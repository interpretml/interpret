// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy
#include <vector>
#include <queue>

#include "ebm_native.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "zones.h"

#include "ebm_internal.hpp"

#include "RandomDeterministic.hpp"
#include "ebm_stats.hpp"
#include "Tensor.hpp"
#include "GradientPair.hpp"
#include "Bin.hpp"
#include "TreeNode.hpp"
#include "SplitPosition.hpp"
#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<ptrdiff_t cCompilerClasses>
INLINE_RELEASE_TEMPLATED static void SumAllBins(
   BoosterShell * const pBoosterShell,
   const Bin<FloatBig, IsClassification(cCompilerClasses), GetCountScores(cCompilerClasses)> * const pBinsEnd,
   const size_t cSamplesTotal,
   const FloatBig weightTotal,
   Bin<FloatBig, IsClassification(cCompilerClasses), GetCountScores(cCompilerClasses)> * const pBinOut
) {
   static constexpr bool bClassification = IsClassification(cCompilerClasses);
   static constexpr size_t cCompilerScores = GetCountScores(cCompilerClasses);

   // these stay the same across boosting rounds, so we can calculate them once at init
   pBinOut->SetCountSamples(cSamplesTotal);
   pBinOut->SetWeight(weightTotal);

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();
   const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, cRuntimeClasses);
   const size_t cScores = GetCountScores(cClasses);

   // if we know how many scores there are, use the memory on the stack where the compiler can optimize access
   GradientPair<FloatBig, bClassification> aSumGradientPairsLocal[cCompilerScores];
   static constexpr bool bUseStackMemory = k_dynamicClassification != cCompilerClasses;
   auto * const aSumGradientPairs = bUseStackMemory ? aSumGradientPairsLocal : pBinOut->GetGradientPairs();

   ZeroGradientPairs(aSumGradientPairs, cScores);

   const auto * const aBins = pBoosterShell->GetBoostingBigBins()->Specialize<FloatBig, bClassification, cCompilerScores>();

#ifndef NDEBUG
   size_t cSamplesTotalDebug = 0;
   FloatBig weightTotalDebug = 0;
#endif // NDEBUG

   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we're accessing allocated memory
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

   EBM_ASSERT(2 <= CountBins(pBinsEnd, aBins, cBytesPerBin)); // we pre-filter out features with only one bin

   const auto * pBin = aBins;
   do {
      ASSERT_BIN_OK(cBytesPerBin, pBin, pBoosterShell->GetDebugBigBinsEnd());
#ifndef NDEBUG
      cSamplesTotalDebug += pBin->GetCountSamples();
      weightTotalDebug += pBin->GetWeight();
#endif // NDEBUG

      const auto * aGradientPairs = pBin->GetGradientPairs();

      size_t iScore = 0;
      do {
         aSumGradientPairs[iScore] += aGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);

      pBin = IndexBin(pBin, cBytesPerBin);
   } while(pBinsEnd != pBin);

   EBM_ASSERT(cSamplesTotal == cSamplesTotalDebug);
   EBM_ASSERT(weightTotalDebug * 0.999 <= weightTotal && weightTotal <= weightTotalDebug * 1.0001);

   if(bUseStackMemory) {
      // if we used registers to collect the gradients and hessians then copy them now to the bin memory

      auto * const aCopyToGradientPairs = pBinOut->GetGradientPairs();
      size_t iScoreCopy = 0;
      do {
         // do not use memset here so that the compiler can keep aSumGradientPairsLocal in registers
         aCopyToGradientPairs[iScoreCopy] = aSumGradientPairs[iScoreCopy];
         ++iScoreCopy;
      } while(cScores != iScoreCopy);
   }
}

// do not inline this.  Not inlining it makes fewer versions that can be called from the more templated functions
template<bool bClassification>
static ErrorEbm Flatten(
   BoosterShell * const pBoosterShell,
   const size_t iDimension,
   const size_t cBins,
   const size_t cSplits
) {
   LOG_0(Trace_Verbose, "Entered Flatten");

   EBM_ASSERT(nullptr != pBoosterShell);
   EBM_ASSERT(iDimension <= k_cDimensionsMax);
   EBM_ASSERT(cSplits < cBins);
   
   ErrorEbm error;

   Tensor * const pInnerTermUpdate = pBoosterShell->GetInnerTermUpdate();

   error = pInnerTermUpdate->SetCountSplits(iDimension, cSplits);
   if(UNLIKELY(Error_None != error)) {
      // already logged
      return error;
   }

   const BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();
   const size_t cScores = GetCountScores(cRuntimeClasses);

   // we checked during init that cScores * cBins can be allocated, so cSplits + 1 must work too
   EBM_ASSERT(!IsMultiplyError(cScores, cSplits + size_t { 1 }));
   error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * (cSplits + size_t { 1 }));
   if(UNLIKELY(Error_None != error)) {
      // already logged
      return error;
   }

   ActiveDataType * pSplits = pInnerTermUpdate->GetSplitPointer(iDimension);
   FloatFast * pUpdateScore = pInnerTermUpdate->GetTensorScoresPointer();

   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we're accessing allocated memory
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

   EBM_ASSERT(!IsOverflowTreeNodeSize(bClassification, cScores)); // we're accessing allocated memory
   const size_t cBytesPerTreeNode = GetTreeNodeSize(bClassification, cScores);

   const auto * const aBins = pBoosterShell->GetBoostingBigBins()->Specialize<FloatBig, bClassification>();
   const auto * const pBinsEnd = IndexBin(aBins, cBytesPerBin * cBins);

   auto * pTreeNode = pBoosterShell->GetTreeNodesTemp<bClassification>();

   TreeNode<bClassification> * pParent = nullptr;
   size_t iSplit;
   while(true) {

   moved_down:;
      if(UNPREDICTABLE(pTreeNode->AFTER_IsSplit())) {
#ifndef NDEBUG
         pTreeNode->SetDebugProgression(2);
#endif // NDEBUG

         pTreeNode->DECONSTRUCT_SetParent(pParent);
         pParent = pTreeNode;
         pTreeNode = GetLeftNode(pTreeNode->AFTER_GetChildren());
         goto moved_down;
      } else {
         const void * pBinLastOrChildren = pTreeNode->DANGEROUS_GetBinLastOrChildren();
         // if the pointer points to the space within the bins, then the TreeNode could not be split
         // and this TreeNode never had children and we never wrote a pointer to the children in this memory
         if(pBinLastOrChildren < aBins || pBinsEnd <= pBinLastOrChildren) {
            EBM_ASSERT(pTreeNode <= pBinLastOrChildren && pBinLastOrChildren < 
               IndexTreeNode(pTreeNode, pBoosterCore->GetCountBytesTreeNodes()));

            // the node was examined and a gain calculated, so it has left and right children.
            // We can retrieve the split location by looking at where the right child would end its range
            const auto * const pRightChild = GetRightNode(pTreeNode->AFTER_GetChildren(), cBytesPerTreeNode);
            pBinLastOrChildren = pRightChild->BEFORE_GetBinLast();
         }
         const auto * const pBinLast = reinterpret_cast<const Bin<FloatBig, bClassification> *>(pBinLastOrChildren);

         EBM_ASSERT(aBins <= pBinLast);
         EBM_ASSERT(pBinLast < pBinsEnd);
         iSplit = CountBins(pBinLast, aBins, cBytesPerBin);
            
         const auto * aGradientPair = pTreeNode->GetGradientPairs();
         size_t iScore = 0;
         do {
            FloatBig updateScore;
            if(bClassification) {
               updateScore = EbmStats::ComputeSinglePartitionUpdate(
                  aGradientPair[iScore].m_sumGradients, aGradientPair[iScore].GetHess());
            } else {
               updateScore = EbmStats::ComputeSinglePartitionUpdate(
                  aGradientPair[iScore].m_sumGradients, pTreeNode->GetWeight());
            }

            *pUpdateScore = SafeConvertFloat<FloatFast>(updateScore);
            ++pUpdateScore;

            ++iScore;
         } while(cScores != iScore);

         pTreeNode = pParent;
         if(nullptr != pTreeNode) {
            goto moved_up;
         }
         break; // this can only happen if our tree has zero splits, but we need to check it
      }

   moved_up:;
      auto * pChildren = pTreeNode->AFTER_GetChildren();
      if(nullptr != pChildren) {
         *pSplits = iSplit;
         ++pSplits;

         pParent = pTreeNode;
         pTreeNode->AFTER_SetChildren(nullptr);
         pTreeNode = GetRightNode(pChildren, cBytesPerTreeNode);
         goto moved_down;
      } else {
         pTreeNode = pTreeNode->DECONSTRUCT_GetParent();
         if(nullptr != pTreeNode) {
            goto moved_up;
         }
         break;
      }
   }
   LOG_0(Trace_Verbose, "Exited Flatten");
   return Error_None;
}


// TODO: it would be easy for us to implement a -1 lookback where we make the first split, find the second split, elimnate the first split and try 
//   again on that side, then re-examine the second split again.  For mains this would be very quick we have found that 2-3 splits are optimimum.  
//   Probably 1 split isn't very good since with 2 splits we can localize a region of high gain in the center somewhere

template<ptrdiff_t cCompilerClasses>
static int FindBestSplitGain(
   RandomDeterministic * const pRng,
   BoosterShell * const pBoosterShell,
   TreeNode<IsClassification(cCompilerClasses), GetCountScores(cCompilerClasses)> * pTreeNode,
   TreeNode<IsClassification(cCompilerClasses), GetCountScores(cCompilerClasses)> * const pTreeNodeScratchSpace,
   const size_t cSamplesLeafMin
) {
   static constexpr bool bClassification = IsClassification(cCompilerClasses);
   static constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;
   static constexpr size_t cCompilerScores = GetCountScores(cCompilerClasses);

   LOG_N(
      Trace_Verbose,
      "Entered FindBestSplitGain: "
      "pRng=%p, "
      "pBoosterShell=%p, "
      "pTreeNode=%p, "
      "pTreeNodeScratchSpace=%p, "
      "cSamplesLeafMin=%zu"
      ,
      static_cast<void *>(pRng),
      static_cast<const void *>(pBoosterShell),
      static_cast<void *>(pTreeNode),
      static_cast<void *>(pTreeNodeScratchSpace),
      cSamplesLeafMin
   );

   if(!pTreeNode->BEFORE_IsSplittable()) {
#ifndef NDEBUG
      pTreeNode->SetDebugProgression(1);
#endif // NDEBUG

      pTreeNode->AFTER_RejectSplit();
      return 1;
   }

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();

   const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, cRuntimeClasses);
   const size_t cScores = GetCountScores(cClasses);

   auto * const pLeftChild = GetLeftNode(pTreeNodeScratchSpace);
#ifndef NDEBUG
   pLeftChild->SetDebugProgression(0);
#endif // NDEBUG

   Bin<FloatBig, bClassification, cCompilerScores> binParent;
   Bin<FloatBig, bClassification, cCompilerScores> binLeft;

   // if we know how many scores there are, use the memory on the stack where the compiler can optimize access
   static constexpr bool bUseStackMemory = k_dynamicClassification != cCompilerClasses;
   const auto * const aParentGradientPairs = bUseStackMemory ? binParent.GetGradientPairs() : pTreeNode->GetGradientPairs();
   auto * const aLeftGradientPairs = bUseStackMemory ? binLeft.GetGradientPairs() : pLeftChild->GetGradientPairs();
   if(bUseStackMemory) {
      binParent.Copy(cScores, *pTreeNode->GetBin());
   } else {
      binParent.SetCountSamples(pTreeNode->GetCountSamples());
      binParent.SetWeight(pTreeNode->GetWeight());
   }
   binLeft.Zero(cScores, aLeftGradientPairs);

   auto * pBinCur = pTreeNode->BEFORE_GetBinFirst();
   const auto * const pBinLast = pTreeNode->BEFORE_GetBinLast();

   pLeftChild->BEFORE_SetBinFirst(pBinCur);

   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we're accessing allocated memory
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);
   EBM_ASSERT(!IsOverflowSplitPositionSize(bClassification, cScores)); // we're accessing allocated memory
   const size_t cBytesPerSplitPosition = GetSplitPositionSize(bClassification, cScores);

   auto * pBestSplitsStart = pBoosterShell->GetSplitPositionsTemp<bClassification, cCompilerScores>();
   auto * pBestSplitsCur = pBestSplitsStart;

   size_t cSamplesRight = binParent.GetCountSamples();

   EBM_ASSERT(0 <= k_gainMin);
   FloatBig bestGain = k_gainMin; // it must at least be this, and maybe it needs to be more
   EBM_ASSERT(0 < cSamplesLeafMin);
   EBM_ASSERT(pBinLast != pBinCur); // then we would be non-splitable and would have exited above
   do {
      ASSERT_BIN_OK(cBytesPerBin, pBinCur, pBoosterShell->GetDebugBigBinsEnd());

      const size_t cSamplesChange = pBinCur->GetCountSamples();
      cSamplesRight -= cSamplesChange;
      if(UNLIKELY(cSamplesRight < cSamplesLeafMin)) {
         break; // we'll just keep subtracting if we continue, so there won't be any more splits, so we're done
      }

      binLeft.SetCountSamples(binLeft.GetCountSamples() + cSamplesChange);
      binLeft.SetWeight(binLeft.GetWeight() + pBinCur->GetWeight());

      const auto * const aBinGradientPairs = pBinCur->GetGradientPairs();

      FloatBig sumHessiansLeft = binLeft.GetWeight();
      FloatBig sumHessiansRight = binParent.GetWeight() - binLeft.GetWeight();
      FloatBig gain = 0;

      size_t iScore = 0;
      do {
         const FloatBig sumGradientsLeft = aLeftGradientPairs[iScore].m_sumGradients + 
            aBinGradientPairs[iScore].m_sumGradients;
         aLeftGradientPairs[iScore].m_sumGradients = sumGradientsLeft;
         const FloatBig sumGradientsRight = aParentGradientPairs[iScore].m_sumGradients - sumGradientsLeft;

         if(bClassification) {
            const FloatBig newSumHessiansLeft = aLeftGradientPairs[iScore].GetHess() + aBinGradientPairs[iScore].GetHess();
            aLeftGradientPairs[iScore].SetHess(newSumHessiansLeft);
            if(bUseLogitBoost) {
               sumHessiansLeft = newSumHessiansLeft;
               sumHessiansRight = aParentGradientPairs[iScore].GetHess() - newSumHessiansLeft;
            }
         }

         // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators 
         // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain
         const FloatBig gainRight = EbmStats::CalcPartialGain(sumGradientsRight, sumHessiansRight);
         EBM_ASSERT(std::isnan(gainRight) || 0 <= gainRight);
         gain += gainRight;

         // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators 
         // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain
         const FloatBig gainLeft = EbmStats::CalcPartialGain(sumGradientsLeft, sumHessiansLeft);
         EBM_ASSERT(std::isnan(gainLeft) || 0 <= gainLeft);
         gain += gainLeft;

         ++iScore;
      } while(cScores != iScore);
      EBM_ASSERT(std::isnan(gain) || 0 <= gain);

      if(LIKELY(cSamplesLeafMin <= binLeft.GetCountSamples())) {
         if(UNLIKELY(/* NaN */ !LIKELY(gain < bestGain))) {
            // propagate NaN values since we stop boosting when we see them

            // it's very possible that we have bins with zero samples in them, in which case we could easily be presented with equally favorable splits
            // or it's even possible for two different possible unrelated sections of bins, or individual bins to have exactly the same gain 
            // (think low count symetric data) we want to avoid any bias of always choosing the higher or lower value to split on, so what we should 
            // do is store the indexes of any ties in a stack and we reset the stack if we later find a gain that's larger than any we have in the stack.
            // The stack needs to be size_t to hold indexes, and we need the stack to be as long as the number of samples - 1, incase all gain for 
            // all bins are the same (potential_splits = bins - 1) after we exit the loop we can examine our stack and choose a random split from all 
            // the equivalent splits available.  eg: we find that items at index 4,7,8,9 all have the same gain, so we pick a random number 
            // between 0 -> 3 to select which one we actually split on
            //
            // DON'T use a floating point epsilon when comparing the gains.  It's not clear what the epsilon should be given that gain is continuously
            // pushed to zero, so we can get very low numbers here eventually.  As an approximation, we should just take the assumption that if two 
            // numbers which have mathematically equality, end up with different gains due to floating point computation issues, that the error will 
            // be roughtly symetric such that either the first or the last could be chosen, which is fine for us since we just want to ensure 
            // randomized picking. Having two mathematically identical gains is pretty rare in any case, except for the situation where multiple bins
            // have zero samples, but in that case we'll have floating point equality too since we'll be adding zero to the floating 
            // points values, which is an exact operation.
            //
            // TODO : implement the randomized splitting described for interaction effect, which can be done the same although we might want to 
            //   include near matches since there is floating point noise there due to the way we sum interaction effect region totals

            // if gain becomes NaN, the first time we come through here we're comparing the non-NaN value in bestGain 
            // with gain, which is false.  Next time we come through here, both bestGain and gain, 
            // and that has a special case of being false!  So, we always choose pBestSplitsStart, which is great because we don't waste 
            // or fill memory unnessarily
            pBestSplitsCur = UNPREDICTABLE(bestGain == gain) ? pBestSplitsCur : pBestSplitsStart;
            bestGain = gain;

            pBestSplitsCur->SetBinPosition(pBinCur);

            pBestSplitsCur->GetLeftSum()->Copy(cScores, binLeft, aLeftGradientPairs);

            pBestSplitsCur = IndexSplitPosition(pBestSplitsCur, cBytesPerSplitPosition);
         } else {
            EBM_ASSERT(!std::isnan(gain));
         }
      }
      pBinCur = IndexBin(pBinCur, cBytesPerBin);
   } while(pBinLast != pBinCur);

   if(UNLIKELY(pBestSplitsStart == pBestSplitsCur)) {
      // no valid splits found
      EBM_ASSERT(k_gainMin == bestGain);

#ifndef NDEBUG
      pTreeNode->SetDebugProgression(1);
#endif // NDEBUG

      pTreeNode->AFTER_RejectSplit();
      return 1;
   }
   EBM_ASSERT(std::isnan(bestGain) || 0 <= bestGain);

   if(UNLIKELY(/* NaN */ !LIKELY(bestGain <= std::numeric_limits<FloatBig>::max()))) {
      // this tests for NaN and +inf

      // we need this test since the priority queue in the function that calls us cannot accept a NaN value
      // since we would break weak ordering with non-ordered NaN comparisons, thus create undefined behavior

#ifndef NDEBUG
      pTreeNode->SetDebugProgression(1);
#endif // NDEBUG

      pTreeNode->AFTER_RejectSplit();
      return -1; // exit boosting with overflow
   }

   FloatBig sumHessiansOverwrite = binParent.GetWeight();
   size_t iScoreParent = 0;
   do {
      const FloatBig sumGradientsParent = aParentGradientPairs[iScoreParent].m_sumGradients;
      if(bClassification) {
         if(bUseLogitBoost) {
            sumHessiansOverwrite = aParentGradientPairs[iScoreParent].GetHess();
         }
      }
      const FloatBig gain1 = EbmStats::CalcPartialGain(sumGradientsParent, sumHessiansOverwrite);
      EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);
      bestGain -= gain1;

      ++iScoreParent;
   } while(cScores != iScoreParent);

   // bestGain could be -inf if the partial gain on the children reached a number close to +inf and then
   // the children were -inf due to floating point noise.  
   EBM_ASSERT(std::isnan(bestGain) || -std::numeric_limits<FloatBig>::infinity() == bestGain || k_epsilonNegativeGainAllowed <= bestGain);
   EBM_ASSERT(std::numeric_limits<FloatBig>::infinity() != bestGain);

   EBM_ASSERT(0 <= k_gainMin);
   if(UNLIKELY(/* NaN */ !LIKELY(k_gainMin <= bestGain))) {
      // do not allow splits on gains that are too small
      // also filter out slightly negative numbers that can arrise from floating point noise

#ifndef NDEBUG
      pTreeNode->SetDebugProgression(1);
#endif // NDEBUG

      pTreeNode->AFTER_RejectSplit();

      // but if the parent partial gain overflowed to +inf and thus we got a -inf gain, then handle as an overflow
      return /* NaN */ std::numeric_limits<FloatBig>::lowest() <= bestGain ? 1 : -1;
   }
   EBM_ASSERT(!std::isnan(bestGain));
   EBM_ASSERT(!std::isinf(bestGain));
   EBM_ASSERT(0 <= bestGain);

   const size_t cTies = CountSplitPositions(pBestSplitsCur, pBestSplitsStart, cBytesPerSplitPosition);
   if(UNLIKELY(1 < cTies)) {
      const size_t iRandom = pRng->NextFast(cTies);
      pBestSplitsStart = IndexSplitPosition(pBestSplitsStart, cBytesPerSplitPosition * iRandom);
   }

   const auto * const pBestBinPosition = pBestSplitsStart->GetBinPosition();
   pLeftChild->BEFORE_SetBinLast(pBestBinPosition);

   memcpy(pLeftChild->GetBin(), pBestSplitsStart->GetLeftSum(), cBytesPerBin);

   const auto * const pBinFirst = IndexBin(pBestBinPosition, cBytesPerBin);
   ASSERT_BIN_OK(cBytesPerBin, pBinFirst, pBoosterShell->GetDebugBigBinsEnd());


   EBM_ASSERT(!IsOverflowTreeNodeSize(bClassification, cScores)); // we're accessing allocated memory
   const size_t cBytesPerTreeNode = GetTreeNodeSize(bClassification, cScores);
   auto * const pRightChild = GetRightNode(pTreeNodeScratchSpace, cBytesPerTreeNode);
#ifndef NDEBUG
   pRightChild->SetDebugProgression(0);
#endif // NDEBUG
   pRightChild->BEFORE_SetBinLast(pBinLast);
   pRightChild->BEFORE_SetBinFirst(pBinFirst);

   pRightChild->GetBin()->SetCountSamples(binParent.GetCountSamples() - pBestSplitsStart->GetLeftSum()->GetCountSamples());
   pRightChild->GetBin()->SetWeight(binParent.GetWeight() - pBestSplitsStart->GetLeftSum()->GetWeight());

   auto * const aRightGradientPairs = pRightChild->GetGradientPairs();
   const auto * const aBestGradientPairs = pBestSplitsStart->GetLeftSum()->GetGradientPairs();
   size_t iScoreCopy = 0;
   do {
      auto temp = aParentGradientPairs[iScoreCopy];
      temp -= aBestGradientPairs[iScoreCopy];
      aRightGradientPairs[iScoreCopy] = temp;

      ++iScoreCopy;
   } while(cScores != iScoreCopy);


   // IMPORTANT!! : We need to finish all our calls that use pTreeNode->m_UNION.m_beforeGainCalc BEFORE setting 
   // anything in m_UNION.m_afterGainCalc as we do below this comment!
#ifndef NDEBUG
   pTreeNode->SetDebugProgression(1);
#endif // NDEBUG


   pTreeNode->AFTER_SetChildren(pTreeNodeScratchSpace);
   pTreeNode->AFTER_SetSplitGain(bestGain);

   LOG_N(Trace_Verbose, "Exited FindBestSplitGain: gain=%le", bestGain);

   return 0;
}

template<bool bClassification>
class CompareNodeGain final {
public:
   INLINE_ALWAYS bool operator() (const TreeNode<bClassification> * const & lhs, const TreeNode<bClassification> * const & rhs) const noexcept {
      // NEVER check for exact equality (as a precondition is ok), since then we'd violate the weak ordering rule
      // https://medium.com/@shiansu/strict-weak-ordering-and-the-c-stl-f7dcfa4d4e07
      return lhs->AFTER_GetSplitGain() < rhs->AFTER_GetSplitGain();
   }
};

template<ptrdiff_t cCompilerClasses>
class PartitionOneDimensionalBoostingInternal final {
public:

   PartitionOneDimensionalBoostingInternal() = delete; // this is a static class.  Do not construct

   static ErrorEbm Func(
      RandomDeterministic * const pRng,
      BoosterShell * const pBoosterShell,
      const size_t cBins,
      const size_t iDimension,
      const size_t cSamplesLeafMin,
      const size_t cSplitsMax,
      const size_t cSamplesTotal,
      const FloatBig weightTotal,
      double * const pTotalGain
   ) {
      EBM_ASSERT(2 <= cBins); // filter these out at the start where we can handle this case easily
      EBM_ASSERT(1 <= cSplitsMax); // filter these out at the start where we can handle this case easily
      EBM_ASSERT(nullptr != pTotalGain);

      static constexpr bool bClassification = IsClassification(cCompilerClasses);
      static constexpr size_t cCompilerScores = GetCountScores(cCompilerClasses);

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();
      const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, cRuntimeClasses);
      const size_t cScores = GetCountScores(cClasses);

      EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we're accessing allocated memory
      const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

      auto * const pRootTreeNode = pBoosterShell->GetTreeNodesTemp<bClassification, cCompilerScores>();

#ifndef NDEBUG
      pRootTreeNode->SetDebugProgression(0);
#endif // NDEBUG

      const auto * const aBins = pBoosterShell->GetBoostingBigBins()->Specialize<FloatBig, bClassification, cCompilerScores>();
      const auto * const pBinsEnd = IndexBin(aBins, cBytesPerBin * cBins);
      const auto * const pBinsLast = NegativeIndexBin(pBinsEnd, cBytesPerBin);

      pRootTreeNode->BEFORE_SetBinFirst(aBins);
      pRootTreeNode->BEFORE_SetBinLast(pBinsLast);
      ASSERT_BIN_OK(cBytesPerBin, pRootTreeNode->BEFORE_GetBinLast(), pBoosterShell->GetDebugBigBinsEnd());

      SumAllBins<cCompilerClasses>(pBoosterShell, pBinsEnd, cSamplesTotal, weightTotal, pRootTreeNode->GetBin());

      EBM_ASSERT(!IsOverflowTreeNodeSize(bClassification, cScores));
      const size_t cBytesPerTreeNode = GetTreeNodeSize(bClassification, cScores);

      auto * pTreeNodeScratchSpace = IndexTreeNode(pRootTreeNode, cBytesPerTreeNode);

      int retFind = FindBestSplitGain<cCompilerClasses>(
         pRng,
         pBoosterShell,
         pRootTreeNode,
         pTreeNodeScratchSpace,
         cSamplesLeafMin
      );
      size_t cSplitsRemaining = cSplitsMax;
      FloatBig totalGain = 0;
      if(UNLIKELY(0 != retFind)) {
         // there will be no splits at all
         if(UNLIKELY(retFind < 0)) {
            // Any negative return means there was an overflow. Signal the overflow by making gain infinite here.
            // We'll flip it later to a negative number to signal to the caller that there was an overflow.
            totalGain = std::numeric_limits<double>::infinity();
         }
      } else {
         // our priority queue comparison function cannot handle NaN gains so we filter out before
         EBM_ASSERT(!std::isnan(pRootTreeNode->AFTER_GetSplitGain()));
         EBM_ASSERT(!std::isinf(pRootTreeNode->AFTER_GetSplitGain()));
         EBM_ASSERT(0 <= pRootTreeNode->AFTER_GetSplitGain());

         try {
            // TODO: someday see if we can replace this with an in-class priority queue that stores it's info inside
            //       the TreeNode datastructure
            std::priority_queue<
               TreeNode<bClassification> *,
               std::vector<TreeNode<bClassification> *>,
               CompareNodeGain<bClassification>
            > nodeGainRanking;

            auto * pTreeNode = pRootTreeNode;

            // The root node used a left and right leaf, so reserve it here
            pTreeNodeScratchSpace = IndexTreeNode(pTreeNodeScratchSpace, cBytesPerTreeNode << 1);

            goto skip_first_push_pop;

            do {
               // there is no way to get the top and pop at the same time.. would be good to get a better queue, but our code isn't bottlenecked by it
               pTreeNode = nodeGainRanking.top()->template Upgrade<cCompilerScores>();
               // In theory we can have nodes with equal gain values here, but this is very very rare to occur in practice
               // We handle equal gain values in FindBestSplitGain because we 
               // can have zero instances in bins, in which case it occurs, but those equivalent situations have been cleansed by
               // the time we reach this code, so the only realistic scenario where we might get equivalent gains is if we had an almost
               // symetric distribution samples bin distributions AND two tail ends that happen to have the same statistics AND
               // either this is our first split, or we've only made a single split in the center in the case where there is symetry in the center
               // Even if all of these things are true, after one non-symetric split, we won't see that scenario anymore since the gradients won't be
               // symetric anymore.  This is so rare, and limited to one split, so we shouldn't bother to handle it since the complexity of doing so
               // outweights the benefits.
               nodeGainRanking.pop();

            skip_first_push_pop:

               // pTreeNode had the highest gain of all the available Nodes, so we will split it.

               // get the gain first, since calling AFTER_SplitNode destroys it
               const FloatBig totalGainUpdate = pTreeNode->AFTER_GetSplitGain();
               EBM_ASSERT(!std::isnan(totalGainUpdate));
               EBM_ASSERT(!std::isinf(totalGainUpdate));
               EBM_ASSERT(0 <= totalGainUpdate);
               totalGain += totalGainUpdate;

               pTreeNode->AFTER_SplitNode();

               auto * const pLeftChild = GetLeftNode(pTreeNode->AFTER_GetChildren());

               retFind = FindBestSplitGain<cCompilerClasses>(
                  pRng,
                  pBoosterShell,
                  pLeftChild,
                  pTreeNodeScratchSpace,
                  cSamplesLeafMin
               );
               // if FindBestSplitGain returned -1 to indicate an 
               // overflow ignore it here. We successfully made a root node split, so we might as well continue 
               // with the successful tree that we have which can make progress in boosting down the residuals
               if(0 == retFind) {
                  pTreeNodeScratchSpace = IndexTreeNode(pTreeNodeScratchSpace, cBytesPerTreeNode << 1);
                  // our priority queue comparison function cannot handle NaN gains so we filter out before
                  EBM_ASSERT(!std::isnan(pLeftChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(!std::isinf(pLeftChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(0 <= pLeftChild->AFTER_GetSplitGain());
                  nodeGainRanking.push(pLeftChild->Downgrade());
               }

               auto * const pRightChild = GetRightNode(pTreeNode->AFTER_GetChildren(), cBytesPerTreeNode);

               retFind = FindBestSplitGain<cCompilerClasses>(
                  pRng,
                  pBoosterShell,
                  pRightChild,
                  pTreeNodeScratchSpace,
                  cSamplesLeafMin
               );
               // if FindBestSplitGain returned -1 to indicate an 
               // overflow ignore it here. We successfully made a root node split, so we might as well continue 
               // with the successful tree that we have which can make progress in boosting down the residuals
               if(0 == retFind) {
                  pTreeNodeScratchSpace = IndexTreeNode(pTreeNodeScratchSpace, cBytesPerTreeNode << 1);
                  // our priority queue comparison function cannot handle NaN gains so we filter out before
                  EBM_ASSERT(!std::isnan(pRightChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(!std::isinf(pRightChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(0 <= pRightChild->AFTER_GetSplitGain());
                  nodeGainRanking.push(pRightChild->Downgrade());
               }

               --cSplitsRemaining;
            } while(0 != cSplitsRemaining && UNLIKELY(!nodeGainRanking.empty()));

            EBM_ASSERT(!std::isnan(totalGain));
            EBM_ASSERT(0 <= totalGain);

            EBM_ASSERT(CountBytes(pTreeNodeScratchSpace, pRootTreeNode) <= pBoosterCore->GetCountBytesTreeNodes());
         } catch(const std::bad_alloc &) {
            // calling anything inside nodeGainRanking can throw exceptions
            LOG_0(Trace_Warning, "WARNING PartitionOneDimensionalBoosting out of memory exception");
            return Error_OutOfMemory;
         } catch(...) {
            // calling anything inside nodeGainRanking can throw exceptions
            LOG_0(Trace_Warning, "WARNING PartitionOneDimensionalBoosting exception");
            return Error_UnexpectedInternal;
         }
      }
      *pTotalGain = static_cast<double>(totalGain);
      const size_t cSplits = cSplitsMax - cSplitsRemaining;
      return Flatten<bClassification>(pBoosterShell, iDimension, cBins, cSplits);
   }
};

extern ErrorEbm PartitionOneDimensionalBoosting(
   RandomDeterministic * const pRng,
   BoosterShell * const pBoosterShell,
   const size_t cBins,
   const size_t iDimension,
   const size_t cSamplesLeafMin,
   const size_t cSplitsMax,
   const size_t cSamplesTotal,
   const FloatBig weightTotal,
   double * const pTotalGain
) {
   LOG_0(Trace_Verbose, "Entered PartitionOneDimensionalBoosting");

   ErrorEbm error;

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();

   if(IsClassification(cRuntimeClasses)) {
      if(IsBinaryClassification(cRuntimeClasses)) {
         error = PartitionOneDimensionalBoostingInternal<2>::Func(
            pRng,
            pBoosterShell,
            cBins,
            iDimension,
            cSamplesLeafMin,
            cSplitsMax,
            cSamplesTotal,
            weightTotal,
            pTotalGain
         );
      } else if(3 == cRuntimeClasses) {
         error = PartitionOneDimensionalBoostingInternal<3>::Func(
            pRng,
            pBoosterShell,
            cBins,
            iDimension,
            cSamplesLeafMin,
            cSplitsMax,
            cSamplesTotal,
            weightTotal,
            pTotalGain
         );
      } else {
         error = PartitionOneDimensionalBoostingInternal<k_dynamicClassification>::Func(
            pRng,
            pBoosterShell,
            cBins,
            iDimension,
            cSamplesLeafMin,
            cSplitsMax,
            cSamplesTotal,
            weightTotal,
            pTotalGain
         );
      }
   } else {
      EBM_ASSERT(IsRegression(cRuntimeClasses));
      error = PartitionOneDimensionalBoostingInternal<k_regression>::Func(
         pRng,
         pBoosterShell,
         cBins,
         iDimension,
         cSamplesLeafMin,
         cSplitsMax,
         cSamplesTotal,
         weightTotal,
         pTotalGain
      );
   }

   LOG_0(Trace_Verbose, "Exited PartitionOneDimensionalBoosting");

   return error;
}

} // DEFINED_ZONE_NAME
