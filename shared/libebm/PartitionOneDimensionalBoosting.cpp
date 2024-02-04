// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy
#include <vector>
#include <queue>

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT

#define ZONE_main
#include "zones.h"

#include "GradientPair.hpp"
#include "Bin.hpp"

#include "ebm_internal.hpp"
#include "RandomDeterministic.hpp"
#include "ebm_stats.hpp"
#include "Tensor.hpp"
#include "TreeNode.hpp"
#include "SplitPosition.hpp"
#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<bool bHessian, size_t cCompilerScores>
INLINE_RELEASE_TEMPLATED static void SumAllBins(BoosterShell* const pBoosterShell,
      const Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)>* const pBinsEnd,
      const size_t cSamplesTotal,
      const FloatMain weightTotal,
      Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)>* const pBinOut) {
   // these stay the same across boosting rounds, so we can calculate them once at init
   pBinOut->SetCountSamples(static_cast<UIntMain>(cSamplesTotal));
   pBinOut->SetWeight(weightTotal);

   BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pBoosterCore->GetCountScores());

   // if we know how many scores there are, use the memory on the stack where the compiler can optimize access
   GradientPair<FloatMain, bHessian> aSumGradientPairsLocal[GetArrayScores(cCompilerScores)];
   static constexpr bool bUseStackMemory = k_dynamicScores != cCompilerScores;
   auto* const aSumGradientPairs = bUseStackMemory ? aSumGradientPairsLocal : pBinOut->GetGradientPairs();

   ZeroGradientPairs(aSumGradientPairs, cScores);

   const auto* const aBins = pBoosterShell->GetBoostingMainBins()
                                   ->Specialize<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)>();

#ifndef NDEBUG
   UIntMain cSamplesTotalDebug = 0;
   FloatMain weightTotalDebug = 0;
#endif // NDEBUG

   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(bHessian, cScores);

   EBM_ASSERT(2 <= CountBins(pBinsEnd, aBins, cBytesPerBin)); // we pre-filter out features with only one bin

   const auto* pBin = aBins;
   do {
      ASSERT_BIN_OK(cBytesPerBin, pBin, pBoosterShell->GetDebugMainBinsEnd());
#ifndef NDEBUG
      cSamplesTotalDebug += pBin->GetCountSamples();
      weightTotalDebug += pBin->GetWeight();
#endif // NDEBUG

      const auto* aGradientPairs = pBin->GetGradientPairs();

      size_t iScore = 0;
      do {
         aSumGradientPairs[iScore] += aGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);

      pBin = IndexBin(pBin, cBytesPerBin);
   } while(pBinsEnd != pBin);

   EBM_ASSERT(cSamplesTotal == static_cast<size_t>(cSamplesTotalDebug));
   EBM_ASSERT(weightTotalDebug * 0.999 <= weightTotal && weightTotal <= weightTotalDebug * 1.0001);

   if(bUseStackMemory) {
      // if we used registers to collect the gradients and hessians then copy them now to the bin memory

      auto* const aCopyToGradientPairs = pBinOut->GetGradientPairs();
      size_t iScoreCopy = 0;
      do {
         // do not use memset here so that the compiler can keep aSumGradientPairsLocal in registers
         aCopyToGradientPairs[iScoreCopy] = aSumGradientPairs[iScoreCopy];
         ++iScoreCopy;
      } while(cScores != iScoreCopy);
   }
}

// do not inline this.  Not inlining it makes fewer versions that can be called from the more templated functions
template<bool bHessian>
static ErrorEbm Flatten(BoosterShell* const pBoosterShell,
      const TermBoostFlags flags,
      const size_t iDimension,
      const size_t cBins,
      const size_t cSlices) {
   LOG_0(Trace_Verbose, "Entered Flatten");

   EBM_ASSERT(nullptr != pBoosterShell);
   EBM_ASSERT(iDimension <= k_cDimensionsMax);
   EBM_ASSERT(cSlices <= cBins);

   ErrorEbm error;

   Tensor* const pInnerTermUpdate = pBoosterShell->GetInnerTermUpdate();

   error = pInnerTermUpdate->SetCountSlices(iDimension, cSlices);
   if(UNLIKELY(Error_None != error)) {
      // already logged
      return error;
   }

   const BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   const size_t cScores = pBoosterCore->GetCountScores();

   EBM_ASSERT(!IsMultiplyError(cScores, cSlices));
   error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * cSlices);
   if(UNLIKELY(Error_None != error)) {
      // already logged
      return error;
   }

   const bool bUpdateWithHessian = bHessian && 0 == (TermBoostFlags_DisableNewtonUpdate & flags);

   UIntSplit* pSplit = pInnerTermUpdate->GetSplitPointer(iDimension);
   FloatScore* pUpdateScore = pInnerTermUpdate->GetTensorScoresPointer();

   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(bHessian, cScores);

   EBM_ASSERT(!IsOverflowTreeNodeSize(bHessian, cScores)); // we're accessing allocated memory
   const size_t cBytesPerTreeNode = GetTreeNodeSize(bHessian, cScores);

   const auto* const aBins = pBoosterShell->GetBoostingMainBins()->Specialize<FloatMain, UIntMain, bHessian>();
   const auto* const pBinsEnd = IndexBin(aBins, cBytesPerBin * cBins);

   auto* pTreeNode = pBoosterShell->GetTreeNodesTemp<bHessian>();

   TreeNode<bHessian>* pParent = nullptr;
   size_t iEdge;
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
         const void* pBinLastOrChildren = pTreeNode->DANGEROUS_GetBinLastOrChildren();
         // if the pointer points to the space within the bins, then the TreeNode could not be split
         // and this TreeNode never had children and we never wrote a pointer to the children in this memory
         if(pBinLastOrChildren < aBins || pBinsEnd <= pBinLastOrChildren) {
            EBM_ASSERT(pTreeNode <= pBinLastOrChildren &&
                  pBinLastOrChildren < IndexTreeNode(pTreeNode, pBoosterCore->GetCountBytesTreeNodes()));

            // the node was examined and a gain calculated, so it has left and right children.
            // We can retrieve the split location by looking at where the right child would end its range
            const auto* const pRightChild = GetRightNode(pTreeNode->AFTER_GetChildren(), cBytesPerTreeNode);
            pBinLastOrChildren = pRightChild->BEFORE_GetBinLast();
         }
         const auto* const pBinLast = reinterpret_cast<const Bin<FloatMain, UIntMain, bHessian>*>(pBinLastOrChildren);

         EBM_ASSERT(aBins <= pBinLast);
         EBM_ASSERT(pBinLast < pBinsEnd);
         iEdge = CountBins(pBinLast, aBins, cBytesPerBin) + 1;

         const auto* aGradientPair = pTreeNode->GetGradientPairs();
         size_t iScore = 0;
         do {
            FloatCalc updateScore;
            if(bUpdateWithHessian) {
               updateScore = ComputeSinglePartitionUpdate(static_cast<FloatCalc>(aGradientPair[iScore].m_sumGradients),
                     static_cast<FloatCalc>(aGradientPair[iScore].GetHess()));
            } else {
               updateScore = ComputeSinglePartitionUpdate(static_cast<FloatCalc>(aGradientPair[iScore].m_sumGradients),
                     static_cast<FloatCalc>(pTreeNode->GetWeight()));
            }

            *pUpdateScore = static_cast<FloatScore>(updateScore);
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
      auto* pChildren = pTreeNode->AFTER_GetChildren();
      if(nullptr != pChildren) {
         // we checked earlier that countBins could be converted to a UIntSplit
         EBM_ASSERT(!IsConvertError<UIntSplit>(iEdge));
         *pSplit = static_cast<UIntSplit>(iEdge);
         ++pSplit;

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

// TODO: it would be easy for us to implement a -1 lookback where we make the first split, find the second split,
// elimnate the first split and try
//   again on that side, then re-examine the second split again.  For mains this would be very quick we have found that
//   2-3 splits are optimimum. Probably 1 split isn't very good since with 2 splits we can localize a region of high
//   gain in the center somewhere

template<bool bHessian, size_t cCompilerScores>
static int FindBestSplitGain(RandomDeterministic* const pRng,
      BoosterShell* const pBoosterShell,
      const TermBoostFlags flags,
      TreeNode<bHessian, GetArrayScores(cCompilerScores)>* pTreeNode,
      TreeNode<bHessian, GetArrayScores(cCompilerScores)>* const pTreeNodeScratchSpace,
      const size_t cSamplesLeafMin) {

   LOG_N(Trace_Verbose,
         "Entered FindBestSplitGain: "
         "pRng=%p, "
         "pBoosterShell=%p, "
         "pTreeNode=%p, "
         "pTreeNodeScratchSpace=%p, "
         "cSamplesLeafMin=%zu",
         static_cast<void*>(pRng),
         static_cast<const void*>(pBoosterShell),
         static_cast<void*>(pTreeNode),
         static_cast<void*>(pTreeNodeScratchSpace),
         cSamplesLeafMin);

   if(!pTreeNode->BEFORE_IsSplittable()) {
#ifndef NDEBUG
      pTreeNode->SetDebugProgression(1);
#endif // NDEBUG

      pTreeNode->AFTER_RejectSplit();
      return 1;
   }

   BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pBoosterCore->GetCountScores());

   auto* const pLeftChild = GetLeftNode(pTreeNodeScratchSpace);
#ifndef NDEBUG
   pLeftChild->SetDebugProgression(0);
#endif // NDEBUG

   Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> binParent;
   Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> binLeft;

   // if we know how many scores there are, use the memory on the stack where the compiler can optimize access
   static constexpr bool bUseStackMemory = k_dynamicScores != cCompilerScores;
   const auto* const aParentGradientPairs =
         bUseStackMemory ? binParent.GetGradientPairs() : pTreeNode->GetGradientPairs();
   auto* const aLeftGradientPairs = bUseStackMemory ? binLeft.GetGradientPairs() : pLeftChild->GetGradientPairs();
   if(bUseStackMemory) {
      binParent.Copy(cScores, *pTreeNode->GetBin());
   } else {
      binParent.SetCountSamples(pTreeNode->GetCountSamples());
      binParent.SetWeight(pTreeNode->GetWeight());
   }
   binLeft.Zero(cScores, aLeftGradientPairs);

   auto* pBinCur = pTreeNode->BEFORE_GetBinFirst();
   const auto* const pBinLast = pTreeNode->BEFORE_GetBinLast();

   pLeftChild->BEFORE_SetBinFirst(pBinCur);

   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(bHessian, cScores);
   EBM_ASSERT(!IsOverflowSplitPositionSize(bHessian, cScores)); // we're accessing allocated memory
   const size_t cBytesPerSplitPosition = GetSplitPositionSize(bHessian, cScores);

   auto* pBestSplitsStart = pBoosterShell->GetSplitPositionsTemp<bHessian, GetArrayScores(cCompilerScores)>();
   auto* pBestSplitsCur = pBestSplitsStart;

   UIntMain cSamplesRight = binParent.GetCountSamples();

   EBM_ASSERT(FloatCalc{0} <= k_gainMin);
   FloatCalc bestGain = k_gainMin; // it must at least be this, and maybe it needs to be more
   EBM_ASSERT(0 < cSamplesLeafMin);
   EBM_ASSERT(pBinLast != pBinCur); // then we would be non-splitable and would have exited above
   do {
      ASSERT_BIN_OK(cBytesPerBin, pBinCur, pBoosterShell->GetDebugMainBinsEnd());

      const UIntMain cSamplesChange = pBinCur->GetCountSamples();
      cSamplesRight -= cSamplesChange;
      if(UNLIKELY(cSamplesRight < cSamplesLeafMin)) {
         break; // we'll just keep subtracting if we continue, so there won't be any more splits, so we're done
      }

      binLeft.SetCountSamples(binLeft.GetCountSamples() + cSamplesChange);
      binLeft.SetWeight(binLeft.GetWeight() + pBinCur->GetWeight());

      const auto* const aBinGradientPairs = pBinCur->GetGradientPairs();

      FloatMain sumHessiansLeft = binLeft.GetWeight();
      FloatMain sumHessiansRight = binParent.GetWeight() - binLeft.GetWeight();
      FloatCalc gain = 0;

      size_t iScore = 0;
      do {
         const FloatMain sumGradientsLeft =
               aLeftGradientPairs[iScore].m_sumGradients + aBinGradientPairs[iScore].m_sumGradients;
         aLeftGradientPairs[iScore].m_sumGradients = sumGradientsLeft;
         const FloatMain sumGradientsRight = aParentGradientPairs[iScore].m_sumGradients - sumGradientsLeft;

         if(bHessian) {
            const FloatMain newSumHessiansLeft =
                  aLeftGradientPairs[iScore].GetHess() + aBinGradientPairs[iScore].GetHess();
            aLeftGradientPairs[iScore].SetHess(newSumHessiansLeft);
            if(0 == (TermBoostFlags_DisableNewtonGain & flags)) {
               sumHessiansLeft = newSumHessiansLeft;
               sumHessiansRight = aParentGradientPairs[iScore].GetHess() - newSumHessiansLeft;
            }
         }

         // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators
         // (but only do this after we've determined the best node splitting score for classification, and the
         // NewtonRaphsonStep for gain
         const FloatCalc gainRight = CalcPartialGain(
               static_cast<FloatCalc>(sumGradientsRight), static_cast<FloatCalc>(sumHessiansRight));
         EBM_ASSERT(std::isnan(gainRight) || 0 <= gainRight);
         gain += gainRight;

         // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators
         // (but only do this after we've determined the best node splitting score for classification, and the
         // NewtonRaphsonStep for gain
         const FloatCalc gainLeft = CalcPartialGain(
               static_cast<FloatCalc>(sumGradientsLeft), static_cast<FloatCalc>(sumHessiansLeft));
         EBM_ASSERT(std::isnan(gainLeft) || 0 <= gainLeft);
         gain += gainLeft;

         ++iScore;
      } while(cScores != iScore);
      EBM_ASSERT(std::isnan(gain) || 0 <= gain);

      if(LIKELY(cSamplesLeafMin <= binLeft.GetCountSamples())) {
         if(UNLIKELY(/* NaN */ !LIKELY(gain < bestGain))) {
            // propagate NaN values since we stop boosting when we see them

            // it's very possible that we have bins with zero samples in them, in which case we could easily be
            // presented with equally favorable splits or it's even possible for two different possible unrelated
            // sections of bins, or individual bins to have exactly the same gain (think low count symetric data) we
            // want to avoid any bias of always choosing the higher or lower value to split on, so what we should do is
            // store the indexes of any ties in a stack and we reset the stack if we later find a gain that's larger
            // than any we have in the stack. The stack needs to be size_t to hold indexes, and we need the stack to be
            // as long as the number of samples - 1, incase all gain for all bins are the same (potential_splits = bins
            // - 1) after we exit the loop we can examine our stack and choose a random split from all the equivalent
            // splits available.  eg: we find that items at index 4,7,8,9 all have the same gain, so we pick a random
            // number between 0 -> 3 to select which one we actually split on
            //
            // DON'T use a floating point epsilon when comparing the gains.  It's not clear what the epsilon should be
            // given that gain is continuously pushed to zero, so we can get very low numbers here eventually.  As an
            // approximation, we should just take the assumption that if two numbers which have mathematically equality,
            // end up with different gains due to floating point computation issues, that the error will be roughtly
            // symetric such that either the first or the last could be chosen, which is fine for us since we just want
            // to ensure randomized picking. Having two mathematically identical gains is pretty rare in any case,
            // except for the situation where multiple bins have zero samples, but in that case we'll have floating
            // point equality too since we'll be adding zero to the floating points values, which is an exact operation.
            //
            // TODO : implement the randomized splitting described for interaction effect, which can be done the same
            // although we might want to
            //   include near matches since there is floating point noise there due to the way we sum interaction effect
            //   region totals

            // if gain becomes NaN, the first time we come through here we're comparing the non-NaN value in bestGain
            // with gain, which is false.  Next time we come through here, both bestGain and gain,
            // and that has a special case of being false!  So, we always choose pBestSplitsStart, which is great
            // because we don't waste or fill memory unnessarily
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

   if(UNLIKELY(/* NaN */ !LIKELY(bestGain <= std::numeric_limits<FloatCalc>::max()))) {
      // this tests for NaN and +inf

      // we need this test since the priority queue in the function that calls us cannot accept a NaN value
      // since we would break weak ordering with non-ordered NaN comparisons, thus create undefined behavior

#ifndef NDEBUG
      pTreeNode->SetDebugProgression(1);
#endif // NDEBUG

      pTreeNode->AFTER_RejectSplit();
      return -1; // exit boosting with overflow
   }

   FloatMain sumHessiansOverwrite = binParent.GetWeight();
   size_t iScoreParent = 0;
   do {
      const FloatMain sumGradientsParent = aParentGradientPairs[iScoreParent].m_sumGradients;
      if(bHessian) {
         if(0 == (TermBoostFlags_DisableNewtonGain & flags)) {
            sumHessiansOverwrite = aParentGradientPairs[iScoreParent].GetHess();
         }
      }
      const FloatCalc gain1 = CalcPartialGain(
            static_cast<FloatCalc>(sumGradientsParent), static_cast<FloatCalc>(sumHessiansOverwrite));
      EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);
      bestGain -= gain1;

      ++iScoreParent;
   } while(cScores != iScoreParent);

   // bestGain could be -inf if the partial gain on the children reached a number close to +inf and then
   // the children were -inf due to floating point noise.
   EBM_ASSERT(std::isnan(bestGain) || -std::numeric_limits<FloatCalc>::infinity() == bestGain ||
         k_epsilonNegativeGainAllowed <= bestGain);
   EBM_ASSERT(std::numeric_limits<FloatCalc>::infinity() != bestGain);

   EBM_ASSERT(FloatCalc{0} <= k_gainMin);
   if(UNLIKELY(/* NaN */ !LIKELY(k_gainMin <= bestGain))) {
      // do not allow splits on gains that are too small
      // also filter out slightly negative numbers that can arrise from floating point noise

#ifndef NDEBUG
      pTreeNode->SetDebugProgression(1);
#endif // NDEBUG

      pTreeNode->AFTER_RejectSplit();

      // but if the parent partial gain overflowed to +inf and thus we got a -inf gain, then handle as an overflow
      return /* NaN */ std::numeric_limits<FloatCalc>::lowest() <= bestGain ? 1 : -1;
   }
   EBM_ASSERT(!std::isnan(bestGain));
   EBM_ASSERT(!std::isinf(bestGain));
   EBM_ASSERT(0 <= bestGain);

   const size_t cTies = CountSplitPositions(pBestSplitsCur, pBestSplitsStart, cBytesPerSplitPosition);
   if(UNLIKELY(1 < cTies)) {
      const size_t iRandom = pRng->NextFast(cTies);
      pBestSplitsStart = IndexSplitPosition(pBestSplitsStart, cBytesPerSplitPosition * iRandom);
   }

   const auto* const pBestBinPosition = pBestSplitsStart->GetBinPosition();
   pLeftChild->BEFORE_SetBinLast(pBestBinPosition);

   memcpy(pLeftChild->GetBin(), pBestSplitsStart->GetLeftSum(), cBytesPerBin);

   const auto* const pBinFirst = IndexBin(pBestBinPosition, cBytesPerBin);
   ASSERT_BIN_OK(cBytesPerBin, pBinFirst, pBoosterShell->GetDebugMainBinsEnd());

   EBM_ASSERT(!IsOverflowTreeNodeSize(bHessian, cScores)); // we're accessing allocated memory
   const size_t cBytesPerTreeNode = GetTreeNodeSize(bHessian, cScores);
   auto* const pRightChild = GetRightNode(pTreeNodeScratchSpace, cBytesPerTreeNode);
#ifndef NDEBUG
   pRightChild->SetDebugProgression(0);
#endif // NDEBUG
   pRightChild->BEFORE_SetBinLast(pBinLast);
   pRightChild->BEFORE_SetBinFirst(pBinFirst);

   pRightChild->GetBin()->SetCountSamples(
         binParent.GetCountSamples() - pBestSplitsStart->GetLeftSum()->GetCountSamples());
   pRightChild->GetBin()->SetWeight(binParent.GetWeight() - pBestSplitsStart->GetLeftSum()->GetWeight());

   auto* const aRightGradientPairs = pRightChild->GetGradientPairs();
   const auto* const aBestGradientPairs = pBestSplitsStart->GetLeftSum()->GetGradientPairs();
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

template<bool bHessian> class CompareNodeGain final {
 public:
   INLINE_ALWAYS bool operator()(
         const TreeNode<bHessian>* const& lhs, const TreeNode<bHessian>* const& rhs) const noexcept {
      // NEVER check for exact equality (as a precondition is ok), since then we'd violate the weak ordering rule
      // https://medium.com/@shiansu/strict-weak-ordering-and-the-c-stl-f7dcfa4d4e07
      return lhs->AFTER_GetSplitGain() < rhs->AFTER_GetSplitGain();
   }
};

template<bool bHessian, size_t cCompilerScores> class PartitionOneDimensionalBoostingInternal final {
 public:
   PartitionOneDimensionalBoostingInternal() = delete; // this is a static class.  Do not construct

   static ErrorEbm Func(RandomDeterministic* const pRng,
         BoosterShell* const pBoosterShell,
         const TermBoostFlags flags,
         const size_t cBins,
         const size_t iDimension,
         const size_t cSamplesLeafMin,
         const size_t cSplitsMax,
         const size_t cSamplesTotal,
         const FloatMain weightTotal,
         double* const pTotalGain) {
      EBM_ASSERT(2 <= cBins); // filter these out at the start where we can handle this case easily
      EBM_ASSERT(1 <= cSplitsMax); // filter these out at the start where we can handle this case easily
      EBM_ASSERT(nullptr != pTotalGain);

      BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
      const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pBoosterCore->GetCountScores());

      const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(bHessian, cScores);

      auto* const pRootTreeNode = pBoosterShell->GetTreeNodesTemp<bHessian, GetArrayScores(cCompilerScores)>();

#ifndef NDEBUG
      pRootTreeNode->SetDebugProgression(0);
#endif // NDEBUG

      const auto* const aBins = pBoosterShell->GetBoostingMainBins()
                                      ->Specialize<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)>();
      const auto* const pBinsEnd = IndexBin(aBins, cBytesPerBin * cBins);
      const auto* const pBinsLast = NegativeIndexBin(pBinsEnd, cBytesPerBin);

      pRootTreeNode->BEFORE_SetBinFirst(aBins);
      pRootTreeNode->BEFORE_SetBinLast(pBinsLast);
      ASSERT_BIN_OK(cBytesPerBin, pRootTreeNode->BEFORE_GetBinLast(), pBoosterShell->GetDebugMainBinsEnd());

      SumAllBins<bHessian, cCompilerScores>(
            pBoosterShell, pBinsEnd, cSamplesTotal, weightTotal, pRootTreeNode->GetBin());

      EBM_ASSERT(!IsOverflowTreeNodeSize(bHessian, cScores));
      const size_t cBytesPerTreeNode = GetTreeNodeSize(bHessian, cScores);

      auto* pTreeNodeScratchSpace = IndexTreeNode(pRootTreeNode, cBytesPerTreeNode);

      int retFind = FindBestSplitGain<bHessian, cCompilerScores>(
            pRng, pBoosterShell, flags, pRootTreeNode, pTreeNodeScratchSpace, cSamplesLeafMin);
      size_t cSplitsRemaining = cSplitsMax;
      FloatCalc totalGain = 0;
      if(UNLIKELY(0 != retFind)) {
         // there will be no splits at all
         if(UNLIKELY(retFind < 0)) {
            // Any negative return means there was an overflow. Signal the overflow by making gain infinite here.
            // We'll flip it later to a negative number to signal to the caller that there was an overflow.
            totalGain = std::numeric_limits<FloatCalc>::infinity();
         }
      } else {
         // our priority queue comparison function cannot handle NaN gains so we filter out before
         EBM_ASSERT(!std::isnan(pRootTreeNode->AFTER_GetSplitGain()));
         EBM_ASSERT(!std::isinf(pRootTreeNode->AFTER_GetSplitGain()));
         EBM_ASSERT(0 <= pRootTreeNode->AFTER_GetSplitGain());

         try {
            // TODO: someday see if we can replace this with an in-class priority queue that stores it's info inside
            //       the TreeNode datastructure
            std::priority_queue<TreeNode<bHessian>*, std::vector<TreeNode<bHessian>*>, CompareNodeGain<bHessian>>
                  nodeGainRanking;

            auto* pTreeNode = pRootTreeNode;

            // The root node used a left and right leaf, so reserve it here
            pTreeNodeScratchSpace = IndexTreeNode(pTreeNodeScratchSpace, cBytesPerTreeNode << 1);

            goto skip_first_push_pop;

            do {
               // there is no way to get the top and pop at the same time.. would be good to get a better queue, but our
               // code isn't bottlenecked by it
               pTreeNode = nodeGainRanking.top()->template Upgrade<GetArrayScores(cCompilerScores)>();
               // In theory we can have nodes with equal gain values here, but this is very very rare to occur in
               // practice We handle equal gain values in FindBestSplitGain because we can have zero instances in bins,
               // in which case it occurs, but those equivalent situations have been cleansed by the time we reach this
               // code, so the only realistic scenario where we might get equivalent gains is if we had an almost
               // symetric distribution samples bin distributions AND two tail ends that happen to have the same
               // statistics AND either this is our first split, or we've only made a single split in the center in the
               // case where there is symetry in the center Even if all of these things are true, after one non-symetric
               // split, we won't see that scenario anymore since the gradients won't be symetric anymore.  This is so
               // rare, and limited to one split, so we shouldn't bother to handle it since the complexity of doing so
               // outweights the benefits.
               nodeGainRanking.pop();

            skip_first_push_pop:

               // pTreeNode had the highest gain of all the available Nodes, so we will split it.

               // get the gain first, since calling AFTER_SplitNode destroys it
               const FloatCalc totalGainUpdate = pTreeNode->AFTER_GetSplitGain();
               EBM_ASSERT(!std::isnan(totalGainUpdate));
               EBM_ASSERT(!std::isinf(totalGainUpdate));
               EBM_ASSERT(0 <= totalGainUpdate);
               totalGain += totalGainUpdate;

               pTreeNode->AFTER_SplitNode();

               auto* const pLeftChild = GetLeftNode(pTreeNode->AFTER_GetChildren());

               retFind = FindBestSplitGain<bHessian, cCompilerScores>(pRng,
                     pBoosterShell,
                     flags,
                     pLeftChild,
                     pTreeNodeScratchSpace,
                     cSamplesLeafMin);
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

               auto* const pRightChild = GetRightNode(pTreeNode->AFTER_GetChildren(), cBytesPerTreeNode);

               retFind = FindBestSplitGain<bHessian, cCompilerScores>(pRng,
                     pBoosterShell,
                     flags,
                     pRightChild,
                     pTreeNodeScratchSpace,
                     cSamplesLeafMin);
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
         } catch(const std::bad_alloc&) {
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
      return Flatten<bHessian>(pBoosterShell, flags, iDimension, cBins, cSplits + 1);
   }
};

extern ErrorEbm PartitionOneDimensionalBoosting(RandomDeterministic* const pRng,
      BoosterShell* const pBoosterShell,
      const TermBoostFlags flags,
      const size_t cBins,
      const size_t iDimension,
      const size_t cSamplesLeafMin,
      const size_t cSplitsMax,
      const size_t cSamplesTotal,
      const FloatMain weightTotal,
      double* const pTotalGain) {
   LOG_0(Trace_Verbose, "Entered PartitionOneDimensionalBoosting");

   ErrorEbm error;

   BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   const size_t cRuntimeScores = pBoosterCore->GetCountScores();

   EBM_ASSERT(1 <= cRuntimeScores);
   if(pBoosterCore->IsHessian()) {
      if(size_t{1} == cRuntimeScores) {
         error = PartitionOneDimensionalBoostingInternal<true, k_oneScore>::Func(pRng,
               pBoosterShell,
               flags,
               cBins,
               iDimension,
               cSamplesLeafMin,
               cSplitsMax,
               cSamplesTotal,
               weightTotal,
               pTotalGain);
      } else if(size_t{3} == cRuntimeScores) {
         // 3 classes
         error = PartitionOneDimensionalBoostingInternal<true, 3>::Func(pRng,
               pBoosterShell,
               flags,
               cBins,
               iDimension,
               cSamplesLeafMin,
               cSplitsMax,
               cSamplesTotal,
               weightTotal,
               pTotalGain);
      } else {
         // muticlass
         error = PartitionOneDimensionalBoostingInternal<true, k_dynamicScores>::Func(pRng,
               pBoosterShell,
               flags,
               cBins,
               iDimension,
               cSamplesLeafMin,
               cSplitsMax,
               cSamplesTotal,
               weightTotal,
               pTotalGain);
      }
   } else {
      if(size_t{1} == cRuntimeScores) {
         error = PartitionOneDimensionalBoostingInternal<false, k_oneScore>::Func(pRng,
               pBoosterShell,
               flags,
               cBins,
               iDimension,
               cSamplesLeafMin,
               cSplitsMax,
               cSamplesTotal,
               weightTotal,
               pTotalGain);
      } else {
         // Odd: gradient multiclass. Allow it, but do not optimize for it
         error = PartitionOneDimensionalBoostingInternal<false, k_dynamicScores>::Func(pRng,
               pBoosterShell,
               flags,
               cBins,
               iDimension,
               cSamplesLeafMin,
               cSplitsMax,
               cSamplesTotal,
               weightTotal,
               pTotalGain);
      }
   }

   LOG_0(Trace_Verbose, "Exited PartitionOneDimensionalBoosting");

   return error;
}

} // namespace DEFINED_ZONE_NAME
