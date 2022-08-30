// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy
#include <vector>
#include <queue>

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "Tensor.hpp"
#include "ebm_stats.hpp"
#include "BoosterShell.hpp"

#include "Feature.hpp"
#include "Term.hpp"

#include "GradientPair.hpp"
#include "Bin.hpp"

#include "TreeNode.hpp"
#include "TreeSweep.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// TODO: in theory, a malicious caller could overflow our stack if they pass us data that will grow a sufficiently deep tree.  Consider changing this 
//   recursive function to handle that
template<bool bClassification>
static void Flatten(
   const TreeNode<bClassification> * const pTreeNode,
   ActiveDataType ** const ppSplits, 
   FloatFast ** const ppUpdateScore, 
   const size_t cScores
) {
   // don't log this since we call it recursively.  Log where the root is called
   if(UNPREDICTABLE(pTreeNode->AFTER_IsSplit())) {
      EBM_ASSERT(!IsOverflowTreeNodeSize(bClassification, cScores)); // we're accessing allocated memory
      const size_t cBytesPerTreeNode = GetTreeNodeSize(bClassification, cScores);
      const TreeNode<bClassification> * const pLeftChild = GetLeftNode(
         pTreeNode->AFTER_GetChildren(), cBytesPerTreeNode);
      Flatten<bClassification>(pLeftChild, ppSplits, ppUpdateScore, cScores);
      **ppSplits = pTreeNode->AFTER_GetSplitVal();
      ++(*ppSplits);
      const TreeNode<bClassification> * const pRightChild = GetRightNode(
         pTreeNode->AFTER_GetChildren(), cBytesPerTreeNode);
      Flatten<bClassification>(pRightChild, ppSplits, ppUpdateScore, cScores);
   } else {
      FloatFast * pUpdateScoreCur = *ppUpdateScore;
      FloatFast * const pUpdateScoreNext = pUpdateScoreCur + cScores;
      *ppUpdateScore = pUpdateScoreNext;

      const auto * pGradientPair = pTreeNode->GetGradientPairs();

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
      FloatBig zeroLogit = 0;
#endif // ZERO_FIRST_MULTICLASS_LOGIT

      do {
         FloatBig updateScore;
         if(bClassification) {
            updateScore = EbmStats::ComputeSinglePartitionUpdate(
               pGradientPair->m_sumGradients, pGradientPair->GetSumHessians());

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
            if(2 <= cScores) {
               if(pTreeNode->GetGradientPairs() == pGradientPair) {
                  zeroLogit = updateScore;
               }
               updateScore -= zeroLogit;
            }
#endif // ZERO_FIRST_MULTICLASS_LOGIT

         } else {
            updateScore = EbmStats::ComputeSinglePartitionUpdate(
               pGradientPair->m_sumGradients, pTreeNode->GetWeight());
         }
         *pUpdateScoreCur = SafeConvertFloat<FloatFast>(updateScore);

         ++pGradientPair;
         ++pUpdateScoreCur;
      } while(pUpdateScoreNext != pUpdateScoreCur);
   }
}

// TODO: it would be easy for us to implement a -1 lookback where we make the first split, find the second split, elimnate the first split and try 
//   again on that side, then re-examine the second split again.  For mains this would be very quick we have found that 2-3 splits are optimimum.  
//   Probably 1 split isn't very good since with 2 splits we can localize a region of high gain in the center somewhere

template<ptrdiff_t cCompilerClasses>
static int FindBestSplitGain(
   RandomDeterministic * const pRng,
   BoosterShell * const pBoosterShell,
   TreeNode<IsClassification(cCompilerClasses)> * pTreeNode,
   TreeNode<IsClassification(cCompilerClasses)> * const pTreeNodeStackSpace,
   const size_t cSamplesLeafMin
) {
   constexpr bool bClassification = IsClassification(cCompilerClasses);
   constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;

   LOG_N(
      Trace_Verbose,
      "Entered FindBestSplitGain: "
      "pRng=%p, "
      "pBoosterShell=%p, "
      "pTreeNode=%p, "
      "pTreeNodeStackSpace=%p, "
      "cSamplesLeafMin=%zu"
      ,
      static_cast<void *>(pRng),
      static_cast<const void *>(pBoosterShell),
      static_cast<void *>(pTreeNode),
      static_cast<void *>(pTreeNodeStackSpace),
      cSamplesLeafMin
   );

   if(!pTreeNode->BEFORE_IsSplittable()) {
#ifndef NDEBUG
      pTreeNode->SetDoneGainCalc(true);
#endif // NDEBUG

      pTreeNode->AFTER_RejectSplit();
      return 1;
   }

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();

   const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, cRuntimeClasses);
   const size_t cScores = GetCountScores(cClasses);

   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we're accessing allocated memory
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

   auto * pBinCur = pTreeNode->BEFORE_GetBinFirst();
   const auto * const pBinLast = pTreeNode->BEFORE_GetBinLast();

   EBM_ASSERT(!IsOverflowTreeNodeSize(bClassification, cScores)); // we're accessing allocated memory
   const size_t cBytesPerTreeNode = GetTreeNodeSize(bClassification, cScores);

   TreeNode<bClassification> * const pLeftChild = GetLeftNode(pTreeNodeStackSpace, cBytesPerTreeNode);
   TreeNode<bClassification> * const pRightChild = GetRightNode(pTreeNodeStackSpace, cBytesPerTreeNode);

#ifndef NDEBUG
   pLeftChild->SetDoneGainCalc(false);
   pRightChild->SetDoneGainCalc(false);
#endif // NDEBUG

   // we are not using the memory in our next TreeNode children yet, so use it as our temporary accumulation memory
   pLeftChild->GetBin()->Zero(cBytesPerBin);
   pRightChild->GetBin()->Copy(*pTreeNode->GetBin(), cScores);

   pLeftChild->BEFORE_SetBinFirst(pBinCur);
   pRightChild->BEFORE_SetBinLast(pBinLast);

   EBM_ASSERT(!IsOverflowTreeSweepSize(bClassification, cScores)); // we're accessing allocated memory
   const size_t cBytesPerTreeSweep = GetTreeSweepSize(bClassification, cScores);

   TreeSweep<bClassification> * pTreeSweepStart =
      static_cast<TreeSweep<bClassification> *>(pBoosterShell->GetEquivalentSplits());
   TreeSweep<bClassification> * pTreeSweepCur = pTreeSweepStart;

   EBM_ASSERT(0 <= k_gainMin);
   FloatBig BEST_gain = k_gainMin; // it must at least be this, and maybe it needs to be more
   EBM_ASSERT(0 < cSamplesLeafMin);
   EBM_ASSERT(pBinLast != pBinCur); // we wouldn't call this function on a non-splittable node
   do {
      ASSERT_BIN_OK(cBytesPerBin, pBinCur, pBoosterShell->GetBinsBigEndDebug());

      // TODO: In the future we should add the left, then subtract from the parent to get the right, for numeracy
      //       since then we'll be guaranteed that at least they sum to the total instead of having the left and
      //       right drift away from the total over time from floating point noise
      pRightChild->GetBin()->Subtract(*pBinCur, cScores);
      pLeftChild->GetBin()->Add(*pBinCur, cScores);

      const size_t cSamplesRight = pRightChild->GetBin()->GetCountSamples();
      const size_t cSamplesLeft = pLeftChild->GetBin()->GetCountSamples();

      const FloatBig weightRight = pRightChild->GetBin()->GetWeight();
      const FloatBig weightLeft = pLeftChild->GetBin()->GetWeight();

      if(UNLIKELY(cSamplesRight < cSamplesLeafMin)) {
         break; // we'll just keep subtracting if we continue, so there won't be any more splits, so we're done
      }

      if(LIKELY(cSamplesLeafMin <= cSamplesLeft)) {
         EBM_ASSERT(0 < cSamplesRight);
         EBM_ASSERT(0 < cSamplesLeft);

         FloatBig sumHessiansRight = weightRight;
         FloatBig sumHessiansLeft = weightLeft;
         FloatBig gain = 0;

         // TODO: We can probably move the partial gain calculation into a function of the Bin class
         auto * const aLeftSweepGradientPairs = pLeftChild->GetBin()->GetGradientPairs();
         auto * const aRightSweepGradientPairs = pRightChild->GetBin()->GetGradientPairs();
         for(size_t iScore = 0; iScore < cScores; ++iScore) {
            const FloatBig sumGradientsLeft = aLeftSweepGradientPairs[iScore].m_sumGradients;
            const FloatBig sumGradientsRight = aRightSweepGradientPairs[iScore].m_sumGradients;

            if(bClassification) {
               if(bUseLogitBoost) {
                  sumHessiansLeft = aLeftSweepGradientPairs[iScore].GetSumHessians();
                  sumHessiansRight = aRightSweepGradientPairs[iScore].GetSumHessians();
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
         }
         EBM_ASSERT(std::isnan(gain) || 0 <= gain);

         if(UNLIKELY(/* NaN */ !LIKELY(gain < BEST_gain))) {
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

            // if gain becomes NaN, the first time we come through here we're comparing the non-NaN value in BEST_gain 
            // with gain, which is false.  Next time we come through here, both BEST_gain and gain, 
            // and that has a special case of being false!  So, we always choose pTreeSweepStart, which is great because we don't waste 
            // or fill memory unnessarily
            pTreeSweepCur = UNPREDICTABLE(BEST_gain == gain) ? pTreeSweepCur : pTreeSweepStart;
            BEST_gain = gain;

            pTreeSweepCur->SetBestBin(pBinCur);
            pTreeSweepCur->GetBestLeftBin()->Copy(*pLeftChild->GetBin(), cScores);

            pTreeSweepCur = AddBytesTreeSweep(pTreeSweepCur, cBytesPerTreeSweep);
         } else {
            EBM_ASSERT(!std::isnan(gain));
         }
      }
      pBinCur = IndexBin(pBinCur, cBytesPerBin);
   } while(pBinLast != pBinCur);

   if(UNLIKELY(pTreeSweepStart == pTreeSweepCur)) {
      // no valid splits found
      EBM_ASSERT(k_gainMin == BEST_gain);

#ifndef NDEBUG
      pTreeNode->SetDoneGainCalc(true);
#endif // NDEBUG

      pTreeNode->AFTER_RejectSplit();

      return 1;
   }
   EBM_ASSERT(std::isnan(BEST_gain) || 0 <= BEST_gain);

   if(UNLIKELY(/* NaN */ !LIKELY(BEST_gain <= std::numeric_limits<FloatBig>::max()))) {
      // this tests for NaN and +inf

      // we need this test since the priority queue in the function that calls us cannot accept a NaN value
      // since we would break weak ordering with non-ordered NaN comparisons, thus create undefined behavior

#ifndef NDEBUG
      pTreeNode->SetDoneGainCalc(true);
#endif // NDEBUG

      pTreeNode->AFTER_RejectSplit();

      return -1; // exit boosting with overflow
   }

   FloatBig sumHessiansOverwrite = pTreeNode->GetWeight();
   const auto * pGainGradientPair = pTreeNode->GetGradientPairs();

   for(size_t iScore = 0; iScore < cScores; ++iScore) {
      const FloatBig sumGradientsParent = pGainGradientPair[iScore].m_sumGradients;
      if(bClassification) {
         if(bUseLogitBoost) {
            sumHessiansOverwrite = pGainGradientPair[iScore].GetSumHessians();
         }
      }
      const FloatBig gain1 = EbmStats::CalcPartialGain(sumGradientsParent, sumHessiansOverwrite);
      EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);
      BEST_gain -= gain1;
   }

   // BEST_gain could be -inf if the partial gain on the children reached a number close to +inf and then
   // the children were -inf due to floating point noise.  
   EBM_ASSERT(std::isnan(BEST_gain) || -std::numeric_limits<FloatBig>::infinity() == BEST_gain || k_epsilonNegativeGainAllowed <= BEST_gain);
   EBM_ASSERT(std::numeric_limits<FloatBig>::infinity() != BEST_gain);

   EBM_ASSERT(0 <= k_gainMin);
   if(UNLIKELY(/* NaN */ !LIKELY(k_gainMin <= BEST_gain))) {
      // do not allow splits on gains that are too small
      // also filter out slightly negative numbers that can arrise from floating point noise

#ifndef NDEBUG
      pTreeNode->SetDoneGainCalc(true);
#endif // NDEBUG

      pTreeNode->AFTER_RejectSplit();

      // but if the parent partial gain overflowed to +inf and thus we got a -inf gain, then handle as an overflow
      return /* NaN */ std::numeric_limits<FloatBig>::lowest() <= BEST_gain ? 1 : -1;
   }
   EBM_ASSERT(!std::isnan(BEST_gain));
   EBM_ASSERT(!std::isinf(BEST_gain));
   EBM_ASSERT(0 <= BEST_gain);

   const size_t cSweepItems = CountTreeSweep(pTreeSweepStart, pTreeSweepCur, cBytesPerTreeSweep);
   if(UNLIKELY(1 < cSweepItems)) {
      const size_t iRandom = pRng->NextFast(cSweepItems);
      pTreeSweepStart = AddBytesTreeSweep(pTreeSweepStart, cBytesPerTreeSweep * iRandom);
   }

   const auto * const BEST_pBin = pTreeSweepStart->GetBestBin();
   pLeftChild->BEFORE_SetBinLast(BEST_pBin);

   pLeftChild->GetBin()->Copy(*pTreeSweepStart->GetBestLeftBin(), cScores);

   const auto * const BEST_pBinNext = IndexBin(BEST_pBin, cBytesPerBin);
   ASSERT_BIN_OK(cBytesPerBin, BEST_pBinNext, pBoosterShell->GetBinsBigEndDebug());

   pRightChild->BEFORE_SetBinFirst(BEST_pBinNext);

   pRightChild->GetBin()->Copy(*pTreeNode->GetBin(), cScores);
   pRightChild->GetBin()->Subtract(*pTreeSweepStart->GetBestLeftBin(), cScores);

   // if there were zero samples in the entire dataset then we shouldn't have found a split worth making and we 
   // should have handled the empty dataset earlier
   EBM_ASSERT(0 < pTreeNode->GetCountSamples());


   // IMPORTANT!! : We need to finish all our calls that use pTreeNode->m_UNION.m_beforeGainCalc BEFORE setting 
   // anything in m_UNION.m_afterGainCalc as we do below this comment!
#ifndef NDEBUG
   pTreeNode->SetDoneGainCalc(true);
#endif // NDEBUG


   pTreeNode->AFTER_SetChildren(pTreeNodeStackSpace);
   pTreeNode->AFTER_SetSplitGain(BEST_gain);

   BinBase * const aBinsBase = pBoosterShell->GetBinBaseBig();
   const auto * const aBins = aBinsBase->Specialize<FloatBig, bClassification>();

   EBM_ASSERT(reinterpret_cast<const char *>(aBins) <= reinterpret_cast<const char *>(BEST_pBin));
   EBM_ASSERT(0 == (reinterpret_cast<const char *>(BEST_pBin) - reinterpret_cast<const char *>(aBins)) % cBytesPerBin);
   const size_t iSplit = (reinterpret_cast<const char *>(BEST_pBin) - reinterpret_cast<const char *>(aBins)) / cBytesPerBin;
   pTreeNode->AFTER_SetSplitVal(iSplit);

   LOG_N(Trace_Verbose, "Exited FindBestSplitGain: splitVal=%zu, gain=%le", iSplit, BEST_gain);

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
      const size_t cLeavesMax,
      double * const pTotalGain
   ) {
      EBM_ASSERT(2 <= cBins); // filter these out at the start where we can handle this case easily
      EBM_ASSERT(2 <= cLeavesMax); // filter these out at the start where we can handle this case easily
      EBM_ASSERT(nullptr != pTotalGain);

      ErrorEbm error;

      constexpr bool bClassification = IsClassification(cCompilerClasses);

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();
      const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, cRuntimeClasses);
      const size_t cScores = GetCountScores(cClasses);

      EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we're accessing allocated memory
      const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

      TreeNode<bClassification> * pRootTreeNode =
         static_cast<TreeNode<bClassification> *>(pBoosterShell->GetThreadByteBuffer2());

#ifndef NDEBUG
      pRootTreeNode->SetDoneGainCalc(false);
#endif // NDEBUG

      BinBase * const aBinsBase = pBoosterShell->GetBinBaseBig();
      const auto * const aBins = aBinsBase->Specialize<FloatBig, bClassification>();

      pRootTreeNode->BEFORE_SetBinFirst(aBins);
      pRootTreeNode->BEFORE_SetBinLast(IndexBin(aBins, cBytesPerBin * (cBins - 1)));
      ASSERT_BIN_OK(cBytesPerBin, pRootTreeNode->BEFORE_GetBinLast(), pBoosterShell->GetBinsBigEndDebug());

      // TODO: move the summation code as a static function in this file and call it here giving it the memory of the RootTreeNode to fill in
      pRootTreeNode->GetBin()->Copy(*pBoosterShell->GetSumAllBins<bClassification>(), cScores);

      EBM_ASSERT(!IsOverflowTreeNodeSize(bClassification, cScores));
      const size_t cBytesPerTreeNode = GetTreeNodeSize(bClassification, cScores);

      TreeNode<bClassification> * pTreeNodeStackSpace = IndexTreeNode(pRootTreeNode, cBytesPerTreeNode);

      int retFind = FindBestSplitGain<cCompilerClasses>(
         pRng,
         pBoosterShell,
         pRootTreeNode,
         pTreeNodeStackSpace,
         cSamplesLeafMin
      );
      size_t cLeaves = 1;
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

            TreeNode<bClassification> * pTreeNode = pRootTreeNode;

            // The root node used a left and right leaf, so reserve it here
            pTreeNodeStackSpace = IndexTreeNode(pTreeNodeStackSpace, cBytesPerTreeNode << 1);

            goto skip_first_push_pop;

            do {
               // there is no way to get the top and pop at the same time.. would be good to get a better queue, but our code isn't bottlenecked by it
               pTreeNode = nodeGainRanking.top();
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

               TreeNode<bClassification> * const pLeftChild =
                  GetLeftNode(pTreeNode->AFTER_GetChildren(), cBytesPerTreeNode);

               retFind = FindBestSplitGain<cCompilerClasses>(
                  pRng,
                  pBoosterShell,
                  pLeftChild,
                  pTreeNodeStackSpace,
                  cSamplesLeafMin
               );
               // if FindBestSplitGain returned -1 to indicate an 
               // overflow ignore it here. We successfully made a root node split, so we might as well continue 
               // with the successful tree that we have which can make progress in boosting down the residuals
               if(0 == retFind) {
                  pTreeNodeStackSpace = IndexTreeNode(pTreeNodeStackSpace, cBytesPerTreeNode << 1);
                  // our priority queue comparison function cannot handle NaN gains so we filter out before
                  EBM_ASSERT(!std::isnan(pLeftChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(!std::isinf(pLeftChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(0 <= pLeftChild->AFTER_GetSplitGain());
                  nodeGainRanking.push(pLeftChild);
               }

               TreeNode<bClassification> * const pRightChild =
                  GetRightNode(pTreeNode->AFTER_GetChildren(), cBytesPerTreeNode);

               retFind = FindBestSplitGain<cCompilerClasses>(
                  pRng,
                  pBoosterShell,
                  pRightChild,
                  pTreeNodeStackSpace,
                  cSamplesLeafMin
               );
               // if FindBestSplitGain returned -1 to indicate an 
               // overflow ignore it here. We successfully made a root node split, so we might as well continue 
               // with the successful tree that we have which can make progress in boosting down the residuals
               if(0 == retFind) {
                  pTreeNodeStackSpace = IndexTreeNode(pTreeNodeStackSpace, cBytesPerTreeNode << 1);
                  // our priority queue comparison function cannot handle NaN gains so we filter out before
                  EBM_ASSERT(!std::isnan(pRightChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(!std::isinf(pRightChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(0 <= pRightChild->AFTER_GetSplitGain());
                  nodeGainRanking.push(pRightChild);
               }

               ++cLeaves;
            } while(cLeaves < cLeavesMax && UNLIKELY(!nodeGainRanking.empty()));

            EBM_ASSERT(!std::isnan(totalGain));
            EBM_ASSERT(0 <= totalGain);

            EBM_ASSERT(static_cast<size_t>(reinterpret_cast<char *>(pTreeNodeStackSpace) -
               reinterpret_cast<char *>(pRootTreeNode)) <= pBoosterCore->GetCountBytesSplitting());

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

      Tensor * const pInnerTermUpdate = pBoosterShell->GetInnerTermUpdate();

      error = pInnerTermUpdate->SetCountSplits(iDimension, cLeaves - size_t { 1 });
      if(UNLIKELY(Error_None != error)) {
         // already logged
         return error;
      }
      if(IsMultiplyError(cScores, cLeaves)) {
         LOG_0(Trace_Warning, "WARNING PartitionOneDimensionalBoosting IsMultiplyError(cScores, cLeaves)");
         return Error_OutOfMemory;
      }
      error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * cLeaves);
      if(UNLIKELY(Error_None != error)) {
         // already logged
         return error;
      }
      ActiveDataType * pSplits = pInnerTermUpdate->GetSplitPointer(iDimension);
      FloatFast * pUpdateScore = pInnerTermUpdate->GetTensorScoresPointer();

      LOG_0(Trace_Verbose, "Entered Flatten");
      Flatten(pRootTreeNode, &pSplits, &pUpdateScore, cScores);
      LOG_0(Trace_Verbose, "Exited Flatten");

      EBM_ASSERT(pInnerTermUpdate->GetSplitPointer(iDimension) <= pSplits);
      EBM_ASSERT(static_cast<size_t>(pSplits - pInnerTermUpdate->GetSplitPointer(iDimension)) == cLeaves - 1);
      EBM_ASSERT(pInnerTermUpdate->GetTensorScoresPointer() < pUpdateScore);
      EBM_ASSERT(static_cast<size_t>(pUpdateScore - pInnerTermUpdate->GetTensorScoresPointer()) == cScores * cLeaves);

      return Error_None;
   }
};

extern ErrorEbm PartitionOneDimensionalBoosting(
   RandomDeterministic * const pRng,
   BoosterShell * const pBoosterShell,
   const size_t cBins,
   const size_t iDimension,
   const size_t cSamplesLeafMin,
   const size_t cLeavesMax,
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
            cLeavesMax,
            pTotalGain
         );
      } else {
         error = PartitionOneDimensionalBoostingInternal<k_dynamicClassification>::Func(
            pRng,
            pBoosterShell,
            cBins,
            iDimension,
            cSamplesLeafMin,
            cLeavesMax,
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
         cLeavesMax,
         pTotalGain
      );
   }

   LOG_0(Trace_Verbose, "Exited PartitionOneDimensionalBoosting");

   return error;
}

} // DEFINED_ZONE_NAME
