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

#include "CompressibleTensor.hpp"
#include "ebm_stats.hpp"
#include "BoosterShell.hpp"

#include "Feature.hpp"
#include "FeatureGroup.hpp"

#include "HistogramTargetEntry.hpp"
#include "HistogramBucket.hpp"

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
   FloatEbmType ** const ppValues, 
   const size_t cVectorLength
) {
   // don't log this since we call it recursively.  Log where the root is called
   if(UNPREDICTABLE(pTreeNode->WAS_THIS_NODE_SPLIT())) {
      EBM_ASSERT(!GetTreeNodeSizeOverflow(bClassification, cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerTreeNode = GetTreeNodeSize(bClassification, cVectorLength);
      const TreeNode<bClassification> * const pLeftChild = GetLeftTreeNodeChild<bClassification>(
         pTreeNode->AFTER_GetTreeNodeChildren(), cBytesPerTreeNode);
      Flatten<bClassification>(pLeftChild, ppSplits, ppValues, cVectorLength);
      **ppSplits = pTreeNode->AFTER_GetSplitValue();
      ++(*ppSplits);
      const TreeNode<bClassification> * const pRightChild = GetRightTreeNodeChild<bClassification>(
         pTreeNode->AFTER_GetTreeNodeChildren(), cBytesPerTreeNode);
      Flatten<bClassification>(pRightChild, ppSplits, ppValues, cVectorLength);
   } else {
      FloatEbmType * pValuesCur = *ppValues;
      FloatEbmType * const pValuesNext = pValuesCur + cVectorLength;
      *ppValues = pValuesNext;

      const auto * pHistogramTargetEntry = pTreeNode->GetHistogramTargetEntry();

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
      FloatEbmType zeroLogit = FloatEbmType { 0 };
#endif // ZERO_FIRST_MULTICLASS_LOGIT

      do {
         FloatEbmType update;
         if(bClassification) {
            update = EbmStats::ComputeSinglePartitionUpdate(
               pHistogramTargetEntry->m_sumGradients, pHistogramTargetEntry->GetSumHessians());

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
            if(2 <= cVectorLength) {
               if(pTreeNode->GetHistogramTargetEntry() == pHistogramTargetEntry) {
                  zeroLogit = update;
               }
               update -= zeroLogit;
            }
#endif // ZERO_FIRST_MULTICLASS_LOGIT

         } else {
            update = EbmStats::ComputeSinglePartitionUpdate(
               pHistogramTargetEntry->m_sumGradients, pTreeNode->GetWeight());
         }
         *pValuesCur = update;

         ++pHistogramTargetEntry;
         ++pValuesCur;
      } while(pValuesNext != pValuesCur);
   }
}

// TODO: it would be easy for us to implement a -1 lookback where we make the first split, find the second split, elimnate the first split and try 
//   again on that side, then re-examine the second split again.  For mains this would be very quick we have found that 2-3 splits are optimimum.  
//   Probably 1 split isn't very good since with 2 splits we can localize a region of high gain in the center somewhere

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
static int ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint(
   BoosterShell * const pBoosterShell,
   TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTreeNode,
   TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pTreeNodeChildrenAvailableStorageSpaceCur,
   const size_t cSamplesRequiredForChildSplitMin
) {
   constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

   LOG_N(
      TraceLevelVerbose,
      "Entered ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint: pBoosterShell=%p, pTreeNode=%p, "
      "pTreeNodeChildrenAvailableStorageSpaceCur=%p, cSamplesRequiredForChildSplitMin=%zu",
      static_cast<const void *>(pBoosterShell),
      static_cast<void *>(pTreeNode),
      static_cast<void *>(pTreeNodeChildrenAvailableStorageSpaceCur),
      cSamplesRequiredForChildSplitMin
   );
   constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();

   const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
      compilerLearningTypeOrCountTargetClasses,
      runtimeLearningTypeOrCountTargetClasses
   );
   const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);

   // it's tempting to want to use GetSumHistogramTargetEntryArray here instead of 
   // GetSumHistogramTargetEntryLeft, but the problem with that is that we sometimes re-do our work
   // when we exceed our memory size by goto retry_with_bigger_tree_node_children_array.  When that happens
   // we need to retrieve the original sum which resides at GetSumHistogramTargetEntryArray
   // since the memory pointed to at pRootTreeNode is freed and re-allocated.
   // So, DO NOT DO: pBoosterShell->GetSumHistogramTargetEntryArray()->
   //   GetHistogramTargetEntry<bClassification>();
   auto * const aSumHistogramTargetEntryLeft = pBoosterShell->GetSumHistogramTargetEntryLeft<bClassification>();

   for(size_t i = 0; i < cVectorLength; ++i) {
      aSumHistogramTargetEntryLeft[i].Zero();
   }

   auto * const aSumHistogramTargetEntryRight = pBoosterShell->GetSumHistogramTargetEntryRight<bClassification>();
   const auto * pHistogramTargetEntryInit = pTreeNode->GetHistogramTargetEntry();
   for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
      // TODO : memcpy this instead
      aSumHistogramTargetEntryRight[iVector] = pHistogramTargetEntryInit[iVector];
   }

   auto * pHistogramBucketEntryCur = pTreeNode->BEFORE_GetHistogramBucketEntryFirst();
   const auto * const pHistogramBucketEntryLast = pTreeNode->BEFORE_GetHistogramBucketEntryLast();

   EBM_ASSERT(!GetTreeNodeSizeOverflow(bClassification, cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerTreeNode = GetTreeNodeSize(bClassification, cVectorLength);

   TreeNode<bClassification> * const pLeftChildInit =
      GetLeftTreeNodeChild<bClassification>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);
   TreeNode<bClassification> * const pRightChildInit =
      GetRightTreeNodeChild<bClassification>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);

#ifndef NDEBUG
   pLeftChildInit->SetExaminedForPossibleSplitting(false);
   pRightChildInit->SetExaminedForPossibleSplitting(false);
#endif // NDEBUG

   pLeftChildInit->BEFORE_SetHistogramBucketEntryFirst(pHistogramBucketEntryCur);
   pRightChildInit->BEFORE_SetHistogramBucketEntryLast(pHistogramBucketEntryLast);

   EBM_ASSERT(!GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<FloatEbmType>(bClassification, cVectorLength);
   EBM_ASSERT(!GetTreeSweepSizeOverflow(bClassification, cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerTreeSweep = GetTreeSweepSize(bClassification, cVectorLength);

   TreeSweep<bClassification> * pTreeSweepStart =
      static_cast<TreeSweep<bClassification> *>(pBoosterShell->GetEquivalentSplits());
   TreeSweep<bClassification> * pTreeSweepCur = pTreeSweepStart;

   size_t cSamplesRight = pTreeNode->AMBIGUOUS_GetCountSamples();
   size_t cSamplesLeft = 0;

   FloatEbmType weightRight = pTreeNode->GetWeight();
   FloatEbmType weightLeft = 0;

   EBM_ASSERT(FloatEbmType { 0 } <= k_gainMin);
   FloatEbmType BEST_gain = k_gainMin; // it must at least be this, and maybe it needs to be more
   EBM_ASSERT(0 < cSamplesRequiredForChildSplitMin);
   EBM_ASSERT(pHistogramBucketEntryLast != pHistogramBucketEntryCur); // we wouldn't call this function on a non-splittable node
   do {
      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucketEntryCur, pBoosterShell->GetHistogramBucketsEndDebug());

      const size_t CHANGE_cSamples = pHistogramBucketEntryCur->GetCountSamplesInBucket();
      cSamplesRight -= CHANGE_cSamples;
      if(UNLIKELY(cSamplesRight < cSamplesRequiredForChildSplitMin)) {
         break; // we'll just keep subtracting if we continue, so there won't be any more splits, so we're done
      }
      cSamplesLeft += CHANGE_cSamples;


      const FloatEbmType CHANGE_weight = pHistogramBucketEntryCur->GetWeightInBucket();
      weightRight -= CHANGE_weight;
      weightLeft += CHANGE_weight;

      const auto * pHistogramTargetEntry = pHistogramBucketEntryCur->GetHistogramTargetEntry();

      if(LIKELY(cSamplesRequiredForChildSplitMin <= cSamplesLeft)) {
         EBM_ASSERT(0 < cSamplesRight);
         EBM_ASSERT(0 < cSamplesLeft);

         FloatEbmType sumHessiansRight = weightRight;
         FloatEbmType sumHessiansLeft = weightLeft;
         FloatEbmType gain = 0;

         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            const FloatEbmType CHANGE_sumGradients = pHistogramTargetEntry[iVector].m_sumGradients;
            const FloatEbmType sumGradientsRight = aSumHistogramTargetEntryRight[iVector].m_sumGradients - CHANGE_sumGradients;
            aSumHistogramTargetEntryRight[iVector].m_sumGradients = sumGradientsRight;
            const FloatEbmType sumGradientsLeft = aSumHistogramTargetEntryLeft[iVector].m_sumGradients + CHANGE_sumGradients;
            aSumHistogramTargetEntryLeft[iVector].m_sumGradients = sumGradientsLeft;

            if(bClassification) {
               const FloatEbmType CHANGE_sumHessians = pHistogramTargetEntry[iVector].GetSumHessians();
               const FloatEbmType newSumHessiansLeft = aSumHistogramTargetEntryLeft[iVector].GetSumHessians() + CHANGE_sumHessians;
               aSumHistogramTargetEntryLeft[iVector].SetSumHessians(newSumHessiansLeft);
               if(bUseLogitBoost) {
                  sumHessiansLeft = newSumHessiansLeft;
                  sumHessiansRight = aSumHistogramTargetEntryRight[iVector].GetSumHessians() - CHANGE_sumHessians;
                  aSumHistogramTargetEntryRight[iVector].SetSumHessians(sumHessiansRight);
               }
            }

            // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators 
            // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain
            const FloatEbmType gainRight = EbmStats::CalcPartialGain(sumGradientsRight, sumHessiansRight);
            EBM_ASSERT(std::isnan(gainRight) || FloatEbmType { 0 } <= gainRight);
            gain += gainRight;

            // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators 
            // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain
            const FloatEbmType gainLeft = EbmStats::CalcPartialGain(sumGradientsLeft, sumHessiansLeft);
            EBM_ASSERT(std::isnan(gainLeft) || FloatEbmType { 0 } <= gainLeft);
            gain += gainLeft;
         }
         EBM_ASSERT(std::isnan(gain) || FloatEbmType { 0 } <= gain);

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
            // randomized picking. Having two mathematically identical gains is pretty rare in any case, except for the situation where one bucket 
            // has bins with zero samples, but in that case we'll have floating point equality too since we'll be adding zero to the floating 
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

            pTreeSweepCur->SetBestHistogramBucketEntry(pHistogramBucketEntryCur);
            pTreeSweepCur->SetCountBestSamplesLeft(cSamplesLeft);
            pTreeSweepCur->SetBestWeightLeft(weightLeft);
            memcpy(
               pTreeSweepCur->GetBestHistogramTargetEntry(), aSumHistogramTargetEntryLeft,
               sizeof(*aSumHistogramTargetEntryLeft) * cVectorLength
            );

            pTreeSweepCur = AddBytesTreeSweep(pTreeSweepCur, cBytesPerTreeSweep);
         } else {
            EBM_ASSERT(!std::isnan(gain));
         }
      } else {
         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            const FloatEbmType CHANGE_sumGradients = pHistogramTargetEntry[iVector].m_sumGradients;
            aSumHistogramTargetEntryRight[iVector].m_sumGradients -= CHANGE_sumGradients;
            aSumHistogramTargetEntryLeft[iVector].m_sumGradients += CHANGE_sumGradients;
            if(bClassification) {
               const FloatEbmType CHANGE_sumHessians = pHistogramTargetEntry[iVector].GetSumHessians();
               aSumHistogramTargetEntryLeft[iVector].SetSumHessians(aSumHistogramTargetEntryLeft[iVector].GetSumHessians() + CHANGE_sumHessians);
               if(bUseLogitBoost) {
                  aSumHistogramTargetEntryRight[iVector].SetSumHessians(aSumHistogramTargetEntryRight[iVector].GetSumHessians() - CHANGE_sumHessians);
               }
            }
         }
      }
      pHistogramBucketEntryCur = GetHistogramBucketByIndex(cBytesPerHistogramBucket, pHistogramBucketEntryCur, 1);
   } while(pHistogramBucketEntryLast != pHistogramBucketEntryCur);

   if(UNLIKELY(pTreeSweepStart == pTreeSweepCur)) {
      // no valid splits found
      EBM_ASSERT(k_gainMin == BEST_gain);
      return 1;
   }
   EBM_ASSERT(std::isnan(BEST_gain) || FloatEbmType { 0 } <= BEST_gain);

   if(UNLIKELY(/* NaN */ !LIKELY(BEST_gain <= std::numeric_limits<FloatEbmType>::max()))) {
      // this tests for NaN and +inf

      // we need this test since the priority queue in the function that calls us cannot accept a NaN value
      // since we would break weak ordering with non-ordered NaN comparisons, thus create undefined behavior

      return -1; // exit boosting with overflow
   }

   FloatEbmType sumHessiansOverwrite = pTreeNode->GetWeight();
   const auto * pHistEntryParent = pTreeNode->GetHistogramTargetEntry();

   for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
      const FloatEbmType sumGradientsParent = pHistEntryParent[iVector].m_sumGradients;
      if(bClassification) {
         if(bUseLogitBoost) {
            sumHessiansOverwrite = pHistEntryParent[iVector].GetSumHessians();
         }
      }
      const FloatEbmType gain1 = EbmStats::CalcPartialGain(sumGradientsParent, sumHessiansOverwrite);
      EBM_ASSERT(std::isnan(gain1) || FloatEbmType { 0 } <= gain1);
      BEST_gain -= gain1;
   }

   // BEST_gain could be -inf if the partial gain on the children reached a number close to +inf and then
   // the children were -inf due to floating point noise.  
   EBM_ASSERT(std::isnan(BEST_gain) || -std::numeric_limits<FloatEbmType>::infinity() == BEST_gain || k_epsilonNegativeGainAllowed <= BEST_gain);
   EBM_ASSERT(std::numeric_limits<FloatEbmType>::infinity() != BEST_gain);

   EBM_ASSERT(FloatEbmType { 0 } <= k_gainMin);
   if(UNLIKELY(/* NaN */ !LIKELY(k_gainMin <= BEST_gain))) {
      // do not allow splits on gains that are too small
      // also filter out slightly negative numbers that can arrise from floating point noise

      // but if the parent partial gain overflowed to +inf and thus we got a -inf gain, then handle as an overflow
      return /* NaN */ std::numeric_limits<FloatEbmType>::lowest() <= BEST_gain ? 1 : -1;
   }
   EBM_ASSERT(!std::isnan(BEST_gain));
   EBM_ASSERT(!std::isinf(BEST_gain));
   EBM_ASSERT(FloatEbmType { 0 } <= BEST_gain);

   RandomStream * const pRandomStream = pBoosterShell->GetRandomStream();

   const size_t cSweepItems = CountTreeSweep(pTreeSweepStart, pTreeSweepCur, cBytesPerTreeSweep);
   if(UNLIKELY(1 < cSweepItems)) {
      const size_t iRandom = pRandomStream->Next(cSweepItems);
      pTreeSweepStart = AddBytesTreeSweep(pTreeSweepStart, cBytesPerTreeSweep * iRandom);
   }

   TreeNode<bClassification> * const pLeftChild = 
      GetLeftTreeNodeChild<bClassification>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);

   const auto * const BEST_pHistogramBucketEntry = pTreeSweepStart->GetBestHistogramBucketEntry();
   pLeftChild->BEFORE_SetHistogramBucketEntryLast(BEST_pHistogramBucketEntry);
   const size_t BEST_cSamplesLeft = pTreeSweepStart->GetCountBestSamplesLeft();
   pLeftChild->AMBIGUOUS_SetCountSamples(BEST_cSamplesLeft);

   const FloatEbmType BEST_weightLeft = pTreeSweepStart->GetBestWeightLeft();
   pLeftChild->SetWeight(BEST_weightLeft);

   const auto * const BEST_pHistogramBucketEntryNext = 
      GetHistogramBucketByIndex(cBytesPerHistogramBucket, BEST_pHistogramBucketEntry, 1);
   ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, BEST_pHistogramBucketEntryNext, pBoosterShell->GetHistogramBucketsEndDebug());

   TreeNode<bClassification> * const pRightChild = GetRightTreeNodeChild<bClassification>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);

   pRightChild->BEFORE_SetHistogramBucketEntryFirst(BEST_pHistogramBucketEntryNext);
   const size_t cSamplesParent = pTreeNode->AMBIGUOUS_GetCountSamples();
   // if there were zero samples in the entire dataset then we shouldn't have found a split worth making and we 
   // should have handled the empty dataset earlier
   EBM_ASSERT(0 < cSamplesParent);
   pRightChild->AMBIGUOUS_SetCountSamples(cSamplesParent - BEST_cSamplesLeft);

   const FloatEbmType weightParent = pTreeNode->GetWeight();
   pRightChild->SetWeight(weightParent - BEST_weightLeft);

   auto * pHistogramTargetEntryLeftChild = pLeftChild->GetHistogramTargetEntry();

   auto * pHistogramTargetEntryRightChild = pRightChild->GetHistogramTargetEntry();

   const auto * pHistogramTargetEntryTreeNode = pTreeNode->GetHistogramTargetEntry();

   const auto * pHistogramTargetEntrySweep = pTreeSweepStart->GetBestHistogramTargetEntry();

   for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
      const FloatEbmType BEST_sumGradientsLeft = pHistogramTargetEntrySweep[iVector].m_sumGradients;
      pHistogramTargetEntryLeftChild[iVector].m_sumGradients = BEST_sumGradientsLeft;
      const FloatEbmType sumGradientsParent = pHistogramTargetEntryTreeNode[iVector].m_sumGradients;
      pHistogramTargetEntryRightChild[iVector].m_sumGradients = sumGradientsParent - BEST_sumGradientsLeft;

      if(bClassification) {
         const FloatEbmType BEST_sumHessiansLeft = pHistogramTargetEntrySweep[iVector].GetSumHessians();
         pHistogramTargetEntryLeftChild[iVector].SetSumHessians(BEST_sumHessiansLeft);
         const FloatEbmType sumHessiansParent = pHistogramTargetEntryTreeNode[iVector].GetSumHessians();
         pHistogramTargetEntryRightChild[iVector].SetSumHessians(sumHessiansParent - BEST_sumHessiansLeft);
      }
   }

   // IMPORTANT!! : we need to finish all our calls that use this->m_UNION.m_beforeExaminationForPossibleSplitting BEFORE setting anything in 
   // m_UNION.m_afterExaminationForPossibleSplitting as we do below this comment!  The call above to this->GetSamples() needs to be done above 
   // these lines because it uses m_UNION.m_beforeExaminationForPossibleSplitting for classification!
#ifndef NDEBUG
   pTreeNode->SetExaminedForPossibleSplitting(true);
#endif // NDEBUG


   pTreeNode->AFTER_SetTreeNodeChildren(pTreeNodeChildrenAvailableStorageSpaceCur);
   pTreeNode->AFTER_SetSplitGain(BEST_gain);

   HistogramBucketBase * const aHistogramBucketBase = pBoosterShell->GetHistogramBucketBase();
   const auto * const aHistogramBucket = aHistogramBucketBase->GetHistogramBucket<FloatEbmType, bClassification>();

   EBM_ASSERT(reinterpret_cast<const char *>(aHistogramBucket) <= reinterpret_cast<const char *>(BEST_pHistogramBucketEntry));
   EBM_ASSERT(0 == (reinterpret_cast<const char *>(BEST_pHistogramBucketEntry) - reinterpret_cast<const char *>(aHistogramBucket)) % cBytesPerHistogramBucket);
   pTreeNode->AFTER_SetSplitValue((reinterpret_cast<const char *>(BEST_pHistogramBucketEntry) - 
      reinterpret_cast<const char *>(aHistogramBucket)) / cBytesPerHistogramBucket);

   LOG_N(
      TraceLevelVerbose,
      "Exited ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint: splitValue=%zu, gain=%le",
      static_cast<size_t>(pTreeNode->AFTER_GetSplitValue()),
      pTreeNode->AFTER_GetSplitGain()
   );

   return 0;
}

template<bool bClassification>
class CompareTreeNodeSplittingGain final {
public:
   INLINE_ALWAYS bool operator() (const TreeNode<bClassification> * const & lhs, const TreeNode<bClassification> * const & rhs) const noexcept {
      // NEVER check for exact equality (as a precondition is ok), since then we'd violate the weak ordering rule
      // https://medium.com/@shiansu/strict-weak-ordering-and-the-c-stl-f7dcfa4d4e07
      return lhs->AFTER_GetSplitGain() < rhs->AFTER_GetSplitGain();
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class PartitionOneDimensionalBoostingInternal final {
public:

   PartitionOneDimensionalBoostingInternal() = delete; // this is a static class.  Do not construct

   static ErrorEbmType Func(
      BoosterShell * const pBoosterShell,
      const size_t cHistogramBuckets,
      const size_t cSamplesTotal,
      const FloatEbmType weightTotal,
      const size_t iDimension,
      const size_t cSamplesRequiredForChildSplitMin,
      const size_t cLeavesMax,
      FloatEbmType * const pTotalGain
   ) {
      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      ErrorEbmType error;

      HistogramBucketBase * const aHistogramBucketBase = pBoosterShell->GetHistogramBucketBase();
      const auto * const aHistogramBucket = 
         aHistogramBucketBase->GetHistogramBucket<FloatEbmType, bClassification>();

      HistogramTargetEntryBase * const aSumHistogramTargetEntryBase =
         pBoosterShell->GetSumHistogramTargetEntryArray();
      const auto * const aSumHistogramTargetEntry =
         aSumHistogramTargetEntryBase->GetHistogramTargetEntry<FloatEbmType, bClassification>();

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);

      EBM_ASSERT(nullptr != pTotalGain);
      EBM_ASSERT(1 <= cSamplesTotal); // filter these out at the start where we can handle this case easily
      EBM_ASSERT(2 <= cHistogramBuckets); // filter these out at the start where we can handle this case easily
      EBM_ASSERT(2 <= cLeavesMax); // filter these out at the start where we can handle this case easily

      // there will be at least one split

      if(GetTreeNodeSizeOverflow(bClassification, cVectorLength)) {
         LOG_0(TraceLevelWarning, "WARNING PartitionOneDimensionalBoosting GetTreeNodeSizeOverflow<bClassification>(cVectorLength)");
         return Error_OutOfMemory; // we haven't accessed this TreeNode memory yet, so we don't know if it overflows yet
      }
      const size_t cBytesPerTreeNode = GetTreeNodeSize(bClassification, cVectorLength);
      EBM_ASSERT(!GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<FloatEbmType>(bClassification, cVectorLength);

   retry_with_bigger_tree_node_children_array:

      size_t cBytesBuffer2 = pBoosterShell->GetThreadByteBuffer2Size();
      // we need 1 TreeNode for the root, 1 for the left child of the root and 1 for the right child of the root
      const size_t cBytesInitialNeededAllocation = 3 * cBytesPerTreeNode;
      if(cBytesBuffer2 < cBytesInitialNeededAllocation) {
         // TODO : we can eliminate this check as long as we ensure that the ThreadByteBuffer2 is always initialized to be equal to the size of three 
         // TreeNodes (left and right) == GET_SIZEOF_ONE_TREE_NODE_CHILDREN(cBytesPerTreeNode), or the number of bins (interactions multiply bins) on the 
         // highest bin count feature
         error = pBoosterShell->GrowThreadByteBuffer2(cBytesInitialNeededAllocation);
         if(Error_None != error) {
            // already logged
            return error;
         }
         cBytesBuffer2 = pBoosterShell->GetThreadByteBuffer2Size();
         EBM_ASSERT(cBytesInitialNeededAllocation <= cBytesBuffer2);
      }
      TreeNode<bClassification> * pRootTreeNode =
         static_cast<TreeNode<bClassification> *>(pBoosterShell->GetThreadByteBuffer2());

#ifndef NDEBUG
      pRootTreeNode->SetExaminedForPossibleSplitting(false);
#endif // NDEBUG

      pRootTreeNode->BEFORE_SetHistogramBucketEntryFirst(aHistogramBucket);
      pRootTreeNode->BEFORE_SetHistogramBucketEntryLast(
         GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBucket, cHistogramBuckets - 1)
      );
      ASSERT_BINNED_BUCKET_OK(
         cBytesPerHistogramBucket,
         pRootTreeNode->BEFORE_GetHistogramBucketEntryLast(),
         pBoosterShell->GetHistogramBucketsEndDebug()
      );
      pRootTreeNode->AMBIGUOUS_SetCountSamples(cSamplesTotal);
      pRootTreeNode->SetWeight(weightTotal);

      // copying existing mem
      memcpy(
         pRootTreeNode->GetHistogramTargetEntry(),
         aSumHistogramTargetEntry,
         cVectorLength * sizeof(*aSumHistogramTargetEntry)
      );

      CompressibleTensor * const pSmallChangeToModelOverwriteSingleSamplingSet =
         pBoosterShell->GetOverwritableModelUpdate();

      size_t cLeaves;
      const int retExamine = ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint<compilerLearningTypeOrCountTargetClasses>(
         pBoosterShell,
         pRootTreeNode,
         AddBytesTreeNode<bClassification>(pRootTreeNode, cBytesPerTreeNode),
         cSamplesRequiredForChildSplitMin
      );
      if(UNLIKELY(0 != retExamine)) {
         // there will be no splits at all

         // any negative gain means there was an overflow.  Let the caller decide if they want to ignore it
         *pTotalGain = UNLIKELY(retExamine < 0) ? std::numeric_limits<FloatEbmType>::infinity() : FloatEbmType { 0 };

         error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountSplits(iDimension, 0);
         if(UNLIKELY(Error_None != error)) {
            // already logged
            return error;
         }

         // we don't need to call EnsureValueCapacity because by default we start with a value capacity of 2 * cVectorLength
         if(bClassification) {
            FloatEbmType * const aValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
            FloatEbmType zeroLogit = FloatEbmType { 0 };
#endif // ZERO_FIRST_MULTICLASS_LOGIT

            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               FloatEbmType update = EbmStats::ComputeSinglePartitionUpdate(
                  pRootTreeNode->GetHistogramTargetEntry()[iVector].m_sumGradients, pRootTreeNode->GetHistogramTargetEntry()[iVector].GetSumHessians()
               );

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
               if(IsMulticlass(compilerLearningTypeOrCountTargetClasses)) {
                  if(size_t { 0 } == iVector) {
                     zeroLogit = update;
                  }
                  update -= zeroLogit;
               }
#endif // ZERO_FIRST_MULTICLASS_LOGIT

               aValues[iVector] = update;
            }
         } else {
            EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
            const FloatEbmType smallChangeToModel = EbmStats::ComputeSinglePartitionUpdate(
               pRootTreeNode->GetHistogramTargetEntry()[0].m_sumGradients, weightTotal
            );
            FloatEbmType * pValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
            pValues[0] = smallChangeToModel;
         }

         return Error_None;
      }

      // our priority queue comparison function cannot handle NaN gains so we filter out before
      EBM_ASSERT(!std::isnan(pRootTreeNode->AFTER_GetSplitGain()));
      EBM_ASSERT(!std::isinf(pRootTreeNode->AFTER_GetSplitGain()));
      EBM_ASSERT(FloatEbmType { 0 } <= pRootTreeNode->AFTER_GetSplitGain());

      if(UNPREDICTABLE(PREDICTABLE(2 == cLeavesMax) || UNPREDICTABLE(2 == cHistogramBuckets))) {
         // there will be exactly 1 split, which is a special case that we can return faster without as much overhead as the multiple split case

         EBM_ASSERT(2 != cHistogramBuckets || !GetLeftTreeNodeChild<bClassification>(
            pRootTreeNode->AFTER_GetTreeNodeChildren(), cBytesPerTreeNode)->IsSplittable() &&
            !GetRightTreeNodeChild<bClassification>(
               pRootTreeNode->AFTER_GetTreeNodeChildren(),
               cBytesPerTreeNode
               )->IsSplittable()
         );

         error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountSplits(iDimension, 1);
         if(UNLIKELY(Error_None != error)) {
            // already logged
            return error;
         }

         ActiveDataType * pSplits = pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimension);
         pSplits[0] = pRootTreeNode->AFTER_GetSplitValue();

         // we don't need to call EnsureValueCapacity because by default we start with a value capacity of 2 * cVectorLength

         // TODO : we don't need to get the right and left pointer from the root.. we know where they will be
         const TreeNode<bClassification> * const pLeftChild = GetLeftTreeNodeChild<bClassification>(
            pRootTreeNode->AFTER_GetTreeNodeChildren(),
            cBytesPerTreeNode
         );
         const TreeNode<bClassification> * const pRightChild = GetRightTreeNodeChild<bClassification>(
            pRootTreeNode->AFTER_GetTreeNodeChildren(),
            cBytesPerTreeNode
         );

         const auto * pHistogramTargetEntryLeftChild = pLeftChild->GetHistogramTargetEntry();

         const auto * pHistogramTargetEntryRightChild = pRightChild->GetHistogramTargetEntry();

         FloatEbmType * const aValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
         if(bClassification) {

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
            FloatEbmType zeroLogit0 = FloatEbmType { 0 };
            FloatEbmType zeroLogit1 = FloatEbmType { 0 };
#endif // ZERO_FIRST_MULTICLASS_LOGIT

            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               FloatEbmType update0 = EbmStats::ComputeSinglePartitionUpdate(
                  pHistogramTargetEntryLeftChild[iVector].m_sumGradients,
                  pHistogramTargetEntryLeftChild[iVector].GetSumHessians()
               );
               FloatEbmType update1 = EbmStats::ComputeSinglePartitionUpdate(
                  pHistogramTargetEntryRightChild[iVector].m_sumGradients,
                  pHistogramTargetEntryRightChild[iVector].GetSumHessians()
               );

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
               if(IsMulticlass(compilerLearningTypeOrCountTargetClasses)) {
                  if(size_t { 0 } == iVector) {
                     zeroLogit0 = update0;
                     zeroLogit1 = update1;
                  }
                  update0 -= zeroLogit0;
                  update1 -= zeroLogit1;
               }
#endif // ZERO_FIRST_MULTICLASS_LOGIT

               aValues[iVector] = update0;
               aValues[cVectorLength + iVector] = update1;
            }
         } else {
            EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
            aValues[0] = EbmStats::ComputeSinglePartitionUpdate(
               pHistogramTargetEntryLeftChild[0].m_sumGradients,
               pLeftChild->GetWeight()
            );
            aValues[1] = EbmStats::ComputeSinglePartitionUpdate(
               pHistogramTargetEntryRightChild[0].m_sumGradients,
               pRightChild->GetWeight()
            );
         }

         const FloatEbmType totalGain = pRootTreeNode->EXTRACT_GAIN_BEFORE_SPLITTING();
         EBM_ASSERT(!std::isnan(totalGain));
         EBM_ASSERT(!std::isinf(totalGain));
         EBM_ASSERT(FloatEbmType { 0 } <= totalGain);
         *pTotalGain = totalGain;
         return Error_None;
      }

      // it's very likely that there will be more than 1 split below this point.  The only case where we wouldn't 
      // split below is if both our children nodes don't have enough cases to split, but that should rare

      // typically we train on stumps, so often this priority queue is overhead since with 2-3 splits the 
      // overhead is too large to benefit, but we also aren't bottlenecked if we only have 2-3 splits, 
      // so we don't care about performance issues.  On the other hand, we don't want to change this to an 
      // array scan because in theory the user can specify very deep trees, and we don't want to hang on 
      // an O(N^2) operation if they do.  So, let's keep the priority queue, and only the priority queue 
      // since it handles all scenarios without any real cost and is simpler
      // than implementing an optional array scan PLUS a priority queue for deep trees.

      // TODO: someday see if we can replace this with an in-class priority queue that stores it's info inside
      //       the TreeNode datastructure

      try {
         std::priority_queue<
            TreeNode<bClassification> *,
            std::vector<TreeNode<bClassification> *>,
            CompareTreeNodeSplittingGain<bClassification>
         > bestTreeNodeToSplit;

         cLeaves = size_t { 1 };
         TreeNode<bClassification> * pParentTreeNode = pRootTreeNode;

         // we skip 3 tree nodes.  The root, the left child of the root, and the right child of the root
         TreeNode<bClassification> * pTreeNodeChildrenAvailableStorageSpaceCur =
            AddBytesTreeNode<bClassification>(pRootTreeNode, cBytesInitialNeededAllocation);

         FloatEbmType totalGain = FloatEbmType { 0 };

         goto skip_first_push_pop;

         do {
            // there is no way to get the top and pop at the same time.. would be good to get a better queue, but our code isn't bottlenecked by it
            pParentTreeNode = bestTreeNodeToSplit.top();
            // In theory we can have nodes with equal gain values here, but this is very very rare to occur in practice
            // We handle equal gain values in ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint because we 
            // can have zero instnaces in bins, in which case it occurs, but those equivalent situations have been cleansed by
            // the time we reach this code, so the only realistic scenario where we might get equivalent gains is if we had an almost
            // symetric distribution samples bin distributions AND two tail ends that happen to have the same statistics AND
            // either this is our first split, or we've only made a single split in the center in the case where there is symetry in the center
            // Even if all of these things are true, after one non-symetric split, we won't see that scenario anymore since the gradients won't be
            // symetric anymore.  This is so rare, and limited to one split, so we shouldn't bother to handle it since the complexity of doing so
            // outweights the benefits.
            bestTreeNodeToSplit.pop();

         skip_first_push_pop:

            // ONLY AFTER WE'VE POPPED pParentTreeNode OFF the priority queue is it considered to have been split.  Calling SPLIT_THIS_NODE makes it formal
            const FloatEbmType totalGainUpdate = pParentTreeNode->EXTRACT_GAIN_BEFORE_SPLITTING();
            EBM_ASSERT(!std::isnan(totalGainUpdate));
            EBM_ASSERT(!std::isinf(totalGainUpdate));
            EBM_ASSERT(FloatEbmType { 0 } <= totalGainUpdate);
            totalGain += totalGainUpdate;

            pParentTreeNode->SPLIT_THIS_NODE();

            TreeNode<bClassification> * const pLeftChild =
               GetLeftTreeNodeChild<bClassification>(
                  pParentTreeNode->AFTER_GetTreeNodeChildren(),
                  cBytesPerTreeNode
               );
            if(pLeftChild->IsSplittable()) {
               TreeNode<bClassification> * pTreeNodeChildrenAvailableStorageSpaceNext =
                  AddBytesTreeNode<bClassification>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode << 1);
               if(cBytesBuffer2 <
                  static_cast<size_t>(reinterpret_cast<char *>(pTreeNodeChildrenAvailableStorageSpaceNext) - reinterpret_cast<char *>(pRootTreeNode))) {
                  error = pBoosterShell->GrowThreadByteBuffer2(cBytesPerTreeNode);
                  if(Error_None != error) {
                     // already logged
                     return error;
                  }
                  goto retry_with_bigger_tree_node_children_array;
               }
               // the act of splitting it implicitly sets INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED
               // because splitting sets splitGain to a non-illegalGain value
               if(0 == ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint<compilerLearningTypeOrCountTargetClasses>(
                  pBoosterShell,
                  pLeftChild,
                  pTreeNodeChildrenAvailableStorageSpaceCur,
                  cSamplesRequiredForChildSplitMin
               )) {
                  pTreeNodeChildrenAvailableStorageSpaceCur = pTreeNodeChildrenAvailableStorageSpaceNext;
                  // our priority queue comparison function cannot handle NaN gains so we filter out before
                  EBM_ASSERT(!std::isnan(pLeftChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(!std::isinf(pLeftChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(FloatEbmType { 0 } <= pLeftChild->AFTER_GetSplitGain());
                  bestTreeNodeToSplit.push(pLeftChild);
               } else {
                  // if ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint returned -1 to indicate an 
                  // overflow ignore it here. We successfully made a root node split, so we might as well continue 
                  // with the successful tree that we have which can make progress in boosting down the residuals

                  goto no_left_split;
               }
            } else {
            no_left_split:;
               // we aren't going to split this TreeNode because we can't. We need to set the splitGain value 
               // here because otherwise it is filled with garbage that could be NaN (meaning the node was a branch) 
               // we can't call INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED before calling SplitTreeNode 
               // because INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED sets 
               // m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain and the 
               // m_UNION.m_beforeExaminationForPossibleSplitting values are 
               // needed if we had decided to call ExamineNodeForSplittingAndDetermineBestPossibleSplit

#ifndef NDEBUG
               pLeftChild->SetExaminedForPossibleSplitting(true);
#endif // NDEBUG

               pLeftChild->INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED();
            }

            TreeNode<bClassification> * const pRightChild = GetRightTreeNodeChild<bClassification>(
               pParentTreeNode->AFTER_GetTreeNodeChildren(),
               cBytesPerTreeNode
            );
            if(pRightChild->IsSplittable()) {
               TreeNode<bClassification> * pTreeNodeChildrenAvailableStorageSpaceNext =
                  AddBytesTreeNode<bClassification>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode << 1);
               if(cBytesBuffer2 <
                  static_cast<size_t>(reinterpret_cast<char *>(pTreeNodeChildrenAvailableStorageSpaceNext) - reinterpret_cast<char *>(pRootTreeNode))) {
                  error = pBoosterShell->GrowThreadByteBuffer2(cBytesPerTreeNode);
                  if(Error_None != error) {
                     // already logged
                     return error;
                  }
                  goto retry_with_bigger_tree_node_children_array;
               }
               // the act of splitting it implicitly sets INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED 
               // because splitting sets splitGain to a non-NaN value
               if(0 == ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint<compilerLearningTypeOrCountTargetClasses>(
                  pBoosterShell,
                  pRightChild,
                  pTreeNodeChildrenAvailableStorageSpaceCur,
                  cSamplesRequiredForChildSplitMin
               )) {
                  pTreeNodeChildrenAvailableStorageSpaceCur = pTreeNodeChildrenAvailableStorageSpaceNext;
                  // our priority queue comparison function cannot handle NaN gains so we filter out before
                  EBM_ASSERT(!std::isnan(pRightChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(!std::isinf(pRightChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(FloatEbmType { 0 } <= pRightChild->AFTER_GetSplitGain());
                  bestTreeNodeToSplit.push(pRightChild);
               } else {
                  // if ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint returned -1 to indicate an 
                  // overflow ignore it here. We successfully made a root node split, so we might as well continue 
                  // with the successful tree that we have which can make progress in boosting down the residuals

                  goto no_right_split;
               }
            } else {
            no_right_split:;
               // we aren't going to split this TreeNode because we can't. We need to set the splitGain value 
               // here because otherwise it is filled with garbage that could be NaN (meaning the node was a branch) 
               // we can't call INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED before calling SplitTreeNode 
               // because INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED sets 
               // m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain and the 
               // m_UNION.m_beforeExaminationForPossibleSplitting values are 
               // needed if we had decided to call ExamineNodeForSplittingAndDetermineBestPossibleSplit

#ifndef NDEBUG
               pRightChild->SetExaminedForPossibleSplitting(true);
#endif // NDEBUG

               pRightChild->INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED();
            }
            ++cLeaves;
         } while(cLeaves < cLeavesMax && UNLIKELY(!bestTreeNodeToSplit.empty()));
         // we DON'T need to call SetLeafAfterDone() on any items that remain in the bestTreeNodeToSplit queue because everything in that queue has set 
         // a non-NaN gain value


         EBM_ASSERT(!std::isnan(totalGain));
         EBM_ASSERT(FloatEbmType { 0 } <= totalGain);

         *pTotalGain = totalGain;
         EBM_ASSERT(
            static_cast<size_t>(reinterpret_cast<char *>(pTreeNodeChildrenAvailableStorageSpaceCur) - reinterpret_cast<char *>(pRootTreeNode)) <= cBytesBuffer2
         );
      } catch(const std::bad_alloc &) {
         // calling anything inside bestTreeNodeToSplit can throw exceptions
         LOG_0(TraceLevelWarning, "WARNING PartitionOneDimensionalBoosting out of memory exception");
         return Error_OutOfMemory;
      } catch(...) {
         // calling anything inside bestTreeNodeToSplit can throw exceptions
         LOG_0(TraceLevelWarning, "WARNING PartitionOneDimensionalBoosting exception");
         return Error_UnexpectedInternal;
      }

      error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountSplits(iDimension, cLeaves - size_t { 1 });
      if(UNLIKELY(Error_None != error)) {
         // already logged
         return error;
      }
      if(IsMultiplyError(cVectorLength, cLeaves)) {
         LOG_0(TraceLevelWarning, "WARNING PartitionOneDimensionalBoosting IsMultiplyError(cVectorLength, cLeaves)");
         return Error_OutOfMemory;
      }
      error = pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * cLeaves);
      if(UNLIKELY(Error_None != error)) {
         // already logged
         return error;
      }
      ActiveDataType * pSplits = pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimension);
      FloatEbmType * pValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();

      LOG_0(TraceLevelVerbose, "Entered Flatten");
      Flatten<bClassification>(pRootTreeNode, &pSplits, &pValues, cVectorLength);
      LOG_0(TraceLevelVerbose, "Exited Flatten");

      EBM_ASSERT(pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimension) <= pSplits);
      EBM_ASSERT(static_cast<size_t>(pSplits - pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimension)) == cLeaves - 1);
      EBM_ASSERT(pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer() < pValues);
      EBM_ASSERT(static_cast<size_t>(pValues - pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()) == cVectorLength * cLeaves);

      return Error_None;
   }
};

extern ErrorEbmType PartitionOneDimensionalBoosting(
   BoosterShell * const pBoosterShell,
   const size_t cHistogramBuckets,
   const size_t cSamplesTotal,
   const FloatEbmType weightTotal,
   const size_t iDimension,
   const size_t cSamplesRequiredForChildSplitMin,
   const size_t cLeavesMax,
   FloatEbmType * const pTotalGain
) {
   LOG_0(TraceLevelVerbose, "Entered PartitionOneDimensionalBoosting");

   ErrorEbmType error;

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();

   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      if(IsBinaryClassification(runtimeLearningTypeOrCountTargetClasses)) {
         error = PartitionOneDimensionalBoostingInternal<2>::Func(
            pBoosterShell,
            cHistogramBuckets,
            cSamplesTotal,
            weightTotal,
            iDimension,
            cSamplesRequiredForChildSplitMin,
            cLeavesMax,
            pTotalGain
         );
      } else {
         error = PartitionOneDimensionalBoostingInternal<k_dynamicClassification>::Func(
            pBoosterShell,
            cHistogramBuckets,
            cSamplesTotal,
            weightTotal,
            iDimension,
            cSamplesRequiredForChildSplitMin,
            cLeavesMax,
            pTotalGain
         );
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      error = PartitionOneDimensionalBoostingInternal<k_regression>::Func(
         pBoosterShell,
         cHistogramBuckets,
         cSamplesTotal,
         weightTotal,
         iDimension,
         cSamplesRequiredForChildSplitMin,
         cLeavesMax,
         pTotalGain
      );
   }

   LOG_0(TraceLevelVerbose, "Exited PartitionOneDimensionalBoosting");

   return error;
}

} // DEFINED_ZONE_NAME
