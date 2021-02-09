// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy
#include <vector>
#include <queue>

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "SegmentedTensor.h"
#include "EbmStats.h"
#include "ThreadStateBoosting.h"

#include "FeatureAtomic.h"
#include "FeatureGroup.h"

#include "HistogramTargetEntry.h"
#include "HistogramBucket.h"

#include "TreeNode.h"
#include "TreeSweep.h"

// TODO: in theory, a malicious caller could overflow our stack if they pass us data that will grow a sufficiently deep tree.  Consider changing this 
//   recursive function to handle that
template<bool bClassification>
static void Flatten(
   const TreeNode<bClassification> * const pTreeNode,
   ActiveDataType ** const ppDivisions, 
   FloatEbmType ** const ppValues, 
   const size_t cVectorLength
) {
   // don't log this since we call it recursively.  Log where the root is called
   if(UNPREDICTABLE(pTreeNode->WAS_THIS_NODE_SPLIT())) {
      EBM_ASSERT(!GetTreeNodeSizeOverflow(bClassification, cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerTreeNode = GetTreeNodeSize(bClassification, cVectorLength);
      const TreeNode<bClassification> * const pLeftChild = GetLeftTreeNodeChild<bClassification>(
         pTreeNode->AFTER_GetTreeNodeChildren(), cBytesPerTreeNode);
      Flatten<bClassification>(pLeftChild, ppDivisions, ppValues, cVectorLength);
      **ppDivisions = pTreeNode->AFTER_GetDivisionValue();
      ++(*ppDivisions);
      const TreeNode<bClassification> * const pRightChild = GetRightTreeNodeChild<bClassification>(
         pTreeNode->AFTER_GetTreeNodeChildren(), cBytesPerTreeNode);
      Flatten<bClassification>(pRightChild, ppDivisions, ppValues, cVectorLength);
   } else {
      FloatEbmType * pValuesCur = *ppValues;
      FloatEbmType * const pValuesNext = pValuesCur + cVectorLength;
      *ppValues = pValuesNext;

      const HistogramTargetEntry<bClassification> * pHistogramTargetEntry = 
         pTreeNode->GetHistogramTargetEntry();
      do {
         FloatEbmType smallChangeToModel;
         if(bClassification) {
            smallChangeToModel = EbmStats::ComputeSmallChangeForOneSegmentClassificationLogOdds(
               pHistogramTargetEntry->m_sumResidualError, pHistogramTargetEntry->GetSumDenominator());
         } else {
            smallChangeToModel = EbmStats::ComputeSmallChangeForOneSegmentRegression(
               pHistogramTargetEntry->m_sumResidualError, static_cast<FloatEbmType>(pTreeNode->AMBIGUOUS_GetSamples()));
         }
         *pValuesCur = smallChangeToModel;

         ++pHistogramTargetEntry;
         ++pValuesCur;
      } while(pValuesNext != pValuesCur);
   }
}

// TODO: it would be easy for us to implement a -1 lookback where we make the first cut, find the second cut, elimnate the first cut and try 
//   again on that side, then re-examine the second cut again.  For mains this would be very quick we have found that 2-3 cuts are optimimum.  
//   Probably 1 cut isn't very good since with 2 cuts we can localize a region of high gain in the center somewhere

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
static bool ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint(
   ThreadStateBoosting * const pThreadStateBoosting,
   TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTreeNode,
   TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pTreeNodeChildrenAvailableStorageSpaceCur,
   const size_t cSamplesRequiredForChildSplitMin
) {
   constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

   LOG_N(
      TraceLevelVerbose,
      "Entered ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint: pThreadStateBoosting=%p, pTreeNode=%p, "
      "pTreeNodeChildrenAvailableStorageSpaceCur=%p, cSamplesRequiredForChildSplitMin=%zu",
      static_cast<const void *>(pThreadStateBoosting),
      static_cast<void *>(pTreeNode),
      static_cast<void *>(pTreeNodeChildrenAvailableStorageSpaceCur),
      cSamplesRequiredForChildSplitMin
   );

   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();

   const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
      compilerLearningTypeOrCountTargetClasses,
      runtimeLearningTypeOrCountTargetClasses
   );
   const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);

   // it's tempting to want to use GetSumHistogramTargetEntryArray here instead of 
   // GetSumHistogramTargetEntry1Array, but the problem with that is that we sometimes re-do our work
   // when we exceed our memory size by goto retry_with_bigger_tree_node_children_array.  When that happens
   // we need to retrieve the original sum which resides at GetSumHistogramTargetEntryArray
   // since the memory pointed to at pRootTreeNode is freed and re-allocated.
   // So, DO NOT DO: pThreadStateBoosting->GetSumHistogramTargetEntryArray()->
   //   GetHistogramTargetEntry<bClassification>();
   HistogramTargetEntry<bClassification> * const aSumHistogramTargetEntryLeft =
      pThreadStateBoosting->GetSumHistogramTargetEntry1Array<bClassification>();

   for(size_t i = 0; i < cVectorLength; ++i) {
      aSumHistogramTargetEntryLeft[i].Zero();
   }

   FloatEbmType * const aSumResidualErrorsRight = pThreadStateBoosting->GetTempFloatVector();
   const HistogramTargetEntry<bClassification> * pHistogramTargetEntryInit = 
      pTreeNode->GetHistogramTargetEntry();
   for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
      aSumResidualErrorsRight[iVector] = pHistogramTargetEntryInit[iVector].m_sumResidualError;
   }

   const HistogramBucket<bClassification> * pHistogramBucketEntryCur =
      pTreeNode->BEFORE_GetHistogramBucketEntryFirst();
   const HistogramBucket<bClassification> * const pHistogramBucketEntryLast =
      pTreeNode->BEFORE_GetHistogramBucketEntryLast();

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

   EBM_ASSERT(!GetHistogramBucketSizeOverflow(bClassification, cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);
   EBM_ASSERT(!GetTreeSweepSizeOverflow(bClassification, cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerTreeSweep = GetTreeSweepSize(bClassification, cVectorLength);

   TreeSweep<bClassification> * pTreeSweepStart =
      static_cast<TreeSweep<bClassification> *>(pThreadStateBoosting->GetEquivalentSplits());
   TreeSweep<bClassification> * pTreeSweepCur = pTreeSweepStart;

   size_t cSamplesRight = pTreeNode->AMBIGUOUS_GetSamples();
   size_t cSamplesLeft = 0;
   FloatEbmType BEST_nodeSplittingScore = k_illegalGain;
   EBM_ASSERT(0 < cSamplesRequiredForChildSplitMin);
   EBM_ASSERT(pHistogramBucketEntryLast != pHistogramBucketEntryCur); // we wouldn't call this function on a non-splittable node
   do {
      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucketEntryCur, pThreadStateBoosting->GetHistogramBucketsEndDebug());

      const size_t CHANGE_cSamples = pHistogramBucketEntryCur->GetCountSamplesInBucket();
      cSamplesRight -= CHANGE_cSamples;
      if(UNLIKELY(cSamplesRight < cSamplesRequiredForChildSplitMin)) {
         break; // we'll just keep subtracting if we continue, so there won't be any more splits, so we're done
      }
      cSamplesLeft += CHANGE_cSamples;

      const HistogramTargetEntry<bClassification> * pHistogramTargetEntry =
         pHistogramBucketEntryCur->GetHistogramTargetEntry();

      if(LIKELY(cSamplesRequiredForChildSplitMin <= cSamplesLeft)) {
         EBM_ASSERT(0 < cSamplesRight);
         EBM_ASSERT(0 < cSamplesLeft);

         const FloatEbmType cSamplesRightFloatEbmType = static_cast<FloatEbmType>(cSamplesRight);
         const FloatEbmType cSamplesLeftFloatEbmType = static_cast<FloatEbmType>(cSamplesLeft);
         FloatEbmType nodeSplittingScore = 0;

         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            const FloatEbmType CHANGE_sumResidualError = pHistogramTargetEntry[iVector].m_sumResidualError;

            const FloatEbmType sumResidualErrorRight = aSumResidualErrorsRight[iVector] - CHANGE_sumResidualError;
            aSumResidualErrorsRight[iVector] = sumResidualErrorRight;

            // TODO : we can make this faster by doing the division in ComputeNodeSplittingScore after we add all the numerators 
            // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain
            const FloatEbmType nodeSplittingScoreRight = EbmStats::ComputeNodeSplittingScore(sumResidualErrorRight, cSamplesRightFloatEbmType);
            EBM_ASSERT(std::isnan(nodeSplittingScoreRight) || FloatEbmType { 0 } <= nodeSplittingScoreRight);
            nodeSplittingScore += nodeSplittingScoreRight;

            const FloatEbmType sumResidualErrorLeft = aSumHistogramTargetEntryLeft[iVector].m_sumResidualError + CHANGE_sumResidualError;
            aSumHistogramTargetEntryLeft[iVector].m_sumResidualError = sumResidualErrorLeft;

            // TODO : we can make this faster by doing the division in ComputeNodeSplittingScore after we add all the numerators 
            // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain
            const FloatEbmType nodeSplittingScoreLeft = EbmStats::ComputeNodeSplittingScore(sumResidualErrorLeft, cSamplesLeftFloatEbmType);
            EBM_ASSERT(std::isnan(nodeSplittingScoreLeft) || FloatEbmType { 0 } <= nodeSplittingScoreLeft);
            nodeSplittingScore += nodeSplittingScoreLeft;

            if(bClassification) {
               aSumHistogramTargetEntryLeft[iVector].SetSumDenominator(
                  aSumHistogramTargetEntryLeft[iVector].GetSumDenominator() +
                  pHistogramTargetEntry[iVector].GetSumDenominator()
               );
            }
         }
         EBM_ASSERT(std::isnan(nodeSplittingScore) || FloatEbmType { 0 } <= nodeSplittingScore);

         // if we get a NaN result, we'd like to propagate it by making bestSplit NaN.  The rules for NaN values say that non equality comparisons are 
         // all false so, let's flip this comparison such that it should be true for NaN values.  If the compiler violates NaN comparions rules, no big deal.
         // NaN values will get us soon and shut down boosting.
         if(UNLIKELY(/* DO NOT CHANGE THIS WITHOUT READING THE ABOVE. WE DO THIS STRANGE COMPARISON FOR NaN values*/
            !(nodeSplittingScore < BEST_nodeSplittingScore))) {
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

            // if nodeSplittingScore becomes NaN, the first time we come through here we're comparing the non-NaN value in BEST_nodeSplittingScore 
            // with nodeSplittingScore, which is false.  Next time we come through here, both BEST_nodeSplittingScore and nodeSplittingScore, 
            // and that has a special case of being false!  So, we always choose pTreeSweepStart, which is great because we don't waste 
            // or fill memory unnessarily
            pTreeSweepCur = UNPREDICTABLE(BEST_nodeSplittingScore == nodeSplittingScore) ? pTreeSweepCur : pTreeSweepStart;
            BEST_nodeSplittingScore = nodeSplittingScore;

            pTreeSweepCur->SetBestHistogramBucketEntry(pHistogramBucketEntryCur);
            pTreeSweepCur->SetCountBestSamplesLeft(cSamplesLeft);
            memcpy(
               pTreeSweepCur->GetBestHistogramTargetEntry(), aSumHistogramTargetEntryLeft,
               sizeof(*aSumHistogramTargetEntryLeft) * cVectorLength
            );

            pTreeSweepCur = AddBytesTreeSweep(pTreeSweepCur, cBytesPerTreeSweep);
         }
      } else {
         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            const FloatEbmType CHANGE_sumResidualError =
               pHistogramTargetEntry[iVector].m_sumResidualError;

            aSumResidualErrorsRight[iVector] -= CHANGE_sumResidualError;
            aSumHistogramTargetEntryLeft[iVector].m_sumResidualError += CHANGE_sumResidualError;
            if(bClassification) {
               aSumHistogramTargetEntryLeft[iVector].SetSumDenominator(
                  aSumHistogramTargetEntryLeft[iVector].GetSumDenominator() +
                  pHistogramTargetEntry[iVector].GetSumDenominator()
               );
            }
         }
      }
      pHistogramBucketEntryCur = GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pHistogramBucketEntryCur, 1);
   } while(pHistogramBucketEntryLast != pHistogramBucketEntryCur);

   // handle the case where BEST_nodeSplittingScore is +infinity 
   // don't use std::inf because with some compiler flags on some compilers that isn't reliable
   // if we include a case that was equal to max, then ok, no harm done.
   if(UNLIKELY(UNLIKELY(pTreeSweepStart == pTreeSweepCur) || UNLIKELY(std::isnan(BEST_nodeSplittingScore)) ||
      UNLIKELY(std::numeric_limits<FloatEbmType>::max() <= BEST_nodeSplittingScore))) {

      // we didn't find any valid splits, or we hit an overflow
      return true;
   }
   EBM_ASSERT(FloatEbmType { 0 } <= BEST_nodeSplittingScore);

   RandomStream * const pRandomStream = pBooster->GetRandomStream();

   const size_t cSweepItems = CountTreeSweep(pTreeSweepStart, pTreeSweepCur, cBytesPerTreeSweep);
   if(UNLIKELY(1 < cSweepItems)) {
      const size_t iRandom = pRandomStream->Next(cSweepItems);
      pTreeSweepStart = AddBytesTreeSweep(pTreeSweepStart, cBytesPerTreeSweep * iRandom);
   }

   TreeNode<bClassification> * const pLeftChild =
      GetLeftTreeNodeChild<bClassification>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);

   const HistogramBucket<bClassification> * const BEST_pHistogramBucketEntry = pTreeSweepStart->GetBestHistogramBucketEntry();
   pLeftChild->BEFORE_SetHistogramBucketEntryLast(BEST_pHistogramBucketEntry);
   const size_t BEST_cSamplesLeft = pTreeSweepStart->GetCountBestSamplesLeft();
   pLeftChild->AMBIGUOUS_SetSamples(BEST_cSamplesLeft);

   const HistogramBucket<bClassification> * const BEST_pHistogramBucketEntryNext =
      GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, BEST_pHistogramBucketEntry, 1);
   ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, BEST_pHistogramBucketEntryNext, pThreadStateBoosting->GetHistogramBucketsEndDebug());

   TreeNode<bClassification> * const pRightChild = GetRightTreeNodeChild<bClassification>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);

   pRightChild->BEFORE_SetHistogramBucketEntryFirst(BEST_pHistogramBucketEntryNext);
   const size_t cSamplesParent = pTreeNode->AMBIGUOUS_GetSamples();
   pRightChild->AMBIGUOUS_SetSamples(cSamplesParent - BEST_cSamplesLeft);

   const FloatEbmType cSamplesParentFloatEbmType = static_cast<FloatEbmType>(cSamplesParent);

   // if the total samples is 0 then we should be using our specialty handling of that case
   // if the total samples if not 0, then our splitting code should never split any node that has zero on either the left or right, so no new 
   // parent should ever have zero samples
   EBM_ASSERT(0 < cSamplesParent);

   HistogramTargetEntry<bClassification> * pHistogramTargetEntryLeftChild =
      pLeftChild->GetHistogramTargetEntry();

   HistogramTargetEntry<bClassification> * pHistogramTargetEntryRightChild =
      pRightChild->GetHistogramTargetEntry();

   const HistogramTargetEntry<bClassification> * pHistogramTargetEntryTreeNode =
      pTreeNode->GetHistogramTargetEntry();

   const HistogramTargetEntry<bClassification> * pHistogramTargetEntrySweep =
      pTreeSweepStart->GetBestHistogramTargetEntry();

   // TODO: usually we've done this calculation for the parent already.  Why not keep the result arround to avoid extra work?
   FloatEbmType originalParentScore = 0;
   for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
      const FloatEbmType BEST_sumResidualErrorLeft = pHistogramTargetEntrySweep[iVector].m_sumResidualError;
      pHistogramTargetEntryLeftChild[iVector].m_sumResidualError = BEST_sumResidualErrorLeft;

      const FloatEbmType sumResidualErrorParent = pHistogramTargetEntryTreeNode[iVector].m_sumResidualError;
      pHistogramTargetEntryRightChild[iVector].m_sumResidualError = sumResidualErrorParent - BEST_sumResidualErrorLeft;

      const FloatEbmType originalParentScoreUpdate = EbmStats::ComputeNodeSplittingScore(sumResidualErrorParent, cSamplesParentFloatEbmType);
      EBM_ASSERT(std::isnan(originalParentScoreUpdate) || FloatEbmType { 0 } <= originalParentScoreUpdate);
      originalParentScore += originalParentScoreUpdate;

      if(bClassification) {
         const FloatEbmType BEST_sumDenominatorLeft = pHistogramTargetEntrySweep[iVector].GetSumDenominator();
         pHistogramTargetEntryLeftChild[iVector].SetSumDenominator(BEST_sumDenominatorLeft);
         pHistogramTargetEntryRightChild[iVector].SetSumDenominator(
            pHistogramTargetEntryTreeNode[iVector].GetSumDenominator() - BEST_sumDenominatorLeft
         );
      }
   }
   EBM_ASSERT(std::isnan(originalParentScore) || FloatEbmType { 0 } <= originalParentScore);


   // IMPORTANT!! : we need to finish all our calls that use this->m_UNION.m_beforeExaminationForPossibleSplitting BEFORE setting anything in 
   // m_UNION.m_afterExaminationForPossibleSplitting as we do below this comment!  The call above to this->GetSamples() needs to be done above 
   // these lines because it uses m_UNION.m_beforeExaminationForPossibleSplitting for classification!
#ifndef NDEBUG
   pTreeNode->SetExaminedForPossibleSplitting(true);
#endif // NDEBUG


   pTreeNode->AFTER_SetTreeNodeChildren(pTreeNodeChildrenAvailableStorageSpaceCur);
   const FloatEbmType splitGain = BEST_nodeSplittingScore - originalParentScore;
   // mathematically BEST_nodeSplittingScore should be bigger than originalParentScore, and the result positive, but these are numbers that are calculated
   //   from sumation, so they are inaccurate, and we could get a slighly negative number outcome

   // for regression, BEST_nodeSplittingScore and originalParentScore can be infinity.  There is a super-super-super-rare case where we can have 
   // originalParentScore overflow to +infinity due to numeric issues, but not BEST_nodeSplittingScore, and then the subtration causes the 
   // result to be -infinity. The universe will probably die of heat death before we get a -infinity value, but perhaps an adversarial dataset 
   // could trigger it, and we don't want someone giving us data to use a vulnerability in our system, so check for it!
   // within a set, no split should make our model worse.  It might in our validation set, but not within the training set
   EBM_ASSERT(std::isnan(splitGain) || (!bClassification) && std::isinf(splitGain) || k_epsilonNegativeGainAllowed <= splitGain);
   pTreeNode->AFTER_SetSplitGain(splitGain);

   HistogramBucketBase * const aHistogramBucketBase = pThreadStateBoosting->GetHistogramBucketBase();
   const HistogramBucket<bClassification> * const aHistogramBucket =
      aHistogramBucketBase->GetHistogramBucket<bClassification>();

   EBM_ASSERT(reinterpret_cast<const char *>(aHistogramBucket) <= reinterpret_cast<const char *>(BEST_pHistogramBucketEntry));
   EBM_ASSERT(0 == (reinterpret_cast<const char *>(BEST_pHistogramBucketEntry) - reinterpret_cast<const char *>(aHistogramBucket)) % cBytesPerHistogramBucket);
   pTreeNode->AFTER_SetDivisionValue((reinterpret_cast<const char *>(BEST_pHistogramBucketEntry) - 
      reinterpret_cast<const char *>(aHistogramBucket)) / cBytesPerHistogramBucket);

   LOG_N(
      TraceLevelVerbose,
      "Exited ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint: divisionValue=%zu, nodeSplittingScore=%" FloatEbmTypePrintf,
      static_cast<size_t>(pTreeNode->AFTER_GetDivisionValue()),
      pTreeNode->AFTER_GetSplitGain()
   );

   return false;
}

template<bool bClassification>
class CompareTreeNodeSplittingGain final {
public:
   // TODO : check how efficient this is.  Is there a faster way to to this
   INLINE_ALWAYS bool operator() (const TreeNode<bClassification> * const & lhs, const TreeNode<bClassification> * const & rhs) const noexcept {

      // TODO : VERY IMPORTANT, review if this is sorting in the correct direction!!!

      // NEVER check for exact equality (as a precondition is ok), since then we'd violate the weak ordering rule
      // https://medium.com/@shiansu/strict-weak-ordering-and-the-c-stl-f7dcfa4d4e07
      return lhs->AFTER_GetSplitGain() < rhs->AFTER_GetSplitGain();
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class GrowDecisionTreeInternal final {
public:

   GrowDecisionTreeInternal() = delete; // this is a static class.  Do not construct

   static bool Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const size_t cHistogramBuckets,
      const size_t cSamplesTotal,
      const size_t cSamplesRequiredForChildSplitMin,
      const size_t cLeavesMax,
      FloatEbmType * const pTotalGain
   ) {
      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      HistogramBucketBase * const aHistogramBucketBase = pThreadStateBoosting->GetHistogramBucketBase();
      const HistogramBucket<bClassification> * const aHistogramBucket =
         aHistogramBucketBase->GetHistogramBucket<bClassification>();

      HistogramTargetEntryBase * const aSumHistogramTargetEntryBase =
         pThreadStateBoosting->GetSumHistogramTargetEntryArray();
      const HistogramTargetEntry<bClassification> * const aSumHistogramTargetEntry =
         aSumHistogramTargetEntryBase->GetHistogramTargetEntry<bClassification>();

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();

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
         LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree GetTreeNodeSizeOverflow<bClassification>(cVectorLength)");
         return true; // we haven't accessed this TreeNode memory yet, so we don't know if it overflows yet
      }
      const size_t cBytesPerTreeNode = GetTreeNodeSize(bClassification, cVectorLength);
      EBM_ASSERT(!GetHistogramBucketSizeOverflow(bClassification, cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);

   retry_with_bigger_tree_node_children_array:

      size_t cBytesBuffer2 = pThreadStateBoosting->GetThreadByteBuffer2Size();
      // we need 1 TreeNode for the root, 1 for the left child of the root and 1 for the right child of the root
      const size_t cBytesInitialNeededAllocation = 3 * cBytesPerTreeNode;
      if(cBytesBuffer2 < cBytesInitialNeededAllocation) {
         // TODO : we can eliminate this check as long as we ensure that the ThreadByteBuffer2 is always initialized to be equal to the size of three 
         // TreeNodes (left and right) == GET_SIZEOF_ONE_TREE_NODE_CHILDREN(cBytesPerTreeNode), or the number of bins (interactions multiply bins) on the 
         // highest bin count feature
         if(pThreadStateBoosting->GrowThreadByteBuffer2(cBytesInitialNeededAllocation)) {
            LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree pThreadStateBoosting->GrowThreadByteBuffer2(cBytesInitialNeededAllocation)");
            return true;
         }
         cBytesBuffer2 = pThreadStateBoosting->GetThreadByteBuffer2Size();
         EBM_ASSERT(cBytesInitialNeededAllocation <= cBytesBuffer2);
      }
      TreeNode<bClassification> * pRootTreeNode =
         static_cast<TreeNode<bClassification> *>(pThreadStateBoosting->GetThreadByteBuffer2());

#ifndef NDEBUG
      pRootTreeNode->SetExaminedForPossibleSplitting(false);
#endif // NDEBUG

      pRootTreeNode->BEFORE_SetHistogramBucketEntryFirst(aHistogramBucket);
      pRootTreeNode->BEFORE_SetHistogramBucketEntryLast(
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, aHistogramBucket, cHistogramBuckets - 1)
      );
      ASSERT_BINNED_BUCKET_OK(
         cBytesPerHistogramBucket,
         pRootTreeNode->BEFORE_GetHistogramBucketEntryLast(),
         pThreadStateBoosting->GetHistogramBucketsEndDebug()
      );
      pRootTreeNode->AMBIGUOUS_SetSamples(cSamplesTotal);

      // copying existing mem
      memcpy(
         pRootTreeNode->GetHistogramTargetEntry(),
         aSumHistogramTargetEntry,
         cVectorLength * sizeof(*aSumHistogramTargetEntry)
      );

      SegmentedTensor * const pSmallChangeToModelOverwriteSingleSamplingSet =
         pThreadStateBoosting->GetSmallChangeToModelOverwriteSingleSamplingSet();

      size_t cLeaves;
      if(ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint<compilerLearningTypeOrCountTargetClasses>(
         pThreadStateBoosting,
         pRootTreeNode,
         AddBytesTreeNode<bClassification>(pRootTreeNode, cBytesPerTreeNode),
         cSamplesRequiredForChildSplitMin
      )) {
         // there will be no splits at all
         if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 0))) {
            LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 0)");
            return true;
         }

         // we don't need to call EnsureValueCapacity because by default we start with a value capacity of 2 * cVectorLength
         if(bClassification) {
            FloatEbmType * const aValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               const FloatEbmType smallChangeToModel = EbmStats::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                  pRootTreeNode->GetHistogramTargetEntry()[iVector].m_sumResidualError, pRootTreeNode->GetHistogramTargetEntry()[iVector].GetSumDenominator()
               );
               aValues[iVector] = smallChangeToModel;
            }
         } else {
            EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
            const FloatEbmType smallChangeToModel = EbmStats::ComputeSmallChangeForOneSegmentRegression(
               pRootTreeNode->GetHistogramTargetEntry()[0].m_sumResidualError, static_cast<FloatEbmType>(cSamplesTotal)
            );
            FloatEbmType * pValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
            pValues[0] = smallChangeToModel;
         }

         *pTotalGain = FloatEbmType { 0 };
         return false;
      }

      if(UNPREDICTABLE(PREDICTABLE(2 == cLeavesMax) || UNPREDICTABLE(2 == cHistogramBuckets))) {
         // there will be exactly 1 split, which is a special case that we can return faster without as much overhead as the multiple split case

         EBM_ASSERT(2 != cHistogramBuckets || !GetLeftTreeNodeChild<bClassification>(
            pRootTreeNode->AFTER_GetTreeNodeChildren(), cBytesPerTreeNode)->IsSplittable() &&
            !GetRightTreeNodeChild<bClassification>(
               pRootTreeNode->AFTER_GetTreeNodeChildren(),
               cBytesPerTreeNode
               )->IsSplittable()
         );

         if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1))) {
            LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1)");
            return true;
         }

         ActiveDataType * pDivisions = pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0);
         pDivisions[0] = pRootTreeNode->AFTER_GetDivisionValue();

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

         const HistogramTargetEntry<bClassification> * pHistogramTargetEntryLeftChild =
            pLeftChild->GetHistogramTargetEntry();

         const HistogramTargetEntry<bClassification> * pHistogramTargetEntryRightChild =
            pRightChild->GetHistogramTargetEntry();

         FloatEbmType * const aValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
         if(bClassification) {
            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               aValues[iVector] = EbmStats::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                  pHistogramTargetEntryLeftChild[iVector].m_sumResidualError,
                  pHistogramTargetEntryLeftChild[iVector].GetSumDenominator()
               );
               aValues[cVectorLength + iVector] = EbmStats::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                  pHistogramTargetEntryRightChild[iVector].m_sumResidualError,
                  pHistogramTargetEntryRightChild[iVector].GetSumDenominator()
               );
            }
         } else {
            EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
            aValues[0] = EbmStats::ComputeSmallChangeForOneSegmentRegression(
               pHistogramTargetEntryLeftChild[0].m_sumResidualError,
               static_cast<FloatEbmType>(pLeftChild->AMBIGUOUS_GetSamples())
            );
            aValues[1] = EbmStats::ComputeSmallChangeForOneSegmentRegression(
               pHistogramTargetEntryRightChild[0].m_sumResidualError,
               static_cast<FloatEbmType>(pRightChild->AMBIGUOUS_GetSamples())
            );
         }

         const FloatEbmType totalGain = pRootTreeNode->EXTRACT_GAIN_BEFORE_SPLITTING();
         EBM_ASSERT(std::isnan(totalGain) || (!bClassification) && std::isinf(totalGain) || k_epsilonNegativeGainAllowed <= totalGain);
         // don't normalize totalGain here, because we normalize the average outside of this function!
         *pTotalGain = totalGain;
         return false;
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
            // either this is our first cut, or we've only made a single cut in the center in the case where there is symetry in the center
            // Even if all of these things are true, after one non-symetric cut, we won't see that scenario anymore since the residuals won't be
            // symetric anymore.  This is so rare, and limited to one cut, so we shouldn't bother to handle it since the complexity of doing so
            // outweights the benefits.
            bestTreeNodeToSplit.pop();

         skip_first_push_pop:

            // ONLY AFTER WE'VE POPPED pParentTreeNode OFF the priority queue is it considered to have been split.  Calling SPLIT_THIS_NODE makes it formal
            const FloatEbmType totalGainUpdate = pParentTreeNode->EXTRACT_GAIN_BEFORE_SPLITTING();
            EBM_ASSERT(std::isnan(totalGainUpdate) || (!bClassification) && std::isinf(totalGainUpdate) ||
               k_epsilonNegativeGainAllowed <= totalGainUpdate);
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
                  if(pThreadStateBoosting->GrowThreadByteBuffer2(cBytesPerTreeNode)) {
                     LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree pThreadStateBoosting->GrowThreadByteBuffer2(cBytesPerTreeNode)");
                     return true;
                  }
                  goto retry_with_bigger_tree_node_children_array;
               }
               // the act of splitting it implicitly sets INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED
               // because splitting sets splitGain to a non-illegalGain value
               if(!ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint<compilerLearningTypeOrCountTargetClasses>(
                  pThreadStateBoosting,
                  pLeftChild,
                  pTreeNodeChildrenAvailableStorageSpaceCur,
                  cSamplesRequiredForChildSplitMin
               )) {
                  pTreeNodeChildrenAvailableStorageSpaceCur = pTreeNodeChildrenAvailableStorageSpaceNext;
                  bestTreeNodeToSplit.push(pLeftChild);
               } else {
                  goto no_left_split;
               }
            } else {
            no_left_split:;
               // we aren't going to split this TreeNode because we can't.  We need to set the splitGain value here because otherwise it is filled with 
               // garbage that could be k_illegalGain (meaning the node was a branch) we can't call INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED 
               // before calling SplitTreeNode because INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED sets 
               // m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain and the m_UNION.m_beforeExaminationForPossibleSplitting values are 
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
                  if(pThreadStateBoosting->GrowThreadByteBuffer2(cBytesPerTreeNode)) {
                     LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree pThreadStateBoosting->GrowThreadByteBuffer2(cBytesPerTreeNode)");
                     return true;
                  }
                  goto retry_with_bigger_tree_node_children_array;
               }
               // the act of splitting it implicitly sets INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED 
               // because splitting sets splitGain to a non-NaN value
               if(!ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint<compilerLearningTypeOrCountTargetClasses>(
                  pThreadStateBoosting,
                  pRightChild,
                  pTreeNodeChildrenAvailableStorageSpaceCur,
                  cSamplesRequiredForChildSplitMin
               )) {
                  pTreeNodeChildrenAvailableStorageSpaceCur = pTreeNodeChildrenAvailableStorageSpaceNext;
                  bestTreeNodeToSplit.push(pRightChild);
               } else {
                  goto no_right_split;
               }
            } else {
            no_right_split:;
               // we aren't going to split this TreeNode because we can't.  We need to set the splitGain value here because otherwise it is filled with 
               // garbage that could be k_illegalGain (meaning the node was a branch) we can't call INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED 
               // before calling SplitTreeNode because INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED sets 
               // m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain and the m_UNION.m_beforeExaminationForPossibleSplitting values are needed 
               // if we had decided to call ExamineNodeForSplittingAndDetermineBestPossibleSplit

#ifndef NDEBUG
               pRightChild->SetExaminedForPossibleSplitting(true);
#endif // NDEBUG

               pRightChild->INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED();
            }
            ++cLeaves;
         } while(cLeaves < cLeavesMax && UNLIKELY(!bestTreeNodeToSplit.empty()));
         // we DON'T need to call SetLeafAfterDone() on any items that remain in the bestTreeNodeToSplit queue because everything in that queue has set 
         // a non-NaN nodeSplittingScore value

         // regression can be -infinity or slightly negative in extremely rare circumstances.
         // See ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint for details, and the equivalent interaction function
         EBM_ASSERT(std::isnan(totalGain) || (!bClassification) && std::isinf(totalGain) ||
            k_epsilonNegativeGainAllowed <= totalGain);
         // we might as well dump this value out to our pointer, even if later fail the function below.  If the function is failed, we make no guarantees 
         // about what we did with the value pointed to at *pTotalGain don't normalize totalGain here, because we normalize the average outside of this function!
         *pTotalGain = totalGain;
         EBM_ASSERT(
            static_cast<size_t>(reinterpret_cast<char *>(pTreeNodeChildrenAvailableStorageSpaceCur) - reinterpret_cast<char *>(pRootTreeNode)) <= cBytesBuffer2
         );
      } catch(...) {
         // calling anything inside bestTreeNodeToSplit can throw exceptions
         LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree exception");
         return true;
      }

      if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, cLeaves - size_t { 1 }))) {
         LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, cLeaves - size_t { 1 })");
         return true;
      }
      if(IsMultiplyError(cVectorLength, cLeaves)) {
         LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree IsMultiplyError(cVectorLength, cLeaves)");
         return true;
      }
      if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * cLeaves))) {
         LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * cLeaves");
         return true;
      }
      ActiveDataType * pDivisions = pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0);
      FloatEbmType * pValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();

      LOG_0(TraceLevelVerbose, "Entered Flatten");
      Flatten<bClassification>(pRootTreeNode, &pDivisions, &pValues, cVectorLength);
      LOG_0(TraceLevelVerbose, "Exited Flatten");

      EBM_ASSERT(pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0) <= pDivisions);
      EBM_ASSERT(static_cast<size_t>(pDivisions - pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)) == cLeaves - 1);
      EBM_ASSERT(pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer() < pValues);
      EBM_ASSERT(static_cast<size_t>(pValues - pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()) == cVectorLength * cLeaves);

      return false;
   }
};

extern bool GrowDecisionTree(
   ThreadStateBoosting * const pThreadStateBoosting,
   const size_t cHistogramBuckets,
   const size_t cSamplesTotal,
   const size_t cSamplesRequiredForChildSplitMin,
   const size_t cLeavesMax,
   FloatEbmType * const pTotalGain
) {
   LOG_0(TraceLevelVerbose, "Entered GrowDecisionTree");

   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();

   bool bRet;
   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      if(IsBinaryClassification(runtimeLearningTypeOrCountTargetClasses)) {
         bRet = GrowDecisionTreeInternal<2>::Func(
            pThreadStateBoosting,
            cHistogramBuckets,
            cSamplesTotal,
            cSamplesRequiredForChildSplitMin,
            cLeavesMax,
            pTotalGain
         );
      } else {
         bRet = GrowDecisionTreeInternal<k_dynamicClassification>::Func(
            pThreadStateBoosting,
            cHistogramBuckets,
            cSamplesTotal,
            cSamplesRequiredForChildSplitMin,
            cLeavesMax,
            pTotalGain
         );
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      bRet = GrowDecisionTreeInternal<k_regression>::Func(
         pThreadStateBoosting,
         cHistogramBuckets,
         cSamplesTotal,
         cSamplesRequiredForChildSplitMin,
         cLeavesMax,
         pTotalGain
      );
   }

   LOG_0(TraceLevelVerbose, "Exited GrowDecisionTree");

   return bRet;
}
