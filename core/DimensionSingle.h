// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DIMENSION_SINGLE_H
#define DIMENSION_SINGLE_H

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "SegmentedTensor.h"
#include "EbmStatistics.h"
#include "CachedThreadResources.h"
#include "FeatureCore.h"
#include "SamplingWithReplacement.h"
#include "HistogramBucket.h"

#include "TreeNode.h"

template<bool bClassification>
struct SweepTreeNode {
   size_t m_cBestInstancesLeft;
   const HistogramBucket<bClassification> * m_pBestHistogramBucketEntry;
   
   // use the "struct hack" since Flexible array member method is not available in C++
   // m_aHistogramBucketVectorEntry must be the last item in this struct
   HistogramBucketVectorEntry<bClassification> m_aBestHistogramBucketVectorEntry[1];
};
template<bool bClassification>
EBM_INLINE bool GetSweepTreeNodeSizeOverflow(const size_t cVectorLength) {
   return IsMultiplyError(sizeof(HistogramBucketVectorEntry<bClassification>), cVectorLength) ? true : IsAddError(sizeof(SweepTreeNode<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>), sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength) ? true : false;
}
template<bool bClassification>
EBM_INLINE size_t GetSweepTreeNodeSize(const size_t cVectorLength) {
   return sizeof(SweepTreeNode<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>) + sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength;
}
template<bool bClassification>
EBM_INLINE SweepTreeNode<bClassification> * AddBytesSweepTreeNode(SweepTreeNode<bClassification> * const pSweepTreeNode, const size_t countBytesAdd) {
   return reinterpret_cast<SweepTreeNode<bClassification> *>(reinterpret_cast<char *>(pSweepTreeNode) + countBytesAdd);
}


template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
void ExamineNodeForPossibleSplittingAndDetermineBestSplitPoint(TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTreeNode, CachedBoostingThreadResources<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pCachedThreadResources, TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pTreeNodeChildrenAvailableStorageSpaceCur, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   LOG_N(TraceLevelVerbose, "Entered SplitTreeNode: pTreeNode=%p, pTreeNodeChildrenAvailableStorageSpaceCur=%p", static_cast<void *>(pTreeNode), static_cast<void *>(pTreeNodeChildrenAvailableStorageSpaceCur));

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses);
   EBM_ASSERT(!GetTreeNodeSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerTreeNode = GetTreeNodeSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);

   const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pHistogramBucketEntryCur = pTreeNode->m_UNION.m_beforeExaminationForPossibleSplitting.m_pHistogramBucketEntryFirst;
   const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pHistogramBucketEntryLast = pTreeNode->m_UNION.m_beforeExaminationForPossibleSplitting.m_pHistogramBucketEntryLast;

   TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pLeftChild1 = GetLeftTreeNodeChild<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);
   pLeftChild1->m_UNION.m_beforeExaminationForPossibleSplitting.m_pHistogramBucketEntryFirst = pHistogramBucketEntryCur;
   TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pRightChild1 = GetRightTreeNodeChild<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);
   pRightChild1->m_UNION.m_beforeExaminationForPossibleSplitting.m_pHistogramBucketEntryLast = pHistogramBucketEntryLast;

   size_t cInstancesLeft = pHistogramBucketEntryCur->m_cInstancesInBucket;
   size_t cInstancesRight = pTreeNode->GetInstances() - cInstancesLeft;

   HistogramBucketVectorEntry<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aSumHistogramBucketVectorEntryLeft = pCachedThreadResources->m_aSumHistogramBucketVectorEntry1;
   FractionalDataType * const aSumResidualErrorsRight = pCachedThreadResources->m_aSumResidualErrors2;
   HistogramBucketVectorEntry<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aSumHistogramBucketVectorEntryBest = pCachedThreadResources->m_aSumHistogramBucketVectorEntryBest;
   FractionalDataType BEST_nodeSplittingScore = 0;
   for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
      const FractionalDataType sumResidualErrorLeft = ARRAY_TO_POINTER_CONST(pHistogramBucketEntryCur->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError;
      const FractionalDataType sumResidualErrorRight = ARRAY_TO_POINTER_CONST(pTreeNode->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError - sumResidualErrorLeft;

      BEST_nodeSplittingScore += EbmStatistics::ComputeNodeSplittingScore(sumResidualErrorLeft, static_cast<FractionalDataType>(cInstancesLeft)) + EbmStatistics::ComputeNodeSplittingScore(sumResidualErrorRight, static_cast<FractionalDataType>(cInstancesRight));

      aSumHistogramBucketVectorEntryLeft[iVector].m_sumResidualError = sumResidualErrorLeft;
      aSumHistogramBucketVectorEntryBest[iVector].m_sumResidualError = sumResidualErrorLeft;
      aSumResidualErrorsRight[iVector] = sumResidualErrorRight;
      if(IsClassification(compilerLearningTypeOrCountTargetClasses)) {
         FractionalDataType sumDenominator1 = ARRAY_TO_POINTER_CONST(pHistogramBucketEntryCur->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator();
         aSumHistogramBucketVectorEntryLeft[iVector].SetSumDenominator(sumDenominator1);
         aSumHistogramBucketVectorEntryBest[iVector].SetSumDenominator(sumDenominator1);
      }
   }

   EBM_ASSERT(0 <= BEST_nodeSplittingScore);
   const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * BEST_pHistogramBucketEntry = pHistogramBucketEntryCur;
   size_t BEST_cInstancesLeft = cInstancesLeft;
   for(pHistogramBucketEntryCur = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucketEntryCur, 1); pHistogramBucketEntryLast != pHistogramBucketEntryCur; pHistogramBucketEntryCur = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucketEntryCur, 1)) {
      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucketEntryCur, aHistogramBucketsEndDebug);

      const size_t CHANGE_cInstances = pHistogramBucketEntryCur->m_cInstancesInBucket;
      cInstancesLeft += CHANGE_cInstances;
      cInstancesRight -= CHANGE_cInstances;

      FractionalDataType nodeSplittingScore = 0;
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         if(IsClassification(compilerLearningTypeOrCountTargetClasses)) {
            aSumHistogramBucketVectorEntryLeft[iVector].SetSumDenominator(aSumHistogramBucketVectorEntryLeft[iVector].GetSumDenominator() + ARRAY_TO_POINTER_CONST(pHistogramBucketEntryCur->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator());
         }

         const FractionalDataType CHANGE_sumResidualError = ARRAY_TO_POINTER_CONST(pHistogramBucketEntryCur->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError;
         const FractionalDataType sumResidualErrorLeft = aSumHistogramBucketVectorEntryLeft[iVector].m_sumResidualError + CHANGE_sumResidualError;
         const FractionalDataType sumResidualErrorRight = aSumResidualErrorsRight[iVector] - CHANGE_sumResidualError;

         aSumHistogramBucketVectorEntryLeft[iVector].m_sumResidualError = sumResidualErrorLeft;
         aSumResidualErrorsRight[iVector] = sumResidualErrorRight;

         // TODO : we can make this faster by doing the division in ComputeNodeSplittingScore after we add all the numerators
         const FractionalDataType nodeSplittingScoreOneVector = EbmStatistics::ComputeNodeSplittingScore(sumResidualErrorLeft, static_cast<FractionalDataType>(cInstancesLeft)) + EbmStatistics::ComputeNodeSplittingScore(sumResidualErrorRight, static_cast<FractionalDataType>(cInstancesRight));
         EBM_ASSERT(0 <= nodeSplittingScore);
         nodeSplittingScore += nodeSplittingScoreOneVector;
      }
      EBM_ASSERT(0 <= nodeSplittingScore);

      if(UNLIKELY(BEST_nodeSplittingScore < nodeSplittingScore)) {
         // it's very possible that we have bins with zero instances in them, in which case we could easily be presented with equally favorable splits
         // or it's even possible for two different possible unrelated sections of bins, or individual bins to have exactly the same gain (think low count symetric data)
         // we want to avoid any bias of always choosing the higher or lower value to split on, so what we should do is store the indexes of any ties in a stack
         // and we reset the stack if we later find a gain that's larger than any we have in the stack.  The stack needs to be size_t to hold indexes, and we need
         // the stack to be as long as the number of instances - 1, incase all gain for all bins are the same (potential_splits = bins - 1)
         // after we exit the loop we can examine our stack and choose a random split from all the equivalent splits available
         // eg: we find that items at index 4,7,8,9 all have the same gain, so we pick a random number between 0 -> 3 to select which one we actually split on
         //
         // TODO : implement the randomized splitting described above.  Also, do this for interaction effect which can be done the same althoug we might want to include near matches since there is floating point noise there due to the way we sum interaction effect region totals

         BEST_nodeSplittingScore = nodeSplittingScore;
         BEST_pHistogramBucketEntry = pHistogramBucketEntryCur;
         BEST_cInstancesLeft = cInstancesLeft;
         memcpy(aSumHistogramBucketVectorEntryBest, aSumHistogramBucketVectorEntryLeft, sizeof(*aSumHistogramBucketVectorEntryBest) * cVectorLength);
      }
   }

   TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pLeftChild = GetLeftTreeNodeChild<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);
   TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pRightChild = GetRightTreeNodeChild<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);

   pLeftChild->m_UNION.m_beforeExaminationForPossibleSplitting.m_pHistogramBucketEntryLast = BEST_pHistogramBucketEntry;
   pLeftChild->SetInstances(BEST_cInstancesLeft);

   const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const BEST_pHistogramBucketEntryNext = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, BEST_pHistogramBucketEntry, 1);
   ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, BEST_pHistogramBucketEntryNext, aHistogramBucketsEndDebug);

   pRightChild->m_UNION.m_beforeExaminationForPossibleSplitting.m_pHistogramBucketEntryFirst = BEST_pHistogramBucketEntryNext;
   size_t cInstancesParent = pTreeNode->GetInstances();
   pRightChild->SetInstances(cInstancesParent - BEST_cInstancesLeft);

   FractionalDataType originalParentScore = 0;
   for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
      ARRAY_TO_POINTER(pLeftChild->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError = aSumHistogramBucketVectorEntryBest[iVector].m_sumResidualError;
      if(IsClassification(compilerLearningTypeOrCountTargetClasses)) {
         ARRAY_TO_POINTER(pLeftChild->m_aHistogramBucketVectorEntry)[iVector].SetSumDenominator(aSumHistogramBucketVectorEntryBest[iVector].GetSumDenominator());
      }

      const FractionalDataType sumResidualErrorParent = ARRAY_TO_POINTER(pTreeNode->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError;
      originalParentScore += EbmStatistics::ComputeNodeSplittingScore(sumResidualErrorParent, static_cast<FractionalDataType>(cInstancesParent));

      ARRAY_TO_POINTER(pRightChild->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError = sumResidualErrorParent - aSumHistogramBucketVectorEntryBest[iVector].m_sumResidualError;
      if(IsClassification(compilerLearningTypeOrCountTargetClasses)) {
         ARRAY_TO_POINTER(pRightChild->m_aHistogramBucketVectorEntry)[iVector].SetSumDenominator(ARRAY_TO_POINTER(pTreeNode->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator() - aSumHistogramBucketVectorEntryBest[iVector].GetSumDenominator());
      }
   }



   // IMPORTANT!! : we need to finish all our calls that use this->m_UNION.m_beforeExaminationForPossibleSplitting BEFORE setting anything in m_UNION.m_afterExaminationForPossibleSplitting as we do below this comment!  The call above to this->GetInstances() needs to be done above these lines because it uses m_UNION.m_beforeExaminationForPossibleSplitting for classification!



   pTreeNode->m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren = pTreeNodeChildrenAvailableStorageSpaceCur;
   FractionalDataType splitGain = originalParentScore - BEST_nodeSplittingScore;
   if(UNLIKELY(std::isnan(splitGain))) {
      // it is possible that nodeSplittingScoreParent could reach infinity and BEST_nodeSplittingScore infinity, and the subtraction of those values leads to NaN
      // if gain became NaN via overlfow, that would be bad since we use NaN to indicate that a node has not been split
      splitGain = FractionalDataType { 0 };
   }
   pTreeNode->m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain = splitGain;
   pTreeNode->m_UNION.m_afterExaminationForPossibleSplitting.m_divisionValue = (BEST_pHistogramBucketEntry->m_bucketValue + BEST_pHistogramBucketEntryNext->m_bucketValue) / 2;

   EBM_ASSERT(pTreeNode->m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain <= 0.0000000001); // within a set, no split should make our model worse.  It might in our validation set, but not within this set

   LOG_N(TraceLevelVerbose, "Exited SplitTreeNode: divisionValue=%zu, nodeSplittingScore=%" FractionalDataTypePrintf, static_cast<size_t>(pTreeNode->m_UNION.m_afterExaminationForPossibleSplitting.m_divisionValue), pTreeNode->m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain);
}

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
bool GrowDecisionTree(CachedBoostingThreadResources<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pCachedThreadResources, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, const size_t cHistogramBuckets, const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucket, const size_t cInstancesTotal, const HistogramBucketVectorEntry<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aSumHistogramBucketVectorEntry, const size_t cTreeSplitsMax, const size_t cInstancesRequiredForParentSplitMin, SegmentedTensor<ActiveDataType, FractionalDataType> * const pSmallChangeToModelOverwriteSingleSamplingSet, FractionalDataType * const pTotalGain
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   LOG_0(TraceLevelVerbose, "Entered GrowDecisionTree");

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses);

   EBM_ASSERT(nullptr != pTotalGain);
   EBM_ASSERT(1 <= cInstancesTotal); // filter these out at the start where we can handle this case easily
   EBM_ASSERT(1 <= cHistogramBuckets); // cHistogramBuckets could only be zero if cInstancesTotal.  We should filter out that special case at our entry point though!!
   if(UNLIKELY(cInstancesTotal < cInstancesRequiredForParentSplitMin || 1 == cHistogramBuckets || 0 == cTreeSplitsMax)) {
      // there will be no splits at all

      if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 0))) {
         LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 0)");
         return true;
      }

      // we don't need to call EnsureValueCapacity because by default we start with a value capacity of 2 * cVectorLength

      if(IsClassification(compilerLearningTypeOrCountTargetClasses)) {
         FractionalDataType * aValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            FractionalDataType smallChangeToModel = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(aSumHistogramBucketVectorEntry[iVector].m_sumResidualError, aSumHistogramBucketVectorEntry[iVector].GetSumDenominator());
            aValues[iVector] = smallChangeToModel;
         }
      } else {
         EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
         FractionalDataType smallChangeToModel = EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(aSumHistogramBucketVectorEntry[0].m_sumResidualError, static_cast<FractionalDataType>(cInstancesTotal));
         FractionalDataType * pValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
         pValues[0] = smallChangeToModel;
      }

      LOG_0(TraceLevelVerbose, "Exited GrowDecisionTree via not enough data to split");
      *pTotalGain = 0;
      return false;
   }

   // there will be at least one split

   if(GetTreeNodeSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)) {
      LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree GetTreeNodeSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)");
      return true; // we haven't accessed this TreeNode memory yet, so we don't know if it overflows yet
   }
   const size_t cBytesPerTreeNode = GetTreeNodeSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);

retry_with_bigger_tree_node_children_array:
   size_t cBytesBuffer2 = pCachedThreadResources->GetThreadByteBuffer2Size();
   const size_t cBytesInitialNeededAllocation = 3 * cBytesPerTreeNode; // we need 1 TreeNode for the root, 1 for the left child of the root and 1 for the right child of the root
   if(cBytesBuffer2 < cBytesInitialNeededAllocation) {
      // TODO : we can eliminate this check as long as we ensure that the ThreadByteBuffer2 is always initialized to be equal to the size of three TreeNodes (left and right) == GET_SIZEOF_ONE_TREE_NODE_CHILDREN(cBytesPerTreeNode)
      if(pCachedThreadResources->GrowThreadByteBuffer2(cBytesInitialNeededAllocation)) {
         LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree pCachedThreadResources->GrowThreadByteBuffer2(cBytesInitialNeededAllocation)");
         return true;
      }
      cBytesBuffer2 = pCachedThreadResources->GetThreadByteBuffer2Size();
      EBM_ASSERT(cBytesInitialNeededAllocation <= cBytesBuffer2);
   }
   TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pRootTreeNode = static_cast<TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> *>(pCachedThreadResources->GetThreadByteBuffer2());

   pRootTreeNode->m_UNION.m_beforeExaminationForPossibleSplitting.m_pHistogramBucketEntryFirst = aHistogramBucket;
   pRootTreeNode->m_UNION.m_beforeExaminationForPossibleSplitting.m_pHistogramBucketEntryLast = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aHistogramBucket, cHistogramBuckets - 1);
   ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pRootTreeNode->m_UNION.m_beforeExaminationForPossibleSplitting.m_pHistogramBucketEntryLast, aHistogramBucketsEndDebug);
   pRootTreeNode->SetInstances(cInstancesTotal);

   memcpy(ARRAY_TO_POINTER(pRootTreeNode->m_aHistogramBucketVectorEntry), aSumHistogramBucketVectorEntry, cVectorLength * sizeof(*aSumHistogramBucketVectorEntry)); // copying existing mem

   ExamineNodeForPossibleSplittingAndDetermineBestSplitPoint<compilerLearningTypeOrCountTargetClasses>(pRootTreeNode, pCachedThreadResources, AddBytesTreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pRootTreeNode, cBytesPerTreeNode), runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
      , aHistogramBucketsEndDebug
#endif // NDEBUG
   );

   if(UNPREDICTABLE(PREDICTABLE(1 == cTreeSplitsMax) || UNPREDICTABLE(2 == cHistogramBuckets))) {
      // there will be exactly 1 split, which is a special case that we can return faster without as much overhead as the multiple split case

      EBM_ASSERT(2 != cHistogramBuckets || !GetLeftTreeNodeChild<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pRootTreeNode->m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren, cBytesPerTreeNode)->IsSplittable(cInstancesRequiredForParentSplitMin) && !GetRightTreeNodeChild<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pRootTreeNode->m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren, cBytesPerTreeNode)->IsSplittable(cInstancesRequiredForParentSplitMin));

      if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1))) {
         LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1)");
         return true;
      }

      ActiveDataType * pDivisions = pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0);
      pDivisions[0] = pRootTreeNode->m_UNION.m_afterExaminationForPossibleSplitting.m_divisionValue;

      // we don't need to call EnsureValueCapacity because by default we start with a value capacity of 2 * cVectorLength

      // TODO : we don't need to get the right and left pointer from the root.. we know where they will be
      const TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pLeftChild = GetLeftTreeNodeChild<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pRootTreeNode->m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren, cBytesPerTreeNode);
      const TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pRightChild = GetRightTreeNodeChild<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pRootTreeNode->m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren, cBytesPerTreeNode);

      FractionalDataType * const aValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
      if(IsClassification(compilerLearningTypeOrCountTargetClasses)) {
         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            aValues[iVector] = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(ARRAY_TO_POINTER_CONST(pLeftChild->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, ARRAY_TO_POINTER_CONST(pLeftChild->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator());
            aValues[cVectorLength + iVector] = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(ARRAY_TO_POINTER_CONST(pRightChild->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, ARRAY_TO_POINTER_CONST(pRightChild->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator());
         }
      } else {
         EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
         aValues[0] = EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(ARRAY_TO_POINTER_CONST(pLeftChild->m_aHistogramBucketVectorEntry)[0].m_sumResidualError, static_cast<FractionalDataType>(pLeftChild->GetInstances()));
         aValues[1] = EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(ARRAY_TO_POINTER_CONST(pRightChild->m_aHistogramBucketVectorEntry)[0].m_sumResidualError, static_cast<FractionalDataType>(pRightChild->GetInstances()));
      }

      LOG_0(TraceLevelVerbose, "Exited GrowDecisionTree via one tree split");
      *pTotalGain = pRootTreeNode->EXTRACT_GAIN_BEFORE_SPLITTING();
      return false;
   }

   // it's very very likely that there will be more than 1 split below this point.  The only case where we wouldn't split below is if both our children nodes dont't have enough cases
   // to split, but that should be very rare

   // TODO: there are three types of queues that we should try out -> dyamically picking a stragety is a single predictable if statement, so shouldn't cause a lot of overhead
   //       1) When the data is the smallest(1-5 items), just iterate over all items in our TreeNode buffer looking for the best Node.  Zero the value on any nodes that have been removed from the queue.  For 1 or 2 instructions in the loop WITHOUT a branch we can probably save the pointer to the first TreeNode with data so that we can start from there next time we loop
   //       2) When the data is a tiny bit bigger and there are holes in our array of TreeNodes, we can maintain a pointer and value in a separate list and zip through the values and then go to the pointer to the best node.  Since the list is unordered, when we find a TreeNode to remove, we just move the last one into the hole
   //       3) The full fleged priority queue below
   size_t cSplits;
   try {
      std::priority_queue<TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> *, std::vector<TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> *>, CompareTreeNodeSplittingGain<IsClassification(compilerLearningTypeOrCountTargetClasses)>> * pBestTreeNodeToSplit = &pCachedThreadResources->m_bestTreeNodeToSplit.m_queue;

      // it is ridiculous that we need to do this in order to clear the tree (there is no "clear" function), but inside this queue is a chunk of memory, and we want to ensure that the chunk of memory stays in L1 cache, so we pop all the previous garbage off instead of allocating a new one!
      while(!pBestTreeNodeToSplit->empty()) {
         pBestTreeNodeToSplit->pop();
      }

      cSplits = 0;
      TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pParentTreeNode = pRootTreeNode;

      // we skip 3 tree nodes.  The root, the left child of the root, and the right child of the root
      TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTreeNodeChildrenAvailableStorageSpaceCur = AddBytesTreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pRootTreeNode, cBytesInitialNeededAllocation);

      FractionalDataType totalGain = 0;

      goto skip_first_push_pop;

      do {
         // there is no way to get the top and pop at the same time.. would be good to get a better queue, but our code isn't bottlenecked by it
         pParentTreeNode = pBestTreeNodeToSplit->top();
         // TODO : after we pop the TreeNode, we should peek to see if the next best one has exactly the same gain, and if so we should continue pulling them until we find a lower gain,
         //        then we should randomly choose a node to split, and re-insert the remaining nodes back into the heap
         pBestTreeNodeToSplit->pop();

      skip_first_push_pop:

         // ONLY AFTER WE'VE POPPED pParentTreeNode OFF the priority queue is it considered to have been split.  Calling SPLIT_THIS_NODE makes it formal
         totalGain += pParentTreeNode->EXTRACT_GAIN_BEFORE_SPLITTING();
         pParentTreeNode->SPLIT_THIS_NODE();

         TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pLeftChild = GetLeftTreeNodeChild<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pParentTreeNode->m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren, cBytesPerTreeNode);
         if(pLeftChild->IsSplittable(cInstancesRequiredForParentSplitMin)) {
            TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTreeNodeChildrenAvailableStorageSpaceNext = AddBytesTreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode << 1);
            if(cBytesBuffer2 < static_cast<size_t>(reinterpret_cast<char *>(pTreeNodeChildrenAvailableStorageSpaceNext) - reinterpret_cast<char *>(pRootTreeNode))) {
               if(pCachedThreadResources->GrowThreadByteBuffer2(cBytesPerTreeNode)) {
                  LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree pCachedThreadResources->GrowThreadByteBuffer2(cBytesPerTreeNode)");
                  return true;
               }
               goto retry_with_bigger_tree_node_children_array;
            }
            // the act of splitting it implicitly sets INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED because splitting sets splitGain to a non-NaN value
            ExamineNodeForPossibleSplittingAndDetermineBestSplitPoint<compilerLearningTypeOrCountTargetClasses>(pLeftChild, pCachedThreadResources, pTreeNodeChildrenAvailableStorageSpaceCur, runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
               , aHistogramBucketsEndDebug
#endif // NDEBUG
            );
            pTreeNodeChildrenAvailableStorageSpaceCur = pTreeNodeChildrenAvailableStorageSpaceNext;
            pBestTreeNodeToSplit->push(pLeftChild);
         } else {
            // we aren't going to split this TreeNode because we can't.  We need to set the splitGain value here because otherwise it is filled with garbage that could be NaN (meaning the node was a branch)
            // we can't call INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED before calling SplitTreeNode because INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED sets m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain and the m_UNION.m_beforeExaminationForPossibleSplitting values are needed if we had decided to call ExamineNodeForSplittingAndDetermineBestPossibleSplit
            pLeftChild->INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED();
         }

         TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pRightChild = GetRightTreeNodeChild<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pParentTreeNode->m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren, cBytesPerTreeNode);
         if(pRightChild->IsSplittable(cInstancesRequiredForParentSplitMin)) {
            TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTreeNodeChildrenAvailableStorageSpaceNext = AddBytesTreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode << 1);
            if(cBytesBuffer2 < static_cast<size_t>(reinterpret_cast<char *>(pTreeNodeChildrenAvailableStorageSpaceNext) - reinterpret_cast<char *>(pRootTreeNode))) {
               if(pCachedThreadResources->GrowThreadByteBuffer2(cBytesPerTreeNode)) {
                  LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree pCachedThreadResources->GrowThreadByteBuffer2(cBytesPerTreeNode)");
                  return true;
               }
               goto retry_with_bigger_tree_node_children_array;
            }
            // the act of splitting it implicitly sets INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED because splitting sets splitGain to a non-NaN value
            ExamineNodeForPossibleSplittingAndDetermineBestSplitPoint<compilerLearningTypeOrCountTargetClasses>(pRightChild, pCachedThreadResources, pTreeNodeChildrenAvailableStorageSpaceCur, runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
               , aHistogramBucketsEndDebug
#endif // NDEBUG
            );
            pTreeNodeChildrenAvailableStorageSpaceCur = pTreeNodeChildrenAvailableStorageSpaceNext;
            pBestTreeNodeToSplit->push(pRightChild);
         } else {
            // we aren't going to split this TreeNode because we can't.  We need to set the splitGain value here because otherwise it is filled with garbage that could be NaN (meaning the node was a branch)
            // we can't call INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED before calling SplitTreeNode because INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED sets m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain and the m_UNION.m_beforeExaminationForPossibleSplitting values are needed if we had decided to call ExamineNodeForSplittingAndDetermineBestPossibleSplit
            pRightChild->INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED();
         }
         ++cSplits;
      } while(cSplits < cTreeSplitsMax && UNLIKELY(!pBestTreeNodeToSplit->empty()));
      // we DON'T need to call SetLeafAfterDone() on any items that remain in the pBestTreeNodeToSplit queue because everything in that queue has set a non-NaN nodeSplittingScore value

      // we might as well dump this value out to our pointer, even if later fail the function below.  If the function is failed, we make no guarantees about what we did with the value pointed to at *pTotalGain
      *pTotalGain = totalGain;
      EBM_ASSERT(static_cast<size_t>(reinterpret_cast<char *>(pTreeNodeChildrenAvailableStorageSpaceCur) - reinterpret_cast<char *>(pRootTreeNode)) <= cBytesBuffer2);
   } catch(...) {
      LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree exception");
      return true;
   }

   if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, cSplits))) {
      LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, cSplits)");
      return true;
   }
   if(IsMultiplyError(cVectorLength, cSplits + 1)) {
      LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree IsMultiplyError(cVectorLength, cSplits + 1)");
      return true;
   }
   if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * (cSplits + 1)))) {
      LOG_0(TraceLevelWarning, "WARNING GrowDecisionTree pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * (cSplits + 1)");
      return true;
   }
   ActiveDataType * pDivisions = pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0);
   FractionalDataType * pValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();

   LOG_0(TraceLevelVerbose, "Entered Flatten");
   pRootTreeNode->Flatten(&pDivisions, &pValues, cVectorLength);
   LOG_0(TraceLevelVerbose, "Exited Flatten");

   EBM_ASSERT(pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0) <= pDivisions);
   EBM_ASSERT(static_cast<size_t>(pDivisions - pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)) == cSplits);
   EBM_ASSERT(pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer() < pValues);
   EBM_ASSERT(static_cast<size_t>(pValues - pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()) == cVectorLength * (cSplits + 1));

   LOG_0(TraceLevelVerbose, "Exited GrowDecisionTree via normal exit");
   return false;
}

// TODO : make variable ordering consistent with BinDataSet call below (put the feature first since that's a definition that happens before the training data set)
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
bool BoostZeroDimensional(CachedBoostingThreadResources<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pCachedThreadResources, const SamplingMethod * const pTrainingSet, SegmentedTensor<ActiveDataType, FractionalDataType> * const pSmallChangeToModelOverwriteSingleSamplingSet, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses) {
   LOG_0(TraceLevelVerbose, "Entered BoostZeroDimensional");

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses);
   if(GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)");
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);
   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pHistogramBucket = static_cast<HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> *>(pCachedThreadResources->GetThreadByteBuffer1(cBytesPerHistogramBucket));
   if(UNLIKELY(nullptr == pHistogramBucket)) {
      LOG_0(TraceLevelWarning, "WARNING nullptr == pHistogramBucket");
      return true;
   }
   memset(pHistogramBucket, 0, cBytesPerHistogramBucket);

   BinDataSetTrainingZeroDimensions<compilerLearningTypeOrCountTargetClasses>(pHistogramBucket, pTrainingSet, runtimeLearningTypeOrCountTargetClasses);

   const HistogramBucketVectorEntry<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aSumHistogramBucketVectorEntry = ARRAY_TO_POINTER(pHistogramBucket->m_aHistogramBucketVectorEntry);
   if(IsClassification(compilerLearningTypeOrCountTargetClasses)) {
      FractionalDataType * aValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         FractionalDataType smallChangeToModel = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(aSumHistogramBucketVectorEntry[iVector].m_sumResidualError, aSumHistogramBucketVectorEntry[iVector].GetSumDenominator());
         aValues[iVector] = smallChangeToModel;
      }
   } else {
      EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
      FractionalDataType smallChangeToModel = EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(aSumHistogramBucketVectorEntry[0].m_sumResidualError, static_cast<FractionalDataType>(pHistogramBucket->m_cInstancesInBucket));
      FractionalDataType * pValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
      pValues[0] = smallChangeToModel;
   }

   LOG_0(TraceLevelVerbose, "Exited BoostZeroDimensional");
   return false;
}

// TODO : make variable ordering consistent with BinDataSet call below (put the feature first since that's a definition that happens before the training data set)
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
bool BoostSingleDimensional(CachedBoostingThreadResources<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pCachedThreadResources, const SamplingMethod * const pTrainingSet, const FeatureCombinationCore * const pFeatureCombination, const size_t cTreeSplitsMax, const size_t cInstancesRequiredForParentSplitMin, SegmentedTensor<ActiveDataType, FractionalDataType> * const pSmallChangeToModelOverwriteSingleSamplingSet, FractionalDataType * const pTotalGain, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses) {
   LOG_0(TraceLevelVerbose, "Entered BoostSingleDimensional");

   EBM_ASSERT(1 == pFeatureCombination->m_cFeatures);
   size_t cTotalBuckets = ARRAY_TO_POINTER_CONST(pFeatureCombination->m_FeatureCombinationEntry)[0].m_pFeature->m_cBins;

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses);
   if(GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "WARNING GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)");
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);
   if(IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "WARNING IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)");
      return true;
   }
   const size_t cBytesBuffer = cTotalBuckets * cBytesPerHistogramBucket;
   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets = static_cast<HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> *>(pCachedThreadResources->GetThreadByteBuffer1(cBytesBuffer));
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING BoostSingleDimensional nullptr == aHistogramBuckets");
      return true;
   }
   // !!! VERY IMPORTANT: zero our one extra bucket for BuildFastTotals to use for multi-dimensional !!!!
   memset(aHistogramBuckets, 0, cBytesBuffer);

#ifndef NDEBUG
   const unsigned char * const aHistogramBucketsEndDebug = reinterpret_cast<unsigned char *>(aHistogramBuckets) + cBytesBuffer;
#endif // NDEBUG

   BinDataSetTraining<compilerLearningTypeOrCountTargetClasses, 1>(aHistogramBuckets, pFeatureCombination, pTrainingSet, runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
      , aHistogramBucketsEndDebug
#endif // NDEBUG
   );

   HistogramBucketVectorEntry<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aSumHistogramBucketVectorEntry = pCachedThreadResources->m_aSumHistogramBucketVectorEntry;
   memset(aSumHistogramBucketVectorEntry, 0, sizeof(*aSumHistogramBucketVectorEntry) * cVectorLength); // can't overflow, accessing existing memory

   size_t cHistogramBuckets = ARRAY_TO_POINTER_CONST(pFeatureCombination->m_FeatureCombinationEntry)[0].m_pFeature->m_cBins;
   EBM_ASSERT(1 <= cHistogramBuckets); // this function can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on (dimensions with 1 bin don't contribute anything since they always have the same value)
   size_t cInstancesTotal;
   cHistogramBuckets = CompressHistogramBuckets<compilerLearningTypeOrCountTargetClasses>(pTrainingSet, cHistogramBuckets, aHistogramBuckets, &cInstancesTotal, aSumHistogramBucketVectorEntry, runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
      , aHistogramBucketsEndDebug
#endif // NDEBUG
   );

   EBM_ASSERT(1 <= cInstancesTotal);
   EBM_ASSERT(1 <= cHistogramBuckets);

   bool bRet = GrowDecisionTree<compilerLearningTypeOrCountTargetClasses>(pCachedThreadResources, runtimeLearningTypeOrCountTargetClasses, cHistogramBuckets, aHistogramBuckets, cInstancesTotal, aSumHistogramBucketVectorEntry, cTreeSplitsMax, cInstancesRequiredForParentSplitMin, pSmallChangeToModelOverwriteSingleSamplingSet, pTotalGain
#ifndef NDEBUG
      , aHistogramBucketsEndDebug
#endif // NDEBUG
   );

   LOG_0(TraceLevelVerbose, "Exited BoostSingleDimensional");
   return bRet;
}

#endif // DIMENSION_SINGLE_H
