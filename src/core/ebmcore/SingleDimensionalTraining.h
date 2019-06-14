// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TREE_NODE_H
#define TREE_NODE_H

#include <type_traits> // std::is_pod
#include <assert.h>
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // TML_INLINE
#include "SegmentedRegion.h"
#include "EbmStatistics.h"
#include "CachedThreadResources.h"
#include "AttributeInternal.h"
#include "SamplingWithReplacement.h"
#include "BinnedBucket.h"

template<bool bRegression>
class TreeNode;

template<bool bRegression>
constexpr TML_INLINE const size_t GetTreeNodeSizeOverflow(size_t cVectorLength) {
   return IsMultiplyError(sizeof(PredictionStatistics<bRegression>), cVectorLength) ? true : IsAddError(sizeof(TreeNode<bRegression>) - sizeof(PredictionStatistics<bRegression>), sizeof(PredictionStatistics<bRegression>) * cVectorLength) ? true : false;
}
template<bool bRegression>
constexpr TML_INLINE const size_t GetTreeNodeSize(size_t cVectorLength) {
   return sizeof(TreeNode<bRegression>) - sizeof(PredictionStatistics<bRegression>) + sizeof(PredictionStatistics<bRegression>) * cVectorLength;
}
template<bool bRegression>
constexpr TML_INLINE TreeNode<bRegression> * AddBytesTreeNode(TreeNode<bRegression> * pTreeNode, size_t countBytesAdd) {
   return reinterpret_cast<TreeNode<bRegression> *>(reinterpret_cast<char *>(pTreeNode) + countBytesAdd);
}
template<bool bRegression>
constexpr TML_INLINE TreeNode<bRegression> * GetLeftTreeNodeChild(TreeNode<bRegression> * pTreeNodeChildren, size_t countBytesTreeNode) {
   return pTreeNodeChildren;
}
template<bool bRegression>
constexpr TML_INLINE TreeNode<bRegression> * GetRightTreeNodeChild(TreeNode<bRegression> * pTreeNodeChildren, size_t countBytesTreeNode) {
   return AddBytesTreeNode<bRegression>(pTreeNodeChildren, countBytesTreeNode);
}

template<bool bRegression>
class TreeNodeData;

template<>
class TreeNodeData<false> {
   // classification version of the TreeNodeData

public:

   struct BeforeSplit {
      const BinnedBucket<false> * pBinnedBucketEntryFirst;
      const BinnedBucket<false> * pBinnedBucketEntryLast;
      size_t cCases;
   };

   struct AfterSplit {
      TreeNode<false> * pTreeNodeChildren;
      FractionalDataType nodeSplittingScore; // put this at the top so that our priority queue can access it directly without adding anything to the pointer (this is slightly more efficient on intel systems at least)
      ActiveDataType divisionValue;
   };

   union TreeNodeDataUnion {
      // we can save precious L1 cache space by keeping only what we need
      BeforeSplit beforeSplit;
      AfterSplit afterSplit;

      static_assert(std::is_pod<BeforeSplit>::value, "BeforeSplit must be POD (Plain Old Data) if we are going to use it in a union!");
      static_assert(std::is_pod<AfterSplit>::value, "AfterSplit must be POD (Plain Old Data) if we are going to use it in a union!");
   };

   TreeNodeDataUnion m_UNION;
   PredictionStatistics<false> aPredictionStatistics[1];

   TML_INLINE size_t GetCases() const {
      return m_UNION.beforeSplit.cCases;
   }
   TML_INLINE void SetCases(size_t cCases) {
      m_UNION.beforeSplit.cCases = cCases;
   }
};

template<>
class TreeNodeData<true> {
   // regression version of the TreeNodeData
public:

   struct BeforeSplit {
      const BinnedBucket<true> * pBinnedBucketEntryFirst;
      const BinnedBucket<true> * pBinnedBucketEntryLast;
   };

   struct AfterSplit {
      TreeNode<true> * pTreeNodeChildren;
      FractionalDataType nodeSplittingScore; // put this at the top so that our priority queue can access it directly without adding anything to the pointer (this is slightly more efficient on intel systems at least)
      ActiveDataType divisionValue;
   };

   union TreeNodeDataUnion {
      // we can save precious L1 cache space by keeping only what we need
      BeforeSplit beforeSplit;
      AfterSplit afterSplit;

      static_assert(std::is_pod<BeforeSplit>::value, "BeforeSplit must be POD (Plain Old Data) if we are going to use it in a union!");
      static_assert(std::is_pod<AfterSplit>::value, "AfterSplit must be POD (Plain Old Data) if we are going to use it in a union!");
   };

   TreeNodeDataUnion m_UNION;

   size_t m_cCases;
   PredictionStatistics<true> aPredictionStatistics[1];

   TML_INLINE size_t GetCases() const {
      return m_cCases;
   }
   TML_INLINE void SetCases(size_t cCases) {
      m_cCases = cCases;
   }
};

template<bool bRegression>
class TreeNode final : public TreeNodeData<bRegression> {
public:

   TML_INLINE bool IsSplittable(size_t cCasesRequiredForSplitParentMin) const {
      return this->m_UNION.beforeSplit.pBinnedBucketEntryLast != this->m_UNION.beforeSplit.pBinnedBucketEntryFirst && cCasesRequiredForSplitParentMin <= this->GetCases();
   }

   TML_INLINE void SetTrunkAfterDone() {
      constexpr static FractionalDataType nan = std::numeric_limits<FractionalDataType>::quiet_NaN();
      this->m_UNION.afterSplit.nodeSplittingScore = nan;
   }

   TML_INLINE void SetLeafAfterDone() {
      this->m_UNION.afterSplit.nodeSplittingScore = 0;
   }

   TML_INLINE bool IsTrunkAfterDone() const {
      return std::isnan(this->m_UNION.afterSplit.nodeSplittingScore);
   }

   template<ptrdiff_t countCompilerClassificationTargetStates>
   void SplitTreeNode(CachedTrainingThreadResources<bRegression> * const pCachedThreadResources, TreeNode<bRegression> * const pTreeNodeChildrenAvailableStorageSpaceCur, const size_t cTargetStates
#ifndef NDEBUG
      , const unsigned char * const aBinnedBucketsEndDebug
#endif // NDEBUG
   ) {
      LOG(TraceLevelVerbose, "Entered SplitTreeNode this=%p, pTreeNodeChildrenAvailableStorageSpaceCur=%p", static_cast<void *>(this), static_cast<void *>(pTreeNodeChildrenAvailableStorageSpaceCur));

      static_assert(IsRegression(countCompilerClassificationTargetStates) == bRegression, "regression types must match");

      const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
      assert(!GetTreeNodeSizeOverflow<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerTreeNode = GetTreeNodeSize<bRegression>(cVectorLength);
      assert(!GetBinnedBucketSizeOverflow<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerBinnedBucket = GetBinnedBucketSize<bRegression>(cVectorLength);

      const BinnedBucket<bRegression> * pBinnedBucketEntryCur = this->m_UNION.beforeSplit.pBinnedBucketEntryFirst;
      const BinnedBucket<bRegression> * const pBinnedBucketEntryLast = this->m_UNION.beforeSplit.pBinnedBucketEntryLast;

      TreeNode<bRegression> * const pLeftChild1 = GetLeftTreeNodeChild<bRegression>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);
      pLeftChild1->m_UNION.beforeSplit.pBinnedBucketEntryFirst = pBinnedBucketEntryCur;
      TreeNode<bRegression> * const pRightChild1 = GetRightTreeNodeChild<bRegression>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);
      pRightChild1->m_UNION.beforeSplit.pBinnedBucketEntryLast = pBinnedBucketEntryLast;

      size_t cCases1 = pBinnedBucketEntryCur->cCasesInBucket;
      size_t cCases2 = this->GetCases() - cCases1;

      PredictionStatistics<bRegression> * const aSumPredictionStatistics1 = pCachedThreadResources->m_aSumPredictionStatistics1;
      FractionalDataType * const aSumResidualErrors2 = pCachedThreadResources->m_aSumResidualErrors2;
      PredictionStatistics<bRegression> * const aSumPredictionStatisticsBest = pCachedThreadResources->m_aSumPredictionStatisticsBest;
      FractionalDataType BEST_nodeSplittingScoreChildren = 0;
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         const FractionalDataType sumResidualError1 = pBinnedBucketEntryCur->aPredictionStatistics[iVector].sumResidualError;
         const FractionalDataType sumResidualError2 = this->aPredictionStatistics[iVector].sumResidualError - sumResidualError1;

         BEST_nodeSplittingScoreChildren += ComputeNodeSplittingScore(sumResidualError1, cCases1) + ComputeNodeSplittingScore(sumResidualError2, cCases2);

         aSumPredictionStatistics1[iVector].sumResidualError = sumResidualError1;
         aSumPredictionStatisticsBest[iVector].sumResidualError = sumResidualError1;
         aSumResidualErrors2[iVector] = sumResidualError2;
         if(!bRegression) {
            FractionalDataType sumDenominator1 = pBinnedBucketEntryCur->aPredictionStatistics[iVector].GetSumDenominator();
            aSumPredictionStatistics1[iVector].SetSumDenominator(sumDenominator1);
            aSumPredictionStatisticsBest[iVector].SetSumDenominator(sumDenominator1);
         }
      }

      assert(0 <= BEST_nodeSplittingScoreChildren);
      const BinnedBucket<bRegression> * BEST_pBinnedBucketEntry = pBinnedBucketEntryCur;
      size_t BEST_cCases1 = cCases1;
      for(pBinnedBucketEntryCur = GetBinnedBucketByIndex<bRegression>(cBytesPerBinnedBucket, pBinnedBucketEntryCur, 1); pBinnedBucketEntryLast != pBinnedBucketEntryCur; pBinnedBucketEntryCur = GetBinnedBucketByIndex<bRegression>(cBytesPerBinnedBucket, pBinnedBucketEntryCur, 1)) {
         ASSERT_BINNED_BUCKET_OK(cBytesPerBinnedBucket, pBinnedBucketEntryCur, aBinnedBucketsEndDebug);

         const size_t CHANGE_cCases = pBinnedBucketEntryCur->cCasesInBucket;
         cCases1 += CHANGE_cCases;
         cCases2 -= CHANGE_cCases;

         FractionalDataType nodeSplittingScoreChildren = 0;
         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            if(!bRegression) {
               aSumPredictionStatistics1[iVector].SetSumDenominator(aSumPredictionStatistics1[iVector].GetSumDenominator() + pBinnedBucketEntryCur->aPredictionStatistics[iVector].GetSumDenominator());
            }

            const FractionalDataType CHANGE_sumResidualError = pBinnedBucketEntryCur->aPredictionStatistics[iVector].sumResidualError;
            const FractionalDataType sumResidualError1 = aSumPredictionStatistics1[iVector].sumResidualError + CHANGE_sumResidualError;
            const FractionalDataType sumResidualError2 = aSumResidualErrors2[iVector] - CHANGE_sumResidualError;

            aSumPredictionStatistics1[iVector].sumResidualError = sumResidualError1;
            aSumResidualErrors2[iVector] = sumResidualError2;

            // TODO : we can make this faster by doing the division in ComputeNodeSplittingScore after we add all the numerators
            const FractionalDataType nodeSplittingScoreChildrenOneVector = ComputeNodeSplittingScore(sumResidualError1, cCases1) + ComputeNodeSplittingScore(sumResidualError2, cCases2);
            assert(0 <= nodeSplittingScoreChildren);
            nodeSplittingScoreChildren += nodeSplittingScoreChildrenOneVector;
         }
         assert(0 <= nodeSplittingScoreChildren);

         if(UNLIKELY(BEST_nodeSplittingScoreChildren < nodeSplittingScoreChildren)) {
            // TODO : randomly choose a node if BEST_entropyTotalChildren == entropyTotalChildren, but if there are 3 choice make sure that each has a 1/3 probability of being selected (same as interview question to select a random line from a file)
            BEST_nodeSplittingScoreChildren = nodeSplittingScoreChildren;
            BEST_pBinnedBucketEntry = pBinnedBucketEntryCur;
            BEST_cCases1 = cCases1;
            memcpy(aSumPredictionStatisticsBest, aSumPredictionStatistics1, sizeof(*aSumPredictionStatisticsBest) * cVectorLength);
         }
      }

      TreeNode<bRegression> * const pLeftChild = GetLeftTreeNodeChild<bRegression>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);
      TreeNode<bRegression> * const pRightChild = GetRightTreeNodeChild<bRegression>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);

      pLeftChild->m_UNION.beforeSplit.pBinnedBucketEntryLast = BEST_pBinnedBucketEntry;
      pLeftChild->SetCases(BEST_cCases1);

      const BinnedBucket<bRegression> * const BEST_pBinnedBucketEntryNext = GetBinnedBucketByIndex<bRegression>(cBytesPerBinnedBucket, BEST_pBinnedBucketEntry, 1);
      ASSERT_BINNED_BUCKET_OK(cBytesPerBinnedBucket, BEST_pBinnedBucketEntryNext, aBinnedBucketsEndDebug);

      pRightChild->m_UNION.beforeSplit.pBinnedBucketEntryFirst = BEST_pBinnedBucketEntryNext;
      size_t cCasesParent = this->GetCases();
      pRightChild->SetCases(cCasesParent - BEST_cCases1);

      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         pLeftChild->aPredictionStatistics[iVector].sumResidualError = aSumPredictionStatisticsBest[iVector].sumResidualError;
         if(!bRegression) {
            pLeftChild->aPredictionStatistics[iVector].SetSumDenominator(aSumPredictionStatisticsBest[iVector].GetSumDenominator());
         }

         pRightChild->aPredictionStatistics[iVector].sumResidualError = this->aPredictionStatistics[iVector].sumResidualError - aSumPredictionStatisticsBest[iVector].sumResidualError;
         if(!bRegression) {
            pRightChild->aPredictionStatistics[iVector].SetSumDenominator(this->aPredictionStatistics[iVector].GetSumDenominator() - aSumPredictionStatisticsBest[iVector].GetSumDenominator());
         }
      }

      FractionalDataType nodeSplittingScoreParent = 0;
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         const FractionalDataType sumResidualErrorParent = this->aPredictionStatistics[iVector].sumResidualError;
         nodeSplittingScoreParent += ComputeNodeSplittingScore(sumResidualErrorParent, cCasesParent);
      }

      // IMPORTANT!! : we need to finish all our calls that use this->m_UNION.beforeSplit BEFORE setting anything in m_UNION.afterSplit as we do below this comment!  The call above to this->GetCases() needs to be done above these lines because it uses m_UNION.beforeSplit for classification!
      this->m_UNION.afterSplit.pTreeNodeChildren = pTreeNodeChildrenAvailableStorageSpaceCur;
      this->m_UNION.afterSplit.nodeSplittingScore = nodeSplittingScoreParent - BEST_nodeSplittingScoreChildren;
      this->m_UNION.afterSplit.divisionValue = (BEST_pBinnedBucketEntry->bucketValue + BEST_pBinnedBucketEntryNext->bucketValue) / 2;

      assert(this->m_UNION.afterSplit.nodeSplittingScore <= 0.0000000001); // within a set, no split should make our model worse.  It might in our validation set, but not within this set

      LOG(TraceLevelVerbose, "Exited SplitTreeNode divisionValue=%zu, nodeSplittingScore=%" FractionalDataTypePrintf, static_cast<size_t>(this->m_UNION.afterSplit.divisionValue), this->m_UNION.afterSplit.nodeSplittingScore);
   }

   // TODO: in theory, a malicious caller could overflow our stack if they pass us data that will grow a sufficiently deep tree.  Consider changing this recursive function to handle that
   // TODO: specialize this function for cases where we have hard coded vector lengths so that we don't have to pass in the cVectorLength parameter
   void Flatten(ActiveDataType ** const ppDivisions, FractionalDataType ** const ppValues, const size_t cVectorLength) const {
      if(UNPREDICTABLE(IsTrunkAfterDone())) {
         assert(!GetTreeNodeSizeOverflow<bRegression>(cVectorLength)); // we're accessing allocated memory
         const size_t cBytesPerTreeNode = GetTreeNodeSize<bRegression>(cVectorLength);
         const TreeNode<bRegression> * const pLeftChild = GetLeftTreeNodeChild<bRegression>(this->m_UNION.afterSplit.pTreeNodeChildren, cBytesPerTreeNode);
         pLeftChild->Flatten(ppDivisions, ppValues, cVectorLength);
         **ppDivisions = this->m_UNION.afterSplit.divisionValue;
         ++(*ppDivisions);
         const TreeNode<bRegression> * const pRightChild = GetRightTreeNodeChild<bRegression>(this->m_UNION.afterSplit.pTreeNodeChildren, cBytesPerTreeNode);
         pRightChild->Flatten(ppDivisions, ppValues, cVectorLength);
      } else {
         FractionalDataType * pValuesCur = *ppValues;
         FractionalDataType * const pValuesNext = pValuesCur + cVectorLength;
         *ppValues = pValuesNext;

         const PredictionStatistics<bRegression> * pPredictionStatistics = &this->aPredictionStatistics[0];
         do {
            FractionalDataType smallChangeToModel;
            if(bRegression) {
               smallChangeToModel = ComputeSmallChangeInRegressionPredictionForOneSegment(pPredictionStatistics->sumResidualError, this->GetCases());
            } else {
               smallChangeToModel = ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pPredictionStatistics->sumResidualError, pPredictionStatistics->GetSumDenominator());
            }
            *pValuesCur = smallChangeToModel;

            ++pPredictionStatistics;
            ++pValuesCur;
         } while(pValuesNext != pValuesCur);
      }
   }

   static_assert(std::is_pod<ActiveDataType>::value, "We want to keep our TreeNode compact and without a virtual pointer table for fitting in L1 cache as much as possible");
};
static_assert(std::is_pod<TreeNode<false>>::value, "We want to keep our TreeNode compact and without a virtual pointer table for fitting in L1 cache as much as possible");
static_assert(std::is_pod<TreeNode<true>>::value, "We want to keep our TreeNode compact and without a virtual pointer table for fitting in L1 cache as much as possible");

template<ptrdiff_t countCompilerClassificationTargetStates>
bool GrowDecisionTree(CachedTrainingThreadResources<IsRegression(countCompilerClassificationTargetStates)> * const pCachedThreadResources, const size_t cTargetStates, const size_t cBinnedBuckets, const BinnedBucket<IsRegression(countCompilerClassificationTargetStates)> * const aBinnedBucket, const size_t cCasesTotal, const PredictionStatistics<IsRegression(countCompilerClassificationTargetStates)> * const aSumPredictionStatistics, const size_t cTreeSplitsMax, const size_t cCasesRequiredForSplitParentMin, SegmentedRegionCore<ActiveDataType, FractionalDataType> * const pSmallChangeToModelOverwriteSingleSamplingSet
#ifndef NDEBUG
   , const unsigned char * const aBinnedBucketsEndDebug
#endif // NDEBUG
) {
   LOG(TraceLevelVerbose, "Entered GrowDecisionTree");

   const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);

   assert(0 != cBinnedBuckets);
   if(UNLIKELY(cCasesTotal < cCasesRequiredForSplitParentMin || 1 == cBinnedBuckets || 0 == cTreeSplitsMax)) {
      if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 0))) {
         LOG(TraceLevelWarning, "WARNING GrowDecisionTree pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 0)");
         return true;
      }

      // we don't need to call EnsureValueCapacity because by default we start with a value capacity of 2 * cVectorLength

      if(IsRegression(countCompilerClassificationTargetStates)) {
         FractionalDataType smallChangeToModel = ComputeSmallChangeInRegressionPredictionForOneSegment(aSumPredictionStatistics[0].sumResidualError, cCasesTotal);
         FractionalDataType * pValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
         pValues[0] = smallChangeToModel;
      } else {
         assert(IsClassification(countCompilerClassificationTargetStates));
         FractionalDataType * aValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            FractionalDataType smallChangeToModel = ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(aSumPredictionStatistics[iVector].sumResidualError, aSumPredictionStatistics[iVector].GetSumDenominator());
            aValues[iVector] = smallChangeToModel;
         }
      }

      LOG(TraceLevelVerbose, "Exited GrowDecisionTree via not enough data to split");
      return false;
   }

   if(GetTreeNodeSizeOverflow<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength)) {
      LOG(TraceLevelWarning, "WARNING GrowDecisionTree GetTreeNodeSizeOverflow<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength)");
      return true; // we haven't accessed this TreeNode memory yet, so we don't know if it overflows yet
   }
   const size_t cBytesPerTreeNode = GetTreeNodeSize<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength);
   assert(!GetBinnedBucketSizeOverflow<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerBinnedBucket = GetBinnedBucketSize<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength);

retry_with_bigger_tree_node_children_array:
   size_t cBytesBuffer2 = pCachedThreadResources->GetThreadByteBuffer2Size();
   const size_t cBytesInitialNeededAllocation = 3 * cBytesPerTreeNode; // we need 1 TreeNode for the root, 1 for the left child of the root and 1 for the right child of the root
   if(cBytesBuffer2 < cBytesInitialNeededAllocation) {
      // TODO : we can eliminate this check as long as we ensure that the ThreadByteBuffer2 is always initialized to be equal to the size of three TreeNodes (left and right) == GET_SIZEOF_ONE_TREE_NODE_CHILDREN(cBytesPerTreeNode)
      if(pCachedThreadResources->GrowThreadByteBuffer2(cBytesInitialNeededAllocation)) {
         LOG(TraceLevelWarning, "WARNING GrowDecisionTree pCachedThreadResources->GrowThreadByteBuffer2(cBytesInitialNeededAllocation)");
         return true;
      }
      cBytesBuffer2 = pCachedThreadResources->GetThreadByteBuffer2Size();
      assert(cBytesInitialNeededAllocation <= cBytesBuffer2);
   }
   TreeNode<IsRegression(countCompilerClassificationTargetStates)> * pRootTreeNode = static_cast<TreeNode<IsRegression(countCompilerClassificationTargetStates)> *>(pCachedThreadResources->GetThreadByteBuffer2());

   pRootTreeNode->m_UNION.beforeSplit.pBinnedBucketEntryFirst = aBinnedBucket;
   pRootTreeNode->m_UNION.beforeSplit.pBinnedBucketEntryLast = GetBinnedBucketByIndex<IsRegression(countCompilerClassificationTargetStates)>(cBytesPerBinnedBucket, aBinnedBucket, cBinnedBuckets - 1);
   ASSERT_BINNED_BUCKET_OK(cBytesPerBinnedBucket, pRootTreeNode->m_UNION.beforeSplit.pBinnedBucketEntryLast, aBinnedBucketsEndDebug);
   pRootTreeNode->SetCases(cCasesTotal);

   memcpy(&pRootTreeNode->aPredictionStatistics[0], aSumPredictionStatistics, cVectorLength * sizeof(*aSumPredictionStatistics)); // copying existing mem

   pRootTreeNode->template SplitTreeNode<countCompilerClassificationTargetStates>(pCachedThreadResources, AddBytesTreeNode<IsRegression(countCompilerClassificationTargetStates)>(pRootTreeNode, cBytesPerTreeNode), cTargetStates
#ifndef NDEBUG
      , aBinnedBucketsEndDebug
#endif // NDEBUG
   );

   if(PREDICTABLE(1 == cTreeSplitsMax)) {
      if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1))) {
         LOG(TraceLevelWarning, "WARNING GrowDecisionTree pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1)");
         return true;
      }

      ActiveDataType * pDivisions = pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0);
      pDivisions[0] = pRootTreeNode->m_UNION.afterSplit.divisionValue;

      // we don't need to call EnsureValueCapacity because by default we start with a value capacity of 2 * cVectorLength

      // TODO : we don't need to get the right and left pointer from the root.. we know where they will be
      const TreeNode<IsRegression(countCompilerClassificationTargetStates)> * const pLeftChild = GetLeftTreeNodeChild<IsRegression(countCompilerClassificationTargetStates)>(pRootTreeNode->m_UNION.afterSplit.pTreeNodeChildren, cBytesPerTreeNode);
      const TreeNode<IsRegression(countCompilerClassificationTargetStates)> * const pRightChild = GetRightTreeNodeChild<IsRegression(countCompilerClassificationTargetStates)>(pRootTreeNode->m_UNION.afterSplit.pTreeNodeChildren, cBytesPerTreeNode);

      FractionalDataType * const aValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
      if(IsRegression(countCompilerClassificationTargetStates)) {
         aValues[0] = ComputeSmallChangeInRegressionPredictionForOneSegment(pLeftChild->aPredictionStatistics[0].sumResidualError, pLeftChild->GetCases());
         aValues[1] = ComputeSmallChangeInRegressionPredictionForOneSegment(pRightChild->aPredictionStatistics[0].sumResidualError, pRightChild->GetCases());
      } else {
         assert(IsClassification(countCompilerClassificationTargetStates));
         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            aValues[iVector] = ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pLeftChild->aPredictionStatistics[iVector].sumResidualError, pLeftChild->aPredictionStatistics[iVector].GetSumDenominator());
            aValues[cVectorLength + iVector] = ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pRightChild->aPredictionStatistics[iVector].sumResidualError, pRightChild->aPredictionStatistics[iVector].GetSumDenominator());
         }
      }

      LOG(TraceLevelVerbose, "Exited GrowDecisionTree via one tree split");
      return false;
   }

   // TODO: there are three types of queues that we should try out -> dyamically picking a stragety is a single predictable if statement, so shouldn't cause a lot of overhead
   //       1) When the data is the smallest(1-5 items), just iterate over all items in our TreeNode buffer looking for the best Node.  Zero the value on any nodes that have been removed from the queue.  For 1 or 2 instructions in the loop WITHOUT a branch we can probably save the pointer to the first TreeNode with data so that we can start from there next time we loop
   //       2) When the data is a tiny bit bigger and there are holes in our array of TreeNodes, we can maintain a pointer and value in a separate list and zip through the values and then go to the pointer to the best node.  Since the list is unordered, when we find a TreeNode to remove, we just move the last one into the hole
   //       3) The full fleged priority queue below
   size_t cSplits;
   try {
      std::priority_queue<TreeNode<IsRegression(countCompilerClassificationTargetStates)> *, std::vector<TreeNode<IsRegression(countCompilerClassificationTargetStates)> *>, CompareTreeNodeSplittingScore<IsRegression(countCompilerClassificationTargetStates)>> * pBestTreeNodeToSplit = &pCachedThreadResources->m_bestTreeNodeToSplit;

      // it is ridiculous that we need to do this in order to clear the tree (there is no "clear" function), but inside this queue is a chunk of memory, and we want to ensure that the chunk of memory stays in L1 cache, so we pop all the previous garbage off instead of allocating a new one!
      while(!pBestTreeNodeToSplit->empty()) {
         pBestTreeNodeToSplit->pop();
      }

      cSplits = 0;
      TreeNode<IsRegression(countCompilerClassificationTargetStates)> * pParentTreeNode = pRootTreeNode;

      // we skip 3 tree nodes.  The root, the left child of the root, and the right child of the root
      TreeNode<IsRegression(countCompilerClassificationTargetStates)> * pTreeNodeChildrenAvailableStorageSpaceCur = AddBytesTreeNode<IsRegression(countCompilerClassificationTargetStates)>(pRootTreeNode, cBytesInitialNeededAllocation);

      goto skip_first_push_pop;

      do {
         // there is no way to get the top and pop at the same time.. would be good to get a better queue, but our code isn't bottlenecked by it
         pParentTreeNode = pBestTreeNodeToSplit->top();
         pBestTreeNodeToSplit->pop();

      skip_first_push_pop:

         pParentTreeNode->SetTrunkAfterDone();

         TreeNode<IsRegression(countCompilerClassificationTargetStates)> * const pLeftChild = GetLeftTreeNodeChild<IsRegression(countCompilerClassificationTargetStates)>(pParentTreeNode->m_UNION.afterSplit.pTreeNodeChildren, cBytesPerTreeNode);
         if(pLeftChild->IsSplittable(cCasesRequiredForSplitParentMin)) {
            TreeNode<IsRegression(countCompilerClassificationTargetStates)> * pTreeNodeChildrenAvailableStorageSpaceNext = AddBytesTreeNode<IsRegression(countCompilerClassificationTargetStates)>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode << 1);
            if(cBytesBuffer2 < static_cast<size_t>(reinterpret_cast<char *>(pTreeNodeChildrenAvailableStorageSpaceNext) - reinterpret_cast<char *>(pRootTreeNode))) {
               if(pCachedThreadResources->GrowThreadByteBuffer2(cBytesPerTreeNode)) {
                  LOG(TraceLevelWarning, "WARNING GrowDecisionTree pCachedThreadResources->GrowThreadByteBuffer2(cBytesPerTreeNode)");
                  return true;
               }
               goto retry_with_bigger_tree_node_children_array;
            }
            // the act of splitting it implicitly sets SetLeafAfterDone because splitting sets nodeSplittingScore to a non-NaN value
            pLeftChild->template SplitTreeNode<countCompilerClassificationTargetStates>(pCachedThreadResources, pTreeNodeChildrenAvailableStorageSpaceCur, cTargetStates
#ifndef NDEBUG
               , aBinnedBucketsEndDebug
#endif // NDEBUG
            );
            pTreeNodeChildrenAvailableStorageSpaceCur = pTreeNodeChildrenAvailableStorageSpaceNext;
            pBestTreeNodeToSplit->push(pLeftChild);
         } else {
            // we aren't going to split this TreeNode because we can't.  We need to set the nodeSplittingScore value here because otherwise it is filled with garbage that could be NaN (meaning the node was a branch)
            // we can't call SetLeafAfterDone before calling SplitTreeNode because SetLeafAfterDone sets m_UNION.afterSplit.nodeSplittingScore and the m_UNION.beforeSplit values are needed if we had decided to call SplitTreeNode
            pLeftChild->SetLeafAfterDone();
         }

         TreeNode<IsRegression(countCompilerClassificationTargetStates)> * const pRightChild = GetRightTreeNodeChild<IsRegression(countCompilerClassificationTargetStates)>(pParentTreeNode->m_UNION.afterSplit.pTreeNodeChildren, cBytesPerTreeNode);
         if(pRightChild->IsSplittable(cCasesRequiredForSplitParentMin)) {
            TreeNode<IsRegression(countCompilerClassificationTargetStates)> * pTreeNodeChildrenAvailableStorageSpaceNext = AddBytesTreeNode<IsRegression(countCompilerClassificationTargetStates)>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode << 1);
            if(cBytesBuffer2 < static_cast<size_t>(reinterpret_cast<char *>(pTreeNodeChildrenAvailableStorageSpaceNext) - reinterpret_cast<char *>(pRootTreeNode))) {
               if(pCachedThreadResources->GrowThreadByteBuffer2(cBytesPerTreeNode)) {
                  LOG(TraceLevelWarning, "WARNING GrowDecisionTree pCachedThreadResources->GrowThreadByteBuffer2(cBytesPerTreeNode)");
                  return true;
               }
               goto retry_with_bigger_tree_node_children_array;
            }
            // the act of splitting it implicitly sets SetLeafAfterDone because splitting sets nodeSplittingScore to a non-NaN value
            pRightChild->template SplitTreeNode<countCompilerClassificationTargetStates>(pCachedThreadResources, pTreeNodeChildrenAvailableStorageSpaceCur, cTargetStates
#ifndef NDEBUG
               , aBinnedBucketsEndDebug
#endif // NDEBUG
            );
            pTreeNodeChildrenAvailableStorageSpaceCur = pTreeNodeChildrenAvailableStorageSpaceNext;
            pBestTreeNodeToSplit->push(pRightChild);
         } else {
            // we aren't going to split this TreeNode because we can't.  We need to set the nodeSplittingScore value here because otherwise it is filled with garbage that could be NaN (meaning the node was a branch)
            // we can't call SetLeafAfterDone before calling SplitTreeNode because SetLeafAfterDone sets m_UNION.afterSplit.nodeSplittingScore and the m_UNION.beforeSplit values are needed if we had decided to call SplitTreeNode
            pRightChild->SetLeafAfterDone();
         }
         ++cSplits;
      } while(cSplits < cTreeSplitsMax && UNLIKELY(!pBestTreeNodeToSplit->empty()));
      // we DON'T need to call SetLeafAfterDone() on any items that remain in the pBestTreeNodeToSplit queue because everything in that queue has set a non-NaN nodeSplittingScore value

      assert(static_cast<size_t>(reinterpret_cast<char *>(pTreeNodeChildrenAvailableStorageSpaceCur) - reinterpret_cast<char *>(pRootTreeNode)) <= cBytesBuffer2);
   } catch(...) {
      LOG(TraceLevelWarning, "WARNING GrowDecisionTree exception");
      return true;
   }

   if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, cSplits))) {
      LOG(TraceLevelWarning, "WARNING GrowDecisionTree pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, cSplits)");
      return true;
   }
   if(IsMultiplyError(cVectorLength, cSplits + 1)) {
      LOG(TraceLevelWarning, "WARNING GrowDecisionTree IsMultiplyError(cVectorLength, cSplits + 1)");
      return true;
   }
   if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * (cSplits + 1)))) {
      LOG(TraceLevelWarning, "WARNING GrowDecisionTree pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * (cSplits + 1)");
      return true;
   }
   ActiveDataType * pDivisions = pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0);
   FractionalDataType * pValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
   pRootTreeNode->Flatten(&pDivisions, &pValues, cVectorLength);
   assert(pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0) <= pDivisions);
   assert(static_cast<size_t>(pDivisions - pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)) == cSplits);
   assert(pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer() < pValues);
   assert(static_cast<size_t>(pValues - pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()) == cVectorLength * (cSplits + 1));

   LOG(TraceLevelVerbose, "Exited GrowDecisionTree via normal exit");
   return false;
}

// TODO : make variable ordering consistent with BinDataSet call below (put the attribute first since that's a definition that happens before the training data set)
template<ptrdiff_t countCompilerClassificationTargetStates>
bool TrainSingleDimensional(CachedTrainingThreadResources<IsRegression(countCompilerClassificationTargetStates)> * const pCachedThreadResources, const SamplingMethod * const pTrainingSet, const AttributeCombinationCore * const pAttributeCombination, const size_t cTreeSplitsMax, const size_t cCasesRequiredForSplitParentMin, SegmentedRegionCore<ActiveDataType, FractionalDataType> * const pSmallChangeToModelOverwriteSingleSamplingSet, const size_t cTargetStates) {
   LOG(TraceLevelVerbose, "Entered TrainSingleDimensional");

   size_t cTotalBuckets = 1;
   for(size_t iDimension = 0; iDimension < pAttributeCombination->m_cAttributes; ++iDimension) {
      const size_t cStates = pAttributeCombination->m_AttributeCombinationEntry[iDimension].m_pAttribute->m_cStates;
      assert(!IsMultiplyError(cTotalBuckets, cStates)); // we check for simple multiplication overflow from m_cStates in TmlTrainingState->Initialize when we unpack attributeCombinationIndexes
      cTotalBuckets *= cStates;
   }

   const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
   if(GetBinnedBucketSizeOverflow<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength)) {
      // TODO : move this to initialization where we execute it only once (it needs to be in the attribute combination loop though)
      LOG(TraceLevelWarning, "WARNING TODO fill this in");
      return true;
   }
   const size_t cBytesPerBinnedBucket = GetBinnedBucketSize<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength);
   if(IsMultiplyError(cTotalBuckets, cBytesPerBinnedBucket)) {
      // TODO : move this to initialization where we execute it only once (it needs to be in the attribute combination loop though)
      LOG(TraceLevelWarning, "WARNING TODO fill this in");
      return true;
   }
   const size_t cBytesBuffer = cTotalBuckets * cBytesPerBinnedBucket;
   BinnedBucket<IsRegression(countCompilerClassificationTargetStates)> * const aBinnedBuckets = static_cast<BinnedBucket<IsRegression(countCompilerClassificationTargetStates)> *>(pCachedThreadResources->GetThreadByteBuffer1(cBytesBuffer));
   if(UNLIKELY(nullptr == aBinnedBuckets)) {
      LOG(TraceLevelWarning, "WARNING TrainSingleDimensional nullptr == aBinnedBuckets");
      return true;
   }
   // !!! VERY IMPORTANT: zero our one extra bucket for BuildFastTotals to use for multi-dimensional !!!!
   memset(aBinnedBuckets, 0, cBytesBuffer);

#ifndef NDEBUG
   const unsigned char * const aBinnedBucketsEndDebug = reinterpret_cast<unsigned char *>(aBinnedBuckets) + cBytesBuffer;
#endif // NDEBUG

   BinDataSetTraining<countCompilerClassificationTargetStates, 1>(aBinnedBuckets, pAttributeCombination, pTrainingSet, cTargetStates
#ifndef NDEBUG
      , aBinnedBucketsEndDebug
#endif // NDEBUG
   );

   PredictionStatistics<IsRegression(countCompilerClassificationTargetStates)> * const aSumPredictionStatistics = pCachedThreadResources->m_aSumPredictionStatistics;
   memset(aSumPredictionStatistics, 0, sizeof(*aSumPredictionStatistics) * cVectorLength); // can't overflow, accessing existing memory

   size_t cBinnedBuckets = pAttributeCombination->m_AttributeCombinationEntry[0].m_pAttribute->m_cStates;
   size_t cCasesTotal;
   cBinnedBuckets = CompressBinnedBuckets<countCompilerClassificationTargetStates>(pTrainingSet, cBinnedBuckets, aBinnedBuckets, &cCasesTotal, aSumPredictionStatistics, cTargetStates
#ifndef NDEBUG
      , aBinnedBucketsEndDebug
#endif // NDEBUG
   );

   bool bRet = GrowDecisionTree<countCompilerClassificationTargetStates>(pCachedThreadResources, cTargetStates, cBinnedBuckets, aBinnedBuckets, cCasesTotal, aSumPredictionStatistics, cTreeSplitsMax, cCasesRequiredForSplitParentMin, pSmallChangeToModelOverwriteSingleSamplingSet
#ifndef NDEBUG
      , aBinnedBucketsEndDebug
#endif // NDEBUG
   );

   LOG(TraceLevelVerbose, "Exited TrainSingleDimensional");
   return bRet;
}

#endif // TREE_NODE_H
