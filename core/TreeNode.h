// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TREE_NODE_H
#define TREE_NODE_H

#include <type_traits> // std::is_pod
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "EbmStatistics.h"
#include "HistogramBucket.h"

template<bool bClassification>
class TreeNode;

template<bool bClassification>
EBM_INLINE size_t GetTreeNodeSizeOverflow(size_t cVectorLength) {
   return IsMultiplyError(sizeof(HistogramBucketVectorEntry<bClassification>), cVectorLength) ? true : IsAddError(sizeof(TreeNode<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>), sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength) ? true : false;
}
template<bool bClassification>
EBM_INLINE size_t GetTreeNodeSize(size_t cVectorLength) {
   return sizeof(TreeNode<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>) + sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength;
}
template<bool bClassification>
EBM_INLINE TreeNode<bClassification> * AddBytesTreeNode(TreeNode<bClassification> * pTreeNode, size_t countBytesAdd) {
   return reinterpret_cast<TreeNode<bClassification> *>(reinterpret_cast<char *>(pTreeNode) + countBytesAdd);
}
template<bool bClassification>
EBM_INLINE TreeNode<bClassification> * GetLeftTreeNodeChild(TreeNode<bClassification> * pTreeNodeChildren, size_t countBytesTreeNode) {
   UNUSED(countBytesTreeNode);
   return pTreeNodeChildren;
}
template<bool bClassification>
EBM_INLINE TreeNode<bClassification> * GetRightTreeNodeChild(TreeNode<bClassification> * pTreeNodeChildren, size_t countBytesTreeNode) {
   return AddBytesTreeNode<bClassification>(pTreeNodeChildren, countBytesTreeNode);
}

template<bool bClassification>
class TreeNodeData;

template<>
class TreeNodeData<true> {
   // classification version of the TreeNodeData

public:

   struct BeforeExaminationForPossibleSplitting {
      const HistogramBucket<true> * pHistogramBucketEntryFirst;
      const HistogramBucket<true> * pHistogramBucketEntryLast;
      size_t cInstances;
   };

   struct AfterExaminationForPossibleSplitting {
      TreeNode<true> * pTreeNodeChildren;
      FractionalDataType splitGain; // put this at the top so that our priority queue can access it directly without adding anything to the pointer (this is slightly more efficient on intel systems at least)
      ActiveDataType divisionValue;
   };

   union TreeNodeDataUnion {
      // we can save precious L1 cache space by keeping only what we need
      BeforeExaminationForPossibleSplitting beforeExaminationForPossibleSplitting;
      AfterExaminationForPossibleSplitting afterExaminationForPossibleSplitting;

      static_assert(std::is_pod<BeforeExaminationForPossibleSplitting>::value, "BeforeSplit must be POD (Plain Old Data) if we are going to use it in a union!");
      static_assert(std::is_pod<AfterExaminationForPossibleSplitting>::value, "AfterSplit must be POD (Plain Old Data) if we are going to use it in a union!");
   };

   TreeNodeDataUnion m_UNION;
   HistogramBucketVectorEntry<true> aHistogramBucketVectorEntry[1];

   EBM_INLINE size_t GetInstances() const {
      return m_UNION.beforeExaminationForPossibleSplitting.cInstances;
   }
   EBM_INLINE void SetInstances(size_t cInstances) {
      m_UNION.beforeExaminationForPossibleSplitting.cInstances = cInstances;
   }
};

template<>
class TreeNodeData<false> {
   // regression version of the TreeNodeData
public:

   struct BeforeExaminationForPossibleSplitting {
      const HistogramBucket<false> * pHistogramBucketEntryFirst;
      const HistogramBucket<false> * pHistogramBucketEntryLast;
   };

   struct AfterExaminationForPossibleSplitting {
      TreeNode<false> * pTreeNodeChildren;
      FractionalDataType splitGain; // put this at the top so that our priority queue can access it directly without adding anything to the pointer (this is slightly more efficient on intel systems at least)
      ActiveDataType divisionValue;
   };

   union TreeNodeDataUnion {
      // we can save precious L1 cache space by keeping only what we need
      BeforeExaminationForPossibleSplitting beforeExaminationForPossibleSplitting;
      AfterExaminationForPossibleSplitting afterExaminationForPossibleSplitting;

      static_assert(std::is_pod<BeforeExaminationForPossibleSplitting>::value, "BeforeSplit must be POD (Plain Old Data) if we are going to use it in a union!");
      static_assert(std::is_pod<AfterExaminationForPossibleSplitting>::value, "AfterSplit must be POD (Plain Old Data) if we are going to use it in a union!");
   };

   TreeNodeDataUnion m_UNION;

   size_t m_cInstances;
   HistogramBucketVectorEntry<false> aHistogramBucketVectorEntry[1];

   EBM_INLINE size_t GetInstances() const {
      return m_cInstances;
   }
   EBM_INLINE void SetInstances(size_t cInstances) {
      m_cInstances = cInstances;
   }
};

template<bool bClassification>
class TreeNode final : public TreeNodeData<bClassification> {
public:

   EBM_INLINE bool IsSplittable(size_t cInstancesRequiredForParentSplitMin) const {
      return this->m_UNION.beforeExaminationForPossibleSplitting.pHistogramBucketEntryLast != this->m_UNION.beforeExaminationForPossibleSplitting.pHistogramBucketEntryFirst && cInstancesRequiredForParentSplitMin <= this->GetInstances();
   }

   EBM_INLINE FractionalDataType EXTRACT_GAIN_BEFORE_SPLITTING() {
      EBM_ASSERT(this->m_UNION.afterExaminationForPossibleSplitting.splitGain <= 0);
      return this->m_UNION.afterExaminationForPossibleSplitting.splitGain;
   }

   EBM_INLINE void SPLIT_THIS_NODE() {
      this->m_UNION.afterExaminationForPossibleSplitting.splitGain = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   }

   EBM_INLINE void INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED() {
      // we aren't going to split this TreeNode because we can't.  We need to set the splitGain value here because otherwise it is filled with garbage that could be NaN (meaning the node was a branch)
      // we can't call INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED before calling SplitTreeNode because INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED sets m_UNION.afterExaminationForPossibleSplitting.splitGain and the m_UNION.beforeExaminationForPossibleSplitting values are needed if we had decided to call ExamineNodeForSplittingAndDetermineBestPossibleSplit
      this->m_UNION.afterExaminationForPossibleSplitting.splitGain = FractionalDataType { 0 };
   }

   EBM_INLINE bool WAS_THIS_NODE_SPLIT() const {
      return std::isnan(this->m_UNION.afterExaminationForPossibleSplitting.splitGain);
   }

   // TODO: in theory, a malicious caller could overflow our stack if they pass us data that will grow a sufficiently deep tree.  Consider changing this recursive function to handle that
   // TODO: specialize this function for cases where we have hard coded vector lengths so that we don't have to pass in the cVectorLength parameter
   void Flatten(ActiveDataType ** const ppDivisions, FractionalDataType ** const ppValues, const size_t cVectorLength) const {
      // don't log this since we call it recursively.  Log where the root is called
      if(UNPREDICTABLE(WAS_THIS_NODE_SPLIT())) {
         EBM_ASSERT(!GetTreeNodeSizeOverflow<bClassification>(cVectorLength)); // we're accessing allocated memory
         const size_t cBytesPerTreeNode = GetTreeNodeSize<bClassification>(cVectorLength);
         const TreeNode<bClassification> * const pLeftChild = GetLeftTreeNodeChild<bClassification>(this->m_UNION.afterExaminationForPossibleSplitting.pTreeNodeChildren, cBytesPerTreeNode);
         pLeftChild->Flatten(ppDivisions, ppValues, cVectorLength);
         **ppDivisions = this->m_UNION.afterExaminationForPossibleSplitting.divisionValue;
         ++(*ppDivisions);
         const TreeNode<bClassification> * const pRightChild = GetRightTreeNodeChild<bClassification>(this->m_UNION.afterExaminationForPossibleSplitting.pTreeNodeChildren, cBytesPerTreeNode);
         pRightChild->Flatten(ppDivisions, ppValues, cVectorLength);
      } else {
         FractionalDataType * pValuesCur = *ppValues;
         FractionalDataType * const pValuesNext = pValuesCur + cVectorLength;
         *ppValues = pValuesNext;

         const HistogramBucketVectorEntry<bClassification> * pHistogramBucketVectorEntry = &this->aHistogramBucketVectorEntry[0];
         do {
            FractionalDataType smallChangeToModel;
            if(bClassification) {
               smallChangeToModel = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pHistogramBucketVectorEntry->sumResidualError, pHistogramBucketVectorEntry->GetSumDenominator());
            } else {
               smallChangeToModel = EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(pHistogramBucketVectorEntry->sumResidualError, this->GetInstances());
            }
            *pValuesCur = smallChangeToModel;

            ++pHistogramBucketVectorEntry;
            ++pValuesCur;
         } while(pValuesNext != pValuesCur);
      }
   }

   static_assert(std::is_pod<ActiveDataType>::value, "We want to keep our TreeNode compact and without a virtual pointer table for fitting in L1 cache as much as possible");
};
static_assert(std::is_pod<TreeNode<false>>::value, "We want to keep our TreeNode compact and without a virtual pointer table for fitting in L1 cache as much as possible");
static_assert(std::is_pod<TreeNode<true>>::value, "We want to keep our TreeNode compact and without a virtual pointer table for fitting in L1 cache as much as possible");

#endif // TREE_NODE_H
