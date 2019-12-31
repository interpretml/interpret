// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TREE_NODE_H
#define TREE_NODE_H

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "EbmStatistics.h"
#include "HistogramBucket.h"

template<bool bClassification>
struct TreeNode;

template<bool bClassification>
EBM_INLINE size_t GetTreeNodeSizeOverflow(const size_t cVectorLength) {
   return IsMultiplyError(sizeof(HistogramBucketVectorEntry<bClassification>), cVectorLength) ? true : IsAddError(sizeof(TreeNode<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>), sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength) ? true : false;
}
template<bool bClassification>
EBM_INLINE size_t GetTreeNodeSize(const size_t cVectorLength) {
   return sizeof(TreeNode<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>) + sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength;
}
template<bool bClassification>
EBM_INLINE TreeNode<bClassification> * AddBytesTreeNode(TreeNode<bClassification> * const pTreeNode, const size_t cBytesAdd) {
   return reinterpret_cast<TreeNode<bClassification> *>(reinterpret_cast<char *>(pTreeNode) + cBytesAdd);
}
template<bool bClassification>
EBM_INLINE TreeNode<bClassification> * GetLeftTreeNodeChild(TreeNode<bClassification> * const pTreeNodeChildren, const size_t cBytesTreeNode) {
   UNUSED(cBytesTreeNode);
   return pTreeNodeChildren;
}
template<bool bClassification>
EBM_INLINE TreeNode<bClassification> * GetRightTreeNodeChild(TreeNode<bClassification> * const pTreeNodeChildren, const size_t cBytesTreeNode) {
   return AddBytesTreeNode<bClassification>(pTreeNodeChildren, cBytesTreeNode);
}

template<bool bClassification>
struct TreeNodeData;

template<>
struct TreeNodeData<true> {
   // classification version of the TreeNodeData

public:

   struct BeforeExaminationForPossibleSplitting {
      const HistogramBucket<true> * m_pHistogramBucketEntryFirst;
      const HistogramBucket<true> * m_pHistogramBucketEntryLast;
      size_t m_cInstances;
   };

   struct AfterExaminationForPossibleSplitting {
      TreeNode<true> * m_pTreeNodeChildren;
      FractionalDataType m_splitGain; // put this at the top so that our priority queue can access it directly without adding anything to the pointer (this is slightly more efficient on intel systems at least)
      ActiveDataType m_divisionValue;
   };

   union TreeNodeDataUnion {
      // we can save precious L1 cache space by keeping only what we need
      BeforeExaminationForPossibleSplitting m_beforeExaminationForPossibleSplitting;
      AfterExaminationForPossibleSplitting m_afterExaminationForPossibleSplitting;

      static_assert(std::is_standard_layout<BeforeExaminationForPossibleSplitting>::value, "BeforeSplit must be standard layout classes if we are going to use it in a union!");
      static_assert(std::is_standard_layout<AfterExaminationForPossibleSplitting>::value, "AfterSplit must be standard layout classes if we are going to use it in a union!");
   };

   TreeNodeDataUnion m_UNION;
   // use the "struct hack" since Flexible array member method is not available in C++
   // m_aHistogramBucketVectorEntry must be the last item in this struct
   HistogramBucketVectorEntry<true> m_aHistogramBucketVectorEntry[1];

   EBM_INLINE size_t GetInstances() const {
      return m_UNION.m_beforeExaminationForPossibleSplitting.m_cInstances;
   }
   EBM_INLINE void SetInstances(size_t cInstances) {
      m_UNION.m_beforeExaminationForPossibleSplitting.m_cInstances = cInstances;
   }
};

template<>
struct TreeNodeData<false> {
   // regression version of the TreeNodeData
public:

   struct BeforeExaminationForPossibleSplitting {
      const HistogramBucket<false> * m_pHistogramBucketEntryFirst;
      const HistogramBucket<false> * m_pHistogramBucketEntryLast;
   };

   struct AfterExaminationForPossibleSplitting {
      TreeNode<false> * m_pTreeNodeChildren;
      FractionalDataType m_splitGain; // put this at the top so that our priority queue can access it directly without adding anything to the pointer (this is slightly more efficient on intel systems at least)
      ActiveDataType m_divisionValue;
   };

   union TreeNodeDataUnion {
      // we can save precious L1 cache space by keeping only what we need
      BeforeExaminationForPossibleSplitting m_beforeExaminationForPossibleSplitting;
      AfterExaminationForPossibleSplitting m_afterExaminationForPossibleSplitting;

      static_assert(std::is_standard_layout<BeforeExaminationForPossibleSplitting>::value, "BeforeSplit must be a standard layout class if we are going to use it in a union!");
      static_assert(std::is_standard_layout<AfterExaminationForPossibleSplitting>::value, "AfterSplit must be a standard layout class if we are going to use it in a union!");
   };

   TreeNodeDataUnion m_UNION;

   size_t m_cInstances;
   // use the "struct hack" since Flexible array member method is not available in C++
   // aHistogramBucketVectorEntry must be the last item in this struct
   HistogramBucketVectorEntry<false> m_aHistogramBucketVectorEntry[1];

   EBM_INLINE size_t GetInstances() const {
      return m_cInstances;
   }
   EBM_INLINE void SetInstances(size_t cInstances) {
      m_cInstances = cInstances;
   }
};

template<bool bClassification>
struct TreeNode final : public TreeNodeData<bClassification> {
   // this struct CANNOT have any data in it.  All data MUST be put into TreeNodeData.  TreeNodeData uses the "struct hack", which means that it has a variable sized data array at the end that would overwrite any data that we put here

public:

   EBM_INLINE bool IsSplittable(size_t cInstancesRequiredForParentSplitMin) const {
      return this->m_UNION.m_beforeExaminationForPossibleSplitting.m_pHistogramBucketEntryLast != this->m_UNION.m_beforeExaminationForPossibleSplitting.m_pHistogramBucketEntryFirst && cInstancesRequiredForParentSplitMin <= this->GetInstances();
   }

   EBM_INLINE FractionalDataType EXTRACT_GAIN_BEFORE_SPLITTING() {
      EBM_ASSERT(-0.000000001 <= this->m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain);
      return this->m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain;
   }

   EBM_INLINE void SPLIT_THIS_NODE() {
      this->m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain = k_illegalGain;
   }

   EBM_INLINE void INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED() {
      // we aren't going to split this TreeNode because we can't.  We need to set the splitGain value here because otherwise it is filled with garbage that could be NaN (meaning the node was a branch)
      // we can't call INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED before calling SplitTreeNode because INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED sets m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain and the m_UNION.m_beforeExaminationForPossibleSplitting values are needed if we had decided to call ExamineNodeForSplittingAndDetermineBestPossibleSplit
      this->m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain = FractionalDataType { 0 };
   }

   EBM_INLINE bool WAS_THIS_NODE_SPLIT() const {
      return k_illegalGain == this->m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain;
   }

   // TODO: in theory, a malicious caller could overflow our stack if they pass us data that will grow a sufficiently deep tree.  Consider changing this recursive function to handle that
   // TODO: specialize this function for cases where we have hard coded vector lengths so that we don't have to pass in the cVectorLength parameter
   void Flatten(ActiveDataType ** const ppDivisions, FractionalDataType ** const ppValues, const size_t cVectorLength) const {
      // don't log this since we call it recursively.  Log where the root is called
      if(UNPREDICTABLE(WAS_THIS_NODE_SPLIT())) {
         EBM_ASSERT(!GetTreeNodeSizeOverflow<bClassification>(cVectorLength)); // we're accessing allocated memory
         const size_t cBytesPerTreeNode = GetTreeNodeSize<bClassification>(cVectorLength);
         const TreeNode<bClassification> * const pLeftChild = GetLeftTreeNodeChild<bClassification>(this->m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren, cBytesPerTreeNode);
         pLeftChild->Flatten(ppDivisions, ppValues, cVectorLength);
         **ppDivisions = this->m_UNION.m_afterExaminationForPossibleSplitting.m_divisionValue;
         ++(*ppDivisions);
         const TreeNode<bClassification> * const pRightChild = GetRightTreeNodeChild<bClassification>(this->m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren, cBytesPerTreeNode);
         pRightChild->Flatten(ppDivisions, ppValues, cVectorLength);
      } else {
         FractionalDataType * pValuesCur = *ppValues;
         FractionalDataType * const pValuesNext = pValuesCur + cVectorLength;
         *ppValues = pValuesNext;

         const HistogramBucketVectorEntry<bClassification> * pHistogramBucketVectorEntry = ARRAY_TO_POINTER_CONST(this->m_aHistogramBucketVectorEntry);
         do {
            FractionalDataType smallChangeToModel;
            if(bClassification) {
               smallChangeToModel = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pHistogramBucketVectorEntry->m_sumResidualError, pHistogramBucketVectorEntry->GetSumDenominator());
            } else {
               smallChangeToModel = EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(pHistogramBucketVectorEntry->m_sumResidualError, static_cast<FractionalDataType>(this->GetInstances()));
            }
            *pValuesCur = smallChangeToModel;

            ++pHistogramBucketVectorEntry;
            ++pValuesCur;
         } while(pValuesNext != pValuesCur);
      }
   }
};
static_assert(std::is_standard_layout<TreeNode<false>>::value && std::is_standard_layout<TreeNode<true>>::value, "We want to keep our TreeNode compact and without a virtual pointer table for fitting in L1 cache as much as possible");

#endif // TREE_NODE_H
