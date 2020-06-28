// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TREE_NODE_H
#define TREE_NODE_H

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "EbmStatisticUtils.h"
#include "HistogramBucket.h"

template<bool bClassification>
struct TreeNode;

template<bool bClassification>
EBM_INLINE size_t GetTreeNodeSizeOverflow(const size_t cVectorLength) {
   return IsMultiplyError(sizeof(HistogramBucketVectorEntry<bClassification>), cVectorLength) ? true : IsAddError(sizeof(TreeNode<bClassification>) - 
      sizeof(HistogramBucketVectorEntry<bClassification>), sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength) ? true : false;
}
template<bool bClassification>
EBM_INLINE size_t GetTreeNodeSize(const size_t cVectorLength) {
   return sizeof(TreeNode<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>) + 
      sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength;
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
      // put this at the top so that our priority queue can access it directly without adding anything to the pointer 
      // (this is slightly more efficient on intel systems at least)
      FloatEbmType m_splitGain;
      ActiveDataType m_divisionValue;
   };

   union TreeNodeDataUnion {
      // we can save precious L1 cache space by keeping only what we need
      BeforeExaminationForPossibleSplitting m_beforeExaminationForPossibleSplitting;
      AfterExaminationForPossibleSplitting m_afterExaminationForPossibleSplitting;

      static_assert(
         std::is_standard_layout<BeforeExaminationForPossibleSplitting>::value, 
         "BeforeSplit must be standard layout classes if we are going to use it in a union!");
      static_assert(
         std::is_standard_layout<AfterExaminationForPossibleSplitting>::value, 
         "AfterSplit must be standard layout classes if we are going to use it in a union!");
   };

   TreeNodeDataUnion m_UNION;
   // use the "struct hack" since Flexible array member method is not available in C++
   // m_aHistogramBucketVectorEntry must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   HistogramBucketVectorEntry<true> m_aHistogramBucketVectorEntry[1];

   EBM_INLINE size_t GetInstances() const {
      return m_UNION.m_beforeExaminationForPossibleSplitting.m_cInstances;
   }
   EBM_INLINE void SetInstances(size_t cInstances) {
      m_UNION.m_beforeExaminationForPossibleSplitting.m_cInstances = cInstances;
   }
};
static_assert(std::is_standard_layout<TreeNodeData<true>>::value,
   "TreeNodeData uses the struct hack, so it needs to be standard layout so that we can depend on the placement of member data items");

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
      // put this at the top so that our priority queue can access it directly without adding anything to the pointer 
      // (this is slightly more efficient on intel systems at least)
      FloatEbmType m_splitGain;
      ActiveDataType m_divisionValue;
   };

   union TreeNodeDataUnion {
      // we can save precious L1 cache space by keeping only what we need
      BeforeExaminationForPossibleSplitting m_beforeExaminationForPossibleSplitting;
      AfterExaminationForPossibleSplitting m_afterExaminationForPossibleSplitting;

      static_assert(
         std::is_standard_layout<BeforeExaminationForPossibleSplitting>::value, 
         "BeforeSplit must be a standard layout class if we are going to use it in a union!"
      );
      static_assert(
         std::is_standard_layout<AfterExaminationForPossibleSplitting>::value, 
         "AfterSplit must be a standard layout class if we are going to use it in a union!"
      );
   };

   TreeNodeDataUnion m_UNION;

   size_t m_cInstances;
   // use the "struct hack" since Flexible array member method is not available in C++
   // aHistogramBucketVectorEntry must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   HistogramBucketVectorEntry<false> m_aHistogramBucketVectorEntry[1];

   EBM_INLINE size_t GetInstances() const {
      return m_cInstances;
   }
   EBM_INLINE void SetInstances(size_t cInstances) {
      m_cInstances = cInstances;
   }
};
static_assert(std::is_standard_layout<TreeNodeData<false>>::value,
   "TreeNodeData uses the struct hack, so it needs to be standard layout so that we can depend on the placement of member data items");

template<bool bClassification>
struct TreeNode final : public TreeNodeData<bClassification> {
   // this struct CANNOT have any data in it.  All data MUST be put into TreeNodeData.  TreeNodeData uses the "struct hack", which means that it has a 
   // variable sized data array at the end that would overwrite any data that we put here

public:

   EBM_INLINE bool IsSplittable() const {
      return this->m_UNION.m_beforeExaminationForPossibleSplitting.m_pHistogramBucketEntryLast != 
         this->m_UNION.m_beforeExaminationForPossibleSplitting.m_pHistogramBucketEntryFirst;
   }

   EBM_INLINE FloatEbmType EXTRACT_GAIN_BEFORE_SPLITTING() {
      // m_splitGain is the result of a subtraction between a memory location and a calculation
      // if there is a difference in the number of bits between these two (some floating point processors store more bits)
      // then we could get a negative number, even if mathematically it can't be less than zero
      const FloatEbmType splitGain = this->m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain;
      // in ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint we can get a -infinity gain as a special extremely unlikely case for regression
      EBM_ASSERT(std::isnan(splitGain) || (!bClassification) && std::isinf(splitGain) || k_epsilonNegativeGainAllowed <= splitGain);
      return splitGain;
   }

   EBM_INLINE void SPLIT_THIS_NODE() {
      this->m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain = k_illegalGain;
   }

   EBM_INLINE void INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED() {
      // we aren't going to split this TreeNode because we can't.  We need to set the splitGain value here because otherwise it is filled with garbage 
      // that could be NaN (meaning the node was a branch) we can't call INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED before calling SplitTreeNode 
      // because INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED sets m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain and the 
      // m_UNION.m_beforeExaminationForPossibleSplitting values are needed if we had decided to call ExamineNodeForSplittingAndDetermineBestPossibleSplit
      this->m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain = FloatEbmType { 0 };
   }

   EBM_INLINE bool WAS_THIS_NODE_SPLIT() const {
      return k_illegalGain == this->m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain;
   }

   // TODO: in theory, a malicious caller could overflow our stack if they pass us data that will grow a sufficiently deep tree.  Consider changing this 
   //   recursive function to handle that
   void Flatten(ActiveDataType ** const ppDivisions, FloatEbmType ** const ppValues, const size_t cVectorLength) const {
      // don't log this since we call it recursively.  Log where the root is called
      if(UNPREDICTABLE(WAS_THIS_NODE_SPLIT())) {
         EBM_ASSERT(!GetTreeNodeSizeOverflow<bClassification>(cVectorLength)); // we're accessing allocated memory
         const size_t cBytesPerTreeNode = GetTreeNodeSize<bClassification>(cVectorLength);
         const TreeNode<bClassification> * const pLeftChild = GetLeftTreeNodeChild<bClassification>(
            this->m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren, cBytesPerTreeNode);
         pLeftChild->Flatten(ppDivisions, ppValues, cVectorLength);
         **ppDivisions = this->m_UNION.m_afterExaminationForPossibleSplitting.m_divisionValue;
         ++(*ppDivisions);
         const TreeNode<bClassification> * const pRightChild = GetRightTreeNodeChild<bClassification>(
            this->m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren, cBytesPerTreeNode);
         pRightChild->Flatten(ppDivisions, ppValues, cVectorLength);
      } else {
         FloatEbmType * pValuesCur = *ppValues;
         FloatEbmType * const pValuesNext = pValuesCur + cVectorLength;
         *ppValues = pValuesNext;

         const HistogramBucketVectorEntry<bClassification> * pHistogramBucketVectorEntry = ArrayToPointer(this->m_aHistogramBucketVectorEntry);
         do {
            FloatEbmType smallChangeToModel;
            if(bClassification) {
               smallChangeToModel = EbmStatistics::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                  pHistogramBucketVectorEntry->m_sumResidualError, pHistogramBucketVectorEntry->GetSumDenominator());
            } else {
               smallChangeToModel = EbmStatistics::ComputeSmallChangeForOneSegmentRegression(
                  pHistogramBucketVectorEntry->m_sumResidualError, static_cast<FloatEbmType>(this->GetInstances()));
            }
            *pValuesCur = smallChangeToModel;

            ++pHistogramBucketVectorEntry;
            ++pValuesCur;
         } while(pValuesNext != pValuesCur);
      }
   }
};
static_assert(std::is_standard_layout<TreeNode<false>>::value && std::is_standard_layout<TreeNode<true>>::value, 
   "TreeNode uses the struct hack, so it needs to be standard layout so that we can depend on the placement of member data items");

template<bool bClassification>
struct SweepTreeNode {
   size_t m_cBestInstancesLeft;
   const HistogramBucket<bClassification> * m_pBestHistogramBucketEntry;

   // use the "struct hack" since Flexible array member method is not available in C++
   // m_aHistogramBucketVectorEntry must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   HistogramBucketVectorEntry<bClassification> m_aBestHistogramBucketVectorEntry[1];
};
static_assert(
   std::is_standard_layout<SweepTreeNode<false>>::value && std::is_standard_layout<SweepTreeNode<true>>::value,
   "using the struct hack requires that our class have guaranteed member positions, hense it needs to be standard layout");

template<bool bClassification>
EBM_INLINE bool GetSweepTreeNodeSizeOverflow(const size_t cVectorLength) {
   return IsMultiplyError(sizeof(HistogramBucketVectorEntry<bClassification>), cVectorLength) ?
      true :
      IsAddError(sizeof(SweepTreeNode<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>),
         sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength) ? true : false;
}
template<bool bClassification>
EBM_INLINE size_t GetSweepTreeNodeSize(const size_t cVectorLength) {
   return sizeof(SweepTreeNode<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>) +
      sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength;
}
template<bool bClassification>
EBM_INLINE SweepTreeNode<bClassification> * AddBytesSweepTreeNode(SweepTreeNode<bClassification> * const pSweepTreeNode, const size_t cBytesAdd) {
   return reinterpret_cast<SweepTreeNode<bClassification> *>(reinterpret_cast<char *>(pSweepTreeNode) + cBytesAdd);
}
template<bool bClassification>
EBM_INLINE size_t CountSweepTreeNode(
   const SweepTreeNode<bClassification> * const pSweepTreeNodeStart,
   const SweepTreeNode<bClassification> * const pSweepTreeNodeCur,
   const size_t cBytesPerSweepTreeNode
) {
   EBM_ASSERT(reinterpret_cast<const char *>(pSweepTreeNodeStart) <= reinterpret_cast<const char *>(pSweepTreeNodeCur));
   const size_t cBytesDiff = reinterpret_cast<const char *>(pSweepTreeNodeCur) - reinterpret_cast<const char *>(pSweepTreeNodeStart);
   EBM_ASSERT(0 == cBytesDiff % cBytesPerSweepTreeNode);
   return cBytesDiff / cBytesPerSweepTreeNode;
}

#endif // TREE_NODE_H
