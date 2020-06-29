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
};
static_assert(std::is_standard_layout<TreeNode<false>>::value && std::is_standard_layout<TreeNode<true>>::value, 
   "TreeNode uses the struct hack, so it needs to be standard layout so that we can depend on the placement of member data items");

EBM_INLINE size_t GetTreeNodeSizeOverflow(const bool bClassification, const size_t cVectorLength) {
   const size_t cBytesHistogramTargetEntry = bClassification ?
      sizeof(HistogramBucketVectorEntry<true>) :
      sizeof(HistogramBucketVectorEntry<false>);

   if(UNLIKELY(IsMultiplyError(cBytesHistogramTargetEntry, cVectorLength))) {
      return true;
   }

   const size_t cBytesTreeNodeComponent = bClassification ?
      (sizeof(TreeNode<true>) - sizeof(HistogramBucketVectorEntry<true>)) :
      (sizeof(TreeNode<false>) - sizeof(HistogramBucketVectorEntry<false>));

   if(UNLIKELY(IsAddError(cBytesTreeNodeComponent, cBytesHistogramTargetEntry * cVectorLength))) {
      return true;
   }

   return false;
}

EBM_INLINE size_t GetTreeNodeSize(const bool bClassification, const size_t cVectorLength) {
   const size_t cBytesTreeNodeComponent = bClassification ?
      sizeof(TreeNode<true>) - sizeof(HistogramBucketVectorEntry<true>) :
      sizeof(TreeNode<false>) - sizeof(HistogramBucketVectorEntry<false>);

   const size_t cBytesHistogramTargetEntry = bClassification ?
      sizeof(HistogramBucketVectorEntry<true>) :
      sizeof(HistogramBucketVectorEntry<false>);

   return cBytesTreeNodeComponent + cBytesHistogramTargetEntry * cVectorLength;
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

#endif // TREE_NODE_H
