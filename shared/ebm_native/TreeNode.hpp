// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TREE_NODE_HPP
#define TREE_NODE_HPP

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "ebm_stats.hpp"
#include "HistogramBucket.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<bool bClassification>
struct TreeNode;

template<bool bClassification>
struct TreeNodeData;

template<>
struct TreeNodeData<true> {
   // classification version of the TreeNodeData

   TreeNodeData() = default; // preserve our POD status
   ~TreeNodeData() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   struct BeforeExaminationForPossibleSplitting final {
      BeforeExaminationForPossibleSplitting() = default; // preserve our POD status
      ~BeforeExaminationForPossibleSplitting() = default; // preserve our POD status
      void * operator new(std::size_t) = delete; // we only use malloc/free in this library
      void operator delete (void *) = delete; // we only use malloc/free in this library

      const Bin<FloatBig, true> * m_pBinFirst;
      const Bin<FloatBig, true> * m_pBinLast;
      size_t m_cSamples;
   };
   static_assert(std::is_standard_layout<BeforeExaminationForPossibleSplitting>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
   static_assert(std::is_trivial<BeforeExaminationForPossibleSplitting>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");
   static_assert(std::is_pod<BeforeExaminationForPossibleSplitting>::value,
      "We use a lot of C constructs, so disallow non-POD types in general");

   struct AfterExaminationForPossibleSplitting final {
      AfterExaminationForPossibleSplitting() = default; // preserve our POD status
      ~AfterExaminationForPossibleSplitting() = default; // preserve our POD status
      void * operator new(std::size_t) = delete; // we only use malloc/free in this library
      void operator delete (void *) = delete; // we only use malloc/free in this library

      TreeNode<true> * m_pTreeNodeChildren;
      // put this at the top so that our priority queue can access it directly without adding anything to the pointer 
      // (this is slightly more efficient on intel systems at least)
      FloatBig m_splitGain;
      ActiveDataType m_splitValue;
   };
   static_assert(std::is_standard_layout<AfterExaminationForPossibleSplitting>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
   static_assert(std::is_trivial<AfterExaminationForPossibleSplitting>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");
   static_assert(std::is_pod<AfterExaminationForPossibleSplitting>::value,
      "We use a lot of C constructs, so disallow non-POD types in general");

   union TreeNodeDataUnion final {

#ifndef __SUNPRO_CC

      // the Oracle Developer Studio compiler has what I think is a bug by making any class that includes 
      // TreeNodeDataUnion fields turn into non-trivial classes, so exclude the Oracle compiler
      // from these protections

      TreeNodeDataUnion() = default; // preserve our POD status
      ~TreeNodeDataUnion() = default; // preserve our POD status
      void * operator new(std::size_t) = delete; // we only use malloc/free in this library
      void operator delete (void *) = delete; // we only use malloc/free in this library

#endif // __SUNPRO_CC

      // we can save precious L1 cache space by keeping only what we need
      BeforeExaminationForPossibleSplitting m_beforeExaminationForPossibleSplitting;
      AfterExaminationForPossibleSplitting m_afterExaminationForPossibleSplitting;
   };
   static_assert(std::is_standard_layout<TreeNodeDataUnion>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
   static_assert(std::is_trivial<TreeNodeDataUnion>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");
   static_assert(std::is_pod<TreeNodeDataUnion>::value,
      "We use a lot of C constructs, so disallow non-POD types in general");

   INLINE_ALWAYS size_t AMBIGUOUS_GetCountSamples() const {
      EBM_ASSERT(!IsExaminedForPossibleSplitting());
      return m_UNION.m_beforeExaminationForPossibleSplitting.m_cSamples;
   }
   INLINE_ALWAYS void AMBIGUOUS_SetCountSamples(const size_t cSamples) {
      EBM_ASSERT(!IsExaminedForPossibleSplitting());
      m_UNION.m_beforeExaminationForPossibleSplitting.m_cSamples = cSamples;
   }

   INLINE_ALWAYS FloatBig GetWeight() const {
      return m_weight;
   }
   INLINE_ALWAYS void SetWeight(const FloatBig weight) {
      m_weight = weight;
   }

   INLINE_ALWAYS const Bin<FloatBig, true> * BEFORE_GetBinFirst() const {
      EBM_ASSERT(!IsExaminedForPossibleSplitting());
      return m_UNION.m_beforeExaminationForPossibleSplitting.m_pBinFirst;
   }
   INLINE_ALWAYS void BEFORE_SetBinFirst(
      const Bin<FloatBig, true> * const pBinFirst)
   {
      EBM_ASSERT(!IsExaminedForPossibleSplitting());
      m_UNION.m_beforeExaminationForPossibleSplitting.m_pBinFirst = pBinFirst;
   }

   INLINE_ALWAYS const Bin<FloatBig, true> * BEFORE_GetBinLast() const {
      EBM_ASSERT(!IsExaminedForPossibleSplitting());
      return m_UNION.m_beforeExaminationForPossibleSplitting.m_pBinLast;
   }
   INLINE_ALWAYS void BEFORE_SetBinLast(
      const Bin<FloatBig, true> * const pBinLast)
   {
      EBM_ASSERT(!IsExaminedForPossibleSplitting());
      m_UNION.m_beforeExaminationForPossibleSplitting.m_pBinLast = pBinLast;
   }

   INLINE_ALWAYS const TreeNode<true> * AFTER_GetTreeNodeChildren() const {
      EBM_ASSERT(IsExaminedForPossibleSplitting());
      return m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren;
   }
   INLINE_ALWAYS TreeNode<true> * AFTER_GetTreeNodeChildren() {
      EBM_ASSERT(IsExaminedForPossibleSplitting());
      return m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren;
   }
   INLINE_ALWAYS void AFTER_SetTreeNodeChildren(TreeNode<true> * const pTreeNodeChildren) {
      EBM_ASSERT(IsExaminedForPossibleSplitting());
      m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren = pTreeNodeChildren;
   }

   INLINE_ALWAYS FloatBig AFTER_GetSplitGain() const {
      EBM_ASSERT(IsExaminedForPossibleSplitting());
      return m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain;
   }
   INLINE_ALWAYS void AFTER_SetSplitGain(const FloatBig splitGain) {
      EBM_ASSERT(IsExaminedForPossibleSplitting());
      m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain = splitGain;
   }

   INLINE_ALWAYS ActiveDataType AFTER_GetSplitValue() const {
      EBM_ASSERT(IsExaminedForPossibleSplitting());
      return m_UNION.m_afterExaminationForPossibleSplitting.m_splitValue;
   }
   INLINE_ALWAYS void AFTER_SetSplitValue(const ActiveDataType splitValue) {
      EBM_ASSERT(IsExaminedForPossibleSplitting());
      m_UNION.m_afterExaminationForPossibleSplitting.m_splitValue = splitValue;
   }

   INLINE_ALWAYS const HistogramTargetEntry<FloatBig, true> * GetHistogramTargetEntry() const {
      return ArrayToPointer(m_aHistogramTargetEntry);
   }
   INLINE_ALWAYS HistogramTargetEntry<FloatBig, true> * GetHistogramTargetEntry() {
      return ArrayToPointer(m_aHistogramTargetEntry);
   }

#ifndef NDEBUG
   INLINE_ALWAYS bool IsExaminedForPossibleSplitting() const {
      return m_bExaminedForPossibleSplitting;
   }
   INLINE_ALWAYS void SetExaminedForPossibleSplitting(const bool bExaminedForPossibleSplitting) {
      if(bExaminedForPossibleSplitting) {
         // we set this to false when it's random memory, 
         // but we only flip it to true from an initialized state of false
         EBM_ASSERT(!m_bExaminedForPossibleSplitting);
      }
      m_bExaminedForPossibleSplitting = bExaminedForPossibleSplitting;
   }
#endif // NDEBUG

private:

#ifndef NDEBUG
   bool m_bExaminedForPossibleSplitting;
#endif // NDEBUG

   FloatBig m_weight;

   TreeNodeDataUnion m_UNION;
   // use the "struct hack" since Flexible array member method is not available in C++
   // m_aHistogramTargetEntry must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   HistogramTargetEntry<FloatBig, true> m_aHistogramTargetEntry[1];
};
static_assert(std::is_standard_layout<TreeNodeData<true>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<TreeNodeData<true>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<TreeNodeData<true>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

template<>
struct TreeNodeData<false> {
   // regression version of the TreeNodeData

   TreeNodeData() = default; // preserve our POD status
   ~TreeNodeData() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   struct BeforeExaminationForPossibleSplitting final {
      BeforeExaminationForPossibleSplitting() = default; // preserve our POD status
      ~BeforeExaminationForPossibleSplitting() = default; // preserve our POD status
      void * operator new(std::size_t) = delete; // we only use malloc/free in this library
      void operator delete (void *) = delete; // we only use malloc/free in this library

      const Bin<FloatBig, false> * m_pBinFirst;
      const Bin<FloatBig, false> * m_pBinLast;
   };
   static_assert(std::is_standard_layout<BeforeExaminationForPossibleSplitting>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
   static_assert(std::is_trivial<BeforeExaminationForPossibleSplitting>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");
   static_assert(std::is_pod<BeforeExaminationForPossibleSplitting>::value,
      "We use a lot of C constructs, so disallow non-POD types in general");

   struct AfterExaminationForPossibleSplitting final {
      AfterExaminationForPossibleSplitting() = default; // preserve our POD status
      ~AfterExaminationForPossibleSplitting() = default; // preserve our POD status
      void * operator new(std::size_t) = delete; // we only use malloc/free in this library
      void operator delete (void *) = delete; // we only use malloc/free in this library

      TreeNode<false> * m_pTreeNodeChildren;
      // put this at the top so that our priority queue can access it directly without adding anything to the pointer 
      // (this is slightly more efficient on intel systems at least)
      FloatBig m_splitGain;
      ActiveDataType m_splitValue;
   };
   static_assert(std::is_standard_layout<AfterExaminationForPossibleSplitting>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
   static_assert(std::is_trivial<AfterExaminationForPossibleSplitting>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");
   static_assert(std::is_pod<AfterExaminationForPossibleSplitting>::value,
      "We use a lot of C constructs, so disallow non-POD types in general");

   union TreeNodeDataUnion final {

#ifndef __SUNPRO_CC

      // the Oracle Developer Studio compiler has what I think is a bug by making any class that includes 
      // TreeNodeDataUnion fields turn into non-trivial classes, so exclude the Oracle compiler
      // from these protections

      TreeNodeDataUnion() = default; // preserve our POD status
      ~TreeNodeDataUnion() = default; // preserve our POD status
      void * operator new(std::size_t) = delete; // we only use malloc/free in this library
      void operator delete (void *) = delete; // we only use malloc/free in this library

#endif // __SUNPRO_CC

      // we can save precious L1 cache space by keeping only what we need
      BeforeExaminationForPossibleSplitting m_beforeExaminationForPossibleSplitting;
      AfterExaminationForPossibleSplitting m_afterExaminationForPossibleSplitting;
   };
   static_assert(std::is_standard_layout<TreeNodeDataUnion>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
   static_assert(std::is_trivial<TreeNodeDataUnion>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");
   static_assert(std::is_pod<TreeNodeDataUnion>::value,
      "We use a lot of C constructs, so disallow non-POD types in general");

   INLINE_ALWAYS size_t AMBIGUOUS_GetCountSamples() const {
      return m_cSamples;
   }
   INLINE_ALWAYS void AMBIGUOUS_SetCountSamples(const size_t cSamples) {
      m_cSamples = cSamples;
   }

   INLINE_ALWAYS FloatBig GetWeight() const {
      return m_weight;
   }
   INLINE_ALWAYS void SetWeight(const FloatBig weight) {
      m_weight = weight;
   }

   INLINE_ALWAYS const Bin<FloatBig, false> * BEFORE_GetBinFirst() const {
      EBM_ASSERT(!IsExaminedForPossibleSplitting());
      return m_UNION.m_beforeExaminationForPossibleSplitting.m_pBinFirst;
   }
   INLINE_ALWAYS void BEFORE_SetBinFirst(
      const Bin<FloatBig, false> * const pBinFirst)
   {
      EBM_ASSERT(!IsExaminedForPossibleSplitting());
      m_UNION.m_beforeExaminationForPossibleSplitting.m_pBinFirst = pBinFirst;
   }

   INLINE_ALWAYS const Bin<FloatBig, false> * BEFORE_GetBinLast() const {
      EBM_ASSERT(!IsExaminedForPossibleSplitting());
      return m_UNION.m_beforeExaminationForPossibleSplitting.m_pBinLast;
   }
   INLINE_ALWAYS void BEFORE_SetBinLast(
      const Bin<FloatBig, false> * const pBinLast)
   {
      EBM_ASSERT(!IsExaminedForPossibleSplitting());
      m_UNION.m_beforeExaminationForPossibleSplitting.m_pBinLast = pBinLast;
   }

   INLINE_ALWAYS const TreeNode<false> * AFTER_GetTreeNodeChildren() const {
      EBM_ASSERT(IsExaminedForPossibleSplitting());
      return m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren;
   }
   INLINE_ALWAYS TreeNode<false> * AFTER_GetTreeNodeChildren() {
      EBM_ASSERT(IsExaminedForPossibleSplitting());
      return m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren;
   }
   INLINE_ALWAYS void AFTER_SetTreeNodeChildren(TreeNode<false> * const pTreeNodeChildren) {
      EBM_ASSERT(IsExaminedForPossibleSplitting());
      m_UNION.m_afterExaminationForPossibleSplitting.m_pTreeNodeChildren = pTreeNodeChildren;
   }

   INLINE_ALWAYS FloatBig AFTER_GetSplitGain() const {
      EBM_ASSERT(IsExaminedForPossibleSplitting());
      return m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain;
   }
   INLINE_ALWAYS void AFTER_SetSplitGain(const FloatBig splitGain) {
      EBM_ASSERT(IsExaminedForPossibleSplitting());
      m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain = splitGain;
   }

   INLINE_ALWAYS ActiveDataType AFTER_GetSplitValue() const {
      EBM_ASSERT(IsExaminedForPossibleSplitting());
      return m_UNION.m_afterExaminationForPossibleSplitting.m_splitValue;
   }
   INLINE_ALWAYS void AFTER_SetSplitValue(const ActiveDataType splitValue) {
      EBM_ASSERT(IsExaminedForPossibleSplitting());
      m_UNION.m_afterExaminationForPossibleSplitting.m_splitValue = splitValue;
   }

   INLINE_ALWAYS const HistogramTargetEntry<FloatBig, false> * GetHistogramTargetEntry() const {
      return ArrayToPointer(m_aHistogramTargetEntry);
   }
   INLINE_ALWAYS HistogramTargetEntry<FloatBig, false> * GetHistogramTargetEntry() {
      return ArrayToPointer(m_aHistogramTargetEntry);
   }

#ifndef NDEBUG
   INLINE_ALWAYS bool IsExaminedForPossibleSplitting() const {
      return m_bExaminedForPossibleSplitting;
   }
   INLINE_ALWAYS void SetExaminedForPossibleSplitting(const bool bExaminedForPossibleSplitting) {
      if(bExaminedForPossibleSplitting) {
         // we set this to false when it's random memory, 
         // but we only flip it to true from an initialized state of false
         EBM_ASSERT(!m_bExaminedForPossibleSplitting);
      }
      m_bExaminedForPossibleSplitting = bExaminedForPossibleSplitting;
   }
#endif // NDEBUG

private:

#ifndef NDEBUG
   bool m_bExaminedForPossibleSplitting;
#endif // NDEBUG

   FloatBig m_weight;

   TreeNodeDataUnion m_UNION;

   size_t m_cSamples;
   // use the "struct hack" since Flexible array member method is not available in C++
   // m_aHistogramTargetEntry must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   HistogramTargetEntry<FloatBig, false> m_aHistogramTargetEntry[1];
};
static_assert(std::is_standard_layout<TreeNodeData<false>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<TreeNodeData<false>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<TreeNodeData<false>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

template<bool bClassification>
struct TreeNode final : public TreeNodeData<bClassification> {
   // this struct CANNOT have any data in it.  All data MUST be put into TreeNodeData.  TreeNodeData uses the "struct hack", which means that it has a 
   // variable sized data array at the end that would overwrite any data that we put here

public:

   TreeNode() = default; // preserve our POD status
   ~TreeNode() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS bool IsSplittable() const {
      return this->BEFORE_GetBinLast() != 
         this->BEFORE_GetBinFirst();
   }

   INLINE_ALWAYS FloatBig EXTRACT_GAIN_BEFORE_SPLITTING() {
      // m_splitGain is the result of a subtraction between a memory location and a calculation
      // if there is a difference in the number of bits between these two (some floating point processors store more bits)
      // then we could get a negative number, even if mathematically it can't be less than zero
      const FloatBig splitGain = this->AFTER_GetSplitGain();
      // our priority queue cannot handle NaN values so we filter them out before adding them
      EBM_ASSERT(!std::isnan(splitGain));
      EBM_ASSERT(!std::isinf(splitGain));
      // in ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint we can get a -infinity gain as a special extremely unlikely case for regression
      EBM_ASSERT(0 <= splitGain);
      return splitGain;
   }

   INLINE_ALWAYS void SPLIT_THIS_NODE() {
      this->AFTER_SetSplitGain(k_illegalGainFloat);
   }

   INLINE_ALWAYS void INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED() {
      // we aren't going to split this TreeNode because we can't.  We need to set the splitGain value here because otherwise it is filled with garbage 
      // that could be NaN (meaning the node was a branch) we can't call INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED before calling SplitTreeNode 
      // because INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED sets m_UNION.m_afterExaminationForPossibleSplitting.m_splitGain and the 
      // m_UNION.m_beforeExaminationForPossibleSplitting values are needed if we had decided to call ExamineNodeForSplittingAndDetermineBestPossibleSplit
      this->AFTER_SetSplitGain(0);
   }

   INLINE_ALWAYS bool WAS_THIS_NODE_SPLIT() const {
      return k_illegalGainFloat == this->AFTER_GetSplitGain();
   }
};
static_assert(std::is_standard_layout<TreeNode<true>>::value && std::is_standard_layout<TreeNode<false>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<TreeNode<true>>::value && std::is_trivial<TreeNode<false>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<TreeNode<true>>::value && std::is_pod<TreeNode<false>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

INLINE_ALWAYS bool GetTreeNodeSizeOverflow(const bool bClassification, const size_t cScores) {
   const size_t cBytesHistogramTargetEntry = GetHistogramTargetEntrySize<FloatBig>(bClassification);

   if(UNLIKELY(IsMultiplyError(cBytesHistogramTargetEntry, cScores))) {
      return true;
   }

   size_t cBytesTreeNodeComponent;
   if(bClassification) {
      cBytesTreeNodeComponent = sizeof(TreeNode<true>);
   } else {
      cBytesTreeNodeComponent = sizeof(TreeNode<false>);
   }
   cBytesTreeNodeComponent -= cBytesHistogramTargetEntry;

   if(UNLIKELY(IsAddError(cBytesTreeNodeComponent, cBytesHistogramTargetEntry * cScores))) {
      return true;
   }

   return false;
}

INLINE_ALWAYS size_t GetTreeNodeSize(const bool bClassification, const size_t cScores) {
   const size_t cBytesHistogramTargetEntry = GetHistogramTargetEntrySize<FloatBig>(bClassification);

   size_t cBytesTreeNodeComponent;
   if(bClassification) {
      cBytesTreeNodeComponent = sizeof(TreeNode<true>);
   } else {
      cBytesTreeNodeComponent = sizeof(TreeNode<false>);
   }
   cBytesTreeNodeComponent -= cBytesHistogramTargetEntry;

   return cBytesTreeNodeComponent + cBytesHistogramTargetEntry * cScores;
}

template<bool bClassification>
INLINE_ALWAYS TreeNode<bClassification> * AddBytesTreeNode(TreeNode<bClassification> * const pTreeNode, const size_t cBytesAdd) {
   return reinterpret_cast<TreeNode<bClassification> *>(reinterpret_cast<char *>(pTreeNode) + cBytesAdd);
}

template<bool bClassification>
INLINE_ALWAYS const TreeNode<bClassification> * AddBytesTreeNode(const TreeNode<bClassification> * const pTreeNode, const size_t cBytesAdd) {
   return reinterpret_cast<const TreeNode<bClassification> *>(reinterpret_cast<const char *>(pTreeNode) + cBytesAdd);
}

template<bool bClassification>
INLINE_ALWAYS TreeNode<bClassification> * GetLeftTreeNodeChild(TreeNode<bClassification> * const pTreeNodeChildren, const size_t cBytesTreeNode) {
   UNUSED(cBytesTreeNode);
   return pTreeNodeChildren;
}

template<bool bClassification>
INLINE_ALWAYS const TreeNode<bClassification> * GetLeftTreeNodeChild(const TreeNode<bClassification> * const pTreeNodeChildren, const size_t cBytesTreeNode) {
   UNUSED(cBytesTreeNode);
   return pTreeNodeChildren;
}

template<bool bClassification>
INLINE_ALWAYS TreeNode<bClassification> * GetRightTreeNodeChild(TreeNode<bClassification> * const pTreeNodeChildren, const size_t cBytesTreeNode) {
   return AddBytesTreeNode<bClassification>(pTreeNodeChildren, cBytesTreeNode);
}

template<bool bClassification>
INLINE_ALWAYS const TreeNode<bClassification> * GetRightTreeNodeChild(const TreeNode<bClassification> * const pTreeNodeChildren, const size_t cBytesTreeNode) {
   return AddBytesTreeNode<bClassification>(pTreeNodeChildren, cBytesTreeNode);
}

} // DEFINED_ZONE_NAME

#endif // TREE_NODE_HPP
