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

#include "Bin.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<bool bClassification>
struct TreeNode final {
   friend bool IsOverflowTreeNodeSize(const bool, const size_t);
   friend size_t GetTreeNodeSize(const bool, const size_t);

   TreeNode() = default; // preserve our POD status
   ~TreeNode() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS const Bin<FloatBig, bClassification> * BEFORE_GetBinFirst() const {
      EBM_ASSERT(!m_bDoneGainCalc);
      return m_UNION.m_beforeGainCalc.m_pBinFirst;
   }
   INLINE_ALWAYS void BEFORE_SetBinFirst(const Bin<FloatBig, bClassification> * const pBinFirst) {
      EBM_ASSERT(!m_bDoneGainCalc);
      m_UNION.m_beforeGainCalc.m_pBinFirst = pBinFirst;
   }

   INLINE_ALWAYS const Bin<FloatBig, bClassification> * BEFORE_GetBinLast() const {
      EBM_ASSERT(!m_bDoneGainCalc);
      return m_UNION.m_beforeGainCalc.m_pBinLast;
   }
   INLINE_ALWAYS void BEFORE_SetBinLast(const Bin<FloatBig, bClassification> * const pBinLast) {
      EBM_ASSERT(!m_bDoneGainCalc);
      m_UNION.m_beforeGainCalc.m_pBinLast = pBinLast;
   }

   INLINE_ALWAYS bool BEFORE_IsSplittable() const {
      EBM_ASSERT(!m_bDoneGainCalc);
      return this->BEFORE_GetBinLast() != this->BEFORE_GetBinFirst();
   }



   INLINE_ALWAYS const TreeNode<bClassification> * AFTER_GetTreeNodeChildren() const {
      EBM_ASSERT(m_bDoneGainCalc);
      return m_UNION.m_afterGainCalc.m_pTreeNodeChildren;
   }
   INLINE_ALWAYS TreeNode<bClassification> * AFTER_GetTreeNodeChildren() {
      EBM_ASSERT(m_bDoneGainCalc);
      return m_UNION.m_afterGainCalc.m_pTreeNodeChildren;
   }
   INLINE_ALWAYS void AFTER_SetTreeNodeChildren(TreeNode<bClassification> * const pTreeNodeChildren) {
      EBM_ASSERT(m_bDoneGainCalc);
      m_UNION.m_afterGainCalc.m_pTreeNodeChildren = pTreeNodeChildren;
   }

   INLINE_ALWAYS ActiveDataType AFTER_GetSplitVal() const {
      EBM_ASSERT(m_bDoneGainCalc);
      return m_UNION.m_afterGainCalc.m_splitVal;
   }
   INLINE_ALWAYS void AFTER_SetSplitVal(const ActiveDataType splitVal) {
      EBM_ASSERT(m_bDoneGainCalc);
      m_UNION.m_afterGainCalc.m_splitVal = splitVal;
   }

   INLINE_ALWAYS FloatBig AFTER_GetSplitGain() const {
      EBM_ASSERT(m_bDoneGainCalc);
      EBM_ASSERT(!m_bSplitDecided);

      const FloatBig splitGain = m_UNION.m_afterGainCalc.m_splitGain;

      // our priority queue cannot handle NaN values so we filter them out before adding them
      EBM_ASSERT(!std::isnan(splitGain));
      EBM_ASSERT(!std::isinf(splitGain));
      EBM_ASSERT(0 <= splitGain);

      return splitGain;
   }
   INLINE_ALWAYS void AFTER_SetSplitGain(const FloatBig splitGain) {
      EBM_ASSERT(m_bDoneGainCalc);
      EBM_ASSERT(!m_bSplitDecided);

      // our priority queue cannot handle NaN values so we filter them out before adding them
      EBM_ASSERT(!std::isnan(splitGain));
      EBM_ASSERT(!std::isinf(splitGain));
      EBM_ASSERT(0 <= splitGain);

      m_UNION.m_afterGainCalc.m_splitGain = splitGain;
   }

   INLINE_ALWAYS void AFTER_RejectSplit() {
      EBM_ASSERT(m_bDoneGainCalc);
      EBM_ASSERT(!m_bSplitDecided);

#ifndef NDEBUG
      m_bSplitDecided = true;
#endif // NDEBUG

      // we aren't going to split this TreeNode because we can't.  We need to set the splitGain value here because 
      // otherwise it is filled with garbage that could be NaN (meaning the node was a branch) we can't call 
      // AFTER_RejectSplit before calling SplitTreeNode because 
      // AFTER_RejectSplit sets 
      // m_UNION.m_afterGainCalc.m_splitGain and the 
      // m_UNION.m_beforeGainCalc values are needed if we had decided to call 
      // FindBestSplitGain
      m_UNION.m_afterGainCalc.m_splitGain = 0;
   }

   INLINE_ALWAYS void AFTER_SplitNode() {
      EBM_ASSERT(m_bDoneGainCalc);
      EBM_ASSERT(!m_bSplitDecided);

#ifndef NDEBUG
      m_bSplitDecided = true;
#endif // NDEBUG

      m_UNION.m_afterGainCalc.m_splitGain = k_illegalGainFloat;
   }

   INLINE_ALWAYS bool AFTER_IsSplit() const {
      EBM_ASSERT(m_bDoneGainCalc);
      EBM_ASSERT(m_bSplitDecided);
      return k_illegalGainFloat == m_UNION.m_afterGainCalc.m_splitGain;
   }


   INLINE_ALWAYS size_t GetCountSamples() const {
      return m_bin.GetCountSamples();
   }

   INLINE_ALWAYS FloatBig GetWeight() const {
      return m_bin.GetWeight();
   }

   INLINE_ALWAYS const GradientPair<FloatBig, bClassification> * GetGradientPairs() const {
      return m_bin.GetGradientPairs();
   }

   INLINE_ALWAYS const Bin<FloatBig, bClassification> * GetBin() const {
      return &m_bin;
   }
   INLINE_ALWAYS Bin<FloatBig, bClassification> * GetBin() {
      return &m_bin;
   }

#ifndef NDEBUG
   INLINE_ALWAYS void SetDoneGainCalc(const bool bDoneGainCalc) {
      if(bDoneGainCalc) {
         // we set this to false when it's random memory, 
         // but we only flip it to true from an initialized state of false
         EBM_ASSERT(!m_bDoneGainCalc);
         EBM_ASSERT(!m_bSplitDecided);
      }
      m_bDoneGainCalc = bDoneGainCalc;
      m_bSplitDecided = false;
   }
#endif // NDEBUG

private:

   struct BeforeGainCalc final {
      BeforeGainCalc() = default; // preserve our POD status
      ~BeforeGainCalc() = default; // preserve our POD status
      void * operator new(std::size_t) = delete; // we only use malloc/free in this library
      void operator delete (void *) = delete; // we only use malloc/free in this library

      const Bin<FloatBig, bClassification> * m_pBinFirst;
      const Bin<FloatBig, bClassification> * m_pBinLast;
   };
   static_assert(std::is_standard_layout<BeforeGainCalc>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
   static_assert(std::is_trivial<BeforeGainCalc>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");
   static_assert(std::is_pod<BeforeGainCalc>::value,
      "We use a lot of C constructs, so disallow non-POD types in general");

   struct AfterGainCalc final {
      AfterGainCalc() = default; // preserve our POD status
      ~AfterGainCalc() = default; // preserve our POD status
      void * operator new(std::size_t) = delete; // we only use malloc/free in this library
      void operator delete (void *) = delete; // we only use malloc/free in this library

      TreeNode<bClassification> * m_pTreeNodeChildren;
      ActiveDataType m_splitVal; // TODO: make this a size_t for here since it doesn't need cross-platform in the TreeNode
      FloatBig m_splitGain;
   };
   static_assert(std::is_standard_layout<AfterGainCalc>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
   static_assert(std::is_trivial<AfterGainCalc>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");
   static_assert(std::is_pod<AfterGainCalc>::value,
      "We use a lot of C constructs, so disallow non-POD types in general");

   union TreeNodeUnion final {

#ifndef __SUNPRO_CC

      // the Oracle Developer Studio compiler has what I think is a bug by making any class that includes 
      // TreeNodeUnion fields turn into non-trivial classes, so exclude the Oracle compiler
      // from these protections

      TreeNodeUnion() = default; // preserve our POD status
      ~TreeNodeUnion() = default; // preserve our POD status
      void * operator new(std::size_t) = delete; // we only use malloc/free in this library
      void operator delete (void *) = delete; // we only use malloc/free in this library

#endif // __SUNPRO_CC

      // we can save precious L1 cache space by keeping only what we need
      BeforeGainCalc m_beforeGainCalc;
      AfterGainCalc m_afterGainCalc;
   };
   static_assert(std::is_standard_layout<TreeNodeUnion>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
   static_assert(std::is_trivial<TreeNodeUnion>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");
   static_assert(std::is_pod<TreeNodeUnion>::value,
      "We use a lot of C constructs, so disallow non-POD types in general");

#ifndef NDEBUG
   bool m_bDoneGainCalc;
   bool m_bSplitDecided;
#endif // NDEBUG

   TreeNodeUnion m_UNION;

   // use the "struct hack" since Flexible array member method is not available in C++
   // m_bin must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived

   Bin<FloatBig, bClassification> m_bin;
};
static_assert(std::is_standard_layout<TreeNode<true>>::value && std::is_standard_layout<TreeNode<false>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<TreeNode<true>>::value && std::is_trivial<TreeNode<false>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<TreeNode<true>>::value && std::is_pod<TreeNode<false>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

INLINE_ALWAYS bool IsOverflowTreeNodeSize(const bool bClassification, const size_t cScores) {
   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // check this before calling us
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

   size_t cBytesTreeNodeComponent;
   if(bClassification) {
      cBytesTreeNodeComponent = sizeof(TreeNode<true>) - sizeof(TreeNode<true>::m_bin);
   } else {
      cBytesTreeNodeComponent = sizeof(TreeNode<false>) - sizeof(TreeNode<false>::m_bin);
   }

   if(UNLIKELY(IsAddError(cBytesTreeNodeComponent, cBytesPerBin))) {
      return true;
   }

   return false;
}

INLINE_ALWAYS size_t GetTreeNodeSize(const bool bClassification, const size_t cScores) {
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

   size_t cBytesTreeNodeComponent;
   if(bClassification) {
      cBytesTreeNodeComponent = sizeof(TreeNode<true>) - sizeof(TreeNode<true>::m_bin);
   } else {
      cBytesTreeNodeComponent = sizeof(TreeNode<false>) - sizeof(TreeNode<false>::m_bin);
   }

   return cBytesTreeNodeComponent + cBytesPerBin;
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
