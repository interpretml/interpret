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
      EBM_ASSERT(0 == m_debugProgressionStage);
      return m_UNION.m_beforeGainCalc.m_pBinFirst;
   }
   INLINE_ALWAYS void BEFORE_SetBinFirst(const Bin<FloatBig, bClassification> * const pBinFirst) {
      EBM_ASSERT(0 == m_debugProgressionStage);
      m_UNION.m_beforeGainCalc.m_pBinFirst = pBinFirst;
   }

   INLINE_ALWAYS const Bin<FloatBig, bClassification> * BEFORE_GetBinLast() const {
      EBM_ASSERT(0 == m_debugProgressionStage);
      return reinterpret_cast<const Bin<FloatBig, bClassification> *>(pBinLastOrChildren);
   }
   INLINE_ALWAYS void BEFORE_SetBinLast(const Bin<FloatBig, bClassification> * const pBinLast) {
      EBM_ASSERT(0 == m_debugProgressionStage);
      // we aren't going to modify pBinLast, but we're storing it in a shared pointer, so remove the const for now
      pBinLastOrChildren = const_cast<void *>(static_cast<const void *>(pBinLast));
   }

   INLINE_ALWAYS bool BEFORE_IsSplittable() const {
      EBM_ASSERT(0 == m_debugProgressionStage);
      return this->BEFORE_GetBinLast() != this->BEFORE_GetBinFirst();
   }


   INLINE_ALWAYS const void * DANGEROUS_GetBinLastOrChildren() const {
      EBM_ASSERT(1 == m_debugProgressionStage);
      return pBinLastOrChildren;
   }


   INLINE_ALWAYS const TreeNode<bClassification> * AFTER_GetChildren() const {
      EBM_ASSERT(1 == m_debugProgressionStage || 2 == m_debugProgressionStage);
      return reinterpret_cast<const TreeNode<bClassification> *>(pBinLastOrChildren);
   }
   INLINE_ALWAYS TreeNode<bClassification> * AFTER_GetChildren() {
      EBM_ASSERT(1 == m_debugProgressionStage || 2 == m_debugProgressionStage);
      return reinterpret_cast<TreeNode<bClassification> *>(pBinLastOrChildren);
   }
   INLINE_ALWAYS void AFTER_SetChildren(TreeNode<bClassification> * const pChildren) {
      EBM_ASSERT(1 == m_debugProgressionStage || 2 == m_debugProgressionStage);
      pBinLastOrChildren = pChildren;
   }


   INLINE_ALWAYS FloatBig AFTER_GetSplitGain() const {
      EBM_ASSERT(1 == m_debugProgressionStage);

      const FloatBig splitGain = m_UNION.m_afterGainCalc.m_splitGain;

      // our priority queue cannot handle NaN values so we filter them out before adding them
      EBM_ASSERT(!std::isnan(splitGain));
      EBM_ASSERT(!std::isinf(splitGain));
      EBM_ASSERT(0 <= splitGain);

      return splitGain;
   }
   INLINE_ALWAYS void AFTER_SetSplitGain(const FloatBig splitGain) {
      EBM_ASSERT(1 == m_debugProgressionStage);

      // this is only called if there is a legal gain value. If the TreeNode cannot be split call AFTER_RejectSplit.

      // our priority queue cannot handle NaN values so we filter them out before adding them
      EBM_ASSERT(!std::isnan(splitGain));
      EBM_ASSERT(!std::isinf(splitGain));
      EBM_ASSERT(0 <= splitGain);

      m_UNION.m_afterGainCalc.m_splitGain = splitGain;
   }

   INLINE_ALWAYS void AFTER_RejectSplit() {
      EBM_ASSERT(1 == m_debugProgressionStage);

      // This TreeNode could not be split, so it won't be added to the priority queue, and it does not have a gain.
      // 
      // If the TreeNode could have been split, then we would have set the TreeNode::m_afterGainCalc::m_splitGain
      // value, and put the TreeNode onto the priority queue to decide later if it has a high enough gain to split.
      // If a TreeNode was subsequently the highest in the priority queue, then it would then be split, and the 
      // m_splitGain value would be set to NaN to indicate that the TreeNode was split. 
      //
      // Since this function has been called, it was determined this TreeNode could not be split, but the m_splitGain
      // value has not been set yet, and m_UNION is currently in the BeforeGainCalc state, so m_splitGain is
      // filled with random garbage.
      // 
      // We need to set the m_splitGain value then to something other than NaN to indicate that it was not split.

      m_UNION.m_afterGainCalc.m_splitGain = 0;
   }

   INLINE_ALWAYS void AFTER_SplitNode() {
      EBM_ASSERT(1 == m_debugProgressionStage);
      m_UNION.m_afterGainCalc.m_splitGain = std::numeric_limits<FloatBig>::quiet_NaN();
   }

   INLINE_ALWAYS bool AFTER_IsSplit() const {
      EBM_ASSERT(1 == m_debugProgressionStage);
      return std::isnan(m_UNION.m_afterGainCalc.m_splitGain);
   }


   INLINE_ALWAYS TreeNode<bClassification> * DECONSTRUCT_GetParent() {
      EBM_ASSERT(2 == m_debugProgressionStage);
      return m_UNION.m_deconstruct.m_pParent;
   }
   INLINE_ALWAYS void DECONSTRUCT_SetParent(TreeNode<bClassification> * const pParent) {
      EBM_ASSERT(2 == m_debugProgressionStage);
      m_UNION.m_deconstruct.m_pParent = pParent;
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
   INLINE_ALWAYS void SetDebugProgression(const int stage) {
      EBM_ASSERT(0 == stage || m_debugProgressionStage < stage); // always progress after initialization
      m_debugProgressionStage = stage;
   }
#endif // NDEBUG

private:

   struct BeforeGainCalc final {
      const Bin<FloatBig, bClassification> * m_pBinFirst;
   };

   struct AfterGainCalc final {
      FloatBig m_splitGain;
   };

   struct Deconstruct final {
      TreeNode<bClassification> * m_pParent;
   };

   union TreeNodeUnion final {
      BeforeGainCalc m_beforeGainCalc;
      AfterGainCalc m_afterGainCalc;
      Deconstruct m_deconstruct;
   };

#ifndef NDEBUG
   int m_debugProgressionStage;
#endif // NDEBUG

   void * pBinLastOrChildren;
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
INLINE_ALWAYS TreeNode<bClassification> * IndexTreeNode(
   TreeNode<bClassification> * const pTreeNode, 
   const size_t iByte
) {
   return reinterpret_cast<TreeNode<bClassification> *>(reinterpret_cast<char *>(pTreeNode) + iByte);
}

template<bool bClassification>
INLINE_ALWAYS const TreeNode<bClassification> * IndexTreeNode(
   const TreeNode<bClassification> * const pTreeNode, 
   const size_t iByte
) {
   return reinterpret_cast<const TreeNode<bClassification> *>(reinterpret_cast<const char *>(pTreeNode) + iByte);
}

template<bool bClassification>
INLINE_ALWAYS TreeNode<bClassification> * GetLeftNode(
   TreeNode<bClassification> * const pChildren, 
   const size_t cBytesPerTreeNode
) {
   UNUSED(cBytesPerTreeNode);
   return pChildren;
}

template<bool bClassification>
INLINE_ALWAYS const TreeNode<bClassification> * GetLeftNode(
   const TreeNode<bClassification> * const pChildren, 
   const size_t cBytesPerTreeNode
) {
   UNUSED(cBytesPerTreeNode);
   return pChildren;
}

template<bool bClassification>
INLINE_ALWAYS TreeNode<bClassification> * GetRightNode(
   TreeNode<bClassification> * const pChildren, 
   const size_t cBytesPerTreeNode
) {
   return IndexTreeNode<bClassification>(pChildren, cBytesPerTreeNode);
}

template<bool bClassification>
INLINE_ALWAYS const TreeNode<bClassification> * GetRightNode(
   const TreeNode<bClassification> * const pChildren, 
   const size_t cBytesPerTreeNode
) {
   return IndexTreeNode<bClassification>(pChildren, cBytesPerTreeNode);
}

} // DEFINED_ZONE_NAME

#endif // TREE_NODE_HPP
