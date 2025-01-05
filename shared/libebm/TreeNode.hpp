// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TREE_NODE_HPP
#define TREE_NODE_HPP

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT
#include "unzoned.h" // FloatMain

#include "common.hpp" // IsAddError
#include "Bin.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

static bool IsOverflowTreeNodeSize(const bool bHessian, const size_t cScores);
static size_t GetTreeNodeSize(const bool bHessian, const size_t cScores);

#ifndef NDEBUG
enum class DebugStage {
   Initialized = 1,
   SetFirstOrLastBin = 2,
   SetFirstAndLastBin = 3,
   QueuingRejected = 4,
   Queued = 5,
   Split = 6,
   DestructTraversedLeft = 7,
   DestructTraversedRight = 8,
   DONE = 99
};
#endif // NDEBUG

template<bool bHessian, size_t cCompilerScores = 1> struct TreeNode final {
   friend bool IsOverflowTreeNodeSize(const bool, const size_t);
   friend size_t GetTreeNodeSize(const bool, const size_t);

   TreeNode() = default; // preserve our POD status
   ~TreeNode() = default; // preserve our POD status
   void* operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete(void*) = delete; // we only use malloc/free in this library

   inline void Init() {
#ifndef NDEBUG
      m_debugProgressionStage = DebugStage::Initialized;
#endif // NDEBUG
   }

   inline void BEFORE_SetBinFirst(
         const Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores>* const* const pBinFirst) {
#ifndef NDEBUG
      EBM_ASSERT(nullptr != pBinFirst);
      if(DebugStage::Initialized == m_debugProgressionStage) {
         m_debugProgressionStage = DebugStage::SetFirstOrLastBin;
      } else {
         EBM_ASSERT(DebugStage::SetFirstOrLastBin == m_debugProgressionStage);
         m_debugProgressionStage = DebugStage::SetFirstAndLastBin;
      }
#endif // NDEBUG
      m_UNION.m_beforeGainCalc.m_pBinFirst = pBinFirst;
   }

   inline void BEFORE_SetBinLast(
         const Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores>* const* const pBinLast) {
#ifndef NDEBUG
      EBM_ASSERT(nullptr != pBinLast);
      if(DebugStage::Initialized == m_debugProgressionStage) {
         m_debugProgressionStage = DebugStage::SetFirstOrLastBin;
      } else {
         EBM_ASSERT(DebugStage::SetFirstOrLastBin == m_debugProgressionStage);
         m_debugProgressionStage = DebugStage::SetFirstAndLastBin;
      }
#endif // NDEBUG
      // we aren't going to modify pBinLast, but we're storing it in a shared pointer, so remove the const for now
      pPointerBinLastOrChildren = const_cast<void*>(static_cast<const void*>(pBinLast));
   }

   inline const Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores>* const* BEFORE_GetBinFirst() const {
      EBM_ASSERT(DebugStage::SetFirstAndLastBin == m_debugProgressionStage);
      return m_UNION.m_beforeGainCalc.m_pBinFirst;
   }

   inline const Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores>* const* BEFORE_GetBinLast() const {
      EBM_ASSERT(DebugStage::SetFirstAndLastBin == m_debugProgressionStage ||
            DebugStage::QueuingRejected == m_debugProgressionStage);
      return reinterpret_cast<const Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores>* const*>(
            pPointerBinLastOrChildren);
   }

   inline void AFTER_RejectSplit() {
#ifndef NDEBUG
      EBM_ASSERT(DebugStage::SetFirstAndLastBin == m_debugProgressionStage);
      m_debugProgressionStage = DebugStage::QueuingRejected;
#endif // NDEBUG

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

      m_UNION.m_afterGainCalc.m_splitGain = FloatMain{0};
   }

   inline void AFTER_SetSplitGain(const FloatMain splitGain, TreeNode* const pChildren) {
#ifndef NDEBUG
      // our priority queue cannot handle NaN values so we filter them out before adding them
      EBM_ASSERT(!std::isnan(splitGain));
      EBM_ASSERT(!std::isinf(splitGain));
      EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= splitGain);
      EBM_ASSERT(nullptr != pChildren);

      EBM_ASSERT(DebugStage::SetFirstAndLastBin == m_debugProgressionStage);
      m_debugProgressionStage = DebugStage::Queued;
#endif // NDEBUG

      m_UNION.m_afterGainCalc.m_splitGain = splitGain;
      pPointerBinLastOrChildren = pChildren;
   }

   inline FloatMain AFTER_GetSplitGain() const {
      EBM_ASSERT(DebugStage::Queued == m_debugProgressionStage);

      const FloatMain splitGain = m_UNION.m_afterGainCalc.m_splitGain;

      // our priority queue cannot handle NaN values so we filter them out before adding them
      EBM_ASSERT(!std::isnan(splitGain));
      EBM_ASSERT(!std::isinf(splitGain));
      EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= splitGain);

      return splitGain;
   }

   inline TreeNode* AFTER_GetChildren() {
      EBM_ASSERT(DebugStage::Queued == m_debugProgressionStage);
      return reinterpret_cast<TreeNode*>(pPointerBinLastOrChildren);
   }

   inline bool AFTER_IsSplit() const {
      EBM_ASSERT(DebugStage::QueuingRejected == m_debugProgressionStage ||
            DebugStage::Queued == m_debugProgressionStage || DebugStage::Split == m_debugProgressionStage);
      return std::isnan(m_UNION.m_afterGainCalc.m_splitGain);
   }

   inline bool AFTER_IsSplittable() const {
      EBM_ASSERT(
            DebugStage::QueuingRejected == m_debugProgressionStage || DebugStage::Queued == m_debugProgressionStage);
      return 0.0 != m_UNION.m_afterGainCalc.m_splitGain;
   }

   inline void AFTER_SplitNode() {
#ifndef NDEBUG
      EBM_ASSERT(DebugStage::Queued == m_debugProgressionStage);
      m_debugProgressionStage = DebugStage::Split;
#endif // NDEBUG
      m_UNION.m_afterGainCalc.m_splitGain = std::numeric_limits<FloatMain>::quiet_NaN();
   }

   inline TreeNode* DECONSTRUCT_TraverseLeftAndMark(TreeNode* const pParent) {
#ifndef NDEBUG
      EBM_ASSERT(DebugStage::Split == m_debugProgressionStage);
      m_debugProgressionStage = DebugStage::DestructTraversedLeft;
#endif // NDEBUG
      m_UNION.m_deconstruct.m_pParent = pParent;
      // return the left child
      return GetLeftNode(reinterpret_cast<TreeNode*>(pPointerBinLastOrChildren));
   }

   inline bool DECONSTRUCT_IsRightChildTraversal() {
      EBM_ASSERT(DebugStage::DestructTraversedLeft == m_debugProgressionStage ||
            DebugStage::DestructTraversedRight == m_debugProgressionStage);
      return nullptr == pPointerBinLastOrChildren;
   }
   inline TreeNode* DECONSTRUCT_TraverseRightAndMark(const size_t cBytesPerTreeNode) {
#ifndef NDEBUG
      EBM_ASSERT(DebugStage::DestructTraversedLeft == m_debugProgressionStage);
      m_debugProgressionStage = DebugStage::DestructTraversedRight;
#endif // NDEBUG

      TreeNode* const pRightChild =
            GetRightNode(reinterpret_cast<TreeNode*>(pPointerBinLastOrChildren), cBytesPerTreeNode);
      pPointerBinLastOrChildren = nullptr;
      return pRightChild;
   }

   inline TreeNode* DECONSTRUCT_GetParent() {
#ifndef NDEBUG
      EBM_ASSERT(DebugStage::DestructTraversedRight == m_debugProgressionStage);
      m_debugProgressionStage = DebugStage::DONE;
#endif // NDEBUG
      return m_UNION.m_deconstruct.m_pParent;
   }

   inline Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores>* GetBin() { return &m_bin; }
   inline const Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores>* GetBin() const { return &m_bin; }

   template<size_t cNewCompilerScores> inline TreeNode<bHessian, cNewCompilerScores>* Upgrade() {
      return reinterpret_cast<TreeNode<bHessian, cNewCompilerScores>*>(this);
   }
   inline TreeNode<bHessian, 1>* Downgrade() { return reinterpret_cast<TreeNode<bHessian, 1>*>(this); }
   inline const TreeNode<bHessian, 1>* Downgrade() const {
      return reinterpret_cast<const TreeNode<bHessian, 1>*>(this);
   }

 private:
   struct BeforeGainCalc final {
      const Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores>* const* m_pBinFirst;
   };

   struct AfterGainCalc final {
      FloatMain m_splitGain;
   };

   struct Deconstruct final {
      TreeNode* m_pParent;
   };

   union TreeNodeUnion final {
      BeforeGainCalc m_beforeGainCalc;
      AfterGainCalc m_afterGainCalc;
      Deconstruct m_deconstruct;
   };

#ifndef NDEBUG
   DebugStage m_debugProgressionStage;
#endif // NDEBUG

   void* pPointerBinLastOrChildren;
   TreeNodeUnion m_UNION;

   // IMPORTANT: m_bin must be in the last position for the struct hack and this must be standard layout
   Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores> m_bin;
};
static_assert(std::is_standard_layout<TreeNode<true>>::value && std::is_standard_layout<TreeNode<false>>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<TreeNode<true>>::value && std::is_trivial<TreeNode<false>>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<TreeNode<true>>::value && std::is_pod<TreeNode<false>>::value,
      "We use a lot of C constructs, so disallow non-POD types in general");

inline static bool IsOverflowTreeNodeSize(const bool bHessian, const size_t cScores) {
   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);

   size_t cBytesTreeNodeComponent;
   if(bHessian) {
      typedef TreeNode<true> OffsetType;
      cBytesTreeNodeComponent = offsetof(OffsetType, m_bin);
   } else {
      typedef TreeNode<false> OffsetType;
      cBytesTreeNodeComponent = offsetof(OffsetType, m_bin);
   }

   if(UNLIKELY(IsAddError(cBytesTreeNodeComponent, cBytesPerBin))) {
      return true;
   }

   return false;
}

inline static size_t GetTreeNodeSize(const bool bHessian, const size_t cScores) {
   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);

   size_t cBytesTreeNodeComponent;
   if(bHessian) {
      typedef TreeNode<true> OffsetType;
      cBytesTreeNodeComponent = offsetof(OffsetType, m_bin);
   } else {
      typedef TreeNode<false> OffsetType;
      cBytesTreeNodeComponent = offsetof(OffsetType, m_bin);
   }

   return cBytesTreeNodeComponent + cBytesPerBin;
}

template<bool bHessian, size_t cCompilerScores>
inline static TreeNode<bHessian, cCompilerScores>* IndexTreeNode(
      TreeNode<bHessian, cCompilerScores>* const pTreeNode, const size_t iByte) {
   return IndexByte(pTreeNode, iByte);
}

template<bool bHessian, size_t cCompilerScores>
inline static TreeNode<bHessian, cCompilerScores>* GetLeftNode(TreeNode<bHessian, cCompilerScores>* const pChildren) {
   return pChildren;
}

template<bool bHessian, size_t cCompilerScores>
inline static TreeNode<bHessian, cCompilerScores>* GetRightNode(
      TreeNode<bHessian, cCompilerScores>* const pChildren, const size_t cBytesPerTreeNode) {
   return IndexTreeNode(pChildren, cBytesPerTreeNode);
}

} // namespace DEFINED_ZONE_NAME

#endif // TREE_NODE_HPP
