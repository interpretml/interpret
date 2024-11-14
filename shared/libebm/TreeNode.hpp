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

static bool IsOverflowTreeNodeMultiSize(const bool bHessian, const size_t cScores);
static size_t GetTreeNodeMultiSize(const bool bHessian, const size_t cScores);

template<bool bHessian, size_t cCompilerScores = 1> struct TreeNode final {
   friend bool IsOverflowTreeNodeSize(const bool, const size_t);
   friend size_t GetTreeNodeSize(const bool, const size_t);

   TreeNode() = default; // preserve our POD status
   ~TreeNode() = default; // preserve our POD status
   void* operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete(void*) = delete; // we only use malloc/free in this library

   inline const Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores>* BEFORE_GetBinFirst() const {
      EBM_ASSERT(0 == m_debugProgressionStage);
      return m_UNION.m_beforeGainCalc.m_pBinFirst;
   }
   inline void BEFORE_SetBinFirst(
         const Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores>* const pBinFirst) {
      EBM_ASSERT(0 == m_debugProgressionStage);
      m_UNION.m_beforeGainCalc.m_pBinFirst = pBinFirst;
   }

   inline const Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores>* BEFORE_GetBinLast() const {
      EBM_ASSERT(0 == m_debugProgressionStage);
      return reinterpret_cast<const Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores>*>(
            pBinLastOrChildren);
   }
   inline void BEFORE_SetBinLast(
         const Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores>* const pBinLast) {
      EBM_ASSERT(0 == m_debugProgressionStage);
      // we aren't going to modify pBinLast, but we're storing it in a shared pointer, so remove the const for now
      pBinLastOrChildren = const_cast<void*>(static_cast<const void*>(pBinLast));
   }

   inline bool BEFORE_IsSplittable() const {
      EBM_ASSERT(0 == m_debugProgressionStage);
      return this->BEFORE_GetBinLast() != this->BEFORE_GetBinFirst();
   }

   inline const void* DANGEROUS_GetBinLastOrChildren() const {
      EBM_ASSERT(1 == m_debugProgressionStage);
      return pBinLastOrChildren;
   }

   inline TreeNode* AFTER_GetChildren() {
      EBM_ASSERT(1 == m_debugProgressionStage || 2 == m_debugProgressionStage);
      return reinterpret_cast<TreeNode*>(pBinLastOrChildren);
   }
   inline void AFTER_SetChildren(TreeNode* const pChildren) {
      EBM_ASSERT(1 == m_debugProgressionStage || 2 == m_debugProgressionStage);
      pBinLastOrChildren = pChildren;
   }

   inline FloatMain AFTER_GetSplitGain() const {
      EBM_ASSERT(1 == m_debugProgressionStage);

      const FloatMain splitGain = m_UNION.m_afterGainCalc.m_splitGain;

      // our priority queue cannot handle NaN values so we filter them out before adding them
      EBM_ASSERT(!std::isnan(splitGain));
      EBM_ASSERT(!std::isinf(splitGain));
      EBM_ASSERT(FloatMain{0} <= splitGain);

      return splitGain;
   }
   inline void AFTER_SetSplitGain(const FloatMain splitGain) {
      EBM_ASSERT(1 == m_debugProgressionStage);

      // this is only called if there is a legal gain value. If the TreeNode cannot be split call AFTER_RejectSplit.

      // our priority queue cannot handle NaN values so we filter them out before adding them
      EBM_ASSERT(!std::isnan(splitGain));
      EBM_ASSERT(!std::isinf(splitGain));
      EBM_ASSERT(FloatMain{0} <= splitGain);

      m_UNION.m_afterGainCalc.m_splitGain = splitGain;
   }

   inline void AFTER_RejectSplit() {
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

      m_UNION.m_afterGainCalc.m_splitGain = FloatMain{0};
   }

   inline void AFTER_SplitNode() {
      EBM_ASSERT(1 == m_debugProgressionStage);
      m_UNION.m_afterGainCalc.m_splitGain = std::numeric_limits<FloatMain>::quiet_NaN();
   }

   inline bool AFTER_IsSplit() const {
      EBM_ASSERT(1 == m_debugProgressionStage);
      return std::isnan(m_UNION.m_afterGainCalc.m_splitGain);
   }

   inline TreeNode* DECONSTRUCT_GetParent() {
      EBM_ASSERT(2 == m_debugProgressionStage);
      return m_UNION.m_deconstruct.m_pParent;
   }
   inline void DECONSTRUCT_SetParent(TreeNode* const pParent) {
      EBM_ASSERT(2 == m_debugProgressionStage);
      m_UNION.m_deconstruct.m_pParent = pParent;
   }

   inline Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores>* GetBin() { return &m_bin; }

   template<size_t cNewCompilerScores> inline TreeNode<bHessian, cNewCompilerScores>* Upgrade() {
      return reinterpret_cast<TreeNode<bHessian, cNewCompilerScores>*>(this);
   }
   inline TreeNode<bHessian, 1>* Downgrade() { return reinterpret_cast<TreeNode<bHessian, 1>*>(this); }

#ifndef NDEBUG
   inline void SetDebugProgression(const int stage) {
      EBM_ASSERT(0 == stage || m_debugProgressionStage < stage); // always progress after initialization
      m_debugProgressionStage = stage;
   }
#endif // NDEBUG

 private:
   struct BeforeGainCalc final {
      const Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores>* m_pBinFirst;
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
   int m_debugProgressionStage;
#endif // NDEBUG

   void* pBinLastOrChildren;
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

template<bool bHessian, size_t cCompilerScores = 1> struct TreeNodeMulti final {
   friend bool IsOverflowTreeNodeMultiSize(const bool, const size_t);
   friend size_t GetTreeNodeMultiSize(const bool, const size_t);

   TreeNodeMulti() = default; // preserve our POD status
   ~TreeNodeMulti() = default; // preserve our POD status
   void* operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete(void*) = delete; // we only use malloc/free in this library

   inline void SetSplitGain(const FloatMain splitGain) {
      // our priority queue cannot handle NaN values so we filter them out before adding them
      EBM_ASSERT(!std::isnan(splitGain));
      EBM_ASSERT(!std::isinf(splitGain));
      EBM_ASSERT(FloatMain{0} <= splitGain);

      m_splitGain = splitGain;
   }
   inline bool IsSplit() const { return std::isnan(m_splitGain); }
   inline FloatMain GetSplitGain() const {
      // our priority queue cannot handle NaN values so we filter them out before adding them
      EBM_ASSERT(!std::isnan(m_splitGain));
      EBM_ASSERT(!std::isinf(m_splitGain));
      EBM_ASSERT(FloatMain{0} <= m_splitGain);

      return m_splitGain;
   }
   inline void SplitNode() { m_splitGain = std::numeric_limits<FloatMain>::quiet_NaN(); }

   inline void SetDimensionIndex(const size_t iDimension) { m_iDimension = iDimension; }
   inline size_t GetDimensionIndex() const { return m_iDimension; }

   inline void SetSplitIndex(const size_t iSplit) { m_iSplit = iSplit; }
   inline size_t GetSplitIndex() const { return m_iSplit; }

   inline void SetParent(TreeNodeMulti* const pParent) { m_pParent = pParent; }
   inline TreeNodeMulti* GetParent() { return m_pParent; }

   inline void SetChildren(TreeNodeMulti* const pChildren) { m_pChildren = pChildren; }
   inline TreeNodeMulti* GetChildren() { return m_pChildren; }

   inline Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores>* GetBin() { return &m_bin; }

   template<size_t cNewCompilerScores> inline TreeNodeMulti<bHessian, cNewCompilerScores>* Upgrade() {
      return reinterpret_cast<TreeNodeMulti<bHessian, cNewCompilerScores>*>(this);
   }
   inline TreeNodeMulti<bHessian, 1>* Downgrade() { return reinterpret_cast<TreeNodeMulti<bHessian, 1>*>(this); }

 private:
   FloatMain m_splitGain;
   size_t m_iDimension;
   size_t m_iSplit;
   TreeNodeMulti* m_pParent;
   TreeNodeMulti* m_pChildren;

   // IMPORTANT: m_bin must be in the last position for the struct hack and this must be standard layout
   Bin<FloatMain, UIntMain, true, true, bHessian, cCompilerScores> m_bin;
};
static_assert(
      std::is_standard_layout<TreeNodeMulti<true>>::value && std::is_standard_layout<TreeNodeMulti<false>>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<TreeNodeMulti<true>>::value && std::is_trivial<TreeNodeMulti<false>>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<TreeNodeMulti<true>>::value && std::is_pod<TreeNodeMulti<false>>::value,
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

inline static bool IsOverflowTreeNodeMultiSize(const bool bHessian, const size_t cScores) {
   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);

   size_t cBytesTreeNodeMultiComponent;
   if(bHessian) {
      typedef TreeNodeMulti<true> OffsetType;
      cBytesTreeNodeMultiComponent = offsetof(OffsetType, m_bin);
   } else {
      typedef TreeNodeMulti<false> OffsetType;
      cBytesTreeNodeMultiComponent = offsetof(OffsetType, m_bin);
   }

   if(UNLIKELY(IsAddError(cBytesTreeNodeMultiComponent, cBytesPerBin))) {
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

inline static size_t GetTreeNodeMultiSize(const bool bHessian, const size_t cScores) {
   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);

   size_t cBytesTreeNodeMultiComponent;
   if(bHessian) {
      typedef TreeNodeMulti<true> OffsetType;
      cBytesTreeNodeMultiComponent = offsetof(OffsetType, m_bin);
   } else {
      typedef TreeNodeMulti<false> OffsetType;
      cBytesTreeNodeMultiComponent = offsetof(OffsetType, m_bin);
   }

   return cBytesTreeNodeMultiComponent + cBytesPerBin;
}

template<bool bHessian, size_t cCompilerScores>
inline static TreeNode<bHessian, cCompilerScores>* IndexTreeNode(
      TreeNode<bHessian, cCompilerScores>* const pTreeNode, const size_t iByte) {
   return IndexByte(pTreeNode, iByte);
}

template<bool bHessian, size_t cCompilerScores>
inline static TreeNodeMulti<bHessian, cCompilerScores>* IndexTreeNodeMulti(
      TreeNodeMulti<bHessian, cCompilerScores>* const pTreeNodeMulti, const size_t iByte) {
   return IndexByte(pTreeNodeMulti, iByte);
}

template<bool bHessian, size_t cCompilerScores>
inline static TreeNode<bHessian, cCompilerScores>* GetLeftNode(TreeNode<bHessian, cCompilerScores>* const pChildren) {
   return pChildren;
}

template<bool bHessian, size_t cCompilerScores>
inline static TreeNodeMulti<bHessian, cCompilerScores>* GetLowNode(
      TreeNodeMulti<bHessian, cCompilerScores>* const pChildren, const size_t cBytesPerTreeNodeMulti) {
   return IndexTreeNodeMulti(pChildren, cBytesPerTreeNodeMulti);
}

template<bool bHessian, size_t cCompilerScores>
inline static TreeNode<bHessian, cCompilerScores>* GetRightNode(
      TreeNode<bHessian, cCompilerScores>* const pChildren, const size_t cBytesPerTreeNode) {
   return IndexTreeNode(pChildren, cBytesPerTreeNode);
}

template<bool bHessian, size_t cCompilerScores>
inline static TreeNodeMulti<bHessian, cCompilerScores>* GetHighNode(
      TreeNodeMulti<bHessian, cCompilerScores>* const pChildren) {
   return pChildren;
}

} // namespace DEFINED_ZONE_NAME

#endif // TREE_NODE_HPP
