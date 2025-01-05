// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TREE_NODE_MULTI_HPP
#define TREE_NODE_MULTI_HPP

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

static bool IsOverflowTreeNodeMultiSize(const bool bHessian, const size_t cScores);
static size_t GetTreeNodeMultiSize(const bool bHessian, const size_t cScores);

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
inline static TreeNodeMulti<bHessian, cCompilerScores>* IndexTreeNodeMulti(
      TreeNodeMulti<bHessian, cCompilerScores>* const pTreeNodeMulti, const size_t iByte) {
   return IndexByte(pTreeNodeMulti, iByte);
}

template<bool bHessian, size_t cCompilerScores>
inline static TreeNodeMulti<bHessian, cCompilerScores>* GetLowNode(
      TreeNodeMulti<bHessian, cCompilerScores>* const pChildren, const size_t cBytesPerTreeNodeMulti) {
   return IndexTreeNodeMulti(pChildren, cBytesPerTreeNodeMulti);
}

template<bool bHessian, size_t cCompilerScores>
inline static TreeNodeMulti<bHessian, cCompilerScores>* GetHighNode(
      TreeNodeMulti<bHessian, cCompilerScores>* const pChildren) {
   return pChildren;
}

} // namespace DEFINED_ZONE_NAME

#endif // TREE_NODE_MULTI_HPP
