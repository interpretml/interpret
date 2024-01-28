// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef SPLIT_POSITION_HPP
#define SPLIT_POSITION_HPP

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

static bool IsOverflowSplitPositionSize(const bool bHessian, const size_t cScores);
static size_t GetSplitPositionSize(bool bHessian, const size_t cScores);

template<bool bHessian, size_t cCompilerScores = 1> struct SplitPosition final {
   friend bool IsOverflowSplitPositionSize(const bool, const size_t);
   friend size_t GetSplitPositionSize(const bool, const size_t);

 private:
   const Bin<FloatMain, UIntMain, bHessian, cCompilerScores>* m_pBinPosition;

   // IMPORTANT: m_leftSum must be in the last position for the struct hack and this must be standard layout
   Bin<FloatMain, UIntMain, bHessian, cCompilerScores> m_leftSum;

 public:
   SplitPosition() = default; // preserve our POD status
   ~SplitPosition() = default; // preserve our POD status
   void* operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete(void*) = delete; // we only use malloc/free in this library

   inline const Bin<FloatMain, UIntMain, bHessian, cCompilerScores>* GetBinPosition() const { return m_pBinPosition; }
   inline void SetBinPosition(const Bin<FloatMain, UIntMain, bHessian, cCompilerScores>* const pBinPosition) {
      m_pBinPosition = pBinPosition;
   }

   inline Bin<FloatMain, UIntMain, bHessian, cCompilerScores>* GetLeftSum() { return &m_leftSum; }
};
static_assert(
      std::is_standard_layout<SplitPosition<true>>::value && std::is_standard_layout<SplitPosition<false>>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<SplitPosition<true>>::value && std::is_trivial<SplitPosition<false>>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<SplitPosition<true>>::value && std::is_pod<SplitPosition<false>>::value,
      "We use a lot of C constructs, so disallow non-POD types in general");

inline static bool IsOverflowSplitPositionSize(const bool bHessian, const size_t cScores) {
   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(bHessian, cScores);

   size_t cBytesSplitPositionComponent;
   if(bHessian) {
      typedef SplitPosition<true> OffsetType;
      cBytesSplitPositionComponent = offsetof(OffsetType, m_leftSum);
   } else {
      typedef SplitPosition<false> OffsetType;
      cBytesSplitPositionComponent = offsetof(OffsetType, m_leftSum);
   }

   if(UNLIKELY(IsAddError(cBytesSplitPositionComponent, cBytesPerBin))) {
      return true;
   }

   return false;
}

inline static size_t GetSplitPositionSize(bool bHessian, const size_t cScores) {
   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(bHessian, cScores);

   size_t cBytesSplitPositionComponent;
   if(bHessian) {
      typedef SplitPosition<true> OffsetType;
      cBytesSplitPositionComponent = offsetof(OffsetType, m_leftSum);
   } else {
      typedef SplitPosition<false> OffsetType;
      cBytesSplitPositionComponent = offsetof(OffsetType, m_leftSum);
   }

   return cBytesSplitPositionComponent + cBytesPerBin;
}

template<bool bHessian, size_t cCompilerScores>
inline static SplitPosition<bHessian, cCompilerScores>* IndexSplitPosition(
      SplitPosition<bHessian, cCompilerScores>* const pSplitPosition, const size_t iByte) {
   return IndexByte(pSplitPosition, iByte);
}

template<bool bHessian, size_t cCompilerScores>
inline static size_t CountSplitPositions(const SplitPosition<bHessian, cCompilerScores>* const pSplitPositionHigh,
      const SplitPosition<bHessian, cCompilerScores>* const pSplitPositionLow,
      const size_t cBytesPerSplitPosition) {
   const size_t cBytesDiff = CountBytes(pSplitPositionHigh, pSplitPositionLow);
   EBM_ASSERT(0 == cBytesDiff % cBytesPerSplitPosition);
   return cBytesDiff / cBytesPerSplitPosition;
}

} // namespace DEFINED_ZONE_NAME

#endif // SPLIT_POSITION_HPP
