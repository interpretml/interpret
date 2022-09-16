// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef SPLIT_POSITION_HPP
#define SPLIT_POSITION_HPP

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
struct SplitPosition final {
   friend static bool IsOverflowSplitPositionSize(const bool, const size_t);
   friend static size_t GetSplitPositionSize(const bool, const size_t);

private:
   const Bin<FloatBig, bClassification> * m_pBinPosition;

   // IMPORTANT: m_leftSum must be in the last position for the struct hack and this must be standard layout
   Bin<FloatBig, bClassification> m_leftSum;

public:

   SplitPosition() = default; // preserve our POD status
   ~SplitPosition() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS const auto * GetBinPosition() const {
      return m_pBinPosition;
   }
   INLINE_ALWAYS void SetBinPosition(const Bin<FloatBig, bClassification> * const pBinPosition) {
      m_pBinPosition = pBinPosition;
   }

   INLINE_ALWAYS auto * GetLeftSum() {
      return &m_leftSum;
   }
};
static_assert(std::is_standard_layout<SplitPosition<true>>::value && std::is_standard_layout<SplitPosition<false>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<SplitPosition<true>>::value && std::is_trivial<SplitPosition<false>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<SplitPosition<true>>::value && std::is_pod<SplitPosition<false>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

INLINE_ALWAYS static bool IsOverflowSplitPositionSize(const bool bClassification, const size_t cScores) {
   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // check this before calling us
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

   size_t cBytesSplitPositionComponent;
   if(bClassification) {
      cBytesSplitPositionComponent = sizeof(SplitPosition<true>) - sizeof(SplitPosition<true>::m_leftSum);
   } else {
      cBytesSplitPositionComponent = sizeof(SplitPosition<false>) - sizeof(SplitPosition<false>::m_leftSum);
   }

   if(UNLIKELY(IsAddError(cBytesSplitPositionComponent, cBytesPerBin))) {
      return true;
   }

   return false;
}

INLINE_ALWAYS static size_t GetSplitPositionSize(bool bClassification, const size_t cScores) {
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

   size_t cBytesSplitPositionComponent;
   if(bClassification) {
      cBytesSplitPositionComponent = sizeof(SplitPosition<true>) - sizeof(SplitPosition<true>::m_leftSum);
   } else {
      cBytesSplitPositionComponent = sizeof(SplitPosition<false>) - sizeof(SplitPosition<false>::m_leftSum);
   }

   return cBytesSplitPositionComponent + cBytesPerBin;
}

template<bool bClassification>
INLINE_ALWAYS static auto * IndexSplitPosition(
   SplitPosition<bClassification> * const pSplitPosition, 
   const size_t iByte
) {
   return reinterpret_cast<SplitPosition<bClassification> *>(reinterpret_cast<char *>(pSplitPosition) + iByte);
}

template<bool bClassification>
INLINE_ALWAYS static size_t CountSplitPositions(
   const SplitPosition<bClassification> * const pSplitPositionLow,
   const SplitPosition<bClassification> * const pSplitPositionHigh,
   const size_t cBytesPerSplitPosition
) {
   EBM_ASSERT(pSplitPositionLow <= pSplitPositionHigh);
   const size_t cBytesDiff = reinterpret_cast<const char *>(pSplitPositionHigh) - 
      reinterpret_cast<const char *>(pSplitPositionLow);
   EBM_ASSERT(0 == cBytesDiff % cBytesPerSplitPosition);
   return cBytesDiff / cBytesPerSplitPosition;
}

} // DEFINED_ZONE_NAME

#endif // SPLIT_POSITION_HPP
