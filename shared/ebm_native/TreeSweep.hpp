// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TREE_SWEEP_HPP
#define TREE_SWEEP_HPP

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "HistogramTargetEntry.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<typename TFloat, bool bClassification>
struct Bin;

template<bool bClassification>
struct TreeSweep final {
private:
   size_t m_cBestSamplesLeft;
   FloatBig m_bestWeightLeft;
   const Bin<FloatBig, bClassification> * m_pBestBin;

   // use the "struct hack" since Flexible array member method is not available in C++
   // m_aBestGradientPairs must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   GradientPair<FloatBig, bClassification> m_aBestGradientPairs[1];

public:

   TreeSweep() = default; // preserve our POD status
   ~TreeSweep() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS size_t GetCountBestSamplesLeft() const {
      return m_cBestSamplesLeft;
   }

   INLINE_ALWAYS void SetCountBestSamplesLeft(const size_t cBestSamplesLeft) {
      m_cBestSamplesLeft = cBestSamplesLeft;
   }

   INLINE_ALWAYS FloatBig GetBestWeightLeft() const {
      return m_bestWeightLeft;
   }

   INLINE_ALWAYS void SetBestWeightLeft(const FloatBig bestWeightLeft) {
      m_bestWeightLeft = bestWeightLeft;
   }

   INLINE_ALWAYS const Bin<FloatBig, bClassification> * GetBestBin() const {
      return m_pBestBin;
   }

   INLINE_ALWAYS void SetBestBin(const Bin<FloatBig, bClassification> * pBestBin) {
      m_pBestBin = pBestBin;
   }

   INLINE_ALWAYS GradientPair<FloatBig, bClassification> * GetBestGradientPairs() {
      return ArrayToPointer(m_aBestGradientPairs);
   }
};
static_assert(std::is_standard_layout<TreeSweep<true>>::value && std::is_standard_layout<TreeSweep<false>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<TreeSweep<true>>::value && std::is_trivial<TreeSweep<false>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<TreeSweep<true>>::value && std::is_pod<TreeSweep<false>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

INLINE_ALWAYS bool GetTreeSweepSizeOverflow(const bool bClassification, const size_t cScores) {
   const size_t cBytesPerGradientPair = GetGradientPairSize<FloatBig>(bClassification);

   if(UNLIKELY(IsMultiplyError(cBytesPerGradientPair, cScores))) {
      return true;
   }

   size_t cBytesTreeSweepComponent;
   if(bClassification) {
      cBytesTreeSweepComponent = sizeof(TreeSweep<true>);
   } else {
      cBytesTreeSweepComponent = sizeof(TreeSweep<false>);
   }
   cBytesTreeSweepComponent -= cBytesPerGradientPair;

   if(UNLIKELY(IsAddError(cBytesTreeSweepComponent, cBytesPerGradientPair * cScores))) {
      return true;
   }

   return false;
}

INLINE_ALWAYS size_t GetTreeSweepSize(bool bClassification, const size_t cScores) {
   const size_t cBytesPerGradientPair = GetGradientPairSize<FloatBig>(bClassification);

   size_t cBytesTreeSweepComponent;
   if(bClassification) {
      cBytesTreeSweepComponent = sizeof(TreeSweep<true>);
   } else {
      cBytesTreeSweepComponent = sizeof(TreeSweep<false>);
   }
   cBytesTreeSweepComponent -= cBytesPerGradientPair;

   return cBytesTreeSweepComponent + cBytesPerGradientPair * cScores;
}

template<bool bClassification>
INLINE_ALWAYS TreeSweep<bClassification> * AddBytesTreeSweep(TreeSweep<bClassification> * const pTreeSweep, const size_t cBytesAdd) {
   return reinterpret_cast<TreeSweep<bClassification> *>(reinterpret_cast<char *>(pTreeSweep) + cBytesAdd);
}

template<bool bClassification>
INLINE_ALWAYS size_t CountTreeSweep(
   const TreeSweep<bClassification> * const pTreeSweepStart,
   const TreeSweep<bClassification> * const pTreeSweepCur,
   const size_t cBytesPerTreeSweep
) {
   EBM_ASSERT(reinterpret_cast<const char *>(pTreeSweepStart) <= reinterpret_cast<const char *>(pTreeSweepCur));
   const size_t cBytesDiff = reinterpret_cast<const char *>(pTreeSweepCur) - reinterpret_cast<const char *>(pTreeSweepStart);
   EBM_ASSERT(0 == cBytesDiff % cBytesPerTreeSweep);
   return cBytesDiff / cBytesPerTreeSweep;
}

} // DEFINED_ZONE_NAME

#endif // TREE_SWEEP_HPP
