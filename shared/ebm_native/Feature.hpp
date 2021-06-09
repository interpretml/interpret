// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef FEATURE_ATOMIC_H
#define FEATURE_ATOMIC_H

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class FeatureAtomic final {
   size_t m_cBins;
   size_t m_iFeatureAtomicData;
   bool m_bCategorical;

public:

   FeatureAtomic() = default; // preserve our POD status
   ~FeatureAtomic() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS void Initialize(const size_t cBins, const size_t iFeatureAtomicData, const bool bCategorical) noexcept {
      m_cBins = cBins;
      m_iFeatureAtomicData = iFeatureAtomicData;
      m_bCategorical = bCategorical;
   }

   INLINE_ALWAYS size_t GetCountBins() const noexcept {
      StopClangAnalysis(); // clang seems to think we're reading uninitialized data here, but we aren't
      return m_cBins;
   }

   INLINE_ALWAYS size_t GetIndexFeatureAtomicData() const noexcept {
      return m_iFeatureAtomicData;
   }

   INLINE_ALWAYS bool GetIsCategorical() const noexcept {
      return m_bCategorical;
   }
};
static_assert(std::is_standard_layout<FeatureAtomic>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<FeatureAtomic>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<FeatureAtomic>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // FEATURE_ATOMIC_H
