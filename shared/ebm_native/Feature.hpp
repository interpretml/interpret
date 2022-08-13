// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef FEATURE_HPP
#define FEATURE_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class Feature final {
   size_t m_cBins;
   bool m_bMissing;
   bool m_bUnknown;
   bool m_bNominal;

public:

   Feature() = default; // preserve our POD status
   ~Feature() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS void Initialize(
      const size_t cBins, 
      const bool bMissing, 
      const bool bUnknown, 
      const bool bNominal
   ) noexcept {
      m_cBins = cBins;
      m_bMissing = bMissing;
      m_bUnknown = bUnknown;
      m_bNominal = bNominal;
   }

   INLINE_ALWAYS size_t GetCountBins() const noexcept {
      StopClangAnalysis(); // clang seems to think we're reading uninitialized data here, but we aren't
      return m_cBins;
   }

   INLINE_ALWAYS bool IsMissing() const noexcept {
      return m_bMissing;
   }

   INLINE_ALWAYS bool IsUnknown() const noexcept {
      return m_bUnknown;
   }

   INLINE_ALWAYS bool IsNominal() const noexcept {
      return m_bNominal;
   }
};
static_assert(std::is_standard_layout<Feature>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<Feature>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<Feature>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // FEATURE_HPP
