// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef FEATURE_ATOMIC_H
#define FEATURE_ATOMIC_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // INLINE_ALWAYS

class Feature final {
   size_t m_cBins;
   size_t m_iFeatureData;
   bool m_bCategorical;

public:

   Feature() = default; // preserve our POD status
   ~Feature() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS void Initialize(const size_t cBins, const size_t iFeatureData, const bool bCategorical) {
      m_cBins = cBins;
      m_iFeatureData = iFeatureData;
      m_bCategorical = bCategorical;
   }

   INLINE_ALWAYS size_t GetCountBins() const {
      StopClangAnalysis(); // clang seems to think we're reading uninitialized data here, but we aren't
      return m_cBins;
   }

   INLINE_ALWAYS size_t GetIndexFeatureData() const {
      return m_iFeatureData;
   }

   INLINE_ALWAYS bool GetIsCategorical() const {
      return m_bCategorical;
   }
};
static_assert(std::is_standard_layout<Feature>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<Feature>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<Feature>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

#endif // FEATURE_ATOMIC_H
