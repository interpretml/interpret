// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef FEATURE_HPP
#define FEATURE_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT
#include "zones.h"

#include "bridge_cpp.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class FeatureBoosting final {
   size_t m_cBins;
   bool m_bMissing;
   bool m_bUnknown;
   bool m_bNominal;

public:

   FeatureBoosting() = default; // preserve our POD status
   ~FeatureBoosting() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   inline void Initialize(
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

   inline size_t GetCountBins() const noexcept {
      return m_cBins;
   }

   inline bool IsMissing() const noexcept {
      return m_bMissing;
   }

   inline bool IsUnknown() const noexcept {
      return m_bUnknown;
   }

   inline bool IsNominal() const noexcept {
      return m_bNominal;
   }
};
static_assert(std::is_standard_layout<FeatureBoosting>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<FeatureBoosting>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<FeatureBoosting>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

class FeatureInteraction final {
   size_t m_cBins;
   bool m_bMissing;
   bool m_bUnknown;
   bool m_bNominal;
   ptrdiff_t m_cItemsPerBitPack;

public:

   FeatureInteraction() = default; // preserve our POD status
   ~FeatureInteraction() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   inline void Initialize(
      const size_t cBins,
      const bool bMissing,
      const bool bUnknown,
      const bool bNominal
   ) noexcept {
      m_cBins = cBins;
      m_bMissing = bMissing;
      m_bUnknown = bUnknown;
      m_bNominal = bNominal;

      ptrdiff_t cItemsPerBitPack = k_cItemsPerBitPackNone;
      if(size_t { 1 } < cBins) {
         const size_t cBitsRequiredMin = CountBitsRequired(cBins - 1);
         EBM_ASSERT(1 <= cBitsRequiredMin); // 1 < cTensorBins otherwise we'd have filtered it out above
         cItemsPerBitPack = static_cast<ptrdiff_t>(GetCountItemsBitPacked<StorageDataType>(cBitsRequiredMin));
      }
      m_cItemsPerBitPack = cItemsPerBitPack;
   }

   inline ptrdiff_t GetFeatureBitPack() const noexcept {
      return m_cItemsPerBitPack;
   }

   inline size_t GetCountBins() const noexcept {
      return m_cBins;
   }

   inline bool IsMissing() const noexcept {
      return m_bMissing;
   }

   inline bool IsUnknown() const noexcept {
      return m_bUnknown;
   }

   inline bool IsNominal() const noexcept {
      return m_bNominal;
   }
};
static_assert(std::is_standard_layout<FeatureInteraction>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<FeatureInteraction>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<FeatureInteraction>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // FEATURE_HPP
