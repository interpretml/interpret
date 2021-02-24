// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef FEATURE_GROUP_H
#define FEATURE_GROUP_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureAtomic.h"

struct FeatureGroupEntry final {
   FeatureGroupEntry() = default; // preserve our POD status
   ~FeatureGroupEntry() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   // TODO : we can put the entire Feature data into this location instead of using a pointer
   const FeatureAtomic * m_pFeatureAtomic;
};
static_assert(std::is_standard_layout<FeatureGroupEntry>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<FeatureGroupEntry>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<FeatureGroupEntry>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

class FeatureGroup final {
   ptrdiff_t m_cItemsPerBitPack;
   size_t m_cDimensions;
   size_t m_cSignificantDimensions;
   size_t m_iInputData;
   int m_cLogEnterGenerateModelUpdateMessages;
   int m_cLogExitGenerateModelUpdateMessages;
   int m_cLogEnterApplyModelUpdateMessages;
   int m_cLogExitApplyModelUpdateMessages;

   // use the "struct hack" since Flexible array member method is not available in C++
   // m_FeatureGroupEntry must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   FeatureGroupEntry m_FeatureGroupEntry[1];

public:

   FeatureGroup() = default; // preserve our POD status
   ~FeatureGroup() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS static constexpr size_t GetFeatureGroupCountBytes(const size_t cFeatures) noexcept {
      return sizeof(FeatureGroup) - sizeof(FeatureGroupEntry) + sizeof(FeatureGroupEntry) * cFeatures;
   }

   INLINE_ALWAYS static void Free(FeatureGroup * const pFeatureGroup) noexcept {
      free(pFeatureGroup);
   }

   INLINE_ALWAYS void Initialize(const size_t cFeatures, const size_t iFeatureGroup) noexcept {
      m_cDimensions = cFeatures;
      m_iInputData = iFeatureGroup;
      m_cLogEnterGenerateModelUpdateMessages = 2;
      m_cLogExitGenerateModelUpdateMessages = 2;
      m_cLogEnterApplyModelUpdateMessages = 2;
      m_cLogExitApplyModelUpdateMessages = 2;
   }

   static FeatureGroup * Allocate(const size_t cFeatures, const size_t iFeatureGroup) noexcept;
   static FeatureGroup ** AllocateFeatureGroups(const size_t cFeatureGroups) noexcept;
   static void FreeFeatureGroups(const size_t cFeatureGroups, FeatureGroup ** apFeatureGroups) noexcept;

   INLINE_ALWAYS void SetBitPack(const ptrdiff_t cItemsPerBitPack) noexcept {
      EBM_ASSERT(k_cItemsPerBitPackDynamic2 != cItemsPerBitPack);
      EBM_ASSERT(k_cItemsPerBitPackNone == cItemsPerBitPack ||
         k_cItemsPerBitPackMin2 <= cItemsPerBitPack &&
         cItemsPerBitPack <= k_cItemsPerBitPackMax2
      );
      m_cItemsPerBitPack = cItemsPerBitPack;
   }

   INLINE_ALWAYS ptrdiff_t GetBitPack() const noexcept {
      EBM_ASSERT(k_cItemsPerBitPackDynamic2 != m_cItemsPerBitPack);
      EBM_ASSERT(k_cItemsPerBitPackNone == m_cItemsPerBitPack || 
         k_cItemsPerBitPackMin2 <= m_cItemsPerBitPack && 
         m_cItemsPerBitPack <= k_cItemsPerBitPackMax2
      );
      return m_cItemsPerBitPack;
   }

   INLINE_ALWAYS size_t GetIndexInputData() const noexcept {
      return m_iInputData;
   }

   INLINE_ALWAYS size_t GetCountDimensions() const noexcept {
      EBM_ASSERT(m_cSignificantDimensions <= m_cDimensions);
      return m_cDimensions;
   }

   INLINE_ALWAYS size_t GetCountSignificantDimensions() const noexcept {
      EBM_ASSERT(m_cSignificantDimensions <= m_cDimensions);
      return m_cSignificantDimensions;
   }

   INLINE_ALWAYS void SetCountSignificantFeatures(const size_t cSignificantDimensions) noexcept {
      m_cSignificantDimensions = cSignificantDimensions;
   }

   INLINE_ALWAYS const FeatureGroupEntry * GetFeatureGroupEntries() const noexcept {
      return ArrayToPointer(m_FeatureGroupEntry);
   }
   INLINE_ALWAYS FeatureGroupEntry * GetFeatureGroupEntries() noexcept {
      return ArrayToPointer(m_FeatureGroupEntry);
   }

   INLINE_ALWAYS int * GetPointerCountLogEnterGenerateModelUpdateMessages() noexcept {
      return &m_cLogEnterGenerateModelUpdateMessages;
   }

   INLINE_ALWAYS int * GetPointerCountLogExitGenerateModelUpdateMessages() noexcept {
      return &m_cLogExitGenerateModelUpdateMessages;
   }

   INLINE_ALWAYS int * GetPointerCountLogEnterApplyModelUpdateMessages() noexcept {
      return &m_cLogEnterApplyModelUpdateMessages;
   }

   INLINE_ALWAYS int * GetPointerCountLogExitApplyModelUpdateMessages() noexcept {
      return &m_cLogExitApplyModelUpdateMessages;
   }
};
static_assert(std::is_standard_layout<FeatureGroup>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<FeatureGroup>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<FeatureGroup>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");


#endif // FEATURE_GROUP_H
