// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef FEATURE_COMBINATION_H
#define FEATURE_COMBINATION_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureAtomic.h"

struct FeatureGroupEntry final {
   FeatureGroupEntry() = default; // preserve our POD status
   ~FeatureGroupEntry() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   // TODO : we can copy the entire Feature data into this location instead of using a pointer
   const Feature * m_pFeature;
};
static_assert(std::is_standard_layout<FeatureGroupEntry>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<FeatureGroupEntry>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<FeatureGroupEntry>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

class FeatureGroup final {
   size_t m_cItemsPerBitPackedDataUnit;
   size_t m_cFeatures;
   size_t m_iInputData;
   int m_cLogEnterGenerateModelFeatureGroupUpdateMessages;
   int m_cLogExitGenerateModelFeatureGroupUpdateMessages;
   int m_cLogEnterApplyModelFeatureGroupUpdateMessages;
   int m_cLogExitApplyModelFeatureGroupUpdateMessages;

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

   INLINE_ALWAYS static constexpr size_t GetFeatureGroupCountBytes(const size_t cFeatures) {
      return sizeof(FeatureGroup) - sizeof(FeatureGroupEntry) + sizeof(FeatureGroupEntry) * cFeatures;
   }

   INLINE_ALWAYS static void Free(FeatureGroup * const pFeatureGroup) {
      free(pFeatureGroup);
   }

   INLINE_ALWAYS void Initialize(const size_t cFeatures, const size_t iFeatureGroup) {
      m_cFeatures = cFeatures;
      m_iInputData = iFeatureGroup;
      m_cLogEnterGenerateModelFeatureGroupUpdateMessages = 2;
      m_cLogExitGenerateModelFeatureGroupUpdateMessages = 2;
      m_cLogEnterApplyModelFeatureGroupUpdateMessages = 2;
      m_cLogExitApplyModelFeatureGroupUpdateMessages = 2;
   }

   static FeatureGroup * Allocate(const size_t cFeatures, const size_t iFeatureGroup);
   static FeatureGroup ** AllocateFeatureGroups(const size_t cFeatureGroups);
   static void FreeFeatureGroups(const size_t cFeatureGroups, FeatureGroup ** apFeatureGroups);

   INLINE_ALWAYS void SetCountItemsPerBitPackedDataUnit(const size_t cItemsPerBitPackedDataUnit) {
      m_cItemsPerBitPackedDataUnit = cItemsPerBitPackedDataUnit;
   }

   INLINE_ALWAYS size_t GetCountItemsPerBitPackedDataUnit() const {
      return m_cItemsPerBitPackedDataUnit;
   }

   INLINE_ALWAYS size_t GetIndexInputData() const {
      return m_iInputData;
   }

   INLINE_ALWAYS size_t GetCountFeatures() const {
      return m_cFeatures;
   }

   INLINE_ALWAYS const FeatureGroupEntry * GetFeatureGroupEntries() const {
      return ArrayToPointer(m_FeatureGroupEntry);
   }
   INLINE_ALWAYS FeatureGroupEntry * GetFeatureGroupEntries() {
      return ArrayToPointer(m_FeatureGroupEntry);
   }

   INLINE_ALWAYS int * GetPointerCountLogEnterGenerateModelFeatureGroupUpdateMessages() {
      return &m_cLogEnterGenerateModelFeatureGroupUpdateMessages;
   }

   INLINE_ALWAYS int * GetPointerCountLogExitGenerateModelFeatureGroupUpdateMessages() {
      return &m_cLogExitGenerateModelFeatureGroupUpdateMessages;
   }

   INLINE_ALWAYS int * GetPointerCountLogEnterApplyModelFeatureGroupUpdateMessages() {
      return &m_cLogEnterApplyModelFeatureGroupUpdateMessages;
   }

   INLINE_ALWAYS int * GetPointerCountLogExitApplyModelFeatureGroupUpdateMessages() {
      return &m_cLogExitApplyModelFeatureGroupUpdateMessages;
   }
};
static_assert(std::is_standard_layout<FeatureGroup>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<FeatureGroup>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<FeatureGroup>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");


#endif // FEATURE_COMBINATION_H
