// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef FEATURE_COMBINATION_H
#define FEATURE_COMBINATION_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureAtomic.h"

struct FeatureCombinationEntry final {
   FeatureCombinationEntry() = default; // preserve our POD status
   ~FeatureCombinationEntry() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   // TODO : we can copy the entire Feature data into this location instead of using a pointer
   const Feature * m_pFeature;
};
static_assert(std::is_standard_layout<FeatureCombinationEntry>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<FeatureCombinationEntry>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<FeatureCombinationEntry>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

class FeatureCombination final {
   size_t m_cItemsPerBitPackedDataUnit;
   size_t m_cFeatures;
   size_t m_iInputData;
   unsigned int m_cLogEnterGenerateModelFeatureCombinationUpdateMessages;
   unsigned int m_cLogExitGenerateModelFeatureCombinationUpdateMessages;
   unsigned int m_cLogEnterApplyModelFeatureCombinationUpdateMessages;
   unsigned int m_cLogExitApplyModelFeatureCombinationUpdateMessages;

   // use the "struct hack" since Flexible array member method is not available in C++
   // m_FeatureCombinationEntry must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   FeatureCombinationEntry m_FeatureCombinationEntry[1];

public:

   FeatureCombination() = default; // preserve our POD status
   ~FeatureCombination() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS static constexpr size_t GetFeatureCombinationCountBytes(const size_t cFeatures) {
      return sizeof(FeatureCombination) - sizeof(FeatureCombinationEntry) +
         sizeof(FeatureCombinationEntry) * cFeatures;
   }

   INLINE_ALWAYS static void Free(FeatureCombination * const pFeatureCombination) {
      free(pFeatureCombination);
   }

   INLINE_ALWAYS void Initialize(const size_t cFeatures, const size_t iFeatureCombination) {
      m_cFeatures = cFeatures;
      m_iInputData = iFeatureCombination;
      m_cLogEnterGenerateModelFeatureCombinationUpdateMessages = 2;
      m_cLogExitGenerateModelFeatureCombinationUpdateMessages = 2;
      m_cLogEnterApplyModelFeatureCombinationUpdateMessages = 2;
      m_cLogExitApplyModelFeatureCombinationUpdateMessages = 2;
   }

   static FeatureCombination * Allocate(const size_t cFeatures, const size_t iFeatureCombination);
   static FeatureCombination ** AllocateFeatureCombinations(const size_t cFeatureCombinations);
   static void FreeFeatureCombinations(const size_t cFeatureCombinations, FeatureCombination ** apFeatureCombinations);

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

   INLINE_ALWAYS FeatureCombinationEntry * GetFeatureCombinationEntries() {
      return ArrayToPointer(m_FeatureCombinationEntry);
   }

   INLINE_ALWAYS const FeatureCombinationEntry * GetFeatureCombinationEntries() const {
      return ArrayToPointer(m_FeatureCombinationEntry);
   }

   INLINE_ALWAYS unsigned int * GetPointerCountLogEnterGenerateModelFeatureCombinationUpdateMessages() {
      return &m_cLogEnterGenerateModelFeatureCombinationUpdateMessages;
   }

   INLINE_ALWAYS unsigned int * GetPointerCountLogExitGenerateModelFeatureCombinationUpdateMessages() {
      return &m_cLogExitGenerateModelFeatureCombinationUpdateMessages;
   }

   INLINE_ALWAYS unsigned int * GetPointerCountLogEnterApplyModelFeatureCombinationUpdateMessages() {
      return &m_cLogEnterApplyModelFeatureCombinationUpdateMessages;
   }

   INLINE_ALWAYS unsigned int * GetPointerCountLogExitApplyModelFeatureCombinationUpdateMessages() {
      return &m_cLogExitApplyModelFeatureCombinationUpdateMessages;
   }
};
static_assert(std::is_standard_layout<FeatureCombination>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<FeatureCombination>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<FeatureCombination>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");


#endif // FEATURE_COMBINATION_H
