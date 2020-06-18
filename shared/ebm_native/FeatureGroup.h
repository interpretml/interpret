// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef FEATURE_COMBINATION_H
#define FEATURE_COMBINATION_H

#include <string.h> // memset
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "Feature.h"

struct FeatureCombinationEntry final {
   // TODO : we can copy the entire Feature data into this location instead of using a pointer
   const Feature * m_pFeature;
};

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

   EBM_INLINE static constexpr size_t GetFeatureCombinationCountBytes(const size_t cFeatures) {
      return sizeof(FeatureCombination) - sizeof(FeatureCombinationEntry) +
         sizeof(FeatureCombinationEntry) * cFeatures;
   }

   EBM_INLINE void Free() {
      free(this);
   }

   EBM_INLINE void Initialize(const size_t cFeatures, const size_t iFeatureCombination) {
      m_cFeatures = cFeatures;
      m_iInputData = iFeatureCombination;
      m_cLogEnterGenerateModelFeatureCombinationUpdateMessages = 2;
      m_cLogExitGenerateModelFeatureCombinationUpdateMessages = 2;
      m_cLogEnterApplyModelFeatureCombinationUpdateMessages = 2;
      m_cLogExitApplyModelFeatureCombinationUpdateMessages = 2;
   }

   EBM_INLINE static FeatureCombination * Allocate(const size_t cFeatures, const size_t iFeatureCombination) {
      const size_t cBytes = GetFeatureCombinationCountBytes(cFeatures);
      EBM_ASSERT(0 < cBytes);
      FeatureCombination * const pFeatureCombination = static_cast<FeatureCombination *>(EbmMalloc<void, false>(cBytes));
      if(UNLIKELY(nullptr == pFeatureCombination)) {
         return nullptr;
      }
      pFeatureCombination->Initialize(cFeatures, iFeatureCombination);
      return pFeatureCombination;
   }

   EBM_INLINE static FeatureCombination ** AllocateFeatureCombinations(const size_t cFeatureCombinations) {
      LOG_0(TraceLevelInfo, "Entered FeatureCombination::AllocateFeatureCombinations");

      EBM_ASSERT(0 < cFeatureCombinations);
      FeatureCombination ** const apFeatureCombinations = EbmMalloc<FeatureCombination *, true>(cFeatureCombinations);

      LOG_0(TraceLevelInfo, "Exited FeatureCombination::AllocateFeatureCombinations");
      return apFeatureCombinations;
   }

   EBM_INLINE static void FreeFeatureCombinations(const size_t cFeatureCombinations, FeatureCombination ** apFeatureCombinations) {
      LOG_0(TraceLevelInfo, "Entered FeatureCombination::FreeFeatureCombinations");
      if(nullptr != apFeatureCombinations) {
         EBM_ASSERT(0 < cFeatureCombinations);
         for(size_t i = 0; i < cFeatureCombinations; ++i) {
            if(nullptr != apFeatureCombinations[i]) {
               apFeatureCombinations[i]->Free();
            }
         }
         free(apFeatureCombinations);
      }
      LOG_0(TraceLevelInfo, "Exited FeatureCombination::FreeFeatureCombinations");
   }

   EBM_INLINE void SetCountItemsPerBitPackedDataUnit(const size_t cItemsPerBitPackedDataUnit) {
      m_cItemsPerBitPackedDataUnit = cItemsPerBitPackedDataUnit;
   }

   EBM_INLINE size_t GetCountItemsPerBitPackedDataUnit() const {
      return m_cItemsPerBitPackedDataUnit;
   }

   EBM_INLINE size_t GetIndexInputData() const {
      return m_iInputData;
   }

   EBM_INLINE size_t GetCountFeatures() const {
      return m_cFeatures;
   }

   FeatureCombinationEntry * GetFeatureCombinationEntries() {
      return &m_FeatureCombinationEntry[0];
   }

   const FeatureCombinationEntry * GetFeatureCombinationEntries() const {
      return &m_FeatureCombinationEntry[0];
   }

   unsigned int * GetPointerCountLogEnterGenerateModelFeatureCombinationUpdateMessages() {
      return &m_cLogEnterGenerateModelFeatureCombinationUpdateMessages;
   }

   unsigned int * GetPointerCountLogExitGenerateModelFeatureCombinationUpdateMessages() {
      return &m_cLogExitGenerateModelFeatureCombinationUpdateMessages;
   }

   unsigned int * GetPointerCountLogEnterApplyModelFeatureCombinationUpdateMessages() {
      return &m_cLogEnterApplyModelFeatureCombinationUpdateMessages;
   }

   unsigned int * GetPointerCountLogExitApplyModelFeatureCombinationUpdateMessages() {
      return &m_cLogExitApplyModelFeatureCombinationUpdateMessages;
   }
};
static_assert(
   std::is_standard_layout<FeatureCombination>::value, 
   "We have an array at the end of this stucture, so we don't want anyone else derriving something and putting data there, and non-standard layout "
   "data is probably undefined as to what the space after gets filled with");

#endif // FEATURE_COMBINATION_H
