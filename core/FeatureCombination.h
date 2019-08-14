// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef ATTRIBUTE_COMBINATION_H
#define ATTRIBUTE_COMBINATION_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // TML_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "Feature.h"

class FeatureCombination final {
public:

   struct FeatureCombinationEntry {
      const Feature * m_pFeature;
   };

   size_t m_cItemsPerBitPackDataUnit;
   size_t m_cFeatures;
   size_t m_iInputData;
   unsigned int m_cLogEnterGenerateModelFeatureCombinationUpdateMessages;
   unsigned int m_cLogExitGenerateModelFeatureCombinationUpdateMessages;
   unsigned int m_cLogEnterApplyModelFeatureCombinationUpdateMessages;
   unsigned int m_cLogExitApplyModelFeatureCombinationUpdateMessages;
   FeatureCombinationEntry m_FeatureCombinationEntry[1];

   TML_INLINE static size_t GetFeatureCombinationCountBytes(const size_t cFeatures) {
      return sizeof(FeatureCombination) - sizeof(FeatureCombination::FeatureCombinationEntry) + sizeof(FeatureCombination::FeatureCombinationEntry) * cFeatures;
   }

   TML_INLINE void Initialize(const size_t cFeatures, const size_t iFeatureCombination) {
      m_cFeatures = cFeatures;
      m_iInputData = iFeatureCombination;
      m_cLogEnterGenerateModelFeatureCombinationUpdateMessages = 2;
      m_cLogExitGenerateModelFeatureCombinationUpdateMessages = 2;
      m_cLogEnterApplyModelFeatureCombinationUpdateMessages = 2;
      m_cLogExitApplyModelFeatureCombinationUpdateMessages = 2;
   }

   TML_INLINE static FeatureCombination * Allocate(const size_t cFeatures, const size_t iFeatureCombination) {
      const size_t cBytes = GetFeatureCombinationCountBytes(cFeatures);
      EBM_ASSERT(0 < cBytes);
      FeatureCombination * const pFeatureCombination = static_cast<FeatureCombination *>(malloc(cBytes));
      if(UNLIKELY(nullptr == pFeatureCombination)) {
         return nullptr;
      }
      pFeatureCombination->Initialize(cFeatures, iFeatureCombination);
      return pFeatureCombination;
   }

   TML_INLINE static void Free(FeatureCombination * const pFeatureCombination) {
      free(pFeatureCombination);
   }

   TML_INLINE static FeatureCombination ** AllocateFeatureCombinations(const size_t cFeatureCombinations) {
      LOG(TraceLevelInfo, "Entered FeatureCombination::AllocateFeatureCombinations");

      EBM_ASSERT(0 < cFeatureCombinations);
      FeatureCombination ** const apFeatureCombinations = new (std::nothrow) FeatureCombination * [cFeatureCombinations];
      if(LIKELY(nullptr != apFeatureCombinations)) {
         // we need to set this to zero otherwise our destructor will attempt to free garbage memory pointers if we prematurely call the destructor
         EBM_ASSERT(!IsMultiplyError(sizeof(*apFeatureCombinations), cFeatureCombinations)); // if we were able to allocate this, then we should be able to calculate how much memory to zero
         memset(apFeatureCombinations, 0, sizeof(*apFeatureCombinations) * cFeatureCombinations);
      }
      LOG(TraceLevelInfo, "Exited FeatureCombination::AllocateFeatureCombinations");
      return apFeatureCombinations;
   }

   TML_INLINE static void FreeFeatureCombinations(const size_t cFeatureCombinations, FeatureCombination ** apFeatureCombinations) {
      LOG(TraceLevelInfo, "Entered FeatureCombination::FreeFeatureCombinations");
      if(nullptr != apFeatureCombinations) {
         EBM_ASSERT(0 < cFeatureCombinations);
         for(size_t i = 0; i < cFeatureCombinations; ++i) {
            FeatureCombination::Free(apFeatureCombinations[i]);
         }
         delete[] apFeatureCombinations;
      }
      LOG(TraceLevelInfo, "Exited FeatureCombination::FreeFeatureCombinations");
   }
};
static_assert(std::is_pod<FeatureCombination>::value, "We have an array at the end of this stucture, so we don't want anyone else derriving something and putting data there, and non-POD data is probably undefined as to what the space after gets filled with");

// these need to be declared AFTER the class above since the size of FeatureCombination isn't set until the class has been completely declared, and constexpr needs the size before constexpr
constexpr size_t GetFeatureCombinationCountBytesConst(const size_t cFeatures) {
   return sizeof(FeatureCombination) - sizeof(FeatureCombination::FeatureCombinationEntry) + sizeof(FeatureCombination::FeatureCombinationEntry) * cFeatures;
}
constexpr size_t k_cBytesFeatureCombinationMax = GetFeatureCombinationCountBytesConst(k_cDimensionsMax);

#ifndef NDEBUG
class FeatureCombinationCheck final {
public:
   FeatureCombinationCheck() {
      // we need two separate functions for determining the maximum size of FeatureCombination, so let's check that they match at runtime
      EBM_ASSERT(k_cBytesFeatureCombinationMax == FeatureCombination::GetFeatureCombinationCountBytes(k_cDimensionsMax));
   }
};
static FeatureCombinationCheck DEBUG_FeatureCombinationCheck; // yes, this gets duplicated for each include, but it's just for debug..
#endif // NDEBUG

#endif // ATTRIBUTE_COMBINATION_H
