// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef ATTRIBUTE_COMBINATION_H
#define ATTRIBUTE_COMBINATION_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // TML_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "AttributeInternal.h"

class AttributeCombinationCore final {
public:

   struct AttributeCombinationEntry {
      AttributeInternalCore * m_pAttribute;
   };

   size_t m_cItemsPerBitPackDataUnit;
   size_t m_cAttributes;
   size_t m_iInputData;
   unsigned int m_cLogEnterMessages;
   unsigned int m_cLogExitMessages;
   AttributeCombinationEntry m_AttributeCombinationEntry[1];

   TML_INLINE static AttributeCombinationCore * Allocate(const size_t cAttributes, const size_t iAttributeCombination) {
      EBM_ASSERT(0 < cAttributes);

      const size_t cBytes = sizeof(AttributeCombinationCore) - sizeof(AttributeCombinationEntry) + sizeof(AttributeCombinationEntry) * cAttributes;
      AttributeCombinationCore * const pAttributeCombination = static_cast<AttributeCombinationCore *>(malloc(cBytes));
      if(UNLIKELY(nullptr == pAttributeCombination)) {
         return nullptr;
      }
      pAttributeCombination->m_cAttributes = cAttributes;
      pAttributeCombination->m_iInputData = iAttributeCombination;
      pAttributeCombination->m_cLogEnterMessages = 2;
      pAttributeCombination->m_cLogExitMessages = 2;
      return pAttributeCombination;
   }

   TML_INLINE static void Free(AttributeCombinationCore * const pAttributeCombination) {
      free(pAttributeCombination);
   }

   TML_INLINE static AttributeCombinationCore ** AllocateAttributeCombinations(const size_t cAttributeCombinations) {
      LOG(TraceLevelInfo, "Entered AttributeCombinationCore::AllocateAttributeCombinations");

      EBM_ASSERT(0 < cAttributeCombinations);
      AttributeCombinationCore ** const apAttributeCombinations = new (std::nothrow) AttributeCombinationCore * [cAttributeCombinations];
      if (LIKELY(nullptr != apAttributeCombinations)) {
         // we need to set this to zero otherwise our destructor will attempt to free garbage memory pointers if we prematurely call the destructor
         EBM_ASSERT(!IsMultiplyError(sizeof(*apAttributeCombinations), cAttributeCombinations)); // if we were able to allocate this, then we should be able to calculate how much memory to zero
         memset(apAttributeCombinations, 0, sizeof(*apAttributeCombinations) * cAttributeCombinations);
      }
      LOG(TraceLevelInfo, "Exited AttributeCombinationCore::AllocateAttributeCombinations");
      return apAttributeCombinations;
   }

   TML_INLINE static void FreeAttributeCombinations(const size_t cAttributeCombinations, AttributeCombinationCore ** apAttributeCombinations) {
      LOG(TraceLevelInfo, "Entered AttributeCombinationCore::FreeAttributeCombinations");
      if(nullptr != apAttributeCombinations) {
         for(size_t i = 0; i < cAttributeCombinations; ++i) {
            AttributeCombinationCore::Free(apAttributeCombinations[i]);
         }
         delete[] apAttributeCombinations;
      }
      LOG(TraceLevelInfo, "Exited AttributeCombinationCore::FreeAttributeCombinations");
   }
};
static_assert(std::is_pod<AttributeCombinationCore>::value, "We have an array at the end of this stucture, so we don't want anyone else derriving something and putting data there, and non-POD data is probably undefined as to what the space after gets filled with");

#endif // ATTRIBUTE_COMBINATION_H
