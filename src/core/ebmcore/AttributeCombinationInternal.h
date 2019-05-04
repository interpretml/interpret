// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef ATTRIBUTE_COMBINATION_H
#define ATTRIBUTE_COMBINATION_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // TML_INLINE
#include "AttributeInternal.h"

class AttributeCombinationCore final {
public:

   struct AttributeCombinationEntry {
      AttributeInternalCore * m_pAttribute;
   };

   size_t m_cItemsPerBitPackDataUnit;
   size_t m_cAttributes;
   size_t m_iInputData;
   AttributeCombinationEntry m_AttributeCombinationEntry[1];

   TML_INLINE static AttributeCombinationCore * Allocate(const size_t cAttributes, const size_t iAttributeCombination) {
      assert(0 < cAttributes);

      const size_t cBytes = sizeof(AttributeCombinationCore) - sizeof(AttributeCombinationEntry) + sizeof(AttributeCombinationEntry) * cAttributes;
      AttributeCombinationCore * const pAttributeCombination = static_cast<AttributeCombinationCore *>(malloc(cBytes));
      if(UNLIKELY(nullptr == pAttributeCombination)) {
         return nullptr;
      }
      pAttributeCombination->m_cAttributes = cAttributes;
      pAttributeCombination->m_iInputData = iAttributeCombination;
      return pAttributeCombination;
   }

   TML_INLINE static void Free(AttributeCombinationCore * const pAttributeCombination) {
      free(pAttributeCombination);
   }

   TML_INLINE static void FreeAttributeCombinations(const size_t cAttributeCombinations, AttributeCombinationCore ** apAttributeCombinations) {
      if(nullptr != apAttributeCombinations) {
         for(size_t i = 0; i < cAttributeCombinations; ++i) {
            AttributeCombinationCore::Free(apAttributeCombinations[i]);
         }
         delete[] apAttributeCombinations;
      }
   }
};
static_assert(std::is_pod<AttributeCombinationCore>::value, "We have an array at the end of this stucture, so we don't want anyone else derriving something and putting data there, and non-POD data is probably undefined as to what the space after gets filled with");

#endif // ATTRIBUTE_COMBINATION_H
