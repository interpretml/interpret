// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <assert.h>
#include <string.h> // memset
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebmcore.h" // FractionalDataType
#include "EbmInternal.h" // AttributeTypeCore
#include "AttributeInternal.h"
#include "AttributeSet.h" // our DataSetInternal.h file can get away with a forward reference to AttributeSetInternal, but we use fields from AttributeSetInternal below
#include "DataSetByAttribute.h"

DataSetInternalCore::~DataSetInternalCore() {
   free(m_aResidualErrors);
   free(m_aTargetData);
   if(nullptr != m_aaData) {
      assert(0 < m_pAttributeSet->GetCountAttributes());
      void ** paData = m_aaData;
      const void * const * const paDataEnd = m_aaData + m_pAttributeSet->GetCountAttributes();
      do {
         free(*paData); // this can be nullptr if we experienced an error in the middle of allocating data
         ++paData;
      } while(paDataEnd != paData);
      free(m_aaData);
   }
}

bool DataSetInternalCore::Initialize(const size_t cTargetBits, const bool bAllocateResidualErrors, const size_t cVectorLength) {
   if(bAllocateResidualErrors) {
      if(IsMultiplyError(m_cCases, cVectorLength)) {
         return true;
      }

      const size_t cElements = m_cCases * cVectorLength;

      if(IsMultiplyError(sizeof(FractionalDataType), cElements)) {
         return true;
      }

      const size_t cBytesResidualErrors = cElements * sizeof(FractionalDataType);

      m_aResidualErrors = static_cast<FractionalDataType *>(malloc(cBytesResidualErrors));
      if(nullptr == m_aResidualErrors) {
         return true;
      }
   }

   assert(cTargetBits <= k_cBitsForSizeTCore);
   if(0 != cTargetBits) {
      if(IsMultiplyError(m_cCases, cTargetBits)) {
         // in theory we might overflow this value because we're using bits instead of bytes, but in practice, we couldn't have more than 6 input attributes of 1 byte size before we'd overflow memory allocation
         return true;
      }

      const size_t cTargetDataBits = m_cCases * cTargetBits;
      if(std::numeric_limits<size_t>::max() - 7 < cTargetDataBits) {
         return true;
      }
      const size_t cTargetDataBytes = (cTargetDataBits + 7) / 8; // round up to the nearest byte

      m_aTargetData = malloc(cTargetDataBytes);
      if(nullptr == m_aTargetData) {
         return true;
      }
   }

   const size_t cAttributes = m_pAttributeSet->GetCountAttributes();
   assert(0 < cAttributes);
   const size_t cBytesMemory = sizeof(*m_aaData) * cAttributes;
   m_aaData = static_cast<void **>(malloc(cBytesMemory));
   if(nullptr == m_aaData) {
      return true;
   }
   memset(m_aaData, 0, cBytesMemory); // if there is an error allocating one of our data arrays, we don't want to free random memory in our destructor, so zero it!

   void * aData;
   size_t cBytesDataItem;
   for(const AttributeInternalCore * const pInputAttribute : m_pAttributeSet->m_inputAttributes) {
      cBytesDataItem = sizeof(StorageDataTypeCore);
      aData = static_cast<void *>(malloc(cBytesDataItem * m_cCases));
      if(nullptr == aData) {
         return true;
      }
      m_aaData[pInputAttribute->m_iAttributeData] = aData;
   }
   return false;
}

