// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_INTERNAL_H
#define DATA_SET_INTERNAL_H

#include <assert.h>
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebmcore.h" // FractionalDataType
#include "EbmInternal.h" // TML_INLINE
#include "AttributeInternal.h"

class DataSetInternalCore final {
   FractionalDataType * m_aResidualErrors;
   void * m_aTargetData;
   void ** m_aaData; // TODO : rename this variable once we've made it based on an AttributeCombination
   size_t m_cCases;

public:
   const size_t m_cAttributes;

   TML_INLINE DataSetInternalCore(const size_t cAttributes, const size_t cCases)
      : m_aResidualErrors(nullptr)
      , m_aTargetData(nullptr)
      , m_aaData(nullptr)
      , m_cCases(cCases)
      , m_cAttributes(cAttributes) {
      assert(0 < cAttributes);
      assert(0 < m_cCases);
   }

   ~DataSetInternalCore();

   TML_INLINE StorageDataTypeCore * GetDataPointer(const AttributeInternalCore * const pAttribute) {
      return static_cast<StorageDataTypeCore *>(m_aaData[pAttribute->m_iAttributeData]);
   }
   TML_INLINE const StorageDataTypeCore * GetDataPointer(const AttributeInternalCore * const pAttribute) const {
      return static_cast<StorageDataTypeCore *>(m_aaData[pAttribute->m_iAttributeData]);
   }
   TML_INLINE FractionalDataType * GetResidualPointer() {
      return m_aResidualErrors;
   }
   TML_INLINE const FractionalDataType * GetResidualPointer() const {
      return m_aResidualErrors;
   }

   TML_INLINE void * GetTargetDataPointer() {
      return m_aTargetData;
   }

   TML_INLINE size_t GetCountCases() const {
      return m_cCases;
   }

   bool Initialize(const size_t cTargetBits, const bool bAllocateResidualErrors, const size_t cVectorLength);
};

#endif // DATA_SET_INTERNAL_H
