// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_INTERNAL_H
#define DATA_SET_INTERNAL_H

#include <assert.h>
#include <stddef.h> // size_t, ptrdiff_t

#include "ebmcore.h" // FractionalDataType
#include "EbmInternal.h" // TML_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "AttributeInternal.h"

// TODO: rename this to DataSetByAttribute
class DataSetInternalCore final {
   const FractionalDataType * const m_aResidualErrors;
   const StorageDataTypeCore * const * const m_aaInputData;
   const size_t m_cCases;
   const size_t m_cAttributes;

public:

   DataSetInternalCore(const bool bRegression, const size_t cAttributes, const AttributeInternalCore * const aAttributes, const size_t cCases, const IntegerDataType * const aInputDataFrom, const void * const aTargetData, const FractionalDataType * const aPredictionScores, const size_t cTargetStates, const int iZeroResidual);
   ~DataSetInternalCore();

   TML_INLINE bool IsError() const {
      return nullptr == m_aResidualErrors || nullptr == m_aaInputData;
   }

   TML_INLINE const FractionalDataType * GetResidualPointer() const {
      EBM_ASSERT(nullptr != m_aResidualErrors);
      return m_aResidualErrors;
   }
   // TODO: we can change this to take the m_iInputData value directly, which we get from the user! (this also applies to the other dataset)
   // TODO: rename this to GetInputDataPointer
   TML_INLINE const StorageDataTypeCore * GetDataPointer(const AttributeInternalCore * const pAttribute) const {
      EBM_ASSERT(nullptr != pAttribute);
      EBM_ASSERT(pAttribute->m_iAttributeData < m_cAttributes);
      EBM_ASSERT(nullptr != m_aaInputData);
      return m_aaInputData[pAttribute->m_iAttributeData];
   }
   TML_INLINE size_t GetCountCases() const {
      return m_cCases;
   }
   TML_INLINE size_t GetCountAttributes() const {
      return m_cAttributes;
   }
};

#endif // DATA_SET_INTERNAL_H
