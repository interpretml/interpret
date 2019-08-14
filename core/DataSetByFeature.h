// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_INTERNAL_H
#define DATA_SET_INTERNAL_H

#include <stddef.h> // size_t, ptrdiff_t

#include "ebmcore.h" // FractionalDataType
#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureCore.h"

// TODO: rename this to DataSetByFeature
class DataSetByFeature final {
   const FractionalDataType * const m_aResidualErrors;
   const StorageDataTypeCore * const * const m_aaInputData;
   const size_t m_cInstances;
   const size_t m_cFeatures;

public:

   DataSetByFeature(const bool bRegression, const size_t cFeatures, const FeatureCore * const aFeatures, const size_t cInstances, const IntegerDataType * const aInputDataFrom, const void * const aTargetData, const FractionalDataType * const aPredictorScores, const size_t cTargetStates);
   ~DataSetByFeature();

   EBM_INLINE bool IsError() const {
      return nullptr == m_aResidualErrors || 0 != m_cFeatures && nullptr == m_aaInputData;
   }

   EBM_INLINE const FractionalDataType * GetResidualPointer() const {
      EBM_ASSERT(nullptr != m_aResidualErrors);
      return m_aResidualErrors;
   }
   // TODO: we can change this to take the m_iInputData value directly, which we get from the user! (this also applies to the other dataset)
   // TODO: rename this to GetInputDataPointer
   EBM_INLINE const StorageDataTypeCore * GetDataPointer(const FeatureCore * const pFeature) const {
      EBM_ASSERT(nullptr != pFeature);
      EBM_ASSERT(pFeature->m_iFeatureData < m_cFeatures);
      EBM_ASSERT(nullptr != m_aaInputData);
      return m_aaInputData[pFeature->m_iFeatureData];
   }
   EBM_INLINE size_t GetCountInstances() const {
      return m_cInstances;
   }
   EBM_INLINE size_t GetCountFeatures() const {
      return m_cFeatures;
   }
};

#endif // DATA_SET_INTERNAL_H
