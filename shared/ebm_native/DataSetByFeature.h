// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_BY_FEATURE_H
#define DATA_SET_BY_FEATURE_H

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "Feature.h"

class DataSetByFeature final {
   const FloatEbmType * const m_aResidualErrors;
   const StorageDataType * const * const m_aaInputData;
   const size_t m_cInstances;
   const size_t m_cFeatures;

public:

   DataSetByFeature(
      const size_t cFeatures, 
      const Feature * const aFeatures, 
      const size_t cInstances, 
      const IntEbmType * const aInputDataFrom, 
      const void * const aTargetData, 
      const FloatEbmType * const aPredictorScores, 
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
   );
   ~DataSetByFeature();

   EBM_INLINE bool IsError() const {
      return nullptr == m_aResidualErrors || (0 != m_cFeatures && nullptr == m_aaInputData);
   }

   EBM_INLINE const FloatEbmType * GetResidualPointer() const {
      EBM_ASSERT(nullptr != m_aResidualErrors);
      return m_aResidualErrors;
   }
   // TODO: we can change this to take the m_iFeatureData value directly, which we get from a loop index
   EBM_INLINE const StorageDataType * GetInputDataPointer(const Feature * const pFeature) const {
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

#endif // DATA_SET_BY_FEATURE_H
