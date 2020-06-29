// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_BY_FEATURE_H
#define DATA_SET_BY_FEATURE_H

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureAtomic.h"

class DataSetByFeature final {
   FloatEbmType * m_aResidualErrors;
   StorageDataType * * m_aaInputData;
   size_t m_cInstances;
   size_t m_cFeatures;

   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

public:

   DataSetByFeature() = default; // preserve our POD status
   ~DataSetByFeature() = default; // preserve our POD status

   void Destruct();

   EBM_INLINE void InitializeZero() {
      m_aResidualErrors = nullptr;
      m_aaInputData = nullptr;
      m_cInstances = 0;
      m_cFeatures = 0;
   }

   bool Initialize(
      const size_t cFeatures, 
      const Feature * const aFeatures, 
      const size_t cInstances, 
      const IntEbmType * const aInputDataFrom, 
      const void * const aTargetData, 
      const FloatEbmType * const aPredictorScores, 
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
   );

   EBM_INLINE const FloatEbmType * GetResidualPointer() const {
      EBM_ASSERT(nullptr != m_aResidualErrors);
      return m_aResidualErrors;
   }
   // TODO: we can change this to take the m_iFeatureData value directly, which we get from a loop index
   EBM_INLINE const StorageDataType * GetInputDataPointer(const Feature * const pFeature) const {
      EBM_ASSERT(nullptr != pFeature);
      EBM_ASSERT(pFeature->GetIndexFeatureData() < m_cFeatures);
      EBM_ASSERT(nullptr != m_aaInputData);
      return m_aaInputData[pFeature->GetIndexFeatureData()];
   }
   EBM_INLINE size_t GetCountInstances() const {
      return m_cInstances;
   }
   EBM_INLINE size_t GetCountFeatures() const {
      return m_cFeatures;
   }
};
static_assert(std::is_standard_layout<DataSetByFeature>::value,
   "not required, but keep everything standard_layout since some of our classes use the struct hack");
static_assert(std::is_pod<DataSetByFeature>::value,
   "not required, but keep things closer to C by being POD");

#endif // DATA_SET_BY_FEATURE_H
