// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_BY_FEATURE_H
#define DATA_SET_BY_FEATURE_H

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureAtomic.h"

class DataSetByFeature final {
   FloatEbmType * m_aResidualErrors;
   StorageDataType * * m_aaInputData;
   size_t m_cInstances;
   size_t m_cFeatures;

public:

   DataSetByFeature() = default; // preserve our POD status
   ~DataSetByFeature() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   void Destruct();

   INLINE_ALWAYS void InitializeZero() {
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

   INLINE_ALWAYS const FloatEbmType * GetResidualPointer() const {
      EBM_ASSERT(nullptr != m_aResidualErrors);
      return m_aResidualErrors;
   }
   // TODO: we can change this to take the m_iFeatureData value directly, which we get from a loop index
   INLINE_ALWAYS const StorageDataType * GetInputDataPointer(const Feature * const pFeature) const {
      EBM_ASSERT(nullptr != pFeature);
      EBM_ASSERT(pFeature->GetIndexFeatureData() < m_cFeatures);
      EBM_ASSERT(nullptr != m_aaInputData);
      return m_aaInputData[pFeature->GetIndexFeatureData()];
   }
   INLINE_ALWAYS size_t GetCountInstances() const {
      return m_cInstances;
   }
   INLINE_ALWAYS size_t GetCountFeatures() const {
      return m_cFeatures;
   }
};
static_assert(std::is_standard_layout<DataSetByFeature>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<DataSetByFeature>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<DataSetByFeature>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

#endif // DATA_SET_BY_FEATURE_H
