// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_BY_FEATURE_COMBINATION_H
#define DATA_SET_BY_FEATURE_COMBINATION_H

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureGroup.h"

class DataSetByFeatureCombination final {
   FloatEbmType * m_aResidualErrors;
   FloatEbmType * m_aPredictorScores;
   StorageDataType * m_aTargetData;
   StorageDataType * * m_aaInputData;
   size_t m_cInstances;
   size_t m_cFeatureCombinations;

public:

   DataSetByFeatureCombination() = default; // preserve our POD status
   ~DataSetByFeatureCombination() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   EBM_INLINE void InitializeZero() {
      m_aResidualErrors = nullptr;
      m_aPredictorScores = nullptr;
      m_aTargetData = nullptr;
      m_aaInputData = nullptr;
      m_cInstances = 0;
      m_cFeatureCombinations = 0;
   }

   void Destruct();

   bool Initialize(
      const bool bAllocateResidualErrors, 
      const bool bAllocatePredictorScores, 
      const bool bAllocateTargetData, 
      const size_t cFeatureCombinations, 
      const FeatureCombination * const * const apFeatureCombination, 
      const size_t cInstances, 
      const IntEbmType * const aInputDataFrom, 
      const void * const aTargets, 
      const FloatEbmType * const aPredictorScoresFrom, 
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
   );

   EBM_INLINE FloatEbmType * GetResidualPointer() {
      EBM_ASSERT(nullptr != m_aResidualErrors);
      return m_aResidualErrors;
   }
   EBM_INLINE const FloatEbmType * GetResidualPointer() const {
      EBM_ASSERT(nullptr != m_aResidualErrors);
      return m_aResidualErrors;
   }
   EBM_INLINE FloatEbmType * GetPredictorScores() {
      EBM_ASSERT(nullptr != m_aPredictorScores);
      return m_aPredictorScores;
   }
   EBM_INLINE const StorageDataType * GetTargetDataPointer() const {
      EBM_ASSERT(nullptr != m_aTargetData);
      return m_aTargetData;
   }
   // TODO: we can change this to take the GetIndexInputData() value directly, which we get from a loop index
   EBM_INLINE const StorageDataType * GetInputDataPointer(const FeatureCombination * const pFeatureCombination) const {
      EBM_ASSERT(nullptr != pFeatureCombination);
      EBM_ASSERT(pFeatureCombination->GetIndexInputData() < m_cFeatureCombinations);
      EBM_ASSERT(nullptr != m_aaInputData);
      return m_aaInputData[pFeatureCombination->GetIndexInputData()];
   }
   EBM_INLINE size_t GetCountInstances() const {
      return m_cInstances;
   }
   EBM_INLINE size_t GetCountFeatureCombinations() const {
      return m_cFeatureCombinations;
   }
};
static_assert(std::is_standard_layout<DataSetByFeatureCombination>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<DataSetByFeatureCombination>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<DataSetByFeatureCombination>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

#endif // DATA_SET_BY_FEATURE_COMBINATION_H
