// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_BY_FEATURE_COMBINATION_H
#define DATA_SET_BY_FEATURE_COMBINATION_H

#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureCombination.h"

class DataSetByFeatureCombination final {
   FloatEbmType * m_aResidualErrors;
   FloatEbmType * m_aPredictorScores;
   StorageDataType * m_aTargetData;
   StorageDataType * * m_aaInputData;
   size_t m_cInstances;
   size_t m_cFeatureCombinations;

public:

   ~DataSetByFeatureCombination();

   INLINE_RELEASE DataSetByFeatureCombination() {
      // TODO: when we're embedded inside a POD struct we can eliminate this
      memset(this, 0, sizeof(*this));

      // we must be zeroed before being called so that we don't try and free randomized pointers
      EBM_ASSERT(nullptr == m_aResidualErrors);
      EBM_ASSERT(nullptr == m_aPredictorScores);
      EBM_ASSERT(nullptr == m_aTargetData);
      EBM_ASSERT(nullptr == m_aaInputData);
   }

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
      const size_t cVectorLength
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
   // TODO: we can change this to take the m_iInputData value directly, which we get from a loop index
   EBM_INLINE const StorageDataType * GetInputDataPointer(const FeatureCombination * const pFeatureCombination) const {
      EBM_ASSERT(nullptr != pFeatureCombination);
      EBM_ASSERT(pFeatureCombination->m_iInputData < m_cFeatureCombinations);
      EBM_ASSERT(nullptr != m_aaInputData);
      return m_aaInputData[pFeatureCombination->m_iInputData];
   }
   EBM_INLINE size_t GetCountInstances() const {
      return m_cInstances;
   }
   EBM_INLINE size_t GetCountFeatureCombinations() const {
      return m_cFeatureCombinations;
   }
};
static_assert(std::is_standard_layout<DataSetByFeatureCombination>::value,
   "we use memset to zero this, so it needs to be standard layout");

#endif // DATA_SET_BY_FEATURE_COMBINATION_H
