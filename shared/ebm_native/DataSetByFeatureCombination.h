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

// TODO: let's take how clean this class is (with almost everything const and the arrays constructed in initialization list) and apply it to as many other classes as we can
class DataSetByFeatureCombination final {
   FloatEbmType * const m_aResidualErrors;
   FloatEbmType * const m_aPredictorScores;
   const StorageDataType * const m_aTargetData;
   const StorageDataType * const * const m_aaInputData;
   const size_t m_cInstances;
   const size_t m_cFeatureCombinations;

   const bool m_bAllocateResidualErrors;
   const bool m_bAllocatePredictorScores;
   const bool m_bAllocateTargetData;

public:

   DataSetByFeatureCombination(const bool bAllocateResidualErrors, const bool bAllocatePredictorScores, const bool bAllocateTargetData, const size_t cFeatureCombinations, const FeatureCombination * const * const apFeatureCombination, const size_t cInstances, const IntEbmType * const aInputDataFrom, const void * const aTargets, const FloatEbmType * const aPredictorScoresFrom, const size_t cVectorLength);
   ~DataSetByFeatureCombination();

   EBM_INLINE bool IsError() const {
      return (m_bAllocateResidualErrors && nullptr == m_aResidualErrors) || (m_bAllocatePredictorScores && nullptr == m_aPredictorScores) || (m_bAllocateTargetData && nullptr == m_aTargetData) || (0 != m_cFeatureCombinations && nullptr == m_aaInputData);
   }

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

#endif // DATA_SET_BY_FEATURE_COMBINATION_H
