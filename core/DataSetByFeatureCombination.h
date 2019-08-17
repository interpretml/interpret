// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_ATTRIBUTE_COMBINATION_H
#define DATA_SET_ATTRIBUTE_COMBINATION_H

#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebmcore.h" // FractionalDataType
#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureCombinationCore.h"

// TODO: let's take how clean this class is (with almost everything const and the arrays constructed in initialization list) and apply it to as many other classes as we can
// TODO: rename this to DataSetByFeatureCombination
class DataSetByFeatureCombination final {
   FractionalDataType * const m_aResidualErrors;
   FractionalDataType * const m_aPredictorScores;
   const StorageDataTypeCore * const m_aTargetData;
   const StorageDataTypeCore * const * const m_aaInputData;
   const size_t m_cInstances;
   const size_t m_cFeatureCombinations;

public:

   DataSetByFeatureCombination(const bool bAllocateResidualErrors, const bool bAllocatePredictorScores, const bool bAllocateTargetData, const size_t cFeatureCombinations, const FeatureCombinationCore * const * const apFeatureCombination, const size_t cInstances, const IntegerDataType * const aInputDataFrom, const void * const aTargets, const FractionalDataType * const aPredictorScoresFrom, const size_t cVectorLength);
   ~DataSetByFeatureCombination();

   EBM_INLINE bool IsError() const {
      return nullptr == m_aResidualErrors || nullptr == m_aPredictorScores || nullptr == m_aTargetData || 0 != m_cFeatureCombinations && nullptr == m_aaInputData;
   }

   EBM_INLINE FractionalDataType * GetResidualPointer() {
      EBM_ASSERT(nullptr != m_aResidualErrors);
      return m_aResidualErrors;
   }
   EBM_INLINE const FractionalDataType * GetResidualPointer() const {
      EBM_ASSERT(nullptr != m_aResidualErrors);
      return m_aResidualErrors;
   }
   EBM_INLINE FractionalDataType * GetPredictorScores() {
      EBM_ASSERT(nullptr != m_aPredictorScores);
      return m_aPredictorScores;
   }
   EBM_INLINE const FractionalDataType * GetPredictorScores() const {
      EBM_ASSERT(nullptr != m_aPredictorScores);
      return m_aPredictorScores;
   }
   EBM_INLINE const StorageDataTypeCore * GetTargetDataPointer() const {
      EBM_ASSERT(nullptr != m_aTargetData);
      return m_aTargetData;
   }
   // TODO: we can change this to take the m_iInputData value directly, which we get from the user! (this also applies to the other dataset)
   // TODO: rename this to GetInputDataPointer
   EBM_INLINE const StorageDataTypeCore * GetDataPointer(const FeatureCombinationCore * const pFeatureCombination) const {
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

#endif // DATA_SET_ATTRIBUTE_COMBINATION_H
