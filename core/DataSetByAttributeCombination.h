// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_ATTRIBUTE_COMBINATION_H
#define DATA_SET_ATTRIBUTE_COMBINATION_H

#include <assert.h>
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebmcore.h" // FractionalDataType
#include "EbmInternal.h" // TML_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "AttributeCombinationInternal.h"

// TODO: let's take how clean this class is (with almost everything const and the arrays constructed in initialization list) and apply it to as many other classes as we can
// TODO: rename this to DataSetByAttributeCombination
class DataSetAttributeCombination final {
   FractionalDataType * const m_aResidualErrors;
   FractionalDataType * const m_aPredictionScores;
   const StorageDataTypeCore * const m_aTargetData;
   const StorageDataTypeCore * const * const m_aaInputData;
   const size_t m_cCases;
   const size_t m_cAttributeCombinations;

public:

   DataSetAttributeCombination(const bool bAllocateResidualErrors, const bool bAllocatePredictionScores, const bool bAllocateTargetData, const size_t cAttributeCombinations, const AttributeCombinationCore * const * const apAttributeCombination, const size_t cCases, const IntegerDataType * const aInputDataFrom, const void * const aTargets, const FractionalDataType * const aPredictionScoresFrom, const size_t cVectorLength);
   ~DataSetAttributeCombination();

   TML_INLINE bool IsError() const {
      return nullptr == m_aResidualErrors || nullptr == m_aPredictionScores || nullptr == m_aTargetData || 0 != m_cAttributeCombinations && nullptr == m_aaInputData;
   }

   TML_INLINE FractionalDataType * GetResidualPointer() {
      EBM_ASSERT(nullptr != m_aResidualErrors);
      return m_aResidualErrors;
   }
   TML_INLINE const FractionalDataType * GetResidualPointer() const {
      EBM_ASSERT(nullptr != m_aResidualErrors);
      return m_aResidualErrors;
   }
   TML_INLINE FractionalDataType * GetPredictionScores() {
      EBM_ASSERT(nullptr != m_aPredictionScores);
      return m_aPredictionScores;
   }
   TML_INLINE const FractionalDataType * GetPredictionScores() const {
      EBM_ASSERT(nullptr != m_aPredictionScores);
      return m_aPredictionScores;
   }
   TML_INLINE const StorageDataTypeCore * GetTargetDataPointer() const {
      EBM_ASSERT(nullptr != m_aTargetData);
      return m_aTargetData;
   }
   // TODO: we can change this to take the m_iInputData value directly, which we get from the user! (this also applies to the other dataset)
   // TODO: rename this to GetInputDataPointer
   TML_INLINE const StorageDataTypeCore * GetDataPointer(const AttributeCombinationCore * const pAttributeCombination) const {
      EBM_ASSERT(nullptr != pAttributeCombination);
      EBM_ASSERT(pAttributeCombination->m_iInputData < m_cAttributeCombinations);
      EBM_ASSERT(nullptr != m_aaInputData);
      return m_aaInputData[pAttributeCombination->m_iInputData];
   }
   TML_INLINE size_t GetCountCases() const {
      return m_cCases;
   }
   TML_INLINE size_t GetCountAttributeCombinations() const {
      return m_cAttributeCombinations;
   }
};

#endif // DATA_SET_ATTRIBUTE_COMBINATION_H
