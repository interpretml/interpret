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
#include "AttributeCombinationInternal.h"

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

   TML_INLINE bool IsError() {
      return nullptr == m_aResidualErrors || nullptr == m_aPredictionScores || nullptr == m_aTargetData || nullptr == m_aaInputData;
   }

   TML_INLINE FractionalDataType * GetResidualPointer() {
      return m_aResidualErrors;
   }
   TML_INLINE const FractionalDataType * GetResidualPointer() const {
      return m_aResidualErrors;
   }
   TML_INLINE FractionalDataType * GetPredictionScores() {
      return m_aPredictionScores;
   }
   TML_INLINE const FractionalDataType * GetPredictionScores() const {
      return m_aPredictionScores;
   }
   TML_INLINE const StorageDataTypeCore * GetTargetDataPointer() const {
      return m_aTargetData;
   }
   TML_INLINE const StorageDataTypeCore * GetDataPointer(const AttributeCombinationCore * const pAttributeCombination) const {
      return static_cast<const StorageDataTypeCore *>(m_aaInputData[pAttributeCombination->m_iInputData]);
   }
   TML_INLINE size_t GetCountCases() const {
      return m_cCases;
   }
   TML_INLINE size_t GetCountAttributeCombinations() const {
      return m_cAttributeCombinations;
   }
};

#endif // DATA_SET_ATTRIBUTE_COMBINATION_H
