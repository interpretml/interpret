// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <assert.h>
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebmcore.h" // FractionalDataType
#include "EbmInternal.h" // AttributeTypeCore
#include "AttributeInternal.h"
#include "DataSetByAttribute.h"
#include "InitializeResiduals.h"

TML_INLINE static const FractionalDataType * ConstructResidualErrors(const bool bRegression, const size_t cCases, const void * const aTargetData, const FractionalDataType * const aPredictionScores, const size_t cTargetStates, const int iZeroResidual) {
   assert(1 <= cCases);
   assert(nullptr != aTargetData);

   const size_t cVectorLength = GetVectorLengthFlatCore(cTargetStates);
   assert(1 <= cVectorLength);

   if (IsMultiplyError(cCases, cVectorLength)) {
      return nullptr;
   }

   const size_t cElements = cCases * cVectorLength;

   if (IsMultiplyError(sizeof(FractionalDataType), cElements)) {
      return nullptr;
   }

   const size_t cBytes = sizeof(FractionalDataType) * cElements;
   FractionalDataType * aResidualErrors = static_cast<FractionalDataType *>(malloc(cBytes));

   InitializeResidualsFlat(bRegression, cCases, aTargetData, aPredictionScores, aResidualErrors, cTargetStates, iZeroResidual);

   return aResidualErrors;
}

TML_INLINE static const StorageDataTypeCore * const * ConstructInputData(const size_t cAttributes, const AttributeInternalCore * const aAttributes, const size_t cCases, const IntegerDataType * const aInputDataFrom) {
   assert(0 < cAttributes);
   assert(nullptr != aAttributes);
   assert(0 < cCases);
   assert(nullptr != aInputDataFrom);

   if(IsMultiplyError(sizeof(StorageDataTypeCore), cCases)) {
      // we're checking this early instead of checking it inside our loop
      return nullptr;
   }
   const size_t cSubBytesData = sizeof(StorageDataTypeCore) * cCases;

   if (IsMultiplyError(sizeof(void *), cAttributes)) {
      return nullptr;
   }
   const size_t cBytesMemoryArray = sizeof(void *) * cAttributes;
   StorageDataTypeCore ** const aaInputDataTo = static_cast<StorageDataTypeCore * *>(malloc(cBytesMemoryArray));
   if (nullptr == aaInputDataTo) {
      return nullptr;
   }

   StorageDataTypeCore ** paInputDataTo = aaInputDataTo;
   const AttributeInternalCore * pAttribute = aAttributes;
   const AttributeInternalCore * const pAttributeEnd = aAttributes + cAttributes;
   do {
      StorageDataTypeCore * pInputDataTo = static_cast<StorageDataTypeCore *>(malloc(cSubBytesData));
      if (nullptr == pInputDataTo) {
         goto free_all;
      }
      *paInputDataTo = pInputDataTo;
      ++paInputDataTo;

      // TODO : eliminate the counts here and use pointers
      for (size_t iCase = 0; iCase < cCases; ++iCase) {
         // TODO: eliminate this extra internal index lookup (very bad!).  Since the attributes will be in-order, we can probably just use a single pointer to the input data and just keep incrementing it over all the attributes
         const IntegerDataType data = aInputDataFrom[pAttribute->m_iAttributeData * cCases + iCase];
         assert(0 <= data);
         assert((IsNumberConvertable<size_t, IntegerDataType>(data))); // data must be lower than cTargetStates and cTargetStates fits into a size_t which we checked earlier
         assert(static_cast<size_t>(data) < pAttribute->m_cStates);
         assert((IsNumberConvertable<StorageDataTypeCore, IntegerDataType>(data)));
         pInputDataTo[iCase] = static_cast<StorageDataTypeCore>(data);
      }

      ++pAttribute;
   } while (pAttributeEnd != pAttribute);

   return aaInputDataTo;

free_all:
   while (aaInputDataTo != paInputDataTo) {
      --paInputDataTo;
      free(*paInputDataTo);
   }
   free(aaInputDataTo);
   return nullptr;
}

DataSetInternalCore::DataSetInternalCore(const bool bRegression, const size_t cAttributes, const AttributeInternalCore * const aAttributes, const size_t cCases, const IntegerDataType * const aInputDataFrom, const void * const aTargetData, const FractionalDataType * const aPredictionScores, const size_t cTargetStates, const int iZeroResidual)
   : m_aResidualErrors(ConstructResidualErrors(bRegression, cCases, aTargetData, aPredictionScores, cTargetStates, iZeroResidual))
   , m_aaInputData(ConstructInputData(cAttributes, aAttributes, cCases, aInputDataFrom))
   , m_cCases(cCases)
   , m_cAttributes(cAttributes) {

   assert(0 < cCases);
   assert(0 < cAttributes);
}

DataSetInternalCore::~DataSetInternalCore() {
   free(const_cast<FractionalDataType *>(m_aResidualErrors));
   if (nullptr != m_aaInputData) {
      assert(0 < m_cAttributes);
      const StorageDataTypeCore * const * paInputData = m_aaInputData;
      const StorageDataTypeCore * const * const paInputDataEnd = m_aaInputData + m_cAttributes;
      do {
         assert(nullptr != *paInputData);
         free(const_cast<StorageDataTypeCore *>(*paInputData));
         ++paInputData;
      } while (paInputDataEnd != paInputData);
      free(const_cast<StorageDataTypeCore * *>(m_aaInputData));
   }
}
