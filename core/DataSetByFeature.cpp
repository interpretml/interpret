// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebmcore.h" // FractionalDataType
#include "EbmInternal.h" // FeatureTypeCore
#include "Logging.h" // EBM_ASSERT & LOG
#include "Feature.h"
#include "DataSetByFeature.h"
#include "InitializeResiduals.h"

EBM_INLINE static const FractionalDataType * ConstructResidualErrors(const bool bRegression, const size_t cCases, const void * const aTargetData, const FractionalDataType * const aPredictionScores, const size_t cTargetStates) {
   LOG(TraceLevelInfo, "Entered DataSetByFeature::ConstructResidualErrors");

   EBM_ASSERT(1 <= cCases);
   EBM_ASSERT(nullptr != aTargetData);

   const size_t cVectorLength = GetVectorLengthFlatCore(cTargetStates);
   EBM_ASSERT(1 <= cVectorLength);

   if(IsMultiplyError(cCases, cVectorLength)) {
      LOG(TraceLevelWarning, "WARNING DataSetByFeature::ConstructResidualErrors IsMultiplyError(cCases, cVectorLength)");
      return nullptr;
   }

   const size_t cElements = cCases * cVectorLength;

   if(IsMultiplyError(sizeof(FractionalDataType), cElements)) {
      LOG(TraceLevelWarning, "WARNING DataSetByFeature::ConstructResidualErrors IsMultiplyError(sizeof(FractionalDataType), cElements)");
      return nullptr;
   }

   const size_t cBytes = sizeof(FractionalDataType) * cElements;
   FractionalDataType * aResidualErrors = static_cast<FractionalDataType *>(malloc(cBytes));

   if(bRegression) {
      InitializeResiduals<k_Regression>(cCases, aTargetData, aPredictionScores, aResidualErrors, 0);
   } else {
      if(2 == cTargetStates) {
         InitializeResiduals<2>(cCases, aTargetData, aPredictionScores, aResidualErrors, 2);
      } else {
         InitializeResiduals<k_DynamicClassification>(cCases, aTargetData, aPredictionScores, aResidualErrors, cTargetStates);
      }
   }

   LOG(TraceLevelInfo, "Exited DataSetByFeature::ConstructResidualErrors");
   return aResidualErrors;
}

EBM_INLINE static const StorageDataTypeCore * const * ConstructInputData(const size_t cFeatures, const FeatureCore * const aFeatures, const size_t cCases, const IntegerDataType * const aInputDataFrom) {
   LOG(TraceLevelInfo, "Entered DataSetByFeature::ConstructInputData");

   EBM_ASSERT(0 < cFeatures);
   EBM_ASSERT(nullptr != aFeatures);
   EBM_ASSERT(0 < cCases);
   EBM_ASSERT(nullptr != aInputDataFrom);

   if(IsMultiplyError(sizeof(StorageDataTypeCore), cCases)) {
      // we're checking this early instead of checking it inside our loop
      LOG(TraceLevelWarning, "WARNING DataSetByFeature::ConstructInputData IsMultiplyError(sizeof(StorageDataTypeCore), cCases)");
      return nullptr;
   }
   const size_t cSubBytesData = sizeof(StorageDataTypeCore) * cCases;

   if(IsMultiplyError(sizeof(void *), cFeatures)) {
      LOG(TraceLevelWarning, "WARNING DataSetByFeature::ConstructInputData IsMultiplyError(sizeof(void *), cFeatures)");
      return nullptr;
   }
   const size_t cBytesMemoryArray = sizeof(void *) * cFeatures;
   StorageDataTypeCore ** const aaInputDataTo = static_cast<StorageDataTypeCore * *>(malloc(cBytesMemoryArray));
   if(nullptr == aaInputDataTo) {
      LOG(TraceLevelWarning, "WARNING DataSetByFeature::ConstructInputData nullptr == aaInputDataTo");
      return nullptr;
   }

   StorageDataTypeCore ** paInputDataTo = aaInputDataTo;
   const FeatureCore * pFeature = aFeatures;
   const FeatureCore * const pFeatureEnd = aFeatures + cFeatures;
   do {
      StorageDataTypeCore * pInputDataTo = static_cast<StorageDataTypeCore *>(malloc(cSubBytesData));
      if(nullptr == pInputDataTo) {
         LOG(TraceLevelWarning, "WARNING DataSetByFeature::ConstructInputData nullptr == pInputDataTo");
         goto free_all;
      }
      *paInputDataTo = pInputDataTo;
      ++paInputDataTo;

      const IntegerDataType * pInputDataFrom = &aInputDataFrom[pFeature->m_iFeatureData * cCases];
      const IntegerDataType * pInputDataFromEnd = &pInputDataFrom[cCases];
      do {
         const IntegerDataType data = *pInputDataFrom;
         EBM_ASSERT(0 <= data);
         EBM_ASSERT((IsNumberConvertable<size_t, IntegerDataType>(data))); // data must be lower than cTargetStates and cTargetStates fits into a size_t which we checked earlier
         EBM_ASSERT(static_cast<size_t>(data) < pFeature->m_cStates);
         EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, IntegerDataType>(data)));
         *pInputDataTo = static_cast<StorageDataTypeCore>(data);
         ++pInputDataTo;
         ++pInputDataFrom;
      } while(pInputDataFromEnd != pInputDataFrom);

      ++pFeature;
   } while(pFeatureEnd != pFeature);

   LOG(TraceLevelInfo, "Exited DataSetByFeature::ConstructInputData");
   return aaInputDataTo;

free_all:
   while(aaInputDataTo != paInputDataTo) {
      --paInputDataTo;
      free(*paInputDataTo);
   }
   free(aaInputDataTo);
   return nullptr;
}

DataSetByFeature::DataSetByFeature(const bool bRegression, const size_t cFeatures, const FeatureCore * const aFeatures, const size_t cCases, const IntegerDataType * const aInputDataFrom, const void * const aTargetData, const FractionalDataType * const aPredictionScores, const size_t cTargetStates)
   : m_aResidualErrors(ConstructResidualErrors(bRegression, cCases, aTargetData, aPredictionScores, cTargetStates))
   , m_aaInputData(0 == cFeatures ? nullptr : ConstructInputData(cFeatures, aFeatures, cCases, aInputDataFrom))
   , m_cCases(cCases)
   , m_cFeatures(cFeatures) {

   EBM_ASSERT(0 < cCases);
}

DataSetByFeature::~DataSetByFeature() {
   LOG(TraceLevelInfo, "Entered ~DataSetByFeature");

   FractionalDataType * aResidualErrors = const_cast<FractionalDataType *>(m_aResidualErrors);
   free(aResidualErrors);
   if(nullptr != m_aaInputData) {
      EBM_ASSERT(1 <= m_cFeatures);
      const StorageDataTypeCore * const * paInputData = m_aaInputData;
      const StorageDataTypeCore * const * const paInputDataEnd = m_aaInputData + m_cFeatures;
      do {
         EBM_ASSERT(nullptr != *paInputData);
         free(const_cast<StorageDataTypeCore *>(*paInputData));
         ++paInputData;
      } while(paInputDataEnd != paInputData);
      free(const_cast<StorageDataTypeCore * *>(m_aaInputData));
   }

   LOG(TraceLevelInfo, "Exited ~DataSetByFeature");
}
