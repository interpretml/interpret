// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // FeatureType
#include "Logging.h" // EBM_ASSERT & LOG
#include "Feature.h"
#include "DataSetByFeature.h"
#include "InitializeResiduals.h"

EBM_INLINE static const FloatEbmType * ConstructResidualErrors(
   const size_t cInstances, 
   const void * const aTargetData, 
   const FloatEbmType * const aPredictorScores, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses) {
   LOG_0(TraceLevelInfo, "Entered DataSetByFeature::ConstructResidualErrors");

   EBM_ASSERT(1 <= cInstances);
   EBM_ASSERT(nullptr != aTargetData);
   EBM_ASSERT(nullptr != aPredictorScores);

   const size_t cVectorLength = GetVectorLengthFlat(runtimeLearningTypeOrCountTargetClasses);
   EBM_ASSERT(1 <= cVectorLength);

   if(IsMultiplyError(cInstances, cVectorLength)) {
      LOG_0(TraceLevelWarning, "WARNING DataSetByFeature::ConstructResidualErrors IsMultiplyError(cInstances, cVectorLength)");
      return nullptr;
   }

   const size_t cElements = cInstances * cVectorLength;

   if(IsMultiplyError(sizeof(FloatEbmType), cElements)) {
      LOG_0(TraceLevelWarning, "WARNING DataSetByFeature::ConstructResidualErrors IsMultiplyError(sizeof(FloatEbmType), cElements)");
      return nullptr;
   }

   const size_t cBytes = sizeof(FloatEbmType) * cElements;
   FloatEbmType * aResidualErrors = static_cast<FloatEbmType *>(malloc(cBytes));

   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      if(ptrdiff_t { 2 } == runtimeLearningTypeOrCountTargetClasses) {
         InitializeResiduals<2>(cInstances, aTargetData, aPredictorScores, aResidualErrors, ptrdiff_t { 2 });
      } else {
         InitializeResiduals<k_DynamicClassification>(cInstances, aTargetData, aPredictorScores, aResidualErrors, runtimeLearningTypeOrCountTargetClasses);
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      InitializeResiduals<k_Regression>(cInstances, aTargetData, aPredictorScores, aResidualErrors, k_Regression);
   }

   LOG_0(TraceLevelInfo, "Exited DataSetByFeature::ConstructResidualErrors");
   return aResidualErrors;
}

EBM_INLINE static const StorageDataType * const * ConstructInputData(
   const size_t cFeatures, 
   const Feature * const aFeatures, 
   const size_t cInstances, 
   const IntEbmType * 
   const aBinnedData
) {
   LOG_0(TraceLevelInfo, "Entered DataSetByFeature::ConstructInputData");

   EBM_ASSERT(0 < cFeatures);
   EBM_ASSERT(nullptr != aFeatures);
   EBM_ASSERT(0 < cInstances);
   EBM_ASSERT(nullptr != aBinnedData);

   if(IsMultiplyError(sizeof(StorageDataType), cInstances)) {
      // we're checking this early instead of checking it inside our loop
      LOG_0(TraceLevelWarning, "WARNING DataSetByFeature::ConstructInputData IsMultiplyError(sizeof(StorageDataType), cInstances)");
      return nullptr;
   }
   const size_t cSubBytesData = sizeof(StorageDataType) * cInstances;

   if(IsMultiplyError(sizeof(void *), cFeatures)) {
      LOG_0(TraceLevelWarning, "WARNING DataSetByFeature::ConstructInputData IsMultiplyError(sizeof(void *), cFeatures)");
      return nullptr;
   }
   const size_t cBytesMemoryArray = sizeof(void *) * cFeatures;
   StorageDataType ** const aaInputDataTo = static_cast<StorageDataType * *>(malloc(cBytesMemoryArray));
   if(nullptr == aaInputDataTo) {
      LOG_0(TraceLevelWarning, "WARNING DataSetByFeature::ConstructInputData nullptr == aaInputDataTo");
      return nullptr;
   }

   StorageDataType ** paInputDataTo = aaInputDataTo;
   const Feature * pFeature = aFeatures;
   const Feature * const pFeatureEnd = aFeatures + cFeatures;
   do {
      StorageDataType * pInputDataTo = static_cast<StorageDataType *>(malloc(cSubBytesData));
      if(nullptr == pInputDataTo) {
         LOG_0(TraceLevelWarning, "WARNING DataSetByFeature::ConstructInputData nullptr == pInputDataTo");
         goto free_all;
      }
      *paInputDataTo = pInputDataTo;
      ++paInputDataTo;

      const IntEbmType * pInputDataFrom = &aBinnedData[pFeature->m_iFeatureData * cInstances];
      const IntEbmType * pInputDataFromEnd = &pInputDataFrom[cInstances];
      do {
         const IntEbmType data = *pInputDataFrom;
         EBM_ASSERT(0 <= data);
         EBM_ASSERT((IsNumberConvertable<size_t, IntEbmType>(data))); // data must be lower than cBins and cBins fits into a size_t which we checked earlier
         EBM_ASSERT(static_cast<size_t>(data) < pFeature->m_cBins);
         EBM_ASSERT((IsNumberConvertable<StorageDataType, IntEbmType>(data)));
         *pInputDataTo = static_cast<StorageDataType>(data);
         ++pInputDataTo;
         ++pInputDataFrom;
      } while(pInputDataFromEnd != pInputDataFrom);

      ++pFeature;
   } while(pFeatureEnd != pFeature);

   LOG_0(TraceLevelInfo, "Exited DataSetByFeature::ConstructInputData");
   return aaInputDataTo;

free_all:
   while(aaInputDataTo != paInputDataTo) {
      --paInputDataTo;
      free(*paInputDataTo);
   }
   free(aaInputDataTo);
   return nullptr;
}

DataSetByFeature::DataSetByFeature(
   const size_t cFeatures, 
   const Feature * const aFeatures, 
   const size_t cInstances, 
   const IntEbmType * const aBinnedData, 
   const void * const aTargetData, 
   const FloatEbmType * const aPredictorScores, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
)
   : m_aResidualErrors(ConstructResidualErrors(cInstances, aTargetData, aPredictorScores, runtimeLearningTypeOrCountTargetClasses))
   , m_aaInputData(0 == cFeatures ? nullptr : ConstructInputData(cFeatures, aFeatures, cInstances, aBinnedData))
   , m_cInstances(cInstances)
   , m_cFeatures(cFeatures) {

   EBM_ASSERT(0 < cInstances);
}

DataSetByFeature::~DataSetByFeature() {
   LOG_0(TraceLevelInfo, "Entered ~DataSetByFeature");

   FloatEbmType * aResidualErrors = const_cast<FloatEbmType *>(m_aResidualErrors);
   free(aResidualErrors);
   if(nullptr != m_aaInputData) {
      EBM_ASSERT(1 <= m_cFeatures);
      const StorageDataType * const * paInputData = m_aaInputData;
      const StorageDataType * const * const paInputDataEnd = m_aaInputData + m_cFeatures;
      do {
         EBM_ASSERT(nullptr != *paInputData);
         free(const_cast<StorageDataType *>(*paInputData));
         ++paInputData;
      } while(paInputDataEnd != paInputData);
      free(const_cast<StorageDataType * *>(m_aaInputData));
   }

   LOG_0(TraceLevelInfo, "Exited ~DataSetByFeature");
}
