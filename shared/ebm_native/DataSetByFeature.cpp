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

EBM_INLINE static FloatEbmType * ConstructResidualErrors(
   const size_t cInstances, 
   const void * const aTargetData, 
   const FloatEbmType * const aPredictorScores, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
) {
   LOG_0(TraceLevelInfo, "Entered DataSetByFeature::ConstructResidualErrors");

   EBM_ASSERT(1 <= cInstances);
   EBM_ASSERT(nullptr != aTargetData);
   EBM_ASSERT(nullptr != aPredictorScores);

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
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
      if(IsBinaryClassification(runtimeLearningTypeOrCountTargetClasses)) {
         const bool bError = InitializeResiduals<2>::Func(
            cInstances, 
            aTargetData, 
            aPredictorScores, 
            aResidualErrors, 
            ptrdiff_t { 2 }
         );
         if(bError) {
#ifdef EXPAND_BINARY_LOGITS
            free(aResidualErrors);
            LOG_0(TraceLevelWarning, "WARNING DataSetByFeature::ConstructResidualErrors InitializeResiduals");
            return nullptr;
#else // EXPAND_BINARY_LOGITS
            EBM_ASSERT(false);
#endif // EXPAND_BINARY_LOGITS
         }
      } else {
         const bool bError = InitializeResiduals<k_DynamicClassification>::Func(
            cInstances, 
            aTargetData, 
            aPredictorScores, 
            aResidualErrors, 
            runtimeLearningTypeOrCountTargetClasses
         );
         if(bError) {
            free(aResidualErrors);
            LOG_0(TraceLevelWarning, "WARNING DataSetByFeature::ConstructResidualErrors InitializeResiduals");
            return nullptr;
         }
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      const bool bError = InitializeResiduals<k_Regression>::Func(cInstances, aTargetData, aPredictorScores, aResidualErrors, k_Regression);
      if(bError) {
         EBM_ASSERT(false);
      }
   }

   LOG_0(TraceLevelInfo, "Exited DataSetByFeature::ConstructResidualErrors");
   return aResidualErrors;
}

EBM_INLINE static StorageDataType * * ConstructInputData(
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

DataSetByFeature::~DataSetByFeature() {
   LOG_0(TraceLevelInfo, "Entered DataSetByFeature::~DataSetByFeature");

   free(m_aResidualErrors);
   if(nullptr != m_aaInputData) {
      EBM_ASSERT(1 <= m_cFeatures);
      StorageDataType ** paInputData = m_aaInputData;
      const StorageDataType * const * const paInputDataEnd = m_aaInputData + m_cFeatures;
      do {
         EBM_ASSERT(nullptr != *paInputData);
         free(*paInputData);
         ++paInputData;
      } while(paInputDataEnd != paInputData);
      free(m_aaInputData);
   }

   LOG_0(TraceLevelInfo, "Exited DataSetByFeature::~DataSetByFeature");
}

bool DataSetByFeature::Initialize(
   const size_t cFeatures, 
   const Feature * const aFeatures, 
   const size_t cInstances, 
   const IntEbmType * const aBinnedData, 
   const void * const aTargetData, 
   const FloatEbmType * const aPredictorScores, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
) {
   EBM_ASSERT(nullptr == m_aResidualErrors); // we expect to start with zeroed values
   EBM_ASSERT(nullptr == m_aaInputData); // we expect to start with zeroed values
   EBM_ASSERT(0 == m_cInstances); // we expect to start with zeroed values

   LOG_0(TraceLevelInfo, "Entered DataSetByFeature::Initialize");

   if(0 != cInstances) {
      // if cInstances is zero, then we don't need to allocate anything since we won't use them anyways

      FloatEbmType * aResidualErrors = ConstructResidualErrors(cInstances, aTargetData, aPredictorScores, runtimeLearningTypeOrCountTargetClasses);
      if(nullptr == aResidualErrors) {
         goto exit_error;
      }
      if(0 != cFeatures) {
         StorageDataType ** const aaInputData = ConstructInputData(cFeatures, aFeatures, cInstances, aBinnedData);
         if(nullptr == aaInputData) {
            free(aResidualErrors);
            goto exit_error;
         }
         m_aaInputData = aaInputData;
      }
      m_aResidualErrors = aResidualErrors;
      m_cInstances = cInstances;
   }
   m_cFeatures = cFeatures;

   LOG_0(TraceLevelInfo, "Exited DataSetByFeature::Initialize");
   return false;

exit_error:;
   LOG_0(TraceLevelWarning, "WARNING Exited DataSetByFeature::Initialize");
   return true;
}
