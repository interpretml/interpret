// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h"
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureAtomic.h"
#include "DataSetInteraction.h"

extern bool InitializeResiduals(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const size_t cSamples,
   const void * const aTargetData,
   const FloatEbmType * const aPredictorScores,
   FloatEbmType * pResidualError
);

INLINE_RELEASE_UNTEMPLATED static FloatEbmType * ConstructResidualErrors(
   const size_t cSamples, 
   const void * const aTargetData, 
   const FloatEbmType * const aPredictorScores, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
) {
   LOG_0(TraceLevelInfo, "Entered DataFrameInteraction::ConstructResidualErrors");

   EBM_ASSERT(1 <= cSamples);
   EBM_ASSERT(nullptr != aTargetData);
   EBM_ASSERT(nullptr != aPredictorScores);
   // runtimeLearningTypeOrCountTargetClasses can only be zero if there are zero samples and we shouldn't get here
   EBM_ASSERT(0 != runtimeLearningTypeOrCountTargetClasses);

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
   EBM_ASSERT(1 <= cVectorLength);

   if(UNLIKELY(IsMultiplyError(cSamples, cVectorLength))) {
      LOG_0(TraceLevelWarning, "WARNING ConstructResidualErrors IsMultiplyError(cSamples, cVectorLength)");
      return nullptr;
   }

   const size_t cElements = cSamples * cVectorLength;
   FloatEbmType * aResidualErrors = EbmMalloc<FloatEbmType>(cElements);
   if(UNLIKELY(nullptr == aResidualErrors)) {
      LOG_0(TraceLevelWarning, "WARNING ConstructResidualErrors nullptr == aResidualErrors");
      return nullptr;
   }

   if(UNLIKELY(InitializeResiduals(
      runtimeLearningTypeOrCountTargetClasses,
      cSamples,
      aTargetData,
      aPredictorScores,
      aResidualErrors
   ))) {
      // error already logged
      free(aResidualErrors);
      return nullptr;
   }

   LOG_0(TraceLevelInfo, "Exited ConstructResidualErrors");
   return aResidualErrors;
}

INLINE_RELEASE_UNTEMPLATED static StorageDataType * * ConstructInputData(
   const size_t cFeatures, 
   const Feature * const aFeatures, 
   const size_t cSamples, 
   const IntEbmType * const aBinnedData
) {
   LOG_0(TraceLevelInfo, "Entered DataFrameInteraction::ConstructInputData");

   EBM_ASSERT(0 < cFeatures);
   EBM_ASSERT(nullptr != aFeatures);
   EBM_ASSERT(0 < cSamples);
   EBM_ASSERT(nullptr != aBinnedData);

   StorageDataType ** const aaInputDataTo = EbmMalloc<StorageDataType *>(cFeatures);
   if(nullptr == aaInputDataTo) {
      LOG_0(TraceLevelWarning, "WARNING DataFrameInteraction::ConstructInputData nullptr == aaInputDataTo");
      return nullptr;
   }

   StorageDataType ** paInputDataTo = aaInputDataTo;
   const Feature * pFeature = aFeatures;
   const Feature * const pFeatureEnd = aFeatures + cFeatures;
   do {
      StorageDataType * pInputDataTo = EbmMalloc<StorageDataType>(cSamples);
      if(nullptr == pInputDataTo) {
         LOG_0(TraceLevelWarning, "WARNING DataFrameInteraction::ConstructInputData nullptr == pInputDataTo");
         goto free_all;
      }
      *paInputDataTo = pInputDataTo;
      ++paInputDataTo;

      const IntEbmType * pInputDataFrom = &aBinnedData[pFeature->GetIndexFeatureData() * cSamples];
      const IntEbmType * pInputDataFromEnd = &pInputDataFrom[cSamples];
      do {
         const IntEbmType inputData = *pInputDataFrom;
         if(inputData < 0) {
            LOG_0(TraceLevelError, "ERROR DataFrameInteraction::ConstructInputData inputData value cannot be negative");
            goto free_all;
         }
         if(!IsNumberConvertable<StorageDataType>(inputData)) {
            LOG_0(TraceLevelError, "ERROR DataFrameInteraction::ConstructInputData inputData value too big to reference memory");
            goto free_all;
         }
         if(!IsNumberConvertable<size_t>(inputData)) {
            LOG_0(TraceLevelError, "ERROR DataFrameInteraction::ConstructInputData inputData value too big to reference memory");
            goto free_all;
         }
         const size_t iData = static_cast<size_t>(inputData);
         if(pFeature->GetCountBins() <= iData) {
            LOG_0(TraceLevelError, "ERROR DataFrameInteraction::ConstructInputData iData value must be less than the number of bins");
            goto free_all;
         }
         *pInputDataTo = static_cast<StorageDataType>(inputData);
         ++pInputDataTo;
         ++pInputDataFrom;
      } while(pInputDataFromEnd != pInputDataFrom);

      ++pFeature;
   } while(pFeatureEnd != pFeature);

   LOG_0(TraceLevelInfo, "Exited DataFrameInteraction::ConstructInputData");
   return aaInputDataTo;

free_all:
   while(aaInputDataTo != paInputDataTo) {
      --paInputDataTo;
      free(*paInputDataTo);
   }
   free(aaInputDataTo);
   return nullptr;
}

WARNING_PUSH
WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
void DataFrameInteraction::Destruct() {
   LOG_0(TraceLevelInfo, "Entered DataFrameInteraction::Destruct");

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

   LOG_0(TraceLevelInfo, "Exited DataFrameInteraction::Destruct");
}
WARNING_POP

bool DataFrameInteraction::Initialize(
   const size_t cFeatures, 
   const Feature * const aFeatures, 
   const size_t cSamples, 
   const IntEbmType * const aBinnedData, 
   const void * const aTargetData, 
   const FloatEbmType * const aPredictorScores, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
) {
   EBM_ASSERT(nullptr == m_aResidualErrors); // we expect to start with zeroed values
   EBM_ASSERT(nullptr == m_aaInputData); // we expect to start with zeroed values
   EBM_ASSERT(0 == m_cSamples); // we expect to start with zeroed values

   LOG_0(TraceLevelInfo, "Entered DataFrameInteraction::Initialize");

   if(0 != cSamples) {
      // runtimeLearningTypeOrCountTargetClasses can only be zero if 
      // there are zero samples and we shouldn't get past this point
      EBM_ASSERT(0 != runtimeLearningTypeOrCountTargetClasses);

      // if cSamples is zero, then we don't need to allocate anything since we won't use them anyways

      // check our targets since we don't use them other than for initializing residuals
      if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
         const IntEbmType * pTargetFrom = static_cast<const IntEbmType *>(aTargetData);
         const IntEbmType * const pTargetFromEnd = pTargetFrom + cSamples;
         const size_t countTargetClasses = static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses);
         do {
            const IntEbmType data = *pTargetFrom;
            if(data < 0) {
               LOG_0(TraceLevelError, "ERROR DataFrameInteraction::Initialize target value cannot be negative");
               return true;
            }
            if(!IsNumberConvertable<StorageDataType>(data)) {
               LOG_0(TraceLevelError, "ERROR DataFrameInteraction::Initialize data target too big to reference memory");
               return true;
            }
            if(!IsNumberConvertable<size_t>(data)) {
               LOG_0(TraceLevelError, "ERROR DataFrameInteraction::Initialize data target too big to reference memory");
               return true;
            }
            const size_t iData = static_cast<size_t>(data);
            if(countTargetClasses <= iData) {
               LOG_0(TraceLevelError, "ERROR DataFrameInteraction::Initialize target value larger than number of classes");
               return true;
            }
            ++pTargetFrom;
         } while(pTargetFromEnd != pTargetFrom);
      }

      FloatEbmType * aResidualErrors = ConstructResidualErrors(cSamples, aTargetData, aPredictorScores, runtimeLearningTypeOrCountTargetClasses);
      if(nullptr == aResidualErrors) {
         goto exit_error;
      }
      if(0 != cFeatures) {
         StorageDataType ** const aaInputData = ConstructInputData(cFeatures, aFeatures, cSamples, aBinnedData);
         if(nullptr == aaInputData) {
            free(aResidualErrors);
            goto exit_error;
         }
         m_aaInputData = aaInputData;
      }
      m_aResidualErrors = aResidualErrors;
      m_cSamples = cSamples;
   }
   m_cFeatures = cFeatures;

   LOG_0(TraceLevelInfo, "Exited DataFrameInteraction::Initialize");
   return false;

exit_error:;
   LOG_0(TraceLevelWarning, "WARNING Exited DataFrameInteraction::Initialize");
   return true;
}
