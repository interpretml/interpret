// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // FeatureType
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureAtomic.h"
#include "DataSetInteraction.h"

extern void InitializeResiduals(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const size_t cInstances,
   const void * const aTargetData,
   const FloatEbmType * const aPredictorScores,
   FloatEbmType * const aTempFloatVector,
   FloatEbmType * pResidualError
);

INLINE_RELEASE static FloatEbmType * ConstructResidualErrors(
   const size_t cInstances, 
   const void * const aTargetData, 
   const FloatEbmType * const aPredictorScores, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
) {
   LOG_0(TraceLevelInfo, "Entered DataSetByFeature::ConstructResidualErrors");

   EBM_ASSERT(1 <= cInstances);
   EBM_ASSERT(nullptr != aTargetData);
   EBM_ASSERT(nullptr != aPredictorScores);
   // runtimeLearningTypeOrCountTargetClasses can only be zero if there are zero instances and we shouldn't get here
   EBM_ASSERT(0 != runtimeLearningTypeOrCountTargetClasses);

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
   EBM_ASSERT(1 <= cVectorLength);

   if(IsMultiplyError(cInstances, cVectorLength)) {
      LOG_0(TraceLevelWarning, "WARNING DataSetByFeature::ConstructResidualErrors IsMultiplyError(cInstances, cVectorLength)");
      return nullptr;
   }

   const size_t cElements = cInstances * cVectorLength;
   FloatEbmType * aResidualErrors = EbmMalloc<FloatEbmType>(cElements);

   FloatEbmType * aTempFloatVector = nullptr;
   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      if(!IsBinaryClassification(runtimeLearningTypeOrCountTargetClasses)) {
         aTempFloatVector = EbmMalloc<FloatEbmType>(cVectorLength);
         if(UNLIKELY(nullptr == aTempFloatVector)) {
            LOG_0(TraceLevelWarning, "WARNING DataSetByFeature::ConstructResidualErrors nullptr == aTempFloatVector");
            free(aResidualErrors);
            return nullptr;
         }
      }
   }
   if(0 != cInstances) {
      InitializeResiduals(
         runtimeLearningTypeOrCountTargetClasses,
         cInstances,
         aTargetData,
         aPredictorScores,
         aTempFloatVector,
         aResidualErrors
      );
   }
   free(aTempFloatVector);

   LOG_0(TraceLevelInfo, "Exited DataSetByFeature::ConstructResidualErrors");
   return aResidualErrors;
}

INLINE_RELEASE static StorageDataType * * ConstructInputData(
   const size_t cFeatures, 
   const Feature * const aFeatures, 
   const size_t cInstances, 
   const IntEbmType * const aBinnedData
) {
   LOG_0(TraceLevelInfo, "Entered DataSetByFeature::ConstructInputData");

   EBM_ASSERT(0 < cFeatures);
   EBM_ASSERT(nullptr != aFeatures);
   EBM_ASSERT(0 < cInstances);
   EBM_ASSERT(nullptr != aBinnedData);

   StorageDataType ** const aaInputDataTo = EbmMalloc<StorageDataType *>(cFeatures);
   if(nullptr == aaInputDataTo) {
      LOG_0(TraceLevelWarning, "WARNING DataSetByFeature::ConstructInputData nullptr == aaInputDataTo");
      return nullptr;
   }

   StorageDataType ** paInputDataTo = aaInputDataTo;
   const Feature * pFeature = aFeatures;
   const Feature * const pFeatureEnd = aFeatures + cFeatures;
   do {
      StorageDataType * pInputDataTo = EbmMalloc<StorageDataType>(cInstances);
      if(nullptr == pInputDataTo) {
         LOG_0(TraceLevelWarning, "WARNING DataSetByFeature::ConstructInputData nullptr == pInputDataTo");
         goto free_all;
      }
      *paInputDataTo = pInputDataTo;
      ++paInputDataTo;

      const IntEbmType * pInputDataFrom = &aBinnedData[pFeature->GetIndexFeatureData() * cInstances];
      const IntEbmType * pInputDataFromEnd = &pInputDataFrom[cInstances];
      do {
         const IntEbmType inputData = *pInputDataFrom;
         if(inputData < 0) {
            LOG_0(TraceLevelError, "ERROR DataSetByFeature::ConstructInputData inputData value cannot be negative");
            goto free_all;
         }
         if(!IsNumberConvertable<StorageDataType, IntEbmType>(inputData)) {
            LOG_0(TraceLevelError, "ERROR DataSetByFeature::ConstructInputData inputData value too big to reference memory");
            goto free_all;
         }
         if(!IsNumberConvertable<size_t, IntEbmType>(inputData)) {
            LOG_0(TraceLevelError, "ERROR DataSetByFeature::ConstructInputData inputData value too big to reference memory");
            goto free_all;
         }
         const size_t iData = static_cast<size_t>(inputData);
         if(pFeature->GetCountBins() <= iData) {
            LOG_0(TraceLevelError, "ERROR DataSetByFeature::ConstructInputData iData value must be less than the number of bins");
            goto free_all;
         }
         *pInputDataTo = static_cast<StorageDataType>(inputData);
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

WARNING_PUSH
WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
void DataSetByFeature::Destruct() {
   LOG_0(TraceLevelInfo, "Entered DataSetByFeature::Destruct");

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

   LOG_0(TraceLevelInfo, "Exited DataSetByFeature::Destruct");
}
WARNING_POP

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
      // runtimeLearningTypeOrCountTargetClasses can only be zero if 
      // there are zero instances and we shouldn't get past this point
      EBM_ASSERT(0 != runtimeLearningTypeOrCountTargetClasses);

      // if cInstances is zero, then we don't need to allocate anything since we won't use them anyways

      // check our targets since we don't use them other than for initializing residuals
      if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
         const IntEbmType * pTargetFrom = static_cast<const IntEbmType *>(aTargetData);
         const IntEbmType * const pTargetFromEnd = pTargetFrom + cInstances;
         const size_t countTargetClasses = static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses);
         do {
            const IntEbmType data = *pTargetFrom;
            if(data < 0) {
               LOG_0(TraceLevelError, "ERROR DataSetByFeature::Initialize target value cannot be negative");
               return true;
            }
            if(!IsNumberConvertable<StorageDataType, IntEbmType>(data)) {
               LOG_0(TraceLevelError, "ERROR DataSetByFeature::Initialize data target too big to reference memory");
               return true;
            }
            if(!IsNumberConvertable<size_t, IntEbmType>(data)) {
               LOG_0(TraceLevelError, "ERROR DataSetByFeature::Initialize data target too big to reference memory");
               return true;
            }
            const size_t iData = static_cast<size_t>(data);
            if(countTargetClasses <= iData) {
               LOG_0(TraceLevelError, "ERROR DataSetByFeature::Initialize target value larger than number of classes");
               return true;
            }
            ++pTargetFrom;
         } while(pTargetFromEnd != pTargetFrom);
      }

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
