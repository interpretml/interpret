// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "Feature.hpp"
#include "DataSetInteraction.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern ErrorEbmType InitializeGradientsAndHessians(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const size_t cSamples,
   const void * const aTargetData,
   const FloatEbmType * const aPredictorScores,
   FloatEbmType * pGradient
);

INLINE_RELEASE_UNTEMPLATED static ErrorEbmType ConstructGradientsAndHessians(
   const bool bAllocateHessians,
   const size_t cSamples, 
   const void * const aTargetData, 
   const FloatEbmType * const aPredictorScores, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   FloatEbmType ** paGradientsAndHessiansOut
) {
   LOG_0(TraceLevelInfo, "Entered ConstructGradientsAndHessians");

   EBM_ASSERT(1 <= cSamples);
   EBM_ASSERT(nullptr != aTargetData);
   EBM_ASSERT(nullptr != aPredictorScores);
   // runtimeLearningTypeOrCountTargetClasses can only be zero if there are zero samples and we shouldn't get here
   EBM_ASSERT(0 != runtimeLearningTypeOrCountTargetClasses);

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
   EBM_ASSERT(1 <= cVectorLength);

   const size_t cStorageItems = bAllocateHessians ? 2 : 1;
   if(IsMultiplyError(cVectorLength, cStorageItems)) {
      LOG_0(TraceLevelWarning, "WARNING ConstructGradientsAndHessians IsMultiplyError(cVectorLength, cStorageItems)");
      return Error_OutOfMemory;
   }
   const size_t cStorageItemsPerSample = cVectorLength * cStorageItems;

   if(UNLIKELY(IsMultiplyError(cStorageItemsPerSample, cSamples))) {
      LOG_0(TraceLevelWarning, "WARNING ConstructGradientsAndHessians IsMultiplyError(cStorageItemsPerSample, cSamples)");
      return Error_OutOfMemory;
   }

   const size_t cElements = cStorageItemsPerSample * cSamples;
   FloatEbmType * aGradientsAndHessians = EbmMalloc<FloatEbmType>(cElements);
   if(UNLIKELY(nullptr == aGradientsAndHessians)) {
      LOG_0(TraceLevelWarning, "WARNING ConstructGradientsAndHessians nullptr == aGradientsAndHessians");
      return Error_OutOfMemory;
   }
   *paGradientsAndHessiansOut = aGradientsAndHessians; // transfer ownership for future deletion

   const ErrorEbmType error = InitializeGradientsAndHessians(
      runtimeLearningTypeOrCountTargetClasses,
      cSamples,
      aTargetData,
      aPredictorScores,
      aGradientsAndHessians
   );
   if(UNLIKELY(Error_None != error)) {
      // error already logged
      return error;
   }

   LOG_0(TraceLevelInfo, "Exited ConstructGradientsAndHessians");
   return Error_None;
}

INLINE_RELEASE_UNTEMPLATED static StorageDataType * * ConstructInputData(
   const size_t cFeatures, 
   const Feature * const aFeatures, 
   const size_t cSamples, 
   const IntEbmType * const aBinnedData
) {
   LOG_0(TraceLevelInfo, "Entered DataSetInteraction::ConstructInputData");

   EBM_ASSERT(0 < cFeatures);
   EBM_ASSERT(nullptr != aFeatures);
   EBM_ASSERT(0 < cSamples);
   EBM_ASSERT(nullptr != aBinnedData);

   StorageDataType ** const aaInputDataTo = EbmMalloc<StorageDataType *>(cFeatures);
   if(nullptr == aaInputDataTo) {
      LOG_0(TraceLevelWarning, "WARNING DataSetInteraction::ConstructInputData nullptr == aaInputDataTo");
      return nullptr;
   }

   StorageDataType ** paInputDataTo = aaInputDataTo;
   const Feature * pFeature = aFeatures;
   const Feature * const pFeatureEnd = aFeatures + cFeatures;
   do {
      StorageDataType * pInputDataTo = EbmMalloc<StorageDataType>(cSamples);
      if(nullptr == pInputDataTo) {
         LOG_0(TraceLevelWarning, "WARNING DataSetInteraction::ConstructInputData nullptr == pInputDataTo");
         goto free_all;
      }
      *paInputDataTo = pInputDataTo;
      ++paInputDataTo;

      const IntEbmType * pInputDataFrom = &aBinnedData[pFeature->GetIndexFeatureData() * cSamples];
      const IntEbmType * pInputDataFromEnd = &pInputDataFrom[cSamples];
      do {
         const IntEbmType inputData = *pInputDataFrom;
         if(inputData < 0) {
            LOG_0(TraceLevelError, "ERROR DataSetInteraction::ConstructInputData inputData value cannot be negative");
            goto free_all;
         }
         if(IsConvertError<StorageDataType>(inputData)) {
            LOG_0(TraceLevelError, "ERROR DataSetInteraction::ConstructInputData inputData value too big to reference memory");
            goto free_all;
         }
         if(IsConvertError<size_t>(inputData)) {
            LOG_0(TraceLevelError, "ERROR DataSetInteraction::ConstructInputData inputData value too big to reference memory");
            goto free_all;
         }
         const size_t iData = static_cast<size_t>(inputData);
         if(pFeature->GetCountBins() <= iData) {
            LOG_0(TraceLevelError, "ERROR DataSetInteraction::ConstructInputData iData value must be less than the number of bins");
            goto free_all;
         }
         *pInputDataTo = static_cast<StorageDataType>(inputData);
         ++pInputDataTo;
         ++pInputDataFrom;
      } while(pInputDataFromEnd != pInputDataFrom);

      ++pFeature;
   } while(pFeatureEnd != pFeature);

   LOG_0(TraceLevelInfo, "Exited DataSetInteraction::ConstructInputData");
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
void DataSetInteraction::Destruct() {
   LOG_0(TraceLevelInfo, "Entered DataSetInteraction::Destruct");

   free(m_aGradientsAndHessians);
   free(m_aWeights);
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

   LOG_0(TraceLevelInfo, "Exited DataSetInteraction::Destruct");
}
WARNING_POP

ErrorEbmType DataSetInteraction::Initialize(
   const bool bAllocateHessians,
   const size_t cFeatures,
   const Feature * const aFeatures, 
   const size_t cSamples, 
   const IntEbmType * const aBinnedData, 
   const FloatEbmType * const aWeights,
   const void * const aTargetData,
   const FloatEbmType * const aPredictorScores, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
) {
   EBM_ASSERT(nullptr == m_aGradientsAndHessians); // we expect to start with zeroed values
   EBM_ASSERT(nullptr == m_aaInputData); // we expect to start with zeroed values
   EBM_ASSERT(0 == m_cSamples); // we expect to start with zeroed values

   LOG_0(TraceLevelInfo, "Entered DataSetInteraction::Initialize");

   if(0 != cSamples) {
      // runtimeLearningTypeOrCountTargetClasses can only be zero if 
      // there are zero samples and we shouldn't get past this point
      EBM_ASSERT(0 != runtimeLearningTypeOrCountTargetClasses);

      // if cSamples is zero, then we don't need to allocate anything since we won't use them anyways

      // check our targets since we don't use them other than for initializing
      if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
         const IntEbmType * pTargetFrom = static_cast<const IntEbmType *>(aTargetData);
         const IntEbmType * const pTargetFromEnd = pTargetFrom + cSamples;
         const size_t countTargetClasses = static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses);
         do {
            const IntEbmType data = *pTargetFrom;
            if(data < 0) {
               LOG_0(TraceLevelError, "ERROR DataSetInteraction::Initialize target value cannot be negative");
               return Error_IllegalParamValue;
            }
            if(IsConvertError<StorageDataType>(data)) {
               LOG_0(TraceLevelError, "ERROR DataSetInteraction::Initialize data target too big to reference memory");
               return Error_IllegalParamValue;
            }
            if(IsConvertError<size_t>(data)) {
               LOG_0(TraceLevelError, "ERROR DataSetInteraction::Initialize data target too big to reference memory");
               return Error_IllegalParamValue;
            }
            const size_t iData = static_cast<size_t>(data);
            if(countTargetClasses <= iData) {
               LOG_0(TraceLevelError, "ERROR DataSetInteraction::Initialize target value larger than number of classes");
               return Error_IllegalParamValue;
            }
            ++pTargetFrom;
         } while(pTargetFromEnd != pTargetFrom);
      }

      EBM_ASSERT(nullptr == m_aWeights);
      m_weightTotal = static_cast<FloatEbmType>(cSamples);
      if(nullptr != aWeights) {
         if(IsMultiplyError(sizeof(*aWeights), cSamples)) {
            LOG_0(TraceLevelWarning,
               "WARNING DataSetInteraction::Initialize IsMultiplyError(sizeof(*aWeights), cSamples)");
            return Error_IllegalParamValue;
         }
         if(!CheckAllWeightsEqual(cSamples, aWeights)) {
            const FloatEbmType total = AddPositiveFloatsSafe(cSamples, aWeights);
            if(std::isnan(total) || std::isinf(total) || total <= FloatEbmType { 0 }) {
               LOG_0(TraceLevelWarning, "WARNING DataSetInteraction::Initialize std::isnan(total) || std::isinf(total) || total <= FloatEbmType { 0 }");
               return Error_UserParamValue;
            }
            // if they were all zero then we'd ignore the weights param.  If there are negative numbers it might add
            // to zero though so check it after checking for negative
            EBM_ASSERT(FloatEbmType { 0 } != total);
            m_weightTotal = total;

            const size_t cBytes = sizeof(*aWeights) * cSamples;
            FloatEbmType * aWeightInternal = static_cast<FloatEbmType *>(malloc(cBytes));
            if(UNLIKELY(nullptr == aWeightInternal)) {
               LOG_0(TraceLevelWarning, "WARNING DataSetInteraction::Initialize nullptr == pWeightInternal");
               return Error_OutOfMemory;
            }
            m_aWeights = aWeightInternal;
            memcpy(aWeightInternal, aWeights, cBytes);
         }
      }

      ErrorEbmType error = ConstructGradientsAndHessians(bAllocateHessians, cSamples, aTargetData, aPredictorScores, runtimeLearningTypeOrCountTargetClasses, &m_aGradientsAndHessians);
      if(Error_None != error) {
         // we should have already logged the failure
         return error;
      }

      if(0 != cFeatures) {
         StorageDataType ** const aaInputData = ConstructInputData(cFeatures, aFeatures, cSamples, aBinnedData);
         if(nullptr == aaInputData) {
            return Error_OutOfMemory;
         }
         m_aaInputData = aaInputData;
      }

      m_cSamples = cSamples;
   }
   m_cFeatures = cFeatures;

   LOG_0(TraceLevelInfo, "Exited DataSetInteraction::Initialize");
   return Error_None;
}

} // DEFINED_ZONE_NAME
