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


// TODO: remove data_set_shared.hpp and ebm_stats.hpp
#include "data_set_shared.hpp"
#include "ebm_stats.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern ErrorEbmType InitializeGradientsAndHessians(
   const unsigned char * const pDataSetShared,
   const BagEbmType direction,
   const BagEbmType * const aBag,
   const double * const aInitScores,
   const size_t cSetSamples,
   FloatFast * const aGradientAndHessian
);

extern ErrorEbmType ExtractWeights(
   const unsigned char * const pDataSetShared,
   const BagEbmType direction,
   const size_t cAllSamples,
   const BagEbmType * const aBag,
   const size_t cSetSamples,
   FloatFast ** ppWeightsOut
);

INLINE_RELEASE_UNTEMPLATED static ErrorEbmType ConstructGradientsAndHessians(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const bool bAllocateHessians,
   const unsigned char * const pDataSetShared,
   const BagEbmType * const aBag,
   const double * const aInitScores,
   const size_t cSetSamples,
   FloatFast ** paGradientsAndHessiansOut
) {
   LOG_0(TraceLevelInfo, "Entered ConstructGradientsAndHessians");

   // runtimeLearningTypeOrCountTargetClasses can only be zero if there are zero samples and we shouldn't get here
   EBM_ASSERT(0 != runtimeLearningTypeOrCountTargetClasses);
   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(1 <= cSetSamples);
   EBM_ASSERT(nullptr != paGradientsAndHessiansOut);
   EBM_ASSERT(nullptr == *paGradientsAndHessiansOut);

   ErrorEbmType error;

   const size_t cScores = GetCountScores(runtimeLearningTypeOrCountTargetClasses);
   EBM_ASSERT(1 <= cScores);

   const size_t cStorageItems = bAllocateHessians ? 2 : 1;
   if(IsMultiplyError(cScores, cStorageItems, cSetSamples)) {
      LOG_0(TraceLevelWarning, "WARNING ConstructGradientsAndHessians IsMultiplyError(cScores, cStorageItems, cSamples)");
      return Error_OutOfMemory;
   }
   const size_t cElements = cScores * cStorageItems * cSetSamples;

   FloatFast * aGradientsAndHessians = EbmMalloc<FloatFast>(cElements);
   if(UNLIKELY(nullptr == aGradientsAndHessians)) {
      LOG_0(TraceLevelWarning, "WARNING ConstructGradientsAndHessians nullptr == aGradientsAndHessians");
      return Error_OutOfMemory;
   }
   *paGradientsAndHessiansOut = aGradientsAndHessians; // transfer ownership for future deletion

   error = InitializeGradientsAndHessians(
      pDataSetShared,
      BagEbmType { 1 },
      aBag,
      aInitScores,
      cSetSamples,
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
   const unsigned char * const pDataSetShared,
   const BagEbmType * const aBag,
   const size_t cSetSamples,
   const size_t cFeatures
) {
   LOG_0(TraceLevelInfo, "Entered DataSetInteraction::ConstructInputData");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(0 < cSetSamples);
   EBM_ASSERT(0 < cFeatures);

   StorageDataType ** const aaInputDataTo = EbmMalloc<StorageDataType *>(cFeatures);
   if(nullptr == aaInputDataTo) {
      LOG_0(TraceLevelWarning, "WARNING DataSetInteraction::ConstructInputData nullptr == aaInputDataTo");
      return nullptr;
   }

   size_t iFeature = 0;
   do {
      StorageDataType * pInputDataTo = EbmMalloc<StorageDataType>(cSetSamples);
      if(nullptr == pInputDataTo) {
         LOG_0(TraceLevelWarning, "WARNING DataSetInteraction::ConstructInputData nullptr == pInputDataTo");
         goto free_all;
      }
      aaInputDataTo[iFeature] = pInputDataTo;

      size_t cBins;
      bool bMissing;
      bool bUnknown;
      bool bNominal;
      bool bSparse;
      SharedStorageDataType defaultValueSparse;
      size_t cNonDefaultsSparse;
      const void * aInputDataFrom = GetDataSetSharedFeature(
         pDataSetShared,
         iFeature,
         &cBins,
         &bMissing,
         &bUnknown,
         &bNominal,
         &bSparse,
         &defaultValueSparse,
         &cNonDefaultsSparse
      );
      EBM_ASSERT(nullptr != aInputDataFrom);
      EBM_ASSERT(!bSparse); // we don't support sparse yet

      ++iFeature;

      const BagEbmType * pBag = aBag;
      BagEbmType countBagged = 0;
      size_t iData = 0;

      const SharedStorageDataType * pInputDataFrom = static_cast<const SharedStorageDataType *>(aInputDataFrom);
      const StorageDataType * pInputDataToEnd = &pInputDataTo[cSetSamples];
      do {
         while(countBagged <= BagEbmType { 0 }) {
            const SharedStorageDataType inputData = *pInputDataFrom;
            ++pInputDataFrom;

            EBM_ASSERT(!IsConvertError<size_t>(inputData));
            iData = static_cast<size_t>(inputData);
            if(cBins <= iData) {
               LOG_0(TraceLevelError, "ERROR DataSetInteraction::ConstructInputData iData value must be less than the number of bins");
               goto free_all;
            }

            countBagged = 1;
            if(nullptr != pBag) {
               countBagged = *pBag;
               ++pBag;
            }
         }
         EBM_ASSERT(0 < countBagged);
         --countBagged;

         if(IsConvertError<StorageDataType>(iData)) {
            // we can remove this check once we get into bit packing this since we'll have checked it beforehand
            LOG_0(TraceLevelError, "ERROR DataSetInteraction::ConstructInputData iData value too big to reference memory");
            goto free_all;
         }

         *pInputDataTo = static_cast<StorageDataType>(iData);
         ++pInputDataTo;
      } while(pInputDataToEnd != pInputDataTo);
      EBM_ASSERT(0 == countBagged);
   } while(cFeatures != iFeature);

   LOG_0(TraceLevelInfo, "Exited DataSetInteraction::ConstructInputData");
   return aaInputDataTo;

free_all:
   while(0 != iFeature) {
      --iFeature;
      free(aaInputDataTo[iFeature]);
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
   const unsigned char * const pDataSetShared,
   const size_t cAllSamples,
   const BagEbmType * const aBag,
   const double * const aInitScores,
   const size_t cSetSamples,
   const size_t cWeights,
   const size_t cFeatures
) {
   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(cSetSamples <= cAllSamples);

   EBM_ASSERT(nullptr == m_aGradientsAndHessians); // we expect to start with zeroed values
   EBM_ASSERT(nullptr == m_aaInputData); // we expect to start with zeroed values
   EBM_ASSERT(0 == m_cSamples); // we expect to start with zeroed values

   LOG_0(TraceLevelInfo, "Entered DataSetInteraction::Initialize");

   ErrorEbmType error;

   ptrdiff_t runtimeLearningTypeOrCountTargetClasses;
   GetDataSetSharedTarget(pDataSetShared, 0, &runtimeLearningTypeOrCountTargetClasses);

   if(0 != cSetSamples) {
      // runtimeLearningTypeOrCountTargetClasses can only be zero if 
      // there are zero samples and we shouldn't get past this point
      EBM_ASSERT(0 != runtimeLearningTypeOrCountTargetClasses);

      // if cSamples is zero, then we don't need to allocate anything since we won't use them anyways

      EBM_ASSERT(nullptr == m_aWeights);
      m_weightTotal = static_cast<FloatBig>(cSetSamples);
      if(0 != cWeights) {
         error = ExtractWeights(
            pDataSetShared,
            BagEbmType { 1 },
            cAllSamples,
            aBag,
            cSetSamples,
            &m_aWeights
         );
         if(Error_None != error) {
            // error already logged
            return error;
         }
         if(nullptr != m_aWeights) {
            const FloatBig total = AddPositiveFloatsSafeBig(cSetSamples, m_aWeights);
            if(std::isnan(total) || std::isinf(total) || total <= 0) {
               LOG_0(TraceLevelWarning, "WARNING DataSetInteraction::Initialize std::isnan(total) || std::isinf(total) || total <= 0");
               return Error_UserParamValue;
            }
            // if they were all zero then we'd ignore the weights param.  If there are negative numbers it might add
            // to zero though so check it after checking for negative
            EBM_ASSERT(0 != total);
            m_weightTotal = total;
         }
      }

      error = ConstructGradientsAndHessians(
         runtimeLearningTypeOrCountTargetClasses,
         bAllocateHessians,
         pDataSetShared,
         aBag,
         aInitScores,
         cSetSamples,
         &m_aGradientsAndHessians
      );
      if(Error_None != error) {
         // we should have already logged the failure
         return error;
      }

      if(0 != cFeatures) {
         StorageDataType ** const aaInputData = ConstructInputData(
            pDataSetShared,
            aBag,
            cSetSamples,
            cFeatures
         );
         if(nullptr == aaInputData) {
            return Error_OutOfMemory;
         }
         m_aaInputData = aaInputData;
      }

      m_cSamples = cSetSamples;
   }
   m_cFeatures = cFeatures;

   LOG_0(TraceLevelInfo, "Exited DataSetInteraction::Initialize");
   return Error_None;
}

} // DEFINED_ZONE_NAME
