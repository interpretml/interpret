// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "EbmInternal.h"

#include "RandomStream.h" // our header didn't need the full definition, but we use the RandomStream in here, so we need it
#include "DataFrameBoosting.h"
#include "SamplingSet.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

SamplingSet * SamplingSet::GenerateSingleSamplingSet(
   RandomStream * const pRandomStream, 
   const DataFrameBoosting * const pOriginDataFrame,
   const FloatEbmType * const aWeights
) {
   LOG_0(TraceLevelVerbose, "Entered SamplingSet::GenerateSingleSamplingSet");

   EBM_ASSERT(nullptr != pRandomStream);
   EBM_ASSERT(nullptr != pOriginDataFrame);

   const size_t cSamples = pOriginDataFrame->GetCountSamples();
   EBM_ASSERT(0 < cSamples); // if there were no samples, we wouldn't be called

   size_t * const aCountOccurrences = EbmMalloc<size_t>(cSamples);
   if(nullptr == aCountOccurrences) {
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateSingleSamplingSet nullptr == aCountOccurrences");
      return nullptr;
   }
   FloatEbmType * const aWeightsInternal = EbmMalloc<FloatEbmType>(cSamples);
   if(nullptr == aWeightsInternal) {
      free(aCountOccurrences);
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateSingleSamplingSet nullptr == aWeightsInternal");
      return nullptr;
   }
   SamplingSet * pRet = EbmMalloc<SamplingSet>();
   if(nullptr == pRet) {
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateSingleSamplingSet nullptr == pRet");
      free(aWeightsInternal);
      free(aCountOccurrences);
      return nullptr;
   }

   for(size_t i = 0; i < cSamples; ++i) {
      aCountOccurrences[i] = size_t { 0 };
   }
   for(size_t iSample = 0; iSample < cSamples; ++iSample) {
      const size_t iCountOccurrences = pRandomStream->Next(cSamples);
      ++aCountOccurrences[iCountOccurrences];
   }

   for(size_t iSample = 0; iSample < cSamples; ++iSample) {
      FloatEbmType weight = static_cast<FloatEbmType>(aCountOccurrences[iSample]);
      if(nullptr != aWeights) {
         weight *= aWeights[iSample];
      }
      aWeightsInternal[iSample] = weight;
   }
   const FloatEbmType total = AddPositiveFloatsSafe(cSamples, aWeightsInternal);
   if(std::isnan(total) || std::isinf(total) || total < FloatEbmType { 0 }) {
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateSingleSamplingSet std::isnan(total) || std::isinf(total) || total < FloatEbmType { 0 }");
      free(aWeightsInternal);
      free(aCountOccurrences);
      free(pRet);
      return nullptr;
   }

   pRet->m_pOriginDataFrame = pOriginDataFrame;
   pRet->m_aCountOccurrences = aCountOccurrences;
   pRet->m_aWeights = aWeightsInternal;
   pRet->m_weightTotal = total;

   LOG_0(TraceLevelVerbose, "Exited SamplingSet::GenerateSingleSamplingSet");
   return pRet;
}

SamplingSet * SamplingSet::GenerateFlatSamplingSet(
   const DataFrameBoosting * const pOriginDataFrame,
   const FloatEbmType * const aWeights
) {
   LOG_0(TraceLevelInfo, "Entered SamplingSet::GenerateFlatSamplingSet");

   // TODO: someday eliminate the need for generating this flat set by specially handling the case of no internal bagging
   EBM_ASSERT(nullptr != pOriginDataFrame);
   const size_t cSamples = pOriginDataFrame->GetCountSamples();
   EBM_ASSERT(0 < cSamples); // if there were no samples, we wouldn't be called

   size_t * const aCountOccurrences = EbmMalloc<size_t>(cSamples);
   if(nullptr == aCountOccurrences) {
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateFlatSamplingSet nullptr == aCountOccurrences");
      return nullptr;
   }
   for(size_t iSample = 0; iSample < cSamples; ++iSample) {
      aCountOccurrences[iSample] = 1;
   }

   FloatEbmType * const aWeightsInternal = EbmMalloc<FloatEbmType>(cSamples);
   if(nullptr == aWeightsInternal) {
      free(aCountOccurrences);
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateFlatSamplingSet nullptr == aWeightsInternal");
      return nullptr;
   }
   if(nullptr == aWeights) {
      for(size_t iSample = 0; iSample < cSamples; ++iSample) {
         aWeightsInternal[iSample] = 1;
      }
   } else {
      for(size_t iSample = 0; iSample < cSamples; ++iSample) {
         aWeightsInternal[iSample] = aWeights[iSample];
      }
   }

   SamplingSet * pRet = EbmMalloc<SamplingSet>();
   if(nullptr == pRet) {
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateFlatSamplingSet nullptr == pRet");
      free(aWeightsInternal);
      free(aCountOccurrences);
      return nullptr;
   }

   const FloatEbmType total = AddPositiveFloatsSafe(cSamples, aWeightsInternal);
   if(std::isnan(total) || std::isinf(total) || total < FloatEbmType { 0 }) {
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateFlatSamplingSet std::isnan(total) || std::isinf(total) || total < FloatEbmType { 0 }");
      free(aWeightsInternal);
      free(aCountOccurrences);
      free(pRet);
      return nullptr;
   }

   pRet->m_pOriginDataFrame = pOriginDataFrame;
   pRet->m_aCountOccurrences = aCountOccurrences;
   pRet->m_aWeights = aWeightsInternal;
   pRet->m_weightTotal = total;

   LOG_0(TraceLevelInfo, "Exited SamplingSet::GenerateFlatSamplingSet");
   return pRet;
}

void SamplingSet::FreeSamplingSet(SamplingSet * const pSamplingSet) {
   LOG_0(TraceLevelInfo, "Entered SamplingSet::FreeSamplingSet");
   if(nullptr != pSamplingSet) {
      free(pSamplingSet->m_aCountOccurrences);
      free(pSamplingSet->m_aWeights);
      free(pSamplingSet);
   }
   LOG_0(TraceLevelInfo, "Exited SamplingSet::FreeSamplingSet");
}

WARNING_PUSH
WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
void SamplingSet::FreeSamplingSets(const size_t cSamplingSets, SamplingSet ** const apSamplingSets) {
   LOG_0(TraceLevelInfo, "Entered SamplingSet::FreeSamplingSets");
   if(LIKELY(nullptr != apSamplingSets)) {
      const size_t cSamplingSetsAfterZero = 0 == cSamplingSets ? 1 : cSamplingSets;
      for(size_t iSamplingSet = 0; iSamplingSet < cSamplingSetsAfterZero; ++iSamplingSet) {
         FreeSamplingSet(apSamplingSets[iSamplingSet]);
      }
      free(apSamplingSets);
   }
   LOG_0(TraceLevelInfo, "Exited SamplingSet::FreeSamplingSets");
}
WARNING_POP

SamplingSet ** SamplingSet::GenerateSamplingSets(
   RandomStream * const pRandomStream, 
   const DataFrameBoosting * const pOriginDataFrame, 
   const FloatEbmType * const aWeights,
   const size_t cSamplingSets
) {
   LOG_0(TraceLevelInfo, "Entered SamplingSet::GenerateSamplingSets");

   EBM_ASSERT(nullptr != pRandomStream);
   EBM_ASSERT(nullptr != pOriginDataFrame);

   const size_t cSamplingSetsAfterZero = 0 == cSamplingSets ? 1 : cSamplingSets;

   SamplingSet ** apSamplingSets = EbmMalloc<SamplingSet *>(cSamplingSetsAfterZero);
   if(UNLIKELY(nullptr == apSamplingSets)) {
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateSamplingSets nullptr == apSamplingSets");
      return nullptr;
   }
   for(size_t i = 0; i < cSamplingSetsAfterZero; ++i) {
      apSamplingSets[i] = nullptr;
   }

   if(0 == cSamplingSets) {
      // zero is a special value that really means allocate one set that contains all samples.
      SamplingSet * const pSingleSamplingSet = GenerateFlatSamplingSet(pOriginDataFrame, aWeights);
      if(UNLIKELY(nullptr == pSingleSamplingSet)) {
         LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateSamplingSets nullptr == pSingleSamplingSet");
         free(apSamplingSets);
         return nullptr;
      }
      apSamplingSets[0] = pSingleSamplingSet;
   } else {
      for(size_t iSamplingSet = 0; iSamplingSet < cSamplingSets; ++iSamplingSet) {
         SamplingSet * const pSingleSamplingSet = GenerateSingleSamplingSet(pRandomStream, pOriginDataFrame, aWeights);
         if(UNLIKELY(nullptr == pSingleSamplingSet)) {
            LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateSamplingSets nullptr == pSingleSamplingSet");
            FreeSamplingSets(cSamplingSets, apSamplingSets);
            return nullptr;
         }
         apSamplingSets[iSamplingSet] = pSingleSamplingSet;
      }
   }
   LOG_0(TraceLevelInfo, "Exited SamplingSet::GenerateSamplingSets");
   return apSamplingSets;
}

} // DEFINED_ZONE_NAME
