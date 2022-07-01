// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "RandomStream.hpp" // our header didn't need the full definition, but we use the RandomDeterministic in here, so we need it
#include "DataSetBoosting.hpp"
#include "SamplingSet.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

SamplingSet * SamplingSet::GenerateSingleSamplingSet(
   RandomDeterministic * const pRandomDeterministic,
   const DataSetBoosting * const pOriginDataSet,
   const FloatFast * const aWeights
) {
   LOG_0(TraceLevelVerbose, "Entered SamplingSet::GenerateSingleSamplingSet");

   EBM_ASSERT(nullptr != pRandomDeterministic);
   EBM_ASSERT(nullptr != pOriginDataSet);

   SamplingSet * pRet = EbmMalloc<SamplingSet>();
   if(nullptr == pRet) {
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateSingleSamplingSet nullptr == pRet");
      return nullptr;
   }
   pRet->InitializeUnfailing();

   const size_t cSamples = pOriginDataSet->GetCountSamples();
   EBM_ASSERT(0 < cSamples); // if there were no samples, we wouldn't be called

   size_t * const aCountOccurrences = EbmMalloc<size_t>(cSamples);
   if(nullptr == aCountOccurrences) {
      pRet->Free();
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateSingleSamplingSet nullptr == aCountOccurrences");
      return nullptr;
   }
   pRet->m_aCountOccurrences = aCountOccurrences;

   FloatFast * const aWeightsInternal = EbmMalloc<FloatFast>(cSamples);
   if(nullptr == aWeightsInternal) {
      pRet->Free();
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateSingleSamplingSet nullptr == aWeightsInternal");
      return nullptr;
   }
   pRet->m_aWeights = aWeightsInternal;

   for(size_t i = 0; i < cSamples; ++i) {
      // TODO: use memset
      aCountOccurrences[i] = size_t { 0 };
   }

   for(size_t iSample = 0; iSample < cSamples; ++iSample) {
      const size_t iCountOccurrences = pRandomDeterministic->NextFast(cSamples);
      ++aCountOccurrences[iCountOccurrences];
   }

   FloatBig total;
   if(nullptr == aWeights) {
      for(size_t iSample = 0; iSample < cSamples; ++iSample) {
         const FloatFast weight = static_cast<FloatFast>(aCountOccurrences[iSample]);
         aWeightsInternal[iSample] = weight;
      }
      total = static_cast<FloatBig>(cSamples);
#ifndef NDEBUG
      const FloatBig debugTotal = AddPositiveFloatsSafeBig(cSamples, aWeightsInternal);
      EBM_ASSERT(debugTotal * 0.999 <= total && total <= 1.0001 * debugTotal);
#endif // NDEBUG
   } else {
      for(size_t iSample = 0; iSample < cSamples; ++iSample) {
         FloatFast weight = static_cast<FloatFast>(aCountOccurrences[iSample]);
         weight *= aWeights[iSample];
         aWeightsInternal[iSample] = weight;
      }
      total = AddPositiveFloatsSafeBig(cSamples, aWeightsInternal);
      if(std::isnan(total) || std::isinf(total) || total <= 0) {
         pRet->Free();
         LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateSingleSamplingSet std::isnan(total) || std::isinf(total) || total <= 0");
         return nullptr;
      }
   }
   // if they were all zero then we'd ignore the weights param.  If there are negative numbers it might add
   // to zero though so check it after checking for negative
   EBM_ASSERT(0 != total);

   pRet->m_pOriginDataSet = pOriginDataSet;
   pRet->m_weightTotal = total;

   LOG_0(TraceLevelVerbose, "Exited SamplingSet::GenerateSingleSamplingSet");
   return pRet;
}

SamplingSet * SamplingSet::GenerateFlatSamplingSet(
   const DataSetBoosting * const pOriginDataSet,
   const FloatFast * const aWeights
) {
   LOG_0(TraceLevelInfo, "Entered SamplingSet::GenerateFlatSamplingSet");

   // TODO: someday eliminate the need for generating this flat set by specially handling the case of no internal bagging
   EBM_ASSERT(nullptr != pOriginDataSet);

   SamplingSet * const pRet = EbmMalloc<SamplingSet>();
   if(nullptr == pRet) {
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateFlatSamplingSet nullptr == pRet");
      return nullptr;
   }
   pRet->InitializeUnfailing();

   const size_t cSamples = pOriginDataSet->GetCountSamples();
   EBM_ASSERT(0 < cSamples); // if there were no samples, we wouldn't be called

   size_t * const aCountOccurrences = EbmMalloc<size_t>(cSamples);
   if(nullptr == aCountOccurrences) {
      pRet->Free();
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateFlatSamplingSet nullptr == aCountOccurrences");
      return nullptr;
   }
   pRet->m_aCountOccurrences = aCountOccurrences;

   FloatFast * const aWeightsInternal = EbmMalloc<FloatFast>(cSamples);
   if(nullptr == aWeightsInternal) {
      pRet->Free();
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateFlatSamplingSet nullptr == aWeightsInternal");
      return nullptr;
   }
   pRet->m_aWeights = aWeightsInternal;

   FloatBig total;
   if(nullptr == aWeights) {
      for(size_t iSample = 0; iSample < cSamples; ++iSample) {
         aCountOccurrences[iSample] = 1;
         aWeightsInternal[iSample] = 1;
      }
      total = static_cast<FloatBig>(cSamples);
   } else {
      total = AddPositiveFloatsSafeBig(cSamples, aWeights);
      if(std::isnan(total) || std::isinf(total) || total <= 0) {
         pRet->Free();
         LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateFlatSamplingSet std::isnan(total) || std::isinf(total) || total <= 0");
         return nullptr;
      }
      memcpy(aWeightsInternal, aWeights, sizeof(aWeights[0]) * cSamples);
      for(size_t iSample = 0; iSample < cSamples; ++iSample) {
         aCountOccurrences[iSample] = 1;
      }
   }
   // if they were all zero then we'd ignore the weights param.  If there are negative numbers it might add
   // to zero though so check it after checking for negative
   EBM_ASSERT(0 != total);

   pRet->m_pOriginDataSet = pOriginDataSet;
   pRet->m_weightTotal = total;

   LOG_0(TraceLevelInfo, "Exited SamplingSet::GenerateFlatSamplingSet");
   return pRet;
}

void SamplingSet::Free() {
   free(m_aCountOccurrences);
   free(m_aWeights);
   free(this);
}

void SamplingSet::InitializeUnfailing() {
   m_aCountOccurrences = nullptr;
   m_aWeights = nullptr;
}

WARNING_PUSH
WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
void SamplingSet::FreeSamplingSets(const size_t cSamplingSets, SamplingSet ** const apSamplingSets) {
   LOG_0(TraceLevelInfo, "Entered SamplingSet::FreeSamplingSets");
   if(LIKELY(nullptr != apSamplingSets)) {
      const size_t cSamplingSetsAfterZero = 0 == cSamplingSets ? 1 : cSamplingSets;
      for(size_t iSamplingSet = 0; iSamplingSet < cSamplingSetsAfterZero; ++iSamplingSet) {
         SamplingSet * const pSamplingSet = apSamplingSets[iSamplingSet];
         if(nullptr != pSamplingSet) {
            pSamplingSet->Free();
         }
      }
      free(apSamplingSets);
   }
   LOG_0(TraceLevelInfo, "Exited SamplingSet::FreeSamplingSets");
}
WARNING_POP

SamplingSet ** SamplingSet::GenerateSamplingSets(
   RandomDeterministic * const pRandomDeterministic,
   const DataSetBoosting * const pOriginDataSet, 
   const FloatFast * const aWeights,
   const size_t cSamplingSets
) {
   LOG_0(TraceLevelInfo, "Entered SamplingSet::GenerateSamplingSets");

   EBM_ASSERT(nullptr != pRandomDeterministic);
   EBM_ASSERT(nullptr != pOriginDataSet);

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
      SamplingSet * const pSingleSamplingSet = GenerateFlatSamplingSet(pOriginDataSet, aWeights);
      if(UNLIKELY(nullptr == pSingleSamplingSet)) {
         LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateSamplingSets nullptr == pSingleSamplingSet");
         free(apSamplingSets);
         return nullptr;
      }
      apSamplingSets[0] = pSingleSamplingSet;
   } else {
      for(size_t iSamplingSet = 0; iSamplingSet < cSamplingSets; ++iSamplingSet) {
         SamplingSet * const pSingleSamplingSet = GenerateSingleSamplingSet(pRandomDeterministic, pOriginDataSet, aWeights);
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
