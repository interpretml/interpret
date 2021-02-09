// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // INLINE_ALWAYS & UNLIKLEY
#include "Logging.h" // EBM_ASSERT & LOG
#include "RandomStream.h" // our header didn't need the full definition, but we use the RandomStream in here, so we need it
#include "DataSetBoosting.h"
#include "SamplingSet.h"

SamplingSet * SamplingSet::GenerateSingleSamplingSet(
   RandomStream * const pRandomStream, 
   const DataFrameBoosting * const pOriginDataSet
) {
   LOG_0(TraceLevelVerbose, "Entered SamplingSet::GenerateSingleSamplingSet");

   EBM_ASSERT(nullptr != pRandomStream);
   EBM_ASSERT(nullptr != pOriginDataSet);

   const size_t cSamples = pOriginDataSet->GetCountSamples();
   EBM_ASSERT(0 < cSamples); // if there were no samples, we wouldn't be called

   size_t * const aCountOccurrences = EbmMalloc<size_t>(cSamples);
   if(nullptr == aCountOccurrences) {
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateSingleSamplingSet nullptr == aCountOccurrences");
      return nullptr;
   }
   for(size_t i = 0; i < cSamples; ++i) {
      aCountOccurrences[i] = size_t { 0 };
   }

   for(size_t iSample = 0; iSample < cSamples; ++iSample) {
      const size_t iCountOccurrences = pRandomStream->Next(cSamples);
      ++aCountOccurrences[iCountOccurrences];
   }

   SamplingSet * pRet = EbmMalloc<SamplingSet>();
   if(nullptr == pRet) {
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateSingleSamplingSet nullptr == pRet");
      free(aCountOccurrences);
      return nullptr;
   }

   pRet->m_pOriginDataSet = pOriginDataSet;
   pRet->m_aCountOccurrences = aCountOccurrences;

   LOG_0(TraceLevelVerbose, "Exited SamplingSet::GenerateSingleSamplingSet");
   return pRet;
}

SamplingSet * SamplingSet::GenerateFlatSamplingSet(const DataFrameBoosting * const pOriginDataSet) {
   LOG_0(TraceLevelInfo, "Entered SamplingSet::GenerateFlatSamplingSet");

   // TODO: someday eliminate the need for generating this flat set by specially handling the case of no internal bagging
   EBM_ASSERT(nullptr != pOriginDataSet);
   const size_t cSamples = pOriginDataSet->GetCountSamples();
   EBM_ASSERT(0 < cSamples); // if there were no samples, we wouldn't be called

   size_t * const aCountOccurrences = EbmMalloc<size_t>(cSamples);
   if(nullptr == aCountOccurrences) {
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateFlatSamplingSet nullptr == aCountOccurrences");
      return nullptr;
   }

   for(size_t iSample = 0; iSample < cSamples; ++iSample) {
      aCountOccurrences[iSample] = 1;
   }

   SamplingSet * pRet = EbmMalloc<SamplingSet>();
   if(nullptr == pRet) {
      LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateFlatSamplingSet nullptr == pRet");
      free(aCountOccurrences);
      return nullptr;
   }

   pRet->m_pOriginDataSet = pOriginDataSet;
   pRet->m_aCountOccurrences = aCountOccurrences;

   LOG_0(TraceLevelInfo, "Exited SamplingSet::GenerateFlatSamplingSet");
   return pRet;
}

WARNING_PUSH
WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
void SamplingSet::FreeSamplingSets(const size_t cSamplingSets, SamplingSet ** const apSamplingSets) {
   LOG_0(TraceLevelInfo, "Entered SamplingSet::FreeSamplingSets");
   if(LIKELY(nullptr != apSamplingSets)) {
      const size_t cSamplingSetsAfterZero = 0 == cSamplingSets ? 1 : cSamplingSets;
      for(size_t iSamplingSet = 0; iSamplingSet < cSamplingSetsAfterZero; ++iSamplingSet) {
         if(nullptr != apSamplingSets[iSamplingSet]) {
            free(apSamplingSets[iSamplingSet]->m_aCountOccurrences);
            free(apSamplingSets[iSamplingSet]);
         }
      }
      free(apSamplingSets);
   }
   LOG_0(TraceLevelInfo, "Exited SamplingSet::FreeSamplingSets");
}
WARNING_POP

SamplingSet ** SamplingSet::GenerateSamplingSets(
   RandomStream * const pRandomStream, 
   const DataFrameBoosting * const pOriginDataSet, 
   const size_t cSamplingSets
) {
   LOG_0(TraceLevelInfo, "Entered SamplingSet::GenerateSamplingSets");

   EBM_ASSERT(nullptr != pRandomStream);
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
      SamplingSet * const pSingleSamplingSet = GenerateFlatSamplingSet(pOriginDataSet);
      if(UNLIKELY(nullptr == pSingleSamplingSet)) {
         LOG_0(TraceLevelWarning, "WARNING SamplingSet::GenerateSamplingSets nullptr == pSingleSamplingSet");
         free(apSamplingSets);
         return nullptr;
      }
      apSamplingSets[0] = pSingleSamplingSet;
   } else {
      for(size_t iSamplingSet = 0; iSamplingSet < cSamplingSets; ++iSamplingSet) {
         SamplingSet * const pSingleSamplingSet = GenerateSingleSamplingSet(pRandomStream, pOriginDataSet);
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
