// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <string.h> // memset
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // TML_INLINE & UNLIKLEY
#include "Logging.h" // EBM_ASSERT & LOG
#include "RandomStream.h" // our header didn't need the full definition, but we use the RandomStream in here, so we need it
#include "DataSetByFeatureCombination.h" // we use an iterator which requires a full definition.  TODO : in the future we'll be eliminating the iterator, so check back here to see if we can eliminate this include file
#include "SamplingWithReplacement.h"

SamplingWithReplacement::~SamplingWithReplacement() {
   LOG(TraceLevelInfo, "Entered ~SamplingWithReplacement");
   free(const_cast<size_t *>(m_aCountOccurrences));
   LOG(TraceLevelInfo, "Exited ~SamplingWithReplacement");
}

size_t SamplingWithReplacement::GetTotalCountCaseOccurrences() const {
      // for SamplingWithReplacement (bootstrap sampling), we have the same number of cases as our original dataset
   size_t cTotalCountCaseOccurrences = m_pOriginDataSet->GetCountCases();
#ifndef NDEBUG
   size_t cTotalCountCaseOccurrencesDebug = 0;
   for(size_t i = 0; i < m_pOriginDataSet->GetCountCases(); ++i) {
      cTotalCountCaseOccurrencesDebug += m_aCountOccurrences[i];
   }
   EBM_ASSERT(cTotalCountCaseOccurrencesDebug == cTotalCountCaseOccurrences);
#endif // NDEBUG
   return cTotalCountCaseOccurrences;
}

SamplingWithReplacement * SamplingWithReplacement::GenerateSingleSamplingSet(RandomStream * const pRandomStream, const DataSetByFeatureCombination * const pOriginDataSet) {
   LOG(TraceLevelVerbose, "Entered SamplingWithReplacement::GenerateSingleSamplingSet");

   EBM_ASSERT(nullptr != pRandomStream);
   EBM_ASSERT(nullptr != pOriginDataSet);

   const size_t cCases = pOriginDataSet->GetCountCases();
   EBM_ASSERT(0 < cCases); // if there were no cases, we wouldn't be called

   if(IsMultiplyError(sizeof(size_t), cCases)) {
      LOG(TraceLevelWarning, "WARNING SamplingWithReplacement::GenerateSingleSamplingSet IsMultiplyError(sizeof(size_t), cCases)");
      return nullptr;
   }
   const size_t cBytesData = sizeof(size_t) * cCases;
   size_t * const aCountOccurrences = static_cast<size_t *>(malloc(cBytesData));
   if(nullptr == aCountOccurrences) {
      LOG(TraceLevelWarning, "WARNING SamplingWithReplacement::GenerateSingleSamplingSet nullptr == aCountOccurrences");
      return nullptr;
   }

   memset(aCountOccurrences, 0, cBytesData);

   try {
      for(size_t iCase = 0; iCase < cCases; ++iCase) {
         const size_t iCountOccurrences = pRandomStream->Next(size_t { 0 }, cCases - 1);
         ++aCountOccurrences[iCountOccurrences];
      }
   } catch(...) {
      // Next could in theory throw an exception
      LOG(TraceLevelWarning, "WARNING SamplingWithReplacement::GenerateSingleSamplingSet exception");
      free(aCountOccurrences);
      return nullptr;
   }

   SamplingWithReplacement * pRet = new (std::nothrow) SamplingWithReplacement(pOriginDataSet, aCountOccurrences);
   if(nullptr == pRet) {
      LOG(TraceLevelWarning, "WARNING SamplingWithReplacement::GenerateSingleSamplingSet nullptr == pRet");
      free(aCountOccurrences);
      return nullptr;
   }

   LOG(TraceLevelVerbose, "Exited SamplingWithReplacement::GenerateSingleSamplingSet");
   return pRet;
}

SamplingWithReplacement * SamplingWithReplacement::GenerateFlatSamplingSet(const DataSetByFeatureCombination * const pOriginDataSet) {
   LOG(TraceLevelInfo, "Entered SamplingWithReplacement::GenerateFlatSamplingSet");

   // TODO: someday eliminate the need for generating this flat set by specially handling the case of no internal bagging
   EBM_ASSERT(nullptr != pOriginDataSet);
   const size_t cCases = pOriginDataSet->GetCountCases();
   EBM_ASSERT(0 < cCases); // if there were no cases, we wouldn't be called

   const size_t cBytesData = sizeof(size_t) * cCases;
   size_t * const aCountOccurrences = static_cast<size_t *>(malloc(cBytesData));
   if(nullptr == aCountOccurrences) {
      LOG(TraceLevelWarning, "WARNING SamplingWithReplacement::GenerateFlatSamplingSet nullptr == aCountOccurrences");
      return nullptr;
   }

   for(size_t iCase = 0; iCase < cCases; ++iCase) {
      aCountOccurrences[iCase] = 1;
   }

   SamplingWithReplacement * pRet = new (std::nothrow) SamplingWithReplacement(pOriginDataSet, aCountOccurrences);
   if(nullptr == pRet) {
      LOG(TraceLevelWarning, "WARNING SamplingWithReplacement::GenerateFlatSamplingSet nullptr == pRet");
      free(aCountOccurrences);
   }

   LOG(TraceLevelInfo, "Exited SamplingWithReplacement::GenerateFlatSamplingSet");
   return pRet;
}

void SamplingWithReplacement::FreeSamplingSets(const size_t cSamplingSets, SamplingMethod ** apSamplingSets) {
   LOG(TraceLevelInfo, "Entered SamplingWithReplacement::FreeSamplingSets");
   if(LIKELY(nullptr != apSamplingSets)) {
      const size_t cSamplingSetsAfterZero = 0 == cSamplingSets ? 1 : cSamplingSets;
      for(size_t iSamplingSet = 0; iSamplingSet < cSamplingSetsAfterZero; ++iSamplingSet) {
         delete apSamplingSets[iSamplingSet];
      }
      delete[] apSamplingSets;
   }
   LOG(TraceLevelInfo, "Exited SamplingWithReplacement::FreeSamplingSets");
}

SamplingMethod ** SamplingWithReplacement::GenerateSamplingSets(RandomStream * const pRandomStream, const DataSetByFeatureCombination * const pOriginDataSet, const size_t cSamplingSets) {
   LOG(TraceLevelInfo, "Entered SamplingWithReplacement::GenerateSamplingSets");

   EBM_ASSERT(nullptr != pRandomStream);
   EBM_ASSERT(nullptr != pOriginDataSet);

   const size_t cSamplingSetsAfterZero = 0 == cSamplingSets ? 1 : cSamplingSets;

   SamplingMethod ** apSamplingSets = new (std::nothrow) SamplingMethod *[cSamplingSetsAfterZero];
   if(UNLIKELY(nullptr == apSamplingSets)) {
      LOG(TraceLevelWarning, "WARNING SamplingWithReplacement::GenerateSamplingSets nullptr == apSamplingSets");
      return nullptr;
   }
   if(0 == cSamplingSets) {
      SamplingWithReplacement * const pSingleSamplingSet = GenerateFlatSamplingSet(pOriginDataSet);
      if(UNLIKELY(nullptr == pSingleSamplingSet)) {
         LOG(TraceLevelWarning, "WARNING SamplingWithReplacement::GenerateSamplingSets nullptr == pSingleSamplingSet");
         free(apSamplingSets);
         return nullptr;
      }
      apSamplingSets[0] = pSingleSamplingSet;
   } else {
      memset(apSamplingSets, 0, sizeof(*apSamplingSets) * cSamplingSets);
      for(size_t iSamplingSet = 0; iSamplingSet < cSamplingSets; ++iSamplingSet) {
         SamplingWithReplacement * const pSingleSamplingSet = GenerateSingleSamplingSet(pRandomStream, pOriginDataSet);
         if(UNLIKELY(nullptr == pSingleSamplingSet)) {
            LOG(TraceLevelWarning, "WARNING SamplingWithReplacement::GenerateSamplingSets nullptr == pSingleSamplingSet");
            FreeSamplingSets(cSamplingSets, apSamplingSets);
            return nullptr;
         }
         apSamplingSets[iSamplingSet] = pSingleSamplingSet;
      }
   }
   LOG(TraceLevelInfo, "Exited SamplingWithReplacement::GenerateSamplingSets");
   return apSamplingSets;
}