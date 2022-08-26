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

#include "RandomDeterministic.hpp" // our header didn't need the full definition, but we use the RandomDeterministic in here, so we need it
#include "DataSetBoosting.hpp"
#include "InnerBag.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

InnerBag * InnerBag::GenerateSingleInnerBag(
   RandomDeterministic * const pRng,
   const DataSetBoosting * const pOriginDataSet,
   const FloatFast * const aWeights
) {
   LOG_0(Trace_Verbose, "Entered InnerBag::GenerateSingleInnerBag");

   EBM_ASSERT(nullptr != pRng);
   EBM_ASSERT(nullptr != pOriginDataSet);

   InnerBag * pRet = EbmMalloc<InnerBag>();
   if(nullptr == pRet) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag nullptr == pRet");
      return nullptr;
   }
   pRet->InitializeUnfailing();

   const size_t cSamples = pOriginDataSet->GetCountSamples();
   EBM_ASSERT(0 < cSamples); // if there were no samples, we wouldn't be called

   size_t * const aCountOccurrences = EbmMalloc<size_t>(cSamples);
   if(nullptr == aCountOccurrences) {
      pRet->Free();
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag nullptr == aCountOccurrences");
      return nullptr;
   }
   pRet->m_aCountOccurrences = aCountOccurrences;

   FloatFast * const aWeightsInternal = EbmMalloc<FloatFast>(cSamples);
   if(nullptr == aWeightsInternal) {
      pRet->Free();
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag nullptr == aWeightsInternal");
      return nullptr;
   }
   pRet->m_aWeights = aWeightsInternal;

   for(size_t i = 0; i < cSamples; ++i) {
      // TODO: use memset
      aCountOccurrences[i] = size_t { 0 };
   }

   // the compiler understands the internal state of this RNG and can locate its internal state into CPU registers
   RandomDeterministic cpuRng;
   cpuRng.Initialize(*pRng); // move the RNG from memory into CPU registers
   for(size_t iSample = 0; iSample < cSamples; ++iSample) {
      const size_t iCountOccurrences = cpuRng.NextFast(cSamples);
      ++aCountOccurrences[iCountOccurrences];
   }
   pRng->Initialize(cpuRng); // move the RNG from the CPU registers back into memory

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
         LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag std::isnan(total) || std::isinf(total) || total <= 0");
         return nullptr;
      }
   }
   // if they were all zero then we'd ignore the weights param.  If there are negative numbers it might add
   // to zero though so check it after checking for negative
   EBM_ASSERT(0 != total);

   pRet->m_weightTotal = total;

   LOG_0(Trace_Verbose, "Exited InnerBag::GenerateSingleInnerBag");
   return pRet;
}

InnerBag * InnerBag::GenerateFlatInnerBag(
   const DataSetBoosting * const pOriginDataSet,
   const FloatFast * const aWeights
) {
   LOG_0(Trace_Info, "Entered InnerBag::GenerateFlatInnerBag");

   // TODO: someday eliminate the need for generating this flat set by specially handling the case of no internal bagging
   EBM_ASSERT(nullptr != pOriginDataSet);

   InnerBag * const pRet = EbmMalloc<InnerBag>();
   if(nullptr == pRet) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateFlatInnerBag nullptr == pRet");
      return nullptr;
   }
   pRet->InitializeUnfailing();

   const size_t cSamples = pOriginDataSet->GetCountSamples();
   EBM_ASSERT(0 < cSamples); // if there were no samples, we wouldn't be called

   size_t * const aCountOccurrences = EbmMalloc<size_t>(cSamples);
   if(nullptr == aCountOccurrences) {
      pRet->Free();
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateFlatInnerBag nullptr == aCountOccurrences");
      return nullptr;
   }
   pRet->m_aCountOccurrences = aCountOccurrences;

   FloatFast * const aWeightsInternal = EbmMalloc<FloatFast>(cSamples);
   if(nullptr == aWeightsInternal) {
      pRet->Free();
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateFlatInnerBag nullptr == aWeightsInternal");
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
         LOG_0(Trace_Warning, "WARNING InnerBag::GenerateFlatInnerBag std::isnan(total) || std::isinf(total) || total <= 0");
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

   pRet->m_weightTotal = total;

   LOG_0(Trace_Info, "Exited InnerBag::GenerateFlatInnerBag");
   return pRet;
}

void InnerBag::Free() {
   free(m_aCountOccurrences);
   free(m_aWeights);
   free(this);
}

void InnerBag::InitializeUnfailing() {
   m_aCountOccurrences = nullptr;
   m_aWeights = nullptr;
}

WARNING_PUSH
WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
void InnerBag::FreeInnerBags(const size_t cInnerBags, InnerBag ** const apInnerBags) {
   LOG_0(Trace_Info, "Entered InnerBag::FreeInnerBags");
   if(LIKELY(nullptr != apInnerBags)) {
      const size_t cInnerBagsAfterZero = 0 == cInnerBags ? size_t { 1 } : cInnerBags;
      for(size_t iInnerBag = 0; iInnerBag < cInnerBagsAfterZero; ++iInnerBag) {
         InnerBag * const pInnerBag = apInnerBags[iInnerBag];
         if(nullptr != pInnerBag) {
            pInnerBag->Free();
         }
      }
      free(apInnerBags);
   }
   LOG_0(Trace_Info, "Exited InnerBag::FreeInnerBags");
}
WARNING_POP

InnerBag ** InnerBag::GenerateInnerBags(
   RandomDeterministic * const pRng,
   const DataSetBoosting * const pOriginDataSet, 
   const FloatFast * const aWeights,
   const size_t cInnerBags
) {
   LOG_0(Trace_Info, "Entered InnerBag::GenerateInnerBags");

   EBM_ASSERT(nullptr != pRng);
   EBM_ASSERT(nullptr != pOriginDataSet);

   const size_t cInnerBagsAfterZero = 0 == cInnerBags ? size_t { 1 } : cInnerBags;

   InnerBag ** apInnerBags = EbmMalloc<InnerBag *>(cInnerBagsAfterZero);
   if(UNLIKELY(nullptr == apInnerBags)) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateInnerBags nullptr == apInnerBags");
      return nullptr;
   }
   for(size_t i = 0; i < cInnerBagsAfterZero; ++i) {
      apInnerBags[i] = nullptr;
   }

   if(0 == cInnerBags) {
      // zero is a special value that really means allocate one set that contains all samples.
      InnerBag * const pSingleInnerBag = GenerateFlatInnerBag(pOriginDataSet, aWeights);
      if(UNLIKELY(nullptr == pSingleInnerBag)) {
         LOG_0(Trace_Warning, "WARNING InnerBag::GenerateInnerBags nullptr == pSingleInnerBag");
         free(apInnerBags);
         return nullptr;
      }
      apInnerBags[0] = pSingleInnerBag;
   } else {
      for(size_t iInnerBag = 0; iInnerBag < cInnerBags; ++iInnerBag) {
         InnerBag * const pSingleInnerBag = GenerateSingleInnerBag(pRng, pOriginDataSet, aWeights);
         if(UNLIKELY(nullptr == pSingleInnerBag)) {
            LOG_0(Trace_Warning, "WARNING InnerBag::GenerateInnerBags nullptr == pSingleInnerBag");
            FreeInnerBags(cInnerBags, apInnerBags);
            return nullptr;
         }
         apInnerBags[iInnerBag] = pSingleInnerBag;
      }
   }
   LOG_0(Trace_Info, "Exited InnerBag::GenerateInnerBags");
   return apInnerBags;
}

} // DEFINED_ZONE_NAME
