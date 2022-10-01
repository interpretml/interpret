// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "logging.h" // EBM_ASSERT

#include "ebm_internal.hpp" // AddPositiveFloatsSafeBig
#include "RandomDeterministic.hpp" // RandomDeterministic
#include "InnerBag.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

InnerBag * InnerBag::GenerateSingleInnerBag(
   RandomDeterministic * const pRng,
   const size_t cSamples,
   const FloatFast * const aWeights
) {
   LOG_0(Trace_Verbose, "Entered InnerBag::GenerateSingleInnerBag");

   EBM_ASSERT(nullptr != pRng);

   InnerBag * pRet = static_cast<InnerBag *>(malloc(sizeof(InnerBag)));
   if(nullptr == pRet) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag nullptr == pRet");
      return nullptr;
   }
   pRet->InitializeUnfailing();

   EBM_ASSERT(1 <= cSamples); // if there were no samples, we wouldn't be called

   if(IsMultiplyError(sizeof(size_t), cSamples)) {
      pRet->Free();
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag IsMultiplyError(sizeof(size_t), cSamples)");
      return nullptr;
   }
   const size_t cBytesCountOccurrencesCapacity = sizeof(size_t) * cSamples;
   size_t * const aCountOccurrences = static_cast<size_t *>(malloc(cBytesCountOccurrencesCapacity));
   if(nullptr == aCountOccurrences) {
      pRet->Free();
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag nullptr == aCountOccurrences");
      return nullptr;
   }
   pRet->m_aCountOccurrences = aCountOccurrences;

   if(IsMultiplyError(sizeof(FloatFast), cSamples)) {
      pRet->Free();
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag IsMultiplyError(sizeof(FloatFast), cSamples)");
      return nullptr;
   }
   FloatFast * const aWeightsInternal = static_cast<FloatFast *>(malloc(sizeof(FloatFast) * cSamples));
   if(nullptr == aWeightsInternal) {
      pRet->Free();
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag nullptr == aWeightsInternal");
      return nullptr;
   }
   pRet->m_aWeights = aWeightsInternal;

   memset(aCountOccurrences, 0, cBytesCountOccurrencesCapacity);

   // the compiler understands the internal state of this RNG and can locate its internal state into CPU registers
   RandomDeterministic cpuRng;
   cpuRng.Initialize(*pRng); // move the RNG from memory into CPU registers
   size_t iSample = 0;
   do {
      const size_t iCountOccurrences = cpuRng.NextFast(cSamples);
      ++aCountOccurrences[iCountOccurrences];
      ++iSample;
   } while(cSamples != iSample);
   pRng->Initialize(cpuRng); // move the RNG from the CPU registers back into memory

   const size_t * pCountOccurrences = aCountOccurrences;
   const size_t * const pCountOccurrencesEnd = &aCountOccurrences[cSamples];
   FloatFast * pWeightsInternal = aWeightsInternal;
   FloatBig total;
   if(nullptr == aWeights) {
      do {
         *pWeightsInternal = static_cast<FloatFast>(*pCountOccurrences);
         ++pWeightsInternal;
         ++pCountOccurrences;
      } while(pCountOccurrencesEnd != pCountOccurrences);
      total = static_cast<FloatBig>(cSamples);
#ifndef NDEBUG
      const FloatBig debugTotal = AddPositiveFloatsSafeBig(cSamples, aWeightsInternal);
      EBM_ASSERT(debugTotal * 0.999 <= total && total <= 1.0001 * debugTotal);
#endif // NDEBUG
   } else {
      const FloatFast * pWeight = aWeights;
      do {
         *pWeightsInternal = *pWeight * static_cast<FloatFast>(*pCountOccurrences);
         ++pWeight;
         ++pWeightsInternal;
         ++pCountOccurrences;
      } while(pCountOccurrencesEnd != pCountOccurrences);
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
   const size_t cSamples,
   const FloatFast * const aWeights
) {
   LOG_0(Trace_Info, "Entered InnerBag::GenerateFlatInnerBag");

   // TODO: someday eliminate the need for generating this flat set by specially handling the case of no internal bagging

   InnerBag * const pRet = static_cast<InnerBag *>(malloc(sizeof(InnerBag)));
   if(nullptr == pRet) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateFlatInnerBag nullptr == pRet");
      return nullptr;
   }
   pRet->InitializeUnfailing();

   EBM_ASSERT(1 <= cSamples); // if there were no samples, we wouldn't be called

   if(IsMultiplyError(sizeof(size_t), cSamples)) {
      pRet->Free();
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateFlatInnerBag IsMultiplyError(sizeof(size_t), cSamples)");
      return nullptr;
   }
   size_t * const aCountOccurrences = static_cast<size_t *>(malloc(sizeof(size_t) * cSamples));
   if(nullptr == aCountOccurrences) {
      pRet->Free();
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateFlatInnerBag nullptr == aCountOccurrences");
      return nullptr;
   }
   pRet->m_aCountOccurrences = aCountOccurrences;

   if(IsMultiplyError(sizeof(FloatFast), cSamples)) {
      pRet->Free();
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateFlatInnerBag IsMultiplyError(sizeof(FloatFast), cSamples)");
      return nullptr;
   }
   const size_t cBytesWeightsInternalCapacity = sizeof(FloatFast) * cSamples;
   FloatFast * const aWeightsInternal = static_cast<FloatFast *>(malloc(cBytesWeightsInternalCapacity));
   if(nullptr == aWeightsInternal) {
      pRet->Free();
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateFlatInnerBag nullptr == aWeightsInternal");
      return nullptr;
   }
   pRet->m_aWeights = aWeightsInternal;

   size_t * pCountOccurrences = aCountOccurrences;
   const size_t * const pCountOccurrencesEnd = &aCountOccurrences[cSamples];
   FloatBig total;
   if(nullptr == aWeights) {
      FloatFast * pWeightsInternal = aWeightsInternal;
      do {
         *pWeightsInternal = 1;
         *pCountOccurrences = 1;

         ++pWeightsInternal;
         ++pCountOccurrences;
      } while(pCountOccurrencesEnd != pCountOccurrences);
      total = static_cast<FloatBig>(cSamples);
   } else {
      total = AddPositiveFloatsSafeBig(cSamples, aWeights);
      if(std::isnan(total) || std::isinf(total) || total <= 0) {
         pRet->Free();
         LOG_0(Trace_Warning, "WARNING InnerBag::GenerateFlatInnerBag std::isnan(total) || std::isinf(total) || total <= 0");
         return nullptr;
      }
      memcpy(aWeightsInternal, aWeights, cBytesWeightsInternalCapacity);
      do {
         *pCountOccurrences = 1;
         ++pCountOccurrences;
      } while(pCountOccurrencesEnd != pCountOccurrences);
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
      const size_t cInnerBagsAfterZero = size_t { 0 } == cInnerBags ? size_t { 1 } : cInnerBags;
      size_t iInnerBag = 0;
      do {
         InnerBag * const pInnerBag = apInnerBags[iInnerBag];
         if(nullptr != pInnerBag) {
            pInnerBag->Free();
         }
         ++iInnerBag;
      } while(cInnerBagsAfterZero != iInnerBag);
      free(apInnerBags);
   }
   LOG_0(Trace_Info, "Exited InnerBag::FreeInnerBags");
}
WARNING_POP

InnerBag ** InnerBag::GenerateInnerBags(
   RandomDeterministic * const pRng,
   const size_t cSamples,
   const FloatFast * const aWeights,
   const size_t cInnerBags
) {
   LOG_0(Trace_Info, "Entered InnerBag::GenerateInnerBags");

   EBM_ASSERT(nullptr != pRng);

   const size_t cInnerBagsAfterZero = size_t { 0 } == cInnerBags ? size_t { 1 } : cInnerBags;

   if(IsMultiplyError(sizeof(InnerBag *), cInnerBagsAfterZero)) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateInnerBags IsMultiplyError(sizeof(InnerBag *), cInnerBagsAfterZero)");
      return nullptr;
   }
   InnerBag ** apInnerBags = static_cast<InnerBag **>(malloc(sizeof(InnerBag *) * cInnerBagsAfterZero));
   if(UNLIKELY(nullptr == apInnerBags)) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateInnerBags nullptr == apInnerBags");
      return nullptr;
   }

   InnerBag ** ppInnerBagInit = apInnerBags;
   const InnerBag * const * const ppInnerBagsEnd = &apInnerBags[cInnerBagsAfterZero];
   do {
      *ppInnerBagInit = nullptr;
      ++ppInnerBagInit;
   } while(ppInnerBagsEnd != ppInnerBagInit);

   if(size_t { 0 } == cInnerBags) {
      // zero is a special value that really means allocate one set that contains all samples.
      InnerBag * const pSingleInnerBag = GenerateFlatInnerBag(cSamples, aWeights);
      if(UNLIKELY(nullptr == pSingleInnerBag)) {
         LOG_0(Trace_Warning, "WARNING InnerBag::GenerateInnerBags nullptr == pSingleInnerBag");
         free(apInnerBags);
         return nullptr;
      }
      apInnerBags[0] = pSingleInnerBag;
   } else {
      InnerBag ** ppInnerBag = apInnerBags;
      do {
         InnerBag * const pSingleInnerBag = GenerateSingleInnerBag(pRng, cSamples, aWeights);
         if(UNLIKELY(nullptr == pSingleInnerBag)) {
            LOG_0(Trace_Warning, "WARNING InnerBag::GenerateInnerBags nullptr == pSingleInnerBag");
            FreeInnerBags(cInnerBags, apInnerBags);
            return nullptr;
         }
         *ppInnerBag = pSingleInnerBag;
         ++ppInnerBag;
      } while(ppInnerBagsEnd != ppInnerBag);
   }
   LOG_0(Trace_Info, "Exited InnerBag::GenerateInnerBags");
   return apInnerBags;
}

} // DEFINED_ZONE_NAME
