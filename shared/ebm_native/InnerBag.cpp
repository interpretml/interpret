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
#include "RandomNondeterministic.hpp" // RandomNondeterministic
#include "InnerBag.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

ErrorEbm InnerBag::GenerateSingleInnerBag(
   void * const rng,
   const size_t cSamples,
   const FloatFast * const aWeights,
   InnerBag ** const ppOut
) {
   LOG_0(Trace_Verbose, "Entered InnerBag::GenerateSingleInnerBag");

   EBM_ASSERT(nullptr != ppOut);
   EBM_ASSERT(nullptr == *ppOut);

   InnerBag * pRet = static_cast<InnerBag *>(malloc(sizeof(InnerBag)));
   if(nullptr == pRet) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag nullptr == pRet");
      return Error_OutOfMemory;
   }
   pRet->InitializeUnfailing();
   *ppOut = pRet;

   EBM_ASSERT(1 <= cSamples); // if there were no samples, we wouldn't be called

   if(IsMultiplyError(sizeof(size_t), cSamples)) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag IsMultiplyError(sizeof(size_t), cSamples)");
      return Error_OutOfMemory;
   }
   const size_t cBytesCountOccurrencesCapacity = sizeof(size_t) * cSamples;
   size_t * const aCountOccurrences = static_cast<size_t *>(malloc(cBytesCountOccurrencesCapacity));
   if(nullptr == aCountOccurrences) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag nullptr == aCountOccurrences");
      return Error_OutOfMemory;
   }
   pRet->m_aCountOccurrences = aCountOccurrences;

   if(IsMultiplyError(sizeof(FloatFast), cSamples)) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag IsMultiplyError(sizeof(FloatFast), cSamples)");
      return Error_OutOfMemory;
   }
   FloatFast * const aWeightsInternal = static_cast<FloatFast *>(malloc(sizeof(FloatFast) * cSamples));
   if(nullptr == aWeightsInternal) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag nullptr == aWeightsInternal");
      return Error_OutOfMemory;
   }
   pRet->m_aWeights = aWeightsInternal;

   memset(aCountOccurrences, 0, cBytesCountOccurrencesCapacity);

   // the compiler understands the internal state of this RNG and can locate its internal state into CPU registers
   RandomDeterministic cpuRng;
   if(nullptr == rng) {
      // Inner bags are not used when building a differentially private model, so
      // we can use low-quality non-determinism.  Generate a non-deterministic seed
      uint64_t seed;
      try {
         RandomNondeterministic<uint64_t> randomGenerator;
         seed = randomGenerator.Next(std::numeric_limits<uint64_t>::max());
      } catch(const std::bad_alloc &) {
         LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag Out of memory in std::random_device");
         return Error_OutOfMemory;
      } catch(...) {
         LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag Unknown error in std::random_device");
         return Error_UnexpectedInternal;
      }
      cpuRng.Initialize(seed);
   } else {
      const RandomDeterministic * const pRng = reinterpret_cast<RandomDeterministic *>(rng);
      cpuRng.Initialize(*pRng); // move the RNG from memory into CPU registers
   }

   size_t iSample = 0;
   do {
      const size_t iCountOccurrences = cpuRng.NextFast(cSamples);
      ++aCountOccurrences[iCountOccurrences];
      ++iSample;
   } while(cSamples != iSample);

   if(nullptr != rng) {
      RandomDeterministic * pRng = reinterpret_cast<RandomDeterministic *>(rng);
      pRng->Initialize(cpuRng); // move the RNG from memory into CPU registers
   }

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
         LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag std::isnan(total) || std::isinf(total) || total <= 0");
         return Error_UserParamVal;
      }
   }
   // if they were all zero then we'd ignore the weights param.  If there are negative numbers it might add
   // to zero though so check it after checking for negative
   EBM_ASSERT(0 != total);

   pRet->m_weightTotal = total;

   LOG_0(Trace_Verbose, "Exited InnerBag::GenerateSingleInnerBag");
   return Error_None;
}

InnerBag * InnerBag::GenerateFlatInnerBag(
   const size_t cSamples,
   const FloatFast * const aWeights
) {
   LOG_0(Trace_Info, "Entered InnerBag::GenerateFlatInnerBag");

   EBM_ASSERT(1 <= cSamples); // if there were no samples, we wouldn't be called

   InnerBag * const pRet = static_cast<InnerBag *>(malloc(sizeof(InnerBag)));
   if(nullptr == pRet) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateFlatInnerBag nullptr == pRet");
      return nullptr;
   }
   pRet->InitializeUnfailing();

   pRet->m_weightTotal = static_cast<FloatBig>(cSamples);
   if(nullptr != aWeights) {
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

      FloatBig total;
      total = AddPositiveFloatsSafeBig(cSamples, aWeights);
      if(std::isnan(total) || std::isinf(total) || total <= 0) {
         pRet->Free();
         LOG_0(Trace_Warning, "WARNING InnerBag::GenerateFlatInnerBag std::isnan(total) || std::isinf(total) || total <= 0");
         return nullptr;
      }
      pRet->m_weightTotal = total;

      memcpy(aWeightsInternal, aWeights, cBytesWeightsInternalCapacity);
   }

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

ErrorEbm InnerBag::GenerateInnerBags(
   void * const rng,
   const size_t cSamples,
   const FloatFast * const aWeights,
   const size_t cInnerBags,
   InnerBag *** const papOut
) {
   LOG_0(Trace_Info, "Entered InnerBag::GenerateInnerBags");

   EBM_ASSERT(nullptr != papOut);
   EBM_ASSERT(nullptr == *papOut);

   const size_t cInnerBagsAfterZero = size_t { 0 } == cInnerBags ? size_t { 1 } : cInnerBags;

   if(IsMultiplyError(sizeof(InnerBag *), cInnerBagsAfterZero)) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateInnerBags IsMultiplyError(sizeof(InnerBag *), cInnerBagsAfterZero)");
      return Error_OutOfMemory;
   }
   InnerBag ** apInnerBags = static_cast<InnerBag **>(malloc(sizeof(InnerBag *) * cInnerBagsAfterZero));
   if(UNLIKELY(nullptr == apInnerBags)) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateInnerBags nullptr == apInnerBags");
      return Error_OutOfMemory;
   }

   InnerBag ** ppInnerBagInit = apInnerBags;
   const InnerBag * const * const ppInnerBagsEnd = &apInnerBags[cInnerBagsAfterZero];
   do {
      *ppInnerBagInit = nullptr;
      ++ppInnerBagInit;
   } while(ppInnerBagsEnd != ppInnerBagInit);

   *papOut = apInnerBags;

   if(size_t { 0 } == cInnerBags) {
      // zero is a special value that really means allocate one set that contains all samples.
      InnerBag * const pSingleInnerBag = GenerateFlatInnerBag(cSamples, aWeights);
      if(UNLIKELY(nullptr == pSingleInnerBag)) {
         LOG_0(Trace_Warning, "WARNING InnerBag::GenerateInnerBags nullptr == pSingleInnerBag");
         return Error_OutOfMemory;
      }
      apInnerBags[0] = pSingleInnerBag;
   } else {
      InnerBag ** ppInnerBag = apInnerBags;
      do {
         const ErrorEbm error = GenerateSingleInnerBag(rng, cSamples, aWeights, ppInnerBag);
         if(UNLIKELY(Error_None != error)) {
            return error;
         }
         ++ppInnerBag;
      } while(ppInnerBagsEnd != ppInnerBag);
   }
   LOG_0(Trace_Info, "Exited InnerBag::GenerateInnerBags");
   return Error_None;
}

} // DEFINED_ZONE_NAME
