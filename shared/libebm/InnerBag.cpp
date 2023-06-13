// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "logging.h" // EBM_ASSERT

#include "ebm_internal.hpp" // AddPositiveFloatsSafe
#include "RandomDeterministic.hpp" // RandomDeterministic
#include "RandomNondeterministic.hpp" // RandomNondeterministic
#include "InnerBag.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

ErrorEbm InnerBag::InitializeRealInnerBag(
   void * const rng,
   const size_t cSamples,
   const FloatFast * const aWeights
) {
   LOG_0(Trace_Verbose, "Entered InnerBag::GenerateSingleInnerBag");

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
   m_aCountOccurrences = aCountOccurrences;

   if(IsMultiplyError(sizeof(FloatFast), cSamples)) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag IsMultiplyError(sizeof(FloatFast), cSamples)");
      return Error_OutOfMemory;
   }
   FloatFast * const aWeightsInternal = static_cast<FloatFast *>(malloc(sizeof(FloatFast) * cSamples));
   if(nullptr == aWeightsInternal) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag nullptr == aWeightsInternal");
      return Error_OutOfMemory;
   }
   m_aWeights = aWeightsInternal;

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
   double total;
   if(nullptr == aWeights) {
      do {
         *pWeightsInternal = static_cast<FloatFast>(*pCountOccurrences);
         ++pWeightsInternal;
         ++pCountOccurrences;
      } while(pCountOccurrencesEnd != pCountOccurrences);
      total = static_cast<double>(cSamples);
#ifndef NDEBUG
      const double debugTotal = AddPositiveFloatsSafe<double>(cSamples, aWeightsInternal);
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
      total = AddPositiveFloatsSafe<double>(cSamples, aWeightsInternal);
      if(std::isnan(total) || std::isinf(total) || total <= double { 0 }) {
         LOG_0(Trace_Warning, "WARNING InnerBag::GenerateSingleInnerBag std::isnan(total) || std::isinf(total) || total <= 0");
         return Error_UserParamVal;
      }
   }
   // if they were all zero then we'd ignore the weights param.  If there are negative numbers it might add
   // to zero though so check it after checking for negative
   EBM_ASSERT(0 != total);

   m_weightTotal = total;

   LOG_0(Trace_Verbose, "Exited InnerBag::GenerateSingleInnerBag");
   return Error_None;
}

ErrorEbm InnerBag::InitializeFakeInnerBag(
   const size_t cSamples,
   const FloatFast * const aWeights
) {
   LOG_0(Trace_Info, "Entered InnerBag::GenerateFlatInnerBag");

   EBM_ASSERT(1 <= cSamples); // if there were no samples, we wouldn't be called

   m_weightTotal = static_cast<double>(cSamples);
   if(nullptr != aWeights) {
      if(IsMultiplyError(sizeof(FloatFast), cSamples)) {
         LOG_0(Trace_Warning, "WARNING InnerBag::GenerateFlatInnerBag IsMultiplyError(sizeof(FloatFast), cSamples)");
         return Error_OutOfMemory;
      }
      const size_t cBytesWeightsInternalCapacity = sizeof(FloatFast) * cSamples;
      FloatFast * const aWeightsInternal = static_cast<FloatFast *>(malloc(cBytesWeightsInternalCapacity));
      if(nullptr == aWeightsInternal) {
         LOG_0(Trace_Warning, "WARNING InnerBag::GenerateFlatInnerBag nullptr == aWeightsInternal");
         return Error_OutOfMemory;
      }
      m_aWeights = aWeightsInternal;

      double total = AddPositiveFloatsSafe<double>(cSamples, aWeights);
      if(std::isnan(total) || std::isinf(total) || total <= double { 0 }) {
         LOG_0(Trace_Warning, "WARNING InnerBag::GenerateFlatInnerBag std::isnan(total) || std::isinf(total) || total <= 0");
         return Error_UserParamVal;
      }
      m_weightTotal = total;

      memcpy(aWeightsInternal, aWeights, cBytesWeightsInternalCapacity);
   }

   LOG_0(Trace_Info, "Exited InnerBag::GenerateFlatInnerBag");
   return Error_None;
}

void InnerBag::FreeInnerBags(const size_t cInnerBags, InnerBag * const aInnerBags) {
   LOG_0(Trace_Info, "Entered InnerBag::FreeInnerBags");
   if(LIKELY(nullptr != aInnerBags)) {
      const size_t cInnerBagsAfterZero = size_t { 0 } == cInnerBags ? size_t { 1 } : cInnerBags;
      InnerBag * pInnerBag = aInnerBags;
      const InnerBag * const pInnerBagsEnd = aInnerBags + cInnerBagsAfterZero;
      do {
         free(pInnerBag->m_aCountOccurrences);
         free(pInnerBag->m_aWeights);
         ++pInnerBag;
      } while(pInnerBagsEnd != pInnerBag);
      free(aInnerBags);
   }
   LOG_0(Trace_Info, "Exited InnerBag::FreeInnerBags");
}

InnerBag * InnerBag::AllocateInnerBags(const size_t cInnerBags) {
   LOG_0(Trace_Info, "Entered InnerBag::AllocateInnerBags");

   const size_t cInnerBagsAfterZero = size_t { 0 } == cInnerBags ? size_t { 1 } : cInnerBags;

   if(IsMultiplyError(sizeof(InnerBag), cInnerBagsAfterZero)) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateInnerBags IsMultiplyError(sizeof(InnerBag), cInnerBagsAfterZero)");
      return nullptr;
   }
   InnerBag * aInnerBag = static_cast<InnerBag *>(malloc(sizeof(InnerBag) * cInnerBagsAfterZero));
   if(UNLIKELY(nullptr == aInnerBag)) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateInnerBags nullptr == aInnerBag");
      return nullptr;
   }

   InnerBag * pInnerBag = aInnerBag;
   const InnerBag * const pInnerBagsEnd = &aInnerBag[cInnerBagsAfterZero];
   do {
      pInnerBag->m_aCountOccurrences = nullptr;
      pInnerBag->m_aWeights = nullptr;
      ++pInnerBag;
   } while(pInnerBagsEnd != pInnerBag);

   LOG_0(Trace_Info, "Exited InnerBag::AllocateInnerBags");
   return aInnerBag;
}

} // DEFINED_ZONE_NAME
