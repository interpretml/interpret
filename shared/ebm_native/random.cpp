// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include "ebm_native.h" // EBM_API_BODY
#include "logging.h" // LOG_
#include "zones.h"

#include "RandomDeterministic.hpp"
#include "RandomNondeterministic.hpp"
#include "GaussianDistribution.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME


EBM_API_BODY IntEbm EBM_CALLING_CONVENTION MeasureRNG() {
   return sizeof(RandomDeterministic);
}

EBM_API_BODY void EBM_CALLING_CONVENTION InitRNG(SeedEbm seed, void * rngOut) {
   RandomDeterministic * const pRng = reinterpret_cast<RandomDeterministic *>(rngOut);
   pRng->Initialize(seed);
}

EBM_API_BODY void EBM_CALLING_CONVENTION CopyRNG(void * rng, void * rngOut) {
   memcpy(rngOut, rng, sizeof(RandomDeterministic));
}

EBM_API_BODY void EBM_CALLING_CONVENTION BranchRNG(void * rng, void * rngOut) {
   RandomDeterministic * const pRng = reinterpret_cast<RandomDeterministic *>(rng);
   RandomDeterministic * const pRngOut = reinterpret_cast<RandomDeterministic *>(rngOut);

   const uint64_t seed = pRng->Next(std::numeric_limits<uint64_t>::max());
   // NOTE: it might be better to generate a seed that has all 128 bits of
   //       our internal state, but 64 bits should be good enough for now
   pRngOut->Initialize(seed);
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION GenerateSeed(void * rng, SeedEbm * seedOut) {
   if(nullptr == seedOut) {
      LOG_0(Trace_Warning, "WARNING GenerateSeed nullptr == seedOut");
      return Error_None;
   }
   if(nullptr == rng) {
      SeedEbm seed;
      try {
         RandomNondeterministic<USeedEbm> randomGenerator;
         seed = randomGenerator.NextNegative();
      } catch(const std::bad_alloc &) {
         LOG_0(Trace_Warning, "WARNING GenerateSeed Out of memory in std::random_device");
         // this cannot be relied on by the caller but hopefully a zero seed will be more obviously an error
         *seedOut = SeedEbm { 0 };
         return Error_OutOfMemory;
      } catch(...) {
         LOG_0(Trace_Warning, "WARNING GenerateSeed Unknown error in std::random_device");
         // this cannot be relied on by the caller but hopefully a zero seed will be more obviously an error
         *seedOut = SeedEbm { 0 };
         return Error_UnexpectedInternal;
      }
      *seedOut = seed;
      return Error_None;
   } else {
      RandomDeterministic * const pRng = reinterpret_cast<RandomDeterministic *>(rng);
      *seedOut = pRng->Next<SeedEbm>();
      return Error_None;
   }
}

// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterGenerateGaussianRandom = 25;
static int g_cLogExitGenerateGaussianRandom = 25;

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION GenerateGaussianRandom(
   void * rng,
   double stddev,
   IntEbm count,
   double * randomOut
) {
   LOG_COUNTED_N(
      &g_cLogEnterGenerateGaussianRandom,
      Trace_Info,
      Trace_Verbose,
      "Entered GenerateGaussianRandom: "
      "rng=%p, "
      "stddev=%le, "
      "count=%" IntEbmPrintf ", "
      "randomOut=%p"
      ,
      rng,
      stddev,
      count,
      static_cast<const void *>(randomOut)
   );

   if(UNLIKELY(count <= IntEbm { 0 })) {
      if(UNLIKELY(count < IntEbm { 0 })) {
         LOG_0(Trace_Error, "ERROR GenerateGaussianRandom count < IntEbm { 0 }");
         return Error_IllegalParamVal;
      } else {
         LOG_COUNTED_0(
            &g_cLogExitGenerateGaussianRandom,
            Trace_Info,
            Trace_Verbose,
            "GenerateGaussianRandom zero items requested");
         return Error_None;
      }
   }
   if(UNLIKELY(IsConvertError<size_t>(count))) {
      LOG_0(Trace_Error, "ERROR GenerateGaussianRandom IsConvertError<size_t>(count)");
      return Error_IllegalParamVal;
   }
   const size_t c = static_cast<size_t>(count);
   if(UNLIKELY(IsMultiplyError(sizeof(*randomOut), c))) {
      LOG_0(Trace_Error, "ERROR GenerateGaussianRandom IsMultiplyError(sizeof(*randomOut), c)");
      return Error_IllegalParamVal;
   }

   if(UNLIKELY(nullptr == randomOut)) {
      LOG_0(Trace_Error, "ERROR GenerateGaussianRandom nullptr == randomOut");
      return Error_IllegalParamVal;
   }

   if(UNLIKELY(std::isnan(stddev))) {
      LOG_0(Trace_Error, "ERROR GenerateGaussianRandom stddev cannot be NaN");
      return Error_IllegalParamVal;
   }
   if(UNLIKELY(std::isinf(stddev))) {
      LOG_0(Trace_Error, "ERROR GenerateGaussianRandom stddev cannot be +-infinity");
      return Error_IllegalParamVal;
   }
   if(UNLIKELY(stddev < 0)) {
      // TODO: do we handle 0?  We would write out all zeros..

      LOG_0(Trace_Error, "ERROR GenerateGaussianRandom stddev <= 0");
      return Error_IllegalParamVal;
   }

   GaussianDistribution gaussian(stddev);

   double * pRandom = randomOut;
   const double * const pRandomEnd = randomOut + count;
   if(nullptr != rng) {
      RandomDeterministic * const pRng = reinterpret_cast<RandomDeterministic *>(rng);
      do {
         *pRandom = gaussian.Sample(*pRng, 1.0);
         ++pRandom;
      } while(pRandomEnd != pRandom);
   } else {
      try {
         RandomNondeterministic<uint64_t> randomGenerator;
         do {
            *pRandom = gaussian.Sample(randomGenerator, 1.0);
            ++pRandom;
         } while(pRandomEnd != pRandom);
      } catch(const std::bad_alloc &) {
         LOG_0(Trace_Warning, "WARNING GenerateGaussianRandom Out of memory allocating randomGenerator");
         return Error_OutOfMemory;
      } catch(...) {
         LOG_0(Trace_Warning, "WARNING GenerateGaussianRandom Unknown error");
         return Error_UnexpectedInternal;
      }
   }

   LOG_COUNTED_0(
      &g_cLogExitGenerateGaussianRandom,
      Trace_Info,
      Trace_Verbose,
      "Exited GenerateGaussianRandom");

   return Error_None;
}

} // DEFINED_ZONE_NAME
