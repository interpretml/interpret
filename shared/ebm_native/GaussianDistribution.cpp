// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"
#include "RandomStream.hpp"
#include "RandomNondeterministic.hpp"

#include "GaussianDistribution.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterGenerateGaussianRandomCountParametersMessages = 25;
static int g_cLogExitGenerateGaussianRandomCountParametersMessages = 25;

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GenerateGaussianRandom(
   BoolEbmType isDeterministic,
   SeedEbmType randomSeed,
   double stddev,
   IntEbmType count,
   double * randomOut
) {
   LOG_COUNTED_N(
      &g_cLogEnterGenerateGaussianRandomCountParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered GenerateGaussianRandom: "
      "isDeterministic=%s, "
      "randomSeed=%" SeedEbmTypePrintf ", "
      "stddev=%le, "
      "count=%" IntEbmTypePrintf ", "
      "randomOut=%p"
      ,
      ObtainTruth(isDeterministic),
      randomSeed,
      stddev,
      count,
      static_cast<const void *>(randomOut)
   );

   if(UNLIKELY(count <= IntEbmType { 0 })) {
      if(UNLIKELY(count < IntEbmType { 0 })) {
         LOG_0(TraceLevelError, "ERROR GenerateGaussianRandom count < IntEbmType { 0 }");
         return Error_IllegalParamValue;
      } else {
         LOG_COUNTED_0(
            &g_cLogExitGenerateGaussianRandomCountParametersMessages,
            TraceLevelInfo,
            TraceLevelVerbose,
            "GenerateGaussianRandom zero items requested");
         return Error_None;
      }
   }
   if(UNLIKELY(IsConvertError<size_t>(count))) {
      LOG_0(TraceLevelError, "ERROR GenerateGaussianRandom IsConvertError<size_t>(count)");
      return Error_IllegalParamValue;
   }
   const size_t c = static_cast<size_t>(count);
   if(UNLIKELY(IsMultiplyError(sizeof(*randomOut), c))) {
      LOG_0(TraceLevelError, "ERROR GenerateGaussianRandom IsMultiplyError(sizeof(*randomOut), c)");
      return Error_IllegalParamValue;
   }

   if(UNLIKELY(nullptr == randomOut)) {
      LOG_0(TraceLevelError, "ERROR GenerateGaussianRandom nullptr == randomOut");
      return Error_IllegalParamValue;
   }

   if(UNLIKELY(std::isnan(stddev))) {
      LOG_0(TraceLevelError, "ERROR GenerateGaussianRandom stddev cannot be NaN");
      return Error_IllegalParamValue;
   }
   if(UNLIKELY(std::isinf(stddev))) {
      LOG_0(TraceLevelError, "ERROR GenerateGaussianRandom stddev cannot be +-infinity");
      return Error_IllegalParamValue;
   }
   if(UNLIKELY(stddev < 0)) {
      // TODO: do we handle 0?  We would write out all zeros..

      LOG_0(TraceLevelError, "ERROR GenerateGaussianRandom stddev <= 0");
      return Error_IllegalParamValue;
   }

   GaussianDistribution gaussian(stddev);

   double * pRandom = randomOut;
   const double * const pRandomEnd = randomOut + count;
   if(EBM_FALSE != isDeterministic) {
      RandomDeterministic randomGenerator;
      randomGenerator.InitializeUnsigned(randomSeed, k_gaussianRandomizationMix);
      do {
         *pRandom = gaussian.Sample(randomGenerator, 1.0);
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
         LOG_0(TraceLevelWarning, "WARNING GenerateGaussianRandom Out of memory allocating BoosterCore");
         return Error_OutOfMemory;
      } catch(...) {
         LOG_0(TraceLevelWarning, "WARNING GenerateGaussianRandom Unknown error");
         return Error_UnexpectedInternal;
      }
   }

   LOG_COUNTED_0(
      &g_cLogExitGenerateGaussianRandomCountParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Exited GenerateGaussianRandom");

   return Error_None;
}

} // DEFINED_ZONE_NAME
