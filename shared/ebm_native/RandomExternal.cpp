// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include "ebm_native.h" // IntEbmType
#include "EbmInternal.h" // INLINE_ALWAYS
#include "RandomStream.h"

EBM_NATIVE_IMPORT_EXPORT_BODY SeedEbmType EBM_NATIVE_CALLING_CONVENTION GenerateRandomNumber(
   SeedEbmType randomSeed,
   SeedEbmType stageRandomizationMix
) {
   RandomStream randomStream;
   // this is a bit inefficient in that we go through a complete regeneration of the internal state,
   // but it gives us a simple interface
   randomStream.InitializeSigned(randomSeed, stageRandomizationMix);
   SeedEbmType ret = randomStream.NextSeed();
   return ret;
}

// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterSampleWithoutReplacementParametersMessages = 5;
static int g_cLogExitSampleWithoutReplacementParametersMessages = 5;

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION SampleWithoutReplacement(
   SeedEbmType randomSeed,
   IntEbmType countTrainingSamples,
   IntEbmType countValidationSamples,
   IntEbmType * sampleCountsOut
) {
   LOG_COUNTED_N(
      &g_cLogEnterSampleWithoutReplacementParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered SampleWithoutReplacement: "
      "randomSeed=%" SeedEbmTypePrintf ", "
      "countTrainingSamples=%" IntEbmTypePrintf ", "
      "countValidationSamples=%" IntEbmTypePrintf ", "
      "sampleCountsOut=%p"
      ,
      randomSeed,
      countTrainingSamples,
      countValidationSamples,
      static_cast<void *>(sampleCountsOut)
   );

   if(UNLIKELY(nullptr == sampleCountsOut)) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement nullptr == sampleCountsOut");
      return;
   }

   if(UNLIKELY(countTrainingSamples < IntEbmType { 0 })) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement countTrainingSamples < IntEbmType { 0 }");
      return;
   }
   if(UNLIKELY(!IsNumberConvertable<size_t>(countTrainingSamples))) {
      LOG_0(TraceLevelWarning, "WARNING SampleWithoutReplacement !IsNumberConvertable<size_t>(countTrainingSamples)");
      return;
   }
   const size_t cTrainingSamples = static_cast<size_t>(countTrainingSamples);

   if(UNLIKELY(countValidationSamples < IntEbmType { 0 })) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement countValidationSamples < IntEbmType { 0 }");
      return;
   }
   if(UNLIKELY(!IsNumberConvertable<size_t>(countValidationSamples))) {
      LOG_0(TraceLevelWarning, "WARNING SampleWithoutReplacement !IsNumberConvertable<size_t>(countValidationSamples)");
      return;
   }
   const size_t cValidationSamples = static_cast<size_t>(countValidationSamples);

   if(UNLIKELY(IsAddError(cTrainingSamples, cValidationSamples))) {
      LOG_0(TraceLevelWarning, "WARNING SampleWithoutReplacement IsAddError(cTrainingSamples, cValidationSamples)");
      return;
   }
   size_t cSamplesRemaining = cTrainingSamples + cValidationSamples;
   if(UNLIKELY(size_t { 0 } == cSamplesRemaining)) {
      // there's nothing for us to fill the array with
      return;
   }
   if(UNLIKELY(IsMultiplyError(cSamplesRemaining, sizeof(*sampleCountsOut)))) {
      LOG_0(TraceLevelWarning, "WARNING SampleWithoutReplacement IsMultiplyError(cSamples, sizeof(*sampleCountsOut))");
      return;
   }

   size_t cTrainingRemaining = cTrainingSamples;

   RandomStream randomStream;
   randomStream.InitializeUnsigned(randomSeed, k_samplingWithoutReplacementRandomizationMix);

   IntEbmType * pSampleCountsOut = sampleCountsOut;
   do {
      const size_t iRandom = randomStream.Next(cSamplesRemaining);
      const bool bTrainingSample = UNPREDICTABLE(iRandom < cTrainingRemaining);
      cTrainingRemaining = UNPREDICTABLE(bTrainingSample) ? cTrainingRemaining - size_t { 1 } : cTrainingRemaining;
      *pSampleCountsOut = UNPREDICTABLE(bTrainingSample) ? IntEbmType { 1 } : IntEbmType { -1 };
      ++pSampleCountsOut;
      --cSamplesRemaining;
   } while(0 != cSamplesRemaining);
   EBM_ASSERT(0 == cTrainingRemaining); // this should be all used up too now

   LOG_COUNTED_0(
      &g_cLogExitSampleWithoutReplacementParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Exited SampleWithoutReplacement"
   );
}
