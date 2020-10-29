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
   IntEbmType countSamples,
   IntEbmType * trainingCountsOut
) {
   LOG_COUNTED_N(
      &g_cLogEnterSampleWithoutReplacementParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered SampleWithoutReplacement: "
      "randomSeed=%" SeedEbmTypePrintf ", "
      "countTrainingSamples=%" IntEbmTypePrintf ", "
      "countSamples=%" IntEbmTypePrintf ", "
      "trainingCountsOut=%p"
      ,
      randomSeed,
      countTrainingSamples,
      countSamples,
      static_cast<void *>(trainingCountsOut)
   );

   if(UNLIKELY(nullptr == trainingCountsOut)) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement nullptr == trainingCountsOut");
      return;
   }

   if(UNLIKELY(countSamples <= IntEbmType { 0 })) {
      if(UNLIKELY(countSamples < IntEbmType { 0 })) {
         LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement countSamples < IntEbmType { 0 }");
      }
      return;
   }
   if(UNLIKELY(!IsNumberConvertable<size_t>(countSamples))) {
      LOG_0(TraceLevelWarning, "WARNING SampleWithoutReplacement !IsNumberConvertable<size_t>(countSamples)");
      return;
   }
   size_t cSamplesRemaining = static_cast<size_t>(countSamples);
   if(UNLIKELY(IsMultiplyError(cSamplesRemaining, sizeof(*trainingCountsOut)))) {
      LOG_0(TraceLevelWarning, "WARNING SampleWithoutReplacement IsMultiplyError(cSamples, sizeof(*trainingCountsOut))");
      return;
   }

   if(UNLIKELY(countTrainingSamples < IntEbmType { 0 })) {
      // this is a stupid input.  Fix it, but give the caller a warning so they can correct their code
      LOG_0(TraceLevelWarning, "WARNING SampleWithoutReplacement countTrainingSamples shouldn't be negative");
      countTrainingSamples = IntEbmType { 0 };
   }
   if(UNLIKELY(countSamples < countTrainingSamples)) {
      // this is a stupid input.  Fix it, but give the caller a warning so they can correct their code
      LOG_0(TraceLevelWarning, "WARNING SampleWithoutReplacement countTrainingSamples shouldn't be higher than countSamples");
      countTrainingSamples = countSamples;
   }
   // countTrainingSamples can't be negative or higher than countSamples, so it can be converted to size_t
   size_t cTrainingRemaining = static_cast<size_t>(countTrainingSamples);

   RandomStream randomStream;
   randomStream.InitializeUnsigned(randomSeed, k_samplingWithoutReplacementRandomizationMix);

   IntEbmType * pTrainingCountsOut = trainingCountsOut;
   do {
      const size_t iRandom = randomStream.Next(cSamplesRemaining);
      const bool bTrainingSample = UNPREDICTABLE(iRandom < cTrainingRemaining);
      cTrainingRemaining = UNPREDICTABLE(bTrainingSample) ? cTrainingRemaining - size_t { 1 } : cTrainingRemaining;
      *pTrainingCountsOut = UNPREDICTABLE(bTrainingSample) ? IntEbmType { 1 } : IntEbmType { 0 };
      ++pTrainingCountsOut;
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
