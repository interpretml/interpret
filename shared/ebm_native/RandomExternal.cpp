// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include "ebm_native.h" // IntEbmType
#include "EbmInternal.h" // INLINE_ALWAYS
#include "RandomStream.h"

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateRandomNumber(IntEbmType randomSeed) {
   RandomStream randomStream;
   // this is a bit inefficient in that we go through a complete regeneration of the internal state,
   // but it gives us a simple interface
   randomStream.Initialize(randomSeed);
   IntEbmType ret = randomStream.NextEbmInt();
   return ret;
}

// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterSamplingWithoutReplacementParametersMessages = 5;
static int g_cLogExitSamplingWithoutReplacementParametersMessages = 5;

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION SamplingWithoutReplacement(
   IntEbmType randomSeed,
   IntEbmType countIncluded,
   IntEbmType countSamples,
   IntEbmType * isIncludedOut
) {
   LOG_COUNTED_N(
      &g_cLogEnterSamplingWithoutReplacementParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered SamplingWithoutReplacement: "
      "randomSeed=%" IntEbmTypePrintf ", "
      "countIncluded=%" IntEbmTypePrintf ", "
      "countSamples=%" IntEbmTypePrintf ", "
      "isIncludedOut=%p"
      ,
      randomSeed,
      countIncluded,
      countSamples,
      static_cast<void *>(isIncludedOut)
   );

   if(UNLIKELY(nullptr == isIncludedOut)) {
      LOG_0(TraceLevelError, "ERROR SamplingWithoutReplacement nullptr == isIncludedOut");
      return;
   }

   if(UNLIKELY(countSamples <= IntEbmType { 0 })) {
      if(UNLIKELY(countSamples < IntEbmType { 0 })) {
         LOG_0(TraceLevelError, "ERROR SamplingWithoutReplacement countSamples < IntEbmType { 0 }");
      }
      return;
   }
   if(UNLIKELY(!IsNumberConvertable<size_t>(countSamples))) {
      LOG_0(TraceLevelWarning, "WARNING SamplingWithoutReplacement !IsNumberConvertable<size_t>(countSamples)");
      return;
   }
   size_t cSamplesRemaining = static_cast<size_t>(countSamples);
   if(UNLIKELY(IsMultiplyError(cSamplesRemaining, sizeof(*isIncludedOut)))) {
      LOG_0(TraceLevelWarning, "WARNING SamplingWithoutReplacement IsMultiplyError(cSamples, sizeof(*isIncludedOut))");
      return;
   }

   if(UNLIKELY(countIncluded < IntEbmType { 0 })) {
      // this is a stupid input.  Fix it, but give the caller a warning so they can correct their code
      LOG_0(TraceLevelWarning, "WARNING SamplingWithoutReplacement countIncluded shouldn't be negative");
      countIncluded = IntEbmType { 0 };
   }
   if(UNLIKELY(countSamples < countIncluded)) {
      // this is a stupid input.  Fix it, but give the caller a warning so they can correct their code
      LOG_0(TraceLevelWarning, "WARNING SamplingWithoutReplacement countIncluded shouldn't be higher than countSamples");
      countIncluded = countSamples;
   }
   // countIncluded can't be negative or higher than countSamples, so it can be converted to size_t
   size_t cIncludedRemaining = static_cast<size_t>(countIncluded);

   RandomStream randomStream;
   randomStream.Initialize(randomSeed);

   IntEbmType * pbIncluded = isIncludedOut;
   do {
      const size_t iRandom = randomStream.Next(cSamplesRemaining);
      const bool bIncluded = UNPREDICTABLE(iRandom < cIncludedRemaining);
      cIncludedRemaining = UNPREDICTABLE(bIncluded) ? cIncludedRemaining - size_t { 1 } : cIncludedRemaining;
      *pbIncluded = UNPREDICTABLE(bIncluded) ? EBM_TRUE : EBM_FALSE;
      ++pbIncluded;
      --cSamplesRemaining;
   } while(0 != cSamplesRemaining);
   EBM_ASSERT(0 == cIncludedRemaining); // this should be all used up too now

   LOG_COUNTED_0(
      &g_cLogExitSamplingWithoutReplacementParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Exited SamplingWithoutReplacement"
   );
}
