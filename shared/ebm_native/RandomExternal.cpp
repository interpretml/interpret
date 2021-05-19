// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "EbmInternal.h"

#include "RandomStream.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

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


static int g_cLogEnterStratifiedSamplingWithoutReplacementParametersMessages = 5;
static int g_cLogExitStratifiedSamplingWithoutReplacementParametersMessages = 5;


EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION StratifiedSamplingWithoutReplacement(
   SeedEbmType randomSeed,
   IntEbmType countTargetClasses,
   IntEbmType countTrainingSamples,
   IntEbmType countValidationSamples,
   IntEbmType* targets,
   IntEbmType* sampleCountsOut
) {
   struct TargetSamplingCounts {
      size_t m_cTraining;
      size_t m_cTotalRemaining;
   };

   LOG_COUNTED_N(
      &g_cLogEnterStratifiedSamplingWithoutReplacementParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered StratifiedSamplingWithoutReplacement: "
      "randomSeed=%" SeedEbmTypePrintf ", "
      "countTargetClasses=%" IntEbmTypePrintf ", "
      "countTrainingSamples=%" IntEbmTypePrintf ", "
      "countValidationSamples=%" IntEbmTypePrintf ", "
      "targets=%p, "
      "sampleCountsOut=%p"
      ,
      randomSeed,
      countTargetClasses,
      countTrainingSamples,
      countValidationSamples,
      static_cast<void*>(targets),
      static_cast<void*>(sampleCountsOut)
   );

   if (UNLIKELY(nullptr == targets)) {
      LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement nullptr == targets");
      return Error_InvalidParameter;
   }

   if (UNLIKELY(nullptr == sampleCountsOut)) {
      LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement nullptr == sampleCountsOut");
      return Error_InvalidParameter;
   }

   if (UNLIKELY(countTrainingSamples < IntEbmType{ 0 })) {
      LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement countTrainingSamples < IntEbmType{ 0 }");
      return Error_InvalidParameter;
   }
   if (UNLIKELY(!IsNumberConvertable<size_t>(countTrainingSamples))) {
      LOG_0(TraceLevelWarning, "WARNING StratifiedSamplingWithoutReplacement !IsNumberConvertable<size_t>(countTrainingSamples)");
      return Error_InvalidParameter;
   }
   const size_t cTrainingSamples = static_cast<size_t>(countTrainingSamples);

   if (UNLIKELY(countValidationSamples < IntEbmType{ 0 })) {
      LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement countValidationSamples < IntEbmType{ 0 }");
      return Error_InvalidParameter;
   }
   if (UNLIKELY(!IsNumberConvertable<size_t>(countValidationSamples))) {
      LOG_0(TraceLevelWarning, "WARNING StratifiedSamplingWithoutReplacement !IsNumberConvertable<size_t>(countValidationSamples)");
      return Error_InvalidParameter;
   }
   const size_t cValidationSamples = static_cast<size_t>(countValidationSamples);

   if (UNLIKELY(IsAddError(cTrainingSamples, cValidationSamples))) {
      LOG_0(TraceLevelWarning, "WARNING StratifiedSamplingWithoutReplacement IsAddError(countTrainingSamples, countValidationSamples))");
      return Error_InvalidParameter;
   }

   size_t cSamples = cTrainingSamples + cValidationSamples;
   if (UNLIKELY(cSamples == 0)) {
      // there's nothing for us to fill the sampleCountsOut with
      return Error_None;
   }

   if (countTargetClasses <= 0) {
      LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement countTargetClasses can't be negative or zero");
      return Error_InvalidParameter;
   }
   if (!IsNumberConvertable<size_t>(countTargetClasses)) {
      LOG_0(TraceLevelWarning, "WARNING StratifiedSamplingWithoutReplacement !IsNumberConvertable<size_t>(countTargetClasses)");
      return Error_InvalidParameter;
   }
   const size_t cTargetClasses = static_cast<size_t>(countTargetClasses);

   if (UNLIKELY(IsMultiplyError(cSamples, sizeof(*sampleCountsOut)))) {
      LOG_0(TraceLevelWarning, "WARNING StratifiedSamplingWithoutReplacement IsMultiplyError(cSamples, sizeof(*sampleCountsOut))");
      return Error_InvalidParameter;
   }

   if (UNLIKELY(IsMultiplyError(cTargetClasses, sizeof(TargetSamplingCounts)))) {
      LOG_0(TraceLevelWarning, "WARNING StratifiedSamplingWithoutReplacement IsMultiplyError(cTargetClasses, sizeof(TargetSamplingCounts))");
      return Error_InvalidParameter;
   }

   if (UNLIKELY(cTrainingSamples < cTargetClasses)) {
      LOG_0(TraceLevelWarning, "WARNING StratifiedSamplingWithoutReplacement cTrainingSamples < cTargetClasses");
   }

   if (UNLIKELY(cValidationSamples < cTargetClasses)) {
      LOG_0(TraceLevelWarning, "WARNING StratifiedSamplingWithoutReplacement cValidationSamples < cTargetClasses");
   }

   const size_t targetSamplingCountsSize = sizeof(TargetSamplingCounts) * cTargetClasses;
   TargetSamplingCounts* aTargetSamplingCounts = static_cast<TargetSamplingCounts*>(malloc(targetSamplingCountsSize));

   if (UNLIKELY(nullptr == aTargetSamplingCounts)) {
      LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement out of memory nullptr == aTargetSamplingCounts");
      return Error_OutOfMemory;
   }

   memset(aTargetSamplingCounts, 0, targetSamplingCountsSize);

   // calculate number of samples per label in the target
   for (size_t i = 0; i < cSamples; i++) {
      IntEbmType label = targets[i];

      if (UNLIKELY(label < 0 || label >= countTargetClasses)) {
         LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement label >= cTargetClasses");
         free(aTargetSamplingCounts);
         return Error_InvalidParameter;
      }

      ++aTargetSamplingCounts[label].m_cTotalRemaining;
   }

   // This stratified sampling algorithm guarantees:
   // (1) Either the train/validation counts work out perfectly for each class -or- there is at 
   //     least one class with a count above the ideal training count and at least one class with
   //     a training count below the ideal count,
   // (2) Given a sufficient amount of training samples, if a class has only one sample, it 
   //     should go to training,
   // (3) Given a sufficient amount of training samples, if a class only has two samples, one 
   //     should go to train and one should go to test,
   // (4) If a class has enough samples to hit the target train/validation count, its actual
   //     train/validation count should be no more than one away from the ideal count. 
   // 
   // Given these guarantees, the sketch of this algorithm is that for the common case where there 
   // are enough training samples to have more than one sample per class, we initialize the count 
   // of the training samples per class to be the floor of the ideal training count.  This will 
   // leave some amount of samples to be "leftover".  We assign leftovers to classes by determining
   // which class will get closest to its ideal training count by giving it one more training 
   // sample.  If there is more than one class that gets the same improvement, we'll randomly 
   // assign the "leftover" to one of the classes.
   // 
   // In addition to having leftovers as a result of taking the floor of the ideal training count 
   // of each class, we decrement the ideal training count of each class by 1 and consider those
   // samples leftovers as well.  This assures us we have enough leftovers to give 1 to any classes
   // that have 0 training samples when looking at leftovers.  We use this to achieve the 2nd 
   // guarantee that any class with at 1 sample will get at least one sample assigned to training.
   //
   // For the odd cases where there aren't enough training samples given to give at least one 
   // sample to each class, we'll let all the training samples be considered leftover and allow our
   // boosting of improvement for classes with no samples to drive how assignment of training 
   // samples is done as ideal training counts are impossible to achieve, but we'll try to assign
   // at least one training sample to each class that has samples.

   double idealTrainingProportion = static_cast<double>(cTrainingSamples) / cSamples;
   EBM_ASSERT(!std::isnan(idealTrainingProportion)); // since we checked cSamples not zero above
   EBM_ASSERT(!std::isinf(idealTrainingProportion)); // since we checked cSamples not zero above
   EBM_ASSERT(0 <= idealTrainingProportion);
   EBM_ASSERT(idealTrainingProportion <= 1);

   size_t globalLeftover = cTrainingSamples;

   if (cTrainingSamples > cTargetClasses) {

      size_t cClassesWithSamples = 0;

      for (size_t iTargetClass = 0; iTargetClass < cTargetClasses; iTargetClass++) {
         double fTrainingPerClass = std::floor(idealTrainingProportion * aTargetSamplingCounts[iTargetClass].m_cTotalRemaining);
         size_t cTrainingPerClass = static_cast<size_t>(fTrainingPerClass);
         if (0 < cTrainingPerClass) {
            --cTrainingPerClass;
         }
         cClassesWithSamples = (aTargetSamplingCounts[iTargetClass].m_cTotalRemaining > 0) ? cClassesWithSamples + 1 : cClassesWithSamples;
         aTargetSamplingCounts[iTargetClass].m_cTraining = cTrainingPerClass;
         EBM_ASSERT(cTrainingPerClass <= globalLeftover);
         globalLeftover -= cTrainingPerClass;
      }

      EBM_ASSERT(cClassesWithSamples <= globalLeftover);
   }

   EBM_ASSERT(globalLeftover <= cSamples);

   const size_t mostImprovedClassesCapacity = sizeof(size_t) * cTargetClasses;
   size_t* aMostImprovedClasses = static_cast<size_t*>(malloc(mostImprovedClassesCapacity));

   if (UNLIKELY(nullptr == aMostImprovedClasses)) {
      LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement out of memory nullptr == aMostImprovedClasses");
      free(aTargetSamplingCounts);
      return Error_OutOfMemory;
   }

   RandomStream randomStream;
   randomStream.InitializeUnsigned(randomSeed, k_stratifiedSamplingWithoutReplacementRandomizationMix);

   for (size_t iLeftover = 0; iLeftover < globalLeftover; iLeftover++) {
      double maxImprovement = std::numeric_limits<double>::lowest();
      size_t mostImprovedClassesSize = 0;
      memset(aMostImprovedClasses, 0, mostImprovedClassesCapacity);

      for (size_t iTargetClass = 0; iTargetClass < cTargetClasses; iTargetClass++) {
         const size_t cClassTraining = aTargetSamplingCounts[iTargetClass].m_cTraining;
         const size_t cClassRemaining = aTargetSamplingCounts[iTargetClass].m_cTotalRemaining;

         if (cClassTraining == cClassRemaining) {
            continue;
         }
         EBM_ASSERT(0 < cClassRemaining); // because cClassTraining == cClassRemaining if cClassRemaining is zero

         double idealClassTraining = idealTrainingProportion * static_cast<double>(cClassRemaining);
         double curTrainingDiff = idealClassTraining - cClassTraining;
         double newTrainingDiff = idealClassTraining - (cClassTraining + 1);
         double improvement = (curTrainingDiff * curTrainingDiff) - (newTrainingDiff * newTrainingDiff);
         
         if (0 == cClassTraining) {
            // improvement should not be able to be larger than 9
            improvement += 32;
         } else if (cClassTraining + 1 == cClassRemaining) {
            // improvement should not be able to be larger than 9
            improvement -= 32;
         }
         
         if (improvement > maxImprovement) {
            maxImprovement = improvement;
            memset(aMostImprovedClasses, 0, mostImprovedClassesCapacity);
            mostImprovedClassesSize = 0;
         }
         
         if (improvement == maxImprovement) {
            aMostImprovedClasses[mostImprovedClassesSize] = iTargetClass;
            ++mostImprovedClassesSize;
         }
      }
      EBM_ASSERT(std::numeric_limits<double>::lowest() != maxImprovement);

      // If more than one class has the same max improvement, randomly select between the classes
      // to give the leftover to.
      size_t iRandom = randomStream.Next(mostImprovedClassesSize);
      ++aTargetSamplingCounts[aMostImprovedClasses[iRandom]].m_cTraining;
   }

#ifndef NDEBUG
   size_t assignedTrainingCount = 0;
   for (size_t iTargetClass = 0; iTargetClass < cTargetClasses; iTargetClass++) {
      assignedTrainingCount += aTargetSamplingCounts[iTargetClass].m_cTraining;
   }
   EBM_ASSERT(assignedTrainingCount == cTrainingSamples);
#endif

   for (size_t iSample = 0; iSample < cSamples; iSample++) {
      TargetSamplingCounts* pTargetSample = &aTargetSamplingCounts[targets[iSample]];
      EBM_ASSERT(pTargetSample->m_cTotalRemaining > 0);
      const size_t iRandom = randomStream.Next(pTargetSample->m_cTotalRemaining);
      const bool bTrainingSample = UNPREDICTABLE(iRandom < pTargetSample->m_cTraining);

      if (UNPREDICTABLE(bTrainingSample)) {
         --pTargetSample->m_cTraining;
         sampleCountsOut[iSample] = IntEbmType{ 1 };
      }
      else {
         sampleCountsOut[iSample] = IntEbmType{ -1 };
      }

      --pTargetSample->m_cTotalRemaining;
   }

#ifndef NDEBUG
   for (size_t iTargetClass = 0; iTargetClass < cTargetClasses; iTargetClass++) {
      EBM_ASSERT(aTargetSamplingCounts[iTargetClass].m_cTraining == 0);
      EBM_ASSERT(aTargetSamplingCounts[iTargetClass].m_cTotalRemaining == 0);
   }
#endif

   free(aTargetSamplingCounts);
   free(aMostImprovedClasses);

   LOG_COUNTED_0(
      &g_cLogExitStratifiedSamplingWithoutReplacementParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Exited StratifiedSamplingWithoutReplacement"
   );

   return Error_None;
}

} // DEFINED_ZONE_NAME
