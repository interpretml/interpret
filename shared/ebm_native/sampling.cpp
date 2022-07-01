// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "data_set_shared.hpp"
#include "RandomStream.hpp"
#include "RandomNondeterministic.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// TODO: for DP-EBMs we cryptographically secure random number generators that we can use cross platform
// that generate uniform distributions and then normal distributions.  
// To generate the normal distribution we can use the Box-Muller method for which the code seems pretty simple:
// https://stackoverflow.com/questions/34903356/c11-random-number-distributions-are-not-consistent-across-platforms-what-al
// To generate uniform distributions, the C++ std seems to first generate a random number between 0 and 1 then scale it
// Generating a random number between 0 and 1 can be done by taking 0.5 and either choosing it or not based on a
// random bit, then multiplying it by 0.5 to get 0.25 where we again add it or not based on a random bit.  At the end
// we should have a perfectly random number between (0, 1].
// Next we need a way to pass random numbers between C++ and python/R.  We can change our random number generator
// to take any string which we can encrypt using AES256-CTR
// https://en.wikipedia.org/wiki/Cryptographically-secure_pseudorandom_number_generator
// we can pass the state back using hexidecimal strings with 256/16 = 16 characters
// Idea: if we have the caller pad their initial string with spaces to 16 characters, maybe we can avoid having
// an init function and just use a single function that modifies the 16 character string?
// We need to also provide utilities to generate normal distributions using the random number

EBM_NATIVE_IMPORT_EXPORT_BODY SeedEbmType EBM_NATIVE_CALLING_CONVENTION GenerateDeterministicSeed(
   SeedEbmType randomSeed,
   SeedEbmType stageRandomizationMix
) {
   RandomDeterministic randomDeterministic;
   // this is a bit inefficient in that we go through a complete regeneration of the internal state,
   // but it gives us a simple interface
   randomDeterministic.InitializeSigned(randomSeed, stageRandomizationMix);
   SeedEbmType ret = randomDeterministic.NextSeed();
   return ret;
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GenerateNondeterministicSeed(
   SeedEbmType * randomSeedOut
) {
   if(UNLIKELY(nullptr == randomSeedOut)) {
      LOG_0(TraceLevelError, "ERROR GenerateNondeterministicSeed nullptr == randomSeedOut");
      return Error_IllegalParamValue;
   }

   try {
      RandomNondeterministic<uint32_t> randomGenerator;
      const SeedEbmType ret = randomGenerator.NextSeed();
      *randomSeedOut = ret;
   } catch(const std::bad_alloc &) {
      LOG_0(TraceLevelWarning, "WARNING GenerateNondeterministicSeed Out of memory in std::random_device");
      return Error_OutOfMemory;
   } catch(...) {
      LOG_0(TraceLevelWarning, "WARNING GenerateNondeterministicSeed Unknown error in std::random_device");
      return Error_UnexpectedInternal;
   }
   return Error_None;
}


// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterSampleWithoutReplacementParametersMessages = 5;
static int g_cLogExitSampleWithoutReplacementParametersMessages = 5;

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION SampleWithoutReplacement(
   BoolEbmType isDeterministic,
   SeedEbmType randomSeed,
   IntEbmType countTrainingSamples,
   IntEbmType countValidationSamples,
   BagEbmType * sampleCountsOut
) {
   LOG_COUNTED_N(
      &g_cLogEnterSampleWithoutReplacementParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered SampleWithoutReplacement: "
      "isDeterministic=%s, "
      "randomSeed=%" SeedEbmTypePrintf ", "
      "countTrainingSamples=%" IntEbmTypePrintf ", "
      "countValidationSamples=%" IntEbmTypePrintf ", "
      "sampleCountsOut=%p"
      ,
      ObtainTruth(isDeterministic),
      randomSeed,
      countTrainingSamples,
      countValidationSamples,
      static_cast<void *>(sampleCountsOut)
   );

   if(UNLIKELY(nullptr == sampleCountsOut)) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement nullptr == sampleCountsOut");
      return Error_IllegalParamValue;
   }

   if(UNLIKELY(countTrainingSamples < IntEbmType { 0 })) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement countTrainingSamples < IntEbmType { 0 }");
      return Error_IllegalParamValue;
   }
   if(UNLIKELY(IsConvertError<size_t>(countTrainingSamples))) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement IsConvertError<size_t>(countTrainingSamples)");
      return Error_IllegalParamValue;
   }
   const size_t cTrainingSamples = static_cast<size_t>(countTrainingSamples);

   if(UNLIKELY(countValidationSamples < IntEbmType { 0 })) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement countValidationSamples < IntEbmType { 0 }");
      return Error_IllegalParamValue;
   }
   if(UNLIKELY(IsConvertError<size_t>(countValidationSamples))) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement IsConvertError<size_t>(countValidationSamples)");
      return Error_IllegalParamValue;
   }
   const size_t cValidationSamples = static_cast<size_t>(countValidationSamples);

   if(UNLIKELY(IsAddError(cTrainingSamples, cValidationSamples))) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement IsAddError(cTrainingSamples, cValidationSamples)");
      return Error_IllegalParamValue;
   }
   size_t cSamplesRemaining = cTrainingSamples + cValidationSamples;
   if(UNLIKELY(size_t { 0 } == cSamplesRemaining)) {
      // there's nothing for us to fill the array with
      return Error_None;
   }
   if(UNLIKELY(IsMultiplyError(sizeof(*sampleCountsOut), cSamplesRemaining))) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement IsMultiplyError(sizeof(*sampleCountsOut), cSamplesRemaining)");
      return Error_IllegalParamValue;
   }

   size_t cTrainingRemaining = cTrainingSamples;

   BagEbmType * pSampleCountsOut = sampleCountsOut;
   if(EBM_FALSE != isDeterministic) {
      RandomDeterministic randomGenerator;
      randomGenerator.InitializeUnsigned(randomSeed, k_samplingWithoutReplacementRandomizationMix);
      do {
         const size_t iRandom = randomGenerator.NextFast(cSamplesRemaining);
         const bool bTrainingSample = UNPREDICTABLE(iRandom < cTrainingRemaining);
         cTrainingRemaining = UNPREDICTABLE(bTrainingSample) ? cTrainingRemaining - size_t { 1 } : cTrainingRemaining;
         *pSampleCountsOut = UNPREDICTABLE(bTrainingSample) ? BagEbmType { 1 } : BagEbmType { -1 };
         ++pSampleCountsOut;
         --cSamplesRemaining;
      } while(0 != cSamplesRemaining);
   } else {
      try {
         RandomNondeterministic<size_t> randomGenerator;
         do {
            const size_t iRandom = randomGenerator.NextFast(cSamplesRemaining);
            const bool bTrainingSample = UNPREDICTABLE(iRandom < cTrainingRemaining);
            cTrainingRemaining = UNPREDICTABLE(bTrainingSample) ? cTrainingRemaining - size_t { 1 } : cTrainingRemaining;
            *pSampleCountsOut = UNPREDICTABLE(bTrainingSample) ? BagEbmType { 1 } : BagEbmType { -1 };
            ++pSampleCountsOut;
            --cSamplesRemaining;
         } while(0 != cSamplesRemaining);
      } catch(const std::bad_alloc &) {
         LOG_0(TraceLevelWarning, "WARNING GenerateGaussianRandom Out of memory in std::random_device");
         return Error_OutOfMemory;
      } catch(...) {
         LOG_0(TraceLevelWarning, "WARNING GenerateGaussianRandom Unknown error in std::random_device");
         return Error_UnexpectedInternal;
      }
   }
   EBM_ASSERT(0 == cTrainingRemaining); // this should be all used up too now

   LOG_COUNTED_0(
      &g_cLogExitSampleWithoutReplacementParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Exited SampleWithoutReplacement"
   );
   return Error_None;
}


static int g_cLogEnterStratifiedSamplingWithoutReplacementParametersMessages = 5;
static int g_cLogExitStratifiedSamplingWithoutReplacementParametersMessages = 5;

WARNING_PUSH
WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION StratifiedSamplingWithoutReplacement(
   SeedEbmType randomSeed,
   IntEbmType countTargetClasses,
   IntEbmType countTrainingSamples,
   IntEbmType countValidationSamples,
   IntEbmType * targets,
   BagEbmType * sampleCountsOut
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
      return Error_IllegalParamValue;
   }

   if (UNLIKELY(nullptr == sampleCountsOut)) {
      LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement nullptr == sampleCountsOut");
      return Error_IllegalParamValue;
   }

   if (UNLIKELY(countTrainingSamples < IntEbmType{ 0 })) {
      LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement countTrainingSamples < IntEbmType{ 0 }");
      return Error_IllegalParamValue;
   }
   if (UNLIKELY(IsConvertError<size_t>(countTrainingSamples))) {
      LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement IsConvertError<size_t>(countTrainingSamples)");
      return Error_IllegalParamValue;
   }
   const size_t cTrainingSamples = static_cast<size_t>(countTrainingSamples);

   if (UNLIKELY(countValidationSamples < IntEbmType{ 0 })) {
      LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement countValidationSamples < IntEbmType{ 0 }");
      return Error_IllegalParamValue;
   }
   if (UNLIKELY(IsConvertError<size_t>(countValidationSamples))) {
      LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement IsConvertError<size_t>(countValidationSamples)");
      return Error_IllegalParamValue;
   }
   const size_t cValidationSamples = static_cast<size_t>(countValidationSamples);

   if (UNLIKELY(IsAddError(cTrainingSamples, cValidationSamples))) {
      LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement IsAddError(countTrainingSamples, countValidationSamples))");
      return Error_IllegalParamValue;
   }

   size_t cSamples = cTrainingSamples + cValidationSamples;
   if (UNLIKELY(cSamples == 0)) {
      // there's nothing for us to fill the sampleCountsOut with
      return Error_None;
   }

   if (countTargetClasses <= 0) {
      LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement countTargetClasses can't be negative or zero");
      return Error_IllegalParamValue;
   }
   if (IsConvertError<size_t>(countTargetClasses)) {
      LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement IsConvertError<size_t>(countTargetClasses)");
      return Error_IllegalParamValue;
   }
   const size_t cTargetClasses = static_cast<size_t>(countTargetClasses);

   if (UNLIKELY(IsMultiplyError(sizeof(*sampleCountsOut), cSamples))) {
      LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement IsMultiplyError(sizeof(*sampleCountsOut), cSamples)");
      return Error_IllegalParamValue;
   }

   if (UNLIKELY(IsMultiplyError(sizeof(TargetSamplingCounts), cTargetClasses))) {
      LOG_0(TraceLevelWarning, "WARNING StratifiedSamplingWithoutReplacement IsMultiplyError(sizeof(TargetSamplingCounts), cTargetClasses)");
      return Error_OutOfMemory;
   }

   if (UNLIKELY(cTrainingSamples < cTargetClasses)) {
      LOG_0(TraceLevelWarning, "WARNING StratifiedSamplingWithoutReplacement cTrainingSamples < cTargetClasses");
   }

   if (UNLIKELY(cValidationSamples < cTargetClasses)) {
      LOG_0(TraceLevelWarning, "WARNING StratifiedSamplingWithoutReplacement cValidationSamples < cTargetClasses");
   }

   const size_t targetSamplingCountsSize = sizeof(TargetSamplingCounts) * cTargetClasses;
   TargetSamplingCounts* pTargetSamplingCounts = static_cast<TargetSamplingCounts*>(malloc(targetSamplingCountsSize));

   if (UNLIKELY(nullptr == pTargetSamplingCounts)) {
      LOG_0(TraceLevelWarning, "WARNING StratifiedSamplingWithoutReplacement out of memory on aTargetSamplingCounts");
      return Error_OutOfMemory;
   }

   memset(pTargetSamplingCounts, 0, targetSamplingCountsSize);

   // calculate number of samples per label in the target
   for (size_t i = 0; i < cSamples; i++) {
      IntEbmType label = targets[i];

      if (UNLIKELY(label < 0 || label >= countTargetClasses)) {
         LOG_0(TraceLevelError, "ERROR StratifiedSamplingWithoutReplacement label >= cTargetClasses");
         free(pTargetSamplingCounts);
         return Error_IllegalParamValue;
      }

      (pTargetSamplingCounts + label)->m_cTotalRemaining++;
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
         size_t cClassTotalRemaining = (pTargetSamplingCounts + iTargetClass)->m_cTotalRemaining;
         double fTrainingPerClass = std::floor(idealTrainingProportion * cClassTotalRemaining);
         size_t cTrainingPerClass = static_cast<size_t>(fTrainingPerClass);
         if (0 < cTrainingPerClass) {
            --cTrainingPerClass;
         }
         cClassesWithSamples = (cClassTotalRemaining > 0) ? cClassesWithSamples + 1 : cClassesWithSamples;
         (pTargetSamplingCounts + iTargetClass)->m_cTraining = cTrainingPerClass;
         EBM_ASSERT(cTrainingPerClass <= globalLeftover);
         globalLeftover -= cTrainingPerClass;
      }

      EBM_ASSERT(cClassesWithSamples <= globalLeftover);
   }

   EBM_ASSERT(globalLeftover <= cSamples);

   const size_t mostImprovedClassesCapacity = sizeof(size_t) * cTargetClasses;
   size_t* pMostImprovedClasses = static_cast<size_t*>(malloc(mostImprovedClassesCapacity));

   if (UNLIKELY(nullptr == pMostImprovedClasses)) {
      LOG_0(TraceLevelWarning, "WARNING StratifiedSamplingWithoutReplacement out of memory on pMostImprovedClasses");
      free(pTargetSamplingCounts);
      return Error_OutOfMemory;
   }

   RandomDeterministic randomDeterministic;
   randomDeterministic.InitializeUnsigned(randomSeed, k_stratifiedSamplingWithoutReplacementRandomizationMix);

   for (size_t iLeftover = 0; iLeftover < globalLeftover; iLeftover++) {
      double maxImprovement = std::numeric_limits<double>::lowest();
      size_t mostImprovedClassesSize = 0;
      memset(pMostImprovedClasses, 0, mostImprovedClassesCapacity);

      for (size_t iTargetClass = 0; iTargetClass < cTargetClasses; iTargetClass++) {
         const size_t cClassTraining = (pTargetSamplingCounts + iTargetClass)->m_cTraining;
         const size_t cClassRemaining = (pTargetSamplingCounts + iTargetClass)->m_cTotalRemaining;

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
            memset(pMostImprovedClasses, 0, mostImprovedClassesCapacity);
            mostImprovedClassesSize = 0;
         }
         
         if (improvement == maxImprovement) {
            *(pMostImprovedClasses + mostImprovedClassesSize) = iTargetClass;
            ++mostImprovedClassesSize;
         }
      }
      EBM_ASSERT(std::numeric_limits<double>::lowest() != maxImprovement);

      // If more than one class has the same max improvement, randomly select between the classes
      // to give the leftover to.
      size_t iRandom = randomDeterministic.NextFast(mostImprovedClassesSize);
      size_t classToImprove = *(pMostImprovedClasses + iRandom);
      (pTargetSamplingCounts + classToImprove)->m_cTraining++;
   }

#ifndef NDEBUG
   const TargetSamplingCounts* pTargetSamplingCountsEnd = pTargetSamplingCounts + cTargetClasses;
   size_t assignedTrainingCount = 0;

   for (TargetSamplingCounts* pTargetSamplingCountsCur = pTargetSamplingCounts;
      pTargetSamplingCountsEnd != pTargetSamplingCountsCur;
      ++pTargetSamplingCountsCur) {
      assignedTrainingCount += pTargetSamplingCountsCur->m_cTraining;
   }

   EBM_ASSERT(assignedTrainingCount == cTrainingSamples);
#endif

   for (size_t iSample = 0; iSample < cSamples; iSample++) {
      TargetSamplingCounts* pTargetSample = pTargetSamplingCounts + targets[iSample];
      EBM_ASSERT(pTargetSample->m_cTotalRemaining > 0);
      const size_t iRandom = randomDeterministic.NextFast(pTargetSample->m_cTotalRemaining);
      const bool bTrainingSample = UNPREDICTABLE(iRandom < pTargetSample->m_cTraining);

      if (UNPREDICTABLE(bTrainingSample)) {
         --pTargetSample->m_cTraining;
         sampleCountsOut[iSample] = BagEbmType{ 1 };
      }
      else {
         sampleCountsOut[iSample] = BagEbmType{ -1 };
      }

      --pTargetSample->m_cTotalRemaining;
   }

#ifndef NDEBUG
   for (TargetSamplingCounts* pTargetSamplingCountsCur = pTargetSamplingCounts;
      pTargetSamplingCountsEnd != pTargetSamplingCountsCur;
      ++pTargetSamplingCountsCur) {
      EBM_ASSERT(pTargetSamplingCountsCur->m_cTraining == 0);
      EBM_ASSERT(pTargetSamplingCountsCur->m_cTotalRemaining == 0);
   }
#endif

   free(pTargetSamplingCounts);
   free(pMostImprovedClasses);

   LOG_COUNTED_0(
      &g_cLogExitStratifiedSamplingWithoutReplacementParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Exited StratifiedSamplingWithoutReplacement"
   );

   return Error_None;
}
WARNING_POP

extern ErrorEbmType Unbag(
   const size_t cSamples,
   const BagEbmType * const aBag,
   size_t * const pcTrainingSamplesOut,
   size_t * const pcValidationSamplesOut
) {
   EBM_ASSERT(nullptr != pcTrainingSamplesOut);
   EBM_ASSERT(nullptr != pcValidationSamplesOut);

   size_t cTrainingSamples = cSamples;
   size_t cValidationSamples = 0;
   if(nullptr != aBag) {
      cTrainingSamples = 0;
      if(0 != cSamples) {
         const BagEbmType * pBag = aBag;
         const BagEbmType * const pBagEnd = aBag + cSamples;
         do {
            BagEbmType sampleDefinition = *pBag;
            if(sampleDefinition < BagEbmType { 0 }) {
               if(IsConvertError<ptrdiff_t>(sampleDefinition)) {
                  LOG_0(TraceLevelError, "ERROR Unbag IsConvertError<ptrdiff_t>(sampleDefinition)");
                  return Error_IllegalParamValue;
               }
               ptrdiff_t cSampleDefinitionSigned = static_cast<ptrdiff_t>(sampleDefinition);
               // by creating a ptrdiff_t with "ptrdiff_t { ... }" the compiler is suposed to give us an 
               // error if for some reason the negation of the max fails
               if(cSampleDefinitionSigned < ptrdiff_t { -std::numeric_limits<ptrdiff_t>::max() }) {
                  LOG_0(TraceLevelError, "ERROR Unbag cSampleDefinitionSigned < ptrdiff_t { -std::numeric_limits<ptrdiff_t>::max() }");
                  return Error_IllegalParamValue;
               }
               cSampleDefinitionSigned = -cSampleDefinitionSigned;
               const size_t cSampleDefinition = static_cast<size_t>(cSampleDefinitionSigned);
               if(IsAddError(cValidationSamples, cSampleDefinition)) {
                  LOG_0(TraceLevelError, "ERROR Unbag IsAddError(cValidationSamples, cSampleDefinition)");
                  return Error_IllegalParamValue;
               }
               cValidationSamples += cSampleDefinition;
            } else {
               if(IsConvertError<size_t>(sampleDefinition)) {
                  LOG_0(TraceLevelError, "ERROR Unbag IsConvertError<size_t>(sampleDefinition)");
                  return Error_IllegalParamValue;
               }
               const size_t cSampleDefinition = static_cast<size_t>(sampleDefinition);
               if(IsAddError(cTrainingSamples, cSampleDefinition)) {
                  LOG_0(TraceLevelError, "ERROR Unbag IsAddError(cTrainingSamples, cSampleDefinition)");
                  return Error_IllegalParamValue;
               }
               cTrainingSamples += cSampleDefinition;
            }
            ++pBag;
         } while(pBagEnd != pBag);
      }
   }
   *pcTrainingSamplesOut = cTrainingSamples;
   *pcValidationSamplesOut = cValidationSamples;
   return Error_None;
}

INLINE_RELEASE_UNTEMPLATED static bool CheckWeightsEqual(
   const BagEbmType direction,
   const size_t cAllSamples,
   const BagEbmType * pBag,
   const FloatFast * pWeights
) {
   EBM_ASSERT(BagEbmType { -1 } == direction || BagEbmType { 1 } == direction);
   EBM_ASSERT(1 <= cAllSamples);
   EBM_ASSERT(nullptr != pWeights);

   FloatFast firstWeight = std::numeric_limits<FloatFast>::quiet_NaN();
   const FloatFast * const pWeightsEnd = pWeights + cAllSamples;
   const bool isLoopTraining = BagEbmType { 0 } < direction;
   do {
      BagEbmType countBagged = 1;
      if(nullptr != pBag) {
         countBagged = *pBag;
         ++pBag;
      }
      if(BagEbmType { 0 } != countBagged) {
         const bool isItemTraining = BagEbmType { 0 } < countBagged;
         if(isLoopTraining == isItemTraining) {
            const FloatFast weight = *pWeights;
            // this relies on the property that NaN is not equal to everything, including NaN
            if(UNLIKELY(firstWeight != weight)) {
               if(!std::isnan(firstWeight)) {
                  // if firstWeight or *pWeight is NaN this should trigger, which is good since we don't want to
                  // replace arrays containing all NaN weights with weights of 1
                  return false;
               }
               firstWeight = weight;
            }
         }
      }
      ++pWeights;
   } while(LIKELY(pWeightsEnd != pWeights));
   return true;
}

extern ErrorEbmType ExtractWeights(
   const unsigned char * const pDataSetShared,
   const BagEbmType direction,
   const size_t cAllSamples,
   const BagEbmType * const aBag,
   const size_t cSetSamples,
   FloatFast ** ppWeightsOut
) {
   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbmType { -1 } == direction || BagEbmType { 1 } == direction);
   EBM_ASSERT(0 < cSetSamples);
   EBM_ASSERT(cSetSamples <= cAllSamples);
   EBM_ASSERT(0 < cAllSamples); // from the previous two rules
   EBM_ASSERT(nullptr != ppWeightsOut);
   EBM_ASSERT(nullptr == *ppWeightsOut);

   const FloatFast * const aWeights = GetDataSetSharedWeight(pDataSetShared, 0);
   EBM_ASSERT(nullptr != aWeights);
   if(!CheckWeightsEqual(direction, cAllSamples, aBag, aWeights)) {
      const size_t cBytes = sizeof(*aWeights) * cSetSamples;
      FloatFast * const aRet = static_cast<FloatFast *>(malloc(cBytes));
      if(UNLIKELY(nullptr == aRet)) {
         LOG_0(TraceLevelWarning, "WARNING ExtractWeights nullptr == aRet");
         return Error_OutOfMemory;
      }
      *ppWeightsOut = aRet;

      const BagEbmType * pBag = aBag;
      const FloatFast * pWeightFrom = aWeights;
      FloatFast * pWeightTo = aRet;
      FloatFast * pWeightToEnd = aRet + cSetSamples;
      const bool isLoopTraining = BagEbmType { 0 } < direction;
      do {
         BagEbmType countBagged = 1;
         if(nullptr != pBag) {
            countBagged = *pBag;
            ++pBag;
         }
         if(BagEbmType { 0 } != countBagged) {
            const bool isItemTraining = BagEbmType { 0 } < countBagged;
            if(isLoopTraining == isItemTraining) {
               const FloatFast weight = *pWeightFrom;
               do {
                  EBM_ASSERT(pWeightTo < pWeightToEnd);
                  *pWeightTo = weight;
                  ++pWeightTo;
                  countBagged -= direction;
               } while(BagEbmType { 0 } != countBagged);
            }
         }
         ++pWeightFrom;
      } while(pWeightToEnd != pWeightTo);
   }
   return Error_None;
}


} // DEFINED_ZONE_NAME
