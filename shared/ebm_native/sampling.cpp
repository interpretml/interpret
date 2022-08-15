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

EBM_API_BODY SeedEbm EBM_CALLING_CONVENTION GenerateSeed(
   SeedEbm seed,
   SeedEbm randomMix
) {
   RandomDeterministic randomDeterministic;
   // this is a bit inefficient in that we go through a complete regeneration of the internal state,
   // but it gives us a simple interface
   randomDeterministic.InitializeSigned(seed, randomMix);
   SeedEbm ret = randomDeterministic.NextSeed();
   return ret;
}

// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterSampleWithoutReplacement = 5;
static int g_cLogExitSampleWithoutReplacement = 5;

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION SampleWithoutReplacement(
   BoolEbm isDeterministic,
   SeedEbm seed,
   IntEbm countTrainingSamples,
   IntEbm countValidationSamples,
   BagEbm * bagOut
) {
   LOG_COUNTED_N(
      &g_cLogEnterSampleWithoutReplacement,
      Trace_Info,
      Trace_Verbose,
      "Entered SampleWithoutReplacement: "
      "isDeterministic=%s, "
      "seed=%" SeedEbmPrintf ", "
      "countTrainingSamples=%" IntEbmPrintf ", "
      "countValidationSamples=%" IntEbmPrintf ", "
      "bagOut=%p"
      ,
      ObtainTruth(isDeterministic),
      seed,
      countTrainingSamples,
      countValidationSamples,
      static_cast<void *>(bagOut)
   );

   if(UNLIKELY(nullptr == bagOut)) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement nullptr == bagOut");
      return Error_IllegalParamVal;
   }

   if(UNLIKELY(countTrainingSamples < IntEbm { 0 })) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement countTrainingSamples < IntEbm { 0 }");
      return Error_IllegalParamVal;
   }
   if(UNLIKELY(IsConvertError<size_t>(countTrainingSamples))) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement IsConvertError<size_t>(countTrainingSamples)");
      return Error_IllegalParamVal;
   }
   const size_t cTrainingSamples = static_cast<size_t>(countTrainingSamples);

   if(UNLIKELY(countValidationSamples < IntEbm { 0 })) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement countValidationSamples < IntEbm { 0 }");
      return Error_IllegalParamVal;
   }
   if(UNLIKELY(IsConvertError<size_t>(countValidationSamples))) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement IsConvertError<size_t>(countValidationSamples)");
      return Error_IllegalParamVal;
   }
   const size_t cValidationSamples = static_cast<size_t>(countValidationSamples);

   if(UNLIKELY(IsAddError(cTrainingSamples, cValidationSamples))) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement IsAddError(cTrainingSamples, cValidationSamples)");
      return Error_IllegalParamVal;
   }
   size_t cSamplesRemaining = cTrainingSamples + cValidationSamples;
   if(UNLIKELY(size_t { 0 } == cSamplesRemaining)) {
      // there's nothing for us to fill the array with
      return Error_None;
   }
   if(UNLIKELY(IsMultiplyError(sizeof(*bagOut), cSamplesRemaining))) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement IsMultiplyError(sizeof(*bagOut), cSamplesRemaining)");
      return Error_IllegalParamVal;
   }

   size_t cTrainingRemaining = cTrainingSamples;

   BagEbm * pSampleReplicationOut = bagOut;
   if(EBM_FALSE != isDeterministic) {
      RandomDeterministic randomGenerator;
      randomGenerator.InitializeUnsigned(seed, k_samplingWithoutReplacementRandomizationMix);
      do {
         const size_t iRandom = randomGenerator.NextFast(cSamplesRemaining);
         const bool bTrainingSample = UNPREDICTABLE(iRandom < cTrainingRemaining);
         cTrainingRemaining = UNPREDICTABLE(bTrainingSample) ? cTrainingRemaining - size_t { 1 } : cTrainingRemaining;
         *pSampleReplicationOut = UNPREDICTABLE(bTrainingSample) ? BagEbm { 1 } : BagEbm { -1 };
         ++pSampleReplicationOut;
         --cSamplesRemaining;
      } while(0 != cSamplesRemaining);
   } else {
      try {
         RandomNondeterministic<size_t> randomGenerator;
         do {
            const size_t iRandom = randomGenerator.NextFast(cSamplesRemaining);
            const bool bTrainingSample = UNPREDICTABLE(iRandom < cTrainingRemaining);
            cTrainingRemaining = UNPREDICTABLE(bTrainingSample) ? cTrainingRemaining - size_t { 1 } : cTrainingRemaining;
            *pSampleReplicationOut = UNPREDICTABLE(bTrainingSample) ? BagEbm { 1 } : BagEbm { -1 };
            ++pSampleReplicationOut;
            --cSamplesRemaining;
         } while(0 != cSamplesRemaining);
      } catch(const std::bad_alloc &) {
         LOG_0(Trace_Warning, "WARNING SampleWithoutReplacement Out of memory in std::random_device");
         return Error_OutOfMemory;
      } catch(...) {
         LOG_0(Trace_Warning, "WARNING SampleWithoutReplacement Unknown error in std::random_device");
         return Error_UnexpectedInternal;
      }
   }
   EBM_ASSERT(0 == cTrainingRemaining); // this should be all used up too now

   LOG_COUNTED_0(
      &g_cLogExitSampleWithoutReplacement,
      Trace_Info,
      Trace_Verbose,
      "Exited SampleWithoutReplacement"
   );
   return Error_None;
}


static int g_cLogEnterSampleWithoutReplacementStratified = 5;
static int g_cLogExitSampleWithoutReplacementStratified = 5;

WARNING_PUSH
WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION SampleWithoutReplacementStratified(
   BoolEbm isDeterministic,
   SeedEbm seed,
   IntEbm countClasses,
   IntEbm countTrainingSamples,
   IntEbm countValidationSamples,
   IntEbm * targets,
   BagEbm * bagOut
) {
   struct TargetSamplingCounts {
      size_t m_cTraining;
      size_t m_cTotalRemaining;
   };

   LOG_COUNTED_N(
      &g_cLogEnterSampleWithoutReplacementStratified,
      Trace_Info,
      Trace_Verbose,
      "Entered SampleWithoutReplacementStratified: "
      "isDeterministic=%s, "
      "seed=%" SeedEbmPrintf ", "
      "countClasses=%" IntEbmPrintf ", "
      "countTrainingSamples=%" IntEbmPrintf ", "
      "countValidationSamples=%" IntEbmPrintf ", "
      "targets=%p, "
      "bagOut=%p"
      ,
      ObtainTruth(isDeterministic),
      seed,
      countClasses,
      countTrainingSamples,
      countValidationSamples,
      static_cast<void*>(targets),
      static_cast<void*>(bagOut)
   );

   if (UNLIKELY(nullptr == targets)) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified nullptr == targets");
      return Error_IllegalParamVal;
   }

   if (UNLIKELY(nullptr == bagOut)) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified nullptr == bagOut");
      return Error_IllegalParamVal;
   }

   if (UNLIKELY(countTrainingSamples < IntEbm{ 0 })) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified countTrainingSamples < IntEbm{ 0 }");
      return Error_IllegalParamVal;
   }
   if (UNLIKELY(IsConvertError<size_t>(countTrainingSamples))) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified IsConvertError<size_t>(countTrainingSamples)");
      return Error_IllegalParamVal;
   }
   const size_t cTrainingSamples = static_cast<size_t>(countTrainingSamples);

   if (UNLIKELY(countValidationSamples < IntEbm{ 0 })) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified countValidationSamples < IntEbm{ 0 }");
      return Error_IllegalParamVal;
   }
   if (UNLIKELY(IsConvertError<size_t>(countValidationSamples))) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified IsConvertError<size_t>(countValidationSamples)");
      return Error_IllegalParamVal;
   }
   const size_t cValidationSamples = static_cast<size_t>(countValidationSamples);

   if (UNLIKELY(IsAddError(cTrainingSamples, cValidationSamples))) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified IsAddError(countTrainingSamples, countValidationSamples))");
      return Error_IllegalParamVal;
   }

   size_t cSamples = cTrainingSamples + cValidationSamples;
   if (UNLIKELY(cSamples == 0)) {
      // there's nothing for us to fill the bagOut with
      return Error_None;
   }

   if (countClasses <= 0) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified countClasses can't be negative or zero");
      return Error_IllegalParamVal;
   }
   if (IsConvertError<size_t>(countClasses)) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified IsConvertError<size_t>(countClasses)");
      return Error_IllegalParamVal;
   }
   const size_t cClasses = static_cast<size_t>(countClasses);

   if (UNLIKELY(IsMultiplyError(sizeof(*bagOut), cSamples))) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified IsMultiplyError(sizeof(*bagOut), cSamples)");
      return Error_IllegalParamVal;
   }

   if (UNLIKELY(IsMultiplyError(sizeof(TargetSamplingCounts), cClasses))) {
      LOG_0(Trace_Warning, "WARNING SampleWithoutReplacementStratified IsMultiplyError(sizeof(TargetSamplingCounts), cClasses)");
      return Error_OutOfMemory;
   }

   if (UNLIKELY(cTrainingSamples < cClasses)) {
      LOG_0(Trace_Warning, "WARNING SampleWithoutReplacementStratified cTrainingSamples < cClasses");
   }

   if (UNLIKELY(cValidationSamples < cClasses)) {
      LOG_0(Trace_Warning, "WARNING SampleWithoutReplacementStratified cValidationSamples < cClasses");
   }

   if(EBM_FALSE == isDeterministic) {
      // SampleWithoutReplacementStratified is not called when building a differentially private model, so
      // we can use low-quality non-determinism.  Generate a non-deterministic seed
      try {
         RandomNondeterministic<uint32_t> randomGenerator;
         seed = randomGenerator.NextSeed();
      } catch(const std::bad_alloc &) {
         LOG_0(Trace_Warning, "WARNING SampleWithoutReplacementStratified Out of memory in std::random_device");
         return Error_OutOfMemory;
      } catch(...) {
         LOG_0(Trace_Warning, "WARNING SampleWithoutReplacementStratified Unknown error in std::random_device");
         return Error_UnexpectedInternal;
      }
   }

   const size_t targetSamplingCountsSize = sizeof(TargetSamplingCounts) * cClasses;
   TargetSamplingCounts* pTargetSamplingCounts = static_cast<TargetSamplingCounts*>(malloc(targetSamplingCountsSize));

   if (UNLIKELY(nullptr == pTargetSamplingCounts)) {
      LOG_0(Trace_Warning, "WARNING SampleWithoutReplacementStratified out of memory on aTargetSamplingCounts");
      return Error_OutOfMemory;
   }

   memset(pTargetSamplingCounts, 0, targetSamplingCountsSize);

   // calculate number of samples per label in the target
   for (size_t i = 0; i < cSamples; i++) {
      IntEbm label = targets[i];

      if (UNLIKELY(label < 0 || label >= countClasses)) {
         LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified label >= cClasses");
         free(pTargetSamplingCounts);
         return Error_IllegalParamVal;
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

   if (cTrainingSamples > cClasses) {
      size_t cClassesWithSamples = 0;

      for (size_t iClass = 0; iClass < cClasses; iClass++) {
         size_t cClassTotalRemaining = (pTargetSamplingCounts + iClass)->m_cTotalRemaining;
         double fTrainingPerClass = std::floor(idealTrainingProportion * cClassTotalRemaining);
         size_t cTrainingPerClass = static_cast<size_t>(fTrainingPerClass);
         if (0 < cTrainingPerClass) {
            --cTrainingPerClass;
         }
         cClassesWithSamples = (cClassTotalRemaining > 0) ? cClassesWithSamples + 1 : cClassesWithSamples;
         (pTargetSamplingCounts + iClass)->m_cTraining = cTrainingPerClass;
         EBM_ASSERT(cTrainingPerClass <= globalLeftover);
         globalLeftover -= cTrainingPerClass;
      }

      EBM_ASSERT(cClassesWithSamples <= globalLeftover);
   }

   EBM_ASSERT(globalLeftover <= cSamples);

   const size_t mostImprovedClassesCapacity = sizeof(size_t) * cClasses;
   size_t* pMostImprovedClasses = static_cast<size_t*>(malloc(mostImprovedClassesCapacity));

   if (UNLIKELY(nullptr == pMostImprovedClasses)) {
      LOG_0(Trace_Warning, "WARNING SampleWithoutReplacementStratified out of memory on pMostImprovedClasses");
      free(pTargetSamplingCounts);
      return Error_OutOfMemory;
   }

   RandomDeterministic randomDeterministic;
   randomDeterministic.InitializeUnsigned(seed, k_sampleWithoutReplacementStratifiedRandomizationMix);

   for (size_t iLeftover = 0; iLeftover < globalLeftover; iLeftover++) {
      double maxImprovement = std::numeric_limits<double>::lowest();
      size_t mostImprovedClassesSize = 0;
      memset(pMostImprovedClasses, 0, mostImprovedClassesCapacity);

      for (size_t iClass = 0; iClass < cClasses; iClass++) {
         const size_t cClassTraining = (pTargetSamplingCounts + iClass)->m_cTraining;
         const size_t cClassRemaining = (pTargetSamplingCounts + iClass)->m_cTotalRemaining;

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
            *(pMostImprovedClasses + mostImprovedClassesSize) = iClass;
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
   const TargetSamplingCounts * pTargetSamplingCountsEnd = pTargetSamplingCounts + cClasses;
   size_t assignedTrainingCount = 0;

   for (TargetSamplingCounts * pTargetSamplingCountsCur = pTargetSamplingCounts;
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
         bagOut[iSample] = BagEbm{ 1 };
      }
      else {
         bagOut[iSample] = BagEbm{ -1 };
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
      &g_cLogExitSampleWithoutReplacementStratified,
      Trace_Info,
      Trace_Verbose,
      "Exited SampleWithoutReplacementStratified"
   );

   return Error_None;
}
WARNING_POP

extern ErrorEbm Unbag(
   const size_t cSamples,
   const BagEbm * const aBag,
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
         const BagEbm * pSampleReplication = aBag;
         const BagEbm * const pSampleReplicationEnd = aBag + cSamples;
         do {
            const BagEbm replication = *pSampleReplication;
            if(replication < BagEbm { 0 }) {
               if(IsConvertError<ptrdiff_t>(replication)) {
                  LOG_0(Trace_Error, "ERROR Unbag IsConvertError<ptrdiff_t>(replication)");
                  return Error_IllegalParamVal;
               }
               ptrdiff_t replicationSigned = static_cast<ptrdiff_t>(replication);
               // by creating a ptrdiff_t with "ptrdiff_t { ... }" the compiler is suposed to give us an 
               // error if for some reason the negation of the max fails
               if(replicationSigned < ptrdiff_t { -std::numeric_limits<ptrdiff_t>::max() }) {
                  LOG_0(Trace_Error, "ERROR Unbag replicationSigned < ptrdiff_t { -std::numeric_limits<ptrdiff_t>::max() }");
                  return Error_IllegalParamVal;
               }
               replicationSigned = -replicationSigned;
               const size_t replicationUnsigned = static_cast<size_t>(replicationSigned);
               if(IsAddError(cValidationSamples, replicationUnsigned)) {
                  LOG_0(Trace_Error, "ERROR Unbag IsAddError(cValidationSamples, replicationUnsigned)");
                  return Error_IllegalParamVal;
               }
               cValidationSamples += replicationUnsigned;
            } else {
               if(IsConvertError<size_t>(replication)) {
                  LOG_0(Trace_Error, "ERROR Unbag IsConvertError<size_t>(replication)");
                  return Error_IllegalParamVal;
               }
               const size_t replicationUnsigned = static_cast<size_t>(replication);
               if(IsAddError(cTrainingSamples, replicationUnsigned)) {
                  LOG_0(Trace_Error, "ERROR Unbag IsAddError(cTrainingSamples, replicationUnsigned)");
                  return Error_IllegalParamVal;
               }
               cTrainingSamples += replicationUnsigned;
            }
            ++pSampleReplication;
         } while(pSampleReplicationEnd != pSampleReplication);
      }
   }
   *pcTrainingSamplesOut = cTrainingSamples;
   *pcValidationSamplesOut = cValidationSamples;
   return Error_None;
}

INLINE_RELEASE_UNTEMPLATED static bool CheckWeightsEqual(
   const BagEbm direction,
   const size_t cAllSamples,
   const BagEbm * const aBag,
   const FloatFast * pWeights
) {
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(1 <= cAllSamples);
   EBM_ASSERT(nullptr != pWeights);

   FloatFast firstWeight = std::numeric_limits<FloatFast>::quiet_NaN();
   const FloatFast * const pWeightsEnd = pWeights + cAllSamples;
   const bool isLoopTraining = BagEbm { 0 } < direction;
   const BagEbm * pSampleReplication = aBag;
   do {
      BagEbm replication = 1;
      if(nullptr != pSampleReplication) {
         replication = *pSampleReplication;
         ++pSampleReplication;
      }
      if(BagEbm { 0 } != replication) {
         const bool isItemTraining = BagEbm { 0 } < replication;
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

extern ErrorEbm ExtractWeights(
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const size_t cAllSamples,
   const BagEbm * const aBag,
   const size_t cSetSamples,
   FloatFast ** ppWeightsOut
) {
   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
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
         LOG_0(Trace_Warning, "WARNING ExtractWeights nullptr == aRet");
         return Error_OutOfMemory;
      }
      *ppWeightsOut = aRet;

      const BagEbm * pSampleReplication = aBag;
      const FloatFast * pWeightFrom = aWeights;
      FloatFast * pWeightTo = aRet;
      FloatFast * pWeightToEnd = aRet + cSetSamples;
      const bool isLoopTraining = BagEbm { 0 } < direction;
      do {
         BagEbm replication = 1;
         if(nullptr != pSampleReplication) {
            replication = *pSampleReplication;
            ++pSampleReplication;
         }
         if(BagEbm { 0 } != replication) {
            const bool isItemTraining = BagEbm { 0 } < replication;
            if(isLoopTraining == isItemTraining) {
               const FloatFast weight = *pWeightFrom;
               do {
                  EBM_ASSERT(pWeightTo < pWeightToEnd);
                  *pWeightTo = weight;
                  ++pWeightTo;
                  replication -= direction;
               } while(BagEbm { 0 } != replication);
            }
         }
         ++pWeightFrom;
      } while(pWeightToEnd != pWeightTo);
   }
   return Error_None;
}


} // DEFINED_ZONE_NAME
