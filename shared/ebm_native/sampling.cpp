// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include "ebm_native.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // LIKELY
#include "zones.h"

#include "RandomDeterministic.hpp"
#include "RandomNondeterministic.hpp"
#include "dataset_shared.hpp" // GetDataSetSharedWeight

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterSampleWithoutReplacement = 5;
static int g_cLogExitSampleWithoutReplacement = 5;

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION SampleWithoutReplacement(
   void * rng,
   IntEbm countTrainingSamples,
   IntEbm countValidationSamples,
   BagEbm * bagOut
) {
   LOG_COUNTED_N(
      &g_cLogEnterSampleWithoutReplacement,
      Trace_Info,
      Trace_Verbose,
      "Entered SampleWithoutReplacement: "
      "rng=%p, "
      "countTrainingSamples=%" IntEbmPrintf ", "
      "countValidationSamples=%" IntEbmPrintf ", "
      "bagOut=%p"
      ,
      rng,
      countTrainingSamples,
      countValidationSamples,
      static_cast<void *>(bagOut)
   );

   if(UNLIKELY(IsConvertError<size_t>(countTrainingSamples))) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement IsConvertError<size_t>(countTrainingSamples)");
      return Error_IllegalParamVal;
   }
   const size_t cTrainingSamples = static_cast<size_t>(countTrainingSamples);

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
      LOG_COUNTED_0(
         &g_cLogExitSampleWithoutReplacement,
         Trace_Info,
         Trace_Verbose,
         "Exited SampleWithoutReplacement with zero elements"
      );
      return Error_None;
   }
   if(UNLIKELY(IsMultiplyError(sizeof(*bagOut), cSamplesRemaining))) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement IsMultiplyError(sizeof(*bagOut), cSamplesRemaining)");
      return Error_IllegalParamVal;
   }

   if(UNLIKELY(nullptr == bagOut)) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement nullptr == bagOut");
      return Error_IllegalParamVal;
   }

   size_t cTrainingRemaining = cTrainingSamples;

   BagEbm * pSampleReplicationOut = bagOut;
   if(nullptr != rng) {
      RandomDeterministic * const pRng = reinterpret_cast<RandomDeterministic *>(rng);
      // the compiler understands the internal state of this RNG and can locate its internal state into CPU registers
      RandomDeterministic cpuRng;
      cpuRng.Initialize(*pRng); // move the RNG from memory into CPU registers
      do {
         const size_t iRandom = cpuRng.NextFast(cSamplesRemaining);
         const bool bTrainingSample = UNPREDICTABLE(iRandom < cTrainingRemaining);
         cTrainingRemaining -= UNPREDICTABLE(bTrainingSample) ? size_t { 1 } : size_t { 0 };
         *pSampleReplicationOut = UNPREDICTABLE(bTrainingSample) ? BagEbm { 1 } : BagEbm { -1 };
         ++pSampleReplicationOut;
         --cSamplesRemaining;
      } while(0 != cSamplesRemaining);
      pRng->Initialize(cpuRng); // move the RNG from the CPU registers back into memory
   } else {
      try {
         RandomNondeterministic<size_t> randomGenerator;
         do {
            const size_t iRandom = randomGenerator.NextFast(cSamplesRemaining);
            const bool bTrainingSample = UNPREDICTABLE(iRandom < cTrainingRemaining);
            cTrainingRemaining -= UNPREDICTABLE(bTrainingSample) ? size_t { 1 } : size_t { 0 };
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


EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION SampleWithoutReplacementStratified(
   void * rng,
   IntEbm countClasses,
   IntEbm countTrainingSamples,
   IntEbm countValidationSamples,
   const IntEbm * targets,
   BagEbm * bagOut
) {
   struct TargetClass {
      size_t m_cTrainingSamples;
      size_t m_cSamples;
   };

   LOG_COUNTED_N(
      &g_cLogEnterSampleWithoutReplacementStratified,
      Trace_Info,
      Trace_Verbose,
      "Entered SampleWithoutReplacementStratified: "
      "rng=%p, "
      "countClasses=%" IntEbmPrintf ", "
      "countTrainingSamples=%" IntEbmPrintf ", "
      "countValidationSamples=%" IntEbmPrintf ", "
      "targets=%p, "
      "bagOut=%p"
      ,
      rng,
      countClasses,
      countTrainingSamples,
      countValidationSamples,
      static_cast<const void *>(targets),
      static_cast<void *>(bagOut)
   );

   if(UNLIKELY(IsConvertError<size_t>(countTrainingSamples))) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified IsConvertError<size_t>(countTrainingSamples)");
      return Error_IllegalParamVal;
   }
   const size_t cTrainingSamples = static_cast<size_t>(countTrainingSamples);

   if(UNLIKELY(IsConvertError<size_t>(countValidationSamples))) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified IsConvertError<size_t>(countValidationSamples)");
      return Error_IllegalParamVal;
   }
   const size_t cValidationSamples = static_cast<size_t>(countValidationSamples);

   if(UNLIKELY(IsAddError(cTrainingSamples, cValidationSamples))) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified IsAddError(countTrainingSamples, countValidationSamples))");
      return Error_IllegalParamVal;
   }

   const size_t cSamples = cTrainingSamples + cValidationSamples;
   if(UNLIKELY(size_t { 0 } == cSamples)) {
      LOG_COUNTED_0(
         &g_cLogExitSampleWithoutReplacementStratified,
         Trace_Info,
         Trace_Verbose,
         "Exited SampleWithoutReplacementStratified with zero samples"
      );
      return Error_None;
   }

   if(UNLIKELY(IsMultiplyError(EbmMax(sizeof(*targets), sizeof(*bagOut)), cSamples))) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified IsMultiplyError(EbmMax(sizeof(*targets), sizeof(*bagOut)), cSamples)");
      return Error_IllegalParamVal;
   }

   if(UNLIKELY(nullptr == targets)) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified nullptr == targets");
      return Error_IllegalParamVal;
   }

   if(UNLIKELY(nullptr == bagOut)) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified nullptr == bagOut");
      return Error_IllegalParamVal;
   }

   if(UNLIKELY(countClasses <= IntEbm { 0 })) {
      // countClasses cannot be zero since 1 <= cSamples
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified countClasses <= IntEbm { 0 }");
      return Error_IllegalParamVal;
   }
   if(IsConvertError<size_t>(countClasses)) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified IsConvertError<size_t>(countClasses)");
      return Error_IllegalParamVal;
   }
   const size_t cClasses = static_cast<size_t>(countClasses);
   EBM_ASSERT(1 <= cClasses);

   if(UNLIKELY(IsMultiplyError(sizeof(TargetClass), cClasses))) {
      LOG_0(Trace_Warning, "WARNING SampleWithoutReplacementStratified IsMultiplyError(sizeof(TargetClass), cClasses)");
      return Error_OutOfMemory;
   }
   const size_t cBytesAllTargetClasses = sizeof(TargetClass) * cClasses;

   if(UNLIKELY(cTrainingSamples < cClasses)) {
      LOG_0(Trace_Warning, "WARNING SampleWithoutReplacementStratified cTrainingSamples < cClasses");
   }
   if(UNLIKELY(cValidationSamples < cClasses)) {
      LOG_0(Trace_Warning, "WARNING SampleWithoutReplacementStratified cValidationSamples < cClasses");
   }

   // the compiler understands the internal state of this RNG and can locate its internal state into CPU registers
   RandomDeterministic cpuRng;
   if(nullptr == rng) {
      // SampleWithoutReplacementStratified is not called when building a differentially private model, so
      // we can use low-quality non-determinism.  Generate a non-deterministic seed
      uint64_t seed;
      try {
         RandomNondeterministic<uint64_t> randomGenerator;
         seed = randomGenerator.Next(std::numeric_limits<uint64_t>::max());
      } catch(const std::bad_alloc &) {
         LOG_0(Trace_Warning, "WARNING SampleWithoutReplacementStratified Out of memory in std::random_device");
         return Error_OutOfMemory;
      } catch(...) {
         LOG_0(Trace_Warning, "WARNING SampleWithoutReplacementStratified Unknown error in std::random_device");
         return Error_UnexpectedInternal;
      }
      cpuRng.Initialize(seed);
   } else {
      const RandomDeterministic * const pRng = reinterpret_cast<RandomDeterministic *>(rng);
      cpuRng.Initialize(*pRng); // move the RNG from memory into CPU registers
   }

   EBM_ASSERT(1 <= cBytesAllTargetClasses); // we cannot have zero classes where there are any samples
   TargetClass * const aTargetClasses = static_cast<TargetClass *>(malloc(cBytesAllTargetClasses));
   if(UNLIKELY(nullptr == aTargetClasses)) {
      LOG_0(Trace_Warning, "WARNING SampleWithoutReplacementStratified out of memory on aTargetClasses");
      return Error_OutOfMemory;
   }
   // the C++ says memset legal to use for setting classes with unsigned integer types
   memset(aTargetClasses, 0, cBytesAllTargetClasses);

   const TargetClass * const pTargetClassesEnd = IndexByte(aTargetClasses, cBytesAllTargetClasses);
   EBM_ASSERT(pTargetClassesEnd != aTargetClasses);

   // determine number of samples per class in the target
   const IntEbm * pTargetInit = targets;
   const IntEbm * const pTargetsEnd = &targets[cSamples];
   do {
      const IntEbm indexClass = *pTargetInit;
      if(indexClass < 0) {
         LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified indexClass < 0");
         free(aTargetClasses);
         return Error_IllegalParamVal;
      }
      if(UNLIKELY(countClasses <= indexClass)) {
         LOG_0(Trace_Error, "ERROR SampleWithoutReplacementStratified countClasses <= indexClass");
         free(aTargetClasses);
         return Error_IllegalParamVal;
      }
      const size_t iClass = static_cast<size_t>(indexClass);
      ++aTargetClasses[iClass].m_cSamples;
      ++pTargetInit;
   } while(pTargetsEnd != pTargetInit);

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

   const double idealTrainingProportion = static_cast<double>(cTrainingSamples) / cSamples;
   EBM_ASSERT(!std::isnan(idealTrainingProportion)); // since we checked cSamples not zero above
   EBM_ASSERT(!std::isinf(idealTrainingProportion)); // since we checked cSamples not zero above
   EBM_ASSERT(0 <= idealTrainingProportion);
   EBM_ASSERT(idealTrainingProportion <= 1);

   size_t cLeftoverTrainingSamples = cTrainingSamples;
   if(cClasses < cTrainingSamples) {
      TargetClass * pTargetClass = aTargetClasses;
      do {
         size_t cClassSamples = pTargetClass->m_cSamples;
         double trainingPerClass = std::floor(idealTrainingProportion * cClassSamples);
         size_t cTrainingPerClass = static_cast<size_t>(trainingPerClass);
         if(0 < cTrainingPerClass) {
            --cTrainingPerClass;
         }
         pTargetClass->m_cTrainingSamples = cTrainingPerClass;
         EBM_ASSERT(cTrainingPerClass <= cLeftoverTrainingSamples);
         cLeftoverTrainingSamples -= cTrainingPerClass;
         ++pTargetClass;
      } while(pTargetClassesEnd != pTargetClass);
   }
   EBM_ASSERT(cLeftoverTrainingSamples <= cSamples);

   if(IsMultiplyError(sizeof(TargetClass *), cClasses)) {
      LOG_0(Trace_Warning, "WARNING SampleWithoutReplacementStratified IsMultiplyError(sizeof(TargetClass *), cClasses)");
      free(aTargetClasses);
      return Error_OutOfMemory;
   }
   TargetClass ** const apMostImprovedClasses = static_cast<TargetClass **>(malloc(sizeof(TargetClass *) * cClasses));
   if(UNLIKELY(nullptr == apMostImprovedClasses)) {
      LOG_0(Trace_Warning, "WARNING SampleWithoutReplacementStratified out of memory on apMostImprovedClasses");
      free(aTargetClasses);
      return Error_OutOfMemory;
   }

   while(0 != cLeftoverTrainingSamples) {
      double bestImprovement = -std::numeric_limits<double>::infinity();
      TargetClass ** ppMostImprovedClasses = apMostImprovedClasses;
      TargetClass * pTargetClass = aTargetClasses;
      do {
         const size_t cClassTrainingSamples = pTargetClass->m_cTrainingSamples;
         const size_t cClassSamples = pTargetClass->m_cSamples;

         if(cClassTrainingSamples != cClassSamples) {
            EBM_ASSERT(0 < cClassSamples); // because cClassTrainingSamples == cClassSamples if cClassSamples is zero

            double idealClassTraining = idealTrainingProportion * static_cast<double>(cClassSamples);
            double curTrainingDiff = idealClassTraining - cClassTrainingSamples;
            const size_t cClassTrainingSamplesPlusOne = cClassTrainingSamples + 1;
            double newTrainingDiff = idealClassTraining - cClassTrainingSamplesPlusOne;
            double improvement = (curTrainingDiff * curTrainingDiff) - (newTrainingDiff * newTrainingDiff);

            if(0 == cClassTrainingSamples) {
               // improvement should not be able to be larger than 9
               improvement += 32;
            } else if(cClassTrainingSamples + 1 == cClassSamples) {
               // improvement should not be able to be larger than 9
               improvement -= 32;
            }

            if(bestImprovement <= improvement) {
               ppMostImprovedClasses = 
                  LIKELY(improvement != bestImprovement) ? apMostImprovedClasses : ppMostImprovedClasses;
               *ppMostImprovedClasses = pTargetClass;
               ++ppMostImprovedClasses;
               bestImprovement = improvement;
            }
         }
         ++pTargetClass;
      } while(pTargetClassesEnd != pTargetClass);
      EBM_ASSERT(-std::numeric_limits<double>::infinity() != bestImprovement);

      // If more than one class has the same max improvement, randomly select between the classes
      // to give the leftover to.
      const size_t cMostImproved = ppMostImprovedClasses - apMostImprovedClasses;
      EBM_ASSERT(1 <= cMostImproved);
      const size_t iRandom = cpuRng.NextFast(cMostImproved);
      TargetClass * const pMostImprovedClasses = apMostImprovedClasses[iRandom];
      ++pMostImprovedClasses->m_cTrainingSamples;

      --cLeftoverTrainingSamples;
   }

#ifndef NDEBUG
   size_t cTrainingSamplesDebug = 0;
   size_t cSamplesDebug = 0;
   for(size_t iClassDebug = 0; iClassDebug < cClasses; ++iClassDebug) {
      cTrainingSamplesDebug += aTargetClasses[iClassDebug].m_cTrainingSamples;
      cSamplesDebug += aTargetClasses[iClassDebug].m_cSamples;
   }
   EBM_ASSERT(cTrainingSamplesDebug == cTrainingSamples);
   EBM_ASSERT(cSamplesDebug == cSamples);
#endif

   const IntEbm * pTarget = targets;
   BagEbm * pSampleReplicationOut = bagOut;
   do {
      const IntEbm indexClass = *pTarget;
      EBM_ASSERT(0 <= indexClass);
      EBM_ASSERT(indexClass < countClasses);

      TargetClass * const pTargetClass = &aTargetClasses[static_cast<size_t>(indexClass)];
      EBM_ASSERT(1 <= pTargetClass->m_cSamples);
      const size_t iRandom = cpuRng.NextFast(pTargetClass->m_cSamples);
      const bool bTrainingSample = UNPREDICTABLE(iRandom < pTargetClass->m_cTrainingSamples);

      *pSampleReplicationOut = UNPREDICTABLE(bTrainingSample) ? BagEbm { 1 } : BagEbm { -1 };
      const size_t cSubtract = UNPREDICTABLE(bTrainingSample) ? size_t { 1 } : size_t { 0 };
      pTargetClass->m_cTrainingSamples -= cSubtract;
      --pTargetClass->m_cSamples;

      ++pSampleReplicationOut;
      ++pTarget;
   } while(pTargetsEnd != pTarget);

   if(nullptr != rng) {
      RandomDeterministic * pRng = reinterpret_cast<RandomDeterministic *>(rng);
      pRng->Initialize(cpuRng); // move the RNG from memory into CPU registers
   }

#ifndef NDEBUG
   for(size_t iClassDebug = 0; iClassDebug < cClasses; ++iClassDebug) {
      EBM_ASSERT(0 == aTargetClasses[iClassDebug].m_cTrainingSamples);
      EBM_ASSERT(0 == aTargetClasses[iClassDebug].m_cSamples);
   }
#endif

   free(aTargetClasses);
   free(apMostImprovedClasses);

   LOG_COUNTED_0(
      &g_cLogExitSampleWithoutReplacementStratified,
      Trace_Info,
      Trace_Verbose,
      "Exited SampleWithoutReplacementStratified"
   );

   return Error_None;
}

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
               if(IsAbsCastError<size_t>(replication)) {
                  LOG_0(Trace_Error, "ERROR Unbag IsAbsCastError<size_t>(replication)");
                  return Error_IllegalParamVal;
               }
               const size_t replicationUnsigned = AbsCast<size_t>(replication);
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
   const BagEbm * const aBag,
   const FloatFast * pWeights,
   const size_t cSetSamples
) {
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(1 <= cSetSamples);
   EBM_ASSERT(nullptr != pWeights);

   FloatFast firstWeight = std::numeric_limits<FloatFast>::quiet_NaN();
   const bool isLoopTraining = BagEbm { 0 } < direction;
   ptrdiff_t cSetSamplesRemaining = static_cast<ptrdiff_t>(cSetSamples);
   if(!isLoopTraining) {
      // make cSetSamplesRemaining the same sign as the bags we want to match
      cSetSamplesRemaining = -cSetSamplesRemaining;
   }
   const BagEbm * pSampleReplication = aBag;
   EBM_ASSERT(nullptr != aBag || isLoopTraining); // if pSampleReplication is nullptr then we have no validation samples
   do {
      BagEbm replication = 1;
      if(nullptr != pSampleReplication) {
         bool isItemTraining;
         do {
            do {
               replication = *pSampleReplication;
               ++pSampleReplication;
               ++pWeights;
            } while(BagEbm { 0 } == replication);
            isItemTraining = BagEbm { 0 } < replication;
         } while(isLoopTraining != isItemTraining);
         --pWeights;
      }

      const FloatFast weight = *pWeights;
      ++pWeights;

      // this relies on the property that NaN is not equal to everything, including NaN
      if(UNLIKELY(firstWeight != weight)) {
         // we need the check of weight as NaN to handle the case [NaN, 9]
         if(!std::isnan(firstWeight) || std::isnan(weight)) {
            // if there are any NaN values exit and do not replace with weights of 1 even if all values are NaN
            return false;
         }
         firstWeight = weight;
      }
      cSetSamplesRemaining -= static_cast<ptrdiff_t>(replication);
   } while(LIKELY(0 != cSetSamplesRemaining));
   return true;
}

extern ErrorEbm ExtractWeights(
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const BagEbm * const aBag,
   const size_t cSetSamples,
   FloatFast ** ppWeightsOut
) {
   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(1 <= cSetSamples);
   EBM_ASSERT(nullptr != ppWeightsOut);
   EBM_ASSERT(nullptr == *ppWeightsOut);

   const FloatFast * const aWeights = GetDataSetSharedWeight(pDataSetShared, 0);
   EBM_ASSERT(nullptr != aWeights);
   if(!CheckWeightsEqual(direction, aBag, aWeights, cSetSamples)) {
      if(IsMultiplyError(sizeof(FloatFast), cSetSamples)) {
         LOG_0(Trace_Warning, "WARNING ExtractWeights IsMultiplyError(sizeof(FloatFast), cSetSamples)");
         return Error_OutOfMemory;
      }
      FloatFast * const aRet = static_cast<FloatFast *>(malloc(sizeof(FloatFast) * cSetSamples));
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
      EBM_ASSERT(nullptr != aBag || isLoopTraining); // if aBag is nullptr then we have no validation samples
      do {
         BagEbm replication = 1;
         if(nullptr != pSampleReplication) {
            bool isItemTraining;
            do {
               do {
                  replication = *pSampleReplication;
                  ++pSampleReplication;
                  ++pWeightFrom;
               } while(BagEbm { 0 } == replication);
               isItemTraining = BagEbm { 0 } < replication;
            } while(isLoopTraining != isItemTraining);
            --pWeightFrom;
         }

         const FloatFast weight = *pWeightFrom;
         ++pWeightFrom;

         // if weight is NaN or +-inf then we'll find that out when we sum the weights
         do {
            EBM_ASSERT(pWeightTo < pWeightToEnd);
            *pWeightTo = weight;
            ++pWeightTo;

            replication -= direction;
         } while(BagEbm { 0 } != replication);
      } while(pWeightToEnd != pWeightTo);
   }
   return Error_None;
}

} // DEFINED_ZONE_NAME
