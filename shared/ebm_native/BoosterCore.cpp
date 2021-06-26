// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <thread>

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "common_cpp.hpp"
#include "ebm_internal.hpp"

#include "compute_accessors.hpp"

#include "RandomStream.hpp"
#include "SegmentedTensor.hpp"
#include "ebm_stats.hpp"
// feature includes
#include "Feature.hpp"
// FeatureGroup.h depends on FeatureInternal.h
#include "FeatureGroup.hpp"
// dataset depends on features
#include "DataSetBoosting.hpp"
// samples is somewhat independent from datasets, but relies on an indirect coupling with them
#include "SamplingSet.hpp"
#include "TreeSweep.hpp"

#include "BoosterShell.hpp"
#include "BoosterCore.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern ErrorEbmType InitializeGradientsAndHessians(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const size_t cSamples,
   const void * const aTargetData,
   const FloatEbmType * const aPredictorScores,
   FloatEbmType * pGradient
);

INLINE_ALWAYS static size_t GetCountItemsBitPacked(const size_t cBits) {
   EBM_ASSERT(size_t { 1 } <= cBits);
   return k_cBitsForStorageType / cBits;
}

void BoosterCore::DeleteSegmentedTensors(const size_t cFeatureGroups, SegmentedTensor ** const apSegmentedTensors) {
   LOG_0(TraceLevelInfo, "Entered DeleteSegmentedTensors");

   if(UNLIKELY(nullptr != apSegmentedTensors)) {
      EBM_ASSERT(0 < cFeatureGroups);
      SegmentedTensor ** ppSegmentedTensors = apSegmentedTensors;
      const SegmentedTensor * const * const ppSegmentedTensorsEnd = &apSegmentedTensors[cFeatureGroups];
      do {
         SegmentedTensor::Free(*ppSegmentedTensors);
         ++ppSegmentedTensors;
      } while(ppSegmentedTensorsEnd != ppSegmentedTensors);
      free(apSegmentedTensors);
   }
   LOG_0(TraceLevelInfo, "Exited DeleteSegmentedTensors");
}

ErrorEbmType BoosterCore::InitializeSegmentedTensors(
   const size_t cFeatureGroups, 
   const FeatureGroup * const * const apFeatureGroups, 
   const size_t cVectorLength,
   SegmentedTensor *** papSegmentedTensorsOut)
{
   LOG_0(TraceLevelInfo, "Entered InitializeSegmentedTensors");

   EBM_ASSERT(0 < cFeatureGroups);
   EBM_ASSERT(nullptr != apFeatureGroups);
   EBM_ASSERT(1 <= cVectorLength);
   EBM_ASSERT(nullptr != papSegmentedTensorsOut);
   EBM_ASSERT(nullptr == *papSegmentedTensorsOut);

   SegmentedTensor ** const apSegmentedTensors = EbmMalloc<SegmentedTensor *>(cFeatureGroups);
   if(UNLIKELY(nullptr == apSegmentedTensors)) {
      LOG_0(TraceLevelWarning, "WARNING InitializeSegmentedTensors nullptr == apSegmentedTensors");
      return Error_OutOfMemory;
   }
   for(size_t i = 0; i < cFeatureGroups; ++i) {
      apSegmentedTensors[i] = nullptr;
   }
   *papSegmentedTensorsOut = apSegmentedTensors; // transfer ownership for future deletion

   SegmentedTensor ** ppSegmentedTensors = apSegmentedTensors;
   for(size_t iFeatureGroup = 0; iFeatureGroup < cFeatureGroups; ++iFeatureGroup) {
      const FeatureGroup * const pFeatureGroup = apFeatureGroups[iFeatureGroup];
      SegmentedTensor * const pSegmentedTensors = 
         SegmentedTensor::Allocate(pFeatureGroup->GetCountSignificantDimensions(), cVectorLength);
      if(UNLIKELY(nullptr == pSegmentedTensors)) {
         LOG_0(TraceLevelWarning, "WARNING InitializeSegmentedTensors nullptr == pSegmentedTensors");
         return Error_OutOfMemory;
      }
      *ppSegmentedTensors = pSegmentedTensors; // transfer ownership for future deletion

      const ErrorEbmType error = pSegmentedTensors->Expand(pFeatureGroup);
      if(Error_None != error) {
         // already logged
         return error;
      }

      ++ppSegmentedTensors;
   }

   LOG_0(TraceLevelInfo, "Exited InitializeSegmentedTensors");
   return Error_None;
}

void BoosterCore::Free(BoosterCore * const pBoosterCore) {
   LOG_0(TraceLevelInfo, "Entered BoosterCore::Free");
   if(nullptr != pBoosterCore) {
      // for reference counting in general, a release is needed during the decrement and aquire is needed if freeing
      // https://www.boost.org/doc/libs/1_59_0/doc/html/atomic/usage_examples.html
      // We need to ensure that writes on this thread are not allowed to be re-ordered to a point below the 
      // decrement because if we happened to decrement to 2, and then get interrupted, and annother thread
      // decremented to 1 after us, we don't want our unclean writes to memory to be visible in the other thread
      // so we use memory_order_release on the decrement.
      if(size_t { 1 } == pBoosterCore->m_REFERENCE_COUNT.fetch_sub(1, std::memory_order_release)) {
         // we need to ensure that reads on this thread do not get reordered to a point before the decrement, otherwise
         // another thread might write some information, write the decrement to 2, then our thread decrements to 1
         // and then if we're allowed to read from data that occured before our decrement to 1 then we could have
         // stale data from before the other thread decrementing.  If our thread isn't freeing the memory though
         // we don't have to worry about staleness, so only use memory_order_acquire if we're going to delete the
         // object
         std::atomic_thread_fence(std::memory_order_acquire);
         LOG_0(TraceLevelInfo, "INFO BoosterCore::Free deleting BoosterCore");
         delete pBoosterCore;
      }
   }
   LOG_0(TraceLevelInfo, "Exited BoosterCore::Free");
}

static int g_TODO_removeThisThreadTest = 0;
void TODO_removeThisThreadTest() {
   g_TODO_removeThisThreadTest = 1;
}

ErrorEbmType BoosterCore::Create(
   BoosterShell * const pBoosterShell,
   const SeedEbmType randomSeed,
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const size_t cFeatures,
   const size_t cFeatureGroups,
   const size_t cSamplingSets,
   const FloatEbmType * const optionalTempParams,
   const BoolEbmType * const aFeaturesCategorical,
   const IntEbmType * const aFeaturesBinCount,
   const IntEbmType * const aFeatureGroupsDimensionCount,
   const IntEbmType * const aFeatureGroupsFeatureIndexes, 
   const size_t cTrainingSamples, 
   const void * const aTrainingTargets, 
   const IntEbmType * const aTrainingBinnedData, 
   const FloatEbmType * const aTrainingWeights,
   const FloatEbmType * const aTrainingPredictorScores,
   const size_t cValidationSamples, 
   const void * const aValidationTargets, 
   const IntEbmType * const aValidationBinnedData, 
   const FloatEbmType * const aValidationWeights,
   const FloatEbmType * const aValidationPredictorScores
) {
   // optionalTempParams isn't used by default.  It's meant to provide an easy way for python or other higher
   // level languages to pass EXPERIMENTAL temporary parameters easily to the C++ code.
   UNUSED(optionalTempParams);

   LOG_0(TraceLevelInfo, "Entered BoosterCore::Create");

   //try {
   //   // TODO: eliminate this code I added to test that threads are available on the majority of our systems
   //   std::thread testThread(TODO_removeThisThreadTest);
   //   testThread.join();
   //   if(0 == g_TODO_removeThisThreadTest) {
   //      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create thread not started");
   //      return Error_UnexpectedInternal;
   //   }
   //} catch(const std::bad_alloc &) {
   //   LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create thread start out of memory");
   //   return Error_OutOfMemory;
   //} catch(...) {
   //   // the C++ standard doesn't really seem to say what kind of exceptions we'd get for various errors, so
   //   // about the best we can do is catch(...) since the exact exceptions seem to be implementation specific
   //   LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create thread start failed");
   //   return Error_ThreadStartFailed;
   //}
   //LOG_0(TraceLevelInfo, "INFO BoosterCore::Create thread started");

   BoosterCore * pBoosterCore;
   try {
      pBoosterCore = new BoosterCore();
   } catch(const std::bad_alloc &) {
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create Out of memory allocating BoosterCore");
      return Error_OutOfMemory;
   } catch(...) {
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create Unknown error");
      return Error_UnexpectedInternal;
   }
   if(nullptr == pBoosterCore) {
      // this should be impossible since bad_alloc should have been thrown, but let's be untrusting
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == pBoosterCore");
      return Error_OutOfMemory;
   }
   // give ownership of our object to pBoosterShell
   pBoosterShell->SetBoosterCore(pBoosterCore);

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelInfo, "BoosterCore::Create starting feature processing");
   if(0 != cFeatures) {
      pBoosterCore->m_cFeatures = cFeatures;
      pBoosterCore->m_aFeatures = EbmMalloc<Feature>(cFeatures);
      if(nullptr == pBoosterCore->m_aFeatures) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == pBoosterCore->m_aFeatures");
         return Error_OutOfMemory;
      }

      const BoolEbmType * pFeatureCategorical = aFeaturesCategorical;
      const IntEbmType * pFeatureBinCount = aFeaturesBinCount;
      size_t iFeatureInitialize = size_t { 0 };
      do {
         const IntEbmType countBins = *pFeatureBinCount;
         if(countBins < 0) {
            LOG_0(TraceLevelError, "ERROR BoosterCore::Create countBins cannot be negative");
            return Error_IllegalParamValue;
         }
         if(0 == countBins && (0 != cTrainingSamples || 0 != cValidationSamples)) {
            LOG_0(TraceLevelError, "ERROR BoosterCore::Create countBins cannot be zero if either 0 < cTrainingSamples OR 0 < cValidationSamples");
            return Error_IllegalParamValue;
         }
         if(!IsNumberConvertable<size_t>(countBins)) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create countBins is too high for us to allocate enough memory");
            return Error_IllegalParamValue;
         }
         const size_t cBins = static_cast<size_t>(countBins);
         if(0 == cBins) {
            // we can handle 0 == cBins even though that's a degenerate case that shouldn't be boosted on.  0 bins
            // can only occur if there were zero training and zero validation cases since the 
            // features would require a value, even if it was 0.
            LOG_0(TraceLevelInfo, "INFO BoosterCore::Create feature with 0 values");
         } else if(1 == cBins) {
            // we can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on. 
            // Dimensions with 1 bin don't contribute anything since they always have the same value.
            LOG_0(TraceLevelInfo, "INFO BoosterCore::Create feature with 1 value");
         }
         const BoolEbmType isCategorical = *pFeatureCategorical;
         if(EBM_FALSE != isCategorical && EBM_TRUE != isCategorical) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create featuresCategorical should either be EBM_TRUE or EBM_FALSE");
         }
         const bool bCategorical = EBM_FALSE != isCategorical;

         pBoosterCore->m_aFeatures[iFeatureInitialize].Initialize(cBins, iFeatureInitialize, bCategorical);

         ++pFeatureCategorical;
         ++pFeatureBinCount;

         ++iFeatureInitialize;
      } while(cFeatures != iFeatureInitialize);
   }
   LOG_0(TraceLevelInfo, "BoosterCore::Create done feature processing");

   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);
   size_t cBytesArrayEquivalentSplitMax = 0;

   EBM_ASSERT(nullptr == pBoosterCore->m_apCurrentModel);
   EBM_ASSERT(nullptr == pBoosterCore->m_apBestModel);

   LOG_0(TraceLevelInfo, "BoosterCore::Create starting feature group processing");
   if(0 != cFeatureGroups) {
      pBoosterCore->m_cFeatureGroups = cFeatureGroups;
      pBoosterCore->m_apFeatureGroups = FeatureGroup::AllocateFeatureGroups(cFeatureGroups);
      if(UNLIKELY(nullptr == pBoosterCore->m_apFeatureGroups)) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create 0 != m_cFeatureGroups && nullptr == m_apFeatureGroups");
         return Error_OutOfMemory;
      }

      if(GetTreeSweepSizeOverflow(bClassification, cVectorLength)) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create GetTreeSweepSizeOverflow(bClassification, cVectorLength)");
         return Error_OutOfMemory;
      }
      const size_t cBytesPerTreeSweep = GetTreeSweepSize(bClassification, cVectorLength);

      const IntEbmType * pFeatureGroupFeatureIndexes = aFeatureGroupsFeatureIndexes;
      size_t iFeatureGroup = 0;
      do {
         const IntEbmType countDimensions = aFeatureGroupsDimensionCount[iFeatureGroup];
         if(countDimensions < 0) {
            LOG_0(TraceLevelError, "ERROR BoosterCore::Create countDimensions cannot be negative");
            return Error_IllegalParamValue;
         }
         if(!IsNumberConvertable<size_t>(countDimensions)) {
            // if countDimensions exceeds the size of size_t, then we wouldn't be able to find it
            // in the array passed to us
            LOG_0(TraceLevelError, "ERROR BoosterCore::Create countDimensions is too high to index");
            // you can't really have more than size_t countDimensions since each dimension is a feature
            // in a feature group, and our caller can't really have more than size_t of those
            return Error_IllegalParamValue;
         }
         const size_t cDimensions = static_cast<size_t>(countDimensions);
         FeatureGroup * const pFeatureGroup = FeatureGroup::Allocate(cDimensions, iFeatureGroup);
         if(nullptr == pFeatureGroup) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == pFeatureGroup");
            return Error_OutOfMemory;
         }
         // assign our pointer directly to our array right now so that we can't loose the memory if we decide to exit due to an error below
         pBoosterCore->m_apFeatureGroups[iFeatureGroup] = pFeatureGroup;

         size_t cSignificantDimensions = 0;
         ptrdiff_t cItemsPerBitPack = k_cItemsPerBitPackNone;
         if(UNLIKELY(0 == cDimensions)) {
            LOG_0(TraceLevelInfo, "INFO BoosterCore::Create empty feature group");
         } else {
            if(nullptr == pFeatureGroupFeatureIndexes) {
               LOG_0(TraceLevelError, "ERROR BoosterCore::Create aFeatureGroupsFeatureIndexes is null when there are FeatureGroups with non-zero numbers of features");
               return Error_IllegalParamValue;
            }
            size_t cEquivalentSplits = 1;
            size_t cTensorBins = 1;
            FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
            FeatureGroupEntry * pFeatureGroupEntryEnd = pFeatureGroupEntry + cDimensions;
            do {
               const IntEbmType indexFeatureInterop = *pFeatureGroupFeatureIndexes;
               if(indexFeatureInterop < 0) {
                  LOG_0(TraceLevelError, "ERROR BoosterCore::Create aFeatureGroupsFeatureIndexes value cannot be negative");
                  return Error_IllegalParamValue;
               }
               if(!IsNumberConvertable<size_t>(indexFeatureInterop)) {
                  LOG_0(TraceLevelError, "ERROR BoosterCore::Create aFeatureGroupsFeatureIndexes value too big to reference memory");
                  return Error_IllegalParamValue;
               }
               const size_t iFeature = static_cast<size_t>(indexFeatureInterop);

               if(cFeatures <= iFeature) {
                  LOG_0(TraceLevelError, "ERROR BoosterCore::Create aFeatureGroupsFeatureIndexes value must be less than the number of features");
                  return Error_IllegalParamValue;
               }

               EBM_ASSERT(1 <= cFeatures);
               EBM_ASSERT(nullptr != pBoosterCore->m_aFeatures);

               Feature * const pInputFeature = &pBoosterCore->m_aFeatures[iFeature];
               pFeatureGroupEntry->m_pFeature = pInputFeature;

               const size_t cBins = pInputFeature->GetCountBins();
               if(LIKELY(size_t { 1 } < cBins)) {
                  // if we have only 1 bin, then we can eliminate the feature from consideration since the resulting tensor loses one dimension but is 
                  // otherwise indistinquishable from the original data
                  ++cSignificantDimensions;
                  if(IsMultiplyError(cTensorBins, cBins)) {
                     // if this overflows, we definetly won't be able to allocate it
                     LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create IsMultiplyError(cTensorStates, cBins)");
                     return Error_OutOfMemory;
                  }
                  cTensorBins *= cBins;
                  cEquivalentSplits *= cBins - 1; // we can only split between the bins
               } else {
                  LOG_0(TraceLevelInfo, "INFO BoosterCore::Create feature group with no useful features");
               }

               ++pFeatureGroupFeatureIndexes;
               ++pFeatureGroupEntry;
            } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);

            if(LIKELY(0 != cSignificantDimensions)) {
               EBM_ASSERT(1 < cTensorBins);

               if(k_cDimensionsMax < cSignificantDimensions) {
                  // if we try to run with more than k_cDimensionsMax we'll exceed our memory capacity, so let's exit here instead
                  LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create k_cDimensionsMax < cSignificantDimensions");
                  return Error_OutOfMemory;
               }

               size_t cBytesArrayEquivalentSplit;
               if(1 == cSignificantDimensions) {
                  if(IsMultiplyError(cEquivalentSplits, cBytesPerTreeSweep)) {
                     LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create IsMultiplyError(cEquivalentSplits, cBytesPerTreeSweep)");
                     return Error_OutOfMemory;
                  }
                  cBytesArrayEquivalentSplit = cEquivalentSplits * cBytesPerTreeSweep;
               } else {
                  // TODO : someday add equal gain multidimensional randomized picking.  It's rather hard though with the existing sweep functions for 
                  // multidimensional right now
                  cBytesArrayEquivalentSplit = 0;
               }
               if(cBytesArrayEquivalentSplitMax < cBytesArrayEquivalentSplit) {
                  cBytesArrayEquivalentSplitMax = cBytesArrayEquivalentSplit;
               }

               const size_t cBitsRequiredMin = CountBitsRequired(cTensorBins - 1);
               EBM_ASSERT(1 <= cBitsRequiredMin); // 1 < cTensorBins otherwise we'd have filtered it out above
               cItemsPerBitPack = static_cast<ptrdiff_t>(GetCountItemsBitPacked(cBitsRequiredMin));
            }
         }
         pFeatureGroup->SetCountSignificantFeatures(cSignificantDimensions);
         pFeatureGroup->SetBitPack(cItemsPerBitPack);

         ++iFeatureGroup;
      } while(iFeatureGroup < cFeatureGroups);

      if(!bClassification || ptrdiff_t { 2 } <= runtimeLearningTypeOrCountTargetClasses) {
         ErrorEbmType error = InitializeSegmentedTensors(cFeatureGroups, pBoosterCore->m_apFeatureGroups, cVectorLength, &pBoosterCore->m_apCurrentModel);
         if(Error_None != error) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == m_apCurrentModel");
            return error;
         }
         error = InitializeSegmentedTensors(cFeatureGroups, pBoosterCore->m_apFeatureGroups, cVectorLength, &pBoosterCore->m_apBestModel);
         if(Error_None != error) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == m_apBestModel");
            return error;
         }
      }
   }
   LOG_0(TraceLevelInfo, "BoosterCore::Create finished feature group processing");

   pBoosterCore->m_cBytesArrayEquivalentSplitMax = cBytesArrayEquivalentSplitMax;

   const ErrorEbmType error1 = pBoosterCore->m_trainingSet.Initialize(
      true,
      bClassification,
      bClassification,
      bClassification,
      cFeatureGroups,
      pBoosterCore->m_apFeatureGroups,
      cTrainingSamples,
      aTrainingBinnedData,
      aTrainingTargets,
      aTrainingPredictorScores,
      runtimeLearningTypeOrCountTargetClasses
   );
   if(Error_None != error1) {
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create m_trainingSet.Initialize");
      return error1;
   }

   const ErrorEbmType error2 = pBoosterCore->m_validationSet.Initialize(
      !bClassification,
      false,
      bClassification,
      bClassification,
      cFeatureGroups,
      pBoosterCore->m_apFeatureGroups,
      cValidationSamples,
      aValidationBinnedData,
      aValidationTargets,
      aValidationPredictorScores,
      runtimeLearningTypeOrCountTargetClasses
   );
   if(Error_None != error2) {
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create m_validationSet.Initialize");
      return error2;
   }

   pBoosterCore->m_randomStream.InitializeUnsigned(randomSeed, k_boosterRandomizationMix);

   EBM_ASSERT(nullptr == pBoosterCore->m_apSamplingSets);
   if(0 != cTrainingSamples) {
      pBoosterCore->m_cSamplingSets = cSamplingSets;
      pBoosterCore->m_apSamplingSets = SamplingSet::GenerateSamplingSets(&pBoosterCore->m_randomStream, &pBoosterCore->m_trainingSet, aTrainingWeights, cSamplingSets);
      if(UNLIKELY(nullptr == pBoosterCore->m_apSamplingSets)) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == m_apSamplingSets");
         return Error_OutOfMemory;
      }
   }

   EBM_ASSERT(nullptr == pBoosterCore->m_aValidationWeights);
   pBoosterCore->m_validationWeightTotal = static_cast<FloatEbmType>(cValidationSamples);
   if(0 != cValidationSamples && nullptr != aValidationWeights) {
      if(IsMultiplyError(sizeof(*aValidationWeights), cValidationSamples)) {
         LOG_0(TraceLevelWarning,
            "WARNING BoosterCore::Create IsMultiplyError(sizeof(*aValidationWeights), cValidationSamples)");
         return Error_IllegalParamValue;
      }
      if(!CheckAllWeightsEqual(cValidationSamples, aValidationWeights)) {
         const FloatEbmType total = AddPositiveFloatsSafe(cValidationSamples, aValidationWeights);
         if(std::isnan(total) || std::isinf(total) || total <= FloatEbmType { 0 }) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create std::isnan(total) || std::isinf(total) || total <= FloatEbmType { 0 }");
            return Error_UserParamValue;
         }
         // if they were all zero then we'd ignore the weights param.  If there are negative numbers it might add
         // to zero though so check it after checking for negative
         EBM_ASSERT(FloatEbmType { 0 } != total);
         pBoosterCore->m_validationWeightTotal = total;

         const size_t cBytes = sizeof(*aValidationWeights) * cValidationSamples;
         FloatEbmType * pValidationWeightInternal = static_cast<FloatEbmType *>(malloc(cBytes));
         if(UNLIKELY(nullptr == pValidationWeightInternal)) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == pValidationWeightInternal");
            return Error_OutOfMemory;
         }
         pBoosterCore->m_aValidationWeights = pValidationWeightInternal;
         memcpy(pValidationWeightInternal, aValidationWeights, cBytes);
      }
   }

   if(bClassification) {
      if(0 != cTrainingSamples) {
         const ErrorEbmType error = InitializeGradientsAndHessians(
            runtimeLearningTypeOrCountTargetClasses,
            cTrainingSamples,
            aTrainingTargets,
            aTrainingPredictorScores,
            pBoosterCore->m_trainingSet.GetGradientsAndHessiansPointer()
         );
         if(Error_None != error) {
            // error already logged
            return error;
         }
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      if(0 != cTrainingSamples) {
#ifndef NDEBUG
         const ErrorEbmType error =
#endif // NDEBUG
         InitializeGradientsAndHessians(
            k_regression,
            cTrainingSamples,
            aTrainingTargets,
            aTrainingPredictorScores,
            pBoosterCore->m_trainingSet.GetGradientsAndHessiansPointer()
         );
         EBM_ASSERT(Error_None == error); // InitializeGradientsAndHessians doesn't allocate on regression
      }
      if(0 != cValidationSamples) {
#ifndef NDEBUG
         const ErrorEbmType error =
#endif // NDEBUG
         InitializeGradientsAndHessians(
            k_regression,
            cValidationSamples,
            aValidationTargets,
            aValidationPredictorScores,
            pBoosterCore->m_validationSet.GetGradientsAndHessiansPointer()
         );
         EBM_ASSERT(Error_None == error); // InitializeGradientsAndHessians doesn't allocate on regression
      }
   }

   pBoosterCore->m_runtimeLearningTypeOrCountTargetClasses = runtimeLearningTypeOrCountTargetClasses;
   pBoosterCore->m_bestModelMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::max() };

   LOG_0(TraceLevelInfo, "Exited BoosterCore::Create");
   return Error_None;
}

} // DEFINED_ZONE_NAME
