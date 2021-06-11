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

#include "BoosterCore.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern bool InitializeGradientsAndHessians(
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

SegmentedTensor ** BoosterCore::InitializeSegmentedTensors(
   const size_t cFeatureGroups, 
   const FeatureGroup * const * const apFeatureGroups, 
   const size_t cVectorLength) 
{
   LOG_0(TraceLevelInfo, "Entered InitializeSegmentedTensors");

   EBM_ASSERT(0 < cFeatureGroups);
   EBM_ASSERT(nullptr != apFeatureGroups);
   EBM_ASSERT(1 <= cVectorLength);

   SegmentedTensor ** const apSegmentedTensors = EbmMalloc<SegmentedTensor *>(cFeatureGroups);
   if(UNLIKELY(nullptr == apSegmentedTensors)) {
      LOG_0(TraceLevelWarning, "WARNING InitializeSegmentedTensors nullptr == apSegmentedTensors");
      return nullptr;
   }
   for(size_t i = 0; i < cFeatureGroups; ++i) {
      apSegmentedTensors[i] = nullptr;
   }

   SegmentedTensor ** ppSegmentedTensors = apSegmentedTensors;
   for(size_t iFeatureGroup = 0; iFeatureGroup < cFeatureGroups; ++iFeatureGroup) {
      const FeatureGroup * const pFeatureGroup = apFeatureGroups[iFeatureGroup];
      SegmentedTensor * const pSegmentedTensors = 
         SegmentedTensor::Allocate(pFeatureGroup->GetCountSignificantDimensions(), cVectorLength);
      if(UNLIKELY(nullptr == pSegmentedTensors)) {
         LOG_0(TraceLevelWarning, "WARNING InitializeSegmentedTensors nullptr == pSegmentedTensors");
         DeleteSegmentedTensors(cFeatureGroups, apSegmentedTensors);
         return nullptr;
      }

      if(pSegmentedTensors->Expand(pFeatureGroup)) {
         LOG_0(TraceLevelWarning, "WARNING InitializeSegmentedTensors pSegmentedTensors->Expand(pFeatureGroup)");
         DeleteSegmentedTensors(cFeatureGroups, apSegmentedTensors);
         return nullptr;
      }

      *ppSegmentedTensors = pSegmentedTensors;
      ++ppSegmentedTensors;
   }

   LOG_0(TraceLevelInfo, "Exited InitializeSegmentedTensors");
   return apSegmentedTensors;
}

void BoosterCore::Free(BoosterCore * const pBoosterCore) {
   LOG_0(TraceLevelInfo, "Entered BoosterCore::Free");
   if(nullptr != pBoosterCore) {
      pBoosterCore->m_trainingSet.Destruct();
      pBoosterCore->m_validationSet.Destruct();

      SamplingSet::FreeSamplingSets(pBoosterCore->m_cSamplingSets, pBoosterCore->m_apSamplingSets);
      free(pBoosterCore->m_aValidationWeights);

      FeatureGroup::FreeFeatureGroups(pBoosterCore->m_cFeatureGroups, pBoosterCore->m_apFeatureGroups);

      free(pBoosterCore->m_aFeatures);

      DeleteSegmentedTensors(pBoosterCore->m_cFeatureGroups, pBoosterCore->m_apCurrentModel);
      DeleteSegmentedTensors(pBoosterCore->m_cFeatureGroups, pBoosterCore->m_apBestModel);

      free(pBoosterCore);
   }
   LOG_0(TraceLevelInfo, "Exited BoosterCore::Free");
}

static int g_TODO_removeThisThreadTest = 0;
void TODO_removeThisThreadTest() {
   g_TODO_removeThisThreadTest = 1;
}

BoosterCore * BoosterCore::Create(
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

   try {
      // TODO: eliminate this code I added to test that threads are available on the majority of our systems
      std::thread testThread(TODO_removeThisThreadTest);
      testThread.join();
      if(0 == g_TODO_removeThisThreadTest) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create thread not started");
         return nullptr;
      }
   } catch(...) {
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create thread start failed");
      return nullptr;
   }

   LOG_0(TraceLevelInfo, "Entered BoosterCore::Create thread started");

   BoosterCore * const pBoosterCore = EbmMalloc<BoosterCore>();
   if(UNLIKELY(nullptr == pBoosterCore)) {
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == pBoosterCore");
      return nullptr;
   }
   pBoosterCore->InitializeZero();

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelInfo, "BoosterCore::Create starting feature processing");
   if(0 != cFeatures) {
      pBoosterCore->m_aFeatures = EbmMalloc<Feature>(cFeatures);
      if(nullptr == pBoosterCore->m_aFeatures) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == pBoosterCore->m_aFeatures");
         BoosterCore::Free(pBoosterCore);
         return nullptr;
      }
      pBoosterCore->m_cFeatures = cFeatures;

      const BoolEbmType * pFeatureCategorical = aFeaturesCategorical;
      const IntEbmType * pFeatureBinCount = aFeaturesBinCount;
      size_t iFeatureInitialize = size_t { 0 };
      do {
         const IntEbmType countBins = *pFeatureBinCount;
         if(countBins < 0) {
            LOG_0(TraceLevelError, "ERROR BoosterCore::Create countBins cannot be negative");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
         if(0 == countBins && (0 != cTrainingSamples || 0 != cValidationSamples)) {
            LOG_0(TraceLevelError, "ERROR BoosterCore::Create countBins cannot be zero if either 0 < cTrainingSamples OR 0 < cValidationSamples");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
         if(!IsNumberConvertable<size_t>(countBins)) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create countBins is too high for us to allocate enough memory");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
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
         BoosterCore::Free(pBoosterCore);
         return nullptr;
      }

      if(GetTreeSweepSizeOverflow(bClassification, cVectorLength)) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create GetTreeSweepSizeOverflow(bClassification, cVectorLength)");
         BoosterCore::Free(pBoosterCore);
         return nullptr;
      }
      const size_t cBytesPerTreeSweep = GetTreeSweepSize(bClassification, cVectorLength);

      const IntEbmType * pFeatureGroupFeatureIndexes = aFeatureGroupsFeatureIndexes;
      size_t iFeatureGroup = 0;
      do {
         const IntEbmType countDimensions = aFeatureGroupsDimensionCount[iFeatureGroup];
         if(countDimensions < 0) {
            LOG_0(TraceLevelError, "ERROR BoosterCore::Create countDimensions cannot be negative");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
         if(!IsNumberConvertable<size_t>(countDimensions)) {
            // if countDimensions exceeds the size of size_t, then we wouldn't be able to find it
            // in the array passed to us
            LOG_0(TraceLevelError, "ERROR BoosterCore::Create countDimensions is too high to index");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
         const size_t cDimensions = static_cast<size_t>(countDimensions);
         FeatureGroup * const pFeatureGroup = FeatureGroup::Allocate(cDimensions, iFeatureGroup);
         if(nullptr == pFeatureGroup) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == pFeatureGroup");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
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
               BoosterCore::Free(pBoosterCore);
               return nullptr;
            }
            size_t cEquivalentSplits = 1;
            size_t cTensorBins = 1;
            FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
            FeatureGroupEntry * pFeatureGroupEntryEnd = pFeatureGroupEntry + cDimensions;
            do {
               const IntEbmType indexFeatureInterop = *pFeatureGroupFeatureIndexes;
               if(indexFeatureInterop < 0) {
                  LOG_0(TraceLevelError, "ERROR BoosterCore::Create aFeatureGroupsFeatureIndexes value cannot be negative");
                  BoosterCore::Free(pBoosterCore);
                  return nullptr;
               }
               if(!IsNumberConvertable<size_t>(indexFeatureInterop)) {
                  LOG_0(TraceLevelError, "ERROR BoosterCore::Create aFeatureGroupsFeatureIndexes value too big to reference memory");
                  BoosterCore::Free(pBoosterCore);
                  return nullptr;
               }
               const size_t iFeature = static_cast<size_t>(indexFeatureInterop);

               if(cFeatures <= iFeature) {
                  LOG_0(TraceLevelError, "ERROR BoosterCore::Create aFeatureGroupsFeatureIndexes value must be less than the number of features");
                  BoosterCore::Free(pBoosterCore);
                  return nullptr;
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
                     BoosterCore::Free(pBoosterCore);
                     return nullptr;
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
                  BoosterCore::Free(pBoosterCore);
                  return nullptr;
               }

               size_t cBytesArrayEquivalentSplit;
               if(1 == cSignificantDimensions) {
                  if(IsMultiplyError(cEquivalentSplits, cBytesPerTreeSweep)) {
                     LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create IsMultiplyError(cEquivalentSplits, cBytesPerTreeSweep)");
                     BoosterCore::Free(pBoosterCore);
                     return nullptr;
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
         pBoosterCore->m_apCurrentModel = InitializeSegmentedTensors(cFeatureGroups, pBoosterCore->m_apFeatureGroups, cVectorLength);
         if(nullptr == pBoosterCore->m_apCurrentModel) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == m_apCurrentModel");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
         pBoosterCore->m_apBestModel = InitializeSegmentedTensors(cFeatureGroups, pBoosterCore->m_apFeatureGroups, cVectorLength);
         if(nullptr == pBoosterCore->m_apBestModel) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == m_apBestModel");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
      }
   }
   LOG_0(TraceLevelInfo, "BoosterCore::Create finished feature group processing");

   pBoosterCore->m_cBytesArrayEquivalentSplitMax = cBytesArrayEquivalentSplitMax;

   if(pBoosterCore->m_trainingSet.Initialize(
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
   )) {
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create m_trainingSet.Initialize");
      BoosterCore::Free(pBoosterCore);
      return nullptr;
   }

   if(pBoosterCore->m_validationSet.Initialize(
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
   )) {
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create m_validationSet.Initialize");
      BoosterCore::Free(pBoosterCore);
      return nullptr;
   }

   pBoosterCore->m_randomStream.InitializeUnsigned(randomSeed, k_boosterRandomizationMix);

   EBM_ASSERT(nullptr == pBoosterCore->m_apSamplingSets);
   if(0 != cTrainingSamples) {
      pBoosterCore->m_cSamplingSets = cSamplingSets;
      pBoosterCore->m_apSamplingSets = SamplingSet::GenerateSamplingSets(&pBoosterCore->m_randomStream, &pBoosterCore->m_trainingSet, aTrainingWeights, cSamplingSets);
      if(UNLIKELY(nullptr == pBoosterCore->m_apSamplingSets)) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == m_apSamplingSets");
         BoosterCore::Free(pBoosterCore);
         return nullptr;
      }
   }

   EBM_ASSERT(nullptr == pBoosterCore->m_aValidationWeights);
   pBoosterCore->m_validationWeightTotal = static_cast<FloatEbmType>(cValidationSamples);
   if(0 != cValidationSamples && nullptr != aValidationWeights) {
      if(IsMultiplyError(sizeof(*aValidationWeights), cValidationSamples)) {
         LOG_0(TraceLevelWarning,
            "WARNING BoosterCore::Create IsMultiplyError(sizeof(*aValidationWeights), cValidationSamples)");
         BoosterCore::Free(pBoosterCore);
         return nullptr;
      }
      if(!CheckAllWeightsEqual(cValidationSamples, aValidationWeights)) {
         const FloatEbmType total = AddPositiveFloatsSafe(cValidationSamples, aValidationWeights);
         if(std::isnan(total) || std::isinf(total) || total <= FloatEbmType { 0 }) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create std::isnan(total) || std::isinf(total) || total <= FloatEbmType { 0 }");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
         // if they were all zero then we'd ignore the weights param.  If there are negative numbers it might add
         // to zero though so check it after checking for negative
         EBM_ASSERT(FloatEbmType { 0 } != total);
         pBoosterCore->m_validationWeightTotal = total;

         const size_t cBytes = sizeof(*aValidationWeights) * cValidationSamples;
         FloatEbmType * pValidationWeightInternal = static_cast<FloatEbmType *>(malloc(cBytes));
         if(UNLIKELY(nullptr == pValidationWeightInternal)) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == pValidationWeightInternal");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
         pBoosterCore->m_aValidationWeights = pValidationWeightInternal;
         memcpy(pValidationWeightInternal, aValidationWeights, cBytes);
      }
   }

   if(bClassification) {
      if(0 != cTrainingSamples) {
         if(InitializeGradientsAndHessians(
            runtimeLearningTypeOrCountTargetClasses,
            cTrainingSamples,
            aTrainingTargets,
            aTrainingPredictorScores,
            pBoosterCore->m_trainingSet.GetGradientsAndHessiansPointer()
         )) {
            // error already logged
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      if(0 != cTrainingSamples) {
#ifndef NDEBUG
         const bool isFailed = 
#endif // NDEBUG
         InitializeGradientsAndHessians(
            k_regression,
            cTrainingSamples,
            aTrainingTargets,
            aTrainingPredictorScores,
            pBoosterCore->m_trainingSet.GetGradientsAndHessiansPointer()
         );
         EBM_ASSERT(!isFailed);
      }
      if(0 != cValidationSamples) {
#ifndef NDEBUG
         const bool isFailed =
#endif // NDEBUG
         InitializeGradientsAndHessians(
            k_regression,
            cValidationSamples,
            aValidationTargets,
            aValidationPredictorScores,
            pBoosterCore->m_validationSet.GetGradientsAndHessiansPointer()
         );
         EBM_ASSERT(!isFailed);
      }
   }

   pBoosterCore->m_runtimeLearningTypeOrCountTargetClasses = runtimeLearningTypeOrCountTargetClasses;
   pBoosterCore->m_bestModelMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::max() };

   LOG_0(TraceLevelInfo, "Exited BoosterCore::Create");
   return pBoosterCore;
}

} // DEFINED_ZONE_NAME
