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

BoosterCore * BoosterCore::Allocate(
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

   LOG_0(TraceLevelInfo, "Entered BoosterCore::Initialize");

   try {
      // TODO: eliminate this code I added to test that threads are available on the majority of our systems
      std::thread testThread(TODO_removeThisThreadTest);
      testThread.join();
      if(0 == g_TODO_removeThisThreadTest) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize thread not started");
         return nullptr;
      }
   } catch(...) {
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize thread start failed");
      return nullptr;
   }

   LOG_0(TraceLevelInfo, "Entered BoosterCore::Initialize thread started");

   BoosterCore * const pBoosterCore = EbmMalloc<BoosterCore>();
   if(UNLIKELY(nullptr == pBoosterCore)) {
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize nullptr == pBoosterCore");
      return nullptr;
   }
   pBoosterCore->InitializeZero();

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelInfo, "BoosterCore::Initialize starting feature processing");
   if(0 != cFeatures) {
      pBoosterCore->m_aFeatures = EbmMalloc<Feature>(cFeatures);
      if(nullptr == pBoosterCore->m_aFeatures) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize nullptr == pBoosterCore->m_aFeatures");
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
            LOG_0(TraceLevelError, "ERROR BoosterCore::Initialize countBins cannot be negative");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
         if(0 == countBins && (0 != cTrainingSamples || 0 != cValidationSamples)) {
            LOG_0(TraceLevelError, "ERROR BoosterCore::Initialize countBins cannot be zero if either 0 < cTrainingSamples OR 0 < cValidationSamples");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
         if(!IsNumberConvertable<size_t>(countBins)) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize countBins is too high for us to allocate enough memory");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
         const size_t cBins = static_cast<size_t>(countBins);
         if(0 == cBins) {
            // we can handle 0 == cBins even though that's a degenerate case that shouldn't be boosted on.  0 bins
            // can only occur if there were zero training and zero validation cases since the 
            // features would require a value, even if it was 0.
            LOG_0(TraceLevelInfo, "INFO BoosterCore::Initialize feature with 0 values");
         } else if(1 == cBins) {
            // we can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on. 
            // Dimensions with 1 bin don't contribute anything since they always have the same value.
            LOG_0(TraceLevelInfo, "INFO BoosterCore::Initialize feature with 1 value");
         }
         const BoolEbmType isCategorical = *pFeatureCategorical;
         if(EBM_FALSE != isCategorical && EBM_TRUE != isCategorical) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize featuresCategorical should either be EBM_TRUE or EBM_FALSE");
         }
         const bool bCategorical = EBM_FALSE != isCategorical;

         pBoosterCore->m_aFeatures[iFeatureInitialize].Initialize(cBins, iFeatureInitialize, bCategorical);

         ++pFeatureCategorical;
         ++pFeatureBinCount;

         ++iFeatureInitialize;
      } while(cFeatures != iFeatureInitialize);
   }
   LOG_0(TraceLevelInfo, "BoosterCore::Initialize done feature processing");

   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);
   size_t cBytesArrayEquivalentSplitMax = 0;

   EBM_ASSERT(nullptr == pBoosterCore->m_apCurrentModel);
   EBM_ASSERT(nullptr == pBoosterCore->m_apBestModel);

   LOG_0(TraceLevelInfo, "BoosterCore::Initialize starting feature group processing");
   if(0 != cFeatureGroups) {
      pBoosterCore->m_cFeatureGroups = cFeatureGroups;
      pBoosterCore->m_apFeatureGroups = FeatureGroup::AllocateFeatureGroups(cFeatureGroups);
      if(UNLIKELY(nullptr == pBoosterCore->m_apFeatureGroups)) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize 0 != m_cFeatureGroups && nullptr == m_apFeatureGroups");
         BoosterCore::Free(pBoosterCore);
         return nullptr;
      }

      if(GetTreeSweepSizeOverflow(bClassification, cVectorLength)) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize GetTreeSweepSizeOverflow(bClassification, cVectorLength)");
         BoosterCore::Free(pBoosterCore);
         return nullptr;
      }
      const size_t cBytesPerTreeSweep = GetTreeSweepSize(bClassification, cVectorLength);

      const IntEbmType * pFeatureGroupFeatureIndexes = aFeatureGroupsFeatureIndexes;
      size_t iFeatureGroup = 0;
      do {
         const IntEbmType countDimensions = aFeatureGroupsDimensionCount[iFeatureGroup];
         if(countDimensions < 0) {
            LOG_0(TraceLevelError, "ERROR BoosterCore::Initialize countDimensions cannot be negative");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
         if(!IsNumberConvertable<size_t>(countDimensions)) {
            // if countDimensions exceeds the size of size_t, then we wouldn't be able to find it
            // in the array passed to us
            LOG_0(TraceLevelError, "ERROR BoosterCore::Initialize countDimensions is too high to index");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
         const size_t cDimensions = static_cast<size_t>(countDimensions);
         FeatureGroup * const pFeatureGroup = FeatureGroup::Allocate(cDimensions, iFeatureGroup);
         if(nullptr == pFeatureGroup) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize nullptr == pFeatureGroup");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
         // assign our pointer directly to our array right now so that we can't loose the memory if we decide to exit due to an error below
         pBoosterCore->m_apFeatureGroups[iFeatureGroup] = pFeatureGroup;

         size_t cSignificantDimensions = 0;
         ptrdiff_t cItemsPerBitPack = k_cItemsPerBitPackNone;
         if(UNLIKELY(0 == cDimensions)) {
            LOG_0(TraceLevelInfo, "INFO BoosterCore::Initialize empty feature group");
         } else {
            if(nullptr == pFeatureGroupFeatureIndexes) {
               LOG_0(TraceLevelError, "ERROR BoosterCore::Initialize aFeatureGroupsFeatureIndexes is null when there are FeatureGroups with non-zero numbers of features");
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
                  LOG_0(TraceLevelError, "ERROR BoosterCore::Initialize aFeatureGroupsFeatureIndexes value cannot be negative");
                  BoosterCore::Free(pBoosterCore);
                  return nullptr;
               }
               if(!IsNumberConvertable<size_t>(indexFeatureInterop)) {
                  LOG_0(TraceLevelError, "ERROR BoosterCore::Initialize aFeatureGroupsFeatureIndexes value too big to reference memory");
                  BoosterCore::Free(pBoosterCore);
                  return nullptr;
               }
               const size_t iFeature = static_cast<size_t>(indexFeatureInterop);

               if(cFeatures <= iFeature) {
                  LOG_0(TraceLevelError, "ERROR BoosterCore::Initialize aFeatureGroupsFeatureIndexes value must be less than the number of features");
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
                     LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize IsMultiplyError(cTensorStates, cBins)");
                     BoosterCore::Free(pBoosterCore);
                     return nullptr;
                  }
                  cTensorBins *= cBins;
                  cEquivalentSplits *= cBins - 1; // we can only split between the bins
               } else {
                  LOG_0(TraceLevelInfo, "INFO BoosterCore::Initialize feature group with no useful features");
               }

               ++pFeatureGroupFeatureIndexes;
               ++pFeatureGroupEntry;
            } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);

            if(LIKELY(0 != cSignificantDimensions)) {
               EBM_ASSERT(1 < cTensorBins);

               if(k_cDimensionsMax < cSignificantDimensions) {
                  // if we try to run with more than k_cDimensionsMax we'll exceed our memory capacity, so let's exit here instead
                  LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize k_cDimensionsMax < cSignificantDimensions");
                  BoosterCore::Free(pBoosterCore);
                  return nullptr;
               }

               size_t cBytesArrayEquivalentSplit;
               if(1 == cSignificantDimensions) {
                  if(IsMultiplyError(cEquivalentSplits, cBytesPerTreeSweep)) {
                     LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize IsMultiplyError(cEquivalentSplits, cBytesPerTreeSweep)");
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
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize nullptr == m_apCurrentModel");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
         pBoosterCore->m_apBestModel = InitializeSegmentedTensors(cFeatureGroups, pBoosterCore->m_apFeatureGroups, cVectorLength);
         if(nullptr == pBoosterCore->m_apBestModel) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize nullptr == m_apBestModel");
            BoosterCore::Free(pBoosterCore);
            return nullptr;
         }
      }
   }
   LOG_0(TraceLevelInfo, "BoosterCore::Initialize finished feature group processing");

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
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize m_trainingSet.Initialize");
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
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize m_validationSet.Initialize");
      BoosterCore::Free(pBoosterCore);
      return nullptr;
   }

   pBoosterCore->m_randomStream.InitializeUnsigned(randomSeed, k_boosterRandomizationMix);

   EBM_ASSERT(nullptr == pBoosterCore->m_apSamplingSets);
   if(0 != cTrainingSamples) {
      pBoosterCore->m_cSamplingSets = cSamplingSets;
      pBoosterCore->m_apSamplingSets = SamplingSet::GenerateSamplingSets(&pBoosterCore->m_randomStream, &pBoosterCore->m_trainingSet, aTrainingWeights, cSamplingSets);
      if(UNLIKELY(nullptr == pBoosterCore->m_apSamplingSets)) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize nullptr == m_apSamplingSets");
         BoosterCore::Free(pBoosterCore);
         return nullptr;
      }
   }

   EBM_ASSERT(nullptr == pBoosterCore->m_aValidationWeights);
   pBoosterCore->m_validationWeightTotal = static_cast<FloatEbmType>(cValidationSamples);
   if(0 != cValidationSamples && nullptr != aValidationWeights) {
      if(IsMultiplyError(sizeof(*aValidationWeights), cValidationSamples)) {
         LOG_0(TraceLevelWarning,
            "WARNING BoosterCore::Initialize IsMultiplyError(sizeof(*aValidationWeights), cValidationSamples)");
         BoosterCore::Free(pBoosterCore);
         return nullptr;
      }
      if(!CheckAllWeightsEqual(cValidationSamples, aValidationWeights)) {
         const FloatEbmType total = AddPositiveFloatsSafe(cValidationSamples, aValidationWeights);
         if(std::isnan(total) || std::isinf(total) || total <= FloatEbmType { 0 }) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize std::isnan(total) || std::isinf(total) || total <= FloatEbmType { 0 }");
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
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Initialize nullptr == pValidationWeightInternal");
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

   LOG_0(TraceLevelInfo, "Exited BoosterCore::Initialize");
   return pBoosterCore;
}

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
static BoosterCore * AllocateBoosting(
   const SeedEbmType randomSeed,
   const IntEbmType countFeatures, 
   const BoolEbmType * const aFeaturesCategorical,
   const IntEbmType * const aFeaturesBinCount,
   const IntEbmType countFeatureGroups,
   const IntEbmType * const aFeatureGroupsDimensionCount,
   const IntEbmType * const aFeatureGroupsFeatureIndexes, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, 
   const IntEbmType countTrainingSamples, 
   const void * const trainingTargets, 
   const IntEbmType * const trainingBinnedData, 
   const FloatEbmType * const aTrainingWeights,
   const FloatEbmType * const trainingPredictorScores, 
   const IntEbmType countValidationSamples, 
   const void * const validationTargets, 
   const IntEbmType * const validationBinnedData, 
   const FloatEbmType * const aValidationWeights, 
   const FloatEbmType * const validationPredictorScores,
   const IntEbmType countInnerBags,
   const FloatEbmType * const optionalTempParams
) {
   // TODO : give AllocateBoosting the same calling parameter order as CreateClassificationBooster

   if(countFeatures < 0) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting countFeatures must be positive");
      return nullptr;
   }
   if(0 != countFeatures && nullptr == aFeaturesCategorical) {
      // TODO: in the future maybe accept null aFeaturesCategorical and assume there are no missing values
      LOG_0(TraceLevelError, "ERROR AllocateBoosting aFeaturesCategorical cannot be nullptr if 0 < countFeatures");
      return nullptr;
   }
   if(0 != countFeatures && nullptr == aFeaturesBinCount) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting aFeaturesBinCount cannot be nullptr if 0 < countFeatures");
      return nullptr;
   }
   if(countFeatureGroups < 0) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting countFeatureGroups must be positive");
      return nullptr;
   }
   if(0 != countFeatureGroups && nullptr == aFeatureGroupsDimensionCount) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting aFeatureGroupsDimensionCount cannot be nullptr if 0 < countFeatureGroups");
      return nullptr;
   }
   // aFeatureGroupsFeatureIndexes -> it's legal for aFeatureGroupsFeatureIndexes to be nullptr if there are no features indexed by our featureGroups.  
   // FeatureGroups can have zero features, so it could be legal for this to be null even if there are aFeatureGroupsDimensionCount
   if(countTrainingSamples < 0) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting countTrainingSamples must be positive");
      return nullptr;
   }
   if(0 != countTrainingSamples && nullptr == trainingTargets) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting trainingTargets cannot be nullptr if 0 < countTrainingSamples");
      return nullptr;
   }
   if(0 != countTrainingSamples && 0 != countFeatures && nullptr == trainingBinnedData) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting trainingBinnedData cannot be nullptr if 0 < countTrainingSamples AND 0 < countFeatures");
      return nullptr;
   }
   if(0 != countTrainingSamples && nullptr == trainingPredictorScores) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting trainingPredictorScores cannot be nullptr if 0 < countTrainingSamples");
      return nullptr;
   }
   if(countValidationSamples < 0) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting countValidationSamples must be positive");
      return nullptr;
   }
   if(0 != countValidationSamples && nullptr == validationTargets) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting validationTargets cannot be nullptr if 0 < countValidationSamples");
      return nullptr;
   }
   if(0 != countValidationSamples && 0 != countFeatures && nullptr == validationBinnedData) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting validationBinnedData cannot be nullptr if 0 < countValidationSamples AND 0 < countFeatures");
      return nullptr;
   }
   if(0 != countValidationSamples && nullptr == validationPredictorScores) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting validationPredictorScores cannot be nullptr if 0 < countValidationSamples");
      return nullptr;
   }
   if(countInnerBags < 0) {
      // 0 means use the full set (good value).  1 means make a single bag (this is useless but allowed for comparison purposes).  2+ are good numbers of bag
      LOG_0(TraceLevelError, "ERROR AllocateBoosting countInnerBags must be positive");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t>(countFeatures)) {
      // the caller should not have been able to allocate enough memory in "features" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR AllocateBoosting !IsNumberConvertable<size_t>(countFeatures)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t>(countFeatureGroups)) {
      // the caller should not have been able to allocate enough memory in "aFeatureGroupsDimensionCount" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR AllocateBoosting !IsNumberConvertable<size_t>(countFeatureGroups)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t>(countTrainingSamples)) {
      // the caller should not have been able to allocate enough memory in "trainingTargets" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR AllocateBoosting !IsNumberConvertable<size_t>(countTrainingSamples)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t>(countValidationSamples)) {
      // the caller should not have been able to allocate enough memory in "validationTargets" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR AllocateBoosting !IsNumberConvertable<size_t>(countValidationSamples)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t>(countInnerBags)) {
      // this is just a warning since the caller doesn't pass us anything material, but if it's this high
      // then our allocation would fail since it can't even in pricipal fit into memory
      LOG_0(TraceLevelWarning, "WARNING AllocateBoosting !IsNumberConvertable<size_t>(countInnerBags)");
      return nullptr;
   }

   size_t cFeatures = static_cast<size_t>(countFeatures);
   size_t cFeatureGroups = static_cast<size_t>(countFeatureGroups);
   size_t cTrainingSamples = static_cast<size_t>(countTrainingSamples);
   size_t cValidationSamples = static_cast<size_t>(countValidationSamples);
   size_t cInnerBags = static_cast<size_t>(countInnerBags);

   size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   if(IsMultiplyError(cVectorLength, cTrainingSamples)) {
      // the caller should not have been able to allocate enough memory in "trainingPredictorScores" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR AllocateBoosting IsMultiplyError(cVectorLength, cTrainingSamples)");
      return nullptr;
   }
   if(IsMultiplyError(cVectorLength, cValidationSamples)) {
      // the caller should not have been able to allocate enough memory in "validationPredictorScores" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR AllocateBoosting IsMultiplyError(cVectorLength, cValidationSamples)");
      return nullptr;
   }

   BoosterCore * const pBoosterCore = BoosterCore::Allocate(
      randomSeed,
      runtimeLearningTypeOrCountTargetClasses,
      cFeatures,
      cFeatureGroups,
      cInnerBags,
      optionalTempParams,
      aFeaturesCategorical,
      aFeaturesBinCount,
      aFeatureGroupsDimensionCount,
      aFeatureGroupsFeatureIndexes,
      cTrainingSamples,
      trainingTargets,
      trainingBinnedData,
      aTrainingWeights, 
      trainingPredictorScores,
      cValidationSamples,
      validationTargets,
      validationBinnedData,
      aValidationWeights,
      validationPredictorScores
   );
   if(UNLIKELY(nullptr == pBoosterCore)) {
      LOG_0(TraceLevelWarning, "WARNING AllocateBoosting pBoosterCore->Initialize");
      return nullptr;
   }
   return pBoosterCore;
}

EBM_NATIVE_IMPORT_EXPORT_BODY BoosterHandle EBM_NATIVE_CALLING_CONVENTION CreateClassificationBooster(
   SeedEbmType randomSeed,
   IntEbmType countTargetClasses,
   IntEbmType countFeatures,
   const BoolEbmType * featuresCategorical,
   const IntEbmType * featuresBinCount,
   IntEbmType countFeatureGroups,
   const IntEbmType * featureGroupsDimensionCount,
   const IntEbmType * featureGroupsFeatureIndexes,
   IntEbmType countTrainingSamples,
   const IntEbmType * trainingBinnedData,
   const IntEbmType * trainingTargets,
   const FloatEbmType * trainingWeights,
   const FloatEbmType * trainingPredictorScores,
   IntEbmType countValidationSamples,
   const IntEbmType * validationBinnedData,
   const IntEbmType * validationTargets,
   const FloatEbmType * validationWeights,
   const FloatEbmType * validationPredictorScores,
   IntEbmType countInnerBags,
   const FloatEbmType * optionalTempParams
) {
   LOG_N(
      TraceLevelInfo, 
      "Entered CreateClassificationBooster: "
      "randomSeed=%" SeedEbmTypePrintf ", "
      "countTargetClasses=%" IntEbmTypePrintf ", "
      "countFeatures=%" IntEbmTypePrintf ", "
      "featuresCategorical=%p, "
      "featuresBinCount=%p, "
      "countFeatureGroups=%" IntEbmTypePrintf ", "
      "featureGroupsDimensionCount=%p, "
      "featureGroupsFeatureIndexes=%p, "
      "countTrainingSamples=%" IntEbmTypePrintf ", "
      "trainingBinnedData=%p, "
      "trainingTargets=%p, "
      "trainingWeights=%p, "
      "trainingPredictorScores=%p, "
      "countValidationSamples=%" IntEbmTypePrintf ", "
      "validationBinnedData=%p, "
      "validationTargets=%p, "
      "validationWeights=%p, "
      "validationPredictorScores=%p, "
      "countInnerBags=%" IntEbmTypePrintf ", "
      "optionalTempParams=%p"
      ,
      randomSeed,
      countTargetClasses,
      countFeatures, 
      static_cast<const void *>(featuresCategorical),
      static_cast<const void *>(featuresBinCount),
      countFeatureGroups,
      static_cast<const void *>(featureGroupsDimensionCount),
      static_cast<const void *>(featureGroupsFeatureIndexes), 
      countTrainingSamples, 
      static_cast<const void *>(trainingBinnedData), 
      static_cast<const void *>(trainingTargets), 
      static_cast<const void *>(trainingWeights),
      static_cast<const void *>(trainingPredictorScores),
      countValidationSamples, 
      static_cast<const void *>(validationBinnedData), 
      static_cast<const void *>(validationTargets), 
      static_cast<const void *>(validationWeights),
      static_cast<const void *>(validationPredictorScores),
      countInnerBags, 
      static_cast<const void *>(optionalTempParams)
      );
   if(countTargetClasses < 0) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster countTargetClasses can't be negative");
      return nullptr;
   }
   if(0 == countTargetClasses && (0 != countTrainingSamples || 0 != countValidationSamples)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster countTargetClasses can't be zero unless there are no training and no validation cases");
      return nullptr;
   }
   if(!IsNumberConvertable<ptrdiff_t>(countTargetClasses)) {
      LOG_0(TraceLevelWarning, "WARNING CreateClassificationBooster !IsNumberConvertable<ptrdiff_t>(countTargetClasses)");
      return nullptr;
   }
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = static_cast<ptrdiff_t>(countTargetClasses);
   const BoosterHandle boosterHandle = reinterpret_cast<BoosterHandle>(AllocateBoosting(
      randomSeed, 
      countFeatures, 
      featuresCategorical,
      featuresBinCount,
      countFeatureGroups,
      featureGroupsDimensionCount,
      featureGroupsFeatureIndexes, 
      runtimeLearningTypeOrCountTargetClasses, 
      countTrainingSamples, 
      trainingTargets, 
      trainingBinnedData, 
      trainingWeights, 
      trainingPredictorScores, 
      countValidationSamples, 
      validationTargets, 
      validationBinnedData, 
      validationWeights, 
      validationPredictorScores, 
      countInnerBags,
      optionalTempParams
   ));
   LOG_N(TraceLevelInfo, "Exited CreateClassificationBooster %p", static_cast<void *>(boosterHandle));
   return boosterHandle;
}

EBM_NATIVE_IMPORT_EXPORT_BODY BoosterHandle EBM_NATIVE_CALLING_CONVENTION CreateRegressionBooster(
   SeedEbmType randomSeed,
   IntEbmType countFeatures,
   const BoolEbmType * featuresCategorical,
   const IntEbmType * featuresBinCount,
   IntEbmType countFeatureGroups,
   const IntEbmType * featureGroupsDimensionCount,
   const IntEbmType * featureGroupsFeatureIndexes,
   IntEbmType countTrainingSamples,
   const IntEbmType * trainingBinnedData,
   const FloatEbmType * trainingTargets,
   const FloatEbmType * trainingWeights,
   const FloatEbmType * trainingPredictorScores,
   IntEbmType countValidationSamples,
   const IntEbmType * validationBinnedData,
   const FloatEbmType * validationTargets,
   const FloatEbmType * validationWeights,
   const FloatEbmType * validationPredictorScores,
   IntEbmType countInnerBags,
   const FloatEbmType * optionalTempParams
) {
   LOG_N(
      TraceLevelInfo, 
      "Entered CreateRegressionBooster: "
      "randomSeed=%" SeedEbmTypePrintf ", "
      "countFeatures=%" IntEbmTypePrintf ", "
      "featuresCategorical=%p, "
      "featuresBinCount=%p, "
      "countFeatureGroups=%" IntEbmTypePrintf ", "
      "featureGroupsDimensionCount=%p, "
      "featureGroupsFeatureIndexes=%p, "
      "countTrainingSamples=%" IntEbmTypePrintf ", "
      "trainingBinnedData=%p, "
      "trainingTargets=%p, "
      "trainingWeights=%p, "
      "trainingPredictorScores=%p, "
      "countValidationSamples=%" IntEbmTypePrintf ", "
      "validationBinnedData=%p, "
      "validationTargets=%p, "
      "validationWeights=%p, "
      "validationPredictorScores=%p, "
      "countInnerBags=%" IntEbmTypePrintf ", "
      "optionalTempParams=%p"
      ,
      randomSeed,
      countFeatures,
      static_cast<const void *>(featuresCategorical),
      static_cast<const void *>(featuresBinCount),
      countFeatureGroups,
      static_cast<const void *>(featureGroupsDimensionCount),
      static_cast<const void *>(featureGroupsFeatureIndexes), 
      countTrainingSamples, 
      static_cast<const void *>(trainingBinnedData), 
      static_cast<const void *>(trainingTargets), 
      static_cast<const void *>(trainingWeights),
      static_cast<const void *>(trainingPredictorScores),
      countValidationSamples, 
      static_cast<const void *>(validationBinnedData), 
      static_cast<const void *>(validationTargets), 
      static_cast<const void *>(validationWeights),
      static_cast<const void *>(validationPredictorScores),
      countInnerBags, 
      static_cast<const void *>(optionalTempParams)
   );
   const BoosterHandle boosterHandle = reinterpret_cast<BoosterHandle>(AllocateBoosting(
      randomSeed, 
      countFeatures, 
      featuresCategorical,
      featuresBinCount,
      countFeatureGroups, 
      featureGroupsDimensionCount,
      featureGroupsFeatureIndexes, 
      k_regression, 
      countTrainingSamples, 
      trainingTargets, 
      trainingBinnedData, 
      trainingWeights, 
      trainingPredictorScores, 
      countValidationSamples, 
      validationTargets, 
      validationBinnedData, 
      validationWeights,
      validationPredictorScores, 
      countInnerBags,
      optionalTempParams
   ));
   LOG_N(TraceLevelInfo, "Exited CreateRegressionBooster %p", static_cast<void *>(boosterHandle));
   return boosterHandle;
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GetBestModelFeatureGroup(
   BoosterHandle boosterHandle,
   IntEbmType indexFeatureGroup,
   FloatEbmType * modelFeatureGroupTensorOut
) {
   LOG_N(
      TraceLevelInfo,
      "Entered GetBestModelFeatureGroup: "
      "boosterHandle=%p, "
      "indexFeatureGroup=%" IntEbmTypePrintf ", "
      "modelFeatureGroupTensorOut=%p, "
      ,
      static_cast<void *>(boosterHandle),
      indexFeatureGroup,
      modelFeatureGroupTensorOut
   );

   BoosterCore * pBoosterCore = reinterpret_cast<BoosterCore *>(boosterHandle);
   if(nullptr == pBoosterCore) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup boosterHandle cannot be nullptr");
      return IntEbmType { 1 };
   }
   if(indexFeatureGroup < 0) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup indexFeatureGroup must be positive");
      return IntEbmType { 1 };
   }
   if(!IsNumberConvertable<size_t>(indexFeatureGroup)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup indexFeatureGroup is too high to index");
      return IntEbmType { 1 };
   }
   size_t iFeatureGroup = static_cast<size_t>(indexFeatureGroup);
   if(pBoosterCore->GetCountFeatureGroups() <= iFeatureGroup) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup indexFeatureGroup above the number of feature groups that we have");
      return IntEbmType { 1 };
   }

   if(ptrdiff_t { 0 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses() ||
      ptrdiff_t { 1 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses()) {
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.  
      // If there were logits in this model, they'd all be infinity, but you could alternatively think of this 
      // model as having no logits, since the number of logits can be one less than the number of target classes.
      LOG_0(TraceLevelInfo, "Exited GetBestModelFeatureGroup no model");
      return IntEbmType { 0 };
   }

   if(nullptr == modelFeatureGroupTensorOut) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup modelFeatureGroupTensorOut cannot be nullptr");
      return IntEbmType { 1 };
   }

   // if pBoosterCore->GetFeatureGroups() is nullptr, then m_cFeatureGroups was 0, but we checked above that 
   // iFeatureGroup was less than cFeatureGroups
   EBM_ASSERT(nullptr != pBoosterCore->GetFeatureGroups());

   const FeatureGroup * const pFeatureGroup = pBoosterCore->GetFeatureGroups()[iFeatureGroup];
   const size_t cDimensions = pFeatureGroup->GetCountDimensions();
   size_t cValues = GetVectorLength(pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses());
   if(0 != cDimensions) {
      const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
      const FeatureGroupEntry * const pFeatureGroupEntryEnd = &pFeatureGroupEntry[cDimensions];
      do {
         const size_t cBins = pFeatureGroupEntry->m_pFeature->GetCountBins();
         // we've allocated this memory, so it should be reachable, so these numbers should multiply
         EBM_ASSERT(!IsMultiplyError(cBins, cValues));
         cValues *= cBins;
         ++pFeatureGroupEntry;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
   }

   // if pBoosterCore->GetBestModel() is nullptr, then either:
   //    1) m_cFeatureGroups was 0, but we checked above that iFeatureGroup was less than cFeatureGroups
   //    2) If m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0, but we checked for this above too
   EBM_ASSERT(nullptr != pBoosterCore->GetBestModel());

   SegmentedTensor * const pBestModel = pBoosterCore->GetBestModel()[iFeatureGroup];
   EBM_ASSERT(nullptr != pBestModel);
   EBM_ASSERT(pBestModel->GetExpanded()); // the model should have been expanded at startup
   FloatEbmType * const pValues = pBestModel->GetValuePointer();
   EBM_ASSERT(nullptr != pValues);

   EBM_ASSERT(!IsMultiplyError(sizeof(*pValues), cValues));
   memcpy(modelFeatureGroupTensorOut, pValues, sizeof(*pValues) * cValues);

   LOG_0(TraceLevelInfo, "Exited GetBestModelFeatureGroup");
   return IntEbmType { 0 };
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GetCurrentModelFeatureGroup(
   BoosterHandle boosterHandle,
   IntEbmType indexFeatureGroup,
   FloatEbmType * modelFeatureGroupTensorOut
) {
   LOG_N(
      TraceLevelInfo,
      "Entered GetCurrentModelFeatureGroup: "
      "boosterHandle=%p, "
      "indexFeatureGroup=%" IntEbmTypePrintf ", "
      "modelFeatureGroupTensorOut=%p, "
      ,
      static_cast<void *>(boosterHandle),
      indexFeatureGroup,
      modelFeatureGroupTensorOut
   );

   BoosterCore * pBoosterCore = reinterpret_cast<BoosterCore *>(boosterHandle);
   if(nullptr == pBoosterCore) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup boosterHandle cannot be nullptr");
      return IntEbmType { 1 };
   }
   if(indexFeatureGroup < 0) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup indexFeatureGroup must be positive");
      return IntEbmType { 1 };
   }
   if(!IsNumberConvertable<size_t>(indexFeatureGroup)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup indexFeatureGroup is too high to index");
      return IntEbmType { 1 };
   }
   size_t iFeatureGroup = static_cast<size_t>(indexFeatureGroup);
   if(pBoosterCore->GetCountFeatureGroups() <= iFeatureGroup) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup indexFeatureGroup above the number of feature groups that we have");
      return IntEbmType { 1 };
   }

   if(ptrdiff_t { 0 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses() || 
      ptrdiff_t { 1 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses()) 
   {
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.  
      // If there were logits in this model, they'd all be infinity, but you could alternatively think of this 
      // model as having no logits, since the number of logits can be one less than the number of target classes.
      LOG_0(TraceLevelInfo, "Exited GetCurrentModelFeatureGroup no model");
      return IntEbmType { 0 };
   }

   if(nullptr == modelFeatureGroupTensorOut) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup modelFeatureGroupTensorOut cannot be nullptr");
      return IntEbmType { 1 };
   }

   // if pBoosterCore->GetFeatureGroups() is nullptr, then m_cFeatureGroups was 0, but we checked above that 
   // iFeatureGroup was less than cFeatureGroups
   EBM_ASSERT(nullptr != pBoosterCore->GetFeatureGroups());

   const FeatureGroup * const pFeatureGroup = pBoosterCore->GetFeatureGroups()[iFeatureGroup];
   const size_t cDimensions = pFeatureGroup->GetCountDimensions();
   size_t cValues = GetVectorLength(pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses());
   if(0 != cDimensions) {
      const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
      const FeatureGroupEntry * const pFeatureGroupEntryEnd = &pFeatureGroupEntry[cDimensions];
      do {
         const size_t cBins = pFeatureGroupEntry->m_pFeature->GetCountBins();
         // we've allocated this memory, so it should be reachable, so these numbers should multiply
         EBM_ASSERT(!IsMultiplyError(cBins, cValues));
         cValues *= cBins;
         ++pFeatureGroupEntry;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
   }

   // if pBoosterCore->GetCurrentModel() is nullptr, then either:
   //    1) m_cFeatureGroups was 0, but we checked above that iFeatureGroup was less than cFeatureGroups
   //    2) If m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0, but we checked for this above too
   EBM_ASSERT(nullptr != pBoosterCore->GetCurrentModel());

   SegmentedTensor * const pCurrentModel = pBoosterCore->GetCurrentModel()[iFeatureGroup];
   EBM_ASSERT(nullptr != pCurrentModel);
   EBM_ASSERT(pCurrentModel->GetExpanded()); // the model should have been expanded at startup
   FloatEbmType * const pValues = pCurrentModel->GetValuePointer();
   EBM_ASSERT(nullptr != pValues);

   EBM_ASSERT(!IsMultiplyError(sizeof(*pValues), cValues));
   memcpy(modelFeatureGroupTensorOut, pValues, sizeof(*pValues) * cValues);

   LOG_0(TraceLevelInfo, "Exited GetCurrentModelFeatureGroup");
   return IntEbmType { 0 };
}

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION FreeBooster(
   BoosterHandle boosterHandle
) {
   LOG_N(TraceLevelInfo, "Entered FreeBooster: boosterHandle=%p", static_cast<void *>(boosterHandle));

   BoosterCore * pBoosterCore = reinterpret_cast<BoosterCore *>(boosterHandle);

   // it's legal to call free on nullptr, just like for free().  This is checked inside BoosterCore::Free()
   BoosterCore::Free(pBoosterCore);

   LOG_0(TraceLevelInfo, "Exited FreeBooster");
}

} // DEFINED_ZONE_NAME
