// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebm_native.h"
#include "EbmInternal.h"
// very independent includes
#include "Logging.h" // EBM_ASSERT & LOG
#include "RandomStream.h"
#include "SegmentedTensor.h"
#include "EbmStatisticUtils.h"
// feature includes
#include "FeatureAtomic.h"
// FeatureGroup.h depends on FeatureInternal.h
#include "FeatureGroup.h"
// dataset depends on features
#include "DataSetBoosting.h"
// samples is somewhat independent from datasets, but relies on an indirect coupling with them
#include "SamplingSet.h"
#include "TreeSweep.h"

#include "Booster.h"

extern void InitializeResiduals(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const size_t cSamples,
   const void * const aTargetData,
   const FloatEbmType * const aPredictorScores,
   FloatEbmType * const aTempFloatVector,
   FloatEbmType * pResidualError
);

INLINE_ALWAYS static size_t GetCountItemsBitPacked(const size_t cBits) {
   EBM_ASSERT(size_t { 1 } <= cBits);
   return k_cBitsForStorageType / cBits;
}

void EbmBoostingState::DeleteSegmentedTensors(const size_t cFeatureGroups, SegmentedTensor ** const apSegmentedTensors) {
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

SegmentedTensor ** EbmBoostingState::InitializeSegmentedTensors(
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
         SegmentedTensor::Allocate(pFeatureGroup->GetCountFeatures(), cVectorLength);
      if(UNLIKELY(nullptr == pSegmentedTensors)) {
         LOG_0(TraceLevelWarning, "WARNING InitializeSegmentedTensors nullptr == pSegmentedTensors");
         DeleteSegmentedTensors(cFeatureGroups, apSegmentedTensors);
         return nullptr;
      }

      if(0 == pFeatureGroup->GetCountFeatures()) {
         // if there are zero dimensions, then we have a tensor with 1 item, and we're already expanded
         pSegmentedTensors->SetExpanded();
      } else {
         // if our segmented region has no dimensions, then it's already a fully expanded with 1 bin

         // TODO optimize the next few lines
         // TODO there might be a nicer way to expand this at allocation time (fill with zeros is easier)
         // we want to return a pointer to our interior state in the GetCurrentModelFeatureGroup and GetBestModelFeatureGroup functions.  
         // For simplicity we don't transmit the divions, so we need to expand our SegmentedRegion before returning the easiest way to ensure that the 
         // SegmentedRegion is expanded is to start it off expanded, and then we don't have to check later since anything merged into an expanded 
         // SegmentedRegion will itself be expanded
         size_t acDivisionIntegersEnd[k_cDimensionsMax];
         size_t iDimension = 0;
         do {
            acDivisionIntegersEnd[iDimension] = pFeatureGroup->GetFeatureGroupEntries()[iDimension].m_pFeature->GetCountBins();
            ++iDimension;
         } while(iDimension < pFeatureGroup->GetCountFeatures());

         if(pSegmentedTensors->Expand(acDivisionIntegersEnd)) {
            LOG_0(TraceLevelWarning, "WARNING InitializeSegmentedTensors pSegmentedTensors->Expand(acDivisionIntegersEnd)");
            DeleteSegmentedTensors(cFeatureGroups, apSegmentedTensors);
            return nullptr;
         }
      }

      *ppSegmentedTensors = pSegmentedTensors;
      ++ppSegmentedTensors;
   }

   LOG_0(TraceLevelInfo, "Exited InitializeSegmentedTensors");
   return apSegmentedTensors;
}

void EbmBoostingState::Free(EbmBoostingState * const pBoostingState) {
   LOG_0(TraceLevelInfo, "Entered EbmBoostingState::Free");
   if(nullptr != pBoostingState) {
      pBoostingState->m_trainingSet.Destruct();
      pBoostingState->m_validationSet.Destruct();

      CachedBoostingThreadResources::Free(pBoostingState->m_pCachedThreadResources);

      SamplingSet::FreeSamplingSets(pBoostingState->m_cSamplingSets, pBoostingState->m_apSamplingSets);

      FeatureGroup::FreeFeatureGroups(pBoostingState->m_cFeatureGroups, pBoostingState->m_apFeatureGroups);

      free(pBoostingState->m_aFeatures);

      DeleteSegmentedTensors(pBoostingState->m_cFeatureGroups, pBoostingState->m_apCurrentModel);
      DeleteSegmentedTensors(pBoostingState->m_cFeatureGroups, pBoostingState->m_apBestModel);
      SegmentedTensor::Free(pBoostingState->m_pSmallChangeToModelOverwriteSingleSamplingSet);
      SegmentedTensor::Free(pBoostingState->m_pSmallChangeToModelAccumulatedFromSamplingSets);

      free(pBoostingState);
   }
   LOG_0(TraceLevelInfo, "Exited EbmBoostingState::Free");
}

EbmBoostingState * EbmBoostingState::Allocate(
   const SeedEbmType randomSeed,
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const size_t cFeatures,
   const size_t cFeatureGroups,
   const size_t cSamplingSets,
   const FloatEbmType * const optionalTempParams,
   const EbmNativeFeature * const aFeatures,
   const EbmNativeFeatureGroup * const aFeatureGroups, 
   const IntEbmType * featureGroupIndexes, 
   const size_t cTrainingSamples, 
   const void * const aTrainingTargets, 
   const IntEbmType * const aTrainingBinnedData, 
   const FloatEbmType * const aTrainingPredictorScores, 
   const size_t cValidationSamples, 
   const void * const aValidationTargets, 
   const IntEbmType * const aValidationBinnedData, 
   const FloatEbmType * const aValidationPredictorScores
) {
   // optionalTempParams isn't used by default.  It's meant to provide an easy way for python or other higher
   // level languages to pass EXPERIMENTAL temporary parameters easily to the C++ code.
   UNUSED(optionalTempParams);

   LOG_0(TraceLevelInfo, "Entered EbmBoostingState::Initialize");

   EbmBoostingState * const pBooster = EbmMalloc<EbmBoostingState>();
   if(UNLIKELY(nullptr == pBooster)) {
      LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == pBooster");
      return nullptr;
   }
   pBooster->InitializeZero();

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   pBooster->m_pSmallChangeToModelOverwriteSingleSamplingSet = 
      SegmentedTensor::Allocate(k_cDimensionsMax, cVectorLength);
   if(UNLIKELY(nullptr == pBooster->m_pSmallChangeToModelOverwriteSingleSamplingSet)) {
      LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == m_pSmallChangeToModelOverwriteSingleSamplingSet");
      EbmBoostingState::Free(pBooster);
      return nullptr;
   }

   pBooster->m_pSmallChangeToModelAccumulatedFromSamplingSets = 
      SegmentedTensor::Allocate(k_cDimensionsMax, cVectorLength);
   if(UNLIKELY(nullptr == pBooster->m_pSmallChangeToModelAccumulatedFromSamplingSets)) {
      LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == m_pSmallChangeToModelAccumulatedFromSamplingSets");
      EbmBoostingState::Free(pBooster);
      return nullptr;
   }

   LOG_0(TraceLevelInfo, "EbmBoostingState::Initialize starting feature processing");
   if(0 != cFeatures) {
      pBooster->m_aFeatures = EbmMalloc<Feature>(cFeatures);
      if(nullptr == pBooster->m_aFeatures) {
         LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == pBooster->m_aFeatures");
         EbmBoostingState::Free(pBooster);
         return nullptr;
      }
      pBooster->m_cFeatures = cFeatures;

      const EbmNativeFeature * pFeatureInitialize = aFeatures;
      const EbmNativeFeature * const pFeatureEnd = &aFeatures[cFeatures];
      EBM_ASSERT(pFeatureInitialize < pFeatureEnd);
      size_t iFeatureInitialize = 0;
      do {
         static_assert(FeatureType::Ordinal == static_cast<FeatureType>(FeatureTypeOrdinal), 
            "FeatureType::Ordinal must have the same value as FeatureTypeOrdinal");
         static_assert(FeatureType::Nominal == static_cast<FeatureType>(FeatureTypeNominal), 
            "FeatureType::Nominal must have the same value as FeatureTypeNominal");
         if(FeatureTypeOrdinal != pFeatureInitialize->featureType && FeatureTypeNominal != pFeatureInitialize->featureType) {
            LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize featureType must either be FeatureTypeOrdinal or FeatureTypeNominal");
            EbmBoostingState::Free(pBooster);
            return nullptr;
         }
         FeatureType featureType = static_cast<FeatureType>(pFeatureInitialize->featureType);

         IntEbmType countBins = pFeatureInitialize->countBins;
         if(countBins < 0) {
            LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize countBins cannot be negative");
            EbmBoostingState::Free(pBooster);
            return nullptr;
         }
         if(0 == countBins && (0 != cTrainingSamples || 0 != cValidationSamples)) {
            LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize countBins cannot be zero if either 0 < cTrainingSamples OR 0 < cValidationSamples");
            EbmBoostingState::Free(pBooster);
            return nullptr;
         }
         if(!IsNumberConvertable<size_t>(countBins)) {
            LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize countBins is too high for us to allocate enough memory");
            EbmBoostingState::Free(pBooster);
            return nullptr;
         }
         size_t cBins = static_cast<size_t>(countBins);
         if(0 == cBins) {
            // we can handle 0 == cBins even though that's a degenerate case that shouldn't be boosted on.  0 bins
            // can only occur if there were zero training and zero validation cases since the 
            // features would require a value, even if it was 0.
            LOG_0(TraceLevelInfo, "INFO EbmBoostingState::Initialize feature with 0 values");
         } else if(1 == cBins) {
            // we can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on. 
            // Dimensions with 1 bin don't contribute anything since they always have the same value.
            LOG_0(TraceLevelInfo, "INFO EbmBoostingState::Initialize feature with 1 value");
         }
         if(EBM_FALSE != pFeatureInitialize->hasMissing && EBM_TRUE != pFeatureInitialize->hasMissing) {
            LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize hasMissing must either be EBM_TRUE or EBM_FALSE");
            EbmBoostingState::Free(pBooster);
            return nullptr;
         }
         const bool bMissing = EBM_FALSE != pFeatureInitialize->hasMissing;

         pBooster->m_aFeatures[iFeatureInitialize].Initialize(cBins, iFeatureInitialize, featureType, bMissing);

         EBM_ASSERT(EBM_FALSE == pFeatureInitialize->hasMissing); // TODO : implement this, then remove this assert
         EBM_ASSERT(FeatureTypeOrdinal == pFeatureInitialize->featureType); // TODO : implement this, then remove this assert

         ++iFeatureInitialize;
         ++pFeatureInitialize;
      } while(pFeatureEnd != pFeatureInitialize);
   }
   LOG_0(TraceLevelInfo, "EbmBoostingState::Initialize done feature processing");

   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);
   size_t cBytesArrayEquivalentSplitMax = 0;

   EBM_ASSERT(nullptr == pBooster->m_apCurrentModel);
   EBM_ASSERT(nullptr == pBooster->m_apBestModel);

   LOG_0(TraceLevelInfo, "EbmBoostingState::Initialize starting feature group processing");
   if(0 != cFeatureGroups) {
      pBooster->m_cFeatureGroups = cFeatureGroups;
      pBooster->m_apFeatureGroups = FeatureGroup::AllocateFeatureGroups(cFeatureGroups);
      if(UNLIKELY(nullptr == pBooster->m_apFeatureGroups)) {
         LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize 0 != m_cFeatureGroups && nullptr == m_apFeatureGroups");
         EbmBoostingState::Free(pBooster);
         return nullptr;
      }

      if(GetSweepTreeNodeSizeOverflow(bClassification, cVectorLength)) {
         LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize GetSweepTreeNodeSizeOverflow(bClassification, cVectorLength)");
         EbmBoostingState::Free(pBooster);
         return nullptr;
      }
      size_t cBytesPerSweepTreeNode = GetSweepTreeNodeSize(bClassification, cVectorLength);

      const IntEbmType * pFeatureGroupIndex = featureGroupIndexes;
      size_t iFeatureGroup = 0;
      do {
         const EbmNativeFeatureGroup * const pFeatureGroupInterop = &aFeatureGroups[iFeatureGroup];
         const IntEbmType countFeaturesInGroup = pFeatureGroupInterop->countFeaturesInGroup;
         if(countFeaturesInGroup < 0) {
            LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize countFeaturesInGroup cannot be negative");
            EbmBoostingState::Free(pBooster);
            return nullptr;
         }
         if(!IsNumberConvertable<size_t>(countFeaturesInGroup)) {
            // if countFeaturesInGroup exceeds the size of size_t, then we wouldn't be able to find it
            // in the array passed to us
            LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize countFeaturesInGroup is too high to index");
            EbmBoostingState::Free(pBooster);
            return nullptr;
         }
         size_t cFeaturesInGroup = static_cast<size_t>(countFeaturesInGroup);
         size_t cSignificantFeaturesInGroup = 0;
         const IntEbmType * pFeatureGroupIndexEnd = pFeatureGroupIndex;
         if(UNLIKELY(0 == cFeaturesInGroup)) {
            LOG_0(TraceLevelInfo, "INFO EbmBoostingState::Initialize empty feature group");
         } else {
            if(nullptr == pFeatureGroupIndex) {
               LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize featureGroupIndexes is null when there are FeatureGroups with non-zero numbers of features");
               EbmBoostingState::Free(pBooster);
               return nullptr;
            }
            pFeatureGroupIndexEnd += cFeaturesInGroup;
            const IntEbmType * pFeatureGroupIndexTemp = pFeatureGroupIndex;
            do {
               const IntEbmType indexFeatureInterop = *pFeatureGroupIndexTemp;
               if(indexFeatureInterop < 0) {
                  LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize featureGroupIndexes value cannot be negative");
                  EbmBoostingState::Free(pBooster);
                  return nullptr;
               }
               if(!IsNumberConvertable<size_t>(indexFeatureInterop)) {
                  LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize featureGroupIndexes value too big to reference memory");
                  EbmBoostingState::Free(pBooster);
                  return nullptr;
               }
               const size_t iFeatureForGroup = static_cast<size_t>(indexFeatureInterop);

               if(cFeatures <= iFeatureForGroup) {
                  LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize featureGroupIndexes value must be less than the number of features");
                  EbmBoostingState::Free(pBooster);
                  return nullptr;
               }

               EBM_ASSERT(1 <= cFeatures);
               EBM_ASSERT(nullptr != pBooster->m_aFeatures);

               Feature * const pInputFeature = &pBooster->m_aFeatures[iFeatureForGroup];
               if(LIKELY(1 < pInputFeature->GetCountBins())) {
                  // if we have only 1 bin, then we can eliminate the feature from consideration since the resulting tensor loses one dimension but is 
                  // otherwise indistinquishable from the original data
                  ++cSignificantFeaturesInGroup;
               } else {
                  LOG_0(TraceLevelInfo, "INFO EbmBoostingState::Initialize feature group with no useful features");
               }
               ++pFeatureGroupIndexTemp;
            } while(pFeatureGroupIndexEnd != pFeatureGroupIndexTemp);

            if(k_cDimensionsMax < cSignificantFeaturesInGroup) {
               // if we try to run with more than k_cDimensionsMax we'll exceed our memory capacity, so let's exit here instead
               LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize k_cDimensionsMax < cSignificantFeaturesInGroup");
               EbmBoostingState::Free(pBooster);
               return nullptr;
            }
         }

         FeatureGroup * pFeatureGroup = FeatureGroup::Allocate(cSignificantFeaturesInGroup, iFeatureGroup);
         if(nullptr == pFeatureGroup) {
            LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == pFeatureGroup");
            EbmBoostingState::Free(pBooster);
            return nullptr;
         }
         // assign our pointer directly to our array right now so that we can't loose the memory if we decide to exit due to an error below
         pBooster->m_apFeatureGroups[iFeatureGroup] = pFeatureGroup;

         if(UNLIKELY(0 != cSignificantFeaturesInGroup)) {
            EBM_ASSERT(nullptr != featureGroupIndexes);
            size_t cEquivalentSplits = 1;
            size_t cTensorBins = 1;
            FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
            do {
               const IntEbmType indexFeatureInterop = *pFeatureGroupIndex;
               EBM_ASSERT(0 <= indexFeatureInterop);
               EBM_ASSERT(IsNumberConvertable<size_t>(indexFeatureInterop)); // this was checked above
               const size_t iFeatureForGroup = static_cast<size_t>(indexFeatureInterop);
               EBM_ASSERT(iFeatureForGroup < cFeatures);
               const Feature * const pInputFeature = &pBooster->m_aFeatures[iFeatureForGroup];
               const size_t cBins = pInputFeature->GetCountBins();
               if(LIKELY(1 < cBins)) {
                  // if we have only 1 bin, then we can eliminate the feature from consideration since the resulting tensor loses one dimension but is 
                  // otherwise indistinquishable from the original data
                  pFeatureGroupEntry->m_pFeature = pInputFeature;
                  ++pFeatureGroupEntry;
                  if(IsMultiplyError(cTensorBins, cBins)) {
                     // if this overflows, we definetly won't be able to allocate it
                     LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize IsMultiplyError(cTensorStates, cBins)");
                     EbmBoostingState::Free(pBooster);
                     return nullptr;
                  }
                  cTensorBins *= cBins;
                  cEquivalentSplits *= cBins - 1; // we can only split between the bins
               }
               ++pFeatureGroupIndex;
            } while(pFeatureGroupIndexEnd != pFeatureGroupIndex);
            EBM_ASSERT(1 < cTensorBins);

            size_t cBytesArrayEquivalentSplit;
            if(1 == cSignificantFeaturesInGroup) {
               if(IsMultiplyError(cEquivalentSplits, cBytesPerSweepTreeNode)) {
                  LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize IsMultiplyError(cEquivalentSplits, cBytesPerSweepTreeNode)");
                  EbmBoostingState::Free(pBooster);
                  return nullptr;
               }
               cBytesArrayEquivalentSplit = cEquivalentSplits * cBytesPerSweepTreeNode;
            } else {
               // TODO : someday add equal gain multidimensional randomized picking.  It's rather hard though with the existing sweep functions for 
               // multidimensional right now
               cBytesArrayEquivalentSplit = 0;
            }
            if(cBytesArrayEquivalentSplitMax < cBytesArrayEquivalentSplit) {
               cBytesArrayEquivalentSplitMax = cBytesArrayEquivalentSplit;
            }

            // if cSignificantFeaturesInGroup is zero, don't both initializing pFeatureGroup->GetCountItemsPerBitPackedDataUnit()
            const size_t cBitsRequiredMin = CountBitsRequired(cTensorBins - 1);
            EBM_ASSERT(1 <= cBitsRequiredMin); // 1 < cTensorBins otherwise we'd have filtered it out above
            pFeatureGroup->SetCountItemsPerBitPackedDataUnit(GetCountItemsBitPacked(cBitsRequiredMin));
         }
         pFeatureGroupIndex = pFeatureGroupIndexEnd;

         ++iFeatureGroup;
      } while(iFeatureGroup < cFeatureGroups);

      if(!bClassification || ptrdiff_t { 2 } <= runtimeLearningTypeOrCountTargetClasses) {
         pBooster->m_apCurrentModel = InitializeSegmentedTensors(cFeatureGroups, pBooster->m_apFeatureGroups, cVectorLength);
         if(nullptr == pBooster->m_apCurrentModel) {
            LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == m_apCurrentModel");
            EbmBoostingState::Free(pBooster);
            return nullptr;
         }
         pBooster->m_apBestModel = InitializeSegmentedTensors(cFeatureGroups, pBooster->m_apFeatureGroups, cVectorLength);
         if(nullptr == pBooster->m_apBestModel) {
            LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == m_apBestModel");
            EbmBoostingState::Free(pBooster);
            return nullptr;
         }
      }
   }
   LOG_0(TraceLevelInfo, "EbmBoostingState::Initialize finished feature group processing");

   pBooster->m_pCachedThreadResources = CachedBoostingThreadResources::Allocate(
      runtimeLearningTypeOrCountTargetClasses,
      cBytesArrayEquivalentSplitMax
   );
   if(UNLIKELY(nullptr == pBooster->m_pCachedThreadResources)) {
      LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == m_pCachedThreadResources");
      EbmBoostingState::Free(pBooster);
      return nullptr;
   }

   if(pBooster->m_trainingSet.Initialize(
      true, 
      bClassification, 
      bClassification, 
      cFeatureGroups, 
      pBooster->m_apFeatureGroups,
      cTrainingSamples, 
      aTrainingBinnedData, 
      aTrainingTargets, 
      aTrainingPredictorScores, 
      runtimeLearningTypeOrCountTargetClasses
   )) {
      LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize m_trainingSet.Initialize");
      EbmBoostingState::Free(pBooster);
      return nullptr;
   }

   if(pBooster->m_validationSet.Initialize(
      !bClassification, 
      bClassification, 
      bClassification, 
      cFeatureGroups, 
      pBooster->m_apFeatureGroups,
      cValidationSamples, 
      aValidationBinnedData, 
      aValidationTargets, 
      aValidationPredictorScores, 
      runtimeLearningTypeOrCountTargetClasses
   )) {
      LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize m_validationSet.Initialize");
      EbmBoostingState::Free(pBooster);
      return nullptr;
   }

   pBooster->m_randomStream.InitializeUnsigned(randomSeed, k_boosterRandomizationMix);

   EBM_ASSERT(nullptr == pBooster->m_apSamplingSets);
   if(0 != cTrainingSamples) {
      pBooster->m_cSamplingSets = cSamplingSets;
      pBooster->m_apSamplingSets = SamplingSet::GenerateSamplingSets(&pBooster->m_randomStream, &pBooster->m_trainingSet, cSamplingSets);
      if(UNLIKELY(nullptr == pBooster->m_apSamplingSets)) {
         LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == m_apSamplingSets");
         EbmBoostingState::Free(pBooster);
         return nullptr;
      }
   }

   if(bClassification) {
      if(0 != cTrainingSamples) {
         InitializeResiduals(
            runtimeLearningTypeOrCountTargetClasses,
            cTrainingSamples,
            aTrainingTargets,
            aTrainingPredictorScores,
            pBooster->GetCachedThreadResources()->GetTempFloatVector(),
            pBooster->m_trainingSet.GetResidualPointer()
         );
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      if(0 != cTrainingSamples) {
         InitializeResiduals(
            k_regression,
            cTrainingSamples,
            aTrainingTargets,
            aTrainingPredictorScores,
            nullptr,
            pBooster->m_trainingSet.GetResidualPointer()
         );
      }
      if(0 != cValidationSamples) {
         InitializeResiduals(
            k_regression,
            cValidationSamples,
            aValidationTargets,
            aValidationPredictorScores,
            nullptr,
            pBooster->m_validationSet.GetResidualPointer()
         );
      }
   }

   pBooster->m_runtimeLearningTypeOrCountTargetClasses = runtimeLearningTypeOrCountTargetClasses;
   pBooster->m_bestModelMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::max() };

   LOG_0(TraceLevelInfo, "Exited EbmBoostingState::Initialize");
   return pBooster;
}

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
static EbmBoostingState * AllocateBoosting(
   const SeedEbmType randomSeed,
   const IntEbmType countFeatures, 
   const EbmNativeFeature * const features, 
   const IntEbmType countFeatureGroups, 
   const EbmNativeFeatureGroup * const featureGroups, 
   const IntEbmType * const featureGroupIndexes, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, 
   const IntEbmType countTrainingSamples, 
   const void * const trainingTargets, 
   const IntEbmType * const trainingBinnedData, 
   const FloatEbmType * const trainingPredictorScores, 
   const IntEbmType countValidationSamples, 
   const void * const validationTargets, 
   const IntEbmType * const validationBinnedData, 
   const FloatEbmType * const validationPredictorScores, 
   const IntEbmType countInnerBags,
   const FloatEbmType * const optionalTempParams
) {
   // TODO : give AllocateBoosting the same calling parameter order as InitializeBoostingClassification

   if(countFeatures < 0) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting countFeatures must be positive");
      return nullptr;
   }
   if(0 != countFeatures && nullptr == features) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting features cannot be nullptr if 0 < countFeatures");
      return nullptr;
   }
   if(countFeatureGroups < 0) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting countFeatureGroups must be positive");
      return nullptr;
   }
   if(0 != countFeatureGroups && nullptr == featureGroups) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting featureGroups cannot be nullptr if 0 < countFeatureGroups");
      return nullptr;
   }
   // featureGroupIndexes -> it's legal for featureGroupIndexes to be nullptr if there are no features indexed by our featureGroups.  
   // FeatureGroups can have zero features, so it could be legal for this to be null even if there are featureGroups
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
      // the caller should not have been able to allocate enough memory in "featureGroups" if this didn't fit in memory
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

   EbmBoostingState * const pEbmBoostingState = EbmBoostingState::Allocate(
      randomSeed,
      runtimeLearningTypeOrCountTargetClasses,
      cFeatures,
      cFeatureGroups,
      cInnerBags,
      optionalTempParams,
      features,
      featureGroups,
      featureGroupIndexes,
      cTrainingSamples,
      trainingTargets,
      trainingBinnedData,
      trainingPredictorScores,
      cValidationSamples,
      validationTargets,
      validationBinnedData,
      validationPredictorScores
   );
   if(UNLIKELY(nullptr == pEbmBoostingState)) {
      LOG_0(TraceLevelWarning, "WARNING AllocateBoosting pEbmBoostingState->Initialize");
      return nullptr;
   }
   return pEbmBoostingState;
}

EBM_NATIVE_IMPORT_EXPORT_BODY PEbmBoosting EBM_NATIVE_CALLING_CONVENTION InitializeBoostingClassification(
   SeedEbmType randomSeed,
   IntEbmType countTargetClasses,
   IntEbmType countFeatures,
   const EbmNativeFeature * features,
   IntEbmType countFeatureGroups,
   const EbmNativeFeatureGroup * featureGroups,
   const IntEbmType * featureGroupIndexes,
   IntEbmType countTrainingSamples,
   const IntEbmType * trainingBinnedData,
   const IntEbmType * trainingTargets,
   const FloatEbmType * trainingPredictorScores,
   IntEbmType countValidationSamples,
   const IntEbmType * validationBinnedData,
   const IntEbmType * validationTargets,
   const FloatEbmType * validationPredictorScores,
   IntEbmType countInnerBags,
   const FloatEbmType * optionalTempParams
) {
   LOG_N(
      TraceLevelInfo, 
      "Entered InitializeBoostingClassification: "
      "randomSeed=%" SeedEbmTypePrintf ", "
      "countTargetClasses=%" IntEbmTypePrintf ", "
      "countFeatures=%" IntEbmTypePrintf ", "
      "features=%p, "
      "countFeatureGroups=%" IntEbmTypePrintf ", "
      "featureGroups=%p, "
      "featureGroupIndexes=%p, "
      "countTrainingSamples=%" IntEbmTypePrintf ", "
      "trainingBinnedData=%p, "
      "trainingTargets=%p, "
      "trainingPredictorScores=%p, "
      "countValidationSamples=%" IntEbmTypePrintf ", "
      "validationBinnedData=%p, "
      "validationTargets=%p, "
      "validationPredictorScores=%p, "
      "countInnerBags=%" IntEbmTypePrintf ", "
      "optionalTempParams=%p"
      ,
      randomSeed,
      countTargetClasses,
      countFeatures, 
      static_cast<const void *>(features), 
      countFeatureGroups, 
      static_cast<const void *>(featureGroups), 
      static_cast<const void *>(featureGroupIndexes), 
      countTrainingSamples, 
      static_cast<const void *>(trainingBinnedData), 
      static_cast<const void *>(trainingTargets), 
      static_cast<const void *>(trainingPredictorScores), 
      countValidationSamples, 
      static_cast<const void *>(validationBinnedData), 
      static_cast<const void *>(validationTargets), 
      static_cast<const void *>(validationPredictorScores), 
      countInnerBags, 
      static_cast<const void *>(optionalTempParams)
      );
   if(countTargetClasses < 0) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification countTargetClasses can't be negative");
      return nullptr;
   }
   if(0 == countTargetClasses && (0 != countTrainingSamples || 0 != countValidationSamples)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification countTargetClasses can't be zero unless there are no training and no validation cases");
      return nullptr;
   }
   if(!IsNumberConvertable<ptrdiff_t>(countTargetClasses)) {
      LOG_0(TraceLevelWarning, "WARNING InitializeBoostingClassification !IsNumberConvertable<ptrdiff_t>(countTargetClasses)");
      return nullptr;
   }
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = static_cast<ptrdiff_t>(countTargetClasses);
   const PEbmBoosting pEbmBoosting = reinterpret_cast<PEbmBoosting>(AllocateBoosting(
      randomSeed, 
      countFeatures, 
      features, 
      countFeatureGroups, 
      featureGroups, 
      featureGroupIndexes, 
      runtimeLearningTypeOrCountTargetClasses, 
      countTrainingSamples, 
      trainingTargets, 
      trainingBinnedData, 
      trainingPredictorScores, 
      countValidationSamples, 
      validationTargets, 
      validationBinnedData, 
      validationPredictorScores, 
      countInnerBags,
      optionalTempParams
   ));
   LOG_N(TraceLevelInfo, "Exited InitializeBoostingClassification %p", static_cast<void *>(pEbmBoosting));
   return pEbmBoosting;
}

EBM_NATIVE_IMPORT_EXPORT_BODY PEbmBoosting EBM_NATIVE_CALLING_CONVENTION InitializeBoostingRegression(
   SeedEbmType randomSeed,
   IntEbmType countFeatures,
   const EbmNativeFeature * features,
   IntEbmType countFeatureGroups,
   const EbmNativeFeatureGroup * featureGroups,
   const IntEbmType * featureGroupIndexes,
   IntEbmType countTrainingSamples,
   const IntEbmType * trainingBinnedData,
   const FloatEbmType * trainingTargets,
   const FloatEbmType * trainingPredictorScores,
   IntEbmType countValidationSamples,
   const IntEbmType * validationBinnedData,
   const FloatEbmType * validationTargets,
   const FloatEbmType * validationPredictorScores,
   IntEbmType countInnerBags,
   const FloatEbmType * optionalTempParams
) {
   LOG_N(
      TraceLevelInfo, 
      "Entered InitializeBoostingRegression: "
      "randomSeed=%" SeedEbmTypePrintf ", "
      "countFeatures=%" IntEbmTypePrintf ", "
      "features=%p, "
      "countFeatureGroups=%" IntEbmTypePrintf ", "
      "featureGroups=%p, "
      "featureGroupIndexes=%p, "
      "countTrainingSamples=%" IntEbmTypePrintf ", "
      "trainingBinnedData=%p, "
      "trainingTargets=%p, "
      "trainingPredictorScores=%p, "
      "countValidationSamples=%" IntEbmTypePrintf ", "
      "validationBinnedData=%p, "
      "validationTargets=%p, "
      "validationPredictorScores=%p, "
      "countInnerBags=%" IntEbmTypePrintf ", "
      "optionalTempParams=%p"
      ,
      randomSeed,
      countFeatures,
      static_cast<const void *>(features), 
      countFeatureGroups, 
      static_cast<const void *>(featureGroups), 
      static_cast<const void *>(featureGroupIndexes), 
      countTrainingSamples, 
      static_cast<const void *>(trainingBinnedData), 
      static_cast<const void *>(trainingTargets), 
      static_cast<const void *>(trainingPredictorScores), 
      countValidationSamples, 
      static_cast<const void *>(validationBinnedData), 
      static_cast<const void *>(validationTargets), 
      static_cast<const void *>(validationPredictorScores), 
      countInnerBags, 
      static_cast<const void *>(optionalTempParams)
   );
   const PEbmBoosting pEbmBoosting = reinterpret_cast<PEbmBoosting>(AllocateBoosting(
      randomSeed, 
      countFeatures, 
      features, 
      countFeatureGroups, 
      featureGroups, 
      featureGroupIndexes, 
      k_regression, 
      countTrainingSamples, 
      trainingTargets, 
      trainingBinnedData, 
      trainingPredictorScores, 
      countValidationSamples, 
      validationTargets, 
      validationBinnedData, 
      validationPredictorScores, 
      countInnerBags,
      optionalTempParams
   ));
   LOG_N(TraceLevelInfo, "Exited InitializeBoostingRegression %p", static_cast<void *>(pEbmBoosting));
   return pEbmBoosting;
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION BoostingStep(
   PEbmBoosting ebmBoosting,
   IntEbmType indexFeatureGroup,
   GenerateUpdateOptionsType options,
   FloatEbmType learningRate,
   IntEbmType countSamplesRequiredForChildSplitMin,
   const IntEbmType * leavesMax,
   const FloatEbmType * trainingWeights,
   const FloatEbmType * validationWeights,
   FloatEbmType * validationMetricOut
) {
   EbmBoostingState * pEbmBoostingState = reinterpret_cast<EbmBoostingState *>(ebmBoosting);
   if(nullptr == pEbmBoostingState) {
      LOG_0(TraceLevelError, "ERROR BoostingStep ebmBoosting cannot be nullptr");
      return 1;
   }

   if(IsClassification(pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses())) {
      // we need to special handle this case because if we call GenerateModelUpdate, we'll get back a nullptr for the model (since there is no model) 
      // and we'll return 1 from this function.  We'd like to return 0 (success) here, so we handle it ourselves
      if(pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses() <= ptrdiff_t { 1 }) {
         // if there is only 1 target class for classification, then we can predict the output with 100% accuracy.  The model is a tensor with zero 
         // length array logits, which means for our representation that we have zero items in the array total.
         // since we can predit the output with 100% accuracy, our gain will be 0.
         if(nullptr != validationMetricOut) {
            *validationMetricOut = FloatEbmType { 0 };
         }
         LOG_0(TraceLevelWarning, "WARNING BoostingStep pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses <= ptrdiff_t { 1 }");
         return 0;
      }
   }

   FloatEbmType gain; // we toss this value, but we still need to get it
   FloatEbmType * pModelFeatureGroupUpdateTensor = GenerateModelFeatureGroupUpdate(
      ebmBoosting, 
      indexFeatureGroup, 
      options,
      learningRate,
      countSamplesRequiredForChildSplitMin, 
      leavesMax, 
      trainingWeights, 
      validationWeights, 
      &gain
   );
   if(nullptr == pModelFeatureGroupUpdateTensor) {
      // if we get back a nullptr from GenerateModelFeatureGroupUpdate it either means that there's only
      // 1 class in our classification problem, or it means we encountered an error.  We assume here that
      // it was an error since the caller can check ahead of time if there was only 1 class before calling us
      if(nullptr != validationMetricOut) {
         *validationMetricOut = FloatEbmType { 0 };
      }
      return 1;
   }
   return ApplyModelFeatureGroupUpdate(ebmBoosting, indexFeatureGroup, pModelFeatureGroupUpdateTensor, validationMetricOut);
}

EBM_NATIVE_IMPORT_EXPORT_BODY FloatEbmType * EBM_NATIVE_CALLING_CONVENTION GetBestModelFeatureGroup(
   PEbmBoosting ebmBoosting,
   IntEbmType indexFeatureGroup
) {
   LOG_N(
      TraceLevelInfo, 
      "Entered GetBestModelFeatureGroup: ebmBoosting=%p, indexFeatureGroup=%" IntEbmTypePrintf, 
      static_cast<void *>(ebmBoosting), 
      indexFeatureGroup
   );

   EbmBoostingState * pEbmBoostingState = reinterpret_cast<EbmBoostingState *>(ebmBoosting);
   if(nullptr == pEbmBoostingState) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup ebmBoosting cannot be nullptr");
      return nullptr;
   }
   if(indexFeatureGroup < 0) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup indexFeatureGroup must be positive");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t>(indexFeatureGroup)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup indexFeatureGroup is too high to index");
      return nullptr;
   }
   size_t iFeatureGroup = static_cast<size_t>(indexFeatureGroup);
   if(pEbmBoostingState->GetCountFeatureGroups() <= iFeatureGroup) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup indexFeatureGroup above the number of feature groups that we have");
      return nullptr;
   }
   if(nullptr == pEbmBoostingState->GetBestModel()) {
      // if pEbmBoostingState->m_apBestModel is nullptr, then either:
      //    1) m_cFeatureGroups was 0, in which case this function would have undefined behavior since the caller needs to indicate a valid 
      //       indexFeatureGroup, which is impossible, so we can do anything we like, include the below actions.
      //    2) m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0 (and the learning type is classification), 
      //       which is legal, which we need to handle here
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.  If there were logits in this model, 
      // they'd all be infinity, but you could alternatively think of this model as having zero logits, since the number of logits can be one 
      // less than the number of target classification classes.  A model with zero logits is empty, and has zero items.  We want to return a tensor 
      // with 0 items in it, so we could either return a pointer to some random memory that can't be accessed, or we can return nullptr.  
      // We return a nullptr in the hopes that our caller will either handle it or throw a nicer exception.

      LOG_0(TraceLevelInfo, "Exited GetBestModelFeatureGroup no model");
      return nullptr;
   }

   SegmentedTensor * pBestModel = pEbmBoostingState->GetBestModel()[iFeatureGroup];
   EBM_ASSERT(nullptr != pBestModel);
   EBM_ASSERT(pBestModel->GetExpanded()); // the model should have been expanded at startup
   FloatEbmType * pRet = pBestModel->GetValuePointer();
   EBM_ASSERT(nullptr != pRet);

   LOG_N(TraceLevelInfo, "Exited GetBestModelFeatureGroup %p", static_cast<void *>(pRet));
   return pRet;
}

EBM_NATIVE_IMPORT_EXPORT_BODY FloatEbmType * EBM_NATIVE_CALLING_CONVENTION GetCurrentModelFeatureGroup(
   PEbmBoosting ebmBoosting,
   IntEbmType indexFeatureGroup
) {
   LOG_N(
      TraceLevelInfo, 
      "Entered GetCurrentModelFeatureGroup: ebmBoosting=%p, indexFeatureGroup=%" IntEbmTypePrintf, 
      static_cast<void *>(ebmBoosting), 
      indexFeatureGroup
   );

   EbmBoostingState * pEbmBoostingState = reinterpret_cast<EbmBoostingState *>(ebmBoosting);
   if(nullptr == pEbmBoostingState) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup ebmBoosting cannot be nullptr");
      return nullptr;
   }
   if(indexFeatureGroup < 0) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup indexFeatureGroup must be positive");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t>(indexFeatureGroup)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup indexFeatureGroup is too high to index");
      return nullptr;
   }
   size_t iFeatureGroup = static_cast<size_t>(indexFeatureGroup);
   if(pEbmBoostingState->GetCountFeatureGroups() <= iFeatureGroup) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup indexFeatureGroup above the number of feature groups that we have");
      return nullptr;
   }
   if(nullptr == pEbmBoostingState->GetCurrentModel()) {
      // if pEbmBoostingState->m_apCurrentModel is nullptr, then either:
      //    1) m_cFeatureGroups was 0, in which case this function would have undefined behavior since the caller needs to indicate a valid 
      //       indexFeatureGroup, which is impossible, so we can do anything we like, include the below actions.
      //    2) m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0 (and the learning type is classification), which is legal, 
      //       which we need to handle here
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.  If there were logits 
      // in this model, they'd all be infinity, but you could alternatively think of this model as having zero logits, since the number of 
      // logits can be one less than the number of target classification classes.  A model with zero logits is empty, and has zero items.  
      // We want to return a tensor with 0 items in it, so we could either return a pointer to some random memory that can't be accessed, 
      // or we can return nullptr.  We return a nullptr in the hopes that our caller will either handle it or throw a nicer exception.

      LOG_0(TraceLevelInfo, "Exited GetCurrentModelFeatureGroup no model");
      return nullptr;
   }

   SegmentedTensor * pCurrentModel = pEbmBoostingState->GetCurrentModel()[iFeatureGroup];
   EBM_ASSERT(nullptr != pCurrentModel);
   EBM_ASSERT(pCurrentModel->GetExpanded()); // the model should have been expanded at startup
   FloatEbmType * pRet = pCurrentModel->GetValuePointer();
   EBM_ASSERT(nullptr != pRet);

   LOG_N(TraceLevelInfo, "Exited GetCurrentModelFeatureGroup %p", static_cast<void *>(pRet));
   return pRet;
}

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION FreeBoosting(
   PEbmBoosting ebmBoosting
) {
   LOG_N(TraceLevelInfo, "Entered FreeBoosting: ebmBoosting=%p", static_cast<void *>(ebmBoosting));

   EbmBoostingState * pEbmBoostingState = reinterpret_cast<EbmBoostingState *>(ebmBoosting);

   // it's legal to call free on nullptr, just like for free().  This is checked inside EbmBoostingState::Free()
   EbmBoostingState::Free(pEbmBoostingState);

   LOG_0(TraceLevelInfo, "Exited FreeBoosting");
}
