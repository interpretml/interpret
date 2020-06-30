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
// FeatureCombination.h depends on FeatureInternal.h
#include "FeatureGroup.h"
// dataset depends on features
#include "DataSetBoosting.h"
// samples is somewhat independent from datasets, but relies on an indirect coupling with them
#include "SamplingSet.h"
#include "TreeSweep.h"

#include "Booster.h"

extern void InitializeResiduals(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const size_t cInstances,
   const void * const aTargetData,
   const FloatEbmType * const aPredictorScores,
   FloatEbmType * const aTempFloatVector,
   FloatEbmType * pResidualError
);

EBM_INLINE static size_t GetCountItemsBitPacked(const size_t cBits) {
   EBM_ASSERT(size_t { 1 } <= cBits);
   return k_cBitsForStorageType / cBits;
}

void EbmBoostingState::DeleteSegmentedTensors(const size_t cFeatureCombinations, SegmentedTensor ** const apSegmentedTensors) {
   LOG_0(TraceLevelInfo, "Entered DeleteSegmentedTensors");

   if(UNLIKELY(nullptr != apSegmentedTensors)) {
      EBM_ASSERT(0 < cFeatureCombinations);
      SegmentedTensor ** ppSegmentedTensors = apSegmentedTensors;
      const SegmentedTensor * const * const ppSegmentedTensorsEnd = &apSegmentedTensors[cFeatureCombinations];
      do {
         SegmentedTensor::Free(*ppSegmentedTensors);
         ++ppSegmentedTensors;
      } while(ppSegmentedTensorsEnd != ppSegmentedTensors);
      free(apSegmentedTensors);
   }
   LOG_0(TraceLevelInfo, "Exited DeleteSegmentedTensors");
}

SegmentedTensor ** EbmBoostingState::InitializeSegmentedTensors(
   const size_t cFeatureCombinations, 
   const FeatureCombination * const * const apFeatureCombinations, 
   const size_t cVectorLength) 
{
   LOG_0(TraceLevelInfo, "Entered InitializeSegmentedTensors");

   EBM_ASSERT(0 < cFeatureCombinations);
   EBM_ASSERT(nullptr != apFeatureCombinations);
   EBM_ASSERT(1 <= cVectorLength);

   SegmentedTensor ** const apSegmentedTensors = EbmMalloc<SegmentedTensor *>(cFeatureCombinations);
   if(UNLIKELY(nullptr == apSegmentedTensors)) {
      LOG_0(TraceLevelWarning, "WARNING InitializeSegmentedTensors nullptr == apSegmentedTensors");
      return nullptr;
   }
   for(size_t i = 0; i < cFeatureCombinations; ++i) {
      apSegmentedTensors[i] = nullptr;
   }

   SegmentedTensor ** ppSegmentedTensors = apSegmentedTensors;
   for(size_t iFeatureCombination = 0; iFeatureCombination < cFeatureCombinations; ++iFeatureCombination) {
      const FeatureCombination * const pFeatureCombination = apFeatureCombinations[iFeatureCombination];
      SegmentedTensor * const pSegmentedTensors = 
         SegmentedTensor::Allocate(pFeatureCombination->GetCountFeatures(), cVectorLength);
      if(UNLIKELY(nullptr == pSegmentedTensors)) {
         LOG_0(TraceLevelWarning, "WARNING InitializeSegmentedTensors nullptr == pSegmentedTensors");
         DeleteSegmentedTensors(cFeatureCombinations, apSegmentedTensors);
         return nullptr;
      }

      if(0 == pFeatureCombination->GetCountFeatures()) {
         // if there are zero dimensions, then we have a tensor with 1 item, and we're already expanded
         pSegmentedTensors->SetExpanded();
      } else {
         // if our segmented region has no dimensions, then it's already a fully expanded with 1 bin

         // TODO optimize the next few lines
         // TODO there might be a nicer way to expand this at allocation time (fill with zeros is easier)
         // we want to return a pointer to our interior state in the GetCurrentModelFeatureCombination and GetBestModelFeatureCombination functions.  
         // For simplicity we don't transmit the divions, so we need to expand our SegmentedRegion before returning the easiest way to ensure that the 
         // SegmentedRegion is expanded is to start it off expanded, and then we don't have to check later since anything merged into an expanded 
         // SegmentedRegion will itself be expanded
         size_t acDivisionIntegersEnd[k_cDimensionsMax];
         size_t iDimension = 0;
         do {
            acDivisionIntegersEnd[iDimension] = pFeatureCombination->GetFeatureCombinationEntries()[iDimension].m_pFeature->GetCountBins();
            ++iDimension;
         } while(iDimension < pFeatureCombination->GetCountFeatures());

         if(pSegmentedTensors->Expand(acDivisionIntegersEnd)) {
            LOG_0(TraceLevelWarning, "WARNING InitializeSegmentedTensors pSegmentedTensors->Expand(acDivisionIntegersEnd)");
            DeleteSegmentedTensors(cFeatureCombinations, apSegmentedTensors);
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

      FeatureCombination::FreeFeatureCombinations(pBoostingState->m_cFeatureCombinations, pBoostingState->m_apFeatureCombinations);

      free(pBoostingState->m_aFeatures);

      DeleteSegmentedTensors(pBoostingState->m_cFeatureCombinations, pBoostingState->m_apCurrentModel);
      DeleteSegmentedTensors(pBoostingState->m_cFeatureCombinations, pBoostingState->m_apBestModel);
      SegmentedTensor::Free(pBoostingState->m_pSmallChangeToModelOverwriteSingleSamplingSet);
      SegmentedTensor::Free(pBoostingState->m_pSmallChangeToModelAccumulatedFromSamplingSets);

      free(pBoostingState);
   }
   LOG_0(TraceLevelInfo, "Exited EbmBoostingState::Free");
}

EbmBoostingState * EbmBoostingState::Allocate(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const size_t cFeatures,
   const size_t cFeatureCombinations,
   const size_t cSamplingSets,
   const FloatEbmType * const optionalTempParams,
   const EbmNativeFeature * const aFeatures,
   const EbmNativeFeatureCombination * const aFeatureCombinations, 
   const IntEbmType * featureCombinationIndexes, 
   const size_t cTrainingInstances, 
   const void * const aTrainingTargets, 
   const IntEbmType * const aTrainingBinnedData, 
   const FloatEbmType * const aTrainingPredictorScores, 
   const size_t cValidationInstances, 
   const void * const aValidationTargets, 
   const IntEbmType * const aValidationBinnedData, 
   const FloatEbmType * const aValidationPredictorScores,
   const IntEbmType randomSeed
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
         if(0 == countBins && (0 != cTrainingInstances || 0 != cValidationInstances)) {
            LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize countBins cannot be zero if either 0 < cTrainingInstances OR 0 < cValidationInstances");
            EbmBoostingState::Free(pBooster);
            return nullptr;
         }
         if(!IsNumberConvertable<size_t, IntEbmType>(countBins)) {
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
         bool bMissing = EBM_FALSE != pFeatureInitialize->hasMissing;

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

   LOG_0(TraceLevelInfo, "EbmBoostingState::Initialize starting feature combination processing");
   if(0 != cFeatureCombinations) {
      pBooster->m_cFeatureCombinations = cFeatureCombinations;
      pBooster->m_apFeatureCombinations = FeatureCombination::AllocateFeatureCombinations(cFeatureCombinations);
      if(UNLIKELY(nullptr == pBooster->m_apFeatureCombinations)) {
         LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize 0 != m_cFeatureCombinations && nullptr == m_apFeatureCombinations");
         EbmBoostingState::Free(pBooster);
         return nullptr;
      }

      if(GetSweepTreeNodeSizeOverflow(bClassification, cVectorLength)) {
         LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize GetSweepTreeNodeSizeOverflow(bClassification, cVectorLength)");
         EbmBoostingState::Free(pBooster);
         return nullptr;
      }
      size_t cBytesPerSweepTreeNode = GetSweepTreeNodeSize(bClassification, cVectorLength);

      const IntEbmType * pFeatureCombinationIndex = featureCombinationIndexes;
      size_t iFeatureCombination = 0;
      do {
         const EbmNativeFeatureCombination * const pFeatureCombinationInterop = &aFeatureCombinations[iFeatureCombination];
         const IntEbmType countFeaturesInCombination = pFeatureCombinationInterop->countFeaturesInCombination;
         if(countFeaturesInCombination < 0) {
            LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize countFeaturesInCombination cannot be negative");
            EbmBoostingState::Free(pBooster);
            return nullptr;
         }
         if(!IsNumberConvertable<size_t, IntEbmType>(countFeaturesInCombination)) {
            // if countFeaturesInCombination exceeds the size of size_t, then we wouldn't be able to find it
            // in the array passed to us
            LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize countFeaturesInCombination is too high to index");
            EbmBoostingState::Free(pBooster);
            return nullptr;
         }
         size_t cFeaturesInCombination = static_cast<size_t>(countFeaturesInCombination);
         size_t cSignificantFeaturesInCombination = 0;
         const IntEbmType * pFeatureCombinationIndexEnd = pFeatureCombinationIndex;
         if(UNLIKELY(0 == cFeaturesInCombination)) {
            LOG_0(TraceLevelInfo, "INFO EbmBoostingState::Initialize empty feature combination");
         } else {
            if(nullptr == pFeatureCombinationIndex) {
               LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize featureCombinationIndexes is null when there are FeatureCombinations with non-zero numbers of features");
               EbmBoostingState::Free(pBooster);
               return nullptr;
            }
            pFeatureCombinationIndexEnd += cFeaturesInCombination;
            const IntEbmType * pFeatureCombinationIndexTemp = pFeatureCombinationIndex;
            do {
               const IntEbmType indexFeatureInterop = *pFeatureCombinationIndexTemp;
               if(indexFeatureInterop < 0) {
                  LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize featureCombinationIndexes value cannot be negative");
                  EbmBoostingState::Free(pBooster);
                  return nullptr;
               }
               if(!IsNumberConvertable<size_t, IntEbmType>(indexFeatureInterop)) {
                  LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize featureCombinationIndexes value too big to reference memory");
                  EbmBoostingState::Free(pBooster);
                  return nullptr;
               }
               const size_t iFeatureForCombination = static_cast<size_t>(indexFeatureInterop);

               if(cFeatures <= iFeatureForCombination) {
                  LOG_0(TraceLevelError, "ERROR EbmBoostingState::Initialize featureCombinationIndexes value must be less than the number of features");
                  EbmBoostingState::Free(pBooster);
                  return nullptr;
               }

               EBM_ASSERT(1 <= cFeatures);
               EBM_ASSERT(nullptr != pBooster->m_aFeatures);

               Feature * const pInputFeature = &pBooster->m_aFeatures[iFeatureForCombination];
               if(LIKELY(1 < pInputFeature->GetCountBins())) {
                  // if we have only 1 bin, then we can eliminate the feature from consideration since the resulting tensor loses one dimension but is 
                  // otherwise indistinquishable from the original data
                  ++cSignificantFeaturesInCombination;
               } else {
                  LOG_0(TraceLevelInfo, "INFO EbmBoostingState::Initialize feature combination with no useful features");
               }
               ++pFeatureCombinationIndexTemp;
            } while(pFeatureCombinationIndexEnd != pFeatureCombinationIndexTemp);

            if(k_cDimensionsMax < cSignificantFeaturesInCombination) {
               // if we try to run with more than k_cDimensionsMax we'll exceed our memory capacity, so let's exit here instead
               LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize k_cDimensionsMax < cSignificantFeaturesInCombination");
               EbmBoostingState::Free(pBooster);
               return nullptr;
            }
         }

         FeatureCombination * pFeatureCombination = FeatureCombination::Allocate(cSignificantFeaturesInCombination, iFeatureCombination);
         if(nullptr == pFeatureCombination) {
            LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == pFeatureCombination");
            EbmBoostingState::Free(pBooster);
            return nullptr;
         }
         // assign our pointer directly to our array right now so that we can't loose the memory if we decide to exit due to an error below
         pBooster->m_apFeatureCombinations[iFeatureCombination] = pFeatureCombination;

         if(UNLIKELY(0 != cSignificantFeaturesInCombination)) {
            EBM_ASSERT(nullptr != featureCombinationIndexes);
            size_t cEquivalentSplits = 1;
            size_t cTensorBins = 1;
            FeatureCombinationEntry * pFeatureCombinationEntry = pFeatureCombination->GetFeatureCombinationEntries();
            do {
               const IntEbmType indexFeatureInterop = *pFeatureCombinationIndex;
               EBM_ASSERT(0 <= indexFeatureInterop);
               EBM_ASSERT((IsNumberConvertable<size_t, IntEbmType>(indexFeatureInterop))); // this was checked above
               const size_t iFeatureForCombination = static_cast<size_t>(indexFeatureInterop);
               EBM_ASSERT(iFeatureForCombination < cFeatures);
               const Feature * const pInputFeature = &pBooster->m_aFeatures[iFeatureForCombination];
               const size_t cBins = pInputFeature->GetCountBins();
               if(LIKELY(1 < cBins)) {
                  // if we have only 1 bin, then we can eliminate the feature from consideration since the resulting tensor loses one dimension but is 
                  // otherwise indistinquishable from the original data
                  pFeatureCombinationEntry->m_pFeature = pInputFeature;
                  ++pFeatureCombinationEntry;
                  if(IsMultiplyError(cTensorBins, cBins)) {
                     // if this overflows, we definetly won't be able to allocate it
                     LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize IsMultiplyError(cTensorStates, cBins)");
                     EbmBoostingState::Free(pBooster);
                     return nullptr;
                  }
                  cTensorBins *= cBins;
                  cEquivalentSplits *= cBins - 1; // we can only split between the bins
               }
               ++pFeatureCombinationIndex;
            } while(pFeatureCombinationIndexEnd != pFeatureCombinationIndex);
            EBM_ASSERT(1 < cTensorBins);

            size_t cBytesArrayEquivalentSplit;
            if(1 == cSignificantFeaturesInCombination) {
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

            // if cSignificantFeaturesInCombination is zero, don't both initializing pFeatureCombination->GetCountItemsPerBitPackedDataUnit()
            const size_t cBitsRequiredMin = CountBitsRequired(cTensorBins - 1);
            EBM_ASSERT(1 <= cBitsRequiredMin); // 1 < cTensorBins otherwise we'd have filtered it out above
            pFeatureCombination->SetCountItemsPerBitPackedDataUnit(GetCountItemsBitPacked(cBitsRequiredMin));
         }
         pFeatureCombinationIndex = pFeatureCombinationIndexEnd;

         ++iFeatureCombination;
      } while(iFeatureCombination < cFeatureCombinations);

      if(!bClassification || ptrdiff_t { 2 } <= runtimeLearningTypeOrCountTargetClasses) {
         pBooster->m_apCurrentModel = InitializeSegmentedTensors(cFeatureCombinations, pBooster->m_apFeatureCombinations, cVectorLength);
         if(nullptr == pBooster->m_apCurrentModel) {
            LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == m_apCurrentModel");
            EbmBoostingState::Free(pBooster);
            return nullptr;
         }
         pBooster->m_apBestModel = InitializeSegmentedTensors(cFeatureCombinations, pBooster->m_apFeatureCombinations, cVectorLength);
         if(nullptr == pBooster->m_apBestModel) {
            LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == m_apBestModel");
            EbmBoostingState::Free(pBooster);
            return nullptr;
         }
      }
   }
   LOG_0(TraceLevelInfo, "EbmBoostingState::Initialize finished feature combination processing");

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
      cFeatureCombinations, 
      pBooster->m_apFeatureCombinations,
      cTrainingInstances, 
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
      cFeatureCombinations, 
      pBooster->m_apFeatureCombinations,
      cValidationInstances, 
      aValidationBinnedData, 
      aValidationTargets, 
      aValidationPredictorScores, 
      runtimeLearningTypeOrCountTargetClasses
   )) {
      LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize m_validationSet.Initialize");
      EbmBoostingState::Free(pBooster);
      return nullptr;
   }

   pBooster->m_randomStream.Initialize(randomSeed);

   EBM_ASSERT(nullptr == pBooster->m_apSamplingSets);
   if(0 != cTrainingInstances) {
      pBooster->m_cSamplingSets = cSamplingSets;
      pBooster->m_apSamplingSets = SamplingSet::GenerateSamplingSets(&pBooster->m_randomStream, &pBooster->m_trainingSet, cSamplingSets);
      if(UNLIKELY(nullptr == pBooster->m_apSamplingSets)) {
         LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == m_apSamplingSets");
         EbmBoostingState::Free(pBooster);
         return nullptr;
      }
   }

   if(bClassification) {
      if(0 != cTrainingInstances) {
         InitializeResiduals(
            runtimeLearningTypeOrCountTargetClasses,
            cTrainingInstances,
            aTrainingTargets,
            aTrainingPredictorScores,
            pBooster->GetCachedThreadResources()->GetTempFloatVector(),
            pBooster->m_trainingSet.GetResidualPointer()
         );
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      if(0 != cTrainingInstances) {
         InitializeResiduals(
            k_regression,
            cTrainingInstances,
            aTrainingTargets,
            aTrainingPredictorScores,
            nullptr,
            pBooster->m_trainingSet.GetResidualPointer()
         );
      }
      if(0 != cValidationInstances) {
         InitializeResiduals(
            k_regression,
            cValidationInstances,
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
   const IntEbmType randomSeed, 
   const IntEbmType countFeatures, 
   const EbmNativeFeature * const features, 
   const IntEbmType countFeatureCombinations, 
   const EbmNativeFeatureCombination * const featureCombinations, 
   const IntEbmType * const featureCombinationIndexes, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, 
   const IntEbmType countTrainingInstances, 
   const void * const trainingTargets, 
   const IntEbmType * const trainingBinnedData, 
   const FloatEbmType * const trainingPredictorScores, 
   const IntEbmType countValidationInstances, 
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
   if(countFeatureCombinations < 0) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting countFeatureCombinations must be positive");
      return nullptr;
   }
   if(0 != countFeatureCombinations && nullptr == featureCombinations) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting featureCombinations cannot be nullptr if 0 < countFeatureCombinations");
      return nullptr;
   }
   // featureCombinationIndexes -> it's legal for featureCombinationIndexes to be nullptr if there are no features indexed by our featureCombinations.  
   // FeatureCombinations can have zero features, so it could be legal for this to be null even if there are featureCombinations
   if(countTrainingInstances < 0) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting countTrainingInstances must be positive");
      return nullptr;
   }
   if(0 != countTrainingInstances && nullptr == trainingTargets) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting trainingTargets cannot be nullptr if 0 < countTrainingInstances");
      return nullptr;
   }
   if(0 != countTrainingInstances && 0 != countFeatures && nullptr == trainingBinnedData) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting trainingBinnedData cannot be nullptr if 0 < countTrainingInstances AND 0 < countFeatures");
      return nullptr;
   }
   if(0 != countTrainingInstances && nullptr == trainingPredictorScores) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting trainingPredictorScores cannot be nullptr if 0 < countTrainingInstances");
      return nullptr;
   }
   if(countValidationInstances < 0) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting countValidationInstances must be positive");
      return nullptr;
   }
   if(0 != countValidationInstances && nullptr == validationTargets) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting validationTargets cannot be nullptr if 0 < countValidationInstances");
      return nullptr;
   }
   if(0 != countValidationInstances && 0 != countFeatures && nullptr == validationBinnedData) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting validationBinnedData cannot be nullptr if 0 < countValidationInstances AND 0 < countFeatures");
      return nullptr;
   }
   if(0 != countValidationInstances && nullptr == validationPredictorScores) {
      LOG_0(TraceLevelError, "ERROR AllocateBoosting validationPredictorScores cannot be nullptr if 0 < countValidationInstances");
      return nullptr;
   }
   if(countInnerBags < 0) {
      // 0 means use the full set (good value).  1 means make a single bag (this is useless but allowed for comparison purposes).  2+ are good numbers of bag
      LOG_0(TraceLevelError, "ERROR AllocateBoosting countInnerBags must be positive");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntEbmType>(countFeatures)) {
      // the caller should not have been able to allocate enough memory in "features" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR AllocateBoosting !IsNumberConvertable<size_t, IntEbmType>(countFeatures)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntEbmType>(countFeatureCombinations)) {
      // the caller should not have been able to allocate enough memory in "featureCombinations" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR AllocateBoosting !IsNumberConvertable<size_t, IntEbmType>(countFeatureCombinations)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntEbmType>(countTrainingInstances)) {
      // the caller should not have been able to allocate enough memory in "trainingTargets" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR AllocateBoosting !IsNumberConvertable<size_t, IntEbmType>(countTrainingInstances)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntEbmType>(countValidationInstances)) {
      // the caller should not have been able to allocate enough memory in "validationTargets" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR AllocateBoosting !IsNumberConvertable<size_t, IntEbmType>(countValidationInstances)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntEbmType>(countInnerBags)) {
      // this is just a warning since the caller doesn't pass us anything material, but if it's this high
      // then our allocation would fail since it can't even in pricipal fit into memory
      LOG_0(TraceLevelWarning, "WARNING AllocateBoosting !IsNumberConvertable<size_t, IntEbmType>(countInnerBags)");
      return nullptr;
   }

   size_t cFeatures = static_cast<size_t>(countFeatures);
   size_t cFeatureCombinations = static_cast<size_t>(countFeatureCombinations);
   size_t cTrainingInstances = static_cast<size_t>(countTrainingInstances);
   size_t cValidationInstances = static_cast<size_t>(countValidationInstances);
   size_t cInnerBags = static_cast<size_t>(countInnerBags);

   size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   if(IsMultiplyError(cVectorLength, cTrainingInstances)) {
      // the caller should not have been able to allocate enough memory in "trainingPredictorScores" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR AllocateBoosting IsMultiplyError(cVectorLength, cTrainingInstances)");
      return nullptr;
   }
   if(IsMultiplyError(cVectorLength, cValidationInstances)) {
      // the caller should not have been able to allocate enough memory in "validationPredictorScores" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR AllocateBoosting IsMultiplyError(cVectorLength, cValidationInstances)");
      return nullptr;
   }

   EbmBoostingState * const pEbmBoostingState = EbmBoostingState::Allocate(
      runtimeLearningTypeOrCountTargetClasses,
      cFeatures,
      cFeatureCombinations,
      cInnerBags,
      optionalTempParams,
      features,
      featureCombinations,
      featureCombinationIndexes,
      cTrainingInstances,
      trainingTargets,
      trainingBinnedData,
      trainingPredictorScores,
      cValidationInstances,
      validationTargets,
      validationBinnedData,
      validationPredictorScores,
      randomSeed
   );
   if(UNLIKELY(nullptr == pEbmBoostingState)) {
      LOG_0(TraceLevelWarning, "WARNING AllocateBoosting pEbmBoostingState->Initialize");
      return nullptr;
   }
   return pEbmBoostingState;
}

EBM_NATIVE_IMPORT_EXPORT_BODY PEbmBoosting EBM_NATIVE_CALLING_CONVENTION InitializeBoostingClassification(
   IntEbmType countTargetClasses,
   IntEbmType countFeatures,
   const EbmNativeFeature * features,
   IntEbmType countFeatureCombinations,
   const EbmNativeFeatureCombination * featureCombinations,
   const IntEbmType * featureCombinationIndexes,
   IntEbmType countTrainingInstances,
   const IntEbmType * trainingBinnedData,
   const IntEbmType * trainingTargets,
   const FloatEbmType * trainingPredictorScores,
   IntEbmType countValidationInstances,
   const IntEbmType * validationBinnedData,
   const IntEbmType * validationTargets,
   const FloatEbmType * validationPredictorScores,
   IntEbmType countInnerBags,
   IntEbmType randomSeed,
   const FloatEbmType * optionalTempParams
) {
   LOG_N(TraceLevelInfo, "Entered InitializeBoostingClassification: countTargetClasses=%" IntEbmTypePrintf ", countFeatures=%" IntEbmTypePrintf 
      ", features=%p, countFeatureCombinations=%" IntEbmTypePrintf ", featureCombinations=%p, featureCombinationIndexes=%p, countTrainingInstances=%" 
      IntEbmTypePrintf ", trainingBinnedData=%p, trainingTargets=%p, trainingPredictorScores=%p, countValidationInstances=%" 
      IntEbmTypePrintf ", validationBinnedData=%p, validationTargets=%p, validationPredictorScores=%p, countInnerBags=%" 
      IntEbmTypePrintf ", randomSeed=%" IntEbmTypePrintf ", optionalTempParams=%p",
      countTargetClasses, 
      countFeatures, 
      static_cast<const void *>(features), 
      countFeatureCombinations, 
      static_cast<const void *>(featureCombinations), 
      static_cast<const void *>(featureCombinationIndexes), 
      countTrainingInstances, 
      static_cast<const void *>(trainingBinnedData), 
      static_cast<const void *>(trainingTargets), 
      static_cast<const void *>(trainingPredictorScores), 
      countValidationInstances, 
      static_cast<const void *>(validationBinnedData), 
      static_cast<const void *>(validationTargets), 
      static_cast<const void *>(validationPredictorScores), 
      countInnerBags, 
      randomSeed,
      static_cast<const void *>(optionalTempParams)
      );
   if(countTargetClasses < 0) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification countTargetClasses can't be negative");
      return nullptr;
   }
   if(0 == countTargetClasses && (0 != countTrainingInstances || 0 != countValidationInstances)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification countTargetClasses can't be zero unless there are no training and no validation cases");
      return nullptr;
   }
   if(!IsNumberConvertable<ptrdiff_t, IntEbmType>(countTargetClasses)) {
      LOG_0(TraceLevelWarning, "WARNING InitializeBoostingClassification !IsNumberConvertable<ptrdiff_t, IntEbmType>(countTargetClasses)");
      return nullptr;
   }
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = static_cast<ptrdiff_t>(countTargetClasses);
   const PEbmBoosting pEbmBoosting = reinterpret_cast<PEbmBoosting>(AllocateBoosting(
      randomSeed, 
      countFeatures, 
      features, 
      countFeatureCombinations, 
      featureCombinations, 
      featureCombinationIndexes, 
      runtimeLearningTypeOrCountTargetClasses, 
      countTrainingInstances, 
      trainingTargets, 
      trainingBinnedData, 
      trainingPredictorScores, 
      countValidationInstances, 
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
   IntEbmType countFeatures,
   const EbmNativeFeature * features,
   IntEbmType countFeatureCombinations,
   const EbmNativeFeatureCombination * featureCombinations,
   const IntEbmType * featureCombinationIndexes,
   IntEbmType countTrainingInstances,
   const IntEbmType * trainingBinnedData,
   const FloatEbmType * trainingTargets,
   const FloatEbmType * trainingPredictorScores,
   IntEbmType countValidationInstances,
   const IntEbmType * validationBinnedData,
   const FloatEbmType * validationTargets,
   const FloatEbmType * validationPredictorScores,
   IntEbmType countInnerBags,
   IntEbmType randomSeed,
   const FloatEbmType * optionalTempParams
) {
   LOG_N(TraceLevelInfo, "Entered InitializeBoostingRegression: countFeatures=%" IntEbmTypePrintf ", features=%p, countFeatureCombinations=%" 
      IntEbmTypePrintf ", featureCombinations=%p, featureCombinationIndexes=%p, countTrainingInstances=%" IntEbmTypePrintf 
      ", trainingBinnedData=%p, trainingTargets=%p, trainingPredictorScores=%p, countValidationInstances=%" IntEbmTypePrintf 
      ", validationBinnedData=%p, validationTargets=%p, validationPredictorScores=%p, countInnerBags=%" IntEbmTypePrintf 
      ", randomSeed=%" IntEbmTypePrintf ", optionalTempParams=%p",
      countFeatures, 
      static_cast<const void *>(features), 
      countFeatureCombinations, 
      static_cast<const void *>(featureCombinations), 
      static_cast<const void *>(featureCombinationIndexes), 
      countTrainingInstances, 
      static_cast<const void *>(trainingBinnedData), 
      static_cast<const void *>(trainingTargets), 
      static_cast<const void *>(trainingPredictorScores), 
      countValidationInstances, 
      static_cast<const void *>(validationBinnedData), 
      static_cast<const void *>(validationTargets), 
      static_cast<const void *>(validationPredictorScores), 
      countInnerBags, 
      randomSeed,
      static_cast<const void *>(optionalTempParams)
   );
   const PEbmBoosting pEbmBoosting = reinterpret_cast<PEbmBoosting>(AllocateBoosting(
      randomSeed, 
      countFeatures, 
      features, 
      countFeatureCombinations, 
      featureCombinations, 
      featureCombinationIndexes, 
      k_regression, 
      countTrainingInstances, 
      trainingTargets, 
      trainingBinnedData, 
      trainingPredictorScores, 
      countValidationInstances, 
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
   IntEbmType indexFeatureCombination,
   FloatEbmType learningRate,
   IntEbmType countTreeSplitsMax,
   IntEbmType countInstancesRequiredForChildSplitMin,
   const FloatEbmType * trainingWeights,
   const FloatEbmType * validationWeights,
   FloatEbmType * validationMetricReturn
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
         if(nullptr != validationMetricReturn) {
            *validationMetricReturn = FloatEbmType { 0 };
         }
         LOG_0(TraceLevelWarning, "WARNING BoostingStep pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses <= ptrdiff_t { 1 }");
         return 0;
      }
   }

   FloatEbmType gain; // we toss this value, but we still need to get it
   FloatEbmType * pModelFeatureCombinationUpdateTensor = GenerateModelFeatureCombinationUpdate(
      ebmBoosting, 
      indexFeatureCombination, 
      learningRate, 
      countTreeSplitsMax, 
      countInstancesRequiredForChildSplitMin, 
      trainingWeights, 
      validationWeights, 
      &gain
   );
   if(nullptr == pModelFeatureCombinationUpdateTensor) {
      // if we get back a nullptr from GenerateModelFeatureCombinationUpdate it either means that there's only
      // 1 class in our classification problem, or it means we encountered an error.  We assume here that
      // it was an error since the caller can check ahead of time if there was only 1 class before calling us
      if(nullptr != validationMetricReturn) {
         *validationMetricReturn = FloatEbmType { 0 };
      }
      return 1;
   }
   return ApplyModelFeatureCombinationUpdate(ebmBoosting, indexFeatureCombination, pModelFeatureCombinationUpdateTensor, validationMetricReturn);
}

EBM_NATIVE_IMPORT_EXPORT_BODY FloatEbmType * EBM_NATIVE_CALLING_CONVENTION GetBestModelFeatureCombination(
   PEbmBoosting ebmBoosting,
   IntEbmType indexFeatureCombination
) {
   LOG_N(
      TraceLevelInfo, 
      "Entered GetBestModelFeatureCombination: ebmBoosting=%p, indexFeatureCombination=%" IntEbmTypePrintf, 
      static_cast<void *>(ebmBoosting), 
      indexFeatureCombination
   );

   EbmBoostingState * pEbmBoostingState = reinterpret_cast<EbmBoostingState *>(ebmBoosting);
   if(nullptr == pEbmBoostingState) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureCombination ebmBoosting cannot be nullptr");
      return nullptr;
   }
   if(indexFeatureCombination < 0) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureCombination indexFeatureCombination must be positive");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntEbmType>(indexFeatureCombination)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureCombination indexFeatureCombination is too high to index");
      return nullptr;
   }
   size_t iFeatureCombination = static_cast<size_t>(indexFeatureCombination);
   if(pEbmBoostingState->GetCountFeatureCombinations() <= iFeatureCombination) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureCombination indexFeatureCombination above the number of feature groups that we have");
      return nullptr;
   }
   if(nullptr == pEbmBoostingState->GetBestModel()) {
      // if pEbmBoostingState->m_apBestModel is nullptr, then either:
      //    1) m_cFeatureCombinations was 0, in which case this function would have undefined behavior since the caller needs to indicate a valid 
      //       indexFeatureCombination, which is impossible, so we can do anything we like, include the below actions.
      //    2) m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0 (and the learning type is classification), 
      //       which is legal, which we need to handle here
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.  If there were logits in this model, 
      // they'd all be infinity, but you could alternatively think of this model as having zero logits, since the number of logits can be one 
      // less than the number of target classification classes.  A model with zero logits is empty, and has zero items.  We want to return a tensor 
      // with 0 items in it, so we could either return a pointer to some random memory that can't be accessed, or we can return nullptr.  
      // We return a nullptr in the hopes that our caller will either handle it or throw a nicer exception.

      LOG_0(TraceLevelInfo, "Exited GetBestModelFeatureCombination no model");
      return nullptr;
   }

   SegmentedTensor * pBestModel = pEbmBoostingState->GetBestModel()[iFeatureCombination];
   EBM_ASSERT(nullptr != pBestModel);
   EBM_ASSERT(pBestModel->GetExpanded()); // the model should have been expanded at startup
   FloatEbmType * pRet = pBestModel->GetValuePointer();
   EBM_ASSERT(nullptr != pRet);

   LOG_N(TraceLevelInfo, "Exited GetBestModelFeatureCombination %p", static_cast<void *>(pRet));
   return pRet;
}

EBM_NATIVE_IMPORT_EXPORT_BODY FloatEbmType * EBM_NATIVE_CALLING_CONVENTION GetCurrentModelFeatureCombination(
   PEbmBoosting ebmBoosting,
   IntEbmType indexFeatureCombination
) {
   LOG_N(
      TraceLevelInfo, 
      "Entered GetCurrentModelFeatureCombination: ebmBoosting=%p, indexFeatureCombination=%" IntEbmTypePrintf, 
      static_cast<void *>(ebmBoosting), 
      indexFeatureCombination
   );

   EbmBoostingState * pEbmBoostingState = reinterpret_cast<EbmBoostingState *>(ebmBoosting);
   if(nullptr == pEbmBoostingState) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureCombination ebmBoosting cannot be nullptr");
      return nullptr;
   }
   if(indexFeatureCombination < 0) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureCombination indexFeatureCombination must be positive");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntEbmType>(indexFeatureCombination)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureCombination indexFeatureCombination is too high to index");
      return nullptr;
   }
   size_t iFeatureCombination = static_cast<size_t>(indexFeatureCombination);
   if(pEbmBoostingState->GetCountFeatureCombinations() <= iFeatureCombination) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureCombination indexFeatureCombination above the number of feature groups that we have");
      return nullptr;
   }
   if(nullptr == pEbmBoostingState->GetCurrentModel()) {
      // if pEbmBoostingState->m_apCurrentModel is nullptr, then either:
      //    1) m_cFeatureCombinations was 0, in which case this function would have undefined behavior since the caller needs to indicate a valid 
      //       indexFeatureCombination, which is impossible, so we can do anything we like, include the below actions.
      //    2) m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0 (and the learning type is classification), which is legal, 
      //       which we need to handle here
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.  If there were logits 
      // in this model, they'd all be infinity, but you could alternatively think of this model as having zero logits, since the number of 
      // logits can be one less than the number of target classification classes.  A model with zero logits is empty, and has zero items.  
      // We want to return a tensor with 0 items in it, so we could either return a pointer to some random memory that can't be accessed, 
      // or we can return nullptr.  We return a nullptr in the hopes that our caller will either handle it or throw a nicer exception.

      LOG_0(TraceLevelInfo, "Exited GetCurrentModelFeatureCombination no model");
      return nullptr;
   }

   SegmentedTensor * pCurrentModel = pEbmBoostingState->GetCurrentModel()[iFeatureCombination];
   EBM_ASSERT(nullptr != pCurrentModel);
   EBM_ASSERT(pCurrentModel->GetExpanded()); // the model should have been expanded at startup
   FloatEbmType * pRet = pCurrentModel->GetValuePointer();
   EBM_ASSERT(nullptr != pRet);

   LOG_N(TraceLevelInfo, "Exited GetCurrentModelFeatureCombination %p", static_cast<void *>(pRet));
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
