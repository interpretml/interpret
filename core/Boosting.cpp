// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <string.h> // memset
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebmcore.h"

#include "EbmInternal.h"
// very independent includes
#include "Logging.h" // EBM_ASSERT & LOG
#include "InitializeResiduals.h"
#include "RandomStream.h"
#include "SegmentedTensor.h"
#include "EbmStatistics.h"
// this depends on TreeNode pointers, but doesn't require the full definition of TreeNode
#include "CachedThreadResources.h"
// feature includes
#include "FeatureCore.h"
// FeatureCombination.h depends on FeatureInternal.h
#include "FeatureCombinationCore.h"
// dataset depends on features
#include "DataSetByFeatureCombination.h"
// samples is somewhat independent from datasets, but relies on an indirect coupling with them
#include "SamplingWithReplacement.h"
// TreeNode depends on almost everything
#include "DimensionSingle.h"
#include "DimensionMultiple.h"

#include "EbmBoostingState.h"

void EbmBoostingState::DeleteSegmentedTensors(const size_t cFeatureCombinations, SegmentedTensor<ActiveDataType, FractionalDataType> ** const apSegmentedTensors) {
   LOG_0(TraceLevelInfo, "Entered DeleteSegmentedTensors");

   if(UNLIKELY(nullptr != apSegmentedTensors)) {
      EBM_ASSERT(0 < cFeatureCombinations);
      SegmentedTensor<ActiveDataType, FractionalDataType> ** ppSegmentedTensors = apSegmentedTensors;
      const SegmentedTensor<ActiveDataType, FractionalDataType> * const * const ppSegmentedTensorsEnd = &apSegmentedTensors[cFeatureCombinations];
      do {
         SegmentedTensor<ActiveDataType, FractionalDataType>::Free(*ppSegmentedTensors);
         ++ppSegmentedTensors;
      } while(ppSegmentedTensorsEnd != ppSegmentedTensors);
      delete[] apSegmentedTensors;
   }
   LOG_0(TraceLevelInfo, "Exited DeleteSegmentedTensors");
}

SegmentedTensor<ActiveDataType, FractionalDataType> ** EbmBoostingState::InitializeSegmentedTensors(const size_t cFeatureCombinations, const FeatureCombinationCore * const * const apFeatureCombinations, const size_t cVectorLength) {
   LOG_0(TraceLevelInfo, "Entered InitializeSegmentedTensors");

   EBM_ASSERT(0 < cFeatureCombinations);
   EBM_ASSERT(nullptr != apFeatureCombinations);
   EBM_ASSERT(1 <= cVectorLength);

   SegmentedTensor<ActiveDataType, FractionalDataType> ** const apSegmentedTensors = new (std::nothrow) SegmentedTensor<ActiveDataType, FractionalDataType> *[cFeatureCombinations];
   if(UNLIKELY(nullptr == apSegmentedTensors)) {
      LOG_0(TraceLevelWarning, "WARNING InitializeSegmentedTensors nullptr == apSegmentedTensors");
      return nullptr;
   }
   memset(apSegmentedTensors, 0, sizeof(*apSegmentedTensors) * cFeatureCombinations); // this needs to be done immediately after allocation otherwise we might attempt to free random garbage on an error

   SegmentedTensor<ActiveDataType, FractionalDataType> ** ppSegmentedTensors = apSegmentedTensors;
   for(size_t iFeatureCombination = 0; iFeatureCombination < cFeatureCombinations; ++iFeatureCombination) {
      const FeatureCombinationCore * const pFeatureCombination = apFeatureCombinations[iFeatureCombination];
      SegmentedTensor<ActiveDataType, FractionalDataType> * const pSegmentedTensors = SegmentedTensor<ActiveDataType, FractionalDataType>::Allocate(pFeatureCombination->m_cFeatures, cVectorLength);
      if(UNLIKELY(nullptr == pSegmentedTensors)) {
         LOG_0(TraceLevelWarning, "WARNING InitializeSegmentedTensors nullptr == pSegmentedTensors");
         DeleteSegmentedTensors(cFeatureCombinations, apSegmentedTensors);
         return nullptr;
      }

      if(0 == pFeatureCombination->m_cFeatures) {
         // if there are zero dimensions, then we have a tensor with 1 item, and we're already expanded
         pSegmentedTensors->m_bExpanded = true;
      } else {
         // if our segmented region has no dimensions, then it's already a fully expanded with 1 bin

         // TODO optimize the next few lines
         // TODO there might be a nicer way to expand this at allocation time (fill with zeros is easier)
         // we want to return a pointer to our interior state in the GetCurrentModelFeatureCombination and GetBestModelFeatureCombination functions.  For simplicity we don't transmit the divions, so we need to expand our SegmentedRegion before returning
         // the easiest way to ensure that the SegmentedRegion is expanded is to start it off expanded, and then we don't have to check later since anything merged into an expanded SegmentedRegion will itself be expanded
         size_t acDivisionIntegersEnd[k_cDimensionsMax];
         size_t iDimension = 0;
         do {
            acDivisionIntegersEnd[iDimension] = ARRAY_TO_POINTER_CONST(pFeatureCombination->m_FeatureCombinationEntry)[iDimension].m_pFeature->m_cBins;
            ++iDimension;
         } while(iDimension < pFeatureCombination->m_cFeatures);

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

bool EbmBoostingState::Initialize(const EbmCoreFeature * const aFeatures, const EbmCoreFeatureCombination * const aFeatureCombinations, const IntegerDataType * featureCombinationIndexes, const size_t cTrainingInstances, const void * const aTrainingTargets, const IntegerDataType * const aTrainingBinnedData, const FractionalDataType * const aTrainingPredictorScores, const size_t cValidationInstances, const void * const aValidationTargets, const IntegerDataType * const aValidationBinnedData, const FractionalDataType * const aValidationPredictorScores) {
   LOG_0(TraceLevelInfo, "Entered EbmBoostingState::Initialize");

   const bool bClassification = IsClassification(m_runtimeLearningTypeOrCountTargetClasses);

   if(bClassification) {
      if(m_cachedThreadResourcesUnion.classification.IsError()) {
         LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize m_cachedThreadResourcesUnion.classification.IsError()");
         return true;
      }
   } else {
      EBM_ASSERT(IsRegression(m_runtimeLearningTypeOrCountTargetClasses));
      if(m_cachedThreadResourcesUnion.regression.IsError()) {
         LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize m_cachedThreadResourcesUnion.regression.IsError()");
         return true;
      }
   }

   if(0 != m_cFeatures && nullptr == m_aFeatures) {
      LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize 0 != m_cFeatures && nullptr == m_aFeatures");
      return true;
   }

   if(UNLIKELY(0 != m_cFeatureCombinations && nullptr == m_apFeatureCombinations)) {
      LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize 0 != m_cFeatureCombinations && nullptr == m_apFeatureCombinations");
      return true;
   }

   if(UNLIKELY(nullptr == m_pSmallChangeToModelOverwriteSingleSamplingSet)) {
      LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == m_pSmallChangeToModelOverwriteSingleSamplingSet");
      return true;
   }

   if(UNLIKELY(nullptr == m_pSmallChangeToModelAccumulatedFromSamplingSets)) {
      LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == m_pSmallChangeToModelAccumulatedFromSamplingSets");
      return true;
   }

   if(UNLIKELY(!m_randomStream.IsSuccess())) {
      LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize m_randomStream.IsError()");
      return true;
   }

   LOG_0(TraceLevelInfo, "EbmBoostingState::Initialize starting feature processing");
   if(0 != m_cFeatures) {
      EBM_ASSERT(!IsMultiplyError(m_cFeatures, sizeof(*aFeatures))); // if this overflows then our caller should not have been able to allocate the array
      const EbmCoreFeature * pFeatureInitialize = aFeatures;
      const EbmCoreFeature * const pFeatureEnd = &aFeatures[m_cFeatures];
      EBM_ASSERT(pFeatureInitialize < pFeatureEnd);
      size_t iFeatureInitialize = 0;
      do {
         static_assert(FeatureTypeCore::OrdinalCore == static_cast<FeatureTypeCore>(FeatureTypeOrdinal), "FeatureTypeCore::OrdinalCore must have the same value as FeatureTypeOrdinal");
         static_assert(FeatureTypeCore::NominalCore == static_cast<FeatureTypeCore>(FeatureTypeNominal), "FeatureTypeCore::NominalCore must have the same value as FeatureTypeNominal");
         EBM_ASSERT(FeatureTypeOrdinal == pFeatureInitialize->featureType || FeatureTypeNominal == pFeatureInitialize->featureType);
         FeatureTypeCore featureTypeCore = static_cast<FeatureTypeCore>(pFeatureInitialize->featureType);

         IntegerDataType countBins = pFeatureInitialize->countBins;
         EBM_ASSERT(0 <= countBins); // we can handle 1 == cBins or 0 == cBins even though that's a degenerate case that shouldn't be boosted on (dimensions with 1 bin don't contribute anything since they always have the same value).  0 cases could only occur if there were zero training and zero validation cases since the features would require a value, even if it was 0
         if(!IsNumberConvertable<size_t, IntegerDataType>(countBins)) {
            LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize !IsNumberConvertable<size_t, IntegerDataType>(countBins)");
            return true;
         }
         size_t cBins = static_cast<size_t>(countBins);
         if(cBins <= 1) {
            EBM_ASSERT(0 != cBins || 0 == cTrainingInstances && 0 == cValidationInstances);
            LOG_0(TraceLevelInfo, "INFO EbmBoostingState::Initialize feature with 0/1 values");
         }

         EBM_ASSERT(0 == pFeatureInitialize->hasMissing || 1 == pFeatureInitialize->hasMissing);
         bool bMissing = 0 != pFeatureInitialize->hasMissing;

         // this is an in-place new, so there is no new memory allocated, and we already knew where it was going, so we don't need the resulting pointer returned
         new (&m_aFeatures[iFeatureInitialize]) FeatureCore(cBins, iFeatureInitialize, featureTypeCore, bMissing);
         // we don't allocate memory and our constructor doesn't have errors, so we shouldn't have an error here

         EBM_ASSERT(0 == pFeatureInitialize->hasMissing); // TODO : implement this, then remove this assert
         EBM_ASSERT(FeatureTypeOrdinal == pFeatureInitialize->featureType); // TODO : implement this, then remove this assert

         ++iFeatureInitialize;
         ++pFeatureInitialize;
      } while(pFeatureEnd != pFeatureInitialize);
   }
   LOG_0(TraceLevelInfo, "EbmBoostingState::Initialize done feature processing");

   const size_t cVectorLength = GetVectorLengthFlatCore(m_runtimeLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelInfo, "EbmBoostingState::Initialize starting feature combination processing");
   if(0 != m_cFeatureCombinations) {
      size_t cBytesPerSweepTreeNode = 0;
      if(bClassification) {
         if(GetSweepTreeNodeSizeOverflow<true>(cVectorLength)) {
            LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize GetSweepTreeNodeSizeOverflow<true>(cVectorLength)");
            return true;
         }
         cBytesPerSweepTreeNode = GetSweepTreeNodeSize<true>(cVectorLength);
      } else {
         if(GetSweepTreeNodeSizeOverflow<false>(cVectorLength)) {
            LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize GetSweepTreeNodeSizeOverflow<false>(cVectorLength)");
            return true;
         }
         cBytesPerSweepTreeNode = GetSweepTreeNodeSize<false>(cVectorLength);
      }
      size_t cBytesArrayEquivalentSplitMax = 0;

      const IntegerDataType * pFeatureCombinationIndex = featureCombinationIndexes;
      size_t iFeatureCombination = 0;
      do {
         const EbmCoreFeatureCombination * const pFeatureCombinationInterop = &aFeatureCombinations[iFeatureCombination];

         IntegerDataType countFeaturesInCombination = pFeatureCombinationInterop->countFeaturesInCombination;
         EBM_ASSERT(0 <= countFeaturesInCombination);
         if(!IsNumberConvertable<size_t, IntegerDataType>(countFeaturesInCombination)) {
            LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize !IsNumberConvertable<size_t, IntegerDataType>(countFeaturesInCombination)");
            return true;
         }
         size_t cFeaturesInCombination = static_cast<size_t>(countFeaturesInCombination);
         size_t cSignificantFeaturesInCombination = 0;
         const IntegerDataType * const pFeatureCombinationIndexEnd = pFeatureCombinationIndex + cFeaturesInCombination;
         if(UNLIKELY(0 == cFeaturesInCombination)) {
            LOG_0(TraceLevelInfo, "INFO EbmBoostingState::Initialize empty feature combination");
         } else {
            EBM_ASSERT(nullptr != featureCombinationIndexes);
            const IntegerDataType * pFeatureCombinationIndexTemp = pFeatureCombinationIndex;
            do {
               const IntegerDataType indexFeatureInterop = *pFeatureCombinationIndexTemp;
               EBM_ASSERT(0 <= indexFeatureInterop);
               if(!IsNumberConvertable<size_t, IntegerDataType>(indexFeatureInterop)) {
                  LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize !IsNumberConvertable<size_t, IntegerDataType>(indexFeatureInterop)");
                  return true;
               }
               const size_t iFeatureForCombination = static_cast<size_t>(indexFeatureInterop);
               EBM_ASSERT(iFeatureForCombination < m_cFeatures);
               FeatureCore * const pInputFeature = &m_aFeatures[iFeatureForCombination];
               if(LIKELY(1 < pInputFeature->m_cBins)) {
                  // if we have only 1 bin, then we can eliminate the feature from consideration since the resulting tensor loses one dimension but is otherwise indistinquishable from the original data
                  ++cSignificantFeaturesInCombination;
               } else {
                  LOG_0(TraceLevelInfo, "INFO EbmBoostingState::Initialize feature combination with no useful features");
               }
               ++pFeatureCombinationIndexTemp;
            } while(pFeatureCombinationIndexEnd != pFeatureCombinationIndexTemp);

            if(k_cDimensionsMax < cSignificantFeaturesInCombination) {
               // if we try to run with more than k_cDimensionsMax we'll exceed our memory capacity, so let's exit here instead
               LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize k_cDimensionsMax < cSignificantFeaturesInCombination");
               return true;
            }
         }

         FeatureCombinationCore * pFeatureCombination = FeatureCombinationCore::Allocate(cSignificantFeaturesInCombination, iFeatureCombination);
         if(nullptr == pFeatureCombination) {
            LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == pFeatureCombination");
            return true;
         }
         // assign our pointer directly to our array right now so that we can't loose the memory if we decide to exit due to an error below
         m_apFeatureCombinations[iFeatureCombination] = pFeatureCombination;

         if(LIKELY(0 == cSignificantFeaturesInCombination)) {
            // move our index forward to the next feature.  
            // We won't be executing the loop below that would otherwise increment it by the number of features in this feature combination
            pFeatureCombinationIndex = pFeatureCombinationIndexEnd;
         } else {
            EBM_ASSERT(nullptr != featureCombinationIndexes);
            size_t cEquivalentSplits = 1;
            size_t cTensorBins = 1;
            FeatureCombinationCore::FeatureCombinationEntry * pFeatureCombinationEntry = ARRAY_TO_POINTER(pFeatureCombination->m_FeatureCombinationEntry);
            do {
               const IntegerDataType indexFeatureInterop = *pFeatureCombinationIndex;
               EBM_ASSERT(0 <= indexFeatureInterop);
               EBM_ASSERT((IsNumberConvertable<size_t, IntegerDataType>(indexFeatureInterop))); // this was checked above
               const size_t iFeatureForCombination = static_cast<size_t>(indexFeatureInterop);
               EBM_ASSERT(iFeatureForCombination < m_cFeatures);
               const FeatureCore * const pInputFeature = &m_aFeatures[iFeatureForCombination];
               const size_t cBins = pInputFeature->m_cBins;
               if(LIKELY(1 < cBins)) {
                  // if we have only 1 bin, then we can eliminate the feature from consideration since the resulting tensor loses one dimension but is otherwise indistinquishable from the original data
                  pFeatureCombinationEntry->m_pFeature = pInputFeature;
                  ++pFeatureCombinationEntry;
                  if(IsMultiplyError(cTensorBins, cBins)) {
                     // if this overflows, we definetly won't be able to allocate it
                     LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize IsMultiplyError(cTensorStates, cBins)");
                     return true;
                  }
                  cTensorBins *= cBins;
                  cEquivalentSplits *= cBins - 1; // we can only split between the bins
               }
               ++pFeatureCombinationIndex;
            } while(pFeatureCombinationIndexEnd != pFeatureCombinationIndex);

            size_t cBytesArrayEquivalentSplit;
            if(1 == cSignificantFeaturesInCombination) {
               if(IsMultiplyError(cEquivalentSplits, cBytesPerSweepTreeNode)) {
                  LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize IsMultiplyError(cEquivalentSplits, cBytesPerSweepTreeNode)");
                  return true;
               }
               cBytesArrayEquivalentSplit = cEquivalentSplits * cBytesPerSweepTreeNode;
            } else {
               // TODO : someday add equal gain multidimensional randomized picking.  It's rather hard though with the existing sweep functions for multidimensional right now
               cBytesArrayEquivalentSplit = 0;
            }
            if(cBytesArrayEquivalentSplitMax < cBytesArrayEquivalentSplit) {
               cBytesArrayEquivalentSplitMax = cBytesArrayEquivalentSplit;
            }

            // if cSignificantFeaturesInCombination is zero, don't both initializing pFeatureCombination->m_cItemsPerBitPackDataUnit
            const size_t cBitsRequiredMin = CountBitsRequiredCore(cTensorBins - 1);
            pFeatureCombination->m_cItemsPerBitPackDataUnit = GetCountItemsBitPacked(cBitsRequiredMin);
         }
         ++iFeatureCombination;
      } while(iFeatureCombination < m_cFeatureCombinations);

      if(0 != cBytesArrayEquivalentSplitMax) {
         void * const aEquivalentSplits = malloc(cBytesArrayEquivalentSplitMax);
         if(bClassification) {
            m_cachedThreadResourcesUnion.classification.m_aEquivalentSplits = aEquivalentSplits;
         } else {
            m_cachedThreadResourcesUnion.regression.m_aEquivalentSplits = aEquivalentSplits;
         }
      }
   }
   LOG_0(TraceLevelInfo, "EbmBoostingState::Initialize finished feature combination processing");

   LOG_0(TraceLevelInfo, "Entered DataSetByFeatureCombination for m_pTrainingSet");
   if(0 != cTrainingInstances) {
      m_pTrainingSet = new (std::nothrow) DataSetByFeatureCombination(true, bClassification, bClassification, m_cFeatureCombinations, m_apFeatureCombinations, cTrainingInstances, aTrainingBinnedData, aTrainingTargets, aTrainingPredictorScores, cVectorLength);
      if(nullptr == m_pTrainingSet || m_pTrainingSet->IsError()) {
         LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == m_pTrainingSet || m_pTrainingSet->IsError()");
         return true;
      }
   }
   LOG_N(TraceLevelInfo, "Exited DataSetByFeatureCombination for m_pTrainingSet %p", static_cast<void *>(m_pTrainingSet));

   LOG_0(TraceLevelInfo, "Entered DataSetByFeatureCombination for m_pValidationSet");
   if(0 != cValidationInstances) {
      m_pValidationSet = new (std::nothrow) DataSetByFeatureCombination(!bClassification, bClassification, bClassification, m_cFeatureCombinations, m_apFeatureCombinations, cValidationInstances, aValidationBinnedData, aValidationTargets, aValidationPredictorScores, cVectorLength);
      if(nullptr == m_pValidationSet || m_pValidationSet->IsError()) {
         LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == m_pValidationSet || m_pValidationSet->IsError()");
         return true;
      }
   }
   LOG_N(TraceLevelInfo, "Exited DataSetByFeatureCombination for m_pValidationSet %p", static_cast<void *>(m_pValidationSet));

   EBM_ASSERT(nullptr == m_apSamplingSets);
   if(0 != cTrainingInstances) {
      m_apSamplingSets = SamplingWithReplacement::GenerateSamplingSets(&m_randomStream, m_pTrainingSet, m_cSamplingSets);
      if(UNLIKELY(nullptr == m_apSamplingSets)) {
         LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == m_apSamplingSets");
         return true;
      }
   }

   EBM_ASSERT(nullptr == m_apCurrentModel);
   EBM_ASSERT(nullptr == m_apBestModel);
   if(0 != m_cFeatureCombinations && (!bClassification || ptrdiff_t { 2 } <= m_runtimeLearningTypeOrCountTargetClasses)) {
      m_apCurrentModel = InitializeSegmentedTensors(m_cFeatureCombinations, m_apFeatureCombinations, cVectorLength);
      if(nullptr == m_apCurrentModel) {
         LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == m_apCurrentModel");
         return true;
      }
      m_apBestModel = InitializeSegmentedTensors(m_cFeatureCombinations, m_apFeatureCombinations, cVectorLength);
      if(nullptr == m_apBestModel) {
         LOG_0(TraceLevelWarning, "WARNING EbmBoostingState::Initialize nullptr == m_apBestModel");
         return true;
      }
   }

   if(bClassification) {
      if(size_t { 2 } == static_cast<size_t>(m_runtimeLearningTypeOrCountTargetClasses)) {
         if(0 != cTrainingInstances) {
            InitializeResiduals<2>(cTrainingInstances, aTrainingTargets, aTrainingPredictorScores, m_pTrainingSet->GetResidualPointer(), ptrdiff_t { 2 });
         }
      } else {
         if(0 != cTrainingInstances) {
            InitializeResiduals<k_DynamicClassification>(cTrainingInstances, aTrainingTargets, aTrainingPredictorScores, m_pTrainingSet->GetResidualPointer(), m_runtimeLearningTypeOrCountTargetClasses);
         }
      }
   } else {
      EBM_ASSERT(IsRegression(m_runtimeLearningTypeOrCountTargetClasses));
      if(0 != cTrainingInstances) {
         InitializeResiduals<k_Regression>(cTrainingInstances, aTrainingTargets, aTrainingPredictorScores, m_pTrainingSet->GetResidualPointer(), k_Regression);
      }
      if(0 != cValidationInstances) {
         InitializeResiduals<k_Regression>(cValidationInstances, aValidationTargets, aValidationPredictorScores, m_pValidationSet->GetResidualPointer(), k_Regression);
      }
   }

   LOG_0(TraceLevelInfo, "Exited EbmBoostingState::Initialize");
   return false;
}

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
template<unsigned int cInputBits, unsigned int cTargetBits, ptrdiff_t compilerLearningTypeOrCountTargetClasses>
static void TrainingSetTargetFeatureLoop(const FeatureCombinationCore * const pFeatureCombination, DataSetByFeatureCombination * const pTrainingSet, const FractionalDataType * const aModelFeatureCombinationUpdateTensor, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses) {
   LOG_0(TraceLevelVerbose, "Entered TrainingSetTargetFeatureLoop");

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses);
   const size_t cInstances = pTrainingSet->GetCountInstances();
   EBM_ASSERT(0 < cInstances);

   if(0 == pFeatureCombination->m_cFeatures) {
      FractionalDataType * pResidualError = pTrainingSet->GetResidualPointer();
      const FractionalDataType * const pResidualErrorEnd = pResidualError + cVectorLength * cInstances;
      if(IsRegression(compilerLearningTypeOrCountTargetClasses)) {
         const FractionalDataType smallChangeToPrediction = aModelFeatureCombinationUpdateTensor[0];
         do {
            // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
            const FractionalDataType residualError = EbmStatistics::ComputeRegressionResidualError(*pResidualError - smallChangeToPrediction);
            *pResidualError = residualError;
            ++pResidualError;
         } while(pResidualErrorEnd != pResidualError);
      } else {
         EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));
         FractionalDataType * pTrainingPredictorScores = pTrainingSet->GetPredictorScores();
         const StorageDataTypeCore * pTargetData = pTrainingSet->GetTargetDataPointer();
         if(IsBinaryClassification(compilerLearningTypeOrCountTargetClasses)) {
            const FractionalDataType smallChangeToPredictorScores = aModelFeatureCombinationUpdateTensor[0];
            do {
               StorageDataTypeCore targetData = *pTargetData;
               // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
               const FractionalDataType trainingPredictorScore = *pTrainingPredictorScores + smallChangeToPredictorScores;
               *pTrainingPredictorScores = trainingPredictorScore;
               const FractionalDataType residualError = EbmStatistics::ComputeClassificationResidualErrorBinaryclass(trainingPredictorScore, targetData);
               *pResidualError = residualError;
               ++pResidualError;
               ++pTrainingPredictorScores;
               ++pTargetData;
            } while(pResidualErrorEnd != pResidualError);
         } else {
            const FractionalDataType * pValues = aModelFeatureCombinationUpdateTensor;
            do {
               StorageDataTypeCore targetData = *pTargetData;
               FractionalDataType sumExp = 0;
               size_t iVector1 = 0;
               do {
                  // TODO : because there is only one bin for a zero feature feature combination, we could move these values to the stack where the compiler could reason about their visibility and optimize small arrays into registers
                  const FractionalDataType smallChangeToPredictorScores = pValues[iVector1];
                  // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
                  const FractionalDataType trainingPredictorScores = pTrainingPredictorScores[iVector1] + smallChangeToPredictorScores;
                  pTrainingPredictorScores[iVector1] = trainingPredictorScores;
                  sumExp += EbmExp(trainingPredictorScores);
                  ++iVector1;
               } while(iVector1 < cVectorLength);

               EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, size_t>(cVectorLength)));
               const StorageDataTypeCore cVectorLengthStorage = static_cast<StorageDataTypeCore>(cVectorLength);
               StorageDataTypeCore iVector2 = 0;
               do {
                  // TODO : we're calculating exp(predictionScore) above, and then again in ComputeClassificationResidualErrorMulticlass.  exp(..) is expensive so we should just do it once instead and store the result in a small memory array here
                  const FractionalDataType residualError = EbmStatistics::ComputeClassificationResidualErrorMulticlass(sumExp, pTrainingPredictorScores[iVector2], targetData, iVector2);
                  *pResidualError = residualError;
                  ++pResidualError;
                  ++iVector2;
               } while(iVector2 < cVectorLengthStorage);
               // TODO: this works as a way to remove one parameter, but it obviously insn't as efficient as omitting the parameter
               // 
               // this works out in the math as making the first model vector parameter equal to zero, which in turn removes one degree of freedom
               // from the model vector parameters.  Since the model vector weights need to be normalized to sum to a probabilty of 100%, we can set the first
               // one to the constant 1 (0 in log space) and force the other parameters to adjust to that scale which fixes them to a single valid set of values
               // insted of allowing them to be scaled.  
               // Probability = exp(T1 + I1) / [exp(T1 + I1) + exp(T2 + I2) + exp(T3 + I3)] => we can add a constant inside each exp(..) term, which will be multiplication outside the exp(..), which
               // means the numerator and denominator are multiplied by the same constant, which cancels eachother out.  We can thus set exp(T2 + I2) to exp(0) and adjust the other terms
               constexpr bool bZeroingResiduals = 0 <= k_iZeroResidual;
               if(bZeroingResiduals) {
                  *(pResidualError - (cVectorLength - static_cast<size_t>(k_iZeroResidual))) = 0;
               }
               pTrainingPredictorScores += cVectorLength;
               ++pTargetData;
            } while(pResidualErrorEnd != pResidualError);
         }
      }
      LOG_0(TraceLevelVerbose, "Exited TrainingSetTargetFeatureLoop - Zero dimensions");
      return;
   }

   const size_t cItemsPerBitPackDataUnit = pFeatureCombination->m_cItemsPerBitPackDataUnit;
   EBM_ASSERT(1 <= cItemsPerBitPackDataUnit);
   EBM_ASSERT(cItemsPerBitPackDataUnit <= k_cBitsForStorageType);
   const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackDataUnit);
   EBM_ASSERT(1 <= cBitsPerItemMax);
   EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
   const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);

   const StorageDataTypeCore * pInputData = pTrainingSet->GetInputDataPointer(pFeatureCombination);
   FractionalDataType * pResidualError = pTrainingSet->GetResidualPointer();

   if(IsRegression(compilerLearningTypeOrCountTargetClasses)) {
      // this shouldn't overflow since we're accessing existing memory
      const FractionalDataType * const pResidualErrorTrueEnd = pResidualError + cVectorLength * cInstances;
      const FractionalDataType * pResidualErrorExit = pResidualErrorTrueEnd;
      size_t cItemsRemaining = cInstances;
      if(cInstances <= cItemsPerBitPackDataUnit) {
         goto one_last_loop_regression;
      }
      pResidualErrorExit = pResidualErrorTrueEnd - cVectorLength * ((cInstances - 1) % cItemsPerBitPackDataUnit + 1);
      EBM_ASSERT(pResidualError < pResidualErrorExit);
      EBM_ASSERT(pResidualErrorExit < pResidualErrorTrueEnd);

      do {
         cItemsRemaining = cItemsPerBitPackDataUnit;
         // TODO : jumping back into this loop and changing cItemsRemaining to a dynamic value that isn't compile time determinable
         // causes this function to NOT be optimized as much as it could if we had two separate loops.  We're just trying this out for now though
      one_last_loop_regression:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iTensorBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            const size_t iTensorBin = maskBits & iTensorBinCombined;
            const FractionalDataType smallChangeToPrediction = aModelFeatureCombinationUpdateTensor[iTensorBin * cVectorLength];
            // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
            const FractionalDataType residualError = EbmStatistics::ComputeRegressionResidualError(*pResidualError - smallChangeToPrediction);
            *pResidualError = residualError;
            ++pResidualError;

            iTensorBinCombined >>= cBitsPerItemMax;
            // TODO : try replacing cItemsRemaining with a pResidualErrorInnerLoopEnd which eliminates one subtact operation, but might make it harder for the compiler to optimize the loop away
            --cItemsRemaining;
         } while(0 != cItemsRemaining);
      } while(pResidualErrorExit != pResidualError);

      // first time through?
      if(pResidualErrorTrueEnd != pResidualError) {
         EBM_ASSERT(0 == (pResidualErrorTrueEnd - pResidualError) % cVectorLength);
         cItemsRemaining = (pResidualErrorTrueEnd - pResidualError) / cVectorLength;
         EBM_ASSERT(0 < cItemsRemaining);
         EBM_ASSERT(cItemsRemaining <= cItemsPerBitPackDataUnit);

         pResidualErrorExit = pResidualErrorTrueEnd;

         goto one_last_loop_regression;
      }
   } else {
      EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));
      FractionalDataType * pTrainingPredictorScores = pTrainingSet->GetPredictorScores();
      const StorageDataTypeCore * pTargetData = pTrainingSet->GetTargetDataPointer();

      // this shouldn't overflow since we're accessing existing memory
      const FractionalDataType * const pResidualErrorTrueEnd = pResidualError + cVectorLength * cInstances;
      const FractionalDataType * pResidualErrorExit = pResidualErrorTrueEnd;
      size_t cItemsRemaining = cInstances;
      if(cInstances <= cItemsPerBitPackDataUnit) {
         goto one_last_loop_classification;
      }
      pResidualErrorExit = pResidualErrorTrueEnd - cVectorLength * ((cInstances - 1) % cItemsPerBitPackDataUnit + 1);
      EBM_ASSERT(pResidualError < pResidualErrorExit);
      EBM_ASSERT(pResidualErrorExit < pResidualErrorTrueEnd);

      do {
         cItemsRemaining = cItemsPerBitPackDataUnit;
         // TODO : jumping back into this loop and changing cItemsRemaining to a dynamic value that isn't compile time determinable
         // causes this function to NOT be optimized as much as it could if we had two separate loops.  We're just trying this out for now though
      one_last_loop_classification:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iTensorBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            StorageDataTypeCore targetData = *pTargetData;

            const size_t iTensorBin = maskBits & iTensorBinCombined;
            const FractionalDataType * pValues = &aModelFeatureCombinationUpdateTensor[iTensorBin * cVectorLength];

            if(IsBinaryClassification(compilerLearningTypeOrCountTargetClasses)) {
               const FractionalDataType smallChangeToPredictorScores = pValues[0];
               // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
               const FractionalDataType trainingPredictorScore = *pTrainingPredictorScores + smallChangeToPredictorScores;
               *pTrainingPredictorScores = trainingPredictorScore;
               const FractionalDataType residualError = EbmStatistics::ComputeClassificationResidualErrorBinaryclass(trainingPredictorScore, targetData);
               *pResidualError = residualError;
               ++pResidualError;
            } else {
               FractionalDataType sumExp = 0;
               size_t iVector1 = 0;
               do {
                  const FractionalDataType smallChangeToPredictorScores = pValues[iVector1];
                  // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
                  const FractionalDataType trainingPredictorScores = pTrainingPredictorScores[iVector1] + smallChangeToPredictorScores;
                  pTrainingPredictorScores[iVector1] = trainingPredictorScores;
                  sumExp += EbmExp(trainingPredictorScores);
                  ++iVector1;
               } while(iVector1 < cVectorLength);

               EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, size_t>(cVectorLength)));
               const StorageDataTypeCore cVectorLengthStorage = static_cast<StorageDataTypeCore>(cVectorLength);
               StorageDataTypeCore iVector2 = 0;
               do {
                  // TODO : we're calculating exp(predictionScore) above, and then again in ComputeClassificationResidualErrorMulticlass.  exp(..) is expensive so we should just do it once instead and store the result in a small memory array here
                  const FractionalDataType residualError = EbmStatistics::ComputeClassificationResidualErrorMulticlass(sumExp, pTrainingPredictorScores[iVector2], targetData, iVector2);
                  *pResidualError = residualError;
                  ++pResidualError;
                  ++iVector2;
               } while(iVector2 < cVectorLengthStorage);
               // TODO: this works as a way to remove one parameter, but it obviously insn't as efficient as omitting the parameter
               // 
               // this works out in the math as making the first model vector parameter equal to zero, which in turn removes one degree of freedom
               // from the model vector parameters.  Since the model vector weights need to be normalized to sum to a probabilty of 100%, we can set the first
               // one to the constant 1 (0 in log space) and force the other parameters to adjust to that scale which fixes them to a single valid set of values
               // insted of allowing them to be scaled.  
               // Probability = exp(T1 + I1) / [exp(T1 + I1) + exp(T2 + I2) + exp(T3 + I3)] => we can add a constant inside each exp(..) term, which will be multiplication outside the exp(..), which
               // means the numerator and denominator are multiplied by the same constant, which cancels eachother out.  We can thus set exp(T2 + I2) to exp(0) and adjust the other terms
               constexpr bool bZeroingResiduals = 0 <= k_iZeroResidual;
               if(bZeroingResiduals) {
                  *(pResidualError - (cVectorLength - static_cast<size_t>(k_iZeroResidual))) = 0;
               }
            }
            pTrainingPredictorScores += cVectorLength;
            ++pTargetData;

            iTensorBinCombined >>= cBitsPerItemMax;
            // TODO : try replacing cItemsRemaining with a pResidualErrorInnerLoopEnd which eliminates one subtact operation, but might make it harder for the compiler to optimize the loop away
            --cItemsRemaining;
         } while(0 != cItemsRemaining);
      } while(pResidualErrorExit != pResidualError);

      // first time through?
      if(pResidualErrorTrueEnd != pResidualError) {
         EBM_ASSERT(0 == (pResidualErrorTrueEnd - pResidualError) % cVectorLength);
         cItemsRemaining = (pResidualErrorTrueEnd - pResidualError) / cVectorLength;
         EBM_ASSERT(0 < cItemsRemaining);
         EBM_ASSERT(cItemsRemaining <= cItemsPerBitPackDataUnit);

         pResidualErrorExit = pResidualErrorTrueEnd;

         goto one_last_loop_classification;
      }
   }
   LOG_0(TraceLevelVerbose, "Exited TrainingSetTargetFeatureLoop");
}

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
template<unsigned int cInputBits, ptrdiff_t compilerLearningTypeOrCountTargetClasses>
static void TrainingSetInputFeatureLoop(const FeatureCombinationCore * const pFeatureCombination, DataSetByFeatureCombination * const pTrainingSet, const FractionalDataType * const aModelFeatureCombinationUpdateTensor, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses) {
   if(static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses) <= 1 << 1) {
      TrainingSetTargetFeatureLoop<cInputBits, 1, compilerLearningTypeOrCountTargetClasses>(pFeatureCombination, pTrainingSet, aModelFeatureCombinationUpdateTensor, runtimeLearningTypeOrCountTargetClasses);
   } else if(static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses) <= 1 << 2) {
      TrainingSetTargetFeatureLoop<cInputBits, 2, compilerLearningTypeOrCountTargetClasses>(pFeatureCombination, pTrainingSet, aModelFeatureCombinationUpdateTensor, runtimeLearningTypeOrCountTargetClasses);
   } else if(static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses) <= 1 << 4) {
      TrainingSetTargetFeatureLoop<cInputBits, 4, compilerLearningTypeOrCountTargetClasses>(pFeatureCombination, pTrainingSet, aModelFeatureCombinationUpdateTensor, runtimeLearningTypeOrCountTargetClasses);
   } else if(static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses) <= 1 << 8) {
      TrainingSetTargetFeatureLoop<cInputBits, 8, compilerLearningTypeOrCountTargetClasses>(pFeatureCombination, pTrainingSet, aModelFeatureCombinationUpdateTensor, runtimeLearningTypeOrCountTargetClasses);
   } else if(static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses) <= 1 << 16) {
      TrainingSetTargetFeatureLoop<cInputBits, 16, compilerLearningTypeOrCountTargetClasses>(pFeatureCombination, pTrainingSet, aModelFeatureCombinationUpdateTensor, runtimeLearningTypeOrCountTargetClasses);
   } else if(static_cast<uint64_t>(runtimeLearningTypeOrCountTargetClasses) <= uint64_t { 1 } << 32) {
      // if this is a 32 bit system, then m_cBins can't be 0x100000000 or above, because we would have checked that when converting the 64 bit numbers into size_t, and m_cBins will be promoted to a 64 bit number for the above comparison
      // if this is a 64 bit system, then this comparison is fine

      // TODO : perhaps we should change m_cBins into m_iBinMax so that we don't need to do the above promotion to 64 bits.. we can make it <= 0xFFFFFFFF.  Write a function to fill the lowest bits with ones for any number of bits

      TrainingSetTargetFeatureLoop<cInputBits, 32, compilerLearningTypeOrCountTargetClasses>(pFeatureCombination, pTrainingSet, aModelFeatureCombinationUpdateTensor, runtimeLearningTypeOrCountTargetClasses);
   } else {
      // our interface doesn't allow more than 64 bits, so even if size_t was bigger then we don't need to examine higher
      static_assert(63 == CountBitsRequiredPositiveMax<IntegerDataType>(), "");
      TrainingSetTargetFeatureLoop<cInputBits, 64, compilerLearningTypeOrCountTargetClasses>(pFeatureCombination, pTrainingSet, aModelFeatureCombinationUpdateTensor, runtimeLearningTypeOrCountTargetClasses);
   }
}

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
template<unsigned int cInputBits, unsigned int cTargetBits, ptrdiff_t compilerLearningTypeOrCountTargetClasses>
static FractionalDataType ValidationSetTargetFeatureLoop(const FeatureCombinationCore * const pFeatureCombination, DataSetByFeatureCombination * const pValidationSet, const FractionalDataType * const aModelFeatureCombinationUpdateTensor, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses) {
   LOG_0(TraceLevelVerbose, "Entering ValidationSetTargetFeatureLoop");

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses);
   const size_t cInstances = pValidationSet->GetCountInstances();
   EBM_ASSERT(0 < cInstances);

   if(0 == pFeatureCombination->m_cFeatures) {
      if(IsRegression(compilerLearningTypeOrCountTargetClasses)) {
         FractionalDataType * pResidualError = pValidationSet->GetResidualPointer();
         const FractionalDataType * const pResidualErrorEnd = pResidualError + cInstances;

         const FractionalDataType smallChangeToPrediction = aModelFeatureCombinationUpdateTensor[0];

         FractionalDataType meanSquareError = 0;
         do {
            // this will apply a small fix to our existing ValidationPredictorScores, either positive or negative, whichever is needed
            const FractionalDataType residualError = EbmStatistics::ComputeRegressionResidualError(*pResidualError - smallChangeToPrediction);
            meanSquareError += EbmStatistics::ComputeRegressionSingleInstanceMeanSquaredError(residualError);
            *pResidualError = residualError;
            ++pResidualError;
         } while(pResidualErrorEnd != pResidualError);

         meanSquareError /= pValidationSet->GetCountInstances();
         LOG_0(TraceLevelVerbose, "Exited ValidationSetTargetFeatureLoop - Zero dimensions");
         return meanSquareError;
      } else {
         EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));
         FractionalDataType * pValidationPredictorScores = pValidationSet->GetPredictorScores();
         const StorageDataTypeCore * pTargetData = pValidationSet->GetTargetDataPointer();

         const FractionalDataType * const pValidationPredictionEnd = pValidationPredictorScores + cVectorLength * cInstances;

         FractionalDataType sumLogLoss = 0;
         if(IsBinaryClassification(compilerLearningTypeOrCountTargetClasses)) {
            const FractionalDataType smallChangeToPredictorScores = aModelFeatureCombinationUpdateTensor[0];
            do {
               StorageDataTypeCore targetData = *pTargetData;
               // this will apply a small fix to our existing ValidationPredictorScores, either positive or negative, whichever is needed
               const FractionalDataType validationPredictorScores = *pValidationPredictorScores + smallChangeToPredictorScores;
               *pValidationPredictorScores = validationPredictorScores;
               sumLogLoss += EbmStatistics::ComputeClassificationSingleInstanceLogLossBinaryclass(validationPredictorScores, targetData);
               ++pValidationPredictorScores;
               ++pTargetData;
            } while(pValidationPredictionEnd != pValidationPredictorScores);
         } else {
            const FractionalDataType * pValues = aModelFeatureCombinationUpdateTensor;
            do {
               StorageDataTypeCore targetData = *pTargetData;
               FractionalDataType sumExp = 0;
               size_t iVector = 0;
               do {
                  const FractionalDataType smallChangeToPredictorScores = pValues[iVector];
                  // this will apply a small fix to our existing validationPredictorScores, either positive or negative, whichever is needed

                  const FractionalDataType validationPredictorScores = *pValidationPredictorScores + smallChangeToPredictorScores;
                  *pValidationPredictorScores = validationPredictorScores;
                  sumExp += EbmExp(validationPredictorScores);
                  ++pValidationPredictorScores;

                  // TODO : consider replacing iVector with pValidationPredictorScoresInnerEnd
                  ++iVector;
               } while(iVector < cVectorLength);
               // TODO: store the result of std::exp above for the index that we care about above since exp(..) is going to be expensive and probably even more expensive than an unconditional branch
               sumLogLoss += EbmStatistics::ComputeClassificationSingleInstanceLogLossMulticlass(sumExp, pValidationPredictorScores - cVectorLength, targetData);
               ++pTargetData;
            } while(pValidationPredictionEnd != pValidationPredictorScores);
         }
         LOG_0(TraceLevelVerbose, "Exited ValidationSetTargetFeatureLoop - Zero dimensions");
         return sumLogLoss /= pValidationSet->GetCountInstances();
      }
      EBM_ASSERT(false);
   }

   const size_t cItemsPerBitPackDataUnit = pFeatureCombination->m_cItemsPerBitPackDataUnit;
   EBM_ASSERT(1 <= cItemsPerBitPackDataUnit);
   EBM_ASSERT(cItemsPerBitPackDataUnit <= k_cBitsForStorageType);
   const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackDataUnit);
   EBM_ASSERT(1 <= cBitsPerItemMax);
   EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
   const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);
   const StorageDataTypeCore * pInputData = pValidationSet->GetInputDataPointer(pFeatureCombination);

   if(IsRegression(compilerLearningTypeOrCountTargetClasses)) {
      FractionalDataType meanSquareError = 0;
      FractionalDataType * pResidualError = pValidationSet->GetResidualPointer();

      // this shouldn't overflow since we're accessing existing memory
      const FractionalDataType * const pResidualErrorTrueEnd = pResidualError + cVectorLength * cInstances;
      const FractionalDataType * pResidualErrorExit = pResidualErrorTrueEnd;
      size_t cItemsRemaining = cInstances;
      if(cInstances <= cItemsPerBitPackDataUnit) {
         goto one_last_loop_regression;
      }
      pResidualErrorExit = pResidualErrorTrueEnd - cVectorLength * ((cInstances - 1) % cItemsPerBitPackDataUnit + 1);
      EBM_ASSERT(pResidualError < pResidualErrorExit);
      EBM_ASSERT(pResidualErrorExit < pResidualErrorTrueEnd);

      do {
         cItemsRemaining = cItemsPerBitPackDataUnit;
         // TODO : jumping back into this loop and changing cItemsRemaining to a dynamic value that isn't compile time determinable
         // causes this function to NOT be optimized as much as it could if we had two separate loops.  We're just trying this out for now though
      one_last_loop_regression:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iTensorBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            const size_t iTensorBin = maskBits & iTensorBinCombined;
            const FractionalDataType smallChangeToPrediction = aModelFeatureCombinationUpdateTensor[iTensorBin * cVectorLength];
            // this will apply a small fix to our existing ValidationPredictorScores, either positive or negative, whichever is needed
            const FractionalDataType residualError = EbmStatistics::ComputeRegressionResidualError(*pResidualError - smallChangeToPrediction);
            meanSquareError += EbmStatistics::ComputeRegressionSingleInstanceMeanSquaredError(residualError);
            *pResidualError = residualError;
            ++pResidualError;

            iTensorBinCombined >>= cBitsPerItemMax;
            // TODO : try replacing cItemsRemaining with a pResidualErrorInnerLoopEnd which eliminates one subtact operation, but might make it harder for the compiler to optimize the loop away
            --cItemsRemaining;
         } while(0 != cItemsRemaining);
      } while(pResidualErrorExit != pResidualError);

      // first time through?
      if(pResidualErrorTrueEnd != pResidualError) {
         EBM_ASSERT(0 == (pResidualErrorTrueEnd - pResidualError) % cVectorLength);
         cItemsRemaining = (pResidualErrorTrueEnd - pResidualError) / cVectorLength;
         EBM_ASSERT(0 < cItemsRemaining);
         EBM_ASSERT(cItemsRemaining <= cItemsPerBitPackDataUnit);

         pResidualErrorExit = pResidualErrorTrueEnd;

         goto one_last_loop_regression;
      }

      meanSquareError /= pValidationSet->GetCountInstances();
      LOG_0(TraceLevelVerbose, "Exited ValidationSetTargetFeatureLoop");
      return meanSquareError;
   } else {
      EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));
      FractionalDataType sumLogLoss = 0;

      const StorageDataTypeCore * pTargetData = pValidationSet->GetTargetDataPointer();
      FractionalDataType * pValidationPredictorScores = pValidationSet->GetPredictorScores();

      // this shouldn't overflow since we're accessing existing memory
      const FractionalDataType * const pValidationPredictorScoresTrueEnd = pValidationPredictorScores + cVectorLength * cInstances;
      const FractionalDataType * pValidationPredictorScoresExit = pValidationPredictorScoresTrueEnd;
      size_t cItemsRemaining = cInstances;
      if(cInstances <= cItemsPerBitPackDataUnit) {
         goto one_last_loop_classification;
      }
      pValidationPredictorScoresExit = pValidationPredictorScoresTrueEnd - cVectorLength * ((cInstances - 1) % cItemsPerBitPackDataUnit + 1);
      EBM_ASSERT(pValidationPredictorScores < pValidationPredictorScoresExit);
      EBM_ASSERT(pValidationPredictorScoresExit < pValidationPredictorScoresTrueEnd);

      do {
         cItemsRemaining = cItemsPerBitPackDataUnit;
         // TODO : jumping back into this loop and changing cItemsRemaining to a dynamic value that isn't compile time determinable
         // causes this function to NOT be optimized as much as it could if we had two separate loops.  We're just trying this out for now though
      one_last_loop_classification:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iTensorBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            StorageDataTypeCore targetData = *pTargetData;

            const size_t iTensorBin = maskBits & iTensorBinCombined;
            const FractionalDataType * pValues = &aModelFeatureCombinationUpdateTensor[iTensorBin * cVectorLength];

            if(IsBinaryClassification(compilerLearningTypeOrCountTargetClasses)) {
               const FractionalDataType smallChangeToPredictorScores = pValues[0];
               // this will apply a small fix to our existing ValidationPredictorScores, either positive or negative, whichever is needed
               const FractionalDataType validationPredictorScores = *pValidationPredictorScores + smallChangeToPredictorScores;
               *pValidationPredictorScores = validationPredictorScores;
               sumLogLoss += EbmStatistics::ComputeClassificationSingleInstanceLogLossBinaryclass(validationPredictorScores, targetData);
               ++pValidationPredictorScores;
            } else {
               FractionalDataType sumExp = 0;
               size_t iVector = 0;
               do {
                  const FractionalDataType smallChangeToPredictorScores = pValues[iVector];
                  // this will apply a small fix to our existing validationPredictorScores, either positive or negative, whichever is needed

                  const FractionalDataType validationPredictorScores = *pValidationPredictorScores + smallChangeToPredictorScores;
                  *pValidationPredictorScores = validationPredictorScores;
                  sumExp += EbmExp(validationPredictorScores);
                  ++pValidationPredictorScores;

                  // TODO : consider replacing iVector with pValidationPredictorScoresInnerEnd
                  ++iVector;
               } while(iVector < cVectorLength);
               // TODO: store the result of std::exp above for the index that we care about above since exp(..) is going to be expensive and probably even more expensive than an unconditional branch
               sumLogLoss += EbmStatistics::ComputeClassificationSingleInstanceLogLossMulticlass(sumExp, pValidationPredictorScores - cVectorLength, targetData);
            }
            ++pTargetData;

            iTensorBinCombined >>= cBitsPerItemMax;
            // TODO : try replacing cItemsRemaining with a pResidualErrorInnerLoopEnd which eliminates one subtact operation, but might make it harder for the compiler to optimize the loop away
            --cItemsRemaining;
         } while(0 != cItemsRemaining);
      } while(pValidationPredictorScoresExit != pValidationPredictorScores);

      // first time through?
      if(pValidationPredictorScoresTrueEnd != pValidationPredictorScores) {
         EBM_ASSERT(0 == (pValidationPredictorScoresTrueEnd - pValidationPredictorScores) % cVectorLength);
         cItemsRemaining = (pValidationPredictorScoresTrueEnd - pValidationPredictorScores) / cVectorLength;
         EBM_ASSERT(0 < cItemsRemaining);
         EBM_ASSERT(cItemsRemaining <= cItemsPerBitPackDataUnit);

         pValidationPredictorScoresExit = pValidationPredictorScoresTrueEnd;

         goto one_last_loop_classification;
      }

      LOG_0(TraceLevelVerbose, "Exited ValidationSetTargetFeatureLoop");
      return sumLogLoss /= pValidationSet->GetCountInstances();
   }
}

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
template<unsigned int cInputBits, ptrdiff_t compilerLearningTypeOrCountTargetClasses>
static FractionalDataType ValidationSetInputFeatureLoop(const FeatureCombinationCore * const pFeatureCombination, DataSetByFeatureCombination * const pValidationSet, const FractionalDataType * const aModelFeatureCombinationUpdateTensor, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses) {
   if(static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses) <= 1 << 1) {
      return ValidationSetTargetFeatureLoop<cInputBits, 1, compilerLearningTypeOrCountTargetClasses>(pFeatureCombination, pValidationSet, aModelFeatureCombinationUpdateTensor, runtimeLearningTypeOrCountTargetClasses);
   } else if(static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses) <= 1 << 2) {
      return ValidationSetTargetFeatureLoop<cInputBits, 2, compilerLearningTypeOrCountTargetClasses>(pFeatureCombination, pValidationSet, aModelFeatureCombinationUpdateTensor, runtimeLearningTypeOrCountTargetClasses);
   } else if(static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses) <= 1 << 4) {
      return ValidationSetTargetFeatureLoop<cInputBits, 4, compilerLearningTypeOrCountTargetClasses>(pFeatureCombination, pValidationSet, aModelFeatureCombinationUpdateTensor, runtimeLearningTypeOrCountTargetClasses);
   } else if(static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses) <= 1 << 8) {
      return ValidationSetTargetFeatureLoop<cInputBits, 8, compilerLearningTypeOrCountTargetClasses>(pFeatureCombination, pValidationSet, aModelFeatureCombinationUpdateTensor, runtimeLearningTypeOrCountTargetClasses);
   } else if(static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses) <= 1 << 16) {
      return ValidationSetTargetFeatureLoop<cInputBits, 16, compilerLearningTypeOrCountTargetClasses>(pFeatureCombination, pValidationSet, aModelFeatureCombinationUpdateTensor, runtimeLearningTypeOrCountTargetClasses);
   } else if(static_cast<uint64_t>(runtimeLearningTypeOrCountTargetClasses) <= uint64_t { 1 } << 32) {
      // if this is a 32 bit system, then m_cBins can't be 0x100000000 or above, because we would have checked that when converting the 64 bit numbers into size_t, and m_cBins will be promoted to a 64 bit number for the above comparison
      // if this is a 64 bit system, then this comparison is fine

      // TODO : perhaps we should change m_cBins into m_iBinMax so that we don't need to do the above promotion to 64 bits.. we can make it <= 0xFFFFFFFF.  Write a function to fill the lowest bits with ones for any number of bits

      return ValidationSetTargetFeatureLoop<cInputBits, 32, compilerLearningTypeOrCountTargetClasses>(pFeatureCombination, pValidationSet, aModelFeatureCombinationUpdateTensor, runtimeLearningTypeOrCountTargetClasses);
   } else {
      // our interface doesn't allow more than 64 bits, so even if size_t was bigger then we don't need to examine higher
      static_assert(63 == CountBitsRequiredPositiveMax<IntegerDataType>(), "");
      return ValidationSetTargetFeatureLoop<cInputBits, 64, compilerLearningTypeOrCountTargetClasses>(pFeatureCombination, pValidationSet, aModelFeatureCombinationUpdateTensor, runtimeLearningTypeOrCountTargetClasses);
   }
}

#ifndef NDEBUG
void CheckTargets(const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, const size_t cInstances, const void * const aTargets) {
   if(0 != cInstances) {
      if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
         const IntegerDataType * pTarget = static_cast<const IntegerDataType *>(aTargets);
         const IntegerDataType * const pTargetEnd = pTarget + cInstances;
         do {
            const IntegerDataType target = *pTarget;
            EBM_ASSERT(0 <= target);
            EBM_ASSERT((IsNumberConvertable<ptrdiff_t, IntegerDataType>(target))); // data must be lower than runtimeLearningTypeOrCountTargetClasses and runtimeLearningTypeOrCountTargetClasses fits into a ptrdiff_t which we checked earlier
            EBM_ASSERT(static_cast<ptrdiff_t>(target) < runtimeLearningTypeOrCountTargetClasses);
            ++pTarget;
         } while(pTargetEnd != pTarget);
      } else {
         EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
         const FractionalDataType * pTarget = static_cast<const FractionalDataType *>(aTargets);
         const FractionalDataType * const pTargetEnd = pTarget + cInstances;
         do {
            const FractionalDataType target = *pTarget;
            EBM_ASSERT(!std::isnan(target));
            EBM_ASSERT(!std::isinf(target));
            ++pTarget;
         } while(pTargetEnd != pTarget);
      }
   }
}
#endif // NDEBUG

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
EbmBoostingState * AllocateCoreBoosting(const IntegerDataType randomSeed, const IntegerDataType countFeatures, const EbmCoreFeature * const features, const IntegerDataType countFeatureCombinations, const EbmCoreFeatureCombination * const featureCombinations, const IntegerDataType * const featureCombinationIndexes, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, const IntegerDataType countTrainingInstances, const void * const trainingTargets, const IntegerDataType * const trainingBinnedData, const FractionalDataType * const trainingPredictorScores, const IntegerDataType countValidationInstances, const void * const validationTargets, const IntegerDataType * const validationBinnedData, const FractionalDataType * const validationPredictorScores, const IntegerDataType countInnerBags) {
   // TODO : give AllocateCoreBoosting the same calling parameter order as InitializeBoostingClassification
   // TODO: turn these EBM_ASSERTS into log errors!!  Small checks like this of our wrapper's inputs hardly cost anything, and catch issues faster

   // randomSeed can be any value
   EBM_ASSERT(0 <= countFeatures);
   EBM_ASSERT(0 == countFeatures || nullptr != features);
   EBM_ASSERT(0 <= countFeatureCombinations);
   EBM_ASSERT(0 == countFeatureCombinations || nullptr != featureCombinations);
   // featureCombinationIndexes -> it's legal for featureCombinationIndexes to be nullptr if there are no features indexed by our featureCombinations.  FeatureCombinations can have zero features, so it could be legal for this to be null even if there are featureCombinations
   // countTargetClasses is checked by our caller since it's only valid for classification at this point
   EBM_ASSERT(0 <= countTrainingInstances);
   EBM_ASSERT(0 == countTrainingInstances || nullptr != trainingTargets);
   EBM_ASSERT(0 == countTrainingInstances || 0 == countFeatures || nullptr != trainingBinnedData);
   // trainingPredictorScores can be null
   EBM_ASSERT(0 <= countValidationInstances);
   EBM_ASSERT(0 == countValidationInstances || nullptr != validationTargets);
   EBM_ASSERT(0 == countValidationInstances || 0 == countFeatures || nullptr != validationBinnedData);
   // validationPredictorScores can be null
   EBM_ASSERT(0 <= countInnerBags); // 0 means use the full set (good value).  1 means make a single bag (this is useless but allowed for comparison purposes).  2+ are good numbers of bag

   if(!IsNumberConvertable<size_t, IntegerDataType>(countFeatures)) {
      LOG_0(TraceLevelWarning, "WARNING AllocateCore !IsNumberConvertable<size_t, IntegerDataType>(countFeatures)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntegerDataType>(countFeatureCombinations)) {
      LOG_0(TraceLevelWarning, "WARNING AllocateCore !IsNumberConvertable<size_t, IntegerDataType>(countFeatureCombinations)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntegerDataType>(countTrainingInstances)) {
      LOG_0(TraceLevelWarning, "WARNING AllocateCore !IsNumberConvertable<size_t, IntegerDataType>(countTrainingInstances)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntegerDataType>(countValidationInstances)) {
      LOG_0(TraceLevelWarning, "WARNING AllocateCore !IsNumberConvertable<size_t, IntegerDataType>(countValidationInstances)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntegerDataType>(countInnerBags)) {
      LOG_0(TraceLevelWarning, "WARNING AllocateCore !IsNumberConvertable<size_t, IntegerDataType>(countInnerBags)");
      return nullptr;
   }

   size_t cFeatures = static_cast<size_t>(countFeatures);
   size_t cFeatureCombinations = static_cast<size_t>(countFeatureCombinations);
   size_t cTrainingInstances = static_cast<size_t>(countTrainingInstances);
   size_t cValidationInstances = static_cast<size_t>(countValidationInstances);
   size_t cInnerBags = static_cast<size_t>(countInnerBags);

   size_t cVectorLength = GetVectorLengthFlatCore(runtimeLearningTypeOrCountTargetClasses);

   if(IsMultiplyError(cVectorLength, cTrainingInstances)) {
      LOG_0(TraceLevelWarning, "WARNING AllocateCore IsMultiplyError(cVectorLength, cTrainingInstances)");
      return nullptr;
   }
   if(IsMultiplyError(cVectorLength, cValidationInstances)) {
      LOG_0(TraceLevelWarning, "WARNING AllocateCore IsMultiplyError(cVectorLength, cValidationInstances)");
      return nullptr;
   }

#ifndef NDEBUG
   CheckTargets(runtimeLearningTypeOrCountTargetClasses, cTrainingInstances, trainingTargets);
   CheckTargets(runtimeLearningTypeOrCountTargetClasses, cValidationInstances, validationTargets);
#endif // NDEBUG

   LOG_0(TraceLevelInfo, "Entered EbmBoostingState");
   EbmBoostingState * const pEbmBoostingState = new (std::nothrow) EbmBoostingState(runtimeLearningTypeOrCountTargetClasses, cFeatures, cFeatureCombinations, cInnerBags, randomSeed);
   LOG_N(TraceLevelInfo, "Exited EbmBoostingState %p", static_cast<void *>(pEbmBoostingState));
   if(UNLIKELY(nullptr == pEbmBoostingState)) {
      LOG_0(TraceLevelWarning, "WARNING AllocateCore nullptr == pEbmBoostingState");
      return nullptr;
   }
   if(UNLIKELY(pEbmBoostingState->Initialize(features, featureCombinations, featureCombinationIndexes, cTrainingInstances, trainingTargets, trainingBinnedData, trainingPredictorScores, cValidationInstances, validationTargets, validationBinnedData, validationPredictorScores))) {
      LOG_0(TraceLevelWarning, "WARNING AllocateCore pEbmBoostingState->Initialize");
      delete pEbmBoostingState;
      return nullptr;
   }
   return pEbmBoostingState;
}

EBMCORE_IMPORT_EXPORT_BODY PEbmBoosting EBMCORE_CALLING_CONVENTION InitializeBoostingClassification(
   IntegerDataType countTargetClasses,
   IntegerDataType countFeatures,
   const EbmCoreFeature * features,
   IntegerDataType countFeatureCombinations,
   const EbmCoreFeatureCombination * featureCombinations,
   const IntegerDataType * featureCombinationIndexes,
   IntegerDataType countTrainingInstances,
   const IntegerDataType * trainingBinnedData,
   const IntegerDataType * trainingTargets,
   const FractionalDataType * trainingPredictorScores,
   IntegerDataType countValidationInstances,
   const IntegerDataType * validationBinnedData,
   const IntegerDataType * validationTargets,
   const FractionalDataType * validationPredictorScores,
   IntegerDataType countInnerBags,
   IntegerDataType randomSeed
) {
   LOG_N(TraceLevelInfo, "Entered InitializeBoostingClassification: countTargetClasses=%" IntegerDataTypePrintf ", countFeatures=%" IntegerDataTypePrintf ", features=%p, countFeatureCombinations=%" IntegerDataTypePrintf ", featureCombinations=%p, featureCombinationIndexes=%p, countTrainingInstances=%" IntegerDataTypePrintf ", trainingBinnedData=%p, trainingTargets=%p, trainingPredictorScores=%p, countValidationInstances=%" IntegerDataTypePrintf ", validationBinnedData=%p, validationTargets=%p, validationPredictorScores=%p, countInnerBags=%" IntegerDataTypePrintf ", randomSeed=%" IntegerDataTypePrintf, countTargetClasses, countFeatures, static_cast<const void *>(features), countFeatureCombinations, static_cast<const void *>(featureCombinations), static_cast<const void *>(featureCombinationIndexes), countTrainingInstances, static_cast<const void *>(trainingBinnedData), static_cast<const void *>(trainingTargets), static_cast<const void *>(trainingPredictorScores), countValidationInstances, static_cast<const void *>(validationBinnedData), static_cast<const void *>(validationTargets), static_cast<const void *>(validationPredictorScores), countInnerBags, randomSeed);
   if(countTargetClasses < 0) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification countTargetClasses can't be negative");
      return nullptr;
   }
   if(0 == countTargetClasses && (0 != countTrainingInstances || 0 != countValidationInstances)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification countTargetClasses can't be zero unless there are no training and no validation cases");
      return nullptr;
   }
   if(!IsNumberConvertable<ptrdiff_t, IntegerDataType>(countTargetClasses)) {
      LOG_0(TraceLevelWarning, "WARNING InitializeBoostingClassification !IsNumberConvertable<ptrdiff_t, IntegerDataType>(countTargetClasses)");
      return nullptr;
   }
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = static_cast<ptrdiff_t>(countTargetClasses);
   const PEbmBoosting pEbmBoosting = reinterpret_cast<PEbmBoosting>(AllocateCoreBoosting(randomSeed, countFeatures, features, countFeatureCombinations, featureCombinations, featureCombinationIndexes, runtimeLearningTypeOrCountTargetClasses, countTrainingInstances, trainingTargets, trainingBinnedData, trainingPredictorScores, countValidationInstances, validationTargets, validationBinnedData, validationPredictorScores, countInnerBags));
   LOG_N(TraceLevelInfo, "Exited InitializeBoostingClassification %p", static_cast<void *>(pEbmBoosting));
   return pEbmBoosting;
}

EBMCORE_IMPORT_EXPORT_BODY PEbmBoosting EBMCORE_CALLING_CONVENTION InitializeBoostingRegression(
   IntegerDataType countFeatures,
   const EbmCoreFeature * features,
   IntegerDataType countFeatureCombinations,
   const EbmCoreFeatureCombination * featureCombinations,
   const IntegerDataType * featureCombinationIndexes,
   IntegerDataType countTrainingInstances,
   const IntegerDataType * trainingBinnedData,
   const FractionalDataType * trainingTargets,
   const FractionalDataType * trainingPredictorScores,
   IntegerDataType countValidationInstances,
   const IntegerDataType * validationBinnedData,
   const FractionalDataType * validationTargets,
   const FractionalDataType * validationPredictorScores,
   IntegerDataType countInnerBags,
   IntegerDataType randomSeed
) {
   LOG_N(TraceLevelInfo, "Entered InitializeBoostingRegression: countFeatures=%" IntegerDataTypePrintf ", features=%p, countFeatureCombinations=%" IntegerDataTypePrintf ", featureCombinations=%p, featureCombinationIndexes=%p, countTrainingInstances=%" IntegerDataTypePrintf ", trainingBinnedData=%p, trainingTargets=%p, trainingPredictorScores=%p, countValidationInstances=%" IntegerDataTypePrintf ", validationBinnedData=%p, validationTargets=%p, validationPredictorScores=%p, countInnerBags=%" IntegerDataTypePrintf ", randomSeed=%" IntegerDataTypePrintf, countFeatures, static_cast<const void *>(features), countFeatureCombinations, static_cast<const void *>(featureCombinations), static_cast<const void *>(featureCombinationIndexes), countTrainingInstances, static_cast<const void *>(trainingBinnedData), static_cast<const void *>(trainingTargets), static_cast<const void *>(trainingPredictorScores), countValidationInstances, static_cast<const void *>(validationBinnedData), static_cast<const void *>(validationTargets), static_cast<const void *>(validationPredictorScores), countInnerBags, randomSeed);
   const PEbmBoosting pEbmBoosting = reinterpret_cast<PEbmBoosting>(AllocateCoreBoosting(randomSeed, countFeatures, features, countFeatureCombinations, featureCombinations, featureCombinationIndexes, k_Regression, countTrainingInstances, trainingTargets, trainingBinnedData, trainingPredictorScores, countValidationInstances, validationTargets, validationBinnedData, validationPredictorScores, countInnerBags));
   LOG_N(TraceLevelInfo, "Exited InitializeBoostingRegression %p", static_cast<void *>(pEbmBoosting));
   return pEbmBoosting;
}

template<bool bClassification>
EBM_INLINE CachedBoostingThreadResources<bClassification> * GetCachedThreadResources(EbmBoostingState * pEbmBoostingState);
template<>
EBM_INLINE CachedBoostingThreadResources<true> * GetCachedThreadResources<true>(EbmBoostingState * pEbmBoostingState) {
   return &pEbmBoostingState->m_cachedThreadResourcesUnion.classification;
}
template<>
EBM_INLINE CachedBoostingThreadResources<false> * GetCachedThreadResources<false>(EbmBoostingState * pEbmBoostingState) {
   return &pEbmBoostingState->m_cachedThreadResourcesUnion.regression;
}

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
static FractionalDataType * GenerateModelFeatureCombinationUpdatePerTargetClasses(EbmBoostingState * const pEbmBoostingState, const size_t iFeatureCombination, const FractionalDataType learningRate, const size_t cTreeSplitsMax, const size_t cInstancesRequiredForParentSplitMin, const size_t cInstancesRequiredForChildSplitMin, const FractionalDataType * const aTrainingWeights, const FractionalDataType * const aValidationWeights, FractionalDataType * const pGainReturn) {
   // TODO remove this after we use aTrainingWeights and aValidationWeights into the GenerateModelFeatureCombinationUpdatePerTargetClasses function
   UNUSED(aTrainingWeights);
   UNUSED(aValidationWeights);

   LOG_0(TraceLevelVerbose, "Entered GenerateModelFeatureCombinationUpdatePerTargetClasses");

   if(nullptr != pGainReturn) {
      *pGainReturn = 0; // always set this, even on errors.  We might as well do it here at the top
   }

   const size_t cSamplingSetsAfterZero = (0 == pEbmBoostingState->m_cSamplingSets) ? 1 : pEbmBoostingState->m_cSamplingSets;
   CachedBoostingThreadResources<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pCachedThreadResources = GetCachedThreadResources<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pEbmBoostingState);
   const FeatureCombinationCore * const pFeatureCombination = pEbmBoostingState->m_apFeatureCombinations[iFeatureCombination];
   const size_t cDimensions = pFeatureCombination->m_cFeatures;

   pEbmBoostingState->m_pSmallChangeToModelAccumulatedFromSamplingSets->SetCountDimensions(cDimensions);
   pEbmBoostingState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Reset();

   // if pEbmBoostingState->m_apSamplingSets is nullptr, then we should have zero training instances
   // we can't be partially constructed here since then we wouldn't have returned our state pointer to our caller
   EBM_ASSERT(!pEbmBoostingState->m_apSamplingSets == !pEbmBoostingState->m_pTrainingSet); // m_pTrainingSet and m_apSamplingSets should be the same null-ness in that they should either both be null or both be non-null (although different non-null values)
   FractionalDataType totalGain = 0;
   if(nullptr != pEbmBoostingState->m_apSamplingSets) {
      pEbmBoostingState->m_pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDimensions(cDimensions);

      for(size_t iSamplingSet = 0; iSamplingSet < cSamplingSetsAfterZero; ++iSamplingSet) {
         FractionalDataType gain = 0;
         if(0 == pFeatureCombination->m_cFeatures) {
            if(BoostZeroDimensional<compilerLearningTypeOrCountTargetClasses>(pCachedThreadResources, pEbmBoostingState->m_apSamplingSets[iSamplingSet], pEbmBoostingState->m_pSmallChangeToModelOverwriteSingleSamplingSet, pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses)) {
               return nullptr;
            }
         } else if(1 == pFeatureCombination->m_cFeatures) {
            if(BoostSingleDimensional<compilerLearningTypeOrCountTargetClasses>(&pEbmBoostingState->m_randomStream, pCachedThreadResources, pEbmBoostingState->m_apSamplingSets[iSamplingSet], pFeatureCombination, cTreeSplitsMax, cInstancesRequiredForParentSplitMin, cInstancesRequiredForChildSplitMin, pEbmBoostingState->m_pSmallChangeToModelOverwriteSingleSamplingSet, &gain, pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses)) {
               return nullptr;
            }
         } else {
            if(BoostMultiDimensional<compilerLearningTypeOrCountTargetClasses, 0>(pCachedThreadResources, pEbmBoostingState->m_apSamplingSets[iSamplingSet], pFeatureCombination, pEbmBoostingState->m_pSmallChangeToModelOverwriteSingleSamplingSet, cInstancesRequiredForChildSplitMin, pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses)) {
               return nullptr;
            }
         }
         totalGain += gain;
         // TODO : when we thread this code, let's have each thread take a lock and update the combined line segment.  They'll each do it while the others are working, so there should be no blocking and our final result won't require adding by the main thread
         if(pEbmBoostingState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Add(*pEbmBoostingState->m_pSmallChangeToModelOverwriteSingleSamplingSet)) {
            return nullptr;
         }
      }
      totalGain /= static_cast<FractionalDataType>(cSamplingSetsAfterZero);

      LOG_0(TraceLevelVerbose, "GenerateModelFeatureCombinationUpdatePerTargetClasses done sampling set loop");

      // we need to divide by the number of sampling sets that we constructed this from.
      // We also need to slow down our growth so that the more relevant Features get a chance to grow first so we multiply by a user defined learning rate
      if(IsClassification(compilerLearningTypeOrCountTargetClasses)) {
#ifdef EXPAND_BINARY_LOGITS
         constexpr bool bExpandBinaryLogits = true;
#else // EXPAND_BINARY_LOGITS
         constexpr bool bExpandBinaryLogits = false;
#endif // EXPAND_BINARY_LOGITS

         //if(0 <= k_iZeroResidual || ptrdiff_t { 2 } == pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses && bExpandBinaryLogits) {
         //   EBM_ASSERT(ptrdiff_t { 2 } <= pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses);
         //   // TODO : for classification with residual zeroing, is our learning rate essentially being inflated as pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses goes up?  If so, maybe we should divide by pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses here to keep learning rates as equivalent as possible..  Actually, I think the real solution here is that 
         //   pEbmBoostingState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Multiply(learningRate / cSamplingSetsAfterZero * (pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses - 1) / pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses);
         //} else {
         //   // TODO : for classification, is our learning rate essentially being inflated as pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses goes up?  If so, maybe we should divide by pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses here to keep learning rates equivalent as possible
         //   pEbmBoostingState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Multiply(learningRate / cSamplingSetsAfterZero);
         //}

         constexpr bool bDividing = bExpandBinaryLogits && ptrdiff_t { 2 } == compilerLearningTypeOrCountTargetClasses;
         if(bDividing) {
            pEbmBoostingState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Multiply(learningRate / cSamplingSetsAfterZero / 2);
         } else {
            pEbmBoostingState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Multiply(learningRate / cSamplingSetsAfterZero);
         }
      } else {
         pEbmBoostingState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Multiply(learningRate / cSamplingSetsAfterZero);
      }
   }

   if(0 != cDimensions) {
      // pEbmBoostingState->m_pSmallChangeToModelAccumulatedFromSamplingSets was reset above, so it isn't expanded.  We want to expand it before calling ValidationSetInputFeatureLoop so that we can more efficiently lookup the results by index rather than do a binary search
      size_t acDivisionIntegersEnd[k_cDimensionsMax];
      size_t iDimension = 0;
      do {
         acDivisionIntegersEnd[iDimension] = ARRAY_TO_POINTER_CONST(pFeatureCombination->m_FeatureCombinationEntry)[iDimension].m_pFeature->m_cBins;
         ++iDimension;
      } while(iDimension < cDimensions);
      if(pEbmBoostingState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Expand(acDivisionIntegersEnd)) {
         return nullptr;
      }
   }

   if(nullptr != pGainReturn) {
      *pGainReturn = totalGain;
   }

   LOG_0(TraceLevelVerbose, "Exited GenerateModelFeatureCombinationUpdatePerTargetClasses");
   return pEbmBoostingState->m_pSmallChangeToModelAccumulatedFromSamplingSets->m_aValues;
}

template<ptrdiff_t possibleCompilerLearningTypeOrCountTargetClasses>
EBM_INLINE FractionalDataType * CompilerRecursiveGenerateModelFeatureCombinationUpdate(const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, EbmBoostingState * const pEbmBoostingState, const size_t iFeatureCombination, const FractionalDataType learningRate, const size_t cTreeSplitsMax, const size_t cInstancesRequiredForParentSplitMin, const size_t cInstancesRequiredForChildSplitMin, const FractionalDataType * const aTrainingWeights, const FractionalDataType * const aValidationWeights, FractionalDataType * const pGainReturn) {
   static_assert(IsClassification(possibleCompilerLearningTypeOrCountTargetClasses), "possibleCompilerLearningTypeOrCountTargetClasses needs to be a classification");
   EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
   if(possibleCompilerLearningTypeOrCountTargetClasses == runtimeLearningTypeOrCountTargetClasses) {
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);
      return GenerateModelFeatureCombinationUpdatePerTargetClasses<possibleCompilerLearningTypeOrCountTargetClasses>(pEbmBoostingState, iFeatureCombination, learningRate, cTreeSplitsMax, cInstancesRequiredForParentSplitMin, cInstancesRequiredForChildSplitMin, aTrainingWeights, aValidationWeights, pGainReturn);
   } else {
      return CompilerRecursiveGenerateModelFeatureCombinationUpdate<possibleCompilerLearningTypeOrCountTargetClasses + 1>(runtimeLearningTypeOrCountTargetClasses, pEbmBoostingState, iFeatureCombination, learningRate, cTreeSplitsMax, cInstancesRequiredForParentSplitMin, cInstancesRequiredForChildSplitMin, aTrainingWeights, aValidationWeights, pGainReturn);
   }
}

template<>
EBM_INLINE FractionalDataType * CompilerRecursiveGenerateModelFeatureCombinationUpdate<k_cCompilerOptimizedTargetClassesMax + 1>(const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, EbmBoostingState * const pEbmBoostingState, const size_t iFeatureCombination, const FractionalDataType learningRate, const size_t cTreeSplitsMax, const size_t cInstancesRequiredForParentSplitMin, const size_t cInstancesRequiredForChildSplitMin, const FractionalDataType * const aTrainingWeights, const FractionalDataType * const aValidationWeights, FractionalDataType * const pGainReturn) {
   UNUSED(runtimeLearningTypeOrCountTargetClasses);
   // it is logically possible, but uninteresting to have a classification with 1 target class, so let our runtime system handle those unlikley and uninteresting cases
   static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");
   EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
   EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < runtimeLearningTypeOrCountTargetClasses);
   return GenerateModelFeatureCombinationUpdatePerTargetClasses<k_DynamicClassification>(pEbmBoostingState, iFeatureCombination, learningRate, cTreeSplitsMax, cInstancesRequiredForParentSplitMin, cInstancesRequiredForChildSplitMin, aTrainingWeights, aValidationWeights, pGainReturn);
}

// we made this a global because if we had put this variable inside the EbmBoostingState object, then we would need to dereference that before getting the count.  By making this global we can send a log message incase a bad EbmBoostingState object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more times than desired, but we can live with that
static unsigned int g_cLogGenerateModelFeatureCombinationUpdateParametersMessages = 10;

// TODO : change this so that our caller allocates the memory that contains the update, but this is complicated in various ways
//        we don't want to just copy the internal tensor into the memory region that our caller provides, and we want to work with
//        compressed representations of the SegmentedTensor object while we're building it, so we'll work within the memory the caller
//        provides, but that means we'll potentially need more memory than the full tensor, and we'll need to put some header info
//        at the start, so the caller can't treat this memory as a pure tensor.
//        So:
//          1) provide a function that returns the maximum memory needed.  A smart caller will call this once on each feature_combination, choose the max and allocate it once
//          2) return a compressed complete SegmentedTensor to the caller inside an opaque memory region (return the exact size that we require to the caller for copying)
//          3) if caller wants a simplified tensor, then they call a separate function that expands the tensor and returns a pointer to the memory inside the opaque object
//          4) ApplyModelFeatureCombinationUpdate will take an opaque SegmentedTensor, and expand it if needed
//        The benefit of returning a compressed object is that we don't have to do the work of expanding it if the caller decides not to use it (which might happen in greedy algorithms)
//        The other benefit of returning a compressed object is that our caller can store/copy it faster
//        The other benefit of returning a compressed object is that it can be copied from process to process faster
//        Lastly, with the memory allocated by our caller, we can call GenerateModelFeatureCombinationUpdate in parallel on multiple feature_combinations.  Right now you can't call it in parallel since we're updating our internal single tensor

EBMCORE_IMPORT_EXPORT_BODY FractionalDataType * EBMCORE_CALLING_CONVENTION GenerateModelFeatureCombinationUpdate(
   PEbmBoosting ebmBoosting,
   IntegerDataType indexFeatureCombination,
   FractionalDataType learningRate,
   IntegerDataType countTreeSplitsMax,
   IntegerDataType countInstancesRequiredForParentSplitMin,
   const FractionalDataType * trainingWeights,
   const FractionalDataType * validationWeights,
   FractionalDataType * gainReturn
) {
   LOG_COUNTED_N(&g_cLogGenerateModelFeatureCombinationUpdateParametersMessages, TraceLevelInfo, TraceLevelVerbose, "GenerateModelFeatureCombinationUpdate parameters: ebmBoosting=%p, indexFeatureCombination=%" IntegerDataTypePrintf ", learningRate=%" FractionalDataTypePrintf ", countTreeSplitsMax=%" IntegerDataTypePrintf ", countInstancesRequiredForParentSplitMin=%" IntegerDataTypePrintf ", trainingWeights=%p, validationWeights=%p, gainReturn=%p", static_cast<void *>(ebmBoosting), indexFeatureCombination, learningRate, countTreeSplitsMax, countInstancesRequiredForParentSplitMin, static_cast<const void *>(trainingWeights), static_cast<const void *>(validationWeights), static_cast<void *>(gainReturn));

   EbmBoostingState * pEbmBoostingState = reinterpret_cast<EbmBoostingState *>(ebmBoosting);
   EBM_ASSERT(nullptr != pEbmBoostingState);

   EBM_ASSERT(0 <= indexFeatureCombination);
   EBM_ASSERT((IsNumberConvertable<size_t, IntegerDataType>(indexFeatureCombination))); // we wouldn't have allowed the creation of an feature set larger than size_t
   size_t iFeatureCombination = static_cast<size_t>(indexFeatureCombination);
   EBM_ASSERT(iFeatureCombination < pEbmBoostingState->m_cFeatureCombinations);
   EBM_ASSERT(nullptr != pEbmBoostingState->m_apFeatureCombinations); // this is true because 0 < pEbmBoostingState->m_cFeatureCombinations since our caller needs to pass in a valid indexFeatureCombination to this function

   LOG_COUNTED_0(&pEbmBoostingState->m_apFeatureCombinations[iFeatureCombination]->m_cLogEnterGenerateModelFeatureCombinationUpdateMessages, TraceLevelInfo, TraceLevelVerbose, "Entered GenerateModelFeatureCombinationUpdate");

   EBM_ASSERT(!std::isnan(learningRate));
   EBM_ASSERT(!std::isinf(learningRate));

   EBM_ASSERT(0 <= countTreeSplitsMax);
   size_t cTreeSplitsMax = static_cast<size_t>(countTreeSplitsMax);
   if(!IsNumberConvertable<size_t, IntegerDataType>(countTreeSplitsMax)) {
      // we can never exceed a size_t number of splits, so let's just set it to the maximum if we were going to overflow because it will generate the same results as if we used the true number
      cTreeSplitsMax = std::numeric_limits<size_t>::max();
   }

   EBM_ASSERT(0 <= countInstancesRequiredForParentSplitMin); // if there is 1 instance, then it can't be split, but we accept this input from our user
   size_t cInstancesRequiredForParentSplitMin = static_cast<size_t>(countInstancesRequiredForParentSplitMin);
   if(!IsNumberConvertable<size_t, IntegerDataType>(countInstancesRequiredForParentSplitMin)) {
      // we can never exceed a size_t number of instances, so let's just set it to the maximum if we were going to overflow because it will generate the same results as if we used the true number
      cInstancesRequiredForParentSplitMin = std::numeric_limits<size_t>::max();
   }

   EBM_ASSERT(nullptr == trainingWeights); // TODO : implement this later
   EBM_ASSERT(nullptr == validationWeights); // TODO : implement this later
   // validationMetricReturn can be nullptr

   FractionalDataType * aModelFeatureCombinationUpdateTensor;
   if(IsClassification(pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses)) {
      if(pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses <= ptrdiff_t { 1 }) {
         // if there is only 1 target class for classification, then we can predict the output with 100% accuracy.  The model is a tensor with zero length array logits, which means for our representation that we have zero items in the array total.
         // since we can predit the output with 100% accuracy, our gain will be 0.
         if(nullptr != gainReturn) {
            *gainReturn = 0;
         }
         LOG_0(TraceLevelWarning, "WARNING GenerateModelFeatureCombinationUpdate pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses <= ptrdiff_t { 1 }");
         return nullptr;
      }
      aModelFeatureCombinationUpdateTensor = CompilerRecursiveGenerateModelFeatureCombinationUpdate<2>(pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses, pEbmBoostingState, iFeatureCombination, learningRate, cTreeSplitsMax, cInstancesRequiredForParentSplitMin, TODO_REMOVE_THIS_DEFAULT_cInstancesRequiredForChildSplitMin, trainingWeights, validationWeights, gainReturn);
   } else {
      EBM_ASSERT(IsRegression(pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses));
      aModelFeatureCombinationUpdateTensor = GenerateModelFeatureCombinationUpdatePerTargetClasses<k_Regression>(pEbmBoostingState, iFeatureCombination, learningRate, cTreeSplitsMax, cInstancesRequiredForParentSplitMin, TODO_REMOVE_THIS_DEFAULT_cInstancesRequiredForChildSplitMin, trainingWeights, validationWeights, gainReturn);
   }

   if(nullptr != gainReturn) {
      // TODO: make all the epsilons a single constant in our header (e-7?)
      EBM_ASSERT(-0.000000001 <= *gainReturn);
      LOG_COUNTED_N(&pEbmBoostingState->m_apFeatureCombinations[iFeatureCombination]->m_cLogExitGenerateModelFeatureCombinationUpdateMessages, TraceLevelInfo, TraceLevelVerbose, "Exited GenerateModelFeatureCombinationUpdate %" FractionalDataTypePrintf, *gainReturn);
   } else {
      LOG_COUNTED_0(&pEbmBoostingState->m_apFeatureCombinations[iFeatureCombination]->m_cLogExitGenerateModelFeatureCombinationUpdateMessages, TraceLevelInfo, TraceLevelVerbose, "Exited GenerateModelFeatureCombinationUpdate no gain");
   }
   if(nullptr == aModelFeatureCombinationUpdateTensor) {
      LOG_0(TraceLevelWarning, "WARNING GenerateModelFeatureCombinationUpdate returned nullptr");
   }
   return aModelFeatureCombinationUpdateTensor;
}

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
static IntegerDataType ApplyModelFeatureCombinationUpdatePerTargetClasses(EbmBoostingState * const pEbmBoostingState, const size_t iFeatureCombination, const FractionalDataType * const aModelFeatureCombinationUpdateTensor, FractionalDataType * const pValidationMetricReturn) {
   LOG_0(TraceLevelVerbose, "Entered ApplyModelFeatureCombinationUpdatePerTargetClasses");

   EBM_ASSERT(nullptr != pEbmBoostingState->m_apCurrentModel); // m_apCurrentModel can be null if there are no featureCombinations (but we have an feature combination index), or if the target has 1 or 0 classes (which we check before calling this function), so it shouldn't be possible to be null
   EBM_ASSERT(nullptr != pEbmBoostingState->m_apBestModel); // m_apCurrentModel can be null if there are no featureCombinations (but we have an feature combination index), or if the target has 1 or 0 classes (which we check before calling this function), so it shouldn't be possible to be null
   EBM_ASSERT(nullptr != aModelFeatureCombinationUpdateTensor); // aModelFeatureCombinationUpdateTensor is checked for nullptr before calling this function   

   pEbmBoostingState->m_apCurrentModel[iFeatureCombination]->AddExpanded(aModelFeatureCombinationUpdateTensor);

   const FeatureCombinationCore * const pFeatureCombination = pEbmBoostingState->m_apFeatureCombinations[iFeatureCombination];

   // if the count of training instances is zero, then pEbmBoostingState->m_pTrainingSet will be nullptr
   if(nullptr != pEbmBoostingState->m_pTrainingSet) {
      // TODO : move the target bits branch inside TrainingSetInputFeatureLoop to here outside instead of the feature combination.  The target # of bits is extremely predictable and so we get to only process one sub branch of code below that.  If we do feature combinations here then we have to keep in instruction cache a whole bunch of options
      TrainingSetInputFeatureLoop<1, compilerLearningTypeOrCountTargetClasses>(pFeatureCombination, pEbmBoostingState->m_pTrainingSet, aModelFeatureCombinationUpdateTensor, pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses);
   }

   FractionalDataType modelMetric = 0;
   if(nullptr != pEbmBoostingState->m_pValidationSet) {
      // if there is no validation set, it's pretty hard to know what the metric we'll get for our validation set
      // we could in theory return anything from zero to infinity or possibly, NaN (probably legally the best), but we return 0 here
      // because we want to kick our caller out of any loop it might be calling us in.  Infinity and NaN are odd values that might cause problems in
      // a caller that isn't expecting those values, so 0 is the safest option, and our caller can avoid the situation entirely by not calling
      // us with zero count validation sets

      // if the count of validation set is zero, then pEbmBoostingState->m_pValidationSet will be nullptr
      // if the count of training instances is zero, don't update the best model (it will stay as all zeros), and we don't need to update our non-existant training set either
      // C++ doesn't define what happens when you compare NaN to annother number.  It probably follows IEEE 754, but it isn't guaranteed, so let's check for zero instances in the validation set this better way   https://stackoverflow.com/questions/31225264/what-is-the-result-of-comparing-a-number-with-nan

      // TODO : move the target bits branch inside TrainingSetInputFeatureLoop to here outside instead of the feature combination.  The target # of bits is extremely predictable and so we get to only process one sub branch of code below that.  If we do feature combinations here then we have to keep in instruction cache a whole bunch of options

      modelMetric = ValidationSetInputFeatureLoop<1, compilerLearningTypeOrCountTargetClasses>(pFeatureCombination, pEbmBoostingState->m_pValidationSet, aModelFeatureCombinationUpdateTensor, pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses);

      // modelMetric is either logloss (classification) or mean squared error (mse) (regression).  In either case we want to minimize it.
      if(LIKELY(modelMetric < pEbmBoostingState->m_bestModelMetric)) {
         // we keep on improving, so this is more likely than not, and we'll exit if it becomes negative a lot
         pEbmBoostingState->m_bestModelMetric = modelMetric;

         // TODO : in the future don't copy over all SegmentedTensors.  We only need to copy the ones that changed, which we can detect if we use a linked list and array lookup for the same data structure
         size_t iModel = 0;
         size_t iModelEnd = pEbmBoostingState->m_cFeatureCombinations;
         do {
            if(pEbmBoostingState->m_apBestModel[iModel]->Copy(*pEbmBoostingState->m_apCurrentModel[iModel])) {
               if(nullptr != pValidationMetricReturn) {
                  *pValidationMetricReturn = 0; // on error set it to something instead of random bits
               }
               LOG_0(TraceLevelVerbose, "Exited ApplyModelFeatureCombinationUpdatePerTargetClasses with memory allocation error in copy");
               return 1;
            }
            ++iModel;
         } while(iModel != iModelEnd);
      }
   }
   if(nullptr != pValidationMetricReturn) {
      *pValidationMetricReturn = modelMetric;
   }

   LOG_0(TraceLevelVerbose, "Exited ApplyModelFeatureCombinationUpdatePerTargetClasses");
   return 0;
}

template<ptrdiff_t possibleCompilerLearningTypeOrCountTargetClasses>
EBM_INLINE IntegerDataType CompilerRecursiveApplyModelFeatureCombinationUpdate(const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, EbmBoostingState * const pEbmBoostingState, const size_t iFeatureCombination, const FractionalDataType * const aModelFeatureCombinationUpdateTensor, FractionalDataType * const pValidationMetricReturn) {
   static_assert(IsClassification(possibleCompilerLearningTypeOrCountTargetClasses), "possibleCompilerLearningTypeOrCountTargetClasses needs to be a classification");
   EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
   if(possibleCompilerLearningTypeOrCountTargetClasses == runtimeLearningTypeOrCountTargetClasses) {
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);
      return ApplyModelFeatureCombinationUpdatePerTargetClasses<possibleCompilerLearningTypeOrCountTargetClasses>(pEbmBoostingState, iFeatureCombination, aModelFeatureCombinationUpdateTensor, pValidationMetricReturn);
   } else {
      return CompilerRecursiveApplyModelFeatureCombinationUpdate<possibleCompilerLearningTypeOrCountTargetClasses + 1>(runtimeLearningTypeOrCountTargetClasses, pEbmBoostingState, iFeatureCombination, aModelFeatureCombinationUpdateTensor, pValidationMetricReturn);
   }
}

template<>
EBM_INLINE IntegerDataType CompilerRecursiveApplyModelFeatureCombinationUpdate<k_cCompilerOptimizedTargetClassesMax + 1>(const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, EbmBoostingState * const pEbmBoostingState, const size_t iFeatureCombination, const FractionalDataType * const aModelFeatureCombinationUpdateTensor, FractionalDataType * const pValidationMetricReturn) {
   UNUSED(runtimeLearningTypeOrCountTargetClasses);
   // it is logically possible, but uninteresting to have a classification with 1 target class, so let our runtime system handle those unlikley and uninteresting cases
   static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");
   EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
   EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < runtimeLearningTypeOrCountTargetClasses);
   return ApplyModelFeatureCombinationUpdatePerTargetClasses<k_DynamicClassification>(pEbmBoostingState, iFeatureCombination, aModelFeatureCombinationUpdateTensor, pValidationMetricReturn);
}

// we made this a global because if we had put this variable inside the EbmBoostingState object, then we would need to dereference that before getting the count.  By making this global we can send a log message incase a bad EbmBoostingState object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more times than desired, but we can live with that
static unsigned int g_cLogApplyModelFeatureCombinationUpdateParametersMessages = 10;

EBMCORE_IMPORT_EXPORT_BODY IntegerDataType EBMCORE_CALLING_CONVENTION ApplyModelFeatureCombinationUpdate(
   PEbmBoosting ebmBoosting,
   IntegerDataType indexFeatureCombination,
   const FractionalDataType * modelFeatureCombinationUpdateTensor,
   FractionalDataType * validationMetricReturn
) {
   LOG_COUNTED_N(&g_cLogApplyModelFeatureCombinationUpdateParametersMessages, TraceLevelInfo, TraceLevelVerbose, "ApplyModelFeatureCombinationUpdate parameters: ebmBoosting=%p, indexFeatureCombination=%" IntegerDataTypePrintf ", modelFeatureCombinationUpdateTensor=%p, validationMetricReturn=%p", static_cast<void *>(ebmBoosting), indexFeatureCombination, static_cast<const void *>(modelFeatureCombinationUpdateTensor), static_cast<void *>(validationMetricReturn));

   EbmBoostingState * pEbmBoostingState = reinterpret_cast<EbmBoostingState *>(ebmBoosting);
   EBM_ASSERT(nullptr != pEbmBoostingState);

   EBM_ASSERT(0 <= indexFeatureCombination);
   EBM_ASSERT((IsNumberConvertable<size_t, IntegerDataType>(indexFeatureCombination))); // we wouldn't have allowed the creation of an feature set larger than size_t
   size_t iFeatureCombination = static_cast<size_t>(indexFeatureCombination);
   EBM_ASSERT(iFeatureCombination < pEbmBoostingState->m_cFeatureCombinations);
   EBM_ASSERT(nullptr != pEbmBoostingState->m_apFeatureCombinations); // this is true because 0 < pEbmBoostingState->m_cFeatureCombinations since our caller needs to pass in a valid indexFeatureCombination to this function

   LOG_COUNTED_0(&pEbmBoostingState->m_apFeatureCombinations[iFeatureCombination]->m_cLogEnterApplyModelFeatureCombinationUpdateMessages, TraceLevelInfo, TraceLevelVerbose, "Entered ApplyModelFeatureCombinationUpdate");

   // modelFeatureCombinationUpdateTensor can be nullptr (then nothing gets updated)
   // validationMetricReturn can be nullptr

   if(nullptr == modelFeatureCombinationUpdateTensor) {
      if(nullptr != validationMetricReturn) {
         *validationMetricReturn = 0;
      }
      LOG_COUNTED_0(&pEbmBoostingState->m_apFeatureCombinations[iFeatureCombination]->m_cLogExitApplyModelFeatureCombinationUpdateMessages, TraceLevelInfo, TraceLevelVerbose, "Exited ApplyModelFeatureCombinationUpdate from null modelFeatureCombinationUpdateTensor");
      return 0;
   }

   IntegerDataType ret;
   if(IsClassification(pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses)) {
      if(pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses <= ptrdiff_t { 1 }) {
         // if there is only 1 target class for classification, then we can predict the output with 100% accuracy.  The model is a tensor with zero length array logits, which means for our representation that we have zero items in the array total.
         // since we can predit the output with 100% accuracy, our log loss is 0.
         if(nullptr != validationMetricReturn) {
            *validationMetricReturn = 0;
         }
         LOG_COUNTED_0(&pEbmBoostingState->m_apFeatureCombinations[iFeatureCombination]->m_cLogExitApplyModelFeatureCombinationUpdateMessages, TraceLevelInfo, TraceLevelVerbose, "Exited ApplyModelFeatureCombinationUpdate from runtimeLearningTypeOrCountTargetClasses <= 1");
         return 0;
      }
      ret = CompilerRecursiveApplyModelFeatureCombinationUpdate<2>(pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses, pEbmBoostingState, iFeatureCombination, modelFeatureCombinationUpdateTensor, validationMetricReturn);
   } else {
      EBM_ASSERT(IsRegression(pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses));
      ret = ApplyModelFeatureCombinationUpdatePerTargetClasses<k_Regression>(pEbmBoostingState, iFeatureCombination, modelFeatureCombinationUpdateTensor, validationMetricReturn);
   }
   if(0 != ret) {
      LOG_N(TraceLevelWarning, "WARNING ApplyModelFeatureCombinationUpdate returned %" IntegerDataTypePrintf, ret);
   }
   if(nullptr != validationMetricReturn) {
      EBM_ASSERT(std::isnan(*validationMetricReturn) || -0.0000001 <= *validationMetricReturn); // both log loss and RMSE need to be above zero
      LOG_COUNTED_N(&pEbmBoostingState->m_apFeatureCombinations[iFeatureCombination]->m_cLogExitApplyModelFeatureCombinationUpdateMessages, TraceLevelInfo, TraceLevelVerbose, "Exited ApplyModelFeatureCombinationUpdate %" FractionalDataTypePrintf, *validationMetricReturn);
   } else {
      LOG_COUNTED_0(&pEbmBoostingState->m_apFeatureCombinations[iFeatureCombination]->m_cLogExitApplyModelFeatureCombinationUpdateMessages, TraceLevelInfo, TraceLevelVerbose, "Exited ApplyModelFeatureCombinationUpdate.  No validation pointer.");
   }
   return ret;
}

EBMCORE_IMPORT_EXPORT_BODY IntegerDataType EBMCORE_CALLING_CONVENTION BoostingStep(
   PEbmBoosting ebmBoosting,
   IntegerDataType indexFeatureCombination,
   FractionalDataType learningRate,
   IntegerDataType countTreeSplitsMax,
   IntegerDataType countInstancesRequiredForParentSplitMin,
   const FractionalDataType * trainingWeights,
   const FractionalDataType * validationWeights,
   FractionalDataType * validationMetricReturn
) {
   EbmBoostingState * pEbmBoostingState = reinterpret_cast<EbmBoostingState *>(ebmBoosting);
   EBM_ASSERT(nullptr != pEbmBoostingState);

   if(IsClassification(pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses)) {
      // we need to special handle this case because if we call GenerateModelUpdate, we'll get back a nullptr for the model (since there is no model) and we'll return 1 from this function.  We'd like to return 0 (success) here, so we handle it ourselves
      if(pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses <= ptrdiff_t { 1 }) {
         // if there is only 1 target class for classification, then we can predict the output with 100% accuracy.  The model is a tensor with zero length array logits, which means for our representation that we have zero items in the array total.
         // since we can predit the output with 100% accuracy, our gain will be 0.
         if(nullptr != validationMetricReturn) {
            *validationMetricReturn = 0;
         }
         LOG_0(TraceLevelWarning, "WARNING BoostingStep pEbmBoostingState->m_runtimeLearningTypeOrCountTargetClasses <= ptrdiff_t { 1 }");
         return 0;
      }
   }

   FractionalDataType gain; // we toss this value, but we still need to get it
   FractionalDataType * pModelFeatureCombinationUpdateTensor = GenerateModelFeatureCombinationUpdate(ebmBoosting, indexFeatureCombination, learningRate, countTreeSplitsMax, countInstancesRequiredForParentSplitMin, trainingWeights, validationWeights, &gain);
   if(nullptr == pModelFeatureCombinationUpdateTensor) {
      EBM_ASSERT(nullptr == validationMetricReturn || 0 == *validationMetricReturn); // rely on GenerateModelUpdate to set the validationMetricReturn to zero on error
      return 1;
   }
   return ApplyModelFeatureCombinationUpdate(ebmBoosting, indexFeatureCombination, pModelFeatureCombinationUpdateTensor, validationMetricReturn);
}

EBMCORE_IMPORT_EXPORT_BODY FractionalDataType * EBMCORE_CALLING_CONVENTION GetBestModelFeatureCombination(
   PEbmBoosting ebmBoosting,
   IntegerDataType indexFeatureCombination
) {
   LOG_N(TraceLevelInfo, "Entered GetBestModelFeatureCombination: ebmBoosting=%p, indexFeatureCombination=%" IntegerDataTypePrintf, static_cast<void *>(ebmBoosting), indexFeatureCombination);

   EbmBoostingState * pEbmBoostingState = reinterpret_cast<EbmBoostingState *>(ebmBoosting);
   EBM_ASSERT(nullptr != pEbmBoostingState);
   EBM_ASSERT(0 <= indexFeatureCombination);
   EBM_ASSERT((IsNumberConvertable<size_t, IntegerDataType>(indexFeatureCombination))); // we wouldn't have allowed the creation of an feature set larger than size_t
   size_t iFeatureCombination = static_cast<size_t>(indexFeatureCombination);
   EBM_ASSERT(iFeatureCombination < pEbmBoostingState->m_cFeatureCombinations);

   if(nullptr == pEbmBoostingState->m_apBestModel) {
      // if pEbmBoostingState->m_apBestModel is nullptr, then either:
      //    1) m_cFeatureCombinations was 0, in which case this function would have undefined behavior since the caller needs to indicate a valid indexFeatureCombination, which is impossible, so we can do anything we like, include the below actions.
      //    2) m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0 (and the learning type is classification), which is legal, which we need to handle here
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.  If there were logits in this model, they'd all be infinity, but you could alternatively think of this model as having zero logits, since the number of logits can be one less than the number of target classification classes.  A model with zero logits is empty, and has zero items.  We want to return a tensor with 0 items in it, so we could either return a pointer to some random memory that can't be accessed, or we can return nullptr.  We return a nullptr in the hopes that our caller will either handle it or throw a nicer exception.

      LOG_0(TraceLevelInfo, "Exited GetBestModelFeatureCombination no model");
      return nullptr;
   }

   SegmentedTensor<ActiveDataType, FractionalDataType> * pBestModel = pEbmBoostingState->m_apBestModel[iFeatureCombination];
   EBM_ASSERT(nullptr != pBestModel);
   EBM_ASSERT(pBestModel->m_bExpanded); // the model should have been expanded at startup
   FractionalDataType * pRet = pBestModel->GetValuePointer();
   EBM_ASSERT(nullptr != pRet);

   LOG_N(TraceLevelInfo, "Exited GetBestModelFeatureCombination %p", static_cast<void *>(pRet));
   return pRet;
}

EBMCORE_IMPORT_EXPORT_BODY FractionalDataType * EBMCORE_CALLING_CONVENTION GetCurrentModelFeatureCombination(
   PEbmBoosting ebmBoosting,
   IntegerDataType indexFeatureCombination
) {
   LOG_N(TraceLevelInfo, "Entered GetCurrentModelFeatureCombination: ebmBoosting=%p, indexFeatureCombination=%" IntegerDataTypePrintf, static_cast<void *>(ebmBoosting), indexFeatureCombination);

   EbmBoostingState * pEbmBoostingState = reinterpret_cast<EbmBoostingState *>(ebmBoosting);
   EBM_ASSERT(nullptr != pEbmBoostingState);
   EBM_ASSERT(0 <= indexFeatureCombination);
   EBM_ASSERT((IsNumberConvertable<size_t, IntegerDataType>(indexFeatureCombination))); // we wouldn't have allowed the creation of an feature set larger than size_t
   size_t iFeatureCombination = static_cast<size_t>(indexFeatureCombination);
   EBM_ASSERT(iFeatureCombination < pEbmBoostingState->m_cFeatureCombinations);

   if(nullptr == pEbmBoostingState->m_apCurrentModel) {
      // if pEbmBoostingState->m_apCurrentModel is nullptr, then either:
      //    1) m_cFeatureCombinations was 0, in which case this function would have undefined behavior since the caller needs to indicate a valid indexFeatureCombination, which is impossible, so we can do anything we like, include the below actions.
      //    2) m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0 (and the learning type is classification), which is legal, which we need to handle here
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.  If there were logits in this model, they'd all be infinity, but you could alternatively think of this model as having zero logits, since the number of logits can be one less than the number of target classification classes.  A model with zero logits is empty, and has zero items.  We want to return a tensor with 0 items in it, so we could either return a pointer to some random memory that can't be accessed, or we can return nullptr.  We return a nullptr in the hopes that our caller will either handle it or throw a nicer exception.

      LOG_0(TraceLevelInfo, "Exited GetCurrentModelFeatureCombination no model");
      return nullptr;
   }

   SegmentedTensor<ActiveDataType, FractionalDataType> * pCurrentModel = pEbmBoostingState->m_apCurrentModel[iFeatureCombination];
   EBM_ASSERT(nullptr != pCurrentModel);
   EBM_ASSERT(pCurrentModel->m_bExpanded); // the model should have been expanded at startup
   FractionalDataType * pRet = pCurrentModel->GetValuePointer();
   EBM_ASSERT(nullptr != pRet);

   LOG_N(TraceLevelInfo, "Exited GetCurrentModelFeatureCombination %p", static_cast<void *>(pRet));
   return pRet;
}

EBMCORE_IMPORT_EXPORT_BODY void EBMCORE_CALLING_CONVENTION FreeBoosting(
   PEbmBoosting ebmBoosting
) {
   LOG_N(TraceLevelInfo, "Entered FreeBoosting: ebmBoosting=%p", static_cast<void *>(ebmBoosting));
   EbmBoostingState * pEbmBoostingState = reinterpret_cast<EbmBoostingState *>(ebmBoosting);
   // pEbmBoostingState == nullptr is legal, just like delete/free
   delete pEbmBoostingState;
   LOG_0(TraceLevelInfo, "Exited FreeBoosting");
}
