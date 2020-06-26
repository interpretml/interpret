// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebm_native.h"
#include "EbmInternal.h"
#include "Logging.h" // EBM_ASSERT & LOG
// feature includes
#include "Feature.h"
// dataset depends on features
#include "DataSetInteraction.h"
#include "CachedThreadResourcesInteraction.h"
// depends on the above
#include "DimensionMultiple.h"

#include "InteractionDetection.h"

void EbmInteractionState::Free(EbmInteractionState * const pInteractionDetection) {
   LOG_0(TraceLevelInfo, "Entered EbmInteractionState::Free");

   if(nullptr != pInteractionDetection) {
      pInteractionDetection->m_dataSet.Destruct();
      free(pInteractionDetection->m_aFeatures);
      free(pInteractionDetection);
   }

   LOG_0(TraceLevelInfo, "Exited EbmInteractionState::Free");
}

EbmInteractionState * EbmInteractionState::Allocate(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const size_t cFeatures,
   const FloatEbmType * const optionalTempParams,
   const EbmNativeFeature * const aNativeFeatures,
   const size_t cInstances,
   const void * const aTargets,
   const IntEbmType * const aBinnedData,
   const FloatEbmType * const aPredictorScores
) {
   // optionalTempParams isn't used by default.  It's meant to provide an easy way for python or other higher
   // level languages to pass EXPERIMENTAL temporary parameters easily to the C++ code.
   UNUSED(optionalTempParams);

   LOG_0(TraceLevelInfo, "Entered EbmInteractionState::Allocate");

   LOG_0(TraceLevelInfo, "EbmInteractionState::Allocate starting feature processing");
   Feature * aFeatures = nullptr;
   if(0 != cFeatures) {
      aFeatures = EbmMalloc<Feature>(cFeatures);
      if(nullptr == aFeatures) {
         LOG_0(TraceLevelWarning, "WARNING EbmInteractionState::Allocate nullptr == aFeatures");
         return nullptr;
      }
      const EbmNativeFeature * pFeatureInitialize = aNativeFeatures;
      const EbmNativeFeature * const pFeatureEnd = &aNativeFeatures[cFeatures];
      EBM_ASSERT(pFeatureInitialize < pFeatureEnd);
      size_t iFeatureInitialize = 0;
      do {
         static_assert(
            FeatureType::Ordinal == static_cast<FeatureType>(FeatureTypeOrdinal), "FeatureType::Ordinal must have the same value as FeatureTypeOrdinal"
            );
         static_assert(
            FeatureType::Nominal == static_cast<FeatureType>(FeatureTypeNominal), "FeatureType::Nominal must have the same value as FeatureTypeNominal"
            );
         if(FeatureTypeOrdinal != pFeatureInitialize->featureType && FeatureTypeNominal != pFeatureInitialize->featureType) {
            LOG_0(TraceLevelError, "ERROR EbmInteractionState::Allocate featureType must either be FeatureTypeOrdinal or FeatureTypeNominal");
            free(aFeatures);
            return nullptr;
         }
         FeatureType featureType = static_cast<FeatureType>(pFeatureInitialize->featureType);

         IntEbmType countBins = pFeatureInitialize->countBins;
         if(countBins < 0) {
            LOG_0(TraceLevelError, "ERROR EbmInteractionState::Allocate countBins cannot be negative");
            free(aFeatures);
            return nullptr;
         }
         if(0 == countBins && 0 != cInstances) {
            LOG_0(TraceLevelError, "ERROR EbmInteractionState::Allocate countBins cannot be zero if 0 < cInstances");
            free(aFeatures);
            return nullptr;
         }
         if(!IsNumberConvertable<size_t, IntEbmType>(countBins)) {
            LOG_0(TraceLevelWarning, "WARNING EbmInteractionState::Allocate countBins is too high for us to allocate enough memory");
            free(aFeatures);
            return nullptr;
         }
         size_t cBins = static_cast<size_t>(countBins);
         if(0 == cBins) {
            // we can handle 0 == cBins even though that's a degenerate case that shouldn't be boosted on.  0 bins
            // can only occur if there were zero training and zero validation cases since the 
            // features would require a value, even if it was 0.
            LOG_0(TraceLevelInfo, "INFO EbmInteractionState::Allocate feature with 0 values");
         } else if(1 == cBins) {
            // we can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on. 
            // Dimensions with 1 bin don't contribute anything since they always have the same value.
            LOG_0(TraceLevelInfo, "INFO EbmInteractionState::Allocate feature with 1 value");
         }
         if(EBM_FALSE != pFeatureInitialize->hasMissing && EBM_TRUE != pFeatureInitialize->hasMissing) {
            LOG_0(TraceLevelError, "ERROR EbmInteractionState::Allocate hasMissing must either be EBM_TRUE or EBM_FALSE");
            free(aFeatures);
            return nullptr;
         }
         bool bMissing = EBM_FALSE != pFeatureInitialize->hasMissing;

         aFeatures[iFeatureInitialize].Initialize(cBins, iFeatureInitialize, featureType, bMissing);

         EBM_ASSERT(EBM_FALSE == pFeatureInitialize->hasMissing); // TODO : implement this, then remove this assert
         EBM_ASSERT(FeatureTypeOrdinal == pFeatureInitialize->featureType); // TODO : implement this, then remove this assert

         ++iFeatureInitialize;
         ++pFeatureInitialize;
      } while(pFeatureEnd != pFeatureInitialize);
   }
   LOG_0(TraceLevelInfo, "EbmInteractionState::Allocate done feature processing");

   EbmInteractionState * const pRet = EbmMalloc<EbmInteractionState>();
   if(nullptr == pRet) {
      free(aFeatures);
      return nullptr;
   }
   pRet->InitializeZero();

   pRet->m_runtimeLearningTypeOrCountTargetClasses = runtimeLearningTypeOrCountTargetClasses;
   pRet->m_cFeatures = cFeatures;
   pRet->m_aFeatures = aFeatures;
   pRet->m_cLogEnterMessages = 1000;
   pRet->m_cLogExitMessages = 1000;

   if(pRet->m_dataSet.Initialize(
      cFeatures,
      aFeatures,
      cInstances,
      aBinnedData,
      aTargets,
      aPredictorScores,
      runtimeLearningTypeOrCountTargetClasses
   )) {
      LOG_0(TraceLevelWarning, "WARNING EbmInteractionState::Allocate m_dataSet.Initialize");
      EbmInteractionState::Free(pRet);
      return nullptr;
   }

   LOG_0(TraceLevelInfo, "Exited EbmInteractionState::Allocate");
   return pRet;
}

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
EbmInteractionState * AllocateInteraction(
   IntEbmType countFeatures, 
   const EbmNativeFeature * features, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, 
   IntEbmType countInstances, 
   const void * targets, 
   const IntEbmType * binnedData, 
   const FloatEbmType * predictorScores,
   const FloatEbmType * const optionalTempParams
) {
   // TODO : give AllocateInteraction the same calling parameter order as InitializeInteractionClassification

   if(countFeatures < 0) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction countFeatures must be positive");
      return nullptr;
   }
   if(0 != countFeatures && nullptr == features) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction features cannot be nullptr if 0 < countFeatures");
      return nullptr;
   }
   if(countInstances < 0) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction countInstances must be positive");
      return nullptr;
   }
   if(0 != countInstances && nullptr == targets) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction targets cannot be nullptr if 0 < countInstances");
      return nullptr;
   }
   if(0 != countInstances && 0 != countFeatures && nullptr == binnedData) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction binnedData cannot be nullptr if 0 < countInstances AND 0 < countFeatures");
      return nullptr;
   }
   if(0 != countInstances && nullptr == predictorScores) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction predictorScores cannot be nullptr if 0 < countInstances");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntEbmType>(countFeatures)) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction !IsNumberConvertable<size_t, IntEbmType>(countFeatures)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntEbmType>(countInstances)) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction !IsNumberConvertable<size_t, IntEbmType>(countInstances)");
      return nullptr;
   }

   size_t cFeatures = static_cast<size_t>(countFeatures);
   size_t cInstances = static_cast<size_t>(countInstances);

   EbmInteractionState * const pEbmInteractionState = EbmInteractionState::Allocate(
      runtimeLearningTypeOrCountTargetClasses,
      cFeatures,
      optionalTempParams,
      features,
      cInstances,
      targets,
      binnedData,
      predictorScores
   );
   if(UNLIKELY(nullptr == pEbmInteractionState)) {
      LOG_0(TraceLevelWarning, "WARNING AllocateInteraction nullptr == pEbmInteractionState");
      return nullptr;
   }
   return pEbmInteractionState;
}

EBM_NATIVE_IMPORT_EXPORT_BODY PEbmInteraction EBM_NATIVE_CALLING_CONVENTION InitializeInteractionClassification(
   IntEbmType countTargetClasses,
   IntEbmType countFeatures,
   const EbmNativeFeature * features,
   IntEbmType countInstances,
   const IntEbmType * binnedData,
   const IntEbmType * targets,
   const FloatEbmType * predictorScores,
   const FloatEbmType * optionalTempParams
) {
   LOG_N(
      TraceLevelInfo, 
      "Entered InitializeInteractionClassification: countTargetClasses=%" IntEbmTypePrintf ", countFeatures=%" IntEbmTypePrintf 
      ", features=%p, countInstances=%" IntEbmTypePrintf ", binnedData=%p, targets=%p, predictorScores=%p, optionalTempParams=%p",
      countTargetClasses, 
      countFeatures, 
      static_cast<const void *>(features), 
      countInstances, 
      static_cast<const void *>(binnedData), 
      static_cast<const void *>(targets), 
      static_cast<const void *>(predictorScores),
      static_cast<const void *>(optionalTempParams)
   );
   if(countTargetClasses < 0) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification countTargetClasses can't be negative");
      return nullptr;
   }
   if(0 == countTargetClasses && 0 != countInstances) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification countTargetClasses can't be zero unless there are no instances");
      return nullptr;
   }
   if(!IsNumberConvertable<ptrdiff_t, IntEbmType>(countTargetClasses)) {
      LOG_0(TraceLevelWarning, "WARNING InitializeInteractionClassification !IsNumberConvertable<ptrdiff_t, IntEbmType>(countTargetClasses)");
      return nullptr;
   }
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = static_cast<ptrdiff_t>(countTargetClasses);
   PEbmInteraction pEbmInteraction = reinterpret_cast<PEbmInteraction>(AllocateInteraction(
      countFeatures, 
      features, 
      runtimeLearningTypeOrCountTargetClasses, 
      countInstances, 
      targets, 
      binnedData, 
      predictorScores,
      optionalTempParams
   ));
   LOG_N(TraceLevelInfo, "Exited InitializeInteractionClassification %p", static_cast<void *>(pEbmInteraction));
   return pEbmInteraction;
}

EBM_NATIVE_IMPORT_EXPORT_BODY PEbmInteraction EBM_NATIVE_CALLING_CONVENTION InitializeInteractionRegression(
   IntEbmType countFeatures,
   const EbmNativeFeature * features,
   IntEbmType countInstances,
   const IntEbmType * binnedData,
   const FloatEbmType * targets,
   const FloatEbmType * predictorScores,
   const FloatEbmType * optionalTempParams
) {
   LOG_N(TraceLevelInfo, "Entered InitializeInteractionRegression: countFeatures=%" IntEbmTypePrintf ", features=%p, countInstances=%" IntEbmTypePrintf 
      ", binnedData=%p, targets=%p, predictorScores=%p, optionalTempParams=%p",
      countFeatures, 
      static_cast<const void *>(features), 
      countInstances, 
      static_cast<const void *>(binnedData), 
      static_cast<const void *>(targets), 
      static_cast<const void *>(predictorScores),
      static_cast<const void *>(optionalTempParams)
   );
   PEbmInteraction pEbmInteraction = reinterpret_cast<PEbmInteraction>(AllocateInteraction(
      countFeatures, 
      features, 
      k_Regression, 
      countInstances, 
      targets, 
      binnedData, 
      predictorScores,
      optionalTempParams
   ));
   LOG_N(TraceLevelInfo, "Exited InitializeInteractionRegression %p", static_cast<void *>(pEbmInteraction));
   return pEbmInteraction;
}

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
static IntEbmType GetInteractionScorePerTargetClasses(
   EbmInteractionState * const pEbmInteractionState, 
   const FeatureCombination * const pFeatureCombination, 
   const size_t cInstancesRequiredForChildSplitMin, 
   FloatEbmType * const pInteractionScoreReturn
) {
   // TODO : be smarter about our CachedInteractionThreadResources, otherwise why have it?
   CachedInteractionThreadResources * const pCachedThreadResources = CachedInteractionThreadResources::Allocate();
   if(nullptr == pCachedThreadResources) {
      return 1;
   }

   if(CalculateInteractionScore<compilerLearningTypeOrCountTargetClasses, 0>(
      pCachedThreadResources,
      pEbmInteractionState,
      pFeatureCombination, 
      cInstancesRequiredForChildSplitMin, 
      pInteractionScoreReturn
   )) {
      pCachedThreadResources->Free();
      return 1;
   }
   pCachedThreadResources->Free();
   return 0;
}

template<ptrdiff_t possibleCompilerLearningTypeOrCountTargetClasses>
EBM_INLINE IntEbmType CompilerRecursiveGetInteractionScore(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, 
   EbmInteractionState * const pEbmInteractionState, 
   const FeatureCombination * const pFeatureCombination, 
   const size_t cInstancesRequiredForChildSplitMin, 
   FloatEbmType * const pInteractionScoreReturn
) {
   static_assert(IsClassification(possibleCompilerLearningTypeOrCountTargetClasses), 
      "possibleCompilerLearningTypeOrCountTargetClasses needs to be a classification");
   EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
   if(runtimeLearningTypeOrCountTargetClasses == possibleCompilerLearningTypeOrCountTargetClasses) {
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);
      return GetInteractionScorePerTargetClasses<possibleCompilerLearningTypeOrCountTargetClasses>(
         pEbmInteractionState, 
         pFeatureCombination, 
         cInstancesRequiredForChildSplitMin, 
         pInteractionScoreReturn
      );
   } else {
      return CompilerRecursiveGetInteractionScore<possibleCompilerLearningTypeOrCountTargetClasses + 1>(
         runtimeLearningTypeOrCountTargetClasses, 
         pEbmInteractionState, 
         pFeatureCombination, 
         cInstancesRequiredForChildSplitMin, 
         pInteractionScoreReturn
      );
   }
}

template<>
EBM_INLINE IntEbmType CompilerRecursiveGetInteractionScore<k_cCompilerOptimizedTargetClassesMax + 1>(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, 
   EbmInteractionState * const pEbmInteractionState, 
   const FeatureCombination * const pFeatureCombination, 
   const size_t cInstancesRequiredForChildSplitMin, 
   FloatEbmType * const pInteractionScoreReturn
) {
   UNUSED(runtimeLearningTypeOrCountTargetClasses);
   // it is logically possible, but uninteresting to have a classification with 1 target class, 
   // so let our runtime system handle those unlikley and uninteresting cases
   static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");
   EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
   EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < runtimeLearningTypeOrCountTargetClasses);
   return GetInteractionScorePerTargetClasses<k_DynamicClassification>(
      pEbmInteractionState, 
      pFeatureCombination, 
      cInstancesRequiredForChildSplitMin, 
      pInteractionScoreReturn
   );
}

// we made this a global because if we had put this variable inside the EbmInteractionState object, then we would need to dereference that before getting 
// the count.  By making this global we can send a log message incase a bad EbmInteractionState object is sent into us we only decrease the count if the 
// count is non-zero, so at worst if there is a race condition then we'll output this log message more times than desired, but we can live with that
static unsigned int g_cLogGetInteractionScoreParametersMessages = 10;

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GetInteractionScore(
   PEbmInteraction ebmInteraction,
   IntEbmType countFeaturesInCombination,
   const IntEbmType * featureIndexes,
   IntEbmType countInstancesRequiredForChildSplitMin,
   FloatEbmType * interactionScoreReturn
) {
   LOG_COUNTED_N(
      &g_cLogGetInteractionScoreParametersMessages, 
      TraceLevelInfo, 
      TraceLevelVerbose, 
      "GetInteractionScore parameters: ebmInteraction=%p, countFeaturesInCombination=%" IntEbmTypePrintf ", featureIndexes=%p, countInstancesRequiredForChildSplitMin=%" IntEbmTypePrintf ", interactionScoreReturn=%p", 
      static_cast<void *>(ebmInteraction), 
      countFeaturesInCombination, 
      static_cast<const void *>(featureIndexes), 
      countInstancesRequiredForChildSplitMin,
      static_cast<void *>(interactionScoreReturn)
   );

   EbmInteractionState * pEbmInteractionState = reinterpret_cast<EbmInteractionState *>(ebmInteraction);
   if(nullptr == pEbmInteractionState) {
      if(LIKELY(nullptr != interactionScoreReturn)) {
         *interactionScoreReturn = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR GetInteractionScore ebmInteraction cannot be nullptr");
      return 1;
   }

   LOG_COUNTED_0(pEbmInteractionState->GetPointerCountLogEnterMessages(), TraceLevelInfo, TraceLevelVerbose, "Entered GetInteractionScore");

   if(countFeaturesInCombination < 0) {
      if(LIKELY(nullptr != interactionScoreReturn)) {
         *interactionScoreReturn = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR GetInteractionScore countFeaturesInCombination must be positive");
      return 1;
   }
   if(0 != countFeaturesInCombination && nullptr == featureIndexes) {
      if(LIKELY(nullptr != interactionScoreReturn)) {
         *interactionScoreReturn = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR GetInteractionScore featureIndexes cannot be nullptr if 0 < countFeaturesInCombination");
      return 1;
   }
   if(!IsNumberConvertable<size_t, IntEbmType>(countFeaturesInCombination)) {
      if(LIKELY(nullptr != interactionScoreReturn)) {
         *interactionScoreReturn = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR GetInteractionScore countFeaturesInCombination too large to index");
      return 1;
   }
   size_t cFeaturesInCombination = static_cast<size_t>(countFeaturesInCombination);
   if(0 == cFeaturesInCombination) {
      LOG_0(TraceLevelInfo, "INFO GetInteractionScore empty feature combination");
      if(nullptr != interactionScoreReturn) {
         // we return the lowest value possible for the interaction score, but we don't return an error since we handle it even though we'd prefer our 
         // caler be smarter about this condition
         *interactionScoreReturn = FloatEbmType { 0 };
      }
      return 0;
   }
   if(0 == pEbmInteractionState->GetDataSetByFeature()->GetCountInstances()) {
      // if there are zero instances, there isn't much basis to say whether there are interactions, so just return zero
      LOG_0(TraceLevelInfo, "INFO GetInteractionScore zero instances");
      if(nullptr != interactionScoreReturn) {
         // we return the lowest value possible for the interaction score, but we don't return an error since we handle it even though we'd prefer our 
         // caler be smarter about this condition
         *interactionScoreReturn = 0;
      }
      return 0;
   }

   size_t cInstancesRequiredForChildSplitMin = size_t { 1 }; // this is the min value
   if(IntEbmType { 1 } <= countInstancesRequiredForChildSplitMin) {
      cInstancesRequiredForChildSplitMin = static_cast<size_t>(countInstancesRequiredForChildSplitMin);
      if(!IsNumberConvertable<size_t, IntEbmType>(countInstancesRequiredForChildSplitMin)) {
         // we can never exceed a size_t number of instances, so let's just set it to the maximum if we were going to overflow because it will generate 
         // the same results as if we used the true number
         cInstancesRequiredForChildSplitMin = std::numeric_limits<size_t>::max();
      }
   } else {
      LOG_0(TraceLevelWarning, "WARNING GetInteractionScore countInstancesRequiredForChildSplitMin can't be less than 1.  Adjusting to 1.");
   }

   const Feature * const aFeatures = pEbmInteractionState->GetFeatures();
   const IntEbmType * pFeatureCombinationIndex = featureIndexes;
   const IntEbmType * const pFeatureCombinationIndexEnd = featureIndexes + cFeaturesInCombination;

   do {
      const IntEbmType indexFeatureInterop = *pFeatureCombinationIndex;
      if(indexFeatureInterop < 0) {
         if(LIKELY(nullptr != interactionScoreReturn)) {
            *interactionScoreReturn = FloatEbmType { 0 };
         }
         LOG_0(TraceLevelError, "ERROR GetInteractionScore featureIndexes value cannot be negative");
         return 1;
      }
      if(!IsNumberConvertable<size_t, IntEbmType>(indexFeatureInterop)) {
         if(LIKELY(nullptr != interactionScoreReturn)) {
            *interactionScoreReturn = FloatEbmType { 0 };
         }
         LOG_0(TraceLevelError, "ERROR GetInteractionScore featureIndexes value too big to reference memory");
         return 1;
      }
      const size_t iFeatureForCombination = static_cast<size_t>(indexFeatureInterop);
      if(pEbmInteractionState->GetCountFeatures() <= iFeatureForCombination) {
         if(LIKELY(nullptr != interactionScoreReturn)) {
            *interactionScoreReturn = FloatEbmType { 0 };
         }
         LOG_0(TraceLevelError, "ERROR GetInteractionScore featureIndexes value must be less than the number of features");
         return 1;
      }
      const Feature * const pFeature = &aFeatures[iFeatureForCombination];
      if(pFeature->GetCountBins() <= 1) {
         if(nullptr != interactionScoreReturn) {
            // we return the lowest value possible for the interaction score, but we don't return an error since we handle it even though we'd prefer 
            // our caler be smarter about this condition
            *interactionScoreReturn = 0;
         }
         LOG_0(TraceLevelInfo, "INFO GetInteractionScore feature with 0/1 value");
         return 0;
      }
      ++pFeatureCombinationIndex;
   } while(pFeatureCombinationIndexEnd != pFeatureCombinationIndex);

   if(k_cDimensionsMax < cFeaturesInCombination) {
      // if we try to run with more than k_cDimensionsMax we'll exceed our memory capacity, so let's exit here instead
      LOG_0(TraceLevelWarning, "WARNING GetInteractionScore k_cDimensionsMax < cFeaturesInCombination");
      return 1;
   }

   // put the pFeatureCombination object on the stack. We want to put it into a FeatureCombination object since we want to share code with boosting, 
   // which calls things like building the tensor totals (which is templated to be compiled many times)
   char FeatureCombinationBuffer[FeatureCombination::GetFeatureCombinationCountBytes(k_cDimensionsMax)];
   FeatureCombination * const pFeatureCombination = reinterpret_cast<FeatureCombination *>(&FeatureCombinationBuffer);
   pFeatureCombination->Initialize(cFeaturesInCombination, 0);

   pFeatureCombinationIndex = featureIndexes; // restart from the start
   FeatureCombinationEntry * pFeatureCombinationEntry = pFeatureCombination->GetFeatureCombinationEntries();
   do {
      const IntEbmType indexFeatureInterop = *pFeatureCombinationIndex;
      EBM_ASSERT(0 <= indexFeatureInterop);
      EBM_ASSERT((IsNumberConvertable<size_t, IntEbmType>(indexFeatureInterop))); // we already checked indexFeatureInterop was good above
      size_t iFeatureForCombination = static_cast<size_t>(indexFeatureInterop);
      EBM_ASSERT(iFeatureForCombination < pEbmInteractionState->GetCountFeatures());
      const Feature * const pFeature = &aFeatures[iFeatureForCombination];
      EBM_ASSERT(2 <= pFeature->GetCountBins()); // we should have filtered out anything with 1 bin above

      pFeatureCombinationEntry->m_pFeature = pFeature;
      ++pFeatureCombinationEntry;
      ++pFeatureCombinationIndex;
   } while(pFeatureCombinationIndexEnd != pFeatureCombinationIndex);

   IntEbmType ret;
   if(IsClassification(pEbmInteractionState->GetRuntimeLearningTypeOrCountTargetClasses())) {
      if(pEbmInteractionState->GetRuntimeLearningTypeOrCountTargetClasses() <= ptrdiff_t { 1 }) {
         LOG_0(TraceLevelInfo, "INFO GetInteractionScore target with 0/1 classes");
         if(nullptr != interactionScoreReturn) {
            // if there is only 1 classification target, then we can predict the outcome with 100% accuracy and there is no need for logits or 
            // interactions or anything else.  We return 0 since interactions have no benefit
            *interactionScoreReturn = FloatEbmType { 0 };
         }
         return 0;
      }
      ret = CompilerRecursiveGetInteractionScore<2>(
         pEbmInteractionState->GetRuntimeLearningTypeOrCountTargetClasses(),
         pEbmInteractionState, 
         pFeatureCombination, 
         cInstancesRequiredForChildSplitMin, 
         interactionScoreReturn
      );
   } else {
      EBM_ASSERT(IsRegression(pEbmInteractionState->GetRuntimeLearningTypeOrCountTargetClasses()));
      ret = GetInteractionScorePerTargetClasses<k_Regression>(
         pEbmInteractionState, 
         pFeatureCombination, 
         cInstancesRequiredForChildSplitMin, 
         interactionScoreReturn
      );
   }
   if(0 != ret) {
      LOG_N(TraceLevelWarning, "WARNING GetInteractionScore returned %" IntEbmTypePrintf, ret);
   }
   if(nullptr != interactionScoreReturn) {
      // if *interactionScoreReturn was negative for floating point instability reasons, we zero it so that we don't return a negative number to our caller
      EBM_ASSERT(FloatEbmType { 0 } <= *interactionScoreReturn);
      LOG_COUNTED_N(
         pEbmInteractionState->GetPointerCountLogExitMessages(),
         TraceLevelInfo, 
         TraceLevelVerbose, 
         "Exited GetInteractionScore %" FloatEbmTypePrintf, *interactionScoreReturn
      );
   } else {
      LOG_COUNTED_0(pEbmInteractionState->GetPointerCountLogExitMessages(), TraceLevelInfo, TraceLevelVerbose, "Exited GetInteractionScore");
   }
   return ret;
}

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION FreeInteraction(
   PEbmInteraction ebmInteraction
) {
   LOG_N(TraceLevelInfo, "Entered FreeInteraction: ebmInteraction=%p", static_cast<void *>(ebmInteraction));
   EbmInteractionState * pEbmInteractionState = reinterpret_cast<EbmInteractionState *>(ebmInteraction);

   // pEbmInteractionState is allowed to be nullptr.  We handle that inside EbmInteractionState::Free
   EbmInteractionState::Free(pEbmInteractionState);
   
   LOG_0(TraceLevelInfo, "Exited FreeInteraction");
}
