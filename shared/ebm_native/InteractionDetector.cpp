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
#include "FeatureAtomic.h"
#include "FeatureGroup.h"
// dataset depends on features
#include "DataSetInteraction.h"
#include "CachedThreadResourcesInteraction.h"

#include "InteractionDetector.h"

void InteractionDetector::Free(InteractionDetector * const pInteractionDetector) {
   LOG_0(TraceLevelInfo, "Entered InteractionDetector::Free");

   if(nullptr != pInteractionDetector) {
      pInteractionDetector->m_dataSet.Destruct();
      free(pInteractionDetector->m_aFeatures);
      free(pInteractionDetector);
   }

   LOG_0(TraceLevelInfo, "Exited InteractionDetector::Free");
}

InteractionDetector * InteractionDetector::Allocate(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const size_t cFeatures,
   const FloatEbmType * const optionalTempParams,
   const BoolEbmType * const aFeaturesCategorical,
   const IntEbmType * const aFeaturesBinCount,
   const size_t cSamples,
   const void * const aTargets,
   const IntEbmType * const aBinnedData,
   const FloatEbmType * const aWeights, 
   const FloatEbmType * const aPredictorScores
) {
   // optionalTempParams isn't used by default.  It's meant to provide an easy way for python or other higher
   // level languages to pass EXPERIMENTAL temporary parameters easily to the C++ code.
   UNUSED(optionalTempParams);

   // TODO : implement weights 
   UNUSED(aWeights);
   EBM_ASSERT(nullptr == aWeights);

   LOG_0(TraceLevelInfo, "Entered InteractionDetector::Allocate");

   LOG_0(TraceLevelInfo, "InteractionDetector::Allocate starting feature processing");
   Feature * aFeatures = nullptr;
   if(0 != cFeatures) {
      aFeatures = EbmMalloc<Feature>(cFeatures);
      if(nullptr == aFeatures) {
         LOG_0(TraceLevelWarning, "WARNING InteractionDetector::Allocate nullptr == aFeatures");
         return nullptr;
      }

      const BoolEbmType * pFeatureCategorical = aFeaturesCategorical;
      const IntEbmType * pFeatureBinCount = aFeaturesBinCount;
      size_t iFeatureInitialize = 0;
      do {
         const IntEbmType countBins = *pFeatureBinCount;
         if(countBins < 0) {
            LOG_0(TraceLevelError, "ERROR InteractionDetector::Allocate countBins cannot be negative");
            free(aFeatures);
            return nullptr;
         }
         if(0 == countBins && 0 != cSamples) {
            LOG_0(TraceLevelError, "ERROR InteractionDetector::Allocate countBins cannot be zero if 0 < cSamples");
            free(aFeatures);
            return nullptr;
         }
         if(!IsNumberConvertable<size_t>(countBins)) {
            LOG_0(TraceLevelWarning, "WARNING InteractionDetector::Allocate countBins is too high for us to allocate enough memory");
            free(aFeatures);
            return nullptr;
         }
         const size_t cBins = static_cast<size_t>(countBins);
         if(0 == cBins) {
            // we can handle 0 == cBins even though that's a degenerate case that shouldn't be boosted on.  0 bins
            // can only occur if there were zero training and zero validation cases since the 
            // features would require a value, even if it was 0.
            LOG_0(TraceLevelInfo, "INFO InteractionDetector::Allocate feature with 0 values");
         } else if(1 == cBins) {
            // we can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on. 
            // Dimensions with 1 bin don't contribute anything since they always have the same value.
            LOG_0(TraceLevelInfo, "INFO InteractionDetector::Allocate feature with 1 value");
         }
         const BoolEbmType isCategorical = *pFeatureCategorical;
         if(EBM_FALSE != isCategorical && EBM_TRUE != isCategorical) {
            LOG_0(TraceLevelWarning, "WARNING InteractionDetector::Initialize featuresCategorical should either be EBM_TRUE or EBM_FALSE");
         }
         const bool bCategorical = EBM_FALSE != isCategorical;

         aFeatures[iFeatureInitialize].Initialize(cBins, iFeatureInitialize, bCategorical);

         ++pFeatureCategorical;
         ++pFeatureBinCount;

         ++iFeatureInitialize;
      } while(cFeatures != iFeatureInitialize);
   }
   LOG_0(TraceLevelInfo, "InteractionDetector::Allocate done feature processing");

   InteractionDetector * const pRet = EbmMalloc<InteractionDetector>();
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
      cSamples,
      aBinnedData,
      aTargets,
      aPredictorScores,
      runtimeLearningTypeOrCountTargetClasses
   )) {
      LOG_0(TraceLevelWarning, "WARNING InteractionDetector::Allocate m_dataSet.Initialize");
      InteractionDetector::Free(pRet);
      return nullptr;
   }

   LOG_0(TraceLevelInfo, "Exited InteractionDetector::Allocate");
   return pRet;
}

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
static InteractionDetector * AllocateInteraction(
   const IntEbmType countFeatures, 
   const BoolEbmType * const aFeaturesCategorical,
   const IntEbmType * const aFeaturesBinCount,
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const IntEbmType countSamples, 
   const void * const targets, 
   const IntEbmType * const binnedData, 
   const FloatEbmType * const aWeights, 
   const FloatEbmType * const predictorScores,
   const FloatEbmType * const optionalTempParams
) {
   // TODO : give AllocateInteraction the same calling parameter order as CreateClassificationInteractionDetector

   if(countFeatures < 0) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction countFeatures must be positive");
      return nullptr;
   }
   if(0 != countFeatures && nullptr == aFeaturesCategorical) {
      // TODO: in the future maybe accept null aFeaturesCategorical and assume there are no missing values
      LOG_0(TraceLevelError, "ERROR AllocateInteraction aFeaturesCategorical cannot be nullptr if 0 < countFeatures");
      return nullptr;
   }
   if(0 != countFeatures && nullptr == aFeaturesBinCount) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction aFeaturesBinCount cannot be nullptr if 0 < countFeatures");
      return nullptr;
   }
   if(countSamples < 0) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction countSamples must be positive");
      return nullptr;
   }
   if(0 != countSamples && nullptr == targets) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction targets cannot be nullptr if 0 < countSamples");
      return nullptr;
   }
   if(0 != countSamples && 0 != countFeatures && nullptr == binnedData) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction binnedData cannot be nullptr if 0 < countSamples AND 0 < countFeatures");
      return nullptr;
   }
   if(0 != countSamples && nullptr == predictorScores) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction predictorScores cannot be nullptr if 0 < countSamples");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t>(countFeatures)) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction !IsNumberConvertable<size_t>(countFeatures)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t>(countSamples)) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction !IsNumberConvertable<size_t>(countSamples)");
      return nullptr;
   }

   size_t cFeatures = static_cast<size_t>(countFeatures);
   size_t cSamples = static_cast<size_t>(countSamples);

   InteractionDetector * const pInteractionDetector = InteractionDetector::Allocate(
      runtimeLearningTypeOrCountTargetClasses,
      cFeatures,
      optionalTempParams,
      aFeaturesCategorical,
      aFeaturesBinCount,
      cSamples,
      targets,
      binnedData,
      aWeights, 
      predictorScores
   );
   if(UNLIKELY(nullptr == pInteractionDetector)) {
      LOG_0(TraceLevelWarning, "WARNING AllocateInteraction nullptr == pInteractionDetector");
      return nullptr;
   }
   return pInteractionDetector;
}

EBM_NATIVE_IMPORT_EXPORT_BODY InteractionDetectorHandle EBM_NATIVE_CALLING_CONVENTION CreateClassificationInteractionDetector(
   IntEbmType countTargetClasses,
   IntEbmType countFeatures,
   const BoolEbmType * featuresCategorical,
   const IntEbmType * featuresBinCount,
   IntEbmType countSamples,
   const IntEbmType * binnedData,
   const IntEbmType * targets,
   const FloatEbmType * weights,
   const FloatEbmType * predictorScores,
   const FloatEbmType * optionalTempParams
) {
   LOG_N(
      TraceLevelInfo, 
      "Entered CreateClassificationInteractionDetector: "
      "countTargetClasses=%" IntEbmTypePrintf ", "
      "countFeatures=%" IntEbmTypePrintf ", "
      "featuresCategorical=%p, "
      "featuresBinCount=%p, "
      "countSamples=%" IntEbmTypePrintf ", "
      "binnedData=%p, "
      "targets=%p, "
      "weights=%p, "
      "predictorScores=%p, "
      "optionalTempParams=%p"
      ,
      countTargetClasses, 
      countFeatures, 
      static_cast<const void *>(featuresCategorical),
      static_cast<const void *>(featuresBinCount),
      countSamples,
      static_cast<const void *>(binnedData), 
      static_cast<const void *>(targets), 
      static_cast<const void *>(weights), 
      static_cast<const void *>(predictorScores),
      static_cast<const void *>(optionalTempParams)
   );
   if(countTargetClasses < 0) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationInteractionDetector countTargetClasses can't be negative");
      return nullptr;
   }
   if(0 == countTargetClasses && 0 != countSamples) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationInteractionDetector countTargetClasses can't be zero unless there are no samples");
      return nullptr;
   }
   if(!IsNumberConvertable<ptrdiff_t>(countTargetClasses)) {
      LOG_0(TraceLevelWarning, "WARNING CreateClassificationInteractionDetector !IsNumberConvertable<ptrdiff_t>(countTargetClasses)");
      return nullptr;
   }
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = static_cast<ptrdiff_t>(countTargetClasses);
   const InteractionDetectorHandle interactionDetectorHandle = reinterpret_cast<InteractionDetectorHandle>(AllocateInteraction(
      countFeatures, 
      featuresCategorical,
      featuresBinCount,
      runtimeLearningTypeOrCountTargetClasses,
      countSamples, 
      targets, 
      binnedData, 
      weights,
      predictorScores,
      optionalTempParams
   ));
   LOG_N(TraceLevelInfo, "Exited CreateClassificationInteractionDetector %p", static_cast<void *>(interactionDetectorHandle));
   return interactionDetectorHandle;
}

EBM_NATIVE_IMPORT_EXPORT_BODY InteractionDetectorHandle EBM_NATIVE_CALLING_CONVENTION CreateRegressionInteractionDetector(
   IntEbmType countFeatures,
   const BoolEbmType * featuresCategorical,
   const IntEbmType * featuresBinCount,
   IntEbmType countSamples,
   const IntEbmType * binnedData,
   const FloatEbmType * targets,
   const FloatEbmType * weights, 
   const FloatEbmType * predictorScores,
   const FloatEbmType * optionalTempParams
) {
   LOG_N(TraceLevelInfo, "Entered CreateRegressionInteractionDetector: "
      "countFeatures=%" IntEbmTypePrintf ", "
      "featuresCategorical=%p, "
      "featuresBinCount=%p, "
      "countSamples=%" IntEbmTypePrintf ", "
      "binnedData=%p, "
      "targets=%p, "
      "weights=%p, "
      "predictorScores=%p, "
      "optionalTempParams=%p"
      ,
      countFeatures, 
      static_cast<const void *>(featuresCategorical),
      static_cast<const void *>(featuresBinCount),
      countSamples,
      static_cast<const void *>(binnedData), 
      static_cast<const void *>(targets), 
      static_cast<const void *>(weights), 
      static_cast<const void *>(predictorScores),
      static_cast<const void *>(optionalTempParams)
   );
   const InteractionDetectorHandle interactionDetectorHandle = reinterpret_cast<InteractionDetectorHandle>(AllocateInteraction(
      countFeatures, 
      featuresCategorical,
      featuresBinCount,
      k_regression,
      countSamples, 
      targets, 
      binnedData, 
      weights, 
      predictorScores,
      optionalTempParams
   ));
   LOG_N(TraceLevelInfo, "Exited CreateRegressionInteractionDetector %p", static_cast<void *>(interactionDetectorHandle));
   return interactionDetectorHandle;
}

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION FreeInteractionDetector(
   InteractionDetectorHandle interactionDetectorHandle
) {
   LOG_N(TraceLevelInfo, "Entered FreeInteractionDetector: interactionDetectorHandle=%p", static_cast<void *>(interactionDetectorHandle));
   InteractionDetector * pInteractionDetector = reinterpret_cast<InteractionDetector *>(interactionDetectorHandle);

   // pInteractionDetector is allowed to be nullptr.  We handle that inside InteractionDetector::Free
   InteractionDetector::Free(pInteractionDetector);
   
   LOG_0(TraceLevelInfo, "Exited FreeInteractionDetector");
}
