// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "EbmInternal.h"

// feature includes
#include "FeatureAtomic.h"
#include "FeatureGroup.h"
// dataset depends on features
#include "DataFrameInteraction.h"
#include "ThreadStateInteraction.h"

#include "InteractionDetector.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

void InteractionDetector::Free(InteractionDetector * const pInteractionDetector) {
   LOG_0(TraceLevelInfo, "Entered InteractionDetector::Free");

   if(nullptr != pInteractionDetector) {
      pInteractionDetector->m_dataFrame.Destruct();
      free(pInteractionDetector->m_aFeatureAtomics);
      free(pInteractionDetector);
   }

   LOG_0(TraceLevelInfo, "Exited InteractionDetector::Free");
}

InteractionDetector * InteractionDetector::Allocate(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const size_t cFeatureAtomics,
   const FloatEbmType * const optionalTempParams,
   const BoolEbmType * const aFeatureAtomicsCategorical,
   const IntEbmType * const aFeatureAtomicsBinCount,
   const size_t cSamples,
   const void * const aTargets,
   const IntEbmType * const aBinnedData,
   const FloatEbmType * const aWeights, 
   const FloatEbmType * const aPredictorScores
) {
   // optionalTempParams isn't used by default.  It's meant to provide an easy way for python or other higher
   // level languages to pass EXPERIMENTAL temporary parameters easily to the C++ code.
   UNUSED(optionalTempParams);

   LOG_0(TraceLevelInfo, "Entered InteractionDetector::Allocate");

   LOG_0(TraceLevelInfo, "InteractionDetector::Allocate starting feature processing");
   FeatureAtomic * aFeatureAtomics = nullptr;
   if(0 != cFeatureAtomics) {
      aFeatureAtomics = EbmMalloc<FeatureAtomic>(cFeatureAtomics);
      if(nullptr == aFeatureAtomics) {
         LOG_0(TraceLevelWarning, "WARNING InteractionDetector::Allocate nullptr == aFeatures");
         return nullptr;
      }

      const BoolEbmType * pFeatureCategorical = aFeatureAtomicsCategorical;
      const IntEbmType * pFeatureBinCount = aFeatureAtomicsBinCount;
      size_t iFeatureAtomicInitialize = 0;
      do {
         const IntEbmType countBins = *pFeatureBinCount;
         if(countBins < 0) {
            LOG_0(TraceLevelError, "ERROR InteractionDetector::Allocate countBins cannot be negative");
            free(aFeatureAtomics);
            return nullptr;
         }
         if(0 == countBins && 0 != cSamples) {
            LOG_0(TraceLevelError, "ERROR InteractionDetector::Allocate countBins cannot be zero if 0 < cSamples");
            free(aFeatureAtomics);
            return nullptr;
         }
         if(!IsNumberConvertable<size_t>(countBins)) {
            LOG_0(TraceLevelWarning, "WARNING InteractionDetector::Allocate countBins is too high for us to allocate enough memory");
            free(aFeatureAtomics);
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
            LOG_0(TraceLevelWarning, "WARNING InteractionDetector::Initialize featureAtomicsCategorical should either be EBM_TRUE or EBM_FALSE");
         }
         const bool bCategorical = EBM_FALSE != isCategorical;

         aFeatureAtomics[iFeatureAtomicInitialize].Initialize(cBins, iFeatureAtomicInitialize, bCategorical);

         ++pFeatureCategorical;
         ++pFeatureBinCount;

         ++iFeatureAtomicInitialize;
      } while(cFeatureAtomics != iFeatureAtomicInitialize);
   }
   LOG_0(TraceLevelInfo, "InteractionDetector::Allocate done feature processing");

   InteractionDetector * const pRet = EbmMalloc<InteractionDetector>();
   if(nullptr == pRet) {
      free(aFeatureAtomics);
      return nullptr;
   }
   pRet->InitializeZero();

   pRet->m_runtimeLearningTypeOrCountTargetClasses = runtimeLearningTypeOrCountTargetClasses;
   pRet->m_cFeatureAtomics = cFeatureAtomics;
   pRet->m_aFeatureAtomics = aFeatureAtomics;
   pRet->m_cLogEnterMessages = 1000;
   pRet->m_cLogExitMessages = 1000;

   if(pRet->m_dataFrame.Initialize(
      IsClassification(runtimeLearningTypeOrCountTargetClasses),
      cFeatureAtomics,
      aFeatureAtomics,
      cSamples,
      aBinnedData,
      aWeights,
      aTargets,
      aPredictorScores,
      runtimeLearningTypeOrCountTargetClasses
   )) {
      LOG_0(TraceLevelWarning, "WARNING InteractionDetector::Allocate m_dataFrame.Initialize");
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
   const IntEbmType countFeatureAtomics, 
   const BoolEbmType * const aFeatureAtomicsCategorical,
   const IntEbmType * const aFeatureAtomicsBinCount,
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const IntEbmType countSamples, 
   const void * const targets, 
   const IntEbmType * const binnedData, 
   const FloatEbmType * const aWeights, 
   const FloatEbmType * const predictorScores,
   const FloatEbmType * const optionalTempParams
) {
   // TODO : give AllocateInteraction the same calling parameter order as CreateClassificationInteractionDetector

   if(countFeatureAtomics < 0) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction countFeatureAtomics must be positive");
      return nullptr;
   }
   if(0 != countFeatureAtomics && nullptr == aFeatureAtomicsCategorical) {
      // TODO: in the future maybe accept null aFeatureAtomicsCategorical and assume there are no missing values
      LOG_0(TraceLevelError, "ERROR AllocateInteraction aFeatureAtomicsCategorical cannot be nullptr if 0 < countFeatureAtomics");
      return nullptr;
   }
   if(0 != countFeatureAtomics && nullptr == aFeatureAtomicsBinCount) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction aFeatureAtomicsBinCount cannot be nullptr if 0 < countFeatureAtomics");
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
   if(0 != countSamples && 0 != countFeatureAtomics && nullptr == binnedData) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction binnedData cannot be nullptr if 0 < countSamples AND 0 < countFeatureAtomics");
      return nullptr;
   }
   if(0 != countSamples && nullptr == predictorScores) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction predictorScores cannot be nullptr if 0 < countSamples");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t>(countFeatureAtomics)) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction !IsNumberConvertable<size_t>(countFeatureAtomics)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t>(countSamples)) {
      LOG_0(TraceLevelError, "ERROR AllocateInteraction !IsNumberConvertable<size_t>(countSamples)");
      return nullptr;
   }

   size_t cFeatureAtomics = static_cast<size_t>(countFeatureAtomics);
   size_t cSamples = static_cast<size_t>(countSamples);

   InteractionDetector * const pInteractionDetector = InteractionDetector::Allocate(
      runtimeLearningTypeOrCountTargetClasses,
      cFeatureAtomics,
      optionalTempParams,
      aFeatureAtomicsCategorical,
      aFeatureAtomicsBinCount,
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
   IntEbmType countFeatureAtomics,
   const BoolEbmType * featureAtomicsCategorical,
   const IntEbmType * featureAtomicsBinCount,
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
      "countFeatureAtomics=%" IntEbmTypePrintf ", "
      "featureAtomicsCategorical=%p, "
      "featureAtomicsBinCount=%p, "
      "countSamples=%" IntEbmTypePrintf ", "
      "binnedData=%p, "
      "targets=%p, "
      "weights=%p, "
      "predictorScores=%p, "
      "optionalTempParams=%p"
      ,
      countTargetClasses, 
      countFeatureAtomics, 
      static_cast<const void *>(featureAtomicsCategorical),
      static_cast<const void *>(featureAtomicsBinCount),
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
      countFeatureAtomics, 
      featureAtomicsCategorical,
      featureAtomicsBinCount,
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
   IntEbmType countFeatureAtomics,
   const BoolEbmType * featureAtomicsCategorical,
   const IntEbmType * featureAtomicsBinCount,
   IntEbmType countSamples,
   const IntEbmType * binnedData,
   const FloatEbmType * targets,
   const FloatEbmType * weights, 
   const FloatEbmType * predictorScores,
   const FloatEbmType * optionalTempParams
) {
   LOG_N(TraceLevelInfo, "Entered CreateRegressionInteractionDetector: "
      "countFeatureAtomics=%" IntEbmTypePrintf ", "
      "featureAtomicsCategorical=%p, "
      "featureAtomicsBinCount=%p, "
      "countSamples=%" IntEbmTypePrintf ", "
      "binnedData=%p, "
      "targets=%p, "
      "weights=%p, "
      "predictorScores=%p, "
      "optionalTempParams=%p"
      ,
      countFeatureAtomics, 
      static_cast<const void *>(featureAtomicsCategorical),
      static_cast<const void *>(featureAtomicsBinCount),
      countSamples,
      static_cast<const void *>(binnedData), 
      static_cast<const void *>(targets), 
      static_cast<const void *>(weights), 
      static_cast<const void *>(predictorScores),
      static_cast<const void *>(optionalTempParams)
   );
   const InteractionDetectorHandle interactionDetectorHandle = reinterpret_cast<InteractionDetectorHandle>(AllocateInteraction(
      countFeatureAtomics, 
      featureAtomicsCategorical,
      featureAtomicsBinCount,
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

} // DEFINED_ZONE_NAME
