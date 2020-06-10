// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef EBM_INTERACTION_STATE_H
#define EBM_INTERACTION_STATE_H

#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebm_native.h"
#include "EbmInternal.h"
#include "Logging.h" // EBM_ASSERT & LOG
// feature includes
#include "Feature.h"
// dataset depends on features
#include "DataSetByFeature.h"

class EbmInteractionState {
public:
   const ptrdiff_t m_runtimeLearningTypeOrCountTargetClasses;

   const size_t m_cFeatures;
   // TODO : in the future, we can allocate this inside a function so that even the objects inside are const
   Feature * const m_aFeatures;

   DataSetByFeature m_dataSet;

   unsigned int m_cLogEnterMessages;
   unsigned int m_cLogExitMessages;

   EBM_INLINE EbmInteractionState(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, 
      const size_t cFeatures, 
      const FloatEbmType * const optionalTempParams
   )
      : m_runtimeLearningTypeOrCountTargetClasses(runtimeLearningTypeOrCountTargetClasses)
      , m_cFeatures(cFeatures)
      , m_aFeatures(0 == cFeatures || IsMultiplyError(sizeof(Feature), cFeatures) ? nullptr : static_cast<Feature *>(malloc(sizeof(Feature) * cFeatures)))
      , m_dataSet()
      , m_cLogEnterMessages(1000)
      , m_cLogExitMessages(1000) 
   {
      // optionalTempParams isn't used by default.  It's meant to provide an easy way for python or other higher
      // level languages to pass EXPERIMENTAL temporary parameters easily to the C++ code.
      UNUSED(optionalTempParams);
   }

   EBM_INLINE ~EbmInteractionState() {
      LOG_0(TraceLevelInfo, "Entered ~EbmInteractionState");

      free(m_aFeatures);

      LOG_0(TraceLevelInfo, "Exited ~EbmInteractionState");
   }

   EBM_INLINE bool InitializeInteraction(
      const EbmNativeFeature * const aFeatures, 
      const size_t cInstances, 
      const void * const aTargets, 
      const IntEbmType * const aBinnedData, 
      const FloatEbmType * const aPredictorScores
   ) {
      LOG_0(TraceLevelInfo, "Entered InitializeInteraction");

      if(0 != m_cFeatures && nullptr == m_aFeatures) {
         LOG_0(TraceLevelWarning, "WARNING InitializeInteraction 0 != m_cFeatures && nullptr == m_aFeatures");
         return true;
      }

      LOG_0(TraceLevelInfo, "InitializeInteraction starting feature processing");
      if(0 != m_cFeatures) {
         EBM_ASSERT(!IsMultiplyError(m_cFeatures, sizeof(*aFeatures))); // if this overflows then our caller should not have been able to allocate the array
         const EbmNativeFeature * pFeatureInitialize = aFeatures;
         const EbmNativeFeature * const pFeatureEnd = &aFeatures[m_cFeatures];
         EBM_ASSERT(pFeatureInitialize < pFeatureEnd);
         size_t iFeatureInitialize = 0;
         do {
            static_assert(
               FeatureType::Ordinal == static_cast<FeatureType>(FeatureTypeOrdinal), "FeatureType::Ordinal must have the same value as FeatureTypeOrdinal"
            );
            static_assert(
               FeatureType::Nominal == static_cast<FeatureType>(FeatureTypeNominal), "FeatureType::Nominal must have the same value as FeatureTypeNominal"
            );
            EBM_ASSERT(FeatureTypeOrdinal == pFeatureInitialize->featureType || FeatureTypeNominal == pFeatureInitialize->featureType);
            FeatureType featureType = static_cast<FeatureType>(pFeatureInitialize->featureType);

            IntEbmType countBins = pFeatureInitialize->countBins;
            // we can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on (dimensions with 1 bin don't contribute anything 
            // since they always have the same value)
            EBM_ASSERT(0 <= countBins);
            if(!IsNumberConvertable<size_t, IntEbmType>(countBins)) {
               LOG_0(TraceLevelWarning, "WARNING InitializeInteraction !IsNumberConvertable<size_t, IntEbmType>(countBins)");
               return true;
            }
            size_t cBins = static_cast<size_t>(countBins);
            if(cBins <= 1) {
               EBM_ASSERT(0 != cBins || 0 == cInstances);
               LOG_0(TraceLevelInfo, "INFO InitializeInteraction feature with 0/1 value");
            }

            EBM_ASSERT(EBM_FALSE == pFeatureInitialize->hasMissing || EBM_TRUE == pFeatureInitialize->hasMissing);
            bool bMissing = EBM_FALSE != pFeatureInitialize->hasMissing;

            // this is an in-place new, so there is no new memory allocated, and we already knew where it was going, so we don't need the 
            // resulting pointer returned
            new (&m_aFeatures[iFeatureInitialize]) Feature(cBins, iFeatureInitialize, featureType, bMissing);
            // we don't allocate memory and our constructor doesn't have errors, so we shouldn't have an error here

            EBM_ASSERT(EBM_FALSE == pFeatureInitialize->hasMissing); // TODO : implement this, then remove this assert
            EBM_ASSERT(FeatureTypeOrdinal == pFeatureInitialize->featureType); // TODO : implement this, then remove this assert

            ++iFeatureInitialize;
            ++pFeatureInitialize;
         } while(pFeatureEnd != pFeatureInitialize);
      }
      LOG_0(TraceLevelInfo, "InitializeInteraction done feature processing");

      if(0 != cInstances) {
         const bool bError = m_dataSet.Initialize(
            m_cFeatures, 
            m_aFeatures, 
            cInstances, 
            aBinnedData, 
            aTargets, 
            aPredictorScores, 
            m_runtimeLearningTypeOrCountTargetClasses
         );
         if(bError) {
            LOG_0(TraceLevelWarning, "WARNING InitializeInteraction m_dataSet.Initialize");
            return true;
         }
      }

      LOG_0(TraceLevelInfo, "Exited InitializeInteraction");
      return false;
   }
};

#endif // EBM_INTERACTION_STATE_H
