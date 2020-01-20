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
   FeatureCore * const m_aFeatures;
   DataSetByFeature * m_pDataSet;

   unsigned int m_cLogEnterMessages;
   unsigned int m_cLogExitMessages;

   EBM_INLINE EbmInteractionState(const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, const size_t cFeatures)
      : m_runtimeLearningTypeOrCountTargetClasses(runtimeLearningTypeOrCountTargetClasses)
      , m_cFeatures(cFeatures)
      , m_aFeatures(0 == cFeatures || IsMultiplyError(sizeof(FeatureCore), cFeatures) ? nullptr : static_cast<FeatureCore *>(malloc(sizeof(FeatureCore) * cFeatures)))
      , m_pDataSet(nullptr)
      , m_cLogEnterMessages(1000)
      , m_cLogExitMessages(1000) {
   }

   EBM_INLINE ~EbmInteractionState() {
      LOG_0(TraceLevelInfo, "Entered ~EbmInteractionState");

      delete m_pDataSet;
      free(m_aFeatures);

      LOG_0(TraceLevelInfo, "Exited ~EbmInteractionState");
   }

   EBM_INLINE bool InitializeInteraction(const EbmCoreFeature * const aFeatures, const size_t cInstances, const void * const aTargets, const IntegerDataType * const aBinnedData, const FractionalDataType * const aPredictorScores) {
      LOG_0(TraceLevelInfo, "Entered InitializeInteraction");

      if(0 != m_cFeatures && nullptr == m_aFeatures) {
         LOG_0(TraceLevelWarning, "WARNING InitializeInteraction 0 != m_cFeatures && nullptr == m_aFeatures");
         return true;
      }

      LOG_0(TraceLevelInfo, "InitializeInteraction starting feature processing");
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
            EBM_ASSERT(0 <= countBins); // we can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on (dimensions with 1 bin don't contribute anything since they always have the same value)
            if(!IsNumberConvertable<size_t, IntegerDataType>(countBins)) {
               LOG_0(TraceLevelWarning, "WARNING InitializeInteraction !IsNumberConvertable<size_t, IntegerDataType>(countBins)");
               return true;
            }
            size_t cBins = static_cast<size_t>(countBins);
            if(cBins <= 1) {
               EBM_ASSERT(0 != cBins || 0 == cInstances);
               LOG_0(TraceLevelInfo, "INFO InitializeInteraction feature with 0/1 value");
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
      LOG_0(TraceLevelInfo, "InitializeInteraction done feature processing");

      LOG_0(TraceLevelInfo, "Entered DataSetByFeature");
      EBM_ASSERT(nullptr == m_pDataSet);
      if(0 != cInstances) {
         m_pDataSet = new (std::nothrow) DataSetByFeature(m_cFeatures, m_aFeatures, cInstances, aBinnedData, aTargets, aPredictorScores, m_runtimeLearningTypeOrCountTargetClasses);
         if(nullptr == m_pDataSet || m_pDataSet->IsError()) {
            LOG_0(TraceLevelWarning, "WARNING InitializeInteraction nullptr == pDataSet || pDataSet->IsError()");
            return true;
         }
      }
      LOG_0(TraceLevelInfo, "Exited DataSetByFeature");

      LOG_0(TraceLevelInfo, "Exited InitializeInteraction");
      return false;
   }
};

#endif // EBM_INTERACTION_STATE_H