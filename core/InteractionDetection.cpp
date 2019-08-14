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
#include "Logging.h" // EBM_ASSERT & LOG
// attribute includes
#include "AttributeInternal.h"
// dataset depends on attributes
#include "DataSetByAttribute.h"
// depends on the above
#include "MultiDimensionalTraining.h"

// TODO : rename this to EbmInteractionState
class TmlInteractionState {
public:
   const bool m_bRegression;
   const size_t m_cTargetStates;

   const size_t m_cAttributes;
   // TODO : in the future, we can allocate this inside a function so that even the objects inside are const
   AttributeInternalCore * const m_aAttributes;
   DataSetInternalCore * m_pDataSet;

   unsigned int m_cLogEnterMessages;
   unsigned int m_cLogExitMessages;

   TmlInteractionState(const bool bRegression, const size_t cTargetStates, const size_t cAttributes)
      : m_bRegression(bRegression)
      , m_cTargetStates(cTargetStates)
      , m_cAttributes(cAttributes)
      , m_aAttributes(0 == cAttributes || IsMultiplyError(sizeof(AttributeInternalCore), cAttributes) ? nullptr : static_cast<AttributeInternalCore *>(malloc(sizeof(AttributeInternalCore) * cAttributes)))
      , m_pDataSet(nullptr)
      , m_cLogEnterMessages (1000)
      , m_cLogExitMessages(1000) {
   }

   ~TmlInteractionState() {
      LOG(TraceLevelInfo, "Entered ~EbmInteractionState");

      delete m_pDataSet;
      free(m_aAttributes);

      LOG(TraceLevelInfo, "Exited ~EbmInteractionState");
   }

   bool InitializeInteraction(const EbmCoreFeature * const aAttributes, const size_t cCases, const void * const aTargets, const IntegerDataType * const aInputData, const FractionalDataType * const aPredictionScores) {
      LOG(TraceLevelInfo, "Entered InitializeInteraction");

      if(0 != m_cAttributes && nullptr == m_aAttributes) {
         LOG(TraceLevelWarning, "WARNING InitializeInteraction 0 != m_cAttributes && nullptr == m_aAttributes");
         return true;
      }

      LOG(TraceLevelInfo, "InitializeInteraction starting attribute processing");
      if(0 != m_cAttributes) {
         EBM_ASSERT(!IsMultiplyError(m_cAttributes, sizeof(*aAttributes))); // if this overflows then our caller should not have been able to allocate the array
         const EbmCoreFeature * pAttributeInitialize = aAttributes;
         const EbmCoreFeature * const pAttributeEnd = &aAttributes[m_cAttributes];
         EBM_ASSERT(pAttributeInitialize < pAttributeEnd);
         size_t iAttributeInitialize = 0;
         do {
            static_assert(FeatureTypeCore::OrdinalCore == static_cast<FeatureTypeCore>(FeatureTypeOrdinal), "FeatureTypeCore::OrdinalCore must have the same value as FeatureTypeOrdinal");
            static_assert(FeatureTypeCore::NominalCore == static_cast<FeatureTypeCore>(FeatureTypeNominal), "FeatureTypeCore::NominalCore must have the same value as FeatureTypeNominal");
            EBM_ASSERT(FeatureTypeOrdinal == pAttributeInitialize->featureType || FeatureTypeNominal == pAttributeInitialize->featureType);
            FeatureTypeCore featureTypeCore = static_cast<FeatureTypeCore>(pAttributeInitialize->featureType);

            IntegerDataType countStates = pAttributeInitialize->countBins;
            EBM_ASSERT(0 <= countStates); // we can handle 1 == cStates even though that's a degenerate case that shouldn't be trained on (dimensions with 1 state don't contribute anything since they always have the same value)
            if(!IsNumberConvertable<size_t, IntegerDataType>(countStates)) {
               LOG(TraceLevelWarning, "WARNING InitializeInteraction !IsNumberConvertable<size_t, IntegerDataType>(countStates)");
               return true;
            }
            size_t cStates = static_cast<size_t>(countStates);
            if(cStates <= 1) {
               EBM_ASSERT(0 != cStates || 0 == cCases);
               LOG(TraceLevelInfo, "INFO InitializeInteraction feature with 0/1 value");
            }

            EBM_ASSERT(0 == pAttributeInitialize->hasMissing || 1 == pAttributeInitialize->hasMissing);
            bool bMissing = 0 != pAttributeInitialize->hasMissing;

            // this is an in-place new, so there is no new memory allocated, and we already knew where it was going, so we don't need the resulting pointer returned
            new (&m_aAttributes[iAttributeInitialize]) AttributeInternalCore(cStates, iAttributeInitialize, featureTypeCore, bMissing);
            // we don't allocate memory and our constructor doesn't have errors, so we shouldn't have an error here

            EBM_ASSERT(0 == pAttributeInitialize->hasMissing); // TODO : implement this, then remove this assert
            EBM_ASSERT(FeatureTypeOrdinal == pAttributeInitialize->featureType); // TODO : implement this, then remove this assert

            ++iAttributeInitialize;
            ++pAttributeInitialize;
         } while(pAttributeEnd != pAttributeInitialize);
      }
      LOG(TraceLevelInfo, "InitializeInteraction done attribute processing");

      LOG(TraceLevelInfo, "Entered DataSetInternalCore");
      EBM_ASSERT(nullptr == m_pDataSet);
      if(0 != cCases) {
         m_pDataSet = new (std::nothrow) DataSetInternalCore(m_bRegression, m_cAttributes, m_aAttributes, cCases, aInputData, aTargets, aPredictionScores, m_cTargetStates);
         if(nullptr == m_pDataSet || m_pDataSet->IsError()) {
            LOG(TraceLevelWarning, "WARNING InitializeInteraction nullptr == pDataSet || pDataSet->IsError()");
            return true;
         }
      }
      LOG(TraceLevelInfo, "Exited DataSetInternalCore");

      LOG(TraceLevelInfo, "Exited InitializeInteraction");
      return false;
   }
};





// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
TmlInteractionState * AllocateCoreInteraction(bool bRegression, IntegerDataType countAttributes, const EbmCoreFeature * attributes, IntegerDataType countTargetStates, IntegerDataType countCases, const void * targets, const IntegerDataType * data, const FractionalDataType * predictionScores) {
   EBM_ASSERT(0 <= countAttributes);
   EBM_ASSERT(0 == countAttributes || nullptr != attributes);
   EBM_ASSERT(bRegression && 0 == countTargetStates || !bRegression && (1 <= countTargetStates || 0 == countTargetStates && 0 == countCases));
   EBM_ASSERT(0 <= countCases);
   EBM_ASSERT(0 == countCases || nullptr != targets);
   EBM_ASSERT(0 == countCases || 0 == countAttributes || nullptr != data);
   // predictionScores can be null

   if(!IsNumberConvertable<size_t, IntegerDataType>(countAttributes)) {
      LOG(TraceLevelWarning, "WARNING AllocateCoreInteraction !IsNumberConvertable<size_t, IntegerDataType>(countAttributes)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntegerDataType>(countTargetStates)) {
      LOG(TraceLevelWarning, "WARNING AllocateCoreInteraction !IsNumberConvertable<size_t, IntegerDataType>(countTargetStates)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntegerDataType>(countCases)) {
      LOG(TraceLevelWarning, "WARNING AllocateCoreInteraction !IsNumberConvertable<size_t, IntegerDataType>(countCases)");
      return nullptr;
   }

   size_t cAttributes = static_cast<size_t>(countAttributes);
   size_t cTargetStates = static_cast<size_t>(countTargetStates);
   size_t cCases = static_cast<size_t>(countCases);

   LOG(TraceLevelInfo, "Entered EbmInteractionState");
   TmlInteractionState * const pEbmInteractionState = new (std::nothrow) TmlInteractionState(bRegression, cTargetStates, cAttributes);
   LOG(TraceLevelInfo, "Exited EbmInteractionState %p", static_cast<void *>(pEbmInteractionState));
   if(UNLIKELY(nullptr == pEbmInteractionState)) {
      LOG(TraceLevelWarning, "WARNING AllocateCoreInteraction nullptr == pEbmInteractionState");
      return nullptr;
   }
   if(UNLIKELY(pEbmInteractionState->InitializeInteraction(attributes, cCases, targets, data, predictionScores))) {
      LOG(TraceLevelWarning, "WARNING AllocateCoreInteraction pEbmInteractionState->InitializeInteraction");
      delete pEbmInteractionState;
      return nullptr;
   }
   return pEbmInteractionState;
}

EBMCORE_IMPORT_EXPORT PEbmInteraction EBMCORE_CALLING_CONVENTION InitializeInteractionRegression(IntegerDataType countAttributes, const EbmCoreFeature * attributes, IntegerDataType countCases, const FractionalDataType * targets, const IntegerDataType * data, const FractionalDataType * predictionScores) {
   LOG(TraceLevelInfo, "Entered InitializeInteractionRegression: countAttributes=%" IntegerDataTypePrintf ", attributes=%p, countCases=%" IntegerDataTypePrintf ", targets=%p, data=%p, predictionScores=%p", countAttributes, static_cast<const void *>(attributes), countCases, static_cast<const void *>(targets), static_cast<const void *>(data), static_cast<const void *>(predictionScores));
   PEbmInteraction pEbmInteraction = reinterpret_cast<PEbmInteraction>(AllocateCoreInteraction(true, countAttributes, attributes, 0, countCases, targets, data, predictionScores));
   LOG(TraceLevelInfo, "Exited InitializeInteractionRegression %p", static_cast<void *>(pEbmInteraction));
   return pEbmInteraction;
}

EBMCORE_IMPORT_EXPORT PEbmInteraction EBMCORE_CALLING_CONVENTION InitializeInteractionClassification(IntegerDataType countAttributes, const EbmCoreFeature * attributes, IntegerDataType countTargetStates, IntegerDataType countCases, const IntegerDataType * targets, const IntegerDataType * data, const FractionalDataType * predictionScores) {
   LOG(TraceLevelInfo, "Entered InitializeInteractionClassification: countAttributes=%" IntegerDataTypePrintf ", attributes=%p, countTargetStates=%" IntegerDataTypePrintf ", countCases=%" IntegerDataTypePrintf ", targets=%p, data=%p, predictionScores=%p", countAttributes, static_cast<const void *>(attributes), countTargetStates, countCases, static_cast<const void *>(targets), static_cast<const void *>(data), static_cast<const void *>(predictionScores));
   PEbmInteraction pEbmInteraction = reinterpret_cast<PEbmInteraction>(AllocateCoreInteraction(false, countAttributes, attributes, countTargetStates, countCases, targets, data, predictionScores));
   LOG(TraceLevelInfo, "Exited InitializeInteractionClassification %p", static_cast<void *>(pEbmInteraction));
   return pEbmInteraction;
}

template<ptrdiff_t countCompilerClassificationTargetStates>
static IntegerDataType GetInteractionScorePerTargetStates(TmlInteractionState * const pEbmInteractionState, const AttributeCombinationCore * const pAttributeCombination, FractionalDataType * const pInteractionScoreReturn) {
   // TODO : be smarter about our CachedInteractionThreadResources, otherwise why have it?
   CachedInteractionThreadResources * const pCachedThreadResources = new (std::nothrow) CachedInteractionThreadResources();

   if(CalculateInteractionScore<countCompilerClassificationTargetStates, 0>(pEbmInteractionState->m_cTargetStates, pCachedThreadResources, pEbmInteractionState->m_pDataSet, pAttributeCombination, pInteractionScoreReturn)) {
      delete pCachedThreadResources;
      return 1;
   }
   delete pCachedThreadResources;
   return 0;
}

template<ptrdiff_t iPossibleCompilerOptimizedTargetStates>
TML_INLINE IntegerDataType CompilerRecursiveGetInteractionScore(const size_t cRuntimeTargetStates, TmlInteractionState * const pEbmInteractionState, const AttributeCombinationCore * const pAttributeCombination, FractionalDataType * const pInteractionScoreReturn) {
   EBM_ASSERT(IsClassification(iPossibleCompilerOptimizedTargetStates));
   if(cRuntimeTargetStates == iPossibleCompilerOptimizedTargetStates) {
      EBM_ASSERT(cRuntimeTargetStates <= k_cCompilerOptimizedTargetStatesMax);
      return GetInteractionScorePerTargetStates<iPossibleCompilerOptimizedTargetStates>(pEbmInteractionState, pAttributeCombination, pInteractionScoreReturn);
   } else {
      return CompilerRecursiveGetInteractionScore<iPossibleCompilerOptimizedTargetStates + 1>(cRuntimeTargetStates, pEbmInteractionState, pAttributeCombination, pInteractionScoreReturn);
   }
}

template<>
TML_INLINE IntegerDataType CompilerRecursiveGetInteractionScore<k_cCompilerOptimizedTargetStatesMax + 1>(const size_t cRuntimeTargetStates, TmlInteractionState * const pEbmInteractionState, const AttributeCombinationCore * const pAttributeCombination, FractionalDataType * const pInteractionScoreReturn) {
   UNUSED(cRuntimeTargetStates);
   // it is logically possible, but uninteresting to have a classification with 1 target state, so let our runtime system handle those unlikley and uninteresting cases
   EBM_ASSERT(k_cCompilerOptimizedTargetStatesMax < cRuntimeTargetStates);
   return GetInteractionScorePerTargetStates<k_DynamicClassification>(pEbmInteractionState, pAttributeCombination, pInteractionScoreReturn);
}

// we made this a global because if we had put this variable inside the TmlInteractionState object, then we would need to dereference that before getting the count.  By making this global we can send a log message incase a bad TmlInteractionState object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more times than desired, but we can live with that
static unsigned int g_cLogGetInteractionScoreParametersMessages = 10;

EBMCORE_IMPORT_EXPORT IntegerDataType EBMCORE_CALLING_CONVENTION GetInteractionScore(PEbmInteraction ebmInteraction, IntegerDataType countFeaturesInCombination, const IntegerDataType * attributeIndexes, FractionalDataType * interactionScoreReturn) {
   LOG_COUNTED(&g_cLogGetInteractionScoreParametersMessages, TraceLevelInfo, TraceLevelVerbose, "GetInteractionScore parameters: ebmInteraction=%p, countFeaturesInCombination=%" IntegerDataTypePrintf ", attributeIndexes=%p, interactionScoreReturn=%p", static_cast<void *>(ebmInteraction), countFeaturesInCombination, static_cast<const void *>(attributeIndexes), static_cast<void *>(interactionScoreReturn));

   EBM_ASSERT(nullptr != ebmInteraction);
   TmlInteractionState * pEbmInteractionState = reinterpret_cast<TmlInteractionState *>(ebmInteraction);

   LOG_COUNTED(&pEbmInteractionState->m_cLogEnterMessages, TraceLevelInfo, TraceLevelVerbose, "Entered GetInteractionScore");

   EBM_ASSERT(0 <= countFeaturesInCombination);
   EBM_ASSERT(0 == countFeaturesInCombination || nullptr != attributeIndexes);
   // interactionScoreReturn can be nullptr

   if(!IsNumberConvertable<size_t, IntegerDataType>(countFeaturesInCombination)) {
      LOG(TraceLevelWarning, "WARNING GetInteractionScore !IsNumberConvertable<size_t, IntegerDataType>(countFeaturesInCombination)");
      return 1;
   }
   size_t cFeaturesInCombination = static_cast<size_t>(countFeaturesInCombination);
   if(0 == cFeaturesInCombination) {
      LOG(TraceLevelInfo, "INFO GetInteractionScore empty attribute combination");
      if(nullptr != interactionScoreReturn) {
         *interactionScoreReturn = 0; // we return the lowest value possible for the interaction score, but we don't return an error since we handle it even though we'd prefer our caler be smarter about this condition
      }
      return 0;
   }

   if(nullptr == pEbmInteractionState->m_pDataSet) {
      // if pEbmInteractionState->m_pDataSet is null, then we have a dataset with zero cases.  If there are zero data cases, there isn't much basis to say whether there are interactions, so just return zero
      LOG(TraceLevelInfo, "INFO GetInteractionScore zero cases");
      if(nullptr != interactionScoreReturn) {
         *interactionScoreReturn = 0; // we return the lowest value possible for the interaction score, but we don't return an error since we handle it even though we'd prefer our caler be smarter about this condition
      }
      return 0;
   }

   const AttributeInternalCore * const aAttributes = pEbmInteractionState->m_aAttributes;
   const IntegerDataType * pAttributeCombinationIndex = attributeIndexes;
   const IntegerDataType * const pAttributeCombinationIndexEnd = attributeIndexes + cFeaturesInCombination;

   do {
      const IntegerDataType indexAttributeInterop = *pAttributeCombinationIndex;
      EBM_ASSERT(0 <= indexAttributeInterop);
      if(!IsNumberConvertable<size_t, IntegerDataType>(indexAttributeInterop)) {
         LOG(TraceLevelWarning, "WARNING GetInteractionScore !IsNumberConvertable<size_t, IntegerDataType>(indexAttributeInterop)");
         return 1;
      }
      size_t iAttributeForCombination = static_cast<size_t>(indexAttributeInterop);
      EBM_ASSERT(iAttributeForCombination < pEbmInteractionState->m_cAttributes);
      const AttributeInternalCore * const pAttribute = &aAttributes[iAttributeForCombination];
      if(pAttribute->m_cStates <= 1) {
         LOG(TraceLevelInfo, "INFO GetInteractionScore feature with 0/1 value");
         if(nullptr != interactionScoreReturn) {
            *interactionScoreReturn = 0; // we return the lowest value possible for the interaction score, but we don't return an error since we handle it even though we'd prefer our caler be smarter about this condition
         }
         return 0;
      }
      ++pAttributeCombinationIndex;
   } while(pAttributeCombinationIndexEnd != pAttributeCombinationIndex);

   if(k_cDimensionsMax < cFeaturesInCombination) {
      // if we try to run with more than k_cDimensionsMax we'll exceed our memory capacity, so let's exit here instead
      LOG(TraceLevelWarning, "WARNING GetInteractionScore k_cDimensionsMax < cFeaturesInCombination");
      return 1;
   }

   // put the pAttributeCombination object on the stack. We want to put it into a AttributeCombinationCore object since we want to share code with training, which calls things like building the tensor totals (which is templated to be compiled many times)
   char AttributeCombinationBuffer[k_cBytesAttributeCombinationMax];
   AttributeCombinationCore * const pAttributeCombination = reinterpret_cast<AttributeCombinationCore *>(&AttributeCombinationBuffer);
   pAttributeCombination->Initialize(cFeaturesInCombination, 0);

   pAttributeCombinationIndex = attributeIndexes; // restart from the start
   AttributeCombinationCore::AttributeCombinationEntry * pAttributeCombinationEntry = &pAttributeCombination->m_AttributeCombinationEntry[0];
   do {
      const IntegerDataType indexAttributeInterop = *pAttributeCombinationIndex;
      EBM_ASSERT(0 <= indexAttributeInterop);
      EBM_ASSERT((IsNumberConvertable<size_t, IntegerDataType>(indexAttributeInterop))); // we already checked indexAttributeInterop was good above
      size_t iAttributeForCombination = static_cast<size_t>(indexAttributeInterop);
      EBM_ASSERT(iAttributeForCombination < pEbmInteractionState->m_cAttributes);
      const AttributeInternalCore * const pAttribute = &aAttributes[iAttributeForCombination];
      EBM_ASSERT(2 <= pAttribute->m_cStates); // we should have filtered out anything with 1 state above

      pAttributeCombinationEntry->m_pAttribute = pAttribute;
      ++pAttributeCombinationEntry;
      ++pAttributeCombinationIndex;
   } while(pAttributeCombinationIndexEnd != pAttributeCombinationIndex);

   IntegerDataType ret;
   if(pEbmInteractionState->m_bRegression) {
      ret = GetInteractionScorePerTargetStates<k_Regression>(pEbmInteractionState, pAttributeCombination, interactionScoreReturn);
   } else {
      const size_t cTargetStates = pEbmInteractionState->m_cTargetStates;
      if(cTargetStates <= 1) {
         LOG(TraceLevelInfo, "INFO GetInteractionScore target with 0/1 states");
         if(nullptr != interactionScoreReturn) {
            *interactionScoreReturn = 0; // if there is only 1 classification target, then we can predict the outcome with 100% accuracy and there is no need for logits or interactions or anything else.  We return 0 since interactions have no benefit
         }
         return 0;
      }
      ret = CompilerRecursiveGetInteractionScore<2>(cTargetStates, pEbmInteractionState, pAttributeCombination, interactionScoreReturn);
   }
   if(0 != ret) {
      LOG(TraceLevelWarning, "WARNING GetInteractionScore returned %" IntegerDataTypePrintf, ret);
   }
   if(nullptr != interactionScoreReturn) {
      EBM_ASSERT(0 <= *interactionScoreReturn);
      LOG_COUNTED(&pEbmInteractionState->m_cLogExitMessages, TraceLevelInfo, TraceLevelVerbose, "Exited GetInteractionScore %" FractionalDataTypePrintf, *interactionScoreReturn);
   } else {
      LOG_COUNTED(&pEbmInteractionState->m_cLogExitMessages, TraceLevelInfo, TraceLevelVerbose, "Exited GetInteractionScore");
   }
   return ret;
}

EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION FreeInteraction(PEbmInteraction ebmInteraction) {
   LOG(TraceLevelInfo, "Entered FreeInteraction: ebmInteraction=%p", static_cast<void *>(ebmInteraction));
   TmlInteractionState * pEbmInteractionState = reinterpret_cast<TmlInteractionState *>(ebmInteraction);
   EBM_ASSERT(nullptr != pEbmInteractionState);
   delete pEbmInteractionState;
   LOG(TraceLevelInfo, "Exited FreeInteraction");
}
