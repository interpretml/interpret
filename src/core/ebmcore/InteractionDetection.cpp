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
      , m_aAttributes(IsMultiplyError(sizeof(AttributeInternalCore), cAttributes) ? nullptr : static_cast<AttributeInternalCore *>(malloc(sizeof(AttributeInternalCore) * cAttributes)))
      , m_pDataSet(nullptr)
      , m_cLogEnterMessages (1000)
      , m_cLogExitMessages(1000) {
      assert(0 < cAttributes); // we can't allocate zero byte arrays.  This is checked when we were initially called, but I'm leaving it here again as documentation
   }

   ~TmlInteractionState() {
      LOG(TraceLevelInfo, "Entered ~EbmInteractionState");

      delete m_pDataSet;
      free(m_aAttributes);

      LOG(TraceLevelInfo, "Exited ~EbmInteractionState");
   }

   bool InitializeInteraction(const EbmAttribute * const aAttributes, const size_t cCases, const void * const aTargets, const IntegerDataType * const aInputData, const FractionalDataType * const aPredictionScores) {
      LOG(TraceLevelInfo, "Entered InitializeInteraction");

      if(nullptr == m_aAttributes) {
         LOG(TraceLevelWarning, "WARNING InitializeInteraction nullptr == m_aAttributes");
         return true;
      }

      LOG(TraceLevelInfo, "InitializeInteraction starting attribute processing");
      assert(!IsMultiplyError(m_cAttributes, sizeof(*aAttributes))); // if this overflows then our caller should not have been able to allocate the array
      const EbmAttribute * pAttributeInitialize = aAttributes;
      const EbmAttribute * const pAttributeEnd = &aAttributes[m_cAttributes];
      assert(pAttributeInitialize < pAttributeEnd);
      size_t iAttributeInitialize = 0;
      do {
         static_assert(AttributeTypeCore::OrdinalCore == static_cast<AttributeTypeCore>(AttributeTypeOrdinal), "AttributeTypeCore::OrdinalCore must have the same value as AttributeTypeOrdinal");
         static_assert(AttributeTypeCore::NominalCore == static_cast<AttributeTypeCore>(AttributeTypeNominal), "AttributeTypeCore::NominalCore must have the same value as AttributeTypeNominal");
         assert(AttributeTypeOrdinal == pAttributeInitialize->attributeType || AttributeTypeNominal == pAttributeInitialize->attributeType);
         AttributeTypeCore attributeTypeCore = static_cast<AttributeTypeCore>(pAttributeInitialize->attributeType);

         IntegerDataType countStates = pAttributeInitialize->countStates;
         assert(1 <= countStates); // we can handle 1 == cStates even though that's a degenerate case that shouldn't be trained on (dimensions with 1 state don't contribute anything since they always have the same value)
         if(1 == countStates) {
            LOG(TraceLevelError, "ERROR InitializeInteraction Our higher level caller should filter out features with a single state since these provide no useful information for interactions");
         }
         if(!IsNumberConvertable<size_t, IntegerDataType>(countStates)) {
            LOG(TraceLevelWarning, "WARNING InitializeInteraction !IsNumberConvertable<size_t, IntegerDataType>(countStates)");
            return true;
         }
         size_t cStates = static_cast<size_t>(countStates);

         assert(0 == pAttributeInitialize->hasMissing || 1 == pAttributeInitialize->hasMissing);
         bool bMissing = 0 != pAttributeInitialize->hasMissing;

         AttributeInternalCore * pAttribute = new (&m_aAttributes[iAttributeInitialize]) AttributeInternalCore(cStates, iAttributeInitialize, attributeTypeCore, bMissing);
         // we don't allocate memory and our constructor doesn't have errors, so we shouldn't have an error here

         assert(0 == pAttributeInitialize->hasMissing); // TODO : implement this, then remove this assert
         assert(AttributeTypeOrdinal == pAttributeInitialize->attributeType); // TODO : implement this, then remove this assert

         ++iAttributeInitialize;
         ++pAttributeInitialize;
      } while(pAttributeEnd != pAttributeInitialize);
      LOG(TraceLevelInfo, "InitializeInteraction done attribute processing");

      LOG(TraceLevelInfo, "Entered DataSetInternalCore");
      DataSetInternalCore * pDataSet = new (std::nothrow) DataSetInternalCore(m_bRegression, m_cAttributes, m_aAttributes, cCases, aInputData, aTargets, aPredictionScores, m_cTargetStates, k_iZeroResidual);
      if(nullptr == pDataSet || pDataSet->IsError()) {
         LOG(TraceLevelWarning, "WARNING InitializeInteraction nullptr == pDataSet || pDataSet->IsError()");
         return true;
      }
      LOG(TraceLevelInfo, "Exited DataSetInternalCore");

      assert(nullptr == m_pDataSet);
      m_pDataSet = pDataSet;

      LOG(TraceLevelInfo, "Exited InitializeInteraction");
      return false;
   }
};





// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
TmlInteractionState * AllocateCoreInteraction(bool bRegression, IntegerDataType countAttributes, const EbmAttribute * attributes, IntegerDataType countTargetStates, IntegerDataType countCases, const void * targets, const IntegerDataType * data, const FractionalDataType * predictionScores) {
   // bRegression is set in our program, so our caller can't pass in invalid data
   assert(1 <= countAttributes);
   assert(nullptr != attributes);
   assert(bRegression || 2 <= countTargetStates);
   assert(1 <= countCases);
   assert(nullptr != targets);
   assert(nullptr != data);
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

EBMCORE_IMPORT_EXPORT PEbmInteraction EBMCORE_CALLING_CONVENTION InitializeInteractionRegression(IntegerDataType countAttributes, const EbmAttribute * attributes, IntegerDataType countCases, const FractionalDataType * targets, const IntegerDataType * data, const FractionalDataType * predictionScores) {
   LOG(TraceLevelInfo, "Entered InitializeInteractionRegression: countAttributes=%" IntegerDataTypePrintf ", attributes=%p, countCases=%" IntegerDataTypePrintf ", targets=%p, data=%p, predictionScores=%p", countAttributes, static_cast<const void *>(attributes), countCases, static_cast<const void *>(targets), static_cast<const void *>(data), static_cast<const void *>(predictionScores));
   PEbmInteraction pEbmInteraction = reinterpret_cast<PEbmInteraction>(AllocateCoreInteraction(true, countAttributes, attributes, 0, countCases, targets, data, predictionScores));
   LOG(TraceLevelInfo, "Exited InitializeInteractionRegression %p", static_cast<void *>(pEbmInteraction));
   return pEbmInteraction;
}

EBMCORE_IMPORT_EXPORT PEbmInteraction EBMCORE_CALLING_CONVENTION InitializeInteractionClassification(IntegerDataType countAttributes, const EbmAttribute * attributes, IntegerDataType countTargetStates, IntegerDataType countCases, const IntegerDataType * targets, const IntegerDataType * data, const FractionalDataType * predictionScores) {
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
   assert(IsClassification(iPossibleCompilerOptimizedTargetStates));
   if(cRuntimeTargetStates == iPossibleCompilerOptimizedTargetStates) {
      assert(cRuntimeTargetStates <= k_cCompilerOptimizedTargetStatesMax);
      return GetInteractionScorePerTargetStates<iPossibleCompilerOptimizedTargetStates>(pEbmInteractionState, pAttributeCombination, pInteractionScoreReturn);
   } else {
      return CompilerRecursiveGetInteractionScore<iPossibleCompilerOptimizedTargetStates + 1>(cRuntimeTargetStates, pEbmInteractionState, pAttributeCombination, pInteractionScoreReturn);
   }
}
template<>
TML_INLINE IntegerDataType CompilerRecursiveGetInteractionScore<k_cCompilerOptimizedTargetStatesMax + 1>(const size_t cRuntimeTargetStates, TmlInteractionState * const pEbmInteractionState, const AttributeCombinationCore * const pAttributeCombination, FractionalDataType * const pInteractionScoreReturn) {
   // it is logically possible, but uninteresting to have a classification with 1 target state, so let our runtime system handle those unlikley and uninteresting cases
   assert(k_cCompilerOptimizedTargetStatesMax < cRuntimeTargetStates || 1 == cRuntimeTargetStates);
   return GetInteractionScorePerTargetStates<k_DynamicClassification>(pEbmInteractionState, pAttributeCombination, pInteractionScoreReturn);
}

// we made this a global because if we had put this variable inside the TmlInteractionState object, then we would need to dereference that before getting the count.  By making this global we can send a log message incase a bad TmlInteractionState object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more times than desired, but we can live with that
static unsigned int g_cLogGetInteractionScoreParametersMessages = 10;

EBMCORE_IMPORT_EXPORT IntegerDataType EBMCORE_CALLING_CONVENTION GetInteractionScore(PEbmInteraction ebmInteraction, IntegerDataType countAttributesInCombination, const IntegerDataType * attributeIndexes, FractionalDataType * interactionScoreReturn) {
   LOG_COUNTED(&g_cLogGetInteractionScoreParametersMessages, TraceLevelInfo, TraceLevelVerbose, "GetInteractionScore parameters: ebmInteraction=%p, countAttributesInCombination=%" IntegerDataTypePrintf ", attributeIndexes=%p, interactionScoreReturn=%p", static_cast<void *>(ebmInteraction), countAttributesInCombination, static_cast<const void *>(attributeIndexes), static_cast<void *>(interactionScoreReturn));

   assert(nullptr != ebmInteraction);
   TmlInteractionState * pEbmInteractionState = reinterpret_cast<TmlInteractionState *>(ebmInteraction);

   LOG_COUNTED(&pEbmInteractionState->m_cLogEnterMessages, TraceLevelInfo, TraceLevelVerbose, "Entered GetInteractionScore");

   assert(1 <= countAttributesInCombination);
   assert(nullptr != attributeIndexes);

   if(!IsNumberConvertable<size_t, IntegerDataType>(countAttributesInCombination)) {
      LOG(TraceLevelWarning, "WARNING GetInteractionScore !IsNumberConvertable<size_t, IntegerDataType>(countAttributesInCombination)");
      return 1;
   }
   size_t cAttributesInCombination = static_cast<size_t>(countAttributesInCombination);
   if(k_cDimensionsMax < cAttributesInCombination) {
      // if we try to run with more than k_cDimensionsMax we'll exceed our memory capacity, so let's exit here instead
      LOG(TraceLevelWarning, "WARNING GetInteractionScore k_cDimensionsMax < cAttributesInCombination");
      return 1;
   }

   // TODO : !! change our code so that we don't need to allocate an AttributeCombinationCore each time we do an interaction score calculation
   AttributeCombinationCore * pAttributeCombination = AttributeCombinationCore::Allocate(cAttributesInCombination, 0);
   if(nullptr == pAttributeCombination) {
      LOG(TraceLevelWarning, "WARNING GetInteractionScore nullptr == pAttributeCombination");
      return 1;
   }
   AttributeInternalCore * const aAttributes = pEbmInteractionState->m_aAttributes;
   for(size_t iAttributeInCombination = 0; iAttributeInCombination < cAttributesInCombination; ++iAttributeInCombination) {
      IntegerDataType indexAttributeInterop = attributeIndexes[iAttributeInCombination];
      assert(0 <= indexAttributeInterop);
      if(!IsNumberConvertable<size_t, IntegerDataType>(indexAttributeInterop)) {
         LOG(TraceLevelWarning, "WARNING GetInteractionScore !IsNumberConvertable<size_t, IntegerDataType>(indexAttributeInterop)");
         AttributeCombinationCore::Free(pAttributeCombination);
         return 1;
      }
      // we already checked indexAttributeInterop was good above
      size_t iAttributeForCombination = static_cast<size_t>(indexAttributeInterop);
      assert(iAttributeForCombination < pEbmInteractionState->m_cAttributes);
      AttributeInternalCore * const pAttribute = &aAttributes[iAttributeForCombination];
      pAttributeCombination->m_AttributeCombinationEntry[iAttributeInCombination].m_pAttribute = pAttribute;
   }

   IntegerDataType ret;
   if(pEbmInteractionState->m_bRegression) {
      ret = GetInteractionScorePerTargetStates<k_Regression>(pEbmInteractionState, pAttributeCombination, interactionScoreReturn);
   } else {
      const size_t cTargetStates = pEbmInteractionState->m_cTargetStates;
      ret = CompilerRecursiveGetInteractionScore<2>(cTargetStates, pEbmInteractionState, pAttributeCombination, interactionScoreReturn);
   }
   AttributeCombinationCore::Free(pAttributeCombination);

   if(0 != ret) {
      LOG(TraceLevelWarning, "WARNING GetInteractionScore returned %" IntegerDataTypePrintf, ret);
   }
   LOG_COUNTED(&pEbmInteractionState->m_cLogExitMessages, TraceLevelInfo, TraceLevelVerbose, "Exited GetInteractionScore %" FractionalDataTypePrintf, *interactionScoreReturn);
   return ret;
}

EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION CancelInteraction(PEbmInteraction ebmInteraction) {
   LOG(TraceLevelInfo, "Entered CancelInteraction: ebmInteraction=%p", static_cast<void *>(ebmInteraction));
   TmlInteractionState * pEbmInteractionState = reinterpret_cast<TmlInteractionState *>(ebmInteraction);
   assert(nullptr != pEbmInteractionState);
   LOG(TraceLevelInfo, "Exited CancelInteraction");
}

EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION FreeInteraction(PEbmInteraction ebmInteraction) {
   LOG(TraceLevelInfo, "Entered FreeInteraction: ebmInteraction=%p", static_cast<void *>(ebmInteraction));
   TmlInteractionState * pEbmInteractionState = reinterpret_cast<TmlInteractionState *>(ebmInteraction);
   assert(nullptr != pEbmInteractionState);
   delete pEbmInteractionState;
   LOG(TraceLevelInfo, "Exited FreeInteraction");
}
