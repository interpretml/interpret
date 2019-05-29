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

class TmlInteractionState {
public:
   const bool m_bRegression;
   const size_t m_cTargetStates;

   const size_t m_cAttributes;
   // TODO : in the future, we can allocate this inside a function so that even the objects inside are const
   AttributeInternalCore * const m_aAttributes;
   DataSetInternalCore * m_pDataSet;

   TmlInteractionState(const bool bRegression, const size_t cTargetStates, const size_t cAttributes)
      : m_bRegression(bRegression)
      , m_cTargetStates(cTargetStates)
      , m_cAttributes(cAttributes)
      , m_aAttributes(IsMultiplyError(sizeof(AttributeInternalCore), cAttributes) ? nullptr : static_cast<AttributeInternalCore *>(malloc(sizeof(AttributeInternalCore) * cAttributes)))
      , m_pDataSet(nullptr) {
      assert(0 < cAttributes); // we can't allocate zero byte arrays.  This is checked when we were initially called, but I'm leaving it here again as documentation
   }

   ~TmlInteractionState() {
      delete m_pDataSet;
      free(m_aAttributes);
   }

   bool InitializeInteraction(const EbmAttribute * const aAttributes, const size_t cCases, const void * const aTargets, const IntegerDataType * const aInputData, const FractionalDataType * const aPredictionScores) {
      if(nullptr == m_aAttributes) {
         return true;
      }

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
         assert(2 <= countStates);
         if(!IsNumberConvertable<size_t, IntegerDataType>(countStates)) {
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


      DataSetInternalCore * pDataSet = new (std::nothrow) DataSetInternalCore(m_bRegression, m_cAttributes, m_aAttributes, cCases, aInputData, aTargets, aPredictionScores, m_cTargetStates, k_iZeroResidual);
      if(nullptr == pDataSet || pDataSet->IsError()) {
         return true;
      }

      assert(nullptr == m_pDataSet);
      m_pDataSet = pDataSet;

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
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntegerDataType>(countTargetStates)) {
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntegerDataType>(countCases)) {
      return nullptr;
   }

   size_t cAttributes = static_cast<size_t>(countAttributes);
   size_t cTargetStates = static_cast<size_t>(countTargetStates);
   size_t cCases = static_cast<size_t>(countCases);

   TmlInteractionState * const PEbmInteractionState = new (std::nothrow) TmlInteractionState(bRegression, cTargetStates, cAttributes);
   if(UNLIKELY(nullptr == PEbmInteractionState)) {
      return nullptr;
   }
   if(UNLIKELY(PEbmInteractionState->InitializeInteraction(attributes, cCases, targets, data, predictionScores))) {
      delete PEbmInteractionState;
      return nullptr;
   }
   return PEbmInteractionState;
}

EBMCORE_IMPORT_EXPORT PEbmInteraction EBMCORE_CALLING_CONVENTION InitializeInteractionRegression(IntegerDataType countAttributes, const EbmAttribute * attributes, IntegerDataType countCases, const FractionalDataType * targets, const IntegerDataType * data, const FractionalDataType * predictionScores) {
   return reinterpret_cast<PEbmInteraction>(AllocateCoreInteraction(true, countAttributes, attributes, 0, countCases, targets, data, predictionScores));
}

EBMCORE_IMPORT_EXPORT PEbmInteraction EBMCORE_CALLING_CONVENTION InitializeInteractionClassification(IntegerDataType countAttributes, const EbmAttribute * attributes, IntegerDataType countTargetStates, IntegerDataType countCases, const IntegerDataType * targets, const IntegerDataType * data, const FractionalDataType * predictionScores) {
   return reinterpret_cast<PEbmInteraction>(AllocateCoreInteraction(false, countAttributes, attributes, countTargetStates, countCases, targets, data, predictionScores));
}

template<ptrdiff_t countCompilerClassificationTargetStates>
static IntegerDataType GetInteractionScorePerTargetStates(TmlInteractionState * const PEbmInteractionState, const AttributeCombinationCore * const pAttributeCombination, FractionalDataType * const interactionScoreReturn) {
   // TODO : be smarter about our CachedInteractionThreadResources, otherwise why have it?
   CachedInteractionThreadResources * const pCachedThreadResources = new CachedInteractionThreadResources();

   if(CalculateInteractionScore<countCompilerClassificationTargetStates, 0>(PEbmInteractionState->m_cTargetStates, pCachedThreadResources, PEbmInteractionState->m_pDataSet, pAttributeCombination, interactionScoreReturn)) {
      delete pCachedThreadResources;
      return 1;
   }
   delete pCachedThreadResources;
   return 0;
}

template<ptrdiff_t iPossibleCompilerOptimizedTargetStates>
TML_INLINE IntegerDataType CompilerRecursiveGetInteractionScore(const size_t cRuntimeTargetStates, TmlInteractionState * const PEbmInteractionState, const AttributeCombinationCore * const pAttributeCombination, FractionalDataType * const interactionScoreReturn) {
   assert(IsClassification(iPossibleCompilerOptimizedTargetStates));
   if(cRuntimeTargetStates == iPossibleCompilerOptimizedTargetStates) {
      assert(cRuntimeTargetStates <= k_cCompilerOptimizedTargetStatesMax);
      return GetInteractionScorePerTargetStates<iPossibleCompilerOptimizedTargetStates>(PEbmInteractionState, pAttributeCombination, interactionScoreReturn);
   } else {
      return CompilerRecursiveGetInteractionScore<iPossibleCompilerOptimizedTargetStates + 1>(cRuntimeTargetStates, PEbmInteractionState, pAttributeCombination, interactionScoreReturn);
   }
}
template<>
TML_INLINE IntegerDataType CompilerRecursiveGetInteractionScore<k_cCompilerOptimizedTargetStatesMax + 1>(const size_t cRuntimeTargetStates, TmlInteractionState * const PEbmInteractionState, const AttributeCombinationCore * const pAttributeCombination, FractionalDataType * const interactionScoreReturn) {
   // it is logically possible, but uninteresting to have a classification with 1 target state, so let our runtime system handle those unlikley and uninteresting cases
   assert(k_cCompilerOptimizedTargetStatesMax < cRuntimeTargetStates || 1 == cRuntimeTargetStates);
   return GetInteractionScorePerTargetStates<k_DynamicClassification>(PEbmInteractionState, pAttributeCombination, interactionScoreReturn);
}

EBMCORE_IMPORT_EXPORT IntegerDataType EBMCORE_CALLING_CONVENTION GetInteractionScore(PEbmInteraction ebmInteraction, IntegerDataType countAttributesInCombination, const IntegerDataType * attributeIndexes, FractionalDataType * interactionScoreReturn) {
   TmlInteractionState * PEbmInteractionState = reinterpret_cast<TmlInteractionState *>(ebmInteraction);
   assert(nullptr != PEbmInteractionState);
   assert(1 <= countAttributesInCombination);
   assert(nullptr != attributeIndexes);

   if(!IsNumberConvertable<size_t, IntegerDataType>(countAttributesInCombination)) {
      return 1;
   }
   size_t cAttributesInCombination = static_cast<size_t>(countAttributesInCombination);
   if(k_cDimensionsMax < cAttributesInCombination) {
      // if we try to run with more than k_cDimensionsMax we'll exceed our memory capacity, so let's exit here instead
      return 1;
   }

   // TODO : !! change our code so that we don't need to allocate an AttributeCombinationCore each time we do an interaction score calculation
   AttributeCombinationCore * pAttributeCombination = AttributeCombinationCore::Allocate(cAttributesInCombination, 0);
   if(nullptr == pAttributeCombination) {
      return 1;
   }
   for(size_t iAttributeInCombination = 0; iAttributeInCombination < cAttributesInCombination; ++iAttributeInCombination) {
      IntegerDataType indexAttributeInterop = attributeIndexes[iAttributeInCombination];
      assert(0 <= indexAttributeInterop);
      if(!IsNumberConvertable<size_t, IntegerDataType>(indexAttributeInterop)) {
         AttributeCombinationCore::Free(pAttributeCombination);
         return 1;
      }
      // we already checked indexAttributeInterop was good above
      size_t iAttributeForCombination = static_cast<size_t>(indexAttributeInterop);
      pAttributeCombination->m_AttributeCombinationEntry[iAttributeInCombination].m_pAttribute = &PEbmInteractionState->m_aAttributes[iAttributeForCombination];
   }

   IntegerDataType ret;
   if(PEbmInteractionState->m_bRegression) {
      ret = GetInteractionScorePerTargetStates<k_Regression>(PEbmInteractionState, pAttributeCombination, interactionScoreReturn);
   } else {
      const size_t cTargetStates = PEbmInteractionState->m_cTargetStates;
      ret = CompilerRecursiveGetInteractionScore<2>(cTargetStates, PEbmInteractionState, pAttributeCombination, interactionScoreReturn);
   }
   AttributeCombinationCore::Free(pAttributeCombination);
   return ret;
}

EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION CancelInteraction(PEbmInteraction ebmInteraction) {
   TmlInteractionState * PEbmInteractionState = reinterpret_cast<TmlInteractionState *>(ebmInteraction);
   assert(nullptr != PEbmInteractionState);
}

EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION FreeInteraction(PEbmInteraction ebmInteraction) {
   TmlInteractionState * PEbmInteractionState = reinterpret_cast<TmlInteractionState *>(ebmInteraction);
   assert(nullptr != PEbmInteractionState);
   delete PEbmInteractionState;
}
