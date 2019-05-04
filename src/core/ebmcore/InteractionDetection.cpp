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
#include "AttributeSet.h"
// dataset depends on attributes
#include "DataSetByAttribute.h"
// depends on the above
#include "MultiDimensionalTraining.h"

// TODO: can this be merged with the InitializeInteractionErrorCore function in the TransparentMLCoreTraining.cpp file
// TODO: handle null in pPredictionScores like we did for training
//
// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
template<ptrdiff_t countCompilerClassificationTargetStates>
static void InitializeInteractionErrorCore(DataSetInternalCore * const pDataSet, const FractionalDataType * pPredictionScores, int iZeroResidual) {
   const size_t cTargetStates = pDataSet->m_pAttributeSet->m_cTargetStates;
   const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
   FractionalDataType * pResidualError = pDataSet->GetResidualPointer();
   const FractionalDataType * const pResidualErrorEnd = pResidualError + cVectorLength * pDataSet->GetCountCases();

   if(IsRegression(countCompilerClassificationTargetStates)) {
      FractionalDataType * pTargetData = static_cast<FractionalDataType *>(pDataSet->GetTargetDataPointer());
      for(; pResidualErrorEnd != pResidualError; ++pTargetData) {
         FractionalDataType residualError = static_cast<FractionalDataType>(*pTargetData) - *pPredictionScores;
         *pResidualError = residualError;
         ++pResidualError;
         ++pPredictionScores;
      }
   } else {
      StorageDataTypeCore * pTargetData = static_cast<StorageDataTypeCore *>(pDataSet->GetTargetDataPointer());
      for(; pResidualErrorEnd != pResidualError; ++pTargetData) {
         assert(IsClassification(countCompilerClassificationTargetStates));
         if(IsBinaryClassification(countCompilerClassificationTargetStates)) {
            FractionalDataType residualError = ComputeClassificationResidualErrorBinaryclass(*pPredictionScores, *pTargetData);
            *pResidualError = residualError;
            ++pResidualError;
            ++pPredictionScores;
         } else {
            FractionalDataType sumExp = 0;
            const FractionalDataType * pPredictionScoresTemp = pPredictionScores;
            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               sumExp += std::exp(*pPredictionScoresTemp);
               ++pPredictionScoresTemp;
            }

            assert((IsNumberConvertable<StorageDataTypeCore, size_t>(cVectorLength)));
            const StorageDataTypeCore cVectorLengthStorage = static_cast<StorageDataTypeCore>(cVectorLength);
            for(StorageDataTypeCore iVector = 0; iVector < cVectorLengthStorage; ++iVector) {
               const FractionalDataType residualError = ComputeClassificationResidualErrorMulticlass(sumExp, *pPredictionScores, *pTargetData, iVector);
               *pResidualError = residualError;
               ++pResidualError;
               ++pPredictionScores;
            }
            // TODO: this works as a way to remove one parameter, but it obviously insn't as efficient as omitting the parameter
            // 
            // this works out in the math as making the first model vector parameter equal to zero, which in turn removes one degree of freedom
            // from the model vector parameters.  Since the model vector weights need to be normalized to sum to a probabilty of 100%, we can set the first
            // one to the constant 1 (0 in log space) and force the other parameters to adjust to that scale which fixes them to a single valid set of values
            // insted of allowing them to be scaled.  
            // Probability = exp(T1 + I1) / [exp(T1 + I1) + exp(T2 + I2) + exp(T3 + I3)] => we can add a constant inside each exp(..) term, which will be multiplication outside the exp(..), which
            // means the numerator and denominator are multiplied by the same constant, which cancels eachother out.  We can thus set exp(T2 + I2) to exp(0) and adjust the other terms
            if(0 <= iZeroResidual) {
               pResidualError[static_cast<ptrdiff_t>(iZeroResidual) - static_cast<ptrdiff_t>(cVectorLength)] = 0;
            }
         }
      }
   }
}

class TmlInteractionState {
public:
   const bool m_bRegression;
   DataSetInternalCore * m_pDataSet;
   AttributeSetInternalCore * m_pAttributeSet;

   TmlInteractionState(const bool bRegression)
      : m_bRegression(bRegression)
      , m_pDataSet(nullptr)
      , m_pAttributeSet(nullptr) {
   }

   ~TmlInteractionState() {
      delete m_pDataSet;
      delete m_pAttributeSet;
   }

   bool InitializeInteraction(const size_t cAttributes, const EbmAttribute * const aAttributes, const size_t cTargetStates, const size_t cCases, const void * const aTargets, const IntegerDataType * const aData, const FractionalDataType * const aPredictionScores) {
      try {
         assert(nullptr == m_pAttributeSet);
         m_pAttributeSet = new (std::nothrow) AttributeSetInternalCore(cTargetStates);
         if(nullptr == m_pAttributeSet) {
            return true;
         }

         assert(!IsMultiplyError(cAttributes, sizeof(*aAttributes))); // if this overflows then our caller should not have been able to allocate the array
         const EbmAttribute * pAttributeInitialize = aAttributes;
         const EbmAttribute * const pAttributeEnd = &aAttributes[cAttributes];
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

            AttributeInternalCore * pAttribute = m_pAttributeSet->AddAttribute(cStates, iAttributeInitialize, attributeTypeCore, bMissing);
            if(nullptr == pAttribute) {
               return true;
            }

            assert(0 == pAttributeInitialize->hasMissing); // TODO : implement this, then remove this assert
            assert(AttributeTypeOrdinal == pAttributeInitialize->attributeType); // TODO : implement this, then remove this assert

            ++iAttributeInitialize;
            ++pAttributeInitialize;
         } while(pAttributeEnd != pAttributeInitialize);

         DataSetInternalCore * pDataSet = new (std::nothrow) DataSetInternalCore(m_pAttributeSet, cCases);
         if(nullptr == pDataSet) {
            return true;
         }

         size_t cVectorLength = GetVectorLengthFlatCore(cTargetStates);

         //if(pDataSet->Initialize(m_bRegression ? 0 : k_cBitsForSizeTCore, true, cVectorLength)) {
         //   return true;
         //}
         // TODO : can we eliminate the target values and just keep the residuals for interactions (we have a loop that stores the data below, but maybe we can just copy our input data to the residuals)
         if(pDataSet->Initialize(m_bRegression ? sizeof(FractionalDataType) * 8 : k_cBitsForSizeTCore, true, cVectorLength)) {
            return true;
         }

         assert(!IsMultiplyError(cAttributes, cCases)); // if this overflows then our caller should not have been able to allocate the array
         assert(!IsMultiplyError(cAttributes * cCases, sizeof(*aData))); // if this overflows then our caller should not have been able to allocate the array
                                                                                         // TODO : eliminate the counts here and use pointers
         for(size_t iAttribute = 0; iAttribute < cAttributes; ++iAttribute) {
            AttributeInternalCore * pAttribute = m_pAttributeSet->m_inputAttributes[iAttribute];
            StorageDataTypeCore * pData = pDataSet->GetDataPointer(pAttribute);
            // TODO : eliminate the counts here and use pointers
            for(size_t iCase = 0; iCase < cCases; ++iCase) {
               const IntegerDataType data = aData[iAttribute * cCases + iCase];
               assert(0 <= data);
               assert((IsNumberConvertable<size_t, IntegerDataType>(data))); // data must be lower than cTargetStates and cTargetStates fits into a size_t which we checked earlier
               assert(static_cast<size_t>(data) < pAttribute->m_cStates);
               assert((IsNumberConvertable<StorageDataTypeCore, IntegerDataType>(data)));
               pData[iCase] = static_cast<StorageDataTypeCore>(data);
            }
         }

         if(m_bRegression) {
            const FractionalDataType * pTarget = static_cast<const FractionalDataType *>(aTargets);
            FractionalDataType * pData = static_cast<FractionalDataType *>(pDataSet->GetTargetDataPointer());
            // TODO : do loop here
            for(size_t iCase = 0; iCase < cCases; ++iCase, ++pTarget) {
               const FractionalDataType data = *pTarget;
               assert(!std::isnan(data));
               assert(!std::isinf(data));
               pData[iCase] = data;
            }
         } else {
            const IntegerDataType * pTarget = static_cast<const IntegerDataType *>(aTargets);
            StorageDataTypeCore * pData = static_cast<StorageDataTypeCore *>(pDataSet->GetTargetDataPointer());
            // TODO : do loop here
            for(size_t iCase = 0; iCase < cCases; ++iCase, ++pTarget) {
               const IntegerDataType data = *pTarget;
               assert(0 <= data);
               assert((IsNumberConvertable<size_t, IntegerDataType>(data))); // data must be lower than cTargetStates and cTargetStates fits into a size_t which we checked earlier
               assert(static_cast<size_t>(data) < cTargetStates);
               assert((IsNumberConvertable<StorageDataTypeCore, IntegerDataType>(data)));
               pData[iCase] = static_cast<StorageDataTypeCore>(data);
            }
         }

         assert(nullptr == m_pDataSet);
         m_pDataSet = pDataSet;

         if(m_bRegression) {
            InitializeInteractionErrorCore<k_Regression>(pDataSet, aPredictionScores, k_iZeroResidual);
         } else {
            if(2 == cTargetStates) {
               InitializeInteractionErrorCore<2>(pDataSet, aPredictionScores, k_iZeroResidual);
            } else {
               InitializeInteractionErrorCore<k_DynamicClassification>(pDataSet, aPredictionScores, k_iZeroResidual);
            }
         }

         return false;
      } catch(...) {
         // this is here to catch exceptions from (TODO is this required?), but it could also catch errors if we put any other C++ types in here later
         return true;
      }
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

   TmlInteractionState * const PEbmInteractionState = new (std::nothrow) TmlInteractionState(bRegression);
   if(UNLIKELY(nullptr == PEbmInteractionState)) {
      return nullptr;
   }
   if(UNLIKELY(PEbmInteractionState->InitializeInteraction(cAttributes, attributes, cTargetStates, cCases, targets, data, predictionScores))) {
      delete PEbmInteractionState;
      return nullptr;
   }
   return PEbmInteractionState;
}

EBMCORE_IMPORT_EXPORT PEbmInteraction EBMCORE_CALLING_CONVENTION InitializeInteractionRegression(IntegerDataType countAttributes, const EbmAttribute* attributes, IntegerDataType countCases, const FractionalDataType* targets, const IntegerDataType* data, const FractionalDataType* predictionScores) {
   return reinterpret_cast<PEbmInteraction>(AllocateCoreInteraction(true, countAttributes, attributes, 0, countCases, targets, data, predictionScores));
}

EBMCORE_IMPORT_EXPORT PEbmInteraction EBMCORE_CALLING_CONVENTION InitializeInteractionClassification(IntegerDataType countAttributes, const EbmAttribute* attributes, IntegerDataType countTargetStates, IntegerDataType countCases, const IntegerDataType* targets, const IntegerDataType* data, const FractionalDataType* predictionScores) {
   return reinterpret_cast<PEbmInteraction>(AllocateCoreInteraction(false, countAttributes, attributes, countTargetStates, countCases, targets, data, predictionScores));
}

template<ptrdiff_t countCompilerClassificationTargetStates>
static IntegerDataType GetInteractionScorePerTargetStates(TmlInteractionState * const PEbmInteractionState, const AttributeCombinationCore * const pAttributeCombination, FractionalDataType * const interactionScoreReturn) {
   // TODO : be smarter about our CachedInteractionThreadResources, otherwise why have it?
   CachedInteractionThreadResources * const pCachedThreadResources = new CachedInteractionThreadResources();

   if(CalculateInteractionScore<countCompilerClassificationTargetStates, 0>(pCachedThreadResources, PEbmInteractionState->m_pDataSet, pAttributeCombination, interactionScoreReturn)) {
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

   AttributeCombinationCore * pAttributeCombination = AttributeCombinationCore::Allocate(cAttributesInCombination, 0);
   if(nullptr == pAttributeCombination) {
      return 1;
   }
   for(size_t iAttributeInCombination = 0; iAttributeInCombination < cAttributesInCombination; ++iAttributeInCombination) {
      IntegerDataType indexAttributeInterop = attributeIndexes[iAttributeInCombination];
      assert(0 <= indexAttributeInterop);
      if(!IsNumberConvertable<size_t, IntegerDataType>(indexAttributeInterop)) {
         return 1;
      }
      // we already checked indexAttributeInterop was good above
      size_t iAttributeForCombination = static_cast<size_t>(indexAttributeInterop);
      pAttributeCombination->m_AttributeCombinationEntry[iAttributeInCombination].m_pAttribute = PEbmInteractionState->m_pAttributeSet->m_inputAttributes[iAttributeForCombination];
   }

   if(PEbmInteractionState->m_bRegression) {
      return GetInteractionScorePerTargetStates<k_Regression>(PEbmInteractionState, pAttributeCombination, interactionScoreReturn);
   } else {
      size_t cTargetStates = PEbmInteractionState->m_pAttributeSet->m_cTargetStates;
      return CompilerRecursiveGetInteractionScore<2>(cTargetStates, PEbmInteractionState, pAttributeCombination, interactionScoreReturn);
   }
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
