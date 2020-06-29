// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "EbmInternal.h"
// very independent includes
#include "Logging.h" // EBM_ASSERT & LOG
#include "EbmStatisticUtils.h"
// FeatureCombination.h depends on FeatureInternal.h
#include "FeatureGroup.h"
// dataset depends on features
#include "DataSetBoosting.h"

#include "Booster.h"

// C++ does not allow partial function specialization, so we need to use these cumbersome static class functions to do partial function specialization

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class ApplyModelUpdateValidationZeroFeatures {
public:
   static FloatEbmType Func(
      EbmBoostingState * const pEbmBoostingState,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClasses), "must be classification");
      static_assert(!IsBinaryClassification(compilerLearningTypeOrCountTargetClasses), "must be multiclass");

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses();
      DataSetByFeatureCombination * const pValidationSet = pEbmBoostingState->GetValidationSet();

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      const size_t cInstances = pValidationSet->GetCountInstances();
      EBM_ASSERT(0 < cInstances);

      FloatEbmType sumLogLoss = FloatEbmType { 0 };
      const StorageDataType * pTargetData = pValidationSet->GetTargetDataPointer();
      FloatEbmType * pPredictorScores = pValidationSet->GetPredictorScores();
      const FloatEbmType * const pPredictorScoresEnd = pPredictorScores + cInstances * cVectorLength;
      do {
         size_t targetData = static_cast<size_t>(*pTargetData);
         ++pTargetData;

         const FloatEbmType * pValues = aModelFeatureCombinationUpdateTensor;
         FloatEbmType itemExp = FloatEbmType { 0 };
         FloatEbmType sumExp = FloatEbmType { 0 };
         size_t iVector = 0;
         do {
            // TODO : because there is only one bin for a zero feature feature combination, we could move these values to the stack where the
            // compiler could reason about their visibility and optimize small arrays into registers
            const FloatEbmType smallChangeToPredictorScores = *pValues;
            ++pValues;
            // this will apply a small fix to our existing ValidationPredictorScores, either positive or negative, whichever is needed
            const FloatEbmType predictorScore = *pPredictorScores + smallChangeToPredictorScores;
            *pPredictorScores = predictorScore;
            ++pPredictorScores;
            const FloatEbmType oneExp = EbmExp(predictorScore);
            itemExp = iVector == targetData ? oneExp : itemExp;
            sumExp += oneExp;
            ++iVector;
         } while(iVector < cVectorLength);
         const FloatEbmType instanceLogLoss = EbmStatistics::ComputeSingleInstanceLogLossMulticlass(
            sumExp,
            itemExp
         );

         EBM_ASSERT(std::isnan(instanceLogLoss) || -k_epsilonLogLoss <= instanceLogLoss);
         sumLogLoss += instanceLogLoss;

      } while(pPredictorScoresEnd != pPredictorScores);
      return sumLogLoss / cInstances;
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<>
class ApplyModelUpdateValidationZeroFeatures<2> {
public:
   static FloatEbmType Func(
      EbmBoostingState * const pEbmBoostingState,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      DataSetByFeatureCombination * const pValidationSet = pEbmBoostingState->GetValidationSet();
      const size_t cInstances = pValidationSet->GetCountInstances();
      EBM_ASSERT(0 < cInstances);

      FloatEbmType sumLogLoss = 0;
      const StorageDataType * pTargetData = pValidationSet->GetTargetDataPointer();
      FloatEbmType * pPredictorScores = pValidationSet->GetPredictorScores();
      const FloatEbmType * const pPredictorScoresEnd = pPredictorScores + cInstances;
      const FloatEbmType smallChangeToPredictorScores = aModelFeatureCombinationUpdateTensor[0];
      do {
         size_t targetData = static_cast<size_t>(*pTargetData);
         ++pTargetData;
         // this will apply a small fix to our existing ValidationPredictorScores, either positive or negative, whichever is needed
         const FloatEbmType predictorScore = *pPredictorScores + smallChangeToPredictorScores;
         *pPredictorScores = predictorScore;
         ++pPredictorScores;
         const FloatEbmType instanceLogLoss = EbmStatistics::ComputeSingleInstanceLogLossBinaryClassification(predictorScore, targetData);
         EBM_ASSERT(std::isnan(instanceLogLoss) || FloatEbmType { 0 } <= instanceLogLoss);
         sumLogLoss += instanceLogLoss;
      } while(pPredictorScoresEnd != pPredictorScores);
      return sumLogLoss / cInstances;
   }
};
#endif // EXPAND_BINARY_LOGITS

template<>
class ApplyModelUpdateValidationZeroFeatures<k_regression> {
public:
   static FloatEbmType Func(
      EbmBoostingState * const pEbmBoostingState,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      DataSetByFeatureCombination * const pValidationSet = pEbmBoostingState->GetValidationSet();
      const size_t cInstances = pValidationSet->GetCountInstances();
      EBM_ASSERT(0 < cInstances);

      FloatEbmType sumSquareError = FloatEbmType { 0 };
      FloatEbmType * pResidualError = pValidationSet->GetResidualPointer();
      const FloatEbmType * const pResidualErrorEnd = pResidualError + cInstances;
      const FloatEbmType smallChangeToPrediction = aModelFeatureCombinationUpdateTensor[0];
      do {
         // this will apply a small fix to our existing ValidationPredictorScores, either positive or negative, whichever is needed
         const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorRegression(*pResidualError - smallChangeToPrediction);
         const FloatEbmType instanceSquaredError = EbmStatistics::ComputeSingleInstanceSquaredErrorRegression(residualError);
         EBM_ASSERT(std::isnan(instanceSquaredError) || FloatEbmType { 0 } <= instanceSquaredError);
         sumSquareError += instanceSquaredError;
         *pResidualError = residualError;
         ++pResidualError;
      } while(pResidualErrorEnd != pResidualError);
      return sumSquareError / cInstances;
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class ApplyModelUpdateValidationZeroFeaturesTarget {
public:
   EBM_INLINE static FloatEbmType Func(
      EbmBoostingState * const pEbmBoostingState,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         return ApplyModelUpdateValidationZeroFeatures<compilerLearningTypeOrCountTargetClassesPossible>::Func(
            pEbmBoostingState,
            aModelFeatureCombinationUpdateTensor
         );
      } else {
         return ApplyModelUpdateValidationZeroFeaturesTarget<
            compilerLearningTypeOrCountTargetClassesPossible + 1
         >::Func(
            pEbmBoostingState,
            aModelFeatureCombinationUpdateTensor
         );
      }
   }
};

template<>
class ApplyModelUpdateValidationZeroFeaturesTarget<k_cCompilerOptimizedTargetClassesMax + 1> {
public:
   EBM_INLINE static FloatEbmType Func(
      EbmBoostingState * const pEbmBoostingState,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses());

      return ApplyModelUpdateValidationZeroFeatures<k_dynamicClassification>::Func(
         pEbmBoostingState,
         aModelFeatureCombinationUpdateTensor
      );
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerCountItemsPerBitPackedDataUnit>
class ApplyModelUpdateValidationInternal {
public:
   static FloatEbmType Func(
      EbmBoostingState * const pEbmBoostingState,
      const FeatureCombination * const pFeatureCombination,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClasses), "must be classification");
      static_assert(!IsBinaryClassification(compilerLearningTypeOrCountTargetClasses), "must be multiclass");

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses();
      const size_t runtimeCountItemsPerBitPackedDataUnit = pFeatureCombination->GetCountItemsPerBitPackedDataUnit();
      DataSetByFeatureCombination * const pValidationSet = pEbmBoostingState->GetValidationSet();

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      const size_t cInstances = pValidationSet->GetCountInstances();
      EBM_ASSERT(0 < cInstances);
      EBM_ASSERT(0 < pFeatureCombination->GetCountFeatures());

      const size_t cItemsPerBitPackedDataUnit = GET_COUNT_ITEMS_PER_BIT_PACKED_DATA_UNIT(
         compilerCountItemsPerBitPackedDataUnit,
         runtimeCountItemsPerBitPackedDataUnit
      );
      EBM_ASSERT(1 <= cItemsPerBitPackedDataUnit);
      EBM_ASSERT(cItemsPerBitPackedDataUnit <= k_cBitsForStorageType);
      const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackedDataUnit);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);

      FloatEbmType sumLogLoss = FloatEbmType { 0 };
      const StorageDataType * pInputData = pValidationSet->GetInputDataPointer(pFeatureCombination);
      const StorageDataType * pTargetData = pValidationSet->GetTargetDataPointer();
      FloatEbmType * pPredictorScores = pValidationSet->GetPredictorScores();

      // this shouldn't overflow since we're accessing existing memory
      const FloatEbmType * const pPredictorScoresTrueEnd = pPredictorScores + cInstances * cVectorLength;
      const FloatEbmType * pPredictorScoresExit = pPredictorScoresTrueEnd;
      const FloatEbmType * pPredictorScoresInnerEnd = pPredictorScoresTrueEnd;
      if(cInstances <= cItemsPerBitPackedDataUnit) {
         goto one_last_loop;
      }
      pPredictorScoresExit = pPredictorScoresTrueEnd - ((cInstances - 1) % cItemsPerBitPackedDataUnit + 1) * cVectorLength;
      EBM_ASSERT(pPredictorScores < pPredictorScoresExit);
      EBM_ASSERT(pPredictorScoresExit < pPredictorScoresTrueEnd);

      do {
         pPredictorScoresInnerEnd = pPredictorScores + cItemsPerBitPackedDataUnit * cVectorLength;
         // jumping back into this loop and changing pPredictorScoresInnerEnd to a dynamic value that isn't compile time determinable causes this 
         // function to NOT be optimized for templated cItemsPerBitPackedDataUnit, but that's ok since avoiding one unpredictable branch here is negligible
      one_last_loop:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iTensorBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            size_t targetData = static_cast<size_t>(*pTargetData);
            ++pTargetData;

            const size_t iTensorBin = maskBits & iTensorBinCombined;
            const FloatEbmType * pValues = &aModelFeatureCombinationUpdateTensor[iTensorBin * cVectorLength];
            FloatEbmType itemExp = FloatEbmType { 0 };
            FloatEbmType sumExp = FloatEbmType { 0 };
            size_t iVector = 0;
            do {
               const FloatEbmType smallChangeToPredictorScores = *pValues;
               ++pValues;
               // this will apply a small fix to our existing ValidationPredictorScores, either positive or negative, whichever is needed
               const FloatEbmType predictorScore = *pPredictorScores + smallChangeToPredictorScores;
               *pPredictorScores = predictorScore;
               ++pPredictorScores;
               const FloatEbmType oneExp = EbmExp(predictorScore);
               itemExp = iVector == targetData ? oneExp : itemExp;
               sumExp += oneExp;
               ++iVector;
            } while(iVector < cVectorLength);
            const FloatEbmType instanceLogLoss = EbmStatistics::ComputeSingleInstanceLogLossMulticlass(
               sumExp,
               itemExp
            );

            EBM_ASSERT(std::isnan(instanceLogLoss) || -k_epsilonLogLoss <= instanceLogLoss);
            sumLogLoss += instanceLogLoss;
            iTensorBinCombined >>= cBitsPerItemMax;
         } while(pPredictorScoresInnerEnd != pPredictorScores);
      } while(pPredictorScoresExit != pPredictorScores);

      // first time through?
      if(pPredictorScoresTrueEnd != pPredictorScores) {
         pPredictorScoresInnerEnd = pPredictorScoresTrueEnd;
         pPredictorScoresExit = pPredictorScoresTrueEnd;
         goto one_last_loop;
      }
      return sumLogLoss / cInstances;
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<size_t compilerCountItemsPerBitPackedDataUnit>
class ApplyModelUpdateValidationInternal<2, compilerCountItemsPerBitPackedDataUnit> {
public:
   static FloatEbmType Func(
      EbmBoostingState * const pEbmBoostingState,
      const FeatureCombination * const pFeatureCombination,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      const size_t runtimeCountItemsPerBitPackedDataUnit = pFeatureCombination->GetCountItemsPerBitPackedDataUnit();
      DataSetByFeatureCombination * const pValidationSet = pEbmBoostingState->GetValidationSet();

      const size_t cInstances = pValidationSet->GetCountInstances();
      EBM_ASSERT(0 < cInstances);
      EBM_ASSERT(0 < pFeatureCombination->GetCountFeatures());

      const size_t cItemsPerBitPackedDataUnit = GET_COUNT_ITEMS_PER_BIT_PACKED_DATA_UNIT(
         compilerCountItemsPerBitPackedDataUnit,
         runtimeCountItemsPerBitPackedDataUnit
      );
      EBM_ASSERT(1 <= cItemsPerBitPackedDataUnit);
      EBM_ASSERT(cItemsPerBitPackedDataUnit <= k_cBitsForStorageType);
      const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackedDataUnit);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);

      FloatEbmType sumLogLoss = FloatEbmType { 0 };
      const StorageDataType * pInputData = pValidationSet->GetInputDataPointer(pFeatureCombination);
      const StorageDataType * pTargetData = pValidationSet->GetTargetDataPointer();
      FloatEbmType * pPredictorScores = pValidationSet->GetPredictorScores();

      // this shouldn't overflow since we're accessing existing memory
      const FloatEbmType * const pPredictorScoresTrueEnd = pPredictorScores + cInstances;
      const FloatEbmType * pPredictorScoresExit = pPredictorScoresTrueEnd;
      const FloatEbmType * pPredictorScoresInnerEnd = pPredictorScoresTrueEnd;
      if(cInstances <= cItemsPerBitPackedDataUnit) {
         goto one_last_loop;
      }
      pPredictorScoresExit = pPredictorScoresTrueEnd - ((cInstances - 1) % cItemsPerBitPackedDataUnit + 1);
      EBM_ASSERT(pPredictorScores < pPredictorScoresExit);
      EBM_ASSERT(pPredictorScoresExit < pPredictorScoresTrueEnd);

      do {
         pPredictorScoresInnerEnd = pPredictorScores + cItemsPerBitPackedDataUnit;
         // jumping back into this loop and changing pPredictorScoresInnerEnd to a dynamic value that isn't compile time determinable causes this 
         // function to NOT be optimized for templated cItemsPerBitPackedDataUnit, but that's ok since avoiding one unpredictable branch here is negligible
      one_last_loop:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iTensorBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            size_t targetData = static_cast<size_t>(*pTargetData);
            ++pTargetData;

            const size_t iTensorBin = maskBits & iTensorBinCombined;

            const FloatEbmType smallChangeToPredictorScores = aModelFeatureCombinationUpdateTensor[iTensorBin];
            // this will apply a small fix to our existing ValidationPredictorScores, either positive or negative, whichever is needed
            const FloatEbmType predictorScore = *pPredictorScores + smallChangeToPredictorScores;
            *pPredictorScores = predictorScore;
            ++pPredictorScores;
            const FloatEbmType instanceLogLoss = EbmStatistics::ComputeSingleInstanceLogLossBinaryClassification(predictorScore, targetData);

            EBM_ASSERT(std::isnan(instanceLogLoss) || FloatEbmType { 0 } <= instanceLogLoss);
            sumLogLoss += instanceLogLoss;

            iTensorBinCombined >>= cBitsPerItemMax;
         } while(pPredictorScoresInnerEnd != pPredictorScores);
      } while(pPredictorScoresExit != pPredictorScores);

      // first time through?
      if(pPredictorScoresTrueEnd != pPredictorScores) {
         pPredictorScoresInnerEnd = pPredictorScoresTrueEnd;
         pPredictorScoresExit = pPredictorScoresTrueEnd;
         goto one_last_loop;
      }
      return sumLogLoss / cInstances;
   }
};
#endif // EXPAND_BINARY_LOGITS

template<size_t compilerCountItemsPerBitPackedDataUnit>
class ApplyModelUpdateValidationInternal<k_regression, compilerCountItemsPerBitPackedDataUnit> {
public:
   static FloatEbmType Func(
      EbmBoostingState * const pEbmBoostingState,
      const FeatureCombination * const pFeatureCombination,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      const size_t runtimeCountItemsPerBitPackedDataUnit = pFeatureCombination->GetCountItemsPerBitPackedDataUnit();
      DataSetByFeatureCombination * const pValidationSet = pEbmBoostingState->GetValidationSet();

      const size_t cInstances = pValidationSet->GetCountInstances();
      EBM_ASSERT(0 < cInstances);
      EBM_ASSERT(0 < pFeatureCombination->GetCountFeatures());

      const size_t cItemsPerBitPackedDataUnit = GET_COUNT_ITEMS_PER_BIT_PACKED_DATA_UNIT(
         compilerCountItemsPerBitPackedDataUnit,
         runtimeCountItemsPerBitPackedDataUnit
      );
      EBM_ASSERT(1 <= cItemsPerBitPackedDataUnit);
      EBM_ASSERT(cItemsPerBitPackedDataUnit <= k_cBitsForStorageType);
      const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackedDataUnit);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);

      FloatEbmType sumSquareError = FloatEbmType { 0 };
      FloatEbmType * pResidualError = pValidationSet->GetResidualPointer();
      const StorageDataType * pInputData = pValidationSet->GetInputDataPointer(pFeatureCombination);

      // this shouldn't overflow since we're accessing existing memory
      const FloatEbmType * const pResidualErrorTrueEnd = pResidualError + cInstances;
      const FloatEbmType * pResidualErrorExit = pResidualErrorTrueEnd;
      const FloatEbmType * pResidualErrorInnerEnd = pResidualErrorTrueEnd;
      if(cInstances <= cItemsPerBitPackedDataUnit) {
         goto one_last_loop;
      }
      pResidualErrorExit = pResidualErrorTrueEnd - ((cInstances - 1) % cItemsPerBitPackedDataUnit + 1);
      EBM_ASSERT(pResidualError < pResidualErrorExit);
      EBM_ASSERT(pResidualErrorExit < pResidualErrorTrueEnd);

      do {
         pResidualErrorInnerEnd = pResidualError + cItemsPerBitPackedDataUnit;
         // jumping back into this loop and changing pPredictorScoresInnerEnd to a dynamic value that isn't compile time determinable causes this 
         // function to NOT be optimized for templated cItemsPerBitPackedDataUnit, but that's ok since avoiding one unpredictable branch here is negligible
      one_last_loop:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iTensorBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            const size_t iTensorBin = maskBits & iTensorBinCombined;

            const FloatEbmType smallChangeToPrediction = aModelFeatureCombinationUpdateTensor[iTensorBin];
            // this will apply a small fix to our existing ValidationPredictorScores, either positive or negative, whichever is needed
            const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorRegression(*pResidualError - smallChangeToPrediction);
            const FloatEbmType instanceSquaredError = EbmStatistics::ComputeSingleInstanceSquaredErrorRegression(residualError);
            EBM_ASSERT(std::isnan(instanceSquaredError) || FloatEbmType { 0 } <= instanceSquaredError);
            sumSquareError += instanceSquaredError;
            *pResidualError = residualError;
            ++pResidualError;

            iTensorBinCombined >>= cBitsPerItemMax;
         } while(pResidualErrorInnerEnd != pResidualError);
      } while(pResidualErrorExit != pResidualError);

      // first time through?
      if(pResidualErrorTrueEnd != pResidualError) {
         pResidualErrorInnerEnd = pResidualErrorTrueEnd;
         pResidualErrorExit = pResidualErrorTrueEnd;
         goto one_last_loop;
      }
      return sumSquareError / cInstances;
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class ApplyModelUpdateValidationNormalTarget {
public:
   EBM_INLINE static FloatEbmType Func(
      EbmBoostingState * const pEbmBoostingState,
      const FeatureCombination * const pFeatureCombination,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         return ApplyModelUpdateValidationInternal<compilerLearningTypeOrCountTargetClassesPossible, k_cItemsPerBitPackedDataUnitDynamic>::Func(
            pEbmBoostingState,
            pFeatureCombination,
            aModelFeatureCombinationUpdateTensor
         );
      } else {
         return ApplyModelUpdateValidationNormalTarget<
            compilerLearningTypeOrCountTargetClassesPossible + 1
         >::Func(
            pEbmBoostingState,
            pFeatureCombination,
            aModelFeatureCombinationUpdateTensor
         );
      }
   }
};

template<>
class ApplyModelUpdateValidationNormalTarget<k_cCompilerOptimizedTargetClassesMax + 1> {
public:
   EBM_INLINE static FloatEbmType Func(
      EbmBoostingState * const pEbmBoostingState,
      const FeatureCombination * const pFeatureCombination,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses());

      return ApplyModelUpdateValidationInternal<k_dynamicClassification, k_cItemsPerBitPackedDataUnitDynamic>::Func(
         pEbmBoostingState,
         pFeatureCombination,
         aModelFeatureCombinationUpdateTensor
      );
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerCountItemsPerBitPackedDataUnitPossible>
class ApplyModelUpdateValidationSIMDPacking {
public:
   EBM_INLINE static FloatEbmType Func(
      EbmBoostingState * const pEbmBoostingState,
      const FeatureCombination * const pFeatureCombination,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      const size_t runtimeCountItemsPerBitPackedDataUnit = pFeatureCombination->GetCountItemsPerBitPackedDataUnit();

      EBM_ASSERT(1 <= runtimeCountItemsPerBitPackedDataUnit);
      EBM_ASSERT(runtimeCountItemsPerBitPackedDataUnit <= k_cBitsForStorageType);
      static_assert(compilerCountItemsPerBitPackedDataUnitPossible <= k_cBitsForStorageType, "We can't have this many items in a data pack.");
      if(compilerCountItemsPerBitPackedDataUnitPossible == runtimeCountItemsPerBitPackedDataUnit) {
         return ApplyModelUpdateValidationInternal<compilerLearningTypeOrCountTargetClasses, compilerCountItemsPerBitPackedDataUnitPossible>::Func(
            pEbmBoostingState,
            pFeatureCombination,
            aModelFeatureCombinationUpdateTensor
         );
      } else {
         return ApplyModelUpdateValidationSIMDPacking<
            compilerLearningTypeOrCountTargetClasses,
            GetNextCountItemsBitPacked(compilerCountItemsPerBitPackedDataUnitPossible)
         >::Func(
            pEbmBoostingState,
            pFeatureCombination,
            aModelFeatureCombinationUpdateTensor
         );
      }
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class ApplyModelUpdateValidationSIMDPacking<compilerLearningTypeOrCountTargetClasses, k_cItemsPerBitPackedDataUnitDynamic> {
public:
   EBM_INLINE static FloatEbmType Func(
      EbmBoostingState * const pEbmBoostingState,
      const FeatureCombination * const pFeatureCombination,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      EBM_ASSERT(1 <= pFeatureCombination->GetCountItemsPerBitPackedDataUnit());
      EBM_ASSERT(pFeatureCombination->GetCountItemsPerBitPackedDataUnit() <= k_cBitsForStorageType);
      return ApplyModelUpdateValidationInternal<compilerLearningTypeOrCountTargetClasses, k_cItemsPerBitPackedDataUnitDynamic>::Func(
         pEbmBoostingState,
         pFeatureCombination,
         aModelFeatureCombinationUpdateTensor
      );
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class ApplyModelUpdateValidationSIMDTarget {
public:
   EBM_INLINE static FloatEbmType Func(
      EbmBoostingState * const pEbmBoostingState,
      const FeatureCombination * const pFeatureCombination,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         return ApplyModelUpdateValidationSIMDPacking<
            compilerLearningTypeOrCountTargetClassesPossible,
            k_cItemsPerBitPackedDataUnitMax
         >::Func(
            pEbmBoostingState,
            pFeatureCombination,
            aModelFeatureCombinationUpdateTensor
         );
      } else {
         return ApplyModelUpdateValidationSIMDTarget<
            compilerLearningTypeOrCountTargetClassesPossible + 1
         >::Func(
            pEbmBoostingState,
            pFeatureCombination,
            aModelFeatureCombinationUpdateTensor
         );
      }
   }
};

template<>
class ApplyModelUpdateValidationSIMDTarget<k_cCompilerOptimizedTargetClassesMax + 1> {
public:
   EBM_INLINE static FloatEbmType Func(
      EbmBoostingState * const pEbmBoostingState,
      const FeatureCombination * const pFeatureCombination,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses());

      return ApplyModelUpdateValidationSIMDPacking<
         k_dynamicClassification,
         k_cItemsPerBitPackedDataUnitMax
      >::Func(
         pEbmBoostingState,
         pFeatureCombination,
         aModelFeatureCombinationUpdateTensor
      );
   }
};

extern FloatEbmType ApplyModelUpdateValidation(
   EbmBoostingState * const pEbmBoostingState,
   const FeatureCombination * const pFeatureCombination,
   const FloatEbmType * const aModelFeatureCombinationUpdateTensor
) {
   LOG_0(TraceLevelVerbose, "Entered ApplyModelUpdateValidation");

   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses();

   FloatEbmType ret;
   if(0 == pFeatureCombination->GetCountFeatures()) {
      if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
         ret = ApplyModelUpdateValidationZeroFeaturesTarget<2>::Func(
            pEbmBoostingState,
            aModelFeatureCombinationUpdateTensor
         );
      } else {
         EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
         ret = ApplyModelUpdateValidationZeroFeatures<k_regression>::Func(
            pEbmBoostingState,
            aModelFeatureCombinationUpdateTensor
         );
      }
   } else {
      if(k_bUseSIMD) {
         // TODO : enable SIMD(AVX-512) to work

         // 64 - do 8 at a time and unroll the loop 8 times.  These are bool features and are common.  Put the unrolled inner loop into a function
         // 32 - do 8 at a time and unroll the loop 4 times.  These are bool features and are common.  Put the unrolled inner loop into a function
         // 21 - do 8 at a time and unroll the loop 3 times (ignore the last 3 with a mask)
         // 16 - do 8 at a time and unroll the loop 2 times.  These are bool features and are common.  Put the unrolled inner loop into a function
         // 12 - do 8 of them, shift the low 4 upwards and then load the next 12 and take the top 4, repeat.
         // 10 - just drop this down to packing 8 together
         // 9 - just drop this down to packing 8 together
         // 8 - do all 8 at a time without an inner loop.  This is one of the most common values.  256 binned values
         // 7,6,5,4,3,2,1 - use a mask to exclude the non-used conditions and process them like the 8.  These are rare since they require more than 256 values

         if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
            ret = ApplyModelUpdateValidationSIMDTarget<2>::Func(
               pEbmBoostingState,
               pFeatureCombination,
               aModelFeatureCombinationUpdateTensor
            );
         } else {
            EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
            ret = ApplyModelUpdateValidationSIMDPacking<
               k_regression,
               k_cItemsPerBitPackedDataUnitMax
            >::Func(
               pEbmBoostingState,
               pFeatureCombination,
               aModelFeatureCombinationUpdateTensor
            );
         }
      } else {
         // there isn't much benefit in eliminating the loop that unpacks a data unit unless we're also unpacking that to SIMD code
         // Our default packing structure is to bin continuous values to 256 values, and we have 64 bit packing structures, so we usually
         // have more than 8 values per memory fetch.  Eliminating the inner loop for multiclass is valuable since we can have low numbers like 3 class,
         // 4 class, etc, but by the time we get to 8 loops with exp inside and a lot of other instructures we should worry that our code expansion
         // will exceed the L1 instruction cache size.  With SIMD we do 8 times the work in the same number of instructions so these are lesser issues

         if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
            ret = ApplyModelUpdateValidationNormalTarget<2>::Func(
               pEbmBoostingState,
               pFeatureCombination,
               aModelFeatureCombinationUpdateTensor
            );
         } else {
            EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
            ret = ApplyModelUpdateValidationInternal<k_regression, k_cItemsPerBitPackedDataUnitDynamic>::Func(
               pEbmBoostingState,
               pFeatureCombination,
               aModelFeatureCombinationUpdateTensor
            );
         }
      }
   }

   EBM_ASSERT(std::isnan(ret) || -k_epsilonLogLoss <= ret);
   if(UNLIKELY(UNLIKELY(std::isnan(ret)) || UNLIKELY(std::isinf(ret)))) {
      // set the metric so high that this round of boosting will be rejected.  The worst metric is std::numeric_limits<FloatEbmType>::max(),
      // Set it to that so that this round of boosting won't be accepted if our caller is using early stopping
      ret = std::numeric_limits<FloatEbmType>::max();
   } else {
      if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
         if(UNLIKELY(ret < FloatEbmType { 0 })) {
            // regression can't be negative since squares are pretty well insulated from ever doing that

            // Multiclass can return small negative numbers, so we need to clean up the value retunred so that it isn't negative

            // binary classification can't return a negative number provided the log function
            // doesn't ever return a negative number for numbers exactly equal to 1 or higher
            // BUT we're going to be using or trying approximate log functions, and those might not
            // be guaranteed to return a positive or zero number, so let's just always check for numbers less than zero and round up
            EBM_ASSERT(IsMulticlass(runtimeLearningTypeOrCountTargetClasses));

            // because of floating point inexact reasons, ComputeSingleInstanceLogLossMulticlass can return a negative number
            // so correct this before we return.  Any negative numbers were really meant to be zero
            ret = FloatEbmType { 0 };
         }
      }
   }
   EBM_ASSERT(!std::isnan(ret));
   EBM_ASSERT(!std::isinf(ret));
   EBM_ASSERT(FloatEbmType { 0 } <= ret);

   LOG_0(TraceLevelVerbose, "Exited ApplyModelUpdateValidation");

   return ret;
}
