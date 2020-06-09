// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef OPTIMIZED_APPLY_MODEL_UPDATE_TRAINING_H
#define OPTIMIZED_APPLY_MODEL_UPDATE_TRAINING_H

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "EbmInternal.h"
// very independent includes
#include "Logging.h" // EBM_ASSERT & LOG
#include "EbmStatistics.h"
// FeatureCombination.h depends on FeatureInternal.h
#include "FeatureCombination.h"
// dataset depends on features
#include "DataSetByFeatureCombination.h"

// C++ does not allow partial function specialization, so we need to use these cumbersome static class functions to do partial function specialization

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class OptimizedApplyModelUpdateTrainingZeroFeatures {
public:
   static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      DataSetByFeatureCombination * const pTrainingSet,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor,
      FloatEbmType * const aTempFloatVector
   ) {
      EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));
      EBM_ASSERT(!IsBinaryClassification(compilerLearningTypeOrCountTargetClasses));

      FloatEbmType aLocalExpVector[
         k_DynamicClassification == compilerLearningTypeOrCountTargetClasses ? 1 : GetVectorLength(compilerLearningTypeOrCountTargetClasses)
      ];
      FloatEbmType * const aExpVector = k_DynamicClassification == compilerLearningTypeOrCountTargetClasses ? aTempFloatVector : aLocalExpVector;

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      const size_t cInstances = pTrainingSet->GetCountInstances();
      EBM_ASSERT(0 < cInstances);

      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();
      const StorageDataType * pTargetData = pTrainingSet->GetTargetDataPointer();
      FloatEbmType * pPredictorScores = pTrainingSet->GetPredictorScores();
      const FloatEbmType * const pPredictorScoresEnd = pPredictorScores + cInstances * cVectorLength;
      do {
         size_t targetData = static_cast<size_t>(*pTargetData);
         ++pTargetData;

         const FloatEbmType * pValues = aModelFeatureCombinationUpdateTensor;

         FloatEbmType * pExpVector = aExpVector;

         FloatEbmType sumExp = FloatEbmType { 0 };
         size_t iVector = 0;
         do {
            // TODO : because there is only one bin for a zero feature feature combination, we could move these values to the stack where the
            // compiler could reason about their visibility and optimize small arrays into registers
            const FloatEbmType smallChangeToPredictorScores = *pValues;
            ++pValues;
            // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
            const FloatEbmType predictorScore = *pPredictorScores + smallChangeToPredictorScores;
            *pPredictorScores = predictorScore;
            ++pPredictorScores;
            const FloatEbmType oneExp = EbmExp(predictorScore);
            *pExpVector = oneExp;
            ++pExpVector;
            sumExp += oneExp;
            ++iVector;
         } while(iVector < cVectorLength);
         // TODO: store the result of std::exp above for the index that we care about above since exp(..) is going to be expensive and probably 
         // even more expensive than an unconditional branch
         pExpVector -= cVectorLength;
         iVector = 0;
         do {
            const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorMulticlass(
               sumExp,
               *pExpVector,
               targetData,
               iVector
            );
            ++pExpVector;
            *pResidualError = residualError;
            ++pResidualError;
            ++iVector;
         } while(iVector < cVectorLength);
         // TODO: this works as a way to remove one parameter, but it obviously insn't as efficient as omitting the parameter
         // 
         // this works out in the math as making the first model vector parameter equal to zero, which in turn removes one degree of freedom
         // from the model vector parameters.  Since the model vector weights need to be normalized to sum to a probabilty of 100%, we can set the first
         // one to the constant 1 (0 in log space) and force the other parameters to adjust to that scale which fixes them to a single valid set of 
         // values insted of allowing them to be scaled.  
         // Probability = exp(T1 + I1) / [exp(T1 + I1) + exp(T2 + I2) + exp(T3 + I3)] => we can add a constant inside each exp(..) term, which 
         // will be multiplication outside the exp(..), which means the numerator and denominator are multiplied by the same constant, which cancels 
         // eachother out.  We can thus set exp(T2 + I2) to exp(0) and adjust the other terms
         constexpr bool bZeroingResiduals = 0 <= k_iZeroResidual;
         if(bZeroingResiduals) {
            *(pResidualError - (cVectorLength - static_cast<size_t>(k_iZeroResidual))) = 0;
         }
      } while(pPredictorScoresEnd != pPredictorScores);
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<>
class OptimizedApplyModelUpdateTrainingZeroFeatures<2> {
public:
   static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      DataSetByFeatureCombination * const pTrainingSet,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor,
      FloatEbmType * const aTempFloatVector
   ) {
      UNUSED(runtimeLearningTypeOrCountTargetClasses);
      UNUSED(aTempFloatVector);
      const size_t cInstances = pTrainingSet->GetCountInstances();
      EBM_ASSERT(0 < cInstances);

      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();
      const StorageDataType * pTargetData = pTrainingSet->GetTargetDataPointer();
      FloatEbmType * pPredictorScores = pTrainingSet->GetPredictorScores();
      const FloatEbmType * const pPredictorScoresEnd = pPredictorScores + cInstances;
      const FloatEbmType smallChangeToPredictorScores = aModelFeatureCombinationUpdateTensor[0];
      do {
         size_t targetData = static_cast<size_t>(*pTargetData);
         ++pTargetData;
         // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
         const FloatEbmType predictorScore = *pPredictorScores + smallChangeToPredictorScores;
         *pPredictorScores = predictorScore;
         ++pPredictorScores;
         const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorBinaryClassification(predictorScore, targetData);
         *pResidualError = residualError;
         ++pResidualError;
      } while(pPredictorScoresEnd != pPredictorScores);
   }
};
#endif // EXPAND_BINARY_LOGITS

template<>
class OptimizedApplyModelUpdateTrainingZeroFeatures<k_Regression> {
public:
   static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      DataSetByFeatureCombination * const pTrainingSet,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor,
      FloatEbmType * const aTempFloatVector
   ) {
      UNUSED(runtimeLearningTypeOrCountTargetClasses);
      UNUSED(aTempFloatVector);
      const size_t cInstances = pTrainingSet->GetCountInstances();
      EBM_ASSERT(0 < cInstances);


      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();
      const FloatEbmType * const pResidualErrorEnd = pResidualError + cInstances;
      const FloatEbmType smallChangeToPrediction = aModelFeatureCombinationUpdateTensor[0];
      do {
         // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
         const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorRegression(*pResidualError - smallChangeToPrediction);
         *pResidualError = residualError;
         ++pResidualError;
      } while(pResidualErrorEnd != pResidualError);
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerCountItemsPerBitPackedDataUnit>
class OptimizedApplyModelUpdateTrainingInternal {
public:
   static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t runtimeCountItemsPerBitPackedDataUnit,
      const FeatureCombination * const pFeatureCombination,
      DataSetByFeatureCombination * const pTrainingSet,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor,
      FloatEbmType * const aTempFloatVector
   ) {
      EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));
      EBM_ASSERT(!IsBinaryClassification(compilerLearningTypeOrCountTargetClasses));

      FloatEbmType aLocalExpVector[
         k_DynamicClassification == compilerLearningTypeOrCountTargetClasses ? 1 : GetVectorLength(compilerLearningTypeOrCountTargetClasses)
      ];
      FloatEbmType * const aExpVector = k_DynamicClassification == compilerLearningTypeOrCountTargetClasses ? aTempFloatVector : aLocalExpVector;

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      const size_t cInstances = pTrainingSet->GetCountInstances();
      EBM_ASSERT(0 < cInstances);
      EBM_ASSERT(0 < pFeatureCombination->m_cFeatures);

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

      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();
      const StorageDataType * pInputData = pTrainingSet->GetInputDataPointer(pFeatureCombination);
      const StorageDataType * pTargetData = pTrainingSet->GetTargetDataPointer();
      FloatEbmType * pPredictorScores = pTrainingSet->GetPredictorScores();

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

            FloatEbmType * pExpVector = aExpVector;

            FloatEbmType sumExp = FloatEbmType { 0 };
            size_t iVector = 0;
            do {
               const FloatEbmType smallChangeToPredictorScores = *pValues;
               ++pValues;
               // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
               const FloatEbmType predictorScore = *pPredictorScores + smallChangeToPredictorScores;
               *pPredictorScores = predictorScore;
               ++pPredictorScores;
               const FloatEbmType oneExp = EbmExp(predictorScore);
               *pExpVector = oneExp;
               ++pExpVector;
               sumExp += oneExp;
               ++iVector;
            } while(iVector < cVectorLength);
            // TODO: store the result of std::exp above for the index that we care about above since exp(..) is going to be expensive and 
            // probably even more expensive than an unconditional branch
            pExpVector -= cVectorLength;
            iVector = 0;
            do {
               const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorMulticlass(
                  sumExp,
                  *pExpVector,
                  targetData,
                  iVector
               );
               ++pExpVector;
               *pResidualError = residualError;
               ++pResidualError;
               ++iVector;
            } while(iVector < cVectorLength);
            // TODO: this works as a way to remove one parameter, but it obviously insn't as efficient as omitting the parameter
            // 
            // this works out in the math as making the first model vector parameter equal to zero, which in turn removes one degree of freedom
            // from the model vector parameters.  Since the model vector weights need to be normalized to sum to a probabilty of 100%, we can set the 
            // first one to the constant 1 (0 in log space) and force the other parameters to adjust to that scale which fixes them to a single valid 
            // set of values insted of allowing them to be scaled.  
            // Probability = exp(T1 + I1) / [exp(T1 + I1) + exp(T2 + I2) + exp(T3 + I3)] => we can add a constant inside each exp(..) term, which 
            // will be multiplication outside the exp(..), which means the numerator and denominator are multiplied by the same constant, which 
            // cancels eachother out.  We can thus set exp(T2 + I2) to exp(0) and adjust the other terms
            constexpr bool bZeroingResiduals = 0 <= k_iZeroResidual;
            if(bZeroingResiduals) {
               *(pResidualError - (cVectorLength - static_cast<size_t>(k_iZeroResidual))) = 0;
            }

            iTensorBinCombined >>= cBitsPerItemMax;
         } while(pPredictorScoresInnerEnd != pPredictorScores);
      } while(pPredictorScoresExit != pPredictorScores);

      // first time through?
      if(pPredictorScoresTrueEnd != pPredictorScores) {
         pPredictorScoresInnerEnd = pPredictorScoresTrueEnd;
         pPredictorScoresExit = pPredictorScoresTrueEnd;
         goto one_last_loop;
      }
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<size_t compilerCountItemsPerBitPackedDataUnit>
class OptimizedApplyModelUpdateTrainingInternal<2, compilerCountItemsPerBitPackedDataUnit> {
public:
   static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t runtimeCountItemsPerBitPackedDataUnit,
      const FeatureCombination * const pFeatureCombination,
      DataSetByFeatureCombination * const pTrainingSet,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor,
      FloatEbmType * const aTempFloatVector
   ) {
      UNUSED(runtimeLearningTypeOrCountTargetClasses);
      UNUSED(aTempFloatVector);
      const size_t cInstances = pTrainingSet->GetCountInstances();
      EBM_ASSERT(0 < cInstances);
      EBM_ASSERT(0 < pFeatureCombination->m_cFeatures);

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

      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();
      const StorageDataType * pInputData = pTrainingSet->GetInputDataPointer(pFeatureCombination);
      const StorageDataType * pTargetData = pTrainingSet->GetTargetDataPointer();
      FloatEbmType * pPredictorScores = pTrainingSet->GetPredictorScores();

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
            // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
            const FloatEbmType predictorScore = *pPredictorScores + smallChangeToPredictorScores;
            *pPredictorScores = predictorScore;
            ++pPredictorScores;
            const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorBinaryClassification(predictorScore, targetData);

            *pResidualError = residualError;
            ++pResidualError;

            iTensorBinCombined >>= cBitsPerItemMax;
         } while(pPredictorScoresInnerEnd != pPredictorScores);
      } while(pPredictorScoresExit != pPredictorScores);

      // first time through?
      if(pPredictorScoresTrueEnd != pPredictorScores) {
         pPredictorScoresInnerEnd = pPredictorScoresTrueEnd;
         pPredictorScoresExit = pPredictorScoresTrueEnd;
         goto one_last_loop;
      }
   }
};
#endif // EXPAND_BINARY_LOGITS

template<size_t compilerCountItemsPerBitPackedDataUnit>
class OptimizedApplyModelUpdateTrainingInternal<k_Regression, compilerCountItemsPerBitPackedDataUnit> {
public:
   static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t runtimeCountItemsPerBitPackedDataUnit,
      const FeatureCombination * const pFeatureCombination,
      DataSetByFeatureCombination * const pTrainingSet,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor,
      FloatEbmType * const aTempFloatVector
   ) {
      UNUSED(runtimeLearningTypeOrCountTargetClasses);
      UNUSED(aTempFloatVector);
      const size_t cInstances = pTrainingSet->GetCountInstances();
      EBM_ASSERT(0 < cInstances);
      EBM_ASSERT(0 < pFeatureCombination->m_cFeatures);

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


      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();
      const StorageDataType * pInputData = pTrainingSet->GetInputDataPointer(pFeatureCombination);


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
            // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
            const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorRegression(*pResidualError - smallChangeToPrediction);



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
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerCountItemsPerBitPackedDataUnitPossible>
class OptimizedApplyModelUpdateTrainingCompiler {
public:
   EBM_INLINE static void MagicCompilerLoopFunction(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t runtimeCountItemsPerBitPackedDataUnit,
      const FeatureCombination * const pFeatureCombination,
      DataSetByFeatureCombination * const pTrainingSet,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor,
      FloatEbmType * const aTempFloatVector
   ) {
      EBM_ASSERT(1 <= runtimeCountItemsPerBitPackedDataUnit);
      EBM_ASSERT(runtimeCountItemsPerBitPackedDataUnit <= k_cBitsForStorageType);
      static_assert(compilerCountItemsPerBitPackedDataUnitPossible <= k_cBitsForStorageType, "We can't have this many items in a data pack.");
      if(compilerCountItemsPerBitPackedDataUnitPossible == runtimeCountItemsPerBitPackedDataUnit) {
         OptimizedApplyModelUpdateTrainingInternal<compilerLearningTypeOrCountTargetClasses, compilerCountItemsPerBitPackedDataUnitPossible>::Func(
            runtimeLearningTypeOrCountTargetClasses,
            runtimeCountItemsPerBitPackedDataUnit,
            pFeatureCombination,
            pTrainingSet,
            aModelFeatureCombinationUpdateTensor,
            aTempFloatVector
         );
      } else {
         OptimizedApplyModelUpdateTrainingCompiler<
            compilerLearningTypeOrCountTargetClasses,
            GetNextCountItemsBitPacked(compilerCountItemsPerBitPackedDataUnitPossible)
         >::MagicCompilerLoopFunction(
            runtimeLearningTypeOrCountTargetClasses,
            runtimeCountItemsPerBitPackedDataUnit,
            pFeatureCombination,
            pTrainingSet,
            aModelFeatureCombinationUpdateTensor,
            aTempFloatVector
         );
      }
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class OptimizedApplyModelUpdateTrainingCompiler<compilerLearningTypeOrCountTargetClasses, k_cItemsPerBitPackedDataUnitDynamic> {
public:
   EBM_INLINE static void MagicCompilerLoopFunction(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t runtimeCountItemsPerBitPackedDataUnit,
      const FeatureCombination * const pFeatureCombination,
      DataSetByFeatureCombination * const pTrainingSet,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor,
      FloatEbmType * const aTempFloatVector
   ) {
      EBM_ASSERT(1 <= runtimeCountItemsPerBitPackedDataUnit);
      EBM_ASSERT(runtimeCountItemsPerBitPackedDataUnit <= k_cBitsForStorageType);
      OptimizedApplyModelUpdateTrainingInternal<compilerLearningTypeOrCountTargetClasses, k_cItemsPerBitPackedDataUnitDynamic>::Func(
         runtimeLearningTypeOrCountTargetClasses,
         runtimeCountItemsPerBitPackedDataUnit,
         pFeatureCombination,
         pTrainingSet,
         aModelFeatureCombinationUpdateTensor,
         aTempFloatVector
      );
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
EBM_INLINE static void OptimizedApplyModelUpdateTraining(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const bool bUseSIMD,
   const FeatureCombination * const pFeatureCombination,
   DataSetByFeatureCombination * const pTrainingSet,
   const FloatEbmType * const aModelFeatureCombinationUpdateTensor,
   FloatEbmType * const aTempFloatVector
) {
   LOG_0(TraceLevelVerbose, "Entered OptimizedApplyModelUpdateTraining");

   if(0 == pFeatureCombination->m_cFeatures) {
      OptimizedApplyModelUpdateTrainingZeroFeatures<compilerLearningTypeOrCountTargetClasses>::Func(
         runtimeLearningTypeOrCountTargetClasses,
         pTrainingSet,
         aModelFeatureCombinationUpdateTensor,
         aTempFloatVector
      );
   } else {
      if(bUseSIMD) {
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

         OptimizedApplyModelUpdateTrainingCompiler<
            compilerLearningTypeOrCountTargetClasses,
            k_cItemsPerBitPackedDataUnitMax
         >::MagicCompilerLoopFunction(
            runtimeLearningTypeOrCountTargetClasses,
            pFeatureCombination->m_cItemsPerBitPackedDataUnit,
            pFeatureCombination,
            pTrainingSet,
            aModelFeatureCombinationUpdateTensor,
            aTempFloatVector
         );
      } else {
         // there isn't much benefit in eliminating the loop that unpacks a data unit unless we're also unpacking that to SIMD code
         // Our default packing structure is to bin continuous values to 256 values, and we have 64 bit packing structures, so we usually
         // have more than 8 values per memory fetch.  Eliminating the inner loop for multiclass is valuable since we can have low numbers like 3 class,
         // 4 class, etc, but by the time we get to 8 loops with exp inside and a lot of other instructures we should worry that our code expansion
         // will exceed the L1 instruction cache size.  With SIMD we do 8 times the work in the same number of instructions so these are lesser issues
         OptimizedApplyModelUpdateTrainingInternal<compilerLearningTypeOrCountTargetClasses, k_cItemsPerBitPackedDataUnitDynamic>::Func(
            runtimeLearningTypeOrCountTargetClasses,
            pFeatureCombination->m_cItemsPerBitPackedDataUnit,
            pFeatureCombination,
            pTrainingSet,
            aModelFeatureCombinationUpdateTensor,
            aTempFloatVector
         );
      }
   }

   LOG_0(TraceLevelVerbose, "Exited OptimizedApplyModelUpdateTraining");
}

#endif // OPTIMIZED_APPLY_MODEL_UPDATE_TRAINING_H
