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
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));
      EBM_ASSERT(!IsBinaryClassification(compilerLearningTypeOrCountTargetClasses));

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      const size_t cInstances = pTrainingSet->GetCountInstances();
      EBM_ASSERT(0 < cInstances);

      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();
      const FloatEbmType * const pResidualErrorEnd = pResidualError + cVectorLength * cInstances;
      FloatEbmType * pTrainingPredictorScores = pTrainingSet->GetPredictorScores();
      const StorageDataType * pTargetData = pTrainingSet->GetTargetDataPointer();
      const FloatEbmType * pValues = aModelFeatureCombinationUpdateTensor;
      do {
         StorageDataType targetData = *pTargetData;
         FloatEbmType sumExp = 0;
         size_t iVector1 = 0;
         do {
            // TODO : because there is only one bin for a zero feature feature combination, we could move these values to the stack where the
            // compiler could reason about their visibility and optimize small arrays into registers
            const FloatEbmType smallChangeToPredictorScores = pValues[iVector1];
            // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
            const FloatEbmType trainingPredictorScores = pTrainingPredictorScores[iVector1] + smallChangeToPredictorScores;
            pTrainingPredictorScores[iVector1] = trainingPredictorScores;
            sumExp += EbmExp(trainingPredictorScores);
            ++iVector1;
         } while(iVector1 < cVectorLength);

         EBM_ASSERT((IsNumberConvertable<StorageDataType, size_t>(cVectorLength)));
         const StorageDataType cVectorLengthStorage = static_cast<StorageDataType>(cVectorLength);
         StorageDataType iVector2 = 0;
         do {
            // TODO : we're calculating exp(predictionScore) above, and then again in ComputeResidualErrorMulticlass.  exp(..) is expensive so we 
            // should just do it once instead and store the result in a small memory array here
            const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorMulticlass(
               sumExp,
               pTrainingPredictorScores[iVector2],
               targetData,
               iVector2
            );
            *pResidualError = residualError;
            ++pResidualError;
            ++iVector2;
         } while(iVector2 < cVectorLengthStorage);
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
         pTrainingPredictorScores += cVectorLength;
         ++pTargetData;
      } while(pResidualErrorEnd != pResidualError);
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<>
class OptimizedApplyModelUpdateTrainingZeroFeatures<2> {
public:
   static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      DataSetByFeatureCombination * const pTrainingSet,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      UNUSED(runtimeLearningTypeOrCountTargetClasses);
      const size_t cInstances = pTrainingSet->GetCountInstances();
      EBM_ASSERT(0 < cInstances);

      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();
      const FloatEbmType * const pResidualErrorEnd = pResidualError + cInstances;
      FloatEbmType * pTrainingPredictorScores = pTrainingSet->GetPredictorScores();
      const StorageDataType * pTargetData = pTrainingSet->GetTargetDataPointer();
      const FloatEbmType smallChangeToPredictorScores = aModelFeatureCombinationUpdateTensor[0];
      do {
         StorageDataType targetData = *pTargetData;
         // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
         const FloatEbmType trainingPredictorScore = *pTrainingPredictorScores + smallChangeToPredictorScores;
         *pTrainingPredictorScores = trainingPredictorScore;
         const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorBinaryClassification(trainingPredictorScore, targetData);
         *pResidualError = residualError;
         ++pResidualError;
         ++pTrainingPredictorScores;
         ++pTargetData;
      } while(pResidualErrorEnd != pResidualError);
   }
};
#endif // EXPAND_BINARY_LOGITS

template<>
class OptimizedApplyModelUpdateTrainingZeroFeatures<k_Regression> {
public:
   static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      DataSetByFeatureCombination * const pTrainingSet,
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      UNUSED(runtimeLearningTypeOrCountTargetClasses);
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
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));
      EBM_ASSERT(!IsBinaryClassification(compilerLearningTypeOrCountTargetClasses));

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

      const StorageDataType * pInputData = pTrainingSet->GetInputDataPointer(pFeatureCombination);
      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();

      FloatEbmType * pTrainingPredictorScores = pTrainingSet->GetPredictorScores();
      const StorageDataType * pTargetData = pTrainingSet->GetTargetDataPointer();

      // this shouldn't overflow since we're accessing existing memory
      const FloatEbmType * const pResidualErrorTrueEnd = pResidualError + cVectorLength * cInstances;
      const FloatEbmType * pResidualErrorExit = pResidualErrorTrueEnd;
      size_t cItemsRemaining = cInstances;
      if(cInstances <= cItemsPerBitPackedDataUnit) {
         goto one_last_loop_classification;
      }
      pResidualErrorExit = pResidualErrorTrueEnd - cVectorLength * ((cInstances - 1) % cItemsPerBitPackedDataUnit + 1);
      EBM_ASSERT(pResidualError < pResidualErrorExit);
      EBM_ASSERT(pResidualErrorExit < pResidualErrorTrueEnd);

      do {
         cItemsRemaining = cItemsPerBitPackedDataUnit;
         // TODO : jumping back into this loop and changing cItemsRemaining to a dynamic value that isn't compile time determinable
         // causes this function to NOT be optimized as much as it could if we had two separate loops.  We're just trying this out for now though
      one_last_loop_classification:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iTensorBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            StorageDataType targetData = *pTargetData;

            const size_t iTensorBin = maskBits & iTensorBinCombined;
            const FloatEbmType * pValues = &aModelFeatureCombinationUpdateTensor[iTensorBin * cVectorLength];

            FloatEbmType sumExp = 0;
            size_t iVector1 = 0;
            do {
               const FloatEbmType smallChangeToPredictorScores = pValues[iVector1];
               // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
               const FloatEbmType trainingPredictorScores = pTrainingPredictorScores[iVector1] + smallChangeToPredictorScores;
               pTrainingPredictorScores[iVector1] = trainingPredictorScores;
               sumExp += EbmExp(trainingPredictorScores);
               ++iVector1;
            } while(iVector1 < cVectorLength);

            EBM_ASSERT((IsNumberConvertable<StorageDataType, size_t>(cVectorLength)));
            const StorageDataType cVectorLengthStorage = static_cast<StorageDataType>(cVectorLength);
            StorageDataType iVector2 = 0;
            do {
               // TODO : we're calculating exp(predictionScore) above, and then again in ComputeResidualErrorMulticlass.  exp(..) is expensive so we 
               // should just do it once instead and store the result in a small memory array here
               const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorMulticlass(
                  sumExp,
                  pTrainingPredictorScores[iVector2],
                  targetData,
                  iVector2
               );
               *pResidualError = residualError;
               ++pResidualError;
               ++iVector2;
            } while(iVector2 < cVectorLengthStorage);
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
            pTrainingPredictorScores += cVectorLength;
            ++pTargetData;

            iTensorBinCombined >>= cBitsPerItemMax;
            // TODO : try replacing cItemsRemaining with a pResidualErrorInnerLoopEnd which eliminates one subtact operation, but might make it harder 
            // for the compiler to optimize the loop away
            --cItemsRemaining;
         } while(0 != cItemsRemaining);
      } while(pResidualErrorExit != pResidualError);

      // first time through?
      if(pResidualErrorTrueEnd != pResidualError) {
         EBM_ASSERT(0 == (pResidualErrorTrueEnd - pResidualError) % cVectorLength);
         cItemsRemaining = (pResidualErrorTrueEnd - pResidualError) / cVectorLength;
         EBM_ASSERT(0 < cItemsRemaining);
         EBM_ASSERT(cItemsRemaining <= cItemsPerBitPackedDataUnit);

         pResidualErrorExit = pResidualErrorTrueEnd;

         goto one_last_loop_classification;
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
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      UNUSED(runtimeLearningTypeOrCountTargetClasses);
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

      const StorageDataType * pInputData = pTrainingSet->GetInputDataPointer(pFeatureCombination);
      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();

      FloatEbmType * pTrainingPredictorScores = pTrainingSet->GetPredictorScores();
      const StorageDataType * pTargetData = pTrainingSet->GetTargetDataPointer();

      // this shouldn't overflow since we're accessing existing memory
      const FloatEbmType * const pResidualErrorTrueEnd = pResidualError + cInstances;
      const FloatEbmType * pResidualErrorExit = pResidualErrorTrueEnd;
      size_t cItemsRemaining = cInstances;
      if(cInstances <= cItemsPerBitPackedDataUnit) {
         goto one_last_loop_classification;
      }
      pResidualErrorExit = pResidualErrorTrueEnd - ((cInstances - 1) % cItemsPerBitPackedDataUnit + 1);
      EBM_ASSERT(pResidualError < pResidualErrorExit);
      EBM_ASSERT(pResidualErrorExit < pResidualErrorTrueEnd);

      do {
         cItemsRemaining = cItemsPerBitPackedDataUnit;
         // TODO : jumping back into this loop and changing cItemsRemaining to a dynamic value that isn't compile time determinable
         // causes this function to NOT be optimized as much as it could if we had two separate loops.  We're just trying this out for now though
      one_last_loop_classification:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iTensorBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            StorageDataType targetData = *pTargetData;

            const size_t iTensorBin = maskBits & iTensorBinCombined;
            const FloatEbmType * pValues = &aModelFeatureCombinationUpdateTensor[iTensorBin];

            const FloatEbmType smallChangeToPredictorScores = pValues[0];
            // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
            const FloatEbmType trainingPredictorScore = *pTrainingPredictorScores + smallChangeToPredictorScores;
            *pTrainingPredictorScores = trainingPredictorScore;
            const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorBinaryClassification(trainingPredictorScore, targetData);
            *pResidualError = residualError;
            ++pResidualError;
            pTrainingPredictorScores += 1;
            ++pTargetData;

            iTensorBinCombined >>= cBitsPerItemMax;
            // TODO : try replacing cItemsRemaining with a pResidualErrorInnerLoopEnd which eliminates one subtact operation, but might make it harder 
            // for the compiler to optimize the loop away
            --cItemsRemaining;
         } while(0 != cItemsRemaining);
      } while(pResidualErrorExit != pResidualError);

      // first time through?
      if(pResidualErrorTrueEnd != pResidualError) {
         cItemsRemaining = pResidualErrorTrueEnd - pResidualError;
         EBM_ASSERT(0 < cItemsRemaining);
         EBM_ASSERT(cItemsRemaining <= cItemsPerBitPackedDataUnit);

         pResidualErrorExit = pResidualErrorTrueEnd;

         goto one_last_loop_classification;
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
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      UNUSED(runtimeLearningTypeOrCountTargetClasses);
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

      const StorageDataType * pInputData = pTrainingSet->GetInputDataPointer(pFeatureCombination);
      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();


      // this shouldn't overflow since we're accessing existing memory
      const FloatEbmType * const pResidualErrorTrueEnd = pResidualError + cInstances;
      const FloatEbmType * pResidualErrorExit = pResidualErrorTrueEnd;
      size_t cItemsRemaining = cInstances;
      if(cInstances <= cItemsPerBitPackedDataUnit) {
         goto one_last_loop_regression;
      }
      pResidualErrorExit = pResidualErrorTrueEnd - ((cInstances - 1) % cItemsPerBitPackedDataUnit + 1);
      EBM_ASSERT(pResidualError < pResidualErrorExit);
      EBM_ASSERT(pResidualErrorExit < pResidualErrorTrueEnd);

      do {
         cItemsRemaining = cItemsPerBitPackedDataUnit;
         // TODO : jumping back into this loop and changing cItemsRemaining to a dynamic value that isn't compile time determinable
         // causes this function to NOT be optimized as much as it could if we had two separate loops.  We're just trying this out for now though
      one_last_loop_regression:;
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
            // TODO : try replacing cItemsRemaining with a pResidualErrorInnerLoopEnd which eliminates one subtact operation, but might make it 
            // harder for the compiler to optimize the loop away
            --cItemsRemaining;
         } while(0 != cItemsRemaining);
      } while(pResidualErrorExit != pResidualError);

      // first time through?
      if(pResidualErrorTrueEnd != pResidualError) {
         cItemsRemaining = pResidualErrorTrueEnd - pResidualError;
         EBM_ASSERT(0 < cItemsRemaining);
         EBM_ASSERT(cItemsRemaining <= cItemsPerBitPackedDataUnit);

         pResidualErrorExit = pResidualErrorTrueEnd;

         goto one_last_loop_regression;
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
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
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
            aModelFeatureCombinationUpdateTensor
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
            aModelFeatureCombinationUpdateTensor
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
      const FloatEbmType * const aModelFeatureCombinationUpdateTensor
   ) {
      EBM_ASSERT(1 <= runtimeCountItemsPerBitPackedDataUnit);
      EBM_ASSERT(runtimeCountItemsPerBitPackedDataUnit <= k_cBitsForStorageType);
      OptimizedApplyModelUpdateTrainingInternal<compilerLearningTypeOrCountTargetClasses, k_cItemsPerBitPackedDataUnitDynamic>::Func(
         runtimeLearningTypeOrCountTargetClasses,
         runtimeCountItemsPerBitPackedDataUnit,
         pFeatureCombination,
         pTrainingSet,
         aModelFeatureCombinationUpdateTensor
      );
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
EBM_INLINE static void OptimizedApplyModelUpdateTraining(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const bool bUseSIMD,
   const FeatureCombination * const pFeatureCombination,
   DataSetByFeatureCombination * const pTrainingSet,
   const FloatEbmType * const aModelFeatureCombinationUpdateTensor
) {
   LOG_0(TraceLevelVerbose, "Entered OptimizedApplyModelUpdateTraining");

   if(0 == pFeatureCombination->m_cFeatures) {
      OptimizedApplyModelUpdateTrainingZeroFeatures<compilerLearningTypeOrCountTargetClasses>::Func(
         runtimeLearningTypeOrCountTargetClasses,
         pTrainingSet,
         aModelFeatureCombinationUpdateTensor
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

         //EBM_ASSERT(false); // not implemented yet
         //OptimizedApplyModelUpdateTrainingCompiler<
         //   compilerLearningTypeOrCountTargetClasses,
         //   k_cItemsPerBitPackedDataUnitMax
         //>::MagicCompilerLoopFunction(
         //   runtimeLearningTypeOrCountTargetClasses,
         //   pFeatureCombination->m_cItemsPerBitPackedDataUnit,
         //   pFeatureCombination,
         //   pTrainingSet,
         //   aModelFeatureCombinationUpdateTensor
         //);

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
            aModelFeatureCombinationUpdateTensor
         );
      }
   }

   LOG_0(TraceLevelVerbose, "Exited OptimizedApplyModelUpdateTraining");
}

#endif // OPTIMIZED_APPLY_MODEL_UPDATE_TRAINING_H
