// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "EbmInternal.h"
// very independent includes
#include "Logging.h" // EBM_ASSERT & LOG
#include "ApproximateMath.h"
#include "EbmStats.h"
// FeatureGroup.h depends on FeatureInternal.h
#include "FeatureGroup.h"
// dataset depends on features
#include "DataFrameBoosting.h"

#include "Booster.h"
#include "ThreadStateBoosting.h"

// C++ does not allow partial function specialization, so we need to use these cumbersome static class functions to do partial function specialization

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class ApplyModelUpdateTrainingZeroFeatures final {
public:

   ApplyModelUpdateTrainingZeroFeatures() = delete; // this is a static class.  Do not construct

   static void Func(ThreadStateBoosting * const pThreadStateBoosting) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClasses), "must be classification");
      static_assert(!IsBinaryClassification(compilerLearningTypeOrCountTargetClasses), "must be multiclass");

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
      DataFrameBoosting * const pTrainingSet = pBooster->GetTrainingSet();
      FloatEbmType * const aTempFloatVector = pThreadStateBoosting->GetTempFloatVector();

      FloatEbmType aLocalExpVector[
         k_dynamicClassification == compilerLearningTypeOrCountTargetClasses ? 1 : GetVectorLength(compilerLearningTypeOrCountTargetClasses)
      ];
      FloatEbmType * const aExpVector = k_dynamicClassification == compilerLearningTypeOrCountTargetClasses ? aTempFloatVector : aLocalExpVector;

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      const size_t cSamples = pTrainingSet->GetCountSamples();
      EBM_ASSERT(1 <= cSamples);

      const FloatEbmType * const aModelFeatureGroupUpdateTensor = pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->GetValuePointer();
      EBM_ASSERT(nullptr != aModelFeatureGroupUpdateTensor);

      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();
      const StorageDataType * pTargetData = pTrainingSet->GetTargetDataPointer();
      FloatEbmType * pPredictorScores = pTrainingSet->GetPredictorScores();
      const FloatEbmType * const pPredictorScoresEnd = pPredictorScores + cSamples * cVectorLength;
      do {
         size_t targetData = static_cast<size_t>(*pTargetData);
         ++pTargetData;

         const FloatEbmType * pValues = aModelFeatureGroupUpdateTensor;
         FloatEbmType * pExpVector = aExpVector;
         FloatEbmType sumExp = FloatEbmType { 0 };
         size_t iVector = 0;
         do {
            // TODO : because there is only one bin for a zero feature feature group, we could move these values to the stack where the
            // compiler could reason about their visibility and optimize small arrays into registers
            const FloatEbmType smallChangeToPredictorScores = *pValues;
            ++pValues;
            // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
            const FloatEbmType predictorScore = *pPredictorScores + smallChangeToPredictorScores;
            *pPredictorScores = predictorScore;
            ++pPredictorScores;
            const FloatEbmType oneExp = ExpForResidualsMulticlass(predictorScore);
            *pExpVector = oneExp;
            ++pExpVector;
            sumExp += oneExp;
            ++iVector;
         } while(iVector < cVectorLength);
         pExpVector -= cVectorLength;
         iVector = 0;
         do {
            const FloatEbmType residualError = EbmStats::ComputeResidualErrorMulticlass(
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
            *(pResidualError - (static_cast<ptrdiff_t>(cVectorLength) - k_iZeroResidual)) = 0;
         }
      } while(pPredictorScoresEnd != pPredictorScores);
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<>
class ApplyModelUpdateTrainingZeroFeatures<2> final {
public:

   ApplyModelUpdateTrainingZeroFeatures() = delete; // this is a static class.  Do not construct

   static void Func(ThreadStateBoosting * const pThreadStateBoosting) {
      UNUSED(pThreadStateBoosting);

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      DataFrameBoosting * const pTrainingSet = pBooster->GetTrainingSet();
      const size_t cSamples = pTrainingSet->GetCountSamples();
      EBM_ASSERT(1 <= cSamples);

      const FloatEbmType * const aModelFeatureGroupUpdateTensor = pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->GetValuePointer();
      EBM_ASSERT(nullptr != aModelFeatureGroupUpdateTensor);

      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();
      const StorageDataType * pTargetData = pTrainingSet->GetTargetDataPointer();
      FloatEbmType * pPredictorScores = pTrainingSet->GetPredictorScores();
      const FloatEbmType * const pPredictorScoresEnd = pPredictorScores + cSamples;
      const FloatEbmType smallChangeToPredictorScores = aModelFeatureGroupUpdateTensor[0];
      do {
         size_t targetData = static_cast<size_t>(*pTargetData);
         ++pTargetData;
         // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
         const FloatEbmType predictorScore = *pPredictorScores + smallChangeToPredictorScores;
         *pPredictorScores = predictorScore;
         ++pPredictorScores;
         const FloatEbmType residualError = EbmStats::ComputeResidualErrorBinaryClassification(predictorScore, targetData);
         *pResidualError = residualError;
         ++pResidualError;
      } while(pPredictorScoresEnd != pPredictorScores);
   }
};
#endif // EXPAND_BINARY_LOGITS

template<>
class ApplyModelUpdateTrainingZeroFeatures<k_regression> final {
public:

   ApplyModelUpdateTrainingZeroFeatures() = delete; // this is a static class.  Do not construct

   static void Func(ThreadStateBoosting * const pThreadStateBoosting) {
      UNUSED(pThreadStateBoosting);

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      DataFrameBoosting * const pTrainingSet = pBooster->GetTrainingSet();
      const size_t cSamples = pTrainingSet->GetCountSamples();
      EBM_ASSERT(1 <= cSamples);

      const FloatEbmType * const aModelFeatureGroupUpdateTensor = pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->GetValuePointer();
      EBM_ASSERT(nullptr != aModelFeatureGroupUpdateTensor);

      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();
      const FloatEbmType * const pResidualErrorEnd = pResidualError + cSamples;
      const FloatEbmType smallChangeToPrediction = aModelFeatureGroupUpdateTensor[0];
      do {
         // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
         const FloatEbmType residualError = EbmStats::ComputeResidualErrorRegression(*pResidualError - smallChangeToPrediction);
         *pResidualError = residualError;
         ++pResidualError;
      } while(pResidualErrorEnd != pResidualError);
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class ApplyModelUpdateTrainingZeroFeaturesTarget final {
public:

   ApplyModelUpdateTrainingZeroFeaturesTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(ThreadStateBoosting * const pThreadStateBoosting) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         ApplyModelUpdateTrainingZeroFeatures<compilerLearningTypeOrCountTargetClassesPossible>::Func(
            pThreadStateBoosting
         );
      } else {
         ApplyModelUpdateTrainingZeroFeaturesTarget<
            compilerLearningTypeOrCountTargetClassesPossible + 1
         >::Func(
            pThreadStateBoosting
         );
      }
   }
};

template<>
class ApplyModelUpdateTrainingZeroFeaturesTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   ApplyModelUpdateTrainingZeroFeaturesTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(ThreadStateBoosting * const pThreadStateBoosting) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses());

      ApplyModelUpdateTrainingZeroFeatures<k_dynamicClassification>::Func(pThreadStateBoosting);
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerCountItemsPerBitPackedDataUnit>
class ApplyModelUpdateTrainingInternal final {
public:

   ApplyModelUpdateTrainingInternal() = delete; // this is a static class.  Do not construct

   static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClasses), "must be classification");
      static_assert(!IsBinaryClassification(compilerLearningTypeOrCountTargetClasses), "must be multiclass");

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
      DataFrameBoosting * const pTrainingSet = pBooster->GetTrainingSet();
      FloatEbmType * const aTempFloatVector = pThreadStateBoosting->GetTempFloatVector();

      FloatEbmType aLocalExpVector[
         k_dynamicClassification == compilerLearningTypeOrCountTargetClasses ? 1 : GetVectorLength(compilerLearningTypeOrCountTargetClasses)
      ];
      FloatEbmType * const aExpVector = k_dynamicClassification == compilerLearningTypeOrCountTargetClasses ? aTempFloatVector : aLocalExpVector;

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      const size_t cSamples = pTrainingSet->GetCountSamples();
      EBM_ASSERT(1 <= cSamples);
      EBM_ASSERT(1 <= pFeatureGroup->GetCountSignificantFeatures());

      const size_t cItemsPerBitPackedDataUnit = GET_COUNT_ITEMS_PER_BIT_PACKED_DATA_UNIT(
         compilerCountItemsPerBitPackedDataUnit,
         pFeatureGroup->GetCountItemsPerBitPackedDataUnit()
      );
      EBM_ASSERT(1 <= cItemsPerBitPackedDataUnit);
      EBM_ASSERT(cItemsPerBitPackedDataUnit <= k_cBitsForStorageType);
      const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackedDataUnit);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);

      const FloatEbmType * const aModelFeatureGroupUpdateTensor = pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->GetValuePointer();
      EBM_ASSERT(nullptr != aModelFeatureGroupUpdateTensor);

      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();
      const StorageDataType * pInputData = pTrainingSet->GetInputDataPointer(pFeatureGroup);
      const StorageDataType * pTargetData = pTrainingSet->GetTargetDataPointer();
      FloatEbmType * pPredictorScores = pTrainingSet->GetPredictorScores();

      // this shouldn't overflow since we're accessing existing memory
      const FloatEbmType * const pPredictorScoresTrueEnd = pPredictorScores + cSamples * cVectorLength;
      const FloatEbmType * pPredictorScoresExit = pPredictorScoresTrueEnd;
      const FloatEbmType * pPredictorScoresInnerEnd = pPredictorScoresTrueEnd;
      if(cSamples <= cItemsPerBitPackedDataUnit) {
         goto one_last_loop;
      }
      pPredictorScoresExit = pPredictorScoresTrueEnd - ((cSamples - 1) % cItemsPerBitPackedDataUnit + 1) * cVectorLength;
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
            const FloatEbmType * pValues = &aModelFeatureGroupUpdateTensor[iTensorBin * cVectorLength];
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
               const FloatEbmType oneExp = ExpForResidualsMulticlass(predictorScore);
               *pExpVector = oneExp;
               ++pExpVector;
               sumExp += oneExp;
               ++iVector;
            } while(iVector < cVectorLength);
            pExpVector -= cVectorLength;
            iVector = 0;
            do {
               const FloatEbmType residualError = EbmStats::ComputeResidualErrorMulticlass(
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
               *(pResidualError - (static_cast<ptrdiff_t>(cVectorLength) - k_iZeroResidual)) = 0;
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
class ApplyModelUpdateTrainingInternal<2, compilerCountItemsPerBitPackedDataUnit> final {
public:

   ApplyModelUpdateTrainingInternal() = delete; // this is a static class.  Do not construct

   static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) {
      UNUSED(pThreadStateBoosting);

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      const size_t runtimeCountItemsPerBitPackedDataUnit = pFeatureGroup->GetCountItemsPerBitPackedDataUnit();
      DataFrameBoosting * const pTrainingSet = pBooster->GetTrainingSet();

      const size_t cSamples = pTrainingSet->GetCountSamples();
      EBM_ASSERT(1 <= cSamples);
      EBM_ASSERT(1 <= pFeatureGroup->GetCountSignificantFeatures());

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

      const FloatEbmType * const aModelFeatureGroupUpdateTensor = pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->GetValuePointer();
      EBM_ASSERT(nullptr != aModelFeatureGroupUpdateTensor);

      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();
      const StorageDataType * pInputData = pTrainingSet->GetInputDataPointer(pFeatureGroup);
      const StorageDataType * pTargetData = pTrainingSet->GetTargetDataPointer();
      FloatEbmType * pPredictorScores = pTrainingSet->GetPredictorScores();

      // this shouldn't overflow since we're accessing existing memory
      const FloatEbmType * const pPredictorScoresTrueEnd = pPredictorScores + cSamples;
      const FloatEbmType * pPredictorScoresExit = pPredictorScoresTrueEnd;
      const FloatEbmType * pPredictorScoresInnerEnd = pPredictorScoresTrueEnd;
      if(cSamples <= cItemsPerBitPackedDataUnit) {
         goto one_last_loop;
      }
      pPredictorScoresExit = pPredictorScoresTrueEnd - ((cSamples - 1) % cItemsPerBitPackedDataUnit + 1);
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

            const FloatEbmType smallChangeToPredictorScores = aModelFeatureGroupUpdateTensor[iTensorBin];
            // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
            const FloatEbmType predictorScore = *pPredictorScores + smallChangeToPredictorScores;
            *pPredictorScores = predictorScore;
            ++pPredictorScores;
            const FloatEbmType residualError = EbmStats::ComputeResidualErrorBinaryClassification(predictorScore, targetData);

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
class ApplyModelUpdateTrainingInternal<k_regression, compilerCountItemsPerBitPackedDataUnit> final {
public:

   ApplyModelUpdateTrainingInternal() = delete; // this is a static class.  Do not construct

   static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) {
      UNUSED(pThreadStateBoosting);

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      const size_t runtimeCountItemsPerBitPackedDataUnit = pFeatureGroup->GetCountItemsPerBitPackedDataUnit();
      DataFrameBoosting * const pTrainingSet = pBooster->GetTrainingSet();

      const size_t cSamples = pTrainingSet->GetCountSamples();
      EBM_ASSERT(1 <= cSamples);
      EBM_ASSERT(1 <= pFeatureGroup->GetCountSignificantFeatures());

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

      const FloatEbmType * const aModelFeatureGroupUpdateTensor = pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->GetValuePointer();
      EBM_ASSERT(nullptr != aModelFeatureGroupUpdateTensor);

      FloatEbmType * pResidualError = pTrainingSet->GetResidualPointer();
      const StorageDataType * pInputData = pTrainingSet->GetInputDataPointer(pFeatureGroup);

      // this shouldn't overflow since we're accessing existing memory
      const FloatEbmType * const pResidualErrorTrueEnd = pResidualError + cSamples;
      const FloatEbmType * pResidualErrorExit = pResidualErrorTrueEnd;
      const FloatEbmType * pResidualErrorInnerEnd = pResidualErrorTrueEnd;
      if(cSamples <= cItemsPerBitPackedDataUnit) {
         goto one_last_loop;
      }
      pResidualErrorExit = pResidualErrorTrueEnd - ((cSamples - 1) % cItemsPerBitPackedDataUnit + 1);
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

            const FloatEbmType smallChangeToPrediction = aModelFeatureGroupUpdateTensor[iTensorBin];
            // this will apply a small fix to our existing TrainingPredictorScores, either positive or negative, whichever is needed
            const FloatEbmType residualError = EbmStats::ComputeResidualErrorRegression(*pResidualError - smallChangeToPrediction);

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

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class ApplyModelUpdateTrainingNormalTarget final {
public:

   ApplyModelUpdateTrainingNormalTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         ApplyModelUpdateTrainingInternal<compilerLearningTypeOrCountTargetClassesPossible, k_cItemsPerBitPackedDataUnitDynamic>::Func(
            pThreadStateBoosting,
            pFeatureGroup
         );
      } else {
         ApplyModelUpdateTrainingNormalTarget<
            compilerLearningTypeOrCountTargetClassesPossible + 1
         >::Func(
            pThreadStateBoosting,
            pFeatureGroup
         );
      }
   }
};

template<>
class ApplyModelUpdateTrainingNormalTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   ApplyModelUpdateTrainingNormalTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses());

      ApplyModelUpdateTrainingInternal<k_dynamicClassification, k_cItemsPerBitPackedDataUnitDynamic>::Func(
         pThreadStateBoosting,
         pFeatureGroup
      );
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerCountItemsPerBitPackedDataUnitPossible>
class ApplyModelUpdateTrainingSIMDPacking final {
public:

   ApplyModelUpdateTrainingSIMDPacking() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) {
      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      const size_t runtimeCountItemsPerBitPackedDataUnit = pFeatureGroup->GetCountItemsPerBitPackedDataUnit();

      EBM_ASSERT(1 <= runtimeCountItemsPerBitPackedDataUnit);
      EBM_ASSERT(runtimeCountItemsPerBitPackedDataUnit <= k_cBitsForStorageType);
      static_assert(compilerCountItemsPerBitPackedDataUnitPossible <= k_cBitsForStorageType, "We can't have this many items in a data pack.");
      if(compilerCountItemsPerBitPackedDataUnitPossible == runtimeCountItemsPerBitPackedDataUnit) {
         ApplyModelUpdateTrainingInternal<compilerLearningTypeOrCountTargetClasses, compilerCountItemsPerBitPackedDataUnitPossible>::Func(
            pThreadStateBoosting,
            pFeatureGroup
         );
      } else {
         ApplyModelUpdateTrainingSIMDPacking<
            compilerLearningTypeOrCountTargetClasses,
            GetNextCountItemsBitPacked(compilerCountItemsPerBitPackedDataUnitPossible)
         >::Func(
            pThreadStateBoosting,
            pFeatureGroup
         );
      }
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class ApplyModelUpdateTrainingSIMDPacking<compilerLearningTypeOrCountTargetClasses, k_cItemsPerBitPackedDataUnitDynamic> final {
public:

   ApplyModelUpdateTrainingSIMDPacking() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) {
      EBM_ASSERT(1 <= pFeatureGroup->GetCountItemsPerBitPackedDataUnit());
      EBM_ASSERT(pFeatureGroup->GetCountItemsPerBitPackedDataUnit() <= k_cBitsForStorageType);
      ApplyModelUpdateTrainingInternal<compilerLearningTypeOrCountTargetClasses, k_cItemsPerBitPackedDataUnitDynamic>::Func(
         pThreadStateBoosting,
         pFeatureGroup
      );
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class ApplyModelUpdateTrainingSIMDTarget final {
public:

   ApplyModelUpdateTrainingSIMDTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         ApplyModelUpdateTrainingSIMDPacking<
            compilerLearningTypeOrCountTargetClassesPossible,
            k_cItemsPerBitPackedDataUnitMax
         >::Func(
            pThreadStateBoosting,
            pFeatureGroup
         );
      } else {
         ApplyModelUpdateTrainingSIMDTarget<
            compilerLearningTypeOrCountTargetClassesPossible + 1
         >::Func(
            pThreadStateBoosting,
            pFeatureGroup
         );
      }
   }
};

template<>
class ApplyModelUpdateTrainingSIMDTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   ApplyModelUpdateTrainingSIMDTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses());

      ApplyModelUpdateTrainingSIMDPacking<k_dynamicClassification, k_cItemsPerBitPackedDataUnitMax>::Func(
         pThreadStateBoosting,
         pFeatureGroup
      );
   }
};

extern void ApplyModelUpdateTraining(
   ThreadStateBoosting * const pThreadStateBoosting,
   const FeatureGroup * const pFeatureGroup
) {
   LOG_0(TraceLevelVerbose, "Entered ApplyModelUpdateTraining");

   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();

   if(0 == pFeatureGroup->GetCountSignificantFeatures()) {
      if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
         ApplyModelUpdateTrainingZeroFeaturesTarget<2>::Func(pThreadStateBoosting);
      } else {
         EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
         ApplyModelUpdateTrainingZeroFeatures<k_regression>::Func(pThreadStateBoosting);
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
            ApplyModelUpdateTrainingSIMDTarget<2>::Func(pThreadStateBoosting, pFeatureGroup);
         } else {
            EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
            ApplyModelUpdateTrainingSIMDPacking<k_regression, k_cItemsPerBitPackedDataUnitMax>::Func(
               pThreadStateBoosting,
               pFeatureGroup
            );
         }
      } else {
         // there isn't much benefit in eliminating the loop that unpacks a data unit unless we're also unpacking that to SIMD code
         // Our default packing structure is to bin continuous values to 256 values, and we have 64 bit packing structures, so we usually
         // have more than 8 values per memory fetch.  Eliminating the inner loop for multiclass is valuable since we can have low numbers like 3 class,
         // 4 class, etc, but by the time we get to 8 loops with exp inside and a lot of other instructures we should worry that our code expansion
         // will exceed the L1 instruction cache size.  With SIMD we do 8 times the work in the same number of instructions so these are lesser issues

         if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
            ApplyModelUpdateTrainingNormalTarget<2>::Func(pThreadStateBoosting, pFeatureGroup);
         } else {
            EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
            ApplyModelUpdateTrainingInternal<k_regression, k_cItemsPerBitPackedDataUnitDynamic>::Func(
               pThreadStateBoosting,
               pFeatureGroup
            );
         }
      }
   }

   LOG_0(TraceLevelVerbose, "Exited ApplyModelUpdateTraining");
}
