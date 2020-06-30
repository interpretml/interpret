// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "EbmStatisticUtils.h"
#include "Logging.h" // EBM_ASSERT & LOG

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class InitializeResidualsInternal final {
public:

   InitializeResidualsInternal() = delete; // this is a static class.  Do not construct

   static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t cInstances,
      const void * const aTargetData,
      const FloatEbmType * const aPredictorScores,
      FloatEbmType * const aTempFloatVector,
      FloatEbmType * pResidualError
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClasses), "must be classification");
      static_assert(!IsBinaryClassification(compilerLearningTypeOrCountTargetClasses), "must be multiclass");

      LOG_0(TraceLevelInfo, "Entered InitializeResiduals");

      // TODO : review this function to see if iZeroResidual was set to a valid index, does that affect the number of items in pPredictorScores (I assume so), 
      //   and does it affect any calculations below like sumExp += std::exp(predictionScore) and the equivalent.  Should we use cVectorLength or 
      //   runtimeLearningTypeOrCountTargetClasses for some of the addition
      // TODO : !!! re-examine the idea of zeroing one of the residuals with iZeroResidual after we have the ability to test large numbers of datasets
      EBM_ASSERT(0 < cInstances);
      EBM_ASSERT(nullptr != aTargetData);
      EBM_ASSERT(nullptr != aPredictorScores);
      EBM_ASSERT(nullptr != pResidualError);

      FloatEbmType aLocalExpVector[
         k_dynamicClassification == compilerLearningTypeOrCountTargetClasses ? 1 : GetVectorLength(compilerLearningTypeOrCountTargetClasses)
      ];
      FloatEbmType * const aExpVector = k_dynamicClassification == compilerLearningTypeOrCountTargetClasses ? aTempFloatVector : aLocalExpVector;

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);

      const IntEbmType * pTargetData = static_cast<const IntEbmType *>(aTargetData);
      const FloatEbmType * pPredictorScores = aPredictorScores;
      const FloatEbmType * const pResidualErrorEnd = pResidualError + cInstances * cVectorLength;

      do {
         const IntEbmType targetOriginal = *pTargetData;
         ++pTargetData;
         EBM_ASSERT(0 <= targetOriginal);
         // if we can't fit it, then we should increase our StorageDataType size!
         EBM_ASSERT((IsNumberConvertable<size_t, IntEbmType>(targetOriginal)));
         const size_t target = static_cast<size_t>(targetOriginal);
         EBM_ASSERT(target < static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses));
         FloatEbmType * pExpVector = aExpVector;

         FloatEbmType sumExp = FloatEbmType { 0 };
         // TODO : eventually eliminate this subtract variable once we've decided how to handle removing one logit
         const FloatEbmType subtract = 
            0 <= k_iZeroClassificationLogitAtInitialize ? pPredictorScores[k_iZeroClassificationLogitAtInitialize] : FloatEbmType { 0 };

         size_t iVector = 0;
         do {
            const FloatEbmType predictorScore = *pPredictorScores - subtract;
            ++pPredictorScores;
            const FloatEbmType oneExp = EbmExp(predictorScore);
            *pExpVector = oneExp;
            ++pExpVector;
            sumExp += oneExp;
            ++iVector;
         } while(iVector < cVectorLength);

         // go back to the start so that we can iterate again
         pExpVector -= cVectorLength;

         iVector = 0;
         do {
            const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorMulticlass(sumExp, *pExpVector, target, iVector);
            ++pExpVector;
            *pResidualError = residualError;
            ++pResidualError;
            ++iVector;
         } while(iVector < cVectorLength);

         // TODO: this works as a way to remove one parameter, but it obviously insn't as efficient as omitting the parameter
         // 
         // this works out in the math as making the first model vector parameter equal to zero, which in turn removes one degree of freedom
         // from the model vector parameters.  Since the model vector weights need to be normalized to sum to a probabilty of 100%, we can set the first
         // one to the constant 1 (0 in log space) and force the other parameters to adjust to that scale which fixes them to a single valid set of values
         // insted of allowing them to be scaled.  
         // Probability = exp(T1 + I1) / [exp(T1 + I1) + exp(T2 + I2) + exp(T3 + I3)] => we can add a constant inside each exp(..) term, which will be 
         // multiplication outside the exp(..), which means the numerator and denominator are multiplied by the same constant, which cancels eachother out.
         // We can thus set exp(T2 + I2) to exp(0) and adjust the other terms
         constexpr bool bZeroingResiduals = 0 <= k_iZeroResidual;
         if(bZeroingResiduals) {
            pResidualError[k_iZeroResidual - static_cast<ptrdiff_t>(cVectorLength)] = 0;
         }
      } while(pResidualErrorEnd != pResidualError);

      LOG_0(TraceLevelInfo, "Exited InitializeResiduals");
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<>
class InitializeResidualsInternal<2> final {
public:

   InitializeResidualsInternal() = delete; // this is a static class.  Do not construct

   static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t cInstances,
      const void * const aTargetData,
      const FloatEbmType * const aPredictorScores,
      FloatEbmType * const aTempFloatVector,
      FloatEbmType * pResidualError
   ) {
      UNUSED(runtimeLearningTypeOrCountTargetClasses);
      UNUSED(aTempFloatVector);
      LOG_0(TraceLevelInfo, "Entered InitializeResiduals");

      // TODO : review this function to see if iZeroResidual was set to a valid index, does that affect the number of items in pPredictorScores (I assume so), 
      //   and does it affect any calculations below like sumExp += std::exp(predictionScore) and the equivalent.  Should we use cVectorLength or 
      //   runtimeLearningTypeOrCountTargetClasses for some of the addition
      // TODO : !!! re-examine the idea of zeroing one of the residuals with iZeroResidual after we have the ability to test large numbers of datasets
      EBM_ASSERT(0 < cInstances);
      EBM_ASSERT(nullptr != aTargetData);
      EBM_ASSERT(nullptr != aPredictorScores);
      EBM_ASSERT(nullptr != pResidualError);

      const IntEbmType * pTargetData = static_cast<const IntEbmType *>(aTargetData);
      const FloatEbmType * pPredictorScores = aPredictorScores;
      const FloatEbmType * const pResidualErrorEnd = pResidualError + cInstances;

      do {
         const IntEbmType targetOriginal = *pTargetData;
         ++pTargetData;
         EBM_ASSERT(0 <= targetOriginal);
         // if we can't fit it, then we should increase our StorageDataType size!
         EBM_ASSERT((IsNumberConvertable<size_t, IntEbmType>(targetOriginal)));
         const size_t target = static_cast<size_t>(targetOriginal);
         EBM_ASSERT(target < static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses));
         const FloatEbmType predictionScore = *pPredictorScores;
         ++pPredictorScores;
         const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorBinaryClassification(predictionScore, target);
         *pResidualError = residualError;
         ++pResidualError;
      } while(pResidualErrorEnd != pResidualError);
      LOG_0(TraceLevelInfo, "Exited InitializeResiduals");
   }
};
#endif // EXPAND_BINARY_LOGITS

template<>
class InitializeResidualsInternal<k_regression> final {
public:

   InitializeResidualsInternal() = delete; // this is a static class.  Do not construct

   static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t cInstances,
      const void * const aTargetData,
      const FloatEbmType * const aPredictorScores,
      FloatEbmType * const aTempFloatVector,
      FloatEbmType * pResidualError
   ) {
      UNUSED(runtimeLearningTypeOrCountTargetClasses);
      UNUSED(aTempFloatVector);
      LOG_0(TraceLevelInfo, "Entered InitializeResiduals");

      // TODO : review this function to see if iZeroResidual was set to a valid index, does that affect the number of items in pPredictorScores (I assume so), 
      //   and does it affect any calculations below like sumExp += std::exp(predictionScore) and the equivalent.  Should we use cVectorLength or 
      //   runtimeLearningTypeOrCountTargetClasses for some of the addition
      // TODO : !!! re-examine the idea of zeroing one of the residuals with iZeroResidual after we have the ability to test large numbers of datasets
      EBM_ASSERT(0 < cInstances);
      EBM_ASSERT(nullptr != aTargetData);
      EBM_ASSERT(nullptr != aPredictorScores);
      EBM_ASSERT(nullptr != pResidualError);

      const FloatEbmType * pTargetData = static_cast<const FloatEbmType *>(aTargetData);
      const FloatEbmType * pPredictorScores = aPredictorScores;
      const FloatEbmType * const pResidualErrorEnd = pResidualError + cInstances;
      do {
         // TODO : our caller should handle NaN *pTargetData values, which means that the target is missing, which means we should delete that instance 
         //   from the input data

         // if data is NaN, we pass this along and NaN propagation will ensure that we stop boosting immediately.
         // There is no need to check it here since we already have graceful detection later for other reasons.

         const FloatEbmType data = *pTargetData;
         ++pTargetData;
         // TODO: NaN target values essentially mean missing, so we should be filtering those instances out, but our caller should do that so 
         //   that we don't need to do the work here per outer bag.  Our job in C++ is just not to crash or return inexplicable values.
         const FloatEbmType predictionScore = *pPredictorScores;
         ++pPredictorScores;
         const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorRegressionInit(predictionScore, data);
         *pResidualError = residualError;
         ++pResidualError;
      } while(pResidualErrorEnd != pResidualError);
      LOG_0(TraceLevelInfo, "Exited InitializeResiduals");
   }
};

extern void InitializeResiduals(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const size_t cInstances,
   const void * const aTargetData,
   const FloatEbmType * const aPredictorScores,
   FloatEbmType * const aTempFloatVector,
   FloatEbmType * pResidualError
) {
   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      if(IsBinaryClassification(runtimeLearningTypeOrCountTargetClasses)) {
         InitializeResidualsInternal<2>::Func(
            runtimeLearningTypeOrCountTargetClasses,
            cInstances,
            aTargetData,
            aPredictorScores,
            aTempFloatVector,
            pResidualError
         );
      } else {
         InitializeResidualsInternal<k_dynamicClassification>::Func(
            runtimeLearningTypeOrCountTargetClasses,
            cInstances,
            aTargetData,
            aPredictorScores,
            aTempFloatVector,
            pResidualError
         );
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      InitializeResidualsInternal<k_regression>::Func(
         runtimeLearningTypeOrCountTargetClasses,
         cInstances,
         aTargetData,
         aPredictorScores,
         aTempFloatVector,
         pResidualError
      );
   }
}
