// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// TODO: rename this file InitializeGradients.cpp

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "EbmStats.h"
#include "Logging.h" // EBM_ASSERT & LOG

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class InitializeGradientsInternal final {
public:

   InitializeGradientsInternal() = delete; // this is a static class.  Do not construct

   static bool Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t cSamples,
      const void * const aTargetData,
      const FloatEbmType * const aPredictorScores,
      FloatEbmType * pGradient
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClasses), "must be classification");
      static_assert(!IsBinaryClassification(compilerLearningTypeOrCountTargetClasses), "must be multiclass");

      LOG_0(TraceLevelInfo, "Entered InitializeGradients");

      // TODO : review this function to see if iZeroLogit was set to a valid index, does that affect the number of items in pPredictorScores (I assume so), 
      //   and does it affect any calculations below like sumExp += std::exp(predictionScore) and the equivalent.  Should we use cVectorLength or 
      //   runtimeLearningTypeOrCountTargetClasses for some of the addition
      // TODO : !!! re-examine the idea of zeroing one of the logits with iZeroLogit after we have the ability to test large numbers of datasets
      EBM_ASSERT(0 < cSamples);
      EBM_ASSERT(nullptr != aTargetData);
      EBM_ASSERT(nullptr != aPredictorScores);
      EBM_ASSERT(nullptr != pGradient);

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);

      FloatEbmType aLocalExpVector[
         k_dynamicClassification == compilerLearningTypeOrCountTargetClasses ? 1 : GetVectorLength(compilerLearningTypeOrCountTargetClasses)
      ];
      FloatEbmType * const aExpVector = k_dynamicClassification == compilerLearningTypeOrCountTargetClasses ? EbmMalloc<FloatEbmType>(cVectorLength) : aLocalExpVector;
      if(UNLIKELY(nullptr == aExpVector)) {
         LOG_0(TraceLevelWarning, "WARNING InitializeGradients nullptr == aExpVector");
         return true;
      }

      const IntEbmType * pTargetData = static_cast<const IntEbmType *>(aTargetData);
      const FloatEbmType * pPredictorScores = aPredictorScores;
      const FloatEbmType * const pGradientsEnd = pGradient + cSamples * cVectorLength;

      do {
         const IntEbmType targetOriginal = *pTargetData;
         ++pTargetData;
         EBM_ASSERT(0 <= targetOriginal);
         // if we can't fit it, then we should increase our StorageDataType size!
         EBM_ASSERT(IsNumberConvertable<size_t>(targetOriginal));
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
            const FloatEbmType oneExp = ExpForMulticlass(predictorScore);
            *pExpVector = oneExp;
            ++pExpVector;
            sumExp += oneExp;
            ++iVector;
         } while(iVector < cVectorLength);

         // go back to the start so that we can iterate again
         pExpVector -= cVectorLength;

         iVector = 0;
         do {
            const FloatEbmType gradient = EbmStats::TransformScoreToGradientMulticlass(sumExp, *pExpVector, target, iVector);
            ++pExpVector;
            *pGradient = gradient;
            ++pGradient;
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
         constexpr bool bZeroingLogits = 0 <= k_iZeroLogit;
         if(bZeroingLogits) {
            pGradient[k_iZeroLogit - static_cast<ptrdiff_t>(cVectorLength)] = 0;
         }
      } while(pGradientsEnd != pGradient);

      if(UNLIKELY(aExpVector != aLocalExpVector)) {
         free(aExpVector);
      }

      LOG_0(TraceLevelInfo, "Exited InitializeGradients");
      return false;
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<>
class InitializeGradientsInternal<2> final {
public:

   InitializeGradientsInternal() = delete; // this is a static class.  Do not construct

   static bool Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t cSamples,
      const void * const aTargetData,
      const FloatEbmType * const aPredictorScores,
      FloatEbmType * pGradient
   ) {
      UNUSED(runtimeLearningTypeOrCountTargetClasses);
      LOG_0(TraceLevelInfo, "Entered InitializeGradients");

      // TODO : review this function to see if iZeroLogit was set to a valid index, does that affect the number of items in pPredictorScores (I assume so), 
      //   and does it affect any calculations below like sumExp += std::exp(predictionScore) and the equivalent.  Should we use cVectorLength or 
      //   runtimeLearningTypeOrCountTargetClasses for some of the addition
      // TODO : !!! re-examine the idea of zeroing one of the logits with iZeroLogit after we have the ability to test large numbers of datasets
      EBM_ASSERT(0 < cSamples);
      EBM_ASSERT(nullptr != aTargetData);
      EBM_ASSERT(nullptr != aPredictorScores);
      EBM_ASSERT(nullptr != pGradient);

      const IntEbmType * pTargetData = static_cast<const IntEbmType *>(aTargetData);
      const FloatEbmType * pPredictorScores = aPredictorScores;
      const FloatEbmType * const pGradientsEnd = pGradient + cSamples;

      do {
         const IntEbmType targetOriginal = *pTargetData;
         ++pTargetData;
         EBM_ASSERT(0 <= targetOriginal);
         // if we can't fit it, then we should increase our StorageDataType size!
         EBM_ASSERT(IsNumberConvertable<size_t>(targetOriginal));
         const size_t target = static_cast<size_t>(targetOriginal);
         EBM_ASSERT(target < static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses));
         const FloatEbmType predictionScore = *pPredictorScores;
         ++pPredictorScores;
         const FloatEbmType gradient = EbmStats::TransformScoreToGradientBinaryClassification(predictionScore, target);
         *pGradient = gradient;
         ++pGradient;
      } while(pGradientsEnd != pGradient);
      LOG_0(TraceLevelInfo, "Exited InitializeGradients");
      return false;
   }
};
#endif // EXPAND_BINARY_LOGITS

template<>
class InitializeGradientsInternal<k_regression> final {
public:

   InitializeGradientsInternal() = delete; // this is a static class.  Do not construct

   static bool Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t cSamples,
      const void * const aTargetData,
      const FloatEbmType * const aPredictorScores,
      FloatEbmType * pGradient
   ) {
      UNUSED(runtimeLearningTypeOrCountTargetClasses);
      LOG_0(TraceLevelInfo, "Entered InitializeGradients");

      // TODO : review this function to see if iZeroLogit was set to a valid index, does that affect the number of items in pPredictorScores (I assume so), 
      //   and does it affect any calculations below like sumExp += std::exp(predictionScore) and the equivalent.  Should we use cVectorLength or 
      //   runtimeLearningTypeOrCountTargetClasses for some of the addition
      // TODO : !!! re-examine the idea of zeroing one of the logits with iZeroLogit after we have the ability to test large numbers of datasets
      EBM_ASSERT(0 < cSamples);
      EBM_ASSERT(nullptr != aTargetData);
      EBM_ASSERT(nullptr != aPredictorScores);
      EBM_ASSERT(nullptr != pGradient);

      const FloatEbmType * pTargetData = static_cast<const FloatEbmType *>(aTargetData);
      const FloatEbmType * pPredictorScores = aPredictorScores;
      const FloatEbmType * const pGradientsEnd = pGradient + cSamples;
      do {
         // TODO : our caller should handle NaN *pTargetData values, which means that the target is missing, which means we should delete that sample 
         //   from the input data

         // if data is NaN, we pass this along and NaN propagation will ensure that we stop boosting immediately.
         // There is no need to check it here since we already have graceful detection later for other reasons.

         const FloatEbmType data = *pTargetData;
         ++pTargetData;
         // TODO: NaN target values essentially mean missing, so we should be filtering those samples out, but our caller should do that so 
         //   that we don't need to do the work here per outer bag.  Our job in C++ is just not to crash or return inexplicable values.
         const FloatEbmType predictionScore = *pPredictorScores;
         ++pPredictorScores;
         const FloatEbmType gradient = EbmStats::ComputeGradientRegressionMSEInit(predictionScore, data);
         *pGradient = gradient;
         ++pGradient;
      } while(pGradientsEnd != pGradient);
      LOG_0(TraceLevelInfo, "Exited InitializeGradients");
      return false;
   }
};

extern bool InitializeGradients(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const size_t cSamples,
   const void * const aTargetData,
   const FloatEbmType * const aPredictorScores,
   FloatEbmType * pGradient
) {
   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      if(IsBinaryClassification(runtimeLearningTypeOrCountTargetClasses)) {
         return InitializeGradientsInternal<2>::Func(
            runtimeLearningTypeOrCountTargetClasses,
            cSamples,
            aTargetData,
            aPredictorScores,
            pGradient
         );
      } else {
         return InitializeGradientsInternal<k_dynamicClassification>::Func(
            runtimeLearningTypeOrCountTargetClasses,
            cSamples,
            aTargetData,
            aPredictorScores,
            pGradient
         );
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      return InitializeGradientsInternal<k_regression>::Func(
         runtimeLearningTypeOrCountTargetClasses,
         cSamples,
         aTargetData,
         aPredictorScores,
         pGradient
      );
   }
}
