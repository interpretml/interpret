// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// TODO: rename this file InitializeGradients.cpp

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "EbmInternal.h"

#include "EbmStats.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

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
      FloatEbmType * pGradientAndHessian
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClasses), "must be classification");
      static_assert(!IsBinaryClassification(compilerLearningTypeOrCountTargetClasses), "must be multiclass");

      LOG_0(TraceLevelInfo, "Entered InitializeGradients");

      EBM_ASSERT(0 < cSamples);
      EBM_ASSERT(nullptr != aTargetData);
      EBM_ASSERT(nullptr != aPredictorScores);
      EBM_ASSERT(nullptr != pGradientAndHessian);

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
      const FloatEbmType * const pGradientAndHessiansEnd = pGradientAndHessian + 2 * cSamples * cVectorLength;

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

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
         FloatEbmType zeroLogit = FloatEbmType { 0 };
#endif // ZERO_FIRST_MULTICLASS_LOGIT

         size_t iVector = 0;
         do {
            FloatEbmType predictorScore = *pPredictorScores;

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
            if(IsMulticlass(compilerLearningTypeOrCountTargetClasses)) {
               if(size_t { 0 } == iVector) {
                  zeroLogit = predictorScore;
               }
               predictorScore -= zeroLogit;
            }
#endif // ZERO_FIRST_MULTICLASS_LOGIT

            ++pPredictorScores;
            const FloatEbmType oneExp = ExpForMulticlass<false>(predictorScore);
            *pExpVector = oneExp;
            ++pExpVector;
            sumExp += oneExp;
            ++iVector;
         } while(iVector < cVectorLength);

         // go back to the start so that we can iterate again
         pExpVector -= cVectorLength;

         iVector = 0;
         do {
            FloatEbmType gradient;
            FloatEbmType hessian;
            EbmStats::TransformScoreToGradientAndHessianMulticlass(sumExp, *pExpVector, target, iVector, gradient, hessian);
            ++pExpVector;
            *pGradientAndHessian = gradient;
            *(pGradientAndHessian + 1) = hessian;
            pGradientAndHessian += 2;
            ++iVector;
         } while(iVector < cVectorLength);
      } while(pGradientAndHessiansEnd != pGradientAndHessian);

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
      FloatEbmType * pGradientAndHessian
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
      EBM_ASSERT(nullptr != pGradientAndHessian);

      const IntEbmType * pTargetData = static_cast<const IntEbmType *>(aTargetData);
      const FloatEbmType * pPredictorScores = aPredictorScores;
      const FloatEbmType * const pGradientAndHessiansEnd = pGradientAndHessian + 2 * cSamples;

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
         *pGradientAndHessian = gradient;
         *(pGradientAndHessian + 1) = EbmStats::CalculateHessianFromGradientBinaryClassification(gradient);
         pGradientAndHessian += 2;
      } while(pGradientAndHessiansEnd != pGradientAndHessian);
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
      FloatEbmType * pGradientAndHessian
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
      EBM_ASSERT(nullptr != pGradientAndHessian);

      const FloatEbmType * pTargetData = static_cast<const FloatEbmType *>(aTargetData);
      const FloatEbmType * pPredictorScores = aPredictorScores;
      const FloatEbmType * const pGradientAndHessiansEnd = pGradientAndHessian + cSamples;
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
         *pGradientAndHessian = gradient;
         ++pGradientAndHessian;
      } while(pGradientAndHessiansEnd != pGradientAndHessian);
      LOG_0(TraceLevelInfo, "Exited InitializeGradients");
      return false;
   }
};

extern bool InitializeGradients(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const size_t cSamples,
   const void * const aTargetData,
   const FloatEbmType * const aPredictorScores,
   FloatEbmType * pGradientAndHessian
) {
   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      if(IsBinaryClassification(runtimeLearningTypeOrCountTargetClasses)) {
         return InitializeGradientsInternal<2>::Func(
            runtimeLearningTypeOrCountTargetClasses,
            cSamples,
            aTargetData,
            aPredictorScores,
            pGradientAndHessian
         );
      } else {
         return InitializeGradientsInternal<k_dynamicClassification>::Func(
            runtimeLearningTypeOrCountTargetClasses,
            cSamples,
            aTargetData,
            aPredictorScores,
            pGradientAndHessian
         );
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      return InitializeGradientsInternal<k_regression>::Func(
         runtimeLearningTypeOrCountTargetClasses,
         cSamples,
         aTargetData,
         aPredictorScores,
         pGradientAndHessian
      );
   }
}

} // DEFINED_ZONE_NAME
