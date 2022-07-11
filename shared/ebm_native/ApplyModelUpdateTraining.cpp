// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "approximate_math.hpp"
#include "ebm_stats.hpp"
// FeatureGroup.hpp depends on FeatureInternal.h
#include "FeatureGroup.hpp"
// dataset depends on features
#include "DataSetBoosting.hpp"

#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// C++ does not allow partial function specialization, so we need to use these cumbersome static class functions to do partial function specialization

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class ApplyTermUpdateTrainingZeroFeatures final {
public:

   ApplyTermUpdateTrainingZeroFeatures() = delete; // this is a static class.  Do not construct

   static void Func(BoosterShell * const pBoosterShell) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClasses), "must be classification");
      static_assert(!IsBinaryClassification(compilerLearningTypeOrCountTargetClasses), "must be multiclass");

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
      DataSetBoosting * const pTrainingSet = pBoosterCore->GetTrainingSet();
      FloatFast * const aTempFloatVector = pBoosterShell->GetTempFloatVector();

      FloatFast aLocalExpVector[
         k_dynamicClassification == compilerLearningTypeOrCountTargetClasses ? 1 : GetVectorLength(compilerLearningTypeOrCountTargetClasses)
      ];
      FloatFast * const aExpVector = k_dynamicClassification == compilerLearningTypeOrCountTargetClasses ? aTempFloatVector : aLocalExpVector;

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      const size_t cSamples = pTrainingSet->GetCountSamples();
      EBM_ASSERT(1 <= cSamples);

      const FloatFast * const aUpdateScores = pBoosterShell->GetTermUpdate()->GetScoresPointer();
      EBM_ASSERT(nullptr != aUpdateScores);

      FloatFast * pGradientAndHessian = pTrainingSet->GetGradientsAndHessiansPointer();
      const StorageDataType * pTargetData = pTrainingSet->GetTargetDataPointer();
      FloatFast * pSampleScore = pTrainingSet->GetSampleScores();
      const FloatFast * const pSampleScoresEnd = pSampleScore + cSamples * cVectorLength;
      do {
         size_t targetData = static_cast<size_t>(*pTargetData);
         ++pTargetData;

         const FloatFast * pUpdateScore = aUpdateScores;
         FloatFast * pExpVector = aExpVector;
         FloatFast sumExp = FloatFast { 0 };
         size_t iVector = 0;
         do {
            // TODO : because there is only one bin for a zero feature feature group, we could move these values to the stack where the
            // compiler could reason about their visibility and optimize small arrays into registers
            const FloatFast updateScore = *pUpdateScore;
            ++pUpdateScore;
            // this will apply a small fix to our existing TrainingSampleScores, either positive or negative, whichever is needed
            const FloatFast sampleScore = *pSampleScore + updateScore;

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
            if(IsMulticlass(compilerLearningTypeOrCountTargetClasses)) {
               if(size_t { 0 } == iVector) {
                  EBM_ASSERT(0 == updateScore);
                  EBM_ASSERT(0 == sampleScore);
               }
            }
#endif // ZERO_FIRST_MULTICLASS_LOGIT

            *pSampleScore = sampleScore;
            ++pSampleScore;
            const FloatFast oneExp = ExpForMulticlass<false>(sampleScore);
            *pExpVector = oneExp;
            ++pExpVector;
            sumExp += oneExp;
            ++iVector;
         } while(iVector < cVectorLength);
         pExpVector -= cVectorLength;
         iVector = 0;
         do {
            FloatFast gradient;
            FloatFast hessian;
            EbmStats::InverseLinkFunctionThenCalculateGradientAndHessianMulticlass(
               sumExp,
               *pExpVector,
               targetData,
               iVector,
               gradient,
               hessian
            );
            ++pExpVector;
            *pGradientAndHessian = gradient;
            *(pGradientAndHessian + 1) = hessian;
            pGradientAndHessian += 2;
            ++iVector;
         } while(iVector < cVectorLength);
      } while(pSampleScoresEnd != pSampleScore);
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<>
class ApplyTermUpdateTrainingZeroFeatures<2> final {
public:

   ApplyTermUpdateTrainingZeroFeatures() = delete; // this is a static class.  Do not construct

   static void Func(BoosterShell * const pBoosterShell) {
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      DataSetBoosting * const pTrainingSet = pBoosterCore->GetTrainingSet();
      const size_t cSamples = pTrainingSet->GetCountSamples();
      EBM_ASSERT(1 <= cSamples);

      const FloatFast * const aUpdateScores = pBoosterShell->GetTermUpdate()->GetScoresPointer();
      EBM_ASSERT(nullptr != aUpdateScores);

      FloatFast * pGradientAndHessian = pTrainingSet->GetGradientsAndHessiansPointer();
      const StorageDataType * pTargetData = pTrainingSet->GetTargetDataPointer();
      FloatFast * pSampleScore = pTrainingSet->GetSampleScores();
      const FloatFast * const pSampleScoresEnd = pSampleScore + cSamples;
      const FloatFast updateScore = aUpdateScores[0];
      do {
         size_t targetData = static_cast<size_t>(*pTargetData);
         ++pTargetData;
         // this will apply a small fix to our existing TrainingSampleScores, either positive or negative, whichever is needed
         const FloatFast sampleScore = *pSampleScore + updateScore;
         *pSampleScore = sampleScore;
         ++pSampleScore;
         const FloatFast gradient = EbmStats::InverseLinkFunctionThenCalculateGradientBinaryClassification(sampleScore, targetData);
         *pGradientAndHessian = gradient;
         *(pGradientAndHessian + 1) = EbmStats::CalculateHessianFromGradientBinaryClassification(gradient);
         pGradientAndHessian += 2;
      } while(pSampleScoresEnd != pSampleScore);
   }
};
#endif // EXPAND_BINARY_LOGITS

template<>
class ApplyTermUpdateTrainingZeroFeatures<k_regression> final {
public:

   ApplyTermUpdateTrainingZeroFeatures() = delete; // this is a static class.  Do not construct

   static void Func(BoosterShell * const pBoosterShell) {
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      DataSetBoosting * const pTrainingSet = pBoosterCore->GetTrainingSet();
      const size_t cSamples = pTrainingSet->GetCountSamples();
      EBM_ASSERT(1 <= cSamples);

      const FloatFast * const aUpdateScores = pBoosterShell->GetTermUpdate()->GetScoresPointer();
      EBM_ASSERT(nullptr != aUpdateScores);

      // no hessian for regression
      FloatFast * pGradient = pTrainingSet->GetGradientsAndHessiansPointer();
      const FloatFast * const pGradientsEnd = pGradient + cSamples;
      const FloatFast updateScore = aUpdateScores[0];
      do {
         // this will apply a small fix to our existing TrainingSampleScores, either positive or negative, whichever is needed
         const FloatFast gradient = EbmStats::ComputeGradientRegressionMSEFromOriginalGradient(*pGradient, updateScore);
         *pGradient = gradient;
         ++pGradient;
      } while(pGradientsEnd != pGradient);
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class ApplyTermUpdateTrainingZeroFeaturesTarget final {
public:

   ApplyTermUpdateTrainingZeroFeaturesTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(BoosterShell * const pBoosterShell) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         ApplyTermUpdateTrainingZeroFeatures<compilerLearningTypeOrCountTargetClassesPossible>::Func(
            pBoosterShell
         );
      } else {
         ApplyTermUpdateTrainingZeroFeaturesTarget<
            compilerLearningTypeOrCountTargetClassesPossible + 1
         >::Func(
            pBoosterShell
         );
      }
   }
};

template<>
class ApplyTermUpdateTrainingZeroFeaturesTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   ApplyTermUpdateTrainingZeroFeaturesTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(BoosterShell * const pBoosterShell) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pBoosterShell->GetBoosterCore()->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pBoosterShell->GetBoosterCore()->GetRuntimeLearningTypeOrCountTargetClasses());

      ApplyTermUpdateTrainingZeroFeatures<k_dynamicClassification>::Func(pBoosterShell);
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerBitPack>
class ApplyTermUpdateTrainingInternal final {
public:

   ApplyTermUpdateTrainingInternal() = delete; // this is a static class.  Do not construct

   static void Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClasses), "must be classification");
      static_assert(!IsBinaryClassification(compilerLearningTypeOrCountTargetClasses), "must be multiclass");

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
      DataSetBoosting * const pTrainingSet = pBoosterCore->GetTrainingSet();
      FloatFast * const aTempFloatVector = pBoosterShell->GetTempFloatVector();

      FloatFast aLocalExpVector[
         k_dynamicClassification == compilerLearningTypeOrCountTargetClasses ? 1 : GetVectorLength(compilerLearningTypeOrCountTargetClasses)
      ];
      FloatFast * const aExpVector = k_dynamicClassification == compilerLearningTypeOrCountTargetClasses ? aTempFloatVector : aLocalExpVector;

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      const size_t cSamples = pTrainingSet->GetCountSamples();
      EBM_ASSERT(1 <= cSamples);
      EBM_ASSERT(1 <= pTerm->GetCountSignificantDimensions());

      const size_t cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, pTerm->GetBitPack());
      EBM_ASSERT(1 <= cItemsPerBitPack);
      EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
      const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPack);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);

      const FloatFast * const aUpdateScores = pBoosterShell->GetTermUpdate()->GetScoresPointer();
      EBM_ASSERT(nullptr != aUpdateScores);

      FloatFast * pGradientAndHessian = pTrainingSet->GetGradientsAndHessiansPointer();
      const StorageDataType * pInputData = pTrainingSet->GetInputDataPointer(pTerm);
      const StorageDataType * pTargetData = pTrainingSet->GetTargetDataPointer();
      FloatFast * pSampleScore = pTrainingSet->GetSampleScores();

      // this shouldn't overflow since we're accessing existing memory
      const FloatFast * const pSampleScoresTrueEnd = pSampleScore + cSamples * cVectorLength;
      const FloatFast * pSampleScoresExit = pSampleScoresTrueEnd;
      const FloatFast * pSampleScoresInnerEnd = pSampleScoresTrueEnd;
      if(cSamples <= cItemsPerBitPack) {
         goto one_last_loop;
      }
      pSampleScoresExit = pSampleScoresTrueEnd - ((cSamples - 1) % cItemsPerBitPack + 1) * cVectorLength;
      EBM_ASSERT(pSampleScore < pSampleScoresExit);
      EBM_ASSERT(pSampleScoresExit < pSampleScoresTrueEnd);

      do {
         pSampleScoresInnerEnd = pSampleScore + cItemsPerBitPack * cVectorLength;
         // jumping back into this loop and changing pSampleScoresInnerEnd to a dynamic value that isn't compile time determinable causes this 
         // function to NOT be optimized for templated cItemsPerBitPack, but that's ok since avoiding one unpredictable branch here is negligible
      one_last_loop:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iTensorBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            size_t targetData = static_cast<size_t>(*pTargetData);
            ++pTargetData;

            const size_t iTensorBin = maskBits & iTensorBinCombined;
            const FloatFast * pUpdateScore = &aUpdateScores[iTensorBin * cVectorLength];
            FloatFast * pExpVector = aExpVector;
            FloatFast sumExp = 0;
            size_t iVector = 0;
            do {
               const FloatFast updateScore = *pUpdateScore;
               ++pUpdateScore;
               // this will apply a small fix to our existing TrainingSampleScores, either positive or negative, whichever is needed
               const FloatFast sampleScore = *pSampleScore + updateScore;

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
               if(IsMulticlass(compilerLearningTypeOrCountTargetClasses)) {
                  if(size_t { 0 } == iVector) {
                     EBM_ASSERT(0 == updateScore);
                     EBM_ASSERT(0 == sampleScore);
                  }
               }
#endif // ZERO_FIRST_MULTICLASS_LOGIT

               *pSampleScore = sampleScore;
               ++pSampleScore;
               const FloatFast oneExp = ExpForMulticlass<false>(sampleScore);
               *pExpVector = oneExp;
               ++pExpVector;
               sumExp += oneExp;
               ++iVector;
            } while(iVector < cVectorLength);
            pExpVector -= cVectorLength;
            iVector = 0;
            do {
               FloatFast gradient;
               FloatFast hessian;
               EbmStats::InverseLinkFunctionThenCalculateGradientAndHessianMulticlass(
                  sumExp,
                  *pExpVector,
                  targetData,
                  iVector,
                  gradient,
                  hessian
               );
               ++pExpVector;
               *pGradientAndHessian = gradient;
               *(pGradientAndHessian + 1) = hessian;
               pGradientAndHessian += 2;
               ++iVector;
            } while(iVector < cVectorLength);

            iTensorBinCombined >>= cBitsPerItemMax;
         } while(pSampleScoresInnerEnd != pSampleScore);
      } while(pSampleScoresExit != pSampleScore);

      // first time through?
      if(pSampleScoresTrueEnd != pSampleScore) {
         pSampleScoresInnerEnd = pSampleScoresTrueEnd;
         pSampleScoresExit = pSampleScoresTrueEnd;
         goto one_last_loop;
      }
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<size_t compilerBitPack>
class ApplyTermUpdateTrainingInternal<2, compilerBitPack> final {
public:

   ApplyTermUpdateTrainingInternal() = delete; // this is a static class.  Do not construct

   static void Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm
   ) {
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const size_t runtimeBitPack = pTerm->GetBitPack();
      DataSetBoosting * const pTrainingSet = pBoosterCore->GetTrainingSet();

      const size_t cSamples = pTrainingSet->GetCountSamples();
      EBM_ASSERT(1 <= cSamples);
      EBM_ASSERT(1 <= pTerm->GetCountSignificantDimensions());

      const size_t cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, runtimeBitPack);
      EBM_ASSERT(1 <= cItemsPerBitPack);
      EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
      const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPack);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);

      const FloatFast * const aUpdateScores = pBoosterShell->GetTermUpdate()->GetScoresPointer();
      EBM_ASSERT(nullptr != aUpdateScores);

      FloatFast * pGradientAndHessian = pTrainingSet->GetGradientsAndHessiansPointer();
      const StorageDataType * pInputData = pTrainingSet->GetInputDataPointer(pTerm);
      const StorageDataType * pTargetData = pTrainingSet->GetTargetDataPointer();
      FloatFast * pSampleScore = pTrainingSet->GetSampleScores();

      // this shouldn't overflow since we're accessing existing memory
      const FloatFast * const pSampleScoresTrueEnd = pSampleScore + cSamples;
      const FloatFast * pSampleScoresExit = pSampleScoresTrueEnd;
      const FloatFast * pSampleScoresInnerEnd = pSampleScoresTrueEnd;
      if(cSamples <= cItemsPerBitPack) {
         goto one_last_loop;
      }
      pSampleScoresExit = pSampleScoresTrueEnd - ((cSamples - 1) % cItemsPerBitPack + 1);
      EBM_ASSERT(pSampleScore < pSampleScoresExit);
      EBM_ASSERT(pSampleScoresExit < pSampleScoresTrueEnd);

      do {
         pSampleScoresInnerEnd = pSampleScore + cItemsPerBitPack;
         // jumping back into this loop and changing pSampleScoresInnerEnd to a dynamic value that isn't compile time determinable causes this 
         // function to NOT be optimized for templated cItemsPerBitPack, but that's ok since avoiding one unpredictable branch here is negligible
      one_last_loop:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iTensorBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            size_t targetData = static_cast<size_t>(*pTargetData);
            ++pTargetData;

            const size_t iTensorBin = maskBits & iTensorBinCombined;

            const FloatFast updateScore = aUpdateScores[iTensorBin];
            // this will apply a small fix to our existing TrainingSampleScores, either positive or negative, whichever is needed
            const FloatFast sampleScore = *pSampleScore + updateScore;
            *pSampleScore = sampleScore;
            ++pSampleScore;
            const FloatFast gradient = EbmStats::InverseLinkFunctionThenCalculateGradientBinaryClassification(sampleScore, targetData);

            *pGradientAndHessian = gradient;
            *(pGradientAndHessian + 1) = EbmStats::CalculateHessianFromGradientBinaryClassification(gradient);
            pGradientAndHessian += 2;

            iTensorBinCombined >>= cBitsPerItemMax;
         } while(pSampleScoresInnerEnd != pSampleScore);
      } while(pSampleScoresExit != pSampleScore);

      // first time through?
      if(pSampleScoresTrueEnd != pSampleScore) {
         pSampleScoresInnerEnd = pSampleScoresTrueEnd;
         pSampleScoresExit = pSampleScoresTrueEnd;
         goto one_last_loop;
      }
   }
};
#endif // EXPAND_BINARY_LOGITS

template<size_t compilerBitPack>
class ApplyTermUpdateTrainingInternal<k_regression, compilerBitPack> final {
public:

   ApplyTermUpdateTrainingInternal() = delete; // this is a static class.  Do not construct

   static void Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm
   ) {
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const size_t runtimeBitPack = pTerm->GetBitPack();
      DataSetBoosting * const pTrainingSet = pBoosterCore->GetTrainingSet();

      const size_t cSamples = pTrainingSet->GetCountSamples();
      EBM_ASSERT(1 <= cSamples);
      EBM_ASSERT(1 <= pTerm->GetCountSignificantDimensions());

      const size_t cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, runtimeBitPack);
      EBM_ASSERT(1 <= cItemsPerBitPack);
      EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
      const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPack);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);

      const FloatFast * const aUpdateScores = pBoosterShell->GetTermUpdate()->GetScoresPointer();
      EBM_ASSERT(nullptr != aUpdateScores);

      // No hessians for regression
      FloatFast * pGradient = pTrainingSet->GetGradientsAndHessiansPointer();
      const StorageDataType * pInputData = pTrainingSet->GetInputDataPointer(pTerm);

      // this shouldn't overflow since we're accessing existing memory
      const FloatFast * const pGradientTrueEnd = pGradient + cSamples;
      const FloatFast * pGradientExit = pGradientTrueEnd;
      const FloatFast * pGradientInnerEnd = pGradientTrueEnd;
      if(cSamples <= cItemsPerBitPack) {
         goto one_last_loop;
      }
      pGradientExit = pGradientTrueEnd - ((cSamples - 1) % cItemsPerBitPack + 1);
      EBM_ASSERT(pGradient < pGradientExit);
      EBM_ASSERT(pGradientExit < pGradientTrueEnd);

      do {
         pGradientInnerEnd = pGradient + cItemsPerBitPack;
         // jumping back into this loop and changing pSampleScoresInnerEnd to a dynamic value that isn't compile time determinable causes this 
         // function to NOT be optimized for templated cItemsPerBitPack, but that's ok since avoiding one unpredictable branch here is negligible
      one_last_loop:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iTensorBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            const size_t iTensorBin = maskBits & iTensorBinCombined;

            const FloatFast updateScore = aUpdateScores[iTensorBin];
            // this will apply a small fix to our existing TrainingSampleScores, either positive or negative, whichever is needed
            const FloatFast gradient = EbmStats::ComputeGradientRegressionMSEFromOriginalGradient(*pGradient, updateScore);

            *pGradient = gradient;
            ++pGradient;

            iTensorBinCombined >>= cBitsPerItemMax;
         } while(pGradientInnerEnd != pGradient);
      } while(pGradientExit != pGradient);

      // first time through?
      if(pGradientTrueEnd != pGradient) {
         pGradientInnerEnd = pGradientTrueEnd;
         pGradientExit = pGradientTrueEnd;
         goto one_last_loop;
      }
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class ApplyTermUpdateTrainingNormalTarget final {
public:

   ApplyTermUpdateTrainingNormalTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         ApplyTermUpdateTrainingInternal<compilerLearningTypeOrCountTargetClassesPossible, k_cItemsPerBitPackDynamic>::Func(
            pBoosterShell,
            pTerm
         );
      } else {
         ApplyTermUpdateTrainingNormalTarget<
            compilerLearningTypeOrCountTargetClassesPossible + 1
         >::Func(
            pBoosterShell,
            pTerm
         );
      }
   }
};

template<>
class ApplyTermUpdateTrainingNormalTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   ApplyTermUpdateTrainingNormalTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pBoosterShell->GetBoosterCore()->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pBoosterShell->GetBoosterCore()->GetRuntimeLearningTypeOrCountTargetClasses());

      ApplyTermUpdateTrainingInternal<k_dynamicClassification, k_cItemsPerBitPackDynamic>::Func(
         pBoosterShell,
         pTerm
      );
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerBitPack>
class ApplyTermUpdateTrainingSIMDPacking final {
public:

   ApplyTermUpdateTrainingSIMDPacking() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm
   ) {
      const size_t runtimeBitPack = pTerm->GetBitPack();

      EBM_ASSERT(1 <= runtimeBitPack);
      EBM_ASSERT(runtimeBitPack <= k_cBitsForStorageType);
      static_assert(compilerBitPack <= k_cBitsForStorageType, "We can't have this many items in a data pack.");
      if(compilerBitPack == runtimeBitPack) {
         ApplyTermUpdateTrainingInternal<compilerLearningTypeOrCountTargetClasses, compilerBitPack>::Func(
            pBoosterShell,
            pTerm
         );
      } else {
         ApplyTermUpdateTrainingSIMDPacking<
            compilerLearningTypeOrCountTargetClasses,
            GetNextCountItemsBitPacked(compilerBitPack)
         >::Func(
            pBoosterShell,
            pTerm
         );
      }
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class ApplyTermUpdateTrainingSIMDPacking<compilerLearningTypeOrCountTargetClasses, k_cItemsPerBitPackDynamic> final {
public:

   ApplyTermUpdateTrainingSIMDPacking() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm
   ) {
      EBM_ASSERT(1 <= pTerm->GetBitPack());
      EBM_ASSERT(pTerm->GetBitPack() <= static_cast<ptrdiff_t>(k_cBitsForStorageType));
      ApplyTermUpdateTrainingInternal<compilerLearningTypeOrCountTargetClasses, k_cItemsPerBitPackDynamic>::Func(
         pBoosterShell,
         pTerm
      );
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class ApplyTermUpdateTrainingSIMDTarget final {
public:

   ApplyTermUpdateTrainingSIMDTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         ApplyTermUpdateTrainingSIMDPacking<
            compilerLearningTypeOrCountTargetClassesPossible,
            k_cItemsPerBitPackMax
         >::Func(
            pBoosterShell,
            pTerm
         );
      } else {
         ApplyTermUpdateTrainingSIMDTarget<
            compilerLearningTypeOrCountTargetClassesPossible + 1
         >::Func(
            pBoosterShell,
            pTerm
         );
      }
   }
};

template<>
class ApplyTermUpdateTrainingSIMDTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   ApplyTermUpdateTrainingSIMDTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pBoosterShell->GetBoosterCore()->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pBoosterShell->GetBoosterCore()->GetRuntimeLearningTypeOrCountTargetClasses());

      ApplyTermUpdateTrainingSIMDPacking<k_dynamicClassification, k_cItemsPerBitPackMax>::Func(
         pBoosterShell,
         pTerm
      );
   }
};

extern void ApplyTermUpdateTraining(
   BoosterShell * const pBoosterShell,
   const Term * const pTerm
) {
   LOG_0(TraceLevelVerbose, "Entered ApplyTermUpdateTraining");

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();

   if(0 == pTerm->GetCountSignificantDimensions()) {
      if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
         ApplyTermUpdateTrainingZeroFeaturesTarget<2>::Func(pBoosterShell);
      } else {
         EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
         ApplyTermUpdateTrainingZeroFeatures<k_regression>::Func(pBoosterShell);
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
            ApplyTermUpdateTrainingSIMDTarget<2>::Func(pBoosterShell, pTerm);
         } else {
            EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
            ApplyTermUpdateTrainingSIMDPacking<k_regression, k_cItemsPerBitPackMax>::Func(
               pBoosterShell,
               pTerm
            );
         }
      } else {
         // there isn't much benefit in eliminating the loop that unpacks a data unit unless we're also unpacking that to SIMD code
         // Our default packing structure is to bin continuous values to 256 values, and we have 64 bit packing structures, so we usually
         // have more than 8 values per memory fetch.  Eliminating the inner loop for multiclass is valuable since we can have low numbers like 3 class,
         // 4 class, etc, but by the time we get to 8 loops with exp inside and a lot of other instructures we should worry that our code expansion
         // will exceed the L1 instruction cache size.  With SIMD we do 8 times the work in the same number of instructions so these are lesser issues

         if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
            ApplyTermUpdateTrainingNormalTarget<2>::Func(pBoosterShell, pTerm);
         } else {
            EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
            ApplyTermUpdateTrainingInternal<k_regression, k_cItemsPerBitPackDynamic>::Func(
               pBoosterShell,
               pTerm
            );
         }
      }
   }

   LOG_0(TraceLevelVerbose, "Exited ApplyTermUpdateTraining");
}

} // DEFINED_ZONE_NAME
