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
// Term.hpp depends on Feature.h
#include "Term.hpp"
// dataset depends on features
#include "DataSetBoosting.hpp"

#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// C++ does not allow partial function specialization, so we need to use these cumbersome static class functions to do partial function specialization

template<ptrdiff_t cCompilerClasses>
class ApplyTermUpdateValidationZeroFeatures final {
public:

   ApplyTermUpdateValidationZeroFeatures() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(BoosterShell * const pBoosterShell) {
      static_assert(IsClassification(cCompilerClasses), "must be classification");
      static_assert(!IsBinaryClassification(cCompilerClasses), "must be multiclass");

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const FloatFast * const aUpdateScores = pBoosterShell->GetTermUpdate()->GetTensorScoresPointer();
      EBM_ASSERT(nullptr != aUpdateScores);

      const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();
      DataSetBoosting * const pValidationSet = pBoosterCore->GetValidationSet();
      const FloatFast * pWeight = pBoosterCore->GetValidationWeights();
#ifndef NDEBUG
      FloatFast weightTotalDebug = 0;
#endif // NDEBUG

      const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, cRuntimeClasses);
      const size_t cScores = GetCountScores(cClasses);
      const size_t cSamples = pValidationSet->GetCountSamples();
      EBM_ASSERT(0 < cSamples);

      FloatFast sumLogLoss = 0;
      const StorageDataType * pTargetData = pValidationSet->GetTargetDataPointer();
      FloatFast * pSampleScore = pValidationSet->GetSampleScores();
      const FloatFast * const pSampleScoresEnd = pSampleScore + cSamples * cScores;
      do {
         size_t targetData = static_cast<size_t>(*pTargetData);
         ++pTargetData;

         const FloatFast * pUpdateScore = aUpdateScores;
         FloatFast itemExp = 0;
         FloatFast sumExp = 0;
         size_t iScore = 0;
         do {
            // TODO : because there is only one bin for a zero feature feature group, we could move these values to the stack where the
            // compiler could reason about their visibility and optimize small arrays into registers
            const FloatFast updateScore = *pUpdateScore;
            ++pUpdateScore;
            // this will apply a small fix to our existing ValidationSampleScores, either positive or negative, whichever is needed
            const FloatFast sampleScore = *pSampleScore + updateScore;
            *pSampleScore = sampleScore;
            ++pSampleScore;
            const FloatFast oneExp = ExpForLogLossMulticlass<false>(sampleScore);
            itemExp = iScore == targetData ? oneExp : itemExp;
            sumExp += oneExp;
            ++iScore;
         } while(iScore < cScores);
         const FloatFast sampleLogLoss = EbmStats::ComputeSingleSampleLogLossMulticlass(
            sumExp,
            itemExp
         );

         EBM_ASSERT(std::isnan(sampleLogLoss) || -k_epsilonLogLoss <= sampleLogLoss);

         FloatFast weight = 1;
         if(nullptr != pWeight) {
            // TODO: template this check away
            weight = *pWeight;
            ++pWeight;
#ifndef NDEBUG
            weightTotalDebug += weight;
#endif // NDEBUG
         }
         sumLogLoss += sampleLogLoss * weight;
      } while(pSampleScoresEnd != pSampleScore);

      EBM_ASSERT(nullptr == pWeight || pBoosterCore->GetValidationWeightTotal() * 0.999 <= weightTotalDebug &&
         weightTotalDebug <= 1.001 * pBoosterCore->GetValidationWeightTotal());
      EBM_ASSERT(nullptr != pWeight || static_cast<FloatBig>(cSamples) == pBoosterCore->GetValidationWeightTotal());

      return static_cast<double>(sumLogLoss);
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<>
class ApplyTermUpdateValidationZeroFeatures<2> final {
public:

   ApplyTermUpdateValidationZeroFeatures() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(BoosterShell * const pBoosterShell) {
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const FloatFast * const aUpdateScores = pBoosterShell->GetTermUpdate()->GetTensorScoresPointer();
      EBM_ASSERT(nullptr != aUpdateScores);

      DataSetBoosting * const pValidationSet = pBoosterCore->GetValidationSet();
      const size_t cSamples = pValidationSet->GetCountSamples();
      EBM_ASSERT(0 < cSamples);

      const FloatFast * pWeight = pBoosterCore->GetValidationWeights();
#ifndef NDEBUG
      FloatFast weightTotalDebug = 0;
#endif // NDEBUG

      FloatFast sumLogLoss = 0;
      const StorageDataType * pTargetData = pValidationSet->GetTargetDataPointer();
      FloatFast * pSampleScore = pValidationSet->GetSampleScores();
      const FloatFast * const pSampleScoresEnd = pSampleScore + cSamples;
      const FloatFast updateScore = aUpdateScores[0];
      do {
         size_t targetData = static_cast<size_t>(*pTargetData);
         ++pTargetData;
         // this will apply a small fix to our existing ValidationSampleScores, either positive or negative, whichever is needed
         const FloatFast sampleScore = *pSampleScore + updateScore;
         *pSampleScore = sampleScore;
         ++pSampleScore;
         const FloatFast sampleLogLoss = EbmStats::ComputeSingleSampleLogLossBinaryClassification(sampleScore, targetData);
         EBM_ASSERT(std::isnan(sampleLogLoss) || 0 <= sampleLogLoss);

         FloatFast weight = 1;
         if(nullptr != pWeight) {
            // TODO: template this check away
            weight = *pWeight;
            ++pWeight;
#ifndef NDEBUG
            weightTotalDebug += weight;
#endif // NDEBUG
         }
         sumLogLoss += sampleLogLoss * weight;
      } while(pSampleScoresEnd != pSampleScore);

      EBM_ASSERT(nullptr == pWeight || pBoosterCore->GetValidationWeightTotal() * 0.999 <= weightTotalDebug &&
         weightTotalDebug <= 1.001 * pBoosterCore->GetValidationWeightTotal());
      EBM_ASSERT(nullptr != pWeight || static_cast<FloatBig>(cSamples) == pBoosterCore->GetValidationWeightTotal());

      return static_cast<double>(sumLogLoss);
   }
};
#endif // EXPAND_BINARY_LOGITS

template<>
class ApplyTermUpdateValidationZeroFeatures<k_regression> final {
public:

   ApplyTermUpdateValidationZeroFeatures() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(BoosterShell * const pBoosterShell) {
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const FloatFast * const aUpdateScores = pBoosterShell->GetTermUpdate()->GetTensorScoresPointer();
      EBM_ASSERT(nullptr != aUpdateScores);

      DataSetBoosting * const pValidationSet = pBoosterCore->GetValidationSet();
      const size_t cSamples = pValidationSet->GetCountSamples();
      EBM_ASSERT(0 < cSamples);

      const FloatFast * pWeight = pBoosterCore->GetValidationWeights();
#ifndef NDEBUG
      FloatFast weightTotalDebug = 0;
#endif // NDEBUG

      FloatFast sumSquareError = 0;
      // no hessians for regression
      FloatFast * pGradient = pValidationSet->GetGradientsAndHessiansPointer();
      const FloatFast * const pGradientsEnd = pGradient + cSamples;
      const FloatFast updateScore = aUpdateScores[0];
      do {
         // this will apply a small fix to our existing ValidationSampleScores, either positive or negative, whichever is needed
         const FloatFast gradient = EbmStats::ComputeGradientRegressionMSEFromOriginalGradient(*pGradient, updateScore);
         const FloatFast singleSampleSquaredError = EbmStats::ComputeSingleSampleSquaredErrorRegressionFromGradient(gradient);
         EBM_ASSERT(std::isnan(singleSampleSquaredError) || 0 <= singleSampleSquaredError);

         FloatFast weight = 1;
         if(nullptr != pWeight) {
            // TODO: template this check away
            weight = *pWeight;
            ++pWeight;
#ifndef NDEBUG
            weightTotalDebug += weight;
#endif // NDEBUG
         }
         sumSquareError += singleSampleSquaredError * weight;
         *pGradient = gradient;
         ++pGradient;
      } while(pGradientsEnd != pGradient);

      EBM_ASSERT(nullptr == pWeight || pBoosterCore->GetValidationWeightTotal() * 0.999 <= weightTotalDebug &&
         weightTotalDebug <= 1.001 * pBoosterCore->GetValidationWeightTotal());
      EBM_ASSERT(nullptr != pWeight || static_cast<FloatBig>(cSamples) == pBoosterCore->GetValidationWeightTotal());

      return static_cast<double>(sumSquareError);
   }
};

template<ptrdiff_t cPossibleClasses>
class ApplyTermUpdateValidationZeroFeaturesTarget final {
public:

   ApplyTermUpdateValidationZeroFeaturesTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(BoosterShell * const pBoosterShell) {
      static_assert(IsClassification(cPossibleClasses), "cPossibleClasses needs to be a classification");
      static_assert(cPossibleClasses <= k_cCompilerClassesMax, "We can't have this many items in a data pack.");

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();
      EBM_ASSERT(IsClassification(cRuntimeClasses));
      EBM_ASSERT(cRuntimeClasses <= k_cCompilerClassesMax);

      if(cPossibleClasses == cRuntimeClasses) {
         return ApplyTermUpdateValidationZeroFeatures<cPossibleClasses>::Func(
            pBoosterShell
         );
      } else {
         return ApplyTermUpdateValidationZeroFeaturesTarget<
            cPossibleClasses + 1
         >::Func(
            pBoosterShell
         );
      }
   }
};

template<>
class ApplyTermUpdateValidationZeroFeaturesTarget<k_cCompilerClassesMax + 1> final {
public:

   ApplyTermUpdateValidationZeroFeaturesTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(BoosterShell * const pBoosterShell) {
      static_assert(IsClassification(k_cCompilerClassesMax), "k_cCompilerClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pBoosterShell->GetBoosterCore()->GetCountClasses()));
      EBM_ASSERT(k_cCompilerClassesMax < pBoosterShell->GetBoosterCore()->GetCountClasses());

      return ApplyTermUpdateValidationZeroFeatures<k_dynamicClassification>::Func(pBoosterShell);
   }
};

template<ptrdiff_t cCompilerClasses, size_t compilerBitPack>
class ApplyTermUpdateValidationInternal final {
public:

   ApplyTermUpdateValidationInternal() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      BoosterShell * const pBoosterShell,
      const size_t iTerm
   ) {
      static_assert(IsClassification(cCompilerClasses), "must be classification");
      static_assert(!IsBinaryClassification(cCompilerClasses), "must be multiclass");

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
      const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];

      const FloatFast * const aUpdateScores = pBoosterShell->GetTermUpdate()->GetTensorScoresPointer();
      EBM_ASSERT(nullptr != aUpdateScores);

      const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();
      const size_t runtimeBitPack = pTerm->GetBitPack();
      DataSetBoosting * const pValidationSet = pBoosterCore->GetValidationSet();
      const FloatFast * pWeight = pBoosterCore->GetValidationWeights();
#ifndef NDEBUG
      FloatFast weightTotalDebug = 0;
#endif // NDEBUG

      const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, cRuntimeClasses);
      const size_t cScores = GetCountScores(cClasses);
      const size_t cSamples = pValidationSet->GetCountSamples();
      EBM_ASSERT(1 <= cSamples);
      EBM_ASSERT(1 <= pTerm->GetCountRealDimensions());

      const size_t cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, runtimeBitPack);
      EBM_ASSERT(1 <= cItemsPerBitPack);
      EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
      const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPack);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      const StorageDataType maskBits = (~StorageDataType { 0 }) >> (k_cBitsForStorageType - cBitsPerItemMax);

      FloatFast sumLogLoss = 0;
      const StorageDataType * pInputData = pValidationSet->GetInputDataPointer(iTerm);
      const StorageDataType * pTargetData = pValidationSet->GetTargetDataPointer();
      FloatFast * pSampleScore = pValidationSet->GetSampleScores();

      // this shouldn't overflow since we're accessing existing memory
      const FloatFast * const pSampleScoresTrueEnd = pSampleScore + cSamples * cScores;
      const FloatFast * pSampleScoresExit = pSampleScoresTrueEnd;
      const FloatFast * pSampleScoresInnerEnd = pSampleScoresTrueEnd;
      if(cSamples <= cItemsPerBitPack) {
         goto one_last_loop;
      }
      pSampleScoresExit = pSampleScoresTrueEnd - ((cSamples - 1) % cItemsPerBitPack + 1) * cScores;
      EBM_ASSERT(pSampleScore < pSampleScoresExit);
      EBM_ASSERT(pSampleScoresExit < pSampleScoresTrueEnd);

      do {
         pSampleScoresInnerEnd = pSampleScore + cItemsPerBitPack * cScores;
         // jumping back into this loop and changing pSampleScoresInnerEnd to a dynamic value that isn't compile time determinable causes this 
         // function to NOT be optimized for templated cItemsPerBitPack, but that's ok since avoiding one unpredictable branch here is negligible
      one_last_loop:;
         // we store the already multiplied dimensional value in *pInputData
         StorageDataType iTensorBinCombined = *pInputData;
         ++pInputData;
         do {
            size_t targetData = static_cast<size_t>(*pTargetData);
            ++pTargetData;

            const size_t iTensorBin = static_cast<size_t>(maskBits & iTensorBinCombined);
            const FloatFast * pUpdateScore = &aUpdateScores[iTensorBin * cScores];
            FloatFast itemExp = 0;
            FloatFast sumExp = 0;
            size_t iScore = 0;
            do {
               const FloatFast updateScore = *pUpdateScore;
               ++pUpdateScore;
               // this will apply a small fix to our existing ValidationSampleScores, either positive or negative, whichever is needed
               const FloatFast sampleScore = *pSampleScore + updateScore;
               *pSampleScore = sampleScore;
               ++pSampleScore;
               const FloatFast oneExp = ExpForLogLossMulticlass<false>(sampleScore);
               itemExp = iScore == targetData ? oneExp : itemExp;
               sumExp += oneExp;
               ++iScore;
            } while(iScore < cScores);
            const FloatFast sampleLogLoss = EbmStats::ComputeSingleSampleLogLossMulticlass(
               sumExp,
               itemExp
            );

            EBM_ASSERT(std::isnan(sampleLogLoss) || -k_epsilonLogLoss <= sampleLogLoss);

            FloatFast weight = 1;
            if(nullptr != pWeight) {
               // TODO: template this check away
               weight = *pWeight;
               ++pWeight;
#ifndef NDEBUG
               weightTotalDebug += weight;
#endif // NDEBUG
            }
            sumLogLoss += sampleLogLoss * weight;
            iTensorBinCombined >>= cBitsPerItemMax;
         } while(pSampleScoresInnerEnd != pSampleScore);
      } while(pSampleScoresExit != pSampleScore);

      // first time through?
      if(pSampleScoresTrueEnd != pSampleScore) {
         pSampleScoresInnerEnd = pSampleScoresTrueEnd;
         pSampleScoresExit = pSampleScoresTrueEnd;
         goto one_last_loop;
      }

      EBM_ASSERT(nullptr == pWeight || pBoosterCore->GetValidationWeightTotal() * 0.999 <= weightTotalDebug &&
         weightTotalDebug <= 1.001 * pBoosterCore->GetValidationWeightTotal());
      EBM_ASSERT(nullptr != pWeight || static_cast<FloatBig>(cSamples) == pBoosterCore->GetValidationWeightTotal());

      return static_cast<double>(sumLogLoss);
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<size_t compilerBitPack>
class ApplyTermUpdateValidationInternal<2, compilerBitPack> final {
public:

   ApplyTermUpdateValidationInternal() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      BoosterShell * const pBoosterShell,
      const size_t iTerm
   ) {
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
      const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];

      const FloatFast * const aUpdateScores = pBoosterShell->GetTermUpdate()->GetTensorScoresPointer();
      EBM_ASSERT(nullptr != aUpdateScores);

      const size_t runtimeBitPack = pTerm->GetBitPack();
      DataSetBoosting * const pValidationSet = pBoosterCore->GetValidationSet();
      const FloatFast * pWeight = pBoosterCore->GetValidationWeights();
#ifndef NDEBUG
      FloatFast weightTotalDebug = 0;
#endif // NDEBUG

      const size_t cSamples = pValidationSet->GetCountSamples();
      EBM_ASSERT(1 <= cSamples);
      EBM_ASSERT(1 <= pTerm->GetCountRealDimensions());

      const size_t cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, runtimeBitPack);
      EBM_ASSERT(1 <= cItemsPerBitPack);
      EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
      const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPack);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      const StorageDataType maskBits = (~StorageDataType { 0 }) >> (k_cBitsForStorageType - cBitsPerItemMax);

      FloatFast sumLogLoss = 0;
      const StorageDataType * pInputData = pValidationSet->GetInputDataPointer(iTerm);
      const StorageDataType * pTargetData = pValidationSet->GetTargetDataPointer();
      FloatFast * pSampleScore = pValidationSet->GetSampleScores();

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
         StorageDataType iTensorBinCombined = *pInputData;
         ++pInputData;
         do {
            size_t targetData = static_cast<size_t>(*pTargetData);
            ++pTargetData;

            const size_t iTensorBin = static_cast<size_t>(maskBits & iTensorBinCombined);

            const FloatFast updateScore = aUpdateScores[iTensorBin];
            // this will apply a small fix to our existing ValidationSampleScores, either positive or negative, whichever is needed
            const FloatFast sampleScore = *pSampleScore + updateScore;
            *pSampleScore = sampleScore;
            ++pSampleScore;
            const FloatFast sampleLogLoss = EbmStats::ComputeSingleSampleLogLossBinaryClassification(sampleScore, targetData);

            EBM_ASSERT(std::isnan(sampleLogLoss) || 0 <= sampleLogLoss);

            FloatFast weight = 1;
            if(nullptr != pWeight) {
               // TODO: template this check away
               weight = *pWeight;
               ++pWeight;
#ifndef NDEBUG
               weightTotalDebug += weight;
#endif // NDEBUG
            }
            sumLogLoss += sampleLogLoss * weight;

            iTensorBinCombined >>= cBitsPerItemMax;
         } while(pSampleScoresInnerEnd != pSampleScore);
      } while(pSampleScoresExit != pSampleScore);

      // first time through?
      if(pSampleScoresTrueEnd != pSampleScore) {
         pSampleScoresInnerEnd = pSampleScoresTrueEnd;
         pSampleScoresExit = pSampleScoresTrueEnd;
         goto one_last_loop;
      }

      EBM_ASSERT(nullptr == pWeight || pBoosterCore->GetValidationWeightTotal() * 0.999 <= weightTotalDebug &&
         weightTotalDebug <= 1.001 * pBoosterCore->GetValidationWeightTotal());
      EBM_ASSERT(nullptr != pWeight || static_cast<FloatBig>(cSamples) == pBoosterCore->GetValidationWeightTotal());

      return static_cast<double>(sumLogLoss);
   }
};
#endif // EXPAND_BINARY_LOGITS

template<size_t compilerBitPack>
class ApplyTermUpdateValidationInternal<k_regression, compilerBitPack> final {
public:

   ApplyTermUpdateValidationInternal() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      BoosterShell * const pBoosterShell,
      const size_t iTerm
   ) {
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
      const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];

      const FloatFast * const aUpdateScores = pBoosterShell->GetTermUpdate()->GetTensorScoresPointer();
      EBM_ASSERT(nullptr != aUpdateScores);

      const size_t runtimeBitPack = pTerm->GetBitPack();
      DataSetBoosting * const pValidationSet = pBoosterCore->GetValidationSet();
      const FloatFast * pWeight = pBoosterCore->GetValidationWeights();
#ifndef NDEBUG
      FloatFast weightTotalDebug = 0;
#endif // NDEBUG

      const size_t cSamples = pValidationSet->GetCountSamples();
      EBM_ASSERT(1 <= cSamples);
      EBM_ASSERT(1 <= pTerm->GetCountRealDimensions());

      const size_t cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, runtimeBitPack);
      EBM_ASSERT(1 <= cItemsPerBitPack);
      EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
      const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPack);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      const StorageDataType maskBits = (~StorageDataType { 0 }) >> (k_cBitsForStorageType - cBitsPerItemMax);

      FloatFast sumSquareError = 0;
      // no hessians for regression
      FloatFast * pGradient = pValidationSet->GetGradientsAndHessiansPointer();
      const StorageDataType * pInputData = pValidationSet->GetInputDataPointer(iTerm);

      // this shouldn't overflow since we're accessing existing memory
      const FloatFast * const pGradientsTrueEnd = pGradient + cSamples;
      const FloatFast * pGradientsExit = pGradientsTrueEnd;
      const FloatFast * pGradientsInnerEnd = pGradientsTrueEnd;
      if(cSamples <= cItemsPerBitPack) {
         goto one_last_loop;
      }
      pGradientsExit = pGradientsTrueEnd - ((cSamples - 1) % cItemsPerBitPack + 1);
      EBM_ASSERT(pGradient < pGradientsExit);
      EBM_ASSERT(pGradientsExit < pGradientsTrueEnd);

      do {
         pGradientsInnerEnd = pGradient + cItemsPerBitPack;
         // jumping back into this loop and changing pSampleScoresInnerEnd to a dynamic value that isn't compile time determinable causes this 
         // function to NOT be optimized for templated cItemsPerBitPack, but that's ok since avoiding one unpredictable branch here is negligible
      one_last_loop:;
         // we store the already multiplied dimensional value in *pInputData
         StorageDataType iTensorBinCombined = *pInputData;
         ++pInputData;
         do {
            const size_t iTensorBin = static_cast<size_t>(maskBits & iTensorBinCombined);

            const FloatFast updateScore = aUpdateScores[iTensorBin];
            // this will apply a small fix to our existing ValidationSampleScores, either positive or negative, whichever is needed
            const FloatFast gradient = EbmStats::ComputeGradientRegressionMSEFromOriginalGradient(*pGradient, updateScore);
            const FloatFast sampleSquaredError = EbmStats::ComputeSingleSampleSquaredErrorRegressionFromGradient(gradient);
            EBM_ASSERT(std::isnan(sampleSquaredError) || 0 <= sampleSquaredError);

            FloatFast weight = 1;
            if(nullptr != pWeight) {
               // TODO: template this check away
               weight = *pWeight;
               ++pWeight;
#ifndef NDEBUG
               weightTotalDebug += weight;
#endif // NDEBUG
            }
            sumSquareError += sampleSquaredError * weight;
            *pGradient = gradient;
            ++pGradient;

            iTensorBinCombined >>= cBitsPerItemMax;
         } while(pGradientsInnerEnd != pGradient);
      } while(pGradientsExit != pGradient);

      // first time through?
      if(pGradientsTrueEnd != pGradient) {
         pGradientsInnerEnd = pGradientsTrueEnd;
         pGradientsExit = pGradientsTrueEnd;
         goto one_last_loop;
      }

      EBM_ASSERT(nullptr == pWeight || pBoosterCore->GetValidationWeightTotal() * 0.999 <= weightTotalDebug &&
         weightTotalDebug <= 1.001 * pBoosterCore->GetValidationWeightTotal());
      EBM_ASSERT(nullptr != pWeight || static_cast<FloatBig>(cSamples) == pBoosterCore->GetValidationWeightTotal());

      return static_cast<double>(sumSquareError);
   }
};

template<ptrdiff_t cPossibleClasses>
class ApplyTermUpdateValidationNormalTarget final {
public:

   ApplyTermUpdateValidationNormalTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      BoosterShell * const pBoosterShell,
      const size_t iTerm
   ) {
      static_assert(IsClassification(cPossibleClasses), "cPossibleClasses needs to be a classification");
      static_assert(cPossibleClasses <= k_cCompilerClassesMax, "We can't have this many items in a data pack.");

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();
      EBM_ASSERT(IsClassification(cRuntimeClasses));
      EBM_ASSERT(cRuntimeClasses <= k_cCompilerClassesMax);

      if(cPossibleClasses == cRuntimeClasses) {
         return ApplyTermUpdateValidationInternal<cPossibleClasses, k_cItemsPerBitPackDynamic>::Func(
            pBoosterShell,
            iTerm
         );
      } else {
         return ApplyTermUpdateValidationNormalTarget<
            cPossibleClasses + 1
         >::Func(
            pBoosterShell,
            iTerm
         );
      }
   }
};

template<>
class ApplyTermUpdateValidationNormalTarget<k_cCompilerClassesMax + 1> final {
public:

   ApplyTermUpdateValidationNormalTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      BoosterShell * const pBoosterShell,
      const size_t iTerm
   ) {
      static_assert(IsClassification(k_cCompilerClassesMax), "k_cCompilerClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pBoosterShell->GetBoosterCore()->GetCountClasses()));
      EBM_ASSERT(k_cCompilerClassesMax < pBoosterShell->GetBoosterCore()->GetCountClasses());

      return ApplyTermUpdateValidationInternal<k_dynamicClassification, k_cItemsPerBitPackDynamic>::Func(
         pBoosterShell,
         iTerm
      );
   }
};

template<ptrdiff_t cCompilerClasses, size_t compilerBitPack>
class ApplyTermUpdateValidationSIMDPacking final {
public:

   ApplyTermUpdateValidationSIMDPacking() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      BoosterShell * const pBoosterShell,
      const size_t iTerm
   ) {
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
      const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];
      const size_t runtimeBitPack = pTerm->GetBitPack();

      EBM_ASSERT(1 <= runtimeBitPack);
      EBM_ASSERT(runtimeBitPack <= k_cBitsForStorageType);
      static_assert(compilerBitPack <= k_cBitsForStorageType, "We can't have this many items in a data pack.");
      if(compilerBitPack == runtimeBitPack) {
         return ApplyTermUpdateValidationInternal<cCompilerClasses, compilerBitPack>::Func(
            pBoosterShell,
            iTerm
         );
      } else {
         return ApplyTermUpdateValidationSIMDPacking<
            cCompilerClasses,
            GetNextCountItemsBitPacked(compilerBitPack)
         >::Func(
            pBoosterShell,
            iTerm
         );
      }
   }
};

template<ptrdiff_t cCompilerClasses>
class ApplyTermUpdateValidationSIMDPacking<cCompilerClasses, k_cItemsPerBitPackDynamic> final {
public:

   ApplyTermUpdateValidationSIMDPacking() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      BoosterShell * const pBoosterShell,
      const size_t iTerm
   ) {
      EBM_ASSERT(iTerm < pBoosterShell->GetBoosterCore()->GetCountTerms());
      EBM_ASSERT(1 <= pBoosterShell->GetBoosterCore()->GetTerms()[iTerm]->GetBitPack());
      EBM_ASSERT(pBoosterShell->GetBoosterCore()->GetTerms()[iTerm]->GetBitPack() <= static_cast<ptrdiff_t>(k_cBitsForStorageType));
      return ApplyTermUpdateValidationInternal<
         cCompilerClasses, 
         k_cItemsPerBitPackDynamic
      >::Func(
         pBoosterShell,
         iTerm
      );
   }
};

template<ptrdiff_t cPossibleClasses>
class ApplyTermUpdateValidationSIMDTarget final {
public:

   ApplyTermUpdateValidationSIMDTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      BoosterShell * const pBoosterShell,
      const size_t iTerm
   ) {
      static_assert(IsClassification(cPossibleClasses), "cPossibleClasses needs to be a classification");
      static_assert(cPossibleClasses <= k_cCompilerClassesMax, "We can't have this many items in a data pack.");

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();
      EBM_ASSERT(IsClassification(cRuntimeClasses));
      EBM_ASSERT(cRuntimeClasses <= k_cCompilerClassesMax);

      if(cPossibleClasses == cRuntimeClasses) {
         return ApplyTermUpdateValidationSIMDPacking<
            cPossibleClasses,
            k_cItemsPerBitPackMax
         >::Func(
            pBoosterShell,
            iTerm
         );
      } else {
         return ApplyTermUpdateValidationSIMDTarget<
            cPossibleClasses + 1
         >::Func(
            pBoosterShell,
            iTerm
         );
      }
   }
};

template<>
class ApplyTermUpdateValidationSIMDTarget<k_cCompilerClassesMax + 1> final {
public:

   ApplyTermUpdateValidationSIMDTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      BoosterShell * const pBoosterShell,
      const size_t iTerm
   ) {
      static_assert(IsClassification(k_cCompilerClassesMax), "k_cCompilerClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pBoosterShell->GetBoosterCore()->GetCountClasses()));
      EBM_ASSERT(k_cCompilerClassesMax < pBoosterShell->GetBoosterCore()->GetCountClasses());

      return ApplyTermUpdateValidationSIMDPacking<
         k_dynamicClassification,
         k_cItemsPerBitPackMax
      >::Func(
         pBoosterShell,
         iTerm
      );
   }
};

extern double ApplyTermUpdateValidation(BoosterShell * const pBoosterShell, const size_t iTerm) {
   LOG_0(Trace_Verbose, "Entered ApplyTermUpdateValidation");

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];

   const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();

   double ret;
   if(0 == pTerm->GetCountRealDimensions()) {
      if(IsClassification(cRuntimeClasses)) {
         ret = ApplyTermUpdateValidationZeroFeaturesTarget<2>::Func(pBoosterShell);
      } else {
         EBM_ASSERT(IsRegression(cRuntimeClasses));
         ret = ApplyTermUpdateValidationZeroFeatures<k_regression>::Func(pBoosterShell);
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

         if(IsClassification(cRuntimeClasses)) {
            ret = ApplyTermUpdateValidationSIMDTarget<2>::Func(
               pBoosterShell,
               iTerm
            );
         } else {
            EBM_ASSERT(IsRegression(cRuntimeClasses));
            ret = ApplyTermUpdateValidationSIMDPacking<k_regression, k_cItemsPerBitPackMax>::Func(
               pBoosterShell,
               iTerm
            );
         }
      } else {
         // there isn't much benefit in eliminating the loop that unpacks a data unit unless we're also unpacking that to SIMD code
         // Our default packing structure is to bin continuous values to 256 values, and we have 64 bit packing structures, so we usually
         // have more than 8 values per memory fetch.  Eliminating the inner loop for multiclass is valuable since we can have low numbers like 3 class,
         // 4 class, etc, but by the time we get to 8 loops with exp inside and a lot of other instructures we should worry that our code expansion
         // will exceed the L1 instruction cache size.  With SIMD we do 8 times the work in the same number of instructions so these are lesser issues

         if(IsClassification(cRuntimeClasses)) {
            ret = ApplyTermUpdateValidationNormalTarget<2>::Func(
               pBoosterShell,
               iTerm
            );
         } else {
            EBM_ASSERT(IsRegression(cRuntimeClasses));
            ret = ApplyTermUpdateValidationInternal<k_regression, k_cItemsPerBitPackDynamic>::Func(
               pBoosterShell,
               iTerm
            );
         }
      }
   }

   EBM_ASSERT(std::isnan(ret) || -k_epsilonLogLoss <= ret);
   if(UNLIKELY(/* NaN */ !LIKELY(0.0 <= ret))) {
      // this also checks for NaN since NaN < anything is FALSE

      // if we've overflowed to a NaN, then conver it to +inf since +inf is our general overflow marker
      // if we've gotten a value that's slightly negative, which can happen for numeracy reasons, clip to zero

      ret = std::isnan(ret) ? std::numeric_limits<double>::infinity() : double { 0 };
   }
   EBM_ASSERT(!std::isnan(ret));
   EBM_ASSERT(0.0 <= ret);

   LOG_0(Trace_Verbose, "Exited ApplyTermUpdateValidation");

   return ret;
}

} // DEFINED_ZONE_NAME
