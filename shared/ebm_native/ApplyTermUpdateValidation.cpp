// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT
#include "zones.h"

#include "approximate_math.hpp"
#include "ebm_stats.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// C++ does not allow partial function specialization, so we need to use these cumbersome static class functions to do partial function specialization

template<ptrdiff_t cCompilerClasses>
class ApplyTermUpdateValidationZeroFeatures final {
public:

   ApplyTermUpdateValidationZeroFeatures() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      const ptrdiff_t cRuntimeClasses,
      const ptrdiff_t runtimeBitPack,
      FloatFast * const aMulticlassMidwayTemp,
      const FloatFast * const aUpdateScores,
      const size_t cSamples,
      const StorageDataType * const aInputData,
      const void * const aTargetData,
      const FloatFast * const aWeight,
      FloatFast * const aSampleScore,
      FloatFast * const aGradientAndHessian
   ) {
      UNUSED(runtimeBitPack);
      UNUSED(aMulticlassMidwayTemp);
      UNUSED(aInputData);
      UNUSED(aGradientAndHessian);

      static_assert(IsClassification(cCompilerClasses), "must be classification");
      static_assert(!IsBinaryClassification(cCompilerClasses), "must be multiclass");

      EBM_ASSERT(nullptr != aUpdateScores);

      const FloatFast * pWeight = aWeight;

      const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, cRuntimeClasses);
      const size_t cScores = GetCountScores(cClasses);
      EBM_ASSERT(0 < cSamples);

      FloatFast sumLogLoss = 0;
      const StorageDataType * pTargetData = reinterpret_cast<const StorageDataType *>(aTargetData);
      FloatFast * pSampleScore = aSampleScore;
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
         }
         sumLogLoss += sampleLogLoss * weight;
      } while(pSampleScoresEnd != pSampleScore);

      return static_cast<double>(sumLogLoss);
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<>
class ApplyTermUpdateValidationZeroFeatures<2> final {
public:

   ApplyTermUpdateValidationZeroFeatures() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      const ptrdiff_t cRuntimeClasses,
      const ptrdiff_t runtimeBitPack,
      FloatFast * const aMulticlassMidwayTemp,
      const FloatFast * const aUpdateScores,
      const size_t cSamples,
      const StorageDataType * const aInputData,
      const void * const aTargetData,
      const FloatFast * const aWeight,
      FloatFast * const aSampleScore,
      FloatFast * const aGradientAndHessian
   ) {
      UNUSED(cRuntimeClasses);
      UNUSED(runtimeBitPack);
      UNUSED(aMulticlassMidwayTemp);
      UNUSED(aInputData);
      UNUSED(aGradientAndHessian);

      EBM_ASSERT(nullptr != aUpdateScores);

      EBM_ASSERT(0 < cSamples);

      const FloatFast * pWeight = aWeight;

      FloatFast sumLogLoss = 0;
      const StorageDataType * pTargetData = reinterpret_cast<const StorageDataType *>(aTargetData);
      FloatFast * pSampleScore = aSampleScore;
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
         }
         sumLogLoss += sampleLogLoss * weight;
      } while(pSampleScoresEnd != pSampleScore);

      return static_cast<double>(sumLogLoss);
   }
};
#endif // EXPAND_BINARY_LOGITS

template<>
class ApplyTermUpdateValidationZeroFeatures<k_regression> final {
public:

   ApplyTermUpdateValidationZeroFeatures() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      const ptrdiff_t cRuntimeClasses,
      const ptrdiff_t runtimeBitPack,
      FloatFast * const aMulticlassMidwayTemp,
      const FloatFast * const aUpdateScores,
      const size_t cSamples,
      const StorageDataType * const aInputData,
      const void * const aTargetData,
      const FloatFast * const aWeight,
      FloatFast * const aSampleScore,
      FloatFast * const aGradientAndHessian
   ) {
      UNUSED(cRuntimeClasses);
      UNUSED(runtimeBitPack);
      UNUSED(aMulticlassMidwayTemp);
      UNUSED(aInputData);
      UNUSED(aTargetData);
      UNUSED(aSampleScore);

      EBM_ASSERT(nullptr != aUpdateScores);
      EBM_ASSERT(0 < cSamples);

      const FloatFast * pWeight = aWeight;

      FloatFast sumSquareError = 0;
      // no hessians for regression
      FloatFast * pGradient = aGradientAndHessian;
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
         }
         sumSquareError += singleSampleSquaredError * weight;
         *pGradient = gradient;
         ++pGradient;
      } while(pGradientsEnd != pGradient);

      return static_cast<double>(sumSquareError);
   }
};

template<ptrdiff_t cPossibleClasses>
class ApplyTermUpdateValidationZeroFeaturesTarget final {
public:

   ApplyTermUpdateValidationZeroFeaturesTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      const ptrdiff_t cRuntimeClasses,
      const ptrdiff_t runtimeBitPack,
      FloatFast * const aMulticlassMidwayTemp,
      const FloatFast * const aUpdateScores,
      const size_t cSamples,
      const StorageDataType * const aInputData,
      const void * const aTargetData,
      const FloatFast * const aWeight,
      FloatFast * const aSampleScore,
      FloatFast * const aGradientAndHessian
   ) {
      static_assert(IsClassification(cPossibleClasses), "cPossibleClasses needs to be a classification");
      static_assert(cPossibleClasses <= k_cCompilerClassesMax, "We can't have this many items in a data pack.");

      EBM_ASSERT(IsClassification(cRuntimeClasses));
      EBM_ASSERT(cRuntimeClasses <= k_cCompilerClassesMax);

      if(cPossibleClasses == cRuntimeClasses) {
         return ApplyTermUpdateValidationZeroFeatures<cPossibleClasses>::Func(
            cRuntimeClasses,
            runtimeBitPack,
            aMulticlassMidwayTemp,
            aUpdateScores,
            cSamples,
            aInputData,
            aTargetData,
            aWeight,
            aSampleScore,
            aGradientAndHessian
         );
      } else {
         return ApplyTermUpdateValidationZeroFeaturesTarget<
            cPossibleClasses + 1
         >::Func(
            cRuntimeClasses,
            runtimeBitPack,
            aMulticlassMidwayTemp,
            aUpdateScores,
            cSamples,
            aInputData,
            aTargetData,
            aWeight,
            aSampleScore,
            aGradientAndHessian
         );
      }
   }
};

template<>
class ApplyTermUpdateValidationZeroFeaturesTarget<k_cCompilerClassesMax + 1> final {
public:

   ApplyTermUpdateValidationZeroFeaturesTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      const ptrdiff_t cRuntimeClasses,
      const ptrdiff_t runtimeBitPack,
      FloatFast * const aMulticlassMidwayTemp,
      const FloatFast * const aUpdateScores,
      const size_t cSamples,
      const StorageDataType * const aInputData,
      const void * const aTargetData,
      const FloatFast * const aWeight,
      FloatFast * const aSampleScore,
      FloatFast * const aGradientAndHessian
   ) {
      static_assert(IsClassification(k_cCompilerClassesMax), "k_cCompilerClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(cRuntimeClasses));
      EBM_ASSERT(k_cCompilerClassesMax < cRuntimeClasses);

      return ApplyTermUpdateValidationZeroFeatures<k_dynamicClassification>::Func(
         cRuntimeClasses,
         runtimeBitPack,
         aMulticlassMidwayTemp,
         aUpdateScores,
         cSamples,
         aInputData,
         aTargetData,
         aWeight,
         aSampleScore,
         aGradientAndHessian
      );
   }
};

template<ptrdiff_t cCompilerClasses, size_t compilerBitPack>
class ApplyTermUpdateValidationInternal final {
public:

   ApplyTermUpdateValidationInternal() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      const ptrdiff_t cRuntimeClasses,
      const ptrdiff_t runtimeBitPack,
      FloatFast * const aMulticlassMidwayTemp,
      const FloatFast * const aUpdateScores,
      const size_t cSamples,
      const StorageDataType * const aInputData,
      const void * const aTargetData,
      const FloatFast * const aWeight,
      FloatFast * const aSampleScore,
      FloatFast * const aGradientAndHessian
   ) {
      UNUSED(aMulticlassMidwayTemp);
      UNUSED(aGradientAndHessian);

      static_assert(IsClassification(cCompilerClasses), "must be classification");
      static_assert(!IsBinaryClassification(cCompilerClasses), "must be multiclass");

      EBM_ASSERT(nullptr != aUpdateScores);

      const FloatFast * pWeight = aWeight;

      const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, cRuntimeClasses);
      const size_t cScores = GetCountScores(cClasses);
      EBM_ASSERT(1 <= cSamples);

      const size_t cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, runtimeBitPack);
      EBM_ASSERT(1 <= cItemsPerBitPack);
      EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
      const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPack);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      const StorageDataType maskBits = (~StorageDataType { 0 }) >> (k_cBitsForStorageType - cBitsPerItemMax);

      FloatFast sumLogLoss = 0;
      const StorageDataType * pInputData = aInputData;
      const StorageDataType * pTargetData = reinterpret_cast<const StorageDataType *>(aTargetData);
      FloatFast * pSampleScore = aSampleScore;

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

      return static_cast<double>(sumLogLoss);
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<size_t compilerBitPack>
class ApplyTermUpdateValidationInternal<2, compilerBitPack> final {
public:

   ApplyTermUpdateValidationInternal() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      const ptrdiff_t cRuntimeClasses,
      const ptrdiff_t runtimeBitPack,
      FloatFast * const aMulticlassMidwayTemp,
      const FloatFast * const aUpdateScores,
      const size_t cSamples,
      const StorageDataType * const aInputData,
      const void * const aTargetData,
      const FloatFast * const aWeight,
      FloatFast * const aSampleScore,
      FloatFast * const aGradientAndHessian
   ) {
      UNUSED(cRuntimeClasses);
      UNUSED(aMulticlassMidwayTemp);
      UNUSED(aGradientAndHessian);

      EBM_ASSERT(nullptr != aUpdateScores);

      const FloatFast * pWeight = aWeight;

      EBM_ASSERT(1 <= cSamples);

      const size_t cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, runtimeBitPack);
      EBM_ASSERT(1 <= cItemsPerBitPack);
      EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
      const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPack);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      const StorageDataType maskBits = (~StorageDataType { 0 }) >> (k_cBitsForStorageType - cBitsPerItemMax);

      FloatFast sumLogLoss = 0;
      const StorageDataType * pInputData = aInputData;
      const StorageDataType * pTargetData = reinterpret_cast<const StorageDataType *>(aTargetData);
      FloatFast * pSampleScore = aSampleScore;

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

      return static_cast<double>(sumLogLoss);
   }
};
#endif // EXPAND_BINARY_LOGITS

template<size_t compilerBitPack>
class ApplyTermUpdateValidationInternal<k_regression, compilerBitPack> final {
public:

   ApplyTermUpdateValidationInternal() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      const ptrdiff_t cRuntimeClasses,
      const ptrdiff_t runtimeBitPack,
      FloatFast * const aMulticlassMidwayTemp,
      const FloatFast * const aUpdateScores,
      const size_t cSamples,
      const StorageDataType * const aInputData,
      const void * const aTargetData,
      const FloatFast * const aWeight,
      FloatFast * const aSampleScore,
      FloatFast * const aGradientAndHessian
   ) {
      UNUSED(cRuntimeClasses);
      UNUSED(aMulticlassMidwayTemp);
      UNUSED(aTargetData);
      UNUSED(aSampleScore);

      EBM_ASSERT(nullptr != aUpdateScores);

      const FloatFast * pWeight = aWeight;

      EBM_ASSERT(1 <= cSamples);

      const size_t cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, runtimeBitPack);
      EBM_ASSERT(1 <= cItemsPerBitPack);
      EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
      const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPack);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      const StorageDataType maskBits = (~StorageDataType { 0 }) >> (k_cBitsForStorageType - cBitsPerItemMax);

      FloatFast sumSquareError = 0;
      // no hessians for regression
      FloatFast * pGradient = aGradientAndHessian;
      const StorageDataType * pInputData = aInputData;

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

      return static_cast<double>(sumSquareError);
   }
};

template<ptrdiff_t cPossibleClasses>
class ApplyTermUpdateValidationNormalTarget final {
public:

   ApplyTermUpdateValidationNormalTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      const ptrdiff_t cRuntimeClasses,
      const ptrdiff_t runtimeBitPack,
      FloatFast * const aMulticlassMidwayTemp,
      const FloatFast * const aUpdateScores,
      const size_t cSamples,
      const StorageDataType * const aInputData,
      const void * const aTargetData,
      const FloatFast * const aWeight,
      FloatFast * const aSampleScore,
      FloatFast * const aGradientAndHessian
   ) {
      static_assert(IsClassification(cPossibleClasses), "cPossibleClasses needs to be a classification");
      static_assert(cPossibleClasses <= k_cCompilerClassesMax, "We can't have this many items in a data pack.");

      EBM_ASSERT(IsClassification(cRuntimeClasses));
      EBM_ASSERT(cRuntimeClasses <= k_cCompilerClassesMax);

      if(cPossibleClasses == cRuntimeClasses) {
         return ApplyTermUpdateValidationInternal<cPossibleClasses, k_cItemsPerBitPackDynamic>::Func(
            cRuntimeClasses,
            runtimeBitPack,
            aMulticlassMidwayTemp,
            aUpdateScores,
            cSamples,
            aInputData,
            aTargetData,
            aWeight,
            aSampleScore,
            aGradientAndHessian
         );
      } else {
         return ApplyTermUpdateValidationNormalTarget<
            cPossibleClasses + 1
         >::Func(
            cRuntimeClasses,
            runtimeBitPack,
            aMulticlassMidwayTemp,
            aUpdateScores,
            cSamples,
            aInputData,
            aTargetData,
            aWeight,
            aSampleScore,
            aGradientAndHessian
         );
      }
   }
};

template<>
class ApplyTermUpdateValidationNormalTarget<k_cCompilerClassesMax + 1> final {
public:

   ApplyTermUpdateValidationNormalTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      const ptrdiff_t cRuntimeClasses,
      const ptrdiff_t runtimeBitPack,
      FloatFast * const aMulticlassMidwayTemp,
      const FloatFast * const aUpdateScores,
      const size_t cSamples,
      const StorageDataType * const aInputData,
      const void * const aTargetData,
      const FloatFast * const aWeight,
      FloatFast * const aSampleScore,
      FloatFast * const aGradientAndHessian
   ) {
      static_assert(IsClassification(k_cCompilerClassesMax), "k_cCompilerClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(cRuntimeClasses));
      EBM_ASSERT(k_cCompilerClassesMax < cRuntimeClasses);

      return ApplyTermUpdateValidationInternal<k_dynamicClassification, k_cItemsPerBitPackDynamic>::Func(
         cRuntimeClasses,
         runtimeBitPack,
         aMulticlassMidwayTemp,
         aUpdateScores,
         cSamples,
         aInputData,
         aTargetData,
         aWeight,
         aSampleScore,
         aGradientAndHessian
      );
   }
};

extern double ApplyTermUpdateValidation(
   const ptrdiff_t cRuntimeClasses,
   const ptrdiff_t runtimeBitPack,
   FloatFast * const aMulticlassMidwayTemp,
   const FloatFast * const aUpdateScores,
   const size_t cSamples,
   const StorageDataType * const aInputData,
   const void * const aTargetData,
   const FloatFast * const aWeight,
   FloatFast * const aSampleScore,
   FloatFast * const aGradientAndHessian
) {
   LOG_0(Trace_Verbose, "Entered ApplyTermUpdateValidation");

   double ret;
   if(k_cItemsPerBitPackNone == runtimeBitPack) {
      if(IsClassification(cRuntimeClasses)) {
         ret = ApplyTermUpdateValidationZeroFeaturesTarget<2>::Func(
            cRuntimeClasses,
            runtimeBitPack,
            aMulticlassMidwayTemp,
            aUpdateScores,
            cSamples,
            aInputData,
            aTargetData,
            aWeight,
            aSampleScore,
            aGradientAndHessian
         );
      } else {
         EBM_ASSERT(IsRegression(cRuntimeClasses));
         ret = ApplyTermUpdateValidationZeroFeatures<k_regression>::Func(
            cRuntimeClasses,
            runtimeBitPack,
            aMulticlassMidwayTemp,
            aUpdateScores,
            cSamples,
            aInputData,
            aTargetData,
            aWeight,
            aSampleScore,
            aGradientAndHessian
         );
      }
   } else {
      if(IsClassification(cRuntimeClasses)) {
         ret = ApplyTermUpdateValidationNormalTarget<2>::Func(
            cRuntimeClasses,
            runtimeBitPack,
            aMulticlassMidwayTemp,
            aUpdateScores,
            cSamples,
            aInputData,
            aTargetData,
            aWeight,
            aSampleScore,
            aGradientAndHessian
         );
      } else {
         EBM_ASSERT(IsRegression(cRuntimeClasses));
         ret = ApplyTermUpdateValidationInternal<k_regression, k_cItemsPerBitPackDynamic>::Func(
            cRuntimeClasses,
            runtimeBitPack,
            aMulticlassMidwayTemp,
            aUpdateScores,
            cSamples,
            aInputData,
            aTargetData,
            aWeight,
            aSampleScore,
            aGradientAndHessian
         );
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
