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

struct ApplyValidation {
   // For now this is localized to this module, but in the future we will switch to the same named class in bridge_c.h

   ptrdiff_t m_cClasses;
   ptrdiff_t m_cPack;
   FloatFast * m_aMulticlassMidwayTemp;
   const FloatFast * m_aUpdateTensorScores;
   size_t m_cSamples;
   const StorageDataType * m_aPacked;
   const void * m_aTargets;
   const FloatFast * m_aWeights;
   FloatFast * m_aSampleScores;
   FloatFast * m_aGradientsAndHessians;
   double m_metricOut;
};

// C++ does not allow partial function specialization, so we need to use these cumbersome static class functions to do partial function specialization

template<ptrdiff_t cCompilerClasses, ptrdiff_t compilerBitPack>
struct ApplyTermUpdateValidationInternal final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyValidation * const pData) {
      static_assert(IsClassification(cCompilerClasses), "must be classification");
      static_assert(!IsBinaryClassification(cCompilerClasses), "must be multiclass");
      const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, pData->m_cClasses);
      const size_t cScores = GetCountScores(cClasses);

      const FloatFast * const aUpdateTensorScores = pData->m_aUpdateTensorScores;
      EBM_ASSERT(nullptr != aUpdateTensorScores);
      EBM_ASSERT(1 <= pData->m_cSamples);

      const FloatFast * pWeight = pData->m_aWeights;

      FloatFast sumLogLoss = 0;
      const StorageDataType * pTargetData = reinterpret_cast<const StorageDataType *>(pData->m_aTargets);
      FloatFast * pSampleScore = pData->m_aSampleScores;
      const FloatFast * const pSampleScoresTrueEnd = pSampleScore + pData->m_cSamples * cScores;
      const FloatFast * pSampleScoresExit = pSampleScoresTrueEnd;
      const FloatFast * pSampleScoresInnerEnd = pSampleScoresTrueEnd;

      size_t cItemsPerBitPack = 0;
      size_t cBitsPerItemMax = 0;
      StorageDataType maskBits = 0;
      const StorageDataType * pInputData = nullptr;
      StorageDataType iTensorBinCombined = 0;

      constexpr bool bZeroDimensional = k_cItemsPerBitPackNone == compilerBitPack;
      if(bZeroDimensional) {
         goto zero_dimensional;
      }

      cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, pData->m_cPack);
      EBM_ASSERT(1 <= cItemsPerBitPack);
      EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
      cBitsPerItemMax = GetCountBits(cItemsPerBitPack);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      maskBits = (~StorageDataType { 0 }) >> (k_cBitsForStorageType - cBitsPerItemMax);

      pInputData = pData->m_aPacked;

      if(pData->m_cSamples <= cItemsPerBitPack) {
         goto one_last_loop;
      }
      pSampleScoresExit = pSampleScoresTrueEnd - ((pData->m_cSamples - 1) % cItemsPerBitPack + 1) * cScores;
      EBM_ASSERT(pSampleScore < pSampleScoresExit);
      EBM_ASSERT(pSampleScoresExit < pSampleScoresTrueEnd);

      do {
         pSampleScoresInnerEnd = pSampleScore + cItemsPerBitPack * cScores;
         // jumping back into this loop and changing pSampleScoresInnerEnd to a dynamic value that isn't compile time determinable causes this 
         // function to NOT be optimized for templated cItemsPerBitPack, but that's ok since avoiding one unpredictable branch here is negligible
      one_last_loop:;
         // we store the already multiplied dimensional value in *pInputData
         iTensorBinCombined = *pInputData;
         ++pInputData;
         do {
         zero_dimensional:;

            size_t targetData = static_cast<size_t>(*pTargetData);
            ++pTargetData;

            const size_t iTensorBin = static_cast<size_t>(maskBits & iTensorBinCombined);

            const FloatFast * pUpdateScore = &aUpdateTensorScores[iTensorBin * cScores];
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

            const FloatFast sampleLogLoss = EbmStats::ComputeSingleSampleLogLossMulticlass(sumExp, itemExp);

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

      pData->m_metricOut = static_cast<double>(sumLogLoss);
      return Error_None;
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<ptrdiff_t compilerBitPack>
struct ApplyTermUpdateValidationInternal<2, compilerBitPack> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyValidation * const pData) {
      const FloatFast * const aUpdateTensorScores = pData->m_aUpdateTensorScores;
      EBM_ASSERT(nullptr != aUpdateTensorScores);
      EBM_ASSERT(1 <= pData->m_cSamples);

      const FloatFast * pWeight = pData->m_aWeights;

      FloatFast sumLogLoss = 0;
      const StorageDataType * pTargetData = reinterpret_cast<const StorageDataType *>(pData->m_aTargets);
      FloatFast * pSampleScore = pData->m_aSampleScores;
      const FloatFast * const pSampleScoresTrueEnd = pSampleScore + pData->m_cSamples;
      const FloatFast * pSampleScoresExit = pSampleScoresTrueEnd;
      const FloatFast * pSampleScoresInnerEnd = pSampleScoresTrueEnd;

      size_t cItemsPerBitPack = 0;
      size_t cBitsPerItemMax = 0;
      StorageDataType maskBits = 0;
      const StorageDataType * pInputData = nullptr;
      StorageDataType iTensorBinCombined = 0;

      constexpr bool bZeroDimensional = k_cItemsPerBitPackNone == compilerBitPack;
      if(bZeroDimensional) {
         goto zero_dimensional;
      }

      cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, pData->m_cPack);
      EBM_ASSERT(1 <= cItemsPerBitPack);
      EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
      cBitsPerItemMax = GetCountBits(cItemsPerBitPack);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      maskBits = (~StorageDataType { 0 }) >> (k_cBitsForStorageType - cBitsPerItemMax);

      pInputData = pData->m_aPacked;

      if(pData->m_cSamples <= cItemsPerBitPack) {
         goto one_last_loop;
      }
      pSampleScoresExit = pSampleScoresTrueEnd - ((pData->m_cSamples - 1) % cItemsPerBitPack + 1);
      EBM_ASSERT(pSampleScore < pSampleScoresExit);
      EBM_ASSERT(pSampleScoresExit < pSampleScoresTrueEnd);

      do {
         pSampleScoresInnerEnd = pSampleScore + cItemsPerBitPack;
         // jumping back into this loop and changing pSampleScoresInnerEnd to a dynamic value that isn't compile time determinable causes this 
         // function to NOT be optimized for templated cItemsPerBitPack, but that's ok since avoiding one unpredictable branch here is negligible
      one_last_loop:;
         // we store the already multiplied dimensional value in *pInputData
         iTensorBinCombined = *pInputData;
         ++pInputData;
         do {
         zero_dimensional:;

            size_t targetData = static_cast<size_t>(*pTargetData);
            ++pTargetData;

            const size_t iTensorBin = static_cast<size_t>(maskBits & iTensorBinCombined);

            const FloatFast updateScore = aUpdateTensorScores[iTensorBin];
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

      pData->m_metricOut = static_cast<double>(sumLogLoss);
      return Error_None;
   }
};
#endif // EXPAND_BINARY_LOGITS

template<ptrdiff_t compilerBitPack>
struct ApplyTermUpdateValidationInternal<k_regression, compilerBitPack> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyValidation * const pData) {
      const FloatFast * const aUpdateTensorScores = pData->m_aUpdateTensorScores;
      EBM_ASSERT(nullptr != aUpdateTensorScores);
      EBM_ASSERT(1 <= pData->m_cSamples);

      const FloatFast * pWeight = pData->m_aWeights;

      FloatFast sumSquareError = 0;
      // no hessians for regression
      FloatFast * pGradient = pData->m_aGradientsAndHessians;
      const FloatFast * const pGradientsTrueEnd = pGradient + pData->m_cSamples;
      const FloatFast * pGradientsExit = pGradientsTrueEnd;
      const FloatFast * pGradientsInnerEnd = pGradientsTrueEnd;

      size_t cItemsPerBitPack = 0;
      size_t cBitsPerItemMax = 0;
      StorageDataType maskBits = 0;
      const StorageDataType * pInputData = nullptr;
      StorageDataType iTensorBinCombined = 0;

      constexpr bool bZeroDimensional = k_cItemsPerBitPackNone == compilerBitPack;
      if(bZeroDimensional) {
         goto zero_dimensional;
      }

      cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, pData->m_cPack);
      EBM_ASSERT(1 <= cItemsPerBitPack);
      EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
      cBitsPerItemMax = GetCountBits(cItemsPerBitPack);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      maskBits = (~StorageDataType { 0 }) >> (k_cBitsForStorageType - cBitsPerItemMax);

      pInputData = pData->m_aPacked;

      if(pData->m_cSamples <= cItemsPerBitPack) {
         goto one_last_loop;
      }
      pGradientsExit = pGradientsTrueEnd - ((pData->m_cSamples - 1) % cItemsPerBitPack + 1);
      EBM_ASSERT(pGradient < pGradientsExit);
      EBM_ASSERT(pGradientsExit < pGradientsTrueEnd);

      do {
         pGradientsInnerEnd = pGradient + cItemsPerBitPack;
         // jumping back into this loop and changing pSampleScoresInnerEnd to a dynamic value that isn't compile time determinable causes this 
         // function to NOT be optimized for templated cItemsPerBitPack, but that's ok since avoiding one unpredictable branch here is negligible
      one_last_loop:;
         // we store the already multiplied dimensional value in *pInputData
         iTensorBinCombined = *pInputData;
         ++pInputData;
         do {
         zero_dimensional:;

            const size_t iTensorBin = static_cast<size_t>(maskBits & iTensorBinCombined);

            const FloatFast updateScore = aUpdateTensorScores[iTensorBin];
            // this will apply a small fix to our existing ValidationSampleScores, either positive or negative, whichever is needed
            const FloatFast gradient = EbmStats::ComputeGradientRegressionMSEFromOriginalGradient(*pGradient, updateScore);
            *pGradient = gradient;
            ++pGradient;

            const FloatFast sampleSquaredError = EbmStats::ComputeSingleSampleSquaredErrorRegressionFromGradient(gradient);

            EBM_ASSERT(std::isnan(sampleSquaredError) || 0 <= sampleSquaredError);

            FloatFast weight = 1;
            if(nullptr != pWeight) {
               // TODO: template this check away
               weight = *pWeight;
               ++pWeight;
            }
            sumSquareError += sampleSquaredError * weight;

            iTensorBinCombined >>= cBitsPerItemMax;
         } while(pGradientsInnerEnd != pGradient);
      } while(pGradientsExit != pGradient);

      // first time through?
      if(pGradientsTrueEnd != pGradient) {
         pGradientsInnerEnd = pGradientsTrueEnd;
         pGradientsExit = pGradientsTrueEnd;
         goto one_last_loop;
      }

      pData->m_metricOut = static_cast<double>(sumSquareError);
      return Error_None;
   }
};

template<ptrdiff_t cCompilerClasses>
INLINE_RELEASE_TEMPLATED static ErrorEbm BitPackPre(ApplyValidation * const pData) {
   if(k_cItemsPerBitPackNone == pData->m_cPack) {
      return ApplyTermUpdateValidationInternal<cCompilerClasses, k_cItemsPerBitPackNone>::Func(pData);
   } else {
      return ApplyTermUpdateValidationInternal<cCompilerClasses, k_cItemsPerBitPackDynamic>::Func(pData);
   }
}

template<ptrdiff_t cPossibleClasses>
struct CountClasses final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyValidation * const pData) {
      if(cPossibleClasses == pData->m_cClasses) {
         return BitPackPre<cPossibleClasses>(pData);
      } else {
         return CountClasses<cPossibleClasses + 1>::Func(pData);
      }
   }
};

template<>
struct CountClasses<k_cCompilerClassesMax + 1> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyValidation * const pData) {
      return BitPackPre<k_dynamicClassification>(pData);
   }
};

extern ErrorEbm ApplyTermUpdateValidation(
   const ptrdiff_t cRuntimeClasses,
   const ptrdiff_t runtimeBitPack,
   FloatFast * const aMulticlassMidwayTemp,
   const FloatFast * const aUpdateScores,
   const size_t cSamples,
   const StorageDataType * const aInputData,
   const void * const aTargetData,
   const FloatFast * const aWeight,
   FloatFast * const aSampleScore,
   FloatFast * const aGradientAndHessian,
   double * const pMetricOut
) {
   LOG_0(Trace_Verbose, "Entered ApplyTermUpdateValidation");

   ApplyValidation data;
   data.m_cClasses = cRuntimeClasses;
   data.m_cPack = runtimeBitPack;
   data.m_aMulticlassMidwayTemp = aMulticlassMidwayTemp;
   data.m_aUpdateTensorScores = aUpdateScores;
   data.m_cSamples = cSamples;
   data.m_aPacked = aInputData;
   data.m_aTargets = aTargetData;
   data.m_aWeights = aWeight;
   data.m_aSampleScores = aSampleScore;
   data.m_aGradientsAndHessians = aGradientAndHessian;

   ErrorEbm error;
   if(IsClassification(cRuntimeClasses)) {
      error = CountClasses<2>::Func(&data);
   } else {
      EBM_ASSERT(IsRegression(cRuntimeClasses));
      error = BitPackPre<k_regression>(&data);
   }

   if(Error_None != error) {
      return error;
   }

   EBM_ASSERT(std::isnan(data.m_metricOut) || -k_epsilonLogLoss <= data.m_metricOut);
   if(UNLIKELY(/* NaN */ !LIKELY(0.0 <= data.m_metricOut))) {
      // this also checks for NaN since NaN < anything is FALSE

      // if we've overflowed to a NaN, then conver it to +inf since +inf is our general overflow marker
      // if we've gotten a value that's slightly negative, which can happen for numeracy reasons, clip to zero

      data.m_metricOut = std::isnan(data.m_metricOut) ? std::numeric_limits<double>::infinity() : double { 0 };
   }
   EBM_ASSERT(!std::isnan(data.m_metricOut));
   EBM_ASSERT(0.0 <= data.m_metricOut);

   LOG_0(Trace_Verbose, "Exited ApplyTermUpdateValidation");

   *pMetricOut = data.m_metricOut;
   return Error_None;
}

} // DEFINED_ZONE_NAME
