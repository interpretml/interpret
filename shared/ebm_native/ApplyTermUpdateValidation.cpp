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
   bool m_bCalcMetric;
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

template<ptrdiff_t cCompilerClasses, ptrdiff_t compilerBitPack, bool bCalcMetric, bool bWeight, bool bKeepGradHess>
struct ApplyTermUpdateValidationInternal final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyValidation * const pData) {
      static_assert(IsClassification(cCompilerClasses), "must be classification");
      static_assert(!IsBinaryClassification(cCompilerClasses), "must be multiclass");
      constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == compilerBitPack;
      constexpr bool bGetExp = bCalcMetric || bKeepGradHess;
      constexpr bool bGetTarget = bCalcMetric || bKeepGradHess;

      FloatFast aLocalExpVector[GetCountScores(cCompilerClasses)];
      FloatFast * const aExps = k_dynamicClassification == cCompilerClasses ? pData->m_aMulticlassMidwayTemp : aLocalExpVector;

      const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, pData->m_cClasses);
      const size_t cScores = GetCountScores(cClasses);

      const ptrdiff_t cPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, pData->m_cPack);

      const FloatFast * const aUpdateTensorScores = pData->m_aUpdateTensorScores;
      EBM_ASSERT(nullptr != aUpdateTensorScores);

      const size_t cSamples = pData->m_cSamples;
      EBM_ASSERT(1 <= cSamples);
      FloatFast * pSampleScore = pData->m_aSampleScores;
      const FloatFast * const pSampleScoresEnd = pSampleScore + cSamples * cScores;

      size_t cBitsPerItemMax;
      ptrdiff_t cShift;
      ptrdiff_t cShiftReset;
      size_t maskBits;
      const StorageDataType * pInputData;

      FloatFast updateScore;
      const FloatFast * aBinScores;

      if(bCompilerZeroDimensional) {
         aBinScores = aUpdateTensorScores;
      } else {
         EBM_ASSERT(k_cItemsPerBitPackNone != cPack); // we require this condition to be templated

         const size_t cItemsPerBitPack = static_cast<size_t>(cPack);
         EBM_ASSERT(1 <= cItemsPerBitPack);
         EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
         cBitsPerItemMax = GetCountBits(cItemsPerBitPack);

         cShift = static_cast<ptrdiff_t>((cSamples - 1) % cItemsPerBitPack * cBitsPerItemMax);
         cShiftReset = static_cast<ptrdiff_t>(cBitsPerItemMax * (cItemsPerBitPack - 1));

         EBM_ASSERT(1 <= cBitsPerItemMax);
         EBM_ASSERT(cBitsPerItemMax <= k_cBitsForSizeT);
         maskBits = (~size_t { 0 }) >> (k_cBitsForSizeT - cBitsPerItemMax);

         pInputData = pData->m_aPacked;
      }

      const StorageDataType * pTargetData;
      if(bGetTarget) {
         pTargetData = reinterpret_cast<const StorageDataType *>(pData->m_aTargets);
      }

      FloatFast * pGradientAndHessian;
      if(bKeepGradHess) {
         pGradientAndHessian = pData->m_aGradientsAndHessians;
      }

      const FloatFast * pWeight;
      if(bWeight) {
         pWeight = pData->m_aWeights;
      }

      FloatFast sumLogLoss;
      if(bCalcMetric) {
         sumLogLoss = 0;
      }
      do {
         StorageDataType iTensorBinCombined;
         if(!bCompilerZeroDimensional) {
            // we store the already multiplied dimensional value in *pInputData
            iTensorBinCombined = *pInputData;
            ++pInputData;
         }
         while(true) {
            if(!bCompilerZeroDimensional) {
               const size_t iTensorBin = static_cast<size_t>(iTensorBinCombined >> cShift) & maskBits;
               aBinScores = &aUpdateTensorScores[iTensorBin * cScores];
            }

            FloatFast sumExp;
            if(bGetExp) {
               sumExp = 0;
            }
            size_t iScore1 = 0;
            do {
               updateScore = aBinScores[iScore1];

               const FloatFast sampleScore = pSampleScore[iScore1] + updateScore;
               pSampleScore[iScore1] = sampleScore;

               if(bGetExp) {
                  const FloatFast oneExp = ExpForMulticlass<false>(sampleScore);
                  sumExp += oneExp;
                  aExps[iScore1] = oneExp;
               }

               ++iScore1;
            } while(cScores != iScore1);

            size_t targetData;
            if(bGetTarget) {
               targetData = static_cast<size_t>(*pTargetData);
               ++pTargetData;
            }

            pSampleScore += cScores;

            if(bKeepGradHess) {
               const FloatFast sumExpInverted = FloatFast { 1 } / sumExp;

               size_t iScore2 = 0;
               do {
                  FloatFast gradient;
                  FloatFast hessian;
                  EbmStats::InverseLinkFunctionThenCalculateGradientAndHessianMulticlassForNonTarget(
                     sumExpInverted,
                     aExps[iScore2],
                     gradient,
                     hessian
                  );
                  pGradientAndHessian[iScore2 << 1] = gradient;
                  pGradientAndHessian[(iScore2 << 1) + 1] = hessian;
                  ++iScore2;
               } while(cScores != iScore2);

               pGradientAndHessian[targetData << 1] =
                  EbmStats::MulticlassFixTargetGradient(pGradientAndHessian[targetData << 1]);

               pGradientAndHessian += cScores << 1;
            }

            if(bCalcMetric) {
               const FloatFast itemExp = aExps[targetData];

               FloatFast sampleLogLoss = EbmStats::ComputeSingleSampleLogLossMulticlass(sumExp, itemExp);
               EBM_ASSERT(std::isnan(sampleLogLoss) || -k_epsilonLogLoss <= sampleLogLoss);

               if(bWeight) {
                  const FloatFast weight = *pWeight;
                  sampleLogLoss *= weight;
                  ++pWeight;
               }
               sumLogLoss += sampleLogLoss;
            }

            if(bCompilerZeroDimensional) {
               if(pSampleScoresEnd == pSampleScore) {
                  break;
               }
            } else {
               cShift -= cBitsPerItemMax;
               if(cShift < 0) {
                  break;
               }
            }
         }
         if(bCompilerZeroDimensional) {
            break;
         }
         cShift = cShiftReset;
      } while(pSampleScoresEnd != pSampleScore);

      if(bCalcMetric) {
         pData->m_metricOut = static_cast<double>(sumLogLoss);
      }

      return Error_None;
   }
};

#ifndef EXPAND_BINARY_LOGITS
template<ptrdiff_t compilerBitPack, bool bCalcMetric, bool bWeight, bool bKeepGradHess>
struct ApplyTermUpdateValidationInternal<2, compilerBitPack, bCalcMetric, bWeight, bKeepGradHess> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyValidation * const pData) {
      constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == compilerBitPack;
      constexpr bool bGetTarget = bCalcMetric || bKeepGradHess;

      const ptrdiff_t cPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, pData->m_cPack);

      const FloatFast * const aUpdateTensorScores = pData->m_aUpdateTensorScores;
      EBM_ASSERT(nullptr != aUpdateTensorScores);

      const size_t cSamples = pData->m_cSamples;
      EBM_ASSERT(1 <= cSamples);
      FloatFast * pSampleScore = pData->m_aSampleScores;
      const FloatFast * const pSampleScoresEnd = pSampleScore + cSamples;

      size_t cBitsPerItemMax;
      ptrdiff_t cShift;
      ptrdiff_t cShiftReset;
      size_t maskBits;
      const StorageDataType * pInputData;

      FloatFast updateScore;

      if(bCompilerZeroDimensional) {
         updateScore = aUpdateTensorScores[0];
      } else {
         EBM_ASSERT(k_cItemsPerBitPackNone != cPack); // we require this condition to be templated

         const size_t cItemsPerBitPack = static_cast<size_t>(cPack);
         EBM_ASSERT(1 <= cItemsPerBitPack);
         EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
         cBitsPerItemMax = GetCountBits(cItemsPerBitPack);

         cShift = static_cast<ptrdiff_t>((cSamples - 1) % cItemsPerBitPack * cBitsPerItemMax);
         cShiftReset = static_cast<ptrdiff_t>(cBitsPerItemMax * (cItemsPerBitPack - 1));

         EBM_ASSERT(1 <= cBitsPerItemMax);
         EBM_ASSERT(cBitsPerItemMax <= k_cBitsForSizeT);
         maskBits = (~size_t { 0 }) >> (k_cBitsForSizeT - cBitsPerItemMax);

         pInputData = pData->m_aPacked;
      }

      const StorageDataType * pTargetData;
      if(bGetTarget) {
         pTargetData = reinterpret_cast<const StorageDataType *>(pData->m_aTargets);
      }

      FloatFast * pGradientAndHessian;
      if(bKeepGradHess) {
         pGradientAndHessian = pData->m_aGradientsAndHessians;
      }

      const FloatFast * pWeight;
      if(bWeight) {
         pWeight = pData->m_aWeights;
      }

      FloatFast sumLogLoss;
      if(bCalcMetric) {
         sumLogLoss = 0;
      }
      do {
         StorageDataType iTensorBinCombined;
         if(!bCompilerZeroDimensional) {
            // we store the already multiplied dimensional value in *pInputData
            iTensorBinCombined = *pInputData;
            ++pInputData;
         }
         while(true) {
            if(!bCompilerZeroDimensional) {
               const size_t iTensorBin = static_cast<size_t>(iTensorBinCombined >> cShift) & maskBits;
               updateScore = aUpdateTensorScores[iTensorBin];
            }

            size_t targetData;
            if(bGetTarget) {
               targetData = static_cast<size_t>(*pTargetData);
               ++pTargetData;
            }

            const FloatFast sampleScore = *pSampleScore + updateScore;
            *pSampleScore = sampleScore;
            ++pSampleScore;

            if(bKeepGradHess) {
               const FloatFast gradient = EbmStats::InverseLinkFunctionThenCalculateGradientBinaryClassification(sampleScore, targetData);
               *pGradientAndHessian = gradient;
               *(pGradientAndHessian + 1) = EbmStats::CalculateHessianFromGradientBinaryClassification(gradient);
               pGradientAndHessian += 2;
            }

            if(bCalcMetric) {
               FloatFast sampleLogLoss = EbmStats::ComputeSingleSampleLogLossBinaryClassification(sampleScore, targetData);
               EBM_ASSERT(std::isnan(sampleLogLoss) || 0 <= sampleLogLoss);

               if(bWeight) {
                  const FloatFast weight = *pWeight;
                  sampleLogLoss *= weight;
                  ++pWeight;
               }
               sumLogLoss += sampleLogLoss;
            }

            if(bCompilerZeroDimensional) {
               if(pSampleScoresEnd == pSampleScore) {
                  break;
               }
            } else {
               cShift -= cBitsPerItemMax;
               if(cShift < 0) {
                  break;
               }
            }
         }
         if(bCompilerZeroDimensional) {
            break;
         }
         cShift = cShiftReset;
      } while(pSampleScoresEnd != pSampleScore);

      if(bCalcMetric) {
         pData->m_metricOut = static_cast<double>(sumLogLoss);
      }

      return Error_None;
   }
};
#endif // EXPAND_BINARY_LOGITS

template<ptrdiff_t compilerBitPack, bool bCalcMetric, bool bWeight, bool bKeepGradHess>
struct ApplyTermUpdateValidationInternal<k_regression, compilerBitPack, bCalcMetric, bWeight, bKeepGradHess> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyValidation * const pData) {
      static_assert(bKeepGradHess, "for MSE regression we should always keep the gradients");

      constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == compilerBitPack;

      const ptrdiff_t cPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, pData->m_cPack);

      const FloatFast * const aUpdateTensorScores = pData->m_aUpdateTensorScores;
      EBM_ASSERT(nullptr != aUpdateTensorScores);

      const size_t cSamples = pData->m_cSamples;
      EBM_ASSERT(1 <= cSamples);
      FloatFast * pGradient = pData->m_aGradientsAndHessians; // no hessians for regression
      const FloatFast * const pGradientsEnd = pGradient + cSamples;

      size_t cBitsPerItemMax;
      ptrdiff_t cShift;
      ptrdiff_t cShiftReset;
      size_t maskBits;
      const StorageDataType * pInputData;

      FloatFast updateScore;

      if(bCompilerZeroDimensional) {
         updateScore = aUpdateTensorScores[0];
      } else {
         EBM_ASSERT(k_cItemsPerBitPackNone != cPack); // we require this condition to be templated

         const size_t cItemsPerBitPack = static_cast<size_t>(cPack);
         EBM_ASSERT(1 <= cItemsPerBitPack);
         EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
         cBitsPerItemMax = GetCountBits(cItemsPerBitPack);

         cShift = static_cast<ptrdiff_t>((cSamples - 1) % cItemsPerBitPack * cBitsPerItemMax);
         cShiftReset = static_cast<ptrdiff_t>(cBitsPerItemMax * (cItemsPerBitPack - 1));

         EBM_ASSERT(1 <= cBitsPerItemMax);
         EBM_ASSERT(cBitsPerItemMax <= k_cBitsForSizeT);
         maskBits = (~size_t { 0 }) >> (k_cBitsForSizeT - cBitsPerItemMax);

         pInputData = pData->m_aPacked;
      }

      const FloatFast * pWeight;
      if(bWeight) {
         pWeight = pData->m_aWeights;
      }

      FloatFast sumSquareError;
      if(bCalcMetric) {
         sumSquareError = 0;
      }
      do {
         StorageDataType iTensorBinCombined;
         if(!bCompilerZeroDimensional) {
            // we store the already multiplied dimensional value in *pInputData
            iTensorBinCombined = *pInputData;
            ++pInputData;
         }
         while(true) {
            if(!bCompilerZeroDimensional) {
               const size_t iTensorBin = static_cast<size_t>(iTensorBinCombined >> cShift) & maskBits;
               updateScore = aUpdateTensorScores[iTensorBin];
            }

            const FloatFast gradient = EbmStats::ComputeGradientRegressionMSEFromOriginalGradient(*pGradient) + updateScore;
            *pGradient = gradient;
            ++pGradient;

            if(bCalcMetric) {
               FloatFast sampleSquaredError = EbmStats::ComputeSingleSampleSquaredErrorRegressionFromGradient(gradient);
               EBM_ASSERT(std::isnan(sampleSquaredError) || 0 <= sampleSquaredError);

               if(bWeight) {
                  const FloatFast weight = *pWeight;
                  sampleSquaredError *= weight;
                  ++pWeight;
               }
               sumSquareError += sampleSquaredError;
            }

            if(bCompilerZeroDimensional) {
               if(pGradientsEnd == pGradient) {
                  break;
               }
            } else {
               cShift -= cBitsPerItemMax;
               if(cShift < 0) {
                  break;
               }
            }
         }
         if(bCompilerZeroDimensional) {
            break;
         }
         cShift = cShiftReset;
      } while(pGradientsEnd != pGradient);

      if(bCalcMetric) {
         pData->m_metricOut = static_cast<double>(sumSquareError);
      }

      return Error_None;
   }
};

template<ptrdiff_t cCompilerClasses, ptrdiff_t compilerBitPack>
struct FinalOptions final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyValidation * const pData) {
      if(pData->m_bCalcMetric) {
         constexpr bool bCalcMetric = true;
         if(nullptr != pData->m_aWeights) {
            constexpr bool bWeight = true;
            // if we are calculating the metric then we are doing validation and we do not need the gradients
            EBM_ASSERT(nullptr == pData->m_aGradientsAndHessians);
            constexpr bool bKeepGradHess = false;
            return ApplyTermUpdateValidationInternal<cCompilerClasses, compilerBitPack, bCalcMetric, bWeight, bKeepGradHess>::Func(pData);
         } else {
            constexpr bool bWeight = false;
            // if we are calculating the metric then we are doing validation and we do not need the gradients
            EBM_ASSERT(nullptr == pData->m_aGradientsAndHessians);
            constexpr bool bKeepGradHess = false;
            return ApplyTermUpdateValidationInternal<cCompilerClasses, compilerBitPack, bCalcMetric, bWeight, bKeepGradHess>::Func(pData);
         }
      } else {
         constexpr bool bCalcMetric = false;
         EBM_ASSERT(nullptr == pData->m_aWeights);
         constexpr bool bWeight = false; // if we are not calculating the metric then we never need the weights
         if(nullptr != pData->m_aGradientsAndHessians) {
            constexpr bool bKeepGradHess = true;
            return ApplyTermUpdateValidationInternal<cCompilerClasses, compilerBitPack, bCalcMetric, bWeight, bKeepGradHess>::Func(pData);
         } else {
            // currently this branch is not taken, but if would be if we wanted to allow in the future
            // non-metric calculating validation boosting.  For instance if we wanted to substitute an alternate
            // metric or if for performance reasons we only want to calculate the metric every N rounds of boosting
            constexpr bool bKeepGradHess = false;
            return ApplyTermUpdateValidationInternal<cCompilerClasses, compilerBitPack, bCalcMetric, bWeight, bKeepGradHess>::Func(pData);
         }
      }
   }
};

template<ptrdiff_t compilerBitPack>
struct FinalOptions<k_regression, compilerBitPack> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyValidation * const pData) {
      constexpr ptrdiff_t cCompilerClasses = k_regression;
      if(pData->m_bCalcMetric) {
         constexpr bool bCalcMetric = true;
         if(nullptr != pData->m_aWeights) {
            constexpr bool bWeight = true;
            EBM_ASSERT(nullptr != pData->m_aGradientsAndHessians); // we always keep gradients for regression
            constexpr bool bKeepGradHess = true;
            return ApplyTermUpdateValidationInternal<cCompilerClasses, compilerBitPack, bCalcMetric, bWeight, bKeepGradHess>::Func(pData);
         } else {
            constexpr bool bWeight = false;
            EBM_ASSERT(nullptr != pData->m_aGradientsAndHessians); // we always keep gradients for regression
            constexpr bool bKeepGradHess = true;
            return ApplyTermUpdateValidationInternal<cCompilerClasses, compilerBitPack, bCalcMetric, bWeight, bKeepGradHess>::Func(pData);
         }
      } else {
         constexpr bool bCalcMetric = false;
         EBM_ASSERT(nullptr == pData->m_aWeights);
         constexpr bool bWeight = false; // if we are not calculating the metric then we never need the weights
         EBM_ASSERT(nullptr != pData->m_aGradientsAndHessians); // we always keep gradients for regression
         constexpr bool bKeepGradHess = true;
         return ApplyTermUpdateValidationInternal<cCompilerClasses, compilerBitPack, bCalcMetric, bWeight, bKeepGradHess>::Func(pData);
      }
   }
};

template<ptrdiff_t cCompilerClasses>
INLINE_RELEASE_TEMPLATED static ErrorEbm BitPack(ApplyValidation * const pData) {
   if(k_cItemsPerBitPackNone != pData->m_cPack) {
      return FinalOptions<cCompilerClasses, k_cItemsPerBitPackDynamic>::Func(pData);
   } else {
      // this needs to be special cased because otherwise we would inject comparisons into the dynamic version
      return FinalOptions<cCompilerClasses, k_cItemsPerBitPackNone>::Func(pData);
   }
}

template<ptrdiff_t cPossibleClasses>
struct CountClasses final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyValidation * const pData) {
      if(cPossibleClasses == pData->m_cClasses) {
         return BitPack<cPossibleClasses>(pData);
      } else {
         return CountClasses<cPossibleClasses + 1>::Func(pData);
      }
   }
};

template<>
struct CountClasses<k_cCompilerClassesMax + 1> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyValidation * const pData) {
      return BitPack<k_dynamicClassification>(pData);
   }
};

extern ErrorEbm ApplyTermUpdateValidation(
   const ptrdiff_t cRuntimeClasses,
   const ptrdiff_t runtimeBitPack,
   const bool bCalcMetric,
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
   data.m_bCalcMetric = bCalcMetric;
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
      error = BitPack<k_regression>(&data);
   }
   if(Error_None != error) {
      return error;
   }

   if(bCalcMetric) {
      EBM_ASSERT(std::isnan(data.m_metricOut) || -k_epsilonLogLoss <= data.m_metricOut);
      if(UNLIKELY(/* NaN */ !LIKELY(0.0 <= data.m_metricOut))) {
         // this also checks for NaN since NaN < anything is FALSE

         // if we've overflowed to a NaN, then conver it to +inf since +inf is our general overflow marker
         // if we've gotten a value that's slightly negative, which can happen for numeracy reasons, clip to zero

         data.m_metricOut = std::isnan(data.m_metricOut) ? std::numeric_limits<double>::infinity() : double { 0 };
      }
      EBM_ASSERT(!std::isnan(data.m_metricOut));
      EBM_ASSERT(0.0 <= data.m_metricOut);

      *pMetricOut = data.m_metricOut;
   }
   LOG_0(Trace_Verbose, "Exited ApplyTermUpdateValidation");

   return Error_None;
}

} // DEFINED_ZONE_NAME
