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

template<ptrdiff_t cCompilerClasses, ptrdiff_t compilerBitPack, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
struct ApplyUpdateInternal final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyUpdateBridge * const pData) {
      static_assert(IsClassification(cCompilerClasses), "must be classification");
      static_assert(!IsBinaryClassification(cCompilerClasses), "must be multiclass");
      static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == compilerBitPack;
      static constexpr bool bGetExp = bCalcMetric || bKeepGradHess;
      static constexpr bool bGetTarget = bCalcMetric || bKeepGradHess;

      FloatFast aLocalExpVector[GetCountScores(cCompilerClasses)];
      FloatFast * aExps;
      if(bGetExp) {
         static constexpr bool bDynamicClasses = k_dynamicClassification == cCompilerClasses;
         if(bDynamicClasses) {
            EBM_ASSERT(nullptr != pData->m_aMulticlassMidwayTemp);
            aExps = pData->m_aMulticlassMidwayTemp;
         } else {
            aExps = aLocalExpVector;
         }
      }

      const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, pData->m_cClasses);
      const size_t cScores = GetCountScores(cClasses);

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
         const ptrdiff_t cPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, pData->m_cPack);
         EBM_ASSERT(k_cItemsPerBitPackNone != cPack); // we require this condition to be templated

         const size_t cItemsPerBitPack = static_cast<size_t>(cPack);
         EBM_ASSERT(1 <= cItemsPerBitPack);
         EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);

         cBitsPerItemMax = GetCountBits<StorageDataType>(cItemsPerBitPack);
         EBM_ASSERT(1 <= cBitsPerItemMax);
         EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);

         cShift = static_cast<ptrdiff_t>((cSamples - 1) % cItemsPerBitPack * cBitsPerItemMax);
         cShiftReset = static_cast<ptrdiff_t>((cItemsPerBitPack - 1) * cBitsPerItemMax);

         maskBits = static_cast<size_t>(MakeLowMask<StorageDataType>(cBitsPerItemMax));

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

            FloatFast weight;
            if(bWeight) {
               weight = *pWeight;
               ++pWeight;
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
                  if(bWeight) {
                     // This is only used during the initialization of interaction detection. For boosting
                     // we currently multiply by the weight during bin summation instead since we use the weight
                     // there to include the inner bagging counts of occurences.
                     // Whether this multiplication happens or not is controlled by the caller by passing in the
                     // weight array or not.
                     gradient *= weight;
                     hessian *= weight;
                  }
                  pGradientAndHessian[iScore2 << 1] = gradient;
                  pGradientAndHessian[(iScore2 << 1) + 1] = hessian;
                  ++iScore2;
               } while(cScores != iScore2);

               pGradientAndHessian[targetData << 1] = EbmStats::MulticlassFixTargetGradient(
                  pGradientAndHessian[targetData << 1], bWeight ? weight : FloatFast { 1 });

               pGradientAndHessian += cScores << 1;
            }

            if(bCalcMetric) {
               const FloatFast itemExp = aExps[targetData];

               FloatFast sampleLogLoss = EbmStats::ComputeSingleSampleLogLossMulticlass(sumExp, itemExp);
               EBM_ASSERT(std::isnan(sampleLogLoss) || -k_epsilonLogLoss <= sampleLogLoss);

               if(bWeight) {
                  sampleLogLoss *= weight;
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
template<ptrdiff_t compilerBitPack, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
struct ApplyUpdateInternal<2, compilerBitPack, bKeepGradHess, bCalcMetric, bWeight> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyUpdateBridge * const pData) {
      static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == compilerBitPack;
      static constexpr bool bGetTarget = bCalcMetric || bKeepGradHess;

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
         const ptrdiff_t cPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, pData->m_cPack);
         EBM_ASSERT(k_cItemsPerBitPackNone != cPack); // we require this condition to be templated

         const size_t cItemsPerBitPack = static_cast<size_t>(cPack);
         EBM_ASSERT(1 <= cItemsPerBitPack);
         EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);

         cBitsPerItemMax = GetCountBits<StorageDataType>(cItemsPerBitPack);
         EBM_ASSERT(1 <= cBitsPerItemMax);
         EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);

         cShift = static_cast<ptrdiff_t>((cSamples - 1) % cItemsPerBitPack * cBitsPerItemMax);
         cShiftReset = static_cast<ptrdiff_t>((cItemsPerBitPack - 1) * cBitsPerItemMax);

         maskBits = static_cast<size_t>(MakeLowMask<StorageDataType>(cBitsPerItemMax));

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

            FloatFast weight;
            if(bWeight) {
               weight = *pWeight;
               ++pWeight;
            }

            if(bKeepGradHess) {
               FloatFast gradient = EbmStats::InverseLinkFunctionThenCalculateGradientBinaryClassification(sampleScore, targetData);
               FloatFast hessian = EbmStats::CalculateHessianFromGradientBinaryClassification(gradient);
               if(bWeight) {
                  // This is only used during the initialization of interaction detection. For boosting
                  // we currently multiply by the weight during bin summation instead since we use the weight
                  // there to include the inner bagging counts of occurences.
                  // Whether this multiplication happens or not is controlled by the caller by passing in the
                  // weight array or not.
                  gradient *= weight;
                  hessian *= weight;
               }
               *pGradientAndHessian = gradient;
               *(pGradientAndHessian + 1) = hessian;
               pGradientAndHessian += 2;
            }

            if(bCalcMetric) {
               FloatFast sampleLogLoss = EbmStats::ComputeSingleSampleLogLossBinaryClassification(sampleScore, targetData);
               EBM_ASSERT(std::isnan(sampleLogLoss) || 0 <= sampleLogLoss);

               if(bWeight) {
                  sampleLogLoss *= weight;
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

template<ptrdiff_t compilerBitPack, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
struct ApplyUpdateInternal<k_regression, compilerBitPack, bKeepGradHess, bCalcMetric, bWeight> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyUpdateBridge * const pData) {
      static_assert(bKeepGradHess, "for MSE regression we should always keep the gradients");

      static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == compilerBitPack;

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
         const ptrdiff_t cPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, pData->m_cPack);
         EBM_ASSERT(k_cItemsPerBitPackNone != cPack); // we require this condition to be templated

         const size_t cItemsPerBitPack = static_cast<size_t>(cPack);
         EBM_ASSERT(1 <= cItemsPerBitPack);
         EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);

         cBitsPerItemMax = GetCountBits<StorageDataType>(cItemsPerBitPack);
         EBM_ASSERT(1 <= cBitsPerItemMax);
         EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);

         cShift = static_cast<ptrdiff_t>((cSamples - 1) % cItemsPerBitPack * cBitsPerItemMax);
         cShiftReset = static_cast<ptrdiff_t>((cItemsPerBitPack - 1) * cBitsPerItemMax);

         maskBits = static_cast<size_t>(MakeLowMask<StorageDataType>(cBitsPerItemMax));

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

            // for MSE regression we cannot put the weight into the gradient like we could with other objectives
            // for regression or for classification because we only preserve the gradient and to calculate the
            // square error we need the original gradient and not the weight multiplied gradient... well we could
            // do it but it would require a division. A better way would be to have two FloatFast arrays: a 
            // non-weight adjusted one and a weight adjusted one for when inner bags are used
            // NOTE: For interactions we can and do put the weight into the gradient because we never update it
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
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyUpdateBridge * const pData) {
      if(nullptr != pData->m_aGradientsAndHessians) {
         static constexpr bool bKeepGradHess = true;

         // if we are updating the gradients then we are doing training and do not need to calculate the metric
         EBM_ASSERT(!pData->m_bCalcMetric);
         static constexpr bool bCalcMetric = false;

         if(nullptr != pData->m_aWeights) {
            static constexpr bool bWeight = true;

            // this branch will only be taking during interaction initialization

            return ApplyUpdateInternal<cCompilerClasses, compilerBitPack, bKeepGradHess, bCalcMetric, bWeight>::Func(pData);
         } else {
            static constexpr bool bWeight = false;
            return ApplyUpdateInternal<cCompilerClasses, compilerBitPack, bKeepGradHess, bCalcMetric, bWeight>::Func(pData);
         }
      } else {
         static constexpr bool bKeepGradHess = false;

         if(pData->m_bCalcMetric) {
            static constexpr bool bCalcMetric = true;

            if(nullptr != pData->m_aWeights) {
               static constexpr bool bWeight = true;
               return ApplyUpdateInternal<cCompilerClasses, compilerBitPack, bKeepGradHess, bCalcMetric, bWeight>::Func(pData);
            } else {
               static constexpr bool bWeight = false;
               return ApplyUpdateInternal<cCompilerClasses, compilerBitPack, bKeepGradHess, bCalcMetric, bWeight>::Func(pData);
            }
         } else {
            static constexpr bool bCalcMetric = false;

            // currently this branch is not taken, but if would be if we wanted to allow in the future
            // non-metric calculating validation for boosting.  For instance if we wanted to substitute an alternate
            // metric or if for performance reasons we only want to calculate the metric every N rounds of boosting

            EBM_ASSERT(nullptr == pData->m_aWeights);
            static constexpr bool bWeight = false; // if we are not calculating the metric or updating gradients then we never need the weights

            return ApplyUpdateInternal<cCompilerClasses, compilerBitPack, bKeepGradHess, bCalcMetric, bWeight>::Func(pData);
         }
      }
   }
};

template<ptrdiff_t compilerBitPack>
struct FinalOptions<k_regression, compilerBitPack> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyUpdateBridge * const pData) {
      static constexpr ptrdiff_t cCompilerClasses = k_regression;

      EBM_ASSERT(nullptr != pData->m_aGradientsAndHessians); // we always keep gradients for regression
      static constexpr bool bKeepGradHess = true;

      if(pData->m_bCalcMetric) {
         static constexpr bool bCalcMetric = true;

         if(nullptr != pData->m_aWeights) {
            static constexpr bool bWeight = true;
            return ApplyUpdateInternal<cCompilerClasses, compilerBitPack, bKeepGradHess, bCalcMetric, bWeight>::Func(pData);
         } else {
            static constexpr bool bWeight = false;
            return ApplyUpdateInternal<cCompilerClasses, compilerBitPack, bKeepGradHess, bCalcMetric, bWeight>::Func(pData);
         }
      } else {
         static constexpr bool bCalcMetric = false;

         EBM_ASSERT(nullptr == pData->m_aWeights);
         static constexpr bool bWeight = false; // if we are not calculating the metric then we never need the weights

         return ApplyUpdateInternal<cCompilerClasses, compilerBitPack, bKeepGradHess, bCalcMetric, bWeight>::Func(pData);
      }
   }
};

template<ptrdiff_t cCompilerClasses>
INLINE_RELEASE_TEMPLATED static ErrorEbm BitPack(ApplyUpdateBridge * const pData) {
   if(k_cItemsPerBitPackNone != pData->m_cPack) {
      return FinalOptions<cCompilerClasses, k_cItemsPerBitPackDynamic>::Func(pData);
   } else {
      // this needs to be special cased because otherwise we would inject comparisons into the dynamic version
      return FinalOptions<cCompilerClasses, k_cItemsPerBitPackNone>::Func(pData);
   }
}

template<ptrdiff_t cPossibleClasses>
struct CountClasses final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyUpdateBridge * const pData) {
      if(cPossibleClasses == pData->m_cClasses) {
         return BitPack<cPossibleClasses>(pData);
      } else {
         return CountClasses<cPossibleClasses + 1>::Func(pData);
      }
   }
};

template<>
struct CountClasses<k_cCompilerClassesMax + 1> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(ApplyUpdateBridge * const pData) {
      return BitPack<k_dynamicClassification>(pData);
   }
};

extern ErrorEbm ApplyUpdate(ApplyUpdateBridge * const pData) {
   LOG_0(Trace_Verbose, "Entered ApplyUpdate");

   EBM_ASSERT(nullptr != pData);

   ErrorEbm error;
   if(IsClassification(pData->m_cClasses)) {
      error = CountClasses<2>::Func(pData);
   } else {
      EBM_ASSERT(IsRegression(pData->m_cClasses));
      error = BitPack<k_regression>(pData);
   }
   if(Error_None != error) {
      return error;
   }

   if(pData->m_bCalcMetric) {
      double metricOut = pData->m_metricOut;
      EBM_ASSERT(std::isnan(metricOut) || -k_epsilonLogLoss <= metricOut);
      if(UNLIKELY(/* NaN */ !LIKELY(0.0 <= metricOut))) {
         // this also checks for NaN since NaN < anything is FALSE

         // if we've overflowed to a NaN, then conver it to +inf since +inf is our general overflow marker
         // if we've gotten a value that's slightly negative, which can happen for numeracy reasons, clip to zero

         pData->m_metricOut = std::isnan(metricOut) ? std::numeric_limits<double>::infinity() : double { 0 };
      }
      EBM_ASSERT(!std::isnan(pData->m_metricOut));
      EBM_ASSERT(0.0 <= pData->m_metricOut);
   }

   LOG_0(Trace_Verbose, "Exited ApplyUpdate");

   return Error_None;
}

} // DEFINED_ZONE_NAME
