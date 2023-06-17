// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// Do not use this file as a reference for other objectives. LogLoss is special.

template<typename TFloat>
struct LogLossBinaryObjective final : public BinaryObjective {
   OBJECTIVE_CONSTANTS_BOILERPLATE(LogLossBinaryObjective, MINIMIZE_METRIC, Link_logit)

   inline LogLossBinaryObjective(const Config & config) {
      if(1 != config.cOutputs) {
         // we share the tag "log_loss" with multiclass classification
         throw SkipRegistrationException();
      }
   }

   inline double LinkParam() const noexcept {
      return std::numeric_limits<double>::quiet_NaN();
   }

   inline double LearningRateAdjustmentDifferentialPrivacy() const noexcept {
      return 1.0; // typically leave this at 1.0 (unmodified)
   }

   inline double LearningRateAdjustmentGradientBoosting() const noexcept {
      return 1.0; // typically leave this at 1.0 (unmodified)
   }

   inline double LearningRateAdjustmentHessianBoosting() const noexcept {
      return 1.0; // typically leave this at 1.0 (unmodified)
   }

   inline double GainAdjustmentGradientBoosting() const noexcept {
      return 1.0; // typically leave this at 1.0 (unmodified)
   }

   inline double GainAdjustmentHessianBoosting() const noexcept {
      return 1.0; // typically leave this at 1.0 (unmodified)
   }

   inline double GradientConstant() const noexcept {
      return 1.0;
   }

   inline double HessianConstant() const noexcept {
      return 1.0;
   }

   inline double FinishMetric(const double metricSum) const noexcept {
      return metricSum;
   }

   GPU_DEVICE inline TFloat CalcMetric(const TFloat score, const TFloat target) const noexcept {
      // This function is here to signal the LogLossBinaryObjective class abilities, but it will not be called
      UNUSED(score);
      UNUSED(target);
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat score, const TFloat target) const noexcept {
      // This function is here to signal the LogLossBinaryObjective class abilities, but it will not be called
      UNUSED(score);
      UNUSED(target);
      return 0.0;
   }

   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat score, const TFloat target) const noexcept {
      // This function is here to signal the LogLossBinaryObjective class abilities, but it will not be called
      UNUSED(score);
      UNUSED(target);
      return GradientHessian<TFloat>(0.0, 0.0);
   }

   template<size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   GPU_DEVICE void InjectedApplyUpdate(ApplyUpdateBridge * const pData) const {
      static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == cCompilerPack;
      static constexpr bool bGetTarget = bCalcMetric || bKeepGradHess;

#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pData);
      EBM_ASSERT(nullptr != pData->m_aUpdateTensorScores);
      EBM_ASSERT(1 <= pData->m_cSamples);
      EBM_ASSERT(nullptr != pData->m_aSampleScores);
#endif // GPU_COMPILE

      const FloatFast * const aUpdateTensorScores = reinterpret_cast<const FloatFast *>(pData->m_aUpdateTensorScores);

      const size_t cSamples = pData->m_cSamples;

      FloatFast * pSampleScore = reinterpret_cast<FloatFast *>(pData->m_aSampleScores);
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
         const ptrdiff_t cPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pData->m_cPack);

         const size_t cItemsPerBitPack = static_cast<size_t>(cPack);

         cBitsPerItemMax = GetCountBits<StorageDataType>(cItemsPerBitPack);

         cShift = static_cast<ptrdiff_t>((cSamples - 1) % cItemsPerBitPack * cBitsPerItemMax);
         cShiftReset = static_cast<ptrdiff_t>((cItemsPerBitPack - 1) * cBitsPerItemMax);

         maskBits = static_cast<size_t>(MakeLowMask<StorageDataType>(cBitsPerItemMax));

         pInputData = pData->m_aPacked;
#ifndef GPU_COMPILE
         EBM_ASSERT(k_cItemsPerBitPackNone != cPack); // we require this condition to be templated
         EBM_ASSERT(1 <= cItemsPerBitPack);
         EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
         EBM_ASSERT(1 <= cBitsPerItemMax);
         EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
         EBM_ASSERT(nullptr != pInputData);
#endif // GPU_COMPILE
      }

      const StorageDataType * pTargetData;
      if(bGetTarget) {
         pTargetData = reinterpret_cast<const StorageDataType *>(pData->m_aTargets);
#ifndef GPU_COMPILE
         EBM_ASSERT(nullptr != pTargetData);
#endif // GPU_COMPILE
      }

      FloatFast * pGradientAndHessian;
      if(bKeepGradHess) {
         pGradientAndHessian = reinterpret_cast<FloatFast *>(pData->m_aGradientsAndHessians);
#ifndef GPU_COMPILE
         EBM_ASSERT(nullptr != pGradientAndHessian);
#endif // GPU_COMPILE
      }

      const FloatFast * pWeight;
      if(bWeight) {
         pWeight = reinterpret_cast<const FloatFast *>(pData->m_aWeights);
#ifndef GPU_COMPILE
         EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
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
   }
};
