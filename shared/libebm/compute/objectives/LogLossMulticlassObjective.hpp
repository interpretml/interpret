// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// Do not use this file as a reference for other objectives. LogLoss is special.

template<typename TFloat>
struct LogLossMulticlassObjective final : public MulticlassObjective {
   OBJECTIVE_CONSTANTS_BOILERPLATE(LogLossMulticlassObjective, MINIMIZE_METRIC, Link_logit)

   inline LogLossMulticlassObjective(const Config & config) {
      if(1 == config.cOutputs) {
         // we share the tag "log_loss" with binary classification
         throw SkipRegistrationException();
      }

      if(config.cOutputs <= 0) {
         throw ParamMismatchWithConfigException();
      }

      if(config.isDifferentiallyPrivate) {
         throw NonPrivateRegistrationException();
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
      // This function is here to signal the LogLossMulticlassObjective class abilities, but it will not be called
      UNUSED(score);
      UNUSED(target);
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat score, const TFloat target) const noexcept {
      // This function is here to signal the LogLossMulticlassObjective class abilities, but it will not be called
      UNUSED(score);
      UNUSED(target);
      return 0.0;
   }

   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat score, const TFloat target) const noexcept {
      // This function is here to signal the LogLossMulticlassObjective class abilities, but it will not be called
      UNUSED(score);
      UNUSED(target);
      return GradientHessian<TFloat>(0.0, 0.0);
   }

   template<size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bKeepGradHess, bool bCalcMetric, bool bWeight, bool bHessian>
   GPU_DEVICE void InjectedApplyUpdate(ApplyUpdateBridge * const pData) const {
      static_assert(bCalcMetric || !bWeight, "bWeight can only be true if bCalcMetric is true");

      static constexpr bool bDynamic = k_dynamicScores == cCompilerScores;
      static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == cCompilerPack;
      static constexpr bool bGetExp = bCalcMetric || bKeepGradHess;
      static constexpr bool bGetTarget = bCalcMetric || bKeepGradHess;

#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pData);
      EBM_ASSERT(nullptr != pData->m_aMulticlassMidwayTemp);
#endif // GPU_COMPILE

      FloatFast aLocalExpVector[bDynamic ? size_t { 1 } : cCompilerScores];
      FloatFast * aExps;
      if(bGetExp) {
         if(bDynamic) {
            aExps = reinterpret_cast<FloatFast *>(pData->m_aMulticlassMidwayTemp);
         } else {
            aExps = aLocalExpVector;
         }
      }

      const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pData->m_cScores);

      const FloatFast * const aUpdateTensorScores = reinterpret_cast<const FloatFast *>(pData->m_aUpdateTensorScores);

      const size_t cSamples = pData->m_cSamples;

      FloatFast * pSampleScore = reinterpret_cast<FloatFast *>(pData->m_aSampleScores);
      const FloatFast * const pSampleScoresEnd = pSampleScore + cSamples * cScores;

#ifndef GPU_COMPILE
      EBM_ASSERT(3 <= cScores);
      EBM_ASSERT(nullptr != aUpdateTensorScores);
      EBM_ASSERT(1 <= cSamples);
      EBM_ASSERT(nullptr != pSampleScore);
#endif // GPU_COMPILE

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
      FloatFast sumLogLoss;
      if(bCalcMetric) {
         if(bWeight) {
            pWeight = reinterpret_cast<const FloatFast *>(pData->m_aWeights);
#ifndef GPU_COMPILE
            EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
         }
         sumLogLoss = 0.0;
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

               if(bWeight) {
                  FloatFast weight;
                  weight = *pWeight;
                  ++pWeight;
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
