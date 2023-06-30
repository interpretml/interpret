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
      static_assert(k_dynamicScores == cCompilerScores || 2 <= cCompilerScores, "Multiclass needs more than 1 score");
      static_assert(!bKeepGradHess || !bCalcMetric, "bKeepGradHess and bCalcMetric cannot both be true");
      static_assert(bKeepGradHess || !bHessian, "bHessian can only be true if bKeepGradHess is true");
      static_assert(bCalcMetric || !bWeight, "bWeight can only be true if bCalcMetric is true");

      static constexpr bool bDynamic = k_dynamicScores == cCompilerScores;
      static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == cCompilerPack;
      static constexpr bool bGetExp = bCalcMetric || bKeepGradHess;
      static constexpr bool bGetTarget = bCalcMetric || bKeepGradHess;

#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pData);
      EBM_ASSERT(nullptr != pData->m_aMulticlassMidwayTemp);
      EBM_ASSERT(nullptr != pData->m_aUpdateTensorScores);
      EBM_ASSERT(1 <= pData->m_cSamples);
      EBM_ASSERT(0 == pData->m_cSamples % TFloat::k_cSIMDPack);
      EBM_ASSERT(nullptr != pData->m_aSampleScores);
      EBM_ASSERT(2 <= pData->m_cScores);
      EBM_ASSERT(k_dynamicScores == cCompilerScores || cCompilerScores == pData->m_cScores);
#endif // GPU_COMPILE

      alignas(SIMD_BYTE_ALIGNMENT) typename TFloat::T 
         aLocalExpVector[bDynamic ? size_t { 1 } : (cCompilerScores * size_t { TFloat::k_cSIMDPack })];
      typename TFloat::T * aExps;
      if(bGetExp) {
         if(bDynamic) {
            aExps = reinterpret_cast<typename TFloat::T *>(pData->m_aMulticlassMidwayTemp);
         } else {
            aExps = aLocalExpVector;
         }
      }

      const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pData->m_cScores);
      const typename TFloat::TInt::T cCastScores = static_cast<typename TFloat::TInt::T>(cScores);


      const typename TFloat::T * const aUpdateTensorScores = reinterpret_cast<const typename TFloat::T *>(pData->m_aUpdateTensorScores);

      const size_t cSamples = pData->m_cSamples;

      typename TFloat::T * pSampleScore = reinterpret_cast<typename TFloat::T *>(pData->m_aSampleScores);
      const typename TFloat::T * const pSampleScoresEnd = pSampleScore + cSamples * cScores;

      int cBitsPerItemMax;
      int cShift;
      int cShiftReset;
      typename TFloat::TInt maskBits;
      const typename TFloat::TInt::T * pInputData;

      if(!bCompilerZeroDimensional) {
         const ptrdiff_t cPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pData->m_cPack);
#ifndef GPU_COMPILE
         EBM_ASSERT(k_cItemsPerBitPackNone != cPack); // we require this condition to be templated
         EBM_ASSERT(1 <= cPack);
#endif // GPU_COMPILE

         const int cItemsPerBitPack = static_cast<int>(cPack);
#ifndef GPU_COMPILE
         EBM_ASSERT(1 <= cItemsPerBitPack);
         EBM_ASSERT(static_cast<size_t>(cItemsPerBitPack) <= CountBitsRequiredPositiveMax<typename TFloat::TInt::T>());
#endif // GPU_COMPILE

         cBitsPerItemMax = static_cast<int>(GetCountBits<typename TFloat::TInt::T>(static_cast<size_t>(cItemsPerBitPack)));
#ifndef GPU_COMPILE
         EBM_ASSERT(1 <= cBitsPerItemMax);
         EBM_ASSERT(static_cast<size_t>(cBitsPerItemMax) <= CountBitsRequiredPositiveMax<typename TFloat::TInt::T>());
#endif // GPU_COMPILE

         cShift = static_cast<int>((cSamples - size_t { 1 }) % static_cast<size_t>(cItemsPerBitPack)) * cBitsPerItemMax;
         cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;

         maskBits = MakeLowMask<typename TFloat::TInt::T>(cBitsPerItemMax);

         pInputData = reinterpret_cast<const typename TFloat::TInt::T *>(pData->m_aPacked);
#ifndef GPU_COMPILE
         EBM_ASSERT(nullptr != pInputData);
#endif // GPU_COMPILE
      }

      const typename TFloat::TInt::T * pTargetData;
      if(bGetTarget) {
         pTargetData = reinterpret_cast<const typename TFloat::TInt::T *>(pData->m_aTargets);
#ifndef GPU_COMPILE
         EBM_ASSERT(nullptr != pTargetData);
#endif // GPU_COMPILE
      }

      typename TFloat::T * pGradientAndHessian;
      if(bKeepGradHess) {
         pGradientAndHessian = reinterpret_cast<typename TFloat::T *>(pData->m_aGradientsAndHessians);
#ifndef GPU_COMPILE
         EBM_ASSERT(nullptr != pGradientAndHessian);
#endif // GPU_COMPILE
      }

      const typename TFloat::T * pWeight;
      TFloat metricSum;
      if(bCalcMetric) {
         if(bWeight) {
            pWeight = reinterpret_cast<const typename TFloat::T *>(pData->m_aWeights);
#ifndef GPU_COMPILE
            EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
         }
         metricSum = 0.0;
      }
      do {
         typename TFloat::TInt iTensorBinCombined;
         if(!bCompilerZeroDimensional) {
            iTensorBinCombined = TFloat::TInt::Load(pInputData);
            pInputData += TFloat::TInt::k_cSIMDPack;
         }
         while(true) {
            typename TFloat::TInt iTensorBin;
            if(!bCompilerZeroDimensional) {
               iTensorBin = (iTensorBinCombined >> cShift) & maskBits;
               iTensorBin *= cCastScores;
            }

            TFloat sumExp;
            if(bGetExp) {
               sumExp = 0.0;
            }
            size_t iScore1 = 0;
            do {
               TFloat updateScore;
               if(!bCompilerZeroDimensional) {
                  updateScore = TFloat::Load(aUpdateTensorScores, iTensorBin);
                  iTensorBin += 1;
               } else {
                  updateScore = aUpdateTensorScores[iScore1];
               }

               TFloat sampleScore = TFloat::Load(pSampleScore);
               sampleScore += updateScore;
               sampleScore.Store(pSampleScore);
               pSampleScore += TFloat::k_cSIMDPack;

               if(bGetExp) {
                  const TFloat oneExp = ApplyFunction(sampleScore, [](typename TFloat::T x) { return ExpForMulticlass<false>(x); });
                  sumExp += oneExp;
                  oneExp.Store(&aExps[iScore1 * TFloat::k_cSIMDPack]);
               }

               ++iScore1;
            } while(cScores != iScore1);

            typename TFloat::TInt target;
            if(bGetTarget) {
               target = TFloat::TInt::Load(pTargetData);
               pTargetData += TFloat::TInt::k_cSIMDPack;
            }

            if(bKeepGradHess) {
               const TFloat sumExpInverted = TFloat { 1.0 } / sumExp;

               size_t iScore2 = 0;
               do {
                  const TFloat itemExp = TFloat::Load(&aExps[iScore2 * TFloat::k_cSIMDPack]);
                  const TFloat gradient = itemExp * sumExpInverted;

                  if(bHessian) {
                     const TFloat hessian = gradient * (TFloat { 1.0 } - gradient);
                     gradient.Store(&pGradientAndHessian[iScore2 * (TFloat::k_cSIMDPack * 2)]);
                     hessian.Store(&pGradientAndHessian[iScore2 * (TFloat::k_cSIMDPack * 2) + TFloat::k_cSIMDPack]);
                  } else {
                     gradient.Store(&pGradientAndHessian[iScore2 * TFloat::k_cSIMDPack]);
                  }

                  ++iScore2;
               } while(cScores != iScore2);

               if(bHessian) {
                  target = target << (TFloat::k_cSIMDShift + 1);
               } else {
                  target = target << TFloat::k_cSIMDShift;
               }
               target += TFloat::TInt::MakeIndexes();

               // TODO: after we finish sorting our dataset, all the target values in this datasubset will be
               // identical, so instead of calling LoadScattered and SaveScattered we'll be able to call
               // LoadAligned and SaveAligned
               TFloat adjust = TFloat::Load(pGradientAndHessian, target);
               adjust -= 1.0;
               adjust.Store(pGradientAndHessian, target);

               if(bHessian) {
                  pGradientAndHessian += (TFloat::k_cSIMDPack + TFloat::k_cSIMDPack) * cScores;
               } else {
                  pGradientAndHessian += TFloat::k_cSIMDPack * cScores;
               }
            } else if(bCalcMetric) {
               target = target << TFloat::k_cSIMDShift;
               target += TFloat::TInt::MakeIndexes();

               // TODO: after we finish sorting our dataset, all the target values in this datasubset will be
               // identical, so instead of calling LoadScattered we'll be able to call LoadAligned
               const TFloat itemExp = TFloat::Load(aExps, target);
               const TFloat invertedProbability = sumExp / itemExp;
               TFloat metric =
                  ApplyFunction(invertedProbability, [](typename TFloat::T x) { return LogForLogLoss<false>(x); });

               if(bWeight) {
                  const TFloat weight = TFloat::Load(pWeight);
                  pWeight += TFloat::k_cSIMDPack;
                  metric *= weight;
               }
               metricSum += metric;
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
         pData->m_metricOut = static_cast<double>(Sum(metricSum));
      }
   }
};
