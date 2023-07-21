// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// Do not use this file as a reference for other objectives. LogLoss is special.

template<typename TFloat>
struct LogLossMulticlassObjective : MulticlassObjective {
   OBJECTIVE_CONSTANTS_BOILERPLATE(LogLossMulticlassObjective, MINIMIZE_METRIC, Link_logit, k_cItemsPerBitPackDynamic, k_cItemsPerBitPackDynamic)

   inline LogLossMulticlassObjective(const Config & config) {
      if(1 == config.cOutputs) {
         // we share the tag "log_loss" with binary classification
         throw SkipRegistrationException();
      }

      if(config.cOutputs <= 0) {
         throw ParamMismatchWithConfigException();
      }

      if(config.isDifferentialPrivacy) {
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

   GPU_DEVICE inline TFloat CalcMetric(const TFloat & score, const TFloat & target) const noexcept {
      // This function is here to signal the LogLossMulticlassObjective class abilities, but it will not be called
      UNUSED(score);
      UNUSED(target);
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat & score, const TFloat & target) const noexcept {
      // This function is here to signal the LogLossMulticlassObjective class abilities, but it will not be called
      UNUSED(score);
      UNUSED(target);
      return 0.0;
   }

   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat & score, const TFloat & target) const noexcept {
      // This function is here to signal the LogLossMulticlassObjective class abilities, but it will not be called
      UNUSED(score);
      UNUSED(target);
      return GradientHessian<TFloat>(0.0, 0.0);
   }

   template<size_t cCompilerScores, bool bValidation, bool bWeight, bool bHessian, int cCompilerPack>
   GPU_DEVICE NEVER_INLINE void InjectedApplyUpdate(ApplyUpdateBridge * const pData) const {
      static_assert(k_dynamicScores == cCompilerScores || 2 <= cCompilerScores, "Multiclass needs more than 1 score");
      static_assert(!bValidation || !bHessian, "bHessian can only be true if bValidation is false");
      static_assert(bValidation || !bWeight, "bWeight can only be true if bValidation is true");

      static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == cCompilerPack;
      static constexpr bool bDynamic = k_dynamicScores == cCompilerScores;

#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pData);
      EBM_ASSERT(nullptr != pData->m_aUpdateTensorScores);
      EBM_ASSERT(1 <= pData->m_cSamples);
      EBM_ASSERT(0 == pData->m_cSamples % TFloat::k_cSIMDPack);
      EBM_ASSERT(nullptr != pData->m_aSampleScores);
      EBM_ASSERT(2 <= pData->m_cScores);
      EBM_ASSERT(k_dynamicScores == cCompilerScores || cCompilerScores == pData->m_cScores);
      EBM_ASSERT(nullptr != pData->m_aMulticlassMidwayTemp);
      EBM_ASSERT(nullptr != pData->m_aTargets);
#endif // GPU_COMPILE

      alignas(alignof(TFloat)) typename TFloat::T
         aLocalExpVector[bDynamic ? size_t { 1 } : (cCompilerScores * size_t { TFloat::k_cSIMDPack })];
      typename TFloat::T * const aExps = bDynamic ? 
         reinterpret_cast<typename TFloat::T *>(pData->m_aMulticlassMidwayTemp) : aLocalExpVector;

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
         const int cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pData->m_cPack);
#ifndef GPU_COMPILE
         EBM_ASSERT(k_cItemsPerBitPackNone != cItemsPerBitPack); // we require this condition to be templated
         EBM_ASSERT(1 <= cItemsPerBitPack);
         EBM_ASSERT(cItemsPerBitPack <= COUNT_BITS(typename TFloat::TInt::T));
#endif // GPU_COMPILE

         cBitsPerItemMax = GetCountBits<typename TFloat::TInt::T>(cItemsPerBitPack);
#ifndef GPU_COMPILE
         EBM_ASSERT(1 <= cBitsPerItemMax);
         EBM_ASSERT(cBitsPerItemMax <= COUNT_BITS(typename TFloat::TInt::T));
#endif // GPU_COMPILE

         cShift = static_cast<int>(((cSamples >> TFloat::k_cSIMDShift) - size_t { 1 }) % static_cast<size_t>(cItemsPerBitPack)) * cBitsPerItemMax;
         cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;

         maskBits = MakeLowMask<typename TFloat::TInt::T>(cBitsPerItemMax);

         pInputData = reinterpret_cast<const typename TFloat::TInt::T *>(pData->m_aPacked);
#ifndef GPU_COMPILE
         EBM_ASSERT(nullptr != pInputData);
#endif // GPU_COMPILE
      }

      const typename TFloat::TInt::T * pTargetData = reinterpret_cast<const typename TFloat::TInt::T *>(pData->m_aTargets);

      const typename TFloat::T * pWeight;
      TFloat metricSum;
      typename TFloat::T * pGradientAndHessian;
      if(bValidation) {
         if(bWeight) {
            pWeight = reinterpret_cast<const typename TFloat::T *>(pData->m_aWeights);
#ifndef GPU_COMPILE
            EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
         }
         metricSum = 0.0;
      } else {
         pGradientAndHessian = reinterpret_cast<typename TFloat::T *>(pData->m_aGradientsAndHessians);
#ifndef GPU_COMPILE
         EBM_ASSERT(nullptr != pGradientAndHessian);
#endif // GPU_COMPILE
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
               // TODO: this multiplication is expensive since there isn't a good SIMD multiplication until SSE 4.1
               // and even then it has high latency and cost.  We could avoid it entirely by changing the memory
               // layout of the tensor at aUpdateTensorScores.  If we made cScores separate tensors, where we put
               // all the updates for each class, then we could use the non-multiplied indexes to fetch the
               // tensor bins from the first class, then we would add cTensorBins * sizeof(TFloat) to the pointer
               // that is the base of each tensor.  This elimaintes all multiplication and we just need to add
               // the value in a register to a pointer each iteration.  It also reduces the amount of memory we
               // need to access each load, which might be an issue for some big tensors.  It also eliminates the
               // "iTensorBin += 1" instruction below since we'll be doing that to the pointer instead of the indexes
               iTensorBin = Multiply<typename TFloat::TInt, typename TFloat::TInt::T, k_dynamicScores != cCompilerScores && 1 != TFloat::k_cSIMDPack, static_cast<typename TFloat::TInt::T>(cCompilerScores)>(iTensorBin, cCastScores);
            }

            TFloat sumExp = 0.0;
            size_t iScore1 = 0;
            do {
               TFloat updateScore;
               if(!bCompilerZeroDimensional) {
                  updateScore = TFloat::Load(aUpdateTensorScores, iTensorBin);
                  iTensorBin = iTensorBin + 1;
               } else {
                  updateScore = aUpdateTensorScores[iScore1];
               }

               TFloat sampleScore = TFloat::Load(pSampleScore);
               sampleScore += updateScore;
               sampleScore.Store(pSampleScore);
               pSampleScore += TFloat::k_cSIMDPack;

               const TFloat oneExp = ApplyFunc([](typename TFloat::T x) { return ExpForMulticlass<false>(x); }, sampleScore);
               oneExp.Store(&aExps[iScore1 << TFloat::k_cSIMDShift]);
               sumExp += oneExp;

               ++iScore1;
            } while(cScores != iScore1);

            typename TFloat::TInt target = TFloat::TInt::Load(pTargetData);
            pTargetData += TFloat::TInt::k_cSIMDPack;

            if(bValidation) {
               // TODO: instead of writing the exp values to memory, since we just need 1 and the sum, 
               // we could use an if selector to keep only the one that matches our target and we don't need
               // to store (or re-load) from memory.  This also saves us a gathering load, which will be expensive
               // in latency

               target = target << TFloat::k_cSIMDShift;
               target = target + TFloat::TInt::MakeIndexes();

               // TODO: after we finish sorting our dataset, all the target values in this datasubset will be
               // identical, so instead of calling LoadScattered we'll be able to call LoadAligned
               const TFloat itemExp = TFloat::Load(aExps, target);
               const TFloat invertedProbability = FastApproxDivide(sumExp, itemExp);
               TFloat metric =
                  ApplyFunc([](typename TFloat::T x) { return LogForLogLoss<false>(x); }, invertedProbability);

               if(bWeight) {
                  const TFloat weight = TFloat::Load(pWeight);
                  pWeight += TFloat::k_cSIMDPack;
                  metricSum = FusedMultiplyAdd(metric, weight, metricSum);
               } else {
                  metricSum += metric;
               }
            } else {
               // this Reciprocal is fast and is more SIMD-able, but it does create some complications.
               // When sumExp gets somewhat large, arround +4.5 or above, then the sumExp can get to be something
               // in the order of +100.  The inverse of that is around 0.01. We can then later multiply a number
               // close to 100 (if one of the classes dominate a bin) by 0.01 and we can sometimes get a number just
               // slightly above 1.0.  It might just be 1.000001, but then to calculate the hessian we subtract from
               // 1.0, leaving us with a small negative number. We could potentially shift the scores of the classes
               // to make the dominant class have a score of 0, but that gives our approximate exp a skew that I
               // this is better left more randomized. I tink as long as we tollerate negative hessians by excluding
               // hessians below a certain value we're ok, since we probably want to ignore these low hessians anyways
               // and even if we wanted to continue boosting, the other classes will continue to be boosted on
               // and our main class will stop growing more positive around +5, which is a fairly big value anyways
               const TFloat sumExpInverted = FastApproxReciprocal(sumExp);

               size_t iScore2 = 0;
               do {
                  const TFloat itemExp = TFloat::Load(&aExps[iScore2 << TFloat::k_cSIMDShift]);
                  const TFloat gradient = itemExp * sumExpInverted;

                  if(bHessian) {
                     const TFloat hessian = FusedNegateMultiplyAdd(gradient, gradient, gradient);
                     gradient.Store(&pGradientAndHessian[iScore2 << (TFloat::k_cSIMDShift + 1)]);
                     hessian.Store(&pGradientAndHessian[(iScore2 << (TFloat::k_cSIMDShift + 1)) + TFloat::k_cSIMDPack]);
                  } else {
                     gradient.Store(&pGradientAndHessian[iScore2 << TFloat::k_cSIMDShift]);
                  }

                  ++iScore2;
               } while(cScores != iScore2);

               if(bHessian) {
                  target = target << (TFloat::k_cSIMDShift + 1);
               } else {
                  target = target << TFloat::k_cSIMDShift;
               }
               target = target + TFloat::TInt::MakeIndexes();

               // TODO: after we finish sorting our dataset, all the target values in this datasubset will be
               // identical, so instead of calling LoadScattered and SaveScattered we'll be able to call
               // LoadAligned and SaveAligned
               TFloat adjust = TFloat::Load(pGradientAndHessian, target);
               adjust -= 1.0;
               adjust.Store(pGradientAndHessian, target);

               pGradientAndHessian += cScores << (bHessian ? (TFloat::k_cSIMDShift + 1) : TFloat::k_cSIMDShift);
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

      if(bValidation) {
         pData->m_metricOut = static_cast<double>(Sum(metricSum));
      }
   }
};
