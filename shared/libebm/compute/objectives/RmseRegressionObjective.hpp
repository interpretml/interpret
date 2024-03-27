// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// Do not use this file as a reference for other objectives. RMSE is special.

template<typename TFloat> struct RmseRegressionObjective : RegressionObjective {
   using TFloatInternal = TFloat;
   static constexpr bool k_bRmse = true;
   static constexpr bool k_bApprox = false;
   static constexpr BoolEbm k_bMaximizeMetric = MINIMIZE_METRIC;
   static constexpr LinkEbm k_linkFunction = Link_identity;
   static constexpr TaskEbm k_task = IdentifyTask(k_linkFunction);
   static constexpr int k_cItemsPerBitPackMax = 64;
   static constexpr int k_cItemsPerBitPackMin = 1;
   static ErrorEbm StaticApplyUpdate(const Objective* const pThis, ApplyUpdateBridge* const pData) {
      return (static_cast<const RmseRegressionObjective<TFloat>*>(pThis))
            ->ParentApplyUpdate<const RmseRegressionObjective<TFloat>>(pData);
   }
   template<typename T = void, typename std::enable_if<AccelerationFlags_NONE == TFloat::k_zone, T>::type* = nullptr>
   static double StaticFinishMetric(const Objective* const pThis, const double metricSum) {
      return (static_cast<const RmseRegressionObjective<TFloat>*>(pThis))->FinishMetric(metricSum);
   }
   template<typename T = void, typename std::enable_if<AccelerationFlags_NONE == TFloat::k_zone, T>::type* = nullptr>
   static BoolEbm StaticCheckTargets(const Objective* const pThis, const size_t c, const void* const aTargets) {
      return (static_cast<const RmseRegressionObjective<TFloat>*>(pThis))
            ->ParentCheckTargets<const RmseRegressionObjective<TFloat>>(c, aTargets);
   }
   void FillWrapper(const AccelerationFlags zones, void* const pWrapperOut) noexcept {
      FillObjectiveWrapper<RmseRegressionObjective>(zones, pWrapperOut);
   }

   inline RmseRegressionObjective(const Config& config) {
      if(1 != config.cOutputs) {
         throw ParamMismatchWithConfigException();
      }
   }

   inline bool CheckRegressionTarget(const double target) const noexcept {
      return std::isnan(target) || std::isinf(target);
   }

   inline double LinkParam() const noexcept { return std::numeric_limits<double>::quiet_NaN(); }

   inline double LearningRateAdjustmentDifferentialPrivacy() const noexcept {
      // we follow the gradient adjustment for DP since we have a similar change in rate and we want to make
      // our results comparable. The DP paper uses this adjusted rate.

      // WARNING: do not change this rate without accounting for it in the privacy budget!
      return 0.5;
   }

   inline double LearningRateAdjustmentGradientBoosting() const noexcept {
      // the hessian is 2.0 for RMSE. If we change to gradient boosting we divide by the weight/count which is
      // normalized to 1, so we double the effective learning rate without this adjustment.  We want
      // gradient boosting and hessian boosting to have similar rates, and this adjustment makes it that way
      return 0.5;
   }

   inline double LearningRateAdjustmentHessianBoosting() const noexcept {
      // this is the reference point
      return 1.0;
   }

   inline double GainAdjustmentGradientBoosting() const noexcept {
      // the hessian is 2.0 for RMSE. If we change to gradient boosting we divide by the weight/count which is
      // normalized to 1, so we double the effective learning rate without this adjustment.  We want
      // gradient boosting and hessian boosting to have similar rates, and this adjustment makes it that way
      return 0.5;
   }

   inline double GainAdjustmentHessianBoosting() const noexcept {
      // this is the reference point
      return 1.0;
   }

   inline double GradientConstant() const noexcept { return 2.0; }

   inline double HessianConstant() const noexcept { return 2.0; }

   inline double FinishMetric(const double metricSum) const noexcept {
      // TODO for now we return mse in actual fact, but we don't really expose the final value in pyton
      // so it's academic at the moment. MSE and RMSE have the same ordering, so we early stop at essentially
      // the same time. MSE has the benefit of exactness between platforms since the sqrt function isn't
      // guaranteed to give the same results. Once we've implemented our own tailor series approximations
      // then we can get exactness between platforms and then there will be no reason not to expose
      // RMSE instead of MSE
      return metricSum;
      // return std::sqrt(metricSum); // finish the 'r' in 'rmse'
   }

   GPU_DEVICE inline TFloat CalcMetric(const TFloat& score, const TFloat& target) const noexcept {
      // This function is here to signal the RmseRegressionObjective class abilities, but it will not be called
      UNUSED(score);
      UNUSED(target);
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat& score, const TFloat& target) const noexcept {
      // This function is here to signal the RmseRegressionObjective class abilities, but it will not be called
      UNUSED(score);
      UNUSED(target);
   }

   template<bool bValidation,
         bool bWeight,
         bool bHessian,
         bool bDisableApprox,
         size_t cCompilerScores,
         int cCompilerPack>
   GPU_DEVICE NEVER_INLINE void InjectedApplyUpdate(ApplyUpdateBridge* const pData) const {
      static_assert(k_oneScore == cCompilerScores, "for RMSE regression there should always be one score");
      static_assert(!bHessian, "for RMSE regression we should never need the hessians");
      static_assert(bValidation || !bWeight, "bWeight can only be true if bValidation is true");
      static_assert(!bDisableApprox, "Approximations cannot be disabled on RMSE since there are none on RMSE");

      static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == cCompilerPack;
      static constexpr bool bFixedSizePack =
            k_cItemsPerBitPackNone != cCompilerPack && k_cItemsPerBitPackDynamic != cCompilerPack;

#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pData);
      EBM_ASSERT(nullptr != pData->m_aUpdateTensorScores);
      EBM_ASSERT(1 <= pData->m_cSamples);
      EBM_ASSERT(0 == pData->m_cSamples % size_t{TFloat::k_cSIMDPack});
      EBM_ASSERT(0 == pData->m_cSamples % size_t{(bFixedSizePack ? cCompilerPack : 1) * TFloat::k_cSIMDPack});
      EBM_ASSERT(nullptr == pData->m_aSampleScores);
      EBM_ASSERT(1 == pData->m_cScores);
      EBM_ASSERT(nullptr != pData->m_aGradientsAndHessians);
#endif // GPU_COMPILE

      const typename TFloat::T* const aUpdateTensorScores =
            reinterpret_cast<const typename TFloat::T*>(pData->m_aUpdateTensorScores);

      const size_t cSamples = pData->m_cSamples;

      typename TFloat::T* pGradient =
            reinterpret_cast<typename TFloat::T*>(pData->m_aGradientsAndHessians); // no hessians for regression
      const typename TFloat::T* const pGradientsEnd = pGradient + cSamples;

      int cBitsPerItemMax;
      int cShift;
      int cShiftReset;
      typename TFloat::TInt maskBits;
      const typename TFloat::TInt::T* pInputData;

      TFloat updateScore;

      int cItemsPerBitPack;
      if(bCompilerZeroDimensional) {
         updateScore = aUpdateTensorScores[0];
      } else {
         cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pData->m_cPack);
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

         if(!bFixedSizePack) {
            cShift = static_cast<int>(
                           ((cSamples >> TFloat::k_cSIMDShift) - size_t{1}) % static_cast<size_t>(cItemsPerBitPack)) *
                  cBitsPerItemMax;
         }
         cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;

         maskBits = MakeLowMask<typename TFloat::TInt::T>(cBitsPerItemMax);

         pInputData = reinterpret_cast<const typename TFloat::TInt::T*>(pData->m_aPacked);
#ifndef GPU_COMPILE
         EBM_ASSERT(nullptr != pInputData);
#endif // GPU_COMPILE
      }

      const typename TFloat::T* pWeight;
      TFloat metricSum;
      if(bValidation) {
         if(bWeight) {
            pWeight = reinterpret_cast<const typename TFloat::T*>(pData->m_aWeights);
#ifndef GPU_COMPILE
            EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
         }
         metricSum = 0.0;
      }
      do {
         // TODO: the speed of this loop can probably be improved by:
         //   1) fetch the score from memory (predictable load is fast)
         //   2) issue the gather operation FOR THE NEXT loop(unpredictable load is slow)
         //   3) move the fetched gather operation from the previous loop into a new register
         //   4) do the computation using the fetched score and updateScore from the previous loop iteration
         // This will allow the CPU to do the gathering operation in the background while it works on computation.
         // Probably we want to put the code below inside the loop into an inline function that we can call
         // either at the start during init or the end once the rest is done.. not sure which.

         typename TFloat::TInt iTensorBinCombined;
         if(!bCompilerZeroDimensional) {
            iTensorBinCombined = TFloat::TInt::Load(pInputData);
            pInputData += TFloat::TInt::k_cSIMDPack;
         }
         int i;
         if(bFixedSizePack) {
            i = 0;
            cShift = cShiftReset;
         }
         while(true) {
            if(!bCompilerZeroDimensional) {
               const typename TFloat::TInt iTensorBin = (iTensorBinCombined >> cShift) & maskBits;
               updateScore = TFloat::Load(aUpdateTensorScores, iTensorBin);
            }

            TFloat gradient = TFloat::Load(pGradient);
            gradient += updateScore;
            gradient.Store(pGradient);
            pGradient += TFloat::k_cSIMDPack;

            if(bValidation) {
               // we use RMSE so get the squared error part here
               if(bWeight) {
                  const TFloat weight = TFloat::Load(pWeight);
                  pWeight += TFloat::k_cSIMDPack;
                  metricSum = FusedMultiplyAdd(gradient * gradient, weight, metricSum);
               } else {
                  metricSum = FusedMultiplyAdd(gradient, gradient, metricSum);
               }
            }

            if(bCompilerZeroDimensional) {
               if(pGradientsEnd == pGradient) {
                  break;
               }
            } else {
               cShift -= cBitsPerItemMax;
               if(bFixedSizePack) {
                  ++i;
                  if(cItemsPerBitPack <= i) {
                     break;
                  }
               } else {
                  if(cShift < 0) {
                     break;
                  }
               }
            }
         }
         if(bCompilerZeroDimensional) {
            break;
         }
         if(!bFixedSizePack) {
            cShift = cShiftReset;
         }
      } while(pGradientsEnd != pGradient);

      if(bValidation) {
         pData->m_metricOut += static_cast<double>(Sum(metricSum));
      }
   }
};
