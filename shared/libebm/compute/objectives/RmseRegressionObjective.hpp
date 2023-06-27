// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// Do not use this file as a reference for other objectives. RMSE is special.

template<typename TFloat>
struct RmseRegressionObjective final : public RegressionObjective {
public:
   static constexpr bool k_bRmse = true;
   static constexpr BoolEbm k_bMaximizeMetric = MINIMIZE_METRIC;
   static constexpr LinkEbm k_linkFunction = Link_identity;
   static constexpr OutputType k_outputType = GetOutputType(k_linkFunction);
   static ErrorEbm StaticApplyUpdate(const Objective * const pThis, ApplyUpdateBridge * const pData) {
      return (static_cast<const RmseRegressionObjective<TFloat> *>(pThis))->ParentApplyUpdate<const RmseRegressionObjective<TFloat>, TFloat>(pData);
   }
   template<typename T = void, typename std::enable_if<TFloat::bCpu, T>::type * = nullptr>
   static double StaticFinishMetric(const Objective * const pThis, const double metricSum) {
      return (static_cast<const RmseRegressionObjective<TFloat> *>(pThis))->FinishMetric(metricSum);
   }
   template<typename T = void, typename std::enable_if<TFloat::bCpu, T>::type * = nullptr>
   static BoolEbm StaticCheckTargets(const Objective * const pThis, const size_t c, const void * const aTargets) {
      return (static_cast<const RmseRegressionObjective<TFloat> *>(pThis))->ParentCheckTargets<const RmseRegressionObjective<TFloat>, TFloat>(c, aTargets);
   }
   void FillWrapper(void * const pWrapperOut) noexcept {
      FillObjectiveWrapper<RmseRegressionObjective, TFloat>(pWrapperOut);
   }

   inline RmseRegressionObjective(const Config & config) {
      if(1 != config.cOutputs) {
         throw ParamMismatchWithConfigException();
      }
   }

   inline bool CheckRegressionTarget(const double target) const noexcept {
      return std::isnan(target) || std::isinf(target);
   }

   inline double LinkParam() const noexcept {
      return std::numeric_limits<double>::quiet_NaN();
   }

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

   inline double GradientConstant() const noexcept {
      return 2.0;
   }

   inline double HessianConstant() const noexcept {
      return 2.0;
   }

   inline double FinishMetric(const double metricSum) const noexcept {
      // TODO for now we return mse in actual fact, but we don't really expose the final value in pyton
      // so it's academic at the moment. MSE and RMSE have the same ordering, so we early stop at essentially
      // the same time. MSE has the benefit of exactness between platforms since the sqrt function isn't
      // guaranteed to give the same results. Once we've implemented our own tailor series approximations
      // then we can get exactness between platforms and then there will be no reason not to expose
      // RMSE instead of MSE
      return metricSum;
      //return std::sqrt(metricSum); // finish the 'r' in 'rmse'
   }

   GPU_DEVICE inline TFloat CalcMetric(const TFloat score, const TFloat target) const noexcept {
      // This function is here to signal the RmseRegressionObjective class abilities, but it will not be called
      UNUSED(score);
      UNUSED(target);
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat score, const TFloat target) const noexcept {
      // This function is here to signal the RmseRegressionObjective class abilities, but it will not be called
      UNUSED(score);
      UNUSED(target);
      return 0.0;
   }


   template<size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   GPU_DEVICE void InjectedApplyUpdate(ApplyUpdateBridge * const pData) const {
      static_assert(k_oneScore == cCompilerScores, "for RMSE regression there should always be one score");
      static_assert(!bHessian, "for RMSE regression we should never need the hessians");
      static_assert(bKeepGradHess, "for RMSE regression we should always keep the gradients");

      static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == cCompilerPack;

#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pData);
      EBM_ASSERT(nullptr != pData->m_aUpdateTensorScores);
      EBM_ASSERT(1 <= pData->m_cSamples);
      EBM_ASSERT(nullptr != pData->m_aGradientsAndHessians);
#endif // GPU_COMPILE

      const typename TFloat::T * const aUpdateTensorScores = reinterpret_cast<const typename TFloat::T *>(pData->m_aUpdateTensorScores);

      const size_t cSamples = pData->m_cSamples;

      typename TFloat::T * pGradient = reinterpret_cast<typename TFloat::T *>(pData->m_aGradientsAndHessians); // no hessians for regression
      const typename TFloat::T * const pGradientsEnd = pGradient + cSamples;

      size_t cBitsPerItemMax;
      ptrdiff_t cShift;
      ptrdiff_t cShiftReset;
      size_t maskBits;
      const StorageDataType * pInputData;

      alignas(SIMD_BYTE_ALIGNMENT) typename TFloat::T updateScores[TFloat::k_cSIMDPack];
      TFloat updateScore;

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

      const typename TFloat::T * pWeight;
      if(bWeight) {
         pWeight = reinterpret_cast<const typename TFloat::T *>(pData->m_aWeights);
#ifndef GPU_COMPILE
         EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
      }

      TFloat metricSum;
      if(bCalcMetric) {
         metricSum = 0.0;
      }
      do {
         alignas(SIMD_BYTE_ALIGNMENT) StorageDataType iTensorBinCombined[TFloat::k_cSIMDPack];
         if(!bCompilerZeroDimensional) {
            // we store the already multiplied dimensional value in *pInputData
            for(int i = 0; i < TFloat::k_cSIMDPack; ++i) {
               iTensorBinCombined[i] = pInputData[i];
            }
            pInputData += TFloat::k_cSIMDPack;
         }
         while(true) {
            if(!bCompilerZeroDimensional) {
               // in later versions of SIMD there are scatter/gather intrinsics that do this in one operation
               for(int i = 0; i < TFloat::k_cSIMDPack; ++i) {
                  const size_t iTensorBin = static_cast<size_t>(iTensorBinCombined[i] >> cShift) & maskBits;
                  updateScores[i] = aUpdateTensorScores[iTensorBin];
               }
               updateScore.LoadAligned(updateScores);
            }

            // for RMSE regression we cannot put the weight into the gradient like we could with other objectives
            // for regression or for classification because we only preserve the gradient and to calculate the
            // square error we need the original gradient and not the weight multiplied gradient... well we could
            // do it but it would require a division. A better way would be to have two TFloat arrays: a 
            // non-weight adjusted one and a weight adjusted one for when inner bags are used
            // NOTE: For interactions we can and do put the weight into the gradient because we never update it

            TFloat gradient;
            gradient.LoadAligned(pGradient);
            gradient += updateScore;
            gradient.SaveAligned(pGradient);
            pGradient += TFloat::k_cSIMDPack;

            if(bCalcMetric) {
               // we use RMSE so get the squared error part here
               TFloat metric = gradient * gradient;
               if(bWeight) {
                  TFloat weight;
                  weight.LoadAligned(pWeight);
                  metric *= weight;
                  pWeight += TFloat::k_cSIMDPack;
               }
               metricSum += metric;
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
         pData->m_metricOut = static_cast<double>(Sum(metricSum));
      }
   }
};
