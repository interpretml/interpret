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
   static ErrorEbm StaticApplyUpdate(const Objective * const pThis, ApplyUpdateBridge * const pData) {
      return (static_cast<const RmseRegressionObjective<TFloat> *>(pThis))->ParentApplyUpdate<const RmseRegressionObjective<TFloat>, TFloat>(pData);
   }
   template<typename T = void, typename std::enable_if<TFloat::bCpu, T>::type * = nullptr>
   static double StaticFinishMetric(const Objective * const pThis, const double metricSum) {
      return (static_cast<const RmseRegressionObjective<TFloat> *>(pThis))->FinishMetric(metricSum);
   }
   void FillWrapper(void * const pWrapperOut) noexcept {
      FillObjectiveWrapper<RmseRegressionObjective, TFloat>(pWrapperOut);
   }

   inline RmseRegressionObjective(const Config & config) {
      if(1 != config.cOutputs) {
         throw ParamMismatchWithConfigException();
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
      static_assert(bKeepGradHess, "for RMSE regression we should always keep the gradients");

      static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == cCompilerPack;

      const FloatFast * const aUpdateTensorScores = reinterpret_cast<const FloatFast *>(pData->m_aUpdateTensorScores);

      const size_t cSamples = pData->m_cSamples;

      FloatFast * pGradient = reinterpret_cast<FloatFast *>(pData->m_aGradientsAndHessians); // no hessians for regression
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
         const ptrdiff_t cPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pData->m_cPack);

         const size_t cItemsPerBitPack = static_cast<size_t>(cPack);

         cBitsPerItemMax = GetCountBits<StorageDataType>(cItemsPerBitPack);

         cShift = static_cast<ptrdiff_t>((cSamples - 1) % cItemsPerBitPack * cBitsPerItemMax);
         cShiftReset = static_cast<ptrdiff_t>((cItemsPerBitPack - 1) * cBitsPerItemMax);

         maskBits = static_cast<size_t>(MakeLowMask<StorageDataType>(cBitsPerItemMax));

         pInputData = pData->m_aPacked;
      }

      const FloatFast * pWeight;
      if(bWeight) {
         pWeight = reinterpret_cast<const FloatFast *>(pData->m_aWeights);
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

            // for RMSE regression we cannot put the weight into the gradient like we could with other objectives
            // for regression or for classification because we only preserve the gradient and to calculate the
            // square error we need the original gradient and not the weight multiplied gradient... well we could
            // do it but it would require a division. A better way would be to have two FloatFast arrays: a 
            // non-weight adjusted one and a weight adjusted one for when inner bags are used
            // NOTE: For interactions we can and do put the weight into the gradient because we never update it
            const FloatFast gradient = EbmStats::ComputeGradientRegressionRmseFromOriginalGradient(*pGradient) + updateScore;
            *pGradient = gradient;
            ++pGradient;

            if(bCalcMetric) {
               FloatFast sampleSquaredError = EbmStats::ComputeSingleSampleSquaredErrorRegressionFromGradient(gradient);

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
   }
};
