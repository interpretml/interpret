// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!

// Do not use this file as a reference for other loss functions. MSE is special.

// DO NOT INCLUDE ANY FILES IN THIS FILE. THEY WILL NOT BE ZONED PROPERLY

template<typename TFloat>
struct MseRegressionLoss final : public RegressionLoss {
public:
   static constexpr bool k_bMse = true;
   static constexpr bool k_bVectorized = true;
   static ErrorEbm ApplyUpdate(const Loss * const pThis, ApplyUpdateBridge * const pData) {
      return (static_cast<const MseRegressionLoss<TFloat> *>(pThis))->LossApplyUpdate<const MseRegressionLoss<TFloat>, TFloat>(pData);
   }
   void FillWrapper(void * const pWrapperOut) noexcept {
      LossFillWrapper<MseRegressionLoss, TFloat>(pWrapperOut);
   }

   inline MseRegressionLoss(const Config & config) {
      if(1 != config.cOutputs) {
         throw ParamMismatchWithConfigException();
      }
   }

   inline double GradientMultiple() const noexcept {
      return 1.0;
   }

   inline double HessianMultiple() const noexcept {
      return 1.0;
   }

   inline void CalcMetric(TFloat prediction, TFloat target, TFloat & metric) const noexcept {
      // This function is here to signal the loss class abilities, but it will not be called
      UNUSED(prediction);
      UNUSED(target);
      UNUSED(metric);
   }

   inline void CalcGrad(TFloat prediction, TFloat target, TFloat & gradient) const noexcept {
      // This function is here to signal the loss class abilities, but it will not be called
      UNUSED(prediction);
      UNUSED(target);
      UNUSED(gradient);
   }

   inline void CalcGradMetric(TFloat prediction, TFloat target, TFloat & gradient, TFloat & metric) const noexcept {
      // This function is here to signal the loss class abilities, but it will not be called
      UNUSED(prediction);
      UNUSED(target);
      UNUSED(gradient);
      UNUSED(metric);
   }

   template<size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   GPU_DEVICE void InteriorApplyUpdateTemplated(ApplyUpdateBridge * const pData) const {
      static_assert(bKeepGradHess, "for MSE regression we should always keep the gradients");

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
