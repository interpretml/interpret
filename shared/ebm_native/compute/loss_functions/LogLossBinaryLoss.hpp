// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!

// Do not use this file as a reference for other loss functions. LogLoss is special.

// DO NOT INCLUDE ANY FILES IN THIS FILE. THEY WILL NOT BE ZONED PROPERLY

template<typename TFloat>
struct LogLossBinaryLoss final : public BinaryLoss {
   static constexpr bool k_bMse = false;
   LOSS_CLASS_CONSTANTS_BOILERPLATE(true)
   LOSS_CLASS_VIRTUAL_BOILERPLATE(LogLossBinaryLoss)

   inline LogLossBinaryLoss(const Config & config) {
      if(1 != config.cOutputs) {
         // we share the tag "log_loss" with multiclass classification
         throw SkipRegistrationException();
      }
   }

   inline double GetFinalMultiplier() const noexcept {
      return 1.0;
   }

   template<size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   GPU_DEVICE void InteriorApplyUpdateTemplated(ApplyUpdateBridge * const pData) const {
      static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == cCompilerPack;
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
         const ptrdiff_t cPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pData->m_cPack);
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
   }
};
