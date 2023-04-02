// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!

// Do not use this file as a reference for other loss functions. LogLoss is special.

// DO NOT INCLUDE ANY FILES IN THIS FILE. THEY WILL NOT BE ZONED PROPERLY

template<typename TFloat>
struct LogLossMulticlassLoss final : public MulticlassLoss {
   static constexpr bool k_bMse = false;
   LOSS_CLASS_CONSTANTS_BOILERPLATE(true)
   LOSS_CLASS_VIRTUAL_BOILERPLATE(LogLossMulticlassLoss)

   inline LogLossMulticlassLoss(const Config & config) {
      if(1 == config.cOutputs) {
         // we share the tag "log_loss" with binary classification
         throw SkipRegistrationException();
      }

      if(config.cOutputs <= 0) {
         throw ParamMismatchWithConfigException();
      }
   }

   inline double GetFinalMultiplier() const noexcept {
      return 1.0;
   }

   template<size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   GPU_DEVICE void InteriorApplyUpdateTemplated(ApplyUpdateBridge * const pData) const {
      static_assert(IsClassification(cCompilerClasses), "must be classification");
      static_assert(!IsBinaryClassification(cCompilerClasses), "must be multiclass");
      static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == cCompilerPack;
      static constexpr bool bGetExp = bCalcMetric || bKeepGradHess;
      static constexpr bool bGetTarget = bCalcMetric || bKeepGradHess;

      FloatFast aLocalExpVector[GetCountScores(cCompilerClasses)];
      FloatFast * aExps;
      if(bGetExp) {
         static constexpr bool bDynamic = k_dynamicClassification == cCompilerClasses;
         if(bDynamic) {
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
   }
};
