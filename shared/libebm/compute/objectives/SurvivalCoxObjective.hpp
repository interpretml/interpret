// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// Cox Proportional Hazards partial likelihood objective for survival analysis.
//
// Target encoding: survival (time, event) is encoded in a single target value:
//   - Positive target = uncensored event time (time = target)
//   - Negative target = censored observation (time = -target)
//   - Zero target = invalid (rejected by CheckRegressionTarget)
//
// The Python wrapper encodes: target = time * (2*event - 1)
//
// This objective uses a custom InjectedApplyUpdate because Cox partial likelihood gradients
// depend on risk sets (all samples with time >= current time), making them inherently
// non-sample-independent. AccelerationFlags_NONE is required since the risk-set computation
// is sequential.
//
// Algorithm: Breslow partial likelihood with continuous tie handling.
// Link: identity (score = log hazard ratio directly).

template<typename TFloat> struct SurvivalCoxObjective : RegressionObjective {
   // The Breslow partial likelihood has global risk sets: each sample's gradient depends on all
   // other samples' scores and event times. The `true` below for k_bSingleSubsetRequired tells
   // the BoosterCore to keep all training samples in one un-chunked subset.
   OBJECTIVE_CONSTANTS_BOILERPLATE(SurvivalCoxObjective,
         MINIMIZE_METRIC,
         Objective_Other,
         Link_identity,
         true,
         false,
         true,
         k_cItemsPerBitPackUndefined,
         k_cItemsPerBitPackUndefined)

   inline SurvivalCoxObjective(const Config& config) {
      if(1 != config.cOutputs) {
         throw ParamMismatchWithConfigException();
      }
      if(config.isDifferentialPrivacy) {
         throw NonPrivateRegistrationException();
      }
   }

   inline bool CheckRegressionTarget(const double target) const noexcept {
      return std::isnan(target) || std::isinf(target) || 0.0 == target;
   }

   inline double LinkParam() const noexcept { return std::numeric_limits<double>::quiet_NaN(); }

   inline double LearningRateAdjustmentDifferentialPrivacy() const noexcept { return 1.0; }
   inline double LearningRateAdjustmentGradientBoosting() const noexcept { return 1.0; }
   inline double LearningRateAdjustmentHessianBoosting() const noexcept { return 1.0; }
   inline double GainAdjustmentGradientBoosting() const noexcept { return 1.0; }
   inline double GainAdjustmentHessianBoosting() const noexcept { return 1.0; }
   inline double GradientConstant() const noexcept { return 1.0; }
   inline double HessianConstant() const noexcept { return 1.0; }
   inline double FinishMetric(const double metricSum) const noexcept { return metricSum; }

   // These functions are here to signal class capabilities but will not be called since
   // InjectedApplyUpdate is overridden.
   GPU_DEVICE inline TFloat CalcMetric(const TFloat& score, const TFloat& target) const noexcept {
      UNUSED(score);
      UNUSED(target);
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat& score, const TFloat& target) const noexcept {
      UNUSED(score);
      UNUSED(target);
   }

   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(
         const TFloat& score, const TFloat& target) const noexcept {
      UNUSED(score);
      UNUSED(target);
   }

   template<bool bCollapsed,
         bool bValidation,
         bool bWeight,
         bool bHessian,
         bool bUseApprox,
         size_t cCompilerScores,
         int cCompilerPack>
   GPU_DEVICE NEVER_INLINE ErrorEbm InjectedApplyUpdate(ApplyUpdateBridge* const pData) const {
      static_assert(k_oneScore == cCompilerScores, "Cox regression should have one score");
      static_assert(!bValidation || !bHessian, "bHessian can only be true if bValidation is false");
      static_assert(bValidation || !bWeight, "bWeight can only be true if bValidation is true");
      static_assert(!bUseApprox, "Cox partial likelihood has no approximation");
      static_assert(k_cItemsPerBitPackUndefined == cCompilerPack);

      EBM_ASSERT(nullptr != pData);
      EBM_ASSERT(nullptr != pData->m_aUpdateTensorScores);
      EBM_ASSERT(1 <= pData->m_cSamples);
      EBM_ASSERT(0 == pData->m_cSamples % size_t{TFloat::k_cSIMDPack});
      EBM_ASSERT(nullptr != pData->m_aSampleScores);
      EBM_ASSERT(1 == pData->m_cScores);
      EBM_ASSERT(nullptr != pData->m_aTargets);

      const typename TFloat::T* const aUpdateTensorScores =
            reinterpret_cast<const typename TFloat::T*>(pData->m_aUpdateTensorScores);

      const size_t cSamples = pData->m_cSamples;

      typename TFloat::T* pSampleScore = reinterpret_cast<typename TFloat::T*>(pData->m_aSampleScores);
      const typename TFloat::T* const pSampleScoresEnd = pSampleScore + cSamples;

      int cBitsPerItemMax;
      int cShift;
      int cShiftReset;
      typename TFloat::TInt maskBits;
      const typename TFloat::TInt::T* pInputData;

      TFloat updateScore;

      if(bCollapsed) {
         updateScore = aUpdateTensorScores[0];
      } else {
         const int cItemsPerBitPack = pData->m_cPack;
         EBM_ASSERT(1 <= cItemsPerBitPack);
         EBM_ASSERT(cItemsPerBitPack <= COUNT_BITS(typename TFloat::TInt::T));

         cBitsPerItemMax = GetCountBits<typename TFloat::TInt::T>(cItemsPerBitPack);
         EBM_ASSERT(1 <= cBitsPerItemMax);
         EBM_ASSERT(cBitsPerItemMax <= COUNT_BITS(typename TFloat::TInt::T));

         maskBits = MakeLowMask<typename TFloat::TInt::T>(cBitsPerItemMax);

         pInputData = reinterpret_cast<const typename TFloat::TInt::T*>(pData->m_aPacked);
         EBM_ASSERT(nullptr != pInputData);

         cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;
         cShift = static_cast<int>((cSamples >> TFloat::k_cSIMDShift) % static_cast<size_t>(cItemsPerBitPack)) *
               cBitsPerItemMax;
         updateScore = TFloat::Load(aUpdateTensorScores, (TFloat::TInt::Load(pInputData) >> cShift) & maskBits);
         cShift -= cBitsPerItemMax;
         if(cShift < 0) {
            cShift = cShiftReset;
            pInputData += TFloat::TInt::k_cSIMDPack;
         }
      }

      do {
         typename TFloat::TInt iTensorBinCombined;
         if(!bCollapsed) {
            iTensorBinCombined = TFloat::TInt::Load(pInputData);
            pInputData += TFloat::TInt::k_cSIMDPack;
         }
         while(true) {
            TFloat sampleScore = TFloat::Load(pSampleScore);

            typename TFloat::TInt iTensorBin;
            if(!bCollapsed) {
               iTensorBin = (iTensorBinCombined >> cShift) & maskBits;
            }
            sampleScore += updateScore;
            if(!bCollapsed) {
               updateScore = TFloat::Load(aUpdateTensorScores, iTensorBin);
            }

            sampleScore.Store(pSampleScore);
            pSampleScore += TFloat::k_cSIMDPack;

            if(bCollapsed) {
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
         if(bCollapsed) {
            break;
         }
         cShift = cShiftReset;
      } while(pSampleScoresEnd != pSampleScore);

      // =====================================================================
      // Phase 2: Cox partial likelihood gradient/hessian or metric computation
      // =====================================================================

      const typename TFloat::T* const aScores = reinterpret_cast<const typename TFloat::T*>(pData->m_aSampleScores);
      const typename TFloat::T* const aTargets = reinterpret_cast<const typename TFloat::T*>(pData->m_aTargets);

      // Allocate temporary arrays
      const size_t cbScalar = sizeof(typename TFloat::T) * cSamples;
      const size_t cbIndex = sizeof(size_t) * cSamples;

      typename TFloat::T* const aTime = static_cast<typename TFloat::T*>(malloc(cbScalar));
      typename TFloat::T* const aExpScore = static_cast<typename TFloat::T*>(malloc(cbScalar));
      typename TFloat::T* const aRiskSetSum = static_cast<typename TFloat::T*>(malloc(cbScalar));
      size_t* const aSortedIdx = static_cast<size_t*>(malloc(cbIndex));

      if(nullptr == aTime || nullptr == aExpScore || nullptr == aRiskSetSum || nullptr == aSortedIdx) {
         free(aTime);
         free(aExpScore);
         free(aRiskSetSum);
         free(aSortedIdx);
         return Error_OutOfMemory;
      }

      // Decode targets and compute exp(score)
      for(size_t i = 0; i < cSamples; ++i) {
         const typename TFloat::T target = aTargets[i];
         aTime[i] = target > typename TFloat::T{0} ? target : -target;
         aExpScore[i] = std::exp(static_cast<double>(aScores[i]));
         aSortedIdx[i] = i;
      }

      // Heapsort aSortedIdx by ascending aTime values
      if(size_t{1} < cSamples) {
         // Build max heap
         for(size_t k = cSamples / 2; k > 0; --k) {
            size_t parent = k - 1;
            while(true) {
               size_t largest = parent;
               const size_t left = 2 * parent + 1;
               const size_t right = 2 * parent + 2;
               if(left < cSamples && aTime[aSortedIdx[left]] > aTime[aSortedIdx[largest]]) {
                  largest = left;
               }
               if(right < cSamples && aTime[aSortedIdx[right]] > aTime[aSortedIdx[largest]]) {
                  largest = right;
               }
               if(largest == parent) {
                  break;
               }
               const size_t tmp = aSortedIdx[parent];
               aSortedIdx[parent] = aSortedIdx[largest];
               aSortedIdx[largest] = tmp;
               parent = largest;
            }
         }
         // Extract elements from heap in sorted order
         for(size_t end = cSamples - 1; end > 0; --end) {
            const size_t tmp = aSortedIdx[0];
            aSortedIdx[0] = aSortedIdx[end];
            aSortedIdx[end] = tmp;
            // Sift down the new root in the reduced heap
            size_t parent = 0;
            while(true) {
               size_t largest = parent;
               const size_t left = 2 * parent + 1;
               const size_t right = 2 * parent + 2;
               if(left < end && aTime[aSortedIdx[left]] > aTime[aSortedIdx[largest]]) {
                  largest = left;
               }
               if(right < end && aTime[aSortedIdx[right]] > aTime[aSortedIdx[largest]]) {
                  largest = right;
               }
               if(largest == parent) {
                  break;
               }
               const size_t tmp2 = aSortedIdx[parent];
               aSortedIdx[parent] = aSortedIdx[largest];
               aSortedIdx[largest] = tmp2;
               parent = largest;
            }
         }
      }

      // Backward pass: compute cumulative risk set sums (largest to smallest time)
      {
         typename TFloat::T riskSum = typename TFloat::T{0};
         for(size_t jRev = cSamples; jRev > 0; --jRev) {
            const size_t j = jRev - 1;
            riskSum += aExpScore[aSortedIdx[j]];
            aRiskSetSum[j] = riskSum;
         }
      }

      if(bValidation) {
         // Compute negative partial log-likelihood as the validation metric
         typename TFloat::T metric = typename TFloat::T{0};

         const typename TFloat::T* pWeight = nullptr;
         if(bWeight) {
            pWeight = reinterpret_cast<const typename TFloat::T*>(pData->m_aWeights);
            EBM_ASSERT(nullptr != pWeight);
         }

         for(size_t j = 0; j < cSamples; ++j) {
            const size_t i = aSortedIdx[j];
            if(aTargets[i] > typename TFloat::T{0}) {
               // This sample is an uncensored event
               typename TFloat::T contribution = std::log(static_cast<double>(aRiskSetSum[j])) - aScores[i];
               if(bWeight) {
                  contribution *= pWeight[i];
               }
               metric += contribution;
            }
         }

         pData->m_metricOut += static_cast<double>(metric);
      } else {
         // Forward pass: compute gradients (and hessians if needed) using Breslow method
         //
         // For each sample i at sorted position j:
         //   gradient[i] = exp(score[i]) * cumH - event[i]
         //   hessian[i]  = exp(score[i]) * cumH - exp(score[i])^2 * cumH2
         //
         // where cumH  = sum over prior events k: 1/riskSetSum[k]
         //       cumH2 = sum over prior events k: 1/riskSetSum[k]^2

         typename TFloat::T* pGradientAndHessian =
               reinterpret_cast<typename TFloat::T*>(pData->m_aGradientsAndHessians);
         EBM_ASSERT(nullptr != pGradientAndHessian);

         const size_t cStride =
               bHessian ? size_t{TFloat::k_cSIMDPack} + size_t{TFloat::k_cSIMDPack} : size_t{TFloat::k_cSIMDPack};

         typename TFloat::T cumH = typename TFloat::T{0};
         typename TFloat::T cumH2 = typename TFloat::T{0};

         for(size_t j = 0; j < cSamples; ++j) {
            const size_t i = aSortedIdx[j];
            const bool bIsEvent = aTargets[i] > typename TFloat::T{0};

            if(bIsEvent) {
               const typename TFloat::T invRisk = typename TFloat::T{1} / aRiskSetSum[j];
               cumH += invRisk;
               cumH2 += invRisk * invRisk;
            }

            const typename TFloat::T gradient =
                  aExpScore[i] * cumH - (bIsEvent ? typename TFloat::T{1} : typename TFloat::T{0});

            // Write to sample i's position in the gradient/hessian array
            typename TFloat::T* const pOut = pGradientAndHessian + i * cStride;
            const TFloat gradientFloat = gradient;
            gradientFloat.Store(pOut);

            if(bHessian) {
               const typename TFloat::T hessian = aExpScore[i] * cumH - aExpScore[i] * aExpScore[i] * cumH2;
               const TFloat hessianFloat = hessian;
               hessianFloat.Store(pOut + TFloat::k_cSIMDPack);
            }
         }
      }

      free(aTime);
      free(aExpScore);
      free(aRiskSetSum);
      free(aSortedIdx);
      return Error_None;
   }
};
