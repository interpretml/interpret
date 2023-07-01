// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// Do not use this file as a reference for other objectives. LogLoss is special.

template<typename TFloat>
struct LogLossBinaryObjective final : public BinaryObjective {
   OBJECTIVE_CONSTANTS_BOILERPLATE(LogLossBinaryObjective, MINIMIZE_METRIC, Link_logit)

   inline LogLossBinaryObjective(const Config & config) {
      if(1 != config.cOutputs) {
         // we share the tag "log_loss" with multiclass classification
         throw SkipRegistrationException();
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
      // This function is here to signal the LogLossBinaryObjective class abilities, but it will not be called
      UNUSED(score);
      UNUSED(target);
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat score, const TFloat target) const noexcept {
      // This function is here to signal the LogLossBinaryObjective class abilities, but it will not be called
      UNUSED(score);
      UNUSED(target);
      return 0.0;
   }

   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat score, const TFloat target) const noexcept {
      // This function is here to signal the LogLossBinaryObjective class abilities, but it will not be called
      UNUSED(score);
      UNUSED(target);
      return GradientHessian<TFloat>(0.0, 0.0);
   }

   template<size_t cCompilerScores, bool bKeepGradHess, bool bCalcMetric, bool bWeight, bool bHessian, ptrdiff_t cCompilerPack>
   GPU_DEVICE void InjectedApplyUpdate(ApplyUpdateBridge * const pData) const {
      static_assert(k_oneScore == cCompilerScores, "We special case the classifiers so do not need to handle them");
      static_assert(!bKeepGradHess || !bCalcMetric, "bKeepGradHess and bCalcMetric cannot both be true");
      static_assert(bKeepGradHess || !bHessian, "bHessian can only be true if bKeepGradHess is true");
      static_assert(bCalcMetric || !bWeight, "bWeight can only be true if bCalcMetric is true");

      static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == cCompilerPack;
      static constexpr bool bGetTarget = bCalcMetric || bKeepGradHess;

#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pData);
      EBM_ASSERT(nullptr != pData->m_aUpdateTensorScores);
      EBM_ASSERT(1 <= pData->m_cSamples);
      EBM_ASSERT(0 == pData->m_cSamples % TFloat::k_cSIMDPack);
      EBM_ASSERT(nullptr != pData->m_aSampleScores);
      EBM_ASSERT(1 == pData->m_cScores);
#endif // GPU_COMPILE

      const typename TFloat::T * const aUpdateTensorScores = reinterpret_cast<const typename TFloat::T *>(pData->m_aUpdateTensorScores);

      const size_t cSamples = pData->m_cSamples;

      typename TFloat::T * pSampleScore = reinterpret_cast<typename TFloat::T *>(pData->m_aSampleScores);
      const typename TFloat::T * const pSampleScoresEnd = pSampleScore + cSamples;

      int cBitsPerItemMax;
      int cShift;
      int cShiftReset;
      typename TFloat::TInt maskBits;
      const typename TFloat::TInt::T * pInputData;

      TFloat updateScore;

      if(bCompilerZeroDimensional) {
         updateScore = aUpdateTensorScores[0];
      } else {
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
            if(!bCompilerZeroDimensional) {
               const typename TFloat::TInt iTensorBin = (iTensorBinCombined >> cShift) & maskBits;
               updateScore = TFloat::Load(aUpdateTensorScores, iTensorBin);
            }

            typename TFloat::TInt target;
            if(bGetTarget) {
               target = TFloat::TInt::Load(pTargetData);
               pTargetData += TFloat::TInt::k_cSIMDPack;
            }

            TFloat sampleScore = TFloat::Load(pSampleScore);
            sampleScore += updateScore;
            sampleScore.Store(pSampleScore);
            pSampleScore += TFloat::k_cSIMDPack;

            if(bKeepGradHess) {

               // gradient will be 0.0 if we perfectly predict the target with 100% certainty.  
               //    To do so, sampleScore would need to be either +infinity or -infinity
               // gradient will be +1.0 if actual value was 1 but we incorrectly predicted with 
               //    100% certainty that it was 0 by having sampleScore be -infinity
               // gradient will be -1.0 if actual value was 0 but we incorrectly predicted with 
               //    100% certainty that it was 1 by having sampleScore be +infinity
               //
               // gradient will be +0.5 if actual value was 1 but we were 50%/50% by having sampleScore be 0
               // gradient will be -0.5 if actual value was 0 but we were 50%/50% by having sampleScore be 0

               // TODO : In the future we'll sort our data by the target value and process them together, so we'll 
               //    know ahead of time if 0 == target or 1 == target.  This means we can pre-select the numerator
               //    to be either +1 or -1 as a template controlled constant that doesn't need to be runtime selected
               // TODO : In the future we'll sort our data by the target value and process them together. Once that
               //    happens we can eliminate the runtime check that can negate sampleScore AND we can also 
               //    avoid the negation itself by calling ExpForBinaryClassification with a templated parameter
               //    to use negative constants that will effectively take the exp of -sampleScore for no cost
               // 
               // !!! IMPORTANT: when using an approximate exp function, the formula used to compute the gradients becomes very
               //                important.  We want something that is balanced from positive to negative, which this version
               //                does IF the classes are roughly balanced since the positive or negative value is
               //                determined by only the target, unlike if we used a forumala that relied
               //                on the exp function returning a 1 at the 0 value, which our approximate exp does not give
               //                In time, you'd expect boosting to make targets with 0 more negative, leading to a positive
               //                term in the exp, and targets with 1 more positive, leading to a positive term in the exp
               //                So both classes get the same treatment in terms of errors in the exp function (both in the
               //                positive domain)
               //                We do still want the error of the positive cases and the error of the negative cases to
               //                sum to zero in the aggregate, so we want to choose our exp function to have average error
               //                sums of zero.
               //                I've made a copy of this formula as a comment to reference to what is good in-case the 
               //                formula is changed in the code without reading this comment
               //                const FLOAT gradient = (UNPREDICTABLE(0 == target) ? FloatFast { -1 } : FloatFast { 1 }) / (FloatFast{ 1 } + ExpForBinaryClassification(UNPREDICTABLE(0 == target) ? -sampleScore : sampleScore));
               // !!! IMPORTANT: SEE ABOVE

               const TFloat numerator = IfEqual(typename TFloat::TInt(0), target, TFloat(1), TFloat(-1));
               TFloat denominator = IfEqual(typename TFloat::TInt(0), target, -sampleScore, sampleScore);
               denominator = ApplyFunction(denominator, [](typename TFloat::T x) { return ExpForBinaryClassification<false>(x); });
               denominator += 1.0;
               const TFloat gradient = numerator / denominator;

               if(bHessian) {
                  // normally you would calculate the hessian from the class probability, but for classification it's possible
                  // to calculate from the gradient since our gradient is (r - p) where r is either 0 or 1, and our 
                  // hessian is p * (1 - p).  By taking the absolute value of (r - p) we're at a positive distance from either
                  // 0 or 1, and then we flip sides on "p" and "(1 - p)".  For binary classification this is useful since
                  // we can calcualte our gradient directly in a more exact way (especially when approximates are involved)
                  // and then calculate the hessian without subtracting from 1, which also introduces unbalanced floating point
                  // noise, unlike the more balanaced approach we're taking here


                  // Here are the hessian values for various gradient inputs (but this function in isolation isn't useful):
                  // -1     -> 0
                  // -0.999 -> 0.000999
                  // -0.5   -> 0.25
                  // -0.001 -> 0.000999
                  // 0      -> 0
                  // +0.001 -> 0.000999
                  // +0.5   -> 0.25
                  // +0.999 -> 0.000999
                  // +1     -> 0

                  // when we use this hessian term retuned inside ComputeSinglePartitionUpdate, if there was only
                  //   a single hessian term, or multiple similar ones, at the limit we'd get the following for the following inputs:
                  //   boosting is working propertly and we're close to zero error:
                  //     - slice_term_score_update = sumGradient / sumHessian => gradient / [gradient * (1 - gradient)] => 
                  //       gradient / [gradient * (1)] => +-1  but we multiply this by the learningRate of 0.01 (default), to get +-0.01               
                  //   when boosting is making a mistake, but is certain about it's prediction:
                  //     - slice_term_score_update = sumGradient / sumHessian => gradient / [gradient * (1 - gradient)] => +-1 / [1 * (0)] => 
                  //       +-infinity
                  //       but this shouldn't really happen inside the training set, because as the error gets bigger our boosting algorithm will correct corse by
                  //       updating in the opposite direction.  Divergence to infinity is a possibility in the validation set, but the training set pushes it's error to 
                  //       zero.  It may be possible to construct an adversarial dataset with negatively correlated features that cause a bouncing around that leads to 
                  //       divergence, but that seems unlikely in a normal dataset
                  //   our resulting function looks like this:
                  // 
                  //  small_term_score_update
                  //          |     *
                  //          |     *
                  //          |     *
                  //          |    * 
                  //          |   *  
                  //      0.01|*     
                  //          |      
                  //  -1-------------1--- gradient
                  //          |
                  //         *|-0.01
                  //      *   |
                  //     *    |
                  //    *     |
                  //    *     |
                  //    *     |
                  //
                  //   We have +-infinity asympotes at +-1
                  //   We have a discontinuity at 0, where we flip from positive to negative
                  //   the overall effect is that we train more on errors (error is +-1), and less on things with close to zero error


                  // !!! IMPORTANT: Newton-Raphson step, as illustrated in Friedman's original paper (https://statweb.stanford.edu/~jhf/ftp/trebst.pdf, page 9). Note that
                  //   they are using t * (2 - t) since they have a 2 in their objective

                  const TFloat absGradient = Abs(gradient);
                  const TFloat hessian = absGradient * (1.0 - absGradient);

                  gradient.Store(pGradientAndHessian);
                  hessian.Store(pGradientAndHessian + TFloat::k_cSIMDPack);
                  pGradientAndHessian += TFloat::k_cSIMDPack + TFloat::k_cSIMDPack;
               } else {
                  gradient.Store(pGradientAndHessian);
                  pGradientAndHessian += TFloat::k_cSIMDPack;
               }
            } else if(bCalcMetric) {
               // TODO: similar to the gradient calculation above, once we sort our data by the target values we
               //       will be able to pass all the targets==0 and target==1 in to a single call to this function
               //       and we can therefore template the target value.  We can then call ExpForBinaryClassification
               //       with a TEMPLATED parameter that indicates it if should negative sampleScore within the function
               //       This will eliminate both the IfEqual call, and also the negation, so it's a great optimization.
               
               TFloat metric = IfEqual(typename TFloat::TInt(0), target, sampleScore, -sampleScore);
               metric = ApplyFunction(metric, [](typename TFloat::T x) { return ExpForBinaryClassification<false>(x); });
               metric += 1.0;
               metric = ApplyFunction(metric, [](typename TFloat::T x) { return LogForLogLoss<false>(x); });

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
