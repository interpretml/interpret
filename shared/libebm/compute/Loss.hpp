// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !!! NOTE: To add a new loss/objective function in C++, follow the steps listed at the top of the "Loss.cpp" file !!!

#ifndef LOSS_HPP
#define LOSS_HPP

#include <stddef.h> // size_t, ptrdiff_t
#include <memory> // shared_ptr, unique_ptr

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // INLINE_ALWAYS
#include "zones.h"

#include "zoned_bridge_cpp_functions.hpp" // FunctionPointersCpp
#include "compute.hpp" // GPU_GLOBAL

// Nomenclature used in this package:
// - objective: We can use any metric for early stopping, so our list of objectives is identical to the
//   list of metrics that we provide internally. Not all objectives are differentiable though, so for
//   some objectives we need to choose a reasonable differentiable loss function that we can optimize via 
//   gradient boosting. As an example, someone could request an 'auc' objective which uses a 
//   'log_loss' loss since 'auc' is not a differentiable function. This follows the catboost approach:
//   https://catboost.ai/en/docs/concepts/loss-functions
// - loss function: A differentiable cost/error function that we can use for optimization via gradient boosting
// - link function: For prediction we need the reverse/inverse link function, sometimes also known as 
//   the mean/activation(in NN) function. Multiple loss functions can share a single link function, so for 
//   simplicity we record the appropriate link function in our model since the original loss function 
//   and objectives are extraneous information when using the model to make predictions.  
// - In this package the choice of objective determines the loss function, which determines the link function.
//   If more flexibility is needed, custom objectives can be used.

struct ApplyUpdateBridge;

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class Registration;
typedef const std::vector<std::shared_ptr<const Registration>> (* REGISTER_LOSSES_FUNCTION)();

struct SingletaskLoss;
struct BinaryLoss;
struct MulticlassLoss;
struct RegressionLoss;

struct MultitaskLoss;
struct BinaryMultitaskLoss;
struct MulticlassMultitaskLoss;
struct RegressionMultitaskLoss;


template<typename TFloat>
class GradientHessian {
   TFloat m_gradient;
   TFloat m_hessian;

public:
   template<typename T1, typename T2>
   GPU_DEVICE inline GradientHessian(const T1 gradient, const T2 hessian) : 
      m_gradient(TFloat(gradient)), 
      m_hessian(TFloat(hessian)) {
   }

   GPU_DEVICE inline TFloat GetGradient() const noexcept { return m_gradient; }
   GPU_DEVICE inline TFloat GetHessian() const noexcept { return m_hessian; }
};
template<typename TFloat>
GPU_DEVICE inline GradientHessian<TFloat> MakeGradientHessian(const TFloat gradient, const TFloat hessian) {
   return GradientHessian<TFloat>(gradient, hessian);
}
template<typename TFloat>
GPU_DEVICE inline GradientHessian<TFloat> MakeGradientHessian(const TFloat gradient, const double hessian) {
   return GradientHessian<TFloat>(gradient, hessian);
}
template<typename TFloat>
GPU_DEVICE inline GradientHessian<TFloat> MakeGradientHessian(const double gradient, const TFloat hessian) {
   return GradientHessian<TFloat>(gradient, hessian);
}
template<typename TFloat>
GPU_DEVICE inline GradientHessian<TFloat> MakeGradientHessian(const double gradient, const double hessian) {
   return GradientHessian<TFloat>(gradient, hessian);
}


template<typename TLoss, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
GPU_GLOBAL static void RemoteApplyUpdate(const Loss * const pLoss, ApplyUpdateBridge * const pData) {
   const TLoss * const pLossSpecific = static_cast<const TLoss *>(pLoss);
   pLossSpecific->template InjectedApplyUpdate<cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
}


struct Registrable {
   // TODO: move this into its own file once we create Metric classes that are also Registrable
protected:
   Registrable() = default;
   ~Registrable() = default;
};

struct Loss : public Registrable {
private:

   // Welcome to the demented hall of mirrors.. a prison for your mind
   // And no, I did not make this to purposely torment you

   template<class TLoss, typename TFloat>
   struct HasHessianInternal {
      // use SFINAE to determine if TLoss has the function CalcGradientHessian with the correct signature

      template<typename T>
      static auto check(T * p) -> decltype(p->CalcGradientHessian(TFloat { 0 }, TFloat { 0 }), std::true_type());

      static std::false_type check(...);

      using internal_type = decltype(check(static_cast<typename std::remove_reference<TLoss>::type *>(nullptr)));
      static constexpr bool value = internal_type::value;
   };
   template<typename TLoss, typename TFloat>
   constexpr static bool HasHessian() {
      // use SFINAE to determine if TLoss has the function CalcGradientHessian with the correct signature
      return HasHessianInternal<TLoss, TFloat>::value;
   }

   template<typename TLoss>
   constexpr static bool IsEdgeLoss() {
      return
         std::is_base_of<BinaryLoss, TLoss>::value ||
         std::is_base_of<MulticlassLoss, TLoss>::value ||
         std::is_base_of<RegressionLoss, TLoss>::value ||
         std::is_base_of<BinaryMultitaskLoss, TLoss>::value ||
         std::is_base_of<MulticlassMultitaskLoss, TLoss>::value ||
         std::is_base_of<RegressionMultitaskLoss, TLoss>::value;
   }


   // if we have multiple scores AND multiple bitpacks, then we have two nested loops in our final function
   // and the compiler will only unroll the inner loop.  That inner loop will be for the scores, so there
   // is not much benefit in generating hard coded loop counts for the bitpacks, so short circut the
   // bit packing to use the dynamic value if we don't have the single bin case.  This also solves
   // part of our template blowup issue of having N * M starting point templates where N is the number
   // of scores and M is the number of bit packs.  If we use 8 * 16 that's already 128 copies of the
   // templated function at this point and more later.  Reducing this to just 16 is very very helpful.
   template<typename TLoss, typename TFloat, typename std::enable_if<!TLoss::IsMultiScore, void>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm TypeApplyUpdate(ApplyUpdateBridge * const pData) const {
      if(k_cItemsPerBitPackNone == pData->m_cPack) {
         return HessianApplyUpdate<TLoss, TFloat, k_oneScore, k_cItemsPerBitPackNone>(pData);
      } else {
         return BitPack<TLoss, TFloat, k_oneScore, k_cItemsPerBitPackMax>::Func(this, pData);
      }
   }
   template<typename TLoss, typename TFloat, typename std::enable_if<TLoss::IsMultiScore && std::is_base_of<MulticlassMultitaskLoss, TLoss>::value, void>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm TypeApplyUpdate(ApplyUpdateBridge * const pData) const {
      if(k_cItemsPerBitPackNone == pData->m_cPack) {
         // don't blow up our complexity if we have only 1 bin.. just use dynamic for the count of scores
         return HessianApplyUpdate<TLoss, TFloat, k_dynamicScores, k_cItemsPerBitPackNone>(pData);
      } else {
         // if our inner loop is dynamic scores, then the compiler won't do a full unwind of the bit pack
         // loop, so just short circuit it to using dynamic
         return HessianApplyUpdate<TLoss, TFloat, k_dynamicScores, k_cItemsPerBitPackDynamic>(pData);
      }
   }
   template<typename TLoss, typename TFloat, typename std::enable_if<TLoss::IsMultiScore && !std::is_base_of<MulticlassMultitaskLoss, TLoss>::value, void>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm TypeApplyUpdate(ApplyUpdateBridge * const pData) const {
      if(k_cItemsPerBitPackNone == pData->m_cPack) {
         // don't blow up our complexity if we have only 1 bin.. just use dynamic for the count of scores
         return HessianApplyUpdate<TLoss, TFloat, k_dynamicScores, k_cItemsPerBitPackNone>(pData);
      } else {
         return CountScores<TLoss, TFloat, (k_cCompilerScoresMax < k_cCompilerScoresStart ? k_dynamicScores : k_cCompilerScoresStart)>::Func(this, pData);
      }
   }


   template<typename TLoss, typename TFloat, size_t cCompilerScores>
   struct CountScores final {
      INLINE_ALWAYS static ErrorEbm Func(const Loss * const pLoss, ApplyUpdateBridge * const pData) {
         if(cCompilerScores == pData->m_cScores) {
            return pLoss->HessianApplyUpdate<TLoss, TFloat, cCompilerScores, k_cItemsPerBitPackDynamic>(pData);
         } else {
            return CountScores<TLoss, TFloat, k_cCompilerScoresMax == cCompilerScores ? k_dynamicScores : cCompilerScores + 1>::Func(pLoss, pData);
         }
      }
   };
   template<typename TLoss, typename TFloat>
   struct CountScores<TLoss, TFloat, k_dynamicScores> final {
      INLINE_ALWAYS static ErrorEbm Func(const Loss * const pLoss, ApplyUpdateBridge * const pData) {
         return pLoss->HessianApplyUpdate<TLoss, TFloat, k_dynamicScores, k_cItemsPerBitPackDynamic>(pData);
      }
   };


   // in our current format cCompilerScores will always be 1, but just in case we change our code to allow
   // for special casing multiclass with compile time unrolling of the compiler pack, leave cCompilerScores here
   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack>
   struct BitPack final {
      INLINE_ALWAYS static ErrorEbm Func(const Loss * const pLoss, ApplyUpdateBridge * const pData) {
         if(cCompilerPack == pData->m_cPack) {
            return pLoss->HessianApplyUpdate<TLoss, TFloat, cCompilerScores, cCompilerPack>(pData);
         } else {
            return BitPack<TLoss, TFloat, cCompilerScores, GetNextBitPack(cCompilerPack)>::Func(pLoss, pData);
         }
      }
   };
   template<typename TLoss, typename TFloat, size_t cCompilerScores>
   struct BitPack<TLoss, TFloat, cCompilerScores, k_cItemsPerBitPackLast> final {
      INLINE_ALWAYS static ErrorEbm Func(const Loss * const pLoss, ApplyUpdateBridge * const pData) {
         return pLoss->HessianApplyUpdate<TLoss, TFloat, cCompilerScores, k_cItemsPerBitPackLast>(pData);
      }
   };


   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, typename std::enable_if<HasHessian<TLoss, TFloat>(), void>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm HessianApplyUpdate(ApplyUpdateBridge * const pData) const {
      if(pData->m_bHessianNeeded) {
         return OptionsApplyUpdate<TLoss, TFloat, cCompilerScores, cCompilerPack, true>(pData);
      } else {
         return OptionsApplyUpdate<TLoss, TFloat, cCompilerScores, cCompilerPack, false>(pData);
      }
   }
   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, typename std::enable_if<!HasHessian<TLoss, TFloat>(), void>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm HessianApplyUpdate(ApplyUpdateBridge * const pData) const {
      return OptionsApplyUpdate<TLoss, TFloat, cCompilerScores, cCompilerPack, false>(pData);
   }


   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, typename std::enable_if<!TLoss::k_bMse, void>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm OptionsApplyUpdate(ApplyUpdateBridge * const pData) const {
      if(nullptr != pData->m_aGradientsAndHessians) {
         static constexpr bool bKeepGradHess = true;

         // if we are updating the gradients then we are doing training and do not need to calculate the metric
         EBM_ASSERT(!pData->m_bCalcMetric);
         static constexpr bool bCalcMetric = false;

         if(nullptr != pData->m_aWeights) {
            static constexpr bool bWeight = true;

            // this branch will only be taking during interaction initialization

            return OperatorApplyUpdate<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
         } else {
            static constexpr bool bWeight = false;
            return OperatorApplyUpdate<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
         }
      } else {
         static constexpr bool bKeepGradHess = false;

         if(pData->m_bCalcMetric) {
            static constexpr bool bCalcMetric = true;

            if(nullptr != pData->m_aWeights) {
               static constexpr bool bWeight = true;
               return OperatorApplyUpdate<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
            } else {
               static constexpr bool bWeight = false;
               return OperatorApplyUpdate<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
            }
         } else {
            static constexpr bool bCalcMetric = false;

            // currently this branch is not taken, but if would be if we wanted to allow in the future
            // non-metric calculating validation for boosting.  For instance if we wanted to substitute an alternate
            // metric or if for performance reasons we only want to calculate the metric every N rounds of boosting

            EBM_ASSERT(nullptr == pData->m_aWeights);
            static constexpr bool bWeight = false; // if we are not calculating the metric or updating gradients then we never need the weights

            return OperatorApplyUpdate<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
         }
      }
   }
   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, typename std::enable_if<TLoss::k_bMse, void>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm OptionsApplyUpdate(ApplyUpdateBridge * const pData) const {
      EBM_ASSERT(nullptr != pData->m_aGradientsAndHessians); // we always keep gradients for regression
      static constexpr bool bKeepGradHess = true;

      if(pData->m_bCalcMetric) {
         static constexpr bool bCalcMetric = true;

         if(nullptr != pData->m_aWeights) {
            static constexpr bool bWeight = true;
            return OperatorApplyUpdate<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
         } else {
            static constexpr bool bWeight = false;
            return OperatorApplyUpdate<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
         }
      } else {
         static constexpr bool bCalcMetric = false;

         EBM_ASSERT(nullptr == pData->m_aWeights);
         static constexpr bool bWeight = false; // if we are not calculating the metric then we never need the weights

         return OperatorApplyUpdate<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
      }
   }


   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   INLINE_RELEASE_TEMPLATED ErrorEbm OperatorApplyUpdate(ApplyUpdateBridge * const pData) const {
      return TFloat::template OperatorApplyUpdate<TLoss, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(this, pData);
   }

protected:

   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   GPU_DEVICE void ChildApplyUpdate(ApplyUpdateBridge * const pData) const {
      const TLoss * const pLoss = static_cast<const TLoss *>(this);

      static_assert(k_oneScore == cCompilerScores, "We special case the classifiers so do not need to handle them");
      static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == cCompilerPack;
      static constexpr bool bGetTarget = bCalcMetric || bKeepGradHess;

      const typename TFloat::T * const aUpdateTensorScores = reinterpret_cast<const typename TFloat::T *>(pData->m_aUpdateTensorScores);

      const size_t cSamples = pData->m_cSamples;

      typename TFloat::T * pSampleScore = reinterpret_cast<typename TFloat::T *>(pData->m_aSampleScores);
      const typename TFloat::T * const pSampleScoresEnd = pSampleScore + cSamples;

      size_t cBitsPerItemMax;
      ptrdiff_t cShift;
      ptrdiff_t cShiftReset;
      size_t maskBits;
      const StorageDataType * pInputData;

      alignas(16) typename TFloat::T updateScores[TFloat::cPack];
      TFloat updateScore;

      if(bCompilerZeroDimensional) {
         const typename TFloat::T singleScore = aUpdateTensorScores[0];
         for(int i = 0; i < TFloat::cPack; ++i) {
            updateScores[i] = singleScore;
         }
         updateScore.LoadAligned(updateScores);
      } else {
         const ptrdiff_t cPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pData->m_cPack);

         const size_t cItemsPerBitPack = static_cast<size_t>(cPack);

         cBitsPerItemMax = GetCountBits<StorageDataType>(cItemsPerBitPack);

         cShift = static_cast<ptrdiff_t>((cSamples - 1) % cItemsPerBitPack * cBitsPerItemMax);
         cShiftReset = static_cast<ptrdiff_t>((cItemsPerBitPack - 1) * cBitsPerItemMax);

         maskBits = static_cast<size_t>(MakeLowMask<StorageDataType>(cBitsPerItemMax));

         pInputData = pData->m_aPacked;
      }

      const typename TFloat::T * pTargetData;
      if(bGetTarget) {
         pTargetData = reinterpret_cast<const typename TFloat::T *>(pData->m_aTargets);
      }

      typename TFloat::T * pGradientAndHessian;
      if(bKeepGradHess) {
         pGradientAndHessian = reinterpret_cast<typename TFloat::T *>(pData->m_aGradientsAndHessians);
      }

      const typename TFloat::T * pWeight;
      if(bWeight) {
         pWeight = reinterpret_cast<const typename TFloat::T *>(pData->m_aWeights);
      }

      TFloat sumMetric;
      if(bCalcMetric) {
         sumMetric = 0.0;
      }
      do {
         alignas(16) StorageDataType iTensorBinCombined[TFloat::cPack];
         if(!bCompilerZeroDimensional) {
            // we store the already multiplied dimensional value in *pInputData
            for(int i = 0; i < TFloat::cPack; ++i) {
               iTensorBinCombined[i] = pInputData[i];
            }
            pInputData += TFloat::cPack;
         }
         while(true) {
            if(!bCompilerZeroDimensional) {
               // in later versions of SIMD there are scatter/gather intrinsics that do this in one operation
               for(int i = 0; i < TFloat::cPack; ++i) {
                  const size_t iTensorBin = static_cast<size_t>(iTensorBinCombined[i] >> cShift) & maskBits;
                  updateScores[i] = aUpdateTensorScores[iTensorBin];
               }
               updateScore.LoadAligned(updateScores);
            }

            TFloat target;
            if(bGetTarget) {
               target.LoadAligned(pTargetData);
               pTargetData += TFloat::cPack;
            }

            TFloat sampleScore;
            sampleScore.LoadAligned(pSampleScore);
            sampleScore += updateScore;
            sampleScore.SaveAligned(pSampleScore);
            pSampleScore += TFloat::cPack;

            TFloat weight;
            if(bWeight) {
               weight.LoadAligned(pWeight);
               pWeight += TFloat::cPack;
            }

            TFloat prediction;
            if(bGetTarget) {
               prediction = pLoss->InverseLinkFunction(sampleScore);
            }

            if(bKeepGradHess) {
               TFloat gradient;
               TFloat hessian;
               if(bHessian) {
                  const GradientHessian<TFloat> gradientHessian = pLoss->CalcGradientHessian(prediction, target);
                  gradient = gradientHessian.GetGradient();
                  hessian = gradientHessian.GetHessian();
               } else {
                  gradient = pLoss->CalcGradient(prediction, target);
               }
               if(bWeight) {
                  // This is only used during the initialization of interaction detection. For boosting
                  // we currently multiply by the weight during bin summation instead since we use the weight
                  // there to include the inner bagging counts of occurences.
                  // Whether this multiplication happens or not is controlled by the caller by passing in the
                  // weight array or not.
                  gradient *= weight;
                  hessian *= weight;
               }
               gradient.SaveAligned(pGradientAndHessian);
               pGradientAndHessian += TFloat::cPack;
               if(bHessian) {
                  hessian.SaveAligned(pGradientAndHessian);
                  pGradientAndHessian += TFloat::cPack;
               }
            }

            if(bCalcMetric) {
               TFloat metric = pLoss->CalcMetric(prediction, target);
               if(bWeight) {
                  metric *= weight;
               }
               sumMetric += metric;
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
         pData->m_metricOut = static_cast<double>(Sum(sumMetric));
      }
   }


   template<typename TLoss, typename TFloat>
   INLINE_RELEASE_TEMPLATED ErrorEbm ParentApplyUpdate(ApplyUpdateBridge * const pData) const {
      static_assert(IsEdgeLoss<TLoss>(), "TLoss must inherit from one of the children of the Loss class");
      return TypeApplyUpdate<TLoss, TFloat>(pData);
   }


   template<typename TLoss, typename TFloat>
   INLINE_RELEASE_TEMPLATED void FillLossWrapper(void * const pWrapperOut) noexcept {
      EBM_ASSERT(nullptr != pWrapperOut);
      LossWrapper * const pLossWrapperOut = static_cast<LossWrapper *>(pWrapperOut);
      FunctionPointersCpp * const pFunctionPointers =
         static_cast<FunctionPointersCpp *>(pLossWrapperOut->m_pFunctionPointersCpp);
      EBM_ASSERT(nullptr != pFunctionPointers);

      pFunctionPointers->m_pApplyUpdateCpp = &TLoss::StaticApplyUpdate;

      pLossWrapperOut->m_linkFunction = TLoss::k_linkFunction;

      auto linkParam = (static_cast<TLoss *>(this))->LinkParam();
      static_assert(std::is_same<decltype(linkParam), double>::value, "this->LinkParam() should return a double");
      pLossWrapperOut->m_linkParam = linkParam;

      auto gradientMultiple = (static_cast<TLoss *>(this))->GradientMultiple();
      static_assert(std::is_same<decltype(gradientMultiple), double>::value, "this->GradientMultiple() should return a double");
      auto hessianMultiple = (static_cast<TLoss *>(this))->HessianMultiple();
      static_assert(std::is_same<decltype(hessianMultiple), double>::value, "this->HessianMultiple() should return a double");

      pLossWrapperOut->m_gradientMultiple = gradientMultiple;
      pLossWrapperOut->m_hessianMultiple = hessianMultiple;
      pLossWrapperOut->m_bLossHasHessian = HasHessian<TLoss, TFloat>() ? EBM_TRUE : EBM_FALSE;
      pLossWrapperOut->m_bMse = TLoss::k_bMse ? EBM_TRUE : EBM_FALSE;

      pLossWrapperOut->m_pLoss = this;
   }

   Loss() = default;
   ~Loss() = default;

public:

   static ErrorEbm CreateLoss(
      const REGISTER_LOSSES_FUNCTION registerLossesFunction,
      const Config * const pConfig,
      const char * const sLoss,
      const char * const sLossEnd,
      LossWrapper * const pLossWrapperOut
   ) noexcept;
};
static_assert(std::is_standard_layout<Loss>::value && std::is_trivially_copyable<Loss>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

// TODO: include ranking
//
// We use the following terminology:
// Target      : the thing we're trying to predict.  For classification this is the label.  For regression this 
//               is what we're predicting directly.  Target and Output seem to be used interchangeably in other 
//               packages.  We choose Target here.
// Score       : the values we use to generate predictions.  For classification these are logits.  For regression these
//               are the predictions themselves.  For multiclass there are N scores per target when there are N classes.
//               For multiclass you could eliminate one score to get N-1 scores, but we don't use that trick in this 
//               package yet.
// Prediction  : the prediction of the model.  We output scores in our model and generate predictions from them.
//               For multiclass the scores are the logits, and the predictions would be the outputs of softmax.
//               We have N scores per target for an N class multiclass problem.
// Binary      : binary classification.  Target is 0 or 1
// Multiclass  : multiclass classification.  Target is 0, 1, 2, ... 
// Regression  : regression
// Multioutput : a model that can predict multiple different things.  A single model could predict binary, 
//               multiclass, regression, etc. different targets.
// Multitask   : A slightly more restricted form of multioutput where training jointly optimizes the targets.
//               The different targets can still be of different types (binary, multiclass, regression, etc), but
//               importantly they share a single loss function.  In C++ we deal only with multitask since otherwise 
//               it would make more sense to train the targets separately.  In higher level languages the models can 
//               either be Multitask or Multioutput depending on how they were generated.
// Multilabel  : A more restricted version of multitask where the tasks are all binary classification.  We use
//               the term MultitaskBinary* here since it fits better into our ontology.
// 
// The most general loss function that we could handle in C++ would be to take a custom loss function that jointly 
// optimizes a multitask problem that contains regression, binary, and multiclass tasks.  This would be: 
// "MultitaskLossCustom"

struct SingletaskLoss : public Loss {
protected:
   SingletaskLoss() = default;
   ~SingletaskLoss() = default;
};

struct BinaryLoss : public SingletaskLoss {
protected:
   BinaryLoss() = default;
   ~BinaryLoss() = default;
public:
   static constexpr bool IsMultiScore = false;
};

struct MulticlassLoss : public SingletaskLoss {
protected:
   MulticlassLoss() = default;
   ~MulticlassLoss() = default;
public:
   static constexpr bool IsMultiScore = true;
};

struct RegressionLoss : public SingletaskLoss {
protected:
   RegressionLoss() = default;
   ~RegressionLoss() = default;
public:
   static constexpr bool IsMultiScore = false;
};

struct MultitaskLoss : public Loss {
protected:
   MultitaskLoss() = default;
   ~MultitaskLoss() = default;
public:
   static constexpr bool IsMultiScore = true;
};

struct BinaryMultitaskLoss : public MultitaskLoss {
protected:
   BinaryMultitaskLoss() = default;
   ~BinaryMultitaskLoss() = default;
};

struct MulticlassMultitaskLoss : public MultitaskLoss {
protected:
   MulticlassMultitaskLoss() = default;
   ~MulticlassMultitaskLoss() = default;
};

struct RegressionMultitaskLoss : public MultitaskLoss {
protected:
   RegressionMultitaskLoss() = default;
   ~RegressionMultitaskLoss() = default;
};


#define LOSS_CONSTANTS_BOILERPLATE(__EBM_TYPE, __LINK_FUNCTION) \
   public: \
      static constexpr bool k_bMse = false; \
      static constexpr LinkEbm k_linkFunction = (__LINK_FUNCTION); \
      static ErrorEbm StaticApplyUpdate(const Loss * const pThis, ApplyUpdateBridge * const pData) { \
         return (static_cast<const __EBM_TYPE<TFloat> *>(pThis))->ParentApplyUpdate<const __EBM_TYPE<TFloat>, TFloat>(pData); \
      } \
      void FillWrapper(void * const pWrapperOut) noexcept { \
         static_assert( \
            std::is_same<__EBM_TYPE<TFloat>, typename std::remove_pointer<decltype(this)>::type>::value, \
            "*Loss types mismatch"); \
         FillLossWrapper<typename std::remove_pointer<decltype(this)>::type, TFloat>(pWrapperOut); \
      }

#define LOSS_TEMPLATE_BOILERPLATE \
   public: \
      template<size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight> \
      GPU_DEVICE void InjectedApplyUpdate(ApplyUpdateBridge * const pData) const { \
         Loss::ChildApplyUpdate<typename std::remove_pointer<decltype(this)>::type, TFloat, \
            cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData); \
      }

#define LOSS_BOILERPLATE(__EBM_TYPE, __LINK_FUNCTION) \
   LOSS_CONSTANTS_BOILERPLATE(__EBM_TYPE, __LINK_FUNCTION) \
   LOSS_TEMPLATE_BOILERPLATE

} // DEFINED_ZONE_NAME

#endif // LOSS_HPP
