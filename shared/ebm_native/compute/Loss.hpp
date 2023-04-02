// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !!! NOTE: To add a new loss/objective function in C++, follow the steps listed at the top of the "Loss.cpp" file !!!

#ifndef LOSS_HPP
#define LOSS_HPP

#include <stddef.h> // size_t, ptrdiff_t
#include <memory> // shared_ptr, unique_ptr

#include "ebm_native.h" // ErrorEbm
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


template<typename TLoss, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
GPU_GLOBAL static void ExecuteApplyUpdate(const Loss * const pLoss, ApplyUpdateBridge * const pData) {
   const TLoss * const pLossSpecific = static_cast<const TLoss *>(pLoss);
   pLossSpecific->template InteriorApplyUpdateTemplated<cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
}

struct Registrable {
   // TODO: move this into its own file once we create Metric classes that are also Registrable
protected:
   Registrable() = default;
   ~Registrable() = default;
};

struct Loss : public Registrable {
   // Welcome to the demented hall of mirrors.. a prison for your mind
   // And no, I did not make this to purposely torment you


   // if we have multiple scores AND multiple bitpacks, then we have two nested loops in our final function
   // and the compiler will only unroll the inner loop.  That inner loop will be for the scores, so there
   // is not much benefit in generating hard coded loop counts for the bitpacks, so short circut the
   // bit packing to use the dynamic value if we don't have the single bin case.  This also solves
   // part of our template blowup issue of having N * M starting point templates where N is the number
   // of scores and M is the number of bit packs.  If we use 8 * 16 that's already 128 copies of the
   // templated function at this point and more later.  Reducing this to just 16 is very very helpful.
   template<typename TLoss, typename TFloat, typename std::enable_if<!TLoss::IsMultiScore, TLoss>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm CountScoresPreApplyUpdate(ApplyUpdateBridge * const pData) const {
      if(k_cItemsPerBitPackNone == pData->m_cPack) {
         return BitPackPostApplyUpdate<TLoss, TFloat, k_oneScore, k_cItemsPerBitPackNone>(pData);
      } else {
         return BitPack<TLoss, TFloat, k_oneScore, k_cItemsPerBitPackMax>::ApplyUpdate(this, pData);
      }
   }
   template<typename TLoss, typename TFloat, typename std::enable_if<TLoss::IsMultiScore && std::is_base_of<MulticlassMultitaskLoss, TLoss>::value, TLoss>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm CountScoresPreApplyUpdate(ApplyUpdateBridge * const pData) const {
      if(k_cItemsPerBitPackNone == pData->m_cPack) {
         // don't blow up our complexity if we have only 1 bin.. just use dynamic for the count of scores
         return BitPackPostApplyUpdate<TLoss, TFloat, k_dynamicScores, k_cItemsPerBitPackNone>(pData);
      } else {
         // if our inner loop is dynamic scores, then the compiler won't do a full unwind of the bit pack
         // loop, so just short circuit it to using dynamic
         return BitPackPostApplyUpdate<TLoss, TFloat, k_dynamicScores, k_cItemsPerBitPackDynamic>(pData);
      }
   }
   template<typename TLoss, typename TFloat, typename std::enable_if<TLoss::IsMultiScore && !std::is_base_of<MulticlassMultitaskLoss, TLoss>::value, TLoss>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm CountScoresPreApplyUpdate(ApplyUpdateBridge * const pData) const {
      if(k_cItemsPerBitPackNone == pData->m_cPack) {
         // don't blow up our complexity if we have only 1 bin.. just use dynamic for the count of scores
         return BitPackPostApplyUpdate<TLoss, TFloat, k_dynamicScores, k_cItemsPerBitPackNone>(pData);
      } else {
         return CountScores<TLoss, TFloat, (k_cCompilerScoresMax < k_cCompilerScoresStart ? k_dynamicScores : k_cCompilerScoresStart)>::ApplyUpdate(this, pData);
      }
   }
   template<typename TLoss, typename TFloat, size_t cCompilerScores>
   struct CountScores final {
      INLINE_ALWAYS static ErrorEbm ApplyUpdate(const Loss * const pLoss, ApplyUpdateBridge * const pData) {
         if(cCompilerScores == pData->m_cRuntimeScores) {
            return pLoss->BitPackPostApplyUpdate<TLoss, TFloat, cCompilerScores, k_cItemsPerBitPackDynamic>(pData);
         } else {
            return CountScores<TLoss, TFloat, k_cCompilerScoresMax == cCompilerScores ? k_dynamicScores : cCompilerScores + 1>::ApplyUpdate(pLoss, pData);
         }
      }
   };
   template<typename TLoss, typename TFloat>
   struct CountScores<TLoss, TFloat, k_dynamicScores> final {
      INLINE_ALWAYS static ErrorEbm ApplyUpdate(const Loss * const pLoss, ApplyUpdateBridge * const pData) {
         return pLoss->BitPackPostApplyUpdate<TLoss, TFloat, k_dynamicScores, k_cItemsPerBitPackDynamic>(pData);
      }
   };


   // in our current format cCompilerScores will always be 1, but just in case we change our code to allow
   // for special casing multiclass with compile time unrolling of the compiler pack, leave cCompilerScores here
   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack>
   struct BitPack final {
      INLINE_ALWAYS static ErrorEbm ApplyUpdate(const Loss * const pLoss, ApplyUpdateBridge * const pData) {
         if(cCompilerPack == pData->m_cPack) {
            return pLoss->BitPackPostApplyUpdate<TLoss, TFloat, cCompilerScores, cCompilerPack>(pData);
         } else {
            return BitPack<TLoss, TFloat, cCompilerScores, GetNextBitPack(cCompilerPack)>::ApplyUpdate(pLoss, pData);
         }
      }
   };
   template<typename TLoss, typename TFloat, size_t cCompilerScores>
   struct BitPack<TLoss, TFloat, cCompilerScores, k_cItemsPerBitPackLast> final {
      INLINE_ALWAYS static ErrorEbm ApplyUpdate(const Loss * const pLoss, ApplyUpdateBridge * const pData) {
         return pLoss->BitPackPostApplyUpdate<TLoss, TFloat, cCompilerScores, k_cItemsPerBitPackLast>(pData);
      }
   };
   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack>
   INLINE_RELEASE_TEMPLATED ErrorEbm BitPackPostApplyUpdate(ApplyUpdateBridge * const pData) const {
      return AttachHessian<TLoss, TFloat, cCompilerScores, cCompilerPack, HasCalculateHessianFunction<TLoss, TFloat>()>::ApplyUpdate(this, pData);
   }

   template<typename TLoss, bool bHessian>
   struct ApplyHessian;
   template<typename TLoss>
   struct ApplyHessian<TLoss, true> final {
      GPU_DEVICE INLINE_ALWAYS static void Func() {
      }
   };
   template<typename TLoss>
   struct ApplyHessian<TLoss, false> final {
      GPU_DEVICE INLINE_ALWAYS static void Func() {
      }
   };

   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   struct Shared final {
      GPU_DEVICE static void ApplyUpdate(const TLoss * const pLoss, ApplyUpdateBridge * const pData) {
         UNUSED(pLoss);
         UNUSED(pData);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this
      }
   };
   template<typename TLoss, typename TFloat, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   struct Shared <TLoss, TFloat, k_oneScore, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight> final {
      GPU_DEVICE static void ApplyUpdate(const TLoss * const pLoss, ApplyUpdateBridge * const pData) {
         UNUSED(pLoss);
         UNUSED(pData);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this
      }
   };

   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   INLINE_RELEASE_TEMPLATED ErrorEbm CallbackTFloat(ApplyUpdateBridge * const pData) const {
      return TFloat::template ApplyUpdate<TLoss, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(this, pData);
   }

   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, typename std::enable_if<!TLoss::k_bMse, TLoss>::type* = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm FinalOptions(ApplyUpdateBridge * const pData) const {
      if (nullptr != pData->m_aGradientsAndHessians) {
         static constexpr bool bKeepGradHess = true;

         // if we are updating the gradients then we are doing training and do not need to calculate the metric
         EBM_ASSERT(!pData->m_bCalcMetric);
         static constexpr bool bCalcMetric = false;

         if (nullptr != pData->m_aWeights) {
            static constexpr bool bWeight = true;

            // this branch will only be taking during interaction initialization

            return CallbackTFloat<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
         }
         else {
            static constexpr bool bWeight = false;
            return CallbackTFloat<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
         }
      }
      else {
         static constexpr bool bKeepGradHess = false;

         if (pData->m_bCalcMetric) {
            static constexpr bool bCalcMetric = true;

            if (nullptr != pData->m_aWeights) {
               static constexpr bool bWeight = true;
               return CallbackTFloat<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
            }
            else {
               static constexpr bool bWeight = false;
               return CallbackTFloat<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
            }
         }
         else {
            static constexpr bool bCalcMetric = false;

            // currently this branch is not taken, but if would be if we wanted to allow in the future
            // non-metric calculating validation for boosting.  For instance if we wanted to substitute an alternate
            // metric or if for performance reasons we only want to calculate the metric every N rounds of boosting

            EBM_ASSERT(nullptr == pData->m_aWeights);
            static constexpr bool bWeight = false; // if we are not calculating the metric or updating gradients then we never need the weights

            return CallbackTFloat<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
         }
      }
   }

   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, typename std::enable_if<TLoss::k_bMse, TLoss>::type* = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbm FinalOptions(ApplyUpdateBridge * const pData) const {
      EBM_ASSERT(nullptr != pData->m_aGradientsAndHessians); // we always keep gradients for regression
      static constexpr bool bKeepGradHess = true;

      if (pData->m_bCalcMetric) {
         static constexpr bool bCalcMetric = true;

         if (nullptr != pData->m_aWeights) {
            static constexpr bool bWeight = true;
            return CallbackTFloat<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
         }
         else {
            static constexpr bool bWeight = false;
            return CallbackTFloat<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
         }
      }
      else {
         static constexpr bool bCalcMetric = false;

         EBM_ASSERT(nullptr == pData->m_aWeights);
         static constexpr bool bWeight = false; // if we are not calculating the metric then we never need the weights

         return CallbackTFloat<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData);
      }
   }

   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian>
   struct AttachHessian;
   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack>
   struct AttachHessian<TLoss, TFloat, cCompilerScores, cCompilerPack, true> final {
      INLINE_RELEASE_TEMPLATED static ErrorEbm ApplyUpdate(const Loss * const pLoss, ApplyUpdateBridge * const pData) {
         if(pData->m_bHessianNeeded) {
            return pLoss->FinalOptions<TLoss, TFloat, cCompilerScores, cCompilerPack, true>(pData);
         } else {
            return pLoss->FinalOptions<TLoss, TFloat, cCompilerScores, cCompilerPack, false>(pData);
         }
      }
   };
   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack>
   struct AttachHessian<TLoss, TFloat, cCompilerScores, cCompilerPack, false> final {
      INLINE_RELEASE_TEMPLATED static ErrorEbm ApplyUpdate(const Loss * const pLoss, ApplyUpdateBridge * const pData) {
         return pLoss->FinalOptions<TLoss, TFloat, cCompilerScores, cCompilerPack, false>(pData);
      }
   };


   template<class TLoss, typename TFloat>
   struct HasCalculateHessianFunctionInternal {
      // use SFINAE to find out if the target class has the function with the correct signature
      struct TrueStruct {
      };
      struct FalseStruct {
      };

      // under SFINAE, this first version of the NotInvokedCheck function disappears if it fails due to the lack of 
      // the CalculateHessian function with the correct parameters.  It would be better to use a more general
      // version that only looks for the function name, but I don't know if that's possible without taking a pointer
      // to the function, which we don't want to do since our function is meant to be 100% inline, and taking
      // a pointer might create a second non-inlined copy of the function with some compilers.  I think
      // burrying the inline function inside a decltype will prevent it's materialization.
      //
      // use float as the calling type on purpose since any of our Simd types should convert up from a float and
      // we shouldn't get compiler warnings for upgrades on the float type to doubles

      template<class TCheck>
      static TrueStruct NotInvokedCheck(TCheck const * pCheck,
         typename std::enable_if<
         std::is_same<TFloat, decltype(pCheck->CalculateHessian(TFloat { 0 }, TFloat { 0 }))>::value
         >::type * = nullptr);
      static FalseStruct NotInvokedCheck(...);
      static constexpr bool value = std::is_same<TrueStruct, 
         decltype(HasCalculateHessianFunctionInternal::NotInvokedCheck(static_cast<typename std::remove_reference<TLoss>::type *>(nullptr)))
         >::value;
   };

protected:

   template<typename TLoss, typename TFloat>
   constexpr static bool HasCalculateHessianFunction() {
      // use SFINAE to find out if our Loss class has the function CalculateHessian with the correct signature
      return HasCalculateHessianFunctionInternal<TLoss, TFloat>::value;
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

   template<typename TLoss, typename TFloat, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   GPU_DEVICE INLINE_RELEASE_TEMPLATED void InteriorApplyUpdate(ApplyUpdateBridge * const pData) const {
      const TLoss * const pLossSpecific = static_cast<const TLoss *>(this);
      Shared<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>::ApplyUpdate(pLossSpecific, pData);
   }

   template<typename TLoss, typename TFloat>
   INLINE_RELEASE_TEMPLATED ErrorEbm LossApplyUpdate(ApplyUpdateBridge * const pData) const {
      static_assert(IsEdgeLoss<TLoss>(), "TLoss must inherit from one of the children of the Loss class");
      return CountScoresPreApplyUpdate<TLoss, TFloat>(pData);
   }


   template<typename TLoss, typename TFloat>
   INLINE_RELEASE_TEMPLATED void LossFillWrapper(void * const pWrapperOut) noexcept {
      EBM_ASSERT(nullptr != pWrapperOut);
      LossWrapper * const pLossWrapperOut = static_cast<LossWrapper *>(pWrapperOut);
      FunctionPointersCpp * const pFunctionPointers =
         static_cast<FunctionPointersCpp *>(pLossWrapperOut->m_pFunctionPointersCpp);
      EBM_ASSERT(nullptr != pFunctionPointers);

      pFunctionPointers->m_pApplyUpdateCpp = &TLoss::ApplyUpdate;

      auto multiplier = (static_cast<TLoss *>(this))->GetFinalMultiplier();
      static_assert(std::is_same<decltype(multiplier), double>::value, "this->GetFinalMultiplier() should return a double");
      pLossWrapperOut->m_updateMultiple = multiplier;
      pLossWrapperOut->m_bLossHasHessian = HasCalculateHessianFunction<TLoss, TFloat>() ? EBM_TRUE : EBM_FALSE;
      pLossWrapperOut->m_bSuperSuperSpecialLossWhereTargetNotNeededOnlyMseLossQualifies = EBM_FALSE;

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
static_assert(std::is_standard_layout<Loss>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");
#if !(defined(__GNUC__) && __GNUC__ < 5)
static_assert(std::is_trivially_copyable<Loss>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");
#endif // !(defined(__GNUC__) && __GNUC__ < 5)

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


#define LOSS_CLASS_BOILERPLATE(__EBM_TYPE, isVectorized) \
   LOSS_CLASS_CONSTANTS_BOILERPLATE(isVectorized) \
   LOSS_CLASS_TEMPLATE_BOILERPLATE \
   LOSS_CLASS_VIRTUAL_BOILERPLATE(__EBM_TYPE)

// TODO: use the isVectorized static constexpr to control construction of the Loss structs
#define LOSS_CLASS_CONSTANTS_BOILERPLATE(isVectorized) \
   public: \
      static constexpr bool k_bVectorized = (isVectorized);

#define LOSS_CLASS_TEMPLATE_BOILERPLATE \
   public: \
      template<size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight> \
      GPU_DEVICE void InteriorApplyUpdateTemplated(ApplyUpdateBridge * const pData) const { \
         Loss::InteriorApplyUpdate<typename std::remove_pointer<decltype(this)>::type, TFloat, \
            cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pData); \
      }

#define LOSS_CLASS_VIRTUAL_BOILERPLATE(__EBM_TYPE) \
   public: \
      static ErrorEbm ApplyUpdate(const Loss * const pThis, ApplyUpdateBridge * const pData) { \
         return (static_cast<const __EBM_TYPE<TFloat> *>(pThis))->LossApplyUpdate<const __EBM_TYPE<TFloat>, TFloat>(pData); \
      } \
      void FillWrapper(void * const pWrapperOut) noexcept { \
         static_assert( \
            std::is_same<__EBM_TYPE<TFloat>, typename std::remove_pointer<decltype(this)>::type>::value, \
            "*Loss types mismatch"); \
         LossFillWrapper<typename std::remove_pointer<decltype(this)>::type, TFloat>(pWrapperOut); \
      }

} // DEFINED_ZONE_NAME

#endif // LOSS_HPP
