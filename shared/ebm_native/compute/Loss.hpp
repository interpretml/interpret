// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !!! NOTE: To add a new loss/objective function in C++, follow the steps listed at the top of the "Loss.cpp" file !!!

#ifndef LOSS_H
#define LOSS_H

#include <stddef.h> // size_t, ptrdiff_t
#include <memory> // shared_ptr, unique_ptr

#include "ebm_native.h"
#include "logging.h"
#include "common_c.h" // INLINE_ALWAYS
#include "bridge_c.h"
#include "zones.h"

#include "compute.hpp"

#include "zoned_bridge_cpp_functions.hpp"
#include "registration_exceptions.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class Registration;
typedef const std::vector<std::shared_ptr<const Registration>> (* REGISTER_LOSSES_FUNCTION)();

struct LossSingletask;
struct LossBinary;
struct LossMulticlass;
struct LossRegression;

struct LossMultitask;
struct LossMultitaskBinary;
struct LossMultitaskMulticlass;
struct LossMultitaskRegression;


template<template <typename, typename, ptrdiff_t, ptrdiff_t, bool> class TExecute, typename TLoss, typename TFloat, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian>
GPU_GLOBAL static void ExecuteApplyTraining(
   const Loss * const pLoss, 
   const ptrdiff_t cRuntimeScores,
   const ptrdiff_t cRuntimePack
) {
   TLoss * const pLossSpecific = static_cast<TLoss *>(pLoss);
   TExecute<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian>::ApplyTraining(
      pLossSpecific, 
      cRuntimeScores,
      cRuntimePack
   );
}
template<template <typename, typename, ptrdiff_t, ptrdiff_t, bool> class TExecute, typename TLoss, typename TFloat, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian>
GPU_GLOBAL static void ExecuteApplyValidation(
   const Loss * const pLoss, 
   const ptrdiff_t cRuntimeScores,
   const ptrdiff_t cRuntimePack,
   FloatEbmType * const pMetricOut
) {
   TLoss * const pLossSpecific = static_cast<TLoss *>(pLoss);
   TExecute<TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian>::ApplyValidation(
      pLossSpecific, 
      cRuntimeScores,
      cRuntimePack,
      pMetricOut
   );
}

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
   template<typename TLoss, typename std::enable_if<!TLoss::IsMultiScore, TLoss>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbmType CountScoresPreApplyTraining(ApplyTrainingData * const pData) const {
      if(k_cItemsPerBitPackNone == pData->m_cRuntimePack) {
         return BitPackPostApplyTraining<TLoss, k_oneScore, k_cItemsPerBitPackNone>(pData);
      } else {
         return BitPack<TLoss, k_oneScore, k_cItemsPerBitPackMax2>::ApplyTraining(this, pData);
      }
   }
   template<typename TLoss, typename std::enable_if<!TLoss::IsMultiScore, TLoss>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbmType CountScoresPreApplyValidation(ApplyValidationData * const pData) const {
      if(k_cItemsPerBitPackNone == pData->m_cRuntimePack) {
         return BitPackPostApplyValidation<TLoss, k_oneScore, k_cItemsPerBitPackNone>(pData);
      } else {
         return BitPack<TLoss, k_oneScore, k_cItemsPerBitPackMax2>::ApplyValidation(this, pData);
      }
   }
   template<typename TLoss, typename std::enable_if<TLoss::IsMultiScore && std::is_base_of<LossMultitaskMulticlass, TLoss>::value, TLoss>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbmType CountScoresPreApplyTraining(ApplyTrainingData * const pData) const {
      if(k_cItemsPerBitPackNone == pData->m_cRuntimePack) {
         // don't blow up our complexity if we have only 1 bin.. just use dynamic for the count of scores
         return BitPackPostApplyTraining<TLoss, k_dynamicClassification, k_cItemsPerBitPackNone>(pData);
      } else {
         // if our inner loop is dynamic scores, then the compiler won't do a full unwind of the bit pack
         // loop, so just short circuit it to using dynamic
         return BitPackPostApplyTraining<TLoss, k_dynamicClassification, k_cItemsPerBitPackDynamic2>(pData);
      }
   }
   template<typename TLoss, typename std::enable_if<TLoss::IsMultiScore && std::is_base_of<LossMultitaskMulticlass, TLoss>::value, TLoss>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbmType CountScoresPreApplyValidation(ApplyValidationData * const pData) const {
      if(k_cItemsPerBitPackNone == pData->m_cRuntimePack) {
         // don't blow up our complexity if we have only 1 bin.. just use dynamic for the count of scores
         return BitPackPostApplyValidation<TLoss, k_dynamicClassification, k_cItemsPerBitPackNone>(pData);
      } else {
         // if our inner loop is dynamic scores, then the compiler won't do a full unwind of the bit pack
         // loop, so just short circuit it to using dynamic
         return BitPackPostApplyValidation<TLoss, k_dynamicClassification, k_cItemsPerBitPackDynamic2>(pData);
      }
   }
   template<typename TLoss, typename std::enable_if<TLoss::IsMultiScore && !std::is_base_of<LossMultitaskMulticlass, TLoss>::value, TLoss>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbmType CountScoresPreApplyTraining(ApplyTrainingData * const pData) const {
      if(k_cItemsPerBitPackNone == pData->m_cRuntimePack) {
         // don't blow up our complexity if we have only 1 bin.. just use dynamic for the count of scores
         return BitPackPostApplyTraining<TLoss, k_dynamicClassification, k_cItemsPerBitPackNone>(pData);
      } else {
         return CountScores<TLoss, (k_cCompilerOptimizedTargetClassesMax2 < k_cCompilerOptimizedTargetClassesStart2 ? k_dynamicClassification : k_cCompilerOptimizedTargetClassesStart2)>::ApplyTraining(this, pData);
      }
   }
   template<typename TLoss, typename std::enable_if<TLoss::IsMultiScore && !std::is_base_of<LossMultitaskMulticlass, TLoss>::value, TLoss>::type * = nullptr>
   INLINE_RELEASE_TEMPLATED ErrorEbmType CountScoresPreApplyValidation(ApplyValidationData * const pData) const {
      if(k_cItemsPerBitPackNone == pData->m_cRuntimePack) {
         // don't blow up our complexity if we have only 1 bin.. just use dynamic for the count of scores
         return BitPackPostApplyValidation<TLoss, k_dynamicClassification, k_cItemsPerBitPackNone>(pData);
      } else {
         return CountScores<TLoss, (k_cCompilerOptimizedTargetClassesMax2 < k_cCompilerOptimizedTargetClassesStart2 ? k_dynamicClassification : k_cCompilerOptimizedTargetClassesStart2)>::ApplyValidation(this, pData);
      }
   }
   template<typename TLoss, ptrdiff_t cCompilerScores>
   struct CountScores final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ApplyTrainingData * const pData) {
         if(cCompilerScores == pData->m_cRuntimeScores) {
            return pLoss->BitPackPostApplyTraining<TLoss, cCompilerScores, k_cItemsPerBitPackDynamic2>(pData);
         } else {
            return CountScores<TLoss, k_cCompilerOptimizedTargetClassesMax2 == cCompilerScores ? k_dynamicClassification : cCompilerScores + 1>::ApplyTraining(pLoss, pData);
         }
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ApplyValidationData * const pData) {
         if(cCompilerScores == pData->m_cRuntimeScores) {
            return pLoss->BitPackPostApplyValidation<TLoss, cCompilerScores, k_cItemsPerBitPackDynamic2>(pData);
         } else {
            return CountScores<TLoss, k_cCompilerOptimizedTargetClassesMax2 == cCompilerScores ? k_dynamicClassification : cCompilerScores + 1>::ApplyValidation(pLoss, pData);
         }
      }
   };
   template<typename TLoss>
   struct CountScores<TLoss, k_dynamicClassification> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ApplyTrainingData * const pData) {
         return pLoss->BitPackPostApplyTraining<TLoss, k_dynamicClassification, k_cItemsPerBitPackDynamic2>(pData);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ApplyValidationData * const pData) {
         return pLoss->BitPackPostApplyValidation<TLoss, k_dynamicClassification, k_cItemsPerBitPackDynamic2>(pData);
      }
   };


   // in our current format cCompilerScores will always be 1, but just in case we change our code to allow
   // for special casing multiclass with compile time unrolling of the compiler pack, leave cCompilerScores here
   template<typename TLoss, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack>
   struct BitPack final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ApplyTrainingData * const pData) {
         if(cCompilerPack == pData->m_cRuntimePack) {
            return pLoss->BitPackPostApplyTraining<TLoss, cCompilerScores, cCompilerPack>(pData);
         } else {
            return BitPack<TLoss, cCompilerScores, GetNextBitPack(cCompilerPack)>::ApplyTraining(pLoss, pData);
         }
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ApplyValidationData * const pData) {
         if(cCompilerPack == pData->m_cRuntimePack) {
            return pLoss->BitPackPostApplyValidation<TLoss, cCompilerScores, cCompilerPack>(pData);
         } else {
            return BitPack<TLoss, cCompilerScores, GetNextBitPack(cCompilerPack)>::ApplyValidation(pLoss, pData);
         }
      }
   };
   template<typename TLoss, ptrdiff_t cCompilerScores>
   struct BitPack<TLoss, cCompilerScores, k_cItemsPerBitPackLast> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ApplyTrainingData * const pData) {
         return pLoss->BitPackPostApplyTraining<TLoss, cCompilerScores, k_cItemsPerBitPackLast>(pData);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ApplyValidationData * const pData) {
         return pLoss->BitPackPostApplyValidation<TLoss, cCompilerScores, k_cItemsPerBitPackLast>(pData);
      }
   };
   template<typename TLoss, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType BitPackPostApplyTraining(ApplyTrainingData * const pData) const {
      const TLoss * const pTLoss = static_cast<const TLoss *>(this);
      return pTLoss->template ApplyTrainingTemplated<cCompilerScores, cCompilerPack>(pData);
   }
   template<typename TLoss, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType BitPackPostApplyValidation(ApplyValidationData * const pData) const {
      const TLoss * const pTLoss = static_cast<const TLoss *>(this);
      return pTLoss->template ApplyValidationTemplated<cCompilerScores, cCompilerPack>(pData);
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

   template<typename TLoss, typename TFloat, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian>
   struct Shared final {
      GPU_DEVICE static void ApplyTraining(
         const TLoss * const pLoss,
         const ptrdiff_t cRuntimeScores,
         const ptrdiff_t cRuntimePack
      ) {
         UNUSED(pLoss);
         UNUSED(cRuntimeScores);
         UNUSED(cRuntimePack);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this
      }
      GPU_DEVICE static void ApplyValidation(
         const TLoss * const pLoss,
         const ptrdiff_t cRuntimeScores,
         const ptrdiff_t cRuntimePack,
         FloatEbmType * const pMetricOut
      ) {
         UNUSED(pLoss);
         UNUSED(cRuntimeScores);
         UNUSED(cRuntimePack);
         UNUSED(pMetricOut);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this
      }
   };
   template<typename TLoss, typename TFloat, ptrdiff_t cCompilerScores, bool bHessian>
   struct Shared<TLoss, TFloat, cCompilerScores, k_cItemsPerBitPackNone, bHessian> final {
      GPU_DEVICE static void ApplyTraining(
         const TLoss * const pLoss,
         const ptrdiff_t cRuntimeScores,
         const ptrdiff_t cRuntimePack
      ) {
         UNUSED(pLoss);
         UNUSED(cRuntimeScores);
         UNUSED(cRuntimePack);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this
      }
      GPU_DEVICE static void ApplyValidation(
         const TLoss * const pLoss,
         const ptrdiff_t cRuntimeScores,
         const ptrdiff_t cRuntimePack,
         FloatEbmType * const pMetricOut
      ) {
         UNUSED(pLoss);
         UNUSED(cRuntimeScores);
         UNUSED(cRuntimePack);
         UNUSED(pMetricOut);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this
      }
   };
   template<typename TLoss, typename TFloat, ptrdiff_t cCompilerPack, bool bHessian>
   struct Shared <TLoss, TFloat, k_oneScore, cCompilerPack, bHessian> final {
      GPU_DEVICE static void ApplyTraining(
         const TLoss * const pLoss,
         const ptrdiff_t cRuntimeScores,
         const ptrdiff_t cRuntimePack
      ) {
         UNUSED(pLoss);
         UNUSED(cRuntimeScores);
         UNUSED(cRuntimePack);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this
      }
      GPU_DEVICE static void ApplyValidation(
         const TLoss * const pLoss,
         const ptrdiff_t cRuntimeScores,
         const ptrdiff_t cRuntimePack,
         FloatEbmType * const pMetricOut
      ) {
         UNUSED(pLoss);
         UNUSED(cRuntimeScores);
         UNUSED(cRuntimePack);
         UNUSED(pMetricOut);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this
      }
   };
   template<typename TLoss, typename TFloat, bool bHessian>
   struct Shared<TLoss, TFloat, k_oneScore, k_cItemsPerBitPackNone, bHessian> final {
      GPU_DEVICE static void ApplyTraining(
         const TLoss * const pLoss, 
         const ptrdiff_t cRuntimeScores,
         const ptrdiff_t cRuntimePack
      ) {
         UNUSED(pLoss);
         UNUSED(cRuntimeScores);
         UNUSED(cRuntimePack);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this
      }
      GPU_DEVICE static void ApplyValidation(
         const TLoss * const pLoss,
         const ptrdiff_t cRuntimeScores,
         const ptrdiff_t cRuntimePack,
         FloatEbmType * const pMetricOut
      ) {
         UNUSED(pLoss);
         UNUSED(cRuntimeScores);
         UNUSED(cRuntimePack);
         UNUSED(pMetricOut);

         ApplyHessian<TLoss, bHessian>::Func(); // TODO: use this
      }
   };


   template<typename TLoss, typename TFloat, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian>
   struct AttachHessian;
   template<typename TLoss, typename TFloat, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack>
   struct AttachHessian<TLoss, TFloat, cCompilerScores, cCompilerPack, true> final {
      INLINE_RELEASE_TEMPLATED static ErrorEbmType ApplyTraining(const Loss * const pLoss, ApplyTrainingData * const pData) {
         if(pData->m_bHessianNeeded) {
            return TFloat::template ApplyTraining<Shared, TLoss, TFloat, cCompilerScores, cCompilerPack, true>(pLoss, pData);
         } else {
            return TFloat::template ApplyTraining<Shared, TLoss, TFloat, cCompilerScores, cCompilerPack, false>(pLoss, pData);
         }
      }
      INLINE_RELEASE_TEMPLATED static ErrorEbmType ApplyValidation(const Loss * const pLoss, ApplyValidationData * const pData) {
         if(pData->m_bHessianNeeded) {
            return TFloat::template ApplyValidation<Shared, TLoss, TFloat, cCompilerScores, cCompilerPack, true>(pLoss, pData);
         } else {
            return TFloat::template ApplyValidation<Shared, TLoss, TFloat, cCompilerScores, cCompilerPack, false>(pLoss, pData);
         }
      }
   };
   template<typename TLoss, typename TFloat, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack>
   struct AttachHessian<TLoss, TFloat, cCompilerScores, cCompilerPack, false> final {
      INLINE_RELEASE_TEMPLATED static ErrorEbmType ApplyTraining(const Loss * const pLoss, ApplyTrainingData * const pData) {
         return TFloat::template ApplyTraining<Shared, TLoss, TFloat, cCompilerScores, cCompilerPack, false>(pLoss, pData);
      }
      INLINE_RELEASE_TEMPLATED static ErrorEbmType ApplyValidation(const Loss * const pLoss, ApplyValidationData * const pData) {
         return TFloat::template ApplyValidation<Shared, TLoss, TFloat, cCompilerScores, cCompilerPack, false>(pLoss, pData);
      }
   };


   template<class TLoss>
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
         std::is_same<FloatEbmType, decltype(pCheck->CalculateHessian(float { 0 }, float { 0 }))>::value
         >::type * = nullptr);
      static FalseStruct NotInvokedCheck(...);
      static constexpr bool value = std::is_same<TrueStruct, 
         decltype(HasCalculateHessianFunctionInternal::NotInvokedCheck(static_cast<typename std::remove_reference<TLoss>::type *>(nullptr)))
         >::value;
   };

protected:

   template<typename TLoss>
   constexpr static bool HasCalculateHessianFunction() {
      // use SFINAE to find out if our Loss class has the function CalculateHessian with the correct signature
      return HasCalculateHessianFunctionInternal<TLoss>::value;
   }

   template<typename TLoss>
   constexpr static bool IsEdgeLoss() { 
      return
         std::is_base_of<LossBinary, TLoss>::value ||
         std::is_base_of<LossMulticlass, TLoss>::value ||
         std::is_base_of<LossRegression, TLoss>::value ||
         std::is_base_of<LossMultitaskBinary, TLoss>::value ||
         std::is_base_of<LossMultitaskMulticlass, TLoss>::value ||
         std::is_base_of<LossMultitaskRegression, TLoss>::value;
   }


   template<typename TLoss, typename TFloat, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType SharedApplyTraining(ApplyTrainingData * const pData) const {
      static_assert(IsEdgeLoss<TLoss>(), "TLoss must inherit from one of the children of the Loss class");
      return AttachHessian<TLoss, TFloat, cCompilerScores, cCompilerPack, HasCalculateHessianFunction<TLoss>()>::ApplyTraining(this, pData);
   }
   template<typename TLoss, typename TFloat, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType SharedApplyValidation(ApplyValidationData * const pData) const {
      static_assert(IsEdgeLoss<TLoss>(), "TLoss must inherit from one of the children of the Loss class");
      return AttachHessian<TLoss, TFloat, cCompilerScores, cCompilerPack, HasCalculateHessianFunction<TLoss>()>::ApplyValidation(this, pData);
   }


   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyTraining(ApplyTrainingData * const pData) const {
      static_assert(IsEdgeLoss<TLoss>(), "TLoss must inherit from one of the children of the Loss class");
      return CountScoresPreApplyTraining<TLoss>(pData);
   }
   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyValidation(ApplyValidationData * const pData) const {
      static_assert(IsEdgeLoss<TLoss>(), "TLoss must inherit from one of the children of the Loss class");
      return CountScoresPreApplyValidation<TLoss>(pData);
   }


   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED void LossFillWrapper(
      const FloatEbmType updateMultiple,
      void * const pWrapperOut
   ) const noexcept {
      EBM_ASSERT(nullptr != pWrapperOut);
      LossWrapper * const pLossWrapperOut = static_cast<LossWrapper *>(pWrapperOut);
      FunctionPointersCpp * const pFunctionPointers =
         static_cast<FunctionPointersCpp *>(pLossWrapperOut->m_pFunctionPointersCpp);
      EBM_ASSERT(nullptr != pFunctionPointers);

      pFunctionPointers->m_pApplyTrainingCpp = &TLoss::ApplyTraining;
      pFunctionPointers->m_pApplyValidationCpp = &TLoss::ApplyValidation;

      pLossWrapperOut->m_updateMultiple = updateMultiple;
      pLossWrapperOut->m_bLossHasHessian = HasCalculateHessianFunction<TLoss>() ? EBM_TRUE : EBM_FALSE;
      pLossWrapperOut->m_bSuperSuperSpecialLossWhereTargetNotNeededOnlyMseLossQualifies = EBM_FALSE;

      pLossWrapperOut->m_pLoss = this;
   }

   Loss() = default;

public:

   static ErrorEbmType CreateLoss(
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
// "LossMultitaskCustom"

struct LossSingletask : public Loss {
protected:
   LossSingletask() = default;
};

struct LossBinary : public LossSingletask {
protected:
   LossBinary() = default;
public:
   static constexpr bool IsMultiScore = false;
};

struct LossMulticlass : public LossSingletask {
protected:
   LossMulticlass() = default;
public:
   static constexpr bool IsMultiScore = true;
};

struct LossRegression : public LossSingletask {
protected:
   LossRegression() = default;
public:
   static constexpr bool IsMultiScore = false;
};

struct LossMultitask : public Loss {
protected:
   LossMultitask() = default;
public:
   static constexpr bool IsMultiScore = true;
};

struct LossMultitaskBinary : public LossMultitask {
protected:
   LossMultitaskBinary() = default;
};

struct LossMultitaskMulticlass : public LossMultitask {
protected:
   LossMultitaskMulticlass() = default;
};

struct LossMultitaskRegression : public LossMultitask {
protected:
   LossMultitaskRegression() = default;
};


#define LOSS_CLASS_BOILERPLATE(__EBM_TYPE, isVectorized, __updateMultiple) \
   LOSS_CLASS_CONSTANTS_BOILERPLATE(isVectorized) \
   LOSS_CLASS_TEMPLATE_BOILERPLATE \
   LOSS_CLASS_VIRTUAL_BOILERPLATE(__EBM_TYPE, (__updateMultiple))

// TODO: use the isVectorized constexpr to control construction of the Loss structs
#define LOSS_CLASS_CONSTANTS_BOILERPLATE(isVectorized) \
   public: \
      constexpr static bool k_bVectorized = (isVectorized);

#define LOSS_CLASS_TEMPLATE_BOILERPLATE \
   public: \
      template<ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack> \
      ErrorEbmType ApplyTrainingTemplated(ApplyTrainingData * const pData) const { \
         return Loss::SharedApplyTraining<typename std::remove_pointer<decltype(this)>::type, TFloat, \
            cCompilerScores, cCompilerPack>(pData); \
      } \
      template<ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack> \
      ErrorEbmType ApplyValidationTemplated(ApplyValidationData * const pData) const { \
         return Loss::SharedApplyValidation<typename std::remove_pointer<decltype(this)>::type, TFloat, \
            cCompilerScores, cCompilerPack>(pData); \
      }

#define LOSS_CLASS_VIRTUAL_BOILERPLATE(__EBM_TYPE, __updateMultiple) \
   public: \
      static ErrorEbmType ApplyTraining(const Loss * const pThis, ApplyTrainingData * const pData) { \
         return (static_cast<const __EBM_TYPE<TFloat> *>(pThis))->LossApplyTraining<const __EBM_TYPE<TFloat>>(pData); \
      } \
      static ErrorEbmType ApplyValidation(const Loss * const pThis, ApplyValidationData * const pData) { \
         return (static_cast<const __EBM_TYPE<TFloat> *>(pThis))->LossApplyValidation<const __EBM_TYPE<TFloat>>(pData); \
      } \
      void FillWrapper(void * const pWrapperOut) const noexcept { \
         static_assert( \
            std::is_same<const __EBM_TYPE<TFloat>, typename std::remove_pointer<decltype(this)>::type>::value, \
            "*Loss types mismatch"); \
         LossFillWrapper<typename std::remove_pointer<decltype(this)>::type>( \
            static_cast<FloatEbmType>(__updateMultiple), \
            pWrapperOut); \
      }

} // DEFINED_ZONE_NAME

#endif // LOSS_H
