// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef LOSS_H
#define LOSS_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureGroup.h"
#include "ThreadStateBoosting.h"

class Loss {

   template<typename TLoss, ptrdiff_t compilerCountClasses>
   struct CountClasses final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         if(compilerCountClasses == pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()) {
            return BitPack<TLoss, compilerCountClasses, k_cItemsPerBitPackMax2>::ApplyTraining(pLoss, pThreadStateBoosting, pFeatureGroup);
         } else {
            return CountClasses<TLoss, compilerCountClasses + 1>::ApplyTraining(pLoss, pThreadStateBoosting, pFeatureGroup);
         }
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         if(compilerCountClasses == pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()) {
            return BitPack<TLoss, compilerCountClasses, k_cItemsPerBitPackMax2>::ApplyValidation(pLoss, pThreadStateBoosting, pFeatureGroup, pMetricOut);
         } else {
            return CountClasses<TLoss, compilerCountClasses + 1>::ApplyValidation(pLoss, pThreadStateBoosting, pFeatureGroup, pMetricOut);
         }
      }
   };
   template<typename TLoss>
   struct CountClasses<TLoss, k_cCompilerOptimizedTargetClassesMax + 1> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         return BitPack<TLoss, k_dynamicClassification, k_cItemsPerBitPackMax2>::ApplyTraining(pLoss, pThreadStateBoosting, pFeatureGroup);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         return BitPack<TLoss, k_dynamicClassification, k_cItemsPerBitPackMax2>::ApplyValidation(pLoss, pThreadStateBoosting, pFeatureGroup, pMetricOut);
      }
   };


   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   struct BitPack final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         if(compilerBitPack == pFeatureGroup->GetBitPack()) {
            return pLoss->BitPackApplyTraining<TLoss, compilerCountClasses, compilerBitPack>(pThreadStateBoosting, pFeatureGroup);
         } else {
            return BitPack<TLoss, compilerCountClasses, GetNextBitPack(compilerBitPack)>::ApplyTraining(pLoss, pThreadStateBoosting, pFeatureGroup);
         }
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         if(compilerBitPack == pFeatureGroup->GetBitPack()) {
            return pLoss->BitPackApplyValidation<TLoss, compilerCountClasses, compilerBitPack>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
         } else {
            return BitPack<TLoss, compilerCountClasses, GetNextBitPack(compilerBitPack)>::ApplyValidation(pLoss, pThreadStateBoosting, pFeatureGroup, pMetricOut);
         }
      }
   };
   template<typename TLoss, ptrdiff_t compilerCountClasses>
   struct BitPack<TLoss, compilerCountClasses, k_cItemsPerBitPackNone> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         return pLoss->BitPackApplyTraining<TLoss, compilerCountClasses, k_cItemsPerBitPackNone>(pThreadStateBoosting, pFeatureGroup);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         return pLoss->BitPackApplyValidation<TLoss, compilerCountClasses, k_cItemsPerBitPackNone>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
      }
   };


   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   struct RemoveMulticlass final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         const TLoss * const pTLoss = static_cast<const TLoss *>(pLoss);
         return pTLoss->ApplyTrainingTemplated<compilerCountClasses, compilerBitPack>(pThreadStateBoosting, pFeatureGroup);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         const TLoss * const pTLoss = static_cast<const TLoss *>(pLoss);
         return pTLoss->ApplyValidationTemplated<compilerCountClasses, compilerBitPack>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
      }
   };
   template<typename TLoss, ptrdiff_t compilerBitPack>
   struct RemoveMulticlass<TLoss, k_regression, compilerBitPack> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         const TLoss * const pTLoss = static_cast<const TLoss *>(pLoss);
         return pTLoss->ApplyTrainingTemplated<compilerBitPack>(pThreadStateBoosting, pFeatureGroup);
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         const TLoss * const pTLoss = static_cast<const TLoss *>(pLoss);
         return pTLoss->ApplyValidationTemplated<compilerBitPack>(pThreadStateBoosting, pFeatureGroup, pMetricOut);
      }
   };


   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType BitPackApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      return RemoveMulticlass<TLoss, compilerCountClasses, compilerBitPack>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   INLINE_RELEASE_TEMPLATED ErrorEbmType BitPackApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      return RemoveMulticlass<TLoss, compilerCountClasses, compilerBitPack>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }


   template<typename TLoss, ptrdiff_t compilerBitPack>
   struct Shared final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         UNUSED(pMetricOut);
         return Error_None;
      }
   };
   template<typename TLoss>
   struct Shared<TLoss, k_cItemsPerBitPackNone> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         UNUSED(pMetricOut);
         return Error_None;
      }
   };


   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   struct SharedMulticlass final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         UNUSED(pMetricOut);
         return Error_None;
      }
   };
   template<typename TLoss, ptrdiff_t compilerCountClasses>
   struct SharedMulticlass<TLoss, compilerCountClasses, k_cItemsPerBitPackNone> final {
      INLINE_ALWAYS static ErrorEbmType ApplyTraining(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         return Error_None;
      }
      INLINE_ALWAYS static ErrorEbmType ApplyValidation(const Loss * const pLoss, ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) {
         UNUSED(pLoss);
         UNUSED(pThreadStateBoosting);
         UNUSED(pFeatureGroup);
         UNUSED(pMetricOut);
         return Error_None;
      }
   };

protected:

   Loss() = default;

   template<typename TLoss, ptrdiff_t compilerBitPack>
   ErrorEbmType SharedApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return Shared<TLoss, compilerBitPack>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss, ptrdiff_t compilerBitPack>
   ErrorEbmType SharedApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return Shared<TLoss, compilerBitPack>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }


   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   ErrorEbmType SharedApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return SharedMulticlass<TLoss, compilerCountClasses, compilerBitPack>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }

   template<typename TLoss, ptrdiff_t compilerCountClasses, ptrdiff_t compilerBitPack>
   ErrorEbmType SharedApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return SharedMulticlass<TLoss, compilerCountClasses, compilerBitPack>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }


   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyTraining(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return BitPack<TLoss, k_regression, k_cItemsPerBitPackMax2>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyValidation(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return BitPack<TLoss, k_regression, k_cItemsPerBitPackMax2>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }


   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyTrainingMulticlass(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return CountClasses<TLoss, 3>::ApplyTraining(this, pThreadStateBoosting, pFeatureGroup);
   }
   template<typename TLoss>
   INLINE_RELEASE_TEMPLATED ErrorEbmType LossApplyValidationMulticlass(ThreadStateBoosting * const pThreadStateBoosting, const FeatureGroup * const pFeatureGroup, FloatEbmType * const pMetricOut) const {
      static_assert(std::is_base_of<Loss, TLoss>::value, "TLoss must inherit from Loss");
      return CountClasses<TLoss, 3>::ApplyValidation(this, pThreadStateBoosting, pFeatureGroup, pMetricOut);
   }

public:

   virtual FloatEbmType GetUpdateMultiple() const {
      // TODO: we need to actually use this now..
      return FloatEbmType { 1 };
   }

   virtual ErrorEbmType ApplyTraining(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup
   ) const = 0;

   virtual ErrorEbmType ApplyValidation(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      FloatEbmType * const pMetricOut
   ) const = 0;

   static ErrorEbmType CreateLoss(
      const char * const sLoss,
      const size_t cTargetClasses,
      const Loss ** const ppLoss
   ) noexcept;

   virtual ~Loss() = default;
};

typedef ErrorEbmType (*ATTEMPT_CREATE_OBJECTIVE)(
   const char * sLoss,
   size_t countTargetClasses,
   const Loss ** const ppLoss
);







//class LossBinaryLogLoss final : public Loss {
//   // TODO: put this in it's own file
//public:
//
//   INLINE_ALWAYS LossBinaryLogLoss() {
//   }
//
//   template <typename T>
//   static INLINE_ALWAYS T CalculatePrediction(const T score) {
//      return score * 10000;
//   }
//
//   template <typename T>
//   static INLINE_ALWAYS T GetGradientFromBoolTarget(const bool target, const T prediction) {
//      return -100000;
//   }
//
//   template <typename T>
//   static INLINE_ALWAYS T GetHessianFromBoolTargetAndGradient(const bool target, const T gradient) {
//      // normally we'd get the hessian from the prediction, but for binary logistic regression it's a bit better
//      // to use the gradient, and the mathematics is a special case where we can do this.
//      return -100000;
//   }
//
//   INLINE_ALWAYS FloatEbmType GetUpdateMultiple() const noexcept override {
//      return FloatEbmType { 1 };
//   }
//
//};
//
//class LossRegressionMSE final : public Loss {
//   // TODO: put this in it's own file
//public:
//
//   INLINE_ALWAYS LossRegressionMSE() {
//   }
//
//   // for MSE regression, we get target - score at initialization and only store the gradients, so we never
//   // make a prediction, so we don't need CalculatePrediction(...)
//
//   template <typename T>
//   static INLINE_ALWAYS T GetGradientFromGradientPrev(const T target, const T gradientPrev) {
//      // for MSE regression, we get target - score at initialization and only store the gradients, so we
//      // never need the targets.  We just work from the previous gradients.
//
//      return -100000;
//   }
//
//   INLINE_ALWAYS FloatEbmType GetUpdateMultiple() const noexcept override {
//      return FloatEbmType { 1 };
//   }
//
//};

#endif // LOSS_H
