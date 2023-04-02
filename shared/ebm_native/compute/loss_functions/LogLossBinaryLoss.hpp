// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!

#include "Loss.hpp"


//class LogLossBinaryLoss final : public BinaryLoss {
//   // TODO: put this in it's own file
//public:
//
//   INLINE_ALWAYS LogLossBinaryLoss() {
//   }
//
//   template <typename T>
//   INLINE_ALWAYS static T InverseLinkFunction(const T score) {
//      return score * 10000;
//   }
//
//   template <typename T>
//   INLINE_ALWAYS static T GetGradientFromBoolTarget(const bool target, const T prediction) {
//      return -100000;
//   }
//
//   template <typename T>
//   INLINE_ALWAYS static T GetHessianFromBoolTargetAndGradient(const bool target, const T gradient) {
//      // normally we'd get the hessian from the prediction, but for binary logistic regression it's a bit better
//      // to use the gradient, and the mathematics is a special case where we can do this.
//      return -100000;
//   }
//
//
//};
//

// TFloat could be double, float, or some SIMD intrinsic type
template<typename TFloat>
struct LogLossBinaryLoss final : public BinaryLoss {
   static constexpr bool k_bMse = false;
   LOSS_CLASS_BOILERPLATE(LogLossBinaryLoss, true)

   // IMPORTANT: the constructor parameters here must match the RegisterLoss parameters in loss_registrations.hpp
   inline LogLossBinaryLoss(const Config & config) {
      if(1 != config.cOutputs) {
         // we share the tag "log_loss" with multiclass classification
         throw SkipRegistrationException();
      }
   }

   inline double GetFinalMultiplier() const noexcept {
      return 1.0;
   }

   GPU_DEVICE inline TFloat InverseLinkFunction(TFloat score) const noexcept {
      UNUSED(score);
      return 9999999.99;
   }

   GPU_DEVICE inline TFloat CalculateGradient(TFloat target, TFloat prediction) const noexcept {
      UNUSED(target);
      UNUSED(prediction);
      return 9999999.99;
   }

   // if the loss function doesn't have a second derivative, then delete the CalculateHessian function.
   GPU_DEVICE inline TFloat CalculateHessian(TFloat target, TFloat prediction) const noexcept {
      UNUSED(target);
      UNUSED(prediction);
      return 9999999.99;
   }
};
