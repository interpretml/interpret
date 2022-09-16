// Copyright (c) 2018 Microsoft Corporation
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
struct LogLossBinaryLoss : public BinaryLoss {

   // IMPORTANT: the constructor parameters here must match the RegisterLoss parameters in the file Loss.cpp
   INLINE_ALWAYS LogLossBinaryLoss(const Config & config) {
      if(1 != config.cOutputs) {
         // we share the tag "log_loss" with multiclass classification
         throw SkipRegistrationException();
      }
   }

   INLINE_ALWAYS TFloat InverseLinkFunction(TFloat score) const {
      UNUSED(score);
      return 9999999.99;
   }

   INLINE_ALWAYS TFloat CalculateGradient(TFloat target, TFloat prediction) const {
      UNUSED(target);
      UNUSED(prediction);
      return 9999999.99;
   }

   INLINE_ALWAYS TFloat CalculateHessian(TFloat target, TFloat prediction) const {
      UNUSED(target);
      UNUSED(prediction);
      return 9999999.99;
   }
};
