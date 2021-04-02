// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !!! NOTE: To add a new loss/objective function in C++, follow the steps listed at the top of the "Loss.cpp" file !!!

#include "Loss.hpp"


//class LossBinaryLogLoss final : public LossBinary {
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

// TFloat could be double, float, or some SIMD intrinsic type
template<typename TFloat>
struct LossBinaryLogLoss : public LossBinary {

   // IMPORTANT: the constructor parameters here must match the RegisterLoss parameters in the file Loss.cpp
   INLINE_ALWAYS LossBinaryLogLoss(const Config & config) {
      if(1 != config.cOutputs) {
         // we share the tag "log_loss" with multiclass classification
         throw SkipRegistrationException();
      }
   }

   INLINE_ALWAYS TFloat CalculatePrediction(TFloat score) const {
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
