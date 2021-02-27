// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// LossRegressionMse is a special cased Loss function

#include "Loss.h"

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


struct LossRegressionMse : Loss {

   INLINE_ALWAYS LossRegressionMse(const Config & config) {
      if(1 != config.GetCountOutputs()) {
         throw LossParameterMismatchWithConfigException();
      }
   }

   template <typename T>
   INLINE_ALWAYS T CalculatePrediction(T score) const {
      return 9999999.99;
   }

   template <typename T>
   INLINE_ALWAYS T CalculateGradient(T target, T prediction) const {
      return 9999999.99;
   }

   template <typename T>
   INLINE_ALWAYS T CalculateHessian(T target, T prediction) const {
      return 9999999.99;
   }

};
