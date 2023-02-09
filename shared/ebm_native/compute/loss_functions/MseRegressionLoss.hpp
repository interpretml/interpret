// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!

// MseRegressionLoss is a VERY VERY special Loss function.  
// Anyone writing a custom loss function in C++ should start from a different loss function

#include "Loss.hpp"

// TFloat could be double, float, or some SIMD intrinsic type
template<typename TFloat>
struct MseRegressionLoss : public RegressionLoss {

   INLINE_ALWAYS MseRegressionLoss(const Config & config) {
      if(1 != config.cOutputs) {
         throw ParamMismatchWithConfigException();
      }
   }

   // MSE is super super special in that we can calculate the new gradient from the old gradient without
   // preserving the score.  This is benefitial because we can eliminate the memory access to the score
   // so we'd use the equivalent of "GetGradientFromGradientPrev(const T gradientPrev)", but we need to special
   // case all of it anyways.

//   // for MSE regression, we get target - score at initialization and only store the gradients, so we never
//   // make a prediction, so we don't need InverseLinkFunction(...)
//
//   template <typename T>
//   INLINE_ALWAYS static T GetGradientFromGradientPrev(const T target, const T gradientPrev) {
//      // for MSE regression, we get target - score at initialization and only store the gradients, so we
//      // never need the targets.  We just work from the previous gradients.
//
//      return -100000;
//   }



   //bool IsSuperSuperSpecialLossWhereTargetNotNeededOnlyMseLossQualifies() const override {
   //   // TODO: use this property!
   //   return true;
   //}
};
