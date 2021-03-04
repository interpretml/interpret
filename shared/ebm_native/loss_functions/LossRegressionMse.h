// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// LossRegressionMse is a VERY VERY special Loss function.  
// Anyone writing a custom loss function in C++ should start from a different loss function

#include "Loss.h"

struct LossRegressionMse : public LossRegression {

   INLINE_ALWAYS LossRegressionMse(const Config & config) {
      if(1 != config.GetCountOutputs()) {
         throw ParameterMismatchWithConfigException();
      }
   }

   // MSE is super super special in that we can calculate the new gradient from the old gradient without
   // preserving the score.  This is benefitial because we can eliminate the memory access to the score
   // so we'd use the equivalent of "GetGradientFromGradientPrev(const T gradientPrev)", but we need to special
   // case all of it anyways.

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



   bool IsSuperSuperSpecialLossWhereTargetNotNeededOnlyMseLossQualifies() const override {
      // TODO: use this property!
      return true;
   }
};
