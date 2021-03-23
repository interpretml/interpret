// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !!! NOTE: To add a new loss/objective function in C++, follow the steps listed at the top of the "Loss.cpp" file !!!

#include "Loss.h"

// TFloat could be double, float, or some SIMD intrinsic type
template <typename TFloat>
struct LossRegressionPseudoHuber : public LossRegression {

   TFloat m_deltaInverted;

   // IMPORTANT: the constructor parameters here must match the RegisterLoss parameters in the file Loss.cpp
   INLINE_ALWAYS LossRegressionPseudoHuber(const Config & config, TFloat delta) {
      if(1 != config.GetCountOutputs()) {
         throw ParameterMismatchWithConfigException();
      }

      if(delta.IsAnyEqual(TFloat(0)) || delta.IsAnyNaN() || delta.IsAnyInf()) {
         throw ParameterValueOutOfRangeException();
      }

      TFloat deltaInverted = TFloat(1) / delta;
      if(deltaInverted.IsAnyInf()) {
         throw ParameterValueOutOfRangeException();
      }

      m_deltaInverted = deltaInverted;
   }

   INLINE_ALWAYS TFloat CalculatePrediction(TFloat score) const {
      return score;
   }

   INLINE_ALWAYS TFloat CalculateGradient(TFloat target, TFloat prediction) const {
      TFloat residualNegative = prediction - target;
      TFloat residualNegativeFraction = residualNegative * m_deltaInverted;
      TFloat calc = TFloat(1) + residualNegativeFraction * residualNegativeFraction;
      TFloat sqrtCalc = calc.Sqrt();
      // the calculations above are shared with the hessian, so the compiler should combine them.
      return residualNegative / sqrtCalc;
   }

   // if the loss function doesn't have a second derivative, then delete the CalculateHessian function.
   INLINE_ALWAYS TFloat CalculateHessian(TFloat target, TFloat prediction) const {
      TFloat residualNegative = prediction - target;
      TFloat residualNegativeFraction = residualNegative * m_deltaInverted;
      TFloat calc = TFloat(1) + residualNegativeFraction * residualNegativeFraction;
      TFloat sqrtCalc = calc.Sqrt();
      // the calculations above are shared with the hessian, so the compiler should combine them.
      return TFloat(1) / (calc * sqrtCalc);
   }

   LOSS_CLASS_BOILERPLATE_PUT_AT_END_OF_CLASS(true)
};
