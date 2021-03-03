// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !!! NOTE: To add a new loss/objective function in C++, follow the steps listed at the top of the "Loss.cpp" file !!!

#include "Loss.h"

struct LossRegressionPseudoHuber : Loss {

   FloatEbmType m_deltaInverted;

   // IMPORTANT: the constructor parameters here must match the RegisterLoss parameters in the file Loss.cpp
   INLINE_ALWAYS LossRegressionPseudoHuber(const Config & config, FloatEbmType delta) {
      if(1 != config.GetCountOutputs()) {
         throw ParameterMismatchWithConfigException();
      }

      if(0 == delta || std::isnan(delta) || std::isinf(delta)) {
         throw ParameterValueOutOfRangeException();
      }

      FloatEbmType deltaInverted = FloatEbmType { 1 } / delta;
      if(std::isinf(deltaInverted)) {
         throw ParameterValueOutOfRangeException();
      }

      m_deltaInverted = deltaInverted;
   }

   template <typename T>
   INLINE_ALWAYS T CalculatePrediction(T score) const {
      return score;
   }

   template <typename T>
   INLINE_ALWAYS T CalculateGradient(T target, T prediction) const {
      T residualNegative = prediction - target;
      T residualNegativeFraction = residualNegative * static_cast<T>(m_deltaInverted);
      T calc = T { 1 } + residualNegativeFraction * residualNegativeFraction;
      T sqrtCalc = std::sqrt(calc);
      return residualNegative / sqrtCalc;
   }

   template <typename T>
   INLINE_ALWAYS T CalculateHessian(T target, T prediction) const {
      T residualNegative = prediction - target;
      T residualNegativeFraction = residualNegative * static_cast<T>(m_deltaInverted);
      T calc = T { 1 } + residualNegativeFraction * residualNegativeFraction;
      T sqrtCalc = std::sqrt(calc);
      return T { 1 } / (calc * sqrtCalc);
   }

   LOSS_DEFAULT_MECHANICS_PUT_AT_END_OF_CLASS
};
