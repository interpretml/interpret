// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// Steps for adding a new objective in C++:
//   1) Copy one of the existing Loss*.h include files (like this one) into a new renamed Loss*.h file
//   2) Modify the class below to handle your new Loss function
//   3) Add [#include "Loss*.h"] to the list of other include files near the top of the Loss.cpp file
//   4) Add the Loss* type to the list of loss registrations in RegisterLosses() near the top of the Loss.cpp file
//   5) Recompile the C++ with either build.sh or build.bat depending on your operating system
//   6) Enjoy your new Loss function, and send us a PR on Github if you think others would benefit  :-)

#include "Loss.h"

struct LossRegressionPseudoHuber final : public Loss {

   FloatEbmType m_deltaInverted;

   // IMPORTANT: the constructor parameters here must match the LossRegistration parameters in the file Loss.cpp
   INLINE_ALWAYS LossRegressionPseudoHuber(const Config & config, const FloatEbmType delta) {
      if(1 != config.GetCountOutputs()) {
         throw LossParameterMismatchWithConfigException();
      }

      if(FloatEbmType { 0 } == delta || std::isnan(delta) || std::isinf(delta)) {
         throw LossParameterValueOutOfRangeException();
      }

      const FloatEbmType deltaInverted = FloatEbmType { 1 } / delta;
      if(std::isinf(deltaInverted)) {
         throw LossParameterValueOutOfRangeException();
      }

      m_deltaInverted = deltaInverted;
   }

   template <typename T>
   INLINE_ALWAYS T CalculatePrediction(const T score) const {
      return score;
   }

   template <typename T>
   INLINE_ALWAYS T CalculateGradient(const T target, const T prediction) const {
      const T residualNegative = prediction - target;
      const T residualNegativeFraction = residualNegative * static_cast<T>(m_deltaInverted);
      const T calc = T { 1 } + residualNegativeFraction * residualNegativeFraction;
      const T sqrtCalc = std::sqrt(calc);
      return residualNegative / sqrtCalc;
   }

   template <typename T>
   INLINE_ALWAYS T CalculateHessian(const T target, const T prediction) const {
      const T residualNegative = prediction - target;
      const T residualNegativeFraction = residualNegative * static_cast<T>(m_deltaInverted);
      const T calc = T { 1 } + residualNegativeFraction * residualNegativeFraction;
      const T sqrtCalc = std::sqrt(calc);
      return T { 1 } / (calc * sqrtCalc);
   }

   LOSS_DEFAULT_MECHANICS_PUT_AT_END_OF_CLASS
};
