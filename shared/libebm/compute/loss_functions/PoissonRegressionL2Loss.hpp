// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!

// DO NOT INCLUDE FILES IN THIS FILE. They will not be zoned properly. Include files go into the operator *.cpp files.

// TFloat is a datatype that could hold inside a double, float, or some SIMD intrinsic type.
// See sse2_32.cpp, cuda_32.cpp, and cpu_64.cpp as examples where TFloat operators are defined.
template<typename TFloat>
struct PoissonRegressionL2Loss : RegressionLoss {
   LOSS_BOILERPLATE(PoissonRegressionL2Loss, Link_log)

   TFloat m_maxDeltaStep;
   // The constructor parameters following config must match the RegisterLoss parameters in loss_registrations.hpp
   inline PoissonRegressionL2Loss(const Config & config, const double maxDeltaStep) {
      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
      }
      if(maxDeltaStep == 0.0 || std::isnan(maxDeltaStep) || std::isinf(maxDeltaStep)) {
         throw ParamValOutOfRangeException();
      }
      m_maxDeltaStep = maxDeltaStep;
   }


   inline double GradientMultiple() const noexcept {
      return 1.0;
   }

   inline double HessianMultiple() const noexcept {
      return 1.0;
   }

   GPU_DEVICE inline TFloat InverseLinkFunction(const TFloat score) const noexcept {
      // Poisson regression uses a log link function
      return Exp(score);
   }
   //Different GBM Implementations
   // https://github.com/h2oai/h2o-3/blob/master/h2o-core/src/main/java/hex/DistributionFactory.java
   // https://github.com/microsoft/LightGBM/blob/master/src/objective/regression_objective.hpp
   //
   GPU_DEVICE inline TFloat CalcMetric(const TFloat prediction, const TFloat target) const noexcept {
      const TFloat epsilon = static_cast<TFloat>(1e-8);
      const TFloat trueVal = 2 * (prediction - target);
      const TFloat falseVal = -2 * (target * Log(prediction / target) - (prediction - target));
      return IfLess(target, epsilon, trueVal, falseVal);
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat prediction, const TFloat target) const noexcept {
      TFloat gradient = prediction - target;
      return gradient;
   }

   // If the loss function doesn't have a second derivative, then delete the CalcGradientHessian function.
   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat prediction, const TFloat target) const noexcept {
      TFloat gradient = prediction - target;
      TFloat hessian = prediction * m_maxDeltaStep;
      return MakeGradientHessian(gradient, hessian);
   }
};
