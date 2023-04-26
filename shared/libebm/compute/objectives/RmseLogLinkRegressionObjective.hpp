// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// TFloat is a datatype that could hold inside a double, float, or some SIMD intrinsic type.
// See sse2_32.cpp, cuda_32.cpp, and cpu_64.cpp as examples where TFloat operators are defined.
template<typename TFloat>
struct RmseLogLinkRegressionObjective : RegressionObjective {
   OBJECTIVE_BOILERPLATE(RmseLogLinkRegressionObjective, MINIMIZE_METRIC, Link_log)

   inline RmseLogLinkRegressionObjective(const Config & config) {
      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
      }
   }

   inline double LinkParam() const noexcept {
      return std::numeric_limits<double>::quiet_NaN();
   }

   inline double GradientConstant() const noexcept {
      return 2.0;
   }

   inline double HessianConstant() const noexcept {
      return 2.0;
   }

   inline double FinishMetric(const double metricSum) const noexcept {
      return std::sqrt(metricSum); // finish the 'r' in 'rmse'
   }

   GPU_DEVICE inline TFloat CalcMetric(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = Exp(score); // log link function
      const TFloat metric = (prediction - target) * (prediction - target);
      return metric;
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = Exp(score); // log link function
      const TFloat gradient = prediction - target;
      return gradient;
   }

   // If the loss function doesn't have a second derivative, then delete the CalcGradientHessian function.
   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat score, const TFloat target) const noexcept {
      // TODO: we can eliminate this function once we support that for non-rmse
      const TFloat prediction = Exp(score); // log link function
      const TFloat gradient = prediction - target;
      return MakeGradientHessian(gradient, 1.0);
   }
};
