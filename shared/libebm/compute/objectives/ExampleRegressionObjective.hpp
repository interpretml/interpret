// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// TFloat is a wrapper that could hold a double, float, or SIMD intrinsic type. It can also expose GPU operations.
// See sse2_32.cpp, cpu_64.cpp, and cuda_32.cu as examples where TFloat operators are defined.
template<typename TFloat>
struct ExampleRegressionObjective : RegressionObjective {
   OBJECTIVE_BOILERPLATE(ExampleRegressionObjective, Link_identity)

   TFloat m_param0;
   TFloat m_param1;

   // The constructor parameters following config must match the RegisterObjective parameters in objective_registrations.hpp
   inline ExampleRegressionObjective(const Config & config, const double param0, const double param1) {
      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
      }

      m_param0 = param0;
      m_param1 = param1;
   }

   inline double LinkParam() const noexcept {
      return std::numeric_limits<double>::quiet_NaN();
   }

   inline double GradientConstant() const noexcept {
      return 1.0;
   }

   inline double HessianConstant() const noexcept {
      return 1.0;
   }

   GPU_DEVICE inline TFloat InverseLinkFunction(const TFloat score) const noexcept {
      // Identity link function
      return score;
   }

   GPU_DEVICE inline TFloat CalcMetric(const TFloat prediction, const TFloat target) const noexcept {
      const TFloat error = prediction - target;
      return error * error;
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat prediction, const TFloat target) const noexcept {
      const TFloat error = prediction - target;
      // Alternatively, the 2.0 factor could be moved to GradientConstant()
      const TFloat gradient = 2.0 * error;
      return gradient;
   }

   // If the loss function doesn't have a second derivative, then delete the CalcGradientHessian function.
   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat prediction, const TFloat target) const noexcept {
      const TFloat error = prediction - target;
      // Alternatively, the 2.0 factor could be moved to GradientConstant() and HessianConstant()
      const TFloat gradient = 2.0 * error;
      const TFloat hessian = 2.0;
      return MakeGradientHessian(gradient, hessian);
   }
};
