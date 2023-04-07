// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!

// TFloat is a wrapper that could hold a double, float, or SIMD intrinsic type. It can also expose GPU operations.
// See sse2_32.cpp, cpu_64.cpp, and cuda_32.cu as examples where TFloat operators are defined.
template<typename TFloat>
struct ExampleRegressionLoss : RegressionLoss {
   LOSS_BOILERPLATE(ExampleRegressionLoss)

   TFloat m_param0;
   TFloat m_param1;

   // The constructor parameters following config must match the RegisterLoss parameters in loss_registrations.hpp
   inline ExampleRegressionLoss(const Config & config, const double param0, const double param1) {
      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
      }

      m_param0 = param0;
      m_param1 = param1;
   }

   inline double GradientMultiple() const noexcept {
      return 1.0;
   }

   inline double HessianMultiple() const noexcept {
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
      return error;
   }

   // If the loss function doesn't have a second derivative, then delete the CalcGradientHessian function.
   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat prediction, const TFloat target) const noexcept {
      const TFloat error = prediction - target;
      return MakeGradientHessian(error, 1.0);
   }
};
