// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// TFloat is a wrapper that could hold a double, float, or SIMD intrinsic type. It can also expose GPU operations.
// See sse2_32.cpp, cpu_64.cpp, and cuda_32.cu as examples where TFloat operators are defined.
template<typename TFloat>
struct ExampleRegressionObjective : RegressionObjective {
   OBJECTIVE_BOILERPLATE(ExampleRegressionObjective, MINIMIZE_METRIC, Link_identity)

   static constexpr double Two = 2.0;

   TFloat m_param0;
   TFloat m_param1;

   // The constructor parameters following config must match the RegisterObjective parameters in objective_registrations.hpp
   inline ExampleRegressionObjective(const Config & config, const double param0, const double param1) {
      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
      }

      if(config.isDifferentiallyPrivate) {
         throw NonPrivateRegistrationException();
      }

      m_param0 = param0;
      m_param1 = param1;
   }

   inline double LinkParam() const noexcept {
      // only Link_power and the custom link functions use the LinkParam
      return std::numeric_limits<double>::quiet_NaN();
   }

   inline double LearningRateAdjustmentDifferentialPrivacy() const noexcept {
      // WARNING: Do not change this rate without accounting for it in the privacy budget if this objective supports DP
      // typically leave this at 1.0 (unmodified)
      return 1.0;
   }

   inline double LearningRateAdjustmentGradientBoosting() const noexcept {
      // typically leave this at 1.0 (unmodified)
      return 1.0;
   }

   inline double LearningRateAdjustmentHessianBoosting() const noexcept {
      // typically leave this at 1.0 (unmodified)
      return 1.0;
   }

   inline double GainAdjustmentGradientBoosting() const noexcept {
      // typically leave this at 1.0 (unmodified)
      return 1.0;
   }

   inline double GainAdjustmentHessianBoosting() const noexcept {
      // typically leave this at 1.0 (unmodified)
      return 1.0;
   }

   inline double GradientConstant() const noexcept {
      return 1.0;
   }

   inline double HessianConstant() const noexcept {
      return 1.0;
   }

   inline double FinishMetric(const double metricSum) const noexcept {
      return metricSum; // return MSE, but if we wanted to return RMSE we would take the sqrt here
   }

   GPU_DEVICE inline TFloat CalcMetric(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = score; // identity link function
      const TFloat error = prediction - target;
      return error * error;
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = score; // identity link function
      const TFloat error = prediction - target;
      // Alternatively, the 2.0 factor could be moved to GradientConstant()
      const TFloat gradient = Two * error;
      return gradient;
   }

   // If the loss function doesn't have a second derivative, then delete the CalcGradientHessian function.
   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = score; // identity link function
      const TFloat error = prediction - target;
      // Alternatively, the 2.0 factors could be moved to GradientConstant() and HessianConstant()
      const TFloat gradient = Two * error;
      const TFloat hessian = Two;
      return MakeGradientHessian(gradient, hessian);
   }
};
