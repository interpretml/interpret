// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// TFloat is a datatype that could hold inside a double, float, or some SIMD intrinsic type.
// See cpu_64.cpp, avx2_32.cpp, and cuda_32.cu as examples where TFloat operators are defined.
template<typename TFloat> struct GammaDevianceRegressionObjective : RegressionObjective {
   OBJECTIVE_BOILERPLATE(GammaDevianceRegressionObjective, MINIMIZE_METRIC, Objective_Other, Link_log, true)

   inline GammaDevianceRegressionObjective(const Config& config) {
      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
      }

      if(config.isDifferentialPrivacy) {
         throw NonPrivateRegistrationException();
      }
   }

   inline bool CheckRegressionTarget(const double target) const noexcept {
      return std::isnan(target) || std::isinf(target) || target <= 0.0;
   }

   inline double LinkParam() const noexcept { return std::numeric_limits<double>::quiet_NaN(); }

   inline double LearningRateAdjustmentDifferentialPrivacy() const noexcept {
      return 1.0; // typically leave this at 1.0 (unmodified)
   }

   inline double LearningRateAdjustmentGradientBoosting() const noexcept {
      return 1.0; // typically leave this at 1.0 (unmodified)
   }

   inline double LearningRateAdjustmentHessianBoosting() const noexcept {
      return 1.0; // typically leave this at 1.0 (unmodified)
   }

   inline double GainAdjustmentGradientBoosting() const noexcept {
      return 1.0; // typically leave this at 1.0 (unmodified)
   }

   inline double GainAdjustmentHessianBoosting() const noexcept {
      return 1.0; // typically leave this at 1.0 (unmodified)
   }

   inline double GradientConstant() const noexcept { return 1.0; }

   inline double HessianConstant() const noexcept { return 1.0; }

   inline double FinishMetric(const double metricSum) const noexcept { return 2.0 * metricSum; }

   GPU_DEVICE inline TFloat CalcMetric(const TFloat& score, const TFloat& target) const noexcept {
      const TFloat invPrediction = Exp<true>(score); // log link function
      const TFloat frac = target * invPrediction;
      const TFloat metric = frac - 1.0 - Log(frac);
      return metric;
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat& score, const TFloat& target) const noexcept {
      const TFloat invPrediction = Exp<true>(score); // log link function
      const TFloat gradient = FusedNegateMultiplyAdd(target, invPrediction, 1.0);
      return gradient;
   }

   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(
         const TFloat& score, const TFloat& target) const noexcept {
      const TFloat invPrediction = Exp<true>(score); // log link function
      const TFloat gradient = FusedNegateMultiplyAdd(target, invPrediction, 1.0);
      const TFloat hessian = target * invPrediction;
      return MakeGradientHessian(gradient, hessian);
   }
};
