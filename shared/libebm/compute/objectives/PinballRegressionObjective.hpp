// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// TFloat is a datatype that could hold inside a double, float, or some SIMD intrinsic type.
// See cpu_64.cpp, avx2_32.cpp, and cuda_32.cu as examples where TFloat operators are defined.
template<typename TFloat> struct PinballRegressionObjective : RegressionObjective {
   OBJECTIVE_BOILERPLATE(PinballRegressionObjective, MINIMIZE_METRIC, Objective_Other, Link_identity, false)

   TFloat m_alpha;
   TFloat m_oneMinusAlpha;

   // The constructor parameters following config must match the RegisterObjective parameters in
   // objective_registrations.hpp
   inline PinballRegressionObjective(const Config& config, const double alpha) {
      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
      }

      if(config.isDifferentialPrivacy) {
         throw NonPrivateRegistrationException();
      }

      if(std::isnan(alpha) || alpha <= 0.0 || 1.0 <= alpha) {
         throw ParamValOutOfRangeException();
      }

      m_alpha = alpha;
      m_oneMinusAlpha = 1.0 - alpha;
   }

   inline bool CheckRegressionTarget(const double target) const noexcept {
      return std::isnan(target) || std::isinf(target);
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

   inline double FinishMetric(const double metricSum) const noexcept { return metricSum; }

   GPU_DEVICE inline TFloat CalcMetric(const TFloat& score, const TFloat& target) const noexcept {
      const TFloat prediction = score; // identity link function
      const TFloat error = prediction - target;
      return IfThenElse(error < 0.0, -m_alpha * error, m_oneMinusAlpha * error);
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat& score, const TFloat& target) const noexcept {
      const TFloat prediction = score; // identity link function
      const TFloat error = prediction - target;
      return IfThenElse(error < 0.0, -m_alpha, m_oneMinusAlpha);
   }
};
