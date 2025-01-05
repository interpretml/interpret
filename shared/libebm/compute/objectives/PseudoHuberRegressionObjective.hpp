// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// TFloat is a datatype that could hold inside a double, float, or some SIMD intrinsic type.
// See cpu_64.cpp, avx2_32.cpp, and cuda_32.cu as examples where TFloat operators are defined.
template<typename TFloat> struct PseudoHuberRegressionObjective : RegressionObjective {
   OBJECTIVE_BOILERPLATE(PseudoHuberRegressionObjective, MINIMIZE_METRIC, Objective_Other, Link_identity, true)

   TFloat m_deltaInverted;
   double m_deltaSquared;

   // The constructor parameters following config must match the RegisterObjective parameters in
   // objective_registrations.hpp
   inline PseudoHuberRegressionObjective(const Config& config, const double delta) {
      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
      }

      if(config.isDifferentialPrivacy) {
         throw NonPrivateRegistrationException();
      }

      if(std::isnan(delta) || delta <= 0.0 || std::isinf(delta)) {
         throw ParamValOutOfRangeException();
      }

      const double deltaSquared = delta * delta;
      if(std::isinf(deltaSquared)) {
         throw ParamValOutOfRangeException();
      }
      m_deltaSquared = deltaSquared;

      const double deltaInverted = 1.0 / delta;
      if(std::isinf(deltaInverted)) {
         throw ParamValOutOfRangeException();
      }
      m_deltaInverted = deltaInverted;
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

   inline double FinishMetric(const double metricSum) const noexcept { return m_deltaSquared * metricSum; }

   GPU_DEVICE inline TFloat CalcMetric(const TFloat& score, const TFloat& target) const noexcept {
      const TFloat prediction = score; // identity link function
      const TFloat error = prediction - target;
      const TFloat errorFraction = error * m_deltaInverted;
      const TFloat calc = FusedMultiplyAdd(errorFraction, errorFraction, 1.0);
      const TFloat sqrtCalc = Sqrt(calc);
      const TFloat metric = sqrtCalc - 1.0; // TODO: this subtraction of 1.0 could be moved to FinishMetric if we passed
                                            // the total outer bag weight to FinishMetric
      return metric;
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat& score, const TFloat& target) const noexcept {
      const TFloat prediction = score; // identity link function
      const TFloat error = prediction - target;
      const TFloat errorFraction = error * m_deltaInverted;
      const TFloat calc = FusedMultiplyAdd(errorFraction, errorFraction, 1.0);
      const TFloat sqrtCalc = Sqrt(calc);
      const TFloat gradient = FastApproxDivide(error, sqrtCalc);
      return gradient;
   }

   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(
         const TFloat& score, const TFloat& target) const noexcept {
      const TFloat prediction = score; // identity link function
      const TFloat error = prediction - target;
      const TFloat errorFraction = error * m_deltaInverted;
      const TFloat calc = FusedMultiplyAdd(errorFraction, errorFraction, 1.0);
      const TFloat sqrtCalc = Sqrt(calc);
      const TFloat gradient = FastApproxDivide(error, sqrtCalc);
      const TFloat hessian = FastApproxReciprocal(calc * sqrtCalc);
      return MakeGradientHessian(gradient, hessian);
   }
};
