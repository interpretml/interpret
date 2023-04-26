// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// TFloat is a datatype that could hold inside a double, float, or some SIMD intrinsic type.
// See sse2_32.cpp, cuda_32.cpp, and cpu_64.cpp as examples where TFloat operators are defined.
template<typename TFloat>
struct PseudoHuberRegressionObjective : RegressionObjective {
   OBJECTIVE_BOILERPLATE(PseudoHuberRegressionObjective, MINIMIZE_METRIC, Link_identity)

   TFloat m_deltaInverted;
   TFloat m_deltaSquared;

   // The constructor parameters following config must match the RegisterObjective parameters in objective_registrations.hpp
   inline PseudoHuberRegressionObjective(const Config & config, const double delta) {
      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
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

   inline double LinkParam() const noexcept {
      return std::numeric_limits<double>::quiet_NaN();
   }

   inline double GradientConstant() const noexcept {
      return 1.0;
   }

   inline double HessianConstant() const noexcept {
      return 1.0;
   }

   inline double FinishMetric(const double metricSum) const noexcept {
      return metricSum;
   }

   GPU_DEVICE inline TFloat CalcMetric(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = score; // identity link function
      const TFloat error = prediction - target;
      const TFloat errorFraction = error * m_deltaInverted;
      const TFloat calc = errorFraction * errorFraction + 1.0;
      const TFloat sqrtCalc = Sqrt(calc);
      const TFloat metric = m_deltaSquared * (sqrtCalc - 1.0);
      return metric;
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = score; // identity link function
      const TFloat error = prediction - target;
      const TFloat errorFraction = error * m_deltaInverted;
      const TFloat calc = errorFraction * errorFraction + 1.0;
      const TFloat sqrtCalc = Sqrt(calc);
      const TFloat gradient = error / sqrtCalc;
      return gradient;
   }

   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = score; // identity link function
      const TFloat error = prediction - target;
      const TFloat errorFraction = error * m_deltaInverted;
      const TFloat calc = errorFraction * errorFraction + 1.0;
      const TFloat sqrtCalc = Sqrt(calc);
      const TFloat gradient = error / sqrtCalc;
      const TFloat hessian = 1.0 / (calc * sqrtCalc);
      return MakeGradientHessian(gradient, hessian);
   }
};
