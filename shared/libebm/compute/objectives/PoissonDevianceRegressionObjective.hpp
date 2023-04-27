// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// TFloat is a datatype that could hold inside a double, float, or some SIMD intrinsic type.
// See sse2_32.cpp, cuda_32.cpp, and cpu_64.cpp as examples where TFloat operators are defined.
template<typename TFloat>
struct PoissonDevianceRegressionObjective : RegressionObjective {
   OBJECTIVE_BOILERPLATE(PoissonDevianceRegressionObjective, MINIMIZE_METRIC, Link_log)

   double m_maxDeltaStepExp;
   // The constructor parameters following config must match the RegisterObjective parameters in objective_registrations.hpp
   inline PoissonDevianceRegressionObjective(const Config & config, const double maxDeltaStep) {
      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
      }
      if(std::isnan(maxDeltaStep) || maxDeltaStep <= 0.0 || std::isinf(maxDeltaStep)) {
         throw ParamValOutOfRangeException();
      }
      const double maxDeltaStepExp = std::exp(maxDeltaStep);
      if(std::isinf(maxDeltaStepExp)) {
         throw ParamValOutOfRangeException();
      }
      m_maxDeltaStepExp = maxDeltaStepExp;
   }

   inline double LinkParam() const noexcept {
      return std::numeric_limits<double>::quiet_NaN();
   }

   inline double GradientConstant() const noexcept {
      return 1.0;
   }

   inline double HessianConstant() const noexcept {
      return m_maxDeltaStepExp;
   }

   inline double FinishMetric(const double metricSum) const noexcept {
      return 2.0 * metricSum;
   }

   //Different GBM Implementations
   // https://github.com/h2oai/h2o-3/blob/master/h2o-core/src/main/java/hex/DistributionFactory.java
   // https://github.com/microsoft/LightGBM/blob/master/src/objective/regression_objective.hpp
   //
   GPU_DEVICE inline TFloat CalcMetric(const TFloat score, const TFloat target) const noexcept {
      static const TFloat epsilon = 1e-8;

      const TFloat prediction = Exp(score); // log link function
      const TFloat error = prediction - target;
      const TFloat extra = target * Log(target / prediction);
      const TFloat conditionalExtra = IfLess(target, epsilon, 0.0, extra);
      return error + conditionalExtra;
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = Exp(score); // log link function
      const TFloat gradient = prediction - target;
      return gradient;
   }

   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = Exp(score); // log link function
      const TFloat gradient = prediction - target;
      const TFloat hessian = prediction;
      return MakeGradientHessian(gradient, hessian);
   }
};
