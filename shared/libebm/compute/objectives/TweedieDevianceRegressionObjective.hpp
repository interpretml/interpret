// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// TFloat is a datatype that could hold inside a double, float, or some SIMD intrinsic type.
// See sse2_32.cpp, cuda_32.cpp, and cpu_64.cpp as examples where TFloat operators are defined.
template<typename TFloat>
struct TweedieDevianceRegressionObjective : RegressionObjective {
   OBJECTIVE_BOILERPLATE(TweedieDevianceRegressionObjective, MINIMIZE_METRIC, Link_log)

   double m_variancePowerParam;

   // The constructor parameters following config must match the RegisterObjective parameters in objective_registrations.hpp
   inline TweedieDevianceRegressionObjective(const Config & config, double variancePower) {
      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
      }
      if(config.isDifferentiallyPrivate) {
         throw NonPrivateRegistrationException();
      }
      if(variancePower < 1.0 || variancePower > 2.0) {
         // TODO: Implement Tweedie for other Powers. For now skip this registration
         throw SkipRegistrationException();
      }
      // for a discussion on variance_power and link_power, see:
      // https://search.r-project.org/CRAN/refmans/statmod/html/tweedie.html
      // https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/tweedie_link_power.html

      m_variancePowerParam = variancePower;
   }

   inline double LinkParam() const noexcept {
      return std::numeric_limits<double>::quiet_NaN();
   }

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

   inline double GradientConstant() const noexcept {
      return 1.0;
   }

   inline double HessianConstant() const noexcept {
      return 1.0;
   }

   inline double FinishMetric(const double metricSum) const noexcept {
      return 2*metricSum;
   }

   GPU_DEVICE inline TFloat CalcMetric(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = Exp(score); // log link function
      const TFloat metric = -target * Exp((1 - m_variancePowerParam) * score) / (1 - m_variancePowerParam)  + Exp((2 - m_variancePowerParam) * score) / (2 - m_variancePowerParam);
      //const TFloat metric = 1
      return metric;
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = Exp(score); // log link function
      const TFloat exp1Score = Exp((1 - m_variancePowerParam) * score);
      const TFloat exp2Score = Exp((2 - m_variancePowerParam) * score);
      const TFloat gradient = (-target * exp1Score + exp2Score);
      //const TFloat gradient = 1
      return gradient;
   }

   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = Exp(score); // log link function
      const TFloat exp1Score = Exp((1 - m_variancePowerParam) * score);
      const TFloat exp2Score = Exp((2 - m_variancePowerParam) * score);
      const TFloat gradient = (-target * exp1Score + exp2Score);
      const TFloat hessian = (-target * (1 - m_variancePowerParam) * exp1Score + (2 - m_variancePowerParam) * exp2Score);
      return MakeGradientHessian(gradient, hessian);
   }
};
