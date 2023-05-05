// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// TFloat is a datatype that could hold inside a double, float, or some SIMD intrinsic type.
// See sse2_32.cpp, cuda_32.cpp, and cpu_64.cpp as examples where TFloat operators are defined.
template<typename TFloat>
struct TweedieDevianceRegressionObjective : RegressionObjective {
   OBJECTIVE_BOILERPLATE(TweedieDevianceRegressionObjective, MINIMIZE_METRIC, Link_log)

   TFloat m_variancePowerParamSub1;
   TFloat m_variancePowerParamSub2;
   TFloat m_inverseVariancePowerParamSub1;
   TFloat m_inverseVariancePowerParamSub2;

   // The constructor parameters following config must match the RegisterObjective parameters in objective_registrations.hpp
   inline TweedieDevianceRegressionObjective(const Config & config, double variancePower) {
      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
      }
      if(config.isDifferentiallyPrivate) {
         throw NonPrivateRegistrationException();
      }
      if(variancePower <= 1.0 || 2.0 <= variancePower) {
         // TODO: Implement Tweedie for other Powers
         throw ParamValOutOfRangeException();
      }
      // for a discussion on variance_power and link_power, see:
      // https://search.r-project.org/CRAN/refmans/statmod/html/tweedie.html
      // https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/tweedie_link_power.html

      const double variancePowerParamSub1 = 1.0 - variancePower;
      const double variancePowerParamSub2 = 2.0 - variancePower;

      m_variancePowerParamSub1 = variancePowerParamSub1;
      m_variancePowerParamSub2 = variancePowerParamSub2;
      m_inverseVariancePowerParamSub1 = 1.0 / variancePowerParamSub1;
      m_inverseVariancePowerParamSub2 = 1.0 / variancePowerParamSub2;
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
      return 2.0 * metricSum;
   }

   GPU_DEVICE inline TFloat CalcMetric(const TFloat score, const TFloat target) const noexcept {
      const TFloat exp1Score = Exp(m_variancePowerParamSub1 * score);
      const TFloat exp2Score = Exp(m_variancePowerParamSub2 * score);
      const TFloat metric = exp2Score * m_inverseVariancePowerParamSub2 - target * m_inverseVariancePowerParamSub1 * exp1Score;
      return metric;
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat score, const TFloat target) const noexcept {
      const TFloat exp1Score = Exp(m_variancePowerParamSub1 * score);
      const TFloat exp2Score = Exp(m_variancePowerParamSub2 * score);
      const TFloat targetTimesExp1Score = target * exp1Score;
      const TFloat gradient = exp2Score - targetTimesExp1Score;
      return gradient;
   }

   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat score, const TFloat target) const noexcept {
      const TFloat exp1Score = Exp(m_variancePowerParamSub1 * score);
      const TFloat exp2Score = Exp(m_variancePowerParamSub2 * score);
      const TFloat targetTimesExp1Score = target * exp1Score;
      const TFloat gradient = exp2Score - targetTimesExp1Score;
      const TFloat hessian = m_variancePowerParamSub2 * exp2Score - m_variancePowerParamSub1 * targetTimesExp1Score;
      return MakeGradientHessian(gradient, hessian);
   }
};
