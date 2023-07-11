// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// TFloat is a datatype that could hold inside a double, float, or some SIMD intrinsic type.
// See sse2_32.cpp, cuda_32.cpp, and cpu_64.cpp as examples where TFloat operators are defined.
template<typename TFloat>
struct PoissonDevianceRegressionObjective : RegressionObjective {
   OBJECTIVE_BOILERPLATE(PoissonDevianceRegressionObjective, MINIMIZE_METRIC, Link_log)

   // The constructor parameters following config must match the RegisterObjective parameters in objective_registrations.hpp
   inline PoissonDevianceRegressionObjective(const Config & config) {
      // XGBoost and LightGBM have a max_delta_step parameter which is set to 0.7 by default. This parameter's only
      // actual effect is that it changes the effective learning rate by a multiple of 1/e^max_delta_step, which
      // is a parameter that you can set already from the public interface.
      // They add this parameter value to the prediction before taking the exp for the hessian calculation, but you can 
      // extract it to outside the exp by multiplying the hessian by e^max_delta_step instead.
      // To get an equivalent rate to XGBoost/LightGBM for Poisson, multiply our learning rate by
      // 0.49658530379140951.

      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
      }

      if(config.isDifferentialPrivacy) {
         throw NonPrivateRegistrationException();
      }
   }

   inline bool CheckRegressionTarget(const double target) const noexcept {
      return std::isnan(target) || std::isinf(target) || target < 0.0;
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

   //Different GBM Implementations
   // https://github.com/h2oai/h2o-3/blob/master/h2o-core/src/main/java/hex/DistributionFactory.java
   // https://github.com/microsoft/LightGBM/blob/master/src/objective/regression_objective.hpp
   //
   GPU_DEVICE inline TFloat CalcMetric(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = Exp(score); // log link function
      const TFloat error = prediction - target;
      const TFloat frac = target / prediction;
      const TFloat extra = target * Log(frac);
      // target == 0 is legal in poisson and we need to treat the extra variable as a zero in that case as if the
      // Log succeeded and was subsequently multiplied by 0. But due to floating point considerations the
      // fraction inside the log can underflow to zero or a denormal if target is a small non-zero number. We want to 
      // treat denormals as zero, so check if less than the minimum non-denormal as our check for zero.
      const TFloat conditionalExtra = IfLess(frac, std::numeric_limits<typename TFloat::T>::min(), 0.0, extra);
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
