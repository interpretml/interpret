// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// TFloat is a datatype that could hold inside a double, float, or some SIMD intrinsic type.
// See cpu_64.cpp, avx2_32.cpp, and cuda_32.cu as examples where TFloat operators are defined.
template<typename TFloat> struct RmseLogLinkRegressionObjective : RegressionObjective {
   OBJECTIVE_BOILERPLATE(RmseLogLinkRegressionObjective, MINIMIZE_METRIC, Link_log, false)

   inline RmseLogLinkRegressionObjective(const Config& config) {
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

   inline double LinkParam() const noexcept { return std::numeric_limits<double>::quiet_NaN(); }

   inline double LearningRateAdjustmentDifferentialPrivacy() const noexcept {
      // we follow the gradient adjustment for DP since we have a similar change in rate and we want to make
      // our results comparable. The DP paper uses this adjusted rate.

      // WARNING: do not change this rate without accounting for it in the privacy budget!
      return 0.5;
   }

   inline double LearningRateAdjustmentGradientBoosting() const noexcept {
      // the hessian is 2.0 for RMSE. If we change to gradient boosting we divide by the weight/count which is
      // normalized to 1, so we double the effective learning rate without this adjustment.  We want
      // gradient boosting and hessian boosting to have similar rates, and this adjustment makes it that way
      return 0.5;
   }

   inline double LearningRateAdjustmentHessianBoosting() const noexcept {
      // this is the reference point
      return 1.0;
   }

   inline double GainAdjustmentGradientBoosting() const noexcept {
      // the hessian is 2.0 for RMSE. If we change to gradient boosting we divide by the weight/count which is
      // normalized to 1, so we double the effective gain without this adjustment.  We want
      // gradient boosting and hessian boosting to have similar rates, and this adjustment makes it that way
      return 0.5;
   }

   inline double GainAdjustmentHessianBoosting() const noexcept {
      // this is the reference point
      return 1.0;
   }

   inline double GradientConstant() const noexcept { return 2.0; }

   inline double HessianConstant() const noexcept { return 2.0; }

   inline double FinishMetric(const double metricSum) const noexcept {
      return std::sqrt(metricSum); // finish the 'r' in 'rmse'
   }

   GPU_DEVICE inline TFloat CalcMetric(const TFloat& score, const TFloat& target) const noexcept {
      const TFloat prediction = Exp(score); // log link function
      const TFloat error = prediction - target;
      const TFloat metric = error * error;
      return metric;
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat& score, const TFloat& target) const noexcept {
      const TFloat prediction = Exp(score); // log link function
      const TFloat gradient = prediction - target;
      return gradient;
   }
};
