// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// TFloat is a datatype that could hold inside a double, float, or some SIMD intrinsic type.
// See sse2_32.cpp, cuda_32.cpp, and cpu_64.cpp as examples where TFloat operators are defined.
template<typename TFloat>
struct TweedieRegressionObjective : RegressionObjective {
   OBJECTIVE_BOILERPLATE(TweedieRegressionObjective, MINIMIZE_METRIC, Link_power)

   double m_linkParam;

   // The constructor parameters following config must match the RegisterObjective parameters in objective_registrations.hpp
   inline TweedieRegressionObjective(const Config & config, double variancePower, double linkPower) {
      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
      }

      // for a discussion on variance_power and link_power, see:
      // https://search.r-project.org/CRAN/refmans/statmod/html/tweedie.html
      // https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/tweedie_link_power.html

      if(std::isnan(variancePower)) {
         if(std::isnan(linkPower)) {
            variancePower = 0.0;
            linkPower = 1.0;
         } else {
            variancePower = 1.0 - linkPower;
         }
      } else if(std::isnan(linkPower)) {
         linkPower = 1.0 - variancePower;
      }

      if(variancePower == 0.0 && linkPower == 1.0) {
         // TODO: skip this registration to allow a more specialized and optimized version of Tweedie to handle it
         // TODO: add any other common link parameter special casing
         // throw SkipRegistrationException();
      }

      m_linkParam = linkPower;

      // TODO: Implement Tweedie. For now skip this registration
      throw SkipRegistrationException();
   }

   inline double LinkParam() const noexcept {
      return m_linkParam;
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
      return metricSum;
   }

   GPU_DEVICE inline TFloat CalcMetric(const TFloat score, const TFloat target) const noexcept {
      UNUSED(score);
      UNUSED(target);

      const TFloat prediction = Exp(score); // log link function
      //Incomplete Implementation
      //const TFloat metric = 1
      return 1;
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat score, const TFloat target) const noexcept {
      UNUSED(score);
      UNUSED(target);

      const TFloat prediction = Exp(score); // log link function
      //Incomplete Implementation
      //const TFloat gradient = 1
      return 1;
   }

   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat score, const TFloat target) const noexcept {
      UNUSED(score);
      UNUSED(target);

      const TFloat prediction = Exp(score); // log link function
      //Incomplete Implementation
      const TFloat gradient = 1;
      const TFloat hessian = 1;
      return MakeGradientHessian(gradient, hessian);
   }
};
