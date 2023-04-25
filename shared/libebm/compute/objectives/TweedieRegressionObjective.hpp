// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// TFloat is a datatype that could hold inside a double, float, or some SIMD intrinsic type.
// See sse2_32.cpp, cuda_32.cpp, and cpu_64.cpp as examples where TFloat operators are defined.
template<typename TFloat>
struct TweedieRegressionObjective : RegressionObjective {
   OBJECTIVE_BOILERPLATE(TweedieRegressionObjective, Link_log)

   // The constructor parameters following config must match the RegisterObjective parameters in objective_registrations.hpp
   inline TweedieRegressionObjective(const Config & config) {
      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
      }
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

   GPU_DEVICE inline TFloat CalcMetric(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = Exp(score); // log link function
      //Incomplete Implementation
      //const TFloat metric = 1
      return 1;
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = Exp(score); // log link function
      //Incomplete Implementation
      //const TFloat gradient = 1
      return 1;
   }

   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat score, const TFloat target) const noexcept {
      const TFloat prediction = Exp(score); // log link function
      //Incomplete Implementation
      //const TFloat gradient = 1;
      //const TFloat hessian = 1;
      return MakeGradientHessian(1, 1);
   }
};
