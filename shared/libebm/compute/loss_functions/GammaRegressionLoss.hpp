// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!

// DO NOT INCLUDE FILES IN THIS FILE. They will not be zoned properly. Include files go into the operator *.cpp files.

// TFloat is a datatype that could hold inside a double, float, or some SIMD intrinsic type.
// See sse2_32.cpp, cuda_32.cpp, and cpu_64.cpp as examples where TFloat operators are defined.
template<typename TFloat>
struct GammaRegressionLoss : RegressionLoss {
   LOSS_BOILERPLATE(GammaRegressionLoss, Link_log)

   // The constructor parameters following config must match the RegisterLoss parameters in loss_registrations.hpp
   inline GammaRegressionLoss(const Config & config) {
      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
      }
   }

   inline double GradientMultiple() const noexcept {
      return 1.0;
   }

   inline double HessianMultiple() const noexcept {
      return 1.0;
   }

   GPU_DEVICE inline TFloat InverseLinkFunction(const TFloat score) const noexcept {
      // Gamma regression uses a log link function
      return score.Exp();
   }

   GPU_DEVICE inline TFloat CalcMetric(const TFloat prediction, const TFloat target) const noexcept {
      TFloat metric =  2 * ((target - prediction)/prediction - (target / prediction).Log());
      return metric;
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat prediction, const TFloat target) const noexcept {
      TFloat gradient = 1.0 - (target / prediction);
      return gradient;
   }

   // If the loss function doesn't have a second derivative, then delete the CalcGradientHessian function.
   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat prediction, const TFloat target) const noexcept {
      TFloat gradient = 1.0 - (target / prediction);
      TFloat hessian = (target / prediction);
      return MakeGradientHessian(gradient, hessian);
   }
};
