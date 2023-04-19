// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!

// DO NOT INCLUDE FILES IN THIS FILE. They will not be zoned properly. Include files go into the operator *.cpp files.

// TFloat is a datatype that could hold inside a double, float, or some SIMD intrinsic type.
// See sse2_32.cpp, cuda_32.cpp, and cpu_64.cpp as examples where TFloat operators are defined.
template<typename TFloat>
struct PseudoHuberRegressionLoss : RegressionLoss {
   LOSS_BOILERPLATE(PseudoHuberRegressionLoss, Link_identity)

   TFloat m_deltaInverted;
   TFloat m_deltaSquared;

   // The constructor parameters following config must match the RegisterLoss parameters in loss_registrations.hpp
   inline PseudoHuberRegressionLoss(const Config & config, const double delta) {
      if(config.cOutputs != 1) {
         throw ParamMismatchWithConfigException();
      }

      if(delta == 0.0 || std::isnan(delta) || std::isinf(delta)) {
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

   inline double GradientMultiple() const noexcept {
      return 1.0;
   }

   inline double HessianMultiple() const noexcept {
      return 1.0;
   }

   GPU_DEVICE inline TFloat InverseLinkFunction(const TFloat score) const noexcept {
      // pseudo_huber uses an identity link function
      return score;
   }

   GPU_DEVICE inline TFloat CalcMetric(const TFloat prediction, const TFloat target) const noexcept {
      TFloat error = prediction - target;
      TFloat errorFraction = error * m_deltaInverted;
      TFloat calc = errorFraction * errorFraction + 1.0;
      TFloat sqrtCalc = Sqrt(calc);
      TFloat metric = m_deltaSquared * (sqrtCalc - 1.0);
      return metric;
   }

   GPU_DEVICE inline TFloat CalcGradient(const TFloat prediction, const TFloat target) const noexcept {
      TFloat error = prediction - target;
      TFloat errorFraction = error * m_deltaInverted;
      TFloat calc = errorFraction * errorFraction + 1.0;
      TFloat sqrtCalc = Sqrt(calc);
      TFloat gradient = error / sqrtCalc;
      return gradient;
   }

   // If the loss function doesn't have a second derivative, then delete the CalcGradientHessian function.
   GPU_DEVICE inline GradientHessian<TFloat> CalcGradientHessian(const TFloat prediction, const TFloat target) const noexcept {
      TFloat error = prediction - target;
      TFloat errorFraction = error * m_deltaInverted;
      TFloat calc = errorFraction * errorFraction + 1.0;
      TFloat sqrtCalc = Sqrt(calc);
      TFloat gradient = error / sqrtCalc;
      TFloat hessian = 1.0 / (calc * sqrtCalc);
      return MakeGradientHessian(gradient, hessian);
   }
};
