// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!

// DO NOT INCLUDE ANY FILES IN THIS FILE. THEY WILL NOT BE ZONED PROPERLY

// TFloat could be double, float, or some SIMD intrinsic type
template<typename TFloat>
struct PseudoHuberRegressionLoss final : public RegressionLoss {
   LOSS_CLASS_BOILERPLATE(PseudoHuberRegressionLoss, true)

   TFloat m_deltaInverted;

   // IMPORTANT: the constructor parameters here must match the RegisterLoss parameters in loss_registrations.hpp
   inline PseudoHuberRegressionLoss(const Config & config, TFloat delta) {
      if(1 != config.cOutputs) {
         throw ParamMismatchWithConfigException();
      }

      if(delta.IsAnyEqual(TFloat(0)) || delta.IsAnyNaN() || delta.IsAnyInf()) {
         throw ParamValOutOfRangeException();
      }

      TFloat deltaInverted = TFloat(1) / delta;
      if(deltaInverted.IsAnyInf()) {
         throw ParamValOutOfRangeException();
      }

      m_deltaInverted = deltaInverted;
   }

   inline double GetFinalMultiplier() const noexcept {
      return 1.0;
   }

   GPU_DEVICE inline TFloat InverseLinkFunction(TFloat score) const noexcept {
      return score;
   }

   GPU_DEVICE inline TFloat CalculateGradient(TFloat target, TFloat prediction) const noexcept {
      TFloat residualNegative = prediction - target;
      TFloat residualNegativeFraction = residualNegative * m_deltaInverted;
      TFloat calc = TFloat(1) + residualNegativeFraction * residualNegativeFraction;
      TFloat sqrtCalc = calc.Sqrt();
      // the calculations above are shared with the hessian, so the compiler should combine them.
      return residualNegative / sqrtCalc;
   }

   // if the loss function doesn't have a second derivative, then delete the CalculateHessian function.
   GPU_DEVICE inline TFloat CalculateHessian(TFloat target, TFloat prediction) const noexcept {
      TFloat residualNegative = prediction - target;
      TFloat residualNegativeFraction = residualNegative * m_deltaInverted;
      TFloat calc = TFloat(1) + residualNegativeFraction * residualNegativeFraction;
      TFloat sqrtCalc = calc.Sqrt();
      // the calculations above are shared with the hessian, so the compiler should combine them.
      return TFloat(1) / (calc * sqrtCalc);
   }
};
