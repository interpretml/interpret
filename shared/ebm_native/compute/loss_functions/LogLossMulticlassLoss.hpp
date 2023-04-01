// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!

#include "Loss.hpp"

// TFloat could be double, float, or some SIMD intrinsic type
template<typename TFloat>
struct LogLossMulticlassLoss final : public MulticlassLoss {
   LOSS_CLASS_BOILERPLATE(LogLossMulticlassLoss, true)

   // IMPORTANT: the constructor parameters here must match the RegisterLoss parameters in loss_registrations.hpp
   inline LogLossMulticlassLoss(const Config & config) {
      if(1 == config.cOutputs) {
         // we share the tag "log_loss" with binary classification
         throw SkipRegistrationException();
      }

      if(config.cOutputs <= 0) {
         throw ParamMismatchWithConfigException();
      }
   }

   inline double GetFinalMultiplier() const noexcept {
      return 1.0;
   }

   GPU_DEVICE inline TFloat InverseLinkFunctionPass1(size_t countClasses, TFloat * pointerScores, TFloat * pointerTempStorage, const TFloat & tempVal) const noexcept {
      //TODO implement
      // use the countClasses since it can be a templated constant, unlike m_countClasses
      return 999;
   }

   GPU_DEVICE inline TFloat InverseLinkFunctionPass2(size_t countClasses, TFloat * pointerScores, TFloat * pointerTempStorage, const TFloat & tempVal) const noexcept {
      //TODO implement
      // use the countClasses since it can be a templated constant, unlike m_countClasses
      return 999;
   }

   //TODO USE THIS FORMAT FOR MULTICLASS: GPU_DEVICE INLINE_ALWAYS TFloat CalculateGradient(TFloat target, TFloat prediction, TFloat tempStorage, TFloat tempVal) const {
   GPU_DEVICE inline TFloat CalculateGradient(TFloat target, TFloat prediction) const noexcept {
      //TODO implement
      return 999.9999;
   }

   // if the loss function doesn't have a second derivative, then delete the CalculateHessian function.
   GPU_DEVICE inline TFloat CalculateHessian(TFloat target, TFloat prediction, TFloat tempStorage, TFloat tempVal) const noexcept {
      //TODO implement
      return 999.9999;
   }
};
