// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!

#include "Loss.hpp"

// TFloat could be double, float, or some SIMD intrinsic type
template<typename TFloat>
struct LossMulticlassLogLoss : public LossMulticlass {
   LOSS_CLASS_BOILERPLATE(LossMulticlassLogLoss, true, 1)

   // IMPORTANT: the constructor parameters here must match the RegisterLoss parameters in the file Loss.cpp
   INLINE_ALWAYS LossMulticlassLogLoss(const Config & config) {
      UNUSED(config);

      if(1 == config.cOutputs) {
         // we share the tag "log_loss" with binary classification
         throw SkipRegistrationException();
      }

      if(config.cOutputs <= 0) {
         throw ParameterMismatchWithConfigException();
      }
   }

   GPU_DEVICE INLINE_ALWAYS TFloat InverseLinkFunctionPass1(size_t countTargetClasses, TFloat * pointerScores, TFloat * pointerTempStorage, const TFloat & tempValue) const {
      //TODO implement
      // use the countTargetClasses since it can be a templated constant, unlike m_countTargetClasses
      return 999;
   }

   GPU_DEVICE INLINE_ALWAYS TFloat InverseLinkFunctionPass2(size_t countTargetClasses, TFloat * pointerScores, TFloat * pointerTempStorage, const TFloat & tempValue) const {
      //TODO implement
      // use the countTargetClasses since it can be a templated constant, unlike m_countTargetClasses
      return 999;
   }

   //TODO USE THIS FORMAT FOR MULTICLASS: GPU_DEVICE INLINE_ALWAYS TFloat CalculateGradient(TFloat target, TFloat prediction, TFloat tempStorage, TFloat tempValue) const {
   GPU_DEVICE INLINE_ALWAYS TFloat CalculateGradient(TFloat target, TFloat prediction) const {
      //TODO implement
      return 999.9999;
   }

   // if the loss function doesn't have a second derivative, then delete the CalculateHessian function.
   GPU_DEVICE INLINE_ALWAYS TFloat CalculateHessian(TFloat target, TFloat prediction, TFloat tempStorage, TFloat tempValue) const {
      //TODO implement
      return 999.9999;
   }
};
