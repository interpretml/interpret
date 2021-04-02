// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !!! NOTE: To add a new loss/objective function in C++, follow the steps listed at the top of the "Loss.cpp" file !!!

#include "Loss.hpp"

// TFloat could be double, float, or some SIMD intrinsic type
template<typename TFloat>
struct LossMulticlassLogLoss : public LossMulticlass {
   LOSS_CLASS_BOILERPLATE(LossMulticlassLogLoss, true, 1)

   size_t m_countTargetClasses;

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

      m_countTargetClasses = config.cOutputs;
   }
};
