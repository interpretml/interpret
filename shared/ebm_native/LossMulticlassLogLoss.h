// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !!! NOTE: To add a new loss/objective function in C++, follow the steps listed at the top of the "Loss.cpp" file !!!

#include "Loss.h"

struct LossMulticlassLogLoss : Loss {

   size_t m_countTargetClasses;

   // IMPORTANT: the constructor parameters here must match the RegisterLoss parameters in the file Loss.cpp
   INLINE_ALWAYS LossMulticlassLogLoss(const Config & config) {
      if(1 == config.GetCountOutputs()) {
         // we share the tag "log_loss" with binary classification
         throw SkipLossException();
      }

      if(config.GetCountOutputs() <= 0) {
         throw LossParameterMismatchWithConfigException();
      }

      m_countTargetClasses = config.GetCountOutputs();
   }

   LOSS_MULTI_DEFAULT_MECHANICS_PUT_AT_END_OF_CLASS
};
