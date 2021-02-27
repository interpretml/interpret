// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// Steps for adding a new objective in C++:
//   1) Copy one of the existing Loss*.h include files (like this one) into a new renamed Loss*.h file
//   2) Modify the class below to handle your new Loss function
//   3) Add [#include "Loss*.h"] to the list of other include files near the top of the Loss.cpp file
//   4) Add the Loss* type to the list of loss registrations in RegisterLosses() near the top of the Loss.cpp file
//   5) Recompile the C++ with either build.sh or build.bat depending on your operating system
//   6) Enjoy your new Loss function, and send us a PR on Github if you think others would benefit  :-)

#include "Loss.h"

struct LossMulticlassLogLoss final : public Loss {

   size_t m_countTargetClasses;

   // IMPORTANT: the constructor parameters here must match the LossRegistration parameters in the file Loss.cpp
   INLINE_ALWAYS LossMulticlassLogLoss(const Config & config) {
      if(2 == config.GetCountOutputs()) {
         // we share the tag "log_loss" with binary classification
         throw SkipLossException();
      }

      if(config.GetCountOutputs() <= 1) {
         throw LossParameterMismatchWithConfigException();
      }

      m_countTargetClasses = config.GetCountOutputs();
   }

   LOSS_MULTI_DEFAULT_MECHANICS_PUT_AT_END_OF_CLASS
};
