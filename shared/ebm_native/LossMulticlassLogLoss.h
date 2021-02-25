// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// Steps for adding a new objective in C++:
//   1) Copy one of the existing Loss*.h include files (like this one) into a new renamed Loss*.h file
//   2) Modify the class below to handle your new Loss function
//   3) Add [#include "Loss*.h"] to the list of other include files near the top of the Loss.cpp file
//   4) Add [Loss*::AttemptCreateLoss] to the list of objectives in k_registeredLosss in Loss.cpp
//   5) Recompile the C++ with either build.sh or build.bat depending on your operating system
//   6) Enjoy your new Loss function, and send us a PR on Github if you think others would benefit  :-)

// IMPORTANT: This file should only be included ONCE in the project, and that place should be in the Loss.cpp file

#include <stddef.h>

#include "EbmInternal.h"
#include "Logging.h"
#include "FeatureGroup.h"
#include "ThreadStateBoosting.h"

#include "Loss.h"

class LossMulticlassSoftmax final : public Loss {

   size_t m_countTargetClasses;

   INLINE_ALWAYS LossMulticlassSoftmax(const size_t countTargetClasses) {
      m_countTargetClasses = countTargetClasses;
   }

public:

   static ErrorEbmType AttemptCreateLoss(
      const char * sLoss,
      size_t countTargetClasses,
      const Loss ** const ppLoss
   ) {
      EBM_ASSERT(nullptr != sLoss);
      EBM_ASSERT(nullptr != ppLoss);
      EBM_ASSERT(nullptr == *ppLoss);

      static const char k_sLossTag[] = "multiclass_softmax";
      sLoss = IsStringEqualsCaseInsensitive(sLoss, k_sLossTag);
      if(nullptr == sLoss) {
         // we are not the specified objective
         return Error_None;
      }
      if(0 != *sLoss) {
         // we are not the specified objective, but the objective could still be something with a longer string
         // eg: the given tag was "something_else:" but our tag was "something:", so we matched on "something" only
         return Error_None;
      }

      *ppLoss = new LossMulticlassSoftmax(countTargetClasses);
      return Error_None;
   }

   LOSS_MULTICLASS_DEFAULT_MECHANICS_PUT_AT_END_OF_CLASS

};
