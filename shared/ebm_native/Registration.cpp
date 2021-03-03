// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t
#include <vector>
#include <algorithm>

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureGroup.h"
#include "ThreadStateBoosting.h"

#include "Loss.h"
#include "Registration.h"

void Registration::FinalCheckParameters(const char * sRegistration, std::vector<const char *> & usedLocations) {
   std::sort(usedLocations.begin(), usedLocations.end());

   for(const char * sParam : usedLocations) {
      if(sParam != sRegistration) {
         throw EbmException(Error_LossParameterUnknown);
      }
      sRegistration = strchr(sRegistration, k_paramSeparator);
      if(nullptr == sRegistration) {
         return;
      }
      ++sRegistration;
      if(0 == *SkipWhitespace(sRegistration)) {
         return;
      }
   }
   if(0 != *SkipWhitespace(sRegistration)) {
      throw EbmException(Error_LossParameterUnknown);
   }
}

const char * Registration::CheckRegistrationName(const char * sRegistration) const {
   EBM_ASSERT(nullptr != sRegistration);

   sRegistration = IsStringEqualsCaseInsensitive(sRegistration, m_sRegistrationName);
   if(nullptr == sRegistration) {
      // we are not the specified registration function
      return nullptr;
   }
   if(0 != *sRegistration) {
      if(k_typeTerminator != *sRegistration) {
         // we are not the specified objective, but the objective could still be something with a longer string
         // eg: the given tag was "something_else:" but our tag was "something:", so we matched on "something" only
         return nullptr;
      }
      sRegistration = SkipWhitespace(sRegistration + 1);
   }
   return sRegistration;
}

