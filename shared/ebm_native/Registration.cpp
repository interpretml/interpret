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

#include "Config.h"
#include "Registrable.h"
#include "Registration.h"

std::unique_ptr<const Registrable> Registration::CreateRegistrable(
   const std::vector<std::shared_ptr<const Registration>> registrations,
   const char * const sRegistration,
   const Config * const pConfig
) {
   EBM_ASSERT(nullptr != sRegistration);
   EBM_ASSERT(nullptr != pConfig);

   LOG_0(TraceLevelInfo, "Entered Registrable::CreateRegistrable");
   for(const std::shared_ptr<const Registration> & registration : registrations) {
      if(nullptr != registration) {
         try {
            std::unique_ptr<const Registrable> pRegistrable = registration->AttemptCreate(*pConfig, sRegistration);
            if(nullptr != pRegistrable) {
               // found it!
               LOG_0(TraceLevelInfo, "Exited Registrable::CreateRegistrable");
               // we're exiting the area where exceptions are regularily thrown 
               return pRegistrable;
            }
         } catch(const SkipRegistrationException &) {
            // the specific Registrable function is saying it isn't a match (based on parameters in the Config object probably)
         }
      }
   }
   LOG_0(TraceLevelWarning, "WARNING Registrable::CreateRegistrable registration unknown");
   return nullptr;
}

void Registration::FinalCheckParameters(const char * sRegistration, std::vector<const char *> & usedLocations) {
   std::sort(usedLocations.begin(), usedLocations.end());

   for(const char * sParam : usedLocations) {
      if(sParam != sRegistration) {
         throw ParameterUnknownException();
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
      throw ParameterUnknownException();
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

