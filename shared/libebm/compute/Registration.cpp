// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include <stddef.h> // size_t, ptrdiff_t
#include <vector>
#include <algorithm>

#include "Registration.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

void Registration::CheckParamNames(const char * const sParamName, std::vector<const char *> usedParamNames) {
   EBM_ASSERT(nullptr != sParamName);

   // yes, this is exponential, but it's only exponential for parameters that we define in this executable so
   // we have complete control, and objective/metric params should not exceed a handfull
   for(const char * const sOtherParamName : usedParamNames) {
      EBM_ASSERT(nullptr != sOtherParamName);

      const char * const sParamNameEnd = IsStringEqualsCaseInsensitive(sParamName, sOtherParamName);
      if(nullptr != sParamNameEnd) {
         if('\0' == *sParamNameEnd) {
            throw DuplicateParamNameException();
         }
      }
   }
   usedParamNames.push_back(sParamName);
}

bool Registration::CreateRegistrable(
   const Config * const pConfig,
   const char * sRegistration,
   const char * sRegistrationEnd,
   void * const pWrapperOut,
   const std::vector<std::shared_ptr<const Registration>> & registrations
) {
   EBM_ASSERT(nullptr != pConfig);
   EBM_ASSERT(nullptr != sRegistration);
   EBM_ASSERT(nullptr != sRegistrationEnd);
   EBM_ASSERT(sRegistration < sRegistrationEnd); // empty string not allowed
   EBM_ASSERT('\0' != *sRegistration);
   EBM_ASSERT(!(0x20 == *sRegistration || (0x9 <= *sRegistration && *sRegistration <= 0xd)));
   EBM_ASSERT('\0' == *sRegistrationEnd || k_registrationSeparator == *sRegistrationEnd);
   EBM_ASSERT(nullptr != pWrapperOut);

   LOG_0(Trace_Info, "Entered Registrable::CreateRegistrable");

   bool bNoMatch = true;
   for(const std::shared_ptr<const Registration> & registration : registrations) {
      if(nullptr != registration) {
         bNoMatch = registration->AttemptCreate(pConfig, sRegistration, sRegistrationEnd, pWrapperOut);
         if(!bNoMatch) {
            break;
         }
      }
   }

   LOG_0(Trace_Info, "Exited Registrable::CreateRegistrable");
   return bNoMatch;
}

void Registration::FinalCheckParams(
   const char * sRegistration,
   const char * const sRegistrationEnd,
   const size_t cUsedParams
) {
   if(cUsedParams != CountParams(sRegistration, sRegistrationEnd)) {
      // our counts don't match up, so there are strings in the sRegistration string that we didn't
      // process as params.
      throw ParamUnknownException();
   }
}

} // DEFINED_ZONE_NAME
