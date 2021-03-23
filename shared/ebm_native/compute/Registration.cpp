// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t
#include <vector>
#include <algorithm>

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "Config.hpp"
#include "Registrable.hpp"
#include "Registration.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

static bool CheckForIllegalCharacters(const char * s) noexcept {
   if(nullptr != s) {
      // to be generously safe towards people adding new loss/metric registrations, check for nullptr
      while(true) {
         const char chr = *s;
         if('\0' == chr) {
            return false;
         }
         if(0x20 == chr || (0x9 <= chr && chr <= 0xd)) {
            // whitespace is illegal
            break;
         }
         if(Registration::k_registrationSeparator == chr ||
            Registration::k_paramSeparator == chr ||
            Registration::k_valueSeparator == chr ||
            Registration::k_typeTerminator == chr
         ) {
            break;
         }
         ++s;
      }
   }
   return true;
}

ParamBase::ParamBase(const char * const sParamName) : 
   m_sParamName(sParamName) 
{
   if(CheckForIllegalCharacters(sParamName)) {
      throw IllegalParamNameException();
   }
}

Registration::Registration(const char * const sRegistrationName) : 
   m_sRegistrationName(sRegistrationName) 
{
   if(CheckForIllegalCharacters(sRegistrationName)) {
      throw IllegalRegistrationNameException();
   }
}

void Registration::CheckParamNames(const char * const sParamName, std::vector<const char *> usedParamNames) {
   EBM_ASSERT(nullptr != sParamName);

   // yes, this is exponential, but it's only exponential for parameters that we define in this executable so
   // we have complete control, and loss params should not exceed a handfull
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

std::vector<std::unique_ptr<const Registrable>> Registration::CreateRegistrables(
   const Config & config,
   const char * sRegistration,
   const std::vector<std::shared_ptr<const Registration>> & registrations
) {
   EBM_ASSERT(nullptr != sRegistration);

   LOG_0(TraceLevelInfo, "Entered Registrable::CreateRegistrables");

   std::vector<std::unique_ptr<const Registrable>> registrables;
   while(true) {
      sRegistration = SkipWhitespace(sRegistration);
      const char * sRegistrationEnd = strchr(sRegistration, k_registrationSeparator);
      if(nullptr == sRegistrationEnd) {
         // find the null terminator then
         sRegistrationEnd = sRegistration + strlen(sRegistration);
      }
      if(sRegistrationEnd != sRegistration) {
         // we allow empty registrations like ",,,something_legal,,,  something_else  , " since the intent is clear
         for(const std::shared_ptr<const Registration> & registration : registrations) {
            if(nullptr != registration) {
               // normally we shouldn't have nullptr registrations, but let's not complain if someone is writing
               // their own custom one and accidentally puts one in.  We still understand the intent.
               std::unique_ptr<const Registrable> pRegistrable =
                  registration->AttemptCreate(config, sRegistration, sRegistrationEnd);
               if(nullptr != pRegistrable) {
                  registrables.emplace_back(pRegistrable.release());
                  goto next_registration; // pick only the first one per registrable in the sRegistration string
               }
            }
         }
         // we didn't find anything!
         throw RegistrationUnknownException();
      }
   next_registration:;
      if('\0' == *sRegistrationEnd) {
         break;
      }
      EBM_ASSERT(k_registrationSeparator == *sRegistrationEnd);

      sRegistration = sRegistrationEnd + 1;
   }
   LOG_0(TraceLevelInfo, "Exited Registrable::CreateRegistrables");
   return registrables;
}

void Registration::FinalCheckParameters(
   const char * sRegistration,
   const char * const sRegistrationEnd,
   const size_t cUsedParams
) {
   EBM_ASSERT(nullptr != sRegistration);
   EBM_ASSERT(nullptr != sRegistrationEnd);
   EBM_ASSERT(sRegistration <= sRegistrationEnd);

   // cUsedParams will have been filled by the time we reach this point since all the calls to UnpackParam
   // are guaranteed to have occured before we get called.

   size_t cRemainingParams = cUsedParams;
   while(true) {
      // first let's find what we would consider as the next valid param
      while(true) {
         sRegistration = SkipWhitespace(sRegistration);
         EBM_ASSERT(sRegistration <= sRegistrationEnd);
         if(k_paramSeparator != *sRegistration) {
            break;
         }
         ++sRegistration; // get past the ';' character
      }
      EBM_ASSERT(sRegistration <= sRegistrationEnd);
      if(sRegistrationEnd == sRegistration) {
         break;
      }
      --cRemainingParams; // this will underflow if we're missing a param, but underflow for unsigned is legal

      sRegistration = strchr(sRegistration, k_paramSeparator);
      if(nullptr == sRegistration || sRegistrationEnd <= sRegistration) {
         break;
      }
      ++sRegistration; // skip past the ';' character
   }
   if(size_t { 0 } != cRemainingParams) {
      // our counts don't match up, so there are strings in the sRegistration string that we didn't
      // process as params.  cRemainingParams should be a very big number since we would have underflowed
      throw ParameterUnknownException();
   }
}

const char * Registration::CheckRegistrationName(
   const char * sRegistration, 
   const char * const sRegistrationEnd
) const {
   EBM_ASSERT(nullptr != sRegistration);
   EBM_ASSERT(nullptr != sRegistrationEnd);
   EBM_ASSERT(sRegistration <= sRegistrationEnd);

   sRegistration = IsStringEqualsCaseInsensitive(sRegistration, m_sRegistrationName);
   if(nullptr == sRegistration) {
      // we are not the specified registration function
      return nullptr;
   }
   if(sRegistrationEnd != sRegistration) {
      if(k_typeTerminator != *sRegistration) {
         // we are not the specified objective, but the objective could still be something with a longer string
         // eg: the given tag was "something_else:" but our tag was "something:", so we matched on "something" only
         return nullptr;
      }
      sRegistration = SkipWhitespace(sRegistration + 1);
   }
   EBM_ASSERT(sRegistration <= sRegistrationEnd);
   return sRegistration;
}

} // DEFINED_ZONE_NAME
