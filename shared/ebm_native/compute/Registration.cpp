// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <vector>
#include <algorithm>

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
         if(k_registrationSeparator == chr ||
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
   // we have complete control, and loss/metric params should not exceed a handfull
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
   EBM_ASSERT(!(0x20 == *(sRegistrationEnd - 1) || (0x9 <= *(sRegistrationEnd - 1) && *(sRegistrationEnd - 1) <= 0xd)));
   EBM_ASSERT('\0' == *sRegistrationEnd || k_registrationSeparator == *sRegistrationEnd || 0x20 == *sRegistrationEnd || (0x9 <= *sRegistrationEnd && *sRegistrationEnd <= 0xd));
   EBM_ASSERT(nullptr != pWrapperOut);

   LOG_0(Trace_Info, "Entered Registrable::CreateRegistrable");

   bool bNoMatch = true;
   for(const std::shared_ptr<const Registration> & registration : registrations) {
      if(nullptr != registration) {
         // normally we shouldn't have nullptr registrations, but let's not complain if someone is writing
         // their own custom one and accidentally puts one in.  We still understand the intent.
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
   EBM_ASSERT(nullptr != sRegistration);
   EBM_ASSERT(nullptr != sRegistrationEnd);
   EBM_ASSERT(sRegistration <= sRegistrationEnd); // sRegistration contains the part after the tag now
   EBM_ASSERT(!(0x20 == *sRegistration || (0x9 <= *sRegistration && *sRegistration <= 0xd)));
   EBM_ASSERT(!(0x20 == *(sRegistrationEnd - 1) || (0x9 <= *(sRegistrationEnd - 1) && *(sRegistrationEnd - 1) <= 0xd)));
   EBM_ASSERT('\0' == *sRegistrationEnd || k_registrationSeparator == *sRegistrationEnd || 0x20 == *sRegistrationEnd || (0x9 <= *sRegistrationEnd && *sRegistrationEnd <= 0xd));

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
      if(sRegistrationEnd <= sRegistration) {
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
      throw ParamUnknownException();
   }
}

const char * Registration::CheckRegistrationName(
   const char * sRegistration, 
   const char * const sRegistrationEnd
) const {
   EBM_ASSERT(nullptr != sRegistration);
   EBM_ASSERT(nullptr != sRegistrationEnd);
   EBM_ASSERT(sRegistration < sRegistrationEnd); // empty string not allowed
   EBM_ASSERT('\0' != *sRegistration);
   EBM_ASSERT(!(0x20 == *sRegistration || (0x9 <= *sRegistration && *sRegistration <= 0xd)));
   EBM_ASSERT(!(0x20 == *(sRegistrationEnd - 1) || (0x9 <= *(sRegistrationEnd - 1) && *(sRegistrationEnd - 1) <= 0xd)));
   EBM_ASSERT('\0' == *sRegistrationEnd || k_registrationSeparator == *sRegistrationEnd || 0x20 == *sRegistrationEnd || (0x9 <= *sRegistrationEnd && *sRegistrationEnd <= 0xd));

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
