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

ParamBase::ParamBase(const char * const sParamName) : 
   m_sParamName(sParamName) 
{
   if(EBM_FALSE != CheckForIllegalCharacters(sParamName)) {
      throw IllegalParamNameException();
   }
}

Registration::Registration(const bool bCpuOnly, const char * const sRegistrationName) :
   m_sRegistrationName(sRegistrationName),
   m_bCpuOnly(bCpuOnly)
{
   if(EBM_FALSE != CheckForIllegalCharacters(sRegistrationName)) {
      throw IllegalRegistrationNameException();
   }
}

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
   EBM_ASSERT(nullptr != sRegistration);
   EBM_ASSERT(nullptr != sRegistrationEnd);
   EBM_ASSERT(sRegistration <= sRegistrationEnd); // sRegistration contains the part after the tag now
   EBM_ASSERT(!(0x20 == *sRegistration || (0x9 <= *sRegistration && *sRegistration <= 0xd)));
   EBM_ASSERT('\0' == *sRegistrationEnd || k_registrationSeparator == *sRegistrationEnd);

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
      throw ParamUnknownException();
   }
}

} // DEFINED_ZONE_NAME
