// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h>

#include "libebm.h" // BoolEbm
#include "logging.h"
#include "common_c.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

INTERNAL_IMPORT_EXPORT_BODY const char * SkipWhitespace(const char * s) {
   char oneChar = *s;
   while(0x20 == oneChar || (0x9 <= oneChar && oneChar <= 0xd)) {
      // skip whitespace
      ++s;
      oneChar = *s;
   }
   return s;
}

INTERNAL_IMPORT_EXPORT_BODY const char * ConvertStringToFloat(const char * const s, double * const pResultOut) {
   // we skip beginning whitespaces (strtod guarantees this)
   // unlike strtod, we also skip trailing whitespaces

   EBM_ASSERT(NULL != s);
   EBM_ASSERT(NULL != pResultOut);

   // the C++ standard says this about strtod:
   //   If the subject sequence is empty or does not have the expected form, no
   //   conversion is performed; the value of nptr is stored in the object
   //   pointed to by endptr, provided that endptr is not a null pointer.
   //
   // But, I'm unwilling to trust that there is no variation in the C++ runtime libraries, so I'll do my best to 
   // trust but verify by setting sNext before calling strtod, even though that involves a const cast
   char * sNext = (char *)(s);
   const double ret = strtod(s, &sNext);
   if(s == sNext || NULL == sNext) {
      // technically, sNext should never be nullptr, but we're again verifying our trust of the C++ library
      return NULL;
   }
   *pResultOut = ret;
   return SkipWhitespace(sNext);
}

INTERNAL_IMPORT_EXPORT_BODY const char * IsStringEqualsCaseInsensitive(const char * sMain, const char * sLabel) {
   // this function returns nullptr if there is no match, otherwise it returns a pointer to the 
   // first non-whitespace character following a successfully equal comparison

   char mainChar = *sMain;
   EBM_ASSERT(0x20 != mainChar && (mainChar < 0x9 || 0xd < mainChar));
   char labelChar = *sLabel;
   while('\0' != labelChar) {
      if('A' <= mainChar && mainChar <= 'Z') {
         mainChar += 'a' - 'A';
      }
      if('A' <= labelChar && labelChar <= 'Z') {
         // in theory within our executable we could ensure that all labels are lower case, but we want
         // people to tweak the objective and metric registrations, so let's be defensive here and do a full
         // case insensitive compare
         labelChar += 'a' - 'A';
      }
      if(mainChar != labelChar) {
         return NULL;
      }
      ++sMain;
      ++sLabel;
      mainChar = *sMain;
      labelChar = *sLabel;
   }
   while(0x20 == mainChar || (0x9 <= mainChar && mainChar <= 0xd)) {
      // skip whitespace
      ++sMain;
      mainChar = *sMain;
   }
   return sMain;
}

INTERNAL_IMPORT_EXPORT_BODY BoolEbm IsStringEqualsForgiving(const char * sMain, const char * sLabel) {
   sMain = IsStringEqualsCaseInsensitive(sMain, sLabel);
   if(NULL == sMain || '\0' != *sMain) {
      return EBM_FALSE;
   }
   return EBM_TRUE;
}

INTERNAL_IMPORT_EXPORT_BODY BoolEbm CheckForIllegalCharacters(const char * s) {
   if(NULL != s) {
      // to be generously safe towards people adding new objective/metric registrations, check for nullptr
      while(EBM_TRUE) {
         const char chr = *s;
         if('\0' == chr) {
            return EBM_FALSE;
         }
         if(0x20 == chr || (0x9 <= chr && chr <= 0xd)) {
            // whitespace is illegal
            break;
         }
         if(k_registrationSeparator == chr ||
            k_paramSeparator == chr ||
            k_valueSeparator == chr ||
            k_typeTerminator == chr
         ) {
            break;
         }
         ++s;
      }
   }
   return EBM_TRUE;
}

INTERNAL_IMPORT_EXPORT_BODY const char * CheckRegistrationName(
   const char * sRegistration, 
   const char * const sRegistrationEnd,
   const char * const sRegistrationName
) {
   EBM_ASSERT(NULL != sRegistration);
   EBM_ASSERT(NULL != sRegistrationEnd);
   EBM_ASSERT(sRegistration < sRegistrationEnd); // empty string not allowed
   EBM_ASSERT('\0' != *sRegistration);
   EBM_ASSERT(!(0x20 == *sRegistration || (0x9 <= *sRegistration && *sRegistration <= 0xd)));
   EBM_ASSERT('\0' == *sRegistrationEnd || k_registrationSeparator == *sRegistrationEnd);

   sRegistration = IsStringEqualsCaseInsensitive(sRegistration, sRegistrationName);
   if(NULL == sRegistration) {
      // we are not the specified registration function
      return NULL;
   }
   EBM_ASSERT(sRegistration <= sRegistrationEnd);
   if(sRegistrationEnd != sRegistration) {
      if(k_typeTerminator != *sRegistration) {
         // we are not the specified objective, but the objective could still be something with a longer string
         // eg: the given tag was "something_else:" but our tag was "something:", so we matched on "something" only
         return NULL;
      }
      sRegistration = SkipWhitespace(sRegistration + 1);
   }
   return sRegistration;
}

INTERNAL_IMPORT_EXPORT_BODY size_t CountParams(
   const char * sRegistration,
   const char * const sRegistrationEnd
) {
   EBM_ASSERT(NULL != sRegistration);
   EBM_ASSERT(NULL != sRegistrationEnd);
   EBM_ASSERT(sRegistration <= sRegistrationEnd); // sRegistration contains the part after the tag now
   EBM_ASSERT(!(0x20 == *sRegistration || (0x9 <= *sRegistration && *sRegistration <= 0xd)));
   EBM_ASSERT('\0' == *sRegistrationEnd || k_registrationSeparator == *sRegistrationEnd);

   // cUsedParams will have been filled by the time we reach this point since all the calls to UnpackParam
   // are guaranteed to have occured before we get called.

   size_t cParams = 0;
   while(EBM_TRUE) {
      // first let's find what we would consider as the next valid param
      while(EBM_TRUE) {
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
      ++cParams;

      sRegistration = strchr(sRegistration, k_paramSeparator);
      if(NULL == sRegistration || sRegistrationEnd <= sRegistration) {
         break;
      }
      ++sRegistration; // skip past the ';' character
   }
   return cParams;
}

INTERNAL_IMPORT_EXPORT_BODY void * AlignedAlloc(const size_t cBytes) {
   EBM_ASSERT(0 != cBytes);
   if(SIZE_MAX - (sizeof(void *) + SIMD_BYTE_ALIGNMENT - 1) < cBytes) {
      return NULL;
   }
   const size_t cPaddedBytes = sizeof(void *) + SIMD_BYTE_ALIGNMENT - 1 + cBytes;
   void * const p = malloc(cPaddedBytes);
   if(NULL == p) {
      return NULL;
   }

   uintptr_t pointer = REINTERPRET_CAST(uintptr_t, p);
   pointer = (pointer + STATIC_CAST(uintptr_t, sizeof(void *) + SIMD_BYTE_ALIGNMENT - 1)) & 
      STATIC_CAST(uintptr_t, ~STATIC_CAST(uintptr_t, SIMD_BYTE_ALIGNMENT - 1));
   *(REINTERPRET_CAST(void **, pointer) - 1) = p;
   return REINTERPRET_CAST(void *, pointer);
}
INTERNAL_IMPORT_EXPORT_BODY void AlignedFree(void * const p) {
   if(NULL != p) {
      free(*(REINTERPRET_CAST(void **, p) - 1));
   }
}
INTERNAL_IMPORT_EXPORT_BODY void * AlignedRealloc(void * const p, const size_t cOldBytes, const size_t cNewBytes) {
   EBM_ASSERT(NULL != p);
   EBM_ASSERT(0 != cOldBytes);
   EBM_ASSERT(0 != cNewBytes);
   EBM_ASSERT(cOldBytes < cNewBytes);

   void * const pNew = AlignedAlloc(cNewBytes);
   if(pNew == NULL) {
      // identically to realloc, we do NOT free the old memory if there is not enough memory
      return NULL;
   }
   memcpy(pNew, p, cOldBytes); // NOLINT(clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling)

   AlignedFree(p);
   return pNew;
}

#ifdef __cplusplus
}
#endif // __cplusplus
