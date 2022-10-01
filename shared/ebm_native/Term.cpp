// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "common_c.h" // UNLIKELY

#include "Term.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

Term * Term::Allocate(const size_t cDimensions) noexcept {
   const size_t cBytes = GetTermCountBytes(cDimensions);
   EBM_ASSERT(1 <= cBytes);
   Term * const pTerm = static_cast<Term *>(malloc(cBytes));
   if(UNLIKELY(nullptr == pTerm)) {
      return nullptr;
   }
   pTerm->Initialize(cDimensions);
   return pTerm;
}

Term ** Term::AllocateTerms(const size_t cTerms) noexcept {
   LOG_0(Trace_Info, "Entered Term::AllocateTerms");

   if(IsMultiplyError(sizeof(Term *), cTerms)) {
      LOG_0(Trace_Warning, "WARNING Term::AllocateTerms IsMultiplyError(sizeof(Term *), cTerms)");
      return nullptr;
   }

   EBM_ASSERT(1 <= cTerms);
   Term ** const apTerms = static_cast<Term **>(malloc(sizeof(Term *) * cTerms));
   if(nullptr != apTerms) {
      Term ** ppTerm = apTerms;
      const Term * const * const ppTermsEnd = &apTerms[cTerms];
      do {
         *ppTerm = nullptr;
         ++ppTerm;
      } while(ppTermsEnd != ppTerm);
   }

   LOG_0(Trace_Info, "Exited Term::AllocateTerms");
   return apTerms;
}

void Term::FreeTerms(const size_t cTerms, Term ** apTerms) noexcept {
   LOG_0(Trace_Info, "Entered Term::FreeTerms");
   if(nullptr != apTerms) {
      EBM_ASSERT(0 < cTerms);
      for(size_t iTerm = 0; iTerm < cTerms; ++iTerm) {
         if(nullptr != apTerms[iTerm]) {
            Term::Free(apTerms[iTerm]);
         }
      }
      free(apTerms);
   }
   LOG_0(Trace_Info, "Exited Term::FreeTerms");
}

} // DEFINED_ZONE_NAME
