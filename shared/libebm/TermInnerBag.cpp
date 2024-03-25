// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "logging.h" // EBM_ASSERT
#include "unzoned.h" // AlignedFree

#define ZONE_main
#include "zones.h"

#include "ebm_internal.hpp"
#include "Term.hpp"
#include "TermInnerBag.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

TermInnerBag** TermInnerBag::AllocateTermInnerBags(const size_t cTerms) {
   LOG_0(Trace_Info, "Entered TermInnerBag::AllocateTermInnerBags");

   if(IsMultiplyError(sizeof(void*), cTerms)) {
      LOG_0(Trace_Warning, "WARNING TermInnerBag::AllocateTermInnerBags IsMultiplyError(sizeof(void *), cTerms)");
      return nullptr;
   }
   TermInnerBag** aaTermInnerBag = static_cast<TermInnerBag**>(malloc(sizeof(void*) * cTerms));
   if(nullptr == aaTermInnerBag) {
      LOG_0(Trace_Warning, "WARNING TermInnerBag::AllocateTermInnerBags nullptr == aaTermInnerBag");
      return nullptr;
   }

   TermInnerBag** paTermInnerBag = aaTermInnerBag;
   const TermInnerBag* const* const paTermInnerBagEnd = &aaTermInnerBag[cTerms];
   do {
      *paTermInnerBag = nullptr;
      ++paTermInnerBag;
   } while(paTermInnerBagEnd != paTermInnerBag);

   LOG_0(Trace_Info, "Exited TermInnerBag::AllocateTermInnerBags");
   return aaTermInnerBag;
}

ErrorEbm TermInnerBag::InitTermInnerBags(const size_t cTerms,
      const Term* const* const apTerms,
      TermInnerBag** const aaTermInnerBags,
      const size_t cInnerBags) {
   LOG_0(Trace_Info, "Entered TermInnerBag::InitTermInnerBags");

   EBM_ASSERT(1 <= cTerms);
   EBM_ASSERT(nullptr != apTerms);
   EBM_ASSERT(nullptr != aaTermInnerBags);

   const size_t cInnerBagsAfterZero = size_t{0} == cInnerBags ? size_t{1} : cInnerBags;
   if(IsMultiplyError(sizeof(TermInnerBag), cInnerBagsAfterZero)) {
      LOG_0(Trace_Warning, "WARNING TermInnerBag::InitTermInnerBags IsMultiplyError(sizeof(TermInnerBag), cInnerBagsAfterZero)");
      return Error_OutOfMemory;
   }
   const size_t cTermInnerBagBytes = sizeof(TermInnerBag) * cInnerBagsAfterZero;

   const Term* const* ppTerm = apTerms; 

   TermInnerBag** paTermInnerBag = aaTermInnerBags;
   const TermInnerBag* const* const paTermInnerBagEnd = &aaTermInnerBags[cTerms];
   do {
      const Term* const pTerm = *ppTerm;
      ++ppTerm;
      const size_t cBins = pTerm->GetCountTensorBins();

      if(IsMultiplyError(sizeof(UIntMain), cBins)) {
         LOG_0(Trace_Warning, "WARNING TermInnerBag::InitTermInnerBags IsMultiplyError(sizeof(UIntMain), cBins)");
         return Error_OutOfMemory;
      }
      const size_t cBytesCounts = sizeof(UIntMain) * cBins;

      if(IsMultiplyError(sizeof(FloatPrecomp), cBins)) {
         LOG_0(Trace_Warning, "WARNING TermInnerBag::InitTermInnerBags IsMultiplyError(sizeof(FloatPrecomp), cBins)");
         return Error_OutOfMemory;
      }
      const size_t cBytesWeights = sizeof(FloatPrecomp) * cBins;


      TermInnerBag* const aTermInnerBag = static_cast<TermInnerBag*>(malloc(cTermInnerBagBytes));
      if(nullptr == aTermInnerBag) {
         LOG_0(Trace_Warning, "WARNING TermInnerBag::InitTermInnerBags nullptr == aTermInnerBag");
         return Error_OutOfMemory;
      }
      *paTermInnerBag = aTermInnerBag;

      TermInnerBag* pTermInnerBag = aTermInnerBag;
      const TermInnerBag* const pTermInnerBagEnd = IndexByte(aTermInnerBag, cTermInnerBagBytes);
      do {
         pTermInnerBag->collapsedCount = 0;
         pTermInnerBag->collapsedWeight = 0;
         pTermInnerBag->m_aCounts = nullptr;
         pTermInnerBag->m_aWeights = nullptr;
         ++pTermInnerBag;
      } while(pTermInnerBagEnd != pTermInnerBag);

      if(size_t{1} != cBins) {
         pTermInnerBag = aTermInnerBag;
         do {
            UIntMain* aBinCounts = static_cast<UIntMain*>(AlignedAlloc(cBytesCounts));
            if(nullptr == aBinCounts) {
               LOG_0(Trace_Warning, "WARNING TermInnerBag::InitTermInnerBags nullptr == aBinCounts");
               return Error_OutOfMemory;
            }
            pTermInnerBag->m_aCounts = aBinCounts;

            FloatPrecomp* aBinWeights = static_cast<FloatPrecomp*>(AlignedAlloc(cBytesWeights));
            if(nullptr == aBinWeights) {
               LOG_0(Trace_Warning, "WARNING TermInnerBag::InitTermInnerBags nullptr == aBinWeights");
               return Error_OutOfMemory;
            }
            pTermInnerBag->m_aWeights = aBinWeights;

            memset(aBinCounts, 0, cBytesCounts);
            memset(aBinWeights, 0, cBytesWeights);

            ++pTermInnerBag;
         } while(pTermInnerBagEnd != pTermInnerBag);
      }
      ++paTermInnerBag;
   } while(paTermInnerBagEnd != paTermInnerBag);

   LOG_0(Trace_Info, "Exited TermInnerBag::InitTermInnerBags");
   return Error_None;
}

// Visual Studio compiler seems to not like the index addition by 1 to make cInnerBagsAfterZero
WARNING_PUSH
WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
void TermInnerBag::FreeTermInnerBags(
      const size_t cTerms, TermInnerBag** const aaTermInnerBags, const size_t cInnerBags) {
   LOG_0(Trace_Info, "Entered TermInnerBag::FreeTermInnerBags");

   if(LIKELY(nullptr != aaTermInnerBags)) {
      EBM_ASSERT(1 <= cTerms);

      const size_t cInnerBagsAfterZero = size_t{0} == cInnerBags ? size_t{1} : cInnerBags;
      const size_t cTermInnerBagBytes = sizeof(TermInnerBag) * cInnerBagsAfterZero;

      TermInnerBag** paTermInnerBag = aaTermInnerBags;
      const TermInnerBag* const* const paTermInnerBagEnd = &aaTermInnerBags[cTerms];
      do {
         TermInnerBag* pTermInnerBag = *paTermInnerBag;
         if(nullptr != pTermInnerBag) {
            const TermInnerBag* const pTermInnerBagEnd = IndexByte(pTermInnerBag, cTermInnerBagBytes);
            do {
               AlignedFree(pTermInnerBag->m_aCounts);
               AlignedFree(pTermInnerBag->m_aWeights);
               ++pTermInnerBag;
            } while(pTermInnerBagEnd != pTermInnerBag);
            free(*paTermInnerBag);
         }
         ++paTermInnerBag;
      } while(paTermInnerBagEnd != paTermInnerBag);
      free(aaTermInnerBags);
   }
   LOG_0(Trace_Info, "Exited TermInnerBag::FreeTermInnerBags");
}
WARNING_POP

} // namespace DEFINED_ZONE_NAME
