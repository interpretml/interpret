// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "logging.h" // EBM_ASSERT

#include "ebm_internal.hpp"
#include "InnerBag.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

void InnerBag::FreeInnerBags(const size_t cInnerBags, InnerBag * const aInnerBags) {
   LOG_0(Trace_Info, "Entered InnerBag::FreeInnerBags");
   if(LIKELY(nullptr != aInnerBags)) {
      const size_t cInnerBagsAfterZero = size_t { 0 } == cInnerBags ? size_t { 1 } : cInnerBags;
      InnerBag * pInnerBag = aInnerBags;
      const InnerBag * const pInnerBagsEnd = aInnerBags + cInnerBagsAfterZero;
      do {
         free(pInnerBag->m_aCountOccurrences);
         free(pInnerBag->m_aWeights);
         ++pInnerBag;
      } while(pInnerBagsEnd != pInnerBag);
      free(aInnerBags);
   }
   LOG_0(Trace_Info, "Exited InnerBag::FreeInnerBags");
}

InnerBag * InnerBag::AllocateInnerBags(const size_t cInnerBags) {
   LOG_0(Trace_Info, "Entered InnerBag::AllocateInnerBags");

   const size_t cInnerBagsAfterZero = size_t { 0 } == cInnerBags ? size_t { 1 } : cInnerBags;

   if(IsMultiplyError(sizeof(InnerBag), cInnerBagsAfterZero)) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateInnerBags IsMultiplyError(sizeof(InnerBag), cInnerBagsAfterZero)");
      return nullptr;
   }
   InnerBag * aInnerBag = static_cast<InnerBag *>(malloc(sizeof(InnerBag) * cInnerBagsAfterZero));
   if(UNLIKELY(nullptr == aInnerBag)) {
      LOG_0(Trace_Warning, "WARNING InnerBag::GenerateInnerBags nullptr == aInnerBag");
      return nullptr;
   }

   InnerBag * pInnerBag = aInnerBag;
   const InnerBag * const pInnerBagsEnd = &aInnerBag[cInnerBagsAfterZero];
   do {
      pInnerBag->m_aCountOccurrences = nullptr;
      pInnerBag->m_aWeights = nullptr;
      ++pInnerBag;
   } while(pInnerBagsEnd != pInnerBag);

   LOG_0(Trace_Info, "Exited InnerBag::AllocateInnerBags");
   return aInnerBag;
}

} // DEFINED_ZONE_NAME
