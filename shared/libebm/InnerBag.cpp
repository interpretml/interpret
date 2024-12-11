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
#include "InnerBag.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

SubsetInnerBag* SubsetInnerBag::AllocateSubsetInnerBags(const size_t cInnerBags) {
   LOG_0(Trace_Info, "Entered SubsetInnerBag::AllocateSubsetInnerBags");

   const size_t cInnerBagsAfterZero = size_t{0} == cInnerBags ? size_t{1} : cInnerBags;

   if(IsMultiplyError(sizeof(SubsetInnerBag), cInnerBagsAfterZero)) {
      LOG_0(Trace_Warning,
            "WARNING SubsetInnerBag::AllocateSubsetInnerBags IsMultiplyError(sizeof(SubsetInnerBag), "
            "cInnerBagsAfterZero)");
      return nullptr;
   }
   SubsetInnerBag* aSubsetInnerBag = static_cast<SubsetInnerBag*>(malloc(sizeof(SubsetInnerBag) * cInnerBagsAfterZero));
   if(UNLIKELY(nullptr == aSubsetInnerBag)) {
      LOG_0(Trace_Warning, "WARNING SubsetInnerBag::AllocateSubsetInnerBags nullptr == aSubsetInnerBag");
      return nullptr;
   }

   SubsetInnerBag* pSubsetInnerBag = aSubsetInnerBag;
   const SubsetInnerBag* const pSubsetInnerBagsEnd = &aSubsetInnerBag[cInnerBagsAfterZero];
   do {
      pSubsetInnerBag->m_aWeights = nullptr;
      ++pSubsetInnerBag;
   } while(pSubsetInnerBagsEnd != pSubsetInnerBag);

   LOG_0(Trace_Info, "Exited SubsetInnerBag::AllocateSubsetInnerBags");
   return aSubsetInnerBag;
}

// Visual Studio compiler seems to not like the index addition by 1 to make cInnerBagsAfterZero
WARNING_PUSH
WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
void SubsetInnerBag::FreeSubsetInnerBags(const size_t cInnerBags, SubsetInnerBag* const aSubsetInnerBags) {
   LOG_0(Trace_Info, "Entered SubsetInnerBag::FreeSubsetInnerBags");

   if(LIKELY(nullptr != aSubsetInnerBags)) {
      const size_t cInnerBagsAfterZero = size_t{0} == cInnerBags ? size_t{1} : cInnerBags;
      SubsetInnerBag* pSubsetInnerBag = aSubsetInnerBags;
      const SubsetInnerBag* const pSubsetInnerBagsEnd = aSubsetInnerBags + cInnerBagsAfterZero;
      do {
         AlignedFree(pSubsetInnerBag->m_aWeights);
         ++pSubsetInnerBag;
      } while(pSubsetInnerBagsEnd != pSubsetInnerBag);
      free(aSubsetInnerBags);
   }

   LOG_0(Trace_Info, "Exited SubsetInnerBag::FreeSubsetInnerBags");
}
WARNING_POP

} // namespace DEFINED_ZONE_NAME
