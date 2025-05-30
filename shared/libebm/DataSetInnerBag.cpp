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
#include "TermInnerBag.hpp"
#include "DataSetInnerBag.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

DataSetInnerBag* DataSetInnerBag::AllocateDataSetInnerBags(const size_t cInnerBags) {
   LOG_0(Trace_Info, "Entered DataSetInnerBag::AllocateDataSetInnerBags");

   const size_t cInnerBagsAfterZero = size_t{0} == cInnerBags ? size_t{1} : cInnerBags;

   if(IsMultiplyError(sizeof(DataSetInnerBag), cInnerBagsAfterZero)) {
      LOG_0(Trace_Warning,
            "WARNING DataSetInnerBag::AllocateDataSetInnerBags IsMultiplyError(sizeof(DataSetInnerBag), "
            "cInnerBagsAfterZero)");
      return nullptr;
   }
   DataSetInnerBag* aDataSetInnerBag =
         static_cast<DataSetInnerBag*>(malloc(sizeof(DataSetInnerBag) * cInnerBagsAfterZero));
   if(UNLIKELY(nullptr == aDataSetInnerBag)) {
      LOG_0(Trace_Warning, "WARNING DataSetInnerBag::AllocateDataSetInnerBags nullptr == aDataSetInnerBag");
      return nullptr;
   }

   DataSetInnerBag* pDataSetInnerBag = aDataSetInnerBag;
   const DataSetInnerBag* const pDataSetInnerBagsEnd = &aDataSetInnerBag[cInnerBagsAfterZero];
   do {
      pDataSetInnerBag->m_totalCount = 0;
      pDataSetInnerBag->m_totalWeight = 0;
      pDataSetInnerBag->m_aTermInnerBags = nullptr;
      ++pDataSetInnerBag;
   } while(pDataSetInnerBagsEnd != pDataSetInnerBag);

   LOG_0(Trace_Info, "Exited DataSetInnerBag::AllocateDataSetInnerBags");
   return aDataSetInnerBag;
}

// Visual Studio compiler seems to not like the index addition by 1 to make cInnerBagsAfterZero
WARNING_PUSH
WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
void DataSetInnerBag::FreeDataSetInnerBags(
      const size_t cInnerBags, DataSetInnerBag* const aDataSetInnerBags, const size_t cTerms) {
   LOG_0(Trace_Info, "Entered DataSetInnerBag::FreeDataSetInnerBags");

   if(LIKELY(nullptr != aDataSetInnerBags)) {
      const size_t cInnerBagsAfterZero = size_t{0} == cInnerBags ? size_t{1} : cInnerBags;
      DataSetInnerBag* pDataSetInnerBag = aDataSetInnerBags;
      const DataSetInnerBag* const pDataSetInnerBagsEnd = aDataSetInnerBags + cInnerBagsAfterZero;
      do {
         TermInnerBag* const aTermInnerBags = pDataSetInnerBag->m_aTermInnerBags;
         if(nullptr != aTermInnerBags) {
            EBM_ASSERT(1 <= cTerms);
            TermInnerBag* pTermInnerBag = aTermInnerBags;
            const TermInnerBag* const pTermInnerBagEnd = pTermInnerBag + cTerms;
            do {
               TermInnerBag::FreeTermInnerBag(pTermInnerBag);
               ++pTermInnerBag;
            } while(pTermInnerBagEnd != pTermInnerBag);
            free(aTermInnerBags);
         }
         ++pDataSetInnerBag;
      } while(pDataSetInnerBagsEnd != pDataSetInnerBag);
      free(aDataSetInnerBags);
   }

   LOG_0(Trace_Info, "Exited DataSetInnerBag::FreeDataSetInnerBags");
}
WARNING_POP

} // namespace DEFINED_ZONE_NAME
