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

void TermInnerBag::FreeTermInnerBag(TermInnerBag* const pTermInnerBag) {
   LOG_0(Trace_Info, "Entered TermInnerBag::FreeTermInnerBag");

   EBM_ASSERT(nullptr != pTermInnerBag);

   AlignedFree(pTermInnerBag->m_aCounts);
   AlignedFree(pTermInnerBag->m_aWeights);

   LOG_0(Trace_Info, "Exited TermInnerBag::FreeTermInnerBags");
}

} // namespace DEFINED_ZONE_NAME
