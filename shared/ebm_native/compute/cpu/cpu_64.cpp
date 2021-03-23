// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include "ebm_native.h"
#include "bridge_c.h"
#include "common_c.h"
#include "zones.h"

// we use DEFINED_ZONE_NAME in order to give the contents below separate names to the compiler and
// avoid very very very bad "one definition rule" violations which are nasty undefined behavior violations
namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME




} // DEFINED_ZONE_NAME

INTERNAL_IMPORT_EXPORT_BODY ErrorEbmType CreateLoss_Cpu_64(
   const size_t cOutputs,
   const char * const sLoss,
   const char * const sLossEnd,
   const void ** const ppLossOut
) {
   UNUSED(cOutputs);
   UNUSED(sLoss);
   UNUSED(sLossEnd);
   UNUSED(ppLossOut);

   return Error_None;
}
