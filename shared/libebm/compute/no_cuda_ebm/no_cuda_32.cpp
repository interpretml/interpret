// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include "libebm.h"
#include "logging.h"
#include "common_c.h"
#include "bridge_c.h"
#include "zones.h"

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm CreateObjective_Cuda_32(
   const Config * const pConfig,
   const char * const sObjective,
   const char * const sObjectiveEnd,
   ObjectiveWrapper * const pObjectiveWrapperOut
) {
   UNUSED(pConfig);
   UNUSED(sObjective);
   UNUSED(sObjectiveEnd);
   UNUSED(pObjectiveWrapperOut);

   return Error_UnexpectedInternal;
}
