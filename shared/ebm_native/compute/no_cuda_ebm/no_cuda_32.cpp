// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include "ebm_native.h"
#include "logging.h"
#include "common_c.h"
#include "bridge_c.h"
#include "zones.h"

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm CreateLoss_Cuda_32(
   const Config * const pConfig,
   const char * const sLoss,
   const char * const sLossEnd,
   LossWrapper * const pLossWrapperOut
) {
   UNUSED(pConfig);
   UNUSED(sLoss);
   UNUSED(sLossEnd);
   UNUSED(pLossWrapperOut);

   return Error_UnexpectedInternal;
}
