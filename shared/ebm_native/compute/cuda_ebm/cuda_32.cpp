// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <cmath>

#include "ebm_native.h"
#include "logging.h"
#include "common_c.h"
#include "bridge_c.h"
#include "zones.h"

#include "common_cpp.hpp"
#include "bridge_cpp.hpp"

#include "Registrable.hpp"
#include "Registration.hpp"
#include "Loss.hpp"

bool TestCuda();
INTERNAL_IMPORT_EXPORT_BODY ErrorEbmType CreateLoss_Cuda_32(
   const Config * const pConfig,
   const char * const sLoss,
   const char * const sLossEnd,
   LossWrapper * const pLossWrapperOut
) {
   UNUSED(pConfig);
   UNUSED(sLoss);
   UNUSED(sLossEnd);
   UNUSED(pLossWrapperOut);

   TestCuda();

   return Error_UnknownInternalError;
}
