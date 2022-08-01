// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef ZONED_BRIDGE_C_FUNCTIONS_HPP
#define ZONED_BRIDGE_C_FUNCTIONS_HPP

#include "ebm_native.h"
#include "bridge_c.h"
#include "zones.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

INTERNAL_IMPORT_EXPORT_INCLUDE ErrorEbm MAKE_ZONED_C_FUNCTION_NAME(ApplyTraining)(
   const LossWrapper * const pLossWrapper,
   ApplyTrainingData * const pData
);

INTERNAL_IMPORT_EXPORT_INCLUDE ErrorEbm MAKE_ZONED_C_FUNCTION_NAME(ApplyValidation)(
   const LossWrapper * const pLossWrapper,
   ApplyValidationData * const pData
);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // ZONED_BRIDGE_C_FUNCTIONS_HPP
