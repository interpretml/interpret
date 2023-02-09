// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <vector>
#include <algorithm>

#include "ebm_native.h" // ErrorEbm
#include "bridge_c.h" // INTERNAL_IMPORT_EXPORT_BODY
#include "zones.h"

#include "zoned_bridge_c_functions.h"
#include "zoned_bridge_cpp_functions.hpp"

// the static member functions in our classes are extern "CPP" functions, so we need to bridge our extern "C"
// functions (which are the only thing we can can safely bridge over different compilation flags) to extern "CPP"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct Loss;

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm MAKE_ZONED_C_FUNCTION_NAME(ApplyTraining) (
   const LossWrapper * const pLossWrapper,
   ApplyTrainingData * const pData
) {
   const Loss * const pLoss = static_cast<const Loss *>(pLossWrapper->m_pLoss);
   const APPLY_TRAINING_CPP pApplyTrainingCpp = 
      (static_cast<FunctionPointersCpp *>(pLossWrapper->m_pFunctionPointersCpp))->m_pApplyTrainingCpp;
   return (*pApplyTrainingCpp)(pLoss, pData);
}

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm MAKE_ZONED_C_FUNCTION_NAME(ApplyValidation) (
   const LossWrapper * const pLossWrapper,
   ApplyValidationData * const pData
) {
   const Loss * const pLoss = static_cast<const Loss *>(pLossWrapper->m_pLoss);
   const APPLY_VALIDATION_CPP pApplyValidationCpp = 
      (static_cast<FunctionPointersCpp *>(pLossWrapper->m_pFunctionPointersCpp))->m_pApplyValidationCpp;
   return (*pApplyValidationCpp)(pLoss, pData);
}

} // DEFINED_ZONE_NAME
