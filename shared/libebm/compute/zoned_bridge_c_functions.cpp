// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <vector>
#include <algorithm>

#include "libebm.h" // ErrorEbm
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

struct Objective;

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm MAKE_ZONED_C_FUNCTION_NAME(ApplyUpdate) (
   const ObjectiveWrapper * const pObjectiveWrapper,
   ApplyUpdateBridge * const pData
) {
   const Objective * const pObjective = static_cast<const Objective *>(pObjectiveWrapper->m_pObjective);
   const APPLY_UPDATE_CPP pApplyUpdateCpp = 
      (static_cast<FunctionPointersCpp *>(pObjectiveWrapper->m_pFunctionPointersCpp))->m_pApplyUpdateCpp;
   return (*pApplyUpdateCpp)(pObjective, pData);
}

} // DEFINED_ZONE_NAME
