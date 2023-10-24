// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include <stddef.h> // size_t, ptrdiff_t
#include <vector>
#include <algorithm>

#include "libebm.h" // ErrorEbm
#include "logging.h"
#include "bridge_c.h" // INTERNAL_IMPORT_EXPORT_BODY
#include "zones.h"

#include "common_cpp.hpp"
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

   // all our memory should be aligned. It is required by SIMD for correctness or performance
   EBM_ASSERT(IsAligned(pData->m_aMulticlassMidwayTemp));
   EBM_ASSERT(IsAligned(pData->m_aUpdateTensorScores));
   EBM_ASSERT(IsAligned(pData->m_aPacked));
   EBM_ASSERT(IsAligned(pData->m_aTargets));
   EBM_ASSERT(IsAligned(pData->m_aWeights));
   EBM_ASSERT(IsAligned(pData->m_aSampleScores));
   EBM_ASSERT(IsAligned(pData->m_aGradientsAndHessians));

   return (*pApplyUpdateCpp)(pObjective, pData);
}

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm MAKE_ZONED_C_FUNCTION_NAME(BinSumsBoosting)(
   const ObjectiveWrapper * const pObjectiveWrapper,
   BinSumsBoostingBridge * const pParams
) {
   const BIN_SUMS_BOOSTING_CPP pBinSumsBoostingCpp =
      (static_cast<FunctionPointersCpp *>(pObjectiveWrapper->m_pFunctionPointersCpp))->m_pBinSumsBoostingCpp;

   //// all our memory should be aligned. It is required by SIMD for correctness or performance
   EBM_ASSERT(IsAligned(pParams->m_aGradientsAndHessians));
   EBM_ASSERT(IsAligned(pParams->m_aWeights));
   EBM_ASSERT(IsAligned(pParams->m_pCountOccurrences));
   EBM_ASSERT(IsAligned(pParams->m_aPacked));
   EBM_ASSERT(IsAligned(pParams->m_aFastBins));

   return (*pBinSumsBoostingCpp)(pParams);
}

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm MAKE_ZONED_C_FUNCTION_NAME(BinSumsInteraction)(
   const ObjectiveWrapper * const pObjectiveWrapper,
   BinSumsInteractionBridge * const pParams
) {
   const BIN_SUMS_INTERACTION_CPP pBinSumsInteractionCpp =
      (static_cast<FunctionPointersCpp *>(pObjectiveWrapper->m_pFunctionPointersCpp))->m_pBinSumsInteractionCpp;

#ifndef NDEBUG
   //// all our memory should be aligned. It is required by SIMD for correctness or performance
   EBM_ASSERT(IsAligned(pParams->m_aGradientsAndHessians));
   EBM_ASSERT(IsAligned(pParams->m_aWeights));
   EBM_ASSERT(IsAligned(pParams->m_aFastBins));
   for(size_t iDebug = 0; iDebug < pParams->m_cRuntimeRealDimensions; ++iDebug) {
      EBM_ASSERT(IsAligned(pParams->m_aaPacked[iDebug]));
   }
#endif // NDEBUG

   return (*pBinSumsInteractionCpp)(pParams);
}

#ifdef ZONE_cpu
INTERNAL_IMPORT_EXPORT_BODY double MAKE_ZONED_C_FUNCTION_NAME(FinishMetric) (
   const ObjectiveWrapper * const pObjectiveWrapper,
   const double metricSum
) {
   const Objective * const pObjective = static_cast<const Objective *>(pObjectiveWrapper->m_pObjective);
   const FINISH_METRIC_CPP pFinishMetricCpp =
      (static_cast<const FunctionPointersCpp *>(pObjectiveWrapper->m_pFunctionPointersCpp))->m_pFinishMetricCpp;
   return (*pFinishMetricCpp)(pObjective, metricSum);
}

INTERNAL_IMPORT_EXPORT_BODY BoolEbm MAKE_ZONED_C_FUNCTION_NAME(CheckTargets) (
   const ObjectiveWrapper * const pObjectiveWrapper,
   const size_t c, 
   const void * const aTargets
) {
   EBM_ASSERT(nullptr != pObjectiveWrapper);
   EBM_ASSERT(nullptr != aTargets);
   const Objective * const pObjective = static_cast<const Objective *>(pObjectiveWrapper->m_pObjective);
   EBM_ASSERT(nullptr != pObjective);
   const CHECK_TARGETS_CPP pCheckTargetsCpp =
      (static_cast<const FunctionPointersCpp *>(pObjectiveWrapper->m_pFunctionPointersCpp))->m_pCheckTargetsCpp;
   EBM_ASSERT(nullptr != pCheckTargetsCpp);
   return (*pCheckTargetsCpp)(pObjective, c, aTargets);
}
#endif // ZONE_cpu

} // DEFINED_ZONE_NAME
