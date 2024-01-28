// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef BRIDGE_C_H
#define BRIDGE_C_H

#include <stdlib.h> // free

#include "libebm.h" // ErrorEbm, BoolEbm, etc..
#include "logging.h"
#include "unzoned.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef double FloatShared;

typedef uint64_t UIntBig;
typedef double FloatBig;
typedef uint32_t UIntSmall;
typedef float FloatSmall;

static_assert(sizeof(UIntSmall) < sizeof(UIntBig), "UIntBig must be able to contain UIntSmall");
static_assert(sizeof(FloatSmall) < sizeof(FloatBig), "FloatBig must be able to contain FloatSmall");

struct ApplyUpdateBridge {
   size_t m_cScores;
   int m_cPack;

   BoolEbm m_bHessianNeeded;

   BoolEbm m_bValidation;
   BoolEbm m_bDisableApprox;
   void* m_aMulticlassMidwayTemp; // float or double
   const void* m_aUpdateTensorScores; // float or double
   size_t m_cSamples;
   const void* m_aPacked; // uint64_t or uint32_t
   const void* m_aTargets; // uint64_t or uint32_t or float or double
   const void* m_aWeights; // float or double
   void* m_aSampleScores; // float or double
   void* m_aGradientsAndHessians; // float or double

   double m_metricOut;
};

struct BinSumsBoostingBridge {
   BoolEbm m_bHessian;
   size_t m_cScores;

   int m_cPack;

   size_t m_cSamples;
   const void* m_aGradientsAndHessians; // float or double
   const void* m_aWeights; // float or double
   const uint8_t* m_pCountOccurrences;
   const void* m_aPacked; // uint64_t or uint32_t

   void* m_aFastBins; // Bin<...> (can't use BinBase * since this is only C here)

#ifndef NDEBUG
   const void* m_pDebugFastBinsEnd;
#endif // NDEBUG
};

struct BinSumsInteractionBridge {
   BoolEbm m_bHessian;
   size_t m_cScores;

   size_t m_cSamples;
   const void* m_aGradientsAndHessians; // float or double
   const void* m_aWeights; // float or double

   size_t m_cRuntimeRealDimensions;
   size_t m_acBins[k_cDimensionsMax];
   int m_acItemsPerBitPack[k_cDimensionsMax];
   const void* m_aaPacked[k_cDimensionsMax]; // uint64_t or uint32_t

   void* m_aFastBins; // Bin<...> (can't use BinBase * since this is only C here)

#ifndef NDEBUG
   const void* m_pDebugFastBinsEnd;
#endif // NDEBUG
};

struct ObjectiveWrapper;

// these are extern "C" function pointers so we can't call anything other than an extern "C" function with them
typedef ErrorEbm (*APPLY_UPDATE_C)(const ObjectiveWrapper* const pObjectiveWrapper, ApplyUpdateBridge* const pData);
typedef double (*FINISH_METRIC_C)(const ObjectiveWrapper* const pObjectiveWrapper, const double metricSum);
typedef BoolEbm (*CHECK_TARGETS_C)(
      const ObjectiveWrapper* const pObjectiveWrapper, const size_t c, const void* const aTargets);

typedef ErrorEbm (*BIN_SUMS_BOOSTING_C)(
      const ObjectiveWrapper* const pObjectiveWrapper, BinSumsBoostingBridge* const pParams);
typedef ErrorEbm (*BIN_SUMS_INTERACTION_C)(
      const ObjectiveWrapper* const pObjectiveWrapper, BinSumsInteractionBridge* const pParams);

struct ObjectiveWrapper {
   APPLY_UPDATE_C m_pApplyUpdateC;
   BIN_SUMS_BOOSTING_C m_pBinSumsBoostingC;
   BIN_SUMS_INTERACTION_C m_pBinSumsInteractionC;
   // everything below here the C++ *Objective specific class needs to fill out

   // this needs to be void since our Registrable object is C++ visible and we cannot define it initially
   // here in this C file since our object needs to be a POD and thus can't inherit data
   // and it cannot be empty either since empty structures are not compliant in all C compilers
   // https://stackoverflow.com/questions/755305/empty-structure-in-c?rq=1
   void* m_pObjective;

   BoolEbm m_bMaximizeMetric;

   LinkEbm m_linkFunction;
   double m_linkParam;

   double m_learningRateAdjustmentDifferentialPrivacy;
   double m_learningRateAdjustmentGradientBoosting;
   double m_learningRateAdjustmentHessianBoosting;
   double m_gainAdjustmentGradientBoosting;
   double m_gainAdjustmentHessianBoosting;

   double m_gradientConstant;
   double m_hessianConstant;
   BoolEbm m_bObjectiveHasHessian;
   BoolEbm m_bRmse;

   size_t m_cSIMDPack;

   size_t m_cFloatBytes;
   size_t m_cUIntBytes;

   AccelerationFlags m_zones;

   // these are C++ function pointer definitions that exist per-zone, and must remain hidden in the C interface
   void* m_pFunctionPointersCpp;
};

inline static void InitializeObjectiveWrapperUnfailing(ObjectiveWrapper* const pObjectiveWrapper) {
   pObjectiveWrapper->m_pObjective = NULL;
   pObjectiveWrapper->m_bMaximizeMetric = EBM_FALSE;
   pObjectiveWrapper->m_linkFunction = Link_ERROR;
   pObjectiveWrapper->m_linkParam = 0.0;
   pObjectiveWrapper->m_learningRateAdjustmentDifferentialPrivacy = 0.0;
   pObjectiveWrapper->m_learningRateAdjustmentGradientBoosting = 0.0;
   pObjectiveWrapper->m_learningRateAdjustmentHessianBoosting = 0.0;
   pObjectiveWrapper->m_gainAdjustmentGradientBoosting = 0.0;
   pObjectiveWrapper->m_gainAdjustmentHessianBoosting = 0.0;
   pObjectiveWrapper->m_gradientConstant = 0.0;
   pObjectiveWrapper->m_hessianConstant = 0.0;
   pObjectiveWrapper->m_bObjectiveHasHessian = EBM_FALSE;
   pObjectiveWrapper->m_bRmse = EBM_FALSE;
   pObjectiveWrapper->m_cSIMDPack = 0;
   pObjectiveWrapper->m_cFloatBytes = 0;
   pObjectiveWrapper->m_cUIntBytes = 0;
   pObjectiveWrapper->m_pFunctionPointersCpp = NULL;
}

inline static void FreeObjectiveWrapperInternals(ObjectiveWrapper* const pObjectiveWrapper) {
   AlignedFree(pObjectiveWrapper->m_pObjective);
   free(pObjectiveWrapper->m_pFunctionPointersCpp);
}

struct Config {
   // don't use m_ notation here, mostly to make it cleaner for people writing *Objective classes
   size_t cOutputs;
   BoolEbm isDifferentialPrivacy;
};

INTERNAL_IMPORT_EXPORT_INCLUDE ErrorEbm CreateObjective_Cpu_64(const Config* const pConfig,
      const char* const sObjective,
      const char* const sObjectiveEnd,
      ObjectiveWrapper* const pObjectiveWrapperOut);

INTERNAL_IMPORT_EXPORT_INCLUDE ErrorEbm CreateObjective_Avx512f_32(const Config* const pConfig,
      const char* const sObjective,
      const char* const sObjectiveEnd,
      ObjectiveWrapper* const pObjectiveWrapperOut);

INTERNAL_IMPORT_EXPORT_INCLUDE ErrorEbm CreateObjective_Avx2_32(const Config* const pConfig,
      const char* const sObjective,
      const char* const sObjectiveEnd,
      ObjectiveWrapper* const pObjectiveWrapperOut);

INTERNAL_IMPORT_EXPORT_INCLUDE ErrorEbm CreateObjective_Cuda_32(const Config* const pConfig,
      const char* const sObjective,
      const char* const sObjectiveEnd,
      ObjectiveWrapper* const pObjectiveWrapperOut);

INTERNAL_IMPORT_EXPORT_INCLUDE ErrorEbm CreateMetric_Cpu_64(
      const Config* const pConfig, const char* const sMetric, const char* const sMetricEnd
      //   MetricWrapper * const pMetricWrapperOut,
);

INTERNAL_IMPORT_EXPORT_INCLUDE double FinishMetricC(
      const ObjectiveWrapper* const pObjectiveWrapper, const double metricSum);
INTERNAL_IMPORT_EXPORT_INCLUDE BoolEbm CheckTargetsC(
      const ObjectiveWrapper* const pObjectiveWrapper, const size_t c, const void* const aTargets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // BRIDGE_C_H
