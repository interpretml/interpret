// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef BRIDGE_C_H
#define BRIDGE_C_H

#include <stdlib.h> // free

#include "libebm.h" // ErrorEbm, BoolEbm, etc..

#include "common_c.h"

#ifdef __cplusplus
extern "C" {
#define INTERNAL_IMPORT_EXPORT_BODY extern "C"
#else // __cplusplus
#define INTERNAL_IMPORT_EXPORT_BODY extern
#endif // __cplusplus

#define INTERNAL_IMPORT_EXPORT_INCLUDE extern


typedef size_t StorageDataType;
typedef UIntEbm ActiveDataType; // TODO: in most places we could use size_t for this and only use the uint64 version where we have cross-platform considerations.

struct ApplyUpdateBridge {
   size_t m_cScores;
   ptrdiff_t m_cPack;

   BoolEbm m_bHessianNeeded;

   bool m_bCalcMetric; // TODO: should this be BoolEbm?
   void * m_aMulticlassMidwayTemp;
   const void * m_aUpdateTensorScores;
   size_t m_cSamples;
   const StorageDataType * m_aPacked;
   const void * m_aTargets;
   const void * m_aWeights;
   void * m_aSampleScores;
   void * m_aGradientsAndHessians;

   double m_metricOut;
};

struct ObjectiveWrapper;

// these are extern "C" function pointers so we can't call anything other than an extern "C" function with them
typedef ErrorEbm (* APPLY_UPDATE_C)(const ObjectiveWrapper * const pObjectiveWrapper, ApplyUpdateBridge * const pData);
typedef double (* FINISH_METRIC_C)(const ObjectiveWrapper * const pObjectiveWrapper, const double metricSum);

struct ObjectiveWrapper {
   APPLY_UPDATE_C m_pApplyUpdateC;
   FINISH_METRIC_C m_pFinishMetricC;
   // everything below here the C++ *Objective specific class needs to fill out

   // this needs to be void since our Registrable object is C++ visible and we cannot define it initially 
   // here in this C file since our object needs to be a POD and thus can't inherit data
   // and it cannot be empty either since empty structures are not compliant in all C compilers
   // https://stackoverflow.com/questions/755305/empty-structure-in-c?rq=1
   void * m_pObjective;

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

   // these are C++ function pointer definitions that exist per-zone, and must remain hidden in the C interface
   void * m_pFunctionPointersCpp;
};

inline static void InitializeObjectiveWrapperUnfailing(ObjectiveWrapper * const pObjectiveWrapper) {
   pObjectiveWrapper->m_pObjective = NULL;
   pObjectiveWrapper->m_pFunctionPointersCpp = NULL;
}

inline static void FreeObjectiveWrapperInternals(ObjectiveWrapper * const pObjectiveWrapper) {
   free(pObjectiveWrapper->m_pObjective);
   free(pObjectiveWrapper->m_pFunctionPointersCpp);
}

struct Config {
   // don't use m_ notation here, mostly to make it cleaner for people writing *Objective classes
   size_t cOutputs;
};

INTERNAL_IMPORT_EXPORT_INCLUDE ErrorEbm CreateObjective_Cpu_64(
   const Config * const pConfig,
   const char * const sObjective,
   const char * const sObjectiveEnd,
   ObjectiveWrapper * const pObjectiveWrapperOut
);

INTERNAL_IMPORT_EXPORT_INCLUDE ErrorEbm CreateObjective_Cuda_32(
   const Config * const pConfig,
   const char * const sObjective,
   const char * const sObjectiveEnd,
   ObjectiveWrapper * const pObjectiveWrapperOut
);

INTERNAL_IMPORT_EXPORT_INCLUDE ErrorEbm CreateMetric_Cpu_64(
   const Config * const pConfig,
   const char * const sMetric,
   const char * const sMetricEnd
   //   MetricWrapper * const pMetricWrapperOut,
);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // BRIDGE_C_H
