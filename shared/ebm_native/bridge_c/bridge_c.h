// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef BRIDGE_C_H
#define BRIDGE_C_H

#include <stdlib.h> // free

#include "ebm_native.h"
#include "logging.h"
#include "common_c.h"

#ifdef __cplusplus
extern "C" {
#define INTERNAL_IMPORT_EXPORT_BODY extern "C"
#else // __cplusplus
#define INTERNAL_IMPORT_EXPORT_BODY extern
#endif // __cplusplus

#define INTERNAL_IMPORT_EXPORT_INCLUDE extern

typedef size_t StorageDataType;
typedef UIntEbmType ActiveDataType;

struct ApplyTrainingData {
   ptrdiff_t m_cRuntimeScores;
   ptrdiff_t m_cRuntimePack;
   BoolEbmType m_bHessianNeeded;
};

struct ApplyValidationData {
   ptrdiff_t m_cRuntimeScores;
   ptrdiff_t m_cRuntimePack;
   BoolEbmType m_bHessianNeeded;
   double m_metricOut;
};

struct LossWrapper;

// these are extern "C" function pointers so we can't call anything other than an extern "C" function with them
typedef ErrorEbmType (* APPLY_TRAINING_C)(const LossWrapper * const pLossWrapper, ApplyTrainingData * const pData);
typedef ErrorEbmType (* APPLY_VALIDATION_C)(const LossWrapper * const pLossWrapper, ApplyValidationData * const pData);

struct LossWrapper {
   APPLY_TRAINING_C m_pApplyTrainingC;
   APPLY_VALIDATION_C m_pApplyValidationC;
   // everything below here the C++ *Loss specific class needs to fill out

   // this needs to be void since our Registrable object is C++ visible and we cannot define it initially 
   // here in this C file since our object needs to be a POD and thus can't inherit data
   // and it cannot be empty either since empty structures are not compliant in all C compilers
   // https://stackoverflow.com/questions/755305/empty-structure-in-c?rq=1
   void * m_pLoss;
   double m_updateMultiple;
   BoolEbmType m_bLossHasHessian;
   BoolEbmType m_bSuperSuperSpecialLossWhereTargetNotNeededOnlyMseLossQualifies;
   // these are C++ function pointer definitions that exist per-zone, and must remain hidden in the C interface
   void * m_pFunctionPointersCpp;
};

INLINE_ALWAYS static void InitializeLossWrapperUnfailing(LossWrapper * const pLossWrapper) {
   pLossWrapper->m_pLoss = NULL;
   pLossWrapper->m_pFunctionPointersCpp = NULL;
}

INLINE_ALWAYS static void FreeLossWrapperInternals(LossWrapper * const pLossWrapper) {
   free(pLossWrapper->m_pLoss);
   free(pLossWrapper->m_pFunctionPointersCpp);
}

struct Config {
   // don't use m_ notation here, mostly to make it cleaner for people writing *Loss classes
   size_t cOutputs;
};

INTERNAL_IMPORT_EXPORT_INCLUDE ErrorEbmType CreateLoss_Cpu_64(
   const Config * const pConfig,
   const char * const sLoss,
   const char * const sLossEnd,
   LossWrapper * const pLossWrapperOut
);

INTERNAL_IMPORT_EXPORT_INCLUDE ErrorEbmType CreateLoss_Cuda_32(
   const Config * const pConfig,
   const char * const sLoss,
   const char * const sLossEnd,
   LossWrapper * const pLossWrapperOut
);

INTERNAL_IMPORT_EXPORT_INCLUDE ErrorEbmType CreateMetric_Cpu_64(
   const Config * const pConfig,
   const char * const sMetric,
   const char * const sMetricEnd
   //   MetricWrapper * const pMetricWrapperOut,
);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // BRIDGE_C_H
