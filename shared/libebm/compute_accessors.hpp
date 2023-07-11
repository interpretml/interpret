// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef COMPUTE_ACCESSORS_HPP
#define COMPUTE_ACCESSORS_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h"
#include "bridge_c.h" // CreateObjective_*
#include "zones.h"

#include "common_cpp.hpp" // INLINE_RELEASE_UNTEMPLATED

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

INLINE_RELEASE_UNTEMPLATED static ErrorEbm GetObjective(
   const Config * const pConfig,
   const char * sObjective,
   ObjectiveWrapper * const pCpuObjectiveWrapperOut,
   ObjectiveWrapper * const pSIMDObjectiveWrapperOut
) noexcept {
   EBM_ASSERT(nullptr != pConfig);
   EBM_ASSERT(nullptr != pCpuObjectiveWrapperOut);
   EBM_ASSERT(nullptr == pCpuObjectiveWrapperOut->m_pObjective);
   EBM_ASSERT(nullptr == pCpuObjectiveWrapperOut->m_pFunctionPointersCpp);

   EBM_ASSERT(nullptr == pSIMDObjectiveWrapperOut || nullptr == pSIMDObjectiveWrapperOut->m_pObjective);
   EBM_ASSERT(nullptr == pSIMDObjectiveWrapperOut || nullptr == pSIMDObjectiveWrapperOut->m_pFunctionPointersCpp);

   if(nullptr == sObjective) {
      return Error_ObjectiveUnknown;
   }
   sObjective = SkipWhitespace(sObjective);
   if('\0' == *sObjective) {
      return Error_ObjectiveUnknown;
   }

   const char * const sObjectiveEnd = sObjective + strlen(sObjective);

   ErrorEbm error;

   error = CreateObjective_Cpu_64(pConfig, sObjective, sObjectiveEnd, pCpuObjectiveWrapperOut);
   if(Error_None != error) {
      return error;
   }

   if(nullptr != pSIMDObjectiveWrapperOut) {
      // TODO: add a flag in the pCpuObjectiveWrapperOut struct that indicates if the objective can be SIMDed
      //       we first make the cpu version and if that says it can't be SIMDed then we shouldn't try

#ifdef BRIDGE_SSE2_32
      LOG_0(Trace_Info, "INFO GetObjective creating SSE2 SIMD Objective");
      error = CreateObjective_Sse2_32(pConfig, sObjective, sObjectiveEnd, pSIMDObjectiveWrapperOut);
      if(Error_None != error) {
         return error;
      }
#endif //BRIDGE_SSE2_32
   }

   return Error_None;
}

INLINE_RELEASE_UNTEMPLATED static ErrorEbm GetMetrics(
   const Config * const pConfig,
   const char * sMetric
//   MetricWrapper * const aMetricWrapperOut
) noexcept {
   EBM_ASSERT(nullptr != pConfig);
   //EBM_ASSERT(nullptr != pMetricWrapperOut);
   //aMetricWrapperOut->m_pMetric = nullptr;
   //aMetricWrapperOut->m_pFunctionPointersCpp = nullptr;

   if(nullptr == sMetric) {
      // it's legal to have no metrics
      return Error_None;
   }
   while(true) {
      sMetric = SkipWhitespace(sMetric);
      const char * sMetricEnd = strchr(sMetric, k_registrationSeparator);
      if(nullptr == sMetricEnd) {
         // find the null terminator then
         sMetricEnd = sMetric + strlen(sMetric);
      }
      if(sMetricEnd != sMetric) {
         // we allow empty registrations like ",,,something_legal,,,  something_else  , " since the intent is clear
         ErrorEbm error;

         error = CreateMetric_Cpu_64(pConfig, sMetric, sMetricEnd);
         if(Error_None != error) {
            return error;
         }

         // TODO: for now let's return after we find the first metric, but in the future we'll want to return
         //       some kind of list of them
         return error;
      }
      if('\0' == *sMetricEnd) {
         return Error_None;
      }
      EBM_ASSERT(k_registrationSeparator == *sMetricEnd);

      sMetric = sMetricEnd + 1;
   }
}

} // DEFINED_ZONE_NAME

#endif // COMPUTE_ACCESSORS_HPP
