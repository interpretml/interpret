// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef COMPUTE_ACCESSORS_HPP
#define COMPUTE_ACCESSORS_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h"
#include "bridge_c.h" // CreateLoss_*
#include "zones.h"

#include "common_cpp.hpp" // INLINE_RELEASE_UNTEMPLATED

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

INLINE_RELEASE_UNTEMPLATED static ErrorEbm GetLoss(
   const bool bClassification,
   const Config * const pConfig,
   const char * sLoss,
   LossWrapper * const pLossWrapperOut
) noexcept {
   EBM_ASSERT(nullptr != pConfig);
   EBM_ASSERT(nullptr != pLossWrapperOut);
   EBM_ASSERT(nullptr == pLossWrapperOut->m_pLoss);
   EBM_ASSERT(nullptr == pLossWrapperOut->m_pFunctionPointersCpp);

   static const char g_sMse[] = "mse";
   static const char g_sLogLoss[] = "log_loss";

   if(nullptr == sLoss) {
      sLoss = bClassification ? g_sLogLoss : g_sMse;
   }
   sLoss = SkipWhitespace(sLoss);
   if('\0' == *sLoss) {
      sLoss = bClassification ? g_sLogLoss : g_sMse;
   }
   const char * const sLossEnd = sLoss + strlen(sLoss);

   ErrorEbm error;

   error = CreateLoss_Cpu_64(pConfig, sLoss, sLossEnd, pLossWrapperOut);

   return error;
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
