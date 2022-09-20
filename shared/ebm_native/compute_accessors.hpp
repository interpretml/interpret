// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef COMPUTE_ACCESSORS_HPP
#define COMPUTE_ACCESSORS_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // SkipEndWhitespaceWhenGuaranteedNonWhitespace
#include "bridge_c.h" // CreateLoss_*
#include "zones.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

INLINE_ALWAYS static ErrorEbm GetLoss(
   const Config * const pConfig,
   const char * sLoss,
   LossWrapper * const pLossWrapperOut
) noexcept {
   EBM_ASSERT(nullptr != pConfig);
   EBM_ASSERT(nullptr != pLossWrapperOut);
   pLossWrapperOut->m_pLoss = nullptr;
   pLossWrapperOut->m_pFunctionPointersCpp = nullptr;

   if(nullptr == sLoss) {
      // TODO: in the future use a default
      return Error_LossUnknown;
   }
   sLoss = SkipWhitespace(sLoss);
   if('\0' == *sLoss) {
      // TODO: in the future use a default
      return Error_LossUnknown;
   }
   const char * const sLossEnd = SkipEndWhitespaceWhenGuaranteedNonWhitespace(sLoss + strlen(sLoss));

   ErrorEbm error;

   error = CreateLoss_Cpu_64(pConfig, sLoss, sLossEnd, pLossWrapperOut);

   return error;
}

INLINE_ALWAYS static ErrorEbm GetMetrics(
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
      const char * sMetricSeparator = strchr(sMetric, k_registrationSeparator);
      if(nullptr == sMetricSeparator) {
         // find the null terminator then
         sMetricSeparator = sMetric + strlen(sMetric);
      }
      if(sMetricSeparator != sMetric) {
         // we allow empty registrations like ",,,something_legal,,,  something_else  , " since the intent is clear

         const char * const sMetricEnd = SkipEndWhitespaceWhenGuaranteedNonWhitespace(sMetricSeparator);
         ErrorEbm error;

         error = CreateMetric_Cpu_64(pConfig, sMetric, sMetricEnd);
         if(Error_None != error) {
            return error;
         }

         // TODO: for now let's return after we find the first metric, but in the future we'll want to return
         //       some kind of list of them
         return error;
      }
      if('\0' == *sMetricSeparator) {
         return Error_None;
      }
      EBM_ASSERT(k_registrationSeparator == *sMetricSeparator);

      sMetric = sMetricSeparator + 1;
   }
}

} // DEFINED_ZONE_NAME

#endif // COMPUTE_ACCESSORS_HPP
