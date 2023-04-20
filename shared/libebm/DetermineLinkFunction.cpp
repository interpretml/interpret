// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // LIKELY
#include "zones.h"

#include "compute_accessors.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION DetermineLinkFunction(
   BoolEbm isClassification, 
   const char * objective, 
   LinkEbm * linkOut,
   double * linkParamOut
) {
   LossWrapper loss;

   Config config;
   config.cOutputs = 1; // this is kind of cheating, but it should work
   const ErrorEbm error = GetLoss(!!isClassification, &config, objective, &loss);
   if(Error_None != error) {
      if(nullptr != linkOut) {
         *linkOut = Link_ERROR;
      }
      if(nullptr != linkParamOut) {
         *linkParamOut = std::numeric_limits<double>::quiet_NaN();
      }
      return error;
   }

   // this leaves the contents that are not pointers
   FreeLossWrapperInternals(&loss);

   if(nullptr != linkOut) {
      *linkOut = loss.m_linkFunction;
   }
   if(nullptr != linkParamOut) {
      *linkParamOut = loss.m_linkParam;
   }
   return Error_None;
}

EBM_API_BODY const char * EBM_CALLING_CONVENTION GetLinkFunctionString(LinkEbm link) {
   static const char g_sERROR[] = "ERROR";
   static const char g_sCustom[] = "custom";
   static const char g_sPower[] = "power";
   static const char g_sLogit[] = "logit";
   static const char g_sProbit[] = "probit";
   static const char g_sCloglog[] = "cloglog";
   static const char g_sLoglog[] = "loglog";
   static const char g_sCauchit[] = "cauchit";
   static const char g_sIdentity[] = "identity";
   static const char g_sLog[] = "log";
   static const char g_sInverse[] = "inverse";
   static const char g_sInverseSquare[] = "inverse_square";
   static const char g_sSqrt[] = "sqrt";

   switch(link) {
   case Link_custom:
      return g_sCustom;
   case Link_power:
      return g_sPower;
   case Link_logit:
      return g_sLogit;
   case Link_probit:
      return g_sProbit;
   case Link_cloglog:
      return g_sCloglog;
   case Link_loglog:
      return g_sLoglog;
   case Link_cauchit:
      return g_sCauchit;
   case Link_identity:
      return g_sIdentity;
   case Link_log:
      return g_sLog;
   case Link_inverse:
      return g_sInverse;
   case Link_inverse_square:
      return g_sInverseSquare;
   case Link_sqrt:
      return g_sSqrt;
   default:
      return g_sERROR;
   }
}

} // DEFINED_ZONE_NAME
