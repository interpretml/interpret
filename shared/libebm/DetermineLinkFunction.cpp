// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "unzoned.h" // LIKELY

#define ZONE_main
#include "zones.h"

#include "bridge.hpp" // IdentifyTask

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern ErrorEbm GetObjective(
   const Config * const pConfig,
   const char * sObjective,
   const AccelerationFlags acceleration,
   ObjectiveWrapper * const pCpuObjectiveWrapperOut,
   ObjectiveWrapper * const pSIMDObjectiveWrapperOut
) noexcept;

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION DetermineLinkFunction(
   LinkFlags flags,
   const char * objective,
   IntEbm countClasses,
   LinkEbm * linkOut,
   double * linkParamOut
) {
   LOG_N(
      Trace_Info,
      "Entered DetermineLinkFunction: "
      "flags=0x%" ULinkFlagsPrintf ", "
      "objective=%p, "
      "countClasses=%" IntEbmPrintf ", "
      "linkOut=%p, "
      "linkParamOut=%p"
      ,
      static_cast<ULinkFlags>(flags), // signed to unsigned conversion is defined behavior in C++
      static_cast<const void *>(objective),
      countClasses,
      static_cast<void *>(linkOut),
      static_cast<void *>(linkParamOut)
   );

   if(IsConvertError<ptrdiff_t>(countClasses)) {
      LOG_0(Trace_Error, "ERROR DetermineLinkFunction IsConvertError<ptrdiff_t>(countClasses)");
      return Error_IllegalParamVal;
   }
   const ptrdiff_t cClasses = static_cast<ptrdiff_t>(countClasses);

   if(ptrdiff_t { 0 } == cClasses || ptrdiff_t { 1 } == cClasses) {
      if(nullptr != linkOut) {
         *linkOut = Link_monoclassification;
      }
      if(nullptr != linkParamOut) {
         *linkParamOut = std::numeric_limits<double>::quiet_NaN();
      }

      LOG_0(Trace_Info, "Exited DetermineLinkFunction");

      return Error_None;
   }

   size_t cScores;
   if(0 != (LinkFlags_BinaryAsMulticlass & flags)) {
      cScores = cClasses < ptrdiff_t { 2 } ? size_t { 1 } : static_cast<size_t>(cClasses);
   } else {
      cScores = cClasses <= ptrdiff_t { 2 } ? size_t { 1 } : static_cast<size_t>(cClasses);
   }

   ObjectiveWrapper objectiveWrapper;
   InitializeObjectiveWrapperUnfailing(&objectiveWrapper);

   Config config;
   config.cOutputs = cScores;
   config.isDifferentialPrivacy = 0 != (LinkFlags_DifferentialPrivacy & flags) ? EBM_TRUE : EBM_FALSE;
   const ErrorEbm error = GetObjective(&config, objective, AccelerationFlags_NONE, &objectiveWrapper, nullptr);
   if(Error_None != error) {
      LOG_0(Trace_Error, "ERROR DetermineLinkFunction GetObjective failed");

      if(nullptr != linkOut) {
         *linkOut = Link_ERROR;
      }
      if(nullptr != linkParamOut) {
         *linkParamOut = std::numeric_limits<double>::quiet_NaN();
      }
      return error;
   }

   // this leaves the contents that are not pointers
   FreeObjectiveWrapperInternals(&objectiveWrapper);

   if(nullptr != linkOut) {
      *linkOut = objectiveWrapper.m_linkFunction;
   }
   if(nullptr != linkParamOut) {
      *linkParamOut = objectiveWrapper.m_linkParam;
   }

   LOG_0(Trace_Info, "Exited DetermineLinkFunction");

   return Error_None;
}

static const char g_sCustomRegression[] = "custom_regression";
static const char g_sCustomRanking[] = "custom_ranking";
static const char g_sMonoClassification[] = "monoclassification";
static const char g_sCustomBinaryClassification[] = "custom_binary";
static const char g_sCustomOvrClassification[] = "custom_ovr";
static const char g_sCustomMultinomialClassification[] = "custom_multinomial";
static const char g_sMultinominalLogit[] = "mlogit";
static const char g_sOvrLogit[] = "vlogit";
static const char g_sLogit[] = "logit";
static const char g_sProbit[] = "probit";
static const char g_sCloglog[] = "cloglog";
static const char g_sLoglog[] = "loglog";
static const char g_sCauchit[] = "cauchit";
static const char g_sPower[] = "power";
static const char g_sIdentity[] = "identity";
static const char g_sLog[] = "log";
static const char g_sInverse[] = "inverse";
static const char g_sInverseSquare[] = "inverse_square";
static const char g_sSqrt[] = "sqrt";

EBM_API_BODY const char * EBM_CALLING_CONVENTION IdentifyLinkFunctionStr(LinkEbm link) {
   switch(link) {
   case Link_custom_regression:
      return g_sCustomRegression;
   case Link_custom_ranking:
      return g_sCustomRanking;
   case Link_monoclassification:
      return g_sMonoClassification;
   case Link_custom_binary:
      return g_sCustomBinaryClassification;
   case Link_custom_ovr:
      return g_sCustomOvrClassification;
   case Link_custom_multinomial:
      return g_sCustomMultinomialClassification;
   case Link_mlogit:
      return g_sMultinominalLogit;
   case Link_vlogit:
      return g_sOvrLogit;
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
   case Link_power:
      return g_sPower;
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
      return nullptr;
   }
}

EBM_API_BODY LinkEbm EBM_CALLING_CONVENTION IdentifyLinkFunctionInt(const char * link) {
   if(nullptr != link) {
      link = SkipWhitespace(link);
      if(IsStringEqualsForgiving(link, g_sCustomRegression))
         return Link_custom_regression;
      if(IsStringEqualsForgiving(link, g_sCustomRanking))
         return Link_custom_ranking;
      if(IsStringEqualsForgiving(link, g_sMonoClassification))
         return Link_monoclassification;
      if(IsStringEqualsForgiving(link, g_sCustomBinaryClassification))
         return Link_custom_binary;
      if(IsStringEqualsForgiving(link, g_sCustomOvrClassification))
         return Link_custom_ovr;
      if(IsStringEqualsForgiving(link, g_sCustomMultinomialClassification))
         return Link_custom_multinomial;
      if(IsStringEqualsForgiving(link, g_sMultinominalLogit))
         return Link_mlogit;
      if(IsStringEqualsForgiving(link, g_sOvrLogit))
         return Link_vlogit;
      if(IsStringEqualsForgiving(link, g_sLogit))
         return Link_logit;
      if(IsStringEqualsForgiving(link, g_sProbit))
         return Link_probit;
      if(IsStringEqualsForgiving(link, g_sCloglog))
         return Link_cloglog;
      if(IsStringEqualsForgiving(link, g_sLoglog))
         return Link_loglog;
      if(IsStringEqualsForgiving(link, g_sCauchit))
         return Link_cauchit;
      if(IsStringEqualsForgiving(link, g_sPower))
         return Link_power;
      if(IsStringEqualsForgiving(link, g_sIdentity))
         return Link_identity;
      if(IsStringEqualsForgiving(link, g_sLog))
         return Link_log;
      if(IsStringEqualsForgiving(link, g_sInverse))
         return Link_inverse;
      if(IsStringEqualsForgiving(link, g_sInverseSquare))
         return Link_inverse_square;
      if(IsStringEqualsForgiving(link, g_sSqrt))
         return Link_sqrt;
   }
   return Link_ERROR;
}

EBM_API_BODY TaskEbm EBM_CALLING_CONVENTION IdentifyTaskInt(LinkEbm link) {
   return IdentifyTask(link);
}

static const char g_sClassification[] = "classification";
static const char g_sRegression[] = "regression";
static const char g_sRanking[] = "ranking";

EBM_API_BODY const char * EBM_CALLING_CONVENTION IdentifyTaskStr(const char * link) {
   const TaskEbm task = IdentifyTask(IdentifyLinkFunctionInt(link));
   if(Task_GeneralClassification <= task) {
      return g_sClassification;
   }
   if(Task_Regression == task) {
      return g_sRegression;
   }
   if(Task_Ranking == task) {
      return g_sRanking;
   }
   return nullptr;
}

} // DEFINED_ZONE_NAME
