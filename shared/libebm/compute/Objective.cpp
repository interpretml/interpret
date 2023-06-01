// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <memory> // shared_ptr, unique_ptr
#include <vector>

#include "zoned_bridge_c_functions.h"
#include "registration_exceptions.hpp"
#include "Registration.hpp"
#include "Objective.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

ErrorEbm Objective::CreateObjective(
   const REGISTER_OBJECTIVES_FUNCTION registerObjectivesFunction,
   const Config * const pConfig,
   const char * const sObjective,
   const char * const sObjectiveEnd,
   ObjectiveWrapper * const pObjectiveWrapperOut
) noexcept {
   EBM_ASSERT(nullptr != registerObjectivesFunction);
   EBM_ASSERT(nullptr != pConfig);
   EBM_ASSERT(1 <= pConfig->cOutputs);
   EBM_ASSERT(EBM_FALSE == pConfig->isDifferentiallyPrivate || EBM_TRUE == pConfig->isDifferentiallyPrivate);
   EBM_ASSERT(nullptr != sObjective);
   EBM_ASSERT(nullptr != sObjectiveEnd);
   EBM_ASSERT(sObjective < sObjectiveEnd); // empty string not allowed
   EBM_ASSERT('\0' != *sObjective);
   EBM_ASSERT(!(0x20 == *sObjective || (0x9 <= *sObjective && *sObjective <= 0xd)));
   EBM_ASSERT('\0' == *sObjectiveEnd);
   EBM_ASSERT(nullptr != pObjectiveWrapperOut);
   EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
   EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pFunctionPointersCpp);

   LOG_0(Trace_Info, "Entered Objective::CreateObjective");

   void * const pFunctionPointersCpp = malloc(sizeof(FunctionPointersCpp));
   ErrorEbm error = Error_OutOfMemory;
   if(nullptr != pFunctionPointersCpp) {
      pObjectiveWrapperOut->m_pFunctionPointersCpp = pFunctionPointersCpp;
      try {
         const std::vector<std::shared_ptr<const Registration>> registrations = (*registerObjectivesFunction)();
         const bool bFailed = Registration::CreateRegistrable(pConfig, sObjective, sObjectiveEnd, pObjectiveWrapperOut, registrations);
         if(!bFailed) {
            EBM_ASSERT(nullptr != pObjectiveWrapperOut->m_pObjective);
            pObjectiveWrapperOut->m_pApplyUpdateC = MAKE_ZONED_C_FUNCTION_NAME(ApplyUpdate);
#ifdef ZONE_cpu
            pObjectiveWrapperOut->m_pFinishMetricC = MAKE_ZONED_C_FUNCTION_NAME(FinishMetric);
            pObjectiveWrapperOut->m_pCheckTargetsC = MAKE_ZONED_C_FUNCTION_NAME(CheckTargets);
#else // ZONE_cpu
            pObjectiveWrapperOut->m_pFinishMetricC = nullptr;
            pObjectiveWrapperOut->m_pCheckTargetsC = nullptr;
#endif // ZONE_cpu

            LOG_0(Trace_Info, "Exited Objective::CreateObjective");
            return Error_None;
         }
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Info, "Exited Objective::CreateObjective unknown objective");
         error = Error_ObjectiveUnknown;
      } catch(const ParamValMalformedException &) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective ParamValMalformedException");
         error = Error_ObjectiveParamValMalformed;
      } catch(const ParamUnknownException &) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective ParamUnknownException");
         error = Error_ObjectiveParamUnknown;
      } catch(const RegistrationConstructorException &) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective RegistrationConstructorException");
         error = Error_ObjectiveConstructorException;
      } catch(const ParamValOutOfRangeException &) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective ParamValOutOfRangeException");
         error = Error_ObjectiveParamValOutOfRange;
      } catch(const ParamMismatchWithConfigException &) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective ParamMismatchWithConfigException");
         error = Error_ObjectiveParamMismatchWithConfig;
      } catch(const IllegalRegistrationNameException &) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective IllegalRegistrationNameException");
         error = Error_ObjectiveIllegalRegistrationName;
      } catch(const IllegalParamNameException &) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective IllegalParamNameException");
         error = Error_ObjectiveIllegalParamName;
      } catch(const DuplicateParamNameException &) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective DuplicateParamNameException");
         error = Error_ObjectiveDuplicateParamName;
      } catch(const NonPrivateRegistrationException &) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective NonPrivateRegistrationException");
         error = Error_ObjectiveNonPrivate;
      } catch(const NonPrivateParamException &) {
         EBM_ASSERT(nullptr == pObjectiveWrapperOut->m_pObjective);
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective NonPrivateParamException");
         error = Error_ObjectiveParamNonPrivate;
      } catch(const std::bad_alloc &) {
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective Out of Memory");
         error = Error_OutOfMemory;
      } catch(...) {
         LOG_0(Trace_Warning, "WARNING Objective::CreateObjective internal error, unknown exception");
         error = Error_UnexpectedInternal;
      }
      free(pObjectiveWrapperOut->m_pObjective); // this is legal if pObjectiveWrapper->m_pObjective is nullptr
      pObjectiveWrapperOut->m_pObjective = nullptr;

      free(pObjectiveWrapperOut->m_pFunctionPointersCpp); // this is legal if pObjectiveWrapper->m_pFunctionPointersCpp is nullptr
      pObjectiveWrapperOut->m_pFunctionPointersCpp = nullptr;
   }
   return error;
}

} // DEFINED_ZONE_NAME
