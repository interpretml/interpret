// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t
#include <memory> // shared_ptr, unique_ptr
#include <vector>

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "zoned_bridge_c_functions.h"
#include "zoned_bridge_cpp_functions.hpp"
#include "EbmException.hpp"
#include "Registrable.hpp"
#include "Registration.hpp"
#include "Loss.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

ErrorEbmType Loss::CreateLoss(
   const REGISTER_LOSSES_FUNCTION registerLossesFunction,
   const Config * const pConfig,
   const char * const sLoss,
   const char * const sLossEnd,
   LossWrapper * const pLossWrapperOut
) noexcept {
   EBM_ASSERT(nullptr != registerLossesFunction);
   EBM_ASSERT(nullptr != pConfig);
   EBM_ASSERT(1 <= pConfig->cOutputs);
   EBM_ASSERT(nullptr != sLoss);
   EBM_ASSERT(nullptr != sLossEnd);
   EBM_ASSERT(sLoss < sLossEnd); // empty string not allowed
   EBM_ASSERT('\0' != *sLoss);
   EBM_ASSERT(!(0x20 == *sLoss || (0x9 <= *sLoss && *sLoss <= 0xd)));
   EBM_ASSERT(!(0x20 == *(sLossEnd - 1) || (0x9 <= *(sLossEnd - 1) && *(sLossEnd - 1) <= 0xd)));
   EBM_ASSERT('\0' == *sLossEnd || 0x20 == *sLossEnd || (0x9 <= *sLossEnd && *sLossEnd <= 0xd));
   EBM_ASSERT(nullptr != pLossWrapperOut);
   EBM_ASSERT(nullptr == pLossWrapperOut->m_pLoss);

   LOG_0(TraceLevelInfo, "Entered Loss::CreateLoss");

   ErrorEbmType error;
   try {
      const std::vector<std::shared_ptr<const Registration>> registrations = (*registerLossesFunction)();
      const bool bFailed = Registration::CreateRegistrable(pConfig, sLoss, sLossEnd, pLossWrapperOut, registrations);
      if(!bFailed) {
         EBM_ASSERT(nullptr != pLossWrapperOut->m_pLoss);
         pLossWrapperOut->m_pApplyTrainingC = MAKE_ZONED_C_FUNCTION_NAME(ApplyTraining);
         pLossWrapperOut->m_pApplyValidationC = MAKE_ZONED_C_FUNCTION_NAME(ApplyValidation);
         LOG_0(TraceLevelInfo, "Exited Loss::CreateLoss");
         return Error_None;
      }
      EBM_ASSERT(nullptr == pLossWrapperOut->m_pLoss);
      LOG_0(TraceLevelInfo, "Exited Loss::CreateLoss unknown loss");
      error = Error_LossUnknown;
   } catch(const ParameterValueMalformedException &) {
      EBM_ASSERT(nullptr == pLossWrapperOut->m_pLoss);
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss ParameterValueMalformedException");
      error = Error_LossParameterValueMalformed;
   } catch(const ParameterUnknownException &) {
      EBM_ASSERT(nullptr == pLossWrapperOut->m_pLoss);
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss ParameterUnknownException");
      error = Error_LossParameterUnknown;
   } catch(const RegistrationConstructorException &) {
      EBM_ASSERT(nullptr == pLossWrapperOut->m_pLoss);
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss RegistrationConstructorException");
      error = Error_LossConstructorException;
   } catch(const ParameterValueOutOfRangeException &) {
      EBM_ASSERT(nullptr == pLossWrapperOut->m_pLoss);
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss ParameterValueOutOfRangeException");
      error = Error_LossParameterValueOutOfRange;
   } catch(const ParameterMismatchWithConfigException &) {
      EBM_ASSERT(nullptr == pLossWrapperOut->m_pLoss);
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss ParameterMismatchWithConfigException");
      error = Error_LossParameterMismatchWithConfig;
   } catch(const IllegalRegistrationNameException &) {
      EBM_ASSERT(nullptr == pLossWrapperOut->m_pLoss);
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss IllegalRegistrationNameException");
      error = Error_LossIllegalRegistrationName;
   } catch(const IllegalParamNameException &) {
      EBM_ASSERT(nullptr == pLossWrapperOut->m_pLoss);
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss IllegalParamNameException");
      error = Error_LossIllegalParamName;
   } catch(const DuplicateParamNameException &) {
      EBM_ASSERT(nullptr == pLossWrapperOut->m_pLoss);
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss DuplicateParamNameException");
      error = Error_LossDuplicateParamName;
   } catch(const EbmException & exception) {
      EBM_ASSERT(nullptr == pLossWrapperOut->m_pLoss);
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss EbmException");
      error = exception.GetError();
   } catch(const std::bad_alloc &) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss Out of Memory");
      error = Error_OutOfMemory;
   } catch(...) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss internal error, unknown exception");
      error = Error_UnknownInternalError;
   }

   free(const_cast<void *>(pLossWrapperOut->m_pLoss)); // this is legal if pLossWrapper->m_pLoss is nullptr
   pLossWrapperOut->m_pLoss = nullptr;

   return error;
}

} // DEFINED_ZONE_NAME
