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

#include "EbmException.hpp"
#include "Config.hpp"
#include "Registrable.hpp"
#include "Registration.hpp"
#include "Loss.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

ErrorEbmType Loss::CreateLoss(
   const REGISTER_LOSSES_FUNCTION registerLossesFunction,
   const size_t cOutputs,
   const char * const sLoss,
   const char * const sLossEnd,
   const void ** const ppLossOut
) noexcept {
   EBM_ASSERT(nullptr != registerLossesFunction);
   EBM_ASSERT(1 <= cOutputs);
   EBM_ASSERT(nullptr != sLoss);
   EBM_ASSERT(nullptr != sLossEnd);
   EBM_ASSERT(sLoss < sLossEnd); // empty string not allowed
   EBM_ASSERT('\0' != *sLoss);
   EBM_ASSERT(!(0x20 == *sLoss || (0x9 <= *sLoss && *sLoss <= 0xd)));
   EBM_ASSERT(!(0x20 == *(sLossEnd - 1) || (0x9 <= *(sLossEnd - 1) && *(sLossEnd - 1) <= 0xd)));
   EBM_ASSERT('\0' == *sLossEnd || 0x20 == *sLossEnd || (0x9 <= *sLossEnd && *sLossEnd <= 0xd));
   EBM_ASSERT(nullptr != ppLossOut);

   LOG_0(TraceLevelInfo, "Entered Loss::CreateLoss");
   try {
      Config config(cOutputs);
      const std::vector<std::shared_ptr<const Registration>> registrations = (*registerLossesFunction)();
      std::unique_ptr<const Registrable> pRegistrable = 
         Registration::CreateRegistrable(config, sLoss, sLossEnd, registrations);
      if(nullptr == pRegistrable) {
         return Error_LossUnknown;
      }
      *ppLossOut = pRegistrable.release();
      LOG_0(TraceLevelInfo, "Exited Loss::CreateLoss");
      return Error_None;
   } catch(const ParameterValueMalformedException &) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss ParameterValueMalformedException");
      return Error_LossParameterValueMalformed;
   } catch(const ParameterUnknownException &) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss ParameterUnknownException");
      return Error_LossParameterUnknown;
   } catch(const RegistrationConstructorException &) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss RegistrationConstructorException");
      return Error_LossConstructorException;
   } catch(const ParameterValueOutOfRangeException &) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss ParameterValueOutOfRangeException");
      return Error_LossParameterValueOutOfRange;
   } catch(const ParameterMismatchWithConfigException &) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss ParameterMismatchWithConfigException");
      return Error_LossParameterMismatchWithConfig;
   } catch(const IllegalRegistrationNameException &) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss IllegalRegistrationNameException");
      return Error_LossIllegalRegistrationName;
   } catch(const IllegalParamNameException &) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss IllegalParamNameException");
      return Error_LossIllegalParamName;
   } catch(const DuplicateParamNameException &) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss DuplicateParamNameException");
      return Error_LossDuplicateParamName;
   } catch(const EbmException & exception) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss EbmException");
      return exception.GetError();
   } catch(const std::bad_alloc &) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss Out of Memory");
      return Error_OutOfMemory;
   } catch(...) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss internal error, unknown exception");
      return Error_UnknownInternalError;
   }
}

FloatEbmType Loss::GetUpdateMultiple() const {
   return FloatEbmType { 1 };
}

bool Loss::IsSuperSuperSpecialLossWhereTargetNotNeededOnlyMseLossQualifies() const {
   return false;
}

} // DEFINED_ZONE_NAME
