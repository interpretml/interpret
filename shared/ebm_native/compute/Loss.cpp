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
   const void ** const ppLossOut
) noexcept {
   EBM_ASSERT(nullptr != registerLossesFunction);
   EBM_ASSERT(1 <= cOutputs);
   EBM_ASSERT(nullptr != sLoss);
   EBM_ASSERT(nullptr != ppLossOut);

   // we only have 1 loss per string, unlike metrics
   const char * const sLossEnd = sLoss + strlen(sLoss);
   // maybe backtrack from the end until the first non-whitespace, and also move forward until the first non-
   // whitespace.  That'll make it easier to create exit conditions since we will know if we hit sLossEnd that
   // we don't need to explore the trailing characters
   UNUSED(sLossEnd);

   LOG_0(TraceLevelInfo, "Entered Loss::CreateLoss");
   try {
      Config config(cOutputs);
      const std::vector<std::shared_ptr<const Registration>> registrations = (*registerLossesFunction)();
      std::vector<std::unique_ptr<const Registrable>> registrables = 
         Registration::CreateRegistrables(config, sLoss, registrations);
      if(registrables.size() < 1) {
         LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss empty loss string");
         return Error_LossUnknown;
      }
      if(1 != registrables.size()) {
         LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss multiple loss functions can't simultaneously exist");
         return Error_LossMultipleSpecified;
      }
      *ppLossOut = registrables[0].release();
      LOG_0(TraceLevelInfo, "Exited Loss::CreateLoss");
      return Error_None;
   } catch(const RegistrationUnknownException &) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss RegistrationUnknownException");
      return Error_LossUnknown;
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
