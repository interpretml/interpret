// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureGroup.h"
#include "ThreadStateBoosting.h"

#include "Loss.h"

// Add any new Loss*.h include files here:
#include "LossPseudoHuber.h"
#include "LossMulticlassSoftmax.h"

// Add any new Loss*::AttemptCreateLoss functions to this list:
static const ATTEMPT_CREATE_LOSS k_registeredLosss[] = {
   LossPseudoHuber::AttemptCreateLoss,
   LossMulticlassSoftmax::AttemptCreateLoss
};

// TODO: simplify this registration process like this:
//
// global for all losses
//struct LossParam {
//   const char * name; // I consume this so won't complicate things for the user by having a non std::string
//   FloatEbmType defaultValue;
//}
//
//// still need to do this in Loss.cpp!
//#include "LossPseudoHuber.h"
//static const LossRegistration k_registeredLosss[] = {
//   {
//      LossRegistration("pseudo_huber", LossPseudoHuber::CreateLoss),
//      LossRegistration("pseudo_huber2", LossPseudoHuber2::CreateLoss, { LossParam("param1", 1.2), LossParam("param2", 1.2), }),
//   }
//
//   // in your loss include file:
//   static Loss * LossPseudoHuber::CreateLoss(
//      std::vector<FloatEbmType> params,
//      size_t countTargetClasses (or replace someday with a "Configure" object that can hold whatever
//   )
//




ErrorEbmType Loss::CreateLoss(
   const char * const sLoss, 
   const size_t cTargetClasses, 
   const Loss ** const ppLoss
) noexcept {
   EBM_ASSERT(nullptr != sLoss);
   EBM_ASSERT(nullptr != ppLoss);

   LOG_0(TraceLevelInfo, "Entered Loss::CreateLoss");

   *ppLoss = nullptr;
   try {
      const ATTEMPT_CREATE_LOSS * pAttemptCreateLossCur = k_registeredLosss;
      const ATTEMPT_CREATE_LOSS * const pAttemptCreateLossEnd = 
         &k_registeredLosss[sizeof(k_registeredLosss) / sizeof(k_registeredLosss[0])];
      while(pAttemptCreateLossEnd != pAttemptCreateLossCur) {
         const ErrorEbmType error = (*pAttemptCreateLossCur)(sLoss, cTargetClasses, ppLoss);
         if(Error_None != error) {
            EBM_ASSERT(nullptr == *ppLoss);
            LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss error in AttemptCreateLoss");
            return error;
         }
         if(nullptr != *ppLoss) {
            LOG_0(TraceLevelInfo, "Exited Loss::CreateLoss");
            return Error_None;
         }
         ++pAttemptCreateLossCur;
      }
   } catch(const std::bad_alloc&) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss Out of Memory");
      return Error_OutOfMemory;
   } catch(...) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss loss construction exception");
      return Error_LossConstructionException;
   }
   LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss loss unknown");
   return Error_LossUnknown;
}

FloatEbmType Loss::GetUpdateMultiple() const {
   return FloatEbmType { 1 };
}
