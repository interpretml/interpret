// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t
#include <vector>
#include <functional>
#include <memory>
#include <algorithm>

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureGroup.h"
#include "ThreadStateBoosting.h"

#include "Loss.h"

// Add any new Loss*.h include files here:
#include "LossPseudoHuber.h"
#include "LossBinaryLogLoss.h"
#include "LossMulticlassLogLoss.h"

// Add any new Loss* types to this list:
static const std::initializer_list<std::shared_ptr<const LossRegistrationBase>> RegisterLosses() {
   // IMPORTANT: the *LossParam types here must match the parameters types in your Loss* constructor
   return {
      LossRegistration<LossRegressionPseudoHuber>("pseudo_huber", FloatLossParam("delta", 1)),
      LossRegistration<LossMulticlassLogLoss>("log_loss"),
      LossRegistration<LossMulticlassLogLoss>("softmax")
   };
}

// !! ANYTHING BELOW THIS POINT ISN'T REQUIRED TO MAKE YOUR OWN CUSTOM LOSS FUNCTION !!

ErrorEbmType Loss::CreateLoss(
   const char * const sLoss, 
   const Config * const pConfig,
   const Loss ** const ppLoss
) noexcept {
   EBM_ASSERT(nullptr != sLoss);
   EBM_ASSERT(nullptr != pConfig);
   EBM_ASSERT(nullptr != ppLoss);

   LOG_0(TraceLevelInfo, "Entered Loss::CreateLoss");
   
   try {
      const std::initializer_list<std::shared_ptr<const LossRegistrationBase>> registeredLosses = RegisterLosses();
      for(const std::shared_ptr<const LossRegistrationBase> & lossRegistration : registeredLosses) {
         if(nullptr == lossRegistration) {
            // hopefully nobody inserts a nullptr, but check anyways
            LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss loss construction exception");
            return Error_LossConstructionException;
         }
         try {
            std::unique_ptr<const Loss> pLoss = lossRegistration->AttemptCreateLoss(*pConfig, sLoss);
            if(nullptr != pLoss) {
               // found it!
               LOG_0(TraceLevelInfo, "Exited Loss::CreateLoss");
               // we're exiting the area where exceptions are regularily thrown 
               *ppLoss = pLoss.release();
               return Error_None;
            }
         } catch(const SkipLossException &) {
            // the specific Loss function is saying it isn't a match (based on parameters in the Config object probably)
         }
      }
   } catch(const EbmException & exception) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss Exception");
      return exception.GetError();
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
