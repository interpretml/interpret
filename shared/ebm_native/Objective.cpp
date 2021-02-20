// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureGroup.h"
#include "ThreadStateBoosting.h"

#include "Objective.h"

// Add any new Objective*.h include files here:
#include "ObjectivePseudoHuber.h"

// Add any new Objective*::AttemptCreateObjective functions to this list:
static const ATTEMPT_CREATE_OBJECTIVE k_registeredObjectives[] = {
   ObjectivePseudoHuber::AttemptCreateObjective
};

ErrorEbmType Objective::CreateObjective(
   const char * const sObjective, 
   const size_t cTargetClasses, 
   const Objective ** const ppObjective
) noexcept {
   EBM_ASSERT(nullptr != sObjective);
   EBM_ASSERT(nullptr != ppObjective);

   LOG_0(TraceLevelInfo, "Entered Objective::CreateObjective");

   *ppObjective = nullptr;
   try {
      const ATTEMPT_CREATE_OBJECTIVE * pAttemptCreateObjectiveCur = k_registeredObjectives;
      const ATTEMPT_CREATE_OBJECTIVE * const pAttemptCreateObjectiveEnd = 
         &k_registeredObjectives[sizeof(k_registeredObjectives) / sizeof(k_registeredObjectives[0])];
      while(pAttemptCreateObjectiveEnd != pAttemptCreateObjectiveCur) {
         const ErrorEbmType error = (*pAttemptCreateObjectiveCur)(sObjective, cTargetClasses, ppObjective);
         if(Error_None != error) {
            EBM_ASSERT(nullptr == *ppObjective);
            LOG_0(TraceLevelWarning, "WARNING Objective::CreateObjective error in AttemptCreateObjective");
            return error;
         }
         if(nullptr != *ppObjective) {
            LOG_0(TraceLevelInfo, "Exited Objective::CreateObjective");
            return Error_None;
         }
         ++pAttemptCreateObjectiveCur;
      }
   } catch(const std::bad_alloc&) {
      LOG_0(TraceLevelWarning, "WARNING Objective::CreateObjective Out of Memory");
      return Error_OutOfMemory;
   } catch(...) {
      LOG_0(TraceLevelWarning, "WARNING Objective::CreateObjective objective construction exception");
      return Error_ObjectiveConstructionException;
   }
   LOG_0(TraceLevelWarning, "WARNING Objective::CreateObjective objective unknown");
   return Error_ObjectiveUnknown;
}
