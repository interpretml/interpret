// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "InteractionCore.hpp"
#include "InteractionShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

void InteractionShell::Free(InteractionShell * const pInteractionShell) {
   LOG_0(TraceLevelInfo, "Entered InteractionShell::Free");

   if(nullptr != pInteractionShell) {
      free(pInteractionShell->m_aThreadByteBuffer1);
      InteractionCore::Free(pInteractionShell->m_pInteractionCore);
      
      // before we free our memory, indicate it was freed so if our higher level language attempts to use it we have
      // a chance to detect the error
      pInteractionShell->m_handleVerification = k_handleVerificationFreed;
      free(pInteractionShell);
   }

   LOG_0(TraceLevelInfo, "Exited InteractionShell::Free");
}

InteractionShell * InteractionShell::Create() {
   LOG_0(TraceLevelInfo, "Entered InteractionShell::Create");

   InteractionShell * const pNew = EbmMalloc<InteractionShell>();
   if(nullptr != pNew) {
      pNew->InitializeUnfailing();
   }

   LOG_0(TraceLevelInfo, "Exited InteractionShell::Create");

   return pNew;
}

HistogramBucketBase * InteractionShell::GetHistogramBucketBase(size_t cBytesRequired) {
   HistogramBucketBase * aBuffer = m_aThreadByteBuffer1;
   if(UNLIKELY(m_cThreadByteBufferCapacity1 < cBytesRequired)) {
      cBytesRequired <<= 1;
      m_cThreadByteBufferCapacity1 = cBytesRequired;
      LOG_N(TraceLevelInfo, "Growing InteractionShell::ThreadByteBuffer1 to %zu", cBytesRequired);

      free(aBuffer);
      aBuffer = static_cast<HistogramBucketBase *>(EbmMalloc<void>(cBytesRequired));
      m_aThreadByteBuffer1 = aBuffer; // store it before checking it incase it's null so that we don't free old memory
      if(nullptr == aBuffer) {
         LOG_0(TraceLevelWarning, "WARNING InteractionShell::GetHistogramBucketBase OutOfMemory");
      }
   }
   return aBuffer;
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CreateInteractionDetector(
   const void * dataSet,
   const BagEbmType * bag,
   const double * predictorScores, // only samples with non-zeros in the bag are included
   const double * optionalTempParams,
   InteractionHandle * interactionHandleOut
) {
   LOG_N(TraceLevelInfo, "Entered CreateInteractionDetector: "
      "dataSet=%p, "
      "bag=%p, "
      "predictorScores=%p, "
      "optionalTempParams=%p, "
      "interactionHandleOut=%p"
      ,
      static_cast<const void *>(dataSet),
      static_cast<const void *>(bag),
      static_cast<const void *>(predictorScores),
      static_cast<const void *>(optionalTempParams),
      static_cast<const void *>(interactionHandleOut)
   );

   ErrorEbmType error;

   if(nullptr == interactionHandleOut) {
      LOG_0(TraceLevelError, "ERROR CreateInteractionDetector nullptr == interactionHandleOut");
      return Error_IllegalParamValue;
   }
   *interactionHandleOut = nullptr; // set this to nullptr as soon as possible so the caller doesn't attempt to free it

   if(nullptr == dataSet) {
      LOG_0(TraceLevelError, "ERROR CreateInteractionDetector nullptr == dataSet");
      return Error_IllegalParamValue;
   }

   InteractionShell * const pInteractionShell = InteractionShell::Create();
   if(UNLIKELY(nullptr == pInteractionShell)) {
      LOG_0(TraceLevelWarning, "WARNING CreateInteractionDetector nullptr == pInteractionShell");
      return Error_OutOfMemory;
   }

   error = InteractionCore::Create(
      pInteractionShell,
      static_cast<const unsigned char *>(dataSet),
      bag,
      predictorScores,
      optionalTempParams
   );
   if(Error_None != error) {
      InteractionShell::Free(pInteractionShell);
      return error;
   }

   const InteractionHandle handle = pInteractionShell->GetHandle();

   LOG_N(TraceLevelInfo, "Exited CreateInteractionDetector: *interactionHandleOut=%p", static_cast<void *>(handle));

   *interactionHandleOut = handle;
   return Error_None;
}

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION FreeInteractionDetector(
   InteractionHandle interactionHandle
) {
   LOG_N(TraceLevelInfo, "Entered FreeInteractionDetector: interactionHandle=%p", static_cast<void *>(interactionHandle));

   InteractionShell * const pInteractionShell = InteractionShell::GetInteractionShellFromHandle(interactionHandle);
   // if the conversion above doesn't work, it'll return null, and our free will not in fact free any memory,
   // but it will not crash. We'll leak memory, but at least we'll log that.

   // it's legal to call free on nullptr, just like for free().  This is checked inside InteractionCore::Free()
   InteractionShell::Free(pInteractionShell);

   LOG_0(TraceLevelInfo, "Exited FreeInteractionDetector");
}

} // DEFINED_ZONE_NAME
