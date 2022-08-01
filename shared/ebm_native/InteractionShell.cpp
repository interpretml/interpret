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
   LOG_0(Trace_Info, "Entered InteractionShell::Free");

   if(nullptr != pInteractionShell) {
      free(pInteractionShell->m_aThreadByteBuffer1Fast);
      free(pInteractionShell->m_aThreadByteBuffer1Big);
      InteractionCore::Free(pInteractionShell->m_pInteractionCore);
      
      // before we free our memory, indicate it was freed so if our higher level language attempts to use it we have
      // a chance to detect the error
      pInteractionShell->m_handleVerification = k_handleVerificationFreed;
      free(pInteractionShell);
   }

   LOG_0(Trace_Info, "Exited InteractionShell::Free");
}

InteractionShell * InteractionShell::Create() {
   LOG_0(Trace_Info, "Entered InteractionShell::Create");

   InteractionShell * const pNew = EbmMalloc<InteractionShell>();
   if(nullptr != pNew) {
      pNew->InitializeUnfailing();
   }

   LOG_0(Trace_Info, "Exited InteractionShell::Create");

   return pNew;
}

BinBase * InteractionShell::GetBinBaseFast(size_t cBytesRequired) {
   BinBase * aBuffer = m_aThreadByteBuffer1Fast;
   if(UNLIKELY(m_cThreadByteBufferCapacity1Fast < cBytesRequired)) {
      cBytesRequired <<= 1;
      m_cThreadByteBufferCapacity1Fast = cBytesRequired;
      LOG_N(Trace_Info, "Growing InteractionShell::ThreadByteBuffer1Fast to %zu", cBytesRequired);

      free(aBuffer);
      aBuffer = static_cast<BinBase *>(EbmMalloc<void>(cBytesRequired));
      m_aThreadByteBuffer1Fast = aBuffer; // store it before checking it incase it's null so that we don't free old memory
      if(nullptr == aBuffer) {
         LOG_0(Trace_Warning, "WARNING InteractionShell::GetBinBaseFast OutOfMemory");
      }
   }
   return aBuffer;
}

BinBase * InteractionShell::GetBinBaseBig(size_t cBytesRequired) {
   BinBase * aBuffer = m_aThreadByteBuffer1Big;
   if(UNLIKELY(m_cThreadByteBufferCapacity1Big < cBytesRequired)) {
      cBytesRequired <<= 1;
      m_cThreadByteBufferCapacity1Big = cBytesRequired;
      LOG_N(Trace_Info, "Growing InteractionShell::ThreadByteBuffer1Big to %zu", cBytesRequired);

      free(aBuffer);
      aBuffer = static_cast<BinBase *>(EbmMalloc<void>(cBytesRequired));
      m_aThreadByteBuffer1Big = aBuffer; // store it before checking it incase it's null so that we don't free old memory
      if(nullptr == aBuffer) {
         LOG_0(Trace_Warning, "WARNING InteractionShell::GetBinBaseBig OutOfMemory");
      }
   }
   return aBuffer;
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION CreateInteractionDetector(
   const void * dataSet,
   const BagEbm * bag,
   const double * initScores, // only samples with non-zeros in the bag are included
   const double * experimentalParams,
   InteractionHandle * interactionHandleOut
) {
   LOG_N(Trace_Info, "Entered CreateInteractionDetector: "
      "dataSet=%p, "
      "bag=%p, "
      "initScores=%p, "
      "experimentalParams=%p, "
      "interactionHandleOut=%p"
      ,
      static_cast<const void *>(dataSet),
      static_cast<const void *>(bag),
      static_cast<const void *>(initScores),
      static_cast<const void *>(experimentalParams),
      static_cast<const void *>(interactionHandleOut)
   );

   ErrorEbm error;

   if(nullptr == interactionHandleOut) {
      LOG_0(Trace_Error, "ERROR CreateInteractionDetector nullptr == interactionHandleOut");
      return Error_IllegalParamVal;
   }
   *interactionHandleOut = nullptr; // set this to nullptr as soon as possible so the caller doesn't attempt to free it

   if(nullptr == dataSet) {
      LOG_0(Trace_Error, "ERROR CreateInteractionDetector nullptr == dataSet");
      return Error_IllegalParamVal;
   }

   InteractionShell * const pInteractionShell = InteractionShell::Create();
   if(UNLIKELY(nullptr == pInteractionShell)) {
      LOG_0(Trace_Warning, "WARNING CreateInteractionDetector nullptr == pInteractionShell");
      return Error_OutOfMemory;
   }

   error = InteractionCore::Create(
      pInteractionShell,
      static_cast<const unsigned char *>(dataSet),
      bag,
      initScores,
      experimentalParams
   );
   if(Error_None != error) {
      InteractionShell::Free(pInteractionShell);
      return error;
   }

   const InteractionHandle handle = pInteractionShell->GetHandle();

   LOG_N(Trace_Info, "Exited CreateInteractionDetector: *interactionHandleOut=%p", static_cast<void *>(handle));

   *interactionHandleOut = handle;
   return Error_None;
}

EBM_API_BODY void EBM_CALLING_CONVENTION FreeInteractionDetector(
   InteractionHandle interactionHandle
) {
   LOG_N(Trace_Info, "Entered FreeInteractionDetector: interactionHandle=%p", static_cast<void *>(interactionHandle));

   InteractionShell * const pInteractionShell = InteractionShell::GetInteractionShellFromHandle(interactionHandle);
   // if the conversion above doesn't work, it'll return null, and our free will not in fact free any memory,
   // but it will not crash. We'll leak memory, but at least we'll log that.

   // it's legal to call free on nullptr, just like for free().  This is checked inside InteractionCore::Free()
   InteractionShell::Free(pInteractionShell);

   LOG_0(Trace_Info, "Exited FreeInteractionDetector");
}

} // DEFINED_ZONE_NAME
