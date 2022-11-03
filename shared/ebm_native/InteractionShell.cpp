// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "common_cpp.hpp"
#include "bridge_cpp.hpp"

#include "InteractionCore.hpp"
#include "InteractionShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern void InitializeMSEGradientsAndHessians(
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const BagEbm * const aBag,
   const double * const aInitScores,
   const size_t cSetSamples,
   FloatFast * const aGradientAndHessian,
   const FloatFast * const aWeight
);

void InteractionShell::Free(InteractionShell * const pInteractionShell) {
   LOG_0(Trace_Info, "Entered InteractionShell::Free");

   if(nullptr != pInteractionShell) {
      free(pInteractionShell->m_aInteractionFastBinsTemp);
      free(pInteractionShell->m_aInteractionBigBins);
      InteractionCore::Free(pInteractionShell->m_pInteractionCore);
      
      // before we free our memory, indicate it was freed so if our higher level language attempts to use it we have
      // a chance to detect the error
      pInteractionShell->m_handleVerification = k_handleVerificationFreed;
      free(pInteractionShell);
   }

   LOG_0(Trace_Info, "Exited InteractionShell::Free");
}

InteractionShell * InteractionShell::Create(InteractionCore * const pInteractionCore) {
   LOG_0(Trace_Info, "Entered InteractionShell::Create");

   InteractionShell * const pNew = static_cast<InteractionShell *>(malloc(sizeof(InteractionShell)));
   if(UNLIKELY(nullptr == pNew)) {
      LOG_0(Trace_Error, "ERROR InteractionShell::Create nullptr == pNew");
      return nullptr;
   }

   pNew->InitializeUnfailing(pInteractionCore);

   LOG_0(Trace_Info, "Exited InteractionShell::Create");

   return pNew;
}

BinBase * InteractionShell::GetInteractionFastBinsTemp(const size_t cBytesPerFastBin, const size_t cFastBins) {
   ANALYSIS_ASSERT(0 != cBytesPerFastBin);

   BinBase * aBuffer = m_aInteractionFastBinsTemp;
   if(UNLIKELY(m_cAllocatedFastBins < cFastBins)) {
      free(aBuffer);
      m_aInteractionFastBinsTemp = nullptr;

      const size_t cItemsGrowth = (cFastBins >> 2) + 16; // cannot overflow
      if(IsAddError(cItemsGrowth, cFastBins)) {
         LOG_0(Trace_Warning, "WARNING InteractionShell::GetInteractionFastBinsTemp IsAddError(cItemsGrowth, cFastBins)");
         return nullptr;
      }
      const size_t cNewAllocatedFastBins = cFastBins + cItemsGrowth;

      m_cAllocatedFastBins = cNewAllocatedFastBins;
      LOG_N(Trace_Info, "Growing Interaction fast bins to %zu", cNewAllocatedFastBins);

      if(IsMultiplyError(cBytesPerFastBin, cNewAllocatedFastBins)) {
         LOG_0(Trace_Warning, "WARNING InteractionShell::GetInteractionFastBinsTemp IsMultiplyError(cBytesPerFastBin, cNewAllocatedFastBins)");
         return nullptr;
      }
      aBuffer = static_cast<BinBase *>(malloc(cBytesPerFastBin * cNewAllocatedFastBins));
      if(nullptr == aBuffer) {
         LOG_0(Trace_Warning, "WARNING InteractionShell::GetInteractionFastBinsTemp OutOfMemory");
         return nullptr;
      }
      m_aInteractionFastBinsTemp = aBuffer;
   }
   return aBuffer;
}

BinBase * InteractionShell::GetInteractionBigBins(const size_t cBytesPerBigBin, const size_t cBigBins) {
   ANALYSIS_ASSERT(0 != cBytesPerBigBin);

   BinBase * aBuffer = m_aInteractionBigBins;
   if(UNLIKELY(m_cAllocatedBigBins < cBigBins)) {
      free(aBuffer);
      m_aInteractionBigBins = nullptr;

      const size_t cItemsGrowth = (cBigBins >> 2) + 16; // cannot overflow
      if(IsAddError(cItemsGrowth, cBigBins)) {
         LOG_0(Trace_Warning, "WARNING InteractionShell::GetInteractionBigBins IsAddError(cItemsGrowth, cBigBins)");
         return nullptr;
      }
      const size_t cNewAllocatedBigBins = cBigBins + cItemsGrowth;

      m_cAllocatedBigBins = cNewAllocatedBigBins;
      LOG_N(Trace_Info, "Growing Interaction big bins to %zu", cNewAllocatedBigBins);

      if(IsMultiplyError(cBytesPerBigBin, cNewAllocatedBigBins)) {
         LOG_0(Trace_Warning, "WARNING InteractionShell::GetInteractionBigBins IsMultiplyError(cBytesPerBigBin, cNewAllocatedBigBins)");
         return nullptr;
      }
      aBuffer = static_cast<BinBase *>(malloc(cBytesPerBigBin * cNewAllocatedBigBins));
      if(nullptr == aBuffer) {
         LOG_0(Trace_Warning, "WARNING InteractionShell::GetInteractionBigBins OutOfMemory");
         return nullptr;
      }
      m_aInteractionBigBins = aBuffer;
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

   InteractionCore * pInteractionCore = nullptr;
   error = InteractionCore::Create(
      static_cast<const unsigned char *>(dataSet),
      bag,
      experimentalParams,
      &pInteractionCore
   );
   if(Error_None != error) {
      // legal to call if nullptr. On error we can get back a legal pInteractionCore to delete
      InteractionCore::Free(pInteractionCore);
      return error;
   }

   InteractionShell * const pInteractionShell = InteractionShell::Create(pInteractionCore);
   if(UNLIKELY(nullptr == pInteractionShell)) {
      // if the memory allocation for pInteractionShell failed then 
      // there was no place to put the pInteractionCore, so free it
      InteractionCore::Free(pInteractionCore);
      return Error_OutOfMemory;
   }

   const ptrdiff_t cClasses = pInteractionCore->GetCountClasses();
   if(IsClassification(cClasses)) {
      error = pInteractionCore->InitializeInteractionGradientsAndHessians(
         static_cast<const unsigned char *>(dataSet),
         bag,
         initScores
      );
      if(Error_None != error) {
         InteractionCore::Free(pInteractionCore);
         return error;
      }
   } else {
      if(!pInteractionCore->GetDataSetInteraction()->IsGradientsAndHessiansNull()) {
         InitializeMSEGradientsAndHessians(
            static_cast<const unsigned char *>(dataSet),
            BagEbm { 1 },
            bag,
            initScores,
            pInteractionCore->GetDataSetInteraction()->GetCountSamples(),
            pInteractionCore->GetDataSetInteraction()->GetGradientsAndHessiansPointer(),
            pInteractionCore->GetDataSetInteraction()->GetWeights()
         );
      }
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
