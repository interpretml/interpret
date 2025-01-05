// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#define ZONE_main
#include "zones.h"

#include "common.hpp"
#include "bridge.hpp"

#include "dataset_shared.hpp" // GetDataSetSharedHeader
#include "InteractionCore.hpp"
#include "InteractionShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern void InitializeRmseGradientsAndHessiansInteraction(const unsigned char* const pDataSetShared,
      const size_t cWeights,
      const double intercept,
      const BagEbm* const aBag,
      const double* const aInitScores,
      DataSetInteraction* const pDataSet);

void InteractionShell::Free(InteractionShell* const pInteractionShell) {
   LOG_0(Trace_Info, "Entered InteractionShell::Free");

   if(nullptr != pInteractionShell) {
      AlignedFree(pInteractionShell->m_aInteractionFastBinsTemp);
      AlignedFree(pInteractionShell->m_aInteractionMainBins);
      InteractionCore::Free(pInteractionShell->m_pInteractionCore);

      // before we free our memory, indicate it was freed so if our higher level language attempts to use it we have
      // a chance to detect the error
      pInteractionShell->m_handleVerification = k_handleVerificationFreed;
      free(pInteractionShell);
   }

   LOG_0(Trace_Info, "Exited InteractionShell::Free");
}

InteractionShell* InteractionShell::Create(InteractionCore* const pInteractionCore) {
   LOG_0(Trace_Info, "Entered InteractionShell::Create");

   InteractionShell* const pNew = static_cast<InteractionShell*>(malloc(sizeof(InteractionShell)));
   if(UNLIKELY(nullptr == pNew)) {
      LOG_0(Trace_Error, "ERROR InteractionShell::Create nullptr == pNew");
      return nullptr;
   }

   pNew->InitializeUnfailing(pInteractionCore);

   LOG_0(Trace_Info, "Exited InteractionShell::Create");

   return pNew;
}

BinBase* InteractionShell::GetInteractionFastBinsTemp(const size_t cBytes) {
   const ErrorEbm error =
         AlignedGrow(reinterpret_cast<void**>(&m_aInteractionFastBinsTemp), &m_cBytesFastBins, cBytes, EBM_FALSE);
   if(Error_None != error) {
      LOG_0(Trace_Warning, "WARNING InteractionShell::GetInteractionFastBinsTemp AlignedGrow failed");
      return nullptr;
   }
   return m_aInteractionFastBinsTemp;
}

BinBase* InteractionShell::GetInteractionMainBins(const size_t cBytesPerMainBin, const size_t cMainBins) {
   if(IsMultiplyError(cBytesPerMainBin, cMainBins)) {
      LOG_0(Trace_Warning,
            "WARNING InteractionShell::GetInteractionMainBins IsMultiplyError(cBytesPerMainBin, cMainBins)");
      return nullptr;
   }
   const size_t cBytes = cBytesPerMainBin * cMainBins;
   const ErrorEbm error =
         AlignedGrow(reinterpret_cast<void**>(&m_aInteractionMainBins), &m_cAllocatedMainBinBytes, cBytes, EBM_FALSE);
   if(Error_None != error) {
      LOG_0(Trace_Warning, "WARNING InteractionShell::GetInteractionMainBins AlignedGrow failed");
      return nullptr;
   }
   return m_aInteractionMainBins;
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION CreateInteractionDetector(const void* dataSet,
      const double* intercept,
      const BagEbm* bag,
      const double* initScores, // only samples with non-zeros in the bag are included
      CreateInteractionFlags flags,
      AccelerationFlags acceleration,
      const char* objective,
      const double* experimentalParams,
      InteractionHandle* interactionHandleOut) {
   LOG_N(Trace_Info,
         "Entered CreateInteractionDetector: "
         "dataSet=%p, "
         "intercept=%p, "
         "bag=%p, "
         "initScores=%p, "
         "flags=0x%" UCreateInteractionFlagsPrintf ", "
         "acceleration=0x%" UAccelerationFlagsPrintf ", "
         "objective=%p, "
         "experimentalParams=%p, "
         "interactionHandleOut=%p",
         static_cast<const void*>(dataSet),
         static_cast<const void*>(intercept),
         static_cast<const void*>(bag),
         static_cast<const void*>(initScores),
         static_cast<UCreateInteractionFlags>(flags), // signed to unsigned conversion is defined behavior in C++
         static_cast<UAccelerationFlags>(acceleration), // signed to unsigned conversion is defined behavior in C++
         static_cast<const void*>(objective), // do not print the string for security reasons
         static_cast<const void*>(experimentalParams),
         static_cast<const void*>(interactionHandleOut));

   ErrorEbm error;

   if(nullptr == interactionHandleOut) {
      LOG_0(Trace_Error, "ERROR CreateInteractionDetector nullptr == interactionHandleOut");
      return Error_IllegalParamVal;
   }
   *interactionHandleOut = nullptr; // set this to nullptr as soon as possible so the caller doesn't attempt to free it

   if(flags &
         ~(CreateInteractionFlags_DifferentialPrivacy | CreateInteractionFlags_UseApprox |
               CreateInteractionFlags_BinaryAsMulticlass)) {
      LOG_0(Trace_Error, "ERROR CreateInteractionDetector flags contains unknown flags. Ignoring extras.");
   }

   if(nullptr == dataSet) {
      LOG_0(Trace_Error, "ERROR CreateInteractionDetector nullptr == dataSet");
      return Error_IllegalParamVal;
   }

   UIntShared countSamples;
   size_t cFeatures;
   size_t cWeights;
   size_t cTargets;
   error = GetDataSetSharedHeader(
         static_cast<const unsigned char*>(dataSet), &countSamples, &cFeatures, &cWeights, &cTargets);
   if(Error_None != error) {
      // already logged
      return error;
   }

   if(IsConvertError<size_t>(countSamples)) {
      LOG_0(Trace_Error, "ERROR CreateInteractionDetector IsConvertError<size_t>(countSamples)");
      return Error_IllegalParamVal;
   }
   size_t cSamples = static_cast<size_t>(countSamples);

   if(size_t{1} < cWeights) {
      LOG_0(Trace_Warning, "WARNING CreateInteractionDetector size_t { 1 } < cWeights");
      return Error_IllegalParamVal;
   }
   if(size_t{1} != cTargets) {
      LOG_0(Trace_Warning, "WARNING CreateInteractionDetector 1 != cTargets");
      return Error_IllegalParamVal;
   }

   InteractionCore* pInteractionCore = nullptr;
   error = InteractionCore::Create(static_cast<const unsigned char*>(dataSet),
         cSamples,
         cFeatures,
         cWeights,
         bag,
         flags,
         acceleration,
         objective,
         experimentalParams,
         &pInteractionCore);
   if(Error_None != error) {
      // legal to call if nullptr. On error we can get back a legal pInteractionCore to delete
      InteractionCore::Free(pInteractionCore);
      return error;
   }

   InteractionShell* const pInteractionShell = InteractionShell::Create(pInteractionCore);
   if(UNLIKELY(nullptr == pInteractionShell)) {
      // if the memory allocation for pInteractionShell failed then
      // there was no place to put the pInteractionCore, so free it
      InteractionCore::Free(pInteractionCore);
      return Error_OutOfMemory;
   }

   if(size_t{0} != pInteractionCore->GetCountScores()) {
      if(!pInteractionCore->IsRmse()) {
         error = pInteractionCore->InitializeInteractionGradientsAndHessians(
               static_cast<const unsigned char*>(dataSet), cWeights, intercept, bag, initScores);
         if(Error_None != error) {
            // DO NOT FREE pInteractionCore since it's owned by pInteractionShell, which we free here
            InteractionShell::Free(pInteractionShell);
            return error;
         }
      } else {
         InitializeRmseGradientsAndHessiansInteraction(static_cast<const unsigned char*>(dataSet),
               cWeights,
               nullptr == intercept ? 0.0 : *intercept,
               bag,
               initScores,
               pInteractionCore->GetDataSetInteraction());
      }
   }

   const InteractionHandle handle = pInteractionShell->GetHandle();

   LOG_N(Trace_Info, "Exited CreateInteractionDetector: *interactionHandleOut=%p", static_cast<void*>(handle));

   *interactionHandleOut = handle;
   return Error_None;
}

EBM_API_BODY void EBM_CALLING_CONVENTION FreeInteractionDetector(InteractionHandle interactionHandle) {
   LOG_N(Trace_Info, "Entered FreeInteractionDetector: interactionHandle=%p", static_cast<void*>(interactionHandle));

   InteractionShell* const pInteractionShell = InteractionShell::GetInteractionShellFromHandle(interactionHandle);
   // if the conversion above doesn't work, it'll return null, and our free will not in fact free any memory,
   // but it will not crash. We'll leak memory, but at least we'll log that.

   // it's legal to call free on nullptr, just like for free().  This is checked inside InteractionCore::Free()
   InteractionShell::Free(pInteractionShell);

   LOG_0(Trace_Info, "Exited FreeInteractionDetector");
}

} // namespace DEFINED_ZONE_NAME
