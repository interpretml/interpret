// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef INTERACTION_SHELL_HPP
#define INTERACTION_SHELL_HPP

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // InteractionHandle
#include "logging.h" // LOG_0
#include "zones.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct BinBase;
class InteractionCore;

class InteractionShell final {
   static constexpr size_t k_handleVerificationOk = 21773; // random 15 bit number
   static constexpr size_t k_handleVerificationFreed = 27913; // random 15 bit number
   size_t m_handleVerification; // this needs to be at the top and make it pointer sized to keep best alignment

   InteractionCore * m_pInteractionCore;

   BinBase * m_aInteractionFastBinsTemp;
   size_t m_cAllocatedFastBins;

   BinBase * m_aInteractionBigBins;
   size_t m_cAllocatedBigBins;

   int m_cLogEnterMessages;
   int m_cLogExitMessages;

#ifndef NDEBUG
   const BinBase * m_pDebugFastBinsEnd;
#endif // NDEBUG

public:

   InteractionShell() = default; // preserve our POD status
   ~InteractionShell() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   inline void InitializeUnfailing(InteractionCore * const pInteractionCore) {
      m_handleVerification = k_handleVerificationOk;
      m_pInteractionCore = pInteractionCore;

      m_aInteractionFastBinsTemp = nullptr;
      m_cAllocatedFastBins = 0;

      m_aInteractionBigBins = nullptr;
      m_cAllocatedBigBins = 0;

      m_cLogEnterMessages = 1000;
      m_cLogExitMessages = 1000;
   }

   static void Free(InteractionShell * const pInteractionShell);
   static InteractionShell * Create(InteractionCore * const pInteractionCore);

   inline static InteractionShell * GetInteractionShellFromHandle(
      const InteractionHandle interactionHandle
   ) {
      if(nullptr == interactionHandle) {
         LOG_0(Trace_Error, "ERROR GetInteractionShellFromHandle null interactionHandle");
         return nullptr;
      }
      InteractionShell * const pInteractionShell = reinterpret_cast<InteractionShell *>(interactionHandle);
      if(k_handleVerificationOk == pInteractionShell->m_handleVerification) {
         return pInteractionShell;
      }
      if(k_handleVerificationFreed == pInteractionShell->m_handleVerification) {
         LOG_0(Trace_Error, "ERROR GetInteractionShellFromHandle attempt to use freed InteractionHandle");
      } else {
         LOG_0(Trace_Error, "ERROR GetInteractionShellFromHandle attempt to use invalid InteractionHandle");
      }
      return nullptr;
   }
   inline InteractionHandle GetHandle() {
      return reinterpret_cast<InteractionHandle>(this);
   }

   inline InteractionCore * GetInteractionCore() {
      EBM_ASSERT(nullptr != m_pInteractionCore);
      return m_pInteractionCore;
   }

   inline int * GetPointerCountLogEnterMessages() {
      return &m_cLogEnterMessages;
   }

   inline int * GetPointerCountLogExitMessages() {
      return &m_cLogExitMessages;
   }

   BinBase * GetInteractionFastBinsTemp(const size_t cBytesPerFastBin, const size_t cFastBins);

   inline BinBase * GetInteractionFastBinsTemp() {
      // call this if the bins were already allocated and we just need the pointer
      return m_aInteractionFastBinsTemp;
   }

   BinBase * GetInteractionBigBins(const size_t cBytesPerBigBin, const size_t cBigBins);

   inline BinBase * GetInteractionBigBins() {
      // call this if the bins were already allocated and we just need the pointer
      return m_aInteractionBigBins;
   }
};
static_assert(std::is_standard_layout<InteractionShell>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<InteractionShell>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<InteractionShell>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // INTERACTION_SHELL_HPP
