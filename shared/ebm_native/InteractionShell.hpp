// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef INTERACTION_SHELL_HPP
#define INTERACTION_SHELL_HPP

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct BinBase;
class InteractionCore;

class InteractionShell final {
   static constexpr size_t k_handleVerificationOk = 27917; // random 15 bit number
   static constexpr size_t k_handleVerificationFreed = 27913; // random 15 bit number
   size_t m_handleVerification; // this needs to be at the top and make it pointer sized to keep best alignment

   InteractionCore * m_pInteractionCore;

   BinBase * m_aThreadByteBuffer1Fast;
   size_t m_cThreadByteBufferCapacity1Fast;

   BinBase * m_aThreadByteBuffer1Big;
   size_t m_cThreadByteBufferCapacity1Big;

   int m_cLogEnterMessages;
   int m_cLogExitMessages;

#ifndef NDEBUG
   const unsigned char * m_pBinsFastEndDebug;
#endif // NDEBUG

public:

   InteractionShell() = default; // preserve our POD status
   ~InteractionShell() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS void InitializeUnfailing() {
      m_handleVerification = k_handleVerificationOk;
      m_pInteractionCore = nullptr;

      m_aThreadByteBuffer1Fast = nullptr;
      m_cThreadByteBufferCapacity1Fast = 0;

      m_aThreadByteBuffer1Big = nullptr;
      m_cThreadByteBufferCapacity1Big = 0;

      m_cLogEnterMessages = 1000;
      m_cLogExitMessages = 1000;
   }

   static void Free(InteractionShell * const pInteractionShell);
   static InteractionShell * Create();

   static INLINE_ALWAYS InteractionShell * GetInteractionShellFromHandle(
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
   INLINE_ALWAYS InteractionHandle GetHandle() {
      return reinterpret_cast<InteractionHandle>(this);
   }

   INLINE_ALWAYS InteractionCore * GetInteractionCore() {
      EBM_ASSERT(nullptr != m_pInteractionCore);
      return m_pInteractionCore;
   }

   INLINE_ALWAYS void SetInteractionCore(InteractionCore * const pInteractionCore) {
      EBM_ASSERT(nullptr != pInteractionCore);
      EBM_ASSERT(nullptr == m_pInteractionCore); // only set it once
      m_pInteractionCore = pInteractionCore;
   }

   INLINE_ALWAYS int * GetPointerCountLogEnterMessages() {
      return &m_cLogEnterMessages;
   }

   INLINE_ALWAYS int * GetPointerCountLogExitMessages() {
      return &m_cLogExitMessages;
   }

   BinBase * GetBinBaseFast(size_t cBytesRequired);

   INLINE_ALWAYS BinBase * GetBinBaseFast() {
      // call this if the bins were already allocated and we just need the pointer
      return m_aThreadByteBuffer1Fast;
   }

   BinBase * GetBinBaseBig(size_t cBytesRequired);

   INLINE_ALWAYS BinBase * GetBinBaseBig() {
      // call this if the bins were already allocated and we just need the pointer
      return m_aThreadByteBuffer1Big;
   }

#ifndef NDEBUG
   INLINE_ALWAYS const unsigned char * GetBinsFastEndDebug() const {
      return m_pBinsFastEndDebug;
   }

   INLINE_ALWAYS void SetBinsFastEndDebug(const unsigned char * const pBinsFastEndDebug) {
      m_pBinsFastEndDebug = pBinsFastEndDebug;
   }
#endif // NDEBUG
};
static_assert(std::is_standard_layout<InteractionShell>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<InteractionShell>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<InteractionShell>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // INTERACTION_SHELL_HPP
