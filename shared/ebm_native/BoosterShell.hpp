// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef BOOSTER_SHELL_HPP
#define BOOSTER_SHELL_HPP

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "RandomDeterministic.hpp"
#include "GradientPair.hpp"
#include "Bin.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct BinBase;
class BoosterCore;

class BoosterShell final {
   static constexpr size_t k_handleVerificationOk = 10995; // random 15 bit number
   static constexpr size_t k_handleVerificationFreed = 25073; // random 15 bit number
   size_t m_handleVerification; // this needs to be at the top and make it pointer sized to keep best alignment

   BoosterCore * m_pBoosterCore;
   size_t m_iTerm;

   Tensor * m_pTermUpdate;
   Tensor * m_pInnerTermUpdate;

   // TODO: can I preallocate m_aThreadByteBuffer1 without resorting to growing if I examine my inputs
   // TODO: can merge m_aThreadByteBuffer2 with the pair boosting memory (if it isn't already merged)

   BinBase * m_aThreadByteBuffer1Fast;
   size_t m_cThreadByteBufferCapacity1Fast;

   BinBase * m_aThreadByteBuffer1Big;
   size_t m_cThreadByteBufferCapacity1Big;

   void * m_aThreadByteBuffer2;

   FloatFast * m_aTempFloatVector;
   void * m_aEquivalentSplits; // we use different structures for mains and multidimension and between classification and regression

#ifndef NDEBUG
   const unsigned char * m_pBinsFastEndDebug;
   const unsigned char * m_pBinsBigEndDebug;
#endif // NDEBUG

public:

   BoosterShell() = default; // preserve our POD status
   ~BoosterShell() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   constexpr static size_t k_illegalTermIndex = size_t { static_cast<size_t>(ptrdiff_t { -1 }) };

   INLINE_ALWAYS void InitializeUnfailing() {
      m_handleVerification = k_handleVerificationOk;
      m_pBoosterCore = nullptr;
      m_iTerm = k_illegalTermIndex;
      m_pTermUpdate = nullptr;
      m_pInnerTermUpdate = nullptr;
      m_aThreadByteBuffer1Fast = nullptr;
      m_cThreadByteBufferCapacity1Fast = 0;
      m_aThreadByteBuffer1Big = nullptr;
      m_cThreadByteBufferCapacity1Big = 0;
      m_aThreadByteBuffer2 = nullptr;
      m_aTempFloatVector = nullptr;
      m_aEquivalentSplits = nullptr;
   }

   static void Free(BoosterShell * const pBoosterShell);
   static BoosterShell * Create();
   ErrorEbm FillAllocations();

   static INLINE_ALWAYS BoosterShell * GetBoosterShellFromHandle(const BoosterHandle boosterHandle) {
      if(nullptr == boosterHandle) {
         LOG_0(Trace_Error, "ERROR GetBoosterShellFromHandle null boosterHandle");
         return nullptr;
      }
      BoosterShell * const pBoosterShell = reinterpret_cast<BoosterShell *>(boosterHandle);
      if(k_handleVerificationOk == pBoosterShell->m_handleVerification) {
         return pBoosterShell;
      }
      if(k_handleVerificationFreed == pBoosterShell->m_handleVerification) {
         LOG_0(Trace_Error, "ERROR GetBoosterShellFromHandle attempt to use freed BoosterHandle");
      } else {
         LOG_0(Trace_Error, "ERROR GetBoosterShellFromHandle attempt to use invalid BoosterHandle");
      }
      return nullptr;
   }
   INLINE_ALWAYS BoosterHandle GetHandle() {
      return reinterpret_cast<BoosterHandle>(this);
   }

   INLINE_ALWAYS BoosterCore * GetBoosterCore() {
      EBM_ASSERT(nullptr != m_pBoosterCore);
      return m_pBoosterCore;
   }

   INLINE_ALWAYS void SetBoosterCore(BoosterCore * const pBoosterCore) {
      EBM_ASSERT(nullptr != pBoosterCore);
      EBM_ASSERT(nullptr == m_pBoosterCore); // only set it once
      m_pBoosterCore = pBoosterCore;
   }

   INLINE_ALWAYS size_t GetTermIndex() {
      return m_iTerm;
   }

   INLINE_ALWAYS void SetTermIndex(const size_t iTerm) {
      m_iTerm = iTerm;
   }

   INLINE_ALWAYS Tensor * GetTermUpdate() {
      return m_pTermUpdate;
   }

   INLINE_ALWAYS Tensor * GetInnerTermUpdate() {
      return m_pInnerTermUpdate;
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

   INLINE_ALWAYS void * GetThreadByteBuffer2() {
      return m_aThreadByteBuffer2;
   }

   INLINE_ALWAYS FloatFast * GetTempFloatVector() {
      return m_aTempFloatVector;
   }

   INLINE_ALWAYS void * GetEquivalentSplits() {
      return m_aEquivalentSplits;
   }


#ifndef NDEBUG
   INLINE_ALWAYS const unsigned char * GetBinsFastEndDebug() const {
      return m_pBinsFastEndDebug;
   }

   INLINE_ALWAYS void SetBinsFastEndDebug(const unsigned char * const pBinsFastEndDebug) {
      m_pBinsFastEndDebug = pBinsFastEndDebug;
   }

   INLINE_ALWAYS const unsigned char * GetBinsBigEndDebug() const {
      return m_pBinsBigEndDebug;
   }

   INLINE_ALWAYS void SetBinsBigEndDebug(const unsigned char * const pBinsBigEndDebug) {
      m_pBinsBigEndDebug = pBinsBigEndDebug;
   }
#endif // NDEBUG
};
static_assert(std::is_standard_layout<BoosterShell>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<BoosterShell>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<BoosterShell>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // BOOSTER_SHELL_HPP
