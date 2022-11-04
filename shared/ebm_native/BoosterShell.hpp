// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef BOOSTER_SHELL_HPP
#define BOOSTER_SHELL_HPP

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // FloatFast
#include "zones.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class Tensor;

struct BinBase;
class BoosterCore;

template<bool bClassification, size_t cCompilerScores>
struct SplitPosition;

template<bool bClassification, size_t cCompilerScores>
struct TreeNode;

class BoosterShell final {
   static constexpr size_t k_handleVerificationOk = 10995; // random 15 bit number
   static constexpr size_t k_handleVerificationFreed = 25073; // random 15 bit number
   size_t m_handleVerification; // this needs to be at the top and make it pointer sized to keep best alignment

   BoosterCore * m_pBoosterCore;
   size_t m_iTerm;

   Tensor * m_pTermUpdate;
   Tensor * m_pInnerTermUpdate;

   // TODO: try to merge some of this memory so that we get more CPU cache residency
   BinBase * m_aBoostingFastBinsTemp;
   BinBase * m_aBoostingBigBins;

   // TODO: I think this can share memory with m_aBoostingFastBinsTemp since the GradientPair always contains a FloatFast, and it always contains enough for the multiclass scores in the first bin, and we always have at least 1 bin, right?
   FloatFast * m_aMulticlassMidwayTemp;

   void * m_aTreeNodesTemp;
   void * m_aSplitPositionsTemp;

#ifndef NDEBUG
   const BinBase * m_pDebugBigBinsEnd;
#endif // NDEBUG

public:

   BoosterShell() = default; // preserve our POD status
   ~BoosterShell() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   static constexpr size_t k_illegalTermIndex = size_t { static_cast<size_t>(ptrdiff_t { -1 }) };

   INLINE_ALWAYS void InitializeUnfailing(BoosterCore * const pBoosterCore) {
      m_handleVerification = k_handleVerificationOk;
      m_pBoosterCore = pBoosterCore;
      m_iTerm = k_illegalTermIndex;
      m_pTermUpdate = nullptr;
      m_pInnerTermUpdate = nullptr;
      m_aBoostingFastBinsTemp = nullptr;
      m_aBoostingBigBins = nullptr;
      m_aMulticlassMidwayTemp = nullptr;
      m_aTreeNodesTemp = nullptr;
      m_aSplitPositionsTemp = nullptr;
   }

   static void Free(BoosterShell * const pBoosterShell);
   static BoosterShell * Create(BoosterCore * const pBoosterCore);
   ErrorEbm FillAllocations();

   INLINE_ALWAYS static BoosterShell * GetBoosterShellFromHandle(const BoosterHandle boosterHandle) {
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

   INLINE_ALWAYS BinBase * GetBoostingFastBinsTemp() {
      // call this if the bins were already allocated and we just need the pointer
      return m_aBoostingFastBinsTemp;
   }

   INLINE_ALWAYS BinBase * GetBoostingBigBins() {
      // call this if the bins were already allocated and we just need the pointer
      return m_aBoostingBigBins;
   }

   INLINE_ALWAYS FloatFast * GetMulticlassMidwayTemp() {
      return m_aMulticlassMidwayTemp;
   }

   template<bool bClassification, size_t cCompilerScores = 1>
   INLINE_ALWAYS TreeNode<bClassification, cCompilerScores> * GetTreeNodesTemp() {
      return static_cast<TreeNode<bClassification, cCompilerScores> *>(m_aTreeNodesTemp);
   }

   template<bool bClassification, size_t cCompilerScores = 1>
   INLINE_ALWAYS SplitPosition<bClassification, cCompilerScores> * GetSplitPositionsTemp() {
      return static_cast<SplitPosition<bClassification, cCompilerScores> *>(m_aSplitPositionsTemp);
   }


#ifndef NDEBUG
   INLINE_ALWAYS const BinBase * GetDebugBigBinsEnd() const {
      return m_pDebugBigBinsEnd;
   }

   INLINE_ALWAYS void SetDebugBigBinsEnd(const BinBase * const pDebugBigBinsEnd) {
      m_pDebugBigBinsEnd = pDebugBigBinsEnd;
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
