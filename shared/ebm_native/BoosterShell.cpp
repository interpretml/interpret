// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "RandomStream.hpp"

#include "CompressibleTensor.hpp"

#include "HistogramTargetEntry.hpp"

#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

void BoosterShell::Free(BoosterShell * const pBoosterShell) {
   LOG_0(TraceLevelInfo, "Entered BoosterShell::Free");

   if(nullptr != pBoosterShell) {
      Tensor::Free(pBoosterShell->m_pTermUpdate);
      Tensor::Free(pBoosterShell->m_pInnerTermUpdate);
      free(pBoosterShell->m_aThreadByteBuffer1Fast);
      free(pBoosterShell->m_aThreadByteBuffer1Big);
      free(pBoosterShell->m_aThreadByteBuffer2);
      free(pBoosterShell->m_aSumAllGradientPairs);
      free(pBoosterShell->m_aLeftGradientPairs);
      free(pBoosterShell->m_aRightGradientPairs);
      free(pBoosterShell->m_aTempFloatVector);
      free(pBoosterShell->m_aEquivalentSplits);
      BoosterCore::Free(pBoosterShell->m_pBoosterCore);

      // before we free our memory, indicate it was freed so if our higher level language attempts to use it we have
      // a chance to detect the error
      pBoosterShell->m_handleVerification = k_handleVerificationFreed;
      free(pBoosterShell);
   }

   LOG_0(TraceLevelInfo, "Exited BoosterShell::Free");
}

BoosterShell * BoosterShell::Create() {
   LOG_0(TraceLevelInfo, "Entered BoosterShell::Create");

   BoosterShell * const pNew = EbmMalloc<BoosterShell>();
   if(LIKELY(nullptr != pNew)) {
      pNew->InitializeUnfailing();
   }

   LOG_0(TraceLevelInfo, "Exited BoosterShell::Create");

   return pNew;
}

ErrorEbmType BoosterShell::FillAllocations() {
   EBM_ASSERT(nullptr != m_pBoosterCore);

   LOG_0(TraceLevelInfo, "Entered BoosterShell::FillAllocations");

   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = m_pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
   const size_t cScores = GetCountScores(runtimeLearningTypeOrCountTargetClasses);
   const size_t cBytesPerGradientPair = GetGradientPairSize<FloatBig>(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      
   m_pTermUpdate = Tensor::Allocate(k_cDimensionsMax, cScores);
   if(nullptr == m_pTermUpdate) {
      goto failed_allocation;
   }

   m_pInnerTermUpdate = Tensor::Allocate(k_cDimensionsMax, cScores);
   if(nullptr == m_pInnerTermUpdate) {
      goto failed_allocation;
   }

   m_aSumAllGradientPairs = EbmMalloc<GradientPairBase>(cScores, cBytesPerGradientPair);
   if(nullptr == m_aSumAllGradientPairs) {
      goto failed_allocation;
   }

   m_aLeftGradientPairs = EbmMalloc<GradientPairBase>(cScores, cBytesPerGradientPair);
   if(nullptr == m_aLeftGradientPairs) {
      goto failed_allocation;
   }

   m_aRightGradientPairs = EbmMalloc<GradientPairBase>(cScores, cBytesPerGradientPair);
   if(nullptr == m_aRightGradientPairs) {
      goto failed_allocation;
   }

   m_aTempFloatVector = EbmMalloc<FloatFast>(cScores);
   if(nullptr == m_aTempFloatVector) {
      goto failed_allocation;
   }

   if(0 != m_pBoosterCore->GetCountBytesArrayEquivalentSplitMax()) {
      m_aEquivalentSplits = EbmMalloc<void>(m_pBoosterCore->GetCountBytesArrayEquivalentSplitMax());
      if(nullptr == m_aEquivalentSplits) {
         goto failed_allocation;
      }
   }

   LOG_0(TraceLevelInfo, "Exited BoosterShell::FillAllocations");
   return Error_None;

failed_allocation:;
   LOG_0(TraceLevelWarning, "WARNING Exited BoosterShell::FillAllocations with allocation failure");
   return Error_OutOfMemory;
}

BinBase * BoosterShell::GetBinBaseFast(size_t cBytesRequired) {
   BinBase * aBuffer = m_aThreadByteBuffer1Fast;
   if(UNLIKELY(m_cThreadByteBufferCapacity1Fast < cBytesRequired)) {
      cBytesRequired <<= 1;
      m_cThreadByteBufferCapacity1Fast = cBytesRequired;
      LOG_N(TraceLevelInfo, "Growing BoosterShell::ThreadByteBuffer1Fast to %zu", cBytesRequired);

      free(aBuffer);
      aBuffer = static_cast<BinBase *>(EbmMalloc<void>(cBytesRequired));
      m_aThreadByteBuffer1Fast = aBuffer; // store it before checking it incase it's null so that we don't free old memory
      if(nullptr == aBuffer) {
         LOG_0(TraceLevelWarning, "WARNING BoosterShell::GetBinBaseFast OutOfMemory");
      }
   }
   return aBuffer;
}

BinBase * BoosterShell::GetBinBaseBig(size_t cBytesRequired) {
   BinBase * aBuffer = m_aThreadByteBuffer1Big;
   if(UNLIKELY(m_cThreadByteBufferCapacity1Big < cBytesRequired)) {
      cBytesRequired <<= 1;
      m_cThreadByteBufferCapacity1Big = cBytesRequired;
      LOG_N(TraceLevelInfo, "Growing BoosterShell::ThreadByteBuffer1Big to %zu", cBytesRequired);

      free(aBuffer);
      aBuffer = static_cast<BinBase *>(EbmMalloc<void>(cBytesRequired));
      m_aThreadByteBuffer1Big = aBuffer; // store it before checking it incase it's null so that we don't free old memory
      if(nullptr == aBuffer) {
         LOG_0(TraceLevelWarning, "WARNING BoosterShell::GetBinBaseBig OutOfMemory");
      }
   }
   return aBuffer;
}

ErrorEbmType BoosterShell::GrowThreadByteBuffer2(const size_t cByteBoundaries) {
   // by adding cByteBoundaries and shifting our existing size, we do 2 things:
   //   1) we ensure that if we have zero size, we'll get some size that we'll get a non-zero size after the shift
   //   2) we'll always get back an odd number of items, which is good because we always have an odd number of TreeNodeChilden
   EBM_ASSERT(0 == m_cThreadByteBufferCapacity2 % cByteBoundaries);
   m_cThreadByteBufferCapacity2 = cByteBoundaries + (m_cThreadByteBufferCapacity2 << 1);
   LOG_N(TraceLevelInfo, "Growing BoosterShell::ThreadByteBuffer2 to %zu", m_cThreadByteBufferCapacity2);

   // our tree objects have internal pointers, so we're going to dispose of our work anyways
   // We can't use realloc since there is no way to check if the array was re-allocated or not without 
   // invoking undefined behavior, so we don't get a benefit if the array can be resized with realloc

   void * aBuffer = m_aThreadByteBuffer2;
   free(aBuffer);
   aBuffer = EbmMalloc<void>(m_cThreadByteBufferCapacity2);
   m_aThreadByteBuffer2 = aBuffer; // store it before checking it incase it's null so that we don't free old memory
   if(UNLIKELY(nullptr == aBuffer)) {
      LOG_0(TraceLevelWarning, "WARNING GrowThreadByteBuffer2 OutOfMemory");
      return Error_OutOfMemory;
   }
   return Error_None;
}

EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION CreateBooster(
   SeedEbmType randomSeed,
   const void * dataSet,
   const BagEbmType * bag,
   const double * initScores,
   IntEbmType countTerms,
   const IntEbmType * dimensionCounts,
   const IntEbmType * featureIndexes,
   IntEbmType countInnerBags,
   const double * optionalTempParams,
   BoosterHandle * boosterHandleOut
) {
   LOG_N(
      TraceLevelInfo,
      "Entered CreateBooster: "
      "randomSeed=%" SeedEbmTypePrintf ", "
      "dataSet=%p, "
      "bag=%p, "
      "initScores=%p, "
      "countTerms=%" IntEbmTypePrintf ", "
      "dimensionCounts=%p, "
      "featureIndexes=%p, "
      "countInnerBags=%" IntEbmTypePrintf ", "
      "optionalTempParams=%p, "
      "boosterHandleOut=%p"
      ,
      randomSeed,
      static_cast<const void *>(dataSet),
      static_cast<const void *>(bag),
      static_cast<const void *>(initScores),
      countTerms,
      static_cast<const void *>(dimensionCounts),
      static_cast<const void *>(featureIndexes),
      countInnerBags,
      static_cast<const void *>(optionalTempParams),
      static_cast<const void *>(boosterHandleOut)
   );

   ErrorEbmType error;

   if(nullptr == boosterHandleOut) {
      LOG_0(TraceLevelError, "ERROR CreateBooster nullptr == boosterHandleOut");
      return Error_IllegalParamValue;
   }
   *boosterHandleOut = nullptr; // set this to nullptr as soon as possible so the caller doesn't attempt to free it

   if(nullptr == dataSet) {
      LOG_0(TraceLevelError, "ERROR CreateBooster nullptr == dataSet");
      return Error_IllegalParamValue;
   }

   if(countTerms < IntEbmType { 0 }) {
      LOG_0(TraceLevelError, "ERROR CreateBooster countTerms must be positive");
      return Error_IllegalParamValue;
   }
   if(IntEbmType { 0 } != countTerms && nullptr == dimensionCounts) {
      LOG_0(TraceLevelError, "ERROR CreateBooster dimensionCounts cannot be null if 0 < countTerms");
      return Error_IllegalParamValue;
   }
   // it's legal for featureIndexes to be null if there are no features indexed by our feature groups
   // dimensionCounts can have zero features, so it could be legal for this to be null even if 0 < countTerms

   if(countInnerBags < IntEbmType { 0 }) {
      // 0 means use the full set. 1 means make a single bag which is useless, but allowed for comparison purposes
      LOG_0(TraceLevelError, "ERROR CreateBooster countInnerBags cannot be negative");
      return Error_UserParamValue;
   }

   if(IsConvertError<size_t>(countTerms)) {
      // the caller should not have been able to allocate memory for dimensionCounts if this wasn't fittable in size_t
      LOG_0(TraceLevelError, "ERROR CreateBooster IsConvertError<size_t>(countTerms)");
      return Error_IllegalParamValue;
   }
   if(IsConvertError<size_t>(countInnerBags)) {
      // this is just a warning since the caller doesn't pass us anything material, but if it's this high
      // then our allocation would fail since it can't even in pricipal fit into memory
      LOG_0(TraceLevelWarning, "WARNING CreateBooster IsConvertError<size_t>(countInnerBags)");
      return Error_OutOfMemory;
   }

   size_t cTerms = static_cast<size_t>(countTerms);
   size_t cInnerBags = static_cast<size_t>(countInnerBags);

   BoosterShell * const pBoosterShell = BoosterShell::Create();
   if(UNLIKELY(nullptr == pBoosterShell)) {
      return Error_OutOfMemory;
   }

   pBoosterShell->GetRandomDeterministic()->InitializeUnsigned(randomSeed, k_boosterRandomizationMix);

   error = BoosterCore::Create(
      pBoosterShell,
      cTerms,
      cInnerBags,
      optionalTempParams,
      dimensionCounts,
      featureIndexes,
      static_cast<const unsigned char *>(dataSet),
      bag,
      initScores
   );
   if(UNLIKELY(Error_None != error)) {
      BoosterShell::Free(pBoosterShell);
      return error;
   }

   error = pBoosterShell->FillAllocations();
   if(Error_None != error) {
      BoosterShell::Free(pBoosterShell);
      return error;
   }

   const BoosterHandle handle = pBoosterShell->GetHandle();

   LOG_N(TraceLevelInfo, "Exited CreateBooster: *boosterHandleOut=%p", static_cast<void *>(handle));

   *boosterHandleOut = handle;
   return Error_None;
}

EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION CreateBoosterView(
   BoosterHandle boosterHandle,
   BoosterHandle * boosterHandleViewOut
) {
   LOG_N(
      TraceLevelInfo,
      "Entered CreateBoosterView: "
      "boosterHandle=%p, "
      "boosterHandleViewOut=%p"
      ,
      static_cast<void *>(boosterHandle),
      static_cast<void *>(boosterHandleViewOut)
   );

   ErrorEbmType error;

   if(UNLIKELY(nullptr == boosterHandleViewOut)) {
      LOG_0(TraceLevelWarning, "WARNING CreateBooster nullptr == boosterHandleViewOut");
      return Error_IllegalParamValue;
   }
   *boosterHandleViewOut = nullptr; // set this as soon as possible so our caller doesn't end up freeing garbage

   BoosterShell * const pBoosterShellOriginal = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShellOriginal) {
      // already logged
      return Error_IllegalParamValue;
   }

   BoosterShell * const pBoosterShellNew = BoosterShell::Create();
   if(UNLIKELY(nullptr == pBoosterShellNew)) {
      LOG_0(TraceLevelWarning, "WARNING CreateBooster nullptr == pBoosterShellNew");
      return Error_OutOfMemory;
   }

   pBoosterShellNew->GetRandomDeterministic()->Initialize(*pBoosterShellOriginal->GetRandomDeterministic());

   BoosterCore * const pBoosterCore = pBoosterShellOriginal->GetBoosterCore();
   pBoosterCore->AddReferenceCount();
   pBoosterShellNew->SetBoosterCore(pBoosterCore); // assume ownership of pBoosterCore reference count increment

   error = pBoosterShellNew->FillAllocations();
   if(Error_None != error) {
      // TODO: we might move the call to FillAllocations to be more lazy incase the caller doesn't use it all
      BoosterShell::Free(pBoosterShellNew);
      return error;
   }

   LOG_0(TraceLevelInfo, "Exited CreateBoosterView");

   *boosterHandleViewOut = pBoosterShellNew->GetHandle();
   return Error_None;
}

EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION GetBestTermScores(
   BoosterHandle boosterHandle,
   IntEbmType indexTerm,
   double * termScoresTensorOut
) {
   LOG_N(
      TraceLevelInfo,
      "Entered GetBestTermScores: "
      "boosterHandle=%p, "
      "indexTerm=%" IntEbmTypePrintf ", "
      "termScoresTensorOut=%p, "
      ,
      static_cast<void *>(boosterHandle),
      indexTerm,
      static_cast<void *>(termScoresTensorOut)
   );

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamValue;
   }

   if(indexTerm < 0) {
      LOG_0(TraceLevelError, "ERROR GetBestTermScores indexTerm must be positive");
      return Error_IllegalParamValue;
   }
   if(IsConvertError<size_t>(indexTerm)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(TraceLevelError, "ERROR GetBestTermScores indexTerm is too high to index");
      return Error_IllegalParamValue;
   }
   size_t iTerm = static_cast<size_t>(indexTerm);

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   if(pBoosterCore->GetCountTerms() <= iTerm) {
      LOG_0(TraceLevelError, "ERROR GetBestTermScores indexTerm above the number of feature groups that we have");
      return Error_IllegalParamValue;
   }

   if(ptrdiff_t { 0 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses() ||
      ptrdiff_t { 1 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses()) {
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.  
      // If there were logits in this model, they'd all be infinity, but you could alternatively think of this 
      // model as having no logits, since the number of logits can be one less than the number of target classes.
      LOG_0(TraceLevelInfo, "Exited GetBestTermScores no scores");
      return Error_None;
   }

   if(nullptr == termScoresTensorOut) {
      LOG_0(TraceLevelError, "ERROR GetBestTermScores termScoresTensorOut cannot be nullptr");
      return Error_IllegalParamValue;
   }

   // if pBoosterCore->GetTerms() is nullptr, then m_cTerms was 0, but we checked above that 
   // iTerm was less than cTerms
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());

   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];
   const size_t cDimensions = pTerm->GetCountDimensions();
   size_t cTensorScores = GetCountScores(pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses());
   if(0 != cDimensions) {
      const TermEntry * pTermEntry = pTerm->GetTermEntries();
      const TermEntry * const pTermEntriesEnd = &pTermEntry[cDimensions];
      do {
         const size_t cBins = pTermEntry->m_pFeature->GetCountBins();
         // we've allocated this memory, so it should be reachable, so these numbers should multiply
         EBM_ASSERT(!IsMultiplyError(cTensorScores, cBins));
         cTensorScores *= cBins;
         ++pTermEntry;
      } while(pTermEntriesEnd != pTermEntry);
   }

   // if pBoosterCore->GetBestModel() is nullptr, then either:
   //    1) m_cTerms was 0, but we checked above that iTerm was less than cTerms
   //    2) If m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0, but we checked for this above too
   EBM_ASSERT(nullptr != pBoosterCore->GetBestModel());

   Tensor * const pBestTermTensor = pBoosterCore->GetBestModel()[iTerm];
   EBM_ASSERT(nullptr != pBestTermTensor);
   EBM_ASSERT(pBestTermTensor->GetExpanded()); // the tensor should have been expanded at startup
   FloatFast * const aTermScores = pBestTermTensor->GetTensorScoresPointer();
   EBM_ASSERT(nullptr != aTermScores);

   EBM_ASSERT(!IsMultiplyError(sizeof(*termScoresTensorOut), cTensorScores));
   EBM_ASSERT(!IsMultiplyError(sizeof(*aTermScores), cTensorScores));
   static_assert(sizeof(*termScoresTensorOut) == sizeof(*aTermScores), "float mismatch");
   memcpy(termScoresTensorOut, aTermScores, sizeof(*aTermScores) * cTensorScores);

   LOG_0(TraceLevelInfo, "Exited GetBestTermScores");
   return Error_None;
}

EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION GetCurrentTermScores(
   BoosterHandle boosterHandle,
   IntEbmType indexTerm,
   double * termScoresTensorOut
) {
   LOG_N(
      TraceLevelInfo,
      "Entered GetCurrentTermScores: "
      "boosterHandle=%p, "
      "indexTerm=%" IntEbmTypePrintf ", "
      "termScoresTensorOut=%p, "
      ,
      static_cast<void *>(boosterHandle),
      indexTerm,
      static_cast<void *>(termScoresTensorOut)
   );

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamValue;
   }

   if(indexTerm < 0) {
      LOG_0(TraceLevelError, "ERROR GetCurrentTermScores indexTerm must be positive");
      return Error_IllegalParamValue;
   }
   if(IsConvertError<size_t>(indexTerm)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(TraceLevelError, "ERROR GetCurrentTermScores indexTerm is too high to index");
      return Error_IllegalParamValue;
   }
   size_t iTerm = static_cast<size_t>(indexTerm);

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   if(pBoosterCore->GetCountTerms() <= iTerm) {
      LOG_0(TraceLevelError, "ERROR GetCurrentTermScores indexTerm above the number of feature groups that we have");
      return Error_IllegalParamValue;
   }

   if(ptrdiff_t { 0 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses() ||
      ptrdiff_t { 1 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses())    {
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.  
      // If there were logits in this model, they'd all be infinity, but you could alternatively think of this 
      // model as having no logits, since the number of logits can be one less than the number of target classes.
      LOG_0(TraceLevelInfo, "Exited GetCurrentTermScores no scores");
      return Error_None;
   }

   if(nullptr == termScoresTensorOut) {
      LOG_0(TraceLevelError, "ERROR GetCurrentTermScores termScoresTensorOut cannot be nullptr");
      return Error_IllegalParamValue;
   }

   // if pBoosterCore->GetTerms() is nullptr, then m_cTerms was 0, but we checked above that 
   // iTerm was less than cTerms
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());

   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];
   const size_t cDimensions = pTerm->GetCountDimensions();
   size_t cTensorScores = GetCountScores(pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses());
   if(0 != cDimensions) {
      const TermEntry * pTermEntry = pTerm->GetTermEntries();
      const TermEntry * const pTermEntriesEnd = &pTermEntry[cDimensions];
      do {
         const size_t cBins = pTermEntry->m_pFeature->GetCountBins();
         // we've allocated this memory, so it should be reachable, so these numbers should multiply
         EBM_ASSERT(!IsMultiplyError(cTensorScores, cBins));
         cTensorScores *= cBins;
         ++pTermEntry;
      } while(pTermEntriesEnd != pTermEntry);
   }

   // if pBoosterCore->GetCurrentModel() is nullptr, then either:
   //    1) m_cTerms was 0, but we checked above that iTerm was less than cTerms
   //    2) If m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0, but we checked for this above too
   EBM_ASSERT(nullptr != pBoosterCore->GetCurrentModel());

   Tensor * const pCurrentTermTensor = pBoosterCore->GetCurrentModel()[iTerm];
   EBM_ASSERT(nullptr != pCurrentTermTensor);
   EBM_ASSERT(pCurrentTermTensor->GetExpanded()); // the tensor should have been expanded at startup
   FloatFast * const aTermScores = pCurrentTermTensor->GetTensorScoresPointer();
   EBM_ASSERT(nullptr != aTermScores);

   EBM_ASSERT(!IsMultiplyError(sizeof(*termScoresTensorOut), cTensorScores));
   EBM_ASSERT(!IsMultiplyError(sizeof(*aTermScores), cTensorScores));
   static_assert(sizeof(*termScoresTensorOut) == sizeof(*aTermScores), "float mismatch");
   memcpy(termScoresTensorOut, aTermScores, sizeof(*aTermScores) * cTensorScores);

   LOG_0(TraceLevelInfo, "Exited GetCurrentTermScores");
   return Error_None;
}

EBM_API_BODY void EBM_CALLING_CONVENTION FreeBooster(
   BoosterHandle boosterHandle
) {
   LOG_N(TraceLevelInfo, "Entered FreeBooster: boosterHandle=%p", static_cast<void *>(boosterHandle));

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   // if the conversion above doesn't work, it'll return null, and our free will not in fact free any memory,
   // but it will not crash. We'll leak memory, but at least we'll log that.

   // it's legal to call free on nullptr, just like for free().  This is checked inside BoosterCore::Free()
   BoosterShell::Free(pBoosterShell);

   LOG_0(TraceLevelInfo, "Exited FreeBooster");
}

} // DEFINED_ZONE_NAME
