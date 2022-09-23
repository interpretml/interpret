// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "RandomDeterministic.hpp" // RandomDeterministic
#include "RandomNondeterministic.hpp" // RandomNondeterministic

#include "Feature.hpp" // Feature
#include "Term.hpp" // Term
#include "Tensor.hpp" // Tensor

#include "BoosterCore.hpp" // BoosterCore
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct BinBase;

void BoosterShell::Free(BoosterShell * const pBoosterShell) {
   LOG_0(Trace_Info, "Entered BoosterShell::Free");

   if(nullptr != pBoosterShell) {
      Tensor::Free(pBoosterShell->m_pTermUpdate);
      Tensor::Free(pBoosterShell->m_pInnerTermUpdate);
      free(pBoosterShell->m_aBoostingFastBinsTemp);
      free(pBoosterShell->m_aBoostingBigBins);
      free(pBoosterShell->m_aMulticlassMidwayTemp);
      free(pBoosterShell->m_aSplitPositionsTemp);
      free(pBoosterShell->m_aTreeNodesTemp);
      BoosterCore::Free(pBoosterShell->m_pBoosterCore);

      // before we free our memory, indicate it was freed so if our higher level language attempts to use it we have
      // a chance to detect the error
      pBoosterShell->m_handleVerification = k_handleVerificationFreed;
      free(pBoosterShell);
   }

   LOG_0(Trace_Info, "Exited BoosterShell::Free");
}

BoosterShell * BoosterShell::Create(BoosterCore * const pBoosterCore) {
   LOG_0(Trace_Info, "Entered BoosterShell::Create");

   BoosterShell * const pNew = EbmMalloc<BoosterShell>();
   if(UNLIKELY(nullptr == pNew)) {
      LOG_0(Trace_Error, "ERROR BoosterShell::Create nullptr == pNew");
      return nullptr;
   }

   pNew->InitializeUnfailing(pBoosterCore); // take full ownership of the BoosterCore

   LOG_0(Trace_Info, "Exited BoosterShell::Create");

   return pNew;
}

ErrorEbm BoosterShell::FillAllocations() {
   EBM_ASSERT(nullptr != m_pBoosterCore);

   LOG_0(Trace_Info, "Entered BoosterShell::FillAllocations");

   const ptrdiff_t cClasses = m_pBoosterCore->GetCountClasses();
   const size_t cScores = GetCountScores(cClasses);
      
   m_pTermUpdate = Tensor::Allocate(k_cDimensionsMax, cScores);
   if(nullptr == m_pTermUpdate) {
      goto failed_allocation;
   }

   m_pInnerTermUpdate = Tensor::Allocate(k_cDimensionsMax, cScores);
   if(nullptr == m_pInnerTermUpdate) {
      goto failed_allocation;
   }

   if(0 != m_pBoosterCore->GetCountBytesFastBins()) {
      m_aBoostingFastBinsTemp = static_cast<BinBase *>(EbmMalloc<void>(m_pBoosterCore->GetCountBytesFastBins()));
      if(nullptr == m_aBoostingFastBinsTemp) {
         goto failed_allocation;
      }
   }

   if(0 != m_pBoosterCore->GetCountBytesBigBins()) {
      m_aBoostingBigBins = static_cast<BinBase *>(EbmMalloc<void>(m_pBoosterCore->GetCountBytesBigBins()));
      if(nullptr == m_aBoostingBigBins) {
         goto failed_allocation;
      }
   }

   m_aMulticlassMidwayTemp = EbmMalloc<FloatFast>(cScores);
   if(nullptr == m_aMulticlassMidwayTemp) {
      goto failed_allocation;
   }

   if(0 != m_pBoosterCore->GetCountBytesSplitPositions()) {
      m_aSplitPositionsTemp = EbmMalloc<void>(m_pBoosterCore->GetCountBytesSplitPositions());
      if(nullptr == m_aSplitPositionsTemp) {
         goto failed_allocation;
      }
   }

   if(0 != m_pBoosterCore->GetCountBytesTreeNodes()) {
      m_aTreeNodesTemp = EbmMalloc<void>(m_pBoosterCore->GetCountBytesTreeNodes());
      if(nullptr == m_aTreeNodesTemp) {
         goto failed_allocation;
      }
   }

   LOG_0(Trace_Info, "Exited BoosterShell::FillAllocations");
   return Error_None;

failed_allocation:;
   LOG_0(Trace_Warning, "WARNING Exited BoosterShell::FillAllocations with allocation failure");
   return Error_OutOfMemory;
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION CreateBooster(
   void * rng,
   const void * dataSet,
   const BagEbm * bag,
   const double * initScores,
   IntEbm countTerms,
   const IntEbm * dimensionCounts,
   const IntEbm * featureIndexes,
   IntEbm countInnerBags,
   const double * experimentalParams,
   BoosterHandle * boosterHandleOut
) {
   LOG_N(
      Trace_Info,
      "Entered CreateBooster: "
      "rng=%p, "
      "dataSet=%p, "
      "bag=%p, "
      "initScores=%p, "
      "countTerms=%" IntEbmPrintf ", "
      "dimensionCounts=%p, "
      "featureIndexes=%p, "
      "countInnerBags=%" IntEbmPrintf ", "
      "experimentalParams=%p, "
      "boosterHandleOut=%p"
      ,
      rng,
      dataSet,
      static_cast<const void *>(bag),
      static_cast<const void *>(initScores),
      countTerms,
      static_cast<const void *>(dimensionCounts),
      static_cast<const void *>(featureIndexes),
      countInnerBags,
      static_cast<const void *>(experimentalParams),
      static_cast<const void *>(boosterHandleOut)
   );

   ErrorEbm error;

   if(nullptr == boosterHandleOut) {
      LOG_0(Trace_Error, "ERROR CreateBooster nullptr == boosterHandleOut");
      return Error_IllegalParamVal;
   }
   *boosterHandleOut = nullptr; // set this to nullptr as soon as possible so the caller doesn't attempt to free it

   if(nullptr == dataSet) {
      LOG_0(Trace_Error, "ERROR CreateBooster nullptr == dataSet");
      return Error_IllegalParamVal;
   }

   if(countTerms < IntEbm { 0 }) {
      LOG_0(Trace_Error, "ERROR CreateBooster countTerms must be positive");
      return Error_IllegalParamVal;
   }
   if(IntEbm { 0 } != countTerms && nullptr == dimensionCounts) {
      LOG_0(Trace_Error, "ERROR CreateBooster dimensionCounts cannot be null if 0 < countTerms");
      return Error_IllegalParamVal;
   }
   // it's legal for featureIndexes to be null if there are no features indexed by our feature groups
   // dimensionCounts can have zero features, so it could be legal for this to be null even if 0 < countTerms

   if(countInnerBags < IntEbm { 0 }) {
      // 0 means use the full set. 1 means make a single bag which is useless, but allowed for comparison purposes
      LOG_0(Trace_Error, "ERROR CreateBooster countInnerBags cannot be negative");
      return Error_UserParamVal;
   }

   if(IsConvertError<size_t>(countTerms)) {
      // the caller should not have been able to allocate memory for dimensionCounts if this wasn't fittable in size_t
      LOG_0(Trace_Error, "ERROR CreateBooster IsConvertError<size_t>(countTerms)");
      return Error_IllegalParamVal;
   }
   if(IsConvertError<size_t>(countInnerBags)) {
      // this is just a warning since the caller doesn't pass us anything material, but if it's this high
      // then our allocation would fail since it can't even in pricipal fit into memory
      LOG_0(Trace_Warning, "WARNING CreateBooster IsConvertError<size_t>(countInnerBags)");
      return Error_OutOfMemory;
   }

   size_t cTerms = static_cast<size_t>(countTerms);
   size_t cInnerBags = static_cast<size_t>(countInnerBags);

   // TODO: move this code down into BoosterCore::Create since we can happily pass down nullptr into there and then use the CPU register trick at the lowest function level
   RandomDeterministic * pRng = reinterpret_cast<RandomDeterministic *>(rng);
   RandomDeterministic rngInternal;
   if(nullptr == pRng) {
      // We use the seed for creating inner bags. Inner bags are not used in DP-EBMs.
      // Since we do not need high-quality non-determinism, generate a non-deterministic seed
      uint64_t seed;
      try {
         RandomNondeterministic<uint64_t> randomGenerator;
         seed = randomGenerator.Next(std::numeric_limits<uint64_t>::max());
      } catch(const std::bad_alloc &) {
         LOG_0(Trace_Warning, "WARNING CreateBooster Out of memory in std::random_device");
         return Error_OutOfMemory;
      } catch(...) {
         LOG_0(Trace_Warning, "WARNING CreateBooster Unknown error in std::random_device");
         return Error_UnexpectedInternal;
      }
      rngInternal.Initialize(seed);
      pRng = &rngInternal;
   }

   // TODO: since BoosterCore is a non-POD C++ class, we should probably move the call to new from inside
   //       BoosterCore::Create to here and wrap it with a try catch at this level and rely on standard C++ behavior
   BoosterCore * pBoosterCore = nullptr;
   error = BoosterCore::Create(
      pRng,
      cTerms,
      cInnerBags,
      experimentalParams,
      dimensionCounts,
      featureIndexes,
      static_cast<const unsigned char *>(dataSet),
      bag,
      initScores,
      &pBoosterCore
   );
   if(UNLIKELY(Error_None != error)) {
      BoosterCore::Free(pBoosterCore); // legal if nullptr.  On error we can get back a legal pBoosterCore to delete
      return error;
   }

   BoosterShell * const pBoosterShell = BoosterShell::Create(pBoosterCore);
   if(UNLIKELY(nullptr == pBoosterShell)) {
      // if the memory allocation for pBoosterShell failed then there was no place to put the pBoosterCore, so free it
      BoosterCore::Free(pBoosterCore);
      return Error_OutOfMemory;
   }

   // TODO: move this into the BoosterShell::Create function
   error = pBoosterShell->FillAllocations();
   if(Error_None != error) {
      BoosterShell::Free(pBoosterShell);
      return error;
   }

   const BoosterHandle handle = pBoosterShell->GetHandle();

   LOG_N(Trace_Info, "Exited CreateBooster: *boosterHandleOut=%p", static_cast<void *>(handle));

   *boosterHandleOut = handle;
   return Error_None;
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION CreateBoosterView(
   BoosterHandle boosterHandle,
   BoosterHandle * boosterHandleViewOut
) {
   LOG_N(
      Trace_Info,
      "Entered CreateBoosterView: "
      "boosterHandle=%p, "
      "boosterHandleViewOut=%p"
      ,
      static_cast<void *>(boosterHandle),
      static_cast<void *>(boosterHandleViewOut)
   );

   ErrorEbm error;

   if(UNLIKELY(nullptr == boosterHandleViewOut)) {
      LOG_0(Trace_Warning, "WARNING CreateBooster nullptr == boosterHandleViewOut");
      return Error_IllegalParamVal;
   }
   *boosterHandleViewOut = nullptr; // set this as soon as possible so our caller doesn't end up freeing garbage

   BoosterShell * const pBoosterShellOriginal = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShellOriginal) {
      // already logged
      return Error_IllegalParamVal;
   }
   BoosterCore * const pBoosterCore = pBoosterShellOriginal->GetBoosterCore();

   BoosterShell * const pBoosterShellNew = BoosterShell::Create(pBoosterCore);
   if(UNLIKELY(nullptr == pBoosterShellNew)) {
      LOG_0(Trace_Warning, "WARNING CreateBooster nullptr == pBoosterShellNew");
      return Error_OutOfMemory;
   }
   pBoosterCore->AddReferenceCount();

   error = pBoosterShellNew->FillAllocations();
   if(Error_None != error) {
      // TODO: we might move the call to FillAllocations to be more lazy incase the caller doesn't use it all
      BoosterShell::Free(pBoosterShellNew);
      return error;
   }

   LOG_0(Trace_Info, "Exited CreateBoosterView");

   *boosterHandleViewOut = pBoosterShellNew->GetHandle();
   return Error_None;
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION GetBestTermScores(
   BoosterHandle boosterHandle,
   IntEbm indexTerm,
   double * termScoresTensorOut
) {
   LOG_N(
      Trace_Info,
      "Entered GetBestTermScores: "
      "boosterHandle=%p, "
      "indexTerm=%" IntEbmPrintf ", "
      "termScoresTensorOut=%p, "
      ,
      static_cast<void *>(boosterHandle),
      indexTerm,
      static_cast<void *>(termScoresTensorOut)
   );

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamVal;
   }

   if(indexTerm < 0) {
      LOG_0(Trace_Error, "ERROR GetBestTermScores indexTerm must be positive");
      return Error_IllegalParamVal;
   }
   if(IsConvertError<size_t>(indexTerm)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(Trace_Error, "ERROR GetBestTermScores indexTerm is too high to index");
      return Error_IllegalParamVal;
   }
   size_t iTerm = static_cast<size_t>(indexTerm);

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   if(pBoosterCore->GetCountTerms() <= iTerm) {
      LOG_0(Trace_Error, "ERROR GetBestTermScores indexTerm above the number of feature groups that we have");
      return Error_IllegalParamVal;
   }

   if(ptrdiff_t { 0 } == pBoosterCore->GetCountClasses() || ptrdiff_t { 1 } == pBoosterCore->GetCountClasses()) {
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.  
      // If there were logits in this model, they'd all be infinity, but you could alternatively think of this 
      // model as having no logits, since the number of logits can be one less than the number of target classes.
      LOG_0(Trace_Info, "Exited GetBestTermScores no scores");
      return Error_None;
   }

   if(nullptr == termScoresTensorOut) {
      LOG_0(Trace_Error, "ERROR GetBestTermScores termScoresTensorOut cannot be nullptr");
      return Error_IllegalParamVal;
   }

   // if pBoosterCore->GetTerms() is nullptr, then m_cTerms was 0, but we checked above that 
   // iTerm was less than cTerms
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());

   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];
   const size_t cDimensions = pTerm->GetCountDimensions();
   size_t cTensorScores = GetCountScores(pBoosterCore->GetCountClasses());
   if(0 != cDimensions) {
      const Feature * const * ppFeature = pTerm->GetFeatures();
      const Feature * const * const ppFeaturesEnd = &ppFeature[cDimensions];
      do {
         const Feature * const pFeature = *ppFeature;
         const size_t cBins = pFeature->GetCountBins();
         // we've allocated this memory, so it should be reachable, so these numbers should multiply
         EBM_ASSERT(!IsMultiplyError(cTensorScores, cBins));
         cTensorScores *= cBins;
         ++ppFeature;
      } while(ppFeaturesEnd != ppFeature);
   }

   // if pBoosterCore->GetBestModel() is nullptr, then either:
   //    1) m_cTerms was 0, but we checked above that iTerm was less than cTerms
   //    2) If m_cClasses was either 1 or 0, but we checked for this above too
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

   LOG_0(Trace_Info, "Exited GetBestTermScores");
   return Error_None;
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION GetCurrentTermScores(
   BoosterHandle boosterHandle,
   IntEbm indexTerm,
   double * termScoresTensorOut
) {
   LOG_N(
      Trace_Info,
      "Entered GetCurrentTermScores: "
      "boosterHandle=%p, "
      "indexTerm=%" IntEbmPrintf ", "
      "termScoresTensorOut=%p, "
      ,
      static_cast<void *>(boosterHandle),
      indexTerm,
      static_cast<void *>(termScoresTensorOut)
   );

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamVal;
   }

   if(indexTerm < 0) {
      LOG_0(Trace_Error, "ERROR GetCurrentTermScores indexTerm must be positive");
      return Error_IllegalParamVal;
   }
   if(IsConvertError<size_t>(indexTerm)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(Trace_Error, "ERROR GetCurrentTermScores indexTerm is too high to index");
      return Error_IllegalParamVal;
   }
   size_t iTerm = static_cast<size_t>(indexTerm);

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   if(pBoosterCore->GetCountTerms() <= iTerm) {
      LOG_0(Trace_Error, "ERROR GetCurrentTermScores indexTerm above the number of feature groups that we have");
      return Error_IllegalParamVal;
   }

   if(ptrdiff_t { 0 } == pBoosterCore->GetCountClasses() || ptrdiff_t { 1 } == pBoosterCore->GetCountClasses()) {
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.  
      // If there were logits in this model, they'd all be infinity, but you could alternatively think of this 
      // model as having no logits, since the number of logits can be one less than the number of target classes.
      LOG_0(Trace_Info, "Exited GetCurrentTermScores no scores");
      return Error_None;
   }

   if(nullptr == termScoresTensorOut) {
      LOG_0(Trace_Error, "ERROR GetCurrentTermScores termScoresTensorOut cannot be nullptr");
      return Error_IllegalParamVal;
   }

   // if pBoosterCore->GetTerms() is nullptr, then m_cTerms was 0, but we checked above that 
   // iTerm was less than cTerms
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());

   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];
   const size_t cDimensions = pTerm->GetCountDimensions();
   size_t cTensorScores = GetCountScores(pBoosterCore->GetCountClasses());
   if(0 != cDimensions) {
      const Feature * const * ppFeature = pTerm->GetFeatures();
      const Feature * const * const ppFeaturesEnd = &ppFeature[cDimensions];
      do {
         const Feature * const pFeature = *ppFeature;
         const size_t cBins = pFeature->GetCountBins();
         // we've allocated this memory, so it should be reachable, so these numbers should multiply
         EBM_ASSERT(!IsMultiplyError(cTensorScores, cBins));
         cTensorScores *= cBins;
         ++ppFeature;
      } while(ppFeaturesEnd != ppFeature);
   }

   // if pBoosterCore->GetCurrentModel() is nullptr, then either:
   //    1) m_cTerms was 0, but we checked above that iTerm was less than cTerms
   //    2) If m_cClasses was either 1 or 0, but we checked for this above too
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

   LOG_0(Trace_Info, "Exited GetCurrentTermScores");
   return Error_None;
}

EBM_API_BODY void EBM_CALLING_CONVENTION FreeBooster(
   BoosterHandle boosterHandle
) {
   LOG_N(Trace_Info, "Entered FreeBooster: boosterHandle=%p", static_cast<void *>(boosterHandle));

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   // if the conversion above doesn't work, it'll return null, and our free will not in fact free any memory,
   // but it will not crash. We'll leak memory, but at least we'll log that.

   // it's legal to call free on nullptr, just like for free().  This is checked inside BoosterCore::Free()
   BoosterShell::Free(pBoosterShell);

   LOG_0(Trace_Info, "Exited FreeBooster");
}

} // DEFINED_ZONE_NAME
