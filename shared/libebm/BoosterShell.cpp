// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#define ZONE_main
#include "zones.h"

#include "RandomDeterministic.hpp" // RandomDeterministic

#include "Feature.hpp" // Feature
#include "Term.hpp" // Term
#include "Transpose.hpp"
#include "Tensor.hpp" // Tensor

#include "BoosterCore.hpp" // BoosterCore
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct BinBase;

extern void InitializeRmseGradientsAndHessiansBoosting(const unsigned char* const pDataSetShared,
      const BagEbm direction,
      const BagEbm* const aBag,
      const double* const aInitScores,
      DataSetBoosting* const pDataSet);

void BoosterShell::Free(BoosterShell* const pBoosterShell) {
   LOG_0(Trace_Info, "Entered BoosterShell::Free");

   if(nullptr != pBoosterShell) {
      Tensor::Free(pBoosterShell->m_pTermUpdate);
      Tensor::Free(pBoosterShell->m_pInnerTermUpdate);
      AlignedFree(pBoosterShell->m_aBoostingFastBinsTemp);
      AlignedFree(pBoosterShell->m_aBoostingMainBins);
      AlignedFree(pBoosterShell->m_aMulticlassMidwayTemp);
      AlignedFree(pBoosterShell->m_aSplitPositionsTemp);
      AlignedFree(pBoosterShell->m_aTreeNodesTemp);
      BoosterCore::Free(pBoosterShell->m_pBoosterCore);

      // before we free our memory, indicate it was freed so if our higher level language attempts to use it we have
      // a chance to detect the error
      pBoosterShell->m_handleVerification = k_handleVerificationFreed;
      free(pBoosterShell);
   }

   LOG_0(Trace_Info, "Exited BoosterShell::Free");
}

BoosterShell* BoosterShell::Create(BoosterCore* const pBoosterCore) {
   LOG_0(Trace_Info, "Entered BoosterShell::Create");

   BoosterShell* const pNew = static_cast<BoosterShell*>(malloc(sizeof(BoosterShell)));
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

   const size_t cScores = m_pBoosterCore->GetCountScores();
   if(size_t{0} != cScores) {
      m_pTermUpdate = Tensor::Allocate(k_cDimensionsMax, cScores);
      if(nullptr == m_pTermUpdate) {
         goto failed_allocation;
      }

      m_pInnerTermUpdate = Tensor::Allocate(k_cDimensionsMax, cScores);
      if(nullptr == m_pInnerTermUpdate) {
         goto failed_allocation;
      }

      if(0 != m_pBoosterCore->GetCountBytesFastBins()) {
         m_aBoostingFastBinsTemp = static_cast<BinBase*>(AlignedAlloc(m_pBoosterCore->GetCountBytesFastBins()));
         if(nullptr == m_aBoostingFastBinsTemp) {
            goto failed_allocation;
         }
      }

      if(0 != m_pBoosterCore->GetCountBytesMainBins()) {
         m_aBoostingMainBins = static_cast<BinBase*>(AlignedAlloc(m_pBoosterCore->GetCountBytesMainBins()));
         if(nullptr == m_aBoostingMainBins) {
            goto failed_allocation;
         }
      }

      if(size_t{1} != cScores) {
         size_t cBytesMulticlassMidwayMax = 0;
         if(0 != GetBoosterCore()->GetTrainingSet()->GetCountSamples()) {
            DataSubsetBoosting* pSubset = GetBoosterCore()->GetTrainingSet()->GetSubsets();
            const DataSubsetBoosting* const pSubsetsEnd =
                  pSubset + GetBoosterCore()->GetTrainingSet()->GetCountSubsets();
            do {
               const size_t cSIMDPack = pSubset->GetObjectiveWrapper()->m_cSIMDPack;
               const size_t cFloatBytes = pSubset->GetObjectiveWrapper()->m_cFloatBytes;
               const size_t cMultiple = cFloatBytes * cSIMDPack;
               if(IsMultiplyError(cMultiple, cScores)) {
                  goto failed_allocation;
               }
               const size_t cBytesMulticlassMidway = cMultiple * cScores;
               cBytesMulticlassMidwayMax = EbmMax(cBytesMulticlassMidwayMax, cBytesMulticlassMidway);

               ++pSubset;
            } while(pSubsetsEnd != pSubset);
         }

         if(0 != GetBoosterCore()->GetValidationSet()->GetCountSamples()) {
            DataSubsetBoosting* pSubset = GetBoosterCore()->GetValidationSet()->GetSubsets();
            const DataSubsetBoosting* const pSubsetsEnd =
                  pSubset + GetBoosterCore()->GetValidationSet()->GetCountSubsets();
            do {
               const size_t cSIMDPack = pSubset->GetObjectiveWrapper()->m_cSIMDPack;
               const size_t cFloatBytes = pSubset->GetObjectiveWrapper()->m_cFloatBytes;
               const size_t cMultiple = cFloatBytes * cSIMDPack;
               if(IsMultiplyError(cMultiple, cScores)) {
                  goto failed_allocation;
               }
               const size_t cBytesMulticlassMidway = cMultiple * cScores;
               cBytesMulticlassMidwayMax = EbmMax(cBytesMulticlassMidwayMax, cBytesMulticlassMidway);

               ++pSubset;
            } while(pSubsetsEnd != pSubset);
         }

         // if there are zero samples, cFloatBytesMax will be zero
         if(0 != cBytesMulticlassMidwayMax) {
            m_aMulticlassMidwayTemp = AlignedAlloc(cBytesMulticlassMidwayMax);
            if(nullptr == m_aMulticlassMidwayTemp) {
               goto failed_allocation;
            }
         }
      }

      if(0 != m_pBoosterCore->GetCountBytesSplitPositions()) {
         m_aSplitPositionsTemp = AlignedAlloc(m_pBoosterCore->GetCountBytesSplitPositions());
         if(nullptr == m_aSplitPositionsTemp) {
            goto failed_allocation;
         }
      }

      if(0 != m_pBoosterCore->GetCountBytesTreeNodes()) {
         m_aTreeNodesTemp = AlignedAlloc(m_pBoosterCore->GetCountBytesTreeNodes());
         if(nullptr == m_aTreeNodesTemp) {
            goto failed_allocation;
         }
      }
   }

   LOG_0(Trace_Info, "Exited BoosterShell::FillAllocations");
   return Error_None;

failed_allocation:;
   LOG_0(Trace_Warning, "WARNING Exited BoosterShell::FillAllocations with allocation failure");
   return Error_OutOfMemory;
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION CreateBooster(void* rng,
      const void* dataSet,
      const BagEbm* bag,
      const double* initScores,
      IntEbm countTerms,
      const IntEbm* dimensionCounts,
      const IntEbm* featureIndexes,
      IntEbm countInnerBags,
      CreateBoosterFlags flags,
      AccelerationFlags acceleration,
      const char* objective,
      const double* experimentalParams,
      BoosterHandle* boosterHandleOut) {
   LOG_N(Trace_Info,
         "Entered CreateBooster: "
         "rng=%p, "
         "dataSet=%p, "
         "bag=%p, "
         "initScores=%p, "
         "countTerms=%" IntEbmPrintf ", "
         "dimensionCounts=%p, "
         "featureIndexes=%p, "
         "countInnerBags=%" IntEbmPrintf ", "
         "flags=0x%" UCreateBoosterFlagsPrintf ", "
         "acceleration=0x%" UAccelerationFlagsPrintf ", "
         "objective=%p, "
         "experimentalParams=%p, "
         "boosterHandleOut=%p",
         rng,
         dataSet,
         static_cast<const void*>(bag),
         static_cast<const void*>(initScores),
         countTerms,
         static_cast<const void*>(dimensionCounts),
         static_cast<const void*>(featureIndexes),
         countInnerBags,
         static_cast<UCreateBoosterFlags>(flags), // signed to unsigned conversion is defined behavior in C++
         static_cast<UAccelerationFlags>(acceleration), // signed to unsigned conversion is defined behavior in C++
         static_cast<const void*>(objective), // do not print the string for security reasons
         static_cast<const void*>(experimentalParams),
         static_cast<const void*>(boosterHandleOut));

   ErrorEbm error;

   if(nullptr == boosterHandleOut) {
      LOG_0(Trace_Error, "ERROR CreateBooster nullptr == boosterHandleOut");
      return Error_IllegalParamVal;
   }
   *boosterHandleOut = nullptr; // set this to nullptr as soon as possible so the caller doesn't attempt to free it

   if(flags &
         ~(CreateBoosterFlags_DifferentialPrivacy | CreateBoosterFlags_DisableApprox |
               CreateBoosterFlags_BinaryAsMulticlass)) {
      LOG_0(Trace_Error, "ERROR CreateBooster flags contains unknown flags. Ignoring extras.");
   }

   if(nullptr == dataSet) {
      LOG_0(Trace_Error, "ERROR CreateBooster nullptr == dataSet");
      return Error_IllegalParamVal;
   }

   if(IsConvertError<size_t>(countTerms)) {
      // the caller should not have been able to allocate memory for dimensionCounts if this wasn't fittable in size_t
      LOG_0(Trace_Error, "ERROR CreateBooster IsConvertError<size_t>(countTerms)");
      return Error_IllegalParamVal;
   }
   const size_t cTerms = static_cast<size_t>(countTerms);

   if(nullptr == dimensionCounts && size_t{0} != cTerms) {
      LOG_0(Trace_Error, "ERROR CreateBooster dimensionCounts cannot be null if 0 < countTerms");
      return Error_IllegalParamVal;
   }
   // it's legal for featureIndexes to be null if there are no features indexed by our terms
   // dimensionCounts can have zero features, so it could be legal for this to be null even if 0 < countTerms

   if(IsConvertError<size_t>(countInnerBags)) {
      // this is just a warning since the caller doesn't pass us anything material, but if it's this high
      // then our allocation would fail since it can't even in pricipal fit into memory
      LOG_0(Trace_Warning, "WARNING CreateBooster IsConvertError<size_t>(countInnerBags)");
      return Error_OutOfMemory;
   }
   const size_t cInnerBags = static_cast<size_t>(countInnerBags);

   // TODO: since BoosterCore is a non-POD C++ class, we should probably move the call to new from inside
   //       BoosterCore::Create to here and wrap it with a try catch at this level and rely on standard C++ behavior
   BoosterCore* pBoosterCore = nullptr;
   error = BoosterCore::Create(rng,
         cTerms,
         cInnerBags,
         experimentalParams,
         dimensionCounts,
         featureIndexes,
         static_cast<const unsigned char*>(dataSet),
         bag,
         initScores,
         flags,
         acceleration,
         objective,
         &pBoosterCore);
   if(UNLIKELY(Error_None != error)) {
      BoosterCore::Free(pBoosterCore); // legal if nullptr.  On error we can get back a legal pBoosterCore to delete
      return error;
   }

   BoosterShell* const pBoosterShell = BoosterShell::Create(pBoosterCore);
   if(UNLIKELY(nullptr == pBoosterShell)) {
      // if the memory allocation for pBoosterShell failed then there was no place to put the pBoosterCore, so free it
      BoosterCore::Free(pBoosterCore);
      return Error_OutOfMemory;
   }

   error = pBoosterShell->FillAllocations();
   if(Error_None != error) {
      BoosterShell::Free(pBoosterShell);
      return error;
   }

   if(size_t{0} != pBoosterCore->GetCountScores()) {
      if(!pBoosterCore->IsRmse()) {
         error = pBoosterCore->InitializeBoosterGradientsAndHessians(pBoosterShell->GetMulticlassMidwayTemp(),
               pBoosterShell->GetTermUpdate()->GetTensorScoresPointer() // initialized to zero at this point
         );
         if(UNLIKELY(Error_None != error)) {
            BoosterShell::Free(pBoosterShell);
            return error;
         }
      } else {
         InitializeRmseGradientsAndHessiansBoosting(
               static_cast<const unsigned char*>(dataSet), BagEbm{1}, bag, initScores, pBoosterCore->GetTrainingSet());
         InitializeRmseGradientsAndHessiansBoosting(static_cast<const unsigned char*>(dataSet),
               BagEbm{-1},
               bag,
               initScores,
               pBoosterCore->GetValidationSet());
      }
   }

   const BoosterHandle handle = pBoosterShell->GetHandle();

   LOG_N(Trace_Info, "Exited CreateBooster: *boosterHandleOut=%p", static_cast<void*>(handle));

   *boosterHandleOut = handle;
   return Error_None;
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION CreateBoosterView(
      BoosterHandle boosterHandle, BoosterHandle* boosterHandleViewOut) {
   LOG_N(Trace_Info,
         "Entered CreateBoosterView: "
         "boosterHandle=%p, "
         "boosterHandleViewOut=%p",
         static_cast<void*>(boosterHandle),
         static_cast<void*>(boosterHandleViewOut));

   ErrorEbm error;

   if(UNLIKELY(nullptr == boosterHandleViewOut)) {
      LOG_0(Trace_Warning, "WARNING CreateBooster nullptr == boosterHandleViewOut");
      return Error_IllegalParamVal;
   }
   *boosterHandleViewOut = nullptr; // set this as soon as possible so our caller doesn't end up freeing garbage

   BoosterShell* const pBoosterShellOriginal = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShellOriginal) {
      // already logged
      return Error_IllegalParamVal;
   }
   BoosterCore* const pBoosterCore = pBoosterShellOriginal->GetBoosterCore();

   BoosterShell* const pBoosterShellNew = BoosterShell::Create(pBoosterCore);
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
      BoosterHandle boosterHandle, IntEbm indexTerm, double* termScoresTensorOut) {
   LOG_N(Trace_Info,
         "Entered GetBestTermScores: "
         "boosterHandle=%p, "
         "indexTerm=%" IntEbmPrintf ", "
         "termScoresTensorOut=%p, ",
         static_cast<void*>(boosterHandle),
         indexTerm,
         static_cast<void*>(termScoresTensorOut));

   BoosterShell* const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamVal;
   }

   if(IsConvertError<size_t>(indexTerm)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(Trace_Error, "ERROR GetBestTermScores indexTerm is too high to index");
      return Error_IllegalParamVal;
   }
   size_t iTerm = static_cast<size_t>(indexTerm);

   BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   if(pBoosterCore->GetCountTerms() <= iTerm) {
      LOG_0(Trace_Error, "ERROR GetBestTermScores indexTerm above the number of terms that we have");
      return Error_IllegalParamVal;
   }

   if(size_t{0} == pBoosterCore->GetCountScores()) {
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.
      // If there were logits in this model, they'd all be infinity, but you could alternatively think of this
      // model as having no logits, since the number of logits can be one less than the number of target classes.

      LOG_0(Trace_Info, "Exited GetBestTermScores no scores");
      return Error_None;
   }
   EBM_ASSERT(nullptr != pBoosterCore->GetBestModel());
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());

   const Term* const pTerm = pBoosterCore->GetTerms()[iTerm];

   size_t cTensorScores = pTerm->GetCountTensorBins();
   if(size_t{0} == cTensorScores) {
      // If we have zero samples and one of the dimensions has 0 bins then there is no tensor, so return now
      // In theory it might be better to zero out the caller's tensor cells (2 ^ n_dimensions), but this condition
      // is almost an error already, so don't try reading/writing memory. We just define this situation as
      // having a zero sized tensor result. The caller can zero their own memory if they want it zero

      // if GetCountTensorBins is 0, then pBoosterCore->GetBestModel()[iTerm] does not contain valid data

      LOG_0(Trace_Warning, "WARNING GetBestTermScores feature with zero bins");
      return Error_None;
   }
   EBM_ASSERT(nullptr != pBoosterCore->GetBestModel()[iTerm]);

   if(nullptr == termScoresTensorOut) {
      LOG_0(Trace_Error, "ERROR GetBestTermScores termScoresTensorOut cannot be nullptr");
      return Error_IllegalParamVal;
   }

   Tensor* const pTensor = pBoosterCore->GetBestModel()[iTerm];
   EBM_ASSERT(nullptr != pTensor);
   EBM_ASSERT(pTensor->GetExpanded()); // the tensor should have been expanded at startup
   FloatScore* const aTermScores = pTensor->GetTensorScoresPointer();
   EBM_ASSERT(nullptr != aTermScores);

   Transpose<true>(pTerm, pBoosterCore->GetCountScores(), termScoresTensorOut, aTermScores);

   LOG_0(Trace_Info, "Exited GetBestTermScores");
   return Error_None;
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION GetCurrentTermScores(
      BoosterHandle boosterHandle, IntEbm indexTerm, double* termScoresTensorOut) {
   LOG_N(Trace_Info,
         "Entered GetCurrentTermScores: "
         "boosterHandle=%p, "
         "indexTerm=%" IntEbmPrintf ", "
         "termScoresTensorOut=%p, ",
         static_cast<void*>(boosterHandle),
         indexTerm,
         static_cast<void*>(termScoresTensorOut));

   BoosterShell* const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamVal;
   }

   if(IsConvertError<size_t>(indexTerm)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(Trace_Error, "ERROR GetCurrentTermScores indexTerm is too high to index");
      return Error_IllegalParamVal;
   }
   size_t iTerm = static_cast<size_t>(indexTerm);

   BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   if(pBoosterCore->GetCountTerms() <= iTerm) {
      LOG_0(Trace_Error, "ERROR GetCurrentTermScores indexTerm above the number of terms that we have");
      return Error_IllegalParamVal;
   }

   if(size_t{0} == pBoosterCore->GetCountScores()) {
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.
      // If there were logits in this model, they'd all be infinity, but you could alternatively think of this
      // model as having no logits, since the number of logits can be one less than the number of target classes.

      LOG_0(Trace_Info, "Exited GetCurrentTermScores no scores");
      return Error_None;
   }
   EBM_ASSERT(nullptr != pBoosterCore->GetCurrentModel());
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());

   const Term* const pTerm = pBoosterCore->GetTerms()[iTerm];

   size_t cTensorScores = pTerm->GetCountTensorBins();
   if(size_t{0} == cTensorScores) {
      // If we have zero samples and one of the dimensions has 0 bins then there is no tensor, so return now
      // In theory it might be better to zero out the caller's tensor cells (2 ^ n_dimensions), but this condition
      // is almost an error already, so don't try reading/writing memory. We just define this situation as
      // having a zero sized tensor result. The caller can zero their own memory if they want it zero

      // if GetCountTensorBins is 0, then pBoosterCore->GetCurrentModel()[iTerm] does not contain valid data

      LOG_0(Trace_Warning, "WARNING GetCurrentTermScores feature with zero bins");
      return Error_None;
   }
   EBM_ASSERT(nullptr != pBoosterCore->GetCurrentModel()[iTerm]);

   if(nullptr == termScoresTensorOut) {
      LOG_0(Trace_Error, "ERROR GetCurrentTermScores termScoresTensorOut cannot be nullptr");
      return Error_IllegalParamVal;
   }

   Tensor* const pTensor = pBoosterCore->GetCurrentModel()[iTerm];
   EBM_ASSERT(nullptr != pTensor);
   EBM_ASSERT(pTensor->GetExpanded()); // the tensor should have been expanded at startup
   FloatScore* const aTermScores = pTensor->GetTensorScoresPointer();
   EBM_ASSERT(nullptr != aTermScores);

   Transpose<true>(pTerm, pBoosterCore->GetCountScores(), termScoresTensorOut, aTermScores);

   LOG_0(Trace_Info, "Exited GetCurrentTermScores");
   return Error_None;
}

EBM_API_BODY void EBM_CALLING_CONVENTION FreeBooster(BoosterHandle boosterHandle) {
   LOG_N(Trace_Info, "Entered FreeBooster: boosterHandle=%p", static_cast<void*>(boosterHandle));

   BoosterShell* const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   // if the conversion above doesn't work, it'll return null, and our free will not in fact free any memory,
   // but it will not crash. We'll leak memory, but at least we'll log that.

   // it's legal to call free on nullptr, just like for free().  This is checked inside BoosterCore::Free()
   BoosterShell::Free(pBoosterShell);

   LOG_0(Trace_Info, "Exited FreeBooster");
}

} // namespace DEFINED_ZONE_NAME
