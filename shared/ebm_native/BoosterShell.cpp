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
      CompressibleTensor::Free(pBoosterShell->m_pSmallChangeToModelAccumulatedFromSamplingSets);
      CompressibleTensor::Free(pBoosterShell->m_pSmallChangeToModelOverwriteSingleSamplingSet);
      free(pBoosterShell->m_aThreadByteBuffer1);
      free(pBoosterShell->m_aThreadByteBuffer2);
      free(pBoosterShell->m_aSumHistogramTargetEntry);
      free(pBoosterShell->m_aSumHistogramTargetEntryLeft);
      free(pBoosterShell->m_aSumHistogramTargetEntryRight);
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
      pNew->InitializeZero();
   }

   LOG_0(TraceLevelInfo, "Exited BoosterShell::Create");

   return pNew;
}

ErrorEbmType BoosterShell::FillAllocations() {
   EBM_ASSERT(nullptr != m_pBoosterCore);

   LOG_0(TraceLevelInfo, "Entered BoosterShell::FillAllocations");

   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = m_pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
   const size_t cBytesPerItem = IsClassification(runtimeLearningTypeOrCountTargetClasses) ?
      sizeof(HistogramTargetEntry<true>) : sizeof(HistogramTargetEntry<false>);

   m_pSmallChangeToModelAccumulatedFromSamplingSets = CompressibleTensor::Allocate(k_cDimensionsMax, cVectorLength);
   if(nullptr == m_pSmallChangeToModelAccumulatedFromSamplingSets) {
      goto failed_allocation;
   }

   m_pSmallChangeToModelOverwriteSingleSamplingSet = CompressibleTensor::Allocate(k_cDimensionsMax, cVectorLength);
   if(nullptr == m_pSmallChangeToModelOverwriteSingleSamplingSet) {
      goto failed_allocation;
   }

   m_aSumHistogramTargetEntry = EbmMalloc<HistogramTargetEntryBase>(cVectorLength, cBytesPerItem);
   if(nullptr == m_aSumHistogramTargetEntry) {
      goto failed_allocation;
   }

   m_aSumHistogramTargetEntryLeft = EbmMalloc<HistogramTargetEntryBase>(cVectorLength, cBytesPerItem);
   if(nullptr == m_aSumHistogramTargetEntryLeft) {
      goto failed_allocation;
   }

   m_aSumHistogramTargetEntryRight = EbmMalloc<HistogramTargetEntryBase>(cVectorLength, cBytesPerItem);
   if(nullptr == m_aSumHistogramTargetEntryRight) {
      goto failed_allocation;
   }

   m_aTempFloatVector = EbmMalloc<FloatEbmType>(cVectorLength);
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

HistogramBucketBase * BoosterShell::GetHistogramBucketBase(size_t cBytesRequired) {
   HistogramBucketBase * aBuffer = m_aThreadByteBuffer1;
   if(UNLIKELY(m_cThreadByteBufferCapacity1 < cBytesRequired)) {
      cBytesRequired <<= 1;
      m_cThreadByteBufferCapacity1 = cBytesRequired;
      LOG_N(TraceLevelInfo, "Growing BoosterShell::ThreadByteBuffer1 to %zu", cBytesRequired);

      free(aBuffer);
      aBuffer = static_cast<HistogramBucketBase *>(EbmMalloc<void>(cBytesRequired));
      m_aThreadByteBuffer1 = aBuffer; // store it before checking it incase it's null so that we don't free old memory
      if(nullptr == aBuffer) {
         LOG_0(TraceLevelWarning, "WARNING BoosterShell::GetHistogramBucketBase OutOfMemory");
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


// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
static ErrorEbmType CreateBoosterInternal(
   const SeedEbmType randomSeed,
   const void * const dataSet,
   const IntEbmType * const bag,
   const FloatEbmType * const predictorScores,
   const IntEbmType countFeatureGroups,
   const IntEbmType * const aFeatureGroupsDimensionCounts,
   const IntEbmType * const aFeatureGroupsFeatureIndexes,
   const IntEbmType countInnerBags,
   const FloatEbmType * const optionalTempParams,
   BoosterHandle * boosterHandleOut
) {
   // TODO : give CreateBoosterInternal the same calling parameter order as CreateBooster... actually even better would be to eliminate CreateBoosterInternal

   EBM_ASSERT(nullptr != boosterHandleOut);
   EBM_ASSERT(nullptr == *boosterHandleOut);

   if(countFeatureGroups < 0) {
      LOG_0(TraceLevelError, "ERROR CreateBoosterInternal countFeatureGroups must be positive");
      return Error_IllegalParamValue;
   }
   if(0 != countFeatureGroups && nullptr == aFeatureGroupsDimensionCounts) {
      LOG_0(TraceLevelError, "ERROR CreateBoosterInternal aFeatureGroupsDimensionCounts cannot be nullptr if 0 < countFeatureGroups");
      return Error_IllegalParamValue;
   }
   // it's legal for aFeatureGroupsFeatureIndexes to be nullptr if there are no features indexed by our feature groups
   // aFeatureGroupsDimensionCounts can have zero features, so it could be legal for this to be nullptr 
   // even if 0 != countFeatureGroups

   if(nullptr == dataSet) {
      LOG_0(TraceLevelError, "ERROR CreateBoosterInternal nullptr == dataSet");
      return Error_IllegalParamValue;
   }

   if(countInnerBags < 0) {
      // 0 means use the full set (good value).  1 means make a single bag (this is useless but allowed for comparison purposes).  2+ are good numbers of bag
      LOG_0(TraceLevelError, "ERROR CreateBoosterInternal countInnerBags must be positive");
      return Error_UserParamValue;
   }
   if(IsConvertError<size_t>(countFeatureGroups)) {
      // the caller should not have been able to allocate enough memory in "aFeatureGroupsDimensionCounts" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR CreateBoosterInternal IsConvertError<size_t>(countFeatureGroups)");
      return Error_IllegalParamValue;
   }
   if(IsConvertError<size_t>(countInnerBags)) {
      // this is just a warning since the caller doesn't pass us anything material, but if it's this high
      // then our allocation would fail since it can't even in pricipal fit into memory
      LOG_0(TraceLevelWarning, "WARNING CreateBoosterInternal IsConvertError<size_t>(countInnerBags)");
      return Error_OutOfMemory;
   }

   size_t cFeatureGroups = static_cast<size_t>(countFeatureGroups);
   size_t cInnerBags = static_cast<size_t>(countInnerBags);

   BoosterShell * const pBoosterShell = BoosterShell::Create();
   if(UNLIKELY(nullptr == pBoosterShell)) {
      LOG_0(TraceLevelWarning, "WARNING CreateBoosterInternal nullptr == pBoosterShell");
      return Error_OutOfMemory;
   }

   const ErrorEbmType error1 = BoosterCore::Create(
      pBoosterShell,
      randomSeed,
      cFeatureGroups,
      cInnerBags,
      optionalTempParams,
      aFeatureGroupsDimensionCounts,
      aFeatureGroupsFeatureIndexes,
      static_cast<const unsigned char *>(dataSet),
      bag,
      predictorScores
   );
   if(UNLIKELY(Error_None != error1)) {
      BoosterShell::Free(pBoosterShell);
      LOG_0(TraceLevelWarning, "WARNING CreateBoosterInternal pBoosterCore->Initialize");
      return error1;
   }

   const ErrorEbmType error2 = pBoosterShell->FillAllocations();
   if(Error_None != error2) {
      BoosterShell::Free(pBoosterShell);
      return error2;
   }

   *boosterHandleOut = pBoosterShell->GetHandle();
   return Error_None;
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CreateBooster(
   SeedEbmType randomSeed,
   const void * dataSet,
   const IntEbmType * bag,
   const FloatEbmType * predictorScores,
   IntEbmType countFeatureGroups,
   const IntEbmType * dimensionCounts,
   const IntEbmType * featureIndexes,
   IntEbmType countInnerBags,
   const FloatEbmType * optionalTempParams,
   BoosterHandle * boosterHandleOut
) {
   LOG_N(
      TraceLevelInfo,
      "Entered CreateBooster: "
      "randomSeed=%" SeedEbmTypePrintf ", "
      "dataSet=%p, "
      "bag=%p, "
      "predictorScores=%p, "
      "countFeatureGroups=%" IntEbmTypePrintf ", "
      "dimensionCounts=%p, "
      "featureIndexes=%p, "
      "countInnerBags=%" IntEbmTypePrintf ", "
      "optionalTempParams=%p, "
      "boosterHandleOut=%p"
      ,
      randomSeed,
      static_cast<const void *>(dataSet),
      static_cast<const void *>(bag),
      static_cast<const void *>(predictorScores),
      countFeatureGroups,
      static_cast<const void *>(dimensionCounts),
      static_cast<const void *>(featureIndexes),
      countInnerBags,
      static_cast<const void *>(optionalTempParams),
      static_cast<const void *>(boosterHandleOut)
   );

   if(nullptr == boosterHandleOut) {
      LOG_0(TraceLevelError, "ERROR CreateBooster nullptr == boosterHandleOut");
      return Error_IllegalParamValue;
   }
   *boosterHandleOut = nullptr; // set this to nullptr as soon as possible so the caller doesn't attempt to free it

   const ErrorEbmType error = CreateBoosterInternal(
      randomSeed,
      dataSet,
      bag,
      predictorScores,
      countFeatureGroups,
      dimensionCounts,
      featureIndexes,
      countInnerBags,
      optionalTempParams,
      boosterHandleOut
   );

   LOG_N(TraceLevelInfo, "Exited CreateBooster: "
      "*boosterHandleOut=%p, "
      "return=%" ErrorEbmTypePrintf
      ,
      static_cast<void *>(*boosterHandleOut),
      error
   );

   return error;
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CreateBoosterView(
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

   if(UNLIKELY(nullptr == boosterHandleViewOut)) {
      LOG_0(TraceLevelWarning, "WARNING CreateBooster nullptr == boosterHandleViewOut");
      return Error_IllegalParamValue;
   }
   *boosterHandleViewOut = nullptr; // set this as soon as possible so our caller doesn't end up freeing garbage

   BoosterShell * const pBoosterShellOriginal = BoosterShell::GetBoosterShellFromBoosterHandle(boosterHandle);
   if(nullptr == pBoosterShellOriginal) {
      // already logged
      return Error_IllegalParamValue;
   }

   BoosterShell * const pBoosterShellNew = BoosterShell::Create();
   if(UNLIKELY(nullptr == pBoosterShellNew)) {
      LOG_0(TraceLevelWarning, "WARNING CreateBooster nullptr == pBoosterShellNew");
      return Error_OutOfMemory;
   }

   BoosterCore * const pBoosterCore = pBoosterShellOriginal->GetBoosterCore();
   pBoosterCore->AddReferenceCount();
   pBoosterShellNew->SetBoosterCore(pBoosterCore); // assume ownership of pBoosterCore reference count increment

   const ErrorEbmType error = pBoosterShellNew->FillAllocations();
   if(Error_None != error) {
      // TODO: we might move the call to FillAllocations to be more lazy incase the caller doesn't use it all
      BoosterShell::Free(pBoosterShellNew);
      return error;
   }

   LOG_0(TraceLevelInfo, "Exited CreateBoosterView");

   *boosterHandleViewOut = pBoosterShellNew->GetHandle();
   return Error_None;
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GetBestModelFeatureGroup(
   BoosterHandle boosterHandle,
   IntEbmType indexFeatureGroup,
   FloatEbmType * modelFeatureGroupTensorOut
) {
   LOG_N(
      TraceLevelInfo,
      "Entered GetBestModelFeatureGroup: "
      "boosterHandle=%p, "
      "indexFeatureGroup=%" IntEbmTypePrintf ", "
      "modelFeatureGroupTensorOut=%p, "
      ,
      static_cast<void *>(boosterHandle),
      indexFeatureGroup,
      static_cast<void *>(modelFeatureGroupTensorOut)
   );

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromBoosterHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamValue;
   }

   if(indexFeatureGroup < 0) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup indexFeatureGroup must be positive");
      return Error_IllegalParamValue;
   }
   if(IsConvertError<size_t>(indexFeatureGroup)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup indexFeatureGroup is too high to index");
      return Error_IllegalParamValue;
   }
   size_t iFeatureGroup = static_cast<size_t>(indexFeatureGroup);

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   if(pBoosterCore->GetCountFeatureGroups() <= iFeatureGroup) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup indexFeatureGroup above the number of feature groups that we have");
      return Error_IllegalParamValue;
   }

   if(ptrdiff_t { 0 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses() ||
      ptrdiff_t { 1 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses()) {
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.  
      // If there were logits in this model, they'd all be infinity, but you could alternatively think of this 
      // model as having no logits, since the number of logits can be one less than the number of target classes.
      LOG_0(TraceLevelInfo, "Exited GetBestModelFeatureGroup no model");
      return Error_None;
   }

   if(nullptr == modelFeatureGroupTensorOut) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup modelFeatureGroupTensorOut cannot be nullptr");
      return Error_IllegalParamValue;
   }

   // if pBoosterCore->GetFeatureGroups() is nullptr, then m_cFeatureGroups was 0, but we checked above that 
   // iFeatureGroup was less than cFeatureGroups
   EBM_ASSERT(nullptr != pBoosterCore->GetFeatureGroups());

   const FeatureGroup * const pFeatureGroup = pBoosterCore->GetFeatureGroups()[iFeatureGroup];
   const size_t cDimensions = pFeatureGroup->GetCountDimensions();
   size_t cValues = GetVectorLength(pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses());
   if(0 != cDimensions) {
      const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
      const FeatureGroupEntry * const pFeatureGroupEntryEnd = &pFeatureGroupEntry[cDimensions];
      do {
         const size_t cBins = pFeatureGroupEntry->m_pFeature->GetCountBins();
         // we've allocated this memory, so it should be reachable, so these numbers should multiply
         EBM_ASSERT(!IsMultiplyError(cValues, cBins));
         cValues *= cBins;
         ++pFeatureGroupEntry;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
   }

   // if pBoosterCore->GetBestModel() is nullptr, then either:
   //    1) m_cFeatureGroups was 0, but we checked above that iFeatureGroup was less than cFeatureGroups
   //    2) If m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0, but we checked for this above too
   EBM_ASSERT(nullptr != pBoosterCore->GetBestModel());

   CompressibleTensor * const pBestModel = pBoosterCore->GetBestModel()[iFeatureGroup];
   EBM_ASSERT(nullptr != pBestModel);
   EBM_ASSERT(pBestModel->GetExpanded()); // the model should have been expanded at startup
   FloatEbmType * const pValues = pBestModel->GetValuePointer();
   EBM_ASSERT(nullptr != pValues);

   EBM_ASSERT(!IsMultiplyError(sizeof(*pValues), cValues));
   memcpy(modelFeatureGroupTensorOut, pValues, sizeof(*pValues) * cValues);

   LOG_0(TraceLevelInfo, "Exited GetBestModelFeatureGroup");
   return Error_None;
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GetCurrentModelFeatureGroup(
   BoosterHandle boosterHandle,
   IntEbmType indexFeatureGroup,
   FloatEbmType * modelFeatureGroupTensorOut
) {
   LOG_N(
      TraceLevelInfo,
      "Entered GetCurrentModelFeatureGroup: "
      "boosterHandle=%p, "
      "indexFeatureGroup=%" IntEbmTypePrintf ", "
      "modelFeatureGroupTensorOut=%p, "
      ,
      static_cast<void *>(boosterHandle),
      indexFeatureGroup,
      static_cast<void *>(modelFeatureGroupTensorOut)
   );

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromBoosterHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamValue;
   }

   if(indexFeatureGroup < 0) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup indexFeatureGroup must be positive");
      return Error_IllegalParamValue;
   }
   if(IsConvertError<size_t>(indexFeatureGroup)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup indexFeatureGroup is too high to index");
      return Error_IllegalParamValue;
   }
   size_t iFeatureGroup = static_cast<size_t>(indexFeatureGroup);

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   if(pBoosterCore->GetCountFeatureGroups() <= iFeatureGroup) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup indexFeatureGroup above the number of feature groups that we have");
      return Error_IllegalParamValue;
   }

   if(ptrdiff_t { 0 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses() ||
      ptrdiff_t { 1 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses())    {
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.  
      // If there were logits in this model, they'd all be infinity, but you could alternatively think of this 
      // model as having no logits, since the number of logits can be one less than the number of target classes.
      LOG_0(TraceLevelInfo, "Exited GetCurrentModelFeatureGroup no model");
      return Error_None;
   }

   if(nullptr == modelFeatureGroupTensorOut) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup modelFeatureGroupTensorOut cannot be nullptr");
      return Error_IllegalParamValue;
   }

   // if pBoosterCore->GetFeatureGroups() is nullptr, then m_cFeatureGroups was 0, but we checked above that 
   // iFeatureGroup was less than cFeatureGroups
   EBM_ASSERT(nullptr != pBoosterCore->GetFeatureGroups());

   const FeatureGroup * const pFeatureGroup = pBoosterCore->GetFeatureGroups()[iFeatureGroup];
   const size_t cDimensions = pFeatureGroup->GetCountDimensions();
   size_t cValues = GetVectorLength(pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses());
   if(0 != cDimensions) {
      const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
      const FeatureGroupEntry * const pFeatureGroupEntryEnd = &pFeatureGroupEntry[cDimensions];
      do {
         const size_t cBins = pFeatureGroupEntry->m_pFeature->GetCountBins();
         // we've allocated this memory, so it should be reachable, so these numbers should multiply
         EBM_ASSERT(!IsMultiplyError(cValues, cBins));
         cValues *= cBins;
         ++pFeatureGroupEntry;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
   }

   // if pBoosterCore->GetCurrentModel() is nullptr, then either:
   //    1) m_cFeatureGroups was 0, but we checked above that iFeatureGroup was less than cFeatureGroups
   //    2) If m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0, but we checked for this above too
   EBM_ASSERT(nullptr != pBoosterCore->GetCurrentModel());

   CompressibleTensor * const pCurrentModel = pBoosterCore->GetCurrentModel()[iFeatureGroup];
   EBM_ASSERT(nullptr != pCurrentModel);
   EBM_ASSERT(pCurrentModel->GetExpanded()); // the model should have been expanded at startup
   FloatEbmType * const pValues = pCurrentModel->GetValuePointer();
   EBM_ASSERT(nullptr != pValues);

   EBM_ASSERT(!IsMultiplyError(sizeof(*pValues), cValues));
   memcpy(modelFeatureGroupTensorOut, pValues, sizeof(*pValues) * cValues);

   LOG_0(TraceLevelInfo, "Exited GetCurrentModelFeatureGroup");
   return Error_None;
}

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION FreeBooster(
   BoosterHandle boosterHandle
) {
   LOG_N(TraceLevelInfo, "Entered FreeBooster: boosterHandle=%p", static_cast<void *>(boosterHandle));

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromBoosterHandle(boosterHandle);
   // if the conversion above doesn't work, it'll return null, and our free will not in fact free any memory,
   // but it will not crash. We'll leak memory, but at least we'll log that.

   // it's legal to call free on nullptr, just like for free().  This is checked inside BoosterCore::Free()
   BoosterShell::Free(pBoosterShell);

   LOG_0(TraceLevelInfo, "Exited FreeBooster");
}

} // DEFINED_ZONE_NAME
