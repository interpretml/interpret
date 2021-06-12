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

#include "SegmentedTensor.hpp"

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
      SegmentedTensor::Free(pBoosterShell->m_pSmallChangeToModelAccumulatedFromSamplingSets);
      SegmentedTensor::Free(pBoosterShell->m_pSmallChangeToModelOverwriteSingleSamplingSet);
      free(pBoosterShell->m_aThreadByteBuffer1);
      free(pBoosterShell->m_aThreadByteBuffer2);
      free(pBoosterShell->m_aSumHistogramTargetEntry);
      free(pBoosterShell->m_aSumHistogramTargetEntryLeft);
      free(pBoosterShell->m_aSumHistogramTargetEntryRight);
      free(pBoosterShell->m_aTempFloatVector);
      free(pBoosterShell->m_aEquivalentSplits);
      BoosterCore::Free(pBoosterShell->m_pBoosterCore);

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

   m_pSmallChangeToModelAccumulatedFromSamplingSets = SegmentedTensor::Allocate(k_cDimensionsMax, cVectorLength);
   if(nullptr == m_pSmallChangeToModelAccumulatedFromSamplingSets) {
      goto failed_allocation;
   }

   m_pSmallChangeToModelOverwriteSingleSamplingSet = SegmentedTensor::Allocate(k_cDimensionsMax, cVectorLength);
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

HistogramBucketBase * BoosterShell::GetHistogramBucketBase(const size_t cBytesRequired) {
   HistogramBucketBase * aBuffer = m_aThreadByteBuffer1;
   if(UNLIKELY(m_cThreadByteBufferCapacity1 < cBytesRequired)) {
      m_cThreadByteBufferCapacity1 = cBytesRequired << 1;
      LOG_N(TraceLevelInfo, "Growing BoosterShell::ThreadByteBuffer1 to %zu", m_cThreadByteBufferCapacity1);

      free(aBuffer);
      aBuffer = static_cast<HistogramBucketBase *>(EbmMalloc<void>(m_cThreadByteBufferCapacity1));
      m_aThreadByteBuffer1 = aBuffer;
   }
   return aBuffer;
}

bool BoosterShell::GrowThreadByteBuffer2(const size_t cByteBoundaries) {
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
   m_aThreadByteBuffer2 = aBuffer;
   if(UNLIKELY(nullptr == aBuffer)) {
      return true;
   }
   return false;
}


// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
static BoosterHandle CreateBooster(
   const SeedEbmType randomSeed,
   const IntEbmType countFeatures,
   const BoolEbmType * const aFeaturesCategorical,
   const IntEbmType * const aFeaturesBinCount,
   const IntEbmType countFeatureGroups,
   const IntEbmType * const aFeatureGroupsDimensionCount,
   const IntEbmType * const aFeatureGroupsFeatureIndexes,
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const IntEbmType countTrainingSamples,
   const void * const trainingTargets,
   const IntEbmType * const trainingBinnedData,
   const FloatEbmType * const aTrainingWeights,
   const FloatEbmType * const trainingPredictorScores,
   const IntEbmType countValidationSamples,
   const void * const validationTargets,
   const IntEbmType * const validationBinnedData,
   const FloatEbmType * const aValidationWeights,
   const FloatEbmType * const validationPredictorScores,
   const IntEbmType countInnerBags,
   const FloatEbmType * const optionalTempParams
) {
   // TODO : give CreateBooster the same calling parameter order as CreateClassificationBooster

   if(countFeatures < 0) {
      LOG_0(TraceLevelError, "ERROR CreateBooster countFeatures must be positive");
      return nullptr;
   }
   if(0 != countFeatures && nullptr == aFeaturesCategorical) {
      // TODO: in the future maybe accept null aFeaturesCategorical and assume there are no missing values
      LOG_0(TraceLevelError, "ERROR CreateBooster aFeaturesCategorical cannot be nullptr if 0 < countFeatures");
      return nullptr;
   }
   if(0 != countFeatures && nullptr == aFeaturesBinCount) {
      LOG_0(TraceLevelError, "ERROR CreateBooster aFeaturesBinCount cannot be nullptr if 0 < countFeatures");
      return nullptr;
   }
   if(countFeatureGroups < 0) {
      LOG_0(TraceLevelError, "ERROR CreateBooster countFeatureGroups must be positive");
      return nullptr;
   }
   if(0 != countFeatureGroups && nullptr == aFeatureGroupsDimensionCount) {
      LOG_0(TraceLevelError, "ERROR CreateBooster aFeatureGroupsDimensionCount cannot be nullptr if 0 < countFeatureGroups");
      return nullptr;
   }
   // aFeatureGroupsFeatureIndexes -> it's legal for aFeatureGroupsFeatureIndexes to be nullptr if there are no features indexed by our featureGroups.  
   // FeatureGroups can have zero features, so it could be legal for this to be null even if there are aFeatureGroupsDimensionCount
   if(countTrainingSamples < 0) {
      LOG_0(TraceLevelError, "ERROR CreateBooster countTrainingSamples must be positive");
      return nullptr;
   }
   if(0 != countTrainingSamples && nullptr == trainingTargets) {
      LOG_0(TraceLevelError, "ERROR CreateBooster trainingTargets cannot be nullptr if 0 < countTrainingSamples");
      return nullptr;
   }
   if(0 != countTrainingSamples && 0 != countFeatures && nullptr == trainingBinnedData) {
      LOG_0(TraceLevelError, "ERROR CreateBooster trainingBinnedData cannot be nullptr if 0 < countTrainingSamples AND 0 < countFeatures");
      return nullptr;
   }
   if(0 != countTrainingSamples && nullptr == trainingPredictorScores) {
      LOG_0(TraceLevelError, "ERROR CreateBooster trainingPredictorScores cannot be nullptr if 0 < countTrainingSamples");
      return nullptr;
   }
   if(countValidationSamples < 0) {
      LOG_0(TraceLevelError, "ERROR CreateBooster countValidationSamples must be positive");
      return nullptr;
   }
   if(0 != countValidationSamples && nullptr == validationTargets) {
      LOG_0(TraceLevelError, "ERROR CreateBooster validationTargets cannot be nullptr if 0 < countValidationSamples");
      return nullptr;
   }
   if(0 != countValidationSamples && 0 != countFeatures && nullptr == validationBinnedData) {
      LOG_0(TraceLevelError, "ERROR CreateBooster validationBinnedData cannot be nullptr if 0 < countValidationSamples AND 0 < countFeatures");
      return nullptr;
   }
   if(0 != countValidationSamples && nullptr == validationPredictorScores) {
      LOG_0(TraceLevelError, "ERROR CreateBooster validationPredictorScores cannot be nullptr if 0 < countValidationSamples");
      return nullptr;
   }
   if(countInnerBags < 0) {
      // 0 means use the full set (good value).  1 means make a single bag (this is useless but allowed for comparison purposes).  2+ are good numbers of bag
      LOG_0(TraceLevelError, "ERROR CreateBooster countInnerBags must be positive");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t>(countFeatures)) {
      // the caller should not have been able to allocate enough memory in "features" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR CreateBooster !IsNumberConvertable<size_t>(countFeatures)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t>(countFeatureGroups)) {
      // the caller should not have been able to allocate enough memory in "aFeatureGroupsDimensionCount" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR CreateBooster !IsNumberConvertable<size_t>(countFeatureGroups)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t>(countTrainingSamples)) {
      // the caller should not have been able to allocate enough memory in "trainingTargets" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR CreateBooster !IsNumberConvertable<size_t>(countTrainingSamples)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t>(countValidationSamples)) {
      // the caller should not have been able to allocate enough memory in "validationTargets" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR CreateBooster !IsNumberConvertable<size_t>(countValidationSamples)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t>(countInnerBags)) {
      // this is just a warning since the caller doesn't pass us anything material, but if it's this high
      // then our allocation would fail since it can't even in pricipal fit into memory
      LOG_0(TraceLevelWarning, "WARNING CreateBooster !IsNumberConvertable<size_t>(countInnerBags)");
      return nullptr;
   }

   size_t cFeatures = static_cast<size_t>(countFeatures);
   size_t cFeatureGroups = static_cast<size_t>(countFeatureGroups);
   size_t cTrainingSamples = static_cast<size_t>(countTrainingSamples);
   size_t cValidationSamples = static_cast<size_t>(countValidationSamples);
   size_t cInnerBags = static_cast<size_t>(countInnerBags);

   size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   if(IsMultiplyError(cVectorLength, cTrainingSamples)) {
      // the caller should not have been able to allocate enough memory in "trainingPredictorScores" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR CreateBooster IsMultiplyError(cVectorLength, cTrainingSamples)");
      return nullptr;
   }
   if(IsMultiplyError(cVectorLength, cValidationSamples)) {
      // the caller should not have been able to allocate enough memory in "validationPredictorScores" if this didn't fit in memory
      LOG_0(TraceLevelError, "ERROR CreateBooster IsMultiplyError(cVectorLength, cValidationSamples)");
      return nullptr;
   }

   BoosterShell * const pBoosterShell = BoosterShell::Create();
   if(UNLIKELY(nullptr == pBoosterShell)) {
      LOG_0(TraceLevelWarning, "WARNING CreateBooster nullptr == pBoosterShell");
      return nullptr;
   }

   // TODO: pass in the pBoosterShell so that BoosterCore can immediately attach itself to the pBoosterShell
   //       this is important in R and other languages that might want to exit with longjump because we can attach
   //       the pBoosterShell object to a managed destructor that'll clean up all our memory allocations
   BoosterCore * const pBoosterCore = BoosterCore::Create(
      randomSeed,
      runtimeLearningTypeOrCountTargetClasses,
      cFeatures,
      cFeatureGroups,
      cInnerBags,
      optionalTempParams,
      aFeaturesCategorical,
      aFeaturesBinCount,
      aFeatureGroupsDimensionCount,
      aFeatureGroupsFeatureIndexes,
      cTrainingSamples,
      trainingTargets,
      trainingBinnedData,
      aTrainingWeights,
      trainingPredictorScores,
      cValidationSamples,
      validationTargets,
      validationBinnedData,
      aValidationWeights,
      validationPredictorScores
   );
   if(UNLIKELY(nullptr == pBoosterCore)) {
      BoosterShell::Free(pBoosterShell);
      LOG_0(TraceLevelWarning, "WARNING CreateBooster pBoosterCore->Initialize");
      return nullptr;
   }

   pBoosterShell->SetBoosterCore(pBoosterCore); // assume ownership of pBoosterCore

   if(Error_None != pBoosterShell->FillAllocations()) {
      // don't free the pBoosterCore since pBoosterShell now owns it
      BoosterShell::Free(pBoosterShell);
      return nullptr;
   }

   return reinterpret_cast<BoosterHandle>(pBoosterShell);
}

EBM_NATIVE_IMPORT_EXPORT_BODY BoosterHandle EBM_NATIVE_CALLING_CONVENTION CreateClassificationBooster(
   SeedEbmType randomSeed,
   IntEbmType countTargetClasses,
   IntEbmType countFeatures,
   const BoolEbmType * featuresCategorical,
   const IntEbmType * featuresBinCount,
   IntEbmType countFeatureGroups,
   const IntEbmType * featureGroupsDimensionCount,
   const IntEbmType * featureGroupsFeatureIndexes,
   IntEbmType countTrainingSamples,
   const IntEbmType * trainingBinnedData,
   const IntEbmType * trainingTargets,
   const FloatEbmType * trainingWeights,
   const FloatEbmType * trainingPredictorScores,
   IntEbmType countValidationSamples,
   const IntEbmType * validationBinnedData,
   const IntEbmType * validationTargets,
   const FloatEbmType * validationWeights,
   const FloatEbmType * validationPredictorScores,
   IntEbmType countInnerBags,
   const FloatEbmType * optionalTempParams
) {
   LOG_N(
      TraceLevelInfo,
      "Entered CreateClassificationBooster: "
      "randomSeed=%" SeedEbmTypePrintf ", "
      "countTargetClasses=%" IntEbmTypePrintf ", "
      "countFeatures=%" IntEbmTypePrintf ", "
      "featuresCategorical=%p, "
      "featuresBinCount=%p, "
      "countFeatureGroups=%" IntEbmTypePrintf ", "
      "featureGroupsDimensionCount=%p, "
      "featureGroupsFeatureIndexes=%p, "
      "countTrainingSamples=%" IntEbmTypePrintf ", "
      "trainingBinnedData=%p, "
      "trainingTargets=%p, "
      "trainingWeights=%p, "
      "trainingPredictorScores=%p, "
      "countValidationSamples=%" IntEbmTypePrintf ", "
      "validationBinnedData=%p, "
      "validationTargets=%p, "
      "validationWeights=%p, "
      "validationPredictorScores=%p, "
      "countInnerBags=%" IntEbmTypePrintf ", "
      "optionalTempParams=%p"
      ,
      randomSeed,
      countTargetClasses,
      countFeatures,
      static_cast<const void *>(featuresCategorical),
      static_cast<const void *>(featuresBinCount),
      countFeatureGroups,
      static_cast<const void *>(featureGroupsDimensionCount),
      static_cast<const void *>(featureGroupsFeatureIndexes),
      countTrainingSamples,
      static_cast<const void *>(trainingBinnedData),
      static_cast<const void *>(trainingTargets),
      static_cast<const void *>(trainingWeights),
      static_cast<const void *>(trainingPredictorScores),
      countValidationSamples,
      static_cast<const void *>(validationBinnedData),
      static_cast<const void *>(validationTargets),
      static_cast<const void *>(validationWeights),
      static_cast<const void *>(validationPredictorScores),
      countInnerBags,
      static_cast<const void *>(optionalTempParams)
   );
   if(countTargetClasses < 0) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster countTargetClasses can't be negative");
      return nullptr;
   }
   if(0 == countTargetClasses && (0 != countTrainingSamples || 0 != countValidationSamples)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster countTargetClasses can't be zero unless there are no training and no validation cases");
      return nullptr;
   }
   if(!IsNumberConvertable<ptrdiff_t>(countTargetClasses)) {
      LOG_0(TraceLevelWarning, "WARNING CreateClassificationBooster !IsNumberConvertable<ptrdiff_t>(countTargetClasses)");
      return nullptr;
   }
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = static_cast<ptrdiff_t>(countTargetClasses);
   const BoosterHandle boosterHandle = CreateBooster(
      randomSeed,
      countFeatures,
      featuresCategorical,
      featuresBinCount,
      countFeatureGroups,
      featureGroupsDimensionCount,
      featureGroupsFeatureIndexes,
      runtimeLearningTypeOrCountTargetClasses,
      countTrainingSamples,
      trainingTargets,
      trainingBinnedData,
      trainingWeights,
      trainingPredictorScores,
      countValidationSamples,
      validationTargets,
      validationBinnedData,
      validationWeights,
      validationPredictorScores,
      countInnerBags,
      optionalTempParams
   );
   LOG_N(TraceLevelInfo, "Exited CreateClassificationBooster %p", static_cast<void *>(boosterHandle));
   return boosterHandle;
}

EBM_NATIVE_IMPORT_EXPORT_BODY BoosterHandle EBM_NATIVE_CALLING_CONVENTION CreateRegressionBooster(
   SeedEbmType randomSeed,
   IntEbmType countFeatures,
   const BoolEbmType * featuresCategorical,
   const IntEbmType * featuresBinCount,
   IntEbmType countFeatureGroups,
   const IntEbmType * featureGroupsDimensionCount,
   const IntEbmType * featureGroupsFeatureIndexes,
   IntEbmType countTrainingSamples,
   const IntEbmType * trainingBinnedData,
   const FloatEbmType * trainingTargets,
   const FloatEbmType * trainingWeights,
   const FloatEbmType * trainingPredictorScores,
   IntEbmType countValidationSamples,
   const IntEbmType * validationBinnedData,
   const FloatEbmType * validationTargets,
   const FloatEbmType * validationWeights,
   const FloatEbmType * validationPredictorScores,
   IntEbmType countInnerBags,
   const FloatEbmType * optionalTempParams
) {
   LOG_N(
      TraceLevelInfo,
      "Entered CreateRegressionBooster: "
      "randomSeed=%" SeedEbmTypePrintf ", "
      "countFeatures=%" IntEbmTypePrintf ", "
      "featuresCategorical=%p, "
      "featuresBinCount=%p, "
      "countFeatureGroups=%" IntEbmTypePrintf ", "
      "featureGroupsDimensionCount=%p, "
      "featureGroupsFeatureIndexes=%p, "
      "countTrainingSamples=%" IntEbmTypePrintf ", "
      "trainingBinnedData=%p, "
      "trainingTargets=%p, "
      "trainingWeights=%p, "
      "trainingPredictorScores=%p, "
      "countValidationSamples=%" IntEbmTypePrintf ", "
      "validationBinnedData=%p, "
      "validationTargets=%p, "
      "validationWeights=%p, "
      "validationPredictorScores=%p, "
      "countInnerBags=%" IntEbmTypePrintf ", "
      "optionalTempParams=%p"
      ,
      randomSeed,
      countFeatures,
      static_cast<const void *>(featuresCategorical),
      static_cast<const void *>(featuresBinCount),
      countFeatureGroups,
      static_cast<const void *>(featureGroupsDimensionCount),
      static_cast<const void *>(featureGroupsFeatureIndexes),
      countTrainingSamples,
      static_cast<const void *>(trainingBinnedData),
      static_cast<const void *>(trainingTargets),
      static_cast<const void *>(trainingWeights),
      static_cast<const void *>(trainingPredictorScores),
      countValidationSamples,
      static_cast<const void *>(validationBinnedData),
      static_cast<const void *>(validationTargets),
      static_cast<const void *>(validationWeights),
      static_cast<const void *>(validationPredictorScores),
      countInnerBags,
      static_cast<const void *>(optionalTempParams)
   );
   const BoosterHandle boosterHandle = CreateBooster(
      randomSeed,
      countFeatures,
      featuresCategorical,
      featuresBinCount,
      countFeatureGroups,
      featureGroupsDimensionCount,
      featureGroupsFeatureIndexes,
      k_regression,
      countTrainingSamples,
      trainingTargets,
      trainingBinnedData,
      trainingWeights,
      trainingPredictorScores,
      countValidationSamples,
      validationTargets,
      validationBinnedData,
      validationWeights,
      validationPredictorScores,
      countInnerBags,
      optionalTempParams
   );
   LOG_N(TraceLevelInfo, "Exited CreateRegressionBooster %p", static_cast<void *>(boosterHandle));
   return boosterHandle;
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GetBestModelFeatureGroup(
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
      modelFeatureGroupTensorOut
   );

   BoosterShell * const pBoosterShell = reinterpret_cast<BoosterShell *>(boosterHandle);
   if(nullptr == pBoosterShell) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup boosterHandle cannot be nullptr");
      return IntEbmType { 1 };
   }
   if(indexFeatureGroup < 0) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup indexFeatureGroup must be positive");
      return IntEbmType { 1 };
   }
   if(!IsNumberConvertable<size_t>(indexFeatureGroup)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup indexFeatureGroup is too high to index");
      return IntEbmType { 1 };
   }
   size_t iFeatureGroup = static_cast<size_t>(indexFeatureGroup);

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   if(pBoosterCore->GetCountFeatureGroups() <= iFeatureGroup) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup indexFeatureGroup above the number of feature groups that we have");
      return IntEbmType { 1 };
   }

   if(ptrdiff_t { 0 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses() ||
      ptrdiff_t { 1 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses()) {
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.  
      // If there were logits in this model, they'd all be infinity, but you could alternatively think of this 
      // model as having no logits, since the number of logits can be one less than the number of target classes.
      LOG_0(TraceLevelInfo, "Exited GetBestModelFeatureGroup no model");
      return IntEbmType { 0 };
   }

   if(nullptr == modelFeatureGroupTensorOut) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup modelFeatureGroupTensorOut cannot be nullptr");
      return IntEbmType { 1 };
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
         EBM_ASSERT(!IsMultiplyError(cBins, cValues));
         cValues *= cBins;
         ++pFeatureGroupEntry;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
   }

   // if pBoosterCore->GetBestModel() is nullptr, then either:
   //    1) m_cFeatureGroups was 0, but we checked above that iFeatureGroup was less than cFeatureGroups
   //    2) If m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0, but we checked for this above too
   EBM_ASSERT(nullptr != pBoosterCore->GetBestModel());

   SegmentedTensor * const pBestModel = pBoosterCore->GetBestModel()[iFeatureGroup];
   EBM_ASSERT(nullptr != pBestModel);
   EBM_ASSERT(pBestModel->GetExpanded()); // the model should have been expanded at startup
   FloatEbmType * const pValues = pBestModel->GetValuePointer();
   EBM_ASSERT(nullptr != pValues);

   EBM_ASSERT(!IsMultiplyError(sizeof(*pValues), cValues));
   memcpy(modelFeatureGroupTensorOut, pValues, sizeof(*pValues) * cValues);

   LOG_0(TraceLevelInfo, "Exited GetBestModelFeatureGroup");
   return IntEbmType { 0 };
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GetCurrentModelFeatureGroup(
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
      modelFeatureGroupTensorOut
   );

   BoosterShell * const pBoosterShell = reinterpret_cast<BoosterShell *>(boosterHandle);
   if(nullptr == pBoosterShell) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup boosterHandle cannot be nullptr");
      return IntEbmType { 1 };
   }
   if(indexFeatureGroup < 0) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup indexFeatureGroup must be positive");
      return IntEbmType { 1 };
   }
   if(!IsNumberConvertable<size_t>(indexFeatureGroup)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup indexFeatureGroup is too high to index");
      return IntEbmType { 1 };
   }
   size_t iFeatureGroup = static_cast<size_t>(indexFeatureGroup);

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   if(pBoosterCore->GetCountFeatureGroups() <= iFeatureGroup) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup indexFeatureGroup above the number of feature groups that we have");
      return IntEbmType { 1 };
   }

   if(ptrdiff_t { 0 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses() ||
      ptrdiff_t { 1 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses())    {
      // for classification, if there is only 1 possible target class, then the probability of that class is 100%.  
      // If there were logits in this model, they'd all be infinity, but you could alternatively think of this 
      // model as having no logits, since the number of logits can be one less than the number of target classes.
      LOG_0(TraceLevelInfo, "Exited GetCurrentModelFeatureGroup no model");
      return IntEbmType { 0 };
   }

   if(nullptr == modelFeatureGroupTensorOut) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup modelFeatureGroupTensorOut cannot be nullptr");
      return IntEbmType { 1 };
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
         EBM_ASSERT(!IsMultiplyError(cBins, cValues));
         cValues *= cBins;
         ++pFeatureGroupEntry;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
   }

   // if pBoosterCore->GetCurrentModel() is nullptr, then either:
   //    1) m_cFeatureGroups was 0, but we checked above that iFeatureGroup was less than cFeatureGroups
   //    2) If m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0, but we checked for this above too
   EBM_ASSERT(nullptr != pBoosterCore->GetCurrentModel());

   SegmentedTensor * const pCurrentModel = pBoosterCore->GetCurrentModel()[iFeatureGroup];
   EBM_ASSERT(nullptr != pCurrentModel);
   EBM_ASSERT(pCurrentModel->GetExpanded()); // the model should have been expanded at startup
   FloatEbmType * const pValues = pCurrentModel->GetValuePointer();
   EBM_ASSERT(nullptr != pValues);

   EBM_ASSERT(!IsMultiplyError(sizeof(*pValues), cValues));
   memcpy(modelFeatureGroupTensorOut, pValues, sizeof(*pValues) * cValues);

   LOG_0(TraceLevelInfo, "Exited GetCurrentModelFeatureGroup");
   return IntEbmType { 0 };
}

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION FreeBooster(
   BoosterHandle boosterHandle
) {
   LOG_N(TraceLevelInfo, "Entered FreeBooster: boosterHandle=%p", static_cast<void *>(boosterHandle));

   BoosterShell * const pBoosterShell = reinterpret_cast<BoosterShell *>(boosterHandle);

   // it's legal to call free on nullptr, just like for free().  This is checked inside BoosterCore::Free()
   BoosterShell::Free(pBoosterShell);

   LOG_0(TraceLevelInfo, "Exited FreeBooster");
}

} // DEFINED_ZONE_NAME
