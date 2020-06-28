// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DIMENSION_SINGLE_H
#define DIMENSION_SINGLE_H

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy
#include <vector>
#include <queue>

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "SegmentedTensor.h"
#include "EbmStatisticUtils.h"
#include "CachedThreadResourcesBoosting.h"
#include "Feature.h"
#include "FeatureGroup.h"
#include "SamplingSet.h"
#include "HistogramTargetEntry.h"
#include "HistogramBucket.h"

void BinBoosting(
   const bool bUseSIMD,
   EbmBoostingState * const pEbmBoostingState,
   const FeatureCombination * const pFeatureCombination,
   const SamplingSet * const pTrainingSet,
   HistogramBucketBase * const aHistogramBucketBase
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
);

void SumHistogramBuckets(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const size_t cHistogramBuckets,
   const HistogramBucketBase * const aHistogramBucketsBase,
   HistogramBucketVectorEntryBase * const aSumHistogramBucketVectorEntryBase
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
   , const size_t cInstancesTotal
#endif // NDEBUG
);

bool GrowDecisionTree(
   EbmBoostingState * const pEbmBoostingState,
   const size_t cHistogramBuckets,
   const HistogramBucketBase * const aHistogramBucketBase,
   const size_t cInstancesTotal,
   const HistogramBucketVectorEntryBase * const aSumHistogramBucketVectorEntryBase,
   const size_t cTreeSplitsMax,
   const size_t cInstancesRequiredForChildSplitMin,
   SegmentedTensor * const pSmallChangeToModelOverwriteSingleSamplingSet,
   FloatEbmType * const pTotalGain
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
);


EBM_INLINE bool BoostZeroDimensional(
   EbmBoostingState * const pEbmBoostingState,
   const SamplingSet * const pTrainingSet,
   SegmentedTensor * const pSmallChangeToModelOverwriteSingleSamplingSet
) {
   LOG_0(TraceLevelVerbose, "Entered BoostZeroDimensional");

   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
   if(GetHistogramBucketSizeOverflow(bClassification, cVectorLength)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)");
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);

   CachedBoostingThreadResources * const pCachedThreadResources = pEbmBoostingState->GetCachedThreadResources();

   HistogramBucketBase * const pHistogramBucket = 
      pCachedThreadResources->GetThreadByteBuffer1(cBytesPerHistogramBucket);
   
   if(UNLIKELY(nullptr == pHistogramBucket)) {
      LOG_0(TraceLevelWarning, "WARNING nullptr == pHistogramBucket");
      return true;
   }

   if(bClassification) {
      pHistogramBucket->GetHistogramBucket<true>()->Zero(cVectorLength);
   } else {
      pHistogramBucket->GetHistogramBucket<false>()->Zero(cVectorLength);
   }

   BinBoosting(
      false,
      pEbmBoostingState,
      nullptr,
      pTrainingSet,
      pHistogramBucket
#ifndef NDEBUG
      , nullptr
#endif // NDEBUG
   );

   FloatEbmType * aValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
   if(bClassification) {
      const HistogramBucketVectorEntry<true> * const aSumHistogramBucketVectorEntry =
         ArrayToPointer(pHistogramBucket->GetHistogramBucket<true>()->m_aHistogramBucketVectorEntry);
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         const FloatEbmType smallChangeToModel = EbmStatistics::ComputeSmallChangeForOneSegmentClassificationLogOdds(
            aSumHistogramBucketVectorEntry[iVector].m_sumResidualError, 
            aSumHistogramBucketVectorEntry[iVector].GetSumDenominator()
         );
         aValues[iVector] = smallChangeToModel;
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      const HistogramBucketVectorEntry<false> * const aSumHistogramBucketVectorEntry =
         ArrayToPointer(pHistogramBucket->GetHistogramBucket<false>()->m_aHistogramBucketVectorEntry);
      const FloatEbmType smallChangeToModel = EbmStatistics::ComputeSmallChangeForOneSegmentRegression(
         aSumHistogramBucketVectorEntry[0].m_sumResidualError, 
         static_cast<FloatEbmType>(pHistogramBucket->GetHistogramBucket<false>()->m_cInstancesInBucket)
      );
      aValues[0] = smallChangeToModel;
   }

   LOG_0(TraceLevelVerbose, "Exited BoostZeroDimensional");
   return false;
}

EBM_INLINE bool BoostSingleDimensional(
   EbmBoostingState * const pEbmBoostingState,
   const FeatureCombination * const pFeatureCombination,
   const SamplingSet * const pTrainingSet,
   const size_t cTreeSplitsMax, 
   const size_t cInstancesRequiredForChildSplitMin, 
   SegmentedTensor * const pSmallChangeToModelOverwriteSingleSamplingSet, 
   FloatEbmType * const pTotalGain
) {
   LOG_0(TraceLevelVerbose, "Entered BoostSingleDimensional");

   EBM_ASSERT(1 == pFeatureCombination->GetCountFeatures());
   size_t cTotalBuckets = pFeatureCombination->GetFeatureCombinationEntries()[0].m_pFeature->GetCountBins();

   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);
   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
   if(GetHistogramBucketSizeOverflow(bClassification, cVectorLength)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "WARNING GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)");
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);
   if(IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "WARNING IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)");
      return true;
   }
   const size_t cBytesBuffer = cTotalBuckets * cBytesPerHistogramBucket;

   CachedBoostingThreadResources * const pCachedThreadResources = pEbmBoostingState->GetCachedThreadResources();

   HistogramBucketBase * const aHistogramBuckets = pCachedThreadResources->GetThreadByteBuffer1(cBytesBuffer);
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING BoostSingleDimensional nullptr == aHistogramBuckets");
      return true;
   }

   HistogramBucketVectorEntryBase * const aSumHistogramBucketVectorEntry =
      pCachedThreadResources->GetSumHistogramBucketVectorEntryArray();

   if(bClassification) {
      HistogramBucket<true> * const aHistogramBucketsLocal = aHistogramBuckets->GetHistogramBucket<true>();
      for(size_t i = 0; i < cTotalBuckets; ++i) {
         HistogramBucket<true> * const pHistogramBucket =
            GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBucketsLocal, i);
         pHistogramBucket->Zero(cVectorLength);
      }

      HistogramBucketVectorEntry<true> * const aSumHistogramBucketVectorEntryLocal = aSumHistogramBucketVectorEntry->GetHistogramBucketVectorEntry<true>();
      for(size_t i = 0; i < cVectorLength; ++i) {
         aSumHistogramBucketVectorEntryLocal[i].Zero();
      }
   } else {
      HistogramBucket<false> * const aHistogramBucketsLocal = aHistogramBuckets->GetHistogramBucket<false>();
      for(size_t i = 0; i < cTotalBuckets; ++i) {
         HistogramBucket<false> * const pHistogramBucket =
            GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBucketsLocal, i);
         pHistogramBucket->Zero(cVectorLength);
      }

      HistogramBucketVectorEntry<false> * const aSumHistogramBucketVectorEntryLocal = aSumHistogramBucketVectorEntry->GetHistogramBucketVectorEntry<false>();
      for(size_t i = 0; i < cVectorLength; ++i) {
         aSumHistogramBucketVectorEntryLocal[i].Zero();
      }
   }

#ifndef NDEBUG
   const unsigned char * const aHistogramBucketsEndDebug = reinterpret_cast<unsigned char *>(aHistogramBuckets) + cBytesBuffer;
#endif // NDEBUG

   BinBoosting(
      false,
      pEbmBoostingState,
      pFeatureCombination,
      pTrainingSet,
      aHistogramBuckets
#ifndef NDEBUG
      , aHistogramBucketsEndDebug
#endif // NDEBUG
   );

   size_t cHistogramBuckets = pFeatureCombination->GetFeatureCombinationEntries()[0].m_pFeature->GetCountBins();
   // dimensions with 1 bin don't contribute anything since they always have the same value, 
   // so we pre-filter these out and handle them separately
   EBM_ASSERT(2 <= cHistogramBuckets);
   SumHistogramBuckets(
      runtimeLearningTypeOrCountTargetClasses,
      cHistogramBuckets,
      aHistogramBuckets, 
      aSumHistogramBucketVectorEntry
#ifndef NDEBUG
      , aHistogramBucketsEndDebug
      , pTrainingSet->GetTotalCountInstanceOccurrences()
#endif // NDEBUG
   );

   const size_t cInstancesTotal = pTrainingSet->GetTotalCountInstanceOccurrences();
   EBM_ASSERT(1 <= cInstancesTotal);

   bool bRet = GrowDecisionTree(
      pEbmBoostingState,
      cHistogramBuckets, 
      aHistogramBuckets, 
      cInstancesTotal, 
      aSumHistogramBucketVectorEntry, 
      cTreeSplitsMax, 
      cInstancesRequiredForChildSplitMin, 
      pSmallChangeToModelOverwriteSingleSamplingSet, 
      pTotalGain
#ifndef NDEBUG
      , aHistogramBucketsEndDebug
#endif // NDEBUG
   );

   LOG_0(TraceLevelVerbose, "Exited BoostSingleDimensional");
   return bRet;
}

#endif // DIMENSION_SINGLE_H
