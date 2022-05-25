// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "CompressibleTensor.hpp"
#include "ebm_stats.hpp"

#include "Feature.hpp"
#include "FeatureGroup.hpp"

#include "HistogramTargetEntry.hpp"
#include "HistogramBucket.hpp"

#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

#include "TensorTotalsSum.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
static FloatEbmType SweepMultiDimensional(
   const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets,
   const FeatureGroup * const pFeatureGroup,
   size_t * const aiPoint,
   const size_t directionVectorLow,
   const unsigned int iDimensionSweep,
   const size_t cSweepBins,
   const size_t cSamplesRequiredForChildSplitMin,
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pHistogramBucketBestAndTemp,
   size_t * const piBestSplit
#ifndef NDEBUG
   , const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

   // don't LOG this!  It would create way too much chatter!

   // TODO : optimize this function

   EBM_ASSERT(1 <= pFeatureGroup->GetCountSignificantDimensions());
   EBM_ASSERT(iDimensionSweep < pFeatureGroup->GetCountSignificantDimensions());
   EBM_ASSERT(0 == (directionVectorLow & (size_t { 1 } << iDimensionSweep)));

   const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
      compilerLearningTypeOrCountTargetClasses,
      runtimeLearningTypeOrCountTargetClasses
   );
   const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow(bClassification, cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);
   EBM_ASSERT(!IsMultiplyError(size_t { 2 }, cBytesPerHistogramBucket)); // we're accessing allocated memory
   const size_t cBytesPerTwoHistogramBuckets = cBytesPerHistogramBucket << 1;

   size_t * const piBin = &aiPoint[iDimensionSweep];
   *piBin = 0;
   size_t directionVectorHigh = directionVectorLow | size_t { 1 } << iDimensionSweep;

   EBM_ASSERT(2 <= cSweepBins);

   size_t iBestSplit = 0;

   HistogramBucket<bClassification> * const pTotalsLow =
      GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pHistogramBucketBestAndTemp, 2);
   ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pTotalsLow, aHistogramBucketsEndDebug);

   HistogramBucket<bClassification> * const pTotalsHigh =
      GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pHistogramBucketBestAndTemp, 3);
   ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pTotalsHigh, aHistogramBucketsEndDebug);

   EBM_ASSERT(0 < cSamplesRequiredForChildSplitMin);

   FloatEbmType bestGain = k_illegalGain;
   size_t iBin = 0;
   do {
      *piBin = iBin;

      EBM_ASSERT(2 == pFeatureGroup->GetCountSignificantDimensions()); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
      TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, 2>(
         runtimeLearningTypeOrCountTargetClasses,
         pFeatureGroup,
         aHistogramBuckets,
         aiPoint,
         directionVectorLow,
         pTotalsLow
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      if(LIKELY(cSamplesRequiredForChildSplitMin <= pTotalsLow->GetCountSamplesInBucket())) {
         EBM_ASSERT(2 == pFeatureGroup->GetCountSignificantDimensions()); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
         TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, 2>(
            runtimeLearningTypeOrCountTargetClasses,
            pFeatureGroup,
            aHistogramBuckets,
            aiPoint,
            directionVectorHigh,
            pTotalsHigh
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
         if(LIKELY(cSamplesRequiredForChildSplitMin <= pTotalsHigh->GetCountSamplesInBucket())) {
            FloatEbmType gain = FloatEbmType { 0 };
            EBM_ASSERT(0 < pTotalsLow->GetCountSamplesInBucket());
            EBM_ASSERT(0 < pTotalsHigh->GetCountSamplesInBucket());

            const FloatEbmType cLowWeightInBucket = pTotalsLow->GetWeightInBucket();
            const FloatEbmType cHighWeightInBucket = pTotalsHigh->GetWeightInBucket();

            HistogramTargetEntry<bClassification> * const pHistogramTargetEntryLow =
               pTotalsLow->GetHistogramTargetEntry();

            HistogramTargetEntry<bClassification> * const pHistogramTargetEntryHigh =
               pTotalsHigh->GetHistogramTargetEntry();

            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators 
               // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

               constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;
               const FloatEbmType gain1 = EbmStats::CalcPartialGain(
                  pHistogramTargetEntryLow[iVector].m_sumGradients, bUseLogitBoost ? pHistogramTargetEntryLow[iVector].GetSumHessians() : cLowWeightInBucket);
               EBM_ASSERT(std::isnan(gain1) || FloatEbmType { 0 } <= gain1);
               gain += gain1;
               const FloatEbmType gain2 = EbmStats::CalcPartialGain(
                  pHistogramTargetEntryHigh[iVector].m_sumGradients, bUseLogitBoost ? pHistogramTargetEntryHigh[iVector].GetSumHessians() : cHighWeightInBucket);
               EBM_ASSERT(std::isnan(gain2) || FloatEbmType { 0 } <= gain2);
               gain += gain2;
            }
            EBM_ASSERT(std::isnan(gain) || FloatEbmType { 0 } <= gain); // sumation of positive numbers should be positive

            if(UNLIKELY(/* NaN */ !LIKELY(gain <= bestGain))) {
               // propagate NaNs

               bestGain = gain;
               iBestSplit = iBin;

               ASSERT_BINNED_BUCKET_OK(
                  cBytesPerHistogramBucket,
                  GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pHistogramBucketBestAndTemp, 1),
                  aHistogramBucketsEndDebug
               );
               ASSERT_BINNED_BUCKET_OK(
                  cBytesPerHistogramBucket,
                  GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pTotalsLow, 1),
                  aHistogramBucketsEndDebug
               );
               memcpy(pHistogramBucketBestAndTemp, pTotalsLow, cBytesPerTwoHistogramBuckets); // this copies both pTotalsLow and pTotalsHigh
            } else {
               EBM_ASSERT(!std::isnan(gain));
            }
         }
      }
      ++iBin;
   } while(iBin < cSweepBins - 1);
   *piBestSplit = iBestSplit;

   EBM_ASSERT(std::isnan(bestGain) || k_illegalGain == bestGain || FloatEbmType { 0 } <= bestGain);
   return bestGain;
}

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class PartitionTwoDimensionalBoostingInternal final {
public:

   PartitionTwoDimensionalBoostingInternal() = delete; // this is a static class.  Do not construct

   WARNING_PUSH
   WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE

   static ErrorEbmType Func(
      BoosterShell * const pBoosterShell,
      const FeatureGroup * const pFeatureGroup,
      const size_t cSamplesRequiredForChildSplitMin,
      HistogramBucketBase * pAuxiliaryBucketZoneBase,
      FloatEbmType * const pTotalGain
#ifndef NDEBUG
      , const HistogramBucketBase * const aHistogramBucketsDebugCopyBase
#endif // NDEBUG
   ) {
      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      ErrorEbmType error;
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();

      HistogramBucketBase * const aHistogramBucketBase = pBoosterShell->GetHistogramBucketBase();
      CompressibleTensor * const pSmallChangeToModelOverwriteSingleSamplingSet =
         pBoosterShell->GetOverwritableModelUpdate();

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses()
      );

      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);

      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pAuxiliaryBucketZone = pAuxiliaryBucketZoneBase->GetHistogramBucket<bClassification>();
      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets = aHistogramBucketBase->GetHistogramBucket<bClassification>();

#ifndef NDEBUG
      const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy = aHistogramBucketsDebugCopyBase->GetHistogramBucket<bClassification>();
#endif // NDEBUG

      size_t aiStart[k_cDimensionsMax];

      EBM_ASSERT(2 == pFeatureGroup->GetCountSignificantDimensions());
      size_t iDimensionLoop = 0;
      size_t iDimension1 = 0;
      size_t iDimension2 = 0;
      size_t cBinsDimension1 = 0;
      size_t cBinsDimension2 = 0;
      const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
      const FeatureGroupEntry * const pFeatureGroupEntryEnd = pFeatureGroupEntry + pFeatureGroup->GetCountDimensions();
      do {
         const size_t cBins = pFeatureGroupEntry->m_pFeature->GetCountBins();
         EBM_ASSERT(size_t { 1 } <= cBins); // we don't boost on empty training sets
         if(size_t { 1 } < cBins) {
            EBM_ASSERT(0 == cBinsDimension2);
            if(0 == cBinsDimension1) {
               iDimension1 = iDimensionLoop;
               cBinsDimension1 = cBins;
            } else {
               iDimension2 = iDimensionLoop;
               cBinsDimension2 = cBins;
            }
         }
         ++iDimensionLoop;
         ++pFeatureGroupEntry;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
      EBM_ASSERT(2 <= cBinsDimension1);
      EBM_ASSERT(2 <= cBinsDimension2);

      FloatEbmType bestGain = k_illegalGain;

      size_t splitFirst1Best;
      size_t splitFirst1LowBest;
      size_t splitFirst1HighBest;

      HistogramBucket<bClassification> * pTotals1LowLowBest =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 0);
      HistogramBucket<bClassification> * pTotals1LowHighBest =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 1);
      HistogramBucket<bClassification> * pTotals1HighLowBest =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 2);
      HistogramBucket<bClassification> * pTotals1HighHighBest =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 3);

      EBM_ASSERT(0 < cSamplesRequiredForChildSplitMin);

      LOG_0(TraceLevelVerbose, "PartitionTwoDimensionalBoostingInternal Starting FIRST bin sweep loop");
      size_t iBin1 = 0;
      do {
         aiStart[0] = iBin1;

         size_t splitSecond1LowBest;
         HistogramBucket<bClassification> * pTotals2LowLowBest =
            GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 4);
         HistogramBucket<bClassification> * pTotals2LowHighBest =
            GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 5);
         const FloatEbmType gain1 = SweepMultiDimensional<compilerLearningTypeOrCountTargetClasses>(
            aHistogramBuckets,
            pFeatureGroup,
            aiStart,
            0x0,
            1,
            cBinsDimension2,
            cSamplesRequiredForChildSplitMin,
            runtimeLearningTypeOrCountTargetClasses,
            pTotals2LowLowBest,
            &splitSecond1LowBest
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
            , pBoosterShell->GetHistogramBucketsEndDebug()
#endif // NDEBUG
            );

         if(LIKELY(/* NaN */ !UNLIKELY(gain1 < FloatEbmType { 0 }))) {
            EBM_ASSERT(std::isnan(gain1) || FloatEbmType { 0 } <= gain1);

            size_t splitSecond1HighBest;
            HistogramBucket<bClassification> * pTotals2HighLowBest =
               GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 8);
            HistogramBucket<bClassification> * pTotals2HighHighBest =
               GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 9);
            const FloatEbmType gain2 = SweepMultiDimensional<compilerLearningTypeOrCountTargetClasses>(
               aHistogramBuckets,
               pFeatureGroup,
               aiStart,
               0x1,
               1,
               cBinsDimension2,
               cSamplesRequiredForChildSplitMin,
               runtimeLearningTypeOrCountTargetClasses,
               pTotals2HighLowBest,
               &splitSecond1HighBest
#ifndef NDEBUG
               , aHistogramBucketsDebugCopy
               , pBoosterShell->GetHistogramBucketsEndDebug()
#endif // NDEBUG
               );

            if(LIKELY(/* NaN */ !UNLIKELY(gain2 < FloatEbmType { 0 }))) {
               EBM_ASSERT(std::isnan(gain2) || FloatEbmType { 0 } <= gain2);

               const FloatEbmType gain = gain1 + gain2;
               if(UNLIKELY(/* NaN */ !LIKELY(gain <= bestGain))) {
                  // propagate NaNs

                  EBM_ASSERT(std::isnan(gain) || FloatEbmType { 0 } <= gain);

                  bestGain = gain;
                  splitFirst1Best = iBin1;
                  splitFirst1LowBest = splitSecond1LowBest;
                  splitFirst1HighBest = splitSecond1HighBest;

                  pTotals1LowLowBest->Copy(*pTotals2LowLowBest, cVectorLength);
                  pTotals1LowHighBest->Copy(*pTotals2LowHighBest, cVectorLength);
                  pTotals1HighLowBest->Copy(*pTotals2HighLowBest, cVectorLength);
                  pTotals1HighHighBest->Copy(*pTotals2HighHighBest, cVectorLength);
               } else {
                  EBM_ASSERT(!std::isnan(gain));
               }
            } else {
               EBM_ASSERT(!std::isnan(gain2));
               EBM_ASSERT(k_illegalGain == gain2);
            }
         } else {
            EBM_ASSERT(!std::isnan(gain1));
            EBM_ASSERT(k_illegalGain == gain1);
         }
         ++iBin1;
      } while(iBin1 < cBinsDimension1 - 1);

      bool bSplitFirst2 = false;

      size_t splitFirst2Best;
      size_t splitFirst2LowBest;
      size_t splitFirst2HighBest;

      HistogramBucket<bClassification> * pTotals2LowLowBest =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 12);
      HistogramBucket<bClassification> * pTotals2LowHighBest =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 13);
      HistogramBucket<bClassification> * pTotals2HighLowBest =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 14);
      HistogramBucket<bClassification> * pTotals2HighHighBest =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 15);

      LOG_0(TraceLevelVerbose, "PartitionTwoDimensionalBoostingInternal Starting SECOND bin sweep loop");
      size_t iBin2 = 0;
      do {
         aiStart[1] = iBin2;

         size_t splitSecond2LowBest;
         HistogramBucket<bClassification> * pTotals1LowLowBestInner =
            GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 16);
         HistogramBucket<bClassification> * pTotals1LowHighBestInner =
            GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 17);
         const FloatEbmType gain1 = SweepMultiDimensional<compilerLearningTypeOrCountTargetClasses>(
            aHistogramBuckets,
            pFeatureGroup,
            aiStart,
            0x0,
            0,
            cBinsDimension1,
            cSamplesRequiredForChildSplitMin,
            runtimeLearningTypeOrCountTargetClasses,
            pTotals1LowLowBestInner,
            &splitSecond2LowBest
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
            , pBoosterShell->GetHistogramBucketsEndDebug()
#endif // NDEBUG
            );

         if(LIKELY(/* NaN */ !UNLIKELY(gain1 < FloatEbmType { 0 }))) {
            EBM_ASSERT(std::isnan(gain1) || FloatEbmType { 0 } <= gain1);

            size_t splitSecond2HighBest;
            HistogramBucket<bClassification> * pTotals1HighLowBestInner =
               GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 20);
            HistogramBucket<bClassification> * pTotals1HighHighBestInner =
               GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 21);
            const FloatEbmType gain2 = SweepMultiDimensional<compilerLearningTypeOrCountTargetClasses>(
               aHistogramBuckets,
               pFeatureGroup,
               aiStart,
               0x2,
               0,
               cBinsDimension1,
               cSamplesRequiredForChildSplitMin,
               runtimeLearningTypeOrCountTargetClasses,
               pTotals1HighLowBestInner,
               &splitSecond2HighBest
#ifndef NDEBUG
               , aHistogramBucketsDebugCopy
               , pBoosterShell->GetHistogramBucketsEndDebug()
#endif // NDEBUG
               );

            if(LIKELY(/* NaN */ !UNLIKELY(gain2 < FloatEbmType { 0 }))) {
               EBM_ASSERT(std::isnan(gain2) || FloatEbmType { 0 } <= gain2);

               const FloatEbmType gain = gain1 + gain2;
               if(UNLIKELY(/* NaN */ !LIKELY(gain <= bestGain))) {
                  // propagate NaNs

                  EBM_ASSERT(std::isnan(gain) || FloatEbmType { 0 } <= gain);

                  bestGain = gain;
                  splitFirst2Best = iBin2;
                  splitFirst2LowBest = splitSecond2LowBest;
                  splitFirst2HighBest = splitSecond2HighBest;

                  pTotals2LowLowBest->Copy(*pTotals1LowLowBestInner, cVectorLength);
                  pTotals2LowHighBest->Copy(*pTotals1LowHighBestInner, cVectorLength);
                  pTotals2HighLowBest->Copy(*pTotals1HighLowBestInner, cVectorLength);
                  pTotals2HighHighBest->Copy(*pTotals1HighHighBestInner, cVectorLength);

                  bSplitFirst2 = true;
               } else {
                  EBM_ASSERT(!std::isnan(gain));
               }
            } else {
               EBM_ASSERT(!std::isnan(gain2));
               EBM_ASSERT(k_illegalGain == gain2);
            }
         } else {
            EBM_ASSERT(!std::isnan(gain1));
            EBM_ASSERT(k_illegalGain == gain1);
         }
         ++iBin2;
      } while(iBin2 < cBinsDimension2 - 1);
      LOG_0(TraceLevelVerbose, "PartitionTwoDimensionalBoostingInternal Done sweep loops");

      EBM_ASSERT(std::isnan(bestGain) || k_illegalGain == bestGain || FloatEbmType { 0 } <= bestGain);

      // the bucket before the pAuxiliaryBucketZoneBase is the last summation bucket of aHistogramBucketsBase, 
      // which contains the totals of all buckets
      const HistogramBucket<bClassification> * const pTotal =
         reinterpret_cast<const HistogramBucket<bClassification> *>(
            reinterpret_cast<const char *>(pAuxiliaryBucketZoneBase) - cBytesPerHistogramBucket);

      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pTotal, pBoosterShell->GetHistogramBucketsEndDebug());

      const HistogramTargetEntry<bClassification> * const pHistogramTargetEntryTotal =
         pTotal->GetHistogramTargetEntry();

      const FloatEbmType weightAll = pTotal->GetWeightInBucket();
      EBM_ASSERT(0 < weightAll);

      *pTotalGain = FloatEbmType { 0 };
      EBM_ASSERT(0 <= k_gainMin);
      if(LIKELY(/* NaN */ !UNLIKELY(bestGain < k_gainMin))) {
         EBM_ASSERT(std::isnan(bestGain) || FloatEbmType { 0 } <= bestGain);

         // signal that we've hit an overflow.  Use +inf here since our caller likes that and will flip to -inf
         *pTotalGain = std::numeric_limits<FloatEbmType>::infinity();
         if(LIKELY(/* NaN */ bestGain <= std::numeric_limits<FloatEbmType>::max())) {
            EBM_ASSERT(!std::isnan(bestGain));
            EBM_ASSERT(FloatEbmType { 0 } <= bestGain);
            EBM_ASSERT(std::numeric_limits<FloatEbmType>::infinity() != bestGain);

            // now subtract the parent partial gain
            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators 
               // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

               constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;
               const FloatEbmType gain1 = EbmStats::CalcPartialGain(
                  pHistogramTargetEntryTotal[iVector].m_sumGradients,
                  bUseLogitBoost ? pHistogramTargetEntryTotal[iVector].GetSumHessians() : weightAll
               );
               EBM_ASSERT(std::isnan(gain1) || FloatEbmType { 0 } <= gain1);
               bestGain -= gain1;
            }

            EBM_ASSERT(std::numeric_limits<FloatEbmType>::infinity() != bestGain);
            EBM_ASSERT(std::isnan(bestGain) || -std::numeric_limits<FloatEbmType>::infinity() == bestGain ||
               k_epsilonNegativeGainAllowed <= bestGain);

            if(LIKELY(/* NaN */ std::numeric_limits<FloatEbmType>::lowest() <= bestGain)) {
               EBM_ASSERT(!std::isnan(bestGain));
               EBM_ASSERT(!std::isinf(bestGain));
               EBM_ASSERT(k_epsilonNegativeGainAllowed <= bestGain);

               *pTotalGain = FloatEbmType { 0 };
               if(LIKELY(k_gainMin <= bestGain)) {
                  *pTotalGain = bestGain;
                  if(bSplitFirst2) {
                     // if bSplitFirst2 is true, then there definetly was a split, so we don't have to check for zero splits
                     error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountSplits(iDimension2, 1);
                     if(Error_None != error) {
                        // already logged
                        return error;
                     }
                     pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimension2)[0] = splitFirst2Best;

                     if(splitFirst2LowBest < splitFirst2HighBest) {
                        error = pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountSplits(iDimension1, 2);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimension1)[0] = splitFirst2LowBest;
                        pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimension1)[1] = splitFirst2HighBest;
                     } else if(splitFirst2HighBest < splitFirst2LowBest) {
                        error = pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountSplits(iDimension1, 2);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimension1)[0] = splitFirst2HighBest;
                        pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimension1)[1] = splitFirst2LowBest;
                     } else {
                        error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountSplits(iDimension1, 1);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }

                        error = pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimension1)[0] = splitFirst2LowBest;
                     }

                     HistogramTargetEntry<bClassification> * const pHistogramTargetEntryTotals2LowLowBest =
                        pTotals2LowLowBest->GetHistogramTargetEntry();
                     HistogramTargetEntry<bClassification> * const pHistogramTargetEntryTotals2LowHighBest =
                        pTotals2LowHighBest->GetHistogramTargetEntry();
                     HistogramTargetEntry<bClassification> * const pHistogramTargetEntryTotals2HighLowBest =
                        pTotals2HighLowBest->GetHistogramTargetEntry();
                     HistogramTargetEntry<bClassification> * const pHistogramTargetEntryTotals2HighHighBest =
                        pTotals2HighHighBest->GetHistogramTargetEntry();

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
                     FloatEbmType zeroLogit0 = FloatEbmType { 0 };
                     FloatEbmType zeroLogit1 = FloatEbmType { 0 };
                     FloatEbmType zeroLogit2 = FloatEbmType { 0 };
                     FloatEbmType zeroLogit3 = FloatEbmType { 0 };
#endif // ZERO_FIRST_MULTICLASS_LOGIT

                     for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
                        FloatEbmType predictionLowLow;
                        FloatEbmType predictionLowHigh;
                        FloatEbmType predictionHighLow;
                        FloatEbmType predictionHighHigh;

                        if(bClassification) {
                           predictionLowLow = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals2LowLowBest[iVector].m_sumGradients,
                              pHistogramTargetEntryTotals2LowLowBest[iVector].GetSumHessians()
                           );
                           predictionLowHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals2LowHighBest[iVector].m_sumGradients,
                              pHistogramTargetEntryTotals2LowHighBest[iVector].GetSumHessians()
                           );
                           predictionHighLow = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals2HighLowBest[iVector].m_sumGradients,
                              pHistogramTargetEntryTotals2HighLowBest[iVector].GetSumHessians()
                           );
                           predictionHighHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals2HighHighBest[iVector].m_sumGradients,
                              pHistogramTargetEntryTotals2HighHighBest[iVector].GetSumHessians()
                           );

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
                           if(IsMulticlass(compilerLearningTypeOrCountTargetClasses)) {
                              if(size_t { 0 } == iVector) {
                                 zeroLogit0 = predictionLowLow;
                                 zeroLogit1 = predictionLowHigh;
                                 zeroLogit2 = predictionHighLow;
                                 zeroLogit3 = predictionHighHigh;
                              }
                              predictionLowLow -= zeroLogit0;
                              predictionLowHigh -= zeroLogit1;
                              predictionHighLow -= zeroLogit2;
                              predictionHighHigh -= zeroLogit3;
                           }
#endif // ZERO_FIRST_MULTICLASS_LOGIT

                        } else {
                           EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
                           predictionLowLow = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals2LowLowBest[iVector].m_sumGradients,
                              pTotals2LowLowBest->GetWeightInBucket()
                           );
                           predictionLowHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals2LowHighBest[iVector].m_sumGradients,
                              pTotals2LowHighBest->GetWeightInBucket()
                           );
                           predictionHighLow = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals2HighLowBest[iVector].m_sumGradients,
                              pTotals2HighLowBest->GetWeightInBucket()
                           );
                           predictionHighHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals2HighHighBest[iVector].m_sumGradients,
                              pTotals2HighHighBest->GetWeightInBucket()
                           );
                        }

                        if(splitFirst2LowBest < splitFirst2HighBest) {
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[0 * cVectorLength + iVector] = predictionLowLow;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[1 * cVectorLength + iVector] = predictionLowHigh;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[2 * cVectorLength + iVector] = predictionLowHigh;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[3 * cVectorLength + iVector] = predictionHighLow;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[4 * cVectorLength + iVector] = predictionHighLow;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[5 * cVectorLength + iVector] = predictionHighHigh;
                        } else if(splitFirst2HighBest < splitFirst2LowBest) {
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[0 * cVectorLength + iVector] = predictionLowLow;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[1 * cVectorLength + iVector] = predictionLowLow;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[2 * cVectorLength + iVector] = predictionLowHigh;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[3 * cVectorLength + iVector] = predictionHighLow;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[4 * cVectorLength + iVector] = predictionHighHigh;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[5 * cVectorLength + iVector] = predictionHighHigh;
                        } else {
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[0 * cVectorLength + iVector] = predictionLowLow;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[1 * cVectorLength + iVector] = predictionLowHigh;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[2 * cVectorLength + iVector] = predictionHighLow;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[3 * cVectorLength + iVector] = predictionHighHigh;
                        }
                     }
                  } else {
                     error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountSplits(iDimension1, 1);
                     if(Error_None != error) {
                        // already logged
                        return error;
                     }
                     pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimension1)[0] = splitFirst1Best;

                     if(splitFirst1LowBest < splitFirst1HighBest) {
                        error = pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }

                        error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountSplits(iDimension2, 2);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimension2)[0] = splitFirst1LowBest;
                        pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimension2)[1] = splitFirst1HighBest;
                     } else if(splitFirst1HighBest < splitFirst1LowBest) {
                        error = pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }

                        error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountSplits(iDimension2, 2);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimension2)[0] = splitFirst1HighBest;
                        pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimension2)[1] = splitFirst1LowBest;
                     } else {
                        error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountSplits(iDimension2, 1);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        error = pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimension2)[0] = splitFirst1LowBest;
                     }

                     HistogramTargetEntry<bClassification> * const pHistogramTargetEntryTotals1LowLowBest =
                        pTotals1LowLowBest->GetHistogramTargetEntry();
                     HistogramTargetEntry<bClassification> * const pHistogramTargetEntryTotals1LowHighBest =
                        pTotals1LowHighBest->GetHistogramTargetEntry();
                     HistogramTargetEntry<bClassification> * const pHistogramTargetEntryTotals1HighLowBest =
                        pTotals1HighLowBest->GetHistogramTargetEntry();
                     HistogramTargetEntry<bClassification> * const pHistogramTargetEntryTotals1HighHighBest =
                        pTotals1HighHighBest->GetHistogramTargetEntry();

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
                     FloatEbmType zeroLogit0 = FloatEbmType { 0 };
                     FloatEbmType zeroLogit1 = FloatEbmType { 0 };
                     FloatEbmType zeroLogit2 = FloatEbmType { 0 };
                     FloatEbmType zeroLogit3 = FloatEbmType { 0 };
#endif // ZERO_FIRST_MULTICLASS_LOGIT

                     for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
                        FloatEbmType predictionLowLow;
                        FloatEbmType predictionLowHigh;
                        FloatEbmType predictionHighLow;
                        FloatEbmType predictionHighHigh;

                        if(bClassification) {
                           predictionLowLow = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals1LowLowBest[iVector].m_sumGradients,
                              pHistogramTargetEntryTotals1LowLowBest[iVector].GetSumHessians()
                           );
                           predictionLowHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals1LowHighBest[iVector].m_sumGradients,
                              pHistogramTargetEntryTotals1LowHighBest[iVector].GetSumHessians()
                           );
                           predictionHighLow = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals1HighLowBest[iVector].m_sumGradients,
                              pHistogramTargetEntryTotals1HighLowBest[iVector].GetSumHessians()
                           );
                           predictionHighHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals1HighHighBest[iVector].m_sumGradients,
                              pHistogramTargetEntryTotals1HighHighBest[iVector].GetSumHessians()
                           );

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
                           if(IsMulticlass(compilerLearningTypeOrCountTargetClasses)) {
                              if(size_t { 0 } == iVector) {
                                 zeroLogit0 = predictionLowLow;
                                 zeroLogit1 = predictionLowHigh;
                                 zeroLogit2 = predictionHighLow;
                                 zeroLogit3 = predictionHighHigh;
                              }
                              predictionLowLow -= zeroLogit0;
                              predictionLowHigh -= zeroLogit1;
                              predictionHighLow -= zeroLogit2;
                              predictionHighHigh -= zeroLogit3;
                           }
#endif // ZERO_FIRST_MULTICLASS_LOGIT
                        } else {
                           EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
                           predictionLowLow = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals1LowLowBest[iVector].m_sumGradients,
                              pTotals1LowLowBest->GetWeightInBucket()
                           );
                           predictionLowHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals1LowHighBest[iVector].m_sumGradients,
                              pTotals1LowHighBest->GetWeightInBucket()
                           );
                           predictionHighLow = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals1HighLowBest[iVector].m_sumGradients,
                              pTotals1HighLowBest->GetWeightInBucket()
                           );
                           predictionHighHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals1HighHighBest[iVector].m_sumGradients,
                              pTotals1HighHighBest->GetWeightInBucket()
                           );
                        }
                        if(splitFirst1LowBest < splitFirst1HighBest) {
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[0 * cVectorLength + iVector] = predictionLowLow;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[1 * cVectorLength + iVector] = predictionHighLow;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[2 * cVectorLength + iVector] = predictionLowHigh;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[3 * cVectorLength + iVector] = predictionHighLow;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[4 * cVectorLength + iVector] = predictionLowHigh;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[5 * cVectorLength + iVector] = predictionHighHigh;
                        } else if(splitFirst1HighBest < splitFirst1LowBest) {
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[0 * cVectorLength + iVector] = predictionLowLow;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[1 * cVectorLength + iVector] = predictionHighLow;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[2 * cVectorLength + iVector] = predictionLowLow;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[3 * cVectorLength + iVector] = predictionHighHigh;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[4 * cVectorLength + iVector] = predictionLowHigh;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[5 * cVectorLength + iVector] = predictionHighHigh;
                        } else {
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[0 * cVectorLength + iVector] = predictionLowLow;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[1 * cVectorLength + iVector] = predictionHighLow;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[2 * cVectorLength + iVector] = predictionLowHigh;
                           pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[3 * cVectorLength + iVector] = predictionHighHigh;
                        }
                     }
                  }
                  return Error_None;
               }
            } else {
               EBM_ASSERT(std::isnan(bestGain) || -std::numeric_limits<FloatEbmType>::infinity() == bestGain);
            }
         } else {
            EBM_ASSERT(std::isnan(bestGain) || std::numeric_limits<FloatEbmType>::infinity() == bestGain);
         }
      } else {
         EBM_ASSERT(!std::isnan(bestGain));
      }

      // there were no good splits found
#ifndef NDEBUG
      const ErrorEbmType errorDebug1 =
#endif // NDEBUG
         pSmallChangeToModelOverwriteSingleSamplingSet->SetCountSplits(iDimension1, 0);
      // we can't fail since we're setting this to zero, so no allocations.  We don't in fact need the split array at all
      EBM_ASSERT(Error_None == errorDebug1);

#ifndef NDEBUG
      const ErrorEbmType errorDebug2 =
#endif // NDEBUG
         pSmallChangeToModelOverwriteSingleSamplingSet->SetCountSplits(iDimension2, 0);
      // we can't fail since we're setting this to zero, so no allocations.  We don't in fact need the split array at all
      EBM_ASSERT(Error_None == errorDebug2);

      // we don't need to call pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity, 
      // since our value capacity would be 1, which is pre-allocated

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
      FloatEbmType zeroLogit = FloatEbmType { 0 };
#endif // ZERO_FIRST_MULTICLASS_LOGIT

      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         FloatEbmType update;
         if(bClassification) {
            update = EbmStats::ComputeSinglePartitionUpdate(
               pHistogramTargetEntryTotal[iVector].m_sumGradients,
               pHistogramTargetEntryTotal[iVector].GetSumHessians()
            );

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
            if(IsMulticlass(compilerLearningTypeOrCountTargetClasses)) {
               if(size_t { 0 } == iVector) {
                  zeroLogit = update;
               }
               update -= zeroLogit;
            }
#endif // ZERO_FIRST_MULTICLASS_LOGIT

         } else {
            EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
            update = EbmStats::ComputeSinglePartitionUpdate(
               pHistogramTargetEntryTotal[iVector].m_sumGradients,
               weightAll
            );
         }

         pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[iVector] = update;
      }
      return Error_None;
   }
   WARNING_POP
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class PartitionTwoDimensionalBoostingTarget final {
public:

   PartitionTwoDimensionalBoostingTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static ErrorEbmType Func(
      BoosterShell * const pBoosterShell,
      const FeatureGroup * const pFeatureGroup,
      const size_t cSamplesRequiredForChildSplitMin,
      HistogramBucketBase * pAuxiliaryBucketZone,
      FloatEbmType * const pTotalGain
#ifndef NDEBUG
      , const HistogramBucketBase * const aHistogramBucketsDebugCopy
#endif // NDEBUG
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         return PartitionTwoDimensionalBoostingInternal<compilerLearningTypeOrCountTargetClassesPossible>::Func(
            pBoosterShell,
            pFeatureGroup,
            cSamplesRequiredForChildSplitMin,
            pAuxiliaryBucketZone,
            pTotalGain
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
#endif // NDEBUG
         );
      } else {
         return PartitionTwoDimensionalBoostingTarget<compilerLearningTypeOrCountTargetClassesPossible + 1>::Func(
            pBoosterShell,
            pFeatureGroup,
            cSamplesRequiredForChildSplitMin,
            pAuxiliaryBucketZone,
            pTotalGain
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
#endif // NDEBUG
         );
      }
   }
};

template<>
class PartitionTwoDimensionalBoostingTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   PartitionTwoDimensionalBoostingTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static ErrorEbmType Func(
      BoosterShell * const pBoosterShell,
      const FeatureGroup * const pFeatureGroup,
      const size_t cSamplesRequiredForChildSplitMin,
      HistogramBucketBase * pAuxiliaryBucketZone,
      FloatEbmType * const pTotalGain
#ifndef NDEBUG
      , const HistogramBucketBase * const aHistogramBucketsDebugCopy
#endif // NDEBUG
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pBoosterShell->GetBoosterCore()->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pBoosterShell->GetBoosterCore()->GetRuntimeLearningTypeOrCountTargetClasses());

      return PartitionTwoDimensionalBoostingInternal<k_dynamicClassification>::Func(
         pBoosterShell,
         pFeatureGroup,
         cSamplesRequiredForChildSplitMin,
         pAuxiliaryBucketZone,
         pTotalGain
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
#endif // NDEBUG
      );
   }
};

extern ErrorEbmType PartitionTwoDimensionalBoosting(
   BoosterShell * const pBoosterShell,
   const FeatureGroup * const pFeatureGroup,
   const size_t cSamplesRequiredForChildSplitMin,
   HistogramBucketBase * pAuxiliaryBucketZone,
   FloatEbmType * const pTotalGain
#ifndef NDEBUG
   , const HistogramBucketBase * const aHistogramBucketsDebugCopy
#endif // NDEBUG
) {
   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();

   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      return PartitionTwoDimensionalBoostingTarget<2>::Func(
         pBoosterShell,
         pFeatureGroup,
         cSamplesRequiredForChildSplitMin,
         pAuxiliaryBucketZone,
         pTotalGain
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
#endif // NDEBUG
      );
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      return PartitionTwoDimensionalBoostingInternal<k_regression>::Func(
         pBoosterShell,
         pFeatureGroup,
         cSamplesRequiredForChildSplitMin,
         pAuxiliaryBucketZone,
         pTotalGain
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
#endif // NDEBUG
      );
   }
}

} // DEFINED_ZONE_NAME
