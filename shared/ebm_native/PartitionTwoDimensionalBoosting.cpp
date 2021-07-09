// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "SegmentedTensor.hpp"
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
   size_t * const piBestCut
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

   size_t iBestCut = 0;

   HistogramBucket<bClassification> * const pTotalsLow =
      GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pHistogramBucketBestAndTemp, 2);
   ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pTotalsLow, aHistogramBucketsEndDebug);

   HistogramBucket<bClassification> * const pTotalsHigh =
      GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pHistogramBucketBestAndTemp, 3);
   ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pTotalsHigh, aHistogramBucketsEndDebug);

   EBM_ASSERT(0 < cSamplesRequiredForChildSplitMin);

   FloatEbmType bestSplit = k_illegalGain;
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
            FloatEbmType splittingScore = FloatEbmType { 0 };
            EBM_ASSERT(0 < pTotalsLow->GetCountSamplesInBucket());
            EBM_ASSERT(0 < pTotalsHigh->GetCountSamplesInBucket());

            const FloatEbmType cLowWeightInBucket = pTotalsLow->GetWeightInBucket();
            const FloatEbmType cHighWeightInBucket = pTotalsHigh->GetWeightInBucket();

            HistogramTargetEntry<bClassification> * const pHistogramTargetEntryLow =
               pTotalsLow->GetHistogramTargetEntry();

            HistogramTargetEntry<bClassification> * const pHistogramTargetEntryHigh =
               pTotalsHigh->GetHistogramTargetEntry();

            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               // TODO : we can make this faster by doing the division in ComputeSinglePartitionGain after we add all the numerators 
               // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

               constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;
               const FloatEbmType splittingScoreUpdate1 = EbmStats::ComputeSinglePartitionGain(
                  pHistogramTargetEntryLow[iVector].m_sumGradients, bUseLogitBoost ? pHistogramTargetEntryLow[iVector].GetSumHessians() : cLowWeightInBucket);
               EBM_ASSERT(std::isnan(splittingScoreUpdate1) || FloatEbmType { 0 } <= splittingScoreUpdate1);
               splittingScore += splittingScoreUpdate1;
               const FloatEbmType splittingScoreUpdate2 = EbmStats::ComputeSinglePartitionGain(
                  pHistogramTargetEntryHigh[iVector].m_sumGradients, bUseLogitBoost ? pHistogramTargetEntryHigh[iVector].GetSumHessians() : cHighWeightInBucket);
               EBM_ASSERT(std::isnan(splittingScoreUpdate2) || FloatEbmType { 0 } <= splittingScoreUpdate2);
               splittingScore += splittingScoreUpdate2;
            }
            EBM_ASSERT(std::isnan(splittingScore) || FloatEbmType { 0 } <= splittingScore); // sumation of positive numbers should be positive

            // if we get a NaN result, we'd like to propagate it by making bestSplit NaN.  The rules for NaN values say that non equality comparisons are 
            // all false so, let's flip this comparison such that it should be true for NaN values.  If the compiler violates NaN comparions rules, 
            // no big deal.  NaN values will get us soon and shut down boosting.
            if(UNLIKELY(/* DO NOT CHANGE THIS WITHOUT READING THE ABOVE. WE DO THIS STRANGE COMPARISON FOR NaN values*/ !(splittingScore <= bestSplit))) {
               bestSplit = splittingScore;
               iBestCut = iBin;

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
               EBM_ASSERT(!std::isnan(splittingScore));
            }
         }
      }
      ++iBin;
   } while(iBin < cSweepBins - 1);
   *piBestCut = iBestCut;

   EBM_ASSERT(std::isnan(bestSplit) || bestSplit == k_illegalGain || FloatEbmType { 0 } <= bestSplit); // sumation of positive numbers should be positive
   return bestSplit;
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
      HistogramBucketBase * const pTotalBase,
      FloatEbmType * const pTotalGain
#ifndef NDEBUG
      , const HistogramBucketBase * const aHistogramBucketsDebugCopyBase
#endif // NDEBUG
   ) {
      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      ErrorEbmType error;
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();

      HistogramBucketBase * const aHistogramBucketBase = pBoosterShell->GetHistogramBucketBase();
      SegmentedTensor * const pSmallChangeToModelOverwriteSingleSamplingSet =
         pBoosterShell->GetOverwritableModelUpdate();

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses()
      );

      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);

      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pAuxiliaryBucketZone = pAuxiliaryBucketZoneBase->GetHistogramBucket<bClassification>();
      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pTotal = pTotalBase->GetHistogramBucket<bClassification>();
      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets = aHistogramBucketBase->GetHistogramBucket<bClassification>();

#ifndef NDEBUG
      const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy = aHistogramBucketsDebugCopyBase->GetHistogramBucket<bClassification>();
#endif // NDEBUG

      size_t aiStart[k_cDimensionsMax];

      FloatEbmType splittingScore;

      EBM_ASSERT(2 == pFeatureGroup->GetCountSignificantDimensions());
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
               cBinsDimension1 = cBins;
            } else {
               cBinsDimension2 = cBins;
            }
         }
         ++pFeatureGroupEntry;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
      EBM_ASSERT(2 <= cBinsDimension1);
      EBM_ASSERT(2 <= cBinsDimension2);

      FloatEbmType bestSplittingScore = k_illegalGain;

      size_t cutFirst1Best;
      size_t cutFirst1LowBest;
      size_t cutFirst1HighBest;

      HistogramBucket<bClassification> * pTotals1LowLowBest =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 0);
      HistogramBucket<bClassification> * pTotals1LowHighBest =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 1);
      HistogramBucket<bClassification> * pTotals1HighLowBest =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 2);
      HistogramBucket<bClassification> * pTotals1HighHighBest =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 3);

      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pTotal, pBoosterShell->GetHistogramBucketsEndDebug());

      EBM_ASSERT(0 < cSamplesRequiredForChildSplitMin);

      FloatEbmType splittingScoreParent = FloatEbmType { 0 };
      EBM_ASSERT(0 < pTotal->GetCountSamplesInBucket());
      const FloatEbmType cWeightInParentBucket = pTotal->GetWeightInBucket();

      HistogramTargetEntry<bClassification> * const pHistogramTargetEntryTotal =
         pTotal->GetHistogramTargetEntry();

      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         // TODO : we can make this faster by doing the division in ComputeSinglePartitionGainParent after we add all the numerators 
         // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

         constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;
         const FloatEbmType splittingScoreParentUpdate = EbmStats::ComputeSinglePartitionGain(
            pHistogramTargetEntryTotal[iVector].m_sumGradients,
            bUseLogitBoost ? pHistogramTargetEntryTotal[iVector].GetSumHessians() : cWeightInParentBucket
         );
         EBM_ASSERT(std::isnan(splittingScoreParentUpdate) || FloatEbmType { 0 } <= splittingScoreParentUpdate);
         splittingScoreParent += splittingScoreParentUpdate;
      }
      EBM_ASSERT(std::isnan(splittingScoreParent) || FloatEbmType { 0 } <= splittingScoreParent); // sumation of positive numbers should be positive

      LOG_0(TraceLevelVerbose, "PartitionTwoDimensionalBoostingInternal Starting FIRST bin sweep loop");
      size_t iBin1 = 0;
      do {
         aiStart[0] = iBin1;

         splittingScore = FloatEbmType { 0 };

         size_t cutSecond1LowBest;
         HistogramBucket<bClassification> * pTotals2LowLowBest =
            GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 4);
         HistogramBucket<bClassification> * pTotals2LowHighBest =
            GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 5);
         const FloatEbmType splittingScoreNew1 = SweepMultiDimensional<compilerLearningTypeOrCountTargetClasses>(
            aHistogramBuckets,
            pFeatureGroup,
            aiStart,
            0x0,
            1,
            cBinsDimension2,
            cSamplesRequiredForChildSplitMin,
            runtimeLearningTypeOrCountTargetClasses,
            pTotals2LowLowBest,
            &cutSecond1LowBest
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
            , pBoosterShell->GetHistogramBucketsEndDebug()
#endif // NDEBUG
            );

         // if we get a NaN result, we'd like to propagate it by making bestSplit NaN.  The rules for NaN values say that non equality comparisons are all
         // false so, let's flip this comparison such that it should be true for NaN values.  If the compiler violates NaN comparions rules, no big deal.  
         // NaN values will get us soon and shut down boosting.
         if(LIKELY(/* DO NOT CHANGE THIS WITHOUT READING THE ABOVE. WE DO THIS STRANGE COMPARISON FOR NaN values*/ !(k_illegalGain == splittingScoreNew1))) {
            EBM_ASSERT(std::isnan(splittingScoreNew1) || FloatEbmType { 0 } <= splittingScoreNew1);
            splittingScore += splittingScoreNew1;

            size_t cutSecond1HighBest;
            HistogramBucket<bClassification> * pTotals2HighLowBest =
               GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 8);
            HistogramBucket<bClassification> * pTotals2HighHighBest =
               GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 9);
            const FloatEbmType splittingScoreNew2 = SweepMultiDimensional<compilerLearningTypeOrCountTargetClasses>(
               aHistogramBuckets,
               pFeatureGroup,
               aiStart,
               0x1,
               1,
               cBinsDimension2,
               cSamplesRequiredForChildSplitMin,
               runtimeLearningTypeOrCountTargetClasses,
               pTotals2HighLowBest,
               &cutSecond1HighBest
#ifndef NDEBUG
               , aHistogramBucketsDebugCopy
               , pBoosterShell->GetHistogramBucketsEndDebug()
#endif // NDEBUG
               );
            // if we get a NaN result, we'd like to propagate it by making bestSplit NaN.  The rules for NaN values say that non equality comparisons are 
            // all false so, let's flip this comparison such that it should be true for NaN values.  If the compiler violates NaN comparions rules, 
            // no big deal.  NaN values will get us soon and shut down boosting.
            if(LIKELY(/* DO NOT CHANGE THIS WITHOUT READING THE ABOVE. WE DO THIS STRANGE COMPARISON FOR NaN values*/
               !(k_illegalGain == splittingScoreNew2))) {
               EBM_ASSERT(std::isnan(splittingScoreNew2) || FloatEbmType { 0 } <= splittingScoreNew2);
               splittingScore += splittingScoreNew2;

               // if we get a NaN result, we'd like to propagate it by making bestSplit NaN.  The rules for NaN values say that non equality comparisons 
               // are all false so, let's flip this comparison such that it should be true for NaN values.  If the compiler violates NaN comparions rules, 
               // no big deal.  NaN values will get us soon and shut down boosting.
               if(UNLIKELY(/* DO NOT CHANGE THIS WITHOUT READING THE ABOVE. WE DO THIS STRANGE COMPARISON FOR NaN values*/
                  !(splittingScore <= bestSplittingScore))) {
                  bestSplittingScore = splittingScore;
                  cutFirst1Best = iBin1;
                  cutFirst1LowBest = cutSecond1LowBest;
                  cutFirst1HighBest = cutSecond1HighBest;

                  pTotals1LowLowBest->Copy(*pTotals2LowLowBest, cVectorLength);
                  pTotals1LowHighBest->Copy(*pTotals2LowHighBest, cVectorLength);
                  pTotals1HighLowBest->Copy(*pTotals2HighLowBest, cVectorLength);
                  pTotals1HighHighBest->Copy(*pTotals2HighHighBest, cVectorLength);
               } else {
                  EBM_ASSERT(!std::isnan(splittingScore));
               }
            } else {
               EBM_ASSERT(!std::isnan(splittingScoreNew2));
               EBM_ASSERT(k_illegalGain == splittingScoreNew2);
            }
         } else {
            EBM_ASSERT(!std::isnan(splittingScoreNew1));
            EBM_ASSERT(k_illegalGain == splittingScoreNew1);
         }
         ++iBin1;
      } while(iBin1 < cBinsDimension1 - 1);

      bool bCutFirst2 = false;

      size_t cutFirst2Best;
      size_t cutFirst2LowBest;
      size_t cutFirst2HighBest;

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

         splittingScore = FloatEbmType { 0 };

         size_t cutSecond2LowBest;
         HistogramBucket<bClassification> * pTotals1LowLowBestInner =
            GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 16);
         HistogramBucket<bClassification> * pTotals1LowHighBestInner =
            GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 17);
         const FloatEbmType splittingScoreNew1 = SweepMultiDimensional<compilerLearningTypeOrCountTargetClasses>(
            aHistogramBuckets,
            pFeatureGroup,
            aiStart,
            0x0,
            0,
            cBinsDimension1,
            cSamplesRequiredForChildSplitMin,
            runtimeLearningTypeOrCountTargetClasses,
            pTotals1LowLowBestInner,
            &cutSecond2LowBest
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
            , pBoosterShell->GetHistogramBucketsEndDebug()
#endif // NDEBUG
            );

         // if we get a NaN result, we'd like to propagate it by making bestSplit NaN.  The rules for NaN values say that non equality comparisons are 
         // all false so, let's flip this comparison such that it should be true for NaN values.  If the compiler violates NaN comparions rules, no big deal.
         // NaN values will get us soon and shut down boosting.
         if(LIKELY(/* DO NOT CHANGE THIS WITHOUT READING THE ABOVE. WE DO THIS STRANGE COMPARISON FOR NaN values*/ !(k_illegalGain == splittingScoreNew1))) {
            EBM_ASSERT(std::isnan(splittingScoreNew1) || FloatEbmType { 0 } <= splittingScoreNew1);
            splittingScore += splittingScoreNew1;

            size_t cutSecond2HighBest;
            HistogramBucket<bClassification> * pTotals1HighLowBestInner =
               GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 20);
            HistogramBucket<bClassification> * pTotals1HighHighBestInner =
               GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 21);
            const FloatEbmType splittingScoreNew2 = SweepMultiDimensional<compilerLearningTypeOrCountTargetClasses>(
               aHistogramBuckets,
               pFeatureGroup,
               aiStart,
               0x2,
               0,
               cBinsDimension1,
               cSamplesRequiredForChildSplitMin,
               runtimeLearningTypeOrCountTargetClasses,
               pTotals1HighLowBestInner,
               &cutSecond2HighBest
#ifndef NDEBUG
               , aHistogramBucketsDebugCopy
               , pBoosterShell->GetHistogramBucketsEndDebug()
#endif // NDEBUG
               );
            // if we get a NaN result, we'd like to propagate it by making bestSplit NaN.  The rules for NaN values say that non equality comparisons are 
            // all false so, let's flip this comparison such that it should be true for NaN values.  If the compiler violates NaN comparions rules, 
            // no big deal.  NaN values will get us soon and shut down boosting.
            if(LIKELY(/* DO NOT CHANGE THIS WITHOUT READING THE ABOVE. WE DO THIS STRANGE COMPARISON FOR NaN values*/
               !(k_illegalGain == splittingScoreNew2))) {
               EBM_ASSERT(std::isnan(splittingScoreNew2) || FloatEbmType { 0 } <= splittingScoreNew2);
               splittingScore += splittingScoreNew2;
               // if we get a NaN result, we'd like to propagate it by making bestSplit NaN.  The rules for NaN values say that non equality comparisons 
               // are all false so, let's flip this comparison such that it should be true for NaN values.  If the compiler violates NaN comparions rules, 
               // no big deal.  NaN values will get us soon and shut down boosting.
               if(UNLIKELY(/* DO NOT CHANGE THIS WITHOUT READING THE ABOVE. WE DO THIS STRANGE COMPARISON FOR NaN values*/
                  !(splittingScore <= bestSplittingScore))) {
                  bestSplittingScore = splittingScore;
                  cutFirst2Best = iBin2;
                  cutFirst2LowBest = cutSecond2LowBest;
                  cutFirst2HighBest = cutSecond2HighBest;

                  pTotals2LowLowBest->Copy(*pTotals1LowLowBestInner, cVectorLength);
                  pTotals2LowHighBest->Copy(*pTotals1LowHighBestInner, cVectorLength);
                  pTotals2HighLowBest->Copy(*pTotals1HighLowBestInner, cVectorLength);
                  pTotals2HighHighBest->Copy(*pTotals1HighHighBestInner, cVectorLength);

                  bCutFirst2 = true;
               } else {
                  EBM_ASSERT(!std::isnan(splittingScore));
               }
            } else {
               EBM_ASSERT(!std::isnan(splittingScoreNew2));
               EBM_ASSERT(k_illegalGain == splittingScoreNew2);
            }
         } else {
            EBM_ASSERT(!std::isnan(splittingScoreNew1));
            EBM_ASSERT(k_illegalGain == splittingScoreNew1);
         }
         ++iBin2;
      } while(iBin2 < cBinsDimension2 - 1);
      LOG_0(TraceLevelVerbose, "PartitionTwoDimensionalBoostingInternal Done sweep loops");

      FloatEbmType gain;
      // if we get a NaN result for bestSplittingScore, we might as well do less work and just create a zero split update right now.  The rules 
      // for NaN values say that non equality comparisons are all false so, let's flip this comparison such that it should be true for NaN values.  
      // If the compiler violates NaN comparions rules, no big deal.  NaN values will get us soon and shut down boosting.
      if(UNLIKELY(/* DO NOT CHANGE THIS WITHOUT READING THE ABOVE. WE DO THIS STRANGE COMPARISON FOR NaN values*/ !(k_illegalGain != bestSplittingScore))) {
         // there were no good cuts found, or we hit a NaN value
#ifndef NDEBUG
         const ErrorEbmType errorDebug1 =
#endif // NDEBUG
         pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 0);
         // we can't fail since we're setting this to zero, so no allocations.  We don't in fact need the division array at all
         EBM_ASSERT(Error_None == errorDebug1);

#ifndef NDEBUG
         const ErrorEbmType errorDebug2 =
#endif // NDEBUG
         pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 0);
         // we can't fail since we're setting this to zero, so no allocations.  We don't in fact need the division array at all
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
                  cWeightInParentBucket
               );
            }

            pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[iVector] = update;
         }
         gain = FloatEbmType { 0 }; // no splits means no gain
      } else {
         EBM_ASSERT(!std::isnan(bestSplittingScore));
         EBM_ASSERT(k_illegalGain != bestSplittingScore);
         if(bCutFirst2) {
            // if bCutFirst2 is true, then there definetly was a cut, so we don't have to check for zero cuts
            error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 1);
            if(Error_None != error) {
               // already logged
               return error;
            }
            pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = cutFirst2Best;

            if(cutFirst2LowBest < cutFirst2HighBest) {
               error = pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6);
               if(Error_None != error) {
                  // already logged
                  return error;
               }
               error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 2);
               if(Error_None != error) {
                  // already logged
                  return error;
               }
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = cutFirst2LowBest;
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[1] = cutFirst2HighBest;
            } else if(cutFirst2HighBest < cutFirst2LowBest) {
               error = pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6);
               if(Error_None != error) {
                  // already logged
                  return error;
               }
               error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 2);
               if(Error_None != error) {
                  // already logged
                  return error;
               }
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = cutFirst2HighBest;
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[1] = cutFirst2LowBest;
            } else {
               error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1);
               if(Error_None != error) {
                  // already logged
                  return error;
               }

               error = pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4);
               if(Error_None != error) {
                  // already logged
                  return error;
               }
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = cutFirst2LowBest;
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

               if(cutFirst2LowBest < cutFirst2HighBest) {
                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[0 * cVectorLength + iVector] = predictionLowLow;
                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[1 * cVectorLength + iVector] = predictionLowHigh;
                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[2 * cVectorLength + iVector] = predictionLowHigh;
                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[3 * cVectorLength + iVector] = predictionHighLow;
                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[4 * cVectorLength + iVector] = predictionHighLow;
                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[5 * cVectorLength + iVector] = predictionHighHigh;
               } else if(cutFirst2HighBest < cutFirst2LowBest) {
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
            error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1);
            if(Error_None != error) {
               // already logged
               return error;
            }
            pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = cutFirst1Best;

            if(cutFirst1LowBest < cutFirst1HighBest) {
               error = pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6);
               if(Error_None != error) {
                  // already logged
                  return error;
               }

               error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 2);
               if(Error_None != error) {
                  // already logged
                  return error;
               }
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = cutFirst1LowBest;
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[1] = cutFirst1HighBest;
            } else if(cutFirst1HighBest < cutFirst1LowBest) {
               error = pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6);
               if(Error_None != error) {
                  // already logged
                  return error;
               }

               error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 2);
               if(Error_None != error) {
                  // already logged
                  return error;
               }
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = cutFirst1HighBest;
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[1] = cutFirst1LowBest;
            } else {
               error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 1);
               if(Error_None != error) {
                  // already logged
                  return error;
               }
               error = pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4);
               if(Error_None != error) {
                  // already logged
                  return error;
               }
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = cutFirst1LowBest;
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
               if(cutFirst1LowBest < cutFirst1HighBest) {
                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[0 * cVectorLength + iVector] = predictionLowLow;
                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[1 * cVectorLength + iVector] = predictionHighLow;
                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[2 * cVectorLength + iVector] = predictionLowHigh;
                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[3 * cVectorLength + iVector] = predictionHighLow;
                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[4 * cVectorLength + iVector] = predictionLowHigh;
                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[5 * cVectorLength + iVector] = predictionHighHigh;
               } else if(cutFirst1HighBest < cutFirst1LowBest) {
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
         // for regression, bestSplittingScore and splittingScoreParent can be infinity.  There is a super-super-super-rare case where we can have 
         // splittingScoreParent overflow to +infinity due to numeric issues, but not bestSplittingScore, and then the subtration causes the result 
         // to be -infinity.  The universe will probably die of heat death before we get a -infinity value, but perhaps an adversarial dataset could 
         // trigger it, and we don't want someone giving us data to use a vulnerability in our system, so check for it!
         gain = bestSplittingScore - splittingScoreParent;
      }

      // TODO: this gain value is untested.  We should build a new test that compares the single feature gains to the multi-dimensional gains by
      // making a pair where one of the dimensions duplicates values in the 0 and 1 bin.  Then the gain should be identical, if there is only 1 split allowed
      *pTotalGain = gain;
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
      HistogramBucketBase * const pTotal,
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
            pTotal,
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
            pTotal,
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
      HistogramBucketBase * const pTotal,
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
         pTotal,
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
   HistogramBucketBase * const pTotal,
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
         pTotal,
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
         pTotal,
         pTotalGain
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
#endif // NDEBUG
      );
   }
}

} // DEFINED_ZONE_NAME
