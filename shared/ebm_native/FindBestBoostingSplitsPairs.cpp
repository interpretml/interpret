// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "SegmentedTensor.h"
#include "EbmStats.h"

#include "FeatureAtomic.h"
#include "FeatureGroup.h"

#include "HistogramTargetEntry.h"
#include "HistogramBucket.h"

#include "Booster.h"
#include "ThreadStateBoosting.h"

#include "TensorTotalsSum.h"

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
static FloatEbmType SweepMultiDiemensional(
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

   EBM_ASSERT(1 <= pFeatureGroup->GetCountSignificantFeatures());
   EBM_ASSERT(iDimensionSweep < pFeatureGroup->GetCountSignificantFeatures());
   EBM_ASSERT(0 == (directionVectorLow & (size_t { 1 } << iDimensionSweep)));

   const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
      compilerLearningTypeOrCountTargetClasses,
      runtimeLearningTypeOrCountTargetClasses
   );
   const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow(bClassification, cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);
   EBM_ASSERT(!IsMultiplyError(2, cBytesPerHistogramBucket)); // we're accessing allocated memory
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

      EBM_ASSERT(2 == pFeatureGroup->GetCountSignificantFeatures()); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
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
         EBM_ASSERT(2 == pFeatureGroup->GetCountSignificantFeatures()); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
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

            FloatEbmType cLowSamplesInBucket = static_cast<FloatEbmType>(pTotalsLow->GetCountSamplesInBucket());
            FloatEbmType cHighSamplesInBucket = static_cast<FloatEbmType>(pTotalsHigh->GetCountSamplesInBucket());

            HistogramBucketVectorEntry<bClassification> * const pHistogramBucketVectorEntryLow =
               pTotalsLow->GetHistogramBucketVectorEntry();

            HistogramBucketVectorEntry<bClassification> * const pHistogramBucketVectorEntryHigh =
               pTotalsHigh->GetHistogramBucketVectorEntry();

            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               // TODO : we can make this faster by doing the division in ComputeNodeSplittingScore after we add all the numerators 
               // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

               const FloatEbmType splittingScoreUpdate1 = EbmStats::ComputeNodeSplittingScore(
                  pHistogramBucketVectorEntryLow[iVector].m_sumResidualError, cLowSamplesInBucket);
               EBM_ASSERT(std::isnan(splittingScoreUpdate1) || FloatEbmType { 0 } <= splittingScoreUpdate1);
               splittingScore += splittingScoreUpdate1;
               const FloatEbmType splittingScoreUpdate2 = EbmStats::ComputeNodeSplittingScore(
                  pHistogramBucketVectorEntryHigh[iVector].m_sumResidualError, cHighSamplesInBucket);
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
class FindBestBoostingSplitPairsInternal final {
public:

   FindBestBoostingSplitPairsInternal() = delete; // this is a static class.  Do not construct

   WARNING_PUSH
   WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE

   static bool Func(
      ThreadStateBoosting * const pThreadStateBoosting,
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

      Booster * const pBooster = pThreadStateBoosting->GetBooster();

      HistogramBucketBase * const aHistogramBucketBase = pThreadStateBoosting->GetHistogramBucketBase();
      SegmentedTensor * const pSmallChangeToModelOverwriteSingleSamplingSet =
         pThreadStateBoosting->GetSmallChangeToModelOverwriteSingleSamplingSet();

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         pBooster->GetRuntimeLearningTypeOrCountTargetClasses()
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

      EBM_ASSERT(2 == pFeatureGroup->GetCountSignificantFeatures());
      size_t cBinsDimension1 = 0;
      size_t cBinsDimension2 = 0;
      const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
      const FeatureGroupEntry * const pFeatureGroupEntryEnd = pFeatureGroupEntry + pFeatureGroup->GetCountFeatures();
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

      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pTotal, pThreadStateBoosting->GetHistogramBucketsEndDebug());

      EBM_ASSERT(0 < cSamplesRequiredForChildSplitMin);

      FloatEbmType splittingScoreParent = FloatEbmType { 0 };
      EBM_ASSERT(0 < pTotal->GetCountSamplesInBucket());
      const FloatEbmType cSamplesInParentBucket = static_cast<FloatEbmType>(pTotal->GetCountSamplesInBucket());

      HistogramBucketVectorEntry<bClassification> * const pHistogramBucketVectorEntryTotal =
         pTotal->GetHistogramBucketVectorEntry();

      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         // TODO : we can make this faster by doing the division in ComputeNodeSplittingScoreParent after we add all the numerators 
         // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

         const FloatEbmType splittingScoreParentUpdate = EbmStats::ComputeNodeSplittingScore(
            pHistogramBucketVectorEntryTotal[iVector].m_sumResidualError,
            cSamplesInParentBucket
         );
         EBM_ASSERT(std::isnan(splittingScoreParentUpdate) || FloatEbmType { 0 } <= splittingScoreParentUpdate);
         splittingScoreParent += splittingScoreParentUpdate;
      }
      EBM_ASSERT(std::isnan(splittingScoreParent) || FloatEbmType { 0 } <= splittingScoreParent); // sumation of positive numbers should be positive

      LOG_0(TraceLevelVerbose, "BoostMultiDimensional Starting FIRST bin sweep loop");
      size_t iBin1 = 0;
      do {
         aiStart[0] = iBin1;

         splittingScore = FloatEbmType { 0 };

         size_t cutSecond1LowBest;
         HistogramBucket<bClassification> * pTotals2LowLowBest =
            GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 4);
         HistogramBucket<bClassification> * pTotals2LowHighBest =
            GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 5);
         const FloatEbmType splittingScoreNew1 = SweepMultiDiemensional<compilerLearningTypeOrCountTargetClasses>(
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
            , pThreadStateBoosting->GetHistogramBucketsEndDebug()
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
            const FloatEbmType splittingScoreNew2 = SweepMultiDiemensional<compilerLearningTypeOrCountTargetClasses>(
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
               , pThreadStateBoosting->GetHistogramBucketsEndDebug()
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

      LOG_0(TraceLevelVerbose, "BoostMultiDimensional Starting SECOND bin sweep loop");
      size_t iBin2 = 0;
      do {
         aiStart[1] = iBin2;

         splittingScore = FloatEbmType { 0 };

         size_t cutSecond2LowBest;
         HistogramBucket<bClassification> * pTotals1LowLowBestInner =
            GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 16);
         HistogramBucket<bClassification> * pTotals1LowHighBestInner =
            GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 17);
         const FloatEbmType splittingScoreNew1 = SweepMultiDiemensional<compilerLearningTypeOrCountTargetClasses>(
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
            , pThreadStateBoosting->GetHistogramBucketsEndDebug()
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
            const FloatEbmType splittingScoreNew2 = SweepMultiDiemensional<compilerLearningTypeOrCountTargetClasses>(
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
               , pThreadStateBoosting->GetHistogramBucketsEndDebug()
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
      LOG_0(TraceLevelVerbose, "BoostMultiDimensional Done sweep loops");

      FloatEbmType gain;
      // if we get a NaN result for bestSplittingScore, we might as well do less work and just create a zero split update right now.  The rules 
      // for NaN values say that non equality comparisons are all false so, let's flip this comparison such that it should be true for NaN values.  
      // If the compiler violates NaN comparions rules, no big deal.  NaN values will get us soon and shut down boosting.
      if(UNLIKELY(/* DO NOT CHANGE THIS WITHOUT READING THE ABOVE. WE DO THIS STRANGE COMPARISON FOR NaN values*/ !(k_illegalGain != bestSplittingScore))) {
         // there were no good cuts found, or we hit a NaN value
#ifndef NDEBUG
         const bool bSetCountDivisions0 =
#endif // NDEBUG
            pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 0);
         // we can't fail since we're setting this to zero, so no allocations.  We don't in fact need the division array at all
         EBM_ASSERT(!bSetCountDivisions0);

#ifndef NDEBUG
         const bool bSetCountDivisions1 =
#endif // NDEBUG
            pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 0);
         // we can't fail since we're setting this to zero, so no allocations.  We don't in fact need the division array at all
         EBM_ASSERT(!bSetCountDivisions1);

         // we don't need to call pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity, 
         // since our value capacity would be 1, which is pre-allocated

         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            FloatEbmType prediction;

            if(bClassification) {
               prediction = EbmStats::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                  pHistogramBucketVectorEntryTotal[iVector].m_sumResidualError,
                  pHistogramBucketVectorEntryTotal[iVector].GetSumDenominator()
               );
            } else {
               EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
               prediction = EbmStats::ComputeSmallChangeForOneSegmentRegression(
                  pHistogramBucketVectorEntryTotal[iVector].m_sumResidualError,
                  cSamplesInParentBucket
               );
            }
            pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[iVector] = prediction;
         }
         gain = FloatEbmType { 0 }; // no splits means no gain
      } else {
         EBM_ASSERT(!std::isnan(bestSplittingScore));
         EBM_ASSERT(k_illegalGain != bestSplittingScore);
         if(bCutFirst2) {
            // if bCutFirst2 is true, then there definetly was a cut, so we don't have to check for zero cuts
            if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 1)) {
               LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 1)");
               return true;
            }
            pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = cutFirst2Best;

            if(cutFirst2LowBest < cutFirst2HighBest) {
               if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)) {
                  LOG_0(
                     TraceLevelWarning,
                     "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)"
                  );
                  return true;
               }
               if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 2)) {
                  LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 2)");
                  return true;
               }
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = cutFirst2LowBest;
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[1] = cutFirst2HighBest;
            } else if(cutFirst2HighBest < cutFirst2LowBest) {
               if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)) {
                  LOG_0(
                     TraceLevelWarning,
                     "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)"
                  );
                  return true;
               }
               if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 2)) {
                  LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 2)");
                  return true;
               }
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = cutFirst2HighBest;
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[1] = cutFirst2LowBest;
            } else {
               if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1)) {
                  LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1)");
                  return true;
               }

               if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4)) {
                  LOG_0(
                     TraceLevelWarning,
                     "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4)"
                  );
                  return true;
               }
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = cutFirst2LowBest;
            }

            HistogramBucketVectorEntry<bClassification> * const pHistogramBucketVectorEntryTotals2LowLowBest =
               pTotals2LowLowBest->GetHistogramBucketVectorEntry();
            HistogramBucketVectorEntry<bClassification> * const pHistogramBucketVectorEntryTotals2LowHighBest =
               pTotals2LowHighBest->GetHistogramBucketVectorEntry();
            HistogramBucketVectorEntry<bClassification> * const pHistogramBucketVectorEntryTotals2HighLowBest =
               pTotals2HighLowBest->GetHistogramBucketVectorEntry();
            HistogramBucketVectorEntry<bClassification> * const pHistogramBucketVectorEntryTotals2HighHighBest =
               pTotals2HighHighBest->GetHistogramBucketVectorEntry();

            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               FloatEbmType predictionLowLow;
               FloatEbmType predictionLowHigh;
               FloatEbmType predictionHighLow;
               FloatEbmType predictionHighHigh;

               if(bClassification) {
                  predictionLowLow = EbmStats::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     pHistogramBucketVectorEntryTotals2LowLowBest[iVector].m_sumResidualError,
                     pHistogramBucketVectorEntryTotals2LowLowBest[iVector].GetSumDenominator()
                  );
                  predictionLowHigh = EbmStats::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     pHistogramBucketVectorEntryTotals2LowHighBest[iVector].m_sumResidualError,
                     pHistogramBucketVectorEntryTotals2LowHighBest[iVector].GetSumDenominator()
                  );
                  predictionHighLow = EbmStats::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     pHistogramBucketVectorEntryTotals2HighLowBest[iVector].m_sumResidualError,
                     pHistogramBucketVectorEntryTotals2HighLowBest[iVector].GetSumDenominator()
                  );
                  predictionHighHigh = EbmStats::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     pHistogramBucketVectorEntryTotals2HighHighBest[iVector].m_sumResidualError,
                     pHistogramBucketVectorEntryTotals2HighHighBest[iVector].GetSumDenominator()
                  );
               } else {
                  EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
                  predictionLowLow = EbmStats::ComputeSmallChangeForOneSegmentRegression(
                     pHistogramBucketVectorEntryTotals2LowLowBest[iVector].m_sumResidualError,
                     static_cast<FloatEbmType>(pTotals2LowLowBest->GetCountSamplesInBucket())
                  );
                  predictionLowHigh = EbmStats::ComputeSmallChangeForOneSegmentRegression(
                     pHistogramBucketVectorEntryTotals2LowHighBest[iVector].m_sumResidualError,
                     static_cast<FloatEbmType>(pTotals2LowHighBest->GetCountSamplesInBucket())
                  );
                  predictionHighLow = EbmStats::ComputeSmallChangeForOneSegmentRegression(
                     pHistogramBucketVectorEntryTotals2HighLowBest[iVector].m_sumResidualError,
                     static_cast<FloatEbmType>(pTotals2HighLowBest->GetCountSamplesInBucket())
                  );
                  predictionHighHigh = EbmStats::ComputeSmallChangeForOneSegmentRegression(
                     pHistogramBucketVectorEntryTotals2HighHighBest[iVector].m_sumResidualError,
                     static_cast<FloatEbmType>(pTotals2HighHighBest->GetCountSamplesInBucket())
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
            if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1)) {
               LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1)");
               return true;
            }
            pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = cutFirst1Best;

            if(cutFirst1LowBest < cutFirst1HighBest) {
               if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)) {
                  LOG_0(
                     TraceLevelWarning,
                     "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)"
                  );
                  return true;
               }

               if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 2)) {
                  LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 2)");
                  return true;
               }
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = cutFirst1LowBest;
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[1] = cutFirst1HighBest;
            } else if(cutFirst1HighBest < cutFirst1LowBest) {
               if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)) {
                  LOG_0(
                     TraceLevelWarning,
                     "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)"
                  );
                  return true;
               }

               if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 2)) {
                  LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 2)");
                  return true;
               }
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = cutFirst1HighBest;
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[1] = cutFirst1LowBest;
            } else {
               if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 1)) {
                  LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 1)");
                  return true;
               }
               if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4)) {
                  LOG_0(
                     TraceLevelWarning,
                     "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4)"
                  );
                  return true;
               }
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = cutFirst1LowBest;
            }

            HistogramBucketVectorEntry<bClassification> * const pHistogramBucketVectorEntryTotals1LowLowBest =
               pTotals1LowLowBest->GetHistogramBucketVectorEntry();
            HistogramBucketVectorEntry<bClassification> * const pHistogramBucketVectorEntryTotals1LowHighBest =
               pTotals1LowHighBest->GetHistogramBucketVectorEntry();
            HistogramBucketVectorEntry<bClassification> * const pHistogramBucketVectorEntryTotals1HighLowBest =
               pTotals1HighLowBest->GetHistogramBucketVectorEntry();
            HistogramBucketVectorEntry<bClassification> * const pHistogramBucketVectorEntryTotals1HighHighBest =
               pTotals1HighHighBest->GetHistogramBucketVectorEntry();

            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               FloatEbmType predictionLowLow;
               FloatEbmType predictionLowHigh;
               FloatEbmType predictionHighLow;
               FloatEbmType predictionHighHigh;

               if(bClassification) {
                  predictionLowLow = EbmStats::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     pHistogramBucketVectorEntryTotals1LowLowBest[iVector].m_sumResidualError,
                     pHistogramBucketVectorEntryTotals1LowLowBest[iVector].GetSumDenominator()
                  );
                  predictionLowHigh = EbmStats::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     pHistogramBucketVectorEntryTotals1LowHighBest[iVector].m_sumResidualError,
                     pHistogramBucketVectorEntryTotals1LowHighBest[iVector].GetSumDenominator()
                  );
                  predictionHighLow = EbmStats::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     pHistogramBucketVectorEntryTotals1HighLowBest[iVector].m_sumResidualError,
                     pHistogramBucketVectorEntryTotals1HighLowBest[iVector].GetSumDenominator()
                  );
                  predictionHighHigh = EbmStats::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     pHistogramBucketVectorEntryTotals1HighHighBest[iVector].m_sumResidualError,
                     pHistogramBucketVectorEntryTotals1HighHighBest[iVector].GetSumDenominator()
                  );
               } else {
                  EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
                  predictionLowLow = EbmStats::ComputeSmallChangeForOneSegmentRegression(
                     pHistogramBucketVectorEntryTotals1LowLowBest[iVector].m_sumResidualError,
                     static_cast<FloatEbmType>(pTotals1LowLowBest->GetCountSamplesInBucket())
                  );
                  predictionLowHigh = EbmStats::ComputeSmallChangeForOneSegmentRegression(
                     pHistogramBucketVectorEntryTotals1LowHighBest[iVector].m_sumResidualError,
                     static_cast<FloatEbmType>(pTotals1LowHighBest->GetCountSamplesInBucket())
                  );
                  predictionHighLow = EbmStats::ComputeSmallChangeForOneSegmentRegression(
                     pHistogramBucketVectorEntryTotals1HighLowBest[iVector].m_sumResidualError,
                     static_cast<FloatEbmType>(pTotals1HighLowBest->GetCountSamplesInBucket())
                  );
                  predictionHighHigh = EbmStats::ComputeSmallChangeForOneSegmentRegression(
                     pHistogramBucketVectorEntryTotals1HighHighBest[iVector].m_sumResidualError,
                     static_cast<FloatEbmType>(pTotals1HighHighBest->GetCountSamplesInBucket())
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
      return false;
   }
   WARNING_POP
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class FindBestBoostingSplitPairsTarget final {
public:

   FindBestBoostingSplitPairsTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static bool Func(
      ThreadStateBoosting * const pThreadStateBoosting,
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

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         return FindBestBoostingSplitPairsInternal<compilerLearningTypeOrCountTargetClassesPossible>::Func(
            pThreadStateBoosting,
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
         return FindBestBoostingSplitPairsTarget<compilerLearningTypeOrCountTargetClassesPossible + 1>::Func(
            pThreadStateBoosting,
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
class FindBestBoostingSplitPairsTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   FindBestBoostingSplitPairsTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static bool Func(
      ThreadStateBoosting * const pThreadStateBoosting,
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

      EBM_ASSERT(IsClassification(pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses());

      return FindBestBoostingSplitPairsInternal<k_dynamicClassification>::Func(
         pThreadStateBoosting,
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

extern bool FindBestBoostingSplitPairs(
   ThreadStateBoosting * const pThreadStateBoosting,
   const FeatureGroup * const pFeatureGroup,
   const size_t cSamplesRequiredForChildSplitMin,
   HistogramBucketBase * pAuxiliaryBucketZone,
   HistogramBucketBase * const pTotal,
   FloatEbmType * const pTotalGain
#ifndef NDEBUG
   , const HistogramBucketBase * const aHistogramBucketsDebugCopy
#endif // NDEBUG
) {
   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();

   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      return FindBestBoostingSplitPairsTarget<2>::Func(
         pThreadStateBoosting,
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
      return FindBestBoostingSplitPairsInternal<k_regression>::Func(
         pThreadStateBoosting,
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
