// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef DIMENSION_MULTIPLE_H
#define DIMENSION_MULTIPLE_H

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "SegmentedTensor.h"
#include "EbmStatisticUtils.h"
#include "CachedThreadResourcesBoosting.h"
#include "CachedThreadResourcesInteraction.h"
#include "Feature.h"
#include "FeatureGroup.h"
#include "SamplingSet.h"
#include "HistogramTargetEntry.h"
#include "HistogramBucket.h"

#include "Booster.h"
#include "InteractionDetection.h"

#include "TensorTotalsSum.h"

void TensorTotalsBuild(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const FeatureCombination * const pFeatureCombination,
   HistogramBucketBase * pBucketAuxiliaryBuildZone,
   HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
   , HistogramBucketBase * const aHistogramBucketsDebugCopy
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
);

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

void BinInteraction(
   EbmInteractionState * const pEbmInteractionState,
   const FeatureCombination * const pFeatureCombination,
   HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
);

FloatEbmType FindBestInteractionGainPairs(
   EbmInteractionState * const pEbmInteractionState,
   const FeatureCombination * const pFeatureCombination,
   const size_t cInstancesRequiredForChildSplitMin,
   HistogramBucketBase * pAuxiliaryBucketZone,
   HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
   , const HistogramBucketBase * const aHistogramBucketsDebugCopy
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
);


template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerCountDimensions>
FloatEbmType SweepMultiDiemensional(
   const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, 
   const FeatureCombination * const pFeatureCombination, 
   size_t * const aiPoint, 
   const size_t directionVectorLow, 
   const unsigned int iDimensionSweep, 
   const size_t cInstancesRequiredForChildSplitMin, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, 
   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pHistogramBucketBestAndTemp, 
   size_t * const piBestCut
#ifndef NDEBUG
   , const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy, 
   const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

   // don't LOG this!  It would create way too much chatter!

   // TODO : optimize this function

   EBM_ASSERT(1 <= pFeatureCombination->GetCountFeatures());
   EBM_ASSERT(iDimensionSweep < pFeatureCombination->GetCountFeatures());
   EBM_ASSERT(0 == (directionVectorLow & (size_t { 1 } << iDimensionSweep)));

   const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
      compilerLearningTypeOrCountTargetClasses,
      runtimeLearningTypeOrCountTargetClasses
   );
   const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bClassification>(cVectorLength);
   EBM_ASSERT(!IsMultiplyError(2, cBytesPerHistogramBucket)); // we're accessing allocated memory
   const size_t cBytesPerTwoHistogramBuckets = cBytesPerHistogramBucket << 1;

   size_t * const piBin = &aiPoint[iDimensionSweep];
   *piBin = 0;
   size_t directionVectorHigh = directionVectorLow | size_t { 1 } << iDimensionSweep;

   const size_t cBins = pFeatureCombination->GetFeatureCombinationEntries()[iDimensionSweep].m_pFeature->GetCountBins();
   EBM_ASSERT(2 <= cBins);

   size_t iBestCut = 0;

   HistogramBucket<bClassification> * const pTotalsLow =
      GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pHistogramBucketBestAndTemp, 2);
   ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pTotalsLow, aHistogramBucketsEndDebug);

   HistogramBucket<bClassification> * const pTotalsHigh =
      GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pHistogramBucketBestAndTemp, 3);
   ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pTotalsHigh, aHistogramBucketsEndDebug);

   EBM_ASSERT(0 < cInstancesRequiredForChildSplitMin);

   FloatEbmType bestSplit = k_illegalGain;
   size_t iBin = 0;
   do {
      *piBin = iBin;

      TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, compilerCountDimensions>(
         runtimeLearningTypeOrCountTargetClasses,
         pFeatureCombination,
         aHistogramBuckets,
         aiPoint,
         directionVectorLow, 
         pTotalsLow
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
      if(LIKELY(cInstancesRequiredForChildSplitMin <= pTotalsLow->m_cInstancesInBucket)) {
         TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, compilerCountDimensions>(
            runtimeLearningTypeOrCountTargetClasses,
            pFeatureCombination,
            aHistogramBuckets,
            aiPoint,
            directionVectorHigh, 
            pTotalsHigh
   #ifndef NDEBUG
            , aHistogramBucketsDebugCopy
            , aHistogramBucketsEndDebug
   #endif // NDEBUG
         );
         if(LIKELY(cInstancesRequiredForChildSplitMin <= pTotalsHigh->m_cInstancesInBucket)) {
            FloatEbmType splittingScore = FloatEbmType { 0 };
            EBM_ASSERT(0 < pTotalsLow->m_cInstancesInBucket);
            EBM_ASSERT(0 < pTotalsHigh->m_cInstancesInBucket);

            FloatEbmType cLowInstancesInBucket = static_cast<FloatEbmType>(pTotalsLow->m_cInstancesInBucket);
            FloatEbmType cHighInstancesInBucket = static_cast<FloatEbmType>(pTotalsHigh->m_cInstancesInBucket);
            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               // TODO : we can make this faster by doing the division in ComputeNodeSplittingScore after we add all the numerators 
               // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

               const FloatEbmType splittingScoreUpdate1 = EbmStatistics::ComputeNodeSplittingScore(
                  ArrayToPointer(pTotalsLow->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, cLowInstancesInBucket);
               EBM_ASSERT(std::isnan(splittingScoreUpdate1) || FloatEbmType { 0 } <= splittingScoreUpdate1);
               splittingScore += splittingScoreUpdate1;
               const FloatEbmType splittingScoreUpdate2 = EbmStatistics::ComputeNodeSplittingScore(
                  ArrayToPointer(pTotalsHigh->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, cHighInstancesInBucket);
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
                  GetHistogramBucketByIndex<bClassification>(
                     cBytesPerHistogramBucket, 
                     pHistogramBucketBestAndTemp, 
                     1
                  ), 
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
   } while(iBin < cBins - 1);
   *piBestCut = iBestCut;

   EBM_ASSERT(std::isnan(bestSplit) || bestSplit == k_illegalGain || FloatEbmType { 0 } <= bestSplit); // sumation of positive numbers should be positive
   return bestSplit;
}

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE

// TODO: for higher dimensional spaces, we need to add/subtract individual cells alot and the denominator isn't required in order to make decisions about
//   where to cut.  For dimensions higher than 2, we might want to copy the tensor to a new tensor AFTER binning that keeps only the residuals and then 
//    go back to our original tensor after splits to determine the denominator
// TODO: do we really require compilerCountDimensions here?  Does it make any of the code below faster... or alternatively, should we puth the 
//    distinction down into a sub-function
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerCountDimensions>
bool BoostMultiDimensional(
   EbmBoostingState * const pEbmBoostingState,
   const FeatureCombination * const pFeatureCombination, 
   const SamplingSet * const pTrainingSet,
   const size_t cInstancesRequiredForChildSplitMin,
   SegmentedTensor * const pSmallChangeToModelOverwriteSingleSamplingSet,
   FloatEbmType * const pTotalGain
) {
   constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelVerbose, "Entered BoostMultiDimensional");

   // TODO: we can just re-generate this code 63 times and eliminate the dynamic cDimensions value.  We can also do this in several other 
   // places like for SegmentedRegion and other critical places
   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(compilerCountDimensions, pFeatureCombination->GetCountFeatures());
   EBM_ASSERT(2 <= cDimensions);

   size_t cAuxillaryBucketsForBuildFastTotals = 0;
   size_t cTotalBucketsMainSpace = 1;
   for(size_t iDimension = 0; iDimension < cDimensions; ++iDimension) {
      const size_t cBins = pFeatureCombination->GetFeatureCombinationEntries()[iDimension].m_pFeature->GetCountBins();
      // we filer out 1 == cBins in allocation.  If cBins could be 1, then we'd need to check at runtime for overflow of cAuxillaryBucketsForBuildFastTotals
      EBM_ASSERT(2 <= cBins);
      // if this wasn't true then we'd have to check IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace) at runtime
      EBM_ASSERT(cAuxillaryBucketsForBuildFastTotals < cTotalBucketsMainSpace);
      // since cBins must be 2 or more, cAuxillaryBucketsForBuildFastTotals must grow slower than cTotalBucketsMainSpace, and we checked at 
      // allocation that cTotalBucketsMainSpace would not overflow
      EBM_ASSERT(!IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace));
      cAuxillaryBucketsForBuildFastTotals += cTotalBucketsMainSpace;
      // we check for simple multiplication overflow from m_cBins in EbmBoostingState->Initialize when we unpack featureCombinationIndexes
      EBM_ASSERT(!IsMultiplyError(cTotalBucketsMainSpace, cBins));
      cTotalBucketsMainSpace *= cBins;
      // if this wasn't true then we'd have to check IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace) at runtime
      EBM_ASSERT(cAuxillaryBucketsForBuildFastTotals < cTotalBucketsMainSpace);
   }
   // we need to reserve 4 PAST the pointer we pass into SweepMultiDiemensional!!!!.  We pass in index 20 at max, so we need 24
   const size_t cAuxillaryBucketsForSplitting = 24;
   const size_t cAuxillaryBuckets = 
      cAuxillaryBucketsForBuildFastTotals < cAuxillaryBucketsForSplitting ? cAuxillaryBucketsForSplitting : cAuxillaryBucketsForBuildFastTotals;
   if(IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)");
      return true;
   }
   const size_t cTotalBuckets =  cTotalBucketsMainSpace + cAuxillaryBuckets;

   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses();

   const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
      compilerLearningTypeOrCountTargetClasses,
      runtimeLearningTypeOrCountTargetClasses
   );
   const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
   if(GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)) {
      LOG_0(
         TraceLevelWarning, 
         "WARNING BoostMultiDimensional GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)"
      );
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bClassification>(cVectorLength);
   if(IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)) {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)");
      return true;
   }
   const size_t cBytesBuffer = cTotalBuckets * cBytesPerHistogramBucket;

   CachedBoostingThreadResources * const pCachedThreadResources = pEbmBoostingState->GetCachedThreadResources();

   // we don't need to free this!  It's tracked and reused by pCachedThreadResources
   HistogramBucket<bClassification> * const aHistogramBuckets =
      static_cast<HistogramBucket<bClassification> *>(pCachedThreadResources->GetThreadByteBuffer1(cBytesBuffer));
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional nullptr == aHistogramBuckets");
      return true;
   }
   for(size_t i = 0; i < cTotalBuckets; ++i) {
      HistogramBucket<bClassification> * const pHistogramBucket =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, aHistogramBuckets, i);
      pHistogramBucket->Zero(cVectorLength);
   }

   HistogramBucket<bClassification> * pAuxiliaryBucketZone =
      GetHistogramBucketByIndex<bClassification>(
         cBytesPerHistogramBucket, 
         aHistogramBuckets, 
         cTotalBucketsMainSpace
      );

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

#ifndef NDEBUG
   // make a copy of the original binned buckets for debugging purposes
   size_t cTotalBucketsDebug = 1;
   for(size_t iDimensionDebug = 0; iDimensionDebug < cDimensions; ++iDimensionDebug) {
      const size_t cBins = pFeatureCombination->GetFeatureCombinationEntries()[iDimensionDebug].m_pFeature->GetCountBins();
      EBM_ASSERT(!IsMultiplyError(cTotalBucketsDebug, cBins)); // we checked this above
      cTotalBucketsDebug *= cBins;
   }
   // we wouldn't have been able to allocate our main buffer above if this wasn't ok
   EBM_ASSERT(!IsMultiplyError(cTotalBucketsDebug, cBytesPerHistogramBucket));
   HistogramBucket<bClassification> * const aHistogramBucketsDebugCopy =
      EbmMalloc<HistogramBucket<bClassification>>(cTotalBucketsDebug, cBytesPerHistogramBucket);
   if(nullptr != aHistogramBucketsDebugCopy) {
      // if we can't allocate, don't fail.. just stop checking
      const size_t cBytesBufferDebug = cTotalBucketsDebug * cBytesPerHistogramBucket;
      memcpy(aHistogramBucketsDebugCopy, aHistogramBuckets, cBytesBufferDebug);
   }
#endif // NDEBUG

   TensorTotalsBuild(
      runtimeLearningTypeOrCountTargetClasses,
      pFeatureCombination,
      pAuxiliaryBucketZone,
      aHistogramBuckets
#ifndef NDEBUG
      , aHistogramBucketsDebugCopy
      , aHistogramBucketsEndDebug
#endif // NDEBUG
   );

   //permutation0
   //gain_permute0
   //  divs0
   //  gain0
   //    divs00
   //    gain00
   //      divs000
   //      gain000
   //      divs001
   //      gain001
   //    divs01
   //    gain01
   //      divs010
   //      gain010
   //      divs011
   //      gain011
   //  divs1
   //  gain1
   //    divs10
   //    gain10
   //      divs100
   //      gain100
   //      divs101
   //      gain101
   //    divs11
   //    gain11
   //      divs110
   //      gain110
   //      divs111
   //      gain111
   //---------------------------
   //permutation1
   //gain_permute1
   //  divs0
   //  gain0
   //    divs00
   //    gain00
   //      divs000
   //      gain000
   //      divs001
   //      gain001
   //    divs01
   //    gain01
   //      divs010
   //      gain010
   //      divs011
   //      gain011
   //  divs1
   //  gain1
   //    divs10
   //    gain10
   //      divs100
   //      gain100
   //      divs101
   //      gain101
   //    divs11
   //    gain11
   //      divs110
   //      gain110
   //      divs111
   //      gain111       *


   //size_t aiDimensionPermutation[k_cDimensionsMax];
   //for(unsigned int iDimensionInitialize = 0; iDimensionInitialize < cDimensions; ++iDimensionInitialize) {
   //   aiDimensionPermutation[iDimensionInitialize] = iDimensionInitialize;
   //}
   //size_t aiDimensionPermutationBest[k_cDimensionsMax];

   // DO this is a fixed length that we should make variable!
   //size_t aDOSplits[1000000];
   //size_t aDOSplitsBest[1000000];

   //do {
   //   size_t aiDimensions[k_cDimensionsMax];
   //   memset(aiDimensions, 0, sizeof(aiDimensions[0]) * cDimensions));
   //   while(true) {


   //      EBM_ASSERT(0 == iDimension);
   //      while(true) {
   //         ++aiDimension[iDimension];
   //         if(aiDimension[iDimension] != 
   //               pFeatureCombinations->GetFeatureCombinationEntries()[aiDimensionPermutation[iDimension]].m_pFeature->m_cBins) {
   //            break;
   //         }
   //         aiDimension[iDimension] = 0;
   //         ++iDimension;
   //         if(iDimension == cDimensions) {
   //            goto move_next_permutation;
   //         }
   //      }
   //   }
   //   move_next_permutation:
   //} while(std::next_permutation(aiDimensionPermutation, &aiDimensionPermutation[cDimensions]));






   size_t aiStart[k_cDimensionsMax];

   if(2 == cDimensions) {
      FloatEbmType splittingScore;

      const size_t cBinsDimension1 = pFeatureCombination->GetFeatureCombinationEntries()[0].m_pFeature->GetCountBins();
      const size_t cBinsDimension2 = pFeatureCombination->GetFeatureCombinationEntries()[1].m_pFeature->GetCountBins();
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

      HistogramBucket<bClassification> * const pTotal = GetHistogramBucketByIndex<bClassification>(
         cBytesPerHistogramBucket, 
         aHistogramBuckets, 
         cTotalBucketsMainSpace - 1
      );
      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pTotal, aHistogramBucketsEndDebug);

      EBM_ASSERT(0 < cInstancesRequiredForChildSplitMin);

      FloatEbmType splittingScoreParent = FloatEbmType { 0 };
      EBM_ASSERT(0 < pTotal->m_cInstancesInBucket);
      FloatEbmType cInstancesInParentBucket = static_cast<FloatEbmType>(pTotal->m_cInstancesInBucket);
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         // TODO : we can make this faster by doing the division in ComputeNodeSplittingScoreParent after we add all the numerators 
         // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

         const FloatEbmType splittingScoreParentUpdate = EbmStatistics::ComputeNodeSplittingScore(
            ArrayToPointer(pTotal->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
            cInstancesInParentBucket
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
         const FloatEbmType splittingScoreNew1 = SweepMultiDiemensional<compilerLearningTypeOrCountTargetClasses, compilerCountDimensions>(
            aHistogramBuckets, 
            pFeatureCombination, 
            aiStart, 
            0x0, 
            1, 
            cInstancesRequiredForChildSplitMin, 
            runtimeLearningTypeOrCountTargetClasses, 
            pTotals2LowLowBest, 
            &cutSecond1LowBest
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
            , aHistogramBucketsEndDebug
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
               GetHistogramBucketByIndex<bClassification>(
                  cBytesPerHistogramBucket, 
                  pAuxiliaryBucketZone, 
                  8
               );
            HistogramBucket<bClassification> * pTotals2HighHighBest =
               GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 9);
            const FloatEbmType splittingScoreNew2 = SweepMultiDiemensional<compilerLearningTypeOrCountTargetClasses, compilerCountDimensions>(
               aHistogramBuckets, 
               pFeatureCombination, 
               aiStart, 
               0x1, 
               1, 
               cInstancesRequiredForChildSplitMin, 
               runtimeLearningTypeOrCountTargetClasses, 
               pTotals2HighLowBest, 
               &cutSecond1HighBest
#ifndef NDEBUG
               , aHistogramBucketsDebugCopy
               , aHistogramBucketsEndDebug
#endif // NDEBUG
               );
            // if we get a NaN result, we'd like to propagate it by making bestSplit NaN.  The rules for NaN values say that non equality comparisons are 
            // all false so, let's flip this comparison such that it should be true for NaN values.  If the compiler violates NaN comparions rules, 
            // no big deal.  NaN values will get us soon and shut down boosting.
            if(LIKELY(/* DO NOT CHANGE THIS WITHOUT READING THE ABOVE. WE DO THIS STRANGE COMPARISON FOR NaN values*/ 
               !(k_illegalGain == splittingScoreNew2))) 
            {
               EBM_ASSERT(std::isnan(splittingScoreNew2) || FloatEbmType { 0 } <= splittingScoreNew2);
               splittingScore += splittingScoreNew2;

               // if we get a NaN result, we'd like to propagate it by making bestSplit NaN.  The rules for NaN values say that non equality comparisons 
               // are all false so, let's flip this comparison such that it should be true for NaN values.  If the compiler violates NaN comparions rules, 
               // no big deal.  NaN values will get us soon and shut down boosting.
               if(UNLIKELY(/* DO NOT CHANGE THIS WITHOUT READING THE ABOVE. WE DO THIS STRANGE COMPARISON FOR NaN values*/ 
                  !(splittingScore <= bestSplittingScore))) 
               {
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
         const FloatEbmType splittingScoreNew1 = SweepMultiDiemensional<compilerLearningTypeOrCountTargetClasses, compilerCountDimensions>(
            aHistogramBuckets, 
            pFeatureCombination, 
            aiStart, 
            0x0, 
            0, 
            cInstancesRequiredForChildSplitMin, 
            runtimeLearningTypeOrCountTargetClasses, 
            pTotals1LowLowBestInner, 
            &cutSecond2LowBest
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
            , aHistogramBucketsEndDebug
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
            const FloatEbmType splittingScoreNew2 = SweepMultiDiemensional<compilerLearningTypeOrCountTargetClasses, compilerCountDimensions>(
               aHistogramBuckets, 
               pFeatureCombination, 
               aiStart, 
               0x2, 
               0, 
               cInstancesRequiredForChildSplitMin, 
               runtimeLearningTypeOrCountTargetClasses, 
               pTotals1HighLowBestInner, 
               &cutSecond2HighBest
#ifndef NDEBUG
               , aHistogramBucketsDebugCopy
               , aHistogramBucketsEndDebug
#endif // NDEBUG
               );
            // if we get a NaN result, we'd like to propagate it by making bestSplit NaN.  The rules for NaN values say that non equality comparisons are 
            // all false so, let's flip this comparison such that it should be true for NaN values.  If the compiler violates NaN comparions rules, 
            // no big deal.  NaN values will get us soon and shut down boosting.
            if(LIKELY(/* DO NOT CHANGE THIS WITHOUT READING THE ABOVE. WE DO THIS STRANGE COMPARISON FOR NaN values*/ 
               !(k_illegalGain == splittingScoreNew2))) 
            {
               EBM_ASSERT(std::isnan(splittingScoreNew2) || FloatEbmType { 0 } <= splittingScoreNew2);
               splittingScore += splittingScoreNew2;
               // if we get a NaN result, we'd like to propagate it by making bestSplit NaN.  The rules for NaN values say that non equality comparisons 
               // are all false so, let's flip this comparison such that it should be true for NaN values.  If the compiler violates NaN comparions rules, 
               // no big deal.  NaN values will get us soon and shut down boosting.
               if(UNLIKELY(/* DO NOT CHANGE THIS WITHOUT READING THE ABOVE. WE DO THIS STRANGE COMPARISON FOR NaN values*/ 
                  !(splittingScore <= bestSplittingScore))) 
               {
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
               prediction = EbmStatistics::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                  ArrayToPointer(pTotal->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                  ArrayToPointer(pTotal->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator()
               );
            } else {
               EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
               prediction = EbmStatistics::ComputeSmallChangeForOneSegmentRegression(
                  ArrayToPointer(pTotal->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                  static_cast<FloatEbmType>(pTotal->m_cInstancesInBucket)
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
   #ifndef NDEBUG
               free(aHistogramBucketsDebugCopy);
   #endif // NDEBUG
               return true;
            }
            pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = cutFirst2Best;

            if(cutFirst2LowBest < cutFirst2HighBest) {
               if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)) {
                  LOG_0(
                     TraceLevelWarning, 
                     "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)"
                  );
   #ifndef NDEBUG
                  free(aHistogramBucketsDebugCopy);
   #endif // NDEBUG
                  return true;
               }
               if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 2)) {
                  LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 2)");
   #ifndef NDEBUG
                  free(aHistogramBucketsDebugCopy);
   #endif // NDEBUG
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
   #ifndef NDEBUG
                  free(aHistogramBucketsDebugCopy);
   #endif // NDEBUG
                  return true;
               }
               if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 2)) {
                  LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 2)");
   #ifndef NDEBUG
                  free(aHistogramBucketsDebugCopy);
   #endif // NDEBUG
                  return true;
               }
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = cutFirst2HighBest;
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[1] = cutFirst2LowBest;
            } else {
               if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1)) {
                  LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1)");
   #ifndef NDEBUG
                  free(aHistogramBucketsDebugCopy);
   #endif // NDEBUG
                  return true;
               }

               if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4)) {
                  LOG_0(
                     TraceLevelWarning, 
                     "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4)"
                  );
   #ifndef NDEBUG
                  free(aHistogramBucketsDebugCopy);
   #endif // NDEBUG
                  return true;
               }
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = cutFirst2LowBest;
            }

            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               FloatEbmType predictionLowLow;
               FloatEbmType predictionLowHigh;
               FloatEbmType predictionHighLow;
               FloatEbmType predictionHighHigh;

               if(bClassification) {
                  predictionLowLow = EbmStatistics::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     ArrayToPointer(pTotals2LowLowBest->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                     pTotals2LowLowBest->m_aHistogramBucketVectorEntry[iVector].GetSumDenominator()
                  );
                  predictionLowHigh = EbmStatistics::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     ArrayToPointer(pTotals2LowHighBest->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                     pTotals2LowHighBest->m_aHistogramBucketVectorEntry[iVector].GetSumDenominator()
                  );
                  predictionHighLow = EbmStatistics::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     ArrayToPointer(pTotals2HighLowBest->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                     pTotals2HighLowBest->m_aHistogramBucketVectorEntry[iVector].GetSumDenominator()
                  );
                  predictionHighHigh = EbmStatistics::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     ArrayToPointer(pTotals2HighHighBest->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                     pTotals2HighHighBest->m_aHistogramBucketVectorEntry[iVector].GetSumDenominator()
                  );
               } else {
                  EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
                  predictionLowLow = EbmStatistics::ComputeSmallChangeForOneSegmentRegression(
                     ArrayToPointer(pTotals2LowLowBest->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                     static_cast<FloatEbmType>(pTotals2LowLowBest->m_cInstancesInBucket)
                  );
                  predictionLowHigh = EbmStatistics::ComputeSmallChangeForOneSegmentRegression(
                     ArrayToPointer(pTotals2LowHighBest->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                     static_cast<FloatEbmType>(pTotals2LowHighBest->m_cInstancesInBucket)
                  );
                  predictionHighLow = EbmStatistics::ComputeSmallChangeForOneSegmentRegression(
                     ArrayToPointer(pTotals2HighLowBest->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                     static_cast<FloatEbmType>(pTotals2HighLowBest->m_cInstancesInBucket)
                  );
                  predictionHighHigh = EbmStatistics::ComputeSmallChangeForOneSegmentRegression(
                     ArrayToPointer(pTotals2HighHighBest->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                     static_cast<FloatEbmType>(pTotals2HighHighBest->m_cInstancesInBucket)
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
#ifndef NDEBUG
               free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
               return true;
            }
            pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = cutFirst1Best;

            if(cutFirst1LowBest < cutFirst1HighBest) {
               if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)) {
                  LOG_0(
                     TraceLevelWarning, 
                     "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)"
                  );
#ifndef NDEBUG
                  free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
                  return true;
               }

               if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 2)) {
                  LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 2)");
#ifndef NDEBUG
                  free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
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
#ifndef NDEBUG
                  free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
                  return true;
               }

               if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 2)) {
                  LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 2)");
#ifndef NDEBUG
                  free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
                  return true;
               }
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = cutFirst1HighBest;
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[1] = cutFirst1LowBest;
            } else {
               if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 1)) {
                  LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 1)");
#ifndef NDEBUG
                  free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
                  return true;
               }
               if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4)) {
                  LOG_0(
                     TraceLevelWarning, 
                     "WARNING BoostMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4)"
                  );
#ifndef NDEBUG
                  free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
                  return true;
               }
               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = cutFirst1LowBest;
            }

            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               FloatEbmType predictionLowLow;
               FloatEbmType predictionLowHigh;
               FloatEbmType predictionHighLow;
               FloatEbmType predictionHighHigh;

               if(bClassification) {
                  predictionLowLow = EbmStatistics::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     ArrayToPointer(pTotals1LowLowBest->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                     ArrayToPointer(pTotals1LowLowBest->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator()
                  );
                  predictionLowHigh = EbmStatistics::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     ArrayToPointer(pTotals1LowHighBest->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                     ArrayToPointer(pTotals1LowHighBest->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator()
                  );
                  predictionHighLow = EbmStatistics::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     ArrayToPointer(pTotals1HighLowBest->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                     ArrayToPointer(pTotals1HighLowBest->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator()
                  );
                  predictionHighHigh = EbmStatistics::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     ArrayToPointer(pTotals1HighHighBest->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                     ArrayToPointer(pTotals1HighHighBest->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator()
                  );
               } else {
                  EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
                  predictionLowLow = EbmStatistics::ComputeSmallChangeForOneSegmentRegression(
                     ArrayToPointer(pTotals1LowLowBest->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                     static_cast<FloatEbmType>(pTotals1LowLowBest->m_cInstancesInBucket)
                  );
                  predictionLowHigh = EbmStatistics::ComputeSmallChangeForOneSegmentRegression(
                     ArrayToPointer(pTotals1LowHighBest->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                     static_cast<FloatEbmType>(pTotals1LowHighBest->m_cInstancesInBucket)
                  );
                  predictionHighLow = EbmStatistics::ComputeSmallChangeForOneSegmentRegression(
                     ArrayToPointer(pTotals1HighLowBest->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                     static_cast<FloatEbmType>(pTotals1HighLowBest->m_cInstancesInBucket)
                  );
                  predictionHighHigh = EbmStatistics::ComputeSmallChangeForOneSegmentRegression(
                     ArrayToPointer(pTotals1HighHighBest->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                     static_cast<FloatEbmType>(pTotals1HighHighBest->m_cInstancesInBucket)
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

      // gain can be -infinity for regression in a super-super-super-rare condition.  
      // See notes above regarding "gain = bestSplittingScore - splittingScoreParent"

      // within a set, no split should make our model worse.  It might in our validation set, but not within the training set
      EBM_ASSERT(std::isnan(gain) || (!bClassification) && std::isinf(gain) ||
         k_epsilonNegativeGainAllowed <= gain); 

      // TODO: this gain value is untested.  We should build a new test that compares the single feature gains to the multi-dimensional gains by
      // making a pair where one of the dimensions duplicates values in the 0 and 1 bin.  Then the gain should be identical, if there is only 1 split allowed
      *pTotalGain = gain;
   } else {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional 2 != dimensions");
  
      // TODO: handle this better
#ifndef NDEBUG
      EBM_ASSERT(false);
      free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
      return true;
   }

#ifndef NDEBUG
   free(aHistogramBucketsDebugCopy);
#endif // NDEBUG

   LOG_0(TraceLevelVerbose, "Exited BoostMultiDimensional");
   return false;
}
WARNING_POP

//template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerCountDimensions>
//bool BoostMultiDimensionalPaulAlgorithm(CachedThreadResources<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pCachedThreadResources, const FeatureInternal * const pTargetFeature, SamplingSet const * const pTrainingSet, const FeatureCombination * const pFeatureCombination, SegmentedRegion<ActiveDataType, FloatEbmType> * const pSmallChangeToModelOverwriteSingleSamplingSet) {
//   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets = BinDataSet<compilerLearningTypeOrCountTargetClasses>(pCachedThreadResources, pFeatureCombination, pTrainingSet, pTargetFeature);
//   if(UNLIKELY(nullptr == aHistogramBuckets)) {
//      return true;
//   }
//
//   BuildFastTotals(pTargetFeature, pFeatureCombination, aHistogramBuckets);
//
//   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(compilerCountDimensions, pFeatureCombination->GetCountFeatures());
//   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses);
//   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
//   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);
//
//   size_t aiStart[k_cDimensionsMax];
//   size_t aiLast[k_cDimensionsMax];
//
//   if(2 == cDimensions) {
//      DO: somehow avoid having a malloc here, either by allocating these when we allocate our big chunck of memory, or as part of pCachedThreadResources
//      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * aDynamicHistogramBuckets = static_cast<HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> *>(malloc(cBytesPerHistogramBucket * ));
//
//      const size_t cBinsDimension1 = pFeatureCombination->GetFeatureCombinationEntries()[0].m_pFeature->m_cBins;
//      const size_t cBinsDimension2 = pFeatureCombination->GetFeatureCombinationEntries()[1].m_pFeature->m_cBins;
//
//      FloatEbmType bestSplittingScore = FloatEbmType { -std::numeric_limits<FloatEbmType>::infinity() };
//
//      if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1)) {
//         free(aDynamicHistogramBuckets);
//#ifndef NDEBUG
//         free(aHistogramBucketsDebugCopy);
//#endif // NDEBUG
//         return true;
//      }
//      if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 1)) {
//         free(aDynamicHistogramBuckets);
//#ifndef NDEBUG
//         free(aHistogramBucketsDebugCopy);
//#endif // NDEBUG
//         return true;
//      }
//      if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4)) {
//         free(aDynamicHistogramBuckets);
//#ifndef NDEBUG
//         free(aHistogramBucketsDebugCopy);
//#endif // NDEBUG
//         return true;
//      }
//
//      for(size_t iBin1 = 0; iBin1 < cBinsDimension1 - 1; ++iBin1) {
//         for(size_t iBin2 = 0; iBin2 < cBinsDimension2 - 1; ++iBin2) {
//            FloatEbmType splittingScore;
//
//            HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTotalsLowLow = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aDynamicHistogramBuckets, 0);
//            HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTotalsHighLow = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aDynamicHistogramBuckets, 1);
//            HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTotalsLowHigh = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aDynamicHistogramBuckets, 2);
//            HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTotalsHighHigh = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aDynamicHistogramBuckets, 3);
//
//            HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTotalsTarget = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aDynamicHistogramBuckets, 4);
//            HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTotalsOther = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aDynamicHistogramBuckets, 5);
//
//            aiStart[0] = 0;
//            aiStart[1] = 0;
//            aiLast[0] = iBin1;
//            aiLast[1] = iBin2;
//            TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, compilerCountDimensions>(runtimeLearningTypeOrCountTargetClasses, pFeatureCombination, aHistogramBuckets, aiStart, aiLast, pTotalsLowLow);
//
//            aiStart[0] = iBin1 + 1;
//            aiStart[1] = 0;
//            aiLast[0] = cBinsDimension1 - 1;
//            aiLast[1] = iBin2;
//            TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, compilerCountDimensions>(runtimeLearningTypeOrCountTargetClasses, pFeatureCombination, aHistogramBuckets, aiStart, aiLast, pTotalsHighLow);
//
//            aiStart[0] = 0;
//            aiStart[1] = iBin2 + 1;
//            aiLast[0] = iBin1;
//            aiLast[1] = cBinsDimension2 - 1;
//            TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, compilerCountDimensions>(runtimeLearningTypeOrCountTargetClasses, pFeatureCombination, aHistogramBuckets, aiStart, aiLast, pTotalsLowHigh);
//
//            aiStart[0] = iBin1 + 1;
//            aiStart[1] = iBin2 + 1;
//            aiLast[0] = cBinsDimension1 - 1;
//            aiLast[1] = cBinsDimension2 - 1;
//            TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, compilerCountDimensions>(runtimeLearningTypeOrCountTargetClasses, pFeatureCombination, aHistogramBuckets, aiStart, aiLast, pTotalsHighHigh);
//
//            // LOW LOW
//            pTotalsTarget->Zero(runtimeLearningTypeOrCountTargetClasses);
//            pTotalsOther->Zero(runtimeLearningTypeOrCountTargetClasses);
//
//            // MODIFY HERE
//            pTotalsTarget->Add(*pTotalsLowLow, runtimeLearningTypeOrCountTargetClasses);
//            pTotalsOther->Add(*pTotalsHighLow, runtimeLearningTypeOrCountTargetClasses);
//            pTotalsOther->Add(*pTotalsLowHigh, runtimeLearningTypeOrCountTargetClasses);
//            pTotalsOther->Add(*pTotalsHighHigh, runtimeLearningTypeOrCountTargetClasses);
//            
//            splittingScore = CalculateRegionSplittingScore<compilerLearningTypeOrCountTargetClasses, compilerCountDimensions>(pTotalsTarget, pTotalsOther, runtimeLearningTypeOrCountTargetClasses);
//            if(bestSplittingScore < splittingScore) {
//               bestSplittingScore = splittingScore;
//
//               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = iBin1;
//               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = iBin2;
//
//               for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
//                  FloatEbmType predictionTarget;
//                  FloatEbmType predictionOther;
//
//                  if(IS_REGRESSION(compilerLearningTypeOrCountTargetClasses)) {
//                     // regression
//                     predictionTarget = ComputeSmallChangeForOneSegmentRegression(ArrayToPointer(pTotalsTarget->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, pTotalsTarget->m_cInstancesInBucket);
//                     predictionOther = ComputeSmallChangeForOneSegmentRegression(ArrayToPointer(pTotalsOther->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, pTotalsOther->m_cInstancesInBucket);
//                  } else {
//                     EBM_ASSERT(IS_CLASSIFICATION(compilerLearningTypeOrCountTargetClasses));
//                     // classification
//                     predictionTarget = ComputeSmallChangeForOneSegmentClassificationLogOdds(ArrayToPointer(pTotalsTarget->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, ArrayToPointer(pTotalsTarget->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator());
//                     predictionOther = ComputeSmallChangeForOneSegmentClassificationLogOdds(ArrayToPointer(pTotalsOther->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, ArrayToPointer(pTotalsOther->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator());
//                  }
//
//                  // MODIFY HERE
//                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[0 * cVectorLength + iVector] = predictionTarget;
//                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[1 * cVectorLength + iVector] = predictionOther;
//                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[2 * cVectorLength + iVector] = predictionOther;
//                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[3 * cVectorLength + iVector] = predictionOther;
//               }
//            }
//
//
//
//
//            // HIGH LOW
//            pTotalsTarget->Zero(runtimeLearningTypeOrCountTargetClasses);
//            pTotalsOther->Zero(runtimeLearningTypeOrCountTargetClasses);
//
//            // MODIFY HERE
//            pTotalsOther->Add(*pTotalsLowLow, runtimeLearningTypeOrCountTargetClasses);
//            pTotalsTarget->Add(*pTotalsHighLow, runtimeLearningTypeOrCountTargetClasses);
//            pTotalsOther->Add(*pTotalsLowHigh, runtimeLearningTypeOrCountTargetClasses);
//            pTotalsOther->Add(*pTotalsHighHigh, runtimeLearningTypeOrCountTargetClasses);
//
//            splittingScore = CalculateRegionSplittingScore<compilerLearningTypeOrCountTargetClasses, compilerCountDimensions>(pTotalsTarget, pTotalsOther, runtimeLearningTypeOrCountTargetClasses);
//            if(bestSplittingScore < splittingScore) {
//               bestSplittingScore = splittingScore;
//
//               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = iBin1;
//               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = iBin2;
//
//               for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
//                  FloatEbmType predictionTarget;
//                  FloatEbmType predictionOther;
//
//                  if(IS_REGRESSION(compilerLearningTypeOrCountTargetClasses)) {
//                     // regression
//                     predictionTarget = ComputeSmallChangeForOneSegmentRegression(ArrayToPointer(pTotalsTarget->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, pTotalsTarget->m_cInstancesInBucket);
//                     predictionOther = ComputeSmallChangeForOneSegmentRegression(ArrayToPointer(pTotalsOther->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, pTotalsOther->m_cInstancesInBucket);
//                  } else {
//                     EBM_ASSERT(IS_CLASSIFICATION(compilerLearningTypeOrCountTargetClasses));
//                     // classification
//                     predictionTarget = ComputeSmallChangeForOneSegmentClassificationLogOdds(ArrayToPointer(pTotalsTarget->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, ArrayToPointer(pTotalsTarget->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator());
//                     predictionOther = ComputeSmallChangeForOneSegmentClassificationLogOdds(ArrayToPointer(pTotalsOther->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, ArrayToPointer(pTotalsOther->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator());
//                  }
//
//                  // MODIFY HERE
//                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[0 * cVectorLength + iVector] = predictionOther;
//                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[1 * cVectorLength + iVector] = predictionTarget;
//                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[2 * cVectorLength + iVector] = predictionOther;
//                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[3 * cVectorLength + iVector] = predictionOther;
//               }
//            }
//
//
//
//
//            // LOW HIGH
//            pTotalsTarget->Zero(runtimeLearningTypeOrCountTargetClasses);
//            pTotalsOther->Zero(runtimeLearningTypeOrCountTargetClasses);
//
//            // MODIFY HERE
//            pTotalsOther->Add(*pTotalsLowLow, runtimeLearningTypeOrCountTargetClasses);
//            pTotalsOther->Add(*pTotalsHighLow, runtimeLearningTypeOrCountTargetClasses);
//            pTotalsTarget->Add(*pTotalsLowHigh, runtimeLearningTypeOrCountTargetClasses);
//            pTotalsOther->Add(*pTotalsHighHigh, runtimeLearningTypeOrCountTargetClasses);
//
//            splittingScore = CalculateRegionSplittingScore<compilerLearningTypeOrCountTargetClasses, compilerCountDimensions>(pTotalsTarget, pTotalsOther, runtimeLearningTypeOrCountTargetClasses);
//            if(bestSplittingScore < splittingScore) {
//               bestSplittingScore = splittingScore;
//
//               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = iBin1;
//               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = iBin2;
//
//               for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
//                  FloatEbmType predictionTarget;
//                  FloatEbmType predictionOther;
//
//                  if(IS_REGRESSION(compilerLearningTypeOrCountTargetClasses)) {
//                     // regression
//                     predictionTarget = ComputeSmallChangeForOneSegmentRegression(ArrayToPointer(pTotalsTarget->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, pTotalsTarget->m_cInstancesInBucket);
//                     predictionOther = ComputeSmallChangeForOneSegmentRegression(ArrayToPointer(pTotalsOther->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, pTotalsOther->m_cInstancesInBucket);
//                  } else {
//                     EBM_ASSERT(IS_CLASSIFICATION(compilerLearningTypeOrCountTargetClasses));
//                     // classification
//                     predictionTarget = ComputeSmallChangeForOneSegmentClassificationLogOdds(ArrayToPointer(pTotalsTarget->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, ArrayToPointer(pTotalsTarget->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator());
//                     predictionOther = ComputeSmallChangeForOneSegmentClassificationLogOdds(ArrayToPointer(pTotalsOther->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, ArrayToPointer(pTotalsOther->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator());
//                  }
//
//                  // MODIFY HERE
//                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[0 * cVectorLength + iVector] = predictionOther;
//                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[1 * cVectorLength + iVector] = predictionOther;
//                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[2 * cVectorLength + iVector] = predictionTarget;
//                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[3 * cVectorLength + iVector] = predictionOther;
//               }
//            }
//
//
//
//            // HIGH HIGH
//            pTotalsTarget->Zero(runtimeLearningTypeOrCountTargetClasses);
//            pTotalsOther->Zero(runtimeLearningTypeOrCountTargetClasses);
//
//            // MODIFY HERE
//            pTotalsOther->Add(*pTotalsLowLow, runtimeLearningTypeOrCountTargetClasses);
//            pTotalsOther->Add(*pTotalsHighLow, runtimeLearningTypeOrCountTargetClasses);
//            pTotalsOther->Add(*pTotalsLowHigh, runtimeLearningTypeOrCountTargetClasses);
//            pTotalsTarget->Add(*pTotalsHighHigh, runtimeLearningTypeOrCountTargetClasses);
//
//            splittingScore = CalculateRegionSplittingScore<compilerLearningTypeOrCountTargetClasses, compilerCountDimensions>(pTotalsTarget, pTotalsOther, runtimeLearningTypeOrCountTargetClasses);
//            if(bestSplittingScore < splittingScore) {
//               bestSplittingScore = splittingScore;
//
//               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = iBin1;
//               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = iBin2;
//
//               for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
//                  FloatEbmType predictionTarget;
//                  FloatEbmType predictionOther;
//
//                  if(IS_REGRESSION(compilerLearningTypeOrCountTargetClasses)) {
//                     // regression
//                     predictionTarget = ComputeSmallChangeForOneSegmentRegression(ArrayToPointer(pTotalsTarget->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, pTotalsTarget->m_cInstancesInBucket);
//                     predictionOther = ComputeSmallChangeForOneSegmentRegression(ArrayToPointer(pTotalsOther->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, pTotalsOther->m_cInstancesInBucket);
//                  } else {
//                     EBM_ASSERT(IS_CLASSIFICATION(compilerLearningTypeOrCountTargetClasses));
//                     // classification
//                     predictionTarget = ComputeSmallChangeForOneSegmentClassificationLogOdds(ArrayToPointer(pTotalsTarget->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, ArrayToPointer(pTotalsTarget->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator());
//                     predictionOther = ComputeSmallChangeForOneSegmentClassificationLogOdds(ArrayToPointer(pTotalsOther->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, ArrayToPointer(pTotalsOther->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator());
//                  }
//
//                  // MODIFY HERE
//                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[0 * cVectorLength + iVector] = predictionOther;
//                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[1 * cVectorLength + iVector] = predictionOther;
//                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[2 * cVectorLength + iVector] = predictionOther;
//                  pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()[3 * cVectorLength + iVector] = predictionTarget;
//               }
//            }
//
//
//
//
//
//
//         }
//      }
//
//      free(aDynamicHistogramBuckets);
//   } else {
//      DO: handle this better
//#ifndef NDEBUG
//      EBM_ASSERT(false); // we only support pairs currently
//      free(aHistogramBucketsDebugCopy);
//#endif // NDEBUG
//      return true;
//   }
//#ifndef NDEBUG
//   free(aHistogramBucketsDebugCopy);
//#endif // NDEBUG
//   return false;
//}




EBM_INLINE static bool CalculateInteractionScore(
   CachedInteractionThreadResources * const pCachedThreadResources,
   EbmInteractionState * const pEbmInteractionState,
   const FeatureCombination * const pFeatureCombination,
   const size_t cInstancesRequiredForChildSplitMin,
   FloatEbmType * const pInteractionScoreReturn
) {
   // TODO : we NEVER use the denominator term in HistogramBucketVectorEntry when calculating interaction scores, but we're spending time calculating 
   // it, and it's taking up precious memory.  We should eliminate the denominator term HERE in our datastructures OR we should think whether we can 
   // use the denominator as part of the gain function!!!

   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pEbmInteractionState->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelVerbose, "Entered CalculateInteractionScore");

   const size_t cDimensions = pFeatureCombination->GetCountFeatures();
   EBM_ASSERT(1 <= cDimensions); // situations with 0 dimensions should have been filtered out before this function was called (but still inside the C++)

   size_t cAuxillaryBucketsForBuildFastTotals = 0;
   size_t cTotalBucketsMainSpace = 1;
   for(size_t iDimension = 0; iDimension < cDimensions; ++iDimension) {
      const size_t cBins = pFeatureCombination->GetFeatureCombinationEntries()[iDimension].m_pFeature->GetCountBins();
      EBM_ASSERT(2 <= cBins); // situations with 1 bin should have been filtered out before this function was called (but still inside the C++)
      // if cBins could be 1, then we'd need to check at runtime for overflow of cAuxillaryBucketsForBuildFastTotals
      // if this wasn't true then we'd have to check IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace) at runtime
      EBM_ASSERT(cAuxillaryBucketsForBuildFastTotals < cTotalBucketsMainSpace);
      // since cBins must be 2 or more, cAuxillaryBucketsForBuildFastTotals must grow slower than cTotalBucketsMainSpace, and we checked at allocation 
      // that cTotalBucketsMainSpace would not overflow
      EBM_ASSERT(!IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace));
      // this can overflow, but if it does then we're guaranteed to catch the overflow via the multiplication check below
      cAuxillaryBucketsForBuildFastTotals += cTotalBucketsMainSpace;
      if(IsMultiplyError(cTotalBucketsMainSpace, cBins)) {
         // unlike in the boosting code where we check at allocation time if the tensor created overflows on multiplication
         // we don't know what combination of features our caller will give us for calculating the interaction scores,
         // so we need to check if our caller gave us a tensor that overflows multiplication
         LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScore IsMultiplyError(cTotalBucketsMainSpace, cBins)");
         return true;
      }
      cTotalBucketsMainSpace *= cBins;
      // if this wasn't true then we'd have to check IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace) at runtime
      EBM_ASSERT(cAuxillaryBucketsForBuildFastTotals < cTotalBucketsMainSpace);
   }

   const size_t cAuxillaryBucketsForSplitting = 4;
   const size_t cAuxillaryBuckets = 
      cAuxillaryBucketsForBuildFastTotals < cAuxillaryBucketsForSplitting ? cAuxillaryBucketsForSplitting : cAuxillaryBucketsForBuildFastTotals;
   if(IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScore IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)");
      return true;
   }
   const size_t cTotalBuckets = cTotalBucketsMainSpace + cAuxillaryBuckets;

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   if(GetHistogramBucketSizeOverflow(bClassification, cVectorLength)) {
      LOG_0(
         TraceLevelWarning, 
         "WARNING CalculateInteractionScore GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)"
      );
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);
   if(IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)) {
      LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScore IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)");
      return true;
   }
   const size_t cBytesBuffer = cTotalBuckets * cBytesPerHistogramBucket;

   // this doesn't need to be freed since it's tracked and re-used by the class CachedInteractionThreadResources
   HistogramBucketBase * const aHistogramBuckets = pCachedThreadResources->GetThreadByteBuffer1(cBytesBuffer);
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScore nullptr == aHistogramBuckets");
      return true;
   }

   if(bClassification) {
      HistogramBucket<true> * const aHistogramBucketsLocal = aHistogramBuckets->GetHistogramBucket<true>();
      for(size_t i = 0; i < cTotalBuckets; ++i) {
         HistogramBucket<true> * const pHistogramBucket =
            GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBucketsLocal, i);
         pHistogramBucket->Zero(cVectorLength);
      }
   } else {
      HistogramBucket<false> * const aHistogramBucketsLocal = aHistogramBuckets->GetHistogramBucket<false>();
      for(size_t i = 0; i < cTotalBuckets; ++i) {
         HistogramBucket<false> * const pHistogramBucket =
            GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBucketsLocal, i);
         pHistogramBucket->Zero(cVectorLength);
      }
   }

   HistogramBucketBase * pAuxiliaryBucketZone =
      GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBuckets, cTotalBucketsMainSpace);

#ifndef NDEBUG
   const unsigned char * const aHistogramBucketsEndDebug = reinterpret_cast<unsigned char *>(aHistogramBuckets) + cBytesBuffer;
#endif // NDEBUG
   
   BinInteraction(
      pEbmInteractionState,
      pFeatureCombination,
      aHistogramBuckets
#ifndef NDEBUG
      , aHistogramBucketsEndDebug
#endif // NDEBUG
   );

#ifndef NDEBUG
   // make a copy of the original binned buckets for debugging purposes
   size_t cTotalBucketsDebug = 1;
   for(size_t iDimensionDebug = 0; iDimensionDebug < cDimensions; ++iDimensionDebug) {
      const size_t cBins = pFeatureCombination->GetFeatureCombinationEntries()[iDimensionDebug].m_pFeature->GetCountBins();
      EBM_ASSERT(!IsMultiplyError(cTotalBucketsDebug, cBins)); // we checked this above
      cTotalBucketsDebug *= cBins;
   }
   // we wouldn't have been able to allocate our main buffer above if this wasn't ok
   EBM_ASSERT(!IsMultiplyError(cTotalBucketsDebug, cBytesPerHistogramBucket));
   HistogramBucketBase * const aHistogramBucketsDebugCopy =
      EbmMalloc<HistogramBucketBase>(cTotalBucketsDebug, cBytesPerHistogramBucket);
   if(nullptr != aHistogramBucketsDebugCopy) {
      // if we can't allocate, don't fail.. just stop checking
      const size_t cBytesBufferDebug = cTotalBucketsDebug * cBytesPerHistogramBucket;
      memcpy(aHistogramBucketsDebugCopy, aHistogramBuckets, cBytesBufferDebug);
   }
#endif // NDEBUG

   TensorTotalsBuild(
      runtimeLearningTypeOrCountTargetClasses,
      pFeatureCombination,
      pAuxiliaryBucketZone,
      aHistogramBuckets
#ifndef NDEBUG
      , aHistogramBucketsDebugCopy
      , aHistogramBucketsEndDebug
#endif // NDEBUG
   );

   if(2 == cDimensions) {
      LOG_0(TraceLevelVerbose, "CalculateInteractionScore Starting bin sweep loop");

      FloatEbmType bestSplittingScore = FindBestInteractionGainPairs(
         pEbmInteractionState,
         pFeatureCombination,
         cInstancesRequiredForChildSplitMin,
         pAuxiliaryBucketZone,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      
      LOG_0(TraceLevelVerbose, "CalculateInteractionScore Done bin sweep loop");

      if(nullptr != pInteractionScoreReturn) {
         // we started our score at zero, and didn't replace with anything lower, so it can't be below zero
         // if we collected a NaN value, then we kept it
         EBM_ASSERT(std::isnan(bestSplittingScore) || FloatEbmType { 0 } <= bestSplittingScore);
         EBM_ASSERT((!bClassification) || !std::isinf(bestSplittingScore));

         // if bestSplittingScore was NaN we make it zero so that it's not included.  If infinity, also don't include it since we overloaded something
         // even though bestSplittingScore shouldn't be +-infinity for classification, we check it for +-infinity 
         // here since it's most efficient to check that the exponential is all ones, which is the case only for +-infinity and NaN, but not others
         if(UNLIKELY(UNLIKELY(std::isnan(bestSplittingScore)) || UNLIKELY(std::isinf(bestSplittingScore)))) {
            bestSplittingScore = FloatEbmType { 0 };
         }
         *pInteractionScoreReturn = bestSplittingScore;
      }
   } else {
      EBM_ASSERT(false); // we only support pairs currently
      LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScore 2 != cDimensions");

      // TODO: handle this better
      if(nullptr != pInteractionScoreReturn) {
         // for now, just return any interactions that have other than 2 dimensions as zero, which means they won't be considered
         *pInteractionScoreReturn = FloatEbmType { 0 };
      }
   }

#ifndef NDEBUG
   free(aHistogramBucketsDebugCopy);
#endif // NDEBUG

   LOG_0(TraceLevelVerbose, "Exited CalculateInteractionScore");
   return false;
}

#endif // DIMENSION_MULTIPLE_H
