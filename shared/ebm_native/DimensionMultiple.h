// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef DIMENSION_MULTIPLE_H
#define DIMENSION_MULTIPLE_H

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "ebm_native.h" // EBM_INLINE
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

#include "Booster.h"

#include "TensorTotalsSum.h"

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

bool FindBestBoostingSplitPairs(
   EbmBoostingState * const pEbmBoostingState,
   const FeatureCombination * const pFeatureCombination,
   const size_t cInstancesRequiredForChildSplitMin,
   HistogramBucketBase * pAuxiliaryBucketZone,
   HistogramBucketBase * const pTotal,
   HistogramBucketBase * const aHistogramBuckets,
   SegmentedTensor * const pSmallChangeToModelOverwriteSingleSamplingSet,
   FloatEbmType * const pTotalGain
#ifndef NDEBUG
   , const HistogramBucketBase * const aHistogramBucketsDebugCopy
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
);

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
   if(GetHistogramBucketSizeOverflow(bClassification, cVectorLength)) {
      LOG_0(
         TraceLevelWarning, 
         "WARNING BoostMultiDimensional GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)"
      );
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);
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






   if(2 == cDimensions) {
      HistogramBucket<bClassification> * const pTotal = GetHistogramBucketByIndex<bClassification>(
         cBytesPerHistogramBucket,
         aHistogramBuckets,
         cTotalBucketsMainSpace - 1
         );

      bool bError = FindBestBoostingSplitPairs(
         pEbmBoostingState,
         pFeatureCombination,
         cInstancesRequiredForChildSplitMin,
         pAuxiliaryBucketZone,
         pTotal,
         aHistogramBuckets,
         pSmallChangeToModelOverwriteSingleSamplingSet,
         pTotalGain
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
      if(bError) {
#ifndef NDEBUG
         free(aHistogramBucketsDebugCopy);
#endif // NDEBUG

         LOG_0(TraceLevelVerbose, "Exited BoostMultiDimensional with Error code");

         return true;
      }

      // gain can be -infinity for regression in a super-super-super-rare condition.  
      // See notes above regarding "gain = bestSplittingScore - splittingScoreParent"

      // within a set, no split should make our model worse.  It might in our validation set, but not within the training set
      EBM_ASSERT(std::isnan(*pTotalGain) || (!bClassification) && std::isinf(*pTotalGain) ||
         k_epsilonNegativeGainAllowed <= *pTotalGain);
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




#endif // DIMENSION_MULTIPLE_H
