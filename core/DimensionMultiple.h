// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef MULTI_DIMENSIONAL_TRAINING_H
#define MULTI_DIMENSIONAL_TRAINING_H

#include <type_traits> // std::is_pod
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "SegmentedTensor.h"
#include "EbmStatistics.h"
#include "CachedThreadResources.h"
#include "FeatureCore.h"
#include "SamplingWithReplacement.h"
#include "HistogramBucket.h"

#ifndef NDEBUG

// TODO: remove the templating on these debug functions.  We don't need to replicate this function 63 times!!
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
void GetTotalsDebugSlow(const HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, const FeatureCombinationCore * const pFeatureCombination, const size_t * const aiStart, const size_t * const aiLast, const size_t cTargetClasses, HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pRet) {
   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->m_cFeatures);
   EBM_ASSERT(1 <= cDimensions); // why bother getting totals if we just have 1 bin
   size_t aiDimensions[k_cDimensionsMax];

   size_t iTensorBin = 0;
   size_t valueMultipleInitialize = 1;
   size_t iDimensionInitialize = 0;
   do {
      const size_t cBins = pFeatureCombination->m_FeatureCombinationEntry[iDimensionInitialize].m_pFeature->m_cBins;
      EBM_ASSERT(aiStart[iDimensionInitialize] < cBins);
      EBM_ASSERT(aiLast[iDimensionInitialize] < cBins);
      EBM_ASSERT(aiStart[iDimensionInitialize] <= aiLast[iDimensionInitialize]);
      EBM_ASSERT(!IsMultiplyError(aiStart[iDimensionInitialize], valueMultipleInitialize)); // aiStart[iDimensionInitialize] is less than cBins, so this should multiply
      iTensorBin += aiStart[iDimensionInitialize] * valueMultipleInitialize;
      EBM_ASSERT(!IsMultiplyError(cBins, valueMultipleInitialize)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
      valueMultipleInitialize *= cBins;
      aiDimensions[iDimensionInitialize] = aiStart[iDimensionInitialize];
      ++iDimensionInitialize;
   } while(iDimensionInitialize < cDimensions);

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we've allocated this, so it should fit
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);
   pRet->template Zero<compilerLearningTypeOrCountTargetClasses>(cTargetClasses);

   while(true) {
      const HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pHistogramBucket = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aHistogramBuckets, iTensorBin);

      pRet->template Add<compilerLearningTypeOrCountTargetClasses>(*pHistogramBucket, cTargetClasses);

      size_t iDimension = 0;
      size_t valueMultipleLoop = 1;
      while(aiDimensions[iDimension] == aiLast[iDimension]) {
         EBM_ASSERT(aiStart[iDimension] <= aiLast[iDimension]);
         EBM_ASSERT(!IsMultiplyError(aiLast[iDimension] - aiStart[iDimension], valueMultipleLoop)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
         iTensorBin -= (aiLast[iDimension] - aiStart[iDimension]) * valueMultipleLoop;

         const size_t cBins = pFeatureCombination->m_FeatureCombinationEntry[iDimension].m_pFeature->m_cBins;
         EBM_ASSERT(!IsMultiplyError(cBins, valueMultipleLoop)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
         valueMultipleLoop *= cBins;

         aiDimensions[iDimension] = aiStart[iDimension];
         ++iDimension;
         if(iDimension == cDimensions) {
            return;
         }
      }
      ++aiDimensions[iDimension];
      iTensorBin += valueMultipleLoop;
   }
}

// TODO: remove the templating on these debug functions.  We don't need to replicate this function 63 times!!
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
void CompareTotalsDebug(const HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, const FeatureCombinationCore * const pFeatureCombination, const size_t * const aiPoint, const size_t directionVector, const size_t cTargetClasses, const HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pComparison) {
   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);

   size_t aiStart[k_cDimensionsMax];
   size_t aiLast[k_cDimensionsMax];
   size_t directionVectorDestroy = directionVector;
   for(size_t iDimensionDebug = 0; iDimensionDebug < pFeatureCombination->m_cFeatures; ++iDimensionDebug) {
      const size_t cBins = pFeatureCombination->m_FeatureCombinationEntry[iDimensionDebug].m_pFeature->m_cBins;
      if(UNPREDICTABLE(0 != (1 & directionVectorDestroy))) {
         aiStart[iDimensionDebug] = aiPoint[iDimensionDebug] + 1;
         aiLast[iDimensionDebug] = cBins - 1;
      } else {
         aiStart[iDimensionDebug] = 0;
         aiLast[iDimensionDebug] = aiPoint[iDimensionDebug];
      }
      directionVectorDestroy >>= 1;
   }

   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pComparison2 = static_cast<HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> *>(malloc(cBytesPerHistogramBucket));
   if(nullptr != pComparison2) {
      // if we can't obtain the memory, then don't do the comparison and exit
      GetTotalsDebugSlow<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, aiLast, cTargetClasses, pComparison2);
      EBM_ASSERT(pComparison->cInstancesInBucket == pComparison2->cInstancesInBucket);
      free(pComparison2);
   }
}

#endif // NDEBUG


//struct CurrentIndexAndCountBins {
//   size_t iCur;
//   // copy cBins to our local stack since we'll be referring to them often and our stack is more compact in cache and less all over the place AND not shared between CPUs
//   size_t cBins;
//};
//
//template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
//void BuildFastTotals(HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, const size_t cTargetClasses, const FeatureCombination * const pFeatureCombination) {
//   // TODO: sort our N-dimensional combinations at program startup so that the longest dimension is first!  That way we can more efficiently walk through contiguous memory better in this function!
//
//   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->m_cFeatures);
//   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
//   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsRegression(compilerLearningTypeOrCountTargetClasses)>(GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses));
//
//#ifndef NDEBUG
//   // make a copy of the original binned buckets for debugging purposes
//   size_t cTotalBucketsDebug = 1;
//   for(size_t iDimensionDebug = 0; iDimensionDebug < pFeatureCombination->m_cFeatures; ++iDimensionDebug) {
//      const size_t cBins = pFeatureCombination->m_FeatureCombinationEntry[iDimensionDebug].m_pFeature->m_cBins;
//      EBM_ASSERT(IsMultiplyError(cTotalBucketsDebug, cBins)); // we're accessing allocated memory, so this should work
//      cTotalBucketsDebug *= cBins;
//   }
//   EBM_ASSERT(IsMultiplyError(cTotalBucketsDebug, cBytesPerHistogramBucket)); // we're accessing allocated memory, so this should work
//   const size_t cBytesBufferDebug = cTotalBucketsDebug * cBytesPerHistogramBucket;
//   TODO : technically, adding cBytesPerHistogramBucket could overflow so we should handle that instead of asserting
//   EBM_ASSERT(IsAddError(cBytesBufferDebug, cBytesPerHistogramBucket)); // we're just allocating one extra bucket.  If we can't add these two numbers then we shouldn't have been able to allocate the array that we're copying from
//   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy = static_cast<HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> *>(malloc(cBytesBufferDebug + cBytesPerHistogramBucket));
//   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pDebugBucket = nullptr;
//   if(nullptr != aHistogramBucketsDebugCopy) {
//      // if we can't obtain the memory, then don't do the comparison and exit
//      memcpy(aHistogramBucketsDebugCopy, aHistogramBuckets, cBytesBufferDebug);
//      pDebugBucket = GetHistogramBucketByIndex<IsRegression(IsRegression(compilerLearningTypeOrCountTargetClasses))>(cBytesPerHistogramBucket, aHistogramBucketsDebugCopy, cTotalBucketsDebug);
//   }
//#endif // NDEBUG
//
//   EBM_ASSERT(0 < cDimensions);
//
//   CurrentIndexAndCountBins currentIndexAndCountBins[k_cDimensionsMax];
//   const CurrentIndexAndCountBins * const pCurrentIndexAndCountBinsEnd = &currentIndexAndCountBins[cDimensions];
//   const FeatureCombination::FeatureCombinationEntry * pFeatureCombinationEntry = &pFeatureCombination->m_FeatureCombinationEntry[0];
//   for(CurrentIndexAndCountBins * pCurrentIndexAndCountBinsInitialize = currentIndexAndCountBins; pCurrentIndexAndCountBinsEnd != pCurrentIndexAndCountBinsInitialize; ++pCurrentIndexAndCountBinsInitialize, ++pFeatureCombinationEntry) {
//      pCurrentIndexAndCountBinsInitialize->iCur = 0;
//      EBM_ASSERT(2 <= pFeatureCombinationEntry->m_pFeature->m_cBins);
//      pCurrentIndexAndCountBinsInitialize->cBins = pFeatureCombinationEntry->m_pFeature->m_cBins;
//   }
//
//   static_assert(k_cDimensionsMax < k_cBitsForSizeT, "reserve the highest bit for bit manipulation space");
//   EBM_ASSERT(cDimensions < k_cBitsForSizeT);
//   const size_t permuteVectorEnd = size_t { 1 } << cDimensions;
//   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pHistogramBucket = aHistogramBuckets;
//
//   goto skip_intro;
//
//   CurrentIndexAndCountBins * pCurrentIndexAndCountBins;
//   size_t iBucket;
//   while(true) {
//      pCurrentIndexAndCountBins->iCur = iBucket;
//      // we're walking through all buckets, so just move to the next one in the flat array, with the knoledge that we'll figure out it's multi-dimenional index below
//      pHistogramBucket = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucket, 1);
//
//   skip_intro:
//
//      // TODO : I think this code below can be made more efficient by storing the sum of all the items in the 0th dimension where we don't subtract the 0th dimension then when we go to sum up the next set we can eliminate half the work!
//
//      size_t permuteVector = 1;
//      do {
//         HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTargetHistogramBucket = pHistogramBucket;
//         bool bPositive = false;
//         size_t permuteVectorDestroy = permuteVector;
//         ptrdiff_t multiplyDimension = -1;
//         pCurrentIndexAndCountBins = &currentIndexAndCountBins[0];
//         do {
//            if(0 != (1 & permuteVectorDestroy)) {
//               if(0 == pCurrentIndexAndCountBins->iCur) {
//                  goto skip_combination;
//               }
//               pTargetHistogramBucket = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pTargetHistogramBucket, multiplyDimension);
//               bPositive = !bPositive;
//            }
//            // TODO: can we eliminate the multiplication by storing the multiples instead of the cBins?
//            multiplyDimension *= pCurrentIndexAndCountBins->cBins;
//            ++pCurrentIndexAndCountBins;
//            permuteVectorDestroy >>= 1;
//         } while(0 != permuteVectorDestroy);
//         if(bPositive) {
//            pHistogramBucket->Add(*pTargetHistogramBucket, cTargetClasses);
//         } else {
//            pHistogramBucket->Subtract(*pTargetHistogramBucket, cTargetClasses);
//         }
//      skip_combination:
//         ++permuteVector;
//      } while(permuteVectorEnd != permuteVector);
//
//#ifndef NDEBUG
//      if(nullptr != aHistogramBucketsDebugCopy) {
//         EBM_ASSERT(nullptr != pDebugBucket);
//         size_t aiStart[k_cDimensionsMax];
//         size_t aiLast[k_cDimensionsMax];
//         for(size_t iDebugDimension = 0; iDebugDimension < cDimensions; ++iDebugDimension) {
//            aiStart[iDebugDimension] = 0;
//            aiLast[iDebugDimension] = currentIndexAndCountBins[iDebugDimension].iCur;
//         }
//         GetTotalsDebugSlow<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBucketsDebugCopy, pFeatureCombination, aiStart, aiLast, cTargetClasses, pDebugBucket);
//         EBM_ASSERT(pDebugBucket->cInstancesInBucket == pHistogramBucket->cInstancesInBucket);
//
//         free(aHistogramBucketsDebugCopy);
//      }
//#endif // NDEBUG
//
//      pCurrentIndexAndCountBins = &currentIndexAndCountBins[0];
//      while(true) {
//         iBucket = pCurrentIndexAndCountBins->iCur + 1;
//         EBM_ASSERT(iBucket <= pCurrentIndexAndCountBins->cBins);
//         if(iBucket != pCurrentIndexAndCountBins->cBins) {
//            break;
//         }
//         pCurrentIndexAndCountBins->iCur = 0;
//         ++pCurrentIndexAndCountBins;
//         if(pCurrentIndexAndCountBinsEnd == pCurrentIndexAndCountBins) {
//            return;
//         }
//      }
//   }
//}
//





//struct CurrentIndexAndCountBins {
//   ptrdiff_t multipliedIndexCur;
//   ptrdiff_t multipleTotal;
//};
//
//template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
//void BuildFastTotals(HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, const size_t cTargetClasses, const FeatureCombination * const pFeatureCombination) {
//   // TODO: sort our N-dimensional combinations at program startup so that the longest dimension is first!  That way we can more efficiently walk through contiguous memory better in this function!
//
//   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->m_cFeatures);
//   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
//   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsRegression(compilerLearningTypeOrCountTargetClasses)>(GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses));
//
//#ifndef NDEBUG
//   // make a copy of the original binned buckets for debugging purposes
//   size_t cTotalBucketsDebug = 1;
//   for(size_t iDimensionDebug = 0; iDimensionDebug < pFeatureCombination->m_cFeatures; ++iDimensionDebug) {
//      const size_t cBins = pFeatureCombination->m_FeatureCombinationEntry[iDimensionDebug].m_pFeature->m_cBins;
//      EBM_ASSERT(IsMultiplyError(cTotalBucketsDebug, cBins)); // we're accessing allocated memory, so this should work
//      cTotalBucketsDebug *= cBins;
//   }
//   EBM_ASSERT(IsMultiplyError(cTotalBucketsDebug, cBytesPerHistogramBucket)); // we're accessing allocated memory, so this should work
//   const size_t cBytesBufferDebug = cTotalBucketsDebug * cBytesPerHistogramBucket;
//   TODO : technically, adding cBytesPerHistogramBucket could overflow so we should handle that instead of asserting
//   EBM_ASSERT(IsAddError(cBytesBufferDebug, cBytesPerHistogramBucket)); // we're just allocating one extra bucket.  If we can't add these two numbers then we shouldn't have been able to allocate the array that we're copying from
//   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy = static_cast<HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> *>(malloc(cBytesBufferDebug + cBytesPerHistogramBucket));
//   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pDebugBucket = nullptr;
//   if(nullptr != aHistogramBucketsDebugCopy) {
//      // if we can't obtain the memory, then don't do the comparison and exit
//      memcpy(aHistogramBucketsDebugCopy, aHistogramBuckets, cBytesBufferDebug);
//      pDebugBucket = GetHistogramBucketByIndex<IsRegression(IsRegression(compilerLearningTypeOrCountTargetClasses))>(cBytesPerHistogramBucket, aHistogramBucketsDebugCopy, cTotalBucketsDebug);
//   }
//#endif // NDEBUG
//
//   EBM_ASSERT(0 < cDimensions);
//
//   CurrentIndexAndCountBins currentIndexAndCountBins[k_cDimensionsMax];
//   const CurrentIndexAndCountBins * const pCurrentIndexAndCountBinsEnd = &currentIndexAndCountBins[cDimensions];
//   const FeatureCombination::FeatureCombinationEntry * pFeatureCombinationEntry = &pFeatureCombination->m_FeatureCombinationEntry[0];
//   ptrdiff_t multipleTotalInitialize = -1;
//   for(CurrentIndexAndCountBins * pCurrentIndexAndCountBinsInitialize = currentIndexAndCountBins; pCurrentIndexAndCountBinsEnd != pCurrentIndexAndCountBinsInitialize; ++pCurrentIndexAndCountBinsInitialize, ++pFeatureCombinationEntry) {
//      pCurrentIndexAndCountBinsInitialize->multipliedIndexCur = 0;
//      EBM_ASSERT(2 <= pFeatureCombinationEntry->m_pFeature->m_cBins);
//      multipleTotalInitialize *= static_cast<ptrdiff_t>(pFeatureCombinationEntry->m_pFeature->m_cBins);
//      pCurrentIndexAndCountBinsInitialize->multipleTotal = multipleTotalInitialize;
//   }
//
//   static_assert(k_cDimensionsMax < k_cBitsForSizeT, "reserve the highest bit for bit manipulation space");
//   EBM_ASSERT(cDimensions < k_cBitsForSizeT);
//   const size_t permuteVectorEnd = size_t { 1 } << cDimensions;
//   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pHistogramBucket = aHistogramBuckets;
//
//   goto skip_intro;
//
//   CurrentIndexAndCountBins * pCurrentIndexAndCountBins;
//   ptrdiff_t multipliedIndexCur;
//   while(true) {
//      pCurrentIndexAndCountBins->multipliedIndexCur = multipliedIndexCur;
//      // we're walking through all buckets, so just move to the next one in the flat array, with the knoledge that we'll figure out it's multi-dimenional index below
//      pHistogramBucket = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucket, 1);
//
//   skip_intro:
//
//      // TODO : I think this code below can be made more efficient by storing the sum of all the items in the 0th dimension where we don't subtract the 0th dimension then when we go to sum up the next set we can eliminate half the work!
//
//      size_t permuteVector = 1;
//      do {
//         HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTargetHistogramBucket = pHistogramBucket;
//         bool bPositive = false;
//         size_t permuteVectorDestroy = permuteVector;
//         ptrdiff_t multipleTotal = -1;
//         pCurrentIndexAndCountBins = &currentIndexAndCountBins[0];
//         do {
//            if(0 != (1 & permuteVectorDestroy)) {
//               // even though our index is multiplied by the total bins until this point, we only care about the zero bin, and zero multiplied by anything is zero
//               if(0 == pCurrentIndexAndCountBins->multipliedIndexCur) {
//                  goto skip_combination;
//               }
//               pTargetHistogramBucket = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pTargetHistogramBucket, multipleTotal);
//               bPositive = !bPositive;
//            }
//            multipleTotal = pCurrentIndexAndCountBins->multipleTotal;
//            ++pCurrentIndexAndCountBins;
//            permuteVectorDestroy >>= 1;
//         } while(0 != permuteVectorDestroy);
//         if(bPositive) {
//            pHistogramBucket->Add(*pTargetHistogramBucket, cTargetClasses);
//         } else {
//            pHistogramBucket->Subtract(*pTargetHistogramBucket, cTargetClasses);
//         }
//      skip_combination:
//         ++permuteVector;
//      } while(permuteVectorEnd != permuteVector);
//
//#ifndef NDEBUG
//      if(nullptr != aHistogramBucketsDebugCopy) {
//         EBM_ASSERT(nullptr != pDebugBucket);
//         size_t aiStart[k_cDimensionsMax];
//         size_t aiLast[k_cDimensionsMax];
//         ptrdiff_t multipleTotalDebug = -1;
//         for(size_t iDebugDimension = 0; iDebugDimension < cDimensions; ++iDebugDimension) {
//            aiStart[iDebugDimension] = 0;
//            aiLast[iDebugDimension] = static_cast<size_t>(currentIndexAndCountBins[iDebugDimension].multipliedIndexCur / multipleTotalDebug);
//            multipleTotalDebug = currentIndexAndCountBins[iDebugDimension].multipleTotal;
//         }
//         GetTotalsDebugSlow<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBucketsDebugCopy, pFeatureCombination, aiStart, aiLast, cTargetClasses, pDebugBucket);
//         EBM_ASSERT(pDebugBucket->cInstancesInBucket == pHistogramBucket->cInstancesInBucket);
//         free(aHistogramBucketsDebugCopy);
//      }
//#endif // NDEBUG
//
//      pCurrentIndexAndCountBins = &currentIndexAndCountBins[0];
//      ptrdiff_t multipleTotal = -1;
//      while(true) {
//         multipliedIndexCur = pCurrentIndexAndCountBins->multipliedIndexCur + multipleTotal;
//         multipleTotal = pCurrentIndexAndCountBins->multipleTotal;
//         if(multipliedIndexCur != multipleTotal) {
//            break;
//         }
//         pCurrentIndexAndCountBins->multipliedIndexCur = 0;
//         ++pCurrentIndexAndCountBins;
//         if(pCurrentIndexAndCountBinsEnd == pCurrentIndexAndCountBins) {
//            return;
//         }
//      }
//   }
//}
//








template<bool bRegression>
struct FastTotalState {
   HistogramBucket<bRegression> * pDimensionalCur;
   HistogramBucket<bRegression> * pDimensionalWrap;
   HistogramBucket<bRegression> * pDimensionalFirst;
   size_t iCur;
   size_t cBins;
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
void BuildFastTotals(HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, const size_t cTargetClasses, const FeatureCombinationCore * const pFeatureCombination, HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pBucketAuxiliaryBuildZone
#ifndef NDEBUG
   , const HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy, const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   LOG(TraceLevelVerbose, "Entered BuildFastTotals");

   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->m_cFeatures);
   EBM_ASSERT(1 <= cDimensions);

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);

   FastTotalState<IsRegression(compilerLearningTypeOrCountTargetClasses)> fastTotalState[k_cDimensionsMax];
   const FastTotalState<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pFastTotalStateEnd = &fastTotalState[cDimensions];
   {
      FastTotalState<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pFastTotalStateInitialize = fastTotalState;
      const FeatureCombinationCore::FeatureCombinationEntry * pFeatureCombinationEntry = &pFeatureCombination->m_FeatureCombinationEntry[0];
      size_t multiply = 1;
      EBM_ASSERT(0 < cDimensions);
      do {
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pBucketAuxiliaryBuildZone, aHistogramBucketsEndDebug);

         size_t cBins = pFeatureCombinationEntry->m_pFeature->m_cBins;
         EBM_ASSERT(1 <= cBins); // this function can handle 1 == cBins even though that's a degenerate case that shouldn't be trained on (dimensions with 1 bin don't contribute anything since they always have the same value)

         pFastTotalStateInitialize->iCur = 0;
         pFastTotalStateInitialize->cBins = cBins;

         pFastTotalStateInitialize->pDimensionalFirst = pBucketAuxiliaryBuildZone;
         pFastTotalStateInitialize->pDimensionalCur = pBucketAuxiliaryBuildZone;
         pBucketAuxiliaryBuildZone = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pBucketAuxiliaryBuildZone, multiply);

#ifndef NDEBUG
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pBucketAuxiliaryBuildZone, -1), aHistogramBucketsEndDebug);
         for(HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pDimensionalCur = pFastTotalStateInitialize->pDimensionalCur; pBucketAuxiliaryBuildZone != pDimensionalCur; pDimensionalCur = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pDimensionalCur, 1)) {
            pDimensionalCur->template AssertZero<compilerLearningTypeOrCountTargetClasses>(cTargetClasses);
         }
#endif // NDEBUG

         // TODO : we don't need either the first or the wrap values since they are the next ones in the list.. we may need to populate one item past the end and make the list one larger
         pFastTotalStateInitialize->pDimensionalWrap = pBucketAuxiliaryBuildZone;

         multiply *= cBins;

         ++pFeatureCombinationEntry;
         ++pFastTotalStateInitialize;
      } while(LIKELY(pFastTotalStateEnd != pFastTotalStateInitialize));
   }

#ifndef NDEBUG
   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pDebugBucket = static_cast<HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> *>(malloc(cBytesPerHistogramBucket));
#endif //NDEBUG

   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pHistogramBucket = aHistogramBuckets;

   while(true) {
      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucket, aHistogramBucketsEndDebug);

      HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pAddPrev = pHistogramBucket;
      for(ptrdiff_t iDimension = cDimensions - 1; 0 <= iDimension ; --iDimension) {
         HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pAddTo = fastTotalState[iDimension].pDimensionalCur;
         pAddTo->template Add<compilerLearningTypeOrCountTargetClasses>(*pAddPrev, cTargetClasses);
         pAddPrev = pAddTo;
         pAddTo = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAddTo, 1);
         if(pAddTo == fastTotalState[iDimension].pDimensionalWrap) {
            pAddTo = fastTotalState[iDimension].pDimensionalFirst;
         }
         fastTotalState[iDimension].pDimensionalCur = pAddTo;
      }
      pHistogramBucket->template Copy<compilerLearningTypeOrCountTargetClasses>(*pAddPrev, cTargetClasses);

#ifndef NDEBUG
      if(nullptr != aHistogramBucketsDebugCopy && nullptr != pDebugBucket) {
         size_t aiStart[k_cDimensionsMax];
         size_t aiLast[k_cDimensionsMax];
         for(size_t iDebugDimension = 0; iDebugDimension < cDimensions; ++iDebugDimension) {
            aiStart[iDebugDimension] = 0;
            aiLast[iDebugDimension] = fastTotalState[iDebugDimension].iCur;
         }
         GetTotalsDebugSlow<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBucketsDebugCopy, pFeatureCombination, aiStart, aiLast, cTargetClasses, pDebugBucket);
         EBM_ASSERT(pDebugBucket->cInstancesInBucket == pHistogramBucket->cInstancesInBucket);
      }
#endif // NDEBUG

      // we're walking through all buckets, so just move to the next one in the flat array, with the knowledge that we'll figure out it's multi-dimenional index below
      pHistogramBucket = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucket, 1);

      FastTotalState<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pFastTotalState = &fastTotalState[0];
      while(true) {
         ++pFastTotalState->iCur;
         if(LIKELY(pFastTotalState->cBins != pFastTotalState->iCur)) {
            break;
         }
         pFastTotalState->iCur = 0;

         EBM_ASSERT(pFastTotalState->pDimensionalFirst == pFastTotalState->pDimensionalCur);
         memset(pFastTotalState->pDimensionalFirst, 0, reinterpret_cast<char *>(pFastTotalState->pDimensionalWrap) - reinterpret_cast<char *>(pFastTotalState->pDimensionalFirst));

         ++pFastTotalState;

         if(UNLIKELY(pFastTotalStateEnd == pFastTotalState)) {
#ifndef NDEBUG
            free(pDebugBucket);
#endif // NDEBUG

            LOG(TraceLevelVerbose, "Exited BuildFastTotals");
            return;
         }
      }
   }
}


struct CurrentIndexAndCountBins {
   ptrdiff_t multipliedIndexCur;
   ptrdiff_t multipleTotal;
};

// TODO : ALL OF THE BELOW!
//- D is the number of dimensions
//- N is the number of cases per dimension(assume all dimensions have the same number of cases for simplicity)
//- when we construct the initial D - dimensional totals, our current algorithm is N^D * 2 ^ (D - 1).We can probably reduce this to N^D * D with a lot of side memory that records the cost of going each direction
//- the above algorithm gives us small slices of work, so it probably can't help us make the next step of calculating the total regional space from a point to a corner any faster since the slices only give us 1 step away and we'd have to iterate through all the slices to get a larger region
//- we currently have one N^D memory region which allows us to calculate the total from any point to any corner in at worst 2 ^ D operations.If we had 2 ^ D memory spaces and were willing to construct them, then we could calculate the total from any point to any corner in 1 operation.If we made a second total region which had the totals from any point to the(1, 1, ..., 1, 1) corner, then we could calculate any point to corer in sqrt(2 ^ D), which is A LOT BETTER and it only takes twice as much memory.For an 8 dimensional space we would need 16 operations instead of 256!
//- to implement an algorithm that uses the(0, 0, ..., 0, 0) totals volume and the(1, 1, ..., 1, 1) volume, just see whether the input vector has more zeros or 1's and then choose the end point that is closest.
//- we can calculate the total from any arbitrary start and end point(instead of just a point to a corner) if we set the end point as the end and iterate through ALL permutations of all #'s of bits.  There doesn't seem to be any simplification that allows us to handle less than the full combinatoral exploration, even if we constructed a totals for each possible 2 ^ D corner
//- if we succeed(with extra memory) to turn the totals construction algorithm into a N^D*D algorithm, then we might be able to use that to calculate the totals dynamically at the same time that we sweep the splitting space for splits.The simplest sweep would be to look at each region from a point to each corner and choose the best split that isolates one of those corners instead of splitting at different poiints in each dimension.If we did the simplest possible thing, then our algorithm would be 2 ^ D*N^D*D OR(2 * N) ^ D*D.If we wanted the more complicated splits, then we might need to first build a totals so that we could determine the "tube totals" and then we could sweep the tube and have the costs on both sides of the split
//- IMEDIATE TASKS :
//- get point to corner working for N - dimensional to(0, 0, ..., 0, 0)
//- get splitting working for N - dimensional
//- have a look at our final dimensionality.Is the totals calculation the bottleneck, or the point to corner totals function ?
//- I think I understand the costs of all implementations of point to corner computation, so don't implement the (1,1,...,1,1) to point algorithm yet.. try implementing the more optimized totals calculation (with more memory).  After we have the optimized totals calculation, then try to re-do the splitting code to do splitting at the same time as totals calculation.  If that isn't better than our existing stuff, then optimzie the point to corner calculation code
//- implement a function that calcualtes the total of any volume using just the(0, 0, ..., 0, 0) totals ..as a debugging function.We might use this for trying out more complicated splits where we allow 2 splits on some axies

// TODO: build a pair and triple specific version of this function.  For pairs we can get ride of the pPrevious and just use the actual cell at (-1,-1) from our current cell, and we can use two loops with everything in memory [look at code above from before we incoporated the previous totals].  Triples would also benefit from pulling things out since we have low iterations of the inner loop and we can access indicies directly without additional add/subtract/bit operations.  Beyond triples, the combinatorial choices start to explode, so we should probably use this general N-dimensional code.
// TODO: after we build pair and triple specific versions of this function, we don't need to have a compiler countCompilerDimensions, since the compiler won't really be able to simpify the loops that are exploding in dimensionality
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
void BuildFastTotalsZeroMemoryIncrease(HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, const size_t cTargetClasses, const FeatureCombinationCore * const pFeatureCombination
#ifndef NDEBUG
   , const HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy, const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   LOG(TraceLevelVerbose, "Entered BuildFastTotalsZeroMemoryIncrease");

   // TODO: sort our N-dimensional combinations at program startup so that the longest dimension is first!  That way we can more efficiently walk through contiguous memory better in this function!

   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->m_cFeatures);
   EBM_ASSERT(1 <= cDimensions);

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);

   CurrentIndexAndCountBins currentIndexAndCountBins[k_cDimensionsMax];
   const CurrentIndexAndCountBins * const pCurrentIndexAndCountBinsEnd = &currentIndexAndCountBins[cDimensions];
   ptrdiff_t multipleTotalInitialize = -1;
   {
      CurrentIndexAndCountBins * pCurrentIndexAndCountBinsInitialize = currentIndexAndCountBins;
      const FeatureCombinationCore::FeatureCombinationEntry * pFeatureCombinationEntry = &pFeatureCombination->m_FeatureCombinationEntry[0];
      EBM_ASSERT(1 <= cDimensions);
      do {
         pCurrentIndexAndCountBinsInitialize->multipliedIndexCur = 0;
         EBM_ASSERT(1 <= pFeatureCombinationEntry->m_pFeature->m_cBins); // this function can handle 1 == cBins even though that's a degenerate case that shouldn't be trained on (dimensions with 1 bin don't contribute anything since they always have the same value)
         multipleTotalInitialize *= static_cast<ptrdiff_t>(pFeatureCombinationEntry->m_pFeature->m_cBins);
         pCurrentIndexAndCountBinsInitialize->multipleTotal = multipleTotalInitialize;
         ++pFeatureCombinationEntry;
         ++pCurrentIndexAndCountBinsInitialize;
      } while(LIKELY(pCurrentIndexAndCountBinsEnd != pCurrentIndexAndCountBinsInitialize));
   }

   // TODO: If we have a compiler cVectorLength, we could put the pPrevious object into our stack since it would have a defined size.  We could then eliminate having to access it through a pointer and we'd just access through the stack pointer
   // TODO: can we put HistogramBucket object onto the stack in other places too?
   // we reserved 1 extra space for these when we binned our buckets
   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pPrevious = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aHistogramBuckets, -multipleTotalInitialize);
   ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pPrevious, aHistogramBucketsEndDebug);

#ifndef NDEBUG
   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pDebugBucket = static_cast<HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> *>(malloc(cBytesPerHistogramBucket));
   pPrevious->AssertZero();
#endif //NDEBUG

   static_assert(k_cDimensionsMax < k_cBitsForSizeTCore, "reserve the highest bit for bit manipulation space");
   EBM_ASSERT(cDimensions < k_cBitsForSizeTCore);
   EBM_ASSERT(2 <= cDimensions);
   const size_t permuteVectorEnd = size_t { 1 } << (cDimensions - 1);
   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pHistogramBucket = aHistogramBuckets;
   
   ptrdiff_t multipliedIndexCur0 = 0;
   const ptrdiff_t multipleTotal0 = currentIndexAndCountBins[0].multipleTotal;

   goto skip_intro;

   CurrentIndexAndCountBins * pCurrentIndexAndCountBins;
   ptrdiff_t multipliedIndexCur;
   while(true) {
      pCurrentIndexAndCountBins->multipliedIndexCur = multipliedIndexCur;

   skip_intro:
      
      // TODO: We're currently reducing the work by a factor of 2 by keeping the pPrevious values.  I think I could reduce the work by annohter factor of 2 if I maintained a 1 dimensional array of previous values for the 2nd dimension.  I think I could reduce by annohter factor of 2 by maintaininng a two dimensional space of previous values, etc..  At the end I think I can remove the combinatorial treatment by adding about the same order of memory as our existing totals space, which is a great tradeoff because then we can figure out a cell by looping N times for N dimensions instead of 2^N!
      //       After we're solved that, I think I can use the resulting intermediate work to avoid the 2^N work in the region totals function that uses our work (this is speculative)
      //       I think instead of storing the totals in the N^D space, I'll end up storing the previous values for the 1st dimension, or maybe I need to keep both.  Or maybe I can eliminate a huge amount of memory in the last dimension by doing a tiny bit of extra work.  I don't know yet.
      //       
      // TODO: before doing the above, I think I want to take what I have and extract a 2-dimensional and 3-dimensional specializations since these don't need the extra complexity.  Especially for 2-D where I don't even need to keep the previous value

      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucket, aHistogramBucketsEndDebug);

      const size_t cInstancesInBucket = pHistogramBucket->cInstancesInBucket + pPrevious->cInstancesInBucket;
      pHistogramBucket->cInstancesInBucket = cInstancesInBucket;
      pPrevious->cInstancesInBucket = cInstancesInBucket;
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         const FractionalDataType sumResidualError = pHistogramBucket->aHistogramBucketVectorEntry[iVector].sumResidualError + pPrevious->aHistogramBucketVectorEntry[iVector].sumResidualError;
         pHistogramBucket->aHistogramBucketVectorEntry[iVector].sumResidualError = sumResidualError;
         pPrevious->aHistogramBucketVectorEntry[iVector].sumResidualError = sumResidualError;

         if(IsClassification(compilerLearningTypeOrCountTargetClasses)) {
            const FractionalDataType sumDenominator = pHistogramBucket->aHistogramBucketVectorEntry[iVector].GetSumDenominator() + pPrevious->aHistogramBucketVectorEntry[iVector].GetSumDenominator();
            pHistogramBucket->aHistogramBucketVectorEntry[iVector].SetSumDenominator(sumDenominator);
            pPrevious->aHistogramBucketVectorEntry[iVector].SetSumDenominator(sumDenominator);
         }
      }

      size_t permuteVector = 1;
      do {
         ptrdiff_t offsetPointer = 0;
         unsigned int evenOdd = 0;
         size_t permuteVectorDestroy = permuteVector;
         // skip the first one since we preserve the total from the previous run instead of adding all the -1 values
         const CurrentIndexAndCountBins * pCurrentIndexAndCountBinsLoop = &currentIndexAndCountBins[1];
         EBM_ASSERT(0 != permuteVectorDestroy);
         do {
            // even though our index is multiplied by the total bins until this point, we only care about the zero bin, and zero multiplied by anything is zero
            if(UNLIKELY(0 != ((0 == pCurrentIndexAndCountBinsLoop->multipliedIndexCur ? 1 : 0) & permuteVectorDestroy))) {
               goto skip_combination;
            }
            offsetPointer = UNPREDICTABLE(0 != (1 & permuteVectorDestroy)) ? pCurrentIndexAndCountBinsLoop[-1].multipleTotal + offsetPointer : offsetPointer;
            evenOdd ^= permuteVectorDestroy; // flip least significant bit if the dimension bit is set
            ++pCurrentIndexAndCountBinsLoop;
            permuteVectorDestroy >>= 1;
            // this (0 != permuteVectorDestroy) condition is somewhat unpredictable because for low dimensions or for low permutations it exits after just a few loops
            // it might be tempting to try and eliminate the loop by templating it and hardcoding the number of iterations based on the number of dimensions, but that would probably
            // be a bad choice because we can exit this loop early when the permutation number is low, and on average that eliminates more than half of the loop iterations
            // the cost of a branch misprediction is probably equal to one complete loop above, but we're reducing it by more than that, and keeping the code more compact by not 
            // exploding the amount of code based on the number of possible dimensions
         } while(LIKELY(0 != permuteVectorDestroy));
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucket, offsetPointer), aHistogramBucketsEndDebug);
         if(UNPREDICTABLE(0 != (1 & evenOdd))) {
            pHistogramBucket->Add(*GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucket, offsetPointer), cTargetClasses);
         } else {
            pHistogramBucket->Subtract(*GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucket, offsetPointer), cTargetClasses);
         }
      skip_combination:
         ++permuteVector;
      } while(LIKELY(permuteVectorEnd != permuteVector));

#ifndef NDEBUG
      size_t aiStart[k_cDimensionsMax];
      size_t aiLast[k_cDimensionsMax];
      ptrdiff_t multipleTotalDebug = -1;
      for(size_t iDebugDimension = 0; iDebugDimension < cDimensions; ++iDebugDimension) {
         aiStart[iDebugDimension] = 0;
         aiLast[iDebugDimension] = static_cast<size_t>((0 == iDebugDimension ? multipliedIndexCur0 : currentIndexAndCountBins[iDebugDimension].multipliedIndexCur) / multipleTotalDebug);
         multipleTotalDebug = currentIndexAndCountBins[iDebugDimension].multipleTotal;
      }
      GetTotalsDebugSlow<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBucketsDebugCopy, pFeatureCombination, aiStart, aiLast, cTargetClasses, pDebugBucket);
      EBM_ASSERT(pDebugBucket->cInstancesInBucket == pHistogramBucket->cInstancesInBucket);
#endif // NDEBUG

      // we're walking through all buckets, so just move to the next one in the flat array, with the knoledge that we'll figure out it's multi-dimenional index below
      pHistogramBucket = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucket, 1);

      // TODO: we are putting storage that would exist in our array from the innermost loop into registers (multipliedIndexCur0 & multipleTotal0).  We can probably do this in many other places as well that use this pattern of indexing via an array

      --multipliedIndexCur0;
      if(LIKELY(multipliedIndexCur0 != multipleTotal0)) {
         goto skip_intro;
      }

      pPrevious->Zero(cTargetClasses);
      multipliedIndexCur0 = 0;
      pCurrentIndexAndCountBins = &currentIndexAndCountBins[1];
      ptrdiff_t multipleTotal = multipleTotal0;
      while(true) {
         multipliedIndexCur = pCurrentIndexAndCountBins->multipliedIndexCur + multipleTotal;
         multipleTotal = pCurrentIndexAndCountBins->multipleTotal;
         if(LIKELY(multipliedIndexCur != multipleTotal)) {
            break;
         }

         pCurrentIndexAndCountBins->multipliedIndexCur = 0;
         ++pCurrentIndexAndCountBins;
         if(UNLIKELY(pCurrentIndexAndCountBinsEnd == pCurrentIndexAndCountBins)) {
#ifndef NDEBUG
            free(pDebugBucket);
#endif // NDEBUG
            return;
         }
      }
   }

   LOG(TraceLevelVerbose, "Exited BuildFastTotalsZeroMemoryIncrease");
}



struct TotalsDimension {
   size_t cIncrement;
   size_t cLast;
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
void GetTotals(const HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, const FeatureCombinationCore * const pFeatureCombination, const size_t * const aiPoint, const size_t directionVector, const size_t cTargetClasses, HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pRet
#ifndef NDEBUG
   , const HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy, const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   // don't LOG this!  It would create way too much chatter!

   static_assert(k_cDimensionsMax < k_cBitsForSizeTCore, "reserve the highest bit for bit manipulation space");
   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->m_cFeatures);
   EBM_ASSERT(1 <= cDimensions);
   EBM_ASSERT(cDimensions < k_cBitsForSizeTCore);

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);

   size_t multipleTotalInitialize = 1;
   size_t startingOffset = 0;
   const FeatureCombinationCore::FeatureCombinationEntry * pFeatureCombinationEntry = &pFeatureCombination->m_FeatureCombinationEntry[0];
   const FeatureCombinationCore::FeatureCombinationEntry * const pFeatureCombinationEntryEnd = &pFeatureCombination->m_FeatureCombinationEntry[cDimensions];
   const size_t * piPointInitialize = aiPoint;

   if(0 == directionVector) {
      // we would require a check in our inner loop below to handle the case of zero FeatureCombinationEntry items, so let's handle it separetly here instead
      EBM_ASSERT(1 <= cDimensions);
      do {
         size_t cBins = pFeatureCombinationEntry->m_pFeature->m_cBins;
         EBM_ASSERT(1 <= cBins); // this function can handle 1 == cBins even though that's a degenerate case that shouldn't be trained on (dimensions with 1 bin don't contribute anything since they always have the same value)
         EBM_ASSERT(*piPointInitialize < cBins);
         EBM_ASSERT(!IsMultiplyError(*piPointInitialize, multipleTotalInitialize)); // we're accessing allocated memory, so this needs to multiply
         size_t addValue = multipleTotalInitialize * (*piPointInitialize);
         EBM_ASSERT(!IsAddError(startingOffset, addValue)); // we're accessing allocated memory, so this needs to add
         startingOffset += addValue;
         EBM_ASSERT(!IsMultiplyError(cBins, multipleTotalInitialize)); // we're accessing allocated memory, so this needs to multiply
         multipleTotalInitialize *= cBins;
         ++pFeatureCombinationEntry;
         ++piPointInitialize;
      } while(LIKELY(pFeatureCombinationEntryEnd != pFeatureCombinationEntry));
      const HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pHistogramBucket = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aHistogramBuckets, startingOffset);
      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pRet, aHistogramBucketsEndDebug);
      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucket, aHistogramBucketsEndDebug);
      pRet->template Copy<compilerLearningTypeOrCountTargetClasses>(*pHistogramBucket, cTargetClasses);
      return;
   }

   //this is a fast way of determining the number of bits (see if the are faster algorithms.. CPU hardware or expoential shifting potentially).  We may use it in the future if we're trying to decide whether to go from (0,0,...,0,0) or (1,1,...,1,1)
   //unsigned int cBits = 0;
   //{
   //   size_t directionVectorDestroy = directionVector;
   //   while(directionVectorDestroy) {
   //      directionVectorDestroy &= (directionVectorDestroy - 1);
   //      ++cBits;
   //   }
   //}

   TotalsDimension totalsDimension[k_cDimensionsMax];
   TotalsDimension * pTotalsDimensionEnd = totalsDimension;
   {
      size_t directionVectorDestroy = directionVector;
      EBM_ASSERT(0 < cDimensions);
      do {
         size_t cBins = pFeatureCombinationEntry->m_pFeature->m_cBins;
         EBM_ASSERT(1 <= cBins); // this function can handle 1 == cBins even though that's a degenerate case that shouldn't be trained on (dimensions with 1 bin don't contribute anything since they always have the same value)
         if(UNPREDICTABLE(0 != (1 & directionVectorDestroy))) {
            EBM_ASSERT(!IsMultiplyError(cBins - 1, multipleTotalInitialize)); // we're accessing allocated memory, so this needs to multiply
            size_t cLast = multipleTotalInitialize * (cBins - 1);
            EBM_ASSERT(!IsMultiplyError(*piPointInitialize, multipleTotalInitialize)); // we're accessing allocated memory, so this needs to multiply
            pTotalsDimensionEnd->cIncrement = multipleTotalInitialize * (*piPointInitialize);
            pTotalsDimensionEnd->cLast = cLast;
            multipleTotalInitialize += cLast;
            ++pTotalsDimensionEnd;
         } else {
            EBM_ASSERT(!IsMultiplyError(*piPointInitialize, multipleTotalInitialize)); // we're accessing allocated memory, so this needs to multiply
            size_t addValue = multipleTotalInitialize * (*piPointInitialize);
            EBM_ASSERT(!IsAddError(startingOffset, addValue)); // we're accessing allocated memory, so this needs to add
            startingOffset += addValue;
            multipleTotalInitialize *= cBins;
         }
         ++pFeatureCombinationEntry;
         ++piPointInitialize;
         directionVectorDestroy >>= 1;
      } while(LIKELY(pFeatureCombinationEntryEnd != pFeatureCombinationEntry));
   }
   const unsigned int cAllBits = static_cast<unsigned int>(pTotalsDimensionEnd - totalsDimension);
   EBM_ASSERT(cAllBits < k_cBitsForSizeTCore);

   pRet->template Zero<compilerLearningTypeOrCountTargetClasses>(cTargetClasses);

   size_t permuteVector = 0;
   do {
      ptrdiff_t offsetPointer = startingOffset;
      size_t evenOdd = cAllBits;
      size_t permuteVectorDestroy = permuteVector;
      const TotalsDimension * pTotalsDimensionLoop = &totalsDimension[0];
      do {
         evenOdd ^= permuteVectorDestroy; // flip least significant bit if the dimension bit is set
         offsetPointer += *(UNPREDICTABLE(0 != (1 & permuteVectorDestroy)) ? &pTotalsDimensionLoop->cLast : &pTotalsDimensionLoop->cIncrement);
         permuteVectorDestroy >>= 1;
         ++pTotalsDimensionLoop;
         // TODO : this (pTotalsDimensionEnd != pTotalsDimensionLoop) condition is somewhat unpredictable since the number of dimensions is small.  Since the number of iterations will remain constant, we can use templates to move this check out of both loop to the completely non-looped outer body and then we eliminate a bunch of unpredictable branches AND a bunch of adds and a lot of other stuff.  If we allow ourselves to come at the vector from either size (0,0,...,0,0) or (1,1,...,1,1) then we only need to hardcode 63/2 loops.
      } while(LIKELY(pTotalsDimensionEnd != pTotalsDimensionLoop));
      const HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pHistogramBucket = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aHistogramBuckets, offsetPointer);
      if(UNPREDICTABLE(0 != (1 & evenOdd))) {
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pRet, aHistogramBucketsEndDebug);
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucket, aHistogramBucketsEndDebug);
         pRet->template Subtract<compilerLearningTypeOrCountTargetClasses>(*pHistogramBucket, cTargetClasses);
      } else {
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pRet, aHistogramBucketsEndDebug);
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucket, aHistogramBucketsEndDebug);
         pRet->template Add<compilerLearningTypeOrCountTargetClasses>(*pHistogramBucket, cTargetClasses);
      }
      ++permuteVector;
   } while(LIKELY(0 == (permuteVector >> cAllBits)));

#ifndef NDEBUG
   if(nullptr != aHistogramBucketsDebugCopy) {
      CompareTotalsDebug<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBucketsDebugCopy, pFeatureCombination, aiPoint, directionVector, cTargetClasses, pRet);
   }
#endif // NDEBUG
}

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
FractionalDataType SweepMultiDiemensional(const HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, const FeatureCombinationCore * const pFeatureCombination, size_t * const aiPoint, const size_t directionVectorLow, const unsigned int iDimensionSweep, const size_t cTargetClasses, HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pHistogramBucketBestAndTemp, size_t * const piBestCut
#ifndef NDEBUG
   , const HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy, const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   // don't LOG this!  It would create way too much chatter!

   // TODO : optimize this function

   EBM_ASSERT(1 <= pFeatureCombination->m_cFeatures);
   EBM_ASSERT(iDimensionSweep < pFeatureCombination->m_cFeatures);
   EBM_ASSERT(0 == (directionVectorLow & (size_t { 1 } << iDimensionSweep)));

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);
   EBM_ASSERT(!IsMultiplyError(2, cBytesPerHistogramBucket)); // we're accessing allocated memory
   const size_t cBytesPerTwoHistogramBuckets = cBytesPerHistogramBucket << 1;

   size_t * const piBin = &aiPoint[iDimensionSweep];
   *piBin = 0;
   size_t directionVectorHigh = directionVectorLow | size_t { 1 } << iDimensionSweep;

   const size_t cBins = pFeatureCombination->m_FeatureCombinationEntry[iDimensionSweep].m_pFeature->m_cBins;
   EBM_ASSERT(2 <= cBins);

   size_t iBestCut = 0;

   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pTotalsLow = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucketBestAndTemp, 2);
   ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pTotalsLow, aHistogramBucketsEndDebug);

   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pTotalsHigh = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucketBestAndTemp, 3);
   ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pTotalsHigh, aHistogramBucketsEndDebug);

   FractionalDataType bestSplit = FractionalDataType { -std::numeric_limits<FractionalDataType>::infinity() };
   size_t iBin = 0;
   do {
      *piBin = iBin;

      GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiPoint, directionVectorLow, cTargetClasses, pTotalsLow
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
#endif // NDEBUG
      );

      GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiPoint, directionVectorHigh, cTargetClasses, pTotalsHigh
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
#endif // NDEBUG
      );

      FractionalDataType splittingScore = 0;
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         splittingScore += 0 == pTotalsLow->cInstancesInBucket ? 0 : EbmStatistics::ComputeNodeSplittingScore(pTotalsLow->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsLow->cInstancesInBucket);
         EBM_ASSERT(0 <= splittingScore);
         splittingScore += 0 == pTotalsHigh->cInstancesInBucket ? 0 : EbmStatistics::ComputeNodeSplittingScore(pTotalsHigh->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsHigh->cInstancesInBucket);
         EBM_ASSERT(0 <= splittingScore);
      }
      EBM_ASSERT(0 <= splittingScore);

      if(bestSplit < splittingScore) {
         bestSplit = splittingScore;
         iBestCut = iBin;

         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucketBestAndTemp, 1), aHistogramBucketsEndDebug);
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pTotalsLow, 1), aHistogramBucketsEndDebug);
         memcpy(pHistogramBucketBestAndTemp, pTotalsLow, cBytesPerTwoHistogramBuckets);
      }
      ++iBin;
   } while(iBin < cBins - 1);
   *piBestCut = iBestCut;
   return bestSplit;
}

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE

// TODO: consider adding controls to disallow cuts that would leave too few cases in a region
// TODO: for higher dimensional spaces, we need to add/subtract individual cells alot and the denominator isn't required in order to make decisions about where to cut.  For dimensions higher than 2, we might want to copy the tensor to a new tensor AFTER binning that keeps only the residuals and then go back to our original tensor after splits to determine the denominator
// TODO: do we really require countCompilerDimensions here?  Does it make any of the code below faster... or alternatively, should we puth the distinction down into a sub-function
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
bool TrainMultiDimensional(CachedTrainingThreadResources<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pCachedThreadResources, const SamplingMethod * const pTrainingSet, const FeatureCombinationCore * const pFeatureCombination, SegmentedTensor<ActiveDataType, FractionalDataType> * const pSmallChangeToModelOverwriteSingleSamplingSet, const size_t cTargetClasses) {
   LOG(TraceLevelVerbose, "Entered TrainMultiDimensional");

   // TODO: we can just re-generate this code 63 times and eliminate the dynamic cDimensions value.  We can also do this in several other places like for SegmentedRegion and other critical places
   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->m_cFeatures);
   EBM_ASSERT(2 <= cDimensions);

   size_t cAuxillaryBucketsForBuildFastTotals = 0;
   size_t cTotalBucketsMainSpace = 1;
   for(size_t iDimension = 0; iDimension < cDimensions; ++iDimension) {
      const size_t cBins = pFeatureCombination->m_FeatureCombinationEntry[iDimension].m_pFeature->m_cBins;
      EBM_ASSERT(2 <= cBins); // we filer out 1 == cBins in allocation.  If cBins could be 1, then we'd need to check at runtime for overflow of cAuxillaryBucketsForBuildFastTotals
      EBM_ASSERT(cAuxillaryBucketsForBuildFastTotals < cTotalBucketsMainSpace); // if this wasn't true then we'd have to check IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace) at runtime
      EBM_ASSERT(!IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace)); // since cBins must be 2 or more, cAuxillaryBucketsForBuildFastTotals must grow slower than cTotalBucketsMainSpace, and we checked at allocation that cTotalBucketsMainSpace would not overflow
      cAuxillaryBucketsForBuildFastTotals += cTotalBucketsMainSpace;
      EBM_ASSERT(!IsMultiplyError(cTotalBucketsMainSpace, cBins)); // we check for simple multiplication overflow from m_cBins in EbmTrainingState->Initialize when we unpack featureCombinationIndexes
      cTotalBucketsMainSpace *= cBins;
      EBM_ASSERT(cAuxillaryBucketsForBuildFastTotals < cTotalBucketsMainSpace); // if this wasn't true then we'd have to check IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace) at runtime
   }
   const size_t cAuxillaryBucketsForSplitting = 24; // we need to reserve 4 PAST the pointer we pass into SweepMultiDiemensional!!!!.  We pass in index 20 at max, so we need 24
   const size_t cAuxillaryBuckets = cAuxillaryBucketsForBuildFastTotals < cAuxillaryBucketsForSplitting ? cAuxillaryBucketsForSplitting : cAuxillaryBucketsForBuildFastTotals;
   if(IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)) {
      LOG(TraceLevelWarning, "WARNING TrainMultiDimensional IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)");
      return true;
   }
   const size_t cTotalBuckets =  cTotalBucketsMainSpace + cAuxillaryBuckets;

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
   if(GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)) {
      LOG(TraceLevelWarning, "WARNING TrainMultiDimensional GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)");
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);
   if(IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)) {
      LOG(TraceLevelWarning, "WARNING TrainMultiDimensional IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)");
      return true;
   }
   const size_t cBytesBuffer = cTotalBuckets * cBytesPerHistogramBucket;

   // we don't need to free this!  It's tracked and reused by pCachedThreadResources
   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets = static_cast<HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> *>(pCachedThreadResources->GetThreadByteBuffer1(cBytesBuffer));
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      LOG(TraceLevelWarning, "WARNING TrainMultiDimensional nullptr == aHistogramBuckets");
      return true;
   }
   memset(aHistogramBuckets, 0, cBytesBuffer);
   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pAuxiliaryBucketZone = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aHistogramBuckets, cTotalBucketsMainSpace);

#ifndef NDEBUG
   const unsigned char * const aHistogramBucketsEndDebug = reinterpret_cast<unsigned char *>(aHistogramBuckets) + cBytesBuffer;
#endif // NDEBUG

   RecursiveBinDataSetTraining<compilerLearningTypeOrCountTargetClasses, 2>::Recursive(cDimensions, aHistogramBuckets, pFeatureCombination, pTrainingSet, cTargetClasses
#ifndef NDEBUG
      , aHistogramBucketsEndDebug
#endif // NDEBUG
   );

#ifndef NDEBUG
   // make a copy of the original binned buckets for debugging purposes
   size_t cTotalBucketsDebug = 1;
   for(size_t iDimensionDebug = 0; iDimensionDebug < cDimensions; ++iDimensionDebug) {
      const size_t cBins = pFeatureCombination->m_FeatureCombinationEntry[iDimensionDebug].m_pFeature->m_cBins;
      EBM_ASSERT(!IsMultiplyError(cTotalBucketsDebug, cBins)); // we checked this above
      cTotalBucketsDebug *= cBins;
   }
   EBM_ASSERT(!IsMultiplyError(cTotalBucketsDebug, cBytesPerHistogramBucket)); // we wouldn't have been able to allocate our main buffer above if this wasn't ok
   const size_t cBytesBufferDebug = cTotalBucketsDebug * cBytesPerHistogramBucket;
   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy = static_cast<HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> *>(malloc(cBytesBufferDebug));
   if(nullptr != aHistogramBucketsDebugCopy) {
      // if we can't allocate, don't fail.. just stop checking
      memcpy(aHistogramBucketsDebugCopy, aHistogramBuckets, cBytesBufferDebug);
   }
#endif // NDEBUG

   BuildFastTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, cTargetClasses, pFeatureCombination, pAuxiliaryBucketZone
#ifndef NDEBUG
      , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
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

   //// TODO this is a fixed length that we should make variable!
   //size_t aTODOSplits[1000000];
   //size_t aTODOSplitsBest[1000000];

   //do {
   //   size_t aiDimensions[k_cDimensionsMax];
   //   memset(aiDimensions, 0, sizeof(aiDimensions[0]) * cDimensions));
   //   while(true) {


   //      EBM_ASSERT(0 == iDimension);
   //      while(true) {
   //         ++aiDimension[iDimension];
   //         if(aiDimension[iDimension] != pFeatureCombinations->m_FeatureCombinationEntry[aiDimensionPermutation[iDimension]].m_pFeature->m_cBins) {
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
      FractionalDataType splittingScore;

      const size_t cBinsDimension1 = pFeatureCombination->m_FeatureCombinationEntry[0].m_pFeature->m_cBins;
      const size_t cBinsDimension2 = pFeatureCombination->m_FeatureCombinationEntry[1].m_pFeature->m_cBins;
      EBM_ASSERT(2 <= cBinsDimension1);
      EBM_ASSERT(2 <= cBinsDimension2);

      FractionalDataType bestSplittingScoreFirst = FractionalDataType { -std::numeric_limits<FractionalDataType>::infinity() };

      size_t cutFirst1Best;
      size_t cutFirst1LowBest;
      size_t cutFirst1HighBest;

      HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotals1LowLowBest = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 0);
      HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotals1LowHighBest = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 1);
      HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotals1HighLowBest = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 2);
      HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotals1HighHighBest = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 3);

      LOG(TraceLevelVerbose, "TrainMultiDimensional Starting FIRST bin sweep loop");
      size_t iBin1 = 0;
      do {
         aiStart[0] = iBin1;

         splittingScore = 0;

         size_t cutSecond1LowBest;
         HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotals2LowLowBest = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 4);
         HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotals2LowHighBest = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 5);
         splittingScore += SweepMultiDiemensional<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, 0x0, 1, cTargetClasses, pTotals2LowLowBest, &cutSecond1LowBest
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
#endif // NDEBUG
            );
         EBM_ASSERT(0 <= splittingScore);

         size_t cutSecond1HighBest;
         HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotals2HighLowBest = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 8);
         HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotals2HighHighBest = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 9);
         splittingScore += SweepMultiDiemensional<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, 0x1, 1, cTargetClasses, pTotals2HighLowBest, &cutSecond1HighBest
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
#endif // NDEBUG
            );
         EBM_ASSERT(0 <= splittingScore);

         if(bestSplittingScoreFirst < splittingScore) {
            bestSplittingScoreFirst = splittingScore;
            cutFirst1Best = iBin1;
            cutFirst1LowBest = cutSecond1LowBest;
            cutFirst1HighBest = cutSecond1HighBest;

            pTotals1LowLowBest->template Copy<compilerLearningTypeOrCountTargetClasses>(*pTotals2LowLowBest, cTargetClasses);
            pTotals1LowHighBest->template Copy<compilerLearningTypeOrCountTargetClasses>(*pTotals2LowHighBest, cTargetClasses);
            pTotals1HighLowBest->template Copy<compilerLearningTypeOrCountTargetClasses>(*pTotals2HighLowBest, cTargetClasses);
            pTotals1HighHighBest->template Copy<compilerLearningTypeOrCountTargetClasses>(*pTotals2HighHighBest, cTargetClasses);
         }
         ++iBin1;
      } while(iBin1 < cBinsDimension1 - 1);

      bool bCutFirst2 = false;

      size_t cutFirst2Best;
      size_t cutFirst2LowBest;
      size_t cutFirst2HighBest;

      HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotals2LowLowBest = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 12);
      HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotals2LowHighBest = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 13);
      HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotals2HighLowBest = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 14);
      HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotals2HighHighBest = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 15);

      LOG(TraceLevelVerbose, "TrainMultiDimensional Starting SECOND bin sweep loop");
      size_t iBin2 = 0;
      do {
         aiStart[1] = iBin2;

         splittingScore = 0;

         size_t cutSecond2LowBest;
         HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotals1LowLowBestInner = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 16);
         HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotals1LowHighBestInner = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 17);
         splittingScore += SweepMultiDiemensional<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, 0x0, 0, cTargetClasses, pTotals1LowLowBestInner, &cutSecond2LowBest
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
#endif // NDEBUG
            );
         EBM_ASSERT(0 <= splittingScore);

         size_t cutSecond2HighBest;
         HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotals1HighLowBestInner = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 20);
         HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotals1HighHighBestInner = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 21);
         splittingScore += SweepMultiDiemensional<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, 0x2, 0, cTargetClasses, pTotals1HighLowBestInner, &cutSecond2HighBest
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
#endif // NDEBUG
            );
         EBM_ASSERT(0 <= splittingScore);

         if(bestSplittingScoreFirst < splittingScore) {
            bestSplittingScoreFirst = splittingScore;
            cutFirst2Best = iBin2;
            cutFirst2LowBest = cutSecond2LowBest;
            cutFirst2HighBest = cutSecond2HighBest;

            pTotals2LowLowBest->template Copy<compilerLearningTypeOrCountTargetClasses>(*pTotals1LowLowBestInner, cTargetClasses);
            pTotals2LowHighBest->template Copy<compilerLearningTypeOrCountTargetClasses>(*pTotals1LowHighBestInner, cTargetClasses);
            pTotals2HighLowBest->template Copy<compilerLearningTypeOrCountTargetClasses>(*pTotals1HighLowBestInner, cTargetClasses);
            pTotals2HighHighBest->template Copy<compilerLearningTypeOrCountTargetClasses>(*pTotals1HighHighBestInner, cTargetClasses);

            bCutFirst2 = true;
         }
         ++iBin2;
      } while(iBin2 < cBinsDimension2 - 1);
      LOG(TraceLevelVerbose, "TrainMultiDimensional Done sweep loops");

      if(bCutFirst2) {
         if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 1)) {
            LOG(TraceLevelWarning, "WARNING TrainMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 1)");
#ifndef NDEBUG
            free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
            return true;
         }
         pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = cutFirst2Best;

         if(cutFirst2LowBest < cutFirst2HighBest) {
            if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)) {
               LOG(TraceLevelWarning, "WARNING TrainMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)");
#ifndef NDEBUG
               free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
               return true;
            }
            if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 2)) {
               LOG(TraceLevelWarning, "WARNING TrainMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 2)");
#ifndef NDEBUG
               free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
               return true;
            }
            pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = cutFirst2LowBest;
            pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[1] = cutFirst2HighBest;
         } else if(cutFirst2HighBest < cutFirst2LowBest) {
            if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)) {
               LOG(TraceLevelWarning, "WARNING TrainMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)");
#ifndef NDEBUG
               free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
               return true;
            }
            if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 2)) {
               LOG(TraceLevelWarning, "WARNING TrainMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 2)");
#ifndef NDEBUG
               free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
               return true;
            }
            pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = cutFirst2HighBest;
            pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[1] = cutFirst2LowBest;
         } else {
            if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1)) {
               LOG(TraceLevelWarning, "WARNING TrainMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1)");
#ifndef NDEBUG
               free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
               return true;
            }

            if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4)) {
               LOG(TraceLevelWarning, "WARNING TrainMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4)");
#ifndef NDEBUG
               free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
               return true;
            }
            pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = cutFirst2LowBest;
         }

         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            FractionalDataType predictionLowLow;
            FractionalDataType predictionLowHigh;
            FractionalDataType predictionHighLow;
            FractionalDataType predictionHighHigh;

            if(IsRegression(compilerLearningTypeOrCountTargetClasses)) {
               // regression
               predictionLowLow = 0 == pTotals2LowLowBest->cInstancesInBucket ? 0 : EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(pTotals2LowLowBest->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotals2LowLowBest->cInstancesInBucket);
               predictionLowHigh = 0 == pTotals2LowHighBest->cInstancesInBucket ? 0 : EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(pTotals2LowHighBest->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotals2LowHighBest->cInstancesInBucket);
               predictionHighLow = 0 == pTotals2HighLowBest->cInstancesInBucket ? 0 : EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(pTotals2HighLowBest->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotals2HighLowBest->cInstancesInBucket);
               predictionHighHigh = 0 == pTotals2HighHighBest->cInstancesInBucket ? 0 : EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(pTotals2HighHighBest->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotals2HighHighBest->cInstancesInBucket);
            } else {
               // classification
               EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));
               predictionLowLow = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pTotals2LowLowBest->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotals2LowLowBest->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
               predictionLowHigh = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pTotals2LowHighBest->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotals2LowHighBest->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
               predictionHighLow = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pTotals2HighLowBest->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotals2HighLowBest->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
               predictionHighHigh = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pTotals2HighHighBest->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotals2HighHighBest->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
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
            LOG(TraceLevelWarning, "WARNING TrainMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1)");
#ifndef NDEBUG
            free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
            return true;
         }
         pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = cutFirst1Best;

         if(cutFirst1LowBest < cutFirst1HighBest) {
            if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)) {
               LOG(TraceLevelWarning, "WARNING TrainMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)");
#ifndef NDEBUG
               free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
               return true;
            }

            if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 2)) {
               LOG(TraceLevelWarning, "WARNING TrainMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 2)");
#ifndef NDEBUG
               free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
               return true;
            }
            pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = cutFirst1LowBest;
            pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[1] = cutFirst1HighBest;
         } else if(cutFirst1HighBest < cutFirst1LowBest) {
            if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)) {
               LOG(TraceLevelWarning, "WARNING TrainMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 6)");
#ifndef NDEBUG
               free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
               return true;
            }

            if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 2)) {
               LOG(TraceLevelWarning, "WARNING TrainMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 2)");
#ifndef NDEBUG
               free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
               return true;
            }
            pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = cutFirst1HighBest;
            pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[1] = cutFirst1LowBest;
         } else {
            if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 1)) {
               LOG(TraceLevelWarning, "WARNING TrainMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(1, 1)");
#ifndef NDEBUG
               free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
               return true;
            }
            if(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4)) {
               LOG(TraceLevelWarning, "WARNING TrainMultiDimensional pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * 4)");
#ifndef NDEBUG
               free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
               return true;
            }
            pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = cutFirst1LowBest;
         }

         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            FractionalDataType predictionLowLow;
            FractionalDataType predictionLowHigh;
            FractionalDataType predictionHighLow;
            FractionalDataType predictionHighHigh;

            if(IsRegression(compilerLearningTypeOrCountTargetClasses)) {
               // regression
               predictionLowLow = 0 == pTotals1LowLowBest->cInstancesInBucket ? 0 : EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(pTotals1LowLowBest->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotals1LowLowBest->cInstancesInBucket);
               predictionLowHigh = 0 == pTotals1LowHighBest->cInstancesInBucket ? 0 : EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(pTotals1LowHighBest->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotals1LowHighBest->cInstancesInBucket);
               predictionHighLow = 0 == pTotals1HighLowBest->cInstancesInBucket ? 0 : EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(pTotals1HighLowBest->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotals1HighLowBest->cInstancesInBucket);
               predictionHighHigh = 0 == pTotals1HighHighBest->cInstancesInBucket ? 0 : EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(pTotals1HighHighBest->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotals1HighHighBest->cInstancesInBucket);
            } else {
               EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));
               // classification
               predictionLowLow = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pTotals1LowLowBest->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotals1LowLowBest->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
               predictionLowHigh = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pTotals1LowHighBest->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotals1LowHighBest->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
               predictionHighLow = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pTotals1HighLowBest->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotals1HighLowBest->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
               predictionHighHigh = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pTotals1HighHighBest->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotals1HighHighBest->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
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
   } else {
      LOG(TraceLevelWarning, "WARNING TrainMultiDimensional 2 != dimensions");
  
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

   LOG(TraceLevelVerbose, "Exited TrainMultiDimensional");
   return false;
}
WARNING_POP

//template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
//bool TrainMultiDimensionalPaulAlgorithm(CachedThreadResources<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pCachedThreadResources, const FeatureInternal * const pTargetFeature, SamplingMethod const * const pTrainingSet, const FeatureCombination * const pFeatureCombination, SegmentedRegion<ActiveDataType, FractionalDataType> * const pSmallChangeToModelOverwriteSingleSamplingSet) {
//   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets = BinDataSet<compilerLearningTypeOrCountTargetClasses>(pCachedThreadResources, pFeatureCombination, pTrainingSet, pTargetFeature);
//   if(UNLIKELY(nullptr == aHistogramBuckets)) {
//      return true;
//   }
//
//   BuildFastTotals(aHistogramBuckets, pTargetFeature, pFeatureCombination);
//
//   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->m_cFeatures);
//   const size_t cTargetClasses = pTargetFeature->m_cBins;
//   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
//   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
//   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);
//
//   size_t aiStart[k_cDimensionsMax];
//   size_t aiLast[k_cDimensionsMax];
//
//   if(2 == cDimensions) {
//      // TODO: somehow avoid having a malloc here, either by allocating these when we allocate our big chunck of memory, or as part of pCachedThreadResources
//      HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * aDynamicHistogramBuckets = static_cast<HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> *>(malloc(cBytesPerHistogramBucket * ));
//
//      const size_t cBinsDimension1 = pFeatureCombination->m_FeatureCombinationEntry[0].m_pFeature->m_cBins;
//      const size_t cBinsDimension2 = pFeatureCombination->m_FeatureCombinationEntry[1].m_pFeature->m_cBins;
//
//      FractionalDataType bestSplittingScore = FractionalDataType { -std::numeric_limits<FractionalDataType>::infinity() };
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
//            FractionalDataType splittingScore;
//
//            HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotalsLowLow = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aDynamicHistogramBuckets, 0);
//            HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotalsHighLow = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aDynamicHistogramBuckets, 1);
//            HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotalsLowHigh = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aDynamicHistogramBuckets, 2);
//            HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotalsHighHigh = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aDynamicHistogramBuckets, 3);
//
//            HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotalsTarget = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aDynamicHistogramBuckets, 4);
//            HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotalsOther = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aDynamicHistogramBuckets, 5);
//
//            aiStart[0] = 0;
//            aiStart[1] = 0;
//            aiLast[0] = iBin1;
//            aiLast[1] = iBin2;
//            GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, aiLast, cTargetClasses, pTotalsLowLow);
//
//            aiStart[0] = iBin1 + 1;
//            aiStart[1] = 0;
//            aiLast[0] = cBinsDimension1 - 1;
//            aiLast[1] = iBin2;
//            GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, aiLast, cTargetClasses, pTotalsHighLow);
//
//            aiStart[0] = 0;
//            aiStart[1] = iBin2 + 1;
//            aiLast[0] = iBin1;
//            aiLast[1] = cBinsDimension2 - 1;
//            GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, aiLast, cTargetClasses, pTotalsLowHigh);
//
//            aiStart[0] = iBin1 + 1;
//            aiStart[1] = iBin2 + 1;
//            aiLast[0] = cBinsDimension1 - 1;
//            aiLast[1] = cBinsDimension2 - 1;
//            GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, aiLast, cTargetClasses, pTotalsHighHigh);
//
//            // LOW LOW
//            pTotalsTarget->Zero(cTargetClasses);
//            pTotalsOther->Zero(cTargetClasses);
//
//            // MODIFY HERE
//            pTotalsTarget->Add(*pTotalsLowLow, cTargetClasses);
//            pTotalsOther->Add(*pTotalsHighLow, cTargetClasses);
//            pTotalsOther->Add(*pTotalsLowHigh, cTargetClasses);
//            pTotalsOther->Add(*pTotalsHighHigh, cTargetClasses);
//            
//            splittingScore = CalculateRegionSplittingScore<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(pTotalsTarget, pTotalsOther, cTargetClasses);
//            if(bestSplittingScore < splittingScore) {
//               bestSplittingScore = splittingScore;
//
//               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = iBin1;
//               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = iBin2;
//
//               for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
//                  FractionalDataType predictionTarget;
//                  FractionalDataType predictionOther;
//
//                  if(IS_REGRESSION(compilerLearningTypeOrCountTargetClasses)) {
//                     // regression
//                     predictionTarget = 0 == pTotalsTarget->cInstancesInBucket ? 0 : ComputeSmallChangeInRegressionPredictionForOneSegment(pTotalsTarget->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsTarget->cInstancesInBucket);
//                     predictionOther = 0 == pTotalsOther->cInstancesInBucket ? 0 : ComputeSmallChangeInRegressionPredictionForOneSegment(pTotalsOther->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsOther->cInstancesInBucket);
//                  } else {
//                     EBM_ASSERT(IS_CLASSIFICATION(compilerLearningTypeOrCountTargetClasses));
//                     // classification
//                     predictionTarget = ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pTotalsTarget->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsTarget->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
//                     predictionOther = ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pTotalsOther->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsOther->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
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
//            pTotalsTarget->Zero(cTargetClasses);
//            pTotalsOther->Zero(cTargetClasses);
//
//            // MODIFY HERE
//            pTotalsOther->Add(*pTotalsLowLow, cTargetClasses);
//            pTotalsTarget->Add(*pTotalsHighLow, cTargetClasses);
//            pTotalsOther->Add(*pTotalsLowHigh, cTargetClasses);
//            pTotalsOther->Add(*pTotalsHighHigh, cTargetClasses);
//
//            splittingScore = CalculateRegionSplittingScore<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(pTotalsTarget, pTotalsOther, cTargetClasses);
//            if(bestSplittingScore < splittingScore) {
//               bestSplittingScore = splittingScore;
//
//               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = iBin1;
//               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = iBin2;
//
//               for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
//                  FractionalDataType predictionTarget;
//                  FractionalDataType predictionOther;
//
//                  if(IS_REGRESSION(compilerLearningTypeOrCountTargetClasses)) {
//                     // regression
//                     predictionTarget = 0 == pTotalsTarget->cInstancesInBucket ? 0 : ComputeSmallChangeInRegressionPredictionForOneSegment(pTotalsTarget->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsTarget->cInstancesInBucket);
//                     predictionOther = 0 == pTotalsOther->cInstancesInBucket ? 0 : ComputeSmallChangeInRegressionPredictionForOneSegment(pTotalsOther->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsOther->cInstancesInBucket);
//                  } else {
//                     EBM_ASSERT(IS_CLASSIFICATION(compilerLearningTypeOrCountTargetClasses));
//                     // classification
//                     predictionTarget = ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pTotalsTarget->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsTarget->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
//                     predictionOther = ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pTotalsOther->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsOther->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
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
//            pTotalsTarget->Zero(cTargetClasses);
//            pTotalsOther->Zero(cTargetClasses);
//
//            // MODIFY HERE
//            pTotalsOther->Add(*pTotalsLowLow, cTargetClasses);
//            pTotalsOther->Add(*pTotalsHighLow, cTargetClasses);
//            pTotalsTarget->Add(*pTotalsLowHigh, cTargetClasses);
//            pTotalsOther->Add(*pTotalsHighHigh, cTargetClasses);
//
//            splittingScore = CalculateRegionSplittingScore<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(pTotalsTarget, pTotalsOther, cTargetClasses);
//            if(bestSplittingScore < splittingScore) {
//               bestSplittingScore = splittingScore;
//
//               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = iBin1;
//               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = iBin2;
//
//               for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
//                  FractionalDataType predictionTarget;
//                  FractionalDataType predictionOther;
//
//                  if(IS_REGRESSION(compilerLearningTypeOrCountTargetClasses)) {
//                     // regression
//                     predictionTarget = 0 == pTotalsTarget->cInstancesInBucket ? 0 : ComputeSmallChangeInRegressionPredictionForOneSegment(pTotalsTarget->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsTarget->cInstancesInBucket);
//                     predictionOther = 0 == pTotalsOther->cInstancesInBucket ? 0 : ComputeSmallChangeInRegressionPredictionForOneSegment(pTotalsOther->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsOther->cInstancesInBucket);
//                  } else {
//                     EBM_ASSERT(IS_CLASSIFICATION(compilerLearningTypeOrCountTargetClasses));
//                     // classification
//                     predictionTarget = ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pTotalsTarget->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsTarget->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
//                     predictionOther = ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pTotalsOther->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsOther->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
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
//            pTotalsTarget->Zero(cTargetClasses);
//            pTotalsOther->Zero(cTargetClasses);
//
//            // MODIFY HERE
//            pTotalsOther->Add(*pTotalsLowLow, cTargetClasses);
//            pTotalsOther->Add(*pTotalsHighLow, cTargetClasses);
//            pTotalsOther->Add(*pTotalsLowHigh, cTargetClasses);
//            pTotalsTarget->Add(*pTotalsHighHigh, cTargetClasses);
//
//            splittingScore = CalculateRegionSplittingScore<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(pTotalsTarget, pTotalsOther, cTargetClasses);
//            if(bestSplittingScore < splittingScore) {
//               bestSplittingScore = splittingScore;
//
//               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)[0] = iBin1;
//               pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(1)[0] = iBin2;
//
//               for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
//                  FractionalDataType predictionTarget;
//                  FractionalDataType predictionOther;
//
//                  if(IS_REGRESSION(compilerLearningTypeOrCountTargetClasses)) {
//                     // regression
//                     predictionTarget = 0 == pTotalsTarget->cInstancesInBucket ? 0 : ComputeSmallChangeInRegressionPredictionForOneSegment(pTotalsTarget->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsTarget->cInstancesInBucket);
//                     predictionOther = 0 == pTotalsOther->cInstancesInBucket ? 0 : ComputeSmallChangeInRegressionPredictionForOneSegment(pTotalsOther->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsOther->cInstancesInBucket);
//                  } else {
//                     EBM_ASSERT(IS_CLASSIFICATION(compilerLearningTypeOrCountTargetClasses));
//                     // classification
//                     predictionTarget = ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pTotalsTarget->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsTarget->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
//                     predictionOther = ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pTotalsOther->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsOther->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
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
//      // TODO: handle this better
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



template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
bool CalculateInteractionScore(const size_t cTargetClasses, CachedInteractionThreadResources * const pCachedThreadResources, const DataSetByFeature * const pDataSet, const FeatureCombinationCore * const pFeatureCombination, FractionalDataType * const pInteractionScoreReturn) {
   // TODO : we NEVER use the denominator term when calculating interaction scores, but we're calculating it and it's taking precious memory.  We should eliminate the denominator term HERE in our datastructures!!!

   LOG(TraceLevelVerbose, "Entered CalculateInteractionScore");

   // TODO: we can just re-generate this code 63 times and eliminate the dynamic cDimensions value.  We can also do this in several other places like for SegmentedRegion and other critical places
   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->m_cFeatures);
   EBM_ASSERT(1 <= cDimensions); // situations with 0 dimensions should have been filtered out before this function was called (but still inside the C++)

   size_t cAuxillaryBucketsForBuildFastTotals = 0;
   size_t cTotalBucketsMainSpace = 1;
   for(size_t iDimension = 0; iDimension < cDimensions; ++iDimension) {
      const size_t cBins = pFeatureCombination->m_FeatureCombinationEntry[iDimension].m_pFeature->m_cBins;
      EBM_ASSERT(2 <= cBins); // situations with 1 bin should have been filtered out before this function was called (but still inside the C++)
      // if cBins could be 1, then we'd need to check at runtime for overflow of cAuxillaryBucketsForBuildFastTotals
      EBM_ASSERT(cAuxillaryBucketsForBuildFastTotals < cTotalBucketsMainSpace); // if this wasn't true then we'd have to check IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace) at runtime
      EBM_ASSERT(!IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace)); // since cBins must be 2 or more, cAuxillaryBucketsForBuildFastTotals must grow slower than cTotalBucketsMainSpace, and we checked at allocation that cTotalBucketsMainSpace would not overflow
      cAuxillaryBucketsForBuildFastTotals += cTotalBucketsMainSpace; // this can overflow, but if it does then we're guaranteed to catch the overflow via the multiplication check below
      if(IsMultiplyError(cTotalBucketsMainSpace, cBins)) {
         // unlike in the training code where we check at allocation time if the tensor created overflows on multiplication
         // we don't know what combination of features our caller will give us for calculating the interaction scores,
         // so we need to check if our caller gave us a tensor that overflows multiplication
         LOG(TraceLevelWarning, "WARNING CalculateInteractionScore IsMultiplyError(cTotalBucketsMainSpace, cBins)");
         return true;
      }
      cTotalBucketsMainSpace *= cBins;
      EBM_ASSERT(cAuxillaryBucketsForBuildFastTotals < cTotalBucketsMainSpace); // if this wasn't true then we'd have to check IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace) at runtime
   }

   const size_t cAuxillaryBucketsForSplitting = 4;
   const size_t cAuxillaryBuckets = cAuxillaryBucketsForBuildFastTotals < cAuxillaryBucketsForSplitting ? cAuxillaryBucketsForSplitting : cAuxillaryBucketsForBuildFastTotals;
   if(IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)) {
      LOG(TraceLevelWarning, "WARNING CalculateInteractionScore IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)");
      return true;
   }
   const size_t cTotalBuckets = cTotalBucketsMainSpace + cAuxillaryBuckets;

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
   if(GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)) {
      LOG(TraceLevelWarning, "WARNING CalculateInteractionScore GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)");
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);
   if(IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)) {
      LOG(TraceLevelWarning, "WARNING CalculateInteractionScore IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)");
      return true;
   }
   const size_t cBytesBuffer = cTotalBuckets * cBytesPerHistogramBucket;

   // this doesn't need to be freed since it's tracked and re-used by the class CachedInteractionThreadResources
   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets = static_cast<HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> *>(pCachedThreadResources->GetThreadByteBuffer1(cBytesBuffer));
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      LOG(TraceLevelWarning, "WARNING CalculateInteractionScore nullptr == aHistogramBuckets");
      return true;
   }
   memset(aHistogramBuckets, 0, cBytesBuffer);

   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pAuxiliaryBucketZone = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aHistogramBuckets, cTotalBucketsMainSpace);

#ifndef NDEBUG
   const unsigned char * const aHistogramBucketsEndDebug = reinterpret_cast<unsigned char *>(aHistogramBuckets) + cBytesBuffer;
#endif // NDEBUG

   // TODO : we don't seem to use the denmoninator in HistogramBucketVectorEntry, so we could remove that variable for classification
   
   // TODO : use the fancy recursive binner that we use in the training version of this function
   BinDataSetInteraction<compilerLearningTypeOrCountTargetClasses>(aHistogramBuckets, pFeatureCombination, pDataSet, cTargetClasses
#ifndef NDEBUG
      , aHistogramBucketsEndDebug
#endif // NDEBUG
      );

#ifndef NDEBUG
   // make a copy of the original binned buckets for debugging purposes
   size_t cTotalBucketsDebug = 1;
   for(size_t iDimensionDebug = 0; iDimensionDebug < cDimensions; ++iDimensionDebug) {
      const size_t cBins = pFeatureCombination->m_FeatureCombinationEntry[iDimensionDebug].m_pFeature->m_cBins;
      EBM_ASSERT(!IsMultiplyError(cTotalBucketsDebug, cBins)); // we checked this above
      cTotalBucketsDebug *= cBins;
   }
   EBM_ASSERT(!IsMultiplyError(cTotalBucketsDebug, cBytesPerHistogramBucket)); // we wouldn't have been able to allocate our main buffer above if this wasn't ok
   const size_t cBytesBufferDebug = cTotalBucketsDebug * cBytesPerHistogramBucket;
   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy = static_cast<HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> *>(malloc(cBytesBufferDebug));
   if(nullptr != aHistogramBucketsDebugCopy) {
      // if we can't allocate, don't fail.. just stop checking
      memcpy(aHistogramBucketsDebugCopy, aHistogramBuckets, cBytesBufferDebug);
   }
#endif // NDEBUG

   BuildFastTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, cTargetClasses, pFeatureCombination, pAuxiliaryBucketZone
#ifndef NDEBUG
      , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
#endif // NDEBUG
      );

   size_t aiStart[k_cDimensionsMax];

   if(2 == cDimensions) {
      HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotalsLowLow = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 0);
      HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotalsLowHigh = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 1);
      HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotalsHighLow = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 2);
      HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pTotalsHighHigh = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 3);

      const size_t cBinsDimension1 = pFeatureCombination->m_FeatureCombinationEntry[0].m_pFeature->m_cBins;
      const size_t cBinsDimension2 = pFeatureCombination->m_FeatureCombinationEntry[1].m_pFeature->m_cBins;
      EBM_ASSERT(1 <= cBinsDimension1); // this function can handle 1 == cBins even though that's a degenerate case that shouldn't be trained on (dimensions with 1 bin don't contribute anything since they always have the same value)
      EBM_ASSERT(1 <= cBinsDimension2); // this function can handle 1 == cBins even though that's a degenerate case that shouldn't be trained on (dimensions with 1 bin don't contribute anything since they always have the same value)

      FractionalDataType bestSplittingScore = FractionalDataType { -std::numeric_limits<FractionalDataType>::infinity() };

      LOG(TraceLevelVerbose, "CalculateInteractionScore Starting bin sweep loop");
      // note : if cBinsDimension1 can be 1 then we can't use a do loop
      for(size_t iBin1 = 0; iBin1 < cBinsDimension1 - 1; ++iBin1) {
         aiStart[0] = iBin1;
         // note : if cBinsDimension2 can be 1 then we can't use a do loop
         for(size_t iBin2 = 0; iBin2 < cBinsDimension2 - 1; ++iBin2) {
            aiStart[1] = iBin2;

            GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, 0x00, cTargetClasses, pTotalsLowLow
#ifndef NDEBUG
               , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
#endif // NDEBUG
               );

            GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, 0x02, cTargetClasses, pTotalsLowHigh
#ifndef NDEBUG
               , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
#endif // NDEBUG
               );

            GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, 0x01, cTargetClasses, pTotalsHighLow
#ifndef NDEBUG
               , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
#endif // NDEBUG
               );

            GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, 0x03, cTargetClasses, pTotalsHighHigh
#ifndef NDEBUG
               , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
#endif // NDEBUG
               );

            FractionalDataType splittingScore = 0;
            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               splittingScore += 0 == pTotalsLowLow->cInstancesInBucket ? 0 : EbmStatistics::ComputeNodeSplittingScore(pTotalsLowLow->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsLowLow->cInstancesInBucket);
               splittingScore += 0 == pTotalsLowHigh->cInstancesInBucket ? 0 : EbmStatistics::ComputeNodeSplittingScore(pTotalsLowHigh->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsLowHigh->cInstancesInBucket);
               splittingScore += 0 == pTotalsHighLow->cInstancesInBucket ? 0 : EbmStatistics::ComputeNodeSplittingScore(pTotalsHighLow->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsHighLow->cInstancesInBucket);
               splittingScore += 0 == pTotalsHighHigh->cInstancesInBucket ? 0 : EbmStatistics::ComputeNodeSplittingScore(pTotalsHighHigh->aHistogramBucketVectorEntry[iVector].sumResidualError, pTotalsHighHigh->cInstancesInBucket);
               EBM_ASSERT(0 <= splittingScore);
            }
            EBM_ASSERT(0 <= splittingScore);

            if(bestSplittingScore < splittingScore) {
               bestSplittingScore = splittingScore;
            }
         }
      }
      LOG(TraceLevelVerbose, "CalculateInteractionScore Done bin sweep loop");

      if(nullptr != pInteractionScoreReturn) {
         *pInteractionScoreReturn = bestSplittingScore;
      }
   } else {
      EBM_ASSERT(false); // we only support pairs currently
      LOG(TraceLevelWarning, "WARNING CalculateInteractionScore 2 != cDimensions");

      // TODO: handle this better
      if(nullptr != pInteractionScoreReturn) {
         *pInteractionScoreReturn = 0; // for now, just return any interactions that have other than 2 dimensions as zero, which means they won't be considered
      }
   }

#ifndef NDEBUG
   free(aHistogramBucketsDebugCopy);
#endif // NDEBUG

   LOG(TraceLevelVerbose, "Exited CalculateInteractionScore");
   return false;
}

#endif // MULTI_DIMENSIONAL_TRAINING_H
