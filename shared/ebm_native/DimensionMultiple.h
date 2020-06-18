// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef DIMENSION_MULTIPLE_H
#define DIMENSION_MULTIPLE_H

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "SegmentedTensor.h"
#include "EbmStatisticUtils.h"
#include "CachedThreadResourcesBoosting.h"
#include "CachedThreadResourcesInteraction.h"
#include "Feature.h"
#include "SamplingSet.h"
#include "HistogramBucket.h"

#ifndef NDEBUG

template<bool bClassification>
void GetTotalsDebugSlow(
   const HistogramBucket<bClassification> * const aHistogramBuckets,
   const FeatureCombination * const pFeatureCombination, 
   const size_t * const aiStart, 
   const size_t * const aiLast, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, 
   HistogramBucket<bClassification> * const pRet
) {
   const size_t cDimensions = pFeatureCombination->GetCountFeatures();
   EBM_ASSERT(1 <= cDimensions); // why bother getting totals if we just have 1 bin
   size_t aiDimensions[k_cDimensionsMax];

   size_t iTensorBin = 0;
   size_t valueMultipleInitialize = 1;
   size_t iDimensionInitialize = 0;
   do {
      const size_t cBins = pFeatureCombination->GetFeatureCombinationEntries()[iDimensionInitialize].m_pFeature->GetCountBins();
      EBM_ASSERT(aiStart[iDimensionInitialize] < cBins);
      EBM_ASSERT(aiLast[iDimensionInitialize] < cBins);
      EBM_ASSERT(aiStart[iDimensionInitialize] <= aiLast[iDimensionInitialize]);
      // aiStart[iDimensionInitialize] is less than cBins, so this should multiply
      EBM_ASSERT(!IsMultiplyError(aiStart[iDimensionInitialize], valueMultipleInitialize));
      iTensorBin += aiStart[iDimensionInitialize] * valueMultipleInitialize;
      EBM_ASSERT(!IsMultiplyError(cBins, valueMultipleInitialize)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
      valueMultipleInitialize *= cBins;
      aiDimensions[iDimensionInitialize] = aiStart[iDimensionInitialize];
      ++iDimensionInitialize;
   } while(iDimensionInitialize < cDimensions);

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
   // we've allocated this, so it should fit
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<bClassification>(cVectorLength));
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bClassification>(cVectorLength);
   pRet->Zero(cVectorLength);

   while(true) {
      const HistogramBucket<bClassification> * const pHistogramBucket =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, aHistogramBuckets, iTensorBin);

      pRet->Add(*pHistogramBucket, cVectorLength);

      size_t iDimension = 0;
      size_t valueMultipleLoop = 1;
      while(aiDimensions[iDimension] == aiLast[iDimension]) {
         EBM_ASSERT(aiStart[iDimension] <= aiLast[iDimension]);
         // we've allocated this memory, so it should be reachable, so these numbers should multiply
         EBM_ASSERT(!IsMultiplyError(aiLast[iDimension] - aiStart[iDimension], valueMultipleLoop));
         iTensorBin -= (aiLast[iDimension] - aiStart[iDimension]) * valueMultipleLoop;

         const size_t cBins = pFeatureCombination->GetFeatureCombinationEntries()[iDimension].m_pFeature->GetCountBins();
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

template<bool bClassification>
void CompareTotalsDebug(
   const HistogramBucket<bClassification> * const aHistogramBuckets,
   const FeatureCombination * const pFeatureCombination, 
   const size_t * const aiPoint, 
   const size_t directionVector, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, 
   const HistogramBucket<bClassification> * const pComparison
) {
   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bClassification>(cVectorLength);

   size_t aiStart[k_cDimensionsMax];
   size_t aiLast[k_cDimensionsMax];
   size_t directionVectorDestroy = directionVector;
   for(size_t iDimensionDebug = 0; iDimensionDebug < pFeatureCombination->GetCountFeatures(); ++iDimensionDebug) {
      const size_t cBins = pFeatureCombination->GetFeatureCombinationEntries()[iDimensionDebug].m_pFeature->GetCountBins();
      if(UNPREDICTABLE(0 != (1 & directionVectorDestroy))) {
         aiStart[iDimensionDebug] = aiPoint[iDimensionDebug] + 1;
         aiLast[iDimensionDebug] = cBins - 1;
      } else {
         aiStart[iDimensionDebug] = 0;
         aiLast[iDimensionDebug] = aiPoint[iDimensionDebug];
      }
      directionVectorDestroy >>= 1;
   }

   HistogramBucket<bClassification> * const pComparison2 = EbmMalloc<HistogramBucket<bClassification>, false>(1, cBytesPerHistogramBucket);
   if(nullptr != pComparison2) {
      // if we can't obtain the memory, then don't do the comparison and exit
      GetTotalsDebugSlow<bClassification>(
         aHistogramBuckets, 
         pFeatureCombination, 
         aiStart, 
         aiLast, 
         runtimeLearningTypeOrCountTargetClasses, 
         pComparison2
      );
      EBM_ASSERT(pComparison->m_cInstancesInBucket == pComparison2->m_cInstancesInBucket);
      free(pComparison2);
   }
}

#endif // NDEBUG


//struct CurrentIndexAndCountBins {
//   size_t m_iCur;
//   // copy cBins to our local stack since we'll be referring to them often and our stack is more compact in cache and less all over the place AND not shared between CPUs
//   size_t m_cBins;
//};
//
//template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
//void BuildFastTotals(HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, const FeatureCombination * const pFeatureCombination) {
//   DO: I THINK THIS HAS ALREADY BEEN HANDLED IN OUR OPERATIONAL VERSION of BuildFastTotals -> sort our N-dimensional combinations at program startup so that the longest dimension is first!  That way we can more efficiently walk through contiguous memory better in this function!
//
//   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->GetCountFeatures());
//   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
//   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses));
//
//#ifndef NDEBUG
//   // make a copy of the original binned buckets for debugging purposes
//   size_t cTotalBucketsDebug = 1;
//   for(size_t iDimensionDebug = 0; iDimensionDebug < pFeatureCombination->GetCountFeatures(); ++iDimensionDebug) {
//      const size_t cBins = pFeatureCombination->GetFeatureCombinationEntries()[iDimensionDebug].m_pFeature->m_cBins;
//      EBM_ASSERT(IsMultiplyError(cTotalBucketsDebug, cBins)); // we're accessing allocated memory, so this should work
//      cTotalBucketsDebug *= cBins;
//   }
//   EBM_ASSERT(IsMultiplyError(cTotalBucketsDebug, cBytesPerHistogramBucket)); // we're accessing allocated memory, so this should work
//   const size_t cBytesBufferDebug = cTotalBucketsDebug * cBytesPerHistogramBucket;
//   DO : ALREADY BEEN HANDLED IN OUR OPERATIONAL VERSION of BuildFastTotals -> technically, adding cBytesPerHistogramBucket could overflow so we should handle that instead of asserting
//   EBM_ASSERT(IsAddError(cBytesBufferDebug, cBytesPerHistogramBucket)); // we're just allocating one extra bucket.  If we can't add these two numbers then we shouldn't have been able to allocate the array that we're copying from
//   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy = static_cast<HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> *>(malloc(cBytesBufferDebug + cBytesPerHistogramBucket));
//   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pDebugBucket = nullptr;
//   if(nullptr != aHistogramBucketsDebugCopy) {
//      // if we can't obtain the memory, then don't do the comparison and exit
//      memcpy(aHistogramBucketsDebugCopy, aHistogramBuckets, cBytesBufferDebug);
//      pDebugBucket = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aHistogramBucketsDebugCopy, cTotalBucketsDebug);
//   }
//#endif // NDEBUG
//
//   EBM_ASSERT(0 < cDimensions);
//
//   CurrentIndexAndCountBins currentIndexAndCountBins[k_cDimensionsMax];
//   const CurrentIndexAndCountBins * const pCurrentIndexAndCountBinsEnd = &currentIndexAndCountBins[cDimensions];
//   const FeatureCombinationEntry * pFeatureCombinationEntry = pFeatureCombination->GetFeatureCombinationEntries();
//   for(CurrentIndexAndCountBins * pCurrentIndexAndCountBinsInitialize = currentIndexAndCountBins; pCurrentIndexAndCountBinsEnd != pCurrentIndexAndCountBinsInitialize; ++pCurrentIndexAndCountBinsInitialize, ++pFeatureCombinationEntry) {
//      pCurrentIndexAndCountBinsInitialize->m_iCur = 0;
//      EBM_ASSERT(2 <= pFeatureCombinationEntry->m_pFeature->m_cBins);
//      pCurrentIndexAndCountBinsInitialize->m_cBins = pFeatureCombinationEntry->m_pFeature->m_cBins;
//   }
//
//   static_assert(k_cDimensionsMax < k_cBitsForSizeT, "reserve the highest bit for bit manipulation space");
//   EBM_ASSERT(cDimensions < k_cBitsForSizeT);
//   const size_t permuteVectorEnd = size_t { 1 } << cDimensions;
//   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pHistogramBucket = aHistogramBuckets;
//
//   goto skip_intro;
//
//   CurrentIndexAndCountBins * pCurrentIndexAndCountBins;
//   size_t iBucket;
//   while(true) {
//      pCurrentIndexAndCountBins->m_iCur = iBucket;
//      // we're walking through all buckets, so just move to the next one in the flat array, with the knoledge that we'll figure out it's multi-dimenional index below
//      pHistogramBucket = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucket, 1);
//
//   skip_intro:
//
//      DO : I THINK THIS HAS ALREADY BEEN HANDLED IN OUR OPERATIONAL VERSION of BuildFastTotals -> I think this code below can be made more efficient by storing the sum of all the items in the 0th dimension where we don't subtract the 0th dimension then when we go to sum up the next set we can eliminate half the work!
//
//      size_t permuteVector = 1;
//      do {
//         HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTargetHistogramBucket = pHistogramBucket;
//         bool bPositive = false;
//         size_t permuteVectorDestroy = permuteVector;
//         ptrdiff_t multiplyDimension = -1;
//         pCurrentIndexAndCountBins = &currentIndexAndCountBins[0];
//         do {
//            if(0 != (1 & permuteVectorDestroy)) {
//               if(0 == pCurrentIndexAndCountBins->m_iCur) {
//                  goto skip_combination;
//               }
//               pTargetHistogramBucket = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pTargetHistogramBucket, multiplyDimension);
//               bPositive = !bPositive;
//            }
//            DO: ALREADY BEEN HANDLED IN OUR OPERATIONAL VERSION of BuildFastTotals -> can we eliminate the multiplication by storing the multiples instead of the cBins?
//            multiplyDimension *= pCurrentIndexAndCountBins->m_cBins;
//            ++pCurrentIndexAndCountBins;
//            permuteVectorDestroy >>= 1;
//         } while(0 != permuteVectorDestroy);
//         if(bPositive) {
//            pHistogramBucket->Add(*pTargetHistogramBucket, runtimeLearningTypeOrCountTargetClasses);
//         } else {
//            pHistogramBucket->Subtract(*pTargetHistogramBucket, runtimeLearningTypeOrCountTargetClasses);
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
//            aiLast[iDebugDimension] = currentIndexAndCountBins[iDebugDimension].m_iCur;
//         }
//         GetTotalsDebugSlow<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBucketsDebugCopy, pFeatureCombination, aiStart, aiLast, runtimeLearningTypeOrCountTargetClasses, pDebugBucket);
//         EBM_ASSERT(pDebugBucket->m_cInstancesInBucket == pHistogramBucket->m_cInstancesInBucket);
//
//         free(aHistogramBucketsDebugCopy);
//      }
//#endif // NDEBUG
//
//      pCurrentIndexAndCountBins = &currentIndexAndCountBins[0];
//      while(true) {
//         iBucket = pCurrentIndexAndCountBins->m_iCur + 1;
//         EBM_ASSERT(iBucket <= pCurrentIndexAndCountBins->m_cBins);
//         if(iBucket != pCurrentIndexAndCountBins->m_cBins) {
//            break;
//         }
//         pCurrentIndexAndCountBins->m_iCur = 0;
//         ++pCurrentIndexAndCountBins;
//         if(pCurrentIndexAndCountBinsEnd == pCurrentIndexAndCountBins) {
//            return;
//         }
//      }
//   }
//}
//





//struct CurrentIndexAndCountBins {
//   ptrdiff_t m_multipliedIndexCur;
//   ptrdiff_t m_multipleTotal;
//};
//
//template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
//void BuildFastTotals(HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, const FeatureCombination * const pFeatureCombination) {
//   DO: I THINK THIS HAS ALREADY BEEN HANDLED IN OUR OPERATIONAL VERSION of BuildFastTotals -> sort our N-dimensional combinations at program startup so that the longest dimension is first!  That way we can more efficiently walk through contiguous memory better in this function!
//
//   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->GetCountFeatures());
//   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
//   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses));
//
//#ifndef NDEBUG
//   // make a copy of the original binned buckets for debugging purposes
//   size_t cTotalBucketsDebug = 1;
//   for(size_t iDimensionDebug = 0; iDimensionDebug < pFeatureCombination->GetCountFeatures(); ++iDimensionDebug) {
//      const size_t cBins = pFeatureCombination->GetFeatureCombinationEntries()[iDimensionDebug].m_pFeature->m_cBins;
//      EBM_ASSERT(IsMultiplyError(cTotalBucketsDebug, cBins)); // we're accessing allocated memory, so this should work
//      cTotalBucketsDebug *= cBins;
//   }
//   EBM_ASSERT(IsMultiplyError(cTotalBucketsDebug, cBytesPerHistogramBucket)); // we're accessing allocated memory, so this should work
//   const size_t cBytesBufferDebug = cTotalBucketsDebug * cBytesPerHistogramBucket;
//   DO : ALREADY BEEN HANDLED IN OUR OPERATIONAL VERSION of BuildFastTotals -> technically, adding cBytesPerHistogramBucket could overflow so we should handle that instead of asserting
//   EBM_ASSERT(IsAddError(cBytesBufferDebug, cBytesPerHistogramBucket)); // we're just allocating one extra bucket.  If we can't add these two numbers then we shouldn't have been able to allocate the array that we're copying from
//   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy = static_cast<HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> *>(malloc(cBytesBufferDebug + cBytesPerHistogramBucket));
//   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pDebugBucket = nullptr;
//   if(nullptr != aHistogramBucketsDebugCopy) {
//      // if we can't obtain the memory, then don't do the comparison and exit
//      memcpy(aHistogramBucketsDebugCopy, aHistogramBuckets, cBytesBufferDebug);
//      pDebugBucket = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aHistogramBucketsDebugCopy, cTotalBucketsDebug);
//   }
//#endif // NDEBUG
//
//   EBM_ASSERT(0 < cDimensions);
//
//   CurrentIndexAndCountBins currentIndexAndCountBins[k_cDimensionsMax];
//   const CurrentIndexAndCountBins * const pCurrentIndexAndCountBinsEnd = &currentIndexAndCountBins[cDimensions];
//   const FeatureCombinationEntry * pFeatureCombinationEntry = pFeatureCombination->GetFeatureCombinationEntries();
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
//   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pHistogramBucket = aHistogramBuckets;
//
//   goto skip_intro;
//
//   CurrentIndexAndCountBins * pCurrentIndexAndCountBins;
//   ptrdiff_t multipliedIndexCur;
//   while(true) {
//      pCurrentIndexAndCountBins->multipliedIndexCur = multipliedIndexCur;
//      // we're walking through all buckets, so just move to the next one in the flat array, with the knoledge that we'll figure out it's multi-dimenional index below
//      pHistogramBucket = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucket, 1);
//
//   skip_intro:
//
//      DO : I THINK THIS HAS ALREADY BEEN HANDLED IN OUR OPERATIONAL VERSION of BuildFastTotals -> I think this code below can be made more efficient by storing the sum of all the items in the 0th dimension where we don't subtract the 0th dimension then when we go to sum up the next set we can eliminate half the work!
//
//      size_t permuteVector = 1;
//      do {
//         HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTargetHistogramBucket = pHistogramBucket;
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
//               pTargetHistogramBucket = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pTargetHistogramBucket, multipleTotal);
//               bPositive = !bPositive;
//            }
//            multipleTotal = pCurrentIndexAndCountBins->multipleTotal;
//            ++pCurrentIndexAndCountBins;
//            permuteVectorDestroy >>= 1;
//         } while(0 != permuteVectorDestroy);
//         if(bPositive) {
//            pHistogramBucket->Add(*pTargetHistogramBucket, runtimeLearningTypeOrCountTargetClasses);
//         } else {
//            pHistogramBucket->Subtract(*pTargetHistogramBucket, runtimeLearningTypeOrCountTargetClasses);
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
//         GetTotalsDebugSlow<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBucketsDebugCopy, pFeatureCombination, aiStart, aiLast, runtimeLearningTypeOrCountTargetClasses, pDebugBucket);
//         EBM_ASSERT(pDebugBucket->m_cInstancesInBucket == pHistogramBucket->m_cInstancesInBucket);
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







// TODO : ALL OF THE BELOW!
//- D is the number of dimensions
//- N is the number of cases per dimension(assume all dimensions have the same number of cases for simplicity)
//- we currently have one N^D memory region which allows us to calculate the total from any point to any corner in at worst 2 ^ D operations.If we had 2 ^ D memory spaces and were willing to construct them, then we could calculate the total from any point to any corner in 1 operation.If we made a second total region which had the totals from any point to the(1, 1, ..., 1, 1) corner, then we could calculate any point to corer in sqrt(2 ^ D), which is A LOT BETTER and it only takes twice as much memory.For an 8 dimensional space we would need 16 operations instead of 256!
//- to implement an algorithm that uses the(0, 0, ..., 0, 0) totals volume and the(1, 1, ..., 1, 1) volume, just see whether the input vector has more zeros or 1's and then choose the end point that is closest.
//- we can calculate the total from any arbitrary start and end point(instead of just a point to a corner) if we set the end point as the end and iterate through ALL permutations of all #'s of bits.  There doesn't seem to be any simplification that allows us to handle less than the full combinatoral exploration, even if we constructed a totals for each possible 2 ^ D corner
//- we can calculate the totals dynamically at the same time that we sweep the splitting space for splits.The simplest sweep would be to look at each region from a point to each corner and choose the best split that isolates one of those corners instead of splitting at different poiints in each dimension.If we did the simplest possible thing, then our algorithm would be 2 ^ D*N^D*D OR(2 * N) ^ D*D.If we wanted the more complicated splits, then we might need to first build a totals so that we could determine the "tube totals" and then we could sweep the tube and have the costs on both sides of the split
//- IMEDIATE TASKS :
//- get point to corner working for N - dimensional to(0, 0, ..., 0, 0)
//- get splitting working for N - dimensional
//- have a look at our final dimensionality.Is the totals calculation the bottleneck, or the point to corner totals function ?
//- I think I understand the costs of all implementations of point to corner computation, so don't implement the (1,1,...,1,1) to point algorithm yet.. try implementing the more optimized totals calculation (with more memory).  After we have the optimized totals calculation, then try to re-do the splitting code to do splitting at the same time as totals calculation.  If that isn't better than our existing stuff, then optimzie the point to corner calculation code
//- implement a function that calcualtes the total of any volume using just the(0, 0, ..., 0, 0) totals ..as a debugging function.We might use this for trying out more complicated splits where we allow 2 splits on some axies
// TODO: build a pair and triple specific version of this function.  For pairs we can get ride of the pPrevious and just use the actual cell at (-1,-1) from our current cell, and we can use two loops with everything in memory [look at code above from before we incoporated the previous totals].  Triples would also benefit from pulling things out since we have low iterations of the inner loop and we can access indicies directly without additional add/subtract/bit operations.  Beyond triples, the combinatorial choices start to explode, so we should probably use this general N-dimensional code.
// TODO: after we build pair and triple specific versions of this function, we don't need to have a compiler countCompilerDimensions, since the compiler won't really be able to simpify the loops that are exploding in dimensionality
// TODO: sort our N-dimensional combinations at initialization so that the longest dimension is first!  That way we can more efficiently walk through contiguous memory better in this function!  After we determine the cuts, we can undo the re-ordering for cutting the tensor, which has just a few cells, so will be efficient
template<bool bClassification>
struct FastTotalState {
   HistogramBucket<bClassification> * m_pDimensionalCur;
   HistogramBucket<bClassification> * m_pDimensionalWrap;
   HistogramBucket<bClassification> * m_pDimensionalFirst;
   size_t m_iCur;
   size_t m_cBins;
};
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
void BuildFastTotals(
   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, 
   const FeatureCombination * const pFeatureCombination, 
   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pBucketAuxiliaryBuildZone
#ifndef NDEBUG
   , const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy, 
   const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelVerbose, "Entered BuildFastTotals");

   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->GetCountFeatures());
   EBM_ASSERT(1 <= cDimensions);

   const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
      compilerLearningTypeOrCountTargetClasses,
      runtimeLearningTypeOrCountTargetClasses
   );
   const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bClassification>(cVectorLength);

   FastTotalState<bClassification> fastTotalState[k_cDimensionsMax];
   const FastTotalState<bClassification> * const pFastTotalStateEnd = &fastTotalState[cDimensions];
   {
      FastTotalState<bClassification> * pFastTotalStateInitialize = fastTotalState;
      const FeatureCombinationEntry * pFeatureCombinationEntry = pFeatureCombination->GetFeatureCombinationEntries();
      size_t multiply = 1;
      EBM_ASSERT(0 < cDimensions);
      do {
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pBucketAuxiliaryBuildZone, aHistogramBucketsEndDebug);

         size_t cBins = pFeatureCombinationEntry->m_pFeature->GetCountBins();
         // this function can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on 
         // (dimensions with 1 bin don't contribute anything since they always have the same value)
         EBM_ASSERT(1 <= cBins);

         pFastTotalStateInitialize->m_iCur = 0;
         pFastTotalStateInitialize->m_cBins = cBins;

         pFastTotalStateInitialize->m_pDimensionalFirst = pBucketAuxiliaryBuildZone;
         pFastTotalStateInitialize->m_pDimensionalCur = pBucketAuxiliaryBuildZone;
         // when we exit, pBucketAuxiliaryBuildZone should be == to aHistogramBucketsEndDebug, which is legal in C++ since it doesn't extend beyond 1 
         // item past the end of the array
         pBucketAuxiliaryBuildZone = GetHistogramBucketByIndex<bClassification>(
            cBytesPerHistogramBucket, 
            pBucketAuxiliaryBuildZone, 
            multiply
         );

#ifndef NDEBUG
         if(pFastTotalStateEnd == pFastTotalStateInitialize + 1) {
            // this is the last iteration, so pBucketAuxiliaryBuildZone should normally point to the memory address one byte past the legal buffer 
            // (normally aHistogramBucketsEndDebug), BUT in rare cases we allocate more memory for the BucketAuxiliaryBuildZone than we use in this 
            // function, so the only thing that we can guarantee is that we're equal or less than aHistogramBucketsEndDebug
            EBM_ASSERT(reinterpret_cast<unsigned char *>(pBucketAuxiliaryBuildZone) <= aHistogramBucketsEndDebug);
         } else {
            // if this isn't the last iteration, then we'll actually be using this memory, so the entire bucket had better be useable
            EBM_ASSERT(reinterpret_cast<unsigned char *>(pBucketAuxiliaryBuildZone) + cBytesPerHistogramBucket <= aHistogramBucketsEndDebug);
         }
         for(HistogramBucket<bClassification> * pDimensionalCur = pFastTotalStateInitialize->m_pDimensionalCur;
            pBucketAuxiliaryBuildZone != pDimensionalCur; 
            pDimensionalCur = GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pDimensionalCur, 1))
         {
            pDimensionalCur->AssertZero(cVectorLength);
         }
#endif // NDEBUG

         // TODO : we don't need either the first or the wrap values since they are the next ones in the list.. we may need to populate one item past 
         // the end and make the list one larger
         pFastTotalStateInitialize->m_pDimensionalWrap = pBucketAuxiliaryBuildZone;

         multiply *= cBins;

         ++pFeatureCombinationEntry;
         ++pFastTotalStateInitialize;
      } while(LIKELY(pFastTotalStateEnd != pFastTotalStateInitialize));
   }

#ifndef NDEBUG
   HistogramBucket<bClassification> * const pDebugBucket =
      EbmMalloc<HistogramBucket<bClassification>, false>(1, cBytesPerHistogramBucket);
#endif //NDEBUG

   HistogramBucket<bClassification> * pHistogramBucket = aHistogramBuckets;

   while(true) {
      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucket, aHistogramBucketsEndDebug);

      HistogramBucket<bClassification> * pAddPrev = pHistogramBucket;
      size_t iDimension = cDimensions;
      do {
         --iDimension;
         HistogramBucket<bClassification> * pAddTo = fastTotalState[iDimension].m_pDimensionalCur;
         pAddTo->Add(*pAddPrev, cVectorLength);
         pAddPrev = pAddTo;
         pAddTo = GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAddTo, 1);
         if(pAddTo == fastTotalState[iDimension].m_pDimensionalWrap) {
            pAddTo = fastTotalState[iDimension].m_pDimensionalFirst;
         }
         fastTotalState[iDimension].m_pDimensionalCur = pAddTo;
      } while(0 != iDimension);
      pHistogramBucket->Copy(*pAddPrev, cVectorLength);

#ifndef NDEBUG
      if(nullptr != aHistogramBucketsDebugCopy && nullptr != pDebugBucket) {
         size_t aiStart[k_cDimensionsMax];
         size_t aiLast[k_cDimensionsMax];
         for(size_t iDebugDimension = 0; iDebugDimension < cDimensions; ++iDebugDimension) {
            aiStart[iDebugDimension] = 0;
            aiLast[iDebugDimension] = fastTotalState[iDebugDimension].m_iCur;
         }
         GetTotalsDebugSlow<bClassification>(
            aHistogramBucketsDebugCopy, 
            pFeatureCombination, 
            aiStart, 
            aiLast, 
            runtimeLearningTypeOrCountTargetClasses, 
            pDebugBucket
         );
         EBM_ASSERT(pDebugBucket->m_cInstancesInBucket == pHistogramBucket->m_cInstancesInBucket);
      }
#endif // NDEBUG

      // we're walking through all buckets, so just move to the next one in the flat array, 
      // with the knowledge that we'll figure out it's multi-dimenional index below
      pHistogramBucket = GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pHistogramBucket, 1);

      FastTotalState<bClassification> * pFastTotalState = &fastTotalState[0];
      while(true) {
         ++pFastTotalState->m_iCur;
         if(LIKELY(pFastTotalState->m_cBins != pFastTotalState->m_iCur)) {
            break;
         }
         pFastTotalState->m_iCur = 0;

         EBM_ASSERT(pFastTotalState->m_pDimensionalFirst == pFastTotalState->m_pDimensionalCur);
         memset(
            pFastTotalState->m_pDimensionalFirst, 
            0, 
            reinterpret_cast<char *>(pFastTotalState->m_pDimensionalWrap) - reinterpret_cast<char *>(pFastTotalState->m_pDimensionalFirst)
         );

         ++pFastTotalState;

         if(UNLIKELY(pFastTotalStateEnd == pFastTotalState)) {
#ifndef NDEBUG
            free(pDebugBucket);
#endif // NDEBUG

            LOG_0(TraceLevelVerbose, "Exited BuildFastTotals");
            return;
         }
      }
   }
}


//struct CurrentIndexAndCountBins {
//   ptrdiff_t m_multipliedIndexCur;
//   ptrdiff_t m_multipleTotal;
//};
//template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
//void BuildFastTotalsZeroMemoryIncrease(HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, const FeatureCombination * const pFeatureCombination
//#ifndef NDEBUG
//   , const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy, const unsigned char * const aHistogramBucketsEndDebug
//#endif // NDEBUG
//) {
//   LOG_0(TraceLevelVerbose, "Entered BuildFastTotalsZeroMemoryIncrease");
//
//   DO: ALREADY BEEN HANDLED IN OUR OPERATIONAL VERSION of BuildFastTotals -> sort our N-dimensional combinations at program startup so that the longest dimension is first!  That way we can more efficiently walk through contiguous memory better in this function!
//
//   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->GetCountFeatures());
//   EBM_ASSERT(1 <= cDimensions);
//
//   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses);
//   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
//   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);
//
//   CurrentIndexAndCountBins currentIndexAndCountBins[k_cDimensionsMax];
//   const CurrentIndexAndCountBins * const pCurrentIndexAndCountBinsEnd = &currentIndexAndCountBins[cDimensions];
//   ptrdiff_t multipleTotalInitialize = -1;
//   {
//      CurrentIndexAndCountBins * pCurrentIndexAndCountBinsInitialize = currentIndexAndCountBins;
//      const FeatureCombinationEntry * pFeatureCombinationEntry = pFeatureCombination->GetFeatureCombinationEntries();
//      EBM_ASSERT(1 <= cDimensions);
//      do {
//         pCurrentIndexAndCountBinsInitialize->multipliedIndexCur = 0;
//         EBM_ASSERT(1 <= pFeatureCombinationEntry->m_pFeature->m_cBins); // this function can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on (dimensions with 1 bin don't contribute anything since they always have the same value)
//         multipleTotalInitialize *= static_cast<ptrdiff_t>(pFeatureCombinationEntry->m_pFeature->m_cBins);
//         pCurrentIndexAndCountBinsInitialize->multipleTotal = multipleTotalInitialize;
//         ++pFeatureCombinationEntry;
//         ++pCurrentIndexAndCountBinsInitialize;
//      } while(LIKELY(pCurrentIndexAndCountBinsEnd != pCurrentIndexAndCountBinsInitialize));
//   }
//
//   // TODO: If we have a compiler cVectorLength, we could put the pPrevious object into our stack since it would have a defined size.  We could then eliminate having to access it through a pointer and we'd just access through the stack pointer
//   // TODO: can we put HistogramBucket object onto the stack in other places too?
//   // we reserved 1 extra space for these when we binned our buckets
//   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pPrevious = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aHistogramBuckets, -multipleTotalInitialize);
//   ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pPrevious, aHistogramBucketsEndDebug);
//
//#ifndef NDEBUG
//   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pDebugBucket = static_cast<HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> *>(malloc(cBytesPerHistogramBucket));
//   pPrevious->AssertZero();
//#endif //NDEBUG
//
//   static_assert(k_cDimensionsMax < k_cBitsForSizeT, "reserve the highest bit for bit manipulation space");
//   EBM_ASSERT(cDimensions < k_cBitsForSizeT);
//   EBM_ASSERT(2 <= cDimensions);
//   const size_t permuteVectorEnd = size_t { 1 } << (cDimensions - 1);
//   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pHistogramBucket = aHistogramBuckets;
//   
//   ptrdiff_t multipliedIndexCur0 = 0;
//   const ptrdiff_t multipleTotal0 = currentIndexAndCountBins[0].multipleTotal;
//
//   goto skip_intro;
//
//   CurrentIndexAndCountBins * pCurrentIndexAndCountBins;
//   ptrdiff_t multipliedIndexCur;
//   while(true) {
//      pCurrentIndexAndCountBins->multipliedIndexCur = multipliedIndexCur;
//
//   skip_intro:
//      
//      // TODO: We're currently reducing the work by a factor of 2 by keeping the pPrevious values.  I think I could reduce the work by annohter factor of 2 if I maintained a 1 dimensional array of previous values for the 2nd dimension.  I think I could reduce by annohter factor of 2 by maintaininng a two dimensional space of previous values, etc..  At the end I think I can remove the combinatorial treatment by adding about the same order of memory as our existing totals space, which is a great tradeoff because then we can figure out a cell by looping N times for N dimensions instead of 2^N!
//      //       After we're solved that, I think I can use the resulting intermediate work to avoid the 2^N work in the region totals function that uses our work (this is speculative)
//      //       I think instead of storing the totals in the N^D space, I'll end up storing the previous values for the 1st dimension, or maybe I need to keep both.  Or maybe I can eliminate a huge amount of memory in the last dimension by doing a tiny bit of extra work.  I don't know yet.
//      //       
//      // TODO: before doing the above, I think I want to take what I have and extract a 2-dimensional and 3-dimensional specializations since these don't need the extra complexity.  Especially for 2-D where I don't even need to keep the previous value
//
//      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucket, aHistogramBucketsEndDebug);
//
//      const size_t cInstancesInBucket = pHistogramBucket->m_cInstancesInBucket + pPrevious->m_cInstancesInBucket;
//      pHistogramBucket->m_cInstancesInBucket = cInstancesInBucket;
//      pPrevious->m_cInstancesInBucket = cInstancesInBucket;
//      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
//         const FloatEbmType sumResidualError = ArrayToPointer(pHistogramBucket->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError + ArrayToPointer(pPrevious->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError;
//         ArrayToPointer(pHistogramBucket->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError = sumResidualError;
//         ArrayToPointer(pPrevious->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError = sumResidualError;
//
//         if(IsClassification(compilerLearningTypeOrCountTargetClasses)) {
//            const FloatEbmType sumDenominator = ArrayToPointer(pHistogramBucket->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator() + ArrayToPointer(pPrevious->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator();
//            ArrayToPointer(pHistogramBucket->m_aHistogramBucketVectorEntry)[iVector].SetSumDenominator(sumDenominator);
//            ArrayToPointer(pPrevious->m_aHistogramBucketVectorEntry)[iVector].SetSumDenominator(sumDenominator);
//         }
//      }
//
//      size_t permuteVector = 1;
//      do {
//         ptrdiff_t offsetPointer = 0;
//         unsigned int evenOdd = 0;
//         size_t permuteVectorDestroy = permuteVector;
//         // skip the first one since we preserve the total from the previous run instead of adding all the -1 values
//         const CurrentIndexAndCountBins * pCurrentIndexAndCountBinsLoop = &currentIndexAndCountBins[1];
//         EBM_ASSERT(0 != permuteVectorDestroy);
//         do {
//            // even though our index is multiplied by the total bins until this point, we only care about the zero bin, and zero multiplied by anything is zero
//            if(UNLIKELY(0 != ((0 == pCurrentIndexAndCountBinsLoop->multipliedIndexCur ? 1 : 0) & permuteVectorDestroy))) {
//               goto skip_combination;
//            }
//            offsetPointer = UNPREDICTABLE(0 != (1 & permuteVectorDestroy)) ? pCurrentIndexAndCountBinsLoop[-1].multipleTotal + offsetPointer : offsetPointer;
//            evenOdd ^= permuteVectorDestroy; // flip least significant bit if the dimension bit is set
//            ++pCurrentIndexAndCountBinsLoop;
//            permuteVectorDestroy >>= 1;
//            // this (0 != permuteVectorDestroy) condition is somewhat unpredictable because for low dimensions or for low permutations it exits after just a few loops
//            // it might be tempting to try and eliminate the loop by templating it and hardcoding the number of iterations based on the number of dimensions, but that would probably
//            // be a bad choice because we can exit this loop early when the permutation number is low, and on average that eliminates more than half of the loop iterations
//            // the cost of a branch misprediction is probably equal to one complete loop above, but we're reducing it by more than that, and keeping the code more compact by not 
//            // exploding the amount of code based on the number of possible dimensions
//         } while(LIKELY(0 != permuteVectorDestroy));
//         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucket, offsetPointer), aHistogramBucketsEndDebug);
//         if(UNPREDICTABLE(0 != (1 & evenOdd))) {
//            pHistogramBucket->Add(*GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucket, offsetPointer), runtimeLearningTypeOrCountTargetClasses);
//         } else {
//            pHistogramBucket->Subtract(*GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucket, offsetPointer), runtimeLearningTypeOrCountTargetClasses);
//         }
//      skip_combination:
//         ++permuteVector;
//      } while(LIKELY(permuteVectorEnd != permuteVector));
//
//#ifndef NDEBUG
//      size_t aiStart[k_cDimensionsMax];
//      size_t aiLast[k_cDimensionsMax];
//      ptrdiff_t multipleTotalDebug = -1;
//      for(size_t iDebugDimension = 0; iDebugDimension < cDimensions; ++iDebugDimension) {
//         aiStart[iDebugDimension] = 0;
//         aiLast[iDebugDimension] = static_cast<size_t>((0 == iDebugDimension ? multipliedIndexCur0 : currentIndexAndCountBins[iDebugDimension].multipliedIndexCur) / multipleTotalDebug);
//         multipleTotalDebug = currentIndexAndCountBins[iDebugDimension].multipleTotal;
//      }
//      GetTotalsDebugSlow<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBucketsDebugCopy, pFeatureCombination, aiStart, aiLast, runtimeLearningTypeOrCountTargetClasses, pDebugBucket);
//      EBM_ASSERT(pDebugBucket->m_cInstancesInBucket == pHistogramBucket->m_cInstancesInBucket);
//#endif // NDEBUG
//
//      // we're walking through all buckets, so just move to the next one in the flat array, with the knoledge that we'll figure out it's multi-dimenional index below
//      pHistogramBucket = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pHistogramBucket, 1);
//
//      // TODO: we are putting storage that would exist in our array from the innermost loop into registers (multipliedIndexCur0 & multipleTotal0).  We can probably do this in many other places as well that use this pattern of indexing via an array
//
//      --multipliedIndexCur0;
//      if(LIKELY(multipliedIndexCur0 != multipleTotal0)) {
//         goto skip_intro;
//      }
//
//      pPrevious->Zero(runtimeLearningTypeOrCountTargetClasses);
//      multipliedIndexCur0 = 0;
//      pCurrentIndexAndCountBins = &currentIndexAndCountBins[1];
//      ptrdiff_t multipleTotal = multipleTotal0;
//      while(true) {
//         multipliedIndexCur = pCurrentIndexAndCountBins->multipliedIndexCur + multipleTotal;
//         multipleTotal = pCurrentIndexAndCountBins->multipleTotal;
//         if(LIKELY(multipliedIndexCur != multipleTotal)) {
//            break;
//         }
//
//         pCurrentIndexAndCountBins->multipliedIndexCur = 0;
//         ++pCurrentIndexAndCountBins;
//         if(UNLIKELY(pCurrentIndexAndCountBinsEnd == pCurrentIndexAndCountBins)) {
//#ifndef NDEBUG
//            free(pDebugBucket);
//#endif // NDEBUG
//            return;
//         }
//      }
//   }
//
//   LOG_0(TraceLevelVerbose, "Exited BuildFastTotalsZeroMemoryIncrease");
//}



struct TotalsDimension {
   size_t m_cIncrement;
   size_t m_cLast;
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
void GetTotals(
   const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, 
   const FeatureCombination * const pFeatureCombination, 
   const size_t * const aiPoint, 
   const size_t directionVector, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, 
   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pRet
#ifndef NDEBUG
   , const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy, 
   const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

   // don't LOG this!  It would create way too much chatter!

   static_assert(k_cDimensionsMax < k_cBitsForSizeT, "reserve the highest bit for bit manipulation space");
   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->GetCountFeatures());
   EBM_ASSERT(1 <= cDimensions);
   EBM_ASSERT(cDimensions < k_cBitsForSizeT);

   const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
      compilerLearningTypeOrCountTargetClasses,
      runtimeLearningTypeOrCountTargetClasses
   );
   const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bClassification>(cVectorLength);

   size_t multipleTotalInitialize = 1;
   size_t startingOffset = 0;
   const FeatureCombinationEntry * pFeatureCombinationEntry = pFeatureCombination->GetFeatureCombinationEntries();
   const FeatureCombinationEntry * const pFeatureCombinationEntryEnd = &pFeatureCombinationEntry[cDimensions];
   const size_t * piPointInitialize = aiPoint;

   if(0 == directionVector) {
      // we would require a check in our inner loop below to handle the case of zero FeatureCombinationEntry items, so let's handle it separetly here instead
      EBM_ASSERT(1 <= cDimensions);
      do {
         size_t cBins = pFeatureCombinationEntry->m_pFeature->GetCountBins();
         // this function can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on 
         // (dimensions with 1 bin don't contribute anything since they always have the same value)
         EBM_ASSERT(1 <= cBins);
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
      const HistogramBucket<bClassification> * const pHistogramBucket =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, aHistogramBuckets, startingOffset);
      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pRet, aHistogramBucketsEndDebug);
      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucket, aHistogramBucketsEndDebug);
      pRet->Copy(*pHistogramBucket, cVectorLength);
      return;
   }

   //this is a fast way of determining the number of bits (see if the are faster algorithms.. CPU hardware or expoential shifting potentially).  
   // We may use it in the future if we're trying to decide whether to go from (0,0,...,0,0) or (1,1,...,1,1)
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
         size_t cBins = pFeatureCombinationEntry->m_pFeature->GetCountBins();
         // this function can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on 
         // (dimensions with 1 bin don't contribute anything since they always have the same value)
         EBM_ASSERT(1 <= cBins);
         if(UNPREDICTABLE(0 != (1 & directionVectorDestroy))) {
            EBM_ASSERT(!IsMultiplyError(cBins - 1, multipleTotalInitialize)); // we're accessing allocated memory, so this needs to multiply
            size_t cLast = multipleTotalInitialize * (cBins - 1);
            EBM_ASSERT(!IsMultiplyError(*piPointInitialize, multipleTotalInitialize)); // we're accessing allocated memory, so this needs to multiply
            pTotalsDimensionEnd->m_cIncrement = multipleTotalInitialize * (*piPointInitialize);
            pTotalsDimensionEnd->m_cLast = cLast;
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
   EBM_ASSERT(cAllBits < k_cBitsForSizeT);

   pRet->Zero(cVectorLength);

   size_t permuteVector = 0;
   do {
      size_t offsetPointer = startingOffset;
      size_t evenOdd = cAllBits;
      size_t permuteVectorDestroy = permuteVector;
      const TotalsDimension * pTotalsDimensionLoop = &totalsDimension[0];
      do {
         evenOdd ^= permuteVectorDestroy; // flip least significant bit if the dimension bit is set
         offsetPointer += *(UNPREDICTABLE(0 != (1 & permuteVectorDestroy)) ? &pTotalsDimensionLoop->m_cLast : &pTotalsDimensionLoop->m_cIncrement);
         permuteVectorDestroy >>= 1;
         ++pTotalsDimensionLoop;
         // TODO : this (pTotalsDimensionEnd != pTotalsDimensionLoop) condition is somewhat unpredictable since the number of dimensions is small.  
         // Since the number of iterations will remain constant, we can use templates to move this check out of both loop to the completely non-looped 
         // outer body and then we eliminate a bunch of unpredictable branches AND a bunch of adds and a lot of other stuff.  If we allow 
         // ourselves to come at the vector from either size (0,0,...,0,0) or (1,1,...,1,1) then we only need to hardcode 63/2 loops.
      } while(LIKELY(pTotalsDimensionEnd != pTotalsDimensionLoop));
      // TODO : eliminate this multiplication of cBytesPerHistogramBucket by offsetPointer by multiplying both the startingOffset and the 
      // m_cLast & m_cIncrement values by cBytesPerHistogramBucket.  We can eliminate this multiplication each loop!
      const HistogramBucket<bClassification> * const pHistogramBucket =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, aHistogramBuckets, offsetPointer);
      // TODO : we can eliminate this really bad unpredictable branch if we use conditional negation on the values in pHistogramBucket.  
      // We can pass in a bool that indicates if we should take the negation value or the original at each step 
      // (so we don't need to store it beyond one value either).  We would then have an Add(bool bSubtract, ...) function
      if(UNPREDICTABLE(0 != (1 & evenOdd))) {
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pRet, aHistogramBucketsEndDebug);
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucket, aHistogramBucketsEndDebug);
         pRet->Subtract(*pHistogramBucket, cVectorLength);
      } else {
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pRet, aHistogramBucketsEndDebug);
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucket, aHistogramBucketsEndDebug);
         pRet->Add(*pHistogramBucket, cVectorLength);
      }
      ++permuteVector;
   } while(LIKELY(0 == (permuteVector >> cAllBits)));

#ifndef NDEBUG
   if(nullptr != aHistogramBucketsDebugCopy) {
      CompareTotalsDebug<bClassification>(
         aHistogramBucketsDebugCopy, 
         pFeatureCombination, 
         aiPoint, 
         directionVector, 
         runtimeLearningTypeOrCountTargetClasses, 
         pRet
      );
   }
#endif // NDEBUG
}

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
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

      GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(
         aHistogramBuckets, 
         pFeatureCombination, 
         aiPoint, 
         directionVectorLow, 
         runtimeLearningTypeOrCountTargetClasses, 
         pTotalsLow
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
#endif // NDEBUG
      );
      if(LIKELY(cInstancesRequiredForChildSplitMin <= pTotalsLow->m_cInstancesInBucket)) {
         GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(
            aHistogramBuckets, 
            pFeatureCombination, 
            aiPoint, 
            directionVectorHigh, 
            runtimeLearningTypeOrCountTargetClasses, 
            pTotalsHigh
   #ifndef NDEBUG
            , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
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

// TODO: Implement a far more efficient boosting algorithm for higher dimensional interactions.  The algorithm works as follows:
//   - instead of first calculating the sums at each point for the hyper-dimensional region from the origin to each point, and then later
//     looking for cuts, we can do both at the same time.  We know the total sums for the entire hyper-dimensional region, and as we're doing our summing
//     up, we can calcualte the gain at that point.  The catch is that we can only calculate the gain of the split between the hyper-dimensional region from
//     our current point to the origin, and the rest of the hyper-dimensional area.  We're using boosting though, so as long as we find some cut that makes 
//     things a bit better, we can continue to improve the overall model, subject of course to overfitting.
//   - After we find the best single cut from the origin to every point (and we've selected the best one), we can then go backwards from the point inside the
//     hyper-dimensional volume back towards the origin to select the best interior region vs the entire remaining hyper-dimensional volume.  Potentially we 
//     could at this point then also calculate the sub regions that would be created if we had made planar cuts along both sides of each dimension.  
//   - Example: if we're cutting a cube, we find the best gain from the (0,0,0) to (5,5,5) gives the highest gain, then we go backwards and find that 
//     (5,5,5) -> (1,2,3) gives the best overall cube. We can then either take the cube as one region and the larger entire volume minus the cube as the 
//     other region, or we can separate the entire space into 27 cubes (9 cubes on each plane)
//   - We then need to generalize this algorithm because we don't only want cuts from a single origin, we need to start from each origin.
//   - So, for an N dimensional region, we have 2^N ways to pick which dimensions we traverse in various orders.  So for 3 dimensions, there are 8 corners
//   - So for a 4 dimensional space, we would need to compute the gains for 2^4 times, and for a 16x16x16x16 volume, we would need to check 1,048,576 cells.
//     That seems doable for a 1GHz machine and if each cell consists of 16 bytes then it would be about 16 MB, which is cache fittable.  
//     Probably anything larger than 4 dimensions would dilute the data too far to make reasonable cuts. We can go deeper if some of the features are 
//     near binary, but in any case we'll probably always be on the edge of cache sufficiency.  As the # of dimensions the CPU cost goes by by factors of 2, 
//     so we'd tend to be able to process smaller tensors for the same amount of time.
//   - For each cell, when computing the totals we need to check N memory locations, so for the example above we would 
//     need 4 * 1,048,576 = 4,194,304 operations.
//   - our main issue is that memory won't be layed our very well.  When we traverse from the origin along the default dimensional arragement then our 
//     memory accesses will be ordered well, but anything else will be a problem
//   - transposing doesn't really help since we only visit each node after the transpose once, so why not pay the penalty when computing the totals
//     rather than pay to transpose then process
//     Our algorithm isn't like matrix multiplication where each cell is used many times.  We just check the cells once.
//   - I think though that we can still traverse our memory in whatever order we want, subject to the origin that we need to examine. So, for example, 
//     in a 3 dimensional volume, if we were starting from the (1,1,0) corner, which will be very close to the end of the 1D memory layout, then we'll 
//     be starting very close to the end of the 1D array.  We can move backwards on the first dimension always, then backwards on the second dimension, 
//     then forwards on the third dimension.  We then at least get some locality on our inner loop which always travels in the best memory order, 
//     and I think we get the best memory ordering for the first N dimensions that have the same direction.  So in this example, we get good memory 
//     ordering for the first two dimensions since they are both backwards.  Have a closer look at this property.  I don't think we can travel in any 
//     arbitrary order though since we always need to be growing our totals from our starting origin given that we maintain 
//     a "tube" computations in N-1 dimensional space
//   - to check these properties out, we probably want to first make a special version of our existing hyper-dimensional totals functions that can start 
//     from any given origin instead of just (0,0,0)
//   - it might be the case that for pairs, we can get better results by using a traditional tree cutting algorithm (the existing one).  I should 
//     implement this algorithm above though regardless as it grows at less complexity than other algorithms, so it would be useful in any case.  
//     After it's implemented, we can compare the results against the existing pair computation code
//   - this pair splitting code should be templated for the numbrer of dimensions.  Nobody is really going to use it above 4-5 dimensions, 
//     but it's nice to have the option, but we don't want to implement 2,3,4,5 dimensional versions
//   - consider writing a pair specific version of this algorithm, also because pairs have different algorithms that could be the same
//   - once we have found our initial cut, we should start from the cut point and work backwards to the origin and find if there are any cubic cuts 
//     that maximize gain
//   - we could in theory try and redo the first cut (lookback) like we'll do in the mains
//   - each time we re-examine a sub region like this, or use lookback, we essentially need to re-do the algorithm, but we're only increasing the time 
//     by a small constant factor
//   - if we find it's benefitial to make full hyper-plane cuts along all the dimensions that we find eg: if our cut points are (1,2,3) -> (5, 6,7) then 
//     we would have 27 smaller cut cubes (9 per 2-D plane) then we just need to do a single full-ish sweep of the space to calcualte the totals for 
//     each of the volumes we have under consideration, but that too isn't too costly
// EXISTING ALGORITHM:
//   - our existing algorithm first determins the totals.  It benefits in that we can do this in a cache efficient way where we process the main tensor 
//     in order, although we do use side
//   - total N-1 planes that we also access per cut.  This first step can be ignored since it costs much less than the next part
//   - after getting the totals, we do some kind of search for places to cut, but we need to calculate the total weights while we do so.  
//     Determining the weights is the most expensive operation
//   - the cost for determining volume totals is variable, but it's worst at the ends, where it takes 2^N checks per test point 
//     (and they are not very cache efficient lookups)
//   - the cost is dominated by the worst case, so we can just assume it's the worst case, reduced by some reasonable factor like 2-ish.
//   - if we generate a totals tensor and a reverse totals tensor (totals from the point opposite to the origin), then it takes 2^(N/2) at worst
//   - In the abstract, if we were willing to generate 2^N totals matricies, we could calculate any total from any origin in O(1) time, 
//     but it would take 2^N times as much memory!
//   - Probably the best solution is to just generate 2 sum total matricies one from origin (0,0,..,0,0) and the other at (1,1,..,1,1).  
//     For a 6 dimensional space, that still only requires 8 operations instead of 64.
//
//   - we could in theory re-implement the above more restricted algorithm that looks for volume cuts from each dimension, but we'd then need 
//     either 2^N times more memory, or twice the memory and 2^(N/2), and during the search we'd be using cache inefficient memory access anyways, 
//     so it seems like there would be not benefit to doing a volume from each origin search vs the method above
//   - the other thing to note is that when training pairs after mains, any main cut in the pair is suposed to have limited gain 
//     (and the limited gain is overfitting too), so we really need to look for combinations of cuts for gain if we use the algorithm of picking a cut 
//     in one dimension, then picking a cut in a different dimension, until all the dimension have been fulfilled, that's the simplest possible 
//     set of cuts that divides the region in a way that cuts all dimensions (without which we could reduce the interaction by at least 1 dimension)
//
//   - there are really 2 algorithms that I know of that we can do otherwise.  
//     1) The first one is a simple cross bar, where we choose a cut point inside, then divide the area up into volumes from that point to 
//        each origin, which is the algorithm that we use for interaction detection.  At each point you need to calculate 2^N volumes, and each one of 
//        those takes 2^(N/2) operations
//   - 2) The algorithm we use for interaction cuts.  We choose one dimension to cut, but we don't calculate gain, we choose the next, ect, and then 
//        sweep each dimension.  We get 1 cut along the main dimension, 2 cuts on the second dimension, 4 cuts on the third, etc.  The problem is 
//        that to be fair, we probably want to permute the order of our dimension cuts, which means N! sweep variations
//        Possilby we could randomize the sweep directions and just do 1 each time, but that seems like it would be problematic, or maybe we 
//        choose a sweep direction per inner bag, and then we at least get variability. After we know our sweep direction, we need to visit each point.  
//        Since all dimensions are fixed and we just sweep one at a time, we have 2^N sweep tubes, and each step requires computing at least one side, 
//        so we pay 2^(N/2) operations
//    
//   - the cross bar sweep seems a little too close to our regional cut while building appraoch, and it takes more work.  The 2^N operations 
//     and # of cells are common between that one and the add while sweep version, but the cross bar has an additional 2^(N/2) term vs N for 
//     the sum while working.  Sum while working would be much better for large numbers of dimensions
//   - the permuted solution has the same number of points to examine as the cross bar, and it has 2^N tubes to sweep vs 2^N volumes on each
//     side of the cross bar to examine, and calculating each costs region costs 2^(N/2), so the permuted solutoin takes N! times 
//     more time than the cross bar solution
//   - so the sweep while gain calculation takes less time to examine cuts from each corner than the cross bar, all solutions have bad pipeline 
//     prediction fetch caracteristics and cache characteristics.
//   - the gain calculate while add has the benefit in that it requires no more memory other than the side planes that are needed for addition 
//     calculation anyways, so it's more memory efficient than either of the other two algorithms
//   
//   - SO, regardless as to whether the other algorithms are better, we'll probably want some form of the corner volume while adding to explore
//     higher dimensional spaces.  We can also give options for sweep cuts for lower dimensions. 2-3 dimensional regions seem reasonable.  
//     Beyond that I'd say just do volume addition cuts
//   - we should examine changing the interaction detection code to use our corner cut solution since we exectute that algorithm 
//     on a lot of potential pairs/interactions

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE

// TODO: consider adding controls to disallow cuts that would leave too few cases in a region (use the same minimum number of cases paraemter as the mains)
// TODO: for higher dimensional spaces, we need to add/subtract individual cells alot and the denominator isn't required in order to make decisions about
//   where to cut.  For dimensions higher than 2, we might want to copy the tensor to a new tensor AFTER binning that keeps only the residuals and then 
//    go back to our original tensor after splits to determine the denominator
// TODO: do we really require countCompilerDimensions here?  Does it make any of the code below faster... or alternatively, should we puth the 
//    distinction down into a sub-function
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
bool BoostMultiDimensional(
   CachedBoostingThreadResources * const pCachedThreadResources, 
   const SamplingSet * const pTrainingSet,
   const FeatureCombination * const pFeatureCombination, 
   SegmentedTensor * const pSmallChangeToModelOverwriteSingleSamplingSet, 
   const size_t cInstancesRequiredForChildSplitMin, 
   FloatEbmType * const pTotalGain, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
) {
   constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelVerbose, "Entered BoostMultiDimensional");

   // TODO: we can just re-generate this code 63 times and eliminate the dynamic cDimensions value.  We can also do this in several other 
   // places like for SegmentedRegion and other critical places
   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->GetCountFeatures());
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

   // we don't need to free this!  It's tracked and reused by pCachedThreadResources
   HistogramBucket<bClassification> * const aHistogramBuckets =
      static_cast<HistogramBucket<bClassification> *>(pCachedThreadResources->GetThreadByteBuffer1(cBytesBuffer));
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional nullptr == aHistogramBuckets");
      return true;
   }
   memset(aHistogramBuckets, 0, cBytesBuffer);
   HistogramBucket<bClassification> * pAuxiliaryBucketZone =
      GetHistogramBucketByIndex<bClassification>(
         cBytesPerHistogramBucket, 
         aHistogramBuckets, 
         cTotalBucketsMainSpace
      );

#ifndef NDEBUG
   const unsigned char * const aHistogramBucketsEndDebug = reinterpret_cast<unsigned char *>(aHistogramBuckets) + cBytesBuffer;
#endif // NDEBUG

   RecursiveBinDataSetTraining<compilerLearningTypeOrCountTargetClasses, 2>::Recursive(
      cDimensions, 
      aHistogramBuckets, 
      pFeatureCombination, 
      pTrainingSet, 
      runtimeLearningTypeOrCountTargetClasses
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
      EbmMalloc<HistogramBucket<bClassification>, false>(cTotalBucketsDebug, cBytesPerHistogramBucket);
   if(nullptr != aHistogramBucketsDebugCopy) {
      // if we can't allocate, don't fail.. just stop checking
      const size_t cBytesBufferDebug = cTotalBucketsDebug * cBytesPerHistogramBucket;
      memcpy(aHistogramBucketsDebugCopy, aHistogramBuckets, cBytesBufferDebug);
   }
#endif // NDEBUG

   BuildFastTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(
      aHistogramBuckets, 
      runtimeLearningTypeOrCountTargetClasses, 
      pFeatureCombination, 
      pAuxiliaryBucketZone
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
         const FloatEbmType splittingScoreNew1 = SweepMultiDiemensional<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(
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
            , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
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
            const FloatEbmType splittingScoreNew2 = SweepMultiDiemensional<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(
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
               , aHistogramBucketsDebugCopy, 
               aHistogramBucketsEndDebug
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
         const FloatEbmType splittingScoreNew1 = SweepMultiDiemensional<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(
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
            , aHistogramBucketsDebugCopy, 
            aHistogramBucketsEndDebug
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
            const FloatEbmType splittingScoreNew2 = SweepMultiDiemensional<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(
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
               , aHistogramBucketsDebugCopy, 
               aHistogramBucketsEndDebug
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

//template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
//bool BoostMultiDimensionalPaulAlgorithm(CachedThreadResources<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pCachedThreadResources, const FeatureInternal * const pTargetFeature, SamplingSet const * const pTrainingSet, const FeatureCombination * const pFeatureCombination, SegmentedRegion<ActiveDataType, FloatEbmType> * const pSmallChangeToModelOverwriteSingleSamplingSet) {
//   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets = BinDataSet<compilerLearningTypeOrCountTargetClasses>(pCachedThreadResources, pFeatureCombination, pTrainingSet, pTargetFeature);
//   if(UNLIKELY(nullptr == aHistogramBuckets)) {
//      return true;
//   }
//
//   BuildFastTotals(aHistogramBuckets, pTargetFeature, pFeatureCombination);
//
//   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->GetCountFeatures());
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
//            GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, aiLast, runtimeLearningTypeOrCountTargetClasses, pTotalsLowLow);
//
//            aiStart[0] = iBin1 + 1;
//            aiStart[1] = 0;
//            aiLast[0] = cBinsDimension1 - 1;
//            aiLast[1] = iBin2;
//            GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, aiLast, runtimeLearningTypeOrCountTargetClasses, pTotalsHighLow);
//
//            aiStart[0] = 0;
//            aiStart[1] = iBin2 + 1;
//            aiLast[0] = iBin1;
//            aiLast[1] = cBinsDimension2 - 1;
//            GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, aiLast, runtimeLearningTypeOrCountTargetClasses, pTotalsLowHigh);
//
//            aiStart[0] = iBin1 + 1;
//            aiStart[1] = iBin2 + 1;
//            aiLast[0] = cBinsDimension1 - 1;
//            aiLast[1] = cBinsDimension2 - 1;
//            GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(aHistogramBuckets, pFeatureCombination, aiStart, aiLast, runtimeLearningTypeOrCountTargetClasses, pTotalsHighHigh);
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
//            splittingScore = CalculateRegionSplittingScore<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(pTotalsTarget, pTotalsOther, runtimeLearningTypeOrCountTargetClasses);
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
//            splittingScore = CalculateRegionSplittingScore<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(pTotalsTarget, pTotalsOther, runtimeLearningTypeOrCountTargetClasses);
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
//            splittingScore = CalculateRegionSplittingScore<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(pTotalsTarget, pTotalsOther, runtimeLearningTypeOrCountTargetClasses);
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
//            splittingScore = CalculateRegionSplittingScore<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(pTotalsTarget, pTotalsOther, runtimeLearningTypeOrCountTargetClasses);
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



template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t countCompilerDimensions>
bool CalculateInteractionScore(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, 
   CachedInteractionThreadResources * const pCachedThreadResources, 
   const DataSetByFeature * const pDataSet, 
   const FeatureCombination * const pFeatureCombination, 
   const size_t cInstancesRequiredForChildSplitMin, 
   FloatEbmType * const pInteractionScoreReturn
) {
   // TODO : we NEVER use the denominator term in HistogramBucketVectorEntry when calculating interaction scores, but we're spending time calculating 
   // it, and it's taking up precious memory.  We should eliminate the denominator term HERE in our datastructures OR we should think whether we can 
   // use the denominator as part of the gain function!!!

   constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelVerbose, "Entered CalculateInteractionScore");

   // TODO: we can just re-generate this code 63 times and eliminate the dynamic cDimensions value.  We can also do this in several other places like 
   // for SegmentedRegion and other critical places
   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(countCompilerDimensions, pFeatureCombination->GetCountFeatures());
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

   const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
      compilerLearningTypeOrCountTargetClasses, 
      runtimeLearningTypeOrCountTargetClasses
   );
   const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);

   if(GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)) {
      LOG_0(
         TraceLevelWarning, 
         "WARNING CalculateInteractionScore GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)"
      );
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bClassification>(cVectorLength);
   if(IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)) {
      LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScore IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)");
      return true;
   }
   const size_t cBytesBuffer = cTotalBuckets * cBytesPerHistogramBucket;

   // this doesn't need to be freed since it's tracked and re-used by the class CachedInteractionThreadResources
   HistogramBucket<bClassification> * const aHistogramBuckets =
      static_cast<HistogramBucket<bClassification> *>(pCachedThreadResources->GetThreadByteBuffer1(cBytesBuffer));
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScore nullptr == aHistogramBuckets");
      return true;
   }
   memset(aHistogramBuckets, 0, cBytesBuffer);

   HistogramBucket<bClassification> * pAuxiliaryBucketZone =
      GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, aHistogramBuckets, cTotalBucketsMainSpace);

#ifndef NDEBUG
   const unsigned char * const aHistogramBucketsEndDebug = reinterpret_cast<unsigned char *>(aHistogramBuckets) + cBytesBuffer;
#endif // NDEBUG

   
   // TODO : use the fancy recursive binner that we use in the boosting version of this function
   BinDataSetInteraction<compilerLearningTypeOrCountTargetClasses>(aHistogramBuckets, pFeatureCombination, pDataSet, runtimeLearningTypeOrCountTargetClasses
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
      EbmMalloc<HistogramBucket<bClassification>, false>(cTotalBucketsDebug, cBytesPerHistogramBucket);
   if(nullptr != aHistogramBucketsDebugCopy) {
      // if we can't allocate, don't fail.. just stop checking
      const size_t cBytesBufferDebug = cTotalBucketsDebug * cBytesPerHistogramBucket;
      memcpy(aHistogramBucketsDebugCopy, aHistogramBuckets, cBytesBufferDebug);
   }
#endif // NDEBUG

   BuildFastTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(
      aHistogramBuckets, 
      runtimeLearningTypeOrCountTargetClasses, 
      pFeatureCombination, 
      pAuxiliaryBucketZone
#ifndef NDEBUG
      , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
#endif // NDEBUG
      );

   size_t aiStart[k_cDimensionsMax];

   if(2 == cDimensions) {
      HistogramBucket<bClassification> * pTotalsLowLow =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 0);
      HistogramBucket<bClassification> * pTotalsLowHigh =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 1);
      HistogramBucket<bClassification> * pTotalsHighLow =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 2);
      HistogramBucket<bClassification> * pTotalsHighHigh =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 3);

      const size_t cBinsDimension1 = pFeatureCombination->GetFeatureCombinationEntries()[0].m_pFeature->GetCountBins();
      const size_t cBinsDimension2 = pFeatureCombination->GetFeatureCombinationEntries()[1].m_pFeature->GetCountBins();
      // this function can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on 
      // (dimensions with 1 bin don't contribute anything since they always have the same value)
      EBM_ASSERT(1 <= cBinsDimension1);
      // this function can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on 
      // (dimensions with 1 bin don't contribute anything since they always have the same value)
      EBM_ASSERT(1 <= cBinsDimension2);

      EBM_ASSERT(0 < cInstancesRequiredForChildSplitMin);

      // never return anything above zero, which might happen due to numeric instability if we set this lower than 0
      FloatEbmType bestSplittingScore = FloatEbmType { 0 };

      LOG_0(TraceLevelVerbose, "CalculateInteractionScore Starting bin sweep loop");
      EBM_ASSERT(1 < cBinsDimension1);
      size_t iBin1 = 0;
      do {
         aiStart[0] = iBin1;
         EBM_ASSERT(1 < cBinsDimension2);
         size_t iBin2 = 0;
         do {
            aiStart[1] = iBin2;

            GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(
               aHistogramBuckets, 
               pFeatureCombination, 
               aiStart, 
               0x00, 
               runtimeLearningTypeOrCountTargetClasses, 
               pTotalsLowLow
#ifndef NDEBUG
               , aHistogramBucketsDebugCopy, 
               aHistogramBucketsEndDebug
#endif // NDEBUG
               );
            if(LIKELY(cInstancesRequiredForChildSplitMin <= pTotalsLowLow->m_cInstancesInBucket)) {
               GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(
                  aHistogramBuckets, 
                  pFeatureCombination, 
                  aiStart, 
                  0x02, 
                  runtimeLearningTypeOrCountTargetClasses, 
                  pTotalsLowHigh
#ifndef NDEBUG
                  , aHistogramBucketsDebugCopy, 
                  aHistogramBucketsEndDebug
#endif // NDEBUG
                  );
               if(LIKELY(cInstancesRequiredForChildSplitMin <= pTotalsLowHigh->m_cInstancesInBucket)) {
                  GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(
                     aHistogramBuckets, 
                     pFeatureCombination, 
                     aiStart, 
                     0x01, 
                     runtimeLearningTypeOrCountTargetClasses, 
                     pTotalsHighLow
#ifndef NDEBUG
                     , aHistogramBucketsDebugCopy, aHistogramBucketsEndDebug
#endif // NDEBUG
                     );
                  if(LIKELY(cInstancesRequiredForChildSplitMin <= pTotalsHighLow->m_cInstancesInBucket)) {
                     GetTotals<compilerLearningTypeOrCountTargetClasses, countCompilerDimensions>(
                        aHistogramBuckets, 
                        pFeatureCombination, 
                        aiStart, 
                        0x03, 
                        runtimeLearningTypeOrCountTargetClasses, 
                        pTotalsHighHigh
#ifndef NDEBUG
                        , aHistogramBucketsDebugCopy, 
                        aHistogramBucketsEndDebug
#endif // NDEBUG
                        );
                     if(LIKELY(cInstancesRequiredForChildSplitMin <= pTotalsHighHigh->m_cInstancesInBucket)) {
                        FloatEbmType splittingScore = 0;

                        FloatEbmType cLowLowInstancesInBucket = static_cast<FloatEbmType>(pTotalsLowLow->m_cInstancesInBucket);
                        FloatEbmType cLowHighInstancesInBucket = static_cast<FloatEbmType>(pTotalsLowHigh->m_cInstancesInBucket);
                        FloatEbmType cHighLowInstancesInBucket = static_cast<FloatEbmType>(pTotalsHighLow->m_cInstancesInBucket);
                        FloatEbmType cHighHighInstancesInBucket = static_cast<FloatEbmType>(pTotalsHighHigh->m_cInstancesInBucket);

                        for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
                           // TODO : we can make this faster by doing the division in ComputeNodeSplittingScore after we add all the numerators 
                           // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

                           const FloatEbmType splittingScoreUpdate1 = EbmStatistics::ComputeNodeSplittingScore(
                              ArrayToPointer(pTotalsLowLow->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, 
                              cLowLowInstancesInBucket
                           );
                           EBM_ASSERT(std::isnan(splittingScoreUpdate1) || FloatEbmType { 0 } <= splittingScoreUpdate1);
                           splittingScore += splittingScoreUpdate1;
                           const FloatEbmType splittingScoreUpdate2 = EbmStatistics::ComputeNodeSplittingScore(
                              ArrayToPointer(pTotalsLowHigh->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, cLowHighInstancesInBucket);
                           EBM_ASSERT(std::isnan(splittingScoreUpdate2) || FloatEbmType { 0 } <= splittingScoreUpdate2);
                           splittingScore += splittingScoreUpdate2;
                           const FloatEbmType splittingScoreUpdate3 = EbmStatistics::ComputeNodeSplittingScore(
                              ArrayToPointer(pTotalsHighLow->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, cHighLowInstancesInBucket);
                           EBM_ASSERT(std::isnan(splittingScoreUpdate3) || FloatEbmType { 0 } <= splittingScoreUpdate3);
                           splittingScore += splittingScoreUpdate3;
                           const FloatEbmType splittingScoreUpdate4 = EbmStatistics::ComputeNodeSplittingScore(
                              ArrayToPointer(pTotalsHighHigh->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError, cHighHighInstancesInBucket);
                           EBM_ASSERT(std::isnan(splittingScoreUpdate4) || FloatEbmType { 0 } <= splittingScoreUpdate4);
                           splittingScore += splittingScoreUpdate4;
                        }
                        EBM_ASSERT(std::isnan(splittingScore) || FloatEbmType { 0 } <= splittingScore); // sumations of positive numbers should be positive

                        // if we get a NaN result, we'd like to propagate it by making bestSplit NaN.  The rules for NaN values say that non equality
                        // comparisons are all false so, let's flip this comparison such that it should be true for NaN values.  If the compiler violates 
                        // NaN comparions rules, no big deal.  NaN values will get us soon and shut down boosting.
                        if(UNLIKELY(/* DO NOT CHANGE THIS WITHOUT READING THE ABOVE. WE DO THIS STRANGE COMPARISON FOR NaN values*/ 
                           !(splittingScore <= bestSplittingScore))) 
                        {
                           bestSplittingScore = splittingScore;
                        } else {
                           EBM_ASSERT(!std::isnan(splittingScore));
                        }
                     }
                  }
               }
            }
            ++iBin2;
         } while(iBin2 < cBinsDimension2 - 1);
         ++iBin1;
      } while(iBin1 < cBinsDimension1 - 1);
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
