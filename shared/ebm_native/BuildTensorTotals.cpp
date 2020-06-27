// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG

#include "Feature.h"
#include "FeatureGroup.h"

#include "HistogramTargetEntry.h"
#include "HistogramBucket.h"

#include "DimensionMultiple.h"

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
// TODO: after we build pair and triple specific versions of this function, we don't need to have a compiler compilerCountDimensions, since the compiler won't really be able to simpify the loops that are exploding in dimensionality
// TODO: sort our N-dimensional combinations at initialization so that the longest dimension is first!  That way we can more efficiently walk through contiguous memory better in this function!  After we determine the cuts, we can undo the re-ordering for cutting the tensor, which has just a few cells, so will be efficient
template<bool bClassification>
struct FastTotalState {
   HistogramBucket<bClassification> * m_pDimensionalCur;
   HistogramBucket<bClassification> * m_pDimensionalWrap;
   HistogramBucket<bClassification> * m_pDimensionalFirst;
   size_t m_iCur;
   size_t m_cBins;
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerCountDimensions>
class BuildTensorTotalsInternal {
public:
   static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const FeatureCombination * const pFeatureCombination,
      HistogramBucketBase * pBucketAuxiliaryBuildZoneBase,
      HistogramBucketBase * const aHistogramBucketBase
#ifndef NDEBUG
      , HistogramBucketBase * const aHistogramBucketsDebugCopyBase
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      LOG_0(TraceLevelVerbose, "Entered BuildFastTotals");

      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pBucketAuxiliaryBuildZone =
         pBucketAuxiliaryBuildZoneBase->GetHistogramBucket<bClassification>();

      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets =
         aHistogramBucketBase->GetHistogramBucket<bClassification>();

      const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(compilerCountDimensions, pFeatureCombination->GetCountFeatures());
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
         EbmMalloc<HistogramBucket<bClassification>>(1, cBytesPerHistogramBucket);

      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * aHistogramBucketsDebugCopy =
         aHistogramBucketsDebugCopyBase->GetHistogramBucket<bClassification>();

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
               runtimeLearningTypeOrCountTargetClasses,
               pFeatureCombination,
               aHistogramBucketsDebugCopy,
               aiStart,
               aiLast,
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
            char * pCur = reinterpret_cast<char *>(pFastTotalState->m_pDimensionalFirst);
            const char * const pEnd = reinterpret_cast<char *>(pFastTotalState->m_pDimensionalWrap);
            EBM_ASSERT(pCur != pEnd);
            do {
               HistogramBucket<bClassification> * pHistogramBucketCur =
                  reinterpret_cast<HistogramBucket<bClassification> *>(pCur);
               pHistogramBucketCur->Zero(cVectorLength);
               pCur += cBytesPerHistogramBucket;
            } while(pEnd != pCur);

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
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerCountDimensionsPossible>
class BuildTensorTotalsDimensions {
public:
   EBM_INLINE static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const FeatureCombination * const pFeatureCombination,
      HistogramBucketBase * pBucketAuxiliaryBuildZone,
      HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
      , HistogramBucketBase * const aHistogramBucketsDebugCopy
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      static_assert(2 <= compilerCountDimensionsPossible, "can't have less than 2 dimensions for interactions");
      static_assert(compilerCountDimensionsPossible <= k_cDimensionsMax, "can't have more than the max dimensions");

      const size_t runtimeCountDimensions = pFeatureCombination->GetCountFeatures();

      EBM_ASSERT(2 <= runtimeCountDimensions);
      EBM_ASSERT(runtimeCountDimensions <= k_cDimensionsMax);
      if(compilerCountDimensionsPossible == runtimeCountDimensions) {
         BuildTensorTotalsInternal<compilerLearningTypeOrCountTargetClasses, compilerCountDimensionsPossible>::Func(
            runtimeLearningTypeOrCountTargetClasses,
            pFeatureCombination,
            pBucketAuxiliaryBuildZone,
            aHistogramBuckets
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      } else {
         BuildTensorTotalsDimensions<compilerLearningTypeOrCountTargetClasses, compilerCountDimensionsPossible + 1>::Func(
            runtimeLearningTypeOrCountTargetClasses,
            pFeatureCombination,
            pBucketAuxiliaryBuildZone,
            aHistogramBuckets
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class BuildTensorTotalsDimensions<compilerLearningTypeOrCountTargetClasses, k_cCompilerOptimizedCountDimensionsMax + 1> {
public:
   EBM_INLINE static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const FeatureCombination * const pFeatureCombination,
      HistogramBucketBase * pBucketAuxiliaryBuildZone,
      HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
      , HistogramBucketBase * const aHistogramBucketsDebugCopy
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      EBM_ASSERT(2 <= pFeatureCombination->GetCountFeatures());
      EBM_ASSERT(pFeatureCombination->GetCountFeatures() <= k_cDimensionsMax);
      BuildTensorTotalsInternal<compilerLearningTypeOrCountTargetClasses, k_dynamicDimensions>::Func(
         runtimeLearningTypeOrCountTargetClasses,
         pFeatureCombination,
         pBucketAuxiliaryBuildZone,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class BuildTensorTotalsTarget {
public:
   EBM_INLINE static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const FeatureCombination * const pFeatureCombination,
      HistogramBucketBase * pBucketAuxiliaryBuildZone,
      HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
      , HistogramBucketBase * const aHistogramBucketsDebugCopy
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         BuildTensorTotalsDimensions<compilerLearningTypeOrCountTargetClassesPossible, 2>::Func(
            runtimeLearningTypeOrCountTargetClasses,
            pFeatureCombination,
            pBucketAuxiliaryBuildZone,
            aHistogramBuckets
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      } else {
         BuildTensorTotalsTarget<compilerLearningTypeOrCountTargetClassesPossible + 1>::Func(
            runtimeLearningTypeOrCountTargetClasses,
            pFeatureCombination,
            pBucketAuxiliaryBuildZone,
            aHistogramBuckets
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<>
class BuildTensorTotalsTarget<k_cCompilerOptimizedTargetClassesMax + 1> {
public:
   EBM_INLINE static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const FeatureCombination * const pFeatureCombination,
      HistogramBucketBase * pBucketAuxiliaryBuildZone,
      HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
      , HistogramBucketBase * const aHistogramBucketsDebugCopy
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < runtimeLearningTypeOrCountTargetClasses);

      BuildTensorTotalsDimensions<k_dynamicClassification, 2>::Func(
         runtimeLearningTypeOrCountTargetClasses,
         pFeatureCombination,
         pBucketAuxiliaryBuildZone,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
};

extern void BuildTensorTotals(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const FeatureCombination * const pFeatureCombination,
   HistogramBucketBase * pBucketAuxiliaryBuildZone,
   HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
   , HistogramBucketBase * const aHistogramBucketsDebugCopy
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      BuildTensorTotalsTarget<2>::Func(
         runtimeLearningTypeOrCountTargetClasses,
         pFeatureCombination,
         pBucketAuxiliaryBuildZone,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      BuildTensorTotalsDimensions<k_regression, 2>::Func(
         runtimeLearningTypeOrCountTargetClasses,
         pFeatureCombination,
         pBucketAuxiliaryBuildZone,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
}
