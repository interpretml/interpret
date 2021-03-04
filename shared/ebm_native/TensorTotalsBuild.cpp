// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG

#include "FeatureAtomic.h"
#include "FeatureGroup.h"

#include "HistogramTargetEntry.h"
#include "HistogramBucket.h"

#include "TensorTotalsSum.h"


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
//     (and the limited gain is overfitting too), so we really need to look for groups of cuts for gain if we use the algorithm of picking a cut 
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
// TODO: after we build pair and triple specific versions of this function, we don't need to have a compiler cCompilerDimensions, since the compiler won't really be able to simpify the loops that are exploding in dimensionality
// TODO: sort our N-dimensional groups at initialization so that the longest dimension is first!  That way we can more efficiently walk through contiguous memory better in this function!  After we determine the cuts, we can undo the re-ordering for cutting the tensor, which has just a few cells, so will be efficient
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t cCompilerDimensions>
class TensorTotalsBuildInternal final {
public:

   TensorTotalsBuildInternal() = delete; // this is a static class.  Do not construct

   static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const FeatureGroup * const pFeatureGroup,
      HistogramBucketBase * pBucketAuxiliaryBuildZoneBase,
      HistogramBucketBase * const aHistogramBucketBase
#ifndef NDEBUG
      , HistogramBucketBase * const aHistogramBucketsDebugCopyBase
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      struct FastTotalState {
         HistogramBucket<bClassification> * m_pDimensionalCur;
         HistogramBucket<bClassification> * m_pDimensionalWrap;
         HistogramBucket<bClassification> * m_pDimensionalFirst;
         size_t m_iCur;
         size_t m_cBins;
      };

      LOG_0(TraceLevelVerbose, "Entered BuildFastTotals");

      HistogramBucket<bClassification> * pBucketAuxiliaryBuildZone =
         pBucketAuxiliaryBuildZoneBase->GetHistogramBucket<bClassification>();

      HistogramBucket<bClassification> * const aHistogramBuckets = 
         aHistogramBucketBase->GetHistogramBucket<bClassification>();

      // TODO: we can get rid of the cCompilerDimensions aspect here by making the 1 or 2 inner loops register/pointer
      //       based and then having a stack based pointer system like the RandomCutState class in CutRandomInternal
      //       to handle any dimensions at the 3rd level and above.  We'll never need to make any additional checks 
      //       on main memory until we reach the 3rd dimension which should be enough for any performance geek
      const size_t cSignificantDimensions = GET_DIMENSIONS(cCompilerDimensions, pFeatureGroup->GetCountSignificantDimensions());
      EBM_ASSERT(1 <= cSignificantDimensions);
      EBM_ASSERT(cSignificantDimensions <= pFeatureGroup->GetCountDimensions());

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      EBM_ASSERT(!GetHistogramBucketSizeOverflow(bClassification, cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);

      FastTotalState fastTotalState[k_cDimensionsMax];
      FastTotalState * pFastTotalStateInitialize = fastTotalState;
      {
         const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
         const FeatureGroupEntry * const pFeatureGroupEntryEnd = pFeatureGroupEntry + pFeatureGroup->GetCountDimensions();
         size_t multiply = 1;
         do {
            ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pBucketAuxiliaryBuildZone, aHistogramBucketsEndDebug);

            const size_t cBins = pFeatureGroupEntry->m_pFeatureAtomic->GetCountBins();
            // cBins can only be 0 if there are zero training and zero validation samples
            // we don't boost or allow interaction updates if there are zero training samples
            EBM_ASSERT(1 <= cBins);
            if(size_t { 1 } < cBins) {
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
               if(&fastTotalState[cSignificantDimensions] == pFastTotalStateInitialize + 1) {
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
               ++pFastTotalStateInitialize;
            }
            ++pFeatureGroupEntry;
         } while(LIKELY(pFeatureGroupEntryEnd != pFeatureGroupEntry));
      }
      EBM_ASSERT(pFastTotalStateInitialize == &fastTotalState[cSignificantDimensions]);

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
         size_t iDimension = cSignificantDimensions;
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
            for(size_t iDebugDimension = 0; iDebugDimension < cSignificantDimensions; ++iDebugDimension) {
               aiStart[iDebugDimension] = 0;
               aiLast[iDebugDimension] = fastTotalState[iDebugDimension].m_iCur;
            }
            TensorTotalsSumDebugSlow<bClassification>(
               runtimeLearningTypeOrCountTargetClasses,
               pFeatureGroup,
               aHistogramBucketsDebugCopy,
               aiStart,
               aiLast,
               pDebugBucket
            );
            EBM_ASSERT(pDebugBucket->GetCountSamplesInBucket() == pHistogramBucket->GetCountSamplesInBucket());
         }
#endif // NDEBUG

         // we're walking through all buckets, so just move to the next one in the flat array, 
         // with the knowledge that we'll figure out it's multi-dimenional index below
         pHistogramBucket = GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pHistogramBucket, 1);

         FastTotalState * pFastTotalState = &fastTotalState[0];
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

            if(UNLIKELY(pFastTotalStateInitialize == pFastTotalState)) {
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

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t cCompilerDimensionsPossible>
class TensorTotalsBuildDimensions final {
public:

   TensorTotalsBuildDimensions() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const FeatureGroup * const pFeatureGroup,
      HistogramBucketBase * pBucketAuxiliaryBuildZone,
      HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
      , HistogramBucketBase * const aHistogramBucketsDebugCopy
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      static_assert(1 <= cCompilerDimensionsPossible, "can't have less than 1 dimension");
      static_assert(cCompilerDimensionsPossible <= k_cDimensionsMax, "can't have more than the max dimensions");

      const size_t cRuntimeDimensions = pFeatureGroup->GetCountSignificantDimensions();

      EBM_ASSERT(1 <= cRuntimeDimensions);
      EBM_ASSERT(cRuntimeDimensions <= k_cDimensionsMax);
      if(cCompilerDimensionsPossible == cRuntimeDimensions) {
         TensorTotalsBuildInternal<compilerLearningTypeOrCountTargetClasses, cCompilerDimensionsPossible>::Func(
            runtimeLearningTypeOrCountTargetClasses,
            pFeatureGroup,
            pBucketAuxiliaryBuildZone,
            aHistogramBuckets
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      } else {
         TensorTotalsBuildDimensions<compilerLearningTypeOrCountTargetClasses, cCompilerDimensionsPossible + 1>::Func(
            runtimeLearningTypeOrCountTargetClasses,
            pFeatureGroup,
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
class TensorTotalsBuildDimensions<compilerLearningTypeOrCountTargetClasses, k_cCompilerOptimizedCountDimensionsMax + 1> final {
public:

   TensorTotalsBuildDimensions() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const FeatureGroup * const pFeatureGroup,
      HistogramBucketBase * pBucketAuxiliaryBuildZone,
      HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
      , HistogramBucketBase * const aHistogramBucketsDebugCopy
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      EBM_ASSERT(1 <= pFeatureGroup->GetCountSignificantDimensions());
      EBM_ASSERT(pFeatureGroup->GetCountSignificantDimensions() <= k_cDimensionsMax);
      TensorTotalsBuildInternal<compilerLearningTypeOrCountTargetClasses, k_dynamicDimensions>::Func(
         runtimeLearningTypeOrCountTargetClasses,
         pFeatureGroup,
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
class TensorTotalsBuildTarget final {
public:

   TensorTotalsBuildTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const FeatureGroup * const pFeatureGroup,
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
         TensorTotalsBuildDimensions<compilerLearningTypeOrCountTargetClassesPossible, 2>::Func(
            runtimeLearningTypeOrCountTargetClasses,
            pFeatureGroup,
            pBucketAuxiliaryBuildZone,
            aHistogramBuckets
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      } else {
         TensorTotalsBuildTarget<compilerLearningTypeOrCountTargetClassesPossible + 1>::Func(
            runtimeLearningTypeOrCountTargetClasses,
            pFeatureGroup,
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
class TensorTotalsBuildTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   TensorTotalsBuildTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const FeatureGroup * const pFeatureGroup,
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

      TensorTotalsBuildDimensions<k_dynamicClassification, 2>::Func(
         runtimeLearningTypeOrCountTargetClasses,
         pFeatureGroup,
         pBucketAuxiliaryBuildZone,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
};

extern void TensorTotalsBuild(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const FeatureGroup * const pFeatureGroup,
   HistogramBucketBase * pBucketAuxiliaryBuildZone,
   HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
   , HistogramBucketBase * const aHistogramBucketsDebugCopy
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      TensorTotalsBuildTarget<2>::Func(
         runtimeLearningTypeOrCountTargetClasses,
         pFeatureGroup,
         pBucketAuxiliaryBuildZone,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      TensorTotalsBuildDimensions<k_regression, 2>::Func(
         runtimeLearningTypeOrCountTargetClasses,
         pFeatureGroup,
         pBucketAuxiliaryBuildZone,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
}

// Boneyard of useful ideas below:

//struct CurrentIndexAndCountBins {
//   size_t m_iCur;
//   // copy cBins to our local stack since we'll be referring to them often and our stack is more compact in cache and less all over the place AND not shared between CPUs
//   size_t m_cBins;
//};
//
//template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t cCompilerDimensions>
//void BuildFastTotals(const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, const FeatureGroup * const pFeatureGroup, HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets) {
//   DO: I THINK THIS HAS ALREADY BEEN HANDLED IN OUR OPERATIONAL VERSION of BuildFastTotals -> sort our N-dimensional groups at program startup so that the longest dimension is first!  That way we can more efficiently walk through contiguous memory better in this function!
//
//   const size_t cDimensions = GET_DIMENSIONS(cCompilerDimensions, pFeatureGroup->GetCountDimensions());
//   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
//   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses));
//
//#ifndef NDEBUG
//   // make a copy of the original binned buckets for debugging purposes
//   size_t cTotalBucketsDebug = 1;
//   for(size_t iDimensionDebug = 0; iDimensionDebug < pFeatureGroup->GetCountDimensions(); ++iDimensionDebug) {
//      const size_t cBins = pFeatureGroup->GetFeatureGroupEntries()[iDimensionDebug].m_pFeature->m_cBins;
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
//   const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
//   for(CurrentIndexAndCountBins * pCurrentIndexAndCountBinsInitialize = currentIndexAndCountBins; pCurrentIndexAndCountBinsEnd != pCurrentIndexAndCountBinsInitialize; ++pCurrentIndexAndCountBinsInitialize, ++pFeatureGroupEntry) {
//      pCurrentIndexAndCountBinsInitialize->m_iCur = 0;
//      EBM_ASSERT(2 <= pFeatureGroupEntry->m_pFeature->m_cBins);
//      pCurrentIndexAndCountBinsInitialize->m_cBins = pFeatureGroupEntry->m_pFeature->m_cBins;
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
//                  goto skip_group;
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
//      skip_group:
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
//         TensorTotalsSumDebugSlow<compilerLearningTypeOrCountTargetClasses, cCompilerDimensions>(runtimeLearningTypeOrCountTargetClasses, pFeatureGroup, aHistogramBucketsDebugCopy, aiStart, aiLast, pDebugBucket);
//         EBM_ASSERT(pDebugBucket->GetCountSamplesInBucket() == pHistogramBucket->GetCountSamplesInBucket());
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
//template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t cCompilerDimensions>
//void BuildFastTotals(const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, const FeatureGroup * const pFeatureGroup, HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets) {
//   DO: I THINK THIS HAS ALREADY BEEN HANDLED IN OUR OPERATIONAL VERSION of BuildFastTotals -> sort our N-dimensional groups at program startup so that the longest dimension is first!  That way we can more efficiently walk through contiguous memory better in this function!
//
//   const size_t cDimensions = GET_DIMENSIONS(cCompilerDimensions, pFeatureGroup->GetCountDimensions());
//   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
//   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses));
//
//#ifndef NDEBUG
//   // make a copy of the original binned buckets for debugging purposes
//   size_t cTotalBucketsDebug = 1;
//   for(size_t iDimensionDebug = 0; iDimensionDebug < pFeatureGroup->GetCountDimensions(); ++iDimensionDebug) {
//      const size_t cBins = pFeatureGroup->GetFeatureGroupEntries()[iDimensionDebug].m_pFeature->m_cBins;
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
//   const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
//   ptrdiff_t multipleTotalInitialize = -1;
//   for(CurrentIndexAndCountBins * pCurrentIndexAndCountBinsInitialize = currentIndexAndCountBins; pCurrentIndexAndCountBinsEnd != pCurrentIndexAndCountBinsInitialize; ++pCurrentIndexAndCountBinsInitialize, ++pFeatureGroupEntry) {
//      pCurrentIndexAndCountBinsInitialize->multipliedIndexCur = 0;
//      EBM_ASSERT(2 <= pFeatureGroupEntry->m_pFeature->m_cBins);
//      multipleTotalInitialize *= static_cast<ptrdiff_t>(pFeatureGroupEntry->m_pFeature->m_cBins);
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
//                  goto skip_group;
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
//      skip_group:
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
//         TensorTotalsSumDebugSlow<compilerLearningTypeOrCountTargetClasses, cCompilerDimensions>(runtimeLearningTypeOrCountTargetClasses, pFeatureGroup, aHistogramBucketsDebugCopy, aiStart, aiLast, pDebugBucket);
//         EBM_ASSERT(pDebugBucket->GetCountSamplesInBucket() == pHistogramBucket->GetCountSamplesInBucket());
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









//struct CurrentIndexAndCountBins {
//   ptrdiff_t m_multipliedIndexCur;
//   ptrdiff_t m_multipleTotal;
//};
//template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t cCompilerDimensions>
//void BuildFastTotalsZeroMemoryIncrease(const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, const FeatureGroup * const pFeatureGroup, HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets
//#ifndef NDEBUG
//   , const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy, const unsigned char * const aHistogramBucketsEndDebug
//#endif // NDEBUG
//) {
//   LOG_0(TraceLevelVerbose, "Entered BuildFastTotalsZeroMemoryIncrease");
//
//   DO: ALREADY BEEN HANDLED IN OUR OPERATIONAL VERSION of BuildFastTotals -> sort our N-dimensional groups at program startup so that the longest dimension is first!  That way we can more efficiently walk through contiguous memory better in this function!
//
//   const size_t cDimensions = GET_DIMENSIONS(cCompilerDimensions, pFeatureGroup->GetCountDimensions());
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
//      const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
//      EBM_ASSERT(1 <= cDimensions);
//      do {
//         pCurrentIndexAndCountBinsInitialize->multipliedIndexCur = 0;
//         EBM_ASSERT(1 <= pFeatureGroupEntry->m_pFeature->m_cBins); // this function can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on (dimensions with 1 bin don't contribute anything since they always have the same value)
//         multipleTotalInitialize *= static_cast<ptrdiff_t>(pFeatureGroupEntry->m_pFeature->m_cBins);
//         pCurrentIndexAndCountBinsInitialize->multipleTotal = multipleTotalInitialize;
//         ++pFeatureGroupEntry;
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
//      const size_t cSamplesInBucket = pHistogramBucket->GetCountSamplesInBucket() + pPrevious->GetCountSamplesInBucket();
//      pHistogramBucket->m_cSamplesInBucket = cSamplesInBucket;
//      pPrevious->m_cSamplesInBucket = cSamplesInBucket;
//      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
//         const FloatEbmType sumGradients = pHistogramBucket->GetHistogramTargetEntry()[iVector].m_sumGradients + pPrevious->GetHistogramTargetEntry()[iVector].m_sumGradients;
//         pHistogramBucket->GetHistogramTargetEntry()[iVector].m_sumGradients = sumGradients;
//         pPrevious->GetHistogramTargetEntry()[iVector].m_sumGradients = sumGradients;
//
//         if(IsClassification(compilerLearningTypeOrCountTargetClasses)) {
//            const FloatEbmType sumHessians = pHistogramBucket->GetHistogramTargetEntry()[iVector].GetSumHessians() + pPrevious->GetHistogramTargetEntry()[iVector].GetSumHessians();
//            pHistogramBucket->GetHistogramTargetEntry()[iVector].SetSumHessians(sumHessians);
//            pPrevious->GetHistogramTargetEntry()[iVector].SetSumHessians(sumHessians);
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
//               goto skip_group;
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
//      skip_group:
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
//      TensorTotalsSumDebugSlow<compilerLearningTypeOrCountTargetClasses, cCompilerDimensions>(runtimeLearningTypeOrCountTargetClasses, pFeatureGroup, aHistogramBucketsDebugCopy, aiStart, aiLast, pDebugBucket);
//      EBM_ASSERT(pDebugBucket->GetCountSamplesInBucket() == pHistogramBucket->GetCountSamplesInBucket());
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




//template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t cCompilerDimensions>
//bool BoostMultiDimensionalPaulAlgorithm(ThreadStateBoosting<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pThreadState, const FeatureInternal * const pTargetFeature, SamplingSet const * const pTrainingSet, const FeatureGroup * const pFeatureGroup, SegmentedRegion<ActiveDataType, FloatEbmType> * const pSmallChangeToModelOverwriteSingleSamplingSet) {
//   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets = BinDataFrame<compilerLearningTypeOrCountTargetClasses>(pThreadState, pFeatureGroup, pTrainingSet, pTargetFeature);
//   if(UNLIKELY(nullptr == aHistogramBuckets)) {
//      return true;
//   }
//
//   BuildFastTotals(pTargetFeature, pFeatureGroup, aHistogramBuckets);
//
//   const size_t cDimensions = GET_DIMENSIONS(cCompilerDimensions, pFeatureGroup->GetCountDimensions());
//   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses);
//   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
//   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);
//
//   size_t aiStart[k_cDimensionsMax];
//   size_t aiLast[k_cDimensionsMax];
//
//   if(2 == cDimensions) {
//      DO: somehow avoid having a malloc here, either by allocating these when we allocate our big chunck of memory, or as part of pThreadState
//      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * aDynamicHistogramBuckets = static_cast<HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> *>(malloc(cBytesPerHistogramBucket * ));
//
//      const size_t cBinsDimension1 = pFeatureGroup->GetFeatureGroupEntries()[0].m_pFeature->m_cBins;
//      const size_t cBinsDimension2 = pFeatureGroup->GetFeatureGroupEntries()[1].m_pFeature->m_cBins;
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
//            TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, cCompilerDimensions>(runtimeLearningTypeOrCountTargetClasses, pFeatureGroup, aHistogramBuckets, aiStart, aiLast, pTotalsLowLow);
//
//            aiStart[0] = iBin1 + 1;
//            aiStart[1] = 0;
//            aiLast[0] = cBinsDimension1 - 1;
//            aiLast[1] = iBin2;
//            TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, cCompilerDimensions>(runtimeLearningTypeOrCountTargetClasses, pFeatureGroup, aHistogramBuckets, aiStart, aiLast, pTotalsHighLow);
//
//            aiStart[0] = 0;
//            aiStart[1] = iBin2 + 1;
//            aiLast[0] = iBin1;
//            aiLast[1] = cBinsDimension2 - 1;
//            TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, cCompilerDimensions>(runtimeLearningTypeOrCountTargetClasses, pFeatureGroup, aHistogramBuckets, aiStart, aiLast, pTotalsLowHigh);
//
//            aiStart[0] = iBin1 + 1;
//            aiStart[1] = iBin2 + 1;
//            aiLast[0] = cBinsDimension1 - 1;
//            aiLast[1] = cBinsDimension2 - 1;
//            TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, cCompilerDimensions>(runtimeLearningTypeOrCountTargetClasses, pFeatureGroup, aHistogramBuckets, aiStart, aiLast, pTotalsHighHigh);
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
//            splittingScore = CalculateRegionSplittingScore<compilerLearningTypeOrCountTargetClasses, cCompilerDimensions>(pTotalsTarget, pTotalsOther, runtimeLearningTypeOrCountTargetClasses);
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
//                     predictionTarget = ComputeSinglePartitionUpdate(pTotalsTarget->GetHistogramTargetEntry()[iVector].m_sumGradients, pTotalsTarget->GetCountSamplesInBucket());
//                     predictionOther = ComputeSinglePartitionUpdate(pTotalsOther->GetHistogramTargetEntry()[iVector].m_sumGradients, pTotalsOther->GetCountSamplesInBucket());
//                  } else {
//                     EBM_ASSERT(IS_CLASSIFICATION(compilerLearningTypeOrCountTargetClasses));
//                     // classification
//                     predictionTarget = ComputeSinglePartitionUpdate(pTotalsTarget->GetHistogramTargetEntry()[iVector].m_sumGradients, pTotalsTarget->GetHistogramTargetEntry()[iVector].GetSumHessians());
//                     predictionOther = ComputeSinglePartitionUpdate(pTotalsOther->GetHistogramTargetEntry()[iVector].m_sumGradients, pTotalsOther->GetHistogramTargetEntry()[iVector].GetSumHessians());
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
//            splittingScore = CalculateRegionSplittingScore<compilerLearningTypeOrCountTargetClasses, cCompilerDimensions>(pTotalsTarget, pTotalsOther, runtimeLearningTypeOrCountTargetClasses);
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
//                     predictionTarget = ComputeSinglePartitionUpdate(pTotalsTarget->GetHistogramTargetEntry()[iVector].m_sumGradients, pTotalsTarget->GetCountSamplesInBucket());
//                     predictionOther = ComputeSinglePartitionUpdate(pTotalsOther->GetHistogramTargetEntry()[iVector].m_sumGradients, pTotalsOther->GetCountSamplesInBucket());
//                  } else {
//                     EBM_ASSERT(IS_CLASSIFICATION(compilerLearningTypeOrCountTargetClasses));
//                     // classification
//                     predictionTarget = ComputeSinglePartitionUpdate(pTotalsTarget->GetHistogramTargetEntry()[iVector].m_sumGradients, pTotalsTarget->GetHistogramTargetEntry()[iVector].GetSumHessians());
//                     predictionOther = ComputeSinglePartitionUpdate(pTotalsOther->GetHistogramTargetEntry()[iVector].m_sumGradients, pTotalsOther->GetHistogramTargetEntry()[iVector].GetSumHessians());
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
//            splittingScore = CalculateRegionSplittingScore<compilerLearningTypeOrCountTargetClasses, cCompilerDimensions>(pTotalsTarget, pTotalsOther, runtimeLearningTypeOrCountTargetClasses);
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
//                     predictionTarget = ComputeSinglePartitionUpdate(pTotalsTarget->GetHistogramTargetEntry()[iVector].m_sumGradients, pTotalsTarget->GetCountSamplesInBucket());
//                     predictionOther = ComputeSinglePartitionUpdate(pTotalsOther->GetHistogramTargetEntry()[iVector].m_sumGradients, pTotalsOther->GetCountSamplesInBucket());
//                  } else {
//                     EBM_ASSERT(IS_CLASSIFICATION(compilerLearningTypeOrCountTargetClasses));
//                     // classification
//                     predictionTarget = ComputeSinglePartitionUpdate(pTotalsTarget->GetHistogramTargetEntry()[iVector].m_sumGradients, pTotalsTarget->GetHistogramTargetEntry()[iVector].GetSumHessians());
//                     predictionOther = ComputeSinglePartitionUpdate(pTotalsOther->GetHistogramTargetEntry()[iVector].m_sumGradients, pTotalsOther->GetHistogramTargetEntry()[iVector].GetSumHessians());
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
//            splittingScore = CalculateRegionSplittingScore<compilerLearningTypeOrCountTargetClasses, cCompilerDimensions>(pTotalsTarget, pTotalsOther, runtimeLearningTypeOrCountTargetClasses);
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
//                     predictionTarget = ComputeSinglePartitionUpdate(pTotalsTarget->GetHistogramTargetEntry()[iVector].m_sumGradients, pTotalsTarget->GetCountSamplesInBucket());
//                     predictionOther = ComputeSinglePartitionUpdate(pTotalsOther->GetHistogramTargetEntry()[iVector].m_sumGradients, pTotalsOther->GetCountSamplesInBucket());
//                  } else {
//                     EBM_ASSERT(IS_CLASSIFICATION(compilerLearningTypeOrCountTargetClasses));
//                     // classification
//                     predictionTarget = ComputeSinglePartitionUpdate(pTotalsTarget->GetHistogramTargetEntry()[iVector].m_sumGradients, pTotalsTarget->GetHistogramTargetEntry()[iVector].GetSumHessians());
//                     predictionOther = ComputeSinglePartitionUpdate(pTotalsOther->GetHistogramTargetEntry()[iVector].m_sumGradients, pTotalsOther->GetHistogramTargetEntry()[iVector].GetSumHessians());
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




