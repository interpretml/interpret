// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "common_cpp.hpp" // IsMultiplyError

#include "ebm_internal.hpp" // k_dynamicDimensions
#include "GradientPair.hpp"
#include "Bin.hpp"
#include "TensorTotalsSum.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME


// TODO: Implement a far more efficient boosting algorithm for higher dimensional interactions.  The algorithm works as follows:
//   - instead of first calculating the sums at each point for the hyper-dimensional region from the origin to each point, and then later
//     looking for splits, we can do both at the same time.  We know the total sums for the entire hyper-dimensional region, and as we're doing our summing
//     up, we can calcualte the gain at that point.  The catch is that we can only calculate the gain of the split between the hyper-dimensional region from
//     our current point to the origin, and the rest of the hyper-dimensional area.  We're using boosting though, so as long as we find some split that makes 
//     things a bit better, we can continue to improve the overall model, subject of course to overfitting.
//   - After we find the best single split from the origin to every point (and we've selected the best one), we can then go backwards from the point inside the
//     hyper-dimensional volume back towards the origin to select the best interior region vs the entire remaining hyper-dimensional volume.  Potentially we 
//     could at this point then also calculate the sub regions that would be created if we had made planar splits along both sides of each dimension.  
//   - Example: if we're splitting a cube, we find the best gain from the (0,0,0) to (5,5,5) gives the highest gain, then we go backwards and find that 
//     (5,5,5) -> (1,2,3) gives the best overall cube. We can then either take the cube as one region and the larger entire volume minus the cube as the 
//     other region, or we can separate the entire space into 27 cubes (9 cubes on each plane)
//   - We then need to generalize this algorithm because we don't only want splits from a single origin, we need to start from each origin.
//   - So, for an N dimensional region, we have 2^N ways to pick which dimensions we traverse in various orders.  So for 3 dimensions, there are 8 corners
//   - So for a 4 dimensional space, we would need to compute the gains for 2^4 times, and for a 16x16x16x16 volume, we would need to check 1,048,576 cells.
//     That seems doable for a 1GHz machine and if each cell consists of 16 bytes then it would be about 16 MB, which is cache fittable.  
//     Probably anything larger than 4 dimensions would dilute the data too far to make reasonable splits. We can go deeper if some of the features are 
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
//   - it might be the case that for pairs, we can get better results by using a traditional tree splitting algorithm (the existing one).  I should 
//     implement this algorithm above though regardless as it grows at less complexity than other algorithms, so it would be useful in any case.  
//     After it's implemented, we can compare the results against the existing pair computation code
//   - this pair splitting code should be templated for the numbrer of dimensions.  Nobody is really going to use it above 4-5 dimensions, 
//     but it's nice to have the option, but we don't want to implement 2,3,4,5 dimensional versions
//   - consider writing a pair specific version of this algorithm, also because pairs have different algorithms that could be the same
//   - once we have found our initial split, we should start from the split point and work backwards to the origin and find if there are any cubic splits 
//     that maximize gain
//   - we could in theory try and redo the first split (lookback) like we'll do in the mains
//   - each time we re-examine a sub region like this, or use lookback, we essentially need to re-do the algorithm, but we're only increasing the time 
//     by a small constant factor
//   - if we find it's benefitial to make full hyper-plane splits along all the dimensions that we find eg: if our split points are (1,2,3) -> (5, 6,7) then 
//     we would have 27 smaller cubes (9 per 2-D plane) then we just need to do a single full-ish sweep of the space to calcualte the totals for 
//     each of the volumes we have under consideration, but that too isn't too costly
// EXISTING ALGORITHM:
//   - our existing algorithm first determins the totals.  It benefits in that we can do this in a cache efficient way where we process the main tensor 
//     in order, although we do use side
//   - total N-1 planes that we also access per split.  This first step can be ignored since it costs much less than the next part
//   - after getting the totals, we do some kind of search for places to splits, but we need to calculate the total weights while we do so.  
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
//   - we could in theory re-implement the above more restricted algorithm that looks for volume splits from each dimension, but we'd then need 
//     either 2^N times more memory, or twice the memory and 2^(N/2), and during the search we'd be using cache inefficient memory access anyways, 
//     so it seems like there would be not benefit to doing a volume from each origin search vs the method above
//   - the other thing to note is that when training pairs after mains, any main split in the pair is suposed to have limited gain 
//     (and the limited gain is overfitting too), so we really need to look for groups of splits for gain if we use the algorithm of picking a split 
//     in one dimension, then picking a split in a different dimension, until all the dimension have been fulfilled, that's the simplest possible 
//     set of splits that divides the region in a way that splits all dimensions (without which we could reduce the interaction by at least 1 dimension)
//
//   - there are really 2 algorithms that I know of that we can do otherwise.  
//     1) The first one is a simple cross bar, where we choose a split point inside, then divide the area up into volumes from that point to 
//        each origin, which is the algorithm that we use for interaction detection.  At each point you need to calculate 2^N volumes, and each one of 
//        those takes 2^(N/2) operations
//   - 2) The algorithm we use for interaction splits.  We choose one dimension to split, but we don't calculate gain, we choose the next, ect, and then 
//        sweep each dimension.  We get 1 split along the main dimension, 2 splits on the second dimension, 4 splits on the third, etc.  The problem is 
//        that to be fair, we probably want to permute the order of our dimension splits, which means N! sweep variations
//        Possilby we could randomize the sweep directions and just do 1 each time, but that seems like it would be problematic, or maybe we 
//        choose a sweep direction per inner bag, and then we at least get variability. After we know our sweep direction, we need to visit each point.  
//        Since all dimensions are fixed and we just sweep one at a time, we have 2^N sweep tubes, and each step requires computing at least one side, 
//        so we pay 2^(N/2) operations
//    
//   - the cross bar sweep seems a little too close to our regional split while building appraoch, and it takes more work.  The 2^N operations 
//     and # of cells are common between that one and the add while sweep version, but the cross bar has an additional 2^(N/2) term vs N for 
//     the sum while working.  Sum while working would be much better for large numbers of dimensions
//   - the permuted solution has the same number of points to examine as the cross bar, and it has 2^N tubes to sweep vs 2^N volumes on each
//     side of the cross bar to examine, and calculating each costs region costs 2^(N/2), so the permuted solutoin takes N! times 
//     more time than the cross bar solution
//   - so the sweep while gain calculation takes less time to examine splits from each corner than the cross bar, all solutions have bad pipeline 
//     prediction fetch caracteristics and cache characteristics.
//   - the gain calculate while add has the benefit in that it requires no more memory other than the side planes that are needed for addition 
//     calculation anyways, so it's more memory efficient than either of the other two algorithms
//   
//   - SO, regardless as to whether the other algorithms are better, we'll probably want some form of the corner volume while adding to explore
//     higher dimensional spaces.  We can also give options for sweep splits for lower dimensions. 2-3 dimensional regions seem reasonable.  
//     Beyond that I'd say just do volume addition splits
//   - we should examine changing the interaction detection code to use our corner split solution since we exectute that algorithm 
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
// TODO: sort our N-dimensional groups at initialization so that the longest dimension is first!  That way we can more efficiently walk through contiguous memory better in this function!  After we determine the splits, we can undo the re-ordering for splitting the tensor, which has just a few cells, so will be efficient
template<ptrdiff_t cCompilerClasses, size_t cCompilerDimensions>
class TensorTotalsBuildInternal final {
public:

   TensorTotalsBuildInternal() = delete; // this is a static class.  Do not construct

   static void Func(
      const ptrdiff_t cRuntimeClasses,
      const size_t cRuntimeRealDimensions,
      const size_t * const acBins,
      BinBase * aAuxiliaryBinsBase,
      BinBase * const aBinsBase
#ifndef NDEBUG
      , BinBase * const aDebugCopyBinsBase
      , const BinBase * const pBinsEndDebug
#endif // NDEBUG
   ) {
      static constexpr bool bClassification = IsClassification(cCompilerClasses);
      static constexpr size_t cCompilerScores = GetCountScores(cCompilerClasses);

      struct FastTotalState {
         Bin<FloatBig, bClassification, cCompilerScores> * m_pDimensionalCur;
         Bin<FloatBig, bClassification, cCompilerScores> * m_pDimensionalWrap;
         Bin<FloatBig, bClassification, cCompilerScores> * m_pDimensionalFirst;
         size_t m_iCur;
         size_t m_cBins;
      };

      LOG_0(Trace_Verbose, "Entered BuildFastTotals");

      auto * pAuxiliaryBin = aAuxiliaryBinsBase->Specialize<FloatBig, bClassification, cCompilerScores>();

      auto * const aBins = aBinsBase->Specialize<FloatBig, bClassification, cCompilerScores>();

      const size_t cRealDimensions = GET_COUNT_DIMENSIONS(cCompilerDimensions, cRuntimeRealDimensions);
      EBM_ASSERT(1 <= cRealDimensions);

      const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, cRuntimeClasses);
      const size_t cScores = GetCountScores(cClasses);
      EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we're accessing allocated memory
      const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

      FastTotalState fastTotalState[k_cDimensionsMax];
      FastTotalState * pFastTotalStateInitialize = fastTotalState;
      {
         const size_t * pcBins = acBins;
         const size_t * const pcBinsEnd = &acBins[cRuntimeRealDimensions];
         size_t iAuxiliaryByte = cBytesPerBin;
         do {
            ASSERT_BIN_OK(cBytesPerBin, pAuxiliaryBin, pBinsEndDebug);

            const size_t cBins = *pcBins;
            // cBins can only be 0 if there are zero training and zero validation samples
            // we don't boost or allow interaction updates if there are zero training samples
            EBM_ASSERT(2 <= cBins);
            pFastTotalStateInitialize->m_iCur = 0;
            pFastTotalStateInitialize->m_cBins = cBins;

            pFastTotalStateInitialize->m_pDimensionalFirst = pAuxiliaryBin;
            pFastTotalStateInitialize->m_pDimensionalCur = pAuxiliaryBin;
            // when we exit, pAuxiliaryBin should be == to pBinsEndDebug, which is legal in C++ since it doesn't extend beyond 1 
            // item past the end of the array
            pAuxiliaryBin = IndexBin(pAuxiliaryBin, iAuxiliaryByte);

#ifndef NDEBUG
            if(&fastTotalState[cRealDimensions] == pFastTotalStateInitialize + 1) {
               // this is the last iteration, so pAuxiliaryBin should normally point to the memory address one byte past the legal buffer 
               // (normally pBinsEndDebug), BUT in rare cases we allocate more memory for the BinAuxiliaryBuildZone than we use in this 
               // function, so the only thing that we can guarantee is that we're equal or less than pBinsEndDebug
               EBM_ASSERT(pAuxiliaryBin <= pBinsEndDebug);
            } else {
               // if this isn't the last iteration, then we'll actually be using this memory, so the entire bin had better be useable
               EBM_ASSERT(IndexBin(pAuxiliaryBin, cBytesPerBin) <= pBinsEndDebug);
            }
            for(auto * pDimensionalCur = pFastTotalStateInitialize->m_pDimensionalCur;
               pAuxiliaryBin != pDimensionalCur; pDimensionalCur = IndexBin(pDimensionalCur, cBytesPerBin))
            {
               pDimensionalCur->AssertZero(cScores);
            }
#endif // NDEBUG

            // TODO : we don't need either the first or the wrap values since they are the next ones in the list.. we may need to populate one item past 
            // the end and make the list one larger
            pFastTotalStateInitialize->m_pDimensionalWrap = pAuxiliaryBin;

            iAuxiliaryByte *= cBins;
            ++pFastTotalStateInitialize;
            ++pcBins;
         } while(LIKELY(pcBinsEnd != pcBins));
      }
      EBM_ASSERT(pFastTotalStateInitialize == &fastTotalState[cRealDimensions]);

#ifndef NDEBUG

      auto * const pDebugBin = static_cast<Bin<FloatBig, bClassification, cCompilerScores> *>(malloc(cBytesPerBin));

      auto * aDebugCopyBins = aDebugCopyBinsBase->Specialize<FloatBig, bClassification, cCompilerScores>();

#endif //NDEBUG

      auto * pBin = aBins;

      while(true) {
         ASSERT_BIN_OK(cBytesPerBin, pBin, pBinsEndDebug);

         // TODO: on the 0th dimension, we could preserve the prev bin sum and avoid at least one read by
         //       eliminating one dimension loop iteration.  This would allow us to keep the 
         //       cSamples, weight, gradient and hessian in CPU regsiters. Keep the loop as a do loop however and 
         //       require that 2 <= cDimensions

         // TODO: if we have 3 dimensions (as an example), we don't have to keep storing the results of the Add
         //       function back into memory.  We can first load the original Bin into CPU registers and then
         //       add all the other dimensions, and at the end write it back to memory

         auto * pAddPrev = pBin;
         size_t iDimension = cRealDimensions;
         do {
            // TODO: Is there any benefit in making a pair/tripple specific version of this function
            //       This loop might be optimizable away (although the loop inside Add might prevent this
            //       loop from disapparing).  Even so, we could probably eliminate the auxillary memory
            //       for pairs and keep both the left bin and the left up bin on the stack and only read
            //       the up bin since that one will move.  The up left bin can be preserved in registers when reading
            //       the up bin. Perhaps we could optimize all dimensions to preserve these things though.

            --iDimension;
            auto * pAddTo = fastTotalState[iDimension].m_pDimensionalCur;
            pAddTo->Add(cScores, *pAddPrev);
            pAddPrev = pAddTo;
            pAddTo = IndexBin(pAddTo, cBytesPerBin);
            if(pAddTo == fastTotalState[iDimension].m_pDimensionalWrap) {
               pAddTo = fastTotalState[iDimension].m_pDimensionalFirst;
            }
            fastTotalState[iDimension].m_pDimensionalCur = pAddTo;
         } while(0 != iDimension);
         memcpy(pBin, pAddPrev, cBytesPerBin);

#ifndef NDEBUG
         if(nullptr != aDebugCopyBins && nullptr != pDebugBin) {
            size_t aiStart[k_cDimensionsMax];
            size_t aiLast[k_cDimensionsMax];
            for(size_t iDebugDimension = 0; iDebugDimension < cRealDimensions; ++iDebugDimension) {
               aiStart[iDebugDimension] = 0;
               aiLast[iDebugDimension] = fastTotalState[iDebugDimension].m_iCur;
            }
            TensorTotalsSumDebugSlow<bClassification>(
               cClasses,
               cRealDimensions,
               aiStart,
               aiLast,
               acBins,
               aDebugCopyBins->Downgrade(),
               *pDebugBin->Downgrade()
            );
            EBM_ASSERT(pDebugBin->GetCountSamples() == pBin->GetCountSamples());
         }
#endif // NDEBUG

         // we're walking through all bins, so just move to the next one in the flat array, 
         // with the knowledge that we'll figure out it's multi-dimenional index below
         pBin = IndexBin(pBin, cBytesPerBin);

         FastTotalState * pFastTotalState = &fastTotalState[0];
         while(true) {
            ++pFastTotalState->m_iCur;
            if(LIKELY(pFastTotalState->m_cBins != pFastTotalState->m_iCur)) {
               break;
            }
            pFastTotalState->m_iCur = 0;

            EBM_ASSERT(pFastTotalState->m_pDimensionalFirst == pFastTotalState->m_pDimensionalCur);

            auto * const pDimensionalFirst = pFastTotalState->m_pDimensionalFirst;
            const auto * const pDimensionalWrap = pFastTotalState->m_pDimensionalWrap;
            EBM_ASSERT(pDimensionalFirst != pDimensionalWrap);
            const size_t cBytesToZero = CountBytes(pDimensionalWrap, pDimensionalFirst);
            pDimensionalFirst->ZeroMem(cBytesToZero);
            ++pFastTotalState;

            if(UNLIKELY(pFastTotalStateInitialize == pFastTotalState)) {
#ifndef NDEBUG
               free(pDebugBin);
#endif // NDEBUG

               LOG_0(Trace_Verbose, "Exited BuildFastTotals");
               return;
            }
         }
      }
   }
};

template<ptrdiff_t cCompilerClasses, size_t cCompilerDimensionsPossible>
class TensorTotalsBuildDimensions final {
public:

   TensorTotalsBuildDimensions() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      const ptrdiff_t cRuntimeClasses,
      const size_t cRealDimensions,
      const size_t * const acBins,
      BinBase * aAuxiliaryBinsBase,
      BinBase * const aBinsBase
#ifndef NDEBUG
      , BinBase * const aDebugCopyBinsBase
      , const BinBase * const pBinsEndDebug
#endif // NDEBUG
   ) {
      static_assert(1 <= cCompilerDimensionsPossible, "can't have less than 1 dimension");
      static_assert(cCompilerDimensionsPossible <= k_cDimensionsMax, "can't have more than the max dimensions");

      EBM_ASSERT(1 <= cRealDimensions);
      EBM_ASSERT(cRealDimensions <= k_cDimensionsMax);
      if(cCompilerDimensionsPossible == cRealDimensions) {
         TensorTotalsBuildInternal<cCompilerClasses, cCompilerDimensionsPossible>::Func(
            cRuntimeClasses,
            cRealDimensions,
            acBins,
            aAuxiliaryBinsBase,
            aBinsBase
#ifndef NDEBUG
            , aDebugCopyBinsBase
            , pBinsEndDebug
#endif // NDEBUG
         );
      } else {
         TensorTotalsBuildDimensions<cCompilerClasses, cCompilerDimensionsPossible + 1>::Func(
            cRuntimeClasses,
            cRealDimensions,
            acBins,
            aAuxiliaryBinsBase,
            aBinsBase
#ifndef NDEBUG
            , aDebugCopyBinsBase
            , pBinsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<ptrdiff_t cCompilerClasses>
class TensorTotalsBuildDimensions<cCompilerClasses, k_cCompilerOptimizedCountDimensionsMax + 1> final {
public:

   TensorTotalsBuildDimensions() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      const ptrdiff_t cRuntimeClasses,
      const size_t cRealDimensions,
      const size_t * const acBins,
      BinBase * aAuxiliaryBinsBase,
      BinBase * const aBinsBase
#ifndef NDEBUG
      , BinBase * const aDebugCopyBinsBase
      , const BinBase * const pBinsEndDebug
#endif // NDEBUG
   ) {
      EBM_ASSERT(1 <= cRealDimensions);
      EBM_ASSERT(cRealDimensions <= k_cDimensionsMax);
      TensorTotalsBuildInternal<cCompilerClasses, k_dynamicDimensions>::Func(
         cRuntimeClasses,
         cRealDimensions,
         acBins,
         aAuxiliaryBinsBase,
         aBinsBase
#ifndef NDEBUG
         , aDebugCopyBinsBase
         , pBinsEndDebug
#endif // NDEBUG
      );
   }
};

template<ptrdiff_t cPossibleClasses>
class TensorTotalsBuildTarget final {
public:

   TensorTotalsBuildTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      const ptrdiff_t cRuntimeClasses,
      const size_t cRealDimensions,
      const size_t * const acBins,
      BinBase * aAuxiliaryBinsBase,
      BinBase * const aBinsBase
#ifndef NDEBUG
      , BinBase * const aDebugCopyBinsBase
      , const BinBase * const pBinsEndDebug
#endif // NDEBUG
   ) {
      static_assert(IsClassification(cPossibleClasses), "cPossibleClasses needs to be a classification");
      static_assert(cPossibleClasses <= k_cCompilerClassesMax, "We can't have this many items in a data pack.");

      EBM_ASSERT(IsClassification(cRuntimeClasses));
      EBM_ASSERT(cRuntimeClasses <= k_cCompilerClassesMax);

      if(cPossibleClasses == cRuntimeClasses) {
         TensorTotalsBuildDimensions<cPossibleClasses, 2>::Func(
            cRuntimeClasses,
            cRealDimensions,
            acBins,
            aAuxiliaryBinsBase,
            aBinsBase
#ifndef NDEBUG
            , aDebugCopyBinsBase
            , pBinsEndDebug
#endif // NDEBUG
         );
      } else {
         TensorTotalsBuildTarget<cPossibleClasses + 1>::Func(
            cRuntimeClasses,
            cRealDimensions,
            acBins,
            aAuxiliaryBinsBase,
            aBinsBase
#ifndef NDEBUG
            , aDebugCopyBinsBase
            , pBinsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<>
class TensorTotalsBuildTarget<k_cCompilerClassesMax + 1> final {
public:

   TensorTotalsBuildTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      const ptrdiff_t cRuntimeClasses,
      const size_t cRealDimensions,
      const size_t * const acBins,
      BinBase * aAuxiliaryBinsBase,
      BinBase * const aBinsBase
#ifndef NDEBUG
      , BinBase * const aDebugCopyBinsBase
      , const BinBase * const pBinsEndDebug
#endif // NDEBUG
   ) {
      static_assert(IsClassification(k_cCompilerClassesMax), "k_cCompilerClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(cRuntimeClasses));
      EBM_ASSERT(k_cCompilerClassesMax < cRuntimeClasses);

      TensorTotalsBuildDimensions<k_dynamicClassification, 2>::Func(
         cRuntimeClasses,
         cRealDimensions,
         acBins,
         aAuxiliaryBinsBase,
         aBinsBase
#ifndef NDEBUG
         , aDebugCopyBinsBase
         , pBinsEndDebug
#endif // NDEBUG
      );
   }
};

extern void TensorTotalsBuild(
   const ptrdiff_t cClasses,
   const size_t cRealDimensions,
   const size_t * const acBins,
   BinBase * aAuxiliaryBinsBase,
   BinBase * const aBinsBase
#ifndef NDEBUG
   , BinBase * const aDebugCopyBinsBase
   , const BinBase * const pBinsEndDebug
#endif // NDEBUG
) {
   if(IsClassification(cClasses)) {
      TensorTotalsBuildTarget<2>::Func(
         cClasses,
         cRealDimensions,
         acBins,
         aAuxiliaryBinsBase,
         aBinsBase
#ifndef NDEBUG
         , aDebugCopyBinsBase
         , pBinsEndDebug
#endif // NDEBUG
      );
   } else {
      EBM_ASSERT(IsRegression(cClasses));
      TensorTotalsBuildDimensions<k_regression, 2>::Func(
         cClasses,
         cRealDimensions,
         acBins,
         aAuxiliaryBinsBase,
         aBinsBase
#ifndef NDEBUG
         , aDebugCopyBinsBase
         , pBinsEndDebug
#endif // NDEBUG
      );
   }
}

// Boneyard of useful ideas below:


//template<ptrdiff_t cCompilerClasses, size_t cCompilerDimensions>
//bool BoostMultiDimensionalPaulAlgorithm(voidvoid * const pThreadState, const FeatureInternal * const pTargetFeature, InnerBag const * const pInnerBag, const Term * const pTerm, SegmentedRegion<ActiveDataType, FloatBig> * const pInnerTermUpdate) {
//   Bin<IsClassification(cCompilerClasses)> * const aBins = BinDataSet<cCompilerClasses>(pThreadState, pTerm, pInnerBag, pTargetFeature);
//   if(UNLIKELY(nullptr == aBins)) {
//      return true;
//   }
//
//   BuildFastTotals(pTargetFeature, pTerm, aBins);
//
//   const size_t cDimensions = GET_DIMENSIONS(cCompilerDimensions, pTerm->GetCountDimensions());
//   const size_t cScores = GET_VECTOR_LENGTH(cCompilerClasses, cRuntimeClasses);
//   EBM_ASSERT(!IsOverflowBinSize<IsClassification(cCompilerClasses)>(cScores)); // we're accessing allocated memory
//   const size_t cBytesPerBin = GetBinSize<IsClassification(cCompilerClasses)>(cScores);
//
//   size_t aiStart[k_cDimensionsMax];
//   size_t aiLast[k_cDimensionsMax];
//
//   if(2 == cDimensions) {
//      DO: somehow avoid having a malloc here, either by allocating these when we allocate our big chunck of memory, or as part of pThreadState
//      Bin<IsClassification(cCompilerClasses)> * aDynamicBins = static_cast<Bin<IsClassification(cCompilerClasses)> *>(malloc(cBytesPerBin * ));
//
//      const size_t cBinsDimension1 = pTerm->GetFeatures()[0].m_pFeature->m_cBins;
//      const size_t cBinsDimension2 = pTerm->GetFeatures()[1].m_pFeature->m_cBins;
//
//      FloatBig bestSplittingScore = FloatBig { -std::numeric_limits<FloatBig>::infinity() };
//
//      if(pInnerTermUpdate->SetCountSplits(0, 1)) {
//         free(aDynamicBins);
//#ifndef NDEBUG
//         free(aDebugCopyBins);
//#endif // NDEBUG
//         return true;
//      }
//      if(pInnerTermUpdate->SetCountSplits(1, 1)) {
//         free(aDynamicBins);
//#ifndef NDEBUG
//         free(aDebugCopyBins);
//#endif // NDEBUG
//         return true;
//      }
//      if(pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * 4)) {
//         free(aDynamicBins);
//#ifndef NDEBUG
//         free(aDebugCopyBins);
//#endif // NDEBUG
//         return true;
//      }
//
//      for(size_t iBin1 = 0; iBin1 < cBinsDimension1 - 1; ++iBin1) {
//         for(size_t iBin2 = 0; iBin2 < cBinsDimension2 - 1; ++iBin2) {
//            FloatBig splittingScore;
//
//            Bin<IsClassification(cCompilerClasses)> * pTotalsLowLow = IndexBin(cBytesPerBin, aDynamicBins, 0);
//            Bin<IsClassification(cCompilerClasses)> * pTotalsHighLow = IndexBin(cBytesPerBin, aDynamicBins, 1);
//            Bin<IsClassification(cCompilerClasses)> * pTotalsLowHigh = IndexBin(cBytesPerBin, aDynamicBins, 2);
//            Bin<IsClassification(cCompilerClasses)> * pTotalsHighHigh = IndexBin(cBytesPerBin, aDynamicBins, 3);
//
//            Bin<IsClassification(cCompilerClasses)> * pTotalsTarget = IndexBin(cBytesPerBin, aDynamicBins, 4);
//            Bin<IsClassification(cCompilerClasses)> * pTotalsOther = IndexBin(cBytesPerBin, aDynamicBins, 5);
//
//            aiStart[0] = 0;
//            aiStart[1] = 0;
//            aiLast[0] = iBin1;
//            aiLast[1] = iBin2;
//            TensorTotalsSum<cCompilerClasses, cCompilerDimensions>(cRuntimeClasses, pTerm, aBins, aiStart, aiLast, pTotalsLowLow);
//
//            aiStart[0] = iBin1 + 1;
//            aiStart[1] = 0;
//            aiLast[0] = cBinsDimension1 - 1;
//            aiLast[1] = iBin2;
//            TensorTotalsSum<cCompilerClasses, cCompilerDimensions>(cRuntimeClasses, pTerm, aBins, aiStart, aiLast, pTotalsHighLow);
//
//            aiStart[0] = 0;
//            aiStart[1] = iBin2 + 1;
//            aiLast[0] = iBin1;
//            aiLast[1] = cBinsDimension2 - 1;
//            TensorTotalsSum<cCompilerClasses, cCompilerDimensions>(cRuntimeClasses, pTerm, aBins, aiStart, aiLast, pTotalsLowHigh);
//
//            aiStart[0] = iBin1 + 1;
//            aiStart[1] = iBin2 + 1;
//            aiLast[0] = cBinsDimension1 - 1;
//            aiLast[1] = cBinsDimension2 - 1;
//            TensorTotalsSum<cCompilerClasses, cCompilerDimensions>(cRuntimeClasses, pTerm, aBins, aiStart, aiLast, pTotalsHighHigh);
//
//            // LOW LOW
//            pTotalsTarget->Zero(cRuntimeClasses);
//            pTotalsOther->Zero(cRuntimeClasses);
//
//            // MODIFY HERE
//            pTotalsTarget->Add(*pTotalsLowLow, cRuntimeClasses);
//            pTotalsOther->Add(*pTotalsHighLow, cRuntimeClasses);
//            pTotalsOther->Add(*pTotalsLowHigh, cRuntimeClasses);
//            pTotalsOther->Add(*pTotalsHighHigh, cRuntimeClasses);
//            
//            splittingScore = CalculateRegionSplittingScore<cCompilerClasses, cCompilerDimensions>(pTotalsTarget, pTotalsOther, cRuntimeClasses);
//            if(bestSplittingScore < splittingScore) {
//               bestSplittingScore = splittingScore;
//
//               pInnerTermUpdate->GetSplitPointer(0)[0] = iBin1;
//               pInnerTermUpdate->GetSplitPointer(1)[0] = iBin2;
//
//               for(size_t iScore = 0; iScore < cScores; ++iScore) {
//                  FloatBig predictionTarget;
//                  FloatBig predictionOther;
//
//                  if(IS_REGRESSION(cCompilerClasses)) {
//                     // regression
//                     predictionTarget = ComputeSinglePartitionUpdate(pTotalsTarget->GetGradientPairs()[iScore].m_sumGradients, pTotalsTarget->GetCountSamples());
//                     predictionOther = ComputeSinglePartitionUpdate(pTotalsOther->GetGradientPairs()[iScore].m_sumGradients, pTotalsOther->GetCountSamples());
//                  } else {
//                     EBM_ASSERT(IS_CLASSIFICATION(cCompilerClasses));
//                     // classification
//                     predictionTarget = ComputeSinglePartitionUpdate(pTotalsTarget->GetGradientPairs()[iScore].m_sumGradients, pTotalsTarget->GetGradientPairs()[iScore].GetHess());
//                     predictionOther = ComputeSinglePartitionUpdate(pTotalsOther->GetGradientPairs()[iScore].m_sumGradients, pTotalsOther->GetGradientPairs()[iScore].GetHess());
//                  }
//
//                  // MODIFY HERE
//                  pInnerTermUpdate->GetTensorScoresPointer()[0 * cScores + iScore] = predictionTarget;
//                  pInnerTermUpdate->GetTensorScoresPointer()[1 * cScores + iScore] = predictionOther;
//                  pInnerTermUpdate->GetTensorScoresPointer()[2 * cScores + iScore] = predictionOther;
//                  pInnerTermUpdate->GetTensorScoresPointer()[3 * cScores + iScore] = predictionOther;
//               }
//            }
//
//
//
//
//            // HIGH LOW
//            pTotalsTarget->Zero(cRuntimeClasses);
//            pTotalsOther->Zero(cRuntimeClasses);
//
//            // MODIFY HERE
//            pTotalsOther->Add(*pTotalsLowLow, cRuntimeClasses);
//            pTotalsTarget->Add(*pTotalsHighLow, cRuntimeClasses);
//            pTotalsOther->Add(*pTotalsLowHigh, cRuntimeClasses);
//            pTotalsOther->Add(*pTotalsHighHigh, cRuntimeClasses);
//
//            splittingScore = CalculateRegionSplittingScore<cCompilerClasses, cCompilerDimensions>(pTotalsTarget, pTotalsOther, cRuntimeClasses);
//            if(bestSplittingScore < splittingScore) {
//               bestSplittingScore = splittingScore;
//
//               pInnerTermUpdate->GetSplitPointer(0)[0] = iBin1;
//               pInnerTermUpdate->GetSplitPointer(1)[0] = iBin2;
//
//               for(size_t iScore = 0; iScore < cScores; ++iScore) {
//                  FloatBig predictionTarget;
//                  FloatBig predictionOther;
//
//                  if(IS_REGRESSION(cCompilerClasses)) {
//                     // regression
//                     predictionTarget = ComputeSinglePartitionUpdate(pTotalsTarget->GetGradientPairs()[iScore].m_sumGradients, pTotalsTarget->GetCountSamples());
//                     predictionOther = ComputeSinglePartitionUpdate(pTotalsOther->GetGradientPairs()[iScore].m_sumGradients, pTotalsOther->GetCountSamples());
//                  } else {
//                     EBM_ASSERT(IS_CLASSIFICATION(cCompilerClasses));
//                     // classification
//                     predictionTarget = ComputeSinglePartitionUpdate(pTotalsTarget->GetGradientPairs()[iScore].m_sumGradients, pTotalsTarget->GetGradientPairs()[iScore].GetHess());
//                     predictionOther = ComputeSinglePartitionUpdate(pTotalsOther->GetGradientPairs()[iScore].m_sumGradients, pTotalsOther->GetGradientPairs()[iScore].GetHess());
//                  }
//
//                  // MODIFY HERE
//                  pInnerTermUpdate->GetTensorScoresPointer()[0 * cScores + iScore] = predictionOther;
//                  pInnerTermUpdate->GetTensorScoresPointer()[1 * cScores + iScore] = predictionTarget;
//                  pInnerTermUpdate->GetTensorScoresPointer()[2 * cScores + iScore] = predictionOther;
//                  pInnerTermUpdate->GetTensorScoresPointer()[3 * cScores + iScore] = predictionOther;
//               }
//            }
//
//
//
//
//            // LOW HIGH
//            pTotalsTarget->Zero(cRuntimeClasses);
//            pTotalsOther->Zero(cRuntimeClasses);
//
//            // MODIFY HERE
//            pTotalsOther->Add(*pTotalsLowLow, cRuntimeClasses);
//            pTotalsOther->Add(*pTotalsHighLow, cRuntimeClasses);
//            pTotalsTarget->Add(*pTotalsLowHigh, cRuntimeClasses);
//            pTotalsOther->Add(*pTotalsHighHigh, cRuntimeClasses);
//
//            splittingScore = CalculateRegionSplittingScore<cCompilerClasses, cCompilerDimensions>(pTotalsTarget, pTotalsOther, cRuntimeClasses);
//            if(bestSplittingScore < splittingScore) {
//               bestSplittingScore = splittingScore;
//
//               pInnerTermUpdate->GetSplitPointer(0)[0] = iBin1;
//               pInnerTermUpdate->GetSplitPointer(1)[0] = iBin2;
//
//               for(size_t iScore = 0; iScore < cScores; ++iScore) {
//                  FloatBig predictionTarget;
//                  FloatBig predictionOther;
//
//                  if(IS_REGRESSION(cCompilerClasses)) {
//                     // regression
//                     predictionTarget = ComputeSinglePartitionUpdate(pTotalsTarget->GetGradientPairs()[iScore].m_sumGradients, pTotalsTarget->GetCountSamples());
//                     predictionOther = ComputeSinglePartitionUpdate(pTotalsOther->GetGradientPairs()[iScore].m_sumGradients, pTotalsOther->GetCountSamples());
//                  } else {
//                     EBM_ASSERT(IS_CLASSIFICATION(cCompilerClasses));
//                     // classification
//                     predictionTarget = ComputeSinglePartitionUpdate(pTotalsTarget->GetGradientPairs()[iScore].m_sumGradients, pTotalsTarget->GetGradientPairs()[iScore].GetHess());
//                     predictionOther = ComputeSinglePartitionUpdate(pTotalsOther->GetGradientPairs()[iScore].m_sumGradients, pTotalsOther->GetGradientPairs()[iScore].GetHess());
//                  }
//
//                  // MODIFY HERE
//                  pInnerTermUpdate->GetTensorScoresPointer()[0 * cScores + iScore] = predictionOther;
//                  pInnerTermUpdate->GetTensorScoresPointer()[1 * cScores + iScore] = predictionOther;
//                  pInnerTermUpdate->GetTensorScoresPointer()[2 * cScores + iScore] = predictionTarget;
//                  pInnerTermUpdate->GetTensorScoresPointer()[3 * cScores + iScore] = predictionOther;
//               }
//            }
//
//
//
//            // HIGH HIGH
//            pTotalsTarget->Zero(cRuntimeClasses);
//            pTotalsOther->Zero(cRuntimeClasses);
//
//            // MODIFY HERE
//            pTotalsOther->Add(*pTotalsLowLow, cRuntimeClasses);
//            pTotalsOther->Add(*pTotalsHighLow, cRuntimeClasses);
//            pTotalsOther->Add(*pTotalsLowHigh, cRuntimeClasses);
//            pTotalsTarget->Add(*pTotalsHighHigh, cRuntimeClasses);
//
//            splittingScore = CalculateRegionSplittingScore<cCompilerClasses, cCompilerDimensions>(pTotalsTarget, pTotalsOther, cRuntimeClasses);
//            if(bestSplittingScore < splittingScore) {
//               bestSplittingScore = splittingScore;
//
//               pInnerTermUpdate->GetSplitPointer(0)[0] = iBin1;
//               pInnerTermUpdate->GetSplitPointer(1)[0] = iBin2;
//
//               for(size_t iScore = 0; iScore < cScores; ++iScore) {
//                  FloatBig predictionTarget;
//                  FloatBig predictionOther;
//
//                  if(IS_REGRESSION(cCompilerClasses)) {
//                     // regression
//                     predictionTarget = ComputeSinglePartitionUpdate(pTotalsTarget->GetGradientPairs()[iScore].m_sumGradients, pTotalsTarget->GetCountSamples());
//                     predictionOther = ComputeSinglePartitionUpdate(pTotalsOther->GetGradientPairs()[iScore].m_sumGradients, pTotalsOther->GetCountSamples());
//                  } else {
//                     EBM_ASSERT(IS_CLASSIFICATION(cCompilerClasses));
//                     // classification
//                     predictionTarget = ComputeSinglePartitionUpdate(pTotalsTarget->GetGradientPairs()[iScore].m_sumGradients, pTotalsTarget->GetGradientPairs()[iScore].GetHess());
//                     predictionOther = ComputeSinglePartitionUpdate(pTotalsOther->GetGradientPairs()[iScore].m_sumGradients, pTotalsOther->GetGradientPairs()[iScore].GetHess());
//                  }
//
//                  // MODIFY HERE
//                  pInnerTermUpdate->GetTensorScoresPointer()[0 * cScores + iScore] = predictionOther;
//                  pInnerTermUpdate->GetTensorScoresPointer()[1 * cScores + iScore] = predictionOther;
//                  pInnerTermUpdate->GetTensorScoresPointer()[2 * cScores + iScore] = predictionOther;
//                  pInnerTermUpdate->GetTensorScoresPointer()[3 * cScores + iScore] = predictionTarget;
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
//      free(aDynamicBins);
//   } else {
//      DO: handle this better
//#ifndef NDEBUG
//      EBM_ASSERT(false); // we only support pairs currently
//      free(aDebugCopyBins);
//#endif // NDEBUG
//      return true;
//   }
//#ifndef NDEBUG
//   free(aDebugCopyBins);
//#endif // NDEBUG
//   return false;
//}

} // DEFINED_ZONE_NAME
