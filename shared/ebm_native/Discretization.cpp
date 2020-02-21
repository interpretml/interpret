// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <algorithm> // sort
#include <cmath> // std::round
#include <vector>
#include <stdio.h>
#include <cmath>

#include "ebm_native.h"

#include "EbmInternal.h"
// very independent includes
#include "Logging.h" // EBM_ASSERT & LOG
#include "RandomStream.h"

constexpr unsigned int k_MiddleSplittingRange = 0x0;
constexpr unsigned int k_FirstSplittingRange = 0x1;
constexpr unsigned int k_LastSplittingRange = 0x2;

struct SplittingRange {
   // we divide the space into long segments of unsplittable equal values separated by zero or more items that we call SplittingRanges.  SplittingRange are
   // where we put the splitting points.  If there are long unsplittable segments at either the start or the end, we can't put split points at those ends, 
   // so these are left out.  SplittingRange can always have a split point, even in the length is zero, since they sit between long ranges.  If there
   // are non-equal values at either end, but not enough items to put a split point, we put those values at the ends into the Unsplittable category

   FloatEbmType * m_pSplittableValuesStart;
   size_t         m_cSplittableItems; // this can be zero
   size_t         m_cUnsplittablePriorItems;
   size_t         m_cUnsplittableSubsequentItems;

   size_t         m_cUnsplittableEitherSideMax;
   size_t         m_cUnsplittableEitherSideMin;

   size_t         m_cSplitsAssigned;
   unsigned int   m_flags;
};

// PK VERIFIED!
INLINE_RELEASE void SortSplittingRangesByCountItemsDescending(
   RandomStream * const pRandomStream, 
   const size_t cSplittingRanges, 
   SplittingRange ** const apSplittingRange
) {
   EBM_ASSERT(1 <= cSplittingRanges);

   // sort in descending order for m_cSplittableItems
   //
   // But some items can have the same primary sort key, so sort secondarily on the pointer to the original object, thus putting them secondarily in index
   // order, which is guaranteed to be a unique ordering. We'll later randomize the order of items that have the same primary sort index, BUT we want our 
   // initial sort order to be replicatable with the same random seed, so we need the initial sort to be stable.
   std::sort(apSplittingRange, apSplittingRange + cSplittingRanges, [](SplittingRange *& pSplittingRange1, SplittingRange *& pSplittingRange2) {
      if(PREDICTABLE(pSplittingRange1->m_cSplittableItems == pSplittingRange2->m_cSplittableItems)) {
         return UNPREDICTABLE(pSplittingRange1->m_pSplittableValuesStart > pSplittingRange2->m_pSplittableValuesStart);
      } else {
         return UNPREDICTABLE(pSplittingRange1->m_cSplittableItems > pSplittingRange2->m_cSplittableItems);
      }
      });

   // find sections that have the same number of items and randomly shuffle the sections with equal numbers of items so that there is no directional preference
   size_t iStartEqualLengthRange = 0;
   size_t cItems = apSplittingRange[0]->m_cSplittableItems;
   for(size_t i = 1; LIKELY(i < cSplittingRanges); ++i) {
      const size_t cNewItems = apSplittingRange[i]->m_cSplittableItems;
      if(PREDICTABLE(cItems != cNewItems)) {
         // we have a real range
         size_t cRemainingItems = i - iStartEqualLengthRange;
         EBM_ASSERT(1 <= cRemainingItems);
         while(PREDICTABLE(1 != cRemainingItems)) {
            const size_t iSwap = pRandomStream->Next(cRemainingItems);
            SplittingRange * pTmp = apSplittingRange[iStartEqualLengthRange];
            apSplittingRange[iStartEqualLengthRange] = apSplittingRange[iStartEqualLengthRange + iSwap];
            apSplittingRange[iStartEqualLengthRange + iSwap] = pTmp;
            ++iStartEqualLengthRange;
            --cRemainingItems;
         }
         iStartEqualLengthRange = i;
         cItems = cNewItems;
      }
   }
   size_t cRemainingItemsOuter = cSplittingRanges - iStartEqualLengthRange;
   EBM_ASSERT(1 <= cRemainingItemsOuter);
   while(PREDICTABLE(1 != cRemainingItemsOuter)) {
      const size_t iSwap = pRandomStream->Next(cRemainingItemsOuter);
      SplittingRange * pTmp = apSplittingRange[iStartEqualLengthRange];
      apSplittingRange[iStartEqualLengthRange] = apSplittingRange[iStartEqualLengthRange + iSwap];
      apSplittingRange[iStartEqualLengthRange + iSwap] = pTmp;
      ++iStartEqualLengthRange;
      --cRemainingItemsOuter;
   }
}

// PK VERIFIED!
// TODO: Is this being used?
INLINE_RELEASE void SortSplittingRangesByCountItemsAscending(
   RandomStream * const pRandomStream, 
   const size_t cSplittingRanges, 
   SplittingRange ** const apSplittingRange
) {
   EBM_ASSERT(1 <= cSplittingRanges);

   // sort in ascending order for m_cSplittableItems
   //
   // But some items can have the same primary sort key, so sort secondarily on the pointer to the original object, thus putting them secondarily in index
   // order, which is guaranteed to be a unique ordering. We'll later randomize the order of items that have the same primary sort index, BUT we want our 
   // initial sort order to be replicatable with the same random seed, so we need the initial sort to be stable.
   std::sort(apSplittingRange, apSplittingRange + cSplittingRanges, [](SplittingRange * & pSplittingRange1, SplittingRange * & pSplittingRange2) {
      if(PREDICTABLE(pSplittingRange1->m_cSplittableItems == pSplittingRange2->m_cSplittableItems)) {
         return UNPREDICTABLE(pSplittingRange1->m_pSplittableValuesStart < pSplittingRange2->m_pSplittableValuesStart);
      } else {
         return UNPREDICTABLE(pSplittingRange1->m_cSplittableItems < pSplittingRange2->m_cSplittableItems);
      }
      });

   // find sections that have the same number of items and randomly shuffle the sections with equal numbers of items so that there is no directional preference
   size_t iStartEqualLengthRange = 0;
   size_t cItems = apSplittingRange[0]->m_cSplittableItems;
   for(size_t i = 1; LIKELY(i < cSplittingRanges); ++i) {
      const size_t cNewItems = apSplittingRange[i]->m_cSplittableItems;
      if(PREDICTABLE(cItems != cNewItems)) {
         // we have a real range
         size_t cRemainingItems = i - iStartEqualLengthRange;
         EBM_ASSERT(1 <= cRemainingItems);
         while(PREDICTABLE(1 != cRemainingItems)) {
            const size_t iSwap = pRandomStream->Next(cRemainingItems);
            SplittingRange * pTmp = apSplittingRange[iStartEqualLengthRange];
            apSplittingRange[iStartEqualLengthRange] = apSplittingRange[iStartEqualLengthRange + iSwap];
            apSplittingRange[iStartEqualLengthRange + iSwap] = pTmp;
            ++iStartEqualLengthRange;
            --cRemainingItems;
         }
         iStartEqualLengthRange = i;
         cItems = cNewItems;
      }
   }
   size_t cRemainingItemsOuter = cSplittingRanges - iStartEqualLengthRange;
   EBM_ASSERT(1 <= cRemainingItemsOuter);
   while(PREDICTABLE(1 != cRemainingItemsOuter)) {
      const size_t iSwap = pRandomStream->Next(cRemainingItemsOuter);
      SplittingRange * pTmp = apSplittingRange[iStartEqualLengthRange];
      apSplittingRange[iStartEqualLengthRange] = apSplittingRange[iStartEqualLengthRange + iSwap];
      apSplittingRange[iStartEqualLengthRange + iSwap] = pTmp;
      ++iStartEqualLengthRange;
      --cRemainingItemsOuter;
   }
}

// PK VERIFIED!
// TODO: Is this being used?
INLINE_RELEASE void SortSplittingRangesByUnsplittableDescending(
   RandomStream * const pRandomStream, 
   const size_t cSplittingRanges, 
   SplittingRange ** const apSplittingRange
) {
   EBM_ASSERT(1 <= cSplittingRanges);

   // sort in descending order for m_cUnsplittableEitherSideMax, m_cUnsplittableEitherSideMin
   //
   // But some items can have the same primary sort key, so sort secondarily on the pointer to the original object, thus putting them secondarily in index
   // order, which is guaranteed to be a unique ordering. We'll later randomize the order of items that have the same primary sort index, BUT we want our 
   // initial sort order to be replicatable with the same random seed, so we need the initial sort to be stable.
   std::sort(apSplittingRange, apSplittingRange + cSplittingRanges, [](SplittingRange * & pSplittingRange1, SplittingRange * & pSplittingRange2) {
      if(PREDICTABLE(pSplittingRange1->m_cUnsplittableEitherSideMax == pSplittingRange2->m_cUnsplittableEitherSideMax)) {
         if(PREDICTABLE(pSplittingRange1->m_cUnsplittableEitherSideMin == pSplittingRange2->m_cUnsplittableEitherSideMin)) {
            return UNPREDICTABLE(pSplittingRange1->m_pSplittableValuesStart > pSplittingRange2->m_pSplittableValuesStart);
         } else {
            return UNPREDICTABLE(pSplittingRange1->m_cUnsplittableEitherSideMin > pSplittingRange2->m_cUnsplittableEitherSideMin);
         }
      } else {
         return UNPREDICTABLE(pSplittingRange1->m_cUnsplittableEitherSideMax > pSplittingRange2->m_cUnsplittableEitherSideMax);
      }
   });

   // find sections that have the same number of items and randomly shuffle the sections with equal numbers of items so that there is no directional preference
   size_t iStartEqualLengthRange = 0;
   size_t cItemsMax = apSplittingRange[0]->m_cUnsplittableEitherSideMax;
   size_t cItemsMin = apSplittingRange[0]->m_cUnsplittableEitherSideMin;
   for(size_t i = 1; LIKELY(i < cSplittingRanges); ++i) {
      const size_t cNewItemsMax = apSplittingRange[i]->m_cUnsplittableEitherSideMax;
      const size_t cNewItemsMin = apSplittingRange[i]->m_cUnsplittableEitherSideMin;
      if(PREDICTABLE(PREDICTABLE(cNewItemsMax != cItemsMax) || PREDICTABLE(cNewItemsMin != cItemsMin))) {
         // we have a real range
         size_t cRemainingItems = i - iStartEqualLengthRange;
         EBM_ASSERT(1 <= cRemainingItems);
         while(PREDICTABLE(1 != cRemainingItems)) {
            const size_t iSwap = pRandomStream->Next(cRemainingItems);
            SplittingRange * pTmp = apSplittingRange[iStartEqualLengthRange];
            apSplittingRange[iStartEqualLengthRange] = apSplittingRange[iStartEqualLengthRange + iSwap];
            apSplittingRange[iStartEqualLengthRange + iSwap] = pTmp;
            ++iStartEqualLengthRange;
            --cRemainingItems;
         }
         iStartEqualLengthRange = i;
         cItemsMax = cNewItemsMax;
         cItemsMin = cNewItemsMin;
      }
   }
   size_t cRemainingItemsOuter = cSplittingRanges - iStartEqualLengthRange;
   EBM_ASSERT(1 <= cRemainingItemsOuter);
   while(PREDICTABLE(1 != cRemainingItemsOuter)) {
      const size_t iSwap = pRandomStream->Next(cRemainingItemsOuter);
      SplittingRange * pTmp = apSplittingRange[iStartEqualLengthRange];
      apSplittingRange[iStartEqualLengthRange] = apSplittingRange[iStartEqualLengthRange + iSwap];
      apSplittingRange[iStartEqualLengthRange + iSwap] = pTmp;
      ++iStartEqualLengthRange;
      --cRemainingItemsOuter;
   }
}

INLINE_RELEASE size_t PossiblyRemoveBinForMissing(const bool bMissing, const IntEbmType countMaximumBins) {
   EBM_ASSERT(IntEbmType { 2 } <= countMaximumBins);
   size_t cMaximumBins = static_cast<size_t>(countMaximumBins);
   if(PREDICTABLE(bMissing)) {
      // if there is a missing value, then we use 0 for the missing value bin, and bump up all other values by 1.  This creates a semi-problem
      // if the number of bins was specified as a power of two like 256, because we now have 257 possible values, and instead of consuming 8
      // bits per value, we're consuming 9.  If we're told to have a maximum of a power of two bins though, in most cases it won't hurt to
      // have one less bin so that we consume less data.  Our countMaximumBins is just a maximum afterall, so we can choose to have less bins.
      // BUT, if the user requests 8 bins or less, then don't reduce the number of bins since then we'll be changing the bin size significantly

      size_t cBits = ((~size_t { 0 }) >> 1) + size_t { 1 };
      do {
         // if cMaximumBins is a power of two equal to or greater than 16, then reduce the number of bins (it's a maximum after all) to one less so that
         // it's more compressible.  If we have 256 bins, we really want 255 bins and 0 to be the missing value, using 256 values and 1 byte of storage
         // some powers of two aren't compressible, like 2^34, which needs to fit into a 64 bit storage, but we don't want to take a dependency
         // on the size of the storage system, which is system dependent, so we just exclude all powers of two
         if(UNLIKELY(cBits == cMaximumBins)) {
            --cMaximumBins;
            break;
         }
         cBits >>= 1;
         // don't allow shrinkage below 16 bins (8 is the first power of two below 16).  By the time we reach 8 bins, we don't want to reduce this
         // by a complete bin.  We can just use the extra bit for the missing bin
         // if we had shrunk down to 7 bits for non-missing, we would have been able to fit in 16 items per data item instead of 21 for 64 bit systems
      } while(UNLIKELY(0x8 != cBits));
   }
   return cMaximumBins;
}

INLINE_RELEASE size_t GetAvgLength(const size_t cInstances, const size_t cMaximumBins, const size_t cMinimumInstancesPerBin) {
   EBM_ASSERT(size_t { 1 } <= cInstances);
   EBM_ASSERT(size_t { 2 } <= cMaximumBins); // if there is just one bin, then you can't have splits, so we exit earlier
   EBM_ASSERT(size_t { 1 } <= cMinimumInstancesPerBin);

   // SplittingRanges are ranges of numbers that we have the guaranteed option of making at least one split within.
   // if there is only one SplittingRange, then we have no choice other than make cuts within the one SplittingRange that we're given
   // if there are multiple SplittingRanges, then every SplittingRanges borders at least one long range of equal values which are unsplittable.
   // cuts are a limited resource, so we want to spend them wisely.  If we have N cuts to give out, we'll first want to ensure that we get a cut
   // within each possible SplittingRange, since these things always border long ranges of unsplittable values.
   //
   // BUT, what happens if we have N SplittingRange, but only N-1 cuts to give out.  In that case we would have to make difficult decisions about where
   // to put the cuts
   //
   // To avoid the bad scenario of having to figure out which SplittingRange won't get a cut, we instead ensure that we can never have more SplittingRanges
   // than we have cuts.  This way every SplittingRanges is guaranteed to have at least 1 cut.
   // 
   // If our avgLength is the ceiling of cInstances / cMaximumBins, then we get this guarantee
   // but std::ceil works on floating point numbers, and it is inexact, especially if cInstances is above the point where floating point numbers can't
   // represent all integer values anymore (at around 2^51 I think)
   // so, instead of taking the std::ceil, we take the floor instead by just converting it to size_t, then we increment the avgLength until we
   // get our guarantee using integer math.  This gives us a true guarantee that we'll have sufficient cuts to give each SplittingRange at least one cut

   // Example of a bad situation if we took the rounded average of cInstances / cMaximumBins:
   // 20 == cInstances, 9 == cMaximumBins (so 8 cuts).  20 / 9 = 2.22222222222.  std::round(2.222222222) = 2.  So avgLength would be 2 if we rounded 20 / 9
   // but if our data is:
   // 0,0|1,1|2,2|3,3|4,4|5,5|6,6|7,7|8,8|9,9
   // then we get 9 SplittingRanges, but we only have 8 cuts to distribute.  And then we get to somehow choose which SplittingRange gets 0 cuts.
   // a better choice would have been to make avgLength 3 instead, so the ceiling.  Then we'd be guaranteed to have 8 or less SplittingRanges

   // our algorithm has the option of not putting cut points in the first and last SplittingRanges, since they could be cMinimumInstancesPerBin long
   // and have a long set of equal values only on one side, which means that a cut there isn't absolutely required.  We still need to take the ceiling
   // for the avgLength though since it's possible to create arbitrarily high number of missing bins.  We have a test that creates 3 missing bins, thereby
   // testing for the case that we don't give the first and last SplittingRanges an initial cut.  In this case, we're still missing a cut for one of the
   // long ranges that we can't fullfil.

   size_t avgLength = static_cast<size_t>(static_cast<FloatEbmType>(cInstances) / static_cast<FloatEbmType>(cMaximumBins));
   avgLength = UNPREDICTABLE(avgLength < cMinimumInstancesPerBin) ? cMinimumInstancesPerBin : avgLength;
   while(true) {
      if(UNLIKELY(IsMultiplyError(avgLength, cMaximumBins))) {
         // cInstances isn't an overflow (we checked when we entered), so if we've reached an overflow in the multiplication, 
         // then our multiplication result must be larger than cInstances, even though we can't perform it, so we're good
         break;
      }
      if(PREDICTABLE(cInstances <= avgLength * cMaximumBins)) {
         break;
      }
      ++avgLength;
   }
   return avgLength;
}

INLINE_RELEASE size_t RemoveMissingValues(const size_t cInstances, FloatEbmType * const aValues) {
   FloatEbmType * pCopyFrom = aValues;
   const FloatEbmType * const pValuesEnd = aValues + cInstances;
   do {
      FloatEbmType val = *pCopyFrom;
      if(UNLIKELY(std::isnan(val))) {
         FloatEbmType * pCopyTo = pCopyFrom;
         goto skip_val;
         do {
            val = *pCopyFrom;
            if(PREDICTABLE(!std::isnan(val))) {
               *pCopyTo = val;
               ++pCopyTo;
            }
         skip_val:
            ++pCopyFrom;
         } while(LIKELY(pValuesEnd != pCopyFrom));
         const size_t cInstancesWithoutMissing = pCopyTo - aValues;
         EBM_ASSERT(cInstancesWithoutMissing < cInstances);
         return cInstancesWithoutMissing;
      }
      ++pCopyFrom;
   } while(LIKELY(pValuesEnd != pCopyFrom));
   return cInstances;
}

INLINE_RELEASE size_t CountSplittingRanges(
   const size_t cInstances,
   const FloatEbmType * const aSingleFeatureValues,
   const size_t avgLength, 
   const size_t cMinimumInstancesPerBin
) {
   EBM_ASSERT(1 <= cInstances);
   EBM_ASSERT(nullptr != aSingleFeatureValues);
   EBM_ASSERT(1 <= avgLength);
   EBM_ASSERT(1 <= cMinimumInstancesPerBin);

   if(cInstances < (cMinimumInstancesPerBin << 1)) {
      // we can't make any cuts if we have less than 2 * cMinimumInstancesPerBin instances, 
      // since we need at least cMinimumInstancesPerBin instances on either side of the cut point
      return 0;
   }
   FloatEbmType rangeValue = *aSingleFeatureValues;
   const FloatEbmType * pSplittableValuesStart = aSingleFeatureValues;
   const FloatEbmType * pStartEqualRange = aSingleFeatureValues;
   const FloatEbmType * pScan = aSingleFeatureValues + 1;
   const FloatEbmType * const pValuesEnd = aSingleFeatureValues + cInstances;
   size_t cSplittingRanges = 0;
   while(pValuesEnd != pScan) {
      const FloatEbmType val = *pScan;
      if(val != rangeValue) {
         size_t cEqualRangeItems = pScan - pStartEqualRange;
         if(avgLength <= cEqualRangeItems) {
            if(aSingleFeatureValues != pSplittableValuesStart || cMinimumInstancesPerBin <= static_cast<size_t>(pStartEqualRange - pSplittableValuesStart)) {
               ++cSplittingRanges;
            }
            pSplittableValuesStart = pScan;
         }
         rangeValue = val;
         pStartEqualRange = pScan;
      }
      ++pScan;
   }
   if(aSingleFeatureValues == pSplittableValuesStart) {
      EBM_ASSERT(0 == cSplittingRanges);

      // we're still on the first splitting range.  We need to make sure that there is at least one possible cut
      // if we require 3 items for a cut, a problematic range like 0 1 3 3 4 5 could look ok, but we can't cut it in the middle!
      const FloatEbmType * pCheckForSplitPoint = aSingleFeatureValues + cMinimumInstancesPerBin;
      EBM_ASSERT(pCheckForSplitPoint <= pValuesEnd);
      const FloatEbmType * pCheckForSplitPointLast = pValuesEnd - cMinimumInstancesPerBin;
      EBM_ASSERT(aSingleFeatureValues <= pCheckForSplitPointLast);
      EBM_ASSERT(aSingleFeatureValues < pCheckForSplitPoint);
      FloatEbmType checkValue = *(pCheckForSplitPoint - 1);
      while(pCheckForSplitPoint <= pCheckForSplitPointLast) {
         if(checkValue != *pCheckForSplitPoint) {
            return 1;
         }
         ++pCheckForSplitPoint;
      }
      // there's no possible place to split, so return
      return 0;
   } else {
      const size_t cItemsLast = static_cast<size_t>(pValuesEnd - pSplittableValuesStart);
      if(cMinimumInstancesPerBin <= cItemsLast) {
         ++cSplittingRanges;
      }
      return cSplittingRanges;
   }
}

INLINE_RELEASE void FillSplittingRangeBasics(
   const size_t cInstances,
   FloatEbmType * const aSingleFeatureValues,
   const size_t avgLength,
   const size_t cMinimumInstancesPerBin,
   const size_t cSplittingRanges,
   SplittingRange * const aSplittingRange
) {
   EBM_ASSERT(1 <= cInstances);
   EBM_ASSERT(nullptr != aSingleFeatureValues);
   EBM_ASSERT(1 <= avgLength);
   EBM_ASSERT(1 <= cMinimumInstancesPerBin);
   EBM_ASSERT(1 <= cSplittingRanges);
   EBM_ASSERT(nullptr != aSplittingRange);

   FloatEbmType rangeValue = *aSingleFeatureValues;
   FloatEbmType * pSplittableValuesStart = aSingleFeatureValues;
   const FloatEbmType * pStartEqualRange = aSingleFeatureValues;
   FloatEbmType * pScan = aSingleFeatureValues + 1;
   const FloatEbmType * const pValuesEnd = aSingleFeatureValues + cInstances;

   SplittingRange * pSplittingRange = aSplittingRange;
   while(pValuesEnd != pScan) {
      const FloatEbmType val = *pScan;
      if(val != rangeValue) {
         size_t cEqualRangeItems = pScan - pStartEqualRange;
         if(avgLength <= cEqualRangeItems) {
            if(aSingleFeatureValues != pSplittableValuesStart || cMinimumInstancesPerBin <= static_cast<size_t>(pStartEqualRange - pSplittableValuesStart)) {
               EBM_ASSERT(pSplittingRange < aSplittingRange + cSplittingRanges);
               pSplittingRange->m_pSplittableValuesStart = pSplittableValuesStart;
               pSplittingRange->m_cSplittableItems = pStartEqualRange - pSplittableValuesStart;
               ++pSplittingRange;
            }
            pSplittableValuesStart = pScan;
         }
         rangeValue = val;
         pStartEqualRange = pScan;
      }
      ++pScan;
   }
   if(pSplittingRange != aSplittingRange + cSplittingRanges) {
      // we're not done, so we have one more to go.. this last one
      EBM_ASSERT(pSplittingRange == aSplittingRange + cSplittingRanges - 1);
      EBM_ASSERT(pSplittableValuesStart < pValuesEnd);
      pSplittingRange->m_pSplittableValuesStart = pSplittableValuesStart;
      EBM_ASSERT(pStartEqualRange < pValuesEnd);
      const size_t cEqualRangeItems = pValuesEnd - pStartEqualRange;
      const FloatEbmType * const pSplittableRangeEnd = avgLength <= cEqualRangeItems ? pStartEqualRange : pValuesEnd;
      pSplittingRange->m_cSplittableItems = pSplittableRangeEnd - pSplittableValuesStart;
   }
}

INLINE_RELEASE void FillSplittingRangeNeighbours(
   const size_t cInstances,
   FloatEbmType * const aSingleFeatureValues,
   const size_t cSplittingRanges,
   SplittingRange * const aSplittingRange
) {
   EBM_ASSERT(1 <= cInstances);
   EBM_ASSERT(nullptr != aSingleFeatureValues);
   EBM_ASSERT(1 <= cSplittingRanges);
   EBM_ASSERT(nullptr != aSplittingRange);

   SplittingRange * pSplittingRange = aSplittingRange;
   size_t cUnsplittablePriorItems = pSplittingRange->m_pSplittableValuesStart - aSingleFeatureValues;
   const FloatEbmType * const aSingleFeatureValuesEnd = aSingleFeatureValues + cInstances;
   if(1 != cSplittingRanges) {
      const SplittingRange * const pSplittingRangeLast = pSplittingRange + cSplittingRanges - 1; // exit without doing the last one
      do {
         const size_t cUnsplittableSubsequentItems =
            (pSplittingRange + 1)->m_pSplittableValuesStart - pSplittingRange->m_pSplittableValuesStart - pSplittingRange->m_cSplittableItems;

         pSplittingRange->m_cUnsplittablePriorItems = cUnsplittablePriorItems;
         pSplittingRange->m_cUnsplittableSubsequentItems = cUnsplittableSubsequentItems;

         cUnsplittablePriorItems = cUnsplittableSubsequentItems;
         ++pSplittingRange;
      } while(pSplittingRangeLast != pSplittingRange);
   }
   const size_t cUnsplittableSubsequentItems =
      aSingleFeatureValuesEnd - pSplittingRange->m_pSplittableValuesStart - pSplittingRange->m_cSplittableItems;

   pSplittingRange->m_cUnsplittablePriorItems = cUnsplittablePriorItems;
   pSplittingRange->m_cUnsplittableSubsequentItems = cUnsplittableSubsequentItems;
}

INLINE_RELEASE size_t FillSplittingRangeRemaining(
   const size_t cSplittingRanges,
   SplittingRange * const aSplittingRange
) {
   EBM_ASSERT(1 <= cSplittingRanges);
   EBM_ASSERT(nullptr != aSplittingRange);

   SplittingRange * pSplittingRange = aSplittingRange;
   const SplittingRange * const pSplittingRangeEnd = pSplittingRange + cSplittingRanges;
   do {
      const size_t cUnsplittablePriorItems = pSplittingRange->m_cUnsplittablePriorItems;
      const size_t cUnsplittableSubsequentItems = pSplittingRange->m_cUnsplittableSubsequentItems;

      pSplittingRange->m_cUnsplittableEitherSideMax = std::max(cUnsplittablePriorItems, cUnsplittableSubsequentItems);
      pSplittingRange->m_cUnsplittableEitherSideMin = std::min(cUnsplittablePriorItems, cUnsplittableSubsequentItems);

      pSplittingRange->m_flags = k_MiddleSplittingRange;
      pSplittingRange->m_cSplitsAssigned = 1;

      ++pSplittingRange;
   } while(pSplittingRangeEnd != pSplittingRange);

   size_t cConsumedSplittingRanges = cSplittingRanges;
   if(1 == cSplittingRanges) {
      aSplittingRange[0].m_flags = k_FirstSplittingRange | k_LastSplittingRange;
      // might as well assign a split to the only SplittingRange.  We'll be stuffing it as full as it can get soon
      EBM_ASSERT(1 == aSplittingRange[0].m_cSplitsAssigned);
   } else {
      aSplittingRange[0].m_flags = k_FirstSplittingRange;
      if(0 == aSplittingRange[0].m_cUnsplittablePriorItems) {
         aSplittingRange[0].m_cSplitsAssigned = 0;
         --cConsumedSplittingRanges;
      }

      --pSplittingRange; // go back to the last one
      pSplittingRange->m_flags = k_LastSplittingRange;
      if(0 == pSplittingRange->m_cUnsplittableSubsequentItems) {
         pSplittingRange->m_cSplitsAssigned = 0;
         --cConsumedSplittingRanges;;
      }
   }
   return cConsumedSplittingRanges;
}

INLINE_RELEASE void FillSplittingRangePointers(
   const size_t cSplittingRanges,
   SplittingRange ** const apSplittingRange,
   SplittingRange * const aSplittingRange
) {
   EBM_ASSERT(1 <= cSplittingRanges);
   EBM_ASSERT(nullptr != apSplittingRange);
   EBM_ASSERT(nullptr != aSplittingRange);

   SplittingRange ** ppSplittingRange = apSplittingRange;
   const SplittingRange * const * const apSplittingRangeEnd = apSplittingRange + cSplittingRanges;
   SplittingRange * pSplittingRange = aSplittingRange;
   do {
      *ppSplittingRange = pSplittingRange;
      ++pSplittingRange;
      ++ppSplittingRange;
   } while(apSplittingRangeEnd != ppSplittingRange);
}

/*
INLINE_RELEASE size_t AssignSecondCuts(
   RandomStream * const pRandomStream,
   const size_t cMaximumBins,
   const size_t cMinimumInstancesPerBin,
   const size_t cSplittingRanges,
   SplittingRange ** const apSplittingRange
) {
   // ok, at this point we've found all the long segments of equal values, and we've identified SplittingRanges where we have the option
   // of cutting, and since we dislike having two long segments joining together, we've guaranteed that all our existing SplittingRanges
   // have at least one cut.  In general, we'd like to isolate the ranges within each SplittingRanges away from the long strings of equal
   // values, since the stuff in between is more likley to be somehow unique.  With 1 cut in each SplittingRange we can ensure that
   // the long ranges are isolated from eachother, but we can't isolate the items between the long ranges.  Next, we'll try and give
   // some of our remaining cuts to the SplittingRange, which will allow them to put cuts on both sides of the ranges, thereby isolating
   // them from the long strings of equal values
   //
   // Unlike our first cut though, we're not guaranteed to have two cuts for every SplittingRange, so we have to be careful about who we give
   // them to.  In general, we don't like small clusters since you can get overfit pretty easily if they don't have sufficient instances
   // to group enough per class value.
   // So, we would prefer to first give our second cuts to the largest SplittingRange, irregardless about how big it's neighbours are
   // let's sort by the splitable items in each SplittingRange and then progress from the biggest to the smallest until we hit the minimum
   // width
   //
   // the first and last SplittingRange might or might not have long ranges of unsplittable values on their tail ends.  If there are no unsplittable
   // values on their other side, then we don't consider them here, since they don't need the second cut in order to isolate them from long ranges
   // of unsplittable values
   //
   // once we reach a SplittingRange with insufficient number of items inside to put cuts on both ends, we're done since none after that point in the sort
   // order will have enough

   EBM_ASSERT(nullptr != pRandomStream);
   EBM_ASSERT(2 <= cMaximumBins);
   EBM_ASSERT(1 <= cMinimumInstancesPerBin);
   EBM_ASSERT(1 <= cSplittingRanges);
   EBM_ASSERT(nullptr != apSplittingRange);

   EBM_ASSERT(cSplittingRanges < cMaximumBins); // if this isn't true then our guranteeed one cut per SplittingRange didn't work
   size_t cCutsRemaining = cMaximumBins - 1 - cSplittingRanges; // we gave one GUARANTEED cut to each SplittingRange so far.  How many cuts are left?
   if(0 != cCutsRemaining) {
      SortSplittingRangesByCountItemsDescending(pRandomStream, cSplittingRanges, apSplittingRange);
      SplittingRange ** ppSplittingRange = apSplittingRange;
      const SplittingRange * const * const ppSplittingRangeEnd = apSplittingRange + cSplittingRanges;
      do {
         SplittingRange * pSplittingRange = *ppSplittingRange;
         if(pSplittingRange->m_cSplittableItems < cMinimumInstancesPerBin) {
            // we can't split this SplittingRange more than once, and all subsequent ones will be the same, so exit
            break;
         }
         if(0 != pSplittingRange->m_cUnsplittableEitherSideMin) {
            // don't give second splits to the the first or last SplittingRanges unless they have unsplittable ranges on their outside
            EBM_ASSERT(1 == pSplittingRange->m_cSplitsAssigned);
            pSplittingRange->m_cSplitsAssigned = 2;
            --cCutsRemaining;
            if(0 == cCutsRemaining) {
               break;
            }
         }
         ++ppSplittingRange;
      } while(ppSplittingRangeEnd != ppSplittingRange);
   }
   return cCutsRemaining;
}
*/


constexpr static char g_pPrintfForRoundTrip[] = "%+.*" FloatEbmTypePrintf;
constexpr static char g_pPrintfLongInt[] = "%ld";
INLINE_RELEASE FloatEbmType FindClean1eFloat(
   const int cCharsFloatPrint,
   char * const pStr,
   const FloatEbmType low, 
   const FloatEbmType high, 
   FloatEbmType val
) {
   // we know that we are very close to 1e[something].  For positive exponents, we have a whole number,
   // which for smaller values is guaranteed to be exact, but for decimal numbers they will all be inexact
   // we could therefore be either "+9.99999999999999999e+299" or "+1.00000000000000000e+300"
   // we just need to check that the number starts with a 1 to be sure that we're the latter

   constexpr int cMantissaTextDigits = std::numeric_limits<FloatEbmType>::max_digits10;
   unsigned int cIterationsRemaining = 100;
   do {
      if(high <= val) {
         // oh no.  how did this happen.  Oh well, just return the high value, which is guaranteed 
         // to split low and high
         break;
      }
      const int cCharsWithoutNullTerminator = snprintf(
         pStr,
         cCharsFloatPrint,
         g_pPrintfForRoundTrip,
         cMantissaTextDigits,
         val
      );
      if(cCharsFloatPrint <= cCharsWithoutNullTerminator) {
         break;
      }
      if(0 == cCharsWithoutNullTerminator) {
         // check this before trying to access the 2nd item in the array
         break;
      }
      if('1' == pStr[1]) {
         // do one last check to verify for sure that we're above val in the end!
         val = low < val ? val : high;
         return val;
      }

      val = std::nextafter(val, std::numeric_limits<FloatEbmType>::max());
      --cIterationsRemaining;
   } while(0 != cIterationsRemaining);
   return high;
}
// checked
INLINE_RELEASE FloatEbmType GeometricMeanSameSign(const FloatEbmType val1, const FloatEbmType val2) {
   EBM_ASSERT(val1 < 0 && val2 < 0 || 0 <= val1 && 0 <= val2);
   FloatEbmType result = val1 * val2;
   if(UNLIKELY(std::isinf(result))) {
      if(PREDICTABLE(val1 < 0)) {
         result = -std::exp((std::log(-val1) + std::log(-val2)) * FloatEbmType { 0.5 });
      } else {
         result = std::exp((std::log(val1) + std::log(val2)) * FloatEbmType { 0.5 });
      }
   } else {
      result = std::sqrt(result);
      if(PREDICTABLE(val1 < 0)) {
         result = -result;
      }
   }
   return result;
}
// checked
constexpr int CountBase10CharactersAbs(int n) {
   // this works for negative numbers too
   return int { 0 } == n / int { 10 } ? int { 1 } : int { 1 } + CountBase10CharactersAbs(n / int { 10 });
}
// checked
constexpr long MaxReprsentation(int cDigits) {
   return int { 1 } == cDigits ? long { 9 } : long { 10 } * MaxReprsentation(cDigits - int { 1 }) + long { 9 };
}
INLINE_RELEASE FloatEbmType GetInterpretableCutPointFloat(const FloatEbmType low, const FloatEbmType high) {
   EBM_ASSERT(low < high);
   EBM_ASSERT(!std::isnan(low));
   EBM_ASSERT(!std::isinf(low));
   EBM_ASSERT(!std::isnan(high));
   EBM_ASSERT(!std::isinf(high));

   if(low < FloatEbmType { 0 } && FloatEbmType { 0 } <= high) {
      // if low is negative and high is positive, a natural cut point is zero.  Also, this solves the issue
      // that we can't take the geometric mean of mixed positive/negative numbers.
      return FloatEbmType { 0 };
   }

   // We want to handle widly different exponentials, so the average of 1e10 and 1e20 is 1e15, not 1e20 minus some 
   // small epsilon, so we use the geometric mean instead of the arithmetic mean.
   //
   // Because of floating point inexactness, geometricMean is NOT GUARANTEED 
   // to be (low < geometricMean && geometricMean <= high).  We generally don't return the geometric mean though,
   // so don't check it here.
   const FloatEbmType geometricMean = GeometricMeanSameSign(low, high);

   constexpr int cMantissaTextDigits = std::numeric_limits<FloatEbmType>::max_digits10;

   // Unfortunately, min_exponent10 doesn't seem to include denormal/subnormal numbers, so although it's the true
   // minimum exponent in terms of the floating point exponential representations, it isn't the true minimum exponent 
   // when considering numbers converted to text.  To counter this, we add 1 extra character.  For double numbers, 
   // we're 3 digits in either case, but in the more general scenario we might go from N to N+1 digits, but I think
   // it's really unlikely to go from N to N+2, since in the simplest case that would be a factor of 10 
   // (if the low number was almost N and the high number was just a bit above N+2), and subnormal numbers 
   // shouldn't increase the exponent by that much ever.

   constexpr int cExponentMaxTextDigits = CountBase10CharactersAbs(std::numeric_limits<FloatEbmType>::max_exponent10);
   constexpr int cExponentMinTextDigits = CountBase10CharactersAbs(std::numeric_limits<FloatEbmType>::min_exponent10);
   constexpr int cExponentTextDigits = 
      1 + cExponentMaxTextDigits < cExponentMinTextDigits ? cExponentMinTextDigits : cExponentMaxTextDigits;
       
   // example: "+9.12345678901234567e+300" (this is when 17 == cMantissaTextDigits, the value for doubles)
   // 3 characters for "+9."
   // cMantissaTextDigits characters for the mantissa text
   // 2 characters for "e+"
   // cExponentTextDigits characters for the exponent text
   // 1 characters for null terminator
   constexpr int cCharsFloatPrint = 3 + cMantissaTextDigits + 2 + cExponentTextDigits + 1;
   char str0[cCharsFloatPrint];
   char str1[cCharsFloatPrint];

   // I don't trust that snprintf has 100% guaranteed formats.  Let's trust, but verify the results, 
   // including indexes of characters like the "e" character

   // snprintf says to use the buffer size for the "n" term, but in alternate unicode versions it says # of characters
   // with the null terminator as one of the characters, so a string of 5 characters plus a null terminator would be 6.
   // For char strings, the number of bytes and the number of characters is the same.  I use number of characters for 
   // future-proofing the n term to unicode versions, so n-1 characters other than the null terminator can fill 
   // the buffer.  According to the docs, snprintf returns the number of characters that would have been written MINUS 
   // the null terminator.
   const int cLowCharsWithoutNullTerminator = snprintf(
      str0, 
      cCharsFloatPrint, 
      g_pPrintfForRoundTrip, 
      cMantissaTextDigits, 
      low
   );
   if(0 <= cLowCharsWithoutNullTerminator && cLowCharsWithoutNullTerminator < cCharsFloatPrint) {
      const int cHighCharsWithoutNullTerminator = snprintf(
         str1, 
         cCharsFloatPrint, 
         g_pPrintfForRoundTrip, 
         cMantissaTextDigits, 
         high
      );
      if(0 <= cHighCharsWithoutNullTerminator && cHighCharsWithoutNullTerminator < cCharsFloatPrint) {
         const char * pLowEChar = strchr(str0, 'e');
         if(nullptr == pLowEChar) {
            EBM_ASSERT(false); // we should be getting lower case 'e', but don't trust sprintf
            pLowEChar = strchr(str0, 'E');
         }
         if(nullptr != pLowEChar) {
            const char * pHighEChar = strchr(str1, 'e');
            if(nullptr == pHighEChar) {
               EBM_ASSERT(false); // we should be getting lower case 'e', but don't trust sprintf
               pHighEChar = strchr(str1, 'E');
            }
            if(nullptr != pHighEChar) {
               // use strtol instead of atoi incase we have a bad input.  atoi has undefined behavior if the
               // number isn't representable as an int.  strtol returns a 0 with bad inputs, or LONG_MAX, or LONG_MIN, 
               // which we handle by checking that our final output is within the range between low and high.
               const long lowExp = strtol(pLowEChar + 1, nullptr, 10);
               const long highExp = strtol(pHighEChar + 1, nullptr, 10);
               // strtol can return LONG_MAX, or LONG_MIN on errors.  We need to cleanse these away since they would
               // exceed the length of our print string
               constexpr long maxText = MaxReprsentation(cExponentTextDigits);
               // assert on this above, but don't trust our sprintf in release either
               if(-maxText <= lowExp && lowExp <= maxText && -maxText <= highExp && highExp <= maxText) {
                  const long double lowLongDouble = static_cast<long double>(low);
                  const long double highLongDouble = static_cast<long double>(high);
                  if(lowExp != highExp) {
                     EBM_ASSERT(lowExp < highExp);

                     str0[0] = '1';
                     str0[1] = 'e';

                     const long lowAvgExp = (lowExp + highExp) >> 1;
                     EBM_ASSERT(lowExp <= lowAvgExp);
                     EBM_ASSERT(lowAvgExp < highExp);
                     const long highAvgExp = lowAvgExp + 1;
                     EBM_ASSERT(lowExp < highAvgExp);
                     EBM_ASSERT(highAvgExp <= highExp);

                     // do the high avg exp first since it's guaranteed to exist and be between the low and high
                     // values, unlike the low avg exp which can be below the low value
                     const int cHighAvgExpWithoutNullTerminator = snprintf(
                        &str0[2],
                        cCharsFloatPrint - 2,
                        g_pPrintfLongInt,
                        highAvgExp
                     );
                     if(0 <= cHighAvgExpWithoutNullTerminator && cHighAvgExpWithoutNullTerminator < cCharsFloatPrint - 2) {
                        // unless something unexpected happens in our framework, str0 should be a valid 
                        // FloatEbmType value, which means it should also be a valid long double value
                        // so we shouldn't get a return of 0 for errors
                        //
                        // highAvgExp <= highExp, so e1HIGH is literally the smallest number that can be represented
                        // with the same exponent as high, so we shouldn't get back an overflow result, but check it
                        // anyways because of floating point jitter

                        // lowExp < highAvgExp, so e1HIGH should be larger than low, but check it
                        // anyways because of floating point jitter

                        const long double highExpLongDouble = strtold(str0, nullptr);

                        if(lowExp + 1 == highExp) {
                           EBM_ASSERT(lowAvgExp == lowExp);
                           // 1eLOW can't be above low since it's literally the lowest value with the same exponent
                           // as our low value.  So, skip all the low value computations

                        only_high_exp:
                           if(lowLongDouble < highExpLongDouble && highExpLongDouble <= highLongDouble) {
                              // we know that highExpLongDouble can be converted to FloatEbmType since it's
                              // between valid FloatEbmTypes, our low and high values.
                              const FloatEbmType highExpFloat = static_cast<FloatEbmType>(highExpLongDouble);
                              return FindClean1eFloat(cCharsFloatPrint, str0, low, high, highExpFloat);
                           } else {
                              // fallthrough case.  Floating point numbers are inexact, so perhaps if they are 
                              // separated by 1 epsilon or something like that and/or the text conversion isn't exact, 
                              // we could get a case where this might happen
                           }
                        } else {
                           const int cLowAvgExpWithoutNullTerminator = snprintf(
                              &str0[2],
                              cCharsFloatPrint - 2,
                              g_pPrintfLongInt,
                              lowAvgExp
                           );
                           if(0 <= cLowAvgExpWithoutNullTerminator && cLowAvgExpWithoutNullTerminator < cCharsFloatPrint - 2) {
                              EBM_ASSERT(lowExp < lowAvgExp);
                              EBM_ASSERT(lowAvgExp < highExp);

                              // unless something unexpected happens in our framework, str0 should be a valid 
                              // FloatEbmType value, which means it should also be a valid long double value
                              // so we shouldn't get a return of 0 for errors
                              //
                              // lowAvgExp is above lowExp and below lowAvgExp, which are both valid FloatEbmTypes
                              // so str0 must contain a valid number that is convertable to FloatEbmTypes
                              // but check this anyways incase there is floating point jitter

                              const long double lowExpLongDouble = strtold(str0, nullptr);

                              if(lowLongDouble < lowExpLongDouble && lowExpLongDouble <= highLongDouble) {
                                 // We know that lowExpLongDouble can be converted now to FloatEbmType since it's
                                 // between valid our low and high FloatEbmType values.
                                 const FloatEbmType lowExpFloat = static_cast<FloatEbmType>(lowExpLongDouble);
                                 if(lowLongDouble < highExpLongDouble && highExpLongDouble <= highLongDouble) {
                                    // we know that highExpLongDouble can be converted now to FloatEbmType since it's
                                    // between valid FloatEbmType, our low and high values.
                                    const FloatEbmType highExpFloat = static_cast<FloatEbmType>(highExpLongDouble);

                                    // take the one that is closest to the geometric mean
                                    //
                                    // we want to compare in terms of exponential distance, so instead of subtacting,
                                    // divide these.  Flip them so that the geometricMean is at the bottom of the low
                                    // one because it's expected to be bigger than the lowExpFloat (the lowest of all
                                    // 3 numbers)
                                    const FloatEbmType lowRatio = lowExpFloat / geometricMean;
                                    const FloatEbmType highRatio = geometricMean / highExpFloat;
                                    // we flipped them, so higher numbers (closer to 1) are bad.  We want small numbers
                                    if(lowRatio < highRatio) {
                                       return FindClean1eFloat(cCharsFloatPrint, str0, low, high, lowExpFloat);
                                    } else {
                                       return FindClean1eFloat(cCharsFloatPrint, str0, low, high, highExpFloat);
                                    }
                                 } else {
                                    return FindClean1eFloat(cCharsFloatPrint, str0, low, high, lowExpFloat);
                                 }
                              } else {
                                 goto only_high_exp;
                              }
                           } else {
                              EBM_ASSERT(false); // this shouldn't happen, but don't trust sprintf
                           }
                        }
                     } else {
                        EBM_ASSERT(false); // this shouldn't happen, but don't trust sprintf
                     }
                  } else {
                     EBM_ASSERT('+' == str0[0] || '-' == str0[0]);
                     EBM_ASSERT('+' == str1[0] || '-' == str1[0]);
                     EBM_ASSERT(str0[0] == str1[0]);

                     // there should somewhere be an 'e" or 'E' character, otherwise we wouldn't have gotten here,
                     // so there must at least be 1 character
                     size_t iChar = 1;
                     // we shouldn't really need to take the min value, but I don't trust floating point number text
                     const size_t iCharEnd = std::min(pLowEChar - str0, pHighEChar - str1);
                     // handle the virtually impossible case of the string starting with 'e' by using iChar < iCharEnd
                     while(LIKELY(iChar < iCharEnd)) {
                        // "+9.1234 5 678901234567e+300" (low)
                        // "+9.1234 6 546545454545e+300" (high)
                        if(UNLIKELY(str0[iChar] != str1[iChar])) {
                           // we know our low value is lower, so this digit should be lower
                           EBM_ASSERT(str0[iChar] < str1[iChar]);
                           // nothing is bigger than '9' for a single digit, so the low value can't be '9'
                           EBM_ASSERT('9' != str0[iChar]);
                           char * pDiffChar = str0 + iChar;
                           memmove(
                              pDiffChar + 1,
                              pLowEChar,
                              static_cast<size_t>(cLowCharsWithoutNullTerminator) - (pLowEChar - str0) + 1
                           );

                           const char charEnd = str1[iChar];
                           char curChar = *pDiffChar;
                           FloatEbmType ret = FloatEbmType { 0 }; // this value should never be used
                           FloatEbmType bestRatio = std::numeric_limits<FloatEbmType>::lowest();
                           char bestChar = 0;
                           do {
                              // start by incrementing the char, since if we chop off trailing digits we won't
                              // end up with a number higher than the low value
                              ++curChar;
                              *pDiffChar = curChar;
                              const long double valLongDouble = strtold(str0, nullptr);
                              if(lowLongDouble < valLongDouble && valLongDouble <= highLongDouble) {
                                 // we know that valLongDouble can be converted to FloatEbmType since it's
                                 // between valid FloatEbmTypes, our low and high values.
                                 const FloatEbmType val = static_cast<FloatEbmType>(valLongDouble);
                                 const FloatEbmType ratio = 
                                    geometricMean < val ? geometricMean / val: val / geometricMean;
                                 EBM_ASSERT(ratio <= FloatEbmType { 1 });
                                 if(bestRatio < ratio) {
                                    bestRatio = ratio;
                                    bestChar = curChar;
                                    ret = val;
                                 }
                              }
                           } while(charEnd != curChar);
                           if(std::numeric_limits<FloatEbmType>::max() != bestRatio) {
                              // once we have our value, try converting it with printf to ensure that it gives 0000s 
                              // at the end (where the text will match up), instead of 9999s.  If we get this, then 
                              // increment the floating point with integer math until it works.

                              // restore str0 to the best string available
                              *pDiffChar = bestChar;

                              unsigned int cIterationsRemaining = 100;
                              do {
                                 int cCheckCharsWithoutNullTerminator = snprintf(
                                    str1,
                                    cCharsFloatPrint,
                                    g_pPrintfForRoundTrip,
                                    cMantissaTextDigits,
                                    ret
                                 );
                                 if(cCheckCharsWithoutNullTerminator < 0 || 
                                    cCharsFloatPrint <= cCheckCharsWithoutNullTerminator) 
                                 {
                                    break;
                                 }
                                 size_t iFindChar = 0;
                                 while(true) {
                                    if(LIKELY(iChar < iFindChar)) {
                                       // all seems good.  We examined up until what was the changing char
                                       return ret;
                                    }
                                    if(str0[iFindChar] != str1[iFindChar]) {
                                       break;
                                    }
                                    ++iFindChar;
                                 }
                                 ret = std::nextafter(ret, std::numeric_limits<FloatEbmType>::max());
                                 --cIterationsRemaining;
                              } while(0 != cIterationsRemaining);
                           }
                           break; // this shouldn't happen, but who knows with floats
                        }
                        ++iChar;
                     }
                     // we should have seen a difference somehwere since our low should be lower than our high,
                     // and we used enough digits for a "round trip" guarantee, but whatever.  Just fall through
                     // and handle it like other close numbers where we just take the geometric mean
                     EBM_ASSERT(false); // this shouldn't happen, but don't trust sprintf
                  }
               } else {
                  EBM_ASSERT(false); // this shouldn't happen, but don't trust sprintf
               }
            } else {
               EBM_ASSERT(false); // this shouldn't happen, but don't trust sprintf
            }
         } else {
            EBM_ASSERT(false); // this shouldn't happen, but don't trust sprintf
         }
      } else {
         EBM_ASSERT(false); // this shouldn't happen, but don't trust sprintf
      }
   } else {
      EBM_ASSERT(false); // this shouldn't happen, but don't trust sprintf
   }
   // something failed, probably due to floating point inexactness.  Let's first try and see if the 
   // geometric mean will work
   if(low < geometricMean && geometricMean <= high) {
      return geometricMean;
   }

   // For interpretability reasons, our digitization puts numbers that are exactly equal to the cut point into the 
   // higher bin. This keeps 2 in the (2, 3] bin if the cut point is 2, so that 2 is lumped in with 2.2, 2.9, etc
   // 
   // We should never reall get to this point in the code, except perhaps in exceptionally contrived cases, like 
   // perahps if two floating poing numbers were separated by 1 epsilon.
   return high;
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateQuantileCutPoints(
   IntEbmType randomSeed,
   IntEbmType countInstances,
   FloatEbmType * singleFeatureValues,
   IntEbmType countMaximumBins,
   IntEbmType countMinimumInstancesPerBin,
   FloatEbmType * cutPointsLowerBoundInclusive,
   IntEbmType * countCutPoints,
   IntEbmType * isMissing,
   FloatEbmType * minValue,
   FloatEbmType * maxValue
) {
   EBM_ASSERT(0 <= countInstances);
   EBM_ASSERT(0 == countInstances || nullptr != singleFeatureValues);
   EBM_ASSERT(0 <= countMaximumBins);
   EBM_ASSERT(0 == countInstances || 0 < countMaximumBins); // countMaximumBins can only be zero if there are no instances, because otherwise you need a bin
   EBM_ASSERT(0 <= countMinimumInstancesPerBin);
   EBM_ASSERT(0 == countInstances || countMaximumBins <= 1 || nullptr != cutPointsLowerBoundInclusive);
   EBM_ASSERT(nullptr != countCutPoints);
   EBM_ASSERT(nullptr != isMissing);
   EBM_ASSERT(nullptr != minValue);
   EBM_ASSERT(nullptr != maxValue);

   LOG_N(TraceLevelInfo, "Entered GenerateQuantileCutPoints: randomSeed=%" IntEbmTypePrintf ", countInstances=%" IntEbmTypePrintf 
      ", singleFeatureValues=%p, countMaximumBins=%" IntEbmTypePrintf ", countMinimumInstancesPerBin=%" IntEbmTypePrintf 
      ", cutPointsLowerBoundInclusive=%p, countCutPoints=%p, isMissing=%p, minValue=%p, maxValue=%p", 
      randomSeed, 
      countInstances, 
      static_cast<void *>(singleFeatureValues), 
      countMaximumBins, 
      countMinimumInstancesPerBin, 
      static_cast<void *>(cutPointsLowerBoundInclusive), 
      static_cast<void *>(countCutPoints),
      static_cast<void *>(isMissing),
      static_cast<void *>(minValue),
      static_cast<void *>(maxValue)
   );

   if(!IsNumberConvertable<size_t, IntEbmType>(countInstances)) {
      LOG_0(TraceLevelWarning, "WARNING GenerateQuantileCutPoints !IsNumberConvertable<size_t, IntEbmType>(countInstances)");
      return 1;
   }

   if(!IsNumberConvertable<size_t, IntEbmType>(countMaximumBins)) {
      LOG_0(TraceLevelWarning, "WARNING GenerateQuantileCutPoints !IsNumberConvertable<size_t, IntEbmType>(countMaximumBins)");
      return 1;
   }

   if(!IsNumberConvertable<size_t, IntEbmType>(countMinimumInstancesPerBin)) {
      LOG_0(TraceLevelWarning, "WARNING GenerateQuantileCutPoints !IsNumberConvertable<size_t, IntEbmType>(countMinimumInstancesPerBin)");
      return 1;
   }

   const size_t cInstancesIncludingMissingValues = static_cast<size_t>(countInstances);

   IntEbmType ret = 0;
   if(0 == cInstancesIncludingMissingValues) {
      *countCutPoints = 0;
      *isMissing = EBM_FALSE;
      *minValue = 0;
      *maxValue = 0;
   } else {
      const size_t cInstances = RemoveMissingValues(cInstancesIncludingMissingValues, singleFeatureValues);

      const bool bMissing = cInstancesIncludingMissingValues != cInstances;
      *isMissing = bMissing ? EBM_TRUE : EBM_FALSE;

      if(0 == cInstances) {
         *countCutPoints = 0;
         *minValue = 0;
         *maxValue = 0;
      } else {
         FloatEbmType * const pValuesEnd = singleFeatureValues + cInstances;
         std::sort(singleFeatureValues, pValuesEnd);
         *minValue = singleFeatureValues[0];
         *maxValue = pValuesEnd[-1];
         if(countMaximumBins <= 1) {
            // if there is only 1 bin, then there can be no cut points, and no point doing any more work here
            *countCutPoints = 0;
         } else {
            const size_t cMinimumInstancesPerBin =
               countMinimumInstancesPerBin <= IntEbmType { 0 } ? size_t { 1 } : static_cast<size_t>(countMinimumInstancesPerBin);
            const size_t cMaximumBins = PossiblyRemoveBinForMissing(bMissing, countMaximumBins);
            EBM_ASSERT(2 <= cMaximumBins); // if we had just one bin then there would be no cuts and we should have exited above
            const size_t avgLength = GetAvgLength(cInstances, cMaximumBins, cMinimumInstancesPerBin);
            EBM_ASSERT(1 <= avgLength);
            const size_t cSplittingRanges = CountSplittingRanges(cInstances, singleFeatureValues, avgLength, cMinimumInstancesPerBin);
            // we GUARANTEE that each SplittingRange can have at least one cut by choosing an avgLength sufficiently long to ensure this property
            EBM_ASSERT(cSplittingRanges < cMaximumBins);
            if(0 == cSplittingRanges) {
               *countCutPoints = 0;
            } else {
               try {
                  RandomStream randomStream(randomSeed);
                  if(!randomStream.IsSuccess()) {
                     goto exit_error;
                  }

                  const size_t cBytesCombined = sizeof(SplittingRange) + sizeof(SplittingRange *);
                  if(IsMultiplyError(cSplittingRanges, cBytesCombined)) {
                     goto exit_error;
                  }
                  // use the same memory allocation for both the Junction items and the pointers to the junctions that we'll use for sorting
                  SplittingRange ** const apSplittingRange = static_cast<SplittingRange **>(malloc(cSplittingRanges * cBytesCombined));
                  if(nullptr == apSplittingRange) {
                     goto exit_error;
                  }
                  SplittingRange * const aSplittingRange = reinterpret_cast<SplittingRange *>(apSplittingRange + cSplittingRanges);

                  FillSplittingRangeBasics(cInstances, singleFeatureValues, avgLength, cMinimumInstancesPerBin, cSplittingRanges, aSplittingRange);
                  FillSplittingRangeNeighbours(cInstances, singleFeatureValues, cSplittingRanges, aSplittingRange);
                  /* const size_t cUsedSplits = */ FillSplittingRangeRemaining(cSplittingRanges, aSplittingRange);
                  /* size_t cCutsRemaining = cMaximumBins - 1 - cUsedSplits; */
                  FillSplittingRangePointers(cSplittingRanges, apSplittingRange, aSplittingRange);

                  //GetInterpretableCutPointFloat(0, 1);
                  //GetInterpretableCutPointFloat(11, 12);
                  //GetInterpretableCutPointFloat(345.33545, 3453.3745);
                  //GetInterpretableCutPointFloat(0.000034533545, 0.0034533545);

                  // generally, having small bins with insufficient data to cover the base rate is more damaging
                  // than the lost opportunity from not cutting big bins.  So, what we want to avoid is having
                  // small bins.  So, create a heap and insert the average bin size AFTER we added a new cut
                  // don't insert any SplittingRanges that cannot legally be cut (so, it's guaranteed to only have
                  // cuttable items).  Now pick off the SplittingRange that has the largest average AFTER adding a cut
                  // and then add the cut, re-calculate the new average, and re-insert into the heap.  Continue
                  // until there are no items in the heap because they've all exhausted the possibilities of cuts
                  // OR until we run out of cuts to dole out.


                  // TODO: 
                  // next, go inwards from each cut point ends
                  // then try to even out the ranges by looking at semi-long ranges that we collided with and attempting
                  // to slide our cut points (without changing the number)

                  // create two sorted lists. One list is sorted by some metric (perhaps minimum size) cost for ADDing 
                  // one cut to an existing SplittingRange.  The second list is sorted by some metric for the cost of 
                  // subtracting one cut to an existing SplittingRange.  Now, we can pick off one cut from the worst
                  // SplittingRange and give that cut to the best SplittingRange.  Continue down both lists
                  // until we reach a situation where swaping them provides no value, or a negative value
                  // REPEAT several cycles (maybe)




                  //// let's assign how many 
                  //ppSplittingRange = apSplittingRange;
                  //const SplittingRange * const * const apSplittingRangesEnd = apSplittingRange + cSplittingRanges;
                  //const SplittingRange * const pSplittingRangesLast = aSplittingRange + cSplittingRanges - 1;
                  //size_t cCutsRemaining = cMaximumBins - 1;
                  //do {
                  //   SplittingRange * pSplittingRange = *ppSplittingRange;
                  //   if(UNLIKELY(cMinimumInstancesPerBin <= pSplittingRange->m_cItemsSplittableBefore)) {
                  //      // of, we've assigned everything for which we basically had no choices.  Now let's take stock of where we are by counting the number

                  //      do {
                  //         const SplittingRange * pSplittingRange = *ppSplittingRange;
                  //         EBM_ASSERT(cMinimumInstancesPerBin <= pSplittingRange->m_cItemsSplittableBefore);




                  //         if(pSplittingRange == aSplittingRange || pSplittingRange == pSplittingRangeLast && 0 == pSplittingRange->m_cItemsUnsplittableAfter) {
                  //            // if we're at the first item, we can't put a cut on the left
                  //            // if we're at the last item, we can't put a cut on the right, unless there is a long string of items on the right, in which case we can
                  //         }
                  //         if(pSplittingRange != aSplittingRange && (pSplittingRange != pSplittingRangeLast || 0 != pSplittingRange->m_cItemsUnsplittableAfter)) {
                  //            // if we're at the first item, we can't put a cut on the left
                  //            // if we're at the last item, we can't put a cut on the right, unless there is a long string of items on the right, in which case we can
                  //         }
                  //         ++ppSplittingRange;
                  //      } while(apSplittingRangeEnd != ppSplittingRange);
                  //      break;
                  //   }
                  //   if(LIKELY(LIKELY(pJunction != aJunctions) && 
                  //      LIKELY(LIKELY(pJunction != pJunctionsLast) || LIKELY(0 != pJunction->m_cItemsUnsplittableAfter))))
                  //   {
                  //      // if we're at the first item, we can't put a cut on the left
                  //      // if we're at the last item, we can't put a cut on the right, unless there is a long string of items on the right, in which case we can
                  //      
                  //      
                  //      
                  //      //pSplittingRange->m_cSplits = 1;
                  //      
                  //      
                  //      
                  //      --cCutsRemaining;
                  //      if(UNLIKELY(0 == cCutsRemaining)) {
                  //         // this shouldn't be possible, but maybe some terrible dataset might achieve this bad result
                  //         // let's not handle this case, since I think it's impossible
                  //         break;
                  //      }
                  //   }
                  //   ++ppSplittingRange;
                  //} while(apSplittingRangeEnd != ppSplittingRange);




                  //SortSplittingRangesByUnsplittableDescending(&randomStream, cSplittingRanges, apSplittingRange);



                  free(apSplittingRange); // both the junctions and the pointers to the junctions are in the same memory allocation

                  // first let's tackle the short ranges between big ranges (or at the tails) where we know there will be a split to separate the big ranges to either
                  // side, but the short range isn't big enough to split.  In otherwords, there are less than cMinimumInstancesPerBin items
                  // we start with the biggest long ranges and essentially try to push whatever mass there is away from them and continue down the list

                  *countCutPoints = 0;
               } catch(...) {
                  ret = 1;
               }
            }
         }
      }
   }
   if(0 != ret) {
      LOG_N(TraceLevelWarning, "WARNING GenerateQuantileCutPoints returned %" IntEbmTypePrintf, ret);
   } else {
      LOG_N(TraceLevelInfo, "Exited GenerateQuantileCutPoints countCutPoints=%" IntEbmTypePrintf ", isMissing=%" IntEbmTypePrintf,
         *countCutPoints,
         *isMissing
      );
   }
   return ret;

exit_error:;
   ret = 1;
   LOG_N(TraceLevelWarning, "WARNING GenerateQuantileCutPoints returned %" IntEbmTypePrintf, ret);
   return ret;
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateImprovedEqualWidthCutPoints(
   IntEbmType countInstances,
   FloatEbmType * singleFeatureValues,
   IntEbmType countMaximumBins,
   FloatEbmType * cutPointsLowerBoundInclusive,
   IntEbmType * countCutPoints,
   IntEbmType * isMissing,
   FloatEbmType * minValue,
   FloatEbmType * maxValue
) {
   UNUSED(countInstances);
   UNUSED(singleFeatureValues);
   UNUSED(countMaximumBins);
   UNUSED(cutPointsLowerBoundInclusive);
   UNUSED(countCutPoints);
   UNUSED(isMissing);
   UNUSED(minValue);
   UNUSED(maxValue);

   // TODO: IMPLEMENT

   return 0;
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateEqualWidthCutPoints(
   IntEbmType countInstances,
   FloatEbmType * singleFeatureValues,
   IntEbmType countMaximumBins,
   FloatEbmType * cutPointsLowerBoundInclusive,
   IntEbmType * countCutPoints,
   IntEbmType * isMissing,
   FloatEbmType * minValue,
   FloatEbmType * maxValue
) {
   UNUSED(countInstances);
   UNUSED(singleFeatureValues);
   UNUSED(countMaximumBins);
   UNUSED(cutPointsLowerBoundInclusive);
   UNUSED(countCutPoints);
   UNUSED(isMissing);
   UNUSED(minValue);
   UNUSED(maxValue);

   // TODO: IMPLEMENT

   return 0;
}

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION Discretize(
   IntEbmType isMissing,
   IntEbmType countCutPoints,
   const FloatEbmType * cutPointsLowerBoundInclusive,
   IntEbmType countInstances,
   const FloatEbmType * singleFeatureValues,
   IntEbmType * singleFeatureDiscretized
) {
   EBM_ASSERT(EBM_FALSE == isMissing || EBM_TRUE == isMissing);
   EBM_ASSERT(0 <= countCutPoints);
   EBM_ASSERT((IsNumberConvertable<size_t, IntEbmType>(countCutPoints))); // this needs to point to real memory, otherwise it's invalid
   EBM_ASSERT(0 == countInstances || 0 == countCutPoints || nullptr != cutPointsLowerBoundInclusive);
   EBM_ASSERT(0 <= countInstances);
   EBM_ASSERT((IsNumberConvertable<size_t, IntEbmType>(countInstances))); // this needs to point to real memory, otherwise it's invalid
   EBM_ASSERT(0 == countInstances || nullptr != singleFeatureValues);
   EBM_ASSERT(0 == countInstances || nullptr != singleFeatureDiscretized);

   if(IntEbmType { 0 } < countInstances) {
      const size_t cCutPoints = static_cast<size_t>(countCutPoints);
#ifndef NDEBUG
      for(size_t iDebug = 1; iDebug < cCutPoints; ++iDebug) {
         EBM_ASSERT(cutPointsLowerBoundInclusive[iDebug - 1] < cutPointsLowerBoundInclusive[iDebug]);
      }
# endif // NDEBUG
      const size_t cInstances = static_cast<size_t>(countInstances);
      const FloatEbmType * pValue = singleFeatureValues;
      const FloatEbmType * const pValueEnd = singleFeatureValues + cInstances;
      IntEbmType * pDiscretized = singleFeatureDiscretized;

      if(size_t { 0 } == cCutPoints) {
         const IntEbmType missingVal = EBM_FALSE != isMissing ? IntEbmType { 0 } : IntEbmType { -1 };
         const IntEbmType nonMissingVal = EBM_FALSE != isMissing ? IntEbmType { 1 } : IntEbmType { 0 };
         do {
            const FloatEbmType val = *pValue;
            const IntEbmType result = UNPREDICTABLE(std::isnan(val)) ? missingVal : nonMissingVal;
            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
      } else {
         const ptrdiff_t highStart = static_cast<ptrdiff_t>(cCutPoints - size_t { 1 });
         if(EBM_FALSE != isMissing) {
            // there are missing values.  We need to bump up all indexes by 1 and make missing zero
            do {
               ptrdiff_t middle = ptrdiff_t { 0 };
               const FloatEbmType val = *pValue;
               if(!std::isnan(val)) {
                  ptrdiff_t high = highStart;
                  ptrdiff_t low = 0;
                  FloatEbmType midVal;
                  do {
                     middle = (low + high) >> 1;
                     EBM_ASSERT(0 <= middle && middle <= highStart);
                     midVal = cutPointsLowerBoundInclusive[static_cast<size_t>(middle)];
                     high = UNPREDICTABLE(midVal <= val) ? high : middle - ptrdiff_t { 1 };
                     low = UNPREDICTABLE(midVal <= val) ? middle + ptrdiff_t { 1 } : low;
                  } while(LIKELY(low <= high));
                  // we bump up all indexes to allow missing to be 0
                  middle = UNPREDICTABLE(midVal <= val) ? middle + ptrdiff_t { 2 } : middle + ptrdiff_t { 1 };
                  EBM_ASSERT(ptrdiff_t { 0 } <= middle - ptrdiff_t { 1 } && middle - ptrdiff_t { 1 } <= static_cast<ptrdiff_t>(cCutPoints));
               }
               *pDiscretized = static_cast<IntEbmType>(middle);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
         } else {
            // there are no missing values, but check anyways. If there are missing anyways, then make them -1
            do {
               ptrdiff_t middle = ptrdiff_t { -1 };
               const FloatEbmType val = *pValue;
               if(!std::isnan(val)) {
                  ptrdiff_t high = highStart;
                  ptrdiff_t low = 0;
                  FloatEbmType midVal;
                  do {
                     middle = (low + high) >> 1;
                     EBM_ASSERT(0 <= middle && middle <= highStart);
                     midVal = cutPointsLowerBoundInclusive[static_cast<size_t>(middle)];
                     high = UNPREDICTABLE(midVal <= val) ? high : middle - ptrdiff_t { 1 };
                     low = UNPREDICTABLE(midVal <= val) ? middle + ptrdiff_t { 1 } : low;
                  } while(LIKELY(low <= high));
                  middle = UNPREDICTABLE(midVal <= val) ? middle + ptrdiff_t { 1 } : middle;
                  EBM_ASSERT(ptrdiff_t { 0 } <= middle && middle <= static_cast<ptrdiff_t>(cCutPoints));
               }
               *pDiscretized = static_cast<IntEbmType>(middle);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
         }
      }
   }
}
