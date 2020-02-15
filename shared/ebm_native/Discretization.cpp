// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <algorithm> // sort
#include <cmath> // std::round
#include <vector>

#include "ebm_native.h"

#include "EbmInternal.h"
// very independent includes
#include "Logging.h" // EBM_ASSERT & LOG
#include "RandomStream.h"

struct Junction {
   // this records the junction between a range of changing values that are splittable to the left, 
   // and a long range of identical values on the right that are unsplittable, or the end of the array
   // in which case m_cItemsUnsplittableAfter will be zero
   FloatEbmType * m_pJunctionFirstUnsplittable;
   size_t         m_cItemsSplittableBefore; // this can be zero
   size_t         m_cItemsUnsplittableAfter;

   // we first pick out ranges that we split with one cut, and then resort the data.  Keeping our single cut here is a performant way to sort the data
   // while keeping it together
   size_t         m_iSingleSplitBetweenJunctions;
};

// PK VERIFIED!
void SortJunctionsBySplittableAscending(RandomStream * const pRandomStream, const size_t cJunctions, Junction ** const apJunctions) {
   EBM_ASSERT(0 < cJunctions);

   // sort in ascending order for m_cItemsSplittableBefore
   //
   // But some items can have the same primary sort key, so sort secondarily on the pointer to the original object, thus putting them secondarily in index
   // order, which is guaranteed to be a unique ordering. We'll later randomize the order of items that have the same primary sort index, BUT we want our 
   // initial sort order to be replicatable with the same random seed, so we need the initial sort to be stable.
   std::sort(apJunctions, apJunctions + cJunctions, [](Junction *& junction1, Junction *& junction2) {
      if(PREDICTABLE(junction1->m_cItemsSplittableBefore == junction2->m_cItemsSplittableBefore)) {
         return UNPREDICTABLE(junction1->m_pJunctionFirstUnsplittable < junction2->m_pJunctionFirstUnsplittable);
      } else {
         return UNPREDICTABLE(junction1->m_cItemsSplittableBefore < junction2->m_cItemsSplittableBefore);
      }
      });

   // find sections that have the same number of items and randomly shuffle the sections with equal numbers of items so that there is no directional preference
   size_t iStartEqualLengthRange = 0;
   size_t cItems = apJunctions[0]->m_cItemsSplittableBefore;
   for(size_t i = 1; LIKELY(i < cJunctions); ++i) {
      const size_t cNewItems = apJunctions[i]->m_cItemsSplittableBefore;
      if(PREDICTABLE(cItems != cNewItems)) {
         // we have a real range
         size_t cRemainingItems = i - iStartEqualLengthRange;
         EBM_ASSERT(1 <= cRemainingItems);
         while(PREDICTABLE(1 != cRemainingItems)) {
            const size_t iSwap = pRandomStream->Next(cRemainingItems);
            Junction * pTmp = apJunctions[iStartEqualLengthRange];
            apJunctions[iStartEqualLengthRange] = apJunctions[iStartEqualLengthRange + iSwap];
            apJunctions[iStartEqualLengthRange + iSwap] = pTmp;
            ++iStartEqualLengthRange;
            --cRemainingItems;
         }
         iStartEqualLengthRange = i;
         cItems = cNewItems;
      }
   }
   size_t cRemainingItemsOuter = cJunctions - iStartEqualLengthRange;
   EBM_ASSERT(1 <= cRemainingItemsOuter);
   while(PREDICTABLE(1 != cRemainingItemsOuter)) {
      const size_t iSwap = pRandomStream->Next(cRemainingItemsOuter);
      Junction * pTmp = apJunctions[iStartEqualLengthRange];
      apJunctions[iStartEqualLengthRange] = apJunctions[iStartEqualLengthRange + iSwap];
      apJunctions[iStartEqualLengthRange + iSwap] = pTmp;
      ++iStartEqualLengthRange;
      --cRemainingItemsOuter;
   }
}

// PK VERIFIED!
void SortJunctionsByUnsplittableDescending(RandomStream * const pRandomStream, const size_t cJunctions, Junction ** const apJunctions) {
   EBM_ASSERT(0 < cJunctions);

   // sort in descending order for m_cItemsUnsplittableAfter
   //
   // But some items can have the same primary sort key, so sort secondarily on the pointer to the original object, thus putting them secondarily in index
   // order, which is guaranteed to be a unique ordering. We'll later randomize the order of items that have the same primary sort index, BUT we want our 
   // initial sort order to be replicatable with the same random seed, so we need the initial sort to be stable.
   std::sort(apJunctions, apJunctions + cJunctions, [](Junction * & junction1, Junction * & junction2) {
      if(PREDICTABLE(junction1->m_cItemsUnsplittableAfter == junction2->m_cItemsUnsplittableAfter)) {
         return UNPREDICTABLE(junction1->m_pJunctionFirstUnsplittable > junction2->m_pJunctionFirstUnsplittable);
      } else {
         return UNPREDICTABLE(junction1->m_cItemsUnsplittableAfter > junction2->m_cItemsUnsplittableAfter);
      }
   });

   // find sections that have the same number of items and randomly shuffle the sections with equal numbers of items so that there is no directional preference
   size_t iStartEqualLengthRange = 0;
   size_t cItems = apJunctions[0]->m_cItemsUnsplittableAfter;
   for(size_t i = 1; LIKELY(i < cJunctions); ++i) {
      const size_t cNewItems = apJunctions[i]->m_cItemsUnsplittableAfter;
      if(PREDICTABLE(cItems != cNewItems)) {
         // we have a real range
         size_t cRemainingItems = i - iStartEqualLengthRange;
         EBM_ASSERT(1 <= cRemainingItems);
         while(PREDICTABLE(1 != cRemainingItems)) {
            const size_t iSwap = pRandomStream->Next(cRemainingItems);
            Junction * pTmp = apJunctions[iStartEqualLengthRange];
            apJunctions[iStartEqualLengthRange] = apJunctions[iStartEqualLengthRange + iSwap];
            apJunctions[iStartEqualLengthRange + iSwap] = pTmp;
            ++iStartEqualLengthRange;
            --cRemainingItems;
         }
         iStartEqualLengthRange = i;
         cItems = cNewItems;
      }
   }
   size_t cRemainingItemsOuter = cJunctions - iStartEqualLengthRange;
   EBM_ASSERT(1 <= cRemainingItemsOuter);
   while(PREDICTABLE(1 != cRemainingItemsOuter)) {
      const size_t iSwap = pRandomStream->Next(cRemainingItemsOuter);
      Junction * pTmp = apJunctions[iStartEqualLengthRange];
      apJunctions[iStartEqualLengthRange] = apJunctions[iStartEqualLengthRange + iSwap];
      apJunctions[iStartEqualLengthRange + iSwap] = pTmp;
      ++iStartEqualLengthRange;
      --cRemainingItemsOuter;
   }
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateQuantileCutPoints(
   IntEbmType randomSeed,
   IntEbmType countInstances,
   FloatEbmType * singleFeatureValues,
   IntEbmType countMaximumBins,
   IntEbmType countMinimumInstancesPerBin,
   FloatEbmType * cutPointsLowerBoundInclusive,
   IntEbmType * countCutPoints,
   IntEbmType * isMissing
) {
   EBM_ASSERT(0 <= countInstances);
   EBM_ASSERT(0 == countInstances || nullptr != singleFeatureValues);
   EBM_ASSERT(0 <= countMaximumBins);
   EBM_ASSERT(0 <= countMinimumInstancesPerBin);
   EBM_ASSERT(0 == countInstances || 0 == countMaximumBins || nullptr != cutPointsLowerBoundInclusive);
   EBM_ASSERT(nullptr != countCutPoints);
   EBM_ASSERT(nullptr != isMissing);

   LOG_N(TraceLevelInfo, "Entered GenerateQuantileCutPoints: randomSeed=%" IntEbmTypePrintf ", countInstances=%" IntEbmTypePrintf 
      ", singleFeatureValues=%p, countMaximumBins=%" IntEbmTypePrintf ", countMinimumInstancesPerBin=%" IntEbmTypePrintf 
      ", cutPointsLowerBoundInclusive=%p, countCutPoints=%p, isMissingPresent=%p", 
      randomSeed, 
      countInstances, 
      static_cast<void *>(singleFeatureValues), 
      countMaximumBins, 
      countMinimumInstancesPerBin, 
      static_cast<void *>(cutPointsLowerBoundInclusive), 
      static_cast<void *>(countCutPoints),
      static_cast<void *>(isMissing)
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
   } else {
      FloatEbmType * pCopyTo = singleFeatureValues;
      FloatEbmType * pCopyFrom = singleFeatureValues;
      const FloatEbmType * const pCopyFromEnd = singleFeatureValues + cInstancesIncludingMissingValues;
      do {
         FloatEbmType val = *pCopyFrom;
         if(!std::isnan(val)) {
            *pCopyTo = val;
            ++pCopyTo;
         }
         ++pCopyFrom;
      } while(pCopyFromEnd != pCopyFrom);

      FloatEbmType * const pEnd = pCopyTo; // make it clear that this is our end now
      const size_t cInstances = pEnd - singleFeatureValues;

      const bool bMissing = pEnd != pCopyFrom;
      *isMissing = bMissing ? EBM_TRUE : EBM_FALSE;

      if(0 == cInstances) {
         *countCutPoints = 0;
      } else {
         try {
            std::sort(singleFeatureValues, pEnd);

            size_t cMaximumBins = static_cast<size_t>(countMaximumBins);
            if(bMissing) {
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
                  if(cBits == cMaximumBins) {
                     --cMaximumBins;
                     break;
                  }
                  cBits >>= 1;
                  // don't allow shrinkage below 16 bins (8 is the first power of two below 16).  By the time we reach 8 bins, we don't want to reduce this
                  // by a complete bin.  We can just use the extra bit for the missing bin
                  // if we had shrunk down to 7 bits for non-missing, we would have been able to fit in 16 items per data item instead of 21 for 64 bit systems
               } while(0x8 != cBits);
            }

            const size_t cMinimumInstancesPerBin = 
               countMinimumInstancesPerBin <= IntEbmType { 0 } ? size_t { 1 } : static_cast<size_t>(countMinimumInstancesPerBin);

            FloatEbmType avgLengthFloatDontUse = static_cast<FloatEbmType>(cInstances) / static_cast<FloatEbmType>(cMaximumBins);
            size_t avgLength = static_cast<size_t>(std::round(avgLengthFloatDontUse));
            if(avgLength < cMinimumInstancesPerBin) {
               avgLength = cMinimumInstancesPerBin;
            }
            EBM_ASSERT(1 <= avgLength);

            FloatEbmType rangeValue = *singleFeatureValues;
            FloatEbmType * pStartEqualRange = singleFeatureValues;
            FloatEbmType * pScan = singleFeatureValues + 1;
            size_t cJunctions = 1; // we always add a junction at the end
            while(pEnd != pScan) {
               const FloatEbmType val = *pScan;
               if(val != rangeValue) {
                  const size_t cEqualRangeItems = pScan - pStartEqualRange;
                  if(avgLength <= cEqualRangeItems) {
                     ++cJunctions;
                  }
                  rangeValue = val;
                  pStartEqualRange = pScan;
               }
               ++pScan;
            }

            RandomStream randomStream(randomSeed);
            if(!randomStream.IsSuccess()) {
               goto exit_error;
            }

            const size_t cBytesCombined = sizeof(Junction) + sizeof(Junction *);
            if(IsMultiplyError(cJunctions, cBytesCombined)) {
               goto exit_error;
            }
            // use the same memory allocation for both the Junction items and the pointers to the junctions that we'll use for sorting
            Junction ** const apJunctions = static_cast<Junction **>(malloc(cJunctions * cBytesCombined));
            if(nullptr == apJunctions) {
               goto exit_error;
            }
            Junction * const aJunctions = reinterpret_cast<Junction *>(apJunctions + cJunctions);

            rangeValue = *singleFeatureValues;
            FloatEbmType * pStartSplittableRange = singleFeatureValues;
            pStartEqualRange = singleFeatureValues;
            pScan = singleFeatureValues + 1;
            Junction * pJunction = aJunctions;
            Junction ** ppJunction = apJunctions;
            while(pEnd != pScan) {
               const FloatEbmType val = *pScan;
               if(val != rangeValue) {
                  const size_t cEqualRangeItems = pScan - pStartEqualRange;
                  if(avgLength <= cEqualRangeItems) {
                     // insert it even if there are zero cuttable items between two large ranges.  We want to know about the existance of the cuttable point
                     // and we do that via a zero item range.  Likewise for ranges with small numbers of items
                     EBM_ASSERT(pJunction < aJunctions + cJunctions);
                     pJunction->m_cItemsSplittableBefore = pStartEqualRange - pStartSplittableRange;
                     pJunction->m_pJunctionFirstUnsplittable = pStartEqualRange;
                     pJunction->m_cItemsUnsplittableAfter = cEqualRangeItems;
                     // no need to set junction.m_iSingleSplitBetweenLongRanges yet

                     EBM_ASSERT(ppJunction < apJunctions + cJunctions);
                     *ppJunction = pJunction;

                     ++pJunction;
                     ++ppJunction;
                     pStartSplittableRange = pScan;
                  }
                  rangeValue = val;
                  pStartEqualRange = pScan;
               }
               ++pScan;
            }
            size_t cEqualRangeItemsOuter = pEnd - pStartEqualRange;
            EBM_ASSERT(pJunction == aJunctions + cJunctions - 1);
            if(avgLength <= cEqualRangeItemsOuter) {
               // we have a long sequence.  Our previous items are a cuttable range

               // insert it even if there are zero cuttable items between two large ranges.  We want to know about the existance of the cuttable point
               // and we do that via a zero item range.  Likewise for ranges with small numbers of items
               pJunction->m_cItemsSplittableBefore = pStartEqualRange - pStartSplittableRange;
               pJunction->m_pJunctionFirstUnsplittable = pStartEqualRange;
               pJunction->m_cItemsUnsplittableAfter = cEqualRangeItemsOuter;
               // no need to set junction.m_iSingleSplitBetweenLongRanges yet
            } else {
               // we have a short sequence.  Our previous items are a cuttable range

               // insert it even if there are zero cuttable items between two large ranges.  We want to know about the existance of the cuttable point
               // and we do that via a zero item range.  Likewise for ranges with small numbers of items
               pJunction->m_cItemsSplittableBefore = pEnd - pStartSplittableRange;
               pJunction->m_pJunctionFirstUnsplittable = pEnd;
               pJunction->m_cItemsUnsplittableAfter = 0;
               // no need to set junction.m_iSingleSplitBetweenLongRanges yet
            }
            EBM_ASSERT(ppJunction == apJunctions + cJunctions - 1);
            *ppJunction = pJunction;

            SortJunctionsBySplittableAscending(&randomStream, cJunctions, apJunctions);
            SortJunctionsByUnsplittableDescending(&randomStream, cJunctions, apJunctions);



            free(apJunctions); // both the junctions and the pointers to the junctions are in the same memory allocation

            // first let's tackle the short ranges between big ranges (or at the tails) where we know there will be a split to separate the big ranges to either
            // side, but the short range isn't big enough to split.  In otherwords, there are less than cMinimumInstancesPerBin items
            // we start with the biggest long ranges and essentially try to push whatever mass there is away from them and continue down the list

            *countCutPoints = 0;
         } catch(...) {
            ret = 1;
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
   IntEbmType * isMissing
) {
   UNUSED(countInstances);
   UNUSED(singleFeatureValues);
   UNUSED(countMaximumBins);
   UNUSED(cutPointsLowerBoundInclusive);
   UNUSED(countCutPoints);
   UNUSED(isMissing);

   // TODO: IMPLEMENT

   return 0;
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateEqualWidthCutPoints(
   IntEbmType countInstances,
   FloatEbmType * singleFeatureValues,
   IntEbmType countMaximumBins,
   FloatEbmType * cutPointsLowerBoundInclusive,
   IntEbmType * countCutPoints,
   IntEbmType * isMissing
) {
   UNUSED(countInstances);
   UNUSED(singleFeatureValues);
   UNUSED(countMaximumBins);
   UNUSED(cutPointsLowerBoundInclusive);
   UNUSED(countCutPoints);
   UNUSED(isMissing);

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
