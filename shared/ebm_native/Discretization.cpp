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
   // and a long range of identical values on the right that are unsplittable
   FloatEbmType * m_pJunctionFirstUnsplittable;
   size_t         m_cItemsSplittableBefore; // this can be zero
   size_t         m_cItemsUnsplittableAfter;

   size_t         m_iSingleSplitBetweenLongRanges;
};

void SortJunctionsByUnsplittable(RandomStream & randomStream, std::vector<Junction> & junctions) {
   EBM_ASSERT(0 < junctions.size());

   // sort first by number of items, but if the number of items is equal, use the pointer as a stand in for the index so that the order is stable
   // accross all implementations.  We'll use random numbers afterwards to make the ordering fair regarding position
   std::sort(junctions.begin(), junctions.end(), [](Junction & junction1, Junction & junction2) {
      if(junction1.m_cItemsUnsplittableAfter == junction2.m_cItemsUnsplittableAfter) {
         return junction1.m_pJunctionFirstUnsplittable > junction2.m_pJunctionFirstUnsplittable;
      } else {
         return junction1.m_cItemsUnsplittableAfter > junction2.m_cItemsUnsplittableAfter;
      }
   });

   // find sections that have the same number of items and randomly shuffle the sections with equal numbers of items so that there is no directional preference
   size_t iStartEqualLengthRange = 0;
   size_t cItems = junctions[0].m_cItemsUnsplittableAfter;
   for(size_t i = 1; i < junctions.size(); ++i) {
      const size_t cNewItems = junctions[i].m_cItemsUnsplittableAfter;
      if(cItems != cNewItems) {
         // we have a real range
         size_t cRemainingItems = i - iStartEqualLengthRange;
         while(1 != cRemainingItems) {
            const size_t iSwap = randomStream.Next(cRemainingItems);
            if(0 != iSwap) {
               Junction tmp = junctions[iStartEqualLengthRange];
               junctions[iStartEqualLengthRange] = junctions[iSwap + iStartEqualLengthRange];
               junctions[iSwap + iStartEqualLengthRange] = tmp;
            }
            ++iStartEqualLengthRange;
            --cRemainingItems;
         }
         iStartEqualLengthRange = i;
         cItems = cNewItems;
      }
   }
   size_t cRemainingItemsOuter = junctions.size() - iStartEqualLengthRange;
   while(1 != cRemainingItemsOuter) {
      const size_t iSwap = randomStream.Next(cRemainingItemsOuter);
      if(0 != iSwap) {
         Junction tmp = junctions[iStartEqualLengthRange];
         junctions[iStartEqualLengthRange] = junctions[iSwap + iStartEqualLengthRange];
         junctions[iSwap + iStartEqualLengthRange] = tmp;
      }
      ++iStartEqualLengthRange;
      --cRemainingItemsOuter;
   }
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateDiscretizationCutPoints(
   const IntEbmType randomSeed,
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

   LOG_N(TraceLevelInfo, "Entered GenerateDiscretizationCutPoints: randomSeed=%" IntEbmTypePrintf ", countInstances=%" IntEbmTypePrintf 
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
      LOG_0(TraceLevelWarning, "WARNING GenerateDiscretizationCutPoints !IsNumberConvertable<size_t, IntEbmType>(countInstances)");
      return 1;
   }

   if(!IsNumberConvertable<size_t, IntEbmType>(countMaximumBins)) {
      LOG_0(TraceLevelWarning, "WARNING GenerateDiscretizationCutPoints !IsNumberConvertable<size_t, IntEbmType>(countMaximumBins)");
      return 1;
   }

   if(!IsNumberConvertable<size_t, IntEbmType>(countMinimumInstancesPerBin)) {
      LOG_0(TraceLevelWarning, "WARNING GenerateDiscretizationCutPoints !IsNumberConvertable<size_t, IntEbmType>(countMinimumInstancesPerBin)");
      return 1;
   }

   const size_t cInstancesIncludingMissingValues = static_cast<size_t>(countInstances);
   const size_t cMinimumInstancesPerBin = countMinimumInstancesPerBin <= IntEbmType { 0 } ? size_t { 1 } : static_cast<size_t>(countMinimumInstancesPerBin);

   IntEbmType ret = 0;
   if(0 == cInstancesIncludingMissingValues) {
      *countCutPoints = 0;
      *isMissing = EBM_FALSE;
   } else {
      try {
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

         const bool bMissing = pEnd != pCopyFrom;
         *isMissing = bMissing ? EBM_TRUE : EBM_FALSE;
         std::sort(singleFeatureValues, pEnd);

         // if we have 16 max bins requested AND missing values, reduce our bins so that we can use one of the 16 bins for the zero
         // but if the user has a maximum less than 16 that isn't a power of two, like 15, then we don't need to reduce our bins to zero since we'll bit pack
         // nicely anyways.
         // if the user requests something like 8 though, we'll just increase it to 9 if there's a zero bin since we don't want to go below a certain point
         // as the bins become more important
         const size_t cMaximumBins = static_cast<size_t>(countMaximumBins) - 
            (bMissing && IntEbmType { 15 } < countMaximumBins ? size_t { 1 } : size_t { 0 });

         const size_t cInstances = pEnd - singleFeatureValues;

         FloatEbmType avgLengthFloatDontUse = static_cast<FloatEbmType>(cInstances) / static_cast<FloatEbmType>(cMaximumBins);
         size_t avgLength = static_cast<size_t>(std::round(avgLengthFloatDontUse));
         if(avgLength < cMinimumInstancesPerBin) {
            avgLength = cMinimumInstancesPerBin;
         }
         EBM_ASSERT(1 <= avgLength);

         std::vector<Junction> junctions;

         FloatEbmType rangeValue = *singleFeatureValues;
         FloatEbmType * pStartSplittableRange = singleFeatureValues;
         FloatEbmType * pStartEqualRange = singleFeatureValues;
         FloatEbmType * pScan = singleFeatureValues + 1;
         while(pEnd != pScan) {
            FloatEbmType val = *pScan;
            if(val != rangeValue) {
               size_t cEqualRangeItems = pScan - pStartEqualRange;
               if(avgLength <= cEqualRangeItems) {
                  // we have a long sequence.  Our previous items are a cuttable range
                  Junction junction;
                  // insert it even if there are zero cuttable items between two large ranges.  We want to know about the existance of the cuttable point
                  // and we do that via a zero item range.  Likewise for ranges with small numbers of items
                  junction.m_cItemsSplittableBefore = pStartEqualRange - pStartSplittableRange;
                  junction.m_pJunctionFirstUnsplittable = pStartEqualRange;
                  junction.m_cItemsUnsplittableAfter = cEqualRangeItems;
                  // no need to set junction.m_iSingleSplitBetweenLongRanges yet
                  junctions.push_back(junction);

                  pStartSplittableRange = pScan;
               }
               rangeValue = val;
               pStartEqualRange = pScan;
            }
            ++pScan;
         }
         size_t cEqualRangeItemsOuter = pEnd - pStartEqualRange;
         if(avgLength <= cEqualRangeItemsOuter) {
            // we have a long sequence.  Our previous items are a cuttable range
            Junction junction;
            // insert it even if there are zero cuttable items between two large ranges.  We want to know about the existance of the cuttable point
            // and we do that via a zero item range.  Likewise for ranges with small numbers of items
            junction.m_cItemsSplittableBefore = pStartEqualRange - pStartSplittableRange;
            junction.m_pJunctionFirstUnsplittable = pStartEqualRange;
            junction.m_cItemsUnsplittableAfter = cEqualRangeItemsOuter;
            // no need to set junction.m_iSingleSplitBetweenLongRanges yet
            junctions.push_back(junction);
         } else {
            // we have a short sequence.  Our previous items are a cuttable range
            Junction junction;
            // insert it even if there are zero cuttable items between two large ranges.  We want to know about the existance of the cuttable point
            // and we do that via a zero item range.  Likewise for ranges with small numbers of items
            junction.m_cItemsSplittableBefore = pEnd - pStartSplittableRange;
            junction.m_pJunctionFirstUnsplittable = pEnd;
            junction.m_cItemsUnsplittableAfter = 0;
            // no need to set junction.m_iSingleSplitBetweenLongRanges yet
            junctions.push_back(junction);
         }

         RandomStream randomStream(randomSeed);
         if(!randomStream.IsSuccess()) {
            goto exit_error;
         }

         EBM_ASSERT(0 < junctions.size());
         SortJunctionsByUnsplittable(randomStream, junctions);

         // first let's tackle the short ranges between big ranges (or at the tails) where we know there will be a split to separate the big ranges to either
         // side, but the short range isn't big enough to split.  In otherwords, there are less than cMinimumInstancesPerBin items



      } catch(...) {
         ret = 1;
      }
   }
   if(0 != ret) {
      LOG_N(TraceLevelWarning, "WARNING GenerateDiscretizationCutPoints returned %" IntEbmTypePrintf, ret);
   } else {
      LOG_N(TraceLevelInfo, "Exited GenerateDiscretizationCutPoints countCutPoints=%" IntEbmTypePrintf ", isMissing=%" IntEbmTypePrintf,
         *countCutPoints,
         *isMissing
      );
   }
   return ret;

exit_error:;
   ret = 1;
   LOG_N(TraceLevelWarning, "WARNING GenerateDiscretizationCutPoints returned %" IntEbmTypePrintf, ret);
   return ret;
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

   LOG_N(TraceLevelInfo, "Entered Discretize: isMissing=%" IntEbmTypePrintf ", countCutPoints=%" IntEbmTypePrintf 
      ", cutPointsLowerBoundInclusive=%p, countInstances=%" IntEbmTypePrintf ", singleFeatureValues=%p, singleFeatureDiscretized=%p",
      isMissing,
      countCutPoints,
      static_cast<const void *>(cutPointsLowerBoundInclusive),
      countInstances,
      static_cast<const void *>(singleFeatureValues),
      static_cast<void *>(singleFeatureDiscretized)
   );

   if(0 < countInstances) {
      const ptrdiff_t missingVal = EBM_FALSE != isMissing ? ptrdiff_t { 0 } : ptrdiff_t { -1 };
      const size_t cCutPoints = static_cast<size_t>(countCutPoints);
      const size_t cInstances = static_cast<size_t>(countInstances);
      const FloatEbmType * pValue = singleFeatureValues;
      const FloatEbmType * pValueEnd = singleFeatureValues + countInstances;
      IntEbmType * pDiscretized = singleFeatureDiscretized;

      if(0 == countCutPoints) {
         memset(singleFeatureDiscretized, 0, sizeof(singleFeatureDiscretized[0]) * cInstances);
      } else {
         do {
            ptrdiff_t middle = missingVal;
            FloatEbmType val = *pValue;
            if(!std::isnan(val)) {
               ptrdiff_t high = cCutPoints - 1;
               ptrdiff_t low = 0;
               FloatEbmType midVal;
               do {
                  middle = (low + high) >> 1;
                  midVal = cutPointsLowerBoundInclusive[middle];
                  if(UNLIKELY(midVal == val)) {
                     // this happens just once during our descent, so it's less likely than continuing searching

                     // TODO: getting exactly equal should be rare for floating points, especially since our cut points are in between the floats
                     //       that we do get.  Can we modify the descent algorithm so that it handles this without a special exit jump

                     goto no_check;
                  }
                  high = UNPREDICTABLE(midVal < val) ? high : middle - 1;
                  low = UNPREDICTABLE(midVal < val) ? middle + 1 : low;
               } while(LIKELY(low <= high));
               middle = UNPREDICTABLE(midVal < val) ? middle + 1 : middle;
            }
         no_check:
            *pDiscretized = static_cast<IntEbmType>(middle);
            ++pDiscretized;
            ++pValue;
         } while(pValueEnd != pValue);
      }
   }

   LOG_0(TraceLevelInfo, "Exited Discretize");
}
