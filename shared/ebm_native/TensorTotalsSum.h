// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef TENSOR_TOTALS_SUM_H
#define TENSOR_TOTALS_SUM_H

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "EbmInternal.h"
// very independent includes
#include "Logging.h" // EBM_ASSERT & LOG

#include "FeatureAtomic.h"
#include "FeatureGroup.h"

#include "HistogramTargetEntry.h"
#include "HistogramBucket.h"

extern void TensorTotalsBuild(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const FeatureGroup * const pFeatureGroup,
   HistogramBucketBase * pBucketAuxiliaryBuildZone,
   HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
   , HistogramBucketBase * const aHistogramBucketsDebugCopy
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
);

#ifndef NDEBUG

template<bool bClassification>
void TensorTotalsSumDebugSlow(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const FeatureGroup * const pFeatureGroup,
   const HistogramBucket<bClassification> * const aHistogramBuckets,
   const size_t * const aiStart,
   const size_t * const aiLast,
   HistogramBucket<bClassification> * const pRet
) {
   const size_t cDimensions = pFeatureGroup->GetCountFeatures();
   EBM_ASSERT(1 <= cDimensions); // why bother getting totals if we just have 1 bin
   size_t aiDimensions[k_cDimensionsMax];

   size_t iTensorBin = 0;
   size_t valueMultipleInitialize = 1;
   size_t iDimensionInitialize = 0;
   do {
      const size_t cBins = pFeatureGroup->GetFeatureGroupEntries()[iDimensionInitialize].m_pFeature->GetCountBins();
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
   EBM_ASSERT(!GetHistogramBucketSizeOverflow(bClassification, cVectorLength));
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);
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

         const size_t cBins = pFeatureGroup->GetFeatureGroupEntries()[iDimension].m_pFeature->GetCountBins();
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
void TensorTotalsCompareDebug(
   const HistogramBucket<bClassification> * const aHistogramBuckets,
   const FeatureGroup * const pFeatureGroup,
   const size_t * const aiPoint,
   const size_t directionVector,
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const HistogramBucket<bClassification> * const pComparison
) {
   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow(bClassification, cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);

   size_t aiStart[k_cDimensionsMax];
   size_t aiLast[k_cDimensionsMax];
   size_t directionVectorDestroy = directionVector;
   for(size_t iDimensionDebug = 0; iDimensionDebug < pFeatureGroup->GetCountFeatures(); ++iDimensionDebug) {
      const size_t cBins = pFeatureGroup->GetFeatureGroupEntries()[iDimensionDebug].m_pFeature->GetCountBins();
      if(UNPREDICTABLE(0 != (1 & directionVectorDestroy))) {
         aiStart[iDimensionDebug] = aiPoint[iDimensionDebug] + 1;
         aiLast[iDimensionDebug] = cBins - 1;
      } else {
         aiStart[iDimensionDebug] = 0;
         aiLast[iDimensionDebug] = aiPoint[iDimensionDebug];
      }
      directionVectorDestroy >>= 1;
   }

   HistogramBucket<bClassification> * const pComparison2 = EbmMalloc<HistogramBucket<bClassification>>(1, cBytesPerHistogramBucket);
   if(nullptr != pComparison2) {
      // if we can't obtain the memory, then don't do the comparison and exit
      TensorTotalsSumDebugSlow<bClassification>(
         runtimeLearningTypeOrCountTargetClasses,
         pFeatureGroup,
         aHistogramBuckets,
         aiStart,
         aiLast,
         pComparison2
         );
      EBM_ASSERT(pComparison->GetCountSamplesInBucket() == pComparison2->GetCountSamplesInBucket());
      free(pComparison2);
   }
}

#endif // NDEBUG

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerCountDimensions>
void TensorTotalsSum(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const FeatureGroup * const pFeatureGroup,
   const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets,
   const size_t * const aiPoint,
   const size_t directionVector,
   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pRet
#ifndef NDEBUG
   , const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   struct TotalsDimension {
      size_t m_cIncrement;
      size_t m_cLast;
   };

   constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

   // don't LOG this!  It would create way too much chatter!

   static_assert(k_cDimensionsMax < k_cBitsForSizeT, "reserve the highest bit for bit manipulation space");
   const size_t cDimensions = GET_ATTRIBUTE_COMBINATION_DIMENSIONS(compilerCountDimensions, pFeatureGroup->GetCountFeatures());
   EBM_ASSERT(1 <= cDimensions);
   EBM_ASSERT(cDimensions < k_cBitsForSizeT);

   const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
      compilerLearningTypeOrCountTargetClasses,
      runtimeLearningTypeOrCountTargetClasses
   );
   const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow(bClassification, cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);

   size_t multipleTotalInitialize = 1;
   size_t startingOffset = 0;
   const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
   const FeatureGroupEntry * const pFeatureGroupEntryEnd = &pFeatureGroupEntry[cDimensions];
   const size_t * piPointInitialize = aiPoint;

   if(0 == directionVector) {
      // we would require a check in our inner loop below to handle the case of zero FeatureGroupEntry items, so let's handle it separetly here instead
      EBM_ASSERT(1 <= cDimensions);
      do {
         size_t cBins = pFeatureGroupEntry->m_pFeature->GetCountBins();
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
         ++pFeatureGroupEntry;
         ++piPointInitialize;
      } while(LIKELY(pFeatureGroupEntryEnd != pFeatureGroupEntry));
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
         size_t cBins = pFeatureGroupEntry->m_pFeature->GetCountBins();
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
         ++pFeatureGroupEntry;
         ++piPointInitialize;
         directionVectorDestroy >>= 1;
      } while(LIKELY(pFeatureGroupEntryEnd != pFeatureGroupEntry));
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
      TensorTotalsCompareDebug<bClassification>(
         aHistogramBucketsDebugCopy,
         pFeatureGroup,
         aiPoint,
         directionVector,
         runtimeLearningTypeOrCountTargetClasses,
         pRet
         );
   }
#endif // NDEBUG
}

#endif // TENSOR_TOTALS_SUM_H