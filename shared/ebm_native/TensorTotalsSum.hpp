// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef TENSOR_TOTALS_SUM_HPP
#define TENSOR_TOTALS_SUM_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "Feature.hpp"
#include "FeatureGroup.hpp"

#include "HistogramTargetEntry.hpp"
#include "HistogramBucket.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

#ifndef NDEBUG

template<bool bClassification>
void TensorTotalsSumDebugSlow(
   const ptrdiff_t cClasses,
   const Term * const pTerm,
   const Bin<FloatBig, bClassification> * const aBins,
   const size_t * const aiStart,
   const size_t * const aiLast,
   Bin<FloatBig, bClassification> * const pRet
) {
   EBM_ASSERT(1 <= pTerm->GetCountSignificantDimensions()); // why bother getting totals if we just have 1 bin
   size_t aiDimensions[k_cDimensionsMax];

   size_t iTensorBin = 0;
   size_t valMultipleInitialize = 1;
   size_t iDimensionInitialize = 0;

   const TermEntry * pTermEntryInit = pTerm->GetTermEntries();
   const TermEntry * const pTermEntriesEnd = pTermEntryInit + pTerm->GetCountDimensions();
   do {
      const size_t cBins = pTermEntryInit->m_pFeature->GetCountBins();
      // cBins can only be 0 if there are zero training and zero validation samples
      // we don't boost or allow interaction updates if there are zero training samples
      EBM_ASSERT(size_t { 1 } <= cBins);
      if(size_t { 1 } < cBins) {
         EBM_ASSERT(aiStart[iDimensionInitialize] < cBins);
         EBM_ASSERT(aiLast[iDimensionInitialize] < cBins);
         EBM_ASSERT(aiStart[iDimensionInitialize] <= aiLast[iDimensionInitialize]);
         // aiStart[iDimensionInitialize] is less than cBins, so this should multiply
         EBM_ASSERT(!IsMultiplyError(valMultipleInitialize, aiStart[iDimensionInitialize]));
         iTensorBin += valMultipleInitialize * aiStart[iDimensionInitialize];
         EBM_ASSERT(!IsMultiplyError(valMultipleInitialize, cBins)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
         valMultipleInitialize *= cBins;
         aiDimensions[iDimensionInitialize] = aiStart[iDimensionInitialize];
         ++iDimensionInitialize;
      }
      ++pTermEntryInit;
   } while(pTermEntriesEnd != pTermEntryInit);

   const size_t cScores = GetCountScores(cClasses);
   // we've allocated this, so it should fit
   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores));
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);
   pRet->Zero(cBytesPerBin);

   const size_t cSignficantDimensions = pTerm->GetCountSignificantDimensions();

   while(true) {
      const auto * const pBin =
         IndexBin(cBytesPerBin, aBins, iTensorBin);

      pRet->Add(*pBin, cScores);

      size_t iDimension = 0;
      size_t valMultipleLoop = 1;
      const TermEntry * pTermEntry = pTerm->GetTermEntries();
      while(aiDimensions[iDimension] == aiLast[iDimension]) {
         EBM_ASSERT(aiStart[iDimension] <= aiLast[iDimension]);
         // we've allocated this memory, so it should be reachable, so these numbers should multiply
         EBM_ASSERT(!IsMultiplyError(valMultipleLoop, aiLast[iDimension] - aiStart[iDimension]));
         iTensorBin -= valMultipleLoop * (aiLast[iDimension] - aiStart[iDimension]);

         size_t cBins;
         do {
            cBins = pTermEntry->m_pFeature->GetCountBins();
            // cBins can only be 0 if there are zero training and zero validation samples
            // we don't boost or allow interaction updates if there are zero training samples
            EBM_ASSERT(size_t { 1 } <= cBins);
            ++pTermEntry;
         } while(cBins <= size_t { 1 }); // skip anything with 1 bin

         EBM_ASSERT(!IsMultiplyError(valMultipleLoop, cBins)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
         valMultipleLoop *= cBins;

         aiDimensions[iDimension] = aiStart[iDimension];
         ++iDimension;
         if(iDimension == cSignficantDimensions) {
            return;
         }
      }
      ++aiDimensions[iDimension];
      iTensorBin += valMultipleLoop;
   }
}

template<bool bClassification>
void TensorTotalsCompareDebug(
   const Bin<FloatBig, bClassification> * const aBins,
   const Term * const pTerm,
   const size_t * const aiPoint,
   const size_t directionVector,
   const ptrdiff_t cClasses,
   const Bin<FloatBig, bClassification> * const pComparison
) {
   const size_t cScores = GetCountScores(cClasses);
   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we're accessing allocated memory
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

   size_t aiStart[k_cDimensionsMax];
   size_t aiLast[k_cDimensionsMax];
   size_t directionVectorDestroy = directionVector;

   const TermEntry * pTermEntry = pTerm->GetTermEntries();
   const TermEntry * const pTermEntriesEnd = pTermEntry + pTerm->GetCountDimensions();

   size_t iDimensionDebug = 0;
   do {
      const size_t cBins = pTermEntry->m_pFeature->GetCountBins();
      // cBins can only be 0 if there are zero training and zero validation samples
      // we don't boost or allow interaction updates if there are zero training samples
      EBM_ASSERT(size_t { 1 } <= cBins);
      if(size_t { 1 } < cBins) {
         if(UNPREDICTABLE(0 != (1 & directionVectorDestroy))) {
            aiStart[iDimensionDebug] = aiPoint[iDimensionDebug] + 1;
            aiLast[iDimensionDebug] = cBins - 1;
         } else {
            aiStart[iDimensionDebug] = 0;
            aiLast[iDimensionDebug] = aiPoint[iDimensionDebug];
         }
         directionVectorDestroy >>= 1;
         ++iDimensionDebug;
      }
      ++pTermEntry;
   } while(pTermEntriesEnd != pTermEntry);

   auto * const pComparison2 = EbmMalloc<Bin<FloatBig, bClassification>>(1, cBytesPerBin);
   if(nullptr != pComparison2) {
      // if we can't obtain the memory, then don't do the comparison and exit
      TensorTotalsSumDebugSlow<bClassification>(
         cClasses,
         pTerm,
         aBins,
         aiStart,
         aiLast,
         pComparison2
         );
      EBM_ASSERT(pComparison->GetCountSamples() == pComparison2->GetCountSamples());
      free(pComparison2);
   }
}

#endif // NDEBUG

// TODO : we're not currently using cCompilerDimensions, so either use it or get rid of it
template<ptrdiff_t cCompilerClasses, size_t cCompilerDimensions>
void TensorTotalsSum(
   const ptrdiff_t cRuntimeClasses,
   const Term * const pTerm,
   const Bin<FloatBig, IsClassification(cCompilerClasses)> * const aBins,
   const size_t * const aiPoint,
   const size_t directionVector,
   Bin<FloatBig, IsClassification(cCompilerClasses)> * const pRet
#ifndef NDEBUG
   , const Bin<FloatBig, IsClassification(cCompilerClasses)> * const aBinsDebugCopy
   , const unsigned char * const pBinsEndDebug
#endif // NDEBUG
) {
   struct TotalsDimension {
      size_t m_cIncrement;
      size_t m_cLast;
   };

   constexpr bool bClassification = IsClassification(cCompilerClasses);

   // don't LOG this!  It would create way too much chatter!

   static_assert(k_cDimensionsMax < k_cBitsForSizeT, "reserve the highest bit for bit manipulation space");
   // TODO: I don't think I'm benefitting much here for pair code since the permute vector thing below won't
   //       be optimized away.  We should probably build special cases for this function for pairs (only 4 options
   //       in an if statement), and tripples (only 8 options in an if statement) and then keep this more general one 
   //       for higher dimensions

   const ptrdiff_t cClasses = GET_COUNT_CLASSES(
      cCompilerClasses,
      cRuntimeClasses
   );
   const size_t cScores = GetCountScores(cClasses);
   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we're accessing allocated memory
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

   size_t multipleTotalInitialize = 1;
   size_t startingOffset = 0;
   const TermEntry * pTermEntry = pTerm->GetTermEntries();
   EBM_ASSERT(1 <= pTerm->GetCountDimensions());
   const TermEntry * const pTermEntriesEnd = &pTermEntry[pTerm->GetCountDimensions()];
   const size_t * piPointInitialize = aiPoint;

   if(0 == directionVector) {
      // we would require a check in our inner loop below to handle the case of zero TermEntry items, so let's handle it separetly here instead
      do {
         const size_t cBins = pTermEntry->m_pFeature->GetCountBins();
         // cBins can only be 0 if there are zero training and zero validation samples
         // we don't boost or allow interaction updates if there are zero training samples
         EBM_ASSERT(size_t { 1 } <= cBins);
         if(size_t { 1 } < cBins) {
            EBM_ASSERT(*piPointInitialize < cBins);
            EBM_ASSERT(!IsMultiplyError(multipleTotalInitialize, *piPointInitialize)); // we're accessing allocated memory, so this needs to multiply
            const size_t addVal = multipleTotalInitialize * (*piPointInitialize);
            EBM_ASSERT(!IsAddError(startingOffset, addVal)); // we're accessing allocated memory, so this needs to add
            startingOffset += addVal;
            EBM_ASSERT(!IsMultiplyError(multipleTotalInitialize, cBins)); // we're accessing allocated memory, so this needs to multiply
            multipleTotalInitialize *= cBins;
            ++piPointInitialize;
         }
         ++pTermEntry;
      } while(LIKELY(pTermEntriesEnd != pTermEntry));
      const auto * const pBin = 
         IndexBin(cBytesPerBin, aBins, startingOffset);
      ASSERT_BIN_OK(cBytesPerBin, pRet, pBinsEndDebug);
      ASSERT_BIN_OK(cBytesPerBin, pBin, pBinsEndDebug);
      pRet->Copy(*pBin, cScores);
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
      do {
         const size_t cBins = pTermEntry->m_pFeature->GetCountBins();
         // cBins can only be 0 if there are zero training and zero validation samples
         // we don't boost or allow interaction updates if there are zero training samples
         EBM_ASSERT(size_t { 1 } <= cBins);
         if(size_t { 1 } < cBins) {
            if(UNPREDICTABLE(0 != (1 & directionVectorDestroy))) {
               EBM_ASSERT(!IsMultiplyError(multipleTotalInitialize, cBins - 1)); // we're accessing allocated memory, so this needs to multiply
               size_t cLast = multipleTotalInitialize * (cBins - 1);
               EBM_ASSERT(!IsMultiplyError(multipleTotalInitialize, *piPointInitialize)); // we're accessing allocated memory, so this needs to multiply
               pTotalsDimensionEnd->m_cIncrement = multipleTotalInitialize * (*piPointInitialize);
               pTotalsDimensionEnd->m_cLast = cLast;
               multipleTotalInitialize += cLast;
               ++pTotalsDimensionEnd;
            } else {
               EBM_ASSERT(!IsMultiplyError(multipleTotalInitialize, *piPointInitialize)); // we're accessing allocated memory, so this needs to multiply
               const size_t addVal = multipleTotalInitialize * (*piPointInitialize);
               EBM_ASSERT(!IsAddError(startingOffset, addVal)); // we're accessing allocated memory, so this needs to add
               startingOffset += addVal;
               multipleTotalInitialize *= cBins;
            }
            ++piPointInitialize;
            directionVectorDestroy >>= 1;
         }
         ++pTermEntry;
      } while(LIKELY(pTermEntriesEnd != pTermEntry));
   }
   const unsigned int cAllBits = static_cast<unsigned int>(pTotalsDimensionEnd - totalsDimension);
   EBM_ASSERT(cAllBits < k_cBitsForSizeT);

   pRet->Zero(cBytesPerBin);

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
      // TODO : eliminate this multiplication of cBytesPerBin by offsetPointer by multiplying both the startingOffset and the 
      // m_cLast & m_cIncrement values by cBytesPerBin.  We can eliminate this multiplication each loop!
      const auto * const pBin =
         IndexBin(cBytesPerBin, aBins, offsetPointer);
      // TODO : we can eliminate this really bad unpredictable branch if we use conditional negation on the values in pBin.  
      // We can pass in a bool that indicates if we should take the negation value or the original at each step 
      // (so we don't need to store it beyond one value either).  We would then have an Add(bool bSubtract, ...) function
      if(UNPREDICTABLE(0 != (1 & evenOdd))) {
         ASSERT_BIN_OK(cBytesPerBin, pRet, pBinsEndDebug);
         ASSERT_BIN_OK(cBytesPerBin, pBin, pBinsEndDebug);
         pRet->Subtract(*pBin, cScores);
      } else {
         ASSERT_BIN_OK(cBytesPerBin, pRet, pBinsEndDebug);
         ASSERT_BIN_OK(cBytesPerBin, pBin, pBinsEndDebug);
         pRet->Add(*pBin, cScores);
      }
      ++permuteVector;
   } while(LIKELY(0 == (permuteVector >> cAllBits)));

#ifndef NDEBUG
   if(nullptr != aBinsDebugCopy) {
      TensorTotalsCompareDebug<bClassification>(
         aBinsDebugCopy,
         pTerm,
         aiPoint,
         directionVector,
         cClasses,
         pRet
         );
   }
#endif // NDEBUG
}

} // DEFINED_ZONE_NAME

#endif // TENSOR_TOTALS_SUM_HPP