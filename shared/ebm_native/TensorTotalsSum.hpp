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
#include "Term.hpp"

#include "GradientPair.hpp"
#include "Bin.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

#ifndef NDEBUG

template<bool bClassification>
void TensorTotalsSumDebugSlow(
   const ptrdiff_t cClasses,
   const size_t cRealDimensions,
   const size_t * const acBins,
   const Bin<FloatBig, bClassification> * const aBins,
   const size_t * const aiStart,
   const size_t * const aiLast,
   Bin<FloatBig, bClassification> * const pRet
) {
   const size_t cScores = GetCountScores(cClasses);
   // we've allocated this, so it should fit
   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores));
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

   EBM_ASSERT(1 <= cRealDimensions); // why bother getting totals if we just have 1 bin
   size_t aiDimensions[k_cDimensionsMax];

   size_t iTensorByte = 0;
   size_t cTensorBytesInitialize = cBytesPerBin;
   size_t iDimensionInitialize = 0;

   const size_t * pcBinsInit = acBins;
   const size_t * const pcBinsInitEnd = &acBins[cRealDimensions];
   do {
      const size_t cBins = *pcBinsInit;
      // cBins can only be 0 if there are zero training and zero validation samples
      // we don't boost or allow interaction updates if there are zero training samples
      EBM_ASSERT(size_t { 2 } <= cBins);
      EBM_ASSERT(aiStart[iDimensionInitialize] < cBins);
      EBM_ASSERT(aiLast[iDimensionInitialize] < cBins);
      EBM_ASSERT(aiStart[iDimensionInitialize] <= aiLast[iDimensionInitialize]);
      // aiStart[iDimensionInitialize] is less than cBins, so this should multiply
      EBM_ASSERT(!IsMultiplyError(cTensorBytesInitialize, aiStart[iDimensionInitialize]));
      iTensorByte += cTensorBytesInitialize * aiStart[iDimensionInitialize];
      EBM_ASSERT(!IsMultiplyError(cTensorBytesInitialize, cBins)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
      cTensorBytesInitialize *= cBins;
      aiDimensions[iDimensionInitialize] = aiStart[iDimensionInitialize];
      ++iDimensionInitialize;

      ++pcBinsInit;
   } while(pcBinsInitEnd != pcBinsInit);

   pRet->ZeroMem(cBytesPerBin);

   while(true) {
      const auto * const pBin = IndexBin(aBins, iTensorByte);

      pRet->Add(*pBin, cScores);

      size_t iDimension = 0;
      size_t cTensorBytesLoop = cBytesPerBin;
      const size_t * pcBins = acBins;
      while(aiDimensions[iDimension] == aiLast[iDimension]) {
         EBM_ASSERT(aiStart[iDimension] <= aiLast[iDimension]);
         // we've allocated this memory, so it should be reachable, so these numbers should multiply
         EBM_ASSERT(!IsMultiplyError(cTensorBytesLoop, aiLast[iDimension] - aiStart[iDimension]));
         iTensorByte -= cTensorBytesLoop * (aiLast[iDimension] - aiStart[iDimension]);

         const size_t cBins = *pcBins;
         EBM_ASSERT(size_t { 2 } <= cBins);
         ++pcBins;

         EBM_ASSERT(!IsMultiplyError(cTensorBytesLoop, cBins)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
         cTensorBytesLoop *= cBins;

         aiDimensions[iDimension] = aiStart[iDimension];
         ++iDimension;
         if(iDimension == cRealDimensions) {
            return;
         }
      }
      ++aiDimensions[iDimension];
      iTensorByte += cTensorBytesLoop;
   }
}

template<bool bClassification>
void TensorTotalsCompareDebug(
   const Bin<FloatBig, bClassification> * const aBins,
   const size_t cRealDimensions,
   const size_t * const acBins,
   const size_t * const aiPoint,
   const size_t directionVector,
   const ptrdiff_t cClasses,
   const size_t & cSamples,
   const FloatBig & weight,
   const GradientPair<FloatBig, bClassification> * const aGradientPairs
) {
   const size_t cScores = GetCountScores(cClasses);
   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we're accessing allocated memory
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

   size_t aiStart[k_cDimensionsMax];
   size_t aiLast[k_cDimensionsMax];
   size_t directionVectorDestroy = directionVector;

   const size_t * pcBins = acBins;
   const size_t * const pcBinsEnd = &acBins[cRealDimensions];

   size_t iDimensionDebug = 0;
   do {
      const size_t cBins = *pcBins;
      // cBins can only be 0 if there are zero training and zero validation samples
      // we don't boost or allow interaction updates if there are zero training samples
      EBM_ASSERT(size_t { 2 } <= cBins);
      if(UNPREDICTABLE(0 != (1 & directionVectorDestroy))) {
         aiStart[iDimensionDebug] = aiPoint[iDimensionDebug] + 1;
         aiLast[iDimensionDebug] = cBins - 1;
      } else {
         aiStart[iDimensionDebug] = 0;
         aiLast[iDimensionDebug] = aiPoint[iDimensionDebug];
      }
      directionVectorDestroy >>= 1;
      ++iDimensionDebug;
      ++pcBins;
   } while(pcBinsEnd != pcBins);

   auto * const pComparison2 = EbmMalloc<Bin<FloatBig, bClassification>>(1, cBytesPerBin);
   if(nullptr != pComparison2) {
      // if we can't obtain the memory, then don't do the comparison and exit
      TensorTotalsSumDebugSlow<bClassification>(
         cClasses,
         cRealDimensions,
         acBins,
         aBins,
         aiStart,
         aiLast,
         pComparison2
      );
      EBM_ASSERT(pComparison2->GetCountSamples() == cSamples);
      UNUSED(weight);
      UNUSED(aGradientPairs);
      //EBM_ASSERT(pComparison2->IsBinClose(cSamples, weight, aGradientPairs, cScores));
      free(pComparison2);
   }
}

#endif // NDEBUG

template<ptrdiff_t cCompilerClasses>
INLINE_ALWAYS static void TensorTotalsSumMulti(
   const ptrdiff_t cRuntimeClasses,
   const size_t cRealDimensions,
   const size_t * const acBins,
   const Bin<FloatBig, IsClassification(cCompilerClasses)> * const aBins,
   const size_t * const aiPoint,
   const size_t directionVector,
   size_t & cSamplesOut,
   FloatBig & weightOut,
   GradientPair<FloatBig, IsClassification(cCompilerClasses)> * const aGradientPairsOut
#ifndef NDEBUG
   , const Bin<FloatBig, IsClassification(cCompilerClasses)> * const aDebugCopyBins
   , const unsigned char * const pBinsEndDebug
#endif // NDEBUG
) {
   struct TotalsDimension {
      size_t m_cIncrement;
      size_t m_cLast;
   };

   constexpr bool bClassification = IsClassification(cCompilerClasses);

   static_assert(k_cDimensionsMax < k_cBitsForSizeT, "reserve the highest bit for bit manipulation space");

   const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, cRuntimeClasses);
   const size_t cScores = GetCountScores(cClasses);
   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we're accessing allocated memory
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

   EBM_ASSERT(1 <= cRealDimensions); // for interactions, we just return 0 for interactions with zero features

   size_t cTensorBytesInitialize = cBytesPerBin;
   const unsigned char * pStartingBin = reinterpret_cast<const unsigned char *>(aBins);

   if(0 == directionVector) {
      // we would require a check in our inner loop below to handle the case of zero Features, so let's handle it separetly here instead

      size_t iDimension = 0;
      do {
         const size_t cBins = acBins[iDimension];
         const size_t iPoint = aiPoint[iDimension];

         EBM_ASSERT(size_t { 2 } <= cBins);
         EBM_ASSERT(iPoint < cBins);
         EBM_ASSERT(!IsMultiplyError(cTensorBytesInitialize, iPoint)); // we're accessing allocated memory
         const size_t addVal = cTensorBytesInitialize * (iPoint);
         pStartingBin += addVal;
         EBM_ASSERT(!IsMultiplyError(cTensorBytesInitialize, cBins)); // we're accessing allocated memory
         cTensorBytesInitialize *= cBins;

         ++iDimension;
      } while(LIKELY(cRealDimensions != iDimension));
      const auto * const pBin = reinterpret_cast<const Bin<FloatBig, bClassification> *>(pStartingBin);
      ASSERT_BIN_OK(cBytesPerBin, pBin, pBinsEndDebug);
      pBin->CopyTo(cSamplesOut, weightOut, aGradientPairsOut, cScores);
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
      size_t iDimension = 0;
      do {
         const size_t cBins = acBins[iDimension];
         const size_t iPoint = aiPoint[iDimension];

         EBM_ASSERT(size_t { 2 } <= cBins);
         EBM_ASSERT(iPoint < cBins);

         EBM_ASSERT(!IsMultiplyError(cTensorBytesInitialize, iPoint)); // we're accessing allocated memory
         const size_t addVal = cTensorBytesInitialize * iPoint;

         if(UNPREDICTABLE(0 != (1 & directionVectorDestroy))) {
            EBM_ASSERT(!IsMultiplyError(cTensorBytesInitialize, cBins - 1)); // we're accessing allocated memory, so this needs to multiply
            size_t cLast = cTensorBytesInitialize * (cBins - 1);
            pTotalsDimensionEnd->m_cIncrement = addVal;
            pTotalsDimensionEnd->m_cLast = cLast;
            cTensorBytesInitialize += cLast;
            ++pTotalsDimensionEnd;
         } else {
            pStartingBin += addVal;
            cTensorBytesInitialize *= cBins;
         }
         directionVectorDestroy >>= 1;

         ++iDimension;
      } while(LIKELY(cRealDimensions != iDimension));
   }
   const unsigned int cProcessingDimensions = static_cast<unsigned int>(pTotalsDimensionEnd - totalsDimension);
   EBM_ASSERT(cProcessingDimensions < k_cBitsForSizeT);

   cSamplesOut = 0;
   weightOut = 0;

   EBM_ASSERT(1 <= cScores);
   size_t iScore = 0;
   do {
      aGradientPairsOut[iScore].Zero();
      ++iScore;
   } while(cScores != iScore);

   size_t dimensionFlags = 0;
   do {
      const unsigned char * pRawBin = pStartingBin;
      size_t evenOdd = cProcessingDimensions;
      size_t dimensionFlagsDestroy = dimensionFlags;
      const TotalsDimension * pTotalsDimensionLoop = totalsDimension;
      do {
         evenOdd ^= dimensionFlagsDestroy; // flip least significant bit if the dimension bit is set
         // TODO: check if it's faster to load both m_cLast and m_cIncrement instead of selecting the right
         // address and loading it.  Loading both would be more prefetch predictable for the CPU
         pRawBin += *(UNPREDICTABLE(0 != (1 & dimensionFlagsDestroy)) ? 
            &pTotalsDimensionLoop->m_cLast : &pTotalsDimensionLoop->m_cIncrement);
         dimensionFlagsDestroy >>= 1;
         ++pTotalsDimensionLoop;
      } while(LIKELY(pTotalsDimensionEnd != pTotalsDimensionLoop));

      const auto * const pBin = reinterpret_cast<const Bin<FloatBig, bClassification> *>(pRawBin);

      if(UNPREDICTABLE(0 != (1 & evenOdd))) {
         ASSERT_BIN_OK(cBytesPerBin, pBin, pBinsEndDebug);
         pBin->SubtractTo(cSamplesOut, weightOut, aGradientPairsOut, cScores);
      } else {
         ASSERT_BIN_OK(cBytesPerBin, pBin, pBinsEndDebug);
         pBin->AddTo(cSamplesOut, weightOut, aGradientPairsOut, cScores);
      }
      ++dimensionFlags;
   } while(LIKELY(0 == (dimensionFlags >> cProcessingDimensions)));

#ifndef NDEBUG
   if(nullptr != aDebugCopyBins) {
      TensorTotalsCompareDebug<bClassification>(
         aDebugCopyBins,
         cRealDimensions,
         acBins,
         aiPoint,
         directionVector,
         cClasses,
         cSamplesOut,
         weightOut,
         aGradientPairsOut
      );
   }
#endif // NDEBUG
}

template<ptrdiff_t cCompilerClasses>
INLINE_ALWAYS static void TensorTotalsSumTripple(
   const ptrdiff_t cRuntimeClasses,
   const size_t * const acBins,
   const Bin<FloatBig, IsClassification(cCompilerClasses)> * const aBins,
   const size_t * const aiPoint,
   const size_t directionVector,
   size_t & cSamplesOut,
   FloatBig & weightOut,
   GradientPair<FloatBig, IsClassification(cCompilerClasses)> * const aGradientPairsOut
#ifndef NDEBUG
   , const Bin<FloatBig, IsClassification(cCompilerClasses)> * const aDebugCopyBins
   , const unsigned char * const pBinsEndDebug
#endif // NDEBUG
) {
   // TODO: make a tripple specific version of this function
   TensorTotalsSumMulti<cCompilerClasses>(
      cRuntimeClasses,
      3,
      acBins,
      aBins,
      aiPoint,
      directionVector,
      cSamplesOut,
      weightOut,
      aGradientPairsOut
#ifndef NDEBUG
      , aDebugCopyBins
      , pBinsEndDebug
#endif // NDEBUG
   );
}

template<ptrdiff_t cCompilerClasses>
INLINE_ALWAYS static void TensorTotalsSumPair(
   const ptrdiff_t cRuntimeClasses,
   const size_t * const acBins,
   const Bin<FloatBig, IsClassification(cCompilerClasses)> * const aBins,
   const size_t * const aiPoint,
   const size_t directionVector,
   size_t & cSamplesOut,
   FloatBig & weightOut,
   GradientPair<FloatBig, IsClassification(cCompilerClasses)> * const aGradientPairsOut
#ifndef NDEBUG
   , const Bin<FloatBig, IsClassification(cCompilerClasses)> * const aDebugCopyBins
   , const unsigned char * const pBinsEndDebug
#endif // NDEBUG
) {
   // TODO: make a pair specific version of this function
   //       For pairs, we'd probably be better off if we did the original thing where we put 4 co-located Bins
   //       (low, low), (low, high), (high, low), (high, high) and then just use the bin demaned by the 
   //       directionVector.  Our algorithm below works well for higher dimensions where this blows up quickly
   //       but doing it the way below really randomizes memory accesses.
   TensorTotalsSumMulti<cCompilerClasses>(
      cRuntimeClasses,
      2,
      acBins,
      aBins,
      aiPoint,
      directionVector,
      cSamplesOut,
      weightOut,
      aGradientPairsOut
#ifndef NDEBUG
      , aDebugCopyBins
      , pBinsEndDebug
#endif // NDEBUG
   );
}

template<ptrdiff_t cCompilerClasses, size_t cCompilerDimensions>
INLINE_ALWAYS static void TensorTotalsSum(
   const ptrdiff_t cRuntimeClasses,
   const size_t cRuntimeRealDimensions,
   const size_t * const acBins,
   const Bin<FloatBig, IsClassification(cCompilerClasses)> * const aBins,
   const size_t * const aiPoint,
   const size_t directionVector,
   size_t & cSamplesOut,
   FloatBig & weightOut,
   GradientPair<FloatBig, IsClassification(cCompilerClasses)> * const aGradientPairsOut
#ifndef NDEBUG
   , const Bin<FloatBig, IsClassification(cCompilerClasses)> * const aDebugCopyBins
   , const unsigned char * const pBinsEndDebug
#endif // NDEBUG
) {
   constexpr bool bPair = (2 == cCompilerDimensions);
   constexpr bool bTripple = (3 == cCompilerDimensions);
   if(bPair) {
      EBM_ASSERT(2 == cRuntimeRealDimensions);
      TensorTotalsSumPair<cCompilerClasses>(
         cRuntimeClasses,
         acBins,
         aBins,
         aiPoint,
         directionVector,
         cSamplesOut,
         weightOut,
         aGradientPairsOut
#ifndef NDEBUG
         , aDebugCopyBins
         , pBinsEndDebug
#endif // NDEBUG
      );
   } else if(bTripple) {
      EBM_ASSERT(3 == cRuntimeRealDimensions);
      TensorTotalsSumTripple<cCompilerClasses>(
         cRuntimeClasses,
         acBins,
         aBins,
         aiPoint,
         directionVector,
         cSamplesOut,
         weightOut,
         aGradientPairsOut
#ifndef NDEBUG
         , aDebugCopyBins
         , pBinsEndDebug
#endif // NDEBUG
      );
   } else {
      EBM_ASSERT(2 != cRuntimeRealDimensions && 3 != cRuntimeRealDimensions);
      TensorTotalsSumMulti<cCompilerClasses>(
         cRuntimeClasses,
         cRuntimeRealDimensions,
         acBins,
         aBins,
         aiPoint,
         directionVector,
         cSamplesOut,
         weightOut,
         aGradientPairsOut
#ifndef NDEBUG
         , aDebugCopyBins
         , pBinsEndDebug
#endif // NDEBUG
      );
   }
}


} // DEFINED_ZONE_NAME

#endif // TENSOR_TOTALS_SUM_HPP