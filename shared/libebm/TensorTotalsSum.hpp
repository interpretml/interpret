// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef TENSOR_TOTALS_SUM_HPP
#define TENSOR_TOTALS_SUM_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT
#include "zones.h"

#include "bridge_cpp.hpp" // GetArrayScores
#include "GradientPair.hpp"
#include "Bin.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct TensorSumDimension {
   size_t m_iPoint;
   size_t m_cBins;
};

#ifndef NDEBUG
#ifdef CHECK_TENSORS

template<bool bHessian>
void TensorTotalsSumDebugSlow(
   const size_t cScores,
   const size_t cRealDimensions,
   const size_t * const aiStart,
   const size_t * const aiLast,
   const size_t * const acBins,
   const Bin<FloatMain, UIntMain, bHessian> * const aBins,
   Bin<FloatMain, UIntMain, bHessian> & binOut
) {
   // we've allocated this, so it should fit
   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(bHessian, cScores);

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

   binOut.ZeroMem(cBytesPerBin);

   while(true) {
      const auto * const pBin = IndexBin(aBins, iTensorByte);

      binOut.Add(cScores, *pBin);

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

template<bool bHessian>
void TensorTotalsCompareDebug(
   const size_t cScores,
   const size_t cRealDimensions,
   const TensorSumDimension * const aDimensions,
   const size_t directionVector,
   const Bin<FloatMain, UIntMain, bHessian> * const aBins,
   const Bin<FloatMain, UIntMain, bHessian> & bin,
   const GradientPair<FloatMain, bHessian> * const aGradientPairs
) {
   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(bHessian, cScores);

   size_t acBins[k_cDimensionsMax];
   size_t aiStart[k_cDimensionsMax];
   size_t aiLast[k_cDimensionsMax];
   size_t directionVectorDestroy = directionVector;

   size_t iDimension = 0;
   do {
      const size_t iPoint = aDimensions[iDimension].m_iPoint;
      const size_t cBins = aDimensions[iDimension].m_cBins;

      acBins[iDimension] = cBins;

      EBM_ASSERT(size_t { 2 } <= cBins);
      if(UNPREDICTABLE(0 != (1 & directionVectorDestroy))) {
         aiStart[iDimension] = iPoint + 1;
         aiLast[iDimension] = cBins - 1;
      } else {
         aiStart[iDimension] = 0;
         aiLast[iDimension] = iPoint;
      }
      directionVectorDestroy >>= 1;
      ++iDimension;
   } while(cRealDimensions != iDimension);

   auto * const pComparison2 = static_cast<Bin<FloatMain, UIntMain, bHessian> *>(malloc(cBytesPerBin));
   if(nullptr != pComparison2) {
      // if we can't obtain the memory, then don't do the comparison and exit
      TensorTotalsSumDebugSlow<bHessian>(
         cScores,
         cRealDimensions,
         aiStart,
         aiLast,
         acBins,
         aBins,
         *pComparison2
      );
      EBM_ASSERT(pComparison2->GetCountSamples() == bin.GetCountSamples());
      UNUSED(aGradientPairs);
      free(pComparison2);
   }
}

#endif // CHECK_TENSORS
#endif // NDEBUG

template<bool bHessian, size_t cCompilerScores>
INLINE_ALWAYS static void TensorTotalsSumMulti(
   const size_t cRuntimeScores,
   const size_t cRealDimensions,
   const TensorSumDimension * const aDimensions,
   const size_t directionVector,
   const Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> * const aBins,
   Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> & binOut,
   GradientPair<FloatMain, bHessian> * const aGradientPairsOut
#ifndef NDEBUG
   , const Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> * const aDebugCopyBins
   , const BinBase * const pBinsEndDebug
#endif // NDEBUG
) {
   // TODO: build a version of this function that can get the sum of any interior volume.  This function
   //       currently only allows us to get the sum from any point to the edge boundaries of the tensor
   //       The algorithm to extend this to get any interor volume 
   //       eg: (start_x, start_y, start_z) to (end_x, end_y, end_z) is similar to always getting the far
   //       end volume.  For a tripple where we're getting the far end vector directionVector = (1, 1, 1) 
   //       we always start from the last Bin that contains the total sum and ablate the planes of each
   //       dimension combination, then the "tubes", then the origin cube.  When we have a starting and ending
   //       point, we treat the ending point like we do below for the sum total last Bin.  If we look at
   //       the point (end_x, end_y, end_z) it contains the sum up until that point, and we can pretend/ignore
   //       all points past that location and then apply our existing algorithm to ablate the volumes 
   //       related to (start_x, start_y, start_z) as if we were getting the far side cube directionVector = (1, 1, 1) 

   struct TotalsDimension {
      size_t m_cIncrement;
      size_t m_cLast;
   };

   static_assert(k_cDimensionsMax < COUNT_BITS(size_t), "reserve the highest bit for bit manipulation space");

   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, cRuntimeScores);
   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(bHessian, cScores);

   EBM_ASSERT(1 <= cRealDimensions); // for interactions, we just return 0 for interactions with zero features
   EBM_ASSERT(cRealDimensions <= k_cDimensionsMax);

   size_t cTensorBytesInitialize = cBytesPerBin;
   const unsigned char * pStartingBin = reinterpret_cast<const unsigned char *>(aBins);

   if(0 == directionVector) {
      // we would require a check in our inner loop below to handle the case of zero Features, so let's handle it separetly here instead

      size_t iDimension = 0;
      do {
         const size_t iPoint = aDimensions[iDimension].m_iPoint;
         const size_t cBins = aDimensions[iDimension].m_cBins;

         EBM_ASSERT(size_t { 2 } <= cBins);
         EBM_ASSERT(iPoint < cBins);
         EBM_ASSERT(!IsMultiplyError(cTensorBytesInitialize, iPoint)); // we're accessing allocated memory
         const size_t addVal = cTensorBytesInitialize * iPoint;
         pStartingBin += addVal;
         EBM_ASSERT(!IsMultiplyError(cTensorBytesInitialize, cBins)); // we're accessing allocated memory
         cTensorBytesInitialize *= cBins;

         ++iDimension;
      } while(LIKELY(cRealDimensions != iDimension));
      const auto * const pBin = reinterpret_cast<const Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> *>(pStartingBin);
      ASSERT_BIN_OK(cBytesPerBin, pBin, pBinsEndDebug);
      binOut.Copy(cScores, *pBin, pBin->GetGradientPairs(), aGradientPairsOut);
      return;
   }

   //this is a fast way of determining the number of bits (see if the are faster algorithms.. CPU hardware or expoential shifting potentially).  
   // We may use it in the future if we're trying to decide whether to go from (0,0,...,0,0) or (1,1,...,1,1)
   //int cBits = 0;
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
         const size_t iPoint = aDimensions[iDimension].m_iPoint;
         const size_t cBins = aDimensions[iDimension].m_cBins;

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
   const int cProcessingDimensions = static_cast<int>(pTotalsDimensionEnd - totalsDimension);
   EBM_ASSERT(cProcessingDimensions < COUNT_BITS(size_t));
   EBM_ASSERT(static_cast<size_t>(cProcessingDimensions) <= cRealDimensions);
   EBM_ASSERT(1 <= cProcessingDimensions);
   // The Clang static analyer just knows that directionVectorDestroy is not zero in the loop above, but it doesn't
   // know if some of the very high bits are set or some of the low ones, so when it iterates by the number of
   // dimesions it doesn't know if it'll hit any bits that would increment pTotalsDimensionEnd.  If pTotalsDimensionEnd
   // is never incremented, then cProcessingDimensions would be zero, and the shift below would become equal to
   // COUNT_BITS(size_t), which would be illegal
   ANALYSIS_ASSERT(0 != cProcessingDimensions);

   binOut.Zero(cScores, aGradientPairsOut);

   // for every dimension that we're processing, set the dimension bit flag to 1 to start
   ptrdiff_t dimensionFlags = static_cast<ptrdiff_t>(MakeLowMask<size_t>(cProcessingDimensions));
   do {
      const unsigned char * pRawBin = pStartingBin;
      size_t evenOdd = 0;
      size_t dimensionFlagsDestroy = static_cast<size_t>(dimensionFlags);
      const TotalsDimension * pTotalsDimensionLoop = totalsDimension;
      do {
         evenOdd ^= dimensionFlagsDestroy; // flip least significant bit if the dimension bit is set
         // TODO: check if it's faster to load both m_cLast and m_cIncrement instead of selecting the right
         // address and loading it.  Loading both would be more prefetch predictable for the CPU
         pRawBin += *(UNPREDICTABLE(0 == (1 & dimensionFlagsDestroy)) ? 
            &pTotalsDimensionLoop->m_cLast : &pTotalsDimensionLoop->m_cIncrement);
         dimensionFlagsDestroy >>= 1;
         ++pTotalsDimensionLoop;
      } while(LIKELY(pTotalsDimensionEnd != pTotalsDimensionLoop));

      const auto * const pBin = reinterpret_cast<const Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> *>(pRawBin);

      // TODO: for pairs and tripples and anything else that we want to make special case code for we can
      // avoid this unpredictable branch, which would be very helpful
      if(UNPREDICTABLE(0 != (1 & evenOdd))) {
         ASSERT_BIN_OK(cBytesPerBin, pBin, pBinsEndDebug);
         binOut.Subtract(cScores, *pBin, pBin->GetGradientPairs(), aGradientPairsOut);
      } else {
         ASSERT_BIN_OK(cBytesPerBin, pBin, pBinsEndDebug);
         binOut.Add(cScores, *pBin, pBin->GetGradientPairs(), aGradientPairsOut);
      }
      --dimensionFlags;
   } while(LIKELY(0 <= dimensionFlags));

#ifndef NDEBUG
   UNUSED(aDebugCopyBins);
#ifdef CHECK_TENSORS
   if(nullptr != aDebugCopyBins) {
      TensorTotalsCompareDebug<bHessian>(
         cScores,
         cRealDimensions,
         aDimensions,
         directionVector,
         aDebugCopyBins->Downgrade(),
         *binOut.Downgrade(),
         aGradientPairsOut
      );
   }
#endif // CHECK_TENSORS
#endif // NDEBUG
}

template<bool bHessian, size_t cCompilerScores>
INLINE_ALWAYS static void TensorTotalsSumTripple(
   const size_t cRuntimeScores,
   const TensorSumDimension * const aDimensions,
   const size_t directionVector,
   const Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> * const aBins,
   Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> & binOut,
   GradientPair<FloatMain, bHessian> * const aGradientPairsOut
#ifndef NDEBUG
   , const Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> * const aDebugCopyBins
   , const BinBase * const pBinsEndDebug
#endif // NDEBUG
) {
   // TODO: make a tripple specific version of this function
   TensorTotalsSumMulti<bHessian, cCompilerScores>(
      cRuntimeScores,
      3,
      aDimensions,
      directionVector,
      aBins,
      binOut,
      aGradientPairsOut
#ifndef NDEBUG
      , aDebugCopyBins
      , pBinsEndDebug
#endif // NDEBUG
   );
}

template<bool bHessian, size_t cCompilerScores>
INLINE_ALWAYS static void TensorTotalsSumPair(
   const size_t cRuntimeScores,
   const TensorSumDimension * const aDimensions,
   const size_t directionVector,
   const Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> * const aBins,
   Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> & binOut,
   GradientPair<FloatMain, bHessian> * const aGradientPairsOut
#ifndef NDEBUG
   , const Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> * const aDebugCopyBins
   , const BinBase * const pBinsEndDebug
#endif // NDEBUG
) {
   // TODO: make a pair specific version of this function
   //       For pairs, we'd probably be better off if we did the original thing where we put 4 co-located Bins
   //       (low, low), (low, high), (high, low), (high, high) and then just use the bin demaned by the 
   //       directionVector.  Our algorithm below works well for higher dimensions where this blows up quickly
   //       but doing it the way below really randomizes memory accesses.
   TensorTotalsSumMulti<bHessian, cCompilerScores>(
      cRuntimeScores,
      2,
      aDimensions,
      directionVector,
      aBins,
      binOut,
      aGradientPairsOut
#ifndef NDEBUG
      , aDebugCopyBins
      , pBinsEndDebug
#endif // NDEBUG
   );
}

template<bool bHessian, size_t cCompilerScores, size_t cCompilerDimensions>
INLINE_ALWAYS static void TensorTotalsSum(
   const size_t cRuntimeScores,
   const size_t cRuntimeRealDimensions,
   const TensorSumDimension * const aDimensions,
   const size_t directionVector,
   const Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> * const aBins,
   Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> & binOut,
   GradientPair<FloatMain, bHessian> * const aGradientPairsOut
#ifndef NDEBUG
   , const Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> * const aDebugCopyBins
   , const BinBase * const pBinsEndDebug
#endif // NDEBUG
) {
   static constexpr bool bPair = (2 == cCompilerDimensions);
   static constexpr bool bTripple = (3 == cCompilerDimensions);
   if(bPair) {
      EBM_ASSERT(2 == cRuntimeRealDimensions);
      TensorTotalsSumPair<bHessian, cCompilerScores>(
         cRuntimeScores,
         aDimensions,
         directionVector,
         aBins,
         binOut,
         aGradientPairsOut
#ifndef NDEBUG
         , aDebugCopyBins
         , pBinsEndDebug
#endif // NDEBUG
      );
   } else if(bTripple) {
      EBM_ASSERT(3 == cRuntimeRealDimensions);
      TensorTotalsSumTripple<bHessian, cCompilerScores>(
         cRuntimeScores,
         aDimensions,
         directionVector,
         aBins,
         binOut,
         aGradientPairsOut
#ifndef NDEBUG
         , aDebugCopyBins
         , pBinsEndDebug
#endif // NDEBUG
      );
   } else {
      EBM_ASSERT(2 != cRuntimeRealDimensions && 3 != cRuntimeRealDimensions);
      TensorTotalsSumMulti<bHessian, cCompilerScores>(
         cRuntimeScores,
         cRuntimeRealDimensions,
         aDimensions,
         directionVector,
         aBins,
         binOut,
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