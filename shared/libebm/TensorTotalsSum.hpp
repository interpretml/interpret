// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#ifndef TENSOR_TOTALS_SUM_HPP
#define TENSOR_TOTALS_SUM_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT

#include "bridge.hpp" // GetArrayScores
#include "GradientPair.hpp"
#include "Bin.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct TensorSumDimension {
   size_t m_iLow;
   size_t m_iHigh;
   size_t m_cBins;
};

#ifndef NDEBUG
#ifdef CHECK_TENSORS

template<bool bHessian>
void TensorTotalsSumDebugSlow(const size_t cScores,
      const size_t cDimensions,
      const size_t* const aiStart,
      const size_t* const aiLast,
      const size_t* const acBins,
      const Bin<FloatMain, UIntMain, true, true, bHessian>* const aBins,
      Bin<FloatMain, UIntMain, true, true, bHessian>& binOut) {
   // we've allocated this, so it should fit
   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);

   EBM_ASSERT(1 <= cDimensions); // why bother getting totals if we just have 1 bin
   size_t aiDimensions[k_cDimensionsMax];

   size_t iTensorByte = 0;
   size_t cTensorBytesInitialize = cBytesPerBin;
   size_t iDimensionInitialize = 0;

   const size_t* pcBinsInit = acBins;
   const size_t* const pcBinsInitEnd = &acBins[cDimensions];
   do {
      const size_t cBins = *pcBinsInit;
      // cBins can only be 0 if there are zero training and zero validation samples
      // we don't boost or allow interaction updates if there are zero training samples
      EBM_ASSERT(size_t{1} <= cBins);
      EBM_ASSERT(aiStart[iDimensionInitialize] < cBins);
      EBM_ASSERT(aiLast[iDimensionInitialize] < cBins);
      EBM_ASSERT(aiStart[iDimensionInitialize] <= aiLast[iDimensionInitialize]);
      // aiStart[iDimensionInitialize] is less than cBins, so this should multiply
      EBM_ASSERT(!IsMultiplyError(cTensorBytesInitialize, aiStart[iDimensionInitialize]));
      iTensorByte += cTensorBytesInitialize * aiStart[iDimensionInitialize];
      EBM_ASSERT(!IsMultiplyError(cTensorBytesInitialize,
            cBins)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
      cTensorBytesInitialize *= cBins;
      aiDimensions[iDimensionInitialize] = aiStart[iDimensionInitialize];
      ++iDimensionInitialize;

      ++pcBinsInit;
   } while(pcBinsInitEnd != pcBinsInit);

   binOut.ZeroMem(cBytesPerBin);

   while(true) {
      const auto* const pBin = IndexBin(aBins, iTensorByte);

      binOut.Add(cScores, *pBin);

      size_t iDimension = 0;
      size_t cTensorBytesLoop = cBytesPerBin;
      const size_t* pcBins = acBins;
      while(aiDimensions[iDimension] == aiLast[iDimension]) {
         EBM_ASSERT(aiStart[iDimension] <= aiLast[iDimension]);
         // we've allocated this memory, so it should be reachable, so these numbers should multiply
         EBM_ASSERT(!IsMultiplyError(cTensorBytesLoop, aiLast[iDimension] - aiStart[iDimension]));
         iTensorByte -= cTensorBytesLoop * (aiLast[iDimension] - aiStart[iDimension]);

         const size_t cBins = *pcBins;
         EBM_ASSERT(size_t{1} <= cBins);
         ++pcBins;

         EBM_ASSERT(!IsMultiplyError(cTensorBytesLoop,
               cBins)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
         cTensorBytesLoop *= cBins;

         aiDimensions[iDimension] = aiStart[iDimension];
         ++iDimension;
         if(iDimension == cDimensions) {
            return;
         }
      }
      ++aiDimensions[iDimension];
      iTensorByte += cTensorBytesLoop;
   }
}

template<bool bHessian>
void TensorTotalsCompareDebug(const size_t cScores,
      const size_t cRealDimensions,
      const TensorSumDimension* const aDimensions,
      const Bin<FloatMain, UIntMain, true, true, bHessian>* const aBins,
      const Bin<FloatMain, UIntMain, true, true, bHessian>& bin,
      const GradientPair<FloatMain, bHessian>* const aGradientPairs) {
   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);

   size_t acBins[k_cDimensionsMax];
   size_t aiStart[k_cDimensionsMax];
   size_t aiLast[k_cDimensionsMax];

   size_t iDimension = 0;
   do {
      aiStart[iDimension] = aDimensions[iDimension].m_iLow;
      aiLast[iDimension] = aDimensions[iDimension].m_iHigh - 1;
      const size_t cBins = aDimensions[iDimension].m_cBins;

      acBins[iDimension] = cBins;
      EBM_ASSERT(size_t{1} <= cBins);
      ++iDimension;
   } while(cRealDimensions != iDimension);

   auto* const pComparison2 = static_cast<Bin<FloatMain, UIntMain, true, true, bHessian>*>(malloc(cBytesPerBin));
   if(nullptr != pComparison2) {
      // if we can't obtain the memory, then don't do the comparison and exit
      TensorTotalsSumDebugSlow<bHessian>(cScores, cRealDimensions, aiStart, aiLast, acBins, aBins, *pComparison2);
      EBM_ASSERT(pComparison2->GetCountSamples() == bin.GetCountSamples());
      UNUSED(aGradientPairs);
      free(pComparison2);
   }
}

#endif // CHECK_TENSORS
#endif // NDEBUG

template<bool bHessian, size_t cCompilerScores>
INLINE_ALWAYS static void TensorTotalsSumMulti(const size_t cRuntimeScores,
      const size_t cDimensions,
      const TensorSumDimension* const aDimensions,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const aBins,
      Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>& binOut,
      GradientPair<FloatMain, bHessian>* const aGradientPairsOut
#ifndef NDEBUG
      ,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const aDebugCopyBins,
      const BinBase* const pBinsEndDebug
#endif // NDEBUG
) {
   struct TotalsDimension {
      size_t m_cIncrement;
      size_t m_cLast;
   };

   static_assert(k_cDimensionsMax < COUNT_BITS(size_t), "reserve the highest bit for bit manipulation space");

   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, cRuntimeScores);
   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);

   EBM_ASSERT(1 <= cDimensions); // for interactions, we just return 0 for interactions with zero features
   EBM_ASSERT(cDimensions <= k_cDimensionsMax);

   size_t cTensorBytesInitialize = cBytesPerBin;
   const unsigned char* pStartingBin = reinterpret_cast<const unsigned char*>(aBins);

   TotalsDimension totalsDimension[k_cDimensionsMax];
   TotalsDimension* pTotalsDimensionEnd = totalsDimension;
   {
      size_t iDimension = 0;
      do {
         const size_t iLow = aDimensions[iDimension].m_iLow;
         const size_t iHigh = aDimensions[iDimension].m_iHigh;
         const size_t cBins = aDimensions[iDimension].m_cBins;

         EBM_ASSERT(size_t{1} <= cBins); // we can handle useless dimensions
         EBM_ASSERT(iLow < cBins);
         EBM_ASSERT(iHigh <= cBins);
         EBM_ASSERT(iLow < iHigh);

         if(UNPREDICTABLE(0 != iLow)) {
            // we're accessing allocated memory, so this needs to multiply
            EBM_ASSERT(!IsMultiplyError(cTensorBytesInitialize, cBins - 1));
            pTotalsDimensionEnd->m_cIncrement = cTensorBytesInitialize * (iLow - 1);
            pTotalsDimensionEnd->m_cLast = cTensorBytesInitialize * (iHigh - 1);
            cTensorBytesInitialize += cTensorBytesInitialize * (cBins - 1);
            ++pTotalsDimensionEnd;
         } else {
            pStartingBin += cTensorBytesInitialize * (iHigh - 1);
            cTensorBytesInitialize *= cBins;
         }
         ++iDimension;
      } while(LIKELY(cDimensions != iDimension));
   }
   const int cProcessingDimensions = static_cast<int>(pTotalsDimensionEnd - totalsDimension);
   if(0 == cProcessingDimensions) {
      // this is a special case where we only need to lookup the answer
      const auto* const pBin =
            reinterpret_cast<const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>*>(
                  pStartingBin);
      ASSERT_BIN_OK(cBytesPerBin, pBin, pBinsEndDebug);
      binOut.Copy(cScores, *pBin, pBin->GetGradientPairs(), aGradientPairsOut);
      return;
   }

   EBM_ASSERT(cProcessingDimensions < COUNT_BITS(size_t));
   EBM_ASSERT(static_cast<size_t>(cProcessingDimensions) <= cDimensions);
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
      const unsigned char* pRawBin = pStartingBin;
      size_t evenOdd = 0;
      size_t dimensionFlagsDestroy = static_cast<size_t>(dimensionFlags);
      const TotalsDimension* pTotalsDimensionLoop = totalsDimension;
      do {
         evenOdd ^= dimensionFlagsDestroy; // flip least significant bit if the dimension bit is set
         // TODO: check if it's faster to load both m_cLast and m_cIncrement instead of selecting the right
         // address and loading it.  Loading both would be more prefetch predictable for the CPU
         pRawBin += *(UNPREDICTABLE(0 == (1 & dimensionFlagsDestroy)) ? &pTotalsDimensionLoop->m_cLast :
                                                                        &pTotalsDimensionLoop->m_cIncrement);
         dimensionFlagsDestroy >>= 1;
         ++pTotalsDimensionLoop;
      } while(LIKELY(pTotalsDimensionEnd != pTotalsDimensionLoop));

      const auto* const pBin =
            reinterpret_cast<const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>*>(
                  pRawBin);

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
            cScores, cDimensions, aDimensions, aDebugCopyBins->Downgrade(), *binOut.Downgrade(), aGradientPairsOut);
   }
#endif // CHECK_TENSORS
#endif // NDEBUG
}

template<bool bHessian, size_t cCompilerScores>
INLINE_ALWAYS static void TensorTotalsSumTripple(const size_t cRuntimeScores,
      const TensorSumDimension* const aDimensions,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const aBins,
      Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>& binOut,
      GradientPair<FloatMain, bHessian>* const aGradientPairsOut
#ifndef NDEBUG
      ,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const aDebugCopyBins,
      const BinBase* const pBinsEndDebug
#endif // NDEBUG
) {
   // TODO: make a tripple specific version of this function
   TensorTotalsSumMulti<bHessian, cCompilerScores>(cRuntimeScores,
         3,
         aDimensions,
         aBins,
         binOut,
         aGradientPairsOut
#ifndef NDEBUG
         ,
         aDebugCopyBins,
         pBinsEndDebug
#endif // NDEBUG
   );
}

template<bool bHessian, size_t cCompilerScores>
INLINE_ALWAYS static void TensorTotalsSumPair(const size_t cRuntimeScores,
      const TensorSumDimension* const aDimensions,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const aBins,
      Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>& binOut,
      GradientPair<FloatMain, bHessian>* const aGradientPairsOut
#ifndef NDEBUG
      ,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const aDebugCopyBins,
      const BinBase* const pBinsEndDebug
#endif // NDEBUG
) {
   // TODO: make a pair specific version of this function
   //       For pairs, we'd probably be better off if we did the original thing where we put 4 co-located Bins
   //       (low, low), (low, high), (high, low), (high, high) and then just use the bin demaned by the
   //       directionVector.  Our algorithm below works well for higher dimensions where this blows up quickly
   //       but doing it the way below really randomizes memory accesses.
   TensorTotalsSumMulti<bHessian, cCompilerScores>(cRuntimeScores,
         2,
         aDimensions,
         aBins,
         binOut,
         aGradientPairsOut
#ifndef NDEBUG
         ,
         aDebugCopyBins,
         pBinsEndDebug
#endif // NDEBUG
   );
}

template<bool bHessian, size_t cCompilerScores, size_t cCompilerDimensions>
INLINE_ALWAYS static void TensorTotalsSum(const size_t cRuntimeScores,
      const size_t cRuntimeDimensions,
      const TensorSumDimension* const aDimensions,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const aBins,
      Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>& binOut,
      GradientPair<FloatMain, bHessian>* const aGradientPairsOut
#ifndef NDEBUG
      ,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const aDebugCopyBins,
      const BinBase* const pBinsEndDebug
#endif // NDEBUG
) {
   static constexpr bool bPair = (2 == cCompilerDimensions);
   static constexpr bool bTripple = (3 == cCompilerDimensions);
   if(bPair) {
      EBM_ASSERT(2 == cRuntimeDimensions);
      TensorTotalsSumPair<bHessian, cCompilerScores>(cRuntimeScores,
            aDimensions,
            aBins,
            binOut,
            aGradientPairsOut
#ifndef NDEBUG
            ,
            aDebugCopyBins,
            pBinsEndDebug
#endif // NDEBUG
      );
   } else if(bTripple) {
      EBM_ASSERT(3 == cRuntimeDimensions);
      TensorTotalsSumTripple<bHessian, cCompilerScores>(cRuntimeScores,
            aDimensions,
            aBins,
            binOut,
            aGradientPairsOut
#ifndef NDEBUG
            ,
            aDebugCopyBins,
            pBinsEndDebug
#endif // NDEBUG
      );
   } else {
      TensorTotalsSumMulti<bHessian, cCompilerScores>(cRuntimeScores,
            cRuntimeDimensions,
            aDimensions,
            aBins,
            binOut,
            aGradientPairsOut
#ifndef NDEBUG
            ,
            aDebugCopyBins,
            pBinsEndDebug
#endif // NDEBUG
      );
   }
}

} // namespace DEFINED_ZONE_NAME

#endif // TENSOR_TOTALS_SUM_HPP