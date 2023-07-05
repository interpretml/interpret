// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef BIN_SUMS_INTERACTION_HPP
#define BIN_SUMS_INTERACTION_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT
#include "zones.h"

#include "common_cpp.hpp" // k_cDimensionsMax
#include "bridge_cpp.hpp" // BinSumsInteractionBridge
#include "GradientPair.hpp" // GradientPair
#include "Bin.hpp" // Bin

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME


//  TODO: Notes on SIMD-ifying
//
//- Let's say we have 8/16 SIMD-streams.  We'll be adding gradients into tensors with these, but
//  if two of the SIMD streams collide in their index, then we'd have a problem with adding. We therefore need to add into
//  SEPARATE tensors (one per SIMD-stream) and then add the final tensors together at the end.
//- Let's say we have a binary feature, so we're packing 64 bits into a single bit pack.  When we access the gradients array
//  we don't want to advance by 64 * 4 * 16 = 4096 bytes per access.  What we can do is carefully locate the bit packs such
//  that we access the non-bit packed data sequentially.  To do this we would locate the first item into the highest bits of the
//  first bitpack, and the second item into the highest bits of the second bitpack.  If we have 16 SIMD streams then we'll
//  load 16 bitpacks at once, and we'll load the next sequential 16 gradient floats at the same time.  Then we'll use the
//  highest bits of the first bitpack to index the first tensor, and the highest bits of the second bitpack and the second float
//  to update the second tensor.  This way we sequentially load both the bitpacks and the floats.
//- Similar to Boosting, we'll need two different kinds of loops.  We'll have a SIMD-optimized one and CPU one that specializes in the "dregs"
//  The "dregs" one will will be for situations where we won't want to load 8/16 in a SIMD-cluster.  We'll do them 1 at a time in a CPU loop
//  Since we're going to be araning the bits in the bitpack by the size of the SIMD-cluster, we'll need to have a different layout for
//  these last dregs
//- on AVX-512 with 16 parallel streams, we only need to process the last 15 items on the CPU. We can start mid-way through
//  each bit pack even if they start from different points
//- since we'll need to pack the bits differently depending on the type of SIMD.  We can proceed as follows:
//  - move entirely towards using 32-bit floats AND build infrastructure to allow for adding together tensors AND being able to process
//    separate datasets.  We'll need this to combine the separate SIMD tensors and to combine the CPU processed data from the SIMD processed data
//  - build a separate SIMD-specialized part of the dataset, or a new dataset that packs bits in the way that we want for our particular SIMD-cluster size
//  - keeping our existing code as-is, copy our exising code into a SIMD-only specialized set of loops in the compute part of the code and start
//    passing clean sets of data that is in our new SIMD-specific datasets.  We'll use the existing code to handle CPU
//  - allow the system to process all the data via CPU (which means it can be inside a single dataset) and compare this result to the result
//    of using the SIMD code pipeline.  Maybe we can simulate all the same access 

template<typename TFloat, bool bHessian, size_t cCompilerScores, size_t cCompilerDimensions, bool bWeight>
static void BinSumsInteractionInternal(BinSumsInteractionBridge * const pParams) {
   static constexpr size_t cArrayScores = GetArrayScores(cCompilerScores);

   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pParams->m_cScores);

   auto * const aBins = reinterpret_cast<BinBase *>(pParams->m_aFastBins)->Specialize<FloatFast, StorageDataType, bHessian, cArrayScores>();
   EBM_ASSERT(nullptr != aBins);

   const size_t cSamples = pParams->m_cSamples;
   EBM_ASSERT(1 <= cSamples);

   const FloatFast * pGradientAndHessian = pParams->m_aGradientsAndHessians;
   const FloatFast * const pGradientsAndHessiansEnd = pGradientAndHessian + (bHessian ? 2 : 1) * cScores * cSamples;

   struct DimensionalData {
      ptrdiff_t m_cShift;
      size_t m_cBitsPerItemMax;
      StorageDataType m_iTensorBinCombined;
      size_t m_maskBits;
      size_t m_cBins;
      const StorageDataType * m_pData;
      ptrdiff_t m_cShiftReset;
   };

   const size_t cRealDimensions = GET_COUNT_DIMENSIONS(cCompilerDimensions, pParams->m_cRuntimeRealDimensions);
   EBM_ASSERT(1 <= cRealDimensions); // for interactions, we just return 0 for interactions with zero features

   EBM_ASSERT(1 == cCompilerDimensions || 1 != pParams->m_cRuntimeRealDimensions); // 1 dimension must be templated

   // this is on the stack and the compiler should be able to optimize these as if they were variables or registers
   DimensionalData aDimensionalData[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
   size_t iDimensionInit = 0;
   do {
      DimensionalData * const pDimensionalData = &aDimensionalData[iDimensionInit];

      const StorageDataType * const pData = pParams->m_aaPacked[iDimensionInit];
      pDimensionalData->m_iTensorBinCombined = *pData;
      pDimensionalData->m_pData = pData + 1;

      const size_t cItemsPerBitPack = pParams->m_acItemsPerBitPack[iDimensionInit];
      EBM_ASSERT(1 <= cItemsPerBitPack);
      EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);

      const size_t cBitsPerItemMax = GetCountBits<StorageDataType>(cItemsPerBitPack);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      pDimensionalData->m_cBitsPerItemMax = cBitsPerItemMax;

      pDimensionalData->m_cShift = static_cast<ptrdiff_t>(((cSamples - 1) % cItemsPerBitPack + 1) * cBitsPerItemMax);
      pDimensionalData->m_cShiftReset = static_cast<ptrdiff_t>((cItemsPerBitPack - 1) * cBitsPerItemMax);

      const size_t maskBits = static_cast<size_t>(MakeLowMask<StorageDataType>(cBitsPerItemMax));
      pDimensionalData->m_maskBits = maskBits;

      pDimensionalData->m_cBins = pParams->m_acBins[iDimensionInit];

      ++iDimensionInit;
   } while(cRealDimensions != iDimensionInit);

   DimensionalData * const aDimensionalDataShifted = &aDimensionalData[1];
   const size_t cRealDimensionsMinusOne = cRealDimensions - 1;

   const size_t cBytesPerBin = GetBinSize<FloatFast, StorageDataType>(bHessian, cScores);

   const FloatFast * pWeight;
   if(bWeight) {
      pWeight = pParams->m_aWeights;
   }

   while(true) {
      size_t cTensorBytes = cBytesPerBin;
      // for SIMD we'll want scatter/gather semantics since each parallel unit must load from a different pointer: 
      // otherwise we'll need to execute the scatter/gather as separate instructions in a templated loop
      // I think we can 6 dimensional 32 bin dimensions with that, and if we need more then we can use the 64
      // bit version that will fetch half of our values and do it twice
      // https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-avx-512-instructions/intrinsics-for-gather-and-scatter-operations/intrinsics-for-int-gather-and-scatter-ops.html
      // https://stackoverflow.com/questions/36971722/how-to-do-an-indirect-load-gather-scatter-in-avx-or-sse-instructions
      //
      // I think I want _mm512_i32gather_epi32.  I think I can use any 64 or 32 bit pointer as long as the index offsets
      // are 32-bit.  I cannot use the scale parameter since it is compile time and limited in values, so I would
      // want my tensors to be co-located into one big chunck of memory and the indexes will all index from the
      // base pointer!  I should be able to handle even very big tensors.  

      unsigned char * pRawBin = reinterpret_cast<unsigned char *>(aBins);
      {
         DimensionalData * const pDimensionalData = &aDimensionalDataShifted[-1];

         pDimensionalData->m_cShift -= pDimensionalData->m_cBitsPerItemMax;
         if(pDimensionalData->m_cShift < ptrdiff_t { 0 }) {
            if(pGradientsAndHessiansEnd == pGradientAndHessian) {
               // we only need to check this for the first dimension since all dimensions will reach
               // this point simultaneously
               return;
            }
            pDimensionalData->m_iTensorBinCombined = *pDimensionalData->m_pData;
            pDimensionalData->m_pData = pDimensionalData->m_pData + 1;
            pDimensionalData->m_cShift = pDimensionalData->m_cShiftReset;
         }

         const size_t iBin = static_cast<size_t>(
            pDimensionalData->m_iTensorBinCombined >> pDimensionalData->m_cShift) & pDimensionalData->m_maskBits;

         const size_t cBins = pDimensionalData->m_cBins;
         // earlier we return an interaction strength of 0.0 on any useless dimensions having 1 bin
         EBM_ASSERT(size_t { 2 } <= cBins);
         EBM_ASSERT(iBin < cBins);

         pRawBin += iBin * cTensorBytes;
         cTensorBytes *= cBins;
      }
      static constexpr bool isNotOneDimensional = 1 != cCompilerDimensions;
      if(isNotOneDimensional) {
         size_t iDimension = 0;
         do {
            DimensionalData * const pDimensionalData = &aDimensionalDataShifted[iDimension];

            pDimensionalData->m_cShift -= pDimensionalData->m_cBitsPerItemMax;
            if(pDimensionalData->m_cShift < ptrdiff_t { 0 }) {
               pDimensionalData->m_iTensorBinCombined = *pDimensionalData->m_pData;
               pDimensionalData->m_pData = pDimensionalData->m_pData + 1;
               pDimensionalData->m_cShift = pDimensionalData->m_cShiftReset;
            }

            const size_t iBin = static_cast<size_t>(
               pDimensionalData->m_iTensorBinCombined >> pDimensionalData->m_cShift) & pDimensionalData->m_maskBits;

            const size_t cBins = pDimensionalData->m_cBins;
            // earlier we return an interaction strength of 0.0 on any useless dimensions having 1 bin
            EBM_ASSERT(size_t { 2 } <= cBins);
            EBM_ASSERT(iBin < cBins);

            pRawBin += iBin * cTensorBytes;
            cTensorBytes *= cBins;

            ++iDimension;
         } while(cRealDimensionsMinusOne != iDimension);
      }

      auto * const pBin = reinterpret_cast<Bin<FloatFast, StorageDataType, bHessian, cArrayScores> *>(pRawBin);
      ASSERT_BIN_OK(cBytesPerBin, pBin, pParams->m_pDebugFastBinsEnd);

      pBin->SetCountSamples(pBin->GetCountSamples() + size_t { 1 });

      if(bWeight) {
         const FloatFast weight = *pWeight;
         pBin->SetWeight(pBin->GetWeight() + weight);
         ++pWeight;
      } else {
         // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
         //       such that we can remove that field optionally
         pBin->SetWeight(pBin->GetWeight() + FloatFast { 1 });
      }

      auto * const aGradientPair = pBin->GetGradientPairs();
      size_t iScore = 0;
      do {
         // TODO: unlike for BinSumsBoosting, we can re-arrange the values inside pGradientAndHessian to whatever
         //       order works best for us. Right now we put k_cSIMDPack gradient values for each sample in the pack
         //       next to eachother, then the k_cSIMDPack hessians, then we do the same for the next class if 
         //       this is multi-class.  If we don't SIMD-ify this code (we could do that if we're willing to use
         //       scattered writes and multiple histograms), then we'll probably want to move the gradient and hessian
         //       for sample 0 to be adjacent, then the gradient & hessian for class 1 of sample 0 to follow that
         //       to keep memory loads in order)
         //       We can do this because we only calculate the gradients & hessians during initialization, unlike
         //       boosting where they constantly change on each boost step

         auto * const pGradientPair = &aGradientPair[iScore];
         const FloatFast gradient = bHessian ? pGradientAndHessian[iScore << 1] : pGradientAndHessian[iScore];
         // DO NOT MULTIPLY gradient BY WEIGHT. WE PRE-MULTIPLIED WHEN WE ALLOCATED pGradientAndHessian
         pGradientPair->m_sumGradients += gradient;
         if(bHessian) {
            const FloatFast hessian = pGradientAndHessian[(iScore << 1) + 1];
            // DO NOT MULTIPLY hessian BY WEIGHT. WE PRE-MULTIPLIED WHEN WE ALLOCATED pGradientAndHessian
            pGradientPair->SetHess(pGradientPair->GetHess() + hessian);
         }
         ++iScore;
      } while(cScores != iScore);
      pGradientAndHessian += bHessian ? cScores << 1 : cScores;
   }
}


template<typename TFloat, bool bHessian, size_t cCompilerScores, size_t cCompilerDimensions, bool bWeight>
INLINE_RELEASE_TEMPLATED ErrorEbm OperatorBinSumsInteraction(BinSumsInteractionBridge * const pParams) {
   // TODO: in the future call back to the the operator class to allow it to inject the code into a GPU (see Objective.hpp for an example):
   // return TFloat::template OperatorBinSumsInteraction<TFloat, bHessian, cCompilerScores, cCompilerDimensions, bWeight>(pParams);
   // and also return the error code returned from that call instead of always Error_None
   BinSumsInteractionInternal<TFloat, bHessian, cCompilerScores, cCompilerDimensions, bWeight>(pParams);

   return Error_None;
}


template<typename TFloat, bool bHessian, size_t cCompilerScores, size_t cCompilerDimensions>
INLINE_RELEASE_TEMPLATED static ErrorEbm FinalOptionsInteraction(BinSumsInteractionBridge * const pParams) {
   if(nullptr != pParams->m_aWeights) {
      static constexpr bool bWeight = true;
      return OperatorBinSumsInteraction<TFloat, bHessian, cCompilerScores, cCompilerDimensions, bWeight>(pParams);
   } else {
      static constexpr bool bWeight = false;
      return OperatorBinSumsInteraction<TFloat, bHessian, cCompilerScores, cCompilerDimensions, bWeight>(pParams);
   }
}


template<typename TFloat, bool bHessian, size_t cCompilerScores, size_t cCompilerDimensionsPossible>
struct CountDimensionsInteraction final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsInteractionBridge * const pParams) {
      if(cCompilerDimensionsPossible == pParams->m_cRuntimeRealDimensions) {
         return FinalOptionsInteraction<TFloat, bHessian, cCompilerScores, cCompilerDimensionsPossible>(pParams);
      } else {
         return CountDimensionsInteraction<TFloat, bHessian, cCompilerScores, cCompilerDimensionsPossible + 1>::Func(pParams);
      }
   }
};
template<typename TFloat, bool bHessian, size_t cCompilerScores>
struct CountDimensionsInteraction<TFloat, bHessian, cCompilerScores, k_cCompilerOptimizedCountDimensionsMax + 1> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsInteractionBridge * const pParams) {
      return FinalOptionsInteraction<TFloat, bHessian, cCompilerScores, k_dynamicDimensions>(pParams);
   }
};


template<typename TFloat, bool bHessian, size_t cPossibleScores>
struct CountClassesInteraction final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsInteractionBridge * const pParams) {
      if(cPossibleScores == pParams->m_cScores) {
         return CountDimensionsInteraction<TFloat, bHessian, cPossibleScores, 1>::Func(pParams);
      } else {
         return CountClassesInteraction<TFloat, bHessian, cPossibleScores + 1>::Func(pParams);
      }
   }
};
template<typename TFloat, bool bHessian>
struct CountClassesInteraction<TFloat, bHessian, k_cCompilerScoresMax + 1> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsInteractionBridge * const pParams) {
      return CountDimensionsInteraction<TFloat, bHessian, k_dynamicScores, 1>::Func(pParams);
   }
};

template<typename TFloat>
INLINE_RELEASE_TEMPLATED static ErrorEbm BinSumsInteraction(BinSumsInteractionBridge * const pParams) {
   LOG_0(Trace_Verbose, "Entered BinSumsInteraction");

#ifndef NDEBUG
   // all our memory should be aligned. It is required by SIMD for correctness or performance
   EBM_ASSERT(IsAligned(pParams->m_aGradientsAndHessians));
   EBM_ASSERT(IsAligned(pParams->m_aWeights));
   EBM_ASSERT(IsAligned(pParams->m_aFastBins));
   for(size_t i = 0 ; i < pParams->m_cRuntimeRealDimensions; ++i) {
      EBM_ASSERT(IsAligned(pParams->m_aaPacked[i]));
   }
#endif // NDEBUG

   ErrorEbm error;

   EBM_ASSERT(1 <= pParams->m_cScores);
   if(EBM_FALSE != pParams->m_bHessian) {
      if(size_t { 1 } != pParams->m_cScores) {
         // muticlass
         error = CountClassesInteraction<TFloat, true, k_cCompilerScoresStart>::Func(pParams);
      } else {
         error = CountDimensionsInteraction<TFloat, true, k_oneScore, 1>::Func(pParams);
      }
   } else {
      if(size_t { 1 } != pParams->m_cScores) {
         // Odd: gradient multiclass. Allow it, but do not optimize for it
         error = FinalOptionsInteraction<TFloat, false, k_dynamicScores, k_dynamicDimensions>(pParams);
      } else {
         error = CountDimensionsInteraction<TFloat, false, k_oneScore, 1>::Func(pParams);
      }
   }

   LOG_0(Trace_Verbose, "Exited BinSumsInteraction");

   return error;
}

} // DEFINED_ZONE_NAME

#endif // BIN_SUMS_INTERACTION_HPP