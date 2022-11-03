// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT
#include "zones.h"

#include "bridge_cpp.hpp" // BinSumsInteractionBridge

#include "ebm_internal.hpp" // k_cDimensionsMax

#include "DataSetInteraction.hpp" // DataSetInteraction
#include "GradientPair.hpp" // GradientPair
#include "Bin.hpp" // Bin
#include "InteractionCore.hpp" // InteractionCore
#include "InteractionShell.hpp" // InteractionShell

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

template<ptrdiff_t cCompilerClasses, size_t cCompilerDimensions, bool bWeight>
INLINE_RELEASE_TEMPLATED static ErrorEbm BinSumsInteractionInternal(BinSumsInteractionBridge * const pParams) {
   static constexpr bool bClassification = IsClassification(cCompilerClasses);
   static constexpr size_t cCompilerScores = GetCountScores(cCompilerClasses);

   auto * const aBins = pParams->m_aFastBins->Specialize<FloatFast, bClassification, cCompilerScores>();

   const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, pParams->m_cClasses);
   const size_t cScores = GetCountScores(cClasses);

   EBM_ASSERT(!IsOverflowBinSize<FloatFast>(bClassification, cScores)); // we're accessing allocated memory
   const size_t cBytesPerBin = GetBinSize<FloatFast>(bClassification, cScores);

   const size_t cSamples = pParams->m_cSamples;
   EBM_ASSERT(1 <= cSamples);

   const FloatFast * pGradientAndHessian = pParams->m_aGradientsAndHessians;
   const FloatFast * const pGradientsAndHessiansEnd = pGradientAndHessian + (bClassification ? 2 : 1) * cScores * cSamples;

   const FloatFast * pWeight = bWeight ? pParams->m_aWeights : nullptr;

   const size_t cRealDimensions = GET_COUNT_DIMENSIONS(cCompilerDimensions, pParams->m_cRuntimeRealDimensions);
   EBM_ASSERT(1 <= cRealDimensions); // for interactions, we just return 0 for interactions with zero features

#ifndef NDEBUG
   FloatFast weightTotalDebug = 0;
#endif // NDEBUG

   struct DimensionalData {
      ptrdiff_t m_cShift;
      size_t m_cBitsPerItemMax;
      StorageDataType m_iTensorBinCombined;
      size_t m_maskBits;
      size_t m_cBins;
      const StorageDataType * m_pData;
      ptrdiff_t m_cShiftReset;
   };

   // this is on the stack and the compiler should be able to optimize these as if they were variables or registers
   DimensionalData aDimensionalData[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
   size_t iDimensionInit = 0;
   do {
      DimensionalData * const pDimensionalData = &aDimensionalData[iDimensionInit];

      const StorageDataType * const pData = pParams->m_aaPacked[iDimensionInit];
      pDimensionalData->m_iTensorBinCombined = *pData;
      pDimensionalData->m_pData = pData + 1;

      const size_t cItemsPerBitPack = pParams->m_acItemsPerBitPack[iDimensionInit];
      EBM_ASSERT(size_t { 1 } <= cItemsPerBitPack);
      EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);

      const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPack);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      pDimensionalData->m_cBitsPerItemMax = cBitsPerItemMax;

      pDimensionalData->m_cShift = static_cast<ptrdiff_t>(((cSamples - 1) % cItemsPerBitPack + 1) * cBitsPerItemMax);
      pDimensionalData->m_cShiftReset = static_cast<ptrdiff_t>((cItemsPerBitPack - 1) * cBitsPerItemMax);

      const size_t maskBits = (~size_t { 0 }) >> (k_cBitsForSizeT - cBitsPerItemMax);
      pDimensionalData->m_maskBits = maskBits;

      pDimensionalData->m_cBins = pParams->m_acBins[iDimensionInit];

      ++iDimensionInit;
   } while(cRealDimensions != iDimensionInit);

   while(true) {
      size_t cTensorBytes = cBytesPerBin;
      unsigned char * pRawBin = reinterpret_cast<unsigned char *>(aBins);
      size_t iDimension = 0;
      do {
         DimensionalData * const pDimensionalData = &aDimensionalData[iDimension];

         ptrdiff_t cShift = pDimensionalData->m_cShift;
         cShift -= pDimensionalData->m_cBitsPerItemMax;
         StorageDataType iTensorBinCombined = pDimensionalData->m_iTensorBinCombined;
         if(cShift < ptrdiff_t { 0 }) {
            const StorageDataType * const pInputData = pDimensionalData->m_pData;
            if(pGradientsAndHessiansEnd == pGradientAndHessian) {
               // TODO: we only need to do this for the first dimension since all dimensions will reach
               // this point simultaneously.  But to do this I would need to separate out the first dimension
               // so do it after we've locked down everything else about this loop
               goto done;
            }
            iTensorBinCombined = *pInputData;
            pDimensionalData->m_pData = pInputData + 1;
            cShift = pDimensionalData->m_cShiftReset;
            pDimensionalData->m_iTensorBinCombined = iTensorBinCombined;
         }
         pDimensionalData->m_cShift = cShift;

         const size_t iBin = static_cast<size_t>(iTensorBinCombined >> cShift) & pDimensionalData->m_maskBits;

         const size_t cBins = pDimensionalData->m_cBins;
         // earlier we return an interaction strength of 0.0 on any useless dimensions having 1 bin
         EBM_ASSERT(size_t { 2 } <= cBins);
         EBM_ASSERT(iBin < cBins);

         pRawBin += iBin * cTensorBytes;
         cTensorBytes *= cBins;
          
         ++iDimension;
      } while(cRealDimensions != iDimension);

      auto * const pBin = reinterpret_cast<Bin<FloatFast, bClassification, cCompilerScores> *>(pRawBin);
      ASSERT_BIN_OK(cBytesPerBin, pBin, pParams->m_pDebugFastBinsEnd);

      pBin->SetCountSamples(pBin->GetCountSamples() + size_t { 1 });

      FloatFast weight = 1;
      if(bWeight) {
         weight = *pWeight;
         ++pWeight;
#ifndef NDEBUG
         weightTotalDebug += weight;
#endif // NDEBUG
      }
      pBin->SetWeight(pBin->GetWeight() + weight);

      auto * const aGradientPair = pBin->GetGradientPairs();
      size_t iScore = 0;
      do {
         auto * const pGradientPair = &aGradientPair[iScore];
         const FloatFast gradient = bClassification ? pGradientAndHessian[iScore << 1] : pGradientAndHessian[iScore];
         // DO NOT MULTIPLY gradient BY WEIGHT. WE PRE-MULTIPLIED WHEN WE ALLOCATED pGradientAndHessian
         pGradientPair->m_sumGradients += gradient;
         if(bClassification) {
            const FloatFast hessian = pGradientAndHessian[(iScore << 1) + 1];
            // DO NOT MULTIPLY hessian BY WEIGHT. WE PRE-MULTIPLIED WHEN WE ALLOCATED pGradientAndHessian
            pGradientPair->SetHess(pGradientPair->GetHess() + hessian);
         }
         ++iScore;
      } while(cScores != iScore);
      pGradientAndHessian += bClassification ? cScores << 1 : cScores;
   }
done:;

   EBM_ASSERT(0 < pParams->totalWeightDebug);
   EBM_ASSERT(nullptr == pWeight || weightTotalDebug * FloatFast { 0.999 } <= pParams->totalWeightDebug &&
      pParams->totalWeightDebug <= FloatFast { 1.001 } * weightTotalDebug);
   EBM_ASSERT(nullptr != pWeight || 
      static_cast<FloatFast>(cSamples) == pParams->totalWeightDebug);

   return Error_None;
}

template<ptrdiff_t cCompilerClasses, size_t cCompilerDimensions>
INLINE_RELEASE_TEMPLATED static ErrorEbm FinalOptions(BinSumsInteractionBridge * const pParams) {
   if(nullptr != pParams->m_aWeights) {
      constexpr bool bWeight = true;
      return BinSumsInteractionInternal<cCompilerClasses, cCompilerDimensions, bWeight>(pParams);
   } else {
      constexpr bool bWeight = false;
      return BinSumsInteractionInternal<cCompilerClasses, cCompilerDimensions, bWeight>(pParams);
   }
}

template<ptrdiff_t cCompilerClasses, size_t cCompilerDimensionsPossible>
class BinSumsInteractionDimensions final {
public:

   BinSumsInteractionDimensions() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsInteractionBridge * const pParams) {
      static_assert(1 <= cCompilerDimensionsPossible, "can't have less than 1 dimension for interactions");
      static_assert(cCompilerDimensionsPossible <= k_cDimensionsMax, "can't have more than the max dimensions");

      EBM_ASSERT(1 <= pParams->m_cRuntimeRealDimensions);
      EBM_ASSERT(pParams->m_cRuntimeRealDimensions <= k_cDimensionsMax);
      if(cCompilerDimensionsPossible == pParams->m_cRuntimeRealDimensions) {
         return FinalOptions<cCompilerClasses, cCompilerDimensionsPossible>(pParams);
      } else {
         return BinSumsInteractionDimensions<cCompilerClasses, cCompilerDimensionsPossible + 1>::Func(pParams);
      }
   }
};

template<ptrdiff_t cCompilerClasses>
class BinSumsInteractionDimensions<cCompilerClasses, k_cCompilerOptimizedCountDimensionsMax + 1> final {
public:

   BinSumsInteractionDimensions() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsInteractionBridge * const pParams) {
      EBM_ASSERT(1 <= pParams->m_cRuntimeRealDimensions);
      EBM_ASSERT(pParams->m_cRuntimeRealDimensions <= k_cDimensionsMax);
      return FinalOptions<cCompilerClasses, k_dynamicDimensions>(pParams);
   }
};

template<ptrdiff_t cPossibleClasses>
class BinSumsInteractionTarget final {
public:

   BinSumsInteractionTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsInteractionBridge * const pParams) {
      static_assert(IsClassification(cPossibleClasses), "cPossibleClasses needs to be a classification");
      static_assert(cPossibleClasses <= k_cCompilerClassesMax, "We can't have this many items in a data pack.");

      const ptrdiff_t cRuntimeClasses = pParams->m_cClasses;
      EBM_ASSERT(IsClassification(cRuntimeClasses));
      EBM_ASSERT(cRuntimeClasses <= k_cCompilerClassesMax);

      if(cPossibleClasses == cRuntimeClasses) {
         return BinSumsInteractionDimensions<cPossibleClasses, 2>::Func(pParams);
      } else {
         return BinSumsInteractionTarget<cPossibleClasses + 1>::Func(pParams);
      }
   }
};

template<>
class BinSumsInteractionTarget<k_cCompilerClassesMax + 1> final {
public:

   BinSumsInteractionTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsInteractionBridge * const pParams) {
      static_assert(IsClassification(k_cCompilerClassesMax), "k_cCompilerClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pParams->m_cClasses));
      EBM_ASSERT(k_cCompilerClassesMax < pParams->m_cClasses);

      return BinSumsInteractionDimensions<k_dynamicClassification, 2>::Func(pParams);
   }
};

extern ErrorEbm BinSumsInteraction(BinSumsInteractionBridge * const pParams) {
   LOG_0(Trace_Verbose, "Entered BinSumsInteraction");

   ErrorEbm error;

   const ptrdiff_t cRuntimeClasses = pParams->m_cClasses;
   if(IsClassification(cRuntimeClasses)) {
      error = BinSumsInteractionTarget<2>::Func(pParams);
   } else {
      EBM_ASSERT(IsRegression(cRuntimeClasses));
      error = BinSumsInteractionDimensions<k_regression, 2>::Func(pParams);
   }

   LOG_0(Trace_Verbose, "Exited BinSumsInteraction");

   return error;
}

} // DEFINED_ZONE_NAME
