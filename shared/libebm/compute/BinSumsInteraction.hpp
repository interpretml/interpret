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

template<typename TFloat, bool bHessian, size_t cCompilerScores, size_t cCompilerDimensions, bool bWeight>
GPU_DEVICE NEVER_INLINE static void BinSumsInteractionInternal(BinSumsInteractionBridge * const pParams) {
   static constexpr size_t cArrayScores = GetArrayScores(cCompilerScores);

#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pParams);
   EBM_ASSERT(1 <= pParams->m_cSamples);
   EBM_ASSERT(0 == pParams->m_cSamples % size_t { TFloat::k_cSIMDPack });
   EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
   EBM_ASSERT(nullptr != pParams->m_aFastBins);
   EBM_ASSERT(k_dynamicScores == cCompilerScores || cCompilerScores == pParams->m_cScores);
   EBM_ASSERT(k_dynamicDimensions == cCompilerDimensions || cCompilerDimensions == pParams->m_cRuntimeRealDimensions);
   EBM_ASSERT(1 <= pParams->m_cRuntimeRealDimensions); // for interactions, we just return 0 for interactions with zero features
   EBM_ASSERT(1 == cCompilerDimensions || 1 != pParams->m_cRuntimeRealDimensions); // 1 dimension must be templated
#endif // GPU_COMPILE

   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pParams->m_cScores);

   auto * const aBins = reinterpret_cast<BinBase *>(pParams->m_aFastBins)->Specialize<typename TFloat::T, typename TFloat::TInt::T, bHessian, cArrayScores>();

   const size_t cSamples = pParams->m_cSamples;

   const typename TFloat::T * pGradientAndHessian = reinterpret_cast<const typename TFloat::T *>(pParams->m_aGradientsAndHessians);
   const typename TFloat::T * const pGradientsAndHessiansEnd = pGradientAndHessian + (bHessian ? size_t { 2 } : size_t { 1 }) * cScores * cSamples;

   struct alignas(EbmMax(alignof(typename TFloat::TInt), alignof(void *), alignof(size_t), alignof(int))) DimensionalData {
      // C struct packing rules say these will be aligned within the struct to sizeof(typename TFloat::TInt)
      // and the compiler should (although some compilers have bugs) align the entire struct on the stack to 
      // alignof(typename TFloat::TInt) from the alignas directive above assuming TFloat::TInt is a large SIMD type
      ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER typename TFloat::TInt iBinCombined;
      ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER typename TFloat::TInt maskBits;
      ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER const typename TFloat::TInt::T * m_pData;
      ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER size_t m_cBins;
      ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER int m_cShift;
      ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER int m_cBitsPerItemMax;
      ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER int m_cShiftReset;
   };

   const size_t cRealDimensions = GET_COUNT_DIMENSIONS(cCompilerDimensions, pParams->m_cRuntimeRealDimensions);

   // this is on the stack and the compiler should be able to optimize these as if they were variables or registers
   DimensionalData aDimensionalData[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];

   size_t iDimensionInit = 0;
   do {
      DimensionalData * const pDimensionalData = &aDimensionalData[iDimensionInit];

      const typename TFloat::TInt::T * const pData = reinterpret_cast<const typename TFloat::TInt::T *>(pParams->m_aaPacked[iDimensionInit]);
      pDimensionalData->iBinCombined = TFloat::TInt::Load(pData);
      pDimensionalData->m_pData = pData + TFloat::TInt::k_cSIMDPack;

      const int cItemsPerBitPack = pParams->m_acItemsPerBitPack[iDimensionInit];
#ifndef GPU_COMPILE
      EBM_ASSERT(1 <= cItemsPerBitPack);
      EBM_ASSERT(cItemsPerBitPack <= COUNT_BITS(typename TFloat::TInt::T));
#endif // GPU_COMPILE

      const int cBitsPerItemMax = GetCountBits<typename TFloat::TInt::T>(cItemsPerBitPack);
#ifndef GPU_COMPILE
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= COUNT_BITS(typename TFloat::TInt::T));
#endif // GPU_COMPILE
      pDimensionalData->m_cBitsPerItemMax = cBitsPerItemMax;

      pDimensionalData->m_cShift = (static_cast<int>(((cSamples >> TFloat::k_cSIMDShift) - size_t { 1 }) % static_cast<size_t>(cItemsPerBitPack)) + 1) * cBitsPerItemMax;
      pDimensionalData->m_cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;

      pDimensionalData->maskBits = MakeLowMask<typename TFloat::TInt::T>(cBitsPerItemMax);

      pDimensionalData->m_cBins = pParams->m_acBins[iDimensionInit];

      ++iDimensionInit;
   } while(cRealDimensions != iDimensionInit);

   DimensionalData * const aDimensionalDataShifted = &aDimensionalData[1];
   const size_t cRealDimensionsMinusOne = cRealDimensions - 1;

   const size_t cBytesPerBin = GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(bHessian, cScores);

   const typename TFloat::T * pWeight;
   if(bWeight) {
      pWeight = reinterpret_cast<const typename TFloat::T *>(pParams->m_aWeights);
#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
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

      Bin<typename TFloat::T, typename TFloat::TInt::T, bHessian, cArrayScores> * apBins[TFloat::k_cSIMDPack];
      TFloat::Execute([aBins, &apBins](const int i) {
         apBins[i] = aBins;
      });
      {
         DimensionalData * const pDimensionalData = &aDimensionalDataShifted[-1];

         pDimensionalData->m_cShift -= pDimensionalData->m_cBitsPerItemMax;
         if(pDimensionalData->m_cShift < 0) {
            if(pGradientsAndHessiansEnd == pGradientAndHessian) {
               // we only need to check this for the first dimension since all dimensions will reach
               // this point simultaneously
               return;
            }
            pDimensionalData->iBinCombined = TFloat::TInt::Load(pDimensionalData->m_pData);
            pDimensionalData->m_pData = pDimensionalData->m_pData + TFloat::TInt::k_cSIMDPack;
            pDimensionalData->m_cShift = pDimensionalData->m_cShiftReset;
         }

         const typename TFloat::TInt iBin = (pDimensionalData->iBinCombined >> pDimensionalData->m_cShift) & pDimensionalData->maskBits;

         const size_t cBins = pDimensionalData->m_cBins;
         // earlier we return an interaction strength of 0.0 on any useless dimensions having 1 bin
#ifndef NDEBUG
#ifndef GPU_COMPILE
         EBM_ASSERT(size_t { 2 } <= cBins);
         TFloat::TInt::Execute([cBins](int, const typename TFloat::TInt::T x) {
            EBM_ASSERT(static_cast<size_t>(x) < cBins);
         }, iBin);
#endif // GPU_COMPILE
#endif // NDEBUG

         TFloat::TInt::Execute([&apBins, cTensorBytes](const int i, const typename TFloat::TInt::T x) {
            apBins[i] = IndexByte(apBins[i], static_cast<size_t>(x) * cTensorBytes);
         }, iBin);

         cTensorBytes *= cBins;
      }
      static constexpr bool isNotOneDimensional = 1 != cCompilerDimensions;
      if(isNotOneDimensional) {
         size_t iDimension = 0;
         do {
            DimensionalData * const pDimensionalData = &aDimensionalDataShifted[iDimension];

            pDimensionalData->m_cShift -= pDimensionalData->m_cBitsPerItemMax;
            if(pDimensionalData->m_cShift < 0) {
               pDimensionalData->iBinCombined = TFloat::TInt::Load(pDimensionalData->m_pData);
               pDimensionalData->m_pData = pDimensionalData->m_pData + TFloat::TInt::k_cSIMDPack;
               pDimensionalData->m_cShift = pDimensionalData->m_cShiftReset;
            }

            const typename TFloat::TInt iBin = (pDimensionalData->iBinCombined >> pDimensionalData->m_cShift) & pDimensionalData->maskBits;

            const size_t cBins = pDimensionalData->m_cBins;
            // earlier we return an interaction strength of 0.0 on any useless dimensions having 1 bin
#ifndef NDEBUG
#ifndef GPU_COMPILE
            EBM_ASSERT(size_t { 2 } <= cBins);
            TFloat::TInt::Execute([cBins](int, const typename TFloat::TInt::T x) {
               EBM_ASSERT(static_cast<size_t>(x) < cBins);
            }, iBin);
#endif // GPU_COMPILE
#endif // NDEBUG

            TFloat::TInt::Execute([&apBins, cTensorBytes](const int i, const typename TFloat::TInt::T x) {
               // TODO: I think this non-SIMD multiplication is the bottleneck for this code. Since it wouldn't
               // change any memory layout, we could have two versions of this loop. One would be the current 
               // code fallback that exectures if the multiplication would exceed a 32-bit integer, and the other could
               // do the multiplication with SIMD to avoid most of the cost when the result will fit into a 32 bit result
               apBins[i] = IndexByte(apBins[i], static_cast<size_t>(x) * cTensorBytes);
            }, iBin);

            cTensorBytes *= cBins;

            ++iDimension;
         } while(cRealDimensionsMinusOne != iDimension);
      }

      TFloat::Execute([apBins](const int i) {
         auto * const pBin = apBins[i];
         // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
         //       such that we can remove that field optionally
         pBin->SetCountSamples(pBin->GetCountSamples() + typename TFloat::TInt::T { 1 });
      });

      if(bWeight) {
         const TFloat weight = TFloat::Load(pWeight);
         pWeight += TFloat::k_cSIMDPack;

         TFloat::Execute([apBins](const int i, const typename TFloat::T x) {
            auto * const pBin = apBins[i];
            // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
            //       such that we can remove that field optionally
            pBin->SetWeight(pBin->GetWeight() + x);
         }, weight);
      } else {
         TFloat::Execute([apBins](const int i) {
            auto * const pBin = apBins[i];
            // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
            //       such that we can remove that field optionally
            pBin->SetWeight(pBin->GetWeight() + typename TFloat::T { 1.0 });
         });
      }

      size_t iScore = 0;
      do {
         if(bHessian) {
            const TFloat gradient = TFloat::Load(&pGradientAndHessian[iScore << (TFloat::k_cSIMDShift + 1)]);
            const TFloat hessian = TFloat::Load(&pGradientAndHessian[(iScore << (TFloat::k_cSIMDShift + 1)) + TFloat::k_cSIMDPack]);
            TFloat::Execute([apBins, iScore](const int i, const typename TFloat::T grad, const typename TFloat::T hess) {
               // BEWARE: unless we generate a separate histogram for each SIMD stream and later merge them, pBin can 
               // point to the same bin in multiple samples within the SIMD pack, so we need to serialize fetching sums
               auto * const pBin = apBins[i];
               auto * const aGradientPair = pBin->GetGradientPairs();
               auto * const pGradientPair = &aGradientPair[iScore];
               typename TFloat::T binGrad = pGradientPair->m_sumGradients;
               typename TFloat::T binHess = pGradientPair->GetHess();
               binGrad += grad;
               binHess += hess;
               pGradientPair->m_sumGradients = binGrad;
               pGradientPair->SetHess(binHess);
            }, gradient, hessian);
         } else {
            const TFloat gradient = TFloat::Load(&pGradientAndHessian[iScore << TFloat::k_cSIMDShift]);
            TFloat::Execute([apBins, iScore](const int i, const typename TFloat::T grad) {
               // BEWARE: unless we generate a separate histogram for each SIMD stream and later merge them, pBin can 
               // point to the same bin in multiple samples within the SIMD pack, so we need to serialize fetching sums
               auto * const pBin = apBins[i];
               auto * const aGradientPair = pBin->GetGradientPairs();
               auto * const pGradientPair = &aGradientPair[iScore];
               pGradientPair->m_sumGradients += grad;
            }, gradient);
         }
         ++iScore;
      } while(cScores != iScore);

      pGradientAndHessian += cScores << (bHessian ? (TFloat::k_cSIMDShift + 1) : TFloat::k_cSIMDShift);
   }
}

template<typename TFloat, bool bHessian, size_t cCompilerScores, size_t cCompilerDimensions, bool bWeight>
GPU_GLOBAL static void RemoteBinSumsInteraction(BinSumsInteractionBridge * const pParams) {
   BinSumsInteractionInternal<TFloat, bHessian, cCompilerScores, cCompilerDimensions, bWeight>(pParams);
}

template<typename TFloat, bool bHessian, size_t cCompilerScores, size_t cCompilerDimensions, bool bWeight>
INLINE_RELEASE_TEMPLATED ErrorEbm OperatorBinSumsInteraction(BinSumsInteractionBridge * const pParams) {
   return TFloat::template OperatorBinSumsInteraction<bHessian, cCompilerScores, cCompilerDimensions, bWeight>(pParams);
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