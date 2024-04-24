// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef BIN_SUMS_BOOSTING_HPP
#define BIN_SUMS_BOOSTING_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT

#include "common.hpp" // Multiply
#include "bridge.hpp" // BinSumsBoostingBridge
#include "GradientPair.hpp"
#include "Bin.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

static constexpr int k_cItemsPerBitPackBoostingMax = 64;
static constexpr int k_cItemsPerBitPackBoostingMin = 1;

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bParallel,
      bool bCollapsed,
      size_t cCompilerScores,
      int cCompilerPack,
      typename std::enable_if<bCollapsed && 1 == cCompilerScores, int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

   static_assert(!bParallel, "BinSumsBoosting specialization for collapsed does not handle parallel bins.");

#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pParams);
   EBM_ASSERT(1 <= pParams->m_cSamples);
   EBM_ASSERT(0 == pParams->m_cSamples % size_t{TFloat::k_cSIMDPack});
   EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
   EBM_ASSERT(nullptr != pParams->m_aFastBins);
   EBM_ASSERT(size_t{1} == pParams->m_cScores);
#endif // GPU_COMPILE

   const size_t cSamples = pParams->m_cSamples;

   auto* const aBins =
         reinterpret_cast<BinBase*>(pParams->m_aFastBins)
               ->Specialize<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian, 1>();

   const typename TFloat::T* pGradientAndHessian =
         reinterpret_cast<const typename TFloat::T*>(pParams->m_aGradientsAndHessians);
   const typename TFloat::T* const pGradientsAndHessiansEnd =
         pGradientAndHessian + (bHessian ? size_t{2} : size_t{1}) * cSamples;

   const typename TFloat::T* pWeight;
   if(bWeight) {
      pWeight = reinterpret_cast<const typename TFloat::T*>(pParams->m_aWeights);
#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
   }

   TFloat gradientTotal = 0;
   TFloat hessianTotal;
   if(bHessian) {
      hessianTotal = 0;
   }
   do {
      TFloat weight;
      if(bWeight) {
         weight = TFloat::Load(pWeight);
         pWeight += TFloat::k_cSIMDPack;
      }

      TFloat gradient = TFloat::Load(pGradientAndHessian);
      TFloat hessian;
      if(bHessian) {
         hessian = TFloat::Load(&pGradientAndHessian[TFloat::k_cSIMDPack]);
      }
      pGradientAndHessian += (bHessian ? size_t{2} : size_t{1}) * TFloat::k_cSIMDPack;

      if(bWeight) {
         gradient *= weight;
         if(bHessian) {
            hessian *= weight;
         }
      }

      gradientTotal += gradient;
      if(bHessian) {
         hessianTotal += hessian;
      }
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);

   auto* const pGradientPair = aBins->GetGradientPairs();
   pGradientPair->m_sumGradients += Sum(gradientTotal);
   if(bHessian) {
      pGradientPair->SetHess(pGradientPair->GetHess() + Sum(hessianTotal));
   }
}

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bParallel,
      bool bCollapsed,
      size_t cCompilerScores,
      int cCompilerPack,
      typename std::enable_if<bCollapsed && 1 != cCompilerScores, int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

   static_assert(!bParallel, "BinSumsBoosting specialization for collapsed does not handle parallel bins.");

   static constexpr size_t cArrayScores = GetArrayScores(cCompilerScores);

#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pParams);
   EBM_ASSERT(1 <= pParams->m_cSamples);
   EBM_ASSERT(0 == pParams->m_cSamples % size_t{TFloat::k_cSIMDPack});
   EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
   EBM_ASSERT(nullptr != pParams->m_aFastBins);
   EBM_ASSERT(k_dynamicScores == cCompilerScores || cCompilerScores == pParams->m_cScores);
#endif // GPU_COMPILE

   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pParams->m_cScores);

   const size_t cSamples = pParams->m_cSamples;

   auto* const aBins =
         reinterpret_cast<BinBase*>(pParams->m_aFastBins)
               ->Specialize<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian, cArrayScores>();

   const typename TFloat::T* pGradientAndHessian =
         reinterpret_cast<const typename TFloat::T*>(pParams->m_aGradientsAndHessians);
   const typename TFloat::T* const pGradientsAndHessiansEnd =
         pGradientAndHessian + (bHessian ? size_t{2} : size_t{1}) * cScores * cSamples;

   const typename TFloat::T* pWeight;
   if(bWeight) {
      pWeight = reinterpret_cast<const typename TFloat::T*>(pParams->m_aWeights);
#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
   }

   do {
      TFloat weight;
      if(bWeight) {
         weight = TFloat::Load(pWeight);
         pWeight += TFloat::k_cSIMDPack;
      }

      size_t iScore = 0;
      do {
         TFloat gradient = TFloat::Load(&pGradientAndHessian[iScore << (TFloat::k_cSIMDShift + 1)]);
         TFloat hessian;
         if(bHessian) {
            hessian = TFloat::Load(&pGradientAndHessian[(iScore << (TFloat::k_cSIMDShift + 1)) + TFloat::k_cSIMDPack]);
         }
         if(bWeight) {
            gradient *= weight;
            if(bHessian) {
               hessian *= weight;
            }
         }

         const typename TFloat::T gradientSum = Sum(gradient);
         typename TFloat::T hessianSum;
         if(bHessian) {
            hessianSum = Sum(hessian);
         }

         auto* const aGradientPair = aBins->GetGradientPairs();
         auto* const pGradientPair = &aGradientPair[iScore];
         typename TFloat::T binGrad = pGradientPair->m_sumGradients;
         typename TFloat::T binHess;
         if(bHessian) {
            binHess = pGradientPair->GetHess();
         }
         binGrad += gradientSum;
         if(bHessian) {
            binHess += hessianSum;
         }
         pGradientPair->m_sumGradients = binGrad;
         if(bHessian) {
            pGradientPair->SetHess(binHess);
         }

         ++iScore;
      } while(cScores != iScore);

      pGradientAndHessian += cScores << (bHessian ? (TFloat::k_cSIMDShift + 1) : TFloat::k_cSIMDShift);
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);
}

#if 0 < PARALLEL_BINS_BYTES_MAX
template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bParallel,
      bool bCollapsed,
      size_t cCompilerScores,
      int cCompilerPack,
      typename std::enable_if<bParallel && !bCollapsed && 1 == cCompilerScores, int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

   static_assert(1 == cCompilerScores, "This specialization of BinSumsBoostingInternal cannot handle multiclass.");
   static constexpr bool bFixedSizePack = k_cItemsPerBitPackUndefined != cCompilerPack;

   static_assert(0 == Bin<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian>::k_offsetGrad,
         "We treat aBins as a flat array of TFloat::T, so the Bin class needs to be ordered in an exact way");
   static_assert(!bHessian ||
               sizeof(typename TFloat::T) ==
                     Bin<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian>::k_offsetHess,
         "We treat aBins as a flat array of TFloat::T, so the Bin class needs to be ordered in an exact way");
   static_assert((bHessian ? size_t{2} : size_t{1}) * sizeof(typename TFloat::T) ==
               sizeof(Bin<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian>),
         "We treat aBins as a flat array of TFloat::T, so the Bin class needs to be ordered in an exact way");

#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pParams);
   EBM_ASSERT(1 <= pParams->m_cSamples);
   EBM_ASSERT(0 == pParams->m_cSamples % size_t{TFloat::k_cSIMDPack});
   EBM_ASSERT(0 == pParams->m_cSamples % size_t{(bFixedSizePack ? cCompilerPack : 1) * TFloat::k_cSIMDPack});
   EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
   EBM_ASSERT(nullptr != pParams->m_aFastBins);
   EBM_ASSERT(size_t{1} == pParams->m_cScores);
#endif // GPU_COMPILE

   const size_t cSamples = pParams->m_cSamples;

   typename TFloat::T* const aBins = reinterpret_cast<typename TFloat::T*>(pParams->m_aFastBins);

   const typename TFloat::T* pGradientAndHessian =
         reinterpret_cast<const typename TFloat::T*>(pParams->m_aGradientsAndHessians);
   const typename TFloat::T* const pGradientsAndHessiansEnd =
         pGradientAndHessian + (bHessian ? size_t{2} : size_t{1}) * cSamples;

   static constexpr typename TFloat::TInt::T cBytesPerBin = static_cast<typename TFloat::TInt::T>(
         GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(false, false, bHessian, size_t{1}));

   // The compiler is normally pretty good about optimizing multiplications into shifts when possible
   // BUT, when compiling for SIMD, it seems to use a SIMD multiplication instruction instead of shifts
   // even when the multiplication has a fixed compile time constant value that is a power of 2, so
   // we manually convert the multiplications into shifts.
   //
   // We also have tried the Multiply templated function that is designed to convert multiplications
   // into shifts, but using that templated function breaks the compiler optimization that unrolls
   // the bitpacking loop.
   //
   constexpr static bool bSmall = 4 == cBytesPerBin;
   constexpr static bool bMed = 8 == cBytesPerBin;
   constexpr static bool bLarge = 16 == cBytesPerBin;
   static_assert(bSmall || bMed || bLarge, "cBytesPerBin size must be small, medium, or large");
   constexpr static int cFixedShift = bSmall ? 2 : bMed ? 3 : 4;
   static_assert(1 << cFixedShift == cBytesPerBin, "cFixedShift must match the BinSize");
   EBM_ASSERT(0 == pParams->m_cBytesFastBins % static_cast<size_t>(cBytesPerBin));

   const typename TFloat::TInt offsets =
         TFloat::TInt::MakeIndexes() * static_cast<typename TFloat::TInt::T>(pParams->m_cBytesFastBins >> cFixedShift);

   const int cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pParams->m_cPack);
#ifndef GPU_COMPILE
   EBM_ASSERT(1 <= cItemsPerBitPack);
   EBM_ASSERT(cItemsPerBitPack <= COUNT_BITS(typename TFloat::TInt::T));
#endif // GPU_COMPILE

   const int cBitsPerItemMax = GetCountBits<typename TFloat::TInt::T>(cItemsPerBitPack);
#ifndef GPU_COMPILE
   EBM_ASSERT(1 <= cBitsPerItemMax);
   EBM_ASSERT(cBitsPerItemMax <= COUNT_BITS(typename TFloat::TInt::T));
#endif // GPU_COMPILE

   const typename TFloat::TInt maskBits = MakeLowMask<typename TFloat::TInt::T>(cBitsPerItemMax);

   const typename TFloat::TInt::T* pInputData = reinterpret_cast<const typename TFloat::TInt::T*>(pParams->m_aPacked);
#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pInputData);
#endif // GPU_COMPILE

   typename TFloat::TInt iTensorBin;
   const int cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;
   int cShift;
   if(bFixedSizePack) {
      iTensorBin = (TFloat::TInt::Load(pInputData) & maskBits) + offsets;
      if(bHessian) {
         iTensorBin = PermuteForInterleaf(iTensorBin);
      }
      pInputData += TFloat::TInt::k_cSIMDPack;
   } else {
      cShift = static_cast<int>((cSamples >> TFloat::k_cSIMDShift) % static_cast<size_t>(cItemsPerBitPack)) *
            cBitsPerItemMax;
      iTensorBin = ((TFloat::TInt::Load(pInputData) >> cShift) & maskBits) + offsets;
      if(bHessian) {
         iTensorBin = PermuteForInterleaf(iTensorBin);
      }
      cShift -= cBitsPerItemMax;
      if(cShift < 0) {
         cShift = cShiftReset;
         pInputData += TFloat::TInt::k_cSIMDPack;
      }
   }

   const typename TFloat::T* pWeight;
   if(bWeight) {
      pWeight = reinterpret_cast<const typename TFloat::T*>(pParams->m_aWeights);
#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
   }

   // We want to structure the loop below so that the load happens immediately after the store
   // because that allows the maximum possible time for the gathering load to happen in the CPU pipeline.
   // To do that we put the store at the top and the load below. But now we need to exectute a
   // store on the first iteration, so load the values from memory here that we'll then store
   // back on the first loop iteration
   typename TFloat::TInt iTensorBinPrev = offsets;
   TFloat bin0;
   TFloat bin1;
   if(!bHessian) {
      bin0 = TFloat::template Load<cFixedShift>(aBins, iTensorBinPrev);
   } else {
      TFloat::template DoubleLoad<cFixedShift>(aBins, iTensorBinPrev, bin0, bin1);
   }

   TFloat gradient = 0;
   TFloat hessian;
   if(bHessian) {
      hessian = 0;
   }

   TFloat gradhess0;
   TFloat gradhess1;

   TFloat weight;
   if(bWeight) {
      weight = 0;
   }
   do {
      const typename TFloat::TInt iTensorBinCombined = TFloat::TInt::Load(pInputData);
      pInputData += TFloat::TInt::k_cSIMDPack;
      if(bFixedSizePack) {
         // If we have a fixed sized cCompilerPack then the compiler should be able to unroll
         // the loop below. The compiler can only do that though if it can guarantee that all
         // iterations of the loop have the name number of loops.  Setting cShift here allows this
         cShift = cShiftReset;
      }
      do {
         if(bWeight) {
            gradient *= weight;
            if(bHessian) {
               hessian *= weight;
            }
            weight = TFloat::Load(pWeight);
            pWeight += TFloat::k_cSIMDPack;
         }

         if(!bHessian) {
            gradhess0 = gradient;
         } else {
            TFloat::Interleaf(gradient, hessian, gradhess0, gradhess1);
         }

         gradient = TFloat::Load(pGradientAndHessian);
         if(bHessian) {
            hessian = TFloat::Load(&pGradientAndHessian[TFloat::k_cSIMDPack]);
         }
         pGradientAndHessian += (bHessian ? size_t{2} : size_t{1}) * TFloat::k_cSIMDPack;

         bin0 += gradhess0;
         if(bHessian) {
            bin1 += gradhess1;
         }

         if(!bHessian) {
            bin0.template Store<cFixedShift>(aBins, iTensorBinPrev);
         } else {
            TFloat::template DoubleStore<cFixedShift>(aBins, iTensorBinPrev, bin0, bin1);
         }

         // This load is a gathering load and is the main bottleneck to EBMs. We want
         // to give it as much time as possible to execute the load before using the bin0
         // or bin1 values, which we do since the addition of the gradient
         // to the bin0 value above is almost an entire loop away. We would like
         // this load to be as early as possible, but we cannot move it before the bin0 Store
         // operation since that operation can change the memory that we're loading here, but the
         // optimal solution is to have as little work done between the Store and this gathering load
         // All the other loads are predictable and should be much faster than this gathering load.

         if(!bHessian) {
            bin0 = TFloat::template Load<cFixedShift>(aBins, iTensorBin);
         } else {
            TFloat::template DoubleLoad<cFixedShift>(aBins, iTensorBin, bin0, bin1);
         }

         iTensorBinPrev = iTensorBin;
         iTensorBin = ((iTensorBinCombined >> cShift) & maskBits) + offsets;
         if(bHessian) {
            iTensorBin = PermuteForInterleaf(iTensorBin);
         }

         cShift -= cBitsPerItemMax;
      } while(0 <= cShift);
      if(!bFixedSizePack) {
         cShift = cShiftReset;
      }
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);

   if(bWeight) {
      gradient *= weight;
      if(bHessian) {
         hessian *= weight;
      }
   }
   if(!bHessian) {
      gradhess0 = gradient;
   } else {
      TFloat::Interleaf(gradient, hessian, gradhess0, gradhess1);
   }

   bin0 += gradhess0;
   if(bHessian) {
      bin1 += gradhess1;
   }

   if(!bHessian) {
      bin0.template Store<cFixedShift>(aBins, iTensorBinPrev);
   } else {
      TFloat::template DoubleStore<cFixedShift>(aBins, iTensorBinPrev, bin0, bin1);
   }
}
#endif // 0 < PARALLEL_BINS_BYTES_MAX

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bParallel,
      bool bCollapsed,
      size_t cCompilerScores,
      int cCompilerPack,
      typename std::enable_if<!bParallel && 1 == TFloat::k_cSIMDPack && !bCollapsed && 1 == cCompilerScores,
            int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

   static_assert(1 == cCompilerScores, "This specialization of BinSumsBoostingInternal cannot handle multiclass.");
   static constexpr bool bFixedSizePack = k_cItemsPerBitPackUndefined != cCompilerPack;

#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pParams);
   EBM_ASSERT(1 <= pParams->m_cSamples);
   EBM_ASSERT(0 == pParams->m_cSamples % size_t{TFloat::k_cSIMDPack});
   EBM_ASSERT(0 == pParams->m_cSamples % size_t{(bFixedSizePack ? cCompilerPack : 1) * TFloat::k_cSIMDPack});
   EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
   EBM_ASSERT(nullptr != pParams->m_aFastBins);
   EBM_ASSERT(size_t{1} == pParams->m_cScores);
#endif // GPU_COMPILE

   const size_t cSamples = pParams->m_cSamples;

   auto* const aBins =
         reinterpret_cast<BinBase*>(pParams->m_aFastBins)
               ->Specialize<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian, size_t{1}>();

   const typename TFloat::T* pGradientAndHessian =
         reinterpret_cast<const typename TFloat::T*>(pParams->m_aGradientsAndHessians);
   const typename TFloat::T* const pGradientsAndHessiansEnd =
         pGradientAndHessian + (bHessian ? size_t{2} : size_t{1}) * cSamples;

   static constexpr typename TFloat::TInt::T cBytesPerBin = static_cast<typename TFloat::TInt::T>(
         GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(false, false, bHessian, size_t{1}));

   // The compiler is normally pretty good about optimizing multiplications into shifts when possible
   // BUT, when compiling for SIMD, it seems to use a SIMD multiplication instruction instead of shifts
   // even when the multiplication has a fixed compile time constant value that is a power of 2, so
   // we manually convert the multiplications into shifts.
   //
   // We also have tried the Multiply templated function that is designed to convert multiplications
   // into shifts, but using that templated function breaks the compiler optimization that unrolls
   // the bitpacking loop.
   //
   constexpr static bool bSmall = 4 == cBytesPerBin;
   constexpr static bool bMed = 8 == cBytesPerBin;
   constexpr static bool bLarge = 16 == cBytesPerBin;
   static_assert(bSmall || bMed || bLarge, "cBytesPerBin size must be small, medium, or large");
   constexpr static int cFixedShift = bSmall ? 2 : bMed ? 3 : 4;
   static_assert(1 << cFixedShift == cBytesPerBin, "cFixedShift must match the BinSize");

   const int cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pParams->m_cPack);
#ifndef GPU_COMPILE
   EBM_ASSERT(1 <= cItemsPerBitPack);
   EBM_ASSERT(cItemsPerBitPack <= COUNT_BITS(typename TFloat::TInt::T));
#endif // GPU_COMPILE

   const int cBitsPerItemMax = GetCountBits<typename TFloat::TInt::T>(cItemsPerBitPack);
#ifndef GPU_COMPILE
   EBM_ASSERT(1 <= cBitsPerItemMax);
   EBM_ASSERT(cBitsPerItemMax <= COUNT_BITS(typename TFloat::TInt::T));
#endif // GPU_COMPILE

   const typename TFloat::TInt::T maskBits = MakeLowMask<typename TFloat::TInt::T>(cBitsPerItemMax);

   const typename TFloat::TInt::T* pInputData = reinterpret_cast<const typename TFloat::TInt::T*>(pParams->m_aPacked);
#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pInputData);
#endif // GPU_COMPILE

   size_t iTensorBin;
   const int cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;
   int cShift;
   if(bFixedSizePack) {
      iTensorBin = static_cast<size_t>(*pInputData & maskBits);
      ++pInputData;
   } else {
      cShift = static_cast<int>((cSamples >> TFloat::k_cSIMDShift) % static_cast<size_t>(cItemsPerBitPack)) *
            cBitsPerItemMax;
      iTensorBin = static_cast<size_t>((*pInputData >> cShift) & maskBits);
      cShift -= cBitsPerItemMax;
      if(cShift < 0) {
         cShift = cShiftReset;
         ++pInputData;
      }
   }

   const typename TFloat::T* pWeight;
   if(bWeight) {
      pWeight = reinterpret_cast<const typename TFloat::T*>(pParams->m_aWeights);
#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
   }

   // We want to structure the loop below so that the load happens immediately after the store
   // because that allows the maximum possible time for the gathering load to happen in the CPU pipeline.
   // To do that we put the store at the top and the load below. But now we need to exectute a
   // store on the first iteration, so load the values from memory here that we'll then store
   // back on the first loop iteration
   auto* pGradientPair = aBins->GetGradientPairs();
   typename TFloat::T binGrad = pGradientPair->m_sumGradients;
   typename TFloat::T binHess;
   if(bHessian) {
      binHess = pGradientPair->GetHess();
   }

   typename TFloat::T gradient = 0;
   typename TFloat::T hessian;
   if(bHessian) {
      hessian = 0;
   }

   typename TFloat::T weight;
   if(bWeight) {
      weight = 0;
   }
   do {
      const typename TFloat::TInt::T iTensorBinCombined = *pInputData;
      ++pInputData;
      if(bFixedSizePack) {
         // If we have a fixed sized cCompilerPack then the compiler should be able to unroll
         // the loop below. The compiler can only do that though if it can guarantee that all
         // iterations of the loop have the name number of loops.  Setting cShift here allows this
         cShift = cShiftReset;
      }
      do {
         if(bWeight) {
            gradient *= weight;
            if(bHessian) {
               hessian *= weight;
            }
            weight = *pWeight;
            ++pWeight;
         }

         binGrad += gradient;
         if(bHessian) {
            binHess += hessian;
         }

         gradient = pGradientAndHessian[0];
         if(bHessian) {
            hessian = pGradientAndHessian[1];
         }

         pGradientPair->m_sumGradients = binGrad;
         if(bHessian) {
            pGradientPair->SetHess(binHess);
         }

         // these loads are unpredictable loads, and therefore take the most time in this loop
         pGradientPair = IndexBin(aBins, iTensorBin << cFixedShift)->GetGradientPairs();
         binGrad = pGradientPair->m_sumGradients;
         if(bHessian) {
            binHess = pGradientPair->GetHess();
         }

         iTensorBin = static_cast<size_t>((iTensorBinCombined >> cShift) & maskBits);

         pGradientAndHessian += bHessian ? size_t{2} : size_t{1};

         cShift -= cBitsPerItemMax;
      } while(0 <= cShift);
      if(!bFixedSizePack) {
         cShift = cShiftReset;
      }
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);

   if(bWeight) {
      gradient *= weight;
      if(bHessian) {
         hessian *= weight;
      }
   }
   binGrad += gradient;
   if(bHessian) {
      binHess += hessian;
   }

   pGradientPair->m_sumGradients = binGrad;
   if(bHessian) {
      pGradientPair->SetHess(binHess);
   }
}

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bParallel,
      bool bCollapsed,
      size_t cCompilerScores,
      int cCompilerPack,
      typename std::enable_if<!bParallel && 1 != TFloat::k_cSIMDPack && !bCollapsed && 1 == cCompilerScores,
            int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

   static_assert(1 == cCompilerScores, "This specialization of BinSumsBoostingInternal cannot handle multiclass.");
   static constexpr bool bFixedSizePack = k_cItemsPerBitPackUndefined != cCompilerPack;

#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pParams);
   EBM_ASSERT(1 <= pParams->m_cSamples);
   EBM_ASSERT(0 == pParams->m_cSamples % size_t{TFloat::k_cSIMDPack});
   EBM_ASSERT(0 == pParams->m_cSamples % size_t{(bFixedSizePack ? cCompilerPack : 1) * TFloat::k_cSIMDPack});
   EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
   EBM_ASSERT(nullptr != pParams->m_aFastBins);
   EBM_ASSERT(size_t{1} == pParams->m_cScores);
#endif // GPU_COMPILE

   const size_t cSamples = pParams->m_cSamples;

   auto* const aBins =
         reinterpret_cast<BinBase*>(pParams->m_aFastBins)
               ->Specialize<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian, size_t{1}>();

   const typename TFloat::T* pGradientAndHessian =
         reinterpret_cast<const typename TFloat::T*>(pParams->m_aGradientsAndHessians);
   const typename TFloat::T* const pGradientsAndHessiansEnd =
         pGradientAndHessian + (bHessian ? size_t{2} : size_t{1}) * cSamples;

   static constexpr typename TFloat::TInt::T cBytesPerBin = static_cast<typename TFloat::TInt::T>(
         GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(false, false, bHessian, size_t{1}));

   // The compiler is normally pretty good about optimizing multiplications into shifts when possible
   // BUT, when compiling for SIMD, it seems to use a SIMD multiplication instruction instead of shifts
   // even when the multiplication has a fixed compile time constant value that is a power of 2, so
   // we manually convert the multiplications into shifts.
   //
   // We also have tried the Multiply templated function that is designed to convert multiplications
   // into shifts, but using that templated function breaks the compiler optimization that unrolls
   // the bitpacking loop.
   //
   constexpr static bool bSmall = 4 == cBytesPerBin;
   constexpr static bool bMed = 8 == cBytesPerBin;
   constexpr static bool bLarge = 16 == cBytesPerBin;
   static_assert(bSmall || bMed || bLarge, "cBytesPerBin size must be small, medium, or large");
   constexpr static int cFixedShift = bSmall ? 2 : bMed ? 3 : 4;
   static_assert(1 << cFixedShift == cBytesPerBin, "cFixedShift must match the BinSize");

   const int cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pParams->m_cPack);
#ifndef GPU_COMPILE
   EBM_ASSERT(1 <= cItemsPerBitPack);
   EBM_ASSERT(cItemsPerBitPack <= COUNT_BITS(typename TFloat::TInt::T));
#endif // GPU_COMPILE

   const int cBitsPerItemMax = GetCountBits<typename TFloat::TInt::T>(cItemsPerBitPack);
#ifndef GPU_COMPILE
   EBM_ASSERT(1 <= cBitsPerItemMax);
   EBM_ASSERT(cBitsPerItemMax <= COUNT_BITS(typename TFloat::TInt::T));
#endif // GPU_COMPILE

   const typename TFloat::TInt maskBits = MakeLowMask<typename TFloat::TInt::T>(cBitsPerItemMax);

   const typename TFloat::TInt::T* pInputData = reinterpret_cast<const typename TFloat::TInt::T*>(pParams->m_aPacked);
#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pInputData);
#endif // GPU_COMPILE

   typename TFloat::TInt iTensorBin;
   const int cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;
   int cShift;
   if(bFixedSizePack) {
      iTensorBin = TFloat::TInt::Load(pInputData) & maskBits;
      iTensorBin = iTensorBin << cFixedShift;
      pInputData += TFloat::TInt::k_cSIMDPack;
   } else {
      cShift = static_cast<int>((cSamples >> TFloat::k_cSIMDShift) % static_cast<size_t>(cItemsPerBitPack)) *
            cBitsPerItemMax;
      iTensorBin = (TFloat::TInt::Load(pInputData) >> cShift) & maskBits;
      iTensorBin = iTensorBin << cFixedShift;
      cShift -= cBitsPerItemMax;
      if(cShift < 0) {
         cShift = cShiftReset;
         pInputData += TFloat::TInt::k_cSIMDPack;
      }
   }

   const typename TFloat::T* pWeight;
   if(bWeight) {
      pWeight = reinterpret_cast<const typename TFloat::T*>(pParams->m_aWeights);
#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
   }

   do {
      const typename TFloat::TInt iTensorBinCombined = TFloat::TInt::Load(pInputData);
      pInputData += TFloat::TInt::k_cSIMDPack;
      if(bFixedSizePack) {
         // If we have a fixed sized cCompilerPack then the compiler should be able to unroll
         // the loop below. The compiler can only do that though if it can guarantee that all
         // iterations of the loop have the name number of loops.  Setting cShift here allows this
         cShift = cShiftReset;
      }
      do {
         TFloat weight;
         if(bWeight) {
            weight = TFloat::Load(pWeight);
            pWeight += TFloat::k_cSIMDPack;
         }

         TFloat gradient = TFloat::Load(pGradientAndHessian);
         TFloat hessian;
         if(bHessian) {
            hessian = TFloat::Load(&pGradientAndHessian[TFloat::k_cSIMDPack]);
         }
         pGradientAndHessian += (bHessian ? size_t{2} : size_t{1}) * TFloat::k_cSIMDPack;

         if(bWeight) {
            gradient *= weight;
            if(bHessian) {
               hessian *= weight;
            }
         }

         // BEWARE: unless we generate a separate histogram for each SIMD stream and later merge them, pBin can
         // point to the same bin in multiple samples within the SIMD pack, so we need to serialize fetching sums
         if(bHessian) {
            TFloat::Execute(
                  [aBins](int,
                        const typename TFloat::TInt::T i,
                        const typename TFloat::T grad,
                        const typename TFloat::T hess) {
                     COVER(COV_BinSumsBoostingInternal_NoWeight_NoReplication_Hessian);
                     auto* const pBin = IndexBin(aBins, static_cast<size_t>(i));
                     auto* const pGradientPair = pBin->GetGradientPairs();
                     typename TFloat::T binGrad = pGradientPair->m_sumGradients;
                     typename TFloat::T binHess = pGradientPair->GetHess();
                     binGrad += grad;
                     binHess += hess;
                     pGradientPair->m_sumGradients = binGrad;
                     pGradientPair->SetHess(binHess);
                  },
                  iTensorBin,
                  gradient,
                  hessian);
         } else {
            TFloat::Execute(
                  [aBins](int, const typename TFloat::TInt::T i, const typename TFloat::T grad) {
                     COVER(COV_BinSumsBoostingInternal_NoWeight_NoReplication_NoHessian);
                     auto* const pBin = IndexBin(aBins, static_cast<size_t>(i));
                     auto* const pGradientPair = pBin->GetGradientPairs();
                     typename TFloat::T binGrad = pGradientPair->m_sumGradients;
                     binGrad += grad;
                     pGradientPair->m_sumGradients = binGrad;
                  },
                  iTensorBin,
                  gradient);
         }

         iTensorBin = (iTensorBinCombined >> cShift) & maskBits;
         iTensorBin = iTensorBin << cFixedShift;

         cShift -= cBitsPerItemMax;
      } while(0 <= cShift);
      if(!bFixedSizePack) {
         cShift = cShiftReset;
      }
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);
}

/*

This code no longer works because we now prefetch the gathering loads by moving all the bit packed data one
bitpack earlier and leave 1 bitpack at the end with zeros

This code works, but seems to take about two times longer than the non-parallel version where bParallel is false
I think this is because for multiclass we load at least 6 floats from the same location (3 gradients 3 hessians) or
more and those are predictable loads after the first one, and the CPU is good at predicting those loads especially
for non-gathering loads. The TFloat::Execute function creates cSIMDPack separate loads which means the CPU will
learn that each unpredictable load is followed by 5+ predictable ones at the same assembly instruction

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bParallel,
      bool bCollapsed,
      size_t cCompilerScores,
      int cCompilerPack,
      typename std::enable_if<bParallel && !bCollapsed && 1 != cCompilerScores, int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

   static constexpr size_t cArrayScores = GetArrayScores(cCompilerScores);

#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pParams);
   EBM_ASSERT(1 <= pParams->m_cSamples);
   EBM_ASSERT(0 == pParams->m_cSamples % size_t{TFloat::k_cSIMDPack});
   EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
   EBM_ASSERT(nullptr != pParams->m_aFastBins);
   EBM_ASSERT(k_dynamicScores == cCompilerScores || cCompilerScores == pParams->m_cScores);
#endif // GPU_COMPILE

   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pParams->m_cScores);

   const size_t cSamples = pParams->m_cSamples;

   auto* const aBins =
         reinterpret_cast<BinBase*>(pParams->m_aFastBins)
               ->Specialize<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian, cArrayScores>();

   const typename TFloat::T* pGradientAndHessian =
         reinterpret_cast<const typename TFloat::T*>(pParams->m_aGradientsAndHessians);
   const typename TFloat::T* const pGradientsAndHessiansEnd =
         pGradientAndHessian + (bHessian ? size_t{2} : size_t{1}) * cScores * cSamples;

   const typename TFloat::TInt::T cBytesPerBin = static_cast<typename TFloat::TInt::T>(
         GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(false, false, bHessian, cScores));

   const int cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pParams->m_cPack);
#ifndef GPU_COMPILE
   EBM_ASSERT(1 <= cItemsPerBitPack);
   EBM_ASSERT(cItemsPerBitPack <= COUNT_BITS(typename TFloat::TInt::T));
#endif // GPU_COMPILE

   const int cBitsPerItemMax = GetCountBits<typename TFloat::TInt::T>(cItemsPerBitPack);
#ifndef GPU_COMPILE
   EBM_ASSERT(1 <= cBitsPerItemMax);
   EBM_ASSERT(cBitsPerItemMax <= COUNT_BITS(typename TFloat::TInt::T));
#endif // GPU_COMPILE

   const typename TFloat::TInt maskBits = MakeLowMask<typename TFloat::TInt::T>(cBitsPerItemMax);

   const typename TFloat::TInt::T* pInputData = reinterpret_cast<const typename TFloat::TInt::T*>(pParams->m_aPacked);
#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pInputData);
#endif // GPU_COMPILE

   const int cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;
   int cShift =
         static_cast<int>(((cSamples >> TFloat::k_cSIMDShift) - size_t{1}) % static_cast<size_t>(cItemsPerBitPack)) *
         cBitsPerItemMax;

   const typename TFloat::T* pWeight;
   if(bWeight) {
      pWeight = reinterpret_cast<const typename TFloat::T*>(pParams->m_aWeights);
#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
   }

   const typename TFloat::TInt offsets =
         TFloat::TInt::MakeIndexes() * static_cast<typename TFloat::TInt::T>(pParams->m_cBytesFastBins);

   static constexpr ptrdiff_t k_offsetGrad =
         Bin<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian>::k_offsetGrad;
   static constexpr ptrdiff_t k_offsetHess =
         Bin<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian>::k_offsetHess;

   typename TFloat::T* const pGrad = IndexByte(reinterpret_cast<typename TFloat::T*>(aBins), k_offsetGrad);
   typename TFloat::T* pHess;
   if(bHessian) {
      pHess = IndexByte(reinterpret_cast<typename TFloat::T*>(aBins), static_cast<size_t>(k_offsetHess));
   }

   static constexpr size_t k_cBytesGradHess = sizeof(GradientPair<typename TFloat::T, bHessian>);

   do {
      const typename TFloat::TInt iTensorBinCombined = TFloat::TInt::Load(pInputData);
      pInputData += TFloat::TInt::k_cSIMDPack;
      do {
         TFloat weight;
         if(bWeight) {
            weight = TFloat::Load(pWeight);
            pWeight += TFloat::k_cSIMDPack;
         }

         typename TFloat::TInt iTensorBin = (iTensorBinCombined >> cShift) & maskBits;

         // normally the compiler is better at optimimizing multiplications into shifs, but it isn't better
         // if TFloat is a SIMD type. For SIMD shifts & adds will almost always be better than multiplication if
         // there are low numbers of shifts, which should be the case for anything with a compile time constant here
         iTensorBin = Multiply < typename TFloat::TInt, typename TFloat::TInt::T,
         k_dynamicScores != cCompilerScores && 1 != TFloat::k_cSIMDPack,
         static_cast<typename TFloat::TInt::T>(GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(
               false, false, bHessian, cCompilerScores)) > (iTensorBin, cBytesPerBin);

         iTensorBin = iTensorBin + offsets;

         size_t iScore = 0;
         do {
            TFloat gradientBin = TFloat::template Load<0>(IndexByte(pGrad, iScore * k_cBytesGradHess), iTensorBin);
            TFloat hessianBin;
            if(bHessian) {
               hessianBin = TFloat::template Load<0>(IndexByte(pHess, iScore * k_cBytesGradHess), iTensorBin);
            }

            TFloat gradient = TFloat::Load(
                  &pGradientAndHessian[iScore << (bHessian ? (TFloat::k_cSIMDShift + 1) : TFloat::k_cSIMDShift)]);
            TFloat hessian;
            if(bHessian) {
               hessian = TFloat::Load(
                  &pGradientAndHessian[(iScore << (TFloat::k_cSIMDShift + 1)) + TFloat::k_cSIMDPack]);
            }

            if(bWeight) {
               gradient *= weight;
               if(bHessian) {
                  hessian *= weight;
               }
            }

            gradientBin += gradient;
            if(bHessian) {
               hessianBin += hessian;
            }

            gradientBin.template Store<0>(IndexByte(pGrad, iScore * k_cBytesGradHess), iTensorBin);
            if(bHessian) {
               hessianBin.template Store<0>(IndexByte(pHess, iScore * k_cBytesGradHess), iTensorBin);
            }
            ++iScore;
         } while(cScores != iScore);

         pGradientAndHessian += cScores << (bHessian ? (TFloat::k_cSIMDShift + 1) : TFloat::k_cSIMDShift);

         cShift -= cBitsPerItemMax;
      } while(0 <= cShift);
      cShift = cShiftReset;
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);
}
*/

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bParallel,
      bool bCollapsed,
      size_t cCompilerScores,
      int cCompilerPack,
      typename std::enable_if<!bParallel && !bCollapsed && 1 != cCompilerScores, int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

   static_assert(!bParallel, "BinSumsBoosting specialization for collapsed does not handle parallel bins.");

   static constexpr size_t cArrayScores = GetArrayScores(cCompilerScores);

#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pParams);
   EBM_ASSERT(1 <= pParams->m_cSamples);
   EBM_ASSERT(0 == pParams->m_cSamples % size_t{TFloat::k_cSIMDPack});
   EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
   EBM_ASSERT(nullptr != pParams->m_aFastBins);
   EBM_ASSERT(k_dynamicScores == cCompilerScores || cCompilerScores == pParams->m_cScores);
#endif // GPU_COMPILE

   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pParams->m_cScores);

   const size_t cSamples = pParams->m_cSamples;

   auto* const aBins =
         reinterpret_cast<BinBase*>(pParams->m_aFastBins)
               ->Specialize<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian, cArrayScores>();

   const typename TFloat::T* pGradientAndHessian =
         reinterpret_cast<const typename TFloat::T*>(pParams->m_aGradientsAndHessians);
   const typename TFloat::T* const pGradientsAndHessiansEnd =
         pGradientAndHessian + (bHessian ? size_t{2} : size_t{1}) * cScores * cSamples;

   const typename TFloat::TInt::T cBytesPerBin = static_cast<typename TFloat::TInt::T>(
         GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(false, false, bHessian, cScores));

   const int cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pParams->m_cPack);
#ifndef GPU_COMPILE
   EBM_ASSERT(1 <= cItemsPerBitPack);
   EBM_ASSERT(cItemsPerBitPack <= COUNT_BITS(typename TFloat::TInt::T));
#endif // GPU_COMPILE

   const int cBitsPerItemMax = GetCountBits<typename TFloat::TInt::T>(cItemsPerBitPack);
#ifndef GPU_COMPILE
   EBM_ASSERT(1 <= cBitsPerItemMax);
   EBM_ASSERT(cBitsPerItemMax <= COUNT_BITS(typename TFloat::TInt::T));
#endif // GPU_COMPILE

   const typename TFloat::TInt maskBits = MakeLowMask<typename TFloat::TInt::T>(cBitsPerItemMax);

   const typename TFloat::TInt::T* pInputData = reinterpret_cast<const typename TFloat::TInt::T*>(pParams->m_aPacked);
#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pInputData);
#endif // GPU_COMPILE

   typename TFloat::TInt iTensorBin;
   const int cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;
   int cShift;
   cShift = static_cast<int>((cSamples >> TFloat::k_cSIMDShift) % static_cast<size_t>(cItemsPerBitPack)) *
         cBitsPerItemMax;
   iTensorBin = (TFloat::TInt::Load(pInputData) >> cShift) & maskBits;
   iTensorBin = Multiply < typename TFloat::TInt, typename TFloat::TInt::T,
   k_dynamicScores != cCompilerScores && 1 != TFloat::k_cSIMDPack,
   static_cast<typename TFloat::TInt::T>(GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(
         false, false, bHessian, cCompilerScores)) > (iTensorBin, cBytesPerBin);
   cShift -= cBitsPerItemMax;
   if(cShift < 0) {
      cShift = cShiftReset;
      pInputData += TFloat::TInt::k_cSIMDPack;
   }

   const typename TFloat::T* pWeight;
   if(bWeight) {
      pWeight = reinterpret_cast<const typename TFloat::T*>(pParams->m_aWeights);
#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
   }

   do {
      const typename TFloat::TInt iTensorBinCombined = TFloat::TInt::Load(pInputData);
      pInputData += TFloat::TInt::k_cSIMDPack;
      do {
         TFloat weight;
         if(bWeight) {
            weight = TFloat::Load(pWeight);
            pWeight += TFloat::k_cSIMDPack;
         }

         Bin<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian, cArrayScores>*
               apBins[TFloat::k_cSIMDPack];
         TFloat::TInt::Execute(
               [aBins, &apBins](const int i, const typename TFloat::TInt::T x) {
                  apBins[i] = IndexBin(aBins, static_cast<size_t>(x));
               },
               iTensorBin);

         size_t iScore = 0;
         do {
            if(bHessian) {
               TFloat gradient = TFloat::Load(&pGradientAndHessian[iScore << (TFloat::k_cSIMDShift + 1)]);
               TFloat hessian =
                     TFloat::Load(&pGradientAndHessian[(iScore << (TFloat::k_cSIMDShift + 1)) + TFloat::k_cSIMDPack]);
               if(bWeight) {
                  gradient *= weight;
                  hessian *= weight;
               }
               TFloat::Execute(
                     [apBins, iScore](const int i, const typename TFloat::T grad, const typename TFloat::T hess) {
                        // BEWARE: unless we generate a separate histogram for each SIMD stream and later merge them,
                        // pBin can point to the same bin in multiple samples within the SIMD pack, so we need to
                        // serialize fetching sums
                        auto* const pBin = apBins[i];
                        auto* const aGradientPair = pBin->GetGradientPairs();
                        auto* const pGradientPair = &aGradientPair[iScore];
                        typename TFloat::T binGrad = pGradientPair->m_sumGradients;
                        typename TFloat::T binHess = pGradientPair->GetHess();
                        binGrad += grad;
                        binHess += hess;
                        pGradientPair->m_sumGradients = binGrad;
                        pGradientPair->SetHess(binHess);
                     },
                     gradient,
                     hessian);
            } else {
               TFloat gradient = TFloat::Load(&pGradientAndHessian[iScore << TFloat::k_cSIMDShift]);
               if(bWeight) {
                  gradient *= weight;
               }
               TFloat::Execute(
                     [apBins, iScore](const int i, const typename TFloat::T grad) {
                        auto* const pBin = apBins[i];
                        auto* const aGradientPair = pBin->GetGradientPairs();
                        auto* const pGradientPair = &aGradientPair[iScore];
                        pGradientPair->m_sumGradients += grad;
                     },
                     gradient);
            }
            ++iScore;
         } while(cScores != iScore);
         pGradientAndHessian += cScores << (bHessian ? (TFloat::k_cSIMDShift + 1) : TFloat::k_cSIMDShift);

         iTensorBin = (iTensorBinCombined >> cShift) & maskBits;

         // normally the compiler is better at optimimizing multiplications into shifs, but it isn't better
         // if TFloat is a SIMD type. For SIMD shifts & adds will almost always be better than multiplication if
         // there are low numbers of shifts, which should be the case for anything with a compile time constant here
         iTensorBin = Multiply < typename TFloat::TInt, typename TFloat::TInt::T,
         k_dynamicScores != cCompilerScores && 1 != TFloat::k_cSIMDPack,
         static_cast<typename TFloat::TInt::T>(GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(
               false, false, bHessian, cCompilerScores)) > (iTensorBin, cBytesPerBin);

         cShift -= cBitsPerItemMax;
      } while(0 <= cShift);
      cShift = cShiftReset;
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);
}

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bParallel,
      bool bCollapsed,
      size_t cCompilerScores,
      int cCompilerPack>
struct BitPack final {
   INLINE_ALWAYS static void Func(BinSumsBoostingBridge* const pParams) {

      static_assert(!bCollapsed, "Cannot be bCollapsed since there would be no bitpacking");
      static_assert(cCompilerPack <= COUNT_BITS(typename TFloat::TInt::T), "cCompilerPack must fit into the bitpack");

      if(cCompilerPack == pParams->m_cPack) {
         size_t cSamples = pParams->m_cSamples;
         const size_t cRemnants = cSamples % static_cast<size_t>(cCompilerPack * TFloat::k_cSIMDPack);
         if(0 != cRemnants) {
            pParams->m_cSamples = cRemnants;
            BinSumsBoostingInternal<TFloat,
                  bHessian,
                  bWeight,
                  bParallel,
                  bCollapsed,
                  cCompilerScores,
                  k_cItemsPerBitPackUndefined>(pParams);

            cSamples -= cRemnants;
            if(0 == cSamples) {
               return;
            }
            pParams->m_cSamples = cSamples;
            if(bWeight) {
               EBM_ASSERT(nullptr != pParams->m_aWeights);
               pParams->m_aWeights = IndexByte(pParams->m_aWeights, sizeof(typename TFloat::T) * cRemnants);
            } else {
               EBM_ASSERT(nullptr == pParams->m_aWeights);
            }

            const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pParams->m_cScores);

            EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
            pParams->m_aGradientsAndHessians = IndexByte(pParams->m_aGradientsAndHessians,
                  sizeof(typename TFloat::T) * (bHessian ? size_t{2} : size_t{1}) * cScores * cRemnants);
         }
         BinSumsBoostingInternal<TFloat, bHessian, bWeight, bParallel, bCollapsed, cCompilerScores, cCompilerPack>(
               pParams);
      } else {
         BitPack<TFloat,
               bHessian,
               bWeight,
               bParallel,
               bCollapsed,
               cCompilerScores,
               GetNextBitPack<typename TFloat::TInt::T>(cCompilerPack, k_cItemsPerBitPackBoostingMin)>::Func(pParams);
      }
   }
};
template<typename TFloat, bool bHessian, bool bWeight, bool bParallel, bool bCollapsed, size_t cCompilerScores>
struct BitPack<TFloat, bHessian, bWeight, bParallel, bCollapsed, cCompilerScores, k_cItemsPerBitPackUndefined>
      final {
   INLINE_ALWAYS static void Func(BinSumsBoostingBridge* const pParams) {

      static_assert(!bCollapsed, "Cannot be bCollapsed since there would be no bitpacking");

      BinSumsBoostingInternal<TFloat,
            bHessian,
            bWeight,
            bParallel,
            bCollapsed,
            cCompilerScores,
            k_cItemsPerBitPackUndefined>(
            pParams);
   }
};

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bParallel,
      bool bCollapsed,
      size_t cCompilerScores,
      typename std::enable_if<!bCollapsed && 1 == cCompilerScores, int>::type = 0>
INLINE_RELEASE_TEMPLATED static void BitPackBoosting(BinSumsBoostingBridge* const pParams) {
   BitPack<TFloat,
         bHessian,
         bWeight,
         bParallel,
         bCollapsed,
         cCompilerScores,
         GetFirstBitPack<typename TFloat::TInt::T>(
               k_cItemsPerBitPackBoostingMax, k_cItemsPerBitPackBoostingMin)>::Func(pParams);
}
template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bParallel,
      bool bCollapsed,
      size_t cCompilerScores,
      typename std::enable_if<bCollapsed || 1 != cCompilerScores, int>::type = 0>
INLINE_RELEASE_TEMPLATED static void BitPackBoosting(BinSumsBoostingBridge* const pParams) {
   BinSumsBoostingInternal<TFloat,
         bHessian,
         bWeight,
         bParallel,
         bCollapsed,
         cCompilerScores,
         k_cItemsPerBitPackUndefined>(
         pParams);
}

template<typename TFloat, bool bHessian, bool bWeight, bool bParallel, bool bCollapsed, size_t cCompilerScores>
GPU_GLOBAL static void RemoteBinSumsBoosting(BinSumsBoostingBridge* const pParams) {
   BitPackBoosting<TFloat, bHessian, bWeight, bParallel, bCollapsed, cCompilerScores>(pParams);
}

template<typename TFloat, bool bHessian, bool bWeight, bool bParallel, bool bCollapsed, size_t cCompilerScores>
INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorBinSumsBoosting(BinSumsBoostingBridge* const pParams) {
   return TFloat::template OperatorBinSumsBoosting<bHessian, bWeight, bParallel, bCollapsed, cCompilerScores>(
         pParams);
}

template<typename TFloat, bool bHessian, bool bWeight, bool bParallel, bool bCollapsed, size_t cPossibleScores>
struct CountClassesBoosting final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsBoostingBridge* const pParams) {
      if(cPossibleScores == pParams->m_cScores) {
         return OperatorBinSumsBoosting<TFloat, bHessian, bWeight, bParallel, bCollapsed, cPossibleScores>(
               pParams);
      } else {
         return CountClassesBoosting<TFloat, bHessian, bWeight, bParallel, bCollapsed, cPossibleScores + 1>::Func(
               pParams);
      }
   }
};
template<typename TFloat, bool bHessian, bool bWeight, bool bParallel, bool bCollapsed>
struct CountClassesBoosting<TFloat, bHessian, bWeight, bParallel, bCollapsed, k_cCompilerScoresMax + 1> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsBoostingBridge* const pParams) {
      return OperatorBinSumsBoosting<TFloat, bHessian, bWeight, bParallel, bCollapsed, k_dynamicScores>(pParams);
   }
};

#if 0 < PARALLEL_BINS_BYTES_MAX
template<typename TFloat, bool bHessian, bool bWeight, bool bParallel, typename std::enable_if<bParallel, int>::type = 0>
INLINE_RELEASE_TEMPLATED static ErrorEbm DoneParallel(BinSumsBoostingBridge* const pParams) {

   // some scatter/gather SIMD instructions are often signed integers and we only use the positive range
   static_assert(0 == PARALLEL_BINS_BYTES_MAX ||
               !IsConvertError<typename std::make_signed<typename TFloat::TInt::T>::type>(PARALLEL_BINS_BYTES_MAX - 1),
         "PARALLEL_BINS_BYTES_MAX is too large");

   EBM_ASSERT(k_cItemsPerBitPackUndefined != pParams->m_cPack); // excluded in caller
   static constexpr bool bCollapsed = false;
   EBM_ASSERT(1 == pParams->m_cScores); // excluded in caller
   return OperatorBinSumsBoosting<TFloat, bHessian, bWeight, bParallel, bCollapsed, k_oneScore>(pParams);
}
#endif // 0 < PARALLEL_BINS_BYTES_MAX


template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bParallel,
      typename std::enable_if<!bParallel && bHessian, int>::type = 0>
INLINE_RELEASE_TEMPLATED static ErrorEbm DoneParallel(BinSumsBoostingBridge* const pParams) {
   if(k_cItemsPerBitPackUndefined == pParams->m_cPack) {
      static constexpr bool bCollapsed = true;
      if(size_t{1} == pParams->m_cScores) {
         return OperatorBinSumsBoosting<TFloat, bHessian, bWeight, bParallel, bCollapsed, k_oneScore>(pParams);
      } else {
         // muticlass, but for a collapsed so don't optimize for it
         return OperatorBinSumsBoosting<TFloat, bHessian, bWeight, bParallel, bCollapsed, k_dynamicScores>(
               pParams);
      }
   } else {
      static constexpr bool bCollapsed = false;
      if(size_t{1} == pParams->m_cScores) {
         return OperatorBinSumsBoosting<TFloat, bHessian, bWeight, bParallel, bCollapsed, k_oneScore>(pParams);
      } else {
         // muticlass
         return CountClassesBoosting<TFloat, bHessian, bWeight, bParallel, bCollapsed, k_cCompilerScoresStart>::
               Func(pParams);
      }
   }
}

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bParallel,
      typename std::enable_if<!bParallel && !bHessian, int>::type = 0>
INLINE_RELEASE_TEMPLATED static ErrorEbm DoneParallel(BinSumsBoostingBridge* const pParams) {
   if(k_cItemsPerBitPackUndefined == pParams->m_cPack) {
      static constexpr bool bCollapsed = true;
      if(size_t{1} == pParams->m_cScores) {
         return OperatorBinSumsBoosting<TFloat, bHessian, bWeight, bParallel, bCollapsed, k_oneScore>(pParams);
      } else {
         // Odd: gradient multiclass. Allow it, but do not optimize for it
         return OperatorBinSumsBoosting<TFloat, bHessian, bWeight, bParallel, bCollapsed, k_dynamicScores>(
               pParams);
      }
   } else {
      static constexpr bool bCollapsed = false;
      if(size_t{1} == pParams->m_cScores) {
         return OperatorBinSumsBoosting<TFloat, bHessian, bWeight, bParallel, bCollapsed, k_oneScore>(pParams);
      } else {
         // Odd: gradient multiclass. Allow it, but do not optimize for it
         return OperatorBinSumsBoosting<TFloat, bHessian, bWeight, bParallel, bCollapsed, k_dynamicScores>(
               pParams);
      }
   }
}

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      typename std::enable_if<1 == TFloat::k_cSIMDPack || 0 == PARALLEL_BINS_BYTES_MAX, int>::type = 0>
INLINE_RELEASE_TEMPLATED static ErrorEbm CheckParallel(BinSumsBoostingBridge* const pParams) {
   EBM_ASSERT(EBM_FALSE == pParams->m_bParallelBins);
   static constexpr bool bParallel = false;
   return DoneParallel<TFloat, bHessian, bWeight, bParallel>(pParams);
}

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      typename std::enable_if<1 != TFloat::k_cSIMDPack && 0 < PARALLEL_BINS_BYTES_MAX, int>::type = 0>
INLINE_RELEASE_TEMPLATED static ErrorEbm CheckParallel(BinSumsBoostingBridge* const pParams) {
   if(pParams->m_bParallelBins) {
      static constexpr bool bParallel = true;
      return DoneParallel<TFloat, bHessian, bWeight, bParallel>(pParams);
   } else {
      static constexpr bool bParallel = false;
      return DoneParallel<TFloat, bHessian, bWeight, bParallel>(pParams);
   }
}

template<typename TFloat>
INLINE_RELEASE_TEMPLATED static ErrorEbm BinSumsBoosting(BinSumsBoostingBridge* const pParams) {
   LOG_0(Trace_Verbose, "Entered BinSumsBoosting");

   // all our memory should be aligned. It is required by SIMD for correctness or performance
   EBM_ASSERT(IsAligned(pParams->m_aGradientsAndHessians));
   EBM_ASSERT(IsAligned(pParams->m_aWeights));
   EBM_ASSERT(IsAligned(pParams->m_aPacked));
   EBM_ASSERT(IsAligned(pParams->m_aFastBins));

   ErrorEbm error;

   EBM_ASSERT(1 <= pParams->m_cScores);

   if(EBM_FALSE != pParams->m_bHessian) {
      static constexpr bool bHessian = true;
      if(nullptr == pParams->m_aWeights) {
         static constexpr bool bWeight = false;
         error = CheckParallel<TFloat, bHessian, bWeight>(pParams);
      } else {
         static constexpr bool bWeight = true;
         error = CheckParallel<TFloat, bHessian, bWeight>(pParams);
      }
   } else {
      static constexpr bool bHessian = false;
      if(nullptr == pParams->m_aWeights) {
         static constexpr bool bWeight = false;
         error = CheckParallel<TFloat, bHessian, bWeight>(pParams);
      } else {
         static constexpr bool bWeight = true;
         error = CheckParallel<TFloat, bHessian, bWeight>(pParams);
      }
   }

   LOG_0(Trace_Verbose, "Exited BinSumsBoosting");

   return error;
}

} // namespace DEFINED_ZONE_NAME

#endif // BIN_SUMS_BOOSTING_HPP