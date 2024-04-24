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
      bool bCollapsed,
      size_t cCompilerScores,
      bool bParallel,
      int cCompilerPack,
      typename std::enable_if<bCollapsed && 1 == cCompilerScores, int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

   static_assert(!bParallel, "BinSumsBoosting specialization for collapsed does not handle parallel bins.");
   static_assert(k_cItemsPerBitPackUndefined == cCompilerPack, "cCompilerPack must match bCollapsed.");

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
      bool bCollapsed,
      size_t cCompilerScores,
      bool bParallel,
      int cCompilerPack,
      typename std::enable_if<bCollapsed && 1 != cCompilerScores, int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

   static_assert(!bParallel, "BinSumsBoosting specialization for collapsed does not handle parallel bins.");
   static_assert(k_cItemsPerBitPackUndefined == cCompilerPack, "cCompilerPack must match bCollapsed.");

#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pParams);
   EBM_ASSERT(1 <= pParams->m_cSamples);
   EBM_ASSERT(0 == pParams->m_cSamples % size_t{TFloat::k_cSIMDPack});
   EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
   EBM_ASSERT(nullptr != pParams->m_aFastBins);
   EBM_ASSERT(k_dynamicScores == cCompilerScores || cCompilerScores == pParams->m_cScores);
#endif // GPU_COMPILE

   static constexpr size_t cArrayScores = GetArrayScores(cCompilerScores);
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

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bCollapsed,
      size_t cCompilerScores,
      bool bParallel,
      int cCompilerPack,
      typename std::enable_if<bParallel && 1 == cCompilerScores, int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

   static_assert(!bCollapsed, "bCollapsed cannot be true for parallel histograms.");
   static_assert(1 != TFloat::k_cSIMDPack, "If k_cSIMDPack is 1 there is no reason to process in parallel.");
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
   EBM_ASSERT(0 != pParams->m_cBytesFastBins);
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
#ifndef GPU_COMPILE
   EBM_ASSERT(0 == pParams->m_cBytesFastBins % static_cast<size_t>(cBytesPerBin));
#endif // GPU_COMPILE

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

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bCollapsed,
      size_t cCompilerScores,
      bool bParallel,
      int cCompilerPack,
      typename std::enable_if<!bCollapsed && 1 == TFloat::k_cSIMDPack && 1 == cCompilerScores,
            int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

   static_assert(!bParallel, "BinSumsBoosting specialization for SIMD pack of 1 does not handle parallel bins.");
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
      bool bCollapsed,
      size_t cCompilerScores,
      bool bParallel,
      int cCompilerPack,
      typename std::enable_if<!bCollapsed && 1 != TFloat::k_cSIMDPack && !bParallel && 1 == cCompilerScores,
            int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

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

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bCollapsed,
      size_t cCompilerScores,
      bool bParallel,
      int cCompilerPack,
      typename std::enable_if<bParallel && 1 != cCompilerScores, int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

   static_assert(!bCollapsed, "bCollapsed cannot be true for parallel histograms.");
   static_assert(1 != TFloat::k_cSIMDPack, "If k_cSIMDPack is 1 there is no reason to process in parallel.");

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
   EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
   EBM_ASSERT(nullptr != pParams->m_aFastBins);
   EBM_ASSERT(k_dynamicScores == cCompilerScores || cCompilerScores == pParams->m_cScores);
   EBM_ASSERT(0 != pParams->m_cBytesFastBins);
#endif // GPU_COMPILE

   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pParams->m_cScores);

   const size_t cSamples = pParams->m_cSamples;

   typename TFloat::T* const aBins = reinterpret_cast<typename TFloat::T*>(pParams->m_aFastBins);

   const typename TFloat::T* pGradientAndHessian =
         reinterpret_cast<const typename TFloat::T*>(pParams->m_aGradientsAndHessians);
   const typename TFloat::T* const pGradientsAndHessiansEnd =
         pGradientAndHessian + (bHessian ? size_t{2} : size_t{1}) * cScores * cSamples;

   const typename TFloat::TInt::T cBytesPerBin = static_cast<typename TFloat::TInt::T>(
         GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(false, false, bHessian, cScores));

#ifndef GPU_COMPILE
   EBM_ASSERT(0 == pParams->m_cBytesFastBins % static_cast<size_t>(cBytesPerBin));
#endif // GPU_COMPILE

   const typename TFloat::TInt offsets =
         TFloat::TInt::MakeIndexes() * static_cast<typename TFloat::TInt::T>(pParams->m_cBytesFastBins);

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
   cShift =
         static_cast<int>((cSamples >> TFloat::k_cSIMDShift) % static_cast<size_t>(cItemsPerBitPack)) * cBitsPerItemMax;
   iTensorBin = (TFloat::TInt::Load(pInputData) >> cShift) & maskBits;

   // normally the compiler is better at optimimizing multiplications into shifs, but it isn't better
   // if TFloat is a SIMD type. For SIMD shifts & adds will almost always be better than multiplication if
   // there are low numbers of shifts, which should be the case for anything with a compile time constant here
   iTensorBin = Multiply < typename TFloat::TInt, typename TFloat::TInt::T,
   k_dynamicScores != cCompilerScores && 1 != TFloat::k_cSIMDPack,
   static_cast<typename TFloat::TInt::T>(GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(
         false, false, bHessian, cCompilerScores)) > (iTensorBin, cBytesPerBin);
   iTensorBin = iTensorBin + offsets;
   if(bHessian) {
      iTensorBin = PermuteForInterleaf(iTensorBin);
   }

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

         size_t iScore = 0;
         do {
            TFloat gradient = TFloat::Load(
                  &pGradientAndHessian[iScore << (bHessian ? (TFloat::k_cSIMDShift + 1) : TFloat::k_cSIMDShift)]);
            TFloat hessian;
            if(bHessian) {
               hessian =
                     TFloat::Load(&pGradientAndHessian[(iScore << (TFloat::k_cSIMDShift + 1)) + TFloat::k_cSIMDPack]);
            }

            TFloat bin0;
            TFloat bin1;
            if(!bHessian) {
               bin0 = TFloat::template Load<0>(IndexByte(aBins, iScore * k_cBytesGradHess), iTensorBin);
            } else {
               TFloat::template DoubleLoad<0>(IndexByte(aBins, iScore * k_cBytesGradHess), iTensorBin, bin0, bin1);
            }

            if(bWeight) {
               gradient *= weight;
               if(bHessian) {
                  hessian *= weight;
               }
            }

            TFloat gradhess0;
            TFloat gradhess1;

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
               bin0.template Store<0>(IndexByte(aBins, iScore * k_cBytesGradHess), iTensorBin);
            } else {
               TFloat::template DoubleStore<0>(IndexByte(aBins, iScore * k_cBytesGradHess), iTensorBin, bin0, bin1);
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

         iTensorBin = iTensorBin + offsets;
         if(bHessian) {
            iTensorBin = PermuteForInterleaf(iTensorBin);
         }

         cShift -= cBitsPerItemMax;
      } while(0 <= cShift);
      cShift = cShiftReset;
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);
}

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bCollapsed,
      size_t cCompilerScores,
      bool bParallel,
      int cCompilerPack,
      typename std::enable_if<!bCollapsed && !bParallel && 1 != cCompilerScores, int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pParams);
   EBM_ASSERT(1 <= pParams->m_cSamples);
   EBM_ASSERT(0 == pParams->m_cSamples % size_t{TFloat::k_cSIMDPack});
   EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
   EBM_ASSERT(nullptr != pParams->m_aFastBins);
   EBM_ASSERT(k_dynamicScores == cCompilerScores || cCompilerScores == pParams->m_cScores);
#endif // GPU_COMPILE

   static constexpr size_t cArrayScores = GetArrayScores(cCompilerScores);
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
   cShift =
         static_cast<int>((cSamples >> TFloat::k_cSIMDShift) % static_cast<size_t>(cItemsPerBitPack)) * cBitsPerItemMax;
   iTensorBin = (TFloat::TInt::Load(pInputData) >> cShift) & maskBits;

   // normally the compiler is better at optimimizing multiplications into shifs, but it isn't better
   // if TFloat is a SIMD type. For SIMD shifts & adds will almost always be better than multiplication if
   // there are low numbers of shifts, which should be the case for anything with a compile time constant here
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
      bool bCollapsed,
      size_t cCompilerScores,
      bool bParallel,
      int cCompilerPack>
struct BitPack final {
   GPU_DEVICE INLINE_RELEASE_TEMPLATED static void Func(BinSumsBoostingBridge* const pParams) {

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
                  bCollapsed,
                  cCompilerScores,
                  bParallel,
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
         BinSumsBoostingInternal<TFloat, bHessian, bWeight, bCollapsed, cCompilerScores, bParallel, cCompilerPack>(
               pParams);
      } else {
         BitPack<TFloat,
               bHessian,
               bWeight,
               bCollapsed,
               cCompilerScores,
               bParallel,
               GetNextBitPack<typename TFloat::TInt::T>(cCompilerPack, k_cItemsPerBitPackBoostingMin)>::Func(pParams);
      }
   }
};
template<typename TFloat, bool bHessian, bool bWeight, bool bCollapsed, size_t cCompilerScores, bool bParallel>
struct BitPack<TFloat, bHessian, bWeight, bCollapsed, cCompilerScores, bParallel, k_cItemsPerBitPackUndefined>
      final {
   GPU_DEVICE INLINE_RELEASE_TEMPLATED static void Func(BinSumsBoostingBridge* const pParams) {

      static_assert(!bCollapsed, "Cannot be bCollapsed since there would be no bitpacking");

      BinSumsBoostingInternal<TFloat,
            bHessian,
            bWeight,
            bCollapsed,
            cCompilerScores,
            bParallel,
            k_cItemsPerBitPackUndefined>(
            pParams);
   }
};

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bCollapsed,
      size_t cCompilerScores,
      bool bParallel,
      typename std::enable_if<!bCollapsed && 1 == cCompilerScores, int>::type = 0>
GPU_DEVICE INLINE_RELEASE_TEMPLATED static void BitPackBoosting(BinSumsBoostingBridge* const pParams) {
   BitPack<TFloat,
         bHessian,
         bWeight,
         bCollapsed,
         cCompilerScores,
         bParallel,
         GetFirstBitPack<typename TFloat::TInt::T>(
               k_cItemsPerBitPackBoostingMax, k_cItemsPerBitPackBoostingMin)>::Func(pParams);
}
template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bCollapsed,
      size_t cCompilerScores,
      bool bParallel,
      typename std::enable_if<bCollapsed || 1 != cCompilerScores, int>::type = 0>
GPU_DEVICE INLINE_RELEASE_TEMPLATED static void BitPackBoosting(BinSumsBoostingBridge* const pParams) {
   BinSumsBoostingInternal<TFloat,
         bHessian,
         bWeight,
         bCollapsed,
         cCompilerScores,
         bParallel,
         k_cItemsPerBitPackUndefined>(
         pParams);
}

template<typename TFloat, bool bHessian, bool bWeight, bool bCollapsed, size_t cCompilerScores, bool bParallel>
GPU_GLOBAL static void RemoteBinSumsBoosting(BinSumsBoostingBridge* const pParams) {
   BitPackBoosting<TFloat, bHessian, bWeight, bCollapsed, cCompilerScores, bParallel>(pParams);
}

template<typename TFloat, bool bHessian, bool bWeight, bool bCollapsed, size_t cCompilerScores, bool bParallel>
INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorBinSumsBoosting(BinSumsBoostingBridge* const pParams) {
   return TFloat::template OperatorBinSumsBoosting<bHessian, bWeight, bCollapsed, cCompilerScores, bParallel>(
         pParams);
}

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bCollapsed,
      size_t cCompilerScores,
      typename std::enable_if<bCollapsed || 1 == TFloat::k_cSIMDPack || 
                  0 == HESSIAN_PARALLEL_BIN_BYTES_MAX && bHessian && 1 == cCompilerScores ||
                  0 == GRADIENT_PARALLEL_BIN_BYTES_MAX && !bHessian && 1 == cCompilerScores ||
                  0 == MULTISCORE_PARALLEL_BIN_BYTES_MAX && bHessian && 1 != cCompilerScores ||
                  !bHessian && 1 != cCompilerScores,
            int>::type = 0>
INLINE_RELEASE_TEMPLATED static ErrorEbm DoneScores(BinSumsBoostingBridge* const pParams) {
   EBM_ASSERT(EBM_FALSE == pParams->m_bParallelBins);
   static constexpr bool bParallel = false;
   return OperatorBinSumsBoosting<TFloat, bHessian, bWeight, bCollapsed, cCompilerScores, bParallel>(pParams);
}

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bCollapsed,
      size_t cCompilerScores,
      typename std::enable_if<!(bCollapsed || 1 == TFloat::k_cSIMDPack || 
                                    0 == HESSIAN_PARALLEL_BIN_BYTES_MAX && bHessian && 1 == cCompilerScores ||
                                    0 == GRADIENT_PARALLEL_BIN_BYTES_MAX && !bHessian && 1 == cCompilerScores ||
                                    0 == MULTISCORE_PARALLEL_BIN_BYTES_MAX && bHessian && 1 != cCompilerScores ||
                                    !bHessian && 1 != cCompilerScores),
            int>::type = 0>
INLINE_RELEASE_TEMPLATED static ErrorEbm DoneScores(BinSumsBoostingBridge* const pParams) {
   if(pParams->m_bParallelBins) {
      EBM_ASSERT(k_cItemsPerBitPackUndefined != pParams->m_cPack); // excluded in caller

      static constexpr bool bParallel = true;
      return OperatorBinSumsBoosting<TFloat, bHessian, bWeight, bCollapsed, cCompilerScores, bParallel>(pParams);
   } else {
      static constexpr bool bParallel = false;
      return OperatorBinSumsBoosting<TFloat, bHessian, bWeight, bCollapsed, cCompilerScores, bParallel>(pParams);
   }
}

template<typename TFloat, bool bHessian, bool bWeight, bool bCollapsed, size_t cPossibleScores>
struct CountClassesBoosting final {
   INLINE_RELEASE_TEMPLATED static ErrorEbm Func(BinSumsBoostingBridge* const pParams) {
      if(cPossibleScores == pParams->m_cScores) {
         return DoneScores<TFloat, bHessian, bWeight, bCollapsed, cPossibleScores>(pParams);
      } else {
         return CountClassesBoosting<TFloat, bHessian, bWeight, bCollapsed, cPossibleScores + 1>::Func(pParams);
      }
   }
};
template<typename TFloat, bool bHessian, bool bWeight, bool bCollapsed>
struct CountClassesBoosting<TFloat, bHessian, bWeight, bCollapsed, k_cCompilerScoresMax + 1> final {
   INLINE_RELEASE_TEMPLATED static ErrorEbm Func(BinSumsBoostingBridge* const pParams) {
      return DoneScores<TFloat, bHessian, bWeight, bCollapsed, k_dynamicScores>(pParams);
   }
};

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bCollapsed,
      typename std::enable_if<bCollapsed || !bHessian, int>::type = 0>
INLINE_RELEASE_TEMPLATED static ErrorEbm CheckScores(BinSumsBoostingBridge* const pParams) {
   if(size_t{1} == pParams->m_cScores) {
      return DoneScores<TFloat, bHessian, bWeight, bCollapsed, k_oneScore>(pParams);
   } else {
      // muticlass, but for a collapsed or non-hessian so don't optimize for it
      return DoneScores<TFloat, bHessian, bWeight, bCollapsed, k_dynamicScores>(pParams);
   }
}

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      bool bCollapsed,
      typename std::enable_if<!bCollapsed && bHessian, int>::type = 0>
INLINE_RELEASE_TEMPLATED static ErrorEbm CheckScores(BinSumsBoostingBridge* const pParams) {
   if(size_t{1} == pParams->m_cScores) {
      return DoneScores<TFloat, bHessian, bWeight, bCollapsed, k_oneScore>(pParams);
   } else {
      // muticlass
      return CountClassesBoosting<TFloat, bHessian, bWeight, bCollapsed, k_cCompilerScoresStart>::Func(pParams);
   }
}

template<typename TFloat>
INLINE_RELEASE_TEMPLATED static ErrorEbm BinSumsBoosting(BinSumsBoostingBridge* const pParams) {
   LOG_0(Trace_Verbose, "Entered BinSumsBoosting");

   // some scatter/gather SIMD instructions are often signed integers and we only use the positive range
   static_assert(0 == HESSIAN_PARALLEL_BIN_BYTES_MAX ||
               !IsConvertError<typename std::make_signed<typename TFloat::TInt::T>::type>(
                     HESSIAN_PARALLEL_BIN_BYTES_MAX - 1),
         "HESSIAN_PARALLEL_BIN_BYTES_MAX is too large");

   static_assert(0 == GRADIENT_PARALLEL_BIN_BYTES_MAX ||
               !IsConvertError<typename std::make_signed<typename TFloat::TInt::T>::type>(
                     GRADIENT_PARALLEL_BIN_BYTES_MAX - 1),
         "GRADIENT_PARALLEL_BIN_BYTES_MAX is too large");

   static_assert(0 == MULTISCORE_PARALLEL_BIN_BYTES_MAX ||
               !IsConvertError<typename std::make_signed<typename TFloat::TInt::T>::type>(
                     MULTISCORE_PARALLEL_BIN_BYTES_MAX - 1),
         "MULTISCORE_PARALLEL_BIN_BYTES_MAX is too large");

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
         if(k_cItemsPerBitPackUndefined == pParams->m_cPack) {
            static constexpr bool bCollapsed = true;
            error = CheckScores<TFloat, bHessian, bWeight, bCollapsed>(pParams);
         } else {
            static constexpr bool bCollapsed = false;
            error = CheckScores<TFloat, bHessian, bWeight, bCollapsed>(pParams);
         }
      } else {
         static constexpr bool bWeight = true;
         if(k_cItemsPerBitPackUndefined == pParams->m_cPack) {
            static constexpr bool bCollapsed = true;
            error = CheckScores<TFloat, bHessian, bWeight, bCollapsed>(pParams);
         } else {
            static constexpr bool bCollapsed = false;
            error = CheckScores<TFloat, bHessian, bWeight, bCollapsed>(pParams);
         }
      }
   } else {
      static constexpr bool bHessian = false;
      if(nullptr == pParams->m_aWeights) {
         static constexpr bool bWeight = false;
         if(k_cItemsPerBitPackUndefined == pParams->m_cPack) {
            static constexpr bool bCollapsed = true;
            error = CheckScores<TFloat, bHessian, bWeight, bCollapsed>(pParams);
         } else {
            static constexpr bool bCollapsed = false;
            error = CheckScores<TFloat, bHessian, bWeight, bCollapsed>(pParams);
         }
      } else {
         static constexpr bool bWeight = true;
         if(k_cItemsPerBitPackUndefined == pParams->m_cPack) {
            static constexpr bool bCollapsed = true;
            error = CheckScores<TFloat, bHessian, bWeight, bCollapsed>(pParams);
         } else {
            static constexpr bool bCollapsed = false;
            error = CheckScores<TFloat, bHessian, bWeight, bCollapsed>(pParams);
         }
      }
   }

   LOG_0(Trace_Verbose, "Exited BinSumsBoosting");

   return error;
}

} // namespace DEFINED_ZONE_NAME

#endif // BIN_SUMS_BOOSTING_HPP