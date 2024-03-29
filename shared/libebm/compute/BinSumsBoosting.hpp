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
      bool bCollapsed,
      bool bHessian,
      bool bWeight,
      size_t cCompilerScores,
      int cCompilerPack,
      typename std::enable_if<bCollapsed, int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

   // TODO: we can improve the zero dimensional scenario quite a bit because we know that all the scores added will
   // eventually be added into the same bin.  Instead of adding the gradients & hessians & weights & counts from
   // each sample to the bin in order, we can just add those values together for all samples in SIMD variables
   // and then add the totals into the bins. We probably want to write a completely separate function for handling
   // it this way though. Also, we want to separate the cCompilerScores==1 implementation since all gradients
   // and hessians are going into the same bin we can store that in a register and just write it out at the end
   // while for multiclass we need to write it to memory in case there are too many.
   //
   // For cCompilerScores==1 where we keep everything in registers we can also sum the floats in the SIMD streams
   // separately, and only at the end call call the SIMD Sum() to add the floats accross the SIMD pack. We can't
   // do that for multiclass without keeping 8 or whatever separate histograms (which we could do but probably
   // isn't worth the complexity)
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

      // TODO: we probably want a templated version of this function for Bins with only 1 cScore so that
      //       we can pre-fetch the weight, count, gradient and hessian before writing them

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
                  [aBins, iScore](int, const typename TFloat::T grad, const typename TFloat::T hess) {
                     auto* const pBin = aBins;
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
                  [aBins, iScore](int, const typename TFloat::T grad) {
                     auto* const pBin = aBins;
                     auto* const aGradientPair = pBin->GetGradientPairs();
                     auto* const pGradientPair = &aGradientPair[iScore];
                     pGradientPair->m_sumGradients += grad;
                  },
                  gradient);
         }
         ++iScore;
      } while(cScores != iScore);

      pGradientAndHessian += cScores << (bHessian ? (TFloat::k_cSIMDShift + 1) : TFloat::k_cSIMDShift);
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);
}

template<typename TFloat,
      bool bCollapsed,
      bool bHessian,
      bool bWeight,
      size_t cCompilerScores,
      int cCompilerPack,
      typename std::enable_if<!bCollapsed && 1 == cCompilerScores, int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

   static_assert(1 == cCompilerScores,
         "This specialization of BinSumsBoostingInternal cannot handle multiclass.");
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

   const typename TFloat::TInt::T cBytesPerBin = static_cast<typename TFloat::TInt::T>(
         GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(false, false, bHessian, size_t{1}));

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

   int cShift;
   if(!bFixedSizePack) {
      cShift =
            static_cast<int>(((cSamples >> TFloat::k_cSIMDShift) - size_t{1}) % static_cast<size_t>(cItemsPerBitPack)) *
            cBitsPerItemMax;
   }
   const int cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;

   const typename TFloat::TInt maskBits = MakeLowMask<typename TFloat::TInt::T>(cBitsPerItemMax);

   const typename TFloat::TInt::T* pInputData = reinterpret_cast<const typename TFloat::TInt::T*>(pParams->m_aPacked);
#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pInputData);
#endif // GPU_COMPILE

   const typename TFloat::T* pWeight;
   if(bWeight) {
      pWeight = reinterpret_cast<const typename TFloat::T*>(pParams->m_aWeights);
#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
   }

   do {
      // TODO: maybe, it might be useful to preload the iTensorBinCombined, weight, gradient, hessian for the next loop in this loop
      // which we could do by allocating an extra item to the end of each memory region and throwing away the last one.  I think
      // this won't have any effect since these loads are predictable loads and the CPU should already have them in cache
      // but it's worth trying.  This optimization might destroy the loop unwinding we currently have the compiler doing
      // where it removes the cShift loop and flattens those assembly instructions.

      const typename TFloat::TInt iTensorBinCombined = TFloat::TInt::Load(pInputData);
      pInputData += TFloat::TInt::k_cSIMDPack;
      if(bFixedSizePack) {
         // If we have a fixed sized cCompilerPack, then we previously made it so that in this call cSamples
         // will divide perfectly into the available bitpacks.  This allows us to guarantee that the loop
         // below will allways execute an identical number of times.  If the compiler is aware of this,
         // and it knows how many times the loop below will execute, then it can eliminate the loop.
         // To do this though, we need to set cShift to a value at the top of the loop instead of allowing
         // it to be set above to a smaller value which can change after the first loop iteration.
         // By setting it here, the compiler knows the value of cShift on each loop iteration.

         // I've verified that on the Microsoft compiler, clang, and g++ this loop below optimizes away using the
         // shifts below. For the binary classification Objective (InjectedApplyUpdate) I was not able
         // to get the compiler to optimize the equivalent loop away with the Microsoft compiler 
         // (clang and g++ work) and I think that was due to the
         // amount of code within the loop rather than anything that was preventing the compiler to
         // reason about the values of the variables within each loop iteration.  For RMSE, which
         // has less code within the loop I was able to get it to optimize away the loop, but I had to
         // add an additional index variable and decrement it by 1 each loop, so it seems we're right on
         // the edge of complexity where the compiler will choose to do this.

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

         typename TFloat::TInt iTensorBin = (iTensorBinCombined >> cShift) & maskBits;

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
         if(bSmall) {
            iTensorBin = iTensorBin << 2;
         } else if(bMed) {
            iTensorBin = iTensorBin << 3;
         } else if(bLarge) {
            iTensorBin = iTensorBin << 4;
         }

         // TODO: the ultimate version of this algorithm would:
         //   1) Write to k_cSIMDPack histograms simutaneously to avoid collisions of indexes
         //   2) Sum up the final histograms using SIMD operations in parallel.  If we hvae k_cSIMDPack
         //      histograms, then we're prefectly suited to sum them, and integers and float32 values shouldn't
         //      have issues since we stay well away from 2^32 integers, and the float values don't have addition
         //      issues anymore (where you can't add a 1 to more than 16 million floats)
         //   3) Only do the above if there aren't too many bins. If we put each sample into it's own bin
         //      for a feature, then we should prefer using this version that keeps only 1 histogram

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
         cShift -= cBitsPerItemMax;
      } while(0 <= cShift);
      if(!bFixedSizePack) {
         cShift = cShiftReset;
      }
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);
}

template<typename TFloat,
      bool bCollapsed,
      bool bHessian,
      bool bWeight,
      size_t cCompilerScores,
      int cCompilerPack,
      typename std::enable_if<!bCollapsed && 1 != cCompilerScores, int>::type = 0>
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

   int cShift =
         static_cast<int>(((cSamples >> TFloat::k_cSIMDShift) - size_t{1}) % static_cast<size_t>(cItemsPerBitPack)) *
         cBitsPerItemMax;
   const int cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;

   const typename TFloat::TInt maskBits = MakeLowMask<typename TFloat::TInt::T>(cBitsPerItemMax);

   const typename TFloat::TInt::T* pInputData = reinterpret_cast<const typename TFloat::TInt::T*>(pParams->m_aPacked);
#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pInputData);
#endif // GPU_COMPILE

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
         Bin<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian, cArrayScores>*
               apBins[TFloat::k_cSIMDPack];
         typename TFloat::TInt iTensorBin = (iTensorBinCombined >> cShift) & maskBits;

         // normally the compiler is better at optimimizing multiplications into shifs, but it isn't better
         // if TFloat is a SIMD type. For SIMD shifts & adds will almost always be better than multiplication if
         // there are low numbers of shifts, which should be the case for anything with a compile time constant here
         iTensorBin = Multiply < typename TFloat::TInt, typename TFloat::TInt::T,
         k_dynamicScores != cCompilerScores && 1 != TFloat::k_cSIMDPack,
         static_cast<typename TFloat::TInt::T>(GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(
               false, false, bHessian, cCompilerScores)) > (iTensorBin, cBytesPerBin);

         TFloat::TInt::Execute(
               [aBins, &apBins](const int i, const typename TFloat::TInt::T x) {
                  apBins[i] = IndexBin(aBins, static_cast<size_t>(x));
               },
               iTensorBin);

         // TODO: the ultimate version of this algorithm would:
         //   1) Write to k_cSIMDPack histograms simutaneously to avoid collisions of indexes
         //   2) Sum up the final histograms using SIMD operations in parallel.  If we hvae k_cSIMDPack
         //      histograms, then we're prefectly suited to sum them, and integers and float32 values shouldn't
         //      have issues since we stay well away from 2^32 integers, and the float values don't have addition
         //      issues anymore (where you can't add a 1 to more than 16 million floats)
         //   3) Only do the above if there aren't too many bins. If we put each sample into it's own bin
         //      for a feature, then we should prefer using this version that keeps only 1 histogram

         TFloat weight;
         if(bWeight) {
            weight = TFloat::Load(pWeight);
            pWeight += TFloat::k_cSIMDPack;
         }

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

         cShift -= cBitsPerItemMax;
      } while(0 <= cShift);
      cShift = cShiftReset;
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);
}

template<typename TFloat, bool bCollapsed, bool bHessian, bool bWeight, size_t cCompilerScores, int cCompilerPack>
struct BitPack final {
   INLINE_ALWAYS static void Func(BinSumsBoostingBridge* const pParams) {

      static_assert(!bCollapsed, "Cannot be bCollapsed since there would be no bitpacking");

      if(cCompilerPack == pParams->m_cPack) {
         size_t cSamples = pParams->m_cSamples;
         const size_t cRemnants = cSamples % static_cast<size_t>(cCompilerPack * TFloat::k_cSIMDPack);
         if(0 != cRemnants) {
            pParams->m_cSamples = cRemnants;
            BinSumsBoostingInternal<TFloat,
                  bCollapsed,
                  bHessian,
                  bWeight,
                  cCompilerScores,
                  k_cItemsPerBitPackUndefined>(pParams);

            cSamples -= cRemnants;
            if(0 == cSamples) {
               return;
            }
            pParams->m_cSamples = cSamples;
            if(bWeight) {
               pParams->m_aWeights = IndexByte(pParams->m_aWeights, sizeof(typename TFloat::T) * cRemnants);
            }
            pParams->m_aGradientsAndHessians = IndexByte(pParams->m_aGradientsAndHessians,
                  sizeof(typename TFloat::T) * (bHessian ? size_t{2} : size_t{1}) * cCompilerScores * cRemnants);
            pParams->m_aPacked =
                  IndexByte(pParams->m_aPacked, sizeof(typename TFloat::TInt::T) * TFloat::TInt::k_cSIMDPack);
         }
         BinSumsBoostingInternal<TFloat, bCollapsed, bHessian, bWeight, cCompilerScores, cCompilerPack>(pParams);
      } else {
         BitPack<TFloat,
               bCollapsed,
               bHessian,
               bWeight,
               cCompilerScores,
               GetNextBitPack<TFloat>(cCompilerPack, k_cItemsPerBitPackBoostingMin)>::Func(pParams);
      }
   }
};
template<typename TFloat, bool bCollapsed, bool bHessian, bool bWeight, size_t cCompilerScores>
struct BitPack<TFloat, bCollapsed, bHessian, bWeight, cCompilerScores, k_cItemsPerBitPackUndefined> final {
   INLINE_ALWAYS static void Func(BinSumsBoostingBridge* const pParams) {

      static_assert(!bCollapsed, "Cannot be bCollapsed since there would be no bitpacking");

      BinSumsBoostingInternal<TFloat, bCollapsed, bHessian, bWeight, cCompilerScores, k_cItemsPerBitPackUndefined>(
            pParams);
   }
};

template<typename TFloat,
      bool bCollapsed,
      bool bHessian,
      bool bWeight,
      size_t cCompilerScores,
      typename std::enable_if<!bCollapsed && 1 == cCompilerScores, int>::type = 0>
INLINE_RELEASE_TEMPLATED static void BitPackBoosting(BinSumsBoostingBridge* const pParams) {
   BitPack<TFloat,
         bCollapsed,
         bHessian,
         bWeight,
         cCompilerScores,
         GetFirstBitPack<TFloat>(k_cItemsPerBitPackBoostingMax, k_cItemsPerBitPackBoostingMin)>::Func(pParams);
}
template<typename TFloat,
      bool bCollapsed,
      bool bHessian,
      bool bWeight,
      size_t cCompilerScores,
      typename std::enable_if<bCollapsed || 1 != cCompilerScores, int>::type = 0>
INLINE_RELEASE_TEMPLATED static void BitPackBoosting(BinSumsBoostingBridge* const pParams) {
   BinSumsBoostingInternal<TFloat, bCollapsed, bHessian, bWeight, cCompilerScores, k_cItemsPerBitPackUndefined>(
         pParams);
}

template<typename TFloat, bool bCollapsed, bool bHessian, bool bWeight, size_t cCompilerScores>
GPU_GLOBAL static void RemoteBinSumsBoosting(BinSumsBoostingBridge* const pParams) {
   BitPackBoosting<TFloat, bCollapsed, bHessian, bWeight, cCompilerScores>(pParams);
}

template<typename TFloat, bool bCollapsed, bool bHessian, bool bWeight, size_t cCompilerScores>
INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorBinSumsBoosting(BinSumsBoostingBridge* const pParams) {
   return TFloat::template OperatorBinSumsBoosting<bCollapsed, bHessian, bWeight, cCompilerScores>(pParams);
}

template<typename TFloat, bool bCollapsed, bool bHessian, bool bWeight, size_t cPossibleScores>
struct CountClassesBoosting final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsBoostingBridge* const pParams) {
      if(cPossibleScores == pParams->m_cScores) {
         return OperatorBinSumsBoosting<TFloat, bCollapsed, bHessian, bWeight, cPossibleScores>(pParams);
      } else {
         return CountClassesBoosting<TFloat, bCollapsed, bHessian, bWeight, cPossibleScores + 1>::Func(pParams);
      }
   }
};
template<typename TFloat, bool bCollapsed, bool bHessian, bool bWeight>
struct CountClassesBoosting<TFloat, bCollapsed, bHessian, bWeight, k_cCompilerScoresMax + 1> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsBoostingBridge* const pParams) {
      return OperatorBinSumsBoosting<TFloat, bCollapsed, bHessian, bWeight, k_dynamicScores>(pParams);
   }
};

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

   if(k_cItemsPerBitPackUndefined == pParams->m_cPack) {
      static constexpr bool bCollapsed = true;
      if(EBM_FALSE != pParams->m_bHessian) {
         static constexpr bool bHessian = true;
         if(nullptr != pParams->m_aWeights) {
            static constexpr bool bWeight = true;
            if(size_t{1} == pParams->m_cScores) {
               error = OperatorBinSumsBoosting<TFloat, bCollapsed, bHessian, bWeight, k_oneScore>(pParams);
            } else {
               // muticlass, but for a collapsed so don't optimize for it
               error = OperatorBinSumsBoosting<TFloat, bCollapsed, bHessian, bWeight, k_dynamicScores>(pParams);
            }
         } else {
            static constexpr bool bWeight = false;
            if(size_t{1} == pParams->m_cScores) {
               error = OperatorBinSumsBoosting<TFloat, bCollapsed, bHessian, bWeight, k_oneScore>(pParams);
            } else {
               // muticlass, but for a collapsed so don't optimize for it
               error = OperatorBinSumsBoosting<TFloat, bCollapsed, bHessian, bWeight, k_dynamicScores>(pParams);
            }
         }
      } else {
         static constexpr bool bHessian = false;
         if(nullptr != pParams->m_aWeights) {
            static constexpr bool bWeight = true;
            if(size_t{1} == pParams->m_cScores) {
               error = OperatorBinSumsBoosting<TFloat, bCollapsed, bHessian, bWeight, k_oneScore>(pParams);
            } else {
               // Odd: gradient multiclass. Allow it, but do not optimize for it
               error = OperatorBinSumsBoosting<TFloat, bCollapsed, bHessian, bWeight, k_dynamicScores>(pParams);
            }
         } else {
            static constexpr bool bWeight = false;
            if(size_t{1} == pParams->m_cScores) {
               error = OperatorBinSumsBoosting<TFloat, bCollapsed, bHessian, bWeight, k_oneScore>(pParams);
            } else {
               // Odd: gradient multiclass. Allow it, but do not optimize for it
               error = OperatorBinSumsBoosting<TFloat, bCollapsed, bHessian, bWeight, k_dynamicScores>(pParams);
            }
         }
      }
   } else {
      static constexpr bool bCollapsed = false;
      if(EBM_FALSE != pParams->m_bHessian) {
         static constexpr bool bHessian = true;
         if(nullptr != pParams->m_aWeights) {
            static constexpr bool bWeight = true;
            if(size_t{1} == pParams->m_cScores) {
               error = OperatorBinSumsBoosting<TFloat, bCollapsed, bHessian, bWeight, k_oneScore>(pParams);
            } else {
               // muticlass
               error =
                     CountClassesBoosting<TFloat, bCollapsed, bHessian, bWeight, k_cCompilerScoresStart>::Func(pParams);
            }
         } else {
            static constexpr bool bWeight = false;
            if(size_t{1} == pParams->m_cScores) {
               error = OperatorBinSumsBoosting<TFloat, bCollapsed, bHessian, bWeight, k_oneScore>(pParams);
            } else {
               // muticlass
               error =
                     CountClassesBoosting<TFloat, bCollapsed, bHessian, bWeight, k_cCompilerScoresStart>::Func(pParams);
            }
         }
      } else {
         static constexpr bool bHessian = false;
         if(nullptr != pParams->m_aWeights) {
            static constexpr bool bWeight = true;
            if(size_t{1} == pParams->m_cScores) {
               error = OperatorBinSumsBoosting<TFloat, bCollapsed, bHessian, bWeight, k_oneScore>(pParams);
            } else {
               // Odd: gradient multiclass. Allow it, but do not optimize for it
               error = OperatorBinSumsBoosting<TFloat, bCollapsed, bHessian, bWeight, k_dynamicScores>(pParams);
            }
         } else {
            static constexpr bool bWeight = false;
            if(size_t{1} == pParams->m_cScores) {
               error = OperatorBinSumsBoosting<TFloat, bCollapsed, bHessian, bWeight, k_oneScore>(pParams);
            } else {
               // Odd: gradient multiclass. Allow it, but do not optimize for it
               error = OperatorBinSumsBoosting<TFloat, bCollapsed, bHessian, bWeight, k_dynamicScores>(pParams);
            }
         }
      }
   }

   LOG_0(Trace_Verbose, "Exited BinSumsBoosting");

   return error;
}

} // namespace DEFINED_ZONE_NAME

#endif // BIN_SUMS_BOOSTING_HPP