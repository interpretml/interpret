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

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      size_t cCompilerScores,
      int cCompilerPack,
      typename std::enable_if<k_cItemsPerBitPackNone == cCompilerPack, int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

   // TODO: we can improve the zero dimensional scenario quite a bit because we know that all the scores added will
   // eventually be added into the same bin.  Instead of adding the gradients & hessians & weights & counts from
   // each sample to the bin in order, we can just add those values together for all samples in SIMD variables
   // and then add the totals into the bins. We probably want to write a completely separate function for handling
   // it this way though.
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

   auto* const aBins = reinterpret_cast<BinBase*>(pParams->m_aFastBins)
                             ->Specialize<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian, cArrayScores>();

   const size_t cSamples = pParams->m_cSamples;

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
      bool bHessian,
      bool bWeight,
      size_t cCompilerScores,
      int cCompilerPack,
      typename std::enable_if<k_cItemsPerBitPackNone != cCompilerPack && 1 == cCompilerScores, int>::type = 0>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge* const pParams) {

#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pParams);
   EBM_ASSERT(1 <= pParams->m_cSamples);
   EBM_ASSERT(0 == pParams->m_cSamples % size_t{TFloat::k_cSIMDPack});
   EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
   EBM_ASSERT(nullptr != pParams->m_aFastBins);
   EBM_ASSERT(size_t{1} == pParams->m_cScores);
#endif // GPU_COMPILE

   auto* const aBins = reinterpret_cast<BinBase*>(pParams->m_aFastBins)
                             ->Specialize<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian, size_t{1}>();

   const size_t cSamples = pParams->m_cSamples;

   const typename TFloat::T* pGradientAndHessian =
         reinterpret_cast<const typename TFloat::T*>(pParams->m_aGradientsAndHessians);
   const typename TFloat::T* const pGradientsAndHessiansEnd =
         pGradientAndHessian + (bHessian ? size_t{2} : size_t{1}) * cSamples;

   const typename TFloat::TInt::T cBytesPerBin = static_cast<typename TFloat::TInt::T>(
         GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(false, false, bHessian, size_t{1}));

   const int cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pParams->m_cPack);
#ifndef GPU_COMPILE
   EBM_ASSERT(k_cItemsPerBitPackNone != cItemsPerBitPack); // we require this condition to be templated
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

         // normally the compiler is better at optimimizing multiplications into shifs, but it isn't better
         // if TFloat is a SIMD type. For SIMD shifts & adds will almost always be better than multiplication if
         // there are low numbers of shifts, which should be the case for anything with a compile time constant here
         iTensorBin = Multiply<typename TFloat::TInt,
               typename TFloat::TInt::T,
               1 != TFloat::k_cSIMDPack,
               static_cast<typename TFloat::TInt::T>(GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(
                     false, false, bHessian, size_t{1}))>(iTensorBin, cBytesPerBin);

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
      cShift = cShiftReset;
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);
}

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      size_t cCompilerScores,
      int cCompilerPack,
      typename std::enable_if<k_cItemsPerBitPackNone != cCompilerPack && 1 != cCompilerScores, int>::type = 0>
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

   auto* const aBins = reinterpret_cast<BinBase*>(pParams->m_aFastBins)
                             ->Specialize<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian, cArrayScores>();

   const size_t cSamples = pParams->m_cSamples;

   const typename TFloat::T* pGradientAndHessian =
         reinterpret_cast<const typename TFloat::T*>(pParams->m_aGradientsAndHessians);
   const typename TFloat::T* const pGradientsAndHessiansEnd =
         pGradientAndHessian + (bHessian ? size_t{2} : size_t{1}) * cScores * cSamples;

   const typename TFloat::TInt::T cBytesPerBin = static_cast<typename TFloat::TInt::T>(
         GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(false, false, bHessian, cScores));

   const int cItemsPerBitPack = GET_ITEMS_PER_BIT_PACK(cCompilerPack, pParams->m_cPack);
#ifndef GPU_COMPILE
   EBM_ASSERT(k_cItemsPerBitPackNone != cItemsPerBitPack); // we require this condition to be templated
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
         Bin<typename TFloat::T, typename TFloat::TInt::T, false, false, bHessian, cArrayScores>* apBins[TFloat::k_cSIMDPack];
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

template<typename TFloat,
      bool bHessian,
      bool bWeight,
      size_t cCompilerScores,
      typename std::enable_if<1 == cCompilerScores, int>::type = 0>
INLINE_RELEASE_TEMPLATED static void BitPackBoosting(BinSumsBoostingBridge* const pParams) {
   if(k_cItemsPerBitPackNone == pParams->m_cPack) {
      // this needs to be special cased because otherwise we would inject comparisons into the dynamic version
      BinSumsBoostingInternal<TFloat, bHessian, bWeight, 1, k_cItemsPerBitPackNone>(pParams);
   } else {
      BinSumsBoostingInternal<TFloat, bHessian, bWeight, 1, k_cItemsPerBitPackDynamic>(pParams);
   }
}
template<typename TFloat,
      bool bHessian,
      bool bWeight,
      size_t cCompilerScores,
      typename std::enable_if<1 != cCompilerScores, int>::type = 0>
INLINE_RELEASE_TEMPLATED static void BitPackBoosting(BinSumsBoostingBridge* const pParams) {
   if(k_cItemsPerBitPackNone == pParams->m_cPack) {
      // this needs to be special cased because otherwise we would inject comparisons into the dynamic version
      BinSumsBoostingInternal<TFloat, bHessian, bWeight, cCompilerScores, k_cItemsPerBitPackNone>(pParams);
   } else {
      BinSumsBoostingInternal<TFloat, bHessian, bWeight, cCompilerScores, k_cItemsPerBitPackDynamic>(pParams);
   }
}

template<typename TFloat, bool bHessian, bool bWeight, size_t cCompilerScores>
GPU_GLOBAL static void RemoteBinSumsBoosting(BinSumsBoostingBridge* const pParams) {
   BitPackBoosting<TFloat, bHessian, bWeight, cCompilerScores>(pParams);
}

template<typename TFloat, bool bHessian, bool bWeight, size_t cCompilerScores>
INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorBinSumsBoosting(BinSumsBoostingBridge* const pParams) {
   return TFloat::template OperatorBinSumsBoosting<bHessian, bWeight, cCompilerScores>(pParams);
}

template<typename TFloat, bool bHessian, bool bWeight, size_t cPossibleScores> struct CountClassesBoosting final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsBoostingBridge* const pParams) {
      if(cPossibleScores == pParams->m_cScores) {
         return OperatorBinSumsBoosting<TFloat, bHessian, bWeight, cPossibleScores>(pParams);
      } else {
         return CountClassesBoosting<TFloat, bHessian, bWeight, cPossibleScores + 1>::Func(pParams);
      }
   }
};
template<typename TFloat, bool bHessian, bool bWeight>
struct CountClassesBoosting<TFloat, bHessian, bWeight, k_cCompilerScoresMax + 1> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsBoostingBridge* const pParams) {
      return OperatorBinSumsBoosting<TFloat, bHessian, bWeight, k_dynamicScores>(pParams);
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
   if(EBM_FALSE != pParams->m_bHessian) {
      static constexpr bool bHessian = true;
      if(nullptr != pParams->m_aWeights) {
         static constexpr bool bWeight = true;
         if(size_t{1} == pParams->m_cScores) {
            error = OperatorBinSumsBoosting<TFloat, bHessian, bWeight, k_oneScore>(pParams);
         } else {
            // muticlass
            error = CountClassesBoosting<TFloat, bHessian, bWeight, k_cCompilerScoresStart>::Func(pParams);
         }
      } else {
         static constexpr bool bWeight = false;
         if(size_t{1} == pParams->m_cScores) {
            error = OperatorBinSumsBoosting<TFloat, bHessian, bWeight, k_oneScore>(pParams);
         } else {
            // muticlass
            error = CountClassesBoosting<TFloat, bHessian, bWeight, k_cCompilerScoresStart>::Func(pParams);
         }
      }
   } else {
      static constexpr bool bHessian = false;
      if(nullptr != pParams->m_aWeights) {
         static constexpr bool bWeight = true;
         if(size_t{1} == pParams->m_cScores) {
            error = OperatorBinSumsBoosting<TFloat, bHessian, bWeight, k_oneScore>(pParams);
         } else {
            // Odd: gradient multiclass. Allow it, but do not optimize for it
            error = OperatorBinSumsBoosting<TFloat, bHessian, bWeight, k_dynamicScores>(pParams);
         }
      } else {
         static constexpr bool bWeight = false;
         if(size_t{1} == pParams->m_cScores) {
            error = OperatorBinSumsBoosting<TFloat, bHessian, bWeight, k_oneScore>(pParams);
         } else {
            // Odd: gradient multiclass. Allow it, but do not optimize for it
            error = OperatorBinSumsBoosting<TFloat, bHessian, bWeight, k_dynamicScores>(pParams);
         }
      }
   }

   LOG_0(Trace_Verbose, "Exited BinSumsBoosting");

   return error;
}

} // namespace DEFINED_ZONE_NAME

#endif // BIN_SUMS_BOOSTING_HPP