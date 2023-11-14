// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef BIN_SUMS_BOOSTING_HPP
#define BIN_SUMS_BOOSTING_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT
#include "zones.h"

#include "common.hpp" // Multiply
#include "bridge.hpp" // BinSumsBoostingBridge
#include "GradientPair.hpp"
#include "Bin.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<
   typename TFloat, 
   bool bHessian, 
   size_t cCompilerScores, 
   bool bWeight, 
   bool bReplication, 
   int cCompilerPack, 
   typename std::enable_if<k_cItemsPerBitPackNone == cCompilerPack, int>::type = 0
>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge * const pParams) {
   static_assert(bWeight || !bReplication, "bReplication cannot be true if bWeight is false");

   // TODO: we can improve the zero dimensional scenario quite a bit because we know that all the scores added will
   // eventually be added into the same bin.  Instead of adding the gradients & hessians & weights & counts from
   // each sample to the bin in order, we can just add those values together for all samples in SIMD variables
   // and then add the totals into the bins. We probably want to write a completely separate function for handling
   // it this way though.
   static constexpr size_t cArrayScores = GetArrayScores(cCompilerScores);

#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pParams);
   EBM_ASSERT(1 <= pParams->m_cSamples);
   EBM_ASSERT(0 == pParams->m_cSamples % size_t { TFloat::k_cSIMDPack });
   EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
   EBM_ASSERT(nullptr != pParams->m_aFastBins);
   EBM_ASSERT(k_dynamicScores == cCompilerScores || cCompilerScores == pParams->m_cScores);
#endif // GPU_COMPILE

   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pParams->m_cScores);

   auto * const aBins = reinterpret_cast<BinBase *>(pParams->m_aFastBins)->Specialize<typename TFloat::T, typename TFloat::TInt::T, bHessian, cArrayScores>();

   const size_t cSamples = pParams->m_cSamples;

   const typename TFloat::T * pGradientAndHessian = reinterpret_cast<const typename TFloat::T *>(pParams->m_aGradientsAndHessians);
   const typename TFloat::T * const pGradientsAndHessiansEnd = pGradientAndHessian + (bHessian ? size_t { 2 } : size_t { 1 }) * cScores * cSamples;

   const typename TFloat::T * pWeight;
   const uint8_t * pCountOccurrences;
   if(bWeight) {
      pWeight = reinterpret_cast<const typename TFloat::T *>(pParams->m_aWeights);
#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
      if(bReplication) {
         pCountOccurrences = pParams->m_pCountOccurrences;
#ifndef GPU_COMPILE
         EBM_ASSERT(nullptr != pCountOccurrences);
#endif // GPU_COMPILE
      }
   }

   do {
      if(bReplication) {
         const typename TFloat::TInt cOccurences = TFloat::TInt::LoadBytes(pCountOccurrences);
         pCountOccurrences += TFloat::k_cSIMDPack;

         TFloat::TInt::Execute([aBins](int, const typename TFloat::TInt::T x) {
            auto * const pBin = aBins;
            // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
            //       such that we can remove that field optionally
            pBin->SetCountSamples(pBin->GetCountSamples() + x);
         }, cOccurences);
      } else {
         TFloat::Execute([aBins](int) {
            auto * const pBin = aBins;
            // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
            //       such that we can remove that field optionally
            pBin->SetCountSamples(pBin->GetCountSamples() + typename TFloat::TInt::T { 1 });
         });
      }

      TFloat weight;
      if(bWeight) {
         weight = TFloat::Load(pWeight);
         pWeight += TFloat::k_cSIMDPack;

         TFloat::Execute([aBins](int, const typename TFloat::T x) {
            auto * const pBin = aBins;
            // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
            //       such that we can remove that field optionally
            pBin->SetWeight(pBin->GetWeight() + x);
         }, weight);
      } else {
         TFloat::Execute([aBins](int) {
            auto * const pBin = aBins;
            // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
            //       such that we can remove that field optionally
            pBin->SetWeight(pBin->GetWeight() + typename TFloat::T { 1.0 });
         });
      }

      // TODO: we probably want a templated version of this function for Bins with only 1 cScore so that
      //       we can pre-fetch the weight, count, gradient and hessian before writing them 

      size_t iScore = 0;
      do {
         if(bHessian) {
            TFloat gradient = TFloat::Load(&pGradientAndHessian[iScore << (TFloat::k_cSIMDShift + 1)]);
            TFloat hessian = TFloat::Load(&pGradientAndHessian[(iScore << (TFloat::k_cSIMDShift + 1)) + TFloat::k_cSIMDPack]);
            if(bWeight) {
               gradient *= weight;
               hessian *= weight;
            }
            TFloat::Execute([aBins, iScore](int, const typename TFloat::T grad, const typename TFloat::T hess) {
               auto * const pBin = aBins;
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
            TFloat gradient = TFloat::Load(&pGradientAndHessian[iScore << TFloat::k_cSIMDShift]);
            if(bWeight) {
               gradient *= weight;
            }
            TFloat::Execute([aBins, iScore](int, const typename TFloat::T grad) {
               auto * const pBin = aBins;
               auto * const aGradientPair = pBin->GetGradientPairs();
               auto * const pGradientPair = &aGradientPair[iScore];
               pGradientPair->m_sumGradients += grad;
            }, gradient);
         }
         ++iScore;
      } while(cScores != iScore);

      pGradientAndHessian += cScores << (bHessian ? (TFloat::k_cSIMDShift + 1) : TFloat::k_cSIMDShift);
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);
}

template<
   typename TFloat, 
   bool bHessian, 
   size_t cCompilerScores, 
   bool bWeight, 
   bool bReplication, 
   int cCompilerPack, 
   typename std::enable_if<k_cItemsPerBitPackNone != cCompilerPack && 1 == cCompilerScores, int>::type = 0
>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge * const pParams) {
   static_assert(bWeight || !bReplication, "bReplication cannot be true if bWeight is false");

#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pParams);
   EBM_ASSERT(1 <= pParams->m_cSamples);
   EBM_ASSERT(0 == pParams->m_cSamples % size_t { TFloat::k_cSIMDPack });
   EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
   EBM_ASSERT(nullptr != pParams->m_aFastBins);
   EBM_ASSERT(size_t { 1 } == pParams->m_cScores);
#endif // GPU_COMPILE

   auto * const aBins = reinterpret_cast<BinBase *>(pParams->m_aFastBins)->Specialize<typename TFloat::T, typename TFloat::TInt::T, bHessian, size_t { 1 }>();

   const size_t cSamples = pParams->m_cSamples;

   const typename TFloat::T * pGradientAndHessian = reinterpret_cast<const typename TFloat::T *>(pParams->m_aGradientsAndHessians);
   const typename TFloat::T * const pGradientsAndHessiansEnd = pGradientAndHessian + (bHessian ? size_t { 2 } : size_t { 1 }) * cSamples;

   const typename TFloat::TInt::T cBytesPerBin = static_cast<typename TFloat::TInt::T>(GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(bHessian, size_t { 1 }));

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

   int cShift = static_cast<int>(((cSamples >> TFloat::k_cSIMDShift) - size_t { 1 }) % static_cast<size_t>(cItemsPerBitPack)) * cBitsPerItemMax;
   const int cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;

   const typename TFloat::TInt maskBits = MakeLowMask<typename TFloat::TInt::T>(cBitsPerItemMax);

   const typename TFloat::TInt::T * pInputData = reinterpret_cast<const typename TFloat::TInt::T *>(pParams->m_aPacked);
#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pInputData);
#endif // GPU_COMPILE

   const typename TFloat::T * pWeight;
   const uint8_t * pCountOccurrences;
   if(bWeight) {
      pWeight = reinterpret_cast<const typename TFloat::T *>(pParams->m_aWeights);
#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
      if(bReplication) {
         pCountOccurrences = pParams->m_pCountOccurrences;
#ifndef GPU_COMPILE
         EBM_ASSERT(nullptr != pCountOccurrences);
#endif // GPU_COMPILE
      }
   }

   do {
      const typename TFloat::TInt iTensorBinCombined = TFloat::TInt::Load(pInputData);
      pInputData += TFloat::TInt::k_cSIMDPack;
      do {
         TFloat weight;
         typename TFloat::TInt cOccurences;
         if(bWeight) {
            weight = TFloat::Load(pWeight);
            pWeight += TFloat::k_cSIMDPack;
            if(bReplication) {
               cOccurences = TFloat::TInt::LoadBytes(pCountOccurrences);
               pCountOccurrences += TFloat::k_cSIMDPack;
            }
         }

         TFloat gradient = TFloat::Load(pGradientAndHessian);
         TFloat hessian;
         if(bHessian) {
            hessian = TFloat::Load(&pGradientAndHessian[TFloat::k_cSIMDPack]);
         }
         pGradientAndHessian += (bHessian ? size_t { 2 } : size_t { 1 }) * TFloat::k_cSIMDPack;

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
         iTensorBin = Multiply<typename TFloat::TInt, typename TFloat::TInt::T, 
            1 != TFloat::k_cSIMDPack, 
            static_cast<typename TFloat::TInt::T>(GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(bHessian, size_t { 1 }))>(
               iTensorBin, cBytesPerBin);

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
         if(bWeight) {
            if(bReplication) {
               if(bHessian) {
                  TFloat::Execute([aBins](
                     int, 
                     const typename TFloat::TInt::T i, 
                     const typename TFloat::TInt::T c, 
                     const typename TFloat::T w, 
                     const typename TFloat::T grad, 
                     const typename TFloat::T hess
                  ) {
                     COVER(COV_BinSumsBoostingInternal_Weight_Replication_Hessian);
                     auto * const pBin = IndexBin(aBins, static_cast<size_t>(i));
                     auto * const pGradientPair = pBin->GetGradientPairs();
                     typename TFloat::TInt::T cBinSamples = pBin->GetCountSamples();
                     typename TFloat::T binWeight = pBin->GetWeight();
                     typename TFloat::T binGrad = pGradientPair->m_sumGradients;
                     typename TFloat::T binHess = pGradientPair->GetHess();
                     cBinSamples += c;
                     binWeight += w;
                     binGrad += grad;
                     binHess += hess;
                     pBin->SetCountSamples(cBinSamples);
                     pBin->SetWeight(binWeight);
                     pGradientPair->m_sumGradients = binGrad;
                     pGradientPair->SetHess(binHess);
                  }, iTensorBin, cOccurences, weight, gradient, hessian);
               } else {
                  TFloat::Execute([aBins](
                     int, 
                     const typename TFloat::TInt::T i, 
                     const typename TFloat::TInt::T c, 
                     const typename TFloat::T w, 
                     const typename TFloat::T grad
                  ) {
                     COVER(COV_BinSumsBoostingInternal_Weight_Replication_NoHessian);
                     auto * const pBin = IndexBin(aBins, static_cast<size_t>(i));
                     auto * const pGradientPair = pBin->GetGradientPairs();
                     typename TFloat::TInt::T cBinSamples = pBin->GetCountSamples();
                     typename TFloat::T binWeight = pBin->GetWeight();
                     typename TFloat::T binGrad = pGradientPair->m_sumGradients;
                     cBinSamples += c;
                     binWeight += w;
                     binGrad += grad;
                     pBin->SetCountSamples(cBinSamples);
                     pBin->SetWeight(binWeight);
                     pGradientPair->m_sumGradients = binGrad;
                  }, iTensorBin, cOccurences, weight, gradient);
               }
            } else {
               if(bHessian) {
                  TFloat::Execute([aBins](
                     int, 
                     const typename TFloat::TInt::T i, 
                     const typename TFloat::T w, 
                     const typename TFloat::T grad, 
                     const typename TFloat::T hess
                  ) {
                     COVER(COV_BinSumsBoostingInternal_Weight_NoReplication_Hessian);
                     auto * const pBin = IndexBin(aBins, static_cast<size_t>(i));
                     auto * const pGradientPair = pBin->GetGradientPairs();
                     typename TFloat::TInt::T cBinSamples = pBin->GetCountSamples(); // TODO: eliminate this by eliminating the field in the future
                     typename TFloat::T binWeight = pBin->GetWeight();
                     typename TFloat::T binGrad = pGradientPair->m_sumGradients;
                     typename TFloat::T binHess = pGradientPair->GetHess();
                     cBinSamples += typename TFloat::TInt::T { 1 }; // TODO: eliminate this by eliminating the field in the future
                     binWeight += w;
                     binGrad += grad;
                     binHess += hess;
                     pBin->SetCountSamples(cBinSamples); // TODO: eliminate this by eliminating the field in the future
                     pBin->SetWeight(binWeight);
                     pGradientPair->m_sumGradients = binGrad;
                     pGradientPair->SetHess(binHess);
                  }, iTensorBin, weight, gradient, hessian);
               } else {
                  TFloat::Execute([aBins](
                     int, 
                     const typename TFloat::TInt::T i, 
                     const typename TFloat::T w, 
                     const typename TFloat::T grad
                  ) {
                     COVER(COV_BinSumsBoostingInternal_Weight_NoReplication_NoHessian);
                     auto * const pBin = IndexBin(aBins, static_cast<size_t>(i));
                     auto * const pGradientPair = pBin->GetGradientPairs();
                     typename TFloat::TInt::T cBinSamples = pBin->GetCountSamples(); // TODO: eliminate this by eliminating the field in the future
                     typename TFloat::T binWeight = pBin->GetWeight();
                     typename TFloat::T binGrad = pGradientPair->m_sumGradients;
                     cBinSamples += typename TFloat::TInt::T { 1 }; // TODO: eliminate this by eliminating the field in the future
                     binWeight += w;
                     binGrad += grad;
                     pBin->SetCountSamples(cBinSamples); // TODO: eliminate this by eliminating the field in the future
                     pBin->SetWeight(binWeight);
                     pGradientPair->m_sumGradients = binGrad;
                  }, iTensorBin, weight, gradient);
               }
            }
         } else {
            if(bHessian) {
               TFloat::Execute([aBins](
                  int, 
                  const typename TFloat::TInt::T i, 
                  const typename TFloat::T grad, 
                  const typename TFloat::T hess
               ) {
                  COVER(COV_BinSumsBoostingInternal_NoWeight_NoReplication_Hessian);
                  auto * const pBin = IndexBin(aBins, static_cast<size_t>(i));
                  auto * const pGradientPair = pBin->GetGradientPairs();
                  typename TFloat::TInt::T cBinSamples = pBin->GetCountSamples(); // TODO: eliminate this by eliminating the field in the future
                  typename TFloat::T binWeight = pBin->GetWeight(); // TODO: eliminate this by eliminating the field in the future
                  typename TFloat::T binGrad = pGradientPair->m_sumGradients;
                  typename TFloat::T binHess = pGradientPair->GetHess();
                  cBinSamples += typename TFloat::TInt::T { 1 }; // TODO: eliminate this by eliminating the field in the future
                  binWeight += typename TFloat::T { 1.0 }; // TODO: eliminate this by eliminating the field in the future
                  binGrad += grad;
                  binHess += hess;
                  pBin->SetCountSamples(cBinSamples); // TODO: eliminate this by eliminating the field in the future
                  pBin->SetWeight(binWeight); // TODO: eliminate this by eliminating the field in the future
                  pGradientPair->m_sumGradients = binGrad;
                  pGradientPair->SetHess(binHess);
               }, iTensorBin, gradient, hessian);
            } else {
               TFloat::Execute([aBins](
                  int, 
                  const typename TFloat::TInt::T i, 
                  const typename TFloat::T grad
               ) {
                  COVER(COV_BinSumsBoostingInternal_NoWeight_NoReplication_NoHessian);
                  auto * const pBin = IndexBin(aBins, static_cast<size_t>(i));
                  auto * const pGradientPair = pBin->GetGradientPairs();
                  typename TFloat::TInt::T cBinSamples = pBin->GetCountSamples(); // TODO: eliminate this by eliminating the field in the future
                  typename TFloat::T binWeight = pBin->GetWeight(); // TODO: eliminate this by eliminating the field in the future
                  typename TFloat::T binGrad = pGradientPair->m_sumGradients;
                  cBinSamples += typename TFloat::TInt::T { 1 }; // TODO: eliminate this by eliminating the field in the future
                  binWeight += typename TFloat::T { 1.0 }; // TODO: eliminate this by eliminating the field in the future
                  binGrad += grad;
                  pBin->SetCountSamples(cBinSamples); // TODO: eliminate this by eliminating the field in the future
                  pBin->SetWeight(binWeight); // TODO: eliminate this by eliminating the field in the future
                  pGradientPair->m_sumGradients = binGrad;
               }, iTensorBin, gradient);
            }
         }

         cShift -= cBitsPerItemMax;
      } while(0 <= cShift);
      cShift = cShiftReset;
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);
}

template<
   typename TFloat, 
   bool bHessian, 
   size_t cCompilerScores, 
   bool bWeight, 
   bool bReplication, 
   int cCompilerPack, 
   typename std::enable_if<k_cItemsPerBitPackNone != cCompilerPack && 1 != cCompilerScores, int>::type = 0
>
GPU_DEVICE NEVER_INLINE static void BinSumsBoostingInternal(BinSumsBoostingBridge * const pParams) {
   static_assert(bWeight || !bReplication, "bReplication cannot be true if bWeight is false");

   static constexpr size_t cArrayScores = GetArrayScores(cCompilerScores);

#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pParams);
   EBM_ASSERT(1 <= pParams->m_cSamples);
   EBM_ASSERT(0 == pParams->m_cSamples % size_t { TFloat::k_cSIMDPack });
   EBM_ASSERT(nullptr != pParams->m_aGradientsAndHessians);
   EBM_ASSERT(nullptr != pParams->m_aFastBins);
   EBM_ASSERT(k_dynamicScores == cCompilerScores || cCompilerScores == pParams->m_cScores);
#endif // GPU_COMPILE

   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pParams->m_cScores);

   auto * const aBins = reinterpret_cast<BinBase *>(pParams->m_aFastBins)->Specialize<typename TFloat::T, typename TFloat::TInt::T, bHessian, cArrayScores>();

   const size_t cSamples = pParams->m_cSamples;

   const typename TFloat::T * pGradientAndHessian = reinterpret_cast<const typename TFloat::T *>(pParams->m_aGradientsAndHessians);
   const typename TFloat::T * const pGradientsAndHessiansEnd = pGradientAndHessian + (bHessian ? size_t { 2 } : size_t { 1 }) * cScores * cSamples;

   const typename TFloat::TInt::T cBytesPerBin = static_cast<typename TFloat::TInt::T>(GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(bHessian, cScores));

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

   int cShift = static_cast<int>(((cSamples >> TFloat::k_cSIMDShift) - size_t { 1 }) % static_cast<size_t>(cItemsPerBitPack)) * cBitsPerItemMax;
   const int cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;

   const typename TFloat::TInt maskBits = MakeLowMask<typename TFloat::TInt::T>(cBitsPerItemMax);

   const typename TFloat::TInt::T * pInputData = reinterpret_cast<const typename TFloat::TInt::T *>(pParams->m_aPacked);
#ifndef GPU_COMPILE
   EBM_ASSERT(nullptr != pInputData);
#endif // GPU_COMPILE

   const typename TFloat::T * pWeight;
   const uint8_t * pCountOccurrences;
   if(bWeight) {
      pWeight = reinterpret_cast<const typename TFloat::T *>(pParams->m_aWeights);
#ifndef GPU_COMPILE
      EBM_ASSERT(nullptr != pWeight);
#endif // GPU_COMPILE
      if(bReplication) {
         pCountOccurrences = pParams->m_pCountOccurrences;
#ifndef GPU_COMPILE
         EBM_ASSERT(nullptr != pCountOccurrences);
#endif // GPU_COMPILE
      }
   }

   do {
      const typename TFloat::TInt iTensorBinCombined = TFloat::TInt::Load(pInputData);
      pInputData += TFloat::TInt::k_cSIMDPack;
      do {
         Bin<typename TFloat::T, typename TFloat::TInt::T, bHessian, cArrayScores> * apBins[TFloat::k_cSIMDPack];
         typename TFloat::TInt iTensorBin = (iTensorBinCombined >> cShift) & maskBits;
            
         // normally the compiler is better at optimimizing multiplications into shifs, but it isn't better
         // if TFloat is a SIMD type. For SIMD shifts & adds will almost always be better than multiplication if
         // there are low numbers of shifts, which should be the case for anything with a compile time constant here
         iTensorBin = Multiply<typename TFloat::TInt, typename TFloat::TInt::T, 
            k_dynamicScores != cCompilerScores && 1 != TFloat::k_cSIMDPack, 
            static_cast<typename TFloat::TInt::T>(GetBinSize<typename TFloat::T, typename TFloat::TInt::T>(bHessian, cCompilerScores))>(
               iTensorBin, cBytesPerBin);
            
         TFloat::TInt::Execute([aBins, &apBins](const int i, const typename TFloat::TInt::T x) {
            apBins[i] = IndexBin(aBins, static_cast<size_t>(x));
         }, iTensorBin);
#ifndef NDEBUG
#ifndef GPU_COMPILE
         TFloat::Execute([cBytesPerBin, apBins, pParams](const int i) {
            ASSERT_BIN_OK(cBytesPerBin, apBins[i], pParams->m_pDebugFastBinsEnd);
         });
#endif // GPU_COMPILE
#endif // NDEBUG

         // TODO: the ultimate version of this algorithm would:
         //   1) Write to k_cSIMDPack histograms simutaneously to avoid collisions of indexes
         //   2) Sum up the final histograms using SIMD operations in parallel.  If we hvae k_cSIMDPack
         //      histograms, then we're prefectly suited to sum them, and integers and float32 values shouldn't
         //      have issues since we stay well away from 2^32 integers, and the float values don't have addition
         //      issues anymore (where you can't add a 1 to more than 16 million floats)
         //   3) Only do the above if there aren't too many bins. If we put each sample into it's own bin
         //      for a feature, then we should prefer using this version that keeps only 1 histogram

         if(bReplication) {
            const typename TFloat::TInt cOccurences = TFloat::TInt::LoadBytes(pCountOccurrences);
            pCountOccurrences += TFloat::k_cSIMDPack;

            TFloat::TInt::Execute([apBins](const int i, const typename TFloat::TInt::T x) {
               auto * const pBin = apBins[i];
               // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
               //       such that we can remove that field optionally
               pBin->SetCountSamples(pBin->GetCountSamples() + x);
            }, cOccurences);
         } else {
            TFloat::Execute([apBins](const int i) {
               auto * const pBin = apBins[i];
               // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
               //       such that we can remove that field optionally
               pBin->SetCountSamples(pBin->GetCountSamples() + typename TFloat::TInt::T { 1 });
            });
         }

         // TODO: if bWeight and bReplication then we can load both together before adding and the storing them
         //       we'd need to write something like we have in the 1 == cCompilerScores function above

         TFloat weight;
         if(bWeight) {
            weight = TFloat::Load(pWeight);
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
               TFloat gradient = TFloat::Load(&pGradientAndHessian[iScore << (TFloat::k_cSIMDShift + 1)]);
               TFloat hessian = TFloat::Load(&pGradientAndHessian[(iScore << (TFloat::k_cSIMDShift + 1)) + TFloat::k_cSIMDPack]);
               if(bWeight) {
                  gradient *= weight;
                  hessian *= weight;
               }
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
               TFloat gradient = TFloat::Load(&pGradientAndHessian[iScore << TFloat::k_cSIMDShift]);
               if(bWeight) {
                  gradient *= weight;
               }
               TFloat::Execute([apBins, iScore](const int i, const typename TFloat::T grad) {
                  auto * const pBin = apBins[i];
                  auto * const aGradientPair = pBin->GetGradientPairs();
                  auto * const pGradientPair = &aGradientPair[iScore];
                  pGradientPair->m_sumGradients += grad;
               }, gradient);
            }
            ++iScore;
         } while(cScores != iScore);

         pGradientAndHessian += cScores << (bHessian ? (TFloat::k_cSIMDShift + 1) : TFloat::k_cSIMDShift);

         cShift -= cBitsPerItemMax;
      } while(0 <= cShift);
      cShift = cShiftReset;
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);
}

template<typename TFloat, bool bHessian, size_t cCompilerScores, bool bWeight, bool bReplication, int cCompilerPack>
GPU_GLOBAL static void RemoteBinSumsBoosting(BinSumsBoostingBridge * const pParams) {
   BinSumsBoostingInternal<TFloat, bHessian, cCompilerScores, bWeight, bReplication, cCompilerPack>(pParams);
}

template<typename TFloat, bool bHessian, size_t cCompilerScores, bool bWeight, bool bReplication, int cCompilerPack>
INLINE_RELEASE_TEMPLATED ErrorEbm OperatorBinSumsBoosting(BinSumsBoostingBridge * const pParams) {
   return TFloat::template OperatorBinSumsBoosting<bHessian, cCompilerScores, bWeight, bReplication, cCompilerPack>(pParams);
}

template<typename TFloat, bool bHessian, size_t cCompilerScores, bool bWeight, bool bReplication>
INLINE_RELEASE_TEMPLATED static ErrorEbm BitPackBoosting(BinSumsBoostingBridge * const pParams) {
   if(k_cItemsPerBitPackNone != pParams->m_cPack) {
      return OperatorBinSumsBoosting<TFloat, bHessian, cCompilerScores, bWeight, bReplication, k_cItemsPerBitPackDynamic>(pParams);
   } else {
      // this needs to be special cased because otherwise we would inject comparisons into the dynamic version
      return OperatorBinSumsBoosting<TFloat, bHessian, cCompilerScores, bWeight, bReplication, k_cItemsPerBitPackNone>(pParams);
   }
}


template<typename TFloat, bool bHessian, size_t cCompilerScores>
INLINE_RELEASE_TEMPLATED static ErrorEbm FinalOptionsBoosting(BinSumsBoostingBridge * const pParams) {
   if(nullptr != pParams->m_aWeights) {
      static constexpr bool bWeight = true;

      if(nullptr != pParams->m_pCountOccurrences) {
         static constexpr bool bReplication = true;
         return BitPackBoosting<TFloat, bHessian, cCompilerScores, bWeight, bReplication>(pParams);
      } else {
         static constexpr bool bReplication = false;
         return BitPackBoosting<TFloat, bHessian, cCompilerScores, bWeight, bReplication>(pParams);
      }
   } else {
      static constexpr bool bWeight = false;

      // we use the weights to hold both the weights and the inner bag counts if there are inner bags
      EBM_ASSERT(nullptr == pParams->m_pCountOccurrences);
      static constexpr bool bReplication = false;

      return BitPackBoosting<TFloat, bHessian, cCompilerScores, bWeight, bReplication>(pParams);
   }
}


template<typename TFloat, bool bHessian, size_t cPossibleScores>
struct CountClassesBoosting final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsBoostingBridge * const pParams) {
      if(cPossibleScores == pParams->m_cScores) {
         return FinalOptionsBoosting<TFloat, bHessian, cPossibleScores>(pParams);
      } else {
         return CountClassesBoosting<TFloat, bHessian, cPossibleScores + 1>::Func(pParams);
      }
   }
};
template<typename TFloat, bool bHessian>
struct CountClassesBoosting<TFloat, bHessian, k_cCompilerScoresMax + 1> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsBoostingBridge * const pParams) {
      return FinalOptionsBoosting<TFloat, bHessian, k_dynamicScores>(pParams);
   }
};

template<typename TFloat>
INLINE_RELEASE_TEMPLATED static ErrorEbm BinSumsBoosting(BinSumsBoostingBridge * const pParams) {
   LOG_0(Trace_Verbose, "Entered BinSumsBoosting");

   // all our memory should be aligned. It is required by SIMD for correctness or performance
   EBM_ASSERT(IsAligned(pParams->m_aGradientsAndHessians));
   EBM_ASSERT(IsAligned(pParams->m_aWeights));
   EBM_ASSERT(IsAligned(pParams->m_pCountOccurrences));
   EBM_ASSERT(IsAligned(pParams->m_aPacked));
   EBM_ASSERT(IsAligned(pParams->m_aFastBins));

   ErrorEbm error;

   EBM_ASSERT(1 <= pParams->m_cScores);
   if(EBM_FALSE != pParams->m_bHessian) {
      if(size_t { 1 } != pParams->m_cScores) {
         // muticlass
         error = CountClassesBoosting<TFloat, true, k_cCompilerScoresStart>::Func(pParams);
      } else {
         error = FinalOptionsBoosting<TFloat, true, k_oneScore>(pParams);
      }
   } else {
      if(size_t { 1 } != pParams->m_cScores) {
         // Odd: gradient multiclass. Allow it, but do not optimize for it
         error = FinalOptionsBoosting<TFloat, false, k_dynamicScores>(pParams);
      } else {
         error = FinalOptionsBoosting<TFloat, false, k_oneScore>(pParams);
      }
   }

   LOG_0(Trace_Verbose, "Exited BinSumsBoosting");

   return error;
}

} // DEFINED_ZONE_NAME

#endif // BIN_SUMS_BOOSTING_HPP