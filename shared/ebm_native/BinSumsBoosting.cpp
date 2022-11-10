// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT
#include "zones.h"

#include "ebm_internal.hpp" // k_cCompilerClassesMax

#include "Term.hpp"
#include "InnerBag.hpp"
#include "GradientPair.hpp"
#include "Bin.hpp"
#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<ptrdiff_t cCompilerClasses, ptrdiff_t compilerBitPack, bool bWeight, bool bReplication>
INLINE_RELEASE_TEMPLATED static ErrorEbm BinSumsBoostingInternal(BinSumsBoostingBridge * const pParams) {
   static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == compilerBitPack;
   static constexpr bool bClassification = IsClassification(cCompilerClasses);
   static constexpr size_t cCompilerScores = GetCountScores(cCompilerClasses);

   const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, pParams->m_cClasses);
   const size_t cScores = GetCountScores(cClasses);

   auto * const aBins = pParams->m_aFastBins->Specialize<FloatFast, bClassification, cCompilerScores>();
   EBM_ASSERT(nullptr != aBins);

   const size_t cSamples = pParams->m_cSamples;
   EBM_ASSERT(1 <= cSamples);

   const FloatFast * pGradientAndHessian = pParams->m_aGradientsAndHessians;
   const FloatFast * const pGradientsAndHessiansEnd = pGradientAndHessian + (bClassification ? 2 : 1) * cScores * cSamples;

   size_t cBytesPerBin;
   size_t cBitsPerItemMax;
   ptrdiff_t cShift;
   ptrdiff_t cShiftReset;
   size_t maskBits;
   const StorageDataType * pInputData;

   Bin<FloatFast, bClassification, cCompilerScores> * pBin;

   if(bCompilerZeroDimensional) {
      pBin = aBins;
   } else {
      EBM_ASSERT(!IsOverflowBinSize<FloatFast>(bClassification, cScores)); // we're accessing allocated memory
      cBytesPerBin = GetBinSize<FloatFast>(bClassification, cScores);

      const ptrdiff_t cPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, pParams->m_cPack);
      EBM_ASSERT(k_cItemsPerBitPackNone != cPack); // we require this condition to be templated

      const size_t cItemsPerBitPack = static_cast<size_t>(cPack);
      EBM_ASSERT(1 <= cItemsPerBitPack);
      EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);

      cBitsPerItemMax = GetCountBits<StorageDataType>(cItemsPerBitPack);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);

      cShift = static_cast<ptrdiff_t>((cSamples - 1) % cItemsPerBitPack * cBitsPerItemMax);
      cShiftReset = static_cast<ptrdiff_t>((cItemsPerBitPack - 1) * cBitsPerItemMax);

      maskBits = static_cast<size_t>(MakeLowMask<StorageDataType>(cBitsPerItemMax));

      pInputData = pParams->m_aPacked;
   }

   const size_t * pCountOccurrences;
   if(bReplication) {
      pCountOccurrences = pParams->m_pCountOccurrences;
   }

   const FloatFast * pWeight;
   if(bWeight) {
      pWeight = pParams->m_aWeights;
   }
#ifndef NDEBUG
   FloatFast weightTotalDebug = 0;
#endif // NDEBUG

   do {
      // this loop gets about twice as slow if you add a single unpredictable branching if statement based on count, even if you still access all the memory
      // in complete sequential order, so we'll probably want to use non-branching instructions for any solution like conditional selection or multiplication
      // this loop gets about 3 times slower if you use a bad pseudo random number generator like rand(), although it might be better if you inlined rand().
      // this loop gets about 10 times slower if you use a proper pseudo random number generator like std::default_random_engine
      // taking all the above together, it seems unlikley we'll use a method of separating sets via single pass randomized set splitting.  Even if count is 
      // stored in memory if shouldn't increase the time spent fetching it by 2 times, unless our bottleneck when threading is overwhelmingly memory pressure
      // related, and even then we could store the count for a single bit aleviating the memory pressure greatly, if we use the right sampling method 

      // TODO : try using a sampling method with non-repeating samples, and put the count into a bit.  Then unwind that loop either at the byte level 
      //   (8 times) or the uint64_t level.  This can be done without branching and doesn't require random number generators

      // we store the already multiplied dimensional value in *pInputData
      StorageDataType iTensorBinCombined;
      if(!bCompilerZeroDimensional) {
         // we store the already multiplied dimensional value in *pInputData
         iTensorBinCombined = *pInputData;
         ++pInputData;
      }
      while(true) {
         if(!bCompilerZeroDimensional) {
            const size_t iTensorBin = static_cast<size_t>(iTensorBinCombined >> cShift) & maskBits;
            pBin = IndexBin(aBins, cBytesPerBin * iTensorBin);
            ASSERT_BIN_OK(cBytesPerBin, pBin, pParams->m_pDebugFastBinsEnd);
         }

         if(bReplication) {
            const size_t cOccurences = *pCountOccurrences;
            pBin->SetCountSamples(pBin->GetCountSamples() + cOccurences);
            ++pCountOccurrences;
         } else {
            pBin->SetCountSamples(pBin->GetCountSamples() + size_t { 1 });
         }

         FloatFast weight;
         if(bWeight) {
            weight = *pWeight;
            pBin->SetWeight(pBin->GetWeight() + weight);
            ++pWeight;
#ifndef NDEBUG
            weightTotalDebug += weight;
#endif // NDEBUG
         } else {
            // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
            //       such that we can remove that field optionally
            pBin->SetWeight(pBin->GetWeight() + FloatFast { 1 });
         }

#ifndef NDEBUG
#ifdef EXPAND_BINARY_LOGITS
         static constexpr bool bExpandBinaryLogits = true;
#else // EXPAND_BINARY_LOGITS
         static constexpr bool bExpandBinaryLogits = false;
#endif // EXPAND_BINARY_LOGITS
         FloatFast gradientTotalDebug = 0;
#endif // NDEBUG

         auto * const aGradientPair = pBin->GetGradientPairs();
         size_t iScore = 0;
         do {
            auto * const pGradientPair = &aGradientPair[iScore];
            FloatFast gradient = bClassification ? pGradientAndHessian[iScore << 1] : pGradientAndHessian[iScore];
#ifndef NDEBUG
            gradientTotalDebug += gradient;
#endif // NDEBUG
            if(bWeight) {
               gradient *= weight;
            }
            pGradientPair->m_sumGradients += gradient;
            if(bClassification) {
               FloatFast hessian = pGradientAndHessian[(iScore << 1) + 1];
               if(bWeight) {
                  hessian *= weight;
               }
               pGradientPair->SetHess(pGradientPair->GetHess() + hessian);
            }
            ++iScore;
         } while(cScores != iScore);
         pGradientAndHessian += bClassification ? cScores << 1 : cScores;

         EBM_ASSERT(!bClassification || ptrdiff_t { 2 } == cClasses && !bExpandBinaryLogits ||
            -k_epsilonGradient < gradientTotalDebug && gradientTotalDebug < k_epsilonGradient);

         if(bCompilerZeroDimensional) {
            if(pGradientsAndHessiansEnd == pGradientAndHessian) {
               break;
            }
         } else {
            cShift -= cBitsPerItemMax;
            if(cShift < 0) {
               break;
            }
         }
      }
      if(bCompilerZeroDimensional) {
         break;
      }
      cShift = cShiftReset;
   } while(pGradientsAndHessiansEnd != pGradientAndHessian);

   EBM_ASSERT(!bWeight || 0 < pParams->m_totalWeightDebug);
   EBM_ASSERT(!bWeight || 0 < weightTotalDebug);
   EBM_ASSERT(!bWeight || (weightTotalDebug * FloatFast { 0.999 } <= pParams->m_totalWeightDebug &&
      pParams->m_totalWeightDebug <= FloatFast { 1.001 } * weightTotalDebug));
   EBM_ASSERT(bWeight || static_cast<FloatFast>(cSamples) == pParams->m_totalWeightDebug);

   return Error_None;
}


template<ptrdiff_t cCompilerClasses, ptrdiff_t compilerBitPack>
INLINE_RELEASE_TEMPLATED static ErrorEbm FinalOptions(BinSumsBoostingBridge * const pParams) {
   if(nullptr != pParams->m_aWeights) {
      static constexpr bool bWeight = true;

      if(nullptr != pParams->m_pCountOccurrences) {
         static constexpr bool bReplication = true;
         return BinSumsBoostingInternal<cCompilerClasses, compilerBitPack, bWeight, bReplication>(pParams);
      } else {
         static constexpr bool bReplication = false;
         return BinSumsBoostingInternal<cCompilerClasses, compilerBitPack, bWeight, bReplication>(pParams);
      }
   } else {
      static constexpr bool bWeight = false;

      // we use the weights to hold both the weights and the inner bag counts if there are inner bags
      EBM_ASSERT(nullptr == pParams->m_pCountOccurrences);
      static constexpr bool bReplication = false;

      return BinSumsBoostingInternal<cCompilerClasses, compilerBitPack, bWeight, bReplication>(pParams);
   }
}


template<ptrdiff_t cCompilerClasses>
INLINE_RELEASE_TEMPLATED static ErrorEbm BitPack(BinSumsBoostingBridge * const pParams) {
   if(k_cItemsPerBitPackNone != pParams->m_cPack) {
      return FinalOptions<cCompilerClasses, k_cItemsPerBitPackDynamic>(pParams);
   } else {
      // this needs to be special cased because otherwise we would inject comparisons into the dynamic version
      return FinalOptions<cCompilerClasses, k_cItemsPerBitPackNone>(pParams);
   }
}


template<ptrdiff_t cPossibleClasses>
struct CountClasses final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsBoostingBridge * const pParams) {
      if(cPossibleClasses == pParams->m_cClasses) {
         return BitPack<cPossibleClasses>(pParams);
      } else {
         return CountClasses<cPossibleClasses + 1>::Func(pParams);
      }
   }
};
template<>
struct CountClasses<k_cCompilerClassesMax + 1> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsBoostingBridge * const pParams) {
      return BitPack<k_dynamicClassification>(pParams);
   }
};


extern ErrorEbm BinSumsBoosting(BinSumsBoostingBridge * const pParams) {
   ErrorEbm error;

   LOG_0(Trace_Verbose, "Entered BinSumsBoosting");

   if(IsClassification(pParams->m_cClasses)) {
      error = CountClasses<2>::Func(pParams);
   } else {
      EBM_ASSERT(IsRegression(pParams->m_cClasses));
      error = BitPack<k_regression>(pParams);
   }

   LOG_0(Trace_Verbose, "Exited BinSumsBoosting");

   return error;
}

} // DEFINED_ZONE_NAME
