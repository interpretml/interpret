// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT
#include "zones.h"

#include "GradientPair.hpp"
#include "Bin.hpp"

#include "ebm_internal.hpp"
#include "Term.hpp"
#include "InnerBag.hpp"
#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<bool bHessian, size_t cCompilerScores, ptrdiff_t compilerBitPack, bool bWeight, bool bReplication>
INLINE_RELEASE_TEMPLATED static ErrorEbm BinSumsBoostingInternal(BinSumsBoostingBridge * const pParams) {
   static constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == compilerBitPack;
   static constexpr size_t cArrayScores = GetArrayScores(cCompilerScores);

   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pParams->m_cScores);

   auto * const aBins = pParams->m_aFastBins->Specialize<FloatFast, bHessian, cArrayScores>();
   EBM_ASSERT(nullptr != aBins);

   const size_t cSamples = pParams->m_cSamples;
   EBM_ASSERT(1 <= cSamples);

   const FloatFast * pGradientAndHessian = pParams->m_aGradientsAndHessians;
   const FloatFast * const pGradientsAndHessiansEnd = pGradientAndHessian + (bHessian ? 2 : 1) * cScores * cSamples;

   size_t cBytesPerBin;
   size_t cBitsPerItemMax;
   ptrdiff_t cShift;
   ptrdiff_t cShiftReset;
   size_t maskBits;
   const StorageDataType * pInputData;

   Bin<FloatFast, bHessian, cArrayScores> * pBin;

   if(bCompilerZeroDimensional) {
      pBin = aBins;
   } else {
      EBM_ASSERT(!IsOverflowBinSize<FloatFast>(bHessian, cScores)); // we're accessing allocated memory
      cBytesPerBin = GetBinSize<FloatFast>(bHessian, cScores);

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
         } else {
            // TODO: In the future we'd like to eliminate this but we need the ability to change the Bin class
            //       such that we can remove that field optionally
            pBin->SetWeight(pBin->GetWeight() + FloatFast { 1 });
         }

         auto * const aGradientPair = pBin->GetGradientPairs();
         size_t iScore = 0;
         do {
            auto * const pGradientPair = &aGradientPair[iScore];
            FloatFast gradient = bHessian ? pGradientAndHessian[iScore << 1] : pGradientAndHessian[iScore];
            if(bWeight) {
               gradient *= weight;
            }
            pGradientPair->m_sumGradients += gradient;
            if(bHessian) {
               FloatFast hessian = pGradientAndHessian[(iScore << 1) + 1];
               if(bWeight) {
                  hessian *= weight;
               }
               pGradientPair->SetHess(pGradientPair->GetHess() + hessian);
            }
            ++iScore;
         } while(cScores != iScore);
         pGradientAndHessian += bHessian ? cScores << 1 : cScores;

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

   return Error_None;
}


template<bool bHessian, size_t cCompilerScores, ptrdiff_t compilerBitPack>
INLINE_RELEASE_TEMPLATED static ErrorEbm FinalOptions(BinSumsBoostingBridge * const pParams) {
   if(nullptr != pParams->m_aWeights) {
      static constexpr bool bWeight = true;

      if(nullptr != pParams->m_pCountOccurrences) {
         static constexpr bool bReplication = true;
         return BinSumsBoostingInternal<bHessian, cCompilerScores, compilerBitPack, bWeight, bReplication>(pParams);
      } else {
         static constexpr bool bReplication = false;
         return BinSumsBoostingInternal<bHessian, cCompilerScores, compilerBitPack, bWeight, bReplication>(pParams);
      }
   } else {
      static constexpr bool bWeight = false;

      // we use the weights to hold both the weights and the inner bag counts if there are inner bags
      EBM_ASSERT(nullptr == pParams->m_pCountOccurrences);
      static constexpr bool bReplication = false;

      return BinSumsBoostingInternal<bHessian, cCompilerScores, compilerBitPack, bWeight, bReplication>(pParams);
   }
}


template<bool bHessian, size_t cCompilerScores>
INLINE_RELEASE_TEMPLATED static ErrorEbm BitPack(BinSumsBoostingBridge * const pParams) {
   if(k_cItemsPerBitPackNone != pParams->m_cPack) {
      return FinalOptions<bHessian, cCompilerScores, k_cItemsPerBitPackDynamic>(pParams);
   } else {
      // this needs to be special cased because otherwise we would inject comparisons into the dynamic version
      return FinalOptions<bHessian, cCompilerScores, k_cItemsPerBitPackNone>(pParams);
   }
}


template<bool bHessian, size_t cPossibleScores>
struct CountClasses final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsBoostingBridge * const pParams) {
      if(cPossibleScores == pParams->m_cScores) {
         return BitPack<bHessian, cPossibleScores>(pParams);
      } else {
         return CountClasses<bHessian, cPossibleScores + 1>::Func(pParams);
      }
   }
};
template<bool bHessian>
struct CountClasses<bHessian, k_cCompilerScoresMax + 1> final {
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BinSumsBoostingBridge * const pParams) {
      return BitPack<bHessian, k_dynamicScores>(pParams);
   }
};


extern ErrorEbm BinSumsBoosting(BinSumsBoostingBridge * const pParams) {
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
         error = CountClasses<true, k_cCompilerScoresStart>::Func(pParams);
      } else {
         error = BitPack<true, k_oneScore>(pParams);
      }
   } else {
      if(size_t { 1 } != pParams->m_cScores) {
         // Odd: gradient multiclass. Allow it, but do not optimize for it
         error = BitPack<false, k_dynamicScores>(pParams);
      } else {
         error = BitPack<false, k_oneScore>(pParams);
      }
   }

   LOG_0(Trace_Verbose, "Exited BinSumsBoosting");

   return error;
}

} // DEFINED_ZONE_NAME
