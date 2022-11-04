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

template<ptrdiff_t cCompilerClasses, ptrdiff_t compilerBitPack>
class BinSumsBoostingInternal final {
public:

   BinSumsBoostingInternal() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static void Func(BinSumsBoostingBridge * const pParams) {
      constexpr bool bCompilerZeroDimensional = k_cItemsPerBitPackNone == compilerBitPack;
      constexpr bool bClassification = IsClassification(cCompilerClasses);
      constexpr size_t cCompilerScores = GetCountScores(cCompilerClasses);

      const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, pParams->m_cClasses);
      const size_t cScores = GetCountScores(cClasses);

      auto * const aBins = pParams->m_aFastBins->Specialize<FloatFast, bClassification, cCompilerScores>();
      EBM_ASSERT(nullptr != aBins);

      const size_t cSamples = pParams->m_cSamples;
      EBM_ASSERT(1 <= cSamples);
      const FloatFast * pGradientAndHessian = pParams->m_aGradientsAndHessians;
      const FloatFast * const pGradientAndHessiansEnd = pGradientAndHessian + (bClassification ? 2 : 1) * cScores * cSamples;

      size_t cBitsPerItemMax;
      ptrdiff_t cShift;
      ptrdiff_t cShiftReset;
      size_t maskBits;
      const StorageDataType * pInputData;
      size_t cBytesPerBin;

      Bin<FloatFast, bClassification, cCompilerScores> * pBin;

      if(bCompilerZeroDimensional) {
         pBin = aBins;
      } else {
         const ptrdiff_t cPack = GET_ITEMS_PER_BIT_PACK(compilerBitPack, pParams->m_cPack);
         EBM_ASSERT(k_cItemsPerBitPackNone != cPack); // we require this condition to be templated

         const size_t cItemsPerBitPack = static_cast<size_t>(cPack);
         EBM_ASSERT(1 <= cItemsPerBitPack);
         EBM_ASSERT(cItemsPerBitPack <= k_cBitsForStorageType);
         cBitsPerItemMax = GetCountBits(cItemsPerBitPack);

         cShift = static_cast<ptrdiff_t>((cSamples - 1) % cItemsPerBitPack * cBitsPerItemMax);
         cShiftReset = static_cast<ptrdiff_t>((cItemsPerBitPack - 1) * cBitsPerItemMax);

         EBM_ASSERT(1 <= cBitsPerItemMax);
         EBM_ASSERT(cBitsPerItemMax <= k_cBitsForSizeT);
         maskBits = (~size_t { 0 }) >> (k_cBitsForSizeT - cBitsPerItemMax);

         pInputData = pParams->m_aPacked;

         EBM_ASSERT(!IsOverflowBinSize<FloatFast>(bClassification, cScores)); // we're accessing allocated memory
         cBytesPerBin = GetBinSize<FloatFast>(bClassification, cScores);
      }

      const size_t * pCountOccurrences = pParams->m_pCountOccurrences;
      const FloatFast * pWeight = pParams->m_aWeights;
      EBM_ASSERT(nullptr != pWeight); // TODO: make this so that we can have a nullptr for weight!
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

            const size_t cOccurences = *pCountOccurrences;
            const FloatFast weight = *pWeight;

#ifndef NDEBUG
            weightTotalDebug += weight;
#endif // NDEBUG

            ++pCountOccurrences;
            ++pWeight;
            pBin->SetCountSamples(pBin->GetCountSamples() + cOccurences);
            pBin->SetWeight(pBin->GetWeight() + weight);

            auto * pGradientPair = pBin->GetGradientPairs();

            size_t iScore = 0;

#ifndef NDEBUG
#ifdef EXPAND_BINARY_LOGITS
            static constexpr bool bExpandBinaryLogits = true;
#else // EXPAND_BINARY_LOGITS
            static constexpr bool bExpandBinaryLogits = false;
#endif // EXPAND_BINARY_LOGITS
            FloatFast gradientTotalDebug = 0;
#endif // NDEBUG
            do {
               const FloatFast gradient = *pGradientAndHessian;
#ifndef NDEBUG
               gradientTotalDebug += gradient;
#endif // NDEBUG
               pGradientPair[iScore].m_sumGradients += gradient * weight;
               if(bClassification) {
                  // TODO : this code gets executed for each InnerBag set.  I could probably execute it once and then all the
                  //   InnerBag sets would have this value, but I would need to store the computation in a new memory place, and it might 
                  //   make more sense to calculate this values in the CPU rather than put more pressure on memory.  I think controlling this should be 
                  //   done in a MACRO and we should use a class to hold the gradient and this computation from that value and then comment out the 
                  //   computation if not necssary and access it through an accessor so that we can make the change entirely via macro
                  const FloatFast hessian = *(pGradientAndHessian + 1);
                  pGradientPair[iScore].SetHess(pGradientPair[iScore].GetHess() + hessian * weight);
               }
               pGradientAndHessian += bClassification ? 2 : 1;
               ++iScore;
               // if we use this specific format where (iScore < cScores) then the compiler collapses alway the loop for small cScores values
               // if we make this (iScore != cScores) then the loop is not collapsed
               // the compiler seems to not mind if we make this a for loop or do loop in terms of collapsing away the loop
            } while(iScore < cScores);

            EBM_ASSERT(
               !bClassification ||
               ptrdiff_t { 2 } == cClasses && !bExpandBinaryLogits ||
               -k_epsilonGradient < gradientTotalDebug && gradientTotalDebug < k_epsilonGradient
            );

            if(bCompilerZeroDimensional) {
               if(pGradientAndHessiansEnd == pGradientAndHessian) {
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
      } while(pGradientAndHessiansEnd != pGradientAndHessian);

      EBM_ASSERT(0 < weightTotalDebug);
      EBM_ASSERT(weightTotalDebug * FloatFast { 0.999 } <= pParams->m_totalWeightDebug &&
         pParams->m_totalWeightDebug <= FloatFast { 1.001 } * weightTotalDebug);
   }
};

template<ptrdiff_t cCompilerClasses>
INLINE_RELEASE_TEMPLATED static void BitPack(BinSumsBoostingBridge * const pParams) {
   if(k_cItemsPerBitPackNone != pParams->m_cPack) {
      BinSumsBoostingInternal<cCompilerClasses, k_cItemsPerBitPackDynamic>::Func(pParams);
   } else {
      // this needs to be special cased because otherwise we would inject comparisons into the dynamic version
      BinSumsBoostingInternal<cCompilerClasses, k_cItemsPerBitPackNone>::Func(pParams);
   }
}

template<ptrdiff_t cPossibleClasses>
class BinSumsBoostingNormalTarget final {
public:

   BinSumsBoostingNormalTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static void Func(BinSumsBoostingBridge * const pParams) {
      static_assert(IsClassification(cPossibleClasses), "cPossibleClasses needs to be a classification");
      static_assert(cPossibleClasses <= k_cCompilerClassesMax, "We can't have this many items in a data pack.");

      EBM_ASSERT(IsClassification(pParams->m_cClasses));
      EBM_ASSERT(pParams->m_cClasses <= k_cCompilerClassesMax);

      if(cPossibleClasses == pParams->m_cClasses) {
         BitPack<cPossibleClasses>(pParams);
      } else {
         BinSumsBoostingNormalTarget<cPossibleClasses + 1>::Func(pParams);
      }
   }
};

template<>
class BinSumsBoostingNormalTarget<k_cCompilerClassesMax + 1> final {
public:

   BinSumsBoostingNormalTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static void Func(BinSumsBoostingBridge * const pParams) {
      static_assert(IsClassification(k_cCompilerClassesMax), "k_cCompilerClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pParams->m_cClasses));
      EBM_ASSERT(k_cCompilerClassesMax < pParams->m_cClasses);

      BitPack<k_dynamicClassification>(pParams);
   }
};

extern void BinSumsBoosting(BinSumsBoostingBridge * const pParams) {
   LOG_0(Trace_Verbose, "Entered BinSumsBoosting");

   if(IsClassification(pParams->m_cClasses)) {
      BinSumsBoostingNormalTarget<2>::Func(pParams);
   } else {
      EBM_ASSERT(IsRegression(pParams->m_cClasses));
      BitPack<k_regression>(pParams);
   }

   LOG_0(Trace_Verbose, "Exited BinSumsBoosting");
}

} // DEFINED_ZONE_NAME
