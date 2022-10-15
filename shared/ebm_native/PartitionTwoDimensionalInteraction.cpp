// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h"
#include "common_c.h" // LIKELY
#include "zones.h"

#include "ebm_stats.hpp"
#include "GradientPair.hpp"
#include "Bin.hpp"
#include "TensorTotalsSum.hpp"
#include "InteractionCore.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<ptrdiff_t cCompilerClasses>
class PartitionTwoDimensionalInteractionInternal final {
public:

   PartitionTwoDimensionalInteractionInternal() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      InteractionCore * const pInteractionCore,
      const size_t cRuntimeRealDimensions,
      const size_t * const acBins,
      const InteractionFlags flags,
      const size_t cSamplesLeafMin,
      BinBase * const aAuxiliaryBinsBase,
      BinBase * const aBinsBase
#ifndef NDEBUG
      , const BinBase * const aDebugCopyBinsBase
      , const BinBase * const pBinsEndDebug
#endif // NDEBUG
   ) {
      static constexpr bool bClassification = IsClassification(cCompilerClasses);
      static constexpr size_t cCompilerScores = GetCountScores(cCompilerClasses);
      static constexpr size_t cCompilerDimensions = 2;

      auto * const aAuxiliaryBins = aAuxiliaryBinsBase->Specialize<FloatBig, bClassification, cCompilerScores>();
      auto * const aBins = aBinsBase->Specialize<FloatBig, bClassification, cCompilerScores>();

#ifndef NDEBUG
      auto * const aDebugCopyBins = aDebugCopyBinsBase->Specialize<FloatBig, bClassification, cCompilerScores>();
#endif // NDEBUG

      const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, pInteractionCore->GetCountClasses());

      const size_t cScores = GetCountScores(cClasses);
      const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

      const size_t cRealDimensions = GET_COUNT_DIMENSIONS(cCompilerDimensions, cRuntimeRealDimensions);
      EBM_ASSERT(k_dynamicDimensions == cCompilerDimensions || cCompilerDimensions == cRuntimeRealDimensions);

      TensorSumDimension aDimensions[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
      size_t iDimensionInit = 0;
      do {
         // move data to a local variable that the compiler can reason about and then eliminate by moving to CPU registers
         aDimensions[iDimensionInit].m_cBins = acBins[iDimensionInit];
         ++iDimensionInit;
      } while(cRealDimensions != iDimensionInit);

      auto * const p_DO_NOT_USE_DIRECTLY_00 = IndexBin(aAuxiliaryBins, cBytesPerBin * 0);
      ASSERT_BIN_OK(cBytesPerBin, p_DO_NOT_USE_DIRECTLY_00, pBinsEndDebug);
      auto * const p_DO_NOT_USE_DIRECTLY_01 = IndexBin(aAuxiliaryBins, cBytesPerBin * 1);
      ASSERT_BIN_OK(cBytesPerBin, p_DO_NOT_USE_DIRECTLY_01, pBinsEndDebug);
      auto * const p_DO_NOT_USE_DIRECTLY_10 = IndexBin(aAuxiliaryBins, cBytesPerBin * 2);
      ASSERT_BIN_OK(cBytesPerBin, p_DO_NOT_USE_DIRECTLY_10, pBinsEndDebug);
      auto * const p_DO_NOT_USE_DIRECTLY_11 = IndexBin(aAuxiliaryBins, cBytesPerBin * 3);
      ASSERT_BIN_OK(cBytesPerBin, p_DO_NOT_USE_DIRECTLY_11, pBinsEndDebug);

      Bin<FloatBig, bClassification, cCompilerScores> bin00;
      Bin<FloatBig, bClassification, cCompilerScores> bin01;
      Bin<FloatBig, bClassification, cCompilerScores> bin10;
      Bin<FloatBig, bClassification, cCompilerScores> bin11;

      // if we know how many scores there are, use the memory on the stack where the compiler can optimize access
      static constexpr bool bUseStackMemory = k_dynamicClassification != cCompilerClasses;
      auto * const aGradientPairs00 = bUseStackMemory ? bin00.GetGradientPairs() : p_DO_NOT_USE_DIRECTLY_00->GetGradientPairs();
      auto * const aGradientPairs01 = bUseStackMemory ? bin01.GetGradientPairs() : p_DO_NOT_USE_DIRECTLY_01->GetGradientPairs();
      auto * const aGradientPairs10 = bUseStackMemory ? bin10.GetGradientPairs() : p_DO_NOT_USE_DIRECTLY_10->GetGradientPairs();
      auto * const aGradientPairs11 = bUseStackMemory ? bin11.GetGradientPairs() : p_DO_NOT_USE_DIRECTLY_11->GetGradientPairs();

      EBM_ASSERT(0 < cSamplesLeafMin);

#ifndef NDEBUG
      bool bAnySplits = false;
#endif // NDEBUG

      // if a negative value were to occur, then it would be due to numeric instability, so clip it to zero here
      FloatBig bestGain = 0;

      EBM_ASSERT(2 <= aDimensions[0].m_cBins); // 1 cBins in any dimension returns an interaction score of 0
      aDimensions[0].m_iPoint = 0;
      do {
         EBM_ASSERT(2 <= aDimensions[1].m_cBins); // 1 cBins in any dimension returns an interaction score of 0
         aDimensions[1].m_iPoint = 0;
         do {
            EBM_ASSERT(2 == cRealDimensions); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
            TensorTotalsSum<cCompilerClasses, cCompilerDimensions>(
               cClasses,
               cRealDimensions,
               aDimensions,
               0x00,
               aBins,
               bin00,
               aGradientPairs00
#ifndef NDEBUG
               , aDebugCopyBins
               , pBinsEndDebug
#endif // NDEBUG
            );
            if(LIKELY(cSamplesLeafMin <= bin00.GetCountSamples())) {
               EBM_ASSERT(2 == cRealDimensions); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
               TensorTotalsSum<cCompilerClasses, cCompilerDimensions>(
                  cClasses,
                  cRealDimensions,
                  aDimensions,
                  0x01,
                  aBins,
                  bin01,
                  aGradientPairs01
#ifndef NDEBUG
                  , aDebugCopyBins
                  , pBinsEndDebug
#endif // NDEBUG
               );
               if(LIKELY(cSamplesLeafMin <= bin01.GetCountSamples())) {
                  EBM_ASSERT(2 == cRealDimensions); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
                  TensorTotalsSum<cCompilerClasses, cCompilerDimensions>(
                     cClasses,
                     cRealDimensions,
                     aDimensions,
                     0x02,
                     aBins,
                     bin10,
                     aGradientPairs10
#ifndef NDEBUG
                     , aDebugCopyBins
                     , pBinsEndDebug
#endif // NDEBUG
                  );
                  if(LIKELY(cSamplesLeafMin <= bin10.GetCountSamples())) {
                     EBM_ASSERT(2 == cRealDimensions); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
                     TensorTotalsSum<cCompilerClasses, cCompilerDimensions>(
                        cClasses,
                        cRealDimensions,
                        aDimensions,
                        0x03,
                        aBins,
                        bin11,
                        aGradientPairs11
#ifndef NDEBUG
                        , aDebugCopyBins
                        , pBinsEndDebug
#endif // NDEBUG
                     );
                     if(LIKELY(cSamplesLeafMin <= bin11.GetCountSamples())) {
#ifndef NDEBUG
                        bAnySplits = true;
#endif // NDEBUG
                        FloatBig gain = 0;

                        for(size_t iScore = 0; iScore < cScores; ++iScore) {
                           // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators 
                           // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

                           static constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;

                           // n = numerator (sum_gradients), d = denominator (sum_hessians or weight)

                           const FloatBig n00 = aGradientPairs00[iScore].m_sumGradients;
                           const FloatBig d00 = bUseLogitBoost ? aGradientPairs00[iScore].GetHess() : bin00.GetWeight();

                           const FloatBig n01 = aGradientPairs01[iScore].m_sumGradients;
                           const FloatBig d01 = bUseLogitBoost ? aGradientPairs01[iScore].GetHess() : bin01.GetWeight();

                           const FloatBig n10 = aGradientPairs10[iScore].m_sumGradients;
                           const FloatBig d10 = bUseLogitBoost ? aGradientPairs10[iScore].GetHess() : bin10.GetWeight();

                           const FloatBig n11 = aGradientPairs11[iScore].m_sumGradients;
                           const FloatBig d11 = bUseLogitBoost ? aGradientPairs11[iScore].GetHess() : bin11.GetWeight();

                           if(0 != (InteractionFlags_Pure & flags)) {
                              // purified gain

                              // If we have a 2x2 matrix of updates, we can purify the updates using an equation
                              // -------------------
                              // |update00|update01|
                              // |-----------------|
                              // |update10|update11|
                              // -------------------
                              //
                              // The update in each cell consists of a main update from feature0, 
                              // a main update from feature1, and a purified update:
                              //   update00 = main0_0 + main1_0 + pure00
                              //   update01 = main0_1 + main1_0 + pure01
                              //   update10 = main0_0 + main1_1 + pure10
                              //   update11 = main0_1 + main1_1 + pure11
                              // We can add and subtract these to remove the main contributions:
                              //   update00 - update01 - update10 + update11 = 
                              //        main0_0 + main1_0 + pure00 
                              //      - main0_1 - main1_0 - pure01 
                              //      - main0_0 - main1_1 - pure10
                              //      + main0_1 + main1_1 + pure11
                              // Which simplifies to:
                              //   update00 - update01 - update10 + update11 = pure00 - pure01 - pure10 + pure11
                              // Purification means the pure update multiplied by the count must sum to zero
                              // across all rows/columns:
                              //   pure00 * count00 + pure01 * count01 = 0
                              //   pure01 * count01 + pure11 * count11 = 0
                              //   pure11 * count11 + pure10 * count10 = 0
                              //   pure10 * count10 + pure00 * count00 = 0
                              // So:
                              //   pure01 = -pure00 * count00 / count01
                              //   pure10 = -pure00 * count00 / count10
                              // And we can relate pure00 to pure11 by adding/subtracting the above:
                              //     pure00 * count00 + pure01 * count01 
                              //   - pure01 * count01 - pure11 * count11 
                              //   - pure11 * count11 - pure10 * count10 
                              //   + pure10 * count10 + pure00 * count00 = 0
                              // which simplifies to:
                              //   2 * pure00 * count00 - 2 * pure11 * count11 = 0
                              // and then:
                              //   pure11 = pure00 * count00 / count11
                              // From the above:
                              //   update00 - update01 - update10 + update11 = pure00 - pure01 - pure10 + pure11
                              // we can substitute to get: 
                              //   update00 - update01 - update10 + update11 = 
                              //      pure00 
                              //      + pure00 * count00 / count01 
                              //      + pure00 * count00 / count10 
                              //      + pure00 * count00 / count11
                              // Which simplifies to:
                              //   pure00 = (update00 - update01 - update10 + update11) /
                              //     (1 + count00 / count01 + count00 / count10 + count00 / count11)
                              // The other pure effects can be derived the same way.

                              // if any of the denominators (weights) are zero then the purified gain will be
                              // zero.  Handle it here to avoid division by zero
                              if(0 != d00 && 0 != d01 && 0 != d10 && 0 != d11) {

                                 // TODO: instead of checking the denominators for zero above, can we do it earlier?
                                 // If we're using hessians then we'd need it here, but we aren't using them yet

                                 // calculate what the full updates would be for non-purified:
                                 // u = update (non-purified)
                                 const FloatBig u00 = n00 / d00;
                                 const FloatBig u01 = n01 / d01;
                                 const FloatBig u10 = n10 / d10;
                                 const FloatBig u11 = n11 / d11;

                                 // common part of equations (positive for 00 & 11 equations, negative for 01 and 10)
                                 const FloatBig common = u00 - u01 - u10 + u11;

                                 // p = purified update
                                 const FloatBig p00 = common / (FloatBig { 1 } + d00 / d01 + d00 / d10 + d00 / d11);
                                 const FloatBig p01 = common / (FloatBig { -1 } - d01 / d00 - d01 / d10 - d01 / d11);
                                 const FloatBig p10 = common / (FloatBig { -1 } - d10 / d00 - d10 / d01 - d10 / d11);
                                 const FloatBig p11 = common / (FloatBig { 1 } + d11 / d00 + d11 / d01 + d11 / d10);

                                 // g = gain
                                 const FloatBig g00 = EbmStats::CalcPartialGainFromUpdate(p00, d00);
                                 const FloatBig g01 = EbmStats::CalcPartialGainFromUpdate(p01, d01);
                                 const FloatBig g10 = EbmStats::CalcPartialGainFromUpdate(p10, d10);
                                 const FloatBig g11 = EbmStats::CalcPartialGainFromUpdate(p11, d11);

#ifndef NDEBUG
                                 // r = reconsituted numerator (after purification)
                                 const FloatBig r00 = p00 * d00;
                                 const FloatBig r01 = p01 * d01;
                                 const FloatBig r10 = p10 * d10;
                                 const FloatBig r11 = p11 * d11;

                                 // purification means summing any direction gives us zero
                                 EBM_ASSERT(std::abs(r00 + r01) < 0.001);
                                 EBM_ASSERT(std::abs(r01 + r11) < 0.001);
                                 EBM_ASSERT(std::abs(r11 + r10) < 0.001);
                                 EBM_ASSERT(std::abs(r10 + r00) < 0.001);

                                 // if all of these added together are zero, then the parent partial gain should also 
                                 // be zero, which means we can avoid calculating the parent partial gain.
                                 EBM_ASSERT(std::abs(r00 + r01 + r10 + r11) < 0.001);

                                 EBM_ASSERT(std::abs(g00 - EbmStats::CalcPartialGain(r00, d00)) < 0.001);
                                 EBM_ASSERT(std::abs(g01 - EbmStats::CalcPartialGain(r01, d01)) < 0.001);
                                 EBM_ASSERT(std::abs(g10 - EbmStats::CalcPartialGain(r10, d10)) < 0.001);
                                 EBM_ASSERT(std::abs(g11 - EbmStats::CalcPartialGain(r11, d11)) < 0.001);
#endif // NDEBUG
                                 gain += g00;
                                 gain += g01;
                                 gain += g10;
                                 gain += g11;
                              }
                           } else {
                              // non-purified gain
                              gain += EbmStats::CalcPartialGain(n00, d00);
                              gain += EbmStats::CalcPartialGain(n01, d01);
                              gain += EbmStats::CalcPartialGain(n10, d10);
                              gain += EbmStats::CalcPartialGain(n11, d11);
                           }
                        }
                        EBM_ASSERT(std::isnan(gain) || 0 <= gain); // sumations of positive numbers should be positive

                        // If we get a NaN result, we'd like to propagate it by making bestGain NaN.  
                        // The rules for NaN values say that non equality comparisons are all false so, 
                        // let's flip this comparison such that it should be true for NaN values.
                        if(UNLIKELY(/* NaN */ !LIKELY(gain <= bestGain))) {
                           bestGain = gain;
                        } else {
                           EBM_ASSERT(!std::isnan(gain));
                        }
                     }
                  }
               }
            }
            ++aDimensions[1].m_iPoint;
         } while(aDimensions[1].m_cBins - 1 != aDimensions[1].m_iPoint);
         ++aDimensions[0].m_iPoint;
      } while(aDimensions[0].m_cBins - 1 != aDimensions[0].m_iPoint);

      // we start from zero, so bestGain can't be negative here
      EBM_ASSERT(std::isnan(bestGain) || 0 <= bestGain);

      if(0 == (InteractionFlags_Pure & flags)) {
         // if we are detecting impure interaction then so far we have only calculated the children partial gain 
         // but we still need to subtract the partial gain of the parent to have
         // gain. All the splits we've analyzed so far though had the same non-split partial gain, so we subtract it here
         // instead of inside the loop.

         // the bin before the aAuxiliaryBins is the last summation bin of aBinsBase, 
         // which contains the totals of all bins
         const auto * const pTotal = NegativeIndexBin(aAuxiliaryBins, cBytesPerBin);
         const FloatBig weightAll = pTotal->GetWeight();
         const auto * const aGradientPairs = pTotal->GetGradientPairs();
         for(size_t iScore = 0; iScore < cScores; ++iScore) {
            // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators 
            // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

            static constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;
            bestGain -= EbmStats::CalcPartialGain(
               aGradientPairs[iScore].m_sumGradients,
               bUseLogitBoost ? aGradientPairs[iScore].GetHess() : weightAll
            );
         }

         // bestGain should be positive, or NaN, BUT it can be slightly negative due to floating point noise
         // it could also be -inf if the parent/total bin overflows, but the children parts did not.
         // bestGain can also be substantially negative if we didn't find any legal cuts and 
         // then we subtracted the base partial gain here from zero

         // if no legal splits were found, then bestGain will be zero.  In theory we should
         // therefore not subtract the parent partial gain, but doing so does no harm since we later set any
         // negative interaction score to zero in the caller of this function.  Due to that we don't
         // need to check here, since any value we subtract from zero will lead to a negative number and
         // then will be zeroed by our caller
         // BUT, for debugging purposes, check here for that condition so that we can check for illegal negative gain.

         EBM_ASSERT(std::isnan(bestGain) ||
            -std::numeric_limits<FloatBig>::infinity() == bestGain ||
            k_epsilonNegativeGainAllowed <= bestGain || !bAnySplits);
      }

      // we clean up bestGain in the caller, since this function is templated and created many times
      return static_cast<double>(bestGain);
   }
};

template<ptrdiff_t cPossibleClasses>
class PartitionTwoDimensionalInteractionTarget final {
public:

   PartitionTwoDimensionalInteractionTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      InteractionCore * const pInteractionCore,
      const size_t cRealDimensions,
      const size_t * const acBins,
      const InteractionFlags flags,
      const size_t cSamplesLeafMin,
      BinBase * aAuxiliaryBinsBase,
      BinBase * const aBinsBase
#ifndef NDEBUG
      , const BinBase * const aDebugCopyBinsBase
      , const BinBase * const pBinsEndDebug
#endif // NDEBUG
   ) {
      static_assert(IsClassification(cPossibleClasses), "cPossibleClasses needs to be a classification");
      static_assert(cPossibleClasses <= k_cCompilerClassesMax, "We can't have this many items in a data pack.");

      const ptrdiff_t cRuntimeClasses = pInteractionCore->GetCountClasses();
      EBM_ASSERT(IsClassification(cRuntimeClasses));
      EBM_ASSERT(cRuntimeClasses <= k_cCompilerClassesMax);

      if(cPossibleClasses == cRuntimeClasses) {
         return PartitionTwoDimensionalInteractionInternal<cPossibleClasses>::Func(
            pInteractionCore,
            cRealDimensions,
            acBins,
            flags,
            cSamplesLeafMin,
            aAuxiliaryBinsBase,
            aBinsBase
#ifndef NDEBUG
            , aDebugCopyBinsBase
            , pBinsEndDebug
#endif // NDEBUG
         );
      } else {
         return PartitionTwoDimensionalInteractionTarget<cPossibleClasses + 1>::Func(
            pInteractionCore,
            cRealDimensions,
            acBins,
            flags,
            cSamplesLeafMin,
            aAuxiliaryBinsBase,
            aBinsBase
#ifndef NDEBUG
            , aDebugCopyBinsBase
            , pBinsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<>
class PartitionTwoDimensionalInteractionTarget<k_cCompilerClassesMax + 1> final {
public:

   PartitionTwoDimensionalInteractionTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(
      InteractionCore * const pInteractionCore,
      const size_t cRealDimensions,
      const size_t * const acBins,
      const InteractionFlags flags,
      const size_t cSamplesLeafMin,
      BinBase * aAuxiliaryBinsBase,
      BinBase * const aBinsBase
#ifndef NDEBUG
      , const BinBase * const aDebugCopyBinsBase
      , const BinBase * const pBinsEndDebug
#endif // NDEBUG
   ) {
      static_assert(IsClassification(k_cCompilerClassesMax), "k_cCompilerClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pInteractionCore->GetCountClasses()));
      EBM_ASSERT(k_cCompilerClassesMax < pInteractionCore->GetCountClasses());

      return PartitionTwoDimensionalInteractionInternal<k_dynamicClassification>::Func(
         pInteractionCore,
         cRealDimensions,
         acBins,
         flags,
         cSamplesLeafMin,
         aAuxiliaryBinsBase,
         aBinsBase
#ifndef NDEBUG
         , aDebugCopyBinsBase
         , pBinsEndDebug
#endif // NDEBUG
      );
   }
};

extern double PartitionTwoDimensionalInteraction(
   InteractionCore * const pInteractionCore,
   const size_t cRealDimensions,
   const size_t * const acBins,
   const InteractionFlags flags,
   const size_t cSamplesLeafMin,
   BinBase * aAuxiliaryBinsBase,
   BinBase * const aBinsBase
#ifndef NDEBUG
   , const BinBase * const aDebugCopyBinsBase
   , const BinBase * const pBinsEndDebug
#endif // NDEBUG
) {
   const ptrdiff_t cRuntimeClasses = pInteractionCore->GetCountClasses();

   if(IsClassification(cRuntimeClasses)) {
      return PartitionTwoDimensionalInteractionTarget<2>::Func(
         pInteractionCore,
         cRealDimensions,
         acBins,
         flags,
         cSamplesLeafMin,
         aAuxiliaryBinsBase,
         aBinsBase
#ifndef NDEBUG
         , aDebugCopyBinsBase
         , pBinsEndDebug
#endif // NDEBUG
      );
   } else {
      EBM_ASSERT(IsRegression(cRuntimeClasses));
      return PartitionTwoDimensionalInteractionInternal<k_regression>::Func(
         pInteractionCore,
         cRealDimensions,
         acBins,
         flags,
         cSamplesLeafMin,
         aAuxiliaryBinsBase,
         aBinsBase
#ifndef NDEBUG
         , aDebugCopyBinsBase
         , pBinsEndDebug
#endif // NDEBUG
      );
   }
}

} // DEFINED_ZONE_NAME
