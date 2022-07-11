// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "ebm_stats.hpp"

#include "Feature.hpp"
#include "FeatureGroup.hpp"

#include "HistogramTargetEntry.hpp"
#include "HistogramBucket.hpp"

#include "InteractionCore.hpp"

#include "TensorTotalsSum.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class PartitionTwoDimensionalInteractionInternal final {
public:

   PartitionTwoDimensionalInteractionInternal() = delete; // this is a static class.  Do not construct

   static double Func(
      InteractionCore * const pInteractionCore,
      const Term * const pTerm,
      const InteractionOptionsType options,
      const size_t cSamplesRequiredForChildSplitMin,
      BinBase * aAuxiliaryBinsBase,
      BinBase * const aBinsBase
#ifndef NDEBUG
      , const BinBase * const aBinsBaseDebugCopy
      , const unsigned char * const pBinsEndDebug
#endif // NDEBUG
   ) {
      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      auto * const aAuxiliaryBins = aAuxiliaryBinsBase->Specialize<FloatBig, bClassification>();

      auto * const aBins = aBinsBase->Specialize<FloatBig, bClassification>();

#ifndef NDEBUG
      auto * const aBinsDebugCopy = aBinsBaseDebugCopy->Specialize<FloatBig, bClassification>();
#endif // NDEBUG

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses()
      );

      const size_t cScores = GetCountScores(learningTypeOrCountTargetClasses);
      const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

      auto * const pTotals00 = IndexBin(cBytesPerBin, aAuxiliaryBins, 0);
      auto * const pTotals01 = IndexBin(cBytesPerBin, aAuxiliaryBins, 1);
      auto * const pTotals10 = IndexBin(cBytesPerBin, aAuxiliaryBins, 2);
      auto * const pTotals11 = IndexBin(cBytesPerBin, aAuxiliaryBins, 3);

      // for interactions we return an interaction score of 0 if any of the dimensions are useless
      EBM_ASSERT(2 == pTerm->GetCountDimensions());
      EBM_ASSERT(2 == pTerm->GetCountSignificantDimensions());

      // we return an interaction score of 0 if any features are useless before calling here
      EBM_ASSERT(pTerm->GetCountDimensions() == pTerm->GetCountSignificantDimensions());
      const size_t cBinsDimension1 = pTerm->GetTermEntries()[0].m_pFeature->GetCountBins();
      const size_t cBinsDimension2 = pTerm->GetTermEntries()[1].m_pFeature->GetCountBins();

      // any pair with a feature with 1 cBins returns an interaction score of 0
      EBM_ASSERT(2 <= cBinsDimension1);
      EBM_ASSERT(2 <= cBinsDimension2);

      EBM_ASSERT(0 < cSamplesRequiredForChildSplitMin);

#ifndef NDEBUG
      bool bAnySplits = false;
#endif // NDEBUG

      // if a negative value were to occur, then it would be due to numeric instability, so clip it to zero here
      FloatBig bestGain = 0;

      size_t aiStart[k_cDimensionsMax];

      EBM_ASSERT(1 < cBinsDimension1);
      size_t iBin1 = 0;
      do {
         aiStart[0] = iBin1;
         EBM_ASSERT(1 < cBinsDimension2);
         size_t iBin2 = 0;
         do {
            aiStart[1] = iBin2;

            EBM_ASSERT(2 == pTerm->GetCountSignificantDimensions()); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
            TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, 2>(
               learningTypeOrCountTargetClasses,
               pTerm,
               aBins,
               aiStart,
               0x00,
               pTotals00
#ifndef NDEBUG
               , aBinsDebugCopy
               , pBinsEndDebug
#endif // NDEBUG
               );
            if(LIKELY(cSamplesRequiredForChildSplitMin <= pTotals00->GetCountSamples())) {
               EBM_ASSERT(2 == pTerm->GetCountSignificantDimensions()); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
               TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, 2>(
                  learningTypeOrCountTargetClasses,
                  pTerm,
                  aBins,
                  aiStart,
                  0x01,
                  pTotals01
#ifndef NDEBUG
                  , aBinsDebugCopy
                  , pBinsEndDebug
#endif // NDEBUG
                  );
               if(LIKELY(cSamplesRequiredForChildSplitMin <= pTotals01->GetCountSamples())) {
                  EBM_ASSERT(2 == pTerm->GetCountSignificantDimensions()); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
                  TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, 2>(
                     learningTypeOrCountTargetClasses,
                     pTerm,
                     aBins,
                     aiStart,
                     0x02,
                     pTotals10
#ifndef NDEBUG
                     , aBinsDebugCopy
                     , pBinsEndDebug
#endif // NDEBUG
                     );
                  if(LIKELY(cSamplesRequiredForChildSplitMin <= pTotals10->GetCountSamples())) {
                     EBM_ASSERT(2 == pTerm->GetCountSignificantDimensions()); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
                     TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, 2>(
                        learningTypeOrCountTargetClasses,
                        pTerm,
                        aBins,
                        aiStart,
                        0x03,
                        pTotals11
#ifndef NDEBUG
                        , aBinsDebugCopy
                        , pBinsEndDebug
#endif // NDEBUG
                        );
                     if(LIKELY(cSamplesRequiredForChildSplitMin <= pTotals11->GetCountSamples())) {
#ifndef NDEBUG
                        bAnySplits = true;
#endif // NDEBUG
                        FloatBig gain = 0;

                        FloatBig weight00 = pTotals00->GetWeight();
                        FloatBig weight01 = pTotals01->GetWeight();
                        FloatBig weight10 = pTotals10->GetWeight();
                        FloatBig weight11 = pTotals11->GetWeight();

                        auto * const pHistogramEntry00 = pTotals00->GetHistogramTargetEntry();
                        auto * const pHistogramEntry01 = pTotals01->GetHistogramTargetEntry();
                        auto * const pHistogramEntry10 = pTotals10->GetHistogramTargetEntry();
                        auto * const pHistogramEntry11 = pTotals11->GetHistogramTargetEntry();

                        for(size_t iScore = 0; iScore < cScores; ++iScore) {
                           // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators 
                           // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

                           constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;

                           // n = numerator (sum_gradients), d = denominator (sum_hessians or weight)

                           const FloatBig n00 = pHistogramEntry00[iScore].m_sumGradients;
                           const FloatBig d00 = bUseLogitBoost ?
                              pHistogramEntry00[iScore].GetSumHessians() : weight00;

                           const FloatBig n01 = pHistogramEntry01[iScore].m_sumGradients;
                           const FloatBig d01 = bUseLogitBoost ?
                              pHistogramEntry01[iScore].GetSumHessians() : weight01;

                           const FloatBig n10 = pHistogramEntry10[iScore].m_sumGradients;
                           const FloatBig d10 = bUseLogitBoost ?
                              pHistogramEntry10[iScore].GetSumHessians() : weight10;

                           const FloatBig n11 = pHistogramEntry11[iScore].m_sumGradients;
                           const FloatBig d11 = bUseLogitBoost ?
                              pHistogramEntry11[iScore].GetSumHessians() : weight11;

                           if(0 != (InteractionOptions_Pure & options)) {
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
            ++iBin2;
         } while(iBin2 < cBinsDimension2 - 1);
         ++iBin1;
      } while(iBin1 < cBinsDimension1 - 1);

      // we start from zero, so bestGain can't be negative here
      EBM_ASSERT(std::isnan(bestGain) || 0 <= bestGain);

      if(0 == (InteractionOptions_Pure & options)) {
         // if we are detecting impure interaction then so far we have only calculated the children partial gain 
         // but we still need to subtract the partial gain of the parent to have
         // gain. All the splits we've analyzed so far though had the same non-split partial gain, so we subtract it here
         // instead of inside the loop.

         // the bin before the aAuxiliaryBinsBase is the last summation bin of aBinsBase, 
         // which contains the totals of all bins
         const auto * const pTotal =
            reinterpret_cast<const Bin<FloatBig, bClassification> *>(
               reinterpret_cast<const char *>(aAuxiliaryBinsBase) - cBytesPerBin);

         const FloatBig weightAll = pTotal->GetWeight();

         const auto * const pHistogramEntryTotal = pTotal->GetHistogramTargetEntry();

         for(size_t iScore = 0; iScore < cScores; ++iScore) {
            // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators 
            // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

            constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;
            bestGain -= EbmStats::CalcPartialGain(
               pHistogramEntryTotal[iScore].m_sumGradients,
               bUseLogitBoost ? pHistogramEntryTotal[iScore].GetSumHessians() : weightAll
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

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class PartitionTwoDimensionalInteractionTarget final {
public:

   PartitionTwoDimensionalInteractionTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static double Func(
      InteractionCore * const pInteractionCore,
      const Term * const pTerm,
      const InteractionOptionsType options,
      const size_t cSamplesRequiredForChildSplitMin,
      BinBase * aAuxiliaryBinsBase,
      BinBase * const aBinsBase
#ifndef NDEBUG
      , const BinBase * const aBinsBaseDebugCopy
      , const unsigned char * const pBinsEndDebug
#endif // NDEBUG
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         return PartitionTwoDimensionalInteractionInternal<compilerLearningTypeOrCountTargetClassesPossible>::Func(
            pInteractionCore,
            pTerm,
            options,
            cSamplesRequiredForChildSplitMin,
            aAuxiliaryBinsBase,
            aBinsBase
#ifndef NDEBUG
            , aBinsBaseDebugCopy
            , pBinsEndDebug
#endif // NDEBUG
         );
      } else {
         return PartitionTwoDimensionalInteractionTarget<compilerLearningTypeOrCountTargetClassesPossible + 1>::Func(
            pInteractionCore,
            pTerm,
            options,
            cSamplesRequiredForChildSplitMin,
            aAuxiliaryBinsBase,
            aBinsBase
#ifndef NDEBUG
            , aBinsBaseDebugCopy
            , pBinsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<>
class PartitionTwoDimensionalInteractionTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   PartitionTwoDimensionalInteractionTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static double Func(
      InteractionCore * const pInteractionCore,
      const Term * const pTerm,
      const InteractionOptionsType options,
      const size_t cSamplesRequiredForChildSplitMin,
      BinBase * aAuxiliaryBinsBase,
      BinBase * const aBinsBase
#ifndef NDEBUG
      , const BinBase * const aBinsBaseDebugCopy
      , const unsigned char * const pBinsEndDebug
#endif // NDEBUG
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses());

      return PartitionTwoDimensionalInteractionInternal<k_dynamicClassification>::Func(
         pInteractionCore,
         pTerm,
         options,
         cSamplesRequiredForChildSplitMin,
         aAuxiliaryBinsBase,
         aBinsBase
#ifndef NDEBUG
         , aBinsBaseDebugCopy
         , pBinsEndDebug
#endif // NDEBUG
      );
   }
};

extern double PartitionTwoDimensionalInteraction(
   InteractionCore * const pInteractionCore,
   const Term * const pTerm,
   const InteractionOptionsType options,
   const size_t cSamplesRequiredForChildSplitMin,
   BinBase * aAuxiliaryBinsBase,
   BinBase * const aBinsBase
#ifndef NDEBUG
   , const BinBase * const aBinsBaseDebugCopy
   , const unsigned char * const pBinsEndDebug
#endif // NDEBUG
) {
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses();

   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      return PartitionTwoDimensionalInteractionTarget<2>::Func(
         pInteractionCore,
         pTerm,
         options,
         cSamplesRequiredForChildSplitMin,
         aAuxiliaryBinsBase,
         aBinsBase
#ifndef NDEBUG
         , aBinsBaseDebugCopy
         , pBinsEndDebug
#endif // NDEBUG
      );
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      return PartitionTwoDimensionalInteractionInternal<k_regression>::Func(
         pInteractionCore,
         pTerm,
         options,
         cSamplesRequiredForChildSplitMin,
         aAuxiliaryBinsBase,
         aBinsBase
#ifndef NDEBUG
         , aBinsBaseDebugCopy
         , pBinsEndDebug
#endif // NDEBUG
      );
   }
}

} // DEFINED_ZONE_NAME
