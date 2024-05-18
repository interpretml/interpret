// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h"
#include "unzoned.h" // LIKELY

#define ZONE_main
#include "zones.h"

#include "GradientPair.hpp"
#include "Bin.hpp"

#include "ebm_internal.hpp"
#include "ebm_stats.hpp"
#include "TensorTotalsSum.hpp"
#include "InteractionCore.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<bool bHessian, size_t cCompilerScores> class PartitionTwoDimensionalInteractionInternal final {
 public:
   PartitionTwoDimensionalInteractionInternal() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(InteractionCore* const pInteractionCore,
         const size_t cRuntimeRealDimensions,
         const size_t* const acBins,
         const CalcInteractionFlags flags,
         const size_t cSamplesLeafMin,
         const double hessianMin,
         BinBase* const aAuxiliaryBinsBase,
         BinBase* const aBinsBase
#ifndef NDEBUG
         ,
         const BinBase* const aDebugCopyBinsBase,
         const BinBase* const pBinsEndDebug
#endif // NDEBUG
   ) {
      static constexpr size_t cCompilerDimensions = 2;

      auto* const aAuxiliaryBins =
            aAuxiliaryBinsBase
                  ->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();
      auto* const aBins =
            aBinsBase->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();

#ifndef NDEBUG
      auto* const aDebugCopyBins =
            aDebugCopyBinsBase
                  ->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();
#endif // NDEBUG

      const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pInteractionCore->GetCountScores());
      const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);

      const size_t cRealDimensions = GET_COUNT_DIMENSIONS(cCompilerDimensions, cRuntimeRealDimensions);
      EBM_ASSERT(k_dynamicDimensions == cCompilerDimensions || cCompilerDimensions == cRuntimeRealDimensions);

      TensorSumDimension
            aDimensions[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
      size_t iDimensionInit = 0;
      do {
         // move data to a local variable that the compiler can reason about and then eliminate by moving to CPU
         // registers
         aDimensions[iDimensionInit].m_cBins = acBins[iDimensionInit];
         ++iDimensionInit;
      } while(cRealDimensions != iDimensionInit);

      auto* const p_DO_NOT_USE_DIRECTLY_00 = IndexBin(aAuxiliaryBins, cBytesPerBin * 0);
      ASSERT_BIN_OK(cBytesPerBin, p_DO_NOT_USE_DIRECTLY_00, pBinsEndDebug);
      auto* const p_DO_NOT_USE_DIRECTLY_01 = IndexBin(aAuxiliaryBins, cBytesPerBin * 1);
      ASSERT_BIN_OK(cBytesPerBin, p_DO_NOT_USE_DIRECTLY_01, pBinsEndDebug);
      auto* const p_DO_NOT_USE_DIRECTLY_10 = IndexBin(aAuxiliaryBins, cBytesPerBin * 2);
      ASSERT_BIN_OK(cBytesPerBin, p_DO_NOT_USE_DIRECTLY_10, pBinsEndDebug);
      auto* const p_DO_NOT_USE_DIRECTLY_11 = IndexBin(aAuxiliaryBins, cBytesPerBin * 3);
      ASSERT_BIN_OK(cBytesPerBin, p_DO_NOT_USE_DIRECTLY_11, pBinsEndDebug);

      Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)> bin00;
      Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)> bin01;
      Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)> bin10;
      Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)> bin11;

      // if we know how many scores there are, use the memory on the stack where the compiler can optimize access
      static constexpr bool bUseStackMemory = k_dynamicScores != cCompilerScores;
      auto* const aGradientPairs00 =
            bUseStackMemory ? bin00.GetGradientPairs() : p_DO_NOT_USE_DIRECTLY_00->GetGradientPairs();
      auto* const aGradientPairs01 =
            bUseStackMemory ? bin01.GetGradientPairs() : p_DO_NOT_USE_DIRECTLY_01->GetGradientPairs();
      auto* const aGradientPairs10 =
            bUseStackMemory ? bin10.GetGradientPairs() : p_DO_NOT_USE_DIRECTLY_10->GetGradientPairs();
      auto* const aGradientPairs11 =
            bUseStackMemory ? bin11.GetGradientPairs() : p_DO_NOT_USE_DIRECTLY_11->GetGradientPairs();

      EBM_ASSERT(0.0 < hessianMin);

#ifndef NDEBUG
      bool bAnySplits = false;
#endif // NDEBUG

      const bool bUseLogitBoost = bHessian && !(CalcInteractionFlags_DisableNewton & flags);

      // if a negative value were to occur, then it would be due to numeric instability, so clip it to zero here
      FloatCalc bestGain = 0;

      EBM_ASSERT(2 == cRealDimensions); // our TensorTotalsSum needs to be templated as dynamic if we want to have
                                        // something other than 2 dimensions

      EBM_ASSERT(2 <= aDimensions[0].m_cBins); // 1 cBins in any dimension returns an interaction score of 0
      aDimensions[0].m_iPoint = 0;
      do {
         EBM_ASSERT(2 <= aDimensions[1].m_cBins); // 1 cBins in any dimension returns an interaction score of 0
         aDimensions[1].m_iPoint = 0;
         do {
            TensorTotalsSum<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
                  cRealDimensions,
                  aDimensions,
                  0x00,
                  aBins,
                  bin00,
                  aGradientPairs00
#ifndef NDEBUG
                  ,
                  aDebugCopyBins,
                  pBinsEndDebug
#endif // NDEBUG
            );
            if(bin00.GetCountSamples() < cSamplesLeafMin) {
               goto next;
            }

            TensorTotalsSum<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
                  cRealDimensions,
                  aDimensions,
                  0x01,
                  aBins,
                  bin01,
                  aGradientPairs01
#ifndef NDEBUG
                  ,
                  aDebugCopyBins,
                  pBinsEndDebug
#endif // NDEBUG
            );
            if(bin01.GetCountSamples() < cSamplesLeafMin) {
               goto next;
            }

            TensorTotalsSum<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
                  cRealDimensions,
                  aDimensions,
                  0x02,
                  aBins,
                  bin10,
                  aGradientPairs10
#ifndef NDEBUG
                  ,
                  aDebugCopyBins,
                  pBinsEndDebug
#endif // NDEBUG
            );
            if(bin10.GetCountSamples() < cSamplesLeafMin) {
               goto next;
            }

            TensorTotalsSum<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
                  cRealDimensions,
                  aDimensions,
                  0x03,
                  aBins,
                  bin11,
                  aGradientPairs11
#ifndef NDEBUG
                  ,
                  aDebugCopyBins,
                  pBinsEndDebug
#endif // NDEBUG
            );
            if(bin11.GetCountSamples() < cSamplesLeafMin) {
               goto next;
            }

            {
#ifndef NDEBUG
               bAnySplits = true;
#endif // NDEBUG
               FloatCalc gain = 0;
               for(size_t iScore = 0; iScore < cScores; ++iScore) {
                  // TODO : we can make this faster by doing the division in CalcPartialGain after we add all
                  // the numerators (but only do this after we've determined the best node splitting score for
                  // classification, and the NewtonRaphsonStep for gain

                  // n = numerator (sum_gradients), d = denominator (sum_hessians or weight)

                  FloatCalc hessian00;
                  if(bHessian) {
                     hessian00 = aGradientPairs00[iScore].GetHess();
                  } else {
                     hessian00 = bin00.GetWeight();
                  }
                  if(hessian00 < hessianMin) {
                     goto next;
                  }

                  FloatCalc hessian01;
                  if(bHessian) {
                     hessian01 = aGradientPairs01[iScore].GetHess();
                  } else {
                     hessian01 = bin01.GetWeight();
                  }
                  if(hessian01 < hessianMin) {
                     goto next;
                  }

                  FloatCalc hessian10;
                  if(bHessian) {
                     hessian10 = aGradientPairs10[iScore].GetHess();
                  } else {
                     hessian10 = bin10.GetWeight();
                  }
                  if(hessian10 < hessianMin) {
                     goto next;
                  }

                  FloatCalc hessian11;
                  if(bHessian) {
                     hessian11 = aGradientPairs11[iScore].GetHess();
                  } else {
                     hessian11 = bin11.GetWeight();
                  }
                  if(hessian11 < hessianMin) {
                     goto next;
                  }

                  const FloatCalc n00 = static_cast<FloatCalc>(aGradientPairs00[iScore].m_sumGradients);
                  const FloatCalc d00 =
                        static_cast<FloatCalc>(bUseLogitBoost ? aGradientPairs00[iScore].GetHess() : bin00.GetWeight());

                  const FloatCalc n01 = static_cast<FloatCalc>(aGradientPairs01[iScore].m_sumGradients);
                  const FloatCalc d01 =
                        static_cast<FloatCalc>(bUseLogitBoost ? aGradientPairs01[iScore].GetHess() : bin01.GetWeight());

                  const FloatCalc n10 = static_cast<FloatCalc>(aGradientPairs10[iScore].m_sumGradients);
                  const FloatCalc d10 =
                        static_cast<FloatCalc>(bUseLogitBoost ? aGradientPairs10[iScore].GetHess() : bin10.GetWeight());

                  const FloatCalc n11 = static_cast<FloatCalc>(aGradientPairs11[iScore].m_sumGradients);
                  const FloatCalc d11 =
                        static_cast<FloatCalc>(bUseLogitBoost ? aGradientPairs11[iScore].GetHess() : bin11.GetWeight());

                  if(CalcInteractionFlags_Pure & flags) {
                     // purified gain

                     // TODO: The interaction score is exactly equivalent to the gain calculated during
                     // boosting, so we can simplify our codebase by eliminating the interaction detection
                     // code if we generalize the boosting code to accept multiple term indexes.
                     // This change would have the additional benefit that we'd be able to use
                     // the more complex splits that we currently handle for boosting during interaction detection for
                     // no additional complexity, and we'll be able to benefit from the even more complex spits that
                     // we'll eventuall support for boosting interactions (allowing more than one cut in each of the
                     // interaction dimensions)
                     //
                     // TODO: We are purififying the simple 2x2 solution below using a simple system of equations
                     // but the solution below can be generalized to handle any size matrix and/or any size
                     // of tensor for 3-way and higher interactions.  The system of equations below were solved 
                     // using the substitution/elimination method, but to solve these in the general case we'll 
                     // need to implement a system of equations solver.  First try something like the matrix or
                     // inverse matrix method, and if that fails use an iterative solution like the
                     // Jacobi or Gauss-Seidel methods. This would be a better solution than the iterative
                     // solution that we currently use in the python purify() function.
                     // 
                     // TODO: Once more efficient purification is done, we can use the same purification
                     // method during boosting where we could then keep the interactions pure while we 
                     // simultaneously boost mains and interactions togehter at the same time.  This would 
                     // be desirable in order to keep from overboosting on mains that are also included 
                     // within interactions.
                     // 
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
                     if(FloatCalc{0} != d00 && FloatCalc{0} != d01 && FloatCalc{0} != d10 && FloatCalc{0} != d11) {

                        // TODO: instead of checking the denominators for zero above, can we do it earlier?
                        // If we're using hessians then we'd need it here, but we aren't using them yet

                        // calculate what the full updates would be for non-purified:
                        // u = update (non-purified)
                        const FloatCalc u00 = n00 / d00;
                        const FloatCalc u01 = n01 / d01;
                        const FloatCalc u10 = n10 / d10;
                        const FloatCalc u11 = n11 / d11;

                        // common part of equations (positive for 00 & 11 equations, negative for 01 and 10)
                        const FloatCalc common = u00 - u01 - u10 + u11;

                        // p = purified update
                        const FloatCalc p00 = common / (FloatCalc{1} + d00 / d01 + d00 / d10 + d00 / d11);
                        const FloatCalc p01 = common / (FloatCalc{-1} - d01 / d00 - d01 / d10 - d01 / d11);
                        const FloatCalc p10 = common / (FloatCalc{-1} - d10 / d00 - d10 / d01 - d10 / d11);
                        const FloatCalc p11 = common / (FloatCalc{1} + d11 / d00 + d11 / d01 + d11 / d10);

                        // g = gain
                        const FloatCalc g00 = CalcPartialGainFromUpdate(p00, d00);
                        const FloatCalc g01 = CalcPartialGainFromUpdate(p01, d01);
                        const FloatCalc g10 = CalcPartialGainFromUpdate(p10, d10);
                        const FloatCalc g11 = CalcPartialGainFromUpdate(p11, d11);

#ifndef NDEBUG
                        // r = reconsituted numerator (after purification)
                        const FloatCalc r00 = p00 * d00;
                        const FloatCalc r01 = p01 * d01;
                        const FloatCalc r10 = p10 * d10;
                        const FloatCalc r11 = p11 * d11;

                        // purification means summing any direction gives us zero
                        EBM_ASSERT(std::abs(r00 + r01) < 0.001);
                        EBM_ASSERT(std::abs(r01 + r11) < 0.001);
                        EBM_ASSERT(std::abs(r11 + r10) < 0.001);
                        EBM_ASSERT(std::abs(r10 + r00) < 0.001);

                        // if all of these added together are zero, then the parent partial gain should also
                        // be zero, which means we can avoid calculating the parent partial gain.
                        EBM_ASSERT(std::abs(r00 + r01 + r10 + r11) < 0.001);

                        EBM_ASSERT(std::abs(g00 - CalcPartialGain(r00, d00)) < 0.001);
                        EBM_ASSERT(std::abs(g01 - CalcPartialGain(r01, d01)) < 0.001);
                        EBM_ASSERT(std::abs(g10 - CalcPartialGain(r10, d10)) < 0.001);
                        EBM_ASSERT(std::abs(g11 - CalcPartialGain(r11, d11)) < 0.001);
#endif // NDEBUG
                        gain += g00;
                        gain += g01;
                        gain += g10;
                        gain += g11;
                     }
                  } else {
                     // non-purified gain
                     gain += CalcPartialGain(n00, d00);
                     gain += CalcPartialGain(n01, d01);
                     gain += CalcPartialGain(n10, d10);
                     gain += CalcPartialGain(n11, d11);
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

         next:;

            ++aDimensions[1].m_iPoint;
         } while(aDimensions[1].m_cBins - 1 != aDimensions[1].m_iPoint);
         ++aDimensions[0].m_iPoint;
      } while(aDimensions[0].m_cBins - 1 != aDimensions[0].m_iPoint);

      // we start from zero, so bestGain can't be negative here
      EBM_ASSERT(std::isnan(bestGain) || 0 <= bestGain);

      if(!(CalcInteractionFlags_Pure & flags)) {
         // if we are detecting impure interaction then so far we have only calculated the children partial gain
         // but we still need to subtract the partial gain of the parent to have
         // gain. All the splits we've analyzed so far though had the same non-split partial gain, so we subtract it
         // here instead of inside the loop.

         // the bin before the aAuxiliaryBins is the last summation bin of aBinsBase,
         // which contains the totals of all bins
         const auto* const pTotal = NegativeIndexBin(aAuxiliaryBins, cBytesPerBin);
         const FloatMain weightAll = pTotal->GetWeight();
         const auto* const aGradientPairs = pTotal->GetGradientPairs();
         for(size_t iScore = 0; iScore < cScores; ++iScore) {
            // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators
            // (but only do this after we've determined the best node splitting score for classification, and the
            // NewtonRaphsonStep for gain

            bestGain -= CalcPartialGain(static_cast<FloatCalc>(aGradientPairs[iScore].m_sumGradients),
                  static_cast<FloatCalc>(bUseLogitBoost ? aGradientPairs[iScore].GetHess() : weightAll));
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

         EBM_ASSERT(std::isnan(bestGain) || -std::numeric_limits<FloatCalc>::infinity() == bestGain ||
               k_epsilonNegativeGainAllowed <= bestGain || !bAnySplits);
      }

      // we clean up bestGain in the caller, since this function is templated and created many times
      return static_cast<double>(bestGain);
   }
};

template<bool bHessian, size_t cPossibleScores> class PartitionTwoDimensionalInteractionTarget final {
 public:
   PartitionTwoDimensionalInteractionTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(InteractionCore* const pInteractionCore,
         const size_t cRealDimensions,
         const size_t* const acBins,
         const CalcInteractionFlags flags,
         const size_t cSamplesLeafMin,
         const double hessianMin,
         BinBase* aAuxiliaryBinsBase,
         BinBase* const aBinsBase
#ifndef NDEBUG
         ,
         const BinBase* const aDebugCopyBinsBase,
         const BinBase* const pBinsEndDebug
#endif // NDEBUG
   ) {
      if(cPossibleScores == pInteractionCore->GetCountScores()) {
         return PartitionTwoDimensionalInteractionInternal<bHessian, cPossibleScores>::Func(pInteractionCore,
               cRealDimensions,
               acBins,
               flags,
               cSamplesLeafMin,
               hessianMin,
               aAuxiliaryBinsBase,
               aBinsBase
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase,
               pBinsEndDebug
#endif // NDEBUG
         );
      } else {
         return PartitionTwoDimensionalInteractionTarget<bHessian, cPossibleScores + 1>::Func(pInteractionCore,
               cRealDimensions,
               acBins,
               flags,
               cSamplesLeafMin,
               hessianMin,
               aAuxiliaryBinsBase,
               aBinsBase
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase,
               pBinsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<bool bHessian> class PartitionTwoDimensionalInteractionTarget<bHessian, k_cCompilerScoresMax + 1> final {
 public:
   PartitionTwoDimensionalInteractionTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(InteractionCore* const pInteractionCore,
         const size_t cRealDimensions,
         const size_t* const acBins,
         const CalcInteractionFlags flags,
         const size_t cSamplesLeafMin,
         const double hessianMin,
         BinBase* aAuxiliaryBinsBase,
         BinBase* const aBinsBase
#ifndef NDEBUG
         ,
         const BinBase* const aDebugCopyBinsBase,
         const BinBase* const pBinsEndDebug
#endif // NDEBUG
   ) {
      return PartitionTwoDimensionalInteractionInternal<bHessian, k_dynamicScores>::Func(pInteractionCore,
            cRealDimensions,
            acBins,
            flags,
            cSamplesLeafMin,
            hessianMin,
            aAuxiliaryBinsBase,
            aBinsBase
#ifndef NDEBUG
            ,
            aDebugCopyBinsBase,
            pBinsEndDebug
#endif // NDEBUG
      );
   }
};

extern double PartitionTwoDimensionalInteraction(InteractionCore* const pInteractionCore,
      const size_t cRealDimensions,
      const size_t* const acBins,
      const CalcInteractionFlags flags,
      const size_t cSamplesLeafMin,
      const double hessianMin,
      BinBase* aAuxiliaryBinsBase,
      BinBase* const aBinsBase
#ifndef NDEBUG
      ,
      const BinBase* const aDebugCopyBinsBase,
      const BinBase* const pBinsEndDebug
#endif // NDEBUG
) {
   const size_t cRuntimeScores = pInteractionCore->GetCountScores();

   EBM_ASSERT(1 <= cRuntimeScores);
   if(pInteractionCore->IsHessian()) {
      if(size_t{1} != cRuntimeScores) {
         // muticlass
         return PartitionTwoDimensionalInteractionTarget<true, k_cCompilerScoresStart>::Func(pInteractionCore,
               cRealDimensions,
               acBins,
               flags,
               cSamplesLeafMin,
               hessianMin,
               aAuxiliaryBinsBase,
               aBinsBase
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase,
               pBinsEndDebug
#endif // NDEBUG
         );
      } else {
         return PartitionTwoDimensionalInteractionInternal<true, k_oneScore>::Func(pInteractionCore,
               cRealDimensions,
               acBins,
               flags,
               cSamplesLeafMin,
               hessianMin,
               aAuxiliaryBinsBase,
               aBinsBase
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase,
               pBinsEndDebug
#endif // NDEBUG
         );
      }
   } else {
      if(size_t{1} != cRuntimeScores) {
         // Odd: gradient multiclass. Allow it, but do not optimize for it
         return PartitionTwoDimensionalInteractionInternal<false, k_dynamicScores>::Func(pInteractionCore,
               cRealDimensions,
               acBins,
               flags,
               cSamplesLeafMin,
               hessianMin,
               aAuxiliaryBinsBase,
               aBinsBase
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase,
               pBinsEndDebug
#endif // NDEBUG
         );
      } else {
         return PartitionTwoDimensionalInteractionInternal<false, k_oneScore>::Func(pInteractionCore,
               cRealDimensions,
               acBins,
               flags,
               cSamplesLeafMin,
               hessianMin,
               aAuxiliaryBinsBase,
               aBinsBase
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase,
               pBinsEndDebug
#endif // NDEBUG
         );
      }
   }
}

} // namespace DEFINED_ZONE_NAME
