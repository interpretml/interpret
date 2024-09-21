// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "unzoned.h" // LIKELY

#define ZONE_main
#include "zones.h"

#include "GradientPair.hpp"
#include "Bin.hpp"

#include "ebm_stats.hpp"
#include "Feature.hpp"
#include "Term.hpp"
#include "Tensor.hpp"
#include "TensorTotalsSum.hpp"
#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<bool bHessian, size_t cCompilerScores, size_t cCompilerDimensions>
static FloatCalc SweepMultiDimensional(const size_t cRuntimeScores,
      const size_t cRuntimeRealDimensions,
      const TermBoostFlags flags,
      const size_t* const aiPoint,
      const size_t* const acBins,
      const size_t directionVectorLow,
      const size_t iDimensionSweep,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const aBins,
      const size_t cSamplesLeafMin,
      const FloatCalc hessianMin,
      Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const pBinBestAndTemp,
      size_t* const piBestSplit
#ifndef NDEBUG
      ,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const aDebugCopyBins,
      const BinBase* const pBinsEndDebug
#endif // NDEBUG
) {
   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, cRuntimeScores);
   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);

   const size_t cRealDimensions = GET_COUNT_DIMENSIONS(cCompilerDimensions, cRuntimeRealDimensions);
   EBM_ASSERT(1 <= cRealDimensions); // for interactions, we just return 0 for interactions with zero features
   EBM_ASSERT(iDimensionSweep < cRealDimensions);
   EBM_ASSERT(0 == (directionVectorLow & (size_t{1} << iDimensionSweep)));

   TensorSumDimension aDimensions[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
   size_t directionDestroy = directionVectorLow;
   size_t iDimensionInit = 0;
   do {
      aDimensions[iDimensionInit].m_cBins = acBins[iDimensionInit];
      if(0 != (size_t{1} & directionDestroy)) {
         aDimensions[iDimensionInit].m_iLow = aiPoint[iDimensionInit] + 1;
         aDimensions[iDimensionInit].m_iHigh = acBins[iDimensionInit];
      } else {
         aDimensions[iDimensionInit].m_iLow = 0;
         aDimensions[iDimensionInit].m_iHigh = aiPoint[iDimensionInit] + 1;
      }
      directionDestroy >>= 1;
      ++iDimensionInit;
   } while(cRealDimensions != iDimensionInit);

   const size_t cSweepCuts = aDimensions[iDimensionSweep].m_cBins - 1;
   EBM_ASSERT(1 <= cSweepCuts); // dimensions with 1 bin are removed earlier

   size_t iBestSplit = 0;

   auto* const p_DO_NOT_USE_DIRECTLY_Low = IndexBin(pBinBestAndTemp, cBytesPerBin * 2);
   ASSERT_BIN_OK(cBytesPerBin, p_DO_NOT_USE_DIRECTLY_Low, pBinsEndDebug);
   auto* const p_DO_NOT_USE_DIRECTLY_High = IndexBin(pBinBestAndTemp, cBytesPerBin * 3);
   ASSERT_BIN_OK(cBytesPerBin, p_DO_NOT_USE_DIRECTLY_High, pBinsEndDebug);

   Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)> binLow;
   Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)> binHigh;

   // if we know how many scores there are, use the memory on the stack where the compiler can optimize access
   static constexpr bool bUseStackMemory = k_dynamicScores != cCompilerScores;
   auto* const aGradientPairsLow =
         bUseStackMemory ? binLow.GetGradientPairs() : p_DO_NOT_USE_DIRECTLY_Low->GetGradientPairs();
   auto* const aGradientPairsHigh =
         bUseStackMemory ? binHigh.GetGradientPairs() : p_DO_NOT_USE_DIRECTLY_High->GetGradientPairs();

   EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= hessianMin);

   const bool bUseLogitBoost = bHessian && !(TermBoostFlags_DisableNewtonGain & flags);

   // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
   EBM_ASSERT(2 == cRealDimensions);

   FloatCalc bestGain = k_illegalGainFloat;
   size_t iBin = 0;
   do {
      aDimensions[iDimensionSweep].m_iLow = 0;
      aDimensions[iDimensionSweep].m_iHigh = iBin + 1;
      TensorTotalsSum<bHessian, cCompilerScores, cCompilerDimensions>(cRuntimeScores,
            cRealDimensions,
            aDimensions,
            aBins,
            binLow,
            aGradientPairsLow
#ifndef NDEBUG
            ,
            aDebugCopyBins,
            pBinsEndDebug
#endif // NDEBUG
      );
      if(binLow.GetCountSamples() < cSamplesLeafMin) {
         goto next;
      }

      aDimensions[iDimensionSweep].m_iLow = iBin + 1;
      aDimensions[iDimensionSweep].m_iHigh = aDimensions[iDimensionSweep].m_cBins;
      TensorTotalsSum<bHessian, cCompilerScores, cCompilerDimensions>(cRuntimeScores,
            cRealDimensions,
            aDimensions,
            aBins,
            binHigh,
            aGradientPairsHigh
#ifndef NDEBUG
            ,
            aDebugCopyBins,
            pBinsEndDebug
#endif // NDEBUG
      );
      if(binHigh.GetCountSamples() < cSamplesLeafMin) {
         goto next;
      }

      // if (0 != (TermBoostFlags_PurifyGain & flags)) {
      //  TODO: At this point we have the bin sums histogram for the tensor, so we can purify the future update
      //  for the cuts we're currently evaluating before calculating the gain. This should give us a more accurate gain
      //  calculation for the purified update. We need to construct the entire tensor here before purifying.
      //  We already calculate purified gain as an option during interaction detection, since the
      //  interaction metric we use is the gain calculation.
      //  See: Use of CalcInteractionFlags_Purify in PartitionTwoDimensionalInteraction.cpp
      //}

      {
         FloatCalc gain = 0;
         EBM_ASSERT(0 < binLow.GetCountSamples());
         EBM_ASSERT(0 < binHigh.GetCountSamples());

         EBM_ASSERT(1 <= cScores);
         size_t iScore = 0;
         do {
            // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the
            // numerators (but only do this after we've determined the best node splitting score for classification,
            // and the NewtonRaphsonStep for gain

            FloatCalc hessianLow;
            if(bUseLogitBoost) {
               hessianLow = static_cast<FloatCalc>(aGradientPairsLow[iScore].GetHess());
            } else {
               hessianLow = static_cast<FloatCalc>(binLow.GetWeight());
            }
            if(hessianLow < hessianMin) {
               goto next;
            }

            FloatCalc hessianHigh;
            if(bUseLogitBoost) {
               hessianHigh = static_cast<FloatCalc>(aGradientPairsHigh[iScore].GetHess());
            } else {
               hessianHigh = static_cast<FloatCalc>(binHigh.GetWeight());
            }
            if(hessianHigh < hessianMin) {
               goto next;
            }

            const FloatCalc gain1 = CalcPartialGain(static_cast<FloatCalc>(aGradientPairsLow[iScore].m_sumGradients), hessianLow);
            EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);
            gain += gain1;

            const FloatCalc gain2 = CalcPartialGain(static_cast<FloatCalc>(aGradientPairsHigh[iScore].m_sumGradients), hessianHigh);
            EBM_ASSERT(std::isnan(gain2) || 0 <= gain2);
            gain += gain2;

            ++iScore;
         } while(cScores != iScore);
         EBM_ASSERT(std::isnan(gain) || 0 <= gain); // sumation of positive numbers should be positive

         if(UNLIKELY(/* NaN */ !LIKELY(gain <= bestGain))) {
            // propagate NaNs

            bestGain = gain;
            iBestSplit = iBin;

            auto* const pTotalsLowOut = IndexBin(pBinBestAndTemp, cBytesPerBin * 0);
            ASSERT_BIN_OK(cBytesPerBin, pTotalsLowOut, pBinsEndDebug);

            pTotalsLowOut->Copy(cScores, binLow, aGradientPairsLow);

            auto* const pTotalsHighOut = IndexBin(pBinBestAndTemp, cBytesPerBin * 1);
            ASSERT_BIN_OK(cBytesPerBin, pTotalsHighOut, pBinsEndDebug);

            pTotalsHighOut->Copy(cScores, binHigh, aGradientPairsHigh);
         } else {
            EBM_ASSERT(!std::isnan(gain));
         }
      }

   next:;

      ++iBin;
   } while(cSweepCuts != iBin);
   *piBestSplit = iBestSplit;

   EBM_ASSERT(std::isnan(bestGain) || k_illegalGainFloat == bestGain || FloatCalc{0} <= bestGain);
   return bestGain;
}

template<bool bHessian, size_t cCompilerScores> class PartitionTwoDimensionalBoostingInternal final {
 public:
   PartitionTwoDimensionalBoostingInternal() = delete; // this is a static class.  Do not construct

   WARNING_PUSH
   WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BoosterShell* const pBoosterShell,
         const TermBoostFlags flags,
         const Term* const pTerm,
         const size_t* const acBins,
         const size_t cSamplesLeafMin,
         const FloatCalc hessianMin,
         BinBase* const aAuxiliaryBinsBase,
         double* const aWeights,
         double* const pTotalGain
#ifndef NDEBUG
         ,
         const BinBase* const aDebugCopyBinsBase
#endif // NDEBUG
   ) {
      static constexpr size_t cCompilerDimensions = 2;

      ErrorEbm error;
      BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();

      auto* const aBins =
            pBoosterShell->GetBoostingMainBins()
                  ->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();
      Tensor* const pInnerTermUpdate = pBoosterShell->GetInnerTermUpdate();

      const size_t cRuntimeScores = pBoosterCore->GetCountScores();
      const size_t cScores = GET_COUNT_SCORES(cCompilerScores, cRuntimeScores);
      const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);

      auto* const aAuxiliaryBins =
            aAuxiliaryBinsBase
                  ->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();

#ifndef NDEBUG
      const auto* const aDebugCopyBins =
            aDebugCopyBinsBase
                  ->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();
#endif // NDEBUG

      size_t aiStart[k_cDimensionsMax];
      // technically this assignment to zero might not be needed, but if we left it uninitialized, then we would later
      // be copying an unitialized memory location into another unitialized memory location which the static clang
      // analysis does not like and which seems might be problematic in some compilers even though not technically
      // undefined behavior according to the standard
      // https://stackoverflow.com/questions/11962457/why-is-using-an-uninitialized-variable-undefined-behavior
      aiStart[1] = 0;

      EBM_ASSERT(2 == pTerm->GetCountRealDimensions());
      EBM_ASSERT(2 <= pTerm->GetCountDimensions());
      size_t iDimensionLoop = 0;
      size_t iDimension1 = 0;
      size_t iDimension2 = 0;
      size_t cBinsDimension1 = 0;
      size_t cBinsDimension2 = 0;
      const TermFeature* pTermFeature = pTerm->GetTermFeatures();
      const TermFeature* const pTermFeaturesEnd = &pTermFeature[pTerm->GetCountDimensions()];
      do {
         const FeatureBoosting* const pFeature = pTermFeature->m_pFeature;
         const size_t cBins = pFeature->GetCountBins();
         EBM_ASSERT(size_t{1} <= cBins); // we don't boost on empty training sets
         if(size_t{1} < cBins) {
            EBM_ASSERT(0 == cBinsDimension2);
            if(0 == cBinsDimension1) {
               iDimension1 = iDimensionLoop;
               cBinsDimension1 = cBins;
            } else {
               iDimension2 = iDimensionLoop;
               cBinsDimension2 = cBins;
            }
         }
         ++iDimensionLoop;
         ++pTermFeature;
      } while(pTermFeaturesEnd != pTermFeature);
      EBM_ASSERT(2 <= cBinsDimension1);
      EBM_ASSERT(2 <= cBinsDimension2);

      FloatCalc bestGain = k_illegalGainFloat;

      size_t splitFirst1Best;
      size_t splitFirst1LowBest;
      size_t splitFirst1HighBest;

      auto* pTotals1LowLowBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 0);
      auto* pTotals1LowHighBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 1);
      auto* pTotals1HighLowBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 2);
      auto* pTotals1HighHighBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 3);

      EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= hessianMin);

      LOG_0(Trace_Verbose, "PartitionTwoDimensionalBoostingInternal Starting FIRST bin sweep loop");
      size_t iBin1 = 0;
      do {
         aiStart[0] = iBin1;
         size_t splitSecond1LowBest;
         auto* pTotals2LowLowBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 4);
         auto* pTotals2LowHighBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 5);
         const FloatCalc gain1 = SweepMultiDimensional<bHessian, cCompilerScores, cCompilerDimensions>(cRuntimeScores,
               pTerm->GetCountRealDimensions(),
               flags,
               aiStart,
               acBins,
               0x0,
               1,
               aBins,
               cSamplesLeafMin,
               hessianMin,
               pTotals2LowLowBest,
               &splitSecond1LowBest
#ifndef NDEBUG
               ,
               aDebugCopyBins,
               pBoosterShell->GetDebugMainBinsEnd()
#endif // NDEBUG
         );

         if(LIKELY(/* NaN */ !UNLIKELY(gain1 < FloatCalc{0}))) {
            EBM_ASSERT(std::isnan(gain1) || FloatCalc{0} <= gain1);

            size_t splitSecond1HighBest;
            auto* pTotals2HighLowBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 8);
            auto* pTotals2HighHighBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 9);
            const FloatCalc gain2 =
                  SweepMultiDimensional<bHessian, cCompilerScores, cCompilerDimensions>(cRuntimeScores,
                        pTerm->GetCountRealDimensions(),
                        flags,
                        aiStart,
                        acBins,
                        0x1,
                        1,
                        aBins,
                        cSamplesLeafMin,
                        hessianMin,
                        pTotals2HighLowBest,
                        &splitSecond1HighBest
#ifndef NDEBUG
                        ,
                        aDebugCopyBins,
                        pBoosterShell->GetDebugMainBinsEnd()
#endif // NDEBUG
                  );

            if(LIKELY(/* NaN */ !UNLIKELY(gain2 < FloatCalc{0}))) {
               EBM_ASSERT(std::isnan(gain2) || FloatCalc{0} <= gain2);

               const FloatCalc gain = gain1 + gain2;
               if(UNLIKELY(/* NaN */ !LIKELY(gain <= bestGain))) {
                  // propagate NaNs

                  EBM_ASSERT(std::isnan(gain) || FloatCalc{0} <= gain);

                  bestGain = gain;
                  splitFirst1Best = iBin1;
                  splitFirst1LowBest = splitSecond1LowBest;
                  splitFirst1HighBest = splitSecond1HighBest;

                  // TODO: we can probably copy all 4 of these with a single memcpy
                  memcpy(pTotals1LowLowBest, pTotals2LowLowBest, cBytesPerBin);
                  memcpy(pTotals1LowHighBest, pTotals2LowHighBest, cBytesPerBin);
                  memcpy(pTotals1HighLowBest, pTotals2HighLowBest, cBytesPerBin);
                  memcpy(pTotals1HighHighBest, pTotals2HighHighBest, cBytesPerBin);
               } else {
                  EBM_ASSERT(!std::isnan(gain));
               }
            } else {
               EBM_ASSERT(!std::isnan(gain2));
               EBM_ASSERT(k_illegalGainFloat == gain2);
            }
         } else {
            EBM_ASSERT(!std::isnan(gain1));
            EBM_ASSERT(k_illegalGainFloat == gain1);
         }
         ++iBin1;
      } while(iBin1 < cBinsDimension1 - 1);

      bool bSplitFirst2 = false;

      size_t splitFirst2Best;
      size_t splitFirst2LowBest;
      size_t splitFirst2HighBest;

      auto* pTotals2LowLowBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 12);
      auto* pTotals2LowHighBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 13);
      auto* pTotals2HighLowBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 14);
      auto* pTotals2HighHighBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 15);

      LOG_0(Trace_Verbose, "PartitionTwoDimensionalBoostingInternal Starting SECOND bin sweep loop");
      size_t iBin2 = 0;
      do {
         aiStart[1] = iBin2;
         size_t splitSecond2LowBest;
         auto* pTotals1LowLowBestInner = IndexBin(aAuxiliaryBins, cBytesPerBin * 16);
         auto* pTotals1LowHighBestInner = IndexBin(aAuxiliaryBins, cBytesPerBin * 17);
         const FloatCalc gain1 = SweepMultiDimensional<bHessian, cCompilerScores, cCompilerDimensions>(cRuntimeScores,
               pTerm->GetCountRealDimensions(),
               flags,
               aiStart,
               acBins,
               0x0,
               0,
               aBins,
               cSamplesLeafMin,
               hessianMin,
               pTotals1LowLowBestInner,
               &splitSecond2LowBest
#ifndef NDEBUG
               ,
               aDebugCopyBins,
               pBoosterShell->GetDebugMainBinsEnd()
#endif // NDEBUG
         );

         if(LIKELY(/* NaN */ !UNLIKELY(gain1 < FloatCalc{0}))) {
            EBM_ASSERT(std::isnan(gain1) || FloatCalc{0} <= gain1);

            size_t splitSecond2HighBest;
            auto* pTotals1HighLowBestInner = IndexBin(aAuxiliaryBins, cBytesPerBin * 20);
            auto* pTotals1HighHighBestInner = IndexBin(aAuxiliaryBins, cBytesPerBin * 21);
            const FloatCalc gain2 =
                  SweepMultiDimensional<bHessian, cCompilerScores, cCompilerDimensions>(cRuntimeScores,
                        pTerm->GetCountRealDimensions(),
                        flags,
                        aiStart,
                        acBins,
                        0x2,
                        0,
                        aBins,
                        cSamplesLeafMin,
                        hessianMin,
                        pTotals1HighLowBestInner,
                        &splitSecond2HighBest
#ifndef NDEBUG
                        ,
                        aDebugCopyBins,
                        pBoosterShell->GetDebugMainBinsEnd()
#endif // NDEBUG
                  );

            if(LIKELY(/* NaN */ !UNLIKELY(gain2 < FloatCalc{0}))) {
               EBM_ASSERT(std::isnan(gain2) || FloatCalc{0} <= gain2);

               const FloatCalc gain = gain1 + gain2;
               if(UNLIKELY(/* NaN */ !LIKELY(gain <= bestGain))) {
                  // propagate NaNs

                  EBM_ASSERT(std::isnan(gain) || 0 <= gain);

                  bestGain = gain;
                  splitFirst2Best = iBin2;
                  splitFirst2LowBest = splitSecond2LowBest;
                  splitFirst2HighBest = splitSecond2HighBest;

                  // TODO: we can probably copy all 4 of these with a single memcpy
                  memcpy(pTotals2LowLowBest, pTotals1LowLowBestInner, cBytesPerBin);
                  memcpy(pTotals2LowHighBest, pTotals1LowHighBestInner, cBytesPerBin);
                  memcpy(pTotals2HighLowBest, pTotals1HighLowBestInner, cBytesPerBin);
                  memcpy(pTotals2HighHighBest, pTotals1HighHighBestInner, cBytesPerBin);

                  bSplitFirst2 = true;
               } else {
                  EBM_ASSERT(!std::isnan(gain));
               }
            } else {
               EBM_ASSERT(!std::isnan(gain2));
               EBM_ASSERT(k_illegalGainFloat == gain2);
            }
         } else {
            EBM_ASSERT(!std::isnan(gain1));
            EBM_ASSERT(k_illegalGainFloat == gain1);
         }
         ++iBin2;
      } while(iBin2 < cBinsDimension2 - 1);
      LOG_0(Trace_Verbose, "PartitionTwoDimensionalBoostingInternal Done sweep loops");

      EBM_ASSERT(std::isnan(bestGain) || k_illegalGainFloat == bestGain || FloatCalc{0} <= bestGain);

      // the bin before the aAuxiliaryBins is the last summation bin of aBinsBase,
      // which contains the totals of all bins
      const auto* const pTotal = NegativeIndexBin(aAuxiliaryBins, cBytesPerBin);

      ASSERT_BIN_OK(cBytesPerBin, pTotal, pBoosterShell->GetDebugMainBinsEnd());

      const auto* const pGradientPairTotal = pTotal->GetGradientPairs();

      const FloatMain weightAll = pTotal->GetWeight();
      EBM_ASSERT(0 < weightAll);

      const bool bUpdateWithHessian = bHessian && !(TermBoostFlags_DisableNewtonUpdate & flags);

      *pTotalGain = 0;
      EBM_ASSERT(FloatCalc{0} <= k_gainMin);
      if(LIKELY(/* NaN */ !UNLIKELY(bestGain < k_gainMin))) {
         EBM_ASSERT(std::isnan(bestGain) || 0 <= bestGain);

         // signal that we've hit an overflow.  Use +inf here since our caller likes that and will flip to -inf
         *pTotalGain = std::numeric_limits<double>::infinity();
         if(LIKELY(/* NaN */ bestGain <= std::numeric_limits<FloatCalc>::max())) {
            EBM_ASSERT(!std::isnan(bestGain));
            EBM_ASSERT(0 <= bestGain);
            EBM_ASSERT(std::numeric_limits<FloatCalc>::infinity() != bestGain);

            const bool bUseLogitBoost = bHessian && !(TermBoostFlags_DisableNewtonGain & flags);

            // now subtract the parent partial gain
            for(size_t iScore = 0; iScore < cScores; ++iScore) {
               // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the
               // numerators (but only do this after we've determined the best node splitting score for classification,
               // and the NewtonRaphsonStep for gain

               const FloatCalc hess =
                     static_cast<FloatCalc>(bUseLogitBoost ? pGradientPairTotal[iScore].GetHess() : weightAll);

               // we would not get there unless there was a legal cut, which requires that hessianMin <= hess
               EBM_ASSERT(hessianMin <= hess);

               const FloatCalc gain1 =
                     CalcPartialGain(static_cast<FloatCalc>(pGradientPairTotal[iScore].m_sumGradients), hess);
               EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);
               bestGain -= gain1;
            }

            EBM_ASSERT(std::numeric_limits<FloatCalc>::infinity() != bestGain);
            EBM_ASSERT(std::isnan(bestGain) || -std::numeric_limits<FloatCalc>::infinity() == bestGain ||
                  k_epsilonNegativeGainAllowed <= bestGain);

            if(LIKELY(/* NaN */ std::numeric_limits<FloatCalc>::lowest() <= bestGain)) {
               EBM_ASSERT(!std::isnan(bestGain));
               EBM_ASSERT(!std::isinf(bestGain));
               EBM_ASSERT(k_epsilonNegativeGainAllowed <= bestGain);

               *pTotalGain = 0;
               if(LIKELY(k_gainMin <= bestGain)) {
                  *pTotalGain = static_cast<double>(bestGain);
                  if(bSplitFirst2) {
                     // if bSplitFirst2 is true, then there definetly was a split, so we don't have to check for zero
                     // splits
                     error = pInnerTermUpdate->SetCountSlices(iDimension2, 2);
                     if(Error_None != error) {
                        // already logged
                        return error;
                     }
                     const size_t temp1 = splitFirst2Best + 1;
                     pInnerTermUpdate->GetSplitPointer(iDimension2)[0] = static_cast<UIntSplit>(temp1);

                     if(splitFirst2LowBest < splitFirst2HighBest) {
                        error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * 6);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        error = pInnerTermUpdate->SetCountSlices(iDimension1, 3);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        const size_t temp2 = splitFirst2LowBest + 1;
                        pInnerTermUpdate->GetSplitPointer(iDimension1)[0] = static_cast<UIntSplit>(temp2);
                        const size_t temp3 = splitFirst2HighBest + 1;
                        pInnerTermUpdate->GetSplitPointer(iDimension1)[1] = static_cast<UIntSplit>(temp3);
                     } else if(splitFirst2HighBest < splitFirst2LowBest) {
                        error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * 6);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        error = pInnerTermUpdate->SetCountSlices(iDimension1, 3);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        const size_t temp4 = splitFirst2HighBest + 1;
                        pInnerTermUpdate->GetSplitPointer(iDimension1)[0] = static_cast<UIntSplit>(temp4);
                        const size_t temp5 = splitFirst2LowBest + 1;
                        pInnerTermUpdate->GetSplitPointer(iDimension1)[1] = static_cast<UIntSplit>(temp5);
                     } else {
                        error = pInnerTermUpdate->SetCountSlices(iDimension1, 2);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }

                        error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * 4);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        const size_t temp6 = splitFirst2LowBest + 1;
                        pInnerTermUpdate->GetSplitPointer(iDimension1)[0] = static_cast<UIntSplit>(temp6);
                     }

                     auto* const pGradientPairTotals2LowLowBest = pTotals2LowLowBest->GetGradientPairs();
                     auto* const pGradientPairTotals2LowHighBest = pTotals2LowHighBest->GetGradientPairs();
                     auto* const pGradientPairTotals2HighLowBest = pTotals2HighLowBest->GetGradientPairs();
                     auto* const pGradientPairTotals2HighHighBest = pTotals2HighHighBest->GetGradientPairs();
                     for(size_t iScore = 0; iScore < cScores; ++iScore) {
                        FloatCalc predictionLowLow;
                        FloatCalc predictionLowHigh;
                        FloatCalc predictionHighLow;
                        FloatCalc predictionHighHigh;

                        FloatCalc weightLowLow;
                        FloatCalc weightLowHigh;
                        FloatCalc weightHighLow;
                        FloatCalc weightHighHigh;

                        if(bUpdateWithHessian) {
                           weightLowLow = static_cast<FloatCalc>(pGradientPairTotals2LowLowBest[iScore].GetHess());
                           predictionLowLow = -CalcNegUpdate(
                                 static_cast<FloatCalc>(pGradientPairTotals2LowLowBest[iScore].m_sumGradients),
                                 weightLowLow);
                           weightLowHigh = static_cast<FloatCalc>(pGradientPairTotals2LowHighBest[iScore].GetHess());
                           predictionLowHigh = -CalcNegUpdate(
                                 static_cast<FloatCalc>(pGradientPairTotals2LowHighBest[iScore].m_sumGradients),
                                 weightLowHigh);
                           weightHighLow = static_cast<FloatCalc>(pGradientPairTotals2HighLowBest[iScore].GetHess());
                           predictionHighLow = -CalcNegUpdate(
                                 static_cast<FloatCalc>(pGradientPairTotals2HighLowBest[iScore].m_sumGradients),
                                 weightHighLow);
                           weightHighHigh = static_cast<FloatCalc>(pGradientPairTotals2HighHighBest[iScore].GetHess());
                           predictionHighHigh = -CalcNegUpdate(
                                 static_cast<FloatCalc>(pGradientPairTotals2HighHighBest[iScore].m_sumGradients),
                                 weightHighHigh);
                        } else {
                           weightLowLow = static_cast<FloatCalc>(pTotals2LowLowBest->GetWeight());
                           predictionLowLow = -CalcNegUpdate(
                                 static_cast<FloatCalc>(pGradientPairTotals2LowLowBest[iScore].m_sumGradients),
                                 weightLowLow);
                           weightLowHigh = static_cast<FloatCalc>(pTotals2LowHighBest->GetWeight());
                           predictionLowHigh = -CalcNegUpdate(
                                 static_cast<FloatCalc>(pGradientPairTotals2LowHighBest[iScore].m_sumGradients),
                                 weightLowHigh);
                           weightHighLow = static_cast<FloatCalc>(pTotals2HighLowBest->GetWeight());
                           predictionHighLow = -CalcNegUpdate(
                                 static_cast<FloatCalc>(pGradientPairTotals2HighLowBest[iScore].m_sumGradients),
                                 weightHighLow);
                           weightHighHigh = static_cast<FloatCalc>(pTotals2HighHighBest->GetWeight());
                           predictionHighHigh = -CalcNegUpdate(
                                 static_cast<FloatCalc>(pGradientPairTotals2HighHighBest[iScore].m_sumGradients),
                                 weightHighHigh);
                        }

                        FloatScore* const aUpdateScores = pInnerTermUpdate->GetTensorScoresPointer();
                        if(splitFirst2LowBest < splitFirst2HighBest) {
                           aUpdateScores[0 * cScores + iScore] = static_cast<FloatScore>(predictionLowLow);
                           aUpdateScores[1 * cScores + iScore] = static_cast<FloatScore>(predictionLowHigh);
                           aUpdateScores[2 * cScores + iScore] = static_cast<FloatScore>(predictionLowHigh);
                           aUpdateScores[3 * cScores + iScore] = static_cast<FloatScore>(predictionHighLow);
                           aUpdateScores[4 * cScores + iScore] = static_cast<FloatScore>(predictionHighLow);
                           aUpdateScores[5 * cScores + iScore] = static_cast<FloatScore>(predictionHighHigh);
                           if(nullptr != aWeights) {
                              aWeights[0 + 6 * iScore] = static_cast<double>(weightLowLow);
                              aWeights[1 + 6 * iScore] = static_cast<double>(weightLowHigh);
                              aWeights[2 + 6 * iScore] = static_cast<double>(weightLowHigh);
                              aWeights[3 + 6 * iScore] = static_cast<double>(weightHighLow);
                              aWeights[4 + 6 * iScore] = static_cast<double>(weightHighLow);
                              aWeights[5 + 6 * iScore] = static_cast<double>(weightHighHigh);
                           }
                        } else if(splitFirst2HighBest < splitFirst2LowBest) {
                           aUpdateScores[0 * cScores + iScore] = static_cast<FloatScore>(predictionLowLow);
                           aUpdateScores[1 * cScores + iScore] = static_cast<FloatScore>(predictionLowLow);
                           aUpdateScores[2 * cScores + iScore] = static_cast<FloatScore>(predictionLowHigh);
                           aUpdateScores[3 * cScores + iScore] = static_cast<FloatScore>(predictionHighLow);
                           aUpdateScores[4 * cScores + iScore] = static_cast<FloatScore>(predictionHighHigh);
                           aUpdateScores[5 * cScores + iScore] = static_cast<FloatScore>(predictionHighHigh);
                           if(nullptr != aWeights) {
                              aWeights[0 + 6 * iScore] = static_cast<double>(weightLowLow);
                              aWeights[1 + 6 * iScore] = static_cast<double>(weightLowLow);
                              aWeights[2 + 6 * iScore] = static_cast<double>(weightLowHigh);
                              aWeights[3 + 6 * iScore] = static_cast<double>(weightHighLow);
                              aWeights[4 + 6 * iScore] = static_cast<double>(weightHighHigh);
                              aWeights[5 + 6 * iScore] = static_cast<double>(weightHighHigh);
                           }
                        } else {
                           aUpdateScores[0 * cScores + iScore] = static_cast<FloatScore>(predictionLowLow);
                           aUpdateScores[1 * cScores + iScore] = static_cast<FloatScore>(predictionLowHigh);
                           aUpdateScores[2 * cScores + iScore] = static_cast<FloatScore>(predictionHighLow);
                           aUpdateScores[3 * cScores + iScore] = static_cast<FloatScore>(predictionHighHigh);
                           if(nullptr != aWeights) {
                              aWeights[0 + 4 * iScore] = static_cast<double>(weightLowLow);
                              aWeights[1 + 4 * iScore] = static_cast<double>(weightLowHigh);
                              aWeights[2 + 4 * iScore] = static_cast<double>(weightHighLow);
                              aWeights[3 + 4 * iScore] = static_cast<double>(weightHighHigh);
                           }
                        }
                     }
                  } else {
                     error = pInnerTermUpdate->SetCountSlices(iDimension1, 2);
                     if(Error_None != error) {
                        // already logged
                        return error;
                     }

                     // The Clang static analyzer believes that splitFirst1Best could contain uninitialized garbage
                     // We can only reach here if bSplitFirst2 is false and if k_illegalGainFloat != bestGain
                     // Since bestGain is only set in two places, and since in one of those bSplitFirst2 is set to true
                     // our code path above must have gone through the section that set both bestGain and
                     // splitFirst1Best.  The Clang static analyzer does not seem to recognize this, so stop it
                     StopClangAnalysis();
                     const size_t temp7 = splitFirst1Best + 1;
                     pInnerTermUpdate->GetSplitPointer(iDimension1)[0] = static_cast<UIntSplit>(temp7);

                     if(splitFirst1LowBest < splitFirst1HighBest) {
                        error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * 6);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }

                        error = pInnerTermUpdate->SetCountSlices(iDimension2, 3);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        const size_t temp8 = splitFirst1LowBest + 1;
                        pInnerTermUpdate->GetSplitPointer(iDimension2)[0] = static_cast<UIntSplit>(temp8);
                        const size_t temp9 = splitFirst1HighBest + 1;
                        pInnerTermUpdate->GetSplitPointer(iDimension2)[1] = static_cast<UIntSplit>(temp9);
                     } else if(splitFirst1HighBest < splitFirst1LowBest) {
                        error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * 6);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }

                        error = pInnerTermUpdate->SetCountSlices(iDimension2, 3);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        const size_t temp10 = splitFirst1HighBest + 1;
                        pInnerTermUpdate->GetSplitPointer(iDimension2)[0] = static_cast<UIntSplit>(temp10);
                        const size_t temp11 = splitFirst1LowBest + 1;
                        pInnerTermUpdate->GetSplitPointer(iDimension2)[1] = static_cast<UIntSplit>(temp11);
                     } else {
                        error = pInnerTermUpdate->SetCountSlices(iDimension2, 2);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * 4);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        const size_t temp12 = splitFirst1LowBest + 1;
                        pInnerTermUpdate->GetSplitPointer(iDimension2)[0] = static_cast<UIntSplit>(temp12);
                     }

                     auto* const pGradientPairTotals1LowLowBest = pTotals1LowLowBest->GetGradientPairs();
                     auto* const pGradientPairTotals1LowHighBest = pTotals1LowHighBest->GetGradientPairs();
                     auto* const pGradientPairTotals1HighLowBest = pTotals1HighLowBest->GetGradientPairs();
                     auto* const pGradientPairTotals1HighHighBest = pTotals1HighHighBest->GetGradientPairs();
                     for(size_t iScore = 0; iScore < cScores; ++iScore) {
                        FloatCalc predictionLowLow;
                        FloatCalc predictionLowHigh;
                        FloatCalc predictionHighLow;
                        FloatCalc predictionHighHigh;

                        FloatCalc weightLowLow;
                        FloatCalc weightLowHigh;
                        FloatCalc weightHighLow;
                        FloatCalc weightHighHigh;

                        if(bUpdateWithHessian) {
                           weightLowLow = static_cast<FloatCalc>(pGradientPairTotals1LowLowBest[iScore].GetHess());
                           predictionLowLow = -CalcNegUpdate(
                                 static_cast<FloatCalc>(pGradientPairTotals1LowLowBest[iScore].m_sumGradients),
                                 weightLowLow);
                           weightLowHigh = static_cast<FloatCalc>(pGradientPairTotals1LowHighBest[iScore].GetHess());
                           predictionLowHigh = -CalcNegUpdate(
                                 static_cast<FloatCalc>(pGradientPairTotals1LowHighBest[iScore].m_sumGradients),
                                 weightLowHigh);
                           weightHighLow = static_cast<FloatCalc>(pGradientPairTotals1HighLowBest[iScore].GetHess());
                           predictionHighLow = -CalcNegUpdate(
                                 static_cast<FloatCalc>(pGradientPairTotals1HighLowBest[iScore].m_sumGradients),
                                 weightHighLow);
                           weightHighHigh = static_cast<FloatCalc>(pGradientPairTotals1HighHighBest[iScore].GetHess());
                           predictionHighHigh = -CalcNegUpdate(
                                 static_cast<FloatCalc>(pGradientPairTotals1HighHighBest[iScore].m_sumGradients),
                                 weightHighHigh);
                        } else {
                           weightLowLow = static_cast<FloatCalc>(pTotals1LowLowBest->GetWeight());
                           predictionLowLow = -CalcNegUpdate(
                                 static_cast<FloatCalc>(pGradientPairTotals1LowLowBest[iScore].m_sumGradients),
                                 weightLowLow);
                           weightLowHigh = static_cast<FloatCalc>(pTotals1LowHighBest->GetWeight());
                           predictionLowHigh = -CalcNegUpdate(
                                 static_cast<FloatCalc>(pGradientPairTotals1LowHighBest[iScore].m_sumGradients),
                                 weightLowHigh);
                           weightHighLow = static_cast<FloatCalc>(pTotals1HighLowBest->GetWeight());
                           predictionHighLow = -CalcNegUpdate(
                                 static_cast<FloatCalc>(pGradientPairTotals1HighLowBest[iScore].m_sumGradients),
                                 weightHighLow);
                           weightHighHigh = static_cast<FloatCalc>(pTotals1HighHighBest->GetWeight());
                           predictionHighHigh = -CalcNegUpdate(
                                 static_cast<FloatCalc>(pGradientPairTotals1HighHighBest[iScore].m_sumGradients),
                                 weightHighHigh);
                        }
                        FloatScore* const aUpdateScores = pInnerTermUpdate->GetTensorScoresPointer();
                        if(splitFirst1LowBest < splitFirst1HighBest) {
                           aUpdateScores[0 * cScores + iScore] = static_cast<FloatScore>(predictionLowLow);
                           aUpdateScores[1 * cScores + iScore] = static_cast<FloatScore>(predictionHighLow);
                           aUpdateScores[2 * cScores + iScore] = static_cast<FloatScore>(predictionLowHigh);
                           aUpdateScores[3 * cScores + iScore] = static_cast<FloatScore>(predictionHighLow);
                           aUpdateScores[4 * cScores + iScore] = static_cast<FloatScore>(predictionLowHigh);
                           aUpdateScores[5 * cScores + iScore] = static_cast<FloatScore>(predictionHighHigh);
                           if(nullptr != aWeights) {
                              aWeights[0 + 6 * iScore] = static_cast<double>(weightLowLow);
                              aWeights[1 + 6 * iScore] = static_cast<double>(weightHighLow);
                              aWeights[2 + 6 * iScore] = static_cast<double>(weightLowHigh);
                              aWeights[3 + 6 * iScore] = static_cast<double>(weightHighLow);
                              aWeights[4 + 6 * iScore] = static_cast<double>(weightLowHigh);
                              aWeights[5 + 6 * iScore] = static_cast<double>(weightHighHigh);
                           }
                        } else if(splitFirst1HighBest < splitFirst1LowBest) {
                           aUpdateScores[0 * cScores + iScore] = static_cast<FloatScore>(predictionLowLow);
                           aUpdateScores[1 * cScores + iScore] = static_cast<FloatScore>(predictionHighLow);
                           aUpdateScores[2 * cScores + iScore] = static_cast<FloatScore>(predictionLowLow);
                           aUpdateScores[3 * cScores + iScore] = static_cast<FloatScore>(predictionHighHigh);
                           aUpdateScores[4 * cScores + iScore] = static_cast<FloatScore>(predictionLowHigh);
                           aUpdateScores[5 * cScores + iScore] = static_cast<FloatScore>(predictionHighHigh);
                           if(nullptr != aWeights) {
                              aWeights[0 + 6 * iScore] = static_cast<double>(weightLowLow);
                              aWeights[1 + 6 * iScore] = static_cast<double>(weightHighLow);
                              aWeights[2 + 6 * iScore] = static_cast<double>(weightLowLow);
                              aWeights[3 + 6 * iScore] = static_cast<double>(weightHighHigh);
                              aWeights[4 + 6 * iScore] = static_cast<double>(weightLowHigh);
                              aWeights[5 + 6 * iScore] = static_cast<double>(weightHighHigh);
                           }
                        } else {
                           aUpdateScores[0 * cScores + iScore] = static_cast<FloatScore>(predictionLowLow);
                           aUpdateScores[1 * cScores + iScore] = static_cast<FloatScore>(predictionHighLow);
                           aUpdateScores[2 * cScores + iScore] = static_cast<FloatScore>(predictionLowHigh);
                           aUpdateScores[3 * cScores + iScore] = static_cast<FloatScore>(predictionHighHigh);
                           if(nullptr != aWeights) {
                              aWeights[0 + 4 * iScore] = static_cast<double>(weightLowLow);
                              aWeights[1 + 4 * iScore] = static_cast<double>(weightHighLow);
                              aWeights[2 + 4 * iScore] = static_cast<double>(weightLowHigh);
                              aWeights[3 + 4 * iScore] = static_cast<double>(weightHighHigh);
                           }
                        }
                     }
                  }
                  return Error_None;
               }
            } else {
               EBM_ASSERT(std::isnan(bestGain) || -std::numeric_limits<FloatCalc>::infinity() == bestGain);
            }
         } else {
            EBM_ASSERT(std::isnan(bestGain) || std::numeric_limits<FloatCalc>::infinity() == bestGain);
         }
      } else {
         EBM_ASSERT(!std::isnan(bestGain));
      }

      // there were no good splits found
#ifndef NDEBUG
      const ErrorEbm errorDebug1 =
#endif // NDEBUG
            pInnerTermUpdate->SetCountSlices(iDimension1, 1);
      // we can't fail since we're setting this to zero, so no allocations.  We don't in fact need the split array at
      // all
      EBM_ASSERT(Error_None == errorDebug1);

#ifndef NDEBUG
      const ErrorEbm errorDebug2 =
#endif // NDEBUG
            pInnerTermUpdate->SetCountSlices(iDimension2, 1);
      // we can't fail since we're setting this to zero, so no allocations.  We don't in fact need the split array at
      // all
      EBM_ASSERT(Error_None == errorDebug2);

      // we don't need to call pInnerTermUpdate->EnsureTensorScoreCapacity,
      // since our value capacity would be 1, which is pre-allocated

      FloatScore* const aUpdateScores = pInnerTermUpdate->GetTensorScoresPointer();
      for(size_t iScore = 0; iScore < cScores; ++iScore) {
         FloatCalc update;
         FloatCalc weight;
         if(bUpdateWithHessian) {
            weight = static_cast<FloatCalc>(pGradientPairTotal[iScore].GetHess());
            update = -CalcNegUpdate(static_cast<FloatCalc>(pGradientPairTotal[iScore].m_sumGradients), weight);
            if(nullptr != aWeights) {
               aWeights[iScore] = static_cast<double>(weight);
            }
         } else {
            weight = static_cast<FloatCalc>(weightAll);
            update = -CalcNegUpdate(static_cast<FloatCalc>(pGradientPairTotal[iScore].m_sumGradients), weight);
            if(nullptr != aWeights) {
               aWeights[iScore] = static_cast<double>(weight);
            }
         }

         aUpdateScores[iScore] = static_cast<FloatScore>(update);
      }
      return Error_None;
   }
   WARNING_POP
};

template<bool bHessian, size_t cPossibleScores> class PartitionTwoDimensionalBoostingTarget final {
 public:
   PartitionTwoDimensionalBoostingTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BoosterShell* const pBoosterShell,
         const TermBoostFlags flags,
         const Term* const pTerm,
         const size_t* const acBins,
         const size_t cSamplesLeafMin,
         const FloatCalc hessianMin,
         BinBase* aAuxiliaryBinsBase,
         double* const aWeights,
         double* const pTotalGain
#ifndef NDEBUG
         ,
         const BinBase* const aDebugCopyBinsBase
#endif // NDEBUG
   ) {
      BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
      if(cPossibleScores == pBoosterCore->GetCountScores()) {
         return PartitionTwoDimensionalBoostingInternal<bHessian, cPossibleScores>::Func(pBoosterShell,
               flags,
               pTerm,
               acBins,
               cSamplesLeafMin,
               hessianMin,
               aAuxiliaryBinsBase,
               aWeights,
               pTotalGain
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase
#endif // NDEBUG
         );
      } else {
         return PartitionTwoDimensionalBoostingTarget<bHessian, cPossibleScores + 1>::Func(pBoosterShell,
               flags,
               pTerm,
               acBins,
               cSamplesLeafMin,
               hessianMin,
               aAuxiliaryBinsBase,
               aWeights,
               pTotalGain
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase
#endif // NDEBUG
         );
      }
   }
};

template<bool bHessian> class PartitionTwoDimensionalBoostingTarget<bHessian, k_cCompilerScoresMax + 1> final {
 public:
   PartitionTwoDimensionalBoostingTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BoosterShell* const pBoosterShell,
         const TermBoostFlags flags,
         const Term* const pTerm,
         const size_t* const acBins,
         const size_t cSamplesLeafMin,
         const FloatCalc hessianMin,
         BinBase* aAuxiliaryBinsBase,
         double* const aWeights,
         double* const pTotalGain
#ifndef NDEBUG
         ,
         const BinBase* const aDebugCopyBinsBase
#endif // NDEBUG
   ) {
      return PartitionTwoDimensionalBoostingInternal<bHessian, k_dynamicScores>::Func(pBoosterShell,
            flags,
            pTerm,
            acBins,
            cSamplesLeafMin,
            hessianMin,
            aAuxiliaryBinsBase,
            aWeights,
            pTotalGain
#ifndef NDEBUG
            ,
            aDebugCopyBinsBase
#endif // NDEBUG
      );
   }
};

extern ErrorEbm PartitionTwoDimensionalBoosting(BoosterShell* const pBoosterShell,
      const TermBoostFlags flags,
      const Term* const pTerm,
      const size_t* const acBins,
      const size_t cSamplesLeafMin,
      const FloatCalc hessianMin,
      BinBase* aAuxiliaryBinsBase,
      double* const aWeights,
      double* const pTotalGain
#ifndef NDEBUG
      ,
      const BinBase* const aDebugCopyBinsBase
#endif // NDEBUG
) {
   BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   const size_t cRuntimeScores = pBoosterCore->GetCountScores();

   EBM_ASSERT(1 <= cRuntimeScores);
   if(pBoosterCore->IsHessian()) {
      if(size_t{1} != cRuntimeScores) {
         // muticlass
         return PartitionTwoDimensionalBoostingTarget<true, k_cCompilerScoresStart>::Func(pBoosterShell,
               flags,
               pTerm,
               acBins,
               cSamplesLeafMin,
               hessianMin,
               aAuxiliaryBinsBase,
               aWeights,
               pTotalGain
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase
#endif // NDEBUG
         );
      } else {
         return PartitionTwoDimensionalBoostingInternal<true, k_oneScore>::Func(pBoosterShell,
               flags,
               pTerm,
               acBins,
               cSamplesLeafMin,
               hessianMin,
               aAuxiliaryBinsBase,
               aWeights,
               pTotalGain
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase
#endif // NDEBUG
         );
      }
   } else {
      if(size_t{1} != cRuntimeScores) {
         // Odd: gradient multiclass. Allow it, but do not optimize for it
         return PartitionTwoDimensionalBoostingInternal<false, k_dynamicScores>::Func(pBoosterShell,
               flags,
               pTerm,
               acBins,
               cSamplesLeafMin,
               hessianMin,
               aAuxiliaryBinsBase,
               aWeights,
               pTotalGain
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase
#endif // NDEBUG
         );
      } else {
         return PartitionTwoDimensionalBoostingInternal<false, k_oneScore>::Func(pBoosterShell,
               flags,
               pTerm,
               acBins,
               cSamplesLeafMin,
               hessianMin,
               aAuxiliaryBinsBase,
               aWeights,
               pTotalGain
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase
#endif // NDEBUG
         );
      }
   }
}

} // namespace DEFINED_ZONE_NAME
