// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "ebm_native.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // LIKELY
#include "zones.h"

#include "ebm_stats.hpp"
#include "Feature.hpp"
#include "Term.hpp"
#include "Tensor.hpp"
#include "GradientPair.hpp"
#include "Bin.hpp"
#include "TensorTotalsSum.hpp"
#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<ptrdiff_t cCompilerClasses, size_t cCompilerDimensions>
static FloatBig SweepMultiDimensional(
   const ptrdiff_t cRuntimeClasses,
   const size_t cRuntimeRealDimensions,
   const size_t * const aiPoint,
   const size_t * const acBins,
   const size_t directionVectorLow,
   const size_t iDimensionSweep,
   const Bin<FloatBig, IsClassification(cCompilerClasses), GetCountScores(cCompilerClasses)> * const aBins,
   const size_t cSamplesLeafMin,
   Bin<FloatBig, IsClassification(cCompilerClasses), GetCountScores(cCompilerClasses)> * const pBinBestAndTemp,
   size_t * const piBestSplit
#ifndef NDEBUG
   , const Bin<FloatBig, IsClassification(cCompilerClasses), GetCountScores(cCompilerClasses)> * const aDebugCopyBins
   , const BinBase * const pBinsEndDebug
#endif // NDEBUG
) {
   static constexpr bool bClassification = IsClassification(cCompilerClasses);
   static constexpr size_t cCompilerScores = GetCountScores(cCompilerClasses);

   const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, cRuntimeClasses);
   const size_t cScores = GetCountScores(cClasses);
   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we're accessing allocated memory
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

   const size_t cRealDimensions = GET_COUNT_DIMENSIONS(cCompilerDimensions, cRuntimeRealDimensions);
   EBM_ASSERT(1 <= cRealDimensions); // for interactions, we just return 0 for interactions with zero features
   EBM_ASSERT(iDimensionSweep < cRealDimensions);
   EBM_ASSERT(0 == (directionVectorLow & (size_t { 1 } << iDimensionSweep)));

   TensorSumDimension aDimensions[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
   size_t iDimensionInit = 0;
   do {
      // move data to a local variable that the compiler can reason about and then eliminate by moving to CPU registers

      aDimensions[iDimensionInit].m_iPoint = aiPoint[iDimensionInit];
      aDimensions[iDimensionInit].m_cBins = acBins[iDimensionInit];

      ++iDimensionInit;
   } while(cRealDimensions != iDimensionInit);

   const size_t directionVectorHigh = directionVectorLow | size_t { 1 } << iDimensionSweep;
   const size_t cSweepCuts = aDimensions[iDimensionSweep].m_cBins - 1;
   EBM_ASSERT(1 <= cSweepCuts); // dimensions with 1 bin are removed earlier

   size_t iBestSplit = 0;

   auto * const p_DO_NOT_USE_DIRECTLY_Low = IndexBin(pBinBestAndTemp, cBytesPerBin * 2);
   ASSERT_BIN_OK(cBytesPerBin, p_DO_NOT_USE_DIRECTLY_Low, pBinsEndDebug);
   auto * const p_DO_NOT_USE_DIRECTLY_High = IndexBin(pBinBestAndTemp, cBytesPerBin * 3);
   ASSERT_BIN_OK(cBytesPerBin, p_DO_NOT_USE_DIRECTLY_High, pBinsEndDebug);

   Bin<FloatBig, bClassification, cCompilerScores> binLow;
   Bin<FloatBig, bClassification, cCompilerScores> binHigh;

   // if we know how many scores there are, use the memory on the stack where the compiler can optimize access
   static constexpr bool bUseStackMemory = k_dynamicClassification != cCompilerClasses;
   auto * const aGradientPairsLow = bUseStackMemory ? binLow.GetGradientPairs() : p_DO_NOT_USE_DIRECTLY_Low->GetGradientPairs();
   auto * const aGradientPairsHigh = bUseStackMemory ? binHigh.GetGradientPairs() : p_DO_NOT_USE_DIRECTLY_High->GetGradientPairs();

   EBM_ASSERT(0 < cSamplesLeafMin);

   FloatBig bestGain = k_illegalGainFloat;
   size_t iBin = 0;
   do {
      aDimensions[iDimensionSweep].m_iPoint = iBin;
      EBM_ASSERT(2 == cRealDimensions); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
      TensorTotalsSum<cCompilerClasses, cCompilerDimensions>(
         cRuntimeClasses,
         cRealDimensions,
         aDimensions,
         directionVectorLow,
         aBins,
         binLow,
         aGradientPairsLow
#ifndef NDEBUG
         , aDebugCopyBins
         , pBinsEndDebug
#endif // NDEBUG
      );
      if(LIKELY(cSamplesLeafMin <= binLow.GetCountSamples())) {
         EBM_ASSERT(2 == cRealDimensions); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
         TensorTotalsSum<cCompilerClasses, cCompilerDimensions>(
            cRuntimeClasses,
            cRealDimensions,
            aDimensions,
            directionVectorHigh,
            aBins,
            binHigh,
            aGradientPairsHigh
#ifndef NDEBUG
            , aDebugCopyBins
            , pBinsEndDebug
#endif // NDEBUG
         );
         if(LIKELY(cSamplesLeafMin <= binHigh.GetCountSamples())) {
            FloatBig gain = 0;
            EBM_ASSERT(0 < binLow.GetCountSamples());
            EBM_ASSERT(0 < binHigh.GetCountSamples());

            EBM_ASSERT(1 <= cScores);
            size_t iScore = 0;
            do {
               // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators 
               // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

               static constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;
               
               const FloatBig gain1 = EbmStats::CalcPartialGain(
                  aGradientPairsLow[iScore].m_sumGradients, bUseLogitBoost ? aGradientPairsLow[iScore].GetHess() : binLow.GetWeight());
               EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);
               gain += gain1;
               
               const FloatBig gain2 = EbmStats::CalcPartialGain(
                  aGradientPairsHigh[iScore].m_sumGradients, bUseLogitBoost ? aGradientPairsHigh[iScore].GetHess() : binHigh.GetWeight());
               EBM_ASSERT(std::isnan(gain2) || 0 <= gain2);
               gain += gain2;

               ++iScore;
            } while(cScores != iScore);
            EBM_ASSERT(std::isnan(gain) || 0 <= gain); // sumation of positive numbers should be positive

            if(UNLIKELY(/* NaN */ !LIKELY(gain <= bestGain))) {
               // propagate NaNs

               bestGain = gain;
               iBestSplit = iBin;

               auto * const pTotalsLowOut = IndexBin(pBinBestAndTemp, cBytesPerBin * 0);
               ASSERT_BIN_OK(cBytesPerBin, pTotalsLowOut, pBinsEndDebug);

               pTotalsLowOut->Copy(cScores, binLow, aGradientPairsLow);

               auto * const pTotalsHighOut = IndexBin(pBinBestAndTemp, cBytesPerBin * 1);
               ASSERT_BIN_OK(cBytesPerBin, pTotalsHighOut, pBinsEndDebug);

               pTotalsHighOut->Copy(cScores, binHigh, aGradientPairsHigh);
            } else {
               EBM_ASSERT(!std::isnan(gain));
            }
         }
      }
      ++iBin;
   } while(cSweepCuts != iBin);
   *piBestSplit = iBestSplit;

   EBM_ASSERT(std::isnan(bestGain) || k_illegalGainFloat == bestGain || 0 <= bestGain);
   return bestGain;
}

template<ptrdiff_t cCompilerClasses>
class PartitionTwoDimensionalBoostingInternal final {
public:

   PartitionTwoDimensionalBoostingInternal() = delete; // this is a static class.  Do not construct

   WARNING_PUSH
   WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm,
      const size_t * const acBins,
      const size_t cSamplesLeafMin,
      BinBase * const aAuxiliaryBinsBase,
      double * const pTotalGain
#ifndef NDEBUG
      , const BinBase * const aDebugCopyBinsBase
#endif // NDEBUG
   ) {
      static constexpr bool bClassification = IsClassification(cCompilerClasses);
      static constexpr size_t cCompilerScores = GetCountScores(cCompilerClasses);
      static constexpr size_t cCompilerDimensions = 2;

      ErrorEbm error;
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();

      auto * const aBins = pBoosterShell->GetBoostingBigBins()->Specialize<FloatBig, bClassification, cCompilerScores>();
      Tensor * const pInnerTermUpdate = pBoosterShell->GetInnerTermUpdate();

      const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();

      const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, cRuntimeClasses);

      const size_t cScores = GetCountScores(cClasses);
      const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

      auto * const aAuxiliaryBins = aAuxiliaryBinsBase->Specialize<FloatBig, bClassification, cCompilerScores>();

#ifndef NDEBUG
      const auto * const aDebugCopyBins = aDebugCopyBinsBase->Specialize<FloatBig, bClassification, cCompilerScores>();
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
      const FeatureBoosting * const * ppFeature = pTerm->GetFeatures();
      const FeatureBoosting * const * const ppFeaturesEnd = &ppFeature[pTerm->GetCountDimensions()];
      do {
         const FeatureBoosting * const pFeature = *ppFeature;
         const size_t cBins = pFeature->GetCountBins();
         EBM_ASSERT(size_t { 1 } <= cBins); // we don't boost on empty training sets
         if(size_t { 1 } < cBins) {
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
         ++ppFeature;
      } while(ppFeaturesEnd != ppFeature);
      EBM_ASSERT(2 <= cBinsDimension1);
      EBM_ASSERT(2 <= cBinsDimension2);

      FloatBig bestGain = k_illegalGainFloat;

      size_t splitFirst1Best;
      size_t splitFirst1LowBest;
      size_t splitFirst1HighBest;

      auto * pTotals1LowLowBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 0);
      auto * pTotals1LowHighBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 1);
      auto * pTotals1HighLowBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 2);
      auto * pTotals1HighHighBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 3);

      EBM_ASSERT(0 < cSamplesLeafMin);

      LOG_0(Trace_Verbose, "PartitionTwoDimensionalBoostingInternal Starting FIRST bin sweep loop");
      size_t iBin1 = 0;
      do {
         aiStart[0] = iBin1;
         size_t splitSecond1LowBest;
         auto * pTotals2LowLowBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 4);
         auto * pTotals2LowHighBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 5);
         const FloatBig gain1 = SweepMultiDimensional<cCompilerClasses, cCompilerDimensions>(
            cRuntimeClasses,
            pTerm->GetCountRealDimensions(),
            aiStart,
            acBins,
            0x0,
            1,
            aBins,
            cSamplesLeafMin,
            pTotals2LowLowBest,
            &splitSecond1LowBest
#ifndef NDEBUG
            , aDebugCopyBins
            , pBoosterShell->GetDebugBigBinsEnd()
#endif // NDEBUG
         );

         if(LIKELY(/* NaN */ !UNLIKELY(gain1 < 0))) {
            EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);

            size_t splitSecond1HighBest;
            auto * pTotals2HighLowBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 8);
            auto * pTotals2HighHighBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 9);
            const FloatBig gain2 = SweepMultiDimensional<cCompilerClasses, cCompilerDimensions>(
               cRuntimeClasses,
               pTerm->GetCountRealDimensions(),
               aiStart,
               acBins,
               0x1,
               1,
               aBins,
               cSamplesLeafMin,
               pTotals2HighLowBest,
               &splitSecond1HighBest
#ifndef NDEBUG
               , aDebugCopyBins
               , pBoosterShell->GetDebugBigBinsEnd()
#endif // NDEBUG
            );

            if(LIKELY(/* NaN */ !UNLIKELY(gain2 < 0))) {
               EBM_ASSERT(std::isnan(gain2) || 0 <= gain2);

               const FloatBig gain = gain1 + gain2;
               if(UNLIKELY(/* NaN */ !LIKELY(gain <= bestGain))) {
                  // propagate NaNs

                  EBM_ASSERT(std::isnan(gain) || 0 <= gain);

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

      auto * pTotals2LowLowBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 12);
      auto * pTotals2LowHighBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 13);
      auto * pTotals2HighLowBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 14);
      auto * pTotals2HighHighBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 15);

      LOG_0(Trace_Verbose, "PartitionTwoDimensionalBoostingInternal Starting SECOND bin sweep loop");
      size_t iBin2 = 0;
      do {
         aiStart[1] = iBin2;
         size_t splitSecond2LowBest;
         auto * pTotals1LowLowBestInner = IndexBin(aAuxiliaryBins, cBytesPerBin * 16);
         auto * pTotals1LowHighBestInner = IndexBin(aAuxiliaryBins, cBytesPerBin * 17);
         const FloatBig gain1 = SweepMultiDimensional<cCompilerClasses, cCompilerDimensions>(
            cRuntimeClasses,
            pTerm->GetCountRealDimensions(),
            aiStart,
            acBins,
            0x0,
            0,
            aBins,
            cSamplesLeafMin,
            pTotals1LowLowBestInner,
            &splitSecond2LowBest
#ifndef NDEBUG
            , aDebugCopyBins
            , pBoosterShell->GetDebugBigBinsEnd()
#endif // NDEBUG
         );

         if(LIKELY(/* NaN */ !UNLIKELY(gain1 < 0))) {
            EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);

            size_t splitSecond2HighBest;
            auto * pTotals1HighLowBestInner = IndexBin(aAuxiliaryBins, cBytesPerBin * 20);
            auto * pTotals1HighHighBestInner = IndexBin(aAuxiliaryBins, cBytesPerBin * 21);
            const FloatBig gain2 = SweepMultiDimensional<cCompilerClasses, cCompilerDimensions>(
               cRuntimeClasses,
               pTerm->GetCountRealDimensions(),
               aiStart,
               acBins,
               0x2,
               0,
               aBins,
               cSamplesLeafMin,
               pTotals1HighLowBestInner,
               &splitSecond2HighBest
#ifndef NDEBUG
               , aDebugCopyBins
               , pBoosterShell->GetDebugBigBinsEnd()
#endif // NDEBUG
            );

            if(LIKELY(/* NaN */ !UNLIKELY(gain2 < 0))) {
               EBM_ASSERT(std::isnan(gain2) || 0 <= gain2);

               const FloatBig gain = gain1 + gain2;
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

      EBM_ASSERT(std::isnan(bestGain) || k_illegalGainFloat == bestGain || 0 <= bestGain);

      // the bin before the aAuxiliaryBins is the last summation bin of aBinsBase, 
      // which contains the totals of all bins
      const auto * const pTotal = NegativeIndexBin(aAuxiliaryBins, cBytesPerBin);

      ASSERT_BIN_OK(cBytesPerBin, pTotal, pBoosterShell->GetDebugBigBinsEnd());

      const auto * const pGradientPairTotal = pTotal->GetGradientPairs();

      const FloatBig weightAll = pTotal->GetWeight();
      EBM_ASSERT(0 < weightAll);

      *pTotalGain = 0;
      EBM_ASSERT(0 <= k_gainMin);
      if(LIKELY(/* NaN */ !UNLIKELY(bestGain < k_gainMin))) {
         EBM_ASSERT(std::isnan(bestGain) || 0 <= bestGain);

         // signal that we've hit an overflow.  Use +inf here since our caller likes that and will flip to -inf
         *pTotalGain = std::numeric_limits<double>::infinity();
         if(LIKELY(/* NaN */ bestGain <= std::numeric_limits<FloatBig>::max())) {
            EBM_ASSERT(!std::isnan(bestGain));
            EBM_ASSERT(0 <= bestGain);
            EBM_ASSERT(std::numeric_limits<FloatBig>::infinity() != bestGain);

            // now subtract the parent partial gain
            for(size_t iScore = 0; iScore < cScores; ++iScore) {
               // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators 
               // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

               static constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;
               const FloatBig gain1 = EbmStats::CalcPartialGain(
                  pGradientPairTotal[iScore].m_sumGradients,
                  bUseLogitBoost ? pGradientPairTotal[iScore].GetHess() : weightAll
               );
               EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);
               bestGain -= gain1;
            }

            EBM_ASSERT(std::numeric_limits<FloatBig>::infinity() != bestGain);
            EBM_ASSERT(std::isnan(bestGain) || -std::numeric_limits<FloatBig>::infinity() == bestGain ||
               k_epsilonNegativeGainAllowed <= bestGain);

            if(LIKELY(/* NaN */ std::numeric_limits<FloatBig>::lowest() <= bestGain)) {
               EBM_ASSERT(!std::isnan(bestGain));
               EBM_ASSERT(!std::isinf(bestGain));
               EBM_ASSERT(k_epsilonNegativeGainAllowed <= bestGain);

               *pTotalGain = 0;
               if(LIKELY(k_gainMin <= bestGain)) {
                  *pTotalGain = static_cast<double>(bestGain);
                  if(bSplitFirst2) {
                     // if bSplitFirst2 is true, then there definetly was a split, so we don't have to check for zero splits
                     error = pInnerTermUpdate->SetCountSplits(iDimension2, 1);
                     if(Error_None != error) {
                        // already logged
                        return error;
                     }
                     pInnerTermUpdate->GetSplitPointer(iDimension2)[0] = splitFirst2Best;

                     if(splitFirst2LowBest < splitFirst2HighBest) {
                        error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * 6);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        error = pInnerTermUpdate->SetCountSplits(iDimension1, 2);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        pInnerTermUpdate->GetSplitPointer(iDimension1)[0] = splitFirst2LowBest;
                        pInnerTermUpdate->GetSplitPointer(iDimension1)[1] = splitFirst2HighBest;
                     } else if(splitFirst2HighBest < splitFirst2LowBest) {
                        error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * 6);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        error = pInnerTermUpdate->SetCountSplits(iDimension1, 2);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        pInnerTermUpdate->GetSplitPointer(iDimension1)[0] = splitFirst2HighBest;
                        pInnerTermUpdate->GetSplitPointer(iDimension1)[1] = splitFirst2LowBest;
                     } else {
                        error = pInnerTermUpdate->SetCountSplits(iDimension1, 1);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }

                        error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * 4);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        pInnerTermUpdate->GetSplitPointer(iDimension1)[0] = splitFirst2LowBest;
                     }

                     auto * const pGradientPairTotals2LowLowBest = pTotals2LowLowBest->GetGradientPairs();
                     auto * const pGradientPairTotals2LowHighBest = pTotals2LowHighBest->GetGradientPairs();
                     auto * const pGradientPairTotals2HighLowBest = pTotals2HighLowBest->GetGradientPairs();
                     auto * const pGradientPairTotals2HighHighBest = pTotals2HighHighBest->GetGradientPairs();
                     for(size_t iScore = 0; iScore < cScores; ++iScore) {
                        FloatBig predictionLowLow;
                        FloatBig predictionLowHigh;
                        FloatBig predictionHighLow;
                        FloatBig predictionHighHigh;

                        if(bClassification) {
                           predictionLowLow = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals2LowLowBest[iScore].m_sumGradients,
                              pGradientPairTotals2LowLowBest[iScore].GetHess()
                           );
                           predictionLowHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals2LowHighBest[iScore].m_sumGradients,
                              pGradientPairTotals2LowHighBest[iScore].GetHess()
                           );
                           predictionHighLow = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals2HighLowBest[iScore].m_sumGradients,
                              pGradientPairTotals2HighLowBest[iScore].GetHess()
                           );
                           predictionHighHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals2HighHighBest[iScore].m_sumGradients,
                              pGradientPairTotals2HighHighBest[iScore].GetHess()
                           );
                        } else {
                           EBM_ASSERT(IsRegression(cCompilerClasses));
                           predictionLowLow = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals2LowLowBest[iScore].m_sumGradients,
                              pTotals2LowLowBest->GetWeight()
                           );
                           predictionLowHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals2LowHighBest[iScore].m_sumGradients,
                              pTotals2LowHighBest->GetWeight()
                           );
                           predictionHighLow = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals2HighLowBest[iScore].m_sumGradients,
                              pTotals2HighLowBest->GetWeight()
                           );
                           predictionHighHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals2HighHighBest[iScore].m_sumGradients,
                              pTotals2HighHighBest->GetWeight()
                           );
                        }

                        FloatFast * const aUpdateScores = pInnerTermUpdate->GetTensorScoresPointer();
                        if(splitFirst2LowBest < splitFirst2HighBest) {
                           aUpdateScores[0 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionLowLow);
                           aUpdateScores[1 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionLowHigh);
                           aUpdateScores[2 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionLowHigh);
                           aUpdateScores[3 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionHighLow);
                           aUpdateScores[4 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionHighLow);
                           aUpdateScores[5 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionHighHigh);
                        } else if(splitFirst2HighBest < splitFirst2LowBest) {
                           aUpdateScores[0 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionLowLow);
                           aUpdateScores[1 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionLowLow);
                           aUpdateScores[2 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionLowHigh);
                           aUpdateScores[3 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionHighLow);
                           aUpdateScores[4 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionHighHigh);
                           aUpdateScores[5 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionHighHigh);
                        } else {
                           aUpdateScores[0 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionLowLow);
                           aUpdateScores[1 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionLowHigh);
                           aUpdateScores[2 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionHighLow);
                           aUpdateScores[3 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionHighHigh);
                        }
                     }
                  } else {
                     error = pInnerTermUpdate->SetCountSplits(iDimension1, 1);
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
                     pInnerTermUpdate->GetSplitPointer(iDimension1)[0] = splitFirst1Best;

                     if(splitFirst1LowBest < splitFirst1HighBest) {
                        error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * 6);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }

                        error = pInnerTermUpdate->SetCountSplits(iDimension2, 2);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        pInnerTermUpdate->GetSplitPointer(iDimension2)[0] = splitFirst1LowBest;
                        pInnerTermUpdate->GetSplitPointer(iDimension2)[1] = splitFirst1HighBest;
                     } else if(splitFirst1HighBest < splitFirst1LowBest) {
                        error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * 6);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }

                        error = pInnerTermUpdate->SetCountSplits(iDimension2, 2);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        pInnerTermUpdate->GetSplitPointer(iDimension2)[0] = splitFirst1HighBest;
                        pInnerTermUpdate->GetSplitPointer(iDimension2)[1] = splitFirst1LowBest;
                     } else {
                        error = pInnerTermUpdate->SetCountSplits(iDimension2, 1);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * 4);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        pInnerTermUpdate->GetSplitPointer(iDimension2)[0] = splitFirst1LowBest;
                     }

                     auto * const pGradientPairTotals1LowLowBest = pTotals1LowLowBest->GetGradientPairs();
                     auto * const pGradientPairTotals1LowHighBest = pTotals1LowHighBest->GetGradientPairs();
                     auto * const pGradientPairTotals1HighLowBest = pTotals1HighLowBest->GetGradientPairs();
                     auto * const pGradientPairTotals1HighHighBest = pTotals1HighHighBest->GetGradientPairs();
                     for(size_t iScore = 0; iScore < cScores; ++iScore) {
                        FloatBig predictionLowLow;
                        FloatBig predictionLowHigh;
                        FloatBig predictionHighLow;
                        FloatBig predictionHighHigh;

                        if(bClassification) {
                           predictionLowLow = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals1LowLowBest[iScore].m_sumGradients,
                              pGradientPairTotals1LowLowBest[iScore].GetHess()
                           );
                           predictionLowHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals1LowHighBest[iScore].m_sumGradients,
                              pGradientPairTotals1LowHighBest[iScore].GetHess()
                           );
                           predictionHighLow = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals1HighLowBest[iScore].m_sumGradients,
                              pGradientPairTotals1HighLowBest[iScore].GetHess()
                           );
                           predictionHighHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals1HighHighBest[iScore].m_sumGradients,
                              pGradientPairTotals1HighHighBest[iScore].GetHess()
                           );
                        } else {
                           EBM_ASSERT(IsRegression(cCompilerClasses));
                           predictionLowLow = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals1LowLowBest[iScore].m_sumGradients,
                              pTotals1LowLowBest->GetWeight()
                           );
                           predictionLowHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals1LowHighBest[iScore].m_sumGradients,
                              pTotals1LowHighBest->GetWeight()
                           );
                           predictionHighLow = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals1HighLowBest[iScore].m_sumGradients,
                              pTotals1HighLowBest->GetWeight()
                           );
                           predictionHighHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals1HighHighBest[iScore].m_sumGradients,
                              pTotals1HighHighBest->GetWeight()
                           );
                        }
                        FloatFast * const aUpdateScores = pInnerTermUpdate->GetTensorScoresPointer();
                        if(splitFirst1LowBest < splitFirst1HighBest) {
                           aUpdateScores[0 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionLowLow);
                           aUpdateScores[1 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionHighLow);
                           aUpdateScores[2 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionLowHigh);
                           aUpdateScores[3 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionHighLow);
                           aUpdateScores[4 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionLowHigh);
                           aUpdateScores[5 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionHighHigh);
                        } else if(splitFirst1HighBest < splitFirst1LowBest) {
                           aUpdateScores[0 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionLowLow);
                           aUpdateScores[1 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionHighLow);
                           aUpdateScores[2 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionLowLow);
                           aUpdateScores[3 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionHighHigh);
                           aUpdateScores[4 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionLowHigh);
                           aUpdateScores[5 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionHighHigh);
                        } else {
                           aUpdateScores[0 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionLowLow);
                           aUpdateScores[1 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionHighLow);
                           aUpdateScores[2 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionLowHigh);
                           aUpdateScores[3 * cScores + iScore] = SafeConvertFloat<FloatFast>(predictionHighHigh);
                        }
                     }
                  }
                  return Error_None;
               }
            } else {
               EBM_ASSERT(std::isnan(bestGain) || -std::numeric_limits<FloatBig>::infinity() == bestGain);
            }
         } else {
            EBM_ASSERT(std::isnan(bestGain) || std::numeric_limits<FloatBig>::infinity() == bestGain);
         }
      } else {
         EBM_ASSERT(!std::isnan(bestGain));
      }

      // there were no good splits found
#ifndef NDEBUG
      const ErrorEbm errorDebug1 =
#endif // NDEBUG
         pInnerTermUpdate->SetCountSplits(iDimension1, 0);
      // we can't fail since we're setting this to zero, so no allocations.  We don't in fact need the split array at all
      EBM_ASSERT(Error_None == errorDebug1);

#ifndef NDEBUG
      const ErrorEbm errorDebug2 =
#endif // NDEBUG
         pInnerTermUpdate->SetCountSplits(iDimension2, 0);
      // we can't fail since we're setting this to zero, so no allocations.  We don't in fact need the split array at all
      EBM_ASSERT(Error_None == errorDebug2);

      // we don't need to call pInnerTermUpdate->EnsureTensorScoreCapacity, 
      // since our value capacity would be 1, which is pre-allocated

      for(size_t iScore = 0; iScore < cScores; ++iScore) {
         FloatBig update;
         if(bClassification) {
            update = EbmStats::ComputeSinglePartitionUpdate(
               pGradientPairTotal[iScore].m_sumGradients,
               pGradientPairTotal[iScore].GetHess()
            );
         } else {
            EBM_ASSERT(IsRegression(cCompilerClasses));
            update = EbmStats::ComputeSinglePartitionUpdate(
               pGradientPairTotal[iScore].m_sumGradients,
               weightAll
            );
         }

         FloatFast * const aUpdateScores = pInnerTermUpdate->GetTensorScoresPointer();
         aUpdateScores[iScore] = SafeConvertFloat<FloatFast>(update);
      }
      return Error_None;
   }
   WARNING_POP
};

template<ptrdiff_t cPossibleClasses>
class PartitionTwoDimensionalBoostingTarget final {
public:

   PartitionTwoDimensionalBoostingTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm,
      const size_t * const acBins,
      const size_t cSamplesLeafMin,
      BinBase * aAuxiliaryBinsBase,
      double * const pTotalGain
#ifndef NDEBUG
      , const BinBase * const aDebugCopyBinsBase
#endif // NDEBUG
   ) {
      static_assert(IsClassification(cPossibleClasses), "cPossibleClasses needs to be a classification");
      static_assert(cPossibleClasses <= k_cCompilerClassesMax, "We can't have this many items in a data pack.");

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();
      EBM_ASSERT(IsClassification(cRuntimeClasses));
      EBM_ASSERT(cRuntimeClasses <= k_cCompilerClassesMax);

      if(cPossibleClasses == cRuntimeClasses) {
         return PartitionTwoDimensionalBoostingInternal<cPossibleClasses>::Func(
            pBoosterShell,
            pTerm,
            acBins,
            cSamplesLeafMin,
            aAuxiliaryBinsBase,
            pTotalGain
#ifndef NDEBUG
            , aDebugCopyBinsBase
#endif // NDEBUG
         );
      } else {
         return PartitionTwoDimensionalBoostingTarget<cPossibleClasses + 1>::Func(
            pBoosterShell,
            pTerm,
            acBins,
            cSamplesLeafMin,
            aAuxiliaryBinsBase,
            pTotalGain
#ifndef NDEBUG
            , aDebugCopyBinsBase
#endif // NDEBUG
         );
      }
   }
};

template<>
class PartitionTwoDimensionalBoostingTarget<k_cCompilerClassesMax + 1> final {
public:

   PartitionTwoDimensionalBoostingTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm,
      const size_t * const acBins,
      const size_t cSamplesLeafMin,
      BinBase * aAuxiliaryBinsBase,
      double * const pTotalGain
#ifndef NDEBUG
      , const BinBase * const aDebugCopyBinsBase
#endif // NDEBUG
   ) {
      static_assert(IsClassification(k_cCompilerClassesMax), "k_cCompilerClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pBoosterShell->GetBoosterCore()->GetCountClasses()));
      EBM_ASSERT(k_cCompilerClassesMax < pBoosterShell->GetBoosterCore()->GetCountClasses());

      return PartitionTwoDimensionalBoostingInternal<k_dynamicClassification>::Func(
         pBoosterShell,
         pTerm,
         acBins,
         cSamplesLeafMin,
         aAuxiliaryBinsBase,
         pTotalGain
#ifndef NDEBUG
         , aDebugCopyBinsBase
#endif // NDEBUG
      );
   }
};

extern ErrorEbm PartitionTwoDimensionalBoosting(
   BoosterShell * const pBoosterShell,
   const Term * const pTerm,
   const size_t * const acBins,
   const size_t cSamplesLeafMin,
   BinBase * aAuxiliaryBinsBase,
   double * const pTotalGain
#ifndef NDEBUG
   , const BinBase * const aDebugCopyBinsBase
#endif // NDEBUG
) {
   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();

   if(IsClassification(cRuntimeClasses)) {
      return PartitionTwoDimensionalBoostingTarget<2>::Func(
         pBoosterShell,
         pTerm,
         acBins,
         cSamplesLeafMin,
         aAuxiliaryBinsBase,
         pTotalGain
#ifndef NDEBUG
         , aDebugCopyBinsBase
#endif // NDEBUG
      );
   } else {
      EBM_ASSERT(IsRegression(cRuntimeClasses));
      return PartitionTwoDimensionalBoostingInternal<k_regression>::Func(
         pBoosterShell,
         pTerm,
         acBins,
         cSamplesLeafMin,
         aAuxiliaryBinsBase,
         pTotalGain
#ifndef NDEBUG
         , aDebugCopyBinsBase
#endif // NDEBUG
      );
   }
}

} // DEFINED_ZONE_NAME
