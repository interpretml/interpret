// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "CompressibleTensor.hpp"
#include "ebm_stats.hpp"

#include "Feature.hpp"
#include "FeatureGroup.hpp"

#include "HistogramTargetEntry.hpp"
#include "HistogramBucket.hpp"

#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

#include "TensorTotalsSum.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<ptrdiff_t cCompilerClasses>
static FloatBig SweepMultiDimensional(
   const Bin<FloatBig, IsClassification(cCompilerClasses)> * const aBins,
   const size_t cRealDimensions,
   const size_t * const acBins,
   size_t * const aiPoint,
   const size_t directionVectorLow,
   const unsigned int iDimensionSweep,
   const size_t cSweepBins,
   const size_t cSamplesLeafMin,
   const ptrdiff_t cRuntimeClasses,
   Bin<FloatBig, IsClassification(cCompilerClasses)> * const pBinBestAndTemp,
   size_t * const piBestSplit
#ifndef NDEBUG
   , const Bin<FloatBig, IsClassification(cCompilerClasses)> * const aBinsDebugCopy
   , const unsigned char * const pBinsEndDebug
#endif // NDEBUG
) {
   constexpr bool bClassification = IsClassification(cCompilerClasses);

   // don't LOG this!  It would create way too much chatter!

   // TODO : optimize this function

   EBM_ASSERT(1 <= cRealDimensions);
   EBM_ASSERT(iDimensionSweep < cRealDimensions);
   EBM_ASSERT(0 == (directionVectorLow & (size_t { 1 } << iDimensionSweep)));

   const ptrdiff_t cClasses = GET_COUNT_CLASSES(
      cCompilerClasses,
      cRuntimeClasses
   );
   const size_t cScores = GetCountScores(cClasses);
   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we're accessing allocated memory
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);
   EBM_ASSERT(!IsMultiplyError(size_t { 2 }, cBytesPerBin)); // we're accessing allocated memory
   const size_t cBytesPerTwoBins = cBytesPerBin << 1;

   size_t * const piBin = &aiPoint[iDimensionSweep];
   *piBin = 0;
   size_t directionVectorHigh = directionVectorLow | size_t { 1 } << iDimensionSweep;

   EBM_ASSERT(2 <= cSweepBins);

   size_t iBestSplit = 0;

   auto * const pTotalsLow = IndexBin(pBinBestAndTemp, cBytesPerBin * 2);
   ASSERT_BIN_OK(cBytesPerBin, pTotalsLow, pBinsEndDebug);

   auto * const pTotalsHigh = IndexBin(pBinBestAndTemp, cBytesPerBin * 3);
   ASSERT_BIN_OK(cBytesPerBin, pTotalsHigh, pBinsEndDebug);

   EBM_ASSERT(0 < cSamplesLeafMin);

   FloatBig bestGain = k_illegalGainFloat;
   size_t iBin = 0;
   do {
      *piBin = iBin;

      EBM_ASSERT(2 == cRealDimensions); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
      TensorTotalsSum<cCompilerClasses, 2>(
         cRuntimeClasses,
         cRealDimensions,
         acBins,
         aBins,
         aiPoint,
         directionVectorLow,
         pTotalsLow
#ifndef NDEBUG
         , aBinsDebugCopy
         , pBinsEndDebug
#endif // NDEBUG
         );
      if(LIKELY(cSamplesLeafMin <= pTotalsLow->GetCountSamples())) {
         EBM_ASSERT(2 == cRealDimensions); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
         TensorTotalsSum<cCompilerClasses, 2>(
            cRuntimeClasses,
            cRealDimensions,
            acBins,
            aBins,
            aiPoint,
            directionVectorHigh,
            pTotalsHigh
#ifndef NDEBUG
            , aBinsDebugCopy
            , pBinsEndDebug
#endif // NDEBUG
         );
         if(LIKELY(cSamplesLeafMin <= pTotalsHigh->GetCountSamples())) {
            FloatBig gain = 0;
            EBM_ASSERT(0 < pTotalsLow->GetCountSamples());
            EBM_ASSERT(0 < pTotalsHigh->GetCountSamples());

            const FloatBig cLowWeight = pTotalsLow->GetWeight();
            const FloatBig cHighWeight = pTotalsHigh->GetWeight();

            auto * const pGradientPairLow = pTotalsLow->GetGradientPairs();

            auto * const pGradientPairHigh = pTotalsHigh->GetGradientPairs();

            for(size_t iScore = 0; iScore < cScores; ++iScore) {
               // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators 
               // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

               constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;
               const FloatBig gain1 = EbmStats::CalcPartialGain(
                  pGradientPairLow[iScore].m_sumGradients, bUseLogitBoost ? pGradientPairLow[iScore].GetSumHessians() : cLowWeight);
               EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);
               gain += gain1;
               const FloatBig gain2 = EbmStats::CalcPartialGain(
                  pGradientPairHigh[iScore].m_sumGradients, bUseLogitBoost ? pGradientPairHigh[iScore].GetSumHessians() : cHighWeight);
               EBM_ASSERT(std::isnan(gain2) || 0 <= gain2);
               gain += gain2;
            }
            EBM_ASSERT(std::isnan(gain) || 0 <= gain); // sumation of positive numbers should be positive

            if(UNLIKELY(/* NaN */ !LIKELY(gain <= bestGain))) {
               // propagate NaNs

               bestGain = gain;
               iBestSplit = iBin;

               ASSERT_BIN_OK(cBytesPerBin, IndexBin(pBinBestAndTemp, cBytesPerBin), pBinsEndDebug);
               ASSERT_BIN_OK(cBytesPerBin, IndexBin(pTotalsLow, cBytesPerBin), pBinsEndDebug);
               memcpy(pBinBestAndTemp, pTotalsLow, cBytesPerTwoBins); // this copies both pTotalsLow and pTotalsHigh
            } else {
               EBM_ASSERT(!std::isnan(gain));
            }
         }
      }
      ++iBin;
   } while(iBin < cSweepBins - 1);
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

   static ErrorEbm Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm,
      const size_t * const acBins,
      const size_t cSamplesLeafMin,
      BinBase * aAuxiliaryBinsBase,
      double * const pTotalGain
#ifndef NDEBUG
      , const BinBase * const aBinsBaseDebugCopy
#endif // NDEBUG
   ) {
      constexpr bool bClassification = IsClassification(cCompilerClasses);

      ErrorEbm error;
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();

      BinBase * const aBinsBase = pBoosterShell->GetBinBaseBig();
      Tensor * const pInnerTermUpdate = pBoosterShell->GetInnerTermUpdate();

      const ptrdiff_t cRuntimeClasses = pBoosterCore->GetCountClasses();

      const ptrdiff_t cClasses = GET_COUNT_CLASSES(cCompilerClasses, cRuntimeClasses);

      const size_t cScores = GetCountScores(cClasses);
      const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

      auto * aAuxiliaryBins = aAuxiliaryBinsBase->Specialize<FloatBig, bClassification>();
      auto * const aBins = aBinsBase->Specialize<FloatBig, bClassification>();

#ifndef NDEBUG
      const auto * const aBinsDebugCopy = aBinsBaseDebugCopy->Specialize<FloatBig, bClassification>();
#endif // NDEBUG

      size_t aiStart[k_cDimensionsMax];

      EBM_ASSERT(2 == pTerm->GetCountRealDimensions());
      size_t iDimensionLoop = 0;
      size_t iDimension1 = 0;
      size_t iDimension2 = 0;
      size_t cBinsDimension1 = 0;
      size_t cBinsDimension2 = 0;
      const Feature * const * ppFeature = pTerm->GetFeatures();
      const Feature * const * const ppFeaturesEnd = &ppFeature[pTerm->GetCountDimensions()];
      do {
         const Feature * const pFeature = *ppFeature;
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
         const FloatBig gain1 = SweepMultiDimensional<cCompilerClasses>(
            aBins,
            pTerm->GetCountRealDimensions(),
            acBins,
            aiStart,
            0x0,
            1,
            cBinsDimension2,
            cSamplesLeafMin,
            cRuntimeClasses,
            pTotals2LowLowBest,
            &splitSecond1LowBest
#ifndef NDEBUG
            , aBinsDebugCopy
            , pBoosterShell->GetBinsBigEndDebug()
#endif // NDEBUG
            );

         if(LIKELY(/* NaN */ !UNLIKELY(gain1 < 0))) {
            EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);

            size_t splitSecond1HighBest;
            auto * pTotals2HighLowBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 8);
            auto * pTotals2HighHighBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 9);
            const FloatBig gain2 = SweepMultiDimensional<cCompilerClasses>(
               aBins,
               pTerm->GetCountRealDimensions(),
               acBins,
               aiStart,
               0x1,
               1,
               cBinsDimension2,
               cSamplesLeafMin,
               cRuntimeClasses,
               pTotals2HighLowBest,
               &splitSecond1HighBest
#ifndef NDEBUG
               , aBinsDebugCopy
               , pBoosterShell->GetBinsBigEndDebug()
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

                  pTotals1LowLowBest->Copy(*pTotals2LowLowBest, cScores);
                  pTotals1LowHighBest->Copy(*pTotals2LowHighBest, cScores);
                  pTotals1HighLowBest->Copy(*pTotals2HighLowBest, cScores);
                  pTotals1HighHighBest->Copy(*pTotals2HighHighBest, cScores);
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
         const FloatBig gain1 = SweepMultiDimensional<cCompilerClasses>(
            aBins,
            pTerm->GetCountRealDimensions(),
            acBins,
            aiStart,
            0x0,
            0,
            cBinsDimension1,
            cSamplesLeafMin,
            cRuntimeClasses,
            pTotals1LowLowBestInner,
            &splitSecond2LowBest
#ifndef NDEBUG
            , aBinsDebugCopy
            , pBoosterShell->GetBinsBigEndDebug()
#endif // NDEBUG
            );

         if(LIKELY(/* NaN */ !UNLIKELY(gain1 < 0))) {
            EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);

            size_t splitSecond2HighBest;
            auto * pTotals1HighLowBestInner = IndexBin(aAuxiliaryBins, cBytesPerBin * 20);
            auto * pTotals1HighHighBestInner = IndexBin(aAuxiliaryBins, cBytesPerBin * 21);
            const FloatBig gain2 = SweepMultiDimensional<cCompilerClasses>(
               aBins,
               pTerm->GetCountRealDimensions(),
               acBins,
               aiStart,
               0x2,
               0,
               cBinsDimension1,
               cSamplesLeafMin,
               cRuntimeClasses,
               pTotals1HighLowBestInner,
               &splitSecond2HighBest
#ifndef NDEBUG
               , aBinsDebugCopy
               , pBoosterShell->GetBinsBigEndDebug()
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

                  pTotals2LowLowBest->Copy(*pTotals1LowLowBestInner, cScores);
                  pTotals2LowHighBest->Copy(*pTotals1LowHighBestInner, cScores);
                  pTotals2HighLowBest->Copy(*pTotals1HighLowBestInner, cScores);
                  pTotals2HighHighBest->Copy(*pTotals1HighHighBestInner, cScores);

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

      // the bin before the aAuxiliaryBinsBase is the last summation bin of aBinsBase, 
      // which contains the totals of all bins
      const auto * const pTotal =
         reinterpret_cast<const Bin<FloatBig, bClassification> *>(
            reinterpret_cast<const char *>(aAuxiliaryBinsBase) - cBytesPerBin);

      ASSERT_BIN_OK(cBytesPerBin, pTotal, pBoosterShell->GetBinsBigEndDebug());

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

               constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;
               const FloatBig gain1 = EbmStats::CalcPartialGain(
                  pGradientPairTotal[iScore].m_sumGradients,
                  bUseLogitBoost ? pGradientPairTotal[iScore].GetSumHessians() : weightAll
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

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
                     FloatBig zeroLogit0 = 0;
                     FloatBig zeroLogit1 = 0;
                     FloatBig zeroLogit2 = 0;
                     FloatBig zeroLogit3 = 0;
#endif // ZERO_FIRST_MULTICLASS_LOGIT

                     for(size_t iScore = 0; iScore < cScores; ++iScore) {
                        FloatBig predictionLowLow;
                        FloatBig predictionLowHigh;
                        FloatBig predictionHighLow;
                        FloatBig predictionHighHigh;

                        if(bClassification) {
                           predictionLowLow = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals2LowLowBest[iScore].m_sumGradients,
                              pGradientPairTotals2LowLowBest[iScore].GetSumHessians()
                           );
                           predictionLowHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals2LowHighBest[iScore].m_sumGradients,
                              pGradientPairTotals2LowHighBest[iScore].GetSumHessians()
                           );
                           predictionHighLow = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals2HighLowBest[iScore].m_sumGradients,
                              pGradientPairTotals2HighLowBest[iScore].GetSumHessians()
                           );
                           predictionHighHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals2HighHighBest[iScore].m_sumGradients,
                              pGradientPairTotals2HighHighBest[iScore].GetSumHessians()
                           );

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
                           if(IsMulticlass(cCompilerClasses)) {
                              if(size_t { 0 } == iScore) {
                                 zeroLogit0 = predictionLowLow;
                                 zeroLogit1 = predictionLowHigh;
                                 zeroLogit2 = predictionHighLow;
                                 zeroLogit3 = predictionHighHigh;
                              }
                              predictionLowLow -= zeroLogit0;
                              predictionLowHigh -= zeroLogit1;
                              predictionHighLow -= zeroLogit2;
                              predictionHighHigh -= zeroLogit3;
                           }
#endif // ZERO_FIRST_MULTICLASS_LOGIT

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

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
                     FloatBig zeroLogit0 = 0;
                     FloatBig zeroLogit1 = 0;
                     FloatBig zeroLogit2 = 0;
                     FloatBig zeroLogit3 = 0;
#endif // ZERO_FIRST_MULTICLASS_LOGIT

                     for(size_t iScore = 0; iScore < cScores; ++iScore) {
                        FloatBig predictionLowLow;
                        FloatBig predictionLowHigh;
                        FloatBig predictionHighLow;
                        FloatBig predictionHighHigh;

                        if(bClassification) {
                           predictionLowLow = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals1LowLowBest[iScore].m_sumGradients,
                              pGradientPairTotals1LowLowBest[iScore].GetSumHessians()
                           );
                           predictionLowHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals1LowHighBest[iScore].m_sumGradients,
                              pGradientPairTotals1LowHighBest[iScore].GetSumHessians()
                           );
                           predictionHighLow = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals1HighLowBest[iScore].m_sumGradients,
                              pGradientPairTotals1HighLowBest[iScore].GetSumHessians()
                           );
                           predictionHighHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pGradientPairTotals1HighHighBest[iScore].m_sumGradients,
                              pGradientPairTotals1HighHighBest[iScore].GetSumHessians()
                           );

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
                           if(IsMulticlass(cCompilerClasses)) {
                              if(size_t { 0 } == iScore) {
                                 zeroLogit0 = predictionLowLow;
                                 zeroLogit1 = predictionLowHigh;
                                 zeroLogit2 = predictionHighLow;
                                 zeroLogit3 = predictionHighHigh;
                              }
                              predictionLowLow -= zeroLogit0;
                              predictionLowHigh -= zeroLogit1;
                              predictionHighLow -= zeroLogit2;
                              predictionHighHigh -= zeroLogit3;
                           }
#endif // ZERO_FIRST_MULTICLASS_LOGIT
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

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
      FloatBig zeroLogit = 0;
#endif // ZERO_FIRST_MULTICLASS_LOGIT

      for(size_t iScore = 0; iScore < cScores; ++iScore) {
         FloatBig update;
         if(bClassification) {
            update = EbmStats::ComputeSinglePartitionUpdate(
               pGradientPairTotal[iScore].m_sumGradients,
               pGradientPairTotal[iScore].GetSumHessians()
            );

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
            if(IsMulticlass(cCompilerClasses)) {
               if(size_t { 0 } == iScore) {
                  zeroLogit = update;
               }
               update -= zeroLogit;
            }
#endif // ZERO_FIRST_MULTICLASS_LOGIT

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

   INLINE_ALWAYS static ErrorEbm Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm,
      const size_t * const acBins,
      const size_t cSamplesLeafMin,
      BinBase * aAuxiliaryBinsBase,
      double * const pTotalGain
#ifndef NDEBUG
      , const BinBase * const aBinsBaseDebugCopy
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
            , aBinsBaseDebugCopy
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
            , aBinsBaseDebugCopy
#endif // NDEBUG
         );
      }
   }
};

template<>
class PartitionTwoDimensionalBoostingTarget<k_cCompilerClassesMax + 1> final {
public:

   PartitionTwoDimensionalBoostingTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static ErrorEbm Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm,
      const size_t * const acBins,
      const size_t cSamplesLeafMin,
      BinBase * aAuxiliaryBinsBase,
      double * const pTotalGain
#ifndef NDEBUG
      , const BinBase * const aBinsBaseDebugCopy
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
         , aBinsBaseDebugCopy
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
   , const BinBase * const aBinsBaseDebugCopy
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
         , aBinsBaseDebugCopy
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
         , aBinsBaseDebugCopy
#endif // NDEBUG
      );
   }
}

} // DEFINED_ZONE_NAME
