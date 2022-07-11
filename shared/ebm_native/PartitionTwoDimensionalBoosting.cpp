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

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
static FloatBig SweepMultiDimensional(
   const Bin<FloatBig, IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aBins,
   const Term * const pTerm,
   size_t * const aiPoint,
   const size_t directionVectorLow,
   const unsigned int iDimensionSweep,
   const size_t cSweepBins,
   const size_t cSamplesRequiredForChildSplitMin,
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   Bin<FloatBig, IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pBinBestAndTemp,
   size_t * const piBestSplit
#ifndef NDEBUG
   , const Bin<FloatBig, IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aBinsDebugCopy
   , const unsigned char * const pBinsEndDebug
#endif // NDEBUG
) {
   constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

   // don't LOG this!  It would create way too much chatter!

   // TODO : optimize this function

   EBM_ASSERT(1 <= pTerm->GetCountSignificantDimensions());
   EBM_ASSERT(iDimensionSweep < pTerm->GetCountSignificantDimensions());
   EBM_ASSERT(0 == (directionVectorLow & (size_t { 1 } << iDimensionSweep)));

   const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
      compilerLearningTypeOrCountTargetClasses,
      runtimeLearningTypeOrCountTargetClasses
   );
   const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cVectorLength);
   EBM_ASSERT(!IsMultiplyError(size_t { 2 }, cBytesPerBin)); // we're accessing allocated memory
   const size_t cBytesPerTwoBins = cBytesPerBin << 1;

   size_t * const piBin = &aiPoint[iDimensionSweep];
   *piBin = 0;
   size_t directionVectorHigh = directionVectorLow | size_t { 1 } << iDimensionSweep;

   EBM_ASSERT(2 <= cSweepBins);

   size_t iBestSplit = 0;

   auto * const pTotalsLow = IndexBin(cBytesPerBin, pBinBestAndTemp, 2);
   ASSERT_BIN_OK(cBytesPerBin, pTotalsLow, pBinsEndDebug);

   auto * const pTotalsHigh = IndexBin(cBytesPerBin, pBinBestAndTemp, 3);
   ASSERT_BIN_OK(cBytesPerBin, pTotalsHigh, pBinsEndDebug);

   EBM_ASSERT(0 < cSamplesRequiredForChildSplitMin);

   FloatBig bestGain = k_illegalGainFloat;
   size_t iBin = 0;
   do {
      *piBin = iBin;

      EBM_ASSERT(2 == pTerm->GetCountSignificantDimensions()); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
      TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, 2>(
         runtimeLearningTypeOrCountTargetClasses,
         pTerm,
         aBins,
         aiPoint,
         directionVectorLow,
         pTotalsLow
#ifndef NDEBUG
         , aBinsDebugCopy
         , pBinsEndDebug
#endif // NDEBUG
         );
      if(LIKELY(cSamplesRequiredForChildSplitMin <= pTotalsLow->GetCountSamples())) {
         EBM_ASSERT(2 == pTerm->GetCountSignificantDimensions()); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
         TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, 2>(
            runtimeLearningTypeOrCountTargetClasses,
            pTerm,
            aBins,
            aiPoint,
            directionVectorHigh,
            pTotalsHigh
#ifndef NDEBUG
            , aBinsDebugCopy
            , pBinsEndDebug
#endif // NDEBUG
         );
         if(LIKELY(cSamplesRequiredForChildSplitMin <= pTotalsHigh->GetCountSamples())) {
            FloatBig gain = 0;
            EBM_ASSERT(0 < pTotalsLow->GetCountSamples());
            EBM_ASSERT(0 < pTotalsHigh->GetCountSamples());

            const FloatBig cLowWeightInBin = pTotalsLow->GetWeight();
            const FloatBig cHighWeightInBin = pTotalsHigh->GetWeight();

            auto * const pHistogramTargetEntryLow = pTotalsLow->GetHistogramTargetEntry();

            auto * const pHistogramTargetEntryHigh = pTotalsHigh->GetHistogramTargetEntry();

            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators 
               // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

               constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;
               const FloatBig gain1 = EbmStats::CalcPartialGain(
                  pHistogramTargetEntryLow[iVector].m_sumGradients, bUseLogitBoost ? pHistogramTargetEntryLow[iVector].GetSumHessians() : cLowWeightInBin);
               EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);
               gain += gain1;
               const FloatBig gain2 = EbmStats::CalcPartialGain(
                  pHistogramTargetEntryHigh[iVector].m_sumGradients, bUseLogitBoost ? pHistogramTargetEntryHigh[iVector].GetSumHessians() : cHighWeightInBin);
               EBM_ASSERT(std::isnan(gain2) || 0 <= gain2);
               gain += gain2;
            }
            EBM_ASSERT(std::isnan(gain) || 0 <= gain); // sumation of positive numbers should be positive

            if(UNLIKELY(/* NaN */ !LIKELY(gain <= bestGain))) {
               // propagate NaNs

               bestGain = gain;
               iBestSplit = iBin;

               ASSERT_BIN_OK(
                  cBytesPerBin,
                  IndexBin(cBytesPerBin, pBinBestAndTemp, 1),
                  pBinsEndDebug
               );
               ASSERT_BIN_OK(
                  cBytesPerBin,
                  IndexBin(cBytesPerBin, pTotalsLow, 1),
                  pBinsEndDebug
               );
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

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class PartitionTwoDimensionalBoostingInternal final {
public:

   PartitionTwoDimensionalBoostingInternal() = delete; // this is a static class.  Do not construct

   WARNING_PUSH
   WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE

   static ErrorEbmType Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm,
      const size_t cSamplesRequiredForChildSplitMin,
      BinBase * aAuxiliaryBinsBase,
      double * const pTotalGain
#ifndef NDEBUG
      , const BinBase * const aBinsBaseDebugCopy
#endif // NDEBUG
   ) {
      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      ErrorEbmType error;
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();

      BinBase * const aBinsBase = pBoosterShell->GetBinBaseBig();
      Tensor * const pInnerTermUpdate =
         pBoosterShell->GetInnerTermUpdate();

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses()
      );

      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cVectorLength);

      auto * aAuxiliaryBins = aAuxiliaryBinsBase->Specialize<FloatBig, bClassification>();
      auto * const aBins = aBinsBase->Specialize<FloatBig, bClassification>();

#ifndef NDEBUG
      const auto * const aBinsDebugCopy = aBinsBaseDebugCopy->Specialize<FloatBig, bClassification>();
#endif // NDEBUG

      size_t aiStart[k_cDimensionsMax];

      EBM_ASSERT(2 == pTerm->GetCountSignificantDimensions());
      size_t iDimensionLoop = 0;
      size_t iDimension1 = 0;
      size_t iDimension2 = 0;
      size_t cBinsDimension1 = 0;
      size_t cBinsDimension2 = 0;
      const TermEntry * pTermEntry = pTerm->GetTermEntries();
      const TermEntry * const pTermEntriesEnd = pTermEntry + pTerm->GetCountDimensions();
      do {
         const size_t cBins = pTermEntry->m_pFeature->GetCountBins();
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
         ++pTermEntry;
      } while(pTermEntriesEnd != pTermEntry);
      EBM_ASSERT(2 <= cBinsDimension1);
      EBM_ASSERT(2 <= cBinsDimension2);

      FloatBig bestGain = k_illegalGainFloat;

      size_t splitFirst1Best;
      size_t splitFirst1LowBest;
      size_t splitFirst1HighBest;

      auto * pTotals1LowLowBest = IndexBin(cBytesPerBin, aAuxiliaryBins, 0);
      auto * pTotals1LowHighBest = IndexBin(cBytesPerBin, aAuxiliaryBins, 1);
      auto * pTotals1HighLowBest = IndexBin(cBytesPerBin, aAuxiliaryBins, 2);
      auto * pTotals1HighHighBest = IndexBin(cBytesPerBin, aAuxiliaryBins, 3);

      EBM_ASSERT(0 < cSamplesRequiredForChildSplitMin);

      LOG_0(TraceLevelVerbose, "PartitionTwoDimensionalBoostingInternal Starting FIRST bin sweep loop");
      size_t iBin1 = 0;
      do {
         aiStart[0] = iBin1;

         size_t splitSecond1LowBest;
         auto * pTotals2LowLowBest = IndexBin(cBytesPerBin, aAuxiliaryBins, 4);
         auto * pTotals2LowHighBest = IndexBin(cBytesPerBin, aAuxiliaryBins, 5);
         const FloatBig gain1 = SweepMultiDimensional<compilerLearningTypeOrCountTargetClasses>(
            aBins,
            pTerm,
            aiStart,
            0x0,
            1,
            cBinsDimension2,
            cSamplesRequiredForChildSplitMin,
            runtimeLearningTypeOrCountTargetClasses,
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
            auto * pTotals2HighLowBest = IndexBin(cBytesPerBin, aAuxiliaryBins, 8);
            auto * pTotals2HighHighBest = IndexBin(cBytesPerBin, aAuxiliaryBins, 9);
            const FloatBig gain2 = SweepMultiDimensional<compilerLearningTypeOrCountTargetClasses>(
               aBins,
               pTerm,
               aiStart,
               0x1,
               1,
               cBinsDimension2,
               cSamplesRequiredForChildSplitMin,
               runtimeLearningTypeOrCountTargetClasses,
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

                  pTotals1LowLowBest->Copy(*pTotals2LowLowBest, cVectorLength);
                  pTotals1LowHighBest->Copy(*pTotals2LowHighBest, cVectorLength);
                  pTotals1HighLowBest->Copy(*pTotals2HighLowBest, cVectorLength);
                  pTotals1HighHighBest->Copy(*pTotals2HighHighBest, cVectorLength);
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

      auto * pTotals2LowLowBest = IndexBin(cBytesPerBin, aAuxiliaryBins, 12);
      auto * pTotals2LowHighBest = IndexBin(cBytesPerBin, aAuxiliaryBins, 13);
      auto * pTotals2HighLowBest = IndexBin(cBytesPerBin, aAuxiliaryBins, 14);
      auto * pTotals2HighHighBest = IndexBin(cBytesPerBin, aAuxiliaryBins, 15);

      LOG_0(TraceLevelVerbose, "PartitionTwoDimensionalBoostingInternal Starting SECOND bin sweep loop");
      size_t iBin2 = 0;
      do {
         aiStart[1] = iBin2;

         size_t splitSecond2LowBest;
         auto * pTotals1LowLowBestInner = IndexBin(cBytesPerBin, aAuxiliaryBins, 16);
         auto * pTotals1LowHighBestInner = IndexBin(cBytesPerBin, aAuxiliaryBins, 17);
         const FloatBig gain1 = SweepMultiDimensional<compilerLearningTypeOrCountTargetClasses>(
            aBins,
            pTerm,
            aiStart,
            0x0,
            0,
            cBinsDimension1,
            cSamplesRequiredForChildSplitMin,
            runtimeLearningTypeOrCountTargetClasses,
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
            auto * pTotals1HighLowBestInner = IndexBin(cBytesPerBin, aAuxiliaryBins, 20);
            auto * pTotals1HighHighBestInner = IndexBin(cBytesPerBin, aAuxiliaryBins, 21);
            const FloatBig gain2 = SweepMultiDimensional<compilerLearningTypeOrCountTargetClasses>(
               aBins,
               pTerm,
               aiStart,
               0x2,
               0,
               cBinsDimension1,
               cSamplesRequiredForChildSplitMin,
               runtimeLearningTypeOrCountTargetClasses,
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

                  pTotals2LowLowBest->Copy(*pTotals1LowLowBestInner, cVectorLength);
                  pTotals2LowHighBest->Copy(*pTotals1LowHighBestInner, cVectorLength);
                  pTotals2HighLowBest->Copy(*pTotals1HighLowBestInner, cVectorLength);
                  pTotals2HighHighBest->Copy(*pTotals1HighHighBestInner, cVectorLength);

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
      LOG_0(TraceLevelVerbose, "PartitionTwoDimensionalBoostingInternal Done sweep loops");

      EBM_ASSERT(std::isnan(bestGain) || k_illegalGainFloat == bestGain || 0 <= bestGain);

      // the bin before the aAuxiliaryBinsBase is the last summation bin of aBinsBase, 
      // which contains the totals of all bins
      const auto * const pTotal =
         reinterpret_cast<const Bin<FloatBig, bClassification> *>(
            reinterpret_cast<const char *>(aAuxiliaryBinsBase) - cBytesPerBin);

      ASSERT_BIN_OK(cBytesPerBin, pTotal, pBoosterShell->GetBinsBigEndDebug());

      const auto * const pHistogramTargetEntryTotal = pTotal->GetHistogramTargetEntry();

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
            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               // TODO : we can make this faster by doing the division in CalcPartialGain after we add all the numerators 
               // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

               constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;
               const FloatBig gain1 = EbmStats::CalcPartialGain(
                  pHistogramTargetEntryTotal[iVector].m_sumGradients,
                  bUseLogitBoost ? pHistogramTargetEntryTotal[iVector].GetSumHessians() : weightAll
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
                        error = pInnerTermUpdate->EnsureScoreCapacity(cVectorLength * 6);
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
                        error = pInnerTermUpdate->EnsureScoreCapacity(cVectorLength * 6);
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

                        error = pInnerTermUpdate->EnsureScoreCapacity(cVectorLength * 4);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        pInnerTermUpdate->GetSplitPointer(iDimension1)[0] = splitFirst2LowBest;
                     }

                     auto * const pHistogramTargetEntryTotals2LowLowBest = pTotals2LowLowBest->GetHistogramTargetEntry();
                     auto * const pHistogramTargetEntryTotals2LowHighBest = pTotals2LowHighBest->GetHistogramTargetEntry();
                     auto * const pHistogramTargetEntryTotals2HighLowBest = pTotals2HighLowBest->GetHistogramTargetEntry();
                     auto * const pHistogramTargetEntryTotals2HighHighBest = pTotals2HighHighBest->GetHistogramTargetEntry();

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
                     FloatBig zeroLogit0 = 0;
                     FloatBig zeroLogit1 = 0;
                     FloatBig zeroLogit2 = 0;
                     FloatBig zeroLogit3 = 0;
#endif // ZERO_FIRST_MULTICLASS_LOGIT

                     for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
                        FloatBig predictionLowLow;
                        FloatBig predictionLowHigh;
                        FloatBig predictionHighLow;
                        FloatBig predictionHighHigh;

                        if(bClassification) {
                           predictionLowLow = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals2LowLowBest[iVector].m_sumGradients,
                              pHistogramTargetEntryTotals2LowLowBest[iVector].GetSumHessians()
                           );
                           predictionLowHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals2LowHighBest[iVector].m_sumGradients,
                              pHistogramTargetEntryTotals2LowHighBest[iVector].GetSumHessians()
                           );
                           predictionHighLow = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals2HighLowBest[iVector].m_sumGradients,
                              pHistogramTargetEntryTotals2HighLowBest[iVector].GetSumHessians()
                           );
                           predictionHighHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals2HighHighBest[iVector].m_sumGradients,
                              pHistogramTargetEntryTotals2HighHighBest[iVector].GetSumHessians()
                           );

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
                           if(IsMulticlass(compilerLearningTypeOrCountTargetClasses)) {
                              if(size_t { 0 } == iVector) {
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
                           EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
                           predictionLowLow = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals2LowLowBest[iVector].m_sumGradients,
                              pTotals2LowLowBest->GetWeight()
                           );
                           predictionLowHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals2LowHighBest[iVector].m_sumGradients,
                              pTotals2LowHighBest->GetWeight()
                           );
                           predictionHighLow = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals2HighLowBest[iVector].m_sumGradients,
                              pTotals2HighLowBest->GetWeight()
                           );
                           predictionHighHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals2HighHighBest[iVector].m_sumGradients,
                              pTotals2HighHighBest->GetWeight()
                           );
                        }

                        FloatFast * const aUpdateScores = pInnerTermUpdate->GetScoresPointer();
                        if(splitFirst2LowBest < splitFirst2HighBest) {
                           aUpdateScores[0 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionLowLow);
                           aUpdateScores[1 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionLowHigh);
                           aUpdateScores[2 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionLowHigh);
                           aUpdateScores[3 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionHighLow);
                           aUpdateScores[4 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionHighLow);
                           aUpdateScores[5 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionHighHigh);
                        } else if(splitFirst2HighBest < splitFirst2LowBest) {
                           aUpdateScores[0 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionLowLow);
                           aUpdateScores[1 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionLowLow);
                           aUpdateScores[2 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionLowHigh);
                           aUpdateScores[3 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionHighLow);
                           aUpdateScores[4 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionHighHigh);
                           aUpdateScores[5 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionHighHigh);
                        } else {
                           aUpdateScores[0 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionLowLow);
                           aUpdateScores[1 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionLowHigh);
                           aUpdateScores[2 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionHighLow);
                           aUpdateScores[3 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionHighHigh);
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
                        error = pInnerTermUpdate->EnsureScoreCapacity(cVectorLength * 6);
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
                        error = pInnerTermUpdate->EnsureScoreCapacity(cVectorLength * 6);
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
                        error = pInnerTermUpdate->EnsureScoreCapacity(cVectorLength * 4);
                        if(Error_None != error) {
                           // already logged
                           return error;
                        }
                        pInnerTermUpdate->GetSplitPointer(iDimension2)[0] = splitFirst1LowBest;
                     }

                     auto * const pHistogramTargetEntryTotals1LowLowBest = pTotals1LowLowBest->GetHistogramTargetEntry();
                     auto * const pHistogramTargetEntryTotals1LowHighBest = pTotals1LowHighBest->GetHistogramTargetEntry();
                     auto * const pHistogramTargetEntryTotals1HighLowBest = pTotals1HighLowBest->GetHistogramTargetEntry();
                     auto * const pHistogramTargetEntryTotals1HighHighBest = pTotals1HighHighBest->GetHistogramTargetEntry();

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
                     FloatBig zeroLogit0 = 0;
                     FloatBig zeroLogit1 = 0;
                     FloatBig zeroLogit2 = 0;
                     FloatBig zeroLogit3 = 0;
#endif // ZERO_FIRST_MULTICLASS_LOGIT

                     for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
                        FloatBig predictionLowLow;
                        FloatBig predictionLowHigh;
                        FloatBig predictionHighLow;
                        FloatBig predictionHighHigh;

                        if(bClassification) {
                           predictionLowLow = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals1LowLowBest[iVector].m_sumGradients,
                              pHistogramTargetEntryTotals1LowLowBest[iVector].GetSumHessians()
                           );
                           predictionLowHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals1LowHighBest[iVector].m_sumGradients,
                              pHistogramTargetEntryTotals1LowHighBest[iVector].GetSumHessians()
                           );
                           predictionHighLow = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals1HighLowBest[iVector].m_sumGradients,
                              pHistogramTargetEntryTotals1HighLowBest[iVector].GetSumHessians()
                           );
                           predictionHighHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals1HighHighBest[iVector].m_sumGradients,
                              pHistogramTargetEntryTotals1HighHighBest[iVector].GetSumHessians()
                           );

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
                           if(IsMulticlass(compilerLearningTypeOrCountTargetClasses)) {
                              if(size_t { 0 } == iVector) {
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
                           EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
                           predictionLowLow = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals1LowLowBest[iVector].m_sumGradients,
                              pTotals1LowLowBest->GetWeight()
                           );
                           predictionLowHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals1LowHighBest[iVector].m_sumGradients,
                              pTotals1LowHighBest->GetWeight()
                           );
                           predictionHighLow = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals1HighLowBest[iVector].m_sumGradients,
                              pTotals1HighLowBest->GetWeight()
                           );
                           predictionHighHigh = EbmStats::ComputeSinglePartitionUpdate(
                              pHistogramTargetEntryTotals1HighHighBest[iVector].m_sumGradients,
                              pTotals1HighHighBest->GetWeight()
                           );
                        }
                        FloatFast * const aUpdateScores = pInnerTermUpdate->GetScoresPointer();
                        if(splitFirst1LowBest < splitFirst1HighBest) {
                           aUpdateScores[0 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionLowLow);
                           aUpdateScores[1 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionHighLow);
                           aUpdateScores[2 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionLowHigh);
                           aUpdateScores[3 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionHighLow);
                           aUpdateScores[4 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionLowHigh);
                           aUpdateScores[5 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionHighHigh);
                        } else if(splitFirst1HighBest < splitFirst1LowBest) {
                           aUpdateScores[0 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionLowLow);
                           aUpdateScores[1 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionHighLow);
                           aUpdateScores[2 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionLowLow);
                           aUpdateScores[3 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionHighHigh);
                           aUpdateScores[4 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionLowHigh);
                           aUpdateScores[5 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionHighHigh);
                        } else {
                           aUpdateScores[0 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionLowLow);
                           aUpdateScores[1 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionHighLow);
                           aUpdateScores[2 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionLowHigh);
                           aUpdateScores[3 * cVectorLength + iVector] = SafeConvertFloat<FloatFast>(predictionHighHigh);
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
      const ErrorEbmType errorDebug1 =
#endif // NDEBUG
         pInnerTermUpdate->SetCountSplits(iDimension1, 0);
      // we can't fail since we're setting this to zero, so no allocations.  We don't in fact need the split array at all
      EBM_ASSERT(Error_None == errorDebug1);

#ifndef NDEBUG
      const ErrorEbmType errorDebug2 =
#endif // NDEBUG
         pInnerTermUpdate->SetCountSplits(iDimension2, 0);
      // we can't fail since we're setting this to zero, so no allocations.  We don't in fact need the split array at all
      EBM_ASSERT(Error_None == errorDebug2);

      // we don't need to call pInnerTermUpdate->EnsureScoreCapacity, 
      // since our value capacity would be 1, which is pre-allocated

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
      FloatBig zeroLogit = 0;
#endif // ZERO_FIRST_MULTICLASS_LOGIT

      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         FloatBig update;
         if(bClassification) {
            update = EbmStats::ComputeSinglePartitionUpdate(
               pHistogramTargetEntryTotal[iVector].m_sumGradients,
               pHistogramTargetEntryTotal[iVector].GetSumHessians()
            );

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
            if(IsMulticlass(compilerLearningTypeOrCountTargetClasses)) {
               if(size_t { 0 } == iVector) {
                  zeroLogit = update;
               }
               update -= zeroLogit;
            }
#endif // ZERO_FIRST_MULTICLASS_LOGIT

         } else {
            EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
            update = EbmStats::ComputeSinglePartitionUpdate(
               pHistogramTargetEntryTotal[iVector].m_sumGradients,
               weightAll
            );
         }

         FloatFast * const aUpdateScores = pInnerTermUpdate->GetScoresPointer();
         aUpdateScores[iVector] = SafeConvertFloat<FloatFast>(update);
      }
      return Error_None;
   }
   WARNING_POP
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class PartitionTwoDimensionalBoostingTarget final {
public:

   PartitionTwoDimensionalBoostingTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static ErrorEbmType Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm,
      const size_t cSamplesRequiredForChildSplitMin,
      BinBase * aAuxiliaryBinsBase,
      double * const pTotalGain
#ifndef NDEBUG
      , const BinBase * const aBinsBaseDebugCopy
#endif // NDEBUG
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         return PartitionTwoDimensionalBoostingInternal<compilerLearningTypeOrCountTargetClassesPossible>::Func(
            pBoosterShell,
            pTerm,
            cSamplesRequiredForChildSplitMin,
            aAuxiliaryBinsBase,
            pTotalGain
#ifndef NDEBUG
            , aBinsBaseDebugCopy
#endif // NDEBUG
         );
      } else {
         return PartitionTwoDimensionalBoostingTarget<compilerLearningTypeOrCountTargetClassesPossible + 1>::Func(
            pBoosterShell,
            pTerm,
            cSamplesRequiredForChildSplitMin,
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
class PartitionTwoDimensionalBoostingTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   PartitionTwoDimensionalBoostingTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static ErrorEbmType Func(
      BoosterShell * const pBoosterShell,
      const Term * const pTerm,
      const size_t cSamplesRequiredForChildSplitMin,
      BinBase * aAuxiliaryBinsBase,
      double * const pTotalGain
#ifndef NDEBUG
      , const BinBase * const aBinsBaseDebugCopy
#endif // NDEBUG
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pBoosterShell->GetBoosterCore()->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pBoosterShell->GetBoosterCore()->GetRuntimeLearningTypeOrCountTargetClasses());

      return PartitionTwoDimensionalBoostingInternal<k_dynamicClassification>::Func(
         pBoosterShell,
         pTerm,
         cSamplesRequiredForChildSplitMin,
         aAuxiliaryBinsBase,
         pTotalGain
#ifndef NDEBUG
         , aBinsBaseDebugCopy
#endif // NDEBUG
      );
   }
};

extern ErrorEbmType PartitionTwoDimensionalBoosting(
   BoosterShell * const pBoosterShell,
   const Term * const pTerm,
   const size_t cSamplesRequiredForChildSplitMin,
   BinBase * aAuxiliaryBinsBase,
   double * const pTotalGain
#ifndef NDEBUG
   , const BinBase * const aBinsBaseDebugCopy
#endif // NDEBUG
) {
   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();

   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      return PartitionTwoDimensionalBoostingTarget<2>::Func(
         pBoosterShell,
         pTerm,
         cSamplesRequiredForChildSplitMin,
         aAuxiliaryBinsBase,
         pTotalGain
#ifndef NDEBUG
         , aBinsBaseDebugCopy
#endif // NDEBUG
      );
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      return PartitionTwoDimensionalBoostingInternal<k_regression>::Func(
         pBoosterShell,
         pTerm,
         cSamplesRequiredForChildSplitMin,
         aAuxiliaryBinsBase,
         pTotalGain
#ifndef NDEBUG
         , aBinsBaseDebugCopy
#endif // NDEBUG
      );
   }
}

} // DEFINED_ZONE_NAME
