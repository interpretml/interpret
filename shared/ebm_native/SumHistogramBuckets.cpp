// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "Feature.hpp"
#include "FeatureGroup.hpp"

#include "HistogramTargetEntry.hpp"
#include "HistogramBucket.hpp"

#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class SumAllBinsInternal final {
public:

   SumAllBinsInternal() = delete; // this is a static class.  Do not construct

   static void Func(
      BoosterShell * const pBoosterShell,
      const size_t cBins
#ifndef NDEBUG
      , const size_t cSamplesTotal
      , const FloatBig weightTotal
#endif // NDEBUG
   ) {
      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();

      GradientPairBase * const aSumAllGradientPairsBase = pBoosterShell->GetSumAllGradientPairs();
      BinBase * const aBinsBase = pBoosterShell->GetBinBaseBig();

      const auto * const aBins = aBinsBase->Specialize<FloatBig, bClassification>();
      auto * const aSumAllGradientPairs = aSumAllGradientPairsBase->Specialize<FloatBig, bClassification>();

      EBM_ASSERT(2 <= cBins); // we pre-filter out features with only one bin

#ifndef NDEBUG
      size_t cSamplesTotalDebug = 0;
      FloatBig weightTotalDebug = 0;
#endif // NDEBUG

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cScores = GetCountScores(learningTypeOrCountTargetClasses);
      EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we're accessing allocated memory
      const size_t cBytesPerBin = GetBinSize<FloatBig>(bClassification, cScores);

      const auto * pCopyFrom = aBins;
      const auto * pCopyFromEnd = IndexBin(cBytesPerBin, aBins, cBins);

      // we do a lot more work in the PartitionOneDimensionalBoosting function per target entry, so if we can compress it by any amount, then it will probably be a win
      // for bin arrays that have a small set of labels, this loop will be fast and result in no movements.  For bin arrays that are long 
      // and have many different labels, we are more likley to find bins with zero items, and that's where we get a win by compressing it down to just the 
      // non-zero bins, even though this requires one more member variable in the bins array
      do {
         ASSERT_BIN_OK(cBytesPerBin, pCopyFrom, pBoosterShell->GetBinsBigEndDebug());
#ifndef NDEBUG
         cSamplesTotalDebug += pCopyFrom->GetCountSamples();
         weightTotalDebug += pCopyFrom->GetWeight();
#endif // NDEBUG

         const auto * pGradientPair = pCopyFrom->GetGradientPairs();

         for(size_t iScore = 0; iScore < cScores; ++iScore) {
            // when building a tree, we start from one end and sweep to the other.  In order to caluculate
            // gain on both sides, we need the sum on both sides, which means when starting from one end
            // we need to know the sum of everything on the other side, so we need to calculate this sum
            // somewhere.  If we have a continuous value and bin it such that many samples are in the same bin
            // then it makes sense to calculate the total of all bins after generating the histograms of the bins
            // since then we just need to sum N bins (where N is the number of bins) vs the # of samples.
            // There is one case though where we might want to calculate the sum while looping the samples,
            // and that is if almost all bins have either 0 or 1 samples, which would happen if we didn't bin at all
            // beforehand.  We'll still want this per-bin sumation though since it's unlikley that all data
            // will be continuous in an ML problem.
            aSumAllGradientPairs[iScore].Add(pGradientPair[iScore]);
         }

         pCopyFrom = IndexBin(cBytesPerBin, pCopyFrom, 1);
      } while(pCopyFromEnd != pCopyFrom);
      EBM_ASSERT(0 == (reinterpret_cast<const char *>(pCopyFrom) - reinterpret_cast<const char *>(aBins)) % cBytesPerBin);

      EBM_ASSERT(cSamplesTotal == cSamplesTotalDebug);
      EBM_ASSERT(weightTotalDebug * 0.999 <= weightTotal && weightTotal <= weightTotalDebug * 1.0001);
   }
};

extern void SumAllBins(
   BoosterShell * const pBoosterShell,
   const size_t cBins
#ifndef NDEBUG
   , const size_t cSamplesTotal
   , const FloatBig weightTotal
#endif // NDEBUG
) {
   LOG_0(TraceLevelVerbose, "Entered SumAllBins");

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();

   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      if(IsBinaryClassification(runtimeLearningTypeOrCountTargetClasses)) {
         SumAllBinsInternal<2>::Func(
            pBoosterShell,
            cBins
#ifndef NDEBUG
            , cSamplesTotal
            , weightTotal
#endif // NDEBUG
         );
      } else {
         SumAllBinsInternal<k_dynamicClassification>::Func(
            pBoosterShell,
            cBins
#ifndef NDEBUG
            , cSamplesTotal
            , weightTotal
#endif // NDEBUG
         );
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      SumAllBinsInternal<k_regression>::Func(
         pBoosterShell,
         cBins
#ifndef NDEBUG
         , cSamplesTotal
         , weightTotal
#endif // NDEBUG
      );
   }

   LOG_0(TraceLevelVerbose, "Exited SumAllBins");
}

} // DEFINED_ZONE_NAME
