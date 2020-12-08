// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderEbmNativeTest.h"

#include "ebm_native.h"
#include "EbmNativeTest.h"

static const TestPriority k_filePriority = TestPriority::Discretize;

TEST_CASE("Discretize, zero samples") {
   UNUSED(testCaseHidden);
   const FloatEbmType binCutsLowerBoundInclusive[] { 1, 2, 2.2, 2.3, 2.5, 2.6, 2.7, 2.8, 2.9 };
   constexpr IntEbmType countCuts = sizeof(binCutsLowerBoundInclusive) / sizeof(binCutsLowerBoundInclusive[0]);
   constexpr IntEbmType  cSamples = 0;

   Discretize(
      cSamples,
      nullptr,
      countCuts,
      binCutsLowerBoundInclusive,
      nullptr
   );

   Discretize(
      cSamples,
      nullptr,
      countCuts,
      binCutsLowerBoundInclusive,
      nullptr
   );
}

TEST_CASE("Discretize, increasing lengths") {
   constexpr size_t cBinCutsEnd = 1024 * 2 + 100;

   constexpr size_t cData = 11 * cBinCutsEnd;
   FloatEbmType * binCutsLowerBoundInclusive = new FloatEbmType[cBinCutsEnd];
   FloatEbmType * featureValues = new FloatEbmType[cData];
   IntEbmType * singleFeatureDiscretized = new IntEbmType[cData];
   for(size_t iCutPoint = 0; iCutPoint < cBinCutsEnd; ++iCutPoint) {
      const FloatEbmType cutPoint = static_cast<FloatEbmType>(iCutPoint);
      binCutsLowerBoundInclusive[iCutPoint] = cutPoint;

      // we have 11 items here, which will put these odd values into various positions for SIMD testing
      // we wrap missing at the first and last positions for additional testing of initial memory slots

      featureValues[11 * iCutPoint + 0] = std::numeric_limits<FloatEbmType>::lowest();
      featureValues[11 * iCutPoint + 1] = std::numeric_limits<FloatEbmType>::quiet_NaN();

      featureValues[11 * iCutPoint + 2] = -std::numeric_limits<FloatEbmType>::denorm_min();
      featureValues[11 * iCutPoint + 3] = std::numeric_limits<FloatEbmType>::denorm_min();

      featureValues[11 * iCutPoint + 4] = std::nextafter(cutPoint, std::numeric_limits<FloatEbmType>::lowest());
      featureValues[11 * iCutPoint + 5] = cutPoint;
      featureValues[11 * iCutPoint + 6] = std::nextafter(cutPoint, std::numeric_limits<FloatEbmType>::max());
      featureValues[11 * iCutPoint + 7] = std::numeric_limits<FloatEbmType>::max();
      featureValues[11 * iCutPoint + 8] = std::numeric_limits<FloatEbmType>::infinity();

      featureValues[11 * iCutPoint + 9] = std::numeric_limits<FloatEbmType>::signaling_NaN();
      featureValues[11 * iCutPoint + 10] = -std::numeric_limits<FloatEbmType>::infinity();
   }

   for(size_t cBinCuts = 0; cBinCuts < cBinCutsEnd; ++cBinCuts) {
      // the first pass fills in all values, then we permute the addresses randomly, but our first and last
      // values are fixed
      const size_t cRemoveLow = 0 == cBinCuts % 3 ? size_t { 0 } : size_t { 1 };
      const size_t cRemoveHigh = 0 == cBinCuts % 7 ? size_t { 0 } : size_t { 1 };

      const size_t cSamples = cData - cRemoveLow - cRemoveHigh;
      memset(singleFeatureDiscretized + cRemoveLow, 0, cSamples * sizeof(*singleFeatureDiscretized));
      Discretize(
         static_cast<IntEbmType>(cSamples),
         featureValues + cRemoveLow,
         cBinCuts,
         binCutsLowerBoundInclusive,
         singleFeatureDiscretized + cRemoveLow
      );

      for(size_t iCutPoint = 0; iCutPoint < cBinCutsEnd; ++iCutPoint) {
         CHECK(singleFeatureDiscretized[11 * iCutPoint + 0] == IntEbmType { 1 });
         CHECK(singleFeatureDiscretized[11 * iCutPoint + 1] == IntEbmType { 0 });

         CHECK(singleFeatureDiscretized[11 * iCutPoint + 2] == IntEbmType { 1 });
         CHECK(singleFeatureDiscretized[11 * iCutPoint + 3] == (size_t { 0 } == cBinCuts ? IntEbmType { 1 } : IntEbmType { 2 }));

         CHECK(singleFeatureDiscretized[11 * iCutPoint + 4] == IntEbmType { 1 } + static_cast<IntEbmType>(std::min(iCutPoint, cBinCuts)));
         CHECK(singleFeatureDiscretized[11 * iCutPoint + 5] == IntEbmType { 1 } + static_cast<IntEbmType>(std::min(iCutPoint + 1, cBinCuts)));
         CHECK(singleFeatureDiscretized[11 * iCutPoint + 6] == IntEbmType { 1 } + static_cast<IntEbmType>(std::min(iCutPoint + 1, cBinCuts)));
         CHECK(singleFeatureDiscretized[11 * iCutPoint + 7] == IntEbmType { 1 } + static_cast<IntEbmType>(cBinCuts));
         CHECK(singleFeatureDiscretized[11 * iCutPoint + 8] == IntEbmType { 1 } + static_cast<IntEbmType>(cBinCuts));

         CHECK(singleFeatureDiscretized[11 * iCutPoint + 9] == IntEbmType { 0 });
         CHECK(singleFeatureDiscretized[11 * iCutPoint + 10] == IntEbmType { 1 });
      }
   }

   delete[] binCutsLowerBoundInclusive;
   delete[] featureValues;
   delete[] singleFeatureDiscretized;
}

