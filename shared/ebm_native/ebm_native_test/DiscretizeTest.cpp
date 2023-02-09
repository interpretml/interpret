// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"

static constexpr TestPriority k_filePriority = TestPriority::Discretize;

TEST_CASE("GetHistogramCutCount, normals") {
   UNUSED(testCaseHidden);
   const double test[] { 1, 2, 3, 5 };
   static constexpr size_t cTest = sizeof(test) / sizeof(test[0]);

   IntEbm result = GetHistogramCutCount(static_cast<IntEbm>(cTest), test);
   CHECK(3 == result);
}

TEST_CASE("GetHistogramCutCount, out of bound inputs") {
   UNUSED(testCaseHidden);
   const double test[] { std::numeric_limits<double>::infinity(), 1, 2, std::numeric_limits<double>::quiet_NaN(), 3, 5, -std::numeric_limits<double>::infinity() };
   static constexpr size_t cTest = sizeof(test) / sizeof(test[0]);

   IntEbm result = GetHistogramCutCount(static_cast<IntEbm>(cTest), test);
   CHECK(3 == result);
}

TEST_CASE("Discretize, zero samples") {
   ErrorEbm error;

   UNUSED(testCaseHidden);
   const double cutsLowerBoundInclusive[] { 1, 2, 2.2, 2.3, 2.5, 2.6, 2.7, 2.8, 2.9 };
   static constexpr IntEbm countCuts = sizeof(cutsLowerBoundInclusive) / sizeof(cutsLowerBoundInclusive[0]);
   static constexpr IntEbm  cSamples = 0;

   error = Discretize(
      cSamples,
      nullptr,
      countCuts,
      cutsLowerBoundInclusive,
      nullptr
   );
   CHECK(Error_None == error);

   error = Discretize(
      cSamples,
      nullptr,
      countCuts,
      cutsLowerBoundInclusive,
      nullptr
   );
   CHECK(Error_None == error);
}

TEST_CASE("Discretize, increasing lengths") {
   static constexpr size_t cCutsEnd = 1024 * 2 + 100;
   static constexpr size_t cData = 11 * cCutsEnd;

   ErrorEbm error;

   double cutsLowerBoundInclusive[cCutsEnd];
   double featureVals[cData];
   IntEbm aiBins[cData];
   for(size_t iCutPoint = 0; iCutPoint < cCutsEnd; ++iCutPoint) {
      const double cutPoint = static_cast<double>(iCutPoint);
      cutsLowerBoundInclusive[iCutPoint] = cutPoint;

      // we have 11 items here, which will put these odd values into various positions for SIMD testing
      // we wrap missing at the first and last positions for additional testing of initial memory slots

      featureVals[11 * iCutPoint + 0] = std::numeric_limits<double>::lowest();
      featureVals[11 * iCutPoint + 1] = std::numeric_limits<double>::quiet_NaN();

      featureVals[11 * iCutPoint + 2] = -std::numeric_limits<double>::denorm_min();
      featureVals[11 * iCutPoint + 3] = std::numeric_limits<double>::denorm_min();

      featureVals[11 * iCutPoint + 4] = FloatTickDecrementTest(cutPoint);
      featureVals[11 * iCutPoint + 5] = cutPoint;
      featureVals[11 * iCutPoint + 6] = FloatTickIncrementTest(cutPoint);
      featureVals[11 * iCutPoint + 7] = std::numeric_limits<double>::max();
      featureVals[11 * iCutPoint + 8] = std::numeric_limits<double>::infinity();

      featureVals[11 * iCutPoint + 9] = std::numeric_limits<double>::signaling_NaN();
      featureVals[11 * iCutPoint + 10] = -std::numeric_limits<double>::infinity();
   }

   for(size_t cCuts = 0; cCuts < cCutsEnd; ++cCuts) {
      // the first pass fills in all values, then we permute the addresses randomly, but our first and last
      // values are fixed
      const size_t cRemoveLow = 0 == cCuts % 3 ? size_t { 0 } : size_t { 1 };
      const size_t cRemoveHigh = 0 == cCuts % 7 ? size_t { 0 } : size_t { 1 };

      const size_t cSamples = cData - cRemoveLow - cRemoveHigh;
      memset(aiBins + cRemoveLow, 0, cSamples * sizeof(*aiBins));
      error = Discretize(
         static_cast<IntEbm>(cSamples),
         featureVals + cRemoveLow,
         cCuts,
         cutsLowerBoundInclusive,
         aiBins + cRemoveLow
      );
      CHECK(Error_None == error);

      for(size_t iCutPoint = 0; iCutPoint < cCutsEnd; ++iCutPoint) {
         CHECK(aiBins[11 * iCutPoint + 0] == IntEbm { 1 });
         CHECK(aiBins[11 * iCutPoint + 1] == IntEbm { 0 });

         CHECK(aiBins[11 * iCutPoint + 2] == IntEbm { 1 });
         CHECK(aiBins[11 * iCutPoint + 3] == (size_t { 0 } == cCuts ? IntEbm { 1 } : IntEbm { 2 }));

         CHECK(aiBins[11 * iCutPoint + 4] == IntEbm { 1 } + static_cast<IntEbm>(std::min(iCutPoint, cCuts)));
         CHECK(aiBins[11 * iCutPoint + 5] == IntEbm { 1 } + static_cast<IntEbm>(std::min(iCutPoint + 1, cCuts)));
         CHECK(aiBins[11 * iCutPoint + 6] == IntEbm { 1 } + static_cast<IntEbm>(std::min(iCutPoint + 1, cCuts)));
         CHECK(aiBins[11 * iCutPoint + 7] == IntEbm { 1 } + static_cast<IntEbm>(cCuts));
         CHECK(aiBins[11 * iCutPoint + 8] == IntEbm { 1 } + static_cast<IntEbm>(cCuts));

         CHECK(aiBins[11 * iCutPoint + 9] == IntEbm { 0 });
         CHECK(aiBins[11 * iCutPoint + 10] == IntEbm { 1 });
      }
   }
}

