// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"

static const TestPriority k_filePriority = TestPriority::Discretize;

TEST_CASE("GetHistogramCutCount, normals") {
   UNUSED(testCaseHidden);
   const FloatEbmType test[] { 1, 2, 3, 5 };
   constexpr size_t cTest = sizeof(test) / sizeof(test[0]);

   IntEbmType result = GetHistogramCutCount(static_cast<IntEbmType>(cTest), test, 0);
   CHECK(3 == result);
}

TEST_CASE("GetHistogramCutCount, out of bound inputs") {
   UNUSED(testCaseHidden);
   const FloatEbmType test[] { std::numeric_limits<FloatEbmType>::infinity(), 1, 2, std::numeric_limits<FloatEbmType>::quiet_NaN(), 3, 5, -std::numeric_limits<FloatEbmType>::infinity() };
   constexpr size_t cTest = sizeof(test) / sizeof(test[0]);

   IntEbmType result = GetHistogramCutCount(static_cast<IntEbmType>(cTest), test, 0);
   CHECK(3 == result);
}

TEST_CASE("Discretize, zero samples") {
   UNUSED(testCaseHidden);
   const FloatEbmType cutsLowerBoundInclusive[] { 1, 2, 2.2, 2.3, 2.5, 2.6, 2.7, 2.8, 2.9 };
   constexpr IntEbmType countCuts = sizeof(cutsLowerBoundInclusive) / sizeof(cutsLowerBoundInclusive[0]);
   constexpr IntEbmType  cSamples = 0;

   ErrorEbmType error = Discretize(
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
   constexpr size_t cCutsEnd = 1024 * 2 + 100;

   constexpr size_t cData = 11 * cCutsEnd;
   FloatEbmType * cutsLowerBoundInclusive = new FloatEbmType[cCutsEnd];
   FloatEbmType * featureValues = new FloatEbmType[cData];
   IntEbmType * singleFeatureDiscretized = new IntEbmType[cData];
   for(size_t iCutPoint = 0; iCutPoint < cCutsEnd; ++iCutPoint) {
      const FloatEbmType cutPoint = static_cast<FloatEbmType>(iCutPoint);
      cutsLowerBoundInclusive[iCutPoint] = cutPoint;

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

   for(size_t cCuts = 0; cCuts < cCutsEnd; ++cCuts) {
      // the first pass fills in all values, then we permute the addresses randomly, but our first and last
      // values are fixed
      const size_t cRemoveLow = 0 == cCuts % 3 ? size_t { 0 } : size_t { 1 };
      const size_t cRemoveHigh = 0 == cCuts % 7 ? size_t { 0 } : size_t { 1 };

      const size_t cSamples = cData - cRemoveLow - cRemoveHigh;
      memset(singleFeatureDiscretized + cRemoveLow, 0, cSamples * sizeof(*singleFeatureDiscretized));
      const ErrorEbmType error = Discretize(
         static_cast<IntEbmType>(cSamples),
         featureValues + cRemoveLow,
         cCuts,
         cutsLowerBoundInclusive,
         singleFeatureDiscretized + cRemoveLow
      );
      CHECK(Error_None == error);

      for(size_t iCutPoint = 0; iCutPoint < cCutsEnd; ++iCutPoint) {
         CHECK(singleFeatureDiscretized[11 * iCutPoint + 0] == IntEbmType { 1 });
         CHECK(singleFeatureDiscretized[11 * iCutPoint + 1] == IntEbmType { 0 });

         CHECK(singleFeatureDiscretized[11 * iCutPoint + 2] == IntEbmType { 1 });
         CHECK(singleFeatureDiscretized[11 * iCutPoint + 3] == (size_t { 0 } == cCuts ? IntEbmType { 1 } : IntEbmType { 2 }));

         CHECK(singleFeatureDiscretized[11 * iCutPoint + 4] == IntEbmType { 1 } + static_cast<IntEbmType>(std::min(iCutPoint, cCuts)));
         CHECK(singleFeatureDiscretized[11 * iCutPoint + 5] == IntEbmType { 1 } + static_cast<IntEbmType>(std::min(iCutPoint + 1, cCuts)));
         CHECK(singleFeatureDiscretized[11 * iCutPoint + 6] == IntEbmType { 1 } + static_cast<IntEbmType>(std::min(iCutPoint + 1, cCuts)));
         CHECK(singleFeatureDiscretized[11 * iCutPoint + 7] == IntEbmType { 1 } + static_cast<IntEbmType>(cCuts));
         CHECK(singleFeatureDiscretized[11 * iCutPoint + 8] == IntEbmType { 1 } + static_cast<IntEbmType>(cCuts));

         CHECK(singleFeatureDiscretized[11 * iCutPoint + 9] == IntEbmType { 0 });
         CHECK(singleFeatureDiscretized[11 * iCutPoint + 10] == IntEbmType { 1 });
      }
   }

   delete[] cutsLowerBoundInclusive;
   delete[] featureValues;
   delete[] singleFeatureDiscretized;
}

