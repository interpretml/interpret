// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderEbmNativeTest.h"

#include "ebm_native.h"
#include "EbmNativeTest.h"

static const TestPriority k_filePriority = TestPriority::Discretize;

TEST_CASE("Discretize, zero samples") {
   UNUSED(testCaseHidden);
   const FloatEbmType cutPointsLowerBoundInclusive[] { 1, 2, 2.2, 2.3, 2.5, 2.6, 2.7, 2.8, 2.9 };
   constexpr IntEbmType countCuts = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   constexpr IntEbmType  cSamples = 0;

   Discretize(
      cSamples,
      nullptr,
      countCuts,
      cutPointsLowerBoundInclusive,
      nullptr
   );

   Discretize(
      cSamples,
      nullptr,
      countCuts,
      cutPointsLowerBoundInclusive,
      nullptr
   );
}

TEST_CASE("Discretize, zero cuts, missing") {
   FloatEbmType featureValues[] { 0, 0.9, 1, 1.1, 1.9, 2, 2.1, std::numeric_limits<FloatEbmType>::quiet_NaN(), 2.75, 3 };
   const IntEbmType expectedDiscretized[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };

   constexpr size_t cSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   static_assert(cSamples == sizeof(expectedDiscretized) / sizeof(expectedDiscretized[0]),
      "cSamples and expectedDiscretized must be the same length"
      );
   constexpr IntEbmType countCuts = 0;
   IntEbmType singleFeatureDiscretized[cSamples];
   const bool bMissing = std::any_of(featureValues, featureValues + cSamples, [](const FloatEbmType val) { return std::isnan(val); });

   Discretize(
      IntEbmType { cSamples },
      featureValues,
      countCuts,
      nullptr,
      singleFeatureDiscretized
   );

   for(size_t i = 0; i < cSamples; ++i) {
      CHECK(expectedDiscretized[i] == singleFeatureDiscretized[i]);
   }
}

TEST_CASE("Discretize, missing") {
   const FloatEbmType cutPointsLowerBoundInclusive[] { 1, 2, 2.2, 2.3, 2.5, 2.6, 2.7, 2.8, 2.9 };
   FloatEbmType featureValues[] { 0, 0.9, 1, 1.1, 1.9, 2, 2.1, std::numeric_limits<FloatEbmType>::quiet_NaN(), 2.75, 3 };
   const IntEbmType expectedDiscretized[] { 0, 0, 1, 1, 1, 2, 2, 10, 7, 9 };

   constexpr size_t cSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   static_assert(cSamples == sizeof(expectedDiscretized) / sizeof(expectedDiscretized[0]),
      "cSamples and expectedDiscretized must be the same length"
      );
   constexpr IntEbmType countCuts = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType singleFeatureDiscretized[cSamples];
   const bool bMissing = std::any_of(featureValues, featureValues + cSamples, [](const FloatEbmType val) { return std::isnan(val); });

   Discretize(
      IntEbmType { cSamples },
      featureValues,
      countCuts,
      cutPointsLowerBoundInclusive,
      singleFeatureDiscretized
   );

   for(size_t i = 0; i < cSamples; ++i) {
      CHECK(expectedDiscretized[i] == singleFeatureDiscretized[i]);
   }
}

TEST_CASE("Discretize, increasing lengths") {
   FloatEbmType featureValues[1];
   IntEbmType singleFeatureDiscretized[1];

   constexpr size_t cCutPointsMax = 1024 * 2 + 100;
   FloatEbmType cutPointsLowerBoundInclusive[cCutPointsMax];
   for(size_t iCutPoint = 0; iCutPoint < cCutPointsMax; ++iCutPoint) {
      cutPointsLowerBoundInclusive[iCutPoint] = static_cast<FloatEbmType>(iCutPoint);
   }
   // this doesn't check 0 cuts, or having missing values
   for(size_t cCutPoints = 1; cCutPoints <= cCutPointsMax; ++cCutPoints) {
      for(size_t iCutPoint = 0; iCutPoint < cCutPoints; ++iCutPoint) {
         // first try it without missing values
         featureValues[0] = cutPointsLowerBoundInclusive[iCutPoint] - FloatEbmType { 0.5 };
         Discretize(
            1,
            featureValues,
            cCutPoints,
            cutPointsLowerBoundInclusive,
            singleFeatureDiscretized
         );
         CHECK(singleFeatureDiscretized[0] == static_cast<IntEbmType>(iCutPoint));

         featureValues[0] = cutPointsLowerBoundInclusive[iCutPoint];
         Discretize(
            1,
            featureValues,
            cCutPoints,
            cutPointsLowerBoundInclusive,
            singleFeatureDiscretized
         );
         CHECK(singleFeatureDiscretized[0] == static_cast<IntEbmType>(iCutPoint) + 1); // any exact matches are inclusive to the upper bound

         featureValues[0] = cutPointsLowerBoundInclusive[iCutPoint] + FloatEbmType { 0.5 };
         Discretize(
            1,
            featureValues,
            cCutPoints,
            cutPointsLowerBoundInclusive,
            singleFeatureDiscretized
         );
         CHECK(singleFeatureDiscretized[0] == static_cast<IntEbmType>(iCutPoint) + 1);

         // now try it indicating that there can be missing values, which should take the 0 value position and bump everything else up
         featureValues[0] = cutPointsLowerBoundInclusive[iCutPoint] - FloatEbmType { 0.5 };
         Discretize(
            1,
            featureValues,
            cCutPoints,
            cutPointsLowerBoundInclusive,
            singleFeatureDiscretized
         );
         CHECK(singleFeatureDiscretized[0] == static_cast<IntEbmType>(iCutPoint));

         featureValues[0] = cutPointsLowerBoundInclusive[iCutPoint];
         Discretize(
            1,
            featureValues,
            cCutPoints,
            cutPointsLowerBoundInclusive,
            singleFeatureDiscretized
         );
         CHECK(singleFeatureDiscretized[0] == static_cast<IntEbmType>(iCutPoint) + 1); // any exact matches are inclusive to the upper bound

         featureValues[0] = cutPointsLowerBoundInclusive[iCutPoint] + FloatEbmType { 0.5 };
         Discretize(
            1,
            featureValues,
            cCutPoints,
            cutPointsLowerBoundInclusive,
            singleFeatureDiscretized
         );
         CHECK(singleFeatureDiscretized[0] == static_cast<IntEbmType>(iCutPoint) + 1);
      }
   }
}

