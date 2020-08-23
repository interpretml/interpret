// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderEbmNativeTest.h"

#include "ebm_native.h"
#include "EbmNativeTest.h"
#include "RandomStreamTest.h"

static const TestPriority k_filePriority = TestPriority::GenerateQuantileCutPoints;

void GetExpectedStats(
   const IntEbmType countSamples,
   const FloatEbmType * const aFeatureValues,
   IntEbmType & countMissingValues,
   FloatEbmType & minNonInfinityValue,
   IntEbmType & countNegativeInfinity,
   FloatEbmType & maxNonInfinityValue,
   IntEbmType & countPositiveInfinity
) {
   countMissingValues = 0;
   minNonInfinityValue = std::numeric_limits<FloatEbmType>::max();
   countNegativeInfinity = 0;
   maxNonInfinityValue = std::numeric_limits<FloatEbmType>::lowest();
   countPositiveInfinity = 0;

   for(IntEbmType i = 0; i < countSamples; ++i) {
      const FloatEbmType val = aFeatureValues[i];
      if(std::isnan(val)) {
         ++countMissingValues;
      } else if(-std::numeric_limits<FloatEbmType>::infinity() == val) {
         ++countNegativeInfinity;
      } else if(std::numeric_limits<FloatEbmType>::infinity() == val) {
         ++countPositiveInfinity;
      } else {
         minNonInfinityValue = std::min(val, minNonInfinityValue);
         maxNonInfinityValue = std::max(val, maxNonInfinityValue);
      }
   }
   if(countMissingValues + countNegativeInfinity + countPositiveInfinity == countSamples) {
      // if everything was a special value, then make the min/max zero
      minNonInfinityValue = 0;
      maxNonInfinityValue = 0;
   }
}

TEST_CASE("GenerateQuantileCutPoints, 0 samples") {
   constexpr IntEbmType countSamplesPerBinMin = -1;

   IntEbmType countCutPoints = -1;
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType isPositiveInfinityPresent;

   IntEbmType ret = GenerateQuantileCutPoints(
      0,
      nullptr,
      countSamplesPerBinMin,
      &countCutPoints,
      nullptr,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &isPositiveInfinityPresent
   );
   CHECK(IntEbmType { 0 } == ret);
   CHECK(IntEbmType { 0 } == countMissingValues);
   CHECK(FloatEbmType { 0 } == minNonInfinityValue);
   CHECK(IntEbmType { 0 } == countNegativeInfinity);
   CHECK(FloatEbmType { 0 } == maxNonInfinityValue);
   CHECK(EBM_FALSE == isPositiveInfinityPresent);
   CHECK(IntEbmType { 0 } == countCutPoints);
}

TEST_CASE("GenerateQuantileCutPoints, only missing") {
   constexpr IntEbmType countSamplesPerBinMin = 1;
   FloatEbmType featureValues[] { std::numeric_limits<FloatEbmType>::quiet_NaN(), std::numeric_limits<FloatEbmType>::quiet_NaN() };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   IntEbmType countCutPoints = 1;
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      reinterpret_cast<FloatEbmType *>(0x1), // this shouldn't be filled, so it would throw an exception if it did 
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   CHECK(IntEbmType { 0 } == countCutPoints);
}

TEST_CASE("GenerateQuantileCutPoints, just one bin") {
   constexpr IntEbmType countSamplesPerBinMin = 1;
   FloatEbmType featureValues[] { 1, 2 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   IntEbmType countCutPoints = 0;
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      nullptr,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   CHECK(IntEbmType { 0 } == countCutPoints);
}

TEST_CASE("GenerateQuantileCutPoints, too small") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 5 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, splitable") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 0, 1, 2, 3 };
   const std::vector<FloatEbmType> expectedCutPoints { 1.5 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, splitable (first interior check not splitable)") {
   constexpr IntEbmType countSamplesPerBinMin = 3;
   FloatEbmType featureValues[] { 0, 1, 5, 5, 7, 8, 9 };
   const std::vector<FloatEbmType> expectedCutPoints { 6 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, splitable except middle isn't available") {
   constexpr IntEbmType countSamplesPerBinMin = 3;
   FloatEbmType featureValues[] { 0, 1, 5, 5, 8, 9 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 5, 5, 5, 5 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 1, 5, 5, 5 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+right") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 5, 5, 5, 9 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable+right") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 1, 5, 5, 9 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+unsplitable") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 4, 4, 6, 6 };
   const std::vector<FloatEbmType> expectedCutPoints { 5 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable+unsplitable") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 1, 4, 4, 6, 6 };
   const std::vector<FloatEbmType> expectedCutPoints { 5 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+unsplitable+right") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 4, 4, 6, 6, 9 };
   const std::vector<FloatEbmType> expectedCutPoints { 5 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+mid+unsplitable") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 4, 4, 5, 6, 6 };
   const std::vector<FloatEbmType> expectedCutPoints { 4.5 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable+mid+unsplitable") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 1, 4, 4, 5, 6, 6 };
   const std::vector<FloatEbmType> expectedCutPoints { 4.5 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+mid+unsplitable+right") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 4, 4, 5, 6, 6, 9 };
   const std::vector<FloatEbmType> expectedCutPoints { 4.5 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+splitable") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 5, 5, 7, 8 };
   const std::vector<FloatEbmType> expectedCutPoints { 6 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable+splitable") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 1, 5, 5, 7, 8 };
   const std::vector<FloatEbmType> expectedCutPoints { 6 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, splitable+unsplitable") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 2, 3, 5, 5 };
   const std::vector<FloatEbmType> expectedCutPoints { 4 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, splitable+unsplitable+right") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 2, 3, 5, 5, 7 };
   const std::vector<FloatEbmType> expectedCutPoints { 4 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, splitable+unsplitable+splitable") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 2, 3, 5, 5, 7, 8 };
   const std::vector<FloatEbmType> expectedCutPoints { 4, 6 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+splitable+unsplitable") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 2, 2, 4, 6, 8, 8 };
   const std::vector<FloatEbmType> expectedCutPoints { 3, 7 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable+splitable+unsplitable") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 1, 2, 2, 4, 6, 8, 8 };
   const std::vector<FloatEbmType> expectedCutPoints { 5 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+splitable+unsplitable+right") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 2, 2, 4, 6, 8, 8, 9 };
   const std::vector<FloatEbmType> expectedCutPoints { 3, 7 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable+splitable+unsplitable+right") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 1, 2, 2, 4, 6, 8, 8, 9 };
   const std::vector<FloatEbmType> expectedCutPoints { 3, 7 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+splitable+unsplitable+splitable") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 1, 1, 2, 3, 5, 5, 7, 8 };
   const std::vector<FloatEbmType> expectedCutPoints { 1.5, 4, 6 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable+splitable+unsplitable+splitable") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 0, 1, 1, 2, 3, 5, 5, 7, 8 };
   const std::vector<FloatEbmType> expectedCutPoints { 1.5, 4, 6 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, splitable+unsplitable+splitable+unsplitable") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 2, 3, 5, 5, 7, 8, 9, 9 };
   const std::vector<FloatEbmType> expectedCutPoints { 4, 6, 8.5 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, splitable+unsplitable+splitable+unsplitable+right") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 2, 3, 5, 5, 7, 8, 9, 9, 10 };
   const std::vector<FloatEbmType> expectedCutPoints { 4, 6, 8.5 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable+splitable+unsplitable+splitable+unsplitable+splitable+unsplitable+right") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9 };
   const std::vector<FloatEbmType> expectedCutPoints { 3.5, 5.5, 7.5 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, infinities") {
   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 
      -std::numeric_limits<FloatEbmType>::infinity(),
      std::numeric_limits<FloatEbmType>::lowest(),
      std::numeric_limits<FloatEbmType>::max(),
      std::numeric_limits<FloatEbmType>::infinity(),
      std::numeric_limits<FloatEbmType>::quiet_NaN(),
      -std::numeric_limits<FloatEbmType>::infinity(),
      std::numeric_limits<FloatEbmType>::lowest(),
      std::numeric_limits<FloatEbmType>::max(),
      std::numeric_limits<FloatEbmType>::infinity(),
      std::numeric_limits<FloatEbmType>::signaling_NaN(),
   };
   const std::vector<FloatEbmType> expectedCutPoints { 0 };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[1000];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, average division sizes that requires the ceiling instead of rounding") {
   // our algorithm makes an internal assumption that we can give each cut point a split.  This is guaranteed if we 
   // make the average length of the equal value long ranges the ceiling of the average samples per bin.  
   // This test stresses that average calculation by having an average bin lenght of 2.2222222222 but if you use 
   // a bin width of 2, then there are 3 cut points that can't get any cuts.  3 cut points means that even if you 
   // don't give the first and last SplittingRanges an actual cut point, which can be reasonalbe since the 
   // first and last SplittingRanges are special in that they may have no long ranges on the tail ends, 
   // you still end up with one or more SplittingRanges that can't have a cut if you don't take the ceiling.

   constexpr IntEbmType countSamplesPerBinMin = 2;
   FloatEbmType featureValues[] { 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16,
      17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29 };
   const std::vector<FloatEbmType> expectedCutPoints {
      0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
      9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5,
      18.5, 19.5, 21.5, 22.5, 23.5, 24.5, 25.5, 27.5
   };

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[26];
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, stress test the guarantee of one split per SplittingRange, by 2") {
   constexpr IntEbmType countSamplesPerBinMin = 1;
   constexpr size_t cItemsPerRange = 10;
   constexpr size_t cInteriorRanges = 3;
   constexpr size_t cRemoveCuts = 1;

   FloatEbmType featureValues[3 + cInteriorRanges * cItemsPerRange];

   std::vector<FloatEbmType> expectedCutPoints;

   featureValues[0] = 0;
   for(size_t iRange = 0; iRange < cInteriorRanges; ++iRange) {
      for(size_t i = 1 + cItemsPerRange * iRange; i < 1 + (cItemsPerRange * (iRange + 1)); ++i) {
         featureValues[i] = iRange + 1;
      }
      expectedCutPoints.push_back(FloatEbmType { 0.5 } + static_cast<FloatEbmType>(iRange));
   }
   expectedCutPoints.push_back(FloatEbmType { 0.5 } + static_cast<FloatEbmType>(cInteriorRanges));

   featureValues[cInteriorRanges * cItemsPerRange + 1] = 1 + cInteriorRanges;
   featureValues[cInteriorRanges * cItemsPerRange + 2] = 1 + cInteriorRanges;

   if(1 == cRemoveCuts) {
      expectedCutPoints.erase(expectedCutPoints.begin());
   } else if(2 == cRemoveCuts) {
      expectedCutPoints.erase(expectedCutPoints.begin());
      expectedCutPoints.erase(expectedCutPoints.end() - 1);
   } else {
      assert(0 == cRemoveCuts);
   }

   constexpr IntEbmType countSamples = sizeof(featureValues) / sizeof(featureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[cInteriorRanges + 1 - cRemoveCuts];
   assert(sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]) == expectedCutPoints.size());
   IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      countSamples,
      featureValues,
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType ret = GenerateQuantileCutPoints(
      countSamples,
      featureValues,
      countSamplesPerBinMin,
      &countCutPoints,
      cutPointsLowerBoundInclusive,
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, randomized fairness check") {
   RandomStreamTest randomStream(randomSeed);
   if(!randomStream.IsSuccess()) {
      exit(1);
   }

   constexpr IntEbmType countSamplesPerBinMin = 1;
   constexpr IntEbmType countSamples = 100;
   FloatEbmType featureValues[countSamples];

   constexpr IntEbmType randomMaxMax = countSamples - 1; // this doesn't need to be exactly countSamples - 1, but this number gives us chunky sets
   size_t cutHistogram[randomMaxMax];
   constexpr size_t cCutHistogram = sizeof(cutHistogram) / sizeof(cutHistogram[0]);
   // our random numbers can be any numbers from 0 to randomMaxMax (inclusive), which gives us randomMaxMax - 1 possible cut points between them
   static_assert(1 == cCutHistogram % 2, "cutHistogram must have a center value that is perfectly in the middle");

   FloatEbmType cutPointsLowerBoundInclusive[9];

   memset(cutHistogram, 0, sizeof(cutHistogram));

   for(int iIteration = 0; iIteration < 100; ++iIteration) {
      for(size_t randomMax = 1; randomMax <= randomMaxMax; randomMax += 2) {
         // since randomMax isn't larger than the number of samples, we'll always be chunky.  This is good for testing range collisions
         for(size_t iSample = 0; iSample < countSamples; ++iSample) {
            bool bMissing = 0 == randomStream.Next(countSamples); // some datasetes will have zero missing values, some will have 1 or more
            size_t iRandom = randomStream.Next(randomMax + 1);
            featureValues[iSample] = bMissing ? std::numeric_limits<FloatEbmType>::quiet_NaN() : static_cast<FloatEbmType>(iRandom);
         }

         IntEbmType countCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
         IntEbmType countMissingValues;
         FloatEbmType minNonInfinityValue;
         IntEbmType countNegativeInfinity;
         FloatEbmType maxNonInfinityValue;
         IntEbmType countPositiveInfinity;

         // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
         IntEbmType countMissingValuesExpected;
         FloatEbmType minNonInfinityValueExpected;
         IntEbmType countNegativeInfinityExpected;
         FloatEbmType maxNonInfinityValueExpected;
         IntEbmType countPositiveInfinityExpected;
         GetExpectedStats(
            countSamples,
            featureValues,
            countMissingValuesExpected,
            minNonInfinityValueExpected,
            countNegativeInfinityExpected,
            maxNonInfinityValueExpected,
            countPositiveInfinityExpected
         );

         IntEbmType ret = GenerateQuantileCutPoints(
            countSamples,
            featureValues,
            countSamplesPerBinMin,
            &countCutPoints,
            cutPointsLowerBoundInclusive,
            &countMissingValues,
            &minNonInfinityValue,
            &countNegativeInfinity,
            &maxNonInfinityValue,
            &countPositiveInfinity
         );
         CHECK(0 == ret);

         CHECK(countMissingValuesExpected == countMissingValues);
         CHECK(minNonInfinityValueExpected == minNonInfinityValue);
         CHECK(countNegativeInfinityExpected == countNegativeInfinity);
         CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
         CHECK(countPositiveInfinityExpected == countPositiveInfinity);

         const size_t cCutPoints = static_cast<size_t>(countCutPoints);
         assert(1 == randomMax % 2); // our random numbers need a center value as well
         constexpr size_t iHistogramExactMiddle = cCutHistogram / 2;
         const size_t iCutExactMiddle = randomMax / 2;
         assert(iCutExactMiddle <= iHistogramExactMiddle);
         const size_t iShiftToMiddle = iHistogramExactMiddle - iCutExactMiddle;
         for(size_t iCutPoint = 0; iCutPoint < cCutPoints; ++iCutPoint) {
            const FloatEbmType cutPoint = cutPointsLowerBoundInclusive[iCutPoint];
            // cutPoint can be a number between 0.5 and (randomMax - 0.5)
            const size_t iCut = static_cast<size_t>(std::round(cutPoint - FloatEbmType { 0.5 }));
            const size_t iSymetricCut = iShiftToMiddle + iCut;
            assert(iSymetricCut < cCutHistogram);
            ++cutHistogram[iSymetricCut];
         }
      }
   }
   size_t cBottomTotal = 0;
   size_t cTopTotal = 0;
   for(size_t i = 0; i < (cCutHistogram + 1) / 2; ++i) {
      size_t iBottom = i;
      size_t iTop = cCutHistogram - 1 - i;

      size_t cBottom = cutHistogram[iBottom];
      size_t cTop = cutHistogram[iTop];
      cBottomTotal += cBottom;
      cTopTotal += cTop;
   }
   const size_t cMax = std::max(cBottomTotal, cTopTotal);
   const size_t cMin = std::min(cBottomTotal, cTopTotal);
   const FloatEbmType ratio = static_cast<FloatEbmType>(cMin) / static_cast<FloatEbmType>(cMax);
   CHECK(0.97 <= ratio || 0 == cMax);
}

TEST_CASE("GenerateQuantileCutPoints, chunky randomized check") {
   RandomStreamTest randomStream(randomSeed);
   if(!randomStream.IsSuccess()) {
      exit(1);
   }

   FloatEbmType cutPointsLowerBoundInclusive[9];
   constexpr size_t cCutPoints = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   constexpr IntEbmType countSamplesPerBinMin = 3;
   constexpr size_t cSamples = 100;
   constexpr size_t maxRandomVal = 70;
   const size_t cLongBinLength = static_cast<size_t>(
      std::ceil(static_cast<FloatEbmType>(cSamples) / static_cast<FloatEbmType>(cCutPoints + 1))
      );
   FloatEbmType featureValues[cSamples];

   for(int iIteration = 0; iIteration < 30000; ++iIteration) {
      memset(featureValues, 0, sizeof(featureValues));

      size_t i = 0;
      size_t cLongRanges = randomStream.Next(6);
      for(size_t iLongRange = 0; iLongRange < cLongRanges; ++iLongRange) {
         size_t cItems = randomStream.Next(cLongBinLength) + cLongBinLength;
         size_t val = randomStream.Next(maxRandomVal) + 1;
         for(size_t iItem = 0; iItem < cItems; ++iItem) {
            featureValues[i % cSamples] = static_cast<FloatEbmType>(val);
            ++i;
         }
      }
      size_t cShortRanges = randomStream.Next(6);
      for(size_t iShortRange = 0; iShortRange < cShortRanges; ++iShortRange) {
         size_t cItems = randomStream.Next(cLongBinLength);
         size_t val = randomStream.Next(maxRandomVal) + 1;
         for(size_t iItem = 0; iItem < cItems; ++iItem) {
            featureValues[i % cSamples] = static_cast<FloatEbmType>(val);
            ++i;
         }
      }
      for(size_t iSample = 0; iSample < cSamples; ++iSample) {
         if(0 == featureValues[iSample]) {
            featureValues[iSample] = static_cast<FloatEbmType>(randomStream.Next(maxRandomVal) + 1);
         }
      }

      const IntEbmType countSamples = static_cast<IntEbmType>(cSamples);

      IntEbmType countCutPoints = static_cast<IntEbmType>(cCutPoints);
      IntEbmType countMissingValues;
      FloatEbmType minNonInfinityValue;
      IntEbmType countNegativeInfinity;
      FloatEbmType maxNonInfinityValue;
      IntEbmType countPositiveInfinity;

      // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies featureValues
      IntEbmType countMissingValuesExpected;
      FloatEbmType minNonInfinityValueExpected;
      IntEbmType countNegativeInfinityExpected;
      FloatEbmType maxNonInfinityValueExpected;
      IntEbmType countPositiveInfinityExpected;
      GetExpectedStats(
         countSamples,
         featureValues,
         countMissingValuesExpected,
         minNonInfinityValueExpected,
         countNegativeInfinityExpected,
         maxNonInfinityValueExpected,
         countPositiveInfinityExpected
      );

      IntEbmType ret = GenerateQuantileCutPoints(
         countSamples,
         featureValues,
         countSamplesPerBinMin,
         &countCutPoints,
         cutPointsLowerBoundInclusive,
         &countMissingValues,
         &minNonInfinityValue,
         &countNegativeInfinity,
         &maxNonInfinityValue,
         &countPositiveInfinity
      );
      CHECK(0 == ret);

      CHECK(countMissingValuesExpected == countMissingValues);
      CHECK(minNonInfinityValueExpected == minNonInfinityValue);
      CHECK(countNegativeInfinityExpected == countNegativeInfinity);
      CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
      CHECK(countPositiveInfinityExpected == countPositiveInfinity);
   }
}

