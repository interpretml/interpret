// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderEbmNativeTest.h"

#include "ebm_native.h"
#include "EbmNativeTest.h"

static const TestPriority k_filePriority = TestPriority::GenerateUniformBinCuts;

constexpr FloatEbmType illegalVal = FloatEbmType { -888.88 };

TEST_CASE("GenerateUniformBinCuts, 0 samples") {
   IntEbmType countBinCuts = 3;

   std::vector<FloatEbmType> featureValues {};
   const std::vector<FloatEbmType> expectedBinCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   GenerateUniformBinCuts(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countBinCuts,
      &binCutsLowerBoundInclusive[0],
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == countMissingValues);
   CHECK(0 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(0 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateUniformBinCuts, only missing") {
   IntEbmType countBinCuts = 3;

   std::vector<FloatEbmType> featureValues { std::numeric_limits<FloatEbmType>::quiet_NaN(), std::numeric_limits<FloatEbmType>::quiet_NaN() };
   const std::vector<FloatEbmType> expectedBinCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   GenerateUniformBinCuts(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countBinCuts,
      &binCutsLowerBoundInclusive[0],
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(2 == countMissingValues);
   CHECK(0 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(0 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateUniformBinCuts, one cut, -infinity") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues { -std::numeric_limits<FloatEbmType>::infinity(), -std::numeric_limits<FloatEbmType>::infinity() };
   const std::vector<FloatEbmType> expectedBinCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   GenerateUniformBinCuts(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countBinCuts,
      &binCutsLowerBoundInclusive[0],
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == countMissingValues);
   CHECK(0 == minNonInfinityValue);
   CHECK(2 == countNegativeInfinity);
   CHECK(0 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateUniformBinCuts, one cut, +infinity") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues { std::numeric_limits<FloatEbmType>::infinity(), std::numeric_limits<FloatEbmType>::infinity() };
   const std::vector<FloatEbmType> expectedBinCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   GenerateUniformBinCuts(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countBinCuts,
      &binCutsLowerBoundInclusive[0],
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == countMissingValues);
   CHECK(0 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(0 == maxNonInfinityValue);
   CHECK(2 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateUniformBinCuts, one cut, -infinity and +infinity") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues { -std::numeric_limits<FloatEbmType>::infinity(), std::numeric_limits<FloatEbmType>::infinity() };
   const std::vector<FloatEbmType> expectedBinCuts { 0 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   GenerateUniformBinCuts(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countBinCuts,
      &binCutsLowerBoundInclusive[0],
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == countMissingValues);
   CHECK(0 == minNonInfinityValue);
   CHECK(1 == countNegativeInfinity);
   CHECK(0 == maxNonInfinityValue);
   CHECK(1 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateUniformBinCuts, one item") {
   IntEbmType countBinCuts = 3;

   std::vector<FloatEbmType> featureValues { 1 };
   const std::vector<FloatEbmType> expectedBinCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   GenerateUniformBinCuts(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countBinCuts,
      &binCutsLowerBoundInclusive[0],
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == countMissingValues);
   CHECK(1 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(1 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateUniformBinCuts, zero cuts") {
   IntEbmType countBinCuts = 0;

   std::vector<FloatEbmType> featureValues { 1, 2, 3, 4 };
   const std::vector<FloatEbmType> expectedBinCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   GenerateUniformBinCuts(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countBinCuts,
      &binCutsLowerBoundInclusive[0],
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == countMissingValues);
   CHECK(1 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(4 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateUniformBinCuts, identical values") {
   IntEbmType countBinCuts = 2;

   std::vector<FloatEbmType> featureValues { 1, 1, std::numeric_limits<FloatEbmType>::quiet_NaN(), 1 };
   const std::vector<FloatEbmType> expectedBinCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   GenerateUniformBinCuts(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countBinCuts,
      &binCutsLowerBoundInclusive[0],
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(1 == countMissingValues);
   CHECK(1 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(1 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateUniformBinCuts, underflow") {
   IntEbmType countBinCuts = 9;

   std::vector<FloatEbmType> featureValues { 0, std::numeric_limits<FloatEbmType>::denorm_min() };
   const std::vector<FloatEbmType> expectedBinCuts { std::numeric_limits<FloatEbmType>::denorm_min() };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   GenerateUniformBinCuts(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countBinCuts,
      &binCutsLowerBoundInclusive[0],
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == countMissingValues);
   CHECK(0 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(std::numeric_limits<FloatEbmType>::denorm_min() == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateUniformBinCuts, normal") {
   IntEbmType countBinCuts = 9;

   std::vector<FloatEbmType> featureValues { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
   const std::vector<FloatEbmType> expectedBinCuts { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   GenerateUniformBinCuts(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countBinCuts,
      &binCutsLowerBoundInclusive[0],
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == countMissingValues);
   CHECK(0 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(10 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateUniformBinCuts, 1 cut, -infinity, lowest, max, and +infinity") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues {
      -std::numeric_limits<FloatEbmType>::infinity(),
      std::numeric_limits<FloatEbmType>::lowest(),
      std::numeric_limits<FloatEbmType>::max(),
      std::numeric_limits<FloatEbmType>::infinity()
   };
   const std::vector<FloatEbmType> expectedBinCuts { 0 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   GenerateUniformBinCuts(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countBinCuts,
      &binCutsLowerBoundInclusive[0],
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == countMissingValues);
   CHECK(std::numeric_limits<FloatEbmType>::lowest() == minNonInfinityValue);
   CHECK(1 == countNegativeInfinity);
   CHECK(std::numeric_limits<FloatEbmType>::max() == maxNonInfinityValue);
   CHECK(1 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateUniformBinCuts, 1 cut, -infinity, lowest + 1, max - 1, and +infinity") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues {
      -std::numeric_limits<FloatEbmType>::infinity(),
      std::nextafter(std::numeric_limits<FloatEbmType>::lowest(), 0),
      std::nextafter(std::numeric_limits<FloatEbmType>::max(), 0),
      std::numeric_limits<FloatEbmType>::infinity()
   };
   const std::vector<FloatEbmType> expectedBinCuts { 0 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   GenerateUniformBinCuts(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countBinCuts,
      &binCutsLowerBoundInclusive[0],
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == countMissingValues);
   CHECK(std::nextafter(std::numeric_limits<FloatEbmType>::lowest(), 0)
      == minNonInfinityValue);
   CHECK(1 == countNegativeInfinity);
   CHECK(std::nextafter(std::numeric_limits<FloatEbmType>::max(), 0)
      == maxNonInfinityValue);
   CHECK(1 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

