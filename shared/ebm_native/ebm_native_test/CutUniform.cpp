// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"

static const TestPriority k_filePriority = TestPriority::CutUniform;

constexpr FloatEbmType illegalVal = FloatEbmType { -888.88 };

TEST_CASE("CutUniform, 0 samples") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues {};
   const std::vector<FloatEbmType> expectedCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0],
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

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutUniform, only missing") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues { std::numeric_limits<FloatEbmType>::quiet_NaN(), std::numeric_limits<FloatEbmType>::quiet_NaN() };
   const std::vector<FloatEbmType> expectedCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0],
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

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutUniform, one cut, -infinity") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues { -std::numeric_limits<FloatEbmType>::infinity(), -std::numeric_limits<FloatEbmType>::infinity() };
   const std::vector<FloatEbmType> expectedCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0],
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

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutUniform, one cut, +infinity") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues { std::numeric_limits<FloatEbmType>::infinity(), std::numeric_limits<FloatEbmType>::infinity() };
   const std::vector<FloatEbmType> expectedCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0],
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

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutUniform, one cut, -infinity and +infinity") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues { -std::numeric_limits<FloatEbmType>::infinity(), std::numeric_limits<FloatEbmType>::infinity() };
   const std::vector<FloatEbmType> expectedCuts { };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0],
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

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutUniform, one cut, -infinity, mid-val, +infinity") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues { -std::numeric_limits<FloatEbmType>::infinity(), 7, std::numeric_limits<FloatEbmType>::infinity() };
   const std::vector<FloatEbmType> expectedCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0],
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == countMissingValues);
   CHECK(7 == minNonInfinityValue);
   CHECK(1 == countNegativeInfinity);
   CHECK(7 == maxNonInfinityValue);
   CHECK(1 == countPositiveInfinity);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutUniform, one item") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues { 1 };
   const std::vector<FloatEbmType> expectedCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0],
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

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutUniform, zero cuts") {
   IntEbmType countCuts = 0;

   std::vector<FloatEbmType> featureValues { 1, 2, 3, 4 };
   const std::vector<FloatEbmType> expectedCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0],
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

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutUniform, identical values") {
   IntEbmType countCuts = 2;

   std::vector<FloatEbmType> featureValues { 1, 1, std::numeric_limits<FloatEbmType>::quiet_NaN(), 1 };
   const std::vector<FloatEbmType> expectedCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0],
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

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutUniform, underflow") {
   IntEbmType countCuts = 9;

   std::vector<FloatEbmType> featureValues { 0, std::numeric_limits<FloatEbmType>::denorm_min() };
   const std::vector<FloatEbmType> expectedCuts { std::numeric_limits<FloatEbmType>::denorm_min() };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0],
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

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutUniform, normal") {
   IntEbmType countCuts = 9;

   std::vector<FloatEbmType> featureValues { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
   const std::vector<FloatEbmType> expectedCuts { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0],
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

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutUniform, 1 cut, -infinity, lowest, max, and +infinity") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues {
      -std::numeric_limits<FloatEbmType>::infinity(),
      std::numeric_limits<FloatEbmType>::lowest(),
      std::numeric_limits<FloatEbmType>::max(),
      std::numeric_limits<FloatEbmType>::infinity()
   };
   const std::vector<FloatEbmType> expectedCuts { 0 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0],
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

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutUniform, 1 cut, -infinity, lowest + 1, max - 1, and +infinity") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues {
      -std::numeric_limits<FloatEbmType>::infinity(),
      std::nextafter(std::numeric_limits<FloatEbmType>::lowest(), 0),
      std::nextafter(std::numeric_limits<FloatEbmType>::max(), 0),
      std::numeric_limits<FloatEbmType>::infinity()
   };
   const std::vector<FloatEbmType> expectedCuts { 0 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0],
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

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

