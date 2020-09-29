// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderEbmNativeTest.h"

#include "ebm_native.h"
#include "EbmNativeTest.h"

static const TestPriority k_filePriority = TestPriority::GenerateWinsorizedBinCuts;

constexpr FloatEbmType illegalVal = FloatEbmType { -888.88 };

TEST_CASE("GenerateWinsorizedBinCuts, 0 samples") {
   IntEbmType countBinCuts = 3;

   std::vector<FloatEbmType> featureValues {};
   const std::vector<FloatEbmType> expectedBinCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

TEST_CASE("GenerateWinsorizedBinCuts, only missing") {
   IntEbmType countBinCuts = 3;

   std::vector<FloatEbmType> featureValues { std::numeric_limits<FloatEbmType>::quiet_NaN(), std::numeric_limits<FloatEbmType>::quiet_NaN() };
   const std::vector<FloatEbmType> expectedBinCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

TEST_CASE("GenerateWinsorizedBinCuts, one item") {
   IntEbmType countBinCuts = 3;

   std::vector<FloatEbmType> featureValues { 1 };
   const std::vector<FloatEbmType> expectedBinCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

TEST_CASE("GenerateWinsorizedBinCuts, zero cuts") {
   IntEbmType countBinCuts = 0;

   std::vector<FloatEbmType> featureValues { 1, 2, 3, 4 };
   const std::vector<FloatEbmType> expectedBinCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

TEST_CASE("GenerateWinsorizedBinCuts, one cut, identical values") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues { 1, 1, std::numeric_limits<FloatEbmType>::quiet_NaN(), 1 };
   const std::vector<FloatEbmType> expectedBinCuts { };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

TEST_CASE("GenerateWinsorizedBinCuts, one cut, even") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues { 1, 2, 3, 4 };
   const std::vector<FloatEbmType> expectedBinCuts { 2.5 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

TEST_CASE("GenerateWinsorizedBinCuts, one cut, odd") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues { 1, 2, 3.5, 4, 5 };
   const std::vector<FloatEbmType> expectedBinCuts { 3 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
   CHECK(0 == countMissingValues);
   CHECK(1 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(5 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateWinsorizedBinCuts, one cut, even, two loops") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues { 1, 2, 2, 4 };
   const std::vector<FloatEbmType> expectedBinCuts { 2.5 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

TEST_CASE("GenerateWinsorizedBinCuts, one cut, odd, two loops") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues { 1, 4, 4, 4, 5 };
   const std::vector<FloatEbmType> expectedBinCuts { 3 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
   CHECK(0 == countMissingValues);
   CHECK(1 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(5 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateWinsorizedBinCuts, one cut, even, two loops, exit up") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues { 2, 2, 2, 4 };
   const std::vector<FloatEbmType> expectedBinCuts { 3 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
   CHECK(0 == countMissingValues);
   CHECK(2 == minNonInfinityValue);
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

TEST_CASE("GenerateWinsorizedBinCuts, one cut, odd, two loops, exit up") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues { 4, 4, 4, 4, 5 };
   const std::vector<FloatEbmType> expectedBinCuts { 4.5 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
   CHECK(0 == countMissingValues);
   CHECK(4 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(5 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateWinsorizedBinCuts, one cut, even, two loops, exit down") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues { 1, 2, 2, 2 };
   const std::vector<FloatEbmType> expectedBinCuts { 1.5 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
   CHECK(0 == countMissingValues);
   CHECK(1 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(2 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateWinsorizedBinCuts, one cut, odd, two loops, exit up") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues { 1, 4, 4, 4, 4 };
   const std::vector<FloatEbmType> expectedBinCuts { 2.5 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

TEST_CASE("GenerateWinsorizedBinCuts, one cut, -infinity") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues { -std::numeric_limits<FloatEbmType>::infinity(), -std::numeric_limits<FloatEbmType>::infinity() };
   const std::vector<FloatEbmType> expectedBinCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

TEST_CASE("GenerateWinsorizedBinCuts, one cut, +infinity") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues { std::numeric_limits<FloatEbmType>::infinity(), std::numeric_limits<FloatEbmType>::infinity() };
   const std::vector<FloatEbmType> expectedBinCuts {};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

TEST_CASE("GenerateWinsorizedBinCuts, one cut, -infinity and +infinity") {
   IntEbmType countBinCuts = 1;

   std::vector<FloatEbmType> featureValues { -std::numeric_limits<FloatEbmType>::infinity(), std::numeric_limits<FloatEbmType>::infinity() };
   const std::vector<FloatEbmType> expectedBinCuts { 0 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

TEST_CASE("GenerateWinsorizedBinCuts, outer test, cuts both sides") {
   IntEbmType countBinCuts = 3;

   std::vector<FloatEbmType> featureValues { 0, 1, 1, 1, 1, 1, 1, 7 };
   const std::vector<FloatEbmType> expectedBinCuts { 0.5, 4 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
   CHECK(0 == countMissingValues);
   CHECK(0 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(7 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateWinsorizedBinCuts, outer test, cut bottom") {
   IntEbmType countBinCuts = 3;

   std::vector<FloatEbmType> featureValues { 0, 1, 1, 1, 1, 1, 1, 1 };
   const std::vector<FloatEbmType> expectedBinCuts { 0.5 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
   CHECK(0 == countMissingValues);
   CHECK(0 == minNonInfinityValue);
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

TEST_CASE("GenerateWinsorizedBinCuts, outer test, cut top") {
   IntEbmType countBinCuts = 3;

   std::vector<FloatEbmType> featureValues { 1, 1, 1, 1, 1, 1, 1, 7 };
   const std::vector<FloatEbmType> expectedBinCuts { 4 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
   CHECK(0 == countMissingValues);
   CHECK(1 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(7 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateWinsorizedBinCuts, outer test, no cuts") {
   IntEbmType countBinCuts = 3;

   std::vector<FloatEbmType> featureValues { 1, 1, 1, 1, 1, 1, 1, 1 };
   const std::vector<FloatEbmType> expectedBinCuts { };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

TEST_CASE("GenerateWinsorizedBinCuts, center, one transition") {
   IntEbmType countBinCuts = 3;

   std::vector<FloatEbmType> featureValues { 1, 1, 1, 1, 2, 2, 2, 2 };
   const std::vector<FloatEbmType> expectedBinCuts { 1.5 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
   CHECK(0 == countMissingValues);
   CHECK(1 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(2 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateWinsorizedBinCuts, center, two transitions") {
   IntEbmType countBinCuts = 3;

   std::vector<FloatEbmType> featureValues { 1, 1, 1, 2, 2, 3, 3, 3 };
   const std::vector<FloatEbmType> expectedBinCuts { 1.5, 2.5 };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
   CHECK(0 == countMissingValues);
   CHECK(1 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(3 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateWinsorizedBinCuts, two cuts") {
   IntEbmType countBinCuts = 2;

   std::vector<FloatEbmType> featureValues { 0, 1, 2, 3, 4, 5 };
   const std::vector<FloatEbmType> expectedBinCuts { 2, std::nextafter(3, std::numeric_limits<FloatEbmType>::max()) };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
   CHECK(0 == countMissingValues);
   CHECK(0 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(5 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateWinsorizedBinCuts, three cuts") {
   IntEbmType countBinCuts = 3;

   std::vector<FloatEbmType> featureValues { 0, 1, 2, 3, 4, 5, 6, 7 };
   const std::vector<FloatEbmType> expectedBinCuts {2, std::nextafter(3.5, std::numeric_limits<FloatEbmType>::max()), std::nextafter(5, std::numeric_limits<FloatEbmType>::max())};

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
   CHECK(0 == countMissingValues);
   CHECK(0 == minNonInfinityValue);
   CHECK(0 == countNegativeInfinity);
   CHECK(7 == maxNonInfinityValue);
   CHECK(0 == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateWinsorizedBinCuts, four cuts") {
   IntEbmType countBinCuts = 4;

   std::vector<FloatEbmType> featureValues { 0, 1, 2, 3, 5, 7, 8, 9, 10 };
   const std::vector<FloatEbmType> expectedBinCuts { 2, 4, 6, std::nextafter(8, std::numeric_limits<FloatEbmType>::max()) };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

TEST_CASE("GenerateWinsorizedBinCuts, one cut, -infinity, lowest, max, and +infinity") {
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

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

TEST_CASE("GenerateWinsorizedBinCuts, one cut, -infinity, lowest + 1, max - 1, and +infinity") {
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

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

TEST_CASE("GenerateWinsorizedBinCuts, 3 cuts, -infinity, lowest, max, and +infinity") {
   IntEbmType countBinCuts = 3;

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

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

TEST_CASE("GenerateWinsorizedBinCuts, 3 cuts, -infinity, lowest + 1, max - 1, and +infinity") {
   IntEbmType countBinCuts = 3;

   std::vector<FloatEbmType> featureValues {
      -std::numeric_limits<FloatEbmType>::infinity(),
      std::nextafter(std::numeric_limits<FloatEbmType>::lowest(), 0),
      std::nextafter(std::numeric_limits<FloatEbmType>::max(), 0),
      std::numeric_limits<FloatEbmType>::infinity()
   };
   const std::vector<FloatEbmType> expectedBinCuts { 
      std::nextafter(std::numeric_limits<FloatEbmType>::lowest(), 0), 
      0,
      std::numeric_limits<FloatEbmType>::max()
   };

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(0 == countBinCuts ? 1 : countBinCuts, illegalVal);

   IntEbmType ret = GenerateWinsorizedBinCuts(
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
   CHECK(0 == ret);
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

