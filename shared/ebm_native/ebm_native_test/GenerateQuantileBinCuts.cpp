// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderEbmNativeTest.h"

#include "ebm_native.h"
#include "EbmNativeTest.h"
#include "RandomStreamTest.h"

static const TestPriority k_filePriority = TestPriority::GenerateQuantileBinCuts;

class CompareFloatWithNan final {
public:
   inline bool operator() (const FloatEbmType & lhs, const FloatEbmType & rhs) const noexcept {
      if(std::isnan(lhs)) {
         return false;
      } else {
         if(std::isnan(rhs)) {
            return true;
         } else {
            // NEVER check for exact equality (as a precondition is ok), since then we'd violate the weak ordering rule
            return lhs < rhs;
         }
      }
   }
};

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

void TestQuantileBinning(
   TestCaseHidden & testCaseHidden,
   const bool bTestReverse,
   const bool bSmart,
   const size_t cBinCutsMax,
   const size_t cSamplesPerBinMin,
   const std::vector<FloatEbmType> featureValues,
   const std::vector<FloatEbmType> expectedBinCuts
) {
   const IntEbmType countBinCutsMax = cBinCutsMax;
   const IntEbmType countSamplesPerBinMin = cSamplesPerBinMin;

   constexpr FloatEbmType illegalVal = FloatEbmType { -888.88 };
   std::vector<FloatEbmType> binCutsLowerBoundInclusive(cBinCutsMax + 2, illegalVal); // allocate values at ends

   std::vector<FloatEbmType> featureValues1(featureValues);
   std::vector<FloatEbmType> featureValues2(featureValues);
   std::transform(featureValues2.begin(), featureValues2.end(), featureValues2.begin(), 
      [](FloatEbmType & val) { return -val; });

   IntEbmType countMissingValues;
   FloatEbmType minNonInfinityValue;
   IntEbmType countNegativeInfinity;
   FloatEbmType maxNonInfinityValue;
   IntEbmType countPositiveInfinity;

   // do this before calling GenerateQuantileBinCuts, since GenerateQuantileBinCuts modifies featureValues
   IntEbmType countMissingValuesExpected;
   FloatEbmType minNonInfinityValueExpected;
   IntEbmType countNegativeInfinityExpected;
   FloatEbmType maxNonInfinityValueExpected;
   IntEbmType countPositiveInfinityExpected;
   GetExpectedStats(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      countMissingValuesExpected,
      minNonInfinityValueExpected,
      countNegativeInfinityExpected,
      maxNonInfinityValueExpected,
      countPositiveInfinityExpected
   );

   IntEbmType countBinCuts = countBinCutsMax;
   IntEbmType ret = GenerateQuantileBinCuts(
      featureValues1.size(),
      0 == featureValues1.size() ? nullptr : &featureValues1[0],
      countSamplesPerBinMin,
      bSmart ? EBM_TRUE : EBM_FALSE,
      k_randomSeed,
      &countBinCuts,
      &binCutsLowerBoundInclusive[1],
      &countMissingValues,
      &minNonInfinityValue,
      &countNegativeInfinity,
      &maxNonInfinityValue,
      &countPositiveInfinity
   );
   CHECK(0 == ret);

   CHECK(illegalVal == binCutsLowerBoundInclusive[0]);
   for(size_t iCheck = static_cast<size_t>(countBinCuts) + 1; iCheck < cBinCutsMax + 2; ++iCheck) {
      CHECK(illegalVal == binCutsLowerBoundInclusive[iCheck]);
   }

   CHECK(countMissingValuesExpected == countMissingValues);
   CHECK(minNonInfinityValueExpected == minNonInfinityValue);
   CHECK(countNegativeInfinityExpected == countNegativeInfinity);
   CHECK(maxNonInfinityValueExpected == maxNonInfinityValue);
   CHECK(countPositiveInfinityExpected == countPositiveInfinity);

   size_t cBinCuts = static_cast<size_t>(countBinCuts);
   CHECK(expectedBinCuts.size() == cBinCuts);
   if(expectedBinCuts.size() == cBinCuts) {
      for(size_t i = 0; i < cBinCuts; ++i) {
         CHECK_APPROX(expectedBinCuts[i], binCutsLowerBoundInclusive[i + 1]);
      }
   }

   if(bTestReverse) {
      // try the reverse now.  We try very hard to ensure that we preserve symmetry in the cutting algorithm

      countBinCuts = countBinCutsMax;
      ret = GenerateQuantileBinCuts(
         featureValues2.size(),
         0 == featureValues2.size() ? nullptr : &featureValues2[0],
         countSamplesPerBinMin,
         bSmart ? EBM_TRUE : EBM_FALSE,
         k_randomSeed,
         &countBinCuts,
         &binCutsLowerBoundInclusive[1],
         &countMissingValues,
         &minNonInfinityValue,
         &countNegativeInfinity,
         &maxNonInfinityValue,
         &countPositiveInfinity
      );
      CHECK(0 == ret);

      CHECK(illegalVal == binCutsLowerBoundInclusive[0]);
      for(size_t iCheck = static_cast<size_t>(countBinCuts) + 1; iCheck < cBinCutsMax + 2; ++iCheck) {
         CHECK(illegalVal == binCutsLowerBoundInclusive[iCheck]);
      }

      CHECK(countMissingValuesExpected == countMissingValues);
      CHECK(-maxNonInfinityValueExpected == minNonInfinityValue);
      CHECK(countPositiveInfinityExpected == countNegativeInfinity);
      CHECK(-minNonInfinityValueExpected == maxNonInfinityValue);
      CHECK(countNegativeInfinityExpected == countPositiveInfinity);

      cBinCuts = static_cast<size_t>(countBinCuts);
      CHECK(expectedBinCuts.size() == cBinCuts);
      if(expectedBinCuts.size() == cBinCuts) {
         for(size_t i = 0; i < cBinCuts; ++i) {
            CHECK_APPROX(-expectedBinCuts[cBinCuts - 1 - i], binCutsLowerBoundInclusive[i + 1]);
         }
      }
   }
}

TEST_CASE("GenerateQuantileBinCuts, 0 samples") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { };
   const std::vector<FloatEbmType> expectedBinCuts { };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, only missing") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { std::numeric_limits<FloatEbmType>::quiet_NaN(), std::numeric_limits<FloatEbmType>::signaling_NaN() };
   const std::vector<FloatEbmType> expectedBinCuts {};

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, zero cuts") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 0;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { 1, 2 };
   const std::vector<FloatEbmType> expectedBinCuts {};

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, too small") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 5 };
   const std::vector<FloatEbmType> expectedBinCuts {};

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, positive and +infinity") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { 11, std::numeric_limits<FloatEbmType>::infinity() };
   const std::vector<FloatEbmType> expectedBinCuts { 4e+154 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, positive and +max") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { 11, std::numeric_limits<FloatEbmType>::max() };
   const std::vector<FloatEbmType> expectedBinCuts { 4e+154 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, one and +max minus one tick backwards") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { FloatEbmType { 1 }, std::nextafter(std::numeric_limits<FloatEbmType>::max(), FloatEbmType { 0 }) };
   const std::vector<FloatEbmType> expectedBinCuts { 1e+154 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, zero and +max") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { 0, std::numeric_limits<FloatEbmType>::max() };
   const std::vector<FloatEbmType> expectedBinCuts { 9e+307 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, negative and +max") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { -11, std::numeric_limits<FloatEbmType>::max() };
   const std::vector<FloatEbmType> expectedBinCuts { 0 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, negative and -infinity") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { -11, -std::numeric_limits<FloatEbmType>::infinity() };
   const std::vector<FloatEbmType> expectedBinCuts { -4e+154 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, negative and lowest") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { -11, std::numeric_limits<FloatEbmType>::lowest() };
   const std::vector<FloatEbmType> expectedBinCuts { -4e+154 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, -1 and lowest plus one tick backwards") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { FloatEbmType { -1 }, std::nextafter(std::numeric_limits<FloatEbmType>::lowest(), FloatEbmType { 0 }) };
   const std::vector<FloatEbmType> expectedBinCuts { -1e+154 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, zero and lowest") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { 0, std::numeric_limits<FloatEbmType>::lowest() };
   const std::vector<FloatEbmType> expectedBinCuts { -9e+307 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, positive and lowest") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { 11, std::numeric_limits<FloatEbmType>::lowest() };
   const std::vector<FloatEbmType> expectedBinCuts { 0 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, splitable") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 0, 1, 2, 3 };
   const std::vector<FloatEbmType> expectedBinCuts { 1.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, splitable (first interior check not splitable)") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 3;
   const std::vector<FloatEbmType> featureValues { 0, 1, 5, 5, 7, 8, 9 };
   const std::vector<FloatEbmType> expectedBinCuts { 6 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, splitable except middle isn't available") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 3;
   const std::vector<FloatEbmType> featureValues { 0, 1, 5, 5, 8, 9 };
   const std::vector<FloatEbmType> expectedBinCuts { };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, unsplitable") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 5, 5, 5, 5 };
   const std::vector<FloatEbmType> expectedBinCuts {};

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, left+unsplitable") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 1, 5, 5, 5 };
   const std::vector<FloatEbmType> expectedBinCuts {};

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, unsplitable+right") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 5, 5, 5, 9 };
   const std::vector<FloatEbmType> expectedBinCuts {};

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, left+unsplitable+right") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 1, 5, 5, 9 };
   const std::vector<FloatEbmType> expectedBinCuts {};

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, unsplitable+unsplitable") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 4, 4, 6, 6 };
   const std::vector<FloatEbmType> expectedBinCuts { 5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, left+unsplitable+unsplitable") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 1, 4, 4, 6, 6, 6 };
   const std::vector<FloatEbmType> expectedBinCuts { 5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, unsplitable+unsplitable+right") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 4, 4, 6, 6, 6, 9 };
   const std::vector<FloatEbmType> expectedBinCuts { 5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, unsplitable+mid+unsplitable") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 4, 4, 4, 5, 6, 6 };
   const std::vector<FloatEbmType> expectedBinCuts { 4.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, left+unsplitable+mid+unsplitable") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 1, 4, 4, 5, 6, 6 };
   const std::vector<FloatEbmType> expectedBinCuts { 4.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, unsplitable+mid+unsplitable+right") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 4, 4, 5, 6, 6, 9 };
   const std::vector<FloatEbmType> expectedBinCuts { 5.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, unsplitable+splitable") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 5, 5, 7, 8 };
   const std::vector<FloatEbmType> expectedBinCuts { 6 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, left+unsplitable+splitable") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 1, 5, 5, 5, 7, 8 };
   const std::vector<FloatEbmType> expectedBinCuts { 6 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, splitable+unsplitable") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 2, 3, 5, 5 };
   const std::vector<FloatEbmType> expectedBinCuts { 4 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, splitable+unsplitable+right") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 1, 2, 3, 5, 5, 7 };
   const std::vector<FloatEbmType> expectedBinCuts { 4 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, splitable+unsplitable+splitable") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 2, 3, 5, 5, 7, 8 };
   const std::vector<FloatEbmType> expectedBinCuts { 4, 6 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, unsplitable+splitable+unsplitable") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 2, 2, 4, 6, 8, 8 };
   const std::vector<FloatEbmType> expectedBinCuts { 3, 7 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, left+unsplitable+splitable+unsplitable") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 1, 2, 2, 4, 5, 6, 8, 8 };
   const std::vector<FloatEbmType> expectedBinCuts { 3, 7 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, unsplitable+splitable+unsplitable+right") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 2, 2, 2, 4, 6, 8, 8, 9 };
   const std::vector<FloatEbmType> expectedBinCuts { 3, 7 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, left+unsplitable+splitable+unsplitable+right") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 1, 2, 2, 4, 6, 8, 8, 9 };
   const std::vector<FloatEbmType> expectedBinCuts { 3, 7 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, unsplitable+splitable+unsplitable+splitable") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 1, 1, 2, 3, 5, 5, 7, 8 };
   const std::vector<FloatEbmType> expectedBinCuts { 1.5, 4, 6 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, left+unsplitable+splitable+unsplitable+splitable") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 0, 1, 1, 2, 3, 5, 5, 5, 7, 8 };
   const std::vector<FloatEbmType> expectedBinCuts { 1.5, 4, 6 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, splitable+unsplitable+splitable+unsplitable") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 2, 3, 5, 5, 7, 8, 9, 9 };
   const std::vector<FloatEbmType> expectedBinCuts { 4, 6, 8.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, splitable+unsplitable+splitable+unsplitable+right") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 2, 3, 5, 5, 5, 7, 8, 9, 9, 10 };
   const std::vector<FloatEbmType> expectedBinCuts { 4, 6, 8.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, left+unsplitable+splitable+unsplitable+splitable+unsplitable+splitable+unsplitable+right") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 1, 2, 2, 3, 4, 4, 4, 5, 6, 6, 7, 8, 8, 9 };
   const std::vector<FloatEbmType> expectedBinCuts { 3.5, 4.5, 7.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, infinities") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 
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
   const std::vector<FloatEbmType> expectedBinCuts { 0 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, average division sizes that requires the ceiling instead of rounding") {
   // our algorithm makes an internal assumption that we can give each cut point a split.  This is guaranteed if we 
   // make the average length of the equal value long ranges the ceiling of the average samples per bin.  
   // This test stresses that average calculation by having an average bin lenght of 2.2222222222 but if you use 
   // a bin width of 2, then there are 3 cut points that can't get any cuts.  3 cut points means that even if you 
   // don't give the first and last SplittingRanges an actual cut point, which can be reasonalbe since the 
   // first and last SplittingRanges are special in that they may have no long ranges on the tail ends, 
   // you still end up with one or more SplittingRanges that can't have a cut if you don't take the ceiling.

   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 26;
   constexpr size_t cSamplesPerBinMin = 2;
   const std::vector<FloatEbmType> featureValues { 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
      13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 
      27, 27, 28, 28, 29, 29, 30, 30 };
   const std::vector<FloatEbmType> expectedBinCuts { 2.5, 4.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5,
      12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5 };
   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, reversibility, 2") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { -1, 1 };
   const std::vector<FloatEbmType> expectedBinCuts { 0 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, reversibility, 3") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { -2, 1, 2 };
   const std::vector<FloatEbmType> expectedBinCuts { 0, 1.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, reversibility") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { -2, -1, 1, 2 };
   const std::vector<FloatEbmType> expectedBinCuts { -1.5, 0, 1.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, imbalanced") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { -3, -1, 2, 3 };
   const std::vector<FloatEbmType> expectedBinCuts { -2, 0, 2.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, extreme tails") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { std::numeric_limits<FloatEbmType>::lowest(), 1, 2, 3, std::numeric_limits<FloatEbmType>::max() };
   const std::vector<FloatEbmType> expectedBinCuts { 0.5, 1.5, 2.5, 3.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, far tails") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { -1, 1, 2, 3, 5 };
   const std::vector<FloatEbmType> expectedBinCuts { 0.5, 1.5, 2.5, 3.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, close tails") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { 0.9, 1, 2, 3, 3.1 };
   const std::vector<FloatEbmType> expectedBinCuts { 0.95, 1.5, 2.5, 3.05 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, non-smart") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = false;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues { std::numeric_limits<FloatEbmType>::lowest(), 0, 1000, 10000000, std::numeric_limits<FloatEbmType>::max() };
   const std::vector<FloatEbmType> expectedBinCuts { -8.9884656743115785e+307, 500, 5000500, 8.9884656743115785e+307 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, overflow interpretable ends") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues {
      std::numeric_limits<FloatEbmType>::lowest(),
      std::nextafter(std::numeric_limits<FloatEbmType>::lowest(), FloatEbmType { 0 }),
      std::nextafter(std::numeric_limits<FloatEbmType>::max(), FloatEbmType { 0 }),
      std::numeric_limits<FloatEbmType>::max()
   };

   const std::vector<FloatEbmType> expectedBinCuts {
      std::nextafter(std::numeric_limits<FloatEbmType>::lowest(), FloatEbmType { 0 }),
      0,
      std::numeric_limits<FloatEbmType>::max()
   };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, maximum non-overflow interpretable ends") {
   constexpr bool bTestReverse = true;
   constexpr bool bSmart = true;
   constexpr size_t cBinCutsMax = 1000;
   constexpr size_t cSamplesPerBinMin = 1;
   const std::vector<FloatEbmType> featureValues {
      std::numeric_limits<FloatEbmType>::lowest(),
      std::nextafter(std::numeric_limits<FloatEbmType>::lowest(), FloatEbmType { 0 }),
      std::numeric_limits<FloatEbmType>::max() - std::nextafter(std::numeric_limits<FloatEbmType>::max(), FloatEbmType { 0 }),
      std::numeric_limits<FloatEbmType>::max()
   };

   const std::vector<FloatEbmType> expectedBinCuts {
      std::nextafter(std::numeric_limits<FloatEbmType>::lowest(), FloatEbmType { 0 }),
      0,
      2.0000000000000001e+300
   };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, stress test the guarantee of one split per SplittingRange, by 2") {
   constexpr IntEbmType countSamplesPerBinMin = 1;
   constexpr size_t cItemsPerRange = 10;
   constexpr size_t cInteriorRanges = 3;
   constexpr size_t cRemoveCuts = 1;

   std::vector<FloatEbmType> featureValues(3 + cInteriorRanges * cItemsPerRange, 0);
   std::vector<FloatEbmType> expectedBinCuts;

   featureValues[0] = 0;
   for(size_t iRange = 0; iRange < cInteriorRanges; ++iRange) {
      for(size_t i = 1 + cItemsPerRange * iRange; i < 1 + (cItemsPerRange * (iRange + 1)); ++i) {
         const size_t iRangePlusOne = iRange + size_t { 1 };
         featureValues[i] = static_cast<FloatEbmType>(iRangePlusOne);
      }
      expectedBinCuts.push_back(FloatEbmType { 0.5 } + static_cast<FloatEbmType>(iRange));
   }
   expectedBinCuts.push_back(FloatEbmType { 0.5 } + static_cast<FloatEbmType>(cInteriorRanges));

   featureValues[cInteriorRanges * cItemsPerRange + 1] = static_cast<FloatEbmType>(1 + cInteriorRanges);
   featureValues[cInteriorRanges * cItemsPerRange + 2] = static_cast<FloatEbmType>(1 + cInteriorRanges);

   static bool bOne = 1 == cRemoveCuts;
   static bool bTwo = 2 == cRemoveCuts;
   if(bOne) {
      expectedBinCuts.erase(expectedBinCuts.begin());
   } else if(bTwo) {
      expectedBinCuts.erase(expectedBinCuts.begin());
      expectedBinCuts.erase(expectedBinCuts.end() - 1);
   } else {
      assert(0 == cRemoveCuts);
   }

   constexpr bool bSmart = true;
   constexpr bool bTestReverse = true;
   constexpr size_t cBinCutsMax = cInteriorRanges + 1 - cRemoveCuts;
   constexpr size_t cSamplesPerBinMin = 1;

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cBinCutsMax,
      cSamplesPerBinMin,
      featureValues,
      expectedBinCuts
   );
}

TEST_CASE("GenerateQuantileBinCuts, randomized fairness check") {
   RandomStreamTest randomStream(k_randomSeed);
   if(!randomStream.IsSuccess()) {
      exit(1);
   }

   constexpr bool bSmart = true;
   constexpr IntEbmType countSamplesPerBinMin = 1;
   constexpr IntEbmType countSamples = 100;
   FloatEbmType featureValues[countSamples]; // preserve these for debugging purposes
   FloatEbmType featureValuesForward[countSamples];
   FloatEbmType featureValuesReversed[countSamples];

   constexpr IntEbmType randomMaxMax = countSamples - 1; // this doesn't need to be exactly countSamples - 1, but this number gives us chunky sets
   size_t cutHistogram[randomMaxMax];
   constexpr size_t cCutHistogram = sizeof(cutHistogram) / sizeof(cutHistogram[0]);
   // our random numbers can be any numbers from 0 to randomMaxMax (inclusive), which gives us randomMaxMax - 1 possible cut points between them
   static_assert(1 == cCutHistogram % 2, "cutHistogram must have a center value that is perfectly in the middle");

   constexpr size_t cBinCuts = 9;
   FloatEbmType binCutsLowerBoundInclusiveForward[cBinCuts];
   FloatEbmType binCutsLowerBoundInclusiveReversed[cBinCuts];

   memset(cutHistogram, 0, sizeof(cutHistogram));

   for(int iIteration = 0; iIteration < 100; ++iIteration) {
      for(size_t randomMax = 1; randomMax <= randomMaxMax; randomMax += 2) {
         // since randomMax isn't larger than the number of samples, we'll always be chunky.  This is good for testing range collisions
         for(size_t iSample = 0; iSample < countSamples; ++iSample) {
            bool bMissing = 0 == randomStream.Next(countSamples); // some datasetes will have zero missing values, some will have 1 or more
            size_t iRandom = randomStream.Next(randomMax + 1) + 1;
            featureValues[iSample] = bMissing ? std::numeric_limits<FloatEbmType>::quiet_NaN() : static_cast<FloatEbmType>(iRandom);
         }

         IntEbmType countMissingValues;
         FloatEbmType minNonInfinityValue;
         IntEbmType countNegativeInfinity;
         FloatEbmType maxNonInfinityValue;
         IntEbmType countPositiveInfinity;

         // do this before calling GenerateQuantileBinCuts, since GenerateQuantileBinCuts modifies featureValues
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

         memcpy(featureValuesForward, featureValues, sizeof(featureValues[0]) * countSamples);

         IntEbmType countBinCutsForward = static_cast<IntEbmType>(cBinCuts);
         IntEbmType ret = GenerateQuantileBinCuts(
            countSamples,
            featureValuesForward,
            countSamplesPerBinMin,
            bSmart ? EBM_TRUE : EBM_FALSE,
            k_randomSeed + iIteration,
            &countBinCutsForward,
            binCutsLowerBoundInclusiveForward,
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

         //DisplayCuts(
         //   countSamples,
         //   featureValues,
         //   cBinCuts + 1,
         //   countSamplesPerBinMin,
         //   countBinCutsForward,
         //   binCutsLowerBoundInclusiveForward,
         //   0 != countMissingValues,
         //   minNonInfinityValue,
         //   maxNonInfinityValue
         //);

         std::transform(featureValues, featureValues + countSamples, featureValuesReversed,
            [](FloatEbmType & val) { return -val; });

         IntEbmType countBinCutsReversed = static_cast<IntEbmType>(cBinCuts);
         ret = GenerateQuantileBinCuts(
            countSamples,
            featureValuesReversed,
            countSamplesPerBinMin,
            bSmart ? EBM_TRUE : EBM_FALSE,
            k_randomSeed + iIteration,
            &countBinCutsReversed,
            binCutsLowerBoundInclusiveReversed,
            &countMissingValues,
            &minNonInfinityValue,
            &countNegativeInfinity,
            &maxNonInfinityValue,
            &countPositiveInfinity
         );
         CHECK(0 == ret);

         CHECK(countMissingValuesExpected == countMissingValues);
         CHECK(-maxNonInfinityValueExpected == minNonInfinityValue);
         CHECK(countPositiveInfinityExpected == countNegativeInfinity);
         CHECK(-minNonInfinityValueExpected == maxNonInfinityValue);
         CHECK(countNegativeInfinityExpected == countPositiveInfinity);

         CHECK(countBinCutsForward == countBinCutsReversed);

         std::sort(featureValues, featureValues + countSamples, CompareFloatWithNan());

         assert(1 == randomMax % 2); // our random numbers need a center value as well
         constexpr size_t iHistogramExactMiddle = cCutHistogram / 2;
         const size_t iCutExactMiddle = randomMax / 2;
         assert(iCutExactMiddle <= iHistogramExactMiddle);
         const size_t iShiftToMiddle = iHistogramExactMiddle - iCutExactMiddle;
         const size_t cBinCutsReturned = static_cast<size_t>(countBinCutsForward);
         for(size_t iCutPoint = 0; iCutPoint < cBinCutsReturned; ++iCutPoint) {
            const FloatEbmType cutPointForward = binCutsLowerBoundInclusiveForward[iCutPoint];
            if(countBinCutsForward == countBinCutsReversed) {
               const FloatEbmType cutPointReversed = -binCutsLowerBoundInclusiveReversed[cBinCutsReturned - 1 - iCutPoint];

               const FloatEbmType cutPointForwardNext = *std::upper_bound(featureValues, featureValues + countSamples - 1 - countMissingValuesExpected, cutPointForward);
               const FloatEbmType cutPointReversedNext = *std::upper_bound(featureValues, featureValues + countSamples - 1 - countMissingValuesExpected, cutPointReversed);

               CHECK_APPROX(cutPointForwardNext, cutPointReversedNext);
            }
            // cutPoint can be a number between 0.5 and (randomMax - 0.5)
            const size_t iCut = static_cast<size_t>(std::round(cutPointForward - FloatEbmType { 1.5 }));
            const size_t iSymetricCut = iShiftToMiddle + iCut;
            assert(iSymetricCut < cCutHistogram);
            if(iSymetricCut < cCutHistogram) {
               ++cutHistogram[iSymetricCut];
            } else {
               // this shouldn't happen, but the compiler is giving a warning because it can't predict iSymetricCut
               assert(false);
            }
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

TEST_CASE("GenerateQuantileBinCuts, chunky randomized check") {
   RandomStreamTest randomStream(k_randomSeed);
   if(!randomStream.IsSuccess()) {
      exit(1);
   }

   constexpr bool bSmart = true;
   constexpr size_t cSamplesMin = 1;
   constexpr size_t cSamplesMax = 250;
   constexpr size_t cBinCutsMin = 1;
   constexpr size_t cBinCutsMax = 70;
   constexpr IntEbmType countSamplesPerBinMinMin = 1;
   constexpr IntEbmType countSamplesPerBinMinMax = 3;

   constexpr size_t randomValMax = 70; // the min is 1 since the value doesn't really matter

   FloatEbmType binCutsLowerBoundInclusiveForward[cBinCutsMax];
   FloatEbmType binCutsLowerBoundInclusiveReversed[cBinCutsMax];

   FloatEbmType featureValues[cSamplesMax]; // preserve these for debugging purposes
   FloatEbmType featureValuesForward[cSamplesMax];
   FloatEbmType featureValuesReversed[cSamplesMax];

   for(size_t iIteration = 0; iIteration < 30000; ++iIteration) {
      const size_t cSamples = randomStream.Next(cSamplesMax - cSamplesMin + 1) + cSamplesMin;
      const size_t cBinCuts = randomStream.Next(cBinCutsMax - cBinCutsMin + 1) + cBinCutsMin;
      const IntEbmType countSamplesPerBinMin = randomStream.Next(countSamplesPerBinMinMax - countSamplesPerBinMinMin + 1) + countSamplesPerBinMinMin;

      const size_t denominator = cBinCuts + size_t { 1 };
      const size_t cLongBinLength = static_cast<size_t>(
         std::ceil(static_cast<FloatEbmType>(cSamples) / static_cast<FloatEbmType>(denominator)));

      memset(featureValues, 0, sizeof(featureValues));

      size_t i = 0;
      size_t cLongRanges = randomStream.Next(6);
      for(size_t iLongRange = 0; iLongRange < cLongRanges; ++iLongRange) {
         size_t cItems = randomStream.Next(cLongBinLength) + cLongBinLength;
         size_t val = randomStream.Next(randomValMax) + 1;
         for(size_t iItem = 0; iItem < cItems; ++iItem) {
            featureValues[i % cSamples] = static_cast<FloatEbmType>(val);
            ++i;
         }
      }
      size_t cShortRanges = randomStream.Next(6);
      for(size_t iShortRange = 0; iShortRange < cShortRanges; ++iShortRange) {
         size_t cItems = randomStream.Next(cLongBinLength);
         size_t val = randomStream.Next(randomValMax) + 1;
         for(size_t iItem = 0; iItem < cItems; ++iItem) {
            featureValues[i % cSamples] = static_cast<FloatEbmType>(val);
            ++i;
         }
      }
      for(size_t iSample = 0; iSample < cSamples; ++iSample) {
         if(0 == featureValues[iSample]) {
            const size_t randomPlusOne = randomStream.Next(randomValMax) + size_t { 1 };
            featureValues[iSample] = static_cast<FloatEbmType>(randomPlusOne);
         }
      }

      const IntEbmType countSamples = static_cast<IntEbmType>(cSamples);

      IntEbmType countMissingValues;
      FloatEbmType minNonInfinityValue;
      IntEbmType countNegativeInfinity;
      FloatEbmType maxNonInfinityValue;
      IntEbmType countPositiveInfinity;

      // do this before calling GenerateQuantileBinCuts, since GenerateQuantileBinCuts modifies featureValues
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

      memcpy(featureValuesForward, featureValues, sizeof(featureValues[0]) * cSamples);

      IntEbmType countBinCutsForward = static_cast<IntEbmType>(cBinCuts);
      IntEbmType ret = GenerateQuantileBinCuts(
         countSamples,
         featureValuesForward,
         countSamplesPerBinMin,
         bSmart ? EBM_TRUE : EBM_FALSE,
         k_randomSeed + static_cast<SeedEbmType>(iIteration),
         &countBinCutsForward,
         binCutsLowerBoundInclusiveForward,
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

      std::transform(featureValues, featureValues + countSamples, featureValuesReversed,
         [](FloatEbmType & val) { return -val; });

      IntEbmType countBinCutsReversed = static_cast<IntEbmType>(cBinCuts);
      ret = GenerateQuantileBinCuts(
         countSamples,
         featureValuesReversed,
         countSamplesPerBinMin,
         bSmart ? EBM_TRUE : EBM_FALSE,
         k_randomSeed + static_cast<SeedEbmType>(iIteration),
         &countBinCutsReversed,
         binCutsLowerBoundInclusiveReversed,
         &countMissingValues,
         &minNonInfinityValue,
         &countNegativeInfinity,
         &maxNonInfinityValue,
         &countPositiveInfinity
      );
      CHECK(0 == ret);

      CHECK(countMissingValuesExpected == countMissingValues);
      CHECK(-maxNonInfinityValueExpected == minNonInfinityValue);
      CHECK(countPositiveInfinityExpected == countNegativeInfinity);
      CHECK(-minNonInfinityValueExpected == maxNonInfinityValue);
      CHECK(countNegativeInfinityExpected == countPositiveInfinity);

      CHECK(countBinCutsForward == countBinCutsReversed);
      if(countBinCutsForward == countBinCutsReversed) {
         std::sort(featureValues, featureValues + countSamples, CompareFloatWithNan());

         const size_t cBinCutsReturned = static_cast<size_t>(countBinCutsForward);
         for(size_t iCutPoint = 0; iCutPoint < cBinCutsReturned; ++iCutPoint) {
            const FloatEbmType cutPointForward = binCutsLowerBoundInclusiveForward[iCutPoint];
            const FloatEbmType cutPointReversed = -binCutsLowerBoundInclusiveReversed[cBinCutsReturned - 1 - iCutPoint];

            const FloatEbmType cutPointForwardNext = *std::upper_bound(featureValues, featureValues + countSamples - 1 - countMissingValuesExpected, cutPointForward);
            const FloatEbmType cutPointReversedNext = *std::upper_bound(featureValues, featureValues + countSamples - 1 - countMissingValuesExpected, cutPointReversed);

            CHECK_APPROX(cutPointForwardNext, cutPointReversedNext);
         }
      }
   }
}

