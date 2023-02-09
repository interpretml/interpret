// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"
#include "RandomStreamTest.hpp"

static constexpr TestPriority k_filePriority = TestPriority::CutQuantile;

class CompareFloatWithNan final {
public:
   inline bool operator() (const double & lhs, const double & rhs) const noexcept {
      // NEVER check for exact equality (as a precondition is ok), since then we'd violate the weak ordering rule
      // https://medium.com/@shiansu/strict-weak-ordering-and-the-c-stl-f7dcfa4d4e07

      if(std::isnan(lhs)) {
         return false;
      } else {
         if(std::isnan(rhs)) {
            return true;
         } else {
            return lhs < rhs;
         }
      }
   }
};

void TestQuantileBinning(
   TestCaseHidden & testCaseHidden,
   const bool bTestReverse,
   const bool bSmart,
   const size_t cCutsMax,
   const size_t cSamplesBinMin,
   const std::vector<double> featureVals,
   const std::vector<double> expectedCuts
) {
   ErrorEbm error;

   const IntEbm countCutsMax = cCutsMax;
   const IntEbm minSamplesBin = cSamplesBinMin;

   static constexpr double illegalVal = double { -888.88 };
   std::vector<double> cutsLowerBoundInclusive(cCutsMax + 2, illegalVal); // allocate values at ends

   std::vector<double> featureVals1(featureVals);
   std::vector<double> featureVals2(featureVals);
   std::transform(featureVals2.begin(), featureVals2.end(), featureVals2.begin(), 
      [](double & val) { return -val; });

   IntEbm countCuts = countCutsMax;
   error = CutQuantile(
      featureVals1.size(),
      0 == featureVals1.size() ? nullptr : &featureVals1[0],
      minSamplesBin,
      bSmart ? EBM_TRUE : EBM_FALSE,
      &countCuts,
      &cutsLowerBoundInclusive[1]
   );
   CHECK(Error_None == error);

   CHECK(illegalVal == cutsLowerBoundInclusive[0]);
   for(size_t iCheck = static_cast<size_t>(countCuts) + 1; iCheck < cCutsMax + 2; ++iCheck) {
      CHECK(illegalVal == cutsLowerBoundInclusive[iCheck]);
   }

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i + 1]);
      }
   }

   if(bTestReverse) {
      // try the reverse now.  We try very hard to ensure that we preserve symmetry in the cutting algorithm

      countCuts = countCutsMax;
      error = CutQuantile(
         featureVals2.size(),
         0 == featureVals2.size() ? nullptr : &featureVals2[0],
         minSamplesBin,
         bSmart ? EBM_TRUE : EBM_FALSE,
         &countCuts,
         &cutsLowerBoundInclusive[1]
      );
      CHECK(Error_None == error);

      CHECK(illegalVal == cutsLowerBoundInclusive[0]);
      for(size_t iCheck = static_cast<size_t>(countCuts) + 1; iCheck < cCutsMax + 2; ++iCheck) {
         CHECK(illegalVal == cutsLowerBoundInclusive[iCheck]);
      }

      cCuts = static_cast<size_t>(countCuts);
      CHECK(expectedCuts.size() == cCuts);
      if(expectedCuts.size() == cCuts) {
         for(size_t i = 0; i < cCuts; ++i) {
            CHECK_APPROX(-expectedCuts[cCuts - 1 - i], cutsLowerBoundInclusive[i + 1]);
         }
      }
   }
}

TEST_CASE("CutQuantile, 0 samples") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { };
   static const std::vector<double> expectedCuts { };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, only missing") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::signaling_NaN() };
   static const std::vector<double> expectedCuts {};

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, zero cuts") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 0;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { 1, 2 };
   static const std::vector<double> expectedCuts {};

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, too small") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 5 };
   static const std::vector<double> expectedCuts {};

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, positive and +infinity") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { 11, std::numeric_limits<double>::infinity() };
   static const std::vector<double> expectedCuts { 4e+154 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, positive and +max") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { 11, std::numeric_limits<double>::max() };
   static const std::vector<double> expectedCuts { 4e+154 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, one and +max minus one tick backwards") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { double { 1 }, FloatTickDecrementTest(std::numeric_limits<double>::max()) };
   static const std::vector<double> expectedCuts { 1e+154 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, zero and +max") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { 0, std::numeric_limits<double>::max() };
   static const std::vector<double> expectedCuts { 9e+307 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, negative and +max") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { -11, std::numeric_limits<double>::max() };
   static const std::vector<double> expectedCuts { 0 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, negative and -infinity") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { -11, -std::numeric_limits<double>::infinity() };
   static const std::vector<double> expectedCuts { -4e+154 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, negative and lowest") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { -11, std::numeric_limits<double>::lowest() };
   static const std::vector<double> expectedCuts { -4e+154 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, -1 and lowest plus one tick backwards") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { double { -1 }, FloatTickIncrementTest(std::numeric_limits<double>::lowest()) };
   static const std::vector<double> expectedCuts { -1e+154 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, zero and lowest") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { 0, std::numeric_limits<double>::lowest() };
   static const std::vector<double> expectedCuts { -9e+307 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, positive and lowest") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { 11, std::numeric_limits<double>::lowest() };
   static const std::vector<double> expectedCuts { 0 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, cuttable") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 0, 1, 2, 3 };
   static const std::vector<double> expectedCuts { 1.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, cuttable (first interior check not cuttable)") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 3;
   static const std::vector<double> featureVals { 0, 1, 5, 5, 7, 8, 9 };
   static const std::vector<double> expectedCuts { 6 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, cuttable except middle isn't available") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 3;
   static const std::vector<double> featureVals { 0, 1, 5, 5, 8, 9 };
   static const std::vector<double> expectedCuts { };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, uncuttable") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 5, 5, 5, 5 };
   static const std::vector<double> expectedCuts {};

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, left+uncuttable") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 1, 5, 5, 5 };
   static const std::vector<double> expectedCuts {};

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, uncuttable+right") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 5, 5, 5, 9 };
   static const std::vector<double> expectedCuts {};

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, left+uncuttable+right") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 1, 5, 5, 9 };
   static const std::vector<double> expectedCuts {};

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, uncuttable+uncuttable") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 4, 4, 6, 6 };
   static const std::vector<double> expectedCuts { 5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, left+uncuttable+uncuttable") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 1, 4, 4, 6, 6, 6 };
   static const std::vector<double> expectedCuts { 5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, uncuttable+uncuttable+right") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 4, 4, 6, 6, 6, 9 };
   static const std::vector<double> expectedCuts { 5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, uncuttable+mid+uncuttable") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 4, 4, 4, 5, 6, 6 };
   static const std::vector<double> expectedCuts { 4.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, left+uncuttable+mid+uncuttable") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 1, 4, 4, 5, 6, 6 };
   static const std::vector<double> expectedCuts { 4.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, uncuttable+mid+uncuttable+right") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 4, 4, 5, 6, 6, 9 };
   static const std::vector<double> expectedCuts { 5.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, uncuttable+cuttable") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 5, 5, 7, 8 };
   static const std::vector<double> expectedCuts { 6 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, left+uncuttable+cuttable") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 1, 5, 5, 5, 7, 8 };
   static const std::vector<double> expectedCuts { 6 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, cuttable+uncuttable") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 2, 3, 5, 5 };
   static const std::vector<double> expectedCuts { 4 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, cuttable+uncuttable+right") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 1, 2, 3, 5, 5, 7 };
   static const std::vector<double> expectedCuts { 4 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, cuttable+uncuttable+cuttable") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 2, 3, 5, 5, 7, 8 };
   static const std::vector<double> expectedCuts { 4, 6 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, uncuttable+cuttable+uncuttable") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 2, 2, 4, 6, 8, 8 };
   static const std::vector<double> expectedCuts { 3, 7 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, left+uncuttable+cuttable+uncuttable") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 1, 2, 2, 4, 5, 6, 8, 8 };
   static const std::vector<double> expectedCuts { 3, 7 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, uncuttable+cuttable+uncuttable+right") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 2, 2, 2, 4, 6, 8, 8, 9 };
   static const std::vector<double> expectedCuts { 3, 7 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, left+uncuttable+cuttable+uncuttable+right") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 1, 2, 2, 4, 6, 8, 8, 9 };
   static const std::vector<double> expectedCuts { 3, 7 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, uncuttable+cuttable+uncuttable+cuttable") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 1, 1, 2, 3, 5, 5, 7, 8 };
   static const std::vector<double> expectedCuts { 1.5, 4, 6 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, left+uncuttable+cuttable+uncuttable+cuttable") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 0, 1, 1, 2, 3, 5, 5, 5, 7, 8 };
   static const std::vector<double> expectedCuts { 1.5, 4, 6 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, cuttable+uncuttable+cuttable+uncuttable") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 2, 3, 5, 5, 7, 8, 9, 9 };
   static const std::vector<double> expectedCuts { 4, 6, 8.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, cuttable+uncuttable+cuttable+uncuttable+right") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 2, 3, 5, 5, 5, 7, 8, 9, 9, 10 };
   static const std::vector<double> expectedCuts { 4, 6, 8.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, left+uncuttable+cuttable+uncuttable+cuttable+uncuttable+cuttable+uncuttable+right") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 1, 2, 2, 3, 4, 4, 4, 5, 6, 6, 7, 8, 8, 9 };
   static const std::vector<double> expectedCuts { 3.5, 4.5, 7.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, infinities") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(),
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::signaling_NaN(),
   };
   static const std::vector<double> expectedCuts { 0 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, average segment sizes that requires the ceiling instead of rounding") {
   // our algorithm makes an internal assumption that we can give each cut point a location.  This is guaranteed if we 
   // make the average length of the equal value long ranges the ceiling of the average samples per bin.  
   // This test stresses that average calculation by having an average bin length of 2.2222222222 but if you use 
   // a bin width of 2, then there are 3 cut points that can't get any cuts.  3 cut points means that even if you 
   // don't give the first and last CuttingRanges an actual cut point, which can be reasonable since the 
   // first and last CuttingRanges are special in that they may have no long ranges on the tail ends, 
   // you still end up with one or more CuttingRanges that can't have a cut if you don't take the ceiling.

   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 26;
   static constexpr size_t cSamplesBinMin = 2;
   static const std::vector<double> featureVals { 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
      13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 
      27, 27, 28, 28, 29, 29, 30, 30 };
   static const std::vector<double> expectedCuts { 2.5, 4.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 
      14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5 };
   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, reversibility, 2") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { -1, 1 };
   static const std::vector<double> expectedCuts { 0 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, reversibility, 3") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { -2, 1, 2 };
   static const std::vector<double> expectedCuts { 0, 1.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, reversibility") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { -2, -1, 1, 2 };
   static const std::vector<double> expectedCuts { -1.5, 0, 1.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, imbalanced") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { -3, -1, 2, 3 };
   static const std::vector<double> expectedCuts { -2, 0, 2.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, extreme tails") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { std::numeric_limits<double>::lowest(), 1, 2, 3, std::numeric_limits<double>::max() };
   static const std::vector<double> expectedCuts { 0.5, 1.5, 2.5, 3.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, far tails") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { -1, 1, 2, 3, 5 };
   static const std::vector<double> expectedCuts { 0.5, 1.5, 2.5, 3.5 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, close tails") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { 0.9, 1, 2, 3, 3.1 };
   static const std::vector<double> expectedCuts { 0.95, 1.5, 2.5, 3.05 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, non-smart") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = false;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals { std::numeric_limits<double>::lowest(), 0, 1000, 10000000, std::numeric_limits<double>::max() };
   static const std::vector<double> expectedCuts { -8.9884656743115785e+307, 500, 5000500, 8.9884656743115785e+307 };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, overflow interpretable ends") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals {
      std::numeric_limits<double>::lowest(),
      FloatTickIncrementTest(std::numeric_limits<double>::lowest()),
      FloatTickDecrementTest(std::numeric_limits<double>::max()),
      std::numeric_limits<double>::max()
   };

   static const std::vector<double> expectedCuts {
      FloatTickIncrementTest(std::numeric_limits<double>::lowest()),
      0,
      std::numeric_limits<double>::max()
   };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, maximum non-overflow interpretable ends") {
   static constexpr bool bTestReverse = true;
   static constexpr bool bSmart = true;
   static constexpr size_t cCutsMax = 1000;
   static constexpr size_t cSamplesBinMin = 1;
   static const std::vector<double> featureVals {
      std::numeric_limits<double>::lowest(),
      FloatTickIncrementTest(std::numeric_limits<double>::lowest()),
      std::numeric_limits<double>::max() - FloatTickDecrementTest(std::numeric_limits<double>::max()),
      std::numeric_limits<double>::max()
   };

   static const std::vector<double> expectedCuts {
      FloatTickIncrementTest(std::numeric_limits<double>::lowest()),
      0,
      2.0000000000000001e+300
   };

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, stress test the guarantee of one cut per CuttingRange, by 2") {
   static constexpr size_t cItemsPerRange = 10;
   static constexpr size_t cInteriorRanges = 3;
   static constexpr size_t cRemoveCuts = 1;

   std::vector<double> featureVals(3 + cInteriorRanges * cItemsPerRange, 0);
   std::vector<double> expectedCuts;

   featureVals[0] = 0;
   for(size_t iRange = 0; iRange < cInteriorRanges; ++iRange) {
      for(size_t i = 1 + cItemsPerRange * iRange; i < 1 + (cItemsPerRange * (iRange + 1)); ++i) {
         const size_t iRangePlusOne = iRange + size_t { 1 };
         featureVals[i] = static_cast<double>(iRangePlusOne);
      }
      expectedCuts.push_back(double { 0.5 } + static_cast<double>(iRange));
   }
   expectedCuts.push_back(double { 0.5 } + static_cast<double>(cInteriorRanges));

   featureVals[cInteriorRanges * cItemsPerRange + 1] = static_cast<double>(1 + cInteriorRanges);
   featureVals[cInteriorRanges * cItemsPerRange + 2] = static_cast<double>(1 + cInteriorRanges);

   static bool bOne = 1 == cRemoveCuts;
   static bool bTwo = 2 == cRemoveCuts;
   if(bOne) {
      expectedCuts.erase(expectedCuts.begin());
   } else if(bTwo) {
      expectedCuts.erase(expectedCuts.begin());
      expectedCuts.erase(expectedCuts.end() - 1);
   } else {
      assert(0 == cRemoveCuts);
   }

   static constexpr bool bSmart = true;
   static constexpr bool bTestReverse = true;
   static constexpr size_t cCutsMax = cInteriorRanges + 1 - cRemoveCuts;
   static constexpr size_t cSamplesBinMin = 1;

   TestQuantileBinning(
      testCaseHidden,
      bTestReverse,
      bSmart,
      cCutsMax,
      cSamplesBinMin,
      featureVals,
      expectedCuts
   );
}

TEST_CASE("CutQuantile, randomized fairness check") {
   ErrorEbm error;

   RandomStreamTest randomStream(k_seed);
   if(!randomStream.IsSuccess()) {
      exit(1);
   }

   static constexpr bool bSmart = true;
   static constexpr IntEbm minSamplesBin = 1;
   static constexpr IntEbm countSamples = 100;
   double featureVals[countSamples]; // preserve these for debugging purposes
   double featureValsForward[countSamples];
   double featureValsReversed[countSamples];

   static constexpr IntEbm randomMaxMax = countSamples - 1; // this doesn't need to be exactly countSamples - 1, but this number gives us chunky sets
   size_t cutHistogram[randomMaxMax] = { 0 };
   static constexpr size_t cCutHistogram = sizeof(cutHistogram) / sizeof(cutHistogram[0]);
   // our random numbers can be any numbers from 0 to randomMaxMax (inclusive), which gives us randomMaxMax - 1 possible cut points between them
   static_assert(1 == cCutHistogram % 2, "cutHistogram must have a center value that is perfectly in the middle");

   static constexpr size_t cCuts = 9;
   double cutsLowerBoundInclusiveForward[cCuts];
   double cutsLowerBoundInclusiveReversed[cCuts];

   for(int iIteration = 0; iIteration < 100; ++iIteration) {
      for(size_t randomMax = 1; randomMax <= randomMaxMax; randomMax += 2) {
         // since randomMax isn't larger than the number of samples, we'll always be chunky.  This is good for testing range collisions
         for(size_t iSample = 0; iSample < countSamples; ++iSample) {
            bool bMissing = 0 == randomStream.Next(countSamples); // some datasetes will have zero missing values, some will have 1 or more
            size_t iRandom = randomStream.Next(randomMax + 1) + 1;
            featureVals[iSample] = bMissing ? std::numeric_limits<double>::quiet_NaN() : static_cast<double>(iRandom);
         }

         size_t countMissingValsExpected = 0;
         const double * pFeatureVal = featureVals;
         while(pFeatureVal != featureVals + countSamples) {
            if(std::isnan(*pFeatureVal)) {
               ++countMissingValsExpected;
            }
            ++pFeatureVal;
         }

         memcpy(featureValsForward, featureVals, sizeof(featureVals[0]) * countSamples);

         IntEbm countCutsForward = static_cast<IntEbm>(cCuts);
         error = CutQuantile(
            countSamples,
            featureValsForward,
            minSamplesBin,
            bSmart ? EBM_TRUE : EBM_FALSE,
            &countCutsForward,
            cutsLowerBoundInclusiveForward
         );
         CHECK(Error_None == error);

         //DisplayCuts(
         //   countSamples,
         //   featureVals,
         //   cCuts + 1,
         //   minSamplesBin,
         //   countCutsForward,
         //   cutsLowerBoundInclusiveForward,
         //   0 != countMissingVals,
         //   minNonInfinityVal,
         //   maxNonInfinityVal
         //);

         std::transform(featureVals, featureVals + countSamples, featureValsReversed,
            [](double & val) { return -val; });

         IntEbm countCutsReversed = static_cast<IntEbm>(cCuts);
         error = CutQuantile(
            countSamples,
            featureValsReversed,
            minSamplesBin,
            bSmart ? EBM_TRUE : EBM_FALSE,
            &countCutsReversed,
            cutsLowerBoundInclusiveReversed
         );
         CHECK(Error_None == error);

         CHECK(countCutsForward == countCutsReversed);

         std::sort(featureVals, featureVals + countSamples, CompareFloatWithNan());

         assert(1 == randomMax % 2); // our random numbers need a center value as well
         static constexpr size_t iHistogramExactMiddle = cCutHistogram / 2;
         const size_t iCutExactMiddle = randomMax / 2;
         assert(iCutExactMiddle <= iHistogramExactMiddle);
         const size_t iShiftToMiddle = iHistogramExactMiddle - iCutExactMiddle;
         const size_t cCutsReturned = static_cast<size_t>(countCutsForward);
         for(size_t iCutPoint = 0; iCutPoint < cCutsReturned; ++iCutPoint) {
            const double cutPointForward = cutsLowerBoundInclusiveForward[iCutPoint];
            if(countCutsForward == countCutsReversed) {
               const double cutPointReversed = -cutsLowerBoundInclusiveReversed[cCutsReturned - 1 - iCutPoint];

               const double cutPointForwardNext = *std::upper_bound(featureVals, featureVals + countSamples - 1 - countMissingValsExpected, cutPointForward);
               const double cutPointReversedNext = *std::upper_bound(featureVals, featureVals + countSamples - 1 - countMissingValsExpected, cutPointReversed);

               CHECK_APPROX(cutPointForwardNext, cutPointReversedNext);
            }
            // cutPoint can be a number between 0.5 and (randomMax - 0.5)
            const size_t iCut = static_cast<size_t>(std::round(cutPointForward - double { 1.5 }));
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
   const double ratio = static_cast<double>(cMin) / static_cast<double>(cMax);
   CHECK(0.97 <= ratio || 0 == cMax);
}

TEST_CASE("CutQuantile, chunky randomized check") {
   ErrorEbm error;

   RandomStreamTest randomStream(k_seed);
   if(!randomStream.IsSuccess()) {
      exit(1);
   }

   static constexpr bool bSmart = true;
   static constexpr size_t cSamplesMin = 1;
   static constexpr size_t cSamplesMax = 250;
   static constexpr size_t cCutsMin = 1;
   static constexpr size_t cCutsMax = 70;
   static constexpr IntEbm minSamplesBinMin = 1;
   static constexpr IntEbm minSamplesBinMax = 3;

   static constexpr size_t randomValMax = 70; // the min is 1 since the value doesn't really matter

   double cutsLowerBoundInclusiveForward[cCutsMax];
   double cutsLowerBoundInclusiveReversed[cCutsMax];

   double featureValsForward[cSamplesMax];
   double featureValsReversed[cSamplesMax];

   for(size_t iIteration = 0; iIteration < 30000; ++iIteration) {
      double featureVals[cSamplesMax] = { 0 }; // preserve these for debugging purposes

      const size_t cSamples = randomStream.Next(cSamplesMax - cSamplesMin + 1) + cSamplesMin;
      const size_t cCuts = randomStream.Next(cCutsMax - cCutsMin + 1) + cCutsMin;
      const IntEbm minSamplesBin = randomStream.Next(minSamplesBinMax - minSamplesBinMin + 1) + minSamplesBinMin;

      const size_t denominator = cCuts + size_t { 1 };
      const size_t cLongBinLength = static_cast<size_t>(
         std::ceil(static_cast<double>(cSamples) / static_cast<double>(denominator)));

      size_t i = 0;
      size_t cLongRanges = randomStream.Next(6);
      for(size_t iLongRange = 0; iLongRange < cLongRanges; ++iLongRange) {
         size_t cItems = randomStream.Next(cLongBinLength) + cLongBinLength;
         size_t val = randomStream.Next(randomValMax) + 1;
         for(size_t iItem = 0; iItem < cItems; ++iItem) {
            featureVals[i % cSamples] = static_cast<double>(val);
            ++i;
         }
      }
      size_t cShortRanges = randomStream.Next(6);
      for(size_t iShortRange = 0; iShortRange < cShortRanges; ++iShortRange) {
         size_t cItems = randomStream.Next(cLongBinLength);
         size_t val = randomStream.Next(randomValMax) + 1;
         for(size_t iItem = 0; iItem < cItems; ++iItem) {
            featureVals[i % cSamples] = static_cast<double>(val);
            ++i;
         }
      }
      for(size_t iSample = 0; iSample < cSamples; ++iSample) {
         if(0 == featureVals[iSample]) {
            const size_t randomPlusOne = randomStream.Next(randomValMax) + size_t { 1 };
            featureVals[iSample] = static_cast<double>(randomPlusOne);
         }
      }

      size_t countMissingValsExpected = 0;
      const double * pFeatureVal = featureVals;
      while(pFeatureVal != featureVals + cSamples) {
         if(std::isnan(*pFeatureVal)) {
            ++countMissingValsExpected;
         }
         ++pFeatureVal;
      }

      const IntEbm countSamples = static_cast<IntEbm>(cSamples);

      memcpy(featureValsForward, featureVals, sizeof(featureVals[0]) * cSamples);

      IntEbm countCutsForward = static_cast<IntEbm>(cCuts);
      error = CutQuantile(
         countSamples,
         featureValsForward,
         minSamplesBin,
         bSmart ? EBM_TRUE : EBM_FALSE,
         &countCutsForward,
         cutsLowerBoundInclusiveForward
      );
      CHECK(Error_None == error);

      std::transform(featureVals, featureVals + countSamples, featureValsReversed,
         [](double & val) { return -val; });

      IntEbm countCutsReversed = static_cast<IntEbm>(cCuts);
      error = CutQuantile(
         countSamples,
         featureValsReversed,
         minSamplesBin,
         bSmart ? EBM_TRUE : EBM_FALSE,
         &countCutsReversed,
         cutsLowerBoundInclusiveReversed
      );
      CHECK(Error_None == error);

      CHECK(countCutsForward == countCutsReversed);
      if(countCutsForward == countCutsReversed) {
         std::sort(featureVals, featureVals + countSamples, CompareFloatWithNan());

         const size_t cCutsReturned = static_cast<size_t>(countCutsForward);
         for(size_t iCutPoint = 0; iCutPoint < cCutsReturned; ++iCutPoint) {
            const double cutPointForward = cutsLowerBoundInclusiveForward[iCutPoint];
            const double cutPointReversed = -cutsLowerBoundInclusiveReversed[cCutsReturned - 1 - iCutPoint];

            const double cutPointForwardNext = *std::upper_bound(featureVals, featureVals + countSamples - 1 - countMissingValsExpected, cutPointForward);
            const double cutPointReversedNext = *std::upper_bound(featureVals, featureVals + countSamples - 1 - countMissingValsExpected, cutPointReversed);

            CHECK_APPROX(cutPointForwardNext, cutPointReversedNext);
         }
      }
   }
}

