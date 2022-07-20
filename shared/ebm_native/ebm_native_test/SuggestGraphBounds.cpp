// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"
#include "RandomStreamTest.hpp"

static const TestPriority k_filePriority = TestPriority::SuggestGraphBounds;

// TODO : re-formulate these tests after we reach agreement on how the graph bounds are suposed to work

TEST_CASE("SuggestGraphBounds, 0 cuts, min -inf, max nan") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 0;
   constexpr double minFeatureVal = -std::numeric_limits<double>::infinity();
   constexpr double lowestCut = -1;
   constexpr double highestCut = -1;
   constexpr double maxFeatureVal = std::numeric_limits<double>::quiet_NaN();

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(-std::numeric_limits<double>::infinity() == lowGraphBound);
   CHECK(-std::numeric_limits<double>::infinity() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 0 cuts, min nan, max inf") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 0;
   constexpr double minFeatureVal = std::numeric_limits<double>::quiet_NaN();
   constexpr double lowestCut = -1;
   constexpr double highestCut = -1;
   constexpr double maxFeatureVal = std::numeric_limits<double>::infinity();

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(std::numeric_limits<double>::infinity() == lowGraphBound);
   CHECK(std::numeric_limits<double>::infinity() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 0 cuts, min 7, max 99") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 0;
   constexpr double minFeatureVal = 7;
   constexpr double lowestCut = -1;
   constexpr double highestCut = -1;
   constexpr double maxFeatureVal = 99;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(7 == lowGraphBound);
   CHECK(99 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, all 6") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr double minFeatureVal = 6;
   constexpr double lowestCut = 6;
   constexpr double highestCut = 6;
   constexpr double maxFeatureVal = 6;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(6 == lowGraphBound);
   CHECK(6 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, progression") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr double minFeatureVal = 6;
   constexpr double lowestCut = 7;
   constexpr double highestCut = 7;
   constexpr double maxFeatureVal = 8;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(6 == lowGraphBound);
   CHECK(8 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, mismatched low high") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr double minFeatureVal = -1;
   constexpr double lowestCut = -2;
   constexpr double highestCut = -2;
   constexpr double maxFeatureVal = 1;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(lowGraphBound < -2);
   CHECK(1 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, mismatched low high") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr double minFeatureVal = -2;
   constexpr double lowestCut = 0;
   constexpr double highestCut = 0;
   constexpr double maxFeatureVal = -1;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(-2 == lowGraphBound);
   CHECK(0 < highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, out of range high") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr double minFeatureVal = 1;
   constexpr double lowestCut = 0;
   constexpr double highestCut = 0;
   constexpr double maxFeatureVal = 2;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(lowGraphBound < 0);
   CHECK(2 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, min -inf") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr double minFeatureVal = -std::numeric_limits<double>::infinity();
   constexpr double lowestCut = std::numeric_limits<double>::lowest() + double { 1e300 };
   constexpr double highestCut = std::numeric_limits<double>::lowest() + double { 1e300 };
   constexpr double maxFeatureVal = std::numeric_limits<double>::lowest() + double { 1.5e300 };

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(minFeatureVal == lowGraphBound);
   CHECK(maxFeatureVal == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, max +inf") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr double minFeatureVal = std::numeric_limits<double>::max() - 1.5e300;
   constexpr double lowestCut = std::numeric_limits<double>::max() - 1e300;
   constexpr double highestCut = std::numeric_limits<double>::max() - 1e300;
   constexpr double maxFeatureVal = std::numeric_limits<double>::infinity();

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(minFeatureVal == lowGraphBound);
   CHECK(maxFeatureVal == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, overflow diff") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr double minFeatureVal = 0;
   constexpr double lowestCut = std::numeric_limits<double>::lowest() + 1e300;
   constexpr double highestCut = lowestCut;
   constexpr double maxFeatureVal = std::numeric_limits<double>::max() - 1e300;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(-std::numeric_limits<double>::infinity() == lowGraphBound);
   CHECK(maxFeatureVal == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, min longest") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr double minFeatureVal = 98;
   constexpr double lowestCut = 100;
   constexpr double highestCut = 100;
   constexpr double maxFeatureVal = 101;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(98 == lowGraphBound);
   CHECK(101 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, max longest") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr double minFeatureVal = 99;
   constexpr double lowestCut = 100;
   constexpr double highestCut = 100;
   constexpr double maxFeatureVal = 102;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(99 == lowGraphBound);
   CHECK(102 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, overflow high") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr double minFeatureVal = std::numeric_limits<double>::max() - 1e307;
   constexpr double lowestCut = std::numeric_limits<double>::max() - 1e306;
   constexpr double highestCut = std::numeric_limits<double>::max() - 1e306;
   constexpr double maxFeatureVal = std::numeric_limits<double>::max() - 1e307;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(minFeatureVal == lowGraphBound);
   CHECK(std::numeric_limits<double>::infinity() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, overflow low") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr double minFeatureVal = std::numeric_limits<double>::lowest() + 1e307;
   constexpr double lowestCut = std::numeric_limits<double>::lowest() + 1e306;
   constexpr double highestCut = std::numeric_limits<double>::lowest() + 1e306;
   constexpr double maxFeatureVal = std::numeric_limits<double>::lowest() + 1e307;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(-std::numeric_limits<double>::infinity() == lowGraphBound);
   CHECK(maxFeatureVal == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 2 cuts") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 2;
   constexpr double minFeatureVal = 5;
   constexpr double lowestCut = 6;
   constexpr double highestCut = 7;
   constexpr double maxFeatureVal = 8;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(5 == lowGraphBound);
   CHECK(8 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 4 cuts") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 4;
   constexpr double minFeatureVal = 5;
   constexpr double lowestCut = 6;
   constexpr double highestCut = 7;
   constexpr double maxFeatureVal = 8;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK_APPROX(lowGraphBound, 5);
   CHECK_APPROX(highGraphBound, 8);
}

TEST_CASE("SuggestGraphBounds, 2 cuts, overflow diff") {
   double lowGraphBound;
   double highGraphBound;

   constexpr IntEbmType countCuts = 2;
   constexpr double minFeatureVal = -1;
   constexpr double lowestCut = std::numeric_limits<double>::lowest();
   constexpr double highestCut = std::numeric_limits<double>::max();
   constexpr double maxFeatureVal = 1;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minFeatureVal,
      maxFeatureVal,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(-std::numeric_limits<double>::infinity() == lowGraphBound);
   CHECK(std::numeric_limits<double>::infinity() == highGraphBound);
}

