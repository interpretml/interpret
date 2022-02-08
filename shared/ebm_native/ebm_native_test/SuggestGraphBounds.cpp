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
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 0;
   constexpr FloatEbmType minValue = -std::numeric_limits<FloatEbmType>::infinity();
   constexpr FloatEbmType lowestCut = -1;
   constexpr FloatEbmType highestCut = -1;
   constexpr FloatEbmType maxValue = std::numeric_limits<FloatEbmType>::quiet_NaN();

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(-std::numeric_limits<FloatEbmType>::infinity() == lowGraphBound);
   CHECK(-std::numeric_limits<FloatEbmType>::infinity() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 0 cuts, min nan, max inf") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 0;
   constexpr FloatEbmType minValue = std::numeric_limits<FloatEbmType>::quiet_NaN();
   constexpr FloatEbmType lowestCut = -1;
   constexpr FloatEbmType highestCut = -1;
   constexpr FloatEbmType maxValue = std::numeric_limits<FloatEbmType>::infinity();

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(std::numeric_limits<FloatEbmType>::infinity() == lowGraphBound);
   CHECK(std::numeric_limits<FloatEbmType>::infinity() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 0 cuts, min 7, max 99") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 0;
   constexpr FloatEbmType minValue = 7;
   constexpr FloatEbmType lowestCut = -1;
   constexpr FloatEbmType highestCut = -1;
   constexpr FloatEbmType maxValue = 99;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(7 == lowGraphBound);
   CHECK(99 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, all 6") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr FloatEbmType minValue = 6;
   constexpr FloatEbmType lowestCut = 6;
   constexpr FloatEbmType highestCut = 6;
   constexpr FloatEbmType maxValue = 6;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(6 == lowGraphBound);
   CHECK(6 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, progression") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr FloatEbmType minValue = 6;
   constexpr FloatEbmType lowestCut = 7;
   constexpr FloatEbmType highestCut = 7;
   constexpr FloatEbmType maxValue = 8;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(6 == lowGraphBound);
   CHECK(8 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, mismatched low high") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr FloatEbmType minValue = -1;
   constexpr FloatEbmType lowestCut = -2;
   constexpr FloatEbmType highestCut = -2;
   constexpr FloatEbmType maxValue = 1;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(lowGraphBound < -2);
   CHECK(1 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, mismatched low high") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr FloatEbmType minValue = -2;
   constexpr FloatEbmType lowestCut = 0;
   constexpr FloatEbmType highestCut = 0;
   constexpr FloatEbmType maxValue = -1;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(-2 == lowGraphBound);
   CHECK(0 < highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, out of range high") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr FloatEbmType minValue = 1;
   constexpr FloatEbmType lowestCut = 0;
   constexpr FloatEbmType highestCut = 0;
   constexpr FloatEbmType maxValue = 2;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(lowGraphBound < 0);
   CHECK(2 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, min -inf") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr FloatEbmType minValue = -std::numeric_limits<FloatEbmType>::infinity();
   constexpr FloatEbmType lowestCut = std::numeric_limits<FloatEbmType>::lowest() + FloatEbmType { 1e300 };
   constexpr FloatEbmType highestCut = std::numeric_limits<FloatEbmType>::lowest() + FloatEbmType { 1e300 };
   constexpr FloatEbmType maxValue = std::numeric_limits<FloatEbmType>::lowest() + FloatEbmType { 1.5e300 };

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(minValue == lowGraphBound);
   CHECK(maxValue == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, max +inf") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr FloatEbmType minValue = std::numeric_limits<FloatEbmType>::max() - 1.5e300;
   constexpr FloatEbmType lowestCut = std::numeric_limits<FloatEbmType>::max() - 1e300;
   constexpr FloatEbmType highestCut = std::numeric_limits<FloatEbmType>::max() - 1e300;
   constexpr FloatEbmType maxValue = std::numeric_limits<FloatEbmType>::infinity();

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(minValue == lowGraphBound);
   CHECK(maxValue == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, overflow diff") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr FloatEbmType minValue = 0;
   constexpr FloatEbmType lowestCut = std::numeric_limits<FloatEbmType>::lowest() + 1e300;
   constexpr FloatEbmType highestCut = lowestCut;
   constexpr FloatEbmType maxValue = std::numeric_limits<FloatEbmType>::max() - 1e300;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(-std::numeric_limits<FloatEbmType>::infinity() == lowGraphBound);
   CHECK(maxValue == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, min longest") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr FloatEbmType minValue = 98;
   constexpr FloatEbmType lowestCut = 100;
   constexpr FloatEbmType highestCut = 100;
   constexpr FloatEbmType maxValue = 101;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(98 == lowGraphBound);
   CHECK(101 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, max longest") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr FloatEbmType minValue = 99;
   constexpr FloatEbmType lowestCut = 100;
   constexpr FloatEbmType highestCut = 100;
   constexpr FloatEbmType maxValue = 102;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(99 == lowGraphBound);
   CHECK(102 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, overflow high") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr FloatEbmType minValue = std::numeric_limits<FloatEbmType>::max() - 1e307;
   constexpr FloatEbmType lowestCut = std::numeric_limits<FloatEbmType>::max() - 1e306;
   constexpr FloatEbmType highestCut = std::numeric_limits<FloatEbmType>::max() - 1e306;
   constexpr FloatEbmType maxValue = std::numeric_limits<FloatEbmType>::max() - 1e307;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(minValue == lowGraphBound);
   CHECK(std::numeric_limits<FloatEbmType>::infinity() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, overflow low") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 1;
   constexpr FloatEbmType minValue = std::numeric_limits<FloatEbmType>::lowest() + 1e307;
   constexpr FloatEbmType lowestCut = std::numeric_limits<FloatEbmType>::lowest() + 1e306;
   constexpr FloatEbmType highestCut = std::numeric_limits<FloatEbmType>::lowest() + 1e306;
   constexpr FloatEbmType maxValue = std::numeric_limits<FloatEbmType>::lowest() + 1e307;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(-std::numeric_limits<FloatEbmType>::infinity() == lowGraphBound);
   CHECK(maxValue == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 2 cuts") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 2;
   constexpr FloatEbmType minValue = 5;
   constexpr FloatEbmType lowestCut = 6;
   constexpr FloatEbmType highestCut = 7;
   constexpr FloatEbmType maxValue = 8;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(5 == lowGraphBound);
   CHECK(8 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 4 cuts") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 4;
   constexpr FloatEbmType minValue = 5;
   constexpr FloatEbmType lowestCut = 6;
   constexpr FloatEbmType highestCut = 7;
   constexpr FloatEbmType maxValue = 8;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK_APPROX(lowGraphBound, 5);
   CHECK_APPROX(highGraphBound, 8);
}

TEST_CASE("SuggestGraphBounds, 2 cuts, overflow diff") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countCuts = 2;
   constexpr FloatEbmType minValue = -1;
   constexpr FloatEbmType lowestCut = std::numeric_limits<FloatEbmType>::lowest();
   constexpr FloatEbmType highestCut = std::numeric_limits<FloatEbmType>::max();
   constexpr FloatEbmType maxValue = 1;

   SuggestGraphBounds(
      countCuts,
      lowestCut,
      highestCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(-std::numeric_limits<FloatEbmType>::infinity() == lowGraphBound);
   CHECK(std::numeric_limits<FloatEbmType>::infinity() == highGraphBound);
}

