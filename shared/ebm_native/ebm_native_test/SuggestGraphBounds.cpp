// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderEbmNativeTest.h"

#include "ebm_native.h"
#include "EbmNativeTest.h"
#include "RandomStreamTest.h"

static const TestPriority k_filePriority = TestPriority::SuggestGraphBounds;

TEST_CASE("SuggestGraphBounds, 0 cuts, min -inf, max nan") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 0;
   constexpr FloatEbmType minValue = -std::numeric_limits<FloatEbmType>::infinity();
   constexpr FloatEbmType lowestBinCut = -1;
   constexpr FloatEbmType highestBinCut = -1;
   constexpr FloatEbmType maxValue = std::numeric_limits<FloatEbmType>::quiet_NaN();

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(std::numeric_limits<FloatEbmType>::lowest() == lowGraphBound);
   CHECK(std::numeric_limits<FloatEbmType>::max() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 0 cuts, min nan, max inf") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 0;
   constexpr FloatEbmType minValue = std::numeric_limits<FloatEbmType>::quiet_NaN();
   constexpr FloatEbmType lowestBinCut = -1;
   constexpr FloatEbmType highestBinCut = -1;
   constexpr FloatEbmType maxValue = std::numeric_limits<FloatEbmType>::infinity();

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(std::numeric_limits<FloatEbmType>::lowest() == lowGraphBound);
   CHECK(std::numeric_limits<FloatEbmType>::max() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 0 cuts, min inf, max -inf") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 0;
   constexpr FloatEbmType minValue = std::numeric_limits<FloatEbmType>::infinity();
   constexpr FloatEbmType lowestBinCut = -1;
   constexpr FloatEbmType highestBinCut = -1;
   constexpr FloatEbmType maxValue = -std::numeric_limits<FloatEbmType>::infinity();

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(std::numeric_limits<FloatEbmType>::lowest() == lowGraphBound);
   CHECK(std::numeric_limits<FloatEbmType>::max() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, low nan, high 0") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 1;
   constexpr FloatEbmType minValue = -1;
   constexpr FloatEbmType lowestBinCut = std::numeric_limits<FloatEbmType>::quiet_NaN();
   constexpr FloatEbmType highestBinCut = 0;
   constexpr FloatEbmType maxValue = 1;

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(-1 == lowGraphBound);
   CHECK(1 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, low 0, high nan") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 1;
   constexpr FloatEbmType minValue = -1;
   constexpr FloatEbmType lowestBinCut = 0;
   constexpr FloatEbmType highestBinCut = std::numeric_limits<FloatEbmType>::quiet_NaN();
   constexpr FloatEbmType maxValue = 1;

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(-1 == lowGraphBound);
   CHECK(1 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, mismatched low high") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 1;
   constexpr FloatEbmType minValue = -1;
   constexpr FloatEbmType lowestBinCut = -0.5;
   constexpr FloatEbmType highestBinCut = 0.5;
   constexpr FloatEbmType maxValue = 1;

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(-1 == lowGraphBound);
   CHECK(1 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, out of range low") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 1;
   constexpr FloatEbmType minValue = -2;
   constexpr FloatEbmType lowestBinCut = 0;
   constexpr FloatEbmType highestBinCut = 0;
   constexpr FloatEbmType maxValue = -1;

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(std::numeric_limits<FloatEbmType>::lowest() == lowGraphBound);
   CHECK(std::numeric_limits<FloatEbmType>::max() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, out of range high") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 1;
   constexpr FloatEbmType minValue = 1;
   constexpr FloatEbmType lowestBinCut = 0;
   constexpr FloatEbmType highestBinCut = 0;
   constexpr FloatEbmType maxValue = 2;

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(std::numeric_limits<FloatEbmType>::lowest() == lowGraphBound);
   CHECK(std::numeric_limits<FloatEbmType>::max() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, min -inf") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 1;
   constexpr FloatEbmType minValue = -std::numeric_limits<FloatEbmType>::infinity();
   constexpr FloatEbmType lowestBinCut = std::numeric_limits<FloatEbmType>::lowest() + 1e300;
   constexpr FloatEbmType highestBinCut = std::numeric_limits<FloatEbmType>::lowest() + 1e300;
   constexpr FloatEbmType maxValue = std::numeric_limits<FloatEbmType>::lowest() + 1.5e300;

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(std::numeric_limits<FloatEbmType>::lowest() == lowGraphBound);
   CHECK(std::numeric_limits<FloatEbmType>::lowest() + 2e300 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, max +inf") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 1;
   constexpr FloatEbmType minValue = std::numeric_limits<FloatEbmType>::max() - 1.5e300;
   constexpr FloatEbmType lowestBinCut = std::numeric_limits<FloatEbmType>::max() - 1e300;
   constexpr FloatEbmType highestBinCut = std::numeric_limits<FloatEbmType>::max() - 1e300;
   constexpr FloatEbmType maxValue = std::numeric_limits<FloatEbmType>::infinity();

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(std::numeric_limits<FloatEbmType>::max() - 2e300 == lowGraphBound);
   CHECK(std::numeric_limits<FloatEbmType>::max() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, overflow diff") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 1;
   constexpr FloatEbmType minValue = std::numeric_limits<FloatEbmType>::lowest() + 1e300;
   constexpr FloatEbmType lowestBinCut = std::numeric_limits<FloatEbmType>::max() - 2e300;
   constexpr FloatEbmType highestBinCut = std::numeric_limits<FloatEbmType>::max() - 2e300;
   constexpr FloatEbmType maxValue = std::numeric_limits<FloatEbmType>::max() - 1e300;

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(std::numeric_limits<FloatEbmType>::lowest() == lowGraphBound);
   CHECK(std::numeric_limits<FloatEbmType>::max() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, min longest") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 1;
   constexpr FloatEbmType minValue = 98;
   constexpr FloatEbmType lowestBinCut = 100;
   constexpr FloatEbmType highestBinCut = 100;
   constexpr FloatEbmType maxValue = 101;

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(98 == lowGraphBound);
   CHECK(102 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, max longest") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 1;
   constexpr FloatEbmType minValue = 99;
   constexpr FloatEbmType lowestBinCut = 100;
   constexpr FloatEbmType highestBinCut = 100;
   constexpr FloatEbmType maxValue = 102;

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(98 == lowGraphBound);
   CHECK(102 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, overflow high") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 1;
   constexpr FloatEbmType minValue = std::numeric_limits<FloatEbmType>::max() - 1e307;
   constexpr FloatEbmType lowestBinCut = std::numeric_limits<FloatEbmType>::max() - 1e306;
   constexpr FloatEbmType highestBinCut = std::numeric_limits<FloatEbmType>::max() - 1e306;
   constexpr FloatEbmType maxValue = std::numeric_limits<FloatEbmType>::max();

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(std::numeric_limits<FloatEbmType>::max() - 1e307 == lowGraphBound);
   CHECK(std::numeric_limits<FloatEbmType>::max() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, overflow low") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 1;
   constexpr FloatEbmType minValue = std::numeric_limits<FloatEbmType>::lowest();
   constexpr FloatEbmType lowestBinCut = std::numeric_limits<FloatEbmType>::lowest() + 1e306;
   constexpr FloatEbmType highestBinCut = std::numeric_limits<FloatEbmType>::lowest() + 1e306;
   constexpr FloatEbmType maxValue = std::numeric_limits<FloatEbmType>::lowest() + 1e307;

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(std::numeric_limits<FloatEbmType>::lowest() == lowGraphBound);
   CHECK(std::numeric_limits<FloatEbmType>::lowest() + 1e307 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 2 cuts") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 2;
   constexpr FloatEbmType minValue = 5;
   constexpr FloatEbmType lowestBinCut = 6;
   constexpr FloatEbmType highestBinCut = 7;
   constexpr FloatEbmType maxValue = 8;

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(5.5 == lowGraphBound);
   CHECK(7.5 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 4 cuts") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 4;
   constexpr FloatEbmType minValue = 5;
   constexpr FloatEbmType lowestBinCut = 6;
   constexpr FloatEbmType highestBinCut = 7;
   constexpr FloatEbmType maxValue = 8;

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK_APPROX(lowGraphBound, 5.8);
   CHECK_APPROX(highGraphBound, 7.2);
}

TEST_CASE("SuggestGraphBounds, 2 cuts, overflow diff") {
   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;

   constexpr IntEbmType countBinCuts = 2;
   constexpr FloatEbmType minValue = std::numeric_limits<FloatEbmType>::lowest();
   constexpr FloatEbmType lowestBinCut = std::numeric_limits<FloatEbmType>::lowest();
   constexpr FloatEbmType highestBinCut = std::numeric_limits<FloatEbmType>::max();
   constexpr FloatEbmType maxValue = std::numeric_limits<FloatEbmType>::max();

   SuggestGraphBounds(
      countBinCuts,
      lowestBinCut,
      highestBinCut,
      minValue,
      maxValue,
      &lowGraphBound,
      &highGraphBound
   );

   CHECK(std::numeric_limits<FloatEbmType>::lowest() == lowGraphBound);
   CHECK(std::numeric_limits<FloatEbmType>::max() == highGraphBound);
}

