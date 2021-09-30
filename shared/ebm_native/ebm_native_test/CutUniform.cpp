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

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );

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

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );

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

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );

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

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );

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
   const std::vector<FloatEbmType> expectedCuts { 0 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );

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
   const std::vector<FloatEbmType> expectedCuts { 0 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );

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

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );

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

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );

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

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );

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

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );

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

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );

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

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );

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

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   CutUniform(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}
