// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"

static const TestPriority k_filePriority = TestPriority::CutWinsorized;

constexpr FloatEbmType illegalVal = FloatEbmType { -888.88 };

TEST_CASE("CutWinsorized, 0 samples") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues {};
   const std::vector<FloatEbmType> expectedCuts {};

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, only missing") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues { std::numeric_limits<FloatEbmType>::quiet_NaN(), std::numeric_limits<FloatEbmType>::quiet_NaN() };
   const std::vector<FloatEbmType> expectedCuts {};

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one item") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues { 1 };
   const std::vector<FloatEbmType> expectedCuts {};

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, zero cuts") {
   IntEbmType countCuts = 0;

   std::vector<FloatEbmType> featureValues { 1, 2, 3, 4 };
   const std::vector<FloatEbmType> expectedCuts {};

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, identical values") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues { 1, 1, std::numeric_limits<FloatEbmType>::quiet_NaN(), 1 };
   const std::vector<FloatEbmType> expectedCuts { };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, even") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues { 1, 2, 3, 4 };
   const std::vector<FloatEbmType> expectedCuts { 2.5 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, odd") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues { 1, 2, 3.5, 4, 5 };
   const std::vector<FloatEbmType> expectedCuts { 3 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, even, two loops") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues { 1, 2, 2, 4 };
   const std::vector<FloatEbmType> expectedCuts { 2.5 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, odd, two loops") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues { 1, 4, 4, 4, 5 };
   const std::vector<FloatEbmType> expectedCuts { 3 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, even, two loops, exit up") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues { 2, 2, 2, 4 };
   const std::vector<FloatEbmType> expectedCuts { 3 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, odd, two loops, exit up") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues { 4, 4, 4, 4, 5 };
   const std::vector<FloatEbmType> expectedCuts { 4.5 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, even, two loops, exit down") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues { 1, 2, 2, 2 };
   const std::vector<FloatEbmType> expectedCuts { 1.5 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, odd, two loops, exit up") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues { 1, 4, 4, 4, 4 };
   const std::vector<FloatEbmType> expectedCuts { 2.5 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, -infinity") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues { -std::numeric_limits<FloatEbmType>::infinity(), -std::numeric_limits<FloatEbmType>::infinity() };
   const std::vector<FloatEbmType> expectedCuts {};

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, +infinity") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues { std::numeric_limits<FloatEbmType>::infinity(), std::numeric_limits<FloatEbmType>::infinity() };
   const std::vector<FloatEbmType> expectedCuts {};

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, -infinity and +infinity") {
   IntEbmType countCuts = 1;

   std::vector<FloatEbmType> featureValues { -std::numeric_limits<FloatEbmType>::infinity(), std::numeric_limits<FloatEbmType>::infinity() };
   const std::vector<FloatEbmType> expectedCuts { 0 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, outer test, cuts both sides") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues { 0, 1, 1, 1, 1, 1, 1, 7 };
   const std::vector<FloatEbmType> expectedCuts { 0.5, 4 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, outer test, cut bottom") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues { 0, 1, 1, 1, 1, 1, 1, 1 };
   const std::vector<FloatEbmType> expectedCuts { 0.5 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, outer test, cut top") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues { 1, 1, 1, 1, 1, 1, 1, 7 };
   const std::vector<FloatEbmType> expectedCuts { 4 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, outer test, no cuts") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues { 1, 1, 1, 1, 1, 1, 1, 1 };
   const std::vector<FloatEbmType> expectedCuts { };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, center, one transition") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues { 1, 1, 1, 1, 2, 2, 2, 2 };
   const std::vector<FloatEbmType> expectedCuts { 1.5 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, center, two transitions") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues { 1, 1, 1, 2, 2, 3, 3, 3 };
   const std::vector<FloatEbmType> expectedCuts { 1.5, 2.5 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, two cuts") {
   IntEbmType countCuts = 2;

   std::vector<FloatEbmType> featureValues { 0, 1, 2, 3, 4, 5 };
   const std::vector<FloatEbmType> expectedCuts { 2, std::nextafter(3, std::numeric_limits<FloatEbmType>::max()) };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, three cuts") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues { 0, 1, 2, 3, 4, 5, 6, 7 };
   const std::vector<FloatEbmType> expectedCuts {2, std::nextafter(3.5, std::numeric_limits<FloatEbmType>::max()), std::nextafter(5, std::numeric_limits<FloatEbmType>::max())};

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, four cuts") {
   IntEbmType countCuts = 4;

   std::vector<FloatEbmType> featureValues { 0, 1, 2, 3, 5, 7, 8, 9, 10 };
   const std::vector<FloatEbmType> expectedCuts { 2, 4, 6, std::nextafter(8, std::numeric_limits<FloatEbmType>::max()) };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, -infinity, lowest, max, and +infinity") {
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

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, -infinity, lowest + 1, max - 1, and +infinity") {
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

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, 3 cuts, -infinity, lowest, max, and +infinity") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues { 
      -std::numeric_limits<FloatEbmType>::infinity(), 
      std::numeric_limits<FloatEbmType>::lowest(), 
      std::numeric_limits<FloatEbmType>::max(),
      std::numeric_limits<FloatEbmType>::infinity()
   };
   const std::vector<FloatEbmType> expectedCuts { 0 };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, 3 cuts, -infinity, lowest + 1, max - 1, and +infinity") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues {
      -std::numeric_limits<FloatEbmType>::infinity(),
      std::nextafter(std::numeric_limits<FloatEbmType>::lowest(), 0),
      std::nextafter(std::numeric_limits<FloatEbmType>::max(), 0),
      std::numeric_limits<FloatEbmType>::infinity()
   };
   const std::vector<FloatEbmType> expectedCuts { 
      std::nextafter(std::numeric_limits<FloatEbmType>::lowest(), 0), 
      0,
      std::numeric_limits<FloatEbmType>::max()
   };

   std::vector<FloatEbmType> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   ErrorEbmType ret = CutWinsorized(
      featureValues.size(),
      0 == featureValues.size() ? nullptr : &featureValues[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == ret);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

