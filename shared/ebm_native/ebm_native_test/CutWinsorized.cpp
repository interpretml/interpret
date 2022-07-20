// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"

static const TestPriority k_filePriority = TestPriority::CutWinsorized;

constexpr double illegalVal = double { -888.88 };

TEST_CASE("CutWinsorized, 0 samples") {
   ErrorEbmType error;

   IntEbmType countCuts = 3;

   std::vector<double> featureVals {};
   const std::vector<double> expectedCuts {};

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, only missing") {
   ErrorEbmType error;

   IntEbmType countCuts = 3;

   std::vector<double> featureVals { std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN() };
   const std::vector<double> expectedCuts {};

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one item") {
   ErrorEbmType error;

   IntEbmType countCuts = 3;

   std::vector<double> featureVals { 1 };
   const std::vector<double> expectedCuts {};

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, zero cuts") {
   ErrorEbmType error;

   IntEbmType countCuts = 0;

   std::vector<double> featureVals { 1, 2, 3, 4 };
   const std::vector<double> expectedCuts {};

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, identical values") {
   ErrorEbmType error;

   IntEbmType countCuts = 1;

   std::vector<double> featureVals { 1, 1, std::numeric_limits<double>::quiet_NaN(), 1 };
   const std::vector<double> expectedCuts { };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, even") {
   ErrorEbmType error;

   IntEbmType countCuts = 1;

   std::vector<double> featureVals { 1, 2, 3, 4 };
   const std::vector<double> expectedCuts { 2.5 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, odd") {
   ErrorEbmType error;

   IntEbmType countCuts = 1;

   std::vector<double> featureVals { 1, 2, 3.5, 4, 5 };
   const std::vector<double> expectedCuts { 3 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, even, two loops") {
   ErrorEbmType error;

   IntEbmType countCuts = 1;

   std::vector<double> featureVals { 1, 2, 2, 4 };
   const std::vector<double> expectedCuts { 2.5 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, odd, two loops") {
   ErrorEbmType error;

   IntEbmType countCuts = 1;

   std::vector<double> featureVals { 1, 4, 4, 4, 5 };
   const std::vector<double> expectedCuts { 3 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, even, two loops, exit up") {
   ErrorEbmType error;

   IntEbmType countCuts = 1;

   std::vector<double> featureVals { 2, 2, 2, 4 };
   const std::vector<double> expectedCuts { 3 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, odd, two loops, exit up") {
   ErrorEbmType error;

   IntEbmType countCuts = 1;

   std::vector<double> featureVals { 4, 4, 4, 4, 5 };
   const std::vector<double> expectedCuts { 4.5 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, even, two loops, exit down") {
   ErrorEbmType error;

   IntEbmType countCuts = 1;

   std::vector<double> featureVals { 1, 2, 2, 2 };
   const std::vector<double> expectedCuts { 1.5 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, odd, two loops, exit up") {
   ErrorEbmType error;

   IntEbmType countCuts = 1;

   std::vector<double> featureVals { 1, 4, 4, 4, 4 };
   const std::vector<double> expectedCuts { 2.5 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, -infinity") {
   ErrorEbmType error;

   IntEbmType countCuts = 1;

   std::vector<double> featureVals { -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity() };
   const std::vector<double> expectedCuts {};

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, +infinity") {
   ErrorEbmType error;

   IntEbmType countCuts = 1;

   std::vector<double> featureVals { std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity() };
   const std::vector<double> expectedCuts {};

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, -infinity and +infinity") {
   ErrorEbmType error;

   IntEbmType countCuts = 1;

   std::vector<double> featureVals { -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity() };
   const std::vector<double> expectedCuts { 0 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, outer test, cuts both sides") {
   ErrorEbmType error;

   IntEbmType countCuts = 3;

   std::vector<double> featureVals { 0, 1, 1, 1, 1, 1, 1, 7 };
   const std::vector<double> expectedCuts { 0.5, 4 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, outer test, cut bottom") {
   ErrorEbmType error;

   IntEbmType countCuts = 3;

   std::vector<double> featureVals { 0, 1, 1, 1, 1, 1, 1, 1 };
   const std::vector<double> expectedCuts { 0.5 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, outer test, cut top") {
   ErrorEbmType error;

   IntEbmType countCuts = 3;

   std::vector<double> featureVals { 1, 1, 1, 1, 1, 1, 1, 7 };
   const std::vector<double> expectedCuts { 4 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, outer test, no cuts") {
   ErrorEbmType error;

   IntEbmType countCuts = 3;

   std::vector<double> featureVals { 1, 1, 1, 1, 1, 1, 1, 1 };
   const std::vector<double> expectedCuts { };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, center, one transition") {
   ErrorEbmType error;

   IntEbmType countCuts = 3;

   std::vector<double> featureVals { 1, 1, 1, 1, 2, 2, 2, 2 };
   const std::vector<double> expectedCuts { 1.5 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, center, two transitions") {
   ErrorEbmType error;

   IntEbmType countCuts = 3;

   std::vector<double> featureVals { 1, 1, 1, 2, 2, 3, 3, 3 };
   const std::vector<double> expectedCuts { 1.5, 2.5 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, two cuts") {
   ErrorEbmType error;

   IntEbmType countCuts = 2;

   std::vector<double> featureVals { 0, 1, 2, 3, 4, 5 };
   const std::vector<double> expectedCuts { 2, FloatTickIncrementTest(3) };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, three cuts") {
   ErrorEbmType error;

   IntEbmType countCuts = 3;

   std::vector<double> featureVals { 0, 1, 2, 3, 4, 5, 6, 7 };
   const std::vector<double> expectedCuts {2, FloatTickIncrementTest(3.5), FloatTickIncrementTest(5)};

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, four cuts") {
   ErrorEbmType error;

   IntEbmType countCuts = 4;

   std::vector<double> featureVals { 0, 1, 2, 3, 5, 7, 8, 9, 10 };
   const std::vector<double> expectedCuts { 2, 4, 6, FloatTickIncrementTest(8) };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, -infinity, lowest, max, and +infinity") {
   ErrorEbmType error;

   IntEbmType countCuts = 1;

   std::vector<double> featureVals {
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::infinity()
   };
   const std::vector<double> expectedCuts { 0 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, one cut, -infinity, lowest + 1, max - 1, and +infinity") {
   ErrorEbmType error;

   IntEbmType countCuts = 1;

   std::vector<double> featureVals {
      -std::numeric_limits<double>::infinity(),
      FloatTickIncrementTest(std::numeric_limits<double>::lowest()),
      FloatTickDecrementTest(std::numeric_limits<double>::max()),
      std::numeric_limits<double>::infinity()
   };
   const std::vector<double> expectedCuts { 0 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, 3 cuts, -infinity, lowest, max, and +infinity") {
   ErrorEbmType error;

   IntEbmType countCuts = 3;

   std::vector<double> featureVals { 
      -std::numeric_limits<double>::infinity(), 
      std::numeric_limits<double>::lowest(), 
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::infinity()
   };
   const std::vector<double> expectedCuts { 0 };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("CutWinsorized, 3 cuts, -infinity, lowest + 1, max - 1, and +infinity") {
   ErrorEbmType error;

   IntEbmType countCuts = 3;

   std::vector<double> featureVals {
      -std::numeric_limits<double>::infinity(),
      FloatTickIncrementTest(std::numeric_limits<double>::lowest()),
      FloatTickDecrementTest(std::numeric_limits<double>::max()),
      std::numeric_limits<double>::infinity()
   };
   const std::vector<double> expectedCuts { 
      FloatTickIncrementTest(std::numeric_limits<double>::lowest()),
      0,
      std::numeric_limits<double>::max()
   };

   std::vector<double> cutsLowerBoundInclusive(
      0 == countCuts ? size_t { 1 } : static_cast<size_t>(countCuts), illegalVal);

   error = CutWinsorized(
      featureVals.size(),
      0 == featureVals.size() ? nullptr : &featureVals[0],
      &countCuts,
      &cutsLowerBoundInclusive[0]
   );
   CHECK(Error_None == error);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cutsLowerBoundInclusive[i]);
      }
   }
}

