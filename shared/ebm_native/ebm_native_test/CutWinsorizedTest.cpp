// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"

static constexpr TestPriority k_filePriority = TestPriority::CutWinsorized;

static constexpr double illegalVal = double { -888.88 };

TEST_CASE("CutWinsorized, 0 samples") {
   ErrorEbm error;

   IntEbm countCuts = 3;

   std::vector<double> featureVals {};
   static const std::vector<double> expectedCuts {};

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
   ErrorEbm error;

   IntEbm countCuts = 3;

   std::vector<double> featureVals { std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN() };
   static const std::vector<double> expectedCuts {};

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
   ErrorEbm error;

   IntEbm countCuts = 3;

   std::vector<double> featureVals { 1 };
   static const std::vector<double> expectedCuts {};

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
   ErrorEbm error;

   IntEbm countCuts = 0;

   std::vector<double> featureVals { 1, 2, 3, 4 };
   static const std::vector<double> expectedCuts {};

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
   ErrorEbm error;

   IntEbm countCuts = 1;

   std::vector<double> featureVals { 1, 1, std::numeric_limits<double>::quiet_NaN(), 1 };
   static const std::vector<double> expectedCuts { };

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
   ErrorEbm error;

   IntEbm countCuts = 1;

   std::vector<double> featureVals { 1, 2, 3, 4 };
   static const std::vector<double> expectedCuts { 2.5 };

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
   ErrorEbm error;

   IntEbm countCuts = 1;

   std::vector<double> featureVals { 1, 2, 3.5, 4, 5 };
   static const std::vector<double> expectedCuts { 3 };

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
   ErrorEbm error;

   IntEbm countCuts = 1;

   std::vector<double> featureVals { 1, 2, 2, 4 };
   static const std::vector<double> expectedCuts { 2.5 };

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
   ErrorEbm error;

   IntEbm countCuts = 1;

   std::vector<double> featureVals { 1, 4, 4, 4, 5 };
   static const std::vector<double> expectedCuts { 3 };

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
   ErrorEbm error;

   IntEbm countCuts = 1;

   std::vector<double> featureVals { 2, 2, 2, 4 };
   static const std::vector<double> expectedCuts { 3 };

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
   ErrorEbm error;

   IntEbm countCuts = 1;

   std::vector<double> featureVals { 4, 4, 4, 4, 5 };
   static const std::vector<double> expectedCuts { 4.5 };

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
   ErrorEbm error;

   IntEbm countCuts = 1;

   std::vector<double> featureVals { 1, 2, 2, 2 };
   static const std::vector<double> expectedCuts { 1.5 };

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
   ErrorEbm error;

   IntEbm countCuts = 1;

   std::vector<double> featureVals { 1, 4, 4, 4, 4 };
   static const std::vector<double> expectedCuts { 2.5 };

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
   ErrorEbm error;

   IntEbm countCuts = 1;

   std::vector<double> featureVals { -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity() };
   static const std::vector<double> expectedCuts {};

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
   ErrorEbm error;

   IntEbm countCuts = 1;

   std::vector<double> featureVals { std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity() };
   static const std::vector<double> expectedCuts {};

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
   ErrorEbm error;

   IntEbm countCuts = 1;

   std::vector<double> featureVals { -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity() };
   static const std::vector<double> expectedCuts { 0 };

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
   ErrorEbm error;

   IntEbm countCuts = 3;

   std::vector<double> featureVals { 0, 1, 1, 1, 1, 1, 1, 7 };
   static const std::vector<double> expectedCuts { 0.5, 4 };

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
   ErrorEbm error;

   IntEbm countCuts = 3;

   std::vector<double> featureVals { 0, 1, 1, 1, 1, 1, 1, 1 };
   static const std::vector<double> expectedCuts { 0.5 };

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
   ErrorEbm error;

   IntEbm countCuts = 3;

   std::vector<double> featureVals { 1, 1, 1, 1, 1, 1, 1, 7 };
   static const std::vector<double> expectedCuts { 4 };

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
   ErrorEbm error;

   IntEbm countCuts = 3;

   std::vector<double> featureVals { 1, 1, 1, 1, 1, 1, 1, 1 };
   static const std::vector<double> expectedCuts { };

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
   ErrorEbm error;

   IntEbm countCuts = 3;

   std::vector<double> featureVals { 1, 1, 1, 1, 2, 2, 2, 2 };
   static const std::vector<double> expectedCuts { 1.5 };

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
   ErrorEbm error;

   IntEbm countCuts = 3;

   std::vector<double> featureVals { 1, 1, 1, 2, 2, 3, 3, 3 };
   static const std::vector<double> expectedCuts { 1.5, 2.5 };

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
   ErrorEbm error;

   IntEbm countCuts = 2;

   std::vector<double> featureVals { 0, 1, 2, 3, 4, 5 };
   static const std::vector<double> expectedCuts { 2, FloatTickIncrementTest(3) };

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
   ErrorEbm error;

   IntEbm countCuts = 3;

   std::vector<double> featureVals { 0, 1, 2, 3, 4, 5, 6, 7 };
   static const std::vector<double> expectedCuts {2, FloatTickIncrementTest(3.5), FloatTickIncrementTest(5)};

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
   ErrorEbm error;

   IntEbm countCuts = 4;

   std::vector<double> featureVals { 0, 1, 2, 3, 5, 7, 8, 9, 10 };
   static const std::vector<double> expectedCuts { 2, 4, 6, FloatTickIncrementTest(8) };

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
   ErrorEbm error;

   IntEbm countCuts = 1;

   std::vector<double> featureVals {
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::infinity()
   };
   static const std::vector<double> expectedCuts { 0 };

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
   ErrorEbm error;

   IntEbm countCuts = 1;

   std::vector<double> featureVals {
      -std::numeric_limits<double>::infinity(),
      FloatTickIncrementTest(std::numeric_limits<double>::lowest()),
      FloatTickDecrementTest(std::numeric_limits<double>::max()),
      std::numeric_limits<double>::infinity()
   };
   static const std::vector<double> expectedCuts { 0 };

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
   ErrorEbm error;

   IntEbm countCuts = 3;

   std::vector<double> featureVals { 
      -std::numeric_limits<double>::infinity(), 
      std::numeric_limits<double>::lowest(), 
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::infinity()
   };
   static const std::vector<double> expectedCuts { 0 };

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
   ErrorEbm error;

   IntEbm countCuts = 3;

   std::vector<double> featureVals {
      -std::numeric_limits<double>::infinity(),
      FloatTickIncrementTest(std::numeric_limits<double>::lowest()),
      FloatTickDecrementTest(std::numeric_limits<double>::max()),
      std::numeric_limits<double>::infinity()
   };
   static const std::vector<double> expectedCuts { 
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

