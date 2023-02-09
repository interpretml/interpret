// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include <random>

#include "ebm_native.h"
#include "ebm_native_test.hpp"

static constexpr TestPriority k_filePriority = TestPriority::CutUniform;

static constexpr double k_minNonSubnormal = 2.2250738585072014e-308; // std::numeric_limits<double>::min()
static constexpr double k_maxNonInf = 1.7976931348623158e+308; // std::numeric_limits<double>::max()
static constexpr double k_subnormToNorm = 4503599627370496.0; // multiplying by this will move a subnormal into a normal

static constexpr double illegalVal = double { -888.88 };
static double * const pIllegal = reinterpret_cast<double *>(1);

TEST_CASE("CutUniform, zero cuts") {
   IntEbm countCuts = CutUniform(100, pIllegal, 0, pIllegal);
   CHECK(0 == countCuts);
}

TEST_CASE("CutUniform, 0 samples") {
   IntEbm countCuts = CutUniform(0, pIllegal, 10, pIllegal);
   CHECK(countCuts == 0);
}

TEST_CASE("CutUniform, 1 sample") {
   IntEbm countCuts = CutUniform(0, pIllegal, 10, pIllegal);
   CHECK(countCuts == 0);
}

TEST_CASE("CutUniform, only missing") {
   std::vector<double> featureVals { 
      std::numeric_limits<double>::quiet_NaN(), 
      std::numeric_limits<double>::quiet_NaN(),
   };
   IntEbm countCuts = CutUniform(featureVals.size(), &featureVals[0], 3, pIllegal);
   CHECK(0 == countCuts);
}

TEST_CASE("CutUniform, only -inf") {
   std::vector<double> featureVals {
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
   };
   IntEbm countCuts = CutUniform(featureVals.size(), &featureVals[0], 3, pIllegal);
   CHECK(0 == countCuts);
}

TEST_CASE("CutUniform, only +inf") {
   std::vector<double> featureVals {
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
   };
   IntEbm countCuts = CutUniform(featureVals.size(), &featureVals[0], 3, pIllegal);
   CHECK(0 == countCuts);
}

TEST_CASE("CutUniform, identical values + missing") {
   std::vector<double> featureVals { 1, 1, std::numeric_limits<double>::quiet_NaN(), 1 };
   IntEbm countCuts = CutUniform(featureVals.size(), &featureVals[0], 10, pIllegal);
   CHECK(0 == countCuts);
}

TEST_CASE("CutUniform, exactly sufficient floting point range for all cuts") {
   std::vector<double> featureVals;
   std::vector<double> expectedCuts;

   double val = 10.0;
   for(int i = 0; i < 1000; ++i) {
      featureVals.push_back(val);
      val = FloatTickIncrementTest(val);
      expectedCuts.push_back(val);
   }
   expectedCuts.pop_back();

   std::vector<double> cuts(featureVals.size() - 1, illegalVal);

   IntEbm countCuts = CutUniform(featureVals.size(), &featureVals[0], cuts.size(), &cuts[0]);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cuts[i]);
      }
   }
}

TEST_CASE("CutUniform, exactly sufficient floting point range for all cuts, cross power 2 high boundary upwards") {
   std::vector<double> featureVals;
   std::vector<double> expectedCuts;

   double val = 8.0;
   for(int i = 0; i < 500; ++i) {
      val = FloatTickDecrementTest(val);
   }
   for(int i = 0; i < 1000; ++i) {
      featureVals.push_back(val);
      val = FloatTickIncrementTest(val);
      expectedCuts.push_back(val);
   }
   expectedCuts.pop_back();

   std::vector<double> cuts(featureVals.size() - 1, illegalVal);

   IntEbm countCuts = CutUniform(featureVals.size(), &featureVals[0], cuts.size(), &cuts[0]);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cuts[i]);
      }
   }
}

TEST_CASE("CutUniform, marginally sufficient floting point range for all cuts, cross power 2 high boundary upwards") {
   std::vector<double> featureVals;

   double val = 8.0;
   for(int i = 0; i < 500; ++i) {
      val = FloatTickDecrementTest(val);
   }
   for(int i = 0; i < 1000; ++i) {
      featureVals.push_back(val);
      val = FloatTickIncrementTest(val);
   }

   std::vector<double> cuts(featureVals.size() - 2, illegalVal);

   IntEbm countCuts = CutUniform(featureVals.size(), &featureVals[0], cuts.size(), &cuts[0]);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(cuts.size() == cCuts);
}

TEST_CASE("CutUniform, insufficient floting point range for all cuts") {
   std::vector<double> featureVals;
   std::vector<double> expectedCuts;

   double val = 10.0;
   for(int i = 0; i < 1000; ++i) {
      featureVals.push_back(val);
      val = FloatTickIncrementTest(val);
      expectedCuts.push_back(val);
   }
   expectedCuts.pop_back();

   std::vector<double> cuts(featureVals.size(), illegalVal);

   IntEbm countCuts = CutUniform(featureVals.size(), &featureVals[0], cuts.size(), &cuts[0]);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cuts[i]);
      }
   }
}

TEST_CASE("CutUniform, one cut, -infinity and +infinity") {
   std::vector<double> featureVals {
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
   };
   std::vector<double> cuts(1, illegalVal);

   IntEbm countCuts = CutUniform(featureVals.size(), &featureVals[0], cuts.size(), &cuts[0]);

   CHECK(1 == countCuts);
   CHECK(0 == cuts[0]);
}

TEST_CASE("CutUniform, mid-point overflow if not special cased") {
   std::vector<double> featureVals {
      std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::max(),
   };
   std::vector<double> cuts(13, illegalVal);

   IntEbm countCuts = CutUniform(featureVals.size(), &featureVals[0], cuts.size(), &cuts[0]);

   CHECK(13 == countCuts);
}

TEST_CASE("CutUniform, infinite diff, even cuts") {
   IntEbm countCuts = 2;

   std::vector<double> featureVals {
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::lowest(),
   };
   static const std::vector<double> expectedCuts { -5.9923104495410517e+307, 5.9923104495410517e+307 };

   std::vector<double> cuts(static_cast<size_t>(countCuts), illegalVal);

   countCuts = CutUniform(featureVals.size(), &featureVals[0], cuts.size(), &cuts[0]);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cuts[i]);
      }
   }
}

TEST_CASE("CutUniform, infinite diff, odd cuts") {
   IntEbm countCuts = 3;

   std::vector<double> featureVals {
      std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::max(),
   };
   static const std::vector<double> expectedCuts { -8.9884656743115785e+307, 0, 8.9884656743115785e+307 };

   std::vector<double> cuts(static_cast<size_t>(countCuts), illegalVal);

   countCuts = CutUniform(featureVals.size(), &featureVals[0], cuts.size(), &cuts[0]);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cuts[i]);
      }
   }
}

TEST_CASE("CutUniform, min and max at interior positions") {
   IntEbm countCuts = 9;

   std::vector<double> featureVals { 1, 2, 3, 4, 5, 0, 10, 6, 7, 8, 9 };
   static const std::vector<double> expectedCuts { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

   std::vector<double> cuts(static_cast<size_t>(countCuts), illegalVal);

   countCuts = CutUniform(featureVals.size(), &featureVals[0], countCuts, &cuts[0]);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cuts[i]);
      }
   }
}

TEST_CASE("CutUniform, low start, hit float resolution before end") {
   std::vector<double> featureVals;

   double val = -std::numeric_limits<double>::min();
   for(int i = 0; i < 5; ++i) {
      // backup a few ticks
      val = FloatTickDecrementTest(val);
   }
   for(size_t iPast = 0; iPast < 10; ) {
      if(std::numeric_limits<double>::min() <= val) {
         ++iPast;
      }
      featureVals.push_back(val);
      val = FloatTickIncrementTest(val);
   }

   // have just 1 hole in the middle
   std::vector<double> cuts(featureVals.size() - 2, illegalVal);

   IntEbm countCuts = CutUniform(featureVals.size(), &featureVals[0], cuts.size(), &cuts[0]);

   CHECK(cuts.size() == static_cast<size_t>(countCuts));
}

TEST_CASE("CutUniform, high start, hit float resolution before end") {
   std::vector<double> featureVals;

   double val = -std::numeric_limits<double>::min();
   for(int i = 0; i < 10; ++i) {
      // backup a few ticks
      val = FloatTickDecrementTest(val);
   }
   for(size_t iPast = 0; iPast < 5; ) {
      if(std::numeric_limits<double>::min() <= val) {
         ++iPast;
      }
      featureVals.push_back(val);
      val = FloatTickIncrementTest(val);
   }

   // have just 1 hole in the middle
   std::vector<double> cuts(featureVals.size() - 2, illegalVal);

   IntEbm countCuts = CutUniform(featureVals.size(), &featureVals[0], cuts.size(), &cuts[0]);

   CHECK(cuts.size() == static_cast<size_t>(countCuts));
}

TEST_CASE("CutUniform, stress test reproducible") {
   size_t iTest;
   IntEbm iCut;
   IntEbm countCuts;

   double lowTest;
   double highTest;
   double oneCut;

   IntEbm testVal;
   IntEbm checkVal;
   IntEbm iTick;

   double result = 0.0;
   double seed = 64906263;

   double featureVals[2] = { 0, 0 };

   static constexpr size_t cInteresting = 19;
   static constexpr double interestingVals[cInteresting] = {
      -k_maxNonInf,
      -3.0,
      -2.0,
      -1.5,
      -1.0,
      -0.5,
      -2 * k_subnormToNorm * k_minNonSubnormal,
      -k_subnormToNorm * k_minNonSubnormal,
      -k_minNonSubnormal,
      0.0,
      k_minNonSubnormal,
      k_subnormToNorm * k_minNonSubnormal,
      2 * k_subnormToNorm * k_minNonSubnormal,
      0.5,
      1.0,
      1.5,
      2.0,
      3.0,
      k_maxNonInf
   };
   static constexpr size_t cCutsMax = 31; // 31 is prime
   double cuts[cCutsMax] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

   // this is a really crappy Middle-square random number generator so that we can replicate it in any language
   for(iTest = 0; iTest < 20000; ++iTest) {
      seed = floor((seed * seed) / 11.0);
      seed = seed - floor(seed / 94906263) * 94906263; // floor(sqrt(SAFE_FLOAT64_AS_INT64_MAX))
      testVal = static_cast<IntEbm>(seed);

      checkVal = testVal % cInteresting;
      lowTest = interestingVals[checkVal];
      testVal = testVal / cInteresting;

      checkVal = testVal % cInteresting;
      highTest = interestingVals[checkVal];
      testVal = testVal / cInteresting;

      checkVal = testVal % 2;
      if(0 == checkVal) {
         testVal = testVal / 2;
         checkVal = testVal % 16;
         for(iTick = 0; iTick < checkVal; ++iTick) {
            if(lowTest != -k_maxNonInf) {
               lowTest = FloatTickDecrementTest(lowTest);
            }
         }
      } else {
         testVal = testVal / 2;
         checkVal = testVal % 16;
         for(iTick = 0; iTick < checkVal; ++iTick) {
            if(lowTest != k_maxNonInf) {
               lowTest = FloatTickIncrementTest(lowTest);
            }
         }
      }
      testVal = testVal / 16;

      checkVal = testVal % 2;
      if(0 == checkVal) {
         testVal = testVal / 2;
         checkVal = testVal % 16;
         for(iTick = 0; iTick < checkVal; ++iTick) {
            if(highTest != -k_maxNonInf) {
               highTest = FloatTickDecrementTest(highTest);
            }
         }
      } else {
         testVal = testVal / 2;
         checkVal = testVal % 16;
         for(iTick = 0; iTick < checkVal; ++iTick) {
            if(highTest != k_maxNonInf) {
               highTest = FloatTickIncrementTest(highTest);
            }
         }
      }
      testVal = testVal / 16;

      featureVals[0] = lowTest;
      featureVals[1] = highTest;

      checkVal = testVal % cCutsMax;
      countCuts = CutUniform(2, featureVals, checkVal, cuts);

      for(iCut = 0; iCut < countCuts; ++iCut) {
         oneCut = cuts[iCut];
         while(-1.0 < oneCut && oneCut < 1.0 && 0.0 != oneCut) {
            oneCut *= 2.0;
         }
         while(oneCut < -1.0 || 1.0 < oneCut) {
            oneCut *= 0.5;
         }
         if(result < 0.0 && 0.0 < oneCut || 0.0 < result && oneCut < 0.0) {
            result += oneCut;
         } else {
            result -= oneCut;
         }
      }
   }

   CHECK(0.083452131729086054 == result);
}

