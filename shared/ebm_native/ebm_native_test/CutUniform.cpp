// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include <random>

#include "ebm_native.h"
#include "ebm_native_test.hpp"

static const TestPriority k_filePriority = TestPriority::CutUniform;

constexpr FloatEbmType illegalVal = FloatEbmType { -888.88 };
static FloatEbmType * const pIllegal = reinterpret_cast<FloatEbmType *>(1);

TEST_CASE("CutUniform, zero cuts") {
   IntEbmType countCuts = CutUniform(100, pIllegal, 0, pIllegal);
   CHECK(0 == countCuts);
}

TEST_CASE("CutUniform, 0 samples") {
   IntEbmType countCuts = CutUniform(0, pIllegal, 10, pIllegal);
   CHECK(countCuts == 0);
}

TEST_CASE("CutUniform, 1 sample") {
   IntEbmType countCuts = CutUniform(0, pIllegal, 10, pIllegal);
   CHECK(countCuts == 0);
}

TEST_CASE("CutUniform, only missing") {
   std::vector<FloatEbmType> featureValues { 
      std::numeric_limits<FloatEbmType>::quiet_NaN(), 
      std::numeric_limits<FloatEbmType>::quiet_NaN(),
   };
   IntEbmType countCuts = CutUniform(featureValues.size(), &featureValues[0], 3, pIllegal);
   CHECK(0 == countCuts);
}

TEST_CASE("CutUniform, only -inf") {
   std::vector<FloatEbmType> featureValues {
      -std::numeric_limits<FloatEbmType>::infinity(),
      -std::numeric_limits<FloatEbmType>::infinity(),
   };
   IntEbmType countCuts = CutUniform(featureValues.size(), &featureValues[0], 3, pIllegal);
   CHECK(0 == countCuts);
}

TEST_CASE("CutUniform, only +inf") {
   std::vector<FloatEbmType> featureValues {
      std::numeric_limits<FloatEbmType>::infinity(),
      std::numeric_limits<FloatEbmType>::infinity(),
   };
   IntEbmType countCuts = CutUniform(featureValues.size(), &featureValues[0], 3, pIllegal);
   CHECK(0 == countCuts);
}

TEST_CASE("CutUniform, identical values + missing") {
   std::vector<FloatEbmType> featureValues { 1, 1, std::numeric_limits<FloatEbmType>::quiet_NaN(), 1 };
   IntEbmType countCuts = CutUniform(featureValues.size(), &featureValues[0], 10, pIllegal);
   CHECK(0 == countCuts);
}

TEST_CASE("CutUniform, exactly sufficient floting point range for all cuts") {
   std::vector<FloatEbmType> featureValues;
   std::vector<FloatEbmType> expectedCuts;

   FloatEbmType val = 10.0;
   for(int i = 0; i < 1000; ++i) {
      featureValues.push_back(val);
      val = std::nextafter(val, std::numeric_limits<FloatEbmType>::max());
      expectedCuts.push_back(val);
   }
   expectedCuts.pop_back();

   std::vector<FloatEbmType> cuts(featureValues.size() - 1, illegalVal);

   IntEbmType countCuts = CutUniform(featureValues.size(), &featureValues[0], cuts.size(), &cuts[0]);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cuts[i]);
      }
   }
}

TEST_CASE("CutUniform, insufficient floting point range for all cuts") {
   std::vector<FloatEbmType> featureValues;
   std::vector<FloatEbmType> expectedCuts;

   FloatEbmType val = 10.0;
   for(int i = 0; i < 1000; ++i) {
      featureValues.push_back(val);
      val = std::nextafter(val, std::numeric_limits<FloatEbmType>::max());
      expectedCuts.push_back(val);
   }
   expectedCuts.pop_back();

   std::vector<FloatEbmType> cuts(featureValues.size(), illegalVal);

   IntEbmType countCuts = CutUniform(featureValues.size(), &featureValues[0], cuts.size(), &cuts[0]);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cuts[i]);
      }
   }
}

TEST_CASE("CutUniform, one cut, -infinity and +infinity") {
   std::vector<FloatEbmType> featureValues {
      std::numeric_limits<FloatEbmType>::infinity(),
      -std::numeric_limits<FloatEbmType>::infinity(),
   };
   std::vector<FloatEbmType> cuts(1, illegalVal);

   IntEbmType countCuts = CutUniform(featureValues.size(), &featureValues[0], cuts.size(), &cuts[0]);

   CHECK(1 == countCuts);
   CHECK(0 == cuts[0]);
}

TEST_CASE("CutUniform, infinite diff, even cuts") {
   IntEbmType countCuts = 2;

   std::vector<FloatEbmType> featureValues {
      std::numeric_limits<FloatEbmType>::max(),
      std::numeric_limits<FloatEbmType>::lowest(),
   };
   const std::vector<FloatEbmType> expectedCuts { -5.9923104495410517e+307, 5.9923104495410517e+307 };

   std::vector<FloatEbmType> cuts(static_cast<size_t>(countCuts), illegalVal);

   countCuts = CutUniform(featureValues.size(), &featureValues[0], cuts.size(), &cuts[0]);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cuts[i]);
      }
   }
}

TEST_CASE("CutUniform, infinite diff, odd cuts") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues {
      std::numeric_limits<FloatEbmType>::lowest(),
      std::numeric_limits<FloatEbmType>::max(),
   };
   const std::vector<FloatEbmType> expectedCuts { -8.9884656743115785e+307, 0, 8.9884656743115785e+307 };

   std::vector<FloatEbmType> cuts(static_cast<size_t>(countCuts), illegalVal);

   countCuts = CutUniform(featureValues.size(), &featureValues[0], cuts.size(), &cuts[0]);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cuts[i]);
      }
   }
}

TEST_CASE("CutUniform, anchor on min") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues { -2, std::nextafter(2, 1000000) };
   const std::vector<FloatEbmType> expectedCuts { -1, 0, 1 };

   std::vector<FloatEbmType> cuts(static_cast<size_t>(countCuts), illegalVal);

   countCuts = CutUniform(featureValues.size(), &featureValues[0], cuts.size(), &cuts[0]);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cuts[i]);
      }
   }
}

TEST_CASE("CutUniform, anchor on max") {
   IntEbmType countCuts = 3;

   std::vector<FloatEbmType> featureValues { std::nextafter(-2, -100000), 2 };
   const std::vector<FloatEbmType> expectedCuts { -1, 0, 1 };

   std::vector<FloatEbmType> cuts(static_cast<size_t>(countCuts), illegalVal);

   countCuts = CutUniform(featureValues.size(), &featureValues[0], cuts.size(), &cuts[0]);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cuts[i]);
      }
   }
}

TEST_CASE("CutUniform, min and max at interior positions") {
   IntEbmType countCuts = 9;

   std::vector<FloatEbmType> featureValues { 1, 2, 3, 4, 5, 0, 10, 6, 7, 8, 9 };
   const std::vector<FloatEbmType> expectedCuts { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

   std::vector<FloatEbmType> cuts(static_cast<size_t>(countCuts), illegalVal);

   countCuts = CutUniform(featureValues.size(), &featureValues[0], countCuts, &cuts[0]);

   size_t cCuts = static_cast<size_t>(countCuts);
   CHECK(expectedCuts.size() == cCuts);
   if(expectedCuts.size() == cCuts) {
      for(size_t i = 0; i < cCuts; ++i) {
         CHECK_APPROX(expectedCuts[i], cuts[i]);
      }
   }
}

TEST_CASE("CutUniform, low start, hit float resolution before end") {
   std::vector<FloatEbmType> featureValues;

   FloatEbmType val = -std::numeric_limits<FloatEbmType>::min();
   for(int i = 0; i < 5; ++i) {
      // backup a few ticks
      val = TickLowerTest(val);
   }
   for(size_t iPast = 0; iPast < 10; ) {
      if(std::numeric_limits<FloatEbmType>::min() <= val) {
         ++iPast;
      }
      featureValues.push_back(val);
      val = TickHigherTest(val);
   }

   // have just 1 hole in the middle
   std::vector<FloatEbmType> cuts(featureValues.size() - 2, illegalVal);

   IntEbmType countCuts = CutUniform(featureValues.size(), &featureValues[0], cuts.size(), &cuts[0]);

   CHECK(cuts.size() == static_cast<size_t>(countCuts));
}

TEST_CASE("CutUniform, high start, hit float resolution before end") {
   std::vector<FloatEbmType> featureValues;

   FloatEbmType val = -std::numeric_limits<FloatEbmType>::min();
   for(int i = 0; i < 10; ++i) {
      // backup a few ticks
      val = TickLowerTest(val);
   }
   for(size_t iPast = 0; iPast < 5; ) {
      if(std::numeric_limits<FloatEbmType>::min() <= val) {
         ++iPast;
      }
      featureValues.push_back(val);
      val = TickHigherTest(val);
   }

   // have just 1 hole in the middle
   std::vector<FloatEbmType> cuts(featureValues.size() - 2, illegalVal);

   IntEbmType countCuts = CutUniform(featureValues.size(), &featureValues[0], cuts.size(), &cuts[0]);

   CHECK(cuts.size() == static_cast<size_t>(countCuts));
}

TEST_CASE("CutUniform, stress test reproducible") {
   constexpr double k_denormToNorm = 4503599627370496.0;
   static_assert(k_denormToNorm == std::numeric_limits<double>::min() / std::numeric_limits<double>::denorm_min(),
      "bad min to denorm_min ratio");

   std::vector<FloatEbmType> featureValues { 0, 0 };

   double interestingValues[] = {
      std::numeric_limits<FloatEbmType>::lowest(),
      -2 * k_denormToNorm * std::numeric_limits<double>::min(),
      -k_denormToNorm * std::numeric_limits<double>::min(),
      -std::numeric_limits<FloatEbmType>::min(),
      std::numeric_limits<FloatEbmType>::min(),
      k_denormToNorm * std::numeric_limits<double>::min(),
      2 * k_denormToNorm * std::numeric_limits<double>::min(),
      std::numeric_limits<FloatEbmType>::max(),
   };
   const size_t cInteresting = static_cast<int>(sizeof(interestingValues) / sizeof(interestingValues[0]));

   // 97 is prime
   std::vector<FloatEbmType> cuts(97, illegalVal);

   double result = 0.0;

   std::mt19937 randomize(42);
   for(int i = 0; i < 10000; ++i) {
      size_t val = static_cast<size_t>(randomize());
      bool isPositiveLow = static_cast<bool>(0x1 & val);
      val >>= 1;
      bool isPositiveHigh = static_cast<bool>(0x1 & val);
      val >>= 1;
      size_t shiftLow = static_cast<size_t>(0x7 & val); // 0-3 shift
      val >>= 3;
      size_t shiftHigh = static_cast<size_t>(0x7 & val); // 0-3 shift
      val >>= 3;
      size_t iInterestingLow = val % cInteresting;
      val /= cInteresting;
      size_t iInterestingHigh = val % cInteresting;
      val /= cInteresting;

      double low = DenormalizeTest(interestingValues[iInterestingLow]);
      double high = DenormalizeTest(interestingValues[iInterestingHigh]);

      if(isPositiveLow) {
         for(size_t iTick = 0; iTick < shiftLow; ++iTick) {
            if(low != std::numeric_limits<FloatEbmType>::lowest()) {
               low = TickLowerTest(low);
            }
         }
      } else {
         for(size_t iTick = 0; iTick < shiftLow; ++iTick) {
            if(low != std::numeric_limits<FloatEbmType>::max()) {
               low = TickHigherTest(low);
            }
         }
      }

      if(isPositiveHigh) {
         for(size_t iTick = 0; iTick < shiftHigh; ++iTick) {
            if(high != std::numeric_limits<FloatEbmType>::lowest()) {
               high = TickLowerTest(high);
            }
         }
      } else {
         for(size_t iTick = 0; iTick < shiftHigh; ++iTick) {
            if(high != std::numeric_limits<FloatEbmType>::max()) {
               high = TickHigherTest(high);
            }
         }
      }

      featureValues[0] = low;
      featureValues[1] = high;

      IntEbmType iCuts = static_cast<IntEbmType>(val % cuts.size());
      IntEbmType countCuts = CutUniform(featureValues.size(), &featureValues[0], iCuts, &cuts[0]);

      size_t cCuts = static_cast<size_t>(countCuts);
      for(size_t iCut = 0; iCut < cCuts; ++iCut) {
         double oneCut = cuts[iCut];
         if(0.0 != oneCut) {
            if(std::abs(oneCut) < 1.0) {
               do {
                  oneCut *= 2.0;
               } while(std::abs(oneCut) < 1.0);
            } else {
               do {
                  oneCut *= 0.5;
               } while(1.0 < std::abs(oneCut));
            }
         }
         if(result < 0.0 && 0.0 < oneCut || 0.0 < result && oneCut < 0.0) {
            result += oneCut;
         } else {
            result -= oneCut;
         }
      }
   }

   CHECK(-0.65942253092906411 == result);
}

