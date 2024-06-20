// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch_test.hpp"

#include <float.h>

#include "libebm.h"
#include "libebm_test.hpp"
#include "RandomStreamTest.hpp"

static constexpr TestPriority k_filePriority = TestPriority::SuggestGraphBounds;

// TODO : re-formulate these tests after we reach agreement on how the graph bounds are suposed to work

TEST_CASE("SuggestGraphBounds, 0 cuts, min -inf, max nan") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 0;
   static constexpr double minFeatureVal = -std::numeric_limits<double>::infinity();
   static constexpr double lowestCut = -1;
   static constexpr double highestCut = -1;
   static constexpr double maxFeatureVal = std::numeric_limits<double>::quiet_NaN();

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(-std::numeric_limits<double>::infinity() == lowGraphBound);
   CHECK(-std::numeric_limits<double>::infinity() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 0 cuts, min nan, max inf") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 0;
   static constexpr double minFeatureVal = std::numeric_limits<double>::quiet_NaN();
   static constexpr double lowestCut = -1;
   static constexpr double highestCut = -1;
   static constexpr double maxFeatureVal = std::numeric_limits<double>::infinity();

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(std::numeric_limits<double>::infinity() == lowGraphBound);
   CHECK(std::numeric_limits<double>::infinity() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 0 cuts, min 7, max 99") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 0;
   static constexpr double minFeatureVal = 7;
   static constexpr double lowestCut = -1;
   static constexpr double highestCut = -1;
   static constexpr double maxFeatureVal = 99;

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(7 == lowGraphBound);
   CHECK(99 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, all 6") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 1;
   static constexpr double minFeatureVal = 6;
   static constexpr double lowestCut = 6;
   static constexpr double highestCut = 6;
   static constexpr double maxFeatureVal = 6;

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(6 == lowGraphBound);
   CHECK(6 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, progression") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 1;
   static constexpr double minFeatureVal = 6;
   static constexpr double lowestCut = 7;
   static constexpr double highestCut = 7;
   static constexpr double maxFeatureVal = 8;

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(6 == lowGraphBound);
   CHECK(8 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, mismatched low high") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 1;
   static constexpr double minFeatureVal = -1;
   static constexpr double lowestCut = -2;
   static constexpr double highestCut = -2;
   static constexpr double maxFeatureVal = 1;

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(lowGraphBound < -2);
   CHECK(1 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, mismatched low high") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 1;
   static constexpr double minFeatureVal = -2;
   static constexpr double lowestCut = 0;
   static constexpr double highestCut = 0;
   static constexpr double maxFeatureVal = -1;

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(-2 == lowGraphBound);
   CHECK(0 < highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, out of range high") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 1;
   static constexpr double minFeatureVal = 1;
   static constexpr double lowestCut = 0;
   static constexpr double highestCut = 0;
   static constexpr double maxFeatureVal = 2;

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(lowGraphBound < 0);
   CHECK(2 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, min -inf") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 1;
   static constexpr double minFeatureVal = -std::numeric_limits<double>::infinity();
   static constexpr double lowestCut = std::numeric_limits<double>::lowest() + double{1e300};
   static constexpr double highestCut = std::numeric_limits<double>::lowest() + double{1e300};
   static constexpr double maxFeatureVal = std::numeric_limits<double>::lowest() + double{1.5e300};

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(minFeatureVal == lowGraphBound);
   CHECK(maxFeatureVal == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, max +inf") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 1;
   static constexpr double minFeatureVal = std::numeric_limits<double>::max() - 1.5e300;
   static constexpr double lowestCut = std::numeric_limits<double>::max() - 1e300;
   static constexpr double highestCut = std::numeric_limits<double>::max() - 1e300;
   static constexpr double maxFeatureVal = std::numeric_limits<double>::infinity();

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(minFeatureVal == lowGraphBound);
   CHECK(maxFeatureVal == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, overflow diff") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 1;
   static constexpr double minFeatureVal = 0;
   static constexpr double lowestCut = std::numeric_limits<double>::lowest() + 1e300;
   static constexpr double highestCut = lowestCut;
   static constexpr double maxFeatureVal = std::numeric_limits<double>::max() - 1e300;

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(-std::numeric_limits<double>::infinity() == lowGraphBound);
   CHECK(maxFeatureVal == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, min longest") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 1;
   static constexpr double minFeatureVal = 98;
   static constexpr double lowestCut = 100;
   static constexpr double highestCut = 100;
   static constexpr double maxFeatureVal = 101;

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(98 == lowGraphBound);
   CHECK(101 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, max longest") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 1;
   static constexpr double minFeatureVal = 99;
   static constexpr double lowestCut = 100;
   static constexpr double highestCut = 100;
   static constexpr double maxFeatureVal = 102;

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(99 == lowGraphBound);
   CHECK(102 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, overflow high") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 1;
   static constexpr double minFeatureVal = std::numeric_limits<double>::max() - 1e307;
   static constexpr double lowestCut = std::numeric_limits<double>::max() - 1e306;
   static constexpr double highestCut = std::numeric_limits<double>::max() - 1e306;
   static constexpr double maxFeatureVal = std::numeric_limits<double>::max() - 1e307;

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(minFeatureVal == lowGraphBound);
   CHECK(std::numeric_limits<double>::infinity() == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 1 cuts, overflow low") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 1;
   static constexpr double minFeatureVal = std::numeric_limits<double>::lowest() + 1e307;
   static constexpr double lowestCut = std::numeric_limits<double>::lowest() + 1e306;
   static constexpr double highestCut = std::numeric_limits<double>::lowest() + 1e306;
   static constexpr double maxFeatureVal = std::numeric_limits<double>::lowest() + 1e307;

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(-std::numeric_limits<double>::infinity() == lowGraphBound);
   CHECK(maxFeatureVal == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 2 cuts") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 2;
   static constexpr double minFeatureVal = 5;
   static constexpr double lowestCut = 6;
   static constexpr double highestCut = 7;
   static constexpr double maxFeatureVal = 8;

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(5 == lowGraphBound);
   CHECK(8 == highGraphBound);
}

TEST_CASE("SuggestGraphBounds, 4 cuts") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 4;
   static constexpr double minFeatureVal = 5;
   static constexpr double lowestCut = 6;
   static constexpr double highestCut = 7;
   static constexpr double maxFeatureVal = 8;

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK_APPROX(lowGraphBound, 5);
   CHECK_APPROX(highGraphBound, 8);
}

TEST_CASE("SuggestGraphBounds, 2 cuts, overflow diff") {
   double lowGraphBound;
   double highGraphBound;

   static constexpr IntEbm countCuts = 2;
   static constexpr double minFeatureVal = -1;
   static constexpr double lowestCut = std::numeric_limits<double>::lowest();
   static constexpr double highestCut = std::numeric_limits<double>::max();
   static constexpr double maxFeatureVal = 1;

   SuggestGraphBounds(countCuts, lowestCut, highestCut, minFeatureVal, maxFeatureVal, &lowGraphBound, &highGraphBound);

   CHECK(-std::numeric_limits<double>::infinity() == lowGraphBound);
   CHECK(std::numeric_limits<double>::infinity() == highGraphBound);
}

TEST_CASE("SafeMean, 4 values") {
   double vals[]{1.0, 2.5, 10, 100};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double mean = -99.0;

   const ErrorEbm error = SafeMean(cVals, 1, vals, &mean);
   CHECK(Error_None == error);
   CHECK(mean == 28.375);
}

TEST_CASE("SafeMean, 4 values in bag") {
   double vals[]{1.0, -99.0, 2.5, -99.0, 10, -99.0, 100, -99.0};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double mean[2];

   const ErrorEbm error = SafeMean(cVals / 2, 2, vals, mean);
   CHECK(Error_None == error);
   CHECK(mean[0] == 28.375);
   CHECK(mean[1] == -99.0);
}

TEST_CASE("SafeMean, -inf") {
   double vals[]{1.0, -INFINITY, 10, 100};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double mean = -99.0;

   const ErrorEbm error = SafeMean(cVals, 1, vals, &mean);
   CHECK(Error_None == error);
   CHECK(mean == -INFINITY);
}

TEST_CASE("SafeMean, +inf") {
   double vals[]{1.0, INFINITY, 10, 100};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double mean = -99.0;

   const ErrorEbm error = SafeMean(cVals, 1, vals, &mean);
   CHECK(Error_None == error);
   CHECK(mean == INFINITY);
}

TEST_CASE("SafeMean, nan") {
   double vals[]{1.0, NAN, 10, 100};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double mean = -99.0;

   const ErrorEbm error = SafeMean(cVals, 1, vals, &mean);
   CHECK(Error_None == error);
   CHECK(std::isnan(mean));
}

TEST_CASE("SafeMean, nan with others") {
   double vals[]{1.0, INFINITY, NAN, -INFINITY};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double mean = -99.0;

   const ErrorEbm error = SafeMean(cVals, 1, vals, &mean);
   CHECK(Error_None == error);
   CHECK(std::isnan(mean));
}

TEST_CASE("SafeMean, +inf and -inf") {
   double vals[]{1.0, INFINITY, -INFINITY, 100};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double mean = -99.0;

   const ErrorEbm error = SafeMean(cVals, 1, vals, &mean);
   CHECK(Error_None == error);
   CHECK(mean == INFINITY);
}

TEST_CASE("SafeMean, -inf and +inf") {
   double vals[]{1.0, -INFINITY, +INFINITY, 100};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double mean = -99.0;

   const ErrorEbm error = SafeMean(cVals, 1, vals, &mean);
   CHECK(Error_None == error);
   CHECK(mean == INFINITY);
}

TEST_CASE("SafeMean, more -inf") {
   double vals[]{1.0, -INFINITY, +INFINITY, -INFINITY};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double mean = -99.0;

   const ErrorEbm error = SafeMean(cVals, 1, vals, &mean);
   CHECK(Error_None == error);
   CHECK(mean == -INFINITY);
}

TEST_CASE("SafeMean, more +inf") {
   double vals[]{1.0, -INFINITY, +INFINITY, +INFINITY};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double mean = -99.0;

   const ErrorEbm error = SafeMean(cVals, 1, vals, &mean);
   CHECK(Error_None == error);
   CHECK(mean == INFINITY);
}

TEST_CASE("SafeMean, no overflow positive") {
   double vals[]{1.0, DBL_MAX, DBL_MAX, -DBL_MAX};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double mean = -99.0;

   const ErrorEbm error = SafeMean(cVals, 1, vals, &mean);
   CHECK(Error_None == error);
   CHECK(std::isfinite(mean));
}

TEST_CASE("SafeMean, no overflow negative") {
   double vals[]{1.0, -DBL_MAX, -DBL_MAX, DBL_MAX};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double mean = -99.0;

   const ErrorEbm error = SafeMean(cVals, 1, vals, &mean);
   CHECK(Error_None == error);
   CHECK(std::isfinite(mean));
}



TEST_CASE("SafeStandardDeviation, 4 values") {
   double vals[]{1.0, 2.5, 10, 100};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double stddev = -99.0;

   const ErrorEbm error = SafeStandardDeviation(cVals, 1, vals, &stddev);
   CHECK(Error_None == error);
   CHECK_APPROX(stddev, 41.493034053922834);
}

TEST_CASE("SafeStandardDeviation, 3 values in bag") {
   double vals[]{1.0, -99.0, 2.5, -99.0, 10, -99.0, 100, -99.0};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double stddev[2];

   const ErrorEbm error = SafeStandardDeviation(cVals / 2, 2, vals, stddev);
   CHECK(Error_None == error);
   CHECK_APPROX(stddev[0], 41.493034053922834);
   CHECK(stddev[1] == 0.0);
}

TEST_CASE("SafeStandardDeviation, -inf") {
   double vals[]{1.0, -INFINITY, 10, 100};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double stddev = -99.0;

   const ErrorEbm error = SafeStandardDeviation(cVals, 1, vals, &stddev);
   CHECK(Error_None == error);
   CHECK(stddev == INFINITY);
}

TEST_CASE("SafeStandardDeviation, +inf") {
   double vals[]{1.0, INFINITY, 10, 100};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double stddev = -99.0;

   const ErrorEbm error = SafeStandardDeviation(cVals, 1, vals, &stddev);
   CHECK(Error_None == error);
   CHECK(stddev == INFINITY);
}

TEST_CASE("SafeStandardDeviation, nan") {
   double vals[]{1.0, NAN, 10, 100};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double stddev = -99.0;

   const ErrorEbm error = SafeStandardDeviation(cVals, 1, vals, &stddev);
   CHECK(Error_None == error);
   CHECK(std::isnan(stddev));
}

TEST_CASE("SafeStandardDeviation, nan with others") {
   double vals[]{1.0, INFINITY, NAN, -INFINITY};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double stddev = -99.0;

   const ErrorEbm error = SafeStandardDeviation(cVals, 1, vals, &stddev);
   CHECK(Error_None == error);
   CHECK(std::isnan(stddev));
}

TEST_CASE("SafeStandardDeviation, +inf and -inf") {
   double vals[]{1.0, INFINITY, -INFINITY, 100};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double stddev = -99.0;

   const ErrorEbm error = SafeStandardDeviation(cVals, 1, vals, &stddev);
   CHECK(Error_None == error);
   CHECK(stddev == INFINITY);
}

TEST_CASE("SafeStandardDeviation, -inf and +inf") {
   double vals[]{1.0, -INFINITY, +INFINITY, 100};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double stddev = -99.0;

   const ErrorEbm error = SafeStandardDeviation(cVals, 1, vals, &stddev);
   CHECK(Error_None == error);
   CHECK(stddev == INFINITY);
}

TEST_CASE("SafeStandardDeviation, more -inf") {
   double vals[]{1.0, -INFINITY, +INFINITY, -INFINITY};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double stddev = -99.0;

   const ErrorEbm error = SafeStandardDeviation(cVals, 1, vals, &stddev);
   CHECK(Error_None == error);
   CHECK(stddev == INFINITY);
}

TEST_CASE("SafeStandardDeviation, more +inf") {
   double vals[]{1.0, -INFINITY, +INFINITY, +INFINITY};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double stddev = -99.0;

   const ErrorEbm error = SafeStandardDeviation(cVals, 1, vals, &stddev);
   CHECK(Error_None == error);
   CHECK(stddev == INFINITY);
}

TEST_CASE("SafeStandardDeviation, no overflow positive") {
   double vals[]{1.0, DBL_MAX, DBL_MAX, -DBL_MAX};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double stddev = -99.0;

   const ErrorEbm error = SafeStandardDeviation(cVals, 1, vals, &stddev);
   CHECK(Error_None == error);
   CHECK(std::isfinite(stddev));
}

TEST_CASE("SafeStandardDeviation, no overflow negative") {
   double vals[]{1.0, -DBL_MAX, -DBL_MAX, DBL_MAX};
   const size_t cVals = sizeof(vals) / sizeof(vals[0]);
   double stddev = -99.0;

   const ErrorEbm error = SafeStandardDeviation(cVals, 1, vals, &stddev);
   CHECK(Error_None == error);
   CHECK(std::isfinite(stddev));
}
