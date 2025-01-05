// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch_test.hpp"

#include <float.h>

#include "libebm.h"
#include "libebm_test.hpp"

static constexpr TestPriority k_filePriority = TestPriority::Purify;

static constexpr double k_tolerancePurityDefault = 1e-6;
static constexpr BoolEbm k_isRandomizedDefault = EBM_TRUE;
static constexpr BoolEbm k_isMulticlassNormalization = EBM_TRUE;

TEST_CASE("Purify pure single dimensional 3, unweighted") {
   const IntEbm dimensionLengths[]{3};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   const double weights[]{2.0, 2.0, 2.0};
   double scores[]{3.0, -1.5, -1.5};
   double scoresExpected[]{3.0, -1.5, -1.5};
   double impurities[1]; // should be ignored
   double residualIntercept;

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         k_isMulticlassNormalization,
         IntEbm{1},
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept);
   CHECK(Error_None == error);
   CHECK(0.0 == residualIntercept); // it started off pure
   for(size_t i = 0; i < sizeof(scoresExpected) / sizeof(scoresExpected[0]); ++i) {
      CHECK_APPROX(scores[i], scoresExpected[i]);
   }

   const double impurity = MeasureImpurity(1, 0, cDimensions, dimensionLengths, weights, scores);
   CHECK(-0.001 < impurity && impurity < 0.001);
}

TEST_CASE("Purify impure single dimensional 3, unweighted") {
   const IntEbm dimensionLengths[]{3};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   const double weights[]{2.0, 2.0, 2.0};
   double scores[]{4.0, 0.0, -1.0};
   double scoresExpected[]{3.0, -1.0, -2.0};
   double impurities[1]; // should be ignored
   double residualIntercept;

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         k_isMulticlassNormalization,
         IntEbm{1},
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept);
   CHECK(Error_None == error);
   CHECK(1.0 == residualIntercept); // it started off pure
   for(size_t i = 0; i < sizeof(scoresExpected) / sizeof(scoresExpected[0]); ++i) {
      CHECK_APPROX(scores[i], scoresExpected[i]);
   }
   const double impurity = MeasureImpurity(1, 0, cDimensions, dimensionLengths, weights, scores);
   CHECK(-0.001 < impurity && impurity < 0.001);
}

TEST_CASE("Purify pure 2x2, unweighted") {
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   const double weights[]{2.0, 2.0, 2.0, 2.0};
   double scores[]{3.0, -3.0, -3.0, 3.0};
   double scoresExpected[]{3.0, -3.0, -3.0, 3.0};
   double impurities[4];
   double impuritiesExpected[]{0.0, 0.0, 0.0, 0.0};
   double residualIntercept;

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         k_isMulticlassNormalization,
         IntEbm{1},
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept);
   CHECK(Error_None == error);
   CHECK(0.0 == residualIntercept); // it started off pure
   for(size_t i = 0; i < sizeof(scoresExpected) / sizeof(scoresExpected[0]); ++i) {
      CHECK_APPROX(scores[i], scoresExpected[i]);
   }
   for(size_t i = 0; i < sizeof(impuritiesExpected) / sizeof(impuritiesExpected[0]); ++i) {
      CHECK_APPROX(impurities[i], impuritiesExpected[i]);
   }
   const double impurity = MeasureImpurity(1, 0, cDimensions, dimensionLengths, weights, scores);
   CHECK(-0.001 < impurity && impurity < 0.001);
}

TEST_CASE("Purify pure 2x2, weighted") {
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   const double weights[]{3.0, 2.0, 2.0, 3.0};
   double scores[]{2.0, -3.0, -3.0, 2.0};
   double scoresExpected[]{2.0, -3.0, -3.0, 2.0};
   double impurities[4];
   double impuritiesExpected[]{0.0, 0.0, 0.0, 0.0};
   double residualIntercept;

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         k_isMulticlassNormalization,
         IntEbm{1},
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept);
   CHECK(Error_None == error);
   CHECK(0.0 == residualIntercept); // it started off pure
   for(size_t i = 0; i < sizeof(scoresExpected) / sizeof(scoresExpected[0]); ++i) {
      CHECK_APPROX(scores[i], scoresExpected[i]);
   }
   for(size_t i = 0; i < sizeof(impuritiesExpected) / sizeof(impuritiesExpected[0]); ++i) {
      CHECK_APPROX(impurities[i], impuritiesExpected[i]);
   }
   const double impurity = MeasureImpurity(1, 0, cDimensions, dimensionLengths, weights, scores);
   CHECK(-0.001 < impurity && impurity < 0.001);
}

TEST_CASE("Purify pure 2x2 + incercept, unweighted") {
   // take the pure 2x2 case, and add an intercept of 1.0
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   const double weights[]{2.0, 2.0, 2.0, 2.0};
   double scores[]{3.0 + 1.0, -3.0 + 1.0, -3.0 + 1.0, 3.0 + 1.0};
   double scoresExpected[]{3.0, -3.0, -3.0, 3.0};
   double impurities[4];
   double impuritiesExpected[]{0.0, 0.0, 0.0, 0.0};
   double residualIntercept;

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         k_isMulticlassNormalization,
         IntEbm{1},
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept);
   CHECK(Error_None == error);
   CHECK(1.0 == residualIntercept); // it started off pure
   for(size_t i = 0; i < sizeof(scoresExpected) / sizeof(scoresExpected[0]); ++i) {
      CHECK_APPROX(scores[i], scoresExpected[i]);
   }
   for(size_t i = 0; i < sizeof(impuritiesExpected) / sizeof(impuritiesExpected[0]); ++i) {
      CHECK_APPROX(impurities[i], impuritiesExpected[i]);
   }
   const double impurity = MeasureImpurity(1, 0, cDimensions, dimensionLengths, weights, scores);
   CHECK(-0.001 < impurity && impurity < 0.001);
}

TEST_CASE("Purify pure 2x2 + intercept, weighted") {
   // take the pure 2x2 case (weighted), and add an intercept of 1.0
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   const double weights[]{3.0, 2.0, 2.0, 3.0};
   double scores[]{2.0 + 1.0, -3.0 + 1.0, -3.0 + 1.0, 2.0 + 1.0};
   double scoresExpected[]{2.0, -3.0, -3.0, 2.0};
   double impurities[4];
   double impuritiesExpected[]{0.0, 0.0, 0.0, 0.0};
   double residualIntercept;

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         k_isMulticlassNormalization,
         IntEbm{1},
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept);
   CHECK(Error_None == error);
   CHECK(1.0 == residualIntercept); // it started off pure
   for(size_t i = 0; i < sizeof(scoresExpected) / sizeof(scoresExpected[0]); ++i) {
      CHECK_APPROX(scores[i], scoresExpected[i]);
   }
   for(size_t i = 0; i < sizeof(impuritiesExpected) / sizeof(impuritiesExpected[0]); ++i) {
      CHECK_APPROX(impurities[i], impuritiesExpected[i]);
   }
   const double impurity = MeasureImpurity(1, 0, cDimensions, dimensionLengths, weights, scores);
   CHECK(-0.001 < impurity && impurity < 0.001);
}

TEST_CASE("Purify pure 2x2 + impurities, unweighted") {
   // take the pure 2x2 case, and add an impure [[1.0, -1.0], [1.0, -1.0]]
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   const double weights[]{2.0, 2.0, 2.0, 2.0};
   double scores[]{3.0 + 1.0 + 1.0, -3.0 + 1.0 - 1.0, -3.0 - 1.0 + 1.0, 3.0 - 1.0 - 1.0};
   double scoresExpected[]{3.0, -3.0, -3.0, 3.0};
   double impurities[4];
   double impuritiesExpected[]{1.0, -1.0, 1.0, -1.0};
   double residualIntercept;

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         k_isMulticlassNormalization,
         IntEbm{1},
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept);
   CHECK(Error_None == error);
   CHECK(0.0 == residualIntercept); // it started off pure
   for(size_t i = 0; i < sizeof(scoresExpected) / sizeof(scoresExpected[0]); ++i) {
      CHECK_APPROX(scores[i], scoresExpected[i]);
   }
   for(size_t i = 0; i < sizeof(impuritiesExpected) / sizeof(impuritiesExpected[0]); ++i) {
      CHECK_APPROX(impurities[i], impuritiesExpected[i]);
   }
   const double impurity = MeasureImpurity(1, 0, cDimensions, dimensionLengths, weights, scores);
   CHECK(-0.001 < impurity && impurity < 0.001);
}

TEST_CASE("Purify pure 2x2 + impurities, weighted") {
   // take the pure 2x2 case, and add an impure [[1.0, -1.0], [1.0, -1.0]]
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   const double weights[]{3.0, 2.0, 2.0, 3.0};
   double scores[]{2.0 + 1.0 + 1.0, -3.0 + 1.0 - 1.0, -3.0 - 1.0 + 1.0, 2.0 - 1.0 - 1.0};
   double scoresExpected[]{2.0, -3.0, -3.0, 2.0};
   double impurities[4];
   double impuritiesExpected[]{1.0, -1.0, 1.0, -1.0};
   double residualIntercept;

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         EBM_FALSE, // the impurities array isn't identifiable, so use a predictable ordering
         k_isMulticlassNormalization,
         IntEbm{1},
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept);
   CHECK(Error_None == error);
   CHECK(-0.000001 < residualIntercept && residualIntercept < 0.000001); // it started off pure
   for(size_t i = 0; i < sizeof(scoresExpected) / sizeof(scoresExpected[0]); ++i) {
      CHECK_APPROX(scores[i], scoresExpected[i]);
   }
   for(size_t i = 0; i < sizeof(impuritiesExpected) / sizeof(impuritiesExpected[0]); ++i) {
      CHECK_APPROX(impurities[i], impuritiesExpected[i]);
   }
   const double impurity = MeasureImpurity(1, 0, cDimensions, dimensionLengths, weights, scores);
   CHECK(-0.001 < impurity && impurity < 0.001);
}

TEST_CASE("Purify simple 3x4") {
   const IntEbm dimensionLengths[]{3, 4};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
   const double weights[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
   double impurities[7];
   double residualIntercept;

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         k_isMulticlassNormalization,
         IntEbm{1},
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept);
   CHECK(Error_None == error);
   CHECK(0.0 != residualIntercept);
   const double impurity = MeasureImpurity(1, 0, cDimensions, dimensionLengths, weights, scores);
   CHECK(-0.001 < impurity && impurity < 0.001);
}

TEST_CASE("Purify simple 3x4, infinite weights") {
   const IntEbm dimensionLengths[]{3, 4};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
   const double weights[]{1,
         std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(),
         4,
         std::numeric_limits<double>::infinity(),
         6,
         7,
         std::numeric_limits<double>::infinity(),
         9,
         10,
         std::numeric_limits<double>::infinity(),
         12};
   double impurities[7];
   double residualIntercept;

   ErrorEbm error = Purify(0.0,
         k_isRandomizedDefault,
         k_isMulticlassNormalization,
         IntEbm{1},
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept);
   CHECK(Error_None == error);
   CHECK_APPROX(residualIntercept, 5.8);
   // const double impurity = MeasureImpurity(1, 0, cDimensions, dimensionLengths, weights, scores);
   // CHECK(-0.001 < impurity && impurity < 0.001);
}

TEST_CASE("Purify simple 3x4, infinite weights, overflow") {
   const IntEbm dimensionLengths[]{3, 4};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{1, DBL_MAX, DBL_MAX, -DBL_MAX, 5, 6, -DBL_MAX, 8, 9, 10, 11, 12};
   const double weights[]{1,
         std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(),
         5,
         6,
         std::numeric_limits<double>::infinity(),
         8,
         9,
         10,
         11,
         12};
   double impurities[7];
   double residualIntercept;

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         k_isMulticlassNormalization,
         IntEbm{1},
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept);
   CHECK(Error_None == error);
   // const double impurity = MeasureImpurity(1, 0, cDimensions, dimensionLengths, weights, scores);
   // CHECK(-0.001 < impurity && impurity < 0.001);
}

TEST_CASE("Purify simple 3x4 with NaN") {
   const IntEbm dimensionLengths[]{3, 4};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{1, 2, 3, 4, std::numeric_limits<double>::quiet_NaN(), 6, 7, 8, 9, 10, 11, 12};
   const double weights[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
   double impurities[7];
   double residualIntercept;

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         k_isMulticlassNormalization,
         IntEbm{1},
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept);
   CHECK(Error_None == error);
   CHECK(0.0 != residualIntercept);
   // const double impurity = MeasureImpurity(1, 0, cDimensions, dimensionLengths, weights, scores);
   // CHECK(-0.001 < impurity && impurity < 0.001);
}

TEST_CASE("Purify simple 3x4 with -inf") {
   const IntEbm dimensionLengths[]{3, 4};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{1, 2, 3, 4, -std::numeric_limits<double>::infinity(), 6, 7, 8, 9, 10, 11, 12};
   const double weights[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
   double impurities[7];
   double residualIntercept;

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         k_isMulticlassNormalization,
         IntEbm{1},
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept);
   CHECK(Error_None == error);
   CHECK(0.0 != residualIntercept);
   // const double impurity = MeasureImpurity(1, 0, cDimensions, dimensionLengths, weights, scores);
   // CHECK(-0.001 < impurity && impurity < 0.001);
}

TEST_CASE("Purify simple 3x4 with overflow") {
   const IntEbm dimensionLengths[]{3, 4};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{1, 2, 3, DBL_MAX, DBL_MAX, DBL_MAX, 7, 8, 9, 10, 11, 12};
   const double weights[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
   double impurities[7];
   double residualIntercept;

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         k_isMulticlassNormalization,
         IntEbm{1},
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept);
   CHECK(Error_None == error);
   CHECK(0.0 != residualIntercept);
   // const double impurity = MeasureImpurity(1, 0, cDimensions, dimensionLengths, weights, scores);
   // CHECK(-0.001 < impurity && impurity < 0.001);
}

TEST_CASE("Purify simple 3x3x3") {
   const IntEbm dimensionLengths[]{3, 3, 3};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};
   const double weights[]{
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};
   double impurities[27];
   double residualIntercept;

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         k_isMulticlassNormalization,
         IntEbm{1},
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept);
   CHECK(Error_None == error);
   CHECK(0.0 != residualIntercept);
   const double impurity = MeasureImpurity(1, 0, cDimensions, dimensionLengths, weights, scores);
   CHECK(-0.001 < impurity && impurity < 0.001);
}

TEST_CASE("Purify simple 3x4x5") {
   const IntEbm dimensionLengths[]{3, 4, 5};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{1,
         2,
         3,
         4,
         5,
         6,
         7,
         8,
         9,
         10,
         11,
         12,
         13,
         14,
         15,
         16,
         17,
         18,
         19,
         20,
         21,
         22,
         23,
         24,
         25,
         26,
         27,
         28,
         29,
         30,
         31,
         32,
         33,
         34,
         35,
         36,
         37,
         38,
         39,
         40,
         41,
         42,
         43,
         44,
         45,
         46,
         47,
         48,
         49,
         50,
         51,
         52,
         53,
         54,
         55,
         56,
         57,
         58,
         59,
         60};

   const double weights[]{1,
         2,
         3,
         4,
         5,
         6,
         7,
         8,
         9,
         10,
         11,
         12,
         13,
         14,
         15,
         16,
         17,
         18,
         19,
         20,
         21,
         22,
         23,
         24,
         25,
         26,
         27,
         28,
         29,
         30,
         31,
         32,
         33,
         34,
         35,
         36,
         37,
         38,
         39,
         40,
         41,
         42,
         43,
         44,
         45,
         46,
         47,
         48,
         49,
         50,
         51,
         52,
         53,
         54,
         55,
         56,
         57,
         58,
         59,
         60};

   // impurities are: 4*5 + 3*5 + 3*4 = 47
   double impurities[47];
   double residualIntercept;

   ErrorEbm error = Purify(0,
         k_isRandomizedDefault,
         k_isMulticlassNormalization,
         IntEbm{1},
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept);
   CHECK(Error_None == error);
   CHECK(0.0 != residualIntercept);
   const double impurity = MeasureImpurity(1, 0, cDimensions, dimensionLengths, weights, scores);
   CHECK(-1e-20 < impurity && impurity < 1e-20);
}

TEST_CASE("Purify simple multiclass 3x4") {
   constexpr IntEbm cClasses = 2;
   const IntEbm dimensionLengths[]{3, 4};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{1, 10, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 9, 11, 10, 12, 11, 13, 12, 14};
   const double weights[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
   double impurities[7 * cClasses];
   double residualIntercept[cClasses];

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         k_isMulticlassNormalization,
         cClasses,
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept[0]);
   CHECK(Error_None == error);
   CHECK(0.0 != residualIntercept[0]);
   CHECK(0.0 != residualIntercept[1]);
   const double impurity0 = MeasureImpurity(cClasses, 0, cDimensions, dimensionLengths, weights, scores);
   CHECK(-0.001 < impurity0 && impurity0 < 0.001);
   const double impurity1 = MeasureImpurity(cClasses, 1, cDimensions, dimensionLengths, weights, scores);
   CHECK(-0.001 < impurity1 && impurity1 < 0.001);
}

TEST_CASE("Purify simple multiclass NaN") {
   constexpr IntEbm cClasses = 3;
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{std::numeric_limits<double>::quiet_NaN(),
         2,
         3,
         1,
         2,
         std::numeric_limits<double>::quiet_NaN(),
         1,
         std::numeric_limits<double>::quiet_NaN(),
         std::numeric_limits<double>::quiet_NaN(),
         std::numeric_limits<double>::quiet_NaN(),
         std::numeric_limits<double>::quiet_NaN(),
         std::numeric_limits<double>::quiet_NaN()};
   const double weights[]{1, 2, 3, 4};
   double impurities[4 * cClasses];
   double residualIntercept[cClasses];

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         EBM_FALSE,
         cClasses,
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept[0]);
   CHECK(Error_None == error);
   // CHECK(0.0 != residualIntercept[0]);
   // CHECK(0.0 != residualIntercept[1]);
   //  const double impurity0 = MeasureImpurity(cClasses, 0, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity0 && impurity0 < 0.001);
   //  const double impurity1 = MeasureImpurity(cClasses, 1, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity1 && impurity1 < 0.001);
}

TEST_CASE("Purify simple multiclass +inf") {
   constexpr IntEbm cClasses = 3;
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{std::numeric_limits<double>::infinity(),
         2,
         3,
         1,
         2,
         std::numeric_limits<double>::infinity(),
         1,
         std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity()};
   const double weights[]{1, 2, 3, 4};
   double impurities[4 * cClasses];
   double residualIntercept[cClasses];

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         EBM_FALSE,
         cClasses,
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept[0]);
   CHECK(Error_None == error);
   // CHECK(0.0 != residualIntercept[0]);
   // CHECK(0.0 != residualIntercept[1]);
   //  const double impurity0 = MeasureImpurity(cClasses, 0, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity0 && impurity0 < 0.001);
   //  const double impurity1 = MeasureImpurity(cClasses, 1, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity1 && impurity1 < 0.001);
}

TEST_CASE("Purify simple multiclass -inf") {
   constexpr IntEbm cClasses = 3;
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{-std::numeric_limits<double>::infinity(),
         2,
         3,
         1,
         2,
         -std::numeric_limits<double>::infinity(),
         1,
         -std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity()};
   const double weights[]{1, 2, 3, 4};
   double impurities[4 * cClasses];
   double residualIntercept[cClasses];

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         EBM_FALSE,
         cClasses,
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept[0]);
   CHECK(Error_None == error);
   // CHECK(0.0 != residualIntercept[0]);
   // CHECK(0.0 != residualIntercept[1]);
   //  const double impurity0 = MeasureImpurity(cClasses, 0, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity0 && impurity0 < 0.001);
   //  const double impurity1 = MeasureImpurity(cClasses, 1, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity1 && impurity1 < 0.001);
}

TEST_CASE("Purify simple multiclass overflow to -inf") {
   constexpr IntEbm cClasses = 3;
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{-DBL_MAX, 2, 3, 1, 2, -DBL_MAX, 1, -DBL_MAX, -DBL_MAX, -DBL_MAX, -DBL_MAX, -DBL_MAX};
   const double weights[]{1, 2, 3, 4};
   double impurities[4 * cClasses];
   double residualIntercept[cClasses];

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         EBM_FALSE,
         cClasses,
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept[0]);
   CHECK(Error_None == error);
   // CHECK(0.0 != residualIntercept[0]);
   // CHECK(0.0 != residualIntercept[1]);
   //  const double impurity0 = MeasureImpurity(cClasses, 0, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity0 && impurity0 < 0.001);
   //  const double impurity1 = MeasureImpurity(cClasses, 1, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity1 && impurity1 < 0.001);
}

TEST_CASE("Purify simple multiclass overflow to +inf") {
   constexpr IntEbm cClasses = 3;
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{DBL_MAX, 2, 3, 1, 2, DBL_MAX, 1, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX};
   const double weights[]{1, 2, 3, 4};
   double impurities[4 * cClasses];
   double residualIntercept[cClasses];

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         EBM_FALSE,
         cClasses,
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept[0]);
   CHECK(Error_None == error);
   // CHECK(0.0 != residualIntercept[0]);
   // CHECK(0.0 != residualIntercept[1]);
   //  const double impurity0 = MeasureImpurity(cClasses, 0, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity0 && impurity0 < 0.001);
   //  const double impurity1 = MeasureImpurity(cClasses, 1, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity1 && impurity1 < 0.001);
}

TEST_CASE("Purify simple multiclass +inf and NaN, overflow-inf,-overflow+inf, inf-inf") {
   constexpr IntEbm cClasses = 3;
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{std::numeric_limits<double>::infinity(),
         2,
         std::numeric_limits<double>::quiet_NaN(),
         DBL_MAX,
         DBL_MAX,
         -std::numeric_limits<double>::infinity(),
         -DBL_MAX,
         -DBL_MAX,
         std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity()};
   const double weights[]{1, 2, 3, 4};
   double impurities[4 * cClasses];
   double residualIntercept[cClasses];

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         EBM_FALSE,
         cClasses,
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept[0]);
   CHECK(Error_None == error);
   // CHECK(0.0 != residualIntercept[0]);
   // CHECK(0.0 != residualIntercept[1]);
   //  const double impurity0 = MeasureImpurity(cClasses, 0, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity0 && impurity0 < 0.001);
   //  const double impurity1 = MeasureImpurity(cClasses, 1, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity1 && impurity1 < 0.001);
}

TEST_CASE("Purify simple multiclass overflow the shifts") {
   constexpr IntEbm cClasses = 3;
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{DBL_MAX,
         DBL_MAX,
         -DBL_MAX,
         -DBL_MAX,
         -DBL_MAX,
         DBL_MAX,
         DBL_MAX * 0.9,
         DBL_MAX * 0.9,
         -DBL_MAX * 0.9,
         -DBL_MAX * 0.9,
         -DBL_MAX * 0.9,
         DBL_MAX * 0.9};
   const double weights[]{1, 2, 3, 4};
   double impurities[4 * cClasses];
   double residualIntercept[cClasses];

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         EBM_FALSE,
         cClasses,
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept[0]);
   CHECK(Error_None == error);
   // CHECK(0.0 != residualIntercept[0]);
   // CHECK(0.0 != residualIntercept[1]);
   //  const double impurity0 = MeasureImpurity(cClasses, 0, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity0 && impurity0 < 0.001);
   //  const double impurity1 = MeasureImpurity(cClasses, 1, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity1 && impurity1 < 0.001);
}

TEST_CASE("Purify simple multiclass 3x4, normalize classes") {
   constexpr IntEbm cClasses = 2;
   const IntEbm dimensionLengths[]{3, 4};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{1, 10, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 9, 11, 10, 12, 11, 13, 12, 14};
   const double weights[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
   double impurities[7 * cClasses];
   double residualIntercept[cClasses];

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         EBM_TRUE,
         cClasses,
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept[0]);
   CHECK(Error_None == error);
   CHECK(0.0 != residualIntercept[0]);
   CHECK(0.0 != residualIntercept[1]);
   const double impurity0 = MeasureImpurity(cClasses, 0, cDimensions, dimensionLengths, weights, scores);
   CHECK(-0.001 < impurity0 && impurity0 < 0.001);
   const double impurity1 = MeasureImpurity(cClasses, 1, cDimensions, dimensionLengths, weights, scores);
   CHECK(-0.001 < impurity1 && impurity1 < 0.001);
}

TEST_CASE("Purify simple multiclass NaN, normalize classes") {
   constexpr IntEbm cClasses = 3;
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{std::numeric_limits<double>::quiet_NaN(),
         2,
         3,
         1,
         2,
         std::numeric_limits<double>::quiet_NaN(),
         1,
         std::numeric_limits<double>::quiet_NaN(),
         std::numeric_limits<double>::quiet_NaN(),
         std::numeric_limits<double>::quiet_NaN(),
         std::numeric_limits<double>::quiet_NaN(),
         std::numeric_limits<double>::quiet_NaN()};
   const double weights[]{1, 2, 3, 4};
   double impurities[4 * cClasses];
   double residualIntercept[cClasses];

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         EBM_TRUE,
         cClasses,
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept[0]);
   CHECK(Error_None == error);
   // CHECK(0.0 != residualIntercept[0]);
   // CHECK(0.0 != residualIntercept[1]);
   //  const double impurity0 = MeasureImpurity(cClasses, 0, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity0 && impurity0 < 0.001);
   //  const double impurity1 = MeasureImpurity(cClasses, 1, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity1 && impurity1 < 0.001);
}

TEST_CASE("Purify simple multiclass +inf, normalize classes") {
   constexpr IntEbm cClasses = 3;
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{std::numeric_limits<double>::infinity(),
         2,
         3,
         1,
         2,
         std::numeric_limits<double>::infinity(),
         1,
         std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity()};
   const double weights[]{1, 2, 3, 4};
   double impurities[4 * cClasses];
   double residualIntercept[cClasses];

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         EBM_TRUE,
         cClasses,
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept[0]);
   CHECK(Error_None == error);
   // CHECK(0.0 != residualIntercept[0]);
   // CHECK(0.0 != residualIntercept[1]);
   //  const double impurity0 = MeasureImpurity(cClasses, 0, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity0 && impurity0 < 0.001);
   //  const double impurity1 = MeasureImpurity(cClasses, 1, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity1 && impurity1 < 0.001);
}

TEST_CASE("Purify simple multiclass -inf, normalize classes") {
   constexpr IntEbm cClasses = 3;
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{-std::numeric_limits<double>::infinity(),
         2,
         3,
         1,
         2,
         -std::numeric_limits<double>::infinity(),
         1,
         -std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity()};
   const double weights[]{1, 2, 3, 4};
   double impurities[4 * cClasses];
   double residualIntercept[cClasses];

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         EBM_TRUE,
         cClasses,
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept[0]);
   CHECK(Error_None == error);
   // CHECK(0.0 != residualIntercept[0]);
   // CHECK(0.0 != residualIntercept[1]);
   //  const double impurity0 = MeasureImpurity(cClasses, 0, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity0 && impurity0 < 0.001);
   //  const double impurity1 = MeasureImpurity(cClasses, 1, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity1 && impurity1 < 0.001);
}

TEST_CASE("Purify simple multiclass overflow to -inf, normalize classes") {
   constexpr IntEbm cClasses = 3;
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{-DBL_MAX, 2, 3, 1, 2, -DBL_MAX, 1, -DBL_MAX, -DBL_MAX, -DBL_MAX, -DBL_MAX, -DBL_MAX};
   const double weights[]{1, 2, 3, 4};
   double impurities[4 * cClasses];
   double residualIntercept[cClasses];

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         EBM_TRUE,
         cClasses,
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept[0]);
   CHECK(Error_None == error);
   // CHECK(0.0 != residualIntercept[0]);
   // CHECK(0.0 != residualIntercept[1]);
   //  const double impurity0 = MeasureImpurity(cClasses, 0, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity0 && impurity0 < 0.001);
   //  const double impurity1 = MeasureImpurity(cClasses, 1, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity1 && impurity1 < 0.001);
}

TEST_CASE("Purify simple multiclass overflow to +inf, normalize classes") {
   constexpr IntEbm cClasses = 3;
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{DBL_MAX, 2, 3, 1, 2, DBL_MAX, 1, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX};
   const double weights[]{1, 2, 3, 4};
   double impurities[4 * cClasses];
   double residualIntercept[cClasses];

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         EBM_TRUE,
         cClasses,
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept[0]);
   CHECK(Error_None == error);
   // CHECK(0.0 != residualIntercept[0]);
   // CHECK(0.0 != residualIntercept[1]);
   //  const double impurity0 = MeasureImpurity(cClasses, 0, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity0 && impurity0 < 0.001);
   //  const double impurity1 = MeasureImpurity(cClasses, 1, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity1 && impurity1 < 0.001);
}

TEST_CASE("Purify simple multiclass +inf and NaN, overflow-inf,-overflow+inf, inf-inf, normalize classes") {
   constexpr IntEbm cClasses = 3;
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{std::numeric_limits<double>::infinity(),
         2,
         std::numeric_limits<double>::quiet_NaN(),
         DBL_MAX,
         DBL_MAX,
         -std::numeric_limits<double>::infinity(),
         -DBL_MAX,
         -DBL_MAX,
         std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity()};
   const double weights[]{1, 2, 3, 4};
   double impurities[4 * cClasses];
   double residualIntercept[cClasses];

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         EBM_TRUE,
         cClasses,
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept[0]);
   CHECK(Error_None == error);
   // CHECK(0.0 != residualIntercept[0]);
   // CHECK(0.0 != residualIntercept[1]);
   //  const double impurity0 = MeasureImpurity(cClasses, 0, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity0 && impurity0 < 0.001);
   //  const double impurity1 = MeasureImpurity(cClasses, 1, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity1 && impurity1 < 0.001);
}

TEST_CASE("Purify simple multiclass overflow the shifts, normalize classes") {
   constexpr IntEbm cClasses = 3;
   const IntEbm dimensionLengths[]{2, 2};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{DBL_MAX,
         DBL_MAX,
         -DBL_MAX,
         -DBL_MAX,
         -DBL_MAX,
         DBL_MAX,
         DBL_MAX * 0.9,
         DBL_MAX * 0.9,
         -DBL_MAX * 0.9,
         -DBL_MAX * 0.9,
         -DBL_MAX * 0.9,
         DBL_MAX * 0.9};
   const double weights[]{1, 2, 3, 4};
   double impurities[4 * cClasses];
   double residualIntercept[cClasses];

   ErrorEbm error = Purify(k_tolerancePurityDefault,
         k_isRandomizedDefault,
         EBM_TRUE,
         cClasses,
         cDimensions,
         dimensionLengths,
         weights,
         scores,
         impurities,
         &residualIntercept[0]);
   CHECK(Error_None == error);
   // CHECK(0.0 != residualIntercept[0]);
   // CHECK(0.0 != residualIntercept[1]);
   //  const double impurity0 = MeasureImpurity(cClasses, 0, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity0 && impurity0 < 0.001);
   //  const double impurity1 = MeasureImpurity(cClasses, 1, cDimensions, dimensionLengths, weights, scores);
   //  CHECK(-0.001 < impurity1 && impurity1 < 0.001);
}
