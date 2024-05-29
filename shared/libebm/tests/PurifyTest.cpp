// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch_test.hpp"

#include "libebm.h"
#include "libebm_test.hpp"

static constexpr TestPriority k_filePriority = TestPriority::Purify;

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

   ErrorEbm error = Purify(1e-6, cDimensions, dimensionLengths, weights, scores, impurities, &residualIntercept);
   CHECK(Error_None == error);
   CHECK(0.0 != residualIntercept);
}

TEST_CASE("Purify simple 3x3x3") {
   const IntEbm dimensionLengths[]{3, 3, 3};
   const size_t cDimensions = sizeof(dimensionLengths) / sizeof(dimensionLengths[0]);
   double scores[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};
   const double weights[]{
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};
   double impurities[27];
   double residualIntercept;

   ErrorEbm error = Purify(1e-6, cDimensions, dimensionLengths, weights, scores, impurities, &residualIntercept);
   CHECK(Error_None == error);
   CHECK(0.0 != residualIntercept);
}
