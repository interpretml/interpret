// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef EBM_INTERNAL_HPP
#define EBM_INTERNAL_HPP

#include <inttypes.h>
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <type_traits> // is_integral
#include <stdlib.h> // free
#include <assert.h> // base assert
#include <string.h> // strcpy

#include "libebm.h"
#include "logging.h" // EBM_ASSERT
#include "unzoned.h"

#include "bridge.h"
#include "common.hpp"
#include "bridge.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

typedef size_t UIntSplit;
typedef uint64_t UIntMain;
typedef double FloatMain;
typedef double FloatCalc;
typedef double FloatScore;


// TODO: put a list of all the epilon constants that we use here throughout (use 1e-7 format).  Make it a percentage based on the data type 
//   minimum eplison from 1 + minimal_change.  If we can make it a constant, then do that, or make it a percentage of a dynamically detected/changing value.  
//   Perhaps take the sqrt of the minimal change from 1?
// when comparing floating point numbers, check this info out: https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/


// TODO: search on all my epsilon values and see if they are being used consistently

// gain should be positive, so any number is essentially illegal, but let's make our number very very negative so that we can't confuse it with small 
// negative values close to zero that might occur due to numeric instability
static constexpr FloatCalc k_illegalGainFloat = std::numeric_limits<FloatCalc>::lowest();
static constexpr double k_illegalGainDouble = std::numeric_limits<double>::lowest();

#ifndef NDEBUG
static constexpr FloatCalc k_epsilonNegativeGainAllowed = FloatCalc { -1e-7 };
#endif // NDEBUG

static constexpr bool k_bUseLogitboost = false;

extern double FloatTickIncrementInternal(double deprecisioned[1]) noexcept;
extern double FloatTickDecrementInternal(double deprecisioned[1]) noexcept;

INLINE_ALWAYS static double FloatTickIncrement(const double val) noexcept {
   // we use an array in the call to FloatTickIncrementInternal to chop off any extended precision bits that might be in the float
   double deprecisioned[1];
   deprecisioned[0] = val;
   return FloatTickIncrementInternal(deprecisioned);
}
INLINE_ALWAYS static double FloatTickDecrement(const double val) noexcept {
   // we use an array in the call to FloatTickDecrementInternal to chop off any extended precision bits that might be in the float
   double deprecisioned[1];
   deprecisioned[0] = val;
   return FloatTickDecrementInternal(deprecisioned);
}
// TODO: call this throughout our code to remove subnormals
INLINE_ALWAYS static double CleanFloat(const double val) noexcept {
   // we use an array in the call to CleanFloats to chop off any extended precision bits that might be in the float
   double deprecisioned[1];
   deprecisioned[0] = val;
   CleanFloats(1, deprecisioned);
   return deprecisioned[0];
}

} // DEFINED_ZONE_NAME

#endif // EBM_INTERNAL_HPP
