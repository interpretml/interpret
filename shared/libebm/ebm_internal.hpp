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
#include "common_c.h"
#include "bridge_c.h"
#include "zones.h"

// TODO: try and remove the dependency on bridge_cpp.hpp and common_cpp.hpp in the future once all our templated performance stuff is in the compute zone
#include "common_cpp.hpp"
#include "bridge_cpp.hpp" // k_cBitsForStorageType

// TODO: try and remove EbmInternal.h from as many places as possible after we've transitioned most of this stuff into common_c.h and common_c.hpp

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<typename TTo, typename TFrom>
INLINE_ALWAYS static TTo SafeConvertFloat(const TFrom val) {
   // TODO: call this function from anywhere we convert floats of any kind

   // In some IEEE-754 compatible implementations, you can get a float exception when converting a float64
   // to a float32 when the result overflows to +-inf or (maybe underflow?).  This is likely to be a compiler
   // specific thing and controlled by the floating point unit flags.  For now let's just do the conversion
   // and if it becomes a problem in the future with over/under flow then explicitly check the range before
   // making it a +-inf manually.  The caller can check the results for +-inf.

   // TODO: If we end up manually checking bounds, then change this so that it doesn't do any checks if TFrom is a float64 or if both are float32
   // TODO: test the overflow/underflow scenario on our compilers to see if they behave well, and check documentation

   return static_cast<TTo>(val);
}

// TODO: put a list of all the epilon constants that we use here throughout (use 1e-7 format).  Make it a percentage based on the data type 
//   minimum eplison from 1 + minimal_change.  If we can make it a constant, then do that, or make it a percentage of a dynamically detected/changing value.  
//   Perhaps take the sqrt of the minimal change from 1?
// when comparing floating point numbers, check this info out: https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/


// TODO: search on all my epsilon values and see if they are being used consistently

// gain should be positive, so any number is essentially illegal, but let's make our number very very negative so that we can't confuse it with small 
// negative values close to zero that might occur due to numeric instability
static constexpr FloatBig k_illegalGainFloat = std::numeric_limits<FloatBig>::lowest();
static constexpr double k_illegalGainDouble = std::numeric_limits<double>::lowest();
static constexpr FloatBig k_epsilonNegativeGainAllowed = FloatBig { -1e-7 };

// there doesn't seem to be a reasonable upper bound for how high you can set the k_cCompilerClassesMax value.  The bottleneck seems to be 
// that setting it too high increases compile time and module size
// this is how much the runtime speeds up if you compile it with hard coded vector sizes
// 200 => 2.65%
// 32  => 3.28%
// 16  => 5.12%
// 8   => 5.34%
// 4   => 8.31%
// TODO: increase this up to something like 16.  I have decreased it to 8 in order to make compiling more efficient, and so that I regularily test the 
//   runtime looped version of our code
static constexpr size_t k_cCompilerScoresMax = 8;
static constexpr size_t k_cCompilerScoresStart = 3;

static constexpr size_t k_cCompilerOptimizedCountDimensionsMax = 3;

static_assert(1 <= k_cCompilerOptimizedCountDimensionsMax,
   "k_cCompilerOptimizedCountDimensionsMax can be 1 if we want to turn off dimension optimization, but 0 or less is disallowed.");
static_assert(k_cCompilerOptimizedCountDimensionsMax <= k_cDimensionsMax,
   "k_cCompilerOptimizedCountDimensionsMax cannot be larger than the maximum number of dimensions.");

static constexpr size_t k_dynamicDimensions = 0;

static constexpr bool k_bUseLogitboost = false;

template<typename TSum, typename TItem>
static TSum AddPositiveFloatsSafe(size_t cVals, const TItem * pVals) {
   // floats have 23 bits of mantissa, so if you add 2^23 of them, the average value is below the threshold where
   // it adds to the sum total value even by the smallest amount.  When that happens the sum stops advancing.
   // This function solves that problem by breaking the loop into 3 sections, which allows us to go back to zero where
   // floats have more resolution

   EBM_ASSERT(nullptr != pVals);
   TSum totalOuter = 0;
   while(size_t { 0 } != cVals) {
      TSum totalMid = 0;
      do {
         EBM_ASSERT(0 != cVals);
         const size_t cInner = ((cVals - 1) % k_cFloatSumLimit) + 1;
         cVals -= cInner;
         EBM_ASSERT(0 == cVals % k_cFloatSumLimit);
         const TItem * const pValsEnd = pVals + cInner;
         TSum totalInner = 0;
         do {
            const TItem val = *pVals;
            if(val < 0) {
               // we often sum up the results of this function, and NaN cannot go back to anything other than NaN
               return std::numeric_limits<TSum>::quiet_NaN();
            }
            totalInner += SafeConvertFloat<TSum>(val);
            ++pVals;
         } while(pValsEnd != pVals);
         totalMid += totalInner;
         EBM_ASSERT(0 == cVals % k_cFloatSumLimit);
      } while(size_t { 0 } != (cVals / k_cFloatSumLimit) % k_cFloatSumLimit);
      totalOuter += totalMid;
   }
   return totalOuter;
}

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
