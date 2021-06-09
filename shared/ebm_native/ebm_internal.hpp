// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef EBM_INTERNAL_H
#define EBM_INTERNAL_H

#include <inttypes.h>
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <type_traits> // is_integral
#include <stdlib.h> // free
#include <assert.h> // base assert
#include <string.h> // strcpy

#include "ebm_native.h"
#include "common_c.h"
#include "bridge_c.h"
#include "zones.h"

// TODO: try and remove the dependency on bridge_cpp.hpp and common_cpp.hpp in the future once all our templated performance stuff is in the compute zone
#include "common_cpp.hpp"
#include "bridge_cpp.hpp"

// TODO: try and remove EbmInternal.h from as many places as possible after we've transitioned most of this stuff into common_c.h and common_c.hpp

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// TODO: put a list of all the epilon constants that we use here throughout (use 1e-7 format).  Make it a percentage based on the FloatEbmType data type 
//   minimum eplison from 1 + minimal_change.  If we can make it a constant, then do that, or make it a percentage of a dynamically detected/changing value.  
//   Perhaps take the sqrt of the minimal change from 1?
// when comparing floating point numbers, check this info out: https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/


// TODO: search on all my epsilon values and see if they are being used consistently

// gain should be positive, so any number is essentially illegal, but let's make our number very very negative so that we can't confuse it with small 
// negative values close to zero that might occur due to numeric instability
constexpr static FloatEbmType k_illegalGain = std::numeric_limits<FloatEbmType>::lowest();
constexpr static FloatEbmType k_epsilonNegativeGainAllowed = -1e-7;
constexpr static FloatEbmType k_epsilonNegativeValidationMetricAllowed = -1e-7;
constexpr static FloatEbmType k_epsilonGradient = 1e-7;
#if defined(FAST_EXP) || defined(FAST_LOG)
// with the approximate exp function we can expect a bit of noise.  We might need to increase this further
constexpr static FloatEbmType k_epsilonGradientForBinaryToMulticlass = 1e-1;
#else // defined(FAST_EXP) || defined(FAST_LOG)
constexpr static FloatEbmType k_epsilonGradientForBinaryToMulticlass = 1e-7;
#endif // defined(FAST_EXP) || defined(FAST_LOG)
constexpr static FloatEbmType k_epsilonLogLoss = 1e-7;

// there doesn't seem to be a reasonable upper bound for how high you can set the k_cCompilerOptimizedTargetClassesMax value.  The bottleneck seems to be 
// that setting it too high increases compile time and module size
// this is how much the runtime speeds up if you compile it with hard coded vector sizes
// 200 => 2.65%
// 32  => 3.28%
// 16  => 5.12%
// 8   => 5.34%
// 4   => 8.31%
// TODO: increase this up to something like 16.  I have decreased it to 8 in order to make compiling more efficient, and so that I regularily test the 
//   runtime looped version of our code

constexpr static ptrdiff_t k_cCompilerOptimizedTargetClassesMax = 8;
constexpr static ptrdiff_t k_cCompilerOptimizedTargetClassesStart = 3;

static_assert(
   2 <= k_cCompilerOptimizedTargetClassesMax,
   "we special case binary classification to have only 1 output.  If we remove the compile time optimization for the binary class situation then we would "
   "output model files with two values instead of our special case 1");

constexpr static size_t k_cCompilerOptimizedCountDimensionsMax = 2;

static_assert(1 <= k_cCompilerOptimizedCountDimensionsMax,
   "k_cCompilerOptimizedCountDimensionsMax can be 1 if we want to turn off dimension optimization, but 0 or less is disallowed.");
static_assert(k_cCompilerOptimizedCountDimensionsMax <= k_cDimensionsMax,
   "k_cCompilerOptimizedCountDimensionsMax cannot be larger than the maximum number of dimensions.");

constexpr static size_t k_dynamicDimensions = 0;

#ifndef TODO_remove_this
constexpr static size_t k_cItemsPerBitPackDynamic = 0;
constexpr static size_t k_cItemsPerBitPackMax = 0; // if there are more than 16 (4 bits), then we should just use a loop since the code will be pretty big
constexpr static size_t k_cItemsPerBitPackMin = 0; // our default binning leads us to 256 values, which is 8 units per 64-bit data pack
INLINE_ALWAYS constexpr static size_t GetNextCountItemsBitPacked(const size_t cItemsBitPackedPrev) noexcept {
   // for 64 bits, the progression is: 64,32,21,16, 12,10,9,8,7,6,5,4,3,2,1 [there are 15 of these]
   // for 32 bits, the progression is: 32,16,10,8,6,5,4,3,2,1 [which are all included in 64 bits]
   return k_cItemsPerBitPackMin == cItemsBitPackedPrev ?
      k_cItemsPerBitPackDynamic : k_cBitsForStorageType / ((k_cBitsForStorageType / cItemsBitPackedPrev) + 1);
}
#endif




// TODO: figure out if we really want/need to template the handling of different bit packing sizes.  It might
//       be the case that for specific bit sizes, like 8x8, we want to keep our memory stride as small as possible
//       but we might also find that we can apply SIMD at the outer loop level in the places where we use bit
//       packing, so we'd load eight 64-bit numbers at a time and then keep all the interior loops.  In this case
//       the only penalty would be one branch mispredict, but we'd be able to loop over 8 bit extractions at a time
//       We might also pay a penalty if our stride length for the outputs is too long, but we'll have to test that
constexpr static bool k_bUseSIMD = false;
constexpr static bool k_bUseLogitboost = false;

template<typename T>
static T AddPositiveFloatsSafe(size_t cVals, const T * pVals) {
   // floats have 23 bits of mantissa, so if you add 2^23 of them, the average value is below the threshold where
   // it adds to the sum total value even by the smallest amount.  When that happens the sum stops advancing.
   // This function solves that problem by breaking the loop into 3 sections, which allows us to go back to zero where
   // floats have more resolution

   EBM_ASSERT(nullptr != pVals);
   T totalOuter = T { 0 };
   while(size_t { 0 } != cVals) {
      T totalMid = T { 0 };
      do {
         EBM_ASSERT(0 != cVals);
         const size_t cInner = ((cVals - 1) % k_cFloatSumLimit) + 1;
         cVals -= cInner;
         EBM_ASSERT(0 == cVals % k_cFloatSumLimit);
         const T * const pValsEnd = pVals + cInner;
         T totalInner = T { 0 };
         do {
            const T val = *pVals;
            if(val < T { 0 }) {
               return std::numeric_limits<T>::lowest();
            }
            totalInner += val;
            ++pVals;
         } while(pValsEnd != pVals);
         totalMid += totalInner;
         EBM_ASSERT(0 == cVals % k_cFloatSumLimit);
      } while(size_t { 0 } != (cVals / k_cFloatSumLimit) % k_cFloatSumLimit);
      totalOuter += totalMid;
   }
   return totalOuter;
}

template<typename T>
static bool CheckAllWeightsEqual(const size_t cWeights, const T * pWeights) {
   EBM_ASSERT(0 != cWeights);
   EBM_ASSERT(nullptr != pWeights);
   const T firstWeight = *pWeights;
   const T * const pWeightsEnd = pWeights + cWeights;
   do {
      if(UNLIKELY(firstWeight != *pWeights)) {
         // if firstWeight or *pWeight is NaN this should trigger, which is good since we don't want to
         // replace arrays containing all NaN weights with weights of 1
         return false;
      }
      ++pWeights;
   } while(LIKELY(pWeightsEnd != pWeights));
   return true;
}


//#define ZERO_FIRST_MULTICLASS_LOGIT

} // DEFINED_ZONE_NAME

#endif // EBM_INTERNAL_H
