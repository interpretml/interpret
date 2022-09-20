// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef COMPUTE_HPP
#define COMPUTE_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "zones.h"

#include "bridge_cpp.hpp" // k_cBitsForStorageType

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

#ifdef __CUDACC__
#define GPU_GLOBAL            __global__
#define GPU_DEVICE            __device__
#define GPU_BOTH              __host__ __device__
#else
#define GPU_GLOBAL
#define GPU_DEVICE
#define GPU_BOTH
#endif

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

static constexpr ptrdiff_t k_cCompilerClassesMax2 = 8;
static constexpr ptrdiff_t k_cCompilerClassesStart2 = 3;

static_assert(
   2 <= k_cCompilerClassesMax2,
   "we special case binary classification to have only 1 output.  If we remove the compile time optimization for the binary class situation then we would "
   "output model files with two values instead of our special case 1");

static constexpr ptrdiff_t k_cItemsPerBitPackMax2 = ptrdiff_t { k_cBitsForStorageType };
static_assert(k_cItemsPerBitPackMax2 <= ptrdiff_t { k_cBitsForStorageType }, "k_cItemsPerBitPackMax too big");
static constexpr ptrdiff_t k_cItemsPerBitPackMin2 = ptrdiff_t { 1 };
static_assert(1 <= k_cItemsPerBitPackMin2 || (k_cItemsPerBitPackDynamic2 == k_cItemsPerBitPackMin2 && k_cItemsPerBitPackDynamic2 == k_cItemsPerBitPackMax2), "k_cItemsPerBitPackMin must be positive and can only be zero if both min and max are zero (which means we only use dynamic)");
static_assert(k_cItemsPerBitPackMin2 <= k_cItemsPerBitPackMax2, "bit pack max less than min");
static_assert(
   k_cItemsPerBitPackDynamic2 == k_cItemsPerBitPackMin2 ||
   k_cItemsPerBitPackMin2 ==
   ptrdiff_t { k_cBitsForStorageType } / (ptrdiff_t { k_cBitsForStorageType } / k_cItemsPerBitPackMin2),
   "k_cItemsPerBitPackMin needs to be on the progression series");
static_assert(
   k_cItemsPerBitPackDynamic2 == k_cItemsPerBitPackMax2 ||
   k_cItemsPerBitPackMax2 ==
   ptrdiff_t { k_cBitsForStorageType } / (ptrdiff_t { k_cBitsForStorageType } / k_cItemsPerBitPackMax2),
   "k_cItemsPerBitPackMax needs to be on the progression series");
// if we cover the entire range of possible bit packing, then we don't need the dynamic case!
static constexpr ptrdiff_t k_cItemsPerBitPackLast = (ptrdiff_t { k_cBitsForStorageType } == k_cItemsPerBitPackMax2 &&
   ptrdiff_t { 1 } == k_cItemsPerBitPackMin2) ? ptrdiff_t { 1 } : k_cItemsPerBitPackDynamic2;
inline constexpr static ptrdiff_t GetNextBitPack(const ptrdiff_t cItemsBitPackedPrev) noexcept {
   // for 64 bits, the progression is: 64,32,21,16,12,10,9,8,7,6,5,4,3,2,1,0 (optionaly),-1 (never occurs in this function)
   // [there are 15 of these + the dynamic case + onebin case]
   // for 32 bits, the progression is: 32,16,10,8,6,5,4,3,2,1,0 (optionaly),-1 (never occurs in this function)
   // [which are all included in 64 bits + the dynamic case + onebin case]
   // we can have bit packs of -1, but this function should never see that value
   // this function should also never see the dynamic value 0 because we should terminate the chain at that point
   return k_cItemsPerBitPackMin2 == cItemsBitPackedPrev ? k_cItemsPerBitPackDynamic2 :
      ptrdiff_t { k_cBitsForStorageType } / ((ptrdiff_t { k_cBitsForStorageType } / cItemsBitPackedPrev) + 1);
}

} // DEFINED_ZONE_NAME

#endif // COMPUTE_HPP
