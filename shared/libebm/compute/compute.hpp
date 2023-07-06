// Copyright (c) 2023 The InterpretML Contributors
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
#define GPU_COMPILE
#define GPU_GLOBAL            __global__
#define GPU_DEVICE            __device__
#define GPU_BOTH              __host__ __device__
#else
#define GPU_GLOBAL
#define GPU_DEVICE
#define GPU_BOTH
#endif

// 64 for k_cItemsPerBitPackMax is too big since it'll replicate the objectives 64 times, and then 32, 21, etc..
// 8 is nice for k_cItemsPerBitPackMax since 2^8 = 256 bins, which gets 8 items packed into each 64 bit number.
static constexpr ptrdiff_t k_cItemsPerBitPackMax = ptrdiff_t { 8 };
// 1 is too low for k_cItemsPerBitPackMin since nobody should have 2^64 bins. 4 is nice since it allows there
// to be 2^16 bins = 65,536 bins. 5 would only allow 2^12 bins = 4096 which someone might want to exceed.
static constexpr ptrdiff_t k_cItemsPerBitPackMin = ptrdiff_t { 4 };

static_assert(k_cItemsPerBitPackMax <= ptrdiff_t { k_cBitsForStorageType }, "k_cItemsPerBitPackMax too big");
static_assert(1 <= k_cItemsPerBitPackMin || (k_cItemsPerBitPackDynamic == k_cItemsPerBitPackMin && k_cItemsPerBitPackDynamic == k_cItemsPerBitPackMax), "k_cItemsPerBitPackMin must be positive and can only be zero if both min and max are zero (which means we only use dynamic)");
static_assert(k_cItemsPerBitPackMin <= k_cItemsPerBitPackMax, "bit pack max less than min");
static_assert(
   k_cItemsPerBitPackDynamic == k_cItemsPerBitPackMin ||
   k_cItemsPerBitPackMin ==
   ptrdiff_t { k_cBitsForStorageType } / (ptrdiff_t { k_cBitsForStorageType } / k_cItemsPerBitPackMin),
   "k_cItemsPerBitPackMin needs to be on the progression series");
static_assert(
   k_cItemsPerBitPackDynamic == k_cItemsPerBitPackMax ||
   k_cItemsPerBitPackMax ==
   ptrdiff_t { k_cBitsForStorageType } / (ptrdiff_t { k_cBitsForStorageType } / k_cItemsPerBitPackMax),
   "k_cItemsPerBitPackMax needs to be on the progression series");
// if we cover the entire range of possible bit packing, then we don't need the dynamic case!
static constexpr ptrdiff_t k_cItemsPerBitPackLast = (ptrdiff_t { k_cBitsForStorageType } == k_cItemsPerBitPackMax &&
   ptrdiff_t { 1 } == k_cItemsPerBitPackMin) ? ptrdiff_t { 1 } : k_cItemsPerBitPackDynamic;
inline constexpr static ptrdiff_t GetNextBitPack(const ptrdiff_t cItemsBitPackedPrev) noexcept {
   // for 64 bits, the progression is: 64,32,21,16,12,10,9,8,7,6,5,4,3,2,1,0 (optionaly),-1 (never occurs in this function)
   // [there are 15 of these + the dynamic case + onebin case]
   // for 32 bits, the progression is: 32,16,10,8,6,5,4,3,2,1,0 (optionaly),-1 (never occurs in this function)
   // [which are all included in 64 bits + the dynamic case + onebin case]
   // we can have bit packs of -1, but this function should never see that value
   // this function should also never see the dynamic value 0 because we should terminate the chain at that point
   return k_cItemsPerBitPackMin == cItemsBitPackedPrev ? k_cItemsPerBitPackDynamic :
      ptrdiff_t { k_cBitsForStorageType } / ((ptrdiff_t { k_cBitsForStorageType } / cItemsBitPackedPrev) + 1);
}

template<typename T, typename U, U multiplicator, int shiftEnd, int shift>
struct Multiplier final {
   GPU_DEVICE inline constexpr static T Multiply(const T val) {
      return (U { 0 } != (multiplicator & (U { 1 } << shift)) ? (val << shift) : T { 0 }) + Multiplier<T, U, multiplicator, shiftEnd, shift + 1>::Multiply(val);
   }
};
template<typename T, typename U, U multiplicator, int shiftEnd>
struct Multiplier<T, U, multiplicator, shiftEnd, shiftEnd> final {
   GPU_DEVICE inline constexpr static T Multiply(const T) {
      return T { 0 };
   }
};
template<typename T, typename U, U multiplicator, int shiftEnd = CountBitsRequiredPositiveMax<U>()>
GPU_DEVICE inline constexpr static T Multiply(const T val) {
   return Multiplier<T, U, multiplicator, shiftEnd, 0>::Multiply(val);
}
template<typename T, typename U, bool bCompileTime, U multiplicator, int shiftEnd = CountBitsRequiredPositiveMax<U>(), typename std::enable_if<bCompileTime, void>::type * = nullptr>
GPU_DEVICE inline constexpr static T Multiply(const T val, const U) {
   return Multiply<T, U, multiplicator, shiftEnd>(val);
}
template<typename T, typename U, bool bCompileTime, U multiplicator, int shiftEnd = CountBitsRequiredPositiveMax<U>(), typename std::enable_if<!bCompileTime, void>::type * = nullptr>
GPU_DEVICE inline constexpr static T Multiply(const T val, const U runtimeMultiplicator) {
   return val * runtimeMultiplicator;
}

static_assert(Multiply<uint32_t, uint32_t, true, 0>(7, 0) == uint32_t { 0 }, "failed Multiply");
static_assert(Multiply<uint32_t, uint32_t, true, 7>(0, 0) == uint32_t { 0 }, "failed Multiply");
static_assert(Multiply<uint32_t, uint32_t, true, 1>(7, 0) == uint32_t { 7 }, "failed Multiply");
static_assert(Multiply<uint32_t, uint32_t, true, 7>(1, 0) == uint32_t { 7 }, "failed Multiply");
static_assert(Multiply<uint32_t, uint32_t, true, 65280>(65536, 0) == uint32_t { 4278190080 }, "failed Multiply");
static_assert(Multiply<uint32_t, uint32_t, true, 65536>(65280, 0) == uint32_t { 4278190080 }, "failed Multiply");
static_assert(Multiply<uint32_t, uint32_t, true, 4294967295>(1, 0) == uint32_t { 4294967295 }, "failed Multiply");
static_assert(Multiply<uint32_t, uint32_t, true, 1>(4294967295, 0) == uint32_t { 4294967295 }, "failed Multiply");

} // DEFINED_ZONE_NAME

#endif // COMPUTE_HPP
