// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef COMPUTE_HPP
#define COMPUTE_HPP

#include "bridge.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME


template<typename T>
inline constexpr static int GetNextBitPack(const int cItemsBitPackedPrev, const int cItemsPerBitPackMin) noexcept {
   // for 64 bits, the progression is: 64,32,21,16,12,10,9,8,7,6,5,4,3,2,1,0 (dynamic). -1 never occurs in this function
   // [there are 15 of these + the dynamic case + onebin case]
   // for 32 bits, the progression is: 32,16,10,8,6,5,4,3,2,1,0 (dynamic). -1 never occurs in this function
   // [which are all included in 64 bits + the dynamic case + onebin case]
   // we can have bit packs of -1, but this function should never see that value
   // this function should also never see the dynamic value 0 because we should terminate the chain at that point
   return COUNT_BITS(T) / ((COUNT_BITS(T) / cItemsBitPackedPrev) + 1) < cItemsPerBitPackMin ? k_cItemsPerBitPackDynamic :
      COUNT_BITS(T) / ((COUNT_BITS(T) / cItemsBitPackedPrev) + 1);
}

template<typename T>
inline constexpr static int GetFirstBitPack(int cItemsPerBitPackMax, const int cItemsPerBitPackMin) noexcept {
   return GetNextBitPack<T>(cItemsPerBitPackMax + 1, cItemsPerBitPackMin);
}

template<typename T, typename U, U multiplicator, int shiftEnd, int shift>
struct MultiplierInternal final {
   GPU_DEVICE inline constexpr static T Func(const T val) {
      return (U { 0 } != (multiplicator & (U { 1 } << shift)) ? (val << shift) : T { 0 }) + MultiplierInternal<T, U, multiplicator, shiftEnd, shift + 1>::Func(val);
   }
};
template<typename T, typename U, U multiplicator, int shiftEnd>
struct MultiplierInternal<T, U, multiplicator, shiftEnd, shiftEnd> final {
   GPU_DEVICE inline constexpr static T Func(const T) {
      return T { 0 };
   }
};
template<typename T, typename U, U multiplicator>
GPU_DEVICE inline constexpr static T MultiplyConst(const T val) {
   // Normally the compiler does a better job at choosing between multiplication or shifting, but it doesn't when
   // T is a SIMD packed datatype. Some SIMD implementation do not have a scalar multiply option in which case
   // this function will be a lot faster than the default of unpacking the SIMD type and multiplying the components.
   // And even if the SIMD implemention has a scalar multiply I think this function will be faster for multiplications
   // with less than 4 bits anywhere in multiplicator, which should most be the case for where we use it.
   return MultiplierInternal<T, U, multiplicator, COUNT_BITS(U), 0>::Func(val);
}
template<typename T, typename U, bool bCompileTime, U multiplicator>
GPU_DEVICE inline constexpr static typename std::enable_if<bCompileTime, T>::type Multiply(const T val, const U) {
   return MultiplyConst<T, U, multiplicator>(val);
}
template<typename T, typename U, bool bCompileTime, U multiplicator>
GPU_DEVICE inline constexpr static typename std::enable_if<!bCompileTime, T>::type Multiply(const T val, const U runtimeMultiplicator) {
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
static_assert(Multiply<uint32_t, uint32_t, true, 2>(4294967295, 0) == uint32_t { 4294967294 }, "failed Multiply");
static_assert(Multiply<uint64_t, uint64_t, true, 2>(4294967295, 0) == uint64_t { 8589934590 }, "failed Multiply");
static_assert(Multiply<uint32_t, uint32_t, true, 4294967295>(2, 0) == uint32_t { 4294967294 }, "failed Multiply");
static_assert(Multiply<uint64_t, uint64_t, true, 4294967295>(2, 0) == uint64_t { 8589934590 }, "failed Multiply");


} // DEFINED_ZONE_NAME

#endif // COMPUTE_HPP
