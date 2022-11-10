// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef COMMON_CPP_HPP
#define COMMON_CPP_HPP

#include <limits> // numeric_limits
#include <type_traits> // std::is_integral, std::enable_if, std::is_signed
#include <stddef.h> // size_t, ptrdiff_t
#include <stdlib.h> // malloc

#include "zones.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// using inlining makes it much harder to debug inline functions (stepping and breakpoints don't work).  
// In debug builds we don't care as much about speed, but we do care about debugability, so we generally 
// want to turn off inlining in debug mode.  BUT, when I make everything non-inlined some trivial wrapper
// functions cause a big slowdown, so we'd rather have two classes of inlining.  The INLINE_ALWAYS
// version that inlines in debug mode, and the INLINE_RELEASE version that only inlines for release builds.  
// BUT, unfortunately, inline functions need to be in headers generally, but if you remove the inline, 
// then you get name collisions on the functions.  Using static is one possible solution, but it can create 
// duplicate copies of the function inside each module that the header is inlucded within if the linker 
// isn't smart.  Another option is to use a dummy template, which forces the compiler to allow 
// definition in a header but combines them afterwards. Lastly, using the non-forced inline works in most 
// cases since the compiler will not inline complicated functions by default.
#ifdef NDEBUG
#define INLINE_RELEASE_UNTEMPLATED INLINE_ALWAYS
#define INLINE_RELEASE_TEMPLATED INLINE_ALWAYS
#else //NDEBUG
#define INLINE_RELEASE_UNTEMPLATED template<bool bUnusedInline = false>
#define INLINE_RELEASE_TEMPLATED
#endif //NDEBUG

// The C++ standard makes it undefined behavior to access memory past the end of an array with a declared length.
// So, without mitigation, the struct hack would be undefined behavior.  We can however formally turn an array 
// into a pointer, thus making our modified struct hack completely legal in C++.  So, for instance, the following
// is illegal in C++:
//
// struct MyStruct { int myInt[1]; };
// MyStruct * pMyStruct = malloc(sizeof(MyStruct) + sizeof(int));
// "pMyStruct->myInt[1] = 3;" 
// 
// Compilers have been getting agressive in using undefined behavior to optimize code, so even though the struct
// hack is still widely used, we don't want to risk invoking undefined behavior. By converting an array 
// into a pointer though with the ArrayToPointer function below, we can make this legal again by always writing: 
//
// "ArrayToPointer(pMyStruct->myInt)[1] = 3;"
//
// I've seen a lot of speculation on the internet that the struct hack is always illegal, but I believe this is
// incorrect using this modified access method.  To illustrate, everything in this example should be completely legal:
//
// struct MyStruct { int myInt[1]; };
// char * pMem = malloc(sizeof(MyStruct) + sizeof(int));
// size_t myOffset = offsetof(MyStruct, myInt);
// int * pInt = reinterpret_cast<int *>(pMem + myOffset);
// pInt[1] = 3;
//
// We endure all this hassle because in a number of places we co-locate memory for performance reasons.  We do allocate 
// sufficient memory for doing this, and we also statically check that our structures are standard layout structures, 
// which is required in order to use the offsetof macro, or in our case array to pointer conversion.
// 
template<typename T>
inline static T * ArrayToPointer(T * const a) noexcept {
   return a;
}
template<typename T>
inline static const T * ArrayToPointer(const T * const a) noexcept {
   return a;
}

template<typename T>
inline static T * IndexByte(T * const p, const size_t iByte) noexcept {
   return reinterpret_cast<T *>(reinterpret_cast<char *>(p) + iByte);
}
template<typename T>
inline static const T * IndexByte(const T * const p, const size_t iByte) noexcept {
   return reinterpret_cast<const T *>(reinterpret_cast<const char *>(p) + iByte);
}

template<typename T>
inline static T * NegativeIndexByte(T * const p, const size_t iByte) noexcept {
   return reinterpret_cast<T *>(reinterpret_cast<char *>(p) - iByte);
}
template<typename T>
inline static const T * NegativeIndexByte(const T * const p, const size_t iByte) noexcept {
   return reinterpret_cast<const T *>(reinterpret_cast<const char *>(p) - iByte);
}

template<typename T>
inline static size_t CountBytes(const T * const pHigh, const T * const pLow) noexcept {
   EBM_ASSERT(pLow <= pHigh);
   return reinterpret_cast<const char *>(pHigh) - reinterpret_cast<const char *>(pLow);
}

// TODO : replace all std::min and std::max and similar comparions that get the min/max with this function
// unlike std::min/std::max, our version has explicit noexcept semantics, constexpr static, and is variadic
template<typename T>
inline constexpr static T EbmMin(T v1, T v2) noexcept {
   return v1 < v2 ? v1 : v2;
}
template<typename T>
inline constexpr static T EbmMax(T v1, T v2) noexcept {
   return v1 < v2 ? v2 : v1;
}

template<typename T, typename... Args>
inline constexpr static T EbmMin(const T v1, const T v2, const Args...args) noexcept {
   return EbmMin(EbmMin(v1, v2), args...);
}

template<typename T, typename... Args>
inline constexpr static T EbmMax(const T v1, const T v2, const Args...args) noexcept {
   return EbmMax(EbmMax(v1, v2), args...);
}

static_assert(EbmMin(1.25, 2.5) == 1.25, "automated test with compiler");
static_assert(EbmMin(2.5, 1.25) == 1.25, "automated test with compiler");

static_assert(EbmMax(1.25, 2.5) == 2.5, "automated test with compiler");
static_assert(EbmMax(2.5, 1.25) == 2.5, "automated test with compiler");

static_assert(EbmMin(1.25, 2.5, 3.75) == 1.25, "automated test with compiler");
static_assert(EbmMin(2.5, 1.25, 3.75) == 1.25, "automated test with compiler");
static_assert(EbmMin(3.75, 2.5, 1.25) == 1.25, "automated test with compiler");

static_assert(EbmMax(3.75, 2.5, 1.25) == 3.75, "automated test with compiler");
static_assert(EbmMax(2.5, 3.75, 1.25) == 3.75, "automated test with compiler");
static_assert(EbmMax(1.25, 2.5, 3.75) == 3.75, "automated test with compiler");

template<typename T>
inline constexpr static T EbmAbs(T v) noexcept {
   return T { 0 } <= v ? v : -v;
}

template<typename T>
inline static bool IsClose(
   T v1,
   T v2,
   T threshold = T { 1e-10 },
   T additiveError = T { 1e-15 },
   T multipleError = T { 0.9999 }
) noexcept {
   //EBM_ASSERT(T { 0 } < threshold);
   //EBM_ASSERT(T { 0 } < additiveError);
   //EBM_ASSERT(additiveError < threshold);
   //EBM_ASSERT(T { 0 } < multipleError);
   //EBM_ASSERT(multipleError < T { 1 });

   if(threshold <= v1) {
      if(v1 < v2) {
         return v2 * multipleError <= v1;
      } else {
         return v1 * multipleError <= v2;
      }
   } else if(threshold <= v2) {
      return v2 * multipleError <= v1;
   } else if(v1 <= -threshold) {
      if(v2 < v1) {
         return v1 <= v2 * multipleError;
      } else {
         return v2 <= v1 * multipleError;
      }
   } else if(v2 <= -threshold) {
      return v1 <= v2 * multipleError;
   } else {
      // the absolute value of both are below threshold, so use additive error
      return EbmAbs(v2 - v1) <= additiveError;
   }
}


// use SFINAE to compile time specialize IsConvertError
// https://www.fluentcpp.com/2019/08/23/how-to-make-sfinae-pretty-and-robust/
//
// the general rules of conversion are as follows:
// calling std::numeric_limits<?>::max() returns an item of that type
// casting and comparing will never give us undefined behavior.  It can give us implementation defined behavior or unspecified behavior, which is legal.
// Undefined behavior results from overflowing negative integers, but we don't add or subtract.
// C/C++ uses value preserving instead of sign preserving.  Generally, if you have two integer numbers that you're comparing then if one type can be 
// converted into the other with no loss in range then that the smaller range integer is converted into the bigger range integer
// if one type can't cover the entire range of the other, then items are converted to UNSIGNED values.  This is probably the most dangerous 
// thing for us to deal with

template<typename TTo, typename TFrom>
inline constexpr static typename std::enable_if<
   std::is_signed<TTo>::value && std::is_signed<TFrom>::value &&
   std::numeric_limits<TTo>::lowest() <= std::numeric_limits<TFrom>::lowest() &&
   std::numeric_limits<TFrom>::max() <= std::numeric_limits<TTo>::max()
   , bool>::type IsConvertError(const TFrom number) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(std::numeric_limits<TTo>::lowest() < 0, "TTo::lowest must be negative");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo::max must be positive");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(std::numeric_limits<TFrom>::lowest() < 0, "TFrom::lowest must be negative");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom::max must be positive");

   static_assert(std::is_same<const TFrom, decltype(number)>::value,
      "this is a stupid check to access the number variable to avoid a compiler warning");

   return false;
}

static_assert(!IsConvertError<int32_t>(int16_t { 32767 }), "automated test with compiler");
static_assert(!IsConvertError<int32_t>(int16_t { 0 }), "automated test with compiler");
static_assert(!IsConvertError<int32_t>(int16_t { -32768 }), "automated test with compiler");
static_assert(!IsConvertError<int16_t>(int16_t { 32767 }), "automated test with compiler");
static_assert(!IsConvertError<int16_t>(int16_t { 0 }), "automated test with compiler");
static_assert(!IsConvertError<int16_t>(int16_t { -32768 }), "automated test with compiler");

template<typename TTo, typename TFrom>
inline constexpr static typename std::enable_if<
   std::is_signed<TTo>::value && std::is_signed<TFrom>::value &&
   !(std::numeric_limits<TTo>::lowest() <= std::numeric_limits<TFrom>::lowest() &&
      std::numeric_limits<TFrom>::max() <= std::numeric_limits<TTo>::max())
   , bool>::type IsConvertError(const TFrom number) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(std::numeric_limits<TTo>::lowest() < 0, "TTo::lowest must be negative");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo::max must be positive");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(std::numeric_limits<TFrom>::lowest() < 0, "TFrom::lowest must be negative");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom::max must be positive");

   static_assert(
      std::numeric_limits<TFrom>::lowest() <= std::numeric_limits<TTo>::lowest() &&
      std::numeric_limits<TTo>::max() <= std::numeric_limits<TFrom>::max(),
      "we have a specialization for when TTo has a larger range, but if TFrom is larger then check that it's larger on both the upper and lower ends"
      );

   return number < TFrom { std::numeric_limits<TTo>::lowest() } || TFrom { std::numeric_limits<TTo>::max() } < number;
}

static_assert(IsConvertError<int8_t>(int16_t { -129 }), "automated test with compiler");
static_assert(!IsConvertError<int8_t>(int16_t { -128 }), "automated test with compiler");
static_assert(!IsConvertError<int8_t>(int16_t { -1 }), "automated test with compiler");
static_assert(!IsConvertError<int8_t>(int16_t { 0 }), "automated test with compiler");
static_assert(!IsConvertError<int8_t>(int16_t { 1 }), "automated test with compiler");
static_assert(!IsConvertError<int8_t>(int16_t { 127 }), "automated test with compiler");
static_assert(IsConvertError<int8_t>(int16_t { 128 }), "automated test with compiler");

template<typename TTo, typename TFrom>
inline constexpr static typename std::enable_if<
   !std::is_signed<TTo>::value && std::is_signed<TFrom>::value &&
   std::numeric_limits<TFrom>::max() <= std::numeric_limits<TTo>::max()
   , bool>::type IsConvertError(const TFrom number) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(0 == std::numeric_limits<TTo>::lowest(), "TTo::lowest must be zero");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo::max must be positive");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(std::numeric_limits<TFrom>::lowest() < 0, "TFrom::lowest must be negative");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom::max must be positive");

   return number < TFrom { 0 };
}

static_assert(!IsConvertError<uint32_t>(int16_t { 32767 }), "automated test with compiler");
static_assert(!IsConvertError<uint32_t>(int16_t { 0 }), "automated test with compiler");
static_assert(IsConvertError<uint32_t>(int16_t { -32768 }), "automated test with compiler");
static_assert(!IsConvertError<uint16_t>(int16_t { 32767 }), "automated test with compiler");
static_assert(!IsConvertError<uint16_t>(int16_t { 0 }), "automated test with compiler");
static_assert(IsConvertError<uint16_t>(int16_t { -32768 }), "automated test with compiler");

template<typename TTo, typename TFrom>
inline constexpr static typename std::enable_if <
   !std::is_signed<TTo>::value && std::is_signed<TFrom>::value &&
   std::numeric_limits<TTo>::max() < std::numeric_limits<TFrom>::max()
   , bool>::type IsConvertError(const TFrom number) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(0 == std::numeric_limits<TTo>::lowest(), "TTo::lowest must be zero");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo::max must be positive");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(std::numeric_limits<TFrom>::lowest() < 0, "TFrom::lowest must be negative");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom::max must be positive");

   return number < TFrom { 0 } || TFrom { std::numeric_limits<TTo>::max() } < number;
}

static_assert(IsConvertError<uint8_t>(int16_t { -32768 }), "automated test with compiler");
static_assert(IsConvertError<uint8_t>(int16_t { -1 }), "automated test with compiler");
static_assert(!IsConvertError<uint8_t>(int16_t { 0 }), "automated test with compiler");
static_assert(!IsConvertError<uint8_t>(int16_t { 255 }), "automated test with compiler");
static_assert(IsConvertError<uint8_t>(int16_t { 256 }), "automated test with compiler");
static_assert(IsConvertError<uint8_t>(int16_t { 32767 }), "automated test with compiler");

template<typename TTo, typename TFrom>
inline constexpr static typename std::enable_if<
   std::is_signed<TTo>::value && !std::is_signed<TFrom>::value &&
   std::numeric_limits<TFrom>::max() <= std::numeric_limits<TTo>::max()
   , bool>::type IsConvertError(const TFrom number) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(std::numeric_limits<TTo>::lowest() < 0, "TTo::lowest must be negative");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo::max must be positive");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(0 == std::numeric_limits<TFrom>::lowest(), "TFrom::lowest must be zero");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom::max must be positive");

   static_assert(std::is_same<const TFrom, decltype(number)>::value,
      "this is a stupid check to access the number variable to avoid a compiler warning");

   return false;
}

static_assert(!IsConvertError<int32_t>(uint16_t { 65535 }), "automated test with compiler");
static_assert(!IsConvertError<int32_t>(uint16_t { 32767 }), "automated test with compiler");
static_assert(!IsConvertError<int32_t>(uint16_t { 0 }), "automated test with compiler");

template<typename TTo, typename TFrom>
inline constexpr static typename std::enable_if <
   std::is_signed<TTo>::value && !std::is_signed<TFrom>::value &&
   std::numeric_limits<TTo>::max() < std::numeric_limits<TFrom>::max()
   , bool>::type IsConvertError(const TFrom number) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(std::numeric_limits<TTo>::lowest() < 0, "TTo::lowest must be negative");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo::max must be positive");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(0 == std::numeric_limits<TFrom>::lowest(), "TFrom::lowest must be zero");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom::max must be positive");

   return TFrom { std::numeric_limits<TTo>::max() } < number;
}

static_assert(IsConvertError<int16_t>(uint16_t { 65535 }), "automated test with compiler");
static_assert(IsConvertError<int16_t>(uint16_t { 32768 }), "automated test with compiler");
static_assert(!IsConvertError<int16_t>(uint16_t { 32767 }), "automated test with compiler");
static_assert(!IsConvertError<int16_t>(uint16_t { 0 }), "automated test with compiler");
static_assert(IsConvertError<int8_t>(uint16_t { 65535 }), "automated test with compiler");
static_assert(IsConvertError<int8_t>(uint16_t { 32768 }), "automated test with compiler");
static_assert(IsConvertError<int8_t>(uint16_t { 32767 }), "automated test with compiler");
static_assert(IsConvertError<int8_t>(uint16_t { 256 }), "automated test with compiler");
static_assert(IsConvertError<int8_t>(uint16_t { 255 }), "automated test with compiler");
static_assert(IsConvertError<int8_t>(uint16_t { 128 }), "automated test with compiler");
static_assert(!IsConvertError<int8_t>(uint16_t { 127 }), "automated test with compiler");
static_assert(!IsConvertError<int8_t>(uint16_t { 0 }), "automated test with compiler");

template<typename TTo, typename TFrom>
inline constexpr static typename std::enable_if<
   !std::is_signed<TTo>::value && !std::is_signed<TFrom>::value &&
   std::numeric_limits<TFrom>::max() <= std::numeric_limits<TTo>::max()
   , bool>::type IsConvertError(const TFrom number) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(0 == std::numeric_limits<TTo>::lowest(), "TTo::lowest must be zero");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo::max must be positive");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(0 == std::numeric_limits<TFrom>::lowest(), "TFrom::lowest must be zero");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom::max must be positive");

   static_assert(std::is_same<const TFrom, decltype(number)>::value,
      "this is a stupid check to access the number variable to avoid a compiler warning");

   return false;
}

static_assert(!IsConvertError<uint32_t>(uint16_t { 65535 }), "automated test with compiler");
static_assert(!IsConvertError<uint32_t>(uint16_t { 0 }), "automated test with compiler");
static_assert(!IsConvertError<uint16_t>(uint16_t { 65535 }), "automated test with compiler");
static_assert(!IsConvertError<uint16_t>(uint16_t { 0 }), "automated test with compiler");

template<typename TTo, typename TFrom>
inline constexpr static typename std::enable_if <
   !std::is_signed<TTo>::value && !std::is_signed<TFrom>::value &&
   std::numeric_limits<TTo>::max() < std::numeric_limits<TFrom>::max()
   , bool>::type IsConvertError(const TFrom number) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(0 == std::numeric_limits<TTo>::lowest(), "TTo::lowest must be zero");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo::max must be positive");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(0 == std::numeric_limits<TFrom>::lowest(), "TFrom::lowest must be zero");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom::max must be positive");

   return TFrom { std::numeric_limits<TTo>::max() } < number;
}

static_assert(IsConvertError<uint8_t>(uint16_t { 65535 }), "automated test with compiler");
static_assert(IsConvertError<uint8_t>(uint16_t { 256 }), "automated test with compiler");
static_assert(!IsConvertError<uint8_t>(uint16_t { 255 }), "automated test with compiler");
static_assert(!IsConvertError<uint8_t>(uint16_t { 0 }), "automated test with compiler");

template<typename T>
constexpr static size_t CountBitsRequired(const T maxValue) noexcept {
   // this is a bit inefficient when called in the runtime, but we don't call it anywhere that's important performance wise.
   return T { 0 } == maxValue ? size_t { 0 } : size_t { 1 } + CountBitsRequired<T>(maxValue / T { 2 });
}

template<typename T>
inline constexpr static size_t CountBitsRequiredPositiveMax() noexcept {
   return CountBitsRequired(std::numeric_limits<T>::max());
}
static_assert(CountBitsRequiredPositiveMax<uint8_t>() == 8, "automated test with compiler");
static_assert(CountBitsRequiredPositiveMax<uint16_t>() == 16, "automated test with compiler");
static_assert(CountBitsRequiredPositiveMax<uint32_t>() == 32, "automated test with compiler");
static_assert(CountBitsRequiredPositiveMax<uint64_t>() == 64, "automated test with compiler");
static_assert(CountBitsRequiredPositiveMax<int8_t>() == 7, "automated test with compiler");
static_assert(CountBitsRequiredPositiveMax<int16_t>() == 15, "automated test with compiler");
static_assert(CountBitsRequiredPositiveMax<int32_t>() == 31, "automated test with compiler");
static_assert(CountBitsRequiredPositiveMax<int64_t>() == 63, "automated test with compiler");

template<typename T>
inline constexpr static T MaxFromCountBits(const size_t cBits) noexcept {
   return 0 == cBits ? T { 0 } : (MaxFromCountBits<T>(cBits - 1) << 1 | T { 1 });
}
static_assert(MaxFromCountBits<uint8_t>(0) == 0, "automated test with compiler");
static_assert(MaxFromCountBits<uint8_t>(1) == 1, "automated test with compiler");
static_assert(MaxFromCountBits<uint8_t>(2) == 3, "automated test with compiler");
static_assert(MaxFromCountBits<uint8_t>(3) == 7, "automated test with compiler");
static_assert(MaxFromCountBits<uint8_t>(4) == 15, "automated test with compiler");
static_assert(MaxFromCountBits<uint8_t>(8) == 255, "automated test with compiler");
static_assert(MaxFromCountBits<uint64_t>(64) == uint64_t { 18446744073709551615u }, "automated test with compiler");

static constexpr size_t k_cBitsForSizeT = CountBitsRequiredPositiveMax<size_t>();

// It is impossible for us to have tensors with more than k_cDimensionsMax dimensions.  
// Our public C interface passes tensors back and forth with our caller with each dimension having
// a minimum of two bins, which only occurs for categoricals with just a missing and unknown bin.
// This should only occur for nominal cateogricals with only missing values.  Other feature types
// will have more bins, and thus will be restricted to even less dimensions. If all dimensions had the minimum 
// of two bins, we would need 2^N cells stored in the tensor.  If the tensor contained cells
// of 1 byte and filled all memory on a 64 bit machine, then we could not have more than 64 dimensions.
// On a real system, we can't fill all memory, and our interface requires tensors of double, so we  
// can subtract bits for the # of bytes used in a double and subtract 1 more because we cannot use all memory.
static constexpr size_t k_cDimensionsMax = k_cBitsForSizeT -
   CountBitsRequired(sizeof(double) / sizeof(uint8_t) - 1) - 1;
static_assert(k_cDimensionsMax < k_cBitsForSizeT, "reserve the highest bit for bit manipulation space");

template<typename T>
inline constexpr static bool IsMultiplyError(const T num1PreferredConstexpr, const T num2) noexcept {
   static_assert(std::is_integral<T>::value, "T must be integral");
   static_assert(std::numeric_limits<T>::is_specialized, "T must be specialized");
   static_assert(std::is_unsigned<T>::value, "T must be unsigned in the current implementation");

   // it will never overflow if num1 is zero or 1.  We need to check zero to avoid division by zero
   return T { 0 } != num1PreferredConstexpr && static_cast<T>(std::numeric_limits<T>::max() / num1PreferredConstexpr) < num2;
}

static_assert(!IsMultiplyError(uint8_t { 0 }, uint8_t { 0 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint8_t { 0 }, uint8_t { 1 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint8_t { 1 }, uint8_t { 0 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint8_t { 1 }, uint8_t { 1 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint8_t { 1 }, uint8_t { 255 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint8_t { 255 }, uint8_t { 1 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint8_t { 0 }, uint8_t { 2 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint8_t { 2 }, uint8_t { 0 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint8_t { 2 }, uint8_t { 2 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint8_t { 2 }, uint8_t { 127 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint8_t { 127 }, uint8_t { 2 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint8_t { 15 }, uint8_t { 17 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint8_t { 17 }, uint8_t { 15 }), "automated test with compiler");
static_assert(IsMultiplyError(uint8_t { 16 }, uint8_t { 16 }), "automated test with compiler");
static_assert(IsMultiplyError(uint8_t { 2 }, uint8_t { 128 }), "automated test with compiler");
static_assert(IsMultiplyError(uint8_t { 128 }, uint8_t { 2 }), "automated test with compiler");
static_assert(IsMultiplyError(uint32_t { 641 }, uint32_t { 6700417 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint32_t { 640 }, uint32_t { 6700417 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint32_t { 641 }, uint32_t { 6700416 }), "automated test with compiler");

template<typename T, typename... Args>
inline constexpr static bool IsMultiplyError(const T num1PreferredConstexpr, const T num2, const Args...args) noexcept {
   // we allow zeros in the parameters, but we report an error if there's an overflow before the 0 is reached
   // since multiplication will overflow if we proceed in the order specified by IsMultiplyError
   return IsMultiplyError(num1PreferredConstexpr, num2) || IsMultiplyError(static_cast<T>(num1PreferredConstexpr * num2), args...);
}

static_assert(!IsMultiplyError(uint8_t { 0 }, uint8_t { 0 }, uint8_t { 0 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint8_t { 0 }, uint8_t { 0 }, uint8_t { 0 }, uint8_t { 0 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint8_t { 1 }, uint8_t { 1 }, uint8_t { 1 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint8_t { 2 }, uint8_t { 2 }, uint8_t { 2 }, uint8_t { 2 }), "automated test with compiler");
static_assert(!IsMultiplyError(uint8_t { 17 }, uint8_t { 15 }, uint8_t { 1 }, uint8_t { 1 }), "automated test with compiler");
static_assert(IsMultiplyError(uint8_t { 17 }, uint8_t { 15 }, uint8_t { 2 }, uint8_t { 1 }), "automated test with compiler");

static_assert(IsMultiplyError(uint8_t { 16 }, uint8_t { 16 }, uint8_t { 0 }), "once we overflow we stay overflowed");
static_assert(!IsMultiplyError(uint8_t { 16 }, uint8_t { 0 }, uint8_t { 16 }), "we never reach overflow with this");


template<typename T>
inline constexpr static bool IsAddError(const T num1PreferredConstexpr, const T num2) noexcept {
   static_assert(std::is_integral<T>::value, "T must be integral");
   static_assert(std::numeric_limits<T>::is_specialized, "T must be specialized");
   static_assert(std::is_unsigned<T>::value, "T must be unsigned in the current implementation");

   // overflow for unsigned values is defined behavior in C++ and it causes a wrap arround
   return static_cast<T>(num1PreferredConstexpr + num2) < num1PreferredConstexpr;
}

static_assert(!IsAddError(uint8_t { 0 }, uint8_t { 0 }), "automated test with compiler");
static_assert(!IsAddError(uint8_t { 0 }, uint8_t { 255 }), "automated test with compiler");
static_assert(!IsAddError(uint8_t { 255 }, uint8_t { 0 }), "automated test with compiler");
static_assert(!IsAddError(uint8_t { 1 }, uint8_t { 254 }), "automated test with compiler");
static_assert(!IsAddError(uint8_t { 254 }, uint8_t { 1 }), "automated test with compiler");
static_assert(!IsAddError(uint8_t { 127 }, uint8_t { 128 }), "automated test with compiler");
static_assert(!IsAddError(uint8_t { 128 }, uint8_t { 127 }), "automated test with compiler");
static_assert(IsAddError(uint8_t { 1 }, uint8_t { 255 }), "automated test with compiler");
static_assert(IsAddError(uint8_t { 255 }, uint8_t { 1 }), "automated test with compiler");
static_assert(IsAddError(uint8_t { 2 }, uint8_t { 254 }), "automated test with compiler");
static_assert(IsAddError(uint8_t { 254 }, uint8_t { 2 }), "automated test with compiler");
static_assert(IsAddError(uint8_t { 128 }, uint8_t { 128 }), "automated test with compiler");
static_assert(IsAddError(uint8_t { 255 }, uint8_t { 255 }), "automated test with compiler");

template<typename T, typename... Args>
inline constexpr static bool IsAddError(const T num1PreferredConstexpr, const T num2, const Args...args) noexcept {
   return IsAddError(num1PreferredConstexpr, num2) || IsAddError(static_cast<T>(num1PreferredConstexpr + num2), args...);
}

static_assert(!IsAddError(uint8_t { 0 }, uint8_t { 0 }, uint8_t { 0 }), "automated test with compiler");
static_assert(!IsAddError(uint8_t { 0 }, uint8_t { 0 }, uint8_t { 0 }, uint8_t { 0 }), "automated test with compiler");
static_assert(!IsAddError(uint8_t { 127 }, uint8_t { 127 }, uint8_t { 1 }), "automated test with compiler");
static_assert(!IsAddError(uint8_t { 127 }, uint8_t { 126 }, uint8_t { 1 }, uint8_t { 1 }), "automated test with compiler");
static_assert(IsAddError(uint8_t { 127 }, uint8_t { 127 }, uint8_t { 1 }, uint8_t { 1 }), "automated test with compiler");
static_assert(IsAddError(uint8_t { 127 }, uint8_t { 127 }, uint8_t { 2 }, uint8_t { 0 }), "automated test with compiler");


template<typename T, typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, void *>::type = nullptr>
struct internal_is_twos_complement {
   // this struct is only defined for negative integral numbers

   // The C standard allows 3 types of signed numbers: "sign and magnitude", "ones’ complement" and "two’s complement"
   // The pre-C++20 is silent on this but in practice interoperability requires the C standard definition
   // C++20 only allows two's compliment
   // https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2218.htm#c-sign
   // https://stackoverflow.com/questions/12231560/correct-way-to-take-absolute-value-of-int-min

   typedef T ST;
   typedef typename std::make_unsigned<T>::type UT;

   // the C++ standard says that converting from signed to unsigned is always defined 
   // behavior and also that the result will be the two's complement, so this is well formed
   static constexpr bool value = static_cast<UT>(std::numeric_limits<ST>::max()) ==
      static_cast<UT>(std::numeric_limits<ST>::lowest()) - UT { 1 };
};
template<typename T>
struct is_twos_complement {
   static constexpr bool value = internal_is_twos_complement<T>::value;
};

static_assert(is_twos_complement<int8_t>::value, "compiler not twos complement");
static_assert(is_twos_complement<int16_t>::value, "compiler not twos complement");
static_assert(is_twos_complement<int32_t>::value, "compiler not twos complement");
static_assert(is_twos_complement<int64_t>::value, "compiler not twos complement");

template<typename T>
inline constexpr static typename std::enable_if<std::is_signed<T>::value, typename std::make_unsigned<T>::type>::type TwosComplementConvert(const T val) noexcept {
   // the C++ standard says that converting from signed to unsigned is always defined behavior and also that the
   // result will be the two's complement, so we can just static_cast this for our result
   return static_cast<typename std::make_unsigned<T>::type>(val);
}

template<typename T>
inline constexpr static typename std::enable_if<std::is_unsigned<T>::value, typename std::make_signed<T>::type>::type TwosComplementConvert(const T val) noexcept {
   typedef T UT;
   typedef typename std::make_signed<T>::type ST;

   static_assert(is_twos_complement<ST>::value, "we only support twos complement negative numbers");

   return val < static_cast<UT>(std::numeric_limits<ST>::lowest()) ? static_cast<ST>(val) :
      static_cast<ST>(val + static_cast<UT>(std::numeric_limits<ST>::lowest())) + std::numeric_limits<ST>::lowest();
}

static_assert(TwosComplementConvert(int8_t { 0 }) == uint8_t { 0 }, "test TwosComplementConvert");
static_assert(TwosComplementConvert(int8_t { 1 }) == uint8_t { 1 }, "test TwosComplementConvert");
static_assert(TwosComplementConvert(int8_t { -1 }) == uint8_t { 255 }, "test TwosComplementConvert");
static_assert(TwosComplementConvert(int8_t { -2 }) == uint8_t { 254 }, "test TwosComplementConvert");
static_assert(TwosComplementConvert(int8_t { -126 }) == uint8_t { 130 }, "test TwosComplementConvert");
static_assert(TwosComplementConvert(int8_t { -127 }) == uint8_t { 129 }, "test TwosComplementConvert");
static_assert(TwosComplementConvert(int8_t { -128 }) == uint8_t { 128 }, "test TwosComplementConvert");

static_assert(TwosComplementConvert(uint8_t { 0 }) == int8_t { 0 }, "test TwosComplementConvert");
static_assert(TwosComplementConvert(uint8_t { 1 }) == int8_t { 1 }, "test TwosComplementConvert");
static_assert(TwosComplementConvert(uint8_t { 255 }) == int8_t { -1 }, "test TwosComplementConvert");
static_assert(TwosComplementConvert(uint8_t { 254 }) == int8_t { -2 }, "test TwosComplementConvert");
static_assert(TwosComplementConvert(uint8_t { 130 }) == int8_t { -126 }, "test TwosComplementConvert");
static_assert(TwosComplementConvert(uint8_t { 129 }) == int8_t { -127 }, "test TwosComplementConvert");
static_assert(TwosComplementConvert(uint8_t { 128 }) == int8_t { -128 }, "test TwosComplementConvert");

template<typename TTo, typename TFrom>
inline constexpr static bool IsAbsCastError(const TFrom val) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(std::is_unsigned<TTo>::value, "TTo must be unsigned");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(std::is_signed<TFrom>::value, "TFrom must be signed");
   static_assert(is_twos_complement<TFrom>::value, "we only support twos complement negative numbers");

   // we make the comparison in the unsigned domain. The C++ standard guarantees that the smaller range unsigned
   // number is promoted to the same size as the bigger one
   return std::numeric_limits<TTo>::max() < (
      TFrom { 0 } <= val ? static_cast<typename std::make_unsigned<TFrom>::type>(val) :
      (std::numeric_limits<TFrom>::lowest() != val ? static_cast<typename std::make_unsigned<TFrom>::type>(-val) :
         static_cast<typename std::make_unsigned<TFrom>::type>(std::numeric_limits<TFrom>::lowest())));
}

static_assert(!IsAbsCastError<uint8_t>(int8_t { 0 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint8_t>(int8_t { 127 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint8_t>(int8_t { -1 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint8_t>(int8_t { -127 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint8_t>(int8_t { -128 }), "failed IsAbsCastError");

static_assert(!IsAbsCastError<uint16_t>(int8_t { 0 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint16_t>(int8_t { 1 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint16_t>(int8_t { 127 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint16_t>(int8_t { -1 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint16_t>(int8_t { -127 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint16_t>(int8_t { -128 }), "failed IsAbsCastError");

static_assert(!IsAbsCastError<uint8_t>(int16_t { 0 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint8_t>(int16_t { 1 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint8_t>(int16_t { 127 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint8_t>(int16_t { 128 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint8_t>(int16_t { 255 }), "failed IsAbsCastError");
static_assert(IsAbsCastError<uint8_t>(int16_t { 256 }), "failed IsAbsCastError");
static_assert(IsAbsCastError<uint8_t>(int16_t { 257 }), "failed IsAbsCastError");
static_assert(IsAbsCastError<uint8_t>(int16_t { 127 * 3 }), "failed IsAbsCastError");
static_assert(IsAbsCastError<uint8_t>(int16_t { 32767 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint8_t>(int16_t { -1 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint8_t>(int16_t { -127 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint8_t>(int16_t { -128 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint8_t>(int16_t { -129 }), "failed IsAbsCastError");
static_assert(!IsAbsCastError<uint8_t>(int16_t { -255 }), "failed IsAbsCastError");
static_assert(IsAbsCastError<uint8_t>(int16_t { -256 }), "failed IsAbsCastError");
static_assert(IsAbsCastError<uint8_t>(int16_t { -257 }), "failed IsAbsCastError");
static_assert(IsAbsCastError<uint8_t>(int16_t { -32767 }), "failed IsAbsCastError");
static_assert(IsAbsCastError<uint8_t>(int16_t { -32768 }), "failed IsAbsCastError");

template<typename TTo, typename TFrom>
inline constexpr static TTo AbsCast(const TFrom val) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(std::is_unsigned<TTo>::value, "TTo must be unsigned");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(std::is_signed<TFrom>::value, "TFrom must be unsigned");
   static_assert(is_twos_complement<TFrom>::value, "we only support twos complement negative numbers");

   return TFrom { 0 } <= val ? static_cast<TTo>(val) : (std::numeric_limits<TFrom>::lowest() == val ?
      static_cast<TTo>(static_cast<typename std::make_unsigned<TFrom>::type>(std::numeric_limits<TFrom>::lowest())) :
      static_cast<TTo>(-val));
}

static_assert(AbsCast<uint8_t>(int8_t { 0 }) == uint8_t { 0 }, "failed AbsCast");
static_assert(AbsCast<uint8_t>(int8_t { 1 }) == uint8_t { 1 }, "failed AbsCast");
static_assert(AbsCast<uint8_t>(int8_t { 127 }) == uint8_t { 127 }, "failed AbsCast");
static_assert(AbsCast<uint8_t>(int8_t { -1 }) == uint8_t { 1 }, "failed AbsCast");
static_assert(AbsCast<uint8_t>(int8_t { -127 }) == uint8_t { 127 }, "failed AbsCast");
static_assert(AbsCast<uint8_t>(int8_t { -128 }) == uint8_t { 128 }, "failed AbsCast");

static_assert(AbsCast<uint16_t>(int8_t { 0 }) == uint16_t { 0 }, "failed AbsCast");
static_assert(AbsCast<uint16_t>(int8_t { 1 }) == uint16_t { 1 }, "failed AbsCast");
static_assert(AbsCast<uint16_t>(int8_t { 127 }) == uint16_t { 127 }, "failed AbsCast");
static_assert(AbsCast<uint16_t>(int8_t { -1 }) == uint16_t { 1 }, "failed AbsCast");
static_assert(AbsCast<uint16_t>(int8_t { -127 }) == uint16_t { 127 }, "failed AbsCast");
static_assert(AbsCast<uint16_t>(int8_t { -128 }) == uint16_t { 128 }, "failed AbsCast");

static_assert(AbsCast<uint8_t>(int16_t { 0 }) == uint8_t { 0 }, "failed AbsCast");
static_assert(AbsCast<uint8_t>(int16_t { 1 }) == uint8_t { 1 }, "failed AbsCast");
static_assert(AbsCast<uint8_t>(int16_t { 127 }) == uint8_t { 127 }, "failed AbsCast");
static_assert(AbsCast<uint8_t>(int16_t { 128 }) == uint8_t { 128 }, "failed AbsCast");
static_assert(AbsCast<uint8_t>(int16_t { 255 }) == uint8_t { 255 }, "failed AbsCast");
static_assert(AbsCast<uint8_t>(int16_t { -1 }) == uint8_t { 1 }, "failed AbsCast");
static_assert(AbsCast<uint8_t>(int16_t { -127 }) == uint8_t { 127 }, "failed AbsCast");
static_assert(AbsCast<uint8_t>(int16_t { -128 }) == uint8_t { 128 }, "failed AbsCast");
static_assert(AbsCast<uint8_t>(int16_t { -255 }) == uint8_t { 255 }, "failed AbsCast");


/*
* These are not used currently, but perhaps someday if our trick to use macros to merge compiler and runtime
* values via templating (see GET_COUNT_CLASSES and GET_ITEMS_PER_BIT_PACK and GET_COUNT_DIMENSIONS) gets too many
* warnings, we might want to change over to passing in std::integral_constant with template<typename TConstantType>
* the way that catboost does for this kind of thing
* 

template<typename T>
struct is_null {
   static constexpr bool value = false;
};
template<>
struct is_null<std::nullptr_t> {
   // C++14 has std::is_null_pointer, but we support C++11, so implement this here
   static constexpr bool value = true;
};
static_assert(!is_null<int>::value, "is_null failed on int");
static_assert(!is_null<int *>::value, "is_null failed on int *");
static_assert(is_null<nullptr_t>::value, "is_null failed on nullptr_t");

template<typename T>
struct extract_integral_constant {
   static constexpr bool is_type = false;
   static constexpr T value = T { 0 };
   using value_type = T;
};
template<typename T, T TVal>
struct extract_integral_constant<std::integral_constant<T, TVal>> {
   static constexpr bool is_type = true;
   static constexpr T value = TVal;
   using value_type = T;
};
static_assert(!extract_integral_constant<int>::is_type, "failed on int is_type");
static_assert(extract_integral_constant<std::integral_constant<int, 5>>::is_type, "failed on integral_constant is_type");
static_assert(extract_integral_constant<int>::value == 0, "failed on int value");
static_assert(extract_integral_constant<std::integral_constant<int, 5>>::value == 5, "failed on integral_constant value");
static_assert(std::is_same<extract_integral_constant<int>::value_type, int>::value, "failed on int value_type");
static_assert(std::is_same<extract_integral_constant<std::integral_constant<int, 5>>::value_type, int>::value, "failed on integral_constant value_type");

template<typename T>
inline constexpr static typename std::enable_if<!extract_integral_constant<T>::is_type, T>::type GetConstexpr(const T defaultNonConst) noexcept {
   return defaultNonConst;
}
template<typename T>
inline constexpr static typename std::enable_if<extract_integral_constant<T>::is_type, typename extract_integral_constant<T>::value_type>::type GetConstexpr(const typename extract_integral_constant<T>::value_type defaultNonConst) noexcept {
   return extract_integral_constant<T>::value;
}
static_assert(int { 9 } == GetConstexpr<int>(int { 9 }), "GetConstexpr failed on int 9");
static_assert(int { 4 } == GetConstexpr<std::integral_constant<int, 4>>(int { 9 }), "GetConstexpr failed on int 9");
*/

} // DEFINED_ZONE_NAME

#endif // COMMON_CPP_HPP
