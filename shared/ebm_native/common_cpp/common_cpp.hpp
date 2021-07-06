// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef COMMON_CPP_HPP
#define COMMON_CPP_HPP

#include <limits> // numeric_limits
#include <type_traits> // std::is_standard_layout, std::is_integral

#include "ebm_native.h"
#include "logging.h"
#include "common_c.h"
#include "zones.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

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
INLINE_ALWAYS static T * ArrayToPointer(T * const a) noexcept {
   return a;
}
template<typename T>
INLINE_ALWAYS static const T * ArrayToPointer(const T * const a) noexcept {
   return a;
}

// TODO : replace all std::min and std::max and similar comparions that get the min/max with this function
// unlike std::min, our version has explicit noexcept semantics
template<typename T>
INLINE_ALWAYS constexpr static T EbmMin(T v1, T v2) noexcept {
   return UNPREDICTABLE(v1 < v2) ? v1 : v2;
}
// unlike std::max, our version has explicit noexcept semantics
template<typename T>
INLINE_ALWAYS constexpr static T EbmMax(T v1, T v2) noexcept {
   return UNPREDICTABLE(v1 < v2) ? v2 : v1;
}

// use SFINAE to compile time specialize IsNumberConvertable
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
using InternalCheckSSN = typename std::enable_if<
   std::is_signed<TTo>::value && std::is_signed<TFrom>::value && 
   std::numeric_limits<TTo>::lowest() <= std::numeric_limits<TFrom>::lowest() && 
   std::numeric_limits<TFrom>::max() <= std::numeric_limits<TTo>::max()
, bool>::type;
template<typename TTo, typename TFrom, InternalCheckSSN<TTo, TFrom> = true>
INLINE_ALWAYS constexpr static bool IsNumberConvertable(const TFrom number) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(std::numeric_limits<TTo>::lowest() < 0, "TTo::lowest must be negative");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo::max must be positive");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(std::numeric_limits<TFrom>::lowest() < 0, "TFrom::lowest must be negative");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom::max must be positive");

   UNUSED(number);

   // our TTo has either a larger range or the same range as TFrom, so there is no need to check anything
   return true;
}

template<typename TTo, typename TFrom>
using InternalCheckSSY = typename std::enable_if<
   std::is_signed<TTo>::value && std::is_signed<TFrom>::value && 
   !(std::numeric_limits<TTo>::lowest() <= std::numeric_limits<TFrom>::lowest() && 
   std::numeric_limits<TFrom>::max() <= std::numeric_limits<TTo>::max())
, bool>::type;
template<typename TTo, typename TFrom, InternalCheckSSY<TTo, TFrom> = true>
INLINE_ALWAYS constexpr static bool IsNumberConvertable(const TFrom number) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(std::numeric_limits<TTo>::lowest() < 0, "TTo::lowest must be negative");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo::max must be positive");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(std::numeric_limits<TFrom>::lowest() < 0, "TFrom::lowest must be negative");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom::max must be positive");

   static_assert(
      std::numeric_limits<TFrom>::lowest() < std::numeric_limits<TTo>::lowest() && 
      std::numeric_limits<TTo>::max() < std::numeric_limits<TFrom>::max(),
      "we have a specialization for when TTo has a larger range, but if TFrom is larger then check that it's larger on both the upper and lower ends"
   );

   return TFrom { std::numeric_limits<TTo>::lowest() } <= number && number <= TFrom { std::numeric_limits<TTo>::max() };
}

template<typename TTo, typename TFrom>
using InternalCheckUSN = typename std::enable_if<
   !std::is_signed<TTo>::value && std::is_signed<TFrom>::value && 
   std::numeric_limits<TFrom>::max() <= std::numeric_limits<TTo>::max()
, bool>::type;
template<typename TTo, typename TFrom, InternalCheckUSN<TTo, TFrom> = true>
INLINE_ALWAYS constexpr static bool IsNumberConvertable(const TFrom number) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(0 == std::numeric_limits<TTo>::lowest(), "TTo::lowest must be zero");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo::max must be positive");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(std::numeric_limits<TFrom>::lowest() < 0, "TFrom::lowest must be negative");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom::max must be positive");

   return TFrom { 0 } <= number;
}

template<typename TTo, typename TFrom>
using InternalCheckUSY = typename std::enable_if<
   !std::is_signed<TTo>::value && std::is_signed<TFrom>::value &&
   std::numeric_limits<TTo>::max() < std::numeric_limits<TFrom>::max()
, bool>::type;
template<typename TTo, typename TFrom, InternalCheckUSY<TTo, TFrom> = true>
INLINE_ALWAYS constexpr static bool IsNumberConvertable(const TFrom number) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(0 == std::numeric_limits<TTo>::lowest(), "TTo::lowest must be zero");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo::max must be positive");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(std::numeric_limits<TFrom>::lowest() < 0, "TFrom::lowest must be negative");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom::max must be positive");

   return TFrom { 0 } <= number && number <= TFrom { std::numeric_limits<TTo>::max() };
}

template<typename TTo, typename TFrom>
using InternalCheckSUN = typename std::enable_if<
   std::is_signed<TTo>::value && !std::is_signed<TFrom>::value && 
   std::numeric_limits<TFrom>::max() <= std::numeric_limits<TTo>::max()
, bool>::type;
template<typename TTo, typename TFrom, InternalCheckSUN<TTo, TFrom> = true>
INLINE_ALWAYS constexpr static bool IsNumberConvertable(const TFrom number) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(std::numeric_limits<TTo>::lowest() < 0, "TTo::lowest must be negative");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo::max must be positive");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(0 == std::numeric_limits<TFrom>::lowest(), "TFrom::lowest must be zero");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom::max must be positive");

   UNUSED(number);

   return true;
}

template<typename TTo, typename TFrom>
using InternalCheckSUY = typename std::enable_if<
   std::is_signed<TTo>::value && !std::is_signed<TFrom>::value && 
   std::numeric_limits<TTo>::max() < std::numeric_limits<TFrom>::max()
   , bool>::type;
template<typename TTo, typename TFrom, InternalCheckSUY<TTo, TFrom> = true>
INLINE_ALWAYS constexpr static bool IsNumberConvertable(const TFrom number) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(std::numeric_limits<TTo>::lowest() < 0, "TTo::lowest must be negative");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo::max must be positive");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(0 == std::numeric_limits<TFrom>::lowest(), "TFrom::lowest must be zero");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom::max must be positive");

   return number <= TFrom { std::numeric_limits<TTo>::max() };
}

template<typename TTo, typename TFrom>
using InternalCheckUUN = typename std::enable_if<
   !std::is_signed<TTo>::value && !std::is_signed<TFrom>::value && 
   std::numeric_limits<TFrom>::max() <= std::numeric_limits<TTo>::max()
, bool>::type;
template<typename TTo, typename TFrom, InternalCheckUUN<TTo, TFrom> = true>
INLINE_ALWAYS constexpr static bool IsNumberConvertable(const TFrom number) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(0 == std::numeric_limits<TTo>::lowest() < 0, "TTo::lowest must be zero");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo::max must be positive");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(0 == std::numeric_limits<TFrom>::lowest(), "TFrom::lowest must be zero");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom::max must be positive");

   UNUSED(number);

   return true;
}

template<typename TTo, typename TFrom>
using InternalCheckUUY = typename std::enable_if<
   !std::is_signed<TTo>::value && !std::is_signed<TFrom>::value && 
   std::numeric_limits<TTo>::max() < std::numeric_limits<TFrom>::max()
   , bool>::type;
template<typename TTo, typename TFrom, InternalCheckUUY<TTo, TFrom> = true>
INLINE_ALWAYS constexpr static bool IsNumberConvertable(const TFrom number) noexcept {
   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(0 == std::numeric_limits<TTo>::lowest() < 0, "TTo::lowest must be zero");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo::max must be positive");

   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");
   static_assert(0 == std::numeric_limits<TFrom>::lowest(), "TFrom::lowest must be zero");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom::max must be positive");

   return number <= TFrom { std::numeric_limits<TTo>::max() };
}


template<typename T>
constexpr static size_t CountBitsRequired(const T maxValue) noexcept {
   // this is a bit inefficient when called in the runtime, but we don't call it anywhere that's important performance wise.
   return T { 0 } == maxValue ? size_t { 0 } : size_t { 1 } + CountBitsRequired<T>(maxValue / T { 2 });
}
template<typename T>
INLINE_ALWAYS constexpr static size_t CountBitsRequiredPositiveMax() noexcept {
   return CountBitsRequired(std::numeric_limits<T>::max());
}

constexpr static size_t k_cBitsForSizeT = CountBitsRequiredPositiveMax<size_t>();

// It's impossible for us to have tensors with more than k_cDimensionsMax dimensions.  Even if we had the minimum 
// number of bins per feature (two), then we would have 2^N memory spaces at our binning step, and 
// that would exceed our memory size if it's greater than the number of bits allowed in a size_t, so on a 
// 64 bit machine, 64 dimensions is a hard maximum.  We can subtract one bit safely, since we know that 
// the rest of our program takes some memory, denying the full 64 bits of memory available.  This extra 
// bit is very helpful since we can then set the 64th bit without overflowing it inside loops and other places
//
// We strip out features with only 1 value since they provide no learning value and they break this nice property
// of having a maximum number of dimensions.
//
// TODO : we can restrict the dimensionatlity even more because HistogramBuckets aren't 1 byte, so we can see 
//        how many would fit into memory.
constexpr static size_t k_cDimensionsMax = k_cBitsForSizeT - 1;
static_assert(k_cDimensionsMax < k_cBitsForSizeT, "reserve the highest bit for bit manipulation space");


WARNING_PUSH
WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO
template<typename T>
INLINE_ALWAYS constexpr static bool IsMultiplyError(const T num1, const T num2) noexcept {
   static_assert(std::is_integral<T>::value, "T must be integral");
   static_assert(std::numeric_limits<T>::is_specialized, "T must be specialized");
   static_assert(!std::is_signed<T>::value, "T must be unsigned in the current implementation");

   // algebraically, we want to know if this is true: std::numeric_limits<T>::max() + 1 <= num1 * num2
   // which can be turned into: (std::numeric_limits<T>::max() + 1 - num1) / num1 + 1 <= num2
   // which can be turned into: (std::numeric_limits<T>::max() + 1 - num1) / num1 < num2
   // which can be turned into: (std::numeric_limits<T>::max() - num1 + 1) / num1 < num2
   // which works if num1 == 1, but does not work if num1 is zero, so check for zero first

   // it will never overflow if num1 is zero or 1.  We need to check zero to avoid division by zero
   return T { 1 } < num1 && ((std::numeric_limits<T>::max() - num1 + 1) / num1 < num2);
}
WARNING_POP

template<typename T>
INLINE_ALWAYS constexpr static bool IsAddError(const T num1, const T num2) noexcept {
   static_assert(std::is_integral<T>::value, "T must be integral");
   static_assert(std::numeric_limits<T>::is_specialized, "T must be specialized");
   static_assert(!std::is_signed<T>::value, "T must be unsigned in the current implementation");

   // overflow for unsigned values is defined behavior in C++ and it causes a wrap arround
   return num1 + num2 < num1;
}

// we use the struct hack in a number of places in this code base for putting memory in the optimial location
// the struct hack isn't valid unless a class/struct is standard layout.  standard layout objects cannot
// be allocated with new and delete, so we need to use malloc and free for a number of our objects.  It was
// getting confusing to having some objects free with free and other objects use delete, so we just turned
// everything into malloc/free to keep to a single convention.
// 
// Also, using std::nothrow on new apparently doesn't always return nullptr on all compilers.  Sometimes it just 
// exits. This library sometimes allocates large amounts of memory and we'd like to gracefully handle the case where
// that large amount of memory is too large.  So, instead of using new[] and delete[] we use malloc and free.
//
// There's also a small subset of cases where we allocate a chunk of memory and use it for heterogenious types
// in which case we use pure malloc and then free instead of these helper functions.  In both cases we still
// use free though, so it's less likely to create bugs by accident.
template<typename T>
INLINE_ALWAYS static T * EbmMalloc() noexcept {
   static_assert(!std::is_same<T, void>::value, "don't try allocating a single void item with EbmMalloc");
   T * const a = static_cast<T *>(malloc(sizeof(T)));
   return a;
}
template<typename T>
INLINE_ALWAYS static T * EbmMalloc(const size_t cItems) noexcept {
   constexpr size_t cBytesPerItem = sizeof(typename std::conditional<std::is_same<T, void>::value, char, T>::type);
   static_assert(0 < cBytesPerItem, "can't have a zero sized item");
   bool bOneByte = 1 == cBytesPerItem;
   if(bOneByte) {
      const size_t cBytes = cItems;
      // TODO: !! BEWARE: we do use realloc in some parts of our program still!!
      T * const a = static_cast<T *>(malloc(cBytes));
      return a;
   } else {
      if(UNLIKELY(IsMultiplyError(cItems, cBytesPerItem))) {
         return nullptr;
      } else {
         const size_t cBytes = cItems * cBytesPerItem;
         // TODO: !! BEWARE: we do use realloc in some parts of our program still!!
         StopClangAnalysis(); // for some reason Clang-analysis thinks cBytes can be zero, despite the assert above.
         T * const a = static_cast<T *>(malloc(cBytes));
         return a;
      }
   }
}
template<typename T>
INLINE_ALWAYS static T * EbmMalloc(const size_t cItems, const size_t cBytesPerItem) noexcept {
   if(UNLIKELY(IsMultiplyError(cItems, cBytesPerItem))) {
      return nullptr;
   } else {
      const size_t cBytes = cItems * cBytesPerItem;
      // TODO: !! BEWARE: we do use realloc in some parts of our program still!!
      T * const a = static_cast<T *>(malloc(cBytes));
      return a;
   }
}

} // DEFINED_ZONE_NAME

#endif // COMMON_CPP_HPP
