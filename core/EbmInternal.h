// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef EBM_INTERNAL_H
#define EBM_INTERNAL_H

#include <inttypes.h>
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <type_traits> // is_integral

// TODO : remove INVALID_POINTER, since there is no portable way to get a pointer that has all the upper bits, and using it is undefined behavior too!
#define INVALID_POINTER (reinterpret_cast<void *>(~uintptr_t { 0 }))
#define UNUSED(x) (void)(x)
// UBSAN really doesn't like it when we access data past the end of a class eg( p->m_a[2], when m_a is declared as an array of 1)
// We do this however in a number of places to co-locate memory for performance reasons.  We do allocate sufficient memory for doing this, and we also statically check that our classes are standard layout structures (even if declared as classes), so accessing that memory is legal.
// this MACRO turns an array reference into a pointer to the same type of object, which resolves any UBSAN warnings

// TODO : the const and non-const macros can probably be unified
#define ARRAY_TO_POINTER(x) (reinterpret_cast<typename std::remove_all_extents<decltype(x)>::type *>(reinterpret_cast<void *>(x)))
#define ARRAY_TO_POINTER_CONST(x) (reinterpret_cast<const typename std::remove_all_extents<decltype(x)>::type *>(reinterpret_cast<const void *>(x)))

// here's how to detect the compiler type for a variety of compilers -> https://sourceforge.net/p/predef/wiki/Compilers/
// disabling warnings with _Pragma detailed info here https://stackoverflow.com/questions/3378560/how-to-disable-gcc-warnings-for-a-few-lines-of-code

#if defined(__clang__) // compiler type

#define WARNING_PUSH _Pragma("clang diagnostic push")
#define WARNING_POP _Pragma("clang diagnostic pop")
#define WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
#define WARNING_DISABLE_SIGNED_UNSIGNED_MISMATCH _Pragma("clang diagnostic ignored \"-Wsign-compare\"")
#define WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO
#define WARNING_DISABLE_NON_LITERAL_PRINTF_STRING _Pragma("clang diagnostic ignored \"-Wformat-nonliteral\"")

#elif defined(__GNUC__) // compiler type

#define WARNING_PUSH _Pragma("GCC diagnostic push")
#define WARNING_POP _Pragma("GCC diagnostic pop")
#define WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE _Pragma("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
#define WARNING_DISABLE_SIGNED_UNSIGNED_MISMATCH _Pragma("GCC diagnostic ignored \"-Wsign-compare\"")
#define WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO
#define WARNING_DISABLE_NON_LITERAL_PRINTF_STRING

#elif defined(__SUNPRO_CC) // compiler type (Oracle Developer Studio)

// The Oracle Developer Studio compiler doesn't seem to have a way to push/pop warning/error messages, but they do have the concept of the "default" which acts as a pop for the specific warning that we turn on/off
// Since we can only default on previously changed warnings, we need to have matching warnings off/default sets, so use WARNING_DEFAULT_* 
// example: WARNING_DISABLE_SOMETHING   _Pragma("error_messages(off,something1,something2)")
// example: WARNING_DEFAULT_SOMETHING   _Pragma("error_messages(default,something1,something2)")

#define WARNING_PUSH
#define WARNING_POP
#define WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
#define WARNING_DISABLE_SIGNED_UNSIGNED_MISMATCH
#define WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO
#define WARNING_DISABLE_NON_LITERAL_PRINTF_STRING

#elif defined(_MSC_VER) // compiler type

#define WARNING_PUSH __pragma(warning(push))
#define WARNING_POP __pragma(warning(pop))
#define WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE __pragma(warning(disable: 4701))
#define WARNING_DISABLE_SIGNED_UNSIGNED_MISMATCH __pragma(warning(disable: 4018))
#define WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO __pragma(warning(disable: 4723))
#define WARNING_DISABLE_NON_LITERAL_PRINTF_STRING

#else  // compiler type
#error compiler not recognized
#endif // compiler type

#if defined(__clang__) || defined(__GNUC__) || defined(__SUNPRO_CC) // compiler
#ifndef __has_builtin
#define __has_builtin(x) 0 // __has_builtin is supported in newer compilers.  On older compilers diable anything we would check with it
#endif // __has_builtin

#define LIKELY(b) __builtin_expect(static_cast<bool>(b), 1)
#define UNLIKELY(b) __builtin_expect(static_cast<bool>(b), 0)
#define PREDICTABLE(b) (b)

#if __has_builtin(__builtin_unpredictable)
#define UNPREDICTABLE(b) __builtin_unpredictable(b)
#else // __has_builtin(__builtin_unpredictable)
#define UNPREDICTABLE(b) (b)
#endif // __has_builtin(__builtin_unpredictable)

#define EBM_INLINE inline __attribute__((always_inline))

// TODO : use EBM_RESTRICT_FUNCTION_RETURN EBM_RESTRICT_PARAM_VARIABLE and EBM_NOALIAS.  This helps performance by telling the compiler that pointers are not aliased
// EBM_RESTRICT_FUNCTION_RETURN tells the compiler that a pointer returned from a function in not aliased in any other part of the program (the memory wasn't reached previously)
#define EBM_RESTRICT_FUNCTION_RETURN __declspec(restrict)
// EBM_RESTRICT_PARAM_VARIABLE tells the compiler that a pointer passed into a function doesn't refer to memory passed in via annohter pointer
#define EBM_RESTRICT_PARAM_VARIABLE __restrict
// EBM_NOALIAS tells the compiler that a function does not modify global state and only modified data DIRECTLY pointed to via it's parameters (first level indirection)
#define EBM_NOALIAS __declspec(noalias)

#elif defined(_MSC_VER) // compiler type

#define LIKELY(b) (b)
#define UNLIKELY(b) (b)
#define PREDICTABLE(b) (b)
#define UNPREDICTABLE(b) (b)
#define EBM_INLINE __forceinline

#else // compiler type
#error compiler not recognized
#endif // compiler type

WARNING_PUSH
WARNING_DISABLE_SIGNED_UNSIGNED_MISMATCH

template<typename TTo, typename TFrom>
constexpr EBM_INLINE bool IsNumberConvertable(const TFrom number) {
   // the general rules of conversion are as follows:
   // calling std::numeric_limits<?>::max() returns an item of that type
   // casting and comparing will never give us undefined behavior.  It can give us implementation defined behavior or unspecified behavior, which is legal.  Undefined behavior results from overflowing negative integers, but we don't add or subtract.
   // C/C++ uses value preserving instead of sign preserving.  Generally, if you have two integer numbers that you're comparing then if one type can be converted into the other with no loss in range then that the smaller range integer is converted into the bigger range integer
   // if one type can't cover the entire range of the other, then items are converted to UNSIGNED values.  This is probably the most dangerous thing for us to deal with

   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");

   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");

   static_assert(std::numeric_limits<TTo>::is_signed || 0 == std::numeric_limits<TTo>::lowest(), "min of an unsigned TTo value must be zero");
   static_assert(std::numeric_limits<TFrom>::is_signed || 0 == std::numeric_limits<TFrom>::lowest(), "min of an unsigned TFrom value must be zero");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo max must be positive");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom max must be positive");
   static_assert(std::numeric_limits<TTo>::is_signed != std::numeric_limits<TFrom>::is_signed || ((std::numeric_limits<TTo>::lowest() <= std::numeric_limits<TFrom>::lowest() && std::numeric_limits<TFrom>::max() <= std::numeric_limits<TTo>::max()) || (std::numeric_limits<TFrom>::lowest() <= std::numeric_limits<TTo>::lowest() && std::numeric_limits<TTo>::max() <= std::numeric_limits<TFrom>::max())), "types should entirely wrap their smaller types or be the same size");

   return std::numeric_limits<TTo>::is_signed ? (std::numeric_limits<TFrom>::is_signed ? (std::numeric_limits<TTo>::lowest() <= number && number <= std::numeric_limits<TTo>::max()) : (number <= std::numeric_limits<TTo>::max())) : (std::numeric_limits<TFrom>::is_signed ? (0 <= number && number <= std::numeric_limits<TTo>::max()) : (number <= std::numeric_limits<TTo>::max()));

   // C++11 is pretty limited for constexpr functions and requires everything to be in 1 line (above).  In C++14 though the below more readable code should be used.
   //if(std::numeric_limits<TTo>::is_signed) {
   //   if(std::numeric_limits<TFrom>::is_signed) {
   //      // To signed from signed
   //      // if both operands are the same size, then they should be the same type
   //      // if one operand is bigger, then both operands will be converted to that type and the result will not have unspecified behavior
   //      return std::numeric_limits<TTo>::lowest() <= number && number <= std::numeric_limits<TTo>::max();
   //   } else {
   //      // To signed from unsigned
   //      // if both operands are the same size, then max will be converted to the unsigned type, but that should be fine as max should fit
   //      // if one operand is bigger, then both operands will be converted to that type and the result will not have unspecified behavior
   //      return number <= std::numeric_limits<TTo>::max();
   //   }
   //} else {
   //   if(std::numeric_limits<TFrom>::is_signed) {
   //      // To unsigned from signed
   //      // the zero comparison is done signed.  If number is negative, then the results of the max comparison are unspecified, but we don't care because it's not undefined and any value true or false will lead to the same answer since the zero comparison was false.
   //      // For the max comparison, if both operands are the same size, then number will be converted to the unsigned type, which will be fine since we already checked that it wasn't zero
   //      // For the max comparison, if one operand is bigger, then both operands will be converted to that type and the result will not have unspecified behavior
   //      return 0 <= number && number <= std::numeric_limits<TTo>::max();
   //   } else {
   //      // To unsigned from unsigned
   //      // both are unsigned, so both will be upconverted to the biggest data type and then compared.  There is no undefined or unspecified behavior here
   //      return number <= std::numeric_limits<TTo>::max();
   //   }
   //}
}

WARNING_POP

enum class FeatureTypeCore { OrdinalCore = 0, NominalCore = 1};

// there doesn't seem to be a reasonable upper bound for how high you can set the k_cCompilerOptimizedTargetClassesMax value.  The bottleneck seems to be that setting it too high increases compile time and module size
// this is how much the runtime speeds up if you compile it with hard coded vector sizes
// 200 => 2.65%
// 32  => 3.28%
// 16  => 5.12%
// 8   => 5.34%
// 4   => 8.31%
// TODO: increase this up to something like 16.  I have decreased it to 8 in order to make compiling more efficient, and so that I regularily test the runtime looped version of our code
constexpr ptrdiff_t k_cCompilerOptimizedTargetClassesMax = 8;
static_assert(2 <= k_cCompilerOptimizedTargetClassesMax, "we special case binary classification to have only 1 output.  If we remove the compile time optimization for the binary class situation then we would output model files with two values instead of our special case 1");

typedef size_t StorageDataTypeCore;
typedef size_t ActiveDataType;

constexpr ptrdiff_t k_Regression = -1;
constexpr ptrdiff_t k_DynamicClassification = 0;
constexpr EBM_INLINE bool IsRegression(const ptrdiff_t learningTypeOrCountTargetClasses) {
   return k_Regression == learningTypeOrCountTargetClasses;
}
constexpr EBM_INLINE bool IsClassification(const ptrdiff_t learningTypeOrCountTargetClasses) {
   return 0 <= learningTypeOrCountTargetClasses;
}
constexpr EBM_INLINE bool IsBinaryClassification(const ptrdiff_t learningTypeOrCountTargetClasses) {
#ifdef EXPAND_BINARY_LOGITS
   return UNUSED(learningTypeOrCountTargetClasses), false;
#else // EXPAND_BINARY_LOGITS
   return 2 == learningTypeOrCountTargetClasses;
#endif // EXPAND_BINARY_LOGITS
}

constexpr EBM_INLINE size_t GetVectorLengthFlatCore(const ptrdiff_t learningTypeOrCountTargetClasses) {
   // this will work for anything except if learningTypeOrCountTargetClasses is set to DYNAMIC_CLASSIFICATION which means we should have passed in the dynamic value since DYNAMIC_CLASSIFICATION is a constant that doesn't tell us anything about the real value
#ifdef EXPAND_BINARY_LOGITS
   return learningTypeOrCountTargetClasses <= ptrdiff_t { 1 } ? size_t { 1 } : static_cast<size_t>(learningTypeOrCountTargetClasses);
#else // EXPAND_BINARY_LOGITS
   return learningTypeOrCountTargetClasses <= ptrdiff_t { 2 } ? size_t { 1 } : static_cast<size_t>(learningTypeOrCountTargetClasses);
#endif // EXPAND_BINARY_LOGITS
}

// THIS NEEDS TO BE A MACRO AND NOT AN INLINE FUNCTION -> an inline function will cause all the parameters to get resolved before calling the function
// We want any arguments to our macro to not get resolved if they are not needed at compile time so that we do less work if it's not needed
// This will effectively turn the variable into a compile time constant if it can be resolved at compile time
// The caller can put pTargetFeature->m_cBins inside the macro call and it will be optimize away if it isn't necessary
// having compile time counts of the target count of classes should allow for loop elimination in most cases and the restoration of SIMD instructions in places where you couldn't do so with variable loop iterations
#define GET_VECTOR_LENGTH(MACRO_compilerLearningTypeOrCountTargetClasses, MACRO_runtimeLearningTypeOrCountTargetClasses) (GetVectorLengthFlatCore(k_DynamicClassification == (MACRO_compilerLearningTypeOrCountTargetClasses) ? (MACRO_runtimeLearningTypeOrCountTargetClasses) : (MACRO_compilerLearningTypeOrCountTargetClasses)))

// THIS NEEDS TO BE A MACRO AND NOT AN INLINE FUNCTION -> an inline function will cause all the parameters to get resolved before calling the function
// We want any arguments to our macro to not get resolved if they are not needed at compile time so that we do less work if it's not needed
// This will effectively turn the variable into a compile time constant if it can be resolved at compile time
// having compile time counts of the target count of classes should allow for loop elimination in most cases and the restoration of SIMD instructions in places where you couldn't do so with variable loop iterations
// TODO: use this macro more
#define GET_ATTRIBUTE_COMBINATION_DIMENSIONS(MACRO_countCompilerDimensions, MACRO_countRuntimeDimensions) ((MACRO_countCompilerDimensions) <= 0 ? static_cast<size_t>(MACRO_countRuntimeDimensions) : static_cast<size_t>(MACRO_countCompilerDimensions))

template<typename T>
constexpr size_t CountBitsRequiredCore(const T maxValue) {
   // this is a bit inefficient when called in the runtime, but we don't call it anywhere that's important performance wise.
   return 0 == maxValue ? 0 : 1 + CountBitsRequiredCore<T>(maxValue / 2);
}
template<typename T>
constexpr size_t CountBitsRequiredPositiveMax() {
   return CountBitsRequiredCore(std::numeric_limits<T>::max());
}

constexpr size_t k_cBitsForSizeTCore = CountBitsRequiredPositiveMax<size_t>();
// it's impossible for us to have more than k_cDimensionsMax dimensions.  Even if we had the minimum number of bin per variable (two), then we would have 2^N memory spaces at our binning step, and that would exceed our memory size if it's greater than the number of bits allowed in a size_t, so on a 64 bit machine, 64 dimensions is a hard maximum.  We can subtract one bit safely, since we know that the rest of our program takes some memory, denying the full 64 bits of memory available.  This extra bit is very helpful since we can then set the 64th bit without overflowing it inside loops and other places
// TODO : we can restrict the dimensionatlity even more because HistogramBuckets aren't 1 byte, so we can see how many would fit into memory.  This isn't a big deal, but it could be nice if we generate static code to handle every possible valid dimension value
constexpr size_t k_cDimensionsMax = k_cBitsForSizeTCore - 1;
static_assert(k_cDimensionsMax < k_cBitsForSizeTCore, "reserve the highest bit for bit manipulation space");

constexpr size_t k_cBitsForStorageType = CountBitsRequiredPositiveMax<StorageDataTypeCore>();

constexpr EBM_INLINE size_t GetCountItemsBitPacked(const size_t cBits) {
   return k_cBitsForStorageType / cBits;
}
constexpr EBM_INLINE size_t GetCountBits(const size_t cItemsBitPacked) {
   return k_cBitsForStorageType / cItemsBitPacked;
}
constexpr EBM_INLINE size_t GetNextCountItemsBitPacked(const size_t cItemsBitPackedPrev) {
   // for 64 bits, the progression is: 64,32,21,16, 12,10,9,8,7,6,5,4,3,2,1
   // for 32 bits, the progression is: 32,16,10,8,6,5,4,3,2,1 [which are all included in 64 bits]
   return k_cBitsForStorageType / ((k_cBitsForStorageType / cItemsBitPackedPrev) + 1);
}

WARNING_PUSH
WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO
constexpr EBM_INLINE bool IsMultiplyError(const size_t num1, const size_t num2) {
   // algebraically, we want to know if this is true: std::numeric_limits<size_t>::max() + 1 <= num1 * num2
   // which can be turned into: (std::numeric_limits<size_t>::max() + 1 - num1) / num1 + 1 <= num2
   // which can be turned into: (std::numeric_limits<size_t>::max() + 1 - num1) / num1 < num2
   // which can be turned into: (std::numeric_limits<size_t>::max() - num1 + 1) / num1 < num2
   // which works if num1 == 1, but does not work if num1 is zero, so check for zero first

   // it will never overflow if num1 is zero
   return 0 != num1 && ((std::numeric_limits<size_t>::max() - num1 + 1) / num1 < num2);
}
WARNING_POP

constexpr EBM_INLINE bool IsAddError(const size_t num1, const size_t num2) {
   // overflow for unsigned values is defined behavior in C++ and it causes a wrap arround
   return num1 + num2 < num1;
}

// TODO eventually, eliminate these variables, and make eliminating logits a part of our regular framework
static constexpr ptrdiff_t k_iZeroResidual = -1;
static constexpr ptrdiff_t k_iZeroClassificationLogitAtInitialize = -1;

#endif // EBM_INTERNAL_H
