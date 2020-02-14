// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef EBM_INTERNAL_H
#define EBM_INTERNAL_H

#include <inttypes.h>
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <type_traits> // is_integral
#include <cmath> // std::exp, std::log
#include <stdlib.h> // malloc, realloc, free

#include "ebm_native.h"

#define UNUSED(x) (void)(x)
// UBSAN really doesn't like it when we access data past the end of a class eg( p->m_a[2], when m_a is declared as an array of 1)
// We do this however in a number of places to co-locate memory for performance reasons.  We do allocate sufficient memory for doing this, and we 
// also statically check that our classes are standard layout structures (even if declared as classes), so accessing that memory is legal.
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

// The Oracle Developer Studio compiler doesn't seem to have a way to push/pop warning/error messages, but they do have the concept of the "default" which 
// acts as a pop for the specific warning that we turn on/off
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

// TODO : use EBM_RESTRICT_FUNCTION_RETURN EBM_RESTRICT_PARAM_VARIABLE and EBM_NOALIAS.  This helps performance by telling the compiler that pointers are 
//   not aliased
// EBM_RESTRICT_FUNCTION_RETURN tells the compiler that a pointer returned from a function in not aliased in any other part of the program 
// (the memory wasn't reached previously)
#define EBM_RESTRICT_FUNCTION_RETURN __declspec(restrict)
// EBM_RESTRICT_PARAM_VARIABLE tells the compiler that a pointer passed into a function doesn't refer to memory passed in via annohter pointer
#define EBM_RESTRICT_PARAM_VARIABLE __restrict
// EBM_NOALIAS tells the compiler that a function does not modify global state and only modified data DIRECTLY pointed to via it's parameters 
// (first level indirection)
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

// TODO: put a list of all the epilon constants that we use here throughout (use 1e-7 format).  Make it a percentage based on the FloatEbmType data type 
//   minimum eplison from 1 + minimal_change.  If we can make it a constant, then do that, or make it a percentage of a dynamically detected/changing value.  
//   Perhaps take the sqrt of the minimal change from 1?
// when comparing floating point numbers, check this info out: https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/


// TODO: do a full top to bottom review of EbmStatistics.h (not the diff, which we've already checked)
// TODO: starting from the input functions, follow all places that we use/create floating point numbers and look for overflow possibilities
// TODO: search on all my epsilon values and see if they are being used consistently

// gain should be positive, so any number is essentially illegal, but let's make our number very very negative so that we can't confuse it with small 
// negative values close to zero that might occur due to numeric instability
constexpr FloatEbmType k_illegalGain = std::numeric_limits<FloatEbmType>::lowest();
constexpr FloatEbmType k_epsilonNegativeGainAllowed = -1e-7;
constexpr FloatEbmType k_epsilonNegativeValidationMetricAllowed = -1e-7;
constexpr FloatEbmType k_epsilonResidualError = 1e-7;
constexpr FloatEbmType k_epsilonLogLoss = 1e-7;

WARNING_PUSH
WARNING_DISABLE_SIGNED_UNSIGNED_MISMATCH
template<typename TTo, typename TFrom>
constexpr EBM_INLINE bool IsNumberConvertable(const TFrom number) {
   // the general rules of conversion are as follows:
   // calling std::numeric_limits<?>::max() returns an item of that type
   // casting and comparing will never give us undefined behavior.  It can give us implementation defined behavior or unspecified behavior, which is legal.
   // Undefined behavior results from overflowing negative integers, but we don't add or subtract.
   // C/C++ uses value preserving instead of sign preserving.  Generally, if you have two integer numbers that you're comparing then if one type can be 
   // converted into the other with no loss in range then that the smaller range integer is converted into the bigger range integer
   // if one type can't cover the entire range of the other, then items are converted to UNSIGNED values.  This is probably the most dangerous 
   // thing for us to deal with

   static_assert(std::is_integral<TTo>::value, "TTo must be integral");
   static_assert(std::is_integral<TFrom>::value, "TFrom must be integral");

   static_assert(std::numeric_limits<TTo>::is_specialized, "TTo must be specialized");
   static_assert(std::numeric_limits<TFrom>::is_specialized, "TFrom must be specialized");

   static_assert(std::numeric_limits<TTo>::is_signed || 0 == std::numeric_limits<TTo>::lowest(), "min of an unsigned TTo value must be zero");
   static_assert(std::numeric_limits<TFrom>::is_signed || 0 == std::numeric_limits<TFrom>::lowest(), "min of an unsigned TFrom value must be zero");
   static_assert(0 <= std::numeric_limits<TTo>::max(), "TTo max must be positive");
   static_assert(0 <= std::numeric_limits<TFrom>::max(), "TFrom max must be positive");
   static_assert(std::numeric_limits<TTo>::is_signed != std::numeric_limits<TFrom>::is_signed || 
      ((std::numeric_limits<TTo>::lowest() <= std::numeric_limits<TFrom>::lowest() && std::numeric_limits<TFrom>::max() <= std::numeric_limits<TTo>::max()) || 
      (std::numeric_limits<TFrom>::lowest() <= std::numeric_limits<TTo>::lowest() && std::numeric_limits<TTo>::max() <= std::numeric_limits<TFrom>::max())), 
      "types should entirely wrap their smaller types or be the same size"
   );

   return std::numeric_limits<TTo>::is_signed ? 
      (std::numeric_limits<TFrom>::is_signed ? (std::numeric_limits<TTo>::lowest() <= number && number <= std::numeric_limits<TTo>::max()) 
         : (number <= std::numeric_limits<TTo>::max())) : (std::numeric_limits<TFrom>::is_signed ? (0 <= number && number <= std::numeric_limits<TTo>::max()) :
         (number <= std::numeric_limits<TTo>::max()));

   // C++11 is pretty limited for constexpr functions and requires everything to be in 1 line (above).  In C++14 though the below more readable code should
   // be used.
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
   //      // the zero comparison is done signed.  If number is negative, then the results of the max comparison are unspecified, but we don't care because 
   //         it's not undefined and any value true or false will lead to the same answer since the zero comparison was false.
   //      // For the max comparison, if both operands are the same size, then number will be converted to the unsigned type, which will be fine since we 
   //         already checked that it wasn't zero
   //      // For the max comparison, if one operand is bigger, then both operands will be converted to that type and the result will not have 
   //         unspecified behavior
   //      return 0 <= number && number <= std::numeric_limits<TTo>::max();
   //   } else {
   //      // To unsigned from unsigned
   //      // both are unsigned, so both will be upconverted to the biggest data type and then compared.  There is no undefined or unspecified behavior here
   //      return number <= std::numeric_limits<TTo>::max();
   //   }
   //}
}
WARNING_POP

enum class FeatureType { Ordinal = 0, Nominal = 1};

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

#ifdef EBM_NATIVE_R
// we get size NOTES if we compile with too many multiclass optimizations in CRAN, so reduce them to the bare minimum
constexpr ptrdiff_t k_cCompilerOptimizedTargetClassesMax = 2;
#else // EBM_NATIVE_R
constexpr ptrdiff_t k_cCompilerOptimizedTargetClassesMax = 8;
#endif // EBM_NATIVE_R

static_assert(
   2 <= k_cCompilerOptimizedTargetClassesMax, 
   "we special case binary classification to have only 1 output.  If we remove the compile time optimization for the binary class situation then we would "
   "output model files with two values instead of our special case 1");

typedef size_t StorageDataType;
typedef UIntEbmType ActiveDataType;

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
constexpr EBM_INLINE bool IsMulticlass(const ptrdiff_t learningTypeOrCountTargetClasses) {
   return IsClassification(learningTypeOrCountTargetClasses) && !IsBinaryClassification(learningTypeOrCountTargetClasses);
}

constexpr EBM_INLINE size_t GetVectorLength(const ptrdiff_t learningTypeOrCountTargetClasses) {
   // this will work for anything except if learningTypeOrCountTargetClasses is set to DYNAMIC_CLASSIFICATION which means we should have passed in the 
   // dynamic value since DYNAMIC_CLASSIFICATION is a constant that doesn't tell us anything about the real value
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
// having compile time counts of the target count of classes should allow for loop elimination in most cases and the restoration of SIMD instructions in
// places where you couldn't do so with variable loop iterations
#define GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(MACRO_compilerLearningTypeOrCountTargetClasses, MACRO_runtimeLearningTypeOrCountTargetClasses) \
   (k_DynamicClassification == (MACRO_compilerLearningTypeOrCountTargetClasses) ? (MACRO_runtimeLearningTypeOrCountTargetClasses) : \
   (MACRO_compilerLearningTypeOrCountTargetClasses))

// THIS NEEDS TO BE A MACRO AND NOT AN INLINE FUNCTION -> an inline function will cause all the parameters to get resolved before calling the function
// We want any arguments to our macro to not get resolved if they are not needed at compile time so that we do less work if it's not needed
// This will effectively turn the variable into a compile time constant if it can be resolved at compile time
// having compile time counts of the target count of classes should allow for loop elimination in most cases and the restoration of SIMD instructions in 
// places where you couldn't do so with variable loop iterations
// TODO: use this macro more
// TODO: do we really need the static_cast to size_t here?
#define GET_ATTRIBUTE_COMBINATION_DIMENSIONS(MACRO_countCompilerDimensions, MACRO_countRuntimeDimensions) \
   ((MACRO_countCompilerDimensions) <= 0 ? static_cast<size_t>(MACRO_countRuntimeDimensions) : static_cast<size_t>(MACRO_countCompilerDimensions))

// THIS NEEDS TO BE A MACRO AND NOT AN INLINE FUNCTION -> an inline function will cause all the parameters to get resolved before calling the function
// We want any arguments to our macro to not get resolved if they are not needed at compile time so that we do less work if it's not needed
// This will effectively turn the variable into a compile time constant if it can be resolved at compile time
// having compile time counts of the target count of classes should allow for loop elimination in most cases and the restoration of SIMD instructions in 
// places where you couldn't do so with variable loop iterations
#define GET_COUNT_ITEMS_PER_BIT_PACKED_DATA_UNIT(MACRO_compilerCountItemsPerBitPackedDataUnit, MACRO_runtimeCountItemsPerBitPackedDataUnit) \
   (k_cItemsPerBitPackedDataUnitDynamic == (MACRO_compilerCountItemsPerBitPackedDataUnit) ? \
      (MACRO_runtimeCountItemsPerBitPackedDataUnit) : (MACRO_compilerCountItemsPerBitPackedDataUnit))

template<typename T>
constexpr size_t CountBitsRequired(const T maxValue) {
   // this is a bit inefficient when called in the runtime, but we don't call it anywhere that's important performance wise.
   return 0 == maxValue ? 0 : 1 + CountBitsRequired<T>(maxValue / 2);
}
template<typename T>
constexpr EBM_INLINE size_t CountBitsRequiredPositiveMax() {
   return CountBitsRequired(std::numeric_limits<T>::max());
}

constexpr size_t k_cBitsForSizeT = CountBitsRequiredPositiveMax<size_t>();
// it's impossible for us to have more than k_cDimensionsMax dimensions.  Even if we had the minimum number of bin per variable (two), then we would have 
// 2^N memory spaces at our binning step, and that would exceed our memory size if it's greater than the number of bits allowed in a size_t, so on a 
// 64 bit machine, 64 dimensions is a hard maximum.  We can subtract one bit safely, since we know that the rest of our program takes some memory, denying 
// the full 64 bits of memory available.  This extra bit is very helpful since we can then set the 64th bit without overflowing it inside loops and 
// other places
// TODO : we can restrict the dimensionatlity even more because HistogramBuckets aren't 1 byte, so we can see how many would fit into memory.  
//   This isn't a big deal, but it could be nice if we generate static code to handle every possible valid dimension value
constexpr size_t k_cDimensionsMax = k_cBitsForSizeT - 1;
static_assert(k_cDimensionsMax < k_cBitsForSizeT, "reserve the highest bit for bit manipulation space");

constexpr size_t k_cBitsForStorageType = CountBitsRequiredPositiveMax<StorageDataType>();

constexpr EBM_INLINE size_t GetCountItemsBitPacked(const size_t cBits) {
   return k_cBitsForStorageType / cBits;
}
constexpr EBM_INLINE size_t GetCountBits(const size_t cItemsBitPacked) {
   return k_cBitsForStorageType / cItemsBitPacked;
}
constexpr size_t k_cItemsPerBitPackedDataUnitDynamic = 0;
constexpr size_t k_cItemsPerBitPackedDataUnitMax = 0; // if there are more than 16 (4 bits), then we should just use a loop since the code will be pretty big
constexpr size_t k_cItemsPerBitPackedDataUnitMin = 0; // our default binning leads us to 256 values, which is 8 units per 64-bit data pack
constexpr EBM_INLINE size_t GetNextCountItemsBitPacked(const size_t cItemsBitPackedPrev) {
   // for 64 bits, the progression is: 64,32,21,16, 12,10,9,8,7,6,5,4,3,2,1 [there are 15 of these]
   // for 32 bits, the progression is: 32,16,10,8,6,5,4,3,2,1 [which are all included in 64 bits]
   return k_cItemsPerBitPackedDataUnitMin == cItemsBitPackedPrev ? 
      k_cItemsPerBitPackedDataUnitDynamic : k_cBitsForStorageType / ((k_cBitsForStorageType / cItemsBitPackedPrev) + 1);
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

// TODO: use this a lot more.  std::nothrow on new apparently doesn't always work, so we should probably catch the allocation exceptions instead or use this
EBM_INLINE void * SmartMalloc(size_t cItems, size_t cBytesPerItem) {
   if(IsMultiplyError(cItems, cBytesPerItem)) {
      return nullptr;
   } else {
      return malloc(cItems * cBytesPerItem);
   }
}

// TODO eventually, eliminate these variables, and make eliminating logits a part of our regular framework
static constexpr ptrdiff_t k_iZeroResidual = -1;
static constexpr ptrdiff_t k_iZeroClassificationLogitAtInitialize = -1;

// TODO eventually consider using these approximate functions for exp and log.  They make a BIG difference!
//#define FAST_EXP
//#define FAST_LOG

#ifdef FAST_EXP
EBM_INLINE FloatEbmType EbmExp(FloatEbmType val) {
   // we use EbmExp to calculate the residual error, but we calculate the residual error with inputs only from the target and our logits
   // so if we introduce some noise in the residual error from approximations to exp, it will be seen and corrected by later boosting steps
   // so it's largely self correcting
   //
   // Exp is also used to calculate the log loss, but in that case we report the log loss, but otherwise don't use it again, so any errors
   // in calculating the log loss don't propegate cyclically
   //
   // when we get our logit update from training a feature, we apply that to both the model AND our per instance array of logits, so we can
   // potentialy diverge there over time, but that's just an addition operation which is going to be exact for many decimal places.
   // that divergence will NOT be affected by noise in the exp function since the noise in the exp function
   // will generate noise in the logit update, but it won't cause a divergence between the model and the error

   // for algorithm, see https://codingforspeed.com/using-faster-exponential-approximation/
   // TODO make the number of multiplications below a copmile time constant so we can try different values (9 in the code below)

   // here's annohter implementation in AVX-512 (with a table)-> http://www.ecs.umass.edu/arith-2018/pdf/arith25_18.pdf

   val = FloatEbmType { 1 } + val / FloatEbmType { 512 };
   val *= val;
   val *= val;
   val *= val;
   val *= val;
   val *= val;
   val *= val;
   val *= val;
   val *= val;
   val *= val;
   return val;
}
#else // FAST_EXP
EBM_INLINE FloatEbmType EbmExp(FloatEbmType val) {
   return std::exp(val);
}
#endif // FAST_EXP

#ifdef FAST_LOG

// possible approaches:
// NOTE: even though memory lookup table approaches look to be the fastest and reasonable approach, we probably want to avoid lookup tables
//   in order to avoid using too much of our cache memory which we need for other things
// https://www.icsi.berkeley.edu/pubs/techreports/TR-07-002.pdf
// https ://tech.ebayinc.com/engineering/fast-approximate-logarithms-part-i-the-basics/
// https://tech.ebayinc.com/engineering/fast-approximate-logarithms-part-ii-rounding-error/
// https://tech.ebayinc.com/engineering/fast-approximate-logarithms-part-iii-the-formulas/


// TODO move this include up into the VS specific parts
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_BitScanReverse64)

template<typename T>
EBM_INLINE unsigned int MostSignificantBit(T val) {
   // TODO this only works in MS compiler.  This also doesn't work for numbers larger than uint64_t.  This has many problems, so review it.
   unsigned long index;
   return _BitScanReverse64(&index, static_cast<unsigned __int64>(val)) ? static_cast<unsigned int>(index) : static_cast<unsigned int>(0);
}

EBM_INLINE FloatEbmType EbmLog(FloatEbmType val) {
   // TODO: also look into whehter std::log1p has a good approximation directly

   // the log function is only used to calculate the log loss on the valididation set only
   // the log loss is calculated for the validation set and then returned as a single number to the caller
   // it never gets used as an input to anything inside our code, so any errors won't cyclically grow

   // TODO : this only handles numbers x > 1.  I think I don't need results for less than x < 1 though, so check into that.   If we do have numbers below 1, 
   //   we should do 1/x and figure out how much to multiply below

   // for various algorithms, see https://stackoverflow.com/questions/9799041/efficient-implementation-of-natural-logarithm-ln-and-exponentiation
 
   // TODO: this isn't going to work for us since we will often get vlaues greater than 2^64 in exp terms.  Let's figure out how to do the alternate where
   // we extract the exponent term directly via IEEE 754
   unsigned int shifts = MostSignificantBit(static_cast<uint64_t>(val));
   val = val / static_cast<FloatEbmType>(uint64_t { 1 } << shifts);

   // this works sorta kinda well for numbers between 1 to 2 (we shifted our number to be within this range)
   // TODO : increase precision of these magic numbers
   val = FloatEbmType { -1.7417939 } + (FloatEbmType { 2.8212026 } + (FloatEbmType { -1.4699568 } + (FloatEbmType { 0.44717955 } + 
      FloatEbmType { -0.056570851 } * val) * val) * val) * val;
   val += static_cast<FloatEbmType>(shifts) * FloatEbmType { 0.69314718 };

   return val;
}
#else // FAST_LOG
EBM_INLINE FloatEbmType EbmLog(FloatEbmType val) {
   return std::log(val);
   // TODO: also look into whehter std::log1p is a good function for this (mostly in terms of speed).  For the most part we don't care about accuracy 
   //   in the low
   // digits since we take the average, and the log loss will therefore be dominated by a few items that we predict strongly won't happen, but do happen.  
}
#endif // FAST_LOG

#endif // EBM_INTERNAL_H
