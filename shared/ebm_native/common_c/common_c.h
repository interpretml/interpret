// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef COMMON_C_H
#define COMMON_C_H

#include <string.h>

#ifdef __cplusplus
extern "C" {
#define EBM_NOEXCEPT noexcept
#else // __cplusplus
#define EBM_NOEXCEPT
#endif // __cplusplus

#define UNUSED(x) (void)(x)
   // TODO: use OUT throughout our codebase
   // this just visually tags parameters as being modified by their caller
#define OUT
#define INOUT

// here's how to detect the compiler type for a variety of compilers -> https://sourceforge.net/p/predef/wiki/Compilers/
// disabling warnings with _Pragma detailed info here https://stackoverflow.com/questions/3378560/how-to-disable-gcc-warnings-for-a-few-lines-of-code
#if defined(__clang__) // compiler type (clang++)

#define WARNING_PUSH _Pragma("clang diagnostic push")
#define WARNING_POP _Pragma("clang diagnostic pop")
#define WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
#define WARNING_DISABLE_UNINITIALIZED_LOCAL_POINTER
#define WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO
#define WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
#define WARNING_BUFFER_OVERRUN
#define WARNING_REDUNDANT_CODE
#define ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER

#if __has_feature(attribute_analyzer_noreturn)
#define ANALYZER_NORETURN __attribute__((analyzer_noreturn))
#else // __has_feature(attribute_analyzer_noreturn)
#define ANALYZER_NORETURN
#endif // __has_feature(attribute_analyzer_noreturn)

#elif defined(__GNUC__) // compiler type (g++)

#define WARNING_PUSH _Pragma("GCC diagnostic push")
#define WARNING_POP _Pragma("GCC diagnostic pop")
#define WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE _Pragma("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
#define WARNING_DISABLE_UNINITIALIZED_LOCAL_POINTER
#define WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO
#define WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
#define WARNING_BUFFER_OVERRUN
#define WARNING_REDUNDANT_CODE
#define ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER

#define ANALYZER_NORETURN

#elif defined(__SUNPRO_CC) // compiler type (Oracle Developer Studio)

// The Oracle Developer Studio compiler doesn't seem to have a way to push/pop warning/error messages, but they do have the concept of the "default" which 
// acts as a pop for the specific warning that we turn on/off
// Since we can only default on previously changed warnings, we need to have matching warnings off/default sets, so use WARNING_DEFAULT_* 
// example: WARNING_DISABLE_SOMETHING   _Pragma("error_messages(off,something1,something2)")
// example: WARNING_DEFAULT_SOMETHING   _Pragma("error_messages(default,something1,something2)")

#define WARNING_PUSH
#define WARNING_POP
#define WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
#define WARNING_DISABLE_UNINITIALIZED_LOCAL_POINTER
#define WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO
#define WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
#define WARNING_BUFFER_OVERRUN
#define WARNING_REDUNDANT_CODE
#define ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER

#define ANALYZER_NORETURN

#elif defined(_MSC_VER) // compiler type (Microsoft Visual Studio compiler)

#define WARNING_PUSH __pragma(warning(push))
#define WARNING_POP __pragma(warning(pop))
#define WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE __pragma(warning(disable: 4701))
#define WARNING_DISABLE_UNINITIALIZED_LOCAL_POINTER __pragma(warning(disable: 4703))
#define WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO __pragma(warning(disable: 4723))
#define WARNING_DISABLE_USING_UNINITIALIZED_MEMORY __pragma(warning(disable: 6001))
#define WARNING_BUFFER_OVERRUN __pragma(warning(disable: 6386))
#define WARNING_REDUNDANT_CODE __pragma(warning(disable: 6287))
#define ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER [[gsl::suppress(type.6)]]

#define ANALYZER_NORETURN

// disable constexpr warning, since GetCountScores is meant to be ambiguous and it's used everywhere
#pragma warning(disable : 26498)
// disable dereferencing NULL pointer, since the static analysis seems to think any access of a pointer is 
// dereferencing a NULL pointer potentially.
#pragma warning(disable : 6011)
// disable dereferencing NULL pointer (same pointer), since the static analysis seems to think any access 
// of a pointer is dereferencing a NULL pointer potentially.
#pragma warning(disable : 28182)
// disable SIMD alignment issues.  We need to align on 16 byte boundaries (64 byte would be better), but we
// use SIMD all over the place, so just disable the general warning
#pragma warning(disable : 4316)

#else  // compiler type
#error compiler not recognized
#endif // compiler type

#if defined(__clang__) || defined(__GNUC__) || defined(__SUNPRO_CC) // compiler
#ifndef __has_builtin
#define __has_builtin(x) 0 // __has_builtin is supported in newer compilers.  On older compilers diable anything we would check with it
#endif // __has_builtin

#define LIKELY(b) __builtin_expect(!!(b), 1)
#define UNLIKELY(b) __builtin_expect(!!(b), 0)
#define PREDICTABLE(b) (b)

#if __has_builtin(__builtin_unpredictable)
#define UNPREDICTABLE(b) __builtin_unpredictable(!!(b))
#else // __has_builtin(__builtin_unpredictable)
#define UNPREDICTABLE(b) (b)
#endif // __has_builtin(__builtin_unpredictable)

#define INLINE_ALWAYS inline __attribute__((always_inline))

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
#define INLINE_ALWAYS inline __forceinline

#else // compiler type
#error compiler not recognized
#endif // compiler type

INLINE_ALWAYS static void StopClangAnalysis() EBM_NOEXCEPT ANALYZER_NORETURN {
}

INLINE_ALWAYS static char * strcpy_NO_WARNINGS(char * const dest, const char * const src) EBM_NOEXCEPT {
   StopClangAnalysis();
   return strcpy(dest, src);
}

#define ANALYSIS_ASSERT(b) \
   do { \
      if(!(b)) \
         StopClangAnalysis(); \
   } while( (void)0, 0)


#define FAST_EXP
#define FAST_LOG

static const char k_registrationSeparator = ',';

extern const char * SkipWhitespace(const char * s);
extern const char * SkipEndWhitespaceWhenGuaranteedNonWhitespace(const char * sEnd);
extern const char * ConvertStringToFloat(
   const char * const s,
   double * const pResultOut
);
extern const char * IsStringEqualsCaseInsensitive(
   const char * sMain,
   const char * sLabel
);

// TODO: use k_cFloatSumLimit in more places (in the histogram generation!!)
//
// floating point numbers have a mantissa of 23 bits.  If you try to sum up more than something in the
// range of 2^23 of numbers with any kind of resolution you'll find that after about 2^23 of them they don't increase
// on average because the additional single increment is below the resolution of a float.  We can get arround this
// though by breaking the work into separate loops where we sum only to a certain high value and then 
// re-do our internal loop again.  The outer loop suffers from the same resolution problem though, so instead
// of 2 loops, use 3 loops which should solve the problem up to arround the range of 2^64.  If we run our
// inner loop until 2^19, then we can handle numbers up to 2^(19 * 3) = 2^57, and then we still get annother
// 4 bits of ultimate resolution until we don't increment on average anymore, so at 2^61.  Our floats are 4 bytes
// so that means we can handle 2^63 floats, and we'll need other memory of course to, so even on a 64 bit machine
// we can handle any number of items.
static const size_t k_cFloatSumLimit = 524288;

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // COMMON_C_H
