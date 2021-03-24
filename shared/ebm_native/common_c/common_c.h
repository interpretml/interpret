// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef COMMON_C_H
#define COMMON_C_H

#include <string.h>

#include "ebm_native.h"

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
#define WARNING_DISABLE_SIGNED_UNSIGNED_MISMATCH _Pragma("clang diagnostic ignored \"-Wsign-compare\"")
#define WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO
#define WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
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
#define WARNING_DISABLE_SIGNED_UNSIGNED_MISMATCH _Pragma("GCC diagnostic ignored \"-Wsign-compare\"")
#define WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO
#define WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
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
#define WARNING_DISABLE_SIGNED_UNSIGNED_MISMATCH
#define WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO
#define WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
#define ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER

#define ANALYZER_NORETURN

#elif defined(_MSC_VER) // compiler type (Microsoft Visual Studio compiler)

#define WARNING_PUSH __pragma(warning(push))
#define WARNING_POP __pragma(warning(pop))
#define WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE __pragma(warning(disable: 4701))
#define WARNING_DISABLE_SIGNED_UNSIGNED_MISMATCH __pragma(warning(disable: 4018))
#define WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO __pragma(warning(disable: 4723))
#define WARNING_DISABLE_USING_UNINITIALIZED_MEMORY __pragma(warning(disable: 6001))
#define ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER [[gsl::suppress(type.6)]]

#define ANALYZER_NORETURN

// disable constexpr warning, since GetVectorLength is meant to be ambiguous and it's used everywhere
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

#define LIKELY(b) __builtin_expect((b), 1)
#define UNLIKELY(b) __builtin_expect((b), 0)
#define PREDICTABLE(b) (b)

#if __has_builtin(__builtin_unpredictable)
#define UNPREDICTABLE(b) __builtin_unpredictable(b)
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

INLINE_ALWAYS static void StopClangAnalysis() EBM_NOEXCEPT ANALYZER_NORETURN {
}

INLINE_ALWAYS static char * strcpy_NO_WARNINGS(char * dest, const char * src) EBM_NOEXCEPT {
   StopClangAnalysis();
   return strcpy(dest, src);
}

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

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // COMMON_C_H
