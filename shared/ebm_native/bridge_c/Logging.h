// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef LOGGING_H
#define LOGGING_H

#include <assert.h>

#include "ebm_native.h" // LOG_MESSAGE_FUNCTION
#include "bridge_c.h" // INTERNAL_IMPORT_EXPORT_INCLUDE 
#include "common_c.h" // UNLIKELY

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

INTERNAL_IMPORT_EXPORT_INCLUDE const char g_trueString[];
INTERNAL_IMPORT_EXPORT_INCLUDE const char g_falseString[];

INLINE_ALWAYS static const char * ObtainTruth(const BoolEbmType b) {
   return EBM_FALSE != b ? g_trueString : g_falseString;
}

INTERNAL_IMPORT_EXPORT_INCLUDE TraceEbmType g_traceLevel;

INTERNAL_IMPORT_EXPORT_INCLUDE void EBM_NATIVE_CALLING_CONVENTION InteralLogWithArguments(const TraceEbmType traceLevel, const char * const pOriginalMessage, ...);
INTERNAL_IMPORT_EXPORT_INCLUDE void EBM_NATIVE_CALLING_CONVENTION InteralLogWithoutArguments(const TraceEbmType traceLevel, const char * const pOriginalMessage);
INTERNAL_IMPORT_EXPORT_INCLUDE void EBM_NATIVE_CALLING_CONVENTION LogAssertFailure(
   const unsigned long long lineNumber,
   const char * const fileName,
   const char * const functionName,
   const char * const assertText
) ANALYZER_NORETURN ;

// We use separate macros for LOG_0 (zero parameters) and LOG_N (variadic parameters) because having zero parameters is non-standardized in C++11
// In C++20, there will be __VA_OPT__, but I don't want to take a dependency on such a new standard yet
// In GCC, you can use ##__VA_ARGS__, but it's non-standard
// In C++11, you can use (but it's more complicated and I can't generate "const static char" string arrays by default):
// constexpr size_t LOG__cArguments = std::tuple_size<decltype(std::make_tuple(__VA_ARGS__))>::value; // this gets the number of arguments
// static const char * LOG__originalMessage = std::get<0>(std::make_tuple(__VA_ARGS__)); // this gets a specific argument by index.  In C++14 it should 
// have a constexpr version

// use a MACRO for LOG_0/LOG_N(..) and LOG_COUNTED_0/LOG_COUNTED_N(..) instead of an inline because:
//   1) we can use static_assert on the log level
//   2) we can get the number of variadic arguments at compile time, allowing the call to InteralLogWithArguments to be optimized away in cases where we 
//      have a simple string
//   3) we can put our input string into a const static char LOG__originalMessage[] instead of keeping it as a string, which is useful in some languages 
//      because I think strings are not const, so by default get put into the read/write data segment instead of readonly
//   3) our variadic arguments won't be evaluated unless they are necessary (log level is set high enough).  If we had inlined them, they might have 
//      needed to be evaluated, depending on the inputs

// MACRO notes for the below:
// using a do loop below gives us a nice look to the macro where the caller needs to use a semi-colon to call it,
// and it can be used after a single if statement without curly braces
//
// the "(void)0, 0" part supresses the conditional expression is constant compiler warning
// 
// we only use each input parameter once, which avoids pre and post decrement issues with macros
//
// we use LOG__ prefixes for any variables that we define to avoid collisions with any code we get inserted into
#define LOG_0(traceLevel, pLogMessage) \
   do { \
      const TraceEbmType LOG__traceLevel = (traceLevel); \
      static_assert(TraceLevelOff < LOG__traceLevel, "traceLevel can't be TraceLevelOff or lower for call to LOG_0(traceLevel, pLogMessage, ...)"); \
      static_assert(LOG__traceLevel <= TraceLevelVerbose, "traceLevel can't be higher than TraceLevelVerbose for call to LOG_0(traceLevel, pLogMessage, ...)"); \
      if(UNLIKELY(LOG__traceLevel <= g_traceLevel)) { \
         const static char LOG__originalMessage[] = pLogMessage; \
         InteralLogWithoutArguments(LOG__traceLevel, LOG__originalMessage); \
      } \
   } while( (void)0, 0)

#define LOG_N(traceLevel, pLogMessage, ...) \
   do { \
      const TraceEbmType LOG__traceLevel = (traceLevel); \
      static_assert(TraceLevelOff < LOG__traceLevel, "traceLevel can't be TraceLevelOff or lower for call to LOG_N(traceLevel, pLogMessage, ...)"); \
      static_assert(LOG__traceLevel <= TraceLevelVerbose, \
         "traceLevel can't be higher than TraceLevelVerbose for call to LOG_N(traceLevel, pLogMessage, ...)"); \
      if(UNLIKELY(LOG__traceLevel <= g_traceLevel)) { \
         const static char LOG__originalMessage[] = pLogMessage; \
         InteralLogWithArguments(LOG__traceLevel, LOG__originalMessage, __VA_ARGS__); \
      } \
   } while( (void)0, 0)

#define LOG_COUNTED_0(pLogCountDecrement, traceLevelBefore, traceLevelAfter, pLogMessage) \
   do { \
      const TraceEbmType LOG__traceLevelBefore = (traceLevelBefore); \
      static_assert(TraceLevelOff < LOG__traceLevelBefore, \
         "traceLevelBefore can't be TraceLevelOff or lower for call to LOG_COUNTED_0(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      static_assert(LOG__traceLevelBefore <= TraceLevelVerbose, \
         "traceLevelBefore can't be higher than TraceLevelVerbose for call to LOG_COUNTED_0(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      const TraceEbmType LOG__traceLevelAfter = (traceLevelAfter); \
      static_assert(TraceLevelOff < LOG__traceLevelAfter, \
         "traceLevelAfter can't be TraceLevelOff or lower for call to LOG_COUNTED_0(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      static_assert(LOG__traceLevelAfter <= TraceLevelVerbose, \
         "traceLevelAfter can't be higher than TraceLevelVerbose for call to LOG_COUNTED_0(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      static_assert(LOG__traceLevelBefore < LOG__traceLevelAfter, \
         "We only support increasing the required trace level after N iterations. It doesn't make sense to have equal values, otherwise just use LOG_0(..)"); \
      const TraceEbmType LOG__traceLevel = g_traceLevel; \
      if(UNLIKELY(LOG__traceLevelBefore <= LOG__traceLevel)) { \
         do { \
            TraceEbmType LOG__traceLevelLogging; \
            if(LIKELY(LOG__traceLevel < LOG__traceLevelAfter)) { \
               int * const LOG__pLogCountDecrement = (pLogCountDecrement); \
               const int LOG__logCount = *LOG__pLogCountDecrement - 1; \
               if(LIKELY(LOG__logCount < 0)) { \
                  break; \
               } \
               *LOG__pLogCountDecrement = LOG__logCount; \
               LOG__traceLevelLogging = LOG__traceLevelBefore; \
            } else { \
               LOG__traceLevelLogging = LOG__traceLevelAfter; \
            }\
            const static char LOG__originalMessage[] = pLogMessage; \
            InteralLogWithoutArguments(LOG__traceLevelLogging, LOG__originalMessage); \
         } while( (void)0, 0); \
      } \
   } while( (void)0, 0)

#define LOG_COUNTED_N(pLogCountDecrement, traceLevelBefore, traceLevelAfter, pLogMessage, ...) \
   do { \
      const TraceEbmType LOG__traceLevelBefore = (traceLevelBefore); \
      static_assert(TraceLevelOff < LOG__traceLevelBefore, \
         "traceLevelBefore can't be TraceLevelOff or lower for call to LOG_COUNTED_N(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      static_assert(LOG__traceLevelBefore <= TraceLevelVerbose, \
         "traceLevelBefore can't be higher than TraceLevelVerbose for call to LOG_COUNTED_N(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      const TraceEbmType LOG__traceLevelAfter = (traceLevelAfter); \
      static_assert(TraceLevelOff < LOG__traceLevelAfter, \
         "traceLevelAfter can't be TraceLevelOff or lower for call to LOG_COUNTED_N(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      static_assert(LOG__traceLevelAfter <= TraceLevelVerbose, \
         "traceLevelAfter can't be higher than TraceLevelVerbose for call to LOG_COUNTED_N(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      static_assert(LOG__traceLevelBefore < LOG__traceLevelAfter, \
         "We only support increasing the required trace level after N iterations and it doesn't make sense to have equal values, otherwise just use LOG_N(...)"); \
      const TraceEbmType LOG__traceLevel = g_traceLevel; \
      if(UNLIKELY(LOG__traceLevelBefore <= LOG__traceLevel)) { \
         do { \
            TraceEbmType LOG__traceLevelLogging; \
            if(LIKELY(LOG__traceLevel < LOG__traceLevelAfter)) { \
               int * const LOG__pLogCountDecrement = (pLogCountDecrement); \
               const int LOG__logCount = *LOG__pLogCountDecrement - 1; \
               if(LIKELY(LOG__logCount < 0)) { \
                  break; \
               } \
               *LOG__pLogCountDecrement = LOG__logCount; \
               LOG__traceLevelLogging = LOG__traceLevelBefore; \
            } else { \
               LOG__traceLevelLogging = LOG__traceLevelAfter; \
            }\
            const static char LOG__originalMessage[] = pLogMessage; \
            InteralLogWithArguments(LOG__traceLevelLogging, LOG__originalMessage, __VA_ARGS__); \
         } while( (void)0, 0); \
      } \
   } while( (void)0, 0)

#ifndef NDEBUG
// the "assert(!  #bCondition)" condition needs some explanation.  At that point we definetly want to assert false, and we also want to include the text
// of the assert that triggered the failure. Any string will have a non-zero pointer, so negating it will always fail, and we'll get to see the text of 
// the original failure in the message this allows us to use whatever behavior has been chosen by the C runtime library implementor for assertion 
// failures without using the undocumented function that assert calls internally on each platform
#define EBM_ASSERT(bCondition) ((void)(LIKELY(bCondition) || (LogAssertFailure(static_cast<unsigned long long>(__LINE__), __FILE__, __func__, #bCondition), assert(!  #bCondition), 0)))
#else // NDEBUG
#define EBM_ASSERT(bCondition) ((void)0)
#endif // NDEBUG

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // LOGGING_H
