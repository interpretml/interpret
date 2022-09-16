// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef LOGGING_H
#define LOGGING_H

#include <assert.h>

#include "ebm_native.h" // TraceEbm

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifdef __clang__ // compiler type (clang++)

#if __has_feature(attribute_analyzer_noreturn)
#define LOGGING_ANALYZER_NORETURN __attribute__((analyzer_noreturn))
#else // __has_feature(attribute_analyzer_noreturn)
#define LOGGING_ANALYZER_NORETURN
#endif // __has_feature(attribute_analyzer_noreturn)

#else // __clang__

#define LOGGING_ANALYZER_NORETURN

#endif // __clang__

extern const char g_sTrue[];
extern const char g_sFalse[];

inline static const char * ObtainTruth(const BoolEbm b) {
   return EBM_FALSE != b ? g_sTrue : g_sFalse;
}

extern TraceEbm g_traceLevel;

extern void InteralLogWithArguments(const TraceEbm traceLevel, const char * const sMessage, ...);
extern void InteralLogWithoutArguments(const TraceEbm traceLevel, const char * const sMessage);
extern void LogAssertFailure(
   const unsigned long long lineNumber,
   const char * const sFileName,
   const char * const sFunctionName,
   const char * const sAssertText
) LOGGING_ANALYZER_NORETURN ;

// We use separate macros for LOG_0 (zero parameters) and LOG_N (variadic parameters) because having zero parameters is non-standardized in C++11
// In C++20, there will be __VA_OPT__, but I don't want to take a dependency on such a new standard yet
// In GCC, you can use ##__VA_ARGS__, but it's non-standard
// In C++11, you can use (but it's more complicated and I can't generate "static const char" string arrays by default):
// static constexpr size_t LOG__cArguments = std::tuple_size<decltype(std::make_tuple(__VA_ARGS__))>::value; // this gets the number of arguments
// static const char * LOG__sMessage = std::get<0>(std::make_tuple(__VA_ARGS__)); // this gets a specific argument by index.  In C++14 it should 
// have a static constexpr version

// use a MACRO for LOG_0/LOG_N(..) and LOG_COUNTED_0/LOG_COUNTED_N(..) instead of an inline because:
//   1) we can use static_assert on the log level
//   2) we can get the number of variadic arguments at compile time, allowing the call to InteralLogWithArguments to be optimized away in cases where we 
//      have a simple string
//   3) we can put our input string into a static const char LOG__sMessage[] instead of keeping it as a string, which is useful in some languages 
//      because I think strings are not const, so by default get put into the read/write data segment instead of readonly
//   3) our variadic arguments won't be evaluated unless they are necessary (log level is set high enough).  If we had inlined them, they might have 
//      needed to be evaluated, depending on the inputs

// MACRO notes for the below:
// using a do loop below gives us a nice look to the macro where the caller needs to use a semi-colon to call it,
// and it can be used after a single if statement without curly braces
//
// the "while( (void)0, 0)" part supresses the conditional expression is constant compiler warning.  The unusual 
// space between the opening braces "( (" is actually important to have to avoid warnings!!
// 
// we only use each input parameter once, which avoids pre and post decrement issues with macros
//
// we use LOG__ prefixes for any variables that we define to avoid collisions with any code we get inserted into
#define LOG_0(traceLevel, sMessage) \
   do { \
      const TraceEbm LOG__traceLevel = (traceLevel); \
      static_assert(Trace_Off < LOG__traceLevel, "traceLevel can't be Trace_Off or lower for call to LOG_0(traceLevel, sMessage, ...)"); \
      static_assert(LOG__traceLevel <= Trace_Verbose, "traceLevel can't be higher than Trace_Verbose for call to LOG_0(traceLevel, sMessage, ...)"); \
      if(LOG__traceLevel <= g_traceLevel) { \
         static const char LOG__sMessage[] = sMessage; \
         InteralLogWithoutArguments(LOG__traceLevel, LOG__sMessage); \
      } \
   } while( (void)0, 0)

#define LOG_N(traceLevel, sMessage, ...) \
   do { \
      const TraceEbm LOG__traceLevel = (traceLevel); \
      static_assert(Trace_Off < LOG__traceLevel, "traceLevel can't be Trace_Off or lower for call to LOG_N(traceLevel, sMessage, ...)"); \
      static_assert(LOG__traceLevel <= Trace_Verbose, \
         "traceLevel can't be higher than Trace_Verbose for call to LOG_N(traceLevel, sMessage, ...)"); \
      if(LOG__traceLevel <= g_traceLevel) { \
         static const char LOG__sMessage[] = sMessage; \
         InteralLogWithArguments(LOG__traceLevel, LOG__sMessage, __VA_ARGS__); \
      } \
   } while( (void)0, 0)

#define LOG_COUNTED_0(pLogCountDecrement, traceLevelBefore, traceLevelAfter, sMessage) \
   do { \
      const TraceEbm LOG__traceLevelBefore = (traceLevelBefore); \
      static_assert(Trace_Off < LOG__traceLevelBefore, \
         "traceLevelBefore can't be Trace_Off or lower for call to LOG_COUNTED_0(pLogCount, traceLevelBefore, traceLevelAfter, sMessage, ...)"); \
      static_assert(LOG__traceLevelBefore <= Trace_Verbose, \
         "traceLevelBefore can't be higher than Trace_Verbose for call to LOG_COUNTED_0(pLogCount, traceLevelBefore, traceLevelAfter, sMessage, ...)"); \
      const TraceEbm LOG__traceLevelAfter = (traceLevelAfter); \
      static_assert(Trace_Off < LOG__traceLevelAfter, \
         "traceLevelAfter can't be Trace_Off or lower for call to LOG_COUNTED_0(pLogCount, traceLevelBefore, traceLevelAfter, sMessage, ...)"); \
      static_assert(LOG__traceLevelAfter <= Trace_Verbose, \
         "traceLevelAfter can't be higher than Trace_Verbose for call to LOG_COUNTED_0(pLogCount, traceLevelBefore, traceLevelAfter, sMessage, ...)"); \
      static_assert(LOG__traceLevelBefore < LOG__traceLevelAfter, \
         "We only support increasing the required trace level after N iterations. It doesn't make sense to have equal values, otherwise just use LOG_0(..)"); \
      const TraceEbm LOG__traceLevel = g_traceLevel; \
      if(LOG__traceLevelBefore <= LOG__traceLevel) { \
         do { \
            TraceEbm LOG__traceLevelLogging; \
            if(LOG__traceLevel < LOG__traceLevelAfter) { \
               int * const LOG__pLogCountDecrement = (pLogCountDecrement); \
               const int LOG__logCount = *LOG__pLogCountDecrement - 1; \
               if(LOG__logCount < 0) { \
                  break; \
               } \
               *LOG__pLogCountDecrement = LOG__logCount; \
               LOG__traceLevelLogging = LOG__traceLevelBefore; \
            } else { \
               LOG__traceLevelLogging = LOG__traceLevelAfter; \
            }\
            static const char LOG__sMessage[] = sMessage; \
            InteralLogWithoutArguments(LOG__traceLevelLogging, LOG__sMessage); \
         } while( (void)0, 0); \
      } \
   } while( (void)0, 0)

#define LOG_COUNTED_N(pLogCountDecrement, traceLevelBefore, traceLevelAfter, sMessage, ...) \
   do { \
      const TraceEbm LOG__traceLevelBefore = (traceLevelBefore); \
      static_assert(Trace_Off < LOG__traceLevelBefore, \
         "traceLevelBefore can't be Trace_Off or lower for call to LOG_COUNTED_N(pLogCount, traceLevelBefore, traceLevelAfter, sMessage, ...)"); \
      static_assert(LOG__traceLevelBefore <= Trace_Verbose, \
         "traceLevelBefore can't be higher than Trace_Verbose for call to LOG_COUNTED_N(pLogCount, traceLevelBefore, traceLevelAfter, sMessage, ...)"); \
      const TraceEbm LOG__traceLevelAfter = (traceLevelAfter); \
      static_assert(Trace_Off < LOG__traceLevelAfter, \
         "traceLevelAfter can't be Trace_Off or lower for call to LOG_COUNTED_N(pLogCount, traceLevelBefore, traceLevelAfter, sMessage, ...)"); \
      static_assert(LOG__traceLevelAfter <= Trace_Verbose, \
         "traceLevelAfter can't be higher than Trace_Verbose for call to LOG_COUNTED_N(pLogCount, traceLevelBefore, traceLevelAfter, sMessage, ...)"); \
      static_assert(LOG__traceLevelBefore < LOG__traceLevelAfter, \
         "We only support increasing the required trace level after N iterations and it doesn't make sense to have equal values, otherwise just use LOG_N(...)"); \
      const TraceEbm LOG__traceLevel = g_traceLevel; \
      if(LOG__traceLevelBefore <= LOG__traceLevel) { \
         do { \
            TraceEbm LOG__traceLevelLogging; \
            if(LOG__traceLevel < LOG__traceLevelAfter) { \
               int * const LOG__pLogCountDecrement = (pLogCountDecrement); \
               const int LOG__logCount = *LOG__pLogCountDecrement - 1; \
               if(LOG__logCount < 0) { \
                  break; \
               } \
               *LOG__pLogCountDecrement = LOG__logCount; \
               LOG__traceLevelLogging = LOG__traceLevelBefore; \
            } else { \
               LOG__traceLevelLogging = LOG__traceLevelAfter; \
            }\
            static const char LOG__sMessage[] = sMessage; \
            InteralLogWithArguments(LOG__traceLevelLogging, LOG__sMessage, __VA_ARGS__); \
         } while( (void)0, 0); \
      } \
   } while( (void)0, 0)

#ifndef NDEBUG
// the "assert(!  #bCondition)" condition needs some explanation.  At that point we definetly want to assert false, and we also want to include the text
// of the assert that triggered the failure. Any string will have a non-zero pointer, so negating it will always fail, and we'll get to see the text of 
// the original failure in the message this allows us to use whatever behavior has been chosen by the C runtime library implementor for assertion 
// failures without using the undocumented function that assert calls internally on each platform
#define EBM_ASSERT(bCondition) ((void)((bCondition) || (LogAssertFailure(STATIC_CAST(unsigned long long, __LINE__), __FILE__, __func__, #bCondition), assert(!  #bCondition), 0)))
#else // NDEBUG
#define EBM_ASSERT(bCondition) ((void)0)
#endif // NDEBUG

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // LOGGING_H
