// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef LOGGING_H
#define LOGGING_H

#include <assert.h>

#include "ebmcore.h" // LOG_MESSAGE_FUNCTION
#include "EbmInternal.h" // UNLIKELY

extern signed char g_traceLevel;
extern LOG_MESSAGE_FUNCTION g_pLogMessageFunc;
extern void InteralLogWithArguments(signed char traceLevel, const char * const pOriginalMessage, ...);
extern const char g_assertLogMessage[];

// We use separate macros for LOG_0 (zero parameters) and LOG_N (variadic parameters) because having zero parameters is non-standardized in C++11
// In C++20, there will be __VA_OPT__, but I don't want to take a dependency on such a new standard yet
// In GCC, you can use ##__VA_ARGS__, but it's non-standard
// In C++11, you can use (but it's more complicated and I can't generate "const static char" string arrays by default):
// constexpr size_t LOG__cArguments = std::tuple_size<decltype(std::make_tuple(__VA_ARGS__))>::value; // this gets the number of arguments
// static const char * LOG__originalMessage = std::get<0>(std::make_tuple(__VA_ARGS__)); // this gets a specific argument by index.  In C++14 it should have a constexpr version

// use a MACRO for LOG_0/LOG_N(..) and LOG_COUNTED_0/LOG_COUNTED_N(..) instead of an inline because:
//   1) we can use static_assert on the log level
//   2) we can get the number of variadic arguments at compile time, allowing the call to InteralLogWithArguments to be optimized away in cases where we have a simple string
//   3) we can put our input string into a const static char LOG__originalMessage[] instead of keeping it as a string, which is useful in some languages because I think strings are not const, so by default get put into the read/write data segment instead of readonly
//   3) our variadic arguments won't be evaluated unless they are necessary (log level is set high enough).  If we had inlined them, they might have needed to be evaluated, depending on the inputs
#define LOG_0(traceLevel, pLogMessage) \
   do { /* using a do loop here gives us a nice look to the macro where the caller needs to use a semi-colon to call it, and it can be used after a single if statement without curly braces */ \
      constexpr signed char LOG__traceLevel = (traceLevel); /* we only use traceLevel once, which avoids pre and post decrement issues with macros */ \
      static_assert(TraceLevelOff < LOG__traceLevel, "traceLevel can't be TraceLevelOff or lower for call to LOG_0(traceLevel, pLogMessage, ...)"); \
      static_assert(LOG__traceLevel <= TraceLevelVerbose, "traceLevel can't be higher than TraceLevelVerbose for call to LOG_0(traceLevel, pLogMessage, ...)"); \
      if(UNLIKELY(LOG__traceLevel <= g_traceLevel)) { \
         assert(nullptr != g_pLogMessageFunc); \
         static const char LOG__originalMessage[] = pLogMessage; \
         (*g_pLogMessageFunc)(LOG__traceLevel, LOG__originalMessage); \
      } \
      /* the "(void)0, 0" part supresses the conditional expression is constant compiler warning */ \
   } while((void)0, 0)

#define LOG_N(traceLevel, pLogMessage, ...) \
   do { /* using a do loop here gives us a nice look to the macro where the caller needs to use a semi-colon to call it, and it can be used after a single if statement without curly braces */ \
      constexpr signed char LOG__traceLevel = (traceLevel); /* we only use traceLevel once, which avoids pre and post decrement issues with macros */ \
      static_assert(TraceLevelOff < LOG__traceLevel, "traceLevel can't be TraceLevelOff or lower for call to LOG_N(traceLevel, pLogMessage, ...)"); \
      static_assert(LOG__traceLevel <= TraceLevelVerbose, "traceLevel can't be higher than TraceLevelVerbose for call to LOG_N(traceLevel, pLogMessage, ...)"); \
      if(UNLIKELY(LOG__traceLevel <= g_traceLevel)) { \
         assert(nullptr != g_pLogMessageFunc); \
         static const char LOG__originalMessage[] = pLogMessage; /* we only use pLogMessage once, which avoids pre and post decrement issues with macros */ \
         InteralLogWithArguments(LOG__traceLevel, LOG__originalMessage, __VA_ARGS__); \
      } \
      /* the "(void)0, 0" part supresses the conditional expression is constant compiler warning */ \
   } while((void)0, 0)

#define LOG_COUNTED_0(pLogCountDecrement, traceLevelBefore, traceLevelAfter, pLogMessage) \
   do { /* using a do loop here gives us a nice look to the macro where the caller needs to use a semi-colon to call it, and it can be used after a single if statement without curly braces */ \
      constexpr signed char LOG__traceLevelBefore = (traceLevelBefore); /* we only use traceLevelBefore once, which avoids pre and post decrement issues with macros */ \
      static_assert(TraceLevelOff < LOG__traceLevelBefore, "traceLevelBefore can't be TraceLevelOff or lower for call to LOG_COUNTED_0(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      static_assert(LOG__traceLevelBefore <= TraceLevelVerbose, "traceLevelBefore can't be higher than TraceLevelVerbose for call to LOG_COUNTED_0(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      constexpr signed char LOG__traceLevelAfter = (traceLevelAfter); /* we only use traceLevelAfter once, which avoids pre and post decrement issues with macros */ \
      static_assert(TraceLevelOff < LOG__traceLevelAfter, "traceLevelAfter can't be TraceLevelOff or lower for call to LOG_COUNTED_0(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      static_assert(LOG__traceLevelAfter <= TraceLevelVerbose, "traceLevelAfter can't be higher than TraceLevelVerbose for call to LOG_COUNTED_0(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      static_assert(LOG__traceLevelBefore < LOG__traceLevelAfter, "We only support increasing the required trace level after N iterations and it doesn't make sense to have equal values, otherwise just use LOG_0(..)"); \
      if(UNLIKELY(LOG__traceLevelBefore <= g_traceLevel)) { \
         static const char LOG__originalMessage[] = pLogMessage; /* we only use pLogMessage once, which avoids pre and post decrement issues with macros */ \
         unsigned int * const LOG__pLogCountDecrement = (pLogCountDecrement); /* we only use pLogCountDecrement once, which avoids pre and post decrement issues with macros */ \
         const unsigned int LOG__logCount = *LOG__pLogCountDecrement; \
         if(0 < LOG__logCount) { \
            *LOG__pLogCountDecrement = LOG__logCount - 1; \
            assert(nullptr != g_pLogMessageFunc); \
            (*g_pLogMessageFunc)(LOG__traceLevelBefore, LOG__originalMessage); \
         } else { \
            if(UNLIKELY(LOG__traceLevelAfter <= g_traceLevel)) { \
               assert(nullptr != g_pLogMessageFunc); \
               (*g_pLogMessageFunc)(LOG__traceLevelAfter, LOG__originalMessage); \
            } \
         } \
      } \
      /* the "(void)0, 0" part supresses the conditional expression is constant compiler warning */ \
   } while((void)0, 0)

#define LOG_COUNTED_N(pLogCountDecrement, traceLevelBefore, traceLevelAfter, pLogMessage, ...) \
   do { /* using a do loop here gives us a nice look to the macro where the caller needs to use a semi-colon to call it, and it can be used after a single if statement without curly braces */ \
      constexpr signed char LOG__traceLevelBefore = (traceLevelBefore); /* we only use traceLevelBefore once, which avoids pre and post decrement issues with macros */ \
      static_assert(TraceLevelOff < LOG__traceLevelBefore, "traceLevelBefore can't be TraceLevelOff or lower for call to LOG_COUNTED_N(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      static_assert(LOG__traceLevelBefore <= TraceLevelVerbose, "traceLevelBefore can't be higher than TraceLevelVerbose for call to LOG_COUNTED_N(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      constexpr signed char LOG__traceLevelAfter = (traceLevelAfter); /* we only use traceLevelAfter once, which avoids pre and post decrement issues with macros */ \
      static_assert(TraceLevelOff < LOG__traceLevelAfter, "traceLevelAfter can't be TraceLevelOff or lower for call to LOG_COUNTED_N(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      static_assert(LOG__traceLevelAfter <= TraceLevelVerbose, "traceLevelAfter can't be higher than TraceLevelVerbose for call to LOG_COUNTED_N(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      static_assert(LOG__traceLevelBefore < LOG__traceLevelAfter, "We only support increasing the required trace level after N iterations and it doesn't make sense to have equal values, otherwise just use LOG_N(..)"); \
      if(UNLIKELY(LOG__traceLevelBefore <= g_traceLevel)) { \
         static const char LOG__originalMessage[] = pLogMessage; /* we only use pLogMessage once, which avoids pre and post decrement issues with macros */ \
         unsigned int * const LOG__pLogCountDecrement = (pLogCountDecrement); /* we only use pLogCountDecrement once, which avoids pre and post decrement issues with macros */ \
         const unsigned int LOG__logCount = *LOG__pLogCountDecrement; \
         if(0 < LOG__logCount) { \
            *LOG__pLogCountDecrement = LOG__logCount - 1; \
            assert(nullptr != g_pLogMessageFunc); \
            InteralLogWithArguments(LOG__traceLevelBefore, LOG__originalMessage, __VA_ARGS__); \
         } else { \
            if(UNLIKELY(LOG__traceLevelAfter <= g_traceLevel)) { \
               assert(nullptr != g_pLogMessageFunc); \
               InteralLogWithArguments(LOG__traceLevelAfter, LOG__originalMessage, __VA_ARGS__); \
            } \
         } \
      } \
      /* the "(void)0, 0" part supresses the conditional expression is constant compiler warning */ \
   } while((void)0, 0)

#ifndef NDEBUG
// the "assert(!#bCondition)" condition needs some explanation.  At that point we definetly want to assert false, and we also want to include the text of the assert that triggered the failure.
// Any string will have a non-zero pointer, so negating it will always fail, and we'll get to see the text of the original failure in the message
// this allows us to use whatever behavior has been chosen by the C runtime library implementor for assertion failures without using the undocumented function that assert calls internally on each platform
#define EBM_ASSERT(bCondition) ((void)(UNLIKELY(bCondition) ? 0 : (assert(UNLIKELY(nullptr != g_pLogMessageFunc)), UNLIKELY(TraceLevelError <= g_traceLevel) ? (InteralLogWithArguments(TraceLevelError, g_assertLogMessage, static_cast<unsigned long long>(__LINE__), __FILE__, __func__, #bCondition), 0) : 0, assert(!   #bCondition), 0)))
#else // NDEBUG
#define EBM_ASSERT(bCondition) ((void)0)
#endif // NDEBUG

#endif // LOGGING_H
