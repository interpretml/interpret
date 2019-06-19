// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef LOGGING_H
#define LOGGING_H

#include <assert.h>
#include <tuple>

#include "ebmcore.h" // LOG_MESSAGE_FUNCTION
#include "EbmInternal.h" // UNLIKELY

extern signed char g_traceLevel;
extern LOG_MESSAGE_FUNCTION g_pLogMessageFunc;
extern void InteralLogWithArguments(signed char traceLevel, const char * const pOriginalMessage, ...);
extern const char g_assertLogMessage[];

// use a MACRO for LOG(..) and LOG_COUNTED(..) instead of an inline because:
//   1) we can use static_assert on the log level
//   2) we can get the number of variadic arguments at compile time, allowing the call to InteralLogWithArguments to be optimized away in cases where we have a simple string
//   3) we can put our input string into a constexpr static char LOG__originalMessage[] instead of keeping it as a string, which is useful in some languages because I think strings are not const, so by default get put into the read/write data segment instead of readonly
//   3) our variadic arguments won't be evaluated unless they are necessary (log level is set high enough).  If we had inlined them, they might have needed to be evaluated, depending on the inputs
#define LOG(traceLevel, pLogMessage, ...) \
   do { /* using a do loop here gives us a nice look to the macro where the caller needs to use a semi-colon to call it, and it can be used after a single if statement without curly braces */ \
      constexpr signed char LOG__traceLevel = (traceLevel); /* we only use traceLevel once, which avoids pre and post decrement issues with macros */ \
      static_assert(TraceLevelOff < LOG__traceLevel, "traceLevel can't be TraceLevelOff or lower for call to LOG(traceLevel, pLogMessage, ...)"); \
      static_assert(LOG__traceLevel <= TraceLevelVerbose, "traceLevel can't be higher than TraceLevelVerbose for call to LOG(traceLevel, pLogMessage, ...)"); \
      if(UNLIKELY(LOG__traceLevel <= g_traceLevel)) { \
         assert(nullptr != g_pLogMessageFunc); \
         constexpr size_t LOG__cArguments = std::tuple_size<decltype(std::make_tuple(__VA_ARGS__))>::value; \
         constexpr static char LOG__originalMessage[] = (pLogMessage); /* we only use pLogMessage once, which avoids pre and post decrement issues with macros */ \
         if(0 == LOG__cArguments) { /* if there are no arguments we might as well send the log directly without reserving stack space for vsnprintf and without log length limitations for stack allocation */ \
            (*g_pLogMessageFunc)(LOG__traceLevel, LOG__originalMessage); \
         } else { \
            InteralLogWithArguments(LOG__traceLevel, LOG__originalMessage, ##__VA_ARGS__); \
         } \
      } \
      /* the "(void)0, 0" part supresses the conditional expression is constant compiler warning */ \
   } while((void)0, 0)

#define LOG_COUNTED(pLogCountDecrement, traceLevelBefore, traceLevelAfter, pLogMessage, ...) \
   do { /* using a do loop here gives us a nice look to the macro where the caller needs to use a semi-colon to call it, and it can be used after a single if statement without curly braces */ \
      constexpr signed char LOG__traceLevelBefore = (traceLevelBefore); /* we only use traceLevelBefore once, which avoids pre and post decrement issues with macros */ \
      static_assert(TraceLevelOff < LOG__traceLevelBefore, "traceLevelBefore can't be TraceLevelOff or lower for call to LOG_COUNTED(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      static_assert(LOG__traceLevelBefore <= TraceLevelVerbose, "traceLevelBefore can't be higher than TraceLevelVerbose for call to LOG_COUNTED(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      constexpr signed char LOG__traceLevelAfter = (traceLevelAfter); /* we only use traceLevelAfter once, which avoids pre and post decrement issues with macros */ \
      static_assert(TraceLevelOff < LOG__traceLevelAfter, "traceLevelAfter can't be TraceLevelOff or lower for call to LOG_COUNTED(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      static_assert(LOG__traceLevelAfter <= TraceLevelVerbose, "traceLevelAfter can't be higher than TraceLevelVerbose for call to LOG_COUNTED(pLogCount, traceLevelBefore, traceLevelAfter, pLogMessage, ...)"); \
      static_assert(LOG__traceLevelBefore < LOG__traceLevelAfter, "We only support increasing the required trace level after N iterations and it doesn't make sense to have equal values, otherwise just use LOG(..)"); \
      if(UNLIKELY(LOG__traceLevelBefore <= g_traceLevel)) { \
         constexpr size_t LOG__cArguments = std::tuple_size<decltype(std::make_tuple(__VA_ARGS__))>::value; \
         constexpr static char LOG__originalMessage[] = (pLogMessage); /* we only use pLogMessage once, which avoids pre and post decrement issues with macros */ \
         unsigned int * const LOG__pLogCountDecrement = (pLogCountDecrement); /* we only use pLogCountDecrement once, which avoids pre and post decrement issues with macros */ \
         const unsigned int LOG__logCount = *LOG__pLogCountDecrement; \
         if(0 < LOG__logCount) { \
            *LOG__pLogCountDecrement = LOG__logCount - 1; \
            assert(nullptr != g_pLogMessageFunc); \
            if(0 == LOG__cArguments) { /* if there are no arguments we might as well send the log directly without reserving stack space for vsnprintf and without log length limitations for stack allocation */ \
               (*g_pLogMessageFunc)(LOG__traceLevelBefore, LOG__originalMessage); \
            } else { \
               InteralLogWithArguments(LOG__traceLevelBefore, LOG__originalMessage, ##__VA_ARGS__); \
            } \
         } else { \
            if(UNLIKELY(LOG__traceLevelAfter <= g_traceLevel)) { \
               assert(nullptr != g_pLogMessageFunc); \
               if(0 == LOG__cArguments) { /* if there are no arguments we might as well send the log directly without reserving stack space for vsnprintf and without log length limitations for stack allocation */ \
                  (*g_pLogMessageFunc)(LOG__traceLevelAfter, LOG__originalMessage); \
               } else { \
                  InteralLogWithArguments(LOG__traceLevelAfter, LOG__originalMessage, ##__VA_ARGS__); \
               } \
            } \
         } \
      } \
      /* the "(void)0, 0" part supresses the conditional expression is constant compiler warning */ \
   } while((void)0, 0)

#ifndef NDEBUG

// the "(void)0, 0" part supresses the conditional expression is constant compiler warning
// using a do loop here gives us a nice look to the macro where the caller needs to use a semi-colon to call it, and it can be used after a single if statement without curly braces
#define EBM_ASSERT(bCondition) \
   do { \
      if(UNLIKELY(!(bCondition))) { \
         if(UNLIKELY(TraceLevelError <= g_traceLevel)) { \
            assert(nullptr != g_pLogMessageFunc); \
            InteralLogWithArguments(TraceLevelError, g_assertLogMessage, static_cast<unsigned long long>(__LINE__), __FILE__, __func__, #bCondition); \
         } \
         assert(!#bCondition); \
      } \
   } while((void)0, 0)

#else // NDEBUG
#define EBM_ASSERT(bCondition) ((void)0)
#endif // NDEBUG

#endif // LOGGING_H
