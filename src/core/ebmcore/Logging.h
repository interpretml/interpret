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

// use a MACRO for LOG(..) and LOG_COUNTED(..) instead of an inline because:
//   1) we can use static_assert on the log level
//   2) we can get the number of variadic arguments at compile time, allowing the call to InteralLogWithArguments to be optimized away in cases where we have a simple string
//   3) we can put our input string into a constexpr static char LOG__originalMessage[] instead of keeping it as a string, which is useful in some languages because I think strings are not const, so by default get put into the read/write data segment instead of readonly
//   3) our variadic arguments won't be evaluated unless they are necessary (log level is set high enough).  If we had inlined them, they might have needed to be evaluated, depending on the inputs
#define LOG(traceLevel, pLogMessage, ...) \
   do { /* using a do loop here gives us a nice look to the macro where the caller needs to use a semi-colon to call it, and it can be used after a single if statement without curly braces */ \
      constexpr signed char LOG__traceLevel = (traceLevel); /* we only use traceLevel once, which avoids pre and post decrement issues with macros */ \
      static_assert(TraceLevelOff < LOG__traceLevel, "traceLevel can't be TraceLevelOff or lower for call to LOG(traceLevel, pLogMessage, ...)"); \
      static_assert(LOG__traceLevel <= TraceLevelVerbose, "traceLevel can't be higher than TraceLevelDebug for call to LOG(traceLevel, pLogMessage, ...)"); \
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

#define LOG_COUNTED(pLogCount, traceLevelBefore, pLogMessage, ...) \
   do { /* using a do loop here gives us a nice look to the macro where the caller needs to use a semi-colon to call it, and it can be used after a single if statement without curly braces */ \
      constexpr signed char LOG__traceLevel = (traceLevel); /* we only use traceLevel once, which avoids pre and post decrement issues with macros */ \
      static_assert(TraceLevelOff < LOG__traceLevel, "traceLevel can't be TraceLevelOff or lower for call to LOG(traceLevel, pLogMessage, ...)"); \
      static_assert(LOG__traceLevel <= TraceLevelVerbose, "traceLevel can't be higher than TraceLevelDebug for call to LOG(traceLevel, pLogMessage, ...)"); \
      if(UNLIKELY(LOG__traceLevel <= g_traceLevel)) { \
         unsigned int * const LOG__pLogCount = (pLogCount); \
         if(0 < *LOG__pLogCount) { \
            --(*LOG__pLogCount); \
            assert(nullptr != g_pLogMessageFunc); \
            constexpr size_t LOG__cArguments = std::tuple_size<decltype(std::make_tuple(__VA_ARGS__))>::value; \
            constexpr static char LOG__originalMessage[] = (pLogMessage); /* we only use pLogMessage once, which avoids pre and post decrement issues with macros */ \
            if(0 == LOG__cArguments) { /* if there are no arguments we might as well send the log directly without reserving stack space for vsnprintf and without log length limitations for stack allocation */ \
               (*g_pLogMessageFunc)(LOG__traceLevel, LOG__originalMessage); \
            } else { \
               InteralLogWithArguments(LOG__traceLevel, LOG__originalMessage, ##__VA_ARGS__); \
            } \
         } \
      } \
      /* the "(void)0, 0" part supresses the conditional expression is constant compiler warning */ \
   } while((void)0, 0)

#endif // LOGGING_H
