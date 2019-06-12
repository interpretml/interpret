// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef LOGGING_H
#define LOGGING_H

#include <assert.h>
#include <tuple>

#include "ebmcore.h" // LOG_MESSAGE_FUNCTION

extern signed char g_traceLevel;
extern LOG_MESSAGE_FUNCTION g_pLogMessageFunc;
extern void InteralLogWithArguments(signed char traceLevel, const char * const pOriginalMessage, ...);

#define LOG(traceLevel, pLogMessage, ...) \
   do { /* using a do loop here gives us a nice look to the macro where the caller needs to use a semi-colon to call it, and it can be used after a single if statement without curly braces */ \
      constexpr signed char LOG__traceLevel = (traceLevel); /* we only use traceLevel once, which avoids pre and post decrement issues with macros */ \
      assert(TraceLevelOff < LOG__traceLevel); /* , "traceLevel can't be TraceLevelOff or lower for call to LOG(traceLevel, pLogMessage, ...)" */ \
      assert(LOG__traceLevel <= TraceLevelVerbose); /* "traceLevel can't be higher than TraceLevelDebug for call to LOG(traceLevel, pLogMessage, ...)" */ \
      if(LOG__traceLevel <= g_traceLevel) { \
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

#endif // LOGGING_H
