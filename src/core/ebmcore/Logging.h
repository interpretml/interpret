// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef LOGGING_H
#define LOGGING_H

#include <assert.h>
#include <tuple>
#include <stdio.h>

#include "ebmcore.h"

#define LOG_MESSAGE_BUFFER_SIZE_MAX 1024

extern signed char g_traceLevel;
extern LOG_MESSAGE_FUNCTION g_pLogMessageFunc;
constexpr static char g_pLoggingParameterError[] = "Error in snprintf parameters for logging.";

#define LOG(traceLevel, pLogMessage, ...) \
   do { /* using a do loop here gives us a nice look to the macro where the caller needs to use a semi-colon to call it, and it can be used after a single if statement without curly braces */ \
      constexpr signed char LOG__traceLevel = (traceLevel); /* we only use traceLevel once, which avoids pre and post decrement issues with macros */ \
      assert(TraceLevelOff < LOG__traceLevel); /* , "traceLevel can't be TraceLevelOff or lower for call to LOG(traceLevel, pLogMessage, ...)" */ \
      assert(LOG__traceLevel <= TraceLevelVerbose); /* "traceLevel can't be higher than TraceLevelDebug for call to LOG(traceLevel, pLogMessage, ...)" */ \
      if(LOG__traceLevel <= g_traceLevel) { \
         constexpr size_t LOG__cArguments = std::tuple_size<decltype(std::make_tuple(__VA_ARGS__))>::value; \
         constexpr static char LOG__originalMessage[] = (pLogMessage); /* we only use pLogMessage once, which avoids pre and post decrement issues with macros */ \
         if(0 == LOG__cArguments) { \
            (*g_pLogMessageFunc)(LOG__traceLevel, LOG__originalMessage); \
         } else { \
            char LOG__messageSpace[LOG_MESSAGE_BUFFER_SIZE_MAX]; \
            /* snprintf specifically says that the count parameter is in bytes of buffer space, but let's be safe and assume someone might change this to a unicode function someday and that new function might be in characters instead of bytes.  For us #bytes == #chars */ \
            if(snprintf(LOG__messageSpace, sizeof(LOG__messageSpace) / sizeof(LOG__messageSpace[0]), LOG__originalMessage, ##__VA_ARGS__) < 0) { \
               (*g_pLogMessageFunc)(LOG__traceLevel, g_pLoggingParameterError); \
            } else { \
               /* if LOG__messageSpace overflows, we just clip the message */ \
               (*g_pLogMessageFunc)(LOG__traceLevel, LOG__messageSpace); \
            } \
         } \
      } \
      /* the "(void)0, 0" part supresses the conditional expression is constant compiler warning */ \
   } while((void)0, 0)

#endif // LOGGING_H
