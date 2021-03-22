// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#define _CRT_SECURE_NO_DEPRECATE

#include <assert.h>
#include <stdio.h>
#include <stdarg.h>

#include "ebm_native.h" // FloatEbmType
#include "bridge_c.h" // INTERNAL_IMPORT_EXPORT_INCLUDE 
#include "common_c.h"
#include "Logging.h"

INTERNAL_IMPORT_EXPORT_BODY const char g_trueString[] = "true";
INTERNAL_IMPORT_EXPORT_BODY const char g_falseString[] = "false";

#ifndef NDEBUG
#define COMPILE_MODE "DEBUG"
#else // NDEBUG
#define COMPILE_MODE "RELEASE"
#endif // NDEBUG

static const char g_assertLogMessage[] = "ASSERT ERROR on line %llu of file \"%s\" in function \"%s\" for condition \"%s\"";
static const char g_pLoggingParameterError[] = "Error in vsnprintf parameters for logging.";

INTERNAL_IMPORT_EXPORT_BODY TraceEbmType g_traceLevel = TraceLevelOff;
static LOG_MESSAGE_FUNCTION g_pLogMessageFunc = NULL;

static const char g_traceOffString[] = "OFF";
static const char g_traceErrorString[] = "ERROR";
static const char g_traceWarningString[] = "WARNING";
static const char g_traceInfoString[] = "INFO";
static const char g_traceVerboseString[] = "VERBOSE";
static const char g_traceIllegalString[] = "ILLEGAL";

EBM_NATIVE_IMPORT_EXPORT_BODY const char * EBM_NATIVE_CALLING_CONVENTION GetTraceLevelString(TraceEbmType traceLevel) {
   switch(traceLevel) {
   case TraceLevelOff:
      return g_traceOffString;
   case TraceLevelError:
      return g_traceErrorString;
   case TraceLevelWarning:
      return g_traceWarningString;
   case TraceLevelInfo:
      return g_traceInfoString;
   case TraceLevelVerbose:
      return g_traceVerboseString;
   default:
      return g_traceIllegalString;
   }
}

// TODO: combine SetLogMessageFunction and SetTraceLevel and verify logMessageFunction hasn't changed.  if set to something new turn off logging!

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION SetLogMessageFunction(LOG_MESSAGE_FUNCTION logMessageFunction) {
   assert(NULL != logMessageFunction);
   assert(NULL == g_pLogMessageFunc); /* "SetLogMessageFunction should only be called once" */
   assert(TraceLevelOff == g_traceLevel);

   g_pLogMessageFunc = logMessageFunction;
}

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION SetTraceLevel(TraceEbmType traceLevel) {
   if(traceLevel < TraceLevelOff || TraceLevelVerbose < traceLevel || NULL == g_pLogMessageFunc) {
      // call SetLogMessageFunction before calling SetTraceLevel unless we're keeping tracing off
      g_traceLevel = TraceLevelOff;
   } else {
      g_traceLevel = traceLevel;

      // this is not an actual error, but ensure that this message gets written to the log so that we know it was properly
      // set, and also test that the callback function works at this early stage instead of waiting for a real error
      const TraceEbmType LOG__traceLevel = TraceLevelWarning;
      if(UNLIKELY(LOG__traceLevel <= traceLevel)) {
         const static char LOG__originalMessage[] = "Native logging trace level set to %s in " COMPILE_MODE;
         InteralLogWithArguments(LOG__traceLevel, LOG__originalMessage, GetTraceLevelString(traceLevel));
      }
   }
}

WARNING_PUSH
WARNING_DISABLE_NON_LITERAL_PRINTF_STRING
INTERNAL_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION InteralLogWithArguments(const TraceEbmType traceLevel, const char * const pOriginalMessage, ...) {
   assert(NULL != g_pLogMessageFunc);

   // this function is here largely to clip the stack memory needed for messageSpace.  If we put the below functionality directly into a MACRO or an 
   // inline function then the memory gets reserved on the stack of the function which calls our logging MACRO.  The reserved memory will be held when 
   // our calling function calls any children functions.  By putting the buffer insdie this purposely separated function we allocate it on the stack, 
   // then immedicately deallocate it, so our caller doesn't need to hold valuable stack space all the way down when calling it's offspring functions.  
   // We also don't need to allocate any stack when logging is turned off.

   va_list args;
   char messageSpace[1024];
   va_start(args, pOriginalMessage);
   // vsnprintf specifically says that the count parameter is in bytes of buffer space, but let's be safe and assume someone might change this to a 
   // unicode function someday and that new function might be in characters instead of bytes.  For us #bytes == #chars.  If a unicode specific version 
   // is in bytes it won't overflow, but it will waste memory

   // clang-tidy says va_list is uninitialized, despite the call to va_start above. This is a known bug in clang-tidy.
   // DETAILS: https://stackoverflow.com/questions/58672959/why-does-clang-tidy-say-vsnprintf-has-an-uninitialized-va-list-argument
   StopClangAnalysis();
   if(vsnprintf(messageSpace, sizeof(messageSpace) / sizeof(messageSpace[0]), pOriginalMessage, args) < 0) {
      (*g_pLogMessageFunc)(traceLevel, g_pLoggingParameterError);
   } else {
      // if messageSpace overflows, we clip the message, but it's still legal
      (*g_pLogMessageFunc)(traceLevel, messageSpace);
   }
   va_end(args);
}
WARNING_POP

INTERNAL_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION InteralLogWithoutArguments(const TraceEbmType traceLevel, const char * const pOriginalMessage) {
   assert(NULL != g_pLogMessageFunc);
   (*g_pLogMessageFunc)(traceLevel, pOriginalMessage);
}

INTERNAL_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION LogAssertFailure(
   const unsigned long long lineNumber, 
   const char * const fileName, 
   const char * const functionName, 
   const char * const assertText
) ANALYZER_NORETURN {
   if(UNLIKELY(TraceLevelError <= g_traceLevel)) {
      InteralLogWithArguments(TraceLevelError, g_assertLogMessage, lineNumber, fileName, functionName, assertText);
   }
}
