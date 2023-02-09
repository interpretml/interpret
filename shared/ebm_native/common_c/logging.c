// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#define _CRT_SECURE_NO_DEPRECATE

#include <assert.h>
#include <stdio.h>
#include <stdarg.h>

#include "logging.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

const char g_sTrue[] = "true";
const char g_sFalse[] = "false";

static const char g_sAssertLogMessage[] = "ASSERT ERROR on line %llu of file \"%s\" in function \"%s\" for condition \"%s\"";
static const char g_sLoggingParamError[] = "Error in vsnprintf parameters for logging.";

TraceEbm g_traceLevel = Trace_Off;
static LogCallbackFunction g_pLogCallbackFunction = NULL;

static const char g_sTraceOff[] = "OFF";
static const char g_sTraceError[] = "ERROR";
static const char g_sTraceWarning[] = "WARNING";
static const char g_sTraceInfo[] = "INFO";
static const char g_sTraceVerbose[] = "VERBOSE";
static const char g_sTraceIllegal[] = "ILLEGAL";

EBM_API_BODY const char * EBM_CALLING_CONVENTION GetTraceLevelString(TraceEbm traceLevel) {
   switch(traceLevel) {
   case Trace_Off:
      return g_sTraceOff;
   case Trace_Error:
      return g_sTraceError;
   case Trace_Warning:
      return g_sTraceWarning;
   case Trace_Info:
      return g_sTraceInfo;
   case Trace_Verbose:
      return g_sTraceVerbose;
   default:
      return g_sTraceIllegal;
   }
}

EBM_API_BODY void EBM_CALLING_CONVENTION SetLogCallback(LogCallbackFunction logCallbackFunction) {
   assert(NULL != logCallbackFunction);
   assert(NULL == g_pLogCallbackFunction); /* SetLogCallback should only be called once */
   assert(Trace_Off == g_traceLevel);

   g_pLogCallbackFunction = logCallbackFunction;
}

#ifndef NDEBUG
#define COMPILE_MODE "DEBUG"
#else // NDEBUG
#define COMPILE_MODE "RELEASE"
#endif // NDEBUG

static const char sStartLogOff[] = "Native logging set to OFF in " COMPILE_MODE " build.";
static const char sStartLogError[] = "Native logging set to ERROR in " COMPILE_MODE " build.";
static const char sStartLogWarning[] = "Native logging set to WARNING in " COMPILE_MODE " build.";
static const char sStartLogInfo[] = "Native logging set to INFO in " COMPILE_MODE " build.";
static const char sStartLogVerbose[] = "Native logging set to VERBOSE in " COMPILE_MODE " build.";
static const char sStartLogIllegal[] = "Native logging set to ILLEGAL in " COMPILE_MODE " build.";

EBM_API_BODY void EBM_CALLING_CONVENTION SetTraceLevel(TraceEbm traceLevel) {
   const char * sMessage;
   switch(traceLevel) {
   case Trace_Off:
      // if the previous logging level allows us to log a message, then do it before turning logging off
      sMessage = Trace_Off == g_traceLevel ? NULL : sStartLogOff;
      break;
   case Trace_Error:
      sMessage = sStartLogError;
      break;
   case Trace_Warning:
      sMessage = sStartLogWarning;
      break;
   case Trace_Info:
      sMessage = sStartLogInfo;
      break;
   case Trace_Verbose:
      sMessage = sStartLogVerbose;
      break;
   default:
      // if the previous logging level allows us to log a message, then do it before turning logging off
      sMessage = Trace_Off == g_traceLevel ? NULL : sStartLogIllegal;
      traceLevel = Trace_Off;
   }

   if(NULL == g_pLogCallbackFunction) {
      assert(Trace_Off == traceLevel && Trace_Off == g_traceLevel);
      traceLevel = Trace_Off;
      sMessage = NULL;
   }

   if(g_traceLevel < traceLevel) {
      // if the new logging level is more permissive, then set it now. log level is an observable within the 
      // log callback even though the log callback shouldn't re-enter, but we are not very trusting.
      g_traceLevel = traceLevel;
   }

   if(NULL != sMessage) {
      // log as an error message to guarantee a starting message is recorded even though this is not an error
      InteralLogWithoutArguments(Trace_Error, sMessage);
   }

   g_traceLevel = traceLevel;
}

extern void InteralLogWithArguments(const TraceEbm traceLevel, const char * const sMessage, ...) {
   assert(NULL != g_pLogCallbackFunction);
   // it is illegal for g_pLogCallbackFunction to be NULL at this point, but in the interest of not crashing check it
   if(NULL != g_pLogCallbackFunction) {
      // this function is here largely to clip the stack memory needed for messageSpace.  If we put the below functionality directly into a MACRO or an 
      // inline function then the memory gets reserved on the stack of the function which calls our logging MACRO.  The reserved memory will be held when 
      // our calling function calls any children functions.  By putting the buffer insdie this purposely separated function we allocate it on the stack, 
      // then immedicately deallocate it, so our caller doesn't need to hold valuable stack space all the way down when calling it's offspring functions.  
      // We also don't need to allocate any stack when logging is turned off.

      va_list args;
      char messageSpace[1024];
      va_start(args, sMessage);
      // vsnprintf specifically says that the count parameter is in bytes of buffer space, but let's be safe and assume someone might change this to a 
      // unicode function someday and that new function might be in characters instead of bytes.  For us #bytes == #chars.  If a unicode specific version 
      // is in bytes it won't overflow, but it will waste memory

      // turn off clang-tidy warning about insecurity of vsnprintf
      // NOLINTNEXTLINE
      if(vsnprintf(messageSpace, sizeof(messageSpace) / sizeof(messageSpace[0]), sMessage, args) < 0) {
         (*g_pLogCallbackFunction)(traceLevel, g_sLoggingParamError);
      } else {
         // if messageSpace overflows, we clip the message, but it's still legal
         (*g_pLogCallbackFunction)(traceLevel, messageSpace);
      }
      va_end(args);
   }
}

extern void InteralLogWithoutArguments(const TraceEbm traceLevel, const char * const sMessage) {
   assert(NULL != g_pLogCallbackFunction);
   // it is illegal for g_pLogCallbackFunction to be NULL at this point, but in the interest of not crashing check it
   if(NULL != g_pLogCallbackFunction) {
      (*g_pLogCallbackFunction)(traceLevel, sMessage);
   }
}

extern void LogAssertFailure(
   const unsigned long long lineNumber,
   const char * const sFileName,
   const char * const sFunctionName,
   const char * const sAssertText
) LOGGING_ANALYZER_NORETURN {
   if(Trace_Error <= g_traceLevel) {
      InteralLogWithArguments(Trace_Error, g_sAssertLogMessage, lineNumber, sFileName, sFunctionName, sAssertText);
   }
}

#ifdef __cplusplus
}
#endif // __cplusplus
