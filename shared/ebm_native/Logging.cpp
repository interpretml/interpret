// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <assert.h>
#include <stdio.h>
#include <stdarg.h>

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // FeatureTypeCore
#include "Logging.h"

extern const char g_assertLogMessage[] = "ASSERT ERROR on line %llu of file \"%s\" in function \"%s\" for condition \"%s\"";
constexpr static char g_pLoggingParameterError[] = "Error in vsnprintf parameters for logging.";

signed char g_traceLevel = TraceLevelOff;
LOG_MESSAGE_FUNCTION g_pLogMessageFunc = nullptr;

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION SetLogMessageFunction(LOG_MESSAGE_FUNCTION logMessageFunction) {
   assert(nullptr != logMessageFunction);
   assert(nullptr == g_pLogMessageFunc); /* "SetLogMessageFunction should only be called once" */
   g_pLogMessageFunc = logMessageFunction;
}

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION SetTraceLevel(signed char traceLevel) {
   assert(TraceLevelOff <= traceLevel);
   assert(traceLevel <= TraceLevelVerbose);
   assert(nullptr != g_pLogMessageFunc); /* "call SetLogMessageFunction before calling SetTraceLevel" */
   g_traceLevel = traceLevel;
}

WARNING_PUSH
WARNING_DISABLE_NON_LITERAL_PRINTF_STRING
extern void InteralLogWithArguments(signed char traceLevel, const char * const pOriginalMessage, ...) {
   // this function is here largely to clip the stack memory needed for messageSpace.  If we put the below functionality directly into a MACRO or an inline function then the 
   // memory gets reserved on the stack of the function which calls our logging MACRO.  The reserved memory will be held when our calling function calls any
   // children functions.  By putting the buffer insdie this purposely separated function we allocate it on the stack, then immedicately deallocate it, so our caller
   // doesn't need to hold valuable stack space all the way down when calling it's offspring functions.  We also don't need to allocate any stack when logging is turned off.

   va_list args;
   char messageSpace[1024];
   va_start(args, pOriginalMessage);
   // vsnprintf specifically says that the count parameter is in bytes of buffer space, but let's be safe and assume someone might change this to a unicode function someday 
   // and that new function might be in characters instead of bytes.  For us #bytes == #chars.  If a unicode specific version is in bytes it won't overflow, but it will waste memory
   if(vsnprintf(messageSpace, sizeof(messageSpace) / sizeof(messageSpace[0]), pOriginalMessage, args) < 0) {
      (*g_pLogMessageFunc)(traceLevel, g_pLoggingParameterError);
   } else {
      // if messageSpace overflows, we clip the message, but it's still legal
      (*g_pLogMessageFunc)(traceLevel, messageSpace);
   }
   va_end(args);
}
WARNING_POP