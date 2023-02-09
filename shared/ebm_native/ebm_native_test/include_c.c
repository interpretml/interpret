// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// this just tests that our include file is C compatible
#include "ebm_native.h"

// include this AFTER ebm_native.h to test that ebm_native.h can stand alone
#include <stdio.h>

static void EBM_CALLING_CONVENTION LogCallback(TraceEbm traceLevel, const char * message) {
   char buffer[1000];
   const size_t cBytesBuffer = sizeof(buffer) / sizeof(buffer[0]);
   snprintf(buffer, cBytesBuffer, "%" TraceEbmPrintf ".%s\n", traceLevel, message);
}

extern void TestCHeaderConstructs() {
   char buffer[1000];
   const size_t cBytesBuffer = sizeof(buffer) / sizeof(buffer[0]);

   BoosterHandle boosterHandle = NULL;
   snprintf(buffer, cBytesBuffer, "%p\n", boosterHandle);

   InteractionHandle interactionHandle = NULL;
   snprintf(buffer, cBytesBuffer, "%p\n", interactionHandle);

   IntEbm testInt = -123;
   snprintf(buffer, cBytesBuffer, "%" IntEbmPrintf "\n", testInt);

   UIntEbm testUInt = 123;
   snprintf(buffer, cBytesBuffer, "%" UIntEbmPrintf "\n", testUInt);

   SeedEbm testSeed = -123;
   snprintf(buffer, cBytesBuffer, "%" SeedEbmPrintf "\n", testSeed);

   BoolEbm testBoolTrue = EBM_TRUE;
   snprintf(buffer, cBytesBuffer, "%" BoolEbmPrintf "\n", testBoolTrue);

   BoolEbm testBoolFalse = EBM_FALSE;
   snprintf(buffer, cBytesBuffer, "%" BoolEbmPrintf "\n", testBoolFalse);

   TraceEbm testTraceOff = Trace_Off;
   snprintf(buffer, cBytesBuffer, "%" TraceEbmPrintf "\n", testTraceOff);

   TraceEbm testTraceError = Trace_Error;
   snprintf(buffer, cBytesBuffer, "%" TraceEbmPrintf "\n", testTraceError);

   TraceEbm testTraceWarning = Trace_Warning;
   snprintf(buffer, cBytesBuffer, "%" TraceEbmPrintf "\n", testTraceWarning);

   TraceEbm testTraceInfo = Trace_Info;
   snprintf(buffer, cBytesBuffer, "%" TraceEbmPrintf "\n", testTraceInfo);

   TraceEbm testTraceVerbose = Trace_Verbose;
   snprintf(buffer, cBytesBuffer, "%" TraceEbmPrintf "\n", testTraceVerbose);

   TraceEbm testTraceIllegal = 9999;
   snprintf(buffer, cBytesBuffer, "%" TraceEbmPrintf "\n", testTraceIllegal);

   snprintf(buffer, cBytesBuffer, "%s\n", GetTraceLevelString(testTraceOff));
   snprintf(buffer, cBytesBuffer, "%s\n", GetTraceLevelString(testTraceError));
   snprintf(buffer, cBytesBuffer, "%s\n", GetTraceLevelString(testTraceWarning));
   snprintf(buffer, cBytesBuffer, "%s\n", GetTraceLevelString(testTraceInfo));
   snprintf(buffer, cBytesBuffer, "%s\n", GetTraceLevelString(testTraceVerbose));
   snprintf(buffer, cBytesBuffer, "%s\n", GetTraceLevelString(testTraceIllegal));

   LogCallbackFunction logCallbackFunction = &LogCallback;
   (*logCallbackFunction)(Trace_Verbose, "I am a test.  What are you?");
}