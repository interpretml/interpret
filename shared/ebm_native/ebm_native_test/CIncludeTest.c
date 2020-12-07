// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// this just tests that our include file is C compatible
#include "ebm_native.h"

// include this AFTER ebm_native.h to test that ebm_native.h can stand alone
#include <stdio.h>

static void EBM_NATIVE_CALLING_CONVENTION LogMessage(TraceEbmType traceLevel, const char * message) {
   char buffer[1000];
   const size_t cBytesBuffer = sizeof(buffer) / sizeof(buffer[0]);
   snprintf(buffer, cBytesBuffer, "%" TraceEbmTypePrintf ".%s\n", traceLevel, message);
}

extern void TestCHeaderConstructs() {
   char buffer[1000];
   const size_t cBytesBuffer = sizeof(buffer) / sizeof(buffer[0]);

   BoosterHandle boosterHandle = NULL;
   snprintf(buffer, cBytesBuffer, "%p\n", boosterHandle);

   InteractionDetectorHandle interactionDetectorHandle = NULL;
   snprintf(buffer, cBytesBuffer, "%p\n", interactionDetectorHandle);

   FloatEbmType testFloat = -123.456;
   snprintf(buffer, cBytesBuffer, "%" FloatEbmTypePrintf "\n", testFloat);

   IntEbmType testInt = -123;
   snprintf(buffer, cBytesBuffer, "%" IntEbmTypePrintf "\n", testInt);

   UIntEbmType testUInt = 123;
   snprintf(buffer, cBytesBuffer, "%" UIntEbmTypePrintf "\n", testUInt);

   SeedEbmType testSeed = -123;
   snprintf(buffer, cBytesBuffer, "%" SeedEbmTypePrintf "\n", testSeed);

   BoolEbmType testBoolTrue = EBM_TRUE;
   snprintf(buffer, cBytesBuffer, "%" BoolEbmTypePrintf "\n", testBoolTrue);

   BoolEbmType testBoolFalse = EBM_FALSE;
   snprintf(buffer, cBytesBuffer, "%" BoolEbmTypePrintf "\n", testBoolFalse);

   FeatureEbmType testFeatureOrdinal = FeatureTypeOrdinal;
   snprintf(buffer, cBytesBuffer, "%" FeatureEbmTypePrintf "\n", testFeatureOrdinal);

   FeatureEbmType testFeatureNominal = FeatureTypeNominal;
   snprintf(buffer, cBytesBuffer, "%" FeatureEbmTypePrintf "\n", testFeatureNominal);

   TraceEbmType testTraceOff = TraceLevelOff;
   snprintf(buffer, cBytesBuffer, "%" TraceEbmTypePrintf "\n", testTraceOff);

   TraceEbmType testTraceError = TraceLevelError;
   snprintf(buffer, cBytesBuffer, "%" TraceEbmTypePrintf "\n", testTraceError);

   TraceEbmType testTraceWarning = TraceLevelWarning;
   snprintf(buffer, cBytesBuffer, "%" TraceEbmTypePrintf "\n", testTraceWarning);

   TraceEbmType testTraceInfo = TraceLevelInfo;
   snprintf(buffer, cBytesBuffer, "%" TraceEbmTypePrintf "\n", testTraceInfo);

   TraceEbmType testTraceVerbose = TraceLevelVerbose;
   snprintf(buffer, cBytesBuffer, "%" TraceEbmTypePrintf "\n", testTraceVerbose);

   TraceEbmType testTraceIllegal = 9999;
   snprintf(buffer, cBytesBuffer, "%" TraceEbmTypePrintf "\n", testTraceIllegal);

   snprintf(buffer, cBytesBuffer, "%s\n", GetTraceLevelString(testTraceOff));
   snprintf(buffer, cBytesBuffer, "%s\n", GetTraceLevelString(testTraceError));
   snprintf(buffer, cBytesBuffer, "%s\n", GetTraceLevelString(testTraceWarning));
   snprintf(buffer, cBytesBuffer, "%s\n", GetTraceLevelString(testTraceInfo));
   snprintf(buffer, cBytesBuffer, "%s\n", GetTraceLevelString(testTraceVerbose));
   snprintf(buffer, cBytesBuffer, "%s\n", GetTraceLevelString(testTraceIllegal));

   LOG_MESSAGE_FUNCTION logMessageFunction = &LogMessage;
   (*logMessageFunction)(TraceLevelVerbose, "I am a test.  What are you?");
}