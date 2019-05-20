// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <assert.h>

#include "ebmcore.h" // FractionalDataType
#include "EbmInternal.h" // AttributeTypeCore

signed char g_traceLevel = 0;
LOG_MESSAGE_FUNCTION g_pLogMessageFunc = nullptr;

EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION SetLogMessageFunction(LOG_MESSAGE_FUNCTION logMessageFunction) {
   assert(nullptr != logMessageFunction);
   assert(nullptr == g_pLogMessageFunc); /* "SetLogMessageFunction should only be called once" */
   g_pLogMessageFunc = logMessageFunction;
}

EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION SetTraceLevel(signed char traceLevel) {
   assert(TraceLevelOff <= traceLevel);
   assert(traceLevel <= TraceLevelDebug);
   assert(nullptr != g_pLogMessageFunc); /* "call SetLogMessageFunction before calling SetTraceLevel" */
   g_traceLevel = traceLevel;
}
