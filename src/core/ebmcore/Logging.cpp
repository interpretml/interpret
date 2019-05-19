// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <assert.h>

#include "ebmcore.h" // FractionalDataType
#include "EbmInternal.h" // AttributeTypeCore

IntegerDataType g_traceLevel = 0;
LOG_MESSAGE_FUNCTION g_pLogMessageFunc = nullptr;

EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION SetLogMessageFunction(LOG_MESSAGE_FUNCTION logMessageFunction) {
   assert(nullptr != logMessageFunction);
   g_pLogMessageFunc = logMessageFunction;
}

EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION SetTraceLevel(IntegerDataType traceLevel) {
   assert(0 <= traceLevel);
   assert(nullptr != g_pLogMessageFunc);
   g_traceLevel = traceLevel;
}
