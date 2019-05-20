// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef LOGGING_H
#define LOGGING_H

#include "ebmcore.h"

extern IntegerDataType g_traceLevel;
extern LOG_MESSAGE_FUNCTION g_pLogMessageFunc;
#define LOG(traceLevel, pLogMessage) ((void)((traceLevel) <= g_traceLevel ? ((*g_pLogMessageFunc)((traceLevel), (pLogMessage)), 0) : 0))

#endif // DATA_SET_INTERNAL_H
