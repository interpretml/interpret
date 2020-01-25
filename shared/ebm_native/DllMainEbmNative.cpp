// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

// this file is only included in Windows builds.  We don't want to require windows.h in our precompiled header since then it will be needed in linux builds,
// which doesn't make sense
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#pragma warning(suppress : 4100)
BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
   switch(ul_reason_for_call) {
   case DLL_PROCESS_ATTACH:
   case DLL_THREAD_ATTACH:
   case DLL_THREAD_DETACH:
   case DLL_PROCESS_DETACH:
      break;
   }
   return TRUE;
}
