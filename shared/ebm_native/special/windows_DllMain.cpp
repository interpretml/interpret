// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

// this file is only included in Windows builds.  We don't want to require windows.h in our precompiled header since then it will be needed in linux builds,
// which doesn't make sense
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

extern BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
   (void)(hModule); // disable unused parameter warnings
   (void)(lpReserved); // disable unused parameter warnings

   switch(ul_reason_for_call) {
   case DLL_PROCESS_ATTACH:
   case DLL_THREAD_ATTACH:
   case DLL_THREAD_DETACH:
   case DLL_PROCESS_DETACH:
      break;
   }
   return TRUE;
}
