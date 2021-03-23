// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <ebm@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "zones.h"
#include "common_c.h"
#include "bridge_c.h"

extern "C" void SafeToCallThisOutsideNamespace() {
   return;
}

// we use DEFINED_ZONE_NAME in order to give the contents below separate names to the compiler and
// avoid very very very bad "one definition rule" violations which are nasty undefined behavior violations
namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class TestClass {
public:
   inline TestClass() {
      SafeToCallThisOutsideNamespace();
   }
};

void SafeToMakeThisOutsideNamespaceCpp() {
   TestClass testClass;
   return;
}

} // DEFINED_ZONE_NAME

extern "C" void SafeToMakeThisOutsideNamespaceC() {
   DEFINED_ZONE_NAME::SafeToMakeThisOutsideNamespaceCpp();
   return;
}
