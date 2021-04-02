// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef REGISTRABLE_H
#define REGISTRABLE_H

#include "zones.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct Registrable {

protected:

   Registrable() = default;

};
static_assert(std::is_standard_layout<Registrable>::value &&
   std::is_trivially_copyable<Registrable>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

} // DEFINED_ZONE_NAME

#endif // REGISTRABLE_H