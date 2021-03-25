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

class Registrable {

protected:

   Registrable() = default;

public:
   virtual ~Registrable() = default;
};

} // DEFINED_ZONE_NAME

#endif // REGISTRABLE_H