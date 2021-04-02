// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef ZONED_BRIDGE_CPP_FUNCTIONS_HPP
#define ZONED_BRIDGE_CPP_FUNCTIONS_HPP

#include "ebm_native.h"
#include "bridge_c.h"
#include "zones.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct Loss;

// these are going to be extern "C++", which we require to call our static member functions per:
// https://www.drdobbs.com/c-theory-and-practice/184403437
typedef ErrorEbmType (* APPLY_TRAINING_CPP)(const Loss * const pLoss, ApplyTrainingData * const pData);
typedef ErrorEbmType (* APPLY_VALIDATION_CPP)(const Loss * const pLoss, ApplyValidationData * const pData);

} // DEFINED_ZONE_NAME

#endif // ZONED_BRIDGE_CPP_FUNCTIONS_HPP
