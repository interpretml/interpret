// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !!! NOTE: To add a new loss/objective function in C++, follow the steps listed at the top of the "Loss.cpp" file !!!

#include "Loss.h"

struct LossMulticlassCrossEntropy : public LossMulticlass {
   // this one needs to be special cased!
};
