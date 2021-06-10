// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!

#include "Loss.hpp"

// TFloat could be double, float, or some SIMD intrinsic type
template <typename TFloat>
struct LogLossBinaryMultitaskLoss : public BinaryMultitaskLoss {
   // this one would more popularily be called LossMultilabelLogLoss.  We're currently calling this
   // LogLossBinaryMultitaskLoss since it fits better into our ontology of Multitask* types having 
   // multiple targets, but consider chaning this to multilabel since it would be more widely recognized that way

   // this one needs to be special cased!
};
