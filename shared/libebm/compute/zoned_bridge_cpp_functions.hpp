// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef ZONED_BRIDGE_CPP_FUNCTIONS_HPP
#define ZONED_BRIDGE_CPP_FUNCTIONS_HPP

#include "libebm.h" // ErrorEbm
#include "zones.h"

struct ApplyUpdateBridge;

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct Objective;

// these are going to be extern "C++", which we require to call our static member functions per:
// https://www.drdobbs.com/c-theory-and-practice/184403437
typedef ErrorEbm (* APPLY_UPDATE_CPP)(const Objective * const pObjective, ApplyUpdateBridge * const pData);
typedef double (* FINISH_METRIC_CPP)(const Objective * const pObjective, const double metricSum);
typedef BoolEbm (* CHECK_TARGETS_CPP)(const Objective * const pObjective, const size_t c, const void * const aTargets);

struct FunctionPointersCpp {
   // unfortunately, function pointers are not interchangable with data pointers since in some architectures
   // they exist in separate memory regions as data, so we can't store them as void * in the Wrappers
   // https://stackoverflow.com/questions/12358843/why-are-function-pointers-and-data-pointers-incompatible-in-c-c

   APPLY_UPDATE_CPP m_pApplyUpdateCpp;
   FINISH_METRIC_CPP m_pFinishMetricCpp;
   CHECK_TARGETS_CPP m_pCheckTargetsCpp;
};

} // DEFINED_ZONE_NAME

#endif // ZONED_BRIDGE_CPP_FUNCTIONS_HPP
