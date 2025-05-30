// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// TFloat could be double, float, or some SIMD intrinsic type
template<typename TFloat> struct LogLossBinaryMultitaskObjective : BinaryMultitaskObjective {
   // this one would more popularily be called LogLossMultilabelObjective.  We're currently calling this
   // LogLossBinaryMultitaskObjective since it fits better into our ontology of Multitask* types having
   // multiple targets, but consider chaning this to multilabel since it would be more widely recognized that way

   // this one needs to be special cased!
};
