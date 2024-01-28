// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new objective in C++ follow the steps at the top of the "objective_registrations.hpp" file !!

// TFloat could be double, float, or some SIMD intrinsic type
template<typename TFloat> struct RmseRegressionMultitaskObjective : RegressionMultitaskObjective {
   // this one needs to be special cased!
};
