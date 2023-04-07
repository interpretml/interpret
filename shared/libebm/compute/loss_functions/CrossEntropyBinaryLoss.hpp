// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!

// DO NOT INCLUDE ANY FILES IN THIS FILE. THEY WILL NOT BE ZONED PROPERLY

// TFloat could be double, float, or some SIMD intrinsic type
template <typename TFloat>
struct CrossEntropyBinaryLoss : public BinaryLoss {
   // this one needs to be special cased!
};
