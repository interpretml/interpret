// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef BRIDGE_CPP_HPP
#define BRIDGE_CPP_HPP

#include "ebm_native.h"
#include "logging.h"
#include "common_c.h"
#include "bridge_c.h"
#include "zones.h"

#include "common_cpp.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

constexpr static ptrdiff_t k_regression = -1;
constexpr static ptrdiff_t k_dynamicClassification = 0;
constexpr static ptrdiff_t k_oneScore = 1;
INLINE_ALWAYS constexpr static bool IsRegression(const ptrdiff_t learningTypeOrCountTargetClasses) noexcept {
   return k_regression == learningTypeOrCountTargetClasses;
}
INLINE_ALWAYS constexpr static bool IsClassification(const ptrdiff_t learningTypeOrCountTargetClasses) noexcept {
   return 0 <= learningTypeOrCountTargetClasses;
}
INLINE_ALWAYS constexpr static bool IsBinaryClassification(const ptrdiff_t learningTypeOrCountTargetClasses) noexcept {
#ifdef EXPAND_BINARY_LOGITS
   return UNUSED(learningTypeOrCountTargetClasses), false;
#else // EXPAND_BINARY_LOGITS
   return 2 == learningTypeOrCountTargetClasses;
#endif // EXPAND_BINARY_LOGITS
}
INLINE_ALWAYS constexpr static bool IsMulticlass(const ptrdiff_t learningTypeOrCountTargetClasses) noexcept {
   return IsClassification(learningTypeOrCountTargetClasses) && !IsBinaryClassification(learningTypeOrCountTargetClasses);
}

INLINE_ALWAYS constexpr static size_t GetVectorLength(const ptrdiff_t learningTypeOrCountTargetClasses) noexcept {
   // this will work for anything except if learningTypeOrCountTargetClasses is set to DYNAMIC_CLASSIFICATION which means we should have passed in the 
   // dynamic value since DYNAMIC_CLASSIFICATION is a constant that doesn't tell us anything about the real value
#ifdef EXPAND_BINARY_LOGITS
   return learningTypeOrCountTargetClasses <= ptrdiff_t { 1 } ? size_t { 1 } : static_cast<size_t>(learningTypeOrCountTargetClasses);
#else // EXPAND_BINARY_LOGITS
   return learningTypeOrCountTargetClasses <= ptrdiff_t { 2 } ? size_t { 1 } : static_cast<size_t>(learningTypeOrCountTargetClasses);
#endif // EXPAND_BINARY_LOGITS
}

// THIS NEEDS TO BE A MACRO AND NOT AN INLINE FUNCTION -> an inline function will cause all the parameters to get resolved before calling the function
// We want any arguments to our macro to not get resolved if they are not needed at compile time so that we do less work if it's not needed
// This will effectively turn the variable into a compile time constant if it can be resolved at compile time
// The caller can put pTargetFeature->m_cBins inside the macro call and it will be optimize away if it isn't necessary
// having compile time counts of the target count of classes should allow for loop elimination in most cases and the restoration of SIMD instructions in
// places where you couldn't do so with variable loop iterations
#define GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(MACRO_compilerLearningTypeOrCountTargetClasses, MACRO_runtimeLearningTypeOrCountTargetClasses) \
   (k_dynamicClassification == (MACRO_compilerLearningTypeOrCountTargetClasses) ? (MACRO_runtimeLearningTypeOrCountTargetClasses) : \
   (MACRO_compilerLearningTypeOrCountTargetClasses))

// THIS NEEDS TO BE A MACRO AND NOT AN INLINE FUNCTION -> an inline function will cause all the parameters to get resolved before calling the function
// We want any arguments to our macro to not get resolved if they are not needed at compile time so that we do less work if it's not needed
// This will effectively turn the variable into a compile time constant if it can be resolved at compile time
// having compile time counts of the target count of classes should allow for loop elimination in most cases and the restoration of SIMD instructions in 
// places where you couldn't do so with variable loop iterations
// TODO: use this macro more
// TODO: do we really need the static_cast to size_t here?
#define GET_DIMENSIONS(MACRO_cCompilerDimensions, MACRO_cRuntimeDimensions) \
   (k_dynamicDimensions == (MACRO_cCompilerDimensions) ? static_cast<size_t>(MACRO_cRuntimeDimensions) : static_cast<size_t>(MACRO_cCompilerDimensions))

// THIS NEEDS TO BE A MACRO AND NOT AN INLINE FUNCTION -> an inline function will cause all the parameters to get resolved before calling the function
// We want any arguments to our macro to not get resolved if they are not needed at compile time so that we do less work if it's not needed
// This will effectively turn the variable into a compile time constant if it can be resolved at compile time
// having compile time counts of the target count of classes should allow for loop elimination in most cases and the restoration of SIMD instructions in 
// places where you couldn't do so with variable loop iterations
#define GET_ITEMS_PER_BIT_PACK(MACRO_compilerBitPack, MACRO_runtimeBitPack) \
   (k_cItemsPerBitPackDynamic2 == (MACRO_compilerBitPack) ? \
      static_cast<size_t>(MACRO_runtimeBitPack) : static_cast<size_t>(MACRO_compilerBitPack))

constexpr static size_t k_cBitsForStorageType = CountBitsRequiredPositiveMax<StorageDataType>();

INLINE_ALWAYS constexpr static size_t GetCountBits(const size_t cItemsBitPacked) noexcept {
   return k_cBitsForStorageType / cItemsBitPacked;
}

constexpr static ptrdiff_t k_cItemsPerBitPackNone = ptrdiff_t { -1 }; // this is for when there is only 1 bin
// TODO : remove the 2 suffixes from these, and verify these are being used!!  AND at the same time verify that we like the sign of anything that uses these constants size_t vs ptrdiff_t
constexpr static ptrdiff_t k_cItemsPerBitPackDynamic2 = ptrdiff_t { 0 };

} // DEFINED_ZONE_NAME

#endif // BRIDGE_CPP_HPP