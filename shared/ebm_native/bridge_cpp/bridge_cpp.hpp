// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef BRIDGE_CPP_HPP
#define BRIDGE_CPP_HPP

#include "ebm_native.h" // bridge_c.h depends on ebm_native.h and we probably will eventually too
#include "logging.h"
#include "bridge_c.h" // StorageDataType
#include "zones.h"

#include "common_cpp.hpp" // CountBitsRequiredPositiveMax
#include "Bin.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

static constexpr ptrdiff_t k_regression = -1;
static constexpr ptrdiff_t k_dynamicClassification = 0;
static constexpr ptrdiff_t k_oneScore = 1;
inline constexpr static bool IsRegression(const ptrdiff_t cClasses) noexcept {
   return k_regression == cClasses;
}
inline constexpr static bool IsClassification(const ptrdiff_t cClasses) noexcept {
   return 0 <= cClasses;
}
inline constexpr static bool IsBinaryClassification(const ptrdiff_t cClasses) noexcept {
#ifdef EXPAND_BINARY_LOGITS
   return UNUSED(cClasses), false;
#else // EXPAND_BINARY_LOGITS
   return 2 == cClasses;
#endif // EXPAND_BINARY_LOGITS
}
inline constexpr static bool IsMulticlass(const ptrdiff_t cClasses) noexcept {
   return IsClassification(cClasses) && !IsBinaryClassification(cClasses);
}

inline constexpr static size_t GetCountScores(const ptrdiff_t cClasses) noexcept {
   // this will work for anything except if cClasses is set to DYNAMIC_CLASSIFICATION which means we should have passed in the 
   // dynamic value since DYNAMIC_CLASSIFICATION is a constant that doesn't tell us anything about the real value
#ifdef EXPAND_BINARY_LOGITS
   return cClasses <= ptrdiff_t { 1 } ? size_t { 1 } : static_cast<size_t>(cClasses);
#else // EXPAND_BINARY_LOGITS
   return cClasses <= ptrdiff_t { 2 } ? size_t { 1 } : static_cast<size_t>(cClasses);
#endif // EXPAND_BINARY_LOGITS
}

// THIS NEEDS TO BE A MACRO AND NOT AN INLINE FUNCTION -> an inline function will cause all the parameters to get resolved before calling the function
// We want any arguments to our macro to not get resolved if they are not needed at compile time so that we do less work if it's not needed
// This will effectively turn the variable into a compile time constant if it can be resolved at compile time
// The caller can put pTargetFeature->m_cBins inside the macro call and it will be optimize away if it isn't necessary
// having compile time counts of the target count of classes should allow for loop elimination in most cases and the restoration of SIMD instructions in
// places where you couldn't do so with variable loop iterations
#define GET_COUNT_CLASSES(MACRO_cCompilerClasses, MACRO_cRuntimeClasses) \
   (k_dynamicClassification == (MACRO_cCompilerClasses) ? (MACRO_cRuntimeClasses) : \
   (MACRO_cCompilerClasses))

// THIS NEEDS TO BE A MACRO AND NOT AN INLINE FUNCTION -> an inline function will cause all the parameters to get resolved before calling the function
// We want any arguments to our macro to not get resolved if they are not needed at compile time so that we do less work if it's not needed
// This will effectively turn the variable into a compile time constant if it can be resolved at compile time
// having compile time counts of the target count of classes should allow for loop elimination in most cases and the restoration of SIMD instructions in 
// places where you couldn't do so with variable loop iterations
// TODO: use this macro more
// TODO: do we really need the static_cast to size_t here?
#define GET_COUNT_DIMENSIONS(MACRO_cCompilerDimensions, MACRO_cRuntimeDimensions) \
   (k_dynamicDimensions == (MACRO_cCompilerDimensions) ? static_cast<size_t>(MACRO_cRuntimeDimensions) : static_cast<size_t>(MACRO_cCompilerDimensions))

// THIS NEEDS TO BE A MACRO AND NOT AN INLINE FUNCTION -> an inline function will cause all the parameters to get resolved before calling the function
// We want any arguments to our macro to not get resolved if they are not needed at compile time so that we do less work if it's not needed
// This will effectively turn the variable into a compile time constant if it can be resolved at compile time
// having compile time counts of the target count of classes should allow for loop elimination in most cases and the restoration of SIMD instructions in 
// places where you couldn't do so with variable loop iterations
#define GET_ITEMS_PER_BIT_PACK(MACRO_compilerBitPack, MACRO_runtimeBitPack) \
   (k_cItemsPerBitPackDynamic2 == (MACRO_compilerBitPack) ? (MACRO_runtimeBitPack) : (MACRO_compilerBitPack))

static constexpr size_t k_cBitsForStorageType = CountBitsRequiredPositiveMax<StorageDataType>();

template<typename T>
inline static size_t GetCountItemsBitPacked(const size_t cBits) noexcept {
   static_assert(std::is_unsigned<T>::value, "T must be unsigned");
   EBM_ASSERT(size_t { 1 } <= cBits);
   EBM_ASSERT(cBits <= CountBitsRequiredPositiveMax<T>());
   return CountBitsRequiredPositiveMax<T>() / cBits;
}
template<typename T>
inline constexpr static size_t GetCountBits(const size_t cItemsBitPacked) noexcept {
   static_assert(std::is_unsigned<T>::value, "T must be unsigned");
   return CountBitsRequiredPositiveMax<T>() / cItemsBitPacked;
}
template<typename T>
inline constexpr static T MakeLowMask(const size_t cBits) noexcept {
   static_assert(std::is_unsigned<T>::value, "T must be unsigned");
   return (~T { 0 }) >> (CountBitsRequiredPositiveMax<T>() - cBits);
}

static constexpr ptrdiff_t k_cItemsPerBitPackNone = ptrdiff_t { -1 }; // this is for when there is only 1 bin
// TODO : remove the 2 suffixes from these, and verify these are being used!!  AND at the same time verify that we like the sign of anything that uses these constants size_t vs ptrdiff_t
static constexpr ptrdiff_t k_cItemsPerBitPackDynamic2 = ptrdiff_t { 0 };

struct BinSumsBoostingBridge {
   ptrdiff_t m_cClasses;
   ptrdiff_t m_cPack;

   size_t m_cSamples;
   const FloatFast * m_aGradientsAndHessians;
   const FloatFast * m_aWeights;
   const size_t * m_pCountOccurrences;
   const StorageDataType * m_aPacked;

   BinBase * m_aFastBins;

#ifndef NDEBUG
   const BinBase * m_pDebugFastBinsEnd;
   FloatFast m_totalWeightDebug;
#endif // NDEBUG
};

struct BinSumsInteractionBridge {
   ptrdiff_t m_cClasses;

   size_t m_cSamples;
   const FloatFast * m_aGradientsAndHessians;
   const FloatFast * m_aWeights;

   size_t m_cRuntimeRealDimensions;
   size_t m_acBins[k_cDimensionsMax];
   size_t m_acItemsPerBitPack[k_cDimensionsMax];
   const StorageDataType * m_aaPacked[k_cDimensionsMax];

   BinBase * m_aFastBins;

#ifndef NDEBUG
   const BinBase * m_pDebugFastBinsEnd;
   FloatFast m_totalWeightDebug;
#endif // NDEBUG
};

} // DEFINED_ZONE_NAME

#endif // BRIDGE_CPP_HPP