// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATASET_SHARED_HPP
#define DATASET_SHARED_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "libebm.h" // UIntEbm

#include "bridge.h" // FloatShared

#include "ebm_internal.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

typedef UIntEbm UIntShared;
struct SparseFeatureDataSetSharedEntry {
   UIntShared m_iSample;
   UIntShared m_nonDefaultVal;
};
static_assert(std::is_standard_layout<SparseFeatureDataSetSharedEntry>::value,
      "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<SparseFeatureDataSetSharedEntry>::value,
      "These structs are shared between processes, so they definetly need to be standard layout and trivial");

extern ErrorEbm GetDataSetSharedHeader(const unsigned char* const pDataSetShared,
      UIntShared* const pcSamplesOut,
      size_t* const pcFeaturesOut,
      size_t* const pcWeightsOut,
      size_t* const pcTargetsOut);

// GetDataSetSharedFeature will return either (SparseFeatureDataSetSharedEntry *) or (UIntShared *)
extern const void* GetDataSetSharedFeature(const unsigned char* const pDataSetShared,
      const size_t iFeature,
      bool* const pbMissingOut,
      bool* const pbUnknownOut,
      bool* const pbNominalOut,
      bool* const pbSparseOut,
      UIntShared* const pcBinsOut,
      UIntShared* const pDefaultValSparseOut,
      size_t* const pcNonDefaultsSparseOut);

extern const FloatShared* GetDataSetSharedWeight(const unsigned char* const pDataSetShared, const size_t iWeight);

// GetDataSetSharedTarget returns (FloatShared *) for regression and (UIntShared *) for classification
extern const void* GetDataSetSharedTarget(
      const unsigned char* const pDataSetShared, const size_t iTarget, ptrdiff_t* const pcClassesOut);

} // namespace DEFINED_ZONE_NAME

#endif // DATASET_SHARED_HPP
