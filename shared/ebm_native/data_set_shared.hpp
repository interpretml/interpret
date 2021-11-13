// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_SHARED_HPP
#define DATA_SET_SHARED_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

typedef UIntEbmType SharedStorageDataType;

struct SparseFeatureDataSetSharedEntry {
   SharedStorageDataType m_iSample;
   SharedStorageDataType m_nonDefaultValue;
};
static_assert(std::is_standard_layout<SparseFeatureDataSetSharedEntry>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<SparseFeatureDataSetSharedEntry>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");

extern ErrorEbmType GetDataSetSharedHeader(
   const unsigned char * const pDataSetShared,
   size_t * const pcSamplesOut,
   size_t * const pcFeaturesOut,
   size_t * const pcWeightsOut,
   size_t * const pcTargetsOut
);

// GetDataSetSharedFeature will return either SparseFeatureDataSetSharedEntry or SharedStorageDataType
extern const void * GetDataSetSharedFeature(
   const unsigned char * const pDataSetShared,
   const size_t iFeature,
   size_t * const pcBinsOut,
   bool * const pbNominalOut,
   bool * const pbSparseOut,
   SharedStorageDataType * const pDefaultValueSparseOut,
   size_t * const pcNonDefaultsSparseOut
);

extern const FloatEbmType * GetDataSetSharedWeight(
   const unsigned char * const pDataSetShared,
   const size_t iWeight
);

extern const void * GetDataSetSharedTarget(
   const unsigned char * const pDataSetShared,
   const size_t iTarget,
   ptrdiff_t * const pRuntimeLearningTypeOrCountTargetClassesOut
);

} // DEFINED_ZONE_NAME

#endif // DATA_SET_SHARED_HPP
