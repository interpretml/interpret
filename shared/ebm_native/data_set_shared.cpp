// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"
#include "data_set_shared.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// TODO PK Implement the following to speed boosting and generating the gradients/hessians in interaction detection:
//   - OBSERVATION: We want sparse feature support in our booster since we don't need to access
//                  memory if there are long segments with just a single value
//   - OBSERVATION: our boosting algorithm is position independent, so we can sort the data by the target feature, which
//   -              helps us because we can move the class number into a loop counter and not fetch the memory, and it allows
//                  us to elimiante a branch when calculating statistics since all samples will have the same target within a loop
//   - OBSERVATION: we'll be sorting on the target, so we can't sort primarily on intput features (secondary sort ok)
//                  So, sparse input features are not typically expected to clump into ranges of non - default parameters
//                  So, we won't use ranges in our representation, so our sparse feature representation will be
//                  class Sparse { size_t index; size_t val; }
//                  This representation is invariant to position, so we'll be able to pre-compute the size before sorting
//   - OBSERVATION: having a secondary sort would allow a few features to have better memory locality since they'll
//                  cluster their updates into a single memory location.  We should pick the features that have the
//                  highest numbers of identical values so that we can get this benefit for the most number of features possible
//   - OBSERVATION: for regression, we might want to sort by increasing absolute values of the target since then we'll
//                  have more precision in the earlier numbers which can have benefits in IEEE 754 where smaller numbers
//                  have more precision in the early additions where the bin sums will be lower
//   - OBSERVATION: We will be sorting on the target values, BUT since the sort on the target will have no discontinuities
//                  We can represent it purely as class Target { size_t count; } and each item in the array is an increment
//                  of the class value(for classification).
//                  Since we know how many classes there are, we will be able to know the size of the array AFTER sorting
//   - OBSERVATION: For interaction detection, we can be asked to check for interactions with up to 64 features together, and if we're compressing
//                  feature data and /or using sparse representations, then any of those features can have any number of compressions.
//                  One example bad situation is having 3 features: one of which is sparse, one of which has 3 items per 64 - bit number, and the
//                  last has 7 items per number.You can't really template this many options.  Even if you had special pair
//                  interaction detection code, which would have 16 * 16 = 256 possible combinations(15 different packs per 64 bit number PLUS sparse)
//                  You wouldn't be able to match up the loops since the first feature would require 3 iterations, and the second 7, so you don't
//                  really get any relief. The only way to partly handle this is to make all features use the same number of bits
//                  (choose the worst case packing)
//                  and then template the combination <number_of_dimensions, number_of_bits> which has 16 * 64 possible combinations, most of which are not 
//                  used. You can get this down to maybe 16 * 4 combinations templated with loops on the others, but then you still can't easily do
//                  sparse features, so you're stuck with dense features if you go this route.
//   - OBSERVATION: For interaction detection, we'll want our template to be: <cCompilerClasses, cDimensions, cDataItemsPerPack>
//                  The main reason is that we want to load data via SIMD, and we can't have branches in order to do that, so we can't bitpack each feature
//                  differently, so they all need to use the same number of bits per pack.
//   - OBSERVATION: For histogram creation and updating, we'll want our template to be: <cCompilerClasses, cDataItemsPerPack>
//   - OBSERVATION: For partitioning, we'll want our template to be: <cCompilerClasses, cDimensions>
//   - OBSERVATION: THIS SECTION IS WRONG -> Branch misprediction is on the order of 12-20 cycles.  When doing interactions, we can template JUST the # of features
//                  since if we didn't then the # of features loop would branch mis-predict per loop, and that's bad
//                  BUT we can keep the compressed 64 bit number for each feature(which can now be in a regsiter since the # of features is templated)
//                  and then we shift them down until we're done, and then relaod the next 64-bit number.  This causes a branch mispredict each time
//                  we need to load from memory, but that's probably less than 1/8 fetches if we have 256 bins on a continuous variable, or maybe less
//                  for things like binary features.This 12 - 20 cycles will be a minor component of the loop cost in that context
//                  A bonus of this method is that we only have one template parameter(and we can limit it to maybe 5 interaction features
//                  with a loop fallback for anything up to 64 features).
//                  A second bonus of this method is that all features can be bit packed for their natural size, which means they stay as compressed
//                  As the mains.
//                  Lastly, if we want to allow sparse features we can do this. If we're templating the number of features and the # of features loop
//                  is unwound by the compiler, then each feature will have it's own code section and the if statement selecting whether a feature is
//                  sparse or not will be predicatble.If we really really wanted to, we could conceivably 
//                  template <count_dense_features, count_sparse_features>, which for low numbers of features is tractable
//   - OBSERVATION: we'll be sorting our target, then secondarily features by some packability metric, 
//   - OBSERVATION: when we make train/validation sets, the size of the sets will be indeterminate until we know the exact indexes for each set since the 
//                  number of sparse features will determine it, BUT we can have python give us the complete memory representation and then we can calcualte 
//                  the size, then return that to pyhton, have python allocate it, then pass us in the memory for a second pass at filling it
//   - OBSERVATION: since sorting this data by target is so expensive, we'll create a special "all feature" data 
//                  represenation that is just features without feature groups.  This representation will be compressed per feature.
//                  and will include a reverse index to work back to the original unsorted indexes
//                  We'll generate the main/interaction training dataset from that directly when python passes us the train/validation set indexes and 
//                  the terms.  We'll also generate train/validation duplicates of this dataset for interaction detection 
//                  (but for interactions we don't need the reverse index lookup)
//   - OBSERVATION: We should be able to completely preserve sparse data representations without expanding them, although we can also detect when dense 
//                  features should be sparsified in our own dataset
// 
// STEPS :
//   - C will fill a temporary index array in the RawArray, sort the data by target with the indexes, and secondarily by input features.  The index array 
//     will remain for reconstructing the original order
//   - Now the memory is read only from now on, and shareable, and the original order can be re-constructed


// header ids
constexpr static SharedStorageDataType k_sharedDataSetWorkingId = 0x46DB; // random 15 bit number
constexpr static SharedStorageDataType k_sharedDataSetErrorId = 0x103; // anything other than our normal id will work
constexpr static SharedStorageDataType k_sharedDataSetDoneId = 0x61E3; // random 15 bit number

// feature ids
constexpr static SharedStorageDataType k_missingFeatureBit = 0x1;
constexpr static SharedStorageDataType k_unknownFeatureBit = 0x2;
constexpr static SharedStorageDataType k_nominalFeatureBit = 0x4;
constexpr static SharedStorageDataType k_sparseFeatureBit = 0x8;
constexpr static SharedStorageDataType k_featureId = 0x2B40; // random 15 bit number with lower 4 bits set to zero

// weight ids
constexpr static SharedStorageDataType k_weightId = 0x61FB; // random 15 bit number

// target ids
constexpr static SharedStorageDataType k_classificationBit = 0x1;
constexpr static SharedStorageDataType k_targetId = 0x5A92; // random 15 bit number with lowest bit set to zero

INLINE_ALWAYS static bool IsFeature(const SharedStorageDataType id) noexcept {
   return (k_missingFeatureBit | k_unknownFeatureBit | k_nominalFeatureBit | k_sparseFeatureBit | k_featureId) ==
      ((k_missingFeatureBit | k_unknownFeatureBit | k_nominalFeatureBit | k_sparseFeatureBit) | id);
}
INLINE_ALWAYS static bool IsMissingFeature(const SharedStorageDataType id) noexcept {
   static_assert(0 == (k_missingFeatureBit & k_featureId), "k_featureId should not be missing");
   EBM_ASSERT(IsFeature(id));
   return 0 != (k_missingFeatureBit & id);
}
INLINE_ALWAYS static bool IsUnknownFeature(const SharedStorageDataType id) noexcept {
   static_assert(0 == (k_unknownFeatureBit & k_featureId), "k_featureId should not be unknown");
   EBM_ASSERT(IsFeature(id));
   return 0 != (k_unknownFeatureBit & id);
}
INLINE_ALWAYS static bool IsNominalFeature(const SharedStorageDataType id) noexcept {
   static_assert(0 == (k_nominalFeatureBit & k_featureId), "k_featureId should not be nominal");
   EBM_ASSERT(IsFeature(id));
   return 0 != (k_nominalFeatureBit & id);
}
INLINE_ALWAYS static bool IsSparseFeature(const SharedStorageDataType id) noexcept {
   static_assert(0 == (k_sparseFeatureBit & k_featureId), "k_featureId should not be sparse");
   EBM_ASSERT(IsFeature(id));
   return 0 != (k_sparseFeatureBit & id);
}
INLINE_ALWAYS static SharedStorageDataType GetFeatureId(
   const bool bMissing,
   const bool bUnknown,
   const bool bNominal,
   const bool bSparse
) noexcept {
   return k_featureId | 
      (bMissing ? k_missingFeatureBit : SharedStorageDataType { 0 }) |
      (bUnknown ? k_unknownFeatureBit : SharedStorageDataType { 0 }) |
      (bNominal ? k_nominalFeatureBit : SharedStorageDataType { 0 }) |
      (bSparse ? k_sparseFeatureBit : SharedStorageDataType { 0 });
}

INLINE_ALWAYS static bool IsTarget(const SharedStorageDataType id) noexcept {
   return (k_classificationBit | k_targetId) == (k_classificationBit | id);
}
INLINE_ALWAYS static bool IsClassificationTarget(const SharedStorageDataType id) noexcept {
   static_assert(0 == (k_classificationBit & k_targetId), "k_targetId should not be classification");
   EBM_ASSERT(IsTarget(id));
   return 0 != (k_classificationBit & id);
}
INLINE_ALWAYS static SharedStorageDataType GetTargetId(const bool bClassification) noexcept {
   return k_targetId |
      (bClassification ? k_classificationBit : SharedStorageDataType { 0 });
}


struct HeaderDataSetShared {
   SharedStorageDataType m_id;
   SharedStorageDataType m_cSamples;
   SharedStorageDataType m_cFeatures;
   SharedStorageDataType m_cWeights;
   SharedStorageDataType m_cTargets;

   // m_offsets needs to be at the bottom of this struct.  We use the struct hack to size this array
   SharedStorageDataType m_offsets[1];
};
static_assert(std::is_standard_layout<HeaderDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<HeaderDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
constexpr static size_t k_cBytesHeaderNoOffset = offsetof(HeaderDataSetShared, m_offsets);
constexpr static SharedStorageDataType k_unfilledOffset = k_cBytesHeaderNoOffset - 1;

struct FeatureDataSetShared {
   SharedStorageDataType m_id; // dense or sparse?  nominal, missing, unknown or not?
   SharedStorageDataType m_cBins;
};
static_assert(std::is_standard_layout<FeatureDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<FeatureDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");

struct SparseFeatureDataSetShared {
   // TODO: implement sparse features
   SharedStorageDataType m_defaultVal;
   SharedStorageDataType m_cNonDefaults;

   // m_nonDefaults needs to be at the bottom of this struct.  We use the struct hack to size this array
   SparseFeatureDataSetSharedEntry m_nonDefaults[1];
};
static_assert(std::is_standard_layout<SparseFeatureDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<SparseFeatureDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");

struct WeightDataSetShared {
   SharedStorageDataType m_id;
};
static_assert(std::is_standard_layout<WeightDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<WeightDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");

struct TargetDataSetShared {
   SharedStorageDataType m_id; // classification or regression
};
static_assert(std::is_standard_layout<TargetDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<TargetDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");

struct ClassificationTargetDataSetShared {
   SharedStorageDataType m_cClasses;
};
static_assert(std::is_standard_layout<ClassificationTargetDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<ClassificationTargetDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");

// No RegressionTargetDataSetShared required

static bool IsHeaderError(
   const size_t cSamples,
   const size_t cBytesAllocated,
   const unsigned char * const pFillMem
) {
   EBM_ASSERT(k_cBytesHeaderNoOffset + sizeof(HeaderDataSetShared::m_offsets[0]) + sizeof(SharedStorageDataType) <= cBytesAllocated); // checked by our caller
   EBM_ASSERT(nullptr != pFillMem);

   const HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<const HeaderDataSetShared *>(pFillMem);
   EBM_ASSERT(k_sharedDataSetWorkingId == pHeaderDataSetShared->m_id); // checked by our caller

   const SharedStorageDataType countFeatures = pHeaderDataSetShared->m_cFeatures;
   if(IsConvertError<size_t>(countFeatures)) {
      // we're being untrusting of the caller manipulating the memory improperly here
      LOG_0(Trace_Error, "ERROR IsHeaderError countFeatures is outside the range of a valid index");
      return true;
   }
   const size_t cFeatures = static_cast<size_t>(countFeatures);

   const SharedStorageDataType countWeights = pHeaderDataSetShared->m_cWeights;
   if(IsConvertError<size_t>(countWeights)) {
      // we're being untrusting of the caller manipulating the memory improperly here
      LOG_0(Trace_Error, "ERROR IsHeaderError countWeights is outside the range of a valid index");
      return true;
   }
   const size_t cWeights = static_cast<size_t>(countWeights);

   const SharedStorageDataType countTargets = pHeaderDataSetShared->m_cTargets;
   if(IsConvertError<size_t>(countTargets)) {
      // we're being untrusting of the caller manipulating the memory improperly here
      LOG_0(Trace_Error, "ERROR IsHeaderError countTargets is outside the range of a valid index");
      return true;
   }
   const size_t cTargets = static_cast<size_t>(countTargets);

   if(IsAddError(cFeatures, cWeights, cTargets)) {
      LOG_0(Trace_Error, "ERROR IsHeaderError IsAddError(cFeatures, cWeights, cTargets)");
      return true;
   }
   const size_t cOffsets = cFeatures + cWeights + cTargets;

   if(IsMultiplyError(sizeof(HeaderDataSetShared::m_offsets[0]), cOffsets)) {
      LOG_0(Trace_Error, "ERROR IsHeaderError IsMultiplyError(sizeof(HeaderDataSetShared::m_offsets[0]), cOffsets)");
      return true;
   }
   const size_t cBytesOffsets = sizeof(HeaderDataSetShared::m_offsets[0]) * cOffsets;

   if(IsAddError(k_cBytesHeaderNoOffset, cBytesOffsets)) {
      LOG_0(Trace_Error, "ERROR IsHeaderError IsAddError(k_cBytesHeaderNoOffset, cBytesOffsets)");
      return true;
   }
   const size_t cBytesHeader = k_cBytesHeaderNoOffset + cBytesOffsets;

   if(cBytesAllocated - sizeof(SharedStorageDataType) < cBytesHeader) {
      LOG_0(Trace_Error, "ERROR IsHeaderError cBytesAllocated - sizeof(SharedStorageDataType) < cBytesHeader");
      return true;
   }

   const SharedStorageDataType indexByte0 = pHeaderDataSetShared->m_offsets[0];
   if(IsConvertError<size_t>(indexByte0)) {
      // we're being untrusting of the caller manipulating the memory improperly here
      LOG_0(Trace_Error, "ERROR IsHeaderError indexByte0 is outside the range of a valid index");
      return true;
   }
   const size_t iByte0 = static_cast<size_t>(indexByte0);

   if(iByte0 != cBytesHeader) {
      // we're being untrusting of the caller manipulating the memory improperly here
      LOG_0(Trace_Error, "ERROR IsHeaderError iByte0 != cBytesHeader");
      return true;
   }

   const SharedStorageDataType * const pInternalState =
      reinterpret_cast<const SharedStorageDataType *>(pFillMem + cBytesAllocated - sizeof(SharedStorageDataType));

   const SharedStorageDataType internalState = *pInternalState;
   if(IsConvertError<size_t>(internalState)) {
      LOG_0(Trace_Error, "ERROR IsHeaderError opaqueState invalid");
      return true;
   }
   const size_t iOffset = static_cast<size_t>(internalState);

   if(cOffsets <= iOffset) {
      LOG_0(Trace_Error, "ERROR IsHeaderError cOffsets <= iOffset");
      return true;
   }

   if(size_t { 0 } == iOffset) {
      if(SharedStorageDataType { 0 } != pHeaderDataSetShared->m_cSamples) {
         LOG_0(Trace_Error, "ERROR IsHeaderError SharedStorageDataType { 0 } != pHeaderDataSetShared->m_cSamples");
         return true;
      }
   } else {
      if(pHeaderDataSetShared->m_cSamples != static_cast<SharedStorageDataType>(cSamples)) {
         LOG_0(Trace_Error, "ERROR IsHeaderError pHeaderDataSetShared->m_cSamples != cSamples");
         return true;
      }

      // if iOffset is 1, we'll just check this once again without any issues
      const SharedStorageDataType indexHighestOffsetPrev = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset - 1];
      if(IsConvertError<size_t>(indexHighestOffsetPrev)) {
         // we're being untrusting of the caller manipulating the memory improperly here
         LOG_0(Trace_Error, "ERROR IsHeaderError indexHighestOffsetPrev is outside the range of a valid index");
         return true;
      }
      const size_t iHighestOffsetPrev = static_cast<size_t>(indexHighestOffsetPrev);

      if(iHighestOffsetPrev < iByte0) {
         // we're being untrusting of the caller manipulating the memory improperly here
         LOG_0(Trace_Error, "ERROR IsHeaderError iHighestOffsetPrev < iByte0");
         return true;
      }

      const SharedStorageDataType indexHighestOffset = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset];
      if(IsConvertError<size_t>(indexHighestOffset)) {
         // we're being untrusting of the caller manipulating the memory improperly here
         LOG_0(Trace_Error, "ERROR IsHeaderError indexHighestOffset is outside the range of a valid index");
         return true;
      }
      const size_t iHighestOffset = static_cast<size_t>(indexHighestOffset);

      if(iHighestOffset < iHighestOffsetPrev) {
         // we're being untrusting of the caller manipulating the memory improperly here
         LOG_0(Trace_Error, "ERROR IsHeaderError iHighestOffset < iHighestOffsetPrev");
         return true;
      }

      // through associativity since iByte0 <= iHighestOffsetPrev && iHighestOffsetPrev <= iHighestOffset
      EBM_ASSERT(iByte0 <= iHighestOffset);
   }

   const size_t iOffsetNext = iOffset + 1; // we verified iOffset < cOffsets above, so this is guaranteed to work
   if(iOffsetNext != cOffsets) {
      const SharedStorageDataType indexHighestOffsetNext = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffsetNext];
      if(k_unfilledOffset != indexHighestOffsetNext) {
         LOG_0(Trace_Error, "ERROR IsHeaderError k_unfilledOffset != indexHighestOffsetNext");
         return true;
      }
   }

   return false;
}

static void LockDataSetShared(unsigned char * const pFillMem) {
   HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
   EBM_ASSERT(k_sharedDataSetWorkingId == pHeaderDataSetShared->m_id);

   // TODO: sort the data by the target (if there is only one target)

   pHeaderDataSetShared->m_id = k_sharedDataSetDoneId; // signal that we finished construction of the data set
}

static IntEbmType AppendHeader(
   const IntEbmType countFeatures,
   const IntEbmType countWeights,
   const IntEbmType countTargets,
   const size_t cBytesAllocated,
   unsigned char * const pFillMem
) {
   EBM_ASSERT(size_t { 0 } == cBytesAllocated && nullptr == pFillMem || nullptr != pFillMem);

   LOG_N(
      Trace_Info,
      "Entered AppendHeader: "
      "countFeatures=%" IntEbmTypePrintf ", "
      "countWeights=%" IntEbmTypePrintf ", "
      "countTargets=%" IntEbmTypePrintf ", "
      "cBytesAllocated=%zu, "
      "pFillMem=%p"
      ,
      countFeatures,
      countWeights,
      countTargets,
      cBytesAllocated,
      static_cast<void *>(pFillMem)
   );

   if(IsConvertErrorDual<size_t, SharedStorageDataType>(countFeatures)) {
      LOG_0(Trace_Error, "ERROR AppendHeader countFeatures is outside the range of a valid index");
      return Error_IllegalParamValue;
   }
   const size_t cFeatures = static_cast<size_t>(countFeatures);

   if(IsConvertErrorDual<size_t, SharedStorageDataType>(countWeights)) {
      LOG_0(Trace_Error, "ERROR AppendHeader countWeights is outside the range of a valid index");
      return Error_IllegalParamValue;
   }
   const size_t cWeights = static_cast<size_t>(countWeights);

   if(IsConvertErrorDual<size_t, SharedStorageDataType>(countTargets)) {
      LOG_0(Trace_Error, "ERROR AppendHeader countTargets is outside the range of a valid index");
      return Error_IllegalParamValue;
   }
   const size_t cTargets = static_cast<size_t>(countTargets);

   if(IsAddError(cFeatures, cWeights, cTargets)) {
      LOG_0(Trace_Error, "ERROR AppendHeader IsAddError(cFeatures, cWeights, cTargets)");
      return Error_IllegalParamValue;
   }
   const size_t cOffsets = cFeatures + cWeights + cTargets;

   if(IsMultiplyError(sizeof(HeaderDataSetShared::m_offsets[0]), cOffsets)) {
      LOG_0(Trace_Error, "ERROR AppendHeader IsMultiplyError(sizeof(HeaderDataSetShared::m_offsets[0]), cOffsets)");
      return Error_IllegalParamValue;
   }
   const size_t cBytesOffsets = sizeof(HeaderDataSetShared::m_offsets[0]) * cOffsets;

   if(IsAddError(k_cBytesHeaderNoOffset, cBytesOffsets)) {
      LOG_0(Trace_Error, "ERROR AppendHeader IsAddError(k_cBytesHeaderNoOffset, cBytesOffsets)");
      return Error_IllegalParamValue;
   }
   const size_t cBytesHeader = k_cBytesHeaderNoOffset + cBytesOffsets;

   if(IsConvertError<SharedStorageDataType>(cBytesHeader)) {
      LOG_0(Trace_Error, "ERROR AppendHeader cBytesHeader is outside the range of a valid size");
      return Error_IllegalParamValue;
   }

   if(nullptr != pFillMem) {
      if(size_t { 0 } == cOffsets) {
         if(cBytesAllocated != cBytesHeader) {
            LOG_0(Trace_Error, "ERROR AppendHeader buffer size and fill size do not agree");
            return Error_IllegalParamValue;
         }
      } else {
         if(cBytesAllocated - sizeof(SharedStorageDataType) < cBytesHeader) {
            LOG_0(Trace_Error, "ERROR AppendHeader cBytesAllocated - sizeof(SharedStorageDataType) < cBytesHeader");
            // don't set the header to bad if we don't have enough memory for the header itself
            return Error_IllegalParamValue;
         }
      }

      HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);

      pHeaderDataSetShared->m_id = k_sharedDataSetWorkingId;
      pHeaderDataSetShared->m_cSamples = 0;
      pHeaderDataSetShared->m_cFeatures = static_cast<SharedStorageDataType>(cFeatures);
      pHeaderDataSetShared->m_cWeights = static_cast<SharedStorageDataType>(cWeights);
      pHeaderDataSetShared->m_cTargets = static_cast<SharedStorageDataType>(cTargets);

      if(size_t { 0 } == cOffsets) {
         // we allow this shared data set to be permissive in it's construction but if there are things like
         // zero targets we expect the booster or interaction detector constructors to give errors
         LockDataSetShared(pFillMem);
      } else {
         SharedStorageDataType * pCur = pHeaderDataSetShared->m_offsets;
         const SharedStorageDataType * const pEnd = pCur + cOffsets;
         do {
            *pCur = k_unfilledOffset;
            ++pCur;
         } while(pEnd != pCur);

         // position our first feature right after the header, or at the target if there are no features
         pHeaderDataSetShared->m_offsets[0] = static_cast<SharedStorageDataType>(cBytesHeader);
         SharedStorageDataType * const pInternalState =
            reinterpret_cast<SharedStorageDataType *>(pFillMem + cBytesAllocated - sizeof(SharedStorageDataType));
         *pInternalState = 0;
      }
      return Error_None;
   }
   if(IsConvertError<IntEbmType>(cBytesHeader)) {
      LOG_0(Trace_Error, "ERROR AppendHeader IsConvertError<IntEbmType>(cBytesHeader)");
      return Error_OutOfMemory;
   }
   return cBytesHeader;
}

static bool DecideIfSparse(const size_t cSamples, const IntEbmType * binIndexes) {
   // For sparsity in the data set shared memory the only thing that matters is compactness since we don't use
   // this memory in any high performance loops

   UNUSED(cSamples);
   UNUSED(binIndexes);

   // TODO: evalute the data to decide if the feature should be sparse or not
   return false;
}

static IntEbmType AppendFeature(
   const IntEbmType countBins,
   const BoolEbmType isMissing,
   const BoolEbmType isUnknown,
   const BoolEbmType isNominal,
   const IntEbmType countSamples,
   const IntEbmType * binIndexes,
   const size_t cBytesAllocated,
   unsigned char * const pFillMem
) {
   EBM_ASSERT(size_t { 0 } == cBytesAllocated && nullptr == pFillMem || 
      nullptr != pFillMem && k_cBytesHeaderNoOffset + sizeof(HeaderDataSetShared::m_offsets[0]) + sizeof(SharedStorageDataType) <= cBytesAllocated);

   LOG_N(
      Trace_Info,
      "Entered AppendFeature: "
      "countBins=%" IntEbmTypePrintf ", "
      "isMissing=%s, "
      "isUnknown=%s, "
      "isNominal=%s, "
      "countSamples=%" IntEbmTypePrintf ", "
      "binIndexes=%p, "
      "cBytesAllocated=%zu, "
      "pFillMem=%p"
      ,
      countBins,
      ObtainTruth(isMissing),
      ObtainTruth(isUnknown),
      ObtainTruth(isNominal),
      countSamples,
      static_cast<const void *>(binIndexes),
      cBytesAllocated,
      static_cast<void *>(pFillMem)
   );

   {
      if(IsConvertErrorDual<size_t, SharedStorageDataType>(countBins)) {
         LOG_0(Trace_Error, "ERROR AppendFeature countBins is outside the range of a valid index");
         goto return_bad;
      }
      if(EBM_FALSE != isMissing && EBM_TRUE != isMissing) {
         LOG_0(Trace_Error, "ERROR AppendFeature isMissing is not EBM_FALSE or EBM_TRUE");
         goto return_bad;
      }

      if(EBM_FALSE != isUnknown && EBM_TRUE != isUnknown) {
         LOG_0(Trace_Error, "ERROR AppendFeature isUnknown is not EBM_FALSE or EBM_TRUE");
         goto return_bad;
      }
      if(EBM_FALSE != isNominal && EBM_TRUE != isNominal) {
         LOG_0(Trace_Error, "ERROR AppendFeature isNominal is not EBM_FALSE or EBM_TRUE");
         goto return_bad;
      }
      if(IsConvertErrorDual<size_t, SharedStorageDataType>(countSamples)) {
         LOG_0(Trace_Error, "ERROR AppendFeature countSamples is outside the range of a valid index");
         goto return_bad;
      }
      const size_t cSamples = static_cast<size_t>(countSamples);

      bool bSparse = false;
      if(size_t { 0 } != cSamples) {
         if(nullptr == binIndexes) {
            LOG_0(Trace_Error, "ERROR AppendFeature nullptr == binIndexes");
            goto return_bad;
         }

         // TODO: handle sparse data someday
         bSparse = DecideIfSparse(cSamples, binIndexes);
      }

      size_t iOffset = 0;
      size_t iByteCur = sizeof(FeatureDataSetShared);
      if(nullptr != pFillMem) {
         if(IsHeaderError(cSamples, cBytesAllocated, pFillMem)) {
            goto return_bad;
         }

         SharedStorageDataType * const pInternalState =
            reinterpret_cast<SharedStorageDataType *>(pFillMem + cBytesAllocated - sizeof(SharedStorageDataType));
         iOffset = static_cast<size_t>(*pInternalState);

         HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);

         const size_t cFeatures = static_cast<size_t>(pHeaderDataSetShared->m_cFeatures);

         // check that we haven't exceeded the number of features
         if(cFeatures <= iOffset) {
            LOG_0(Trace_Error, "ERROR AppendFeature cFeatures <= iOffset");
            goto return_bad;
         }

         const size_t iHighestOffset = static_cast<size_t>(ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset]);

         if(IsAddError(iByteCur, iHighestOffset)) {
            LOG_0(Trace_Error, "ERROR AppendFeature IsAddError(iByteCur, iHighestOffset)");
            goto return_bad;
         }
         iByteCur += iHighestOffset; // if we're going to access FeatureDataSetShared, then check if we have the space
         if(cBytesAllocated < iByteCur) {
            LOG_0(Trace_Error, "ERROR AppendFeature cBytesAllocated < iByteCur");
            goto return_bad;
         }

         EBM_ASSERT(size_t { 0 } == iOffset && SharedStorageDataType { 0 } == pHeaderDataSetShared->m_cSamples ||
            static_cast<SharedStorageDataType>(cSamples) == pHeaderDataSetShared->m_cSamples);
         pHeaderDataSetShared->m_cSamples = static_cast<SharedStorageDataType>(cSamples);

         FeatureDataSetShared * pFeatureDataSetShared = reinterpret_cast<FeatureDataSetShared *>(pFillMem + iHighestOffset);
         pFeatureDataSetShared->m_id = GetFeatureId(
            EBM_FALSE != isMissing,
            EBM_FALSE != isUnknown,
            EBM_FALSE != isNominal,
            bSparse
         );
         pFeatureDataSetShared->m_cBins = static_cast<SharedStorageDataType>(countBins);
      }

      if(size_t { 0 } != cSamples) {
         if(IsMultiplyError(sizeof(SharedStorageDataType), cSamples)) {
            LOG_0(Trace_Error, "ERROR AppendFeature IsMultiplyError(sizeof(SharedStorageDataType), cSamples)");
            goto return_bad;
         }
         const size_t cBytesAllSamples = sizeof(SharedStorageDataType) * cSamples;

         if(IsAddError(iByteCur, cBytesAllSamples)) {
            LOG_0(Trace_Error, "ERROR AppendFeature IsAddError(iByteCur, cBytesAllSamples)");
            goto return_bad;
         }
         const size_t iByteNext = iByteCur + cBytesAllSamples;

         if(nullptr != pFillMem) {
            if(cBytesAllocated < iByteNext) {
               LOG_0(Trace_Error, "ERROR AppendFeature cBytesAllocated < iByteNext");
               goto return_bad;
            }

            if(IsMultiplyError(sizeof(binIndexes[0]), cSamples)) {
               LOG_0(Trace_Error, "ERROR AppendFeature IsMultiplyError(sizeof(binIndexes[0]), cSamples)");
               goto return_bad;
            }
            const IntEbmType * pBinIndex = binIndexes;
            const IntEbmType * const pBinIndexsEnd = binIndexes + cSamples;
            SharedStorageDataType * pFillData = reinterpret_cast<SharedStorageDataType *>(pFillMem + iByteCur);
            do {
               const IntEbmType indexBin = *pBinIndex;
               if(indexBin < IntEbmType { 0 }) {
                  LOG_0(Trace_Error, "ERROR AppendFeature indexBin can't be negative");
                  goto return_bad;
               }
               if(countBins <= indexBin) {
                  LOG_0(Trace_Error, "ERROR AppendFeature countBins <= indexBin");
                  goto return_bad;
               }
               // since countBins can be converted to these, so now can indexBin
               EBM_ASSERT(!IsConvertError<size_t>(indexBin));
               EBM_ASSERT(!IsConvertError<SharedStorageDataType>(indexBin));

               // TODO: bit compact this
               *pFillData = static_cast<SharedStorageDataType>(indexBin);

               ++pFillData;
               ++pBinIndex;
            } while(pBinIndexsEnd != pBinIndex);
            EBM_ASSERT(reinterpret_cast<unsigned char *>(pFillData) == pFillMem + iByteNext);
         }
         iByteCur = iByteNext;
      }

      if(nullptr != pFillMem) {
         HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
         EBM_ASSERT(k_sharedDataSetWorkingId == pHeaderDataSetShared->m_id);

         // In IsHeaderError above we checked that iOffset < cOffsets, and cOffsets was a size_t so this
         // addition should work and all counts should be convertible to size_t 
         EBM_ASSERT(iOffset < std::numeric_limits<size_t>::max());
         ++iOffset;
         const size_t cOffsets = static_cast<size_t>(pHeaderDataSetShared->m_cFeatures) + 
            static_cast<size_t>(pHeaderDataSetShared->m_cWeights) + 
            static_cast<size_t>(pHeaderDataSetShared->m_cTargets);
         
         if(iOffset == cOffsets) {
            if(cBytesAllocated != iByteCur) {
               LOG_0(Trace_Error, "ERROR AppendFeature buffer size and fill size do not agree");
               goto return_bad;
            }

            LockDataSetShared(pFillMem);
         } else {
            if(cBytesAllocated - sizeof(SharedStorageDataType) < iByteCur) {
               LOG_0(Trace_Error, "ERROR AppendFeature cBytesAllocated - sizeof(SharedStorageDataType) < iByteNext");
               goto return_bad;
            }

            if(IsConvertError<SharedStorageDataType>(iOffset)) {
               LOG_0(Trace_Error, "ERROR AppendFeature IsConvertError<IntEbmType>(iOffset)");
               goto return_bad;
            }
            if(IsConvertError<SharedStorageDataType>(iByteCur)) {
               LOG_0(Trace_Error, "ERROR AppendFeature IsConvertError<SharedStorageDataType>(iByteCur)");
               goto return_bad;
            }
            ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset] = static_cast<SharedStorageDataType>(iByteCur);
            SharedStorageDataType * const pInternalState =
               reinterpret_cast<SharedStorageDataType *>(pFillMem + cBytesAllocated - sizeof(SharedStorageDataType));
            *pInternalState = static_cast<SharedStorageDataType>(iOffset); // the offset index is our state
         }
         return Error_None;
      }
      if(IsConvertError<IntEbmType>(iByteCur)) {
         LOG_0(Trace_Error, "ERROR AppendFeature IsConvertError<IntEbmType>(iByteCur)");
         goto return_bad;
      }
      return static_cast<IntEbmType>(iByteCur);
   }

return_bad:;

   if(nullptr != pFillMem) {
      HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
      pHeaderDataSetShared->m_id = k_sharedDataSetErrorId;
   }
   return Error_IllegalParamValue;
}

static IntEbmType AppendWeight(
   const IntEbmType countSamples,
   const double * aWeights,
   const size_t cBytesAllocated,
   unsigned char * const pFillMem
) {
   EBM_ASSERT(size_t { 0 } == cBytesAllocated && nullptr == pFillMem ||
      nullptr != pFillMem && k_cBytesHeaderNoOffset + sizeof(HeaderDataSetShared::m_offsets[0]) + sizeof(SharedStorageDataType) <= cBytesAllocated);

   LOG_N(
      Trace_Info,
      "Entered AppendWeight: "
      "countSamples=%" IntEbmTypePrintf ", "
      "aWeights=%p, "
      "cBytesAllocated=%zu, "
      "pFillMem=%p"
      ,
      countSamples,
      static_cast<const void *>(aWeights),
      cBytesAllocated,
      static_cast<void *>(pFillMem)
   );

   {
      if(IsConvertErrorDual<size_t, SharedStorageDataType>(countSamples)) {
         LOG_0(Trace_Error, "ERROR AppendWeight countSamples is outside the range of a valid index");
         goto return_bad;
      }
      const size_t cSamples = static_cast<size_t>(countSamples);

      size_t iOffset = 0;
      size_t iByteCur = sizeof(WeightDataSetShared);
      if(nullptr != pFillMem) {
         if(IsHeaderError(cSamples, cBytesAllocated, pFillMem)) {
            goto return_bad;
         }

         SharedStorageDataType * const pInternalState =
            reinterpret_cast<SharedStorageDataType *>(pFillMem + cBytesAllocated - sizeof(SharedStorageDataType));

         iOffset = static_cast<size_t>(*pInternalState);

         HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);

         const size_t cFeatures = static_cast<size_t>(pHeaderDataSetShared->m_cFeatures);
         const size_t cWeights = static_cast<size_t>(pHeaderDataSetShared->m_cWeights);

         // check that we're in the weight setting range
         if(iOffset < cFeatures) {
            LOG_0(Trace_Error, "ERROR AppendWeight iOffset < cFeatures");
            goto return_bad;
         }
         if(cFeatures + cWeights <= iOffset) {
            LOG_0(Trace_Error, "ERROR AppendWeight cFeatures + cWeights <= iOffset");
            goto return_bad;
         }

         const size_t iHighestOffset = static_cast<size_t>(ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset]);

         if(IsAddError(iByteCur, iHighestOffset)) {
            LOG_0(Trace_Error, "ERROR AppendWeight IsAddError(iByteCur, iHighestOffset)");
            goto return_bad;
         }
         iByteCur += iHighestOffset; // if we're going to access FeatureDataSetShared, then check if we have the space
         if(cBytesAllocated < iByteCur) {
            LOG_0(Trace_Error, "ERROR AppendWeight cBytesAllocated < iByteCur");
            goto return_bad;
         }

         EBM_ASSERT(size_t { 0 } == iOffset && SharedStorageDataType { 0 } == pHeaderDataSetShared->m_cSamples ||
            static_cast<SharedStorageDataType>(cSamples) == pHeaderDataSetShared->m_cSamples);
         pHeaderDataSetShared->m_cSamples = static_cast<SharedStorageDataType>(cSamples);

         WeightDataSetShared * const pWeightDataSetShared = reinterpret_cast<WeightDataSetShared *>(pFillMem + iHighestOffset);
         pWeightDataSetShared->m_id = k_weightId;
      }

      if(size_t { 0 } != cSamples) {
         if(nullptr == aWeights) {
            LOG_0(Trace_Error, "ERROR AppendWeight nullptr == aWeights");
            goto return_bad;
         }

         if(IsMultiplyError(EbmMax(sizeof(*aWeights), sizeof(FloatFast)), cSamples)) {
            LOG_0(Trace_Error, "ERROR AppendWeight IsMultiplyError(EbmMax(sizeof(*aWeights), sizeof(FloatFast)), cSamples)");
            goto return_bad;
         }
         const size_t cBytesAllSamples = sizeof(FloatFast) * cSamples;

         if(IsAddError(iByteCur, cBytesAllSamples)) {
            LOG_0(Trace_Error, "ERROR AppendWeight IsAddError(iByteCur, cBytesAllSamples)");
            goto return_bad;
         }
         const size_t iByteNext = iByteCur + cBytesAllSamples;
         if(nullptr != pFillMem) {
            if(cBytesAllocated < iByteNext) {
               LOG_0(Trace_Error, "ERROR AppendWeight cBytesAllocated < iByteNext");
               goto return_bad;
            }

            static_assert(sizeof(FloatFast) == sizeof(*aWeights), "float mismatch");
            memcpy(pFillMem + iByteCur, aWeights, cBytesAllSamples);
         }
         iByteCur = iByteNext;
      }

      if(nullptr != pFillMem) {
         HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
         EBM_ASSERT(k_sharedDataSetWorkingId == pHeaderDataSetShared->m_id);

         // In IsHeaderError above we checked that iOffset < cOffsets, and cOffsets was a size_t so this
         // addition should work and all counts should be convertible to size_t 
         EBM_ASSERT(iOffset < std::numeric_limits<size_t>::max());
         ++iOffset;
         const size_t cOffsets = static_cast<size_t>(pHeaderDataSetShared->m_cFeatures) +
            static_cast<size_t>(pHeaderDataSetShared->m_cWeights) +
            static_cast<size_t>(pHeaderDataSetShared->m_cTargets);

         if(iOffset == cOffsets) {
            if(cBytesAllocated != iByteCur) {
               LOG_0(Trace_Error, "ERROR AppendWeight buffer size and fill size do not agree");
               goto return_bad;
            }

            LockDataSetShared(pFillMem);
         } else {
            if(cBytesAllocated - sizeof(SharedStorageDataType) < iByteCur) {
               LOG_0(Trace_Error, "ERROR AppendWeight cBytesAllocated - sizeof(SharedStorageDataType) < iByteCur");
               goto return_bad;
            }

            if(IsConvertError<SharedStorageDataType>(iOffset)) {
               LOG_0(Trace_Error, "ERROR AppendWeight IsConvertError<IntEbmType>(iOffset)");
               goto return_bad;
            }
            if(IsConvertError<SharedStorageDataType>(iByteCur)) {
               LOG_0(Trace_Error, "ERROR AppendWeight IsConvertError<SharedStorageDataType>(iByteCur)");
               goto return_bad;
            }
            ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset] = static_cast<SharedStorageDataType>(iByteCur);
            SharedStorageDataType * const pInternalState =
               reinterpret_cast<SharedStorageDataType *>(pFillMem + cBytesAllocated - sizeof(SharedStorageDataType));
            *pInternalState = static_cast<SharedStorageDataType>(iOffset); // the offset index is our state
         }
         return Error_None;
      }
      if(IsConvertError<IntEbmType>(iByteCur)) {
         LOG_0(Trace_Error, "ERROR AppendWeight IsConvertError<IntEbmType>(iByteCur)");
         goto return_bad;
      }
      return static_cast<IntEbmType>(iByteCur);
   }

return_bad:;

   if(nullptr != pFillMem) {
      HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
      pHeaderDataSetShared->m_id = k_sharedDataSetErrorId;
   }
   return Error_IllegalParamValue;
}

static IntEbmType AppendTarget(
   const bool bClassification,
   const IntEbmType countClasses,
   const IntEbmType countSamples,
   const void * aTargets,
   const size_t cBytesAllocated,
   unsigned char * const pFillMem
) {
   EBM_ASSERT(size_t { 0 } == cBytesAllocated && nullptr == pFillMem ||
      nullptr != pFillMem && k_cBytesHeaderNoOffset + sizeof(HeaderDataSetShared::m_offsets[0]) + sizeof(SharedStorageDataType) <= cBytesAllocated);

   LOG_N(
      Trace_Info,
      "Entered AppendTarget: "
      "bClassification=%s, "
      "countClasses=%" IntEbmTypePrintf ", "
      "countSamples=%" IntEbmTypePrintf ", "
      "aTargets=%p, "
      "cBytesAllocated=%zu, "
      "pFillMem=%p"
      ,
      ObtainTruth(bClassification ? EBM_TRUE : EBM_FALSE),
      countClasses,
      countSamples,
      static_cast<const void *>(aTargets),
      cBytesAllocated,
      static_cast<void *>(pFillMem)
   );

   {
      if(IsConvertErrorDual<size_t, SharedStorageDataType>(countClasses)) {
         LOG_0(Trace_Error, "ERROR AppendTarget countClasses is outside the range of a valid index");
         goto return_bad;
      }
      if(IsConvertErrorDual<size_t, SharedStorageDataType>(countSamples)) {
         LOG_0(Trace_Error, "ERROR AppendTarget countSamples is outside the range of a valid index");
         goto return_bad;
      }
      const size_t cSamples = static_cast<size_t>(countSamples);

      size_t iOffset = 0;
      size_t iByteCur = bClassification ? sizeof(TargetDataSetShared) + sizeof(ClassificationTargetDataSetShared) :
         sizeof(TargetDataSetShared);
      if(nullptr != pFillMem) {
         if(IsHeaderError(cSamples, cBytesAllocated, pFillMem)) {
            goto return_bad;
         }

         SharedStorageDataType * const pInternalState =
            reinterpret_cast<SharedStorageDataType *>(pFillMem + cBytesAllocated - sizeof(SharedStorageDataType));

         iOffset = static_cast<size_t>(*pInternalState);

         HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);

         const size_t cFeatures = static_cast<size_t>(pHeaderDataSetShared->m_cFeatures);
         const size_t cWeights = static_cast<size_t>(pHeaderDataSetShared->m_cWeights);

         // check that we're done the features and weights
         if(iOffset < cFeatures + cWeights) {
            LOG_0(Trace_Error, "ERROR AppendTarget iOffset < cFeatures + cWeights");
            goto return_bad;
         }

         const size_t iHighestOffset = static_cast<size_t>(ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset]);

         if(IsAddError(iByteCur, iHighestOffset)) {
            LOG_0(Trace_Error, "ERROR AppendTarget IsAddError(iByteCur, iHighestOffset)");
            goto return_bad;
         }
         iByteCur += iHighestOffset; // if we're going to access FeatureDataSetShared, then check if we have the space
         if(cBytesAllocated < iByteCur) {
            LOG_0(Trace_Error, "ERROR AppendTarget cBytesAllocated < iByteCur");
            goto return_bad;
         }

         EBM_ASSERT(size_t { 0 } == iOffset && SharedStorageDataType { 0 } == pHeaderDataSetShared->m_cSamples ||
            static_cast<SharedStorageDataType>(cSamples) == pHeaderDataSetShared->m_cSamples);
         pHeaderDataSetShared->m_cSamples = static_cast<SharedStorageDataType>(cSamples);

         unsigned char * const pFillMemTemp = pFillMem + iHighestOffset;
         TargetDataSetShared * const pTargetDataSetShared = reinterpret_cast<TargetDataSetShared *>(pFillMemTemp);
         pTargetDataSetShared->m_id = GetTargetId(bClassification);

         if(bClassification) {
            ClassificationTargetDataSetShared * pClassificationTargetDataSetShared = reinterpret_cast<ClassificationTargetDataSetShared *>(pFillMemTemp + sizeof(TargetDataSetShared));
            pClassificationTargetDataSetShared->m_cClasses = static_cast<SharedStorageDataType>(countClasses);
         }
      }

      if(size_t { 0 } != cSamples) {
         if(nullptr == aTargets) {
            LOG_0(Trace_Error, "ERROR AppendTarget nullptr == aTargets");
            goto return_bad;
         }

         size_t cBytesAllSamples;
         if(bClassification) {
            if(IsMultiplyError(EbmMax(sizeof(IntEbmType), sizeof(SharedStorageDataType)), cSamples)) {
               LOG_0(Trace_Error, "ERROR AppendTarget IsMultiplyError(EbmMax(sizeof(IntEbmType), sizeof(SharedStorageDataType)), cSamples)");
               goto return_bad;
            }
            cBytesAllSamples = sizeof(SharedStorageDataType) * cSamples;
         } else {
            if(IsMultiplyError(EbmMax(sizeof(double), sizeof(FloatFast)), cSamples)) {
               LOG_0(Trace_Error, "ERROR AppendTarget IsMultiplyError(EbmMax(sizeof(double), sizeof(FloatFast)), cSamples)");
               goto return_bad;
            }
            cBytesAllSamples = sizeof(FloatFast) * cSamples;
         }
         if(IsAddError(iByteCur, cBytesAllSamples)) {
            LOG_0(Trace_Error, "ERROR AppendTarget IsAddError(iByteCur, cBytesAllSamples)");
            goto return_bad;
         }
         const size_t iByteNext = iByteCur + cBytesAllSamples;
         if(nullptr != pFillMem) {
            if(cBytesAllocated < iByteNext) {
               LOG_0(Trace_Error, "ERROR AppendTarget cBytesAllocated < iByteNext");
               goto return_bad;
            }
            if(bClassification) {
               const IntEbmType * pTarget = reinterpret_cast<const IntEbmType *>(aTargets);
               if(IsMultiplyError(sizeof(pTarget[0]), cSamples)) {
                  LOG_0(Trace_Error, "ERROR AppendTarget IsMultiplyError(sizeof(SharedStorageDataType), cSamples)");
                  goto return_bad;
               }
               const IntEbmType * const pTargetsEnd = pTarget + cSamples;
               SharedStorageDataType * pFillData = reinterpret_cast<SharedStorageDataType *>(pFillMem + iByteCur);
               do {
                  const IntEbmType target = *pTarget;
                  if(target < IntEbmType { 0 }) {
                     LOG_0(Trace_Error, "ERROR AppendTarget classification target can't be negative");
                     goto return_bad;
                  }
                  if(countClasses <= target) {
                     LOG_0(Trace_Error, "ERROR AppendTarget countClasses <= target");
                     goto return_bad;
                  }
                  // since countClasses can be converted to these, so now can target
                  EBM_ASSERT(!IsConvertError<size_t>(target));
                  EBM_ASSERT(!IsConvertError<SharedStorageDataType>(target));
               
                  // TODO: sort by the target and then convert the target to a count of each index
                  *pFillData = static_cast<SharedStorageDataType>(target);
               
                  ++pFillData;
                  ++pTarget;
               } while(pTargetsEnd != pTarget);
               EBM_ASSERT(reinterpret_cast<unsigned char *>(pFillData) == pFillMem + iByteNext);
            } else {
               static_assert(sizeof(FloatFast) == sizeof(double), "float mismatch");
               memcpy(pFillMem + iByteCur, aTargets, cBytesAllSamples);
            }
         }
         iByteCur = iByteNext;
      }

      if(nullptr != pFillMem) {
         HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
         EBM_ASSERT(k_sharedDataSetWorkingId == pHeaderDataSetShared->m_id);

         // In IsHeaderError above we checked that iOffset < cOffsets, and cOffsets was a size_t so this
         // addition should work and all counts should be convertible to size_t 
         EBM_ASSERT(iOffset < std::numeric_limits<size_t>::max());
         ++iOffset;
         const size_t cOffsets = static_cast<size_t>(pHeaderDataSetShared->m_cFeatures) +
            static_cast<size_t>(pHeaderDataSetShared->m_cWeights) +
            static_cast<size_t>(pHeaderDataSetShared->m_cTargets);

         if(iOffset == cOffsets) {
            if(cBytesAllocated != iByteCur) {
               LOG_0(Trace_Error, "ERROR AppendTarget buffer size and fill size do not agree");
               goto return_bad;
            }

            LockDataSetShared(pFillMem);
         } else {
            if(cBytesAllocated - sizeof(SharedStorageDataType) < iByteCur) {
               LOG_0(Trace_Error, "ERROR AppendTarget cBytesAllocated - sizeof(SharedStorageDataType) < iByteCur");
               goto return_bad;
            }
            if(IsConvertError<SharedStorageDataType>(iOffset)) {
               LOG_0(Trace_Error, "ERROR AppendTarget IsConvertError<IntEbmType>(iOffset)");
               goto return_bad;
            }
            if(IsConvertError<SharedStorageDataType>(iByteCur)) {
               LOG_0(Trace_Error, "ERROR AppendTarget IsConvertError<SharedStorageDataType>(iByteCur)");
               goto return_bad;
            }
            ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset] = static_cast<SharedStorageDataType>(iByteCur);
            SharedStorageDataType * const pInternalState =
               reinterpret_cast<SharedStorageDataType *>(pFillMem + cBytesAllocated - sizeof(SharedStorageDataType));
            *pInternalState = static_cast<SharedStorageDataType>(iOffset); // the offset index is our state
         }
         return Error_None;
      }
      if(IsConvertError<IntEbmType>(iByteCur)) {
         LOG_0(Trace_Error, "ERROR AppendTarget IsConvertError<IntEbmType>(iByteCur)");
         goto return_bad;
      }
      return static_cast<IntEbmType>(iByteCur);
   }

return_bad:;

   if(nullptr != pFillMem) {
      HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
      pHeaderDataSetShared->m_id = k_sharedDataSetErrorId;
   }
   return Error_IllegalParamValue;
}

EBM_API_BODY IntEbmType EBM_CALLING_CONVENTION SizeDataSetHeader(
   IntEbmType countFeatures,
   IntEbmType countWeights,
   IntEbmType countTargets
) {
   return AppendHeader(countFeatures, countWeights, countTargets, 0, nullptr);
}

EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION FillDataSetHeader(
   IntEbmType countFeatures,
   IntEbmType countWeights,
   IntEbmType countTargets,
   IntEbmType countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(Trace_Error, "ERROR FillDataSetHeader nullptr == fillMem");
      return Error_IllegalParamValue;
   }

   if(IsConvertError<size_t>(countBytesAllocated)) {
      LOG_0(Trace_Error, "ERROR FillDataSetHeader countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   const IntEbmType ret = AppendHeader(
      countFeatures, 
      countWeights, 
      countTargets, 
      cBytesAllocated, 
      static_cast<unsigned char *>(fillMem)
   );
   return static_cast<ErrorEbmType>(ret);
}

EBM_API_BODY IntEbmType EBM_CALLING_CONVENTION SizeFeature(
   IntEbmType countBins,
   BoolEbmType isMissing,
   BoolEbmType isUnknown,
   BoolEbmType isNominal,
   IntEbmType countSamples,
   const IntEbmType * binIndexes
) {
   return AppendFeature(
      countBins,
      isMissing,
      isUnknown,
      isNominal,
      countSamples,
      binIndexes,
      0,
      nullptr
   );
}

EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION FillFeature(
   IntEbmType countBins,
   BoolEbmType isMissing,
   BoolEbmType isUnknown,
   BoolEbmType isNominal,
   IntEbmType countSamples,
   const IntEbmType * binIndexes,
   IntEbmType countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(Trace_Error, "ERROR FillFeature nullptr == fillMem");
      return Error_IllegalParamValue;
   }

   if(IsConvertError<size_t>(countBytesAllocated)) {
      LOG_0(Trace_Error, "ERROR FillFeature countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   if(cBytesAllocated < k_cBytesHeaderNoOffset + sizeof(HeaderDataSetShared::m_offsets[0]) + sizeof(SharedStorageDataType)) {
      LOG_0(Trace_Error, "ERROR FillFeature cBytesAllocated < k_cBytesHeaderNoOffset + sizeof(HeaderDataSetShared::m_offsets[0]) + sizeof(SharedStorageDataType)");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }

   HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(fillMem);
   if(k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id) {
      LOG_0(Trace_Error, "ERROR FillFeature k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id");
      // don't set the header to bad since it's already set to something invalid and we don't know why
      return Error_IllegalParamValue;
   }

   const IntEbmType ret = AppendFeature(
      countBins,
      isMissing,
      isUnknown,
      isNominal,
      countSamples,
      binIndexes,
      cBytesAllocated,
      static_cast<unsigned char *>(fillMem)
   );
   return static_cast<ErrorEbmType>(ret);
}

EBM_API_BODY IntEbmType EBM_CALLING_CONVENTION SizeWeight(
   IntEbmType countSamples,
   const double * weights
) {
   return AppendWeight(
      countSamples,
      weights,
      0,
      nullptr
   );
}

EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION FillWeight(
   IntEbmType countSamples,
   const double * weights,
   IntEbmType countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(Trace_Error, "ERROR FillWeight nullptr == fillMem");
      return Error_IllegalParamValue;
   }

   if(IsConvertError<size_t>(countBytesAllocated)) {
      LOG_0(Trace_Error, "ERROR FillWeight countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   if(cBytesAllocated < k_cBytesHeaderNoOffset + sizeof(HeaderDataSetShared::m_offsets[0]) + sizeof(SharedStorageDataType)) {
      LOG_0(Trace_Error, "ERROR FillWeight cBytesAllocated < k_cBytesHeaderNoOffset + sizeof(HeaderDataSetShared::m_offsets[0]) + sizeof(SharedStorageDataType)");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }

   HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(fillMem);
   if(k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id) {
      LOG_0(Trace_Error, "ERROR FillWeight k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id");
      // don't set the header to bad since it's already set to something invalid and we don't know why
      return Error_IllegalParamValue;
   }

   const IntEbmType ret = AppendWeight(
      countSamples,
      weights,
      cBytesAllocated,
      static_cast<unsigned char *>(fillMem)
   );
   return static_cast<ErrorEbmType>(ret);
}

EBM_API_BODY IntEbmType EBM_CALLING_CONVENTION SizeClassificationTarget(
   IntEbmType countClasses,
   IntEbmType countSamples,
   const IntEbmType * targets
) {
   return AppendTarget(
      true,
      countClasses,
      countSamples,
      targets,
      0,
      nullptr
   );
}

EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION FillClassificationTarget(
   IntEbmType countClasses,
   IntEbmType countSamples,
   const IntEbmType * targets,
   IntEbmType countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(Trace_Error, "ERROR FillClassificationTarget nullptr == fillMem");
      return Error_IllegalParamValue;
   }

   if(IsConvertError<size_t>(countBytesAllocated)) {
      LOG_0(Trace_Error, "ERROR FillClassificationTarget countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   if(cBytesAllocated < k_cBytesHeaderNoOffset + sizeof(HeaderDataSetShared::m_offsets[0]) + sizeof(SharedStorageDataType)) {
      LOG_0(Trace_Error, "ERROR FillClassificationTarget cBytesAllocated < k_cBytesHeaderNoOffset + sizeof(HeaderDataSetShared::m_offsets[0]) + sizeof(SharedStorageDataType)");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }

   HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(fillMem);
   if(k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id) {
      LOG_0(Trace_Error, "ERROR FillClassificationTarget k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id");
      // don't set the header to bad since it's already set to something invalid and we don't know why
      return Error_IllegalParamValue;
   }

   const IntEbmType ret = AppendTarget(
      true,
      countClasses,
      countSamples,
      targets,
      cBytesAllocated,
      static_cast<unsigned char *>(fillMem)
   );
   return static_cast<ErrorEbmType>(ret);
}

EBM_API_BODY IntEbmType EBM_CALLING_CONVENTION SizeRegressionTarget(
   IntEbmType countSamples,
   const double * targets
) {
   return AppendTarget(
      false,
      0,
      countSamples,
      targets,
      0,
      nullptr
   );
}

EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION FillRegressionTarget(
   IntEbmType countSamples,
   const double * targets,
   IntEbmType countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(Trace_Error, "ERROR FillRegressionTarget nullptr == fillMem");
      return Error_IllegalParamValue;
   }

   if(IsConvertError<size_t>(countBytesAllocated)) {
      LOG_0(Trace_Error, "ERROR FillRegressionTarget countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   if(cBytesAllocated < k_cBytesHeaderNoOffset + sizeof(HeaderDataSetShared::m_offsets[0]) + sizeof(SharedStorageDataType)) {
      LOG_0(Trace_Error, "ERROR FillRegressionTarget cBytesAllocated < k_cBytesHeaderNoOffset + sizeof(HeaderDataSetShared::m_offsets[0]) + sizeof(SharedStorageDataType)");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }

   HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(fillMem);
   if(k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id) {
      LOG_0(Trace_Error, "ERROR FillRegressionTarget k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id");
      // don't set the header to bad since it's already set to something invalid and we don't know why
      return Error_IllegalParamValue;
   }

   const IntEbmType ret = AppendTarget(
      false,
      0,
      countSamples,
      targets,
      cBytesAllocated,
      static_cast<unsigned char *>(fillMem)
   );
   return static_cast<ErrorEbmType>(ret);
}

extern ErrorEbmType GetDataSetSharedHeader(
   const unsigned char * const pDataSetShared,
   size_t * const pcSamplesOut,
   size_t * const pcFeaturesOut,
   size_t * const pcWeightsOut,
   size_t * const pcTargetsOut
) {
   const HeaderDataSetShared * const pHeaderDataSetShared = 
      reinterpret_cast<const HeaderDataSetShared *>(pDataSetShared);
   
   if(k_sharedDataSetDoneId != pHeaderDataSetShared->m_id) {
      LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader k_sharedDataSetDoneId != pHeaderDataSetShared->m_id");
      return Error_IllegalParamValue;
   }

   const SharedStorageDataType countSamples = pHeaderDataSetShared->m_cSamples;
   if(IsConvertError<size_t>(countSamples)) {
      LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader IsConvertError<size_t>(countSamples)");
      return Error_IllegalParamValue;
   }
   const size_t cSamples = static_cast<size_t>(countSamples);
   *pcSamplesOut = cSamples;

   const SharedStorageDataType countFeatures = pHeaderDataSetShared->m_cFeatures;
   if(IsConvertError<size_t>(countFeatures)) {
      LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader IsConvertError<size_t>(countFeatures)");
      return Error_IllegalParamValue;
   }
   size_t cFeatures = static_cast<size_t>(countFeatures);
   *pcFeaturesOut = cFeatures;

   const SharedStorageDataType countWeights = pHeaderDataSetShared->m_cWeights;
   if(IsConvertError<size_t>(countWeights)) {
      LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader IsConvertError<size_t>(countWeights)");
      return Error_IllegalParamValue;
   }
   size_t cWeights = static_cast<size_t>(countWeights);
   *pcWeightsOut = cWeights;

   const SharedStorageDataType countTargets = pHeaderDataSetShared->m_cTargets;
   if(IsConvertError<size_t>(countTargets)) {
      LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader IsConvertError<size_t>(countTargets)");
      return Error_IllegalParamValue;
   }
   size_t cTargets = static_cast<size_t>(countTargets);
   *pcTargetsOut = cTargets;

   if(IsAddError(cFeatures, cWeights, cTargets)) {
      LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader IsAddError(cFeatures, cWeights, cTargets)");
      return Error_IllegalParamValue;
   }
   const size_t cOffsets = cFeatures + cWeights + cTargets;

   if(IsMultiplyError(cOffsets, sizeof(pHeaderDataSetShared->m_offsets[0]))) {
      LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader IsMultiplyError(cOffsets, sizeof(pHeaderDataSetShared->m_offsets[0]))");
      return Error_IllegalParamValue;
   }
   size_t iOffsetNext = cOffsets * sizeof(pHeaderDataSetShared->m_offsets[0]);
   if(IsAddError(k_cBytesHeaderNoOffset, iOffsetNext)) {
      LOG_0(Trace_Error, "ERROR IsHeaderError IsAddError(k_cBytesHeaderNoOffset, iOffsetNext)");
      return Error_IllegalParamValue;
   }
   iOffsetNext += k_cBytesHeaderNoOffset;

   const SharedStorageDataType * pOffset = pHeaderDataSetShared->m_offsets;
   while(0 != cFeatures) {
      const SharedStorageDataType indexOffsetCur = *pOffset;
      if(IsConvertError<size_t>(indexOffsetCur)) {
         LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader IsConvertError<size_t>(indexOffsetCur)");
         return Error_IllegalParamValue;
      }
      const size_t iOffsetCur = static_cast<size_t>(indexOffsetCur);

      if(iOffsetNext != iOffsetCur) {
         LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader iOffsetNext != offsetCur");
         return Error_IllegalParamValue;
      }
      ++pOffset;

      if(IsAddError(iOffsetNext, sizeof(FeatureDataSetShared))) {
         LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader IsAddError(iOffsetNext, sizeof(FeatureDataSetShared))");
         return Error_IllegalParamValue;
      }
      const FeatureDataSetShared * pFeatureDataSetShared =
         reinterpret_cast<const FeatureDataSetShared *>(pDataSetShared + iOffsetNext);
      iOffsetNext += sizeof(FeatureDataSetShared);

      const SharedStorageDataType id = pFeatureDataSetShared->m_id;
      if(!IsFeature(id)) {
         LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader !IsFeature(id)");
         return Error_IllegalParamValue;
      }

      const SharedStorageDataType countBins = pFeatureDataSetShared->m_cBins;
      if(IsConvertError<size_t>(countBins)) {
         LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader IsConvertError<size_t>(countBins)");
         return Error_IllegalParamValue;
      }
      if(IsSparseFeature(id)) {
         const size_t cBytesSparseHeaderNoOffset = offsetof(SparseFeatureDataSetShared, m_nonDefaults);

         if(IsAddError(iOffsetNext, cBytesSparseHeaderNoOffset)) {
            LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader IsAddError(iOffsetNext, cBytesSparseHeaderNoOffset)");
            return Error_IllegalParamValue;
         }
         const SparseFeatureDataSetShared * const pSparseFeatureDataSetShared =
            reinterpret_cast<const SparseFeatureDataSetShared *>(pDataSetShared + iOffsetNext);
         iOffsetNext += cBytesSparseHeaderNoOffset;

         // TODO: are there any limits to what defaultVal can be?
         //const SharedStorageDataType defaultVal = pSparseFeatureDataSetShared->m_defaultVal;
         //if(IsConvertError<size_t>(defaultVal)) {
         //   LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader IsConvertError<size_t>(defaultVal)");
         //   return Error_IllegalParamValue;
         //}

         const SharedStorageDataType countNonDefaults = pSparseFeatureDataSetShared->m_cNonDefaults;
         if(IsConvertError<size_t>(countNonDefaults)) {
            LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader IsConvertError<size_t>(countNonDefaults)");
            return Error_IllegalParamValue;
         }
         const size_t cNonDefaults = static_cast<size_t>(countNonDefaults);

         if(IsMultiplyError(cNonDefaults, sizeof(pSparseFeatureDataSetShared->m_nonDefaults[0]))) {
            LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader IsMultiplyError(cNonDefaults, sizeof(pSparseFeatureDataSetShared->m_nonDefaults[0]))");
            return Error_IllegalParamValue;
         }
         const size_t cTotalNonDefaults = cNonDefaults * sizeof(pSparseFeatureDataSetShared->m_nonDefaults[0]);
         if(IsAddError(iOffsetNext, cTotalNonDefaults)) {
            LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader IsAddError(iOffsetNext, cTotalNonDefaults)");
               return Error_IllegalParamValue;
         }
         iOffsetNext += cTotalNonDefaults;
      } else {
         if(IsMultiplyError(cSamples, sizeof(SharedStorageDataType))) {
            LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader IsMultiplyError(cSamples, sizeof(SharedStorageDataType))");
            return Error_IllegalParamValue;
         }
         const size_t cTotalMem = cSamples * sizeof(SharedStorageDataType);

         if(IsAddError(iOffsetNext, cTotalMem)) {
            LOG_0(Trace_Error, "ERROR GetDataSetSharedHeader IsAddError(iOffsetNext, cTotalMem)");
            return Error_IllegalParamValue;
         }
         iOffsetNext += cTotalMem;
      }
      --cFeatures;
   }

   // TODO: do the same kind of offset and content checking for weights and targets as we do above for features

   return Error_None;
}

EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION ExtractDataSetHeader(
   const void * dataSet,
   IntEbmType * countSamplesOut,
   IntEbmType * countFeaturesOut,
   IntEbmType * countWeightsOut,
   IntEbmType * countTargetsOut
) {
   ErrorEbmType error;

   if(nullptr == dataSet) {
      LOG_0(Trace_Error, "ERROR ExtractDataSetHeader nullptr == dataSet");
      return Error_IllegalParamValue;
   }

   size_t cSamples;
   size_t cFeatures;
   size_t cWeights;
   size_t cTargets;

   error = GetDataSetSharedHeader(
      static_cast<const unsigned char *>(dataSet),
      &cSamples,
      &cFeatures,
      &cWeights,
      &cTargets
   );
   if(Error_None != error) {
      // already logged
      return error;
   }

   if(IsConvertError<IntEbmType>(cSamples)) {
      // cSamples should have originally came to us as an IntEbmType, but check in case of corruption
      LOG_0(Trace_Error, "ERROR ExtractDataSetHeader IsConvertError<IntEbmType>(cSamples)");
      return Error_IllegalParamValue;
   }
   if(IsConvertError<IntEbmType>(cFeatures)) {
      // cFeatures should have originally came to us as an IntEbmType, but check in case of corruption
      LOG_0(Trace_Error, "ERROR ExtractDataSetHeader IsConvertError<IntEbmType>(cFeatures)");
      return Error_IllegalParamValue;
   }
   if(IsConvertError<IntEbmType>(cWeights)) {
      // cWeights should have originally came to us as an IntEbmType, but check in case of corruption
      LOG_0(Trace_Error, "ERROR ExtractDataSetHeader IsConvertError<IntEbmType>(cWeights)");
      return Error_IllegalParamValue;
   }
   if(IsConvertError<IntEbmType>(cTargets)) {
      // cTargets should have originally came to us as an IntEbmType, but check in case of corruption
      LOG_0(Trace_Error, "ERROR ExtractDataSetHeader IsConvertError<IntEbmType>(cTargets)");
      return Error_IllegalParamValue;
   }

   if(nullptr != countSamplesOut) {
      *countSamplesOut = static_cast<IntEbmType>(cSamples);
   }
   if(nullptr != countFeaturesOut) {
      *countFeaturesOut = static_cast<IntEbmType>(cFeatures);
   }
   if(nullptr != countWeightsOut) {
      *countWeightsOut = static_cast<IntEbmType>(cWeights);
   }
   if(nullptr != countTargetsOut) {
      *countTargetsOut = static_cast<IntEbmType>(cTargets);
   }

   return Error_None;
}

// TODO: make an inline wrapper that forces this to the correct type and have 2 differently named functions
// GetDataSetSharedFeature will return either (SparseFeatureDataSetSharedEntry *) or (SharedStorageDataType *)
extern const void * GetDataSetSharedFeature(
   const unsigned char * const pDataSetShared,
   const size_t iFeature,
   size_t * const pcBinsOut,
   bool * const pbMissingOut,
   bool * const pbUnknownOut,
   bool * const pbNominalOut,
   bool * const pbSparseOut,
   SharedStorageDataType * const pDefaultValSparseOut,
   size_t * const pcNonDefaultsSparseOut
) {
   const HeaderDataSetShared * const pHeaderDataSetShared = 
      reinterpret_cast<const HeaderDataSetShared *>(pDataSetShared);
   EBM_ASSERT(k_sharedDataSetDoneId == pHeaderDataSetShared->m_id);

   EBM_ASSERT(!IsConvertError<size_t>(pHeaderDataSetShared->m_cFeatures));
   EBM_ASSERT(iFeature < static_cast<size_t>(pHeaderDataSetShared->m_cFeatures));

   EBM_ASSERT(!IsMultiplyError(iFeature, sizeof(pHeaderDataSetShared->m_offsets[0])));
   const SharedStorageDataType indexMem = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iFeature];
   EBM_ASSERT(!IsConvertError<size_t>(indexMem));
   const size_t iMem = static_cast<size_t>(indexMem);

   const FeatureDataSetShared * pFeatureDataSetShared = 
      reinterpret_cast<const FeatureDataSetShared *>(pDataSetShared + iMem);

   const SharedStorageDataType id = pFeatureDataSetShared->m_id;
   EBM_ASSERT(IsFeature(id));
   *pbMissingOut = IsMissingFeature(id);
   *pbUnknownOut = IsUnknownFeature(id);
   *pbNominalOut = IsNominalFeature(id);
   const bool bSparse = IsSparseFeature(id);
   *pbSparseOut = bSparse;

   const SharedStorageDataType countBins = pFeatureDataSetShared->m_cBins;
   EBM_ASSERT(!IsConvertError<size_t>(countBins));
   const size_t cBins = static_cast<size_t>(countBins);
   *pcBinsOut = cBins;

   const void * pRet = reinterpret_cast<const void *>(pFeatureDataSetShared + 1);
   if(bSparse) {
      const SparseFeatureDataSetShared * const pSparseFeatureDataSetShared =
         reinterpret_cast<const SparseFeatureDataSetShared *>(pRet);

      *pDefaultValSparseOut = pSparseFeatureDataSetShared->m_defaultVal;
      const SharedStorageDataType countNonDefaults = pSparseFeatureDataSetShared->m_cNonDefaults;
      EBM_ASSERT(!IsConvertError<size_t>(countNonDefaults));
      const size_t cNonDefaults = static_cast<size_t>(countNonDefaults);
      *pcNonDefaultsSparseOut = cNonDefaults;
      pRet = reinterpret_cast<const void *>(pSparseFeatureDataSetShared->m_nonDefaults);
   }
   return pRet;
}

EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION ExtractBinCounts(
   const void * dataSet,
   IntEbmType countFeaturesVerify,
   IntEbmType * binCountsOut
) {
   if(nullptr == dataSet) {
      LOG_0(Trace_Error, "ERROR ExtractBinCounts nullptr == dataSet");
      return Error_IllegalParamValue;
   }

   if(IsConvertError<size_t>(countFeaturesVerify)) {
      LOG_0(Trace_Error, "ERROR ExtractBinCounts IsConvertError<size_t>(countFeaturesVerify)");
      return Error_IllegalParamValue;
   }
   const size_t cFeaturesVerify = static_cast<size_t>(countFeaturesVerify);

   const HeaderDataSetShared * const pHeaderDataSetShared =
      reinterpret_cast<const HeaderDataSetShared *>(dataSet);

   if(k_sharedDataSetDoneId != pHeaderDataSetShared->m_id) {
      LOG_0(Trace_Error, "ERROR ExtractBinCounts k_sharedDataSetDoneId != pHeaderDataSetShared->m_id");
      return Error_IllegalParamValue;
   }

   const SharedStorageDataType countFeatures = pHeaderDataSetShared->m_cFeatures;
   if(IsConvertError<size_t>(countFeatures)) {
      LOG_0(Trace_Error, "ERROR ExtractBinCounts IsConvertError<size_t>(countFeatures)");
      return Error_IllegalParamValue;
   }
   size_t cFeatures = static_cast<size_t>(countFeatures);

   if(cFeatures != cFeaturesVerify) {
      LOG_0(Trace_Error, "ERROR ExtractBinCounts cFeatures != cFeaturesVerify");
      return Error_IllegalParamValue;
   }
   if(size_t { 0 } != cFeatures) {
      if(nullptr == binCountsOut) {
         LOG_0(Trace_Error, "ERROR ExtractBinCounts nullptr == binCountsOut");
         return Error_IllegalParamValue;
      }

      const SharedStorageDataType * pOffset = pHeaderDataSetShared->m_offsets;
      IntEbmType * pcBins = binCountsOut;
      const IntEbmType * const pcBinsEnd = binCountsOut + cFeatures;
      do {
         const SharedStorageDataType indexOffsetCur = *pOffset;
         ++pOffset;

         if(IsConvertError<size_t>(indexOffsetCur)) {
            LOG_0(Trace_Error, "ERROR ExtractBinCounts IsConvertError<size_t>(indexOffsetCur)");
            return Error_IllegalParamValue;
         }
         const size_t iOffsetCur = static_cast<size_t>(indexOffsetCur);

         const FeatureDataSetShared * pFeatureDataSetShared =
            reinterpret_cast<const FeatureDataSetShared *>(static_cast<const char *>(dataSet) + iOffsetCur);

         const SharedStorageDataType id = pFeatureDataSetShared->m_id;
         if(!IsFeature(id)) {
            LOG_0(Trace_Error, "ERROR ExtractBinCounts !IsFeature(id)");
            return Error_IllegalParamValue;
         }

         const SharedStorageDataType countBins = pFeatureDataSetShared->m_cBins;
         if(IsConvertError<IntEbmType>(countBins)) {
            LOG_0(Trace_Error, "ERROR ExtractBinCounts IsConvertError<IntEbmType>(countBins)");
            return Error_IllegalParamValue;
         }

         *pcBins = static_cast<IntEbmType>(countBins);
         ++pcBins;
      } while(pcBinsEnd != pcBins);
   }
   return Error_None;
}

extern const FloatFast * GetDataSetSharedWeight(
   const unsigned char * const pDataSetShared,
   const size_t iWeight
) {
   const HeaderDataSetShared * const pHeaderDataSetShared =
      reinterpret_cast<const HeaderDataSetShared *>(pDataSetShared);
   EBM_ASSERT(k_sharedDataSetDoneId == pHeaderDataSetShared->m_id);

   const SharedStorageDataType countFeatures = pHeaderDataSetShared->m_cFeatures;
   EBM_ASSERT(!IsConvertError<size_t>(countFeatures));
   const size_t cFeatures = static_cast<size_t>(countFeatures);

   EBM_ASSERT(!IsConvertError<size_t>(pHeaderDataSetShared->m_cWeights));
   EBM_ASSERT(iWeight < static_cast<size_t>(pHeaderDataSetShared->m_cWeights));

   EBM_ASSERT(!IsAddError(cFeatures, iWeight));
   const size_t iOffset = cFeatures + iWeight;

   EBM_ASSERT(!IsMultiplyError(iOffset, sizeof(pHeaderDataSetShared->m_offsets[0])));
   const SharedStorageDataType indexMem = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset];
   EBM_ASSERT(!IsConvertError<size_t>(indexMem));
   const size_t iMem = static_cast<size_t>(indexMem);

   const WeightDataSetShared * pWeightDataSetShared =
      reinterpret_cast<const WeightDataSetShared *>(pDataSetShared + iMem);

   EBM_ASSERT(k_weightId == pWeightDataSetShared->m_id);

   return reinterpret_cast<const FloatFast *>(pWeightDataSetShared + 1);
}

// TODO: make an inline wrapper that forces this to the correct type and have 2 differently named functions
// GetDataSetSharedTarget returns (FloatFast *) for regression and (SharedStorageDataType *) for classification
extern const void * GetDataSetSharedTarget(
   const unsigned char * const pDataSetShared,
   const size_t iTarget,
   ptrdiff_t * const pcClassesOut
) {
   const HeaderDataSetShared * const pHeaderDataSetShared =
      reinterpret_cast<const HeaderDataSetShared *>(pDataSetShared);
   EBM_ASSERT(k_sharedDataSetDoneId == pHeaderDataSetShared->m_id);

   const SharedStorageDataType countFeatures = pHeaderDataSetShared->m_cFeatures;
   EBM_ASSERT(!IsConvertError<size_t>(countFeatures));
   const size_t cFeatures = static_cast<size_t>(countFeatures);

   const SharedStorageDataType countWeights = pHeaderDataSetShared->m_cWeights;
   EBM_ASSERT(!IsConvertError<size_t>(countWeights));
   const size_t cWeights = static_cast<size_t>(countWeights);

   EBM_ASSERT(!IsConvertError<size_t>(pHeaderDataSetShared->m_cTargets));
   EBM_ASSERT(iTarget < static_cast<size_t>(pHeaderDataSetShared->m_cTargets));

   EBM_ASSERT(!IsAddError(cFeatures, cWeights, iTarget));
   const size_t iOffset = cFeatures + cWeights + iTarget;

   EBM_ASSERT(!IsMultiplyError(iOffset, sizeof(pHeaderDataSetShared->m_offsets[0])));
   const SharedStorageDataType indexMem = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset];
   EBM_ASSERT(!IsConvertError<size_t>(indexMem));
   const size_t iMem = static_cast<size_t>(indexMem);

   const TargetDataSetShared * pTargetDataSetShared =
      reinterpret_cast<const TargetDataSetShared *>(pDataSetShared + iMem);

   const SharedStorageDataType id = pTargetDataSetShared->m_id;
   EBM_ASSERT(IsTarget(id));

   ptrdiff_t cClasses = k_regression;
   const void * pRet = reinterpret_cast<const void *>(pTargetDataSetShared + 1);
   if(IsClassificationTarget(id)) {
      const ClassificationTargetDataSetShared * const pClassificationTargetDataSetShared =
         reinterpret_cast<const ClassificationTargetDataSetShared *>(pRet);

      const SharedStorageDataType countClasses = pClassificationTargetDataSetShared->m_cClasses;
      EBM_ASSERT(!IsConvertError<ptrdiff_t>(countClasses));
      cClasses = static_cast<ptrdiff_t>(countClasses);
      EBM_ASSERT(0 <= cClasses); // 0 is possible with 0 samples
      pRet = reinterpret_cast<const void *>(pClassificationTargetDataSetShared + 1);
   }
   *pcClassesOut = cClasses;
   return pRet;
}

EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION ExtractTargetClasses(
   const void * dataSet,
   IntEbmType countTargetsVerify,
   IntEbmType * classCountsOut
) {
   if(nullptr == dataSet) {
      LOG_0(Trace_Error, "ERROR ExtractTargetClasses nullptr == dataSet");
      return Error_IllegalParamValue;
   }

   if(IsConvertError<size_t>(countTargetsVerify)) {
      LOG_0(Trace_Error, "ERROR ExtractTargetClasses IsConvertError<size_t>(countTargetsVerify)");
      return Error_IllegalParamValue;
   }
   const size_t cTargetsVerify = static_cast<size_t>(countTargetsVerify);

   const HeaderDataSetShared * const pHeaderDataSetShared =
      reinterpret_cast<const HeaderDataSetShared *>(dataSet);

   if(k_sharedDataSetDoneId != pHeaderDataSetShared->m_id) {
      LOG_0(Trace_Error, "ERROR ExtractTargetClasses k_sharedDataSetDoneId != pHeaderDataSetShared->m_id");
      return Error_IllegalParamValue;
   }

   const SharedStorageDataType countFeatures = pHeaderDataSetShared->m_cFeatures;
   if(IsConvertError<size_t>(countFeatures)) {
      LOG_0(Trace_Error, "ERROR ExtractTargetClasses IsConvertError<size_t>(countFeatures)");
      return Error_IllegalParamValue;
   }
   size_t cFeatures = static_cast<size_t>(countFeatures);

   const SharedStorageDataType countWeights = pHeaderDataSetShared->m_cWeights;
   if(IsConvertError<size_t>(countWeights)) {
      LOG_0(Trace_Error, "ERROR ExtractTargetClasses IsConvertError<size_t>(countWeights)");
      return Error_IllegalParamValue;
   }
   size_t cWeights = static_cast<size_t>(countWeights);

   const SharedStorageDataType countTargets = pHeaderDataSetShared->m_cTargets;
   if(IsConvertError<size_t>(countTargets)) {
      LOG_0(Trace_Error, "ERROR ExtractTargetClasses IsConvertError<size_t>(countTargets)");
      return Error_IllegalParamValue;
   }
   size_t cTargets = static_cast<size_t>(countTargets);

   if(cTargets != cTargetsVerify) {
      LOG_0(Trace_Error, "ERROR ExtractTargetClasses cTargets != cTargetsVerify");
      return Error_IllegalParamValue;
   }

   if(size_t { 0 } != cTargets) {
      if(nullptr == classCountsOut) {
         LOG_0(Trace_Error, "ERROR ExtractTargetClasses nullptr == classCountsOut");
         return Error_IllegalParamValue;
      }

      const SharedStorageDataType * pOffset = &pHeaderDataSetShared->m_offsets[cFeatures + cWeights];
      IntEbmType * pcClasses = classCountsOut;
      const IntEbmType * const pcClassesEnd = classCountsOut + cTargets;
      do {
         const SharedStorageDataType indexOffsetCur = *pOffset;
         ++pOffset;

         if(IsConvertError<size_t>(indexOffsetCur)) {
            LOG_0(Trace_Error, "ERROR ExtractTargetClasses IsConvertError<size_t>(indexOffsetCur)");
            return Error_IllegalParamValue;
         }
         const size_t iOffsetCur = static_cast<size_t>(indexOffsetCur);

         const TargetDataSetShared * pTargetDataSetShared =
            reinterpret_cast<const TargetDataSetShared *>(static_cast<const char *>(dataSet) + iOffsetCur);

         const SharedStorageDataType id = pTargetDataSetShared->m_id;
         if(!IsTarget(id)) {
            LOG_0(Trace_Error, "ERROR ExtractTargetClasses !IsTarget(id)");
            return Error_IllegalParamValue;
         }

         IntEbmType countClasses = IntEbmType { -1 };
         if(IsClassificationTarget(id)) {
            const ClassificationTargetDataSetShared * const pClassificationTargetDataSetShared =
               reinterpret_cast<const ClassificationTargetDataSetShared *>(pTargetDataSetShared + 1);

            const SharedStorageDataType cClasses = pClassificationTargetDataSetShared->m_cClasses;

            if(IsConvertError<IntEbmType>(cClasses)) {
               LOG_0(Trace_Error, "ERROR ExtractTargetClasses IsConvertError<IntEbmType>(cClasses)");
               return Error_IllegalParamValue;
            }

            countClasses = static_cast<IntEbmType>(cClasses);
            if(countClasses < IntEbmType { 0 }) {
               LOG_0(Trace_Error, "ERROR ExtractTargetClasses countClasses < IntEbmType { 0 }");
               return Error_IllegalParamValue;
            }
         }

         *pcClasses = countClasses;
         ++pcClasses;
      } while(pcClassesEnd != pcClasses);
   }
   return Error_None;
}

} // DEFINED_ZONE_NAME
