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

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// the stuff in this file is for handling a raw chunk of shared memory that our caller allocates and which we fill




// TODO PK Implement the following for memory efficiency and speed of initialization :
//   - OBSERVATION: We want sparse feature support in our booster since we don't need to access
//                  memory if there are long segments with just a single value
//   - OBSERVATION: our boosting algorithm is position independent, so we can sort the data by the target feature, which
//   -              helps us because we can move the class number into a loop count and not fetch the memory, and it allows
//                  us to elimiante a branch when calculating statistics since all samples will have the same target within a loop
//   - OBSERVATION: we'll be sorting on the target, so we can't sort primarily on intput features (secondary sort ok)
//                  So, sparse input features are not typically expected to clump into ranges of non - default parameters
//                  So, we won't use ranges in our representation, so our sparse feature representation will be
//                  class Sparse { size_t index; size_t val; }
//                  This representation is invariant to position, so we'll be able to pre-compute the size before sorting
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
//   - OBSERVATION: For templates, always put the more universal template featutres at the end, incase C++ changes such that variadic template/macros
//                  work for us someday (currently they only allow only just typenames or the same datatypes per parameter pack)
//   - OBSERVATION: For interaction detection, we'll want our template to be: <compilerLearningTypeOrCountTargetClasses, cDimensions, cDataItemsPerPack>
//                  The main reason is that we want to load data via SIMD, and we can't have branches in order to do that, so we can't bitpack each feature
//                  differently, so they all need to use the same number of bits per pack.
//   - OBSERVATION: For histogram creation and updating, we'll want our template to be: <compilerLearningTypeOrCountTargetClasses, cDataItemsPerPack>
//   - OBSERVATION: For partitioning, we'll want our template to be: <compilerLearningTypeOrCountTargetClasses, cDimensions>
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
//   - OBSERVATION: since sorting this data by target is so expensive (and the transpose to get it there), we'll create a special "all feature" data 
//                  represenation that is just features without feature groups.  This representation will be compressed per feature.
//                  and will include a reverse index to work back to the original unsorted indexes
//                  We'll generate the main/interaction training dataset from that directly when python passes us the train/validation set indexes and 
//                  the feature_groups.  We'll also generate train/validation duplicates of this dataset for interaction detection 
//                  (but for interactions we don't need the reverse index lookup)
//   - OBSERVATION: We should be able to completely preserve sparse data representations without expanding them, although we can also detect when dense 
//                  features should be sparsified in our own dataset
// 
// STEPS :
//   - C will fill a temporary index array in the RawArray, sort the data by target with the indexes, and secondarily by input features.  The index array 
//     will remain for reconstructing the original order
//   - Now the memory is read only from now on, and shareable, and the original order can be re-constructed
//   - first generate the mains train/validation boosting datasets, then create the interaction sets, then create the pair boosting datasets.  We only 
//     need these in memory one at a time
//   - FOR BOOSTING:
//     - pass the process shared read only RawArray, and the train/validation counts AND the feature_group definitions (we already have the feature 
//       definitions in the RawArray)
//   - FOR INTERACTIONS:
//     - pass the process shared read only RawArray, and the train/validation bools (we already have all feature definitions in the RawArray)
//     - C will do a first pass to determine how much memory it will need (sparse features can be divided unequally per train/validation set, so the 
//       train/validation can't be calculated without a first pass). We have all the data to do this!
//     - C will allocate the memory for the interaction detection dataset
//     - C will do a second pass to fill the data structure and return that to python (no need for a RawArray this time since it isn't shared)
//     - per the notes above, we will bit pack each feature by it's best fit size, and keep sparse features.  We're pretty much just copying data for 
//       interactions into the train/validations sets
//     - After re-ordering the bool lists to the original feature order, we process each feature using the bool to do a non-branching if statements 
//       to select whether each sample for that feature goes into the train or validation set, and handling increments





// header ids
constexpr static SharedStorageDataType k_sharedDataSetWorkingId = 0x46DB; // random 15 bit number
constexpr static SharedStorageDataType k_sharedDataSetErrorId = 0x103; // anything other than our normal id will work
constexpr static SharedStorageDataType k_sharedDataSetDoneId = 0x61E3; // random 15 bit number

// feature ids
constexpr static SharedStorageDataType k_categoricalFeatureBit = 0x1;
constexpr static SharedStorageDataType k_sparseFeatureBit = 0x2;
constexpr static SharedStorageDataType k_featureId = 0x2B44; // random 15 bit number with lower 2 bits set to zero

// weight ids
constexpr static SharedStorageDataType k_weightId = 0x61FB; // random 15 bit number

// target ids
constexpr static SharedStorageDataType k_classificationBit = 0x1;
constexpr static SharedStorageDataType k_targetId = 0x5A92; // random 15 bit number with lowest bit set to zero

INLINE_ALWAYS static bool IsFeature(const SharedStorageDataType id) noexcept {
   return (k_categoricalFeatureBit | k_sparseFeatureBit | k_featureId) ==
      ((k_categoricalFeatureBit | k_sparseFeatureBit) | id);
}
INLINE_ALWAYS static bool IsCategoricalFeature(const SharedStorageDataType id) noexcept {
   static_assert(0 == (k_categoricalFeatureBit & k_featureId), "k_featureId should not be categorical");
   EBM_ASSERT(IsFeature(id));
   return 0 != (k_categoricalFeatureBit & id);
}
INLINE_ALWAYS static bool IsSparseFeature(const SharedStorageDataType id) noexcept {
   static_assert(0 == (k_sparseFeatureBit & k_featureId), "k_featureId should not be sparse");
   EBM_ASSERT(IsFeature(id));
   return 0 != (k_sparseFeatureBit & id);
}
INLINE_ALWAYS static SharedStorageDataType GetFeatureId(const bool bCategorical, const bool bSparse) noexcept {
   return k_featureId | 
      (bCategorical ? k_categoricalFeatureBit : SharedStorageDataType { 0 }) | 
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
constexpr static size_t k_cBytesHeaderNoOffset = sizeof(HeaderDataSetShared) - sizeof(HeaderDataSetShared::m_offsets[0]);
constexpr static SharedStorageDataType k_unfilledOffset = k_cBytesHeaderNoOffset - 1;

struct FeatureDataSetShared {
   SharedStorageDataType m_id; // dense or sparse?  categorical or not?
   SharedStorageDataType m_cBins;
};
static_assert(std::is_standard_layout<FeatureDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<FeatureDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");

// No DenseFeatureDataSetShared required

struct SparseFeatureDataSetSharedEntry {
   SharedStorageDataType m_iSample;
   SharedStorageDataType m_nonDefaultValue;
};
static_assert(std::is_standard_layout<SparseFeatureDataSetSharedEntry>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<SparseFeatureDataSetSharedEntry>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");

struct SparseFeatureDataSetShared {
   // TODO: implement sparse features
   SharedStorageDataType m_defaultValue;
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
   SharedStorageDataType m_cTargetClasses;
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
   EBM_ASSERT(sizeof(HeaderDataSetShared) + sizeof(SharedStorageDataType) <= cBytesAllocated); // checked by our caller
   EBM_ASSERT(nullptr != pFillMem);

   const HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<const HeaderDataSetShared *>(pFillMem);
   EBM_ASSERT(k_sharedDataSetWorkingId == pHeaderDataSetShared->m_id); // checked by our caller

   const SharedStorageDataType countFeatures = pHeaderDataSetShared->m_cFeatures;
   if(IsConvertError<size_t>(countFeatures)) {
      // we're being untrusting of the caller manipulating the memory improperly here
      LOG_0(TraceLevelError, "ERROR IsHeaderError countFeatures is outside the range of a valid index");
      return true;
   }
   const size_t cFeatures = static_cast<size_t>(countFeatures);

   const SharedStorageDataType countWeights = pHeaderDataSetShared->m_cWeights;
   if(IsConvertError<size_t>(countWeights)) {
      // we're being untrusting of the caller manipulating the memory improperly here
      LOG_0(TraceLevelError, "ERROR IsHeaderError countWeights is outside the range of a valid index");
      return true;
   }
   const size_t cWeights = static_cast<size_t>(countWeights);

   const SharedStorageDataType countTargets = pHeaderDataSetShared->m_cTargets;
   if(IsConvertError<size_t>(countTargets)) {
      // we're being untrusting of the caller manipulating the memory improperly here
      LOG_0(TraceLevelError, "ERROR IsHeaderError countTargets is outside the range of a valid index");
      return true;
   }
   const size_t cTargets = static_cast<size_t>(countTargets);

   if(IsAddError(cFeatures, cWeights, cTargets)) {
      LOG_0(TraceLevelError, "ERROR IsHeaderError IsAddError(cFeatures, cWeights, cTargets)");
      return true;
   }
   const size_t cOffsets = cFeatures + cWeights + cTargets;

   if(IsMultiplyError(sizeof(HeaderDataSetShared::m_offsets[0]), cOffsets)) {
      LOG_0(TraceLevelError, "ERROR IsHeaderError IsMultiplyError(sizeof(HeaderDataSetShared::m_offsets[0]), cOffsets)");
      return true;
   }
   const size_t cBytesOffsets = sizeof(HeaderDataSetShared::m_offsets[0]) * cOffsets;

   if(IsAddError(k_cBytesHeaderNoOffset, cBytesOffsets)) {
      LOG_0(TraceLevelError, "ERROR IsHeaderError IsAddError(k_cBytesHeaderNoOffset, cBytesOffsets)");
      return true;
   }
   const size_t cBytesHeader = k_cBytesHeaderNoOffset + cBytesOffsets;

   if(cBytesAllocated - sizeof(SharedStorageDataType) < cBytesHeader) {
      LOG_0(TraceLevelError, "ERROR IsHeaderError cBytesAllocated - sizeof(SharedStorageDataType) < cBytesHeader");
      return true;
   }

   const SharedStorageDataType indexByte0 = pHeaderDataSetShared->m_offsets[0];
   if(IsConvertError<size_t>(indexByte0)) {
      // we're being untrusting of the caller manipulating the memory improperly here
      LOG_0(TraceLevelError, "ERROR IsHeaderError indexByte0 is outside the range of a valid index");
      return true;
   }
   const size_t iByte0 = static_cast<size_t>(indexByte0);

   if(iByte0 != cBytesHeader) {
      // we're being untrusting of the caller manipulating the memory improperly here
      LOG_0(TraceLevelError, "ERROR IsHeaderError iByte0 != cBytesHeader");
      return true;
   }

   const SharedStorageDataType * const pInternalState =
      reinterpret_cast<const SharedStorageDataType *>(pFillMem + cBytesAllocated - sizeof(SharedStorageDataType));

   const SharedStorageDataType internalState = *pInternalState;
   if(IsConvertError<size_t>(internalState)) {
      LOG_0(TraceLevelError, "ERROR IsHeaderError opaqueState invalid");
      return true;
   }
   const size_t iOffset = static_cast<size_t>(internalState);

   if(cOffsets <= iOffset) {
      LOG_0(TraceLevelError, "ERROR IsHeaderError cOffsets <= iOffset");
      return true;
   }

   if(size_t { 0 } == iOffset) {
      if(SharedStorageDataType { 0 } != pHeaderDataSetShared->m_cSamples) {
         LOG_0(TraceLevelError, "ERROR IsHeaderError SharedStorageDataType { 0 } != pHeaderDataSetShared->m_cSamples");
         return true;
      }
   } else {
      if(pHeaderDataSetShared->m_cSamples != static_cast<SharedStorageDataType>(cSamples)) {
         LOG_0(TraceLevelError, "ERROR IsHeaderError pHeaderDataSetShared->m_cSamples != cSamples");
         return true;
      }

      // if iOffset is 1, we'll just check this once again without any issues
      const SharedStorageDataType indexHighestOffsetPrev = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset - 1];
      if(IsConvertError<size_t>(indexHighestOffsetPrev)) {
         // we're being untrusting of the caller manipulating the memory improperly here
         LOG_0(TraceLevelError, "ERROR IsHeaderError indexHighestOffsetPrev is outside the range of a valid index");
         return true;
      }
      const size_t iHighestOffsetPrev = static_cast<size_t>(indexHighestOffsetPrev);

      if(iHighestOffsetPrev < iByte0) {
         // we're being untrusting of the caller manipulating the memory improperly here
         LOG_0(TraceLevelError, "ERROR IsHeaderError iHighestOffsetPrev < iByte0");
         return true;
      }

      const SharedStorageDataType indexHighestOffset = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset];
      if(IsConvertError<size_t>(indexHighestOffset)) {
         // we're being untrusting of the caller manipulating the memory improperly here
         LOG_0(TraceLevelError, "ERROR IsHeaderError indexHighestOffset is outside the range of a valid index");
         return true;
      }
      const size_t iHighestOffset = static_cast<size_t>(indexHighestOffset);

      if(iHighestOffset < iHighestOffsetPrev) {
         // we're being untrusting of the caller manipulating the memory improperly here
         LOG_0(TraceLevelError, "ERROR IsHeaderError iHighestOffset < iHighestOffsetPrev");
         return true;
      }

      // through associativity since iByte0 <= iHighestOffsetPrev && iHighestOffsetPrev <= iHighestOffset
      EBM_ASSERT(iByte0 <= iHighestOffset);
   }

   const size_t iOffsetNext = iOffset + 1; // we verified iOffset < cOffsets above, so this is guaranteed to work
   if(iOffsetNext != cOffsets) {
      const SharedStorageDataType indexHighestOffsetNext = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffsetNext];
      if(k_unfilledOffset != indexHighestOffsetNext) {
         LOG_0(TraceLevelError, "ERROR IsHeaderError k_unfilledOffset != indexHighestOffsetNext");
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
      TraceLevelInfo,
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
      LOG_0(TraceLevelError, "ERROR AppendHeader countFeatures is outside the range of a valid index");
      return Error_IllegalParamValue;
   }
   const size_t cFeatures = static_cast<size_t>(countFeatures);

   if(IsConvertErrorDual<size_t, SharedStorageDataType>(countWeights)) {
      LOG_0(TraceLevelError, "ERROR AppendHeader countWeights is outside the range of a valid index");
      return Error_IllegalParamValue;
   }
   const size_t cWeights = static_cast<size_t>(countWeights);

   if(IsConvertErrorDual<size_t, SharedStorageDataType>(countTargets)) {
      LOG_0(TraceLevelError, "ERROR AppendHeader countTargets is outside the range of a valid index");
      return Error_IllegalParamValue;
   }
   const size_t cTargets = static_cast<size_t>(countTargets);

   if(IsAddError(cFeatures, cWeights, cTargets)) {
      LOG_0(TraceLevelError, "ERROR AppendHeader IsAddError(cFeatures, cWeights, cTargets)");
      return Error_IllegalParamValue;
   }
   const size_t cOffsets = cFeatures + cWeights + cTargets;

   if(IsMultiplyError(sizeof(HeaderDataSetShared::m_offsets[0]), cOffsets)) {
      LOG_0(TraceLevelError, "ERROR AppendHeader IsMultiplyError(sizeof(HeaderDataSetShared::m_offsets[0]), cOffsets)");
      return Error_IllegalParamValue;
   }
   const size_t cBytesOffsets = sizeof(HeaderDataSetShared::m_offsets[0]) * cOffsets;

   if(IsAddError(k_cBytesHeaderNoOffset, cBytesOffsets)) {
      LOG_0(TraceLevelError, "ERROR AppendHeader IsAddError(k_cBytesHeaderNoOffset, cBytesOffsets)");
      return Error_IllegalParamValue;
   }
   const size_t cBytesHeader = k_cBytesHeaderNoOffset + cBytesOffsets;

   if(IsConvertError<SharedStorageDataType>(cBytesHeader)) {
      LOG_0(TraceLevelError, "ERROR AppendHeader cBytesHeader is outside the range of a valid size");
      return Error_IllegalParamValue;
   }

   if(nullptr != pFillMem) {
      if(size_t { 0 } == cOffsets) {
         if(cBytesAllocated != cBytesHeader) {
            LOG_0(TraceLevelError, "ERROR AppendHeader buffer size and fill size do not agree");
            return Error_IllegalParamValue;
         }
      } else {
         if(cBytesAllocated - sizeof(SharedStorageDataType) < cBytesHeader) {
            LOG_0(TraceLevelError, "ERROR AppendHeader cBytesAllocated - sizeof(SharedStorageDataType) < cBytesHeader");
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
      LOG_0(TraceLevelError, "ERROR SizeDataSetHeader IsConvertError<IntEbmType>(cBytesHeader)");
      return Error_OutOfMemory;
   }
   return cBytesHeader;
}

static bool DecideIfSparse(const size_t cSamples, const IntEbmType * aBinnedData) {
   // For sparsity in the data set shared memory the only thing that matters is compactness since we don't use
   // this memory in any high performance loops

   UNUSED(cSamples);
   UNUSED(aBinnedData);

   // TODO: evalute the data to decide if the feature should be sparse or not
   return false;
}

static IntEbmType AppendFeature(
   const BoolEbmType categorical,
   const IntEbmType countBins,
   const IntEbmType countSamples,
   const IntEbmType * aBinnedData,
   const size_t cBytesAllocated,
   unsigned char * const pFillMem
) {
   EBM_ASSERT(size_t { 0 } == cBytesAllocated && nullptr == pFillMem || 
      nullptr != pFillMem && sizeof(HeaderDataSetShared) + sizeof(SharedStorageDataType) <= cBytesAllocated);

   LOG_N(
      TraceLevelInfo,
      "Entered AppendFeature: "
      "categorical=%" BoolEbmTypePrintf ", "
      "countBins=%" IntEbmTypePrintf ", "
      "countSamples=%" IntEbmTypePrintf ", "
      "aBinnedData=%p, "
      "cBytesAllocated=%zu, "
      "pFillMem=%p"
      ,
      categorical,
      countBins,
      countSamples,
      static_cast<const void *>(aBinnedData),
      cBytesAllocated,
      static_cast<void *>(pFillMem)
   );

   {
      if(EBM_FALSE != categorical && EBM_TRUE != categorical) {
         LOG_0(TraceLevelError, "ERROR AppendFeature categorical is not EBM_FALSE or EBM_TRUE");
         goto return_bad;
      }

      if(IsConvertErrorDual<size_t, SharedStorageDataType>(countBins)) {
         LOG_0(TraceLevelError, "ERROR AppendFeature countBins is outside the range of a valid index");
         goto return_bad;
      }

      if(IsConvertErrorDual<size_t, SharedStorageDataType>(countSamples)) {
         LOG_0(TraceLevelError, "ERROR AppendFeature countSamples is outside the range of a valid index");
         goto return_bad;
      }
      const size_t cSamples = static_cast<size_t>(countSamples);

      bool bSparse = false;
      if(size_t { 0 } != cSamples) {
         if(nullptr == aBinnedData) {
            LOG_0(TraceLevelError, "ERROR AppendFeature nullptr == aBinnedData");
            goto return_bad;
         }

         // TODO: handle sparse data someday
         bSparse = DecideIfSparse(cSamples, aBinnedData);
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
            LOG_0(TraceLevelError, "ERROR AppendFeature cFeatures <= iOffset");
            goto return_bad;
         }

         const size_t iHighestOffset = static_cast<size_t>(ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset]);

         if(IsAddError(iByteCur, iHighestOffset)) {
            LOG_0(TraceLevelError, "ERROR AppendFeature IsAddError(iByteCur, iHighestOffset)");
            goto return_bad;
         }
         iByteCur += iHighestOffset; // if we're going to access FeatureDataSetShared, then check if we have the space
         if(cBytesAllocated < iByteCur) {
            LOG_0(TraceLevelError, "ERROR AppendFeature cBytesAllocated < iByteCur");
            goto return_bad;
         }

         EBM_ASSERT(size_t { 0 } == iOffset && SharedStorageDataType { 0 } == pHeaderDataSetShared->m_cSamples ||
            static_cast<SharedStorageDataType>(cSamples) == pHeaderDataSetShared->m_cSamples);
         pHeaderDataSetShared->m_cSamples = static_cast<SharedStorageDataType>(cSamples);

         FeatureDataSetShared * pFeatureDataSetShared = reinterpret_cast<FeatureDataSetShared *>(pFillMem + iHighestOffset);
         pFeatureDataSetShared->m_id = GetFeatureId(EBM_FALSE != categorical, bSparse);
         pFeatureDataSetShared->m_cBins = static_cast<SharedStorageDataType>(countBins);
      }

      if(size_t { 0 } != cSamples) {
         if(IsMultiplyError(sizeof(SharedStorageDataType), cSamples)) {
            LOG_0(TraceLevelError, "ERROR AppendFeature IsMultiplyError(sizeof(SharedStorageDataType), cSamples)");
            goto return_bad;
         }
         const size_t cBytesAllSamples = sizeof(SharedStorageDataType) * cSamples;

         if(IsAddError(iByteCur, cBytesAllSamples)) {
            LOG_0(TraceLevelError, "ERROR AppendFeature IsAddError(iByteCur, cBytesAllSamples)");
            goto return_bad;
         }
         const size_t iByteNext = iByteCur + cBytesAllSamples;

         if(nullptr != pFillMem) {
            if(cBytesAllocated < iByteNext) {
               LOG_0(TraceLevelError, "ERROR AppendFeature cBytesAllocated < iByteNext");
               goto return_bad;
            }

            if(IsMultiplyError(sizeof(aBinnedData[0]), cSamples)) {
               LOG_0(TraceLevelError, "ERROR AppendFeature IsMultiplyError(sizeof(aBinnedData[0]), cSamples)");
               goto return_bad;
            }
            const IntEbmType * pBinnedData = aBinnedData;
            const IntEbmType * const pBinnedDataEnd = aBinnedData + cSamples;
            SharedStorageDataType * pFillData = reinterpret_cast<SharedStorageDataType *>(pFillMem + iByteCur);
            do {
               const IntEbmType binnedData = *pBinnedData;
               if(binnedData < IntEbmType { 0 }) {
                  LOG_0(TraceLevelError, "ERROR AppendFeature binnedData can't be negative");
                  goto return_bad;
               }
               if(countBins <= binnedData) {
                  LOG_0(TraceLevelError, "ERROR AppendFeature countBins <= binnedData");
                  goto return_bad;
               }
               // since countBins can be converted to these, so now can binnedData
               EBM_ASSERT(!IsConvertError<size_t>(binnedData));
               EBM_ASSERT(!IsConvertError<SharedStorageDataType>(binnedData));

               // TODO: bit compact this
               *pFillData = static_cast<SharedStorageDataType>(binnedData);

               ++pFillData;
               ++pBinnedData;
            } while(pBinnedDataEnd != pBinnedData);
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
               LOG_0(TraceLevelError, "ERROR AppendFeature buffer size and fill size do not agree");
               goto return_bad;
            }

            LockDataSetShared(pFillMem);
         } else {
            if(cBytesAllocated - sizeof(SharedStorageDataType) < iByteCur) {
               LOG_0(TraceLevelError, "ERROR AppendFeature cBytesAllocated - sizeof(SharedStorageDataType) < iByteNext");
               goto return_bad;
            }

            if(IsConvertError<SharedStorageDataType>(iOffset)) {
               LOG_0(TraceLevelError, "ERROR AppendFeature IsConvertError<IntEbmType>(iOffset)");
               goto return_bad;
            }
            if(IsConvertError<SharedStorageDataType>(iByteCur)) {
               LOG_0(TraceLevelError, "ERROR AppendFeature IsConvertError<SharedStorageDataType>(iByteCur)");
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
         LOG_0(TraceLevelError, "ERROR SizeFeature IsConvertError<IntEbmType>(cBytes)");
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
   const FloatEbmType * aWeights,
   const size_t cBytesAllocated,
   unsigned char * const pFillMem
) {
   EBM_ASSERT(size_t { 0 } == cBytesAllocated && nullptr == pFillMem ||
      nullptr != pFillMem && sizeof(HeaderDataSetShared) + sizeof(SharedStorageDataType) <= cBytesAllocated);

   LOG_N(
      TraceLevelInfo,
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
         LOG_0(TraceLevelError, "ERROR AppendWeight countSamples is outside the range of a valid index");
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
            LOG_0(TraceLevelError, "ERROR AppendWeight iOffset < cFeatures");
            goto return_bad;
         }
         if(cFeatures + cWeights <= iOffset) {
            LOG_0(TraceLevelError, "ERROR AppendWeight cFeatures + cWeights <= iOffset");
            goto return_bad;
         }

         const size_t iHighestOffset = static_cast<size_t>(ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset]);

         if(IsAddError(iByteCur, iHighestOffset)) {
            LOG_0(TraceLevelError, "ERROR AppendWeight IsAddError(iByteCur, iHighestOffset)");
            goto return_bad;
         }
         iByteCur += iHighestOffset; // if we're going to access FeatureDataSetShared, then check if we have the space
         if(cBytesAllocated < iByteCur) {
            LOG_0(TraceLevelError, "ERROR AppendWeight cBytesAllocated < iByteCur");
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
            LOG_0(TraceLevelError, "ERROR AppendWeight nullptr == aWeights");
            goto return_bad;
         }

         if(IsMultiplyError(sizeof(FloatEbmType), cSamples)) {
            LOG_0(TraceLevelError, "ERROR AppendWeight IsMultiplyError(sizeof(FloatEbmType), cSamples)");
            goto return_bad;
         }
         const size_t cBytesAllSamples = sizeof(FloatEbmType) * cSamples;

         if(IsAddError(iByteCur, cBytesAllSamples)) {
            LOG_0(TraceLevelError, "ERROR AppendWeight IsAddError(iByteCur, cBytesAllSamples)");
            goto return_bad;
         }
         const size_t iByteNext = iByteCur + cBytesAllSamples;
         if(nullptr != pFillMem) {
            if(cBytesAllocated < iByteNext) {
               LOG_0(TraceLevelError, "ERROR AppendWeight cBytesAllocated < iByteNext");
               goto return_bad;
            }

            EBM_ASSERT(!IsMultiplyError(sizeof(FloatEbmType), cSamples)); // checked above
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
               LOG_0(TraceLevelError, "ERROR AppendWeight buffer size and fill size do not agree");
               goto return_bad;
            }

            LockDataSetShared(pFillMem);
         } else {
            if(cBytesAllocated - sizeof(SharedStorageDataType) < iByteCur) {
               LOG_0(TraceLevelError, "ERROR AppendWeight cBytesAllocated - sizeof(SharedStorageDataType) < iByteCur");
               goto return_bad;
            }

            if(IsConvertError<SharedStorageDataType>(iOffset)) {
               LOG_0(TraceLevelError, "ERROR AppendWeight IsConvertError<IntEbmType>(iOffset)");
               goto return_bad;
            }
            if(IsConvertError<SharedStorageDataType>(iByteCur)) {
               LOG_0(TraceLevelError, "ERROR AppendWeight IsConvertError<SharedStorageDataType>(iByteCur)");
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
         LOG_0(TraceLevelError, "ERROR SizeFeature IsConvertError<IntEbmType>(cBytes)");
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
   const IntEbmType countTargetClasses,
   const IntEbmType countSamples,
   const void * aTargets,
   const size_t cBytesAllocated,
   unsigned char * const pFillMem
) {
   EBM_ASSERT(size_t { 0 } == cBytesAllocated && nullptr == pFillMem ||
      nullptr != pFillMem && sizeof(HeaderDataSetShared) + sizeof(SharedStorageDataType) <= cBytesAllocated);

   LOG_N(
      TraceLevelInfo,
      "Entered AppendTarget: "
      "bClassification=%" BoolEbmTypePrintf ", "
      "countTargetClasses=%" IntEbmTypePrintf ", "
      "countSamples=%" IntEbmTypePrintf ", "
      "aTargets=%p, "
      "cBytesAllocated=%zu, "
      "pFillMem=%p"
      ,
      bClassification ? EBM_TRUE : EBM_FALSE,
      countTargetClasses,
      countSamples,
      static_cast<const void *>(aTargets),
      cBytesAllocated,
      static_cast<void *>(pFillMem)
   );

   {
      if(IsConvertErrorDual<size_t, SharedStorageDataType>(countTargetClasses)) {
         LOG_0(TraceLevelError, "ERROR AppendTarget countTargetClasses is outside the range of a valid index");
         goto return_bad;
      }
      if(IsConvertErrorDual<size_t, SharedStorageDataType>(countSamples)) {
         LOG_0(TraceLevelError, "ERROR AppendTarget countSamples is outside the range of a valid index");
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
            LOG_0(TraceLevelError, "ERROR AppendTarget iOffset < cFeatures + cWeights");
            goto return_bad;
         }

         const size_t iHighestOffset = static_cast<size_t>(ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset]);

         if(IsAddError(iByteCur, iHighestOffset)) {
            LOG_0(TraceLevelError, "ERROR AppendTarget IsAddError(iByteCur, iHighestOffset)");
            goto return_bad;
         }
         iByteCur += iHighestOffset; // if we're going to access FeatureDataSetShared, then check if we have the space
         if(cBytesAllocated < iByteCur) {
            LOG_0(TraceLevelError, "ERROR AppendTarget cBytesAllocated < iByteCur");
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
            pClassificationTargetDataSetShared->m_cTargetClasses = static_cast<SharedStorageDataType>(countTargetClasses);
         }
      }

      if(size_t { 0 } != cSamples) {
         if(nullptr == aTargets) {
            LOG_0(TraceLevelError, "ERROR AppendTarget nullptr == aTargets");
            goto return_bad;
         }

         size_t cBytesAllSamples;
         if(bClassification) {
            if(IsMultiplyError(sizeof(SharedStorageDataType), cSamples)) {
               LOG_0(TraceLevelError, "ERROR AppendTarget IsMultiplyError(sizeof(SharedStorageDataType), cSamples)");
               goto return_bad;
            }
            cBytesAllSamples = sizeof(SharedStorageDataType) * cSamples;
         } else {
            if(IsMultiplyError(sizeof(FloatEbmType), cSamples)) {
               LOG_0(TraceLevelError, "ERROR AppendTarget IsMultiplyError(sizeof(FloatEbmType), cSamples)");
               goto return_bad;
            }
            cBytesAllSamples = sizeof(FloatEbmType) * cSamples;
         }
         if(IsAddError(iByteCur, cBytesAllSamples)) {
            LOG_0(TraceLevelError, "ERROR AppendTarget IsAddError(iByteCur, cBytesAllSamples)");
            goto return_bad;
         }
         const size_t iByteNext = iByteCur + cBytesAllSamples;
         if(nullptr != pFillMem) {
            if(cBytesAllocated < iByteNext) {
               LOG_0(TraceLevelError, "ERROR AppendTarget cBytesAllocated < iByteNext");
               goto return_bad;
            }
            if(bClassification) {
               const IntEbmType * pTarget = reinterpret_cast<const IntEbmType *>(aTargets);
               if(IsMultiplyError(sizeof(pTarget[0]), cSamples)) {
                  LOG_0(TraceLevelError, "ERROR AppendTarget IsMultiplyError(sizeof(SharedStorageDataType), cSamples)");
                  goto return_bad;
               }
               const IntEbmType * const pTargetsEnd = pTarget + cSamples;
               SharedStorageDataType * pFillData = reinterpret_cast<SharedStorageDataType *>(pFillMem + iByteCur);
               do {
                  const IntEbmType target = *pTarget;
                  if(target < IntEbmType { 0 }) {
                     LOG_0(TraceLevelError, "ERROR AppendTarget classification target can't be negative");
                     goto return_bad;
                  }
                  if(countTargetClasses <= target) {
                     LOG_0(TraceLevelError, "ERROR AppendTarget countTargetClasses <= target");
                     goto return_bad;
                  }
                  // since countTargetClasses can be converted to these, so now can target
                  EBM_ASSERT(!IsConvertError<size_t>(target));
                  EBM_ASSERT(!IsConvertError<SharedStorageDataType>(target));
               
                  // TODO: sort by the target and then convert the target to a count of each index
                  *pFillData = static_cast<SharedStorageDataType>(target);
               
                  ++pFillData;
                  ++pTarget;
               } while(pTargetsEnd != pTarget);
               EBM_ASSERT(reinterpret_cast<unsigned char *>(pFillData) == pFillMem + iByteNext);
            } else {
               EBM_ASSERT(!IsMultiplyError(sizeof(FloatEbmType), cSamples)); // checked above
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
               LOG_0(TraceLevelError, "ERROR AppendTarget buffer size and fill size do not agree");
               goto return_bad;
            }

            LockDataSetShared(pFillMem);
         } else {
            if(cBytesAllocated - sizeof(SharedStorageDataType) < iByteCur) {
               LOG_0(TraceLevelError, "ERROR AppendTarget cBytesAllocated - sizeof(SharedStorageDataType) < iByteCur");
               goto return_bad;
            }
            if(IsConvertError<SharedStorageDataType>(iOffset)) {
               LOG_0(TraceLevelError, "ERROR AppendTarget IsConvertError<IntEbmType>(iOffset)");
               goto return_bad;
            }
            if(IsConvertError<SharedStorageDataType>(iByteCur)) {
               LOG_0(TraceLevelError, "ERROR AppendTarget IsConvertError<SharedStorageDataType>(iByteCur)");
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
         LOG_0(TraceLevelError, "ERROR SizeFeature IsConvertError<IntEbmType>(cBytes)");
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

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION SizeDataSetHeader(
   IntEbmType countFeatures,
   IntEbmType countWeights,
   IntEbmType countTargets
) {
   return AppendHeader(countFeatures, countWeights, countTargets, 0, nullptr);
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillDataSetHeader(
   IntEbmType countFeatures,
   IntEbmType countWeights,
   IntEbmType countTargets,
   IntEbmType countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(TraceLevelError, "ERROR FillDataSetHeader nullptr == fillMem");
      return Error_IllegalParamValue;
   }

   if(IsConvertError<size_t>(countBytesAllocated)) {
      LOG_0(TraceLevelError, "ERROR FillDataSetHeader countBytesAllocated is outside the range of a valid size");
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

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION SizeFeature(
   BoolEbmType categorical,
   IntEbmType countBins,
   IntEbmType countSamples,
   const IntEbmType * binnedData
) {
   return AppendFeature(
      categorical,
      countBins,
      countSamples,
      binnedData,
      0,
      nullptr
   );
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillFeature(
   BoolEbmType categorical,
   IntEbmType countBins,
   IntEbmType countSamples,
   const IntEbmType * binnedData,
   IntEbmType countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(TraceLevelError, "ERROR FillFeature nullptr == fillMem");
      return Error_IllegalParamValue;
   }

   if(IsConvertError<size_t>(countBytesAllocated)) {
      LOG_0(TraceLevelError, "ERROR FillFeature countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   if(cBytesAllocated < sizeof(HeaderDataSetShared) + sizeof(SharedStorageDataType)) {
      LOG_0(TraceLevelError, "ERROR FillFeature cBytesAllocated < sizeof(HeaderDataSetShared) + sizeof(SharedStorageDataType)");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }

   HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(fillMem);
   if(k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id) {
      LOG_0(TraceLevelError, "ERROR FillFeature k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id");
      // don't set the header to bad since it's already set to something invalid and we don't know why
      return Error_IllegalParamValue;
   }

   const IntEbmType ret = AppendFeature(
      categorical,
      countBins,
      countSamples,
      binnedData,
      cBytesAllocated,
      static_cast<unsigned char *>(fillMem)
   );
   return static_cast<ErrorEbmType>(ret);
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION SizeWeight(
   IntEbmType countSamples,
   const FloatEbmType * weights
) {
   return AppendWeight(
      countSamples,
      weights,
      0,
      nullptr
   );
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillWeight(
   IntEbmType countSamples,
   const FloatEbmType * weights,
   IntEbmType countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(TraceLevelError, "ERROR FillWeight nullptr == fillMem");
      return Error_IllegalParamValue;
   }

   if(IsConvertError<size_t>(countBytesAllocated)) {
      LOG_0(TraceLevelError, "ERROR FillWeight countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   if(cBytesAllocated < sizeof(HeaderDataSetShared) + sizeof(SharedStorageDataType)) {
      LOG_0(TraceLevelError, "ERROR FillWeight cBytesAllocated < sizeof(HeaderDataSetShared) + sizeof(SharedStorageDataType)");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }

   HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(fillMem);
   if(k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id) {
      LOG_0(TraceLevelError, "ERROR FillWeight k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id");
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

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION SizeClassificationTarget(
   IntEbmType countTargetClasses,
   IntEbmType countSamples,
   const IntEbmType * targets
) {
   return AppendTarget(
      true,
      countTargetClasses,
      countSamples,
      targets,
      0,
      nullptr
   );
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillClassificationTarget(
   IntEbmType countTargetClasses,
   IntEbmType countSamples,
   const IntEbmType * targets,
   IntEbmType countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(TraceLevelError, "ERROR FillClassificationTarget nullptr == fillMem");
      return Error_IllegalParamValue;
   }

   if(IsConvertError<size_t>(countBytesAllocated)) {
      LOG_0(TraceLevelError, "ERROR FillClassificationTarget countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   if(cBytesAllocated < sizeof(HeaderDataSetShared) + sizeof(SharedStorageDataType)) {
      LOG_0(TraceLevelError, "ERROR FillClassificationTarget cBytesAllocated < sizeof(HeaderDataSetShared) + sizeof(SharedStorageDataType)");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }

   HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(fillMem);
   if(k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id) {
      LOG_0(TraceLevelError, "ERROR FillClassificationTarget k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id");
      // don't set the header to bad since it's already set to something invalid and we don't know why
      return Error_IllegalParamValue;
   }

   const IntEbmType ret = AppendTarget(
      true,
      countTargetClasses,
      countSamples,
      targets,
      cBytesAllocated,
      static_cast<unsigned char *>(fillMem)
   );
   return static_cast<ErrorEbmType>(ret);
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION SizeRegressionTarget(
   IntEbmType countSamples,
   const FloatEbmType * targets
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

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillRegressionTarget(
   IntEbmType countSamples,
   const FloatEbmType * targets,
   IntEbmType countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(TraceLevelError, "ERROR FillRegressionTarget nullptr == fillMem");
      return Error_IllegalParamValue;
   }

   if(IsConvertError<size_t>(countBytesAllocated)) {
      LOG_0(TraceLevelError, "ERROR FillRegressionTarget countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   if(cBytesAllocated < sizeof(HeaderDataSetShared) + sizeof(SharedStorageDataType)) {
      LOG_0(TraceLevelError, "ERROR FillRegressionTarget cBytesAllocated < sizeof(HeaderDataSetShared) + sizeof(SharedStorageDataType)");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }

   HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(fillMem);
   if(k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id) {
      LOG_0(TraceLevelError, "ERROR FillRegressionTarget k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id");
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

} // DEFINED_ZONE_NAME
