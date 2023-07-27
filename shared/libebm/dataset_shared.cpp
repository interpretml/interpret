// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "logging.h" // EBM_ASSERT
#include "common_c.h"

#include "common_cpp.hpp" // IsConvertError
#include "bridge_cpp.hpp" // GetCountItemsBitPacked

#include "ebm_internal.hpp"
#include "dataset_shared.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// TODO PK Implement the following to speed boosting and generating the gradients/hessians in interaction detection:
//   - We want sparse feature support in our booster since we don't need to access
//     memory if there are long segments with just a single value
//   - our boosting algorithm is position independent, so we can sort the data by the target feature, which
//     helps us because we can move the class number into a loop counter and not fetch the memory, and it allows
//     us to elimiante a branch when calculating statistics since all samples will have the same target within a loop
//   - we'll be sorting on the target, so we can't sort primarily on intput features (secondary sort ok)
//     So, sparse input features are not typically expected to clump into ranges of non - default parameters
//     So, we won't use ranges in our representation, so our sparse feature representation will be
//     class Sparse { size_t index; size_t val; }
//     This representation is invariant to position, so we'll be able to pre-compute the size before sorting
//   - having a secondary sort would allow a few features to have better memory locality since they'll
//     cluster their updates into a single memory location.  We should pick the features that have the
//     highest numbers of identical values so that we can get this benefit for the most number of features possible
//   - for regression, we might want to sort by increasing absolute values of the target since then we'll
//     have more precision in the earlier numbers which can have benefits in IEEE 754 where smaller numbers
//     have more precision in the early additions where the bin sums will be lower, or we could just sort
//     as best we can by input features if that gives us some speed benefit
//   - We will be sorting on the target values, BUT since the sort on the target will have no discontinuities
//     We can represent it purely as class Target { size_t count; } and each item in the array is an increment
//     of the class value(for classification).
//     Since we know how many classes there are, we will be able to know the size of the array AFTER sorting
//   - once we start sorting data, include a reverse index to work back to the original unsorted indexes
//   - For mains we should probably preserve the sparsity, but for pairs we probably want to de-sparsify
//     the data in the boosting and interaction datasets because we do not want to deal with dimensions
//     being either sparse or not and wanting to template that for low dimensions
// 
// STEPS :
//   - C will fill a temporary index array in the RawArray, sort the data by target with the indexes, and secondarily by input features.  The index array 
//     will remain for reconstructing the original order
//   - Now the memory is read only from now on, and shareable, and the original order can be re-constructed

// header ids
static constexpr UIntShared k_sharedDataSetWorkingId = 0x46DB; // random 15 bit number
static constexpr UIntShared k_sharedDataSetErrorId = 0x0103; // anything other than our normal id will work
static constexpr UIntShared k_sharedDataSetDoneId = 0x61E3; // random 15 bit number

// feature ids
static constexpr UIntShared k_missingFeatureBit = 0x1;
static constexpr UIntShared k_unknownFeatureBit = 0x2;
static constexpr UIntShared k_nominalFeatureBit = 0x4;
static constexpr UIntShared k_sparseFeatureBit = 0x8;
static constexpr UIntShared k_featureId = 0x2B40; // random 15 bit number with lower 4 bits set to zero

// weight ids
static constexpr UIntShared k_weightId = 0x31FB; // random 15 bit number

// target ids
static constexpr UIntShared k_classificationBit = 0x1;
static constexpr UIntShared k_targetId = 0x5A92; // random 15 bit number with lowest bit set to zero

INLINE_ALWAYS static bool IsFeature(const UIntShared id) noexcept {
   return (k_missingFeatureBit | k_unknownFeatureBit | k_nominalFeatureBit | k_sparseFeatureBit | k_featureId) ==
      ((k_missingFeatureBit | k_unknownFeatureBit | k_nominalFeatureBit | k_sparseFeatureBit) | id);
}
INLINE_ALWAYS static bool IsMissingFeature(const UIntShared id) noexcept {
   static_assert(0 == (k_missingFeatureBit & k_featureId), "k_featureId should not be missing");
   EBM_ASSERT(IsFeature(id));
   return 0 != (k_missingFeatureBit & id);
}
INLINE_ALWAYS static bool IsUnknownFeature(const UIntShared id) noexcept {
   static_assert(0 == (k_unknownFeatureBit & k_featureId), "k_featureId should not be unknown");
   EBM_ASSERT(IsFeature(id));
   return 0 != (k_unknownFeatureBit & id);
}
INLINE_ALWAYS static bool IsNominalFeature(const UIntShared id) noexcept {
   static_assert(0 == (k_nominalFeatureBit & k_featureId), "k_featureId should not be nominal");
   EBM_ASSERT(IsFeature(id));
   return 0 != (k_nominalFeatureBit & id);
}
INLINE_ALWAYS static bool IsSparseFeature(const UIntShared id) noexcept {
   static_assert(0 == (k_sparseFeatureBit & k_featureId), "k_featureId should not be sparse");
   EBM_ASSERT(IsFeature(id));
   return 0 != (k_sparseFeatureBit & id);
}
INLINE_ALWAYS static UIntShared GetFeatureId(
   const bool bMissing,
   const bool bUnknown,
   const bool bNominal,
   const bool bSparse
) noexcept {
   return k_featureId | 
      (bMissing ? k_missingFeatureBit : UIntShared { 0 }) |
      (bUnknown ? k_unknownFeatureBit : UIntShared { 0 }) |
      (bNominal ? k_nominalFeatureBit : UIntShared { 0 }) |
      (bSparse ? k_sparseFeatureBit : UIntShared { 0 });
}

INLINE_ALWAYS static bool IsTarget(const UIntShared id) noexcept {
   return (k_classificationBit | k_targetId) == (k_classificationBit | id);
}
INLINE_ALWAYS static bool IsClassificationTarget(const UIntShared id) noexcept {
   static_assert(0 == (k_classificationBit & k_targetId), "k_targetId should not be classification");
   EBM_ASSERT(IsTarget(id));
   return 0 != (k_classificationBit & id);
}
INLINE_ALWAYS static UIntShared GetTargetId(const bool bClassification) noexcept {
   return k_targetId | (bClassification ? k_classificationBit : UIntShared { 0 });
}


struct HeaderDataSetShared {
   // m_id should be in the first position since we use it to mark validity
   UIntShared m_id;

   UIntShared m_cSamples;
   UIntShared m_cFeatures;
   UIntShared m_cWeights;
   UIntShared m_cTargets;

   // IMPORTANT: m_offsets must be in the last position for the struct hack and this must be standard layout
   UIntShared m_offsets[1];
};
static_assert(std::is_standard_layout<HeaderDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<HeaderDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");

static const size_t k_cBytesHeaderId = offsetof(HeaderDataSetShared, m_id) + sizeof(HeaderDataSetShared::m_id);
static const size_t k_cBytesHeaderNoOffset = offsetof(HeaderDataSetShared, m_offsets);
static const UIntShared k_unfilledOffset = static_cast<UIntShared>(k_cBytesHeaderNoOffset - size_t { 1 });

struct FeatureDataSetShared {
   UIntShared m_id; // dense or sparse?  nominal, missing, unknown or not?
   UIntShared m_cBins;
};
static_assert(std::is_standard_layout<FeatureDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<FeatureDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");

struct SparseFeatureDataSetShared {
   // TODO: implement sparse features
   UIntShared m_defaultVal;
   UIntShared m_cNonDefaults;

   // IMPORTANT: m_nonDefaults must be in the last position for the struct hack and this must be standard layout
   SparseFeatureDataSetSharedEntry m_nonDefaults[1];
};
static_assert(std::is_standard_layout<SparseFeatureDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<SparseFeatureDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");

struct WeightDataSetShared {
   UIntShared m_id;
};
static_assert(std::is_standard_layout<WeightDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<WeightDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");

struct TargetDataSetShared {
   UIntShared m_id; // classification or regression
};
static_assert(std::is_standard_layout<TargetDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<TargetDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");

struct ClassificationTargetDataSetShared {
   UIntShared m_cClasses;
};
static_assert(std::is_standard_layout<ClassificationTargetDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<ClassificationTargetDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");

// No RegressionTargetDataSetShared required

static bool IsHeaderError(
   const UIntShared countSamples,
   const size_t cBytesAllocated,
   const unsigned char * const pFillMem
) {
   EBM_ASSERT(nullptr != pFillMem);

   // we only call IsHeaderError when adding a section, so we need at least the global header, 
   // one section offset, and at least one section header and/or the state index
   if(cBytesAllocated < k_cBytesHeaderNoOffset + sizeof(HeaderDataSetShared::m_offsets[0]) + sizeof(UIntShared)) {
      LOG_0(Trace_Error, "ERROR IsHeaderError not enough memory allocated for the shared dataset header");
      return true;
   }

   const HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<const HeaderDataSetShared *>(pFillMem);
   EBM_ASSERT(k_sharedDataSetWorkingId == pHeaderDataSetShared->m_id); // checked by our caller

   const UIntShared countFeatures = pHeaderDataSetShared->m_cFeatures;
   if(IsConvertError<size_t>(countFeatures)) {
      // we're being untrusting of the caller manipulating the memory improperly here
      LOG_0(Trace_Error, "ERROR IsHeaderError countFeatures is outside the range of a valid index");
      return true;
   }
   const size_t cFeatures = static_cast<size_t>(countFeatures);

   const UIntShared countWeights = pHeaderDataSetShared->m_cWeights;
   if(IsConvertError<size_t>(countWeights)) {
      // we're being untrusting of the caller manipulating the memory improperly here
      LOG_0(Trace_Error, "ERROR IsHeaderError countWeights is outside the range of a valid index");
      return true;
   }
   const size_t cWeights = static_cast<size_t>(countWeights);

   const UIntShared countTargets = pHeaderDataSetShared->m_cTargets;
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

   // we need a UIntShared at the end to indicate state
   if(cBytesAllocated - sizeof(UIntShared) < cBytesHeader) {
      LOG_0(Trace_Error, "ERROR IsHeaderError cBytesAllocated - sizeof(UIntShared) < cBytesHeader");
      return true;
   }

   const UIntShared indexByte0 = pHeaderDataSetShared->m_offsets[0];
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

   const UIntShared * const pInternalState =
      reinterpret_cast<const UIntShared *>(pFillMem + cBytesAllocated - sizeof(UIntShared));

   const UIntShared internalState = *pInternalState;
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
      if(UIntShared { 0 } != pHeaderDataSetShared->m_cSamples) {
         LOG_0(Trace_Error, "ERROR IsHeaderError UIntShared { 0 } != pHeaderDataSetShared->m_cSamples");
         return true;
      }
   } else {
      if(pHeaderDataSetShared->m_cSamples != countSamples) {
         LOG_0(Trace_Error, "ERROR IsHeaderError pHeaderDataSetShared->m_cSamples != countSamples");
         return true;
      }

      // if iOffset is 1, we'll just check this once again without any issues
      const UIntShared indexHighestOffsetPrev = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset - 1];
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

      const UIntShared indexHighestOffset = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset];
      if(IsConvertError<size_t>(indexHighestOffset)) {
         // we're being untrusting of the caller manipulating the memory improperly here
         LOG_0(Trace_Error, "ERROR IsHeaderError indexHighestOffset is outside the range of a valid index");
         return true;
      }
      const size_t iHighestOffset = static_cast<size_t>(indexHighestOffset);

      if(iHighestOffset <= iHighestOffsetPrev) {
         // we're being untrusting of the caller manipulating the memory improperly here
         LOG_0(Trace_Error, "ERROR IsHeaderError iHighestOffset <= iHighestOffsetPrev");
         return true;
      }

      // through associativity since iByte0 <= iHighestOffsetPrev && iHighestOffsetPrev < iHighestOffset
      EBM_ASSERT(iByte0 < iHighestOffset);
   }

   const size_t iOffsetNext = iOffset + 1; // we verified iOffset < cOffsets above, so this is guaranteed to work
   if(iOffsetNext != cOffsets) {
      const UIntShared indexHighestOffsetNext = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffsetNext];
      if(k_unfilledOffset != indexHighestOffsetNext) {
         LOG_0(Trace_Error, "ERROR IsHeaderError k_unfilledOffset != indexHighestOffsetNext");
         return true;
      }
   }

   return false;
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION CheckDataSet(IntEbm countBytesAllocated, const void * dataSet) {
   // if countBytesAllocated is 0 then we do not check the bytes allocated
   // if countBytesAllocated is positive then countBytesAllocated must exactly equal the dataSet size
   // if countBytesAllocated is negative then -countBytesAllocated must equal or exceed the dataSet size

   // dataSet needs to fit into memory, so for some fields we check if they can be converted to size_t since if they
   // could not then the dataSet would not fit into memory. Other fields where we do not require allocation like
   // the count of bins we do not check to see if they fit into size_t and leave that to the machine and architecture

   if(nullptr == dataSet) {
      LOG_0(Trace_Error, "ERROR CheckDataSet nullptr == dataSet");
      return Error_IllegalParamVal;
   }
   const unsigned char * const pDataSetShared = static_cast<const unsigned char *>(dataSet);

   if(IsAbsCastError<size_t>(countBytesAllocated)) {
      LOG_0(Trace_Error, "ERROR CheckDataSet IsAbsCastError<size_t>(countBytesAllocated)");
      return Error_IllegalParamVal;
   }
   size_t cBytesMax = AbsCast<size_t>(countBytesAllocated);
   if(size_t { 0 } == cBytesMax) {
      // 0 means do not check it
      cBytesMax = std::numeric_limits<size_t>::max();
   }

   if(cBytesMax < k_cBytesHeaderNoOffset) {
      LOG_0(Trace_Error, "ERROR CheckDataSet Not enough space to access HeaderDataSetShared");
      return Error_IllegalParamVal;
   }

   const HeaderDataSetShared * const pHeaderDataSetShared =
      reinterpret_cast<const HeaderDataSetShared *>(pDataSetShared);

   if(k_sharedDataSetDoneId != pHeaderDataSetShared->m_id) {
      LOG_0(Trace_Error, "ERROR CheckDataSet k_sharedDataSetDoneId != pHeaderDataSetShared->m_id");
      return Error_IllegalParamVal;
   }

   // often we'll need to be able to convert countSamples to a size_t, but not always, so do it later
   const UIntShared countSamples = pHeaderDataSetShared->m_cSamples;

   const UIntShared countFeatures = pHeaderDataSetShared->m_cFeatures;
   if(IsConvertError<size_t>(countFeatures)) {
      LOG_0(Trace_Error, "ERROR CheckDataSet IsConvertError<size_t>(countFeatures)");
      return Error_IllegalParamVal;
   }
   size_t cFeatures = static_cast<size_t>(countFeatures);

   const UIntShared countWeights = pHeaderDataSetShared->m_cWeights;
   if(IsConvertError<size_t>(countWeights)) {
      LOG_0(Trace_Error, "ERROR CheckDataSet IsConvertError<size_t>(countWeights)");
      return Error_IllegalParamVal;
   }
   size_t cWeights = static_cast<size_t>(countWeights);

   const UIntShared countTargets = pHeaderDataSetShared->m_cTargets;
   if(IsConvertError<size_t>(countTargets)) {
      LOG_0(Trace_Error, "ERROR CheckDataSet IsConvertError<size_t>(countTargets)");
      return Error_IllegalParamVal;
   }
   size_t cTargets = static_cast<size_t>(countTargets);

   if(IsAddError(cFeatures, cWeights, cTargets)) {
      LOG_0(Trace_Error, "ERROR CheckDataSet IsAddError(cFeatures, cWeights, cTargets)");
      return Error_IllegalParamVal;
   }
   const size_t cOffsets = cFeatures + cWeights + cTargets;

   if(IsMultiplyError(sizeof(pHeaderDataSetShared->m_offsets[0]), cOffsets)) {
      LOG_0(Trace_Error, "ERROR CheckDataSet IsMultiplyError(sizeof(pHeaderDataSetShared->m_offsets[0]), cOffsets)");
      return Error_IllegalParamVal;
   }
   size_t iOffsetNext = sizeof(pHeaderDataSetShared->m_offsets[0]) * cOffsets;
   if(IsAddError(k_cBytesHeaderNoOffset, iOffsetNext)) {
      LOG_0(Trace_Error, "ERROR CheckDataSet IsAddError(k_cBytesHeaderNoOffset, iOffsetNext)");
      return Error_IllegalParamVal;
   }
   iOffsetNext += k_cBytesHeaderNoOffset;

   if(cBytesMax < iOffsetNext) {
      LOG_0(Trace_Error, "ERROR CheckDataSet Not enough space to access HeaderDataSetShared::m_offsets");
      return Error_IllegalParamVal;
   }

   const UIntShared * pOffset = ArrayToPointer(pHeaderDataSetShared->m_offsets);

   if(size_t { 0 } != cFeatures) {
      const UIntShared * const pOffsetEnd = pOffset + cFeatures;
      do {
         const UIntShared indexOffsetCur = *pOffset;
         if(IsConvertError<size_t>(indexOffsetCur)) {
            LOG_0(Trace_Error, "ERROR CheckDataSet IsConvertError<size_t>(indexOffsetCur)");
            return Error_IllegalParamVal;
         }
         size_t iOffsetCur = static_cast<size_t>(indexOffsetCur);

         if(iOffsetNext != iOffsetCur) {
            LOG_0(Trace_Error, "ERROR CheckDataSet iOffsetNext != offsetCur");
            return Error_IllegalParamVal;
         }

         if(IsAddError(iOffsetNext, sizeof(FeatureDataSetShared))) {
            LOG_0(Trace_Error, "ERROR CheckDataSet IsAddError(iOffsetNext, sizeof(FeatureDataSetShared))");
            return Error_IllegalParamVal;
         }
         iOffsetNext += sizeof(FeatureDataSetShared);

         if(cBytesMax < iOffsetNext) {
            LOG_0(Trace_Error, "ERROR CheckDataSet Not enough space to access FeatureDataSetShared");
            return Error_IllegalParamVal;
         }

         const FeatureDataSetShared * pFeatureDataSetShared =
            reinterpret_cast<const FeatureDataSetShared *>(pDataSetShared + iOffsetCur);

         const UIntShared id = pFeatureDataSetShared->m_id;
         if(!IsFeature(id)) {
            LOG_0(Trace_Error, "ERROR CheckDataSet !IsFeature(id)");
            return Error_IllegalParamVal;
         }

         // we do not need to allocate anything based on the count of bins, so allow numbers of bins
         // that would exceed a size_t here and catch those on whatever system this dataset is unpacked on
         const UIntShared countBins = pFeatureDataSetShared->m_cBins;

         if(IsSparseFeature(id)) {
            const size_t cBytesSparseHeaderNoOffset = offsetof(SparseFeatureDataSetShared, m_nonDefaults);

            iOffsetCur = iOffsetNext;
            if(IsAddError(iOffsetNext, cBytesSparseHeaderNoOffset)) {
               LOG_0(Trace_Error, "ERROR CheckDataSet IsAddError(iOffsetNext, cBytesSparseHeaderNoOffset)");
               return Error_IllegalParamVal;
            }
            iOffsetNext += cBytesSparseHeaderNoOffset;

            if(cBytesMax < iOffsetNext) {
               LOG_0(Trace_Error, "ERROR CheckDataSet Not enough space to access SparseFeatureDataSetShared");
               return Error_IllegalParamVal;
            }

            const SparseFeatureDataSetShared * const pSparseFeatureDataSetShared =
               reinterpret_cast<const SparseFeatureDataSetShared *>(pDataSetShared + iOffsetCur);

            const UIntShared defaultVal = pSparseFeatureDataSetShared->m_defaultVal;
            if(countBins <= defaultVal) {
               LOG_0(Trace_Error, "ERROR CheckDataSet countBins <= defaultVal");
               return Error_IllegalParamVal;
            }

            const UIntShared countNonDefaults = pSparseFeatureDataSetShared->m_cNonDefaults;
            if(IsConvertError<size_t>(countNonDefaults)) {
               LOG_0(Trace_Error, "ERROR CheckDataSet IsConvertError<size_t>(countNonDefaults)");
               return Error_IllegalParamVal;
            }
            const size_t cNonDefaults = static_cast<size_t>(countNonDefaults);

            if(IsMultiplyError(sizeof(pSparseFeatureDataSetShared->m_nonDefaults[0]), cNonDefaults)) {
               LOG_0(Trace_Error, "ERROR CheckDataSet IsMultiplyError(sizeof(pSparseFeatureDataSetShared->m_nonDefaults[0]), cNonDefaults)");
               return Error_IllegalParamVal;
            }
            const size_t cTotalNonDefaults = sizeof(pSparseFeatureDataSetShared->m_nonDefaults[0]) * cNonDefaults;
            if(IsAddError(iOffsetNext, cTotalNonDefaults)) {
               LOG_0(Trace_Error, "ERROR CheckDataSet IsAddError(iOffsetNext, cTotalNonDefaults)");
               return Error_IllegalParamVal;
            }
            iOffsetNext += cTotalNonDefaults;

            if(cBytesMax < iOffsetNext) {
               LOG_0(Trace_Error, "ERROR CheckDataSet Not enough space to access SparseFeatureDataSetShared::m_nonDefaults");
               return Error_IllegalParamVal;
            }

            const SparseFeatureDataSetSharedEntry * pNonDefault = ArrayToPointer(pSparseFeatureDataSetShared->m_nonDefaults);
            const SparseFeatureDataSetSharedEntry * const pNonDefaultEnd = &pNonDefault[cNonDefaults];
            while(pNonDefaultEnd != pNonDefault) {
               if(countSamples <= pNonDefault->m_iSample) {
                  LOG_0(Trace_Error, "ERROR CheckDataSet countSamples <= pNonDefault->m_iSample");
                  return Error_IllegalParamVal;
               }

               if(countBins <= pNonDefault->m_nonDefaultVal) {
                  LOG_0(Trace_Error, "ERROR CheckDataSet countBins <= pNonDefault->m_nonDefaultVal");
                  return Error_IllegalParamVal;
               }
               ++pNonDefault;
            }
         } else {
            // if there is only 1 bin we always know what it will be and we do not need to store anything
            if(UIntShared { 0 } != countSamples && UIntShared { 1 } < countBins) {
               const int cBitsRequiredMin = CountBitsRequired(countBins - UIntShared { 1 });
               EBM_ASSERT(1 <= cBitsRequiredMin);
               EBM_ASSERT(cBitsRequiredMin <= COUNT_BITS(UIntShared));

               const int cItemsPerBitPack = GetCountItemsBitPacked<UIntShared>(cBitsRequiredMin);
               EBM_ASSERT(1 <= cItemsPerBitPack);
               EBM_ASSERT(cItemsPerBitPack <= COUNT_BITS(UIntShared));

               EBM_ASSERT(UIntShared { 1 } <= countSamples);
               const UIntShared countDataUnits = (countSamples - UIntShared { 1 }) /
                  static_cast<UIntShared>(cItemsPerBitPack) + UIntShared { 1 };

               if(IsConvertError<size_t>(countDataUnits)) {
                  LOG_0(Trace_Error, "ERROR CheckDataSet IsConvertError<size_t>(countDataUnits)");
                  return Error_IllegalParamVal;
               }
               const size_t cDataUnits = static_cast<size_t>(countDataUnits);

               if(IsMultiplyError(sizeof(UIntShared), cDataUnits)) {
                  LOG_0(Trace_Error, "ERROR CheckDataSet IsMultiplyError(sizeof(UIntShared), cDataUnits)");
                  return Error_IllegalParamVal;
               }
               const size_t cTotalMem = sizeof(UIntShared) * cDataUnits;

               iOffsetCur = iOffsetNext;
               if(IsAddError(iOffsetNext, cTotalMem)) {
                  LOG_0(Trace_Error, "ERROR CheckDataSet IsAddError(iOffsetNext, cTotalMem)");
                  return Error_IllegalParamVal;
               }
               iOffsetNext += cTotalMem;

               if(cBytesMax < iOffsetNext) {
                  LOG_0(Trace_Error, "ERROR CheckDataSet Not enough space to access the bit packed feature values");
                  return Error_IllegalParamVal;
               }

               const int cBitsPerItemMax = GetCountBits<UIntShared>(cItemsPerBitPack);
               EBM_ASSERT(1 <= cBitsPerItemMax);
               EBM_ASSERT(cBitsPerItemMax <= COUNT_BITS(UIntShared));

               int cShift = static_cast<int>((countSamples - UIntShared { 1 }) % static_cast<UIntShared>(cItemsPerBitPack)) * cBitsPerItemMax;
               const int cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;

               const UIntShared maskBits = MakeLowMask<UIntShared>(cBitsPerItemMax);

               const UIntShared * pInputData =
                  reinterpret_cast<const UIntShared *>(pDataSetShared + iOffsetCur);
               const UIntShared * const pInputDataEnd =
                  reinterpret_cast<const UIntShared *>(pDataSetShared + iOffsetNext);

               do {
                  const UIntShared iBinCombined = *pInputData;
                  ++pInputData;
                  do {
                     const UIntShared indexBin = (iBinCombined >> cShift) & maskBits;

                     if(countBins <= indexBin) {
                        LOG_0(Trace_Error, "ERROR CheckDataSet countBins <= indexBin");
                        return Error_IllegalParamVal;
                     }

                     cShift -= cBitsPerItemMax;
                  } while(0 <= cShift);
                  cShift = cShiftReset;
               } while(pInputDataEnd != pInputData);
            }
         }
         ++pOffset;
      } while(pOffsetEnd != pOffset);
   }

   if(size_t { 0 } != cWeights) {
      if(IsConvertError<size_t>(countSamples)) {
         LOG_0(Trace_Error, "ERROR CheckDataSet IsConvertError<size_t>(countSamples)");
         return Error_IllegalParamVal;
      }
      const size_t cSamples = static_cast<size_t>(countSamples);

      const UIntShared * const pOffsetEnd = pOffset + cWeights;
      do {
         const UIntShared indexOffsetCur = *pOffset;
         if(IsConvertError<size_t>(indexOffsetCur)) {
            LOG_0(Trace_Error, "ERROR CheckDataSet IsConvertError<size_t>(indexOffsetCur)");
            return Error_IllegalParamVal;
         }
         size_t iOffsetCur = static_cast<size_t>(indexOffsetCur);

         if(iOffsetNext != iOffsetCur) {
            LOG_0(Trace_Error, "ERROR CheckDataSet iOffsetNext != offsetCur");
            return Error_IllegalParamVal;
         }

         if(IsAddError(iOffsetNext, sizeof(WeightDataSetShared))) {
            LOG_0(Trace_Error, "ERROR CheckDataSet IsAddError(iOffsetNext, sizeof(WeightDataSetShared))");
            return Error_IllegalParamVal;
         }
         iOffsetNext += sizeof(WeightDataSetShared);

         if(cBytesMax < iOffsetNext) {
            LOG_0(Trace_Error, "ERROR CheckDataSet Not enough space to access WeightDataSetShared");
            return Error_IllegalParamVal;
         }

         const WeightDataSetShared * pWeightDataSetShared =
            reinterpret_cast<const WeightDataSetShared *>(pDataSetShared + iOffsetCur);

         const UIntShared id = pWeightDataSetShared->m_id;
         if(k_weightId != id) {
            LOG_0(Trace_Error, "ERROR CheckDataSet k_weightId != id");
            return Error_IllegalParamVal;
         }

         if(IsMultiplyError(sizeof(FloatShared), cSamples)) {
            LOG_0(Trace_Error, "ERROR CheckDataSet IsMultiplyError(sizeof(FloatShared), cSamples)");
            return Error_IllegalParamVal;
         }
         const size_t cTotalMem = sizeof(FloatShared) * cSamples;

         // iOffsetCur = iOffsetNext;
         if(IsAddError(iOffsetNext, cTotalMem)) {
            LOG_0(Trace_Error, "ERROR CheckDataSet IsAddError(iOffsetNext, cTotalMem)");
            return Error_IllegalParamVal;
         }
         iOffsetNext += cTotalMem;

         if(cBytesMax < iOffsetNext) {
            LOG_0(Trace_Error, "ERROR CheckDataSet Not enough space to access the weights");
            return Error_IllegalParamVal;
         }

         // TODO: should I be checking for these bad weight values here or somewhere else?
         //const FloatShared * pInputData =
         //   reinterpret_cast<const FloatShared *>(pDataSetShared + iOffsetCur);
         //const FloatShared * const pInputDataEnd =
         //   reinterpret_cast<const FloatShared *>(pDataSetShared + iOffsetNext);
         //while(pInputDataEnd != pInputData) {
         //   const FloatShared weight = *pInputData;
         //   if(std::isnan(weight)) {
         //      LOG_0(Trace_Error, "ERROR CheckDataSet std::isnan(weight)");
         //      return Error_IllegalParamVal;
         //   }
         //   if(std::isinf(weight)) {
         //      LOG_0(Trace_Error, "ERROR CheckDataSet std::isinf(weight)");
         //      return Error_IllegalParamVal;
         //   }
         //   if(weight < FloatShared { 0 }) {
         //      LOG_0(Trace_Error, "ERROR CheckDataSet weight < FloatShared { 0 }");
         //      return Error_IllegalParamVal;
         //   }
         //   ++pInputData;
         //}

         ++pOffset;
      } while(pOffsetEnd != pOffset);
   }

   if(size_t { 0 } != cTargets) {
      const UIntShared * const pOffsetEnd = pOffset + cTargets;
      do {
         const UIntShared indexOffsetCur = *pOffset;
         if(IsConvertError<size_t>(indexOffsetCur)) {
            LOG_0(Trace_Error, "ERROR CheckDataSet IsConvertError<size_t>(indexOffsetCur)");
            return Error_IllegalParamVal;
         }
         size_t iOffsetCur = static_cast<size_t>(indexOffsetCur);

         if(iOffsetNext != iOffsetCur) {
            LOG_0(Trace_Error, "ERROR CheckDataSet iOffsetNext != offsetCur");
            return Error_IllegalParamVal;
         }

         if(IsAddError(iOffsetNext, sizeof(TargetDataSetShared))) {
            LOG_0(Trace_Error, "ERROR CheckDataSet IsAddError(iOffsetNext, sizeof(TargetDataSetShared))");
            return Error_IllegalParamVal;
         }
         iOffsetNext += sizeof(TargetDataSetShared);

         if(cBytesMax < iOffsetNext) {
            LOG_0(Trace_Error, "ERROR CheckDataSet Not enough space to access TargetDataSetShared");
            return Error_IllegalParamVal;
         }

         const TargetDataSetShared * pTargetDataSetShared =
            reinterpret_cast<const TargetDataSetShared *>(pDataSetShared + iOffsetCur);

         const UIntShared id = pTargetDataSetShared->m_id;
         if(!IsTarget(id)) {
            LOG_0(Trace_Error, "ERROR CheckDataSet !IsTarget(id)");
            return Error_IllegalParamVal;
         }

         if(IsClassificationTarget(id)) {
            if(IsConvertError<size_t>(countSamples)) {
               LOG_0(Trace_Error, "ERROR CheckDataSet IsConvertError<size_t>(countSamples)");
               return Error_IllegalParamVal;
            }
            const size_t cSamples = static_cast<size_t>(countSamples);

            iOffsetCur = iOffsetNext;
            if(IsAddError(iOffsetNext, sizeof(ClassificationTargetDataSetShared))) {
               LOG_0(Trace_Error, "ERROR CheckDataSet IsAddError(iOffsetNext, sizeof(ClassificationTargetDataSetShared))");
               return Error_IllegalParamVal;
            }
            iOffsetNext += sizeof(ClassificationTargetDataSetShared);

            if(cBytesMax < iOffsetNext) {
               LOG_0(Trace_Error, "ERROR CheckDataSet Not enough space to access ClassificationTargetDataSetShared");
               return Error_IllegalParamVal;
            }

            const ClassificationTargetDataSetShared * const pClassificationTargetDataSetShared =
               reinterpret_cast<const ClassificationTargetDataSetShared *>(pDataSetShared + iOffsetCur);

            // we do not need to allocate anything based on the number of classes, so allow numbers of classes
            // that would exceed a size_t here and catch those on whatever system this dataset is unpacked on
            const UIntShared countClasses = pClassificationTargetDataSetShared->m_cClasses;

            if(IsMultiplyError(sizeof(UIntShared), cSamples)) {
               LOG_0(Trace_Error, "ERROR CheckDataSet IsMultiplyError(sizeof(UIntShared), cSamples)");
               return Error_IllegalParamVal;
            }
            const size_t cTotalMem = sizeof(UIntShared) * cSamples;

            iOffsetCur = iOffsetNext;
            if(IsAddError(iOffsetNext, cTotalMem)) {
               LOG_0(Trace_Error, "ERROR CheckDataSet IsAddError(iOffsetNext, cTotalMem)");
               return Error_IllegalParamVal;
            }
            iOffsetNext += cTotalMem;

            if(cBytesMax < iOffsetNext) {
               LOG_0(Trace_Error, "ERROR CheckDataSet Not enough space to access the classification targets");
               return Error_IllegalParamVal;
            }

            const UIntShared * pInputData =
               reinterpret_cast<const UIntShared *>(pDataSetShared + iOffsetCur);
            const UIntShared * const pInputDataEnd =
               reinterpret_cast<const UIntShared *>(pDataSetShared + iOffsetNext);
            while(pInputDataEnd != pInputData) {
               const UIntShared target = *pInputData;
               if(countClasses <= target) {
                  LOG_0(Trace_Error, "ERROR CheckDataSet countClasses <= target");
                  return Error_IllegalParamVal;
               }
               ++pInputData;
            }
         } else {
            if(IsConvertError<size_t>(countSamples)) {
               LOG_0(Trace_Error, "ERROR CheckDataSet IsConvertError<size_t>(countSamples)");
               return Error_IllegalParamVal;
            }
            const size_t cSamples = static_cast<size_t>(countSamples);

            if(IsMultiplyError(sizeof(FloatShared), cSamples)) {
               LOG_0(Trace_Error, "ERROR CheckDataSet IsMultiplyError(sizeof(FloatShared), cSamples)");
               return Error_IllegalParamVal;
            }
            const size_t cTotalMem = sizeof(FloatShared) * cSamples;

            //iOffsetCur = iOffsetNext;
            if(IsAddError(iOffsetNext, cTotalMem)) {
               LOG_0(Trace_Error, "ERROR CheckDataSet IsAddError(iOffsetNext, cTotalMem)");
               return Error_IllegalParamVal;
            }
            iOffsetNext += cTotalMem;

            if(cBytesMax < iOffsetNext) {
               LOG_0(Trace_Error, "ERROR CheckDataSet Not enough space to access the regression targets");
               return Error_IllegalParamVal;
            }

            // TODO: should I be checking for these bad regression targets here or somewhere else?
            //const FloatShared * pInputData =
            //   reinterpret_cast<const FloatShared *>(pDataSetShared + iOffsetCur);
            //const FloatShared * const pInputDataEnd =
            //   reinterpret_cast<const FloatShared *>(pDataSetShared + iOffsetNext);
            //while(pInputDataEnd != pInputData) {
            //   const FloatShared target = *pInputData;
            //   if(std::isnan(target)) {
            //      LOG_0(Trace_Error, "ERROR CheckDataSet std::isnan(target)");
            //      return Error_IllegalParamVal;
            //   }
            //   if(std::isinf(target)) {
            //      LOG_0(Trace_Error, "ERROR CheckDataSet std::isinf(target)");
            //      return Error_IllegalParamVal;
            //   }
            //   ++pInputData;
            //}
         }
         ++pOffset;
      } while(pOffsetEnd != pOffset);
   }

   if(IntEbm { 0 } < countBytesAllocated) {
      if(iOffsetNext != cBytesMax) {
         LOG_0(Trace_Error, "ERROR CheckDataSet dataSet length does not match");
         return Error_IllegalParamVal;
      }
   }

   return Error_None;
}

static ErrorEbm LockDataSetShared(const size_t cBytesAllocated, unsigned char * const pFillMem) {
   HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
   EBM_ASSERT(k_sharedDataSetWorkingId == pHeaderDataSetShared->m_id);

   // TODO: sort the data by the target (if there is only one target)


   // breifly set this to done so that we can check it with our public CheckDataSet function
   pHeaderDataSetShared->m_id = k_sharedDataSetDoneId;

   EBM_ASSERT(!IsConvertError<IntEbm>(cBytesAllocated)); // it came from IntEbm
   const ErrorEbm error = CheckDataSet(static_cast<IntEbm>(cBytesAllocated), pFillMem);
   if(Error_None != error) {
      pHeaderDataSetShared->m_id = k_sharedDataSetErrorId;
   }
   return error;
}

WARNING_PUSH
WARNING_REDUNDANT_CODE
static IntEbm AppendHeader(
   const IntEbm countFeatures,
   const IntEbm countWeights,
   const IntEbm countTargets,
   const size_t cBytesAllocated,
   unsigned char * const pFillMem
) {
   EBM_ASSERT(size_t { 0 } == cBytesAllocated && nullptr == pFillMem || nullptr != pFillMem);

   LOG_N(
      Trace_Info,
      "Entered AppendHeader: "
      "countFeatures=%" IntEbmPrintf ", "
      "countWeights=%" IntEbmPrintf ", "
      "countTargets=%" IntEbmPrintf ", "
      "cBytesAllocated=%zu, "
      "pFillMem=%p"
      ,
      countFeatures,
      countWeights,
      countTargets,
      cBytesAllocated,
      static_cast<void *>(pFillMem)
   );

   if(IsConvertError<size_t>(countFeatures) || IsConvertError<UIntShared>(countFeatures)) {
      LOG_0(Trace_Error, "ERROR AppendHeader countFeatures is outside the range of a valid index");
      return Error_IllegalParamVal;
   }
   const size_t cFeatures = static_cast<size_t>(countFeatures);

   if(IsConvertError<size_t>(countWeights) || IsConvertError<UIntShared>(countWeights)) {
      LOG_0(Trace_Error, "ERROR AppendHeader countWeights is outside the range of a valid index");
      return Error_IllegalParamVal;
   }
   const size_t cWeights = static_cast<size_t>(countWeights);

   if(IsConvertError<size_t>(countTargets) || IsConvertError<UIntShared>(countTargets)) {
      LOG_0(Trace_Error, "ERROR AppendHeader countTargets is outside the range of a valid index");
      return Error_IllegalParamVal;
   }
   const size_t cTargets = static_cast<size_t>(countTargets);

   if(IsAddError(cFeatures, cWeights, cTargets)) {
      LOG_0(Trace_Error, "ERROR AppendHeader IsAddError(cFeatures, cWeights, cTargets)");
      return Error_IllegalParamVal;
   }
   const size_t cOffsets = cFeatures + cWeights + cTargets;

   if(IsMultiplyError(sizeof(HeaderDataSetShared::m_offsets[0]), cOffsets)) {
      LOG_0(Trace_Error, "ERROR AppendHeader IsMultiplyError(sizeof(HeaderDataSetShared::m_offsets[0]), cOffsets)");
      return Error_IllegalParamVal;
   }
   const size_t cBytesOffsets = sizeof(HeaderDataSetShared::m_offsets[0]) * cOffsets;

   // for AppendHeader also check if we have room to add the termination/state index at the end
   // if there are zero offsets it will always work and if there are non-zero offsets then we need it
   if(IsAddError(k_cBytesHeaderNoOffset, cBytesOffsets, sizeof(UIntShared))) {
      LOG_0(Trace_Error, "ERROR AppendHeader IsAddError(k_cBytesHeaderNoOffset, cBytesOffsets, sizeof(UIntShared))");
      return Error_IllegalParamVal;
   }
   const size_t cBytesHeader = k_cBytesHeaderNoOffset + cBytesOffsets;

   if(IsConvertError<UIntShared>(cBytesHeader)) {
      LOG_0(Trace_Error, "ERROR AppendHeader cBytesHeader is outside the range of a valid size");
      return Error_IllegalParamVal;
   }

   if(nullptr != pFillMem) {
      if(size_t { 0 } == cOffsets) {
         if(cBytesAllocated != cBytesHeader) {
            LOG_0(Trace_Error, "ERROR AppendHeader buffer size and fill size do not agree");
            return Error_IllegalParamVal;
         }
      } else {
         // we checked above that we could add sizeof(UIntShared) to cBytesHeader
         if(cBytesAllocated < cBytesHeader + sizeof(UIntShared)) {
            LOG_0(Trace_Error, "ERROR AppendHeader cBytesAllocated < cBytesHeader + sizeof(UIntShared)");
            // don't set the header to bad if we don't have enough memory for the header itself
            return Error_IllegalParamVal;
         }
      }

      HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);

      pHeaderDataSetShared->m_id = k_sharedDataSetWorkingId;
      pHeaderDataSetShared->m_cSamples = 0;
      pHeaderDataSetShared->m_cFeatures = static_cast<UIntShared>(cFeatures);
      pHeaderDataSetShared->m_cWeights = static_cast<UIntShared>(cWeights);
      pHeaderDataSetShared->m_cTargets = static_cast<UIntShared>(cTargets);

      if(size_t { 0 } == cOffsets) {
         // we allow this shared data set to be permissive in it's construction but if there are things like
         // zero targets we expect the booster or interaction detector constructors to give errors
         const ErrorEbm error = LockDataSetShared(cBytesAllocated, pFillMem);
         if(Error_None != error) {
            return error;
         }
      } else {
         UIntShared * pCur = ArrayToPointer(pHeaderDataSetShared->m_offsets);
         const UIntShared * const pEnd = pCur + cOffsets;
         do {
            *pCur = k_unfilledOffset;
            ++pCur;
         } while(pEnd != pCur);

         // position our first item right after the header
         pHeaderDataSetShared->m_offsets[0] = static_cast<UIntShared>(cBytesHeader);
         UIntShared * const pInternalState =
            reinterpret_cast<UIntShared *>(pFillMem + cBytesAllocated - sizeof(UIntShared));
         *pInternalState = 0;
      }
      return Error_None;
   }
   if(IsConvertError<IntEbm>(cBytesHeader)) {
      LOG_0(Trace_Error, "ERROR AppendHeader IsConvertError<IntEbm>(cBytesHeader)");
      return Error_OutOfMemory;
   }
   return cBytesHeader;
}
WARNING_POP

static bool DecideIfSparse(const size_t cSamples, const IntEbm * binIndexes) {
   // For sparsity in the data set shared memory the only thing that matters is compactness since we don't use
   // this memory in any high performance loops

   UNUSED(cSamples);
   UNUSED(binIndexes);

   // TODO: evalute the data to decide if the feature should be sparse or not
   return false;
}

WARNING_PUSH
WARNING_REDUNDANT_CODE
static IntEbm AppendFeature(
   const IntEbm countBins,
   const BoolEbm isMissing,
   const BoolEbm isUnknown,
   const BoolEbm isNominal,
   const IntEbm countSamples,
   const IntEbm * binIndexes,
   const size_t cBytesAllocated,
   unsigned char * const pFillMem
) {
   EBM_ASSERT(size_t { 0 } == cBytesAllocated && nullptr == pFillMem || 
      nullptr != pFillMem && k_cBytesHeaderId <= cBytesAllocated);

   LOG_N(
      Trace_Info,
      "Entered AppendFeature: "
      "countBins=%" IntEbmPrintf ", "
      "isMissing=%s, "
      "isUnknown=%s, "
      "isNominal=%s, "
      "countSamples=%" IntEbmPrintf ", "
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
      if(countBins <= IntEbm { 1 }) {
         LOG_0(Trace_Error, "ERROR AppendFeature countBins must be 2 or larger");
         goto return_bad;
      }
      // we do not need to access memory based on the value of countBins, so we do not need to check if fits into size_t
      if(IsConvertError<UIntShared>(countBins)) {
         LOG_0(Trace_Error, "ERROR AppendFeature countBins is outside the range of a valid index");
         goto return_bad;
      }
      UIntShared cBins = static_cast<UIntShared>(countBins);
      cBins -= EBM_FALSE != isMissing ? UIntShared { 0 } : UIntShared { 1 };
      cBins -= EBM_FALSE != isUnknown ? UIntShared { 0 } : UIntShared { 1 };

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
      if(IsConvertError<size_t>(countSamples) || IsConvertError<UIntShared>(countSamples)) {
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
         if(IsHeaderError(static_cast<UIntShared>(cSamples), cBytesAllocated, pFillMem)) {
            goto return_bad;
         }

         UIntShared * const pInternalState =
            reinterpret_cast<UIntShared *>(pFillMem + cBytesAllocated - sizeof(UIntShared));
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

         EBM_ASSERT(size_t { 0 } == iOffset && UIntShared { 0 } == pHeaderDataSetShared->m_cSamples ||
            static_cast<UIntShared>(cSamples) == pHeaderDataSetShared->m_cSamples);
         pHeaderDataSetShared->m_cSamples = static_cast<UIntShared>(cSamples);

         FeatureDataSetShared * const pFeatureDataSetShared = 
            reinterpret_cast<FeatureDataSetShared *>(pFillMem + iHighestOffset);

         pFeatureDataSetShared->m_id = GetFeatureId(
            EBM_FALSE != isMissing,
            EBM_FALSE != isUnknown,
            EBM_FALSE != isNominal,
            bSparse
         );
         pFeatureDataSetShared->m_cBins = cBins;
      }

      // if there is only 1 bin we always know what it will be and we do not need to store anything
      if(size_t { 0 } != cSamples) {
         const IntEbm * pBinIndex = binIndexes;
         const IntEbm * const pBinIndexsEnd = binIndexes + cSamples;
         if(cBins <= UIntShared { 1 }) {
            if(UIntShared { 0 } == cBins) {
               LOG_0(Trace_Error, "ERROR AppendFeature UIntShared { 0 } == cBins");
               goto return_bad;
            }
            const IntEbm indexBinLegal = EBM_FALSE != isMissing ? IntEbm { 0 } : IntEbm { 1 };
            do {
               const IntEbm indexBin = *pBinIndex;
               if(indexBinLegal != indexBin) {
                  LOG_0(Trace_Error, "ERROR AppendFeature indexBinLegal != indexBin");
                  goto return_bad;
               }
               ++pBinIndex;
            } while(pBinIndexsEnd != pBinIndex);
         } else {
            const int cBitsRequiredMin = CountBitsRequired(cBins - UIntShared { 1 });
            EBM_ASSERT(1 <= cBitsRequiredMin);
            EBM_ASSERT(cBitsRequiredMin <= COUNT_BITS(UIntShared));

            const int cItemsPerBitPack = GetCountItemsBitPacked<UIntShared>(cBitsRequiredMin);
            EBM_ASSERT(1 <= cItemsPerBitPack);
            EBM_ASSERT(cItemsPerBitPack <= COUNT_BITS(UIntShared));

            const int cBitsPerItemMax = GetCountBits<UIntShared>(cItemsPerBitPack);
            EBM_ASSERT(1 <= cBitsPerItemMax);
            EBM_ASSERT(cBitsPerItemMax <= COUNT_BITS(UIntShared));

            EBM_ASSERT(1 <= cSamples);
            const size_t cDataUnits = (cSamples - size_t { 1 }) / static_cast<size_t>(cItemsPerBitPack) + size_t { 1 };

            if(IsMultiplyError(sizeof(UIntShared), cDataUnits)) {
               LOG_0(Trace_Error, "ERROR AppendFeature IsMultiplyError(sizeof(UIntShared), cDataUnits)");
               goto return_bad;
            }
            const size_t cBytesAllSamples = sizeof(UIntShared) * cDataUnits;

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
               UIntShared * pFillData = reinterpret_cast<UIntShared *>(pFillMem + iByteCur);

               int cShift = static_cast<int>((cSamples - size_t { 1 }) % static_cast<size_t>(cItemsPerBitPack)) * cBitsPerItemMax;
               const int cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;
               const IntEbm indexBinIllegal = countBins - (EBM_FALSE != isUnknown ? IntEbm { 0 } : IntEbm { 1 });
               do {
                  UIntShared bits = 0;
                  do {
                     IntEbm indexBin = *pBinIndex;
                     if(indexBinIllegal <= indexBin) {
                        LOG_0(Trace_Error, "ERROR AppendFeature indexBinIllegal <= indexBin");
                        goto return_bad;
                     }
                     if(EBM_FALSE != isMissing) {
                        if(indexBin < IntEbm { 0 }) {
                           LOG_0(Trace_Error, "ERROR AppendFeature indexBin can't be negative");
                           goto return_bad;
                        }
                     } else {
                        if(indexBin <= IntEbm { 0 }) {
                           LOG_0(Trace_Error, "ERROR AppendFeature indexBin <= IntEbm { 0 }");
                           goto return_bad;
                        }
                        --indexBin;
                     }
                     ++pBinIndex;

                     // since countBins can be converted to these, so now can indexBin
                     EBM_ASSERT(!IsConvertError<UIntShared>(indexBin));

                     EBM_ASSERT(0 <= cShift);
                     EBM_ASSERT(cShift < COUNT_BITS(UIntShared));
                     bits |= static_cast<UIntShared>(indexBin) << cShift;
                     cShift -= cBitsPerItemMax;
                  } while(0 <= cShift);
                  cShift = cShiftReset;
                  *pFillData = bits;
                  ++pFillData;
               } while(pBinIndexsEnd != pBinIndex);
               EBM_ASSERT(reinterpret_cast<unsigned char *>(pFillData) == pFillMem + iByteNext);
            }
            iByteCur = iByteNext;
         }
      }

      if(nullptr != pFillMem) {
         HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
         EBM_ASSERT(k_sharedDataSetWorkingId == pHeaderDataSetShared->m_id);

         ++iOffset;
         const size_t cOffsets = static_cast<size_t>(pHeaderDataSetShared->m_cFeatures) + 
            static_cast<size_t>(pHeaderDataSetShared->m_cWeights) + 
            static_cast<size_t>(pHeaderDataSetShared->m_cTargets);
         
         if(iOffset == cOffsets) {
            if(cBytesAllocated != iByteCur) {
               LOG_0(Trace_Error, "ERROR AppendFeature buffer size and fill size do not agree");
               goto return_bad;
            }

            const ErrorEbm error = LockDataSetShared(cBytesAllocated, pFillMem);
            if(Error_None != error) {
               return error;
            }
         } else {
            if(cBytesAllocated - sizeof(UIntShared) < iByteCur) {
               LOG_0(Trace_Error, "ERROR AppendFeature cBytesAllocated - sizeof(UIntShared) < iByteNext");
               goto return_bad;
            }

            if(IsConvertError<UIntShared>(iOffset)) {
               LOG_0(Trace_Error, "ERROR AppendFeature IsConvertError<IntEbm>(iOffset)");
               goto return_bad;
            }
            if(IsConvertError<UIntShared>(iByteCur)) {
               LOG_0(Trace_Error, "ERROR AppendFeature IsConvertError<UIntShared>(iByteCur)");
               goto return_bad;
            }
            ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset] = static_cast<UIntShared>(iByteCur);
            UIntShared * const pInternalState =
               reinterpret_cast<UIntShared *>(pFillMem + cBytesAllocated - sizeof(UIntShared));
            *pInternalState = static_cast<UIntShared>(iOffset); // the offset index is our state
         }
         return Error_None;
      }
      if(IsConvertError<IntEbm>(iByteCur)) {
         LOG_0(Trace_Error, "ERROR AppendFeature IsConvertError<IntEbm>(iByteCur)");
         goto return_bad;
      }
      return static_cast<IntEbm>(iByteCur);
   }

return_bad:;

   if(nullptr != pFillMem) {
      HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
      pHeaderDataSetShared->m_id = k_sharedDataSetErrorId;
   }
   return Error_IllegalParamVal;
}
WARNING_POP

WARNING_PUSH
WARNING_REDUNDANT_CODE
static IntEbm AppendWeight(
   const IntEbm countSamples,
   const double * aWeights,
   const size_t cBytesAllocated,
   unsigned char * const pFillMem
) {
   EBM_ASSERT(size_t { 0 } == cBytesAllocated && nullptr == pFillMem || 
      nullptr != pFillMem && k_cBytesHeaderId <= cBytesAllocated);

   LOG_N(
      Trace_Info,
      "Entered AppendWeight: "
      "countSamples=%" IntEbmPrintf ", "
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
      if(IsConvertError<size_t>(countSamples) || IsConvertError<UIntShared>(countSamples)) {
         LOG_0(Trace_Error, "ERROR AppendWeight countSamples is outside the range of a valid index");
         goto return_bad;
      }
      const size_t cSamples = static_cast<size_t>(countSamples);

      size_t iOffset = 0;
      size_t iByteCur = sizeof(WeightDataSetShared);
      if(nullptr != pFillMem) {
         if(IsHeaderError(static_cast<UIntShared>(cSamples), cBytesAllocated, pFillMem)) {
            goto return_bad;
         }

         UIntShared * const pInternalState =
            reinterpret_cast<UIntShared *>(pFillMem + cBytesAllocated - sizeof(UIntShared));

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

         EBM_ASSERT(size_t { 0 } == iOffset && UIntShared { 0 } == pHeaderDataSetShared->m_cSamples ||
            static_cast<UIntShared>(cSamples) == pHeaderDataSetShared->m_cSamples);
         pHeaderDataSetShared->m_cSamples = static_cast<UIntShared>(cSamples);

         WeightDataSetShared * const pWeightDataSetShared = reinterpret_cast<WeightDataSetShared *>(pFillMem + iHighestOffset);
         pWeightDataSetShared->m_id = k_weightId;
      }

      if(size_t { 0 } != cSamples) {
         if(nullptr == aWeights) {
            LOG_0(Trace_Error, "ERROR AppendWeight nullptr == aWeights");
            goto return_bad;
         }

         if(IsMultiplyError(EbmMax(sizeof(*aWeights), sizeof(FloatShared)), cSamples)) {
            LOG_0(Trace_Error, "ERROR AppendWeight IsMultiplyError(EbmMax(sizeof(*aWeights), sizeof(FloatShared)), cSamples)");
            goto return_bad;
         }
         const size_t cBytesAllSamples = sizeof(FloatShared) * cSamples;

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

            FloatShared * pFill = reinterpret_cast<FloatShared *>(pFillMem + iByteCur);
            const double * pWeight = aWeights;
            const double * const pWeightsEnd = &aWeights[cSamples];
            do {
               const double weight = *pWeight;

               if(std::isnan(weight)) {
                  LOG_0(Trace_Warning, "WARNING AppendWeight std::isnan(weight)");
                  goto return_bad;
               }
               if(std::isinf(weight)) {
                  LOG_0(Trace_Warning, "WARNING AppendWeight std::isinf(weight)");
                  goto return_bad;
               }
               if(weight < static_cast<double>(std::numeric_limits<float>::min())) {
                  // we use floats internally in some places, so limit the weight to a float minimum
                  LOG_0(Trace_Warning, "WARNING AppendWeight weight < static_cast<double>(std::numeric_limits<float>::min())");
                  goto return_bad;
               }
               if(static_cast<double>(std::numeric_limits<float>::max()) < weight) {
                  // we use floats internally in some places, so limit the weight to a float maximum
                  LOG_0(Trace_Warning, "WARNING AppendWeight static_cast<double>(std::numeric_limits<float>::max()) < weight");
                  goto return_bad;
               }

               *pFill = static_cast<FloatShared>(weight);
               ++pFill;
               ++pWeight;
            } while(pWeightsEnd != pWeight);
         }
         iByteCur = iByteNext;
      }

      if(nullptr != pFillMem) {
         HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
         EBM_ASSERT(k_sharedDataSetWorkingId == pHeaderDataSetShared->m_id);

         ++iOffset;
         const size_t cOffsets = static_cast<size_t>(pHeaderDataSetShared->m_cFeatures) +
            static_cast<size_t>(pHeaderDataSetShared->m_cWeights) +
            static_cast<size_t>(pHeaderDataSetShared->m_cTargets);

         if(iOffset == cOffsets) {
            if(cBytesAllocated != iByteCur) {
               LOG_0(Trace_Error, "ERROR AppendWeight buffer size and fill size do not agree");
               goto return_bad;
            }

            const ErrorEbm error = LockDataSetShared(cBytesAllocated, pFillMem);
            if(Error_None != error) {
               return error;
            }
         } else {
            if(cBytesAllocated - sizeof(UIntShared) < iByteCur) {
               LOG_0(Trace_Error, "ERROR AppendWeight cBytesAllocated - sizeof(UIntShared) < iByteCur");
               goto return_bad;
            }

            if(IsConvertError<UIntShared>(iOffset)) {
               LOG_0(Trace_Error, "ERROR AppendWeight IsConvertError<IntEbm>(iOffset)");
               goto return_bad;
            }
            if(IsConvertError<UIntShared>(iByteCur)) {
               LOG_0(Trace_Error, "ERROR AppendWeight IsConvertError<UIntShared>(iByteCur)");
               goto return_bad;
            }
            ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset] = static_cast<UIntShared>(iByteCur);
            UIntShared * const pInternalState =
               reinterpret_cast<UIntShared *>(pFillMem + cBytesAllocated - sizeof(UIntShared));
            *pInternalState = static_cast<UIntShared>(iOffset); // the offset index is our state
         }
         return Error_None;
      }
      if(IsConvertError<IntEbm>(iByteCur)) {
         LOG_0(Trace_Error, "ERROR AppendWeight IsConvertError<IntEbm>(iByteCur)");
         goto return_bad;
      }
      return static_cast<IntEbm>(iByteCur);
   }

return_bad:;

   if(nullptr != pFillMem) {
      HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
      pHeaderDataSetShared->m_id = k_sharedDataSetErrorId;
   }
   return Error_IllegalParamVal;
}
WARNING_POP

WARNING_PUSH
WARNING_REDUNDANT_CODE
static IntEbm AppendTarget(
   const bool bClassification,
   const IntEbm countClasses,
   const IntEbm countSamples,
   const void * aTargets,
   const size_t cBytesAllocated,
   unsigned char * const pFillMem
) {
   EBM_ASSERT(size_t { 0 } == cBytesAllocated && nullptr == pFillMem ||
      nullptr != pFillMem && k_cBytesHeaderId <= cBytesAllocated);

   LOG_N(
      Trace_Info,
      "Entered AppendTarget: "
      "bClassification=%s, "
      "countClasses=%" IntEbmPrintf ", "
      "countSamples=%" IntEbmPrintf ", "
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
      if(IsConvertError<UIntShared>(countClasses)) {
         LOG_0(Trace_Error, "ERROR AppendTarget countClasses is outside the range of a valid index");
         goto return_bad;
      }
      if(IsConvertError<size_t>(countSamples) || IsConvertError<UIntShared>(countSamples)) {
         LOG_0(Trace_Error, "ERROR AppendTarget countSamples is outside the range of a valid index");
         goto return_bad;
      }
      const size_t cSamples = static_cast<size_t>(countSamples);

      size_t iOffset = 0;
      size_t iByteCur = bClassification ? sizeof(TargetDataSetShared) + sizeof(ClassificationTargetDataSetShared) :
         sizeof(TargetDataSetShared);
      if(nullptr != pFillMem) {
         if(IsHeaderError(static_cast<UIntShared>(cSamples), cBytesAllocated, pFillMem)) {
            goto return_bad;
         }

         UIntShared * const pInternalState =
            reinterpret_cast<UIntShared *>(pFillMem + cBytesAllocated - sizeof(UIntShared));

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

         EBM_ASSERT(size_t { 0 } == iOffset && UIntShared { 0 } == pHeaderDataSetShared->m_cSamples ||
            static_cast<UIntShared>(cSamples) == pHeaderDataSetShared->m_cSamples);
         pHeaderDataSetShared->m_cSamples = static_cast<UIntShared>(cSamples);

         unsigned char * const pFillMemTemp = pFillMem + iHighestOffset;
         TargetDataSetShared * const pTargetDataSetShared = reinterpret_cast<TargetDataSetShared *>(pFillMemTemp);
         pTargetDataSetShared->m_id = GetTargetId(bClassification);

         if(bClassification) {
            ClassificationTargetDataSetShared * pClassificationTargetDataSetShared = 
               reinterpret_cast<ClassificationTargetDataSetShared *>(pFillMemTemp + sizeof(TargetDataSetShared));
            pClassificationTargetDataSetShared->m_cClasses = static_cast<UIntShared>(countClasses);
         }
      }

      if(size_t { 0 } != cSamples) {
         if(nullptr == aTargets) {
            LOG_0(Trace_Error, "ERROR AppendTarget nullptr == aTargets");
            goto return_bad;
         }

         size_t cBytesAllSamples;
         if(bClassification) {
            if(IsMultiplyError(EbmMax(sizeof(IntEbm), sizeof(UIntShared)), cSamples)) {
               LOG_0(Trace_Error, "ERROR AppendTarget IsMultiplyError(EbmMax(sizeof(IntEbm), sizeof(UIntShared)), cSamples)");
               goto return_bad;
            }
            cBytesAllSamples = sizeof(UIntShared) * cSamples;
         } else {
            if(IsMultiplyError(EbmMax(sizeof(double), sizeof(FloatShared)), cSamples)) {
               LOG_0(Trace_Error, "ERROR AppendTarget IsMultiplyError(EbmMax(sizeof(double), sizeof(FloatShared)), cSamples)");
               goto return_bad;
            }
            cBytesAllSamples = sizeof(FloatShared) * cSamples;
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
               const IntEbm * pTarget = reinterpret_cast<const IntEbm *>(aTargets);
               if(IsMultiplyError(sizeof(pTarget[0]), cSamples)) {
                  LOG_0(Trace_Error, "ERROR AppendTarget IsMultiplyError(sizeof(UIntShared), cSamples)");
                  goto return_bad;
               }
               const IntEbm * const pTargetsEnd = pTarget + cSamples;
               UIntShared * pFillData = reinterpret_cast<UIntShared *>(pFillMem + iByteCur);
               do {
                  const IntEbm target = *pTarget;
                  if(target < IntEbm { 0 }) {
                     LOG_0(Trace_Error, "ERROR AppendTarget classification target can't be negative");
                     goto return_bad;
                  }
                  if(countClasses <= target) {
                     LOG_0(Trace_Error, "ERROR AppendTarget countClasses <= target");
                     goto return_bad;
                  }
                  // since countClasses can be converted to UIntShared, so now can target
                  EBM_ASSERT(!IsConvertError<UIntShared>(target));
               
                  *pFillData = static_cast<UIntShared>(target);
               
                  ++pFillData;
                  ++pTarget;
               } while(pTargetsEnd != pTarget);
               EBM_ASSERT(reinterpret_cast<unsigned char *>(pFillData) == pFillMem + iByteNext);
            } else {
               FloatShared * pFill = reinterpret_cast<FloatShared *>(pFillMem + iByteCur);
               const double * pTarget = reinterpret_cast<const double *>(aTargets);
               const double * const pTargetsEnd = &pTarget[cSamples];
               do {
                  const double target = *pTarget;
                  const double cleaned = CleanFloat(target);
                  if(std::isnan(cleaned)) {
                     LOG_0(Trace_Error, "ERROR AppendTarget target is NaN");
                     goto return_bad;
                  }
                  if(std::isinf(cleaned)) {
                     LOG_0(Trace_Error, "ERROR AppendTarget target is infinity");
                     goto return_bad;
                  }
                  const FloatShared converted = static_cast<FloatShared>(cleaned);

                  // TODO: clean this float in float32 format

                  if(std::isnan(converted)) {
                     LOG_0(Trace_Error, "ERROR AppendTarget target is NaN after conversion to float32");
                     goto return_bad;
                  }
                  if(std::isinf(converted)) {
                     LOG_0(Trace_Error, "ERROR AppendTarget target is infinity after conversion to float32");
                     goto return_bad;
                  }
                  *pFill = converted;
                  ++pFill;
                  ++pTarget;
               } while(pTargetsEnd != pTarget);
            }
         }
         iByteCur = iByteNext;
      }

      if(nullptr != pFillMem) {
         HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
         EBM_ASSERT(k_sharedDataSetWorkingId == pHeaderDataSetShared->m_id);

         ++iOffset;
         const size_t cOffsets = static_cast<size_t>(pHeaderDataSetShared->m_cFeatures) +
            static_cast<size_t>(pHeaderDataSetShared->m_cWeights) +
            static_cast<size_t>(pHeaderDataSetShared->m_cTargets);

         if(iOffset == cOffsets) {
            if(cBytesAllocated != iByteCur) {
               LOG_0(Trace_Error, "ERROR AppendTarget buffer size and fill size do not agree");
               goto return_bad;
            }

            const ErrorEbm error = LockDataSetShared(cBytesAllocated, pFillMem);
            if(Error_None != error) {
               return error;
            }
         } else {
            if(cBytesAllocated - sizeof(UIntShared) < iByteCur) {
               LOG_0(Trace_Error, "ERROR AppendTarget cBytesAllocated - sizeof(UIntShared) < iByteCur");
               goto return_bad;
            }
            if(IsConvertError<UIntShared>(iOffset)) {
               LOG_0(Trace_Error, "ERROR AppendTarget IsConvertError<IntEbm>(iOffset)");
               goto return_bad;
            }
            if(IsConvertError<UIntShared>(iByteCur)) {
               LOG_0(Trace_Error, "ERROR AppendTarget IsConvertError<UIntShared>(iByteCur)");
               goto return_bad;
            }
            ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset] = static_cast<UIntShared>(iByteCur);
            UIntShared * const pInternalState =
               reinterpret_cast<UIntShared *>(pFillMem + cBytesAllocated - sizeof(UIntShared));
            *pInternalState = static_cast<UIntShared>(iOffset); // the offset index is our state
         }
         return Error_None;
      }
      if(IsConvertError<IntEbm>(iByteCur)) {
         LOG_0(Trace_Error, "ERROR AppendTarget IsConvertError<IntEbm>(iByteCur)");
         goto return_bad;
      }
      return static_cast<IntEbm>(iByteCur);
   }

return_bad:;

   if(nullptr != pFillMem) {
      HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
      pHeaderDataSetShared->m_id = k_sharedDataSetErrorId;
   }
   return Error_IllegalParamVal;
}
WARNING_POP

EBM_API_BODY IntEbm EBM_CALLING_CONVENTION MeasureDataSetHeader(
   IntEbm countFeatures,
   IntEbm countWeights,
   IntEbm countTargets
) {
   return AppendHeader(countFeatures, countWeights, countTargets, 0, nullptr);
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION FillDataSetHeader(
   IntEbm countFeatures,
   IntEbm countWeights,
   IntEbm countTargets,
   IntEbm countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(Trace_Error, "ERROR FillDataSetHeader nullptr == fillMem");
      return Error_IllegalParamVal;
   }

   if(IsConvertError<size_t>(countBytesAllocated)) {
      LOG_0(Trace_Error, "ERROR FillDataSetHeader countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamVal;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   const IntEbm ret = AppendHeader(
      countFeatures, 
      countWeights, 
      countTargets, 
      cBytesAllocated, 
      static_cast<unsigned char *>(fillMem)
   );
   return static_cast<ErrorEbm>(ret);
}

EBM_API_BODY IntEbm EBM_CALLING_CONVENTION MeasureFeature(
   IntEbm countBins,
   BoolEbm isMissing,
   BoolEbm isUnknown,
   BoolEbm isNominal,
   IntEbm countSamples,
   const IntEbm * binIndexes
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

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION FillFeature(
   IntEbm countBins,
   BoolEbm isMissing,
   BoolEbm isUnknown,
   BoolEbm isNominal,
   IntEbm countSamples,
   const IntEbm * binIndexes,
   IntEbm countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(Trace_Error, "ERROR FillFeature nullptr == fillMem");
      return Error_IllegalParamVal;
   }

   if(IsConvertError<size_t>(countBytesAllocated)) {
      LOG_0(Trace_Error, "ERROR FillFeature countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamVal;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   if(cBytesAllocated < k_cBytesHeaderId) {
      LOG_0(Trace_Error, "ERROR FillFeature cBytesAllocated < k_cBytesHeaderId");
      // don't check or set the header to bad if we don't have enough memory for the header id itself
      return Error_IllegalParamVal;
   }

   HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(fillMem);
   if(k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id) {
      LOG_0(Trace_Error, "ERROR FillFeature k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id");
      // don't set the header to bad since it's already set to something invalid and we don't know why
      return Error_IllegalParamVal;
   }

   const IntEbm ret = AppendFeature(
      countBins,
      isMissing,
      isUnknown,
      isNominal,
      countSamples,
      binIndexes,
      cBytesAllocated,
      static_cast<unsigned char *>(fillMem)
   );
   return static_cast<ErrorEbm>(ret);
}

EBM_API_BODY IntEbm EBM_CALLING_CONVENTION MeasureWeight(
   IntEbm countSamples,
   const double * weights
) {
   return AppendWeight(
      countSamples,
      weights,
      0,
      nullptr
   );
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION FillWeight(
   IntEbm countSamples,
   const double * weights,
   IntEbm countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(Trace_Error, "ERROR FillWeight nullptr == fillMem");
      return Error_IllegalParamVal;
   }

   if(IsConvertError<size_t>(countBytesAllocated)) {
      LOG_0(Trace_Error, "ERROR FillWeight countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamVal;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   if(cBytesAllocated < k_cBytesHeaderId) {
      LOG_0(Trace_Error, "ERROR FillWeight cBytesAllocated < k_cBytesHeaderId");
      // don't check or set the header to bad if we don't have enough memory for the header id itself
      return Error_IllegalParamVal;
   }

   HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(fillMem);
   if(k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id) {
      LOG_0(Trace_Error, "ERROR FillWeight k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id");
      // don't set the header to bad since it's already set to something invalid and we don't know why
      return Error_IllegalParamVal;
   }

   const IntEbm ret = AppendWeight(
      countSamples,
      weights,
      cBytesAllocated,
      static_cast<unsigned char *>(fillMem)
   );
   return static_cast<ErrorEbm>(ret);
}

EBM_API_BODY IntEbm EBM_CALLING_CONVENTION MeasureClassificationTarget(
   IntEbm countClasses,
   IntEbm countSamples,
   const IntEbm * targets
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

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION FillClassificationTarget(
   IntEbm countClasses,
   IntEbm countSamples,
   const IntEbm * targets,
   IntEbm countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(Trace_Error, "ERROR FillClassificationTarget nullptr == fillMem");
      return Error_IllegalParamVal;
   }

   if(IsConvertError<size_t>(countBytesAllocated)) {
      LOG_0(Trace_Error, "ERROR FillClassificationTarget countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamVal;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   if(cBytesAllocated < k_cBytesHeaderId) {
      LOG_0(Trace_Error, "ERROR FillClassificationTarget cBytesAllocated < k_cBytesHeaderId");
      // don't check or set the header to bad if we don't have enough memory for the header id itself
      return Error_IllegalParamVal;
   }

   HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(fillMem);
   if(k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id) {
      LOG_0(Trace_Error, "ERROR FillClassificationTarget k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id");
      // don't set the header to bad since it's already set to something invalid and we don't know why
      return Error_IllegalParamVal;
   }

   const IntEbm ret = AppendTarget(
      true,
      countClasses,
      countSamples,
      targets,
      cBytesAllocated,
      static_cast<unsigned char *>(fillMem)
   );
   return static_cast<ErrorEbm>(ret);
}

EBM_API_BODY IntEbm EBM_CALLING_CONVENTION MeasureRegressionTarget(
   IntEbm countSamples,
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

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION FillRegressionTarget(
   IntEbm countSamples,
   const double * targets,
   IntEbm countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(Trace_Error, "ERROR FillRegressionTarget nullptr == fillMem");
      return Error_IllegalParamVal;
   }

   if(IsConvertError<size_t>(countBytesAllocated)) {
      LOG_0(Trace_Error, "ERROR FillRegressionTarget countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamVal;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   if(cBytesAllocated < k_cBytesHeaderId) {
      LOG_0(Trace_Error, "ERROR FillRegressionTarget cBytesAllocated < k_cBytesHeaderId");
      // don't check or set the header to bad if we don't have enough memory for the header id itself
      return Error_IllegalParamVal;
   }

   HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(fillMem);
   if(k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id) {
      LOG_0(Trace_Error, "ERROR FillRegressionTarget k_sharedDataSetWorkingId != pHeaderDataSetShared->m_id");
      // don't set the header to bad since it's already set to something invalid and we don't know why
      return Error_IllegalParamVal;
   }

   const IntEbm ret = AppendTarget(
      false,
      0,
      countSamples,
      targets,
      cBytesAllocated,
      static_cast<unsigned char *>(fillMem)
   );
   return static_cast<ErrorEbm>(ret);
}

extern ErrorEbm GetDataSetSharedHeader(
   const unsigned char * const pDataSetShared,
   UIntShared * const pcSamplesOut,
   size_t * const pcFeaturesOut,
   size_t * const pcWeightsOut,
   size_t * const pcTargetsOut
) {
   EBM_ASSERT(nullptr != pcSamplesOut);
   EBM_ASSERT(nullptr != pcFeaturesOut);
   EBM_ASSERT(nullptr != pcWeightsOut);
   EBM_ASSERT(nullptr != pcTargetsOut);

   const ErrorEbm error = CheckDataSet(0, pDataSetShared);
   if(Error_None != error) {
      return error;
   }
   EBM_ASSERT(nullptr != pDataSetShared); // checked in CheckDataSet

   const HeaderDataSetShared * const pHeaderDataSetShared = 
      reinterpret_cast<const HeaderDataSetShared *>(pDataSetShared);
   EBM_ASSERT(k_sharedDataSetDoneId == pHeaderDataSetShared->m_id);

   // our shared dataset allows some cases where there can be more samples then can fit into a size_t. Caller checks
   *pcSamplesOut = pHeaderDataSetShared->m_cSamples;

   const UIntShared countFeatures = pHeaderDataSetShared->m_cFeatures;
   EBM_ASSERT(!IsConvertError<size_t>(countFeatures)); // if we can fit into memory then this must be convertible
   *pcFeaturesOut = static_cast<size_t>(countFeatures);

   const UIntShared countWeights = pHeaderDataSetShared->m_cWeights;
   EBM_ASSERT(!IsConvertError<size_t>(countWeights)); // if we can fit into memory then this must be convertible
   *pcWeightsOut = static_cast<size_t>(countWeights);

   const UIntShared countTargets = pHeaderDataSetShared->m_cTargets;
   EBM_ASSERT(!IsConvertError<size_t>(countTargets)); // if we can fit into memory then this must be convertible
   *pcTargetsOut = static_cast<size_t>(countTargets);

   return Error_None;
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION ExtractDataSetHeader(
   const void * dataSet,
   IntEbm * countSamplesOut,
   IntEbm * countFeaturesOut,
   IntEbm * countWeightsOut,
   IntEbm * countTargetsOut
) {
   UIntShared countSamples;
   size_t cFeatures;
   size_t cWeights;
   size_t cTargets;

   const ErrorEbm error = GetDataSetSharedHeader(
      static_cast<const unsigned char *>(dataSet),
      &countSamples,
      &cFeatures,
      &cWeights,
      &cTargets
   );
   if(Error_None != error) {
      // already logged
      return error;
   }
   EBM_ASSERT(nullptr != dataSet); // checked in GetDataSetSharedHeader

   if(IsConvertError<IntEbm>(countSamples)) {
      // countSamples should have originally came to us as an IntEbm, but check in case of corruption
      LOG_0(Trace_Error, "ERROR ExtractDataSetHeader IsConvertError<IntEbm>(countSamples)");
      return Error_IllegalParamVal;
   }
   if(IsConvertError<IntEbm>(cFeatures)) {
      // cFeatures should have originally came to us as an IntEbm, but check in case of corruption
      LOG_0(Trace_Error, "ERROR ExtractDataSetHeader IsConvertError<IntEbm>(cFeatures)");
      return Error_IllegalParamVal;
   }
   if(IsConvertError<IntEbm>(cWeights)) {
      // cWeights should have originally came to us as an IntEbm, but check in case of corruption
      LOG_0(Trace_Error, "ERROR ExtractDataSetHeader IsConvertError<IntEbm>(cWeights)");
      return Error_IllegalParamVal;
   }
   if(IsConvertError<IntEbm>(cTargets)) {
      // cTargets should have originally came to us as an IntEbm, but check in case of corruption
      LOG_0(Trace_Error, "ERROR ExtractDataSetHeader IsConvertError<IntEbm>(cTargets)");
      return Error_IllegalParamVal;
   }

   if(nullptr != countSamplesOut) {
      *countSamplesOut = static_cast<IntEbm>(countSamples);
   }
   if(nullptr != countFeaturesOut) {
      *countFeaturesOut = static_cast<IntEbm>(cFeatures);
   }
   if(nullptr != countWeightsOut) {
      *countWeightsOut = static_cast<IntEbm>(cWeights);
   }
   if(nullptr != countTargetsOut) {
      *countTargetsOut = static_cast<IntEbm>(cTargets);
   }

   return Error_None;
}

// TODO: make an inline wrapper that forces this to the correct type and have 2 differently named functions
// GetDataSetSharedFeature will return either (SparseFeatureDataSetSharedEntry *) or (UIntShared *)
extern const void * GetDataSetSharedFeature(
   const unsigned char * const pDataSetShared,
   const size_t iFeature,
   bool * const pbMissingOut,
   bool * const pbUnknownOut,
   bool * const pbNominalOut,
   bool * const pbSparseOut,
   UIntShared * const pcBinsOut,
   UIntShared * const pDefaultValSparseOut,
   size_t * const pcNonDefaultsSparseOut
) {
   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(nullptr != pbMissingOut);
   EBM_ASSERT(nullptr != pbUnknownOut);
   EBM_ASSERT(nullptr != pbNominalOut);
   EBM_ASSERT(nullptr != pbSparseOut);
   EBM_ASSERT(nullptr != pcBinsOut);
   EBM_ASSERT(nullptr != pDefaultValSparseOut);
   EBM_ASSERT(nullptr != pcNonDefaultsSparseOut);

   const HeaderDataSetShared * const pHeaderDataSetShared = 
      reinterpret_cast<const HeaderDataSetShared *>(pDataSetShared);
   EBM_ASSERT(k_sharedDataSetDoneId == pHeaderDataSetShared->m_id);

   EBM_ASSERT(!IsConvertError<size_t>(pHeaderDataSetShared->m_cFeatures));
   EBM_ASSERT(iFeature < static_cast<size_t>(pHeaderDataSetShared->m_cFeatures));

   EBM_ASSERT(!IsMultiplyError(sizeof(pHeaderDataSetShared->m_offsets[0]), iFeature));
   const UIntShared indexMem = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iFeature];
   EBM_ASSERT(!IsConvertError<size_t>(indexMem)); // it is allocated and we trust it (or should have verified it)
   const size_t iMem = static_cast<size_t>(indexMem);

   const FeatureDataSetShared * pFeatureDataSetShared = 
      reinterpret_cast<const FeatureDataSetShared *>(pDataSetShared + iMem);

   const UIntShared id = pFeatureDataSetShared->m_id;
   EBM_ASSERT(IsFeature(id));
   *pbMissingOut = IsMissingFeature(id);
   *pbUnknownOut = IsUnknownFeature(id);
   *pbNominalOut = IsNominalFeature(id);
   const bool bSparse = IsSparseFeature(id);
   *pbSparseOut = bSparse;

   *pcBinsOut = pFeatureDataSetShared->m_cBins;

   const void * pRet = reinterpret_cast<const void *>(pFeatureDataSetShared + 1);
   if(bSparse) {
      const SparseFeatureDataSetShared * const pSparseFeatureDataSetShared =
         reinterpret_cast<const SparseFeatureDataSetShared *>(pRet);

      *pDefaultValSparseOut = pSparseFeatureDataSetShared->m_defaultVal;
      const UIntShared countNonDefaults = pSparseFeatureDataSetShared->m_cNonDefaults;
      EBM_ASSERT(!IsConvertError<size_t>(countNonDefaults)); // it's allocated so must be in memory
      *pcNonDefaultsSparseOut = static_cast<size_t>(countNonDefaults);
      pRet = reinterpret_cast<const void *>(pSparseFeatureDataSetShared->m_nonDefaults);
   }
   return pRet;
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION ExtractBinCounts(
   const void * dataSet,
   IntEbm countFeaturesVerify,
   IntEbm * binCountsOut
) {
   if(nullptr == dataSet) {
      LOG_0(Trace_Error, "ERROR ExtractBinCounts nullptr == dataSet");
      return Error_IllegalParamVal;
   }

   if(IsConvertError<size_t>(countFeaturesVerify)) {
      LOG_0(Trace_Error, "ERROR ExtractBinCounts IsConvertError<size_t>(countFeaturesVerify)");
      return Error_IllegalParamVal;
   }
   const size_t cFeaturesVerify = static_cast<size_t>(countFeaturesVerify);

   const HeaderDataSetShared * const pHeaderDataSetShared =
      reinterpret_cast<const HeaderDataSetShared *>(dataSet);

   if(k_sharedDataSetDoneId != pHeaderDataSetShared->m_id) {
      LOG_0(Trace_Error, "ERROR ExtractBinCounts k_sharedDataSetDoneId != pHeaderDataSetShared->m_id");
      return Error_IllegalParamVal;
   }

   const UIntShared countFeatures = pHeaderDataSetShared->m_cFeatures;
   EBM_ASSERT(!IsConvertError<size_t>(countFeatures)); // it's allocated so must fit into size_t
   size_t cFeatures = static_cast<size_t>(countFeatures);

   if(cFeatures != cFeaturesVerify) {
      LOG_0(Trace_Error, "ERROR ExtractBinCounts cFeatures != cFeaturesVerify");
      return Error_IllegalParamVal;
   }
   if(size_t { 0 } != cFeatures) {
      if(nullptr == binCountsOut) {
         LOG_0(Trace_Error, "ERROR ExtractBinCounts nullptr == binCountsOut");
         return Error_IllegalParamVal;
      }

      const UIntShared * pOffset = ArrayToPointer(pHeaderDataSetShared->m_offsets);
      IntEbm * pcBins = binCountsOut;
      const IntEbm * const pcBinsEnd = binCountsOut + cFeatures;
      do {
         const UIntShared indexOffsetCur = *pOffset;
         ++pOffset;

         EBM_ASSERT(!IsConvertError<size_t>(indexOffsetCur)); // it is in allocated space so size_t must fit
         const size_t iOffsetCur = static_cast<size_t>(indexOffsetCur);

         const FeatureDataSetShared * pFeatureDataSetShared =
            reinterpret_cast<const FeatureDataSetShared *>(static_cast<const char *>(dataSet) + iOffsetCur);

         EBM_ASSERT(IsFeature(pFeatureDataSetShared->m_id));

         UIntShared countBins = pFeatureDataSetShared->m_cBins;
         // countBins originally fit into UIntShared so it should still with these additions
         countBins += IsMissingFeature(pFeatureDataSetShared->m_id) ? UIntShared { 0 } : UIntShared { 1 };
         countBins += IsUnknownFeature(pFeatureDataSetShared->m_id) ? UIntShared { 0 } : UIntShared { 1 };
         if(IsConvertError<IntEbm>(countBins)) {
            LOG_0(Trace_Error, "ERROR ExtractBinCounts IsConvertError<IntEbm>(countBins)");
            return Error_IllegalParamVal;
         }

         *pcBins = static_cast<IntEbm>(countBins);
         ++pcBins;
      } while(pcBinsEnd != pcBins);
   }
   return Error_None;
}

extern const FloatShared * GetDataSetSharedWeight(
   const unsigned char * const pDataSetShared,
   const size_t iWeight
) {
   const HeaderDataSetShared * const pHeaderDataSetShared =
      reinterpret_cast<const HeaderDataSetShared *>(pDataSetShared);
   EBM_ASSERT(k_sharedDataSetDoneId == pHeaderDataSetShared->m_id);

   const UIntShared countFeatures = pHeaderDataSetShared->m_cFeatures;
   EBM_ASSERT(!IsConvertError<size_t>(countFeatures));
   const size_t cFeatures = static_cast<size_t>(countFeatures);

   EBM_ASSERT(!IsConvertError<size_t>(pHeaderDataSetShared->m_cWeights));
   EBM_ASSERT(iWeight < static_cast<size_t>(pHeaderDataSetShared->m_cWeights));

   EBM_ASSERT(!IsAddError(cFeatures, iWeight));
   const size_t iOffset = cFeatures + iWeight;

   EBM_ASSERT(!IsMultiplyError(sizeof(pHeaderDataSetShared->m_offsets[0]), iOffset));
   const UIntShared indexMem = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset];
   EBM_ASSERT(!IsConvertError<size_t>(indexMem));
   const size_t iMem = static_cast<size_t>(indexMem);

   const WeightDataSetShared * pWeightDataSetShared =
      reinterpret_cast<const WeightDataSetShared *>(pDataSetShared + iMem);

   EBM_ASSERT(k_weightId == pWeightDataSetShared->m_id);

   return reinterpret_cast<const FloatShared *>(pWeightDataSetShared + 1);
}

// TODO: make an inline wrapper that forces this to the correct type and have 2 differently named functions
// GetDataSetSharedTarget returns (FloatShared *) for regression and (UIntShared *) for classification
extern const void * GetDataSetSharedTarget(
   const unsigned char * const pDataSetShared,
   const size_t iTarget,
   ptrdiff_t * const pcClassesOut
) {
   const HeaderDataSetShared * const pHeaderDataSetShared =
      reinterpret_cast<const HeaderDataSetShared *>(pDataSetShared);
   EBM_ASSERT(k_sharedDataSetDoneId == pHeaderDataSetShared->m_id);

   const UIntShared countFeatures = pHeaderDataSetShared->m_cFeatures;
   EBM_ASSERT(!IsConvertError<size_t>(countFeatures));
   const size_t cFeatures = static_cast<size_t>(countFeatures);

   const UIntShared countWeights = pHeaderDataSetShared->m_cWeights;
   EBM_ASSERT(!IsConvertError<size_t>(countWeights));
   const size_t cWeights = static_cast<size_t>(countWeights);

   EBM_ASSERT(!IsConvertError<size_t>(pHeaderDataSetShared->m_cTargets));
   EBM_ASSERT(iTarget < static_cast<size_t>(pHeaderDataSetShared->m_cTargets));

   EBM_ASSERT(!IsAddError(cFeatures, cWeights, iTarget));
   const size_t iOffset = cFeatures + cWeights + iTarget;

   EBM_ASSERT(!IsMultiplyError(sizeof(pHeaderDataSetShared->m_offsets[0]), iOffset));
   const UIntShared indexMem = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iOffset];
   EBM_ASSERT(!IsConvertError<size_t>(indexMem));
   const size_t iMem = static_cast<size_t>(indexMem);

   const TargetDataSetShared * pTargetDataSetShared =
      reinterpret_cast<const TargetDataSetShared *>(pDataSetShared + iMem);

   const UIntShared id = pTargetDataSetShared->m_id;
   EBM_ASSERT(IsTarget(id));

   ptrdiff_t cClasses = ptrdiff_t { OutputType_Regression };
   const void * pRet = reinterpret_cast<const void *>(pTargetDataSetShared + 1);
   if(IsClassificationTarget(id)) {
      const ClassificationTargetDataSetShared * const pClassificationTargetDataSetShared =
         reinterpret_cast<const ClassificationTargetDataSetShared *>(pRet);

      const UIntShared countClasses = pClassificationTargetDataSetShared->m_cClasses;
      if(IsConvertError<ptrdiff_t>(countClasses)) {
         LOG_0(Trace_Error, "ERROR GetDataSetSharedTarget IsConvertError<ptrdiff_t>(countClasses)");
         return nullptr;
      }
      cClasses = static_cast<ptrdiff_t>(countClasses);
      EBM_ASSERT(0 <= cClasses); // 0 is possible with 0 samples
      pRet = reinterpret_cast<const void *>(pClassificationTargetDataSetShared + 1);
   }
   *pcClassesOut = cClasses;
   return pRet;
}

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION ExtractTargetClasses(
   const void * dataSet,
   IntEbm countTargetsVerify,
   IntEbm * classCountsOut
) {
   if(nullptr == dataSet) {
      LOG_0(Trace_Error, "ERROR ExtractTargetClasses nullptr == dataSet");
      return Error_IllegalParamVal;
   }

   if(IsConvertError<size_t>(countTargetsVerify)) {
      LOG_0(Trace_Error, "ERROR ExtractTargetClasses IsConvertError<size_t>(countTargetsVerify)");
      return Error_IllegalParamVal;
   }
   const size_t cTargetsVerify = static_cast<size_t>(countTargetsVerify);

   const HeaderDataSetShared * const pHeaderDataSetShared =
      reinterpret_cast<const HeaderDataSetShared *>(dataSet);

   if(k_sharedDataSetDoneId != pHeaderDataSetShared->m_id) {
      LOG_0(Trace_Error, "ERROR ExtractTargetClasses k_sharedDataSetDoneId != pHeaderDataSetShared->m_id");
      return Error_IllegalParamVal;
   }

   const UIntShared countFeatures = pHeaderDataSetShared->m_cFeatures;
   EBM_ASSERT(!IsConvertError<size_t>(countFeatures));
   const size_t cFeatures = static_cast<size_t>(countFeatures);

   const UIntShared countWeights = pHeaderDataSetShared->m_cWeights;
   EBM_ASSERT(!IsConvertError<size_t>(countWeights));
   const size_t cWeights = static_cast<size_t>(countWeights);

   const UIntShared countTargets = pHeaderDataSetShared->m_cTargets;
   EBM_ASSERT(!IsConvertError<size_t>(countTargets));
   const size_t cTargets = static_cast<size_t>(countTargets);

   if(cTargets != cTargetsVerify) {
      LOG_0(Trace_Error, "ERROR ExtractTargetClasses cTargets != cTargetsVerify");
      return Error_IllegalParamVal;
   }

   if(size_t { 0 } != cTargets) {
      if(nullptr == classCountsOut) {
         LOG_0(Trace_Error, "ERROR ExtractTargetClasses nullptr == classCountsOut");
         return Error_IllegalParamVal;
      }
      
      const UIntShared * pOffset = &ArrayToPointer(pHeaderDataSetShared->m_offsets)[cFeatures + cWeights];
      IntEbm * pcClasses = classCountsOut;
      const IntEbm * const pcClassesEnd = classCountsOut + cTargets;
      do {
         const UIntShared indexOffsetCur = *pOffset;
         ++pOffset;

         EBM_ASSERT(!IsConvertError<size_t>(indexOffsetCur));
         const size_t iOffsetCur = static_cast<size_t>(indexOffsetCur);

         const TargetDataSetShared * pTargetDataSetShared =
            reinterpret_cast<const TargetDataSetShared *>(static_cast<const char *>(dataSet) + iOffsetCur);

         const UIntShared id = pTargetDataSetShared->m_id;
         EBM_ASSERT(IsTarget(id));

         IntEbm countClasses = IntEbm { OutputType_Regression };
         if(IsClassificationTarget(id)) {
            const ClassificationTargetDataSetShared * const pClassificationTargetDataSetShared =
               reinterpret_cast<const ClassificationTargetDataSetShared *>(pTargetDataSetShared + 1);

            const UIntShared cClasses = pClassificationTargetDataSetShared->m_cClasses;

            if(IsConvertError<IntEbm>(cClasses)) {
               LOG_0(Trace_Error, "ERROR ExtractTargetClasses IsConvertError<IntEbm>(cClasses)");
               return Error_IllegalParamVal;
            }

            countClasses = static_cast<IntEbm>(cClasses);
         }

         *pcClasses = countClasses;
         ++pcClasses;
      } while(pcClassesEnd != pcClasses);
   }
   return Error_None;
}

} // DEFINED_ZONE_NAME
