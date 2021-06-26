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

// header ids
constexpr static SharedStorageDataType k_sharedDataSetId = 0x46DB; // random 15 bit number
constexpr static SharedStorageDataType k_sharedDataSetErrorId = 0x103; // anything other than our normal id will work

// feature ids
constexpr static SharedStorageDataType k_categoricalFeatureBit = 0x1;
constexpr static SharedStorageDataType k_sparseFeatureBit = 0x2;
constexpr static SharedStorageDataType k_featureId = 0x2B44; // random 15 bit number with lower 2 bits set to zero

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

   // m_offsets needs to be at the bottom of this struct.  We use the struct hack to size this array
   SharedStorageDataType m_offsets[1];
};
static_assert(std::is_standard_layout<HeaderDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");
static_assert(std::is_trivial<HeaderDataSetShared>::value,
   "These structs are shared between processes, so they definetly need to be standard layout and trivial");

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

size_t AppendHeader(const IntEbmType countFeatures, const size_t cBytesAllocated, char * const pFillMem) {
   EBM_ASSERT(size_t { 0 } == cBytesAllocated && nullptr == pFillMem || nullptr != pFillMem);

   LOG_N(
      TraceLevelInfo,
      "Entered AppendHeader: "
      "countFeatures=%" IntEbmTypePrintf ", "
      "cBytesAllocated=%zu, "
      "pFillMem=%p"
      ,
      countFeatures,
      cBytesAllocated,
      static_cast<void *>(pFillMem)
   );

   if(!IsNumberConvertable<size_t>(countFeatures) || !IsNumberConvertable<SharedStorageDataType>(countFeatures)) {
      LOG_0(TraceLevelError, "ERROR AppendHeader countFeatures is outside the range of a valid index");
      return 0;
   }
   const size_t cFeatures = static_cast<size_t>(countFeatures);

   if(IsMultiplyError(sizeof(HeaderDataSetShared::m_offsets[0]), cFeatures)) {
      LOG_0(TraceLevelError, "ERROR AppendHeader IsMultiplyError(sizeof(HeaderDataSetShared::m_offsets[0]), cFeatures)");
      return 0;
   }
   const size_t cBytesOffsets = sizeof(HeaderDataSetShared::m_offsets[0]) * cFeatures;

   if(IsAddError(sizeof(HeaderDataSetShared), cBytesOffsets)) {
      LOG_0(TraceLevelError, "ERROR AppendHeader IsAddError(sizeof(HeaderDataSetShared), cBytesOffsets)");
      return 0;
   }
   // keep the first element of m_offsets because we need one additional offset for accessing the targets
   const size_t cBytesHeader = sizeof(HeaderDataSetShared) + cBytesOffsets;

   if(!IsNumberConvertable<SharedStorageDataType>(cBytesHeader)) {
      LOG_0(TraceLevelError, "ERROR AppendHeader cBytesHeader is outside the range of a valid size");
      return 0;
   }

   if(nullptr != pFillMem) {
      EBM_ASSERT(sizeof(HeaderDataSetShared) <= cBytesAllocated); // checked by our caller

      HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);

      pHeaderDataSetShared->m_id = k_sharedDataSetId;
      // set samples to zero in case there are no features and the caller goes directly to targets
      pHeaderDataSetShared->m_cSamples = 0;
      pHeaderDataSetShared->m_cFeatures = 0;
      // position our first feature right after the header, or at the target if there are no features
      pHeaderDataSetShared->m_offsets[0] = static_cast<SharedStorageDataType>(cBytesHeader);
   }

   return cBytesHeader;
}

bool DecideIfSparse(const size_t cSamples, const IntEbmType * aBinnedData) {
   // For sparsity in the data set shared memory the only thing that matters is compactness since we don't use
   // this memory in any high performance loops

   UNUSED(cSamples);
   UNUSED(aBinnedData);

   // TODO: evalute the data to decide if the feature should be sparse or not
   return false;
}

size_t AppendFeatureData(
   const BoolEbmType categorical,
   const IntEbmType countBins,
   const IntEbmType countSamples,
   const IntEbmType * aBinnedData,
   const size_t cBytesAllocated, 
   char * const pFillMem
) {
   EBM_ASSERT(size_t { 0 } == cBytesAllocated && nullptr == pFillMem || nullptr != pFillMem);

   LOG_N(
      TraceLevelInfo,
      "Entered AppendFeatureData: "
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
         LOG_0(TraceLevelError, "ERROR AppendFeatureData categorical is not EBM_FALSE or EBM_TRUE");
         goto return_bad;
      }

      if(!IsNumberConvertable<size_t>(countBins) || !IsNumberConvertable<SharedStorageDataType>(countBins)) {
         LOG_0(TraceLevelError, "ERROR AppendFeatureData countBins is outside the range of a valid index");
         goto return_bad;
      }

      if(!IsNumberConvertable<size_t>(countSamples) || !IsNumberConvertable<SharedStorageDataType>(countSamples)) {
         LOG_0(TraceLevelError, "ERROR AppendFeatureData countSamples is outside the range of a valid index");
         goto return_bad;
      }
      const size_t cSamples = static_cast<size_t>(countSamples);

      if(nullptr == aBinnedData && size_t { 0 } != cSamples) {
         LOG_0(TraceLevelError, "ERROR AppendFeatureData nullptr == aBinnedData && size_t { 0 } != cSamples");
         goto return_bad;
      }

      // TODO: handle sparse data someday
      const bool bSparse = DecideIfSparse(cSamples, aBinnedData);

      size_t iByteCur = sizeof(FeatureDataSetShared);
      if(nullptr != pFillMem) {
         EBM_ASSERT(sizeof(HeaderDataSetShared) <= cBytesAllocated); // checked by our caller

         HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
         if(k_sharedDataSetId != pHeaderDataSetShared->m_id) {
            return 0; // don't set the header to bad since it's already set to something invalid and we don't know why
         }

         const SharedStorageDataType indexFeature = pHeaderDataSetShared->m_cFeatures;
         if(!IsNumberConvertable<size_t>(indexFeature)) {
            // we're being untrusting of the caller manipulating the memory improperly here
            LOG_0(TraceLevelError, "ERROR AppendFeatureData indexFeature is outside the range of a valid index");
            goto return_bad;
         }
         const size_t iFeature = static_cast<size_t>(indexFeature);

         const SharedStorageDataType offsetsEnd = pHeaderDataSetShared->m_offsets[0];
         if(!IsNumberConvertable<size_t>(offsetsEnd)) {
            LOG_0(TraceLevelError, "ERROR AppendFeatureData m_offsets[0] is outside the range of a valid index");
            goto return_bad;
         }
         const size_t iOffsetsEnd = static_cast<size_t>(offsetsEnd);

         if(cBytesAllocated <= iOffsetsEnd) {
            LOG_0(TraceLevelError, "ERROR AppendFeatureData cBytesAllocated <= iOffsetEndFeatures");
            goto return_bad;
         }

         if(iOffsetsEnd < sizeof(HeaderDataSetShared)) {
            LOG_0(TraceLevelError, "ERROR AppendFeatureData iOffsetsEnd < sizeof(HeaderDataSetShared)");
            goto return_bad;
         }
         const size_t cBytesInOffsets = iOffsetsEnd - sizeof(HeaderDataSetShared);
         if(size_t { 0 } != cBytesInOffsets % sizeof(HeaderDataSetShared::m_offsets[0])) {
            LOG_0(TraceLevelError, "ERROR AppendFeatureData size_t { 0 } != cBytesInOffsets % sizeof(HeaderDataSetShared::m_offsets[0])");
            goto return_bad;
         }
         const size_t cFeatures = cBytesInOffsets / sizeof(HeaderDataSetShared::m_offsets[0]);

         if(cFeatures <= iFeature) {
            LOG_0(TraceLevelError, "ERROR AppendFeatureData cFeatures <= iFeature");
            goto return_bad;
         }

         if(size_t { 0 } == iFeature) {
            if(SharedStorageDataType { 0 } != pHeaderDataSetShared->m_cSamples) {
               LOG_0(TraceLevelError, "ERROR AppendFeatureData SharedStorageDataType { 0 } != pHeaderDataSetShared->m_cSamples");
               goto return_bad;
            }
            pHeaderDataSetShared->m_cSamples = static_cast<SharedStorageDataType>(cSamples);
         } else {
            if(pHeaderDataSetShared->m_cSamples != static_cast<SharedStorageDataType>(cSamples)) {
               LOG_0(TraceLevelError, "ERROR AppendFeatureData pHeaderDataSetShared->m_cSamples != cSamples");
               goto return_bad;
            }
         }

         // this memory location is guaranteed to be below cBytesAllocated since we checked above that the
         // memory space AFTER the m_offsets array existed AND we've checked that our current iFeature doesn't
         // overflow that the m_offsets, so it stands to reason that the memory location at iFeature exists
         const SharedStorageDataType indexHighestOffset = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iFeature];
         if(!IsNumberConvertable<size_t>(indexHighestOffset)) {
            LOG_0(TraceLevelError, "ERROR AppendFeatureData indexByteCur is outside the range of a valid index");
            goto return_bad;
         }
         const size_t iHighestOffset = static_cast<size_t>(indexHighestOffset);

         if(IsAddError(iByteCur, iHighestOffset)) {
            LOG_0(TraceLevelError, "ERROR AppendFeatureData IsAddError(iByteCur, iHighestOffset)");
            goto return_bad;
         }
         iByteCur += iHighestOffset; // if we're going to access FeatureDataSetShared, then check if we have the space
         if(cBytesAllocated <= iByteCur) {
            LOG_0(TraceLevelError, "ERROR AppendFeatureData cBytesAllocated <= iByteCur");
            goto return_bad;
         }

         FeatureDataSetShared * pFeatureDataSetShared = reinterpret_cast<FeatureDataSetShared *>(pFillMem + iHighestOffset);
         pFeatureDataSetShared->m_id = GetFeatureId(EBM_FALSE != categorical, bSparse);
         pFeatureDataSetShared->m_cBins = static_cast<SharedStorageDataType>(countBins);
      }

      if(size_t { 0 } != cSamples) {
         if(IsMultiplyError(sizeof(SharedStorageDataType), cSamples)) {
            LOG_0(TraceLevelError, "ERROR AppendFeatureData IsMultiplyError(sizeof(SharedStorageDataType), cSamples)");
            goto return_bad;
         }
         size_t cBytesAllSamples = sizeof(SharedStorageDataType) * cSamples;
         if(IsAddError(iByteCur, cBytesAllSamples)) {
            LOG_0(TraceLevelError, "ERROR AppendFeatureData IsAddError(iByteCur, cBytesAllSamples)");
            goto return_bad;
         }
         const size_t iByteNext = iByteCur + cBytesAllSamples;
         if(nullptr != pFillMem) {
            if(cBytesAllocated <= iByteNext) {
               LOG_0(TraceLevelError, "ERROR AppendFeatureData cBytesAllocated <= iByteNext");
               goto return_bad;
            }

            if(IsMultiplyError(sizeof(aBinnedData[0]), cSamples)) {
               LOG_0(TraceLevelError, "ERROR AppendFeatureData IsMultiplyError(sizeof(aBinnedData[0]), cSamples)");
               goto return_bad;
            }
            const IntEbmType * pBinnedData = aBinnedData;
            const IntEbmType * const pBinnedDataEnd = aBinnedData + cSamples;
            SharedStorageDataType * pFillData = reinterpret_cast<SharedStorageDataType *>(pFillMem + iByteCur);
            do {
               const IntEbmType binnedData = *pBinnedData;
               if(binnedData < IntEbmType { 0 }) {
                  LOG_0(TraceLevelError, "ERROR AppendFeatureData binnedData can't be negative");
                  goto return_bad;
               }
               if(countBins <= binnedData) {
                  LOG_0(TraceLevelError, "ERROR AppendFeatureData countBins <= binnedData");
                  goto return_bad;
               }
               // since countBins can be converted to these, so now can binnedData
               EBM_ASSERT(IsNumberConvertable<size_t>(binnedData));
               EBM_ASSERT(IsNumberConvertable<SharedStorageDataType>(binnedData));

               // TODO: bit compact this
               *pFillData = static_cast<SharedStorageDataType>(binnedData);

               ++pFillData;
               ++pBinnedData;
            } while(pBinnedDataEnd != pBinnedData);
            EBM_ASSERT(reinterpret_cast<char *>(pFillData) == pFillMem + iByteNext);
         }
         iByteCur = iByteNext;
      }

      if(nullptr != pFillMem) {
         HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
         EBM_ASSERT(k_sharedDataSetId == pHeaderDataSetShared->m_id);

         // earlier we compared pHeaderDataSetShared->m_cFeatures with cFeatures calculated from the size of
         // the offset array, so we know there's at least one possible value greater than our current index value
         size_t iFeature = static_cast<size_t>(pHeaderDataSetShared->m_cFeatures);
         EBM_ASSERT(iFeature < std::numeric_limits<size_t>::max());
         ++iFeature;

         // we checked above that we would be below the first offset, so we can be converted to SharedStorageDataType
         // which is recorded as a SharedStorageDataType
         EBM_ASSERT(IsNumberConvertable<SharedStorageDataType>(iFeature));

         pHeaderDataSetShared->m_cFeatures = static_cast<SharedStorageDataType>(iFeature);
         // on the last feature we put the offset to the target entry

         if(!IsNumberConvertable<SharedStorageDataType>(iByteCur)) {
            LOG_0(TraceLevelError, "ERROR AppendFeatureData !IsNumberConvertable<SharedStorageDataType>(iByteCur)");
            goto return_bad;
         }
         ArrayToPointer(pHeaderDataSetShared->m_offsets)[iFeature] = static_cast<SharedStorageDataType>(iByteCur);
      }

      return iByteCur;
   }

return_bad:;

   if(nullptr != pFillMem) {
      HeaderDataSetShared * pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
      pHeaderDataSetShared->m_id = k_sharedDataSetErrorId;
   }
   return 0;
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION SizeDataSetHeader(IntEbmType countFeatures) {
   const size_t cBytes = AppendHeader(countFeatures, 0, nullptr);

   if(!IsNumberConvertable<IntEbmType>(cBytes)) {
      LOG_0(TraceLevelError, "ERROR SizeDataSetHeader !IsNumberConvertable<IntEbmType>(cBytes)");
      return 0;
   }

   return static_cast<IntEbmType>(cBytes);
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillDataSetHeader(
   IntEbmType countFeatures,
   IntEbmType countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(TraceLevelError, "ERROR FillDataSetHeader nullptr == fillMem");
      return Error_IllegalParamValue;
   }

   if(!IsNumberConvertable<size_t>(countBytesAllocated)) {
      LOG_0(TraceLevelError, "ERROR FillDataSetHeader countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   if(cBytesAllocated < sizeof(HeaderDataSetShared)) {
      LOG_0(TraceLevelError, "ERROR FillDataSetHeader cBytesAllocated < sizeof(HeaderDataSetShared)");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }

   const size_t cBytes = AppendHeader(countFeatures, cBytesAllocated, static_cast<char *>(fillMem));
   return size_t { 0 } == cBytes ? Error_IllegalParamValue : Error_None;
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION SizeDataSetFeature(
   BoolEbmType categorical,
   IntEbmType countBins,
   IntEbmType countSamples,
   const IntEbmType * binnedData
) {
   const size_t cBytes = AppendFeatureData(
      categorical,
      countBins,
      countSamples,
      binnedData,
      0,
      nullptr
   );

   if(!IsNumberConvertable<IntEbmType>(cBytes)) {
      LOG_0(TraceLevelError, "ERROR SizeDataSetFeature !IsNumberConvertable<IntEbmType>(cBytes)");
      return 0;
   }

   return static_cast<IntEbmType>(cBytes);
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillDataSetFeature(
   BoolEbmType categorical,
   IntEbmType countBins,
   IntEbmType countSamples,
   const IntEbmType * binnedData,
   IntEbmType countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(TraceLevelError, "ERROR FillDataSetFeature nullptr == fillMem");
      return Error_IllegalParamValue;
   }

   if(!IsNumberConvertable<size_t>(countBytesAllocated)) {
      LOG_0(TraceLevelError, "ERROR FillDataSetFeature countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   if(cBytesAllocated < sizeof(HeaderDataSetShared)) {
      LOG_0(TraceLevelError, "ERROR FillDataSetFeature cBytesAllocated < sizeof(HeaderDataSetShared)");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }

   const size_t cBytes = AppendFeatureData(
      categorical,
      countBins,
      countSamples,
      binnedData,
      cBytesAllocated,
      static_cast<char *>(fillMem)
   );
   return size_t { 0 } == cBytes ? Error_IllegalParamValue : Error_None;
}

size_t AppendTargets(
   const bool bClassification,
   const IntEbmType countTargetClasses,
   const IntEbmType countSamples,
   const void * aTargets,
   const size_t cBytesAllocated,
   char * const pFillMem
) {
   EBM_ASSERT(size_t { 0 } == cBytesAllocated && nullptr == pFillMem || nullptr != pFillMem);

   LOG_N(
      TraceLevelInfo,
      "Entered AppendTargets: "
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
      if(!IsNumberConvertable<size_t>(countTargetClasses) || !IsNumberConvertable<SharedStorageDataType>(countTargetClasses)) {
         LOG_0(TraceLevelError, "ERROR AppendTargets countTargetClasses is outside the range of a valid index");
         goto return_bad;
      }
      if(!IsNumberConvertable<size_t>(countSamples) || !IsNumberConvertable<SharedStorageDataType>(countSamples)) {
         LOG_0(TraceLevelError, "ERROR AppendTargets countSamples is outside the range of a valid index");
         goto return_bad;
      }
      const size_t cSamples = static_cast<size_t>(countSamples);

      size_t iByteCur = bClassification ? sizeof(TargetDataSetShared) + sizeof(ClassificationTargetDataSetShared) :
         sizeof(TargetDataSetShared);
      if(nullptr != pFillMem) {
         EBM_ASSERT(sizeof(HeaderDataSetShared) <= cBytesAllocated); // checked by our caller

         HeaderDataSetShared * const pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
         if(k_sharedDataSetId != pHeaderDataSetShared->m_id) {
            return 0; // don't set the header to bad since it's already set to something invalid and we don't know why
         }

         const SharedStorageDataType indexFeature = pHeaderDataSetShared->m_cFeatures;
         if(!IsNumberConvertable<size_t>(indexFeature)) {
            // we're being untrusting of the caller manipulating the memory improperly here
            LOG_0(TraceLevelError, "ERROR AppendTargets indexFeature is outside the range of a valid index");
            goto return_bad;
         }
         const size_t iFeature = static_cast<size_t>(indexFeature);

         const SharedStorageDataType offsetsEnd = pHeaderDataSetShared->m_offsets[0];
         if(!IsNumberConvertable<size_t>(offsetsEnd)) {
            LOG_0(TraceLevelError, "ERROR AppendTargets m_offsets[0] is outside the range of a valid index");
            goto return_bad;
         }
         const size_t iOffsetsEnd = static_cast<size_t>(offsetsEnd);

         if(cBytesAllocated <= iOffsetsEnd) {
            LOG_0(TraceLevelError, "ERROR AppendTargets cBytesAllocated <= iOffsetEndFeatures");
            goto return_bad;
         }

         if(iOffsetsEnd < sizeof(HeaderDataSetShared)) {
            LOG_0(TraceLevelError, "ERROR AppendTargets iOffsetsEnd < sizeof(HeaderDataSetShared)");
            goto return_bad;
         }
         const size_t cBytesInOffsets = iOffsetsEnd - sizeof(HeaderDataSetShared);
         if(size_t { 0 } != cBytesInOffsets % sizeof(HeaderDataSetShared::m_offsets[0])) {
            LOG_0(TraceLevelError, "ERROR AppendTargets size_t { 0 } != cBytesInOffsets % sizeof(HeaderDataSetShared::m_offsets[0])");
            goto return_bad;
         }
         const size_t cFeatures = cBytesInOffsets / sizeof(HeaderDataSetShared::m_offsets[0]);

         if(cFeatures != iFeature) {
            LOG_0(TraceLevelError, "ERROR AppendTargets cFeatures != iFeature");
            goto return_bad;
         }

         if(size_t { 0 } == iFeature) {
            if(SharedStorageDataType { 0 } != pHeaderDataSetShared->m_cSamples) {
               LOG_0(TraceLevelError, "ERROR AppendTargets SharedStorageDataType { 0 } != pHeaderDataSetShared->m_cSamples");
               goto return_bad;
            }
            pHeaderDataSetShared->m_cSamples = static_cast<SharedStorageDataType>(cSamples);
         } else {
            if(pHeaderDataSetShared->m_cSamples != static_cast<SharedStorageDataType>(cSamples)) {
               LOG_0(TraceLevelError, "ERROR AppendTargets pHeaderDataSetShared->m_cSamples != cSamples");
               goto return_bad;
            }
         }

         // this memory location is guaranteed to be below cBytesAllocated since we checked above that the
         // memory space AFTER the m_offsets array existed AND we've checked that our current iFeature doesn't
         // overflow that the m_offsets, so it stands to reason that the memory location at iFeature exists
         const SharedStorageDataType indexHighestOffset = ArrayToPointer(pHeaderDataSetShared->m_offsets)[iFeature];
         if(!IsNumberConvertable<size_t>(indexHighestOffset)) {
            LOG_0(TraceLevelError, "ERROR AppendTargets indexByteCur is outside the range of a valid index");
            goto return_bad;
         }
         const size_t iHighestOffset = static_cast<size_t>(indexHighestOffset);

         if(IsAddError(iByteCur, iHighestOffset)) {
            LOG_0(TraceLevelError, "ERROR AppendTargets IsAddError(iByteCur, iHighestOffset)");
            goto return_bad;
         }
         iByteCur += iHighestOffset; // if we're going to access FeatureDataSetShared, then check if we have the space
         if(size_t { 0 } != cSamples) {
            if(cBytesAllocated <= iByteCur) {
               LOG_0(TraceLevelError, "ERROR AppendTargets cBytesAllocated <= iByteCur");
               goto return_bad;
            }
         } else {
            // if there are zero samples, then we should be at the end already
            if(cBytesAllocated != iByteCur) {
               LOG_0(TraceLevelError, "ERROR AppendTargets cBytesAllocated != iByteCur");
               goto return_bad;
            }
         }

         char * pFillMemTemp = pFillMem + iHighestOffset;
         TargetDataSetShared * pTargetDataSetShared = reinterpret_cast<TargetDataSetShared *>(pFillMemTemp);
         pTargetDataSetShared->m_id = GetTargetId(bClassification);

         if(bClassification) {
            ClassificationTargetDataSetShared * pClassificationTargetDataSetShared = reinterpret_cast<ClassificationTargetDataSetShared *>(pFillMemTemp + sizeof(TargetDataSetShared));
            pClassificationTargetDataSetShared->m_cTargetClasses = static_cast<SharedStorageDataType>(countTargetClasses);
         }
      }

      if(size_t { 0 } != cSamples) {
         if(nullptr == aTargets) {
            LOG_0(TraceLevelError, "ERROR AppendTargets nullptr == aTargets");
            goto return_bad;
         }

         size_t cBytesAllSamples;
         if(bClassification) {
            if(IsMultiplyError(sizeof(SharedStorageDataType), cSamples)) {
               LOG_0(TraceLevelError, "ERROR AppendTargets IsMultiplyError(sizeof(SharedStorageDataType), cSamples)");
               goto return_bad;
            }
            cBytesAllSamples = sizeof(SharedStorageDataType) * cSamples;
         } else {
            if(IsMultiplyError(sizeof(FloatEbmType), cSamples)) {
               LOG_0(TraceLevelError, "ERROR AppendTargets IsMultiplyError(sizeof(FloatEbmType), cSamples)");
               goto return_bad;
            }
            cBytesAllSamples = sizeof(FloatEbmType) * cSamples;
         }
         if(IsAddError(iByteCur, cBytesAllSamples)) {
            LOG_0(TraceLevelError, "ERROR AppendTargets IsAddError(iByteCur, cBytesAllSamples)");
            goto return_bad;
         }
         const size_t iByteNext = iByteCur + cBytesAllSamples;
         if(nullptr != pFillMem) {
            // our caller needs to give us the EXACT number of bytes used, otherwise there's a dangerous bug
            if(cBytesAllocated != iByteNext) {
               LOG_0(TraceLevelError, "ERROR AppendTargets cBytesAllocated != iByteNext");
               goto return_bad;
            }
            if(bClassification) {
               const IntEbmType * pTarget = reinterpret_cast<const IntEbmType *>(aTargets);
               if(IsMultiplyError(sizeof(pTarget[0]), cSamples)) {
                  LOG_0(TraceLevelError, "ERROR AppendTargets IsMultiplyError(sizeof(SharedStorageDataType), cSamples)");
                  goto return_bad;
               }
               const IntEbmType * const pTargetsEnd = pTarget + cSamples;
               SharedStorageDataType * pFillData = reinterpret_cast<SharedStorageDataType *>(pFillMem + iByteCur);
               do {
                  const IntEbmType target = *pTarget;
                  if(target < IntEbmType { 0 }) {
                     LOG_0(TraceLevelError, "ERROR AppendTargets classification target can't be negative");
                     goto return_bad;
                  }
                  if(countTargetClasses <= target) {
                     LOG_0(TraceLevelError, "ERROR AppendTargets countTargetClasses <= target");
                     goto return_bad;
                  }
                  // since countTargetClasses can be converted to these, so now can target
                  EBM_ASSERT(IsNumberConvertable<size_t>(target));
                  EBM_ASSERT(IsNumberConvertable<SharedStorageDataType>(target));
               
                  // TODO: sort by the target and then convert the target to a count of each index
                  *pFillData = static_cast<SharedStorageDataType>(target);
               
                  ++pFillData;
                  ++pTarget;
               } while(pTargetsEnd != pTarget);
               EBM_ASSERT(reinterpret_cast<char *>(pFillData) == pFillMem + iByteNext);
            } else {
               EBM_ASSERT(!IsMultiplyError(sizeof(FloatEbmType), cSamples)); // checked above
               memcpy(pFillMem + iByteCur, aTargets, cBytesAllSamples);
            }
         }
         iByteCur = iByteNext;
      }

      return iByteCur;
   }

return_bad:;

   if(nullptr != pFillMem) {
      HeaderDataSetShared * pHeaderDataSetShared = reinterpret_cast<HeaderDataSetShared *>(pFillMem);
      pHeaderDataSetShared->m_id = k_sharedDataSetErrorId;
   }
   return 0;
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION SizeClassificationTargets(
   IntEbmType countTargetClasses,
   IntEbmType countSamples,
   const IntEbmType * targets
) {
   const size_t cBytes = AppendTargets(
      true,
      countTargetClasses,
      countSamples,
      targets,
      0,
      nullptr
   );

   if(!IsNumberConvertable<IntEbmType>(cBytes)) {
      LOG_0(TraceLevelError, "ERROR SizeClassificationTargets !IsNumberConvertable<IntEbmType>(cBytes)");
      return 0;
   }

   return static_cast<IntEbmType>(cBytes);
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillClassificationTargets(
   IntEbmType countTargetClasses,
   IntEbmType countSamples,
   const IntEbmType * targets,
   IntEbmType countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(TraceLevelError, "ERROR FillClassificationTargets nullptr == fillMem");
      return Error_IllegalParamValue;
   }

   if(!IsNumberConvertable<size_t>(countBytesAllocated)) {
      LOG_0(TraceLevelError, "ERROR FillClassificationTargets countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   if(cBytesAllocated < sizeof(HeaderDataSetShared)) {
      LOG_0(TraceLevelError, "ERROR FillClassificationTargets cBytesAllocated < sizeof(HeaderDataSetShared)");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }

   const size_t cBytes = AppendTargets(
      true,
      countTargetClasses,
      countSamples,
      targets,
      cBytesAllocated,
      static_cast<char *>(fillMem)
   );
   return size_t { 0 } == cBytes ? Error_IllegalParamValue : Error_None;
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION SizeRegressionTargets(
   IntEbmType countSamples,
   const FloatEbmType * targets
) {
   const size_t cBytes = AppendTargets(
      false,
      0,
      countSamples,
      targets,
      0,
      nullptr
   );

   if(!IsNumberConvertable<IntEbmType>(cBytes)) {
      LOG_0(TraceLevelError, "ERROR SizeRegressionTargets !IsNumberConvertable<IntEbmType>(cBytes)");
      return 0;
   }

   return static_cast<IntEbmType>(cBytes);
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillRegressionTargets(
   IntEbmType countSamples,
   const FloatEbmType * targets,
   IntEbmType countBytesAllocated,
   void * fillMem
) {
   if(nullptr == fillMem) {
      LOG_0(TraceLevelError, "ERROR FillRegressionTargets nullptr == fillMem");
      return Error_IllegalParamValue;
   }

   if(!IsNumberConvertable<size_t>(countBytesAllocated)) {
      LOG_0(TraceLevelError, "ERROR FillRegressionTargets countBytesAllocated is outside the range of a valid size");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }
   const size_t cBytesAllocated = static_cast<size_t>(countBytesAllocated);

   if(cBytesAllocated < sizeof(HeaderDataSetShared)) {
      LOG_0(TraceLevelError, "ERROR FillRegressionTargets cBytesAllocated < sizeof(HeaderDataSetShared)");
      // don't set the header to bad if we don't have enough memory for the header itself
      return Error_IllegalParamValue;
   }

   const size_t cBytes = AppendTargets(
      false,
      0,
      countSamples,
      targets,
      cBytesAllocated,
      static_cast<char *>(fillMem)
   );
   return size_t { 0 } == cBytes ? Error_IllegalParamValue : Error_None;
}




} // DEFINED_ZONE_NAME
