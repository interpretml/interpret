// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef HISTOGRAM_BUCKET_H
#define HISTOGRAM_BUCKET_H

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t
#include <cmath> // abs
#include <string.h> // memcpy

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "HistogramTargetEntry.h"
#include "Feature.h"
#include "FeatureGroup.h"
#include "DataSetBoosting.h"
#include "DataSetInteraction.h"
#include "SamplingSet.h"

#include "Booster.h"
#include "InteractionDetection.h"

// we don't need to handle multi-dimensional inputs with more than 64 bits total
// the rational is that we need to bin this data, and our binning memory will be N1*N1*...*N(D-1)*N(D)
// So, even for binary input featuers, we would have 2^64 bins, and that would take more memory than a 64 bit machine can have
// Similarily, we need a huge amount of memory in order to bin any data with a combined total of more than 64 bits.
// The worst case I can think of is where we have 3 bins, requiring 2 bit each, and we overflowed at 33 dimensions
// in that bad case, we would have 3^33 bins * 8 bytes minimum per bin = 44472484532444184 bytes, which would take 56 bits to express
// Nobody is every going to build a machine with more than 64 bits, since you need a non-trivial volume of material assuming bits require
// more than several atoms to store.
// we can just return an out of memory error if someone requests a set of features that would sum to more than 64 bits
// we DO need to check for this error condition though, since it's not impossible for someone to request this kind of thing.
// any dimensions with only 1 bin don't count since you would just be multiplying by 1 for each such dimension

template<bool bClassification>
struct HistogramBucket;

struct HistogramBucketBase {
   template<bool bClassification>
   EBM_INLINE HistogramBucket<bClassification> * GetHistogramBucket() {
      return static_cast<HistogramBucket<bClassification> *>(this);
   }
   template<bool bClassification>
   EBM_INLINE const HistogramBucket<bClassification> * GetHistogramBucket() const {
      return static_cast<const HistogramBucket<bClassification> *>(this);
   }
};

template<bool bClassification>
EBM_INLINE bool GetHistogramBucketSizeOverflow(const size_t cVectorLength) {
   return IsMultiplyError(
      sizeof(HistogramBucketVectorEntry<bClassification>), cVectorLength) ? 
      true : 
      IsAddError(
         sizeof(HistogramBucket<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>), 
         sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength
      ) ? true : false;
}
template<bool bClassification>
EBM_INLINE size_t GetHistogramBucketSize(const size_t cVectorLength) {
   return sizeof(HistogramBucket<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>) + 
      sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength;
}
template<bool bClassification>
EBM_INLINE HistogramBucket<bClassification> * GetHistogramBucketByIndex(
   const size_t cBytesPerHistogramBucket, 
   HistogramBucket<bClassification> * const aHistogramBuckets, 
   const size_t iBin
) {
   // TODO : remove the use of this function anywhere performant by making the tensor calculation start with the # of bytes per histogram bucket, 
   // therefore eliminating the need to do the multiplication at the end when finding the index
   return reinterpret_cast<HistogramBucket<bClassification> *>(reinterpret_cast<char *>(aHistogramBuckets) + iBin * cBytesPerHistogramBucket);
}
template<bool bClassification>
EBM_INLINE const HistogramBucket<bClassification> * GetHistogramBucketByIndex(
   const size_t cBytesPerHistogramBucket, 
   const HistogramBucket<bClassification> * const aHistogramBuckets, 
   const size_t iBin
) {
   // TODO : remove the use of this function anywhere performant by making the tensor calculation start with the # of bytes per histogram bucket, 
   //   therefore eliminating the need to do the multiplication at the end when finding the index
   return reinterpret_cast<const HistogramBucket<bClassification> *>(reinterpret_cast<const char *>(aHistogramBuckets) + iBin * cBytesPerHistogramBucket);
}

// keep this as a MACRO so that we don't materialize any of the parameters on non-debug builds
#define ASSERT_BINNED_BUCKET_OK(MACRO_cBytesPerHistogramBucket, MACRO_pHistogramBucket, MACRO_aHistogramBucketsEnd) \
   (EBM_ASSERT(reinterpret_cast<const char *>(MACRO_pHistogramBucket) + static_cast<size_t>(MACRO_cBytesPerHistogramBucket) <= \
      reinterpret_cast<const char *>(MACRO_aHistogramBucketsEnd)))

template<bool bClassification>
struct HistogramBucket final : HistogramBucketBase {

   size_t m_cInstancesInBucket;

   // use the "struct hack" since Flexible array member method is not available in C++
   // aHistogramBucketVectorEntry must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   HistogramBucketVectorEntry<bClassification> m_aHistogramBucketVectorEntry[1];

   EBM_INLINE void Add(const HistogramBucket<bClassification> & other, const size_t cVectorLength) {
      m_cInstancesInBucket += other.m_cInstancesInBucket;
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         ArrayToPointer(m_aHistogramBucketVectorEntry)[iVector].Add(ArrayToPointer(other.m_aHistogramBucketVectorEntry)[iVector]);
      }
   }

   EBM_INLINE void Subtract(const HistogramBucket<bClassification> & other, const size_t cVectorLength) {
      m_cInstancesInBucket -= other.m_cInstancesInBucket;
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         ArrayToPointer(m_aHistogramBucketVectorEntry)[iVector].Subtract(ArrayToPointer(other.m_aHistogramBucketVectorEntry)[iVector]);
      }
   }

   EBM_INLINE void Copy(const HistogramBucket<bClassification> & other, const size_t cVectorLength) {
      EBM_ASSERT(!GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bClassification>(cVectorLength);
      memcpy(this, &other, cBytesPerHistogramBucket);
   }

   EBM_INLINE void Zero(const size_t cVectorLength) {
      m_cInstancesInBucket = size_t { 0 };
      HistogramBucketVectorEntry<bClassification> * pHistogramTargetEntry = ArrayToPointer(m_aHistogramBucketVectorEntry);
      const HistogramBucketVectorEntry<bClassification> * const pHistogramTargetEntryEnd = &pHistogramTargetEntry[cVectorLength];
      EBM_ASSERT(1 <= cVectorLength);
      do {
         pHistogramTargetEntry->Zero();
         ++pHistogramTargetEntry;
      } while(pHistogramTargetEntryEnd != pHistogramTargetEntry);

      AssertZero(cVectorLength);
   }

   EBM_INLINE void AssertZero(const size_t cVectorLength) const {
      UNUSED(cVectorLength);
#ifndef NDEBUG
      EBM_ASSERT(0 == m_cInstancesInBucket);
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         ArrayToPointer(m_aHistogramBucketVectorEntry)[iVector].AssertZero();
      }
#endif // NDEBUG
   }
};
static_assert(std::is_standard_layout<HistogramBucket<false>>::value && std::is_standard_layout<HistogramBucket<true>>::value, 
   "HistogramBucket uses the struct hack, so it needs to be standard layout class otherwise we can't depend on the layout!");

#endif // HISTOGRAM_BUCKET_H
