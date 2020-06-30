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
#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "HistogramTargetEntry.h"
#include "FeatureAtomic.h"
#include "FeatureGroup.h"
#include "DataSetBoosting.h"
#include "DataSetInteraction.h"
#include "SamplingSet.h"

#include "Booster.h"
#include "InteractionDetection.h"

template<bool bClassification>
struct HistogramBucket;

struct HistogramBucketBase {
   HistogramBucketBase() = default; // preserve our POD status
   ~HistogramBucketBase() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   template<bool bClassification>
   INLINE_ALWAYS HistogramBucket<bClassification> * GetHistogramBucket() {
      return static_cast<HistogramBucket<bClassification> *>(this);
   }
   template<bool bClassification>
   INLINE_ALWAYS const HistogramBucket<bClassification> * GetHistogramBucket() const {
      return static_cast<const HistogramBucket<bClassification> *>(this);
   }
};
static_assert(std::is_standard_layout<HistogramBucketBase>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<HistogramBucketBase>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<HistogramBucketBase>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

template<bool bClassification>
struct HistogramBucket final : HistogramBucketBase {
   HistogramBucket() = default; // preserve our POD status
   ~HistogramBucket() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   size_t m_cInstancesInBucket;

   // use the "struct hack" since Flexible array member method is not available in C++
   // aHistogramBucketVectorEntry must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   HistogramBucketVectorEntry<bClassification> m_aHistogramBucketVectorEntry[1];

   INLINE_ALWAYS void Add(const HistogramBucket<bClassification> & other, const size_t cVectorLength) {
      m_cInstancesInBucket += other.m_cInstancesInBucket;
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         ArrayToPointer(m_aHistogramBucketVectorEntry)[iVector].Add(ArrayToPointer(other.m_aHistogramBucketVectorEntry)[iVector]);
      }
   }

   INLINE_ALWAYS void Subtract(const HistogramBucket<bClassification> & other, const size_t cVectorLength) {
      m_cInstancesInBucket -= other.m_cInstancesInBucket;
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         ArrayToPointer(m_aHistogramBucketVectorEntry)[iVector].Subtract(ArrayToPointer(other.m_aHistogramBucketVectorEntry)[iVector]);
      }
   }

   INLINE_ALWAYS void Copy(const HistogramBucket<bClassification> & other, const size_t cVectorLength) {
      const size_t cBytesPerHistogramBucket = 
         sizeof(HistogramBucket<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>) +
         sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength;

      memcpy(this, &other, cBytesPerHistogramBucket);
   }

   INLINE_ALWAYS void Zero(const size_t cVectorLength) {
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

   INLINE_ALWAYS void AssertZero(const size_t cVectorLength) const {
      UNUSED(cVectorLength);
#ifndef NDEBUG
      EBM_ASSERT(0 == m_cInstancesInBucket);
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         ArrayToPointer(m_aHistogramBucketVectorEntry)[iVector].AssertZero();
      }
#endif // NDEBUG
   }
};
static_assert(std::is_standard_layout<HistogramBucket<true>>::value && std::is_standard_layout<HistogramBucket<false>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<HistogramBucket<true>>::value && std::is_trivial<HistogramBucket<false>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<HistogramBucket<true>>::value && std::is_pod<HistogramBucket<false>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

INLINE_ALWAYS bool GetHistogramBucketSizeOverflow(const bool bClassification, const size_t cVectorLength) {
   const size_t cBytesHistogramTargetEntry = bClassification ?
      sizeof(HistogramBucketVectorEntry<true>) :
      sizeof(HistogramBucketVectorEntry<false>);

   if(UNLIKELY(IsMultiplyError(cBytesHistogramTargetEntry, cVectorLength))) {
      return true;
   }

   const size_t cBytesHistogramBucketComponent = bClassification ?
      (sizeof(HistogramBucket<true>) - sizeof(HistogramBucketVectorEntry<true>)) :
      (sizeof(HistogramBucket<false>) - sizeof(HistogramBucketVectorEntry<false>));

   if(UNLIKELY(IsAddError(cBytesHistogramBucketComponent, cBytesHistogramTargetEntry * cVectorLength))) {
      return true;
   }

   return false;
}

INLINE_ALWAYS size_t GetHistogramBucketSize(const bool bClassification, const size_t cVectorLength) {
   // TODO: someday try out bucket sizes that are a power of two.  This would allow us to use a shift when bucketing into histograms
   //       instead of using multiplications.  In that version return the number of bits to shift here to make it easy
   //       to get either the shift required for indexing OR the number of bytes (shift 1 << num_bits)

   const size_t cBytesHistogramBucketComponent = bClassification ?
      sizeof(HistogramBucket<true>) - sizeof(HistogramBucketVectorEntry<true>) :
      sizeof(HistogramBucket<false>) - sizeof(HistogramBucketVectorEntry<false>);

   const size_t cBytesHistogramTargetEntry = bClassification ?
      sizeof(HistogramBucketVectorEntry<true>) :
      sizeof(HistogramBucketVectorEntry<false>);

   return cBytesHistogramBucketComponent + cBytesHistogramTargetEntry * cVectorLength;
}

template<bool bClassification>
INLINE_ALWAYS HistogramBucket<bClassification> * GetHistogramBucketByIndex(
   const size_t cBytesPerHistogramBucket,
   HistogramBucket<bClassification> * const aHistogramBuckets,
   const size_t iBin
) {
   // TODO : remove the use of this function anywhere performant by making the tensor calculation start with the # of bytes per histogram bucket, 
   // therefore eliminating the need to do the multiplication at the end when finding the index
   return reinterpret_cast<HistogramBucket<bClassification> *>(reinterpret_cast<char *>(aHistogramBuckets) + iBin * cBytesPerHistogramBucket);
}

template<bool bClassification>
INLINE_ALWAYS const HistogramBucket<bClassification> * GetHistogramBucketByIndex(
   const size_t cBytesPerHistogramBucket,
   const HistogramBucket<bClassification> * const aHistogramBuckets,
   const size_t iBin
) {
   // TODO : remove the use of this function anywhere performant by making the tensor calculation start with the # of bytes per histogram bucket, 
   //   therefore eliminating the need to do the multiplication at the end when finding the index
   return reinterpret_cast<const HistogramBucket<bClassification> *>(reinterpret_cast<const char *>(aHistogramBuckets) + iBin * cBytesPerHistogramBucket);
}

INLINE_ALWAYS HistogramBucketBase * GetHistogramBucketByIndex(
   const size_t cBytesPerHistogramBucket,
   HistogramBucketBase * const aHistogramBuckets,
   const size_t iBin
) {
   // TODO : remove the use of this function anywhere performant by making the tensor calculation start with the # of bytes per histogram bucket, 
   //   therefore eliminating the need to do the multiplication at the end when finding the index
   return reinterpret_cast<HistogramBucketBase *>(reinterpret_cast<char *>(aHistogramBuckets) + iBin * cBytesPerHistogramBucket);
}

INLINE_ALWAYS const HistogramBucketBase * GetHistogramBucketByIndex(
   const size_t cBytesPerHistogramBucket,
   const HistogramBucketBase * const aHistogramBuckets,
   const size_t iBin
) {
   // TODO : remove the use of this function anywhere performant by making the tensor calculation start with the # of bytes per histogram bucket, 
   //   therefore eliminating the need to do the multiplication at the end when finding the index
   return reinterpret_cast<const HistogramBucketBase *>(reinterpret_cast<const char *>(aHistogramBuckets) + iBin * cBytesPerHistogramBucket);
}

// keep this as a MACRO so that we don't materialize any of the parameters on non-debug builds
#define ASSERT_BINNED_BUCKET_OK(MACRO_cBytesPerHistogramBucket, MACRO_pHistogramBucket, MACRO_aHistogramBucketsEnd) \
   (EBM_ASSERT(reinterpret_cast<const char *>(MACRO_pHistogramBucket) + static_cast<size_t>(MACRO_cBytesPerHistogramBucket) <= \
      reinterpret_cast<const char *>(MACRO_aHistogramBucketsEnd)))

#endif // HISTOGRAM_BUCKET_H
