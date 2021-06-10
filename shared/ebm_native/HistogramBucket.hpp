// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef HISTOGRAM_BUCKET_HPP
#define HISTOGRAM_BUCKET_HPP

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t
#include <cmath> // abs
#include <string.h> // memcpy

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "HistogramTargetEntry.hpp"
#include "Feature.hpp"
#include "FeatureGroup.hpp"
#include "DataSetBoosting.hpp"
#include "DataSetInteraction.hpp"
#include "SamplingSet.hpp"

#include "BoosterCore.hpp"
#include "InteractionCore.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

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
private:

   size_t m_cSamplesInBucket;
   FloatEbmType m_weightInBucket;

   // use the "struct hack" since Flexible array member method is not available in C++
   // aHistogramTargetEntry must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   HistogramTargetEntry<bClassification> m_aHistogramTargetEntry[1];

public:

   HistogramBucket() = default; // preserve our POD status
   ~HistogramBucket() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS size_t GetCountSamplesInBucket() const {
      return m_cSamplesInBucket;
   }
   INLINE_ALWAYS void SetCountSamplesInBucket(const size_t cSamplesInBucket) {
      m_cSamplesInBucket = cSamplesInBucket;
   }

   INLINE_ALWAYS FloatEbmType GetWeightInBucket() const {
      return m_weightInBucket;
   }
   INLINE_ALWAYS void SetWeightInBucket(const FloatEbmType weightInBucket) {
      m_weightInBucket = weightInBucket;
   }

   INLINE_ALWAYS const HistogramTargetEntry<bClassification> * GetHistogramTargetEntry() const {
      return ArrayToPointer(m_aHistogramTargetEntry);
   }
   INLINE_ALWAYS HistogramTargetEntry<bClassification> * GetHistogramTargetEntry() {
      return ArrayToPointer(m_aHistogramTargetEntry);
   }

   INLINE_ALWAYS void Add(const HistogramBucket<bClassification> & other, const size_t cVectorLength) {
      m_cSamplesInBucket += other.m_cSamplesInBucket;
      m_weightInBucket += other.m_weightInBucket;

      HistogramTargetEntry<bClassification> * pHistogramBucketVectorThis = GetHistogramTargetEntry();

      const HistogramTargetEntry<bClassification> * pHistogramBucketVectorOther = 
         other.GetHistogramTargetEntry();

      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         pHistogramBucketVectorThis[iVector].Add(pHistogramBucketVectorOther[iVector]);
      }
   }

   INLINE_ALWAYS void Subtract(const HistogramBucket<bClassification> & other, const size_t cVectorLength) {
      m_cSamplesInBucket -= other.m_cSamplesInBucket;
      m_weightInBucket -= other.m_weightInBucket;

      HistogramTargetEntry<bClassification> * pHistogramBucketVectorThis = GetHistogramTargetEntry();

      const HistogramTargetEntry<bClassification> * pHistogramBucketVectorOther =
         other.GetHistogramTargetEntry();

      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         pHistogramBucketVectorThis[iVector].Subtract(pHistogramBucketVectorOther[iVector]);
      }
   }

   INLINE_ALWAYS void Copy(const HistogramBucket<bClassification> & other, const size_t cVectorLength) {
      const size_t cBytesPerHistogramBucket = 
         sizeof(HistogramBucket<bClassification>) - sizeof(HistogramTargetEntry<bClassification>) +
         sizeof(HistogramTargetEntry<bClassification>) * cVectorLength;

      memcpy(this, &other, cBytesPerHistogramBucket);
   }

   INLINE_ALWAYS void Zero(const size_t cVectorLength) {
      // TODO: make this a function that can operate on an array of HistogramBucket objects with given total size 
      //
      // probably we should get rid of this function, and any others that zero via non-memset ways.  We should use memset instead.  Traditionally
      // C/C++ only guaranteed that memset would lead to zeroed integers, but I think size_t would also quality
      // (check this), and if we have IEEE 754 floats (which we can check), then zeroed memory is a zeroed float

      // C standard guarantees that zeroing integer types (size_t) is a zero, and IEEE 754 guarantees 
      // that zeroing a floating point is zero.  Our HistogramBucket objects are POD and also only contain
      // floating point types and size_t
      //
      // 6.2.6.2 Integer types -> 5. The values of any padding bits are unspecified.A valid (non - trap) 
      // object representation of a signed integer type where the sign bit is zero is a valid object 
      // representation of the corresponding unsigned type, and shall represent the same value.For any 
      // integer type, the object representation where all the bits are zero shall be a representation 
      // of the value zero in that type.
      //
      // static_assert(std::numeric_limits<float>::is_iec559, "memset of floats requires IEEE 754 to guarantee zeros");
      // memset(some_pointer, 0, my_size);


      m_cSamplesInBucket = size_t { 0 };
      m_weightInBucket = size_t { 0 };
      HistogramTargetEntry<bClassification> * pHistogramTargetEntry = GetHistogramTargetEntry();
      const HistogramTargetEntry<bClassification> * const pHistogramTargetEntryEnd = &pHistogramTargetEntry[cVectorLength];
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
      EBM_ASSERT(0 == m_cSamplesInBucket);
      EBM_ASSERT(0 == m_weightInBucket);

      const HistogramTargetEntry<bClassification> * pHistogramBucketVector = GetHistogramTargetEntry();

      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         pHistogramBucketVector[iVector].AssertZero();
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
      sizeof(HistogramTargetEntry<true>) :
      sizeof(HistogramTargetEntry<false>);

   if(UNLIKELY(IsMultiplyError(cBytesHistogramTargetEntry, cVectorLength))) {
      return true;
   }

   const size_t cBytesHistogramBucketComponent = bClassification ?
      (sizeof(HistogramBucket<true>) - sizeof(HistogramTargetEntry<true>)) :
      (sizeof(HistogramBucket<false>) - sizeof(HistogramTargetEntry<false>));

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
      sizeof(HistogramBucket<true>) - sizeof(HistogramTargetEntry<true>) :
      sizeof(HistogramBucket<false>) - sizeof(HistogramTargetEntry<false>);

   const size_t cBytesHistogramTargetEntry = bClassification ?
      sizeof(HistogramTargetEntry<true>) :
      sizeof(HistogramTargetEntry<false>);

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

} // DEFINED_ZONE_NAME

#endif // HISTOGRAM_BUCKET_HPP
