// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef THREAD_STATE_BOOSTING_H
#define THREAD_STATE_BOOSTING_H

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "EbmInternal.h"

#include "HistogramTargetEntry.h"
#include "Booster.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct HistogramBucketBase;

class ThreadStateBoosting final {

   Booster * m_pBooster;
   size_t m_iFeatureGroup;

   SegmentedTensor * m_pSmallChangeToModelAccumulatedFromSamplingSets;
   SegmentedTensor * m_pSmallChangeToModelOverwriteSingleSamplingSet;

   // TODO: can I preallocate m_aThreadByteBuffer1 and m_aThreadByteBuffer2 without resorting to grow them if I examine my inputs

   HistogramBucketBase * m_aThreadByteBuffer1;
   size_t m_cThreadByteBufferCapacity1;

   void * m_aThreadByteBuffer2;
   size_t m_cThreadByteBufferCapacity2;

   FloatEbmType * m_aTempFloatVector;
   void * m_aEquivalentSplits; // we use different structures for mains and multidimension and between classification and regression

   HistogramTargetEntryBase * m_aSumHistogramTargetEntry;
   HistogramTargetEntryBase * m_aSumHistogramTargetEntry1;
   HistogramTargetEntryBase * m_aSumHistogramTargetEntry2;

#ifndef NDEBUG
   const unsigned char * m_aHistogramBucketsEndDebug;
#endif // NDEBUG

public:

   ThreadStateBoosting() = default; // preserve our POD status
   ~ThreadStateBoosting() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   constexpr static size_t k_illegalFeatureGroupIndex = size_t { static_cast<size_t>(ptrdiff_t { -1 }) };

   INLINE_ALWAYS void InitializeZero() {
      m_pBooster = nullptr;
      m_iFeatureGroup = k_illegalFeatureGroupIndex;
      m_pSmallChangeToModelAccumulatedFromSamplingSets = nullptr;
      m_pSmallChangeToModelOverwriteSingleSamplingSet = nullptr;
      m_aThreadByteBuffer1 = nullptr;
      m_cThreadByteBufferCapacity1 = 0;
      m_aThreadByteBuffer2 = nullptr;
      m_cThreadByteBufferCapacity2 = 0;
      m_aTempFloatVector = nullptr;
      m_aEquivalentSplits = nullptr;
      m_aSumHistogramTargetEntry = nullptr;
      m_aSumHistogramTargetEntry1 = nullptr;
      m_aSumHistogramTargetEntry2 = nullptr;
   }

   static void Free(ThreadStateBoosting * const pThreadStateBoosting);
   static ThreadStateBoosting * Allocate(Booster * const pBooster);

   INLINE_ALWAYS Booster * GetBooster() {
      return m_pBooster;
   }

   INLINE_ALWAYS size_t GetFeatureGroupIndex() {
      return m_iFeatureGroup;
   }

   INLINE_ALWAYS void SetFeatureGroupIndex(const size_t val) {
      m_iFeatureGroup = val;
   }

   INLINE_ALWAYS SegmentedTensor * GetAccumulatedModelUpdate() {
      return m_pSmallChangeToModelAccumulatedFromSamplingSets;
   }

   INLINE_ALWAYS SegmentedTensor * GetOverwritableModelUpdate() {
      return m_pSmallChangeToModelOverwriteSingleSamplingSet;
   }

   HistogramBucketBase * GetHistogramBucketBase(const size_t cBytesRequired);

   INLINE_ALWAYS HistogramBucketBase * GetHistogramBucketBase() {
      // call this if the histograms were already allocated and we just need the pointer
      return m_aThreadByteBuffer1;
   }

   bool GrowThreadByteBuffer2(const size_t cByteBoundaries);

   INLINE_ALWAYS void * GetThreadByteBuffer2() {
      return m_aThreadByteBuffer2;
   }

   INLINE_ALWAYS size_t GetThreadByteBuffer2Size() const {
      return m_cThreadByteBufferCapacity2;
   }

   INLINE_ALWAYS FloatEbmType * GetTempFloatVector() {
      return m_aTempFloatVector;
   }

   INLINE_ALWAYS void * GetEquivalentSplits() {
      return m_aEquivalentSplits;
   }

   INLINE_ALWAYS HistogramTargetEntryBase * GetSumHistogramTargetEntryArray() {
      return m_aSumHistogramTargetEntry;
   }

   template<bool bClassification>
   INLINE_ALWAYS HistogramTargetEntry<bClassification> * GetSumHistogramTargetEntry1Array() {
      return static_cast<HistogramTargetEntry<bClassification> *>(m_aSumHistogramTargetEntry1);
   }

   template<bool bClassification>
   INLINE_ALWAYS HistogramTargetEntry<bClassification> * GetSumHistogramTargetEntry2Array() {
      return static_cast<HistogramTargetEntry<bClassification> *>(m_aSumHistogramTargetEntry2);
   }

#ifndef NDEBUG
   INLINE_ALWAYS const unsigned char * GetHistogramBucketsEndDebug() const {
      return m_aHistogramBucketsEndDebug;
   }

   INLINE_ALWAYS void SetHistogramBucketsEndDebug(const unsigned char * const val) {
      m_aHistogramBucketsEndDebug = val;
   }
#endif // NDEBUG
};
static_assert(std::is_standard_layout<ThreadStateBoosting>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<ThreadStateBoosting>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<ThreadStateBoosting>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // THREAD_STATE_BOOSTING_H
