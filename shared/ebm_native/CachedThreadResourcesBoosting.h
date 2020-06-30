// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef CACHED_BOOSTING_THREAD_RESOURCES_H
#define CACHED_BOOSTING_THREAD_RESOURCES_H

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG

#include "HistogramTargetEntry.h"

struct HistogramBucketBase;

class CachedBoostingThreadResources final {
   // TODO: can I preallocate m_aThreadByteBuffer1 and m_aThreadByteBuffer2 without resorting to grow them if I examine my inputs

   HistogramBucketBase * m_aThreadByteBuffer1;
   size_t m_cThreadByteBufferCapacity1;

   void * m_aThreadByteBuffer2;
   size_t m_cThreadByteBufferCapacity2;

   FloatEbmType * m_aTempFloatVector;
   void * m_aEquivalentSplits; // we use different structures for mains and multidimension and between classification and regression

   HistogramBucketVectorEntryBase * m_aSumHistogramBucketVectorEntry;
   HistogramBucketVectorEntryBase * m_aSumHistogramBucketVectorEntry1;

public:

   CachedBoostingThreadResources() = default; // preserve our POD status
   ~CachedBoostingThreadResources() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   EBM_INLINE void InitializeZero() {
      m_aThreadByteBuffer1 = nullptr;
      m_cThreadByteBufferCapacity1 = 0;
      m_aThreadByteBuffer2 = nullptr;
      m_cThreadByteBufferCapacity2 = 0;
      m_aTempFloatVector = nullptr;
      m_aEquivalentSplits = nullptr;
      m_aSumHistogramBucketVectorEntry = nullptr;
      m_aSumHistogramBucketVectorEntry1 = nullptr;
   }

   static void Free(CachedBoostingThreadResources * const pCachedResources);
   static CachedBoostingThreadResources * Allocate(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t cBytesArrayEquivalentSplitMax
   );
   HistogramBucketBase * GetThreadByteBuffer1(const size_t cBytesRequired);
   bool GrowThreadByteBuffer2(const size_t cByteBoundaries);

   EBM_INLINE void * GetThreadByteBuffer2() {
      return m_aThreadByteBuffer2;
   }

   EBM_INLINE size_t GetThreadByteBuffer2Size() const {
      return m_cThreadByteBufferCapacity2;
   }

   EBM_INLINE FloatEbmType * GetTempFloatVector() {
      return m_aTempFloatVector;
   }

   EBM_INLINE void * GetEquivalentSplits() {
      return m_aEquivalentSplits;
   }

   EBM_INLINE HistogramBucketVectorEntryBase * GetSumHistogramBucketVectorEntryArray() {
      return m_aSumHistogramBucketVectorEntry;
   }

   template<bool bClassification>
   EBM_INLINE HistogramBucketVectorEntry<bClassification> * GetSumHistogramBucketVectorEntry1Array() {
      return static_cast<HistogramBucketVectorEntry<bClassification> *>(m_aSumHistogramBucketVectorEntry1);
   }
};
static_assert(std::is_standard_layout<CachedBoostingThreadResources>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<CachedBoostingThreadResources>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<CachedBoostingThreadResources>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

#endif // CACHED_BOOSTING_THREAD_RESOURCES_H
