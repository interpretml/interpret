// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef CACHED_BOOSTING_THREAD_RESOURCES_H
#define CACHED_BOOSTING_THREAD_RESOURCES_H

#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG

#include "HistogramBucketVectorEntry.h"

class CachedBoostingThreadResources {
   // TODO: can I preallocate m_aThreadByteBuffer1 and m_aThreadByteBuffer2 without resorting to grow them if I examine my inputs

   // this allows us to share the memory between underlying data types
   void * m_aThreadByteBuffer1;
   size_t m_cThreadByteBufferCapacity1;

   void * m_aThreadByteBuffer2;
   size_t m_cThreadByteBufferCapacity2;

   void * const m_aSumHistogramBucketVectorEntry;
   void * const m_aSumHistogramBucketVectorEntry1;

public:

   template<bool bClassification>
   EBM_INLINE HistogramBucketVectorEntry<bClassification> * GetSumHistogramBucketVectorEntryArray() {
      return static_cast<HistogramBucketVectorEntry<bClassification> *>(m_aSumHistogramBucketVectorEntry);
   }

   template<bool bClassification>
   EBM_INLINE HistogramBucketVectorEntry<bClassification> * GetSumHistogramBucketVectorEntry1Array() {
      return static_cast<HistogramBucketVectorEntry<bClassification> *>(m_aSumHistogramBucketVectorEntry1);
   }

   FloatEbmType * const m_aTempFloatVector;
   void * m_aEquivalentSplits; // we use different structures for mains and multidimension and between classification and regression

   CachedBoostingThreadResources(const ptrdiff_t runtimeLearningTypeOrCountTargetClasses)
      : m_aThreadByteBuffer1(nullptr)
      , m_cThreadByteBufferCapacity1(0)
      , m_aThreadByteBuffer2(nullptr)
      , m_cThreadByteBufferCapacity2(0)
      // TODO : do we need to check that the multiplication doesn't overflow here
      , m_aSumHistogramBucketVectorEntry(malloc(GetVectorLength(runtimeLearningTypeOrCountTargetClasses) * (IsClassification(runtimeLearningTypeOrCountTargetClasses) ? sizeof(HistogramBucketVectorEntry<true>) : sizeof(HistogramBucketVectorEntry<false>))))
      // TODO : do we need to check that the multiplication doesn't overflow here
      , m_aSumHistogramBucketVectorEntry1(malloc(GetVectorLength(runtimeLearningTypeOrCountTargetClasses) * (IsClassification(runtimeLearningTypeOrCountTargetClasses) ? sizeof(HistogramBucketVectorEntry<true>) : sizeof(HistogramBucketVectorEntry<false>))))
      , m_aTempFloatVector(MallocArray<FloatEbmType>(GetVectorLength(runtimeLearningTypeOrCountTargetClasses)))
      , m_aEquivalentSplits(nullptr)
   {
   }

   ~CachedBoostingThreadResources() {
      LOG_0(TraceLevelInfo, "Entered ~CachedBoostingThreadResources");

      free(m_aThreadByteBuffer1);
      free(m_aThreadByteBuffer2);
      free(m_aSumHistogramBucketVectorEntry);
      free(m_aSumHistogramBucketVectorEntry1);
      free(m_aTempFloatVector);
      free(m_aEquivalentSplits);

      LOG_0(TraceLevelInfo, "Exited ~CachedBoostingThreadResources");
   }

   EBM_INLINE void * GetThreadByteBuffer1(const size_t cBytesRequired) {
      if(UNLIKELY(m_cThreadByteBufferCapacity1 < cBytesRequired)) {
         m_cThreadByteBufferCapacity1 = cBytesRequired << 1;
         LOG_N(TraceLevelInfo, "Growing CachedBoostingThreadResources::ThreadByteBuffer1 to %zu", m_cThreadByteBufferCapacity1);
         // TODO : use malloc here instead of realloc.  We don't need to copy the data, and if we free first then we can either slot the new memory 
         // in the old slot or it can be moved
         void * const aNewThreadByteBuffer = realloc(m_aThreadByteBuffer1, m_cThreadByteBufferCapacity1);
         if(UNLIKELY(nullptr == aNewThreadByteBuffer)) {
            // according to the realloc spec, if realloc fails to allocate the new memory, it returns nullptr BUT the old memory is valid.
            // we leave m_aThreadByteBuffer1 alone in this instance and will free that memory later in the destructor
            return nullptr;
         }
         m_aThreadByteBuffer1 = aNewThreadByteBuffer;
      }
      return m_aThreadByteBuffer1;
   }

   EBM_INLINE bool GrowThreadByteBuffer2(const size_t cByteBoundaries) {
      // by adding cByteBoundaries and shifting our existing size, we do 2 things:
      //   1) we ensure that if we have zero size, we'll get some size that we'll get a non-zero size after the shift
      //   2) we'll always get back an odd number of items, which is good because we always have an odd number of TreeNodeChilden
      EBM_ASSERT(0 == m_cThreadByteBufferCapacity2 % cByteBoundaries);
      m_cThreadByteBufferCapacity2 = cByteBoundaries + (m_cThreadByteBufferCapacity2 << 1);
      LOG_N(TraceLevelInfo, "Growing CachedBoostingThreadResources::ThreadByteBuffer2 to %zu", m_cThreadByteBufferCapacity2);
      // TODO : use malloc here.  our tree objects have internal pointers, so we're going to dispose of our work anyways
      // There is no way to check if the array was re-allocated or not without invoking undefined behavior, 
      // so we don't get a benefit if the array can be resized with realloc
      void * const aNewThreadByteBuffer = realloc(m_aThreadByteBuffer2, m_cThreadByteBufferCapacity2);
      if(UNLIKELY(nullptr == aNewThreadByteBuffer)) {
         // according to the realloc spec, if realloc fails to allocate the new memory, it returns nullptr BUT the old memory is valid.
         // we leave m_aThreadByteBuffer1 alone in this instance and will free that memory later in the destructor
         return true;
      }
      m_aThreadByteBuffer2 = aNewThreadByteBuffer;
      return false;
   }

   EBM_INLINE void * GetThreadByteBuffer2() {
      return m_aThreadByteBuffer2;
   }

   EBM_INLINE size_t GetThreadByteBuffer2Size() const {
      return m_cThreadByteBufferCapacity2;
   }

   EBM_INLINE bool IsError() const {
      return nullptr == m_aSumHistogramBucketVectorEntry || 
         nullptr == m_aSumHistogramBucketVectorEntry1 || nullptr == m_aTempFloatVector;
   }
};

#endif // CACHED_BOOSTING_THREAD_RESOURCES_H
