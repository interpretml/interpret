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

class CachedBoostingThreadResources final {
   // TODO: can I preallocate m_aThreadByteBuffer1 and m_aThreadByteBuffer2 without resorting to grow them if I examine my inputs

   void * m_aThreadByteBuffer1;
   size_t m_cThreadByteBufferCapacity1;

   void * m_aThreadByteBuffer2;
   size_t m_cThreadByteBufferCapacity2;

   FloatEbmType * m_aTempFloatVector;
   void * m_aEquivalentSplits; // we use different structures for mains and multidimension and between classification and regression

   void * m_aSumHistogramBucketVectorEntry;
   void * m_aSumHistogramBucketVectorEntry1;

public:

   INLINE_RELEASE static void Free(CachedBoostingThreadResources * const pCachedResources) {
      LOG_0(TraceLevelInfo, "Entered CachedBoostingThreadResources::Free");

      if(nullptr != pCachedResources) {
         free(pCachedResources->m_aThreadByteBuffer1);
         free(pCachedResources->m_aThreadByteBuffer2);
         free(pCachedResources->m_aSumHistogramBucketVectorEntry);
         free(pCachedResources->m_aSumHistogramBucketVectorEntry1);
         free(pCachedResources->m_aTempFloatVector);
         free(pCachedResources->m_aEquivalentSplits);

         free(pCachedResources);
      }

      LOG_0(TraceLevelInfo, "Exited CachedBoostingThreadResources::Free");
   }

   INLINE_RELEASE static CachedBoostingThreadResources * Allocate(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t cBytesArrayEquivalentSplitMax
   ) {
      LOG_0(TraceLevelInfo, "Entered CachedBoostingThreadResources::Allocate");

      CachedBoostingThreadResources * const pNew = EbmMalloc<CachedBoostingThreadResources, true>();
      if(LIKELY(nullptr != pNew)) {
         const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
         const size_t cBytesPerItem = IsClassification(runtimeLearningTypeOrCountTargetClasses) ?
            sizeof(HistogramBucketVectorEntry<true>) : sizeof(HistogramBucketVectorEntry<false>);

         void * const aSumHistogramBucketVectorEntry = EbmMalloc<void, false>(cVectorLength, cBytesPerItem);
         if(LIKELY(nullptr != aSumHistogramBucketVectorEntry)) {
            pNew->m_aSumHistogramBucketVectorEntry = aSumHistogramBucketVectorEntry;
            void * const aSumHistogramBucketVectorEntry1 = EbmMalloc<void, false>(cVectorLength, cBytesPerItem);
            if(LIKELY(nullptr != aSumHistogramBucketVectorEntry1)) {
               pNew->m_aSumHistogramBucketVectorEntry1 = aSumHistogramBucketVectorEntry1;
               FloatEbmType * const aTempFloatVector = EbmMalloc<FloatEbmType, false>(cVectorLength);
               if(LIKELY(nullptr != aTempFloatVector)) {
                  pNew->m_aTempFloatVector = aTempFloatVector;
                  if(0 != cBytesArrayEquivalentSplitMax) {
                     void * aEquivalentSplits = EbmMalloc<void, false>(cBytesArrayEquivalentSplitMax);
                     if(UNLIKELY(nullptr == aEquivalentSplits)) {
                        goto exit_error;
                     }
                     pNew->m_aEquivalentSplits = aEquivalentSplits;
                  }

                  LOG_0(TraceLevelInfo, "Exited CachedBoostingThreadResources::Allocate");
                  return pNew;
               }
            }
         }
      exit_error:;
         Free(pNew);
      }
      LOG_0(TraceLevelWarning, "WARNING Exited CachedBoostingThreadResources::Allocate with error");
      return nullptr;
   }

   INLINE_RELEASE void * GetThreadByteBuffer1(const size_t cBytesRequired) {
      void * aBuffer = m_aThreadByteBuffer1;
      if(UNLIKELY(m_cThreadByteBufferCapacity1 < cBytesRequired)) {
         m_cThreadByteBufferCapacity1 = cBytesRequired << 1;
         LOG_N(TraceLevelInfo, "Growing CachedBoostingThreadResources::ThreadByteBuffer1 to %zu", m_cThreadByteBufferCapacity1);

         free(aBuffer);
         aBuffer = EbmMalloc<void, false>(m_cThreadByteBufferCapacity1);
         m_aThreadByteBuffer1 = aBuffer;
      }
      return aBuffer;
   }

   INLINE_RELEASE bool GrowThreadByteBuffer2(const size_t cByteBoundaries) {
      // by adding cByteBoundaries and shifting our existing size, we do 2 things:
      //   1) we ensure that if we have zero size, we'll get some size that we'll get a non-zero size after the shift
      //   2) we'll always get back an odd number of items, which is good because we always have an odd number of TreeNodeChilden
      EBM_ASSERT(0 == m_cThreadByteBufferCapacity2 % cByteBoundaries);
      m_cThreadByteBufferCapacity2 = cByteBoundaries + (m_cThreadByteBufferCapacity2 << 1);
      LOG_N(TraceLevelInfo, "Growing CachedBoostingThreadResources::ThreadByteBuffer2 to %zu", m_cThreadByteBufferCapacity2);

      // our tree objects have internal pointers, so we're going to dispose of our work anyways
      // We can't use realloc since there is no way to check if the array was re-allocated or not without 
      // invoking undefined behavior, so we don't get a benefit if the array can be resized with realloc

      void * aBuffer = m_aThreadByteBuffer2;
      free(aBuffer);
      aBuffer = EbmMalloc<void, false>(m_cThreadByteBufferCapacity2);
      m_aThreadByteBuffer2 = aBuffer;
      if(UNLIKELY(nullptr == aBuffer)) {
         return true;
      }
      return false;
   }

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

   template<bool bClassification>
   EBM_INLINE HistogramBucketVectorEntry<bClassification> * GetSumHistogramBucketVectorEntryArray() {
      return static_cast<HistogramBucketVectorEntry<bClassification> *>(m_aSumHistogramBucketVectorEntry);
   }

   template<bool bClassification>
   EBM_INLINE HistogramBucketVectorEntry<bClassification> * GetSumHistogramBucketVectorEntry1Array() {
      return static_cast<HistogramBucketVectorEntry<bClassification> *>(m_aSumHistogramBucketVectorEntry1);
   }
};
static_assert(std::is_standard_layout<CachedBoostingThreadResources>::value,
   "we use malloc to allocate this, so it needs to be standard layout");

#endif // CACHED_BOOSTING_THREAD_RESOURCES_H
