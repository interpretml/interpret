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

class CachedBoostingThreadResourcesPrivate final {
   friend struct CachedBoostingThreadResources;

   // TODO: can I preallocate m_aThreadByteBuffer1 and m_aThreadByteBuffer2 without resorting to grow them if I examine my inputs

   // this allows us to share the memory between underlying data types
   void * aThreadByteBuffer1;
   size_t cThreadByteBufferCapacity1;

   void * aThreadByteBuffer2;
   size_t cThreadByteBufferCapacity2;

   FloatEbmType * aTempFloatVector;
   void * aEquivalentSplits; // we use different structures for mains and multidimension and between classification and regression

   void * aSumHistogramBucketVectorEntry;
   void * aSumHistogramBucketVectorEntry1;
};

struct CachedBoostingThreadResources final {
   CachedBoostingThreadResourcesPrivate m;

   INLINE_RELEASE void Free() {
      LOG_0(TraceLevelInfo, "Entered CachedBoostingThreadResources::Free");

      free(m.aThreadByteBuffer1);
      free(m.aThreadByteBuffer2);
      free(m.aSumHistogramBucketVectorEntry);
      free(m.aSumHistogramBucketVectorEntry1);
      free(m.aTempFloatVector);
      free(m.aEquivalentSplits);

      free(this);

      LOG_0(TraceLevelInfo, "Exited CachedBoostingThreadResources::Free");
   }

   INLINE_RELEASE static CachedBoostingThreadResources * Allocate(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const size_t cBytesArrayEquivalentSplitMax
   ) {
      LOG_0(TraceLevelInfo, "Entered CachedBoostingThreadResources::Allocate");

      CachedBoostingThreadResources * const pNew = EbmMalloc<CachedBoostingThreadResources>();
      if(LIKELY(nullptr != pNew)) {
         const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
         const size_t cBytesPerItem = IsClassification(runtimeLearningTypeOrCountTargetClasses) ?
            sizeof(HistogramBucketVectorEntry<true>) : sizeof(HistogramBucketVectorEntry<false>);

         void * const aSumHistogramBucketVectorEntry = EbmMalloc<void>(cVectorLength, cBytesPerItem);
         if(LIKELY(nullptr != aSumHistogramBucketVectorEntry)) {
            pNew->m.aSumHistogramBucketVectorEntry = aSumHistogramBucketVectorEntry;
            void * const aSumHistogramBucketVectorEntry1 = EbmMalloc<void>(cVectorLength, cBytesPerItem);
            if(LIKELY(nullptr != aSumHistogramBucketVectorEntry1)) {
               pNew->m.aSumHistogramBucketVectorEntry1 = aSumHistogramBucketVectorEntry1;
               FloatEbmType * const aTempFloatVector = EbmMalloc<FloatEbmType>(cVectorLength);
               if(LIKELY(nullptr != aTempFloatVector)) {
                  pNew->m.aTempFloatVector = aTempFloatVector;
                  if(0 != cBytesArrayEquivalentSplitMax) {
                     void * aEquivalentSplits = malloc(cBytesArrayEquivalentSplitMax);
                     if(UNLIKELY(nullptr == aEquivalentSplits)) {
                        goto exit_error;
                     }
                     pNew->m.aEquivalentSplits = aEquivalentSplits;
                  }

                  LOG_0(TraceLevelInfo, "Exited CachedBoostingThreadResources::Allocate");
                  return pNew;
               }
            }
         }
      exit_error:;
         pNew->Free();
      }
      LOG_0(TraceLevelWarning, "WARNING Exited CachedBoostingThreadResources::Allocate with error");
      return nullptr;
   }

   INLINE_RELEASE void * GetThreadByteBuffer1(const size_t cBytesRequired) {
      if(UNLIKELY(m.cThreadByteBufferCapacity1 < cBytesRequired)) {
         m.cThreadByteBufferCapacity1 = cBytesRequired << 1;
         LOG_N(TraceLevelInfo, "Growing CachedBoostingThreadResources::ThreadByteBuffer1 to %zu", m.cThreadByteBufferCapacity1);
         // TODO : use malloc here instead of realloc.  We don't need to copy the data, and if we free first then we can either slot the new memory 
         // in the old slot or it can be moved
         void * const aNewThreadByteBuffer = realloc(m.aThreadByteBuffer1, m.cThreadByteBufferCapacity1);
         if(UNLIKELY(nullptr == aNewThreadByteBuffer)) {
            // according to the realloc spec, if realloc fails to allocate the new memory, it returns nullptr BUT the old memory is valid.
            // we leave m.aThreadByteBuffer1 alone in this instance and will free that memory later in the destructor
            return nullptr;
         }
         m.aThreadByteBuffer1 = aNewThreadByteBuffer;
      }
      return m.aThreadByteBuffer1;
   }

   INLINE_RELEASE bool GrowThreadByteBuffer2(const size_t cByteBoundaries) {
      // by adding cByteBoundaries and shifting our existing size, we do 2 things:
      //   1) we ensure that if we have zero size, we'll get some size that we'll get a non-zero size after the shift
      //   2) we'll always get back an odd number of items, which is good because we always have an odd number of TreeNodeChilden
      EBM_ASSERT(0 == m.cThreadByteBufferCapacity2 % cByteBoundaries);
      m.cThreadByteBufferCapacity2 = cByteBoundaries + (m.cThreadByteBufferCapacity2 << 1);
      LOG_N(TraceLevelInfo, "Growing CachedBoostingThreadResources::ThreadByteBuffer2 to %zu", m.cThreadByteBufferCapacity2);
      // TODO : use malloc here.  our tree objects have internal pointers, so we're going to dispose of our work anyways
      // There is no way to check if the array was re-allocated or not without invoking undefined behavior, 
      // so we don't get a benefit if the array can be resized with realloc
      void * const aNewThreadByteBuffer = realloc(m.aThreadByteBuffer2, m.cThreadByteBufferCapacity2);
      if(UNLIKELY(nullptr == aNewThreadByteBuffer)) {
         // according to the realloc spec, if realloc fails to allocate the new memory, it returns nullptr BUT the old memory is valid.
         // we leave m.aThreadByteBuffer1 alone in this instance and will free that memory later in the destructor
         return true;
      }
      m.aThreadByteBuffer2 = aNewThreadByteBuffer;
      return false;
   }

   EBM_INLINE void * GetThreadByteBuffer2() {
      return m.aThreadByteBuffer2;
   }

   EBM_INLINE size_t GetThreadByteBuffer2Size() const {
      return m.cThreadByteBufferCapacity2;
   }

   EBM_INLINE FloatEbmType * GetTempFloatVector() {
      return m.aTempFloatVector;
   }

   EBM_INLINE void * GetEquivalentSplits() {
      return m.aEquivalentSplits;
   }

   template<bool bClassification>
   EBM_INLINE HistogramBucketVectorEntry<bClassification> * GetSumHistogramBucketVectorEntryArray() {
      return static_cast<HistogramBucketVectorEntry<bClassification> *>(m.aSumHistogramBucketVectorEntry);
   }

   template<bool bClassification>
   EBM_INLINE HistogramBucketVectorEntry<bClassification> * GetSumHistogramBucketVectorEntry1Array() {
      return static_cast<HistogramBucketVectorEntry<bClassification> *>(m.aSumHistogramBucketVectorEntry1);
   }
};
static_assert(std::is_standard_layout<CachedBoostingThreadResources>::value,
   "we use malloc to allocate this, so it needs to be standard layout");

#endif // CACHED_BOOSTING_THREAD_RESOURCES_H
