// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef CACHED_INTERACTION_THREAD_RESOURCES_H
#define CACHED_INTERACTION_THREAD_RESOURCES_H

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG

class CachedInteractionThreadResources final {
   void * m_aThreadByteBuffer1;
   size_t m_cThreadByteBufferCapacity1;

public:

   INLINE_RELEASE void Free() {
      LOG_0(TraceLevelInfo, "Entered CachedInteractionThreadResources::Free");

      free(m_aThreadByteBuffer1);

      free(this);

      LOG_0(TraceLevelInfo, "Exited CachedInteractionThreadResources::Free");
   }

   INLINE_RELEASE static CachedInteractionThreadResources * Allocate() {
      LOG_0(TraceLevelInfo, "Entered CachedInteractionThreadResources::Allocate");

      CachedInteractionThreadResources * const pNew = EbmMalloc<CachedInteractionThreadResources, true>();

      LOG_0(TraceLevelInfo, "Exited CachedInteractionThreadResources::Allocate");

      return pNew;
   }

   INLINE_RELEASE void * GetThreadByteBuffer1(const size_t cBytesRequired) {
      void * aBuffer = m_aThreadByteBuffer1;
      if(UNLIKELY(m_cThreadByteBufferCapacity1 < cBytesRequired)) {
         m_cThreadByteBufferCapacity1 = cBytesRequired << 1;
         LOG_N(TraceLevelInfo, "Growing CachedInteractionThreadResources::ThreadByteBuffer1 to %zu", m_cThreadByteBufferCapacity1);

         free(aBuffer);
         aBuffer = EbmMalloc<void, false>(m_cThreadByteBufferCapacity1);
         m_aThreadByteBuffer1 = aBuffer;
      }
      return aBuffer;
   }
};
static_assert(std::is_standard_layout<CachedInteractionThreadResources>::value,
   "we use malloc to allocate this, so it needs to be standard layout");

#endif // CACHED_INTERACTION_THREAD_RESOURCES_H
