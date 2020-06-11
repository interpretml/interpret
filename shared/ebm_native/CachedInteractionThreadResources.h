// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef CACHED_INTERACTION_THREAD_RESOURCES_H
#define CACHED_INTERACTION_THREAD_RESOURCES_H

#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG

class CachedInteractionThreadResourcesPrivate final {
   friend struct CachedInteractionThreadResources;

   // this allows us to share the memory between underlying data types
   void * aThreadByteBuffer1;
   size_t cThreadByteBufferCapacity1;
};

struct CachedInteractionThreadResources final {
   CachedInteractionThreadResourcesPrivate m;

   INLINE_RELEASE void Free() {
      LOG_0(TraceLevelInfo, "Entered CachedInteractionThreadResources::Free");

      free(m.aThreadByteBuffer1);

      free(this);

      LOG_0(TraceLevelInfo, "Exited CachedInteractionThreadResources::Free");
   }

   INLINE_RELEASE static CachedInteractionThreadResources * Allocate() {
      LOG_0(TraceLevelInfo, "Entered CachedInteractionThreadResources::Allocate");

      CachedInteractionThreadResources * const pNew = EbmMalloc<CachedInteractionThreadResources>();

      LOG_0(TraceLevelInfo, "Exited CachedInteractionThreadResources::Allocate");

      return pNew;
   }

   INLINE_RELEASE void * GetThreadByteBuffer1(const size_t cBytesRequired) {
      if(UNLIKELY(m.cThreadByteBufferCapacity1 < cBytesRequired)) {
         m.cThreadByteBufferCapacity1 = cBytesRequired << 1;
         LOG_N(TraceLevelInfo, "Growing CachedInteractionThreadResources::ThreadByteBuffer1 to %zu", m.cThreadByteBufferCapacity1);
         // TODO : use malloc here instead of realloc.  We don't need to copy the data, and if we free first then we can either slot the new 
         // memory in the old slot or it can be moved
         void * const aNewThreadByteBuffer = realloc(m.aThreadByteBuffer1, m.cThreadByteBufferCapacity1);
         if(UNLIKELY(nullptr == aNewThreadByteBuffer)) {
            // according to the realloc spec, if realloc fails to allocate the new memory, it returns nullptr BUT the old memory is valid.
            // we leave m_aThreadByteBuffer1 alone in this instance and will free that memory later in the destructor
            return nullptr;
         }
         m.aThreadByteBuffer1 = aNewThreadByteBuffer;
      }
      return m.aThreadByteBuffer1;
   }
};
static_assert(std::is_standard_layout<CachedInteractionThreadResources>::value,
   "we use malloc to allocate this, so it needs to be standard layout");
static_assert(sizeof(CachedInteractionThreadResourcesPrivate) == sizeof(CachedInteractionThreadResources),
   "CachedInteractionThreadResources shouldn't contain any data");

#endif // CACHED_INTERACTION_THREAD_RESOURCES_H
