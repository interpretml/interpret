// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "EbmInternal.h"

#include "ThreadStateInteraction.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

void ThreadStateInteraction::Free(ThreadStateInteraction * const pThreadStateInteraction) {
   LOG_0(TraceLevelInfo, "Entered ThreadStateInteraction::Free");

   free(pThreadStateInteraction->m_aThreadByteBuffer1);

   free(pThreadStateInteraction);

   LOG_0(TraceLevelInfo, "Exited ThreadStateInteraction::Free");
}

ThreadStateInteraction * ThreadStateInteraction::Allocate() {
   LOG_0(TraceLevelInfo, "Entered ThreadStateInteraction::Allocate");

   ThreadStateInteraction * const pNew = EbmMalloc<ThreadStateInteraction>();
   if(nullptr != pNew) {
      pNew->InitializeZero();
   }

   LOG_0(TraceLevelInfo, "Exited ThreadStateInteraction::Allocate");

   return pNew;
}

HistogramBucketBase * ThreadStateInteraction::GetHistogramBucketBase(const size_t cBytesRequired) {
   HistogramBucketBase * aBuffer = m_aThreadByteBuffer1;
   if(UNLIKELY(m_cThreadByteBufferCapacity1 < cBytesRequired)) {
      m_cThreadByteBufferCapacity1 = cBytesRequired << 1;
      LOG_N(TraceLevelInfo, "Growing ThreadStateInteraction::ThreadByteBuffer1 to %zu", m_cThreadByteBufferCapacity1);

      free(aBuffer);
      aBuffer = static_cast<HistogramBucketBase *>(EbmMalloc<void>(m_cThreadByteBufferCapacity1));
      m_aThreadByteBuffer1 = aBuffer;
   }
   return aBuffer;
}

} // DEFINED_ZONE_NAME
