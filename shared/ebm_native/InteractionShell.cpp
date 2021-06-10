// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "InteractionShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

void InteractionShell::Free(InteractionShell * const pInteractionShell) {
   LOG_0(TraceLevelInfo, "Entered InteractionShell::Free");

   free(pInteractionShell->m_aThreadByteBuffer1);

   free(pInteractionShell);

   LOG_0(TraceLevelInfo, "Exited InteractionShell::Free");
}

InteractionShell * InteractionShell::Allocate() {
   LOG_0(TraceLevelInfo, "Entered InteractionShell::Allocate");

   InteractionShell * const pNew = EbmMalloc<InteractionShell>();
   if(nullptr != pNew) {
      pNew->InitializeZero();
   }

   LOG_0(TraceLevelInfo, "Exited InteractionShell::Allocate");

   return pNew;
}

HistogramBucketBase * InteractionShell::GetHistogramBucketBase(const size_t cBytesRequired) {
   HistogramBucketBase * aBuffer = m_aThreadByteBuffer1;
   if(UNLIKELY(m_cThreadByteBufferCapacity1 < cBytesRequired)) {
      m_cThreadByteBufferCapacity1 = cBytesRequired << 1;
      LOG_N(TraceLevelInfo, "Growing InteractionShell::ThreadByteBuffer1 to %zu", m_cThreadByteBufferCapacity1);

      free(aBuffer);
      aBuffer = static_cast<HistogramBucketBase *>(EbmMalloc<void>(m_cThreadByteBufferCapacity1));
      m_aThreadByteBuffer1 = aBuffer;
   }
   return aBuffer;
}

} // DEFINED_ZONE_NAME
