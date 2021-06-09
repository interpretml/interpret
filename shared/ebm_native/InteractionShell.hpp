// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef THREAD_STATE_INTERACTION_H
#define THREAD_STATE_INTERACTION_H

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct HistogramBucketBase;

class ThreadStateInteraction final {
   HistogramBucketBase * m_aThreadByteBuffer1;
   size_t m_cThreadByteBufferCapacity1;

public:

   ThreadStateInteraction() = default; // preserve our POD status
   ~ThreadStateInteraction() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS void InitializeZero() {
      m_aThreadByteBuffer1 = nullptr;
      m_cThreadByteBufferCapacity1 = 0;
   }

   static void Free(ThreadStateInteraction * const pThreadStateInteraction);
   static ThreadStateInteraction * Allocate();
   HistogramBucketBase * GetHistogramBucketBase(const size_t cBytesRequired);

};
static_assert(std::is_standard_layout<ThreadStateInteraction>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<ThreadStateInteraction>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<ThreadStateInteraction>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // THREAD_STATE_INTERACTION_H
