// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TREE_SWEEP_H
#define TREE_SWEEP_H

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // INLINE_ALWAYS
#include "logging.h" // EBM_ASSERT & LOG
#include "HistogramTargetEntry.h"

template<bool bClassification>
struct HistogramBucket;

template<bool bClassification>
struct TreeSweep final {
private:
   size_t m_cBestSamplesLeft;
   const HistogramBucket<bClassification> * m_pBestHistogramBucketEntry;

   // use the "struct hack" since Flexible array member method is not available in C++
   // m_aBestHistogramTargetEntry must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   HistogramTargetEntry<bClassification> m_aBestHistogramTargetEntry[1];

public:

   TreeSweep() = default; // preserve our POD status
   ~TreeSweep() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS size_t GetCountBestSamplesLeft() {
      return m_cBestSamplesLeft;
   }

   INLINE_ALWAYS void SetCountBestSamplesLeft(size_t cBestSamplesLeft) {
      m_cBestSamplesLeft = cBestSamplesLeft;
   }

   INLINE_ALWAYS const HistogramBucket<bClassification> * GetBestHistogramBucketEntry() {
      return m_pBestHistogramBucketEntry;
   }

   INLINE_ALWAYS void SetBestHistogramBucketEntry(const HistogramBucket<bClassification> * pBestHistogramBucketEntry) {
      m_pBestHistogramBucketEntry = pBestHistogramBucketEntry;
   }

   INLINE_ALWAYS HistogramTargetEntry<bClassification> * GetBestHistogramTargetEntry() {
      return ArrayToPointer(m_aBestHistogramTargetEntry);
   }
};
static_assert(std::is_standard_layout<TreeSweep<true>>::value && std::is_standard_layout<TreeSweep<false>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<TreeSweep<true>>::value && std::is_trivial<TreeSweep<false>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<TreeSweep<true>>::value && std::is_pod<TreeSweep<false>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

INLINE_ALWAYS bool GetTreeSweepSizeOverflow(const bool bClassification, const size_t cVectorLength) {
   const size_t cBytesHistogramTargetEntry = bClassification ?
      sizeof(HistogramTargetEntry<true>) :
      sizeof(HistogramTargetEntry<false>);

   if(UNLIKELY(IsMultiplyError(cBytesHistogramTargetEntry, cVectorLength))) {
      return true;
   }

   const size_t cBytesTreeSweepComponent = bClassification ?
      (sizeof(TreeSweep<true>) - sizeof(HistogramTargetEntry<true>)) :
      (sizeof(TreeSweep<false>) - sizeof(HistogramTargetEntry<false>));

   if(UNLIKELY(IsAddError(cBytesTreeSweepComponent, cBytesHistogramTargetEntry * cVectorLength))) {
      return true;
   }

   return false;
}

INLINE_ALWAYS size_t GetTreeSweepSize(bool bClassification, const size_t cVectorLength) {
   const size_t cBytesTreeSweepComponent = bClassification ?
      sizeof(TreeSweep<true>) - sizeof(HistogramTargetEntry<true>) :
      sizeof(TreeSweep<false>) - sizeof(HistogramTargetEntry<false>);

   const size_t cBytesHistogramTargetEntry = bClassification ?
      sizeof(HistogramTargetEntry<true>) :
      sizeof(HistogramTargetEntry<false>);

   return cBytesTreeSweepComponent + cBytesHistogramTargetEntry * cVectorLength;
}

template<bool bClassification>
INLINE_ALWAYS TreeSweep<bClassification> * AddBytesTreeSweep(TreeSweep<bClassification> * const pTreeSweep, const size_t cBytesAdd) {
   return reinterpret_cast<TreeSweep<bClassification> *>(reinterpret_cast<char *>(pTreeSweep) + cBytesAdd);
}

template<bool bClassification>
INLINE_ALWAYS size_t CountTreeSweep(
   const TreeSweep<bClassification> * const pTreeSweepStart,
   const TreeSweep<bClassification> * const pTreeSweepCur,
   const size_t cBytesPerTreeSweep
) {
   EBM_ASSERT(reinterpret_cast<const char *>(pTreeSweepStart) <= reinterpret_cast<const char *>(pTreeSweepCur));
   const size_t cBytesDiff = reinterpret_cast<const char *>(pTreeSweepCur) - reinterpret_cast<const char *>(pTreeSweepStart);
   EBM_ASSERT(0 == cBytesDiff % cBytesPerTreeSweep);
   return cBytesDiff / cBytesPerTreeSweep;
}

#endif // TREE_SWEEP_H
