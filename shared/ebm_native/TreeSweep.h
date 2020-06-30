// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TREE_SWEEP_H
#define TREE_SWEEP_H

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "HistogramTargetEntry.h"

template<bool bClassification>
struct HistogramBucket;

template<bool bClassification>
struct SweepTreeNode final {

   SweepTreeNode() = default; // preserve our POD status
   ~SweepTreeNode() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   size_t m_cBestInstancesLeft;
   const HistogramBucket<bClassification> * m_pBestHistogramBucketEntry;

   // use the "struct hack" since Flexible array member method is not available in C++
   // m_aHistogramBucketVectorEntry must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   HistogramBucketVectorEntry<bClassification> m_aBestHistogramBucketVectorEntry[1];
};
static_assert(std::is_standard_layout<SweepTreeNode<true>>::value && std::is_standard_layout<SweepTreeNode<false>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<SweepTreeNode<true>>::value && std::is_trivial<SweepTreeNode<false>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<SweepTreeNode<true>>::value && std::is_pod<SweepTreeNode<false>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

INLINE_ALWAYS bool GetSweepTreeNodeSizeOverflow(const bool bClassification, const size_t cVectorLength) {
   const size_t cBytesHistogramTargetEntry = bClassification ?
      sizeof(HistogramBucketVectorEntry<true>) :
      sizeof(HistogramBucketVectorEntry<false>);

   if(UNLIKELY(IsMultiplyError(cBytesHistogramTargetEntry, cVectorLength))) {
      return true;
   }

   const size_t cBytesTreeSweepComponent = bClassification ?
      (sizeof(SweepTreeNode<true>) - sizeof(HistogramBucketVectorEntry<true>)) :
      (sizeof(SweepTreeNode<false>) - sizeof(HistogramBucketVectorEntry<false>));

   if(UNLIKELY(IsAddError(cBytesTreeSweepComponent, cBytesHistogramTargetEntry * cVectorLength))) {
      return true;
   }

   return false;
}

INLINE_ALWAYS size_t GetSweepTreeNodeSize(bool bClassification, const size_t cVectorLength) {
   const size_t cBytesTreeSweepComponent = bClassification ?
      sizeof(SweepTreeNode<true>) - sizeof(HistogramBucketVectorEntry<true>) :
      sizeof(SweepTreeNode<false>) - sizeof(HistogramBucketVectorEntry<false>);

   const size_t cBytesHistogramTargetEntry = bClassification ?
      sizeof(HistogramBucketVectorEntry<true>) :
      sizeof(HistogramBucketVectorEntry<false>);

   return cBytesTreeSweepComponent + cBytesHistogramTargetEntry * cVectorLength;
}

template<bool bClassification>
INLINE_ALWAYS SweepTreeNode<bClassification> * AddBytesSweepTreeNode(SweepTreeNode<bClassification> * const pSweepTreeNode, const size_t cBytesAdd) {
   return reinterpret_cast<SweepTreeNode<bClassification> *>(reinterpret_cast<char *>(pSweepTreeNode) + cBytesAdd);
}

template<bool bClassification>
INLINE_ALWAYS size_t CountSweepTreeNode(
   const SweepTreeNode<bClassification> * const pSweepTreeNodeStart,
   const SweepTreeNode<bClassification> * const pSweepTreeNodeCur,
   const size_t cBytesPerSweepTreeNode
) {
   EBM_ASSERT(reinterpret_cast<const char *>(pSweepTreeNodeStart) <= reinterpret_cast<const char *>(pSweepTreeNodeCur));
   const size_t cBytesDiff = reinterpret_cast<const char *>(pSweepTreeNodeCur) - reinterpret_cast<const char *>(pSweepTreeNodeStart);
   EBM_ASSERT(0 == cBytesDiff % cBytesPerSweepTreeNode);
   return cBytesDiff / cBytesPerSweepTreeNode;
}

#endif // TREE_SWEEP_H
