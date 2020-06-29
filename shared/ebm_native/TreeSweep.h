// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TREE_SWEEP_H
#define TREE_SWEEP_H

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "HistogramTargetEntry.h"

template<bool bClassification>
struct HistogramBucket;

template<bool bClassification>
struct SweepTreeNode {
   size_t m_cBestInstancesLeft;
   const HistogramBucket<bClassification> * m_pBestHistogramBucketEntry;

   // use the "struct hack" since Flexible array member method is not available in C++
   // m_aHistogramBucketVectorEntry must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   HistogramBucketVectorEntry<bClassification> m_aBestHistogramBucketVectorEntry[1];
};
static_assert(
   std::is_standard_layout<SweepTreeNode<false>>::value && std::is_standard_layout<SweepTreeNode<true>>::value,
   "using the struct hack requires that our class have guaranteed member positions, hense it needs to be standard layout");

template<bool bClassification>
EBM_INLINE bool GetSweepTreeNodeSizeOverflow(const size_t cVectorLength) {
   return IsMultiplyError(sizeof(HistogramBucketVectorEntry<bClassification>), cVectorLength) ?
      true :
      IsAddError(sizeof(SweepTreeNode<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>),
         sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength) ? true : false;
}
template<bool bClassification>
EBM_INLINE size_t GetSweepTreeNodeSize(const size_t cVectorLength) {
   return sizeof(SweepTreeNode<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>) +
      sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength;
}
template<bool bClassification>
EBM_INLINE SweepTreeNode<bClassification> * AddBytesSweepTreeNode(SweepTreeNode<bClassification> * const pSweepTreeNode, const size_t cBytesAdd) {
   return reinterpret_cast<SweepTreeNode<bClassification> *>(reinterpret_cast<char *>(pSweepTreeNode) + cBytesAdd);
}
template<bool bClassification>
EBM_INLINE size_t CountSweepTreeNode(
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
