// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef SEGMENTED_TENSOR_H
#define SEGMENTED_TENSOR_H

#include <type_traits> // std::is_standard_layout
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG

#include "FeatureGroup.h"

// TODO: we need to radically change this data structure so that we can efficiently pass it between machines in a 
// cluster AND within/between a GPU/CPU.  This stucture should be:
//
// IntEbmType m_cBytes; // our caller can fetch the memory size of this SegmentedTensor and memcpy it over the network
//// The first thing our caller should do is call into the C++ to fix the endian nature of this struct
//// 0x3333333333333333 non-expanded, big endian
//// 0x2222222222222222 non-expanded, little endian
//// 0x1111111111111111 expanded, big endian
//// 0x0000000000000000 expanded, little endian
//// if (m_endianAndExpanded < 0x2000000000000000) bExpanded = true;
//// if (0 != (0x1 & m_endianAndExpanded)) bBigEndian = true;
// UIntEbmType m_endianAndIsExpanded;
// NO m_cValueCapacity -> we have a function that calculates the maximum capacity and we allocate it all at the start
// NO m_cVectorLength -> we don't need to pass this arround from process to process since it's global info and can be passed to the individual functions
// NO m_cDimensionsMax -> we pre-determine the maximum size and always allocate the max max size
// NO m_cDimensions; -> we can pass in the FeatureGroup object to know the # of dimensions
// FloatEbmType m_values[]; // a space for our values
// UIntEbmType DIMENSION_1_CUT_POINTS
// UIntEbmType DIMENSION_1_BIN_COUNT -> we find this by traversing the 0th dimension items
// UIntEbmType DIMENSION_0_CUT_POINTS -> we travel backwards by the count
// UIntEbmType DIMENSION_0_BIN_COUNT -> we find this using m_cBytes to find the end, then subtract sizeof(UIntEbmType)
//
// - use BIN_COUNT instead of CUT_COUNT because then we can express tensors with zero bins AND we can still
//   get to the End pointer because our bin dount and cut points are both 64 bit numbers, so subtracting from
//   the current pointer gets us to the next BIN_COUNT value since a bin count of 1 means zero cuts, so subtract 1
//
// Reasons:
//   - our super-parallel algorithm needs to split up the data and have separate processing cores process their data
//     their outputs will be partial histograms. After a single cores does the tree buiding, that core will need to 
//     push the model updates to all the children nodes
//   - in an MPI environment, or a Spark cluster, or in a GPU, we'll need to pass this data structure between nodes, 
//     so it will need to be memcopy-able, which means no pointers (use 64-bit offsets), and it means that the data 
//     needs to be in a single contiguous byte array
//   - we might want our external caller to allocate this memory, because perhaps we might want the MPI communication 
//     layer or other network protocol to sit outside of C++, so we want to allocate the memory just once and we need 
//     to be able to determine the memory size before allocating it.  We know ahead of time how many dimensions AND 
//     the maximum split points in all dimensions, so it's possible to pre-determine this at startup.
//   - our first data structure member is a count of the number of bytes, so any high level language can extract
//     this without knowing our structure internals.  This helps our caller to use memcpy since they know the size
//   - when two tensors are combined, we exectue the following steps:
//     1) Determine the number of new cut points for each dimension.  Store that info on the stack (64 entries max)
//     2) Determine the new value tensor size (with splits).  If that new tensor size PLUS the size of the cut point
//        information is greater than a pure expanded tensor without cuts, then expand the tensor and merge and we're 
//        done. If it's still more compact as a segmented tensor, then continue below.
//     3) memcpy all the dimension cut information to the new end of our data space to give us room
//     4) Expand and merge the values using the non-changed dimension info
//     5) Expand the dimension into upwards starting from the Nth dimension at the top and working downwards
//   - the maximum number of cuts is part of the feature_group definition, so we don't need to store that and pass 
//     that reduntant information arround. We do store the current number of cuts because that changes.  This data 
//     structure should therefore have a dependency on the feature_group definition since we'll need to read the 
//     maximum number of cuts.  The pointer to the feature_group class can be passed in via the stack to any 
//     function that needs that information
//   - use 64 bit values for all offsets, since nobody will ever need more than 64 bits 
//     (you need a non-trivial amount of mass even if you store one bit per atom) and we might pass these 
//     between 64 and 32 bit processes, but a few 64 bit offsets won't be a problem even for a 32-bit process.
//   - EXPANDING:
//     - eventually all our tensors will be expanded when doing the model update, because we want to use array lookups 
//       instead of binary search when applying the model (array lookup is a huge speed boost over binary search)
//     - we can return a non-expanded model to the caller.  We will provide an expand function to expand it if the
//       caller wants to examine the tensor themselves
//     - we might as well also flip to epanded mode whenever all dimensions are fully expanded, even when we think
//       we have a compressed model. For things like bools or low numbers of cuts this could be frequent, and the 
//       non-compressed model is actually smaller when all dimensions have been expanded
//     - once we've been expanded, we no longer need the cut points since the cut points are just incrementing integers

class SegmentedTensor final {
   struct DimensionInfoStack final {
      DimensionInfoStack() = default; // preserve our POD status
      ~DimensionInfoStack() = default; // preserve our POD status
      void * operator new(std::size_t) = delete; // we only use malloc/free in this library
      void operator delete (void *) = delete; // we only use malloc/free in this library

      const ActiveDataType * m_pDivision1;
      const ActiveDataType * m_pDivision2;
      size_t m_cNewDivisions;
   };
   static_assert(std::is_standard_layout<DimensionInfoStack>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
   static_assert(std::is_trivial<DimensionInfoStack>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");
   static_assert(std::is_pod<DimensionInfoStack>::value,
      "We use a lot of C constructs, so disallow non-POD types in general");

   struct DimensionInfoStackExpand final {
      DimensionInfoStackExpand() = default; // preserve our POD status
      ~DimensionInfoStackExpand() = default; // preserve our POD status
      void * operator new(std::size_t) = delete; // we only use malloc/free in this library
      void operator delete (void *) = delete; // we only use malloc/free in this library

      const ActiveDataType * m_pDivision1;
      size_t m_iDivision2;
      size_t m_cNewDivisions;
   };
   static_assert(std::is_standard_layout<DimensionInfoStackExpand>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
   static_assert(std::is_trivial<DimensionInfoStackExpand>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");
   static_assert(std::is_pod<DimensionInfoStackExpand>::value,
      "We use a lot of C constructs, so disallow non-POD types in general");

   struct DimensionInfo final {
      DimensionInfo() = default; // preserve our POD status
      ~DimensionInfo() = default; // preserve our POD status
      void * operator new(std::size_t) = delete; // we only use malloc/free in this library
      void operator delete (void *) = delete; // we only use malloc/free in this library

      // TODO : change m_cDivisions to m_cBins to fit the rest of our framework where we always use bin, but also
      //        to represent tensors with bins that are 0 (truely empty without even 1 bin), and because we need
      //        to multiply by cBins when calculating the tensor volume, so we can never get away or optimize the
      //        need away for cBins, unlike perhaps cCuts which might in some cases allow for tricks to optimize
      size_t m_cDivisions;
      ActiveDataType * m_aDivisions;
      size_t m_cDivisionCapacity;
   };
   static_assert(std::is_standard_layout<DimensionInfo>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
   static_assert(std::is_trivial<DimensionInfo>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");
   static_assert(std::is_pod<DimensionInfo>::value,
      "We use a lot of C constructs, so disallow non-POD types in general");

   // TODO : is this still required after we do tree splitting by pairs??
   // we always allocate our array because we don't want to Require Add(...) to check for the null pointer
   // always allocate one so that we never have to check if we have sufficient storage when we call Reset with one division and two values
   static constexpr size_t k_initialDivisionCapacity = 1;
   static constexpr size_t k_initialValueCapacity = 2;

   size_t m_cValueCapacity;
   size_t m_cVectorLength;
   size_t m_cDimensionsMax;
   size_t m_cDimensions;
   FloatEbmType * m_aValues;
   bool m_bExpanded;
   // use the "struct hack" since Flexible array member method is not available in C++
   // m_aDimensions must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   DimensionInfo m_aDimensions[1];

   INLINE_ALWAYS const DimensionInfo * GetDimensions() const {
      return ArrayToPointer(m_aDimensions);
   }
   INLINE_ALWAYS DimensionInfo * GetDimensions() {
      return ArrayToPointer(m_aDimensions);
   }

public:

   SegmentedTensor() = default; // preserve our POD status
   ~SegmentedTensor() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   // TODO: In the future we'll be splitting our work into small sets of residuals and logits owned by
   // a node in a distributed system.  After each node calculates it's model update (represented by this
   // SegmentedTensor class), we'll need to reduce them accross all nodes, before adding together all the
   // SegmentedTensor classes and sending back a full update to the Nodes.  Since we'll be ferrying info
   // back and forth, we'll want to keep it in a more compressed format keeping division and not expanding
   // to a direct indexable tensor until after recieved by the nodes.  We'll NEED to keep the entire strucutre
   // as a single continuous chunk of memory.  At the very start will be our regular struct (containing the
   // full size of the data region at the top (64 bit since we don't know what processor we'll be on)
   // We know the number of dimensions for an feature group at allocation, so we can put the values right below
   // that.  When we find ourselves expanding dimensions, we can first figure out how much all the values and dimension
   // need to grow and then we can directly move each dimension pointed to object without needing to move the full
   // values array.

   static void Free(SegmentedTensor * const pSegmentedRegion);
   static SegmentedTensor * Allocate(const size_t cDimensionsMax, const size_t cVectorLength);
   void Reset();
   bool SetCountDivisions(const size_t iDimension, const size_t cDivisions);
   bool EnsureValueCapacity(const size_t cValues);
   bool Copy(const SegmentedTensor & rhs);
   bool MultiplyAndCheckForIssues(const FloatEbmType v);
   bool Expand(const FeatureGroup * const pFeatureGroup);
   void AddExpandedWithBadValueProtection(const FloatEbmType * const aFromValues);
   bool Add(const SegmentedTensor & rhs);

#ifndef NDEBUG
   bool IsEqual(const SegmentedTensor & rhs) const;
#endif // NDEBUG

   INLINE_ALWAYS bool GetExpanded() {
      return m_bExpanded;
   }

   INLINE_ALWAYS void SetCountDimensions(const size_t cDimensions) {
      EBM_ASSERT(cDimensions <= m_cDimensionsMax);
      m_cDimensions = cDimensions;
   }

   INLINE_ALWAYS ActiveDataType * GetDivisionPointer(const size_t iDimension) {
      EBM_ASSERT(iDimension < m_cDimensions);
      return GetDimensions()[iDimension].m_aDivisions;
   }

   INLINE_ALWAYS size_t GetCountDivisions(const size_t iDimension) {
      EBM_ASSERT(iDimension < m_cDimensions);
      return GetDimensions()[iDimension].m_cDivisions;
   }

   INLINE_ALWAYS FloatEbmType * GetValuePointer() {
      return m_aValues;
   }
};
static_assert(std::is_standard_layout<SegmentedTensor>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<SegmentedTensor>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<SegmentedTensor>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

#endif // SEGMENTED_TENSOR_H
