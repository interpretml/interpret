// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <type_traits> // std::is_standard_layout
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "ebm_native.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // FloatFast
#include "bridge_c.h" // ActiveDataType
#include "zones.h"

#include "common_cpp.hpp" // ArrayToPointer

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class Term;

// TODO: we need to radically change this data structure so that we can efficiently pass it between machines in a 
// cluster AND within/between a GPU/CPU.  This stucture should be:
//
// IntEbm m_cBytes; // our caller can fetch the memory size of this Tensor and memcpy it over the network
//// The first thing our caller should do is call into the C++ to fix the endian nature of this struct
//// 0x3333333333333333 non-expanded, big endian
//// 0x2222222222222222 non-expanded, little endian
//// 0x1111111111111111 expanded, big endian
//// 0x0000000000000000 expanded, little endian
//// if (m_endianAndExpanded < 0x2000000000000000) bExpanded = true;
//// if (0 != (0x1 & m_endianAndExpanded)) bBigEndian = true;
// UIntEbm m_endianAndIsExpanded;
// NO m_cTensorScoreCapacity -> we have a function that calculates the maximum capacity and we allocate it all at the start
// NO m_cTensorScores -> we don't need to pass this arround from process to process since it's global info and can be passed to the individual functions
// NO m_cDimensionsMax -> we pre-determine the maximum size and always allocate the max max size
// NO m_cDimensions; -> we can pass in the Term object to know the # of dimensions
// FloatFast m_aTensorScores[]; // a space for our values
// UIntEbm DIMENSION_1_SPLIT_POINTS
// UIntEbm DIMENSION_1_BIN_COUNT -> we find this by traversing the 0th dimension items
// UIntEbm DIMENSION_0_SPLIT_POINTS -> we travel backwards by the count
// UIntEbm DIMENSION_0_BIN_COUNT -> we find this using m_cBytes to find the end, then subtract sizeof(UIntEbm)
//
// - use SLICE_COUNT instead of SPLIT_COUNT because then we can express tensors with zero bins AND we can still
//   get to the End pointer because our slices don't and split points are both 64 bit numbers, so subtracting from
//   the current pointer gets us to the next BIN_COUNT value since a bin count of 1 means zero splits, so subtract 1
//
// Reasons:
//   - our super-parallel algorithm needs to split up the data and have separate processing cores process their data
//     their outputs will be partial histograms. After a single cores does the tree buiding, that core will need to 
//     push the term score updates to all the children nodes
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
//     1) Determine the number of new split points for each dimension.  Store that info on the stack (64 entries max)
//     2) Determine the new value tensor size (with splits).  If that new tensor size PLUS the size of the split point
//        information is greater than a pure expanded tensor without splits, then expand the tensor and merge and we're 
//        done. If it's still more compact as a sliceable tensor, then continue below.
//     3) memcpy all the dimension split information to the new end of our data space to give us room
//     4) Expand and merge the values using the non-changed dimension info
//     5) Expand the dimension into upwards starting from the Nth dimension at the top and working downwards
//   - the maximum number of splits is part of the term definition, so we don't need to store that and pass 
//     that reduntant information arround. We do store the current number of splits because that changes.  This data 
//     structure should therefore have a dependency on the term definition since we'll need to read the 
//     maximum number of splits.  The pointer to the term class can be passed in via the stack to any 
//     function that needs that information
//   - use 64 bit values for all offsets, since nobody will ever need more than 64 bits 
//     (you need a non-trivial amount of mass even if you store one bit per atom) and we might pass these 
//     between 64 and 32 bit processes, but a few 64 bit offsets won't be a problem even for a 32-bit process.
//   - EXPANDING:
//     - eventually all our tensors will be expanded when doing the term score update, because we want to use array lookups 
//       instead of binary search when applying the term scores (array lookup is a huge speed boost over binary search)
//     - we can return a non-expanded term score tensor to the caller.  We will provide an expand function to expand it if the
//       caller wants to examine the tensor themselves
//     - we might as well also flip to epanded mode whenever all dimensions are fully expanded, even when we think
//       we have a compressed tensor. For things like bools or low numbers of splits this could be frequent, and the 
//       non-compressed tensor is actually smaller when all dimensions have been expanded
//     - once we've been expanded, we no longer need the split points since the split points are just incrementing integers

class Tensor final {
   struct DimensionInfoStack final {
      DimensionInfoStack() = default; // preserve our POD status
      ~DimensionInfoStack() = default; // preserve our POD status
      void * operator new(std::size_t) = delete; // we only use malloc/free in this library
      void operator delete (void *) = delete; // we only use malloc/free in this library

      const ActiveDataType * m_pSplit1;
      const ActiveDataType * m_pSplit2;
      size_t m_cNewSplits;
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

      const ActiveDataType * m_pSplit1;
      size_t m_iSplit2;
      size_t m_cNewSplits;
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

      // TODO : change m_cSplits to m_cSlices to fit the rest of our framework where we always use bin/slices, but also
      //        to represent tensors with slices that are 0 (truely empty without even 1 bin), and because we need
      //        to multiply by cSlices, when calculating the tensor volume, so we can never get away or optimize the
      //        need away for cSlicess, unlike perhaps cSplits which might in some cases allow for tricks to optimize
      size_t m_cSplits;
      ActiveDataType * m_aSplits;
      size_t m_cSplitCapacity;
   };
   static_assert(std::is_standard_layout<DimensionInfo>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
   static_assert(std::is_trivial<DimensionInfo>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");
   static_assert(std::is_pod<DimensionInfo>::value,
      "We use a lot of C constructs, so disallow non-POD types in general");

   // TODO : is this still required after we do tree splitting by pairs??
   // we always allocate our array because we don't want to Require Add(...) to check for the null pointer
   // always allocate one so that we never have to check if we have sufficient storage when we call Reset with one split and two values
   static constexpr size_t k_initialSplitCapacity = 1;
   static constexpr size_t k_initialTensorCapacity = 2;

   size_t m_cTensorScoreCapacity;
   size_t m_cScores;
   size_t m_cDimensionsMax;
   size_t m_cDimensions;
   FloatFast * m_aTensorScores;
   bool m_bExpanded;

   // IMPORTANT: m_aDimensions must be in the last position for the struct hack and this must be standard layout
   DimensionInfo m_aDimensions[1];

   inline const DimensionInfo * GetDimensions() const {
      return ArrayToPointer(m_aDimensions);
   }
   inline DimensionInfo * GetDimensions() {
      return ArrayToPointer(m_aDimensions);
   }

public:

   Tensor() = default; // preserve our POD status
   ~Tensor() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   // TODO: In the future we'll be splitting our work into small data owned by
   // a node in a distributed system.  After each node calculates it's term score update (represented by this
   // Tensor class), we'll need to reduce them accross all nodes, before adding together all the
   // Tensor classes and sending back a full update to the Nodes.  Since we'll be ferrying info
   // back and forth, we'll want to keep it in a more compressed format keeping split and not expanding
   // to a direct indexable tensor until after recieved by the nodes.  We'll NEED to keep the entire strucutre
   // as a single continuous chunk of memory.  At the very start will be our regular struct (containing the
   // full size of the data region at the top (64 bit since we don't know what processor we'll be on)
   // We know the number of dimensions for an feature group at allocation, so we can put the values right below
   // that.  When we find ourselves expanding dimensions, we can first figure out how much all the values and dimension
   // need to grow and then we can directly move each dimension pointed to object without needing to move the full
   // values array.

   static void Free(Tensor * const pTensor);
   static Tensor * Allocate(const size_t cDimensionsMax, const size_t cScores);
   void Reset();
   ErrorEbm SetCountSplits(const size_t iDimension, const size_t cSplits);
   ErrorEbm EnsureTensorScoreCapacity(const size_t cTensorScores);
   ErrorEbm Copy(const Tensor & rhs);
   bool MultiplyAndCheckForIssues(const double v);
   ErrorEbm Expand(const Term * const pTerm);
   void AddExpandedWithBadValueProtection(const FloatFast * const aFromValues);
   ErrorEbm Add(const Tensor & rhs);

#ifndef NDEBUG
   bool IsEqual(const Tensor & rhs) const;
#endif // NDEBUG

   inline bool GetExpanded() {
      return m_bExpanded;
   }

   inline void SetCountDimensions(const size_t cDimensions) {
      EBM_ASSERT(cDimensions <= m_cDimensionsMax);
      m_cDimensions = cDimensions;
   }

   inline ActiveDataType * GetSplitPointer(const size_t iDimension) {
      EBM_ASSERT(iDimension < m_cDimensions);
      return GetDimensions()[iDimension].m_aSplits;
   }

   inline size_t GetCountSplits(const size_t iDimension) {
      EBM_ASSERT(iDimension < m_cDimensions);
      return GetDimensions()[iDimension].m_cSplits;
   }

   inline FloatFast * GetTensorScoresPointer() {
      return m_aTensorScores;
   }
};
static_assert(std::is_standard_layout<Tensor>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<Tensor>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<Tensor>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // TENSOR_HPP
