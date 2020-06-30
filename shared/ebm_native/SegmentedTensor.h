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

// TODO: we need to radically change this data structure so that we can efficiently pass it between machines in a cluster AND within/between a GPU/CPU
// This stucture should be:
//   EXTERIOR_STRUCTURE -> our external pointers point to here
//     offset_to_start_of_real_data (which is the offset_to_MAIN_DATA_STRUCTURE_in_big_endian_format)
//     size_of_real_data (from offset_to_MAIN_DATA_STRUCTURE_in_big_endian_format to the last valid value)
//   EMPTY_SPACE_FOR_DIMENSIONS_TO_GROW_INTO
//   offset_to_MAIN_DATA_STRUCTURE_in_big_endian_format -> our external caller can reduce our memory size to this point optinally to pass between memory
//     boundaries
//   DIMENSION_0_COUNT (this is the current # of cuts, not the maximum, which we store elsewhere and pass into our class for processing)
//   DIMENSION_0_CUT_POINTS
//   DIMENSION_1_COUNT (this is the current # of cuts, not the maximum, which we store elsewhere and pass into our class for processing)
//   DIMENSION_1_CUT_POINTS
//   MAIN_DATA_STRUCTURE -> our internal pointsers point to here
//     offset_to_DIMENSION_0_COUNT_or_zero_if_value_array_expanded_and_zero_offset_to_MAIN_DATA_STRUCTURE (we can find all the dimensions from this one offset)
//     count_dimensions
//     offset_to_external_structure
//     is_little_endian_bool_in_big_endian_format (little endian vs big endian)
//     all_other_data_that_does_not_resize
//   VALUES_ARRAY_FLAT_EXPANDABLE (we've preallocated enough room for this, but we don't always need it)
//   EMPTY_SPACE_FOR_VALUES_TO_GROW_INTO
// Reasons:
//   - our super-parallel algrithm needs to split up the data and have separate processing cores process their data
//     their outputs will be partial histograms.  After a single cores does the tree buiding, that core will need to push the model updates to all the
//     children nodes
//   - in an MPI environment, or a Spark cluster, or in a GPU, we'll need to pass this data structure between nodes, so it will need to be memcopy-able,
//     which means no pointers (use 64-bit offsets), and it means that the data needs to be in a single contiguous byte array
//   - we might want our external caller to allocate this memory, because perhaps we might want the MPI communication layer or other network protocol to sit 
//     outside of C++,
//     so we want to allocate the memory just once and we need to be able to determine the memory size before allocating it.
//     we know ahead of time how many dimensions AND the maximum split points in all dimensions, so it's possible to pre-determine this
//   - we use an easy/compatible external structure at the top that our caller can read without knowing our deep internals, but it provides enough 
//     information for our external caller to identify the core inner data that they can memcopy to annother memory address space
//   - so, our external caller has a non-changing memory location that indicates the internal offset where they can start copying data from and a number of 
//     bytes to copy when they want to pass this memory to annother memory boundary.  This essentially strips the useless non-used space away for compression
//   - if the caller needs to be allocated on some kind of top boundary, they can optionally start from the top and just add the 
//     offset_to_start_of_real_data and size_of_real_data together to get the size they need to copy from the top.  This will include a little
//     empty data at the top, but the empty dimension data is usually much less than the value data for interaction models, and the dimension data should 
//     almost always be pretty small, so this isn't a problem.
//   - once we recieve the interor data structure we can read offset_to_MAIN_DATA_STRUCTURE_in_big_endian_format, which will be in a machine
//     independent endian format (big endian), and we can then find our MAIN_DATA_STRUCTURE from there, and from MAIN_DATA_STRUCTURE we can reconstruct
//     our original memory size with empty space if we want, optionally
//   - the caller can pass us a pointer to the exterior data, and we can then find the offset_to_MAIN_DATA_STRUCTURE_in_big_endian_format from the header
//     then we can find our MAIN_DATA_STRUCTURE from there
//   - when we expand our tensor, we need to keep the dimension data constant and then we expand the values efficiently 
//     (starting by moving the last value to where it's eventually going to go).  After we've expanded the values, we can expand the dimensions upwards
//     by first moving the 0th dimension where it will ultimately end up, then processing each dimension in order
//   - if we have dimension 0 at the top, and cut points heading downwards, we'll be reading the data forwards as we process it, which efficiently 
//     loads it into the cache
//   - we don't need to update offset_to_MAIN_DATA_STRUCTURE_in_big_endian_format or the values in the EXTERIOR_STRUCTURE until we return from C++, 
//     so let it get out of date while we process things internally
//   - if the tensor is purely for internal consumption, we don't even need to allocate the EXTERIOR_STRUCTURE and 
//     offset_to_MAIN_DATA_STRUCTURE_in_big_endian_format, 
//     so our allocator should have a flag to indicate if these are required or not
//   - the maximum number of cuts is part of the feature_combination definition, so we don't need to store that and pass that reduntant information arround
//     We do store the current number of cuts because that changes.  This data structure should therefore have a dependency on the 
//     feature_combination definition since we'll need to read the maximum number of cuts.  The pointer to the feature_combination class can be passed 
//     in via the stack to any function that needs that information
//   - use 64 bit values for all offsets, since nobody will ever need more than 64 bits 
//     (you need a non-trivial amount of mass even if you store one bit per atom) and
//     we might pass these between 64 and 32 bit processes, but a few 64 bit offsets won't be a problem even for a 32-bit process.
//   - EXPANDING:
//     - eventually all our tensors will be expanded when doing the model update, because we want to use array lookups instead of binary search when
//       applying the model (array lookup is a huge speed boost over binary search)
//     - our current and best models start out expanded since we want an easy way to communicate with our caller, and we eventualy expect almost all the 
//       cuts to be used anyways
//     - we might as well also flip to epanded mode whenever all dimensions are fully expanded, even when we think we have a compressed model.  
//       For things like bools or low numbers of cuts this could be frequent, and the non-compressed model is actually smaller when all dimensions have 
//       been expanded
//     - once we've been expanded, we no longer need the cut points since the cut points are just incrementing integers.  
//       The maximum number of cuts is stored outside of the data structure already because that's common between all machines/GPUs,
//       so we don't need to pass that redundant information arround, so offset_to_start_of_real_data can point directly to
//       MAIN_DATA_STRUCTURE directly (which is most optimial for compression)
//     - IMPORTANT: if we've been expanded AND offset_to_start_of_real_data points directly to MAIN_DATA_STRUCTURE, then if the external caller
//       reduces the data to just the interior used bytes, we still need a way to get to the MAIN_DATA_STRUCTURE.  If we use 
//       offset_to_DIMENSION_0_COUNT_or_zero_if_value_array_expanded_and_zero_offset_to_MAIN_DATA_STRUCTURE to indicate expansion with a zero value then
//       we can also use it for the tripple use as the offset_to_MAIN_DATA_STRUCTURE_in_big_endian_format value.  Our caller will
//       see a zero in what it expects is the offset_to_MAIN_DATA_STRUCTURE_in_big_endian_format, and then it can find MAIN_DATA_STRUCTURE the same
//       way it would normally by adding the zero to the address it has.  The nice thing is that zero is both big_endian and also little_endian, so it 
//       works for both.
//     - the top item in our MAIN_DATA_STRUCTURE MUST be offset_to_DIMENSION_0_COUNT_or_zero_if_value_array_expanded_and_zero_offset_to_MAIN_DATA_STRUCTURE for
//       our efficient tripple use above!

// TODO : put some of this file into .cpp since some of the functions are repeated and they don't need to all be inline!
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
   // We know the number of dimensions for an feature combination at allocation, so we can put the values right below
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
   bool Expand(const size_t * const acValuesPerDimension);
   void AddExpandedWithBadValueProtection(const FloatEbmType * const aFromValues);
   bool Add(const SegmentedTensor & rhs);

#ifndef NDEBUG
   bool IsEqual(const SegmentedTensor & rhs) const;
#endif // NDEBUG

   INLINE_ALWAYS void SetExpanded() {
      m_bExpanded = true;
   }

   INLINE_ALWAYS bool GetExpanded() {
      return m_bExpanded;
   }

   INLINE_ALWAYS FloatEbmType * GetValues() {
      return m_aValues;
   }

   INLINE_ALWAYS void SetCountDimensions(const size_t cDimensions) {
      EBM_ASSERT(cDimensions <= m_cDimensionsMax);
      m_cDimensions = cDimensions;
   }

   INLINE_ALWAYS ActiveDataType * GetDivisionPointer(const size_t iDimension) {
      EBM_ASSERT(iDimension < m_cDimensions);
      return &ArrayToPointer(m_aDimensions)[iDimension].m_aDivisions[0];
   }

   INLINE_ALWAYS FloatEbmType * GetValuePointer() {
      return &m_aValues[0];
   }
};
static_assert(std::is_standard_layout<SegmentedTensor>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<SegmentedTensor>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<SegmentedTensor>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

#endif // SEGMENTED_TENSOR_H
