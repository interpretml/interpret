// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef SEGMENTED_TENSOR_H
#define SEGMENTED_TENSOR_H

#include <string.h> // memset
#include <type_traits> // std::is_standard_layout
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
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
struct SegmentedTensor final {
private:

   struct DimensionInfoStack {
      const ActiveDataType * m_pDivision1;
      const ActiveDataType * m_pDivision2;
      size_t m_cNewDivisions;
   };

   struct DimensionInfoStackExpand {
      const ActiveDataType * m_pDivision1;
      size_t m_iDivision2;
      size_t m_cNewDivisions;
   };

   // TODO : is this still required after we do tree splitting by pairs??
   // we always allocate our array because we don't want to Require Add(...) to check for the null pointer
   // always allocate one so that we never have to check if we have sufficient storage when we call Reset with one division and two values
   static constexpr size_t k_initialDivisionCapacity = 1;
   static constexpr size_t k_initialValueCapacity = 2;

public:

   struct DimensionInfo {
      size_t m_cDivisions;
      ActiveDataType * m_aDivisions;
      size_t m_cDivisionCapacity;
   };

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

   EBM_INLINE static SegmentedTensor * Allocate(const size_t cDimensionsMax, const size_t cVectorLength) {
      EBM_ASSERT(cDimensionsMax <= k_cDimensionsMax);
      EBM_ASSERT(1 <= cVectorLength); // having 0 classes makes no sense, and having 1 class is useless

      if(IsMultiplyError(cVectorLength, k_initialValueCapacity)) {
         LOG_0(TraceLevelWarning, "WARNING Allocate IsMultiplyError(cVectorLength, k_initialValueCapacity)");
         return nullptr;
      }
      const size_t cValueCapacity = cVectorLength * k_initialValueCapacity;
      if(IsMultiplyError(sizeof(FloatEbmType), cValueCapacity)) {
         LOG_0(TraceLevelWarning, "WARNING Allocate IsMultiplyError(sizeof(FloatEbmType), cValueCapacity)");
         return nullptr;
      }
      const size_t cBytesValues = sizeof(FloatEbmType) * cValueCapacity;

      // this can't overflow since cDimensionsMax can't be bigger than k_cDimensionsMax, which is arround 64
      const size_t cBytesSegmentedRegion = sizeof(SegmentedTensor) - sizeof(DimensionInfo) + sizeof(DimensionInfo) * cDimensionsMax;
      SegmentedTensor * const pSegmentedRegion = static_cast<SegmentedTensor *>(malloc(cBytesSegmentedRegion));
      if(UNLIKELY(nullptr == pSegmentedRegion)) {
         LOG_0(TraceLevelWarning, "WARNING Allocate nullptr == pSegmentedRegion");
         return nullptr;
      }
      // we do this so that if we later fail while allocating arrays inside of this that we can exit easily, otherwise we would need to be careful to 
      // only free pointers that had non-initialized garbage inside of them
      memset(pSegmentedRegion, 0, cBytesSegmentedRegion);

      pSegmentedRegion->m_cVectorLength = cVectorLength;
      pSegmentedRegion->m_cDimensionsMax = cDimensionsMax;
      pSegmentedRegion->m_cDimensions = cDimensionsMax;
      pSegmentedRegion->m_cValueCapacity = cValueCapacity;

      FloatEbmType * const aValues = static_cast<FloatEbmType *>(malloc(cBytesValues));
      if(UNLIKELY(nullptr == aValues)) {
         LOG_0(TraceLevelWarning, "WARNING Allocate nullptr == aValues");
         free(pSegmentedRegion); // don't need to call the full Free(*) yet
         return nullptr;
      }
      pSegmentedRegion->m_aValues = aValues;
      // we only need to set the base case to zero, not our entire initial allocation
      // we checked for cVectorLength * k_initialValueCapacity * sizeof(FloatEbmType), and 1 <= k_initialValueCapacity, 
      // so sizeof(FloatEbmType) * cVectorLength can't overflow
      memset(aValues, 0, sizeof(FloatEbmType) * cVectorLength);

      if(0 != cDimensionsMax) {
         DimensionInfo * pDimension = ArrayToPointer(pSegmentedRegion->m_aDimensions);
         size_t iDimension = 0;
         do {
            EBM_ASSERT(0 == pDimension->m_cDivisions);
            pDimension->m_cDivisionCapacity = k_initialDivisionCapacity;
            ActiveDataType * const aDivisions = static_cast<ActiveDataType *>(malloc(sizeof(ActiveDataType) * k_initialDivisionCapacity)); // this multiply can't overflow
            if(UNLIKELY(nullptr == aDivisions)) {
               LOG_0(TraceLevelWarning, "WARNING Allocate nullptr == aDivisions");
               Free(pSegmentedRegion); // free everything!
               return nullptr;
            }
            pDimension->m_aDivisions = aDivisions;
            ++pDimension;
            ++iDimension;
         } while(iDimension < cDimensionsMax);
      }
      return pSegmentedRegion;
   }

   EBM_INLINE static void Free(SegmentedTensor * const pSegmentedRegion) {
      if(LIKELY(nullptr != pSegmentedRegion)) {
         free(pSegmentedRegion->m_aValues);
         if(LIKELY(0 != pSegmentedRegion->m_cDimensionsMax)) {
            const DimensionInfo * pDimensionInfo = ArrayToPointer(pSegmentedRegion->m_aDimensions);
            const DimensionInfo * const pDimensionInfoEnd = &pDimensionInfo[pSegmentedRegion->m_cDimensionsMax];
            do {
               free(pDimensionInfo->m_aDivisions);
               ++pDimensionInfo;
            } while(pDimensionInfoEnd != pDimensionInfo);
         }
         free(pSegmentedRegion);
      }
   }

   EBM_INLINE void SetCountDimensions(const size_t cDimensions) {
      EBM_ASSERT(cDimensions <= m_cDimensionsMax);
      m_cDimensions = cDimensions;
   }

   EBM_INLINE ActiveDataType * GetDivisionPointer(const size_t iDimension) {
      EBM_ASSERT(iDimension < m_cDimensions);
      return &ArrayToPointer(m_aDimensions)[iDimension].m_aDivisions[0];
   }

   EBM_INLINE FloatEbmType * GetValuePointer() {
      return &m_aValues[0];
   }

   EBM_INLINE void Reset() {
      for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
         ArrayToPointer(m_aDimensions)[iDimension].m_cDivisions = 0;
      }
      // we only need to set the base case to zero
      // this can't overflow since we previously allocated this memory
      memset(m_aValues, 0, sizeof(FloatEbmType) * m_cVectorLength);
      m_bExpanded = false;
   }

   EBM_INLINE bool SetCountDivisions(const size_t iDimension, const size_t cDivisions) {
      EBM_ASSERT(iDimension < m_cDimensions);
      DimensionInfo * const pDimension = &ArrayToPointer(m_aDimensions)[iDimension];
      // we shouldn't be able to expand our length after we're been expanded since expanded should be the maximum size already
      EBM_ASSERT(!m_bExpanded || cDivisions <= pDimension->m_cDivisions);
      if(UNLIKELY(pDimension->m_cDivisionCapacity < cDivisions)) {
         EBM_ASSERT(!m_bExpanded); // we shouldn't be able to expand our length after we're been expanded since expanded should be the maximum size already

         if(IsAddError(cDivisions, cDivisions >> 1)) {
            LOG_0(TraceLevelWarning, "WARNING SetCountDivisions IsAddError(cDivisions, cDivisions >> 1)");
            return true;
         }
         // just increase it by 50% since we don't expect to grow our divisions often after an initial period, 
         // and realloc takes some of the cost of growing away
         size_t cNewDivisionCapacity = cDivisions + (cDivisions >> 1);
         LOG_N(TraceLevelInfo, "SetCountDivisions Growing to size %zu", cNewDivisionCapacity);

         if(IsMultiplyError(sizeof(ActiveDataType), cNewDivisionCapacity)) {
            LOG_0(TraceLevelWarning, "WARNING SetCountDivisions IsMultiplyError(sizeof(ActiveDataType), cNewDivisionCapacity)");
            return true;
         }
         size_t cBytes = sizeof(ActiveDataType) * cNewDivisionCapacity;
         ActiveDataType * const aNewDivisions = static_cast<ActiveDataType *>(realloc(pDimension->m_aDivisions, cBytes));
         if(UNLIKELY(nullptr == aNewDivisions)) {
            // according to the realloc spec, if realloc fails to allocate the new memory, it returns nullptr BUT the old memory is valid.
            // we leave m_aThreadByteBuffer1 alone in this instance and will free that memory later in the destructor
            LOG_0(TraceLevelWarning, "WARNING SetCountDivisions nullptr == aNewDivisions");
            return true;
         }
         pDimension->m_aDivisions = aNewDivisions;
         pDimension->m_cDivisionCapacity = cNewDivisionCapacity;
      } // never shrink our array unless the user chooses to Trim()
      pDimension->m_cDivisions = cDivisions;
      return false;
   }

   EBM_INLINE bool EnsureValueCapacity(const size_t cValues) {
      if(UNLIKELY(m_cValueCapacity < cValues)) {
         EBM_ASSERT(!m_bExpanded); // we shouldn't be able to expand our length after we're been expanded since expanded should be the maximum size already

         if(IsAddError(cValues, cValues >> 1)) {
            LOG_0(TraceLevelWarning, "WARNING EnsureValueCapacity IsAddError(cValues, cValues >> 1)");
            return true;
         }
         // just increase it by 50% since we don't expect to grow our values often after an initial period, and realloc takes some of the cost of growing away
         size_t cNewValueCapacity = cValues + (cValues >> 1);
         LOG_N(TraceLevelInfo, "EnsureValueCapacity Growing to size %zu", cNewValueCapacity);

         if(IsMultiplyError(sizeof(FloatEbmType), cNewValueCapacity)) {
            LOG_0(TraceLevelWarning, "WARNING EnsureValueCapacity IsMultiplyError(sizeof(FloatEbmType), cNewValueCapacity)");
            return true;
         }
         size_t cBytes = sizeof(FloatEbmType) * cNewValueCapacity;
         FloatEbmType * const aNewValues = static_cast<FloatEbmType *>(realloc(m_aValues, cBytes));
         if(UNLIKELY(nullptr == aNewValues)) {
            // according to the realloc spec, if realloc fails to allocate the new memory, it returns nullptr BUT the old memory is valid.
            // we leave m_aThreadByteBuffer1 alone in this instance and will free that memory later in the destructor
            LOG_0(TraceLevelWarning, "WARNING EnsureValueCapacity nullptr == aNewValues");
            return true;
         }
         m_aValues = aNewValues;
         m_cValueCapacity = cNewValueCapacity;
      } // never shrink our array unless the user chooses to Trim()
      return false;
   }

   EBM_INLINE bool Copy(const SegmentedTensor & rhs) {
      EBM_ASSERT(m_cDimensions == rhs.m_cDimensions);

      size_t cValues = m_cVectorLength;
      for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
         const DimensionInfo * const pDimension = &ArrayToPointer(rhs.m_aDimensions)[iDimension];
         size_t cDivisions = pDimension->m_cDivisions;
         EBM_ASSERT(!IsMultiplyError(cValues, cDivisions + 1)); // we're copying this memory, so multiplication can't overflow
         cValues *= (cDivisions + 1);
         if(UNLIKELY(SetCountDivisions(iDimension, cDivisions))) {
            LOG_0(TraceLevelWarning, "WARNING Copy SetCountDivisions(iDimension, cDivisions)");
            return true;
         }
         EBM_ASSERT(!IsMultiplyError(sizeof(ActiveDataType), cDivisions)); // we're copying this memory, so multiplication can't overflow
         memcpy(ArrayToPointer(m_aDimensions)[iDimension].m_aDivisions, pDimension->m_aDivisions, sizeof(ActiveDataType) * cDivisions);
      }
      if(UNLIKELY(EnsureValueCapacity(cValues))) {
         LOG_0(TraceLevelWarning, "WARNING Copy EnsureValueCapacity(cValues)");
         return true;
      }
      EBM_ASSERT(!IsMultiplyError(sizeof(FloatEbmType), cValues)); // we're copying this memory, so multiplication can't overflow
      memcpy(m_aValues, rhs.m_aValues, sizeof(FloatEbmType) * cValues);
      m_bExpanded = rhs.m_bExpanded;
      return false;
   }

//#ifndef NDEBUG
//   EBM_INLINE FloatEbmType * GetValue(const ActiveDataType * const aDivisionValue) const {
//      if(0 == m_cDimensions) {
//         return &m_aValues[0]; // there are no dimensions, and only 1 value
//      }
//      const DimensionInfo * pDimension = ArrayToPointer(m_aDimensions);
//      const ActiveDataType * pDivisionValue = aDivisionValue;
//      const ActiveDataType * const pDivisionValueEnd = &aDivisionValue[m_cDimensions];
//      size_t iValue = 0;
//      size_t valueMultiple = m_cVectorLength;
//
//      if(m_bExpanded) {
//         while(true) {
//            const ActiveDataType d = *pDivisionValue;
//            EBM_ASSERT(!IsMultiplyError(d, valueMultiple)); // we're accessing existing memory, so it can't overflow
//            size_t addValue = d * valueMultiple;
//            EBM_ASSERT(!IsAddError(addValue, iValue)); // we're accessing existing memory, so it can't overflow
//            iValue += addValue;
//            ++pDivisionValue;
//            if(pDivisionValueEnd == pDivisionValue) {
//               break;
//            }
//            const size_t cDivisions = pDimension->m_cDivisions;
//            EBM_ASSERT(1 <= cDivisions); // since we're expanded we should have at least one division and two values
//            EBM_ASSERT(!IsMultiplyError(cDivisions + 1, valueMultiple)); // we're accessing existing memory, so it can't overflow
//            valueMultiple *= cDivisions + 1;
//            ++pDimension;
//         }
//      } else {
//         DO: this code is no longer executed because we always expand our models now.  We can probably get rid of it, but I'm leaving it here for a while to decide if there are really no use cases
//         do {
//            const size_t cDivisions = pDimension->m_cDivisions;
//            if(LIKELY(0 != cDivisions)) {
//               const ActiveDataType * const aDivisions = pDimension->m_aDivisions;
//               const ActiveDataType d = *pDivisionValue;
//               ptrdiff_t high = cDivisions - 1;
//               ptrdiff_t middle;
//               ptrdiff_t low = 0;
//               ActiveDataType midVal;
//               do {
//                  middle = (low + high) >> 1;
//                  midVal = aDivisions[middle];
//                  if(UNLIKELY(midVal == d)) {
//                     // this happens just once during our descent, so it's less likely than continuing searching
//                     goto no_check;
//                  }
//                  high = UNPREDICTABLE(midVal < d) ? high : middle - 1;
//                  low = UNPREDICTABLE(midVal < d) ? middle + 1 : low;
//               } while(LIKELY(low <= high));
//               middle = UNPREDICTABLE(midVal < d) ? middle + 1 : middle;
//            no_check:
//               EBM_ASSERT(!IsMultiplyError(middle, valueMultiple)); // we're accessing existing memory, so it can't overflow
//               ptrdiff_t addValue = middle * valueMultiple;
//               EBM_ASSERT(!IsAddError(iValue, addValue)); // we're accessing existing memory, so it can't overflow
//               iValue += addValue;
//               EBM_ASSERT(!IsMultiplyError(valueMultiple, cDivisions + 1)); // we're accessing existing memory, so it can't overflow
//               valueMultiple *= cDivisions + 1;
//            }
//            ++pDimension;
//            ++pDivisionValue;
//         } while(pDivisionValueEnd != pDivisionValue);
//      }
//      return &m_aValues[iValue];
//   }
//#endif // NDEBUG

   EBM_INLINE bool MultiplyAndCheckForIssues(const FloatEbmType v) {
      size_t cValues = 1;
      for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
         // we're accessing existing memory, so it can't overflow
         EBM_ASSERT(!IsMultiplyError(cValues, ArrayToPointer(m_aDimensions)[iDimension].m_cDivisions + 1));
         cValues *= ArrayToPointer(m_aDimensions)[iDimension].m_cDivisions + 1;
      }

      FloatEbmType * pCur = &m_aValues[0];
      FloatEbmType * pEnd = &m_aValues[cValues * m_cVectorLength];
      int bBad = 0;
      // we always have 1 value, even if we have zero divisions
      do {
         const FloatEbmType val = *pCur * v;
         // TODO: these can be done with bitwise operators, which would be good for SIMD.  Check to see what assembly this turns into.
         // since both NaN and +-infinity have the exponential as FF, and no other values do, the best optimized assembly would test the exponential 
         // bits for FF and then OR a 1 if the test is true and 0 if the test is false
         bBad |= std::isnan(val) || std::isinf(val);
         *pCur = val;
         ++pCur;
      } while(pEnd != pCur);
      return !!bBad;
   }

   EBM_INLINE bool Expand(const size_t * const acValuesPerDimension) {
      LOG_0(TraceLevelVerbose, "Entered Expand");

      EBM_ASSERT(1 <= m_cDimensions); // you can't really expand something with zero dimensions
      EBM_ASSERT(nullptr != acValuesPerDimension);
      // ok, checking the max isn't really the best here, but doing this right seems pretty complicated, and this should detect any real problems.
      // don't make this a static assert.  The rest of our class is fine as long as Expand is never called
      EBM_ASSERT(std::numeric_limits<size_t>::max() <= std::numeric_limits<ActiveDataType>::max() &&
         0 == std::numeric_limits<ActiveDataType>::min());
      if(m_bExpanded) {
         // we're already expanded
         LOG_0(TraceLevelVerbose, "Exited Expand");
         return false;
      }

      EBM_ASSERT(m_cDimensions <= k_cDimensionsMax);
      DimensionInfoStackExpand aDimensionInfoStackExpand[k_cDimensionsMax];

      const DimensionInfo * pDimensionFirst1 = ArrayToPointer(m_aDimensions);

      DimensionInfoStackExpand * pDimensionInfoStackFirst = aDimensionInfoStackExpand;
      const DimensionInfoStackExpand * const pDimensionInfoStackEnd = &aDimensionInfoStackExpand[m_cDimensions];
      const size_t * pcValuesPerDimension = acValuesPerDimension;

      size_t cValues1 = 1;
      size_t cNewValues = 1;

      EBM_ASSERT(0 < m_cDimensions);
      // first, get basic counts of how many divisions and values we'll have in our final result
      do {
         const size_t cDivisions1 = pDimensionFirst1->m_cDivisions;

         EBM_ASSERT(!IsMultiplyError(cValues1, cDivisions1 + 1)); // this is accessing existing memory, so it can't overflow
         cValues1 *= cDivisions1 + 1;

         pDimensionInfoStackFirst->m_pDivision1 = &pDimensionFirst1->m_aDivisions[cDivisions1];
         const size_t cValuesPerDimension = *pcValuesPerDimension;
         // we check for simple multiplication overflow from m_cBins in EbmBoostingState->Initialize when we unpack featureCombinationIndexes 
         // and in GetInteractionScore for interactions
         EBM_ASSERT(!IsMultiplyError(cNewValues, cValuesPerDimension));
         cNewValues *= cValuesPerDimension;
         const size_t cNewDivisions = cValuesPerDimension - 1;

         pDimensionInfoStackFirst->m_iDivision2 = cNewDivisions;
         pDimensionInfoStackFirst->m_cNewDivisions = cNewDivisions;

         ++pDimensionFirst1;
         ++pcValuesPerDimension;
         ++pDimensionInfoStackFirst;
      } while(pDimensionInfoStackEnd != pDimensionInfoStackFirst);

      if(IsMultiplyError(cNewValues, m_cVectorLength)) {
         LOG_0(TraceLevelWarning, "WARNING Expand IsMultiplyError(cNewValues, m_cVectorLength)");
         return true;
      }
      const size_t cVectoredNewValues = cNewValues * m_cVectorLength;
      // call EnsureValueCapacity before using the m_aValues pointer since m_aValues might change inside EnsureValueCapacity
      if(UNLIKELY(EnsureValueCapacity(cVectoredNewValues))) {
         LOG_0(TraceLevelWarning, "WARNING Expand EnsureValueCapacity(cVectoredNewValues))");
         return true;
      }

      FloatEbmType * const aValues = m_aValues;
      const DimensionInfo * const aDimension1 = ArrayToPointer(m_aDimensions);

      EBM_ASSERT(cValues1 <= cNewValues);
      EBM_ASSERT(!IsMultiplyError(m_cVectorLength, cValues1)); // we checked against cNewValues above, and cValues1 should be smaller
      const FloatEbmType * pValue1 = &aValues[m_cVectorLength * cValues1];
      FloatEbmType * pValueTop = &aValues[cVectoredNewValues];

      // traverse the values in reverse so that we can put our results at the higher order indexes where we are guaranteed not to overwrite our 
      // existing values which we still need to copy first do the values because we need to refer to the old divisions when making decisions about 
      // where to move next
      while(true) {
         const FloatEbmType * pValue1Move = pValue1;
         const FloatEbmType * const pValueTopEnd = pValueTop - m_cVectorLength;
         do {
            --pValue1Move;
            --pValueTop;
            *pValueTop = *pValue1Move;
         } while(pValueTopEnd != pValueTop);

         // For a single dimensional SegmentedRegion checking here is best.  
         // For two or higher dimensions, we could instead check inside our loop below for when we reach the end of the pDimensionInfoStack, thus 
         // eliminating the check on most loops. We'll spend most of our time working on single features though, so we optimize for that case, but 
         // if we special cased the single dimensional case, then we would want to move this check into the loop below in the case of 
         // multi-dimensioncal SegmentedTensors
         if(UNLIKELY(aValues == pValueTop)) {
            // we've written our final tensor cell, so we're done
            break;
         }

         DimensionInfoStackExpand * pDimensionInfoStackSecond = aDimensionInfoStackExpand;
         const DimensionInfo * pDimensionSecond1 = aDimension1;

         size_t multiplication1 = m_cVectorLength;

         while(true) {
            const ActiveDataType * const pDivision1 = pDimensionInfoStackSecond->m_pDivision1;
            size_t iDivision2 = pDimensionInfoStackSecond->m_iDivision2;

            ActiveDataType * const aDivisions1 = pDimensionSecond1->m_aDivisions;

            if(UNPREDICTABLE(aDivisions1 < pDivision1)) {
               EBM_ASSERT(0 < iDivision2);

               const ActiveDataType * const pDivision1MinusOne = pDivision1 - 1;

               const size_t d1 = static_cast<size_t>(*pDivision1MinusOne);

               --iDivision2;

               const bool bMove = UNPREDICTABLE(iDivision2 <= d1);
               pDimensionInfoStackSecond->m_pDivision1 = bMove ? pDivision1MinusOne : pDivision1;
               pValue1 = bMove ? pValue1 - multiplication1 : pValue1;

               pDimensionInfoStackSecond->m_iDivision2 = iDivision2;
               break;
            } else {
               if(UNPREDICTABLE(0 < iDivision2)) {
                  pDimensionInfoStackSecond->m_iDivision2 = iDivision2 - 1;
                  break;
               } else {
                  pValue1 -= multiplication1; // put us before the beginning.  We'll add the full row first

                  const size_t cDivisions1 = pDimensionSecond1->m_cDivisions;

                  // we're already allocated values, so this is accessing what we've already allocated, so it must not overflow
                  EBM_ASSERT(!IsMultiplyError(multiplication1, 1 + cDivisions1));
                  multiplication1 *= 1 + cDivisions1;

                  // go to the last valid entry back to where we started.  If we don't move down a set, then we re-do this set of numbers
                  pValue1 += multiplication1;

                  pDimensionInfoStackSecond->m_pDivision1 = &aDivisions1[cDivisions1];
                  pDimensionInfoStackSecond->m_iDivision2 = pDimensionInfoStackSecond->m_cNewDivisions;

                  ++pDimensionSecond1;
                  ++pDimensionInfoStackSecond;
                  continue;
               }
            }
         }
      }

      EBM_ASSERT(pValueTop == m_aValues);
      EBM_ASSERT(pValue1 == m_aValues + m_cVectorLength);

      for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
         const size_t cDivisions = acValuesPerDimension[iDimension] - 1;

         if(cDivisions == ArrayToPointer(m_aDimensions)[iDimension].m_cDivisions) {
            continue;
         }

         if(UNLIKELY(SetCountDivisions(iDimension, cDivisions))) {
            LOG_0(TraceLevelWarning, "WARNING Expand SetCountDivisions(iDimension, cDivisions)");
            return true;
         }

         for(size_t iDivision = 0; iDivision < cDivisions; ++iDivision) {
            ArrayToPointer(m_aDimensions)[iDimension].m_aDivisions[iDivision] = iDivision;
         }
      }

      m_bExpanded = true;
      LOG_0(TraceLevelVerbose, "Exited Expand");
      return false;
   }

   EBM_INLINE void AddExpandedWithBadValueProtection(const FloatEbmType * const aFromValues) {
      EBM_ASSERT(m_bExpanded);
      size_t cItems = m_cVectorLength;
      for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
         // this can't overflow since we've already allocated them!
         cItems *= ArrayToPointer(m_aDimensions)[iDimension].m_cDivisions + 1;
      }

      const FloatEbmType * pFromValue = aFromValues;
      FloatEbmType * pToValue = m_aValues;
      const FloatEbmType * const pToValueEnd = m_aValues + cItems;
      do {
         // if we get a NaN value, then just consider it a no-op zero
         // if we get a +infinity, then just make our value the maximum
         // if we get a -infinity, then just make our value the minimum
         // these changes will make us out of sync with the updates to our logits, but it should be at the extremes anyways
         // so, not much real loss there.  Also, if we have NaN, or +-infinity in an update, we'll be stopping boosting soon
         // but we want to preserve the best model that we had

         FloatEbmType val = *pFromValue;
         val = std::isnan(val) ? FloatEbmType { 0 } : val;
         val = *pToValue + val;
         // this is a check for -infinity, without the -infinity value since some compilers make that illegal
         val = val < std::numeric_limits<FloatEbmType>::lowest() ? std::numeric_limits<FloatEbmType>::lowest() : val;
         // this is a check for +infinity, without the +infinity value since some compilers make that illegal
         val = std::numeric_limits<FloatEbmType>::max() < val ? std::numeric_limits<FloatEbmType>::max() : val;
         *pToValue = val;
         ++pFromValue;
         ++pToValue;
      } while(pToValueEnd != pToValue);
   }

   // TODO : consider adding templated cVectorLength and cDimensions to this function.  At worst someone can pass in 0 and use the loops 
   //   without needing to super-optimize it
   EBM_INLINE bool Add(const SegmentedTensor & rhs) {
      DimensionInfoStack dimensionStack[k_cDimensionsMax];

      EBM_ASSERT(m_cDimensions == rhs.m_cDimensions);

      if(0 == m_cDimensions) {
         EBM_ASSERT(1 <= m_cValueCapacity);
         EBM_ASSERT(nullptr != m_aValues);

         FloatEbmType * pTo = &m_aValues[0];
         const FloatEbmType * pFrom = &rhs.m_aValues[0];
         const FloatEbmType * const pToEnd = &pTo[m_cVectorLength];
         do {
            *pTo += *pFrom;
            ++pTo;
            ++pFrom;
         } while(pToEnd != pTo);

         return false;
      }

      if(m_bExpanded) {
         // TODO: the existing code below works, but handle this differently (we can do it more efficiently)
      }

      if(rhs.m_bExpanded) {
         // TODO: the existing code below works, but handle this differently (we can do it more efficiently)
      }

      const DimensionInfo * pDimensionFirst1 = ArrayToPointer(m_aDimensions);
      const DimensionInfo * pDimensionFirst2 = ArrayToPointer(rhs.m_aDimensions);

      DimensionInfoStack * pDimensionInfoStackFirst = dimensionStack;
      const DimensionInfoStack * const pDimensionInfoStackEnd = &dimensionStack[m_cDimensions];

      size_t cValues1 = 1;
      size_t cValues2 = 1;
      size_t cNewValues = 1;

      EBM_ASSERT(0 < m_cDimensions);
      // first, get basic counts of how many divisions and values we'll have in our final result
      do {
         const size_t cDivisions1 = pDimensionFirst1->m_cDivisions;
         ActiveDataType * p1Cur = pDimensionFirst1->m_aDivisions;
         const size_t cDivisions2 = pDimensionFirst2->m_cDivisions;
         ActiveDataType * p2Cur = pDimensionFirst2->m_aDivisions;

         cValues1 *= cDivisions1 + 1; // this can't overflow since we're counting existing allocated memory
         cValues2 *= cDivisions2 + 1; // this can't overflow since we're counting existing allocated memory

         ActiveDataType * const p1End = &p1Cur[cDivisions1];
         ActiveDataType * const p2End = &p2Cur[cDivisions2];

         pDimensionInfoStackFirst->m_pDivision1 = p1End;
         pDimensionInfoStackFirst->m_pDivision2 = p2End;

         size_t cNewSingleDimensionDivisions = 0;

         // processing forwards here is slightly faster in terms of cache fetch efficiency.  We'll then be guaranteed to have the divisions at least
         // in the cache, which will be benefitial when traversing backwards later below
         while(true) {
            if(UNLIKELY(p2End == p2Cur)) {
               // check the other array first.  Most of the time the other array will be shorter since we'll be adding
               // a sequence of Segmented lines and our main line will be in *this, and there will be more segments in general for
               // a line that is added to a lot
               cNewSingleDimensionDivisions += static_cast<size_t>(p1End - p1Cur);
               break;
            }
            if(UNLIKELY(p1End == p1Cur)) {
               cNewSingleDimensionDivisions += static_cast<size_t>(p2End - p2Cur);
               break;
            }
            ++cNewSingleDimensionDivisions; // if we move one or both pointers, we just added annother unique one

            const ActiveDataType d1 = *p1Cur;
            const ActiveDataType d2 = *p2Cur;

            p1Cur = UNPREDICTABLE(d1 <= d2) ? p1Cur + 1 : p1Cur;
            p2Cur = UNPREDICTABLE(d2 <= d1) ? p2Cur + 1 : p2Cur;
         }
         pDimensionInfoStackFirst->m_cNewDivisions = cNewSingleDimensionDivisions;
         // we check for simple multiplication overflow from m_cBins in EbmBoostingState->Initialize when we unpack featureCombinationIndexes and in 
         // GetInteractionScore for interactions
         EBM_ASSERT(!IsMultiplyError(cNewValues, cNewSingleDimensionDivisions + 1));
         cNewValues *= cNewSingleDimensionDivisions + 1;

         ++pDimensionFirst1;
         ++pDimensionFirst2;

         ++pDimensionInfoStackFirst;
      } while(pDimensionInfoStackEnd != pDimensionInfoStackFirst);

      if(IsMultiplyError(cNewValues, m_cVectorLength)) {
         LOG_0(TraceLevelWarning, "WARNING Add IsMultiplyError(cNewValues, m_cVectorLength)");
         return true;
      }
      // call EnsureValueCapacity before using the m_aValues pointer since m_aValues might change inside EnsureValueCapacity
      if(UNLIKELY(EnsureValueCapacity(cNewValues * m_cVectorLength))) {
         LOG_0(TraceLevelWarning, "WARNING Add EnsureValueCapacity(cNewValues * m_cVectorLength)");
         return true;
      }

      const FloatEbmType * pValue2 = &rhs.m_aValues[m_cVectorLength * cValues2];  // we're accessing allocated memory, so it can't overflow
      const DimensionInfo * const aDimension2 = ArrayToPointer(rhs.m_aDimensions);

      FloatEbmType * const aValues = m_aValues;
      const DimensionInfo * const aDimension1 = ArrayToPointer(m_aDimensions);

      const FloatEbmType * pValue1 = &aValues[m_cVectorLength * cValues1]; // we're accessing allocated memory, so it can't overflow
      FloatEbmType * pValueTop = &aValues[m_cVectorLength * cNewValues]; // we're accessing allocated memory, so it can't overflow

      // traverse the values in reverse so that we can put our results at the higher order indexes where we are guaranteed not to overwrite our
      // existing values which we still need to copy first do the values because we need to refer to the old divisions when making decisions about where 
      // to move next
      while(true) {
         const FloatEbmType * pValue1Move = pValue1;
         const FloatEbmType * pValue2Move = pValue2;
         const FloatEbmType * const pValueTopEnd = pValueTop - m_cVectorLength;
         do {
            --pValue1Move;
            --pValue2Move;
            --pValueTop;
            *pValueTop = *pValue1Move + *pValue2Move;
         } while(pValueTopEnd != pValueTop);

         // For a single dimensional SegmentedRegion checking here is best.  
         // For two or higher dimensions, we could instead check inside our loop below for when we reach the end of the pDimensionInfoStack,
         // thus eliminating the check on most loops.  We'll spend most of our time working on single features though, so we optimize for that case, 
         // but if we special cased the single dimensional case, then we would want to move this check into the loop below in the case 
         // of multi-dimensioncal SegmentedTensors
         if(UNLIKELY(aValues == pValueTop)) {
            // we've written our final tensor cell, so we're done
            break;
         }

         DimensionInfoStack * pDimensionInfoStackSecond = dimensionStack;
         const DimensionInfo * pDimensionSecond1 = aDimension1;
         const DimensionInfo * pDimensionSecond2 = aDimension2;

         size_t multiplication1 = m_cVectorLength;
         size_t multiplication2 = m_cVectorLength;

         while(true) {
            const ActiveDataType * const pDivision1 = pDimensionInfoStackSecond->m_pDivision1;
            const ActiveDataType * const pDivision2 = pDimensionInfoStackSecond->m_pDivision2;

            ActiveDataType * const aDivisions1 = pDimensionSecond1->m_aDivisions;
            ActiveDataType * const aDivisions2 = pDimensionSecond2->m_aDivisions;

            if(UNPREDICTABLE(aDivisions1 < pDivision1)) {
               if(UNPREDICTABLE(aDivisions2 < pDivision2)) {
                  const ActiveDataType * const pDivision1MinusOne = pDivision1 - 1;
                  const ActiveDataType * const pDivision2MinusOne = pDivision2 - 1;

                  const ActiveDataType d1 = *pDivision1MinusOne;
                  const ActiveDataType d2 = *pDivision2MinusOne;

                  const bool bMove1 = UNPREDICTABLE(d2 <= d1);
                  pDimensionInfoStackSecond->m_pDivision1 = bMove1 ? pDivision1MinusOne : pDivision1;
                  pValue1 = bMove1 ? pValue1 - multiplication1 : pValue1;

                  const bool bMove2 = UNPREDICTABLE(d1 <= d2);
                  pDimensionInfoStackSecond->m_pDivision2 = bMove2 ? pDivision2MinusOne : pDivision2;
                  pValue2 = bMove2 ? pValue2 - multiplication2 : pValue2;
                  break;
               } else {
                  pValue1 -= multiplication1;
                  pDimensionInfoStackSecond->m_pDivision1 = pDivision1 - 1;
                  break;
               }
            } else {
               if(UNPREDICTABLE(aDivisions2 < pDivision2)) {
                  pValue2 -= multiplication2;
                  pDimensionInfoStackSecond->m_pDivision2 = pDivision2 - 1;
                  break;
               } else {
                  pValue1 -= multiplication1; // put us before the beginning.  We'll add the full row first
                  pValue2 -= multiplication2; // put us before the beginning.  We'll add the full row first

                  const size_t cDivisions1 = pDimensionSecond1->m_cDivisions;
                  const size_t cDivisions2 = pDimensionSecond2->m_cDivisions;

                  EBM_ASSERT(!IsMultiplyError(multiplication1, 1 + cDivisions1)); // we're accessing allocated memory, so it can't overflow
                  multiplication1 *= 1 + cDivisions1;
                  EBM_ASSERT(!IsMultiplyError(multiplication2, 1 + cDivisions2)); // we're accessing allocated memory, so it can't overflow
                  multiplication2 *= 1 + cDivisions2;

                  // go to the last valid entry back to where we started.  If we don't move down a set, then we re-do this set of numbers
                  pValue1 += multiplication1;
                  // go to the last valid entry back to where we started.  If we don't move down a set, then we re-do this set of numbers
                  pValue2 += multiplication2;

                  pDimensionInfoStackSecond->m_pDivision1 = &aDivisions1[cDivisions1];
                  pDimensionInfoStackSecond->m_pDivision2 = &aDivisions2[cDivisions2];
                  ++pDimensionSecond1;
                  ++pDimensionSecond2;
                  ++pDimensionInfoStackSecond;
                  continue;
               }
            }
         }
      }

      EBM_ASSERT(pValueTop == m_aValues);
      EBM_ASSERT(pValue1 == m_aValues + m_cVectorLength);
      EBM_ASSERT(pValue2 == rhs.m_aValues + m_cVectorLength);

      // now finally do the divisions

      const DimensionInfoStack * pDimensionInfoStackCur = dimensionStack;
      const DimensionInfo * pDimension1Cur = aDimension1;
      const DimensionInfo * pDimension2Cur = aDimension2;
      size_t iDimension = 0;
      do {
         const size_t cNewDivisions = pDimensionInfoStackCur->m_cNewDivisions;
         const size_t cOriginalDivisionsBeforeSetting = pDimension1Cur->m_cDivisions;
         
         // this will increase our capacity, if required.  It will also change m_cDivisions, so we get that before calling it.  
         // SetCountDivisions might change m_aValuesAndDivisions, so we need to actually keep it here after getting m_cDivisions but 
         // before set set all our pointers
         if(UNLIKELY(SetCountDivisions(iDimension, cNewDivisions))) {
            LOG_0(TraceLevelWarning, "WARNING Add SetCountDivisions(iDimension, cNewDivisions)");
            return true;
         }
         
         const ActiveDataType * p1Cur = &pDimension1Cur->m_aDivisions[cOriginalDivisionsBeforeSetting];
         const ActiveDataType * p2Cur = &pDimension2Cur->m_aDivisions[pDimension2Cur->m_cDivisions];
         ActiveDataType * pTopCur = &pDimension1Cur->m_aDivisions[cNewDivisions];

         // traverse in reverse so that we can put our results at the higher order indexes where we are guaranteed not to overwrite our existing values
         // which we still need to copy
         while(true) {
            EBM_ASSERT(pDimension1Cur->m_aDivisions <= pTopCur);
            EBM_ASSERT(pDimension1Cur->m_aDivisions <= p1Cur);
            EBM_ASSERT(pDimension2Cur->m_aDivisions <= p2Cur);
            EBM_ASSERT(p1Cur <= pTopCur);
            EBM_ASSERT(static_cast<size_t>(p2Cur - pDimension2Cur->m_aDivisions) <= static_cast<size_t>(pTopCur - pDimension1Cur->m_aDivisions));

            if(UNLIKELY(pTopCur == p1Cur)) {
               // since we've finished the rhs divisions, our SegmentedRegion already has the right divisions in place, so all we need is to add the value
               // of the last region in rhs to our remaining values
               break;
            }
            // pTopCur is an index above pDimension1Cur->m_aDivisions.  p2Cur is an index above pDimension2Cur->m_aDivisions.  We want to decide if they
            // are at the same index above their respective arrays
            if(UNLIKELY(static_cast<size_t>(pTopCur - pDimension1Cur->m_aDivisions) == static_cast<size_t>(p2Cur - pDimension2Cur->m_aDivisions))) {
               EBM_ASSERT(pDimension1Cur->m_aDivisions < pTopCur);
               // direct copy the remaining divisions.  There should be at least one
               memcpy(
                  pDimension1Cur->m_aDivisions, 
                  pDimension2Cur->m_aDivisions, 
                  static_cast<size_t>(pTopCur - pDimension1Cur->m_aDivisions) * sizeof(ActiveDataType)
               );
               break;
            }

            const ActiveDataType * const p1CurMinusOne = p1Cur - 1;
            const ActiveDataType * const p2CurMinusOne = p2Cur - 1;

            const ActiveDataType d1 = *p1CurMinusOne;
            const ActiveDataType d2 = *p2CurMinusOne;

            p1Cur = UNPREDICTABLE(d2 <= d1) ? p1CurMinusOne : p1Cur;
            p2Cur = UNPREDICTABLE(d1 <= d2) ? p2CurMinusOne : p2Cur;

            const ActiveDataType d = UNPREDICTABLE(d1 <= d2) ? d2 : d1;

            --pTopCur; // if we move one or both pointers, we just added annother unique one
            *pTopCur = d;
         }
         ++pDimension1Cur;
         ++pDimension2Cur;
         ++pDimensionInfoStackCur;
         ++iDimension;
      } while(iDimension != m_cDimensions);
      return false;
   }

#ifndef NDEBUG
   EBM_INLINE bool IsEqual(const SegmentedTensor & rhs) const {
      if(m_cDimensions != rhs.m_cDimensions) {
         return false;
      }

      size_t cValues = m_cVectorLength;
      for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
         const DimensionInfo * const pDimension1 = &ArrayToPointer(m_aDimensions)[iDimension];
         const DimensionInfo * const pDimension2 = &ArrayToPointer(rhs.m_aDimensions)[iDimension];

         size_t cDivisions = pDimension1->m_cDivisions;
         if(cDivisions != pDimension2->m_cDivisions) {
            return false;
         }

         if(0 != cDivisions) {
            EBM_ASSERT(!IsMultiplyError(cValues, cDivisions + 1)); // we're accessing allocated memory, so it can't overflow
            cValues *= cDivisions + 1;

            const ActiveDataType * pD1Cur = pDimension1->m_aDivisions;
            const ActiveDataType * pD2Cur = pDimension2->m_aDivisions;
            const ActiveDataType * const pD1End = pD1Cur + cDivisions;
            do {
               if(UNLIKELY(*pD1Cur != *pD2Cur)) {
                  return false;
               }
               ++pD1Cur;
               ++pD2Cur;
            } while(LIKELY(pD1End != pD1Cur));
         }
      }

      const FloatEbmType * pV1Cur = &m_aValues[0];
      const FloatEbmType * pV2Cur = &rhs.m_aValues[0];
      const FloatEbmType * const pV1End = pV1Cur + cValues;
      do {
         if(UNLIKELY(*pV1Cur != *pV2Cur)) {
            return false;
         }
         ++pV1Cur;
         ++pV2Cur;
      } while(LIKELY(pV1End != pV1Cur));

      return true;
   }
#endif // NDEBUG
};
static_assert(
   std::is_standard_layout<SegmentedTensor>::value, 
   "SegmentedRegion uses the struct hack, so it must be a standard layout class.  We use realloc, which isn't compatible with using complex classes.  "
   "Interop data must also be standard layout classes.  Lastly, we put this class into a union, so the destructor needs to be called manually anyways");

#endif // SEGMENTED_TENSOR_H
