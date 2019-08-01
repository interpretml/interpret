// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef SEGMENTED_REGION_H
#define SEGMENTED_REGION_H

// TODO : check for multiplication overflows within this class (we could overflow in several parts)
// TODO : try and make this work with size_t instead of needing ptrdiff_t as we currently do

#include <type_traits> // std::is_pod
#include <assert.h>
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // TML_INLINE
#include "Logging.h" // EBM_ASSERT & LOG

// TODO : after we've optimized a lot more and fit into the python wrapper and we've completely solved the bucketing, consider making SegmentedRegion with variable types that we can switch
// we could put make TDivisions and TValues conditioned on individual functions and tread our allocated memory as a pool of variable types.   We cache SegmentedRegions right now for different types
// but they could all be shared into one class that get morphed.  We're currently getting some type safety that we wouldn't though otherwise, so hold off on this change until we can performance check
// I think we'll find that using size_t as TDivisions is as performant or better than using anything else, so it may be a moot point, in which case leave it as hard coded types and just make all TDivisions size_t, even for mains
// for pairs and triplicates, we already know that the dimensionality aspect requires us to have common division types since we don't want char/short SegmentedRegion classes and all the combinatorial options that would allow
template<typename TDivisions, typename TValues>
class SegmentedRegionCore final {
   struct DimensionInfoStack {
      const TDivisions * pDivision1;
      const TDivisions * pDivision2;
      size_t cNewDivisions;
   };

   struct DimensionInfoStackExpand {
      const TDivisions * pDivision1;
      ptrdiff_t iDivision2;
      size_t cNewDivisions;
   };

   // TODO : is this still required after we do tree splitting by pairs??
   // we always allocate our array because we don't want to Require Add(...) to check for the null pointer
   // always allocate one so that we never have to check if we have sufficient storage when we call Reset with one division and two values
   static constexpr size_t k_initialDivisionCapacity = 1;
   static constexpr size_t k_initialValueCapacity = 2;

public:

   struct DimensionInfo {
      size_t cDivisions;
      TDivisions * aDivisions;
      size_t cDivisionCapacity;
   };

   size_t m_cValueCapacity;
   size_t m_cVectorLength;
   size_t m_cDimensionsMax;
   size_t m_cDimensions;
   TValues * m_aValues;
   bool m_bExpanded;
   // TODO : I lean towards leaving this alone since pointers to SegmentedRegions instead of having an array of compact objects seems fine, but i should look over and consider changing this to eliminate dynamic allocation and replace it with k_cDimensionsMax
   DimensionInfo m_aDimensions[1];

   TML_INLINE static SegmentedRegionCore * Allocate(const size_t cDimensionsMax, const size_t cVectorLength) {
      EBM_ASSERT(cDimensionsMax <= k_cDimensionsMax);
      EBM_ASSERT(1 <= cVectorLength); // having 0 states makes no sense, and having 1 state is useless

      if(IsMultiplyError(cVectorLength, k_initialValueCapacity)) {
         LOG(TraceLevelWarning, "WARNING Allocate IsMultiplyError(cVectorLength, k_initialValueCapacity)");
         return nullptr;
      }
      const size_t cValueCapacity = cVectorLength * k_initialValueCapacity;
      if(IsMultiplyError(sizeof(TValues), cValueCapacity)) {
         LOG(TraceLevelWarning, "WARNING Allocate IsMultiplyError(sizeof(TValues), cValueCapacity)");
         return nullptr;
      }
      const size_t cBytesValues = sizeof(TValues) * cValueCapacity;

      // this can't overflow since cDimensionsMax can't be bigger than k_cDimensionsMax, which is arround 64
      const size_t cBytesSegmentedRegion = sizeof(SegmentedRegionCore) - sizeof(DimensionInfo) + sizeof(DimensionInfo) * cDimensionsMax;
      SegmentedRegionCore * const pSegmentedRegion = static_cast<SegmentedRegionCore *>(malloc(cBytesSegmentedRegion));
      if(UNLIKELY(nullptr == pSegmentedRegion)) {
         LOG(TraceLevelWarning, "WARNING Allocate nullptr == pSegmentedRegion");
         return nullptr;
      }
      memset(pSegmentedRegion, 0, cBytesSegmentedRegion); // we do this so that if we later fail while allocating arrays inside of this that we can exit easily, otherwise we would need to be careful to only free pointers that had non-initialized garbage inside of them

      pSegmentedRegion->m_cVectorLength = cVectorLength;
      pSegmentedRegion->m_cDimensionsMax = cDimensionsMax;
      pSegmentedRegion->m_cDimensions = cDimensionsMax;
      pSegmentedRegion->m_cValueCapacity = cValueCapacity;

      TValues * const aValues = static_cast<TValues *>(malloc(cBytesValues));
      if(UNLIKELY(nullptr == aValues)) {
         LOG(TraceLevelWarning, "WARNING Allocate nullptr == aValues");
         free(pSegmentedRegion); // don't need to call the full Free(*) yet
         return nullptr;
      }
      pSegmentedRegion->m_aValues = aValues;
      // we only need to set the base case to zero, not our entire initial allocation
      memset(aValues, 0, sizeof(TValues) * cVectorLength);

      if(0 != cDimensionsMax) {
         DimensionInfo * pDimension = &pSegmentedRegion->m_aDimensions[0];
         size_t iDimension = 0;
         do {
            EBM_ASSERT(0 == pDimension->cDivisions);
            pDimension->cDivisionCapacity = k_initialDivisionCapacity;
            TDivisions * const aDivisions = static_cast<TDivisions *>(malloc(sizeof(TDivisions) * k_initialDivisionCapacity)); // this multiply can't overflow
            if(UNLIKELY(nullptr == aDivisions)) {
               LOG(TraceLevelWarning, "WARNING Allocate nullptr == aDivisions");
               Free(pSegmentedRegion); // free everything!
               return nullptr;
            }
            pDimension->aDivisions = aDivisions;
            ++pDimension;
            ++iDimension;
         } while(iDimension < cDimensionsMax);
      }
      return pSegmentedRegion;
   }

   TML_INLINE static void Free(SegmentedRegionCore * const pSegmentedRegion) {
      if(UNLIKELY(nullptr != pSegmentedRegion)) {
         free(pSegmentedRegion->m_aValues);
         for(size_t iDimension = 0; iDimension < pSegmentedRegion->m_cDimensionsMax; ++iDimension) {
            free(pSegmentedRegion->m_aDimensions[iDimension].aDivisions);
         }
         free(pSegmentedRegion);
      }
   }

   TML_INLINE void SetCountDimensions(const size_t cDimensions) {
      EBM_ASSERT(cDimensions <= m_cDimensionsMax);
      m_cDimensions = cDimensions;
   }

   TML_INLINE size_t GetStackMemorySizeBytes() {
      EBM_ASSERT(m_cDimensions <= k_cDimensionsMax);
      return sizeof(DimensionInfoStack) * m_cDimensions; // this can't overflow since m_cDimensions is limited in size
   }

   TML_INLINE TDivisions * GetDivisionPointer(const size_t iDimension) {
      EBM_ASSERT(iDimension < m_cDimensions);
      return &m_aDimensions[iDimension].aDivisions[0];
   }

   TML_INLINE TValues * GetValuePointer() {
      return &m_aValues[0];
   }

   TML_INLINE void Reset() {
      for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
         m_aDimensions[iDimension].cDivisions = 0;
      }
      // we only need to set the base case to zero
      // this can't overflow since we previously allocated this memory
      memset(m_aValues, 0, sizeof(TValues) * m_cVectorLength);
      m_bExpanded = false;
   }

   TML_INLINE bool SetCountDivisions(const size_t iDimension, const size_t cDivisions) {
      EBM_ASSERT(iDimension < m_cDimensions);
      DimensionInfo * const pDimension = &m_aDimensions[iDimension];
      EBM_ASSERT(!m_bExpanded || cDivisions <= pDimension->cDivisions); // we shouldn't be able to expand our length after we're been expanded since expanded should be the maximum size already
      if(UNLIKELY(pDimension->cDivisionCapacity < cDivisions)) {
         EBM_ASSERT(!m_bExpanded); // we shouldn't be able to expand our length after we're been expanded since expanded should be the maximum size already

         if(IsAddError(cDivisions, cDivisions >> 1)) {
            LOG(TraceLevelWarning, "WARNING SetCountDivisions IsAddError(cDivisions, cDivisions >> 1)");
            return true;
         }
         size_t cNewDivisionCapacity = cDivisions + (cDivisions >> 1); // just increase it by 50% since we don't expect to grow our divisions often after an initial period, and realloc takes some of the cost of growing away
         LOG(TraceLevelInfo, "SetCountDivisions Growing to size %zu", cNewDivisionCapacity);

         if(IsMultiplyError(sizeof(TDivisions), cNewDivisionCapacity)) {
            LOG(TraceLevelWarning, "WARNING SetCountDivisions IsMultiplyError(sizeof(TDivisions), cNewDivisionCapacity)");
            return true;
         }
         size_t cBytes = sizeof(TDivisions) * cNewDivisionCapacity;
         TDivisions * const aNewDivisions = static_cast<TDivisions *>(realloc(pDimension->aDivisions, cBytes));
         if(UNLIKELY(nullptr == aNewDivisions)) {
            // according to the realloc spec, if realloc fails to allocate the new memory, it returns nullptr BUT the old memory is valid.
            // we leave m_aThreadByteBuffer1 alone in this instance and will free that memory later in the destructor
            LOG(TraceLevelWarning, "WARNING SetCountDivisions nullptr == aNewDivisions");
            return true;
         }
         pDimension->aDivisions = aNewDivisions;
         pDimension->cDivisionCapacity = cNewDivisionCapacity;
      } // never shrink our array unless the user chooses to Trim()
      pDimension->cDivisions = cDivisions;
      return false;
   }

   TML_INLINE bool EnsureValueCapacity(const size_t cValues) {
      if(UNLIKELY(m_cValueCapacity < cValues)) {
         EBM_ASSERT(!m_bExpanded); // we shouldn't be able to expand our length after we're been expanded since expanded should be the maximum size already

         if(IsAddError(cValues, cValues >> 1)) {
            LOG(TraceLevelWarning, "WARNING EnsureValueCapacity IsAddError(cValues, cValues >> 1)");
            return true;
         }
         size_t cNewValueCapacity = cValues + (cValues >> 1); // just increase it by 50% since we don't expect to grow our values often after an initial period, and realloc takes some of the cost of growing away
         LOG(TraceLevelInfo, "EnsureValueCapacity Growing to size %zu", cNewValueCapacity);

         if(IsMultiplyError(sizeof(TValues), cNewValueCapacity)) {
            LOG(TraceLevelWarning, "WARNING EnsureValueCapacity IsMultiplyError(sizeof(TValues), cNewValueCapacity)");
            return true;
         }
         size_t cBytes = sizeof(TValues) * cNewValueCapacity;
         TValues * const aNewValues = static_cast<TValues *>(realloc(m_aValues, cBytes));
         if(UNLIKELY(nullptr == aNewValues)) {
            // according to the realloc spec, if realloc fails to allocate the new memory, it returns nullptr BUT the old memory is valid.
            // we leave m_aThreadByteBuffer1 alone in this instance and will free that memory later in the destructor
            LOG(TraceLevelWarning, "WARNING EnsureValueCapacity nullptr == aNewValues");
            return true;
         }
         m_aValues = aNewValues;
         m_cValueCapacity = cNewValueCapacity;
      } // never shrink our array unless the user chooses to Trim()
      return false;
   }

   TML_INLINE bool Copy(const SegmentedRegionCore & rhs) {
      EBM_ASSERT(m_cDimensions == rhs.m_cDimensions);

      size_t cValues = m_cVectorLength;
      for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
         const DimensionInfo * const pDimension = &rhs.m_aDimensions[iDimension];
         size_t cDivisions = pDimension->cDivisions;
         EBM_ASSERT(!IsMultiplyError(cValues, cDivisions + 1)); // we're copying this memory, so multiplication can't overflow
         cValues *= (cDivisions + 1);
         if(UNLIKELY(SetCountDivisions(iDimension, cDivisions))) {
            LOG(TraceLevelWarning, "WARNING Copy SetCountDivisions(iDimension, cDivisions)");
            return true;
         }
         EBM_ASSERT(!IsMultiplyError(sizeof(TDivisions), cDivisions)); // we're copying this memory, so multiplication can't overflow
         memcpy(m_aDimensions[iDimension].aDivisions, pDimension->aDivisions, sizeof(TDivisions) * cDivisions);
      }
      if(UNLIKELY(EnsureValueCapacity(cValues))) {
         LOG(TraceLevelWarning, "WARNING Copy EnsureValueCapacity(cValues)");
         return true;
      }
      EBM_ASSERT(!IsMultiplyError(sizeof(TValues), cValues)); // we're copying this memory, so multiplication can't overflow
      memcpy(m_aValues, rhs.m_aValues, sizeof(TValues) * cValues);
      m_bExpanded = rhs.m_bExpanded;
      return false;
   }

#ifndef NDEBUG
   TML_INLINE TValues * GetValue(const TDivisions * const aDivisionValue) const {
      if(0 == m_cDimensions) {
         return &m_aValues[0]; // there are no dimensions, and only 1 value
      }
      const DimensionInfo * pDimension = m_aDimensions;
      const TDivisions * pDivisionValue = aDivisionValue;
      const TDivisions * const pDivisionValueEnd = &aDivisionValue[m_cDimensions];
      size_t iValue = 0;
      size_t valueMultiple = m_cVectorLength;

      if(m_bExpanded) {
         while(true) {
            const TDivisions d = *pDivisionValue;
            EBM_ASSERT(!IsMultiplyError(d, valueMultiple)); // we're accessing existing memory, so it can't overflow
            size_t addValue = d * valueMultiple;
            EBM_ASSERT(!IsAddError(addValue, iValue)); // we're accessing existing memory, so it can't overflow
            iValue += addValue;
            ++pDivisionValue;
            if(pDivisionValueEnd == pDivisionValue) {
               break;
            }
            const size_t cDivisions = pDimension->cDivisions;
            EBM_ASSERT(1 <= cDivisions); // since we're expanded we should have at least one division and two values
            EBM_ASSERT(!IsMultiplyError(cDivisions + 1, valueMultiple)); // we're accessing existing memory, so it can't overflow
            valueMultiple *= cDivisions + 1;
            ++pDimension;
         }
      } else {
         // TODO: this code is no longer executed because we always expand our models now.  We can probably get rid of it, but I'm leaving it here for a while to decide if there are really no use cases
         do {
            const size_t cDivisions = pDimension->cDivisions;
            if(LIKELY(0 != cDivisions)) {
               const TDivisions * const aDivisions = pDimension->aDivisions;
               const TDivisions d = *pDivisionValue;
               ptrdiff_t high = cDivisions - 1;
               ptrdiff_t middle;
               ptrdiff_t low = 0;
               TDivisions midVal;
               do {
                  middle = (low + high) >> 1;
                  midVal = aDivisions[middle];
                  if(UNLIKELY(midVal == d)) {
                     // this happens just once during our descent, so it's less likely than continuing searching
                     goto no_check;
                  }
                  high = UNPREDICTABLE(midVal < d) ? high : middle - 1;
                  low = UNPREDICTABLE(midVal < d) ? middle + 1 : low;
               } while(LIKELY(low <= high));
               middle = UNPREDICTABLE(midVal < d) ? middle + 1 : middle;
            no_check:
               EBM_ASSERT(!IsMultiplyError(middle, valueMultiple)); // we're accessing existing memory, so it can't overflow
               ptrdiff_t addValue = middle * valueMultiple;
               EBM_ASSERT(!IsAddError(iValue, addValue)); // we're accessing existing memory, so it can't overflow
               iValue += addValue;
               EBM_ASSERT(!IsMultiplyError(valueMultiple, cDivisions + 1)); // we're accessing existing memory, so it can't overflow
               valueMultiple *= cDivisions + 1;
            }
            ++pDimension;
            ++pDivisionValue;
         } while(pDivisionValueEnd != pDivisionValue);
      }
      return &m_aValues[iValue];
   }
#endif // NDEBUG

   TML_INLINE TValues * GetValueDirect(const size_t index) const {
      EBM_ASSERT(0 == m_cDimensions || m_bExpanded); // this function doesn't make sense unless the underlying data has been expanded
      EBM_ASSERT(!IsMultiplyError(index, m_cVectorLength)); // we're accessing existing memory, so it can't overflow
      return &m_aValues[index * m_cVectorLength];
   }

   TML_INLINE void Multiply(const TValues v) {
      size_t cValues = 1;
      for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
         EBM_ASSERT(!IsMultiplyError(cValues, m_aDimensions[iDimension].cDivisions + 1)); // we're accessing existing memory, so it can't overflow
         cValues *= m_aDimensions[iDimension].cDivisions + 1;
      }

      TValues * pCur = &m_aValues[0];
      TValues * pEnd = &m_aValues[cValues * m_cVectorLength];
      // we always have 1 value, even if we have zero divisions
      do {
         *pCur *= v;
         ++pCur;
      } while(pEnd != pCur);
   }

   bool Expand(const size_t * const acDivisionsPlusOne) {
      LOG(TraceLevelVerbose, "Entered Expand");

      EBM_ASSERT(1 <= m_cDimensions); // you can't really expand something with zero dimensions
      EBM_ASSERT(nullptr != acDivisionsPlusOne);
      // ok, checking the max isn't really the best here, but doing this right seems pretty complicated, and this should detect any real problems.
      // don't make this a static assert.  The rest of our class is fine as long as Expand is never called
      // TODO : see if we can change this back to size_t somehow.  I remember we got negative numbers some places either here or in the Add function and that why I used ptrdiff_t, but if it's just -1, then we can still use size_t.
      EBM_ASSERT(std::numeric_limits<ptrdiff_t>::max() == std::numeric_limits<TDivisions>::max() && std::numeric_limits<ptrdiff_t>::min() == std::numeric_limits<TDivisions>::min());
      if(m_bExpanded) {
         // we're already expanded
         LOG(TraceLevelVerbose, "Exited Expand");
         return false;
      }

      EBM_ASSERT(m_cDimensions <= k_cDimensionsMax);
      DimensionInfoStackExpand aDimensionInfoStackExpand[k_cDimensionsMax];

      const DimensionInfo * pDimensionFirst1 = m_aDimensions;

      DimensionInfoStackExpand * pDimensionInfoStackFirst = aDimensionInfoStackExpand;
      const DimensionInfoStackExpand * const pDimensionInfoStackEnd = &aDimensionInfoStackExpand[m_cDimensions];
      const size_t * pcDivisionsPlusOne = acDivisionsPlusOne;

      size_t cValues1 = 1;
      size_t cNewValues = 1;

      EBM_ASSERT(0 < m_cDimensions);
      // first, get basic counts of how many divisions and values we'll have in our final result
      do {
         const size_t cDivisions1 = pDimensionFirst1->cDivisions;

         EBM_ASSERT(!IsMultiplyError(cValues1, cDivisions1 + 1)); // we check for simple multiplication overflow from m_cStates in TmlTrainingState->Initialize when we unpack attributeCombinationIndexes
         cValues1 *= cDivisions1 + 1;
         const TDivisions * const p1End = &pDimensionFirst1->aDivisions[cDivisions1];

         pDimensionInfoStackFirst->pDivision1 = p1End - 1;
         const size_t cDivisionsPlusOne = *pcDivisionsPlusOne;
         EBM_ASSERT(!IsMultiplyError(cNewValues, cDivisionsPlusOne)); // we check for simple multiplication overflow from m_cStates in TmlTrainingState->Initialize when we unpack attributeCombinationIndexes
         cNewValues *= cDivisionsPlusOne;
         const size_t maxDivision = cDivisionsPlusOne - 2;

         pDimensionInfoStackFirst->iDivision2 = maxDivision;
         pDimensionInfoStackFirst->cNewDivisions = maxDivision; // maxDivision can be a negative number, but I think it isn't used below if it's negative, so converting it to size_t is ok here

         ++pDimensionFirst1;
         ++pcDivisionsPlusOne;
         ++pDimensionInfoStackFirst;
      } while(pDimensionInfoStackEnd != pDimensionInfoStackFirst);

      if(IsMultiplyError(cNewValues, m_cVectorLength)) {
         LOG(TraceLevelWarning, "WARNING Expand IsMultiplyError(cNewValues, m_cVectorLength)");
         return true;
      }
      // call EnsureValueCapacity before using the m_aValues pointer since m_aValues might change inside EnsureValueCapacity
      if(UNLIKELY(EnsureValueCapacity(cNewValues * m_cVectorLength))) {
         LOG(TraceLevelWarning, "WARNING Expand EnsureValueCapacity(cNewValues * m_cVectorLength))");
         return true;
      }

      TValues * const aValues = m_aValues;
      const DimensionInfo * const aDimension1 = m_aDimensions;

      EBM_ASSERT(cValues1 <= cNewValues);
      EBM_ASSERT(!IsMultiplyError(m_cVectorLength, cValues1)); // we checked against cNewValues above, and cValues1 should be smaller
      const TValues * pValue1 = &aValues[m_cVectorLength * cValues1 - 1];
      TValues * pValueTop = &aValues[m_cVectorLength * cNewValues - 1];

      const TValues * const aValuesEnd = aValues - 1;

      // traverse the values in reverse so that we can put our results at the higher order indexes where we are guaranteed not to overwrite our existing values which we still need to copy
      // first do the values because we need to refer to the old divisions when making decisions about where to move next
      while(true) {
         const TValues * pValue1Move = pValue1;
         const TValues * const pValueTopEnd = pValueTop - m_cVectorLength;
         do {
            *pValueTop = *pValue1Move;
            --pValue1Move;
            --pValueTop;
         } while(pValueTopEnd != pValueTop);

         // For a single dimensional SegmentedRegion checking here is best.  
         // For two or higher dimensions, we could instead check inside our loop below for when we reach the end of the pDimensionInfoStack, thus eliminating the check on most loops.  
         // we'll spend most of our time working on single features though, so we optimize for that case, but if we special cased the single dimensional case, then we would want 
         // to move this check into the loop below in the case of multi-dimensioncal SegmentedRegions
         if(UNLIKELY(aValuesEnd == pValueTop)) {
            // we've written our final tensor cell, so we're done
            break;
         }

         DimensionInfoStackExpand * pDimensionInfoStackSecond = aDimensionInfoStackExpand;
         const DimensionInfo * pDimensionSecond1 = aDimension1;

         size_t multiplication1 = m_cVectorLength;

         while(true) {
            const TDivisions * const pDivision1 = pDimensionInfoStackSecond->pDivision1;
            ptrdiff_t iDivision2 = pDimensionInfoStackSecond->iDivision2;

            TDivisions * const aDivisions1 = pDimensionSecond1->aDivisions;

            if(UNPREDICTABLE(aDivisions1 <= pDivision1)) {
               EBM_ASSERT(0 <= iDivision2);
               const TDivisions d1 = *pDivision1;

               const size_t change1 = UNPREDICTABLE(iDivision2 <= d1) ? 1 : 0;
               pDimensionInfoStackSecond->pDivision1 = pDivision1 - change1;
               // this doesn't need to be checked since change1 is either 0 or 1
               // TODO: let's use a conditional here since it's either zero or 1
               pValue1 -= change1 * multiplication1;

               pDimensionInfoStackSecond->iDivision2 = iDivision2 - 1;
               break;
            } else {
               if(UNPREDICTABLE(0 <= iDivision2)) {
                  --iDivision2;
                  pDimensionInfoStackSecond->iDivision2 = iDivision2;
                  break;
               } else {
                  pValue1 -= multiplication1; // put us before the beginning.  We'll add the full row first

                  const size_t cDivisions1 = pDimensionSecond1->cDivisions;

                  EBM_ASSERT(!IsMultiplyError(multiplication1, 1 + cDivisions1)); // we're already allocated values, so this is accessing what we've already allocated, so it must not overflow
                  multiplication1 *= 1 + cDivisions1;

                  pValue1 += multiplication1; // go to the last valid entry back to where we started.  If we don't move down a set, then we re-do this set of numbers

                  pDimensionInfoStackSecond->pDivision1 = &aDivisions1[static_cast<ptrdiff_t>(cDivisions1) - 1];
                  pDimensionInfoStackSecond->iDivision2 = pDimensionInfoStackSecond->cNewDivisions;

                  ++pDimensionSecond1;
                  ++pDimensionInfoStackSecond;
                  continue;
               }
            }
         }
      }

      EBM_ASSERT(pValueTop == m_aValues - 1);
      EBM_ASSERT(pValue1 == m_aValues + m_cVectorLength - 1);

      for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
         const size_t cDivisions = acDivisionsPlusOne[iDimension] - 1;

         if(cDivisions == m_aDimensions[iDimension].cDivisions) {
            continue;
         }

         if(UNLIKELY(SetCountDivisions(iDimension, cDivisions))) {
            LOG(TraceLevelWarning, "WARNING Expand SetCountDivisions(iDimension, cDivisions)");
            return true;
         }

         for(size_t iDivision = 0; iDivision < cDivisions; ++iDivision) {
            m_aDimensions[iDimension].aDivisions[iDivision] = iDivision;
         }
      }

      m_bExpanded = true;
      LOG(TraceLevelVerbose, "Exited Expand");
      return false;
   }

   // TODO : change this to eliminate pStackMemory and replace it with true on stack memory (we know that there can't be more than 63 dimensions)
   // TODO : consider adding templated cVectorLength and cDimensions to this function.  At worst someone can pass in 0 and use the loops without needing to super-optimize it
   bool Add(const SegmentedRegionCore & rhs, void * pStackMemory) {
      EBM_ASSERT(nullptr != pStackMemory);
      EBM_ASSERT(m_cDimensions == rhs.m_cDimensions);

      if(0 == m_cDimensions) {
         EBM_ASSERT(1 <= m_cValueCapacity);
         EBM_ASSERT(nullptr != m_aValues);

         TValues * pTo = &m_aValues[0];
         const TValues * pFrom = &rhs.m_aValues[0];
         const TValues * const pToEnd = &pTo[m_cVectorLength];
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

      const DimensionInfo * pDimensionFirst1 = m_aDimensions;
      const DimensionInfo * pDimensionFirst2 = rhs.m_aDimensions;

      DimensionInfoStack * pDimensionInfoStackFirst = reinterpret_cast<DimensionInfoStack *>(pStackMemory);
      const DimensionInfoStack * const pDimensionInfoStackEnd = &pDimensionInfoStackFirst[m_cDimensions];

      size_t cValues1 = 1;
      size_t cValues2 = 1;
      size_t cNewValues = 1;

      EBM_ASSERT(0 < m_cDimensions);
      // first, get basic counts of how many divisions and values we'll have in our final result
      do {
         const size_t cDivisions1 = pDimensionFirst1->cDivisions;
         TDivisions * p1Cur = pDimensionFirst1->aDivisions;
         const size_t cDivisions2 = pDimensionFirst2->cDivisions;
         TDivisions * p2Cur = pDimensionFirst2->aDivisions;

         cValues1 *= cDivisions1 + 1;
         cValues2 *= cDivisions2 + 1;

         TDivisions * const p1End = &p1Cur[cDivisions1];
         TDivisions * const p2End = &p2Cur[cDivisions2];

         pDimensionInfoStackFirst->pDivision1 = p1End - 1;
         pDimensionInfoStackFirst->pDivision2 = p2End - 1;

         size_t cNewSingleDimensionDivisions = 0;

         // processing forwards here is slightly faster in terms of cache fetch efficiency.  We'll then be guaranteed to have the divisions at least in the cache, which will be benefitial when traversing backwards later below
         while(true) {
            if(UNLIKELY(p2End == p2Cur)) {
               // check the other array first.  Most of the time the other array will be shorter since we'll be adding
               // a sequence of Segmented lines and our main line will be in *this, and there will be more segments in general for
               // a line that is added to a lot
               cNewSingleDimensionDivisions += p1End - p1Cur;
               break;
            }
            if(UNLIKELY(p1End == p1Cur)) {
               cNewSingleDimensionDivisions += p2End - p2Cur;
               break;
            }
            ++cNewSingleDimensionDivisions; // if we move one or both pointers, we just added annother unique one

            const TDivisions d1 = *p1Cur;
            const TDivisions d2 = *p2Cur;

            p1Cur = UNPREDICTABLE(d1 <= d2) ? p1Cur + 1 : p1Cur;
            p2Cur = UNPREDICTABLE(d2 <= d1) ? p2Cur + 1 : p2Cur;
         }
         pDimensionInfoStackFirst->cNewDivisions = cNewSingleDimensionDivisions;
         EBM_ASSERT(!IsMultiplyError(cNewValues, cNewSingleDimensionDivisions + 1)); // we check for simple multiplication overflow from m_cStates in TmlTrainingState->Initialize when we unpack attributeCombinationIndexes
         cNewValues *= cNewSingleDimensionDivisions + 1;

         ++pDimensionFirst1;
         ++pDimensionFirst2;

         ++pDimensionInfoStackFirst;
      } while(pDimensionInfoStackEnd != pDimensionInfoStackFirst);

      if(IsMultiplyError(cNewValues, m_cVectorLength)) {
         LOG(TraceLevelWarning, "WARNING Add IsMultiplyError(cNewValues, m_cVectorLength)");
         return true;
      }
      // call EnsureValueCapacity before using the m_aValues pointer since m_aValues might change inside EnsureValueCapacity
      if(UNLIKELY(EnsureValueCapacity(cNewValues * m_cVectorLength))) {
         LOG(TraceLevelWarning, "WARNING Add EnsureValueCapacity(cNewValues * m_cVectorLength)");
         return true;
      }

      const TValues * pValue2 = &rhs.m_aValues[m_cVectorLength * cValues2 - 1];  // we're accessing allocated memory, so it can't overflow
      const DimensionInfo * const aDimension2 = rhs.m_aDimensions;

      TValues * const aValues = m_aValues;
      const DimensionInfo * const aDimension1 = m_aDimensions;

      const TValues * pValue1 = &aValues[m_cVectorLength * cValues1 - 1]; // we're accessing allocated memory, so it can't overflow
      TValues * pValueTop = &aValues[m_cVectorLength * cNewValues - 1]; // we're accessing allocated memory, so it can't overflow

      const TValues * const aValuesEnd = aValues - 1;

      // traverse the values in reverse so that we can put our results at the higher order indexes where we are guaranteed not to overwrite our existing values which we still need to copy
      // first do the values because we need to refer to the old divisions when making decisions about where to move next
      while(true) {
         const TValues * pValue1Move = pValue1;
         const TValues * pValue2Move = pValue2;
         const TValues * const pValueTopEnd = pValueTop - m_cVectorLength;
         do {
            *pValueTop = *pValue1Move + *pValue2Move;
            --pValue1Move;
            --pValue2Move;
            --pValueTop;
         } while(pValueTopEnd != pValueTop);

         // For a single dimensional SegmentedRegion checking here is best.  
         // For two or higher dimensions, we could instead check inside our loop below for when we reach the end of the pDimensionInfoStack, thus eliminating the check on most loops.  
         // we'll spend most of our time working on single features though, so we optimize for that case, but if we special cased the single dimensional case, then we would want 
         // to move this check into the loop below in the case of multi-dimensioncal SegmentedRegions
         if(UNLIKELY(aValuesEnd == pValueTop)) {
            // we've written our final tensor cell, so we're done
            break;
         }

         DimensionInfoStack * pDimensionInfoStackSecond = reinterpret_cast<DimensionInfoStack *>(pStackMemory);
         const DimensionInfo * pDimensionSecond1 = aDimension1;
         const DimensionInfo * pDimensionSecond2 = aDimension2;

         size_t multiplication1 = m_cVectorLength;
         size_t multiplication2 = m_cVectorLength;

         while(true) {
            const TDivisions * const pDivision1 = pDimensionInfoStackSecond->pDivision1;
            const TDivisions * const pDivision2 = pDimensionInfoStackSecond->pDivision2;

            TDivisions * const aDivisions1 = pDimensionSecond1->aDivisions;
            TDivisions * const aDivisions2 = pDimensionSecond2->aDivisions;

            if(UNPREDICTABLE(aDivisions1 <= pDivision1)) {
               if(UNPREDICTABLE(aDivisions2 <= pDivision2)) {
                  const TDivisions d1 = *pDivision1;
                  const TDivisions d2 = *pDivision2;

                  const size_t change1 = UNPREDICTABLE(d2 <= d1) ? 1 : 0;
                  pDimensionInfoStackSecond->pDivision1 = pDivision1 - change1;
                  pValue1 -= change1 * multiplication1;

                  const size_t change2 = UNPREDICTABLE(d1 <= d2) ? 1 : 0;
                  pDimensionInfoStackSecond->pDivision2 = pDivision2 - change2;
                  pValue2 -= change2 * multiplication2;
                  break;
               } else {
                  pValue1 -= multiplication1;
                  pDimensionInfoStackSecond->pDivision1 = pDivision1 - 1;
                  break;
               }
            } else {
               if(UNPREDICTABLE(aDivisions2 <= pDivision2)) {
                  pValue2 -= multiplication2;
                  pDimensionInfoStackSecond->pDivision2 = pDivision2 - 1;
                  break;
               } else {
                  pValue1 -= multiplication1; // put us before the beginning.  We'll add the full row first
                  pValue2 -= multiplication2; // put us before the beginning.  We'll add the full row first

                  const size_t cDivisions1 = pDimensionSecond1->cDivisions;
                  const size_t cDivisions2 = pDimensionSecond2->cDivisions;

                  EBM_ASSERT(!IsMultiplyError(multiplication1, 1 + cDivisions1)); // we're accessing allocated memory, so it can't overflow
                  multiplication1 *= 1 + cDivisions1;
                  EBM_ASSERT(!IsMultiplyError(multiplication2, 1 + cDivisions2)); // we're accessing allocated memory, so it can't overflow
                  multiplication2 *= 1 + cDivisions2;

                  pValue1 += multiplication1; // go to the last valid entry back to where we started.  If we don't move down a set, then we re-do this set of numbers
                  pValue2 += multiplication2; // go to the last valid entry back to where we started.  If we don't move down a set, then we re-do this set of numbers

                  pDimensionInfoStackSecond->pDivision1 = &aDivisions1[static_cast<ptrdiff_t>(cDivisions1) - 1];
                  pDimensionInfoStackSecond->pDivision2 = &aDivisions2[static_cast<ptrdiff_t>(cDivisions2) - 1];
                  ++pDimensionSecond1;
                  ++pDimensionSecond2;
                  ++pDimensionInfoStackSecond;
                  continue;
               }
            }
         }
      }

      EBM_ASSERT(pValueTop == m_aValues - 1);
      EBM_ASSERT(pValue1 == m_aValues + m_cVectorLength - 1);
      EBM_ASSERT(pValue2 == rhs.m_aValues + m_cVectorLength - 1);

      // now finally do the divisions

      const DimensionInfoStack * pDimensionInfoStackCur = reinterpret_cast<DimensionInfoStack *>(pStackMemory);
      const DimensionInfo * pDimension1Cur = aDimension1;
      const DimensionInfo * pDimension2Cur = aDimension2;
      size_t iDimension = 0;
      do {
         const size_t cNewDivisions = pDimensionInfoStackCur->cNewDivisions;
         const size_t cOriginalDivisionsBeforeSetting = pDimension1Cur->cDivisions;
         
         // this will increase our capacity, if required.  It will also change m_cDivisions, so we get that before calling it.  SetCountDivisions might change m_aValuesAndDivisions, so we need to actually keep it here after getting m_cDivisions but before set set all our pointers
         if(UNLIKELY(SetCountDivisions(iDimension, cNewDivisions))) {
            LOG(TraceLevelWarning, "WARNING Add SetCountDivisions(iDimension, cNewDivisions)");
            return true;
         }
         
         const TDivisions * p1Cur = &pDimension1Cur->aDivisions[static_cast<ptrdiff_t>(cOriginalDivisionsBeforeSetting) - 1];
         const TDivisions * p2Cur = &pDimension2Cur->aDivisions[static_cast<ptrdiff_t>(pDimension2Cur->cDivisions) - 1];
         TDivisions * pTopCur = &pDimension1Cur->aDivisions[static_cast<ptrdiff_t>(cNewDivisions) - 1];
         const ptrdiff_t diffDivisions = reinterpret_cast<char *>(pDimension2Cur->aDivisions) - reinterpret_cast<char *>(pDimension1Cur->aDivisions); // these arrays are not guaranteed to be aligned with each other, so convert to byte pointers

         // traverse in reverse so that we can put our results at the higher order indexes where we are guaranteed not to overwrite our existing values which we still need to copy
         while(true) {
            EBM_ASSERT(&pDimension1Cur->aDivisions[-1] <= pTopCur); // -1 can happen if both our SegmentedRegions have zero divisions
            EBM_ASSERT(&pDimension1Cur->aDivisions[-1] <= p1Cur);
            EBM_ASSERT(&pDimension2Cur->aDivisions[-1] <= p2Cur);
            EBM_ASSERT(p1Cur <= pTopCur);
            EBM_ASSERT(reinterpret_cast<const char *>(p2Cur) <= reinterpret_cast<const char *>(pTopCur) + diffDivisions);

            if(UNLIKELY(pTopCur == p1Cur)) {
               // since we've finished the rhs divisions, our SegmentedRegion already has the right divisions in place, so all we need is to add the value of the last region in rhs to our remaining values
               break;
            }
            // pTopCur is an index above pDimension1Cur->aDivisions.  p2Cur is an index above pDimension2Cur->aDivisions.  We want to decide if they are at the same index above their respective arrays.  Adding diffDivisions to a pointer that references an index in pDimension1Cur->aDivisions turns it into a pointer indexed from pDimension2Cur->aDivisions.  They both point to TValues items, so we can cross reference them this way
            if(UNLIKELY(reinterpret_cast<const char *>(pTopCur) + diffDivisions == reinterpret_cast<const char *>(p2Cur))) {
               EBM_ASSERT(pDimension1Cur->aDivisions <= pTopCur);
               // direct copy the remaining divisions.  There should be at least one
               memcpy(pDimension1Cur->aDivisions, pDimension2Cur->aDivisions, (pTopCur - pDimension1Cur->aDivisions + 1) * sizeof(TDivisions));
               break;
            }

            const TDivisions d1 = *p1Cur;
            const TDivisions d2 = *p2Cur;

            p1Cur = UNPREDICTABLE(d2 <= d1) ? p1Cur - 1 : p1Cur;
            p2Cur = UNPREDICTABLE(d1 <= d2) ? p2Cur - 1 : p2Cur;

            const TDivisions d = UNPREDICTABLE(d1 <= d2) ? d2 : d1;

            *pTopCur = d;
            --pTopCur; // if we move one or both pointers, we just added annother unique one
         }
         ++pDimension1Cur;
         ++pDimension2Cur;
         ++pDimensionInfoStackCur;
         ++iDimension;
      } while(iDimension != m_cDimensions);

      return false;
   }

#ifndef NDEBUG
   bool IsEqual(const SegmentedRegionCore & rhs) const {
      if(m_cDimensions != rhs.m_cDimensions) {
         return false;
      }

      size_t cValues = m_cVectorLength;
      for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
         const DimensionInfo * const pDimension1 = &m_aDimensions[iDimension];
         const DimensionInfo * const pDimension2 = &rhs.m_aDimensions[iDimension];

         size_t cDivisions = pDimension1->cDivisions;
         if(cDivisions != pDimension2->cDivisions) {
            return false;
         }

         if(0 != cDivisions) {
            EBM_ASSERT(!IsMultiplyError(cValues, cDivisions + 1)); // we're accessing allocated memory, so it can't overflow
            cValues *= cDivisions + 1;

            const TDivisions * pD1Cur = pDimension1->aDivisions;
            const TDivisions * pD2Cur = pDimension2->aDivisions;
            const TDivisions * const pD1End = pD1Cur + cDivisions;
            do {
               if(UNLIKELY(*pD1Cur != *pD2Cur)) {
                  return false;
               }
               ++pD1Cur;
               ++pD2Cur;
            } while(LIKELY(pD1End != pD1Cur));
         }
      }

      const TValues * pV1Cur = &m_aValues[0];
      const TValues * pV2Cur = &rhs.m_aValues[0];
      const TValues * const pV1End = pV1Cur + cValues;
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

   static_assert(std::is_pod<TDivisions>::value, "SegmentedRegion must be POD (Plain Old Data).  We use realloc, which isn't compatible with using complex classes.  Interop data must also be PODs.  Lastly, we put this class into a union, so the destructor would need to be called manually anyways");
   static_assert(std::is_pod<TValues>::value, "SegmentedRegion must be POD (Plain Old Data).  We use realloc, which isn't compatible with using complex classes.  Interop data must also be PODs.  Lastly, we put this class into a union, so the destructor would need to be called manually anyways");
};
// SegmentedRegion must be a POD, which it will be if both our D and V types are PODs and SegmentedRegion<char, char> is a POD
static_assert(std::is_pod<SegmentedRegionCore<char, char>>::value, "SegmentedRegion must be POD (Plain Old Data).  We use realloc, which isn't compatible with using complex classes.  Interop data must also be PODs.  Lastly, we put this class into a union, so the destructor needs to be called manually anyways");

#endif // SEGMENTED_REGION_H
