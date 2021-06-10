// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <type_traits> // std::is_standard_layout
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "SegmentedTensor.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

SegmentedTensor * SegmentedTensor::Allocate(const size_t cDimensionsMax, const size_t cVectorLength) {
   EBM_ASSERT(cDimensionsMax <= k_cDimensionsMax);
   EBM_ASSERT(1 <= cVectorLength); // having 0 classes makes no sense, and having 1 class is useless

   if(IsMultiplyError(cVectorLength, k_initialValueCapacity)) {
      LOG_0(TraceLevelWarning, "WARNING Allocate IsMultiplyError(cVectorLength, k_initialValueCapacity)");
      return nullptr;
   }
   const size_t cValueCapacity = cVectorLength * k_initialValueCapacity;

   // this can't overflow since cDimensionsMax can't be bigger than k_cDimensionsMax, which is arround 64
   const size_t cBytesSegmentedRegion = sizeof(SegmentedTensor) - sizeof(DimensionInfo) + sizeof(DimensionInfo) * cDimensionsMax;
   SegmentedTensor * const pSegmentedRegion = EbmMalloc<SegmentedTensor>(1, cBytesSegmentedRegion);
   if(UNLIKELY(nullptr == pSegmentedRegion)) {
      LOG_0(TraceLevelWarning, "WARNING Allocate nullptr == pSegmentedRegion");
      return nullptr;
   }

   pSegmentedRegion->m_cVectorLength = cVectorLength;
   pSegmentedRegion->m_cDimensionsMax = cDimensionsMax;
   pSegmentedRegion->m_cDimensions = cDimensionsMax;
   pSegmentedRegion->m_cValueCapacity = cValueCapacity;
   pSegmentedRegion->m_bExpanded = false;

   FloatEbmType * const aValues = EbmMalloc<FloatEbmType>(cValueCapacity);
   if(UNLIKELY(nullptr == aValues)) {
      LOG_0(TraceLevelWarning, "WARNING Allocate nullptr == aValues");
      free(pSegmentedRegion); // don't need to call the full Free(*) yet
      return nullptr;
   }
   pSegmentedRegion->m_aValues = aValues;
   // we only need to set the base case to zero, not our entire initial allocation
   // we checked for cVectorLength * k_initialValueCapacity * sizeof(FloatEbmType), and 1 <= k_initialValueCapacity, 
   // so sizeof(FloatEbmType) * cVectorLength can't overflow
   for(size_t i = 0; i < cVectorLength; ++i) {
      aValues[i] = FloatEbmType { 0 };
   }

   if(0 != cDimensionsMax) {
      DimensionInfo * pDimension = pSegmentedRegion->GetDimensions();
      size_t iDimension = 0;
      do {
         pDimension->m_cDivisions = 0;
         pDimension->m_cDivisionCapacity = k_initialDivisionCapacity;
         pDimension->m_aDivisions = nullptr;
         ++pDimension;
         ++iDimension;
      } while(iDimension < cDimensionsMax);

      pDimension = pSegmentedRegion->GetDimensions();
      iDimension = 0;
      do {
         ActiveDataType * const aDivisions = EbmMalloc<ActiveDataType>(k_initialDivisionCapacity);
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

void SegmentedTensor::Free(SegmentedTensor * const pSegmentedRegion) {
   if(LIKELY(nullptr != pSegmentedRegion)) {
      free(pSegmentedRegion->m_aValues);
      if(LIKELY(0 != pSegmentedRegion->m_cDimensionsMax)) {
         const DimensionInfo * pDimensionInfo = pSegmentedRegion->GetDimensions();
         const DimensionInfo * const pDimensionInfoEnd = &pDimensionInfo[pSegmentedRegion->m_cDimensionsMax];
         do {
            free(pDimensionInfo->m_aDivisions);
            ++pDimensionInfo;
         } while(pDimensionInfoEnd != pDimensionInfo);
      }
      free(pSegmentedRegion);
   }
}

void SegmentedTensor::Reset() {
   DimensionInfo * pDimensionInfo = GetDimensions();
   for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
      pDimensionInfo[iDimension].m_cDivisions = 0;
   }
   // we only need to set the base case to zero
   // this can't overflow since we previously allocated this memory
   for(size_t i = 0; i < m_cVectorLength; ++i) {
      m_aValues[i] = FloatEbmType { 0 };
   }
   m_bExpanded = false;
}

bool SegmentedTensor::SetCountDivisions(const size_t iDimension, const size_t cDivisions) {
   EBM_ASSERT(iDimension < m_cDimensions);
   DimensionInfo * const pDimension = &GetDimensions()[iDimension];
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

bool SegmentedTensor::EnsureValueCapacity(const size_t cValues) {
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

bool SegmentedTensor::Copy(const SegmentedTensor & rhs) {
   EBM_ASSERT(m_cDimensions == rhs.m_cDimensions);

   const DimensionInfo * pThisDimensionInfo = GetDimensions();
   const DimensionInfo * pRhsDimensionInfo = rhs.GetDimensions();

   size_t cValues = m_cVectorLength;
   for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
      const DimensionInfo * const pDimension = &pRhsDimensionInfo[iDimension];
      size_t cDivisions = pDimension->m_cDivisions;
      EBM_ASSERT(!IsMultiplyError(cValues, cDivisions + 1)); // we're copying this memory, so multiplication can't overflow
      cValues *= (cDivisions + 1);
      if(UNLIKELY(SetCountDivisions(iDimension, cDivisions))) {
         LOG_0(TraceLevelWarning, "WARNING Copy SetCountDivisions(iDimension, cDivisions)");
         return true;
      }
      EBM_ASSERT(!IsMultiplyError(sizeof(ActiveDataType), cDivisions)); // we're copying this memory, so multiplication can't overflow
      memcpy(pThisDimensionInfo[iDimension].m_aDivisions, pDimension->m_aDivisions, sizeof(ActiveDataType) * cDivisions);
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

bool SegmentedTensor::MultiplyAndCheckForIssues(const FloatEbmType v) {

   const DimensionInfo * pThisDimensionInfo = GetDimensions();

   size_t cValues = 1;
   for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
      // we're accessing existing memory, so it can't overflow
      EBM_ASSERT(!IsMultiplyError(cValues, pThisDimensionInfo[iDimension].m_cDivisions + 1));
      cValues *= pThisDimensionInfo[iDimension].m_cDivisions + 1;
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
      // TODO: another issue is that isnan and isinf don't work on some compilers with some compiler settings
      bBad |= std::isnan(val) || std::isinf(val);
      *pCur = val;
      ++pCur;
   } while(pEnd != pCur);
   return !!bBad;
}

bool SegmentedTensor::Expand(const FeatureGroup * const pFeatureGroup) {
   // checking the max isn't really the best here, but doing this right seems pretty complicated
   static_assert(std::numeric_limits<size_t>::max() <= std::numeric_limits<ActiveDataType>::max() &&
      0 == std::numeric_limits<ActiveDataType>::min(), "bad AcitveDataType size");

   LOG_0(TraceLevelVerbose, "Entered Expand");

   if(m_bExpanded) {
      // we're already expanded
      LOG_0(TraceLevelVerbose, "Exited Expand");
      return false;
   }

   EBM_ASSERT(nullptr != pFeatureGroup);
   const size_t cDimensions = pFeatureGroup->GetCountDimensions();
   if(size_t { 0 } != cDimensions) {
      const FeatureGroupEntry * pFeatureGroupEntry1 = pFeatureGroup->GetFeatureGroupEntries();
      const FeatureGroupEntry * const pFeatureGroupEntryEnd = pFeatureGroupEntry1 + cDimensions;
      DimensionInfoStackExpand aDimensionInfoStackExpand[k_cDimensionsMax];
      DimensionInfoStackExpand * pDimensionInfoStackFirst = aDimensionInfoStackExpand;
      const DimensionInfo * pDimensionFirst1 = GetDimensions();
      size_t cValues1 = 1;
      size_t cNewValues = 1;

      // first, get basic counts of how many divisions and values we'll have in our final result
      do {
         const size_t cBins = pFeatureGroupEntry1->m_pFeature->GetCountBins();

         // we check for simple multiplication overflow from m_cBins in Booster::Initialize when we unpack 
         // featureGroupsFeatureIndexes and in CalculateInteractionScore for interactions
         EBM_ASSERT(!IsMultiplyError(cNewValues, cBins));
         cNewValues *= cBins;

         if(size_t { 1 } < cBins) {
            // strip any dimensions which have 1 bin since the tensor shape doesn't change and we 
            // have limited stack memory to store dimension information

            const size_t cDivisions1 = pDimensionFirst1->m_cDivisions;

            EBM_ASSERT(!IsMultiplyError(cValues1, cDivisions1 + 1)); // this is accessing existing memory, so it can't overflow
            cValues1 *= cDivisions1 + 1;

            pDimensionInfoStackFirst->m_pDivision1 = &pDimensionFirst1->m_aDivisions[cDivisions1];

            const size_t cCuts = cBins - size_t { 1 };
            pDimensionInfoStackFirst->m_iDivision2 = cCuts;
            pDimensionInfoStackFirst->m_cNewDivisions = cCuts;

            ++pDimensionFirst1;
            ++pDimensionInfoStackFirst;
         }
         ++pFeatureGroupEntry1;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry1);
      
      if(size_t { 0 } == cNewValues) {
         // there's a really degenerate case where we have zero training and zero validation samples, and the user 
         // specifies zero bins which is legal since there are no bins in the training or validation, in this case
         // the tensor has zero bins in one dimension, so there are zero bins in the entire tensor.  In this case
         // the dimension is still stripped from our view, but we should not expand
         LOG_0(TraceLevelWarning, "WARNING Expand Zero sized tensor");
      } else {
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
         const DimensionInfo * const aDimension1 = GetDimensions();

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
               EBM_ASSERT(aValues <= pValue1Move);
               EBM_ASSERT(aValues <= pValueTop);
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

         const FeatureGroupEntry * pFeatureGroupEntry2 = pFeatureGroup->GetFeatureGroupEntries();
         size_t iDimension = 0;
         do {
            const size_t cBins = pFeatureGroupEntry2->m_pFeature->GetCountBins();
            EBM_ASSERT(size_t { 1 } <= cBins); // we exited above on tensors with zero bins in any dimension
            const size_t cDivisions = cBins - size_t { 1 };
            if(size_t { 0 } < cDivisions) {
               // strip any dimensions which have 1 bin since the tensor shape doesn't change and we 
               // have limited stack memory to store dimension information

               const DimensionInfo * const pDimension = &aDimension1[iDimension];
               if(cDivisions != pDimension->m_cDivisions) {
                  if(UNLIKELY(SetCountDivisions(iDimension, cDivisions))) {
                     LOG_0(TraceLevelWarning, "WARNING Expand SetCountDivisions(iDimension, cDivisions)");
                     return true;
                  }

                  ActiveDataType * const aDivision = pDimension->m_aDivisions;
                  size_t iDivision = 0;
                  do {
                     aDivision[iDivision] = iDivision;
                     ++iDivision;
                  } while(cDivisions != iDivision);
               }
               ++iDimension;
            }
            ++pFeatureGroupEntry2;
         } while(pFeatureGroupEntryEnd != pFeatureGroupEntry2);
      }
   }
   m_bExpanded = true;
   
   LOG_0(TraceLevelVerbose, "Exited Expand");
   return false;
}

void SegmentedTensor::AddExpandedWithBadValueProtection(const FloatEbmType * const aFromValues) {
   EBM_ASSERT(m_bExpanded);
   size_t cItems = m_cVectorLength;

   const DimensionInfo * const aDimension = GetDimensions();

   for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
      // this can't overflow since we've already allocated them!
      cItems *= aDimension[iDimension].m_cDivisions + 1;
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
      // even so far as to make isinf always FALSE with some compiler flags
      // include the equals case so that the compiler is less likely to optimize that out
      val = val <= std::numeric_limits<FloatEbmType>::lowest() ? std::numeric_limits<FloatEbmType>::lowest() : val;
      // this is a check for +infinity, without the +infinity value since some compilers make that illegal
      // even so far as to make isinf always FALSE with some compiler flags
      // include the equals case so that the compiler is less likely to optimize that out
      val = std::numeric_limits<FloatEbmType>::max() <= val ? std::numeric_limits<FloatEbmType>::max() : val;
      *pToValue = val;
      ++pFromValue;
      ++pToValue;
   } while(pToValueEnd != pToValue);
}

// TODO : consider adding templated cVectorLength and cDimensions to this function.  At worst someone can pass in 0 and use the loops 
//   without needing to super-optimize it
bool SegmentedTensor::Add(const SegmentedTensor & rhs) {
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

   const DimensionInfo * pDimensionFirst1 = GetDimensions();
   const DimensionInfo * pDimensionFirst2 = rhs.GetDimensions();

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
      // we check for simple multiplication overflow from m_cBins in Booster::Initialize when we unpack featureGroupsFeatureIndexes and in 
      // CalculateInteractionScore for interactions
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
   const DimensionInfo * const aDimension2 = rhs.GetDimensions();

   FloatEbmType * const aValues = m_aValues;
   const DimensionInfo * const aDimension1 = GetDimensions();

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
bool SegmentedTensor::IsEqual(const SegmentedTensor & rhs) const {
   if(m_cDimensions != rhs.m_cDimensions) {
      return false;
   }

   const DimensionInfo * pThisDimensionInfo = GetDimensions();
   const DimensionInfo * pRhsDimensionInfo = rhs.GetDimensions();

   size_t cValues = m_cVectorLength;
   for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
      const DimensionInfo * const pDimension1 = &pThisDimensionInfo[iDimension];
      const DimensionInfo * const pDimension2 = &pRhsDimensionInfo[iDimension];

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

} // DEFINED_ZONE_NAME
