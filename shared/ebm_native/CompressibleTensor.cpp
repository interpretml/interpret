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

#include "CompressibleTensor.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

CompressibleTensor * CompressibleTensor::Allocate(const size_t cDimensionsMax, const size_t cVectorLength) {
   EBM_ASSERT(cDimensionsMax <= k_cDimensionsMax);
   EBM_ASSERT(1 <= cVectorLength); // having 0 classes makes no sense, and having 1 class is useless

   if(IsMultiplyError(k_initialValueCapacity, cVectorLength)) {
      LOG_0(TraceLevelWarning, "WARNING Allocate IsMultiplyError(k_initialValueCapacity, cVectorLength)");
      return nullptr;
   }
   const size_t cValueCapacity = k_initialValueCapacity * cVectorLength;

   // this can't overflow since cDimensionsMax can't be bigger than k_cDimensionsMax, which is arround 64
   const size_t cBytesCompressibleTensor = sizeof(CompressibleTensor) - sizeof(DimensionInfo) + sizeof(DimensionInfo) * cDimensionsMax;
   CompressibleTensor * const pCompressibleTensor = EbmMalloc<CompressibleTensor>(1, cBytesCompressibleTensor);
   if(UNLIKELY(nullptr == pCompressibleTensor)) {
      LOG_0(TraceLevelWarning, "WARNING Allocate nullptr == pCompressibleTensor");
      return nullptr;
   }

   pCompressibleTensor->m_cVectorLength = cVectorLength;
   pCompressibleTensor->m_cDimensionsMax = cDimensionsMax;
   pCompressibleTensor->m_cDimensions = cDimensionsMax;
   pCompressibleTensor->m_cValueCapacity = cValueCapacity;
   pCompressibleTensor->m_bExpanded = false;

   FloatEbmType * const aValues = EbmMalloc<FloatEbmType>(cValueCapacity);
   if(UNLIKELY(nullptr == aValues)) {
      LOG_0(TraceLevelWarning, "WARNING Allocate nullptr == aValues");
      free(pCompressibleTensor); // don't need to call the full Free(*) yet
      return nullptr;
   }
   pCompressibleTensor->m_aValues = aValues;
   // we only need to set the base case to zero, not our entire initial allocation
   // we checked for cVectorLength * k_initialValueCapacity * sizeof(FloatEbmType), and 1 <= k_initialValueCapacity, 
   // so sizeof(FloatEbmType) * cVectorLength can't overflow
   for(size_t i = 0; i < cVectorLength; ++i) {
      aValues[i] = FloatEbmType { 0 };
   }

   if(0 != cDimensionsMax) {
      DimensionInfo * pDimension = pCompressibleTensor->GetDimensions();
      size_t iDimension = 0;
      do {
         pDimension->m_cSplits = 0;
         pDimension->m_cSplitCapacity = k_initialSplitCapacity;
         pDimension->m_aSplits = nullptr;
         ++pDimension;
         ++iDimension;
      } while(iDimension < cDimensionsMax);

      pDimension = pCompressibleTensor->GetDimensions();
      iDimension = 0;
      do {
         ActiveDataType * const aSplits = EbmMalloc<ActiveDataType>(k_initialSplitCapacity);
         if(UNLIKELY(nullptr == aSplits)) {
            LOG_0(TraceLevelWarning, "WARNING Allocate nullptr == aSplits");
            Free(pCompressibleTensor); // free everything!
            return nullptr;
         }
         pDimension->m_aSplits = aSplits;
         ++pDimension;
         ++iDimension;
      } while(iDimension < cDimensionsMax);
   }
   return pCompressibleTensor;
}

void CompressibleTensor::Free(CompressibleTensor * const pCompressibleTensor) {
   if(LIKELY(nullptr != pCompressibleTensor)) {
      free(pCompressibleTensor->m_aValues);
      if(LIKELY(0 != pCompressibleTensor->m_cDimensionsMax)) {
         const DimensionInfo * pDimensionInfo = pCompressibleTensor->GetDimensions();
         const DimensionInfo * const pDimensionInfoEnd = &pDimensionInfo[pCompressibleTensor->m_cDimensionsMax];
         do {
            free(pDimensionInfo->m_aSplits);
            ++pDimensionInfo;
         } while(pDimensionInfoEnd != pDimensionInfo);
      }
      free(pCompressibleTensor);
   }
}

void CompressibleTensor::Reset() {
   DimensionInfo * pDimensionInfo = GetDimensions();
   for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
      pDimensionInfo[iDimension].m_cSplits = 0;
   }
   // we only need to set the base case to zero
   // this can't overflow since we previously allocated this memory
   for(size_t i = 0; i < m_cVectorLength; ++i) {
      m_aValues[i] = FloatEbmType { 0 };
   }
   m_bExpanded = false;
}

ErrorEbmType CompressibleTensor::SetCountSplits(const size_t iDimension, const size_t cSplits) {
   EBM_ASSERT(iDimension < m_cDimensions);
   DimensionInfo * const pDimension = &GetDimensions()[iDimension];
   // we shouldn't be able to expand our length after we're been expanded since expanded should be the maximum size already
   EBM_ASSERT(!m_bExpanded || cSplits <= pDimension->m_cSplits);
   if(UNLIKELY(pDimension->m_cSplitCapacity < cSplits)) {
      EBM_ASSERT(!m_bExpanded); // we shouldn't be able to expand our length after we're been expanded since expanded should be the maximum size already

      if(IsAddError(cSplits, cSplits >> 1)) {
         LOG_0(TraceLevelWarning, "WARNING SetCountSplits IsAddError(cSplits, cSplits >> 1)");
         return Error_OutOfMemory;
      }
      // just increase it by 50% since we don't expect to grow our splits often after an initial period, 
      // and realloc takes some of the cost of growing away
      size_t cNewSplitCapacity = cSplits + (cSplits >> 1);
      LOG_N(TraceLevelInfo, "SetCountSplits Growing to size %zu", cNewSplitCapacity);

      if(IsMultiplyError(sizeof(ActiveDataType), cNewSplitCapacity)) {
         LOG_0(TraceLevelWarning, "WARNING SetCountSplits IsMultiplyError(sizeof(ActiveDataType), cNewSplitCapacity)");
         return Error_OutOfMemory;
      }
      size_t cBytes = sizeof(ActiveDataType) * cNewSplitCapacity;
      ActiveDataType * const aNewSplits = static_cast<ActiveDataType *>(realloc(pDimension->m_aSplits, cBytes));
      if(UNLIKELY(nullptr == aNewSplits)) {
         // according to the realloc spec, if realloc fails to allocate the new memory, it returns nullptr BUT the old memory is valid.
         // we leave m_aThreadByteBuffer1 alone in this instance and will free that memory later in the destructor
         LOG_0(TraceLevelWarning, "WARNING SetCountSplits nullptr == aNewSplits");
         return Error_OutOfMemory;
      }
      pDimension->m_aSplits = aNewSplits;
      pDimension->m_cSplitCapacity = cNewSplitCapacity;
   } // never shrink our array unless the user chooses to Trim()
   pDimension->m_cSplits = cSplits;
   return Error_None;
}

ErrorEbmType CompressibleTensor::EnsureValueCapacity(const size_t cValues) {
   if(UNLIKELY(m_cValueCapacity < cValues)) {
      EBM_ASSERT(!m_bExpanded); // we shouldn't be able to expand our length after we're been expanded since expanded should be the maximum size already

      if(IsAddError(cValues, cValues >> 1)) {
         LOG_0(TraceLevelWarning, "WARNING EnsureValueCapacity IsAddError(cValues, cValues >> 1)");
         return Error_OutOfMemory;
      }
      // just increase it by 50% since we don't expect to grow our values often after an initial period, and realloc takes some of the cost of growing away
      size_t cNewValueCapacity = cValues + (cValues >> 1);
      LOG_N(TraceLevelInfo, "EnsureValueCapacity Growing to size %zu", cNewValueCapacity);

      if(IsMultiplyError(sizeof(FloatEbmType), cNewValueCapacity)) {
         LOG_0(TraceLevelWarning, "WARNING EnsureValueCapacity IsMultiplyError(sizeof(FloatEbmType), cNewValueCapacity)");
         return Error_OutOfMemory;
      }
      size_t cBytes = sizeof(FloatEbmType) * cNewValueCapacity;
      FloatEbmType * const aNewValues = static_cast<FloatEbmType *>(realloc(m_aValues, cBytes));
      if(UNLIKELY(nullptr == aNewValues)) {
         // according to the realloc spec, if realloc fails to allocate the new memory, it returns nullptr BUT the old memory is valid.
         // we leave m_aThreadByteBuffer1 alone in this instance and will free that memory later in the destructor
         LOG_0(TraceLevelWarning, "WARNING EnsureValueCapacity nullptr == aNewValues");
         return Error_OutOfMemory;
      }
      m_aValues = aNewValues;
      m_cValueCapacity = cNewValueCapacity;
   } // never shrink our array unless the user chooses to Trim()
   return Error_None;
}

ErrorEbmType CompressibleTensor::Copy(const CompressibleTensor & rhs) {
   EBM_ASSERT(m_cDimensions == rhs.m_cDimensions);

   ErrorEbmType error;

   const DimensionInfo * pThisDimensionInfo = GetDimensions();
   const DimensionInfo * pRhsDimensionInfo = rhs.GetDimensions();

   size_t cValues = m_cVectorLength;
   for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
      const DimensionInfo * const pDimension = &pRhsDimensionInfo[iDimension];
      size_t cSplits = pDimension->m_cSplits;
      EBM_ASSERT(!IsMultiplyError(cValues, cSplits + 1)); // we're copying this memory, so multiplication can't overflow
      cValues *= (cSplits + 1);
      error = SetCountSplits(iDimension, cSplits);
      if(UNLIKELY(Error_None != error)) {
         LOG_0(TraceLevelWarning, "WARNING Copy SetCountSplits(iDimension, cSplits)");
         return error;
      }
      EBM_ASSERT(!IsMultiplyError(sizeof(ActiveDataType), cSplits)); // we're copying this memory, so multiplication can't overflow
      memcpy(pThisDimensionInfo[iDimension].m_aSplits, pDimension->m_aSplits, sizeof(ActiveDataType) * cSplits);
   }
   error = EnsureValueCapacity(cValues);
   if(UNLIKELY(Error_None != error)) {
      // already logged
      return error;
   }
   EBM_ASSERT(!IsMultiplyError(sizeof(FloatEbmType), cValues)); // we're copying this memory, so multiplication can't overflow
   memcpy(m_aValues, rhs.m_aValues, sizeof(FloatEbmType) * cValues);
   m_bExpanded = rhs.m_bExpanded;
   return Error_None;
}

bool CompressibleTensor::MultiplyAndCheckForIssues(const FloatEbmType v) {

   const DimensionInfo * pThisDimensionInfo = GetDimensions();

   size_t cValues = 1;
   for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
      // we're accessing existing memory, so it can't overflow
      EBM_ASSERT(!IsMultiplyError(cValues, pThisDimensionInfo[iDimension].m_cSplits + 1));
      cValues *= pThisDimensionInfo[iDimension].m_cSplits + 1;
   }

   FloatEbmType * pCur = &m_aValues[0];
   FloatEbmType * pEnd = &m_aValues[cValues * m_cVectorLength];
   int bBad = 0;
   // we always have 1 value, even if we have zero splits
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

ErrorEbmType CompressibleTensor::Expand(const FeatureGroup * const pFeatureGroup) {
   // checking the max isn't really the best here, but doing this right seems pretty complicated
   static_assert(std::numeric_limits<size_t>::max() <= std::numeric_limits<ActiveDataType>::max() &&
      0 == std::numeric_limits<ActiveDataType>::min(), "bad AcitveDataType size");

   ErrorEbmType error;

   LOG_0(TraceLevelVerbose, "Entered Expand");

   if(m_bExpanded) {
      // we're already expanded
      LOG_0(TraceLevelVerbose, "Exited Expand");
      return Error_None;
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

      // first, get basic counts of how many splits and values we'll have in our final result
      do {
         const size_t cBins = pFeatureGroupEntry1->m_pFeature->GetCountBins();

         // we check for simple multiplication overflow from m_cBins in Booster::Initialize when we unpack 
         // featureGroupsFeatureIndexes and in CalculateInteractionScore for interactions
         EBM_ASSERT(!IsMultiplyError(cNewValues, cBins));
         cNewValues *= cBins;

         const size_t cSplits1 = pDimensionFirst1->m_cSplits;

         EBM_ASSERT(!IsMultiplyError(cValues1, cSplits1 + 1)); // this is accessing existing memory, so it can't overflow
         cValues1 *= cSplits1 + 1;

         pDimensionInfoStackFirst->m_pSplit1 = &pDimensionFirst1->m_aSplits[cSplits1];

         const size_t cSplits = cBins - size_t { 1 };
         pDimensionInfoStackFirst->m_iSplit2 = cSplits;
         pDimensionInfoStackFirst->m_cNewSplits = cSplits;

         ++pDimensionFirst1;
         ++pDimensionInfoStackFirst;
         ++pFeatureGroupEntry1;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry1);
      
      if(size_t { 0 } == cNewValues) {
         // there's a really degenerate case where we have zero training and zero validation samples, and the user 
         // specifies zero bins which is legal since there are no bins in the training or validation, in this case
         // the tensor has zero bins in one dimension, so there are zero bins in the entire tensor.
         LOG_0(TraceLevelWarning, "WARNING Expand Zero sized tensor");
      } else {
         if(IsMultiplyError(m_cVectorLength, cNewValues)) {
            LOG_0(TraceLevelWarning, "WARNING Expand IsMultiplyError(m_cVectorLength, cNewValues)");
            return Error_OutOfMemory;
         }
         const size_t cVectoredNewValues = m_cVectorLength * cNewValues;
         // call EnsureValueCapacity before using the m_aValues pointer since m_aValues might change inside EnsureValueCapacity
         error = EnsureValueCapacity(cVectoredNewValues);
         if(UNLIKELY(Error_None != error)) {
            // already logged
            return error;
         }

         FloatEbmType * const aValues = m_aValues;
         const DimensionInfo * const aDimension1 = GetDimensions();

         EBM_ASSERT(cValues1 <= cNewValues);
         EBM_ASSERT(!IsMultiplyError(m_cVectorLength, cValues1)); // we checked against cNewValues above, and cValues1 should be smaller
         const FloatEbmType * pValue1 = &aValues[m_cVectorLength * cValues1];
         FloatEbmType * pValueTop = &aValues[cVectoredNewValues];

         // traverse the values in reverse so that we can put our results at the higher order indexes where we are guaranteed not to overwrite our 
         // existing values which we still need to copy first do the values because we need to refer to the old splits when making decisions about 
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

            // For a single dimensional CompressibleTensor checking here is best.  
            // For two or higher dimensions, we could instead check inside our loop below for when we reach the end of the pDimensionInfoStack, thus 
            // eliminating the check on most loops. We'll spend most of our time working on single features though, so we optimize for that case, but 
            // if we special cased the single dimensional case, then we would want to move this check into the loop below in the case of 
            // multi-dimensioncal CompressibleTensors
            if(UNLIKELY(aValues == pValueTop)) {
               // we've written our final tensor cell, so we're done
               break;
            }

            DimensionInfoStackExpand * pDimensionInfoStackSecond = aDimensionInfoStackExpand;
            const DimensionInfo * pDimensionSecond1 = aDimension1;

            size_t multiplication1 = m_cVectorLength;

            while(true) {
               const ActiveDataType * const pSplit1 = pDimensionInfoStackSecond->m_pSplit1;
               size_t iSplit2 = pDimensionInfoStackSecond->m_iSplit2;

               ActiveDataType * const aSplits1 = pDimensionSecond1->m_aSplits;

               EBM_ASSERT(static_cast<size_t>(pSplit1 - aSplits1) <= iSplit2);
               if(UNPREDICTABLE(aSplits1 < pSplit1)) {
                  EBM_ASSERT(0 < iSplit2);

                  const ActiveDataType * const pSplit1MinusOne = pSplit1 - 1;

                  const size_t d1 = static_cast<size_t>(*pSplit1MinusOne);

                  --iSplit2;

                  const bool bMove = UNPREDICTABLE(iSplit2 <= d1);
                  pDimensionInfoStackSecond->m_pSplit1 = bMove ? pSplit1MinusOne : pSplit1;
                  pValue1 = bMove ? pValue1 - multiplication1 : pValue1;

                  pDimensionInfoStackSecond->m_iSplit2 = iSplit2;
                  break;
               } else {
                  if(UNPREDICTABLE(0 < iSplit2)) {
                     pDimensionInfoStackSecond->m_iSplit2 = iSplit2 - 1;
                     break;
                  } else {
                     pValue1 -= multiplication1; // put us before the beginning.  We'll add the full row first

                     const size_t cSplits1 = pDimensionSecond1->m_cSplits;

                     // we're already allocated values, so this is accessing what we've already allocated, so it must not overflow
                     EBM_ASSERT(!IsMultiplyError(multiplication1, 1 + cSplits1));
                     multiplication1 *= 1 + cSplits1;

                     // go to the last valid entry back to where we started.  If we don't move down a set, then we re-do this set of numbers
                     pValue1 += multiplication1;

                     pDimensionInfoStackSecond->m_pSplit1 = &aSplits1[cSplits1];
                     pDimensionInfoStackSecond->m_iSplit2 = pDimensionInfoStackSecond->m_cNewSplits;

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
            const size_t cSplits = cBins - size_t { 1 };
            const DimensionInfo * const pDimension = &aDimension1[iDimension];
            if(cSplits != pDimension->m_cSplits) {
               error = SetCountSplits(iDimension, cSplits);
               if(UNLIKELY(Error_None != error)) {
                  // already logged
                  return error;
               }

               // if cSplits is zero then pDimension->m_cSplits must be zero and we'd be filtered out above
               EBM_ASSERT(size_t { 1 } <= cSplits);

               ActiveDataType * const aSplit = pDimension->m_aSplits;
               size_t iSplit = 0;
               do {
                  aSplit[iSplit] = iSplit;
                  ++iSplit;
               } while(cSplits != iSplit);
            }
            ++iDimension;
            ++pFeatureGroupEntry2;
         } while(pFeatureGroupEntryEnd != pFeatureGroupEntry2);
      }
   }
   m_bExpanded = true;
   
   LOG_0(TraceLevelVerbose, "Exited Expand");
   return Error_None;
}

void CompressibleTensor::AddExpandedWithBadValueProtection(const FloatEbmType * const aFromValues) {
   EBM_ASSERT(m_bExpanded);
   size_t cItems = m_cVectorLength;

   const DimensionInfo * const aDimension = GetDimensions();

   for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
      // this can't overflow since we've already allocated them!
      cItems *= aDimension[iDimension].m_cSplits + 1;
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

ErrorEbmType CompressibleTensor::Add(const CompressibleTensor & rhs) {
   ErrorEbmType error;

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

      return Error_None;
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
   // first, get basic counts of how many splits and values we'll have in our final result
   do {
      const size_t cSplits1 = pDimensionFirst1->m_cSplits;
      ActiveDataType * p1Cur = pDimensionFirst1->m_aSplits;
      const size_t cSplits2 = pDimensionFirst2->m_cSplits;
      ActiveDataType * p2Cur = pDimensionFirst2->m_aSplits;

      cValues1 *= cSplits1 + 1; // this can't overflow since we're counting existing allocated memory
      cValues2 *= cSplits2 + 1; // this can't overflow since we're counting existing allocated memory

      ActiveDataType * const p1End = &p1Cur[cSplits1];
      ActiveDataType * const p2End = &p2Cur[cSplits2];

      pDimensionInfoStackFirst->m_pSplit1 = p1End;
      pDimensionInfoStackFirst->m_pSplit2 = p2End;

      size_t cNewSingleDimensionSplits = 0;

      // processing forwards here is slightly faster in terms of cache fetch efficiency.  We'll then be guaranteed to have the splits at least
      // in the cache, which will be benefitial when traversing backwards later below
      while(true) {
         if(UNLIKELY(p2End == p2Cur)) {
            // check the other array first.  Most of the time the other array will be shorter since we'll be adding
            // a sequence of sliced lines and our main line will be in *this, and there will be more slices in general for
            // a line that is added to a lot
            cNewSingleDimensionSplits += static_cast<size_t>(p1End - p1Cur);
            break;
         }
         if(UNLIKELY(p1End == p1Cur)) {
            cNewSingleDimensionSplits += static_cast<size_t>(p2End - p2Cur);
            break;
         }
         ++cNewSingleDimensionSplits; // if we move one or both pointers, we just added annother unique one

         const ActiveDataType d1 = *p1Cur;
         const ActiveDataType d2 = *p2Cur;

         p1Cur = UNPREDICTABLE(d1 <= d2) ? p1Cur + 1 : p1Cur;
         p2Cur = UNPREDICTABLE(d2 <= d1) ? p2Cur + 1 : p2Cur;
      }
      pDimensionInfoStackFirst->m_cNewSplits = cNewSingleDimensionSplits;
      // we check for simple multiplication overflow from m_cBins in Booster::Initialize when we unpack featureGroupsFeatureIndexes and in 
      // CalculateInteractionScore for interactions
      EBM_ASSERT(!IsMultiplyError(cNewValues, cNewSingleDimensionSplits + 1));
      cNewValues *= cNewSingleDimensionSplits + 1;

      ++pDimensionFirst1;
      ++pDimensionFirst2;

      ++pDimensionInfoStackFirst;
   } while(pDimensionInfoStackEnd != pDimensionInfoStackFirst);

   if(IsMultiplyError(m_cVectorLength, cNewValues)) {
      LOG_0(TraceLevelWarning, "WARNING Add IsMultiplyError(m_cVectorLength, cNewValues)");
      return Error_OutOfMemory;
   }
   // call EnsureValueCapacity before using the m_aValues pointer since m_aValues might change inside EnsureValueCapacity
   error = EnsureValueCapacity(m_cVectorLength * cNewValues);
   if(UNLIKELY(Error_None != error)) {
      // already logged
      return error;
   }

   const FloatEbmType * pValue2 = &rhs.m_aValues[m_cVectorLength * cValues2];  // we're accessing allocated memory, so it can't overflow
   const DimensionInfo * const aDimension2 = rhs.GetDimensions();

   FloatEbmType * const aValues = m_aValues;
   const DimensionInfo * const aDimension1 = GetDimensions();

   const FloatEbmType * pValue1 = &aValues[m_cVectorLength * cValues1]; // we're accessing allocated memory, so it can't overflow
   FloatEbmType * pValueTop = &aValues[m_cVectorLength * cNewValues]; // we're accessing allocated memory, so it can't overflow

   // traverse the values in reverse so that we can put our results at the higher order indexes where we are guaranteed not to overwrite our
   // existing values which we still need to copy first do the values because we need to refer to the old splits when making decisions about where 
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

      // For a single dimensional CompressibleTensor checking here is best.  
      // For two or higher dimensions, we could instead check inside our loop below for when we reach the end of the pDimensionInfoStack,
      // thus eliminating the check on most loops.  We'll spend most of our time working on single features though, so we optimize for that case, 
      // but if we special cased the single dimensional case, then we would want to move this check into the loop below in the case 
      // of multi-dimensioncal CompressibleTensors
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
         const ActiveDataType * const pSplit1 = pDimensionInfoStackSecond->m_pSplit1;
         const ActiveDataType * const pSplit2 = pDimensionInfoStackSecond->m_pSplit2;

         ActiveDataType * const aSplits1 = pDimensionSecond1->m_aSplits;
         ActiveDataType * const aSplits2 = pDimensionSecond2->m_aSplits;

         if(UNPREDICTABLE(aSplits1 < pSplit1)) {
            if(UNPREDICTABLE(aSplits2 < pSplit2)) {
               const ActiveDataType * const pSplit1MinusOne = pSplit1 - 1;
               const ActiveDataType * const pSplit2MinusOne = pSplit2 - 1;

               const ActiveDataType d1 = *pSplit1MinusOne;
               const ActiveDataType d2 = *pSplit2MinusOne;

               const bool bMove1 = UNPREDICTABLE(d2 <= d1);
               pDimensionInfoStackSecond->m_pSplit1 = bMove1 ? pSplit1MinusOne : pSplit1;
               pValue1 = bMove1 ? pValue1 - multiplication1 : pValue1;

               const bool bMove2 = UNPREDICTABLE(d1 <= d2);
               pDimensionInfoStackSecond->m_pSplit2 = bMove2 ? pSplit2MinusOne : pSplit2;
               pValue2 = bMove2 ? pValue2 - multiplication2 : pValue2;
               break;
            } else {
               pValue1 -= multiplication1;
               pDimensionInfoStackSecond->m_pSplit1 = pSplit1 - 1;
               break;
            }
         } else {
            if(UNPREDICTABLE(aSplits2 < pSplit2)) {
               pValue2 -= multiplication2;
               pDimensionInfoStackSecond->m_pSplit2 = pSplit2 - 1;
               break;
            } else {
               pValue1 -= multiplication1; // put us before the beginning.  We'll add the full row first
               pValue2 -= multiplication2; // put us before the beginning.  We'll add the full row first

               const size_t cSplits1 = pDimensionSecond1->m_cSplits;
               const size_t cSplits2 = pDimensionSecond2->m_cSplits;

               EBM_ASSERT(!IsMultiplyError(multiplication1, 1 + cSplits1)); // we're accessing allocated memory, so it can't overflow
               multiplication1 *= 1 + cSplits1;
               EBM_ASSERT(!IsMultiplyError(multiplication2, 1 + cSplits2)); // we're accessing allocated memory, so it can't overflow
               multiplication2 *= 1 + cSplits2;

               // go to the last valid entry back to where we started.  If we don't move down a set, then we re-do this set of numbers
               pValue1 += multiplication1;
               // go to the last valid entry back to where we started.  If we don't move down a set, then we re-do this set of numbers
               pValue2 += multiplication2;

               pDimensionInfoStackSecond->m_pSplit1 = &aSplits1[cSplits1];
               pDimensionInfoStackSecond->m_pSplit2 = &aSplits2[cSplits2];
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

   // now finally do the splits

   const DimensionInfoStack * pDimensionInfoStackCur = dimensionStack;
   const DimensionInfo * pDimension1Cur = aDimension1;
   const DimensionInfo * pDimension2Cur = aDimension2;
   size_t iDimension = 0;
   do {
      const size_t cNewSplits = pDimensionInfoStackCur->m_cNewSplits;
      const size_t cOriginalSplitsBeforeSetting = pDimension1Cur->m_cSplits;

      // this will increase our capacity, if required.  It will also change m_cSplits, so we get that before calling it.  
      // SetCountSplits might change m_aValuesAndSplits, so we need to actually keep it here after getting m_cSplits but 
      // before set set all our pointers
      error = SetCountSplits(iDimension, cNewSplits);
      if(UNLIKELY(Error_None != error)) {
         // already logged
         return error;
      }

      const ActiveDataType * p1Cur = &pDimension1Cur->m_aSplits[cOriginalSplitsBeforeSetting];
      const ActiveDataType * p2Cur = &pDimension2Cur->m_aSplits[pDimension2Cur->m_cSplits];
      ActiveDataType * pTopCur = &pDimension1Cur->m_aSplits[cNewSplits];

      // traverse in reverse so that we can put our results at the higher order indexes where we are guaranteed not to overwrite our existing values
      // which we still need to copy
      while(true) {
         EBM_ASSERT(pDimension1Cur->m_aSplits <= pTopCur);
         EBM_ASSERT(pDimension1Cur->m_aSplits <= p1Cur);
         EBM_ASSERT(pDimension2Cur->m_aSplits <= p2Cur);
         EBM_ASSERT(p1Cur <= pTopCur);
         EBM_ASSERT(static_cast<size_t>(p2Cur - pDimension2Cur->m_aSplits) <= static_cast<size_t>(pTopCur - pDimension1Cur->m_aSplits));

         if(UNLIKELY(pTopCur == p1Cur)) {
            // since we've finished the rhs splits, our CompressibleTensor already has the right splits in place, so all we need is to add the value
            // of the last region in rhs to our remaining values
            break;
         }
         // pTopCur is an index above pDimension1Cur->m_aSplits.  p2Cur is an index above pDimension2Cur->m_aSplits.  We want to decide if they
         // are at the same index above their respective arrays
         if(UNLIKELY(static_cast<size_t>(pTopCur - pDimension1Cur->m_aSplits) == static_cast<size_t>(p2Cur - pDimension2Cur->m_aSplits))) {
            EBM_ASSERT(pDimension1Cur->m_aSplits < pTopCur);
            // direct copy the remaining splits.  There should be at least one
            memcpy(
               pDimension1Cur->m_aSplits,
               pDimension2Cur->m_aSplits,
               static_cast<size_t>(pTopCur - pDimension1Cur->m_aSplits) * sizeof(ActiveDataType)
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
   return Error_None;
}

#ifndef NDEBUG
bool CompressibleTensor::IsEqual(const CompressibleTensor & rhs) const {
   if(m_cDimensions != rhs.m_cDimensions) {
      return false;
   }

   const DimensionInfo * pThisDimensionInfo = GetDimensions();
   const DimensionInfo * pRhsDimensionInfo = rhs.GetDimensions();

   size_t cValues = m_cVectorLength;
   for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
      const DimensionInfo * const pDimension1 = &pThisDimensionInfo[iDimension];
      const DimensionInfo * const pDimension2 = &pRhsDimensionInfo[iDimension];

      size_t cSplits = pDimension1->m_cSplits;
      if(cSplits != pDimension2->m_cSplits) {
         return false;
      }

      if(0 != cSplits) {
         EBM_ASSERT(!IsMultiplyError(cValues, cSplits + 1)); // we're accessing allocated memory, so it can't overflow
         cValues *= cSplits + 1;

         const ActiveDataType * pD1Cur = pDimension1->m_aSplits;
         const ActiveDataType * pD2Cur = pDimension2->m_aSplits;
         const ActiveDataType * const pD1End = pD1Cur + cSplits;
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
