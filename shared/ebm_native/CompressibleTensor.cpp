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

   if(IsMultiplyError(k_initialScoreCapacity, cVectorLength)) {
      LOG_0(TraceLevelWarning, "WARNING Allocate IsMultiplyError(k_initialScoreCapacity, cVectorLength)");
      return nullptr;
   }
   const size_t cScoreCapacity = k_initialScoreCapacity * cVectorLength;

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
   pCompressibleTensor->m_cScoreCapacity = cScoreCapacity;
   pCompressibleTensor->m_bExpanded = false;

   FloatFast * const aScores = EbmMalloc<FloatFast>(cScoreCapacity);
   if(UNLIKELY(nullptr == aScores)) {
      LOG_0(TraceLevelWarning, "WARNING Allocate nullptr == aScores");
      free(pCompressibleTensor); // don't need to call the full Free(*) yet
      return nullptr;
   }
   pCompressibleTensor->m_aScores = aScores;
   // we only need to set the base case to zero, not our entire initial allocation
   // we checked for cVectorLength * k_initialScoreCapacity * sizeof(FloatFast), and 1 <= k_initialScoreCapacity, 
   // so sizeof(FloatFast) * cVectorLength can't overflow
   for(size_t i = 0; i < cVectorLength; ++i) {
      aScores[i] = 0;
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
      free(pCompressibleTensor->m_aScores);
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
      m_aScores[i] = 0;
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

ErrorEbmType CompressibleTensor::EnsureScoreCapacity(const size_t cScores) {
   if(UNLIKELY(m_cScoreCapacity < cScores)) {
      EBM_ASSERT(!m_bExpanded); // we shouldn't be able to expand our length after we're been expanded since expanded should be the maximum size already

      if(IsAddError(cScores, cScores >> 1)) {
         LOG_0(TraceLevelWarning, "WARNING EnsureScoreCapacity IsAddError(cScores, cScores >> 1)");
         return Error_OutOfMemory;
      }
      // just increase it by 50% since we don't expect to grow our scores often after an initial period, and realloc takes some of the cost of growing away
      size_t cNewScoreCapacity = cScores + (cScores >> 1);
      LOG_N(TraceLevelInfo, "EnsureScoreCapacity Growing to size %zu", cNewScoreCapacity);

      if(IsMultiplyError(sizeof(FloatFast), cNewScoreCapacity)) {
         LOG_0(TraceLevelWarning, "WARNING EnsureScoreCapacity IsMultiplyError(sizeof(FloatFast), cNewScoreCapacity)");
         return Error_OutOfMemory;
      }
      size_t cBytes = sizeof(FloatFast) * cNewScoreCapacity;
      FloatFast * const aNewScores = static_cast<FloatFast *>(realloc(m_aScores, cBytes));
      if(UNLIKELY(nullptr == aNewScores)) {
         // according to the realloc spec, if realloc fails to allocate the new memory, it returns nullptr BUT the old memory is valid.
         // we leave m_aThreadByteBuffer1 alone in this instance and will free that memory later in the destructor
         LOG_0(TraceLevelWarning, "WARNING EnsureScoreCapacity nullptr == aNewScores");
         return Error_OutOfMemory;
      }
      m_aScores = aNewScores;
      m_cScoreCapacity = cNewScoreCapacity;
   } // never shrink our array unless the user chooses to Trim()
   return Error_None;
}

ErrorEbmType CompressibleTensor::Copy(const CompressibleTensor & rhs) {
   EBM_ASSERT(m_cDimensions == rhs.m_cDimensions);

   ErrorEbmType error;

   const DimensionInfo * pThisDimensionInfo = GetDimensions();
   const DimensionInfo * pRhsDimensionInfo = rhs.GetDimensions();

   size_t cScores = m_cVectorLength;
   for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
      const DimensionInfo * const pDimension = &pRhsDimensionInfo[iDimension];
      size_t cSplits = pDimension->m_cSplits;
      EBM_ASSERT(!IsMultiplyError(cScores, cSplits + 1)); // we're copying this memory, so multiplication can't overflow
      cScores *= (cSplits + 1);
      error = SetCountSplits(iDimension, cSplits);
      if(UNLIKELY(Error_None != error)) {
         LOG_0(TraceLevelWarning, "WARNING Copy SetCountSplits(iDimension, cSplits)");
         return error;
      }
      EBM_ASSERT(!IsMultiplyError(sizeof(ActiveDataType), cSplits)); // we're copying this memory, so multiplication can't overflow
      memcpy(pThisDimensionInfo[iDimension].m_aSplits, pDimension->m_aSplits, sizeof(ActiveDataType) * cSplits);
   }
   error = EnsureScoreCapacity(cScores);
   if(UNLIKELY(Error_None != error)) {
      // already logged
      return error;
   }
   EBM_ASSERT(!IsMultiplyError(sizeof(FloatFast), cScores)); // we're copying this memory, so multiplication can't overflow
   memcpy(m_aScores, rhs.m_aScores, sizeof(FloatFast) * cScores);
   m_bExpanded = rhs.m_bExpanded;
   return Error_None;
}

bool CompressibleTensor::MultiplyAndCheckForIssues(const double v) {
   const FloatFast vFloat = SafeConvertFloat<FloatFast>(v);
   const DimensionInfo * pThisDimensionInfo = GetDimensions();

   size_t cScores = m_cVectorLength;
   for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
      // we're accessing existing memory, so it can't overflow
      EBM_ASSERT(!IsMultiplyError(cScores, pThisDimensionInfo[iDimension].m_cSplits + 1));
      cScores *= pThisDimensionInfo[iDimension].m_cSplits + 1;
   }

   FloatFast * pCur = &m_aScores[0];
   FloatFast * pEnd = &m_aScores[cScores];
   int bBad = 0;
   // we always have 1 score, even if we have zero splits
   do {
      const FloatFast val = *pCur * vFloat;
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

ErrorEbmType CompressibleTensor::Expand(const Term * const pTerm) {
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

   EBM_ASSERT(nullptr != pTerm);
   const size_t cDimensions = pTerm->GetCountDimensions();
   if(size_t { 0 } != cDimensions) {
      const TermEntry * pTermEntry1 = pTerm->GetTermEntries();
      const TermEntry * const pTermEntriesEnd = pTermEntry1 + cDimensions;
      DimensionInfoStackExpand aDimensionInfoStackExpand[k_cDimensionsMax];
      DimensionInfoStackExpand * pDimensionInfoStackFirst = aDimensionInfoStackExpand;
      const DimensionInfo * pDimensionFirst1 = GetDimensions();
      size_t cScores1 = m_cVectorLength;
      size_t cNewScores = m_cVectorLength;

      // first, get basic counts of how many splits and scores we'll have in our final result
      do {
         const size_t cBins = pTermEntry1->m_pFeature->GetCountBins();

         // we check for simple multiplication overflow from m_cBins in Booster::Initialize when we unpack 
         // featureIndexes and in CalcInteractionStrength for interactions
         EBM_ASSERT(!IsMultiplyError(cNewScores, cBins));
         cNewScores *= cBins;

         const size_t cSplits1 = pDimensionFirst1->m_cSplits;

         EBM_ASSERT(!IsMultiplyError(cScores1, cSplits1 + 1)); // this is accessing existing memory, so it can't overflow
         cScores1 *= cSplits1 + 1;

         pDimensionInfoStackFirst->m_pSplit1 = &pDimensionFirst1->m_aSplits[cSplits1];

         const size_t cSplits = cBins - size_t { 1 };
         pDimensionInfoStackFirst->m_iSplit2 = cSplits;
         pDimensionInfoStackFirst->m_cNewSplits = cSplits;

         ++pDimensionFirst1;
         ++pDimensionInfoStackFirst;
         ++pTermEntry1;
      } while(pTermEntriesEnd != pTermEntry1);
      
      if(size_t { 0 } == cNewScores) {
         // there's a really degenerate case where we have zero training and zero validation samples, and the user 
         // specifies zero bins which is legal since there are no bins in the training or validation, in this case
         // the tensor has zero bins in one dimension, so there are zero bins in the entire tensor.
         LOG_0(TraceLevelWarning, "WARNING Expand Zero sized tensor");
      } else {
         // call EnsureScoreCapacity before using the m_aScores pointer since m_aScores might change inside EnsureScoreCapacity
         error = EnsureScoreCapacity(cNewScores);
         if(UNLIKELY(Error_None != error)) {
            // already logged
            return error;
         }

         FloatFast * const aScores = m_aScores;
         const DimensionInfo * const aDimension1 = GetDimensions();

         EBM_ASSERT(cScores1 <= cNewScores);
         const FloatFast * pScore1 = &aScores[cScores1];
         FloatFast * pScoreTop = &aScores[cNewScores];

         // traverse the scores in reverse so that we can put our results at the higher order indexes where we are guaranteed not to overwrite our 
         // existing scores which we still need to copy first do the scores because we need to refer to the old splits when making decisions about 
         // where to move next
         while(true) {
            const FloatFast * pScore1Move = pScore1;
            const FloatFast * const pScoreTopEnd = pScoreTop - m_cVectorLength;
            do {
               --pScore1Move;
               --pScoreTop;
               EBM_ASSERT(aScores <= pScore1Move);
               EBM_ASSERT(aScores <= pScoreTop);
               *pScoreTop = *pScore1Move;
            } while(pScoreTopEnd != pScoreTop);

            // For a single dimensional CompressibleTensor checking here is best.  
            // For two or higher dimensions, we could instead check inside our loop below for when we reach the end of the pDimensionInfoStack, thus 
            // eliminating the check on most loops. We'll spend most of our time working on single features though, so we optimize for that case, but 
            // if we special cased the single dimensional case, then we would want to move this check into the loop below in the case of 
            // multi-dimensioncal CompressibleTensors
            if(UNLIKELY(aScores == pScoreTop)) {
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
                  pScore1 = bMove ? pScore1 - multiplication1 : pScore1;

                  pDimensionInfoStackSecond->m_iSplit2 = iSplit2;
                  break;
               } else {
                  if(UNPREDICTABLE(0 < iSplit2)) {
                     pDimensionInfoStackSecond->m_iSplit2 = iSplit2 - 1;
                     break;
                  } else {
                     pScore1 -= multiplication1; // put us before the beginning.  We'll add the full row first

                     const size_t cSplits1 = pDimensionSecond1->m_cSplits;

                     // we're already allocated scores, so this is accessing what we've already allocated, so it must not overflow
                     EBM_ASSERT(!IsMultiplyError(multiplication1, 1 + cSplits1));
                     multiplication1 *= 1 + cSplits1;

                     // go to the last valid entry back to where we started.  If we don't move down a set, then we re-do this set of numbers
                     pScore1 += multiplication1;

                     pDimensionInfoStackSecond->m_pSplit1 = &aSplits1[cSplits1];
                     pDimensionInfoStackSecond->m_iSplit2 = pDimensionInfoStackSecond->m_cNewSplits;

                     ++pDimensionSecond1;
                     ++pDimensionInfoStackSecond;
                     continue;
                  }
               }
            }
         }

         EBM_ASSERT(pScoreTop == m_aScores);
         EBM_ASSERT(pScore1 == m_aScores + m_cVectorLength);

         const TermEntry * pTermEntry2 = pTerm->GetTermEntries();
         size_t iDimension = 0;
         do {
            const size_t cBins = pTermEntry2->m_pFeature->GetCountBins();
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
            ++pTermEntry2;
         } while(pTermEntriesEnd != pTermEntry2);
      }
   }
   m_bExpanded = true;
   
   LOG_0(TraceLevelVerbose, "Exited Expand");
   return Error_None;
}

void CompressibleTensor::AddExpandedWithBadValueProtection(const FloatFast * const aFromScores) {
   EBM_ASSERT(m_bExpanded);
   size_t cItems = m_cVectorLength;

   const DimensionInfo * const aDimension = GetDimensions();

   for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
      // this can't overflow since we've already allocated them!
      cItems *= aDimension[iDimension].m_cSplits + 1;
   }

   const FloatFast * pFromScore = aFromScores;
   FloatFast * pToScore = m_aScores;
   const FloatFast * const pToScoresEnd = m_aScores + cItems;
   do {
      // if we get a NaN value, then just consider it a no-op zero
      // if we get a +infinity, then just make our value the maximum
      // if we get a -infinity, then just make our value the minimum
      // these changes will make us out of sync with the updates to our logits, but it should be at the extremes anyways
      // so, not much real loss there.  Also, if we have NaN, or +-infinity in an update, we'll be stopping boosting soon
      // but we want to preserve the best term scores that we had

      FloatFast score = *pFromScore;
      score = std::isnan(score) ? FloatFast { 0 } : score;
      score = *pToScore + score;
      // this is a check for -infinity, without the -infinity value since some compilers make that illegal
      // even so far as to make isinf always FALSE with some compiler flags
      // include the equals case so that the compiler is less likely to optimize that out
      score = score <= std::numeric_limits<FloatFast>::lowest() ? std::numeric_limits<FloatFast>::lowest() : score;
      // this is a check for +infinity, without the +infinity value since some compilers make that illegal
      // even so far as to make isinf always FALSE with some compiler flags
      // include the equals case so that the compiler is less likely to optimize that out
      score = std::numeric_limits<FloatFast>::max() <= score ? std::numeric_limits<FloatFast>::max() : score;
      *pToScore = score;
      ++pFromScore;
      ++pToScore;
   } while(pToScoresEnd != pToScore);
}

ErrorEbmType CompressibleTensor::Add(const CompressibleTensor & rhs) {
   ErrorEbmType error;

   DimensionInfoStack dimensionStack[k_cDimensionsMax];

   EBM_ASSERT(m_cDimensions == rhs.m_cDimensions);

   if(0 == m_cDimensions) {
      EBM_ASSERT(1 <= m_cScoreCapacity);
      EBM_ASSERT(nullptr != m_aScores);

      FloatFast * pTo = &m_aScores[0];
      const FloatFast * pFrom = &rhs.m_aScores[0];
      const FloatFast * const pToEnd = &pTo[m_cVectorLength];
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

   size_t cScores1 = m_cVectorLength;
   size_t cScores2 = m_cVectorLength;
   size_t cNewScores = m_cVectorLength;

   EBM_ASSERT(0 < m_cDimensions);
   // first, get basic counts of how many splits and values we'll have in our final result
   do {
      const size_t cSplits1 = pDimensionFirst1->m_cSplits;
      ActiveDataType * p1Cur = pDimensionFirst1->m_aSplits;
      const size_t cSplits2 = pDimensionFirst2->m_cSplits;
      ActiveDataType * p2Cur = pDimensionFirst2->m_aSplits;

      cScores1 *= cSplits1 + 1; // this can't overflow since we're counting existing allocated memory
      cScores2 *= cSplits2 + 1; // this can't overflow since we're counting existing allocated memory

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
      // we check for simple multiplication overflow from m_cBins in Booster::Initialize when we unpack featureIndexes and in 
      // CalcInteractionStrength for interactions
      EBM_ASSERT(!IsMultiplyError(cNewScores, cNewSingleDimensionSplits + 1));
      cNewScores *= cNewSingleDimensionSplits + 1;

      ++pDimensionFirst1;
      ++pDimensionFirst2;

      ++pDimensionInfoStackFirst;
   } while(pDimensionInfoStackEnd != pDimensionInfoStackFirst);

   // call EnsureScoreCapacity before using the m_aScores pointer since m_aScores might change inside EnsureScoreCapacity
   error = EnsureScoreCapacity(cNewScores);
   if(UNLIKELY(Error_None != error)) {
      // already logged
      return error;
   }

   const FloatFast * pScore2 = &rhs.m_aScores[cScores2];  // we're accessing allocated memory, so it can't overflow
   const DimensionInfo * const aDimension2 = rhs.GetDimensions();

   FloatFast * const aScores = m_aScores;
   const DimensionInfo * const aDimension1 = GetDimensions();

   const FloatFast * pScore1 = &aScores[cScores1]; // we're accessing allocated memory, so it can't overflow
   FloatFast * pScoreTop = &aScores[cNewScores]; // we're accessing allocated memory, so it can't overflow

   // traverse the scores in reverse so that we can put our results at the higher order indexes where we are guaranteed not to overwrite our
   // existing scores which we still need to copy first do the scores because we need to refer to the old splits when making decisions about where 
   // to move next
   while(true) {
      const FloatFast * pScore1Move = pScore1;
      const FloatFast * pScore2Move = pScore2;
      const FloatFast * const pScoreTopEnd = pScoreTop - m_cVectorLength;
      do {
         --pScore1Move;
         --pScore2Move;
         --pScoreTop;
         *pScoreTop = *pScore1Move + *pScore2Move;
      } while(pScoreTopEnd != pScoreTop);

      // For a single dimensional CompressibleTensor checking here is best.  
      // For two or higher dimensions, we could instead check inside our loop below for when we reach the end of the pDimensionInfoStack,
      // thus eliminating the check on most loops.  We'll spend most of our time working on single features though, so we optimize for that case, 
      // but if we special cased the single dimensional case, then we would want to move this check into the loop below in the case 
      // of multi-dimensioncal CompressibleTensors
      if(UNLIKELY(aScores == pScoreTop)) {
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
               pScore1 = bMove1 ? pScore1 - multiplication1 : pScore1;

               const bool bMove2 = UNPREDICTABLE(d1 <= d2);
               pDimensionInfoStackSecond->m_pSplit2 = bMove2 ? pSplit2MinusOne : pSplit2;
               pScore2 = bMove2 ? pScore2 - multiplication2 : pScore2;
               break;
            } else {
               pScore1 -= multiplication1;
               pDimensionInfoStackSecond->m_pSplit1 = pSplit1 - 1;
               break;
            }
         } else {
            if(UNPREDICTABLE(aSplits2 < pSplit2)) {
               pScore2 -= multiplication2;
               pDimensionInfoStackSecond->m_pSplit2 = pSplit2 - 1;
               break;
            } else {
               pScore1 -= multiplication1; // put us before the beginning.  We'll add the full row first
               pScore2 -= multiplication2; // put us before the beginning.  We'll add the full row first

               const size_t cSplits1 = pDimensionSecond1->m_cSplits;
               const size_t cSplits2 = pDimensionSecond2->m_cSplits;

               EBM_ASSERT(!IsMultiplyError(multiplication1, 1 + cSplits1)); // we're accessing allocated memory, so it can't overflow
               multiplication1 *= 1 + cSplits1;
               EBM_ASSERT(!IsMultiplyError(multiplication2, 1 + cSplits2)); // we're accessing allocated memory, so it can't overflow
               multiplication2 *= 1 + cSplits2;

               // go to the last valid entry back to where we started.  If we don't move down a set, then we re-do this set of numbers
               pScore1 += multiplication1;
               // go to the last valid entry back to where we started.  If we don't move down a set, then we re-do this set of numbers
               pScore2 += multiplication2;

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

   EBM_ASSERT(pScoreTop == m_aScores);
   EBM_ASSERT(pScore1 == m_aScores + m_cVectorLength);
   EBM_ASSERT(pScore2 == rhs.m_aScores + m_cVectorLength);

   // now finally do the splits

   const DimensionInfoStack * pDimensionInfoStackCur = dimensionStack;
   const DimensionInfo * pDimension1Cur = aDimension1;
   const DimensionInfo * pDimension2Cur = aDimension2;
   size_t iDimension = 0;
   do {
      const size_t cNewSplits = pDimensionInfoStackCur->m_cNewSplits;
      const size_t cOriginalSplitsBeforeSetting = pDimension1Cur->m_cSplits;

      // this will increase our capacity, if required.  It will also change m_cSplits, so we get that before calling it.  
      // SetCountSplits might change m_aScoresAndSplits, so we need to actually keep it here after getting m_cSplits but 
      // before set set all our pointers
      error = SetCountSplits(iDimension, cNewSplits);
      if(UNLIKELY(Error_None != error)) {
         // already logged
         return error;
      }

      const ActiveDataType * p1Cur = &pDimension1Cur->m_aSplits[cOriginalSplitsBeforeSetting];
      const ActiveDataType * p2Cur = &pDimension2Cur->m_aSplits[pDimension2Cur->m_cSplits];
      ActiveDataType * pTopCur = &pDimension1Cur->m_aSplits[cNewSplits];

      // traverse in reverse so that we can put our results at the higher order indexes where we are guaranteed not to overwrite our existing scores
      // which we still need to copy
      while(true) {
         EBM_ASSERT(pDimension1Cur->m_aSplits <= pTopCur);
         EBM_ASSERT(pDimension1Cur->m_aSplits <= p1Cur);
         EBM_ASSERT(pDimension2Cur->m_aSplits <= p2Cur);
         EBM_ASSERT(p1Cur <= pTopCur);
         EBM_ASSERT(static_cast<size_t>(p2Cur - pDimension2Cur->m_aSplits) <= static_cast<size_t>(pTopCur - pDimension1Cur->m_aSplits));

         if(UNLIKELY(pTopCur == p1Cur)) {
            // since we've finished the rhs splits, our CompressibleTensor already has the right splits in place, so all we need is to add the score
            // of the last region in rhs to our remaining scores
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

   size_t cScores = m_cVectorLength;
   for(size_t iDimension = 0; iDimension < m_cDimensions; ++iDimension) {
      const DimensionInfo * const pDimension1 = &pThisDimensionInfo[iDimension];
      const DimensionInfo * const pDimension2 = &pRhsDimensionInfo[iDimension];

      size_t cSplits = pDimension1->m_cSplits;
      if(cSplits != pDimension2->m_cSplits) {
         return false;
      }

      if(0 != cSplits) {
         EBM_ASSERT(!IsMultiplyError(cScores, cSplits + 1)); // we're accessing allocated memory, so it can't overflow
         cScores *= cSplits + 1;

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

   const FloatFast * pV1Cur = &m_aScores[0];
   const FloatFast * pV2Cur = &rhs.m_aScores[0];
   const FloatFast * const pV1End = pV1Cur + cScores;
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
