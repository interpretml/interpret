// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <algorithm> // sort

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "CompressibleTensor.hpp"
#include "ebm_stats.hpp"

#include "Feature.hpp"
#include "FeatureGroup.hpp"

#include "HistogramTargetEntry.hpp"
#include "HistogramBucket.hpp"

#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class PartitionRandomBoostingInternal final {
public:

   PartitionRandomBoostingInternal() = delete; // this is a static class.  Do not construct

   static ErrorEbmType Func(
      BoosterShell * const pBoosterShell,
      const FeatureGroup * const pFeatureGroup,
      const GenerateUpdateOptionsType options,
      const IntEbmType * const aLeavesMax,
      double * const pTotalGain
   ) {
      // THIS RANDOM SPLIT FUNCTION IS PRIMARILY USED FOR DIFFERENTIAL PRIVACY EBMs

      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      // TODO: add a new random_rety option that will retry random splitting for N times and select the one with the best gain
      // TODO: accept the minimum number of items in a split and then refuse to allow the split if we violate it, or
      //       provide a soft trigger that generates 10 random ones and selects the one that violates the least
      //       maybe provide a flag to indicate if we want a hard or soft allowance.  We won't be splitting if we
      //       require a soft allowance and a lot of regions have zeros.
      // TODO: accept 0 == countSamplesRequiredForChildSplitMin as a minimum number of items so that we can always choose to allow a tensor split (for DP)
      // TODO: move most of this code out of this function into a non-templated place

      ErrorEbmType error;
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses()
      );

      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      EBM_ASSERT(!GetHistogramBucketSizeOverflow<FloatBig>(bClassification, cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<FloatBig>(bClassification, cVectorLength);

      HistogramBucketBase * const aHistogramBucketsBase = pBoosterShell->GetHistogramBucketBaseBig();
      auto * const aHistogramBuckets = aHistogramBucketsBase->GetHistogramBucket<FloatBig, bClassification>();

      EBM_ASSERT(1 <= pFeatureGroup->GetCountSignificantDimensions());
      EBM_ASSERT(1 <= pFeatureGroup->GetCountDimensions());

      CompressibleTensor * const pSmallChangeToModelOverwriteSingleSamplingSet =
         pBoosterShell->GetOverwritableModelUpdate();

      const IntEbmType * pLeavesMax1 = aLeavesMax;
      const FeatureGroupEntry * pFeatureGroupEntry1 = pFeatureGroup->GetFeatureGroupEntries();
      const FeatureGroupEntry * const pFeatureGroupEntryEnd = pFeatureGroupEntry1 + pFeatureGroup->GetCountDimensions();
      size_t cSlicesTotal = 0;
      size_t cSlicesPlusRandomMax = 0;
      size_t cCollapsedTensorCells = 1;
      do {
         size_t cLeavesMax;
         if(nullptr == pLeavesMax1) {
            cLeavesMax = size_t { 1 };
         } else {
            const IntEbmType countLeavesMax = *pLeavesMax1;
            ++pLeavesMax1;
            if(countLeavesMax <= IntEbmType { 1 }) {
               cLeavesMax = size_t { 1 };
            } else {
               cLeavesMax = static_cast<size_t>(countLeavesMax);
               if(IsConvertError<size_t>(countLeavesMax)) {
                  // we can never exceed a size_t number of leaves, so let's just set it to the maximum if we 
                  // were going to overflow because it will generate the same results as if we used the true number
                  cLeavesMax = std::numeric_limits<size_t>::max();
               }
            }
         }

         const Feature * const pFeature = pFeatureGroupEntry1->m_pFeature;
         const size_t cBins = pFeature->GetCountBins();
         EBM_ASSERT(size_t { 1 } <= cBins); // we don't boost on empty training sets
         const size_t cSlices = EbmMin(cLeavesMax, cBins);
         const size_t cPossibleSplitLocations = cBins - size_t { 1 };
         if(size_t { 0 } < cPossibleSplitLocations) {
            // drop any dimensions with 1 bin since the tensor is the same without the extra dimension

            if(IsAddError(cSlicesTotal, cPossibleSplitLocations)) {
               LOG_0(TraceLevelWarning, "WARNING PartitionRandomBoostingInternal IsAddError(cSlicesTotal, cPossibleSplitLocations)");
               return Error_OutOfMemory;
            }
            const size_t cSlicesPlusRandom = cSlicesTotal + cPossibleSplitLocations;
            cSlicesPlusRandomMax = EbmMax(cSlicesPlusRandomMax, cSlicesPlusRandom);

            // our histogram is a tensor where we multiply the number of cells on each pass.  Addition of those 
            // same numbers can't be bigger than multiplication unless one of the dimensions is less than 2 wide.  
            // At 2, multiplication and addition would yield the same size.  All other numbers will be bigger for 
            // multiplication, so we can conclude that addition won't overflow since the multiplication didn't
            EBM_ASSERT(!IsAddError(cSlicesTotal, cSlices));
            cSlicesTotal += cSlices;

            EBM_ASSERT(!IsMultiplyError(cCollapsedTensorCells, cSlices)); // our allocated histogram is bigger
            cCollapsedTensorCells *= cSlices;
         }
         ++pFeatureGroupEntry1;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry1);

      // since we subtract 1 from cPossibleSplitLocations, we need to check that our final slice length isn't longer
      cSlicesPlusRandomMax = EbmMax(cSlicesPlusRandomMax, cSlicesTotal);

      if(IsMultiplyError(sizeof(size_t), cSlicesPlusRandomMax)) {
         LOG_0(TraceLevelWarning, "WARNING PartitionRandomBoostingInternal IsMultiplyError(sizeof(size_t), cSlicesPlusRandomMax)");
         return Error_OutOfMemory;
      }
      const size_t cBytesSlicesPlusRandom = sizeof(size_t) * cSlicesPlusRandomMax;

      error = pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * cCollapsedTensorCells);
      if(UNLIKELY(Error_None != error)) {
         // already logged
         return error;
      }

      // our allocated histogram is bigger since it has more elements and the elements contain a size_t
      EBM_ASSERT(!IsMultiplyError(sizeof(size_t), cSlicesTotal));
      const size_t cBytesSlices = sizeof(size_t) * cSlicesTotal;

      // promote to bytes
      EBM_ASSERT(!IsMultiplyError(cBytesPerHistogramBucket, cCollapsedTensorCells)); // our allocated histogram is bigger
      cCollapsedTensorCells *= cBytesPerHistogramBucket;
      if(IsAddError(cBytesSlices, cCollapsedTensorCells)) {
         LOG_0(TraceLevelWarning, "WARNING PartitionRandomBoostingInternal IsAddError(cBytesSlices, cBytesCollapsedTensor1)");
         return Error_OutOfMemory;
      }
      const size_t cBytesSlicesAndCollapsedTensor = cBytesSlices + cCollapsedTensorCells;

      const size_t cBytesBuffer = EbmMax(cBytesSlicesAndCollapsedTensor, cBytesSlicesPlusRandom);

      // TODO: use GrowThreadByteBuffer2 for this, but first we need to change that to allocate void or bytes
      char * const pBuffer = static_cast<char *>(malloc(cBytesBuffer));
      if(UNLIKELY(nullptr == pBuffer)) {
         LOG_0(TraceLevelWarning, "WARNING PartitionRandomBoostingInternal nullptr == pBuffer");
         return Error_OutOfMemory;
      }
      size_t * const acItemsInNextSliceOrBytesInCurrentSlice = reinterpret_cast<size_t *>(pBuffer);

      const IntEbmType * pLeavesMax2 = aLeavesMax;
      RandomStream * const pRandomStream = pBoosterShell->GetRandomStream();
      size_t * pcItemsInNextSliceOrBytesInCurrentSlice2 = acItemsInNextSliceOrBytesInCurrentSlice;
      const FeatureGroupEntry * pFeatureGroupEntry2 = pFeatureGroup->GetFeatureGroupEntries();
      do {
         size_t cTreeSplitsMax;
         if(nullptr == pLeavesMax2) {
            cTreeSplitsMax = size_t { 0 };
         } else {
            const IntEbmType countLeavesMax = *pLeavesMax2;
            ++pLeavesMax2;
            if(countLeavesMax <= IntEbmType { 1 }) {
               cTreeSplitsMax = size_t { 0 };
            } else {
               cTreeSplitsMax = static_cast<size_t>(countLeavesMax) - size_t { 1 };
               if(IsConvertError<size_t>(countLeavesMax)) {
                  // we can never exceed a size_t number of leaves, so let's just set it to the maximum if we 
                  // were going to overflow because it will generate the same results as if we used the true number
                  cTreeSplitsMax = std::numeric_limits<size_t>::max() - size_t { 1 };
               }
            }
         }

         const Feature * const pFeature = pFeatureGroupEntry2->m_pFeature;
         const size_t cBins = pFeature->GetCountBins();
         EBM_ASSERT(size_t { 1 } <= cBins); // we don't boost on empty training sets
         size_t cPossibleSplitLocations = cBins - size_t { 1 };
         if(size_t { 0 } < cPossibleSplitLocations) {
            // drop any dimensions with 1 bin since the tensor is the same without the extra dimension

            if(size_t { 0 } != cTreeSplitsMax) {
               size_t * pFillIndexes = pcItemsInNextSliceOrBytesInCurrentSlice2;
               size_t iPossibleSplitLocations = cPossibleSplitLocations; // 1 means split between bin 0 and bin 1
               do {
                  *pFillIndexes = iPossibleSplitLocations;
                  ++pFillIndexes;
                  --iPossibleSplitLocations;
               } while(size_t { 0 } != iPossibleSplitLocations);

               size_t * pOriginal = pcItemsInNextSliceOrBytesInCurrentSlice2;

               const size_t cSplits = EbmMin(cTreeSplitsMax, cPossibleSplitLocations);
               EBM_ASSERT(1 <= cSplits);
               const size_t * const pcItemsInNextSliceOrBytesInCurrentSliceEnd = pcItemsInNextSliceOrBytesInCurrentSlice2 + cSplits;
               do {
                  const size_t iRandom = pRandomStream->Next(cPossibleSplitLocations);
                  size_t * const pRandomSwap = pcItemsInNextSliceOrBytesInCurrentSlice2 + iRandom;
                  const size_t temp = *pRandomSwap;
                  *pRandomSwap = *pcItemsInNextSliceOrBytesInCurrentSlice2;
                  *pcItemsInNextSliceOrBytesInCurrentSlice2 = temp;
                  --cPossibleSplitLocations;
                  ++pcItemsInNextSliceOrBytesInCurrentSlice2;
               } while(pcItemsInNextSliceOrBytesInCurrentSliceEnd != pcItemsInNextSliceOrBytesInCurrentSlice2);

               std::sort(pOriginal, pcItemsInNextSliceOrBytesInCurrentSlice2);
            }
            *pcItemsInNextSliceOrBytesInCurrentSlice2 = cBins; // index 1 past the last item
            ++pcItemsInNextSliceOrBytesInCurrentSlice2;
         }
         ++pFeatureGroupEntry2;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry2);

      const IntEbmType * pLeavesMax3 = aLeavesMax;
      const size_t * pcBytesInSliceEnd;
      const FeatureGroupEntry * pFeatureGroupEntry3 = pFeatureGroup->GetFeatureGroupEntries();
      size_t * pcItemsInNextSliceOrBytesInCurrentSlice3 = acItemsInNextSliceOrBytesInCurrentSlice;
      size_t cBytesCollapsedTensor3;
      while(true) {
         EBM_ASSERT(pFeatureGroupEntry3 < pFeatureGroupEntryEnd);

         size_t cLeavesMax;
         if(nullptr == pLeavesMax3) {
            cLeavesMax = size_t { 1 };
         } else {
            const IntEbmType countLeavesMax = *pLeavesMax3;
            ++pLeavesMax3;
            if(countLeavesMax <= IntEbmType { 1 }) {
               cLeavesMax = size_t { 1 };
            } else {
               cLeavesMax = static_cast<size_t>(countLeavesMax);
               if(IsConvertError<size_t>(countLeavesMax)) {
                  // we can never exceed a size_t number of leaves, so let's just set it to the maximum if we 
                  // were going to overflow because it will generate the same results as if we used the true number
                  cLeavesMax = std::numeric_limits<size_t>::max();
               }
            }
         }

         // the first dimension is special.  we put byte until next item into it instead of counts remaining
         const Feature * const pFirstFeature = pFeatureGroupEntry3->m_pFeature;
         ++pFeatureGroupEntry3;
         const size_t cFirstBins = pFirstFeature->GetCountBins();
         EBM_ASSERT(size_t { 1 } <= cFirstBins); // we don't boost on empty training sets
         if(size_t { 1 } < cFirstBins) {
            // drop any dimensions with 1 bin since the tensor is the same without the extra dimension

            const size_t cFirstSlices = EbmMin(cLeavesMax, cFirstBins);
            cBytesCollapsedTensor3 = cBytesPerHistogramBucket * cFirstSlices;

            pcBytesInSliceEnd = acItemsInNextSliceOrBytesInCurrentSlice + cFirstSlices;
            size_t iPrev = size_t { 0 };
            do {
               const size_t iCur = *pcItemsInNextSliceOrBytesInCurrentSlice3;
               EBM_ASSERT(iPrev < iCur);
               // turn these into bytes from the previous
               *pcItemsInNextSliceOrBytesInCurrentSlice3 = (iCur - iPrev) * cBytesPerHistogramBucket;
               iPrev = iCur;
               ++pcItemsInNextSliceOrBytesInCurrentSlice3;
            } while(pcBytesInSliceEnd != pcItemsInNextSliceOrBytesInCurrentSlice3);

            // we found a non-eliminated dimension.  We treat the first dimension differently from others, so
            // if our first dimension is eliminated we need to keep looking until we find our first REAL dimension
            break;
         }
      }

      struct RandomSplitState {
         size_t         m_cItemsInSliceRemaining;
         size_t         m_cBytesSubtractResetCollapsedHistogramBucket;

         const size_t * m_pcItemsInNextSlice;
         const size_t * m_pcItemsInNextSliceEnd;
      };
      RandomSplitState randomSplitState[k_cDimensionsMax - size_t { 1 }]; // the first dimension is special cased
      RandomSplitState * pStateInit = &randomSplitState[0];

      for(; pFeatureGroupEntryEnd != pFeatureGroupEntry3; ++pFeatureGroupEntry3) {
         size_t cLeavesMax;
         if(nullptr == pLeavesMax3) {
            cLeavesMax = size_t { 1 };
         } else {
            const IntEbmType countLeavesMax = *pLeavesMax3;
            ++pLeavesMax3;
            if(countLeavesMax <= IntEbmType { 1 }) {
               cLeavesMax = size_t { 1 };
            } else {
               cLeavesMax = static_cast<size_t>(countLeavesMax);
               if(IsConvertError<size_t>(countLeavesMax)) {
                  // we can never exceed a size_t number of leaves, so let's just set it to the maximum if we 
                  // were going to overflow because it will generate the same results as if we used the true number
                  cLeavesMax = std::numeric_limits<size_t>::max();
               }
            }
         }

         const Feature * const pFeature = pFeatureGroupEntry3->m_pFeature;
         const size_t cBins = pFeature->GetCountBins();
         EBM_ASSERT(size_t { 1 } <= cBins); // we don't boost on empty training sets
         if(size_t { 1 } < cBins) {
            // drop any dimensions with 1 bin since the tensor is the same without the extra dimension

            size_t cSlices = EbmMin(cLeavesMax, cBins);

            pStateInit->m_cBytesSubtractResetCollapsedHistogramBucket = cBytesCollapsedTensor3;

            EBM_ASSERT(!IsMultiplyError(cBytesCollapsedTensor3, cSlices)); // our allocated histogram is bigger
            cBytesCollapsedTensor3 *= cSlices;

            const size_t iFirst = *pcItemsInNextSliceOrBytesInCurrentSlice3;
            EBM_ASSERT(1 <= iFirst);
            pStateInit->m_cItemsInSliceRemaining = iFirst;
            pStateInit->m_pcItemsInNextSlice = pcItemsInNextSliceOrBytesInCurrentSlice3;

            size_t iPrev = iFirst;
            for(--cSlices; LIKELY(size_t { 0 } != cSlices); --cSlices) {
               size_t * const pCur = pcItemsInNextSliceOrBytesInCurrentSlice3 + size_t { 1 };
               const size_t iCur = *pCur;
               EBM_ASSERT(iPrev < iCur);
               *pcItemsInNextSliceOrBytesInCurrentSlice3 = iCur - iPrev;
               iPrev = iCur;
               pcItemsInNextSliceOrBytesInCurrentSlice3 = pCur;
            }
            *pcItemsInNextSliceOrBytesInCurrentSlice3 = iFirst;
            ++pcItemsInNextSliceOrBytesInCurrentSlice3;
            pStateInit->m_pcItemsInNextSliceEnd = pcItemsInNextSliceOrBytesInCurrentSlice3;
            ++pStateInit;
         }
      }

      // put the histograms right after our slice array
      auto * const aCollapsedHistogramBuckets =
         reinterpret_cast<HistogramBucket<FloatBig, bClassification> *>(pcItemsInNextSliceOrBytesInCurrentSlice3);

      // TODO: move this into a helper function on the histogram bucket object that zeros N bytes (if we know the bytes).  Mostly as a warning to understand where we're using memset

      // C standard guarantees that zeroing integer types (size_t) is a zero, and IEEE 754 guarantees 
      // that zeroing a floating point is zero.  Our HistogramBucket objects are POD and also only contain
      // floating point types and size_t
      //
      // 6.2.6.2 Integer types -> 5. The values of any padding bits are unspecified.A valid (non - trap) 
      // object representation of a signed integer type where the sign bit is zero is a valid object 
      // representation of the corresponding unsigned type, and shall represent the same value.For any 
      // integer type, the object representation where all the bits are zero shall be a representation 
      // of the value zero in that type.
      static_assert(std::numeric_limits<float>::is_iec559, "memset of floats requires IEEE 754 to guarantee zeros");
      memset(aCollapsedHistogramBuckets, 0, cBytesCollapsedTensor3);
      const auto * const pCollapsedHistogramBucketEnd =
         reinterpret_cast<HistogramBucket<FloatBig, bClassification> *>(reinterpret_cast<char *>(aCollapsedHistogramBuckets) +
            cBytesCollapsedTensor3);

      // we special case the first dimension, so drop it by subtracting
      EBM_ASSERT(&randomSplitState[pFeatureGroup->GetCountSignificantDimensions() - size_t { 1 }] == pStateInit);

      const auto * pHistogramBucket = aHistogramBuckets;
      auto * pCollapsedHistogramBucket1 = aCollapsedHistogramBuckets;

      {
      move_next_slice:;

         // for the first dimension, acItemsInNextSliceOrBytesInCurrentSlice contains the number of bytes to proceed 
         // until the next pHistogramBucketSliceEnd point.  For the second dimension and higher, it contains a 
         // count of items for the NEXT slice.  The 0th element contains the count of items for the
         // 1st slice.  Yeah, it's pretty confusing, but it allows for some pretty compact code in this
         // super critical inner loop without overburdening the CPU registers when we execute the outer loop.
         const size_t * pcItemsInNextSliceOrBytesInCurrentSlice = acItemsInNextSliceOrBytesInCurrentSlice;
         do {
            const auto * const pHistogramBucketSliceEnd =
               reinterpret_cast<const HistogramBucket<FloatBig, bClassification> *>(
               reinterpret_cast<const char *>(pHistogramBucket) + *pcItemsInNextSliceOrBytesInCurrentSlice);

            do {
               ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucket, pBoosterShell->GetHistogramBucketsEndDebugBig());
               pCollapsedHistogramBucket1->Add(*pHistogramBucket, cVectorLength);

               // we're walking through all buckets, so just move to the next one in the flat array, 
               // with the knowledge that we'll figure out it's multi-dimenional index below
               pHistogramBucket = 
                  GetHistogramBucketByIndex(cBytesPerHistogramBucket, pHistogramBucket, 1);

            } while(LIKELY(pHistogramBucketSliceEnd != pHistogramBucket));

            pCollapsedHistogramBucket1 = 
               GetHistogramBucketByIndex(cBytesPerHistogramBucket, pCollapsedHistogramBucket1, 1);

            ++pcItemsInNextSliceOrBytesInCurrentSlice;
         } while(PREDICTABLE(pcBytesInSliceEnd != pcItemsInNextSliceOrBytesInCurrentSlice));

         for(RandomSplitState * pState = randomSplitState; PREDICTABLE(pStateInit != pState); ++pState) {
            EBM_ASSERT(size_t { 1 } <= pState->m_cItemsInSliceRemaining);
            const size_t cItemsInSliceRemaining = pState->m_cItemsInSliceRemaining - size_t { 1 };
            if(LIKELY(size_t { 0 } != cItemsInSliceRemaining)) {
               // ideally, the compiler would move this to the location right above the first loop and it would
               // jump over it on the first loop, but I wasn't able to make the Visual Studio compiler do it

               pState->m_cItemsInSliceRemaining = cItemsInSliceRemaining;
               pCollapsedHistogramBucket1 = reinterpret_cast<HistogramBucket<FloatBig, bClassification> *>(
                  reinterpret_cast<char *>(pCollapsedHistogramBucket1) -
                  pState->m_cBytesSubtractResetCollapsedHistogramBucket);

               goto move_next_slice;
            }

            const size_t * pcItemsInNextSlice = pState->m_pcItemsInNextSlice;
            EBM_ASSERT(pcItemsInNextSliceOrBytesInCurrentSlice <= pcItemsInNextSlice);
            EBM_ASSERT(pcItemsInNextSlice < pState->m_pcItemsInNextSliceEnd);
            pState->m_cItemsInSliceRemaining = *pcItemsInNextSlice;
            ++pcItemsInNextSlice;
            // it would be legal for us to move this assignment into the if statement below, since if we don't
            // enter the if statement we overwrite the value that we just wrote, but writing it here allows the
            // compiler to emit a sinlge jne instruction to move to move_next_slice without using an extra
            // jmp instruction.  Typically we have 3 slices, so we avoid 2 jmp instructions for the cost of
            // 1 extra assignment that'll happen 1/3 of the time when m_pcItemsInNextSliceEnd == pcItemsInNextSlice
            // Something would have to be wrong for us to have less than 2 slices since then we'd be ignoring a
            // dimension, so even in the realistic worst case the 1 jmp instruction balances the extra mov instruction
            pState->m_pcItemsInNextSlice = pcItemsInNextSlice;
            if(UNPREDICTABLE(pState->m_pcItemsInNextSliceEnd != pcItemsInNextSlice)) {
               goto move_next_slice;
            }
            // the end of the previous dimension is the start of our current one
            pState->m_pcItemsInNextSlice = pcItemsInNextSliceOrBytesInCurrentSlice;
            pcItemsInNextSliceOrBytesInCurrentSlice = pcItemsInNextSlice;
         }
      }

      //TODO: retrieve the gain.  Always calculate the gain without respect to the parent and pick the best one
      //      Then, before exiting, on the last one we collapse the collapsed tensor even more into just a single
      //      bin from which we can calculate the parent and subtract the best child from the parent.
      
      //FloatBig gain;
      //FloatBig gainParent = FloatBig { 0 };
      FloatBig gain = 0;


      const FeatureGroupEntry * pFeatureGroupEntry4 = pFeatureGroup->GetFeatureGroupEntries();
      size_t iDimensionWrite = ~size_t { 0 }; // this is -1, but without the compiler warning
      size_t cBinsWrite;
      do {
         cBinsWrite = pFeatureGroupEntry4->m_pFeature->GetCountBins();
         ++iDimensionWrite;
         ++pFeatureGroupEntry4;
      } while(cBinsWrite <= size_t { 1 });

      const size_t * const pcBytesInSliceLast = pcBytesInSliceEnd - size_t { 1 };
      EBM_ASSERT(acItemsInNextSliceOrBytesInCurrentSlice <= pcBytesInSliceLast);
      const size_t cFirstSplits = pcBytesInSliceLast - acItemsInNextSliceOrBytesInCurrentSlice;
      // 3 items in the acItemsInNextSliceOrBytesInCurrentSlice means 2 splits and 
      // one last item to indicate the termination point
      error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountSplits(iDimensionWrite, cFirstSplits);
      if(UNLIKELY(Error_None != error)) {
         // already logged
         free(pBuffer);
         return error;
      }
      const size_t * pcBytesInSlice2 = acItemsInNextSliceOrBytesInCurrentSlice;
      if(LIKELY(size_t { 0 } != cFirstSplits)) {
         ActiveDataType * pSplitFirst = pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimensionWrite);
         // converting negative to positive number is defined behavior in C++ and uses twos compliment
         size_t iSplitFirst = static_cast<size_t>(ptrdiff_t { -1 });
         do {
            EBM_ASSERT(pcBytesInSlice2 < pcBytesInSliceLast);
            EBM_ASSERT(0 != *pcBytesInSlice2);
            EBM_ASSERT(0 == *pcBytesInSlice2 % cBytesPerHistogramBucket);
            iSplitFirst += *pcBytesInSlice2 / cBytesPerHistogramBucket;
            *pSplitFirst = iSplitFirst;
            ++pSplitFirst;
            ++pcBytesInSlice2;
            // the last one is the distance to the end, which we don't include in the update
         } while(LIKELY(pcBytesInSliceLast != pcBytesInSlice2));
      }

      RandomSplitState * pState = randomSplitState;
      if(PREDICTABLE(pStateInit != pState)) {
         do {
            do {
               cBinsWrite = pFeatureGroupEntry4->m_pFeature->GetCountBins();
               ++iDimensionWrite;
               ++pFeatureGroupEntry4;
            } while(cBinsWrite <= size_t { 1 });

            ++pcBytesInSlice2; // we have one less split than we have slices, so move to the next one

            const size_t * pcItemsInNextSliceLast = pState->m_pcItemsInNextSliceEnd - size_t { 1 };
            error = pSmallChangeToModelOverwriteSingleSamplingSet->SetCountSplits(iDimensionWrite, pcItemsInNextSliceLast - pcBytesInSlice2);
            if(Error_None != error) {
               // already logged
               free(pBuffer);
               return error;
            }
            if(pcItemsInNextSliceLast != pcBytesInSlice2) {
               ActiveDataType * pSplit = pSmallChangeToModelOverwriteSingleSamplingSet->GetSplitPointer(iDimensionWrite);
               size_t iSplit2 = *pcItemsInNextSliceLast - size_t { 1 };
               *pSplit = iSplit2;
               --pcItemsInNextSliceLast;
               while(pcItemsInNextSliceLast != pcBytesInSlice2) {
                  iSplit2 += *pcBytesInSlice2;
                  ++pSplit;
                  *pSplit = iSplit2;
                  ++pcBytesInSlice2;
               }
               // increment it once more because our indexes are shifted such that the first one was the last item
               ++pcBytesInSlice2;
            }
            ++pState;
         } while(PREDICTABLE(pStateInit != pState));
      }

      FloatFast * pUpdate = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
      auto * pCollapsedHistogramBucket2 = aCollapsedHistogramBuckets;

      if(0 != (GenerateUpdateOptions_GradientSums & options)) {
         do {
            auto * const pHistogramTargetEntry = pCollapsedHistogramBucket2->GetHistogramTargetEntry();

            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               FloatBig update = EbmStats::ComputeSinglePartitionUpdateGradientSum(pHistogramTargetEntry[iVector].m_sumGradients);

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
               // for DP-EBMs, we can't zero one of the class scores as we can for logits since we're returning a sum
#endif // ZERO_FIRST_MULTICLASS_LOGIT

               *pUpdate = SafeConvertFloat<FloatFast>(update);
               ++pUpdate;
            }
            pCollapsedHistogramBucket2 = GetHistogramBucketByIndex(cBytesPerHistogramBucket, pCollapsedHistogramBucket2, 1);
         } while(pCollapsedHistogramBucketEnd != pCollapsedHistogramBucket2);
      } else {
         do {
            const size_t cSamples = pCollapsedHistogramBucket2->GetCountSamplesInBucket();
            if(UNLIKELY(size_t { 0 } == cSamples)) {
               // TODO: this section can probably be eliminated since ComputeSinglePartitionUpdate now checks
               // for zero in the denominator, but I'm leaving it here to see how the removal of the 
               // GetCountSamplesInBucket property works in the future in combination with the check on hessians

               // normally, we'd eliminate regions where the number of items was zero before putting down a split
               // but for random splits we can't know beforehand if there will be zero splits, so we need to check
               for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
#ifdef ZERO_FIRST_MULTICLASS_LOGIT
                  // if we eliminated the space for the logit, we'd need to eliminate one assignment here
#endif // ZERO_FIRST_MULTICLASS_LOGIT

                  *pUpdate = 0;
                  ++pUpdate;
               }
            } else {
               auto * const pHistogramTargetEntry = pCollapsedHistogramBucket2->GetHistogramTargetEntry();

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
               FloatBig zeroLogit = 0;
#endif // ZERO_FIRST_MULTICLASS_LOGIT

               for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
                  FloatBig update;
                  if(bClassification) {
                     update = EbmStats::ComputeSinglePartitionUpdate(
                        pHistogramTargetEntry[iVector].m_sumGradients,
                        pHistogramTargetEntry[iVector].GetSumHessians()
                     );
#ifdef ZERO_FIRST_MULTICLASS_LOGIT
                     if(IsMulticlass(compilerLearningTypeOrCountTargetClasses)) {
                        if(size_t { 0 } == iVector) {
                           zeroLogit = update;
                        }
                        update -= zeroLogit;
                     }
#endif // ZERO_FIRST_MULTICLASS_LOGIT
                  } else {
                     EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
                     update = EbmStats::ComputeSinglePartitionUpdate(
                        pHistogramTargetEntry[iVector].m_sumGradients,
                        pCollapsedHistogramBucket2->GetWeightInBucket()
                     );
                  }
                  *pUpdate = SafeConvertFloat<FloatFast>(update);
                  ++pUpdate;
               }
            }
            pCollapsedHistogramBucket2 = GetHistogramBucketByIndex(
               cBytesPerHistogramBucket, pCollapsedHistogramBucket2, 1);
         } while(pCollapsedHistogramBucketEnd != pCollapsedHistogramBucket2);
      }

      free(pBuffer);
      *pTotalGain = static_cast<double>(gain);
      return Error_None;
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class PartitionRandomBoostingTarget final {
public:

   PartitionRandomBoostingTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static ErrorEbmType Func(
      BoosterShell * const pBoosterShell,
      const FeatureGroup * const pFeatureGroup,
      const GenerateUpdateOptionsType options,
      const IntEbmType * const aLeavesMax,
      double * const pTotalGain
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         return PartitionRandomBoostingInternal<compilerLearningTypeOrCountTargetClassesPossible>::Func(
            pBoosterShell,
            pFeatureGroup,
            options,
            aLeavesMax,
            pTotalGain
         );
      } else {
         return PartitionRandomBoostingTarget<compilerLearningTypeOrCountTargetClassesPossible + 1>::Func(
            pBoosterShell,
            pFeatureGroup,
            options,
            aLeavesMax,
            pTotalGain
         );
      }
   }
};

template<>
class PartitionRandomBoostingTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   PartitionRandomBoostingTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static ErrorEbmType Func(
      BoosterShell * const pBoosterShell,
      const FeatureGroup * const pFeatureGroup,
      const GenerateUpdateOptionsType options,
      const IntEbmType * const aLeavesMax,
      double * const pTotalGain
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pBoosterShell->GetBoosterCore()->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pBoosterShell->GetBoosterCore()->GetRuntimeLearningTypeOrCountTargetClasses());

      return PartitionRandomBoostingInternal<k_dynamicClassification>::Func(
         pBoosterShell,
         pFeatureGroup,
         options,
         aLeavesMax,
         pTotalGain
      );
   }
};

extern ErrorEbmType PartitionRandomBoosting(
   BoosterShell * const pBoosterShell,
   const FeatureGroup * const pFeatureGroup,
   const GenerateUpdateOptionsType options,
   const IntEbmType * const aLeavesMax,
   double * const pTotalGain
) {
   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();

   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      return PartitionRandomBoostingTarget<2>::Func(
         pBoosterShell,
         pFeatureGroup,
         options,
         aLeavesMax,
         pTotalGain
      );
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      return PartitionRandomBoostingInternal<k_regression>::Func(
         pBoosterShell,
         pFeatureGroup,
         options,
         aLeavesMax,
         pTotalGain
      );
   }
}

} // DEFINED_ZONE_NAME
