// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "SegmentedTensor.h"
#include "EbmStatisticUtils.h"

#include "FeatureAtomic.h"
#include "FeatureGroup.h"

#include "HistogramTargetEntry.h"
#include "HistogramBucket.h"

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class CutRandomInternal final {
public:

   CutRandomInternal() = delete; // this is a static class.  Do not construct

   static bool Func(
      EbmBoostingState * const pEbmBoostingState,
      const FeatureGroup * const pFeatureGroup,
      HistogramBucketBase * const aHistogramBucketsBase,
      const size_t cTreeSplitsMax,
      const GenerateUpdateOptionsType options,
      SegmentedTensor * const pSmallChangeToModelOverwriteSingleSamplingSet,
      FloatEbmType * const pTotalGain
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      // THIS RANDOM CUT FUNCTION IS PRIMARILY USED FOR DIFFERENTIAL PRIVACY EBMs

      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      // TODO: accept the minimum number of items in a cut and then refuse to allow the cut if we violate it
      // TODO: accept 0 as a minimum number of items so that we can always choose to allow a tensor cut (for DP)
      // TODO: we'll need to check our denominator if we allow 0 cuts since we'll have 0/0 then
      // TODO: add a new random_rety option that will retry random cutting for N times and select the one with the best gain
      // TODO: move most of this code out of this function into a non-templated place

      const size_t cLeavesMax = cTreeSplitsMax + size_t { 1 }; // TODO: this should be done by our caller in the future

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses();

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses()
      );

      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      EBM_ASSERT(!GetHistogramBucketSizeOverflow(bClassification, cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);

      HistogramBucket<bClassification> * const aHistogramBuckets = aHistogramBucketsBase->GetHistogramBucket<bClassification>();

      const size_t cDimensions = pFeatureGroup->GetCountFeatures();
      EBM_ASSERT(1 <= cDimensions);

      const FeatureGroupEntry * pFeatureGroupEntry1 = pFeatureGroup->GetFeatureGroupEntries();
      const FeatureGroupEntry * const pFeatureGroupEntryEnd = pFeatureGroupEntry1 + cDimensions;
      size_t cSlicesTotal = 0;
      size_t cSlicesPlusRandomMax = 0;
      size_t cCollapsedTensorCells = 1;
      do {
         const Feature * const pFeature = pFeatureGroupEntry1->m_pFeature;
         const size_t cBins = pFeature->GetCountBins();

         // TODO: test that 2 <= cBins is always true

         EBM_ASSERT(2 <= cBins); // otherwise this dimension would have been eliminated
         const size_t cSlices = EbmMin(cLeavesMax, cBins);
         const size_t cPossibleCutLocations = cBins - size_t { 1 };

         if(IsAddError(cSlicesTotal, cPossibleCutLocations)) {
            LOG_0(TraceLevelWarning, "WARNING CutRandomInternal IsAddError(cSlicesTotal, cPossibleCutLocations)");
            return true;
         }
         const size_t cSlicesPlusRandom = cSlicesTotal + cPossibleCutLocations;
         cSlicesPlusRandomMax = EbmMax(cSlicesPlusRandomMax, cSlicesPlusRandom);

         // our histogram is a tensor where we multiply the number of cells on each pass.  Addition of those 
         // same numbers can't be bigger than multiplication unless one of the dimensions is less than 2 wide.  
         // At 2, multiplication and addition would yield the same size.  All other numbers will be bigger for 
         // multiplication, so we can conclude that addition won't overflow since the multiplication didn't
         EBM_ASSERT(!IsAddError(cSlicesTotal, cSlices));
         cSlicesTotal += cSlices;

         EBM_ASSERT(!IsMultiplyError(cCollapsedTensorCells, cSlices)); // our allocated histogram is bigger
         cCollapsedTensorCells *= cSlices;

         ++pFeatureGroupEntry1;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry1);

      if(UNLIKELY(
         pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * cCollapsedTensorCells))) 
      {
         LOG_0(
            TraceLevelWarning,
            "WARNING CutRandomInternal pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * cCollapsedTensorCells)"
         );
         return true;
      }

      // since we subtract 1 from cPossibleCutLocations, we need to check that our final slice length isn't longer
      cSlicesPlusRandomMax = EbmMax(cSlicesPlusRandomMax, cSlicesTotal);

      if(IsMultiplyError(cSlicesPlusRandomMax, sizeof(size_t))) {
         LOG_0(TraceLevelWarning, "WARNING CutRandomInternal IsMultiplyError(cSlicesPlusRandomMax, sizeof(size_t))");
         return true;
      }
      const size_t cBytesSlicesPlusRandom = cSlicesPlusRandomMax * sizeof(size_t);

      // our allocated histogram is bigger since it has more elements and the elements contain a size_t
      EBM_ASSERT(!IsMultiplyError(cSlicesTotal, sizeof(size_t)));
      const size_t cBytesSlices = cSlicesTotal * sizeof(size_t);

      // promote to bytes
      EBM_ASSERT(!IsMultiplyError(cCollapsedTensorCells, cBytesPerHistogramBucket)); // our allocated histogram is bigger
      cCollapsedTensorCells *= cBytesPerHistogramBucket;
      if(IsAddError(cBytesSlices, cCollapsedTensorCells)) {
         LOG_0(TraceLevelWarning, "WARNING CutRandomInternal IsAddError(cBytesSlices, cBytesCollapsedTensor1)");
         return true;
      }
      const size_t cBytesSlicesAndCollapsedTensor = cBytesSlices + cCollapsedTensorCells;

      const size_t cBytesBuffer = EbmMax(cBytesSlicesAndCollapsedTensor, cBytesSlicesPlusRandom);

      // TODO: use GrowThreadByteBuffer2 for this, but first we need to change that to allocate void or bytes
      char * const pBuffer = static_cast<char *>(malloc(cBytesBuffer));
      if(UNLIKELY(nullptr == pBuffer)) {
         LOG_0(TraceLevelWarning, "WARNING CutRandomInternal nullptr == pBuffer");
         return true;
      }
      size_t * const acItemsInNextSliceOrBytesInCurrentSlice = reinterpret_cast<size_t *>(pBuffer);

      RandomStream * const pRandomStream = pEbmBoostingState->GetRandomStream();
      size_t * pcItemsInNextSliceOrBytesInCurrentSlice2 = acItemsInNextSliceOrBytesInCurrentSlice;
      const FeatureGroupEntry * pFeatureGroupEntry2 = pFeatureGroup->GetFeatureGroupEntries();
      do {
         const Feature * const pFeature = pFeatureGroupEntry2->m_pFeature;
         const size_t cBins = pFeature->GetCountBins();
         EBM_ASSERT(2 <= cBins); // otherwise this dimension would have been eliminated

         if(size_t { 0 } != cTreeSplitsMax) {
            size_t cPossibleCutLocations = cBins - size_t { 1 };
            size_t * pFillIndexes = pcItemsInNextSliceOrBytesInCurrentSlice2;
            size_t iPossibleCutLocation = cPossibleCutLocations; // 1 means cut between bin 0 and bin 1
            do {
               *pFillIndexes = iPossibleCutLocation;
               ++pFillIndexes;
               --iPossibleCutLocation;
            } while(0 != iPossibleCutLocation);

            size_t * pOriginal = pcItemsInNextSliceOrBytesInCurrentSlice2;

            const size_t cCuts = EbmMin(cTreeSplitsMax, cPossibleCutLocations);
            EBM_ASSERT(1 <= cCuts);
            size_t * const pcItemsInNextSliceOrBytesInCurrentSliceEnd = pcItemsInNextSliceOrBytesInCurrentSlice2 + cCuts;
            do {
               const size_t iRandom = pRandomStream->Next(cPossibleCutLocations);
               size_t * const pRandomSwap = pcItemsInNextSliceOrBytesInCurrentSlice2 + iRandom;
               const size_t temp = *pRandomSwap;
               *pRandomSwap = *pcItemsInNextSliceOrBytesInCurrentSlice2;
               *pcItemsInNextSliceOrBytesInCurrentSlice2 = temp;
               --cPossibleCutLocations;
               ++pcItemsInNextSliceOrBytesInCurrentSlice2;
            } while(pcItemsInNextSliceOrBytesInCurrentSliceEnd != pcItemsInNextSliceOrBytesInCurrentSlice2);

            std::sort(pOriginal, pcItemsInNextSliceOrBytesInCurrentSlice2);
         }
         *pcItemsInNextSliceOrBytesInCurrentSlice2 = cBins; // index 1 past the last item
         ++pcItemsInNextSliceOrBytesInCurrentSlice2;
         ++pFeatureGroupEntry2;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry2);

      const size_t * pcBytesInSliceEnd;
      const FeatureGroupEntry * pFeatureGroupEntry3 = pFeatureGroup->GetFeatureGroupEntries();
      size_t * pcItemsInNextSliceOrBytesInCurrentSlice3 = acItemsInNextSliceOrBytesInCurrentSlice;
      size_t cBytesCollapsedTensor3;
      {
         // the first dimension is special.  we put byte until next item into it instead of counts remaining
         const Feature * const pFirstFeature = pFeatureGroupEntry3->m_pFeature;
         const size_t cFirstBins = pFirstFeature->GetCountBins();
         EBM_ASSERT(2 <= cFirstBins); // otherwise this dimension would have been eliminated
         const size_t cFirstSlices = EbmMin(cLeavesMax, cFirstBins);
         cBytesCollapsedTensor3 = cBytesPerHistogramBucket * cFirstSlices;

         pcBytesInSliceEnd = acItemsInNextSliceOrBytesInCurrentSlice + cFirstSlices;
         size_t iPrev = 0;
         do {
            size_t iCur = *pcItemsInNextSliceOrBytesInCurrentSlice3;
            EBM_ASSERT(iPrev < iCur);
            // turn these into bytes from the previous
            *pcItemsInNextSliceOrBytesInCurrentSlice3 = (iCur - iPrev) * cBytesPerHistogramBucket;
            iPrev = iCur;
            ++pcItemsInNextSliceOrBytesInCurrentSlice3;
         } while(pcBytesInSliceEnd != pcItemsInNextSliceOrBytesInCurrentSlice3);
      }

      struct RandomCutState {
         // these change in value
         size_t         m_cItemsInSliceRemaining;
         const size_t * m_pcItemsInNextSlice;

         // these are constant after initialization
         const size_t * m_pcItemsInNextSliceEnd;
         size_t         m_cBytesSubtractResetCollapsedHistogramBucket;
      };
      RandomCutState randomCutState[k_cDimensionsMax - size_t { 1 }]; // the first dimension is special cased
      RandomCutState * pStateInit = &randomCutState[0];

      for(++pFeatureGroupEntry3; pFeatureGroupEntryEnd != pFeatureGroupEntry3; ++pFeatureGroupEntry3) {
         const Feature * const pFeature = pFeatureGroupEntry3->m_pFeature;
         const size_t cBins = pFeature->GetCountBins();
         EBM_ASSERT(2 <= cBins); // otherwise this dimension would have been eliminated
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

      // put the histograms right after our slice array
      HistogramBucket<bClassification> * const aCollapsedHistogramBuckets =
         reinterpret_cast<HistogramBucket<bClassification> *>(pcItemsInNextSliceOrBytesInCurrentSlice3);

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
      const HistogramBucket<bClassification> * const pCollapsedHistogramBucketEnd =
         reinterpret_cast<HistogramBucket<bClassification> *>(reinterpret_cast<char *>(aCollapsedHistogramBuckets) +
            cBytesCollapsedTensor3);

      // we special case the first dimension, so drop it by subtracting
      const RandomCutState * const pStateEnd = &randomCutState[cDimensions - size_t { 1 }];

      const HistogramBucket<bClassification> * pHistogramBucket = aHistogramBuckets;
      HistogramBucket<bClassification> * pCollapsedHistogramBucket1 = aCollapsedHistogramBuckets;

      // we'll move pHistogramBucketSliceEnd to the first slice end inside the loop
      const HistogramBucket<bClassification> * pHistogramBucketSliceEnd = pHistogramBucket;

      {
         goto move_next_slice;

      repeat_current_slice:;

         // Ok, this is a bit confusing, but we get here from calling "goto repeat_current_slice;" below
         // and we set these variables BEFORE calling goto, so that they have valid values
         //
         // putting this code here removes a jmp assembly instruction since the comparison jne operator
         // can jump here directly and we complete the work that we need to do before re-entering the loop.
         // The compiler may not do what we want here, and in fact Visual Studio puts this code inside the
         // if statement below and uses a jmp instruction, but it's worth a try in case a different 
         // compiler does what we want here.

         RandomCutState * pState;
         size_t cItemsInSliceRemaining;

         pState->m_cItemsInSliceRemaining = cItemsInSliceRemaining;
         pCollapsedHistogramBucket1 = reinterpret_cast<HistogramBucket<bClassification> *>(
            reinterpret_cast<char *>(pCollapsedHistogramBucket1) -
            pState->m_cBytesSubtractResetCollapsedHistogramBucket);

      move_next_slice:;

         // for the first dimension, acItemsInNextSliceOrBytesInCurrentSlice contains the number of bytes to proceed 
         // until the next pHistogramBucketSliceEnd point.  For the second dimension and higher, it contains a 
         // count of items for the NEXT slice.  The 0th element contains the count of items for the
         // 1st slice.  Yeah, it's pretty confusing, but it allows for some pretty compact code in this
         // super critical inner loop without overburdening the CPU registers when we execute the outer loop.
         const size_t * pcItemsInNextSliceOrBytesInCurrentSlice = acItemsInNextSliceOrBytesInCurrentSlice;
         do {
            pHistogramBucketSliceEnd = reinterpret_cast<const HistogramBucket<bClassification> *>(
               reinterpret_cast<const char *>(pHistogramBucketSliceEnd) + *pcItemsInNextSliceOrBytesInCurrentSlice);

            do {
               ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucket, aHistogramBucketsEndDebug);
               pCollapsedHistogramBucket1->Add(*pHistogramBucket, cVectorLength);

               // we're walking through all buckets, so just move to the next one in the flat array, 
               // with the knowledge that we'll figure out it's multi-dimenional index below
               pHistogramBucket = 
                  GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pHistogramBucket, 1);

            } while(LIKELY(pHistogramBucketSliceEnd != pHistogramBucket));

            pCollapsedHistogramBucket1 = 
               GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pCollapsedHistogramBucket1, 1);

            ++pcItemsInNextSliceOrBytesInCurrentSlice;
         } while(PREDICTABLE(pcBytesInSliceEnd != pcItemsInNextSliceOrBytesInCurrentSlice));

         for(pState = randomCutState; PREDICTABLE(pStateEnd != pState); ++pState) {
            EBM_ASSERT(size_t { 1 } <= pState->m_cItemsInSliceRemaining);
            cItemsInSliceRemaining = pState->m_cItemsInSliceRemaining - size_t { 1 };
            if(LIKELY(size_t { 0 } != cItemsInSliceRemaining)) {
               goto repeat_current_slice;
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





      //TODO: retrieve the gain by first calculating the gain on the collapsed histogram, then collapse the histogram
      //      into a single total bucket and calculate that and subtract

      //FloatEbmType splittingScore;
      //FloatEbmType splittingScoreParent = FloatEbmType { 0 };
      FloatEbmType gain = FloatEbmType { 0 };






      const size_t * pcBytesInSlice2 = acItemsInNextSliceOrBytesInCurrentSlice;
      const size_t * const pcBytesInSliceLast = pcBytesInSliceEnd - size_t { 1 };
      if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(
         0, pcBytesInSliceLast - acItemsInNextSliceOrBytesInCurrentSlice))) 
      {
         LOG_0(TraceLevelWarning, "WARNING CutRandomInternal SetCountDivisions(0, )");
         free(pBuffer);
         return true;
      }
      ActiveDataType * pDivisionFirst = pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0);
      // converting negative to positive number is defined behavior in C++ and uses twos compliment
      size_t iSplitFirst = static_cast<size_t>(ptrdiff_t { -1 });
      do {
         EBM_ASSERT(0 != *pcBytesInSlice2);
         EBM_ASSERT(0 == *pcBytesInSlice2 % cBytesPerHistogramBucket);
         iSplitFirst += *pcBytesInSlice2 / cBytesPerHistogramBucket;
         *pDivisionFirst = iSplitFirst;
         ++pDivisionFirst;
         ++pcBytesInSlice2;
         // the last one is the distance to the end, which we don't include in the update
      } while(LIKELY(pcBytesInSliceLast != pcBytesInSlice2));

      RandomCutState * pState = randomCutState;
      if(PREDICTABLE(pStateEnd != pState)) {
         size_t iDivision = 0;
         do {
            ++iDivision;
            ++pcBytesInSlice2; // we have one less cut than we have slices, so move to the next one

            const size_t * pcItemsInNextSliceLast = pState->m_pcItemsInNextSliceEnd - size_t { 1 };
            // TODO: check if this still works if we have zero cuts
            if(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(iDivision, pcItemsInNextSliceLast - pcBytesInSlice2)) {
               LOG_0(TraceLevelWarning, "WARNING CutRandomInternal pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(iDivision, pcItemsInNextSliceLast - pcBytesInSlice2)");
               free(pBuffer);
               return true;
            }
            if(pcItemsInNextSliceLast != pcBytesInSlice2) {
               ActiveDataType * pDivision = pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(iDivision);
               size_t iSplit = *pcItemsInNextSliceLast - size_t { 1 };
               *pDivision = iSplit;
               --pcItemsInNextSliceLast;
               while(pcItemsInNextSliceLast != pcBytesInSlice2) {
                  ++pDivision;
                  iSplit += *pcBytesInSlice2;
                  *pDivision = iSplit;
                  ++pcBytesInSlice2;
               }
               // increment it once more because our indexes are shifted such that the first one was the last item
               ++pcBytesInSlice2;
            }
            ++pState;
         } while(PREDICTABLE(pStateEnd != pState));
      }

      FloatEbmType * pUpdate = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
      HistogramBucket<bClassification> * pCollapsedHistogramBucket2 = aCollapsedHistogramBuckets;

      if(0 != (GenerateUpdateOptions_Sums & options)) {
         do {
            HistogramBucketVectorEntry<bClassification> * const pHistogramBucketVectorEntry =
               pCollapsedHistogramBucket2->GetHistogramBucketVectorEntry();

            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               *pUpdate = pHistogramBucketVectorEntry[iVector].m_sumResidualError;
               ++pUpdate;
            }
            pCollapsedHistogramBucket2 = GetHistogramBucketByIndex<bClassification>(
               cBytesPerHistogramBucket, pCollapsedHistogramBucket2, 1);
         } while(pCollapsedHistogramBucketEnd != pCollapsedHistogramBucket2);
      } else {
         do {
            HistogramBucketVectorEntry<bClassification> * const pHistogramBucketVectorEntry =
               pCollapsedHistogramBucket2->GetHistogramBucketVectorEntry();

            for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
               FloatEbmType update;
               if(bClassification) {
                  update = EbmStatistics::ComputeSmallChangeForOneSegmentClassificationLogOdds(
                     pHistogramBucketVectorEntry[iVector].m_sumResidualError,
                     pHistogramBucketVectorEntry[iVector].GetSumDenominator()
                  );
               } else {
                  EBM_ASSERT(IsRegression(compilerLearningTypeOrCountTargetClasses));
                  update = EbmStatistics::ComputeSmallChangeForOneSegmentRegression(
                     pHistogramBucketVectorEntry[iVector].m_sumResidualError,
                     static_cast<FloatEbmType>(pCollapsedHistogramBucket2->GetCountSamplesInBucket())
                  );
               }
               *pUpdate = update;
               ++pUpdate;
            }
            pCollapsedHistogramBucket2 = GetHistogramBucketByIndex<bClassification>(
               cBytesPerHistogramBucket, pCollapsedHistogramBucket2, 1);
         } while(pCollapsedHistogramBucketEnd != pCollapsedHistogramBucket2);
      }

      free(pBuffer);
      *pTotalGain = gain;
      return false;
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class CutRandomTarget final {
public:

   CutRandomTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static bool Func(
      EbmBoostingState * const pEbmBoostingState,
      const FeatureGroup * const pFeatureGroup,
      HistogramBucketBase * const aHistogramBuckets,
      const size_t cTreeSplitsMax,
      const GenerateUpdateOptionsType options,
      SegmentedTensor * const pSmallChangeToModelOverwriteSingleSamplingSet,
      FloatEbmType * const pTotalGain
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         return CutRandomInternal<compilerLearningTypeOrCountTargetClassesPossible>::Func(
            pEbmBoostingState,
            pFeatureGroup,
            aHistogramBuckets,
            cTreeSplitsMax,
            options,
            pSmallChangeToModelOverwriteSingleSamplingSet,
            pTotalGain
#ifndef NDEBUG
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      } else {
         return CutRandomTarget<compilerLearningTypeOrCountTargetClassesPossible + 1>::Func(
            pEbmBoostingState,
            pFeatureGroup,
            aHistogramBuckets,
            cTreeSplitsMax,
            options,
            pSmallChangeToModelOverwriteSingleSamplingSet,
            pTotalGain
#ifndef NDEBUG
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<>
class CutRandomTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   CutRandomTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static bool Func(
      EbmBoostingState * const pEbmBoostingState,
      const FeatureGroup * const pFeatureGroup,
      HistogramBucketBase * const aHistogramBuckets,
      const size_t cTreeSplitsMax,
      const GenerateUpdateOptionsType options,
      SegmentedTensor * const pSmallChangeToModelOverwriteSingleSamplingSet,
      FloatEbmType * const pTotalGain
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses());

      return CutRandomInternal<k_dynamicClassification>::Func(
         pEbmBoostingState,
         pFeatureGroup,
         aHistogramBuckets,
         cTreeSplitsMax,
         options,
         pSmallChangeToModelOverwriteSingleSamplingSet,
         pTotalGain
#ifndef NDEBUG
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
};

extern bool CutRandom(
   EbmBoostingState * const pEbmBoostingState,
   const FeatureGroup * const pFeatureGroup,
   HistogramBucketBase * const aHistogramBuckets,
   const size_t cTreeSplitsMax,
   const GenerateUpdateOptionsType options,
   SegmentedTensor * const pSmallChangeToModelOverwriteSingleSamplingSet,
   FloatEbmType * const pTotalGain
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses();

   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      return CutRandomTarget<2>::Func(
         pEbmBoostingState,
         pFeatureGroup,
         aHistogramBuckets,
         cTreeSplitsMax,
         options,
         pSmallChangeToModelOverwriteSingleSamplingSet,
         pTotalGain
#ifndef NDEBUG
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      return CutRandomInternal<k_regression>::Func(
         pEbmBoostingState,
         pFeatureGroup,
         aHistogramBuckets,
         cTreeSplitsMax,
         options,
         pSmallChangeToModelOverwriteSingleSamplingSet,
         pTotalGain
#ifndef NDEBUG
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
}
