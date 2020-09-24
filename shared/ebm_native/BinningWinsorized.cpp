// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // std::numeric_limits
#include <algorithm> // std::sort

#include "ebm_native.h"
#include "EbmInternal.h"
#include "Logging.h" // EBM_ASSERT & LOG

extern size_t RemoveMissingValuesAndReplaceInfinities(
   size_t cSamples,
   FloatEbmType * const aValues,
   FloatEbmType * const pMinNonInfinityValueOut,
   IntEbmType * const pCountNegativeInfinityOut,
   FloatEbmType * const pMaxNonInfinityValueOut,
   IntEbmType * const pCountPositiveInfinityOut
) noexcept;

// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterGenerateWinsorizedBinCutsParametersMessages = 25;
static int g_cLogExitGenerateWinsorizedBinCutsParametersMessages = 25;

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateWinsorizedBinCuts(
   IntEbmType countSamples,
   FloatEbmType * featureValues,
   IntEbmType * countBinCutsInOut,
   FloatEbmType * binCutsLowerBoundInclusiveOut,
   IntEbmType * countMissingValuesOut,
   FloatEbmType * minNonInfinityValueOut,
   IntEbmType * countNegativeInfinityOut,
   FloatEbmType * maxNonInfinityValueOut,
   IntEbmType * countPositiveInfinityOut
) {
   LOG_COUNTED_N(
      &g_cLogEnterGenerateWinsorizedBinCutsParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered GenerateWinsorizedBinCuts: "
      "countSamples=%" IntEbmTypePrintf ", "
      "featureValues=%p, "
      "countBinCutsInOut=%p, "
      "binCutsLowerBoundInclusiveOut=%p, "
      "countMissingValuesOut=%p, "
      "minNonInfinityValueOut=%p, "
      "countNegativeInfinityOut=%p, "
      "maxNonInfinityValueOut=%p, "
      "countPositiveInfinityOut=%p"
      ,
      countSamples,
      static_cast<void *>(featureValues),
      static_cast<void *>(countBinCutsInOut),
      static_cast<void *>(binCutsLowerBoundInclusiveOut),
      static_cast<void *>(countMissingValuesOut),
      static_cast<void *>(minNonInfinityValueOut),
      static_cast<void *>(countNegativeInfinityOut),
      static_cast<void *>(maxNonInfinityValueOut),
      static_cast<void *>(countPositiveInfinityOut)
   );

   IntEbmType countBinCutsRet;
   IntEbmType countMissingValuesRet;
   FloatEbmType minNonInfinityValueRet;
   IntEbmType countNegativeInfinityRet;
   FloatEbmType maxNonInfinityValueRet;
   IntEbmType countPositiveInfinityRet;
   IntEbmType ret;

   // if there is only 1 bin, then there can be no cut points, and no point doing any more work here
   if(UNLIKELY(nullptr == countBinCutsInOut)) {
      LOG_0(TraceLevelError, "ERROR GenerateWinsorizedBinCuts nullptr == countBinCutsInOut");
      countBinCutsRet = IntEbmType { 0 };
      countMissingValuesRet = IntEbmType { 0 };
      minNonInfinityValueRet = FloatEbmType { 0 };
      countNegativeInfinityRet = IntEbmType { 0 };
      maxNonInfinityValueRet = FloatEbmType { 0 };
      countPositiveInfinityRet = IntEbmType { 0 };
      ret = IntEbmType { 1 };
   } else {
      if(UNLIKELY(countSamples <= IntEbmType { 0 })) {
         countBinCutsRet = IntEbmType { 0 };
         countMissingValuesRet = IntEbmType { 0 };
         minNonInfinityValueRet = FloatEbmType { 0 };
         countNegativeInfinityRet = IntEbmType { 0 };
         maxNonInfinityValueRet = FloatEbmType { 0 };
         countPositiveInfinityRet = IntEbmType { 0 };
         ret = IntEbmType { 0 };
         if(UNLIKELY(countSamples < IntEbmType { 0 })) {
            LOG_0(TraceLevelError, "ERROR GenerateWinsorizedBinCuts countSamples < IntEbmType { 0 }");
            ret = IntEbmType { 1 };
         }
      } else {
         if(UNLIKELY(nullptr == featureValues)) {
            LOG_0(TraceLevelError, "ERROR GenerateWinsorizedBinCuts nullptr == featureValues");

            countBinCutsRet = IntEbmType { 0 };
            countMissingValuesRet = IntEbmType { 0 };
            minNonInfinityValueRet = FloatEbmType { 0 };
            countNegativeInfinityRet = IntEbmType { 0 };
            maxNonInfinityValueRet = FloatEbmType { 0 };
            countPositiveInfinityRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }

         if(UNLIKELY(!IsNumberConvertable<size_t>(countSamples))) {
            LOG_0(TraceLevelWarning, "WARNING GenerateWinsorizedBinCuts !IsNumberConvertable<size_t>(countSamples)");

            countBinCutsRet = IntEbmType { 0 };
            countMissingValuesRet = IntEbmType { 0 };
            minNonInfinityValueRet = FloatEbmType { 0 };
            countNegativeInfinityRet = IntEbmType { 0 };
            maxNonInfinityValueRet = FloatEbmType { 0 };
            countPositiveInfinityRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }

         const size_t cSamplesIncludingMissingValues = static_cast<size_t>(countSamples);

         if(UNLIKELY(IsMultiplyError(sizeof(*featureValues), cSamplesIncludingMissingValues))) {
            LOG_0(TraceLevelError, "ERROR GenerateWinsorizedBinCuts countSamples was too large to fit into featureValues");

            countBinCutsRet = IntEbmType { 0 };
            countMissingValuesRet = IntEbmType { 0 };
            minNonInfinityValueRet = FloatEbmType { 0 };
            countNegativeInfinityRet = IntEbmType { 0 };
            maxNonInfinityValueRet = FloatEbmType { 0 };
            countPositiveInfinityRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }

         // if there are +infinity values in the data we won't be able to separate them
         // from max_float values without having a cut at infinity since we use lower bound inclusivity
         // so we disallow +infinity values by turning them into max_float.  For symmetry we do the same on
         // the -infinity side turning those into lowest_float.  
         const size_t cSamples = RemoveMissingValuesAndReplaceInfinities(
            cSamplesIncludingMissingValues,
            featureValues,
            &minNonInfinityValueRet,
            &countNegativeInfinityRet,
            &maxNonInfinityValueRet,
            &countPositiveInfinityRet
         );

         EBM_ASSERT(cSamples <= cSamplesIncludingMissingValues);
         const size_t cMissingValues = cSamplesIncludingMissingValues - cSamples;
         // this is guaranteed to work since the number of missing values can't exceed the number of original
         // samples, and samples came to us as an IntEbmType
         EBM_ASSERT(IsNumberConvertable<IntEbmType>(cMissingValues));
         countMissingValuesRet = static_cast<IntEbmType>(cMissingValues);

         if(UNLIKELY(size_t { 0 } == cSamples)) {
            countBinCutsRet = IntEbmType { 0 };
            EBM_ASSERT(FloatEbmType { 0 } == minNonInfinityValueRet);
            EBM_ASSERT(IntEbmType { 0 } == countNegativeInfinityRet);
            EBM_ASSERT(FloatEbmType { 0 } == maxNonInfinityValueRet);
            EBM_ASSERT(IntEbmType { 0 } == countPositiveInfinityRet);
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }

         EBM_ASSERT(nullptr != countBinCutsInOut);
         const IntEbmType countBinCuts = *countBinCutsInOut;

         if(UNLIKELY(countBinCuts <= IntEbmType { 0 })) {
            countBinCutsRet = IntEbmType { 0 };
            ret = IntEbmType { 0 };
            if(UNLIKELY(countBinCuts < IntEbmType { 0 })) {
               LOG_0(TraceLevelError, "ERROR GenerateWinsorizedBinCuts countBinCuts can't be negative.");
               ret = IntEbmType { 1 };
            }
            goto exit_with_log;
         }

         if(UNLIKELY(!IsNumberConvertable<size_t>(countBinCuts))) {
            LOG_0(TraceLevelWarning, "WARNING GenerateWinsorizedBinCuts !IsNumberConvertable<size_t>(countBinCuts)");

            countBinCutsRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }

         if(UNLIKELY(nullptr == binCutsLowerBoundInclusiveOut)) {
            // if we have a potential bin cut, then binCutsLowerBoundInclusiveOut shouldn't be nullptr
            LOG_0(TraceLevelError, "ERROR GenerateWinsorizedBinCuts nullptr == binCutsLowerBoundInclusiveOut");

            countBinCutsRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };

            goto exit_with_log;
         }

         // our +infinity values have been turned into max_float and -infinity values have been turned into 
         // lowest_float because we can't have a cut between max_float and +infinity without using a +infinity
         // cut value since we use lower bound inclusivity.  Other than +-infinity, if our dataset isn't completely
         // uniform we just need to find a single cut between values and we can divide the space up between
         // uniform bins between those values.

         std::sort(featureValues, featureValues + cSamples);

         const size_t cBinCuts = static_cast<size_t>(countBinCuts);
         const size_t cBins = cBinCuts + size_t { 1 };
         const size_t cStepInwards = (cSamples - size_t { 1 }) / cBins;

         // position ourselves at low-high and move inwards
         const FloatEbmType * pLow = &featureValues[cStepInwards];
         // position ourselves at high-low and move inwards
         const FloatEbmType * pHigh = &featureValues[cSamples - cStepInwards - size_t { 1 }];

         FloatEbmType lowVal = *pLow;
         FloatEbmType highVal = *pHigh;
         FloatEbmType valCur;

         if(lowVal == highVal) {
            // we need to go outwards since there are no cuts inwards.  We just need a single cut in order
            // to make this work since we can subdivide the space between the cuts

            const FloatEbmType minValue = UNPREDICTABLE(0 == countNegativeInfinityRet) ? 
               minNonInfinityValueRet : std::numeric_limits<FloatEbmType>::lowest();
            const FloatEbmType maxValue = UNPREDICTABLE(0 == countPositiveInfinityRet) ?
               maxNonInfinityValueRet : std::numeric_limits<FloatEbmType>::max();

            if(minValue != lowVal) {
               // there's a transition somehwere on the low side, so let's find it
               ++pLow;
               do {
                  --pLow;
                  valCur = *pLow;
               } while(lowVal != valCur);
               lowVal = valCur;
            }
            if(maxValue != highVal) {
               // there's a transition somehwere on the high side, so let's find it
               --pHigh;
               do {
                  ++pHigh;
                  valCur = *pHigh;
               } while(highVal != valCur);
               highVal = valCur;
            }
            if(lowVal == highVal) {
               // there are no cuts at all
               countBinCutsRet = IntEbmType { 0 };
               ret = IntEbmType { 0 };
               goto exit_with_log;
            }

            // move upwards from the min since we 
            lowVal = std::nextafter(lowVal, std::numeric_limits<FloatEbmType>::max());
         } else {
            // because lowVal != highVal, we know there's a transition that'll exit our loops somewhere in between

            do {
               ++pLow;
               valCur = *pLow;
            } while(lowVal != valCur);
            lowVal = valCur;

            do {
               --pHigh;
               valCur = *pHigh;
            } while(highVal != valCur);
            highVal = valCur;

            if(highVal < lowVal) {
               // there was only a single transition between our points, and now our low is on the high side
               // and our high is on the low side, so flip them

               highVal = lowVal;
               lowVal = valCur;
            }

            // move one up from the highVal since if we put a cut exactly there the high-low value will
            // be included in the highest bin, and we don't want that
            // winsorized binning doesn't have the ability to create humanized cut numbers, so going one up
            // isn't a big deal here, and even if the high value is an integer like 5, one up from that will
            // basically round to 5 in the UI but we'll get a number here that's just slighly higher
            //
            // This won't work if max_float == highVal, but it'll be fine since in that case nextafter will
            // just keep the max_float and we'll have a cut with max_float on the upper bin
            highVal = std::nextafter(highVal, std::numeric_limits<FloatEbmType>::max());
         }

         FloatEbmType * pBinCutsLowerBoundInclusive = binCutsLowerBoundInclusiveOut;
         const FloatEbmType stepValue = (highVal - lowVal) / static_cast<FloatEbmType>(cBins);
         FloatEbmType cutPrev = lowVal;
         size_t iCut = size_t { 1 };
         *pBinCutsLowerBoundInclusive = lowVal;
         ++pBinCutsLowerBoundInclusive;
         do {
            const FloatEbmType cut = lowVal + stepValue * iCut;
            if(cut != cutPrev) {
               // just in case we have floating point inexactness that puts us above the highValue we need to stop
               if(highVal <= cut) {
                  break;
               }
               *pBinCutsLowerBoundInclusive = cut;
               ++pBinCutsLowerBoundInclusive;
               cutPrev = cut;
            }
            ++iCut;
         } while(cBinCuts - 1 != iCut);

         // write the last one manually without resorting to a formula
         // we don't allow our loop above to write out the highVal, so we're guarnateed that we can write it here
         *pBinCutsLowerBoundInclusive = highVal;

         EBM_ASSERT(binCutsLowerBoundInclusiveOut <= pBinCutsLowerBoundInclusive);
         const size_t cBinCutsRet = pBinCutsLowerBoundInclusive - binCutsLowerBoundInclusiveOut + size_t { 1 };

         // this conversion is guaranteed to work since the number of cut points can't exceed the number our user
         // specified, and that value came to us as an IntEbmType
         countBinCutsRet = static_cast<IntEbmType>(cBinCutsRet);
         EBM_ASSERT(countBinCutsRet <= countBinCuts);

         ret = IntEbmType { 0 };
      }

   exit_with_log:;

      EBM_ASSERT(nullptr != countBinCutsInOut);
      *countBinCutsInOut = countBinCutsRet;
   }

   if(LIKELY(nullptr != countMissingValuesOut)) {
      *countMissingValuesOut = countMissingValuesRet;
   }
   if(LIKELY(nullptr != minNonInfinityValueOut)) {
      *minNonInfinityValueOut = minNonInfinityValueRet;
   }
   if(LIKELY(nullptr != countNegativeInfinityOut)) {
      *countNegativeInfinityOut = countNegativeInfinityRet;
   }
   if(LIKELY(nullptr != maxNonInfinityValueOut)) {
      *maxNonInfinityValueOut = maxNonInfinityValueRet;
   }
   if(LIKELY(nullptr != countPositiveInfinityOut)) {
      *countPositiveInfinityOut = countPositiveInfinityRet;
   }

   LOG_COUNTED_N(
      &g_cLogExitGenerateWinsorizedBinCutsParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Exited GenerateWinsorizedBinCuts: "
      "countBinCuts=%" IntEbmTypePrintf ", "
      "countMissingValues=%" IntEbmTypePrintf ", "
      "minNonInfinityValue=%" FloatEbmTypePrintf ", "
      "countNegativeInfinity=%" IntEbmTypePrintf ", "
      "maxNonInfinityValue=%" FloatEbmTypePrintf ", "
      "countPositiveInfinity=%" IntEbmTypePrintf ", "
      "return=%" IntEbmTypePrintf
      ,
      countBinCutsRet,
      countMissingValuesRet,
      minNonInfinityValueRet,
      countNegativeInfinityRet,
      maxNonInfinityValueRet,
      countPositiveInfinityRet,
      ret
   );

   return ret;
}
