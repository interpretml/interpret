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

extern FloatEbmType ArithmeticMean(
   const FloatEbmType low,
   const FloatEbmType high
) noexcept;

// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterGenerateWinsorizedBinCutsParametersMessages = 25;
static int g_cLogExitGenerateWinsorizedBinCutsParametersMessages = 25;

// TODO: add this as a python/R option "winsorized"
EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateWinsorizedBinCuts(
   IntEbmType countSamples,
   const FloatEbmType * featureValues,
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
      static_cast<const void *>(featureValues),
      static_cast<void *>(countBinCutsInOut),
      static_cast<void *>(binCutsLowerBoundInclusiveOut),
      static_cast<void *>(countMissingValuesOut),
      static_cast<void *>(minNonInfinityValueOut),
      static_cast<void *>(countNegativeInfinityOut),
      static_cast<void *>(maxNonInfinityValueOut),
      static_cast<void *>(countPositiveInfinityOut)
   );

   IntEbmType countBinCutsRet = IntEbmType { 0 };
   IntEbmType countMissingValuesRet;
   FloatEbmType minNonInfinityValueRet;
   IntEbmType countNegativeInfinityRet;
   FloatEbmType maxNonInfinityValueRet;
   IntEbmType countPositiveInfinityRet;
   IntEbmType ret;

   if(UNLIKELY(nullptr == countBinCutsInOut)) {
      LOG_0(TraceLevelError, "ERROR GenerateWinsorizedBinCuts nullptr == countBinCutsInOut");
      countMissingValuesRet = IntEbmType { 0 };
      minNonInfinityValueRet = FloatEbmType { 0 };
      countNegativeInfinityRet = IntEbmType { 0 };
      maxNonInfinityValueRet = FloatEbmType { 0 };
      countPositiveInfinityRet = IntEbmType { 0 };
      ret = IntEbmType { 1 };
   } else {
      if(UNLIKELY(countSamples <= IntEbmType { 0 })) {
         // if there's 1 sample, then we can't split it, but we'd still want to determine the min, max, etc
         // so continue processing

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
         if(UNLIKELY(!IsNumberConvertable<size_t>(countSamples))) {
            LOG_0(TraceLevelWarning, "WARNING GenerateWinsorizedBinCuts !IsNumberConvertable<size_t>(countSamples)");

            countMissingValuesRet = IntEbmType { 0 };
            minNonInfinityValueRet = FloatEbmType { 0 };
            countNegativeInfinityRet = IntEbmType { 0 };
            maxNonInfinityValueRet = FloatEbmType { 0 };
            countPositiveInfinityRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }

         if(UNLIKELY(nullptr == featureValues)) {
            LOG_0(TraceLevelError, "ERROR GenerateWinsorizedBinCuts nullptr == featureValues");

            countMissingValuesRet = IntEbmType { 0 };
            minNonInfinityValueRet = FloatEbmType { 0 };
            countNegativeInfinityRet = IntEbmType { 0 };
            maxNonInfinityValueRet = FloatEbmType { 0 };
            countPositiveInfinityRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }

         const size_t cSamplesIncludingMissingValues = static_cast<size_t>(countSamples);

         FloatEbmType * const aFeatureValues = EbmMalloc<FloatEbmType>(cSamplesIncludingMissingValues);
         if(UNLIKELY(nullptr == aFeatureValues)) {
            LOG_0(TraceLevelError, "ERROR GenerateWinsorizedBinCuts nullptr == aFeatureValues");

            countMissingValuesRet = IntEbmType { 0 };
            minNonInfinityValueRet = FloatEbmType { 0 };
            countNegativeInfinityRet = IntEbmType { 0 };
            maxNonInfinityValueRet = FloatEbmType { 0 };
            countPositiveInfinityRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }
         const size_t cBytesFeatureValues = sizeof(*featureValues) * cSamplesIncludingMissingValues;
         memcpy(aFeatureValues, featureValues, cBytesFeatureValues);

         // if there are +infinity values in the data we won't be able to separate them
         // from max_float values without having a cut at infinity since we use lower bound inclusivity
         // so we disallow +infinity values by turning them into max_float.  For symmetry we do the same on
         // the -infinity side turning those into lowest_float.  
         const size_t cSamples = RemoveMissingValuesAndReplaceInfinities(
            cSamplesIncludingMissingValues,
            aFeatureValues,
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

         // we can't really split 0 or 1 samples.  Now that we know our min, max, etc values, we can exit
         // or if there was only 1 non-missing value
         if(LIKELY(size_t { 1 } < cSamples)) {
            EBM_ASSERT(nullptr != countBinCutsInOut);
            const IntEbmType countBinCuts = *countBinCutsInOut;

            if(UNLIKELY(countBinCuts <= IntEbmType { 0 })) {
               free(aFeatureValues);
               ret = IntEbmType { 0 };
               if(UNLIKELY(countBinCuts < IntEbmType { 0 })) {
                  LOG_0(TraceLevelError, "ERROR GenerateWinsorizedBinCuts countBinCuts can't be negative.");
                  ret = IntEbmType { 1 };
               }
               goto exit_with_log;
            }

            if(UNLIKELY(!IsNumberConvertable<size_t>(countBinCuts))) {
               LOG_0(TraceLevelWarning, "WARNING GenerateWinsorizedBinCuts !IsNumberConvertable<size_t>(countBinCuts)");
               free(aFeatureValues);
               ret = IntEbmType { 1 };
               goto exit_with_log;
            }
            const size_t cBinCuts = static_cast<size_t>(countBinCuts);

            if(UNLIKELY(IsMultiplyError(sizeof(*binCutsLowerBoundInclusiveOut), cBinCuts))) {
               LOG_0(TraceLevelError, "ERROR GenerateWinsorizedBinCuts countBinCuts was too large to fit into binCutsLowerBoundInclusiveOut");
               free(aFeatureValues);
               ret = IntEbmType { 1 };
               goto exit_with_log;
            }

            if(UNLIKELY(nullptr == binCutsLowerBoundInclusiveOut)) {
               // if we have a potential bin cut, then binCutsLowerBoundInclusiveOut shouldn't be nullptr
               LOG_0(TraceLevelError, "ERROR GenerateWinsorizedBinCuts nullptr == binCutsLowerBoundInclusiveOut");
               free(aFeatureValues);
               ret = IntEbmType { 1 };
               goto exit_with_log;
            }

            // our +infinity values have been turned into max_float and -infinity values have been turned into 
            // lowest_float because we can't have a cut between max_float and +infinity without using a +infinity
            // cut value since we use lower bound inclusivity.  Other than +-infinity, if our dataset isn't completely
            // uniform we just need to find a single cut between values and we can divide the space up between
            // uniform bins between those values.

            std::sort(aFeatureValues, aFeatureValues + cSamples);

            if(UNLIKELY(size_t { 1 } == cBinCuts)) {
               // if we're only given 1 cut, then we need do so something special since we can't have an upper and
               // lower cut from which to range between.  We want to find the best central cut and use that

               const FloatEbmType minValue = aFeatureValues[0];
               const FloatEbmType maxValue = aFeatureValues[cSamples - size_t { 1 }];

#ifndef NDEBUG
               // if all of the samples are positive infinity then minValue is max, otherwise if there are any
               // negative infinities, then the min will be lowest.  Same for the max, but in reverse.
               const FloatEbmType minValueCompare = UNLIKELY(cSamples == static_cast<size_t>(countPositiveInfinityRet)) ?
                  std::numeric_limits<FloatEbmType>::max() :
                  (UNPREDICTABLE(0 == countNegativeInfinityRet) ?
                     minNonInfinityValueRet : std::numeric_limits<FloatEbmType>::lowest());
               const FloatEbmType maxValueCompare = UNLIKELY(cSamples == static_cast<size_t>(countNegativeInfinityRet)) ?
                  std::numeric_limits<FloatEbmType>::lowest() :
                  (UNPREDICTABLE(0 == countPositiveInfinityRet) ?
                     maxNonInfinityValueRet : std::numeric_limits<FloatEbmType>::max());

               EBM_ASSERT(minValue == minValueCompare);
               EBM_ASSERT(maxValue == maxValueCompare);
#endif

               // if this fails there are no transitions at all, so we can't have a cut
               if(LIKELY(minValue != maxValue)) {
                  const size_t iCenterHigh = cSamples >> 1;
                  // put the low on the high side so that in our first loop we decrement it to it's actual position
                  const FloatEbmType * pLow = &aFeatureValues[iCenterHigh];
                  const FloatEbmType * pHigh = pLow - (size_t { 1 } & (cSamples - size_t { 1 }));

                  FloatEbmType lowCur;
                  FloatEbmType highCur;
                  do {
                     --pLow;
                     ++pHigh;

                     EBM_ASSERT(aFeatureValues <= pLow && pLow < aFeatureValues + cSamples);
                     lowCur = *pLow;
                     EBM_ASSERT(aFeatureValues <= pHigh && pHigh < aFeatureValues + cSamples);
                     highCur = *pHigh;
                     // since minValue != maxValue per above, we know there is a transition SOMEWHERE, which will exit us
                  } while(LIKELY(lowCur == highCur));
                  EBM_ASSERT(lowCur < highCur);

                  // if both lowCur and highCur have changed, we'll get the average value between them, but that'll
                  // put the centerVal either on the low or high bin dependent on the values.  Unlike quantile binning, 
                  // winsorized binning is senitive to the values and not invariant to operations, so this is fine

                  const FloatEbmType avg = ArithmeticMean(lowCur, highCur);
                  *binCutsLowerBoundInclusiveOut = avg;
                  countBinCutsRet = IntEbmType { 1 };
               }
            } else {
               // we check that cBinCuts can be multiplied with sizeof(*binCutsLowerBoundInclusiveOut), and since
               // there is no way an element of binCutsLowerBoundInclusiveOut is as small as 1 byte, we should
               // be able to add one to cBinCuts
               EBM_ASSERT(!IsAddError(cBinCuts, size_t { 1 }));
               const size_t cBins = cBinCuts + size_t { 1 };
               EBM_ASSERT(size_t { 1 } < cSamples);
               const size_t iOuterBound = (cSamples - size_t { 1 }) / cBins;
               EBM_ASSERT(iOuterBound < cSamples);

               // position ourselves at low-high and move inwards
               const FloatEbmType * pLow = &aFeatureValues[iOuterBound];
               // position ourselves at high-low and move inwards
               const FloatEbmType * pHigh = &aFeatureValues[cSamples - iOuterBound - size_t { 1 }];

               EBM_ASSERT(aFeatureValues <= pLow && pLow < aFeatureValues + cSamples);
               EBM_ASSERT(aFeatureValues <= pHigh && pHigh < aFeatureValues + cSamples);

               const FloatEbmType lowOuterVal = *pLow;
               const FloatEbmType highOuterVal = *pHigh;
               EBM_ASSERT(lowOuterVal <= highOuterVal);

               if(UNLIKELY(lowOuterVal == highOuterVal)) {
                  // there are no transitions between our outer values.  We have just 1 single value between them
                  // one way to handle this would be to wrap the value on the low side with the exact value
                  // and epsilon higher on the high side, but that's rather tight and makes the graph look too
                  // tight.  We instead put the two cuts between the outer values and the next transition outwards

                  const FloatEbmType centerVal = lowOuterVal;

                  const FloatEbmType minValue = aFeatureValues[0];
                  const FloatEbmType maxValue = aFeatureValues[cSamples - size_t { 1 }];

#ifndef NDEBUG
                  // if all of the samples are positive infinity then minValue is max, otherwise if there are any
                  // negative infinities, then the min will be lowest.  Same for the max, but in reverse.
                  const FloatEbmType minValueCompare = UNLIKELY(cSamples == static_cast<size_t>(countPositiveInfinityRet)) ?
                     std::numeric_limits<FloatEbmType>::max() :
                     (UNPREDICTABLE(0 == countNegativeInfinityRet) ?
                     minNonInfinityValueRet : std::numeric_limits<FloatEbmType>::lowest());
                  const FloatEbmType maxValueCompare = UNLIKELY(cSamples == static_cast<size_t>(countNegativeInfinityRet)) ?
                     std::numeric_limits<FloatEbmType>::lowest() :
                     (UNPREDICTABLE(0 == countPositiveInfinityRet) ?
                     maxNonInfinityValueRet : std::numeric_limits<FloatEbmType>::max());

                  EBM_ASSERT(minValue == minValueCompare);
                  EBM_ASSERT(maxValue == maxValueCompare);
#endif // NDEBUG

                  FloatEbmType * pBinCutsLowerBoundInclusive = binCutsLowerBoundInclusiveOut;
                  if(PREDICTABLE(minValue != centerVal)) {
                     // there's a transition somewhere on the low side
                     EBM_ASSERT(std::numeric_limits<FloatEbmType>::lowest() < centerVal);
                     EBM_ASSERT(minValue < centerVal);

                     FloatEbmType valCur;
                     do {
                        --pLow;
                        EBM_ASSERT(aFeatureValues <= pLow && pLow < aFeatureValues + cSamples);
                        valCur = *pLow;
                     } while(centerVal == valCur);
                     EBM_ASSERT(valCur < centerVal);

                     const FloatEbmType avg = ArithmeticMean(valCur, centerVal);
                     *pBinCutsLowerBoundInclusive = avg;
                     ++pBinCutsLowerBoundInclusive;
                     ++countBinCutsRet;
                  }
                  if(PREDICTABLE(maxValue != centerVal)) {
                     // there's a transition somewhere on the high side
                     EBM_ASSERT(centerVal < std::numeric_limits<FloatEbmType>::max());
                     EBM_ASSERT(centerVal < maxValue);

                     FloatEbmType valCur;
                     do {
                        ++pHigh;
                        EBM_ASSERT(aFeatureValues <= pHigh && pHigh < aFeatureValues + cSamples);
                        valCur = *pHigh;
                     } while(centerVal == valCur);
                     EBM_ASSERT(centerVal < valCur);

                     const FloatEbmType avg = ArithmeticMean(centerVal, valCur);
                     *pBinCutsLowerBoundInclusive = avg;
                     ++countBinCutsRet;
                  }
               } else {
                  // because lowVal != highVal, we know there's a transition that'll exit our loops somewhere in between

                  FloatEbmType lowInnerVal;
                  do {
                     ++pLow;
                     EBM_ASSERT(aFeatureValues <= pLow && pLow < aFeatureValues + cSamples);
                     lowInnerVal = *pLow;
                  } while(lowOuterVal == lowInnerVal);
                  EBM_ASSERT(std::numeric_limits<FloatEbmType>::lowest() < lowInnerVal);
                  EBM_ASSERT(lowOuterVal < lowInnerVal);
                  EBM_ASSERT(lowInnerVal <= highOuterVal);

                  if(lowInnerVal == highOuterVal) {
                     // there was just a single transition between our two outer values.  This doesn't really give
                     // us anything to wrap, so let's just return a single cut point in the middle of the transition
                     // space

                     EBM_ASSERT(lowOuterVal < highOuterVal);
                     const FloatEbmType avg = ArithmeticMean(lowOuterVal, highOuterVal);
                     *binCutsLowerBoundInclusiveOut = avg;
                     countBinCutsRet = IntEbmType { 1 };
                  } else {
                     FloatEbmType highInnerVal;
                     do {
                        --pHigh;
                        EBM_ASSERT(aFeatureValues <= pHigh && pHigh < aFeatureValues + cSamples);
                        highInnerVal = *pHigh;
                     } while(highOuterVal == highInnerVal);
                     EBM_ASSERT(highInnerVal < std::numeric_limits<FloatEbmType>::max());
                     EBM_ASSERT(highInnerVal < highOuterVal);
                     EBM_ASSERT(lowInnerVal <= highInnerVal);
                     EBM_ASSERT(lowOuterVal < highInnerVal);

                     if(lowInnerVal == highInnerVal) {
                        // there's just one long run of values in the center.  We don't really want to wrap any
                        // number this tightly, so let's put down two cut points in the spaces on either side

                        const FloatEbmType centerVal = lowInnerVal;

                        EBM_ASSERT(lowOuterVal < centerVal);
                        const FloatEbmType avg1 = ArithmeticMean(lowOuterVal, centerVal);
                        binCutsLowerBoundInclusiveOut[0] = avg1;

                        EBM_ASSERT(centerVal < highOuterVal);
                        const FloatEbmType avg2 = ArithmeticMean(centerVal, highOuterVal);
                        binCutsLowerBoundInclusiveOut[1] = avg2;

                        countBinCutsRet = IntEbmType { 2 };
                     } else {
                        // move one up from the highVal since if we put a cut exactly there the high-low value will
                        // be included in the highest bin, and we don't want that
                        // winsorized binning doesn't have the ability to create humanized cut numbers, so going one tick up
                        // isn't a big deal here, and even if the high value is an integer like 5, one up from that will
                        // basically round to 5 in the UI but we'll get a number here that's just slighly higher

                        EBM_ASSERT(highInnerVal < std::numeric_limits<FloatEbmType>::max());
                        highInnerVal = std::nextafter(highInnerVal, std::numeric_limits<FloatEbmType>::max());

                        EBM_ASSERT(lowInnerVal < highInnerVal);
                        EBM_ASSERT(size_t { 2 } <= cBinCuts); // we can put down the low and high ones at least

                        FloatEbmType * pBinCutsLowerBoundInclusive = binCutsLowerBoundInclusiveOut;
                        *pBinCutsLowerBoundInclusive = lowInnerVal;
                        ++pBinCutsLowerBoundInclusive;

                        if(size_t { 2 } < cBinCuts) {
                           while(true) {
                              const size_t cInternalRanges = cBinCuts - size_t { 1 };
                              const FloatEbmType cInternalRangesFloat = static_cast<FloatEbmType>(cInternalRanges);
                              FloatEbmType stepValue = (highInnerVal - lowInnerVal) / cInternalRangesFloat;
                              if(std::isinf(stepValue)) {
                                 // cInternalRangesFloat should be 3 or higher, which should be enough to avoid any numeracy issues
                                 // that might cause us to overflow again
                                 stepValue = highInnerVal / cInternalRangesFloat - lowInnerVal / cInternalRangesFloat;
                                 if(std::isinf(stepValue)) {
                                    // this is probably impossible if correct rounding is guarnateed, but floats have bad guarantees

                                    // if you have 2 internal bins it would be close to an overflow on the subtraction 
                                    // of the divided values.  With 3 bins it isn't obvious to me how you'd get an
                                    // overflow after dividing it up in to separate divisions.  So, let's assume
                                    // that 2 == cBins, so we can just take the average and report one cut
                                    const FloatEbmType cut = ArithmeticMean(lowInnerVal, highInnerVal);
                                    // we always write out a cut at highInnerVal below, and we wouldn't want to do that
                                    // twice if we got back highInnerVal here, but we can only get here with huge
                                    // separations, so it shouldn't be possible to get something even close to 
                                    // highInnerVal
                                    EBM_ASSERT(highInnerVal != cut);
                                    *pBinCutsLowerBoundInclusive = cut;
                                    ++pBinCutsLowerBoundInclusive;
                                    break;
                                 }
                              }
                              FloatEbmType cutPrev = lowInnerVal;
                              size_t iCut = size_t { 1 };
                              do {
                                 const FloatEbmType cut = lowInnerVal + stepValue * static_cast<FloatEbmType>(iCut);
                                 // just in case we have floating point inexactness that puts us above the 
                                 // highValue we need to stop
                                 if(UNLIKELY(highInnerVal <= cut)) {
                                    break;
                                 }
                                 if(LIKELY(cutPrev != cut)) {
                                    EBM_ASSERT(cutPrev < cut);
                                    EBM_ASSERT(binCutsLowerBoundInclusiveOut < pBinCutsLowerBoundInclusive &&
                                       pBinCutsLowerBoundInclusive < binCutsLowerBoundInclusiveOut + cBinCuts - size_t { 1 });

                                    *pBinCutsLowerBoundInclusive = cut;
                                    ++pBinCutsLowerBoundInclusive;
                                    cutPrev = cut;
                                 }
                                 ++iCut;
                              } while(cInternalRanges != iCut);
                              break;
                           }
                        }

                        EBM_ASSERT(binCutsLowerBoundInclusiveOut < pBinCutsLowerBoundInclusive &&
                           pBinCutsLowerBoundInclusive < binCutsLowerBoundInclusiveOut + cBinCuts);

                        // write the last one manually without resorting to a formula.  We don't allow our loop 
                        // above to write out the highVal, so we're guarnateed that we can write it here
                        *pBinCutsLowerBoundInclusive = highInnerVal;

                        const size_t cBinCutsRet = 
                           pBinCutsLowerBoundInclusive - binCutsLowerBoundInclusiveOut + size_t { 1 };

                        // this conversion is guaranteed to work since the number of cut points can't exceed the number our user
                        // specified, and that value came to us as an IntEbmType
                        countBinCutsRet = static_cast<IntEbmType>(cBinCutsRet);
                        EBM_ASSERT(countBinCutsRet <= countBinCuts);
                     }
                  }
               }
            }
         }
         free(aFeatureValues);
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
