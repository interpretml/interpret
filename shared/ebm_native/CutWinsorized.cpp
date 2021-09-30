// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // std::numeric_limits
#include <algorithm> // std::sort

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern size_t RemoveMissingValuesAndReplaceInfinities(const size_t cSamples, FloatEbmType * const aValues) noexcept;

extern FloatEbmType ArithmeticMean(
   const FloatEbmType low,
   const FloatEbmType high
) noexcept;

// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterCutWinsorizedParametersMessages = 25;
static int g_cLogExitCutWinsorizedParametersMessages = 25;

// TODO: add this as a python/R option "winsorized"
EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CutWinsorized(
   IntEbmType countSamples,
   const FloatEbmType * featureValues,
   IntEbmType * countCutsInOut,
   FloatEbmType * cutsLowerBoundInclusiveOut
) {
   LOG_COUNTED_N(
      &g_cLogEnterCutWinsorizedParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered CutWinsorized: "
      "countSamples=%" IntEbmTypePrintf ", "
      "featureValues=%p, "
      "countCutsInOut=%p, "
      "cutsLowerBoundInclusiveOut=%p"
      ,
      countSamples,
      static_cast<const void *>(featureValues),
      static_cast<void *>(countCutsInOut),
      static_cast<void *>(cutsLowerBoundInclusiveOut)
   );

   IntEbmType countCutsRet = IntEbmType { 0 };
   ErrorEbmType ret;

   if(UNLIKELY(nullptr == countCutsInOut)) {
      LOG_0(TraceLevelError, "ERROR CutWinsorized nullptr == countCutsInOut");
      ret = Error_IllegalParamValue;
   } else {
      if(UNLIKELY(countSamples <= IntEbmType { 1 })) {
         // can't cut 1 sample
         ret = Error_None;
         if(UNLIKELY(countSamples < IntEbmType { 0 })) {
            LOG_0(TraceLevelError, "ERROR CutWinsorized countSamples < IntEbmType { 0 }");
            ret = Error_IllegalParamValue;
         }
      } else {
         if(UNLIKELY(IsConvertError<size_t>(countSamples))) {
            LOG_0(TraceLevelWarning, "WARNING CutWinsorized IsConvertError<size_t>(countSamples)");

            ret = Error_IllegalParamValue;
            goto exit_with_log;
         }

         if(UNLIKELY(nullptr == featureValues)) {
            LOG_0(TraceLevelError, "ERROR CutWinsorized nullptr == featureValues");

            ret = Error_IllegalParamValue;
            goto exit_with_log;
         }

         const size_t cSamplesIncludingMissingValues = static_cast<size_t>(countSamples);

         FloatEbmType * const aFeatureValues = EbmMalloc<FloatEbmType>(cSamplesIncludingMissingValues);
         if(UNLIKELY(nullptr == aFeatureValues)) {
            LOG_0(TraceLevelError, "ERROR CutWinsorized nullptr == aFeatureValues");

            ret = Error_OutOfMemory;
            goto exit_with_log;
         }
         const size_t cBytesFeatureValues = sizeof(*featureValues) * cSamplesIncludingMissingValues;
         memcpy(aFeatureValues, featureValues, cBytesFeatureValues);

         // if there are +infinity values in the data we won't be able to separate them
         // from max_float values without having a cut at infinity since we use lower bound inclusivity
         // so we disallow +infinity values by turning them into max_float.  For symmetry we do the same on
         // the -infinity side turning those into lowest_float.  
         const size_t cSamples = RemoveMissingValuesAndReplaceInfinities(cSamplesIncludingMissingValues, aFeatureValues);

         EBM_ASSERT(cSamples <= cSamplesIncludingMissingValues);

         // we can't really cut 0 or 1 samples.  Now that we know our min, max, etc values, we can exit
         // or if there was only 1 non-missing value
         if(LIKELY(size_t { 1 } < cSamples)) {
            EBM_ASSERT(nullptr != countCutsInOut);
            const IntEbmType countCuts = *countCutsInOut;

            if(UNLIKELY(countCuts <= IntEbmType { 0 })) {
               free(aFeatureValues);
               ret = Error_None;
               if(UNLIKELY(countCuts < IntEbmType { 0 })) {
                  LOG_0(TraceLevelError, "ERROR CutWinsorized countCuts can't be negative.");
                  ret = Error_IllegalParamValue;
               }
               goto exit_with_log;
            }

            if(UNLIKELY(IsConvertError<size_t>(countCuts))) {
               LOG_0(TraceLevelWarning, "WARNING CutWinsorized IsConvertError<size_t>(countCuts)");
               free(aFeatureValues);
               ret = Error_IllegalParamValue;
               goto exit_with_log;
            }
            const size_t cCuts = static_cast<size_t>(countCuts);

            if(UNLIKELY(IsMultiplyError(sizeof(*cutsLowerBoundInclusiveOut), cCuts))) {
               LOG_0(TraceLevelError, "ERROR CutWinsorized countCuts was too large to fit into cutsLowerBoundInclusiveOut");
               free(aFeatureValues);
               ret = Error_IllegalParamValue;
               goto exit_with_log;
            }

            if(UNLIKELY(nullptr == cutsLowerBoundInclusiveOut)) {
               // if we have a potential bin cut, then cutsLowerBoundInclusiveOut shouldn't be nullptr
               LOG_0(TraceLevelError, "ERROR CutWinsorized nullptr == cutsLowerBoundInclusiveOut");
               free(aFeatureValues);
               ret = Error_IllegalParamValue;
               goto exit_with_log;
            }

            // our +infinity values have been turned into max_float and -infinity values have been turned into 
            // lowest_float because we can't have a cut between max_float and +infinity without using a +infinity
            // cut value since we use lower bound inclusivity.  Other than +-infinity, if our dataset isn't completely
            // uniform we just need to find a single cut between values and we can divide the space up between
            // uniform bins between those values.

            std::sort(aFeatureValues, aFeatureValues + cSamples);

            if(UNLIKELY(size_t { 1 } == cCuts)) {
               // if we're only given 1 cut, then we need do so something special since we can't have an upper and
               // lower cut from which to range between.  We want to find the best central cut and use that

               const FloatEbmType minValue = aFeatureValues[0];
               const FloatEbmType maxValue = aFeatureValues[cSamples - size_t { 1 }];

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
                  *cutsLowerBoundInclusiveOut = avg;
                  countCutsRet = IntEbmType { 1 };
               }
            } else {
               // we check that cCuts can be multiplied with sizeof(*cutsLowerBoundInclusiveOut), and since
               // there is no way an element of cutsLowerBoundInclusiveOut is as small as 1 byte, we should
               // be able to add one to cCuts
               EBM_ASSERT(!IsAddError(size_t { 1 }, cCuts));
               const size_t cBins = cCuts + size_t { 1 };
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

                  FloatEbmType * pCutsLowerBoundInclusive = cutsLowerBoundInclusiveOut;
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
                     *pCutsLowerBoundInclusive = avg;
                     ++pCutsLowerBoundInclusive;
                     ++countCutsRet;
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
                     *pCutsLowerBoundInclusive = avg;
                     ++countCutsRet;
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
                     *cutsLowerBoundInclusiveOut = avg;
                     countCutsRet = IntEbmType { 1 };
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
                        cutsLowerBoundInclusiveOut[0] = avg1;

                        EBM_ASSERT(centerVal < highOuterVal);
                        const FloatEbmType avg2 = ArithmeticMean(centerVal, highOuterVal);
                        cutsLowerBoundInclusiveOut[1] = avg2;

                        countCutsRet = IntEbmType { 2 };
                     } else {
                        // move one up from the highVal since if we put a cut exactly there the high-low value will
                        // be included in the highest bin, and we don't want that
                        // winsorized binning doesn't have the ability to create humanized cut numbers, so going one tick up
                        // isn't a big deal here, and even if the high value is an integer like 5, one up from that will
                        // basically round to 5 in the UI but we'll get a number here that's just slighly higher

                        EBM_ASSERT(highInnerVal < std::numeric_limits<FloatEbmType>::max());
                        highInnerVal = std::nextafter(highInnerVal, std::numeric_limits<FloatEbmType>::max());

                        EBM_ASSERT(lowInnerVal < highInnerVal);
                        EBM_ASSERT(size_t { 2 } <= cCuts); // we can put down the low and high ones at least

                        FloatEbmType * pCutsLowerBoundInclusive = cutsLowerBoundInclusiveOut;
                        *pCutsLowerBoundInclusive = lowInnerVal;
                        ++pCutsLowerBoundInclusive;

                        if(size_t { 2 } < cCuts) {
                           while(true) {
                              const size_t cInternalRanges = cCuts - size_t { 1 };
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
                                    // overflow after dividing it up in to separate segments.  So, let's assume
                                    // that 2 == cBins, so we can just take the average and report one cut
                                    const FloatEbmType cut = ArithmeticMean(lowInnerVal, highInnerVal);
                                    // we always write out a cut at highInnerVal below, and we wouldn't want to do that
                                    // twice if we got back highInnerVal here, but we can only get here with huge
                                    // separations, so it shouldn't be possible to get something even close to 
                                    // highInnerVal
                                    EBM_ASSERT(highInnerVal != cut);
                                    *pCutsLowerBoundInclusive = cut;
                                    ++pCutsLowerBoundInclusive;
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
                                    EBM_ASSERT(cutsLowerBoundInclusiveOut < pCutsLowerBoundInclusive &&
                                       pCutsLowerBoundInclusive < cutsLowerBoundInclusiveOut + cCuts - size_t { 1 });

                                    *pCutsLowerBoundInclusive = cut;
                                    ++pCutsLowerBoundInclusive;
                                    cutPrev = cut;
                                 }
                                 ++iCut;
                              } while(cInternalRanges != iCut);
                              break;
                           }
                        }

                        EBM_ASSERT(cutsLowerBoundInclusiveOut < pCutsLowerBoundInclusive &&
                           pCutsLowerBoundInclusive < cutsLowerBoundInclusiveOut + cCuts);

                        // write the last one manually without resorting to a formula.  We don't allow our loop 
                        // above to write out the highVal, so we're guarnateed that we can write it here
                        *pCutsLowerBoundInclusive = highInnerVal;

                        const size_t cCutsRet = 
                           pCutsLowerBoundInclusive - cutsLowerBoundInclusiveOut + size_t { 1 };

                        // this conversion is guaranteed to work since the number of cut points can't exceed the number our user
                        // specified, and that value came to us as an IntEbmType
                        countCutsRet = static_cast<IntEbmType>(cCutsRet);
                        EBM_ASSERT(countCutsRet <= countCuts);
                     }
                  }
               }
            }
         }
         free(aFeatureValues);
         ret = Error_None;
      }

   exit_with_log:;

      EBM_ASSERT(nullptr != countCutsInOut);
      *countCutsInOut = countCutsRet;
   }

   LOG_COUNTED_N(
      &g_cLogExitCutWinsorizedParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Exited CutWinsorized: "
      "countCuts=%" IntEbmTypePrintf ", "
      "return=%" ErrorEbmTypePrintf
      ,
      countCutsRet,
      ret
   );

   return ret;
}

} // DEFINED_ZONE_NAME
