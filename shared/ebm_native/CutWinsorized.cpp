// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // std::numeric_limits
#include <algorithm> // std::sort
#include <string.h> // memcpy

#include "ebm_native.h" // EBM_API_BODY
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // LIKELY
#include "zones.h"

#include "common_cpp.hpp" // IsConvertError

#include "ebm_internal.hpp" // FloatTickIncrement

// TODO: check this file for how we handle subnormal numbers.  NEVER RETURN SUBNORMALS!

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern size_t RemoveMissingValsAndReplaceInfinities(const size_t cSamples, double * const aVals) noexcept;

extern double ArithmeticMean(
   const double low,
   const double high
) noexcept;

// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterCutWinsorized = 25;
static int g_cLogExitCutWinsorized = 25;

// TODO: add this as a python/R option "winsorized"
EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION CutWinsorized(
   IntEbm countSamples,
   const double * featureVals,
   IntEbm * countCutsInOut,
   double * cutsLowerBoundInclusiveOut
) {
   LOG_COUNTED_N(
      &g_cLogEnterCutWinsorized,
      Trace_Info,
      Trace_Verbose,
      "Entered CutWinsorized: "
      "countSamples=%" IntEbmPrintf ", "
      "featureVals=%p, "
      "countCutsInOut=%p, "
      "cutsLowerBoundInclusiveOut=%p"
      ,
      countSamples,
      static_cast<const void *>(featureVals),
      static_cast<void *>(countCutsInOut),
      static_cast<void *>(cutsLowerBoundInclusiveOut)
   );

   ErrorEbm error;

   IntEbm countCutsRet = IntEbm { 0 };

   if(UNLIKELY(nullptr == countCutsInOut)) {
      LOG_0(Trace_Error, "ERROR CutWinsorized nullptr == countCutsInOut");
      error = Error_IllegalParamVal;
   } else {
      if(UNLIKELY(countSamples <= IntEbm { 1 })) {
         // can't cut 1 sample
         error = Error_None;
         if(UNLIKELY(countSamples < IntEbm { 0 })) {
            LOG_0(Trace_Error, "ERROR CutWinsorized countSamples < IntEbm { 0 }");
            error = Error_IllegalParamVal;
         }
      } else {
         if(UNLIKELY(IsConvertError<size_t>(countSamples))) {
            LOG_0(Trace_Warning, "WARNING CutWinsorized IsConvertError<size_t>(countSamples)");

            error = Error_IllegalParamVal;
            goto exit_with_log;
         }

         if(UNLIKELY(nullptr == featureVals)) {
            LOG_0(Trace_Error, "ERROR CutWinsorized nullptr == featureVals");

            error = Error_IllegalParamVal;
            goto exit_with_log;
         }

         const size_t cSamplesIncludingMissingVals = static_cast<size_t>(countSamples);

         if(IsMultiplyError(sizeof(double), cSamplesIncludingMissingVals)) {
            LOG_0(Trace_Warning, "WARNING CutWinsorized IsMultiplyError(sizeof(double), cSamplesIncludingMissingVals)");
            error = Error_OutOfMemory;
            goto exit_with_log;
         }
         const size_t cBytesFeatureVals = sizeof(double) * cSamplesIncludingMissingVals;
         double * const aFeatureVals = static_cast<double *>(malloc(cBytesFeatureVals));
         if(UNLIKELY(nullptr == aFeatureVals)) {
            LOG_0(Trace_Error, "ERROR CutWinsorized nullptr == aFeatureVals");

            error = Error_OutOfMemory;
            goto exit_with_log;
         }
         memcpy(aFeatureVals, featureVals, cBytesFeatureVals);

         // if there are +infinity values in the data we won't be able to separate them
         // from max_float values without having a cut at infinity since we use lower bound inclusivity
         // so we disallow +infinity values by turning them into max_float.  For symmetry we do the same on
         // the -infinity side turning those into lowest_float.  
         const size_t cSamples = RemoveMissingValsAndReplaceInfinities(cSamplesIncludingMissingVals, aFeatureVals);

         EBM_ASSERT(cSamples <= cSamplesIncludingMissingVals);

         // we can't really cut 0 or 1 samples.  Now that we know our min, max, etc values, we can exit
         // or if there was only 1 non-missing value
         if(LIKELY(size_t { 1 } < cSamples)) {
            EBM_ASSERT(nullptr != countCutsInOut);
            const IntEbm countCuts = *countCutsInOut;

            if(UNLIKELY(countCuts <= IntEbm { 0 })) {
               free(aFeatureVals);
               error = Error_None;
               if(UNLIKELY(countCuts < IntEbm { 0 })) {
                  LOG_0(Trace_Error, "ERROR CutWinsorized countCuts can't be negative.");
                  error = Error_IllegalParamVal;
               }
               goto exit_with_log;
            }

            if(UNLIKELY(IsConvertError<size_t>(countCuts))) {
               LOG_0(Trace_Warning, "WARNING CutWinsorized IsConvertError<size_t>(countCuts)");
               free(aFeatureVals);
               error = Error_IllegalParamVal;
               goto exit_with_log;
            }
            const size_t cCuts = static_cast<size_t>(countCuts);

            if(UNLIKELY(IsMultiplyError(sizeof(*cutsLowerBoundInclusiveOut), cCuts))) {
               LOG_0(Trace_Error, "ERROR CutWinsorized countCuts was too large to fit into cutsLowerBoundInclusiveOut");
               free(aFeatureVals);
               error = Error_IllegalParamVal;
               goto exit_with_log;
            }

            if(UNLIKELY(nullptr == cutsLowerBoundInclusiveOut)) {
               // if we have a potential bin cut, then cutsLowerBoundInclusiveOut shouldn't be nullptr
               LOG_0(Trace_Error, "ERROR CutWinsorized nullptr == cutsLowerBoundInclusiveOut");
               free(aFeatureVals);
               error = Error_IllegalParamVal;
               goto exit_with_log;
            }

            // our +infinity values have been turned into max_float and -infinity values have been turned into 
            // lowest_float because we can't have a cut between max_float and +infinity without using a +infinity
            // cut value since we use lower bound inclusivity.  Other than +-infinity, if our dataset isn't completely
            // uniform we just need to find a single cut between values and we can divide the space up between
            // uniform bins between those values.

            std::sort(aFeatureVals, aFeatureVals + cSamples);

            if(UNLIKELY(size_t { 1 } == cCuts)) {
               // if we're only given 1 cut, then we need do so something special since we can't have an upper and
               // lower cut from which to range between.  We want to find the best central cut and use that

               const double valMin = aFeatureVals[0];
               const double valMax = aFeatureVals[cSamples - size_t { 1 }];

               // if this fails there are no transitions at all, so we can't have a cut
               if(LIKELY(valMin != valMax)) {
                  const size_t iCenterHigh = cSamples >> 1;
                  // put the low on the high side so that in our first loop we decrement it to it's actual position
                  const double * pLow = &aFeatureVals[iCenterHigh];
                  const double * pHigh = pLow - (size_t { 1 } & (cSamples - size_t { 1 }));

                  double lowCur;
                  double highCur;
                  do {
                     --pLow;
                     ++pHigh;

                     EBM_ASSERT(aFeatureVals <= pLow && pLow < aFeatureVals + cSamples);
                     lowCur = *pLow;
                     EBM_ASSERT(aFeatureVals <= pHigh && pHigh < aFeatureVals + cSamples);
                     highCur = *pHigh;
                     // since valMin != valMax per above, we know there is a transition SOMEWHERE, which will exit us
                  } while(LIKELY(lowCur == highCur));
                  EBM_ASSERT(lowCur < highCur);

                  // if both lowCur and highCur have changed, we'll get the average value between them, but that'll
                  // put the valCenter either on the low or high bin dependent on the values.  Unlike quantile binning, 
                  // winsorized binning is senitive to the values and not invariant to operations, so this is fine

                  const double avg = ArithmeticMean(lowCur, highCur);
                  *cutsLowerBoundInclusiveOut = avg;
                  countCutsRet = IntEbm { 1 };
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
               const double * pLow = &aFeatureVals[iOuterBound];
               // position ourselves at high-low and move inwards
               const double * pHigh = &aFeatureVals[cSamples - iOuterBound - size_t { 1 }];

               EBM_ASSERT(aFeatureVals <= pLow && pLow < aFeatureVals + cSamples);
               EBM_ASSERT(aFeatureVals <= pHigh && pHigh < aFeatureVals + cSamples);

               const double lowOuterVal = *pLow;
               const double highOuterVal = *pHigh;
               EBM_ASSERT(lowOuterVal <= highOuterVal);

               if(UNLIKELY(lowOuterVal == highOuterVal)) {
                  // there are no transitions between our outer values.  We have just 1 single value between them
                  // one way to handle this would be to wrap the value on the low side with the exact value
                  // and epsilon higher on the high side, but that's rather tight and makes the graph look too
                  // tight.  We instead put the two cuts between the outer values and the next transition outwards

                  const double valCenter = lowOuterVal;

                  const double valMin = aFeatureVals[0];
                  const double valMax = aFeatureVals[cSamples - size_t { 1 }];

                  double * pCutsLowerBoundInclusive = cutsLowerBoundInclusiveOut;
                  if(PREDICTABLE(valMin != valCenter)) {
                     // there's a transition somewhere on the low side
                     EBM_ASSERT(std::numeric_limits<double>::lowest() < valCenter);
                     EBM_ASSERT(valMin < valCenter);

                     double valCur;
                     do {
                        --pLow;
                        EBM_ASSERT(aFeatureVals <= pLow && pLow < aFeatureVals + cSamples);
                        valCur = *pLow;
                     } while(valCenter == valCur);
                     EBM_ASSERT(valCur < valCenter);

                     const double avg = ArithmeticMean(valCur, valCenter);
                     *pCutsLowerBoundInclusive = avg;
                     ++pCutsLowerBoundInclusive;
                     ++countCutsRet;
                  }
                  if(PREDICTABLE(valMax != valCenter)) {
                     // there's a transition somewhere on the high side
                     EBM_ASSERT(valCenter < std::numeric_limits<double>::max());
                     EBM_ASSERT(valCenter < valMax);

                     double valCur;
                     do {
                        ++pHigh;
                        EBM_ASSERT(aFeatureVals <= pHigh && pHigh < aFeatureVals + cSamples);
                        valCur = *pHigh;
                     } while(valCenter == valCur);
                     EBM_ASSERT(valCenter < valCur);

                     const double avg = ArithmeticMean(valCenter, valCur);
                     *pCutsLowerBoundInclusive = avg;
                     ++countCutsRet;
                  }
               } else {
                  // because lowVal != highVal, we know there's a transition that'll exit our loops somewhere in between

                  double lowInnerVal;
                  do {
                     ++pLow;
                     EBM_ASSERT(aFeatureVals <= pLow && pLow < aFeatureVals + cSamples);
                     lowInnerVal = *pLow;
                  } while(lowOuterVal == lowInnerVal);
                  EBM_ASSERT(std::numeric_limits<double>::lowest() < lowInnerVal);
                  EBM_ASSERT(lowOuterVal < lowInnerVal);
                  EBM_ASSERT(lowInnerVal <= highOuterVal);

                  if(lowInnerVal == highOuterVal) {
                     // there was just a single transition between our two outer values.  This doesn't really give
                     // us anything to wrap, so let's just return a single cut point in the middle of the transition
                     // space

                     EBM_ASSERT(lowOuterVal < highOuterVal);
                     const double avg = ArithmeticMean(lowOuterVal, highOuterVal);
                     *cutsLowerBoundInclusiveOut = avg;
                     countCutsRet = IntEbm { 1 };
                  } else {
                     double highInnerVal;
                     do {
                        --pHigh;
                        EBM_ASSERT(aFeatureVals <= pHigh && pHigh < aFeatureVals + cSamples);
                        highInnerVal = *pHigh;
                     } while(highOuterVal == highInnerVal);
                     EBM_ASSERT(highInnerVal < std::numeric_limits<double>::max());
                     EBM_ASSERT(highInnerVal < highOuterVal);
                     EBM_ASSERT(lowInnerVal <= highInnerVal);
                     EBM_ASSERT(lowOuterVal < highInnerVal);

                     if(lowInnerVal == highInnerVal) {
                        // there's just one long run of values in the center.  We don't really want to wrap any
                        // number this tightly, so let's put down two cut points in the spaces on either side

                        const double valCenter = lowInnerVal;

                        EBM_ASSERT(lowOuterVal < valCenter);
                        const double avg1 = ArithmeticMean(lowOuterVal, valCenter);
                        cutsLowerBoundInclusiveOut[0] = avg1;

                        EBM_ASSERT(valCenter < highOuterVal);
                        const double avg2 = ArithmeticMean(valCenter, highOuterVal);
                        cutsLowerBoundInclusiveOut[1] = avg2;

                        countCutsRet = IntEbm { 2 };
                     } else {
                        // move one up from the highVal since if we put a cut exactly there the high-low value will
                        // be included in the highest bin, and we don't want that
                        // winsorized binning doesn't have the ability to create rounded cut numbers, so going one tick up
                        // isn't a big deal here, and even if the high value is an integer like 5, one up from that will
                        // basically round to 5 in the UI but we'll get a number here that's just slighly higher

                        EBM_ASSERT(highInnerVal < std::numeric_limits<double>::max());
                        highInnerVal = FloatTickIncrement(highInnerVal);

                        EBM_ASSERT(lowInnerVal < highInnerVal);
                        EBM_ASSERT(size_t { 2 } <= cCuts); // we can put down the low and high ones at least

                        double * pCutsLowerBoundInclusive = cutsLowerBoundInclusiveOut;
                        *pCutsLowerBoundInclusive = lowInnerVal;
                        ++pCutsLowerBoundInclusive;

                        if(size_t { 2 } < cCuts) {
                           while(true) {
                              const size_t cInternalRanges = cCuts - size_t { 1 };
                              const double cInternalRangesFloat = static_cast<double>(cInternalRanges);
                              double stepVal = (highInnerVal - lowInnerVal) / cInternalRangesFloat;
                              if(std::isinf(stepVal)) {
                                 // cInternalRangesFloat should be 3 or higher, which should be enough to avoid any numeracy issues
                                 // that might cause us to overflow again
                                 stepVal = highInnerVal / cInternalRangesFloat - lowInnerVal / cInternalRangesFloat;
                                 if(std::isinf(stepVal)) {
                                    // this is probably impossible if correct rounding is guarnateed, but floats have bad guarantees

                                    // if you have 2 internal bins it would be close to an overflow on the subtraction 
                                    // of the divided values.  With 3 bins it isn't obvious to me how you'd get an
                                    // overflow after dividing it up in to separate segments.  So, let's assume
                                    // that 2 == cBins, so we can just take the average and report one cut
                                    const double cut = ArithmeticMean(lowInnerVal, highInnerVal);
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
                              double cutPrev = lowInnerVal;
                              size_t iCut = size_t { 1 };
                              do {
                                 const double cut = lowInnerVal + stepVal * static_cast<double>(iCut);
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
                        // specified, and that value came to us as an IntEbm
                        countCutsRet = static_cast<IntEbm>(cCutsRet);
                        EBM_ASSERT(countCutsRet <= countCuts);
                     }
                  }
               }
            }
         }
         free(aFeatureVals);
         error = Error_None;
      }

   exit_with_log:;

      EBM_ASSERT(nullptr != countCutsInOut);
      *countCutsInOut = countCutsRet;
   }

   LOG_COUNTED_N(
      &g_cLogExitCutWinsorized,
      Trace_Info,
      Trace_Verbose,
      "Exited CutWinsorized: "
      "countCuts=%" IntEbmPrintf ", "
      "return=%" ErrorEbmPrintf
      ,
      countCutsRet,
      error
   );

   return error;
}

} // DEFINED_ZONE_NAME
