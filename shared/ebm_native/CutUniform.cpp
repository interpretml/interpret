// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // std::numeric_limits

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern FloatEbmType ArithmeticMean(
   const FloatEbmType low,
   const FloatEbmType high
) noexcept;

INLINE_RELEASE_UNTEMPLATED static size_t DetermineBounds(
   const size_t cSamples,
   const FloatEbmType * const aValues,
   FloatEbmType * const pMinValueOut,
   FloatEbmType * const pMaxValueOut
) noexcept {
   EBM_ASSERT(size_t { 1 } <= cSamples);
   EBM_ASSERT(nullptr != aValues);
   EBM_ASSERT(nullptr != pMinValueOut);
   EBM_ASSERT(nullptr != pMaxValueOut);

   // In most cases we believe that for graphing the caller should only need the bin cuts that we'll eventually
   // return, and they'll want to position the graph to include the first and last cuts, and have a little bit of 
   // space both above and below those cuts.  In most cases they shouldn't need the non-infinity min/max values or know
   // whether or not there is +-infinity in the data, BUT on the margins of choosing graphing it might be useful.
   // For example, if the first cut was at 0.1 it might be reasonable to think that the low boundary should be 0,
   // and that would be reasonable if the lowest true value was 0.01, but if the lowest value was actually -0.1,
   // then we might want to instead make our graph start at -1.  Likewise, knowing if there were +-infinity 
   // values in the data probably won't affect the bounds shown, but perhaps the graphing code might want to 
   // somehow indicate the existance of +-infinity values.  The user might write custom graphing code, so we should
   // just return all this information and let the user choose what they want.

   // we really don't want to have cut points that are either -infinity or +infinity because these values are 
   // problematic for serialization, cross language compatibility, human understantability, graphing, etc.
   // In some cases though, +-infinity might carry some information that we do want to capture.  In almost all
   // cases though we can put a cut point between -infinity and the smallest value or +infinity and the largest
   // value.  One corner case is if our data has both max_float and +infinity values.  Our binning uses
   // lower inclusive bounds, so a cut value of max_float will include both max_float and +infinity, so if
   // our algorithm decides to put a cut there we'd be in trouble.  We don't want to make the cut +infinity
   // since that violates our no infinity cut point rule above.  A good compromise is to turn +infinity
   // into max_float.  If we do it here, our cutting algorithm won't need to deal with the odd case of indicating
   // a cut and removing it later.  In theory we could separate -infinity and min_float, since a cut value of
   // min_float would separate the two, but we convert -infinity to min_float here for symmetry with the positive
   // case and for simplicity.

   // when +-infinity values and min_float/max_float values are present, they usually don't represent real values,
   // since it's exceedingly unlikley that min_float or max_float represents a natural value that just happened
   // to not overflow.  When picking our cut points later between values, we should care more about the highest
   // or lowest value that is not min_float/max_float/+-infinity.  So, we convert +-infinity to min_float/max_float
   // here and disregard that value when choosing bin cut points.  We put the bin cut closer to the other value
   // A good point to put the cut is the value that has the same exponent, but increments the top value, so for
   // example, (7.84222e22, +infinity) should have a bin cut value of 8e22).

   // all of this infrastructure gives the user back the maximum amount of information possible, while also avoiding
   // +-infinity values in either the cut points, or the min/max values, which is good since serialization of
   // +-infinity isn't very standardized accross languages.  It's a problem in JSON especially.

   FloatEbmType minValue = std::numeric_limits<FloatEbmType>::infinity();
   FloatEbmType maxValue = -std::numeric_limits<FloatEbmType>::infinity();
   
   size_t cSamplesWithoutMissing = cSamples;
   const FloatEbmType * pValue = aValues;
   const FloatEbmType * const pValuesEnd = aValues + cSamples;
   do {
      const FloatEbmType val = *pValue;
      cSamplesWithoutMissing = UNPREDICTABLE(std::isnan(val)) ? cSamplesWithoutMissing - 1 : cSamplesWithoutMissing;
      maxValue = UNPREDICTABLE(maxValue < val) ? val : maxValue; // this works for NaN values which eval to false
      minValue = UNPREDICTABLE(val < minValue) ? val : minValue; // this works for NaN values which eval to false
      ++pValue;
   } while(LIKELY(pValuesEnd != pValue));
   EBM_ASSERT(cSamplesWithoutMissing <= cSamples);

   *pMinValueOut = minValue;
   *pMaxValueOut = maxValue;

   return cSamplesWithoutMissing;
}

// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterCutUniformParametersMessages = 25;
static int g_cLogExitCutUniformParametersMessages = 25;

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION CutUniform(
   IntEbmType countSamples,
   const FloatEbmType * featureValues,
   IntEbmType * countCutsInOut,
   FloatEbmType * cutsLowerBoundInclusiveOut
) {
   LOG_COUNTED_N(
      &g_cLogEnterCutUniformParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered CutUniform: "
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

   if(UNLIKELY(nullptr == countCutsInOut)) {
      LOG_0(TraceLevelError, "ERROR CutUniform nullptr == countCutsInOut");
   } else {
      if(UNLIKELY(countSamples <= IntEbmType { 1 })) {
         // can't cut 1 sample by itself
         if(UNLIKELY(countSamples < IntEbmType { 0 })) {
            LOG_0(TraceLevelError, "ERROR CutUniform countSamples < IntEbmType { 0 }");
         }
      } else {
         if(UNLIKELY(IsConvertError<size_t>(countSamples))) {
            LOG_0(TraceLevelWarning, "WARNING CutUniform IsConvertError<size_t>(countSamples)");
            goto exit_with_log;
         }

         if(UNLIKELY(nullptr == featureValues)) {
            LOG_0(TraceLevelError, "ERROR CutUniform nullptr == featureValues");
            goto exit_with_log;
         }

         const size_t cSamplesIncludingMissingValues = static_cast<size_t>(countSamples);

         if(UNLIKELY(IsMultiplyError(sizeof(*featureValues), cSamplesIncludingMissingValues))) {
            LOG_0(TraceLevelError, "ERROR CutUniform countSamples was too large to fit into featureValues");
            goto exit_with_log;
         }

         // if there are +infinity values in the data we won't be able to separate them
         // from max_float values without having a cut at infinity since we use lower bound inclusivity
         // so we disallow +infinity values by turning them into max_float.  For symmetry we do the same on
         // the -infinity side turning those into lowest_float.  
         FloatEbmType minValue;
         FloatEbmType maxValue;
         const size_t cSamples = DetermineBounds(
            cSamplesIncludingMissingValues,
            featureValues,
            &minValue,
            &maxValue
         );
         EBM_ASSERT(cSamples <= cSamplesIncludingMissingValues);

         // can't cut 0 or 1 samples
         if(PREDICTABLE(1 < cSamples)) {
            if(minValue == -std::numeric_limits<FloatEbmType>::infinity()) {
               minValue = std::numeric_limits<FloatEbmType>::lowest();
            } else if(minValue == std::numeric_limits<FloatEbmType>::infinity()) {
               // this can only happen if the only data is +infinity
               minValue = std::numeric_limits<FloatEbmType>::max();
            }

            if(maxValue == std::numeric_limits<FloatEbmType>::infinity()) {
               maxValue = std::numeric_limits<FloatEbmType>::max();
            } else if(maxValue == -std::numeric_limits<FloatEbmType>::infinity()) {
               // this can only happen if the only data is -infinity
               maxValue = std::numeric_limits<FloatEbmType>::lowest();
            }

            if(PREDICTABLE(minValue != maxValue)) {
               EBM_ASSERT(minValue < maxValue);
               EBM_ASSERT(nullptr != countCutsInOut);
               const IntEbmType countCuts = *countCutsInOut;

               if(UNLIKELY(countCuts <= IntEbmType { 0 })) {
                  if(UNLIKELY(countCuts < IntEbmType { 0 })) {
                     LOG_0(TraceLevelError, "ERROR CutUniform countCuts can't be negative.");
                  }
                  goto exit_with_log;
               }

               if(UNLIKELY(IsConvertError<size_t>(countCuts))) {
                  LOG_0(TraceLevelWarning, "WARNING CutUniform IsConvertError<size_t>(countCuts)");
                  goto exit_with_log;
               }
               const size_t cCuts = static_cast<size_t>(countCuts);

               if(UNLIKELY(IsMultiplyError(sizeof(*cutsLowerBoundInclusiveOut), cCuts))) {
                  LOG_0(TraceLevelError, "ERROR CutUniform countCuts was too large to fit into cutsLowerBoundInclusiveOut");
                  goto exit_with_log;
               }

               if(UNLIKELY(nullptr == cutsLowerBoundInclusiveOut)) {
                  // if we have a potential bin cut, then cutsLowerBoundInclusiveOut shouldn't be nullptr
                  LOG_0(TraceLevelError, "ERROR CutUniform nullptr == cutsLowerBoundInclusiveOut");
                  goto exit_with_log;
               }

               // we check that cCuts can be multiplied with sizeof(*cutsLowerBoundInclusiveOut), and since
               // there is no way an element of cutsLowerBoundInclusiveOut is as small as 1 byte, we should
               // be able to add one to cCuts
               EBM_ASSERT(!IsAddError(size_t { 1 }, cCuts));
               const size_t cBins = cCuts + size_t { 1 };

               const FloatEbmType cBinsFloat = static_cast<FloatEbmType>(cBins);
               FloatEbmType stepValue = (maxValue - minValue) / cBinsFloat;
               if(std::isinf(stepValue)) {
                  stepValue = maxValue / cBinsFloat - minValue / cBinsFloat;
                  if(std::isinf(stepValue)) {
                     // this is probably impossible if correct rounding is guarnateed, but floats have bad guarantees

                     // if you have 2 internal bins it would be close to an overflow on the subtraction 
                     // of the divided values.  With 3 bins it isn't obvious to me how you'd get an
                     // overflow after dividing it up into separate segments.  So, let's assume
                     // that 2 == cBins, so we can just take the average and report one cut
                     const FloatEbmType avg = ArithmeticMean(minValue, maxValue);
                     *cutsLowerBoundInclusiveOut = avg;
                     countCutsRet = IntEbmType { 1 };
                     goto exit_with_log;
                  }
               }
               if(stepValue <= FloatEbmType { 0 }) {
                  // if stepValue underflows, we can still put a cut between the minValue and maxValue
                  // we can also pickup a free check against odd floating point behavior that returns a negative here

                  const FloatEbmType avg = ArithmeticMean(minValue, maxValue);
                  *cutsLowerBoundInclusiveOut = avg;
                  countCutsRet = IntEbmType { 1 };
               } else {
                  EBM_ASSERT(FloatEbmType { 0 } < stepValue);
                  // we don't want a first cut that's the minValue anyways, since then we'd have zero items in the
                  // lowest bin given that we use lower bound inclusive semantics here
                  FloatEbmType cutPrev = minValue;
                  FloatEbmType * pCutsLowerBoundInclusive = cutsLowerBoundInclusiveOut;
                  size_t iCut = size_t { 1 };
                  do {
                     const FloatEbmType cut = minValue + stepValue * static_cast<FloatEbmType>(iCut);
                     // just in case we have floating point inexactness that puts us above the highValue we need to stop
                     if(UNLIKELY(maxValue < cut)) {
                        // this could only happen due to numeric instability
                        break;
                     }
                     if(LIKELY(cutPrev != cut)) {
                        EBM_ASSERT(cutPrev < cut);
                        EBM_ASSERT(cutsLowerBoundInclusiveOut <= pCutsLowerBoundInclusive &&
                           pCutsLowerBoundInclusive < cutsLowerBoundInclusiveOut + cCuts);

                        *pCutsLowerBoundInclusive = cut;
                        ++pCutsLowerBoundInclusive;
                        cutPrev = cut;
                     }
                     ++iCut;
                  } while(cBins != iCut);

                  const size_t cCutsRet = pCutsLowerBoundInclusive - cutsLowerBoundInclusiveOut;

                  // this conversion is guaranteed to work since the number of cut points can't exceed the number our user
                  // specified, and that value came to us as an IntEbmType
                  countCutsRet = static_cast<IntEbmType>(cCutsRet);
               }
               EBM_ASSERT(countCutsRet <= countCuts);
            }
         }
      }

   exit_with_log:;

      EBM_ASSERT(nullptr != countCutsInOut);
      *countCutsInOut = countCutsRet;
   }

   LOG_COUNTED_N(
      &g_cLogExitCutUniformParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Exited CutUniform: "
      "countCuts=%" IntEbmTypePrintf
      ,
      countCutsRet
   );
}

} // DEFINED_ZONE_NAME
