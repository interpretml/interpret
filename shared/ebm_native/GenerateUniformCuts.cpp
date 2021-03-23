// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // std::numeric_limits

#include "ebm_native.h"
#include "EbmInternal.h"
#include "logging.h" // EBM_ASSERT & LOG

extern FloatEbmType ArithmeticMean(
   const FloatEbmType low,
   const FloatEbmType high
) noexcept;

INLINE_RELEASE_UNTEMPLATED static size_t DetermineBounds(
   size_t cSamples,
   const FloatEbmType * const aValues,
   FloatEbmType * const pMinNonInfinityValueOut,
   IntEbmType * const pCountNegativeInfinityOut,
   FloatEbmType * const pMaxNonInfinityValueOut,
   IntEbmType * const pCountPositiveInfinityOut
) noexcept {
   EBM_ASSERT(size_t { 1 } <= cSamples);
   EBM_ASSERT(nullptr != aValues);
   EBM_ASSERT(nullptr != pMinNonInfinityValueOut);
   EBM_ASSERT(nullptr != pCountNegativeInfinityOut);
   EBM_ASSERT(nullptr != pMaxNonInfinityValueOut);
   EBM_ASSERT(nullptr != pCountPositiveInfinityOut);

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

   FloatEbmType minNonInfinityValue = std::numeric_limits<FloatEbmType>::max();
   size_t cNegativeInfinity = size_t { 0 };
   FloatEbmType maxNonInfinityValue = std::numeric_limits<FloatEbmType>::lowest();
   size_t cPositiveInfinity = size_t { 0 };

   size_t cSamplesWithoutMissing = cSamples;
   const FloatEbmType * pValue = aValues;
   const FloatEbmType * const pValuesEnd = aValues + cSamples;
   do {
      FloatEbmType val = *pValue;
      if(UNLIKELY(std::isnan(val))) {
         EBM_ASSERT(0 < cSamplesWithoutMissing);
         --cSamplesWithoutMissing;
      } else if(PREDICTABLE(std::numeric_limits<FloatEbmType>::infinity() == val)) {
         ++cPositiveInfinity;
      } else if(PREDICTABLE(-std::numeric_limits<FloatEbmType>::infinity() == val)) {
         ++cNegativeInfinity;
      } else {
         maxNonInfinityValue = UNPREDICTABLE(maxNonInfinityValue < val) ? val : maxNonInfinityValue;
         minNonInfinityValue = UNPREDICTABLE(val < minNonInfinityValue) ? val : minNonInfinityValue;
      }
      ++pValue;
   } while(LIKELY(pValuesEnd != pValue));
   EBM_ASSERT(cSamplesWithoutMissing <= cSamples);

   if(UNLIKELY(cNegativeInfinity + cPositiveInfinity == cSamplesWithoutMissing)) {
      // all values were special values (missing, +infinity, -infinity), so make our min/max both zero
      maxNonInfinityValue = FloatEbmType { 0 };
      minNonInfinityValue = FloatEbmType { 0 };
   }

   *pMinNonInfinityValueOut = minNonInfinityValue;
   // this can't overflow since we got our cSamples from an IntEbmType, and we can't have more infinities than that
   *pCountNegativeInfinityOut = static_cast<IntEbmType>(cNegativeInfinity);
   *pMaxNonInfinityValueOut = maxNonInfinityValue;
   // this can't overflow since we got our cSamples from an IntEbmType, and we can't have more infinities than that
   *pCountPositiveInfinityOut = static_cast<IntEbmType>(cPositiveInfinity);

   return cSamplesWithoutMissing;
}

// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterGenerateUniformCutsParametersMessages = 25;
static int g_cLogExitGenerateUniformCutsParametersMessages = 25;

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION GenerateUniformCuts(
   IntEbmType countSamples,
   const FloatEbmType * featureValues,
   IntEbmType * countCutsInOut,
   FloatEbmType * cutsLowerBoundInclusiveOut,
   IntEbmType * countMissingValuesOut,
   FloatEbmType * minNonInfinityValueOut,
   IntEbmType * countNegativeInfinityOut,
   FloatEbmType * maxNonInfinityValueOut,
   IntEbmType * countPositiveInfinityOut
) {
   LOG_COUNTED_N(
      &g_cLogEnterGenerateUniformCutsParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered GenerateUniformCuts: "
      "countSamples=%" IntEbmTypePrintf ", "
      "featureValues=%p, "
      "countCutsInOut=%p, "
      "cutsLowerBoundInclusiveOut=%p, "
      "countMissingValuesOut=%p, "
      "minNonInfinityValueOut=%p, "
      "countNegativeInfinityOut=%p, "
      "maxNonInfinityValueOut=%p, "
      "countPositiveInfinityOut=%p"
      ,
      countSamples,
      static_cast<const void *>(featureValues),
      static_cast<void *>(countCutsInOut),
      static_cast<void *>(cutsLowerBoundInclusiveOut),
      static_cast<void *>(countMissingValuesOut),
      static_cast<void *>(minNonInfinityValueOut),
      static_cast<void *>(countNegativeInfinityOut),
      static_cast<void *>(maxNonInfinityValueOut),
      static_cast<void *>(countPositiveInfinityOut)
   );

   IntEbmType countCutsRet = IntEbmType { 0 };
   IntEbmType countMissingValuesRet;
   FloatEbmType minNonInfinityValueRet;
   IntEbmType countNegativeInfinityRet;
   FloatEbmType maxNonInfinityValueRet;
   IntEbmType countPositiveInfinityRet;

   if(UNLIKELY(nullptr == countCutsInOut)) {
      LOG_0(TraceLevelError, "ERROR GenerateUniformCuts nullptr == countCutsInOut");
      countMissingValuesRet = IntEbmType { 0 };
      minNonInfinityValueRet = FloatEbmType { 0 };
      countNegativeInfinityRet = IntEbmType { 0 };
      maxNonInfinityValueRet = FloatEbmType { 0 };
      countPositiveInfinityRet = IntEbmType { 0 };
   } else {
      if(UNLIKELY(countSamples <= IntEbmType { 0 })) {
         // if there's 1 sample, then we can't split it, but we'd still want to determine the min, max, etc
         // so continue processing

         countMissingValuesRet = IntEbmType { 0 };
         minNonInfinityValueRet = FloatEbmType { 0 };
         countNegativeInfinityRet = IntEbmType { 0 };
         maxNonInfinityValueRet = FloatEbmType { 0 };
         countPositiveInfinityRet = IntEbmType { 0 };
         if(UNLIKELY(countSamples < IntEbmType { 0 })) {
            LOG_0(TraceLevelError, "ERROR GenerateUniformCuts countSamples < IntEbmType { 0 }");
         }
      } else {
         if(UNLIKELY(!IsNumberConvertable<size_t>(countSamples))) {
            LOG_0(TraceLevelWarning, "WARNING GenerateUniformCuts !IsNumberConvertable<size_t>(countSamples)");

            countMissingValuesRet = IntEbmType { 0 };
            minNonInfinityValueRet = FloatEbmType { 0 };
            countNegativeInfinityRet = IntEbmType { 0 };
            maxNonInfinityValueRet = FloatEbmType { 0 };
            countPositiveInfinityRet = IntEbmType { 0 };
            goto exit_with_log;
         }

         if(UNLIKELY(nullptr == featureValues)) {
            LOG_0(TraceLevelError, "ERROR GenerateUniformCuts nullptr == featureValues");

            countMissingValuesRet = IntEbmType { 0 };
            minNonInfinityValueRet = FloatEbmType { 0 };
            countNegativeInfinityRet = IntEbmType { 0 };
            maxNonInfinityValueRet = FloatEbmType { 0 };
            countPositiveInfinityRet = IntEbmType { 0 };
            goto exit_with_log;
         }

         const size_t cSamplesIncludingMissingValues = static_cast<size_t>(countSamples);

         if(UNLIKELY(IsMultiplyError(sizeof(*featureValues), cSamplesIncludingMissingValues))) {
            LOG_0(TraceLevelError, "ERROR GenerateUniformCuts countSamples was too large to fit into featureValues");

            countMissingValuesRet = IntEbmType { 0 };
            minNonInfinityValueRet = FloatEbmType { 0 };
            countNegativeInfinityRet = IntEbmType { 0 };
            maxNonInfinityValueRet = FloatEbmType { 0 };
            countPositiveInfinityRet = IntEbmType { 0 };
            goto exit_with_log;
         }

         // if there are +infinity values in the data we won't be able to separate them
         // from max_float values without having a cut at infinity since we use lower bound inclusivity
         // so we disallow +infinity values by turning them into max_float.  For symmetry we do the same on
         // the -infinity side turning those into lowest_float.  
         const size_t cSamples = DetermineBounds(
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

         // our minValue and maxValue calculations below depend on there being at least 1 sample
         if(size_t { 0 } != cSamples) {
            // if all of the samples are positive infinity then minValue is max, otherwise if there are any
            // negative infinities, then the min will be lowest.  Same for the max, but in reverse.
            const FloatEbmType minValue = UNLIKELY(cSamples == static_cast<size_t>(countPositiveInfinityRet)) ?
               std::numeric_limits<FloatEbmType>::max() :
               (UNPREDICTABLE(0 == countNegativeInfinityRet) ?
                  minNonInfinityValueRet : std::numeric_limits<FloatEbmType>::lowest());
            const FloatEbmType maxValue = UNLIKELY(cSamples == static_cast<size_t>(countNegativeInfinityRet)) ?
               std::numeric_limits<FloatEbmType>::lowest() :
               (UNPREDICTABLE(0 == countPositiveInfinityRet) ?
                  maxNonInfinityValueRet : std::numeric_limits<FloatEbmType>::max());

            if(PREDICTABLE(minValue != maxValue)) {
               EBM_ASSERT(nullptr != countCutsInOut);
               const IntEbmType countCuts = *countCutsInOut;

               if(UNLIKELY(countCuts <= IntEbmType { 0 })) {
                  if(UNLIKELY(countCuts < IntEbmType { 0 })) {
                     LOG_0(TraceLevelError, "ERROR GenerateUniformCuts countCuts can't be negative.");
                  }
                  goto exit_with_log;
               }

               if(UNLIKELY(!IsNumberConvertable<size_t>(countCuts))) {
                  LOG_0(TraceLevelWarning, "WARNING GenerateUniformCuts !IsNumberConvertable<size_t>(countCuts)");
                  goto exit_with_log;
               }
               const size_t cCuts = static_cast<size_t>(countCuts);

               if(UNLIKELY(IsMultiplyError(sizeof(*cutsLowerBoundInclusiveOut), cCuts))) {
                  LOG_0(TraceLevelError, "ERROR GenerateUniformCuts countCuts was too large to fit into cutsLowerBoundInclusiveOut");
                  goto exit_with_log;
               }

               if(UNLIKELY(nullptr == cutsLowerBoundInclusiveOut)) {
                  // if we have a potential bin cut, then cutsLowerBoundInclusiveOut shouldn't be nullptr
                  LOG_0(TraceLevelError, "ERROR GenerateUniformCuts nullptr == cutsLowerBoundInclusiveOut");
                  goto exit_with_log;
               }

               // we check that cCuts can be multiplied with sizeof(*cutsLowerBoundInclusiveOut), and since
               // there is no way an element of cutsLowerBoundInclusiveOut is as small as 1 byte, we should
               // be able to add one to cCuts
               EBM_ASSERT(!IsAddError(cCuts, size_t { 1 }));
               const size_t cBins = cCuts + size_t { 1 };

               const FloatEbmType cBinsFloat = static_cast<FloatEbmType>(cBins);
               FloatEbmType stepValue = (maxValue - minValue) / cBinsFloat;
               if(std::isinf(stepValue)) {
                  stepValue = maxValue / cBinsFloat - minValue / cBinsFloat;
                  if(std::isinf(stepValue)) {
                     // this is probably impossible if correct rounding is guarnateed, but floats have bad guarantees

                     // if you have 2 internal bins it would be close to an overflow on the subtraction 
                     // of the divided values.  With 3 bins it isn't obvious to me how you'd get an
                     // overflow after dividing it up into separate divisions.  So, let's assume
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
      &g_cLogExitGenerateUniformCutsParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Exited GenerateUniformCuts: "
      "countCuts=%" IntEbmTypePrintf ", "
      "countMissingValues=%" IntEbmTypePrintf ", "
      "minNonInfinityValue=%" FloatEbmTypePrintf ", "
      "countNegativeInfinity=%" IntEbmTypePrintf ", "
      "maxNonInfinityValue=%" FloatEbmTypePrintf ", "
      "countPositiveInfinity=%" IntEbmTypePrintf
      ,
      countCutsRet,
      countMissingValuesRet,
      minNonInfinityValueRet,
      countNegativeInfinityRet,
      maxNonInfinityValueRet,
      countPositiveInfinityRet
   );
}
