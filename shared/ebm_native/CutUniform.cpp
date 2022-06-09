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

// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterCutUniformParametersMessages = 25;
static int g_cLogExitCutUniformParametersMessages = 25;

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION CutUniform(
   IntEbmType countSamples,
   const FloatEbmType * featureValues,
   IntEbmType countDesiredCuts,
   FloatEbmType * cutsLowerBoundInclusiveOut
) {
   // DO NOT CHANGE THIS FUNCTION'S ALGORITHM.  IT IS PART OF THE EBM HISTOGRAM SPEC
   //
   // This function is used when choosing histograms cuts. Since we don't store the histogram 
   // cuts in our JSON, this function must always return the same results in all implementations.
   //
   // This function guarantees that it will return countDesiredCuts unless it is impossible to put enough cuts between
   // the min and max values, and in that case it will put a cut into every possible location allowed by floats.
   // This functionality is required because if we have a given number of histogram counts, then we need
   // to return the corresponding number of expected cuts, so returning the maximum number of cuts allowed avoids
   // issues because our caller would have to throw an exception if they got back less cuts than expected.
   // 
   // When later discretizing, we include numbers equal to the cut in the bin because we want
   // a cut of 1.0 to include 1.0 in the same bin as 1.999999... (if the next cut is at 2.0)
   //
   // This means we never return the min value in our cuts since that will be 
   // separated from the min value by the next highest floating point number.
   //
   // This also means that we can return a cut at the max value since a cut there will keep
   // the max value in the highest bin, so we do have an asymmetry that we fundamentally can't 
   // avoid since we need to make a choice whether exact matches fall into the lower or upper bin.
   //
   // Since our cuts can include the max value, this means that we cannot separate +highest from +inf values without 
   // allowing a +inf cut value, which we don't want to do. So, +inf values get converted to +highest
   // before choosing the cuts.  It would be hard to do anything else anyways since +inf is hard to use.
   //
   // For symmetry, we set -inf to lowest before generating uniform cuts.
   //
   // For histograms edges, our caller will add the min and max values to the returned cuts. We do include
   // -inf and +inf values there, so we still preserve the best information when graphing and serializing.
   //
   // We do not allow cut points to be subnormal floats. Even if we can handle subnormals ourselves here, other 
   // callers might not be able to handle them.  They might not serialize to JSON or work in JavaScript, etc.

   LOG_COUNTED_N(
      &g_cLogEnterCutUniformParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered CutUniform: "
      "countSamples=%" IntEbmTypePrintf ", "
      "featureValues=%p, "
      "countDesiredCuts=%" IntEbmTypePrintf ", "
      "cutsLowerBoundInclusiveOut=%p"
      ,
      countSamples,
      static_cast<const void *>(featureValues),
      countDesiredCuts,
      static_cast<void *>(cutsLowerBoundInclusiveOut)
   );

   if(UNLIKELY(countDesiredCuts <= IntEbmType { 0 })) {
      if(UNLIKELY(countDesiredCuts < IntEbmType { 0 })) {
         LOG_0(TraceLevelError, "ERROR CutUniform countDesiredCuts can't be negative.");
      } else {
         LOG_COUNTED_0(
            &g_cLogExitCutUniformParametersMessages,
            TraceLevelInfo,
            TraceLevelVerbose,
            "Exited CutUniform: countCuts=0 due to zero countDesiredCuts"
         );
      }
      return IntEbmType { 0 };
   }
   if(UNLIKELY(IsConvertError<size_t>(countDesiredCuts))) {
      LOG_0(TraceLevelError, "ERROR CutUniform IsConvertError<size_t>(countDesiredCuts)");
      return IntEbmType { 0 };
   }
   const size_t cCuts = static_cast<size_t>(countDesiredCuts);
   if(UNLIKELY(IsMultiplyError(sizeof(*cutsLowerBoundInclusiveOut), cCuts))) {
      LOG_0(TraceLevelError, "ERROR CutUniform countDesiredCuts was too large to fit into cutsLowerBoundInclusiveOut");
      return IntEbmType { 0 };
   }

   if(UNLIKELY(countSamples <= IntEbmType { 1 })) {
      if(UNLIKELY(countSamples < IntEbmType { 0 })) {
         LOG_0(TraceLevelError, "ERROR CutUniform countSamples < IntEbmType { 0 }");
      } else {
         LOG_COUNTED_0(
            &g_cLogExitCutUniformParametersMessages,
            TraceLevelInfo,
            TraceLevelVerbose,
            "Exited CutUniform: countCuts=0 because we can't cut 1 sample"
         );
      }
      return IntEbmType { 0 };
   }
   if(UNLIKELY(IsConvertError<size_t>(countSamples))) {
      LOG_0(TraceLevelError, "ERROR CutUniform IsConvertError<size_t>(countSamples)");
      return IntEbmType { 0 };
   }
   const size_t cSamples = static_cast<size_t>(countSamples);
   if(UNLIKELY(IsMultiplyError(sizeof(*featureValues), cSamples))) {
      LOG_0(TraceLevelError, "ERROR CutUniform countSamples was too large to fit into featureValues");
      return IntEbmType { 0 };
   }

   if(UNLIKELY(nullptr == featureValues)) {
      LOG_0(TraceLevelError, "ERROR CutUniform nullptr == featureValues");
      return IntEbmType { 0 };
   }

   FloatEbmType minValue = std::numeric_limits<FloatEbmType>::infinity();
   FloatEbmType maxValue = -std::numeric_limits<FloatEbmType>::infinity();

   const FloatEbmType * pValue = featureValues;
   const FloatEbmType * const pValuesEnd = featureValues + cSamples;
   do {
      const FloatEbmType val = *pValue;
      maxValue = UNPREDICTABLE(maxValue < val) ? val : maxValue; // this works for NaN values which evals to false
      minValue = UNPREDICTABLE(val < minValue) ? val : minValue; // this works for NaN values which evals to false
      ++pValue;
   } while(LIKELY(pValuesEnd != pValue));

   EBM_ASSERT(!std::isnan(minValue));
   EBM_ASSERT(!std::isnan(maxValue));

   if(UNLIKELY(std::numeric_limits<FloatEbmType>::infinity() == minValue)) {
      if(PREDICTABLE(-std::numeric_limits<FloatEbmType>::infinity() == maxValue)) {
         LOG_COUNTED_0(
            &g_cLogExitCutUniformParametersMessages,
            TraceLevelInfo,
            TraceLevelVerbose,
            "Exited CutUniform: countCuts=0 due to all feature values being missing"
         );
      } else {
         EBM_ASSERT(std::numeric_limits<FloatEbmType>::infinity() == maxValue);
         LOG_COUNTED_0(
            &g_cLogExitCutUniformParametersMessages,
            TraceLevelInfo,
            TraceLevelVerbose,
            "Exited CutUniform: countCuts=0 due to all feature values being +infinity"
         );
      }
      return IntEbmType { 0 };
   }

   if(UNLIKELY(-std::numeric_limits<FloatEbmType>::infinity() == maxValue)) {
      EBM_ASSERT(-std::numeric_limits<FloatEbmType>::infinity() == minValue);

      LOG_COUNTED_0(
         &g_cLogExitCutUniformParametersMessages,
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited CutUniform: countCuts=0 due to all feature values being -infinity"
      );
      return IntEbmType { 0 };
   }

   if(PREDICTABLE(-std::numeric_limits<FloatEbmType>::infinity() == minValue)) {
      minValue = std::numeric_limits<FloatEbmType>::lowest();
   }

   if(PREDICTABLE(std::numeric_limits<FloatEbmType>::infinity() == maxValue)) {
      maxValue = std::numeric_limits<FloatEbmType>::max();
   }

   // make it zero if our caller gave us a subnormal
   minValue = Denormalize(minValue);
   maxValue = Denormalize(maxValue);

   if(UNLIKELY(minValue == maxValue)) {
      LOG_COUNTED_0(
         &g_cLogExitCutUniformParametersMessages,
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited CutUniform: countCuts=0 due to there being only 1 unique value"
      );
      return IntEbmType { 0 };
   }
   EBM_ASSERT(minValue < maxValue);

   if(UNLIKELY(nullptr == cutsLowerBoundInclusiveOut)) {
      // if we have a potential bin cut, then cutsLowerBoundInclusiveOut shouldn't be nullptr
      LOG_0(TraceLevelError, "ERROR CutUniform nullptr == cutsLowerBoundInclusiveOut");
      return IntEbmType { 0 };
   }

   FloatEbmType walkValue = minValue;
   size_t iCut = 0;
   do {
      EBM_ASSERT(walkValue < maxValue);
      walkValue = TickHigher(walkValue);
      cutsLowerBoundInclusiveOut[iCut] = walkValue;
      ++iCut; // increment here so that we return the right # of items in the if statement below
      if(UNLIKELY(walkValue == maxValue)) {
         const IntEbmType countCutsRet = static_cast<IntEbmType>(iCut);
         EBM_ASSERT(countCutsRet <= countDesiredCuts);

         LOG_COUNTED_N(
            &g_cLogExitCutUniformParametersMessages,
            TraceLevelInfo,
            TraceLevelVerbose,
            "Exited CutUniform: "
            "countCuts=%" IntEbmTypePrintf
            ,
            countCutsRet
         );

         return countCutsRet;
      }
   } while(LIKELY(cCuts != iCut));

   EBM_ASSERT(walkValue < maxValue);

   // at this point we can guarantee that we can return countDesiredCuts items since we were able to 
   // fill that many cuts into the buffer.  We could return at any time now with a legal representation
   // if we found ourselves in a situation where we needed to

   // we checked that cCuts can be multiplied with sizeof(*cutsLowerBoundInclusiveOut), and since
   // there is no way an element of cutsLowerBoundInclusiveOut is as small as 1 byte, we should
   // be able to add one to cCuts
   EBM_ASSERT(!IsAddError(size_t { 1 }, cCuts));
   // don't take the reciprocal here because dividing below will be more accurate when rounding
   const size_t cBins = cCuts + size_t { 1 };
   const FloatEbmType cBinsFloat = static_cast<FloatEbmType>(cBins);

   // TODO: if the numbers are really really small in the 10e-308 range then we can have one of our cuts land
   // in the subnormal range which is a really huge gap if we disallow denormals, which we certainly want to do,
   // There's no perfect solution, although our solution below to potentially bunch up many of the cuts on one side
   // is not the best solution.  We could potentially try to figure out how many cuts should go on either side
   // of the subnormal gap and then allocate them proportionally with cuts at -min and +min to wrap the gap, but
   // that's a lot of complicated code to handle what should be an exceedingly rare scenario.

   const FloatEbmType diff = maxValue - minValue;
   if(UNLIKELY(std::numeric_limits<FloatEbmType>::infinity() == diff)) {
      EBM_ASSERT(2.0 <= cBinsFloat);
      const FloatEbmType stepValue = maxValue / cBinsFloat - minValue / cBinsFloat;

      // +inf the the highest non-NaN value, and it has the least significant bit set. The max float has zero
      // in the least significant bit, so if we divide by 2.0, then we don't need to round.  If cBinsFloat is 3.0 or 
      // above then we won't overflow, so in all conditions, even if IEEE-754 rounding is set to something weird like
      // roundTiesToAway we won't overflow when computing the stepValue.  This behavior was tested as well.
      EBM_ASSERT(!std::isinf(stepValue));

      // the first cut should always succeed, since we subtract from maxValue, so even if 
      // the subtraction is zero we succeed in having a cut at maxValue, which is legal for us
      FloatEbmType cutPrev = std::numeric_limits<FloatEbmType>::infinity();
      EBM_ASSERT(cCuts == iCut);
      do {
         FloatEbmType cut;
         const size_t iReversed = cBins - iCut;
         
         // if we go beyond the mid-point index, then (i * stepValue) can overflow, so we need to swap our anchor
         if(PREDICTABLE(iCut < iReversed)) {
            // always subtract from maxValue on the first iteration so that
            // cutPrev has a guarantee that it's maxValue or lower
            EBM_ASSERT(2 <= iReversed);
            cut = minValue + static_cast<FloatEbmType>(iCut) * stepValue;
         } else {
            cut = maxValue - static_cast<FloatEbmType>(iReversed) * stepValue;
         }
         EBM_ASSERT(!std::isinf(cut));

         cut = Denormalize(cut);

         if(UNLIKELY(cutPrev <= cut)) {
            EBM_ASSERT(std::numeric_limits<FloatEbmType>::infinity() != cutPrev);

            // if we didn't advance, then don't put the same cut into the result, advance by one tick
            cut = TickLower(cutPrev);
         }

         if(UNLIKELY(cut <= cutsLowerBoundInclusiveOut[iCut - 1])) {
            // if this happens, the rest of our cuts will be at tick increments, which we've already filled in
            break;
         }
         cutsLowerBoundInclusiveOut[iCut - 1] = cut;

         cutPrev = cut;
         --iCut;
      } while(LIKELY(size_t { 0 } != iCut));
   } else if(PREDICTABLE(std::abs(minValue) < std::abs(maxValue))) {
      // minValue has more precision, so anchor to that point

      // We're going to fill the cut buffer upwards from the minValue to the maxValue.  If we find that 
      // we hit the boundary where the rest of the cuts need to be in tick increments then we want 
      // it to be in the less precise range near the maxValue, so let's re-fill our buffer again starting 
      // from the maxValue and work downwards by ticks so that we can bail with valid results if needed

      EBM_ASSERT(cCuts == iCut);
      walkValue = maxValue;
      do {
         EBM_ASSERT(minValue < walkValue);
         cutsLowerBoundInclusiveOut[iCut - 1] = walkValue;
         walkValue = TickLower(walkValue);
         --iCut;
      } while(LIKELY(size_t { 0 } != iCut));

      // if they are equal then we should have exited above when filling the buffer initially
      EBM_ASSERT(minValue < walkValue);

      FloatEbmType cutPrev = minValue;
      EBM_ASSERT(0 == iCut);
      do {
         ++iCut;

         FloatEbmType cut = minValue + static_cast<FloatEbmType>(iCut) / cBinsFloat * diff;
         cut = Denormalize(cut);

         if(UNLIKELY(cut <= cutPrev)) {
            // if we didn't advance, then don't put the same cut into the result, advance by one tick
            cut = TickHigher(cutPrev);
         }

         if(UNLIKELY(cutsLowerBoundInclusiveOut[iCut - 1] <= cut)) {
            // if this happens, the rest of our cuts will be at tick increments, which we've already filled in
            break;
         }
         cutsLowerBoundInclusiveOut[iCut - 1] = cut;

         cutPrev = cut;
      } while(LIKELY(cCuts != iCut));
   } else {
      // maxValue has more precision, so anchor to that point

      // the first cut should always succeed, since we subtract from maxValue, so even if 
      // the subtraction is zero we succeed in having a cut at maxValue, which is legal for us
      FloatEbmType cutPrev = std::numeric_limits<FloatEbmType>::infinity();
      EBM_ASSERT(cCuts == iCut);
      do {
         const size_t iReversed = cBins - iCut;

         FloatEbmType cut = maxValue - static_cast<FloatEbmType>(iReversed) / cBinsFloat * diff;
         cut = Denormalize(cut);

         if(UNLIKELY(cutPrev <= cut)) {
            EBM_ASSERT(std::numeric_limits<FloatEbmType>::infinity() != cutPrev);

            // if we didn't advance, then don't put the same cut into the result, advance by one tick
            cut = TickLower(cutPrev);
         }

         if(UNLIKELY(cut <= cutsLowerBoundInclusiveOut[iCut - 1])) {
            // if this happens, the rest of our cuts will be at tick increments, which we've already filled in
            break;
         }
         cutsLowerBoundInclusiveOut[iCut - 1] = cut;

         cutPrev = cut;
         --iCut;
      } while(LIKELY(size_t { 0 } != iCut));
   }

   LOG_COUNTED_N(
      &g_cLogExitCutUniformParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Exited CutUniform: "
      "countCuts=%" IntEbmTypePrintf
      ,
      countDesiredCuts
   );

   return countDesiredCuts;
}

} // DEFINED_ZONE_NAME
