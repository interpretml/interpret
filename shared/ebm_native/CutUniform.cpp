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
   const double * featureValues,
   IntEbmType countDesiredCuts,
   double * cutsLowerBoundInclusiveOut
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

   double minValue = std::numeric_limits<double>::infinity();
   double maxValue = -std::numeric_limits<double>::infinity();

   const double * pValue = featureValues;
   const double * const pValuesEnd = featureValues + cSamples;
   do {
      const double val = *pValue;
      maxValue = UNPREDICTABLE(maxValue < val) ? val : maxValue; // this works for NaN values which evals to false
      minValue = UNPREDICTABLE(val < minValue) ? val : minValue; // this works for NaN values which evals to false
      ++pValue;
   } while(LIKELY(pValuesEnd != pValue));

   EBM_ASSERT(!std::isnan(minValue));
   EBM_ASSERT(!std::isnan(maxValue));

   if(UNLIKELY(std::numeric_limits<double>::infinity() == minValue)) {
      if(PREDICTABLE(-std::numeric_limits<double>::infinity() == maxValue)) {
         LOG_COUNTED_0(
            &g_cLogExitCutUniformParametersMessages,
            TraceLevelInfo,
            TraceLevelVerbose,
            "Exited CutUniform: countCuts=0 due to all feature values being missing"
         );
      } else {
         EBM_ASSERT(std::numeric_limits<double>::infinity() == maxValue);
         LOG_COUNTED_0(
            &g_cLogExitCutUniformParametersMessages,
            TraceLevelInfo,
            TraceLevelVerbose,
            "Exited CutUniform: countCuts=0 due to all feature values being +infinity"
         );
      }
      return IntEbmType { 0 };
   }

   if(UNLIKELY(-std::numeric_limits<double>::infinity() == maxValue)) {
      EBM_ASSERT(-std::numeric_limits<double>::infinity() == minValue);

      LOG_COUNTED_0(
         &g_cLogExitCutUniformParametersMessages,
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited CutUniform: countCuts=0 due to all feature values being -infinity"
      );
      return IntEbmType { 0 };
   }

   if(PREDICTABLE(-std::numeric_limits<double>::infinity() == minValue)) {
      minValue = std::numeric_limits<double>::lowest();
   }

   if(PREDICTABLE(std::numeric_limits<double>::infinity() == maxValue)) {
      maxValue = std::numeric_limits<double>::max();
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

   double walkValue = minValue;
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
   const double cBinsFloat = static_cast<double>(cBins);

   double diff = maxValue - minValue;
   if(UNLIKELY(std::numeric_limits<double>::infinity() == diff)) {
      EBM_ASSERT(2.0 <= cBinsFloat);
      double stepValue = maxValue / cBinsFloat - minValue / cBinsFloat;

      // if maxValue is max_float and minValue is min_float those values are represented with mantissas of all
      // 1s and exponents one shy of all 1s.  Dividing by 2.0 reduces the exponent, but keeps all the mantissa bits.
      // Then subtracting the two equally negated results is identical to multiplying by 2.0, which gives back the 
      // original values.  No rounding was required, so it doesn't matter which IEEE-754 rounding mode is enabled
      // We never overflow to +inf.
      EBM_ASSERT(!std::isinf(stepValue));
      // stepValue cannot be a subnormal since the range of a float64 is larger than an int64
      EBM_ASSERT(std::numeric_limits<double>::min() <= stepValue);

      // the first cut should always succeed, since we subtract from maxValue, so even if 
      // the subtraction is zero we succeed in having a cut at maxValue, which is legal for us
      double cutPrev = std::numeric_limits<double>::infinity();
      EBM_ASSERT(cCuts == iCut);
      do {
         double cut;
         const size_t iReversed = cBins - iCut;

         // the stepValue multiple cannot be subnormal since stepValue cannot be subnormal, 
         // but call Denormalize anyways to guarantee that fused multiply add instructions are not used

         // if we go beyond the mid-point index, then (i * stepValue) can overflow, so we need to swap our anchor
         if(PREDICTABLE(iCut < iReversed)) {
            // always subtract from maxValue on the first iteration so that
            // cutPrev has a guarantee that it's maxValue or lower
            EBM_ASSERT(2 <= iReversed);

            double shift = static_cast<double>(iCut) * stepValue;
            shift = Denormalize(shift);
            cut = minValue + shift;
         } else {
            double shift = static_cast<double>(iReversed) * stepValue;
            shift = Denormalize(shift);
            cut = maxValue - shift;
         }
         if(std::isinf(cut)) {
            // this can occur very rarely if min is lowest float, max is maximum float, at the mid-point when 
            // iReversed is half the value of cBins, we're multiplying the step value by just enough to reach
            // the max float, but then we round upwards.  In that case we overflow to +inf, although it was just
            // by one float tick.  In this case adjust down one tick to the max float value
            cut = std::numeric_limits<double>::max();
         }

         cut = Denormalize(cut);

         if(UNLIKELY(cutPrev <= cut)) {
            EBM_ASSERT(std::numeric_limits<double>::infinity() != cutPrev);

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
   } else {
      // This algorithm has some not as nice properties around subnormal numbers.  Specifically, when a cut
      // lands in the large subnormal space we flush it to zero, and our ticks are limited to non-subnormal numbers
      // too.  This can lead to bunching on float tick increment boundaries where we have more ability to preserve
      // float resolution.  We might be able to make this situation better through difficult manipulation, but this
      // "spec" algorithm is already complicated enough, and we want it to be portable accross implementations without
      // being even more complicated, so our solution here is simply to accept the undesirable bunching around
      // subnormal numbers. We cannot fundamentally have good uniform cuts near subnormals anyways since the big
      // subnormal gap does exists and we can't avoid that while disallowing subnormals as cuts.  The real solution
      // is for the user to avoid spans in the range of 10e-308.  These should be quite rare in any case.
      // This algorithm as constructed has the benefit of being reproducible in any environment that is IEEE-754
      // compliant, minus subnormal compliance.


      // our shift value below can enter the subnormal range which can cause problems in some environments
      // we can solve these issues by shifting upwards to the highest numbers that do not overflow

      double multipleCur;
      double maxValueCur;
      double minValueCur;
      double diffCur;

      double multipleNext = 1.0;
      double maxValueNext = maxValue;
      double minValueNext = minValue;
      double diffNext = Denormalize(diff);

      do {
         multipleCur = multipleNext;
         maxValueCur = maxValueNext;
         minValueCur = minValueNext;
         diffCur = diffNext;

         multipleNext = multipleCur * 2.0;

         maxValueNext = maxValue * multipleNext;
         if(std::isinf(maxValueNext)) {
            break;
         }
         minValueNext = minValue * multipleNext;
         if(std::isinf(minValueNext)) {
            break;
         }
         diffNext = maxValueNext - minValueNext;
      } while(!std::isinf(diffNext));
      
      if(PREDICTABLE(std::abs(minValue) < std::abs(maxValue))) {
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

         double cutPrev = minValue;
         EBM_ASSERT(0 == iCut);

         do {
            ++iCut;

            // we need to de-subnormalize the shift to get cross machine compatibility since some machines
            // flush subnormals to zero, although this introduces bunching around the start point if
            // that's a very small number close to the denormals, but I'm not sure there is a solution there
            double shift = static_cast<double>(iCut) / cBinsFloat * diffCur;
            shift = Denormalize(shift);
            double cut = minValueCur + shift;
            cut /= multipleCur;
            cut = Denormalize(cut);

            // TODO: do the algorithm below
            //if(0.0 == cut) {
            //   if(0.0 == Denormalize(diffCur / multipleCur / cBinsFloat)) {
            //      // if our individual steps are less than float_min then we know that -min, 0, and +min are all
            //      // going to be cut points provided we're crossing the 0 band gap.  If the steps are greater than
            //      // float_min then we can trust float rounding to decide which of the -min, 0, or +min values
            //      // will be included.  The underflow condition is harder though because we originally spaced
            //      // our ticks believing there wouldn't be a big gap (in the subnormal space), so we need
            //      // to reconsider this.  We can no longer really provide a uniform set of cuts, so the best
            //      // we can do now is put cuts at -min, 0, and +min and then fill in the cuts on each side of the gap.
            //      return HandleSubnormalGap(...);
            //   }
            //}

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
         double cutPrev = std::numeric_limits<double>::infinity();
         EBM_ASSERT(cCuts == iCut);

         do {
            const size_t iReversed = cBins - iCut;

            // we need to de-subnormalize the shift to get cross machine compatibility since some machines
            // flush subnormals to zero, although this introduces bunching around the start point if
            // that's a very small number close to the denormals, but I'm not sure there is a solution there
            double shift = static_cast<double>(iReversed) / cBinsFloat * diffCur;
            shift = Denormalize(shift);
            double cut = maxValueCur - shift;
            cut /= multipleCur;
            cut = Denormalize(cut);

            // TODO: do the algorithm below
            //if(0.0 == cut) {
            //   if(0.0 == Denormalize(diffCur / multipleCur / cBinsFloat)) {
            //      // if our individual steps are less than float_min then we know that -min, 0, and +min are all
            //      // going to be cut points provided we're crossing the 0 band gap.  If the steps are greater than
            //      // float_min then we can trust float rounding to decide which of the -min, 0, or +min values
            //      // will be included.  The underflow condition is harder though because we originally spaced
            //      // our ticks believing there wouldn't be a big gap (in the subnormal space), so we need
            //      // to reconsider this.  We can no longer really provide a uniform set of cuts, so the best
            //      // we can do now is put cuts at -min, 0, and +min and then fill in the cuts on each side of the gap.
            //      return HandleSubnormalGap(...);
            //   }
            //}

            if(UNLIKELY(cutPrev <= cut)) {
               EBM_ASSERT(std::numeric_limits<double>::infinity() != cutPrev);

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
