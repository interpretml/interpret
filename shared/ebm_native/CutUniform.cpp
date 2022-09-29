// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // std::numeric_limits

#include "ebm_native.h" // EBM_API_BODY
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // LIKELY
#include "zones.h"

#include "common_cpp.hpp" // IsConvertError

#include "ebm_internal.hpp" // CleanFloat, FloatTickIncrement

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// General float info: 
//   https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
//   https://randomascii.wordpress.com/2017/06/19/sometimes-floating-point-math-is-perfect/
//   https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
//
// This is a cross-language portable implementation of nextafter that skips over subnormals per the EBM spec.
// 
// FloatTickIncrementInternal and FloatTickDecrementInternal will return correct and reproducible results even if:
//   - a subnormal input is given. We treat subnormals as equal to zero per the EBM spec
//   - a subnormal is the next tick. We return zero in this case since the EBM spec skips over subnormals
//   - "flush-to-zero" or "denormals-are-zero" ("subnormals-are-zero") is enabled on the CPU
//   - a negative zero input is given
//   - any of the allowed IEEE-754 rounding modes are chosen, since we do not round any values in these functions
//   - we do not require that the CPU/compiler have correct rounding since we start from numbers that have
//     exact representations in IEEE-754 and our operations generate new numbers that also have exact representations
//   - our function would handle NaN, infinity inputs, and infinity outputs, but we don't ever pass these
//     values into these functions in our program, so we do not need to make any guarantees on these
//   - if the environment supports extended precision, we force the environment to round the value to its chosen
//     non-extended precision value by passing inputs in via arrays which cannot contain the extra bits
//     In C++ on Intel we also set compiler flags to only use the SSE2 registers which do not have extended 
//     precision, unlike x87.  ARM does not use extended precision, AFAIK.
//   - FloatTickIncrementInternal and FloatTickDecrementInternal do not make use of multiply and add semantics, 
//     so any fused CPU instructions that use correct rounding on the final result are not used
//
// FloatTickIncrementInternal and FloatTickDecrementInternal have limited requirements to work properly.  We require 
// that the numbers have IEEE-754 format, which has been the case for all common CPUs released since the 1980s.  
// We require that the numbers in this file are converted to their exact IEEE-754 representation, which is a 
// requirement by the IEEE-754 standard when numbers are provided with 17 digits of precision, which we provide.

static constexpr double k_minNonSubnormal = 2.2250738585072014e-308; // std::numeric_limits<double>::min()
static constexpr double k_maxNonInf = 1.7976931348623158e+308; // std::numeric_limits<double>::max()
static constexpr double k_nonDoubleable = 8.9884656743115795e+307; // if you double this power of two you get +inf
static constexpr double k_lastTick = 1.9958403095347198e+292; // the tick for numbers greater or equal to k_nonDoubleable
static constexpr double k_epsilon = 2.2204460492503131e-16; // std::numeric_limits<double>::epsilon()
static constexpr double k_subnormToNorm = 4503599627370496.0; // multiplying by this will move a subnormal into a normal

static_assert(k_subnormToNorm == std::numeric_limits<double>::min() / std::numeric_limits<double>::denorm_min(),
   "bad min to denorm_min ratio");
static_assert(k_epsilon == std::numeric_limits<double>::epsilon(), "bad k_epsilon");
static_assert(k_maxNonInf == std::numeric_limits<double>::max(), "bad k_maxNonInf");
static_assert(k_minNonSubnormal == std::numeric_limits<double>::min(), "bad k_minNonSubnormal");
static_assert(k_nonDoubleable + k_lastTick != k_nonDoubleable, "bad k_lastTick");
static_assert(k_nonDoubleable + k_lastTick / 2 == k_nonDoubleable, "bad k_lastTick");
static_assert(k_nonDoubleable - k_lastTick / 2 != k_nonDoubleable, "bad k_lastTick");
static_assert((k_nonDoubleable - k_lastTick / 2) * 2 <= std::numeric_limits<double>::max(), "bad k_nonDoubleable");
//static_assert(std::numeric_limits<double>::max() < k_nonDoubleable * 2, "bad k_nonDoubleable");

extern double FloatTickIncrementInternal(double deprecisioned[1]) noexcept {
   // This is a cross-language portable implementation of nextafter

   double bound;
   double tick;

   // We pass values in as an array in order to force the elimination of any possible extended precision bits.
   double val = deprecisioned[0];

   // we would handle all of these special values, but we don't use them anywhere, so 
   // simplify our cross-platform port testing by eliminating them as valid inputs
   EBM_ASSERT(!std::isnan(val));
   EBM_ASSERT(!std::isinf(val));
   EBM_ASSERT(std::numeric_limits<double>::max() != val);

   if(val < k_minNonSubnormal) {
      if(-1.0 <= val) {
         if(-k_minNonSubnormal <= val) {
            if(-k_minNonSubnormal < val) {
               return k_minNonSubnormal;
            } else {
               return 0.0;
            }
         }
         // We cannot let tick enter the subnormal range since subnormals are spaced differently from normal floats.
         // By shifting upwards through a multiple we can maintain the same spacing for the epsilons
         
         val *= k_subnormToNorm;
         bound = -1.0 * k_subnormToNorm * 0.5;
         tick = k_epsilon * k_subnormToNorm * 0.5;
         while(bound <= val) {
            bound *= 0.5;
            tick *= 0.5;
            EBM_ASSERT(std::numeric_limits<double>::min() <= tick);
         }
         return (val + tick) / k_subnormToNorm;
      } else {
         if(val < -k_nonDoubleable) {
            // avoid hitting -infinity high by checking for the last doubling
            return val + k_lastTick;
         }
         bound = -2.0;
         tick = k_epsilon;
         while(val < bound) {
            bound *= 2.0;
            tick *= 2.0;
            EBM_ASSERT(!std::isinf(tick));
         }
         return val + tick;
      }
   } else {
      if(val < 1.0) {
         // We cannot let tick enter the subnormal range since subnormals are spaced differently from normal floats.
         // By shifting upwards through a multiple we can maintain the same spacing for the epsilons

         val *= k_subnormToNorm;
         bound = 1.0 * k_subnormToNorm * 0.5;
         tick = k_epsilon * k_subnormToNorm * 0.5;
         while(val < bound) {
            bound *= 0.5;
            tick *= 0.5;
            EBM_ASSERT(std::numeric_limits<double>::min() <= tick);
         }
         return (val + tick) / k_subnormToNorm;
      } else {
         if(k_nonDoubleable <= val) {
            // avoid hitting +infinity high by checking for the last doubling
            // also, if val is +inf we can't exit the loop below without this check
            return val + k_lastTick;
         }
         bound = 2.0;
         tick = k_epsilon;
         while(bound <= val) {
            bound *= 2.0;
            tick *= 2.0;
            EBM_ASSERT(!std::isinf(tick));
         }
         return val + tick;
      }
   }
}

extern double FloatTickDecrementInternal(double deprecisioned[1]) noexcept {
   // This is a cross-language portable implementation of nextafter

   double bound;
   double tick;

   // We pass values in as an array in order to force the elimination of any possible extended precision bits.
   double val = deprecisioned[0];

   // we would handle all of these special values, but we don't use them anywhere, so 
   // simplify our cross-platform port testing by eliminating them as valid inputs
   EBM_ASSERT(!std::isnan(val));
   EBM_ASSERT(!std::isinf(val));
   EBM_ASSERT(std::numeric_limits<double>::lowest() != val);

   if(-k_minNonSubnormal < val) {
      if(val <= 1.0) {
         if(val <= k_minNonSubnormal) {
            if(val < k_minNonSubnormal) {
               return -k_minNonSubnormal;
            } else {
               return 0.0;
            }
         }
         // We cannot let tick enter the subnormal range since subnormals are spaced differently from normal floats.
         // By shifting upwards through a multiple we can maintain the same spacing for the epsilons

         val *= k_subnormToNorm;
         bound = 1.0 * k_subnormToNorm * 0.5;
         tick = k_epsilon * k_subnormToNorm * 0.5;
         while(val <= bound) {
            bound *= 0.5;
            tick *= 0.5;
            EBM_ASSERT(std::numeric_limits<double>::min() <= tick);
         }
         return (val - tick) / k_subnormToNorm;
      } else {
         if(k_nonDoubleable < val) {
            // avoid hitting +infinity high by checking for the last doubling
            return val - k_lastTick;
         }
         bound = 2.0;
         tick = k_epsilon;
         while(bound < val) {
            bound *= 2.0;
            tick *= 2.0;
            EBM_ASSERT(!std::isinf(tick));
         }
         return val - tick;
      }
   } else {
      if(-1.0 < val) {
         // We cannot let tick enter the subnormal range since subnormals are spaced differently from normal floats.
         // By shifting upwards through a multiple we can maintain the same spacing for the epsilons

         val *= k_subnormToNorm;
         bound = -1.0 * k_subnormToNorm * 0.5;
         tick = k_epsilon * k_subnormToNorm * 0.5;
         while(bound < val) {
            bound *= 0.5;
            tick *= 0.5;
            EBM_ASSERT(std::numeric_limits<double>::min() <= tick);
         }
         return (val - tick) / k_subnormToNorm;
      } else {
         if(val <= -k_nonDoubleable) {
            // avoid hitting -infinity high by checking for the last doubling
            // also, if our input was -inf and we allowed it, this would avoid an infinite loop
            return val - k_lastTick;
         }
         bound = -2.0;
         tick = k_epsilon;
         while(val <= bound) {
            bound *= 2.0;
            tick *= 2.0;
            EBM_ASSERT(!std::isinf(tick));
         }
         return val - tick;
      }
   }
}

#ifdef UNDEFINED_TEST_TICK_HIGHER

TEST_CASE("FloatTickIncrementInternal and FloatTickDecrementInternal") {
   double deprecisioned[1];

   deprecisioned[0] = std::numeric_limits<double>::lowest();
   CHECK(nextafter(std::numeric_limits<double>::lowest(), 0.0) == FloatTickIncrementInternal(deprecisioned));
   deprecisioned[0] = nextafter(std::numeric_limits<double>::lowest(), 0.0);
   CHECK(nextafter(nextafter(std::numeric_limits<double>::lowest(), 0.0), 0.0) == FloatTickIncrementInternal(deprecisioned));

   deprecisioned[0] = nextafter(nextafter(-std::numeric_limits<double>::min(), std::numeric_limits<double>::lowest()), std::numeric_limits<double>::lowest());
   CHECK(nextafter(-std::numeric_limits<double>::min(), std::numeric_limits<double>::lowest()) == FloatTickIncrementInternal(deprecisioned));
   deprecisioned[0] = nextafter(-std::numeric_limits<double>::min(), std::numeric_limits<double>::lowest());
   CHECK(-std::numeric_limits<double>::min() == FloatTickIncrementInternal(deprecisioned));
   deprecisioned[0] = -std::numeric_limits<double>::min();
   CHECK(0.0 == FloatTickIncrementInternal(deprecisioned));
   deprecisioned[0] = -DBL_TRUE_MIN;
   CHECK(std::numeric_limits<double>::min() == FloatTickIncrementInternal(deprecisioned));
   deprecisioned[0] = 0.0;
   CHECK(std::numeric_limits<double>::min() == FloatTickIncrementInternal(deprecisioned));
   deprecisioned[0] = DBL_TRUE_MIN;
   CHECK(std::numeric_limits<double>::min() == FloatTickIncrementInternal(deprecisioned));
   deprecisioned[0] = std::numeric_limits<double>::min();
   CHECK(nextafter(std::numeric_limits<double>::min(), std::numeric_limits<double>::max()) == FloatTickIncrementInternal(deprecisioned));
   deprecisioned[0] = nextafter(std::numeric_limits<double>::min(), std::numeric_limits<double>::max());
   CHECK(nextafter(nextafter(std::numeric_limits<double>::min(), std::numeric_limits<double>::max()), std::numeric_limits<double>::max()) == FloatTickIncrementInternal(deprecisioned));

   deprecisioned[0] = nextafter(std::numeric_limits<double>::max(), 0.0);
   CHECK(std::numeric_limits<double>::max() == FloatTickIncrementInternal(deprecisioned));



   deprecisioned[0] = nextafter(std::numeric_limits<double>::lowest(), 0.0);
   CHECK(std::numeric_limits<double>::lowest() == FloatTickDecrementInternal(deprecisioned));

   deprecisioned[0] = nextafter(-std::numeric_limits<double>::min(), std::numeric_limits<double>::lowest());
   CHECK(nextafter(nextafter(-std::numeric_limits<double>::min(), std::numeric_limits<double>::lowest()), std::numeric_limits<double>::lowest()) == FloatTickDecrementInternal(deprecisioned));
   deprecisioned[0] = -std::numeric_limits<double>::min();
   CHECK(nextafter(-std::numeric_limits<double>::min(), std::numeric_limits<double>::lowest()) == FloatTickDecrementInternal(deprecisioned));
   deprecisioned[0] = -DBL_TRUE_MIN;
   CHECK(-std::numeric_limits<double>::min() == FloatTickDecrementInternal(deprecisioned));
   deprecisioned[0] = 0.0;
   CHECK(-std::numeric_limits<double>::min() == FloatTickDecrementInternal(deprecisioned));
   deprecisioned[0] = DBL_TRUE_MIN;
   CHECK(-std::numeric_limits<double>::min() == FloatTickDecrementInternal(deprecisioned));
   deprecisioned[0] = std::numeric_limits<double>::min();
   CHECK(0.0 == FloatTickDecrementInternal(deprecisioned));
   deprecisioned[0] = nextafter(std::numeric_limits<double>::min(), std::numeric_limits<double>::max());
   CHECK(std::numeric_limits<double>::min() == FloatTickDecrementInternal(deprecisioned));
   deprecisioned[0] = nextafter(nextafter(std::numeric_limits<double>::min(), std::numeric_limits<double>::max()), std::numeric_limits<double>::max());
   CHECK(nextafter(std::numeric_limits<double>::min(), std::numeric_limits<double>::max()) == FloatTickDecrementInternal(deprecisioned));

   deprecisioned[0] = nextafter(std::numeric_limits<double>::max(), 0.0);
   CHECK(nextafter(nextafter(std::numeric_limits<double>::max(), 0.0), 0.0) == FloatTickDecrementInternal(deprecisioned));
   deprecisioned[0] = std::numeric_limits<double>::max();
   CHECK(nextafter(std::numeric_limits<double>::max(), 0.0) == FloatTickDecrementInternal(deprecisioned));

   double base = 2 * std::numeric_limits<double>::min();
   do {
      double sweep = base;

      for(int i = 0; i < 5; ++i) {
         sweep = nextafter(sweep, std::numeric_limits<double>::lowest());
      }

      for(int i = 0; i < 11; ++i) {
         deprecisioned[0] = sweep;
         double test = FloatTickIncrementInternal(deprecisioned);
         sweep = nextafter(sweep, std::numeric_limits<double>::max());
         CHECK(sweep == test);
      }

      for(int i = 0; i < 11; ++i) {
         deprecisioned[0] = sweep;
         double test = FloatTickDecrementInternal(deprecisioned);
         sweep = nextafter(sweep, std::numeric_limits<double>::lowest());
         CHECK(sweep == test);
      }

      base *= 2.0;
   } while(!std::isinf(base));
}

#endif // UNDEFINED_TEST_TICK_HIGHER

EBM_API_BODY void EBM_CALLING_CONVENTION CleanFloats(IntEbm count, double * valsInOut) {
   // this function converts extended precision values, if they are present, into non-extended precision values by 
   // virtue of passing them in via arrays stored in memory, and it also converts subnormal values into zero, 
   // per EBM spec requirements.  It also converts negative zeros into zeros.

   if(IsConvertError<size_t>(count)) {
      LOG_0(Trace_Error, "ERROR CleanFloats count is not a valid index into an array");
      return;
   }
   size_t c = static_cast<size_t>(count);
   if(IsMultiplyError(sizeof(*valsInOut), c)) {
      LOG_0(Trace_Error, "ERROR CleanFloats count value too large to index into memory");
      return;
   }
   while(0 != c) {
      --c;
      const double val = valsInOut[c];
      // Use this check for NaN for cross-language portability.  It is not technically needed 
      // if IEEE-754 is followed, since "anything < NaN" and "NaN < anything" are false
      if(!std::isnan(val)) {
         // Don't use the trick of adding and subtracting 1.0020841800044864e-292 since the results could
         // be inconsitent across platforms if subnormals are rounded or not
         if(-k_minNonSubnormal < val && val < k_minNonSubnormal) {
            // val is in the subnormal range, so force it to zero.  
            // DO NOT COMPARE WITH ZERO SINCE SOME CPUs INDICATE TRUE WHEN COMPARING 0.0 TO A SUBNORMAL
            // if the environment violates IEEE-754 with "denormals-are-zero" ("subnormals-are-zero")
            valsInOut[c] = 0.0;
         }
      }
   }
}

EBM_API_BODY IntEbm EBM_CALLING_CONVENTION CutUniform(
   IntEbm countSamples,
   const double * featureVals,
   IntEbm countDesiredCuts,
   double * cutsLowerBoundInclusiveOut
) {
   // DO NOT CHANGE THIS FUNCTION'S ALGORITHM.  IT IS PART OF THE EBM HISTOGRAM SPEC
   //
   // This function is also used when choosing histograms cuts. Since we don't store the histogram 
   // cuts in our JSON/model, this function must always return the same results in all implementations.
   //
   // This function guarantees that it will return countDesiredCuts unless it is impossible to put enough cuts between
   // the min and max values, and in that case it will put a cut into every possible location allowed by floats.
   // This functionality is required because if we have a given number of histogram counts, then we need
   // to return the same number of expected cuts, so returning the maximum number of cuts allowed avoids
   // issues because our caller would have to throw an exception if they got back less cuts than expected.
   // 
   // When discretizing, we include numbers equal to the cut in the bin because we want
   // a cut value of 1.0 to include 1.0 in the same bin as 1.1, and 1.999999... (if the next cut is at 2.0)
   //
   // This means we never return the min value in our cuts since the min value will be separated away by the next 
   // highest floating point number.
   //
   // This also means that we can return a cut at the max value since a cut there will keep
   // the max value in the highest bin, so we do have an asymmetry that we fundamentally cannot  
   // avoid since we need to make a choice whether exact matches fall into the lower or upper bins.
   //
   // Since our cuts can include the max value, this means that we cannot separate +highest from +inf values without 
   // allowing an +inf cut value, which we do not allow in the EBM spec. So, +inf and +highest are indistinguishable
   // numbers for an EBM.
   //
   // For symmetry with the +inf case, we do not allow EBMs to separate -inf from the lowest float.  The lowest 
   // legal cut is therefore one float tick higher than the lowest float.
   //
   // For histograms edges, which includes the min and max value, our caller will add the min and max values 
   // to the returned cuts. We do allow -inf and +inf values there, so we still preserve the best information 
   // when graphing and serializing.
   //
   // We do not allow cut points to be subnormal floats. Some environments still have issues dealing 
   // with them reproducibly, whereas setting them to zero should work in all environments and be reproducible.
   // They might also not serialize/deserialize to JSON in all languages/environments.  Since we
   // set subnormals to zero we cannot separate subnormals from each other, but having data that should 
   // predict different outcomes in the subnormal range should be really rare, and any caller can rectify this
   // anyways by shifting the subnormals into the normal space themselves.
   // 
   // This function should be able to return correct and reproducible results in all environments that I am 
   // aware of. In order for us to return reproducible results we require:
   //   - That the numbers are stored in IEEE-754 format, which has been the case for all common CPUs since the 1980s.
   //   - The numbers in this file are converted into their exact IEEE-754 representation, which is a requirement 
   //     of the original IEEE-754 spec for numbers given with 17 digits of precision. We always provide 17 digits 
   //     here, unless a number has an exact float representation (eg: 2.0 or 0.5).
   //   - We require that we can eliminate any extended precision digits. Generally, this can be done by storing
   //     the values in memory, which we do, since then they need to have defined sizes.  The only processor I am 
   //     aware of that uses extended precision is the original x87 floating point unit, but you can now specify 
   //     for most compilers, which we have done, to use the SSE2 registers, which do not use extended precision. SSE2 
   //     registers are always available in 64 bit mode, and have been available since the year 2000 in 32 bit mode.
   //   - We require correct rounding for addition, subtraction, division and multiplication.  The original IEEE-754
   //     required correct rounding for these operations, and I'm not aware of an implementation that violates this.
   //   - We require that the IEEE-754 rounding mode is set to "Round to nearest, ties to even".  This is the default
   //     IEEE-754 rounding mode, and I'm not aware of a programming language environment that sets it to anything 
   //     other than that by default.  Mostly you would need to set a flag on the CPU to get anything else.
   //     If another IEEE-754 rounding mode is selected, then we can be off by 1 float tick on cuts.
   //     We still guarantee the same number of cuts will be returned.
   //   - we can handle environments where "flush-to-zero" or "denormals-are-zero" ("subnormals-are-zero")
   //     is enabled on the CPU.  We do this by flushing intermediate and final results to zero ourselves.
   //   - we treat negative zero as a zero
   //   - we use NaN values to indicate missing values, but we don't use NaN for computation.  If the environment
   //     noes not support NaN values, then there cannot be missing values, but everything else will work properly
   //     If the language does not support NaN, but floats can still have NaN values internally, then we
   //     handle that case correctly PROVIDED NaN comparison rules are observed.
   //   - some CPUs have fused multiply and add instructions which also use correct rounding on the final
   //     result, which can differ from CPUs that do these operations independently and round the intermediate 
   //     result. We force conformance by breaking the multiplication from the add with a function that 
   //     also writes the results to memory
   // If these are violated, then the results cannot be guaranteed to be reproducible, but even then they should
   // almost always be reproducible except in outlier cases. We do guarantee that we return the same 
   // number of cuts in all cases, and that we will fill all legal cuts when the difference between the min and 
   // max value is insufficient to have enough cuts.
   //
   // If the distance between cuts is a subnormal number, then we can get bunching if some of the cuts fall into
   // the subnormal gap around 0. The underlying issue is that fundamentally floats are approximate things and we 
   // cannot have true uniform cuts. We could make this algorithm a little better, but IMHO it isn't worth the added
   // complexity since we want this code to be cross-language portable, and this only affects extremely tiny numbers 
   // in the range of 10^-308.  Numbers that small are really skirting close to being zero anyways.


   double valMin;
   double valMax;
   size_t iSample;
   double val;
   double walkVal;
   size_t iCut;
   double multiple;
   double halfDiff;
   double cBinsFloat;
   size_t cHalfCuts;
   double cutPrev;

   double numerator;
   double fraction;
   double shift;
   double cutExpanded;
   double cut;

   if(0 == countDesiredCuts) {
      // we need this check because otherwise we could overflow to +inf when we calculate halfDiff below
      // if both valMin and valMax are both very big (close to infinity) and opposite in sign
      return 0;
   }
   if(IsConvertError<size_t>(countDesiredCuts)) {
      LOG_0(Trace_Error, "ERROR CutUniform countDesiredCuts is not a valid index into an array");
      return 0;
   }
   const size_t cCuts = static_cast<size_t>(countDesiredCuts);
   if(IsMultiplyError(sizeof(*cutsLowerBoundInclusiveOut), cCuts)) {
      LOG_0(Trace_Error, "ERROR CutUniform countDesiredCuts value too large to index into memory");
      return 0;
   }

   if(IsConvertError<size_t>(countSamples)) {
      LOG_0(Trace_Error, "ERROR CutUniform countSamples is not a valid index into an array");
      return 0;
   }
   const size_t cSamples = static_cast<size_t>(countSamples);
   if(IsMultiplyError(sizeof(*featureVals), cSamples)) {
      LOG_0(Trace_Error, "ERROR CutUniform countSamples value too large to index into memory");
      return 0;
   }

   if(nullptr == featureVals) {
      LOG_0(Trace_Error, "ERROR CutUniform featureVals cannot be NULL");
      return 0;
   }

   valMin = k_maxNonInf;
   valMax = -k_maxNonInf;

   for(iSample = 0; iSample < cSamples; ++iSample) {
      val = featureVals[iSample];
      // Use this check for NaN for cross-language portability.  It is not technically needed 
      // if IEEE-754 is followed, since "NaN < anything" and "anything < NaN" are false
      if(!std::isnan(val)) {
         if(valMax < val) {
            // this works for NaN values which evals to false
            valMax = val;
         }
         if(val < valMin) {
            // this works for NaN values which evals to false
            valMin = val;
         }
      }
   }

   EBM_ASSERT(!std::isnan(valMin));
   EBM_ASSERT(!std::isnan(valMax));

   EBM_ASSERT(-std::numeric_limits<double>::infinity() != valMax);
   EBM_ASSERT(std::numeric_limits<double>::infinity() != valMin);

   if(k_maxNonInf == valMin && -k_maxNonInf == valMax) {
      // all features are the missing value
      return 0;
   }

   if(valMin <= -k_maxNonInf) {
      // this is a way to check for -inf without using an -inf value which might be less portable
      valMin = -k_maxNonInf;
   }

   if(k_maxNonInf <= valMax) {
      // this is a way to check for +inf without using an +inf value which might be less portable
      valMax = k_maxNonInf;
   }

   // make it zero if our caller gave us a subnormal
   valMin = CleanFloat(valMin);
   valMax = CleanFloat(valMax);

   if(valMin == valMax) {
      return 0;
   }
   EBM_ASSERT(valMin < valMax);

   if(nullptr == cutsLowerBoundInclusiveOut) {
      LOG_0(Trace_Error, "ERROR CutUniform cutsLowerBoundInclusiveOut cannot be NULL");
      return 0;
   }

   walkVal = valMin;
   for(iCut = 0; cCuts != iCut; ++iCut) {
      EBM_ASSERT(walkVal < valMax);
      walkVal = FloatTickIncrement(walkVal);
      cutsLowerBoundInclusiveOut[iCut] = walkVal;
      if(walkVal == valMax) {
         ++iCut;
         return static_cast<IntEbm>(iCut);
      }
   }
   EBM_ASSERT(walkVal < valMax);

   // at this point we can guarantee that we can return countDesiredCuts items since we were able to 
   // fill that many cuts into the buffer.  We could return at any time now with a legal representation
   // if we found ourselves in a situation where we needed to

   // if the halfDiff value below is in the subnormal range, then we can no longer reliably calculate fractions
   // of that value to place cuts that we could otherwise place in the normal range. We can solve these issues by 
   // shifting upwards.  A simple way to do this is to keep shifting until right before the numbers will overflow

   multiple = 1.0;
   while(valMax < k_nonDoubleable && -k_nonDoubleable < valMin && multiple < k_nonDoubleable) {
      // doubling the valMax and valMin just shifts the exponent without changing the mantissa, which
      // is a lossless float operation that we can do and undo. The difference of valMax - valMin doubles
      // each time we double the operands too, so this is also a lossless operation that only changes the mantissa.
      // The only thing we're affecting by shifting exponents this way would be subnormal numbers which are spaced 
      // differently than normal numbers. This effect on subnormals is desired, and in fact is why we are 
      // performing this loop to achieve more resolution.

      valMax *= 2.0;
      EBM_ASSERT(!std::isinf(valMax));
      valMin *= 2.0;
      EBM_ASSERT(!std::isinf(valMin));
      multiple *= 2.0;
      EBM_ASSERT(!std::isinf(multiple));
   }

   // If valMax had the highest value possible and valMin had the lowest (big negative) value possible,
   // dividing by 2.0 would change the exponent but not change the mantissa of an IEEE-754 value.  Then subtracting 
   // these two negatively identical values would be equivalent to multiplying by 2.0, yielding the highest 
   // possible value again. Since the mantissas will not be changed in these operations, there will be no rounding, 
   // so this result is expected under all possible IEEE-754 rounding modes. If the values were less than their 
   // extreme ends, then there could be rounding, but under correct rounding rules the end result could not be 
   // bigger than our result with larger numbers.  Therefore, this operation cannot overflow to +infinity.

   halfDiff = CleanFloat(valMax / 2.0 - valMin / 2.0);
   EBM_ASSERT(!std::isinf(halfDiff));

   // Don't take the reciprocal here because dividing below will be more accurate when rounding
   // 
   // If cBinsFloat is very very big number (larger than 2^53), then it can reach into the region where individual 
   // integers cannot be identified since the float tick is larger than an integer.  We would want to eliminate
   // any extended precision bits, if there were any. We'd also get float bunching, but meh, it's hard to fix,
   // so avoid having 2^53 cuts.
   const size_t cBinsInt = cCuts + 1;
   cBinsFloat = CleanFloat(static_cast<double>(cBinsInt));

   // make the first cut subtract from valMax so that we do not need to check for forward progress on the first loop
   cHalfCuts = cCuts / 2;
   cutPrev = 0.0;

   iCut = cCuts;
   while(cHalfCuts < iCut) {
      --iCut;
      const size_t numeratorInt = (cCuts - iCut) << 1;
      numerator = CleanFloat(static_cast<double>(numeratorInt)); // clear for > 2^53 cuts
      fraction = CleanFloat(numerator / cBinsFloat); // clear extended precision bits
      EBM_ASSERT(fraction <= 1.0);
      shift = CleanFloat(fraction * halfDiff); // clear extended precision bits
      cutExpanded = CleanFloat(valMax - shift); // stop from using fused CPU instructions
      cut = CleanFloat(cutExpanded / multiple); // zero subnormals

      if(cutPrev <= cut && cCuts != iCut + 1) {
         // Do not tick down from our first cut value. We subtract from valMax, so even if 
         // the subtraction is zero we succeed in having a cut at valMax, which is legal.

         // if we didn't advance, then don't put the same cut into the result, advance by one tick
         cut = FloatTickDecrement(cutPrev);
      }
      if(cut <= cutsLowerBoundInclusiveOut[iCut]) {
         // if this happens, the rest of our cuts will be at tick increments, which we've already filled in
         return countDesiredCuts;
      }
      cutsLowerBoundInclusiveOut[iCut] = cut;
      cutPrev = cut;
   }
   while(0 != iCut) {
      const size_t numeratorInt = iCut << 1;
      numerator = CleanFloat(static_cast<double>(numeratorInt)); // clear for > 2^53 cuts
      fraction = CleanFloat(numerator / cBinsFloat); // clear extended precision bits
      EBM_ASSERT(fraction <= 1.0);
      shift = CleanFloat(fraction * halfDiff); // clear extended precision bits
      cutExpanded = CleanFloat(valMin + shift); // stop from using fused CPU instructions
      cut = CleanFloat(cutExpanded / multiple); // zero subnormals

      --iCut;

      if(cutPrev <= cut) {
         // if we didn't advance, then don't put the same cut into the result, advance by one tick
         cut = FloatTickDecrement(cutPrev);
      }
      if(cut <= cutsLowerBoundInclusiveOut[iCut]) {
         // if this happens, the rest of our cuts will be at tick increments, which we've already filled in
         return countDesiredCuts;
      }
      cutsLowerBoundInclusiveOut[iCut] = cut;
      cutPrev = cut;
   }
   return countDesiredCuts;
}

} // DEFINED_ZONE_NAME
