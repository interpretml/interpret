// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef APPROXIMATE_MATH_HPP
#define APPROXIMATE_MATH_HPP

#include <type_traits> // std::is_same
#include <inttypes.h>
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <cmath> // std::exp, std::log
#include <string.h> // memcpy

#include "logging.h" // EBM_ASSERT
#include "common_c.h" // INLINE_ALWAYS, LIKELY, UNLIKELY
#include "zones.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// TODO: try out floats throughout our program instead of doubles.  It'll be important when we move to GPUs and SIMD
// TODO: do more extensive checks for of AUC and calibration using the benchmark system
// TODO: do the massive multi-dataset addExpSchraudolphTerm/addLogSchraudolphTerm  term optimization using our benchmarks

// START SECTION CORRECT ROUNDING : https://en.wikipedia.org/wiki/IEEE_754#Roundings_to_nearest
// IEEE 754 has a concept called correct rounding that says that certain opterations should be rounded as if they were 
//      computed with infinite precision and then rounded to the nearest bit value.  This is theoretically possible, 
//      except for a limited number of operations like y^w for which there is no known theoretical solution
//      https://en.wikipedia.org/wiki/Rounding#Table-maker's_dilemma
// 
// IEEE 754-1985 -> requires correct rounding to nearest for the following operations:
//   - addition, subtraction, multiplication, division, fused multiply add
// IEEE 754-2008 -> RECOMMENDS (but does not require) correct rounding to nearest for these additional operations:
//   - e^x, ln(x), sqrt(x), and others too, but these are the ones we use
//   - float32 values written as text need to be correctly rounded if they have 9-12 digits, 
//     and 9 digits are required in base 10 representation to differentiate all floating point numbers
//     eg: 1234567.89f -> has 9 digits
//
// In my experience, there is still variation between compilers and many are not completely IEEE-754 compilant,
//   even when they claim to be with regards to correct rounding and other operations.  For instance, 
//   Visual Studio in x86 compilation calls a function to handle rounding of floating points to integers, 
//   and that function returns different results than the Intel instruction for conversion, so the x64 compiles 
//   return different results overall.
//
// It's probably not a good strategy to assume that we can get 100% reproducible results, even though with a fully
//   compilant IEEE-754 compiler we could do so.  These approximate function help a bit in that direction though
//   because we eliminate the variations between compilers with respect to exp and log functions, and we never
//   use the sqrt function because for regression we return the MSE instead of the RMSE. We're still
//   beholden to rounding, but if we include a detection of SSE 4.1 (launched in 2006), then we could use the
//   SSE instructions that convert from float to int, so we could probably get conformance in practice for any
//   machine built since then.  I'd assume ARM using Neon (their SIMD implementation) and other newer processors 
//   would have similar conformance in newer machines independent of their compilers.
//
// END SECTION CORRECT ROUNDING


///////////////////////////////////////////// EXP SECTION

// This paper details our fast exp/log approximation -> "A Fast, Compact Approximation of the Exponential Function"
// by Nicol N. Schraudolph
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.4508&rep=rep1&type=pdf
//
// Here's an implementation from the paper, but includes other options, like float/int32_t versions, and log
// https://github.com/ekmett/approximate/blob/master/cbits/fast.c
//
// different algorithm from Paul Mineiro:
// https://github.com/etheory/fastapprox/blob/master/fastapprox/src/fastexp.h
// Here's a description by Paul Mineiro on how his works
// http://www.machinedlearnings.com/2011/06/fast-approximate-logarithm-exponential.html
// and a Mathematica Notebook where the constants were calculated by Paul Mineiro (more precise numbers are available):
// http://web.archive.org/web/20160507083630/https://fastapprox.googlecode.com/svn/trunk/fastapprox/tests/fastapprox.nb
// The Paul Mineiro variants seem to be based on a Pade approximant
// https://math.stackexchange.com/questions/55830/how-to-calculate-ex-with-a-standard-calculator/56064#56064
//
// This blog shows benchmarks and errors for some exp approximation:
// https://deathandthepenguinblog.wordpress.com/2015/04/13/writing-a-faster-exp/
//
// according to this, SSE 4.1 (launched in 2006) solved the performance issues on Intel 
// for fast EXPs that convert to ints
// https://stackoverflow.com/questions/10552280/fast-exp-calculation-possible-to-improve-accuracy-without-losing-too-much-perfo
//
// annohter implementation in AVX-512 with a table, but I think we'd rather avoid the table to preserve cache.
// http://www.ecs.umass.edu/arith-2018/pdf/arith25_18.pdf
//
// This person reduced the "staircase effect" by using 64 bit integers and 32 bit floats.  I think in our case
// though, the error that we care about is the deviation of the line from the true value instead of the staircase
// effect, since we use summary statistics which will average out the staircase effect.  Also, in SIMD 
// implementations, we want to keep everything as 32 bits, so I dont' think this technique is useful to us,
// but it's include as a reference here to show the fit to the sigmoid function
// https://bduvenhage.me/performance/machine_learning/2019/06/04/fast-exp.html
//
// there's yet annother higher accuracy approximation of exp where the calculation is exp(x) = exp(x/2)/exp(-x/2)
// I tried this one and found that the Paul Mineiro version had quite a bit less error, and the real cost for both
// is the division instruction that they both introduce, but this one would be slightly faster so I'm leaving it
// here as a reference incase an intermediate error solution becomes useful in the future.  This solution was
// suggested by Nic Schraudolph (the publisher of the original paper above used in the algorithm below)
// "You can get a much better approximation (piecewise rational instead of linear) at the cost of a single 
//   floating-point division by using better_exp(x) = exp(x/2)/exp(-x/2), where exp() is my published approximation 
//   but you don't need the additive constant anymore, you can use c=0. On machines with hardware division this 
//   is very attractive." - Nic Schraudolph
// https://martin.ankerl.com/2007/02/11/optimized-exponential-functions-for-java/
// https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse


// Schraudolph constants used here:
// The Schraudolph paper uses doubles and 32 bit integers.  We want to use floats and 32 bit integers because
// it'll be more efficient in SIMD to keep the items the same width.  
//
// from the Schraudolph paper, formula 2.1:      i = a * y + (b - c)
// for float32, the bit shift is 23 rather than 20 used in the paper, and the shift constant is 127 vs 1023
// In the paper they were shifting by 20 to reach the exponent in a double, but used integers to do it and left
// the lower integer always zero.  Doubles have 52 bits dedicated to the significant (DBL_MANT_DIG - 1), 
// so after skipping the lower 32 bits, we get a shift of 53-1-32=20 for doubles
// For float32 processing, we don't skip the lowest 32 bits, so we need to shift by:
// (FLT_MANT_DIG - 1) bits = 23 bits.
// a = (1<<23) / ln(2) = 12102203.16156148555068672305845
// Rounding this value to the nearest 9 digits (which guarantees float32 uniqueness) leads to: 12102203.0f
// This 'a' value multiple is always the same and we never change it.
// 
// From the Schraudolph paper, b is calculated as such for float32 value:
// b = 127<<23 = 1065353216
//
// In the Schraudolph paper, there are options for calculating the 'c' value depending on what the user wants to 
// optimize for. The Schraudolph paper includes methods of calculating the constants for optimizing different 
// choices.  We can also run simulations to find other optimization points.  For simulations, the relative error
// repeats at predictable points with the formula y = k * ln(2), so relative error from 0 to ln(2) should equal 
// relative error from -ln(2) to 0 or ln(2) to 2 * ln(2).
// The paper includes theoretical calculations for several desirable outcomes.
// Here are the calculations for various options (remember to subtract 0.5 at the end for the staircase):
//   - k_expTermLowerBound:
//     c = 2^23 * (1 - (ln(ln(2)) + 1) / ln(2)) = 722018.66465506613537075079698488
//     round up to fix for the staircase: round(722018.66465506613537075079698488) = 722019
//     (b - c) is: 1065353216 - 722019 = 1064631197
//   - k_expTermUpperBound:
//     c = -1 (don't use 0 since that wouldn't account for the staircase effect)
//     (b - c) is: 1065353216 - (-1) = 1065353217
//   - k_expTermMinimizeRelativeMeanAbsError:
//     c = 0.045111411 * (2^23) / ln(2) - 0.5 = 545946.96082669957644759009613291
//     (b - c) is: 1065353216 - 545946.96082669957644759009613291 = 1064807269.0391733004235524099039
//     (b - c) rounded to integer is: 1064807269
//     (b - c) rounded to float is:   1064807300.0f
//   - k_expTermMinimizeRelativeRMSE:
//     c = 2^23 * ln(3 / (8 * ln(2)) + 0.5) / ln(2) - 0.5 = 486411.38068434173315475522228717
//     (b - c) is 1065353216 - 486411.38068434173315475522228717 = 1064866804.6193156582668452447777
//     (b - c) rounded to integer is: 1064866805
//   - k_expTermMinimizeRelativeMaximumError:
//     c = (ln(ln(2) + 2/e) - ln(2) - ln(ln(2))) * 2^23 / ln(2) - 0.5 = 
//        0.03027490056158269559309255065937 * 2^23 / ln(2) - 0.5 = 366392.49729234569298343147736888
//     (b - c) is 1065353216 - 366392.49729234569298343147736888 = 1064986823.5027076543070165685226
//     (b - c) rounded to integer is: 1064986824
// 
// Some of these constants can be verified by this independent implementation for float32/int32 here:
// https://github.com/ekmett/approximate/blob/master/cbits/fast.c
// 1065353216 + 1      = 1065353217 for the upper bound
// 1065353216 - 486411 = 1064866805 for the minimize RMSE
// 1065353216 - 722019 = 1064631197 for the lower bound


// interesting items:
// - according to the Schraudolph paper, if you choose an extreme Schraudolph term, the maximum error for any
//   particular exp value is 6.148%.  We care more about softmax error, and we use floats instead of doubles
// - according to the Schraudolph paper, if you choose a specific Schraudolph term, the mean relative absolute value of
//   error for any particular exp value is 1.483%.  Again, we care more about softmax error and we use floats,
//   but my experience is that we're in the same error range
// - for boosting, we sum the gradient terms in order to find the update direction (either positive or negative).  
//   As long as the direction has the correct sign, boosting will tend to make it continue to go in the right 
//   direction, even if the speed has a little variation due to approximate exp noise.  This means that our primary 
//   goal is to balance the errors in our softmax terms such that the positive errors balance the negative errors.
//   Note: minimizing relative mean abs error does NOT lead to minimizing positive/negative bias
// - the 2nd order derivate hessian Newton-Raphson term is summed separately and then divides the sum of the
//   gradients.  It affects the speed of convergence, but not the direction.  We choose to optimize our approximate
//   exp to obtain minimum error in the numerator where it will determine the sign and therefore direction of boosting
// - any consistent bias in the error will probably show up as miscalibration.  Miscalibration can lead to us to
//   trigger early stopping at an unideal point, but in general that seems to be a minor concern.  EBMs are well
//   calibrated though, and we'd like to preserve this property.
// - from experiments, our error is much better controlled if one of the logits is zeroed.  This makes one of the 
//   exp(0) outputs equal to 1, and thus have no error.  This seems to somehow help in terms of normalizing the 
//   range of the schraudolph terms required under various input skews compared with letting all the logits 
//   range wherever they go, and taking the approximate exp of all of them and then softmax
// - the error deviation seems to go down if the non-zeroed logits are all negative.  This makes sense since the
//   exp(0) -> 1 term in the softmax then dominates.  The effect isn't too big though, so having some positive
//   logits isn't the end of the world if it's more efficient to do it that way.  Finding the maximum logit
//   and subtracting that logit from the other logits requires an extra loop and computation, so our implementation
//   tollerates the slight miscalibration introduced instead of the extra CPU work
// - the schraudolph term that achieves average error of zero for the logit that we zero isn't the same 
//   schraudolph term that achieves mean error of zero for the non-zeroed terms.  They are close however.  If
//   we can't have the same ideal schraudolph term, that means we want to pick something in between, but the exact
//   value we'd pick would depend on the balance between number of samples from the zeroed logit class and the
//   non-zeroed class.  We might be able to develop a table with these skews and tailor them to each dataset at the 
//   start, but since they are relatively minor for now we just choose a value in between the two 
//   (average the experimentally determined Schraudolph term), which would be close to ideal IF 50% of the data is 
//   from the zeroed term and 50% of the data is from the non-zeroed term
// - the relative error from the exp function has a periodicity of k_expErrorPeriodicity -> ln(2)
// - relative errors are the same at points 1/2 the periodicity, but with the opposite derivative direction
// - softmax roughly follows this periodicity, but it's warped by the non-linear function, so we get the following
//   experimentally determined skews that have the same Schraudolph terms: -0.696462, -0.323004, 0, 0.370957, 0.695991
// - at the mid-point between skews (-0.509733 and 0.533474) between the 1st and 2nd values from zero we should find 
//   roughly the largest sumation error.  
// - For the skew of -0.509733 our averaged error is only -0.0010472912, but often you'd expect an error that isn't
//   the maximum since in real data there aren't precise boundaries for the probabilities of selection various logits
// - For the skew of +0.533474 our averaged error is only +0.0012019153, but often you'd expect an error that isn't
//   the maximum since in real data there aren't precise boundaries for the probabilities of selection various logits
// - with a very large positive skew of k_expErrorPeriodicity * 9.25, the averaged error is still only -0.0012371404
// - with a very large negative skew of k_expErrorPeriodicity * -9.25, the averaged error is still only +0.0000084012
// - notice above that errors for negative logits tend to be lower, which is consistent with what we described above
// - but even at worst, we're still averaging only 1/1000 of the error of the true softmax.  That's pretty good!
// - with the worse conditions I can think of (7 classes, skew of k_expErrorPeriodicity * 9.25) I managed
//   to get an average softmax error of -0.0386356382, but that's going to be rare
// - by going to 10 classes instead of 3, the average error for the zeroed logit was     -0.0001425264, which is great
// - by going to 10 classes instead of 3, the average error for the non-zeroed logit was +0.0000267197, which is ok
//   because there might be more of the non-zeroed logits given that there are 9 of them to the 1 zeroed logit
// - we don't care too much about the error from the log function.  That error is used only for log loss, and the
//   log loss is only used for early stopping, and early stopping is only used in a relative sense (improvement
//   allows continued improvement), so if there is any skew, it'll just shift the log loss curve upwards or downwards, 
//   but it shouldn't change the minimum point on the curve very much.  We also average the log loss accross many 
//   samples, so the variance should be reduced by a lot on the averaged term
// - the best way to pick the addExpSchraudolphTerm would probably be to use benchmarks on real data to fine tune
//   the parameter such that errors are balanced on a massive multi-dataset way

// the relative error from our approximate exp function has a periodicity of ln(2), so [0, ln(2)) should have the 
// same relative error as [-ln(2) / 2, +ln(2) / 2), [10, 10 + ln(2)) or any other shift.  We can use this property when testing to optimize
// for various outcomes, BUT our inputs needs to be evenly distributed, so we can't use std::nextafter.  
// We need to increment in identical increments like "ln(2) / some_prime_number".  The period [-ln(2) / 2, +ln(2) / 2) is interesting
// because it concentrates in the region with the most reslution for floating point values and we could get a
// proven symmetric look at the data, and we could iterate with an integer which would ensure we have precise
// negative and positive inputs




static constexpr double k_expErrorPeriodicity = 0.69314718055994529; // ln(2)

// this constant does not change for any variation in optimizing for different objectives in Schraudolph
static constexpr float k_expMultiple = 12102203.0f; // (1<<23) / ln(2)

// all of these constants should have exact rounding to float, per IEEE 754 9-12 digit rule (see above)
static constexpr int32_t k_expTermUpperBound                   = 1065353217; // theoretically justified -> 1065353216 + 1
static constexpr int32_t k_expTermMinimizeRelativeMaximumError = 1064986824; // theoretically justified
static constexpr int32_t k_expTermZeroMeanRelativeError        = 1064870596; // experimentally determined.  This unbiases our average such that averages of large numbers of values should be balanced towards zero
static constexpr int32_t k_expTermMinimizeRelativeRMSE         = 1064866805; // theoretically justified -> 1065353216 - 486411
static constexpr int32_t k_expTermMinimizeRelativeMeanAbsError = 1064807269; // theoretically justified -> 1065353216 - 545947
static constexpr int32_t k_expTermLowerBound                   = 1064631197; // theoretically justified -> 1065353216 - 722019

// experimentally determined softmax Schraudolph terms for softmax where one logit is zeroed by subtracing it from the other logits
static constexpr int32_t k_expTermZeroMeanRelativeErrorSoftmaxThreeClassesZeroingSkewMinus0_696462 = 1064873067; // experimentally determined, +-1, from -0.696462 - k_expErrorPeriodicity / 2 to -0.696462 + k_expErrorPeriodicity / 2
// mid-point = -0.509733
static constexpr int32_t k_expTermZeroMeanRelativeErrorSoftmaxThreeClassesZeroingSkewMinus0_323004 = 1064873067; // experimentally determined, +-1, from -0.323004 - k_expErrorPeriodicity / 2 to -0.323004 + k_expErrorPeriodicity / 2
static constexpr int32_t k_expTermZeroMeanRelativeErrorSoftmaxThreeClassesZeroingSkew0      = 1064873067; // experimentally determined, +-1, from -k_expErrorPeriodicity / 2 to +k_expErrorPeriodicity / 2
static constexpr int32_t k_expTermZeroMeanRelativeErrorSoftmaxThreeClassesZeroingSkewPlus0_370957 = 1064873067; // experimentally determined, +-1, from 0.370957 - k_expErrorPeriodicity / 2 to 0.370957 + k_expErrorPeriodicity / 2
// mid-point = 0.533474
static constexpr int32_t k_expTermZeroMeanRelativeErrorSoftmaxThreeClassesZeroingSkewPlus0_695991 = 1064873067; // experimentally determined, +-1, from 0.695991 - k_expErrorPeriodicity / 2 to 0.695991 + k_expErrorPeriodicity / 2

static constexpr int32_t k_expTermZeroMeanRelativeErrorSoftmaxThreeClassesZeroingTermZeroedRange20 = 1064872079; // experimentally determined, +-1, the zeroed class, from -10 * k_expErrorPeriodicity / 2 to +10 * k_expErrorPeriodicity / 2
// USE THIS Schraudolph term for softmax.  It gives 0.001% -> 0.00001 average error, so it's unlikley to miscalibrate 
// the results too much.  Keeping all softmax terms without zeroing any of them leads to an error of about 0.16% or 
// thereabouts, which is much larger than anything observed for zeroed logits
// 
// this value is the average value between k_expTermZeroMeanRelativeErrorSoftmaxThreeClassesZeroingTermZeroedRange20 and
// k_expTermZeroMeanRelativeErrorSoftmaxThreeClassesZeroingTermNonZeroedRange20, which should be approximately correct IF
// the zeroed logit class has roughly the same number of samples as the non-zeroed classes combined.  We don't 
// know the sample class makeup though (and building a table for various sample class skews would be time consuming)
// so taking the average is a pretty good guess, especially since the upper and lower values are fairly close
// 
// this Schraudolph term works relatively well for softmax with 3+ classes, different skews, and if the predicted
// class is the zeroed class or a non-zeroed class.  It also works fairly well if the non-zeroed logits are greater
// than zero after shifting them
static constexpr int32_t k_expTermZeroMeanErrorForSoftmaxWithZeroedLogit = 1064871915; // experimentally determined, AVERAGED between the zeroed and non-zeroed class, from -10 * k_expErrorPeriodicity / 2 to +10 * k_expErrorPeriodicity / 2
static constexpr int32_t k_expTermZeroMeanRelativeErrorSoftmaxThreeClassesZeroingTermNonZeroedRange20 = 1064871750; // experimentally determined, +-1, the non-zeroed class, from -10 * k_expErrorPeriodicity / 2 to +10 * k_expErrorPeriodicity / 2

// TODO: we can probably pick a better k_expTermZeroMeanErrorForSoftmaxWithZeroedLogit above for binary classification 
// where we know for sure the number of classes, and we know the likely sign of the value going into the exp 
// function because we flip the sign of the logit based on the true value, and boosting ensures that we'll be 
// trying to push that value in a consistent direction towards the true value

// these are rough determinations that need to be vetted 
//static constexpr int32_t k_expTermZeroMeanRelativeErrorSoftmaxTwoClassesZeroingSkew0 = 1064873955; // experimentally determined, +-?
//static constexpr int32_t k_expTermZeroMeanRelativeErrorSoftmaxTwoClassesZeroingSkew0_370? = 1064873955; // experimentally determined, +-?
//static constexpr int32_t k_expTermZeroMeanRelativeErrorSoftmaxTwoClassesZeroingSkew0_69982536866359447004608294930875 = 1064873955; // experimentally determined, +-?


// DO NOT USE.  This constant minimizes the mean error for softmax where none of the logits are zeroed.  It has
// too much variability in the output though to be useful when compared against the softmax that zeroes one of the logits
// Using non-zeroed softmax isn't as viable since at modest skews (I tried k_expErrorPeriodicity / 4), achieving zero average error is impossible
// since all valid addition terms from k_expTermLowerBound to k_expTermUpperBound can result in non-zero error averages
// softmax with a zeroed term seems to have much tighter bounds on the mean error even if the absolute mean error is higher
static constexpr int32_t k_expTermZeroMeanRelativeErrorSoftmaxThreeClassesSkew0 = 1064963329; // experimentally determined +-1, non-zeroed, from -k_expErrorPeriodicity / 2 to k_expErrorPeriodicity / 2

// k_expUnderflowPoint is set to a value that prevents us from returning a denormal number. These approximate function 
// don't really work with denomals and start to drift very quickly from the true exp values when we reach denormals.
static constexpr float k_expUnderflowPoint = -87.25f; // this is exactly representable in IEEE 754
static constexpr float k_expOverflowPoint = 88.5f; // this is exactly representable in IEEE 754

// use the integer version for now since in non-SIMD this is probably faster and more exact
#define EXP_INT

template<
   bool bNegateInput = false,
   bool bNaNPossible = true,
   bool bUnderflowPossible = true,
   bool bOverflowPossible = true,
   bool bSpecialCaseZero = false,
   typename T
>
INLINE_ALWAYS static T ExpApproxSchraudolph(T val, const int32_t addExpSchraudolphTerm = k_expTermZeroMeanErrorForSoftmaxWithZeroedLogit) {
   // This function guarnatees non-decreasing monotonicity, so it never decreases with increasing inputs, but
   // it can sometimes yield equal outputs on increasing inputs

   EBM_ASSERT(k_expTermLowerBound <= addExpSchraudolphTerm);
   EBM_ASSERT(addExpSchraudolphTerm <= k_expTermUpperBound);

   const bool bPassNaN = bNaNPossible && UNLIKELY(std::isnan(val));
   if(LIKELY(!bPassNaN)) {
      // we need to check the following before converting val to a float, if a conversion is needed
      // The following things we do below would invoke undefined behavior otherwise:
      //   - converting a big positive double into a float that can't be represented
      //   - converting a big negative double into a float that can't be represented
      //   - converting a large float to an int that can't be represented
      //   - converting +-infinity to an int32_t
      //   - converting a NaN to an int32_t
      // https://stackoverflow.com/questions/10366485/problems-casting-nan-floats-to-int
      // https://docs.microsoft.com/en-us/cpp/c-language/conversions-from-floating-point-types?view=msvc-160

      // NOTE: I think the SIMD intrinsics don't invoke undefined behavior because they are platform specific 
      //       intrinsics and would therefore be defined by the specific platform, so we could in that case pass 
      //       a NaN value through our processing below and check at the end for NaN and then replace it as needed
      //       or convert floats to ints that can't succeed, provided the platform defines how these are handled

      if(bUnderflowPossible) {
         if(bNegateInput) {
            if(UNLIKELY(static_cast<T>(-k_expUnderflowPoint) < val)) {
               return T { 0 };
            }
         } else {
            if(UNLIKELY(val < static_cast<T>(k_expUnderflowPoint))) {
               return T { 0 };
            }
         }
      }
      if(bOverflowPossible) {
         if(bNegateInput) {
            if(UNLIKELY(val < static_cast<T>(-k_expOverflowPoint))) {
               return std::numeric_limits<T>::infinity();
            }
         } else {
            if(UNLIKELY(static_cast<T>(k_expOverflowPoint) < val)) {
               return std::numeric_limits<T>::infinity();
            }
         }
      }
      if(bSpecialCaseZero) {
         if(UNLIKELY(T { 0 } == val)) {
            return T { 1 };
         }
      }

      const float valFloat = static_cast<float>(val);
      static constexpr float signedExpMultiple = bNegateInput ? -k_expMultiple : k_expMultiple;

#ifdef EXP_INT

      // this version does the addition in integer space, so it's maybe faster if there isn't a fused multiply add
      // instruction that works on floats since the integer add will not be slower than a float add, unless the ALU
      // has some transfer time or time to swtich integer/float values.  Integers in the range we're expecting also
      // have a little more precision, which means we can get closer to the ideal addExpSchraudolphTerm that we want

      const int32_t retInt = static_cast<int32_t>(signedExpMultiple * valFloat) + addExpSchraudolphTerm;
#else

      // TODO: test the speed of this form once we have SIMD implemented

      // this version might be faster if there's a cost to switching between int to float.  Fused multiply add is either 
      // just as fast as plain multiply, or pretty close to it though, so throwing in the add might be low cost or free
      const int32_t retInt = static_cast<int32_t>(signedExpMultiple * valFloat + static_cast<float>(addExpSchraudolphTerm));
#endif

      float retFloat;

      // It's undefined behavior in C++ (not C though!) to use a union or a pointer to bit convert between types 
      // using memcpy though is portable and legal since C++ aliasing rules exclude character pointer copies
      // Suposedly, most compilers are smart enough to optimize the memcpy away.

      static_assert(std::numeric_limits<float>::is_iec559, "This hacky function requires IEEE 754 binary layout");
      static_assert(sizeof(retFloat) == sizeof(retInt), "both binary conversion types better have the same size");
      memcpy(&retFloat, &retInt, sizeof(retFloat));

      val = static_cast<T>(retFloat);
   }
   return val;
}

template<
   bool bNegateInput = false,
   bool bNaNPossible = true,
   bool bUnderflowPossible = true,
   bool bOverflowPossible = true,
   bool bSpecialCaseZero = false,
   typename T
>
INLINE_ALWAYS static T ExpApproxBest(T val) {
   // This function DOES NOT guarnatee monotonicity, and in fact I've seen small monotonicity violations, so it's
   // not just a theoretical consideration.  Unlike the ExpApproxSchraudolph, we have discontinuities
   // at the integer rounding points due to floating point rounding inexactness.

   // algorithm from Paul Mineiro.  re-implemented below with additional features for our package.  Also,
   // I slightly modified his constants because float in C++ is fully expressed by having 9 digits in base
   // 10, and I wanted to have the least variability between compilers.  The compiler should be turning his constants
   // and my constants into the same representation, but for compatility between compilers I thought it would be best
   // to get as close to the to compiler representation as possible
   // https://github.com/etheory/fastapprox/blob/master/fastapprox/src/fastexp.h
   // Here's a description by Paul Mineiro on how it works
   // http://www.machinedlearnings.com/2011/06/fast-approximate-logarithm-exponential.html
   // and a Mathematica Notebook where the constants were calculated (and longer ones are available):
   // http://web.archive.org/web/20160507083630/https://fastapprox.googlecode.com/svn/trunk/fastapprox/tests/fastapprox.nb
   // This blog shows benchmarks and errors for this approximation and some others:
   // https://deathandthepenguinblog.wordpress.com/2015/04/13/writing-a-faster-exp/
   // according to this, SSE 4.1 (launched in 2006) solved the performance issues on Intel with fast EXPs 
   // https://stackoverflow.com/questions/10552280/fast-exp-calculation-possible-to-improve-accuracy-without-losing-too-much-perfo
   // If we need a double version, here's one
   // https://bduvenhage.me/performance/machine_learning/2019/06/04/fast-exp.html
   // Here's a different approximation that uses AVX-512, and has a version for doubles
   // https://github.com/jhjourdan/SIMD-math-prims/blob/master/simd_math_prims.h
   // annohter implementation in AVX-512 (with a table, but I think we'd rather avoid the table to preserve cache)
   // http://www.ecs.umass.edu/arith-2018/pdf/arith25_18.pdf
   // more versions of approximate exp from the Schraudolph99 paper "A Fast, Compact Approximation of the Exponential Function"
   // https://github.com/ekmett/approximate/blob/master/cbits/fast.c

   const bool bPassNaN = bNaNPossible && UNLIKELY(std::isnan(val));
   if(LIKELY(!bPassNaN)) {
      // we need to check the following before converting val to a float, if a conversion is needed
      // The following things we do below would invoke undefined behavior otherwise:
      //   - converting a big positive double into a float that can't be represented
      //   - converting a big negative double into a float that can't be represented
      //   - converting a large float to an int that can't be represented
      //   - converting +-infinity to an int32_t
      //   - converting a NaN to an int32_t
      // https://stackoverflow.com/questions/10366485/problems-casting-nan-floats-to-int
      // https://docs.microsoft.com/en-us/cpp/c-language/conversions-from-floating-point-types?view=msvc-160

      // NOTE: I think the SIMD intrinsics don't invoke undefined behavior because they are platform specific 
      //       intrinsics and would therefore be defined by the specific platform, so we could in that case pass 
      //       a NaN value through our processing below and check at the end for NaN and then replace it as needed
      //       or convert floats to ints that can't succeed, provided the platform defines how these are handled

      if(bUnderflowPossible) {
         if(bNegateInput) {
            if(UNLIKELY(static_cast<T>(-k_expUnderflowPoint) < val)) {
               return T { 0 };
            }
         } else {
            if(UNLIKELY(val < static_cast<T>(k_expUnderflowPoint))) {
               return T { 0 };
            }
         }
      }
      if(bOverflowPossible) {
         if(bNegateInput) {
            if(UNLIKELY(val < static_cast<T>(-k_expOverflowPoint))) {
               return std::numeric_limits<T>::infinity();
            }
         } else {
            if(UNLIKELY(static_cast<T>(k_expOverflowPoint) < val)) {
               return std::numeric_limits<T>::infinity();
            }
         }
      }
      if(bSpecialCaseZero) {
         if(UNLIKELY(T { 0 } == val)) {
            return T { 1 };
         }
      }

      float valFloat = static_cast<float>(val);
      float negativeCorrection;

      // if valFloat is zero, give the same negativeCorrection since the output should be indistinquishable betwee +-0
      if(bNegateInput) {
         negativeCorrection = UNPREDICTABLE(valFloat <= 0.00000000f) ? 0.00000000f : 1.00000000f;
      } else {
         negativeCorrection = UNPREDICTABLE(0.00000000f <= valFloat) ? 0.00000000f : 1.00000000f;
      }

      // 9 digits for most accurate float representation of: 1 / ln(2)
      valFloat *= bNegateInput ? -1.44269502f : 1.44269502f;

      // TODO: (static_cast<float>(static_cast<int32_t>(valFloat)) - negativeCorrection) can alternatively be computed as:
      //       - FIRST, look up _mm_round_sd with _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC which I think does
      //         what we want in a single assembly instruction
      //         #define _MM_FROUND_TO_NEAREST_INT    0x00
      //         #define _MM_FROUND_TO_NEG_INF        0x01
      //         #define _MM_FROUND_TO_POS_INF        0x02
      //         #define _MM_FROUND_TO_ZERO           0x03
      //         #define _MM_FROUND_CUR_DIRECTION     0x04
      //         https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_round_sd&expand=4761
      //       - floor(valFloat) - UNPREDICTABLE(0.00000000f <= valFloatOriginal) ? 0.00000000f : 1.00000000f
      //         UNPREDICTABLE(0.00000000f <= valFloatOriginal) ? floor(valFloat) : ceil(valFloat)
      //         UNPREDICTABLE(0.00000000f <= valFloatOriginal) ? floor(valFloat) : floor(valFloat) - 1.00000000f
      //       - I think the floor/ceil version will be the fastest since the CPU will be able to pipeline
      //         both the floor and ceil operation in parallel and results will be ready at almost the same time for the
      //         non-branching selection.  Taking the floor and then subtracting doesn't allow the subtraction to happen
      //         until the floor completes
      //       - I'm pretty sure that converting to an int32_t and then back to a float is slower since it's two
      //         operations, while ceil is just one operation, and it does the same rounding!
      //       - Also, look into if there is an operator that always rounds to the more negative side.  ceil and floor
      //         round towards zero, which means negative numbers are rounding to the more positive side rather
      //         than the more negative side like we need
      //       - for all of these operations, test that they give the same results, especially on whole number boundaries

      // this approximation isn't perfect.  Occasionally it isn't monotonic when valFloat wraps integer boundaries
      const float factor = valFloat - (static_cast<float>(static_cast<int32_t>(valFloat)) - negativeCorrection);

      // original formula from here: https://github.com/etheory/fastapprox/blob/master/fastapprox/src/fastexp.h
      // from the Mathematica Notebook where the constants were calculated (and longer ones are available):
      // http://web.archive.org/web/20160507083630/https://fastapprox.googlecode.com/svn/trunk/fastapprox/tests/fastapprox.nb
      // 121.2740575f is really 121.27405755366965833343835512821308140575
      // 27.7280233f is really 27.72802338968109763318198810783757358521
      // 4.84252568f is really 4.84252568892855574935929971220272175743
      // 1.49012907f is really 1.49012907173427359585695086181761669065
      //
      // 2^23 * 121.27405755366965833343835512821308140575 = 1017320529.3871737252531476533353 => 1.01732051e+09f
      // 2^23 * 27.72802338968109763318198810783757358521 = 232599518.83086597305449149089731 => 2.32599520e+08f
      // 2^23 * 1.49012907173427359585695086181761669065 = 12500108.65218270136039438485505 => 1.25001090e+07f
      // 4.84252568892855574935929971220272175743 => 4.84252548f

      // in theory it would be more accurate for us to keep all the constants small and close to zero 
      // and multiply by the float { 1 << 23 } at the end after all the additions and subtractions, but I've
      // tested this and the error after expanding it is indistinquishable from if we use larger constants
      // I think it's because we don't care about small changes in the final result much since they tend to
      // make very small changes in the floating point output, and this loss in precision is less than the
      // loss in precision that we're getting by using an approximate exp.

      // TODO: we can probably achieve slightly better results by tuning the single addition constant below of
      //       1.01732051e+09f OR 121.274055f to get better summation results similar to Schraudolph, AND/OR we can 
      //       also convert this to a int32_t value so that we have even more fiddling ability to tune 
      //       the output to average to zero.  For now it's good enough though

      const int32_t retInt = static_cast<int32_t>(
#ifdef EXP_HIGHEST_ACCURACY
         float { 1 << 23 } * (valFloat + 121.274055f - 1.49012911f * factor + 27.7280235f / (4.84252548f - factor))
#else
         float { 1 << 23 } * valFloat + 1.01732051e+09f - 1.25001090e+07f * factor + 2.32599520e+08f / (4.84252548f - factor)
#endif
      );

      float retFloat;

      // It's undefined behavior in C++ (not C though!) to use a union or a pointer to bit convert between types 
      // using memcpy though is portable and legal since C++ aliasing rules exclude character pointer copies
      // Suposedly, most compilers are smart enough to optimize the memcpy away.

      static_assert(std::numeric_limits<float>::is_iec559, "This hacky function requires IEEE 754 binary layout");
      static_assert(sizeof(retFloat) == sizeof(retInt), "both binary conversion types better have the same size");
      memcpy(&retFloat, &retInt, sizeof(retFloat));

      val = static_cast<T>(retFloat);
   }
   return val;
}

template<bool bNegateInput = false, typename T>
INLINE_ALWAYS static T ExpForBinaryClassification(const T val) {
#ifdef FAST_EXP
   // the optimal addExpSchraudolphTerm would be different between binary 
   // and multiclass since the softmax form we use is different

   // TODO : tune addExpSchraudolphTerm specifically for binary classification.  
   //        k_expTermZeroMeanErrorForSoftmaxWithZeroedLogit was tuned for softmax with 3 classes and one of them zeroed.  
   //        For binary classification we can assume large positive logits typically, unlike multiclass

   return ExpApproxSchraudolph<bNegateInput, true, true, true, false, T>(val, k_expTermZeroMeanErrorForSoftmaxWithZeroedLogit);
#else // FAST_EXP
   return std::exp(bNegateInput ? -val : val);
#endif // FAST_EXP
}

template<bool bNegateInput = false, typename T>
INLINE_ALWAYS static T ExpForMulticlass(const T val) {
#ifdef FAST_EXP
   // the optimal addExpSchraudolphTerm would be different between binary
   // and multiclass since the softmax form we use is different

   // TODO : tune addExpSchraudolphTerm specifically for multiclass classification.  Currently (at the time that 
   //        I'm writing this, although we have plans to change it) we aren't zeroing a 
   //        logit and k_expTermZeroMeanErrorForSoftmaxWithZeroedLogit is really the wrong constant since it was
   //        tuned for softmax with a zeroed logit

   return ExpApproxSchraudolph<bNegateInput, true, true, true, false, T>(val, k_expTermZeroMeanErrorForSoftmaxWithZeroedLogit);
#else // FAST_EXP
   return std::exp(bNegateInput ? -val : val);
#endif // FAST_EXP
}




///////////////////////////////////////////// LOG SECTION

// this constant does not change for any variation in optimizing for different objectives in Schraudolph
static constexpr float k_logMultiple = 8.26295832e-08f; // ln(2) / (1<<23)

// k_logTermUpperBound = 
// std::nextafter(-(k_expTermLowerBound * ln(2) / (1 << 23))) = 
// std::nextafter(-(1064631197 * ln(2) / (1 << 23)), 0.0f) = 
// std::nextafter(-87.970031802262032630372429596766f, 0.0f)
// we need to take the nextafter, because the round of that number is -87.9700317f otherwise which is below the bound
// -87.9700241f
static constexpr float k_logTermUpperBound = -87.9700241f;

// k_logTermLowerBound = 
// -(k_expTermUpperBound * ln(2) / (1 << 23)) = 
// -(1065353217 * ln(2) / (1 << 23)) = 
// -88.029692013742637244666652182484f
// -88.0296936f (rounded)
// we do not need to take the nextafter, because the round of that number is -88.0296936f is already outside the bound
static constexpr float k_logTermLowerBound = -88.0296936f;

// ln(1) = 0, and we want to be close to that.  Our boosting never goes below 1 for log loss, so if we set
// a minimum of 0.9999 (to account for floating point inexactness), then our minimum is:
static constexpr float k_logTermLowerBoundInputCloseToOne = -88.02955453797396f;

// LOG constants
static constexpr float k_logTermZeroMeanErrorForLogFrom1_To1_5 = -87.9865799f; // experimentally determined.  optimized for input values from 1 to 1.5.  Equivalent to 1064831465


// the more exact log constancts from https://github.com/etheory/fastapprox/blob/master/fastapprox/src/fastlog.h
// listed in http://web.archive.org/web/20160507083630/https://fastapprox.googlecode.com/svn/trunk/fastapprox/tests/fastapprox.nb
// are (we might need them someday if we want to construct a more exact log function):
// 0.69314718f is really: ln(2)
// 1.1920928955078125e-7f is really: 1/8388608 = 1 / (1 << 23)
// 124.22551499f is really: 124.22551499451099260110626945692909242404
// 1.498030302f is really: 1.49803030235745199322015574519753670721
// 1.72587999f is really: 1.72587998888731819829751241367603866776
// 0.3520887068f is really: 0.35208870683243006166779601840602754876

// some log implementations
// https://github.com/etheory/fastapprox/blob/master/fastapprox/src/fastlog.h
// https://www.icsi.berkeley.edu/pubs/techreports/TR-07-002.pdf
// https://tech.ebayinc.com/engineering/fast-approximate-logarithms-part-i-the-basics/
// https://tech.ebayinc.com/engineering/fast-approximate-logarithms-part-ii-rounding-error/
// https://tech.ebayinc.com/engineering/fast-approximate-logarithms-part-iii-the-formulas/
// https://stackoverflow.com/questions/9799041/efficient-implementation-of-natural-logarithm-ln-and-exponentiation
// https://github.com/ekmett/approximate/blob/master/cbits/fast.c

template<
   bool bNegateOutput = false,
   bool bNaNPossible = true,
   bool bNegativePossible = true,
   bool bZeroPossible = true, // if false, positive zero returns a big negative number, negative zero returns a big positive number
   bool bPositiveInfinityPossible = true, // if false, +inf returns a big positive number.  If val can be a double that is above the largest representable float, then setting this is necessary to avoid undefined behavior
   typename T
>
INLINE_ALWAYS static T LogApproxSchraudolph(T val, const float addLogSchraudolphTerm = k_logTermLowerBoundInputCloseToOne) {
   // NOTE: this function will have large errors on denomal inputs, but the results are reliably big negative numbers

   // to get the log, just reverse the approximate exp function steps

   // we can also figure out the inverse constants above.  Take for example the theoretically determined value
   // of k_expTermMinimizeRelativeMaximumError = 1064986824
   // to calculate our addLogSchraudolphTerm for LogApproxSchraudolph, calculate:
   // -(1064986824 * ln(2) / (1 << 23)) = -87.999417112957322202915588367286
   // or inversely, calculate the equivalent exp term:
   // -(-87.999417112957322202915588367286 * (1 << 23) / ln(2)) = 1064986824

   EBM_ASSERT(k_logTermLowerBound <= addLogSchraudolphTerm);
   EBM_ASSERT(addLogSchraudolphTerm <= k_logTermUpperBound);

   const bool bPassNaN = bNaNPossible && !bNegativePossible && UNLIKELY(std::isnan(val));
   if(LIKELY(!bPassNaN)) {
      const bool bPassInfinity = bPositiveInfinityPossible && std::is_same<T, float>::value &&
         UNLIKELY(std::numeric_limits<T>::infinity() == val);
      if(LIKELY(!bPassInfinity)) {
         if(bNegativePossible) {
            // according to IEEE 754, comparing NaN to anything returns false (except itself), so checking if it's 
            // greater or equal to zero will yield false if val is a NaN, and then true after the negation, so this 
            // checks for both of our NaN output conditions.  This needs to be compiled with strict floating point!
            if(UNLIKELY(!(T { 0 } < val))) {
               if(bZeroPossible) {
                  return PREDICTABLE(T { 0 } == val) ? 
                     -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::quiet_NaN();
               } else {
                  return std::numeric_limits<T>::quiet_NaN();
               }
            }
         } else {
            if(bZeroPossible) {
               if(UNLIKELY(T { 0 } == val)) {
                  return -std::numeric_limits<T>::infinity();
               }
            }
         }
         if(!std::is_same<T, float>::value) {
            if(UNLIKELY(static_cast<T>(std::numeric_limits<float>::max()) < val)) {
               // if val is a non-float32, and it has a value outside of the float range, then it would result in 
               // undefined behavior if we converted it to a float, so check it here and return
               return std::numeric_limits<T>::infinity();
            }
         }

         // if val is a float, there are no values which would invoke undefined behavior for the code below since
         // we bit-convert our float to an integer, which will be legal for all possible integer outputs.
         // We then conver that integer to a float with a cast, which should be ok too since there are no integers 
         // that cannot be converted to a float, so there are no undefined behavior possibilities.
         // if val is a double though, and it has a value outside of the float range, then it would result in 
         // undefined behavior if we converted it to a float, so we check for that above.

         const float valFloat = static_cast<float>(val);

         int32_t retInt;

         static_assert(std::numeric_limits<float>::is_iec559, "This hacky function requires IEEE 754 binary layout");
         static_assert(sizeof(retInt) == sizeof(valFloat), "both binary conversion types better have the same size");
         memcpy(&retInt, &valFloat, sizeof(retInt));

         float retFloat = static_cast<float>(retInt);

         // use a fused multiply add assembly instruction, so don't add the addLogSchraudolphTerm prior to multiplying
         if(bNegateOutput) {
            retFloat = (-k_logMultiple) * retFloat + (-addLogSchraudolphTerm);
         } else {
            retFloat = k_logMultiple * retFloat + addLogSchraudolphTerm;
         }

         val = static_cast<T>(retFloat);
      }
   }
   return val;
}

template<bool bNegateOutput = false, typename T>
INLINE_ALWAYS static T LogForLogLoss(const T val) {

   // the log function is only used to calculate the log loss on the valididation set only in our codebase
   // the log loss is calculated for the validation set and then returned as a single number to the caller
   // it never gets used as an input to anything inside our code, so any errors won't cyclically grow

   // for log, we always sum the outputs, and those outputs are used only for early stopping, and early stopping
   // only cares about relative changes (we compare against other log losses computed by this same function),
   // so we should be insensitive to shifts in the output provided they are by a constant amount
   // The only reason to want balanced sums would be if we report the log loss to the user because then we want
   // the most accurate value we can give them, otherwise we should be re-computing it for the user if they want
   // an exact value

   // boosting will tend to push us towards lower and lower log losses.  Our input can't be less than 1 
   // (without floating point imprecision considerations), so 0 is our lowest output, and the input shouldn't be 
   // much more than 2 (which returns about 0.69, which is random chance), and probably quite a bit smaller than that 
   // in general

#ifdef FAST_LOG

   // it is possible in theory for val to be +infinity if the logits get to above k_expOverflowPoint, BUT
   // we don't get undefined behavior if bPositiveInfinityPossible is set to false.  What we get out is a very
   // big positive number.  In fact, we get the biggest legal positive value, which should terminate boosting
   // at that point if early stopping is enabled.  If we return a large positive result here instead of infinity
   // there shouldn't be any real consequences.
   // val can't be negative or zero from the formulas that we calculate before calling this log function
   return LogApproxSchraudolph<bNegateOutput, true, false, false, false, T>(val, k_logTermLowerBoundInputCloseToOne);
#else // FAST_LOG
   T ret = std::log(val);
   ret = bNegateOutput ? -ret : ret;
   return ret;
#endif // FAST_LOG
}

} // DEFINED_ZONE_NAME

#endif // APPROXIMATE_MATH_HPP