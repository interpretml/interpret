// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef APPROXIMATE_MATH_H
#define APPROXIMATE_MATH_H

#include "ebm_native.h"
#include "EbmInternal.h"

// TODO: try out floats throughout our program instead of doubles.  It'll be important when we move to GPUs and SIMD

// !!IMPORTANT!!: these function require being compiled with strict adherdence to floating point

// START SECTION CORRECT ROUNDING : https://en.wikipedia.org/wiki/IEEE_754#Roundings_to_nearest
// IEEE 754 has a concept called correct rounding that says that certain opterations should be rounded as if they were 
//      computed with infinite precision and then rounded to the nearest bit value.  This is theoretically possible, 
//      except for a limited number of operations like y^w for which there is no known theoretical solution
//      https://en.wikipedia.org/wiki/Rounding#Table-maker's_dilemma
// 
// IEEE 754-1985 -> requires correct rounding to nearest for the following operations:
//   - addition, subtraction, multiplication, division, fused multiply add
// IEEE 754-2008 -> RECOMMENDS (but not require) correct rounding to nearest for these additional operations:
//   - e^x, ln(x), sqrt(x), and others too, but these are the ones we use
//   - float32 values written as text need to be correctly rounded if they have 9-12 digits, 
//     and 9 digits are required in base 10 representation to differentiate all floating point numbers
//     eg: 1234567.89f -> has 9 digits
//
// In my experience, there is still variation between compilers and many are not completely IEEE-754 compilant,
//   even when they claim to be with regards to correct rounding and other operations.  For instance, 
//   Visual Studio in x86 compilation calls a function to handle rounding of floating points to integers, 
//   and that function returns different results than the Intel instruction for conversion, so the x64 compiles 
//   return different results.
//
// It's probably not a good strategy to assume that we can get 100% reproducible results, even though with a fully
//   compilant IEEE-754 compiler we could do so.  These approximate function help a bit in that direction though
//   because we eliminate the variations between compilers with respect to exp and log functions.  We're still
//   beholden to rounding, but if we include a detection of SSE 4.1 (launched in 2006), then we could use the
//   SSE instructions that convert from float to int, so we could probably get conformance in practice for any
//   machine built since then.  I'd assume ARM using Neon (their SIMD implementation) and other newer processors 
//   would have similar conformance in newer machines independent of their compilers.
//
// END SECTION CORRECT ROUNDING


// log is the reversal of the exp function, so we can simply reverse the operations of the exp function to obtain
// the log.  As such, our log and exp functions should share the same constants, so that they are exactly reversible.
//
// This paper details our fast exp/log approximation -> "A Fast, Compact Approximation of the Exponential Function"
// by Nicol N. Schraudolph
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.4508&rep=rep1&type=pdf
//
// Here's an implementation from the paper, but includes other options, like float/uint32_t versions, and log
// https://github.com/ekmett/approximate/blob/master/cbits/fast.c
//
// different algorithm from Paul Mineiro:
// https://github.com/etheory/fastapprox/blob/master/fastapprox/src/fastexp.h
// Here's a description by Paul Mineiro on how his works
// http://www.machinedlearnings.com/2011/06/fast-approximate-logarithm-exponential.html
// and a Mathematica Notebook where the constants were calculated for Paul Mineiro (and longer ones are available):
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
// suggested by Nic Schraudolph (the publisher of the original paper above used in the algorithm below)
// "You can get a much better approximation (piecewise rational instead of linear) at the cost of a single 
//   floating-point division by using better_exp(x) = exp(x/2)/exp(-x/2), where exp() is my published approximation 
//   but you don't need the additive constant anymore, you can use c=0. On machines with hardware division this 
//   is very attractive." - Nic Schraudolph
// https://martin.ankerl.com/2007/02/11/optimized-exponential-functions-for-java/
// https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse
//
// FUTURE IDEAS:
// Per the Schraudolph paper, you can choose to optimize for various outputs by tweaking the addition constant 
// This function below chooses to minimize the mean relative error.  There's probably a better fixed constant 
// to use for minimizing the softmax error.  We might be able to figure that out using the method described in 
// the Schraudolph paper to minimize RMSE and mean relative error.
// It's also possible that when using Newton-Raphson updates, optimizing for RMSE would be better since
// that includes a squared term


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
// repeats at predictable points with the formula y = k * ln(2), so error from 0 to ln(2) should equal error
// from -ln(2) to 0 or ln(2) to 2 * ln(2).
// The paper includes theoretical calculations for several theoretically desirable outcomes.  
// Here are the calculations for various options (remember to subtract 0.5 at the end for the staircase):
//   - k_termMinimizeMeanRelativeAbsError:
//     c = 0.045111411 * (2^23) / ln(2) - 0.5 = 545946.96082669957644759009613291
//     (b - c) is: 1065353216 - 545946.96082669957644759009613291 = 1064807269.0391733004235524099039
//     (b - c) rounded to integer is: 1064807269
//     (b - c) rounded to float is: 1064807300.0f
//   - k_termUpperBound:
//     c = -1 (don't use 0 since that wouldn't account for the staircase effect)
//     (b - c) is: 1065353216 - (-1) = 1065353217
//   - k_termLowerBound:
//     c = 2^23 * (1 - (ln(ln(2)) + 1) / ln(2)) = 722018.66465506613537075079698488
//     round up to fix for the staircase: round(722018.66465506613537075079698488) = 722019
//     (b - c) is: 1065353216 - 722019 = 1064631197
//   - k_termMinimizeRelativeRMSE:
//     c = 2^23 * ln(3 / (8 * ln(2)) + 0.5) / ln(2) - 0.5 = 486411.38068434173315475522228717
//     (b - c) is 1065353216 - 486411.38068434173315475522228717 = 1064866804.6193156582668452447777
//     (b - c) rounded to integer is: 1064866805
//     (b - c) rounded to float is: ?
//   - k_termMinimizeMaximumRelativeError:
//     c = (ln(ln(2) + 2/e) - ln(2) - ln(ln(2))) * 2^23 / ln(2) - 0.5 = 
//        0.03027490056158269559309255065937 * 2^23 / ln(2) - 0.5 = 366392.49729234569298343147736888
//     (b - c) is 1065353216 - 366392.49729234569298343147736888 = 1064986823.5027076543070165685226
//     (b - c) rounded to integer is: 1064986824
//     (b - c) rounded to float is: ?
// 
// Some of these constants can be verified by this independent implementation for float32/int32 here:
// https://github.com/ekmett/approximate/blob/master/cbits/fast.c
// 1065353216 + 1      = 1065353217 for the upper bound
// 1065353216 - 486411 = 1064866805 for the minimize RMSE
// 1065353216 - 722019 = 1064631197 for the lower bound
//
// We use summary statistics, and we want to ensure that our summary statistics don't add any bias which would
// miscalibrate our predictions.  EBMs are in general well calibrated, so don't destroy this property with
// an approximate exp function that tilts one way or the other in the updates.  To generate an update we first
// sum the results of the sigmoid/softmax function (binary vs multiclass) and then divide by the sums of the
// denominator terms.  The denominator terms are derivates that we don't care about too much since they won't
// affect the sign of the numerator that we use to determine the direction a logit is getting updated.  So, essentially
// our primary objective is that our sigmoid function has no positive/negative bias for expected inputs.
// We probably can't calculate what our 'c' value should be theoretically, but some trial and error can get
// us a good value.  Note, that minimizing mean relative error does NOT lead to minimizing positive/negative bias
//
// the relative error from our approximate exp function has a periodicity of ln(2), so [0, ln(2)) should have the 
// same relative error as [ln(2), 2 * ln(2)) and [-ln(2), 0).  We can use this property when testing to optimize
// for various outcomes, BUT our inputs needs to be evenly distributed, so we can't use std::nextafter.  
// We need to increment in precise increments like "ln(2) / 10000".  The period [-ln(2) / 2, +ln(2) / 2) is interesting
// because it concentrates in the region with the most reslution for floating point values and we could get a
// proven symmetric look at the data, and we could iterate with an integer which would ensure we have precise
// negative and positive inputs


// interesting items:
// - according to the Schraudolph paper, if you choose an extreme Schraudolph term, the maximum error for any
//   particular exp value is 6.148%.  We care more about softmax error, and we use floats instead of doubles
// - according to the Schraudolph paper, if you choose an ideal Schraudolph term, the abolute value of teh average 
//   error for any particular exp value is 1.483%.  Again, we care more about softmax error and we use floats
//   but my experience is that we're in the same error range
// - for boosting, we sum the residual terms in order to find the update direction (either positive or negative).  
//   As long as the direction has the correct sign, boosting will tend to make it continue to go in the right 
//   direction, even if the speed has a little variation due to approximate exp noise.  This means that our primary 
//   goal is to balance the errors in our softmax terms such that the positive errors balance the negative errors.
// - the 2nd order derivate denominator Newton-Raphson term is summed separately and then divided by the sum of the
//   residuals.  It affects the speed of convergence, but not the direction.  We choose to optimize our approximate
//   exp to obtain minimum error in the numerator where it will determine the sign and therefore direction of boosting
// - any consistent bias in the error will probably show up as miscalibration.  Miscalibration can lead to us to
//   trigger early stopping at an unideal point, but in general that seems to be a minor concern.  EBMs are well
//   calibrated though, and we'd like to preserve this property.
// - our error is much better controlled if one of the logits is zeroed.  This makes one of the exp(0) outputs equal 
//   to 1, and thus have no error.  This seems to somehow help in terms of normalizing the range of the 
//   schraudolph term required under various input skews compared with letting all the logits range wherever they go
//   and taking the approximate exp of all of them and then softmax
// - the error deviation seems to go down if the non-zeroed logits are all negative.  This makes sense since the
//   exp(0) -> 1 term in the softmax then dominates.  The effect isn't too big though, so having some positive
//   logits isn't the end of the world if it's more efficient to do it that way.  Finding the maximum logit
//   and subtracting that logit from the other logits requires an extra loop and computation, so our implementation
//   tollerates the slight miscalibration introduced instead of the extra CPU work
// - the schraudolph term that achieves average error of zero for the logit that we zero isn't the same 
//   schraudolph term that achieves error of zero for the non-zeroed terms.  They are close however.  If
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
// - at the mid-point skews (-0.509733 and 0.533474) between the 1st and second values from zero we should find 
//   roughly the largest sumation error.  
// - For the skew of -0.509733 our averaged error is only -0.0010472912, and in general you'd expect it to be less 
//   since that's close to the max
// - For the skew of 0.533474 our averaged error is only +0.0012019153, and in general you'd expect it to be less 
//   since that's close to the max
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
//   allows continued improvement), so if there is any skew, it'll just shift the log loss curve upwards, but shouldn't
//   change the minimum point very much.  We also average the log loss accross many samples, so the variance should
//   be reduced by a lot on the averaged term
// - the best way to pick the addSchraudolphTerm would probably be to use benchmarks on real data to fine tune
//   the parameter such that errors are balanced on a massive multi-dataset way

// TODO: do the massive multi-dataset addSchraudolphTerm term optimization using our benchmarks

constexpr double k_expErrorPeriodicity = 0.69314718055994529; // ln(2)

// this constant does not change for any variation in optimizing for different objectives in Schraudolph
constexpr float k_expMultiple = 12102203.0f;

// all of these constants should have exact rounding to float, per IEEE 754 9-12 digit rule (see above)
constexpr uint32_t k_termUpperBound                   = 1065353217; // theoretically justified -> 1065353216 + 1
constexpr uint32_t k_termForDivisionMethodDoNotUse    = 1065353216; // use this for the more accurate exp(x/2)/exp(-x/2) method
constexpr uint32_t k_termMinimizeMaximumRelativeError = 1064986824; // theoretically justified
constexpr uint32_t k_termZeroMeanRelativeError        = 1064870596; // experimentally determined.  This unbiases our average such that averages of large numbers of values should be balanced towards zero
constexpr uint32_t k_termMinimizeRelativeRMSE         = 1064866805; // theoretically justified -> 1065353216 - 486411
constexpr uint32_t k_termMinimizeMeanRelativeAbsError = 1064807269; // theoretically justified -> 1065353216 - 545947
constexpr uint32_t k_termLowerBound                   = 1064631197; // theoretically justified -> 1065353216 - 722019


// experimentally determined softmax Schraudolph terms for softmax where one logit is zeroed by subtracing it from the other logits
constexpr uint32_t k_termZeroMeanRelativeErrorSoftmaxThreeClassesZeroingSkewMinus0_696462 = 1064873067; // experimentally determined, +-1, from -0.696462 - k_expErrorPeriodicity / 2 to -0.696462 + k_expErrorPeriodicity / 2
// mid-point = -0.509733
constexpr uint32_t k_termZeroMeanRelativeErrorSoftmaxThreeClassesZeroingSkewMinus0_323004 = 1064873067; // experimentally determined, +-1, from -0.323004 - k_expErrorPeriodicity / 2 to -0.323004 + k_expErrorPeriodicity / 2
constexpr uint32_t k_termZeroMeanRelativeErrorSoftmaxThreeClassesZeroingSkew0      = 1064873067; // experimentally determined, +-1, from - k_expErrorPeriodicity / 2 to +k_expErrorPeriodicity / 2
constexpr uint32_t k_termZeroMeanRelativeErrorSoftmaxThreeClassesZeroingSkewPlus0_370957 = 1064873067; // experimentally determined, +-?, from 0.370957 - k_expErrorPeriodicity / 2 to 0.370957 + k_expErrorPeriodicity / 2
// mid-point = 0.533474
constexpr uint32_t k_termZeroMeanRelativeErrorSoftmaxThreeClassesZeroingSkewPlus0_695991 = 1064873067; // experimentally determined, +-?, from 0.695991 - k_expErrorPeriodicity / 2 to 0.695991 + k_expErrorPeriodicity / 2

constexpr uint32_t k_termZeroMeanRelativeErrorSoftmaxThreeClassesZeroingRange20 = 1064872079; // experimentally determined, +-1, the zeroed class, from -10 * k_expErrorPeriodicity / 2 to +10 * k_expErrorPeriodicity / 2
// USE THIS Schraudolph term for softmax.  It gives 0.001% -> 0.00001 average error, so it's unlikley to miscalibrate 
// the results too much.  Keeping all softmax terms without zeroing any of them leads to an error of about 0.16% or 
// thereabouts, which is much larger than anything observed for zeroed logits
// 
// this value is the average value between k_termZeroMeanRelativeErrorSoftmaxThreeClassesZeroingRange20 and
// k_termZeroMeanRelativeErrorSoftmaxThreeClassesZeroingOtherRange20, which should be approximately correct IF
// the zeroed logit class has roughly the same number of samples as the non-zeroed classes combined.  We don't 
// know the sample class makeup though (and building a table for various sample class skews would be time consuming)
// so taking the average is a pretty good guess, especially since the upper and lower values are fairly close
// 
// this Schraudolph term works relatively well for softmax with 3+ classes, different skews, and if the predicted
// class is the zeroed class or a non-zeroed class.  It also works fairly well if the non-zeroed logits are greater
// than zero after shifting them
constexpr uint32_t k_termZeroSoftmaxMeanError = 1064871915; // experimentally determined, AVERAGED between the zeroed and non-zeroed class, from -10 * k_expErrorPeriodicity / 2 to +10 * k_expErrorPeriodicity / 2
constexpr uint32_t k_termZeroMeanRelativeErrorSoftmaxThreeClassesZeroingOtherRange20 = 1064871750; // experimentally determined, +-1, the non-zeroed class, from -10 * k_expErrorPeriodicity / 2 to +10 * k_expErrorPeriodicity / 2

// TODO: we can probably pick a better k_termZeroSoftmaxMeanError for binary classification where we know for sure
// the number of classes, and we know the likely sign of the value going into the exp function because we flip
// the sign of the logit based on the true value, and boosting ensures that we'll be trying to push that value
// in a consistent direction.

// TODO: we can probably pick a better k_termZeroSoftmaxMeanError for log loss, although that's less of a prioirty
// since for log loss consistent errors don't affect our early stopping since it just shifts the entire curve
// in one direction.  It would be nice to get as accurate a log loss as possible though for reporting purposes.

constexpr uint32_t k_termZeroMeanRelativeErrorSoftmaxTwoClassesZeroingSkew0 = 1064873955; // experimentally determined, +-?
//constexpr uint32_t k_termZeroMeanRelativeErrorSoftmaxTwoClassesZeroingSkew0_370? = 1064873955; // experimentally determined, +-?
constexpr uint32_t k_termZeroMeanRelativeErrorSoftmaxTwoClassesZeroingSkew0_69982536866359447004608294930875 = 1064873955; // experimentally determined, +-?


// DO NOT USE.  This constant minimizes the mean error for softmax where none of the logits are zeroed.  It has
// too much variability in the output though to be useful when compared against the softmax that zeroes one of the logits
constexpr uint32_t k_termZeroMeanRelativeErrorSoftmaxThreeClassesSkew0 = 1064963329; // experimentally determined +-1, non-zeroed, from -k_expErrorPeriodicity / 2 to k_expErrorPeriodicity / 2
// using non-zeroed softmax isn't as viable since at modest skews (I tried k_expErrorPeriodicity / 4), achieving zero average error is impossible
// since all valid addition terms from k_termLowerBound to k_termUpperBound can result in non-zero error averages
// softmax with a zeroed term seems to have much tighter bounds on the mean error even if the absolute mean error is higher






// k_expUnderflowPoint is set to a value that prevents us from returning a denormal number. These approximate function 
// dont't really work with denomals and starts to drift very quickly from the true exp values when we reach denormals.
constexpr float k_expUnderflowPoint = -87.25f; // this is exactly representable in IEEE 754
constexpr float k_expOverflowPoint = 88.5f; // this is exactly representable in IEEE 754

// use the integer version for now since in non-SIMD this is probably faster and more exact
#define EXP_INT

template<
   bool bNaNPossible = true,
   bool bUnderflowPossible = true,
   bool bOverflowPossible = true,
   bool bSpecialCaseZero = false,
   typename T
>
INLINE_ALWAYS T ExpApproxSchraudolph(const T val, const uint32_t addSchraudolphTerm = k_termZeroSoftmaxMeanError) {
   // This function guarnatees non-decreasing monotonicity, so it never decreases with increasing inputs, but
   // it can sometimes yield equal outputs on increasing inputs

   EBM_ASSERT(k_termLowerBound <= addSchraudolphTerm);
   EBM_ASSERT(addSchraudolphTerm <= k_termUpperBound);

   // if T val is a double, then we need to check before converting to a float if we're in-bounds, since converting
   // a double that is outside the float range into a float results in undefined behavior
   const bool bSetNaN = bNaNPossible && UNPREDICTABLE(std::isnan(val));
   const bool bSetZero = bUnderflowPossible && UNPREDICTABLE(val < T { k_expUnderflowPoint });
   const bool bSetInfinity = bOverflowPossible && UNPREDICTABLE(T { k_expOverflowPoint } < val);
   const bool bSetOne = bSpecialCaseZero && UNPREDICTABLE(T { 0 } == val);

   // if val is a NaN, or would result in an underflow or overflow, we set floatVal to zero below in order to avoid 
   // getting undefined behavior further down.  The following things we do below would invoke undefined behavior 
   // otherwise:
   //   - converting a large double into a float that can't be represented
   //   - converting a large float to an int that can't be represented
   //   - converting +-infinity to an int
   //   - converting a NaN to an int
   // https://stackoverflow.com/questions/10366485/problems-casting-nan-floats-to-int
   // https://docs.microsoft.com/en-us/cpp/c-language/conversions-from-floating-point-types?view=msvc-160
   //
   // The compiler should be smart enough to notice that we overwrite the final result with specific values
   // at the end and the zero set below has no effect, so can be eliminated if any of these conditions are true
   //
   // We use branchless operations here for future SIMD conversion.
   float floatVal = bSetNaN || bSetZero || bSetInfinity ? 0.00000000f : static_cast<float>(val);

#ifdef EXP_INT

   // this version does the addition in integer space, so it's maybe faster if there isn't a fused multiply add
   // instruction that works on floats since the integer add will not be slower than a float add, unless the ALU
   // has some transfer time or time to swtich integer/float values.  Integers in the range we're expecting also
   // have a little more precision, which means we can get closer to the ideal mean relative error constant

   const uint32_t retInt = static_cast<uint32_t>(k_expMultiple * floatVal) + addSchraudolphTerm;
#else

   // this version might be faster if there's a cost to switching between int to float.  Fused multiply add is either 
   // just as fast as plain multiply, or pretty close to it though, so throwing in the add might be low cost or free
   const uint32_t retInt = static_cast<uint32_t>(k_expMultiple * floatVal + static_cast<float>(addSchraudolphTerm));
#endif

   float retFloat;

   // It's undefined behavior in C++ (not C though!) to use a union or a pointer to bit convert between types 
   // using memcpy though is portable and legal since C++ aliasing rules exclude character pointer copies
   // Suposedly, most compilers are smart enough to optimize the memcpy away.

   static_assert(std::numeric_limits<float>::is_iec559, "This hacky function requires IEEE 754 binary layout");
   static_assert(sizeof(retInt) == sizeof(retFloat), "both binary conversion types better have the same size");
   memcpy(&retFloat, &retInt, sizeof(retFloat));

   T ret = static_cast<T>(retFloat);

   ret = UNPREDICTABLE(bSetZero) ? T { 0 } : ret;
   ret = UNPREDICTABLE(bSetInfinity) ? std::numeric_limits<T>::infinity() : ret;
   ret = UNPREDICTABLE(bSetOne) ? T { 1 } : ret;

   // do the NaN value last since we then avoid issues with weird NaN comparison rules
   // if the original was a NaN, then return that exact nan value.  There are many possible NaN values 
   // and they are sometimes used to signal conditions, so preserving the exact value without truncation is nice.
   ret = UNPREDICTABLE(bSetNaN) ? val : ret;

   return ret;
}

template<
   bool bNaNPossible = true,
   bool bUnderflowPossible = true,
   bool bOverflowPossible = true,
   bool bSpecialCaseZero = false,
   typename T
>
INLINE_ALWAYS T ExpApproxBest(const T val) {
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

   // TODO: the constants chosen for this function were rounded to float boundaries but we can probably get a more
   //       accurate version by rounding either up or down for each number from their theoretical values.  We could 
   //       randomly try to choose either up or down and examine the squared error on a bunch of exp values that 
   //       we think are most important for us probably numbers in the range (-5 to +5)

   // if T val is a double, then we need to check before converting to a float if we're in-bounds, since converting
   // a double that is outside the float range into a float results in undefined behavior
   const bool bSetNaN = bNaNPossible && UNPREDICTABLE(std::isnan(val));
   const bool bSetZero = bUnderflowPossible && UNPREDICTABLE(val < T { k_expUnderflowPoint });
   const bool bSetInfinity = bOverflowPossible && UNPREDICTABLE(T { k_expOverflowPoint } < val);
   const bool bSetOne = bSpecialCaseZero && UNPREDICTABLE(T { 0 } == val);

   // if val is a NaN, or would result in an underflow or overflow, we set floatVal to zero below in order to avoid 
   // getting undefined behavior further down.  The following things we do below would invoke undefined behavior 
   // otherwise:
   //   - converting a large double into a float that can't be represented
   //   - converting a large float to an int that can't be represented
   //   - converting +-infinity to an int
   //   - converting a NaN to an int
   // https://stackoverflow.com/questions/10366485/problems-casting-nan-floats-to-int
   // https://docs.microsoft.com/en-us/cpp/c-language/conversions-from-floating-point-types?view=msvc-160
   //
   // The compiler should be smart enough to notice that we overwrite the final result with specific values
   // at the end and the zero set below has no effect, so can be eliminated if any of these conditions are true
   //
   // We use branchless operations here for future SIMD conversion.
   float floatVal = bSetNaN || bSetZero || bSetInfinity ? 0.00000000f : static_cast<float>(val);

   const float negativeCorrection = UNPREDICTABLE(0.00000000f <= floatVal) ? 0.00000000f : 1.00000000f;

   floatVal *= 1.44269502f; // 9 digits for most accurate float representation of: 1 / ln(2)

   // TODO: it might be faster to make negativeCorrection an integer and doing the subtraction from the integer
   // cast of floatVal.  To be trickier, we could extract the sign bit without a conditional with 
   // int negativeCorrection = (superHack.m_int >> 31) [get the highest bit of the float, which is the sign]

   // TODO: is there some kind of ceil or floor or something similar that exactly represents this conversion to a positive range

   // this approximation isn't perfect.  Occasionally it isn't monotonic when floatVal wraps integer boundaries
   const float factor = floatVal - (static_cast<float>(static_cast<int32_t>(floatVal)) - negativeCorrection);

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

   // TODO: we can probably achieve slightly better results by tuning the single constant below of
   //       1.01732051e+09f OR 121.274055f to get better summation results similar to Schraudolph, AND/OR we can 
   //       also convert this to a uint32_t value so that we have even more fiddling ability to tune 
   //       the output to average to zero.  For now it's good enough

   const uint32_t retInt = static_cast<uint32_t>(
#ifdef EXP_HIGHEST_ACCURACY
      float { 1 << 23 } * (floatVal + 121.274055f - 1.49012911f * factor + 27.7280235f / (4.84252548f - factor))
#else
      float { 1 << 23 } * floatVal + 1.01732051e+09f - 1.25001090e+07f * factor + 2.32599520e+08f / (4.84252548f - factor)
#endif
   );

   float retFloat;

   // It's undefined behavior in C++ (not C though!) to use a union or a pointer to bit convert between types 
   // using memcpy though is portable and legal since C++ aliasing rules exclude character pointer copies
   // Suposedly, most compilers are smart enough to optimize the memcpy away.

   static_assert(std::numeric_limits<float>::is_iec559, "This hacky function requires IEEE 754 binary layout");
   static_assert(sizeof(retInt) == sizeof(retFloat), "both binary conversion types better have the same size");
   memcpy(&retFloat, &retInt, sizeof(retFloat));

   T ret = static_cast<T>(retFloat);

   ret = UNPREDICTABLE(bSetZero) ? T { 0 } : ret;
   ret = UNPREDICTABLE(bSetInfinity) ? std::numeric_limits<T>::infinity() : ret;
   ret = UNPREDICTABLE(bSetOne) ? T { 1 } : ret;

   // do the NaN value last since we then avoid issues with weird NaN comparison rules
   // if the original was a NaN, then return that exact nan value.  There are many possible NaN values 
   // and they are sometimes used to signal conditions, so preserving the exact value without truncation is nice.
   ret = UNPREDICTABLE(bSetNaN) ? val : ret;

   return ret;
}

// TODO: in the future, see if we can eliminate any of our handling for NaN, underflow, overflow, underflowToZero, overflowToInfinity
template<
   bool bNaNPossible = true,
   bool bUnderflowPossible = true,
   bool bOverflowPossible = true,
   bool bSpecialCaseZero = false,
   typename T
>
INLINE_ALWAYS T ExpForResiduals(const T val) {
#ifdef FAST_EXP
   return ExpApproxBest<bNaNPossible, bUnderflowPossible, bOverflowPossible, bSpecialCaseZero, T>(val);
#else // FAST_EXP
   return std::exp(val);
#endif // FAST_EXP
}

#ifdef FAST_LOG

// TODO: test the fastlog algorithm from the fastlog.h file below
// algorithm from code originally written by Paul Mineiro.  re-implemented here
// https://github.com/etheory/fastapprox/blob/master/fastapprox/src/fastlog.h
// which might have come from this paper:
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.4508&rep=rep1&type=pdf



// possible approaches:
// NOTE: even though memory lookup table approaches look to be the fastest and reasonable approach, we probably want to avoid lookup tables
//   in order to avoid using too much of our cache memory which we need for other things
// https://www.icsi.berkeley.edu/pubs/techreports/TR-07-002.pdf
// https ://tech.ebayinc.com/engineering/fast-approximate-logarithms-part-i-the-basics/
// https://tech.ebayinc.com/engineering/fast-approximate-logarithms-part-ii-rounding-error/
// https://tech.ebayinc.com/engineering/fast-approximate-logarithms-part-iii-the-formulas/


// TODO move this include up into the VS specific parts
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_BitScanReverse64)

template<typename T>
INLINE_ALWAYS unsigned int MostSignificantBit(T val) {
   // TODO this only works in MS compiler.  This also doesn't work for numbers larger than uint64_t.  This has many problems, so review it.
   unsigned long index;
   return _BitScanReverse64(&index, static_cast<unsigned __int64>(val)) ? static_cast<unsigned int>(index) : static_cast<unsigned int>(0);
}

INLINE_ALWAYS FloatEbmType ExpForLogLoss(const FloatEbmType val) {
   // if we're using approximates for the log function, we don't gain any benefit unless we're also using the
   // approximate exp function
   return ExpApproxSchraudolph(val);
}

INLINE_ALWAYS FloatEbmType LogForLogLoss(FloatEbmType val) {
   // TODO: also look into whehter std::log1p has a good approximation directly

   // the log function is only used to calculate the log loss on the valididation set only
   // the log loss is calculated for the validation set and then returned as a single number to the caller
   // it never gets used as an input to anything inside our code, so any errors won't cyclically grow

   // TODO : this only handles numbers x > 1.  I think I don't need results for less than x < 1 though, so check into that.   If we do have numbers below 1, 
   //   we should do 1/x and figure out how much to multiply below

   // for various algorithms, see https://stackoverflow.com/questions/9799041/efficient-implementation-of-natural-logarithm-ln-and-exponentiation

   // TODO: this isn't going to work for us since we will often get vlaues greater than 2^64 in exp terms.  Let's figure out how to do the alternate where
   // we extract the exponent term directly via IEEE 754
   unsigned int shifts = MostSignificantBit(static_cast<uint64_t>(val));
   val = val / static_cast<FloatEbmType>(uint64_t { 1 } << shifts);

   // this works sorta kinda well for numbers between 1 to 2 (we shifted our number to be within this range)
   // TODO : increase precision of these magic numbers
   val = FloatEbmType { -1.7417939 } + (FloatEbmType { 2.8212026 } + (FloatEbmType { -1.4699568 } + (FloatEbmType { 0.44717955 } +
      FloatEbmType { -0.056570851 } *val) * val) * val) * val;
   val += static_cast<FloatEbmType>(shifts) * FloatEbmType {
      0.69314718
   };

   return val;
}
#else // FAST_LOG
INLINE_ALWAYS FloatEbmType ExpForLogLoss(const FloatEbmType val) {
   // if we're using the non-approximate std::log function, we might as well use std::exp as well for the
   // places that we compute Exp for the log loss
   return std::exp(val);
}

INLINE_ALWAYS FloatEbmType LogForLogLoss(const FloatEbmType val) {
   return std::log(val);
   // TODO: also look into whehter std::log1p is a good function for this (mostly in terms of speed).  For the most part we don't care about accuracy 
   //   in the low
   // digits since we take the average, and the log loss will therefore be dominated by a few items that we predict strongly won't happen, but do happen.  
}
#endif // FAST_LOG

#endif // APPROXIMATE_MATH_H