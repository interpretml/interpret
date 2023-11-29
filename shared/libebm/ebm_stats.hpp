// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef EBM_STATS_HPP
#define EBM_STATS_HPP

#include <cmath> // log, exp, etc
#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT
#include "unzoned.h" // INLINE_ALWAYS, LIKELY, UNLIKELY

#include "common.hpp"

#include "ebm_internal.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// TODO: before applying a term score update for classification, check to see that the probability implied by the 
//       logit exceeds 1/number_of_samples in either the positive or negative direction. If it does, then modify 
//       the update so that the probability does not exceed a certainty of 1/number_of_samples, since we 
//       really can't make statements of probability beyond close to that threshold (make this a tunable option)
// TODO: In addition to limiting logits on individual feature bins, we should also limit them on individual samples
//        (make this a tunable option)
// TODO: we should consider having a min_samples, which will eliminate any class with less than that number of 
//       samples, although we'll record the base rate in the model and then we can re-build the tensor with the 
//       base rate in any classes that we removed (make it an optional tunable value)
// TODO: we can also add new classes at prediction time.. they just get 1/n_samples (make it an optional tunable value)

static_assert(
   std::numeric_limits<double>::is_iec559, 
   "If IEEE 754 isn't guaranteed, then we can't use or compare infinity values in any standard way. "
   "We can't even guarantee that infinity exists as a concept."
);

static constexpr FloatCalc k_hessianMin = std::numeric_limits<FloatCalc>::min();
static constexpr FloatCalc k_gainMin = 0;

// HANDLING SPECIAL FLOATING POINT VALUES (NaN/infinities/denormals/-0):
// - it should be virtually impossible to get NaN values anywhere in this code without being given an 
//   adversarial dataset or adversarial input parameters [see notes below], but they are possible so we need to
//   detect/handle them at some level.
// - it should be virtually impossible to get +-infinity values in classification.  We could get +infinity 
//   values if the user gives us regression targets above sqrt(max_float) 
//   [see notes below], but those regression targets are huge and the caller should just avoid them.  
//   We can't avoid the possibility of the user passing us ridiculously huge regression targets, 
//   and huge inputs are something that has no perfect solution.  We just need to figure out how to 
//   deal with these if +-infinities occur.
// - for classification, if we forcibly limit the logits to a positive and negative range defined by 1/n_samples or
//   some constant min/max values, we should avoid +-infinity (this is not implemented at the time of this writing)
// - we don't crash or really cause any problems if denormals are rounded down to zero.  For classification, any
//   denormal value would be indistinguishable from zero in term of the differences in probabilities, so there is
//   no benefit in attempting to preserve denormals.  In theory, for regression a caller might care about numbers
//   in the denormal range.  We don't need to actively zero denormals for regression though, so we'll just ignore
//   this issue and if they occur we'll just let them propagate if the hardware allows.
// - Our approximate exp/log functions zero denormals since they make things slower and aren't useful in the
//   domain of classification where they are used. We also like getting consistent results between environments, 
//   and zeroing denormals is more standard since not all hardware supports denormals (although is it very common today)
// - negative 0 is a no-op for us.  We don't care if -0 becomes +0 in our code.  It matters though in our floating
//   point approximate exp/log functions, so we check for it there.
// - our strategy, for reasons described below, is to allow NaN and +-infinity values to propagate through our 
//   computations.  We will have a single check for these almost-illegal values at the end.  We terminate all 
//   boosting if we see any NaN or +-infinity values.  In the future this might change so that we limit the logit
//   values to a range that makes sense, which would allow boosting to continue for classification beyond where
//   +-infinities would otherwise be generated.
// - we ensure that any NaN or +-infinity values DO NOT CRASH the program, or cause an infinite loop, but 
//   beyond that we give no guarantees.  The caller can see NaN, +-infinity, or simply wrong values if they give 
//   us adversarial inputs, since the cost of handling them is too high and not even guaranteed to work.
//
// - most compilers have flags that give unspecified behavior when either NaN or +-infinity are generated, 
//   or propagated, or in conditional statements involving them.  Our code can't tolerate these switches being
//   enabled since we can sometimes generate these values, and in general we need to accept NaN values from our caller.
// - g++ makes std::isnan fail to work when the compiler flags indicate no NaN values are present, so we don't use 
//   "-ffast-math".  -fno-finite-math-only would solve this, but we still want NaN comparisons to work outside of
//   just std::isnan for NaN propagation.
// - given our need to allow NaN and +-infinity values, we can then also use NaN and infinity propagation to avoid 
//   checking individual samples for overflows, and we propagate the overflows to the final result which we check.
//   We also use some of the odd properties of NaN comparison that rely on a high degree of compliance with IEEE-754
// - most compilers have switches that allow the compiler to re-order math operations.  We've found issues with
//   these switches, so our code needs to be compiled with strict adherence to floating point operations.  Once
//   we found a bug where the compiler was optimizing "log(x) + log(y)" into "log(x * y)", which then overflowed.
// - Generally +-infinity are treated better than NaN under the compiler switches that make floating points 
//   less IEEE 754 compliant
// - many compilers seem to virtually require the switches enabling floating point deviations from IEEE-754 in 
//   order to enable SIMD, since they want to re-order things like addition instructions.
//   (reordering additions is not IEEE-754 compliant).  Since we can't accept IEEE-754 compiler re-orderings
//   we need to do our own SIMD optimizations instead of allowing the compiler to auto-vectorize
// - given that we are very CPU bound, we would REALLY like our code to be as SIMD friendly as possible 
//   (possibly about 16x faster for AVX-512 using float32)
// - any branching instructions inside a loop will destroy the ability to generate SIMD instructions for the loop.  
//   Non-branching conditional statements in SIMD are allowed via bit masks.
// - since we want SIMD, we need to avoid branching instructions that exit the loop, so if we did want to handle 
//   special floating point values, we'd need to either signal an error on any of the computational values and exit
//   the loop for all, or use NaN or infinity propagation to take the error to the end of the loop
// - old hardware used to have penalties of hundreds of CPU cycles for handling: NaN, +-infinity, and denomral 
//   values.  This has generally been resolved on newer hardware.
//   https://www.agner.org/optimize/nan_propagation.pdf
// - even if there are huge penalties for handling special numeric values, we can stop boosting after a single 
//   completed round if they occur and get propagated, and a several hundred CPU cycle penalty will generally be 
//   less than the total time of doing the rest of the training, which is generally at least dozens of 
//   features and hundreds of boosting rounds.
// - for Intel x86, we enable SSE for floating point instead of x87, since x87 has performance problems and 
//   rounding inconsistencies (they use 80 bit floating point values in the registers)
// - there are C/C++ ways of turning floating point exceptions on or off, but most systems don't do that.  We'll 
//   let our caller choose if those exceptions are enabled, and if so they can at least detect the exceptions.  
//   If we wanted to specify that exceptions would not occur, then we could use the special comparison functions 
//   designed to emulate quiet NaN behavior: std::islessequal, etc..
// - even though the is_iec559 static_assert doesn't correctly indicate when IEEE 754 is broken due to compiler 
//   flags, we should check it anyways because if it's false then we could have even more crazy semantics regarding 
//   floating point numbers, such as no infinities, etc.
// - in practice, we can't rely on the compiler having IEEE-754 compatibility, even if is_iec559 is true.  I've
//   personally seen deviations from correct rounding in float to integer conversion.  Some compilers use the 
//   Intel x87 registers which have non-conformant results, etc.  Here are some other compiler examples of deviations:
//   https://randomascii.wordpress.com/2013/07/16/floating-point-determinism/
// - If IEEE-754 was properly followed (which it isn't), floating point operations would have these properties:
//   - SEE reference https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
//   - these operations would have exact outcomes, guaranteed to be rounded correctly to the last bit: 
//     addition, subtraction, multiplication, division and square root in all versions of IEEE-754 (IEEE-754-1985)
//     we don't use sqrt though as we return MSE as our metric for regression
//   - exp and log would be guaranteed to be correctly rounded to the last bin in IEEE-754-2008 and later
//     https://en.wikipedia.org/wiki/IEEE_754
//   - background: https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
//   - BUT, if we use approximate exp and log functions, we could guarantee identical results, at least in 
//     theory in all versions of IEEE-754.  This does not seem to occur in practice, and I believe the culprit is
//     the rounding between floating point and integers is slightly different, at least on VS compilers.
//   - floating point integers (1.0, 2.0, 3.0, etc) have exact representations up to very large numbers where 
//     the jumps between floating point numbers are greater than 1
//   - fractions of a power of 2 (1/2.0, 1/4.0, 1/8.0, etc) have exact representations down to very small numbers
//
// IEEE 754 special value operations:
//   denormal values don't get rounded to zero
//   NaN values get propagated in operations
//   anything compared to a NaN, is false, except for "a != a" (where a is NaN), which is true! 
//        Take note that "a == a" is FALSE for NaN values!!!
//   log(-ANYTHING) = NaN
//   +infinity + -infinity = NaN
//   -infinity + +infinity = NaN
//   -infinity - -infinity = NaN
//   +infinity - +infinity = NaN
//   +-infinity * +-0 = NaN
//   +-0 * +-infinity = NaN
//   +-0 / +-0 = NaN
//   +-infinity / +-infinity = NaN
//   +infinity + +infinity = +infinity
//   +infinity - -infinity = +infinity
//   -infinity + -infinity = -infinity
//   -infinity - +infinity = -infinity
//   +-infinity * +-infinity = +-infinity
//   +-FINITE * +-infinity = +-infinity
//   +-NON_ZERO / +-0 = +-infinity
//   +-FINITE / +-infinity = +-0
//
// WHY NaN or infinity values can't happen in our code, except it truly exceptional, almost adversarial conditions:
// - once we get a NaN value in almost any computation, even for a single sample's score, it'll 
//   spread through our system like a wildfire making everything NaN
// - once we get an infinity value in almost ANY computation, even for a single sample's score, 
//   it'll turn into a NaN soon, and then everything will become a NaN
// - so, to evaluate the possibility of NaN or infinity values, we need to look everywhere that could generates 
//   such values [see table of operations above that can these values]
// - Generally, the reason why in practice we never see NaN or infinity values is that the boosting algorithm 
//   tends to push errors towards zero, and in the places where it's pushing towards infinity, like for exp(logits), 
//   it does so slowly enough that we shouldn't reach infinity in semi-reasonable numbers of boosting rounds. 
//   We can also further dissuade this from occurring by clipping the logits to reasonable positive or negative values
//   that don't exceed the 1/n_samples ranges where we can't really make any kinds of predictions anyways.
// - looking purely at operators, NaN or infinity values in our code could be introduced in these places:
//   - In regression, if the user passes us NaN or +-infinity as one of the regression targets.  This is 
//     almost the only legitimate way to reach +-infinity or NaN.  If this occurs, we'll get a mean squared 
//     error of +infinity on our FIRST attempt at boosting.  We can detect the overflow of +infinity in our 
//     caller and alert the user to the overflow.  This will also cover the case where mean squared error 
//     overflows more gradually during the training.  Even though this is a legitimate overflow, the user 
//     really needs to do something to handle the condition since there isn't really much that we can do with 
//     such large numbers, so pushing the error back to them is the best solution anyways.
//     If we really wanted to, we could eliminate all errors from large regression targets by limiting the 
//     user to a maximum regression target value (of 7.2e+134) if FLOAT is a float64 and much less for float32
//   - In regression, if the user gives us regression targets (either positive or negative) with absolute 
//     values greater than sqrt(max_float), or if several rounds of boosting 
//     leads to large square errors, then we'll overflow to +infinity in 
//     ComputeSingleSampleSquaredErrorRegressionFromGradient. This won't infect any of our existing training sample 
//     scores, or term graphs, but it will lead to us returning to our caller a +infinity mean squared 
//     error value.  The normal response to any big mean squared error would be for our caller to terminate 
//     via early stopping, which is a fine outcome.  If the caller ignores the +infinity mean squared error, 
//     then they can continue to boost without issues, although all the validation scores will end up as NaN.
//     Our caller can detect the +infinity means squared error themselves and optionally throw a nice 
//     exception plus a message for the end user so that they understand what happened.  The user just needs to 
//     not give us such high regression targets.  If we really wanted to, we could eliminate this error by 
//     limiting the user to a maximum regression target value.  The maximum value would be:
//     sqrt(max_float) / 2^64 = 7.2e+134 for float64 and much less for float32.  
//     Since there must be less than 2^64 samples, the square of that number occurring 2^64 times won't overflow
//     a double
//   - In regression, if the user gives us regression targets (either positive or negative) with values below 
//     but close to +-max_float, the sumGradient can reach +-infinity 
//     since they are a sum.  After sumGradient reaches +-infinity, we'll get a graph update with a 
//     +infinity, and some samples with +-infinity scores.  Then, on the next feature that we boost on, 
//     we'll calculate a term score update for some samples inside ComputeSinglePartitionUpdate 
//     as +-infinity/sumHessian, which will be +-infinity (of the same sign). Then, when we go to calculate 
//     our new sample scores, we'll subtract +infinity-(+infinity) or -infinity-(-infinity), which will 
//     result in NaN.  After that, everything melts down to NaN.  The user just needs to not give us such 
//     high regression targets.  If we really wanted to, we could eliminate all errors from large regression 
//     targets by limiting the user to a maximum regression target value (of 7.2e+134)
//   - In multiclass, in TransformScoreToGradientMulticlass, we could get either NaN or +infinity if sumExp was 0 
//     in the division "ourExp / sumExp", (NaN occurs if ourExp was also zero). This could in theory only 
//     occur in multiclass if all the logits were very very negative.  e^(very_big_negative_logit) = 0.  
//     This is VERY VERY unlikely in any normal dataset because all the logits would need to be very very negative. 
//     Logits in multiclass move in opposite directions while training, so having all logits very negative 
//     shouldn't occur, except possibly in exceptionally contrived situations with extreme logits at 
//     initialization.  Our higher level caller though should be getting the initialization logits from previous 
//     boosting using this code, so this shouldn't happen if this code doesn't return such values, which is true.
//   - In multiclass, In TransformScoreToGradientMulticlass NaN can come from infinity/infinity, which can happen 
//     if any of the e^logits terms reach close to infinity. Doubles have a max value around 1.8e308, 
//     and ln(1.8e308) = approx +709.79, so if any logit reaches above 709 we could have a problem.
//     If ANY logit in any class reaches this high, we should get such a result.  We should also get this 
//     result if the sum of a couple very big logits overflows, but the sum would require the terms to be in 
//     the same neighborhood of logits with values around 709, like two with 708, etc.
//     BUT, reaching logits in the range of 709 is almost impossible without adversarial inputs since boosting
//     will always be working hard to push the error toward zero on the training set.
//     MORE INFO on why logits of 709 are virtually impossible:
//     For computing the term score update, when gradient is close to zero, at the limit the sums used in the 
//     term score update of sumGradients/sumHessians are mathematically at worst +-1 (before applying the learningRate):
//     Segment_term_score_update = sumGradients / sumHessians => 
//     gradient / [gradient * (1 - gradient)] => gradient / [gradient * (1)] => 1
//     When our predictor is very certain, but wrong, we get gradients like -1, and 1.  In those 
//     cases we have a big numerator (relatively) and a small denominator, so those cases are weighted extra heavily, 
//     and are weighted much higher than +-1.  The overall effect is that we train more on the errors, but 
//     in these cases the movement is opposite to the existing value, so high positive numbers become more 
//     negative, and vise versa, so they act in a restorative way and don't contribute to runaway logits.
//     For wrong predictions: Update = sumGradient / sumHessian => +-1 / [1 * (1 - 1)] => 
//     +-1 / 0 => +-infinity but the algorithm will fight against +-infinity since it won't let the gradient 
//     values get too large before it aggressively corrects them back in the direction of zero.  If the logits 
//     are traveling in the correct direction, then the logit updates will go down to +-1.  Then, we apply the 
//     learningRate, which is by default 0.01.  If those values hold, then the post-learningRate update is a 
//     maximum of +-0.01, and it will take around 70,900 rounds of boosting for the logit to reach the 709 
//     neighbourhood.  Our default number of rounds is 5000, so we won't get close under normal parameters.
//   - For multiclass, in theory, if we got a big number of +-infinity as a logit update, we could 
//     overflow a section of the graph. We'll hit overflows inside TransformScoreToGradientMulticlass though 
//     first when the logits get to 709-ish numbers, so it isn't really possible for the logits themselves
//     to overflow first to +-infinity in the graphs for multiclass
//   - For binary and multiclass classification, log loss can reach infinity if we are very certain of one or 
//     more predictions which are incorrectly predicted in the validation set.  This is almost impossible to 
//     occur in real datasets, since log loss will be going up before we reach infinity, so if our caller is 
//     using early stopping, then they'll terminate the learning beforehand.
//     It could happen though if the caller doesn't have early stopping, or specifies an exact number of 
//     boosting rounds.  In any case, we might want to limit probabilities to within 1/number_of_samples since 
//     we want to avoid scenarios that completely blow up and exit early
//   - For binary and multiclass classification, ComputeSinglePartitionUpdate has a 
//     division that can overflow to infinity or NaN values making the logits in our graph infinity or NaN.
//   - for any learning, adversarial inputs like ridiculously huge learningRate parameters or NaN values 
//     could cause these conditions
//   - In InverseLinkFunctionThenCalculateGradientBinaryClassification, the exp term could get to zero or +infinity, but division 
//     by infinity leads to zero, so this shouldn't propagate the infinity term, or create new NaN values.  
//     Even if our logits reached +-infinity InverseLinkFunctionThenCalculateGradientBinaryClassification should 
//     return valid numbers
//   - if our boosting code gets passed NaN or +-infinity values as predictor scores 
//     (regression predictions or logits)  Normally, our higher level library (python, R, etc..) calculates 
//     the predictor scores based on prior boosting rounds from within this core library.  So we shouldn't 
//     get such extreme values unless this C++ code generates NaN, or +-infinities, 
//     and we've shown above that this is very unlikely for non-adversarial inputs.
// - Even before our first e^logit reaches +infinity, we might get a scenario where none of the e^logits 
//   are infinity yet, BUT the sum reaches infinity and in that case we'll get BIG_NUMBER/infinity, 
//   which will be zero, which is the opposite of the mathematical result we want of BIG_NUMBER/BIG_NUMBER, 
//   which should be 1, and it will therefore drive that gradient and logit update in the wrong direction, 
//   so if we ever see an infinity in any multiclass denominator term, we're already in a bad situation.  
//   There is a solution where we could subtract a constant from all the multiclass logits, or make the highest
//   one equal to zero and keep the rest as negative values, which is mathematical no-op, but... why?
//
// - ComputeSinglePartitionUpdate will end with a divide by zero if the hessian is zero.
//   The hessian can be zero if the gradient is +-1.  For binary classification, if 1 + ? = 1, due to 
//   numeracy issues, then the denominator term in InverseLinkFunctionThenCalculateGradientBinaryClassification will be one, 
//   leading to numeracy issues later on due to the 1.  Epsilon is the smallest number that you can 
//   add to 1 and get a non-1. For doubles epsilon is 2.2204460492503131e-016.  e^-36.043 and with an 
//   update rate of 0.01, that means around 3604 rounds of boosting.  We're still good given that our default 
//   is a max of 5000 boosting rounds.  Multiclass has the same issues in summing the e^logit values.
// - there is an asymmetry between small logits and large ones.  Only small ones lead to a problem because 
//   for 1 + ?, the ? needs to be small so that the result is 1.  With big values we get a 1/big_value, which is 
//   ok since there is more resolution for floating points around zero.  For binary classification we can 
//   flip the sign of the logits and make large ones small and small ones large somehow since they are the reciprocal,
//   but we'd rather not add more computation for that.
// - after we've decided on a term score update, we can check to see what the probability predicted would be with 
//   just this 1 feature (we'd need to know the intercept if we wanted to do this better) and if that probability 
//   was beyond the 1/number_of_samples or (number_of_samples-1)/number_of_samples probability range, then we can 
//   be pretty confident that the algorithm is being wildly optimistic, since we have insufficient data to 
//   determine a probability that extreme.  So, we can look at the existing feature logits, and calculate if 
//   any individual bin logit plus the logit update exceeds that probability, and then we can trim the update 
//   logit such that the total never goes above that maximum value.  This isn't really for avoiding +-infinity or NaN
//   values completely, but it helps.  This is to ensure that we don't terminate the algorithm prematurely 
//   for a single case that isn't being predicted well due to runaway logits in the training set where there are 
//   no samples of a particular class in a bin.  By limiting beyond 1/number_of_samples probabilities
//   we won't get completely wild in predicting samples in the validation set.
// - for binary classification, we can just check that the absolute value of the logit doesn't exceed 
//   either +-std::log(1/sumHessian), and if it does we can just reduce it to that value.
// - for multiclass, the correction is a bit harder.  We'd prefer not needing to calculate the true probability 
//   of each class, but we can do an approximate job in logit space.  First off, there can only be one class 
//   with a big probability at a time, so we first need to find out if any item exceeds (sumHessian - 1) / sumHessian 
//   probability.  For the biggest probability, the only two values that matter are the biggest and second 
//   biggest logits.  Reduce the biggest probability so that it doesn't exceed the second biggest probability 
//   by std:log(1/sumHessian).  There can be unlimited items with probabilities that are too low.  After fixing 
//   the biggest probability, we need to check to see if any of these are too low.  If any are lower than 
//   std:log(1/sumHessian) from the max logit, the raise them to that level.
// - even though multiclass logits won't average to zero, we can be somewhat confident that none of them will 
//   change by more than learingRate logits per boosting round, so they can't diverge to more than 709 logits 
//   difference for at least 70,900/2=35,400 rounds.  That's still acceptable.
// - probably the first place we'll get an issue is for multiclass, where we might get one logit that 
//   causes e^logit to be large enough that adding all the other e^logit values doesn't add anything to 
//   the higher e^logit value.  Once that happens, then we'll get e^logit/e^logit = 1, then our... TODO complete
// - from conversation with Rich, we don't really know how to do Laplacian smoothing on boosted decision trees, 
//   and it's done post process anyways, so wouldn't help during tree building.  It would apply a skew to the 
//   updates if used during training.
// - EVEN with all of these precautions though, a malicious user could create a training set such that 
//   many individual features could have high logits, and then they could create an exceptional validation 
//   sample that was the opposite of what's in the training data for every such feature.  As we add the 
//   logits together we could then reach a number greater than 709, and overflow double at e^709.
// - to guard against adversarial datasets like this, we can just accept that we might get +-infinity 
//   and then NaN, and just report those extreme results back to the caller, or we could put in a 
//   per-sample logit maximum.  Checking for these extreme logits per sample seems unreasonable given that 
//   it's hard to imagine this happening on a real dataset, so for now we'll just accept that we might 
//   hit +-infinity, or NaN and then report odd results.  Also, getting to this point would require 
//   70,900 rounds of boosting!
// - when we have a bin with zero samples of a particular class, we might get some interesting results.  This needs
//   more exploration (TODO, explore this)
// - In the future (TODO: this needs to be completed) we're going to only update a term when 
//   it improves the overall log loss, so bins with low counts of a class won't be able to hijack the algorithm 
//   as a whole since the algorithm will just turn off terms with bad predictions for the validation set
// - useful resources:
//   - comparing floats -> https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
//   - details on float representations -> https://www.volkerschatz.com/science/float.html
//   - things that are guaranteed and not -> https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
//
// Other issues:
//   - if we have large numbers of samples, around 2^53 (lookup the exact number), then we can get to the point
//     were "N + 1" results in "N", since the next expressible number above "N" is "N + 2".  We can get to this
//     point with sufficient numbers of samples in the histogram generator.  To combat this, if we break the
//     histogram generation in separate work items less than a certain size, we can then merge the resulting histograms
//     which would sidestep the issue since we'd never be adding 1 to a number that was fairly small.

namespace EbmStats {
   INLINE_ALWAYS static FloatCalc CalcPartialGain(const FloatCalc sumGradient, const FloatCalc sumHessian) {
      // typically this is not performance critical, unless the caller has a very large number of bins

      // This gain function used to determine splits is equivalent to minimizing sum of squared error SSE, which 
      // can be seen following the derivation of Equation #7 in Ping Li's paper -> https://arxiv.org/pdf/1203.3491.pdf

      EBM_ASSERT(FloatCalc { 0 } < k_hessianMin);
      const FloatCalc partialGain = UNLIKELY(sumHessian < k_hessianMin) ? 
         FloatCalc { 0 } : sumGradient / sumHessian * sumGradient;

      // This function should not create new NaN values, but if either sumGradient or sumHessian is a NaN then the 
      // result will be a NaN.  This could happen for instance if large value samples added to +inf in one bin and -inf 
      // in another bin, and then the two bins are added together. That would lead to NaN even without NaN samples.

      EBM_ASSERT(std::isnan(sumGradient) || std::isnan(sumHessian) || FloatCalc { 0 } <= partialGain);
      return partialGain;
   }

   INLINE_ALWAYS static FloatCalc CalcPartialGainFromUpdate(const FloatCalc update, const FloatCalc sumHessian) {

      // the update is: sumGradient / sumHessian
      // For gain we want sumGradient * sumGradient / sumHessian
      // we can get there by doing: update * update * sumHessian
      // which can be simplified as: (sumGradient / sumHessian) * (sumGradient / sumHessian) * sumHessian
      // and then: (sumGradient / sumHessian) * sumGradient
      // finally: sumGradient * sumGradient / sumHessian

      EBM_ASSERT(FloatCalc { 0 } < k_hessianMin);
      const FloatCalc partialGain = UNLIKELY(sumHessian < k_hessianMin) ?
         FloatCalc { 0 } : update * update * sumHessian;

      EBM_ASSERT(std::isnan(update) || std::isnan(sumHessian) || FloatCalc { 0 } <= partialGain);
      return partialGain;
   }


   INLINE_ALWAYS static FloatCalc ComputeSinglePartitionUpdate(
      const FloatCalc sumGradient,
      const FloatCalc sumHessian
   ) {
      // this is NOT a performance critical function.  It only gets called AFTER we've decided where to split, so only a few times per Boosting step

      // for regression, sumGradient can be NaN -> if the user gives us regression targets (either positive or negative) with values below but close to
      //   +-std::numeric_limits<FloatMain>::max(), the sumGradient can reach +-infinity since they are a sum.
      //   After sumGradient reaches +-infinity, we'll get a graph update with a +infinity, and some samples with +-infinity scores
      //   Then, on the next feature that we boost on, we'll calculate a term score update for some samples  
      //   inside ComputeSinglePartitionUpdate as +-infinity/sumHessian, which will be +-infinity (of the same sign). 
      //   Then, when we go to calculate our new sample scores, we'll subtract +infinity-(+infinity) or -infinity-(-infinity), 
      //   which will result in NaN.  After that, everything melts down to NaN.  The user just needs to not give us such high regression targets.
      //   If we really wanted to, we could eliminate all errors from large regression targets by limiting the user to a maximum regression target value 
      //   (of 7.2e+134)

      // for classification, sumGradient can be NaN -> We can get a NaN result inside ComputeSinglePartitionUpdate
      //   for sumGradient / sumHessian if both are zero.  Once one segment of one graph has a NaN logit, then some sample will have a NaN
      //   logit, and InverseLinkFunctionThenCalculateGradientBinaryClassification will return a NaN value. Getting both sumGradient and sumHessian to zero is hard.  
      //   sumGradient can always be zero since it's a sum of positive and negative values sumHessian is harder to get to zero, 
      //   since it's a sum of positive numbers.  The sumHessian is the sum of values returned from this function.  gradient 
      //   must be -1, 0, or +1 to make the denominator zero.  -1 and +1 are hard, but not impossible to get to with really inputs, 
      //   since boosting tends to push errors towards 0.  An error of 0 is most likely when the denominator term in either
      //   InverseLinkFunctionThenCalculateGradientBinaryClassification or TransformScoreToGradientMulticlass becomes close to epsilon.  Once that happens
      //   for InverseLinkFunctionThenCalculateGradientBinaryClassification the 1 + epsilon = 1, then we have 1/1, which is exactly 1, then we subtract 1 from 1.
      //   This can happen after as little as 3604 rounds of boosting, if learningRate is 0.01, and every boosting round we update by the limit of
      //   0.01 [see notes at top of EbmStats.h].  It might happen for a single sample even faster if multiple variables boost the logit
      //   upwards.  We just terminate boosting after that many rounds if this occurs.

      // for regression, sumGradient can be any legal value, including +infinity or -infinity
      //
      // for classification, sumGradient CANNOT be +-infinity-> even with an +-infinity logit, see InverseLinkFunctionThenCalculateGradientBinaryClassification and TransformScoreToGradientMulticlass
      //   also, since -cSamples <= sumGradient && sumGradient <= cSamples, and since cSamples must be 64 bits or lower, we cann't overflow 
      //   to infinity when taking the sum

      // for regression, sumHessian can be NaN -> sumHessian is calculated from gradient.  Since gradient can be NaN (as described above), then 
      //   sumHessian can be NaN

      // for classification, sumHessian cannot be infinity -> per the notes in CalculateHessianFromGradientBinaryClassification

      // since this is classification, -cSamples <= sumGradient && sumGradient <= cSamples
      //EBM_ASSERT(!std::isinf(sumGradient)); // a 64-bit number can't get a value larger than a double to overflow to infinity

      // since this is classification, 0 <= sumHessian && sumHessian <= 0.25 * cSamples (see notes in CalculateHessianFromGradientBinaryClassification), 
      // so sumHessian can't be infinity
      //EBM_ASSERT(!std::isinf(sumHessian)); // a 64-bit number can't get a value larger than a double to overflow to infinity

      // sumHessian can be very slightly negative, for floating point numeracy reasons.  On the face, this is bad because it flips the direction
      //   of our update.  We should only hit this condition when our error gets so close to zero that we should be hitting other numeracy issues though
      //   and even if it does happen, it should be EXTREMELY rare, and since we're using boosting, it will recover the lost round next time.
      //   Also, this should only happen when we are at an extremem logit anyways, and going back one step won't hurt anything.  It's not worth
      //   the cost of the comparison to check for the error condition
      //EBM_ASSERT(std::isnan(sumHessian) || -k_epsilonGradient <= sumHessian);

      // sumGradient can always be much smaller than sumHessian, since it's a sum, so it can add positive and negative numbers such that it reaches 
      // almost zero whlie sumHessian is always positive

      // sumHessian can always be much smaller than sumGradient.  Example: 0.999 -> the sumHessian will be 0.000999 (see CalculateHessianFromGradientBinaryClassification)

      // since the denominator term always goes up as we add new numbers (it's always positive), we can only have a zero in that term if ALL samples have 
      // low denominators
      // the denominator can be close to zero only when gradient is -1, 0, or 1 for all cases, although they can be mixed up between these values.
      // Given that our algorithm tries to drive error to zero and it will drive it increasingly strongly the farther it gets from zero, 
      // it seems really unlikley that we'll get a zero numerator by having a series of
      // -1 and 1 values that perfectly cancel eachother out because it means they were driven to the opposite of the correct answer
      // If the denominator is zero, then it's a strong indication that all the gradients are close to zero.
      // if all the gradients are close to zero, then the numerator is also going to be close to zero
      // So, if the sumHessian is close to zero, we can assume the sumGradient numerator will be too.  At this point with all the gradients very close to zero
      // we might as well stop learning on these samples and set the update to zero for this section anyways, but we don't want branches here, 
      // so we'll just leave it


      // for Gradient regression, 1 < cSamples && cSamples < 2^64 (since it's a 64 bit integer in 64 bit address space app
      //EBM_ASSERT(!std::isnan(sumHessian)); // this starts as an integer
      //EBM_ASSERT(!std::isinf(sumHessian)); // this starts as an integer

      // for Gradient regression, -infinity <= sumGradient && sumGradient <= infinity (it's regression which has a larger range)

      // even if we trim inputs of +-infinity from the user to std::numeric_limits<FloatMain>::max() or std::numeric_limits<FloatMain>::lowest(), 
      // we'll still reach +-infinity if we add a bunch of them together, so sumGradient can reach +-infinity.
      // After sumGradient reaches +-infinity, we'll get an update and some samples with +-infinity scores
      // Then, on the next feature we boost on, we'll calculate an term score update for some samples (inside this function) as 
      // +-infinity/sumHessian, which will be +-infinity (of the same sign).  Then, when we go to find our new sample scores, we'll
      // subtract +infinity-(+infinity) or -infinity-(-infinity), which will result in NaN.  After that, everything melts down to NaN.

      // all the weights can be zero in which case even if we have no splits our sumHessian can be zero

      EBM_ASSERT(FloatCalc { 0 } < k_hessianMin);
      return UNLIKELY(sumHessian < k_hessianMin) ? FloatCalc { 0 } : (-sumGradient / sumHessian);

      // return can be NaN if both sumGradient and sumHessian are zero, or if we're propagating a NaN value.  Neither sumGradient nor 
      //   sumHessian can be infinity, so that's not a source of NaN
      // return can be infinity if sumHessian is extremely close to zero, while sumGradient is non-zero (very hard).  This could happen if 
      //   the gradient was near +-1
      // return can be any other positive or negative number
   }


   INLINE_ALWAYS static FloatCalc ComputeSinglePartitionUpdateGradientSum(const FloatCalc sumGradient) {
      // this is NOT a performance critical call.  It only gets called AFTER we've decided where to split, so only a few times per term boost

      return sumGradient;
   }
};

} // DEFINED_ZONE_NAME

#endif // EBM_STATS_HPP