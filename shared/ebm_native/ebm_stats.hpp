// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef EBM_STATS_HPP
#define EBM_STATS_HPP

#include <cmath> // log, exp, etc
#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT
#include "common_c.h" // INLINE_ALWAYS, LIKELY, UNLIKELY
#include "zones.h"

#include "ebm_internal.hpp" // k_epsilonGradient

#include "approximate_math.hpp" // ExpForBinaryClassification, ...

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

// TODO: we should make this the min() instead of the denorm_min().  We want denormals flushed to zero
// because they either case non-determinism due to flush to zero or they cause performance issues, often.
static constexpr FloatBig k_hessianMin = std::numeric_limits<FloatBig>::denorm_min();
static constexpr FloatBig k_gainMin = 0;

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
//     user to a maximum regression target value (of 7.2e+134) if FloatFast is a float64 and much less for float32
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

class EbmStats final {
public:

   EbmStats() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static FloatFast CalculateHessianFromGradientBinaryClassification(const FloatFast gradient) {
      // this function IS performance critical as it's called on every sample * cClasses * cInnerBags

      // normally you would calculate the hessian from the class probability, but for classification it's possible
      // to calculate from the gradient since our gradient is (r - p) where r is either 0 or 1, and our 
      // hessian is p * (1 - p).  By taking the absolute value of (r - p) we're at a positive distance from either
      // 0 or 1, and then we flip sides on "p" and "(1 - p)".  For binary classification this is useful since
      // we can calcualte our gradient directly in a more exact way (especially when approximates are involved)
      // and then calculate the hessian without subtracting from 1, which also introduces unbalanced floating point
      // noise, unlike the more balanaced approach we're taking here



      // gradient can be NaN -> We can get a NaN result inside ComputeSinglePartitionUpdate
      //   for sumGradient / sumHessian if both are zero.  Once one segment of one graph has a NaN logit, then some sample will have a NaN
      //   logit, and InverseLinkFunctionThenCalculateGradientBinaryClassification will return a NaN value. Getting both sumGradient and sumHessian to zero is hard.  
      //   sumGradient can always be zero since it's a sum of positive and negative values sumHessian is harder to get to zero, 
      //   since it's a sum of positive numbers.  The sumHessian is the sum of values returned from this function.  gradient 
      //   must be -1, 0, or +1 to make the denominator zero.  -1 and +1 are hard, but not impossible to get to with really bad inputs, 
      //   since boosting tends to push errors towards 0.  An error of 0 is most likely when the hessian term in either
      //   InverseLinkFunctionThenCalculateGradientBinaryClassification or TransformScoreToGradientMulticlass becomes close to epsilon.  Once that happens
      //   for InverseLinkFunctionThenCalculateGradientBinaryClassification the 1 + epsilon = 1, then we have 1/1, which is exactly 1, then we subtract 1 from 1.
      //   This can happen after as little as 3604 rounds of boosting, if learningRate is 0.01, and every boosting round we update by the limit of
      //   0.01 [see notes at top of EbmStats.h].  It might happen for a single sample even faster if multiple variables boost the logit
      //   upwards.  We just terminate boosting after that many rounds if this occurs.

      // gradient CANNOT be +-infinity -> even with an +-infinity logit, see InverseLinkFunctionThenCalculateGradientBinaryClassification and TransformScoreToGradientMulticlass

      // for binary classification, -1 <= gradient && gradient <= 1 -> see InverseLinkFunctionThenCalculateGradientBinaryClassification

      // for multiclass, -1 - k_epsilonGradient <= gradient && gradient <= 1 -> see TransformScoreToGradientMulticlass

      // propagate NaN or deliver weird results if NaNs aren't properly propaged.  This is acceptable considering how hard you have to 
      // work to generate such an illegal value, and given that we won't crash
      EBM_ASSERT(
         std::isnan(gradient) || 
         !std::isinf(gradient) && -1 - k_epsilonGradient <= gradient && gradient <= 1
      );

      // this function pre-computes a portion of an equation that we'll use later.  We're computing it now since we can share
      // the work of computing it between inner bags.  It's not possible to reason about this function in isolation though.

      // Here are the responses for various inputs for this function (but this function in isolation isn't useful):
      // -1     -> 0
      // -0.999 -> 0.000999
      // -0.5   -> 0.25
      // -0.001 -> 0.000999
      // 0      -> 0
      // +0.001 -> 0.000999
      // +0.5   -> 0.25
      // +0.999 -> 0.000999
      // +1     -> 0

      // when we use this hessian term retuned inside ComputeSinglePartitionUpdate, if there was only
      //   a single hessian term, or multiple similar ones, at the limit we'd get the following for the following inputs:
      //   boosting is working propertly and we're close to zero error:
      //     - slice_term_score_update = sumGradient / sumHessian => gradient / [gradient * (1 - gradient)] => 
      //       gradient / [gradient * (1)] => +-1  but we multiply this by the learningRate of 0.01 (default), to get +-0.01               
      //   when boosting is making a mistake, but is certain about it's prediction:
      //     - slice_term_score_update = sumGradient / sumHessian => gradient / [gradient * (1 - gradient)] => +-1 / [1 * (0)] => 
      //       +-infinity
      //       but this shouldn't really happen inside the training set, because as the error gets bigger our boosting algorithm will correct corse by
      //       updating in the opposite direction.  Divergence to infinity is a possibility in the validation set, but the training set pushes it's error to 
      //       zero.  It may be possible to construct an adversarial dataset with negatively correlated features that cause a bouncing around that leads to 
      //       divergence, but that seems unlikely in a normal dataset
      //   our resulting function looks like this:
      // 
      //  small_term_score_update
      //          |     *
      //          |     *
      //          |     *
      //          |    * 
      //          |   *  
      //      0.01|*     
      //          |      
      //  -1-------------1--- gradient
      //          |
      //         *|-0.01
      //      *   |
      //     *    |
      //    *     |
      //    *     |
      //    *     |
      //
      //   We have +-infinity asympotes at +-1
      //   We have a discontinuity at 0, where we flip from positive to negative
      //   the overall effect is that we train more on errors (error is +-1), and less on things with close to zero error

      // !!! IMPORTANT: Newton-Raphson step, as illustrated in Friedman's original paper (https://statweb.stanford.edu/~jhf/ftp/trebst.pdf, page 9). Note that
      //   they are using t * (2 - t) since they have a 2 in their objective
      const FloatFast absGradient = std::abs(gradient); // abs will return the same type that it is given, either float or double
      const FloatFast hessian = absGradient * (FloatFast { 1 } - absGradient);

      // - it would be somewhat bad if absGradient could get larger than 1, even if just due to floating point error reasons, 
      //   since this would flip the sign on ret to negative.  Later in ComputeSinglePartitionUpdate, 
      //   The sign flip for ret will cause the update to be the opposite from what was desired.  This should only happen
      //   once we've overflowed our boosting rounds though such that 1 + epsilon = 1, after about 3604 boosting rounds
      //   At that point, undoing one round of boosting isn't going to hurt.

      // the maximum return value occurs when gradient is 0.5 (which is exactly representable in floating points), which gives 0.25
      // 0.25 has an exact representations in floating point numbers, at least in IEEE 754, which we statically check for at compile time
      // I don't think we need an espilon at the top figure.  Given that all three numbers invovled (0.5, 1, 0.25) have exact representations, 
      // we should get the exact return of 0.25 for 0.5, and never larger than 0.25
      // IEEE 754 guarantees that these values have exact representations AND it also guarantees that the results of addition and multiplication
      // for these operations are rounded to the nearest bit exactly, so we shouldn't be able to get anything outside of this range.

      // our input allows values slightly larger than 1, after subtracting 1 from 1 + some_small_value, we can get a slight negative value
      // when we get to this point, we're already in a position where we'll get a NaN or infinity or something else that will stop boosting

      // with our abs inputs confined to the range of -k_epsilonGradient -> 1, we can't get a new NaN value here, so only check against propaged NaN 
      // values from gradient
      EBM_ASSERT(std::isnan(gradient) || !std::isinf(hessian) && -k_epsilonGradient <= hessian && hessian <= FloatFast { 0.25 });

      return hessian;
   }

   INLINE_ALWAYS static FloatBig CalcPartialGain(const FloatBig sumGradient, const FloatBig sumHessian) {
      // typically this is not performance critical, unless the caller has a very large number of bins

      // This gain function used to determine splits is equivalent to minimizing sum of squared error SSE, which 
      // can be seen following the derivation of Equation #7 in Ping Li's paper -> https://arxiv.org/pdf/1203.3491.pdf

      EBM_ASSERT(0 < k_hessianMin);
      const FloatBig partialGain = UNLIKELY(sumHessian < k_hessianMin) ? 
         FloatBig { 0 } : sumGradient / sumHessian * sumGradient;

      // This function should not create new NaN values, but if either sumGradient or sumHessian is a NaN then the 
      // result will be a NaN.  This could happen for instance if large value samples added to +inf in one bin and -inf 
      // in another bin, and then the two bins are added together. That would lead to NaN even without NaN samples.

      EBM_ASSERT(std::isnan(sumGradient) || std::isnan(sumHessian) || 0 <= partialGain);
      return partialGain;
   }

   INLINE_ALWAYS static FloatBig CalcPartialGainFromUpdate(const FloatBig update, const FloatBig sumHessian) {
      // the update is: sumGradient / sumHessian
      // For gain we want sumGradient * sumGradient / sumHessian
      // we can get there by doing: update * update * sumHessian
      // which can be simplified as: (sumGradient / sumHessian) * (sumGradient / sumHessian) * sumHessian
      // and then: (sumGradient / sumHessian) * sumGradient
      // finally: sumGradient * sumGradient / sumHessian

      EBM_ASSERT(0 < k_hessianMin);
      const FloatBig partialGain = UNLIKELY(sumHessian < k_hessianMin) ?
         FloatBig { 0 } : update * update * sumHessian;

      EBM_ASSERT(std::isnan(update) || std::isnan(sumHessian) || 0 <= partialGain);
      return partialGain;
   }


   INLINE_ALWAYS static FloatBig ComputeSinglePartitionUpdate(
      const FloatBig sumGradient, 
      const FloatBig sumHessian
   ) {
      // this is NOT a performance critical function.  It only gets called AFTER we've decided where to split, so only a few times per Boosting step

      // for regression, sumGradient can be NaN -> if the user gives us regression targets (either positive or negative) with values below but close to
      //   +-std::numeric_limits<FloatBig>::max(), the sumGradient can reach +-infinity since they are a sum.
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

      // even if we trim inputs of +-infinity from the user to std::numeric_limits<FloatBig>::max() or std::numeric_limits<FloatBig>::lowest(), 
      // we'll still reach +-infinity if we add a bunch of them together, so sumGradient can reach +-infinity.
      // After sumGradient reaches +-infinity, we'll get an update and some samples with +-infinity scores
      // Then, on the next feature we boost on, we'll calculate an term score update for some samples (inside this function) as 
      // +-infinity/sumHessian, which will be +-infinity (of the same sign).  Then, when we go to find our new sample scores, we'll
      // subtract +infinity-(+infinity) or -infinity-(-infinity), which will result in NaN.  After that, everything melts down to NaN.

      // all the weights can be zero in which case even if we have no splits our sumHessian can be zero
      EBM_ASSERT(0 <= sumHessian);

      return (FloatBig { 0 } == sumHessian) ? FloatBig { 0 } : (-sumGradient / sumHessian);

      // return can be NaN if both sumGradient and sumHessian are zero, or if we're propagating a NaN value.  Neither sumGradient nor 
      //   sumHessian can be infinity, so that's not a source of NaN
      // return can be infinity if sumHessian is extremely close to zero, while sumGradient is non-zero (very hard).  This could happen if 
      //   the gradient was near +-1
      // return can be any other positive or negative number
   }


   INLINE_ALWAYS static FloatBig ComputeSinglePartitionUpdateGradientSum(const FloatBig sumGradient) {
      // this is NOT a performance critical call.  It only gets called AFTER we've decided where to split, so only a few times per term boost

      return sumGradient;
   }

   INLINE_ALWAYS static FloatFast ComputeGradientRegressionMSEInit(const FloatFast sampleScore, const FloatFast target) {
      // this function is NOT performance critical as it's called on every sample, but only during initialization.

      // for MSE regression, the gradient is the residual, and we can calculate it once at init and we don't need
      // to keep the original scores when computing the gradient updates.

      // it's possible to reach NaN or +-infinity within this module, so sampleScore can be those values
      // since we can reach such values anyways, we might as well not check for them during initialization and detect the
      // NaN or +-infinity values in one of our boosting rounds and then terminate the algorithm once those special values
      // have propagaged, which we need to handle anyways

      const FloatFast gradient = sampleScore - target;

      // if target and sampleScore are both +infinity or both -infinity, then we'll generate a NaN value

      return gradient;
   }

   INLINE_ALWAYS static FloatFast ComputeGradientRegressionMSEFromOriginalGradient(const FloatFast originalGradient) {
      // this function IS performance critical as it's called on every sample

      // for MSE regression, the gradient is the residual, and we can calculate it once at init and we don't need
      // to keep the original scores when computing the gradient updates, so we only need the previous gradient

      // originalGradient can be +-infinity, or NaN.  See note in ComputeSinglePartitionUpdate

      // this function is here to document where we're calculating regression, like InverseLinkFunctionThenCalculateGradientBinaryClassification below.  It doesn't do anything, 
      //   but it serves as an indication that the calculation would be placed here if we changed it in the future
      return originalGradient;
   }

   INLINE_ALWAYS static FloatFast InverseLinkFunctionThenCalculateGradientBinaryClassification(
      const FloatFast sampleScore, 
      const size_t target
   ) {
      // this IS a performance critical function.  It gets called per sample!

      // sampleScore can be NaN -> We can get a NaN result inside ComputeSinglePartitionUpdate
      //   for sumGradient / sumHessian if both are zero.  Once one segment of one graph has a NaN logit, then some sample will have a NaN
      //   logit

      // sampleScore can be +-infinity -> we can overflow to +-infinity

      EBM_ASSERT(0 == target || 1 == target);

      // this function outputs 0 if we perfectly predict the target with 100% certainty.  To do so, sampleScore would need to be either 
      //   infinity or -infinity
      // this function outputs 1 if actual value was 1 but we incorrectly predicted with 100% certainty that it was 0 by having 
      //   sampleScore be -infinity
      // this function outputs -1 if actual value was 0 but we incorrectly predicted with 100% certainty that it was 1 by having sampleScore 
      //   be infinity
      //
      // this function outputs 0.5 if actual value was 1 but we were 50%/50% by having sampleScore be 0
      // this function outputs -0.5 if actual value was 0 but we were 50%/50% by having sampleScore be 0

      // TODO : In the future we'll sort our data by the target value, so we'll know ahead of time if 0 == target.  We expect 0 to be the 
      //   default target, so we should flip the value of sampleScore so that we don't need to negate it for the default 0 case
      // TODO: we can probably remove the negation on 1 == target via : return  binned_actual_value - 1 + (1 / (np.exp(training_log_odds_prediction) + 1)) 
      // once we've moved to sorted training data
      // exp will return the same type that it is given, either float or double
      // TODO: for the ApproxExp function, we can change the constant to being a negative once we change to sorting by our target value
      //       then we don't need to even take the negative of sampleScore below

      // !!! IMPORTANT: when using an approximate exp function, the formula used to compute the gradients becomes very
      //                important.  We want something that is balanced from positive to negative, which this version
      //                does IF the classes are roughly balanced since the positive or negative value is
      //                determined by only the target, unlike if we used a forumala that relied
      //                on the exp function returning a 1 at the 0 value, which our approximate exp does not give
      //                In time, you'd expect boosting to make targets with 0 more negative, leading to a positive
      //                term in the exp, and targets with 1 more positive, leading to a positive term in the exp
      //                So both classes get the same treatment in terms of errors in the exp function (both in the
      //                positive domain)
      //                We do still want the error of the positive cases and the error of the negative cases to
      //                sum to zero in the aggregate, so we want to choose our exp function to have average error
      //                sums of zero.
      //                I've made a copy of this formula as a comment to reference to what is good in-case the 
      //                formula is changed in the code without reading this comment
      //                const FloatFast gradient = (UNPREDICTABLE(0 == target) ? FloatFast { -1 } : FloatFast { 1 }) / (FloatFast{ 1 } + ExpForBinaryClassification(UNPREDICTABLE(0 == target) ? -sampleScore : sampleScore));
      // !!! IMPORTANT: SEE ABOVE
      const FloatFast gradient = (UNPREDICTABLE(size_t { 0 } == target) ? FloatFast { 1 } : FloatFast { -1 }) / (FloatFast { 1 } +
         ExpForBinaryClassification<false>(UNPREDICTABLE(size_t { 0 } == target) ? -sampleScore : sampleScore));

      // exp always yields a positive number or zero, and I can't imagine any reasonable implementation that would violate this by returning a negative number
      // given that 1.0 is an exactly representable number in IEEE 754, I can't see 1 + exp(anything) ever being less than 1, even with floating point jitter
      // again, given that the numerator 1.0 is an exactly representable number, I can't see (+-1.0 / something_1_or_greater) ever having an absolute value
      // above 1.0 especially, since IEEE 754 guarnatees that addition and division yield numbers rounded correctly to the last binary decimal place
      // IEEE 754, which we check for at compile time, specifies that +-1/infinity = 0

      // gradient cannot be +-infinity -> even if sampleScore is +-infinity we then get +-1 division by 1, or division by +infinity, which are +-1 
      //   or 0 respectively

      // gradient can only be NaN if our inputs are NaN

      // So...
      EBM_ASSERT(std::isnan(sampleScore) || !std::isinf(gradient) && -1 <= gradient && gradient <= 1);

      // gradient can't be +-infinity, since an infinity in the denominator would just lead us to zero for the gradient value!

#ifndef NDEBUG
      const FloatFast expVal = std::exp(sampleScore);
      FloatFast gradientDebug;
      FloatFast hessianDebug;
      InverseLinkFunctionThenCalculateGradientAndHessianMulticlassForNonTarget(FloatFast { 1 } / (FloatFast { 1 } + expVal), expVal, gradientDebug, hessianDebug);
      if(1 == target) {
         gradientDebug = MulticlassFixTargetGradient(gradientDebug, FloatFast { 1 });
      }
      // the TransformScoreToGradientMulticlass can't be +-infinity per notes in TransformScoreToGradientMulticlass, 
      // but it can generate a new NaN value that we wouldn't get in the binary case due to numeric instability issues with having multiple logits
      // if either is a NaN value, then don't compare since we aren't sure that we're exactly equal in those cases because of numeric instability reasons
      EBM_ASSERT(std::isnan(sampleScore) || std::isnan(gradientDebug) || std::abs(gradientDebug - gradient) < k_epsilonGradientForBinaryToMulticlass);
#endif // NDEBUG
      return gradient;
   }

   INLINE_ALWAYS static void InverseLinkFunctionThenCalculateGradientAndHessianMulticlassForNonTarget(
      const FloatFast sumExpInverted,
      const FloatFast itemExp, 
      FloatFast & gradientOut,
      FloatFast & hessianOut
   ) {
      // this IS a performance critical function.  It gets called per sample AND per-class!

      // trainingLogWeight (which calculates itemExp) can be NaN -> We can get a NaN result inside ComputeSinglePartitionUpdate
      //   for sumGradient / sumHessian if both are zero.  Once one segment of one graph has a NaN logit, then some sample will have a NaN
      //   logit
      
      // trainingLogWeight (which calculates itemExp) can be any number from -infinity to +infinity -> through addition, it can overflow to +-infinity

      // sumExpInverted can be NaN -> sampleScore is used when calculating sumExp, so if sampleScore can be NaN, then sumExp can be NaN

      // sumExpInverted can be any number from 0 to +infinity -> each e^logit term can't be less than zero, and I can't imagine any implementation 
      //   that would result in a negative exp result from adding a series of positive values.
      EBM_ASSERT(std::isnan(sumExpInverted) || 0 <= sumExpInverted);

      // itemExp can be anything from 0 to +infinity, or NaN (through propagation)
      EBM_ASSERT(std::isnan(itemExp) || 0 <= itemExp); // no reasonable implementation should lead to a negative exp value

      // mathematically sumExp must be larger than itemExp BUT in practice itemExp might be SLIGHTLY larger due to numerical issues -> 
      //   since sumExp is a sum of positive terms that includes itemExp, it cannot be lower mathematically.
      //   sumExp, having been computed from non-exact floating points, could be numerically slightly outside of the range that we would otherwise 
      //   mathematically expect.  For sample, if EbmExp(trainingLogWeight) resulted in a number that rounds down, but the floating point processor 
      //   preserves extra bits between computations AND if sumExp, which includes the term EbmExp(trainingLogWeight) was rounded down and then subsequently 
      //   added to numbers below the threshold of epsilon at the value of EbmExp(trainingLogWeight), then by the time we get to the division of 
      //   EbmExp(trainingLogWeight) / sumExp could see the numerator as higher, and result in a value slightly greater than 1!

      EBM_ASSERT(std::isnan(sumExpInverted) || itemExp - k_epsilonGradient <= FloatFast { 1 } / sumExpInverted);

      const FloatFast probability = itemExp * sumExpInverted;

      // probability can be NaN -> 
      // - If itemExp AND sumExp are exactly zero or exactly infinity then itemExp / sumExp will lead to NaN
      // - If sumExp is zero, then itemExp pretty much needs to be zero, since if any of the terms in the sumation are
      //   larger than a fraction.  It is very difficult to see how sumExp could be 0 because it would require that we have 3 or more logits that have 
      //   all either been driven very close to zero, but our algorithm drives multiclass logits appart from eachother, so some should be positive, and
      //   therefor the exp of those numbers non-zero
      // - it is possible, but difficult to see how both itemExp AND sumExp could be infinity because all we need is for itemExp to be greater than
      //   about 709.79.  If one itemExp is +infinity then so is sumExp.  Each update is mostly limited to units of 0.01 logits 
      //   (0.01 learningRate * 1 from InverseLinkFunctionThenCalculateGradientBinaryClassification or TransformScoreToGradientMulticlass), so if we've done more than 70,900 boosting 
      //   rounds we can get infinities or NaN values.  This isn't very likekly by itself given that our default is a max of 2000 rounds, but it is possible
      //   if someone is tweaking the parameters way past their natural values

      // probability can be SLIGHTLY larger than 1 due to numeric issues -> this should only happen if sumExp == itemExp approximately, so there can be no 
      //   other logit terms in sumExp, and this would only happen after many many rounds of boosting (see above about 70,900 rounds of boosting).
      // - if probability was slightly larger than 1, we shouldn't expect a crash.  What would happen is that in our next call to CalculateHessianFromGradientBinaryClassification, we
      //   would find our denomiator term as a negative number (normally it MUST be positive).  If that happens, then later when we go to compute the
      //   small term score update, we'll inadvertently flip the sign of the update, but since CalculateHessianFromGradientBinaryClassification was close to the discontinuity at 0,
      //   we know that the update should have a value of 1 * learningRate = 0.01 for default input parameters.  This means that even in the very very
      //   very unlikely case that we flip our sign due to numericacy error, which only happens after an unreasonable number of boosting rounds, we'll
      //   flip the sign on a minor update to the logits.  We can tollerate this sign flip and next round we're not likely to see the same sign flip, so
      //   boosting will recover the mistake with more boosting rounds

      // probability must be positive -> both the numerator and denominator are positive, so no reasonable implementation should lead to a negative number

      // probability can be zero -> sumExp can be infinity when itemExp is non-infinity.  This occurs when only one of the terms has overflowed to +infinity

      // probability can't be infinity -> even if itemExp is slightly bigger than sumExp due to numeric reasons, the division is going to be close to 1
      //   we can't really get an infinity in itemExp without also getting an infinity in sumExp, so probability can't be infinity without getting a NaN

      EBM_ASSERT(std::isnan(probability) || 
         !std::isinf(probability) && 0 <= probability && probability <= 1 + k_epsilonGradient);


      //const FloatFast yi = UNPREDICTABLE(iScore == target) ? FloatFast { 1 } : FloatFast { 0 };

      // if probability cannot be +infinity, and needs to be between 0 and 1 + small_value, or NaN, then gradient can't be inifinity either

      //const FloatFast gradient = probability - yi;
      const FloatFast gradient = probability; // we will later fix this value for when iScore == target by subtracing 1

      // mathematicaly we're limited to the range of range 0 <= probability && probability <= 1, but with floating point issues
      // we can get an probability value slightly larger than 1, which could lead to -1.00000000001-ish results
      // just like for the division by zero conditions, we'd need many many boosting rounds for probability to get to 1, since
      // the sum of e^logit must be about equal to e^logit for this class, which should require thousands of rounds (70,900 or so)
      // also, the boosting algorthm tends to push results to zero, so a result more negative than -1 would be very exceptional
      EBM_ASSERT(std::isnan(probability) || !std::isinf(gradient) && -1 - k_epsilonGradient <= gradient && gradient <= 1);

      const FloatFast hessian = probability * (FloatFast { 1 } - probability);

      gradientOut = gradient;
      hessianOut = hessian;
   }

   INLINE_ALWAYS static FloatFast MulticlassFixTargetGradient(const FloatFast oldGradient, const FloatFast weight) {
      return oldGradient - weight;
   }

   INLINE_ALWAYS static FloatFast ComputeSingleSampleLogLossBinaryClassification(
      const FloatFast sampleScore, 
      const size_t target
   ) {
      // this IS a performance critical function.  It gets called per validation sample!

      // we are confirmed to get the same log loss value as scikit-learn for binary and multiclass classification

      // trainingLogWeight can be NaN -> We can get a NaN result inside ComputeSinglePartitionUpdate
      //   for sumGradient / sumHessian if both are zero.  Once one segment of one graph has a NaN logit, then some sample will have a NaN
      //   logit

      // trainingLogWeight can be any number from -infinity to +infinity -> through addition, it can overflow to +-infinity

      EBM_ASSERT(0 == target || 1 == target);

      const FloatFast ourExp = ExpForBinaryClassification<false>(UNPREDICTABLE(size_t { 0 } == target) ? sampleScore : -sampleScore);
      // no reasonable implementation of exp should lead to a negative value
      EBM_ASSERT(std::isnan(sampleScore) || 0 <= ourExp);

      // exp will always be positive, and when we add 1, we'll always be guaranteed to have a positive number, so log shouldn't ever fail due to negative 
      // numbers the exp term could overlfow to infinity, but that should only happen in pathalogical scenarios where our train set is driving the logits 
      // one way to a very very certain outcome (essentially 100%) and the validation set has the opposite, but in that case our ultimate convergence is 
      // infinity anyways, and we'll be generaly driving up the log loss, so we legitimately want our loop to terminate training since we're getting a 
      // worse and worse model, so going to infinity isn't bad in that case
      const FloatFast singleSampleLogLoss = LogForLogLoss<false>(FloatFast { 1 } + ourExp); // log & exp will return the same type that it is given, either float or double

      // singleSampleLogLoss can be NaN, but only though propagation -> we're never taking the log of any number close to a negative, 
      // so we should only get propagation NaN values

      // singleSampleLogLoss can be +infinity -> can happen when the logit is greater than 709, which can happen after about 70,900 boosting rounds

      // singleSampleLogLoss always positive -> the 1 term inside the log has an exact floating point representation, so no reasonable floating point framework should 
      // make adding a positive number to 1 a number less than 1.  It's hard to see how any reasonable log implementatation that would give a negative 
      // exp given a 1, since 1 has an exact floating point number representation, and it computes to another exact floating point number, and who 
      // would seriously make a log function that take 1 and returns a negative.
      // So, 
      EBM_ASSERT(std::isnan(sampleScore) || 0 <= singleSampleLogLoss); // log(1) == 0
      // TODO : check our approxmiate log above for handling of 1 exactly.  We might need to change the above assert to allow a small negative value
      //   if our approxmiate log doesn't guarantee non-negative results AND numbers slightly larger than 1

#ifndef NDEBUG
      const FloatFast expVal = std::exp(sampleScore);
      const FloatFast singleSampleLogLossDebug = EbmStats::ComputeSingleSampleLogLossMulticlass(
         1 + expVal, 0 == target ? FloatFast { 1 } : expVal
      );
      EBM_ASSERT(std::isnan(singleSampleLogLoss) || std::isinf(singleSampleLogLoss) || std::isnan(singleSampleLogLossDebug) || std::isinf(singleSampleLogLossDebug) || std::abs(singleSampleLogLossDebug - singleSampleLogLoss) < k_epsilonGradientForBinaryToMulticlass);
#endif // NDEBUG

      return singleSampleLogLoss;
   }

   INLINE_ALWAYS static FloatFast ComputeSingleSampleLogLossMulticlass(
      const FloatFast sumExp,
      const FloatFast itemExp
   ) {
      // this IS a performance critical function.  It gets called per validation sample!

      // we are confirmed to get the same log loss value as scikit-learn for binary and multiclass classification

      // aValidationLogWeight (calculates itemExp) numbers can be NaN -> We can get a NaN result inside ComputeSinglePartitionUpdate
      //   for sumGradient / sumHessian if both are zero.  Once one segment of one graph has a NaN logit, then some sample will have a NaN
      //   logit

      // aValidationLogWeight (calculates itemExp) numbers can be any number from -infinity to +infinity -> through addition, it can overflow to +-infinity

      // sumExp can be NaN -> sampleScore is used when calculating sumExp, so if sampleScore can be NaN, then sumExp can be NaN

      // sumExp can be any number from 0 to +infinity -> each e^logit term can't be less than zero, and I can't imagine any implementation 
      //   that would result in a negative exp result from adding a series of positive values.

      EBM_ASSERT(std::isnan(sumExp) || 0 <= sumExp);

      // validationLogWeight (calculates itemExp) can be any number between -infinity to +infinity, or NaN

      // itemExp can be anything from 0 to +infinity, or NaN (through propagation)
      EBM_ASSERT(std::isnan(itemExp) || 0 <= itemExp); // no reasonable implementation of exp should lead to a negative value

      // mathematically sumExp must be larger than itemExp BUT in practice itemExp might be SLIGHTLY larger due to numerical issues -> 
      //   since sumExp is a sum of positive terms that includes itemExp, it cannot be lower mathematically.
      //   sumExp, having been computed from non-exact floating points, could be numerically slightly outside of the range that we would otherwise 
      //   mathematically expect.  For sample, if EbmExp(trainingLogWeight) resulted in a number that rounds down, but the floating point processor 
      //   preserves extra bits between computations AND if sumExp, which includes the term EbmExp(trainingLogWeight) was rounded down and then subsequently 
      //   added to numbers below the threshold of epsilon at the value of EbmExp(trainingLogWeight), then by the time we get to the division of 
      //   EbmExp(trainingLogWeight) / sumExp could see the numerator as higher, and result in a value slightly greater than 1!

      EBM_ASSERT(std::isnan(sumExp) || itemExp - k_epsilonGradient <= sumExp);

      const FloatFast invertedProbability = sumExp / itemExp;

      // invertedProbability can be NaN -> 
      // - If itemExp AND sumExp are exactly zero or exactly infinity then sumExp / itemExp will lead to NaN
      // - If sumExp is zero, then itemExp pretty much needs to be zero, since if any of the terms in the sumation are
      //   larger than a fraction.  It is very difficult to see how sumExp could be 0 because it would require that we have 3 or more logits that have 
      //   all either been driven very close to zero, but our algorithm drives multiclass logits appart from eachother, so some should be positive, and
      //   therefore the exp of those numbers non-zero
      // - it is possible, but difficult to see how both itemExp AND sumExp could be infinity because all we need is for itemExp to be greater than
      //   about 709.79.  If one itemExp is +infinity then so is sumExp.  Each update is mostly limited to units of 0.01 logits 
      //   (0.01 learningRate * 1 from InverseLinkFunctionThenCalculateGradientBinaryClassification or TransformScoreToGradientMulticlass), so if we've done more than 70,900 boosting 
      //   rounds we can get infinities or NaN values.  This isn't very likekly by itself given that our default is a max of 2000 rounds, but it is possible
      //   if someone is tweaking the parameters way past their natural values

      // invertedProbability can be SLIGHTLY smaller than 1 due to numeric issues -> this should only happen if sumExp == itemExp approximately, so there can be no 
      //   other logit terms in sumExp, and this would only happen after many many rounds of boosting (see above about 70,900 rounds of boosting).
      // - if invertedProbability was slightly smaller than 1, we shouldn't expect a crash.  We'll get a slighly negative log, which would otherwise be impossible.
      //   We check this before returning the log loss to our caller, since they do not expect negative log losses.
      
      // invertedProbability must be positive -> both the numerator and denominator are positive, so no reasonable implementation should lead to a negative number

      // invertedProbability can be +infinity -> sumExp can be infinity when itemExp is non-infinity, or itemExp can be sufficiently small to cause a divide by zero.  
      // This occurs when only one of the terms has overflowed to +infinity

      // we can tollerate numbers very very slightly less than 1.  These make the log loss go down slightly as they lead to negative log
      // values, but we can tollerate this drop and rely on other features for the log loss calculation

      EBM_ASSERT(std::isnan(invertedProbability) || 1 - k_epsilonLogLoss <= invertedProbability);

      const FloatFast singleSampleLogLoss = LogForLogLoss<false>(invertedProbability);

      // we're never taking the log of any number close to a negative, so we won't get a NaN result here UNLESS invertedProbability was already NaN and we're NaN 
      // propegating

      // we're using two numbers that probably can't be represented by exact representations
      // so, the fraction might be a tiny bit smaller than one, in which case the output would be a tiny
      // bit negative.  We can just let other subsequent adds cover this up
      EBM_ASSERT(std::isnan(singleSampleLogLoss) || -k_epsilonLogLoss <= singleSampleLogLoss); // log(1) == 0
      return singleSampleLogLoss;
   }

   INLINE_ALWAYS static FloatFast ComputeSingleSampleSquaredErrorRegressionFromGradient(const FloatFast gradient) {
      // this IS a performance critical function.  It gets called per validation sample!

      // for MSE, the gradient is the error and we square it

      // gradient can be +-infinity, or NaN.  See note in ComputeSinglePartitionUpdate

      // we are confirmed to get the same mean squared error value as scikit-learn for regression
      return gradient * gradient;

      // gradient can be anything from 0 to +infinity, or NaN
   }
};

} // DEFINED_ZONE_NAME

#endif // EBM_STATS_HPP