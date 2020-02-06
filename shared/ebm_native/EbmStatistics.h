// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef EBM_STATISTICS_H
#define EBM_STATISTICS_H

#include <cmath> // log, exp, etc.  Use cmath instead of math.h so that we get type overloading for these functions for seemless float/double useage
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG

// TODO: surprisingly, neither ComputeResidualErrorMulticlass, nor ComputeResidualErrorBinaryClassification seems to be able to generate an infinity, 
//       even if we had an infinite logit!  Walk through the implications of this!
// TODO: I've used std::isnan in many asserts.  I should also include std::isinf to document where these can be, and also for correctness where we have 
//       maximum values.  Residual errors can overflow, so continue from there looking for infinity values
// TODO: rename FloatEbmType to DecimalDataType (or something else)
// TODO: enable SSE on x86 Intel to avoid floating point slowdowns, and for almost exact floating point equivalence between machines -> 
//       https://stackoverflow.com/questions/7517588/different-floating-point-result-with-optimization-enabled-compiler-bug
// TODO: before applying a small model update, check to see that the probability implied by the logit exceeds 1/number_of_instances in either the positive 
//       or negative direction. If it does, then modify the update so that the probabily does not exceed a certainty of 1/number_of_instances, since we 
//       really can't make statements of probability beyond close to that threshold.

// TODO: we should consider having a min_instances, which will eliminate any class with less than that number of instances, although we'll record the base 
//       rate in the model and then we can re-build the tensor with the base rate in any classes that we removed
// TODO: we can also add new classes at prediction time.. they just get 1/n_instances/num_new_classes



static_assert(
   std::numeric_limits<FloatEbmType>::is_iec559, 
   "If IEEE 754 isn't guaranteed, then we can't use or compare infinity values in any standard way. "
   "We can't even guarantee that infinity exists as a concept."
);
// HANDLING SPECIAL FLOATING POINT VALUES (NaN/infinities/denormals/-0):
// - it should be virtually impossible to get NaN values anywhere in this code without being given an adversarial dataset or adversarial input parameters 
//   [see notes below]
// - it should be virtually impossible to get +-infinity values in classification.  We could get +infinity values if the user gives us regression targets above 
//   sqrt(std::numeric_limits<FloatEbmType>::max()) [see notes below], but those regression targets are HUGE and the user should just avoid them.  
//   We can't avoid possibility of the user passing us ridiculously huge regression targets, so HUGE inputs is something that has no good solution anyways.
// - we don't crash or really cause any problems if denormals are disabled.  We'd prefer denormals to be enabled since we'll get better and more accurate 
//   results, but we don't rely on them.  We also like getting consisent results between environments, so we'd prefer denormals to exist in all environments.
// - negative 0 is a no-op for us.  We don't care if -0 becomes +0 in our code
// - our strategy, for reasons described below, is to allow NaN and +-infinty values to propagate through our computations.
//   We will have a single check for these almost-illegal values at the end.  We terminate all boosting if we see any NaN or +-infinity values.
// - we ensure that any NaN or +-infinity values DO NOT CRASH the program, or cause an infinite loop, but beyond that we give no guarantees.  
//   The caller can see NaN, +-infinity, or simply wrong values if they give us adversarial inputs, since the cost of handling them is too high and not even 
//   guaranteed to work.
//
// - most compilers have flags that give unspecified behavior when either NaN or +-infinity are generated, or propagated, or in conditional statements 
//   involving them.  
// - many compilers seem to virtually require the switches enabling floating point deviations from IEEE 754 in order to enable SIMD, 
//   since they want to re-order things like addition instructions (reordering additions is non-IEEE 754 compliant)
// - given that we are very CPU bound, we would REALLY like our code to be as SIMD friendly as possible (possibly about 8x faster for AVX-512)
// - so, we'd like to enable those compiler flags that cause deviations from IEEE 754, even though they make it harder for us to reason about the program.
// - Hopefully the compiler won't give us undefined behaviour for special floats under these compiler flags.  Unspecified behavior is much gentler and limits 
//   the damage to just the individual floating point values!  I haven't seen any documentation from any compiler declaring which situation applies though 
//   (undefined behavior vs unspecified behavior), so we'll have to update this if we find any violations.
// - g++ makes std::isnan fail to work when the compiler flags indicate no NaN values are present.  We turn that flag off though 
//   (-ffast-math -fno-finite-math-only) -> https://stackoverflow.com/questions/570669/checking-if-a-double-or-float-is-nan-in-c
// - Generally +-infinity are treated better than NaN under the compiler switches that make floating points less IEEE 754 compliant
// - any branching instructions inside a loop will destroy the ability to generate SIMD instructions for the loop.  Non-branching conditional statements in 
//   SIMD are allowed via bit masks.
// - since we want SIMD, we need to avoid branching instructions that exit the loop, so if we did want to handle special floating point, we'd need to either 
//   signal an error or special
//   condition inside the loop via a bool, or handling the oddity in the floating point value itself.  One way to handle this is to use NaN or infinity 
//   propagation to take the error to the end of the loop
// - old hardware used to have pentalties of hundreds of CPU cycles for handling: NaN, +-infinity, and denomral values.  This has generally been resolved
//   on newer hardware.
//   https://www.agner.org/optimize/nan_propagation.pdf
// - even if there are huge penalties for handling special numeric values, we can stop boosting after a single completed round, and a several hundred CPU cycle 
//   penalty will generally be less than the total time of doing the rest of the training, which is generally at least dozens of features and hundreds of 
//   boosting rounds.
// - for Intel x86, we should enable SSE for floating point instead of x87, since x87 has performance problems and rounding inconsistencies with other systems
// - there are C/C++ ways of turning floating point exceptions on or off, but most systems don't do that.  We'll let our caller choose
//   if those exceptions are enabled, and if so they can at least detect the exceptions.  If we wanted to specify that exceptions would not occur, then
//   we could use the special comparison functions designed to emulate quiet NaN behavior: std::islessequal, etc..
// - even though the is_iec559 doesn't correctly indicate when IEEE 754 is broken due to compiler flags, we should check it anyways because if
//   it's false then we could have even more crazy semantics regarding floating point numbers, such as no infinities, etc.
// - if we have true is_iec559 (IEEE 754) compatibility (checked above), then floating point operations have these properties:
//   - SEE reference https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
//   - these operations have exact outcomes, which are guaranteed to be rounded correctly to the last bit: addition, subtraction, multiplication, division 
//     and square root, although more bits can be present than the exact sizes of the numbers being operated on, like on Intel x87.
//     PER: https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
//   - exp and log do not guarantee exactly identical results between implementations. They can't due to the "Table Maker’s Dilemma".
//     PER: https://randomascii.wordpress.com/2013/07/16/floating-point-determinism/
//   - BUT, if we use approximate exp and log functions, we can guarantee identical results, at least in theory if the language was perfectly IEEE 754 
//     compliant without extended length registers
//   - floating point integers (1.0, 2.0, 3.0, etc) have exact representations up to very large numbers where the jumps between floating point numbers 
//     are greater than 1
//   - fractions of a power of 2 (1/2.0, 1/4.0, 1/8.0, etc) have exact representations down to very small numbers
// IEEE 754 special value operations:
//   denormal values don't get rounded to zero
//   NaN values get propagated in operations
//   anything compared to a NaN, is false, except for "a != a" (where a is NaN), which is true! Take note that "a == a" is FALSE for NaN values!!!
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
// - once we get a NaN value in almost ANY computation, even for a single instance's residual value, it'll spread through our system like a wildfire
//   making everyting NaN
// - once we get an infinity value in almost ANY computation, even for a single instance's residual value, it'll turn into a NaN soon, and then everything 
//   will become a NaN
// - so, to evaluate the possibilty of NaN or infinity values, we need to look everywhere that could generates such values 
//   [see table of operations above that can these values]
// - Generally, the reason why in practice we never see NaN or infinity values is that the boosting algorithm tends to push errors towards zero, and in the
//   places where it's pushing towards inifity, like for logits, it does so slowly enough that we shouldn't reach infinity in any even semi-reasonable 
//   numbers of boosting rounds. We can also further diswade this from occuring, and in fact for probabalistic reasons, we want to push back against 
//   extremely high or low probabilities, so we want to handle the cases were we might be heading there in an unreasonably long number of boosting rounds.
// - looking purely at operators, NaN or infinity values in our code could be introduced in these places:
//   - In regression, if the user passes us NaN or +-infinity as one of the regression targets.  This is almost the only legitimate way to reach +-infinity 
//     or NaN If this occurs, we'll get a mean squared error of +infinity on our FIRST attempt at boosting.  We can detect the overflow of +infinity in our 
//     caller and alert the user to the overflow.  This will also cover the case where mean squared error overflows more gradually during the training.
//     Even though this is a legitimate overflow, the user really needs to do something to handle the condition since there isn't really
//     much that we can do with such large numbers, so pushing the error back to them is the best solution anyways.
//     If we really wanted to, we could eliminate all errors from large regression targets by limiting the user to a maximum regression target value 
//     (of 7.2e+134)
//   - In regression, if the user gives us regression targets (either positive or negative) with absolute values greater than 
//     sqrt(std::numeric_limits<FloatEbmType>::max()), or if several rounds of boosting lead to large square errors, 
//     then we'll overflow to +infinity in ComputeSingleInstanceSquaredErrorRegression. This won't infect any of our exisisting 
//     training instance residuals, or model graphs, but it will lead to us returning to our caller a +infinity mean squared error value.
//     The normal repsonse to any big mean squared error would be for our caller to terminate via early stopping, which is a fine outcome.  
//     If the caller ignores the +infinity mean squared error, then they can continue to boost without issues, although all the validation residuals
//     will end up as NaN.  Our caller can detect the +infinity means squared error themselves and optionally throw a nice exception plus a message 
//     for the end user so that they understand what happened.  The user just needs to not give us such high regression targets.
//     If we really wanted to, we could eliminate this error by limiting the user to a maximum regression target value.  The maximum value would be:
//     sqrt(std::numeric_limits<FloatEbmType>::max()) / 2^64 = 7.2e+134.  Since there must be less than 2^64 instances, the square of that number
//     occuring 2^64 times won't overflow a double
//   - In regression, if the user gives us regression targets (either positive or negative) with values below but close to
//     +-std::numeric_limits<FloatEbmType>::max(), the sumResidualError can reach +-infinity since they are a sum.
//     After sumResidualError reaches +-infinity, we'll get a graph update with a +infinity, and some instances with +-infinity residuals
//     Then, on the next feature that we boost on, we'll calculate a model update for some instances  
//     inside ComputeSmallChangeForOneSegmentRegression as +-infinity/cInstances, which will be +-infinity (of the same sign). 
//     Then, when we go to calculate our new instance residuals, we'll subtract +infinity-(+infinity) or -infinity-(-infinity), 
//     which will result in NaN.  After that, everything melts down to NaN.  The user just needs to not give us such high regression targets.
//     If we really wanted to, we could eliminate all errors from large regression targets by limiting the user to a maximum regression target value
//     (of 7.2e+134)
//
//   - In multiclass, in ComputeResidualErrorMulticlass, we could get either NaN or +infinity if sumExp was 0 in the division "ourExp / sumExp", 
//     (NaN occurs if ourExp was also zero). This could in theory only occur in multiclass if all the logits were very very negative.  
//     e^(very_big_negative_logit) = 0.  This is VERY VERY unlikely in any normal dataset because all the logits would need to be very very negative. 
//     Logits in multiclass move in opposite directions while training, so having all logits very negative shouldn't occur, 
//     except possibly in exceptionally contrived situations with extreme logits at initialization.  Our higher level caller though should 
//     be getting the initialization logits from previous boosting using this code, so this shouldn't happen if this code doesn't return such values 
//     [which is true]
//   - In multiclass, In ComputeResidualErrorMulticlass NaN can come from infinity/infinity, which can happen if any of the e^logits terms reach close to 
//     infinity. Doubles have a max value arround 1.8e308, and ln(1.8e308) = approx +709.79, so if any logit reaches above 709 we could have a problem.
//     If ANY logit in any class reaches this high, we should get such a result.  We should also get this result if the sum of a couple very big logits
//     overflows, but the sum would require the terms to be in the same neighbourhood of logits with values arround 709, like two with 708, etc.
//     BUT, reaching logits in the range of 709 is almost impossible without adversarial inputs (read on)..
//     MORE INFO on why logits of 709 are virtually impossible:
//     For computing the model update, when residualError is close to zero, at the limit the sums used in the 
//     model update of numerator/denominator are mathematically at worst +-1 (before applying the learningRate):
//     Segment_model_update = sumResidualError / sumDenominator => residualError / [residualError * (1 - residualError)] => 
//     residualError / [residualError * (1)] => 1
//     When our predictor is very certain, but wrong, we get residual errors like -1, and 1.  In those cases we have a big numerator (relatively)
//     and a small denominator, so those cases are weighted extra heavily, and are weighted much higher than +-1
//     the overall effect, is that we train more on the errors, but in these cases the movement is opposite to the existing value, 
//     so high positive numbers become more negative, and vise versa, so they act in a restorative way and don't contribute to runnaway logits.
//     for wrong predictions: Update = sumResidualError / sumDenominator => +-1 / [1 * (1 - 1)] => +-1 / 0 => +-infinity
//     but the algorithm will fight against +-infinity since it won't let the residualError values get too large before it agressively corrects them back 
//     in the direction of zero.  If the logits are traveling in the right direction, then the logit updates will go down to +-1.  Then, we apply the 
//     learningRate, which is by default 0.01.  If those values hold, then the post-learningRate update is a maximum of +-0.01, and 
//     it will take arround 70,900 rounds of boosting for the logit to reach the 709 neighbourhood.
//     Our default number of rounds is 2000, so we won't get close under normal parameters.
//   - For multiclass, in theory, if we got a big number of +-infinity as a logit update, we could overflow a section of the graph. We'll hit overflows inside
//     ComputeResidualErrorMulticlass though first when the logits get to 709-ish numbers, so it isn't really possible for the logits themselves
//     to overflow first to +-infinity in the graphs for multiclass

//   - For binary and multiclass classification, log loss can reach infinty if we are very cetain of one or more predictions which are incorrectly 
//     predicted in the validation set.  This is almost impossible to occur in real datasets, since log loss will be going up before we reach infinity, so 
//     if our caller is using early stopping, they'll terminate the learning.
//     It could happen though if the caller doesn't have early stopping, or specifies an exact number of boosting rounds
//     In any case, we might want to limit probabilities to within 1/number_of_instances since we want to avoid scenarios that completely blow up 
//     and exit early

//   - For binary and multiclass classification, ComputeSmallChangeForOneSegmentClassificationLogOdds has a division that can overflow to 
//     infinity or NaN values making the logits in our graph infinity or NaN.
//   - for any learning, adversarial inputs like ridiculously huge learningRate parameters or NaN values could cause these conditions

//   - In ComputeResidualErrorBinaryClassification, the exp term could get to zero or +infinity, but division by infinity leads to zero, so this shouldn't 
//     propagate the infinity term, or create new NaN values.  Even if our logits reached +-infinity ComputeResidualErrorBinaryClassifications should 
//     return valid numbers

//   - if our boosting code gets passed NaN or +-infinity values as predictor scores (regression predictions or logits)
//     Normally, our higher level library (python, R, etc..) calculates the predictor scores based on prior boosting rounds from 
//     within this core library.  So we shouldn't get such extreme values unless this C++ code generates NaN, or +-infinities, 
//     and we've shown above that this is very very unlikely for non-adversarial inputs.
// - Even before our first e^logit reaches +infinity, we might get a scenario where none of the e^logits are infinity yet, BUT the sum reaches infinity
//   and in that case we'll get BIG_NUMBER/infinity, which will be zero, which is the opposite of the mathematical result we want of BIG_NUMBER/BIG_NUMBER, 
//   which should be 1, and it will therefore drive that residual and logit update in the wrong direction, so if we ever see an infinity in any
//   multiclass denominator term, we're already in a bad situation.  There is a solution where we could subtract a constant from all the multiclass logits,
//   which is mathematical no-op, but... why??
// - ComputeSmallChangeForOneSegmentClassificationLogOdds will end with a divide by zero if the denominator is zero.  The denominator will be zero
//   if the residual error is 1.  For binary classification, if 1 + ? = 1, due to numeracy issues, then the denominator term in 
//   ComputeResidualErrorBinaryClassification will be one, leading to numeracy issues later on due to the 1.  Epsilon is the smallest number that you can 
//   add to 1 and get a non-1. For doubles epsilon is 2.2204460492503131e-016.  e^-36.043 and with an update rate of 0.01, that means arround 3604 rounds 
//   of boosting.  We're still good given that our default is a max of 2000 boosting rounds.
//   Multiclass has the same issues in summing the e^logit values.
// - there is an asymetry in that small logits and large ones.  Only small ones lead to a problem because for 1 + ?, the ? needs to be small so that 
//   the result is 1.  With big values we get a 1/big_value, which is ok since there is more resolution for floating points arround zero
//   for binary classification we can flip the sign of the logits and make large ones small and small ones large somehow since they are the reciprical
// - after we've decided on a model update, we can check to see what the probability predicted would be with just this 1 feature 
//   (we'd need to know the intercept)
//   and if that probability was beyond the 1/number_of_instances or (number_of_instances-1)/number_of_instances probability range, then we can 
//   be pretty confident that the algorithm is being wildly optimistic, since we have insufficient data to determine a probability that extreme
//   So, we can look at the existing feature logits, and calculate if any individual bin logits plus the logit update exceeds that probability, 
//   and then we can trim the update logit such that the total never goes above that maximum value.  This isn't really for avoiding +-infinity or NaN
//   but it helps.  This is to ensure that we don't terminate the algorithm prematurely for a single case that isn't being predicted well due to runaway
//   logits in the training set where there are no instances of a particular class in a bin.  By limiting beyond 1/number_of_instances probabilities
//   we won't get completely wild in predicting instances in the validation set.
// - for binary classification, we can just check that the absolute value of the logit doesn't exceed either +-std::log(1/cInstances), and if it does we can
//   just reduce it to that value.
// - for multiclass, the correct is a bit harder.  We'd prefer not needing to calculate the true probabily of each class, but we can do an approximate job in 
//   logit space.  First off, there can only be one class with a big probability at a time, so we first need to find out if any item exceeds 
//   (cInstances - 1) / cInstances probability.  For the biggest probability, the only two values that matter are the biggest and second biggest logits.  
//   Reduce the biggest probability so that it doesn't exceed the second biggest probability by std:log(1/cInstances).
//   There can be unlimited items with probabilities that are too low.  After fixing the biggest probability, we need to check to see if any of these are too
//   low.  If any are lower than std:log(1/cInstances) from the max logit, the raise them to that level
// - even though multiclass logits won't average to zero, we can be sure that none of them will chnage by more than learingRate logits per boosting round, so 
//   they can't diverge to more than 709 logits difference for at least 70,900/2=35,400 rounds.  That's still acceptable 
// - probably the first place we'll get an issue is for multiclass, where we might get one logit that causes e^logit to be large enough that adding all the 
//   other e^logit values doesn't add anything to the higher e^logit value.  Once that happens, then we'll get e^logit/e^logit = 1, then our 
// - from conversation with Rich, we don't really know how to do Laplacian smoothing on boosted decision trees, and it's done post process anyways, 
//   so wouldn't help during tree building.  It would apply a skew to the updates if used during training.
// - EVEN with these precautions though, a malicious user could create a training set such that many individual features could have high logits, 
//   and then they could create an exceptional validation instance that was the opposite of what's in the training data
//   for every such feature.  As we add the logits together we could then reach a number greater than 709, and overflow double at e^709.
// - to guard against adversarial datasets like this, we can just accept that we might get +-infinity and then NaN, and just report
//   those extreme results back to the user, or we could put in a per-instance logit maximum.  Checking for these
//   extreme logits per instance seems unreasonable given that it's hard to imagine this happening on a real dataset, so 
//   for now we'll just accept that we might hit +-infinity, or NaN and then report odd results.  Also, getting to this point would require 
//   70,900 rounds of boosting!
// - when we have a bin with zero instances of a particular class, the gain function might update the logits a few times towards
//   either a positive or negative logit, BUT after a few updates the gain will tend towards zero since it so well predicts these
//   homogenious bins.  It won't focus on such bins and will then swtich to better cut points, unless it's a binary feature or similar where it's
//   required to make the cut if we're using round robin cycling boosting
// - in the future we're going to only update a feature_combination update when it improves the log loss, so bins with low counts of a class
//   won't be able to hijack the algorithm as a whole since the algorithm will just turn off feature_combinations with bad predictions for the validation set
// - useful resources:
//   - comparing floats -> https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
//   - details on float representations -> https://www.volkerschatz.com/science/float.html
//   - things that are guaranteed and not -> https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html

class EbmStatistics final {
   EBM_INLINE EbmStatistics() {
      // DON'T allow anyone to make this static class
   }

public:

   EBM_INLINE static FloatEbmType ComputeNewtonRaphsonStep(const FloatEbmType residualError) {
      // this function IS performance critical as it's called on every instance

      // residualError can be NaN -> We can get a NaN result inside ComputeSmallChangeForOneSegmentClassificationLogOdds
      //   for sumResidualError / sumDenominator if both are zero.  Once one segment of one graph has a NaN logit, then some instance will have a NaN
      //   logit, and ComputeResidualErrorBinaryClassification will return a NaN value. Getting both sumResidualError and sumDenominator to zero is hard.  
      //   sumResidualError can always be zero since it's a sum of positive and negative values sumDenominator is harder to get to zero, 
      //   since it's a sum of positive numbers.  The sumDenominator is the sum of values returned from this function.  absResidualError 
      //   must be -1, 0, or +1 to make the denominator zero.  -1 and +1 are hard, but not impossible to get to with really inputs, 
      //   since boosting tends to push errors towards 0.  An error of 0 is most likely when the denominator term in either
      //   ComputeResidualErrorBinaryClassification or ComputeResidualErrorMulticlass becomes close to epsilon.  Once that happens
      //   for ComputeResidualErrorBinaryClassification the 1 + epsilon = 1, then we have 1/1, which is exactly 1, then we subtrace 1 from 1.
      //   This can happen after as little as 3604 rounds of boosting, if learningRate is 0.01, and every boosting round we update by the limit of
      //   0.01 [see notes at top of EbmStatistics.h].  It might happen for a single instance even faster if multiple variables boost the logit
      //   upwards.  We just terminate boosting after that many rounds if this occurs.

      // residualError CANNOT be +-infinity -> even with an +-infinity logit, see ComputeResidualErrorBinaryClassification and ComputeResidualErrorMulticlass

      // for binary classification, -1 <= residualError && residualError <= 1 -> see ComputeResidualErrorBinaryClassification

      // for multiclass, -1 - k_epsilonResidualError <= residualError && residualError <= 1 -> see ComputeResidualErrorMulticlass

      // propagate NaN or deliver weird results if NaNs aren't properly propaged.  This is acceptable considering how hard you have to 
      // work to generate such an illegal value, and given that we won't crash
      EBM_ASSERT(
         std::isnan(residualError) || 
         !std::isinf(residualError) && FloatEbmType { -1 } - k_epsilonResidualError <= residualError && residualError <= FloatEbmType { 1 }
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

      // when we use this denominator term retuned inside ComputeSmallChangeForOneSegmentClassificationLogOdds, if there was only
      //   a single denominator term, or multiple similar ones, at the limit we'd get the following for the following inputs:
      //   boosting is working propertly and we're close to zero error:
      //     - segment_model_update = sumResidualError / sumDenominator => residualError / [residualError * (1 - residualError)] => 
      //       residualError / [residualError * (1)] => +-1  but we multiply this by the learningRate of 0.01 (default), to get +-0.01               
      //   when boosting is making a mistake, but is certain about it's prediction:
      //     - segment_model_update = sumResidualError / sumDenominator => residualError / [residualError * (1 - residualError)] => +-1 / [1 * (0)] => 
      //       +-infinity
      //       but this shouldn't really happen inside the training set, because as the error gets bigger our boosting algorithm will correct corse by
      //       updating in the opposite direction.  Divergence to infinity is a possibility in the validation set, but the training set pushes it's error to 
      //       zero.  It may be possible to construct an adversarial dataset with negatively correlated features that cause a bouncing arround that leads to 
      //       divergence, but that seems unlikely in a normal dataset
      //   our resulting function looks like this:
      // 
      //  small_model_update
      //          |     *
      //          |     *
      //          |     *
      //          |    * 
      //          |   *  
      //      0.01|*     
      //          |      
      //  -1-------------1--- residualError
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
      //   the overall effect is that we train more on errors (error is +-1), and less on things with close to zero residuals

      // !!! IMPORTANT: Newton-Raphson step, as illustrated in Friedman's original paper (https://statweb.stanford.edu/~jhf/ftp/trebst.pdf, page 9). Note that
      //   they are using t * (2 - t) since they have a 2 in their objective
      const FloatEbmType absResidualError = std::abs(residualError); // abs will return the same type that it is given, either float or double
      const FloatEbmType ret = absResidualError * (FloatEbmType { 1 } - absResidualError);

      // - it would be somewhat bad if absResidualError could get larger than 1, even if just due to floating point error reasons, 
      //   since this would flip the sign on ret to negative.  Later in ComputeSmallChangeForOneSegmentClassificationLogOdds, 
      //   The sign flip for ret will cause the update to be the opposite from what was desired.  This should only happen
      //   once we've overflowed our boosting rounds though such that 1 + epsilon = 1, after about 3604 boosting rounds
      //   At that point, undoing one round of boosting isn't going to hurt.

      // the maximum return value occurs when residual error is 0.5 (which is exactly representable in floating points), which gives 0.25
      // 0.25 has an exact representations in floating point numbers, at least in IEEE 754, which we statically check for at compile time
      // I don't think we need an espilon at the top figure.  Given that all three numbers invovled (0.5, 1, 0.25) have exact representations, 
      // we should get the exact return of 0.25 for 0.5, and never larger than 0.25
      // IEEE 754 guarantees that these values have exact representations AND it also guarantees that the results of addition and multiplication
      // for these operations are rounded to the nearest bit exactly, so we shouldn't be able to get anything outside of this range.

      // our input allows values slightly larger than 1, after subtracting 1 from 1 + some_small_value, we can get a slight negative value
      // when we get to this point, we're already in a position where we'll get a NaN or infinity or something else that will stop boosting

      // with our abs inputs confined to the range of -k_epsilonResidualError -> 1, we can't get a new NaN value here, so only check against propaged NaN 
      // values from residualError
      EBM_ASSERT(std::isnan(residualError) || !std::isinf(ret) && -k_epsilonResidualError <= ret && ret <= FloatEbmType { 0.25 });

      return ret;
   }

   EBM_INLINE static FloatEbmType ComputeNodeSplittingScore(const FloatEbmType sumResidualError, const FloatEbmType cInstances) {
      // this function can SOMETIMES be performance critical as it's called on every histogram bin
      // it will only be performance critical for truely continous numerical features that we're not binning, or for interactions where dimensionality
      // creates many bins

      // for classification, sumResidualError can be NaN -> We can get a NaN result inside ComputeSmallChangeForOneSegmentClassificationLogOdds
      //   for sumResidualError / sumDenominator if both are zero.  Once one segment of one graph has a NaN logit, then some instance will have a NaN
      //   logit, and ComputeResidualErrorBinaryClassification will return a NaN value. Getting both sumResidualError and sumDenominator to zero is hard.  
      //   sumResidualError can always be zero since it's a sum of positive and negative values sumDenominator is harder to get to zero, 
      //   since it's a sum of positive numbers.  The sumDenominator is the sum of values returned from this function.  absResidualError 
      //   must be -1, 0, or +1 to make the denominator zero.  -1 and +1 are hard, but not impossible to get to with really inputs, 
      //   since boosting tends to push errors towards 0.  An error of 0 is most likely when the denominator term in either
      //   ComputeResidualErrorBinaryClassification or ComputeResidualErrorMulticlass becomes close to epsilon.  Once that happens
      //   for ComputeResidualErrorBinaryClassification the 1 + epsilon = 1, then we have 1/1, which is exactly 1, then we subtrace 1 from 1.
      //   This can happen after as little as 3604 rounds of boosting, if learningRate is 0.01, and every boosting round we update by the limit of
      //   0.01 [see notes at top of EbmStatistics.h].  It might happen for a single instance even faster if multiple variables boost the logit
      //   upwards.  We just terminate boosting after that many rounds if this occurs.

      // for regression, residualError can be NaN -> if the user gives us regression targets (either positive or negative) with values below but close to
      //   +-std::numeric_limits<FloatEbmType>::max(), the sumResidualError can reach +-infinity since they are a sum.
      //   After sumResidualError reaches +-infinity, we'll get a graph update with a +infinity, and some instances with +-infinity residuals
      //   Then, on the next feature that we boost on, we'll calculate a model update for some instances  
      //   inside ComputeSmallChangeForOneSegmentRegression as +-infinity/cInstances, which will be +-infinity (of the same sign). 
      //   Then, when we go to calculate our new instance residuals, we'll subtract +infinity-(+infinity) or -infinity-(-infinity), 
      //   which will result in NaN.  After that, everything melts down to NaN.  The user just needs to not give us such high regression targets.
      //   If we really wanted to, we could eliminate all errors from large regression targets by limiting the user to a maximum regression target value 
      //   (of 7.2e+134)

      // for classification, sumResidualError CANNOT be +-infinity-> even with an +-infinity logit, see ComputeResidualErrorBinaryClassification and 
      //   ComputeResidualErrorMulticlass also, since -cInstances <= sumResidualError && sumResidualError <= cInstances, and since cInstances must be 64 bits 
      //   or lower, we cann't overflow to infinity when taking the sum

      // for classification -cInstances <= sumResidualError && sumResidualError <= cInstances
      
      // for regression sumResidualError can be any legal value, including +infinity or -infinity

      // !!! IMPORTANT: This gain function used to determine splits is equivalent to minimizing sum of squared error SSE, which can be seen following the 
      //   derivation of Equation #7 in Ping Li's paper -> https://arxiv.org/pdf/1203.3491.pdf

      // TODO: we're using this node splitting score for both classification and regression.  It is designed to minimize MSE, so should we also then use 
      //    it for classification?  What about the possibility of using Newton-Raphson step in the gain?
      // TODO: we should also add an option to optimize for mean absolute error

      EBM_ASSERT(!std::isnan(cInstances)); // this starts as an integer
      EBM_ASSERT(!std::isinf(cInstances)); // this starts as an integer

#ifdef LEGACY_COMPATIBILITY
      const FloatEbmType ret = LIKELY(FloatEbmType { 0 } != cInstances) ? sumResidualError / cInstances * sumResidualError : FloatEbmType { 0 };
#else // LEGACY_COMPATIBILITY
      EBM_ASSERT(FloatEbmType { 1 } <= cInstances); // we shouldn't be making splits with children with less than 1 instance
      const FloatEbmType ret = sumResidualError / cInstances * sumResidualError;
#endif // LEGACY_COMPATIBILITY

      // for both classification and regression, we're squaring sumResidualError, and cInstances is positive.  No reasonable floating point implementation 
      // should turn this negative

      // for both classification and regression, we shouldn't generate a new NaN value from our calculation since cInstances can't be zero or inifinity, 
      //   but we might need to propagage a NaN sumResidualError value

      // for classification, 0 <= ret && ret <= cInstances + some_epsilon, since sumResidualError can't be infinity or have absolute values above cInstances

      // for regression, the output can be any positive number from zero to +infinity

      EBM_ASSERT(std::isnan(sumResidualError) || FloatEbmType { 0 } <= ret);
      return ret;
   }

   EBM_INLINE static FloatEbmType ComputeSmallChangeForOneSegmentClassificationLogOdds(
      const FloatEbmType sumResidualError, 
      const FloatEbmType sumDenominator
   ) {
      // this is NOT a performance critical function.  It only gets called AFTER we've decided where to split, so only a few times per Boosting step

      // sumResidualError can be NaN -> We can get a NaN result inside ComputeSmallChangeForOneSegmentClassificationLogOdds
      //   for sumResidualError / sumDenominator if both are zero.  Once one segment of one graph has a NaN logit, then some instance will have a NaN
      //   logit, and ComputeResidualErrorBinaryClassification will return a NaN value. Getting both sumResidualError and sumDenominator to zero is hard.  
      //   sumResidualError can always be zero since it's a sum of positive and negative values sumDenominator is harder to get to zero, 
      //   since it's a sum of positive numbers.  The sumDenominator is the sum of values returned from this function.  absResidualError 
      //   must be -1, 0, or +1 to make the denominator zero.  -1 and +1 are hard, but not impossible to get to with really inputs, 
      //   since boosting tends to push errors towards 0.  An error of 0 is most likely when the denominator term in either
      //   ComputeResidualErrorBinaryClassification or ComputeResidualErrorMulticlass becomes close to epsilon.  Once that happens
      //   for ComputeResidualErrorBinaryClassification the 1 + epsilon = 1, then we have 1/1, which is exactly 1, then we subtrace 1 from 1.
      //   This can happen after as little as 3604 rounds of boosting, if learningRate is 0.01, and every boosting round we update by the limit of
      //   0.01 [see notes at top of EbmStatistics.h].  It might happen for a single instance even faster if multiple variables boost the logit
      //   upwards.  We just terminate boosting after that many rounds if this occurs.

      // sumDenominator can be NaN -> sumDenominator is calculated from residualError.  Since residualError can be NaN (as described above), then 
      //   sumDenominator can be NaN

      // sumResidualError CANNOT be +-infinity-> even with an +-infinity logit, see ComputeResidualErrorBinaryClassification and ComputeResidualErrorMulticlass
      //   also, since -cInstances <= sumResidualError && sumResidualError <= cInstances, and since cInstances must be 64 bits or lower, we cann't overflow 
      //   to infinity when taking the sum

      // sumDenominator cannot be infinity -> per the notes in ComputeNewtonRaphsonStep

      // since this is classification, -cInstances <= sumResidualError && sumResidualError <= cInstances
      EBM_ASSERT(!std::isinf(sumResidualError)); // a 64-bit number can't get a value larger than a double to overflow to infinity

      // since this is classification, 0 <= sumDenominator && sumDenominator <= 0.25 * cInstances (see notes in ComputeNewtonRaphsonStep), 
      // so sumDenominator can't be infinity
      EBM_ASSERT(!std::isinf(sumDenominator)); // a 64-bit number can't get a value larger than a double to overflow to infinity

      // sumDenominator can be very slightly negative, for floating point numeracy reasons.  On the face, this is bad because it flips the direction
      //   of our update.  We should only hit this condition when our error gets so close to zero that we should be hitting other numeracy issues though
      //   and even if it does happen, it should be EXTREMELY rare, and since we're using boosting, it will recover the lost round next time.
      //   Also, this should only happen when we are at an extremem logit anyways, and going back one step won't hurt anything.  It's not worth
      //   the cost of the comparison to check for the error condition
      EBM_ASSERT(std::isnan(sumDenominator) || -k_epsilonResidualError <= sumDenominator);

      // sumResidualError can always be much smaller than sumDenominator, since it's a sum, so it can add positive and negative numbers such that it reaches 
      // almost zero whlie sumDenominator is always positive

      // sumDenominator can always be much smaller than sumResidualError.  Example: 0.999 -> the sumDenominator will be 0.000999 (see ComputeNewtonRaphsonStep)

      // since the denominator term always goes up as we add new numbers (it's always positive), we can only have a zero in that term if ALL instances have 
      // low denominators
      // the denominator can be close to zero only when residualError is -1, 0, or 1 for all cases, although they can be mixed up between these values.
      // Given that our algorithm tries to drive residual error to zero and it will drive it increasingly strongly the farther it gets from zero, 
      // it seems really unlikley that we'll get a zero numerator by having a series of
      // -1 and 1 values that perfectly cancel eachother out because it means they were driven to the opposite of the correct answer
      // If the denominator is zero, then it's a strong indication that all the residuals are close to zero.
      // if all the residuals are close to zero, then the numerator is also going to be close to zero
      // So, if the denominator is close to zero, we can assume the numerator will be too.  At this point with all the residual errors very close to zero
      // we might as well stop learning on these instances and set the update to zero for this section anyways, but we don't want branches here, 
      // so we'll just leave it

      return sumResidualError / sumDenominator;

      // return can be NaN if both sumResidualError and sumDenominator are zero, or if we're propagating a NaN value.  Neither sumResidualError nor 
      //   sumDenominator can be infinity, so that's not a source of NaN
      // return can be infinity if sumDenominator is extremely close to zero, while sumResidualError is non-zero (very hard).  This could happen if 
      //   the residualError was near +-1
      // return can be any other positive or negative number
   }

   EBM_INLINE static FloatEbmType ComputeSmallChangeForOneSegmentRegression(const FloatEbmType sumResidualError, const FloatEbmType cInstances) {
      // this is NOT a performance critical call.  It only gets called AFTER we've decided where to split, so only a few times per feature_combination boost

      // sumResidualError can be NaN -> if the user gives us regression targets (either positive or negative) with values below but close to
      //   +-std::numeric_limits<FloatEbmType>::max(), the sumResidualError can reach +-infinity since they are a sum.
      //   After sumResidualError reaches +-infinity, we'll get a graph update with a +infinity, and some instances with +-infinity residuals
      //   Then, on the next feature that we boost on, we'll calculate a model update for some instances  
      //   inside ComputeSmallChangeForOneSegmentRegression as +-infinity/cInstances, which will be +-infinity (of the same sign). 
      //   Then, when we go to calculate our new instance residuals, we'll subtract +infinity-(+infinity) or -infinity-(-infinity), 
      //   which will result in NaN.  After that, everything melts down to NaN.  The user just needs to not give us such high regression targets.
      //   If we really wanted to, we could eliminate all errors from large regression targets by limiting the user to a maximum regression target value 
      //   (of 7.2e+134)

      // sumResidualError can be any legal value, including +infinity or -infinity

      // 1 < cInstances && cInstances < 2^64 (since it's a 64 bit integer in 64 bit address space app
      EBM_ASSERT(!std::isnan(cInstances)); // this starts as an integer
      EBM_ASSERT(!std::isinf(cInstances)); // this starts as an integer

#ifdef LEGACY_COMPATIBILITY
      return LIKELY(FloatEbmType { 0 } != cInstances) ? sumResidualError / cInstances : FloatEbmType { 0 };
#else // LEGACY_COMPATIBILITY
      // -infinity <= sumResidualError && sumResidualError <= infinity (it's regression which has a larger range)

      // even if we trim inputs of +-infinity from the user to std::numeric_limits<FloatEbmType>::max() or std::numeric_limits<FloatEbmType>::min(), 
      // we'll still reach +-infinity if we add a bunch of them together, so sumResidualError can reach +-infinity.
      // After sumResidualError reaches +-infinity, we'll get an update and some instances with +-infinity residuals
      // Then, on the next feature we boost on, we'll calculate an model update for some instances (inside this function) as 
      // +-infinity/cInstances, which will be +-infinity (of the same sign).  Then, when we go to find our new instance residuals, we'll
      // subtract +infinity-(+infinity) or -infinity-(-infinity), which will result in NaN.  After that, everything melts down to NaN.
      EBM_ASSERT(FloatEbmType { 1 } <= cInstances); // we shouldn't be making splits with children with less than 1 instance
      return sumResidualError / cInstances;
#endif // LEGACY_COMPATIBILITY
      
      // since the sumResidualError inputs can be anything, we can return can be anything, including NaN, or +-infinity
   }

   EBM_INLINE static FloatEbmType ComputeResidualErrorRegressionInit(const FloatEbmType predictionScore, const FloatEbmType actualValue) {
      // this function is NOT performance critical as it's called on every instance, but only during initialization.

      // it's possible to reach NaN or +-infinity within this module, so predictionScore can be those values
      // since we can reach such values anyways, we might as well not check for them during initialization and detect the
      // NaN or +-infinity values in one of our boosting rounds and then terminate the algorithm once those special values
      // have propagaged, which we need to handle anyways

      const FloatEbmType result = actualValue - predictionScore;

      // if actualValue and predictionScore are both +infinity or both -infinity, then we'll generate a NaN value

      return result;
   }

   EBM_INLINE static FloatEbmType ComputeResidualErrorRegression(const FloatEbmType value) {
      // this function IS performance critical as it's called on every instance

      // value can be +-infinity, or NaN.  See note in ComputeSmallChangeForOneSegmentRegression

      // this function is here to document where we're calculating regression, like ComputeResidualErrorBinaryClassification below.  It doesn't do anything, 
      //   but it serves as an indication that the calculation would be placed here if we changed it in the future
      return value;
   }

   EBM_INLINE static FloatEbmType ComputeResidualErrorBinaryClassification(
      const FloatEbmType trainingLogOddsPrediction, 
      const size_t binnedActualValue
   ) {
      // this IS a performance critical function.  It gets called per instance!

      // trainingLogOddsPrediction can be NaN -> We can get a NaN result inside ComputeSmallChangeForOneSegmentClassificationLogOdds
      //   for sumResidualError / sumDenominator if both are zero.  Once one segment of one graph has a NaN logit, then some instance will have a NaN
      //   logit

      // trainingLogOddsPrediction can be +-infinity -> we can overflow to +-infinity

      EBM_ASSERT(0 == binnedActualValue || 1 == binnedActualValue);

      // this function outputs 0 if we perfectly predict the target with 100% certainty.  To do so, trainingLogOddsPrediction would need to be either 
      //   infinity or -infinity
      // this function outputs 1 if actual value was 1 but we incorrectly predicted with 100% certainty that it was 0 by having 
      //   trainingLogOddsPrediction be -infinity
      // this function outputs -1 if actual value was 0 but we incorrectly predicted with 100% certainty that it was 1 by having trainingLogOddsPrediction 
      //   be infinity
      //
      // this function outputs 0.5 if actual value was 1 but we were 50%/50% by having trainingLogOddsPrediction be 0
      // this function outputs -0.5 if actual value was 0 but we were 50%/50% by having trainingLogOddsPrediction be 0

      // TODO : In the future we'll sort our data by the target value, so we'll know ahead of time if 0 == binnedActualValue.  We expect 0 to be the 
      //   default target, so we should flip the value of trainingLogOddsPrediction so that we don't need to negate it for the default 0 case
      // exp will return the same type that it is given, either float or double
      const FloatEbmType ret = (UNPREDICTABLE(0 == binnedActualValue) ? FloatEbmType { -1 } : FloatEbmType { 1 }) / (FloatEbmType { 1 } +
         EbmExp(UNPREDICTABLE(0 == binnedActualValue) ? -trainingLogOddsPrediction : trainingLogOddsPrediction)); 

      // exp always yields a positive number or zero, and I can't imagine any reasonable implementation that would violate this by returning a negative number
      // given that 1.0 is an exactly representable number in IEEE 754, I can't see 1 + exp(anything) ever being less than 1, even with floating point jitter
      // again, given that the numerator 1.0 is an exactly representable number, I can't see (+-1.0 / something_1_or_greater) ever having an absolute value
      // above 1.0 especially, since IEEE 754 guarnatees that addition and division yield numbers rounded correctly to the last binary decimal place
      // IEEE 754, which we check for at compile time, specifies that +-1/infinity = 0

      // ret cannot be +-infinity -> even if trainingLogOddsPrediction is +-infinity we then get +-1 division by 1, or division by +infinity, which are +-1 
      //   or 0 respectively

      // ret can only be NaN if our inputs are NaN

      // So...
      EBM_ASSERT(std::isnan(trainingLogOddsPrediction) || !std::isinf(ret) && FloatEbmType { -1 } <= ret && ret <= FloatEbmType { 1 });

      // ret can't be +-infinity, since an infinity in the denominator would just lead us to zero for the ret value!

#ifndef NDEBUG
      const FloatEbmType retDebug = 
         ComputeResidualErrorMulticlass(FloatEbmType { 1 } + EbmExp(trainingLogOddsPrediction), trainingLogOddsPrediction, binnedActualValue, 1);
      // the ComputeResidualErrorMulticlass can't be +-infinity per notes in ComputeResidualErrorMulticlass, 
      // but it can generate a new NaN value that we wouldn't get in the binary case due to numeric instability issues with having multiple logits
      // if either is a NaN value, then don't compare since we aren't sure that we're exactly equal in those cases because of numeric instability reasons
      EBM_ASSERT(std::isnan(trainingLogOddsPrediction) || std::isnan(retDebug) || std::abs(retDebug - ret) < k_epsilonResidualError);
#endif // NDEBUG
      return ret;
   }

   EBM_INLINE static FloatEbmType ComputeResidualErrorMulticlass(
      const FloatEbmType sumExp, 
      const FloatEbmType trainingLogWeight, 
      const size_t binnedActualValue, 
      const size_t iVector
   ) {
      // this IS a performance critical function.  It gets called per instance AND per-class!

      // trainingLogWeight can be NaN -> We can get a NaN result inside ComputeSmallChangeForOneSegmentClassificationLogOdds
      //   for sumResidualError / sumDenominator if both are zero.  Once one segment of one graph has a NaN logit, then some instance will have a NaN
      //   logit
      
      // trainingLogWeight can be any number from -infinity to +infinity -> through addition, it can overflow to +-infinity

      // sumExp can be NaN -> trainingLogOddsPrediction is used when calculating sumExp, so if trainingLogOddsPrediction can be NaN, then sumExp can be NaN

      // sumExp can be any number from 0 to +infinity -> each e^logit term can't be less than zero, and I can't imagine any implementation 
      //   that would result in a negative exp result from adding a series of positive values.
      EBM_ASSERT(std::isnan(sumExp) || FloatEbmType { 0 } <= sumExp);

      const FloatEbmType ourExp = EbmExp(trainingLogWeight);
      // ourExp can be anything from 0 to +infinity, or NaN (through propagation)
      EBM_ASSERT(std::isnan(trainingLogWeight) || FloatEbmType { 0 } <= ourExp); // no reasonable implementation should lead to a negative exp value

      // mathematically sumExp must be larger than ourExp BUT in practice ourExp might be SLIGHTLY larger due to numerical issues -> 
      //   since sumExp is a sum of positive terms that includes ourExp, it cannot be lower mathematically.
      //   sumExp, having been computed from non-exact floating points, could be numerically slightly outside of the range that we would otherwise 
      //   mathematically expect.  For instance, if EbmExp(trainingLogWeight) resulted in a number that rounds down, but the floating point processor 
      //   preserves extra bits between computations AND if sumExp, which includes the term EbmExp(trainingLogWeight) was rounded down and then subsequently 
      //   added to numbers below the threshold of epsilon at the value of EbmExp(trainingLogWeight), then by the time we get to the division of 
      //   EbmExp(trainingLogWeight) / sumExp could see the numerator as higher, and result in a value slightly greater than 1!

      EBM_ASSERT(std::isnan(sumExp) || ourExp - k_epsilonResidualError <= sumExp);

      const FloatEbmType expFraction = ourExp / sumExp;

      // expFraction can be NaN -> 
      // - If ourExp AND sumExp are exactly zero or exactly infinity then ourExp / sumExp will lead to NaN
      // - If sumExp is zero, then ourExp pretty much needs to be zero, since if any of the terms in the sumation are
      //   larger than a fraction.  It is very difficult to see how sumExp could be 0 because it would require that we have 3 or more logits that have 
      //   all either been driven very close to zero, but our algorithm drives multiclass logits appart from eachother, so some should be positive, and
      //   therefor the exp of those numbers non-zero
      // - it is possible, but difficult to see how both ourExp AND sumExp could be infinity because all we need is for ourExp to be greater than
      //   about 709.79.  If one ourExp is +infinity then so is sumExp.  Each update is mostly limited to units of 0.01 logits 
      //   (0.01 learningRate * 1 from ComputeResidualErrorBinaryClassification or ComputeResidualErrorMulticlass), so if we've done more than 70,900 boosting 
      //   rounds we can get infinities or NaN values.  This isn't very likekly by itself given that our default is a max of 2000 rounds, but it is possible
      //   if someone is tweaking the parameters way past their natural values

      // expFraction can be SLIGHTLY larger than 1 due to numeric issues -> this should only happen if sumExp == ourExp approximately, so there can be no 
      //   other logit terms in sumExp, and this would only happen after many many rounds of boosting (see above about 70,900 rounds of boosting).
      // - if expFraction was slightly larger than 1, we shouldn't expect a crash.  What would happen is that in our next call to ComputeNewtonRaphsonStep, we
      //   would find our denomiator term as a negative number (normally it MUST be positive).  If that happens, then later when we go to compute the
      //   small model update, we'll inadvertently flip the sign of the update, but since ComputeNewtonRaphsonStep was close to the discontinuity at 0,
      //   we know that the update should have a value of 1 * learningRate = 0.01 for default input parameters.  This means that even in the very very
      //   very unlikely case that we flip our sign due to numericacy error, which only happens after an unreasonable number of boosting rounds, we'll
      //   flip the sign on a minor update to the logits.  We can tollerate this sign flip and next round we're not likely to see the same sign flip, so
      //   boosting will recover the mistake with more boosting rounds

      // expFraction must be positive -> both the numerator and denominator are positive, so no reasonable implementation should lead to a negative number

      // expFraction can be zero -> sumExp can be infinity when ourExp is non-infinity.  This occurs when only one of the terms has overflowed to +infinity

      // expFraction can't be infinity -> even if ourExp is slightly bigger than sumExp due to numeric reasons, the division is going to be close to 1
      //   we can't really get an infinity in ourExp without also getting an infinity in sumExp, so expFraction can't be infinity without getting a NaN

      EBM_ASSERT(std::isnan(expFraction) || 
         !std::isinf(expFraction) && FloatEbmType { 0 } <= expFraction && expFraction <= FloatEbmType { 1 } + k_epsilonResidualError);


      const FloatEbmType yi = UNPREDICTABLE(iVector == binnedActualValue) ? FloatEbmType { 1 } : FloatEbmType { 0 };

      // if expFraction cannot be +infinity, and needs to be between 0 and 1 + small_value, or NaN, then ret can't be inifinity either

      const FloatEbmType ret = yi - expFraction;

      // mathematicaly we're limited to the range of range 0 <= expFraction && expFraction <= 1, but with floating point issues
      // we can get an expFraction value slightly larger than 1, which could lead to -1.00000000001-ish results
      // just like for the division by zero conditions, we'd need many many boosting rounds for expFraction to get to 1, since
      // the sum of e^logit must be about equal to e^logit for this class, which should require thousands of rounds (70,900 or so)
      // also, the boosting algorthm tends to push results to zero, so a result more negative than -1 would be very exceptional
      EBM_ASSERT(std::isnan(expFraction) || !std::isinf(ret) && FloatEbmType { -1 } - k_epsilonResidualError <= ret && ret <= FloatEbmType { 1 });
      return ret;
   }

   EBM_INLINE static FloatEbmType ComputeSingleInstanceLogLossBinaryClassification(
      const FloatEbmType validationLogOddsPrediction, 
      const size_t binnedActualValue
   ) {
      // this IS a performance critical function.  It gets called per validation instance!

      // we are confirmed to get the same log loss value as scikit-learn for binary and multiclass classification

      // trainingLogWeight can be NaN -> We can get a NaN result inside ComputeSmallChangeForOneSegmentClassificationLogOdds
      //   for sumResidualError / sumDenominator if both are zero.  Once one segment of one graph has a NaN logit, then some instance will have a NaN
      //   logit

      // trainingLogWeight can be any number from -infinity to +infinity -> through addition, it can overflow to +-infinity

      EBM_ASSERT(0 == binnedActualValue || 1 == binnedActualValue);

      const FloatEbmType ourExp = EbmExp(UNPREDICTABLE(0 == binnedActualValue) ? validationLogOddsPrediction : -validationLogOddsPrediction);
      // no reasonable implementation of exp should lead to a negative value
      EBM_ASSERT(std::isnan(validationLogOddsPrediction) || FloatEbmType { 0 } <= ourExp);

      // exp will always be positive, and when we add 1, we'll always be guaranteed to have a positive number, so log shouldn't ever fail due to negative 
      // numbers the exp term could overlfow to infinity, but that should only happen in pathalogical scenarios where our train set is driving the logits 
      // one way to a very very certain outcome (essentially 100%) and the validation set has the opposite, but in that case our ultimate convergence is 
      // infinity anyways, and we'll be generaly driving up the log loss, so we legitimately want our loop to terminate training since we're getting a 
      // worse and worse model, so going to infinity isn't bad in that case
      const FloatEbmType ret = EbmLog(FloatEbmType { 1 } + ourExp); // log & exp will return the same type that it is given, either float or double

      // ret can be NaN, but only though propagation -> we're never taking the log of any number close to a negative, 
      // so we should only get propagation NaN values

      // ret can be +infinity -> can happen when the logit is greater than 709, which can happen after about 70,900 boosting rounds

      // ret always positive -> the 1 term inside the log has an exact floating point representation, so no reasonable floating point framework should 
      // make adding a positive number to 1 a number less than 1.  It's hard to see how any reasonable log implementatation that would give a negative 
      // exp given a 1, since 1 has an exact floating point number representation, and it computes to another exact floating point number, and who 
      // would seriously make a log function that take 1 and returns a negative.
      // So, 
      EBM_ASSERT(std::isnan(validationLogOddsPrediction) || FloatEbmType { 0 } <= ret); // log(1) == 0
      // TODO : check our approxmiate log above for handling of 1 exactly.  We might need to change the above assert to allow a small negative value
      //   if our approxmiate log doesn't guarantee non-negative results AND numbers slightly larger than 1

#ifndef NDEBUG
      FloatEbmType scores[2];
      scores[0] = 0;
      scores[1] = validationLogOddsPrediction;
      const FloatEbmType retDebug = EbmStatistics::ComputeSingleInstanceLogLossMulticlass(1 + EbmExp(validationLogOddsPrediction), scores, binnedActualValue);
      EBM_ASSERT(std::isnan(ret) || std::isinf(ret) || std::isnan(retDebug) || std::isinf(retDebug) || std::abs(retDebug - ret) < k_epsilonResidualError);
#endif // NDEBUG

      return ret;
   }

   EBM_INLINE static FloatEbmType ComputeSingleInstanceLogLossMulticlass(
      const FloatEbmType sumExp, 
      const FloatEbmType * const aValidationLogWeight, 
      const size_t binnedActualValue
   ) {
      // this IS a performance critical function.  It gets called per validation instance!

      // we are confirmed to get the same log loss value as scikit-learn for binary and multiclass classification

      // aValidationLogWeight numbers can be NaN -> We can get a NaN result inside ComputeSmallChangeForOneSegmentClassificationLogOdds
      //   for sumResidualError / sumDenominator if both are zero.  Once one segment of one graph has a NaN logit, then some instance will have a NaN
      //   logit

      // aValidationLogWeight numbers can be any number from -infinity to +infinity -> through addition, it can overflow to +-infinity

      // sumExp can be NaN -> trainingLogOddsPrediction is used when calculating sumExp, so if trainingLogOddsPrediction can be NaN, then sumExp can be NaN

      // sumExp can be any number from 0 to +infinity -> each e^logit term can't be less than zero, and I can't imagine any implementation 
      //   that would result in a negative exp result from adding a series of positive values.

      EBM_ASSERT(std::isnan(sumExp) || FloatEbmType { 0 } <= sumExp);

      const FloatEbmType validationLogWeight = aValidationLogWeight[binnedActualValue];

      // validationLogWeight can be any number between -infinity to +infinity, or NaN

      const FloatEbmType ourExp = EbmExp(validationLogWeight);
      // ourExp can be anything from 0 to +infinity, or NaN (through propagation)
      EBM_ASSERT(std::isnan(validationLogWeight) || FloatEbmType { 0 } <= ourExp); // no reasonable implementation of exp should lead to a negative value

      // mathematically sumExp must be larger than ourExp BUT in practice ourExp might be SLIGHTLY larger due to numerical issues -> 
      //   since sumExp is a sum of positive terms that includes ourExp, it cannot be lower mathematically.
      //   sumExp, having been computed from non-exact floating points, could be numerically slightly outside of the range that we would otherwise 
      //   mathematically expect.  For instance, if EbmExp(trainingLogWeight) resulted in a number that rounds down, but the floating point processor 
      //   preserves extra bits between computations AND if sumExp, which includes the term EbmExp(trainingLogWeight) was rounded down and then subsequently 
      //   added to numbers below the threshold of epsilon at the value of EbmExp(trainingLogWeight), then by the time we get to the division of 
      //   EbmExp(trainingLogWeight) / sumExp could see the numerator as higher, and result in a value slightly greater than 1!

      EBM_ASSERT(std::isnan(sumExp) || ourExp - k_epsilonResidualError <= sumExp);

      const FloatEbmType expFraction = sumExp / ourExp;

      // expFraction can be NaN -> 
      // - If ourExp AND sumExp are exactly zero or exactly infinity then sumExp / ourExp will lead to NaN
      // - If sumExp is zero, then ourExp pretty much needs to be zero, since if any of the terms in the sumation are
      //   larger than a fraction.  It is very difficult to see how sumExp could be 0 because it would require that we have 3 or more logits that have 
      //   all either been driven very close to zero, but our algorithm drives multiclass logits appart from eachother, so some should be positive, and
      //   therefore the exp of those numbers non-zero
      // - it is possible, but difficult to see how both ourExp AND sumExp could be infinity because all we need is for ourExp to be greater than
      //   about 709.79.  If one ourExp is +infinity then so is sumExp.  Each update is mostly limited to units of 0.01 logits 
      //   (0.01 learningRate * 1 from ComputeResidualErrorBinaryClassification or ComputeResidualErrorMulticlass), so if we've done more than 70,900 boosting 
      //   rounds we can get infinities or NaN values.  This isn't very likekly by itself given that our default is a max of 2000 rounds, but it is possible
      //   if someone is tweaking the parameters way past their natural values

      // expFraction can be SLIGHTLY smaller than 1 due to numeric issues -> this should only happen if sumExp == ourExp approximately, so there can be no 
      //   other logit terms in sumExp, and this would only happen after many many rounds of boosting (see above about 70,900 rounds of boosting).
      // - if expFraction was slightly smaller than 1, we shouldn't expect a crash.  We'll get a slighly negative log, which would otherwise be impossible.
      //   We check this before returning the log loss to our caller, since they do not expect negative log losses.
      
      // expFraction must be positive -> both the numerator and denominator are positive, so no reasonable implementation should lead to a negative number

      // expFraction can be +infinity -> sumExp can be infinity when ourExp is non-infinity, or ourExp can be sufficiently small to cause a divide by zero.  
      // This occurs when only one of the terms has overflowed to +infinity

      // we can tollerate numbers very very slightly less than 1.  These make the log loss go down slightly as they lead to negative log
      // values, but we can tollerate this drop and rely on other features for the log loss calculation

      EBM_ASSERT(std::isnan(expFraction) || FloatEbmType { 1 } - k_epsilonLogLoss <= expFraction);

      const FloatEbmType ret = EbmLog(expFraction);

      // we're never taking the log of any number close to a negative, so we won't get a NaN result here UNLESS expFraction was already NaN and we're NaN 
      // propegating

      // we're using two numbers that probably can't be represented by exact representations
      // so, the fraction might be a tiny bit smaller than one, in which case the output would be a tiny
      // bit negative.  We can just let other subsequent adds cover this up
      EBM_ASSERT(std::isnan(ret) || -k_epsilonLogLoss <= ret); // log(1) == 0
      return ret;
   }

   EBM_INLINE static FloatEbmType ComputeSingleInstanceSquaredErrorRegression(const FloatEbmType residualError) {
      // this IS a performance critical function.  It gets called per validation instance!

      // residualError can be +-infinity, or NaN.  See note in ComputeSmallChangeForOneSegmentRegression

      // we are confirmed to get the same mean squared error value as scikit-learn for regression
      return residualError * residualError;

      // residualError can be anything from 0 to +infinity, or NaN
   }
};

#endif // EBM_STATISTICS_H