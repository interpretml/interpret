// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef EBMCORE_H
#define EBMCORE_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include <inttypes.h>

//#define TREAT_BINARY_AS_MULTICLASS

#if defined(__clang__) || defined(__GNUC__)

#define EBMCORE_IMPORT_EXPORT __attribute__ ((visibility ("default")))
#define EBMCORE_CALLING_CONVENTION

#elif defined(_MSC_VER) /* compiler type */

#ifdef EBMCORE_EXPORTS
// we use a .def file in Visual Studio because we can remove the C name mangling entirely (in addition to C++ name mangling), unlike __declspec(dllexport)
#define EBMCORE_IMPORT_EXPORT
#else // EBMCORE_EXPORTS
// __declspec(dllimport) is optional, but having it allows the compiler to make the resulting code more efficient when imported
#define EBMCORE_IMPORT_EXPORT __declspec(dllimport)
#endif // EBMCORE_EXPORTS

// in Windows, __fastcall is used for x64 always, so let's use __fastcall for x86 as well to keep things consistent
#define EBMCORE_CALLING_CONVENTION __fastcall

#else // compiler
#error compiler not recognized
#endif // compiler

typedef struct {
   // this struct is to enforce that our caller doesn't mix EbmTraining and EbmInteraction pointers.  In C/C++ languages the caller will get an error if they try to mix these pointer types.
   char unused;
} *PEbmTraining;
typedef struct {
   // this struct is to enforce that our caller doesn't mix EbmTraining and EbmInteraction pointers.  In C/C++ languages the caller will get an error if they try to mix these pointer types.
   char unused;
} *PEbmInteraction;

typedef double FractionalDataType;
#define FractionalDataTypePrintf "f"
typedef int64_t IntegerDataType;
#define IntegerDataTypePrintf PRId64

const IntegerDataType AttributeTypeOrdinal = 0;
const IntegerDataType AttributeTypeNominal = 1;

typedef struct {
   IntegerDataType attributeType;
   IntegerDataType hasMissing;
   IntegerDataType countStates;
} EbmAttribute;

typedef struct {
   IntegerDataType countAttributesInCombination;
} EbmAttributeCombination;

const signed char TraceLevelOff = 0; // no messages will be output.  SetLogMessageFunction doesn't need to be called if the level is left at this value
const signed char TraceLevelError = 1;
const signed char TraceLevelWarning = 2;
const signed char TraceLevelInfo = 3;
const signed char TraceLevelVerbose = 4;

// all our logging messages are pure ascii (127 values), and therefore also UTF-8
typedef void (EBMCORE_CALLING_CONVENTION * LOG_MESSAGE_FUNCTION)(signed char traceLevel, const char * message);

EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION SetLogMessageFunction(LOG_MESSAGE_FUNCTION logMessageFunction);
EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION SetTraceLevel(signed char traceLevel);

// - I initially considered storing our model files as negated logits [storing them as (0 - mathematical_logit)], but that's a bad choice because:
//   - if you use the wrong formula, you need a negation for binary classification, but the best formula requires a logit without negation 
//     - for calculating binary classification you can use one of these formulas:
//       - prob = 1/(1+e^(-logit)) -> this requries getting the negative logit first, which was the thinking behind storing a negative logit, BUT there is a better computation:
//       - odds = exp(logit); prob = odds / (1 + odds); -> this is the better form.  Calculating the exp(logit) happens first, then we can load a register with 1, and then add the odds into the register with 1, then divide our odds register with the odds plus 1 register.  We need to do a load of 1 into a register, but we need to do that anyways in the alternate form above.  The above alternate form is worse, since we need to load a 1 into a register for the numerator AND copy that to a second register for the denominator addition
//     - So, using a non-negated logit requires 1 less assembly copy instruction (duplicating the 1 value)
//   - for multiclass classification, normal non-negated logits are better, since the forumla is:
//     - prob_i = exp(logit_i) / SUM(logit_1 .. logit_N)
//   - so non-negated logits are better for both binary classification and multiclass classification
//   - TODO: is this a useful property, or in otherwords, do we want to keep negated logits in our "small update model" => the model logits that we expose to python are kept very separate from the "small update logits" that we use internally in C++ for calculating changes to residuals and for calculating metrics.  We could in theory keep the small update logits as negative internally in C++ if that was useful, and then when it came time to update our public models, we could subtract the negated small update logits.  The code that merges the "small update logits" into the public models is the SegmentedRegion::Add(..) function, but we could have a SegmentedRegion::Subtract(..) function that would be equally efficient if we wanted to do that.  In that function we don't benefit from the assembly add property that you get to choose one of the original values to keep when adding two registers together since we can either just load both items into registers and do the subtract, or we can load the value we will subtract into a register and subract that from the pointer indirection to the target memory location.
//   - clearly, keeping normal logits instead of negated logits will be less confusing
//   - keeping negated logits would be even more confusing since we want to keep non-negated values for regression models
//   - when calling InitializeTrainingClassification, the trainingPredictionScores and validationPredictionScores values would logically need to be negated for consistency with the models if we stored the models as negated, so it would be even more confusing
//   - even if it were better to keep negated logits, in order to calculate a probabily from a model, you need to loop over all the "attribute combinations" and get the logit for that "attribute combination" to sum them all together for the combined logit, and that work is going to be far far greater than negating a logit at the end, so whether we keep negated or non-negated logits isn't a big deal computationally
// - shifting logits
//   - for multiclass, we only require K-1 logits for a K-class prediction problem.  If we use K logits, then we can shift all the logits together at will in any particular case/bin WITHOUT changing the intercept by adding a constant accross all logits within the bin.  If we have K-1 logits, then one of the logits is implicitly zero and the others are forced into the only values that make sense relative to the zero by having shifted all the logits so that one of the bins/cases is zero
//   - we can also shift all the logits together (even after reduction to K-1 logits) for any feature by shifting the model's intercept (this allows us to move the graphs up and down)
//   - we center the binary classification graphs by creating/moving an intercept term.  This helps us visually compare different graphs
//   - TODO: figure out this story -> we can't center all lines in a mutlti-class problem, but we'll do something here to center some kind of metric
//   - Xuezhou's method allows us to take any set of K logits and convert them into intelligible graphs using his axioms.  The resulting graph is UNIQUE for any equivalent model, so we can take a reduce K-1 set of logits and reliable create intelligible graphs with K logits
// - intelligible graphing
//   - when graphing multiclass, we should show K lines on the graph, since otherwise one line would always be zero and the other lines would be relative to the zero line.  We use Xuezhou's method to make the K lines intelligible
//   - for binary classification, if we force our model to have 2 logits (K logits instead of K-1), then the logits will be equal and opposite in sign.  This provides no intelligible benefit.  It's better to think of the problem have a default outcome, and the other more rare outcome as what we're interested in.  In that case we only want to present the user with one graph line, and one logit
// - when saving model files to disk, we want the file format to be intelligible too so that the user can modify it there if they want, so
//   - for multiclass, we should use Xuezhou's method AND center any logits before saving them so that they match the graphs
//   - for binary classification, we should save only 1 logit AFTER centering (by changing the intercept term).  Then the logits will match the graphs presented to the user
//   - Unfortunately, this means that a 3-state problem has 3 logits but a binary class problem (2 states) has only 1, so it isn't consistent.  I think though that human intelligibility is more important here than machine consistency when processing numbers.  Also, in most cases when someone else writes code to process our logits, they'll know ahead of time if they need binary or multi-class classification, so they probably won't have to handle both cases anyways.  We probably want to handle them with different code ourselves anyways since it's more efficient to store less data.
//   - it should be legal for the user two tweak the logits in the file, so we should apply Xuezhou method after loading any model files, and if we output the same model we should output it with the corrected Xuezhou method logits (the user can re-tweak them if desired)
// - for binary and multiclass, the target values:
//   - the target variable is a nominal, so the ordering doesn't matter, and the specific value assigned doesn't matter
//   - most binary prediction problems and most multiclass problems will have a dominant case that can be thought of as the "default".  0 is the best value for the default because:
//     - if 0 is always the default accross multiple problem that you're examining, then you don't have to lookup what the maximum value is to know if it's the default.  If you have a 5 state problem or a 12 state problem, you want the default for both to be zero instead of 5 and 12 respectively, which makes it hard to visually scan the data for anomylies
//     - 0 is generally considered false, and 1 is generally considered true.  If we're predicting something rare, then 1 means it happened, which is consistent with what we want
//     - if you were to hash the data, the most likely first value you'll see is the default case, which will then get the 0 value (assuming you use a counter to assign values)
//     - 0 compresses better when zipped, so it's better to have a non-rare 0 value
//     - sometimes datasets are modified by taking an existing output and splitting it into two outputs to have more information.  Usually a rare case will be split, in which case it's better for the non-modified default case to remain zero since one of the split off output values will usually be given the next higher value.  If the default case was the value N-1 then when someone adds a new bin, the N-1 will now be a rare output.  If the default bin were split up, the person doing the splitting can put the rarest of the two types into the N-1 position, keeping the default target as 0
// - for multiclass, when zeroing [turning K logits into K-1 logits], we want to make the first logit as zero because:
//   - the logits are more intelligible for multiclass in this format because as described above, zero should generally be the default case in the data.  If we could only have K-1 graphs, we'd generally want to know how far from the default some prediction was.  Eg: if we had a highly dominant case like in SIDS (survive), and two very rare cases that occur at about the same rate, the dominant case isn't likely to change much over the graph, but the rare cases might change a lot, so we want to not radically jiggle the graph based on changes to an unlikley outcome.  If we had used one of the rare cases as the default it's likelyhood can change many orders of magnitude over the graph, but the default case will not typically have such large swings.
//   - when converting from K logits to K-1, we can process memory in order.  WE read the [0] index and store that value, then start a loop from the [1] index subtracing the value that we stored from the [0] index
//   - when converting from K-1 logits to K, if we're leaving the first of the K logits as zero, we need to copy to new memory anyways, so we just set the first logit to zero and copy the remaining logits in order.  There is no benefit to putting the zero at the start or end in this case since either way allows us to read the origin array in order, and write in order
//   - when converting from K-1 logits to K logits using Xuezhou's method, we'll be need to loop over all the K-1 logits anyways for calculations and from that we can calculate the 0th bin value which we can write first to the [0]th bin in our new memory and then re-loop over our logits writing them to the new memory.  In this case we don't need to store the [0]th bin value while we're looping again over the logits since we can just write it to memory and forget about it.  This one is slightly better for having the 0 bin zeroed
//   - if in the future when we get to zeroing residuals we find that it's better to zero the minor bin (or major), we can re-order the target numbers as we like since their order does not matter. We can put the major or minor case in the zero bin, and still get the benefits above.  If we find that there is a consistent benefit through some metric, we'll want to always reorder anyways
// - binary classification can be thought of as multiclass classification with 2 bins.  For binary classification, we probably also want to make the FIRST hidden bin (we only present 1 logit) as the implicit zero value because
//   - it's consistent with multiclass
//   - if in our original data the 0 value is the default case, then we really want to graph and are interested in the non-default case most of the time.  That means we want the default case zeroed [the 1st bin which is zero], and graph/report the 2nd bin [which is in the array index 1, and non-zero].  Eg: in the age vs death graph, we want the prediction from the logit to predict death and we want it increasing with increasing age.  That means our logit should be the FIRST bin.

EBMCORE_IMPORT_EXPORT PEbmTraining EBMCORE_CALLING_CONVENTION InitializeTrainingRegression(IntegerDataType randomSeed, IntegerDataType countAttributes, const EbmAttribute * attributes, IntegerDataType countAttributeCombinations, const EbmAttributeCombination * attributeCombinations, const IntegerDataType * attributeCombinationIndexes, IntegerDataType countTrainingCases, const FractionalDataType * trainingTargets, const IntegerDataType * trainingData, const FractionalDataType * trainingPredictionScores, IntegerDataType countValidationCases, const FractionalDataType * validationTargets, const IntegerDataType * validationData, const FractionalDataType * validationPredictionScores, IntegerDataType countInnerBags);
EBMCORE_IMPORT_EXPORT PEbmTraining EBMCORE_CALLING_CONVENTION InitializeTrainingClassification(IntegerDataType randomSeed, IntegerDataType countAttributes, const EbmAttribute * attributes, IntegerDataType countAttributeCombinations, const EbmAttributeCombination * attributeCombinations, const IntegerDataType * attributeCombinationIndexes, IntegerDataType countTargetStates, IntegerDataType countTrainingCases, const IntegerDataType * trainingTargets, const IntegerDataType * trainingData, const FractionalDataType * trainingPredictionScores, IntegerDataType countValidationCases, const IntegerDataType * validationTargets, const IntegerDataType * validationData, const FractionalDataType * validationPredictionScores, IntegerDataType countInnerBags);
EBMCORE_IMPORT_EXPORT IntegerDataType EBMCORE_CALLING_CONVENTION TrainingStep(PEbmTraining ebmTraining, IntegerDataType indexAttributeCombination, FractionalDataType learningRate, IntegerDataType countTreeSplitsMax, IntegerDataType countCasesRequiredForSplitParentMin, const FractionalDataType * trainingWeights, const FractionalDataType * validationWeights, FractionalDataType * validationMetricReturn);
EBMCORE_IMPORT_EXPORT FractionalDataType * EBMCORE_CALLING_CONVENTION GetCurrentModel(PEbmTraining ebmTraining, IntegerDataType indexAttributeCombination);
EBMCORE_IMPORT_EXPORT FractionalDataType * EBMCORE_CALLING_CONVENTION GetBestModel(PEbmTraining ebmTraining, IntegerDataType indexAttributeCombination);
EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION CancelTraining(PEbmTraining ebmTraining);
EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION FreeTraining(PEbmTraining ebmTraining);

EBMCORE_IMPORT_EXPORT PEbmInteraction EBMCORE_CALLING_CONVENTION InitializeInteractionRegression(IntegerDataType countAttributes, const EbmAttribute * attributes, IntegerDataType countCases, const FractionalDataType * targets, const IntegerDataType * data, const FractionalDataType * predictionScores);
EBMCORE_IMPORT_EXPORT PEbmInteraction EBMCORE_CALLING_CONVENTION InitializeInteractionClassification(IntegerDataType countAttributes, const EbmAttribute * attributes, IntegerDataType countTargetStates, IntegerDataType countCases, const IntegerDataType * targets, const IntegerDataType * data, const FractionalDataType * predictionScores);
EBMCORE_IMPORT_EXPORT IntegerDataType EBMCORE_CALLING_CONVENTION GetInteractionScore(PEbmInteraction ebmInteraction, IntegerDataType countAttributesInCombination, const IntegerDataType * attributeIndexes, FractionalDataType * interactionScoreReturn);
EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION CancelInteraction(PEbmInteraction ebmInteraction);
EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION FreeInteraction(PEbmInteraction ebmInteraction);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif  // EBMCORE_H
