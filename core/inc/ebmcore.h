// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef EBMCORE_H
#define EBMCORE_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include <inttypes.h>

//#define EXPAND_BINARY_LOGITS
// TODO: implement REDUCE_MULTICLASS_LOGITS
//#define REDUCE_MULTICLASS_LOGITS
#if defined(EXPAND_BINARY_LOGITS) && defined(REDUCE_MULTICLASS_LOGITS)
#error we should not be expanding binary logits while reducing multiclass logits
#endif

#if defined(__clang__) || defined(__GNUC__)

#define EBMCORE_IMPORT_EXPORT __attribute__ ((visibility ("default")))
#define EBMCORE_CALLING_CONVENTION

#elif defined(_MSC_VER) // compiler type

#ifdef EBMCORE_EXPORTS
// we use a .def file in Visual Studio because we can remove the C name mangling entirely (in addition to C++ name mangling), unlike __declspec(dllexport)
#define EBMCORE_IMPORT_EXPORT
#else // EBMCORE_EXPORTS
// __declspec(dllimport) is optional, but having it allows the compiler to make the resulting code more efficient when imported
#define EBMCORE_IMPORT_EXPORT __declspec(dllimport)
#endif // EBMCORE_EXPORTS

#ifdef _WIN64
// _WIN32 is defined even for 64 bit compilations for compatibility, so use _WIN64
// in Windows, __fastcall is used for x64 always.  We don't need to define it, so let's leave it blank for future compatibility (not specifying it means it can be the new default if somehting new comes along later)
#define EBMCORE_CALLING_CONVENTION
#else // _WIN64
// in Windows, __stdcall (otherwise known as WINAPI) is used for the Win32 OS functions.  It is precicely defined by Windows and all languages essentially support it within the Windows ecosystem since they all need to call win32 functions.  Not all languages support CDECL since that's a C/C++ specification.
#define EBMCORE_CALLING_CONVENTION __stdcall
#endif // _WIN64

#else // compiler type
#error compiler not recognized
#endif // compiler type

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

// BINARY VS MULTICLASS AND LOGIT REDUCTION
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
// - which target to zero the residuals/logits for:
//   - we should zero the 0th bin since when checking the target value we can do a comparison to zero which is easier than checking the .Length value which requires an extra register/variable
//   - in the python code we should change the order of the targets for multiclass.  Everything else being equal, and assuming there is no benefit regarding which class is zeroed, we should zero the domiant class since then we can avoid the call to exp(..) on the majority of the data
//   - we need to have the python re-order the target multiclass values, since the multiclass logits get exposed in the model tensor, so our caller needs to have the same indexes because we don't want to re-order these
//   - since we want the dominant class in the 0th index, we might as well have the python sort the target values in multiclass by the dominance
//   - binary classification doesn't benefit/require re-ordering, but we should consider doing it there too for consistency, but that leads to some oddities

// MISSING VALUES (put missing values in the 0th index)
// - mostly because when processing the tensor we can keep 1 bit to indicate if a dimension is missing and maybe 1 more bit to indicate if it's categorical vs having those bits PLUS the total count AND having to do subtraction to determine when to stop the ordinal values.  Missing is an unconditional branch which chooses either 0 or 1 which are special constants that are faster).  Also, if missing is first, when we start processing a tensor, we can have code that creates an initial split that DOESN'T need to consider if there are any other splits in the model (this is best for tensors again)
//    - concepts:
//       - binning float/dobule/int data in python/numpy consists of first calling np.histogram to get the cuts, then calling np.digitize to do the actually binning
//       - binning is done via binary search.  numpy might send the data to c++.  massive amounts of binary searching is probably better in c++, but for most datasets, we can implement this directly in python efficiently
//       - if you use numpy to do the binning, then it creates the bins and turns missing into NaN values.  We think the NaN values are either stored as indexes to missing values or as a bitfield array.  After generating the binned dataset, if using numpy to do the binning you need to then postprocess the data by turning missing values into either zero or N, and if using zero for missing you need to increment all the other values.  It seems that doing your own binning would be better since it's easy and efficient to do
//    - arguments for putting at end :
//       - On the graph, we'd want missing to go on the right since on the left it would shift any graphs to the right or require whitespace when no missing values are present
//       - if when graphing we put the missing value on the right, then maybe we should store it in the model that way(in the Nth position)
//       - for non-missing values, you need a check to know if you start from the 0th or 1st position, but that can be done in a non - branching way(isMissing ? 1 : 0), but in any case it might be more confusing code wise since you might miss the need to start from a different index. BUT, maybe you don't gain much since if the missing is at the end then you need to have you stopping condition (isMissing ? count - 1 : count), which requries a bit more math and doesn't use the zero value which is more efficient in non - branching instructions
//       - one drawback is that if you put it in the 0th position, then you need to mentally shift all the values upwards only when there is a missing value
//    - arguments for putting at start :
//       - when looking in debugger, it's nice to have the exceptional condition first where you can see it easily
//       - having 0 as missing might be ok since it's always at an exact point, whereas we need to keep the index for missing otherwise
//       - probably having 0 as missing is the best since when converting python values to indexes, we'll probably use a hashtable or a sorted list or something, and there we can just code everything to the index without worrying about missing, BUT if we see a missing we just know to put it in the 0th bin without keeping an extra variable indicating the index position
//       - when binning, we don't care if any of the bins are missing (this is the slow part for mains, so for mains there probably isn't any benefit to putting missing at the start)
//       - we want binning to be efficient, so we won't pass in -1 values.  We'll pass in either 0 or (count - 1) values to indicate missing
//       - when cutting mains, it might be slighly nicer to have the missing value in the 0 position since you can do any work on the missing value before entering any loop to process the ordinal values, so you don't have to carry some state over the loop (you can do it before register pressure increases with loop variables, etc).
//       - when cutting N - dimensions, it becomes a lot nicer to have missing in the 0 bin since then we just need to store a single bit to indicate if the tensor's first value is missing.  If we were to put the missing value in the (count - 1) position, we'd need to store the count, the bit if missing, and do some math to calculate the non - missing cut point.All of this is bad
//       - we'll probably want to have special categorical processing since each slice in a tensoor can be considered completely independently.  I don't see any reason to have intermediate versions where we have 3 missing / categorical values and 4 ordinal values
//       - if missing is in the 0th bin, we can do any cuts at the beginning of processing a range, and that means any cut in the model would be the first, so we can initialze it by writing the cut model directly without bothering to handle inserting into the tree at the end

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
