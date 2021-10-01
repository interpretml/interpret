// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// to minimize confusion, we try whenever possible to use common terms with scikit-learn:
// https://scikit-learn.org/stable/glossary.html

#ifndef EBM_NATIVE_H
#define EBM_NATIVE_H

#include <inttypes.h>
#include <float.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#define EBM_EXTERN_C  extern "C"
#define STATIC_CAST(__type, __val)  (static_cast<__type>(__val))
#else // __cplusplus
#define EBM_EXTERN_C
#define STATIC_CAST(__type, __val)  ((__type)(__val))
#endif // __cplusplus

#define EBM_BOOL_CAST(EBM_VAL)                     (STATIC_CAST(BoolEbmType, (EBM_VAL)))
#define EBM_ERROR_CAST(EBM_VAL)                    (STATIC_CAST(ErrorEbmType, (EBM_VAL)))
#define EBM_TRACE_CAST(EBM_VAL)                    (STATIC_CAST(TraceEbmType, (EBM_VAL)))
#define EBM_GENERATE_UPDATE_OPTIONS_CAST(EBM_VAL)  (STATIC_CAST(GenerateUpdateOptionsType, (EBM_VAL)))

//#define EXPAND_BINARY_LOGITS
// TODO: implement REDUCE_MULTICLASS_LOGITS
//#define REDUCE_MULTICLASS_LOGITS
#if defined(EXPAND_BINARY_LOGITS) && defined(REDUCE_MULTICLASS_LOGITS)
#error we should not be expanding binary logits while reducing multiclass logits
#endif

#if defined(__clang__) || defined(__GNUC__) || defined(__SUNPRO_CC)

#define EBM_NATIVE_IMPORT_EXPORT_INCLUDE extern

#ifdef EBM_NATIVE_R
// R has it's own way of exporting functions.  There is a single entry point that describes to 
// R how to call our functions.  Also, we export R specific functions rather than the generic 
// ones that we can consume from other languages
#define EBM_NATIVE_IMPORT_EXPORT_BODY EBM_EXTERN_C
#else // EBM_NATIVE_R
#define EBM_NATIVE_IMPORT_EXPORT_BODY EBM_EXTERN_C __attribute__ ((visibility ("default")))
#endif // EBM_NATIVE_R

#define EBM_NATIVE_CALLING_CONVENTION

#elif defined(_MSC_VER) // compiler type

#ifdef EBM_NATIVE_R 
// R has it's own way of exporting functions.  There is a single entry point that describes to 
// R how to call our functions.  Also, we export R specific functions rather than the generic 
// ones that we can consume from other languages
#define EBM_NATIVE_IMPORT_EXPORT_INCLUDE extern
#define EBM_NATIVE_IMPORT_EXPORT_BODY EBM_EXTERN_C
#else // EBM_NATIVE_R

#ifdef EBM_NATIVE_EXPORTS
// we use a .def file in Visual Studio because we can remove the C name mangling entirely, 
// in addition to C++ name mangling, unlike __declspec(dllexport)
#define EBM_NATIVE_IMPORT_EXPORT_INCLUDE extern
#define EBM_NATIVE_IMPORT_EXPORT_BODY EBM_EXTERN_C
#else // EBM_NATIVE_EXPORTS
// __declspec(dllimport) is optional, but having it allows the compiler to make the 
// resulting code more efficient when imported
#define EBM_NATIVE_IMPORT_EXPORT_INCLUDE extern __declspec(dllimport)
#endif // EBM_NATIVE_EXPORTS

#endif // EBM_NATIVE_R

#ifdef _WIN64
// _WIN32 is defined even for 64 bit compilations for compatibility, so use _WIN64
// in Windows, __fastcall is used for x64 always.  We don't need to define it, so let's leave it blank for 
// future compatibility.  Not specifying it means it can be the new default if somehting new comes along later
#define EBM_NATIVE_CALLING_CONVENTION
#else // _WIN64
// In Windows, __stdcall (otherwise known as WINAPI) is used for the Win32 OS functions.  It is precicely defined 
// by Windows and all languages essentially support it within the Windows ecosystem since they all need to call 
// win32 functions.  Not all languages support CDECL since that's a C/C++ specification.
#define EBM_NATIVE_CALLING_CONVENTION __stdcall
#endif // _WIN64

#else // compiler type
#error compiler not recognized
#endif // compiler type

typedef struct _BoosterHandle {
   // this struct exists to enforce that our caller doesn't mix handle types.
   // In C/C++ languages the caller will get an error if they try to mix these pointer types.
   char unused;
} * BoosterHandle;

typedef struct _InteractionHandle {
   // this struct exists to enforce that our caller doesn't mix handle types.
   // In C/C++ languages the caller will get an error if they try to mix these pointer types.
   char unused;
} * InteractionHandle;

#ifndef PRId32
// this should really be defined, but some compilers aren't compliant
#define PRId32 "d"
#endif // PRId32
#ifndef PRId64
// this should really be defined, but some compilers aren't compliant
#define PRId64 "lld"
#endif // PRId64
#ifndef PRIu64
// this should really be defined, but some compilers aren't compliant
#define PRIu64 "llu"
#endif // PRIu64
#ifndef PRIx64
// this should really be defined, but some compilers aren't compliant
#define PRIx64 "llx"
#endif // PRIx64

// std::numeric_limits<FloatEbmType>::max() -> big positive number
#define FLOAT_EBM_MAX            DBL_MAX
// std::numeric_limits<FloatEbmType>::lowest() -> big negative number.  True in IEEE-754, which we require
#define FLOAT_EBM_LOWEST         (-DBL_MAX)
// std::numeric_limits<FloatEbmType>::min() -> small positive number
#define FLOAT_EBM_MIN            DBL_MIN
// std::numeric_limits<FloatEbmType>::denorm_min() -> small positive number
//#define FLOAT_EBM_DENORM_MIN     DBL_TRUE_MIN -> not supported in g++ version of float.h for now (it's a C11 construct)
// std::numeric_limits<FloatEbmType>::infinity()
#define FLOAT_EBM_POSITIVE_INF   (STATIC_CAST(FloatEbmType, (INFINITY)))
// -std::numeric_limits<FloatEbmType>::infinity()
#define FLOAT_EBM_NEGATIVE_INF   (-FLOAT_EBM_POSITIVE_INF)
// std::numeric_limits<FloatEbmType>::quiet_NaN()
#define FLOAT_EBM_NAN            (STATIC_CAST(FloatEbmType, (NAN)))

// Smaller integers can safely roundtrip to double values and back, but this breaks down at
// exactly 2^53.  2^53 will convert from an integer to a correct double, but 2^53 + 1 will round
// down to 2^53 in IEEE-754 where banker's rounding is used.  So, if we see an integer with 
// 9007199254740992 we would be safe to convert to to a double, BUT if we see a double of
// 9007199254740992.0 we don't know if that was originally 9007199254740992 or 9007199254740993, so
// 9007199254740992 is unsafe if checking as a double.
// https://stackoverflow.com/questions/1848700/biggest-integer-that-can-be-stored-in-a-double
// R has a lower maximum index of 4503599627370496 (R_XLEN_T_MAX) probably to store a bit somewhere.
#define SAFE_FLOAT64_AS_INT_MAX   9007199254740991
// the maximum signed int64 value is 9223372036854775807, BUT numbers above 9223372036854775295 round in IEEE-754
// to a number above that, so if we're converting from float64 to int64, the maximum safe number is 9223372036854775295
// But when we convert 9223372036854775295 to float64 we loose precision and if we output it with 17 decimal digits
// which is the universal round trip format for float64 in IEEE-754, then we get 9.2233720368547748e+18
// When you accurately round that biggest representable float64 to the closest integer, you get 9223372036854774784,
// which having 19 digits is legal as an exact IEEE-754 number since it's within the 17-20 digits that is required
// to accurately round to idenical floats
#define FLOAT64_TO_INT_MAX   9223372036854774784

// TODO: look through our code for places where SAFE_FLOAT64_AS_INT_MAX or FLOAT64_TO_INT_MAX would be useful

// TODO: we can eliminate FloatEbmType and make our interface entirely doubles.  JSON only supports doubles, and it's
// the most cross-language and highest precision type commonly available.  We can move FloatEbmType into our
// internal interface and use it to switch our score representation which is the only place we'd benefit from float32
typedef double FloatEbmType;
// this needs to be in "le" format, since we internally use that format to generate "interpretable" 
// floating point numbers in text format.   See Discretization.cpp for details.
#define FloatEbmTypePrintf "le"
typedef int64_t IntEbmType;
#define IntEbmTypePrintf PRId64
typedef uint64_t UIntEbmType;
#define UIntEbmTypePrintf PRIu64
typedef int32_t SeedEbmType;
#define SeedEbmTypePrintf PRId32
typedef int32_t TraceEbmType;
#define TraceEbmTypePrintf PRId32
typedef int64_t BoolEbmType;
#define BoolEbmTypePrintf PRId64
typedef int32_t ErrorEbmType;
#define ErrorEbmTypePrintf PRId32
typedef int64_t GenerateUpdateOptionsType;
// technically printf hexidecimals are unsigned, so convert it first to unsigned before calling printf
typedef uint64_t UGenerateUpdateOptionsType;
#define UGenerateUpdateOptionsTypePrintf PRIx64

#define EBM_FALSE          (EBM_BOOL_CAST(0))
#define EBM_TRUE           (EBM_BOOL_CAST(1))

#define Error_None                                 (EBM_ERROR_CAST(0))
#define Error_OutOfMemory                          (EBM_ERROR_CAST(-1))
// errors occuring entirely within the C/C++ code
#define Error_UnexpectedInternal                   (EBM_ERROR_CAST(-2))
// input parameters received that are clearly due to bugs in the higher level caller
#define Error_IllegalParamValue                    (EBM_ERROR_CAST(-3))
// input parameters received from the end user that are illegal.  These should have been filtered by our caller
#define Error_UserParamValue                       (EBM_ERROR_CAST(-4))
#define Error_ThreadStartFailed                    (EBM_ERROR_CAST(-5))

#define Error_LossConstructorException             (EBM_ERROR_CAST(-10))
#define Error_LossParamUnknown                     (EBM_ERROR_CAST(-11))
#define Error_LossParamValueMalformed              (EBM_ERROR_CAST(-12))
#define Error_LossParamValueOutOfRange             (EBM_ERROR_CAST(-13))
#define Error_LossParamMismatchWithConfig          (EBM_ERROR_CAST(-14))
#define Error_LossUnknown                          (EBM_ERROR_CAST(-15))
#define Error_LossIllegalRegistrationName          (EBM_ERROR_CAST(-16))
#define Error_LossIllegalParamName                 (EBM_ERROR_CAST(-17))
#define Error_LossDuplicateParamName               (EBM_ERROR_CAST(-18))

#define GenerateUpdateOptions_Default              (EBM_GENERATE_UPDATE_OPTIONS_CAST(0x0000000000000000))
#define GenerateUpdateOptions_DisableNewtonGain    (EBM_GENERATE_UPDATE_OPTIONS_CAST(0x0000000000000001))
#define GenerateUpdateOptions_DisableNewtonUpdate  (EBM_GENERATE_UPDATE_OPTIONS_CAST(0x0000000000000002))
#define GenerateUpdateOptions_GradientSums         (EBM_GENERATE_UPDATE_OPTIONS_CAST(0x0000000000000004))
#define GenerateUpdateOptions_RandomSplits         (EBM_GENERATE_UPDATE_OPTIONS_CAST(0x0000000000000008))

 // no messages will be output
#define TraceLevelOff      (EBM_TRACE_CAST(0))
// invalid inputs to the C library or assert failure before exit
#define TraceLevelError    (EBM_TRACE_CAST(1))
// out of memory or other conditions we can't continue after
#define TraceLevelWarning  (EBM_TRACE_CAST(2))
// odd inputs like features with 1 value or empty feature groups
#define TraceLevelInfo     (EBM_TRACE_CAST(3))
// function calls, logging that helps us trace execution in the library
#define TraceLevelVerbose  (EBM_TRACE_CAST(4))

// all our logging messages are pure ASCII (127 values), and therefore also conform to UTF-8
typedef void (EBM_NATIVE_CALLING_CONVENTION * LOG_MESSAGE_FUNCTION)(TraceEbmType traceLevel, const char * message);

// SetLogMessageFunction does not need to be called if the level is left at TraceLevelOff
EBM_NATIVE_IMPORT_EXPORT_INCLUDE void EBM_NATIVE_CALLING_CONVENTION SetLogMessageFunction(
   LOG_MESSAGE_FUNCTION logMessageFunction
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE void EBM_NATIVE_CALLING_CONVENTION SetTraceLevel(TraceEbmType traceLevel);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE const char * EBM_NATIVE_CALLING_CONVENTION GetTraceLevelString(TraceEbmType traceLevel);

// BINARY VS MULTICLASS AND LOGIT REDUCTION
// - I initially considered storing our model files as negated logits [storing them as (0 - mathematical_logit)], but that's a bad choice because:
//   - if you use the wrong formula, you need a negation for binary classification, but the best formula requires a logit without negation 
//     - for calculating binary classification you can use one of these formulas:
//       - prob = 1/(1+e^(-logit)) -> this requries getting the negative logit first, which was the thinking behind storing a negative logit, 
//         BUT there is a better computation:
//       - odds = exp(logit); prob = odds / (1 + odds); -> this is the better form.  Calculating the exp(logit) happens first, then we can load a 
//         register with 1, and then add the odds into the register with 1, then divide our odds register with the odds plus 1 register.  
//         We need to do a load of 1 into a register, but we need to do that anyways in the alternate form above.  The above alternate form is worse, 
//         since we need to load a 1 into a register for the numerator AND copy that to a second register for the denominator addition
//     - So, using a non-negated logit requires 1 less assembly copy instruction (duplicating the 1 value)
//   - for multiclass classification, normal non-negated logits are better, since the forumla is:
//     - prob_i = exp(logit_i) / SUM(logit_1 .. logit_N)
//   - so non-negated logits are better for both binary classification and multiclass classification
//   - TODO: is this a useful property, or in otherwords, do we want to keep negated logits in our "small update model" => the model logits that we expose
//     to python are kept very separate from the "small update logits" that we use internally in C++ for calculating changes to scores and for 
//     calculating metrics.  We could in theory keep the small update logits as negative internally in C++ if that was useful, and then when it came 
//     time to update our public models, we could subtract the negated small update logits.  The code that merges the "small update logits" into the 
//     public models is the SegmentedRegion::Add(..) function, but we could have a SegmentedRegion::Subtract(..) function that would be equally efficient 
//     if we wanted to do that.  In that function we don't benefit from the assembly add property that you get to choose one of the original values to 
//     keep when adding two registers together since we can either just load both items into registers and do the subtract, or we can load the value we 
//     will subtract into a register and subract that from the pointer indirection to the target memory location.
//   - clearly, keeping normal logits instead of negated logits will be less confusing
//   - keeping negated logits would be even more confusing since we want to keep non-negated values for regression models
//   - when calling CreateClassificationBooster, the trainingPredictorScores and validationPredictorScores values would logically need to be negated 
//     for consistency with the models if we stored the models as negated, so it would be even more confusing
//   - even if it were better to keep negated logits, in order to calculate a probabily from a model, you need to loop over all the "feature groups" 
//     and get the logit for that "feature group" to sum them all together for the combined logit, and that work is going to be far far greater 
//     than negating a logit at the end, so whether we keep negated or non-negated logits isn't a big deal computationally
// - shifting logits
//   - for multiclass, we only require K-1 logits for a K-class prediction problem.  If we use K logits, then we can shift all the logits together at will 
//     in any particular sample/bin WITHOUT changing the intercept by adding a constant accross all logits within the bin.  If we have K-1 logits, then 
//     one of the logits is implicitly zero and the others are forced into the only values that make sense relative to the zero by having shifted all the 
//     logits so that one of the bins/samples is zero
//   - we can also shift all the logits together (even after reduction to K-1 logits) for any feature by shifting the model's intercept 
//     (this allows us to move the graphs up and down)
//   - we center the binary classification graphs by creating/moving an intercept term.  This helps us visually compare different graphs
//   - TODO: figure out this story -> we can't center all lines in a mutlti-class problem, but we'll do something here to center some kind of metric
//   - Xuezhou's method allows us to take any set of K logits and convert them into intelligible graphs using his axioms.  The resulting graph is UNIQUE 
//     for any equivalent model, so we can take a reduce K-1 set of logits and reliable create intelligible graphs with K logits
// - intelligible graphing
//   - when graphing multiclass, we should show K lines on the graph, since otherwise one line would always be zero and the other lines would be relative 
//     to the zero line.  We use Xuezhou's method to make the K lines intelligible
//   - for binary classification, if we force our model to have 2 logits (K logits instead of K-1), then the logits will be equal and opposite in sign.
//     This provides no intelligible benefit.  It's better to think of the problem have a default outcome, and the other more rare outcome as what 
//     we're interested in.  In that case we only want to present the user with one graph line, and one logit
// - when saving model files to disk, we want the file format to be intelligible too so that the user can modify it there if they want, so
//   - for multiclass, we should use Xuezhou's method AND center any logits before saving them so that they match the graphs
//   - for binary classification, we should save only 1 logit AFTER centering (by changing the intercept term).  Then the logits will match the graphs 
//     presented to the user
//   - Unfortunately, this means that a 3-class problem has 3 logits but a binary class problem (2 classes) has only 1, so it isn't consistent.  
//     I think though that human intelligibility is more important here than machine consistency when processing numbers.  Also, in most cases when 
//     someone else writes code to process our logits, they'll know ahead of time if they need binary or multi-class classification, so they probably 
//     won't have to handle both cases anyways.  We probably want to handle them with different code ourselves anyways since it's more efficient to 
//     store less data.
//   - it should be legal for the user two tweak the logits in the file, so we should apply Xuezhou method after loading any model files, and if we 
//     output the same model we should output it with the corrected Xuezhou method logits (the user can re-tweak them if desired)
// - for binary and multiclass, the target values:
//   - the target variable is a nominal, so the ordering doesn't matter, and the specific value assigned doesn't matter
//   - most binary prediction problems and most multiclass problems will have a dominant class that can be thought of as the "default".  0 is the best 
//     value for the default because:
//     - if 0 is always the default accross multiple problem that you're examining, then you don't have to lookup what the maximum value is to know if 
//       it's the default.  If you have a 5 class problem or a 12 class problem, you want the default for both to be zero instead of 5 and 12 respectively, 
//       which makes it hard to visually scan the data for anomylies
//     - 0 is generally considered false, and 1 is generally considered true.  If we're predicting something rare, then 1 means it happened, which is 
//       consistent with what we want
//     - if you were to hash the data, the most likely first value you'll see is the default class, which will then get the 0 value 
//       (assuming you use a counter to assign values)
//     - 0 compresses better when zipped, so it's better to have a non-rare 0 value
//     - sometimes datasets are modified by taking an existing output and splitting it into two outputs to have more information.  Usually a rare class 
//       will be split, in which case it's better for the non-modified default class to remain zero since one of the split off output values will 
//       usually be given the next higher value.  If the default class was the value N-1 then when someone adds a new bin, the N-1 will now be a rare 
//       output.  If the default bin were split up, the person doing the splitting can put the rarest of the two types into the N-1 position, 
//       keeping the default target as 0
// - for multiclass, when zeroing [turning K logits into K-1 logits], we want to make the first logit as zero because:
//   - the logits are more intelligible for multiclass in this format because as described above, zero should generally be the default class in the data.
//     If we could only have K-1 graphs, we'd generally want to know how far from the default some prediction was.  Eg: if we had a highly dominant class 
//     like in SIDS (survive), and two very rare classes that occur at about the same rate, the dominant class isn't likely to change much over the graph, 
//     but the rare classes might change a lot, so we want to not radically jiggle the graph based on changes to an unlikley outcome.  If we had used one 
//     of the rare classes as the default it's likelyhood can change many orders of magnitude over the graph, but the default class will not typically have 
//     such large swings.
//   - when converting from K logits to K-1, we can process memory in order.  WE read the [0] index and store that value, then start a loop from 
//     the [1] index subtracing the value that we stored from the [0] index
//   - when converting from K-1 logits to K, if we're leaving the first of the K logits as zero, we need to copy to new memory anyways, so we just set the 
//     first logit to zero and copy the remaining logits in order.  There is no benefit to putting the zero at the start or end in this case since either 
//     way allows us to read the origin array in order, and write in order
//   - when converting from K-1 logits to K logits using Xuezhou's method, we'll be need to loop over all the K-1 logits anyways for calculations and from 
//     that we can calculate the 0th bin value which we can write first to the [0]th bin in our new memory and then re-loop over our logits writing them 
//     to the new memory.  In this case we don't need to store the [0]th bin value while we're looping again over the logits since we can just write it 
//     to memory and forget about it.  This one is slightly better for having the 0 bin zeroed
//   - if in the future when we get to zeroing logits we find that it's better to zero the minor bin (or major), we can re-order the target numbers as 
//     we like since their order does not matter. We can put the major or minor class in the zero bin, and still get the benefits above.  If we find 
//     that there is a consistent benefit through some metric, we'll want to always reorder anyways
// - binary classification can be thought of as multiclass classification with 2 bins.  For binary classification, we probably also want to make the FIRST 
//     hidden bin (we only present 1 logit) as the implicit zero value because:
//   - it's consistent with multiclass
//   - if in our original data the 0 value is the default class, then we really want to graph and are interested in the non-default class most of the time.
//     That means we want the default class zeroed [the 1st bin which is zero], and graph/report the 2nd bin [which is in the array index 1, and non-zero].
//     Eg: in the age vs death graph, we want the prediction from the logit to predict death and we want it increasing with increasing age.  
//     That means our logit should be the FIRST bin.
// - which target to zero the logits for:
//   - we should zero the 0th bin since when checking the target value we can do a comparison to zero which is easier than checking the .Length value
//     which requires an extra register/variable
//   - in the python code we should change the order of the targets for multiclass.  Everything else being equal, and assuming there is no benefit 
//     regarding which class is zeroed, we should zero the domiant class since then we can avoid the call to exp(..) on the majority of the data
//   - we need to have the python re-order the target multiclass values, since the multiclass logits get exposed in the model tensor, so our caller needs 
//     to have the same indexes because we don't want to re-order these
//   - since we want the dominant class in the 0th index, we might as well have the python sort the target values in multiclass by the dominance
//   - binary classification doesn't benefit/require re-ordering, but we should consider doing it there too for consistency, but that leads to some oddities


// TODO: we should change our interface such that long running work items will return instantly but are working on
//       a background thread.  The caller will get back a token to the work.  They can either start a number of
//       work items simultaneously, or call a blocking function that waits on any/all work items to complete.
//       The log in this world would be a circular buffer and wouldn't be writtent out unless the C++ code was
//       controlling the main thread (either during calls to the non-blocking components, or while the caller is
//       in the waiting function).  We would drop anything that exceeds the circular buffer.  This allows us to have
//       threaded code inside non-threaded languages.

EBM_NATIVE_IMPORT_EXPORT_INCLUDE SeedEbmType EBM_NATIVE_CALLING_CONVENTION GenerateRandomNumber(
   SeedEbmType randomSeed,
   SeedEbmType stageRandomizationMix
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE IntEbmType EBM_NATIVE_CALLING_CONVENTION GetHistogramCutCount(
   IntEbmType countSamples,
   const double * featureValues,
   IntEbmType strategy
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CutQuantile(
   IntEbmType countSamples,
   const FloatEbmType * featureValues,
   IntEbmType countSamplesPerBinMin,
   BoolEbmType isHumanized,
   IntEbmType * countCutsInOut,
   FloatEbmType * cutsLowerBoundInclusiveOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE void EBM_NATIVE_CALLING_CONVENTION CutUniform(
   IntEbmType countSamples,
   const FloatEbmType * featureValues,
   IntEbmType * countCutsInOut,
   FloatEbmType * cutsLowerBoundInclusiveOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CutWinsorized(
   IntEbmType countSamples,
   const FloatEbmType * featureValues,
   IntEbmType * countCutsInOut,
   FloatEbmType * cutsLowerBoundInclusiveOut
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION SuggestGraphBounds(
   IntEbmType countCuts,
   FloatEbmType lowestCut,
   FloatEbmType highestCut,
   FloatEbmType minValue,
   FloatEbmType maxValue,
   FloatEbmType * lowGraphBoundOut,
   FloatEbmType * highGraphBoundOut
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION Discretize(
   IntEbmType countSamples,
   const FloatEbmType * featureValues,
   IntEbmType countCuts,
   const FloatEbmType * cutsLowerBoundInclusive,
   IntEbmType * discretizedOut
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE IntEbmType EBM_NATIVE_CALLING_CONVENTION SizeDataSetHeader(
   IntEbmType countFeatures,
   IntEbmType countWeights,
   IntEbmType countTargets
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillDataSetHeader(
   IntEbmType countFeatures,
   IntEbmType countWeights,
   IntEbmType countTargets,
   IntEbmType countBytesAllocated,
   void * fillMem
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE IntEbmType EBM_NATIVE_CALLING_CONVENTION SizeFeature(
   BoolEbmType categorical,
   IntEbmType countBins,
   IntEbmType countSamples,
   const IntEbmType * binnedData
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillFeature(
   BoolEbmType categorical,
   IntEbmType countBins,
   IntEbmType countSamples,
   const IntEbmType * binnedData,
   IntEbmType countBytesAllocated,
   void * fillMem
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE IntEbmType EBM_NATIVE_CALLING_CONVENTION SizeWeight(
   IntEbmType countSamples,
   const FloatEbmType * weights
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillWeight(
   IntEbmType countSamples,
   const FloatEbmType * weights,
   IntEbmType countBytesAllocated,
   void * fillMem
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE IntEbmType EBM_NATIVE_CALLING_CONVENTION SizeClassificationTarget(
   IntEbmType countTargetClasses,
   IntEbmType countSamples,
   const IntEbmType * targets
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillClassificationTarget(
   IntEbmType countTargetClasses,
   IntEbmType countSamples,
   const IntEbmType * targets,
   IntEbmType countBytesAllocated,
   void * fillMem
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE IntEbmType EBM_NATIVE_CALLING_CONVENTION SizeRegressionTarget(
   IntEbmType countSamples,
   const FloatEbmType * targets
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillRegressionTarget(
   IntEbmType countSamples,
   const FloatEbmType * targets,
   IntEbmType countBytesAllocated,
   void * fillMem
);


EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION Softmax(
   IntEbmType countTargetClasses,
   IntEbmType countSamples,
   const FloatEbmType * logits,
   FloatEbmType * probabilitiesOut
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE void EBM_NATIVE_CALLING_CONVENTION SampleWithoutReplacement(
   SeedEbmType randomSeed,
   IntEbmType countTrainingSamples,
   IntEbmType countValidationSamples,
   IntEbmType * sampleCountsOut
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION StratifiedSamplingWithoutReplacement(
   SeedEbmType randomSeed,
   IntEbmType countTargetClasses,
   IntEbmType countTrainingSamples,
   IntEbmType countValidationSamples,
   IntEbmType* targets,
   IntEbmType* sampleCountsOut
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CreateClassificationBooster(
   SeedEbmType randomSeed,
   IntEbmType countTargetClasses,
   IntEbmType countFeatures,
   const BoolEbmType * featuresCategorical,
   const IntEbmType * featuresBinCount,
   IntEbmType countFeatureGroups,
   const IntEbmType * featureGroupsDimensionCount,
   const IntEbmType * featureGroupsFeatureIndexes,
   IntEbmType countTrainingSamples,
   const IntEbmType * trainingBinnedData,
   const IntEbmType * trainingTargets,
   const FloatEbmType * trainingWeights,
   const FloatEbmType * trainingPredictorScores,
   IntEbmType countValidationSamples,
   const IntEbmType * validationBinnedData,
   const IntEbmType * validationTargets,
   const FloatEbmType * validationWeights,
   const FloatEbmType * validationPredictorScores,
   IntEbmType countInnerBags,
   const FloatEbmType * optionalTempParams,
   BoosterHandle * boosterHandleOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CreateRegressionBooster(
   SeedEbmType randomSeed,
   IntEbmType countFeatures,
   const BoolEbmType * featuresCategorical,
   const IntEbmType * featuresBinCount,
   IntEbmType countFeatureGroups,
   const IntEbmType * featureGroupsDimensionCount, 
   const IntEbmType * featureGroupsFeatureIndexes, 
   IntEbmType countTrainingSamples,
   const IntEbmType * trainingBinnedData, 
   const FloatEbmType * trainingTargets,
   const FloatEbmType * trainingWeights,
   const FloatEbmType * trainingPredictorScores,
   IntEbmType countValidationSamples,
   const IntEbmType * validationBinnedData, 
   const FloatEbmType * validationTargets,
   const FloatEbmType * validationWeights,
   const FloatEbmType * validationPredictorScores,
   IntEbmType countInnerBags,
   const FloatEbmType * optionalTempParams,
   BoosterHandle * boosterHandleOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CreateBoosterView(
   BoosterHandle boosterHandle,
   BoosterHandle * boosterHandleViewOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GenerateModelUpdate(
   BoosterHandle boosterHandle,
   IntEbmType indexFeatureGroup,
   GenerateUpdateOptionsType options, 
   FloatEbmType learningRate, 
   IntEbmType countSamplesRequiredForChildSplitMin, 
   const IntEbmType * leavesMax, 
   FloatEbmType * gainOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GetModelUpdateSplits(
   BoosterHandle boosterHandle,
   IntEbmType indexDimension,
   IntEbmType * countSplitsInOut,
   IntEbmType * splitIndexesOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GetModelUpdateExpanded(
   BoosterHandle boosterHandle,
   FloatEbmType * modelFeatureGroupUpdateTensorOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION SetModelUpdateExpanded(
   BoosterHandle boosterHandle,
   IntEbmType indexFeatureGroup,
   FloatEbmType * modelFeatureGroupUpdateTensor
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION ApplyModelUpdate(
   BoosterHandle boosterHandle,
   FloatEbmType * validationMetricOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GetBestModelFeatureGroup(
   BoosterHandle boosterHandle, 
   IntEbmType indexFeatureGroup,
   FloatEbmType * modelFeatureGroupTensorOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GetCurrentModelFeatureGroup(
   BoosterHandle boosterHandle,
   IntEbmType indexFeatureGroup,
   FloatEbmType * modelFeatureGroupTensorOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE void EBM_NATIVE_CALLING_CONVENTION FreeBooster(
   BoosterHandle boosterHandle
);


EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CreateClassificationInteractionDetector(
   IntEbmType countTargetClasses,
   IntEbmType countFeatures,
   const BoolEbmType * featuresCategorical,
   const IntEbmType * featuresBinCount,
   IntEbmType countSamples,
   const IntEbmType * binnedData,
   const IntEbmType * targets,
   const FloatEbmType * weights,
   const FloatEbmType * predictorScores,
   const FloatEbmType * optionalTempParams,
   InteractionHandle * interactionHandleOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CreateRegressionInteractionDetector(
   IntEbmType countFeatures, 
   const BoolEbmType * featuresCategorical,
   const IntEbmType * featuresBinCount,
   IntEbmType countSamples,
   const IntEbmType * binnedData, 
   const FloatEbmType * targets,
   const FloatEbmType * weights,
   const FloatEbmType * predictorScores,
   const FloatEbmType * optionalTempParams,
   InteractionHandle * interactionHandleOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CalculateInteractionScore(
   InteractionHandle interactionHandle, 
   IntEbmType countDimensions,
   const IntEbmType * featureIndexes,
   IntEbmType countSamplesRequiredForChildSplitMin,
   FloatEbmType * interactionScoreOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE void EBM_NATIVE_CALLING_CONVENTION FreeInteractionDetector(
   InteractionHandle interactionHandle
);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif  // EBM_NATIVE_H
