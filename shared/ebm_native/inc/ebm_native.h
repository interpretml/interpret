// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef EBM_NATIVE_H
#define EBM_NATIVE_H

#include <inttypes.h> // fixed sized integer types and printf strings.  Includes stdint.h

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
#define EBM_INTERACTION_OPTIONS_CAST(EBM_VAL)      (STATIC_CAST(InteractionOptionsType, (EBM_VAL)))

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

typedef int64_t IntEbmType;
#define IntEbmTypePrintf PRId64
typedef uint64_t UIntEbmType;
#define UIntEbmTypePrintf PRIu64
typedef int32_t SeedEbmType;
#define SeedEbmTypePrintf PRId32
typedef int8_t BagEbmType;
#define BagEbmTypePrintf PRId8
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
typedef int64_t InteractionOptionsType;
// technically printf hexidecimals are unsigned, so convert it first to unsigned before calling printf
typedef uint64_t UInteractionOptionsType;
#define UInteractionOptionsTypePrintf PRIx64

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

#define InteractionOptions_Default                 (EBM_INTERACTION_OPTIONS_CAST(0x0000000000000000))
#define InteractionOptions_Pure                    (EBM_INTERACTION_OPTIONS_CAST(0x0000000000000001))

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

EBM_NATIVE_IMPORT_EXPORT_INCLUDE void EBM_NATIVE_CALLING_CONVENTION CleanFloats(IntEbmType count, double * valsInOut);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE SeedEbmType EBM_NATIVE_CALLING_CONVENTION GenerateDeterministicSeed(
   SeedEbmType randomSeed,
   SeedEbmType stageRandomizationMix
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GenerateNondeterministicSeed(
   SeedEbmType * randomSeedOut
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GenerateGaussianRandom(
   BoolEbmType isDeterministic,
   SeedEbmType randomSeed,
   double stddev,
   IntEbmType count,
   double * randomOut
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE IntEbmType EBM_NATIVE_CALLING_CONVENTION GetHistogramCutCount(
   IntEbmType countSamples,
   const double * featureValues,
   IntEbmType strategy
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CutQuantile(
   IntEbmType countSamples,
   const double * featureValues,
   IntEbmType countSamplesPerBinMin,
   BoolEbmType isRounded,
   IntEbmType * countCutsInOut,
   double * cutsLowerBoundInclusiveOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE IntEbmType EBM_NATIVE_CALLING_CONVENTION CutUniform(
   IntEbmType countSamples,
   const double * featureValues,
   IntEbmType countDesiredCuts,
   double * cutsLowerBoundInclusiveOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CutWinsorized(
   IntEbmType countSamples,
   const double * featureValues,
   IntEbmType * countCutsInOut,
   double * cutsLowerBoundInclusiveOut
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION SuggestGraphBounds(
   IntEbmType countCuts,
   double lowestCut,
   double highestCut,
   double minValue,
   double maxValue,
   double * lowGraphBoundOut,
   double * highGraphBoundOut
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION Discretize(
   IntEbmType countSamples,
   const double * featureValues,
   IntEbmType countCuts,
   const double * cutsLowerBoundInclusive,
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
   IntEbmType countBins,
   BoolEbmType missing,
   BoolEbmType unknown,
   BoolEbmType nominal,
   IntEbmType countSamples,
   const IntEbmType * binnedData
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillFeature(
   IntEbmType countBins,
   BoolEbmType missing,
   BoolEbmType unknown,
   BoolEbmType nominal,
   IntEbmType countSamples,
   const IntEbmType * binnedData,
   IntEbmType countBytesAllocated,
   void * fillMem
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE IntEbmType EBM_NATIVE_CALLING_CONVENTION SizeWeight(
   IntEbmType countSamples,
   const double * weights
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillWeight(
   IntEbmType countSamples,
   const double * weights,
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
   const double * targets
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION FillRegressionTarget(
   IntEbmType countSamples,
   const double * targets,
   IntEbmType countBytesAllocated,
   void * fillMem
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION ExtractDataSetHeader(
   const void * dataSet,
   IntEbmType * countSamplesOut,
   IntEbmType * countFeaturesOut,
   IntEbmType * countWeightsOut,
   IntEbmType * countTargetsOut
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION ExtractBinCounts(
   const void * dataSet,
   IntEbmType countFeaturesVerify,
   IntEbmType * binCountsOut
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION ExtractTargetClasses(
   const void * dataSet,
   IntEbmType countTargetsVerify,
   IntEbmType * classCountsOut
);


EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION SampleWithoutReplacement(
   BoolEbmType isDeterministic,
   SeedEbmType randomSeed,
   IntEbmType countTrainingSamples,
   IntEbmType countValidationSamples,
   BagEbmType * sampleCountsOut
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION StratifiedSamplingWithoutReplacement(
   SeedEbmType randomSeed,
   IntEbmType countTargetClasses,
   IntEbmType countTrainingSamples,
   IntEbmType countValidationSamples,
   IntEbmType * targets,
   BagEbmType * sampleCountsOut
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CreateBooster(
   SeedEbmType randomSeed,
   const void * dataSet,
   const BagEbmType * bag,
   const double * initScores, // only samples with non-zeros in the bag are included
   IntEbmType countFeatureGroups,
   const IntEbmType * dimensionCounts,
   const IntEbmType * featureIndexes,
   IntEbmType countInnerBags,
   const double * optionalTempParams,
   BoosterHandle * boosterHandleOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CreateBoosterView(
   BoosterHandle boosterHandle,
   BoosterHandle * boosterHandleViewOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GenerateTermUpdate(
   BoosterHandle boosterHandle,
   IntEbmType indexFeatureGroup,
   GenerateUpdateOptionsType options, 
   double learningRate, 
   IntEbmType countSamplesRequiredForChildSplitMin, 
   const IntEbmType * leavesMax, 
   double * avgGainOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GetTermUpdateSplits(
   BoosterHandle boosterHandle,
   IntEbmType indexDimension,
   IntEbmType * countSplitsInOut,
   IntEbmType * splitIndexesOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GetTermUpdateExpanded(
   BoosterHandle boosterHandle,
   double * updateScoresTensorOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION SetTermUpdateExpanded(
   BoosterHandle boosterHandle,
   IntEbmType indexFeatureGroup,
   double * updateScoresTensor
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION ApplyTermUpdate(
   BoosterHandle boosterHandle,
   double * validationMetricOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GetBestTermScores(
   BoosterHandle boosterHandle, 
   IntEbmType indexFeatureGroup,
   double * termScoresTensorOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GetCurrentTermScores(
   BoosterHandle boosterHandle,
   IntEbmType indexFeatureGroup,
   double * termScoresTensorOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE void EBM_NATIVE_CALLING_CONVENTION FreeBooster(
   BoosterHandle boosterHandle
);

EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CreateInteractionDetector(
   const void * dataSet,
   const BagEbmType * bag,
   const double * initScores, // only samples with non-zeros in the bag are included
   const double * optionalTempParams,
   InteractionHandle * interactionHandleOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CalcInteractionStrength(
   InteractionHandle interactionHandle, 
   IntEbmType countDimensions,
   const IntEbmType * featureIndexes,
   InteractionOptionsType options,
   IntEbmType countSamplesRequiredForChildSplitMin,
   double * avgInteractionStrengthOut
);
EBM_NATIVE_IMPORT_EXPORT_INCLUDE void EBM_NATIVE_CALLING_CONVENTION FreeInteractionDetector(
   InteractionHandle interactionHandle
);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif  // EBM_NATIVE_H
