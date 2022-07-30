// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef EBM_NATIVE_H
#define EBM_NATIVE_H

#include <inttypes.h> // fixed sized integer types and printf strings.  Includes stdint.h

#ifdef __cplusplus
extern "C" {
#define EBM_EXTERN_C  extern "C"
#define STATIC_CAST(type, val)  (static_cast<type>(val))
#else // __cplusplus
#define EBM_EXTERN_C  extern
#define STATIC_CAST(type, val)  ((type)(val))
#endif // __cplusplus

//#define EXPAND_BINARY_LOGITS

#if defined(__clang__) || defined(__GNUC__) || defined(__SUNPRO_CC)

#define EBM_API_INCLUDE extern

#ifdef EBM_NATIVE_R
// R has it's own way of exporting functions. We export R specific functionality via the R_init_interpret function
#define EBM_API_BODY EBM_EXTERN_C
#else // EBM_NATIVE_R
#define EBM_API_BODY EBM_EXTERN_C __attribute__ ((visibility ("default")))
#endif // EBM_NATIVE_R

#define EBM_CALLING_CONVENTION

#elif defined(_MSC_VER) // compiler type

#ifdef EBM_NATIVE_R 
// R has it's own way of exporting functions. We export R specific functionality via the R_init_interpret function
#define EBM_API_INCLUDE extern
#define EBM_API_BODY EBM_EXTERN_C
#else // EBM_NATIVE_R

#ifdef EBM_NATIVE_EXPORTS
// we use a .def file in Visual Studio because we can remove the C name mangling, unlike __declspec(dllexport)
#define EBM_API_INCLUDE extern
#define EBM_API_BODY EBM_EXTERN_C
#else // EBM_NATIVE_EXPORTS
// __declspec(dllimport) is optional, but it allows the compiler to make the code more efficient when imported
#define EBM_API_INCLUDE extern __declspec(dllimport)
#endif // EBM_NATIVE_EXPORTS

#endif // EBM_NATIVE_R

#ifdef _WIN64
// _WIN32 is defined during 64-bit compilations, so use _WIN64
// In x64 Windows, __fastcall is the only calling convention, and it does not need to be specified
#define EBM_CALLING_CONVENTION
#else // _WIN64
// In x86 Windows, __stdcall (WINAPI) is used for Win32 OS functions. It is defined by Windows and most languages 
// support it since they all need to call win32 internally. Not all languages support CDECL since it is C specified.
#define EBM_CALLING_CONVENTION __stdcall
#endif // _WIN64

#else // compiler type
#error compiler not recognized
#endif // compiler type

typedef struct _BoosterHandle {
   uint32_t handleVerification; // should be 25077 if ok
} * BoosterHandle;

typedef struct _InteractionHandle {
   uint32_t handleVerification; // should be 27917 if ok
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

// Smaller integers can safely roundtrip to double values and back, but this breaks down at exactly 2^53.
// 2^53 will convert exactly from an integer to a double and back, but the double 2^53 + 1 will round
// down to the integer 2^53 in IEEE-754 where banker's rounding is used.  So, if we had an integer of
// 9007199254740992 we would be safe to convert it to to a double and back, but if we see a double of
// 9007199254740992.0 we don't know if it was originally the integer 9007199254740992 or 9007199254740993, so
// 9007199254740992 is unsafe if checking as a double.
// https://stackoverflow.com/questions/1848700/biggest-integer-that-can-be-stored-in-a-double
// R has a lower maximum index of 4503599627370496 (R_XLEN_T_MAX) probably to store a bit somewhere.
#define SAFE_FLOAT64_AS_INT64_MAX   9007199254740991

// The maximum signed int64 value is 9223372036854775807, but doubles above 9223372036854775295 round in IEEE-754
// to a number above that, so if we're converting from a float64 to an int64, the maximum safe number is 
// 9223372036854775295. When we convert 9223372036854775295 to a float64 though, we loose precision and if we output 
// it with 17 decimal digits, which is the universal round trip format for float64 in IEEE-754, then we get 
// 9.2233720368547748e+18.  When you accurately round that biggest representable float64 to the closest integer, 
// you get 9223372036854774784, which having 19 digits is legal as an exact IEEE-754 representation since it's 
// within the 17-20 digits that is required by IEEE-754 to give a universally reproducible float value
#define FLOAT64_TO_INT64_MAX   9223372036854774784

// TODO: look through our code for places where SAFE_FLOAT64_AS_INT64_MAX or FLOAT64_TO_INT64_MAX would be useful

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
typedef int64_t BoostFlagsType;
// technically printf hexidecimals are unsigned, so convert it first to unsigned before calling printf
typedef uint64_t UBoostFlagsType;
#define UBoostFlagsTypePrintf PRIx64
typedef int64_t InteractionFlagsType;
// technically printf hexidecimals are unsigned, so convert it first to unsigned before calling printf
typedef uint64_t UInteractionFlagsType;
#define UInteractionFlagsTypePrintf PRIx64

#define BOOL_CAST(val)                 (STATIC_CAST(BoolEbmType, (val)))
#define ERROR_CAST(val)                (STATIC_CAST(ErrorEbmType, (val)))
#define TRACE_CAST(val)                (STATIC_CAST(TraceEbmType, (val)))
#define BOOST_FLAGS_CAST(val)          (STATIC_CAST(BoostFlagsType, (val)))
#define INTERACTION_FLAGS_CAST(val)    (STATIC_CAST(InteractionFlagsType, (val)))

#define EBM_FALSE          (BOOL_CAST(0))
#define EBM_TRUE           (BOOL_CAST(1))

#define Error_None                                 (ERROR_CAST(0))
#define Error_OutOfMemory                          (ERROR_CAST(-1))
// errors occuring entirely within the C/C++ code
#define Error_UnexpectedInternal                   (ERROR_CAST(-2))
// input parameters received that are clearly due to bugs in the higher level caller
#define Error_IllegalParamValue                    (ERROR_CAST(-3))
// input parameters received from the end user that are illegal.  These should have been filtered by our caller
#define Error_UserParamValue                       (ERROR_CAST(-4))
#define Error_ThreadStartFailed                    (ERROR_CAST(-5))

#define Error_LossConstructorException             (ERROR_CAST(-10))
#define Error_LossParamUnknown                     (ERROR_CAST(-11))
#define Error_LossParamValueMalformed              (ERROR_CAST(-12))
#define Error_LossParamValueOutOfRange             (ERROR_CAST(-13))
#define Error_LossParamMismatchWithConfig          (ERROR_CAST(-14))
#define Error_LossUnknown                          (ERROR_CAST(-15))
#define Error_LossIllegalRegistrationName          (ERROR_CAST(-16))
#define Error_LossIllegalParamName                 (ERROR_CAST(-17))
#define Error_LossDuplicateParamName               (ERROR_CAST(-18))

#define BoostFlags_Default                         (BOOST_FLAGS_CAST(0x0000000000000000))
#define BoostFlags_DisableNewtonGain               (BOOST_FLAGS_CAST(0x0000000000000001))
#define BoostFlags_DisableNewtonUpdate             (BOOST_FLAGS_CAST(0x0000000000000002))
#define BoostFlags_GradientSums                    (BOOST_FLAGS_CAST(0x0000000000000004))
#define BoostFlags_RandomSplits                    (BOOST_FLAGS_CAST(0x0000000000000008))

#define InteractionFlags_Default                   (INTERACTION_FLAGS_CAST(0x0000000000000000))
#define InteractionFlags_Pure                      (INTERACTION_FLAGS_CAST(0x0000000000000001))

// No messages will be logged. This is the default.
#define Trace_Off      (TRACE_CAST(0))
// Invalid inputs to the C interface, internal errors, or assert failures before exiting. Cannot continue afterwards.
#define Trace_Error    (TRACE_CAST(1))
// Out of memory or other conditions that are unexpected or odd. Can either return with an error, or continue.
#define Trace_Warning  (TRACE_CAST(2))
// Important informational messages such as entering important functions. Should be reasonable for production systems.
#define Trace_Info     (TRACE_CAST(3))
// All messages logged. Useful for tracing execution in detail. Might log too much detail for production systems.
#define Trace_Verbose  (TRACE_CAST(4))

// All our logging messages are pure ASCII (127 values), and therefore also conform to UTF-8
typedef void (EBM_CALLING_CONVENTION * LOG_CALLBACK)(TraceEbmType traceLevel, const char * message);

// SetLogCallback does not need to be called if the level is left at Trace_Off
EBM_API_INCLUDE void EBM_CALLING_CONVENTION SetLogCallback(LOG_CALLBACK logCallback);
EBM_API_INCLUDE void EBM_CALLING_CONVENTION SetTraceLevel(TraceEbmType traceLevel);
EBM_API_INCLUDE const char * EBM_CALLING_CONVENTION GetTraceLevelString(TraceEbmType traceLevel);

EBM_API_INCLUDE void EBM_CALLING_CONVENTION CleanFloats(IntEbmType count, double * valsInOut);

EBM_API_INCLUDE SeedEbmType EBM_CALLING_CONVENTION GenerateSeed(
   SeedEbmType seed,
   SeedEbmType randomMix
);
EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION GenerateGaussianRandom(
   BoolEbmType isDeterministic,
   SeedEbmType seed,
   double stddev,
   IntEbmType count,
   double * randomOut
);

EBM_API_INCLUDE IntEbmType EBM_CALLING_CONVENTION GetHistogramCutCount(
   IntEbmType countSamples,
   const double * featureVals
);

EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION CutQuantile(
   IntEbmType countSamples,
   const double * featureVals,
   IntEbmType minSamplesBin,
   BoolEbmType isRounded,
   IntEbmType * countCutsInOut,
   double * cutsLowerBoundInclusiveOut
);
EBM_API_INCLUDE IntEbmType EBM_CALLING_CONVENTION CutUniform(
   IntEbmType countSamples,
   const double * featureVals,
   IntEbmType countDesiredCuts,
   double * cutsLowerBoundInclusiveOut
);
EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION CutWinsorized(
   IntEbmType countSamples,
   const double * featureVals,
   IntEbmType * countCutsInOut,
   double * cutsLowerBoundInclusiveOut
);

EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION SuggestGraphBounds(
   IntEbmType countCuts,
   double lowestCut,
   double highestCut,
   double minFeatureVal,
   double maxFeatureVal,
   double * lowGraphBoundOut,
   double * highGraphBoundOut
);

EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION BinFeature(
   IntEbmType countSamples,
   const double * featureVals,
   IntEbmType countCuts,
   const double * cutsLowerBoundInclusive,
   IntEbmType * binIndexesOut
);

EBM_API_INCLUDE IntEbmType EBM_CALLING_CONVENTION SizeDataSetHeader(
   IntEbmType countFeatures,
   IntEbmType countWeights,
   IntEbmType countTargets
);
EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION FillDataSetHeader(
   IntEbmType countFeatures,
   IntEbmType countWeights,
   IntEbmType countTargets,
   IntEbmType countBytesAllocated,
   void * fillMem
);

EBM_API_INCLUDE IntEbmType EBM_CALLING_CONVENTION SizeFeature(
   IntEbmType countBins,
   BoolEbmType isMissing,
   BoolEbmType isUnknown,
   BoolEbmType isNominal,
   IntEbmType countSamples,
   const IntEbmType * binIndexes
);
EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION FillFeature(
   IntEbmType countBins,
   BoolEbmType isMissing,
   BoolEbmType isUnknown,
   BoolEbmType isNominal,
   IntEbmType countSamples,
   const IntEbmType * binIndexes,
   IntEbmType countBytesAllocated,
   void * fillMem
);

EBM_API_INCLUDE IntEbmType EBM_CALLING_CONVENTION SizeWeight(
   IntEbmType countSamples,
   const double * weights
);
EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION FillWeight(
   IntEbmType countSamples,
   const double * weights,
   IntEbmType countBytesAllocated,
   void * fillMem
);

EBM_API_INCLUDE IntEbmType EBM_CALLING_CONVENTION SizeClassificationTarget(
   IntEbmType countClasses,
   IntEbmType countSamples,
   const IntEbmType * targets
);
EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION FillClassificationTarget(
   IntEbmType countClasses,
   IntEbmType countSamples,
   const IntEbmType * targets,
   IntEbmType countBytesAllocated,
   void * fillMem
);

EBM_API_INCLUDE IntEbmType EBM_CALLING_CONVENTION SizeRegressionTarget(
   IntEbmType countSamples,
   const double * targets
);
EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION FillRegressionTarget(
   IntEbmType countSamples,
   const double * targets,
   IntEbmType countBytesAllocated,
   void * fillMem
);

EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION ExtractDataSetHeader(
   const void * dataSet,
   IntEbmType * countSamplesOut,
   IntEbmType * countFeaturesOut,
   IntEbmType * countWeightsOut,
   IntEbmType * countTargetsOut
);

EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION ExtractBinCounts(
   const void * dataSet,
   IntEbmType countFeaturesVerify,
   IntEbmType * binCountsOut
);

EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION ExtractTargetClasses(
   const void * dataSet,
   IntEbmType countTargetsVerify,
   IntEbmType * classCountsOut
);


EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION SampleWithoutReplacement(
   BoolEbmType isDeterministic,
   SeedEbmType seed,
   IntEbmType countTrainingSamples,
   IntEbmType countValidationSamples,
   BagEbmType * bagOut
);

EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION SampleWithoutReplacementStratified(
   BoolEbmType isDeterministic,
   SeedEbmType seed,
   IntEbmType countClasses,
   IntEbmType countTrainingSamples,
   IntEbmType countValidationSamples,
   IntEbmType * targets,
   BagEbmType * bagOut
);

EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION CreateBooster(
   BoolEbmType isDeterministic,
   SeedEbmType seed,
   const void * dataSet,
   const BagEbmType * bag,
   const double * initScores, // only samples with non-zeros in the bag are included
   IntEbmType countTerms,
   const IntEbmType * dimensionCounts,
   const IntEbmType * featureIndexes,
   IntEbmType countInnerBags,
   const double * experimentalParams,
   BoosterHandle * boosterHandleOut
);
// TODO: we might need a function to set the booster's internal seed so that a booster view 
// can either use the same seed as the original booster, or diverge on some new random sequence path
EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION CreateBoosterView(
   BoosterHandle boosterHandle,
   BoosterHandle * boosterHandleViewOut
);
EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION GenerateTermUpdate(
   BoosterHandle boosterHandle,
   IntEbmType indexTerm,
   BoostFlagsType flags, 
   double learningRate, 
   IntEbmType minSamplesLeaf, 
   const IntEbmType * leavesMax, 
   double * avgGainOut
);
// GetTermUpdateSplits must be called before calls to GetTermUpdate/SetTermUpdate
EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION GetTermUpdateSplits(
   BoosterHandle boosterHandle,
   IntEbmType indexDimension,
   IntEbmType * countSplitsInOut,
   IntEbmType * splitIndexesOut
);
EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION GetTermUpdate(
   BoosterHandle boosterHandle,
   double * updateScoresTensorOut
);
EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION SetTermUpdate(
   BoosterHandle boosterHandle,
   IntEbmType indexTerm,
   double * updateScoresTensor
);
EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION ApplyTermUpdate(
   BoosterHandle boosterHandle,
   double * validationMetricOut
);
EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION GetBestTermScores(
   BoosterHandle boosterHandle, 
   IntEbmType indexTerm,
   double * termScoresTensorOut
);
EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION GetCurrentTermScores(
   BoosterHandle boosterHandle,
   IntEbmType indexTerm,
   double * termScoresTensorOut
);
EBM_API_INCLUDE void EBM_CALLING_CONVENTION FreeBooster(
   BoosterHandle boosterHandle
);

EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION CreateInteractionDetector(
   const void * dataSet,
   const BagEbmType * bag,
   const double * initScores, // only samples with non-zeros in the bag are included
   const double * experimentalParams,
   InteractionHandle * interactionHandleOut
);
EBM_API_INCLUDE ErrorEbmType EBM_CALLING_CONVENTION CalcInteractionStrength(
   InteractionHandle interactionHandle, 
   IntEbmType countDimensions,
   const IntEbmType * featureIndexes,
   InteractionFlagsType flags,
   IntEbmType minSamplesLeaf,
   double * avgInteractionStrengthOut
);
EBM_API_INCLUDE void EBM_CALLING_CONVENTION FreeInteractionDetector(
   InteractionHandle interactionHandle
);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif  // EBM_NATIVE_H
