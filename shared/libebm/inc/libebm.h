// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef LIBEBM_H
#define LIBEBM_H

#include <inttypes.h> // Fixed sized integer types and printf strings. The C99 standard says it includes stdint.h

#ifdef __cplusplus
extern "C" {
#define EBM_EXTERN_C           extern "C"
#define STATIC_CAST(type, val) (static_cast<type>(val))
#else // __cplusplus
#define EBM_EXTERN_C           extern
#define STATIC_CAST(type, val) ((type)(val))
#endif // __cplusplus

#if defined(__clang__) || defined(__GNUC__) || defined(__SUNPRO_CC)

#define EBM_API_INCLUDE extern

#ifdef LIBEBM_R
// R has it's own way of exporting functions. We export R specific functionality via the R_init_interpret function
#define EBM_API_BODY EBM_EXTERN_C
#else // LIBEBM_R
#define EBM_API_BODY EBM_EXTERN_C __attribute__((visibility("default")))
#endif // LIBEBM_R

#define EBM_CALLING_CONVENTION

#elif defined(_MSC_VER) // compiler type

#ifdef LIBEBM_R
// R has it's own way of exporting functions. We export R specific functionality via the R_init_interpret function
#define EBM_API_INCLUDE extern
#define EBM_API_BODY    EBM_EXTERN_C
#else // LIBEBM_R

#ifdef LIBEBM_EXPORTS
// we use a .def file in Visual Studio because we can remove the C name mangling, unlike __declspec(dllexport)
#define EBM_API_INCLUDE extern
#define EBM_API_BODY    EBM_EXTERN_C
#else // LIBEBM_EXPORTS
// __declspec(dllimport) is optional, but it allows the compiler to make the code more efficient when imported
#define EBM_API_INCLUDE extern __declspec(dllimport)
#endif // LIBEBM_EXPORTS

#endif // LIBEBM_R

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
#error unsupported compiler type
#endif // compiler type

// some compilers do not define these PRI macros (MinGW for R)
#ifndef PRId8
#define PRId8 "d"
#endif // PRId8
#ifndef PRId32
#define PRId32 "d"
#endif // PRId32
#ifndef PRIx32
#define PRIx32 "x"
#endif // PRIx32
#ifndef PRId64
#define PRId64 "lld"
#endif // PRId64
#ifndef PRIu64
#define PRIu64 "llu"
#endif // PRIu64

typedef int64_t IntEbm;
#define IntEbmPrintf PRId64
typedef uint64_t UIntEbm;
#define UIntEbmPrintf PRIu64
typedef int32_t SeedEbm;
typedef uint32_t USeedEbm;
#define SeedEbmPrintf PRId32
typedef int8_t BagEbm;
#define BagEbmPrintf PRId8
typedef int32_t TraceEbm;
#define TraceEbmPrintf PRId32
typedef int32_t BoolEbm;
#define BoolEbmPrintf PRId32
typedef int32_t MonotoneDirection;
#define MonotoneDirectionPrintf PRId32
typedef int32_t ErrorEbm;
#define ErrorEbmPrintf PRId32
typedef int32_t LinkFlags;
// printf hexidecimals must be unsigned, so convert first to unsigned before calling printf
typedef uint32_t ULinkFlags;
#define ULinkFlagsPrintf PRIx32
typedef int32_t CreateBoosterFlags;
// printf hexidecimals must be unsigned, so convert first to unsigned before calling printf
typedef uint32_t UCreateBoosterFlags;
#define UCreateBoosterFlagsPrintf PRIx32
typedef int32_t TermBoostFlags;
// printf hexidecimals must be unsigned, so convert first to unsigned before calling printf
typedef uint32_t UTermBoostFlags;
#define UTermBoostFlagsPrintf PRIx32
typedef int32_t CreateInteractionFlags;
// printf hexidecimals must be unsigned, so convert first to unsigned before calling printf
typedef uint32_t UCreateInteractionFlags;
#define UCreateInteractionFlagsPrintf PRIx32
typedef int32_t CalcInteractionFlags;
// printf hexidecimals must be unsigned, so convert first to unsigned before calling printf
typedef uint32_t UCalcInteractionFlags;
#define UCalcInteractionFlagsPrintf PRIx32
typedef int32_t AccelerationFlags;
// printf hexidecimals must be unsigned, so convert first to unsigned before calling printf
typedef uint32_t UAccelerationFlags;
#define UAccelerationFlagsPrintf PRIx32
typedef int32_t LinkEbm;
#define LinkEbmPrintf PRId32
typedef int64_t TaskEbm;
#define TaskEbmPrintf PRId64

typedef struct _BoosterHandle {
   uint32_t handleVerification; // should be 10995 if ok. Do not use size_t since that requires an additional header.
}* BoosterHandle;

typedef struct _InteractionHandle {
   uint32_t handleVerification; // should be 21773 if ok. Do not use size_t since that requires an additional header.
}* InteractionHandle;

#define BOOL_CAST(val)                     (STATIC_CAST(BoolEbm, (val)))
#define MONOTONE_CAST(val)                 (STATIC_CAST(MonotoneDirection, (val)))
#define ERROR_CAST(val)                    (STATIC_CAST(ErrorEbm, (val)))
#define LINK_FLAGS_CAST(val)               (STATIC_CAST(LinkFlags, (val)))
#define CREATE_BOOSTER_FLAGS_CAST(val)     (STATIC_CAST(CreateBoosterFlags, (val)))
#define CREATE_INTERACTION_FLAGS_CAST(val) (STATIC_CAST(CreateInteractionFlags, (val)))
#define TERM_BOOST_FLAGS_CAST(val)         (STATIC_CAST(TermBoostFlags, (val)))
#define CALC_INTERACTION_FLAGS_CAST(val)   (STATIC_CAST(CalcInteractionFlags, (val)))
#define ACCELERATION_CAST(val)             (STATIC_CAST(AccelerationFlags, (val)))
#define TRACE_CAST(val)                    (STATIC_CAST(TraceEbm, (val)))
#define LINK_CAST(val)                     (STATIC_CAST(LinkEbm, (val)))
#define TASK_CAST(val)                     (STATIC_CAST(TaskEbm, (val)))

// TODO: look through our code for places where SAFE_FLOAT64_AS_INT64_MAX or FLOAT64_TO_INT64_MAX would be useful

// Smaller integers can safely roundtrip to double values and back, but this breaks down at exactly 2^53.
// 2^53 will convert exactly from an integer to a double and back, but the double 2^53 + 1 will round
// down to the integer 2^53 in IEEE-754 where banker rounding is used.  So, if we had an integer of
// 9007199254740992 we would be safe to convert it to to a double and back, but if we see a double of
// 9007199254740992.0 we don't know if it was originally the integer 9007199254740992 or 9007199254740993, so
// 9007199254740992 is unsafe if checking as a double.
// https://stackoverflow.com/questions/1848700/biggest-integer-that-can-be-stored-in-a-double
// R has a lower maximum index of 4503599627370496 (R_XLEN_T_MAX) probably to store a bit somewhere.
#define SAFE_FLOAT64_AS_INT64_MAX (9007199254740992 - 1)
// float32 values have identical considerations. 2^24 is the last float32 value that has an exact integer
// representation as a float. But the float 2^24 + 1 rounds down to 2^24, so 2^24 - 1 is the last float32 that
// is safe to assume has a unique integer that represents it
#define REPRESENTABLE_INT32_AS_FLOAT32_MAX 16777216
#define SAFE_FLOAT32_AS_INT32_MAX          (REPRESENTABLE_INT32_AS_FLOAT32_MAX - 1)

// The maximum signed int64 value is 9223372036854775807, but doubles above 9223372036854775295 round in IEEE-754
// to a number above that, so if we're converting from a float64 to an int64, the maximum safe number is
// 9223372036854775295. When we convert 9223372036854775295 to a float64 though, we loose precision and if we output
// it with 17 decimal digits, which is the universal round trip format for float64 in IEEE-754, then we get
// 9.2233720368547748e+18.  When you accurately round that biggest representable float64 to the closest integer,
// you get 9223372036854774784, which having 19 digits is legal as an exact IEEE-754 representation since it is
// within the 17-20 digits that is required by IEEE-754 to give a universally reproducible float value
#define FLOAT64_TO_INT64_MAX 9223372036854774784

#define EBM_FALSE (BOOL_CAST(0))
#define EBM_TRUE  (BOOL_CAST(1))

#define MONOTONE_INCREASING (MONOTONE_CAST(1))
#define MONOTONE_NONE       (MONOTONE_CAST(0))
#define MONOTONE_DECREASING (MONOTONE_CAST(-1))

#define MINIMIZE_METRIC EBM_FALSE
#define MAXIMIZE_METRIC EBM_TRUE

#define Error_None        (ERROR_CAST(0))
#define Error_OutOfMemory (ERROR_CAST(-1))
// errors occuring entirely within the C/C++ code
#define Error_UnexpectedInternal (ERROR_CAST(-2))
// bad input values that are due to bugs in the higher level caller
#define Error_IllegalParamVal (ERROR_CAST(-3))
// bad input values that are from the end user. These should have been filtered out by our higher level caller
#define Error_UserParamVal      (ERROR_CAST(-4))
#define Error_ThreadStartFailed (ERROR_CAST(-5))

#define Error_ObjectiveConstructorException    (ERROR_CAST(-10))
#define Error_ObjectiveParamUnknown            (ERROR_CAST(-11))
#define Error_ObjectiveParamValMalformed       (ERROR_CAST(-12))
#define Error_ObjectiveParamValOutOfRange      (ERROR_CAST(-13))
#define Error_ObjectiveParamMismatchWithConfig (ERROR_CAST(-14))
#define Error_ObjectiveUnknown                 (ERROR_CAST(-15))
#define Error_ObjectiveIllegalRegistrationName (ERROR_CAST(-16))
#define Error_ObjectiveIllegalParamName        (ERROR_CAST(-17))
#define Error_ObjectiveDuplicateParamName      (ERROR_CAST(-18))
#define Error_ObjectiveNonPrivate              (ERROR_CAST(-19))
#define Error_ObjectiveParamNonPrivate         (ERROR_CAST(-20))
#define Error_ObjectiveIllegalTarget           (ERROR_CAST(-21))

#define LinkFlags_Default             (LINK_FLAGS_CAST(0x00000000))
#define LinkFlags_DifferentialPrivacy (LINK_FLAGS_CAST(0x00000001))
#define LinkFlags_BinaryAsMulticlass  (LINK_FLAGS_CAST(0x00000002))

#define CreateBoosterFlags_Default             (CREATE_BOOSTER_FLAGS_CAST(0x00000000))
#define CreateBoosterFlags_DifferentialPrivacy (CREATE_BOOSTER_FLAGS_CAST(0x00000001))
#define CreateBoosterFlags_DisableApprox       (CREATE_BOOSTER_FLAGS_CAST(0x00000002))
#define CreateBoosterFlags_BinaryAsMulticlass  (CREATE_BOOSTER_FLAGS_CAST(0x00000004))

#define TermBoostFlags_Default             (TERM_BOOST_FLAGS_CAST(0x00000000))
#define TermBoostFlags_DisableNewtonGain   (TERM_BOOST_FLAGS_CAST(0x00000001))
#define TermBoostFlags_DisableNewtonUpdate (TERM_BOOST_FLAGS_CAST(0x00000002))
#define TermBoostFlags_PurifyGain          (TERM_BOOST_FLAGS_CAST(0x00000004))
#define TermBoostFlags_PurifyUpdate        (TERM_BOOST_FLAGS_CAST(0x00000008))
#define TermBoostFlags_GradientSums        (TERM_BOOST_FLAGS_CAST(0x00000010))
#define TermBoostFlags_RandomSplits        (TERM_BOOST_FLAGS_CAST(0x00000020))

#define CreateInteractionFlags_Default             (CREATE_INTERACTION_FLAGS_CAST(0x00000000))
#define CreateInteractionFlags_DifferentialPrivacy (CREATE_INTERACTION_FLAGS_CAST(0x00000001))
#define CreateInteractionFlags_DisableApprox       (CREATE_INTERACTION_FLAGS_CAST(0x00000002))
#define CreateInteractionFlags_BinaryAsMulticlass  (CREATE_INTERACTION_FLAGS_CAST(0x00000004))

#define CalcInteractionFlags_Default       (CALC_INTERACTION_FLAGS_CAST(0x00000000))
#define CalcInteractionFlags_DisableNewton (CALC_INTERACTION_FLAGS_CAST(0x00000001))
#define CalcInteractionFlags_Purify        (CALC_INTERACTION_FLAGS_CAST(0x00000002))

#define AccelerationFlags_NONE      (ACCELERATION_CAST(0x00000000))
#define AccelerationFlags_Nvidia    (ACCELERATION_CAST(0x00000001))
#define AccelerationFlags_AVX2      (ACCELERATION_CAST(0x00000002))
#define AccelerationFlags_AVX512F   (ACCELERATION_CAST(0x00000004))
#define AccelerationFlags_IntelSIMD (AccelerationFlags_AVX2 | AccelerationFlags_AVX512F)
#define AccelerationFlags_SIMD      (AccelerationFlags_IntelSIMD)
#define AccelerationFlags_GPU       (AccelerationFlags_Nvidia)
#define AccelerationFlags_ALL       (ACCELERATION_CAST(~ACCELERATION_CAST(0)))

// No messages will be logged. This is the default.
#define Trace_Off (TRACE_CAST(0))
// Invalid inputs to the C interface, internal errors, or assert failures before exiting. Cannot continue afterwards.
#define Trace_Error (TRACE_CAST(1))
// Out of memory or other conditions that are unexpected or odd. Can either return with an error, or continue.
#define Trace_Warning (TRACE_CAST(2))
// Important informational messages such as entering important functions. Should be reasonable for production systems.
#define Trace_Info (TRACE_CAST(3))
// All messages logged. Useful for tracing execution in detail. Might log too much detail for production systems.
#define Trace_Verbose (TRACE_CAST(4))

// https://www.sagepub.com/sites/default/files/upm-binaries/21121_Chapter_15.pdf
// https://www.rdocumentation.org/packages/VGAM/versions/1.1-8/topics/Links
#define Link_ERROR (LINK_CAST(0))
// custom (uses link param potentially)
#define Link_custom_regression  (LINK_CAST(1))
#define Link_custom_ranking     (LINK_CAST(2))
#define Link_monoclassification (LINK_CAST(10)) // always 100%, so no customization
#define Link_custom_binary      (LINK_CAST(11))
#define Link_custom_ovr         (LINK_CAST(12))
#define Link_custom_multinomial (LINK_CAST(13))
// multinominal classification
#define Link_mlogit (LINK_CAST(20)) // Multinominal logistic regression
// one vs rest multiclass classification
#define Link_vlogit (LINK_CAST(30)) // Multiclass one vs rest logistic regression
// binary classification
#define Link_logit   (LINK_CAST(40)) // Logistic regression
#define Link_probit  (LINK_CAST(41)) // Probit regression
#define Link_cloglog (LINK_CAST(42)) // Complementary log-log regression
#define Link_loglog  (LINK_CAST(43)) // Log-log regression
#define Link_cauchit (LINK_CAST(44)) // Cauchit regression
// ranking and future
//                                                 reserve 50-89
// regression with link param
#define Link_power (LINK_CAST(90)) // Tweedie regression (although we use log)
// regression
#define Link_identity       (LINK_CAST(100)) // Linear regression
#define Link_log            (LINK_CAST(101)) // Poisson regression
#define Link_inverse        (LINK_CAST(102)) // Gamma regression (although we use log)
#define Link_inverse_square (LINK_CAST(103)) // Inverse Gaussian regression
#define Link_sqrt           (LINK_CAST(104)) // Square root regression

#define Task_Ranking               (TASK_CAST(-3))
#define Task_Regression            (TASK_CAST(-2))
#define Task_Unknown               (TASK_CAST(-1))
#define Task_GeneralClassification (TASK_CAST(0)) // classification with unspecified # classes
#define Task_MonoClassification    (TASK_CAST(1)) // degenerate case of predicting 1 class
#define Task_BinaryClassification  (TASK_CAST(2)) // 2 classes
#define Task_MulticlassPlus        (TASK_CAST(3)) // 3+ classes (the value is the # of classes)

// All our logging messages are pure ASCII (127 values), and therefore also conform to UTF-8
typedef void(EBM_CALLING_CONVENTION* LogCallbackFunction)(TraceEbm traceLevel, const char* message);

// SetLogCallback does not need to be called if the level is left at Trace_Off
EBM_API_INCLUDE void EBM_CALLING_CONVENTION SetLogCallback(LogCallbackFunction logCallbackFunction);
EBM_API_INCLUDE void EBM_CALLING_CONVENTION SetTraceLevel(TraceEbm traceLevel);
EBM_API_INCLUDE const char* EBM_CALLING_CONVENTION GetTraceLevelString(TraceEbm traceLevel);

EBM_API_INCLUDE void EBM_CALLING_CONVENTION CleanFloats(IntEbm count, double* valsInOut);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION SafeMean(
      IntEbm countBags, IntEbm countTensorBins, const double* vals, double* tensorOut);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION SafeStandardDeviation(
      IntEbm countBags, IntEbm countTensorBins, const double* vals, double* tensorOut);

EBM_API_INCLUDE IntEbm EBM_CALLING_CONVENTION MeasureRNG(void);
EBM_API_INCLUDE void EBM_CALLING_CONVENTION InitRNG(SeedEbm seed, void* rngOut);
EBM_API_INCLUDE void EBM_CALLING_CONVENTION CopyRNG(void* rng, void* rngOut);
EBM_API_INCLUDE void EBM_CALLING_CONVENTION BranchRNG(void* rng, void* rngOut);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION GenerateSeed(void* rng, SeedEbm* seedOut);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION GenerateGaussianRandom(
      void* rng, double stddev, IntEbm count, double* randomOut);

EBM_API_INCLUDE double EBM_CALLING_CONVENTION MeasureImpurity(IntEbm countMultiScores,
      IntEbm indexMultiScore,
      IntEbm countDimensions,
      const IntEbm* dimensionLengths,
      const double* weights,
      const double* scores);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION Purify(double tolerance,
      BoolEbm isRandomized,
      BoolEbm isMulticlassNormalization,
      IntEbm countMultiScores,
      IntEbm countDimensions,
      const IntEbm* dimensionLengths,
      const double* weights,
      double* scoresInOut,
      double* impuritiesOut,
      double* interceptOut);

EBM_API_INCLUDE IntEbm EBM_CALLING_CONVENTION GetHistogramCutCount(IntEbm countSamples, const double* featureVals);
// CutUniform does not fail with valid inputs, so we return the number of cuts generated
EBM_API_INCLUDE IntEbm EBM_CALLING_CONVENTION CutUniform(
      IntEbm countSamples, const double* featureVals, IntEbm countDesiredCuts, double* cutsLowerBoundInclusiveOut);

EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION CutQuantile(IntEbm countSamples,
      const double* featureVals,
      IntEbm minSamplesBin,
      BoolEbm isRounded,
      IntEbm* countCutsInOut,
      double* cutsLowerBoundInclusiveOut);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION CutWinsorized(
      IntEbm countSamples, const double* featureVals, IntEbm* countCutsInOut, double* cutsLowerBoundInclusiveOut);

EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION SuggestGraphBounds(IntEbm countCuts,
      double lowestCut,
      double highestCut,
      double minFeatureVal,
      double maxFeatureVal,
      double* lowGraphBoundOut,
      double* highGraphBoundOut);

EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION Discretize(IntEbm countSamples,
      const double* featureVals,
      IntEbm countCuts,
      const double* cutsLowerBoundInclusive,
      IntEbm* binIndexesOut);

EBM_API_INCLUDE IntEbm EBM_CALLING_CONVENTION MeasureDataSetHeader(
      IntEbm countFeatures, IntEbm countWeights, IntEbm countTargets);
EBM_API_INCLUDE IntEbm EBM_CALLING_CONVENTION MeasureFeature(IntEbm countBins,
      BoolEbm isMissing,
      BoolEbm isUnknown,
      BoolEbm isNominal,
      IntEbm countSamples,
      const IntEbm* binIndexes);
EBM_API_INCLUDE IntEbm EBM_CALLING_CONVENTION MeasureWeight(IntEbm countSamples, const double* weights);
EBM_API_INCLUDE IntEbm EBM_CALLING_CONVENTION MeasureClassificationTarget(
      IntEbm countClasses, IntEbm countSamples, const IntEbm* targets);
EBM_API_INCLUDE IntEbm EBM_CALLING_CONVENTION MeasureRegressionTarget(IntEbm countSamples, const double* targets);

EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION FillDataSetHeader(
      IntEbm countFeatures, IntEbm countWeights, IntEbm countTargets, IntEbm countBytesAllocated, void* fillMem);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION FillFeature(IntEbm countBins,
      BoolEbm isMissing,
      BoolEbm isUnknown,
      BoolEbm isNominal,
      IntEbm countSamples,
      const IntEbm* binIndexes,
      IntEbm countBytesAllocated,
      void* fillMem);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION FillWeight(
      IntEbm countSamples, const double* weights, IntEbm countBytesAllocated, void* fillMem);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION FillClassificationTarget(
      IntEbm countClasses, IntEbm countSamples, const IntEbm* targets, IntEbm countBytesAllocated, void* fillMem);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION FillRegressionTarget(
      IntEbm countSamples, const double* targets, IntEbm countBytesAllocated, void* fillMem);

EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION CheckDataSet(IntEbm countBytesAllocated, const void* dataSet);

EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION ExtractDataSetHeader(const void* dataSet,
      IntEbm* countSamplesOut,
      IntEbm* countFeaturesOut,
      IntEbm* countWeightsOut,
      IntEbm* countTargetsOut);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION ExtractNominals(
      const void* dataSet, IntEbm countFeaturesVerify, BoolEbm* nominalsOut);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION ExtractBinCounts(
      const void* dataSet, IntEbm countFeaturesVerify, IntEbm* binCountsOut);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION ExtractTargetClasses(
      const void* dataSet, IntEbm countTargetsVerify, IntEbm* classCountsOut);

EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION SampleWithoutReplacement(
      void* rng, IntEbm countTrainingSamples, IntEbm countValidationSamples, BagEbm* bagOut);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION SampleWithoutReplacementStratified(void* rng,
      IntEbm countClasses,
      IntEbm countTrainingSamples,
      IntEbm countValidationSamples,
      const IntEbm* targets,
      BagEbm* bagOut);

EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION DetermineTask(const char* objective, TaskEbm* taskOut);
EBM_API_INCLUDE const char* EBM_CALLING_CONVENTION GetTaskStr(TaskEbm task);
EBM_API_INCLUDE TaskEbm EBM_CALLING_CONVENTION GetTaskInt(const char* task);

EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION DetermineLinkFunction(
      LinkFlags flags, const char* objective, IntEbm countClasses, LinkEbm* linkOut, double* linkParamOut);
EBM_API_INCLUDE const char* EBM_CALLING_CONVENTION GetLinkFunctionStr(LinkEbm link);
EBM_API_INCLUDE LinkEbm EBM_CALLING_CONVENTION GetLinkFunctionInt(const char* link);

EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION CreateBooster(void* rng,
      const void* dataSet,
      const BagEbm* bag,
      // TODO: add a baseScore parameter here so that we can initialize the mains boosting without initScores
      const double* initScores, // only samples with non-zeros in the bag are included
      IntEbm countTerms,
      const IntEbm* dimensionCounts,
      const IntEbm* featureIndexes,
      IntEbm countInnerBags,
      CreateBoosterFlags flags,
      AccelerationFlags acceleration,
      const char* objective,
      const double* experimentalParams,
      BoosterHandle* boosterHandleOut);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION CreateBoosterView(
      BoosterHandle boosterHandle, BoosterHandle* boosterHandleViewOut);
EBM_API_INCLUDE void EBM_CALLING_CONVENTION FreeBooster(BoosterHandle boosterHandle);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION GenerateTermUpdate(void* rng,
      BoosterHandle boosterHandle,
      IntEbm indexTerm,
      TermBoostFlags flags,
      double learningRate,
      IntEbm minSamplesLeaf,
      double minHessian,
      const IntEbm* leavesMax,
      const MonotoneDirection* direction,
      double* avgGainOut);
// GetTermUpdateSplits must be called before calls to GetTermUpdate/SetTermUpdate
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION GetTermUpdateSplits(
      BoosterHandle boosterHandle, IntEbm indexDimension, IntEbm* countSplitsInOut, IntEbm* splitsOut);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION GetTermUpdate(
      BoosterHandle boosterHandle, double* updateScoresTensorOut);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION SetTermUpdate(
      BoosterHandle boosterHandle, IntEbm indexTerm, const double* updateScoresTensor);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION ApplyTermUpdate(
      BoosterHandle boosterHandle, double* avgValidationMetricOut);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION GetBestTermScores(
      BoosterHandle boosterHandle, IntEbm indexTerm, double* termScoresTensorOut);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION GetCurrentTermScores(
      BoosterHandle boosterHandle, IntEbm indexTerm, double* termScoresTensorOut);

EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION CreateInteractionDetector(const void* dataSet,
      const BagEbm* bag,
      // TODO: add a baseScore parameter here for symmetry with CreateBooster
      const double* initScores, // only samples with non-zeros in the bag are included
      CreateInteractionFlags flags,
      AccelerationFlags acceleration,
      const char* objective,
      const double* experimentalParams,
      InteractionHandle* interactionHandleOut);
EBM_API_INCLUDE void EBM_CALLING_CONVENTION FreeInteractionDetector(InteractionHandle interactionHandle);
EBM_API_INCLUDE ErrorEbm EBM_CALLING_CONVENTION CalcInteractionStrength(InteractionHandle interactionHandle,
      IntEbm countDimensions,
      const IntEbm* featureIndexes,
      CalcInteractionFlags flags,
      IntEbm maxCardinality,
      IntEbm minSamplesLeaf,
      double minHessian,
      double* avgInteractionStrengthOut);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // LIBEBM_H
