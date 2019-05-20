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
typedef int64_t IntegerDataType;

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
const signed char TraceLevelDebug = 5; // our debug library should be called if our caller is set to this level

// all our logging messages are pure ascii (127 values), and therefore also UTF-8
typedef void (EBMCORE_CALLING_CONVENTION * LOG_MESSAGE_FUNCTION)(signed char traceLevel, const char * message);

EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION SetLogMessageFunction(LOG_MESSAGE_FUNCTION logMessageFunction);
EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION SetTraceLevel(signed char traceLevel);

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
