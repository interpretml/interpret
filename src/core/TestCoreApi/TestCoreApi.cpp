// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderTestCoreApi.h"

// we roll our own test framework here since it's nice having no dependencies, and we just need a few simple tests for the C API.
// If we ended up needing something more substantial, I'd consider using doctest ( https://github.com/onqtam/doctest ) because:
//   1) It's a single include file, which is the simplest we could ask for.  Googletest is more heavyweight
//   2) It's MIT licensed, so we could include the header in our project and still keep our license 100% MIT compatible without having two licenses, unlike Catch, or Catch2
//   3) It's fast to compile.
//   4) doctest is very close to having a JUnit output feature.  JUnit isn't really required, our python testing uses JUnit, so it would be nice to have the same format -> https://github.com/onqtam/doctest/blob/master/doc/markdown/roadmap.md   https://github.com/onqtam/doctest/issues/75
//   5) If JUnit is desired in the meantime, there is a converter that will output JUnit -> https://github.com/ujiro99/doctest-junit-report
//
// In case we want to use doctest in the future, use the format of the following: TEST_CASE, CHECK & FAIL_CHECK (continues testing) / REQUIRE & FAIL (stops the current test, but we could just terminate), INFO (print to log file)
// Don't implement this since it would be harder to do: SUBCASE

#include <string>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include "ebmcore.h"

class TestCaseHidden;
typedef void (* TestFunctionHidden)(TestCaseHidden& testCaseHidden);

class TestCaseHidden {
public:
   TestCaseHidden(TestFunctionHidden pTestFunction, std::string description) {
      m_pTestFunction = pTestFunction;
      m_description = description;
      m_bPassed = true;
   }

   TestFunctionHidden m_pTestFunction;
   std::string m_description;
   bool m_bPassed;
};

std::vector<TestCaseHidden> g_allTestsHidden;

inline int RegisterTestHidden(const TestCaseHidden& testCaseHidden) {
   g_allTestsHidden.push_back(testCaseHidden);
   return 0;
}

#define CONCATENATE_STRINGS(t1, t2) t1##t2
#define CONCATENATE_TOKENS(t1, t2) CONCATENATE_STRINGS(t1, t2)
#define TEST_CASE(description) \
   static void CONCATENATE_TOKENS(TEST_FUNCTION_HIDDEN_, __LINE__)(TestCaseHidden& testCaseHidden); \
   static int CONCATENATE_TOKENS(UNUSED_INTEGER_HIDDEN_, __LINE__) = RegisterTestHidden(TestCaseHidden(&CONCATENATE_TOKENS(TEST_FUNCTION_HIDDEN_, __LINE__), description)); \
   static void CONCATENATE_TOKENS(TEST_FUNCTION_HIDDEN_, __LINE__)(TestCaseHidden& testCaseHidden)

inline bool IsApproxEqual(const double value, const double expected, const double percentage) {
   return std::abs(expected - value) < std::abs(expected * percentage);
}

#define CHECK(expression) \
   do { \
      const bool bFailedHidden = !(expression); \
      if(bFailedHidden) { \
         std::cout << " FAILED on \"" #expression "\""; \
         testCaseHidden.m_bPassed = false; \
      } \
   } while((void)0, 0)

#define CHECK_APPROX(value, expected) \
   do { \
      const double valueHidden = (value); \
      const bool bApproxEqualHidden = IsApproxEqual(valueHidden, (expected), 0.01); \
      if(!bApproxEqualHidden) { \
         std::cout << " FAILED on \"" #value "(" << valueHidden << ") approx " #expected "\""; \
         testCaseHidden.m_bPassed = false; \
      } \
   } while((void)0, 0)



constexpr size_t GetVectorLength(size_t cTargetStates) {
#ifdef TREAT_BINARY_AS_MULTICLASS
   return cTargetStates <= 1 ? static_cast<size_t>(1) : static_cast<size_t>(cTargetStates);
#else // TREAT_BINARY_AS_MULTICLASS
   return cTargetStates <= 2 ? static_cast<size_t>(1) : static_cast<size_t>(cTargetStates);
#endif // TREAT_BINARY_AS_MULTICLASS
}

TEST_CASE("AttributeCombination with zero attributes, Training, regression") {
   constexpr IntegerDataType randomSeed = 42;
   constexpr size_t cVectorLength = 1;
   constexpr size_t cAttributes = 1;
   constexpr size_t cAttributeCombinations = 1;
   constexpr size_t cAttributeCombinationsIndexes = 0;
   constexpr size_t cTrainingCases = 1;
   constexpr size_t cValidationCases = 1;
   constexpr IntegerDataType countInnerBags = 0;
   constexpr IntegerDataType countIterations = 2;
   constexpr FractionalDataType learningRate = 0.01;
   constexpr IntegerDataType countTreeSplitsMax = 2;
   constexpr IntegerDataType countCasesRequiredForSplitParentMin = 2;

   EbmAttribute attributes[std::max(std::size_t { 1 }, cAttributes)];
   EbmAttributeCombination combinations[std::max(std::size_t { 1 }, cAttributeCombinations)];
   IntegerDataType combinationIndexes[std::max(std::size_t { 1 }, cAttributeCombinationsIndexes)];
   FractionalDataType trainingTargets[std::max(std::size_t { 1 }, cTrainingCases)];
   IntegerDataType trainingData[std::max(std::size_t { 1 }, cTrainingCases * cAttributes)];
   FractionalDataType trainingPredictionScores[std::max(std::size_t { 1 }, cTrainingCases * cVectorLength)];
   FractionalDataType validationTargets[std::max(std::size_t { 1 }, cValidationCases)];
   IntegerDataType validationData[std::max(std::size_t { 1 }, cValidationCases * cAttributes)];
   FractionalDataType validationPredictionScores[std::max(std::size_t { 1 }, cValidationCases * cVectorLength)];

   attributes[0].attributeType = AttributeTypeOrdinal;
   attributes[0].countStates = 2;
   attributes[0].hasMissing = 0;

   combinations[0].countAttributesInCombination = 0;

   trainingTargets[0] = 10.5;
   trainingData[0] = 0;
   trainingPredictionScores[0] = 0;

   validationTargets[0] = 10.4;
   validationData[0] = 0;
   validationPredictionScores[0] = 0;

   PEbmTraining pEbmTraining = InitializeTrainingRegression(randomSeed, cAttributes, attributes, cAttributeCombinations, combinations, combinationIndexes, cTrainingCases, trainingTargets, trainingData, trainingPredictionScores, cValidationCases, validationTargets, validationData, validationPredictionScores, countInnerBags);

   FractionalDataType validationMetricReturn;
   IntegerDataType result;
   int count = countIterations;
   while(count--) {
      result = TrainingStep(pEbmTraining, 0, learningRate, countTreeSplitsMax, countCasesRequiredForSplitParentMin, nullptr, nullptr, &validationMetricReturn);
      CHECK(0 == result);
      if(0 != result) {
         return;
      }
      if(1 == count) {
         CHECK_APPROX(validationMetricReturn, 10.295000000000000);
      } else if(0 == count) {
         CHECK_APPROX(validationMetricReturn, 10.191050000000001);
      }
   }
   double * pModel = GetCurrentModel(pEbmTraining, 0);
   double modelValue = pModel[0];
   CHECK_APPROX(modelValue, 0.20895000000000000);
   FreeTraining(pEbmTraining);
}

TEST_CASE("AttributeCombination with zero attributes, Training, Binary") {
   constexpr IntegerDataType randomSeed = 42;
   constexpr IntegerDataType countTargetStates = 2;
   constexpr size_t cVectorLength = GetVectorLength(countTargetStates);
   constexpr size_t cAttributes = 1;
   constexpr size_t cAttributeCombinations = 1;
   constexpr size_t cAttributeCombinationsIndexes = 0;
   constexpr size_t cTrainingCases = 1;
   constexpr size_t cValidationCases = 1;
   constexpr IntegerDataType countInnerBags = 0;
   constexpr IntegerDataType countIterations = 2;
   constexpr FractionalDataType learningRate = 0.01;
   constexpr IntegerDataType countTreeSplitsMax = 2;
   constexpr IntegerDataType countCasesRequiredForSplitParentMin = 2;

   EbmAttribute attributes[std::max(std::size_t { 1 }, cAttributes)];
   EbmAttributeCombination combinations[std::max(std::size_t { 1 }, cAttributeCombinations)];
   IntegerDataType combinationIndexes[std::max(std::size_t { 1 }, cAttributeCombinationsIndexes)];
   IntegerDataType trainingTargets[std::max(std::size_t { 1 }, cTrainingCases)];
   IntegerDataType trainingData[std::max(std::size_t { 1 }, cTrainingCases * cAttributes)];
   FractionalDataType trainingPredictionScores[std::max(std::size_t { 1 }, cTrainingCases * cVectorLength)];
   IntegerDataType validationTargets[std::max(std::size_t { 1 }, cValidationCases)];
   IntegerDataType validationData[std::max(std::size_t { 1 }, cValidationCases * cAttributes)];
   FractionalDataType validationPredictionScores[std::max(std::size_t { 1 }, cValidationCases * cVectorLength)];

   attributes[0].attributeType = AttributeTypeOrdinal;
   attributes[0].countStates = 2;
   attributes[0].hasMissing = 0;

   combinations[0].countAttributesInCombination = 0;

   trainingTargets[0] = 0;
   trainingData[0] = 0;
   trainingPredictionScores[0] = 0;

   validationTargets[0] = 0;
   validationData[0] = 0;
   validationPredictionScores[0] = 0;

   PEbmTraining pEbmTraining = InitializeTrainingClassification(randomSeed, cAttributes, attributes, cAttributeCombinations, combinations, combinationIndexes, countTargetStates, cTrainingCases, trainingTargets, trainingData, trainingPredictionScores, cValidationCases, validationTargets, validationData, validationPredictionScores, countInnerBags);

   FractionalDataType validationMetricReturn;
   IntegerDataType result;
   int count = countIterations;
   while(count--) {
      result = TrainingStep(pEbmTraining, 0, learningRate, countTreeSplitsMax, countCasesRequiredForSplitParentMin, nullptr, nullptr, &validationMetricReturn);
      CHECK(0 == result);
      if(0 != result) {
         return;
      }
      if(1 == count) {
         CHECK_APPROX(validationMetricReturn, 0.68319717972663419);
      } else if(0 == count) {
         CHECK_APPROX(validationMetricReturn, 0.67344419889200957);
      }
   }
   double * pModel = GetCurrentModel(pEbmTraining, 0);
   double modelValue = pModel[0];
   CHECK_APPROX(modelValue, -0.039801986733067563);
   FreeTraining(pEbmTraining);
}

TEST_CASE("AttributeCombination with zero attributes, Training, multiclass") {
   constexpr IntegerDataType randomSeed = 42;
   constexpr IntegerDataType countTargetStates = 3;
   constexpr size_t cVectorLength = GetVectorLength(countTargetStates);
   constexpr size_t cAttributes = 1;
   constexpr size_t cAttributeCombinations = 1;
   constexpr size_t cAttributeCombinationsIndexes = 0;
   constexpr size_t cTrainingCases = 1;
   constexpr size_t cValidationCases = 1;
   constexpr IntegerDataType countInnerBags = 0;
   constexpr IntegerDataType countIterations = 2;
   constexpr FractionalDataType learningRate = 0.01;
   constexpr IntegerDataType countTreeSplitsMax = 2;
   constexpr IntegerDataType countCasesRequiredForSplitParentMin = 2;

   EbmAttribute attributes[std::max(std::size_t { 1 }, cAttributes)];
   EbmAttributeCombination combinations[std::max(std::size_t { 1 }, cAttributeCombinations)];
   IntegerDataType combinationIndexes[std::max(std::size_t { 1 }, cAttributeCombinationsIndexes)];
   IntegerDataType trainingTargets[std::max(std::size_t { 1 }, cTrainingCases)];
   IntegerDataType trainingData[std::max(std::size_t { 1 }, cTrainingCases * cAttributes)];
   FractionalDataType trainingPredictionScores[std::max(std::size_t { 1 }, cTrainingCases * cVectorLength)];
   IntegerDataType validationTargets[std::max(std::size_t { 1 }, cValidationCases)];
   IntegerDataType validationData[std::max(std::size_t { 1 }, cValidationCases * cAttributes)];
   FractionalDataType validationPredictionScores[std::max(std::size_t { 1 }, cValidationCases * cVectorLength)];

   attributes[0].attributeType = AttributeTypeOrdinal;
   attributes[0].countStates = 2;
   attributes[0].hasMissing = 0;

   combinations[0].countAttributesInCombination = 0;

   trainingTargets[0] = 0;
   trainingData[0] = 0;
   trainingPredictionScores[0] = 0;
   trainingPredictionScores[1] = 0;
   trainingPredictionScores[2] = 0;

   validationTargets[0] = 0;
   validationData[0] = 0;
   validationPredictionScores[0] = 0;
   validationPredictionScores[1] = 0;
   validationPredictionScores[2] = 0;

   PEbmTraining pEbmTraining = InitializeTrainingClassification(randomSeed, cAttributes, attributes, cAttributeCombinations, combinations, combinationIndexes, countTargetStates, cTrainingCases, trainingTargets, trainingData, trainingPredictionScores, cValidationCases, validationTargets, validationData, validationPredictionScores, countInnerBags);

   FractionalDataType validationMetricReturn;
   IntegerDataType result;
   int count = countIterations;
   while(count--) {
      result = TrainingStep(pEbmTraining, 0, learningRate, countTreeSplitsMax, countCasesRequiredForSplitParentMin, nullptr, nullptr, &validationMetricReturn);
      CHECK(0 == result);
      if(0 != result) {
         return;
      }
      if(1 == count) {
         CHECK_APPROX(validationMetricReturn, 1.0688384008227103);
      } else if(0 == count) {
         CHECK_APPROX(validationMetricReturn, 1.0401627411809615);
      }
   }
   double * pModel = GetCurrentModel(pEbmTraining, 0);
   double modelValue1 = pModel[0];
   double modelValue2 = pModel[1];
   double modelValue3 = pModel[2];
   CHECK_APPROX(modelValue1, 0.059119949636662006);
   CHECK_APPROX(modelValue2, -0.029887518980531450);
   CHECK_APPROX(modelValue3, -0.029887518980531450);
   FreeTraining(pEbmTraining);
}

TEST_CASE("AttributeCombination with zero attributes, Interaction, regression") {
   constexpr size_t cVectorLength = 1;
   constexpr size_t cAttributes = 1;
   constexpr size_t cAttributeCombinationsIndexes = 0;
   constexpr size_t cCases = 1;

   EbmAttribute attributes[std::max(std::size_t { 1 }, cAttributes)];
   IntegerDataType combinationIndexes[std::max(std::size_t { 1 }, cAttributeCombinationsIndexes)];
   FractionalDataType targets[std::max(std::size_t { 1 }, cCases)];
   IntegerDataType data[std::max(std::size_t { 1 }, cCases * cAttributes)];
   FractionalDataType predictionScores[std::max(std::size_t { 1 }, cCases * cVectorLength)];

   attributes[0].attributeType = AttributeTypeOrdinal;
   attributes[0].countStates = 2;
   attributes[0].hasMissing = 0;

   targets[0] = 10.5;
   data[0] = 0;
   predictionScores[0] = 0;

   PEbmInteraction pEbmInteraction = InitializeInteractionRegression(cAttributes, attributes, cCases, targets, data, predictionScores);

   FractionalDataType metricReturn;
   IntegerDataType result;
   result = GetInteractionScore(pEbmInteraction, cAttributeCombinationsIndexes, combinationIndexes, &metricReturn);
   CHECK(0 == result);
   if(0 != result) {
      return;
   }
   CHECK(0 == metricReturn);
   FreeInteraction(pEbmInteraction);
}

TEST_CASE("AttributeCombination with zero attributes, Interaction, Binary") {
   constexpr IntegerDataType countTargetStates = 2;
   constexpr size_t cVectorLength = GetVectorLength(countTargetStates);
   constexpr size_t cAttributes = 1;
   constexpr size_t cAttributeCombinationsIndexes = 0;
   constexpr size_t cCases = 1;

   EbmAttribute attributes[std::max(std::size_t { 1 }, cAttributes)];
   IntegerDataType combinationIndexes[std::max(std::size_t { 1 }, cAttributeCombinationsIndexes)];
   IntegerDataType targets[std::max(std::size_t { 1 }, cCases)];
   IntegerDataType data[std::max(std::size_t { 1 }, cCases * cAttributes)];
   FractionalDataType predictionScores[std::max(std::size_t { 1 }, cCases * cVectorLength)];

   attributes[0].attributeType = AttributeTypeOrdinal;
   attributes[0].countStates = 2;
   attributes[0].hasMissing = 0;

   targets[0] = 0;
   data[0] = 0;
   predictionScores[0] = 0;

   PEbmInteraction pEbmInteraction = InitializeInteractionClassification(cAttributes, attributes, countTargetStates, cCases, targets, data, predictionScores);

   FractionalDataType metricReturn;
   IntegerDataType result;
   result = GetInteractionScore(pEbmInteraction, cAttributeCombinationsIndexes, combinationIndexes, &metricReturn);
   CHECK(0 == result);
   if(0 != result) {
      return;
   }
   CHECK(0 == metricReturn);
   FreeInteraction(pEbmInteraction);
}

TEST_CASE("AttributeCombination with zero attributes, Interaction, multiclass") {
   constexpr IntegerDataType countTargetStates = 3;
   constexpr size_t cVectorLength = GetVectorLength(countTargetStates);
   constexpr size_t cAttributes = 1;
   constexpr size_t cAttributeCombinationsIndexes = 0;
   constexpr size_t cCases = 1;

   EbmAttribute attributes[std::max(std::size_t { 1 }, cAttributes)];
   IntegerDataType combinationIndexes[std::max(std::size_t { 1 }, cAttributeCombinationsIndexes)];
   IntegerDataType targets[std::max(std::size_t { 1 }, cCases)];
   IntegerDataType data[std::max(std::size_t { 1 }, cCases * cAttributes)];
   FractionalDataType predictionScores[std::max(std::size_t { 1 }, cCases * cVectorLength)];

   attributes[0].attributeType = AttributeTypeOrdinal;
   attributes[0].countStates = 2;
   attributes[0].hasMissing = 0;

   targets[0] = 0;
   data[0] = 0;
   predictionScores[0] = 0;
   predictionScores[1] = 0;
   predictionScores[2] = 0;

   PEbmInteraction pEbmInteraction = InitializeInteractionClassification(cAttributes, attributes, countTargetStates, cCases, targets, data, predictionScores);

   FractionalDataType metricReturn;
   IntegerDataType result;
   result = GetInteractionScore(pEbmInteraction, cAttributeCombinationsIndexes, combinationIndexes, &metricReturn);
   CHECK(0 == result);
   if(0 != result) {
      return;
   }
   CHECK(0 == metricReturn);
   FreeInteraction(pEbmInteraction);
}


void EBMCORE_CALLING_CONVENTION LogMessage(signed char traceLevel, const char * message) {
   printf("%d - %s\n", traceLevel, message);
}

int main() {
   SetLogMessageFunction(&LogMessage);
   SetTraceLevel(TraceLevelOff);

   bool bPassed = true;
   for(TestCaseHidden& testCaseHidden : g_allTestsHidden) {
      std::cout << "Starting test: " << testCaseHidden.m_description;
      testCaseHidden.m_pTestFunction(testCaseHidden);
      if(testCaseHidden.m_bPassed) {
         std::cout << " PASSED" << std::endl;
      } else {
         bPassed = false;
         // any failures (there can be multiple) have already been written out
         std::cout << std::endl;
      }
   }

   std::cout << "C API test " << (bPassed ? "PASSED" : "FAILED") << std::endl;
   return bPassed ? 0 : 1;
}
