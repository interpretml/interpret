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
#include <cmath>
#include <cstddef>

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
#ifdef EXPAND_BINARY_LOGITS
   return cTargetStates <= 1 ? size_t { 1 } : static_cast<size_t>(cTargetStates);
#else // EXPAND_BINARY_LOGITS
   return cTargetStates <= 2 ? size_t { 1 } : static_cast<size_t>(cTargetStates);
#endif // EXPAND_BINARY_LOGITS
}

constexpr IntegerDataType randomSeed = 42;

enum class AttributeType : IntegerDataType { Ordinal = AttributeTypeOrdinal, Nominal = AttributeTypeNominal };

class Attribute {
public:

   AttributeType m_attributeType;
   bool m_hasMissing;
   IntegerDataType m_countStates;

   Attribute(AttributeType attributeType, bool hasMissing, IntegerDataType countStates) {
      m_attributeType = attributeType;
      m_hasMissing = hasMissing;
      m_countStates = countStates;
   }
};

class RegressionCase {
public:
   FractionalDataType m_target;
   std::vector<IntegerDataType> m_data;
   FractionalDataType m_score;
   bool m_bNullTrainingPredictionScores;

   RegressionCase(FractionalDataType target, std::vector<IntegerDataType> data) {
      m_target = target;
      m_data = data;
      m_score = 0;
      m_bNullTrainingPredictionScores = true;
   }

   RegressionCase(FractionalDataType target, std::vector<IntegerDataType> data, FractionalDataType score) {
      m_target = target;
      m_data = data;
      m_score = score;
      m_bNullTrainingPredictionScores = false;
   }
};

class ClassificationCase {
public:
   IntegerDataType m_target;
   std::vector<IntegerDataType> m_data;
   std::vector<FractionalDataType> m_logits;
   bool m_bNullTrainingPredictionScores;

   ClassificationCase(IntegerDataType target, std::vector<IntegerDataType> data) {
      m_target = target;
      m_data = data;
      m_bNullTrainingPredictionScores = true;
   }

   ClassificationCase(IntegerDataType target, std::vector<IntegerDataType> data, std::vector<FractionalDataType> logits) {
      m_target = target;
      m_data = data;
      m_logits = logits;
      m_bNullTrainingPredictionScores = false;
   }
};

class TestApi {
public:

   enum class Stage { Attributes, AttributeCombinations, Training, Validation, Initialize, Steps };

   Stage m_stage;
   const ptrdiff_t m_iZeroClassificationLogit;

   std::vector<EbmAttribute> m_attributes;
   std::vector<EbmAttributeCombination> m_attributeCombinations;
   std::vector<IntegerDataType> m_attributeCombinationsIndexes;

   std::vector<FractionalDataType> m_trainingRegressionTargets;
   std::vector<IntegerDataType> m_trainingClassificationTargets;
   std::vector<IntegerDataType> m_trainingData;
   std::vector<FractionalDataType> m_trainingPredictionScores;
   bool m_bNullTrainingPredictionScores;

   std::vector<FractionalDataType> m_validationRegressionTargets;
   std::vector<IntegerDataType> m_validationClassificationTargets;
   std::vector<IntegerDataType> m_validationData;
   std::vector<FractionalDataType> m_validationPredictionScores;
   bool m_bNullValidationPredictionScores;

   PEbmTraining m_pEbmTraining;

   TestApi(const ptrdiff_t iZeroClassificationLogit = ptrdiff_t { -1 }) :
      m_stage(Stage::Attributes),
      m_iZeroClassificationLogit(iZeroClassificationLogit) {
   }

   void AddAttributes(const std::vector<Attribute> attributes) {
      if(Stage::Attributes != m_stage) {
         exit(1);
      }

      for(const Attribute oneAttribute : attributes) {
         EbmAttribute attribute;
         attribute.attributeType = static_cast<IntegerDataType>(oneAttribute.m_attributeType);
         attribute.hasMissing = oneAttribute.m_hasMissing ? IntegerDataType { 1 } : IntegerDataType { 0 };
         attribute.countStates = oneAttribute.m_countStates;
         m_attributes.push_back(attribute);
      }

      m_stage = Stage::AttributeCombinations;
   }

   void AddAttributeCombinations(const std::vector<std::vector<size_t>> attributeCombinations) {
      if(Stage::AttributeCombinations != m_stage) {
         exit(1);
      }

      for(const std::vector<size_t> oneAttributeCombination : attributeCombinations) {
         EbmAttributeCombination attributeCombination;
         attributeCombination.countAttributesInCombination = oneAttributeCombination.size();
         m_attributeCombinations.push_back(attributeCombination);
         for(const size_t oneIndex : oneAttributeCombination) {
            if(oneIndex <= m_attributes.size()) {
               exit(1);
            }
            m_attributeCombinationsIndexes.push_back(oneIndex);
         }
      }

      m_stage = Stage::Training;
   }

   void AddTrainingCases(const std::vector<RegressionCase> cases) {
      if(Stage::Training != m_stage) {
         exit(1);
      }
      const size_t cCases = cases.size();
      if(0 == cCases) {
         return;
      }
      const size_t cAttributes = m_attributes.size();
      const bool bNullTrainingPredictionScores = cases[0].m_bNullTrainingPredictionScores;
      m_bNullTrainingPredictionScores = bNullTrainingPredictionScores;

      for(const RegressionCase oneCase : cases) {
         if(cAttributes != oneCase.m_data.size()) {
            exit(1);
         }
         if(bNullTrainingPredictionScores != oneCase.m_bNullTrainingPredictionScores) {
            exit(1);
         }
         FractionalDataType target = oneCase.m_target;
         if(std::isnan(target)) {
            exit(1);
         }
         if(std::isinf(target)) {
            exit(1);
         }
         m_trainingRegressionTargets.push_back(target);
         if(!bNullTrainingPredictionScores) {
            FractionalDataType score = oneCase.m_score;
            if(std::isnan(score)) {
               exit(1);
            }
            if(std::isinf(score)) {
               exit(1);
            }
            m_trainingPredictionScores.push_back(score);
         }
      }
      for(size_t iAttribute = 0; iAttribute < cAttributes; ++iAttribute) {
         EbmAttribute attribute = m_attributes[iAttribute];
         for(size_t iCase = 0; iCase < cCases; ++iCase) {
            IntegerDataType data = cases[iCase].m_data[iAttribute];
            if(data < 0) {
               exit(1);
            }
            if(attribute.countStates <= data) {
               exit(1);
            }
            m_trainingData.push_back(data);
         }
      }

      m_stage = Stage::Validation;
   }

   void AddTrainingCases(const std::vector<ClassificationCase> cases) {
      if(Stage::Training != m_stage) {
         exit(1);
      }
      const size_t cCases = cases.size();
      if(0 == cCases) {
         return;
      }
      const size_t cAttributes = m_attributes.size();
      const size_t cClasses = cases[0].m_logits.size();
      if(static_cast<ptrdiff_t>(cClasses) <= m_iZeroClassificationLogit) {
         exit(1);
      }
      const bool bNullTrainingPredictionScores = cases[0].m_bNullTrainingPredictionScores;
      m_bNullTrainingPredictionScores = bNullTrainingPredictionScores;

      for(const ClassificationCase oneCase : cases) {
         if(cAttributes != oneCase.m_data.size()) {
            exit(1);
         }
         if(cClasses != oneCase.m_logits.size()) {
            exit(1);
         }
         if(bNullTrainingPredictionScores != oneCase.m_bNullTrainingPredictionScores) {
            exit(1);
         }
         const IntegerDataType target = oneCase.m_target;
         if(target < 0) {
            exit(1);
         }
         if(cClasses <= static_cast<size_t>(target)) {
            exit(1);
         }
         m_trainingClassificationTargets.push_back(target);
         if(!bNullTrainingPredictionScores) {
            ptrdiff_t iLogit = 0;
            for(FractionalDataType oneLogit : oneCase.m_logits) {
               if(std::isnan(oneLogit)) {
                  exit(1);
               }
               if(std::isinf(oneLogit)) {
                  exit(1);
               }
               if(2 == cClasses) {
                  // binary classification
#ifdef EXPAND_BINARY_LOGITS
                  if(m_iZeroClassificationLogit < 0) {
                     m_trainingPredictionScores.push_back(oneLogit - oneCase.m_logits[0]);
                  } else {
                     m_trainingPredictionScores.push_back(oneLogit - oneCase.m_logits[m_iZeroClassificationLogit]);
                  }
#else // EXPAND_BINARY_LOGITS
                  if(m_iZeroClassificationLogit < 0) {
                     if(0 != iLogit) {
                        m_trainingPredictionScores.push_back(oneLogit - oneCase.m_logits[0]);
                     }
                  } else {
                     if(m_iZeroClassificationLogit != iLogit) {
                        m_trainingPredictionScores.push_back(oneLogit - oneCase.m_logits[m_iZeroClassificationLogit]);
                     }
                  }
#endif // EXPAND_BINARY_LOGITS
               } else {
                  // multiclass
#ifdef REDUCE_MULTICLASS_LOGITS
                  if(m_iZeroClassificationLogit < 0) {
                     if(0 != iLogit) {
                        m_trainingPredictionScores.push_back(oneLogit - oneCase.m_logits[0]);
                     }
                  } else {
                     if(m_iZeroClassificationLogit != iLogit) {
                        m_trainingPredictionScores.push_back(oneLogit - oneCase.m_logits[m_iZeroClassificationLogit]);
                     }
                  }
#else // REDUCE_MULTICLASS_LOGITS
                  if(m_iZeroClassificationLogit < 0) {
                     m_trainingPredictionScores.push_back(oneLogit);
                  } else {
                     m_trainingPredictionScores.push_back(oneLogit - oneCase.m_logits[m_iZeroClassificationLogit]);
                  }
#endif // REDUCE_MULTICLASS_LOGITS
               }
               ++iLogit;
            }
         }
      }
      for(size_t iAttribute = 0; iAttribute < cAttributes; ++iAttribute) {
         EbmAttribute attribute = m_attributes[iAttribute];
         for(size_t iCase = 0; iCase < cCases; ++iCase) {
            IntegerDataType data = cases[iCase].m_data[iAttribute];
            if(data < 0) {
               exit(1);
            }
            if(attribute.countStates <= data) {
               exit(1);
            }
            m_trainingData.push_back(data);
         }
      }

      m_stage = Stage::Validation;
   }













   void Initialize(const IntegerDataType countInnerBags) {
      if(Stage::Initialize != m_stage) {
         exit(1);
      }
      if(countInnerBags < IntegerDataType { 0 }) {
         exit(1);
      }
      m_pEbmTraining = InitializeTrainingRegression(randomSeed, m_attributes.size(), &m_attributes[0], m_attributeCombinations.size(), &m_attributeCombinations[0], &m_attributeCombinationsIndexes[0], m_trainingRegressionTargets.size(), &m_trainingRegressionTargets[0], &m_trainingData[0], &m_trainingPredictionScores[0], m_validationRegressionTargets.size(), &m_validationRegressionTargets[0], &m_validationData[0], &m_validationPredictionScores[0], countInnerBags);

      m_stage = Stage::Training;
   }

   FractionalDataType Train(PEbmTraining ebmTraining, IntegerDataType indexAttributeCombination, FractionalDataType learningRate, IntegerDataType countTreeSplitsMax, IntegerDataType countCasesRequiredForSplitParentMin, std::vector<FractionalDataType> trainingWeights, std::vector<FractionalDataType> validationWeights) {
      if(Stage::Training != m_stage) {
         exit(1);
      }
      if(nullptr == ebmTraining) {
         exit(1);
      }
      if(indexAttributeCombination < IntegerDataType { 0 }) {
         exit(1);
      }
      if(m_attributeCombinations.size() <= static_cast<size_t>(indexAttributeCombination)) {
         exit(1);
      }
      if(learningRate < FractionalDataType { 0 }) {
         exit(1);
      }
      if(std::isnan(learningRate)) {
         exit(1);
      }
      if(std::isinf(learningRate)) {
         exit(1);
      }
      if(countTreeSplitsMax < FractionalDataType { 1 }) {
         exit(1);
      }
      if(countCasesRequiredForSplitParentMin < FractionalDataType { 2 }) {
         exit(1);
      }

      FractionalDataType validationMetricReturn;
      IntegerDataType ret = TrainingStep(m_pEbmTraining, indexAttributeCombination, learningRate, countTreeSplitsMax, countCasesRequiredForSplitParentMin, 0 == trainingWeights.size() ? nullptr : &trainingWeights[0], 0 == validationWeights.size() ? nullptr : &validationWeights[0], &validationMetricReturn);
      if(0 != ret) {
         exit(1);
      }
      return validationMetricReturn;
   }




};

TEST_CASE("AttributeCombination with zero attributes, Training, regression") {
   //TestApi mytest = TestApi();
   //mytest.AddAttributes({ Attribute(AttributeType::Ordinal, false, 2) });
   //mytest.AddAttributeCombinations({ { 0, 1 }, { 2 } });
   //mytest.AddTrainingCases({ RegressionCase(10.5, { 0 }, 0) });
   //mytest.AddTrainingCases({ ClassificationCase(10, { 0 }, { 0 , 0 }) });
   //mytest.AddTrainingCases({ RegressionCase(10.5, { 0 }) });
   //mytest.AddTrainingCases({ ClassificationCase(10, { 0 }) });

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

TEST_CASE("AttributeCombination with one attribute with one state, Training, regression") {
   constexpr size_t cVectorLength = 1;
   constexpr size_t cAttributes = 1;
   constexpr size_t cAttributeCombinations = 1;
   constexpr size_t cAttributeCombinationsIndexes = 1;
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
   attributes[0].countStates = 1;
   attributes[0].hasMissing = 0;

   combinations[0].countAttributesInCombination = 1;

   combinationIndexes[0] = 0;

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

TEST_CASE("AttributeCombination with one attribute with one state, Training, Binary") {
   constexpr IntegerDataType countTargetStates = 2;
   constexpr size_t cVectorLength = GetVectorLength(countTargetStates);
   constexpr size_t cAttributes = 1;
   constexpr size_t cAttributeCombinations = 1;
   constexpr size_t cAttributeCombinationsIndexes = 1;
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
   attributes[0].countStates = 1;
   attributes[0].hasMissing = 0;

   combinations[0].countAttributesInCombination = 1;

   combinationIndexes[0] = 0;

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

TEST_CASE("AttributeCombination with one attribute with one state, Training, multiclass") {
   constexpr IntegerDataType countTargetStates = 3;
   constexpr size_t cVectorLength = GetVectorLength(countTargetStates);
   constexpr size_t cAttributes = 1;
   constexpr size_t cAttributeCombinations = 1;
   constexpr size_t cAttributeCombinationsIndexes = 1;
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
   attributes[0].countStates = 1;
   attributes[0].hasMissing = 0;

   combinations[0].countAttributesInCombination = 1;

   combinationIndexes[0] = 0;

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

TEST_CASE("AttributeCombination with one attribute with one state, Interaction, regression") {
   constexpr size_t cVectorLength = 1;
   constexpr size_t cAttributes = 1;
   constexpr size_t cAttributeCombinationsIndexes = 1;
   constexpr size_t cCases = 1;

   EbmAttribute attributes[std::max(std::size_t { 1 }, cAttributes)];
   IntegerDataType combinationIndexes[std::max(std::size_t { 1 }, cAttributeCombinationsIndexes)];
   FractionalDataType targets[std::max(std::size_t { 1 }, cCases)];
   IntegerDataType data[std::max(std::size_t { 1 }, cCases * cAttributes)];
   FractionalDataType predictionScores[std::max(std::size_t { 1 }, cCases * cVectorLength)];

   attributes[0].attributeType = AttributeTypeOrdinal;
   attributes[0].countStates = 1;
   attributes[0].hasMissing = 0;

   combinationIndexes[0] = 0;

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

TEST_CASE("AttributeCombination with one attribute with one state, Interaction, Binary") {
   constexpr IntegerDataType countTargetStates = 2;
   constexpr size_t cVectorLength = GetVectorLength(countTargetStates);
   constexpr size_t cAttributes = 1;
   constexpr size_t cAttributeCombinationsIndexes = 1;
   constexpr size_t cCases = 1;

   EbmAttribute attributes[std::max(std::size_t { 1 }, cAttributes)];
   IntegerDataType combinationIndexes[std::max(std::size_t { 1 }, cAttributeCombinationsIndexes)];
   IntegerDataType targets[std::max(std::size_t { 1 }, cCases)];
   IntegerDataType data[std::max(std::size_t { 1 }, cCases * cAttributes)];
   FractionalDataType predictionScores[std::max(std::size_t { 1 }, cCases * cVectorLength)];

   attributes[0].attributeType = AttributeTypeOrdinal;
   attributes[0].countStates = 1;
   attributes[0].hasMissing = 0;

   combinationIndexes[0] = 0;

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

TEST_CASE("AttributeCombination with one attribute with one state, Interaction, multiclass") {
   constexpr IntegerDataType countTargetStates = 3;
   constexpr size_t cVectorLength = GetVectorLength(countTargetStates);
   constexpr size_t cAttributes = 1;
   constexpr size_t cAttributeCombinationsIndexes = 1;
   constexpr size_t cCases = 1;

   EbmAttribute attributes[std::max(std::size_t { 1 }, cAttributes)];
   IntegerDataType combinationIndexes[std::max(std::size_t { 1 }, cAttributeCombinationsIndexes)];
   IntegerDataType targets[std::max(std::size_t { 1 }, cCases)];
   IntegerDataType data[std::max(std::size_t { 1 }, cCases * cAttributes)];
   FractionalDataType predictionScores[std::max(std::size_t { 1 }, cCases * cVectorLength)];

   attributes[0].attributeType = AttributeTypeOrdinal;
   attributes[0].countStates = 1;
   attributes[0].hasMissing = 0;

   combinationIndexes[0] = 0;

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
