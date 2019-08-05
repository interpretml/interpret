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
#include <assert.h>

#include "ebmcore.h"

#define UNUSED(x) (void)(x)

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
   return std::abs(expected - value) <= std::abs(expected * percentage);
}

// this will ONLY work if used inside the root TEST_CASE function.  The testCaseHidden variable comes from TEST_CASE and should be visible inside the function where CHECK(expression) is called
#define CHECK(expression) \
   do { \
      const bool bFailedHidden = !(expression); \
      if(bFailedHidden) { \
         std::cout << " FAILED on \"" #expression "\""; \
         testCaseHidden.m_bPassed = false; \
      } \
   } while((void)0, 0)

// this will ONLY work if used inside the root TEST_CASE function.  The testCaseHidden variable comes from TEST_CASE and should be visible inside the function where CHECK_APPROX(expression) is called
#define CHECK_APPROX(value, expected) \
   do { \
      const double valueHidden = (value); \
      const bool bApproxEqualHidden = IsApproxEqual(valueHidden, static_cast<double>(expected), double { 0.01 }); \
      if(!bApproxEqualHidden) { \
         std::cout << " FAILED on \"" #value "(" << valueHidden << ") approx " #expected "\""; \
         testCaseHidden.m_bPassed = false; \
      } \
   } while((void)0, 0)

constexpr ptrdiff_t k_learningTypeRegression = ptrdiff_t { -1 };
constexpr bool IsClassification(const ptrdiff_t learningTypeOrCountClassificationStates) {
   return 0 <= learningTypeOrCountClassificationStates;
}

constexpr size_t GetVectorLength(const ptrdiff_t learningTypeOrCountClassificationStates) {
#ifdef EXPAND_BINARY_LOGITS
#ifdef REDUCE_MULTICLASS_LOGITS

   // EXPAND_BINARY_LOGITS && REDUCE_MULTICLASS_LOGITS
#error we should not be expanding binary logits while reducing multiclass logits

#else // REDUCE_MULTICLASS_LOGITS

   // EXPAND_BINARY_LOGITS && !REDUCE_MULTICLASS_LOGITS
   return learningTypeOrCountClassificationStates <= ptrdiff_t { 1 } ? size_t { 1 } : static_cast<size_t>(learningTypeOrCountClassificationStates);

#endif // REDUCE_MULTICLASS_LOGITS
#else // EXPAND_BINARY_LOGITS
#ifdef REDUCE_MULTICLASS_LOGITS

   // !EXPAND_BINARY_LOGITS && REDUCE_MULTICLASS_LOGITS
   return learningTypeOrCountClassificationStates <= ptrdiff_t { 2 } ? size_t { 1 } : static_cast<size_t>(learningTypeOrCountClassificationStates) - size_t { 1 };

#else // REDUCE_MULTICLASS_LOGITS

   // !EXPAND_BINARY_LOGITS && !REDUCE_MULTICLASS_LOGITS
   return learningTypeOrCountClassificationStates <= ptrdiff_t { 2 } ? size_t { 1 } : static_cast<size_t>(learningTypeOrCountClassificationStates);

#endif // REDUCE_MULTICLASS_LOGITS
#endif // EXPAND_BINARY_LOGITS
}

constexpr IntegerDataType randomSeed = 42;
enum class AttributeType : IntegerDataType { Ordinal = AttributeTypeOrdinal, Nominal = AttributeTypeNominal };

class Attribute final {
public:

   const AttributeType m_attributeType;
   const bool m_hasMissing;
   const IntegerDataType m_countStates;

   Attribute(const IntegerDataType countStates, const AttributeType attributeType = AttributeType::Ordinal, const bool hasMissing = false) :
      m_attributeType(attributeType),
      m_hasMissing(hasMissing),
      m_countStates(countStates) {
      if(countStates <= 0) {
         exit(1);
      }
   }
};

class RegressionCase final {
public:
   const FractionalDataType m_target;
   const std::vector<IntegerDataType> m_data;
   const FractionalDataType m_score;
   const bool m_bNullPredictionScores;

   RegressionCase(const FractionalDataType target, const std::vector<IntegerDataType> data) :
      m_target(target),
      m_data(data),
      m_score(0),
      m_bNullPredictionScores(true) {
   }

   RegressionCase(const FractionalDataType target, const std::vector<IntegerDataType> data, const FractionalDataType score) :
      m_target(target),
      m_data(data),
      m_score(score),
      m_bNullPredictionScores(false) {
   }
};

class ClassificationCase final {
public:
   const IntegerDataType m_target;
   const std::vector<IntegerDataType> m_data;
   const std::vector<FractionalDataType> m_logits;
   const bool m_bNullPredictionScores;

   ClassificationCase(const IntegerDataType target, const std::vector<IntegerDataType> data) :
      m_target(target),
      m_data(data),
      m_bNullPredictionScores(true) {
   }

   ClassificationCase(const IntegerDataType target, const std::vector<IntegerDataType> data, const std::vector<FractionalDataType> logits) :
      m_target(target),
      m_data(data),
      m_logits(logits),
      m_bNullPredictionScores(false) {
   }
};

class TestApi {
   enum class Stage {
      Beginning, AttributesAdded, AttributeCombinationsAdded, TrainingAdded, ValidationAdded, InitializedTraining, InteractionAdded, InitializedInteraction
   };

   Stage m_stage;
   const ptrdiff_t m_learningTypeOrCountClassificationStates;
   const ptrdiff_t m_iZeroClassificationLogit;

   std::vector<EbmAttribute> m_attributes;
   std::vector<EbmAttributeCombination> m_attributeCombinations;
   std::vector<IntegerDataType> m_attributeCombinationIndexes;

   std::vector<std::vector<size_t>> m_countStatesByAttributeCombination;

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

   std::vector<FractionalDataType> m_interactionRegressionTargets;
   std::vector<IntegerDataType> m_interactionClassificationTargets;
   std::vector<IntegerDataType> m_interactionData;
   std::vector<FractionalDataType> m_interactionPredictionScores;
   bool m_bNullInteractionPredictionScores;

   PEbmInteraction m_pEbmInteraction;

   const FractionalDataType * GetScores(const size_t iAttributeCombination, const FractionalDataType * const pModel, const std::vector<size_t> indexes) const {
      if(Stage::InitializedTraining != m_stage) {
         exit(1);
      }
      const size_t cVectorLength = GetVectorLength(m_learningTypeOrCountClassificationStates);

      if(m_countStatesByAttributeCombination.size() <= iAttributeCombination) {
         exit(1);
      }
      const std::vector<size_t> countStates = m_countStatesByAttributeCombination[iAttributeCombination];

      const size_t cDimensions = indexes.size();
      if(cDimensions != countStates.size()) {
         exit(1);
      }
      size_t iValue = 0;
      size_t multiple = cVectorLength;
      for(size_t iDimension = 0; iDimension < cDimensions; ++iDimension) {
         if(countStates[iDimension] <= indexes[iDimension]) {
            exit(1);
         }
         iValue += indexes[iDimension] * multiple;
         multiple *= countStates[iDimension];
      }
      return &pModel[iValue];
   }

   FractionalDataType GetScore(const size_t iAttributeCombination, const FractionalDataType * const pModel, const std::vector<size_t> indexes, const size_t iScore) const {
      const FractionalDataType * const aScores = GetScores(iAttributeCombination, pModel, indexes);
      if(!IsClassification(m_learningTypeOrCountClassificationStates)) {
         if(0 != iScore) {
            exit(1);
         }
         return aScores[0];
      }
      if(static_cast<size_t>(m_learningTypeOrCountClassificationStates) <= iScore) {
         exit(1);
      }
      if(2 == m_learningTypeOrCountClassificationStates) {
         // binary classification
#ifdef EXPAND_BINARY_LOGITS
         if(m_iZeroClassificationLogit < 0) {
            return aScores[iScore];
         } else {
            if(m_iZeroClassificationLogit == iScore) {
               return FractionalDataType { 0 };
            } else {
               return aScores[iScore] - aScores[m_iZeroClassificationLogit];
            }
         }
#else // EXPAND_BINARY_LOGITS
         if(m_iZeroClassificationLogit < 0) {
            if(0 == iScore) {
               return FractionalDataType { 0 };
            } else {
               return aScores[0];
            }
         } else {
            if(static_cast<size_t>(m_iZeroClassificationLogit) == iScore) {
               return FractionalDataType { 0 };
            } else {
               return aScores[0];
            }
         }
#endif // EXPAND_BINARY_LOGITS
      } else {
         // multiclass
#ifdef REDUCE_MULTICLASS_LOGITS
         if(m_iZeroClassificationLogit < 0) {
            if(0 == iScore) {
               return FractionalDataType { 0 };
            } else {
               return aScores[iScore - 1];
            }
         } else {
            if(m_iZeroClassificationLogit == iScore) {
               return FractionalDataType { 0 };
            } else if(m_iZeroClassificationLogit < iScore) {
               return aScores[iScore - 1];
            } else {
               return aScores[iScore];
            }
         }
#else // REDUCE_MULTICLASS_LOGITS
         if(m_iZeroClassificationLogit < 0) {
            return aScores[iScore];
         } else {
            return aScores[iScore] - aScores[m_iZeroClassificationLogit];
         }
#endif // REDUCE_MULTICLASS_LOGITS
      }
   }

public:

   TestApi(const ptrdiff_t learningTypeOrCountClassificationStates, const ptrdiff_t iZeroClassificationLogit = ptrdiff_t { -1 }) :
      m_stage(Stage::Beginning),
      m_learningTypeOrCountClassificationStates(learningTypeOrCountClassificationStates),
      m_iZeroClassificationLogit(iZeroClassificationLogit),
      m_bNullTrainingPredictionScores(true),
      m_bNullValidationPredictionScores(true),
      m_pEbmTraining(nullptr),
      m_bNullInteractionPredictionScores(true),
      m_pEbmInteraction(nullptr) {
      if(IsClassification(learningTypeOrCountClassificationStates)) {
         if(learningTypeOrCountClassificationStates <= iZeroClassificationLogit) {
            exit(1);
         }
      } else {
         if(ptrdiff_t { -1 } != iZeroClassificationLogit) {
            exit(1);
         }
      }
   }

   ~TestApi() {
      if(nullptr != m_pEbmTraining) {
         FreeTraining(m_pEbmTraining);
      }
      if(nullptr != m_pEbmInteraction) {
         FreeInteraction(m_pEbmInteraction);
      }
   }

   size_t GetAttributeCombinationsCount() const {
      return m_attributeCombinations.size();
   }

   void AddAttributes(const std::vector<Attribute> attributes) {
      if(Stage::Beginning != m_stage) {
         exit(1);
      }

      for(const Attribute oneAttribute : attributes) {
         EbmAttribute attribute;
         attribute.attributeType = static_cast<IntegerDataType>(oneAttribute.m_attributeType);
         attribute.hasMissing = oneAttribute.m_hasMissing ? IntegerDataType { 1 } : IntegerDataType { 0 };
         attribute.countStates = oneAttribute.m_countStates;
         m_attributes.push_back(attribute);
      }

      m_stage = Stage::AttributesAdded;
   }

   void AddAttributeCombinations(const std::vector<std::vector<size_t>> attributeCombinations) {
      if(Stage::AttributesAdded != m_stage) {
         exit(1);
      }

      for(const std::vector<size_t> oneAttributeCombination : attributeCombinations) {
         EbmAttributeCombination attributeCombination;
         attributeCombination.countAttributesInCombination = oneAttributeCombination.size();
         m_attributeCombinations.push_back(attributeCombination);
         std::vector<size_t> countStates;
         for(const size_t oneIndex : oneAttributeCombination) {
            if(m_attributes.size() <= oneIndex) {
               exit(1);
            }
            m_attributeCombinationIndexes.push_back(oneIndex);
            countStates.push_back(static_cast<size_t>(m_attributes[oneIndex].countStates));
         }
         m_countStatesByAttributeCombination.push_back(countStates);
      }

      m_stage = Stage::AttributeCombinationsAdded;
   }

   void AddTrainingCases(const std::vector<RegressionCase> cases) {
      if(Stage::AttributeCombinationsAdded != m_stage) {
         exit(1);
      }
      if(k_learningTypeRegression != m_learningTypeOrCountClassificationStates) {
         exit(1);
      }
      const size_t cCases = cases.size();
      if(0 != cCases) {
         const size_t cAttributes = m_attributes.size();
         const bool bNullPredictionScores = cases[0].m_bNullPredictionScores;
         m_bNullTrainingPredictionScores = bNullPredictionScores;

         for(const RegressionCase oneCase : cases) {
            if(cAttributes != oneCase.m_data.size()) {
               exit(1);
            }
            if(bNullPredictionScores != oneCase.m_bNullPredictionScores) {
               exit(1);
            }
            const FractionalDataType target = oneCase.m_target;
            if(std::isnan(target)) {
               exit(1);
            }
            if(std::isinf(target)) {
               exit(1);
            }
            m_trainingRegressionTargets.push_back(target);
            if(!bNullPredictionScores) {
               const FractionalDataType score = oneCase.m_score;
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
            const EbmAttribute attribute = m_attributes[iAttribute];
            for(size_t iCase = 0; iCase < cCases; ++iCase) {
               const IntegerDataType data = cases[iCase].m_data[iAttribute];
               if(data < 0) {
                  exit(1);
               }
               if(attribute.countStates <= data) {
                  exit(1);
               }
               m_trainingData.push_back(data);
            }
         }
      }
      m_stage = Stage::TrainingAdded;
   }

   void AddTrainingCases(const std::vector<ClassificationCase> cases) {
      if(Stage::AttributeCombinationsAdded != m_stage) {
         exit(1);
      }
      if(!IsClassification(m_learningTypeOrCountClassificationStates)) {
         exit(1);
      }
      const size_t cCases = cases.size();
      if(0 != cCases) {
         const size_t cAttributes = m_attributes.size();
         const bool bNullPredictionScores = cases[0].m_bNullPredictionScores;
         m_bNullTrainingPredictionScores = bNullPredictionScores;

         for(const ClassificationCase oneCase : cases) {
            if(cAttributes != oneCase.m_data.size()) {
               exit(1);
            }
            if(bNullPredictionScores != oneCase.m_bNullPredictionScores) {
               exit(1);
            }
            const IntegerDataType target = oneCase.m_target;
            if(target < 0) {
               exit(1);
            }
            if(static_cast<size_t>(m_learningTypeOrCountClassificationStates) <= static_cast<size_t>(target)) {
               exit(1);
            }
            m_trainingClassificationTargets.push_back(target);
            if(!bNullPredictionScores) {
               if(static_cast<size_t>(m_learningTypeOrCountClassificationStates) != oneCase.m_logits.size()) {
                  exit(1);
               }
               ptrdiff_t iLogit = 0;
               for(const FractionalDataType oneLogit : oneCase.m_logits) {
                  if(std::isnan(oneLogit)) {
                     exit(1);
                  }
                  if(std::isinf(oneLogit)) {
                     exit(1);
                  }
                  if(2 == m_learningTypeOrCountClassificationStates) {
                     // binary classification
#ifdef EXPAND_BINARY_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        m_trainingPredictionScores.push_back(oneLogit);
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
            const EbmAttribute attribute = m_attributes[iAttribute];
            for(size_t iCase = 0; iCase < cCases; ++iCase) {
               const IntegerDataType data = cases[iCase].m_data[iAttribute];
               if(data < 0) {
                  exit(1);
               }
               if(attribute.countStates <= data) {
                  exit(1);
               }
               m_trainingData.push_back(data);
            }
         }
      }
      m_stage = Stage::TrainingAdded;
   }

   void AddValidationCases(const std::vector<RegressionCase> cases) {
      if(Stage::TrainingAdded != m_stage) {
         exit(1);
      }
      if(k_learningTypeRegression != m_learningTypeOrCountClassificationStates) {
         exit(1);
      }
      const size_t cCases = cases.size();
      if(0 != cCases) {
         const size_t cAttributes = m_attributes.size();
         const bool bNullPredictionScores = cases[0].m_bNullPredictionScores;
         m_bNullValidationPredictionScores = bNullPredictionScores;

         for(const RegressionCase oneCase : cases) {
            if(cAttributes != oneCase.m_data.size()) {
               exit(1);
            }
            if(bNullPredictionScores != oneCase.m_bNullPredictionScores) {
               exit(1);
            }
            const FractionalDataType target = oneCase.m_target;
            if(std::isnan(target)) {
               exit(1);
            }
            if(std::isinf(target)) {
               exit(1);
            }
            m_validationRegressionTargets.push_back(target);
            if(!bNullPredictionScores) {
               const FractionalDataType score = oneCase.m_score;
               if(std::isnan(score)) {
                  exit(1);
               }
               if(std::isinf(score)) {
                  exit(1);
               }
               m_validationPredictionScores.push_back(score);
            }
         }
         for(size_t iAttribute = 0; iAttribute < cAttributes; ++iAttribute) {
            const EbmAttribute attribute = m_attributes[iAttribute];
            for(size_t iCase = 0; iCase < cCases; ++iCase) {
               const IntegerDataType data = cases[iCase].m_data[iAttribute];
               if(data < 0) {
                  exit(1);
               }
               if(attribute.countStates <= data) {
                  exit(1);
               }
               m_validationData.push_back(data);
            }
         }
      }
      m_stage = Stage::ValidationAdded;
   }

   void AddValidationCases(const std::vector<ClassificationCase> cases) {
      if(Stage::TrainingAdded != m_stage) {
         exit(1);
      }
      if(!IsClassification(m_learningTypeOrCountClassificationStates)) {
         exit(1);
      }
      const size_t cCases = cases.size();
      if(0 != cCases) {
         const size_t cAttributes = m_attributes.size();
         const bool bNullPredictionScores = cases[0].m_bNullPredictionScores;
         m_bNullValidationPredictionScores = bNullPredictionScores;

         for(const ClassificationCase oneCase : cases) {
            if(cAttributes != oneCase.m_data.size()) {
               exit(1);
            }
            if(bNullPredictionScores != oneCase.m_bNullPredictionScores) {
               exit(1);
            }
            const IntegerDataType target = oneCase.m_target;
            if(target < 0) {
               exit(1);
            }
            if(static_cast<size_t>(m_learningTypeOrCountClassificationStates) <= static_cast<size_t>(target)) {
               exit(1);
            }
            m_validationClassificationTargets.push_back(target);
            if(!bNullPredictionScores) {
               if(static_cast<size_t>(m_learningTypeOrCountClassificationStates) != oneCase.m_logits.size()) {
                  exit(1);
               }
               ptrdiff_t iLogit = 0;
               for(const FractionalDataType oneLogit : oneCase.m_logits) {
                  if(std::isnan(oneLogit)) {
                     exit(1);
                  }
                  if(std::isinf(oneLogit)) {
                     exit(1);
                  }
                  if(2 == m_learningTypeOrCountClassificationStates) {
                     // binary classification
#ifdef EXPAND_BINARY_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        m_validationPredictionScores.push_back(oneLogit);
                     } else {
                        m_validationPredictionScores.push_back(oneLogit - oneCase.m_logits[m_iZeroClassificationLogit]);
                     }
#else // EXPAND_BINARY_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_validationPredictionScores.push_back(oneLogit - oneCase.m_logits[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_validationPredictionScores.push_back(oneLogit - oneCase.m_logits[m_iZeroClassificationLogit]);
                        }
                     }
#endif // EXPAND_BINARY_LOGITS
                  } else {
                     // multiclass
#ifdef REDUCE_MULTICLASS_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_validationPredictionScores.push_back(oneLogit - oneCase.m_logits[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_validationPredictionScores.push_back(oneLogit - oneCase.m_logits[m_iZeroClassificationLogit]);
                        }
                     }
#else // REDUCE_MULTICLASS_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        m_validationPredictionScores.push_back(oneLogit);
                     } else {
                        m_validationPredictionScores.push_back(oneLogit - oneCase.m_logits[m_iZeroClassificationLogit]);
                     }
#endif // REDUCE_MULTICLASS_LOGITS
                  }
                  ++iLogit;
               }
            }
         }
         for(size_t iAttribute = 0; iAttribute < cAttributes; ++iAttribute) {
            const EbmAttribute attribute = m_attributes[iAttribute];
            for(size_t iCase = 0; iCase < cCases; ++iCase) {
               const IntegerDataType data = cases[iCase].m_data[iAttribute];
               if(data < 0) {
                  exit(1);
               }
               if(attribute.countStates <= data) {
                  exit(1);
               }
               m_validationData.push_back(data);
            }
         }
      }
      m_stage = Stage::ValidationAdded;
   }

   void InitializeTraining(const IntegerDataType countInnerBags = IntegerDataType { 0 }) {
      FractionalDataType trainingRegressionTargets[1];
      IntegerDataType trainingClassificationTargets[1];
      IntegerDataType trainingData[1];
      FractionalDataType validationRegressionTargets[1];
      IntegerDataType validationClassificationTargets[1];
      IntegerDataType validationData[1];
      
      if(Stage::ValidationAdded != m_stage) {
         exit(1);
      }
      if(countInnerBags < IntegerDataType { 0 }) {
         exit(1);
      }

      if(IsClassification(m_learningTypeOrCountClassificationStates)) {
         m_pEbmTraining = InitializeTrainingClassification(randomSeed, m_attributes.size(), 0 == m_attributes.size() ? nullptr : &m_attributes[0], m_attributeCombinations.size(), 0 == m_attributeCombinations.size() ? nullptr : &m_attributeCombinations[0], 0 == m_attributeCombinationIndexes.size() ? nullptr : &m_attributeCombinationIndexes[0], m_learningTypeOrCountClassificationStates, m_trainingClassificationTargets.size(), 0 == m_trainingClassificationTargets.size() ? &trainingClassificationTargets[0] : &m_trainingClassificationTargets[0], 0 == m_trainingData.size() ? &trainingData[0] : &m_trainingData[0], m_bNullTrainingPredictionScores ? nullptr : &m_trainingPredictionScores[0], m_validationClassificationTargets.size(), 0 == m_validationClassificationTargets.size() ? &validationClassificationTargets[0] : &m_validationClassificationTargets[0], 0 == m_validationData.size() ? &validationData[0] : &m_validationData[0], m_bNullValidationPredictionScores ? nullptr : &m_validationPredictionScores[0], countInnerBags);
      } else if(k_learningTypeRegression == m_learningTypeOrCountClassificationStates) {
         m_pEbmTraining = InitializeTrainingRegression(randomSeed, m_attributes.size(), 0 == m_attributes.size() ? nullptr : &m_attributes[0], m_attributeCombinations.size(), 0 == m_attributeCombinations.size() ? nullptr : &m_attributeCombinations[0], 0 == m_attributeCombinationIndexes.size() ? nullptr : &m_attributeCombinationIndexes[0], m_trainingRegressionTargets.size(), 0 == m_trainingRegressionTargets.size() ? &trainingRegressionTargets[0] : &m_trainingRegressionTargets[0], 0 == m_trainingData.size() ? &trainingData[0] : &m_trainingData[0], m_bNullTrainingPredictionScores ? nullptr : &m_trainingPredictionScores[0], m_validationRegressionTargets.size(), 0 == m_validationRegressionTargets.size() ? &validationRegressionTargets[0] : &m_validationRegressionTargets[0], 0 == m_validationData.size() ? &validationData[0] : &m_validationData[0], m_bNullValidationPredictionScores ? nullptr : &m_validationPredictionScores[0], countInnerBags);
      } else {
         exit(1);
      }

      if(nullptr == m_pEbmTraining) {
         exit(1);
      }
      m_stage = Stage::InitializedTraining;
   }

   FractionalDataType Train(const IntegerDataType indexAttributeCombination, const std::vector<FractionalDataType> trainingWeights = {}, const std::vector<FractionalDataType> validationWeights = {}, const FractionalDataType learningRate = FractionalDataType { 0.01 }, const IntegerDataType countTreeSplitsMax = IntegerDataType { 4 }, const IntegerDataType countCasesRequiredForSplitParentMin = IntegerDataType { 2 }) {
      if(Stage::InitializedTraining != m_stage) {
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
      if(countTreeSplitsMax < FractionalDataType { 0 }) {
         exit(1);
      }
      if(countCasesRequiredForSplitParentMin < FractionalDataType { 0 }) {
         exit(1);
      }

      FractionalDataType validationMetricReturn = FractionalDataType { 0 };
      const IntegerDataType ret = TrainingStep(m_pEbmTraining, indexAttributeCombination, learningRate, countTreeSplitsMax, countCasesRequiredForSplitParentMin, 0 == trainingWeights.size() ? nullptr : &trainingWeights[0], 0 == validationWeights.size() ? nullptr : &validationWeights[0], &validationMetricReturn);
      if(0 != ret) {
         exit(1);
      }
      return validationMetricReturn;
   }

   FractionalDataType GetCurrentModelValue(const size_t iAttributeCombination, const std::vector<size_t> indexes, const size_t iScore) const {
      if(Stage::InitializedTraining != m_stage) {
         exit(1);
      }
      FractionalDataType * pModel = GetCurrentModel(m_pEbmTraining, iAttributeCombination);
      FractionalDataType score = GetScore(iAttributeCombination, pModel, indexes, iScore);
      return score;
   }

   void AddInteractionCases(const std::vector<RegressionCase> cases) {
      if(Stage::AttributesAdded != m_stage) {
         exit(1);
      }
      if(k_learningTypeRegression != m_learningTypeOrCountClassificationStates) {
         exit(1);
      }
      const size_t cCases = cases.size();
      if(0 != cCases) {
         const size_t cAttributes = m_attributes.size();
         const bool bNullPredictionScores = cases[0].m_bNullPredictionScores;
         m_bNullInteractionPredictionScores = bNullPredictionScores;

         for(const RegressionCase oneCase : cases) {
            if(cAttributes != oneCase.m_data.size()) {
               exit(1);
            }
            if(bNullPredictionScores != oneCase.m_bNullPredictionScores) {
               exit(1);
            }
            const FractionalDataType target = oneCase.m_target;
            if(std::isnan(target)) {
               exit(1);
            }
            if(std::isinf(target)) {
               exit(1);
            }
            m_interactionRegressionTargets.push_back(target);
            if(!bNullPredictionScores) {
               const FractionalDataType score = oneCase.m_score;
               if(std::isnan(score)) {
                  exit(1);
               }
               if(std::isinf(score)) {
                  exit(1);
               }
               m_interactionPredictionScores.push_back(score);
            }
         }
         for(size_t iAttribute = 0; iAttribute < cAttributes; ++iAttribute) {
            const EbmAttribute attribute = m_attributes[iAttribute];
            for(size_t iCase = 0; iCase < cCases; ++iCase) {
               const IntegerDataType data = cases[iCase].m_data[iAttribute];
               if(data < 0) {
                  exit(1);
               }
               if(attribute.countStates <= data) {
                  exit(1);
               }
               m_interactionData.push_back(data);
            }
         }
      }
      m_stage = Stage::InteractionAdded;
   }

   void AddInteractionCases(const std::vector<ClassificationCase> cases) {
      if(Stage::AttributesAdded != m_stage) {
         exit(1);
      }
      if(!IsClassification(m_learningTypeOrCountClassificationStates)) {
         exit(1);
      }
      const size_t cCases = cases.size();
      if(0 != cCases) {
         const size_t cAttributes = m_attributes.size();
         const bool bNullPredictionScores = cases[0].m_bNullPredictionScores;
         m_bNullInteractionPredictionScores = bNullPredictionScores;

         for(const ClassificationCase oneCase : cases) {
            if(cAttributes != oneCase.m_data.size()) {
               exit(1);
            }
            if(bNullPredictionScores != oneCase.m_bNullPredictionScores) {
               exit(1);
            }
            const IntegerDataType target = oneCase.m_target;
            if(target < 0) {
               exit(1);
            }
            if(static_cast<size_t>(m_learningTypeOrCountClassificationStates) <= static_cast<size_t>(target)) {
               exit(1);
            }
            m_interactionClassificationTargets.push_back(target);
            if(!bNullPredictionScores) {
               if(static_cast<size_t>(m_learningTypeOrCountClassificationStates) != oneCase.m_logits.size()) {
                  exit(1);
               }
               ptrdiff_t iLogit = 0;
               for(const FractionalDataType oneLogit : oneCase.m_logits) {
                  if(std::isnan(oneLogit)) {
                     exit(1);
                  }
                  if(std::isinf(oneLogit)) {
                     exit(1);
                  }
                  if(2 == m_learningTypeOrCountClassificationStates) {
                     // binary classification
#ifdef EXPAND_BINARY_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        m_interactionPredictionScores.push_back(oneLogit);
                     } else {
                        m_interactionPredictionScores.push_back(oneLogit - oneCase.m_logits[m_iZeroClassificationLogit]);
                     }
#else // EXPAND_BINARY_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_interactionPredictionScores.push_back(oneLogit - oneCase.m_logits[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_interactionPredictionScores.push_back(oneLogit - oneCase.m_logits[m_iZeroClassificationLogit]);
                        }
                     }
#endif // EXPAND_BINARY_LOGITS
                  } else {
                     // multiclass
#ifdef REDUCE_MULTICLASS_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_interactionPredictionScores.push_back(oneLogit - oneCase.m_logits[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_interactionPredictionScores.push_back(oneLogit - oneCase.m_logits[m_iZeroClassificationLogit]);
                        }
                     }
#else // REDUCE_MULTICLASS_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        m_interactionPredictionScores.push_back(oneLogit);
                     } else {
                        m_interactionPredictionScores.push_back(oneLogit - oneCase.m_logits[m_iZeroClassificationLogit]);
                     }
#endif // REDUCE_MULTICLASS_LOGITS
                  }
                  ++iLogit;
               }
            }
         }
         for(size_t iAttribute = 0; iAttribute < cAttributes; ++iAttribute) {
            const EbmAttribute attribute = m_attributes[iAttribute];
            for(size_t iCase = 0; iCase < cCases; ++iCase) {
               const IntegerDataType data = cases[iCase].m_data[iAttribute];
               if(data < 0) {
                  exit(1);
               }
               if(attribute.countStates <= data) {
                  exit(1);
               }
               m_interactionData.push_back(data);
            }
         }
      }
      m_stage = Stage::InteractionAdded;
   }

   void InitializeInteraction() {
      FractionalDataType interactionRegressionTargets[1];
      IntegerDataType interactionClassificationTargets[1];
      IntegerDataType interactionData[1];

      if(Stage::InteractionAdded != m_stage) {
         exit(1);
      }

      if(IsClassification(m_learningTypeOrCountClassificationStates)) {
         m_pEbmInteraction = InitializeInteractionClassification(m_attributes.size(), 0 == m_attributes.size() ? nullptr : &m_attributes[0], m_learningTypeOrCountClassificationStates, m_interactionClassificationTargets.size(), 0 == m_interactionClassificationTargets.size() ? &interactionClassificationTargets[0] : &m_interactionClassificationTargets[0], 0 == m_interactionData.size() ? &interactionData[0] : &m_interactionData[0], m_bNullInteractionPredictionScores ? nullptr : &m_interactionPredictionScores[0]);
      } else if(k_learningTypeRegression == m_learningTypeOrCountClassificationStates) {
         m_pEbmInteraction = InitializeInteractionRegression(m_attributes.size(), 0 == m_attributes.size() ? nullptr : &m_attributes[0], m_interactionRegressionTargets.size(), 0 == m_interactionRegressionTargets.size() ? &interactionRegressionTargets[0] : &m_interactionRegressionTargets[0], 0 == m_interactionData.size() ? &interactionData[0] : &m_interactionData[0], m_bNullInteractionPredictionScores ? nullptr : &m_interactionPredictionScores[0]);
      } else {
         exit(1);
      }

      if(nullptr == m_pEbmInteraction) {
         exit(1);
      }
      m_stage = Stage::InitializedInteraction;
   }

   FractionalDataType InteractionScore(const std::vector<IntegerDataType> attributesInCombination) const {
      if(Stage::InitializedInteraction != m_stage) {
         exit(1);
      }
      for(const IntegerDataType oneAttributeIndex : attributesInCombination) {
         if(oneAttributeIndex < IntegerDataType { 0 }) {
            exit(1);
         }
         if(m_attributes.size() <= static_cast<size_t>(oneAttributeIndex)) {
            exit(1);
         }
      }

      FractionalDataType interactionScoreReturn = FractionalDataType { 0 };
      const IntegerDataType ret = GetInteractionScore(m_pEbmInteraction, attributesInCombination.size(), 0 == attributesInCombination.size() ? nullptr : &attributesInCombination[0], &interactionScoreReturn);
      if(0 != ret) {
         exit(1);
      }
      return interactionScoreReturn;
   }
};

TEST_CASE("zero AttributeCombination, training, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddAttributes({});
   test.AddAttributeCombinations({});
   test.AddTrainingCases({ RegressionCase(10, {}) });
   test.AddValidationCases({ RegressionCase(12, {}) });
   test.InitializeTraining();

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just crash/assert if there was a problem
   // training isn't legal since we'd need to specify an attributeCombination index
}

TEST_CASE("zero AttributeCombination, training, binary") {
   TestApi test = TestApi(2);
   test.AddAttributes({});
   test.AddAttributeCombinations({});
   test.AddTrainingCases({ ClassificationCase(1, {}) });
   test.AddValidationCases({ ClassificationCase(1, {}) });
   test.InitializeTraining();

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just crash/assert if there was a problem
   // training isn't legal since we'd need to specify an attributeCombination index
}

TEST_CASE("zero AttributeCombination, training, multiclass") {
   TestApi test = TestApi(3);
   test.AddAttributes({});
   test.AddAttributeCombinations({});
   test.AddTrainingCases({ ClassificationCase(2, {}) });
   test.AddValidationCases({ ClassificationCase(2, {}) });
   test.InitializeTraining();

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just crash/assert if there was a problem
   // training isn't legal since we'd need to specify an attributeCombination index
}

TEST_CASE("AttributeCombination with zero attributes, training, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddAttributes({});
   test.AddAttributeCombinations({ {} });
   test.AddTrainingCases({ RegressionCase(10, {}) });
   test.AddValidationCases({ RegressionCase(12, {}) });
   test.InitializeTraining();

   FractionalDataType validationMetric = std::numeric_limits<FractionalDataType>::quiet_NaN();
   FractionalDataType modelValue = std::numeric_limits<FractionalDataType>::quiet_NaN();
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iAttributeCombination = 0; iAttributeCombination < test.GetAttributeCombinationsCount(); ++iAttributeCombination) {
         validationMetric = test.Train(iAttributeCombination);
         if(0 == iAttributeCombination && 0 == iEpoch) {
            CHECK_APPROX(validationMetric, 11.900000000000000);
            modelValue = test.GetCurrentModelValue(iAttributeCombination, {}, 0);
            CHECK_APPROX(modelValue, 0.1000000000000000);
         }
         if(0 == iAttributeCombination && 1 == iEpoch) {
            CHECK_APPROX(validationMetric, 11.801000000000000);
            modelValue = test.GetCurrentModelValue(iAttributeCombination, {}, 0);
            CHECK_APPROX(modelValue, 0.1990000000000000);
         }
      }
   }
   CHECK_APPROX(validationMetric, 2.0004317124741098);
   modelValue = test.GetCurrentModelValue(0, {}, 0);
   CHECK_APPROX(modelValue, 9.9995682875258822);
}

TEST_CASE("AttributeCombination with zero attributes, training, binary") {
   TestApi test = TestApi(2);
   test.AddAttributes({});
   test.AddAttributeCombinations({ {} });
   test.AddTrainingCases({ ClassificationCase(0, {}) });
   test.AddValidationCases({ ClassificationCase(0, {}) });
   test.InitializeTraining();

   FractionalDataType validationMetric = std::numeric_limits<FractionalDataType>::quiet_NaN();
   FractionalDataType modelValue = std::numeric_limits<FractionalDataType>::quiet_NaN();
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iAttributeCombination = 0; iAttributeCombination < test.GetAttributeCombinationsCount(); ++iAttributeCombination) {
         validationMetric = test.Train(iAttributeCombination);
         if(0 == iAttributeCombination && 0 == iEpoch) {
            CHECK_APPROX(validationMetric, 0.68319717972663419);
            modelValue = test.GetCurrentModelValue(iAttributeCombination, {}, 0);
            CHECK_APPROX(modelValue, 0);
            modelValue = test.GetCurrentModelValue(iAttributeCombination, {}, 1);
            CHECK_APPROX(modelValue, -0.020000000000000000);
         }
         if(0 == iAttributeCombination && 1 == iEpoch) {
            CHECK_APPROX(validationMetric, 0.67344419889200957);
            modelValue = test.GetCurrentModelValue(iAttributeCombination, {}, 0);
            CHECK_APPROX(modelValue, 0);
            modelValue = test.GetCurrentModelValue(iAttributeCombination, {}, 1);
            CHECK_APPROX(modelValue, -0.039801986733067563);
         }
      }
   }
   CHECK_APPROX(validationMetric, 2.2621439908125974e-05);
   modelValue = test.GetCurrentModelValue(0, {}, 0);
   CHECK_APPROX(modelValue, 0);
   modelValue = test.GetCurrentModelValue(0, {}, 1);
   CHECK_APPROX(modelValue, -10.696601122148364);
}

TEST_CASE("AttributeCombination with zero attributes, training, multiclass") {
   TestApi test = TestApi(3);
   test.AddAttributes({ });
   test.AddAttributeCombinations({ {} });
   test.AddTrainingCases({ ClassificationCase(0, {}) });
   test.AddValidationCases({ ClassificationCase(0, {}) });
   test.InitializeTraining();

   FractionalDataType validationMetric = std::numeric_limits<FractionalDataType>::quiet_NaN();
   FractionalDataType modelValue = std::numeric_limits<FractionalDataType>::quiet_NaN();
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iAttributeCombination = 0; iAttributeCombination < test.GetAttributeCombinationsCount(); ++iAttributeCombination) {
         validationMetric = test.Train(iAttributeCombination);
         if(0 == iAttributeCombination && 0 == iEpoch) {
            CHECK_APPROX(validationMetric, 1.0688384008227103);
            modelValue = test.GetCurrentModelValue(iAttributeCombination, {}, 0);
            CHECK_APPROX(modelValue, 0.03000000000000000);
            modelValue = test.GetCurrentModelValue(iAttributeCombination, {}, 1);
            CHECK_APPROX(modelValue, -0.01500000000000000);
            modelValue = test.GetCurrentModelValue(iAttributeCombination, {}, 2);
            CHECK_APPROX(modelValue, -0.01500000000000000);
         }
         if(0 == iAttributeCombination && 1 == iEpoch) {
            CHECK_APPROX(validationMetric, 1.0401627411809615);
            modelValue = test.GetCurrentModelValue(iAttributeCombination, {}, 0);
            CHECK_APPROX(modelValue, 0.059119949636662006);
            modelValue = test.GetCurrentModelValue(iAttributeCombination, {}, 1);
            CHECK_APPROX(modelValue, -0.029887518980531450);
            modelValue = test.GetCurrentModelValue(iAttributeCombination, {}, 2);
            CHECK_APPROX(modelValue, -0.029887518980531450);
         }
      }
   }
   CHECK_APPROX(validationMetric, 1.7171897252232722e-09);
   modelValue = test.GetCurrentModelValue(0, {}, 0);
   CHECK_APPROX(modelValue, 10.643234965479628);
   modelValue = test.GetCurrentModelValue(0, {}, 1);
   CHECK_APPROX(modelValue, -10.232489007525166);
   modelValue = test.GetCurrentModelValue(0, {}, 2);
   CHECK_APPROX(modelValue, -10.232489007525166);
}

TEST_CASE("AttributeCombination with zero attributes, interaction, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddAttributes({});
   test.AddInteractionCases({ RegressionCase(10, {}) });
   test.InitializeInteraction();
   FractionalDataType metricReturn = test.InteractionScore({});
   CHECK(0 == metricReturn);
}

TEST_CASE("AttributeCombination with zero attributes, interaction, binary") {
   TestApi test = TestApi(2);
   test.AddAttributes({});
   test.AddInteractionCases({ ClassificationCase(0, {}) });
   test.InitializeInteraction();
   FractionalDataType metricReturn = test.InteractionScore({});
   CHECK(0 == metricReturn);
}

TEST_CASE("AttributeCombination with zero attributes, interaction, multiclass") {
   TestApi test = TestApi(3);
   test.AddAttributes({});
   test.AddInteractionCases({ ClassificationCase(0, {}) });
   test.InitializeInteraction();
   FractionalDataType metricReturn = test.InteractionScore({});
   CHECK(0 == metricReturn);
}

TEST_CASE("AttributeCombination with one attribute with one or two states is the exact same as zero AttributeCombinations, training, regression") {
   TestApi testZeroAttributesInCombination = TestApi(k_learningTypeRegression);
   testZeroAttributesInCombination.AddAttributes({});
   testZeroAttributesInCombination.AddAttributeCombinations({ {} });
   testZeroAttributesInCombination.AddTrainingCases({ RegressionCase(10, {}) });
   testZeroAttributesInCombination.AddValidationCases({ RegressionCase(12, {}) });
   testZeroAttributesInCombination.InitializeTraining();

   TestApi testOneState = TestApi(k_learningTypeRegression);
   testOneState.AddAttributes({ Attribute(1) });
   testOneState.AddAttributeCombinations({ { 0 } });
   testOneState.AddTrainingCases({ RegressionCase(10, { 0 }) });
   testOneState.AddValidationCases({ RegressionCase(12, { 0 }) });
   testOneState.InitializeTraining();

   TestApi testTwoStates = TestApi(k_learningTypeRegression);
   testTwoStates.AddAttributes({ Attribute(2) });
   testTwoStates.AddAttributeCombinations({ { 0 } });
   testTwoStates.AddTrainingCases({ RegressionCase(10, { 1 }) });
   testTwoStates.AddValidationCases({ RegressionCase(12, { 1 }) });
   testTwoStates.InitializeTraining();

   FractionalDataType validationMetricZeroAttributesInCombination;
   FractionalDataType modelValueZeroAttributesInCombination;
   FractionalDataType validationMetricOneState;
   FractionalDataType modelValueOneState;
   FractionalDataType validationMetricTwoStates;
   FractionalDataType modelValueTwoStates;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroAttributesInCombination.GetAttributeCombinationsCount() == testOneState.GetAttributeCombinationsCount());
      assert(testZeroAttributesInCombination.GetAttributeCombinationsCount() == testTwoStates.GetAttributeCombinationsCount());
      for(size_t iAttributeCombination = 0; iAttributeCombination < testZeroAttributesInCombination.GetAttributeCombinationsCount(); ++iAttributeCombination) {
         validationMetricZeroAttributesInCombination = testZeroAttributesInCombination.Train(iAttributeCombination);
         validationMetricOneState = testOneState.Train(iAttributeCombination);
         CHECK_APPROX(validationMetricZeroAttributesInCombination, validationMetricOneState);
         validationMetricTwoStates = testTwoStates.Train(iAttributeCombination);
         CHECK_APPROX(validationMetricZeroAttributesInCombination, validationMetricTwoStates);

         modelValueZeroAttributesInCombination = testZeroAttributesInCombination.GetCurrentModelValue(iAttributeCombination, {}, 0);
         modelValueOneState = testOneState.GetCurrentModelValue(iAttributeCombination, { 0 }, 0);
         CHECK_APPROX(modelValueZeroAttributesInCombination, modelValueOneState);
         modelValueTwoStates = testTwoStates.GetCurrentModelValue(iAttributeCombination, { 1 }, 0);
         CHECK_APPROX(modelValueZeroAttributesInCombination, modelValueTwoStates);
      }
   }
}

TEST_CASE("AttributeCombination with one attribute with one or two states is the exact same as zero AttributeCombinations, training, binary") {
   TestApi testZeroAttributesInCombination = TestApi(2);
   testZeroAttributesInCombination.AddAttributes({});
   testZeroAttributesInCombination.AddAttributeCombinations({ {} });
   testZeroAttributesInCombination.AddTrainingCases({ ClassificationCase(0, {}) });
   testZeroAttributesInCombination.AddValidationCases({ ClassificationCase(0, {}) });
   testZeroAttributesInCombination.InitializeTraining();

   TestApi testOneState = TestApi(2);
   testOneState.AddAttributes({ Attribute(1) });
   testOneState.AddAttributeCombinations({ { 0 } });
   testOneState.AddTrainingCases({ ClassificationCase(0, { 0 }) });
   testOneState.AddValidationCases({ ClassificationCase(0, { 0 }) });
   testOneState.InitializeTraining();

   TestApi testTwoStates = TestApi(2);
   testTwoStates.AddAttributes({ Attribute(2) });
   testTwoStates.AddAttributeCombinations({ { 0 } });
   testTwoStates.AddTrainingCases({ ClassificationCase(0, { 1 }) });
   testTwoStates.AddValidationCases({ ClassificationCase(0, { 1 }) });
   testTwoStates.InitializeTraining();

   FractionalDataType validationMetricZeroAttributesInCombination;
   FractionalDataType modelValueZeroAttributesInCombination;
   FractionalDataType validationMetricOneState;
   FractionalDataType modelValueOneState;
   FractionalDataType validationMetricTwoStates;
   FractionalDataType modelValueTwoStates;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroAttributesInCombination.GetAttributeCombinationsCount() == testOneState.GetAttributeCombinationsCount());
      assert(testZeroAttributesInCombination.GetAttributeCombinationsCount() == testTwoStates.GetAttributeCombinationsCount());
      for(size_t iAttributeCombination = 0; iAttributeCombination < testZeroAttributesInCombination.GetAttributeCombinationsCount(); ++iAttributeCombination) {
         validationMetricZeroAttributesInCombination = testZeroAttributesInCombination.Train(iAttributeCombination);
         validationMetricOneState = testOneState.Train(iAttributeCombination);
         CHECK_APPROX(validationMetricZeroAttributesInCombination, validationMetricOneState);
         validationMetricTwoStates = testTwoStates.Train(iAttributeCombination);
         CHECK_APPROX(validationMetricZeroAttributesInCombination, validationMetricTwoStates);

         modelValueZeroAttributesInCombination = testZeroAttributesInCombination.GetCurrentModelValue(iAttributeCombination, {}, 0);
         modelValueOneState = testOneState.GetCurrentModelValue(iAttributeCombination, { 0 }, 0);
         CHECK_APPROX(modelValueZeroAttributesInCombination, modelValueOneState);
         modelValueTwoStates = testTwoStates.GetCurrentModelValue(iAttributeCombination, { 1 }, 0);
         CHECK_APPROX(modelValueZeroAttributesInCombination, modelValueTwoStates);

         modelValueZeroAttributesInCombination = testZeroAttributesInCombination.GetCurrentModelValue(iAttributeCombination, {}, 1);
         modelValueOneState = testOneState.GetCurrentModelValue(iAttributeCombination, { 0 }, 1);
         CHECK_APPROX(modelValueZeroAttributesInCombination, modelValueOneState);
         modelValueTwoStates = testTwoStates.GetCurrentModelValue(iAttributeCombination, { 1 }, 1);
         CHECK_APPROX(modelValueZeroAttributesInCombination, modelValueTwoStates);
      }
   }
}

TEST_CASE("AttributeCombination with one attribute with one or two states is the exact same as zero AttributeCombinations, training, multiclass") {
   TestApi testZeroAttributesInCombination = TestApi(3);
   testZeroAttributesInCombination.AddAttributes({});
   testZeroAttributesInCombination.AddAttributeCombinations({ {} });
   testZeroAttributesInCombination.AddTrainingCases({ ClassificationCase(0, {}) });
   testZeroAttributesInCombination.AddValidationCases({ ClassificationCase(0, {}) });
   testZeroAttributesInCombination.InitializeTraining();

   TestApi testOneState = TestApi(3);
   testOneState.AddAttributes({ Attribute(1) });
   testOneState.AddAttributeCombinations({ { 0 } });
   testOneState.AddTrainingCases({ ClassificationCase(0, { 0 }) });
   testOneState.AddValidationCases({ ClassificationCase(0, { 0 }) });
   testOneState.InitializeTraining();

   TestApi testTwoStates = TestApi(3);
   testTwoStates.AddAttributes({ Attribute(2) });
   testTwoStates.AddAttributeCombinations({ { 0 } });
   testTwoStates.AddTrainingCases({ ClassificationCase(0, { 1 }) });
   testTwoStates.AddValidationCases({ ClassificationCase(0, { 1 }) });
   testTwoStates.InitializeTraining();

   FractionalDataType validationMetricZeroAttributesInCombination;
   FractionalDataType modelValueZeroAttributesInCombination;
   FractionalDataType validationMetricOneState;
   FractionalDataType modelValueOneState;
   FractionalDataType validationMetricTwoStates;
   FractionalDataType modelValueTwoStates;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroAttributesInCombination.GetAttributeCombinationsCount() == testOneState.GetAttributeCombinationsCount());
      assert(testZeroAttributesInCombination.GetAttributeCombinationsCount() == testTwoStates.GetAttributeCombinationsCount());
      for(size_t iAttributeCombination = 0; iAttributeCombination < testZeroAttributesInCombination.GetAttributeCombinationsCount(); ++iAttributeCombination) {
         validationMetricZeroAttributesInCombination = testZeroAttributesInCombination.Train(iAttributeCombination);
         validationMetricOneState = testOneState.Train(iAttributeCombination);
         CHECK_APPROX(validationMetricZeroAttributesInCombination, validationMetricOneState);
         validationMetricTwoStates = testTwoStates.Train(iAttributeCombination);
         CHECK_APPROX(validationMetricZeroAttributesInCombination, validationMetricTwoStates);

         modelValueZeroAttributesInCombination = testZeroAttributesInCombination.GetCurrentModelValue(iAttributeCombination, {}, 0);
         modelValueOneState = testOneState.GetCurrentModelValue(iAttributeCombination, { 0 }, 0);
         CHECK_APPROX(modelValueZeroAttributesInCombination, modelValueOneState);
         modelValueTwoStates = testTwoStates.GetCurrentModelValue(iAttributeCombination, { 1 }, 0);
         CHECK_APPROX(modelValueZeroAttributesInCombination, modelValueTwoStates);

         modelValueZeroAttributesInCombination = testZeroAttributesInCombination.GetCurrentModelValue(iAttributeCombination, {}, 1);
         modelValueOneState = testOneState.GetCurrentModelValue(iAttributeCombination, { 0 }, 1);
         CHECK_APPROX(modelValueZeroAttributesInCombination, modelValueOneState);
         modelValueTwoStates = testTwoStates.GetCurrentModelValue(iAttributeCombination, { 1 }, 1);
         CHECK_APPROX(modelValueZeroAttributesInCombination, modelValueTwoStates);

         modelValueZeroAttributesInCombination = testZeroAttributesInCombination.GetCurrentModelValue(iAttributeCombination, {}, 2);
         modelValueOneState = testOneState.GetCurrentModelValue(iAttributeCombination, { 0 }, 2);
         CHECK_APPROX(modelValueZeroAttributesInCombination, modelValueOneState);
         modelValueTwoStates = testTwoStates.GetCurrentModelValue(iAttributeCombination, { 1 }, 2);
         CHECK_APPROX(modelValueZeroAttributesInCombination, modelValueTwoStates);
      }
   }
}

TEST_CASE("AttributeCombination with one attribute with one state, interaction, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddAttributes({ Attribute(1) });
   test.AddInteractionCases({ RegressionCase(10, { 0 }) });
   test.InitializeInteraction();
   FractionalDataType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("AttributeCombination with one attribute with one state, interaction, Binary") {
   TestApi test = TestApi(2);
   test.AddAttributes({ Attribute(1) });
   test.AddInteractionCases({ ClassificationCase(0, { 0 }) });
   test.InitializeInteraction();
   FractionalDataType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("AttributeCombination with one attribute with one state, interaction, multiclass") {
   TestApi test = TestApi(3);
   test.AddAttributes({ Attribute(1) });
   test.AddInteractionCases({ ClassificationCase(0, { 0 }) });
   test.InitializeInteraction();
   FractionalDataType metricReturn = test.InteractionScore({0});
   CHECK(0 == metricReturn);
}

TEST_CASE("Test Rehydration, training, regression") {
   TestApi testContinuous = TestApi(k_learningTypeRegression);
   testContinuous.AddAttributes({});
   testContinuous.AddAttributeCombinations({ {} });
   testContinuous.AddTrainingCases({ RegressionCase(10, {}) });
   testContinuous.AddValidationCases({ RegressionCase(12, {}) });
   testContinuous.InitializeTraining();

   FractionalDataType model0 = 0;

   FractionalDataType validationMetricContinuous;
   FractionalDataType modelValueContinuous;
   FractionalDataType validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(k_learningTypeRegression);
      testRestart.AddAttributes({});
      testRestart.AddAttributeCombinations({ {} });
      testRestart.AddTrainingCases({ RegressionCase(10, {}, model0) });
      testRestart.AddValidationCases({ RegressionCase(12, {}, model0) });
      testRestart.InitializeTraining();

      validationMetricRestart = testRestart.Train(0);
      validationMetricContinuous = testContinuous.Train(0);
      CHECK_APPROX(validationMetricContinuous, validationMetricRestart);

      modelValueContinuous = testContinuous.GetCurrentModelValue(0, {}, 0);
      model0 += testRestart.GetCurrentModelValue(0, {}, 0);
      CHECK_APPROX(modelValueContinuous, model0);
   }
}

TEST_CASE("Test Rehydration, training, binary") {
   TestApi testContinuous = TestApi(2);
   testContinuous.AddAttributes({});
   testContinuous.AddAttributeCombinations({ {} });
   testContinuous.AddTrainingCases({ ClassificationCase(0, {}) });
   testContinuous.AddValidationCases({ ClassificationCase(0, {}) });
   testContinuous.InitializeTraining();

   FractionalDataType model0 = 0;
   FractionalDataType model1 = 0;

   FractionalDataType validationMetricContinuous;
   FractionalDataType modelValueContinuous;
   FractionalDataType validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(2);
      testRestart.AddAttributes({});
      testRestart.AddAttributeCombinations({ {} });
      testRestart.AddTrainingCases({ ClassificationCase(0, {}, { model0, model1 }) });
      testRestart.AddValidationCases({ ClassificationCase(0, {}, { model0, model1 }) });
      testRestart.InitializeTraining();

      validationMetricRestart = testRestart.Train(0);
      validationMetricContinuous = testContinuous.Train(0);
      CHECK_APPROX(validationMetricContinuous, validationMetricRestart);

      modelValueContinuous = testContinuous.GetCurrentModelValue(0, {}, 0);
      model0 += testRestart.GetCurrentModelValue(0, {}, 0);
      CHECK_APPROX(modelValueContinuous, model0);

      modelValueContinuous = testContinuous.GetCurrentModelValue(0, {}, 1);
      model1 += testRestart.GetCurrentModelValue(0, {}, 1);
      CHECK_APPROX(modelValueContinuous, model1);
   }
}

TEST_CASE("Test Rehydration, training, multiclass") {
   TestApi testContinuous = TestApi(3);
   testContinuous.AddAttributes({});
   testContinuous.AddAttributeCombinations({ {} });
   testContinuous.AddTrainingCases({ ClassificationCase(0, {}) });
   testContinuous.AddValidationCases({ ClassificationCase(0, {}) });
   testContinuous.InitializeTraining();

   FractionalDataType model0 = 0;
   FractionalDataType model1 = 0;
   FractionalDataType model2 = 0;

   FractionalDataType validationMetricContinuous;
   FractionalDataType modelValueContinuous;
   FractionalDataType validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(3);
      testRestart.AddAttributes({});
      testRestart.AddAttributeCombinations({ {} });
      testRestart.AddTrainingCases({ ClassificationCase(0, {}, { model0, model1, model2 }) });
      testRestart.AddValidationCases({ ClassificationCase(0, {}, { model0, model1, model2 }) });
      testRestart.InitializeTraining();

      validationMetricRestart = testRestart.Train(0);
      validationMetricContinuous = testContinuous.Train(0);
      CHECK_APPROX(validationMetricContinuous, validationMetricRestart);

      modelValueContinuous = testContinuous.GetCurrentModelValue(0, {}, 0);
      model0 += testRestart.GetCurrentModelValue(0, {}, 0);
      CHECK_APPROX(modelValueContinuous, model0);

      modelValueContinuous = testContinuous.GetCurrentModelValue(0, {}, 1);
      model1 += testRestart.GetCurrentModelValue(0, {}, 1);
      CHECK_APPROX(modelValueContinuous, model1);

      modelValueContinuous = testContinuous.GetCurrentModelValue(0, {}, 2);
      model2 += testRestart.GetCurrentModelValue(0, {}, 2);
      CHECK_APPROX(modelValueContinuous, model2);
   }
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
