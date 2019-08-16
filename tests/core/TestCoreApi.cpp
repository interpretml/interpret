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
#include <string.h>

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






// EBM/interpret specific stuff below here!!

constexpr ptrdiff_t k_learningTypeRegression = ptrdiff_t { -1 };
constexpr bool IsClassification(const ptrdiff_t learningTypeOrCountTargetClasses) {
   return 0 <= learningTypeOrCountTargetClasses;
}

constexpr size_t GetVectorLength(const ptrdiff_t learningTypeOrCountTargetClasses) {
#ifdef EXPAND_BINARY_LOGITS
#ifdef REDUCE_MULTICLASS_LOGITS

   // EXPAND_BINARY_LOGITS && REDUCE_MULTICLASS_LOGITS
#error we should not be expanding binary logits while reducing multiclass logits

#else // REDUCE_MULTICLASS_LOGITS

   // EXPAND_BINARY_LOGITS && !REDUCE_MULTICLASS_LOGITS
   return learningTypeOrCountTargetClasses <= ptrdiff_t { 1 } ? size_t { 1 } : static_cast<size_t>(learningTypeOrCountTargetClasses);

#endif // REDUCE_MULTICLASS_LOGITS
#else // EXPAND_BINARY_LOGITS
#ifdef REDUCE_MULTICLASS_LOGITS

   // !EXPAND_BINARY_LOGITS && REDUCE_MULTICLASS_LOGITS
   return learningTypeOrCountTargetClasses <= ptrdiff_t { 2 } ? size_t { 1 } : static_cast<size_t>(learningTypeOrCountTargetClasses) - size_t { 1 };

#else // REDUCE_MULTICLASS_LOGITS

   // !EXPAND_BINARY_LOGITS && !REDUCE_MULTICLASS_LOGITS
   return learningTypeOrCountTargetClasses <= ptrdiff_t { 2 } ? size_t { 1 } : static_cast<size_t>(learningTypeOrCountTargetClasses);

#endif // REDUCE_MULTICLASS_LOGITS
#endif // EXPAND_BINARY_LOGITS
}

constexpr IntegerDataType randomSeed = 42;
enum class FeatureType : IntegerDataType { Ordinal = FeatureTypeOrdinal, Nominal = FeatureTypeNominal };

class FeatureTest final {
public:

   const FeatureType m_featureType;
   const bool m_hasMissing;
   const IntegerDataType m_countBins;

   FeatureTest(const IntegerDataType countBins, const FeatureType featureType = FeatureType::Ordinal, const bool hasMissing = false) :
      m_featureType(featureType),
      m_hasMissing(hasMissing),
      m_countBins(countBins) {
      if(countBins < 0) {
         exit(1);
      }
   }
};

class RegressionInstance final {
public:
   const FractionalDataType m_target;
   const std::vector<IntegerDataType> m_binnedDataPerFeatureArray;
   const FractionalDataType m_priorPredictorPrediction;
   const bool m_bNullPredictionScores;

   RegressionInstance(const FractionalDataType target, const std::vector<IntegerDataType> binnedDataPerFeatureArray) :
      m_target(target),
      m_binnedDataPerFeatureArray(binnedDataPerFeatureArray),
      m_priorPredictorPrediction(0),
      m_bNullPredictionScores(true) {
   }

   RegressionInstance(const FractionalDataType target, const std::vector<IntegerDataType> binnedDataPerFeatureArray, const FractionalDataType priorPredictorPrediction) :
      m_target(target),
      m_binnedDataPerFeatureArray(binnedDataPerFeatureArray),
      m_priorPredictorPrediction(priorPredictorPrediction),
      m_bNullPredictionScores(false) {
   }
};

class ClassificationInstance final {
public:
   const IntegerDataType m_target;
   const std::vector<IntegerDataType> m_binnedDataPerFeatureArray;
   const std::vector<FractionalDataType> m_priorPredictorPerClassLogits;
   const bool m_bNullPredictionScores;

   ClassificationInstance(const IntegerDataType target, const std::vector<IntegerDataType> binnedDataPerFeatureArray) :
      m_target(target),
      m_binnedDataPerFeatureArray(binnedDataPerFeatureArray),
      m_bNullPredictionScores(true) {
   }

   ClassificationInstance(const IntegerDataType target, const std::vector<IntegerDataType> binnedDataPerFeatureArray, const std::vector<FractionalDataType> priorPredictorPerClassLogits) :
      m_target(target),
      m_binnedDataPerFeatureArray(binnedDataPerFeatureArray),
      m_priorPredictorPerClassLogits(priorPredictorPerClassLogits),
      m_bNullPredictionScores(false) {
   }
};

static constexpr ptrdiff_t k_iZeroClassificationLogitDefault = ptrdiff_t { -1 };
static constexpr IntegerDataType k_countInnerBagsDefault = IntegerDataType { 0 };
static constexpr FractionalDataType k_learningRateDefault = FractionalDataType { 0.01 };
static constexpr IntegerDataType k_countTreeSplitsMaxDefault = IntegerDataType { 4 };
static constexpr IntegerDataType k_countInstancesRequiredForParentSplitMinDefault = IntegerDataType { 2 };

class TestApi {
   enum class Stage {
      Beginning, FeaturesAdded, FeatureCombinationsAdded, TrainingAdded, ValidationAdded, InitializedTraining, InteractionAdded, InitializedInteraction
   };

   Stage m_stage;
   const ptrdiff_t m_learningTypeOrCountTargetClasses;
   const ptrdiff_t m_iZeroClassificationLogit;

   std::vector<EbmCoreFeature> m_features;
   std::vector<EbmCoreFeatureCombination> m_featureCombinations;
   std::vector<IntegerDataType> m_featureCombinationIndexes;

   std::vector<std::vector<size_t>> m_countBinsByFeatureCombination;

   std::vector<FractionalDataType> m_trainingRegressionTargets;
   std::vector<IntegerDataType> m_trainingClassificationTargets;
   std::vector<IntegerDataType> m_trainingBinnedData;
   std::vector<FractionalDataType> m_trainingPredictionScores;
   bool m_bNullTrainingPredictionScores;

   std::vector<FractionalDataType> m_validationRegressionTargets;
   std::vector<IntegerDataType> m_validationClassificationTargets;
   std::vector<IntegerDataType> m_validationBinnedData;
   std::vector<FractionalDataType> m_validationPredictionScores;
   bool m_bNullValidationPredictionScores;

   PEbmTraining m_pEbmTraining;

   std::vector<FractionalDataType> m_interactionRegressionTargets;
   std::vector<IntegerDataType> m_interactionClassificationTargets;
   std::vector<IntegerDataType> m_interactionBinnedData;
   std::vector<FractionalDataType> m_interactionPredictionScores;
   bool m_bNullInteractionPredictionScores;

   PEbmInteraction m_pEbmInteraction;

   const FractionalDataType * GetPredictorScores(const size_t iFeatureCombination, const FractionalDataType * const pModelFeatureCombination, const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures) const {
      if(Stage::InitializedTraining != m_stage) {
         exit(1);
      }
      const size_t cVectorLength = GetVectorLength(m_learningTypeOrCountTargetClasses);

      if(m_countBinsByFeatureCombination.size() <= iFeatureCombination) {
         exit(1);
      }
      const std::vector<size_t> countBins = m_countBinsByFeatureCombination[iFeatureCombination];

      const size_t cDimensions = perDimensionIndexArrayForBinnedFeatures.size();
      if(cDimensions != countBins.size()) {
         exit(1);
      }
      size_t iValue = 0;
      size_t multiple = cVectorLength;
      for(size_t iDimension = 0; iDimension < cDimensions; ++iDimension) {
         if(countBins[iDimension] <= perDimensionIndexArrayForBinnedFeatures[iDimension]) {
            exit(1);
         }
         iValue += perDimensionIndexArrayForBinnedFeatures[iDimension] * multiple;
         multiple *= countBins[iDimension];
      }
      return &pModelFeatureCombination[iValue];
   }

   FractionalDataType GetPredictorScore(const size_t iFeatureCombination, const FractionalDataType * const pModelFeatureCombination, const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures, const size_t iTargetClassOrZero) const {
      const FractionalDataType * const aScores = GetPredictorScores(iFeatureCombination, pModelFeatureCombination, perDimensionIndexArrayForBinnedFeatures);
      if(!IsClassification(m_learningTypeOrCountTargetClasses)) {
         if(0 != iTargetClassOrZero) {
            exit(1);
         }
         return aScores[0];
      }
      if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) <= iTargetClassOrZero) {
         exit(1);
      }
      if(2 == m_learningTypeOrCountTargetClasses) {
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
            if(0 == iTargetClassOrZero) {
               return FractionalDataType { 0 };
            } else {
               return aScores[0];
            }
         } else {
            if(static_cast<size_t>(m_iZeroClassificationLogit) == iTargetClassOrZero) {
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
            return aScores[iTargetClassOrZero];
         } else {
            return aScores[iTargetClassOrZero] - aScores[m_iZeroClassificationLogit];
         }
#endif // REDUCE_MULTICLASS_LOGITS
      }
   }

public:

   TestApi(const ptrdiff_t learningTypeOrCountTargetClasses, const ptrdiff_t iZeroClassificationLogit = k_iZeroClassificationLogitDefault) :
      m_stage(Stage::Beginning),
      m_learningTypeOrCountTargetClasses(learningTypeOrCountTargetClasses),
      m_iZeroClassificationLogit(iZeroClassificationLogit),
      m_bNullTrainingPredictionScores(true),
      m_bNullValidationPredictionScores(true),
      m_pEbmTraining(nullptr),
      m_bNullInteractionPredictionScores(true),
      m_pEbmInteraction(nullptr) {
      if(IsClassification(learningTypeOrCountTargetClasses)) {
         if(learningTypeOrCountTargetClasses <= iZeroClassificationLogit) {
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

   size_t GetFeatureCombinationsCount() const {
      return m_featureCombinations.size();
   }

   void AddFeatures(const std::vector<FeatureTest> features) {
      if(Stage::Beginning != m_stage) {
         exit(1);
      }

      for(const FeatureTest oneFeature : features) {
         EbmCoreFeature feature;
         feature.featureType = static_cast<IntegerDataType>(oneFeature.m_featureType);
         feature.hasMissing = oneFeature.m_hasMissing ? IntegerDataType { 1 } : IntegerDataType { 0 };
         feature.countBins = oneFeature.m_countBins;
         m_features.push_back(feature);
      }

      m_stage = Stage::FeaturesAdded;
   }

   void AddFeatureCombinations(const std::vector<std::vector<size_t>> featureCombinations) {
      if(Stage::FeaturesAdded != m_stage) {
         exit(1);
      }

      for(const std::vector<size_t> oneFeatureCombination : featureCombinations) {
         EbmCoreFeatureCombination featureCombination;
         featureCombination.countFeaturesInCombination = oneFeatureCombination.size();
         m_featureCombinations.push_back(featureCombination);
         std::vector<size_t> countBins;
         for(const size_t oneIndex : oneFeatureCombination) {
            if(m_features.size() <= oneIndex) {
               exit(1);
            }
            m_featureCombinationIndexes.push_back(oneIndex);
            countBins.push_back(static_cast<size_t>(m_features[oneIndex].countBins));
         }
         m_countBinsByFeatureCombination.push_back(countBins);
      }

      m_stage = Stage::FeatureCombinationsAdded;
   }

   void AddTrainingInstances(const std::vector<RegressionInstance> instances) {
      if(Stage::FeatureCombinationsAdded != m_stage) {
         exit(1);
      }
      if(k_learningTypeRegression != m_learningTypeOrCountTargetClasses) {
         exit(1);
      }
      const size_t cCases = instances.size();
      if(0 != cCases) {
         const size_t cFeatures = m_features.size();
         const bool bNullPredictionScores = instances[0].m_bNullPredictionScores;
         m_bNullTrainingPredictionScores = bNullPredictionScores;

         for(const RegressionInstance oneInstance : instances) {
            if(cFeatures != oneInstance.m_binnedDataPerFeatureArray.size()) {
               exit(1);
            }
            if(bNullPredictionScores != oneInstance.m_bNullPredictionScores) {
               exit(1);
            }
            const FractionalDataType target = oneInstance.m_target;
            if(std::isnan(target)) {
               exit(1);
            }
            if(std::isinf(target)) {
               exit(1);
            }
            m_trainingRegressionTargets.push_back(target);
            if(!bNullPredictionScores) {
               const FractionalDataType score = oneInstance.m_priorPredictorPrediction;
               if(std::isnan(score)) {
                  exit(1);
               }
               if(std::isinf(score)) {
                  exit(1);
               }
               m_trainingPredictionScores.push_back(score);
            }
         }
         for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
            const EbmCoreFeature feature = m_features[iFeature];
            for(size_t iCase = 0; iCase < cCases; ++iCase) {
               const IntegerDataType data = instances[iCase].m_binnedDataPerFeatureArray[iFeature];
               if(data < 0) {
                  exit(1);
               }
               if(feature.countBins <= data) {
                  exit(1);
               }
               m_trainingBinnedData.push_back(data);
            }
         }
      }
      m_stage = Stage::TrainingAdded;
   }

   void AddTrainingInstances(const std::vector<ClassificationInstance> instances) {
      if(Stage::FeatureCombinationsAdded != m_stage) {
         exit(1);
      }
      if(!IsClassification(m_learningTypeOrCountTargetClasses)) {
         exit(1);
      }
      const size_t cCases = instances.size();
      if(0 != cCases) {
         const size_t cFeatures = m_features.size();
         const bool bNullPredictionScores = instances[0].m_bNullPredictionScores;
         m_bNullTrainingPredictionScores = bNullPredictionScores;

         for(const ClassificationInstance oneCase : instances) {
            if(cFeatures != oneCase.m_binnedDataPerFeatureArray.size()) {
               exit(1);
            }
            if(bNullPredictionScores != oneCase.m_bNullPredictionScores) {
               exit(1);
            }
            const IntegerDataType target = oneCase.m_target;
            if(target < 0) {
               exit(1);
            }
            if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) <= static_cast<size_t>(target)) {
               exit(1);
            }
            m_trainingClassificationTargets.push_back(target);
            if(!bNullPredictionScores) {
               if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) != oneCase.m_priorPredictorPerClassLogits.size()) {
                  exit(1);
               }
               ptrdiff_t iLogit = 0;
               for(const FractionalDataType oneLogit : oneCase.m_priorPredictorPerClassLogits) {
                  if(std::isnan(oneLogit)) {
                     exit(1);
                  }
                  if(std::isinf(oneLogit)) {
                     exit(1);
                  }
                  if(2 == m_learningTypeOrCountTargetClasses) {
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
                           m_trainingPredictionScores.push_back(oneLogit - oneCase.m_priorPredictorPerClassLogits[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_trainingPredictionScores.push_back(oneLogit - oneCase.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
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
                        m_trainingPredictionScores.push_back(oneLogit - oneCase.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                     }
#endif // REDUCE_MULTICLASS_LOGITS
                  }
                  ++iLogit;
               }
            }
         }
         for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
            const EbmCoreFeature feature = m_features[iFeature];
            for(size_t iCase = 0; iCase < cCases; ++iCase) {
               const IntegerDataType data = instances[iCase].m_binnedDataPerFeatureArray[iFeature];
               if(data < 0) {
                  exit(1);
               }
               if(feature.countBins <= data) {
                  exit(1);
               }
               m_trainingBinnedData.push_back(data);
            }
         }
      }
      m_stage = Stage::TrainingAdded;
   }

   void AddValidationInstances(const std::vector<RegressionInstance> instances) {
      if(Stage::TrainingAdded != m_stage) {
         exit(1);
      }
      if(k_learningTypeRegression != m_learningTypeOrCountTargetClasses) {
         exit(1);
      }
      const size_t cCases = instances.size();
      if(0 != cCases) {
         const size_t cFeatures = m_features.size();
         const bool bNullPredictionScores = instances[0].m_bNullPredictionScores;
         m_bNullValidationPredictionScores = bNullPredictionScores;

         for(const RegressionInstance oneCase : instances) {
            if(cFeatures != oneCase.m_binnedDataPerFeatureArray.size()) {
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
               const FractionalDataType score = oneCase.m_priorPredictorPrediction;
               if(std::isnan(score)) {
                  exit(1);
               }
               if(std::isinf(score)) {
                  exit(1);
               }
               m_validationPredictionScores.push_back(score);
            }
         }
         for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
            const EbmCoreFeature feature = m_features[iFeature];
            for(size_t iCase = 0; iCase < cCases; ++iCase) {
               const IntegerDataType data = instances[iCase].m_binnedDataPerFeatureArray[iFeature];
               if(data < 0) {
                  exit(1);
               }
               if(feature.countBins <= data) {
                  exit(1);
               }
               m_validationBinnedData.push_back(data);
            }
         }
      }
      m_stage = Stage::ValidationAdded;
   }

   void AddValidationInstances(const std::vector<ClassificationInstance> instances) {
      if(Stage::TrainingAdded != m_stage) {
         exit(1);
      }
      if(!IsClassification(m_learningTypeOrCountTargetClasses)) {
         exit(1);
      }
      const size_t cCases = instances.size();
      if(0 != cCases) {
         const size_t cFeatures = m_features.size();
         const bool bNullPredictionScores = instances[0].m_bNullPredictionScores;
         m_bNullValidationPredictionScores = bNullPredictionScores;

         for(const ClassificationInstance oneCase : instances) {
            if(cFeatures != oneCase.m_binnedDataPerFeatureArray.size()) {
               exit(1);
            }
            if(bNullPredictionScores != oneCase.m_bNullPredictionScores) {
               exit(1);
            }
            const IntegerDataType target = oneCase.m_target;
            if(target < 0) {
               exit(1);
            }
            if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) <= static_cast<size_t>(target)) {
               exit(1);
            }
            m_validationClassificationTargets.push_back(target);
            if(!bNullPredictionScores) {
               if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) != oneCase.m_priorPredictorPerClassLogits.size()) {
                  exit(1);
               }
               ptrdiff_t iLogit = 0;
               for(const FractionalDataType oneLogit : oneCase.m_priorPredictorPerClassLogits) {
                  if(std::isnan(oneLogit)) {
                     exit(1);
                  }
                  if(std::isinf(oneLogit)) {
                     exit(1);
                  }
                  if(2 == m_learningTypeOrCountTargetClasses) {
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
                           m_validationPredictionScores.push_back(oneLogit - oneCase.m_priorPredictorPerClassLogits[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_validationPredictionScores.push_back(oneLogit - oneCase.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
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
                        m_validationPredictionScores.push_back(oneLogit - oneCase.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                     }
#endif // REDUCE_MULTICLASS_LOGITS
                  }
                  ++iLogit;
               }
            }
         }
         for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
            const EbmCoreFeature feature = m_features[iFeature];
            for(size_t iCase = 0; iCase < cCases; ++iCase) {
               const IntegerDataType data = instances[iCase].m_binnedDataPerFeatureArray[iFeature];
               if(data < 0) {
                  exit(1);
               }
               if(feature.countBins <= data) {
                  exit(1);
               }
               m_validationBinnedData.push_back(data);
            }
         }
      }
      m_stage = Stage::ValidationAdded;
   }

   void InitializeTraining(const IntegerDataType countInnerBags = k_countInnerBagsDefault) {
      if(Stage::ValidationAdded != m_stage) {
         exit(1);
      }
      if(countInnerBags < IntegerDataType { 0 }) {
         exit(1);
      }

      if(IsClassification(m_learningTypeOrCountTargetClasses)) {
         m_pEbmTraining = InitializeTrainingClassification(randomSeed, m_features.size(), 0 == m_features.size() ? nullptr : &m_features[0], m_featureCombinations.size(), 0 == m_featureCombinations.size() ? nullptr : &m_featureCombinations[0], 0 == m_featureCombinationIndexes.size() ? nullptr : &m_featureCombinationIndexes[0], m_learningTypeOrCountTargetClasses, m_trainingClassificationTargets.size(), 0 == m_trainingClassificationTargets.size() ? nullptr : &m_trainingClassificationTargets[0], 0 == m_trainingBinnedData.size() ? nullptr : &m_trainingBinnedData[0], m_bNullTrainingPredictionScores ? nullptr : &m_trainingPredictionScores[0], m_validationClassificationTargets.size(), 0 == m_validationClassificationTargets.size() ? nullptr : &m_validationClassificationTargets[0], 0 == m_validationBinnedData.size() ? nullptr : &m_validationBinnedData[0], m_bNullValidationPredictionScores ? nullptr : &m_validationPredictionScores[0], countInnerBags);
      } else if(k_learningTypeRegression == m_learningTypeOrCountTargetClasses) {
         m_pEbmTraining = InitializeTrainingRegression(randomSeed, m_features.size(), 0 == m_features.size() ? nullptr : &m_features[0], m_featureCombinations.size(), 0 == m_featureCombinations.size() ? nullptr : &m_featureCombinations[0], 0 == m_featureCombinationIndexes.size() ? nullptr : &m_featureCombinationIndexes[0], m_trainingRegressionTargets.size(), 0 == m_trainingRegressionTargets.size() ? nullptr : &m_trainingRegressionTargets[0], 0 == m_trainingBinnedData.size() ? nullptr : &m_trainingBinnedData[0], m_bNullTrainingPredictionScores ? nullptr : &m_trainingPredictionScores[0], m_validationRegressionTargets.size(), 0 == m_validationRegressionTargets.size() ? nullptr : &m_validationRegressionTargets[0], 0 == m_validationBinnedData.size() ? nullptr : &m_validationBinnedData[0], m_bNullValidationPredictionScores ? nullptr : &m_validationPredictionScores[0], countInnerBags);
      } else {
         exit(1);
      }

      if(nullptr == m_pEbmTraining) {
         exit(1);
      }
      m_stage = Stage::InitializedTraining;
   }

   FractionalDataType Train(const IntegerDataType indexFeatureCombination, const std::vector<FractionalDataType> trainingWeights = {}, const std::vector<FractionalDataType> validationWeights = {}, const FractionalDataType learningRate = k_learningRateDefault, const IntegerDataType countTreeSplitsMax = k_countTreeSplitsMaxDefault, const IntegerDataType countInstancesRequiredForParentSplitMin = k_countInstancesRequiredForParentSplitMinDefault) {
      if(Stage::InitializedTraining != m_stage) {
         exit(1);
      }
      if(indexFeatureCombination < IntegerDataType { 0 }) {
         exit(1);
      }
      if(m_featureCombinations.size() <= static_cast<size_t>(indexFeatureCombination)) {
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
      if(countInstancesRequiredForParentSplitMin < FractionalDataType { 0 }) {
         exit(1);
      }

      FractionalDataType validationMetricReturn = FractionalDataType { 0 };
      const IntegerDataType ret = TrainingStep(m_pEbmTraining, indexFeatureCombination, learningRate, countTreeSplitsMax, countInstancesRequiredForParentSplitMin, 0 == trainingWeights.size() ? nullptr : &trainingWeights[0], 0 == validationWeights.size() ? nullptr : &validationWeights[0], &validationMetricReturn);
      if(0 != ret) {
         exit(1);
      }
      return validationMetricReturn;
   }

   // TODO : change this so that we first call GetCurrentModelExpanded OR GetBestModelExpanded, which will return a tensor expanded as needed THEN  we call an indexing function if desired
   FractionalDataType GetCurrentModelPredictorScore(const size_t iFeatureCombination, const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures, const size_t iTargetClassOrZero) const {
      if(Stage::InitializedTraining != m_stage) {
         exit(1);
      }
      if(m_featureCombinations.size() <= iFeatureCombination) {
         exit(1);
      }
      FractionalDataType * pModelFeatureCombination = GetCurrentModelFeatureCombination(m_pEbmTraining, iFeatureCombination);
      FractionalDataType predictorScore = GetPredictorScore(iFeatureCombination, pModelFeatureCombination, perDimensionIndexArrayForBinnedFeatures, iTargetClassOrZero);
      return predictorScore;
   }

   FractionalDataType GetBestModelPredictorScore(const size_t iFeatureCombination, const std::vector<size_t> indexes, const size_t iScore) const {
      if(Stage::InitializedTraining != m_stage) {
         exit(1);
      }
      if(m_featureCombinations.size() <= iFeatureCombination) {
         exit(1);
      }
      FractionalDataType * pModelFeatureCombination = GetBestModelFeatureCombination(m_pEbmTraining, iFeatureCombination);
      FractionalDataType predictorScore = GetPredictorScore(iFeatureCombination, pModelFeatureCombination, indexes, iScore);
      return predictorScore;
   }

   const FractionalDataType * GetCurrentModelFeatureCombinationRaw(const size_t iFeatureCombination) const {
      if(Stage::InitializedTraining != m_stage) {
         exit(1);
      }
      if(m_featureCombinations.size() <= iFeatureCombination) {
         exit(1);
      }
      FractionalDataType * pModel = GetCurrentModelFeatureCombination(m_pEbmTraining, iFeatureCombination);
      return pModel;
   }

   const FractionalDataType * GetBestModelFeatureCombinationRaw(const size_t iFeatureCombination) const {
      if(Stage::InitializedTraining != m_stage) {
         exit(1);
      }
      if(m_featureCombinations.size() <= iFeatureCombination) {
         exit(1);
      }
      FractionalDataType * pModel = GetBestModelFeatureCombination(m_pEbmTraining, iFeatureCombination);
      return pModel;
   }

   void AddInteractionInstances(const std::vector<RegressionInstance> instances) {
      if(Stage::FeaturesAdded != m_stage) {
         exit(1);
      }
      if(k_learningTypeRegression != m_learningTypeOrCountTargetClasses) {
         exit(1);
      }
      const size_t cCases = instances.size();
      if(0 != cCases) {
         const size_t cFeatures = m_features.size();
         const bool bNullPredictionScores = instances[0].m_bNullPredictionScores;
         m_bNullInteractionPredictionScores = bNullPredictionScores;

         for(const RegressionInstance oneCase : instances) {
            if(cFeatures != oneCase.m_binnedDataPerFeatureArray.size()) {
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
               const FractionalDataType score = oneCase.m_priorPredictorPrediction;
               if(std::isnan(score)) {
                  exit(1);
               }
               if(std::isinf(score)) {
                  exit(1);
               }
               m_interactionPredictionScores.push_back(score);
            }
         }
         for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
            const EbmCoreFeature feature = m_features[iFeature];
            for(size_t iCase = 0; iCase < cCases; ++iCase) {
               const IntegerDataType data = instances[iCase].m_binnedDataPerFeatureArray[iFeature];
               if(data < 0) {
                  exit(1);
               }
               if(feature.countBins <= data) {
                  exit(1);
               }
               m_interactionBinnedData.push_back(data);
            }
         }
      }
      m_stage = Stage::InteractionAdded;
   }

   void AddInteractionInstances(const std::vector<ClassificationInstance> instances) {
      if(Stage::FeaturesAdded != m_stage) {
         exit(1);
      }
      if(!IsClassification(m_learningTypeOrCountTargetClasses)) {
         exit(1);
      }
      const size_t cCases = instances.size();
      if(0 != cCases) {
         const size_t cFeatures = m_features.size();
         const bool bNullPredictionScores = instances[0].m_bNullPredictionScores;
         m_bNullInteractionPredictionScores = bNullPredictionScores;

         for(const ClassificationInstance oneCase : instances) {
            if(cFeatures != oneCase.m_binnedDataPerFeatureArray.size()) {
               exit(1);
            }
            if(bNullPredictionScores != oneCase.m_bNullPredictionScores) {
               exit(1);
            }
            const IntegerDataType target = oneCase.m_target;
            if(target < 0) {
               exit(1);
            }
            if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) <= static_cast<size_t>(target)) {
               exit(1);
            }
            m_interactionClassificationTargets.push_back(target);
            if(!bNullPredictionScores) {
               if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) != oneCase.m_priorPredictorPerClassLogits.size()) {
                  exit(1);
               }
               ptrdiff_t iLogit = 0;
               for(const FractionalDataType oneLogit : oneCase.m_priorPredictorPerClassLogits) {
                  if(std::isnan(oneLogit)) {
                     exit(1);
                  }
                  if(std::isinf(oneLogit)) {
                     exit(1);
                  }
                  if(2 == m_learningTypeOrCountTargetClasses) {
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
                           m_interactionPredictionScores.push_back(oneLogit - oneCase.m_priorPredictorPerClassLogits[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_interactionPredictionScores.push_back(oneLogit - oneCase.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
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
                        m_interactionPredictionScores.push_back(oneLogit - oneCase.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                     }
#endif // REDUCE_MULTICLASS_LOGITS
                  }
                  ++iLogit;
               }
            }
         }
         for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
            const EbmCoreFeature feature = m_features[iFeature];
            for(size_t iCase = 0; iCase < cCases; ++iCase) {
               const IntegerDataType data = instances[iCase].m_binnedDataPerFeatureArray[iFeature];
               if(data < 0) {
                  exit(1);
               }
               if(feature.countBins <= data) {
                  exit(1);
               }
               m_interactionBinnedData.push_back(data);
            }
         }
      }
      m_stage = Stage::InteractionAdded;
   }

   void InitializeInteraction() {
      if(Stage::InteractionAdded != m_stage) {
         exit(1);
      }

      if(IsClassification(m_learningTypeOrCountTargetClasses)) {
         m_pEbmInteraction = InitializeInteractionClassification(m_features.size(), 0 == m_features.size() ? nullptr : &m_features[0], m_learningTypeOrCountTargetClasses, m_interactionClassificationTargets.size(), 0 == m_interactionClassificationTargets.size() ? nullptr : &m_interactionClassificationTargets[0], 0 == m_interactionBinnedData.size() ? nullptr : &m_interactionBinnedData[0], m_bNullInteractionPredictionScores ? nullptr : &m_interactionPredictionScores[0]);
      } else if(k_learningTypeRegression == m_learningTypeOrCountTargetClasses) {
         m_pEbmInteraction = InitializeInteractionRegression(m_features.size(), 0 == m_features.size() ? nullptr : &m_features[0], m_interactionRegressionTargets.size(), 0 == m_interactionRegressionTargets.size() ? nullptr : &m_interactionRegressionTargets[0], 0 == m_interactionBinnedData.size() ? nullptr : &m_interactionBinnedData[0], m_bNullInteractionPredictionScores ? nullptr : &m_interactionPredictionScores[0]);
      } else {
         exit(1);
      }

      if(nullptr == m_pEbmInteraction) {
         exit(1);
      }
      m_stage = Stage::InitializedInteraction;
   }

   FractionalDataType InteractionScore(const std::vector<IntegerDataType> featuresInCombination) const {
      if(Stage::InitializedInteraction != m_stage) {
         exit(1);
      }
      for(const IntegerDataType oneFeatureIndex : featuresInCombination) {
         if(oneFeatureIndex < IntegerDataType { 0 }) {
            exit(1);
         }
         if(m_features.size() <= static_cast<size_t>(oneFeatureIndex)) {
            exit(1);
         }
      }

      FractionalDataType interactionScoreReturn = FractionalDataType { 0 };
      const IntegerDataType ret = GetInteractionScore(m_pEbmInteraction, featuresInCombination.size(), 0 == featuresInCombination.size() ? nullptr : &featuresInCombination[0], &interactionScoreReturn);
      if(0 != ret) {
         exit(1);
      }
      return interactionScoreReturn;
   }
};

TEST_CASE("null validationMetricReturn, training, regression") {
   EbmCoreFeatureCombination combinations[1];
   combinations->countFeaturesInCombination = 0;

   PEbmTraining pEbmTraining = InitializeTrainingRegression(randomSeed, 0, nullptr, 1, combinations, nullptr, 0, nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, 0);
   const IntegerDataType ret = TrainingStep(pEbmTraining, 0, k_learningRateDefault, k_countTreeSplitsMaxDefault, k_countInstancesRequiredForParentSplitMinDefault, nullptr, nullptr, nullptr);
   CHECK(0 == ret);
   FreeTraining(pEbmTraining);
}

TEST_CASE("null validationMetricReturn, training, binary") {
   EbmCoreFeatureCombination combinations[1];
   combinations->countFeaturesInCombination = 0;

   PEbmTraining pEbmTraining = InitializeTrainingClassification(randomSeed, 0, nullptr, 1, combinations, nullptr, 2, 0, nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, 0);
   const IntegerDataType ret = TrainingStep(pEbmTraining, 0, k_learningRateDefault, k_countTreeSplitsMaxDefault, k_countInstancesRequiredForParentSplitMinDefault, nullptr, nullptr, nullptr);
   CHECK(0 == ret);
   FreeTraining(pEbmTraining);
}

TEST_CASE("null validationMetricReturn, training, multiclass") {
   EbmCoreFeatureCombination combinations[1];
   combinations->countFeaturesInCombination = 0;

   PEbmTraining pEbmTraining = InitializeTrainingClassification(randomSeed, 0, nullptr, 1, combinations, nullptr, 3, 0, nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, 0);
   const IntegerDataType ret = TrainingStep(pEbmTraining, 0, k_learningRateDefault, k_countTreeSplitsMaxDefault, k_countInstancesRequiredForParentSplitMinDefault, nullptr, nullptr, nullptr);
   CHECK(0 == ret);
   FreeTraining(pEbmTraining);
}

TEST_CASE("null interactionScoreReturn, interaction, regression") {
   PEbmInteraction pEbmInteraction = InitializeInteractionRegression(0, nullptr, 0, nullptr, nullptr, nullptr);
   const IntegerDataType ret = GetInteractionScore(pEbmInteraction, 0, nullptr, nullptr);
   CHECK(0 == ret);
   FreeInteraction(pEbmInteraction);
}

TEST_CASE("null interactionScoreReturn, interaction, binary") {
   PEbmInteraction pEbmInteraction = InitializeInteractionClassification(0, nullptr, 2, 0, nullptr, nullptr, nullptr);
   const IntegerDataType ret = GetInteractionScore(pEbmInteraction, 0, nullptr, nullptr);
   CHECK(0 == ret);
   FreeInteraction(pEbmInteraction);
}

TEST_CASE("null interactionScoreReturn, interaction, multiclass") {
   PEbmInteraction pEbmInteraction = InitializeInteractionClassification(0, nullptr, 3, 0, nullptr, nullptr, nullptr);
   const IntegerDataType ret = GetInteractionScore(pEbmInteraction, 0, nullptr, nullptr);
   CHECK(0 == ret);
   FreeInteraction(pEbmInteraction);
}

TEST_CASE("zero learning rate, training, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ RegressionInstance(10, {}) });
   test.AddValidationInstances({ RegressionInstance(12, {}) });
   test.InitializeTraining();

   FractionalDataType validationMetric = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   FractionalDataType modelValue = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Train(iFeatureCombination, {}, {}, 0);
         CHECK_APPROX(validationMetric, 12);
         modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
         CHECK_APPROX(modelValue, 0);

         modelValue = test.GetBestModelPredictorScore(iFeatureCombination, {}, 0);
         CHECK_APPROX(modelValue, 0);
      }
   }
}

TEST_CASE("zero learning rate, training, binary") {
   TestApi test = TestApi(2);
   test.AddFeatures({});
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ ClassificationInstance(0, {}) });
   test.AddValidationInstances({ ClassificationInstance(0, {}) });
   test.InitializeTraining();

   FractionalDataType validationMetric = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   FractionalDataType modelValue = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Train(iFeatureCombination, {}, {}, 0);
         CHECK_APPROX(validationMetric, 0.69314718055994529);
         modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
         CHECK_APPROX(modelValue, 0);
         modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 1);
         CHECK_APPROX(modelValue, 0);

         modelValue = test.GetBestModelPredictorScore(iFeatureCombination, {}, 0);
         CHECK_APPROX(modelValue, 0);
         modelValue = test.GetBestModelPredictorScore(iFeatureCombination, {}, 1);
         CHECK_APPROX(modelValue, 0);
      }
   }
}

TEST_CASE("zero learning rate, training, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({});
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ ClassificationInstance(0, {}) });
   test.AddValidationInstances({ ClassificationInstance(0, {}) });
   test.InitializeTraining();

   FractionalDataType validationMetric = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   FractionalDataType modelValue = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Train(iFeatureCombination, {}, {}, 0);
         CHECK_APPROX(validationMetric, 1.0986122886681098);
         modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
         CHECK_APPROX(modelValue, 0);
         modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 1);
         CHECK_APPROX(modelValue, 0);
         modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 2);
         CHECK_APPROX(modelValue, 0);

         modelValue = test.GetBestModelPredictorScore(iFeatureCombination, {}, 0);
         CHECK_APPROX(modelValue, 0);
         modelValue = test.GetBestModelPredictorScore(iFeatureCombination, {}, 1);
         CHECK_APPROX(modelValue, 0);
         modelValue = test.GetBestModelPredictorScore(iFeatureCombination, {}, 2);
         CHECK_APPROX(modelValue, 0);
      }
   }
}

TEST_CASE("negative learning rate, training, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ RegressionInstance(10, {}) });
   test.AddValidationInstances({ RegressionInstance(12, {}) });
   test.InitializeTraining();

   FractionalDataType validationMetric = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   FractionalDataType modelValue = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Train(iFeatureCombination, {}, {}, -k_learningRateDefault);
         if(0 == iFeatureCombination && 0 == iEpoch) {
            CHECK_APPROX(validationMetric, 12.100000000000000);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
            CHECK_APPROX(modelValue, -0.1000000000000000);
         }
         if(0 == iFeatureCombination && 1 == iEpoch) {
            CHECK_APPROX(validationMetric, 12.20100000000000);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
            CHECK_APPROX(modelValue, -0.2010000000000000);
         }
      }
   }
   CHECK_APPROX(validationMetric, 209593.55637813677);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 0);
   CHECK_APPROX(modelValue, -209581.55637813677);
}

TEST_CASE("negative learning rate, training, binary") {
   TestApi test = TestApi(2);
   test.AddFeatures({});
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ ClassificationInstance(0, {}) });
   test.AddValidationInstances({ ClassificationInstance(0, {}) });
   test.InitializeTraining();

   FractionalDataType validationMetric = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   FractionalDataType modelValue = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Train(iFeatureCombination, {}, {}, -k_learningRateDefault);
         if(0 == iFeatureCombination && 0 == iEpoch) {
            CHECK_APPROX(validationMetric, 0.70319717972663420);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
            CHECK_APPROX(modelValue, 0);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 1);
            CHECK_APPROX(modelValue, 0.020000000000000000);
         }
         if(0 == iFeatureCombination && 1 == iEpoch) {
            CHECK_APPROX(validationMetric, 0.71345019889199235);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
            CHECK_APPROX(modelValue, 0);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 1);
            CHECK_APPROX(modelValue, 0.040202013400267564);
         }
      }
   }
   CHECK(std::isinf(validationMetric));
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 0);
   CHECK_APPROX(modelValue, 0);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 1);
   CHECK_APPROX(modelValue, 16785686302.358746);
}

TEST_CASE("negative learning rate, training, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({});
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ ClassificationInstance(0, {}) });
   test.AddValidationInstances({ ClassificationInstance(0, {}) });
   test.InitializeTraining();

   FractionalDataType validationMetric = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   FractionalDataType modelValue = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Train(iFeatureCombination, {}, {}, -k_learningRateDefault);
         if(0 == iFeatureCombination && 0 == iEpoch) {
            CHECK_APPROX(validationMetric, 1.1288361512023379);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
            CHECK_APPROX(modelValue, -0.03000000000000000);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 1);
            CHECK_APPROX(modelValue, 0.01500000000000000);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 2);
            CHECK_APPROX(modelValue, 0.01500000000000000);
         }
         if(0 == iFeatureCombination && 1 == iEpoch) {
            CHECK_APPROX(validationMetric, 1.1602122411839852);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
            CHECK_APPROX(modelValue, -0.060920557198174352);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 1);
            CHECK_APPROX(modelValue, 0.030112481019468545);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 2);
            CHECK_APPROX(modelValue, 0.030112481019468545);
         }
      }
   }
   CHECK(std::isinf(validationMetric));
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 0);
   CHECK_APPROX(modelValue, -10344932.919067673);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 1);
   CHECK_APPROX(modelValue, 19.907994122542746);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 2);
   CHECK_APPROX(modelValue, 19.907994122542746);
}

TEST_CASE("zero countInstancesRequiredForParentSplitMin, training, regression") {
   // TODO : move this into our tests that iterate many loops and compare output for no splitting.  AND also loop this 
   // TODO : add classification binary and multiclass versions of this

   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances({
      RegressionInstance(10, { 0 }),
      RegressionInstance(10, { 1 }),
      });
   test.AddValidationInstances({ RegressionInstance(12, { 1 }) });
   test.InitializeTraining();

   FractionalDataType validationMetric = test.Train(0, {}, {}, k_learningRateDefault, k_countTreeSplitsMaxDefault, 0);
   CHECK_APPROX(validationMetric, 11.900000000000000);
   FractionalDataType modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
   CHECK_APPROX(modelValue, 0.1000000000000000);
}

TEST_CASE("zero countTreeSplitsMax, training, regression") {
   // TODO : move this into our tests that iterate many loops and compare output for no splitting.  AND also loop this 
   // TODO : add classification binary and multiclass versions of this

   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances({ 
      RegressionInstance(10, { 0 }),
      RegressionInstance(10, { 1 }),
      });
   test.AddValidationInstances({ RegressionInstance(12, { 1 }) });
   test.InitializeTraining();

   FractionalDataType validationMetric = test.Train(0, {}, {}, k_learningRateDefault, 0);
   CHECK_APPROX(validationMetric, 11.900000000000000);
   FractionalDataType modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
   CHECK_APPROX(modelValue, 0.1000000000000000);
}


//TEST_CASE("infinite target training set, training, regression") {
//   TestApi test = TestApi(k_learningTypeRegression);
//   test.AddFeatures({ Feature(2) });
//   test.AddFeatureCombinations({ { 0 } });
//   test.AddTrainingCases({ RegressionCase(FractionalDataType { std::numeric_limits<FractionalDataType>::infinity() }, { 1 }) });
//   test.AddValidationCases({ RegressionCase(12, { 1 }) });
//   test.InitializeTraining();
//
//   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
//      FractionalDataType validationMetric = test.Train(0);
//      CHECK_APPROX(validationMetric, 12);
//      FractionalDataType modelValue = test.GetCurrentModelValue(0, { 0 }, 0);
//      CHECK_APPROX(modelValue, 0);
//   }
//}



TEST_CASE("Zero training instances, training, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances(std::vector<RegressionInstance> {});
   test.AddValidationInstances({ RegressionInstance(12, { 1 }) });
   test.InitializeTraining();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FractionalDataType validationMetric = test.Train(0);
      CHECK_APPROX(validationMetric, 12);
      FractionalDataType modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
      CHECK_APPROX(modelValue, 0);
   }
}

TEST_CASE("Zero training instances, training, binary") {
   TestApi test = TestApi(2);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances(std::vector<ClassificationInstance> {});
   test.AddValidationInstances({ ClassificationInstance(0, { 1 }) });
   test.InitializeTraining();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FractionalDataType validationMetric = test.Train(0);
      CHECK_APPROX(validationMetric, 0.69314718055994529);
      FractionalDataType modelValue;
      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
      CHECK_APPROX(modelValue, 0);
      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 1);
      CHECK_APPROX(modelValue, 0);
   }
}

TEST_CASE("Zero training instances, training, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances(std::vector<ClassificationInstance> {});
   test.AddValidationInstances({ ClassificationInstance(0, { 1 }) });
   test.InitializeTraining();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FractionalDataType validationMetric = test.Train(0);
      CHECK_APPROX(validationMetric, 1.0986122886681098);
      FractionalDataType modelValue;
      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
      CHECK_APPROX(modelValue, 0);
      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 1);
      CHECK_APPROX(modelValue, 0);
      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 2);
      CHECK_APPROX(modelValue, 0);
   }
}

TEST_CASE("Zero validation instances, training, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances({ RegressionInstance(10, { 1 }) });
   test.AddValidationInstances(std::vector<RegressionInstance> {});
   test.InitializeTraining();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FractionalDataType validationMetric = test.Train(0);
      CHECK(0 == validationMetric);
      // the current model will continue to update, even though we have no way of evaluating it
      FractionalDataType modelValue;
      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
      if(0 == iEpoch) {
         CHECK_APPROX(modelValue, 0.1000000000000000);
      }
      if(1 == iEpoch) {
         CHECK_APPROX(modelValue, 0.1990000000000000);
      }
      // the best model doesn't update since we don't have any basis to validate any changes
      modelValue = test.GetBestModelPredictorScore(0, { 0 }, 0);
      CHECK_APPROX(modelValue, 0);
   }
}

TEST_CASE("Zero validation instances, training, binary") {
   TestApi test = TestApi(2);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances({ ClassificationInstance(0, { 1 }) });
   test.AddValidationInstances(std::vector<ClassificationInstance> {});
   test.InitializeTraining();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FractionalDataType validationMetric = test.Train(0);
      CHECK(0 == validationMetric);
      // the current model will continue to update, even though we have no way of evaluating it
      FractionalDataType modelValue;
      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
      CHECK_APPROX(modelValue, 0);
      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 1);
      if(0 == iEpoch) {
         CHECK_APPROX(modelValue, -0.020000000000000000);
      }
      if(1 == iEpoch) {
         CHECK_APPROX(modelValue, -0.039801986733067563);
      }
      // the best model doesn't update since we don't have any basis to validate any changes
      modelValue = test.GetBestModelPredictorScore(0, { 0 }, 0);
      CHECK_APPROX(modelValue, 0);
      modelValue = test.GetBestModelPredictorScore(0, { 0 }, 1);
      CHECK_APPROX(modelValue, 0);
   }
}

TEST_CASE("Zero validation instances, training, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances({ ClassificationInstance(0, { 1 }) });
   test.AddValidationInstances(std::vector<ClassificationInstance> {});
   test.InitializeTraining();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FractionalDataType validationMetric = test.Train(0);
      CHECK(0 == validationMetric);
      // the current model will continue to update, even though we have no way of evaluating it
      FractionalDataType modelValue;
      if(0 == iEpoch) {
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
         CHECK_APPROX(modelValue, 0.03000000000000000);
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 1);
         CHECK_APPROX(modelValue, -0.01500000000000000);
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 2);
         CHECK_APPROX(modelValue, -0.01500000000000000);
      }
      if(1 == iEpoch) {
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
         CHECK_APPROX(modelValue, 0.059119949636662006);
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 1);
         CHECK_APPROX(modelValue, -0.029887518980531450);
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 2);
         CHECK_APPROX(modelValue, -0.029887518980531450);
      }
      // the best model doesn't update since we don't have any basis to validate any changes
      modelValue = test.GetBestModelPredictorScore(0, { 0 }, 0);
      CHECK_APPROX(modelValue, 0);
      modelValue = test.GetBestModelPredictorScore(0, { 0 }, 1);
      CHECK_APPROX(modelValue, 0);
      modelValue = test.GetBestModelPredictorScore(0, { 0 }, 2);
      CHECK_APPROX(modelValue, 0);
   }
}

TEST_CASE("Zero interaction instances, interaction, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionInstances(std::vector<RegressionInstance> {});
   test.InitializeInteraction();

   FractionalDataType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("Zero interaction instances, interaction, binary") {
   TestApi test = TestApi(2);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionInstances(std::vector<ClassificationInstance> {});
   test.InitializeInteraction();

   FractionalDataType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("Zero interaction instances, interaction, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionInstances(std::vector<ClassificationInstance> {});
   test.InitializeInteraction();

   FractionalDataType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("features with 0 states, training") {
   // for there to be zero states, there can't be an training data or testing data since then those would be required to have a value for the state
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(0) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances(std::vector<RegressionInstance> {});
   test.AddValidationInstances(std::vector<RegressionInstance> {});
   test.InitializeTraining();

   FractionalDataType validationMetric = test.Train(0);
   CHECK(0 == validationMetric);

   // we're not sure what we'd get back since we aren't allowed to access it, so don't do anything with the return value.  We just want to make sure calling to get the models doesn't crash
   test.GetCurrentModelFeatureCombinationRaw(0);
   test.GetBestModelFeatureCombinationRaw(0);
}

TEST_CASE("features with 0 states, interaction") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(0) });
   test.AddInteractionInstances(std::vector<RegressionInstance> {});
   test.InitializeInteraction();

   FractionalDataType validationMetric = test.InteractionScore({ 0 });
   CHECK(0 == validationMetric);
}

TEST_CASE("classification with 0 possible target states, training") {
   // for there to be zero states, there can't be an training data or testing data since then those would be required to have a value for the state
   TestApi test = TestApi(0);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances(std::vector<ClassificationInstance> {});
   test.AddValidationInstances(std::vector<ClassificationInstance> {});
   test.InitializeTraining();

   CHECK(nullptr == test.GetCurrentModelFeatureCombinationRaw(0));
   CHECK(nullptr == test.GetBestModelFeatureCombinationRaw(0));

   FractionalDataType validationMetric = test.Train(0);
   CHECK(0 == validationMetric);

   CHECK(nullptr == test.GetCurrentModelFeatureCombinationRaw(0));
   CHECK(nullptr == test.GetBestModelFeatureCombinationRaw(0));
}

TEST_CASE("classification with 0 possible target states, interaction") {
   TestApi test = TestApi(0);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionInstances(std::vector<ClassificationInstance> {});
   test.InitializeInteraction();

   FractionalDataType validationMetric = test.InteractionScore({ 0 });
   CHECK(0 == validationMetric);
}

TEST_CASE("classification with 1 possible target, training") {
   TestApi test = TestApi(1);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances({ ClassificationInstance(0, { 1 }) });
   test.AddValidationInstances({ ClassificationInstance(0, { 1 }) });
   test.InitializeTraining();

   CHECK(nullptr == test.GetCurrentModelFeatureCombinationRaw(0));
   CHECK(nullptr == test.GetBestModelFeatureCombinationRaw(0));

   FractionalDataType validationMetric = test.Train(0);
   CHECK(0 == validationMetric);

   CHECK(nullptr == test.GetCurrentModelFeatureCombinationRaw(0));
   CHECK(nullptr == test.GetBestModelFeatureCombinationRaw(0));
}

TEST_CASE("classification with 1 possible target, interaction") {
   TestApi test = TestApi(1);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionInstances({ ClassificationInstance(0, { 1 }) });
   test.InitializeInteraction();

   FractionalDataType validationMetric = test.InteractionScore({ 0 });
   CHECK(0 == validationMetric);
}

TEST_CASE("features with 1 state in various positions, training") {
   TestApi test0 = TestApi(k_learningTypeRegression);
   test0.AddFeatures({ 
      FeatureTest(1),
      FeatureTest(2),
      FeatureTest(2)
      });
   test0.AddFeatureCombinations({ { 0 }, { 1 }, { 2 } });
   test0.AddTrainingInstances({ RegressionInstance(10, { 0, 1, 1 }) });
   test0.AddValidationInstances({ RegressionInstance(12, { 0, 1, 1 }) });
   test0.InitializeTraining();

   TestApi test1 = TestApi(k_learningTypeRegression);
   test1.AddFeatures({
      FeatureTest(2),
      FeatureTest(1),
      FeatureTest(2)
      });
   test1.AddFeatureCombinations({ { 0 }, { 1 }, { 2 } });
   test1.AddTrainingInstances({ RegressionInstance(10, { 1, 0, 1 }) });
   test1.AddValidationInstances({ RegressionInstance(12, { 1, 0, 1 }) });
   test1.InitializeTraining();

   TestApi test2 = TestApi(k_learningTypeRegression);
   test2.AddFeatures({
      FeatureTest(2),
      FeatureTest(2),
      FeatureTest(1)
      });
   test2.AddFeatureCombinations({ { 0 }, { 1 }, { 2 } });
   test2.AddTrainingInstances({ RegressionInstance(10, { 1, 1, 0 }) });
   test2.AddValidationInstances({ RegressionInstance(12, { 1, 1, 0 }) });
   test2.InitializeTraining();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FractionalDataType validationMetric00 = test0.Train(0);
      FractionalDataType validationMetric10 = test1.Train(1);
      CHECK_APPROX(validationMetric00, validationMetric10);
      FractionalDataType validationMetric20 = test2.Train(2);
      CHECK_APPROX(validationMetric00, validationMetric20);

      FractionalDataType validationMetric01 = test0.Train(1);
      FractionalDataType validationMetric11 = test1.Train(2);
      CHECK_APPROX(validationMetric01, validationMetric11);
      FractionalDataType validationMetric21 = test2.Train(0);
      CHECK_APPROX(validationMetric01, validationMetric21);

      FractionalDataType validationMetric02 = test0.Train(2);
      FractionalDataType validationMetric12 = test1.Train(0);
      CHECK_APPROX(validationMetric02, validationMetric12);
      FractionalDataType validationMetric22 = test2.Train(1);
      CHECK_APPROX(validationMetric02, validationMetric22);

      FractionalDataType modelValue000 = test0.GetCurrentModelPredictorScore(0, { 0 }, 0);
      FractionalDataType modelValue010 = test0.GetCurrentModelPredictorScore(1, { 0 }, 0);
      FractionalDataType modelValue011 = test0.GetCurrentModelPredictorScore(1, { 1 }, 0);
      FractionalDataType modelValue020 = test0.GetCurrentModelPredictorScore(2, { 0 }, 0);
      FractionalDataType modelValue021 = test0.GetCurrentModelPredictorScore(2, { 1 }, 0);

      FractionalDataType modelValue110 = test1.GetCurrentModelPredictorScore(1, { 0 }, 0);
      FractionalDataType modelValue120 = test1.GetCurrentModelPredictorScore(2, { 0 }, 0);
      FractionalDataType modelValue121 = test1.GetCurrentModelPredictorScore(2, { 1 }, 0);
      FractionalDataType modelValue100 = test1.GetCurrentModelPredictorScore(0, { 0 }, 0);
      FractionalDataType modelValue101 = test1.GetCurrentModelPredictorScore(0, { 1 }, 0);
      CHECK_APPROX(modelValue110, modelValue000);
      CHECK_APPROX(modelValue120, modelValue010);
      CHECK_APPROX(modelValue121, modelValue011);
      CHECK_APPROX(modelValue100, modelValue020);
      CHECK_APPROX(modelValue101, modelValue021);

      FractionalDataType modelValue220 = test2.GetCurrentModelPredictorScore(2, { 0 }, 0);
      FractionalDataType modelValue200 = test2.GetCurrentModelPredictorScore(0, { 0 }, 0);
      FractionalDataType modelValue201 = test2.GetCurrentModelPredictorScore(0, { 1 }, 0);
      FractionalDataType modelValue210 = test2.GetCurrentModelPredictorScore(1, { 0 }, 0);
      FractionalDataType modelValue211 = test2.GetCurrentModelPredictorScore(1, { 1 }, 0);
      CHECK_APPROX(modelValue220, modelValue000);
      CHECK_APPROX(modelValue200, modelValue010);
      CHECK_APPROX(modelValue201, modelValue011);
      CHECK_APPROX(modelValue210, modelValue020);
      CHECK_APPROX(modelValue211, modelValue021);
   }
}

TEST_CASE("zero FeatureCombinations, training, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddFeatureCombinations({});
   test.AddTrainingInstances({ RegressionInstance(10, {}) });
   test.AddValidationInstances({ RegressionInstance(12, {}) });
   test.InitializeTraining();

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just crash/assert if there was a problem
   // training isn't legal since we'd need to specify an featureCombination index
}

TEST_CASE("zero FeatureCombinations, training, binary") {
   TestApi test = TestApi(2);
   test.AddFeatures({});
   test.AddFeatureCombinations({});
   test.AddTrainingInstances({ ClassificationInstance(1, {}) });
   test.AddValidationInstances({ ClassificationInstance(1, {}) });
   test.InitializeTraining();

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just crash/assert if there was a problem
   // training isn't legal since we'd need to specify an featureCombination index
}

TEST_CASE("zero FeatureCombinations, training, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({});
   test.AddFeatureCombinations({});
   test.AddTrainingInstances({ ClassificationInstance(2, {}) });
   test.AddValidationInstances({ ClassificationInstance(2, {}) });
   test.InitializeTraining();

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just crash/assert if there was a problem
   // training isn't legal since we'd need to specify an featureCombination index
}

TEST_CASE("FeatureCombination with zero features, training, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ RegressionInstance(10, {}) });
   test.AddValidationInstances({ RegressionInstance(12, {}) });
   test.InitializeTraining();

   FractionalDataType validationMetric = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   FractionalDataType modelValue = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Train(iFeatureCombination);
         if(0 == iFeatureCombination && 0 == iEpoch) {
            CHECK_APPROX(validationMetric, 11.900000000000000);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
            CHECK_APPROX(modelValue, 0.1000000000000000);
         }
         if(0 == iFeatureCombination && 1 == iEpoch) {
            CHECK_APPROX(validationMetric, 11.801000000000000);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
            CHECK_APPROX(modelValue, 0.1990000000000000);
         }
      }
   }
   CHECK_APPROX(validationMetric, 2.0004317124741098);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 0);
   CHECK_APPROX(modelValue, 9.9995682875258822);
}

TEST_CASE("FeatureCombination with zero features, training, binary") {
   TestApi test = TestApi(2);
   test.AddFeatures({});
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ ClassificationInstance(0, {}) });
   test.AddValidationInstances({ ClassificationInstance(0, {}) });
   test.InitializeTraining();

   FractionalDataType validationMetric = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   FractionalDataType modelValue = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Train(iFeatureCombination);
         if(0 == iFeatureCombination && 0 == iEpoch) {
            CHECK_APPROX(validationMetric, 0.68319717972663419);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
            CHECK_APPROX(modelValue, 0);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 1);
            CHECK_APPROX(modelValue, -0.020000000000000000);
         }
         if(0 == iFeatureCombination && 1 == iEpoch) {
            CHECK_APPROX(validationMetric, 0.67344419889200957);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
            CHECK_APPROX(modelValue, 0);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 1);
            CHECK_APPROX(modelValue, -0.039801986733067563);
         }
      }
   }
   CHECK_APPROX(validationMetric, 2.2621439908125974e-05);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 0);
   CHECK_APPROX(modelValue, 0);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 1);
   CHECK_APPROX(modelValue, -10.696601122148364);
}

TEST_CASE("FeatureCombination with zero features, training, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ });
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ ClassificationInstance(0, {}) });
   test.AddValidationInstances({ ClassificationInstance(0, {}) });
   test.InitializeTraining();

   FractionalDataType validationMetric = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   FractionalDataType modelValue = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Train(iFeatureCombination);
         if(0 == iFeatureCombination && 0 == iEpoch) {
            CHECK_APPROX(validationMetric, 1.0688384008227103);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
            CHECK_APPROX(modelValue, 0.03000000000000000);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 1);
            CHECK_APPROX(modelValue, -0.01500000000000000);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 2);
            CHECK_APPROX(modelValue, -0.01500000000000000);
         }
         if(0 == iFeatureCombination && 1 == iEpoch) {
            CHECK_APPROX(validationMetric, 1.0401627411809615);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
            CHECK_APPROX(modelValue, 0.059119949636662006);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 1);
            CHECK_APPROX(modelValue, -0.029887518980531450);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 2);
            CHECK_APPROX(modelValue, -0.029887518980531450);
         }
      }
   }
   CHECK_APPROX(validationMetric, 1.7171897252232722e-09);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 0);
   CHECK_APPROX(modelValue, 10.643234965479628);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 1);
   CHECK_APPROX(modelValue, -10.232489007525166);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 2);
   CHECK_APPROX(modelValue, -10.232489007525166);
}

TEST_CASE("FeatureCombination with zero features, interaction, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddInteractionInstances({ RegressionInstance(10, {}) });
   test.InitializeInteraction();
   FractionalDataType metricReturn = test.InteractionScore({});
   CHECK(0 == metricReturn);
}

TEST_CASE("FeatureCombination with zero features, interaction, binary") {
   TestApi test = TestApi(2);
   test.AddFeatures({});
   test.AddInteractionInstances({ ClassificationInstance(0, {}) });
   test.InitializeInteraction();
   FractionalDataType metricReturn = test.InteractionScore({});
   CHECK(0 == metricReturn);
}

TEST_CASE("FeatureCombination with zero features, interaction, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({});
   test.AddInteractionInstances({ ClassificationInstance(0, {}) });
   test.InitializeInteraction();
   FractionalDataType metricReturn = test.InteractionScore({});
   CHECK(0 == metricReturn);
}

TEST_CASE("FeatureCombination with one feature with one or two states is the exact same as zero FeatureCombinations, training, regression") {
   TestApi testZeroFeaturesInCombination = TestApi(k_learningTypeRegression);
   testZeroFeaturesInCombination.AddFeatures({});
   testZeroFeaturesInCombination.AddFeatureCombinations({ {} });
   testZeroFeaturesInCombination.AddTrainingInstances({ RegressionInstance(10, {}) });
   testZeroFeaturesInCombination.AddValidationInstances({ RegressionInstance(12, {}) });
   testZeroFeaturesInCombination.InitializeTraining();

   TestApi testOneState = TestApi(k_learningTypeRegression);
   testOneState.AddFeatures({ FeatureTest(1) });
   testOneState.AddFeatureCombinations({ { 0 } });
   testOneState.AddTrainingInstances({ RegressionInstance(10, { 0 }) });
   testOneState.AddValidationInstances({ RegressionInstance(12, { 0 }) });
   testOneState.InitializeTraining();

   TestApi testTwoStates = TestApi(k_learningTypeRegression);
   testTwoStates.AddFeatures({ FeatureTest(2) });
   testTwoStates.AddFeatureCombinations({ { 0 } });
   testTwoStates.AddTrainingInstances({ RegressionInstance(10, { 1 }) });
   testTwoStates.AddValidationInstances({ RegressionInstance(12, { 1 }) });
   testTwoStates.InitializeTraining();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroFeaturesInCombination.GetFeatureCombinationsCount() == testOneState.GetFeatureCombinationsCount());
      assert(testZeroFeaturesInCombination.GetFeatureCombinationsCount() == testTwoStates.GetFeatureCombinationsCount());
      for(size_t iFeatureCombination = 0; iFeatureCombination < testZeroFeaturesInCombination.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         FractionalDataType validationMetricZeroFeaturesInCombination = testZeroFeaturesInCombination.Train(iFeatureCombination);
         FractionalDataType validationMetricOneState = testOneState.Train(iFeatureCombination);
         CHECK_APPROX(validationMetricZeroFeaturesInCombination, validationMetricOneState);
         FractionalDataType validationMetricTwoStates = testTwoStates.Train(iFeatureCombination);
         CHECK_APPROX(validationMetricZeroFeaturesInCombination, validationMetricTwoStates);

         FractionalDataType modelValueZeroFeaturesInCombination = testZeroFeaturesInCombination.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
         FractionalDataType modelValueOneState = testOneState.GetCurrentModelPredictorScore(iFeatureCombination, { 0 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInCombination, modelValueOneState);
         FractionalDataType modelValueTwoStates = testTwoStates.GetCurrentModelPredictorScore(iFeatureCombination, { 1 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInCombination, modelValueTwoStates);
      }
   }
}

TEST_CASE("FeatureCombination with one feature with one or two states is the exact same as zero FeatureCombinations, training, binary") {
   TestApi testZeroFeaturesInCombination = TestApi(2);
   testZeroFeaturesInCombination.AddFeatures({});
   testZeroFeaturesInCombination.AddFeatureCombinations({ {} });
   testZeroFeaturesInCombination.AddTrainingInstances({ ClassificationInstance(0, {}) });
   testZeroFeaturesInCombination.AddValidationInstances({ ClassificationInstance(0, {}) });
   testZeroFeaturesInCombination.InitializeTraining();

   TestApi testOneState = TestApi(2);
   testOneState.AddFeatures({ FeatureTest(1) });
   testOneState.AddFeatureCombinations({ { 0 } });
   testOneState.AddTrainingInstances({ ClassificationInstance(0, { 0 }) });
   testOneState.AddValidationInstances({ ClassificationInstance(0, { 0 }) });
   testOneState.InitializeTraining();

   TestApi testTwoStates = TestApi(2);
   testTwoStates.AddFeatures({ FeatureTest(2) });
   testTwoStates.AddFeatureCombinations({ { 0 } });
   testTwoStates.AddTrainingInstances({ ClassificationInstance(0, { 1 }) });
   testTwoStates.AddValidationInstances({ ClassificationInstance(0, { 1 }) });
   testTwoStates.InitializeTraining();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroFeaturesInCombination.GetFeatureCombinationsCount() == testOneState.GetFeatureCombinationsCount());
      assert(testZeroFeaturesInCombination.GetFeatureCombinationsCount() == testTwoStates.GetFeatureCombinationsCount());
      for(size_t iFeatureCombination = 0; iFeatureCombination < testZeroFeaturesInCombination.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         FractionalDataType validationMetricZeroFeaturesInCombination = testZeroFeaturesInCombination.Train(iFeatureCombination);
         FractionalDataType validationMetricOneState = testOneState.Train(iFeatureCombination);
         CHECK_APPROX(validationMetricZeroFeaturesInCombination, validationMetricOneState);
         FractionalDataType validationMetricTwoStates = testTwoStates.Train(iFeatureCombination);
         CHECK_APPROX(validationMetricZeroFeaturesInCombination, validationMetricTwoStates);

         FractionalDataType modelValueZeroFeaturesInCombination0 = testZeroFeaturesInCombination.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
         FractionalDataType modelValueOneState0 = testOneState.GetCurrentModelPredictorScore(iFeatureCombination, { 0 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInCombination0, modelValueOneState0);
         FractionalDataType modelValueTwoStates0 = testTwoStates.GetCurrentModelPredictorScore(iFeatureCombination, { 1 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInCombination0, modelValueTwoStates0);

         FractionalDataType modelValueZeroFeaturesInCombination1 = testZeroFeaturesInCombination.GetCurrentModelPredictorScore(iFeatureCombination, {}, 1);
         FractionalDataType modelValueOneState1 = testOneState.GetCurrentModelPredictorScore(iFeatureCombination, { 0 }, 1);
         CHECK_APPROX(modelValueZeroFeaturesInCombination1, modelValueOneState1);
         FractionalDataType modelValueTwoStates1 = testTwoStates.GetCurrentModelPredictorScore(iFeatureCombination, { 1 }, 1);
         CHECK_APPROX(modelValueZeroFeaturesInCombination1, modelValueTwoStates1);
      }
   }
}

TEST_CASE("FeatureCombination with one feature with one or two states is the exact same as zero FeatureCombinations, training, multiclass") {
   TestApi testZeroFeaturesInCombination = TestApi(3);
   testZeroFeaturesInCombination.AddFeatures({});
   testZeroFeaturesInCombination.AddFeatureCombinations({ {} });
   testZeroFeaturesInCombination.AddTrainingInstances({ ClassificationInstance(0, {}) });
   testZeroFeaturesInCombination.AddValidationInstances({ ClassificationInstance(0, {}) });
   testZeroFeaturesInCombination.InitializeTraining();

   TestApi testOneState = TestApi(3);
   testOneState.AddFeatures({ FeatureTest(1) });
   testOneState.AddFeatureCombinations({ { 0 } });
   testOneState.AddTrainingInstances({ ClassificationInstance(0, { 0 }) });
   testOneState.AddValidationInstances({ ClassificationInstance(0, { 0 }) });
   testOneState.InitializeTraining();

   TestApi testTwoStates = TestApi(3);
   testTwoStates.AddFeatures({ FeatureTest(2) });
   testTwoStates.AddFeatureCombinations({ { 0 } });
   testTwoStates.AddTrainingInstances({ ClassificationInstance(0, { 1 }) });
   testTwoStates.AddValidationInstances({ ClassificationInstance(0, { 1 }) });
   testTwoStates.InitializeTraining();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroFeaturesInCombination.GetFeatureCombinationsCount() == testOneState.GetFeatureCombinationsCount());
      assert(testZeroFeaturesInCombination.GetFeatureCombinationsCount() == testTwoStates.GetFeatureCombinationsCount());
      for(size_t iFeatureCombination = 0; iFeatureCombination < testZeroFeaturesInCombination.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         FractionalDataType validationMetricZeroFeaturesInCombination = testZeroFeaturesInCombination.Train(iFeatureCombination);
         FractionalDataType validationMetricOneState = testOneState.Train(iFeatureCombination);
         CHECK_APPROX(validationMetricZeroFeaturesInCombination, validationMetricOneState);
         FractionalDataType validationMetricTwoStates = testTwoStates.Train(iFeatureCombination);
         CHECK_APPROX(validationMetricZeroFeaturesInCombination, validationMetricTwoStates);

         FractionalDataType modelValueZeroFeaturesInCombination0 = testZeroFeaturesInCombination.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
         FractionalDataType modelValueOneState0 = testOneState.GetCurrentModelPredictorScore(iFeatureCombination, { 0 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInCombination0, modelValueOneState0);
         FractionalDataType modelValueTwoStates0 = testTwoStates.GetCurrentModelPredictorScore(iFeatureCombination, { 1 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInCombination0, modelValueTwoStates0);

         FractionalDataType modelValueZeroFeaturesInCombination1 = testZeroFeaturesInCombination.GetCurrentModelPredictorScore(iFeatureCombination, {}, 1);
         FractionalDataType modelValueOneState1 = testOneState.GetCurrentModelPredictorScore(iFeatureCombination, { 0 }, 1);
         CHECK_APPROX(modelValueZeroFeaturesInCombination1, modelValueOneState1);
         FractionalDataType modelValueTwoStates1 = testTwoStates.GetCurrentModelPredictorScore(iFeatureCombination, { 1 }, 1);
         CHECK_APPROX(modelValueZeroFeaturesInCombination1, modelValueTwoStates1);

         FractionalDataType modelValueZeroFeaturesInCombination2 = testZeroFeaturesInCombination.GetCurrentModelPredictorScore(iFeatureCombination, {}, 2);
         FractionalDataType modelValueOneState2 = testOneState.GetCurrentModelPredictorScore(iFeatureCombination, { 0 }, 2);
         CHECK_APPROX(modelValueZeroFeaturesInCombination2, modelValueOneState2);
         FractionalDataType modelValueTwoStates2 = testTwoStates.GetCurrentModelPredictorScore(iFeatureCombination, { 1 }, 2);
         CHECK_APPROX(modelValueZeroFeaturesInCombination2, modelValueTwoStates2);
      }
   }
}

TEST_CASE("3 dimensional featureCombination with one dimension reduced in different ways, training, regression") {
   TestApi test0 = TestApi(k_learningTypeRegression);
   test0.AddFeatures({ FeatureTest(1), FeatureTest(2), FeatureTest(2) });
   test0.AddFeatureCombinations({ { 0, 1, 2 } });
   test0.AddTrainingInstances({ 
      RegressionInstance(9, { 0, 0, 0 }),
      RegressionInstance(10, { 0, 1, 0 }),
      RegressionInstance(11, { 0, 0, 1 }),
      RegressionInstance(12, { 0, 1, 1 }),
      });
   test0.AddValidationInstances({ RegressionInstance(12, { 0, 1, 0 }) });
   test0.InitializeTraining();

   TestApi test1 = TestApi(k_learningTypeRegression);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(1), FeatureTest(2) });
   test1.AddFeatureCombinations({ { 0, 1, 2 } });
   test1.AddTrainingInstances({
      RegressionInstance(9, { 0, 0, 0 }),
      RegressionInstance(10, { 0, 0, 1 }),
      RegressionInstance(11, { 1, 0, 0 }),
      RegressionInstance(12, { 1, 0, 1 }),
      });
   test1.AddValidationInstances({ RegressionInstance(12, { 0, 0, 1 }) });
   test1.InitializeTraining();

   TestApi test2 = TestApi(k_learningTypeRegression);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2), FeatureTest(1) });
   test2.AddFeatureCombinations({ { 0, 1, 2 } });
   test2.AddTrainingInstances({
      RegressionInstance(9, { 0, 0, 0 }),
      RegressionInstance(10, { 1, 0, 0 }),
      RegressionInstance(11, { 0, 1, 0 }),
      RegressionInstance(12, { 1, 1, 0 }),
      });
   test2.AddValidationInstances({ RegressionInstance(12, { 1, 0, 0 }) });
   test2.InitializeTraining();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(test0.GetFeatureCombinationsCount() == test1.GetFeatureCombinationsCount());
      assert(test0.GetFeatureCombinationsCount() == test2.GetFeatureCombinationsCount());
      for(size_t iFeatureCombination = 0; iFeatureCombination < test0.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         FractionalDataType validationMetric0 = test0.Train(iFeatureCombination);
         FractionalDataType validationMetric1 = test1.Train(iFeatureCombination);
         CHECK_APPROX(validationMetric0, validationMetric1);
         FractionalDataType validationMetric2 = test2.Train(iFeatureCombination);
         CHECK_APPROX(validationMetric0, validationMetric2);

         FractionalDataType modelValue01 = test0.GetCurrentModelPredictorScore(iFeatureCombination, { 0, 0, 0 }, 0);
         FractionalDataType modelValue02 = test0.GetCurrentModelPredictorScore(iFeatureCombination, { 0, 0, 1 }, 0);
         FractionalDataType modelValue03 = test0.GetCurrentModelPredictorScore(iFeatureCombination, { 0, 1, 0 }, 0);
         FractionalDataType modelValue04 = test0.GetCurrentModelPredictorScore(iFeatureCombination, { 0, 1, 1 }, 0);

         FractionalDataType modelValue11 = test1.GetCurrentModelPredictorScore(iFeatureCombination, { 0, 0, 0 }, 0);
         FractionalDataType modelValue12 = test1.GetCurrentModelPredictorScore(iFeatureCombination, { 1, 0, 0 }, 0);
         FractionalDataType modelValue13 = test1.GetCurrentModelPredictorScore(iFeatureCombination, { 0, 0, 1 }, 0);
         FractionalDataType modelValue14 = test1.GetCurrentModelPredictorScore(iFeatureCombination, { 1, 0, 1 }, 0);
         CHECK_APPROX(modelValue11, modelValue01);
         CHECK_APPROX(modelValue12, modelValue02);
         CHECK_APPROX(modelValue13, modelValue03);
         CHECK_APPROX(modelValue14, modelValue04);

         FractionalDataType modelValue21 = test2.GetCurrentModelPredictorScore(iFeatureCombination, { 0, 0, 0 }, 0);
         FractionalDataType modelValue22 = test2.GetCurrentModelPredictorScore(iFeatureCombination, { 0, 1, 0 }, 0);
         FractionalDataType modelValue23 = test2.GetCurrentModelPredictorScore(iFeatureCombination, { 1, 0, 0 }, 0);
         FractionalDataType modelValue24 = test2.GetCurrentModelPredictorScore(iFeatureCombination, { 1, 1, 0 }, 0);
         CHECK_APPROX(modelValue21, modelValue01);
         CHECK_APPROX(modelValue22, modelValue02);
         CHECK_APPROX(modelValue23, modelValue03);
         CHECK_APPROX(modelValue24, modelValue04);
      }
   }
}

TEST_CASE("FeatureCombination with one feature with one state, interaction, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(1) });
   test.AddInteractionInstances({ RegressionInstance(10, { 0 }) });
   test.InitializeInteraction();
   FractionalDataType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("FeatureCombination with one feature with one state, interaction, Binary") {
   TestApi test = TestApi(2);
   test.AddFeatures({ FeatureTest(1) });
   test.AddInteractionInstances({ ClassificationInstance(0, { 0 }) });
   test.InitializeInteraction();
   FractionalDataType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("FeatureCombination with one feature with one state, interaction, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(1) });
   test.AddInteractionInstances({ ClassificationInstance(0, { 0 }) });
   test.InitializeInteraction();
   FractionalDataType metricReturn = test.InteractionScore({0});
   CHECK(0 == metricReturn);
}

TEST_CASE("Test Rehydration, training, regression") {
   TestApi testContinuous = TestApi(k_learningTypeRegression);
   testContinuous.AddFeatures({});
   testContinuous.AddFeatureCombinations({ {} });
   testContinuous.AddTrainingInstances({ RegressionInstance(10, {}) });
   testContinuous.AddValidationInstances({ RegressionInstance(12, {}) });
   testContinuous.InitializeTraining();

   FractionalDataType model0 = 0;

   FractionalDataType validationMetricContinuous;
   FractionalDataType modelValueContinuous;
   FractionalDataType validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(k_learningTypeRegression);
      testRestart.AddFeatures({});
      testRestart.AddFeatureCombinations({ {} });
      testRestart.AddTrainingInstances({ RegressionInstance(10, {}, model0) });
      testRestart.AddValidationInstances({ RegressionInstance(12, {}, model0) });
      testRestart.InitializeTraining();

      validationMetricRestart = testRestart.Train(0);
      validationMetricContinuous = testContinuous.Train(0);
      CHECK_APPROX(validationMetricContinuous, validationMetricRestart);

      modelValueContinuous = testContinuous.GetCurrentModelPredictorScore(0, {}, 0);
      model0 += testRestart.GetCurrentModelPredictorScore(0, {}, 0);
      CHECK_APPROX(modelValueContinuous, model0);
   }
}

TEST_CASE("Test Rehydration, training, binary") {
   TestApi testContinuous = TestApi(2);
   testContinuous.AddFeatures({});
   testContinuous.AddFeatureCombinations({ {} });
   testContinuous.AddTrainingInstances({ ClassificationInstance(0, {}) });
   testContinuous.AddValidationInstances({ ClassificationInstance(0, {}) });
   testContinuous.InitializeTraining();

   FractionalDataType model0 = 0;
   FractionalDataType model1 = 0;

   FractionalDataType validationMetricContinuous;
   FractionalDataType modelValueContinuous;
   FractionalDataType validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(2);
      testRestart.AddFeatures({});
      testRestart.AddFeatureCombinations({ {} });
      testRestart.AddTrainingInstances({ ClassificationInstance(0, {}, { model0, model1 }) });
      testRestart.AddValidationInstances({ ClassificationInstance(0, {}, { model0, model1 }) });
      testRestart.InitializeTraining();

      validationMetricRestart = testRestart.Train(0);
      validationMetricContinuous = testContinuous.Train(0);
      CHECK_APPROX(validationMetricContinuous, validationMetricRestart);

      modelValueContinuous = testContinuous.GetCurrentModelPredictorScore(0, {}, 0);
      model0 += testRestart.GetCurrentModelPredictorScore(0, {}, 0);
      CHECK_APPROX(modelValueContinuous, model0);

      modelValueContinuous = testContinuous.GetCurrentModelPredictorScore(0, {}, 1);
      model1 += testRestart.GetCurrentModelPredictorScore(0, {}, 1);
      CHECK_APPROX(modelValueContinuous, model1);
   }
}

TEST_CASE("Test Rehydration, training, multiclass") {
   TestApi testContinuous = TestApi(3);
   testContinuous.AddFeatures({});
   testContinuous.AddFeatureCombinations({ {} });
   testContinuous.AddTrainingInstances({ ClassificationInstance(0, {}) });
   testContinuous.AddValidationInstances({ ClassificationInstance(0, {}) });
   testContinuous.InitializeTraining();

   FractionalDataType model0 = 0;
   FractionalDataType model1 = 0;
   FractionalDataType model2 = 0;

   FractionalDataType validationMetricContinuous;
   FractionalDataType modelValueContinuous;
   FractionalDataType validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(3);
      testRestart.AddFeatures({});
      testRestart.AddFeatureCombinations({ {} });
      testRestart.AddTrainingInstances({ ClassificationInstance(0, {}, { model0, model1, model2 }) });
      testRestart.AddValidationInstances({ ClassificationInstance(0, {}, { model0, model1, model2 }) });
      testRestart.InitializeTraining();

      validationMetricRestart = testRestart.Train(0);
      validationMetricContinuous = testContinuous.Train(0);
      CHECK_APPROX(validationMetricContinuous, validationMetricRestart);

      modelValueContinuous = testContinuous.GetCurrentModelPredictorScore(0, {}, 0);
      model0 += testRestart.GetCurrentModelPredictorScore(0, {}, 0);
      CHECK_APPROX(modelValueContinuous, model0);

      modelValueContinuous = testContinuous.GetCurrentModelPredictorScore(0, {}, 1);
      model1 += testRestart.GetCurrentModelPredictorScore(0, {}, 1);
      CHECK_APPROX(modelValueContinuous, model1);

      modelValueContinuous = testContinuous.GetCurrentModelPredictorScore(0, {}, 2);
      model2 += testRestart.GetCurrentModelPredictorScore(0, {}, 2);
      CHECK_APPROX(modelValueContinuous, model2);
   }
}

void EBMCORE_CALLING_CONVENTION LogMessage(signed char traceLevel, const char * message) {
   UNUSED(traceLevel);
   // don't display the message, but we want to test all our messages, so have them call us here
   strlen(message); // test that the string memory is accessible
//   printf("%d - %s\n", traceLevel, message);
}

int main() {
   SetLogMessageFunction(&LogMessage);
   SetTraceLevel(TraceLevelVerbose);

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
