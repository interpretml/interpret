// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderEbmNativeTest.h"

// we roll our own test framework here since it's nice having no dependencies, and we just need a few simple tests for the C API.
// If we ended up needing something more substantial, I'd consider using doctest ( https://github.com/onqtam/doctest ) because:
//   1) It's a single include file, which is the simplest we could ask for.  Googletest is more heavyweight
//   2) It's MIT licensed, so we could include the header in our project and still keep our license 100% MIT compatible without having two licenses, 
//      unlike Catch, or Catch2
//   3) It's fast to compile.
//   4) doctest is very close to having a JUnit output feature.  JUnit isn't really required, our python testing uses JUnit, so it would be nice to have 
//      the same format -> https://github.com/onqtam/doctest/blob/master/doc/markdown/roadmap.md   https://github.com/onqtam/doctest/issues/75
//   5) If JUnit is desired in the meantime, there is a converter that will output JUnit -> https://github.com/ujiro99/doctest-junit-report
//
// In case we want to use doctest in the future, use the format of the following: TEST_CASE, CHECK & FAIL_CHECK (continues testing) / REQUIRE & FAIL 
//   (stops the current test, but we could just terminate), INFO (print to log file)
// Don't implement this since it would be harder to do: SUBCASE

// TODO : add test for the condition where we overflow the small model update to NaN or +-infinity for regression by using exteme regression values and in 
//   classification by using certainty situations with big learning rates
// TODO : add test for the condition where we overflow the result of adding the small model update to the existing model NaN or +-infinity for regression 
//   by using exteme regression values and in classification by using certainty situations with big learning rates
// TODO : add test for the condition where we overflow the validation regression or classification residuals without overflowing the model update or the 
//   model tensors.  We can do this by having two extreme features that will overflow together

// TODO: write a test to compare gain from single vs multi-dimensional splitting (they use the same underlying function, so if we make a pair where one 
//    feature has duplicates for all 0 and 1 values, then the split if we control it should give us the same gain
// TODO: write some NaN and +infinity tests to check propagation at various points

#include <string>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <assert.h>
#include <string.h>

#include "ebm_native.h"
#include "RandomStream.h"

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
   static int CONCATENATE_TOKENS(UNUSED_INTEGER_HIDDEN_, __LINE__) = \
      RegisterTestHidden(TestCaseHidden(&CONCATENATE_TOKENS(TEST_FUNCTION_HIDDEN_, __LINE__), description)); \
   static void CONCATENATE_TOKENS(TEST_FUNCTION_HIDDEN_, __LINE__)(TestCaseHidden& testCaseHidden)

int g_countEqualityFailures = 0;

inline bool IsApproxEqual(const double value, const double expected, const double percentage) {
   bool isEqual = false;
   if(!std::isnan(value)) {
      if(!std::isnan(expected)) {
         if(!std::isinf(value)) {
            if(!std::isinf(expected)) {
               const double smaller = double { 1 } - percentage;
               const double bigger = double { 1 } + percentage;
               if(0 < value) {
                  if(0 < expected) {
                     if(value <= expected) {
                        // expected is the bigger number in absolute terms
                        if(expected * smaller <= value && value <= expected * bigger) {
                           isEqual = true;
                        }
                     } else {
                        // value is the bigger number in absolute terms
                        if(value * smaller <= expected && expected <= value * bigger) {
                           isEqual = true;
                        }
                     }
                  }
               } else if(value < 0) {
                  if(expected < 0) {
                     if(expected <= value) {
                        // expected is the bigger number in absolute terms (the biggest negative number)
                        if(expected * bigger <= value && value <= expected * smaller) {
                           isEqual = true;
                        }
                     } else {
                        // value is the bigger number in absolute terms (the biggest negative number)
                        if(value * bigger <= expected && expected <= value * smaller) {
                           isEqual = true;
                        }
                     }
                  }
               } else {
                  if(0 == expected) {
                     isEqual = true;
                  }
               }
            }
         }
      }
   }
   if(!isEqual) {
      // we're going to fail!
      ++g_countEqualityFailures; // this doesn't do anything useful but gives us something to break on
   }
   return isEqual;
}

// this will ONLY work if used inside the root TEST_CASE function.  The testCaseHidden variable comes from TEST_CASE and should be visible inside the 
// function where CHECK(expression) is called
#define CHECK(expression) \
   do { \
      const bool bFailedHidden = !(expression); \
      if(bFailedHidden) { \
         std::cout << " FAILED on \"" #expression "\""; \
         testCaseHidden.m_bPassed = false; \
      } \
   } while((void)0, 0)

// this will ONLY work if used inside the root TEST_CASE function.  The testCaseHidden variable comes from TEST_CASE and should be visible inside the 
// function where CHECK_APPROX(expression) is called
#define CHECK_APPROX(value, expected) \
   do { \
      const double valueHidden = (value); \
      const bool bApproxEqualHidden = IsApproxEqual(valueHidden, static_cast<double>(expected), double { 1e-6 }); \
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

constexpr IntEbmType randomSeed = 42;
enum class FeatureType : IntEbmType { Ordinal = FeatureTypeOrdinal, Nominal = FeatureTypeNominal };

class FeatureTest final {
public:

   const FeatureType m_featureType;
   const bool m_hasMissing;
   const IntEbmType m_countBins;

   FeatureTest(const IntEbmType countBins, const FeatureType featureType = FeatureType::Ordinal, const bool hasMissing = false) :
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
   const FloatEbmType m_target;
   const std::vector<IntEbmType> m_binnedDataPerFeatureArray;
   const FloatEbmType m_priorPredictorPrediction;
   const bool m_bNullPredictionScores;

   RegressionInstance(const FloatEbmType target, const std::vector<IntEbmType> binnedDataPerFeatureArray) :
      m_target(target),
      m_binnedDataPerFeatureArray(binnedDataPerFeatureArray),
      m_priorPredictorPrediction(0),
      m_bNullPredictionScores(true) {
   }

   RegressionInstance(const FloatEbmType target, const std::vector<IntEbmType> binnedDataPerFeatureArray, const FloatEbmType priorPredictorPrediction) :
      m_target(target),
      m_binnedDataPerFeatureArray(binnedDataPerFeatureArray),
      m_priorPredictorPrediction(priorPredictorPrediction),
      m_bNullPredictionScores(false) {
   }
};

class ClassificationInstance final {
public:
   const IntEbmType m_target;
   const std::vector<IntEbmType> m_binnedDataPerFeatureArray;
   const std::vector<FloatEbmType> m_priorPredictorPerClassLogits;
   const bool m_bNullPredictionScores;

   ClassificationInstance(const IntEbmType target, const std::vector<IntEbmType> binnedDataPerFeatureArray) :
      m_target(target),
      m_binnedDataPerFeatureArray(binnedDataPerFeatureArray),
      m_bNullPredictionScores(true) {
   }

   ClassificationInstance(
      const IntEbmType target, 
      const std::vector<IntEbmType> binnedDataPerFeatureArray, 
      const std::vector<FloatEbmType> priorPredictorPerClassLogits) 
   :
      m_target(target),
      m_binnedDataPerFeatureArray(binnedDataPerFeatureArray),
      m_priorPredictorPerClassLogits(priorPredictorPerClassLogits),
      m_bNullPredictionScores(false) {
   }
};

static constexpr ptrdiff_t k_iZeroClassificationLogitDefault = ptrdiff_t { -1 };
static constexpr IntEbmType k_countInnerBagsDefault = IntEbmType { 0 };
static constexpr FloatEbmType k_learningRateDefault = FloatEbmType { 0.01 };
static constexpr IntEbmType k_countTreeSplitsMaxDefault = IntEbmType { 4 };
static constexpr IntEbmType k_countInstancesRequiredForChildSplitMinDefault = IntEbmType { 1 };

class TestApi {
   enum class Stage {
      Beginning, FeaturesAdded, FeatureCombinationsAdded, TrainingAdded, ValidationAdded, InitializedBoosting, InteractionAdded, InitializedInteraction
   };

   Stage m_stage;
   const ptrdiff_t m_learningTypeOrCountTargetClasses;
   const ptrdiff_t m_iZeroClassificationLogit;

   std::vector<EbmNativeFeature> m_features;
   std::vector<EbmNativeFeatureCombination> m_featureCombinations;
   std::vector<IntEbmType> m_featureCombinationIndexes;

   std::vector<std::vector<size_t>> m_countBinsByFeatureCombination;

   std::vector<FloatEbmType> m_trainingRegressionTargets;
   std::vector<IntEbmType> m_trainingClassificationTargets;
   std::vector<IntEbmType> m_trainingBinnedData;
   std::vector<FloatEbmType> m_trainingPredictionScores;
   bool m_bNullTrainingPredictionScores;

   std::vector<FloatEbmType> m_validationRegressionTargets;
   std::vector<IntEbmType> m_validationClassificationTargets;
   std::vector<IntEbmType> m_validationBinnedData;
   std::vector<FloatEbmType> m_validationPredictionScores;
   bool m_bNullValidationPredictionScores;

   PEbmBoosting m_pEbmBoosting;

   std::vector<FloatEbmType> m_interactionRegressionTargets;
   std::vector<IntEbmType> m_interactionClassificationTargets;
   std::vector<IntEbmType> m_interactionBinnedData;
   std::vector<FloatEbmType> m_interactionPredictionScores;
   bool m_bNullInteractionPredictionScores;

   PEbmInteraction m_pEbmInteraction;

   const FloatEbmType * GetPredictorScores(
      const size_t iFeatureCombination, 
      const FloatEbmType * const pModelFeatureCombination, 
      const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures) 
   const {
      if(Stage::InitializedBoosting != m_stage) {
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

   FloatEbmType GetPredictorScore(
      const size_t iFeatureCombination, 
      const FloatEbmType * const pModelFeatureCombination, 
      const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures, 
      const size_t iTargetClassOrZero) 
   const {
      const FloatEbmType * const aScores = GetPredictorScores(iFeatureCombination, pModelFeatureCombination, perDimensionIndexArrayForBinnedFeatures);
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
            return aScores[iTargetClassOrZero];
         } else {
            if(static_cast<size_t>(m_iZeroClassificationLogit) == iTargetClassOrZero) {
               return FloatEbmType { 0 };
            } else {
               return aScores[iTargetClassOrZero] - aScores[m_iZeroClassificationLogit];
            }
         }
#else // EXPAND_BINARY_LOGITS
         if(m_iZeroClassificationLogit < 0) {
            if(0 == iTargetClassOrZero) {
               return FloatEbmType { 0 };
            } else {
               return aScores[0];
            }
         } else {
            if(static_cast<size_t>(m_iZeroClassificationLogit) == iTargetClassOrZero) {
               return FloatEbmType { 0 };
            } else {
               return aScores[0];
            }
         }
#endif // EXPAND_BINARY_LOGITS
      } else {
         // multiclass
#ifdef REDUCE_MULTICLASS_LOGITS
         if(m_iZeroClassificationLogit < 0) {
            if(0 == iTargetClassOrZero) {
               return FloatEbmType { 0 };
            } else {
               return aScores[iTargetClassOrZero - 1];
            }
         } else {
            if(staitc_cast<size_t>(m_iZeroClassificationLogit) == iTargetClassOrZero) {
               return FloatEbmType { 0 };
            } else if(staitc_cast<size_t>(m_iZeroClassificationLogit) < iTargetClassOrZero) {
               return aScores[iTargetClassOrZero - 1];
            } else {
               return aScores[iTargetClassOrZero];
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
      m_pEbmBoosting(nullptr),
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
      if(nullptr != m_pEbmBoosting) {
         FreeBoosting(m_pEbmBoosting);
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
         EbmNativeFeature feature;
         feature.featureType = static_cast<IntEbmType>(oneFeature.m_featureType);
         feature.hasMissing = oneFeature.m_hasMissing ? IntEbmType { 1 } : IntEbmType { 0 };
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
         EbmNativeFeatureCombination featureCombination;
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
      const size_t cInstances = instances.size();
      if(0 != cInstances) {
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
            const FloatEbmType target = oneInstance.m_target;
            if(std::isnan(target)) {
               exit(1);
            }
            if(std::isinf(target)) {
               exit(1);
            }
            m_trainingRegressionTargets.push_back(target);
            if(!bNullPredictionScores) {
               const FloatEbmType score = oneInstance.m_priorPredictorPrediction;
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
            const EbmNativeFeature feature = m_features[iFeature];
            for(size_t iInstance = 0; iInstance < cInstances; ++iInstance) {
               const IntEbmType data = instances[iInstance].m_binnedDataPerFeatureArray[iFeature];
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
      const size_t cInstances = instances.size();
      if(0 != cInstances) {
         const size_t cFeatures = m_features.size();
         const bool bNullPredictionScores = instances[0].m_bNullPredictionScores;
         m_bNullTrainingPredictionScores = bNullPredictionScores;

         for(const ClassificationInstance oneInstance : instances) {
            if(cFeatures != oneInstance.m_binnedDataPerFeatureArray.size()) {
               exit(1);
            }
            if(bNullPredictionScores != oneInstance.m_bNullPredictionScores) {
               exit(1);
            }
            const IntEbmType target = oneInstance.m_target;
            if(target < 0) {
               exit(1);
            }
            if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) <= static_cast<size_t>(target)) {
               exit(1);
            }
            m_trainingClassificationTargets.push_back(target);
            if(!bNullPredictionScores) {
               if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) != oneInstance.m_priorPredictorPerClassLogits.size()) {
                  exit(1);
               }
               ptrdiff_t iLogit = 0;
               for(const FloatEbmType oneLogit : oneInstance.m_priorPredictorPerClassLogits) {
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
                        m_trainingPredictionScores.push_back(oneLogit - oneInstance.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                     }
#else // EXPAND_BINARY_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_trainingPredictionScores.push_back(oneLogit - oneInstance.m_priorPredictorPerClassLogits[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_trainingPredictionScores.push_back(oneLogit - oneInstance.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                        }
                     }
#endif // EXPAND_BINARY_LOGITS
                  } else {
                     // multiclass
#ifdef REDUCE_MULTICLASS_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_trainingPredictionScores.push_back(oneLogit - oneInstance.m_logits[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_trainingPredictionScores.push_back(oneLogit - oneInstance.m_logits[m_iZeroClassificationLogit]);
                        }
                     }
#else // REDUCE_MULTICLASS_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        m_trainingPredictionScores.push_back(oneLogit);
                     } else {
                        m_trainingPredictionScores.push_back(oneLogit - oneInstance.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                     }
#endif // REDUCE_MULTICLASS_LOGITS
                  }
                  ++iLogit;
               }
            }
         }
         for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
            const EbmNativeFeature feature = m_features[iFeature];
            for(size_t iInstance = 0; iInstance < cInstances; ++iInstance) {
               const IntEbmType data = instances[iInstance].m_binnedDataPerFeatureArray[iFeature];
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
      const size_t cInstances = instances.size();
      if(0 != cInstances) {
         const size_t cFeatures = m_features.size();
         const bool bNullPredictionScores = instances[0].m_bNullPredictionScores;
         m_bNullValidationPredictionScores = bNullPredictionScores;

         for(const RegressionInstance oneInstance : instances) {
            if(cFeatures != oneInstance.m_binnedDataPerFeatureArray.size()) {
               exit(1);
            }
            if(bNullPredictionScores != oneInstance.m_bNullPredictionScores) {
               exit(1);
            }
            const FloatEbmType target = oneInstance.m_target;
            if(std::isnan(target)) {
               exit(1);
            }
            if(std::isinf(target)) {
               exit(1);
            }
            m_validationRegressionTargets.push_back(target);
            if(!bNullPredictionScores) {
               const FloatEbmType score = oneInstance.m_priorPredictorPrediction;
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
            const EbmNativeFeature feature = m_features[iFeature];
            for(size_t iInstance = 0; iInstance < cInstances; ++iInstance) {
               const IntEbmType data = instances[iInstance].m_binnedDataPerFeatureArray[iFeature];
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
      const size_t cInstances = instances.size();
      if(0 != cInstances) {
         const size_t cFeatures = m_features.size();
         const bool bNullPredictionScores = instances[0].m_bNullPredictionScores;
         m_bNullValidationPredictionScores = bNullPredictionScores;

         for(const ClassificationInstance oneInstance : instances) {
            if(cFeatures != oneInstance.m_binnedDataPerFeatureArray.size()) {
               exit(1);
            }
            if(bNullPredictionScores != oneInstance.m_bNullPredictionScores) {
               exit(1);
            }
            const IntEbmType target = oneInstance.m_target;
            if(target < 0) {
               exit(1);
            }
            if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) <= static_cast<size_t>(target)) {
               exit(1);
            }
            m_validationClassificationTargets.push_back(target);
            if(!bNullPredictionScores) {
               if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) != oneInstance.m_priorPredictorPerClassLogits.size()) {
                  exit(1);
               }
               ptrdiff_t iLogit = 0;
               for(const FloatEbmType oneLogit : oneInstance.m_priorPredictorPerClassLogits) {
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
                        m_validationPredictionScores.push_back(oneLogit - oneInstance.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                     }
#else // EXPAND_BINARY_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_validationPredictionScores.push_back(oneLogit - oneInstance.m_priorPredictorPerClassLogits[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_validationPredictionScores.push_back(oneLogit - oneInstance.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                        }
                     }
#endif // EXPAND_BINARY_LOGITS
                  } else {
                     // multiclass
#ifdef REDUCE_MULTICLASS_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_validationPredictionScores.push_back(oneLogit - oneInstance.m_logits[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_validationPredictionScores.push_back(oneLogit - oneInstance.m_logits[m_iZeroClassificationLogit]);
                        }
                     }
#else // REDUCE_MULTICLASS_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        m_validationPredictionScores.push_back(oneLogit);
                     } else {
                        m_validationPredictionScores.push_back(oneLogit - oneInstance.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                     }
#endif // REDUCE_MULTICLASS_LOGITS
                  }
                  ++iLogit;
               }
            }
         }
         for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
            const EbmNativeFeature feature = m_features[iFeature];
            for(size_t iInstance = 0; iInstance < cInstances; ++iInstance) {
               const IntEbmType data = instances[iInstance].m_binnedDataPerFeatureArray[iFeature];
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

   void InitializeBoosting(const IntEbmType countInnerBags = k_countInnerBagsDefault) {
      if(Stage::ValidationAdded != m_stage) {
         exit(1);
      }
      if(countInnerBags < IntEbmType { 0 }) {
         exit(1);
      }

      const size_t cVectorLength = GetVectorLength(m_learningTypeOrCountTargetClasses);
      if(IsClassification(m_learningTypeOrCountTargetClasses)) {
         if(m_bNullTrainingPredictionScores) {
            m_trainingPredictionScores.resize(cVectorLength * m_trainingClassificationTargets.size());
         }
         if(m_bNullValidationPredictionScores) {
            m_validationPredictionScores.resize(cVectorLength * m_validationClassificationTargets.size());
         }
         m_pEbmBoosting = InitializeBoostingClassification(
            m_learningTypeOrCountTargetClasses, 
            m_features.size(), 
            0 == m_features.size() ? nullptr : &m_features[0], 
            m_featureCombinations.size(), 
            0 == m_featureCombinations.size() ? nullptr : &m_featureCombinations[0], 
            0 == m_featureCombinationIndexes.size() ? nullptr : &m_featureCombinationIndexes[0], 
            m_trainingClassificationTargets.size(), 
            0 == m_trainingBinnedData.size() ? nullptr : &m_trainingBinnedData[0], 
            0 == m_trainingClassificationTargets.size() ? nullptr : &m_trainingClassificationTargets[0], 
            0 == m_trainingClassificationTargets.size() ? nullptr : &m_trainingPredictionScores[0], 
            m_validationClassificationTargets.size(), 
            0 == m_validationBinnedData.size() ? nullptr : &m_validationBinnedData[0], 
            0 == m_validationClassificationTargets.size() ? nullptr : &m_validationClassificationTargets[0], 
            0 == m_validationClassificationTargets.size() ? nullptr : &m_validationPredictionScores[0], 
            countInnerBags, 
            randomSeed,
            nullptr
         );
      } else if(k_learningTypeRegression == m_learningTypeOrCountTargetClasses) {
         if(m_bNullTrainingPredictionScores) {
            m_trainingPredictionScores.resize(cVectorLength * m_trainingRegressionTargets.size());
         }
         if(m_bNullValidationPredictionScores) {
            m_validationPredictionScores.resize(cVectorLength * m_validationRegressionTargets.size());
         }
         m_pEbmBoosting = InitializeBoostingRegression(
            m_features.size(), 
            0 == m_features.size() ? nullptr : &m_features[0], 
            m_featureCombinations.size(), 
            0 == m_featureCombinations.size() ? nullptr : &m_featureCombinations[0], 
            0 == m_featureCombinationIndexes.size() ? nullptr : &m_featureCombinationIndexes[0], 
            m_trainingRegressionTargets.size(), 
            0 == m_trainingBinnedData.size() ? nullptr : &m_trainingBinnedData[0], 
            0 == m_trainingRegressionTargets.size() ? nullptr : &m_trainingRegressionTargets[0], 
            0 == m_trainingRegressionTargets.size() ? nullptr : &m_trainingPredictionScores[0], 
            m_validationRegressionTargets.size(), 
            0 == m_validationBinnedData.size() ? nullptr : &m_validationBinnedData[0], 
            0 == m_validationRegressionTargets.size() ? nullptr : &m_validationRegressionTargets[0], 
            0 == m_validationRegressionTargets.size() ? nullptr : &m_validationPredictionScores[0], 
            countInnerBags, 
            randomSeed,
            nullptr
         );
      } else {
         exit(1);
      }

      if(nullptr == m_pEbmBoosting) {
         printf("\nClean exit with nullptr from InitializeBoosting*.\n");
         exit(1);
      }
      m_stage = Stage::InitializedBoosting;
   }

   FloatEbmType Boost(const IntEbmType indexFeatureCombination, const std::vector<FloatEbmType> trainingWeights = {}, const std::vector<FloatEbmType> validationWeights = {}, const FloatEbmType learningRate = k_learningRateDefault, const IntEbmType countTreeSplitsMax = k_countTreeSplitsMaxDefault, const IntEbmType countInstancesRequiredForChildSplitMin = k_countInstancesRequiredForChildSplitMinDefault) {
      if(Stage::InitializedBoosting != m_stage) {
         exit(1);
      }
      if(indexFeatureCombination < IntEbmType { 0 }) {
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
      if(countTreeSplitsMax < FloatEbmType { 0 }) {
         exit(1);
      }
      if(countInstancesRequiredForChildSplitMin < FloatEbmType { 0 }) {
         exit(1);
      }

      FloatEbmType validationMetricReturn = FloatEbmType { 0 };
      const IntEbmType ret = BoostingStep(
         m_pEbmBoosting, 
         indexFeatureCombination, 
         learningRate, 
         countTreeSplitsMax, 
         countInstancesRequiredForChildSplitMin,
         0 == trainingWeights.size() ? nullptr : &trainingWeights[0], 
         0 == validationWeights.size() ? nullptr : &validationWeights[0], 
         &validationMetricReturn
      );
      if(0 != ret) {
         exit(1);
      }
      return validationMetricReturn;
   }

   FloatEbmType GetBestModelPredictorScore(const size_t iFeatureCombination, const std::vector<size_t> indexes, const size_t iScore) const {
      if(Stage::InitializedBoosting != m_stage) {
         exit(1);
      }
      if(m_featureCombinations.size() <= iFeatureCombination) {
         exit(1);
      }
      FloatEbmType * pModelFeatureCombination = GetBestModelFeatureCombination(m_pEbmBoosting, iFeatureCombination);
      FloatEbmType predictorScore = GetPredictorScore(iFeatureCombination, pModelFeatureCombination, indexes, iScore);
      return predictorScore;
   }

   const FloatEbmType * GetBestModelFeatureCombinationRaw(const size_t iFeatureCombination) const {
      if(Stage::InitializedBoosting != m_stage) {
         exit(1);
      }
      if(m_featureCombinations.size() <= iFeatureCombination) {
         exit(1);
      }
      FloatEbmType * pModel = GetBestModelFeatureCombination(m_pEbmBoosting, iFeatureCombination);
      return pModel;
   }

   FloatEbmType GetCurrentModelPredictorScore(
      const size_t iFeatureCombination, 
      const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures, 
      const size_t iTargetClassOrZero)
   const {
      if(Stage::InitializedBoosting != m_stage) {
         exit(1);
      }
      if(m_featureCombinations.size() <= iFeatureCombination) {
         exit(1);
      }
      FloatEbmType * pModelFeatureCombination = GetCurrentModelFeatureCombination(m_pEbmBoosting, iFeatureCombination);
      FloatEbmType predictorScore = GetPredictorScore(
         iFeatureCombination, 
         pModelFeatureCombination, 
         perDimensionIndexArrayForBinnedFeatures, 
         iTargetClassOrZero
      );
      return predictorScore;
   }

   const FloatEbmType * GetCurrentModelFeatureCombinationRaw(const size_t iFeatureCombination) const {
      if(Stage::InitializedBoosting != m_stage) {
         exit(1);
      }
      if(m_featureCombinations.size() <= iFeatureCombination) {
         exit(1);
      }
      FloatEbmType * pModel = GetCurrentModelFeatureCombination(m_pEbmBoosting, iFeatureCombination);
      return pModel;
   }

   void AddInteractionInstances(const std::vector<RegressionInstance> instances) {
      if(Stage::FeaturesAdded != m_stage) {
         exit(1);
      }
      if(k_learningTypeRegression != m_learningTypeOrCountTargetClasses) {
         exit(1);
      }
      const size_t cInstances = instances.size();
      if(0 != cInstances) {
         const size_t cFeatures = m_features.size();
         const bool bNullPredictionScores = instances[0].m_bNullPredictionScores;
         m_bNullInteractionPredictionScores = bNullPredictionScores;

         for(const RegressionInstance oneInstance : instances) {
            if(cFeatures != oneInstance.m_binnedDataPerFeatureArray.size()) {
               exit(1);
            }
            if(bNullPredictionScores != oneInstance.m_bNullPredictionScores) {
               exit(1);
            }
            const FloatEbmType target = oneInstance.m_target;
            if(std::isnan(target)) {
               exit(1);
            }
            if(std::isinf(target)) {
               exit(1);
            }
            m_interactionRegressionTargets.push_back(target);
            if(!bNullPredictionScores) {
               const FloatEbmType score = oneInstance.m_priorPredictorPrediction;
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
            const EbmNativeFeature feature = m_features[iFeature];
            for(size_t iInstance = 0; iInstance < cInstances; ++iInstance) {
               const IntEbmType data = instances[iInstance].m_binnedDataPerFeatureArray[iFeature];
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
      const size_t cInstances = instances.size();
      if(0 != cInstances) {
         const size_t cFeatures = m_features.size();
         const bool bNullPredictionScores = instances[0].m_bNullPredictionScores;
         m_bNullInteractionPredictionScores = bNullPredictionScores;

         for(const ClassificationInstance oneInstance : instances) {
            if(cFeatures != oneInstance.m_binnedDataPerFeatureArray.size()) {
               exit(1);
            }
            if(bNullPredictionScores != oneInstance.m_bNullPredictionScores) {
               exit(1);
            }
            const IntEbmType target = oneInstance.m_target;
            if(target < 0) {
               exit(1);
            }
            if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) <= static_cast<size_t>(target)) {
               exit(1);
            }
            m_interactionClassificationTargets.push_back(target);
            if(!bNullPredictionScores) {
               if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) != oneInstance.m_priorPredictorPerClassLogits.size()) {
                  exit(1);
               }
               ptrdiff_t iLogit = 0;
               for(const FloatEbmType oneLogit : oneInstance.m_priorPredictorPerClassLogits) {
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
                        m_interactionPredictionScores.push_back(oneLogit - oneInstance.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                     }
#else // EXPAND_BINARY_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_interactionPredictionScores.push_back(oneLogit - oneInstance.m_priorPredictorPerClassLogits[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_interactionPredictionScores.push_back(oneLogit - oneInstance.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                        }
                     }
#endif // EXPAND_BINARY_LOGITS
                  } else {
                     // multiclass
#ifdef REDUCE_MULTICLASS_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_interactionPredictionScores.push_back(oneLogit - oneInstance.m_logits[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_interactionPredictionScores.push_back(oneLogit - oneInstance.m_logits[m_iZeroClassificationLogit]);
                        }
                     }
#else // REDUCE_MULTICLASS_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        m_interactionPredictionScores.push_back(oneLogit);
                     } else {
                        m_interactionPredictionScores.push_back(oneLogit - oneInstance.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                     }
#endif // REDUCE_MULTICLASS_LOGITS
                  }
                  ++iLogit;
               }
            }
         }
         for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
            const EbmNativeFeature feature = m_features[iFeature];
            for(size_t iInstance = 0; iInstance < cInstances; ++iInstance) {
               const IntEbmType data = instances[iInstance].m_binnedDataPerFeatureArray[iFeature];
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

      const size_t cVectorLength = GetVectorLength(m_learningTypeOrCountTargetClasses);
      if(IsClassification(m_learningTypeOrCountTargetClasses)) {
         if(m_bNullInteractionPredictionScores) {
            m_interactionPredictionScores.resize(cVectorLength * m_interactionClassificationTargets.size());
         }
         m_pEbmInteraction = InitializeInteractionClassification(
            m_learningTypeOrCountTargetClasses, 
            m_features.size(), 
            0 == m_features.size() ? nullptr : &m_features[0], 
            m_interactionClassificationTargets.size(), 
            0 == m_interactionBinnedData.size() ? nullptr : &m_interactionBinnedData[0], 
            0 == m_interactionClassificationTargets.size() ? nullptr : &m_interactionClassificationTargets[0], 
            0 == m_interactionClassificationTargets.size() ? nullptr : &m_interactionPredictionScores[0],
            nullptr
         );
      } else if(k_learningTypeRegression == m_learningTypeOrCountTargetClasses) {
         if(m_bNullInteractionPredictionScores) {
            m_interactionPredictionScores.resize(cVectorLength * m_interactionRegressionTargets.size());
         }
         m_pEbmInteraction = InitializeInteractionRegression(
            m_features.size(), 
            0 == m_features.size() ? nullptr : &m_features[0], 
            m_interactionRegressionTargets.size(), 
            0 == m_interactionBinnedData.size() ? nullptr : &m_interactionBinnedData[0], 
            0 == m_interactionRegressionTargets.size() ? nullptr : &m_interactionRegressionTargets[0], 
            0 == m_interactionRegressionTargets.size() ? nullptr : &m_interactionPredictionScores[0],
            nullptr
         );
      } else {
         exit(1);
      }

      if(nullptr == m_pEbmInteraction) {
         exit(1);
      }
      m_stage = Stage::InitializedInteraction;
   }

   FloatEbmType InteractionScore(const std::vector<IntEbmType> featuresInCombination, const IntEbmType countInstancesRequiredForChildSplitMin = k_countInstancesRequiredForChildSplitMinDefault) const {
      if(Stage::InitializedInteraction != m_stage) {
         exit(1);
      }
      for(const IntEbmType oneFeatureIndex : featuresInCombination) {
         if(oneFeatureIndex < IntEbmType { 0 }) {
            exit(1);
         }
         if(m_features.size() <= static_cast<size_t>(oneFeatureIndex)) {
            exit(1);
         }
      }

      FloatEbmType interactionScoreReturn = FloatEbmType { 0 };
      const IntEbmType ret = GetInteractionScore(
         m_pEbmInteraction, 
         featuresInCombination.size(), 
         0 == featuresInCombination.size() ? nullptr : &featuresInCombination[0], 
         countInstancesRequiredForChildSplitMin,
         &interactionScoreReturn
      );
      if(0 != ret) {
         exit(1);
      }
      return interactionScoreReturn;
   }
};

TEST_CASE("test random number generator equivalency") {
   TestApi test = TestApi(2);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });

   std::vector<ClassificationInstance> instances;
   for(int i = 0; i < 1000; ++i) {
      instances.push_back(ClassificationInstance(i % 2, { 0 == (i * 7) % 3 }));
   }

   test.AddTrainingInstances( instances );
   test.AddValidationInstances({ ClassificationInstance(0, { 0 }), ClassificationInstance(1, { 1 }) });

   test.InitializeBoosting(2);

   for(int iEpoch = 0; iEpoch < 100; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         test.Boost(iFeatureCombination);
      }
   }

   FloatEbmType modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 1);
   // this is meant to be an exact check for this value.  We are testing here if we can generate identical results
   // accross different OSes and C/C++ libraries.  We specificed 2 inner samples, which will use the random generator
   // and if there are any differences between environments then this will catch those
   CHECK_APPROX(modelValue, -0.037461811081225427);
}

TEST_CASE("Discretize, zero instances") {
   UNUSED(testCaseHidden);
   const FloatEbmType cutPointsLowerBoundInclusive[] { 1, 2, 2.2, 2.3, 2.5, 2.6, 2.7, 2.8, 2.9 };
   constexpr IntEbmType countCuts = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   constexpr IntEbmType  cInstances = 0;

   Discretize(
      countCuts,
      cutPointsLowerBoundInclusive,
      cInstances,
      nullptr,
      nullptr
   );

   Discretize(
      countCuts,
      cutPointsLowerBoundInclusive,
      cInstances,
      nullptr,
      nullptr
   );
}

TEST_CASE("Discretize, zero cuts, known missing") {
   FloatEbmType singleFeatureValues[] { 0, 0.9, 1, 1.1, 1.9, 2, 2.1, std::numeric_limits<FloatEbmType>::quiet_NaN(), 2.75, 3 };
   const IntEbmType expectedDiscretized[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };

   constexpr size_t cInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   static_assert(cInstances == sizeof(expectedDiscretized) / sizeof(expectedDiscretized[0]),
      "cInstances and expectedDiscretized must be the same length"
      );
   constexpr IntEbmType countCuts = 0;
   IntEbmType singleFeatureDiscretized[cInstances];
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + cInstances, [](const FloatEbmType val) { return std::isnan(val); });

   Discretize(
      countCuts,
      nullptr,
      IntEbmType { cInstances },
      singleFeatureValues,
      singleFeatureDiscretized
   );

   for(size_t i = 0; i < cInstances; ++i) {
      CHECK(expectedDiscretized[i] == singleFeatureDiscretized[i]);
   }
}

TEST_CASE("Discretize, zero cuts, unknown missing") {
   FloatEbmType singleFeatureValues[] { 0, 0.9, 1, 1.1, 1.9, 2, 2.1, std::numeric_limits<FloatEbmType>::quiet_NaN(), 2.75, 3 };
   const IntEbmType expectedDiscretized[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };

   constexpr size_t cInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   static_assert(cInstances == sizeof(expectedDiscretized) / sizeof(expectedDiscretized[0]),
      "cInstances and expectedDiscretized must be the same length"
      );
   constexpr IntEbmType countCuts = 0;
   IntEbmType singleFeatureDiscretized[cInstances];

   Discretize(
      countCuts,
      nullptr,
      IntEbmType { cInstances },
      singleFeatureValues,
      singleFeatureDiscretized
   );

   for(size_t i = 0; i < cInstances; ++i) {
      CHECK(expectedDiscretized[i] == singleFeatureDiscretized[i]);
   }
}

TEST_CASE("Discretize, known missing") {
   const FloatEbmType cutPointsLowerBoundInclusive[] { 1, 2, 2.2, 2.3, 2.5, 2.6, 2.7, 2.8, 2.9 };
   FloatEbmType singleFeatureValues[] { 0, 0.9, 1, 1.1, 1.9, 2, 2.1, std::numeric_limits<FloatEbmType>::quiet_NaN(), 2.75, 3 };
   const IntEbmType expectedDiscretized[] { 0, 0, 1, 1, 1, 2, 2, 10, 7, 9 };

   constexpr size_t cInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   static_assert(cInstances == sizeof(expectedDiscretized) / sizeof(expectedDiscretized[0]),
      "cInstances and expectedDiscretized must be the same length"
      );
   constexpr IntEbmType countCuts = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType singleFeatureDiscretized[cInstances];
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + cInstances, [](const FloatEbmType val) { return std::isnan(val); });

   Discretize(
      countCuts,
      cutPointsLowerBoundInclusive,
      IntEbmType { cInstances },
      singleFeatureValues,
      singleFeatureDiscretized
   );

   for(size_t i = 0; i < cInstances; ++i) {
      CHECK(expectedDiscretized[i] == singleFeatureDiscretized[i]);
   }
}

TEST_CASE("Discretize, unknown missing") {
   const FloatEbmType cutPointsLowerBoundInclusive[] { 1, 2, 2.2, 2.3, 2.5, 2.6, 2.7, 2.8, 2.9 };
   FloatEbmType singleFeatureValues[] { 0, 0.9, 1, 1.1, 1.9, 2, 2.1, std::numeric_limits<FloatEbmType>::quiet_NaN(), 2.75, 3 };
   const IntEbmType expectedDiscretized[] { 0, 0, 1, 1, 1, 2, 2, 10, 7, 9 };

   constexpr size_t cInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   static_assert(cInstances == sizeof(expectedDiscretized) / sizeof(expectedDiscretized[0]),
      "cInstances and expectedDiscretized must be the same length"
      );
   constexpr IntEbmType countCuts = sizeof(cutPointsLowerBoundInclusive) / sizeof(cutPointsLowerBoundInclusive[0]);
   IntEbmType singleFeatureDiscretized[cInstances];

   Discretize(
      countCuts,
      cutPointsLowerBoundInclusive,
      IntEbmType { cInstances },
      singleFeatureValues,
      singleFeatureDiscretized
   );

   for(size_t i = 0; i < cInstances; ++i) {
      CHECK(expectedDiscretized[i] == singleFeatureDiscretized[i]);
   }
}

TEST_CASE("Discretize, increasing lengths") {
   FloatEbmType singleFeatureValues[1];
   IntEbmType singleFeatureDiscretized[1];

   constexpr size_t cCutPointsMax = 259;
   FloatEbmType cutPointsLowerBoundInclusive[cCutPointsMax];
   for(size_t iCutPoint = 0; iCutPoint < cCutPointsMax; ++iCutPoint) {
      cutPointsLowerBoundInclusive[iCutPoint] = static_cast<FloatEbmType>(iCutPoint);
   }
   // this doesn't check 0 cuts, or having missing values
   for(size_t cCutPoints = 1; cCutPoints <= cCutPointsMax; ++cCutPoints) {
      for(size_t iCutPoint = 0; iCutPoint < cCutPoints; ++iCutPoint) {
         // first try it without missing values
         singleFeatureValues[0] = cutPointsLowerBoundInclusive[iCutPoint] - FloatEbmType { 0.5 };
         Discretize(
            cCutPoints,
            cutPointsLowerBoundInclusive,
            1,
            singleFeatureValues,
            singleFeatureDiscretized
         );
         CHECK(singleFeatureDiscretized[0] == static_cast<IntEbmType>(iCutPoint));

         singleFeatureValues[0] = cutPointsLowerBoundInclusive[iCutPoint];
         Discretize(
            cCutPoints,
            cutPointsLowerBoundInclusive,
            1,
            singleFeatureValues,
            singleFeatureDiscretized
         );
         CHECK(singleFeatureDiscretized[0] == static_cast<IntEbmType>(iCutPoint) + 1); // any exact matches are inclusive to the upper bound

         singleFeatureValues[0] = cutPointsLowerBoundInclusive[iCutPoint] + FloatEbmType { 0.5 };
         Discretize(
            cCutPoints,
            cutPointsLowerBoundInclusive,
            1,
            singleFeatureValues,
            singleFeatureDiscretized
         );
         CHECK(singleFeatureDiscretized[0] == static_cast<IntEbmType>(iCutPoint) + 1);

         // now try it indicating that there can be missing values, which should take the 0 value position and bump everything else up
         singleFeatureValues[0] = cutPointsLowerBoundInclusive[iCutPoint] - FloatEbmType { 0.5 };
         Discretize(
            cCutPoints,
            cutPointsLowerBoundInclusive,
            1,
            singleFeatureValues,
            singleFeatureDiscretized
         );
         CHECK(singleFeatureDiscretized[0] == static_cast<IntEbmType>(iCutPoint));

         singleFeatureValues[0] = cutPointsLowerBoundInclusive[iCutPoint];
         Discretize(
            cCutPoints,
            cutPointsLowerBoundInclusive,
            1,
            singleFeatureValues,
            singleFeatureDiscretized
         );
         CHECK(singleFeatureDiscretized[0] == static_cast<IntEbmType>(iCutPoint) + 1); // any exact matches are inclusive to the upper bound

         singleFeatureValues[0] = cutPointsLowerBoundInclusive[iCutPoint] + FloatEbmType { 0.5 };
         Discretize(
            cCutPoints,
            cutPointsLowerBoundInclusive,
            1,
            singleFeatureValues,
            singleFeatureDiscretized
         );
         CHECK(singleFeatureDiscretized[0] == static_cast<IntEbmType>(iCutPoint) + 1);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, 0 instances") {
   constexpr IntEbmType countMaximumBins = 2;
   constexpr IntEbmType countMinimumInstancesPerBin = 1;

   IntEbmType countCutPoints;
   IntEbmType isMissing;
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      0,
      nullptr,
      countMaximumBins,
      countMinimumInstancesPerBin,
      nullptr,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK(EBM_FALSE == isMissing);
   CHECK(FloatEbmType { 0 } == valMin);
   CHECK(FloatEbmType { 0 } == valMax);
   CHECK(0 == countCutPoints);
}

TEST_CASE("GenerateQuantileCutPoints, only missing") {
   constexpr IntEbmType countMaximumBins = 2;
   constexpr IntEbmType countMinimumInstancesPerBin = 1;
   FloatEbmType singleFeatureValues[] { std::numeric_limits<FloatEbmType>::quiet_NaN(), std::numeric_limits<FloatEbmType>::quiet_NaN() };

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      reinterpret_cast<FloatEbmType *>(0x1), // this shouldn't be filled, so it would throw an exception if it did 
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK(EBM_TRUE == isMissing);
   CHECK(FloatEbmType { 0 } == valMin);
   CHECK(FloatEbmType { 0 } == valMax);
   CHECK(0 == countCutPoints);
}

TEST_CASE("GenerateQuantileCutPoints, just one bin") {
   constexpr IntEbmType countMaximumBins = 1;
   constexpr IntEbmType countMinimumInstancesPerBin = 1;
   FloatEbmType singleFeatureValues[] { 1, 2 };

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      nullptr,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 1 } == valMin);
   CHECK(FloatEbmType { 2 } == valMax);
   CHECK(0 == countCutPoints);
}

TEST_CASE("GenerateQuantileCutPoints, too small") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 5 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 5 } == valMin);
   CHECK(FloatEbmType { 5 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, splitable") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 0, 1, 2, 3 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 0 } == valMin);
   CHECK(FloatEbmType { 3 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, splitable (first interior check not splitable)") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 3;
   FloatEbmType singleFeatureValues[] { 0, 1, 5, 5, 7, 8, 9 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 0 } == valMin);
   CHECK(FloatEbmType { 9 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, splitable except middle isn't available") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 3;
   FloatEbmType singleFeatureValues[] { 0, 1, 5, 5, 8, 9 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 0 } == valMin);
   CHECK(FloatEbmType { 9 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 5, 5, 5, 5 };
   const std::vector<FloatEbmType> expectedCutPoints { };

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 5 } == valMin);
   CHECK(FloatEbmType { 5 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 1, 5, 5, 5 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 1 } == valMin);
   CHECK(FloatEbmType { 5 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+right") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 5, 5, 5, 9 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 5 } == valMin);
   CHECK(FloatEbmType { 9 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable+right") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 1, 5, 5, 9 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 1 } == valMin);
   CHECK(FloatEbmType { 9 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+unsplitable") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 4, 4, 6, 6 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 4 } == valMin);
   CHECK(FloatEbmType { 6 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable+unsplitable") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 1, 4, 4, 6, 6 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 1 } == valMin);
   CHECK(FloatEbmType { 6 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+unsplitable+right") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 4, 4, 6, 6, 9 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 4 } == valMin);
   CHECK(FloatEbmType { 9 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+mid+unsplitable") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 4, 4, 5, 6, 6 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 4 } == valMin);
   CHECK(FloatEbmType { 6 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable+mid+unsplitable") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 1, 4, 4, 5, 6, 6 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 1 } == valMin);
   CHECK(FloatEbmType { 6 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+mid+unsplitable+right") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 4, 4, 5, 6, 6, 9 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 4 } == valMin);
   CHECK(FloatEbmType { 9 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+splitable") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 5, 5, 7, 8 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 5 } == valMin);
   CHECK(FloatEbmType { 8 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable+splitable") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 1, 5, 5, 7, 8 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 1 } == valMin);
   CHECK(FloatEbmType { 8 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, splitable+unsplitable") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 2, 3, 5, 5 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 2 } == valMin);
   CHECK(FloatEbmType { 5 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, splitable+unsplitable+right") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 2, 3, 5, 5, 7 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 2 } == valMin);
   CHECK(FloatEbmType { 7 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, splitable+unsplitable+splitable") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 2, 3, 5, 5, 7, 8 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 2 } == valMin);
   CHECK(FloatEbmType { 8 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+splitable+unsplitable") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 2, 2, 4, 6, 8, 8 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 2 } == valMin);
   CHECK(FloatEbmType { 8 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable+splitable+unsplitable") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 1, 2, 2, 4, 6, 8, 8 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 1 } == valMin);
   CHECK(FloatEbmType { 8 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+splitable+unsplitable+right") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 2, 2, 4, 6, 8, 8, 9 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 2 } == valMin);
   CHECK(FloatEbmType { 9 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable+splitable+unsplitable+right") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 1, 2, 2, 4, 6, 8, 8, 9 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 1 } == valMin);
   CHECK(FloatEbmType { 9 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, unsplitable+splitable+unsplitable+splitable") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 1, 1, 2, 3, 5, 5, 7, 8 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 1 } == valMin);
   CHECK(FloatEbmType { 8 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable+splitable+unsplitable+splitable") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 0, 1, 1, 2, 3, 5, 5, 7, 8 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 0 } == valMin);
   CHECK(FloatEbmType { 8 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, splitable+unsplitable+splitable+unsplitable") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 2, 3, 5, 5, 7, 8, 9, 9 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 2 } == valMin);
   CHECK(FloatEbmType { 9 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, splitable+unsplitable+splitable+unsplitable+right") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 2, 3, 5, 5, 7, 8, 9, 9, 10 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 2 } == valMin);
   CHECK(FloatEbmType { 10 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, left+unsplitable+splitable+unsplitable+splitable+unsplitable+splitable+unsplitable+right") {
   constexpr IntEbmType countMaximumBins = 1000;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 1 } == valMin);
   CHECK(FloatEbmType { 9 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, average division sizes that requires the ceiling instead of rounding") {
   // our algorithm makes an internal assumption that we can give each cut point a split.  This is guaranteed if we 
   // make the average length of the equal value long ranges the ceiling of the average instances per bin.  
   // This test stresses that average calculation by having an average bin lenght of 2.2222222222 but if you use 
   // a bin width of 2, then there are 3 cut points that can't get any cuts.  3 cut points means that even if you 
   // don't give the first and last SplittingRanges an actual cut point, which can be reasonalbe since the 
   // first and last SplittingRanges are special in that they may have no long ranges on the tail ends, 
   // you still end up with one or more SplittingRanges that can't have a cut if you don't take the ceiling.

   constexpr IntEbmType countMaximumBins = 27;
   constexpr IntEbmType countMinimumInstancesPerBin = 2;
   FloatEbmType singleFeatureValues[] { 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 
      17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29 };
   const std::vector<FloatEbmType> expectedCutPoints {};

   constexpr IntEbmType countInstances = sizeof(singleFeatureValues) / sizeof(singleFeatureValues[0]);
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];
   IntEbmType countCutPoints;
   IntEbmType isMissing;
   // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
   const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
   FloatEbmType valMin;
   FloatEbmType valMax;

   IntEbmType ret = GenerateQuantileCutPoints(
      randomSeed,
      countInstances,
      singleFeatureValues,
      countMaximumBins,
      countMinimumInstancesPerBin,
      cutPointsLowerBoundInclusive,
      &countCutPoints,
      &isMissing,
      &valMin,
      &valMax
   );
   CHECK(0 == ret);
   CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
   CHECK(FloatEbmType { 0 } == valMin);
   CHECK(FloatEbmType { 29 } == valMax);
   const size_t cCutPoints = static_cast<size_t>(countCutPoints);
   CHECK(expectedCutPoints.size() == cCutPoints);
   if(expectedCutPoints.size() == cCutPoints) {
      for(size_t i = 0; i < cCutPoints; ++i) {
         CHECK_APPROX(expectedCutPoints[i], cutPointsLowerBoundInclusive[i]);
      }
   }
}

TEST_CASE("GenerateQuantileCutPoints, randomized fairness check") {
   RandomStream randomStream(randomSeed);
   if(!randomStream.IsSuccess()) {
      exit(1);
   }

   constexpr IntEbmType countMinimumInstancesPerBin = 1;
   constexpr IntEbmType countInstances = 100;
   FloatEbmType singleFeatureValues[countInstances];

   constexpr IntEbmType randomMaxMax = countInstances - 1; // this doesn't need to be exactly countInstances - 1, but this number gives us chunky sets
   size_t cutHistogram[randomMaxMax];
   constexpr size_t cCutHistogram = sizeof(cutHistogram) / sizeof(cutHistogram[0]);
   // our random numbers can be any numbers from 0 to randomMaxMax (inclusive), which gives us randomMaxMax - 1 possible cut points between them
   static_assert(1 == cCutHistogram % 2, "cutHistogram must have a center value that is perfectly in the middle");

   constexpr IntEbmType countMaximumBins = 10;
   FloatEbmType cutPointsLowerBoundInclusive[countMaximumBins - 1];

   memset(cutHistogram, 0, sizeof(cutHistogram));

   for(int iIteration = 0; iIteration < 100; ++iIteration) {
      for(size_t randomMax = 1; randomMax <= randomMaxMax; randomMax += 2) {
         // since randomMax isn't larger than the number of instances, we'll always be chunky.  This is good for testing range collisions
         for(size_t iInstance = 0; iInstance < countInstances; ++iInstance) {
            bool bMissing = 0 == randomStream.Next(countInstances); // some datasetes will have zero missing values, some will have 1 or more
            size_t iRandom = randomStream.Next(randomMax + 1);
            singleFeatureValues[iInstance] = bMissing ? std::numeric_limits<FloatEbmType>::quiet_NaN() : static_cast<FloatEbmType>(iRandom);
         }
         // do this before calling GenerateQuantileCutPoints, since GenerateQuantileCutPoints modifies singleFeatureValues
         const bool bMissing = std::any_of(singleFeatureValues, singleFeatureValues + countInstances, [](const FloatEbmType val) { return std::isnan(val); });
         IntEbmType countCutPoints;
         IntEbmType isMissing;
         FloatEbmType valMin;
         FloatEbmType valMax;

         IntEbmType ret = GenerateQuantileCutPoints(
            randomSeed + iIteration * (randomMaxMax + 1) + randomMax, // make them all different random seeds
            countInstances,
            singleFeatureValues,
            countMaximumBins,
            countMinimumInstancesPerBin,
            cutPointsLowerBoundInclusive,
            &countCutPoints,
            &isMissing,
            &valMin,
            &valMax
         );
         CHECK(0 == ret);
         CHECK((bMissing ? EBM_TRUE : EBM_FALSE) == isMissing);
         const size_t cCutPoints = static_cast<size_t>(countCutPoints);
         assert(1 == randomMax % 2); // our random numbers need a center value as well
         constexpr size_t iHistogramExactMiddle = cCutHistogram / 2;
         const size_t iCutExactMiddle = randomMax / 2;
         assert(iCutExactMiddle <= iHistogramExactMiddle);
         const size_t iShiftToMiddle = iHistogramExactMiddle - iCutExactMiddle;
         for(size_t iCutPoint = 0; iCutPoint < cCutPoints; ++iCutPoint) {
            const FloatEbmType cutPoint = cutPointsLowerBoundInclusive[iCutPoint];
            // cutPoint can be a number between 0.5 and (randomMax - 0.5)
            const size_t iCut = static_cast<size_t>(std::round(cutPoint - FloatEbmType { 0.5 }));
            const size_t iSymetricCut = iShiftToMiddle + iCut;
            assert(iSymetricCut < cCutHistogram);
            ++cutHistogram[iSymetricCut];
         }
      }
   }
   size_t cBottomTotal = 0;
   size_t cTopTotal = 0;
   for(size_t i = 0; i < (cCutHistogram + 1) / 2; ++i) {
      size_t iBottom = i;
      size_t iTop = cCutHistogram - 1 - i;

      size_t cBottom = cutHistogram[iBottom];
      size_t cTop = cutHistogram[iTop];
      cBottomTotal += cBottom;
      cTopTotal += cTop;
   }
   const size_t cMax = std::max(cBottomTotal, cTopTotal);
   const size_t cMin = std::min(cBottomTotal, cTopTotal);
   const FloatEbmType ratio = static_cast<FloatEbmType>(cMin) / static_cast<FloatEbmType>(cMax);
   CHECK(0.98 <= ratio || 0 == cMax);
}

TEST_CASE("GenerateQuantileCutPoints, chunky randomized check") {
   RandomStream randomStream(randomSeed);
   if(!randomStream.IsSuccess()) {
      exit(1);
   }

   constexpr size_t cMaximumBins = 10;
   constexpr IntEbmType countMinimumInstancesPerBin = 3;
   constexpr size_t cInstances = 100;
   constexpr size_t maxRandomVal = 70;
   const size_t cLongBinLength = static_cast<size_t>(
      std::ceil(static_cast<FloatEbmType>(cInstances) / static_cast<FloatEbmType>(cMaximumBins))
   );
   FloatEbmType singleFeatureValues[cInstances];
   FloatEbmType cutPointsLowerBoundInclusive[cMaximumBins - 1];

   for(int iIteration = 0; iIteration < 30000; ++iIteration) {
      memset(singleFeatureValues, 0, sizeof(singleFeatureValues));

      size_t i = 0;
      size_t cLongRanges = randomStream.Next(6);
      for(size_t iLongRange = 0; iLongRange < cLongRanges; ++iLongRange) {
         size_t cItems = randomStream.Next(cLongBinLength) + cLongBinLength;
         size_t val = randomStream.Next(maxRandomVal) + 1;
         for(size_t iItem = 0; iItem < cItems; ++iItem) {
            singleFeatureValues[i % cInstances] = static_cast<FloatEbmType>(val);
            ++i;
         }
      }
      size_t cShortRanges = randomStream.Next(6);
      for(size_t iShortRange = 0; iShortRange < cShortRanges; ++iShortRange) {
         size_t cItems = randomStream.Next(cLongBinLength);
         size_t val = randomStream.Next(maxRandomVal) + 1;
         for(size_t iItem = 0; iItem < cItems; ++iItem) {
            singleFeatureValues[i % cInstances] = static_cast<FloatEbmType>(val);
            ++i;
         }
      }
      for(size_t iInstance = 0; iInstance < cInstances; ++iInstance) {
         if(0 == singleFeatureValues[iInstance]) {
            singleFeatureValues[iInstance] = static_cast<FloatEbmType>(randomStream.Next(maxRandomVal) + 1);
         }
      }
      IntEbmType countCutPoints;
      IntEbmType isMissing;
      FloatEbmType valMin;
      FloatEbmType valMax;

      std::sort(singleFeatureValues, singleFeatureValues + cInstances);

      IntEbmType ret = GenerateQuantileCutPoints(
         randomSeed + iIteration, // make them all different random seeds
         static_cast<IntEbmType>(cInstances),
         singleFeatureValues,
         static_cast<IntEbmType>(cMaximumBins),
         countMinimumInstancesPerBin,
         cutPointsLowerBoundInclusive,
         &countCutPoints,
         &isMissing,
         &valMin,
         &valMax
      );
      CHECK(0 == ret);
   }
}

TEST_CASE("null validationMetricReturn, boosting, regression") {
   EbmNativeFeatureCombination combinations[1];
   combinations->countFeaturesInCombination = 0;

   PEbmBoosting pEbmBoosting = InitializeBoostingRegression(
      0, 
      nullptr, 
      1, 
      combinations, 
      nullptr, 
      0, 
      nullptr, 
      nullptr, 
      nullptr, 
      0, 
      nullptr, 
      nullptr, 
      nullptr, 
      0, 
      randomSeed,
      nullptr
   );
   const IntEbmType ret = BoostingStep(
      pEbmBoosting, 
      0, 
      k_learningRateDefault, 
      k_countTreeSplitsMaxDefault, 
      k_countInstancesRequiredForChildSplitMinDefault,
      nullptr, 
      nullptr, 
      nullptr
   );
   CHECK(0 == ret);
   FreeBoosting(pEbmBoosting);
}

TEST_CASE("null validationMetricReturn, boosting, binary") {
   EbmNativeFeatureCombination combinations[1];
   combinations->countFeaturesInCombination = 0;

   PEbmBoosting pEbmBoosting = InitializeBoostingClassification(
      2, 
      0, 
      nullptr, 
      1, 
      combinations, 
      nullptr, 
      0, 
      nullptr, 
      nullptr, 
      nullptr, 
      0, 
      nullptr, 
      nullptr, 
      nullptr, 
      0, 
      randomSeed,
      nullptr
   );
   const IntEbmType ret = BoostingStep(
      pEbmBoosting, 
      0, 
      k_learningRateDefault, 
      k_countTreeSplitsMaxDefault, 
      k_countInstancesRequiredForChildSplitMinDefault,
      nullptr, 
      nullptr, 
      nullptr
   );
   CHECK(0 == ret);
   FreeBoosting(pEbmBoosting);
}

TEST_CASE("null validationMetricReturn, boosting, multiclass") {
   EbmNativeFeatureCombination combinations[1];
   combinations->countFeaturesInCombination = 0;

   PEbmBoosting pEbmBoosting = InitializeBoostingClassification(
      3, 
      0, 
      nullptr, 
      1, 
      combinations, 
      nullptr, 
      0, 
      nullptr, 
      nullptr, 
      nullptr, 
      0, 
      nullptr, 
      nullptr, 
      nullptr, 
      0, 
      randomSeed,
      nullptr
   );
   const IntEbmType ret = BoostingStep(
      pEbmBoosting, 
      0, 
      k_learningRateDefault, 
      k_countTreeSplitsMaxDefault, 
      k_countInstancesRequiredForChildSplitMinDefault,
      nullptr, 
      nullptr, 
      nullptr
   );
   CHECK(0 == ret);
   FreeBoosting(pEbmBoosting);
}

TEST_CASE("null interactionScoreReturn, interaction, regression") {
   PEbmInteraction pEbmInteraction = InitializeInteractionRegression(0, nullptr, 0, nullptr, nullptr, nullptr, nullptr);
   const IntEbmType ret = GetInteractionScore(pEbmInteraction, 0, nullptr, k_countInstancesRequiredForChildSplitMinDefault, nullptr);
   CHECK(0 == ret);
   FreeInteraction(pEbmInteraction);
}

TEST_CASE("null interactionScoreReturn, interaction, binary") {
   PEbmInteraction pEbmInteraction = InitializeInteractionClassification(2, 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr);
   const IntEbmType ret = GetInteractionScore(pEbmInteraction, 0, nullptr, k_countInstancesRequiredForChildSplitMinDefault, nullptr);
   CHECK(0 == ret);
   FreeInteraction(pEbmInteraction);
}

TEST_CASE("null interactionScoreReturn, interaction, multiclass") {
   PEbmInteraction pEbmInteraction = InitializeInteractionClassification(3, 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr);
   const IntEbmType ret = GetInteractionScore(pEbmInteraction, 0, nullptr, k_countInstancesRequiredForChildSplitMinDefault, nullptr);
   CHECK(0 == ret);
   FreeInteraction(pEbmInteraction);
}

TEST_CASE("zero learning rate, boosting, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ RegressionInstance(10, {}) });
   test.AddValidationInstances({ RegressionInstance(12, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Boost(iFeatureCombination, {}, {}, 0);
         CHECK_APPROX(validationMetric, 144);
         modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
         CHECK_APPROX(modelValue, 0);

         modelValue = test.GetBestModelPredictorScore(iFeatureCombination, {}, 0);
         CHECK_APPROX(modelValue, 0);
      }
   }
}

TEST_CASE("zero learning rate, boosting, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({});
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ ClassificationInstance(0, {}) });
   test.AddValidationInstances({ ClassificationInstance(0, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Boost(iFeatureCombination, {}, {}, 0);
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

TEST_CASE("zero learning rate, boosting, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({});
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ ClassificationInstance(0, {}) });
   test.AddValidationInstances({ ClassificationInstance(0, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Boost(iFeatureCombination, {}, {}, 0);
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

TEST_CASE("negative learning rate, boosting, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ RegressionInstance(10, {}) });
   test.AddValidationInstances({ RegressionInstance(12, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Boost(iFeatureCombination, {}, {}, -k_learningRateDefault);
         if(0 == iFeatureCombination && 0 == iEpoch) {
            CHECK_APPROX(validationMetric, 146.41);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
            CHECK_APPROX(modelValue, -0.1000000000000000);
         }
         if(0 == iFeatureCombination && 1 == iEpoch) {
            CHECK_APPROX(validationMetric, 148.864401);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
            CHECK_APPROX(modelValue, -0.2010000000000000);
         }
      }
   }
   CHECK_APPROX(validationMetric, 43929458875.235196700295656826033);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 0);
   CHECK_APPROX(modelValue, -209581.55637813677);
}

TEST_CASE("negative learning rate, boosting, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({});
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ ClassificationInstance(0, {}) });
   test.AddValidationInstances({ ClassificationInstance(0, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 50; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Boost(iFeatureCombination, {}, {}, -k_learningRateDefault);
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

   CHECK_APPROX(validationMetric, 1.7158914513238979);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 0);
   CHECK_APPROX(modelValue, 0);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 1);
   CHECK_APPROX(modelValue, 1.5176802847035755);
}

TEST_CASE("negative learning rate, boosting, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({});
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ ClassificationInstance(0, {}) });
   test.AddValidationInstances({ ClassificationInstance(0, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 20; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Boost(iFeatureCombination, {}, {}, -k_learningRateDefault);
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
   CHECK_APPROX(validationMetric, 2.0611718475324357);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 0);
   CHECK_APPROX(modelValue, -0.90755332487264362);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 1);
   CHECK_APPROX(modelValue, 0.32430253082567057);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 2);
   CHECK_APPROX(modelValue, 0.32430253082567057);
}

TEST_CASE("zero countInstancesRequiredForChildSplitMin, boosting, regression") {
   // TODO : call test.Boost many more times in a loop, and verify the output remains the same as previous runs
   // TODO : add classification binary and multiclass versions of this

   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances({
      RegressionInstance(10, { 0 }),
      RegressionInstance(10, { 1 }),
      });
   test.AddValidationInstances({ RegressionInstance(12, { 1 }) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = test.Boost(0, {}, {}, k_learningRateDefault, k_countTreeSplitsMaxDefault, 0);
   CHECK_APPROX(validationMetric, 141.61);
   FloatEbmType modelValue;
   modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
   CHECK_APPROX(modelValue, 0.1000000000000000);
   CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 0));
}

TEST_CASE("zero countTreeSplitsMax, boosting, regression") {
   // TODO : call test.Boost many more times in a loop, and verify the output remains the same as previous runs
   // TODO : add classification binary and multiclass versions of this

   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances({ 
      RegressionInstance(10, { 0 }),
      RegressionInstance(10, { 1 }),
      });
   test.AddValidationInstances({ RegressionInstance(12, { 1 }) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = test.Boost(0, {}, {}, k_learningRateDefault, 0);
   CHECK_APPROX(validationMetric, 141.61);
   FloatEbmType modelValue;
   modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
   CHECK_APPROX(modelValue, 0.1000000000000000);
   CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 0));
}


// TODO: decide what to do with this test
//TEST_CASE("infinite target training set, boosting, regression") {
//   TestApi test = TestApi(k_learningTypeRegression);
//   test.AddFeatures({ Feature(2) });
//   test.AddFeatureCombinations({ { 0 } });
//   test.AddTrainingInstances({ RegressionInstance(FloatEbmType { std::numeric_limits<FloatEbmType>::infinity() }, { 1 }) });
//   test.AddValidationInstances({ RegressionInstance(12, { 1 }) });
//   test.InitializeBoosting();
//
//   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
//      FloatEbmType validationMetric = test.Boost(0);
//      CHECK_APPROX(validationMetric, 12);
//      FloatEbmType modelValue = test.GetCurrentModelValue(0, { 0 }, 0);
//      CHECK_APPROX(modelValue, 0);
//   }
//}



TEST_CASE("Zero training instances, boosting, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances(std::vector<RegressionInstance> {});
   test.AddValidationInstances({ RegressionInstance(12, { 1 }) });
   test.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FloatEbmType validationMetric = test.Boost(0);
      CHECK_APPROX(validationMetric, 144);
      FloatEbmType modelValue;
      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
      CHECK_APPROX(modelValue, 0);
      CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 0));
   }
}

TEST_CASE("Zero training instances, boosting, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances(std::vector<ClassificationInstance> {});
   test.AddValidationInstances({ ClassificationInstance(0, { 1 }) });
   test.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FloatEbmType validationMetric = test.Boost(0);
      CHECK_APPROX(validationMetric, 0.69314718055994529);
      FloatEbmType modelValue;
      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
      CHECK_APPROX(modelValue, 0);
      CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 0));

      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 1);
      CHECK_APPROX(modelValue, 0);
      CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 1));
   }
}

TEST_CASE("Zero training instances, boosting, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances(std::vector<ClassificationInstance> {});
   test.AddValidationInstances({ ClassificationInstance(0, { 1 }) });
   test.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FloatEbmType validationMetric = test.Boost(0);
      CHECK_APPROX(validationMetric, 1.0986122886681098);
      FloatEbmType modelValue;

      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
      CHECK_APPROX(modelValue, 0);
      CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 0));
      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 1);
      CHECK_APPROX(modelValue, 0);
      CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 1));
      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 2);
      CHECK_APPROX(modelValue, 0);
      CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 2));
   }
}

TEST_CASE("Zero validation instances, boosting, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances({ RegressionInstance(10, { 1 }) });
   test.AddValidationInstances(std::vector<RegressionInstance> {});
   test.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FloatEbmType validationMetric = test.Boost(0);
      CHECK(0 == validationMetric);
      // the current model will continue to update, even though we have no way of evaluating it
      FloatEbmType modelValue;
      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
      if(0 == iEpoch) {
         CHECK_APPROX(modelValue, 0.1000000000000000);
      }
      if(1 == iEpoch) {
         CHECK_APPROX(modelValue, 0.1990000000000000);
      }
      CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 0));

      // the best model doesn't update since we don't have any basis to validate any changes
      modelValue = test.GetBestModelPredictorScore(0, { 0 }, 0);
      CHECK_APPROX(modelValue, 0);
      CHECK_APPROX(modelValue, test.GetBestModelPredictorScore(0, { 1 }, 0));
   }
}

TEST_CASE("Zero validation instances, boosting, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances({ ClassificationInstance(0, { 1 }) });
   test.AddValidationInstances(std::vector<ClassificationInstance> {});
   test.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FloatEbmType validationMetric = test.Boost(0);
      CHECK(0 == validationMetric);
      // the current model will continue to update, even though we have no way of evaluating it
      FloatEbmType modelValue;

      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
      CHECK_APPROX(modelValue, 0);
      CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 0));

      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 1);
      if(0 == iEpoch) {
         CHECK_APPROX(modelValue, -0.020000000000000000);
      }
      if(1 == iEpoch) {
         CHECK_APPROX(modelValue, -0.039801986733067563);
      }
      CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 1));

      // the best model doesn't update since we don't have any basis to validate any changes
      modelValue = test.GetBestModelPredictorScore(0, { 0 }, 0);
      CHECK_APPROX(modelValue, 0);
      CHECK_APPROX(modelValue, test.GetBestModelPredictorScore(0, { 1 }, 0));

      modelValue = test.GetBestModelPredictorScore(0, { 0 }, 1);
      CHECK_APPROX(modelValue, 0);
      CHECK_APPROX(modelValue, test.GetBestModelPredictorScore(0, { 1 }, 1));
   }
}

TEST_CASE("Zero validation instances, boosting, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances({ ClassificationInstance(0, { 1 }) });
   test.AddValidationInstances(std::vector<ClassificationInstance> {});
   test.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FloatEbmType validationMetric = test.Boost(0);
      CHECK(0 == validationMetric);
      // the current model will continue to update, even though we have no way of evaluating it
      FloatEbmType modelValue;
      if(0 == iEpoch) {
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
         CHECK_APPROX(modelValue, 0.03000000000000000);
         CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 0));
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 1);
         CHECK_APPROX(modelValue, -0.01500000000000000);
         CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 1));
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 2);
         CHECK_APPROX(modelValue, -0.01500000000000000);
         CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 2));
      }
      if(1 == iEpoch) {
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
         CHECK_APPROX(modelValue, 0.059119949636662006);
         CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 0));
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 1);
         CHECK_APPROX(modelValue, -0.029887518980531450);
         CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 1));
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 2);
         CHECK_APPROX(modelValue, -0.029887518980531450);
         CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 2));
      }
      // the best model doesn't update since we don't have any basis to validate any changes
      modelValue = test.GetBestModelPredictorScore(0, { 0 }, 0);
      CHECK_APPROX(modelValue, 0);
      CHECK_APPROX(modelValue, test.GetBestModelPredictorScore(0, { 1 }, 0));
      modelValue = test.GetBestModelPredictorScore(0, { 0 }, 1);
      CHECK_APPROX(modelValue, 0);
      CHECK_APPROX(modelValue, test.GetBestModelPredictorScore(0, { 1 }, 1));
      modelValue = test.GetBestModelPredictorScore(0, { 0 }, 2);
      CHECK_APPROX(modelValue, 0);
      CHECK_APPROX(modelValue, test.GetBestModelPredictorScore(0, { 1 }, 2));
   }
}

TEST_CASE("Zero interaction instances, interaction, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionInstances(std::vector<RegressionInstance> {});
   test.InitializeInteraction();

   FloatEbmType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("Zero interaction instances, interaction, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionInstances(std::vector<ClassificationInstance> {});
   test.InitializeInteraction();

   FloatEbmType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("Zero interaction instances, interaction, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionInstances(std::vector<ClassificationInstance> {});
   test.InitializeInteraction();

   FloatEbmType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("features with 0 states, boosting") {
   // for there to be zero states, there can't be an training data or testing data since then those would be required to have a value for the state
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(0) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances(std::vector<RegressionInstance> {});
   test.AddValidationInstances(std::vector<RegressionInstance> {});
   test.InitializeBoosting();

   FloatEbmType validationMetric = test.Boost(0);
   CHECK(0 == validationMetric);

   // we're not sure what we'd get back since we aren't allowed to access it, so don't do anything with the return value.  We just want to make sure 
   // calling to get the models doesn't crash
   test.GetBestModelFeatureCombinationRaw(0);
   test.GetCurrentModelFeatureCombinationRaw(0);
}

TEST_CASE("features with 0 states, interaction") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(0) });
   test.AddInteractionInstances(std::vector<RegressionInstance> {});
   test.InitializeInteraction();

   FloatEbmType validationMetric = test.InteractionScore({ 0 });
   CHECK(0 == validationMetric);
}

TEST_CASE("classification with 0 possible target states, boosting") {
   // for there to be zero states, there can't be an training data or testing data since then those would be required to have a value for the state
   TestApi test = TestApi(0);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances(std::vector<ClassificationInstance> {});
   test.AddValidationInstances(std::vector<ClassificationInstance> {});
   test.InitializeBoosting();

   CHECK(nullptr == test.GetBestModelFeatureCombinationRaw(0));
   CHECK(nullptr == test.GetCurrentModelFeatureCombinationRaw(0));

   FloatEbmType validationMetric = test.Boost(0);
   CHECK(0 == validationMetric);

   CHECK(nullptr == test.GetBestModelFeatureCombinationRaw(0));
   CHECK(nullptr == test.GetCurrentModelFeatureCombinationRaw(0));
}

TEST_CASE("classification with 0 possible target states, interaction") {
   TestApi test = TestApi(0);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionInstances(std::vector<ClassificationInstance> {});
   test.InitializeInteraction();

   FloatEbmType validationMetric = test.InteractionScore({ 0 });
   CHECK(0 == validationMetric);
}

TEST_CASE("classification with 1 possible target, boosting") {
   TestApi test = TestApi(1);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureCombinations({ { 0 } });
   test.AddTrainingInstances({ ClassificationInstance(0, { 1 }) });
   test.AddValidationInstances({ ClassificationInstance(0, { 1 }) });
   test.InitializeBoosting();

   CHECK(nullptr == test.GetBestModelFeatureCombinationRaw(0));
   CHECK(nullptr == test.GetCurrentModelFeatureCombinationRaw(0));

   FloatEbmType validationMetric = test.Boost(0);
   CHECK(0 == validationMetric);

   CHECK(nullptr == test.GetBestModelFeatureCombinationRaw(0));
   CHECK(nullptr == test.GetCurrentModelFeatureCombinationRaw(0));
}

TEST_CASE("classification with 1 possible target, interaction") {
   TestApi test = TestApi(1);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionInstances({ ClassificationInstance(0, { 1 }) });
   test.InitializeInteraction();

   FloatEbmType validationMetric = test.InteractionScore({ 0 });
   CHECK(0 == validationMetric);
}

TEST_CASE("features with 1 state in various positions, boosting") {
   TestApi test0 = TestApi(k_learningTypeRegression);
   test0.AddFeatures({ 
      FeatureTest(1),
      FeatureTest(2),
      FeatureTest(2)
      });
   test0.AddFeatureCombinations({ { 0 }, { 1 }, { 2 } });
   test0.AddTrainingInstances({ RegressionInstance(10, { 0, 1, 1 }) });
   test0.AddValidationInstances({ RegressionInstance(12, { 0, 1, 1 }) });
   test0.InitializeBoosting();

   TestApi test1 = TestApi(k_learningTypeRegression);
   test1.AddFeatures({
      FeatureTest(2),
      FeatureTest(1),
      FeatureTest(2)
      });
   test1.AddFeatureCombinations({ { 0 }, { 1 }, { 2 } });
   test1.AddTrainingInstances({ RegressionInstance(10, { 1, 0, 1 }) });
   test1.AddValidationInstances({ RegressionInstance(12, { 1, 0, 1 }) });
   test1.InitializeBoosting();

   TestApi test2 = TestApi(k_learningTypeRegression);
   test2.AddFeatures({
      FeatureTest(2),
      FeatureTest(2),
      FeatureTest(1)
      });
   test2.AddFeatureCombinations({ { 0 }, { 1 }, { 2 } });
   test2.AddTrainingInstances({ RegressionInstance(10, { 1, 1, 0 }) });
   test2.AddValidationInstances({ RegressionInstance(12, { 1, 1, 0 }) });
   test2.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FloatEbmType validationMetric00 = test0.Boost(0);
      FloatEbmType validationMetric10 = test1.Boost(1);
      CHECK_APPROX(validationMetric00, validationMetric10);
      FloatEbmType validationMetric20 = test2.Boost(2);
      CHECK_APPROX(validationMetric00, validationMetric20);

      FloatEbmType validationMetric01 = test0.Boost(1);
      FloatEbmType validationMetric11 = test1.Boost(2);
      CHECK_APPROX(validationMetric01, validationMetric11);
      FloatEbmType validationMetric21 = test2.Boost(0);
      CHECK_APPROX(validationMetric01, validationMetric21);

      FloatEbmType validationMetric02 = test0.Boost(2);
      FloatEbmType validationMetric12 = test1.Boost(0);
      CHECK_APPROX(validationMetric02, validationMetric12);
      FloatEbmType validationMetric22 = test2.Boost(1);
      CHECK_APPROX(validationMetric02, validationMetric22);

      FloatEbmType modelValue000 = test0.GetCurrentModelPredictorScore(0, { 0 }, 0);
      FloatEbmType modelValue010 = test0.GetCurrentModelPredictorScore(1, { 0 }, 0);
      FloatEbmType modelValue011 = test0.GetCurrentModelPredictorScore(1, { 1 }, 0);
      FloatEbmType modelValue020 = test0.GetCurrentModelPredictorScore(2, { 0 }, 0);
      FloatEbmType modelValue021 = test0.GetCurrentModelPredictorScore(2, { 1 }, 0);

      FloatEbmType modelValue110 = test1.GetCurrentModelPredictorScore(1, { 0 }, 0);
      FloatEbmType modelValue120 = test1.GetCurrentModelPredictorScore(2, { 0 }, 0);
      FloatEbmType modelValue121 = test1.GetCurrentModelPredictorScore(2, { 1 }, 0);
      FloatEbmType modelValue100 = test1.GetCurrentModelPredictorScore(0, { 0 }, 0);
      FloatEbmType modelValue101 = test1.GetCurrentModelPredictorScore(0, { 1 }, 0);
      CHECK_APPROX(modelValue110, modelValue000);
      CHECK_APPROX(modelValue120, modelValue010);
      CHECK_APPROX(modelValue121, modelValue011);
      CHECK_APPROX(modelValue100, modelValue020);
      CHECK_APPROX(modelValue101, modelValue021);

      FloatEbmType modelValue220 = test2.GetCurrentModelPredictorScore(2, { 0 }, 0);
      FloatEbmType modelValue200 = test2.GetCurrentModelPredictorScore(0, { 0 }, 0);
      FloatEbmType modelValue201 = test2.GetCurrentModelPredictorScore(0, { 1 }, 0);
      FloatEbmType modelValue210 = test2.GetCurrentModelPredictorScore(1, { 0 }, 0);
      FloatEbmType modelValue211 = test2.GetCurrentModelPredictorScore(1, { 1 }, 0);
      CHECK_APPROX(modelValue220, modelValue000);
      CHECK_APPROX(modelValue200, modelValue010);
      CHECK_APPROX(modelValue201, modelValue011);
      CHECK_APPROX(modelValue210, modelValue020);
      CHECK_APPROX(modelValue211, modelValue021);
   }
}

TEST_CASE("zero FeatureCombinations, boosting, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddFeatureCombinations({});
   test.AddTrainingInstances({ RegressionInstance(10, {}) });
   test.AddValidationInstances({ RegressionInstance(12, {}) });
   test.InitializeBoosting();

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just crash/assert if there was a problem
   // boosting isn't legal since we'd need to specify an featureCombination index
}

TEST_CASE("zero FeatureCombinations, boosting, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({});
   test.AddFeatureCombinations({});
   test.AddTrainingInstances({ ClassificationInstance(1, {}) });
   test.AddValidationInstances({ ClassificationInstance(1, {}) });
   test.InitializeBoosting();

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just crash/assert if there was a problem
   // boosting isn't legal since we'd need to specify an featureCombination index
}

TEST_CASE("zero FeatureCombinations, boosting, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({});
   test.AddFeatureCombinations({});
   test.AddTrainingInstances({ ClassificationInstance(2, {}) });
   test.AddValidationInstances({ ClassificationInstance(2, {}) });
   test.InitializeBoosting();

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just crash/assert if there was a problem
   // boosting isn't legal since we'd need to specify an featureCombination index
}

TEST_CASE("FeatureCombination with zero features, boosting, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ RegressionInstance(10, {}) });
   test.AddValidationInstances({ RegressionInstance(12, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Boost(iFeatureCombination);
         if(0 == iFeatureCombination && 0 == iEpoch) {
            CHECK_APPROX(validationMetric, 141.61);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
            CHECK_APPROX(modelValue, 0.1000000000000000);
         }
         if(0 == iFeatureCombination && 1 == iEpoch) {
            CHECK_APPROX(validationMetric, 139.263601);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
            CHECK_APPROX(modelValue, 0.1990000000000000);
         }
      }
   }
   CHECK_APPROX(validationMetric, 4.001727036272099502004735302456);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 0);
   CHECK_APPROX(modelValue, 9.9995682875258822);
}

TEST_CASE("FeatureCombination with zero features, boosting, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({});
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ ClassificationInstance(0, {}) });
   test.AddValidationInstances({ ClassificationInstance(0, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Boost(iFeatureCombination);
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

TEST_CASE("FeatureCombination with zero features, boosting, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ });
   test.AddFeatureCombinations({ {} });
   test.AddTrainingInstances({ ClassificationInstance(0, {}) });
   test.AddValidationInstances({ ClassificationInstance(0, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureCombination = 0; iFeatureCombination < test.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         validationMetric = test.Boost(iFeatureCombination);
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
   FloatEbmType metricReturn = test.InteractionScore({});
   CHECK(0 == metricReturn);
}

TEST_CASE("FeatureCombination with zero features, interaction, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({});
   test.AddInteractionInstances({ ClassificationInstance(0, {}) });
   test.InitializeInteraction();
   FloatEbmType metricReturn = test.InteractionScore({});
   CHECK(0 == metricReturn);
}

TEST_CASE("FeatureCombination with zero features, interaction, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({});
   test.AddInteractionInstances({ ClassificationInstance(0, {}) });
   test.InitializeInteraction();
   FloatEbmType metricReturn = test.InteractionScore({});
   CHECK(0 == metricReturn);
}

TEST_CASE("FeatureCombination with one feature with one or two states is the exact same as zero FeatureCombinations, boosting, regression") {
   TestApi testZeroFeaturesInCombination = TestApi(k_learningTypeRegression);
   testZeroFeaturesInCombination.AddFeatures({});
   testZeroFeaturesInCombination.AddFeatureCombinations({ {} });
   testZeroFeaturesInCombination.AddTrainingInstances({ RegressionInstance(10, {}) });
   testZeroFeaturesInCombination.AddValidationInstances({ RegressionInstance(12, {}) });
   testZeroFeaturesInCombination.InitializeBoosting();

   TestApi testOneState = TestApi(k_learningTypeRegression);
   testOneState.AddFeatures({ FeatureTest(1) });
   testOneState.AddFeatureCombinations({ { 0 } });
   testOneState.AddTrainingInstances({ RegressionInstance(10, { 0 }) });
   testOneState.AddValidationInstances({ RegressionInstance(12, { 0 }) });
   testOneState.InitializeBoosting();

   TestApi testTwoStates = TestApi(k_learningTypeRegression);
   testTwoStates.AddFeatures({ FeatureTest(2) });
   testTwoStates.AddFeatureCombinations({ { 0 } });
   testTwoStates.AddTrainingInstances({ RegressionInstance(10, { 1 }) });
   testTwoStates.AddValidationInstances({ RegressionInstance(12, { 1 }) });
   testTwoStates.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroFeaturesInCombination.GetFeatureCombinationsCount() == testOneState.GetFeatureCombinationsCount());
      assert(testZeroFeaturesInCombination.GetFeatureCombinationsCount() == testTwoStates.GetFeatureCombinationsCount());
      for(size_t iFeatureCombination = 0; iFeatureCombination < testZeroFeaturesInCombination.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         FloatEbmType validationMetricZeroFeaturesInCombination = testZeroFeaturesInCombination.Boost(iFeatureCombination);
         FloatEbmType validationMetricOneState = testOneState.Boost(iFeatureCombination);
         CHECK_APPROX(validationMetricZeroFeaturesInCombination, validationMetricOneState);
         FloatEbmType validationMetricTwoStates = testTwoStates.Boost(iFeatureCombination);
         CHECK_APPROX(validationMetricZeroFeaturesInCombination, validationMetricTwoStates);

         FloatEbmType modelValueZeroFeaturesInCombination = testZeroFeaturesInCombination.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
         FloatEbmType modelValueOneState = testOneState.GetCurrentModelPredictorScore(iFeatureCombination, { 0 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInCombination, modelValueOneState);
         FloatEbmType modelValueTwoStates = testTwoStates.GetCurrentModelPredictorScore(iFeatureCombination, { 1 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInCombination, modelValueTwoStates);
      }
   }
}

TEST_CASE("FeatureCombination with one feature with one or two states is the exact same as zero FeatureCombinations, boosting, binary") {
   TestApi testZeroFeaturesInCombination = TestApi(2, 0);
   testZeroFeaturesInCombination.AddFeatures({});
   testZeroFeaturesInCombination.AddFeatureCombinations({ {} });
   testZeroFeaturesInCombination.AddTrainingInstances({ ClassificationInstance(0, {}) });
   testZeroFeaturesInCombination.AddValidationInstances({ ClassificationInstance(0, {}) });
   testZeroFeaturesInCombination.InitializeBoosting();

   TestApi testOneState = TestApi(2, 0);
   testOneState.AddFeatures({ FeatureTest(1) });
   testOneState.AddFeatureCombinations({ { 0 } });
   testOneState.AddTrainingInstances({ ClassificationInstance(0, { 0 }) });
   testOneState.AddValidationInstances({ ClassificationInstance(0, { 0 }) });
   testOneState.InitializeBoosting();

   TestApi testTwoStates = TestApi(2, 0);
   testTwoStates.AddFeatures({ FeatureTest(2) });
   testTwoStates.AddFeatureCombinations({ { 0 } });
   testTwoStates.AddTrainingInstances({ ClassificationInstance(0, { 1 }) });
   testTwoStates.AddValidationInstances({ ClassificationInstance(0, { 1 }) });
   testTwoStates.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroFeaturesInCombination.GetFeatureCombinationsCount() == testOneState.GetFeatureCombinationsCount());
      assert(testZeroFeaturesInCombination.GetFeatureCombinationsCount() == testTwoStates.GetFeatureCombinationsCount());
      for(size_t iFeatureCombination = 0; iFeatureCombination < testZeroFeaturesInCombination.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         FloatEbmType validationMetricZeroFeaturesInCombination = testZeroFeaturesInCombination.Boost(iFeatureCombination);
         FloatEbmType validationMetricOneState = testOneState.Boost(iFeatureCombination);
         CHECK_APPROX(validationMetricZeroFeaturesInCombination, validationMetricOneState);
         FloatEbmType validationMetricTwoStates = testTwoStates.Boost(iFeatureCombination);
         CHECK_APPROX(validationMetricZeroFeaturesInCombination, validationMetricTwoStates);

         FloatEbmType modelValueZeroFeaturesInCombination0 = testZeroFeaturesInCombination.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
         FloatEbmType modelValueOneState0 = testOneState.GetCurrentModelPredictorScore(iFeatureCombination, { 0 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInCombination0, modelValueOneState0);
         FloatEbmType modelValueTwoStates0 = testTwoStates.GetCurrentModelPredictorScore(iFeatureCombination, { 1 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInCombination0, modelValueTwoStates0);

         FloatEbmType modelValueZeroFeaturesInCombination1 = testZeroFeaturesInCombination.GetCurrentModelPredictorScore(iFeatureCombination, {}, 1);
         FloatEbmType modelValueOneState1 = testOneState.GetCurrentModelPredictorScore(iFeatureCombination, { 0 }, 1);
         CHECK_APPROX(modelValueZeroFeaturesInCombination1, modelValueOneState1);
         FloatEbmType modelValueTwoStates1 = testTwoStates.GetCurrentModelPredictorScore(iFeatureCombination, { 1 }, 1);
         CHECK_APPROX(modelValueZeroFeaturesInCombination1, modelValueTwoStates1);
      }
   }
}

TEST_CASE("FeatureCombination with one feature with one or two states is the exact same as zero FeatureCombinations, boosting, multiclass") {
   TestApi testZeroFeaturesInCombination = TestApi(3);
   testZeroFeaturesInCombination.AddFeatures({});
   testZeroFeaturesInCombination.AddFeatureCombinations({ {} });
   testZeroFeaturesInCombination.AddTrainingInstances({ ClassificationInstance(0, {}) });
   testZeroFeaturesInCombination.AddValidationInstances({ ClassificationInstance(0, {}) });
   testZeroFeaturesInCombination.InitializeBoosting();

   TestApi testOneState = TestApi(3);
   testOneState.AddFeatures({ FeatureTest(1) });
   testOneState.AddFeatureCombinations({ { 0 } });
   testOneState.AddTrainingInstances({ ClassificationInstance(0, { 0 }) });
   testOneState.AddValidationInstances({ ClassificationInstance(0, { 0 }) });
   testOneState.InitializeBoosting();

   TestApi testTwoStates = TestApi(3);
   testTwoStates.AddFeatures({ FeatureTest(2) });
   testTwoStates.AddFeatureCombinations({ { 0 } });
   testTwoStates.AddTrainingInstances({ ClassificationInstance(0, { 1 }) });
   testTwoStates.AddValidationInstances({ ClassificationInstance(0, { 1 }) });
   testTwoStates.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroFeaturesInCombination.GetFeatureCombinationsCount() == testOneState.GetFeatureCombinationsCount());
      assert(testZeroFeaturesInCombination.GetFeatureCombinationsCount() == testTwoStates.GetFeatureCombinationsCount());
      for(size_t iFeatureCombination = 0; iFeatureCombination < testZeroFeaturesInCombination.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         FloatEbmType validationMetricZeroFeaturesInCombination = testZeroFeaturesInCombination.Boost(iFeatureCombination);
         FloatEbmType validationMetricOneState = testOneState.Boost(iFeatureCombination);
         CHECK_APPROX(validationMetricZeroFeaturesInCombination, validationMetricOneState);
         FloatEbmType validationMetricTwoStates = testTwoStates.Boost(iFeatureCombination);
         CHECK_APPROX(validationMetricZeroFeaturesInCombination, validationMetricTwoStates);

         FloatEbmType modelValueZeroFeaturesInCombination0 = testZeroFeaturesInCombination.GetCurrentModelPredictorScore(iFeatureCombination, {}, 0);
         FloatEbmType modelValueOneState0 = testOneState.GetCurrentModelPredictorScore(iFeatureCombination, { 0 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInCombination0, modelValueOneState0);
         FloatEbmType modelValueTwoStates0 = testTwoStates.GetCurrentModelPredictorScore(iFeatureCombination, { 1 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInCombination0, modelValueTwoStates0);

         FloatEbmType modelValueZeroFeaturesInCombination1 = testZeroFeaturesInCombination.GetCurrentModelPredictorScore(iFeatureCombination, {}, 1);
         FloatEbmType modelValueOneState1 = testOneState.GetCurrentModelPredictorScore(iFeatureCombination, { 0 }, 1);
         CHECK_APPROX(modelValueZeroFeaturesInCombination1, modelValueOneState1);
         FloatEbmType modelValueTwoStates1 = testTwoStates.GetCurrentModelPredictorScore(iFeatureCombination, { 1 }, 1);
         CHECK_APPROX(modelValueZeroFeaturesInCombination1, modelValueTwoStates1);

         FloatEbmType modelValueZeroFeaturesInCombination2 = testZeroFeaturesInCombination.GetCurrentModelPredictorScore(iFeatureCombination, {}, 2);
         FloatEbmType modelValueOneState2 = testOneState.GetCurrentModelPredictorScore(iFeatureCombination, { 0 }, 2);
         CHECK_APPROX(modelValueZeroFeaturesInCombination2, modelValueOneState2);
         FloatEbmType modelValueTwoStates2 = testTwoStates.GetCurrentModelPredictorScore(iFeatureCombination, { 1 }, 2);
         CHECK_APPROX(modelValueZeroFeaturesInCombination2, modelValueTwoStates2);
      }
   }
}

TEST_CASE("3 dimensional featureCombination with one dimension reduced in different ways, boosting, regression") {
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
   test0.InitializeBoosting();

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
   test1.InitializeBoosting();

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
   test2.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(test0.GetFeatureCombinationsCount() == test1.GetFeatureCombinationsCount());
      assert(test0.GetFeatureCombinationsCount() == test2.GetFeatureCombinationsCount());
      for(size_t iFeatureCombination = 0; iFeatureCombination < test0.GetFeatureCombinationsCount(); ++iFeatureCombination) {
         FloatEbmType validationMetric0 = test0.Boost(iFeatureCombination);
         FloatEbmType validationMetric1 = test1.Boost(iFeatureCombination);
         CHECK_APPROX(validationMetric0, validationMetric1);
         FloatEbmType validationMetric2 = test2.Boost(iFeatureCombination);
         CHECK_APPROX(validationMetric0, validationMetric2);

         FloatEbmType modelValue01 = test0.GetCurrentModelPredictorScore(iFeatureCombination, { 0, 0, 0 }, 0);
         FloatEbmType modelValue02 = test0.GetCurrentModelPredictorScore(iFeatureCombination, { 0, 0, 1 }, 0);
         FloatEbmType modelValue03 = test0.GetCurrentModelPredictorScore(iFeatureCombination, { 0, 1, 0 }, 0);
         FloatEbmType modelValue04 = test0.GetCurrentModelPredictorScore(iFeatureCombination, { 0, 1, 1 }, 0);

         FloatEbmType modelValue11 = test1.GetCurrentModelPredictorScore(iFeatureCombination, { 0, 0, 0 }, 0);
         FloatEbmType modelValue12 = test1.GetCurrentModelPredictorScore(iFeatureCombination, { 1, 0, 0 }, 0);
         FloatEbmType modelValue13 = test1.GetCurrentModelPredictorScore(iFeatureCombination, { 0, 0, 1 }, 0);
         FloatEbmType modelValue14 = test1.GetCurrentModelPredictorScore(iFeatureCombination, { 1, 0, 1 }, 0);
         CHECK_APPROX(modelValue11, modelValue01);
         CHECK_APPROX(modelValue12, modelValue02);
         CHECK_APPROX(modelValue13, modelValue03);
         CHECK_APPROX(modelValue14, modelValue04);

         FloatEbmType modelValue21 = test2.GetCurrentModelPredictorScore(iFeatureCombination, { 0, 0, 0 }, 0);
         FloatEbmType modelValue22 = test2.GetCurrentModelPredictorScore(iFeatureCombination, { 0, 1, 0 }, 0);
         FloatEbmType modelValue23 = test2.GetCurrentModelPredictorScore(iFeatureCombination, { 1, 0, 0 }, 0);
         FloatEbmType modelValue24 = test2.GetCurrentModelPredictorScore(iFeatureCombination, { 1, 1, 0 }, 0);
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
   FloatEbmType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("FeatureCombination with one feature with one state, interaction, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({ FeatureTest(1) });
   test.AddInteractionInstances({ ClassificationInstance(0, { 0 }) });
   test.InitializeInteraction();
   FloatEbmType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("FeatureCombination with one feature with one state, interaction, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(1) });
   test.AddInteractionInstances({ ClassificationInstance(0, { 0 }) });
   test.InitializeInteraction();
   FloatEbmType metricReturn = test.InteractionScore({0});
   CHECK(0 == metricReturn);
}

TEST_CASE("Test Rehydration, boosting, regression") {
   TestApi testContinuous = TestApi(k_learningTypeRegression);
   testContinuous.AddFeatures({});
   testContinuous.AddFeatureCombinations({ {} });
   testContinuous.AddTrainingInstances({ RegressionInstance(10, {}) });
   testContinuous.AddValidationInstances({ RegressionInstance(12, {}) });
   testContinuous.InitializeBoosting();

   FloatEbmType model0 = 0;

   FloatEbmType validationMetricContinuous;
   FloatEbmType modelValueContinuous;
   FloatEbmType validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(k_learningTypeRegression);
      testRestart.AddFeatures({});
      testRestart.AddFeatureCombinations({ {} });
      testRestart.AddTrainingInstances({ RegressionInstance(10, {}, model0) });
      testRestart.AddValidationInstances({ RegressionInstance(12, {}, model0) });
      testRestart.InitializeBoosting();

      validationMetricRestart = testRestart.Boost(0);
      validationMetricContinuous = testContinuous.Boost(0);
      CHECK_APPROX(validationMetricContinuous, validationMetricRestart);

      modelValueContinuous = testContinuous.GetCurrentModelPredictorScore(0, {}, 0);
      model0 += testRestart.GetCurrentModelPredictorScore(0, {}, 0);
      CHECK_APPROX(modelValueContinuous, model0);
   }
}

TEST_CASE("Test Rehydration, boosting, binary") {
   TestApi testContinuous = TestApi(2, 0);
   testContinuous.AddFeatures({});
   testContinuous.AddFeatureCombinations({ {} });
   testContinuous.AddTrainingInstances({ ClassificationInstance(0, {}) });
   testContinuous.AddValidationInstances({ ClassificationInstance(0, {}) });
   testContinuous.InitializeBoosting();

   FloatEbmType model0 = 0;
   FloatEbmType model1 = 0;

   FloatEbmType validationMetricContinuous;
   FloatEbmType modelValueContinuous;
   FloatEbmType validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(2, 0);
      testRestart.AddFeatures({});
      testRestart.AddFeatureCombinations({ {} });
      testRestart.AddTrainingInstances({ ClassificationInstance(0, {}, { model0, model1 }) });
      testRestart.AddValidationInstances({ ClassificationInstance(0, {}, { model0, model1 }) });
      testRestart.InitializeBoosting();

      validationMetricRestart = testRestart.Boost(0);
      validationMetricContinuous = testContinuous.Boost(0);
      CHECK_APPROX(validationMetricContinuous, validationMetricRestart);

      modelValueContinuous = testContinuous.GetCurrentModelPredictorScore(0, {}, 0);
      model0 += testRestart.GetCurrentModelPredictorScore(0, {}, 0);
      CHECK_APPROX(modelValueContinuous, model0);

      modelValueContinuous = testContinuous.GetCurrentModelPredictorScore(0, {}, 1);
      model1 += testRestart.GetCurrentModelPredictorScore(0, {}, 1);
      CHECK_APPROX(modelValueContinuous, model1);
   }
}

TEST_CASE("Test Rehydration, boosting, multiclass") {
   TestApi testContinuous = TestApi(3);
   testContinuous.AddFeatures({});
   testContinuous.AddFeatureCombinations({ {} });
   testContinuous.AddTrainingInstances({ ClassificationInstance(0, {}) });
   testContinuous.AddValidationInstances({ ClassificationInstance(0, {}) });
   testContinuous.InitializeBoosting();

   FloatEbmType model0 = 0;
   FloatEbmType model1 = 0;
   FloatEbmType model2 = 0;

   FloatEbmType validationMetricContinuous;
   FloatEbmType modelValueContinuous;
   FloatEbmType validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(3);
      testRestart.AddFeatures({});
      testRestart.AddFeatureCombinations({ {} });
      testRestart.AddTrainingInstances({ ClassificationInstance(0, {}, { model0, model1, model2 }) });
      testRestart.AddValidationInstances({ ClassificationInstance(0, {}, { model0, model1, model2 }) });
      testRestart.InitializeBoosting();

      validationMetricRestart = testRestart.Boost(0);
      validationMetricContinuous = testContinuous.Boost(0);
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

TEST_CASE("Test data bit packing extremes, boosting, regression") {
   for(size_t exponentialBins = 1; exponentialBins < 10; ++exponentialBins) {
      IntEbmType exponential = static_cast<IntEbmType>(std::pow(2, exponentialBins));
      // if we set the number of bins to be exponential, then we'll be just under a bit packing boundary.  4 bins means bits packs 00, 01, 10, and 11
      for(IntEbmType iRange = IntEbmType { -1 }; iRange <= IntEbmType { 1 }; ++iRange) {
         IntEbmType cBins = exponential + iRange; // check one less than the tight fit, the tight fit, and one above the tight fit
         // try everything from 0 instances to 65 instances because for bitpacks with 1 bit, we can have up to 64 packed into a single data value on a 
         // 64 bit machine
         for(size_t cInstances = 1; cInstances < 66; ++cInstances) {
            TestApi test = TestApi(k_learningTypeRegression);
            test.AddFeatures({ FeatureTest(cBins) });
            test.AddFeatureCombinations({ { 0 } });

            std::vector<RegressionInstance> trainingInstances;
            std::vector<RegressionInstance> validationInstances;
            for(size_t iInstance = 0; iInstance < cInstances; ++iInstance) {
               trainingInstances.push_back(RegressionInstance(7, { cBins - 1 }));
               validationInstances.push_back(RegressionInstance(8, { cBins - 1 }));
            }
            test.AddTrainingInstances(trainingInstances);
            test.AddValidationInstances(validationInstances);
            test.InitializeBoosting();

            FloatEbmType validationMetric = test.Boost(0);
            CHECK_APPROX(validationMetric, 62.8849);
            FloatEbmType modelValue = test.GetCurrentModelPredictorScore(0, { static_cast<size_t>(cBins - 1) }, 0);
            CHECK_APPROX(modelValue, 0.07);
         }
      }
   }
}

TEST_CASE("Test data bit packing extremes, boosting, binary") {
   for(size_t exponentialBins = 1; exponentialBins < 10; ++exponentialBins) {
      IntEbmType exponential = static_cast<IntEbmType>(std::pow(2, exponentialBins));
      // if we set the number of bins to be exponential, then we'll be just under a bit packing boundary.  4 bins means bits packs 00, 01, 10, and 11
      for(IntEbmType iRange = IntEbmType { -1 }; iRange <= IntEbmType { 1 }; ++iRange) {
         IntEbmType cBins = exponential + iRange; // check one less than the tight fit, the tight fit, and one above the tight fit
         // try everything from 0 instances to 65 instances because for bitpacks with 1 bit, we can have up to 64 packed into a single data value on 
         // a 64 bit machine
         for(size_t cInstances = 1; cInstances < 66; ++cInstances) {
            TestApi test = TestApi(2, 0);
            test.AddFeatures({ FeatureTest(cBins) });
            test.AddFeatureCombinations({ { 0 } });

            std::vector<ClassificationInstance> trainingInstances;
            std::vector<ClassificationInstance> validationInstances;
            for(size_t iInstance = 0; iInstance < cInstances; ++iInstance) {
               trainingInstances.push_back(ClassificationInstance(0, { cBins - 1 }));
               validationInstances.push_back(ClassificationInstance(1, { cBins - 1 }));
            }
            test.AddTrainingInstances(trainingInstances);
            test.AddValidationInstances(validationInstances);
            test.InitializeBoosting();

            FloatEbmType validationMetric = test.Boost(0);
            CHECK_APPROX(validationMetric, 0.70319717972663420);

            FloatEbmType modelValue;
            modelValue = test.GetCurrentModelPredictorScore(0, { static_cast<size_t>(cBins - 1) }, 0);
            CHECK_APPROX(modelValue, 0);
            modelValue = test.GetCurrentModelPredictorScore(0, { static_cast<size_t>(cBins - 1) }, 1);
            CHECK_APPROX(modelValue, -0.02);
         }
      }
   }
}

TEST_CASE("Test data bit packing extremes, interaction, regression") {
   for(size_t exponentialBins = 1; exponentialBins < 10; ++exponentialBins) {
      IntEbmType exponential = static_cast<IntEbmType>(std::pow(2, exponentialBins));
      // if we set the number of bins to be exponential, then we'll be just under a bit packing boundary.  4 bins means bits packs 00, 01, 10, and 11
      for(IntEbmType iRange = IntEbmType { -1 }; iRange <= IntEbmType { 1 }; ++iRange) {
         IntEbmType cBins = exponential + iRange; // check one less than the tight fit, the tight fit, and one above the tight fit
         // try everything from 0 instances to 65 instances because for bitpacks with 1 bit, we can have up to 64 packed into a single data value on 
         // a 64 bit machine
         for(size_t cInstances = 1; cInstances < 66; ++cInstances) {
            TestApi test = TestApi(k_learningTypeRegression);
            test.AddFeatures({ FeatureTest(2), FeatureTest(cBins) });

            std::vector<RegressionInstance> instances;
            for(size_t iInstance = 0; iInstance < cInstances; ++iInstance) {
               instances.push_back(RegressionInstance(7, { 0, cBins - 1 }));
            }
            test.AddInteractionInstances(instances);
            test.InitializeInteraction();

            FloatEbmType metric = test.InteractionScore({ 0, 1 });
            CHECK_APPROX(metric, 0);
         }
      }
   }
}

TEST_CASE("Test data bit packing extremes, interaction, binary") {
   for(size_t exponentialBins = 1; exponentialBins < 10; ++exponentialBins) {
      IntEbmType exponential = static_cast<IntEbmType>(std::pow(2, exponentialBins));
      // if we set the number of bins to be exponential, then we'll be just under a bit packing boundary.  4 bins means bits packs 00, 01, 10, and 11
      for(IntEbmType iRange = IntEbmType { -1 }; iRange <= IntEbmType { 1 }; ++iRange) {
         IntEbmType cBins = exponential + iRange; // check one less than the tight fit, the tight fit, and one above the tight fit
         // try everything from 0 instances to 65 instances because for bitpacks with 1 bit, we can have up to 64 packed into a single data value on 
         // a 64 bit machine
         for(size_t cInstances = 1; cInstances < 66; ++cInstances) {
            TestApi test = TestApi(2, 0);
            test.AddFeatures({ FeatureTest(2), FeatureTest(cBins) });

            std::vector<ClassificationInstance> instances;
            for(size_t iInstance = 0; iInstance < cInstances; ++iInstance) {
               instances.push_back(ClassificationInstance(1, { 0, cBins - 1 }));
            }
            test.AddInteractionInstances(instances);
            test.InitializeInteraction();

            FloatEbmType metric = test.InteractionScore({ 0, 1 });

            CHECK_APPROX(metric, 0);
         }
      }
   }
}


void EBM_NATIVE_CALLING_CONVENTION LogMessage(signed char traceLevel, const char * message) {
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
