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
#include "EbmNativeTest.h"

extern void FAILED(TestCaseHidden * const pTestCaseHidden) {
   pTestCaseHidden->m_bPassed = false;
}

static int g_countEqualityFailures = 0;

extern std::vector<TestCaseHidden> & GetAllTestsHidden() {
   // putting this static variable inside a function avoids the static initialization order problem 
   static std::vector<TestCaseHidden> g_allTestsHidden;
   return g_allTestsHidden;
}

extern int RegisterTestHidden(const TestCaseHidden & testCaseHidden) {
   GetAllTestsHidden().push_back(testCaseHidden);
   return 0;
}

extern bool IsApproxEqual(const double value, const double expected, const double percentage) {
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

const FloatEbmType * TestApi::GetPredictorScores(
   const size_t iFeatureGroup,
   const FloatEbmType * const pModelFeatureGroup,
   const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures
) const {
   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }
   const size_t cVectorLength = GetVectorLength(m_learningTypeOrCountTargetClasses);

   if(m_countBinsByFeatureGroup.size() <= iFeatureGroup) {
      exit(1);
   }
   const std::vector<size_t> countBins = m_countBinsByFeatureGroup[iFeatureGroup];

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
   return &pModelFeatureGroup[iValue];
}

FloatEbmType TestApi::GetPredictorScore(
   const size_t iFeatureGroup,
   const FloatEbmType * const pModelFeatureGroup,
   const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures,
   const size_t iTargetClassOrZero
) const {
   const FloatEbmType * const aScores = GetPredictorScores(
      iFeatureGroup, 
      pModelFeatureGroup, 
      perDimensionIndexArrayForBinnedFeatures
   );
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

TestApi::TestApi(const ptrdiff_t learningTypeOrCountTargetClasses, const ptrdiff_t iZeroClassificationLogit) :
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

TestApi::~TestApi() {
   if(nullptr != m_pEbmBoosting) {
      FreeBoosting(m_pEbmBoosting);
   }
   if(nullptr != m_pEbmInteraction) {
      FreeInteraction(m_pEbmInteraction);
   }
}

void TestApi::AddFeatures(const std::vector<FeatureTest> features) {
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

void TestApi::AddFeatureGroups(const std::vector<std::vector<size_t>> featureGroups) {
   if(Stage::FeaturesAdded != m_stage) {
      exit(1);
   }

   for(const std::vector<size_t> oneFeatureGroup : featureGroups) {
      EbmNativeFeatureGroup featureGroup;
      featureGroup.countFeaturesInGroup = oneFeatureGroup.size();
      m_featureGroups.push_back(featureGroup);
      std::vector<size_t> countBins;
      for(const size_t oneIndex : oneFeatureGroup) {
         if(m_features.size() <= oneIndex) {
            exit(1);
         }
         m_featureGroupIndexes.push_back(oneIndex);
         countBins.push_back(static_cast<size_t>(m_features[oneIndex].countBins));
      }
      m_countBinsByFeatureGroup.push_back(countBins);
   }

   m_stage = Stage::FeatureGroupsAdded;
}

void TestApi::AddTrainingSamples(const std::vector<RegressionSample> samples) {
   if(Stage::FeatureGroupsAdded != m_stage) {
      exit(1);
   }
   if(k_learningTypeRegression != m_learningTypeOrCountTargetClasses) {
      exit(1);
   }
   const size_t cSamples = samples.size();
   if(0 != cSamples) {
      const size_t cFeatures = m_features.size();
      const bool bNullPredictionScores = samples[0].m_bNullPredictionScores;
      m_bNullTrainingPredictionScores = bNullPredictionScores;

      for(const RegressionSample oneSample : samples) {
         if(cFeatures != oneSample.m_binnedDataPerFeatureArray.size()) {
            exit(1);
         }
         if(bNullPredictionScores != oneSample.m_bNullPredictionScores) {
            exit(1);
         }
         const FloatEbmType target = oneSample.m_target;
         if(std::isnan(target)) {
            exit(1);
         }
         if(std::isinf(target)) {
            exit(1);
         }
         m_trainingRegressionTargets.push_back(target);
         if(!bNullPredictionScores) {
            const FloatEbmType score = oneSample.m_priorPredictorPrediction;
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
         for(size_t iSample = 0; iSample < cSamples; ++iSample) {
            const IntEbmType data = samples[iSample].m_binnedDataPerFeatureArray[iFeature];
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

void TestApi::AddTrainingSamples(const std::vector<ClassificationSample> samples) {
   if(Stage::FeatureGroupsAdded != m_stage) {
      exit(1);
   }
   if(!IsClassification(m_learningTypeOrCountTargetClasses)) {
      exit(1);
   }
   const size_t cSamples = samples.size();
   if(0 != cSamples) {
      const size_t cFeatures = m_features.size();
      const bool bNullPredictionScores = samples[0].m_bNullPredictionScores;
      m_bNullTrainingPredictionScores = bNullPredictionScores;

      for(const ClassificationSample oneSample : samples) {
         if(cFeatures != oneSample.m_binnedDataPerFeatureArray.size()) {
            exit(1);
         }
         if(bNullPredictionScores != oneSample.m_bNullPredictionScores) {
            exit(1);
         }
         const IntEbmType target = oneSample.m_target;
         if(target < 0) {
            exit(1);
         }
         if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) <= static_cast<size_t>(target)) {
            exit(1);
         }
         m_trainingClassificationTargets.push_back(target);
         if(!bNullPredictionScores) {
            if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) != 
               oneSample.m_priorPredictorPerClassLogits.size()) 
            {
               exit(1);
            }
            ptrdiff_t iLogit = 0;
            for(const FloatEbmType oneLogit : oneSample.m_priorPredictorPerClassLogits) {
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
                     m_trainingPredictionScores.push_back(
                        oneLogit - oneSample.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                  }
#else // EXPAND_BINARY_LOGITS
                  if(m_iZeroClassificationLogit < 0) {
                     if(0 != iLogit) {
                        m_trainingPredictionScores.push_back(oneLogit - oneSample.m_priorPredictorPerClassLogits[0]);
                     }
                  } else {
                     if(m_iZeroClassificationLogit != iLogit) {
                        m_trainingPredictionScores.push_back(
                           oneLogit - oneSample.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                     }
                  }
#endif // EXPAND_BINARY_LOGITS
               } else {
                  // multiclass
#ifdef REDUCE_MULTICLASS_LOGITS
                  if(m_iZeroClassificationLogit < 0) {
                     if(0 != iLogit) {
                        m_trainingPredictionScores.push_back(oneLogit - oneSample.m_logits[0]);
                     }
                  } else {
                     if(m_iZeroClassificationLogit != iLogit) {
                        m_trainingPredictionScores.push_back(
                           oneLogit - oneSample.m_logits[m_iZeroClassificationLogit]);
                     }
                  }
#else // REDUCE_MULTICLASS_LOGITS
                  if(m_iZeroClassificationLogit < 0) {
                     m_trainingPredictionScores.push_back(oneLogit);
                  } else {
                     m_trainingPredictionScores.push_back(
                        oneLogit - oneSample.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                  }
#endif // REDUCE_MULTICLASS_LOGITS
               }
               ++iLogit;
            }
         }
      }
      for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
         const EbmNativeFeature feature = m_features[iFeature];
         for(size_t iSample = 0; iSample < cSamples; ++iSample) {
            const IntEbmType data = samples[iSample].m_binnedDataPerFeatureArray[iFeature];
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

void TestApi::AddValidationSamples(const std::vector<RegressionSample> samples) {
   if(Stage::TrainingAdded != m_stage) {
      exit(1);
   }
   if(k_learningTypeRegression != m_learningTypeOrCountTargetClasses) {
      exit(1);
   }
   const size_t cSamples = samples.size();
   if(0 != cSamples) {
      const size_t cFeatures = m_features.size();
      const bool bNullPredictionScores = samples[0].m_bNullPredictionScores;
      m_bNullValidationPredictionScores = bNullPredictionScores;

      for(const RegressionSample oneSample : samples) {
         if(cFeatures != oneSample.m_binnedDataPerFeatureArray.size()) {
            exit(1);
         }
         if(bNullPredictionScores != oneSample.m_bNullPredictionScores) {
            exit(1);
         }
         const FloatEbmType target = oneSample.m_target;
         if(std::isnan(target)) {
            exit(1);
         }
         if(std::isinf(target)) {
            exit(1);
         }
         m_validationRegressionTargets.push_back(target);
         if(!bNullPredictionScores) {
            const FloatEbmType score = oneSample.m_priorPredictorPrediction;
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
         for(size_t iSample = 0; iSample < cSamples; ++iSample) {
            const IntEbmType data = samples[iSample].m_binnedDataPerFeatureArray[iFeature];
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

void TestApi::AddValidationSamples(const std::vector<ClassificationSample> samples) {
   if(Stage::TrainingAdded != m_stage) {
      exit(1);
   }
   if(!IsClassification(m_learningTypeOrCountTargetClasses)) {
      exit(1);
   }
   const size_t cSamples = samples.size();
   if(0 != cSamples) {
      const size_t cFeatures = m_features.size();
      const bool bNullPredictionScores = samples[0].m_bNullPredictionScores;
      m_bNullValidationPredictionScores = bNullPredictionScores;

      for(const ClassificationSample oneSample : samples) {
         if(cFeatures != oneSample.m_binnedDataPerFeatureArray.size()) {
            exit(1);
         }
         if(bNullPredictionScores != oneSample.m_bNullPredictionScores) {
            exit(1);
         }
         const IntEbmType target = oneSample.m_target;
         if(target < 0) {
            exit(1);
         }
         if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) <= static_cast<size_t>(target)) {
            exit(1);
         }
         m_validationClassificationTargets.push_back(target);
         if(!bNullPredictionScores) {
            if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) != 
               oneSample.m_priorPredictorPerClassLogits.size()) 
            {
               exit(1);
            }
            ptrdiff_t iLogit = 0;
            for(const FloatEbmType oneLogit : oneSample.m_priorPredictorPerClassLogits) {
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
                     m_validationPredictionScores.push_back(
                        oneLogit - oneSample.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                  }
#else // EXPAND_BINARY_LOGITS
                  if(m_iZeroClassificationLogit < 0) {
                     if(0 != iLogit) {
                        m_validationPredictionScores.push_back(oneLogit - oneSample.m_priorPredictorPerClassLogits[0]);
                     }
                  } else {
                     if(m_iZeroClassificationLogit != iLogit) {
                        m_validationPredictionScores.push_back(
                           oneLogit - oneSample.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                     }
                  }
#endif // EXPAND_BINARY_LOGITS
               } else {
                  // multiclass
#ifdef REDUCE_MULTICLASS_LOGITS
                  if(m_iZeroClassificationLogit < 0) {
                     if(0 != iLogit) {
                        m_validationPredictionScores.push_back(oneLogit - oneSample.m_logits[0]);
                     }
                  } else {
                     if(m_iZeroClassificationLogit != iLogit) {
                        m_validationPredictionScores.push_back(
                           oneLogit - oneSample.m_logits[m_iZeroClassificationLogit]);
                     }
                  }
#else // REDUCE_MULTICLASS_LOGITS
                  if(m_iZeroClassificationLogit < 0) {
                     m_validationPredictionScores.push_back(oneLogit);
                  } else {
                     m_validationPredictionScores.push_back(
                        oneLogit - oneSample.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                  }
#endif // REDUCE_MULTICLASS_LOGITS
               }
               ++iLogit;
            }
         }
      }
      for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
         const EbmNativeFeature feature = m_features[iFeature];
         for(size_t iSample = 0; iSample < cSamples; ++iSample) {
            const IntEbmType data = samples[iSample].m_binnedDataPerFeatureArray[iFeature];
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

void TestApi::InitializeBoosting(const IntEbmType countInnerBags) {
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
         m_featureGroups.size(),
         0 == m_featureGroups.size() ? nullptr : &m_featureGroups[0],
         0 == m_featureGroupIndexes.size() ? nullptr : &m_featureGroupIndexes[0],
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
         m_featureGroups.size(),
         0 == m_featureGroups.size() ? nullptr : &m_featureGroups[0],
         0 == m_featureGroupIndexes.size() ? nullptr : &m_featureGroupIndexes[0],
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

FloatEbmType TestApi::Boost(
   const IntEbmType indexFeatureGroup, 
   const std::vector<FloatEbmType> trainingWeights, 
   const std::vector<FloatEbmType> validationWeights, 
   const FloatEbmType learningRate, 
   const IntEbmType countTreeSplitsMax, 
   const IntEbmType countSamplesRequiredForChildSplitMin
) {
   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }
   if(indexFeatureGroup < IntEbmType { 0 }) {
      exit(1);
   }
   if(m_featureGroups.size() <= static_cast<size_t>(indexFeatureGroup)) {
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
   if(countSamplesRequiredForChildSplitMin < FloatEbmType { 0 }) {
      exit(1);
   }

   FloatEbmType validationMetricReturn = FloatEbmType { 0 };
   const IntEbmType ret = BoostingStep(
      m_pEbmBoosting,
      indexFeatureGroup,
      learningRate,
      countTreeSplitsMax,
      countSamplesRequiredForChildSplitMin,
      0 == trainingWeights.size() ? nullptr : &trainingWeights[0],
      0 == validationWeights.size() ? nullptr : &validationWeights[0],
      &validationMetricReturn
   );
   if(0 != ret) {
      exit(1);
   }
   return validationMetricReturn;
}

FloatEbmType TestApi::GetBestModelPredictorScore(
   const size_t iFeatureGroup, 
   const std::vector<size_t> indexes, 
   const size_t iScore
) const {
   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }
   if(m_featureGroups.size() <= iFeatureGroup) {
      exit(1);
   }
   FloatEbmType * pModelFeatureGroup = GetBestModelFeatureGroup(m_pEbmBoosting, iFeatureGroup);
   FloatEbmType predictorScore = GetPredictorScore(iFeatureGroup, pModelFeatureGroup, indexes, iScore);
   return predictorScore;
}

const FloatEbmType * TestApi::GetBestModelFeatureGroupRaw(const size_t iFeatureGroup) const {
   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }
   if(m_featureGroups.size() <= iFeatureGroup) {
      exit(1);
   }
   FloatEbmType * pModel = GetBestModelFeatureGroup(m_pEbmBoosting, iFeatureGroup);
   return pModel;
}

FloatEbmType TestApi::GetCurrentModelPredictorScore(
   const size_t iFeatureGroup,
   const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures,
   const size_t iTargetClassOrZero)
   const {
   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }
   if(m_featureGroups.size() <= iFeatureGroup) {
      exit(1);
   }
   FloatEbmType * pModelFeatureGroup = GetCurrentModelFeatureGroup(m_pEbmBoosting, iFeatureGroup);
   FloatEbmType predictorScore = GetPredictorScore(
      iFeatureGroup,
      pModelFeatureGroup,
      perDimensionIndexArrayForBinnedFeatures,
      iTargetClassOrZero
   );
   return predictorScore;
}

const FloatEbmType * TestApi::GetCurrentModelFeatureGroupRaw(const size_t iFeatureGroup) const {
   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }
   if(m_featureGroups.size() <= iFeatureGroup) {
      exit(1);
   }
   FloatEbmType * pModel = GetCurrentModelFeatureGroup(m_pEbmBoosting, iFeatureGroup);
   return pModel;
}

void TestApi::AddInteractionSamples(const std::vector<RegressionSample> samples) {
   if(Stage::FeaturesAdded != m_stage) {
      exit(1);
   }
   if(k_learningTypeRegression != m_learningTypeOrCountTargetClasses) {
      exit(1);
   }
   const size_t cSamples = samples.size();
   if(0 != cSamples) {
      const size_t cFeatures = m_features.size();
      const bool bNullPredictionScores = samples[0].m_bNullPredictionScores;
      m_bNullInteractionPredictionScores = bNullPredictionScores;

      for(const RegressionSample oneSample : samples) {
         if(cFeatures != oneSample.m_binnedDataPerFeatureArray.size()) {
            exit(1);
         }
         if(bNullPredictionScores != oneSample.m_bNullPredictionScores) {
            exit(1);
         }
         const FloatEbmType target = oneSample.m_target;
         if(std::isnan(target)) {
            exit(1);
         }
         if(std::isinf(target)) {
            exit(1);
         }
         m_interactionRegressionTargets.push_back(target);
         if(!bNullPredictionScores) {
            const FloatEbmType score = oneSample.m_priorPredictorPrediction;
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
         for(size_t iSample = 0; iSample < cSamples; ++iSample) {
            const IntEbmType data = samples[iSample].m_binnedDataPerFeatureArray[iFeature];
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

void TestApi::AddInteractionSamples(const std::vector<ClassificationSample> samples) {
   if(Stage::FeaturesAdded != m_stage) {
      exit(1);
   }
   if(!IsClassification(m_learningTypeOrCountTargetClasses)) {
      exit(1);
   }
   const size_t cSamples = samples.size();
   if(0 != cSamples) {
      const size_t cFeatures = m_features.size();
      const bool bNullPredictionScores = samples[0].m_bNullPredictionScores;
      m_bNullInteractionPredictionScores = bNullPredictionScores;

      for(const ClassificationSample oneSample : samples) {
         if(cFeatures != oneSample.m_binnedDataPerFeatureArray.size()) {
            exit(1);
         }
         if(bNullPredictionScores != oneSample.m_bNullPredictionScores) {
            exit(1);
         }
         const IntEbmType target = oneSample.m_target;
         if(target < 0) {
            exit(1);
         }
         if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) <= static_cast<size_t>(target)) {
            exit(1);
         }
         m_interactionClassificationTargets.push_back(target);
         if(!bNullPredictionScores) {
            if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) != 
               oneSample.m_priorPredictorPerClassLogits.size()) 
            {
               exit(1);
            }
            ptrdiff_t iLogit = 0;
            for(const FloatEbmType oneLogit : oneSample.m_priorPredictorPerClassLogits) {
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
                     m_interactionPredictionScores.push_back(
                        oneLogit - oneSample.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                  }
#else // EXPAND_BINARY_LOGITS
                  if(m_iZeroClassificationLogit < 0) {
                     if(0 != iLogit) {
                        m_interactionPredictionScores.push_back(
                           oneLogit - oneSample.m_priorPredictorPerClassLogits[0]);
                     }
                  } else {
                     if(m_iZeroClassificationLogit != iLogit) {
                        m_interactionPredictionScores.push_back(
                           oneLogit - oneSample.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                     }
                  }
#endif // EXPAND_BINARY_LOGITS
               } else {
                  // multiclass
#ifdef REDUCE_MULTICLASS_LOGITS
                  if(m_iZeroClassificationLogit < 0) {
                     if(0 != iLogit) {
                        m_interactionPredictionScores.push_back(oneLogit - oneSample.m_logits[0]);
                     }
                  } else {
                     if(m_iZeroClassificationLogit != iLogit) {
                        m_interactionPredictionScores.push_back(
                           oneLogit - oneSample.m_logits[m_iZeroClassificationLogit]);
                     }
                  }
#else // REDUCE_MULTICLASS_LOGITS
                  if(m_iZeroClassificationLogit < 0) {
                     m_interactionPredictionScores.push_back(oneLogit);
                  } else {
                     m_interactionPredictionScores.push_back(
                        oneLogit - oneSample.m_priorPredictorPerClassLogits[m_iZeroClassificationLogit]);
                  }
#endif // REDUCE_MULTICLASS_LOGITS
               }
               ++iLogit;
            }
         }
      }
      for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
         const EbmNativeFeature feature = m_features[iFeature];
         for(size_t iSample = 0; iSample < cSamples; ++iSample) {
            const IntEbmType data = samples[iSample].m_binnedDataPerFeatureArray[iFeature];
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

void TestApi::InitializeInteraction() {
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

FloatEbmType TestApi::InteractionScore(
   const std::vector<IntEbmType> featuresInGroup, 
   const IntEbmType countSamplesRequiredForChildSplitMin
) const {
   if(Stage::InitializedInteraction != m_stage) {
      exit(1);
   }
   for(const IntEbmType oneFeatureIndex : featuresInGroup) {
      if(oneFeatureIndex < IntEbmType { 0 }) {
         exit(1);
      }
      if(m_features.size() <= static_cast<size_t>(oneFeatureIndex)) {
         exit(1);
      }
   }

   FloatEbmType interactionScoreReturn = FloatEbmType { 0 };
   const IntEbmType ret = CalculateInteractionScore(
      m_pEbmInteraction,
      featuresInGroup.size(),
      0 == featuresInGroup.size() ? nullptr : &featuresInGroup[0],
      countSamplesRequiredForChildSplitMin,
      &interactionScoreReturn
   );
   if(0 != ret) {
      exit(1);
   }
   return interactionScoreReturn;
}

extern void DisplayCuts(
   IntEbmType countSamples,
   FloatEbmType * featureValues,
   IntEbmType countBinsMax,
   IntEbmType countSamplesPerBinMin,
   IntEbmType countCutPoints,
   FloatEbmType * cutPointsLowerBoundInclusive,
   IntEbmType isMissingPresent,
   FloatEbmType minValue,
   FloatEbmType maxValue
) {
   UNUSED(isMissingPresent);
   UNUSED(minValue);
   UNUSED(maxValue);

   size_t cBinsMax = static_cast<size_t>(countBinsMax);
   size_t cCutPoints = static_cast<size_t>(countCutPoints);

   std::vector<FloatEbmType> samples(featureValues, featureValues + countSamples);
   samples.erase(std::remove_if(samples.begin(), samples.end(),
      [](const FloatEbmType & value) { return std::isnan(value); }), samples.end());
   std::sort(samples.begin(), samples.end());

   std::cout << std::endl << std::endl;
   std::cout << "missing=" << (countSamples - samples.size()) << ", countBinsMax=" << countBinsMax << 
      ", countSamplesPerBinMin=" << countSamplesPerBinMin << ", avgBin=" << 
      static_cast<FloatEbmType>(samples.size()) / static_cast<FloatEbmType>(countBinsMax) << std::endl;

   size_t iCut = 0;
   size_t cInBin = 0;
   for(auto val: samples) {
      while(iCut < cCutPoints && cutPointsLowerBoundInclusive[iCut] <= val) {
         std::cout << "| " << cInBin << std::endl;
         cInBin = 0;
         ++iCut;
      }
      std::cout << val << ' ';
      ++cInBin;
   }

   std::cout << "| " << cInBin << std::endl;
   ++iCut;

   while(iCut < cBinsMax) {
      std::cout << "| 0" << std::endl;
      ++iCut;
   }

   std::cout << std::endl << std::endl;
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

   std::vector<TestCaseHidden> g_allTestsHidden = GetAllTestsHidden();
   std::stable_sort(g_allTestsHidden.begin(), g_allTestsHidden.end(),
      [](const TestCaseHidden & lhs, const TestCaseHidden & rhs) {
         return lhs.m_testPriority < rhs.m_testPriority;
      });

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
