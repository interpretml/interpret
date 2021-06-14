// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

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
// TODO : add test for the condition where we overflow the validation regression or classification scores without overflowing the model update or the 
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
#include "ebm_native_test.hpp"

#ifdef _MSC_VER
// we want to be able to put breakpoints in the FAILED function below, so in release mode turn off optimizations
#pragma optimize("", off)
#endif // _MSC_VER
extern void FAILED(const double val, TestCaseHidden * const pTestCaseHidden, const std::string message) {
   UNUSED(val);
   pTestCaseHidden->m_bPassed = false;
   std::cout << message;
}
#ifdef _MSC_VER
// this turns back on whatever optimization settings we have on the command line.  It's like a pop operation
#pragma optimize("", on)
#endif // _MSC_VER

void EBM_NATIVE_CALLING_CONVENTION LogMessage(TraceEbmType traceLevel, const char * message) {
   const size_t cChars = strlen(message); // test that the string memory is accessible
   UNUSED(cChars);
   if(traceLevel <= TraceLevelOff) {
      // don't display log messages during tests, but having this code here makes it easy to turn on when needed
      printf("\n%s: %s\n", GetTraceLevelString(traceLevel), message);
   }
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
   m_bNullTrainingWeights(true),
   m_bNullValidationWeights(true),
   m_bNullTrainingPredictionScores(true),
   m_bNullValidationPredictionScores(true),
   m_boosterHandle(nullptr),
   m_bNullInteractionWeights(true),
   m_bNullInteractionPredictionScores(true),
   m_interactionHandle(nullptr) 
{
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
   if(nullptr != m_boosterHandle) {
      FreeBooster(m_boosterHandle);
   }
   if(nullptr != m_interactionHandle) {
      FreeInteractionDetector(m_interactionHandle);
   }
}

void TestApi::AddFeatures(const std::vector<FeatureTest> features) {
   if(Stage::Beginning != m_stage) {
      exit(1);
   }

   for(const FeatureTest & oneFeature : features) {
      m_featuresCategorical.push_back(oneFeature.m_bCategorical ? EBM_TRUE : EBM_FALSE);
      m_featuresBinCount.push_back(oneFeature.m_countBins);
   }

   m_stage = Stage::FeaturesAdded;
}

void TestApi::AddFeatureGroups(const std::vector<std::vector<size_t>> featureGroups) {
   if(Stage::FeaturesAdded != m_stage) {
      exit(1);
   }

   for(const std::vector<size_t> oneFeatureGroup : featureGroups) {
      m_featureGroupsDimensionCount.push_back(oneFeatureGroup.size());
      std::vector<size_t> countBins;
      for(const size_t oneIndex : oneFeatureGroup) {
         if(m_featuresBinCount.size() <= oneIndex) {
            exit(1);
         }
         m_featureGroupsFeatureIndexes.push_back(oneIndex);
         countBins.push_back(static_cast<size_t>(m_featuresBinCount[oneIndex]));
      }
      m_countBinsByFeatureGroup.push_back(countBins);
   }

   m_stage = Stage::FeatureGroupsAdded;
}

void TestApi::AddTrainingSamples(const std::vector<TestSample> samples) {
   if(Stage::FeatureGroupsAdded != m_stage) {
      exit(1);
   }
   const size_t cSamples = samples.size();
   if(0 != cSamples) {
      const size_t cFeatures = m_featuresBinCount.size();

      const bool bNullPredictionScores = 0 == samples[0].m_priorScore.size();
      m_bNullTrainingPredictionScores = bNullPredictionScores;

      const bool bNullWeights = samples[0].m_bNullWeight;
      m_bNullTrainingWeights = bNullWeights;

      for(const TestSample oneSample : samples) {
         if(cFeatures != oneSample.m_binnedDataPerFeatureArray.size()) {
            exit(1);
         }
         if(bNullPredictionScores != (0 == oneSample.m_priorScore.size())) {
            exit(1);
         }
         if(bNullWeights != oneSample.m_bNullWeight) {
            exit(1);
         }
         const FloatEbmType target = oneSample.m_target;
         if(std::isnan(target)) {
            exit(1);
         }
         if(std::isinf(target)) {
            exit(1);
         }
         if(IsClassification(m_learningTypeOrCountTargetClasses)) {
            const IntEbmType targetInt = static_cast<IntEbmType>(target);
            if(targetInt < IntEbmType { 0 }) {
               exit(1);
            }
            if(static_cast<IntEbmType>(m_learningTypeOrCountTargetClasses) <= targetInt) {
               exit(1);
            }
            m_trainingClassificationTargets.push_back(targetInt);
            if(!bNullPredictionScores) {
               if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) != oneSample.m_priorScore.size()) {
                  exit(1);
               }
               ptrdiff_t iLogit = 0;
               for(const FloatEbmType oneLogit : oneSample.m_priorScore) {
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
                           oneLogit - oneSample.m_priorScore[m_iZeroClassificationLogit]);
                     }
#else // EXPAND_BINARY_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_trainingPredictionScores.push_back(oneLogit - oneSample.m_priorScore[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_trainingPredictionScores.push_back(
                              oneLogit - oneSample.m_priorScore[m_iZeroClassificationLogit]);
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
                           oneLogit - oneSample.m_priorScore[m_iZeroClassificationLogit]);
                     }
#endif // REDUCE_MULTICLASS_LOGITS
                  }
                  ++iLogit;
               }
            }
         } else {
            m_trainingRegressionTargets.push_back(target);
            if(!bNullPredictionScores) {
               const FloatEbmType score = oneSample.m_priorScore[0];
               if(std::isnan(score)) {
                  exit(1);
               }
               if(std::isinf(score)) {
                  exit(1);
               }
               m_trainingPredictionScores.push_back(score);
            }
         }
         if(!bNullWeights) {
            const FloatEbmType weight = oneSample.m_weight;
            if(std::isnan(weight)) {
               exit(1);
            }
            if(std::isinf(weight)) {
               exit(1);
            }
            m_trainingWeights.push_back(weight);
         }
      }
      for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
         const IntEbmType countBins = m_featuresBinCount[iFeature];
         for(size_t iSample = 0; iSample < cSamples; ++iSample) {
            const IntEbmType data = samples[iSample].m_binnedDataPerFeatureArray[iFeature];
            if(data < 0) {
               exit(1);
            }
            if(countBins <= data) {
               exit(1);
            }
            m_trainingBinnedData.push_back(data);
         }
      }
   }
   m_stage = Stage::TrainingAdded;
}

void TestApi::AddValidationSamples(const std::vector<TestSample> samples) {
   if(Stage::TrainingAdded != m_stage) {
      exit(1);
   }
   const size_t cSamples = samples.size();
   if(0 != cSamples) {
      const size_t cFeatures = m_featuresBinCount.size();
      const bool bNullPredictionScores = 0 == samples[0].m_priorScore.size();
      m_bNullValidationPredictionScores = bNullPredictionScores;

      const bool bNullWeights = samples[0].m_bNullWeight;
      m_bNullValidationWeights = bNullWeights;

      for(const TestSample oneSample : samples) {
         if(cFeatures != oneSample.m_binnedDataPerFeatureArray.size()) {
            exit(1);
         }
         if(bNullPredictionScores != (0 == oneSample.m_priorScore.size())) {
            exit(1);
         }
         if(bNullWeights != oneSample.m_bNullWeight) {
            exit(1);
         }
         const FloatEbmType target = oneSample.m_target;
         if(std::isnan(target)) {
            exit(1);
         }
         if(std::isinf(target)) {
            exit(1);
         }
         if(IsClassification(m_learningTypeOrCountTargetClasses)) {
            const IntEbmType targetInt = static_cast<IntEbmType>(target);
            if(targetInt < IntEbmType { 0 }) {
               exit(1);
            }
            if(static_cast<IntEbmType>(m_learningTypeOrCountTargetClasses) <= targetInt) {
               exit(1);
            }
            m_validationClassificationTargets.push_back(targetInt);
            if(!bNullPredictionScores) {
               if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) != oneSample.m_priorScore.size()) {
                  exit(1);
               }
               ptrdiff_t iLogit = 0;
               for(const FloatEbmType oneLogit : oneSample.m_priorScore) {
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
                           oneLogit - oneSample.m_priorScore[m_iZeroClassificationLogit]);
                     }
#else // EXPAND_BINARY_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_validationPredictionScores.push_back(oneLogit - oneSample.m_priorScore[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_validationPredictionScores.push_back(
                              oneLogit - oneSample.m_priorScore[m_iZeroClassificationLogit]);
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
                           oneLogit - oneSample.m_priorScore[m_iZeroClassificationLogit]);
                     }
#endif // REDUCE_MULTICLASS_LOGITS
                  }
                  ++iLogit;
               }
            }
         } else {
            m_validationRegressionTargets.push_back(target);
            if(!bNullPredictionScores) {
               const FloatEbmType score = oneSample.m_priorScore[0];
               if(std::isnan(score)) {
                  exit(1);
               }
               if(std::isinf(score)) {
                  exit(1);
               }
               m_validationPredictionScores.push_back(score);
            }
         }
         if(!bNullWeights) {
            const FloatEbmType weight = oneSample.m_weight;
            if(std::isnan(weight)) {
               exit(1);
            }
            if(std::isinf(weight)) {
               exit(1);
            }
            m_validationWeights.push_back(weight);
         }
      }
      for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
         const IntEbmType countBins = m_featuresBinCount[iFeature];
         for(size_t iSample = 0; iSample < cSamples; ++iSample) {
            const IntEbmType data = samples[iSample].m_binnedDataPerFeatureArray[iFeature];
            if(data < 0) {
               exit(1);
            }
            if(countBins <= data) {
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
   ErrorEbmType error;
   if(IsClassification(m_learningTypeOrCountTargetClasses)) {
      if(m_bNullTrainingPredictionScores) {
         m_trainingPredictionScores.resize(cVectorLength * m_trainingClassificationTargets.size());
      }
      if(m_bNullValidationPredictionScores) {
         m_validationPredictionScores.resize(cVectorLength * m_validationClassificationTargets.size());
      }
      if(m_bNullTrainingWeights) {
         m_trainingWeights.resize(m_trainingClassificationTargets.size());
      }
      if(m_bNullValidationWeights) {
         m_validationWeights.resize(m_validationClassificationTargets.size());
      }
      error = CreateClassificationBooster(
         k_randomSeed,
         m_learningTypeOrCountTargetClasses,
         m_featuresBinCount.size(),
         0 == m_featuresCategorical.size() ? nullptr : &m_featuresCategorical[0],
         0 == m_featuresBinCount.size() ? nullptr : &m_featuresBinCount[0],
         m_featureGroupsDimensionCount.size(),
         0 == m_featureGroupsDimensionCount.size() ? nullptr : &m_featureGroupsDimensionCount[0],
         0 == m_featureGroupsFeatureIndexes.size() ? nullptr : &m_featureGroupsFeatureIndexes[0],
         m_trainingClassificationTargets.size(),
         0 == m_trainingBinnedData.size() ? nullptr : &m_trainingBinnedData[0],
         0 == m_trainingClassificationTargets.size() ? nullptr : &m_trainingClassificationTargets[0],
         0 == m_trainingWeights.size() ? nullptr : &m_trainingWeights[0],
         0 == m_trainingPredictionScores.size() ? nullptr : &m_trainingPredictionScores[0],
         m_validationClassificationTargets.size(),
         0 == m_validationBinnedData.size() ? nullptr : &m_validationBinnedData[0],
         0 == m_validationClassificationTargets.size() ? nullptr : &m_validationClassificationTargets[0],
         0 == m_validationWeights.size() ? nullptr : &m_validationWeights[0],
         0 == m_validationPredictionScores.size() ? nullptr : &m_validationPredictionScores[0],
         countInnerBags,
         nullptr,
         &m_boosterHandle
      );
   } else if(k_learningTypeRegression == m_learningTypeOrCountTargetClasses) {
      if(m_bNullTrainingPredictionScores) {
         m_trainingPredictionScores.resize(cVectorLength * m_trainingRegressionTargets.size());
      }
      if(m_bNullValidationPredictionScores) {
         m_validationPredictionScores.resize(cVectorLength * m_validationRegressionTargets.size());
      }
      if(m_bNullTrainingWeights) {
         m_trainingWeights.resize(m_trainingRegressionTargets.size());
      }
      if(m_bNullValidationWeights) {
         m_validationWeights.resize(m_validationRegressionTargets.size());
      }
      error = CreateRegressionBooster(
         k_randomSeed,
         m_featuresBinCount.size(),
         0 == m_featuresCategorical.size() ? nullptr : &m_featuresCategorical[0],
         0 == m_featuresBinCount.size() ? nullptr : &m_featuresBinCount[0],
         m_featureGroupsDimensionCount.size(),
         0 == m_featureGroupsDimensionCount.size() ? nullptr : &m_featureGroupsDimensionCount[0],
         0 == m_featureGroupsFeatureIndexes.size() ? nullptr : &m_featureGroupsFeatureIndexes[0],
         m_trainingRegressionTargets.size(),
         0 == m_trainingBinnedData.size() ? nullptr : &m_trainingBinnedData[0],
         0 == m_trainingRegressionTargets.size() ? nullptr : &m_trainingRegressionTargets[0],
         0 == m_trainingWeights.size() ? nullptr : &m_trainingWeights[0],
         0 == m_trainingPredictionScores.size() ? nullptr : &m_trainingPredictionScores[0],
         m_validationRegressionTargets.size(),
         0 == m_validationBinnedData.size() ? nullptr : &m_validationBinnedData[0],
         0 == m_validationRegressionTargets.size() ? nullptr : &m_validationRegressionTargets[0],
         0 == m_validationWeights.size() ? nullptr : &m_validationWeights[0],
         0 == m_validationPredictionScores.size() ? nullptr : &m_validationPredictionScores[0],
         countInnerBags,
         nullptr,
         &m_boosterHandle
      );
   } else {
      exit(1);
   }

   if(Error_None != error) {
      printf("\nClean exit with nullptr from InitializeBoosting*.\n");
      exit(1);
   }
   if(nullptr == m_boosterHandle) {
      printf("\nClean exit with nullptr from InitializeBoosting*.\n");
      exit(1);
   }
   m_stage = Stage::InitializedBoosting;
}

FloatEbmType TestApi::Boost(
   const IntEbmType indexFeatureGroup,
   const GenerateUpdateOptionsType options,
   const FloatEbmType learningRate,
   const IntEbmType countSamplesRequiredForChildSplitMin,
   const std::vector<IntEbmType> leavesMax
) {
   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }
   if(indexFeatureGroup < IntEbmType { 0 }) {
      exit(1);
   }
   if(m_featureGroupsDimensionCount.size() <= static_cast<size_t>(indexFeatureGroup)) {
      exit(1);
   }
   if(std::isnan(learningRate)) {
      exit(1);
   }
   if(std::isinf(learningRate)) {
      exit(1);
   }
   if(countSamplesRequiredForChildSplitMin < FloatEbmType { 0 }) {
      exit(1);
   }

   FloatEbmType validationMetricOut = FloatEbmType { 0 };

   const ErrorEbmType retGenerate = GenerateModelUpdate(
      m_boosterHandle,
      indexFeatureGroup,
      options,
      learningRate,
      countSamplesRequiredForChildSplitMin,
      0 == leavesMax.size() ? nullptr : &leavesMax[0],
      nullptr
   );
   if(Error_None != retGenerate) {
      exit(1);
   }
   if(0 != (GenerateUpdateOptions_GradientSums & options)) {
      // if sums are on, then we MUST change the model update

      size_t cValues = GetVectorLength(m_learningTypeOrCountTargetClasses);
      std::vector<size_t> & countBinsByFeatureGroup = m_countBinsByFeatureGroup[static_cast<size_t>(indexFeatureGroup)];

      for(size_t iDimension = 0; iDimension < countBinsByFeatureGroup.size(); ++iDimension) {
         size_t cBins = countBinsByFeatureGroup[iDimension];
         cValues *= cBins;
      }

      FloatEbmType * aMem = new FloatEbmType[cValues];
      memset(aMem, 0, sizeof(*aMem) * cValues);

      const ErrorEbmType retSet = SetModelUpdateExpanded(
         m_boosterHandle,
         indexFeatureGroup,
         aMem
      );

      delete[] aMem;

      if(Error_None != retSet) {
         exit(1);
      }
   }
   const ErrorEbmType ret = ApplyModelUpdate(
      m_boosterHandle,
      &validationMetricOut
   );

   if(Error_None != ret) {
      exit(1);
   }
   return validationMetricOut;
}

FloatEbmType TestApi::GetBestModelPredictorScore(
   const size_t iFeatureGroup, 
   const std::vector<size_t> indexes, 
   const size_t iScore
) const {
   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }

   if(m_countBinsByFeatureGroup.size() <= iFeatureGroup) {
      exit(1);
   }
   const std::vector<size_t> countBins = m_countBinsByFeatureGroup[iFeatureGroup];

   const size_t cDimensions = countBins.size();
   size_t multiple = GetVectorLength(m_learningTypeOrCountTargetClasses);
   for(size_t iDimension = 0; iDimension < cDimensions; ++iDimension) {
      multiple *= countBins[iDimension];
   }

   std::vector<FloatEbmType> model;
   model.resize(multiple);

   const ErrorEbmType ret = GetBestModelFeatureGroup(m_boosterHandle, iFeatureGroup, &model[0]);
   if(Error_None != ret) {
      exit(1);
   }

   const FloatEbmType predictorScore = GetPredictorScore(iFeatureGroup, &model[0], indexes, iScore);
   return predictorScore;
}

const void TestApi::GetBestModelFeatureGroupRaw(const size_t iFeatureGroup, FloatEbmType * const aModelValues) const {
   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }
   if(m_featureGroupsDimensionCount.size() <= iFeatureGroup) {
      exit(1);
   }
   const ErrorEbmType ret = GetBestModelFeatureGroup(m_boosterHandle, iFeatureGroup, aModelValues);
   if(Error_None != ret) {
      exit(1);
   }
}

FloatEbmType TestApi::GetCurrentModelPredictorScore(
   const size_t iFeatureGroup,
   const std::vector<size_t> indexes,
   const size_t iScore
) const {
   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }

   if(m_countBinsByFeatureGroup.size() <= iFeatureGroup) {
      exit(1);
   }
   const std::vector<size_t> countBins = m_countBinsByFeatureGroup[iFeatureGroup];

   const size_t cDimensions = countBins.size();
   size_t multiple = GetVectorLength(m_learningTypeOrCountTargetClasses);
   for(size_t iDimension = 0; iDimension < cDimensions; ++iDimension) {
      multiple *= countBins[iDimension];
   }

   std::vector<FloatEbmType> model;
   model.resize(multiple);

   const ErrorEbmType ret = GetCurrentModelFeatureGroup(m_boosterHandle, iFeatureGroup, &model[0]);
   if(Error_None != ret) {
      exit(1);
   }

   const FloatEbmType predictorScore = GetPredictorScore(iFeatureGroup, &model[0], indexes, iScore);
   return predictorScore;
}

const void TestApi::GetCurrentModelFeatureGroupRaw(const size_t iFeatureGroup, FloatEbmType * const aModelValues) const {
   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }
   if(m_featureGroupsDimensionCount.size() <= iFeatureGroup) {
      exit(1);
   }
   const ErrorEbmType ret = GetCurrentModelFeatureGroup(m_boosterHandle, iFeatureGroup, aModelValues);
   if(Error_None != ret) {
      exit(1);
   }
}

void TestApi::AddInteractionSamples(const std::vector<TestSample> samples) {
   if(Stage::FeaturesAdded != m_stage) {
      exit(1);
   }
   const size_t cSamples = samples.size();
   if(0 != cSamples) {
      const size_t cFeatures = m_featuresBinCount.size();
      const bool bNullPredictionScores = 0 == samples[0].m_priorScore.size();
      m_bNullInteractionPredictionScores = bNullPredictionScores;

      const bool bNullWeights = samples[0].m_bNullWeight;
      m_bNullInteractionWeights = bNullWeights;

      for(const TestSample oneSample : samples) {
         if(cFeatures != oneSample.m_binnedDataPerFeatureArray.size()) {
            exit(1);
         }
         if(bNullPredictionScores != (0 == oneSample.m_priorScore.size())) {
            exit(1);
         }
         if(bNullWeights != oneSample.m_bNullWeight) {
            exit(1);
         }
         const FloatEbmType target = oneSample.m_target;
         if(std::isnan(target)) {
            exit(1);
         }
         if(std::isinf(target)) {
            exit(1);
         }
         if(IsClassification(m_learningTypeOrCountTargetClasses)) {
            const IntEbmType targetInt = static_cast<IntEbmType>(target);
            if(targetInt < IntEbmType { 0 }) {
               exit(1);
            }
            if(static_cast<IntEbmType>(m_learningTypeOrCountTargetClasses) <= targetInt) {
               exit(1);
            }
            m_interactionClassificationTargets.push_back(targetInt);
            if(!bNullPredictionScores) {
               if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) !=
                  oneSample.m_priorScore.size()) {
                  exit(1);
               }
               ptrdiff_t iLogit = 0;
               for(const FloatEbmType oneLogit : oneSample.m_priorScore) {
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
                           oneLogit - oneSample.m_priorScore[m_iZeroClassificationLogit]);
                     }
#else // EXPAND_BINARY_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_interactionPredictionScores.push_back(
                              oneLogit - oneSample.m_priorScore[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_interactionPredictionScores.push_back(
                              oneLogit - oneSample.m_priorScore[m_iZeroClassificationLogit]);
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
                           oneLogit - oneSample.m_priorScore[m_iZeroClassificationLogit]);
                     }
#endif // REDUCE_MULTICLASS_LOGITS
                  }
                  ++iLogit;
               }
            }
         } else {
            m_interactionRegressionTargets.push_back(target);
            if(!bNullPredictionScores) {
               const FloatEbmType score = oneSample.m_priorScore[0];
               if(std::isnan(score)) {
                  exit(1);
               }
               if(std::isinf(score)) {
                  exit(1);
               }
               m_interactionPredictionScores.push_back(score);
            }
         }
         if(!bNullWeights) {
            const FloatEbmType weight = oneSample.m_weight;
            if(std::isnan(weight)) {
               exit(1);
            }
            if(std::isinf(weight)) {
               exit(1);
            }
            m_interactionWeights.push_back(weight);
         }
      }
      for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
         const IntEbmType countBins = m_featuresBinCount[iFeature];
         for(size_t iSample = 0; iSample < cSamples; ++iSample) {
            const IntEbmType data = samples[iSample].m_binnedDataPerFeatureArray[iFeature];
            if(data < 0) {
               exit(1);
            }
            if(countBins <= data) {
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
   ErrorEbmType error;
   if(IsClassification(m_learningTypeOrCountTargetClasses)) {
      if(m_bNullInteractionPredictionScores) {
         m_interactionPredictionScores.resize(cVectorLength * m_interactionClassificationTargets.size());
      }
      if(m_bNullInteractionWeights) {
         m_interactionWeights.resize(m_interactionClassificationTargets.size());
      }
      error = CreateClassificationInteractionDetector(
         m_learningTypeOrCountTargetClasses,
         m_featuresBinCount.size(),
         0 == m_featuresCategorical.size() ? nullptr : &m_featuresCategorical[0],
         0 == m_featuresBinCount.size() ? nullptr : &m_featuresBinCount[0],
         m_interactionClassificationTargets.size(),
         0 == m_interactionBinnedData.size() ? nullptr : &m_interactionBinnedData[0],
         0 == m_interactionClassificationTargets.size() ? nullptr : &m_interactionClassificationTargets[0],
         0 == m_interactionWeights.size() ? nullptr : &m_interactionWeights[0],
         0 == m_interactionPredictionScores.size() ? nullptr : &m_interactionPredictionScores[0],
         nullptr,
         &m_interactionHandle
      );
   } else if(k_learningTypeRegression == m_learningTypeOrCountTargetClasses) {
      if(m_bNullInteractionPredictionScores) {
         m_interactionPredictionScores.resize(cVectorLength * m_interactionRegressionTargets.size());
      }
      if(m_bNullInteractionWeights) {
         m_interactionWeights.resize(m_interactionRegressionTargets.size());
      }
      error = CreateRegressionInteractionDetector(
         m_featuresBinCount.size(),
         0 == m_featuresCategorical.size() ? nullptr : &m_featuresCategorical[0],
         0 == m_featuresBinCount.size() ? nullptr : &m_featuresBinCount[0],
         m_interactionRegressionTargets.size(),
         0 == m_interactionBinnedData.size() ? nullptr : &m_interactionBinnedData[0],
         0 == m_interactionRegressionTargets.size() ? nullptr : &m_interactionRegressionTargets[0],
         0 == m_interactionWeights.size() ? nullptr : &m_interactionWeights[0],
         0 == m_interactionPredictionScores.size() ? nullptr : &m_interactionPredictionScores[0],
         nullptr,
         &m_interactionHandle
      );
   } else {
      exit(1);
   }

   if(Error_None != error) {
      printf("\nClean exit with nullptr from InitializeInteraction*.\n");
      exit(1);
   }
   if(nullptr == m_interactionHandle) {
      printf("\nClean exit with nullptr from InitializeInteraction*.\n");
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
      if(m_featuresBinCount.size() <= static_cast<size_t>(oneFeatureIndex)) {
         exit(1);
      }
   }

   FloatEbmType interactionScoreOut = FloatEbmType { 0 };
   const ErrorEbmType ret = CalculateInteractionScore(
      m_interactionHandle,
      featuresInGroup.size(),
      0 == featuresInGroup.size() ? nullptr : &featuresInGroup[0],
      countSamplesRequiredForChildSplitMin,
      &interactionScoreOut
   );
   if(Error_None != ret) {
      exit(1);
   }
   return interactionScoreOut;
}

extern void DisplayCuts(
   IntEbmType countSamples,
   FloatEbmType * featureValues,
   IntEbmType countBinsMax,
   IntEbmType countSamplesPerBinMin,
   IntEbmType countCuts,
   FloatEbmType * cutsLowerBoundInclusive,
   IntEbmType isMissingPresent,
   FloatEbmType minValue,
   FloatEbmType maxValue
) {
   UNUSED(isMissingPresent);
   UNUSED(minValue);
   UNUSED(maxValue);

   size_t cBinsMax = static_cast<size_t>(countBinsMax);
   size_t cCuts = static_cast<size_t>(countCuts);

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
      while(iCut < cCuts && cutsLowerBoundInclusive[iCut] <= val) {
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

extern "C" void TestCHeaderConstructs();

int main() {
#ifdef _MSC_VER
   // only test on the Visual Studio Compiler since it's easier.  If we support C later then add more compilers
   TestCHeaderConstructs();
#endif // _MSC_VER

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
