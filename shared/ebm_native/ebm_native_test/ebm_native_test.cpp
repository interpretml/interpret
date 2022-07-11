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

// TODO : add test for the condition where we overflow the term update to NaN or +-infinity for regression by using exteme regression values and in 
//   classification by using certainty situations with big learning rates
// TODO : add test for the condition where we overflow the result of adding the term update to the existing term NaN or +-infinity for regression 
//   by using exteme regression values and in classification by using certainty situations with big learning rates
// TODO : add test for the condition where we overflow the validation regression or classification scores without overflowing the term update or the 
//   term tensors.  We can do this by having two extreme features that will overflow together

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

void EBM_CALLING_CONVENTION LogMessage(TraceEbmType traceLevel, const char * message) {
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

const double * TestApi::GetTermScores(
   const size_t iTerm,
   const double * const aTermScores,
   const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures
) const {
   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }
   const size_t cVectorLength = GetVectorLength(m_learningTypeOrCountTargetClasses);

   if(m_termBinCounts.size() <= iTerm) {
      exit(1);
   }
   const std::vector<size_t> countBins = m_termBinCounts[iTerm];

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
   return &aTermScores[iValue];
}

double TestApi::GetTermScore(
   const size_t iTerm,
   const double * const aTermScores,
   const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures,
   const size_t iTargetClassOrZero
) const {
   const double * const aScores = GetTermScores(
      iTerm, 
      aTermScores,
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
            return double { 0 };
         } else {
            return aScores[iTargetClassOrZero] - aScores[m_iZeroClassificationLogit];
         }
      }
#else // EXPAND_BINARY_LOGITS
      if(m_iZeroClassificationLogit < 0) {
         if(0 == iTargetClassOrZero) {
            return double { 0 };
         } else {
            return aScores[0];
         }
      } else {
         if(static_cast<size_t>(m_iZeroClassificationLogit) == iTargetClassOrZero) {
            return double { 0 };
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
            return double { 0 };
         } else {
            return aScores[iTargetClassOrZero - 1];
         }
      } else {
         if(staitc_cast<size_t>(m_iZeroClassificationLogit) == iTargetClassOrZero) {
            return double { 0 };
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
   m_bNullTrainingInitScores(true),
   m_bNullValidationWeights(true),
   m_bNullValidationInitScores(true),
   m_boosterHandle(nullptr),
   m_bNullInteractionWeights(true),
   m_bNullInteractionInitScores(true),
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
      m_featureNominals.push_back(oneFeature.m_bNominal ? EBM_TRUE : EBM_FALSE);
      m_featureBinCounts.push_back(oneFeature.m_countBins);
   }

   m_stage = Stage::FeaturesAdded;
}

void TestApi::AddTerms(const std::vector<std::vector<size_t>> termFeatures) {
   if(Stage::FeaturesAdded != m_stage) {
      exit(1);
   }

   for(const std::vector<size_t> & termFeatureIndexes : termFeatures) {
      m_dimensionCounts.push_back(termFeatureIndexes.size());
      std::vector<size_t> countBins;
      for(const size_t indexFeature : termFeatureIndexes) {
         if(m_featureBinCounts.size() <= indexFeature) {
            exit(1);
         }
         m_featureIndexes.push_back(indexFeature);
         countBins.push_back(static_cast<size_t>(m_featureBinCounts[indexFeature]));
      }
      m_termBinCounts.push_back(countBins);
   }

   m_stage = Stage::TermsAdded;
}

void TestApi::AddTrainingSamples(const std::vector<TestSample> samples) {
   if(Stage::TermsAdded != m_stage) {
      exit(1);
   }
   const size_t cSamples = samples.size();
   if(0 != cSamples) {
      const size_t cFeatures = m_featureBinCounts.size();

      const bool bNullInitScores = 0 == samples[0].m_initScores.size();
      m_bNullTrainingInitScores = bNullInitScores;

      const bool bNullWeights = samples[0].m_bNullWeight;
      m_bNullTrainingWeights = bNullWeights;

      for(const TestSample & oneSample : samples) {
         if(cFeatures != oneSample.m_binnedDataPerFeatureArray.size()) {
            exit(1);
         }
         if(bNullInitScores != (0 == oneSample.m_initScores.size())) {
            exit(1);
         }
         if(bNullWeights != oneSample.m_bNullWeight) {
            exit(1);
         }
         const double target = oneSample.m_target;
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
            if(!bNullInitScores) {
               if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) != oneSample.m_initScores.size()) {
                  exit(1);
               }
               ptrdiff_t iLogit = 0;
               for(const double oneLogit : oneSample.m_initScores) {
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
                        m_trainingInitScores.push_back(oneLogit);
                     } else {
                        m_trainingInitScores.push_back(
                           oneLogit - oneSample.m_initScores[m_iZeroClassificationLogit]);
                     }
#else // EXPAND_BINARY_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_trainingInitScores.push_back(oneLogit - oneSample.m_initScores[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_trainingInitScores.push_back(
                              oneLogit - oneSample.m_initScores[m_iZeroClassificationLogit]);
                        }
                     }
#endif // EXPAND_BINARY_LOGITS
                  } else {
                     // multiclass
#ifdef REDUCE_MULTICLASS_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_trainingInitScores.push_back(oneLogit - oneSample.m_logits[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_trainingInitScores.push_back(
                              oneLogit - oneSample.m_logits[m_iZeroClassificationLogit]);
                        }
                     }
#else // REDUCE_MULTICLASS_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        m_trainingInitScores.push_back(oneLogit);
                     } else {
                        m_trainingInitScores.push_back(
                           oneLogit - oneSample.m_initScores[m_iZeroClassificationLogit]);
                     }
#endif // REDUCE_MULTICLASS_LOGITS
                  }
                  ++iLogit;
               }
            }
         } else {
            m_trainingRegressionTargets.push_back(target);
            if(!bNullInitScores) {
               const double score = oneSample.m_initScores[0];
               if(std::isnan(score)) {
                  exit(1);
               }
               if(std::isinf(score)) {
                  exit(1);
               }
               m_trainingInitScores.push_back(score);
            }
         }
         if(!bNullWeights) {
            const double weight = oneSample.m_weight;
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
         const IntEbmType countBins = m_featureBinCounts[iFeature];
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
      const size_t cFeatures = m_featureBinCounts.size();
      const bool bNullInitScores = 0 == samples[0].m_initScores.size();
      m_bNullValidationInitScores = bNullInitScores;

      const bool bNullWeights = samples[0].m_bNullWeight;
      m_bNullValidationWeights = bNullWeights;

      for(const TestSample & oneSample : samples) {
         if(cFeatures != oneSample.m_binnedDataPerFeatureArray.size()) {
            exit(1);
         }
         if(bNullInitScores != (0 == oneSample.m_initScores.size())) {
            exit(1);
         }
         if(bNullWeights != oneSample.m_bNullWeight) {
            exit(1);
         }
         const double target = oneSample.m_target;
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
            if(!bNullInitScores) {
               if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) != oneSample.m_initScores.size()) {
                  exit(1);
               }
               ptrdiff_t iLogit = 0;
               for(const double oneLogit : oneSample.m_initScores) {
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
                        m_validationInitScores.push_back(oneLogit);
                     } else {
                        m_validationInitScores.push_back(
                           oneLogit - oneSample.m_initScores[m_iZeroClassificationLogit]);
                     }
#else // EXPAND_BINARY_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_validationInitScores.push_back(oneLogit - oneSample.m_initScores[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_validationInitScores.push_back(
                              oneLogit - oneSample.m_initScores[m_iZeroClassificationLogit]);
                        }
                     }
#endif // EXPAND_BINARY_LOGITS
                  } else {
                     // multiclass
#ifdef REDUCE_MULTICLASS_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_validationInitScores.push_back(oneLogit - oneSample.m_logits[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_validationInitScores.push_back(
                              oneLogit - oneSample.m_logits[m_iZeroClassificationLogit]);
                        }
                     }
#else // REDUCE_MULTICLASS_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        m_validationInitScores.push_back(oneLogit);
                     } else {
                        m_validationInitScores.push_back(
                           oneLogit - oneSample.m_initScores[m_iZeroClassificationLogit]);
                     }
#endif // REDUCE_MULTICLASS_LOGITS
                  }
                  ++iLogit;
               }
            }
         } else {
            m_validationRegressionTargets.push_back(target);
            if(!bNullInitScores) {
               const double score = oneSample.m_initScores[0];
               if(std::isnan(score)) {
                  exit(1);
               }
               if(std::isinf(score)) {
                  exit(1);
               }
               m_validationInitScores.push_back(score);
            }
         }
         if(!bNullWeights) {
            const double weight = oneSample.m_weight;
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
         const IntEbmType countBins = m_featureBinCounts[iFeature];
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
   ErrorEbmType error;

   if(Stage::ValidationAdded != m_stage) {
      exit(1);
   }
   if(countInnerBags < IntEbmType { 0 }) {
      exit(1);
   }

   const size_t cVectorLength = GetVectorLength(m_learningTypeOrCountTargetClasses);
   const size_t cFeatures = m_featureBinCounts.size();
   const size_t cTrainingSamples = IsClassification(m_learningTypeOrCountTargetClasses) ? m_trainingClassificationTargets.size() : m_trainingRegressionTargets.size();
   const size_t cValidationSamples = IsClassification(m_learningTypeOrCountTargetClasses) ? m_validationClassificationTargets.size() : m_validationRegressionTargets.size();

   if(m_bNullTrainingInitScores) {
      m_trainingInitScores.resize(cVectorLength * cTrainingSamples);
   }
   if(m_bNullValidationInitScores) {
      m_validationInitScores.resize(cVectorLength * cValidationSamples);
   }
   if(m_bNullTrainingWeights) {
      m_trainingWeights.resize(cTrainingSamples);
   }
   if(m_bNullValidationWeights) {
      m_validationWeights.resize(cValidationSamples);
   }

   IntEbmType size = SizeDataSetHeader(cFeatures, 1, 1);
   for(size_t i = 0; i < cFeatures; ++i) {
      std::vector<IntEbmType> trainingFeatures(m_trainingBinnedData.begin() + i * cTrainingSamples, m_trainingBinnedData.begin() + i * cTrainingSamples + cTrainingSamples);
      std::vector<IntEbmType> validationFeatures(m_validationBinnedData.begin() + i * cValidationSamples, m_validationBinnedData.begin() + i * cValidationSamples + cValidationSamples);

      std::vector<IntEbmType> allFeatures(trainingFeatures);
      allFeatures.insert(allFeatures.end(), validationFeatures.begin(), validationFeatures.end());

      size += SizeFeature(m_featureBinCounts[i], EBM_TRUE, EBM_TRUE, m_featureNominals[i], allFeatures.size(), 0 == allFeatures.size() ? nullptr : &allFeatures[0]);
   }

   std::vector<double> allWeights(m_trainingWeights);
   allWeights.insert(allWeights.end(), m_validationWeights.begin(), m_validationWeights.end());
   size += SizeWeight(allWeights.size(), 0 == allWeights.size() ? nullptr : &allWeights[0]);

   if(IsClassification(m_learningTypeOrCountTargetClasses)) {
      std::vector<IntEbmType> allTargets(m_trainingClassificationTargets);
      allTargets.insert(allTargets.end(), m_validationClassificationTargets.begin(), m_validationClassificationTargets.end());
      size += SizeClassificationTarget(m_learningTypeOrCountTargetClasses, allTargets.size(), 0 == allTargets.size() ? nullptr : &allTargets[0]);
   } else {
      std::vector<double> allTargets(m_trainingRegressionTargets);
      allTargets.insert(allTargets.end(), m_validationRegressionTargets.begin(), m_validationRegressionTargets.end());
      size += SizeRegressionTarget(allTargets.size(), 0 == allTargets.size() ? nullptr : &allTargets[0]);
   }

   void * pDataSet = malloc(static_cast<size_t>(size));

   error = FillDataSetHeader(cFeatures, 1, 1, size, pDataSet);

   for(size_t i = 0; i < cFeatures; ++i) {
      std::vector<IntEbmType> trainingFeatures(m_trainingBinnedData.begin() + i * cTrainingSamples, m_trainingBinnedData.begin() + i * cTrainingSamples + cTrainingSamples);
      std::vector<IntEbmType> validationFeatures(m_validationBinnedData.begin() + i * cValidationSamples, m_validationBinnedData.begin() + i * cValidationSamples + cValidationSamples);

      std::vector<IntEbmType> allFeatures(trainingFeatures);
      allFeatures.insert(allFeatures.end(), validationFeatures.begin(), validationFeatures.end());

      error = FillFeature(m_featureBinCounts[i], EBM_TRUE, EBM_TRUE, m_featureNominals[i], allFeatures.size(), 0 == allFeatures.size() ? nullptr : &allFeatures[0], size, pDataSet);
   }

   error = FillWeight(allWeights.size(), 0 == allWeights.size() ? nullptr : &allWeights[0], size, pDataSet);

   if(IsClassification(m_learningTypeOrCountTargetClasses)) {
      std::vector<IntEbmType> allTargets(m_trainingClassificationTargets);
      allTargets.insert(allTargets.end(), m_validationClassificationTargets.begin(), m_validationClassificationTargets.end());
      error = FillClassificationTarget(m_learningTypeOrCountTargetClasses, allTargets.size(), 0 == allTargets.size() ? nullptr : &allTargets[0], size, pDataSet);
   } else {
      std::vector<double> allTargets(m_trainingRegressionTargets);
      allTargets.insert(allTargets.end(), m_validationRegressionTargets.begin(), m_validationRegressionTargets.end());
      error = FillRegressionTarget(allTargets.size(), 0 == allTargets.size() ? nullptr : &allTargets[0], size, pDataSet);
   }

   std::vector<BagEbmType> bag;
   bag.insert(bag.end(), cTrainingSamples, 1);
   bag.insert(bag.end(), cValidationSamples, -1);

   std::vector<double> allScores(m_trainingInitScores);
   allScores.insert(allScores.end(), m_validationInitScores.begin(), m_validationInitScores.end());

   error = CreateBooster(
      k_randomSeed,
      pDataSet,
      0 == bag.size() ? nullptr : &bag[0],
      0 == allScores.size() ? nullptr : &allScores[0],
      m_dimensionCounts.size(),
      0 == m_dimensionCounts.size() ? nullptr : &m_dimensionCounts[0],
      0 == m_featureIndexes.size() ? nullptr : &m_featureIndexes[0],
      countInnerBags,
      nullptr,
      &m_boosterHandle
   );

   free(pDataSet);

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

BoostRet TestApi::Boost(
   const IntEbmType indexTerm,
   const GenerateUpdateOptionsType options,
   const double learningRate,
   const IntEbmType countSamplesRequiredForChildSplitMin,
   const std::vector<IntEbmType> leavesMax
) {
   ErrorEbmType error;

   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }
   if(indexTerm < IntEbmType { 0 }) {
      exit(1);
   }
   if(m_dimensionCounts.size() <= static_cast<size_t>(indexTerm)) {
      exit(1);
   }
   if(std::isnan(learningRate)) {
      exit(1);
   }
   if(std::isinf(learningRate)) {
      exit(1);
   }
   if(countSamplesRequiredForChildSplitMin < double { 0 }) {
      exit(1);
   }

   double gainAvg = std::numeric_limits<double>::quiet_NaN();
   double validationMetric = std::numeric_limits<double>::quiet_NaN();

   error = GenerateTermUpdate(
      m_boosterHandle,
      indexTerm,
      options,
      learningRate,
      countSamplesRequiredForChildSplitMin,
      0 == leavesMax.size() ? nullptr : &leavesMax[0],
      &gainAvg
   );
   if(Error_None != error) {
      exit(1);
   }
   if(0 != (GenerateUpdateOptions_GradientSums & options)) {
      // if sums are on, then we MUST change the term update

      size_t cUpdateScores = GetVectorLength(m_learningTypeOrCountTargetClasses);
      std::vector<size_t> & dimensionBinCounts = m_termBinCounts[static_cast<size_t>(indexTerm)];

      for(size_t iDimension = 0; iDimension < dimensionBinCounts.size(); ++iDimension) {
         size_t cBins = dimensionBinCounts[iDimension];
         cUpdateScores *= cBins;
      }

      double * aUpdateScores = new double[cUpdateScores];
      memset(aUpdateScores, 0, sizeof(*aUpdateScores) * cUpdateScores);

      error = SetTermUpdateExpanded(
         m_boosterHandle,
         indexTerm,
         aUpdateScores
      );

      delete[] aUpdateScores;

      if(Error_None != error) {
         exit(1);
      }
   }
   error = ApplyTermUpdate(
      m_boosterHandle,
      &validationMetric
   );

   if(Error_None != error) {
      exit(1);
   }
   return BoostRet { gainAvg, validationMetric };
}

double TestApi::GetBestTermScore(
   const size_t iTerm, 
   const std::vector<size_t> indexes, 
   const size_t iScore
) const {
   ErrorEbmType error;

   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }

   if(m_termBinCounts.size() <= iTerm) {
      exit(1);
   }
   const std::vector<size_t> dimensionBinCounts = m_termBinCounts[iTerm];

   const size_t cDimensions = dimensionBinCounts.size();
   size_t multiple = GetVectorLength(m_learningTypeOrCountTargetClasses);
   for(size_t iDimension = 0; iDimension < cDimensions; ++iDimension) {
      multiple *= dimensionBinCounts[iDimension];
   }

   std::vector<double> termScores;
   termScores.resize(multiple);

   error = GetBestTermScores(m_boosterHandle, iTerm, &termScores[0]);
   if(Error_None != error) {
      exit(1);
   }

   const double termScore = GetTermScore(iTerm, &termScores[0], indexes, iScore);
   return termScore;
}

void TestApi::GetBestTermScoresRaw(const size_t iTerm, double * const aTermScores) const {
   ErrorEbmType error;

   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }
   if(m_dimensionCounts.size() <= iTerm) {
      exit(1);
   }
   error = GetBestTermScores(m_boosterHandle, iTerm, aTermScores);
   if(Error_None != error) {
      exit(1);
   }
}

double TestApi::GetCurrentTermScore(
   const size_t iTerm,
   const std::vector<size_t> indexes,
   const size_t iScore
) const {
   ErrorEbmType error;

   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }

   if(m_termBinCounts.size() <= iTerm) {
      exit(1);
   }
   const std::vector<size_t> dimensionBinCounts = m_termBinCounts[iTerm];

   const size_t cDimensions = dimensionBinCounts.size();
   size_t multiple = GetVectorLength(m_learningTypeOrCountTargetClasses);
   for(size_t iDimension = 0; iDimension < cDimensions; ++iDimension) {
      multiple *= dimensionBinCounts[iDimension];
   }

   std::vector<double> termScores;
   termScores.resize(multiple);

   error = GetCurrentTermScores(m_boosterHandle, iTerm, &termScores[0]);
   if(Error_None != error) {
      exit(1);
   }

   const double termScore = GetTermScore(iTerm, &termScores[0], indexes, iScore);
   return termScore;
}

void TestApi::GetCurrentTermScoresRaw(const size_t iTerm, double * const aTermScores) const {
   ErrorEbmType error;

   if(Stage::InitializedBoosting != m_stage) {
      exit(1);
   }
   if(m_dimensionCounts.size() <= iTerm) {
      exit(1);
   }
   error = GetCurrentTermScores(m_boosterHandle, iTerm, aTermScores);
   if(Error_None != error) {
      exit(1);
   }
}

void TestApi::AddInteractionSamples(const std::vector<TestSample> samples) {
   if(Stage::FeaturesAdded != m_stage) {
      exit(1);
   }
   const size_t cSamples = samples.size();
   if(0 != cSamples) {
      const size_t cFeatures = m_featureBinCounts.size();
      const bool bNullInitScores = 0 == samples[0].m_initScores.size();
      m_bNullInteractionInitScores = bNullInitScores;

      const bool bNullWeights = samples[0].m_bNullWeight;
      m_bNullInteractionWeights = bNullWeights;

      for(const TestSample & oneSample : samples) {
         if(cFeatures != oneSample.m_binnedDataPerFeatureArray.size()) {
            exit(1);
         }
         if(bNullInitScores != (0 == oneSample.m_initScores.size())) {
            exit(1);
         }
         if(bNullWeights != oneSample.m_bNullWeight) {
            exit(1);
         }
         const double target = oneSample.m_target;
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
            if(!bNullInitScores) {
               if(static_cast<size_t>(m_learningTypeOrCountTargetClasses) !=
                  oneSample.m_initScores.size()) {
                  exit(1);
               }
               ptrdiff_t iLogit = 0;
               for(const double oneLogit : oneSample.m_initScores) {
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
                        m_interactionInitScores.push_back(oneLogit);
                     } else {
                        m_interactionInitScores.push_back(
                           oneLogit - oneSample.m_initScores[m_iZeroClassificationLogit]);
                     }
#else // EXPAND_BINARY_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_interactionInitScores.push_back(
                              oneLogit - oneSample.m_initScores[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_interactionInitScores.push_back(
                              oneLogit - oneSample.m_initScores[m_iZeroClassificationLogit]);
                        }
                     }
#endif // EXPAND_BINARY_LOGITS
                  } else {
                     // multiclass
#ifdef REDUCE_MULTICLASS_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           m_interactionInitScores.push_back(oneLogit - oneSample.m_logits[0]);
                        }
                     } else {
                        if(m_iZeroClassificationLogit != iLogit) {
                           m_interactionInitScores.push_back(
                              oneLogit - oneSample.m_logits[m_iZeroClassificationLogit]);
                        }
                     }
#else // REDUCE_MULTICLASS_LOGITS
                     if(m_iZeroClassificationLogit < 0) {
                        m_interactionInitScores.push_back(oneLogit);
                     } else {
                        m_interactionInitScores.push_back(
                           oneLogit - oneSample.m_initScores[m_iZeroClassificationLogit]);
                     }
#endif // REDUCE_MULTICLASS_LOGITS
                  }
                  ++iLogit;
               }
            }
         } else {
            m_interactionRegressionTargets.push_back(target);
            if(!bNullInitScores) {
               const double score = oneSample.m_initScores[0];
               if(std::isnan(score)) {
                  exit(1);
               }
               if(std::isinf(score)) {
                  exit(1);
               }
               m_interactionInitScores.push_back(score);
            }
         }
         if(!bNullWeights) {
            const double weight = oneSample.m_weight;
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
         const IntEbmType countBins = m_featureBinCounts[iFeature];
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
   ErrorEbmType error;

   if(Stage::InteractionAdded != m_stage) {
      exit(1);
   }

   const size_t cVectorLength = GetVectorLength(m_learningTypeOrCountTargetClasses);
   const size_t cFeatures = m_featureBinCounts.size();
   const size_t cSamples = IsClassification(m_learningTypeOrCountTargetClasses) ? m_interactionClassificationTargets.size() : m_interactionRegressionTargets.size();

   if(m_bNullInteractionInitScores) {
      m_interactionInitScores.resize(cVectorLength * cSamples);
   }
   if(m_bNullInteractionWeights) {
      m_interactionWeights.resize(cSamples);
   }

   IntEbmType size = SizeDataSetHeader(cFeatures, 1, 1);
   for(size_t i = 0; i < cFeatures; ++i) {
      std::vector<IntEbmType> allFeatures(m_interactionBinnedData.begin() + i * cSamples, m_interactionBinnedData.begin() + i * cSamples + cSamples);
      size += SizeFeature(m_featureBinCounts[i], EBM_TRUE, EBM_TRUE, m_featureNominals[i], allFeatures.size(), 0 == allFeatures.size() ? nullptr : &allFeatures[0]);
   }

   size += SizeWeight(m_interactionWeights.size(), 0 == m_interactionWeights.size() ? nullptr : &m_interactionWeights[0]);

   if(IsClassification(m_learningTypeOrCountTargetClasses)) {
      size += SizeClassificationTarget(m_learningTypeOrCountTargetClasses, cSamples, 0 == cSamples ? nullptr : &m_interactionClassificationTargets[0]);
   } else {
      size += SizeRegressionTarget(cSamples, 0 == cSamples ? nullptr : &m_interactionRegressionTargets[0]);
   }

   void * pDataSet = malloc(static_cast<size_t>(size));

   error = FillDataSetHeader(cFeatures, 1, 1, size, pDataSet);

   for(size_t i = 0; i < cFeatures; ++i) {
      std::vector<IntEbmType> allFeatures(m_interactionBinnedData.begin() + i * cSamples, m_interactionBinnedData.begin() + i * cSamples + cSamples);
      error = FillFeature(m_featureBinCounts[i], EBM_TRUE, EBM_TRUE, m_featureNominals[i], allFeatures.size(), 0 == allFeatures.size() ? nullptr : &allFeatures[0], size, pDataSet);
   }

   error = FillWeight(m_interactionWeights.size(), 0 == m_interactionWeights.size() ? nullptr : &m_interactionWeights[0], size, pDataSet);

   if(IsClassification(m_learningTypeOrCountTargetClasses)) {
      error = FillClassificationTarget(m_learningTypeOrCountTargetClasses, cSamples, 0 == cSamples ? nullptr : &m_interactionClassificationTargets[0], size, pDataSet);
   } else {
      error = FillRegressionTarget(cSamples, 0 == cSamples ? nullptr : &m_interactionRegressionTargets[0], size, pDataSet);
   }

   std::vector<BagEbmType> bag;
   bag.insert(bag.end(), cSamples, 1);

   error = CreateInteractionDetector(
      pDataSet,
      0 == bag.size() ? nullptr : &bag[0],
      0 == m_interactionInitScores.size() ? nullptr : &m_interactionInitScores[0],
      nullptr,
      &m_interactionHandle
   );

   free(pDataSet);

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

double TestApi::TestCalcInteractionStrength(
   const std::vector<IntEbmType> features, 
   const InteractionOptionsType options,
   const IntEbmType countSamplesRequiredForChildSplitMin
) const {
   ErrorEbmType error;

   if(Stage::InitializedInteraction != m_stage) {
      exit(1);
   }
   for(const IntEbmType oneFeatureIndex : features) {
      if(oneFeatureIndex < IntEbmType { 0 }) {
         exit(1);
      }
      if(m_featureBinCounts.size() <= static_cast<size_t>(oneFeatureIndex)) {
         exit(1);
      }
   }

   double avgInteractionStrength = double { 0 };
   error = CalcInteractionStrength(
      m_interactionHandle,
      features.size(),
      0 == features.size() ? nullptr : &features[0],
      options,
      countSamplesRequiredForChildSplitMin,
      &avgInteractionStrength
   );
   if(Error_None != error) {
      exit(1);
   }
   return avgInteractionStrength;
}

extern void DisplayCuts(
   IntEbmType countSamples,
   double * featureValues,
   IntEbmType countBinsMax,
   IntEbmType countSamplesPerBinMin,
   IntEbmType countCuts,
   double * cutsLowerBoundInclusive,
   IntEbmType isMissingPresent,
   double minValue,
   double maxValue
) {
   UNUSED(isMissingPresent);
   UNUSED(minValue);
   UNUSED(maxValue);

   size_t cBinsMax = static_cast<size_t>(countBinsMax);
   size_t cCuts = static_cast<size_t>(countCuts);

   std::vector<double> samples(featureValues, featureValues + countSamples);
   samples.erase(std::remove_if(samples.begin(), samples.end(),
      [](const double & value) { return std::isnan(value); }), samples.end());
   std::sort(samples.begin(), samples.end());

   std::cout << std::endl << std::endl;
   std::cout << "missing=" << (countSamples - samples.size()) << ", countBinsMax=" << countBinsMax << 
      ", countSamplesPerBinMin=" << countSamplesPerBinMin << ", avgBin=" << 
      static_cast<double>(samples.size()) / static_cast<double>(countBinsMax) << std::endl;

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
