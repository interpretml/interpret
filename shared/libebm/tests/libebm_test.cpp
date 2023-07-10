// Copyright (c) 2023 The InterpretML Contributors
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

#include "libebm.h"
#include "libebm_test.hpp"

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

void EBM_CALLING_CONVENTION LogCallback(const TraceEbm traceLevel, const char * const message) {
   const size_t cChars = strlen(message); // test that the string memory is accessible
   UNUSED(cChars);
   if(traceLevel <= Trace_Off) {
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

extern bool IsApproxEqual(const double val, const double expected, const double percentage) {
   bool isEqual = false;
   if(!std::isnan(val)) {
      if(!std::isnan(expected)) {
         if(!std::isinf(val)) {
            if(!std::isinf(expected)) {
               const double smaller = double { 1 } - percentage;
               const double bigger = double { 1 } + percentage;
               if(0 < val) {
                  if(0 < expected) {
                     if(val <= expected) {
                        // expected is the bigger number in absolute terms
                        if(expected * smaller <= val && val <= expected * bigger) {
                           isEqual = true;
                        }
                     } else {
                        // val is the bigger number in absolute terms
                        if(val * smaller <= expected && expected <= val * bigger) {
                           isEqual = true;
                        }
                     }
                  }
               } else if(val < 0) {
                  if(expected < 0) {
                     if(expected <= val) {
                        // expected is the bigger number in absolute terms (the biggest negative number)
                        if(expected * bigger <= val && val <= expected * smaller) {
                           isEqual = true;
                        }
                     } else {
                        // val is the bigger number in absolute terms (the biggest negative number)
                        if(val * bigger <= expected && expected <= val * smaller) {
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

const double * TestBoost::GetTermScores(
   const size_t iTerm,
   const double * const aTermScores,
   const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures
) const {
   const size_t cScores = GetCountScores(m_cClasses);

   const std::vector<IntEbm> featureIndexes = m_termFeatures[iTerm];

   const size_t cDimensions = perDimensionIndexArrayForBinnedFeatures.size();
   if(cDimensions != featureIndexes.size()) {
      throw TestException("dimension mismatch");
   }
   size_t iVal = 0;
   size_t multiple = cScores;

   size_t iDimension = 0;
   for(const IntEbm indexFeature : featureIndexes) {
      const size_t iFeature = static_cast<size_t>(indexFeature);
      const size_t cBins = static_cast<size_t>(m_features[iFeature].m_countBins);

      if(cBins <= perDimensionIndexArrayForBinnedFeatures[iDimension]) {
         throw TestException("bin index beyond count of bins");
      }
      iVal += perDimensionIndexArrayForBinnedFeatures[iDimension] * multiple;
      multiple *= cBins;
      ++iDimension;
   }

   return &aTermScores[iVal];
}


double TestBoost::GetTermScore(
   const size_t iTerm,
   const double * const aTermScores,
   const std::vector<size_t> perDimensionIndexArrayForBinnedFeatures,
   const size_t iClassOrZero
) const {
   const double * const aScores = GetTermScores(
      iTerm,
      aTermScores,
      perDimensionIndexArrayForBinnedFeatures
   );
   if(!IsClassification(m_cClasses)) {
      if(0 != iClassOrZero) {
         throw TestException("iClassOrZero cannot be non-zero for regression");
      }
      return aScores[0];
   }
   if(static_cast<size_t>(m_cClasses) <= iClassOrZero) {
      throw TestException("iClassOrZero beyond the number of classes");
   }
   if(OutputType_BinaryClassification == m_cClasses) {
      // binary classification
#ifdef EXPAND_BINARY_LOGITS
      if(m_iZeroClassificationLogit < 0) {
         return aScores[iClassOrZero];
      } else {
         if(static_cast<size_t>(m_iZeroClassificationLogit) == iClassOrZero) {
            return double { 0 };
         } else {
            return aScores[iClassOrZero] - aScores[m_iZeroClassificationLogit];
         }
      }
#else // EXPAND_BINARY_LOGITS
      if(m_iZeroClassificationLogit < 0) {
         if(0 == iClassOrZero) {
            return double { 0 };
         } else {
            return aScores[0];
         }
      } else {
         if(static_cast<size_t>(m_iZeroClassificationLogit) == iClassOrZero) {
            return double { 0 };
         } else {
            return aScores[0];
         }
      }
#endif // EXPAND_BINARY_LOGITS
   } else {
      // multiclass
      if(m_iZeroClassificationLogit < 0) {
         return aScores[iClassOrZero];
      } else {
         return aScores[iClassOrZero] - aScores[m_iZeroClassificationLogit];
      }
   }
}

TestBoost::TestBoost(
   const OutputType cClasses,
   const std::vector<FeatureTest> features,
   const std::vector<std::vector<IntEbm>> termFeatures,
   const std::vector<TestSample> train,
   const std::vector<TestSample> validation,
   const IntEbm countInnerBags,
   const BoolEbm bDifferentiallyPrivate,
   const char * const sObjective,
   const ptrdiff_t iZeroClassificationLogit
) :
   m_cClasses(cClasses),
   m_features(features),
   m_termFeatures(termFeatures),
   m_iZeroClassificationLogit(iZeroClassificationLogit),
   m_boosterHandle(nullptr) 
{
   ErrorEbm error;

   if(IsClassification(cClasses)) {
      if(static_cast<ptrdiff_t>(cClasses) <= iZeroClassificationLogit) {
         throw TestException("bad iZeroClassificationLogit value for classification");
      }
   } else {
      if(ptrdiff_t { -1 } != iZeroClassificationLogit) {
         throw TestException("bad iZeroClassificationLogit value for regression");
      }
   }

   m_rng.resize(static_cast<size_t>(MeasureRNG()));
   InitRNG(k_seed, &m_rng[0]);

   const size_t cSamples = train.size() + validation.size();

   bool bWeight = false;
   bool bInitScores = false;
   for(const TestSample & sample : train) {
      bWeight |= sample.m_bWeight;
      bInitScores |= sample.m_bScores;
   }
   for(const TestSample & sample : validation) {
      bWeight |= sample.m_bWeight;
      bInitScores |= sample.m_bScores;
   }

   std::vector<BagEbm> bag;
   for(const TestSample & sample : train) {
      if(sample.m_bBag) {
         bag.push_back(sample.m_bagCount);
      } else {
         bag.push_back(1);
      }
   }
   for(const TestSample & sample : validation) {
      if(sample.m_bBag) {
         bag.push_back(sample.m_bagCount);
      } else {
         bag.push_back(-1);
      }
   }

   IntEbm size = 0;
   IntEbm sizeChange = MeasureDataSetHeader(features.size(), bWeight ? 1 : 0, 1);
   if(sizeChange < 0) {
      throw TestException(static_cast<ErrorEbm>(sizeChange), "MeasureDataSetHeader");
   }
   size += sizeChange;

   size_t iFeature = 0;
   for(const FeatureTest & feature : features) {
      std::vector<IntEbm> binIndexes;
      for(const TestSample & sample : train) {
         binIndexes.push_back(sample.m_sampleBinIndexes[iFeature]);
      }
      for(const TestSample & sample : validation) {
         binIndexes.push_back(sample.m_sampleBinIndexes[iFeature]);
      }
      sizeChange = MeasureFeature(
         feature.m_countBins, 
         feature.m_bMissing ? EBM_TRUE : EBM_FALSE, 
         feature.m_bUnknown ? EBM_TRUE : EBM_FALSE,
         feature.m_bNominal ? EBM_TRUE : EBM_FALSE,
         cSamples, 
         0 == binIndexes.size() ? nullptr : &binIndexes[0]
      );
      if(sizeChange < 0) {
         throw TestException(static_cast<ErrorEbm>(sizeChange), "MeasureFeature");
      }
      size += sizeChange;

      ++iFeature;
   }

   if(bWeight) {
      std::vector<double> weights;
      for(const TestSample & sample : train) {
         weights.push_back(sample.m_weight);
      }
      for(const TestSample & sample : validation) {
         weights.push_back(sample.m_weight);
      }
      sizeChange = MeasureWeight(weights.size(), &weights[0]);
      if(sizeChange < 0) {
         throw TestException(static_cast<ErrorEbm>(sizeChange), "MeasureWeight");
      }
      size += sizeChange;
   }

   if(IsClassification(cClasses)) {
      std::vector<IntEbm> targets;
      for(const TestSample & sample : train) {
         targets.push_back(static_cast<IntEbm>(sample.m_target));
      }
      for(const TestSample & sample : validation) {
         targets.push_back(static_cast<IntEbm>(sample.m_target));
      }
      sizeChange = MeasureClassificationTarget(cClasses, targets.size(), 0 == targets.size() ? nullptr : &targets[0]);
      if(sizeChange < 0) {
         throw TestException(static_cast<ErrorEbm>(sizeChange), "MeasureClassificationTarget");
      }
      size += sizeChange;
   } else {
      std::vector<double> targets;
      for(const TestSample & sample : train) {
         targets.push_back(sample.m_target);
      }
      for(const TestSample & sample : validation) {
         targets.push_back(sample.m_target);
      }
      sizeChange = MeasureRegressionTarget(targets.size(), 0 == targets.size() ? nullptr : &targets[0]);
      if(sizeChange < 0) {
         throw TestException(static_cast<ErrorEbm>(sizeChange), "MeasureRegressionTarget");
      }
      size += sizeChange;
   }

   std::vector<unsigned char> dataset(static_cast<size_t>(size));

   error = FillDataSetHeader(features.size(), bWeight ? 1 : 0, 1, size, &dataset[0]);
   if(Error_None != error) {
      throw TestException(error, "FillDataSetHeader");
   }

   iFeature = 0;
   for(const FeatureTest & feature : features) {
      std::vector<IntEbm> binIndexes;
      for(const TestSample & sample : train) {
         binIndexes.push_back(sample.m_sampleBinIndexes[iFeature]);
      }
      for(const TestSample & sample : validation) {
         binIndexes.push_back(sample.m_sampleBinIndexes[iFeature]);
      }
      error = FillFeature(
         feature.m_countBins,
         feature.m_bMissing ? EBM_TRUE : EBM_FALSE,
         feature.m_bUnknown ? EBM_TRUE : EBM_FALSE,
         feature.m_bNominal ? EBM_TRUE : EBM_FALSE,
         cSamples,
         0 == binIndexes.size() ? nullptr : &binIndexes[0], 
         size, 
         &dataset[0]
      );
      if(Error_None != error) {
         throw TestException(error, "FillFeature");
      }
      ++iFeature;
   }

   if(bWeight) {
      std::vector<double> weights;
      for(const TestSample & sample : train) {
         weights.push_back(sample.m_weight);
      }
      for(const TestSample & sample : validation) {
         weights.push_back(sample.m_weight);
      }
      error = FillWeight(weights.size(), &weights[0], size, &dataset[0]);
      if(Error_None != error) {
         throw TestException(error, "FillWeight");
      }
   }

   if(IsClassification(cClasses)) {
      std::vector<IntEbm> targets;
      for(const TestSample & sample : train) {
         targets.push_back(static_cast<IntEbm>(sample.m_target));
      }
      for(const TestSample & sample : validation) {
         targets.push_back(static_cast<IntEbm>(sample.m_target));
      }
      error = FillClassificationTarget(cClasses, targets.size(), 0 == targets.size() ? nullptr : &targets[0], size, &dataset[0]);
      if(Error_None != error) {
         throw TestException(error, "FillClassificationTarget");
      }
   } else {
      std::vector<double> targets;
      for(const TestSample & sample : train) {
         targets.push_back(sample.m_target);
      }
      for(const TestSample & sample : validation) {
         targets.push_back(sample.m_target);
      }
      error = FillRegressionTarget(targets.size(), 0 == targets.size() ? nullptr : &targets[0], size, &dataset[0]);
      if(Error_None != error) {
         throw TestException(error, "FillRegressionTarget");
      }
   }

   const size_t cScores = GetCountScores(cClasses);
   std::vector<double> initScores;
   if(bInitScores) {
      if(IsClassification(cClasses)) {
         for(const TestSample & sample : train) {
            if(sample.m_bScores) {
               if(static_cast<size_t>(cClasses) != sample.m_initScores.size()) {
                  throw TestException(error, "cClasses mismatch with sample.m_initScores");
               }
               ptrdiff_t iLogit = 0;
               for(const double oneLogit : sample.m_initScores) {
                  if(OutputType_BinaryClassification == cClasses) {
                     // binary classification
#ifdef EXPAND_BINARY_LOGITS
                     if(iZeroClassificationLogit < 0) {
                        initScores.push_back(oneLogit);
                     } else {
                        initScores.push_back(oneLogit - sample.m_initScores[iZeroClassificationLogit]);
                     }
#else // EXPAND_BINARY_LOGITS
                     if(iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           initScores.push_back(oneLogit - sample.m_initScores[0]);
                        }
                     } else {
                        if(iZeroClassificationLogit != iLogit) {
                           initScores.push_back(oneLogit - sample.m_initScores[iZeroClassificationLogit]);
                        }
                     }
#endif // EXPAND_BINARY_LOGITS
                  } else {
                     // multiclass
                     if(iZeroClassificationLogit < 0) {
                        initScores.push_back(oneLogit);
                     } else {
                        initScores.push_back(oneLogit - sample.m_initScores[iZeroClassificationLogit]);
                     }
                  }
                  ++iLogit;
               }
            } else {
               initScores.insert(initScores.end(), cScores, 0);
            }
         }
         for(const TestSample & sample : validation) {
            if(sample.m_bScores) {
               if(static_cast<size_t>(cClasses) != sample.m_initScores.size()) {
                  throw TestException(error, "cClasses mismatch with sample.m_initScores");
               }
               ptrdiff_t iLogit = 0;
               for(const double oneLogit : sample.m_initScores) {
                  if(OutputType_BinaryClassification == cClasses) {
                     // binary classification
#ifdef EXPAND_BINARY_LOGITS
                     if(iZeroClassificationLogit < 0) {
                        initScores.push_back(oneLogit);
                     } else {
                        initScores.push_back(oneLogit - sample.m_initScores[iZeroClassificationLogit]);
                     }
#else // EXPAND_BINARY_LOGITS
                     if(iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           initScores.push_back(oneLogit - sample.m_initScores[0]);
                        }
                     } else {
                        if(iZeroClassificationLogit != iLogit) {
                           initScores.push_back(oneLogit - sample.m_initScores[iZeroClassificationLogit]);
                        }
                     }
#endif // EXPAND_BINARY_LOGITS
                  } else {
                     // multiclass
                     if(iZeroClassificationLogit < 0) {
                        initScores.push_back(oneLogit);
                     } else {
                        initScores.push_back(oneLogit - sample.m_initScores[iZeroClassificationLogit]);
                     }
                  }
                  ++iLogit;
               }
            } else {
               initScores.insert(initScores.end(), cScores, 0);
            }
         }
      } else {
         for(const TestSample & sample : train) {
            const double score = sample.m_initScores[0];
            initScores.push_back(score);
         }
         for(const TestSample & sample : validation) {
            const double score = sample.m_initScores[0];
            initScores.push_back(score);
         }
      }
   }

   std::vector<IntEbm> dimensionCounts;
   std::vector<IntEbm> allFeatureIndexes;
   for(const std::vector<IntEbm> & featureIndexes : termFeatures) {
      dimensionCounts.push_back(featureIndexes.size());
      for(const IntEbm indexFeature : featureIndexes) {
         allFeatureIndexes.push_back(indexFeature);
      }
   }

   error = CreateBooster(
      &m_rng[0],
      &dataset[0],
      0 == bag.size() ? nullptr : &bag[0],
      bInitScores ? &initScores[0] : nullptr,
      dimensionCounts.size(),
      0 == dimensionCounts.size() ? nullptr : &dimensionCounts[0],
      0 == allFeatureIndexes.size() ? nullptr : &allFeatureIndexes[0],
      countInnerBags,
      bDifferentiallyPrivate,
      nullptr == sObjective ? (IsClassification(cClasses) ? "log_loss" : "rmse") : sObjective,
      nullptr,
      &m_boosterHandle
   );
   if(Error_None != error) {
      throw TestException(error, "CreateBooster");
   }
   if(nullptr == m_boosterHandle) {
      throw TestException("Clean exit with nullptr from CreateBooster.");
   }
}

TestBoost::~TestBoost() {
   if(nullptr != m_boosterHandle) {
      FreeBooster(m_boosterHandle);
   }
}

BoostRet TestBoost::Boost(
   const IntEbm indexTerm,
   const TermBoostFlags flags,
   const double learningRate,
   const IntEbm minSamplesLeaf,
   const std::vector<IntEbm> leavesMax
) {
   ErrorEbm error;

   double gainAvg = std::numeric_limits<double>::quiet_NaN();
   double validationMetricAvg = std::numeric_limits<double>::quiet_NaN();

   error = GenerateTermUpdate(
      &m_rng[0],
      m_boosterHandle,
      indexTerm,
      flags,
      learningRate,
      minSamplesLeaf,
      0 == leavesMax.size() ? nullptr : &leavesMax[0],
      &gainAvg
   );
   if(Error_None != error) {
      throw TestException(error, "GenerateTermUpdate");
   }
   if(0 != (TermBoostFlags_GradientSums & flags)) {
      // if sums are on, then we MUST change the term update

      size_t cUpdateScores = GetCountScores(m_cClasses);

      for(const IntEbm iFeature : m_termFeatures[static_cast<size_t>(indexTerm)]) {
         const size_t cBins = static_cast<size_t>(m_features[static_cast<size_t>(iFeature)].m_countBins);
         cUpdateScores *= cBins;
      }

      std::vector<double> scoreTensor(cUpdateScores);
      if(0 != cUpdateScores) {
         memset(&scoreTensor[0], 0, sizeof(double) * cUpdateScores);
      }

      error = SetTermUpdate(
         m_boosterHandle,
         indexTerm,
         &scoreTensor[0]
      );
      if(Error_None != error) {
         throw TestException(error, "SetTermUpdate");
      }
   }
   error = ApplyTermUpdate(m_boosterHandle, &validationMetricAvg);
   if(Error_None != error) {
      throw TestException(error, "ApplyTermUpdate");
   }

   return BoostRet { gainAvg, validationMetricAvg };
}


double TestBoost::GetBestTermScore(
   const size_t iTerm,
   const std::vector<size_t> indexes,
   const size_t iScore
) const {
   size_t multiple = GetCountScores(m_cClasses);
   for(const IntEbm iFeature : m_termFeatures[static_cast<size_t>(iTerm)]) {
      const size_t cBins = static_cast<size_t>(m_features[static_cast<size_t>(iFeature)].m_countBins);
      multiple *= cBins;
   }

   std::vector<double> termScores(multiple);

   const ErrorEbm error = GetBestTermScores(m_boosterHandle, iTerm, &termScores[0]);
   if(Error_None != error) {
      throw TestException(error, "GetBestTermScores");
   }

   const double termScore = GetTermScore(iTerm, &termScores[0], indexes, iScore);
   return termScore;
}

void TestBoost::GetBestTermScoresRaw(const size_t iTerm, double * const aTermScores) const {
   const ErrorEbm error = GetBestTermScores(m_boosterHandle, iTerm, aTermScores);
   if(Error_None != error) {
      throw TestException(error, "GetBestTermScores");
   }
}

double TestBoost::GetCurrentTermScore(
   const size_t iTerm,
   const std::vector<size_t> indexes,
   const size_t iScore
) const {
   size_t multiple = GetCountScores(m_cClasses);
   for(const IntEbm iFeature : m_termFeatures[static_cast<size_t>(iTerm)]) {
      const size_t cBins = static_cast<size_t>(m_features[static_cast<size_t>(iFeature)].m_countBins);
      multiple *= cBins;
   }

   std::vector<double> termScores(multiple);

   const ErrorEbm error = GetCurrentTermScores(m_boosterHandle, iTerm, &termScores[0]);
   if(Error_None != error) {
      throw TestException(error, "GetCurrentTermScores");
   }

   const double termScore = GetTermScore(iTerm, &termScores[0], indexes, iScore);
   return termScore;
}

void TestBoost::GetCurrentTermScoresRaw(const size_t iTerm, double * const aTermScores) const {
   const ErrorEbm error = GetCurrentTermScores(m_boosterHandle, iTerm, aTermScores);
   if(Error_None != error) {
      throw TestException(error, "GetCurrentTermScores");
   }
}


TestInteraction::TestInteraction(
   const OutputType cClasses,
   const std::vector<FeatureTest> features,
   const std::vector<TestSample> samples,
   const BoolEbm bDifferentiallyPrivate,
   const char * const sObjective,
   const ptrdiff_t iZeroClassificationLogit
) :
   m_interactionHandle(nullptr) 
{
   UNUSED(bDifferentiallyPrivate);

   ErrorEbm error;

   if(IsClassification(cClasses)) {
      if(static_cast<ptrdiff_t>(cClasses) <= iZeroClassificationLogit) {
         throw TestException("bad iZeroClassificationLogit value for classification");
      }
   } else {
      if(ptrdiff_t { -1 } != iZeroClassificationLogit) {
         throw TestException("bad iZeroClassificationLogit value for regression");
      }
   }

   const size_t cSamples = samples.size();

   bool bWeight = false;
   bool bInitScores = false;
   for(const TestSample & sample : samples) {
      bWeight |= sample.m_bWeight;
      bInitScores |= sample.m_bScores;
   }

   std::vector<BagEbm> bag;
   for(const TestSample & sample : samples) {
      if(sample.m_bBag) {
         bag.push_back(sample.m_bagCount);
      } else {
         bag.push_back(1);
      }
   }

   IntEbm size = 0;
   IntEbm sizeChange = MeasureDataSetHeader(features.size(), bWeight ? 1 : 0, 1);
   if(sizeChange < 0) {
      throw TestException(static_cast<ErrorEbm>(sizeChange), "MeasureDataSetHeader");
   }
   size += sizeChange;

   size_t iFeature = 0;
   for(const FeatureTest & feature : features) {
      std::vector<IntEbm> binIndexes;
      for(const TestSample & sample : samples) {
         binIndexes.push_back(sample.m_sampleBinIndexes[iFeature]);
      }
      sizeChange = MeasureFeature(
         feature.m_countBins,
         feature.m_bMissing ? EBM_TRUE : EBM_FALSE,
         feature.m_bUnknown ? EBM_TRUE : EBM_FALSE,
         feature.m_bNominal ? EBM_TRUE : EBM_FALSE,
         cSamples,
         0 == binIndexes.size() ? nullptr : &binIndexes[0]
      );
      if(sizeChange < 0) {
         throw TestException(static_cast<ErrorEbm>(sizeChange), "MeasureFeature");
      }
      size += sizeChange;

      ++iFeature;
   }

   if(bWeight) {
      std::vector<double> weights;
      for(const TestSample & sample : samples) {
         weights.push_back(sample.m_weight);
      }
      sizeChange = MeasureWeight(weights.size(), &weights[0]);
      if(sizeChange < 0) {
         throw TestException(static_cast<ErrorEbm>(sizeChange), "MeasureWeight");
      }
      size += sizeChange;
   }

   if(IsClassification(cClasses)) {
      std::vector<IntEbm> targets;
      for(const TestSample & sample : samples) {
         targets.push_back(static_cast<IntEbm>(sample.m_target));
      }
      sizeChange = MeasureClassificationTarget(cClasses, targets.size(), 0 == targets.size() ? nullptr : &targets[0]);
      if(sizeChange < 0) {
         throw TestException(static_cast<ErrorEbm>(sizeChange), "MeasureClassificationTarget");
      }
      size += sizeChange;
   } else {
      std::vector<double> targets;
      for(const TestSample & sample : samples) {
         targets.push_back(sample.m_target);
      }
      sizeChange = MeasureRegressionTarget(targets.size(), 0 == targets.size() ? nullptr : &targets[0]);
      if(sizeChange < 0) {
         throw TestException(static_cast<ErrorEbm>(sizeChange), "MeasureRegressionTarget");
      }
      size += sizeChange;
   }

   std::vector<unsigned char> dataset(static_cast<size_t>(size));

   error = FillDataSetHeader(features.size(), bWeight ? 1 : 0, 1, size, &dataset[0]);
   if(Error_None != error) {
      throw TestException(error, "FillDataSetHeader");
   }

   iFeature = 0;
   for(const FeatureTest & feature : features) {
      std::vector<IntEbm> binIndexes;
      for(const TestSample & sample : samples) {
         binIndexes.push_back(sample.m_sampleBinIndexes[iFeature]);
      }
      error = FillFeature(
         feature.m_countBins,
         feature.m_bMissing ? EBM_TRUE : EBM_FALSE,
         feature.m_bUnknown ? EBM_TRUE : EBM_FALSE,
         feature.m_bNominal ? EBM_TRUE : EBM_FALSE,
         cSamples,
         0 == binIndexes.size() ? nullptr : &binIndexes[0],
         size,
         &dataset[0]
      );
      if(Error_None != error) {
         throw TestException(error, "FillFeature");
      }
      ++iFeature;
   }

   if(bWeight) {
      std::vector<double> weights;
      for(const TestSample & sample : samples) {
         weights.push_back(sample.m_weight);
      }
      error = FillWeight(weights.size(), &weights[0], size, &dataset[0]);
      if(Error_None != error) {
         throw TestException(error, "FillWeight");
      }
   }

   if(IsClassification(cClasses)) {
      std::vector<IntEbm> targets;
      for(const TestSample & sample : samples) {
         targets.push_back(static_cast<IntEbm>(sample.m_target));
      }
      error = FillClassificationTarget(cClasses, targets.size(), 0 == targets.size() ? nullptr : &targets[0], size, &dataset[0]);
      if(Error_None != error) {
         throw TestException(error, "FillClassificationTarget");
      }
   } else {
      std::vector<double> targets;
      for(const TestSample & sample : samples) {
         targets.push_back(sample.m_target);
      }
      error = FillRegressionTarget(targets.size(), 0 == targets.size() ? nullptr : &targets[0], size, &dataset[0]);
      if(Error_None != error) {
         throw TestException(error, "FillRegressionTarget");
      }
   }

   const size_t cScores = GetCountScores(cClasses);
   std::vector<double> initScores;
   if(bInitScores) {
      if(IsClassification(cClasses)) {
         for(const TestSample & sample : samples) {
            if(sample.m_bScores) {
               if(static_cast<size_t>(cClasses) != sample.m_initScores.size()) {
                  throw TestException(error, "cClasses mismatch with sample.m_initScores");
               }
               ptrdiff_t iLogit = 0;
               for(const double oneLogit : sample.m_initScores) {
                  if(OutputType_BinaryClassification == cClasses) {
                     // binary classification
#ifdef EXPAND_BINARY_LOGITS
                     if(iZeroClassificationLogit < 0) {
                        initScores.push_back(oneLogit);
                     } else {
                        initScores.push_back(oneLogit - sample.m_initScores[iZeroClassificationLogit]);
                     }
#else // EXPAND_BINARY_LOGITS
                     if(iZeroClassificationLogit < 0) {
                        if(0 != iLogit) {
                           initScores.push_back(oneLogit - sample.m_initScores[0]);
                        }
                     } else {
                        if(iZeroClassificationLogit != iLogit) {
                           initScores.push_back(oneLogit - sample.m_initScores[iZeroClassificationLogit]);
                        }
                     }
#endif // EXPAND_BINARY_LOGITS
                  } else {
                     // multiclass
                     if(iZeroClassificationLogit < 0) {
                        initScores.push_back(oneLogit);
                     } else {
                        initScores.push_back(oneLogit - sample.m_initScores[iZeroClassificationLogit]);
                     }
                  }
                  ++iLogit;
               }
            } else {
               initScores.insert(initScores.end(), cScores, 0);
            }
         }
      } else {
         for(const TestSample & sample : samples) {
            const double score = sample.m_initScores[0];
            initScores.push_back(score);
         }
      }
   }

   error = CreateInteractionDetector(
      &dataset[0],
      0 == bag.size() ? nullptr : &bag[0],
      0 == initScores.size() ? nullptr : &initScores[0],
      EBM_FALSE,
      nullptr == sObjective ? (IsClassification(cClasses) ? "log_loss" : "rmse") : sObjective,
      nullptr,
      &m_interactionHandle
   );

   if(Error_None != error) {
      throw TestException(error, "CreateInteractionDetector");
   }
   if(nullptr == m_interactionHandle) {
      throw TestException("Clean exit with nullptr from CreateInteractionDetector.");
   }
}

TestInteraction::~TestInteraction() {
   if(nullptr != m_interactionHandle) {
      FreeInteractionDetector(m_interactionHandle);
   }
}

double TestInteraction::TestCalcInteractionStrength(
   const std::vector<IntEbm> features,
   const CalcInteractionFlags flags,
   const IntEbm minSamplesLeaf
) const {
   double avgInteractionStrength = double { 0 };
   const ErrorEbm error = CalcInteractionStrength(
      m_interactionHandle,
      features.size(),
      0 == features.size() ? nullptr : &features[0],
      flags,
      0,
      minSamplesLeaf,
      &avgInteractionStrength
   );
   if(Error_None != error) {
      throw TestException(error, "CalcInteractionStrength");
   }
   return avgInteractionStrength;
}


extern void DisplayCuts(
   IntEbm countSamples,
   double * featureVals,
   IntEbm countBinsMax,
   IntEbm minSamplesBin,
   IntEbm countCuts,
   double * cutsLowerBoundInclusive,
   IntEbm isMissingPresent,
   double minFeatureVal,
   double maxFeatureVal
) {
   UNUSED(isMissingPresent);
   UNUSED(minFeatureVal);
   UNUSED(maxFeatureVal);

   size_t cBinsMax = static_cast<size_t>(countBinsMax);
   size_t cCuts = static_cast<size_t>(countCuts);

   std::vector<double> samples(featureVals, featureVals + countSamples);
   samples.erase(std::remove_if(samples.begin(), samples.end(),
      [](const double & val) { return std::isnan(val); }), samples.end());
   std::sort(samples.begin(), samples.end());

   std::cout << std::endl << std::endl;
   std::cout << "missing=" << (countSamples - samples.size()) << ", countBinsMax=" << countBinsMax << 
      ", minSamplesBin=" << minSamplesBin << ", avgBin=" << 
      static_cast<double>(samples.size()) / static_cast<double>(countBinsMax) << std::endl;

   size_t iCut = 0;
   size_t cInBin = 0;
   for(double val: samples) {
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

   SetLogCallback(&LogCallback);
   SetTraceLevel(Trace_Verbose);

   std::vector<TestCaseHidden> g_allTestsHidden = GetAllTestsHidden();
   std::stable_sort(g_allTestsHidden.begin(), g_allTestsHidden.end(),
      [](const TestCaseHidden & lhs, const TestCaseHidden & rhs) {
         return lhs.m_testPriority < rhs.m_testPriority;
      });

   bool bPassed = true;
   for(TestCaseHidden& testCaseHidden : g_allTestsHidden) {
      std::cout << "Starting test: " << testCaseHidden.m_description;
      try {
         testCaseHidden.m_pTestFunction(testCaseHidden);
         if(testCaseHidden.m_bPassed) {
            std::cout << " PASSED" << std::endl;
         } else {
            bPassed = false;
            // any failures (there can be multiple) have already been written out
            std::cout << std::endl;
         }
      } catch(const TestException & except) {
         bPassed = false;
         std::cout << " FAILED on \"" << except.GetMessage() << "\"" << std::endl;
      } catch(const std::bad_alloc &) {
         bPassed = false;
         std::cout << "out of memory";
         std::cout << std::endl;
      } catch(const std::exception &) {
         bPassed = false;
         std::cout << "generic exception in the test framework";
         std::cout << std::endl;
      } catch(...) {
         bPassed = false;
         std::cout << "unknown exception in the test framework";
         std::cout << std::endl;
      }
   }

   std::cout << "C API test " << (bPassed ? "PASSED" : "FAILED") << std::endl;
   return bPassed ? 0 : 1;
}
