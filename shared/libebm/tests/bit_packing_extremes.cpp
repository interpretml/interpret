// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch_test.hpp"

#include "libebm.h"
#include "libebm_test.hpp"

static constexpr TestPriority k_filePriority = TestPriority::BitPackingExtremes;

TEST_CASE("Test data bit packing extremes, boosting, regression") {
   for(size_t exponentialBins = 1; exponentialBins < 10; ++exponentialBins) {
      IntEbm exponential = static_cast<IntEbm>(std::pow(2, exponentialBins));
      // if we set the number of bins to be exponential, then we'll be just under a bit packing boundary.  4 bins means
      // bits packs 00, 01, 10, and 11
      for(IntEbm iRange = IntEbm{-1}; iRange <= IntEbm{1}; ++iRange) {
         IntEbm cBins =
               exponential + iRange; // check one less than the tight fit, the tight fit, and one above the tight fit
         // try everything from 0 samples to 65 samples because for bitpacks with 1 bit, we can have up to 64 packed
         // into a single data value on a 64 bit machine
         for(size_t cSamples = 1; cSamples < 66; ++cSamples) {
            // add one to the bins because our interface needs missing and unseen bins but
            // internally drop the unseen bin to make it possible to have 1 bin

            std::vector<TestSample> trainingSamples;
            std::vector<TestSample> validationSamples;
            for(size_t iSample = 0; iSample < cSamples; ++iSample) {
               trainingSamples.push_back(TestSample({cBins - 1}, 7));
               validationSamples.push_back(TestSample({cBins - 1}, 8));
            }

            TestBoost test = TestBoost(
                  Task_Regression, {FeatureTest(cBins + 1, true, false)}, {{0}}, trainingSamples, validationSamples);

            double validationMetric = test.Boost(0).validationMetric;
            CHECK_APPROX(validationMetric, 62.8849);
            double termScore = test.GetCurrentTermScore(0, {static_cast<size_t>(cBins - 1)}, 0);
            CHECK_APPROX(termScore, 0.07);
         }
      }
   }
}

TEST_CASE("Test data bit packing extremes, boosting, binary") {
   for(size_t exponentialBins = 1; exponentialBins < 10; ++exponentialBins) {
      IntEbm exponential = static_cast<IntEbm>(std::pow(2, exponentialBins));
      // if we set the number of bins to be exponential, then we'll be just under a bit packing boundary.  4 bins means
      // bits packs 00, 01, 10, and 11
      for(IntEbm iRange = IntEbm{-1}; iRange <= IntEbm{1}; ++iRange) {
         IntEbm cBins =
               exponential + iRange; // check one less than the tight fit, the tight fit, and one above the tight fit
         // try everything from 0 samples to 65 samples because for bitpacks with 1 bit, we can have up to 64 packed
         // into a single data value on a 64 bit machine
         for(size_t cSamples = 1; cSamples < 66; ++cSamples) {
            // add one to the bins because our interface needs missing and unseen bins but
            // internally drop the unseen bin to make it possible to have 1 bin

            std::vector<TestSample> trainingSamples;
            std::vector<TestSample> validationSamples;
            for(size_t iSample = 0; iSample < cSamples; ++iSample) {
               trainingSamples.push_back(TestSample({cBins - 1}, 0));
               validationSamples.push_back(TestSample({cBins - 1}, 1));
            }

            TestBoost test = TestBoost(Task_BinaryClassification,
                  {FeatureTest(cBins + 1, true, false)},
                  {{0}},
                  trainingSamples,
                  validationSamples,
                  k_countInnerBagsDefault,
                  k_testCreateBoosterFlags_Default,
                  k_testAccelerationFlags_Default,
                  nullptr,
                  0);

            double validationMetric = test.Boost(0).validationMetric;
            CHECK_APPROX_TOLERANCE(validationMetric, 0.70319717972663420, double{1e-1});

            double termScore;
            termScore = test.GetCurrentTermScore(0, {static_cast<size_t>(cBins - 1)}, 0);
            CHECK(0 == termScore);
            termScore = test.GetCurrentTermScore(0, {static_cast<size_t>(cBins - 1)}, 1);
            CHECK_APPROX_TOLERANCE(termScore, -0.02, double{1e-1});
         }
      }
   }
}

TEST_CASE("Test data bit packing extremes, interaction, regression") {
   for(size_t exponentialBins = 1; exponentialBins < 10; ++exponentialBins) {
      IntEbm exponential = static_cast<IntEbm>(std::pow(2, exponentialBins));
      // if we set the number of bins to be exponential, then we'll be just under a bit packing boundary.  4 bins means
      // bits packs 00, 01, 10, and 11
      for(IntEbm iRange = IntEbm{-1}; iRange <= IntEbm{1}; ++iRange) {
         IntEbm cBins =
               exponential + iRange; // check one less than the tight fit, the tight fit, and one above the tight fit
         // try everything from 0 samples to 65 samples because for bitpacks with 1 bit, we can have up to 64 packed
         // into a single data value on a 64 bit machine
         for(size_t cSamples = 1; cSamples < 66; ++cSamples) {
            // add one to the bins because our interface needs missing and unseen bins but
            // internally drop the unseen bin to make it possible to have 1 bin

            std::vector<TestSample> samples;
            for(size_t iSample = 0; iSample < cSamples; ++iSample) {
               samples.push_back(TestSample({0, cBins - 1}, 7));
            }

            TestInteraction test =
                  TestInteraction(Task_Regression, {FeatureTest(2), FeatureTest(cBins + 1, true, false)}, samples);

            double metric = test.TestCalcInteractionStrength({0, 1});
            CHECK_APPROX(metric, 0);
         }
      }
   }
}

TEST_CASE("Test data bit packing extremes, interaction, binary") {
   for(size_t exponentialBins = 1; exponentialBins < 10; ++exponentialBins) {
      IntEbm exponential = static_cast<IntEbm>(std::pow(2, exponentialBins));
      // if we set the number of bins to be exponential, then we'll be just under a bit packing boundary.  4 bins means
      // bits packs 00, 01, 10, and 11
      for(IntEbm iRange = IntEbm{-1}; iRange <= IntEbm{1}; ++iRange) {
         IntEbm cBins =
               exponential + iRange; // check one less than the tight fit, the tight fit, and one above the tight fit
         // try everything from 0 samples to 65 samples because for bitpacks with 1 bit, we can have up to 64 packed
         // into a single data value on a 64 bit machine
         for(size_t cSamples = 1; cSamples < 66; ++cSamples) {
            // add one to the bins because our interface needs missing and unseen bins but
            // internally drop the unseen bin to make it possible to have 1 bin

            std::vector<TestSample> samples;
            for(size_t iSample = 0; iSample < cSamples; ++iSample) {
               samples.push_back(TestSample({0, cBins - 1}, 1));
            }

            TestInteraction test = TestInteraction(Task_BinaryClassification,
                  {FeatureTest(2), FeatureTest(cBins + 1, true, false)},
                  samples,
                  k_testCreateInteractionFlags_Default,
                  k_testAccelerationFlags_Default,
                  nullptr,
                  0);

            double metric = test.TestCalcInteractionStrength({0, 1});

            CHECK_APPROX(metric, 0);
         }
      }
   }
}
