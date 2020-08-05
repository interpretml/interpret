// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderEbmNativeTest.h"

#include "ebm_native.h"
#include "EbmNativeTest.h"

static const TestPriority k_filePriority = TestPriority::BitPackingExtremes;


TEST_CASE("Test data bit packing extremes, boosting, regression") {
   for(size_t exponentialBins = 1; exponentialBins < 10; ++exponentialBins) {
      IntEbmType exponential = static_cast<IntEbmType>(std::pow(2, exponentialBins));
      // if we set the number of bins to be exponential, then we'll be just under a bit packing boundary.  4 bins means bits packs 00, 01, 10, and 11
      for(IntEbmType iRange = IntEbmType { -1 }; iRange <= IntEbmType { 1 }; ++iRange) {
         IntEbmType cBins = exponential + iRange; // check one less than the tight fit, the tight fit, and one above the tight fit
         // try everything from 0 samples to 65 samples because for bitpacks with 1 bit, we can have up to 64 packed into a single data value on a 
         // 64 bit machine
         for(size_t cSamples = 1; cSamples < 66; ++cSamples) {
            TestApi test = TestApi(k_learningTypeRegression);
            test.AddFeatures({ FeatureTest(cBins) });
            test.AddFeatureGroups({ { 0 } });

            std::vector<RegressionSample> trainingSamples;
            std::vector<RegressionSample> validationSamples;
            for(size_t iSample = 0; iSample < cSamples; ++iSample) {
               trainingSamples.push_back(RegressionSample(7, { cBins - 1 }));
               validationSamples.push_back(RegressionSample(8, { cBins - 1 }));
            }
            test.AddTrainingSamples(trainingSamples);
            test.AddValidationSamples(validationSamples);
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
         // try everything from 0 samples to 65 samples because for bitpacks with 1 bit, we can have up to 64 packed into a single data value on 
         // a 64 bit machine
         for(size_t cSamples = 1; cSamples < 66; ++cSamples) {
            TestApi test = TestApi(2, 0);
            test.AddFeatures({ FeatureTest(cBins) });
            test.AddFeatureGroups({ { 0 } });

            std::vector<ClassificationSample> trainingSamples;
            std::vector<ClassificationSample> validationSamples;
            for(size_t iSample = 0; iSample < cSamples; ++iSample) {
               trainingSamples.push_back(ClassificationSample(0, { cBins - 1 }));
               validationSamples.push_back(ClassificationSample(1, { cBins - 1 }));
            }
            test.AddTrainingSamples(trainingSamples);
            test.AddValidationSamples(validationSamples);
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
         // try everything from 0 samples to 65 samples because for bitpacks with 1 bit, we can have up to 64 packed into a single data value on 
         // a 64 bit machine
         for(size_t cSamples = 1; cSamples < 66; ++cSamples) {
            TestApi test = TestApi(k_learningTypeRegression);
            test.AddFeatures({ FeatureTest(2), FeatureTest(cBins) });

            std::vector<RegressionSample> samples;
            for(size_t iSample = 0; iSample < cSamples; ++iSample) {
               samples.push_back(RegressionSample(7, { 0, cBins - 1 }));
            }
            test.AddInteractionSamples(samples);
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
         // try everything from 0 samples to 65 samples because for bitpacks with 1 bit, we can have up to 64 packed into a single data value on 
         // a 64 bit machine
         for(size_t cSamples = 1; cSamples < 66; ++cSamples) {
            TestApi test = TestApi(2, 0);
            test.AddFeatures({ FeatureTest(2), FeatureTest(cBins) });

            std::vector<ClassificationSample> samples;
            for(size_t iSample = 0; iSample < cSamples; ++iSample) {
               samples.push_back(ClassificationSample(1, { 0, cBins - 1 }));
            }
            test.AddInteractionSamples(samples);
            test.InitializeInteraction();

            FloatEbmType metric = test.InteractionScore({ 0, 1 });

            CHECK_APPROX(metric, 0);
         }
      }
   }
}


