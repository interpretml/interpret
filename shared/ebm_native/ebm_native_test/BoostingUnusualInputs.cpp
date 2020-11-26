// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderEbmNativeTest.h"

#include "ebm_native.h"
#include "EbmNativeTest.h"

static const TestPriority k_filePriority = TestPriority::BoostingUnusualInputs;

TEST_CASE("null validationMetricOut, boosting, regression") {
   EbmNativeFeatureGroup groups[1];
   groups->countFeaturesInGroup = 0;

   PEbmBoosting pEbmBoosting = InitializeBoostingRegression(
      k_randomSeed,
      0,
      nullptr,
      1,
      groups,
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
      nullptr
   );
   const IntEbmType ret = BoostingStep(
      pEbmBoosting,
      0,
      k_learningRateDefault,
      k_countTreeSplitsMaxDefault,
      k_countSamplesRequiredForChildSplitMinDefault,
      GenerateUpdateOptions_Default,
      nullptr,
      nullptr,
      nullptr
   );
   CHECK(0 == ret);
   FreeBoosting(pEbmBoosting);
}

TEST_CASE("null validationMetricOut, boosting, binary") {
   EbmNativeFeatureGroup groups[1];
   groups->countFeaturesInGroup = 0;

   PEbmBoosting pEbmBoosting = InitializeBoostingClassification(
      k_randomSeed,
      2,
      0,
      nullptr,
      1,
      groups,
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
      nullptr
   );
   const IntEbmType ret = BoostingStep(
      pEbmBoosting,
      0,
      k_learningRateDefault,
      k_countTreeSplitsMaxDefault,
      k_countSamplesRequiredForChildSplitMinDefault,
      GenerateUpdateOptions_Default,
      nullptr,
      nullptr,
      nullptr
   );
   CHECK(0 == ret);
   FreeBoosting(pEbmBoosting);
}

TEST_CASE("null validationMetricOut, boosting, multiclass") {
   EbmNativeFeatureGroup groups[1];
   groups->countFeaturesInGroup = 0;

   PEbmBoosting pEbmBoosting = InitializeBoostingClassification(
      k_randomSeed,
      3,
      0,
      nullptr,
      1,
      groups,
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
      nullptr
   );
   const IntEbmType ret = BoostingStep(
      pEbmBoosting,
      0,
      k_learningRateDefault,
      k_countTreeSplitsMaxDefault,
      k_countSamplesRequiredForChildSplitMinDefault,
      GenerateUpdateOptions_Default,
      nullptr,
      nullptr,
      nullptr
   );
   CHECK(0 == ret);
   FreeBoosting(pEbmBoosting);
}

TEST_CASE("zero learning rate, boosting, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddFeatureGroups({ {} });
   test.AddTrainingSamples({ RegressionSample(10, {}) });
   test.AddValidationSamples({ RegressionSample(12, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup, {}, {}, GenerateUpdateOptions_Default, 0);
         CHECK_APPROX(validationMetric, 144);
         modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
         CHECK_APPROX(modelValue, 0);

         modelValue = test.GetBestModelPredictorScore(iFeatureGroup, {}, 0);
         CHECK_APPROX(modelValue, 0);
      }
   }
}

TEST_CASE("zero learning rate, boosting, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({});
   test.AddFeatureGroups({ {} });
   test.AddTrainingSamples({ ClassificationSample(0, {}) });
   test.AddValidationSamples({ ClassificationSample(0, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup, {}, {}, GenerateUpdateOptions_Default, 0);
         CHECK_APPROX_TOLERANCE(validationMetric, 0.69314718055994529, double { 1e-1 });
         modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
         CHECK_APPROX(modelValue, 0);
         modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1);
         CHECK_APPROX(modelValue, 0);

         modelValue = test.GetBestModelPredictorScore(iFeatureGroup, {}, 0);
         CHECK_APPROX(modelValue, 0);
         modelValue = test.GetBestModelPredictorScore(iFeatureGroup, {}, 1);
         CHECK_APPROX(modelValue, 0);
      }
   }
}

TEST_CASE("zero learning rate, boosting, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({});
   test.AddFeatureGroups({ {} });
   test.AddTrainingSamples({ ClassificationSample(0, {}) });
   test.AddValidationSamples({ ClassificationSample(0, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup, {}, {}, GenerateUpdateOptions_Default, 0);
         CHECK_APPROX_TOLERANCE(validationMetric, 1.0986122886681098, double { 1e-1 });
         modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
         CHECK_APPROX(modelValue, 0);
         modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1);
         CHECK_APPROX(modelValue, 0);
         modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 2);
         CHECK_APPROX(modelValue, 0);

         modelValue = test.GetBestModelPredictorScore(iFeatureGroup, {}, 0);
         CHECK_APPROX(modelValue, 0);
         modelValue = test.GetBestModelPredictorScore(iFeatureGroup, {}, 1);
         CHECK_APPROX(modelValue, 0);
         modelValue = test.GetBestModelPredictorScore(iFeatureGroup, {}, 2);
         CHECK_APPROX(modelValue, 0);
      }
   }
}

TEST_CASE("negative learning rate, boosting, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddFeatureGroups({ {} });
   test.AddTrainingSamples({ RegressionSample(10, {}) });
   test.AddValidationSamples({ RegressionSample(12, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup, {}, {}, GenerateUpdateOptions_Default, -k_learningRateDefault);
         if(0 == iFeatureGroup && 0 == iEpoch) {
            CHECK_APPROX(validationMetric, 146.41);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
            CHECK_APPROX(modelValue, -0.1000000000000000);
         }
         if(0 == iFeatureGroup && 1 == iEpoch) {
            CHECK_APPROX(validationMetric, 148.864401);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
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
   test.AddFeatureGroups({ {} });
   test.AddTrainingSamples({ ClassificationSample(0, {}) });
   test.AddValidationSamples({ ClassificationSample(0, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 50; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup, {}, {}, GenerateUpdateOptions_Default, -k_learningRateDefault);
         if(0 == iFeatureGroup && 0 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 0.70319717972663420, double { 1e-1 });
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
            CHECK_APPROX(modelValue, 0);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1);
            CHECK_APPROX_TOLERANCE(modelValue, 0.020000000000000000, double { 1.5e-1 });
         }
         if(0 == iFeatureGroup && 1 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 0.71345019889199235, double { 1e-1 });
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
            CHECK_APPROX(modelValue, 0);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1);
            CHECK_APPROX_TOLERANCE(modelValue, 0.040202013400267564, double { 1.5e-1 });
         }
      }
   }

   CHECK_APPROX_TOLERANCE(validationMetric, 1.7158914513238979, double { 1e-1 });
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 0);
   CHECK_APPROX(modelValue, 0);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 1);
   CHECK_APPROX_TOLERANCE(modelValue, 1.5176802847035755, double { 1e-2 });
}

TEST_CASE("negative learning rate, boosting, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({});
   test.AddFeatureGroups({ {} });
   test.AddTrainingSamples({ ClassificationSample(0, {}) });
   test.AddValidationSamples({ ClassificationSample(0, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 20; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup, {}, {}, GenerateUpdateOptions_Default, -k_learningRateDefault);
         if(0 == iFeatureGroup && 0 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 1.1288361512023379, double { 1e-1 });
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
            CHECK_APPROX(modelValue, -0.03000000000000000);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1);
            CHECK_APPROX(modelValue, 0.01500000000000000);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 2);
            CHECK_APPROX(modelValue, 0.01500000000000000);
         }
         if(0 == iFeatureGroup && 1 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 1.1602122411839852, double { 1e-1 });
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
            CHECK_APPROX_TOLERANCE(modelValue, -0.060920557198174352, double { 1e-2 });
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1);
            CHECK_APPROX_TOLERANCE(modelValue, 0.030112481019468545, double { 1e-2 });
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 2);
            CHECK_APPROX_TOLERANCE(modelValue, 0.030112481019468545, double { 1e-2 });
         }
      }
   }
   CHECK_APPROX_TOLERANCE(validationMetric, 2.0611718475324357, double { 1e-1 });
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 0);
   CHECK_APPROX_TOLERANCE(modelValue, -0.90755332487264362, double { 1e-2 });
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 1);
   CHECK_APPROX_TOLERANCE(modelValue, 0.32430253082567057, double { 1e-2 });
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 2);
   CHECK_APPROX_TOLERANCE(modelValue, 0.32430253082567057, double { 1e-2 });
}

TEST_CASE("zero countSamplesRequiredForChildSplitMin, boosting, regression") {
   // TODO : call test.Boost many more times in a loop, and verify the output remains the same as previous runs
   // TODO : add classification binary and multiclass versions of this

   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples({
      RegressionSample(10, { 0 }),
      RegressionSample(10, { 1 }),
      });
   test.AddValidationSamples({ RegressionSample(12, { 1 }) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = test.Boost(0, {}, {}, GenerateUpdateOptions_Default, k_learningRateDefault, k_countTreeSplitsMaxDefault, 0);
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
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples({
      RegressionSample(10, { 0 }),
      RegressionSample(10, { 1 }),
      });
   test.AddValidationSamples({ RegressionSample(12, { 1 }) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = test.Boost(0, {}, {}, GenerateUpdateOptions_Default, k_learningRateDefault, 0);
   CHECK_APPROX(validationMetric, 141.61);
   FloatEbmType modelValue;
   modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
   CHECK_APPROX(modelValue, 0.1000000000000000);
   CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 0));
}

TEST_CASE("Zero training samples, boosting, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples(std::vector<RegressionSample> {});
   test.AddValidationSamples({ RegressionSample(12, { 1 }) });
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

TEST_CASE("Zero training samples, boosting, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples(std::vector<ClassificationSample> {});
   test.AddValidationSamples({ ClassificationSample(0, { 1 }) });
   test.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FloatEbmType validationMetric = test.Boost(0);
      CHECK_APPROX_TOLERANCE(validationMetric, 0.69314718055994529, double { 1e-1 });
      FloatEbmType modelValue;
      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
      CHECK_APPROX(modelValue, 0);
      CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 0));

      modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 1);
      CHECK_APPROX(modelValue, 0);
      CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 1));
   }
}

TEST_CASE("Zero training samples, boosting, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples(std::vector<ClassificationSample> {});
   test.AddValidationSamples({ ClassificationSample(0, { 1 }) });
   test.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FloatEbmType validationMetric = test.Boost(0);
      CHECK_APPROX_TOLERANCE(validationMetric, 1.0986122886681098, double { 1e-1 });
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

TEST_CASE("Zero validation samples, boosting, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples({ RegressionSample(10, { 1 }) });
   test.AddValidationSamples(std::vector<RegressionSample> {});
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

TEST_CASE("Zero validation samples, boosting, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples({ ClassificationSample(0, { 1 }) });
   test.AddValidationSamples(std::vector<ClassificationSample> {});
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
         CHECK_APPROX_TOLERANCE(modelValue, -0.020000000000000000, double { 1.5e-1 });
      }
      if(1 == iEpoch) {
         CHECK_APPROX_TOLERANCE(modelValue, -0.039801986733067563, double { 1e-1 });
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

TEST_CASE("Zero validation samples, boosting, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples({ ClassificationSample(0, { 1 }) });
   test.AddValidationSamples(std::vector<ClassificationSample> {});
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
         CHECK_APPROX_TOLERANCE(modelValue, 0.059119949636662006, double { 1e-2 });
         CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 0));
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 1);
         CHECK_APPROX_TOLERANCE(modelValue, -0.029887518980531450, double { 1e-2 });
         CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 1));
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 2);
         CHECK_APPROX_TOLERANCE(modelValue, -0.029887518980531450, double { 1e-2 });
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

TEST_CASE("features with 0 states, boosting") {
   // for there to be zero states, there can't be an training data or testing data since then those would be required to have a value for the state
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(0) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples(std::vector<RegressionSample> {});
   test.AddValidationSamples(std::vector<RegressionSample> {});
   test.InitializeBoosting();

   FloatEbmType validationMetric = test.Boost(0);
   CHECK(0 == validationMetric);

   // we're not sure what we'd get back since we aren't allowed to access it, so don't do anything with the return value.  We just want to make sure 
   // calling to get the models doesn't crash
   test.GetBestModelFeatureGroupRaw(0);
   test.GetCurrentModelFeatureGroupRaw(0);
}

TEST_CASE("features with 0 states, interaction") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(0) });
   test.AddInteractionSamples(std::vector<RegressionSample> {});
   test.InitializeInteraction();

   FloatEbmType validationMetric = test.InteractionScore({ 0 });
   CHECK(0 == validationMetric);
}

TEST_CASE("classification with 0 possible target states, boosting") {
   // for there to be zero states, there can't be an training data or testing data since then those would be required to have a value for the state
   TestApi test = TestApi(0);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples(std::vector<ClassificationSample> {});
   test.AddValidationSamples(std::vector<ClassificationSample> {});
   test.InitializeBoosting();

   CHECK(nullptr == test.GetBestModelFeatureGroupRaw(0));
   CHECK(nullptr == test.GetCurrentModelFeatureGroupRaw(0));

   FloatEbmType validationMetric = test.Boost(0);
   CHECK(0 == validationMetric);

   CHECK(nullptr == test.GetBestModelFeatureGroupRaw(0));
   CHECK(nullptr == test.GetCurrentModelFeatureGroupRaw(0));
}

TEST_CASE("classification with 1 possible target, boosting") {
   TestApi test = TestApi(1);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples({ ClassificationSample(0, { 1 }) });
   test.AddValidationSamples({ ClassificationSample(0, { 1 }) });
   test.InitializeBoosting();

   CHECK(nullptr == test.GetBestModelFeatureGroupRaw(0));
   CHECK(nullptr == test.GetCurrentModelFeatureGroupRaw(0));

   FloatEbmType validationMetric = test.Boost(0);
   CHECK(0 == validationMetric);

   CHECK(nullptr == test.GetBestModelFeatureGroupRaw(0));
   CHECK(nullptr == test.GetCurrentModelFeatureGroupRaw(0));
}

TEST_CASE("features with 1 state in various positions, boosting") {
   TestApi test0 = TestApi(k_learningTypeRegression);
   test0.AddFeatures({
      FeatureTest(1),
      FeatureTest(2),
      FeatureTest(2)
      });
   test0.AddFeatureGroups({ { 0 }, { 1 }, { 2 } });
   test0.AddTrainingSamples({ RegressionSample(10, { 0, 1, 1 }) });
   test0.AddValidationSamples({ RegressionSample(12, { 0, 1, 1 }) });
   test0.InitializeBoosting();

   TestApi test1 = TestApi(k_learningTypeRegression);
   test1.AddFeatures({
      FeatureTest(2),
      FeatureTest(1),
      FeatureTest(2)
      });
   test1.AddFeatureGroups({ { 0 }, { 1 }, { 2 } });
   test1.AddTrainingSamples({ RegressionSample(10, { 1, 0, 1 }) });
   test1.AddValidationSamples({ RegressionSample(12, { 1, 0, 1 }) });
   test1.InitializeBoosting();

   TestApi test2 = TestApi(k_learningTypeRegression);
   test2.AddFeatures({
      FeatureTest(2),
      FeatureTest(2),
      FeatureTest(1)
      });
   test2.AddFeatureGroups({ { 0 }, { 1 }, { 2 } });
   test2.AddTrainingSamples({ RegressionSample(10, { 1, 1, 0 }) });
   test2.AddValidationSamples({ RegressionSample(12, { 1, 1, 0 }) });
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

TEST_CASE("zero FeatureGroups, boosting, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddFeatureGroups({});
   test.AddTrainingSamples({ RegressionSample(10, {}) });
   test.AddValidationSamples({ RegressionSample(12, {}) });
   test.InitializeBoosting();

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just crash/assert if there was a problem
   // boosting isn't legal since we'd need to specify an featureGroup index
}

TEST_CASE("zero FeatureGroups, boosting, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({});
   test.AddFeatureGroups({});
   test.AddTrainingSamples({ ClassificationSample(1, {}) });
   test.AddValidationSamples({ ClassificationSample(1, {}) });
   test.InitializeBoosting();

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just crash/assert if there was a problem
   // boosting isn't legal since we'd need to specify an featureGroup index
}

TEST_CASE("zero FeatureGroups, boosting, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({});
   test.AddFeatureGroups({});
   test.AddTrainingSamples({ ClassificationSample(2, {}) });
   test.AddValidationSamples({ ClassificationSample(2, {}) });
   test.InitializeBoosting();

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just crash/assert if there was a problem
   // boosting isn't legal since we'd need to specify an featureGroup index
}

TEST_CASE("FeatureGroup with zero features, boosting, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddFeatureGroups({ {} });
   test.AddTrainingSamples({ RegressionSample(10, {}) });
   test.AddValidationSamples({ RegressionSample(12, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup);
         if(0 == iFeatureGroup && 0 == iEpoch) {
            CHECK_APPROX(validationMetric, 141.61);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
            CHECK_APPROX(modelValue, 0.1000000000000000);
         }
         if(0 == iFeatureGroup && 1 == iEpoch) {
            CHECK_APPROX(validationMetric, 139.263601);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
            CHECK_APPROX(modelValue, 0.1990000000000000);
         }
      }
   }
   CHECK_APPROX(validationMetric, 4.001727036272099502004735302456);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 0);
   CHECK_APPROX(modelValue, 9.9995682875258822);
}

TEST_CASE("FeatureGroup with zero features, boosting, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({});
   test.AddFeatureGroups({ {} });
   test.AddTrainingSamples({ ClassificationSample(0, {}) });
   test.AddValidationSamples({ ClassificationSample(0, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup);
         if(0 == iFeatureGroup && 0 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 0.68319717972663419, double { 1e-1 });
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
            CHECK_APPROX(modelValue, 0);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1);
            CHECK_APPROX_TOLERANCE(modelValue, -0.020000000000000000, double { 1e-1 });
         }
         if(0 == iFeatureGroup && 1 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 0.67344419889200957, double { 1e-1 });
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
            CHECK_APPROX(modelValue, 0);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1);
            CHECK_APPROX_TOLERANCE(modelValue, -0.039801986733067563, double { 1e-1 });
         }
      }
   }
   CHECK_APPROX_TOLERANCE(validationMetric, 2.2621439908125974e-05, double { 1e+1 });
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 0);
   CHECK_APPROX(modelValue, 0);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 1);
   CHECK_APPROX_TOLERANCE(modelValue, -10.696601122148364, double { 1e-2 });
}

TEST_CASE("FeatureGroup with zero features, boosting, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({});
   test.AddFeatureGroups({ {} });
   test.AddTrainingSamples({ ClassificationSample(0, {}) });
   test.AddValidationSamples({ ClassificationSample(0, {}) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup);
         if(0 == iFeatureGroup && 0 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 1.0688384008227103, double { 1e-1 });
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
            CHECK_APPROX(modelValue, 0.03000000000000000);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1);
            CHECK_APPROX(modelValue, -0.01500000000000000);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 2);
            CHECK_APPROX(modelValue, -0.01500000000000000);
         }
         if(0 == iFeatureGroup && 1 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 1.0401627411809615, double { 1e-1 });
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
            CHECK_APPROX_TOLERANCE(modelValue, 0.059119949636662006, double { 1e-2 });
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1);
            CHECK_APPROX_TOLERANCE(modelValue, -0.029887518980531450, double { 1e-2 });
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 2);
            CHECK_APPROX_TOLERANCE(modelValue, -0.029887518980531450, double { 1e-2 });
         }
      }
   }
   CHECK_APPROX_TOLERANCE(validationMetric, 1.7171897252232722e-09, double { 1e+1 });
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 0);
   CHECK_APPROX_TOLERANCE(modelValue, 10.643234965479628, double { 1e-3 });
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 1);
   CHECK_APPROX_TOLERANCE(modelValue, -10.232489007525166, double { 1e-3 });
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 2);
   CHECK_APPROX_TOLERANCE(modelValue, -10.232489007525166, double { 1e-3 });
}

TEST_CASE("FeatureGroup with one feature with one or two states is the exact same as zero FeatureGroups, boosting, regression") {
   TestApi testZeroFeaturesInGroup = TestApi(k_learningTypeRegression);
   testZeroFeaturesInGroup.AddFeatures({});
   testZeroFeaturesInGroup.AddFeatureGroups({ {} });
   testZeroFeaturesInGroup.AddTrainingSamples({ RegressionSample(10, {}) });
   testZeroFeaturesInGroup.AddValidationSamples({ RegressionSample(12, {}) });
   testZeroFeaturesInGroup.InitializeBoosting();

   TestApi testOneState = TestApi(k_learningTypeRegression);
   testOneState.AddFeatures({ FeatureTest(1) });
   testOneState.AddFeatureGroups({ { 0 } });
   testOneState.AddTrainingSamples({ RegressionSample(10, { 0 }) });
   testOneState.AddValidationSamples({ RegressionSample(12, { 0 }) });
   testOneState.InitializeBoosting();

   TestApi testTwoStates = TestApi(k_learningTypeRegression);
   testTwoStates.AddFeatures({ FeatureTest(2) });
   testTwoStates.AddFeatureGroups({ { 0 } });
   testTwoStates.AddTrainingSamples({ RegressionSample(10, { 1 }) });
   testTwoStates.AddValidationSamples({ RegressionSample(12, { 1 }) });
   testTwoStates.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroFeaturesInGroup.GetFeatureGroupsCount() == testOneState.GetFeatureGroupsCount());
      assert(testZeroFeaturesInGroup.GetFeatureGroupsCount() == testTwoStates.GetFeatureGroupsCount());
      for(size_t iFeatureGroup = 0; iFeatureGroup < testZeroFeaturesInGroup.GetFeatureGroupsCount(); ++iFeatureGroup) {
         FloatEbmType validationMetricZeroFeaturesInGroup = testZeroFeaturesInGroup.Boost(iFeatureGroup);
         FloatEbmType validationMetricOneState = testOneState.Boost(iFeatureGroup);
         CHECK_APPROX(validationMetricZeroFeaturesInGroup, validationMetricOneState);
         FloatEbmType validationMetricTwoStates = testTwoStates.Boost(iFeatureGroup);
         CHECK_APPROX(validationMetricZeroFeaturesInGroup, validationMetricTwoStates);

         FloatEbmType modelValueZeroFeaturesInGroup = testZeroFeaturesInGroup.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
         FloatEbmType modelValueOneState = testOneState.GetCurrentModelPredictorScore(iFeatureGroup, { 0 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInGroup, modelValueOneState);
         FloatEbmType modelValueTwoStates = testTwoStates.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInGroup, modelValueTwoStates);
      }
   }
}

TEST_CASE("FeatureGroup with one feature with one or two states is the exact same as zero FeatureGroups, boosting, binary") {
   TestApi testZeroFeaturesInGroup = TestApi(2, 0);
   testZeroFeaturesInGroup.AddFeatures({});
   testZeroFeaturesInGroup.AddFeatureGroups({ {} });
   testZeroFeaturesInGroup.AddTrainingSamples({ ClassificationSample(0, {}) });
   testZeroFeaturesInGroup.AddValidationSamples({ ClassificationSample(0, {}) });
   testZeroFeaturesInGroup.InitializeBoosting();

   TestApi testOneState = TestApi(2, 0);
   testOneState.AddFeatures({ FeatureTest(1) });
   testOneState.AddFeatureGroups({ { 0 } });
   testOneState.AddTrainingSamples({ ClassificationSample(0, { 0 }) });
   testOneState.AddValidationSamples({ ClassificationSample(0, { 0 }) });
   testOneState.InitializeBoosting();

   TestApi testTwoStates = TestApi(2, 0);
   testTwoStates.AddFeatures({ FeatureTest(2) });
   testTwoStates.AddFeatureGroups({ { 0 } });
   testTwoStates.AddTrainingSamples({ ClassificationSample(0, { 1 }) });
   testTwoStates.AddValidationSamples({ ClassificationSample(0, { 1 }) });
   testTwoStates.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroFeaturesInGroup.GetFeatureGroupsCount() == testOneState.GetFeatureGroupsCount());
      assert(testZeroFeaturesInGroup.GetFeatureGroupsCount() == testTwoStates.GetFeatureGroupsCount());
      for(size_t iFeatureGroup = 0; iFeatureGroup < testZeroFeaturesInGroup.GetFeatureGroupsCount(); ++iFeatureGroup) {
         FloatEbmType validationMetricZeroFeaturesInGroup = testZeroFeaturesInGroup.Boost(iFeatureGroup);
         FloatEbmType validationMetricOneState = testOneState.Boost(iFeatureGroup);
         CHECK_APPROX(validationMetricZeroFeaturesInGroup, validationMetricOneState);
         FloatEbmType validationMetricTwoStates = testTwoStates.Boost(iFeatureGroup);
         CHECK_APPROX(validationMetricZeroFeaturesInGroup, validationMetricTwoStates);

         FloatEbmType modelValueZeroFeaturesInGroup0 = testZeroFeaturesInGroup.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
         FloatEbmType modelValueOneState0 = testOneState.GetCurrentModelPredictorScore(iFeatureGroup, { 0 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInGroup0, modelValueOneState0);
         FloatEbmType modelValueTwoStates0 = testTwoStates.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInGroup0, modelValueTwoStates0);

         FloatEbmType modelValueZeroFeaturesInGroup1 = testZeroFeaturesInGroup.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1);
         FloatEbmType modelValueOneState1 = testOneState.GetCurrentModelPredictorScore(iFeatureGroup, { 0 }, 1);
         CHECK_APPROX(modelValueZeroFeaturesInGroup1, modelValueOneState1);
         FloatEbmType modelValueTwoStates1 = testTwoStates.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 1);
         CHECK_APPROX(modelValueZeroFeaturesInGroup1, modelValueTwoStates1);
      }
   }
}

TEST_CASE("FeatureGroup with one feature with one or two states is the exact same as zero FeatureGroups, boosting, multiclass") {
   TestApi testZeroFeaturesInGroup = TestApi(3);
   testZeroFeaturesInGroup.AddFeatures({});
   testZeroFeaturesInGroup.AddFeatureGroups({ {} });
   testZeroFeaturesInGroup.AddTrainingSamples({ ClassificationSample(0, {}) });
   testZeroFeaturesInGroup.AddValidationSamples({ ClassificationSample(0, {}) });
   testZeroFeaturesInGroup.InitializeBoosting();

   TestApi testOneState = TestApi(3);
   testOneState.AddFeatures({ FeatureTest(1) });
   testOneState.AddFeatureGroups({ { 0 } });
   testOneState.AddTrainingSamples({ ClassificationSample(0, { 0 }) });
   testOneState.AddValidationSamples({ ClassificationSample(0, { 0 }) });
   testOneState.InitializeBoosting();

   TestApi testTwoStates = TestApi(3);
   testTwoStates.AddFeatures({ FeatureTest(2) });
   testTwoStates.AddFeatureGroups({ { 0 } });
   testTwoStates.AddTrainingSamples({ ClassificationSample(0, { 1 }) });
   testTwoStates.AddValidationSamples({ ClassificationSample(0, { 1 }) });
   testTwoStates.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroFeaturesInGroup.GetFeatureGroupsCount() == testOneState.GetFeatureGroupsCount());
      assert(testZeroFeaturesInGroup.GetFeatureGroupsCount() == testTwoStates.GetFeatureGroupsCount());
      for(size_t iFeatureGroup = 0; iFeatureGroup < testZeroFeaturesInGroup.GetFeatureGroupsCount(); ++iFeatureGroup) {
         FloatEbmType validationMetricZeroFeaturesInGroup = testZeroFeaturesInGroup.Boost(iFeatureGroup);
         FloatEbmType validationMetricOneState = testOneState.Boost(iFeatureGroup);
         CHECK_APPROX(validationMetricZeroFeaturesInGroup, validationMetricOneState);
         FloatEbmType validationMetricTwoStates = testTwoStates.Boost(iFeatureGroup);
         CHECK_APPROX(validationMetricZeroFeaturesInGroup, validationMetricTwoStates);

         FloatEbmType modelValueZeroFeaturesInGroup0 = testZeroFeaturesInGroup.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
         FloatEbmType modelValueOneState0 = testOneState.GetCurrentModelPredictorScore(iFeatureGroup, { 0 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInGroup0, modelValueOneState0);
         FloatEbmType modelValueTwoStates0 = testTwoStates.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 0);
         CHECK_APPROX(modelValueZeroFeaturesInGroup0, modelValueTwoStates0);

         FloatEbmType modelValueZeroFeaturesInGroup1 = testZeroFeaturesInGroup.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1);
         FloatEbmType modelValueOneState1 = testOneState.GetCurrentModelPredictorScore(iFeatureGroup, { 0 }, 1);
         CHECK_APPROX(modelValueZeroFeaturesInGroup1, modelValueOneState1);
         FloatEbmType modelValueTwoStates1 = testTwoStates.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 1);
         CHECK_APPROX(modelValueZeroFeaturesInGroup1, modelValueTwoStates1);

         FloatEbmType modelValueZeroFeaturesInGroup2 = testZeroFeaturesInGroup.GetCurrentModelPredictorScore(iFeatureGroup, {}, 2);
         FloatEbmType modelValueOneState2 = testOneState.GetCurrentModelPredictorScore(iFeatureGroup, { 0 }, 2);
         CHECK_APPROX(modelValueZeroFeaturesInGroup2, modelValueOneState2);
         FloatEbmType modelValueTwoStates2 = testTwoStates.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 2);
         CHECK_APPROX(modelValueZeroFeaturesInGroup2, modelValueTwoStates2);
      }
   }
}

TEST_CASE("3 dimensional featureGroup with one dimension reduced in different ways, boosting, regression") {
   TestApi test0 = TestApi(k_learningTypeRegression);
   test0.AddFeatures({ FeatureTest(1), FeatureTest(2), FeatureTest(2) });
   test0.AddFeatureGroups({ { 0, 1, 2 } });
   test0.AddTrainingSamples({
      RegressionSample(9, { 0, 0, 0 }),
      RegressionSample(10, { 0, 1, 0 }),
      RegressionSample(11, { 0, 0, 1 }),
      RegressionSample(12, { 0, 1, 1 }),
      });
   test0.AddValidationSamples({ RegressionSample(12, { 0, 1, 0 }) });
   test0.InitializeBoosting();

   TestApi test1 = TestApi(k_learningTypeRegression);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(1), FeatureTest(2) });
   test1.AddFeatureGroups({ { 0, 1, 2 } });
   test1.AddTrainingSamples({
      RegressionSample(9, { 0, 0, 0 }),
      RegressionSample(10, { 0, 0, 1 }),
      RegressionSample(11, { 1, 0, 0 }),
      RegressionSample(12, { 1, 0, 1 }),
      });
   test1.AddValidationSamples({ RegressionSample(12, { 0, 0, 1 }) });
   test1.InitializeBoosting();

   TestApi test2 = TestApi(k_learningTypeRegression);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2), FeatureTest(1) });
   test2.AddFeatureGroups({ { 0, 1, 2 } });
   test2.AddTrainingSamples({
      RegressionSample(9, { 0, 0, 0 }),
      RegressionSample(10, { 1, 0, 0 }),
      RegressionSample(11, { 0, 1, 0 }),
      RegressionSample(12, { 1, 1, 0 }),
      });
   test2.AddValidationSamples({ RegressionSample(12, { 1, 0, 0 }) });
   test2.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(test0.GetFeatureGroupsCount() == test1.GetFeatureGroupsCount());
      assert(test0.GetFeatureGroupsCount() == test2.GetFeatureGroupsCount());
      for(size_t iFeatureGroup = 0; iFeatureGroup < test0.GetFeatureGroupsCount(); ++iFeatureGroup) {
         FloatEbmType validationMetric0 = test0.Boost(iFeatureGroup);
         FloatEbmType validationMetric1 = test1.Boost(iFeatureGroup);
         CHECK_APPROX(validationMetric0, validationMetric1);
         FloatEbmType validationMetric2 = test2.Boost(iFeatureGroup);
         CHECK_APPROX(validationMetric0, validationMetric2);

         FloatEbmType modelValue01 = test0.GetCurrentModelPredictorScore(iFeatureGroup, { 0, 0, 0 }, 0);
         FloatEbmType modelValue02 = test0.GetCurrentModelPredictorScore(iFeatureGroup, { 0, 0, 1 }, 0);
         FloatEbmType modelValue03 = test0.GetCurrentModelPredictorScore(iFeatureGroup, { 0, 1, 0 }, 0);
         FloatEbmType modelValue04 = test0.GetCurrentModelPredictorScore(iFeatureGroup, { 0, 1, 1 }, 0);

         FloatEbmType modelValue11 = test1.GetCurrentModelPredictorScore(iFeatureGroup, { 0, 0, 0 }, 0);
         FloatEbmType modelValue12 = test1.GetCurrentModelPredictorScore(iFeatureGroup, { 1, 0, 0 }, 0);
         FloatEbmType modelValue13 = test1.GetCurrentModelPredictorScore(iFeatureGroup, { 0, 0, 1 }, 0);
         FloatEbmType modelValue14 = test1.GetCurrentModelPredictorScore(iFeatureGroup, { 1, 0, 1 }, 0);
         CHECK_APPROX(modelValue11, modelValue01);
         CHECK_APPROX(modelValue12, modelValue02);
         CHECK_APPROX(modelValue13, modelValue03);
         CHECK_APPROX(modelValue14, modelValue04);

         FloatEbmType modelValue21 = test2.GetCurrentModelPredictorScore(iFeatureGroup, { 0, 0, 0 }, 0);
         FloatEbmType modelValue22 = test2.GetCurrentModelPredictorScore(iFeatureGroup, { 0, 1, 0 }, 0);
         FloatEbmType modelValue23 = test2.GetCurrentModelPredictorScore(iFeatureGroup, { 1, 0, 0 }, 0);
         FloatEbmType modelValue24 = test2.GetCurrentModelPredictorScore(iFeatureGroup, { 1, 1, 0 }, 0);
         CHECK_APPROX(modelValue21, modelValue01);
         CHECK_APPROX(modelValue22, modelValue02);
         CHECK_APPROX(modelValue23, modelValue03);
         CHECK_APPROX(modelValue24, modelValue04);
      }
   }
}


TEST_CASE("Random splitting with 3 features, boosting, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(4) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples({
      ClassificationSample(0, { 0 }),
      ClassificationSample(1, { 1 }),
      ClassificationSample(1, { 2 }),
      ClassificationSample(2, { 3 })
   });
   test.AddValidationSamples({ ClassificationSample(0, { 1 }) });
   test.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         FloatEbmType validationMetricTwoStates = test.Boost(iFeatureGroup, {}, {}, GenerateUpdateOptions_RandomSplits, k_learningRateDefault, 2);
         if(0 == iEpoch) {
            CHECK_APPROX(validationMetricTwoStates, 1.0340957641601563f);

            FloatEbmType modelValueTwoStates0 = test.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 0);
            CHECK_APPROX(modelValueTwoStates0, 0.0075f);

            FloatEbmType modelValueTwoStates1 = test.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 1);
            CHECK_APPROX(modelValueTwoStates1, 0.0075f);

            FloatEbmType modelValueTwoStates2 = test.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 2);
            CHECK_APPROX(modelValueTwoStates2, -0.015f);
         }
      }
   }
}

TEST_CASE("Random splitting with 3 features, boosting, multiclass, sums") {
   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(4) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples({
      ClassificationSample(0, { 0 }),
      ClassificationSample(1, { 1 }),
      ClassificationSample(1, { 2 }),
      ClassificationSample(2, { 3 })
      });
   test.AddValidationSamples({ ClassificationSample(0, { 1 }) });
   test.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         FloatEbmType validationMetricTwoStates = test.Boost(iFeatureGroup, {}, {}, GenerateUpdateOptions_RandomSplits | GenerateUpdateOptions_Sums, k_learningRateDefault, 2);
         if(0 == iEpoch) {
            CHECK_APPROX(validationMetricTwoStates, 1.0372848510742188);

            FloatEbmType modelValueTwoStates0 = test.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 0);
            CHECK_APPROX(modelValueTwoStates0, 0.0033333333333333344f);

            FloatEbmType modelValueTwoStates1 = test.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 1);
            CHECK_APPROX(modelValueTwoStates1, 0.0033333333333333344f);

            FloatEbmType modelValueTwoStates2 = test.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 2);
            CHECK_APPROX(modelValueTwoStates2, -0.0066666666666666662f);
         }
      }
   }
}
