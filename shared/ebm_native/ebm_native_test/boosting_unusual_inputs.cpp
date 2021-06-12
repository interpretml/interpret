// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"

static const TestPriority k_filePriority = TestPriority::BoostingUnusualInputs;

TEST_CASE("null validationMetricOut, boosting, regression") {
   IntEbmType featureGroupsDimensionCount[1];
   featureGroupsDimensionCount[0] = 0;

   const BoosterHandle boosterHandle = CreateRegressionBooster(
      k_randomSeed,
      0,
      nullptr,
      nullptr,
      1,
      featureGroupsDimensionCount,
      nullptr,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      0,
      nullptr
   );
   const IntEbmType retGenerate = GenerateModelUpdate(
      boosterHandle,
      IntEbmType { 0 },
      GenerateUpdateOptions_Default,
      k_learningRateDefault,
      k_countSamplesRequiredForChildSplitMinDefault,
      &k_leavesMaxDefault[0],
      nullptr
   );
   CHECK(0 == retGenerate);
   const IntEbmType ret = ApplyModelUpdate(
      boosterHandle,
      nullptr
   );
   CHECK(0 == ret);
   FreeBooster(boosterHandle);
}

TEST_CASE("null validationMetricOut, boosting, binary") {
   IntEbmType featureGroupsDimensionCount[1];
   featureGroupsDimensionCount[0] = 0;

   const BoosterHandle boosterHandle = CreateClassificationBooster(
      k_randomSeed,
      2,
      0,
      nullptr,
      nullptr,
      1,
      featureGroupsDimensionCount,
      nullptr,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      0,
      nullptr
   );
   const IntEbmType retGenerate = GenerateModelUpdate(
      boosterHandle,
      IntEbmType { 0 },
      GenerateUpdateOptions_Default,
      k_learningRateDefault,
      k_countSamplesRequiredForChildSplitMinDefault,
      &k_leavesMaxDefault[0],
      nullptr
   );
   CHECK(0 == retGenerate);
   const IntEbmType ret = ApplyModelUpdate(
      boosterHandle,
      nullptr
   );
   CHECK(0 == ret);
   FreeBooster(boosterHandle);
}

TEST_CASE("null validationMetricOut, boosting, multiclass") {
   IntEbmType featureGroupsDimensionCount[1];
   featureGroupsDimensionCount[0] = 0;

   const BoosterHandle boosterHandle = CreateClassificationBooster(
      k_randomSeed,
      3,
      0,
      nullptr,
      nullptr,
      1,
      featureGroupsDimensionCount,
      nullptr,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      0,
      nullptr
   );
   const IntEbmType retGenerate = GenerateModelUpdate(
      boosterHandle,
      IntEbmType { 0 },
      GenerateUpdateOptions_Default,
      k_learningRateDefault,
      k_countSamplesRequiredForChildSplitMinDefault,
      &k_leavesMaxDefault[0],
      nullptr
   );
   CHECK(0 == retGenerate);
   const IntEbmType ret = ApplyModelUpdate(
      boosterHandle,
      nullptr
   );
   CHECK(0 == ret);
   FreeBooster(boosterHandle);
}

TEST_CASE("zero learning rate, boosting, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddFeatureGroups({ {} });
   test.AddTrainingSamples({ TestSample({}, 10) });
   test.AddValidationSamples({ TestSample({}, 12) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup, GenerateUpdateOptions_Default, 0);
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
   test.AddTrainingSamples({ TestSample({}, 0) });
   test.AddValidationSamples({ TestSample({}, 0) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup, GenerateUpdateOptions_Default, 0);
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
   test.AddTrainingSamples({ TestSample({}, 0) });
   test.AddValidationSamples({ TestSample({}, 0) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup, GenerateUpdateOptions_Default, 0);
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
   test.AddTrainingSamples({ TestSample({}, 10) });
   test.AddValidationSamples({ TestSample({}, 12) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup, GenerateUpdateOptions_Default, -k_learningRateDefault);
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
   test.AddTrainingSamples({ TestSample({}, 0) });
   test.AddValidationSamples({ TestSample({}, 0) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 50; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup, GenerateUpdateOptions_Default, -k_learningRateDefault);
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
   test.AddTrainingSamples({ TestSample({}, 0) });
   test.AddValidationSamples({ TestSample({}, 0) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 20; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup, GenerateUpdateOptions_Default, -k_learningRateDefault);
         if(0 == iFeatureGroup && 0 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 1.1288361512023379, double { 1e-1 });
            const FloatEbmType zeroLogit = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1) - zeroLogit;
            CHECK_APPROX(modelValue, 0.04500000000000000);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 2) - zeroLogit;
            CHECK_APPROX(modelValue, 0.04500000000000000);
         }
         if(0 == iFeatureGroup && 1 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 1.1602122411839852, double { 1e-1 });
            const FloatEbmType zeroLogit = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1) - zeroLogit;
            CHECK_APPROX_TOLERANCE(modelValue, 0.091033038217642897, double { 1e-2 });
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 2) - zeroLogit;
            CHECK_APPROX_TOLERANCE(modelValue, 0.091033038217642897, double { 1e-2 });
         }
      }
   }
   CHECK_APPROX_TOLERANCE(validationMetric, 2.0611718475324357, double { 1e-1 });
   const FloatEbmType zeroLogit1 = test.GetCurrentModelPredictorScore(0, {}, 0);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 1) - zeroLogit1;
   CHECK_APPROX_TOLERANCE(modelValue, 1.23185585569831419, double { 1e-1 });
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 2) - zeroLogit1;
   CHECK_APPROX_TOLERANCE(modelValue, 1.23185585569831419, double { 1e-1 });
}

TEST_CASE("zero countSamplesRequiredForChildSplitMin, boosting, regression") {
   // TODO : call test.Boost many more times in a loop, and verify the output remains the same as previous runs
   // TODO : add classification binary and multiclass versions of this

   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples({
      TestSample({ 0 }, 10),
      TestSample({ 1 }, 10),
      });
   test.AddValidationSamples({ TestSample({ 1 }, 12) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = test.Boost(0, GenerateUpdateOptions_Default, k_learningRateDefault, 0);
   CHECK_APPROX(validationMetric, 141.61);
   FloatEbmType modelValue;
   modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
   CHECK_APPROX(modelValue, 0.1000000000000000);
   CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 0));
}

TEST_CASE("weights are proportional, boosting, regression") {
   TestApi test1 = TestApi(k_learningTypeRegression);
   test1.AddFeatures({ FeatureTest(2) });
   test1.AddFeatureGroups({ { 0 } });
   test1.AddTrainingSamples({
      TestSample({ 0 }, 10, std::nextafter(0.3, 100)),
      TestSample({ 1 }, 10, 0.3),
      });
   test1.AddValidationSamples({ 
      TestSample({ 1 }, 12, std::nextafter(0.3, 100)),
      TestSample({ 1 }, 12, 0.3)
      });
   test1.InitializeBoosting();
   FloatEbmType validationMetric1 = test1.Boost(0);
   FloatEbmType modelValue1;
   modelValue1 = test1.GetCurrentModelPredictorScore(0, { 0 }, 0);
   CHECK_APPROX(modelValue1, test1.GetCurrentModelPredictorScore(0, { 1 }, 0));


   TestApi test2 = TestApi(k_learningTypeRegression);
   test2.AddFeatures({ FeatureTest(2) });
   test2.AddFeatureGroups({ { 0 } });
   test2.AddTrainingSamples({
      TestSample({ 0 }, 10, std::nextafter(2, 100)),
      TestSample({ 1 }, 10, 2),
      });
   test2.AddValidationSamples({ 
      TestSample({ 1 }, 12, std::nextafter(2, 100)),
      TestSample({ 1 }, 12, 2)
      });
   test2.InitializeBoosting();
   FloatEbmType validationMetric2 = test2.Boost(0);
   FloatEbmType modelValue2;
   modelValue2 = test2.GetCurrentModelPredictorScore(0, { 0 }, 0);
   CHECK_APPROX(modelValue2, test2.GetCurrentModelPredictorScore(0, { 1 }, 0));


   TestApi test3 = TestApi(k_learningTypeRegression);
   test3.AddFeatures({ FeatureTest(2) });
   test3.AddFeatureGroups({ { 0 } });
   test3.AddTrainingSamples({
      TestSample({ 0 }, 10, 0),
      TestSample({ 1 }, 10, 0),
      });
   test3.AddValidationSamples({ 
      TestSample({ 1 }, 12, 0),
      TestSample({ 1 }, 12, 0)
      });
   test3.InitializeBoosting();
   FloatEbmType validationMetric3 = test3.Boost(0);
   FloatEbmType modelValue3;
   modelValue3 = test3.GetCurrentModelPredictorScore(0, { 0 }, 0);
   CHECK_APPROX(modelValue3, test3.GetCurrentModelPredictorScore(0, { 1 }, 0));


   CHECK_APPROX(validationMetric1, validationMetric2);
   CHECK_APPROX(validationMetric1, validationMetric3);
   CHECK_APPROX(modelValue1, modelValue2);
   CHECK_APPROX(modelValue1, modelValue3);
}

TEST_CASE("weights are proportional, boosting, binary") {
   TestApi test1 = TestApi(2);
   test1.AddFeatures({ FeatureTest(2) });
   test1.AddFeatureGroups({ { 0 } });
   test1.AddTrainingSamples({
      TestSample({ 0 }, 0, std::nextafter(0.3, 100)),
      TestSample({ 1 }, 0, 0.3),
      });
   test1.AddValidationSamples({ 
      TestSample({ 1 }, 0, std::nextafter(0.3, 100)),
      TestSample({ 1 }, 0, 0.3)
      });
   test1.InitializeBoosting();
   FloatEbmType validationMetric1 = test1.Boost(0);
   FloatEbmType modelValue1;
   modelValue1 = test1.GetCurrentModelPredictorScore(0, { 0 }, 0);
   CHECK_APPROX(modelValue1, test1.GetCurrentModelPredictorScore(0, { 1 }, 0));


   TestApi test2 = TestApi(2);
   test2.AddFeatures({ FeatureTest(2) });
   test2.AddFeatureGroups({ { 0 } });
   test2.AddTrainingSamples({
      TestSample({ 0 }, 1, std::nextafter(2, 100)),
      TestSample({ 1 }, 1, 2),
      });
   test2.AddValidationSamples({ 
      TestSample({ 1 }, 1, std::nextafter(2, 100)),
      TestSample({ 1 }, 1, 2)
      });
   test2.InitializeBoosting();
   FloatEbmType validationMetric2 = test2.Boost(0);
   FloatEbmType modelValue2;
   modelValue2 = test2.GetCurrentModelPredictorScore(0, { 0 }, 0);
   CHECK_APPROX(modelValue2, test2.GetCurrentModelPredictorScore(0, { 1 }, 0));


   TestApi test3 = TestApi(2);
   test3.AddFeatures({ FeatureTest(2) });
   test3.AddFeatureGroups({ { 0 } });
   test3.AddTrainingSamples({
      TestSample({ 0 }, 0, 0),
      TestSample({ 1 }, 0, 0),
      });
   test3.AddValidationSamples({ 
      TestSample({ 1 }, 0, 0),
      TestSample({ 1 }, 0, 0)
      });
   test3.InitializeBoosting();
   FloatEbmType validationMetric3 = test3.Boost(0);
   FloatEbmType modelValue3;
   modelValue3 = test3.GetCurrentModelPredictorScore(0, { 0 }, 0);
   CHECK_APPROX(modelValue3, test3.GetCurrentModelPredictorScore(0, { 1 }, 0));


   CHECK_APPROX(validationMetric1, validationMetric2);
   CHECK_APPROX(validationMetric1, validationMetric3);
   CHECK_APPROX(modelValue1, modelValue2);
   CHECK_APPROX(modelValue1, modelValue3);
}

TEST_CASE("weights are proportional, boosting, multiclass") {
   TestApi test1 = TestApi(3);
   test1.AddFeatures({ FeatureTest(2) });
   test1.AddFeatureGroups({ { 0 } });
   test1.AddTrainingSamples({
      TestSample({ 0 }, 0, std::nextafter(0.3, 100)),
      TestSample({ 1 }, 0, 0.3),
      });
   test1.AddValidationSamples({ 
      TestSample({ 1 }, 0, std::nextafter(0.3, 100)),
      TestSample({ 1 }, 0, 0.3)
      });
   test1.InitializeBoosting();
   FloatEbmType validationMetric1 = test1.Boost(0);
   FloatEbmType modelValue1;
   modelValue1 = test1.GetCurrentModelPredictorScore(0, { 0 }, 0);
   CHECK_APPROX(modelValue1, test1.GetCurrentModelPredictorScore(0, { 1 }, 0));


   TestApi test2 = TestApi(3);
   test2.AddFeatures({ FeatureTest(2) });
   test2.AddFeatureGroups({ { 0 } });
   test2.AddTrainingSamples({
      TestSample({ 0 }, 1, std::nextafter(2, 100)),
      TestSample({ 1 }, 1, 2),
      });
   test2.AddValidationSamples({ 
      TestSample({ 1 }, 1, std::nextafter(2, 100)),
      TestSample({ 1 }, 1, 2)
      });
   test2.InitializeBoosting();
   FloatEbmType validationMetric2 = test2.Boost(0);
   FloatEbmType modelValue2;
   modelValue2 = test2.GetCurrentModelPredictorScore(0, { 0 }, 1);
   CHECK_APPROX(modelValue2, test2.GetCurrentModelPredictorScore(0, { 1 }, 1));


   TestApi test3 = TestApi(3);
   test3.AddFeatures({ FeatureTest(2) });
   test3.AddFeatureGroups({ { 0 } });
   test3.AddTrainingSamples({
      TestSample({ 0 }, 2, 0),
      TestSample({ 1 }, 2, 0),
      });
   test3.AddValidationSamples({ 
      TestSample({ 1 }, 2, 0),
      TestSample({ 1 }, 2, 0)
      });
   test3.InitializeBoosting();
   FloatEbmType validationMetric3 = test3.Boost(0);
   FloatEbmType modelValue3;
   modelValue3 = test3.GetCurrentModelPredictorScore(0, { 0 }, 2);
   CHECK_APPROX(modelValue3, test3.GetCurrentModelPredictorScore(0, { 1 }, 2));


   CHECK_APPROX(validationMetric1, validationMetric2);
   CHECK_APPROX(validationMetric1, validationMetric3);
   CHECK_APPROX(modelValue1, modelValue2);
   CHECK_APPROX(modelValue1, modelValue3);
}

TEST_CASE("weights totals equivalence, boosting, regression") {
   TestApi test1 = TestApi(k_learningTypeRegression);
   test1.AddFeatures({ FeatureTest(2) });
   test1.AddFeatureGroups({ { 0 } });
   test1.AddTrainingSamples({
      TestSample({ 0 }, 10, 0.15),
      TestSample({ 0 }, 10, 0.15),
      TestSample({ 1 }, 12, 1.2),
      });
   test1.AddValidationSamples({ 
      TestSample({ 0 }, 10, 0.6),
      TestSample({ 1 }, 12, 0.3)
      });
   test1.InitializeBoosting();
   FloatEbmType validationMetric1 = test1.Boost(0);
   FloatEbmType modelValue1;
   modelValue1 = test1.GetCurrentModelPredictorScore(0, { 0 }, 0);


   TestApi test2 = TestApi(k_learningTypeRegression);
   test2.AddFeatures({ FeatureTest(2) });
   test2.AddFeatureGroups({ { 0 } });
   test2.AddTrainingSamples({
      TestSample({ 0 }, 10, 0.5),
      TestSample({ 1 }, 12, 2),
      });
   test2.AddValidationSamples({ 
      TestSample({ 0 }, 10, 1),
      TestSample({ 0 }, 10, 1),
      TestSample({ 1 }, 12, 1)
      });
   test2.InitializeBoosting();
   FloatEbmType validationMetric2 = test2.Boost(0);
   FloatEbmType modelValue2;
   modelValue2 = test2.GetCurrentModelPredictorScore(0, { 0 }, 0);

   CHECK_APPROX(validationMetric1, validationMetric2);
   CHECK_APPROX(modelValue1, modelValue2);
}

TEST_CASE("weights totals equivalence, boosting, binary") {
   TestApi test1 = TestApi(2);
   test1.AddFeatures({ FeatureTest(2) });
   test1.AddFeatureGroups({ { 0 } });
   test1.AddTrainingSamples({
      TestSample({ 0 }, 0, 0.15),
      TestSample({ 0 }, 0, 0.15),
      TestSample({ 1 }, 1, 1.2),
      });
   test1.AddValidationSamples({ 
      TestSample({ 0 }, 0, 0.6),
      TestSample({ 1 }, 1, 0.3)
      });
   test1.InitializeBoosting();
   FloatEbmType validationMetric1 = test1.Boost(0);
   FloatEbmType modelValue1;
   modelValue1 = test1.GetCurrentModelPredictorScore(0, { 0 }, 1);


   TestApi test2 = TestApi(2);
   test2.AddFeatures({ FeatureTest(2) });
   test2.AddFeatureGroups({ { 0 } });
   test2.AddTrainingSamples({
      TestSample({ 0 }, 0, 0.5),
      TestSample({ 1 }, 1, 2),
      });
   test2.AddValidationSamples({ 
      TestSample({ 0 }, 0, 1),
      TestSample({ 0 }, 0, 1),
      TestSample({ 1 }, 1, 1)
      });
   test2.InitializeBoosting();
   FloatEbmType validationMetric2 = test2.Boost(0);
   FloatEbmType modelValue2;
   modelValue2 = test2.GetCurrentModelPredictorScore(0, { 0 }, 1);


   CHECK_APPROX(validationMetric1, validationMetric2);
   CHECK_APPROX(modelValue1, modelValue2);
}

TEST_CASE("weights totals equivalence, boosting, multiclass") {
   TestApi test1 = TestApi(3);
   test1.AddFeatures({ FeatureTest(2) });
   test1.AddFeatureGroups({ { 0 } });
   test1.AddTrainingSamples({
      TestSample({ 0 }, 0, 0.15),
      TestSample({ 0 }, 0, 0.15),
      TestSample({ 1 }, 2, 1.2),
      });
   test1.AddValidationSamples({ 
      TestSample({ 0 }, 0, 0.6),
      TestSample({ 1 }, 2, 0.3)
      });
   test1.InitializeBoosting();
   FloatEbmType validationMetric1 = test1.Boost(0);
   FloatEbmType modelValue1;
   modelValue1 = test1.GetCurrentModelPredictorScore(0, { 0 }, 1);


   TestApi test2 = TestApi(3);
   test2.AddFeatures({ FeatureTest(2) });
   test2.AddFeatureGroups({ { 0 } });
   test2.AddTrainingSamples({
      TestSample({ 0 }, 0, 0.5),
      TestSample({ 1 }, 2, 2),
      });
   test2.AddValidationSamples({
      TestSample({ 0 }, 0, 1),
      TestSample({ 0 }, 0, 1),
      TestSample({ 1 }, 2, 1)
      });
   test2.InitializeBoosting();
   FloatEbmType validationMetric2 = test2.Boost(0);
   FloatEbmType modelValue2;
   modelValue2 = test2.GetCurrentModelPredictorScore(0, { 0 }, 1);


   CHECK_APPROX(validationMetric1, validationMetric2);
   CHECK_APPROX(modelValue1, modelValue2);
}

TEST_CASE("one leavesMax, boosting, regression") {
   // TODO : add classification binary and multiclass versions of this

   static const std::vector<IntEbmType> k_leavesMax = {
      IntEbmType { 1 }
   };

   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples({
      TestSample({ 0 }, 10),
      TestSample({ 1 }, 10),
      });
   test.AddValidationSamples({ TestSample({ 1 }, 12) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = test.Boost(0, GenerateUpdateOptions_Default, k_learningRateDefault, k_countSamplesRequiredForChildSplitMinDefault, k_leavesMax);
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
   test.AddTrainingSamples({});
   test.AddValidationSamples({ TestSample({ 1 }, 12) });
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
   test.AddTrainingSamples({});
   test.AddValidationSamples({ TestSample({ 1 }, 0) });
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
   test.AddTrainingSamples({});
   test.AddValidationSamples({ TestSample({ 1 }, 0) });
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
   test.AddTrainingSamples({ TestSample({ 1 }, 10) });
   test.AddValidationSamples({});
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
   test.AddTrainingSamples({ TestSample({ 1 }, 0) });
   test.AddValidationSamples({});
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
   test.AddTrainingSamples({ TestSample({ 1 }, 0) });
   test.AddValidationSamples({});
   test.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      FloatEbmType validationMetric = test.Boost(0);
      CHECK(0 == validationMetric);
      // the current model will continue to update, even though we have no way of evaluating it
      FloatEbmType modelValue;
      if(0 == iEpoch) {
         const FloatEbmType zeroLogit = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
         CHECK_APPROX(zeroLogit, test.GetCurrentModelPredictorScore(0, { 1 }, 0));
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 1) - zeroLogit;
         CHECK_APPROX(modelValue, -0.04500000000000000);
         CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 1) - zeroLogit);
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 2) - zeroLogit;
         CHECK_APPROX(modelValue, -0.04500000000000000);
         CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 2) - zeroLogit);
      }
      if(1 == iEpoch) {
         const FloatEbmType zeroLogit = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
         CHECK_APPROX(zeroLogit, test.GetCurrentModelPredictorScore(0, { 1 }, 0));
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 1) - zeroLogit;
         CHECK_APPROX_TOLERANCE(modelValue, -0.089007468617193456, double { 1e-2 });
         CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 1) - zeroLogit);
         modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 2) - zeroLogit;
         CHECK_APPROX_TOLERANCE(modelValue, -0.089007468617193456, double { 1e-2 });
         CHECK_APPROX(modelValue, test.GetCurrentModelPredictorScore(0, { 1 }, 2) - zeroLogit);
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
   test.AddTrainingSamples({});
   test.AddValidationSamples({});
   test.InitializeBoosting();

   FloatEbmType validationMetric = test.Boost(0);
   CHECK(0 == validationMetric);

   FloatEbmType model[1];

   // we're not sure what we'd get back since we aren't allowed to access it, so don't do anything with the return value.  We just want to make sure 
   // calling to get the models doesn't crash
   model[0] = 9.99;
   test.GetBestModelFeatureGroupRaw(0, model);
   CHECK(9.99 == model[0]); // the model is a tensor with zero values since one of the dimensions is non-existant
   model[0] = 9.99;
   test.GetCurrentModelFeatureGroupRaw(0, model);
   CHECK(9.99 == model[0]); // the model is a tensor with zero values since one of the dimensions is non-existant
}

TEST_CASE("classification with 0 possible target states, boosting") {
   // for there to be zero states, there can't be an training data or testing data since then those would be required to have a value for the state

   IntEbmType featureGroupsDimensionCount[1];
   featureGroupsDimensionCount[0] = 0;

   const BoosterHandle boosterHandle = CreateClassificationBooster(
      k_randomSeed,
      0,
      0,
      nullptr,
      nullptr,
      1,
      featureGroupsDimensionCount,
      nullptr,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      0,
      nullptr
   );

   CHECK(0 == GetBestModelFeatureGroup(boosterHandle, 0, nullptr));
   CHECK(0 == GetCurrentModelFeatureGroup(boosterHandle, 0, nullptr));

   FloatEbmType gain = 9.99;
   const IntEbmType retGenerate = GenerateModelUpdate(
      boosterHandle,
      IntEbmType { 0 },
      GenerateUpdateOptions_Default,
      k_learningRateDefault,
      k_countSamplesRequiredForChildSplitMinDefault,
      &k_leavesMaxDefault[0],
      &gain
   );
   CHECK(0 == retGenerate);
   CHECK(0 == gain);

   IntEbmType countCuts = 0;
   IntEbmType cutIndexes[1];
   IntEbmType retCuts = GetModelUpdateCuts(
      boosterHandle,
      0,
      &countCuts,
      cutIndexes
   );
   CHECK(1 == retCuts); // we have no dimensions, so 0 is invalid
   CHECK(0 == countCuts);
   
   IntEbmType retGetModel = GetModelUpdateExpanded(boosterHandle, nullptr);
   CHECK(0 == retGetModel);

   IntEbmType retSetModel = SetModelUpdateExpanded(boosterHandle, 0, nullptr);
   CHECK(0 == retSetModel);

   FloatEbmType metric = 9.99;
   const IntEbmType retApply = ApplyModelUpdate(
      boosterHandle,
      &metric
   );
   CHECK(0 == retApply);
   CHECK(0 == metric);

   CHECK(0 == GetBestModelFeatureGroup(boosterHandle, 0, nullptr));
   CHECK(0 == GetCurrentModelFeatureGroup(boosterHandle, 0, nullptr));

   FreeBooster(boosterHandle);
}

TEST_CASE("classification with 1 possible target, boosting") {
   BoolEbmType featuresCategorical[1];
   featuresCategorical[0] = EBM_FALSE;

   IntEbmType featuresBinCount[1];
   featuresBinCount[0] = 2;

   IntEbmType featureGroupsDimensionCount[1];
   featureGroupsDimensionCount[0] = 1;

   IntEbmType featureGroupsFeatureIndexes[1];
   featureGroupsFeatureIndexes[0] = 0;

   const BoosterHandle boosterHandle = CreateClassificationBooster(
      k_randomSeed,
      1,
      1,
      featuresCategorical,
      featuresBinCount,
      1,
      featureGroupsDimensionCount,
      featureGroupsFeatureIndexes,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      0,
      nullptr
   );

   CHECK(0 == GetBestModelFeatureGroup(boosterHandle, 0, nullptr));
   CHECK(0 == GetCurrentModelFeatureGroup(boosterHandle, 0, nullptr));

   FloatEbmType gain = 9.99;
   const IntEbmType retGenerate = GenerateModelUpdate(
      boosterHandle,
      IntEbmType { 0 },
      GenerateUpdateOptions_Default,
      k_learningRateDefault,
      k_countSamplesRequiredForChildSplitMinDefault,
      &k_leavesMaxDefault[0],
      &gain
   );
   CHECK(0 == retGenerate);
   CHECK(0 == gain);

   IntEbmType countCuts = 1;
   IntEbmType cutIndexes[1];
   IntEbmType retCuts = GetModelUpdateCuts(
      boosterHandle,
      0,
      &countCuts,
      cutIndexes
   );
   CHECK(0 == retCuts);
   CHECK(0 == countCuts);

   IntEbmType retGetModel = GetModelUpdateExpanded(boosterHandle, nullptr);
   CHECK(0 == retGetModel);

   IntEbmType retSetModel = SetModelUpdateExpanded(boosterHandle, 0, nullptr);
   CHECK(0 == retSetModel);

   FloatEbmType metric = 9.99;
   const IntEbmType retApply = ApplyModelUpdate(
      boosterHandle,
      &metric
   );
   CHECK(0 == retApply);
   CHECK(0 == metric);

   CHECK(0 == GetBestModelFeatureGroup(boosterHandle, 0, nullptr));
   CHECK(0 == GetCurrentModelFeatureGroup(boosterHandle, 0, nullptr));

   FreeBooster(boosterHandle);
}

TEST_CASE("features with 1 state in various positions, boosting") {
   TestApi test0 = TestApi(k_learningTypeRegression);
   test0.AddFeatures({
      FeatureTest(1),
      FeatureTest(2),
      FeatureTest(2)
      });
   test0.AddFeatureGroups({ { 0 }, { 1 }, { 2 } });
   test0.AddTrainingSamples({ TestSample({ 0, 1, 1 }, 10) });
   test0.AddValidationSamples({ TestSample({ 0, 1, 1 }, 12) });
   test0.InitializeBoosting();

   TestApi test1 = TestApi(k_learningTypeRegression);
   test1.AddFeatures({
      FeatureTest(2),
      FeatureTest(1),
      FeatureTest(2)
      });
   test1.AddFeatureGroups({ { 0 }, { 1 }, { 2 } });
   test1.AddTrainingSamples({ TestSample({ 1, 0, 1 }, 10) });
   test1.AddValidationSamples({ TestSample({ 1, 0, 1 }, 12) });
   test1.InitializeBoosting();

   TestApi test2 = TestApi(k_learningTypeRegression);
   test2.AddFeatures({
      FeatureTest(2),
      FeatureTest(2),
      FeatureTest(1)
      });
   test2.AddFeatureGroups({ { 0 }, { 1 }, { 2 } });
   test2.AddTrainingSamples({ TestSample({ 1, 1, 0 }, 10) });
   test2.AddValidationSamples({ TestSample({ 1, 1, 0 }, 12) });
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
   test.AddTrainingSamples({ TestSample({}, 10) });
   test.AddValidationSamples({ TestSample({}, 12) });
   test.InitializeBoosting();

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just crash/assert if there was a problem
   // boosting isn't legal since we'd need to specify an featureGroup index
}

TEST_CASE("zero FeatureGroups, boosting, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({});
   test.AddFeatureGroups({});
   test.AddTrainingSamples({ TestSample({}, 1) });
   test.AddValidationSamples({ TestSample({}, 1) });
   test.InitializeBoosting();

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just crash/assert if there was a problem
   // boosting isn't legal since we'd need to specify an featureGroup index
}

TEST_CASE("zero FeatureGroups, boosting, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({});
   test.AddFeatureGroups({});
   test.AddTrainingSamples({ TestSample({}, 2) });
   test.AddValidationSamples({ TestSample({}, 2) });
   test.InitializeBoosting();

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just crash/assert if there was a problem
   // boosting isn't legal since we'd need to specify an featureGroup index
}

TEST_CASE("FeatureGroup with zero features, boosting, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddFeatureGroups({ {} });
   test.AddTrainingSamples({ TestSample({}, 10) });
   test.AddValidationSamples({ TestSample({}, 12) });
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
   test.AddTrainingSamples({ TestSample({}, 0) });
   test.AddValidationSamples({ TestSample({}, 0) });
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
   test.AddTrainingSamples({ TestSample({}, 0) });
   test.AddValidationSamples({ TestSample({}, 0) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   FloatEbmType modelValue = FloatEbmType { std::numeric_limits<FloatEbmType>::quiet_NaN() };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup);
         if(0 == iFeatureGroup && 0 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 1.0688384008227103, double { 1e-1 });
            FloatEbmType zeroLogit = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1) - zeroLogit;
            CHECK_APPROX(modelValue, -0.04500000000000000);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 2) - zeroLogit;
            CHECK_APPROX(modelValue, -0.04500000000000000);
         }
         if(0 == iFeatureGroup && 1 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 1.0401627411809615, double { 1e-1 });
            FloatEbmType zeroLogit = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1) - zeroLogit;
            CHECK_APPROX_TOLERANCE(modelValue, -0.089007468617193456, double { 1e-2 });
            modelValue = test.GetCurrentModelPredictorScore(iFeatureGroup, {}, 2) - zeroLogit;
            CHECK_APPROX_TOLERANCE(modelValue, -0.089007468617193456, double { 1e-2 });
         }
      }
   }
   CHECK_APPROX_TOLERANCE(validationMetric, 1.7171897252232722e-09, double { 1e+1 });
   FloatEbmType zeroLogit1 = test.GetCurrentModelPredictorScore(0, {}, 0);
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 1) - zeroLogit1;
   CHECK_APPROX_TOLERANCE(modelValue, -20.875723973004794, double { 1e-3 });
   modelValue = test.GetCurrentModelPredictorScore(0, {}, 2) - zeroLogit1;
   CHECK_APPROX_TOLERANCE(modelValue, -20.875723973004794, double { 1e-3 });
}

TEST_CASE("FeatureGroup with one feature with one or two states is the exact same as zero FeatureGroups, boosting, regression") {
   TestApi testZeroDimensions = TestApi(k_learningTypeRegression);
   testZeroDimensions.AddFeatures({});
   testZeroDimensions.AddFeatureGroups({ {} });
   testZeroDimensions.AddTrainingSamples({ TestSample({}, 10) });
   testZeroDimensions.AddValidationSamples({ TestSample({}, 12) });
   testZeroDimensions.InitializeBoosting();

   TestApi testOneState = TestApi(k_learningTypeRegression);
   testOneState.AddFeatures({ FeatureTest(1) });
   testOneState.AddFeatureGroups({ { 0 } });
   testOneState.AddTrainingSamples({ TestSample({ 0 }, 10) });
   testOneState.AddValidationSamples({ TestSample({ 0 }, 12) });
   testOneState.InitializeBoosting();

   TestApi testTwoStates = TestApi(k_learningTypeRegression);
   testTwoStates.AddFeatures({ FeatureTest(2) });
   testTwoStates.AddFeatureGroups({ { 0 } });
   testTwoStates.AddTrainingSamples({ TestSample({ 1 }, 10) });
   testTwoStates.AddValidationSamples({ TestSample({ 1 }, 12) });
   testTwoStates.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroDimensions.GetFeatureGroupsCount() == testOneState.GetFeatureGroupsCount());
      assert(testZeroDimensions.GetFeatureGroupsCount() == testTwoStates.GetFeatureGroupsCount());
      for(size_t iFeatureGroup = 0; iFeatureGroup < testZeroDimensions.GetFeatureGroupsCount(); ++iFeatureGroup) {
         FloatEbmType validationMetricZeroDimensions = testZeroDimensions.Boost(iFeatureGroup);
         FloatEbmType validationMetricOneState = testOneState.Boost(iFeatureGroup);
         CHECK_APPROX(validationMetricZeroDimensions, validationMetricOneState);
         FloatEbmType validationMetricTwoStates = testTwoStates.Boost(iFeatureGroup);
         CHECK_APPROX(validationMetricZeroDimensions, validationMetricTwoStates);

         FloatEbmType modelValueZeroDimensions = testZeroDimensions.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
         FloatEbmType modelValueOneState = testOneState.GetCurrentModelPredictorScore(iFeatureGroup, { 0 }, 0);
         CHECK_APPROX(modelValueZeroDimensions, modelValueOneState);
         FloatEbmType modelValueTwoStates = testTwoStates.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 0);
         CHECK_APPROX(modelValueZeroDimensions, modelValueTwoStates);
      }
   }
}

TEST_CASE("FeatureGroup with one feature with one or two states is the exact same as zero FeatureGroups, boosting, binary") {
   TestApi testZeroDimensions = TestApi(2, 0);
   testZeroDimensions.AddFeatures({});
   testZeroDimensions.AddFeatureGroups({ {} });
   testZeroDimensions.AddTrainingSamples({ TestSample({}, 0) });
   testZeroDimensions.AddValidationSamples({ TestSample({}, 0) });
   testZeroDimensions.InitializeBoosting();

   TestApi testOneState = TestApi(2, 0);
   testOneState.AddFeatures({ FeatureTest(1) });
   testOneState.AddFeatureGroups({ { 0 } });
   testOneState.AddTrainingSamples({ TestSample({ 0 }, 0) });
   testOneState.AddValidationSamples({ TestSample({ 0 }, 0) });
   testOneState.InitializeBoosting();

   TestApi testTwoStates = TestApi(2, 0);
   testTwoStates.AddFeatures({ FeatureTest(2) });
   testTwoStates.AddFeatureGroups({ { 0 } });
   testTwoStates.AddTrainingSamples({ TestSample({ 1 }, 0) });
   testTwoStates.AddValidationSamples({ TestSample({ 1 }, 0) });
   testTwoStates.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroDimensions.GetFeatureGroupsCount() == testOneState.GetFeatureGroupsCount());
      assert(testZeroDimensions.GetFeatureGroupsCount() == testTwoStates.GetFeatureGroupsCount());
      for(size_t iFeatureGroup = 0; iFeatureGroup < testZeroDimensions.GetFeatureGroupsCount(); ++iFeatureGroup) {
         FloatEbmType validationMetricZeroDimensions = testZeroDimensions.Boost(iFeatureGroup);
         FloatEbmType validationMetricOneState = testOneState.Boost(iFeatureGroup);
         CHECK_APPROX(validationMetricZeroDimensions, validationMetricOneState);
         FloatEbmType validationMetricTwoStates = testTwoStates.Boost(iFeatureGroup);
         CHECK_APPROX(validationMetricZeroDimensions, validationMetricTwoStates);

         FloatEbmType modelValueZeroDimensions0 = testZeroDimensions.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
         FloatEbmType modelValueOneState0 = testOneState.GetCurrentModelPredictorScore(iFeatureGroup, { 0 }, 0);
         CHECK_APPROX(modelValueZeroDimensions0, modelValueOneState0);
         FloatEbmType modelValueTwoStates0 = testTwoStates.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 0);
         CHECK_APPROX(modelValueZeroDimensions0, modelValueTwoStates0);

         FloatEbmType modelValueZeroDimensions1 = testZeroDimensions.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1);
         FloatEbmType modelValueOneState1 = testOneState.GetCurrentModelPredictorScore(iFeatureGroup, { 0 }, 1);
         CHECK_APPROX(modelValueZeroDimensions1, modelValueOneState1);
         FloatEbmType modelValueTwoStates1 = testTwoStates.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 1);
         CHECK_APPROX(modelValueZeroDimensions1, modelValueTwoStates1);
      }
   }
}

TEST_CASE("FeatureGroup with one feature with one or two states is the exact same as zero FeatureGroups, boosting, multiclass") {
   TestApi testZeroDimensions = TestApi(3);
   testZeroDimensions.AddFeatures({});
   testZeroDimensions.AddFeatureGroups({ {} });
   testZeroDimensions.AddTrainingSamples({ TestSample({}, 0) });
   testZeroDimensions.AddValidationSamples({ TestSample({}, 0) });
   testZeroDimensions.InitializeBoosting();

   TestApi testOneState = TestApi(3);
   testOneState.AddFeatures({ FeatureTest(1) });
   testOneState.AddFeatureGroups({ { 0 } });
   testOneState.AddTrainingSamples({ TestSample({ 0 }, 0) });
   testOneState.AddValidationSamples({ TestSample({ 0 }, 0) });
   testOneState.InitializeBoosting();

   TestApi testTwoStates = TestApi(3);
   testTwoStates.AddFeatures({ FeatureTest(2) });
   testTwoStates.AddFeatureGroups({ { 0 } });
   testTwoStates.AddTrainingSamples({ TestSample({ 1 }, 0) });
   testTwoStates.AddValidationSamples({ TestSample({ 1 }, 0) });
   testTwoStates.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroDimensions.GetFeatureGroupsCount() == testOneState.GetFeatureGroupsCount());
      assert(testZeroDimensions.GetFeatureGroupsCount() == testTwoStates.GetFeatureGroupsCount());
      for(size_t iFeatureGroup = 0; iFeatureGroup < testZeroDimensions.GetFeatureGroupsCount(); ++iFeatureGroup) {
         FloatEbmType validationMetricZeroDimensions = testZeroDimensions.Boost(iFeatureGroup);
         FloatEbmType validationMetricOneState = testOneState.Boost(iFeatureGroup);
         CHECK_APPROX(validationMetricZeroDimensions, validationMetricOneState);
         FloatEbmType validationMetricTwoStates = testTwoStates.Boost(iFeatureGroup);
         CHECK_APPROX(validationMetricZeroDimensions, validationMetricTwoStates);

         FloatEbmType modelValueZeroDimensions0 = testZeroDimensions.GetCurrentModelPredictorScore(iFeatureGroup, {}, 0);
         FloatEbmType modelValueOneState0 = testOneState.GetCurrentModelPredictorScore(iFeatureGroup, { 0 }, 0);
         CHECK_APPROX(modelValueZeroDimensions0, modelValueOneState0);
         FloatEbmType modelValueTwoStates0 = testTwoStates.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 0);
         CHECK_APPROX(modelValueZeroDimensions0, modelValueTwoStates0);

         FloatEbmType modelValueZeroDimensions1 = testZeroDimensions.GetCurrentModelPredictorScore(iFeatureGroup, {}, 1);
         FloatEbmType modelValueOneState1 = testOneState.GetCurrentModelPredictorScore(iFeatureGroup, { 0 }, 1);
         CHECK_APPROX(modelValueZeroDimensions1, modelValueOneState1);
         FloatEbmType modelValueTwoStates1 = testTwoStates.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 1);
         CHECK_APPROX(modelValueZeroDimensions1, modelValueTwoStates1);

         FloatEbmType modelValueZeroDimensions2 = testZeroDimensions.GetCurrentModelPredictorScore(iFeatureGroup, {}, 2);
         FloatEbmType modelValueOneState2 = testOneState.GetCurrentModelPredictorScore(iFeatureGroup, { 0 }, 2);
         CHECK_APPROX(modelValueZeroDimensions2, modelValueOneState2);
         FloatEbmType modelValueTwoStates2 = testTwoStates.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 2);
         CHECK_APPROX(modelValueZeroDimensions2, modelValueTwoStates2);
      }
   }
}

TEST_CASE("3 dimensional featureGroup with one dimension reduced in different ways, boosting, regression") {
   TestApi test0 = TestApi(k_learningTypeRegression);
   test0.AddFeatures({ FeatureTest(1), FeatureTest(2), FeatureTest(2) });
   test0.AddFeatureGroups({ { 0, 1, 2 } });
   test0.AddTrainingSamples({
      TestSample({ 0, 0, 0 }, 9),
      TestSample({ 0, 1, 0 }, 10),
      TestSample({ 0, 0, 1 }, 11),
      TestSample({ 0, 1, 1 }, 12),
      });
   test0.AddValidationSamples({ TestSample({ 0, 1, 0 }, 12) });
   test0.InitializeBoosting();

   TestApi test1 = TestApi(k_learningTypeRegression);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(1), FeatureTest(2) });
   test1.AddFeatureGroups({ { 0, 1, 2 } });
   test1.AddTrainingSamples({
      TestSample({ 0, 0, 0 }, 9),
      TestSample({ 0, 0, 1 }, 10),
      TestSample({ 1, 0, 0 }, 11),
      TestSample({ 1, 0, 1 }, 12),
      });
   test1.AddValidationSamples({ TestSample({ 0, 0, 1 }, 12) });
   test1.InitializeBoosting();

   TestApi test2 = TestApi(k_learningTypeRegression);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2), FeatureTest(1) });
   test2.AddFeatureGroups({ { 0, 1, 2 } });
   test2.AddTrainingSamples({
      TestSample({ 0, 0, 0 }, 9),
      TestSample({ 1, 0, 0 }, 10),
      TestSample({ 0, 1, 0 }, 11),
      TestSample({ 1, 1, 0 }, 12),
      });
   test2.AddValidationSamples({ TestSample({ 1, 0, 0 }, 12) });
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
   static const std::vector<IntEbmType> k_leavesMax = {
      IntEbmType { 3 }
   };

   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(4) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples({
      TestSample({ 0 }, 0),
      TestSample({ 1 }, 1),
      TestSample({ 2 }, 1),
      TestSample({ 3 }, 2)
   });
   test.AddValidationSamples({ TestSample({ 1 }, 0) });
   test.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         FloatEbmType validationMetric = test.Boost(iFeatureGroup, GenerateUpdateOptions_RandomSplits, k_learningRateDefault, k_countSamplesRequiredForChildSplitMinDefault, k_leavesMax);
         if(0 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 1.0340957641601563f, double { 1e-1 });

            FloatEbmType zeroLogit = test.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 0);

            FloatEbmType modelValue1 = test.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 1) - zeroLogit;
            CHECK_APPROX(modelValue1, 0.0f);

            FloatEbmType modelValue2 = test.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 2) - zeroLogit;
            CHECK_APPROX(modelValue2, -0.0225f);
         }
      }
   }
}

TEST_CASE("Random splitting with 3 features, boosting, multiclass, sums") {
   static const std::vector<IntEbmType> k_leavesMax = {
      IntEbmType { 3 }
   };

   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(4) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples({
      TestSample({ 0 }, 0),
      TestSample({ 1 }, 1),
      TestSample({ 2 }, 1),
      TestSample({ 3 }, 2)
      });
   test.AddValidationSamples({ TestSample({ 1 }, 0) });
   test.InitializeBoosting();

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         FloatEbmType validationMetric = test.Boost(iFeatureGroup, GenerateUpdateOptions_RandomSplits | GenerateUpdateOptions_GradientSums, k_learningRateDefault, k_countSamplesRequiredForChildSplitMinDefault, k_leavesMax);
         if(0 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 1.0986122886681098, double { 1e-1 });

            // we set our update to zero since this is for sums so we need to set it to something

            FloatEbmType modelValue0 = test.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 0);
            CHECK_APPROX(modelValue0, 0.0f);

            FloatEbmType modelValue1 = test.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 1);
            CHECK_APPROX(modelValue1, 0.0f);

            FloatEbmType modelValue2 = test.GetCurrentModelPredictorScore(iFeatureGroup, { 1 }, 2);
            CHECK_APPROX(modelValue2, 0.0f);
         }
      }
   }
}

TEST_CASE("Random splitting, tripple with one dimension missing, multiclass") {
   constexpr IntEbmType cStates = 7;
   static const std::vector<IntEbmType> k_leavesMax = {
      IntEbmType { 3 },
      IntEbmType { 3 },
      IntEbmType { 3 }
   };

   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(cStates), FeatureTest(1), FeatureTest(cStates) });
   test.AddFeatureGroups({ { 0, 1, 2 } });
   std::vector<TestSample> samples;
   for(IntEbmType i0 = 0; i0 < cStates; ++i0) {
      for(IntEbmType i2 = 0; i2 < cStates; ++i2) {
         // create a few zero spaces where we have no data
         if(i0 != i2) {
            if(i0 < i2) {
               samples.push_back(TestSample({ i0, 0, i2 }, 1));
            } else {
               samples.push_back(TestSample({ i0, 0, i2 }, 2));
            }
         }
      }
   }

   test.AddTrainingSamples(samples);
   test.AddValidationSamples(samples); // evaluate on the train set
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { 0 };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup, GenerateUpdateOptions_RandomSplits, k_learningRateDefault, 1, k_leavesMax);
      }
   }

   CHECK(validationMetric <= 0.00017711094447544644f * 1.1);

   for(IntEbmType i0 = 0; i0 < cStates; ++i0) {
      for(IntEbmType i2 = 0; i2 < cStates; ++i2) {
#if false
         std::cout << std::endl;
         std::cout << i0 << ' ' << '0' << ' ' << i2 << std::endl;

         FloatEbmType modelValue0 = test.GetCurrentModelPredictorScore(0, { static_cast<size_t>(i0), static_cast<size_t>(0), static_cast<size_t>(i2) }, 0);
         std::cout << modelValue0 << std::endl;

         FloatEbmType modelValue1 = test.GetCurrentModelPredictorScore(0, { static_cast<size_t>(i0), static_cast<size_t>(0), static_cast<size_t>(i2) }, 1);
         std::cout << modelValue1 << std::endl;

         FloatEbmType modelValue2 = test.GetCurrentModelPredictorScore(0, { static_cast<size_t>(i0), static_cast<size_t>(0), static_cast<size_t>(i2) }, 2);
         std::cout << modelValue2 << std::endl;
#endif
      }
   }
}

TEST_CASE("Random splitting, pure tripples, multiclass") {
   constexpr IntEbmType cStates = 7;
   static const std::vector<IntEbmType> k_leavesMax = {
      IntEbmType { 3 },
      IntEbmType { 3 },
      IntEbmType { 3 }
   };

   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(cStates), FeatureTest(cStates), FeatureTest(cStates) });
   test.AddFeatureGroups({ { 0, 1, 2 } });
   std::vector<TestSample> samples;
   for(IntEbmType i0 = 0; i0 < cStates; ++i0) {
      for(IntEbmType i1 = 0; i1 < cStates; ++i1) {
         for(IntEbmType i2 = 0; i2 < cStates; ++i2) {
            if(i0 == i1 && i0 == i2) {
               samples.push_back(TestSample({ i0, i1, i2 }, 0));
            } else if(i0 < i1) {
               samples.push_back(TestSample({ i0, i1, i2 }, 1));
            } else {
               samples.push_back(TestSample({ i0, i1, i2 }, 2));
            }
         }
      }
   }

   test.AddTrainingSamples(samples);
   test.AddValidationSamples(samples); // evaluate on the train set
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { 0 };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup, GenerateUpdateOptions_RandomSplits, k_learningRateDefault, 1, k_leavesMax);
      }
   }
   CHECK(validationMetric <= 0.0091562298922079986f * 1.4);

   for(IntEbmType i0 = 0; i0 < cStates; ++i0) {
      for(IntEbmType i1 = 0; i1 < cStates; ++i1) {
         for(IntEbmType i2 = 0; i2 < cStates; ++i2) {
#if false
            std::cout << std::endl;
            std::cout << i0 << ' ' << i1 << ' ' << i2 << std::endl;

            FloatEbmType modelValue0 = test.GetCurrentModelPredictorScore(0, { static_cast<size_t>(i0), static_cast<size_t>(i1), static_cast<size_t>(i2) }, 0);
            std::cout << modelValue0 << std::endl;

            FloatEbmType modelValue1 = test.GetCurrentModelPredictorScore(0, { static_cast<size_t>(i0), static_cast<size_t>(i1), static_cast<size_t>(i2) }, 1);
            std::cout << modelValue1 << std::endl;

            FloatEbmType modelValue2 = test.GetCurrentModelPredictorScore(0, { static_cast<size_t>(i0), static_cast<size_t>(i1), static_cast<size_t>(i2) }, 2);
            std::cout << modelValue2 << std::endl;
#endif
         }
      }
   }
}

TEST_CASE("Random splitting, pure tripples, regression") {
   constexpr IntEbmType cStates = 7;
   static const std::vector<IntEbmType> k_leavesMax = {
      IntEbmType { 3 },
      IntEbmType { 3 },
      IntEbmType { 3 }
   };

   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(cStates), FeatureTest(cStates), FeatureTest(cStates) });
   test.AddFeatureGroups({ { 0, 1, 2 } });
   std::vector<TestSample> samples;
   for(IntEbmType i0 = 0; i0 < cStates; ++i0) {
      for(IntEbmType i1 = 0; i1 < cStates; ++i1) {
         for(IntEbmType i2 = 0; i2 < cStates; ++i2) {
            if(i0 == i1 && i0 == i2) {
               samples.push_back(TestSample({ i0, i1, i2 }, -10));
            } else if(i0 < i1) {
               samples.push_back(TestSample({ i0, i1, i2 }, 1));
            } else {
               samples.push_back(TestSample({ i0, i1, i2 }, 2));
            }
         }
      }
   }

   test.AddTrainingSamples(samples);
   test.AddValidationSamples(samples); // evaluate on the train set
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { 0 };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup, GenerateUpdateOptions_RandomSplits, k_learningRateDefault, 1, k_leavesMax);
      }
   }

   CHECK_APPROX(validationMetric, 1.4656199141470665);

   for(IntEbmType i0 = 0; i0 < cStates; ++i0) {
      for(IntEbmType i1 = 0; i1 < cStates; ++i1) {
         for(IntEbmType i2 = 0; i2 < cStates; ++i2) {
#if false
            std::cout << std::endl;
            std::cout << i0 << ' ' << i1 << ' ' << i2 << std::endl;

            FloatEbmType modelValue0 = test.GetCurrentModelPredictorScore(0, { static_cast<size_t>(i0), static_cast<size_t>(i1), static_cast<size_t>(i2) }, 0);
            std::cout << modelValue0 << std::endl;
#endif
         }
      }
   }
}

TEST_CASE("Random splitting, pure tripples, only 1 leaf, multiclass") {
   constexpr IntEbmType k_cStates = 7;
   constexpr IntEbmType k_countSamplesRequiredForChildSplitMin = 1;
   static const std::vector<IntEbmType> k_leavesMax = {
      IntEbmType { 1 },
      IntEbmType { 1 },
      IntEbmType { 1 }
   };

   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(k_cStates), FeatureTest(k_cStates), FeatureTest(k_cStates) });
   test.AddFeatureGroups({ { 0, 1, 2 } });
   std::vector<TestSample> samples;
   for(IntEbmType i0 = 0; i0 < k_cStates; ++i0) {
      for(IntEbmType i1 = 0; i1 < k_cStates; ++i1) {
         for(IntEbmType i2 = 0; i2 < k_cStates; ++i2) {
            if(i0 == i1 && i0 == i2) {
               samples.push_back(TestSample({ i0, i1, i2 }, 0));
            } else if(i0 < i1) {
               samples.push_back(TestSample({ i0, i1, i2 }, 1));
            } else {
               samples.push_back(TestSample({ i0, i1, i2 }, 2));
            }
         }
      }
   }

   test.AddTrainingSamples(samples);
   test.AddValidationSamples(samples); // evaluate on the train set
   test.InitializeBoosting();

   FloatEbmType validationMetric = FloatEbmType { 0 };
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(
            iFeatureGroup, 
            GenerateUpdateOptions_RandomSplits, 
            k_learningRateDefault,
            k_countSamplesRequiredForChildSplitMin,
            k_leavesMax
         );
      }
   }

   // it can't really benefit from cutting since we only allow the boosting rounds to have 1 leaf
   CHECK(validationMetric <= 0.73616339235889672f * 1.1);
   CHECK(0.73616339235889672f / 1.1 <= validationMetric);

   for(IntEbmType i0 = 0; i0 < k_cStates; ++i0) {
      for(IntEbmType i1 = 0; i1 < k_cStates; ++i1) {
         for(IntEbmType i2 = 0; i2 < k_cStates; ++i2) {
#if false
            std::cout << std::endl;
            std::cout << i0 << ' ' << i1 << ' ' << i2 << std::endl;

            FloatEbmType modelValue0 = test.GetCurrentModelPredictorScore(0, { static_cast<size_t>(i0), static_cast<size_t>(i1), static_cast<size_t>(i2) }, 0);
            std::cout << modelValue0 << std::endl;

            FloatEbmType modelValue1 = test.GetCurrentModelPredictorScore(0, { static_cast<size_t>(i0), static_cast<size_t>(i1), static_cast<size_t>(i2) }, 1);
            std::cout << modelValue1 << std::endl;

            FloatEbmType modelValue2 = test.GetCurrentModelPredictorScore(0, { static_cast<size_t>(i0), static_cast<size_t>(i1), static_cast<size_t>(i2) }, 2);
            std::cout << modelValue2 << std::endl;
#endif
         }
      }
   }
}

TEST_CASE("Random splitting, no cuts, binary, sums") {
   static const std::vector<IntEbmType> k_leavesMax = {
      IntEbmType { 3 }
   };

   TestApi test = TestApi(2);
   test.AddFeatures({ FeatureTest(1) });
   test.AddFeatureGroups({ { 0 } });
   test.AddTrainingSamples({
      TestSample({ 0 }, 0),
      TestSample({ 0 }, 0),
      TestSample({ 0 }, 1),
      TestSample({ 0 }, 1),
      TestSample({ 0 }, 1),
      });
   test.AddValidationSamples({ TestSample({ 0 }, 0) });
   test.InitializeBoosting();

   FloatEbmType validationMetric = 0;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         validationMetric = test.Boost(iFeatureGroup, GenerateUpdateOptions_RandomSplits | GenerateUpdateOptions_GradientSums, k_learningRateDefault, k_countSamplesRequiredForChildSplitMinDefault, k_leavesMax);
         if(0 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 0.69314718055994529f, double { 1e-1 });

            // we set our update to zero since we're getting the sum

            FloatEbmType modelValue0 = test.GetCurrentModelPredictorScore(iFeatureGroup, { 0 }, 0);
            CHECK_APPROX(modelValue0, 0.0f);

            FloatEbmType modelValue1 = test.GetCurrentModelPredictorScore(iFeatureGroup, { 0 }, 1);
            CHECK_APPROX(modelValue1, 0.0f);
         }
      }
   }

   // we're generating updates from gradient sums, which isn't good, so we expect a bad result
   CHECK_APPROX_TOLERANCE(validationMetric, 0.69314718055994529f, double { 1e-1 });
}

