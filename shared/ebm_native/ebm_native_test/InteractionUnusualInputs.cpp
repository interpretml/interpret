// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderEbmNativeTest.h"

#include "ebm_native.h"
#include "EbmNativeTest.h"

static const TestPriority k_filePriority = TestPriority::InteractionUnusualInputs;

TEST_CASE("null interactionScoreOut, interaction, regression") {
   const InteractionDetectorHandle interactionDetectorHandle = CreateRegressionInteractionDetector(0, nullptr, nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr);
   const IntEbmType ret = CalculateInteractionScore(interactionDetectorHandle, 0, nullptr, k_countSamplesRequiredForChildSplitMinDefault, nullptr);
   CHECK(0 == ret);
   FreeInteractionDetector(interactionDetectorHandle);
}

TEST_CASE("null interactionScoreOut, interaction, binary") {
   const InteractionDetectorHandle interactionDetectorHandle = CreateClassificationInteractionDetector(2, 0, nullptr, nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr);
   const IntEbmType ret = CalculateInteractionScore(interactionDetectorHandle, 0, nullptr, k_countSamplesRequiredForChildSplitMinDefault, nullptr);
   CHECK(0 == ret);
   FreeInteractionDetector(interactionDetectorHandle);
}

TEST_CASE("null interactionScoreOut, interaction, multiclass") {
   const InteractionDetectorHandle interactionDetectorHandle = CreateClassificationInteractionDetector(3, 0, nullptr, nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr);
   const IntEbmType ret = CalculateInteractionScore(interactionDetectorHandle, 0, nullptr, k_countSamplesRequiredForChildSplitMinDefault, nullptr);
   CHECK(0 == ret);
   FreeInteractionDetector(interactionDetectorHandle);
}

TEST_CASE("Zero interaction samples, interaction, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionSamples(std::vector<RegressionSample> {});
   test.InitializeInteraction();

   FloatEbmType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("Zero interaction samples, interaction, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionSamples(std::vector<ClassificationSample> {});
   test.InitializeInteraction();

   FloatEbmType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("Zero interaction samples, interaction, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionSamples(std::vector<ClassificationSample> {});
   test.InitializeInteraction();

   FloatEbmType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("classification with 0 possible target states, interaction") {
   TestApi test = TestApi(0);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionSamples(std::vector<ClassificationSample> {});
   test.InitializeInteraction();

   FloatEbmType validationMetric = test.InteractionScore({ 0 });
   CHECK(0 == validationMetric);
}

TEST_CASE("classification with 1 possible target, interaction") {
   TestApi test = TestApi(1);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionSamples({ ClassificationSample(0, { 1 }) });
   test.InitializeInteraction();

   FloatEbmType validationMetric = test.InteractionScore({ 0 });
   CHECK(0 == validationMetric);
}

TEST_CASE("features with 0 states, interaction") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(0) });
   test.AddInteractionSamples(std::vector<RegressionSample> {});
   test.InitializeInteraction();

   FloatEbmType validationMetric = test.InteractionScore({ 0 });
   CHECK(0 == validationMetric);
}

TEST_CASE("FeatureGroup with zero features, interaction, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddInteractionSamples({ RegressionSample(10, {}) });
   test.InitializeInteraction();
   FloatEbmType metricReturn = test.InteractionScore({});
   CHECK(0 == metricReturn);
}

TEST_CASE("FeatureGroup with zero features, interaction, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({});
   test.AddInteractionSamples({ ClassificationSample(0, {}) });
   test.InitializeInteraction();
   FloatEbmType metricReturn = test.InteractionScore({});
   CHECK(0 == metricReturn);
}

TEST_CASE("FeatureGroup with zero features, interaction, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({});
   test.AddInteractionSamples({ ClassificationSample(0, {}) });
   test.InitializeInteraction();
   FloatEbmType metricReturn = test.InteractionScore({});
   CHECK(0 == metricReturn);
}

TEST_CASE("FeatureGroup with one feature with one state, interaction, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(1) });
   test.AddInteractionSamples({ RegressionSample(10, { 0 }) });
   test.InitializeInteraction();
   FloatEbmType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("FeatureGroup with one feature with one state, interaction, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({ FeatureTest(1) });
   test.AddInteractionSamples({ ClassificationSample(0, { 0 }) });
   test.InitializeInteraction();
   FloatEbmType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("FeatureGroup with one feature with one state, interaction, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(1) });
   test.AddInteractionSamples({ ClassificationSample(0, { 0 }) });
   test.InitializeInteraction();
   FloatEbmType metricReturn = test.InteractionScore({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("weights are proportional, interaction, regression") {
   TestApi test1 = TestApi(k_learningTypeRegression);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test1.AddInteractionSamples({ 
      RegressionSample(10.1, { 0, 0 }, 0, std::nextafter(0.3, 100)),
      RegressionSample(20.2, { 0, 1 }, 0, 0.3),
      RegressionSample(30.3, { 1, 0 }, 0, 0.3),
      RegressionSample(40.4, { 1, 1 }, 0, 0.3),
      });
   test1.InitializeInteraction();
   FloatEbmType metricReturn1 = test1.InteractionScore({ 0, 1 });

   TestApi test2 = TestApi(k_learningTypeRegression);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test2.AddInteractionSamples({
      RegressionSample(10.1, { 0, 0 }, 0, std::nextafter(2, 100)),
      RegressionSample(20.2, { 0, 1 }, 0, 2),
      RegressionSample(30.3, { 1, 0 }, 0, 2),
      RegressionSample(40.4, { 1, 1 }, 0, 2),
      });
   test2.InitializeInteraction();
   FloatEbmType metricReturn2 = test2.InteractionScore({ 0, 1 });

   TestApi test3 = TestApi(k_learningTypeRegression);
   test3.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test3.AddInteractionSamples({
      RegressionSample(10.1, { 0, 0 }, 0, 0),
      RegressionSample(20.2, { 0, 1 }, 0, 0),
      RegressionSample(30.3, { 1, 0 }, 0, 0),
      RegressionSample(40.4, { 1, 1 }, 0, 0),
      });
   test3.InitializeInteraction();
   FloatEbmType metricReturn3 = test3.InteractionScore({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
   CHECK_APPROX(metricReturn1, metricReturn3);
}

TEST_CASE("weights are proportional, interaction, binary") {
   TestApi test1 = TestApi(2);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test1.AddInteractionSamples({
      ClassificationSample(0, { 0, 0 }, { 0, 0 }, std::nextafter(0.3, 100)),
      ClassificationSample(1, { 0, 1 }, { 0, 0 }, 0.3),
      ClassificationSample(1, { 1, 0 }, { 0, 0 }, 0.3),
      ClassificationSample(0, { 1, 1 }, { 0, 0 }, 0.3),
      });
   test1.InitializeInteraction();
   FloatEbmType metricReturn1 = test1.InteractionScore({ 0, 1 });

   TestApi test2 = TestApi(2);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test2.AddInteractionSamples({
      ClassificationSample(0, { 0, 0 }, { 0, 0 }, std::nextafter(2, 100)),
      ClassificationSample(1, { 0, 1 }, { 0, 0 }, 2),
      ClassificationSample(1, { 1, 0 }, { 0, 0 }, 2),
      ClassificationSample(0, { 1, 1 }, { 0, 0 }, 2),
      });
   test2.InitializeInteraction();
   FloatEbmType metricReturn2 = test2.InteractionScore({ 0, 1 });

   TestApi test3 = TestApi(2);
   test3.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test3.AddInteractionSamples({
      ClassificationSample(0, { 0, 0 }, { 0, 0 }, 0),
      ClassificationSample(1, { 0, 1 }, { 0, 0 }, 0),
      ClassificationSample(1, { 1, 0 }, { 0, 0 }, 0),
      ClassificationSample(0, { 1, 1 }, { 0, 0 }, 0),
      });
   test3.InitializeInteraction();
   FloatEbmType metricReturn3 = test3.InteractionScore({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
   CHECK_APPROX(metricReturn1, metricReturn3);
}

TEST_CASE("weights are proportional, interaction, multiclass") {
   TestApi test1 = TestApi(3);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test1.AddInteractionSamples({
      ClassificationSample(0, { 0, 0 }, { 0, 0, 0 }, std::nextafter(0.3, 100)),
      ClassificationSample(1, { 0, 1 }, { 0, 0, 0 }, 0.3),
      ClassificationSample(2, { 1, 0 }, { 0, 0, 0 }, 0.3),
      ClassificationSample(0, { 1, 1 }, { 0, 0, 0 }, 0.3),
      });
   test1.InitializeInteraction();
   FloatEbmType metricReturn1 = test1.InteractionScore({ 0, 1 });

   TestApi test2 = TestApi(3);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test2.AddInteractionSamples({
      ClassificationSample(0, { 0, 0 }, { 0, 0, 0 }, std::nextafter(2, 100)),
      ClassificationSample(1, { 0, 1 }, { 0, 0, 0 }, 2),
      ClassificationSample(2, { 1, 0 }, { 0, 0, 0 }, 2),
      ClassificationSample(0, { 1, 1 }, { 0, 0, 0 }, 2),
      });
   test2.InitializeInteraction();
   FloatEbmType metricReturn2 = test2.InteractionScore({ 0, 1 });

   TestApi test3 = TestApi(3);
   test3.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test3.AddInteractionSamples({
      ClassificationSample(0, { 0, 0 }, { 0, 0, 0 }, 0),
      ClassificationSample(1, { 0, 1 }, { 0, 0, 0 }, 0),
      ClassificationSample(2, { 1, 0 }, { 0, 0, 0 }, 0),
      ClassificationSample(0, { 1, 1 }, { 0, 0, 0 }, 0),
      });
   test3.InitializeInteraction();
   FloatEbmType metricReturn3 = test3.InteractionScore({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
   CHECK_APPROX(metricReturn1, metricReturn3);
}

TEST_CASE("weights totals equivalence, interaction, regression") {
   TestApi test1 = TestApi(k_learningTypeRegression);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test1.AddInteractionSamples({
      RegressionSample(10.1, { 0, 0 }, 0, 0.15),
      RegressionSample(10.1, { 0, 0 }, 0, 0.15),
      RegressionSample(20.2, { 0, 1 }, 0, 0.3),
      RegressionSample(30.3, { 1, 0 }, 0, 0.3),
      RegressionSample(40.4, { 1, 1 }, 0, 0.3),
      });
   test1.InitializeInteraction();
   FloatEbmType metricReturn1 = test1.InteractionScore({ 0, 1 });

   TestApi test2 = TestApi(k_learningTypeRegression);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test2.AddInteractionSamples({
      RegressionSample(10.1, { 0, 0 }, 0, 2),
      RegressionSample(20.2, { 0, 1 }, 0, 2),
      RegressionSample(30.3, { 1, 0 }, 0, 1),
      RegressionSample(30.3, { 1, 0 }, 0, 1),
      RegressionSample(40.4, { 1, 1 }, 0, 2),
      });
   test2.InitializeInteraction();
   FloatEbmType metricReturn2 = test2.InteractionScore({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
}

TEST_CASE("weights totals equivalence, interaction, binary") {
   TestApi test1 = TestApi(2);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test1.AddInteractionSamples({
      ClassificationSample(0, { 0, 0 }, { 0, 0 }, 0.3),
      ClassificationSample(1, { 0, 1 }, { 0, 0 }, 0.15),
      ClassificationSample(1, { 0, 1 }, { 0, 0 }, 0.15),
      ClassificationSample(1, { 1, 0 }, { 0, 0 }, 0.3),
      ClassificationSample(0, { 1, 1 }, { 0, 0 }, 0.3),
      });
   test1.InitializeInteraction();
   FloatEbmType metricReturn1 = test1.InteractionScore({ 0, 1 });

   TestApi test2 = TestApi(2);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test2.AddInteractionSamples({
      ClassificationSample(0, { 0, 0 }, { 0, 0 }, 2),
      ClassificationSample(1, { 0, 1 }, { 0, 0 }, 2),
      ClassificationSample(1, { 1, 0 }, { 0, 0 }, 2),
      ClassificationSample(0, { 1, 1 }, { 0, 0 }, 1),
      ClassificationSample(0, { 1, 1 }, { 0, 0 }, 1),
      });
   test2.InitializeInteraction();
   FloatEbmType metricReturn2 = test2.InteractionScore({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
}

TEST_CASE("weights totals equivalence, interaction, multiclass") {
   TestApi test1 = TestApi(3);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test1.AddInteractionSamples({
      ClassificationSample(0, { 0, 0 }, { 0, 0, 0 }, 0.3),
      ClassificationSample(1, { 0, 1 }, { 0, 0, 0 }, 0.15),
      ClassificationSample(1, { 0, 1 }, { 0, 0, 0 }, 0.15),
      ClassificationSample(2, { 1, 0 }, { 0, 0, 0 }, 0.3),
      ClassificationSample(0, { 1, 1 }, { 0, 0, 0 }, 0.3),
      });
   test1.InitializeInteraction();
   FloatEbmType metricReturn1 = test1.InteractionScore({ 0, 1 });

   TestApi test2 = TestApi(3);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test2.AddInteractionSamples({
      ClassificationSample(0, { 0, 0 }, { 0, 0, 0 }, 1),
      ClassificationSample(0, { 0, 0 }, { 0, 0, 0 }, 1),
      ClassificationSample(1, { 0, 1 }, { 0, 0, 0 }, 2),
      ClassificationSample(2, { 1, 0 }, { 0, 0, 0 }, 2),
      ClassificationSample(0, { 1, 1 }, { 0, 0, 0 }, 2),
      });
   test2.InitializeInteraction();
   FloatEbmType metricReturn2 = test2.InteractionScore({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
}
