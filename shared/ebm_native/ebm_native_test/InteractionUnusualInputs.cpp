// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderEbmNativeTest.h"

#include "ebm_native.h"
#include "EbmNativeTest.h"

static const TestPriority k_filePriority = TestPriority::InteractionUnusualInputs;

TEST_CASE("null interactionScoreReturn, interaction, regression") {
   PEbmInteraction pEbmInteraction = InitializeInteractionRegression(0, nullptr, 0, nullptr, nullptr, nullptr, nullptr);
   const IntEbmType ret = CalculateInteractionScore(pEbmInteraction, 0, nullptr, k_countSamplesRequiredForChildSplitMinDefault, nullptr);
   CHECK(0 == ret);
   FreeInteraction(pEbmInteraction);
}

TEST_CASE("null interactionScoreReturn, interaction, binary") {
   PEbmInteraction pEbmInteraction = InitializeInteractionClassification(2, 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr);
   const IntEbmType ret = CalculateInteractionScore(pEbmInteraction, 0, nullptr, k_countSamplesRequiredForChildSplitMinDefault, nullptr);
   CHECK(0 == ret);
   FreeInteraction(pEbmInteraction);
}

TEST_CASE("null interactionScoreReturn, interaction, multiclass") {
   PEbmInteraction pEbmInteraction = InitializeInteractionClassification(3, 0, nullptr, 0, nullptr, nullptr, nullptr, nullptr);
   const IntEbmType ret = CalculateInteractionScore(pEbmInteraction, 0, nullptr, k_countSamplesRequiredForChildSplitMinDefault, nullptr);
   CHECK(0 == ret);
   FreeInteraction(pEbmInteraction);
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


