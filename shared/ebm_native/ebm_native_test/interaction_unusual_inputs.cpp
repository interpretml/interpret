// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"

static constexpr TestPriority k_filePriority = TestPriority::InteractionUnusualInputs;

TEST_CASE("Zero interaction samples, interaction, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionSamples({});
   test.InitializeInteraction();

   double metricReturn = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("Zero interaction samples, interaction, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionSamples({});
   test.InitializeInteraction();

   double metricReturn = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("Zero interaction samples, interaction, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionSamples({});
   test.InitializeInteraction();

   double metricReturn = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("classification with 0 possible target states, interaction") {
   TestApi test = TestApi(0);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionSamples({});
   test.InitializeInteraction();

   double validationMetric = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == validationMetric);
}

TEST_CASE("classification with 1 possible target, interaction") {
   TestApi test = TestApi(1);
   test.AddFeatures({ FeatureTest(2) });
   test.AddInteractionSamples({ TestSample({ 1 }, 0) });
   test.InitializeInteraction();

   double validationMetric = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == validationMetric);
}

TEST_CASE("features with 0 states, interaction") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(0) });
   test.AddInteractionSamples({});
   test.InitializeInteraction();

   double validationMetric = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == validationMetric);
}

TEST_CASE("Term with zero features, interaction, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({});
   test.AddInteractionSamples({ TestSample({}, 10) });
   test.InitializeInteraction();
   double metricReturn = test.TestCalcInteractionStrength({});
   CHECK(0 == metricReturn);
}

TEST_CASE("Term with zero features, interaction, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({});
   test.AddInteractionSamples({ TestSample({}, 0) });
   test.InitializeInteraction();
   double metricReturn = test.TestCalcInteractionStrength({});
   CHECK(0 == metricReturn);
}

TEST_CASE("Term with zero features, interaction, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({});
   test.AddInteractionSamples({ TestSample({}, 0) });
   test.InitializeInteraction();
   double metricReturn = test.TestCalcInteractionStrength({});
   CHECK(0 == metricReturn);
}

TEST_CASE("Term with one feature with one state, interaction, regression") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(1) });
   test.AddInteractionSamples({ TestSample({ 0 }, 10) });
   test.InitializeInteraction();
   double metricReturn = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("Term with one feature with one state, interaction, binary") {
   TestApi test = TestApi(2, 0);
   test.AddFeatures({ FeatureTest(1) });
   test.AddInteractionSamples({ TestSample({ 0 }, 0) });
   test.InitializeInteraction();
   double metricReturn = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("Term with one feature with one state, interaction, multiclass") {
   TestApi test = TestApi(3);
   test.AddFeatures({ FeatureTest(1) });
   test.AddInteractionSamples({ TestSample({ 0 }, 0) });
   test.InitializeInteraction();
   double metricReturn = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("weights are proportional, interaction, regression") {
   TestApi test1 = TestApi(k_learningTypeRegression);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test1.AddInteractionSamples({ 
      TestSample({ 0, 0 }, 10.1, FloatTickIncrementTest(0.3)),
      TestSample({ 0, 1 }, 20.2, 0.3),
      TestSample({ 1, 0 }, 30.3, 0.3),
      TestSample({ 1, 1 }, 40.4, 0.3),
      });
   test1.InitializeInteraction();
   double metricReturn1 = test1.TestCalcInteractionStrength({ 0, 1 });

   TestApi test2 = TestApi(k_learningTypeRegression);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test2.AddInteractionSamples({
      TestSample({ 0, 0 }, 10.1, FloatTickIncrementTest(2)),
      TestSample({ 0, 1 }, 20.2, 2),
      TestSample({ 1, 0 }, 30.3, 2),
      TestSample({ 1, 1 }, 40.4, 2),
      });
   test2.InitializeInteraction();
   double metricReturn2 = test2.TestCalcInteractionStrength({ 0, 1 });

   TestApi test3 = TestApi(k_learningTypeRegression);
   test3.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test3.AddInteractionSamples({
      TestSample({ 0, 0 }, 10.1, 0),
      TestSample({ 0, 1 }, 20.2, 0),
      TestSample({ 1, 0 }, 30.3, 0),
      TestSample({ 1, 1 }, 40.4, 0),
      });
   test3.InitializeInteraction();
   double metricReturn3 = test3.TestCalcInteractionStrength({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
   CHECK_APPROX(metricReturn1, metricReturn3);
}

TEST_CASE("weights are proportional, interaction, binary") {
   TestApi test1 = TestApi(2);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test1.AddInteractionSamples({
      TestSample({ 0, 0 }, 0, FloatTickIncrementTest(0.3)),
      TestSample({ 0, 1 }, 1, 0.3),
      TestSample({ 1, 0 }, 1, 0.3),
      TestSample({ 1, 1 }, 0, 0.3),
      });
   test1.InitializeInteraction();
   double metricReturn1 = test1.TestCalcInteractionStrength({ 0, 1 });

   TestApi test2 = TestApi(2);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test2.AddInteractionSamples({
      TestSample({ 0, 0 }, 0, FloatTickIncrementTest(2)),
      TestSample({ 0, 1 }, 1, 2),
      TestSample({ 1, 0 }, 1, 2),
      TestSample({ 1, 1 }, 0, 2),
      });
   test2.InitializeInteraction();
   double metricReturn2 = test2.TestCalcInteractionStrength({ 0, 1 });

   TestApi test3 = TestApi(2);
   test3.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test3.AddInteractionSamples({
      TestSample({ 0, 0 }, 0, 0),
      TestSample({ 0, 1 }, 1, 0),
      TestSample({ 1, 0 }, 1, 0),
      TestSample({ 1, 1 }, 0, 0),
      });
   test3.InitializeInteraction();
   double metricReturn3 = test3.TestCalcInteractionStrength({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
   CHECK_APPROX(metricReturn1, metricReturn3);
}

TEST_CASE("weights are proportional, interaction, multiclass") {
   TestApi test1 = TestApi(3);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test1.AddInteractionSamples({
      TestSample({ 0, 0 }, 0, FloatTickIncrementTest(0.3)),
      TestSample({ 0, 1 }, 1, 0.3),
      TestSample({ 1, 0 }, 2, 0.3),
      TestSample({ 1, 1 }, 0, 0.3),
      });
   test1.InitializeInteraction();
   double metricReturn1 = test1.TestCalcInteractionStrength({ 0, 1 });

   TestApi test2 = TestApi(3);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test2.AddInteractionSamples({
      TestSample({ 0, 0 }, 0, FloatTickIncrementTest(2)),
      TestSample({ 0, 1 }, 1, 2),
      TestSample({ 1, 0 }, 2, 2),
      TestSample({ 1, 1 }, 0, 2),
      });
   test2.InitializeInteraction();
   double metricReturn2 = test2.TestCalcInteractionStrength({ 0, 1 });

   TestApi test3 = TestApi(3);
   test3.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test3.AddInteractionSamples({
      TestSample({ 0, 0 }, 0, 0),
      TestSample({ 0, 1 }, 1, 0),
      TestSample({ 1, 0 }, 2, 0),
      TestSample({ 1, 1 }, 0, 0),
      });
   test3.InitializeInteraction();
   double metricReturn3 = test3.TestCalcInteractionStrength({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
   CHECK_APPROX(metricReturn1, metricReturn3);
}

TEST_CASE("weights totals equivalence, interaction, regression") {
   TestApi test1 = TestApi(k_learningTypeRegression);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test1.AddInteractionSamples({
      TestSample({ 0, 0 }, 10.1, 0.15),
      TestSample({ 0, 0 }, 10.1, 0.15),
      TestSample({ 0, 1 }, 20.2, 0.3),
      TestSample({ 1, 0 }, 30.3, 0.3),
      TestSample({ 1, 1 }, 40.4, 0.3),
      });
   test1.InitializeInteraction();
   double metricReturn1 = test1.TestCalcInteractionStrength({ 0, 1 });

   TestApi test2 = TestApi(k_learningTypeRegression);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test2.AddInteractionSamples({
      TestSample({ 0, 0 }, 10.1, 2),
      TestSample({ 0, 1 }, 20.2, 2),
      TestSample({ 1, 0 }, 30.3, 1),
      TestSample({ 1, 0 }, 30.3, 1),
      TestSample({ 1, 1 }, 40.4, 2),
      });
   test2.InitializeInteraction();
   double metricReturn2 = test2.TestCalcInteractionStrength({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
}

TEST_CASE("weights totals equivalence, interaction, binary") {
   TestApi test1 = TestApi(2);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test1.AddInteractionSamples({
      TestSample({ 0, 0 }, 0, 0.3),
      TestSample({ 0, 1 }, 1, 0.15),
      TestSample({ 0, 1 }, 1, 0.15),
      TestSample({ 1, 0 }, 1, 0.3),
      TestSample({ 1, 1 }, 0, 0.3),
      });
   test1.InitializeInteraction();
   double metricReturn1 = test1.TestCalcInteractionStrength({ 0, 1 });

   TestApi test2 = TestApi(2);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test2.AddInteractionSamples({
      TestSample({ 0, 0 }, 0, 2),
      TestSample({ 0, 1 }, 1, 2),
      TestSample({ 1, 0 }, 1, 2),
      TestSample({ 1, 1 }, 0, 1),
      TestSample({ 1, 1 }, 0, 1),
      });
   test2.InitializeInteraction();
   double metricReturn2 = test2.TestCalcInteractionStrength({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
}

TEST_CASE("weights totals equivalence, interaction, multiclass") {
   TestApi test1 = TestApi(3);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test1.AddInteractionSamples({
      TestSample({ 0, 0 }, 0, 0.3),
      TestSample({ 0, 1 }, 1, 0.15),
      TestSample({ 0, 1 }, 1, 0.15),
      TestSample({ 1, 0 }, 2, 0.3),
      TestSample({ 1, 1 }, 0, 0.3),
      });
   test1.InitializeInteraction();
   double metricReturn1 = test1.TestCalcInteractionStrength({ 0, 1 });

   TestApi test2 = TestApi(3);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test2.AddInteractionSamples({
      TestSample({ 0, 0 }, 0, 1),
      TestSample({ 0, 0 }, 0, 1),
      TestSample({ 0, 1 }, 1, 2),
      TestSample({ 1, 0 }, 2, 2),
      TestSample({ 1, 1 }, 0, 2),
      });
   test2.InitializeInteraction();
   double metricReturn2 = test2.TestCalcInteractionStrength({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
}

TEST_CASE("purified interaction strength with impure inputs should be zero, interaction, regression") {
   // impure:
   // feature1 = 3, 5
   // feature2 = 11, 7

   // impure:
   // 3 + 11   3 + 7
   // 5 + 11   5 + 7

   // or:
   // 14  10
   // 16  12

   // we can use any random weights for impure inputs, so stress test this!

   TestApi test1 = TestApi(k_learningTypeRegression);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test1.AddInteractionSamples({
      TestSample({ 0, 0 }, (3.0 + 11.0), 24.25),
      TestSample({ 0, 1 }, (3.0 + 7.0), 21.5),
      TestSample({ 1, 0 }, (5.0 + 11.0), 8.125),
      TestSample({ 1, 1 }, (5.0 + 7.0), 11.625),
      });

   test1.InitializeInteraction();
   double metricReturn = test1.TestCalcInteractionStrength({ 0, 1 }, InteractionFlags_Pure);

   CHECK(0 <= metricReturn && metricReturn < 0.0000001);
}

TEST_CASE("purified interaction strength same as pre-purified strength, interaction, regression") {
   // let us construct a matrix that consists of impure effect and pure effect and compare that to the 
   // interaction strength of the purified matrix.  They should be the same.

   // Start by creating a pure interaction and getting the interaction strength:
   // 
   // counts:
   // 2.5  20
   // 1.25 5
   //
   // pure:
   // -16  2
   //  32 -8

   TestApi test1 = TestApi(k_learningTypeRegression);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test1.AddInteractionSamples({
      TestSample({ 0, 0 }, -16.0, 2.5),
      TestSample({ 0, 1 }, 2.0, 20),
      TestSample({ 1, 0 }, 32.0, 1.25),
      TestSample({ 1, 1 }, -8.0, 5),
      });
   test1.InitializeInteraction();
   double metricReturn1 = test1.TestCalcInteractionStrength({ 0, 1 }, InteractionFlags_Pure);

   // to the pure input we add on one   axis: 3, 5
   // to the pure input we add on other axis: 7, 11
   // these should be purified away leaving only the base pure 
   //
   // impure:
   // 3 + 11   3 + 7
   // 5 + 11   5 + 7

   // or:
   // 14  10
   // 16  12

   TestApi test2 = TestApi(k_learningTypeRegression);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test2.AddInteractionSamples({
      TestSample({ 0, 0 }, -16.0 + (3.0 + 11.0), 2.5),
      TestSample({ 0, 1 }, 2.0 + (3.0 + 7.0), 20),
      TestSample({ 1, 0 }, 32.0 + (5.0 + 11.0), 1.25),
      TestSample({ 1, 1 }, -8.0 + (5.0 + 7.0), 5),
      });
   test2.InitializeInteraction();
   double metricReturn2 = test2.TestCalcInteractionStrength({ 0, 1 }, InteractionFlags_Pure);

   CHECK_APPROX(metricReturn1, metricReturn2);
}

TEST_CASE("compare boosting gain to interaction strength, which should be identical") {
   // we use the same algorithm to calculate interaction strength (gain) and during boosting (gain again)
   // so we would expect them to generate the same response

   TestApi test1 = TestApi(k_learningTypeRegression);
   test1.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test1.AddInteractionSamples({
      TestSample({ 0, 0 }, 3, 232.24),
      TestSample({ 0, 1 }, 11, 12.124),
      TestSample({ 1, 0 }, 5, 85.1254),
      TestSample({ 1, 1 }, 7, 1.355),
      });
   test1.InitializeInteraction();
   const double interactionStrength = test1.TestCalcInteractionStrength({ 0, 1 });

   // we have a 2x2 matrix for boosting, which means there is only 1 cut point and it is known
   // so the gain should be from going from a singularity to the 4 quadrants

   TestApi test2 = TestApi(k_learningTypeRegression);
   test2.AddFeatures({ FeatureTest(2), FeatureTest(2) });
   test2.AddTerms({ { 0, 1 } });
   test2.AddTrainingSamples({
      TestSample({ 0, 0 }, 3, 232.24),
      TestSample({ 0, 1 }, 11, 12.124),
      TestSample({ 1, 0 }, 5, 85.1254),
      TestSample({ 1, 1 }, 7, 1.355),
      });
   test2.AddValidationSamples({});
   test2.InitializeBoosting(0);
   const double gainAvg = test2.Boost(0).gainAvg;

   CHECK_APPROX(interactionStrength, gainAvg);
}
