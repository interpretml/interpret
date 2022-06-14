// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"

static const TestPriority k_filePriority = TestPriority::Rehydration;

TEST_CASE("Test Rehydration, boosting, regression") {
   TestApi testContinuous = TestApi(k_learningTypeRegression);
   testContinuous.AddFeatures({});
   testContinuous.AddFeatureGroups({ {} });
   testContinuous.AddTrainingSamples({ TestSample({}, 10) });
   testContinuous.AddValidationSamples({ TestSample({}, 12) });
   testContinuous.InitializeBoosting();

   double model0 = 0;

   double validationMetricContinuous;
   double modelValueContinuous;
   double validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(k_learningTypeRegression);
      testRestart.AddFeatures({});
      testRestart.AddFeatureGroups({ {} });
      testRestart.AddTrainingSamples({ TestSample({}, 10, 1, { model0 }) });
      testRestart.AddValidationSamples({ TestSample({}, 12, 1, { model0 }) });
      testRestart.InitializeBoosting();

      validationMetricRestart = testRestart.Boost(0).validationMetric;
      validationMetricContinuous = testContinuous.Boost(0).validationMetric;
      CHECK_APPROX(validationMetricContinuous, validationMetricRestart);

      modelValueContinuous = testContinuous.GetCurrentModelPredictorScore(0, {}, 0);
      model0 += testRestart.GetCurrentModelPredictorScore(0, {}, 0);
      CHECK_APPROX(modelValueContinuous, model0);
   }
}

TEST_CASE("Test Rehydration, boosting, binary") {
   TestApi testContinuous = TestApi(2, 0);
   testContinuous.AddFeatures({});
   testContinuous.AddFeatureGroups({ {} });
   testContinuous.AddTrainingSamples({ TestSample({}, 0) });
   testContinuous.AddValidationSamples({ TestSample({}, 0) });
   testContinuous.InitializeBoosting();

   double model0 = 0;
   double model1 = 0;

   double validationMetricContinuous;
   double modelValueContinuous;
   double validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(2, 0);
      testRestart.AddFeatures({});
      testRestart.AddFeatureGroups({ {} });
      testRestart.AddTrainingSamples({ TestSample({}, 0, 1, { model0, model1 }) });
      testRestart.AddValidationSamples({ TestSample({}, 0, 1, { model0, model1 }) });
      testRestart.InitializeBoosting();

      validationMetricRestart = testRestart.Boost(0).validationMetric;
      validationMetricContinuous = testContinuous.Boost(0).validationMetric;
      CHECK_APPROX(validationMetricContinuous, validationMetricRestart);

      modelValueContinuous = testContinuous.GetCurrentModelPredictorScore(0, {}, 0);
      model0 += testRestart.GetCurrentModelPredictorScore(0, {}, 0);
      CHECK_APPROX(modelValueContinuous, model0);

      modelValueContinuous = testContinuous.GetCurrentModelPredictorScore(0, {}, 1);
      model1 += testRestart.GetCurrentModelPredictorScore(0, {}, 1);
      CHECK_APPROX(modelValueContinuous, model1);
   }
}

TEST_CASE("Test Rehydration, boosting, multiclass") {
   TestApi testContinuous = TestApi(3);
   testContinuous.AddFeatures({});
   testContinuous.AddFeatureGroups({ {} });
   testContinuous.AddTrainingSamples({ TestSample({}, 0) });
   testContinuous.AddValidationSamples({ TestSample({}, 0) });
   testContinuous.InitializeBoosting();

   double model0 = 0;
   double model1 = 0;
   double model2 = 0;

   double validationMetricContinuous;
   double modelValueContinuous;
   double validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(3);
      testRestart.AddFeatures({});
      testRestart.AddFeatureGroups({ {} });
      testRestart.AddTrainingSamples({ TestSample({}, 0, 1, { model0, model1, model2 }) });
      testRestart.AddValidationSamples({ TestSample({}, 0, 1, { model0, model1, model2 }) });
      testRestart.InitializeBoosting();

      validationMetricRestart = testRestart.Boost(0).validationMetric;
      validationMetricContinuous = testContinuous.Boost(0).validationMetric;
      CHECK_APPROX(validationMetricContinuous, validationMetricRestart);

      modelValueContinuous = testContinuous.GetCurrentModelPredictorScore(0, {}, 0);
      model0 += testRestart.GetCurrentModelPredictorScore(0, {}, 0);
      CHECK_APPROX(modelValueContinuous, model0);

      modelValueContinuous = testContinuous.GetCurrentModelPredictorScore(0, {}, 1);
      model1 += testRestart.GetCurrentModelPredictorScore(0, {}, 1);
      CHECK_APPROX(modelValueContinuous, model1);

      modelValueContinuous = testContinuous.GetCurrentModelPredictorScore(0, {}, 2);
      model2 += testRestart.GetCurrentModelPredictorScore(0, {}, 2);
      CHECK_APPROX(modelValueContinuous, model2);
   }
}

