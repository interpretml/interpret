// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderEbmNativeTest.h"

#include "ebm_native.h"
#include "EbmNativeTest.h"

static const TestPriority k_filePriority = TestPriority::Rehydration;

TEST_CASE("Test Rehydration, boosting, regression") {
   TestApi testContinuous = TestApi(k_learningTypeRegression);
   testContinuous.AddFeatures({});
   testContinuous.AddFeatureGroups({ {} });
   testContinuous.AddTrainingSamples({ RegressionSample(10, {}) });
   testContinuous.AddValidationSamples({ RegressionSample(12, {}) });
   testContinuous.InitializeBoosting();

   FloatEbmType model0 = 0;

   FloatEbmType validationMetricContinuous;
   FloatEbmType modelValueContinuous;
   FloatEbmType validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(k_learningTypeRegression);
      testRestart.AddFeatures({});
      testRestart.AddFeatureGroups({ {} });
      testRestart.AddTrainingSamples({ RegressionSample(10, {}, model0, 1) });
      testRestart.AddValidationSamples({ RegressionSample(12, {}, model0, 1) });
      testRestart.InitializeBoosting();

      validationMetricRestart = testRestart.Boost(0);
      validationMetricContinuous = testContinuous.Boost(0);
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
   testContinuous.AddTrainingSamples({ ClassificationSample(0, {}) });
   testContinuous.AddValidationSamples({ ClassificationSample(0, {}) });
   testContinuous.InitializeBoosting();

   FloatEbmType model0 = 0;
   FloatEbmType model1 = 0;

   FloatEbmType validationMetricContinuous;
   FloatEbmType modelValueContinuous;
   FloatEbmType validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(2, 0);
      testRestart.AddFeatures({});
      testRestart.AddFeatureGroups({ {} });
      testRestart.AddTrainingSamples({ ClassificationSample(0, {}, { model0, model1 }, 1) });
      testRestart.AddValidationSamples({ ClassificationSample(0, {}, { model0, model1 }, 1) });
      testRestart.InitializeBoosting();

      validationMetricRestart = testRestart.Boost(0);
      validationMetricContinuous = testContinuous.Boost(0);
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
   testContinuous.AddTrainingSamples({ ClassificationSample(0, {}) });
   testContinuous.AddValidationSamples({ ClassificationSample(0, {}) });
   testContinuous.InitializeBoosting();

   FloatEbmType model0 = 0;
   FloatEbmType model1 = 0;
   FloatEbmType model2 = 0;

   FloatEbmType validationMetricContinuous;
   FloatEbmType modelValueContinuous;
   FloatEbmType validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(3);
      testRestart.AddFeatures({});
      testRestart.AddFeatureGroups({ {} });
      testRestart.AddTrainingSamples({ ClassificationSample(0, {}, { model0, model1, model2 }, 1) });
      testRestart.AddValidationSamples({ ClassificationSample(0, {}, { model0, model1, model2 }, 1) });
      testRestart.InitializeBoosting();

      validationMetricRestart = testRestart.Boost(0);
      validationMetricContinuous = testContinuous.Boost(0);
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

