// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"

static constexpr TestPriority k_filePriority = TestPriority::Rehydration;

TEST_CASE("Test Rehydration, boosting, regression") {
   TestApi testContinuous = TestApi(k_learningTypeRegression);
   testContinuous.AddFeatures({});
   testContinuous.AddTerms({ {} });
   testContinuous.AddTrainingSamples({ TestSample({}, 10) });
   testContinuous.AddValidationSamples({ TestSample({}, 12) });
   testContinuous.InitializeBoosting();

   double termScore0 = 0;

   double validationMetricContinuous;
   double termScoreContinuous;
   double validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(k_learningTypeRegression);
      testRestart.AddFeatures({});
      testRestart.AddTerms({ {} });
      testRestart.AddTrainingSamples({ TestSample({}, 10, 1, { termScore0 }) });
      testRestart.AddValidationSamples({ TestSample({}, 12, 1, { termScore0 }) });
      testRestart.InitializeBoosting();

      validationMetricRestart = testRestart.Boost(0).validationMetric;
      validationMetricContinuous = testContinuous.Boost(0).validationMetric;
      CHECK_APPROX(validationMetricContinuous, validationMetricRestart);

      termScoreContinuous = testContinuous.GetCurrentTermScore(0, {}, 0);
      termScore0 += testRestart.GetCurrentTermScore(0, {}, 0);
      CHECK_APPROX(termScoreContinuous, termScore0);
   }
}

TEST_CASE("Test Rehydration, boosting, binary") {
   TestApi testContinuous = TestApi(2, 0);
   testContinuous.AddFeatures({});
   testContinuous.AddTerms({ {} });
   testContinuous.AddTrainingSamples({ TestSample({}, 0) });
   testContinuous.AddValidationSamples({ TestSample({}, 0) });
   testContinuous.InitializeBoosting();

   double termScore0 = 0;
   double termScore1 = 0;

   double validationMetricContinuous;
   double termScoreContinuous;
   double validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(2, 0);
      testRestart.AddFeatures({});
      testRestart.AddTerms({ {} });
      testRestart.AddTrainingSamples({ TestSample({}, 0, 1, { termScore0, termScore1 }) });
      testRestart.AddValidationSamples({ TestSample({}, 0, 1, { termScore0, termScore1 }) });
      testRestart.InitializeBoosting();

      validationMetricRestart = testRestart.Boost(0).validationMetric;
      validationMetricContinuous = testContinuous.Boost(0).validationMetric;
      CHECK_APPROX(validationMetricContinuous, validationMetricRestart);

      termScoreContinuous = testContinuous.GetCurrentTermScore(0, {}, 0);
      termScore0 += testRestart.GetCurrentTermScore(0, {}, 0);
      CHECK_APPROX(termScoreContinuous, termScore0);

      termScoreContinuous = testContinuous.GetCurrentTermScore(0, {}, 1);
      termScore1 += testRestart.GetCurrentTermScore(0, {}, 1);
      CHECK_APPROX(termScoreContinuous, termScore1);
   }
}

TEST_CASE("Test Rehydration, boosting, multiclass") {
   TestApi testContinuous = TestApi(3);
   testContinuous.AddFeatures({});
   testContinuous.AddTerms({ {} });
   testContinuous.AddTrainingSamples({ TestSample({}, 0) });
   testContinuous.AddValidationSamples({ TestSample({}, 0) });
   testContinuous.InitializeBoosting();

   double termScore0 = 0;
   double termScore1 = 0;
   double termScore2 = 0;

   double validationMetricContinuous;
   double termScoreContinuous;
   double validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestApi testRestart = TestApi(3);
      testRestart.AddFeatures({});
      testRestart.AddTerms({ {} });
      testRestart.AddTrainingSamples({ TestSample({}, 0, 1, { termScore0, termScore1, termScore2 }) });
      testRestart.AddValidationSamples({ TestSample({}, 0, 1, { termScore0, termScore1, termScore2 }) });
      testRestart.InitializeBoosting();

      validationMetricRestart = testRestart.Boost(0).validationMetric;
      validationMetricContinuous = testContinuous.Boost(0).validationMetric;
      CHECK_APPROX(validationMetricContinuous, validationMetricRestart);

      termScoreContinuous = testContinuous.GetCurrentTermScore(0, {}, 0);
      termScore0 += testRestart.GetCurrentTermScore(0, {}, 0);
      CHECK_APPROX(termScoreContinuous, termScore0);

      termScoreContinuous = testContinuous.GetCurrentTermScore(0, {}, 1);
      termScore1 += testRestart.GetCurrentTermScore(0, {}, 1);
      CHECK_APPROX(termScoreContinuous, termScore1);

      termScoreContinuous = testContinuous.GetCurrentTermScore(0, {}, 2);
      termScore2 += testRestart.GetCurrentTermScore(0, {}, 2);
      CHECK_APPROX(termScoreContinuous, termScore2);
   }
}

