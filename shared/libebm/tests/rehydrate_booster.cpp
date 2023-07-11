// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "libebm.h"
#include "libebm_test.hpp"

static constexpr TestPriority k_filePriority = TestPriority::Rehydration;

TEST_CASE("Test Rehydration, boosting, regression") {
   TestBoost testContinuous = TestBoost(
      OutputType_Regression,
      {},
      { {} },
      { 
         TestSample({}, 10) 
      },
      { 
         TestSample({}, 12) 
      }
   );

   double termScore0 = 0;

   double validationMetricContinuous;
   double termScoreContinuous;
   double validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestBoost testRestart = TestBoost(
         OutputType_Regression,
         {},
         { {} },
         { 
            TestSample({}, 10, 1, { termScore0 }) 
         },
         { 
            TestSample({}, 12, 1, { termScore0 }) 
         }
      );

      validationMetricRestart = testRestart.Boost(0).validationMetric;
      validationMetricContinuous = testContinuous.Boost(0).validationMetric;
      CHECK_APPROX(validationMetricContinuous, validationMetricRestart);

      termScoreContinuous = testContinuous.GetCurrentTermScore(0, {}, 0);
      termScore0 += testRestart.GetCurrentTermScore(0, {}, 0);
      CHECK_APPROX(termScoreContinuous, termScore0);
   }
}

TEST_CASE("Test Rehydration, boosting, binary") {
   TestBoost testContinuous = TestBoost(
      OutputType_BinaryClassification, 
      {},
      { {} },
      { 
         TestSample({}, 0) 
      },
      { 
         TestSample({}, 0) 
      },
      k_countInnerBagsDefault, 
      k_testCreateBoosterFlags_Default,
      nullptr, 
      0
   );

   double termScore0 = 0;
   double termScore1 = 0;

   double validationMetricContinuous;
   double termScoreContinuous;
   double validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestBoost testRestart = TestBoost(
         OutputType_BinaryClassification, 
         {},
         { {} },
         { 
            TestSample({}, 0, 1, { termScore0, termScore1 }) 
         },
         { 
            TestSample({}, 0, 1, { termScore0, termScore1 }) 
         },
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         nullptr, 
         0
      );

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
   TestBoost testContinuous = TestBoost(
      3,
      {},
      { {} },
      { 
         TestSample({}, 0) 
      }, 
      { 
         TestSample({}, 0) 
      }
   );

   double termScore0 = 0;
   double termScore1 = 0;
   double termScore2 = 0;

   double validationMetricContinuous;
   double termScoreContinuous;
   double validationMetricRestart;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      TestBoost testRestart = TestBoost(
         3,
         {},
         { {} },
         { 
            TestSample({}, 0, 1, { termScore0, termScore1, termScore2 }) 
         },
         { 
            TestSample({}, 0, 1, { termScore0, termScore1, termScore2 }) 
         }
      );

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

