// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "libebm.h"
#include "libebm_test.hpp"

static constexpr TestPriority k_filePriority = TestPriority::InteractionUnusualInputs;

TEST_CASE("Zero interaction samples, interaction, regression") {
   TestInteraction test = TestInteraction(
      OutputType_Regression,
      { FeatureTest(2) }, 
      {}
   );

   double metricReturn = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("Zero interaction samples, interaction, binary") {
   TestInteraction test = TestInteraction(
      OutputType_BinaryClassification, 
      { FeatureTest(2) },
      {},
      k_testCreateInteractionFlags_Default,
      nullptr,
      0
   );

   double metricReturn = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("Zero interaction samples, interaction, multiclass") {
   TestInteraction test = TestInteraction(
      3, 
      { FeatureTest(2) },
      {}
   );

   double metricReturn = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("classification with 0 possible target states, interaction") {
   TestInteraction test = TestInteraction(
      0,
      { FeatureTest(2) },
      {}
   );

   double validationMetric = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == validationMetric);
}

TEST_CASE("classification with 1 possible target, interaction") {
   TestInteraction test = TestInteraction(
      OutputType_MonoClassification,
      { FeatureTest(2) },
      { 
         TestSample({ 1 }, 0) 
      }
   );

   double validationMetric = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == validationMetric);
}

TEST_CASE("features with 0 states, interaction") {
   TestInteraction test = TestInteraction(
      OutputType_Regression,
      { FeatureTest(2, false, false) },
      {}
   );

   double validationMetric = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == validationMetric);
}

TEST_CASE("Term with zero features, interaction, regression") {
   TestInteraction test = TestInteraction(
      OutputType_Regression, 
      {}, 
      { 
         TestSample({}, 10) 
      }
   );

   double metricReturn = test.TestCalcInteractionStrength({});
   CHECK(0 == metricReturn);
}

TEST_CASE("Term with zero features, interaction, binary") {
   TestInteraction test = TestInteraction(
      OutputType_BinaryClassification, 
      {},
      { TestSample({}, 0) },
      k_testCreateInteractionFlags_Default,
      nullptr,
      0
   );

   double metricReturn = test.TestCalcInteractionStrength({});
   CHECK(0 == metricReturn);
}

TEST_CASE("Term with zero features, interaction, multiclass") {
   TestInteraction test = TestInteraction(
      3,
      {}, 
      { TestSample({}, 0) }
   );

   double metricReturn = test.TestCalcInteractionStrength({});
   CHECK(0 == metricReturn);
}

TEST_CASE("Term with one feature with one state, interaction, regression") {
   TestInteraction test = TestInteraction(
      OutputType_Regression, 
      { FeatureTest(2, true, false) }, 
      { TestSample({ 0 }, 10) }
   );

   double metricReturn = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("Term with one feature with one state, interaction, binary") {
   TestInteraction test = TestInteraction(
      OutputType_BinaryClassification, 
      { FeatureTest(2, true, false) }, 
      { TestSample({ 0 }, 0) },
      k_testCreateInteractionFlags_Default,
      nullptr,
      0
   );

   double metricReturn = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("Term with one feature with one state, interaction, multiclass") {
   TestInteraction test = TestInteraction(
      3,
      { FeatureTest(2, true, false) },
      { 
         TestSample({ 0 }, 0) 
      }
   );

   double metricReturn = test.TestCalcInteractionStrength({ 0 });
   CHECK(0 == metricReturn);
}

TEST_CASE("weights are proportional, interaction, regression") {
   TestInteraction test1 = TestInteraction(
      OutputType_Regression, 
      { FeatureTest(2), FeatureTest(2) },
      { 
         TestSample({ 0, 0 }, 10.1, FloatTickIncrementTest(0.3)),
         TestSample({ 0, 1 }, 20.2, 0.3),
         TestSample({ 1, 0 }, 30.3, 0.3),
         TestSample({ 1, 1 }, 40.4, 0.3),
      }
   );

   double metricReturn1 = test1.TestCalcInteractionStrength({ 0, 1 });

   TestInteraction test2 = TestInteraction(
      OutputType_Regression,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, 10.1, FloatTickIncrementTest(2)),
         TestSample({ 0, 1 }, 20.2, 2),
         TestSample({ 1, 0 }, 30.3, 2),
         TestSample({ 1, 1 }, 40.4, 2),
      }
   );

   double metricReturn2 = test2.TestCalcInteractionStrength({ 0, 1 });

   TestInteraction test3 = TestInteraction(
      OutputType_Regression,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, 10.1, 0.125),
         TestSample({ 0, 1 }, 20.2, 0.125),
         TestSample({ 1, 0 }, 30.3, 0.125),
         TestSample({ 1, 1 }, 40.4, 0.125),
      }
   );

   double metricReturn3 = test3.TestCalcInteractionStrength({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
   CHECK_APPROX(metricReturn1, metricReturn3);
}

TEST_CASE("weights are proportional, interaction, binary") {
   TestInteraction test1 = TestInteraction(
      OutputType_BinaryClassification,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, 0, FloatTickIncrementTest(0.3)),
         TestSample({ 0, 1 }, 1, 0.3),
         TestSample({ 1, 0 }, 1, 0.3),
         TestSample({ 1, 1 }, 0, 0.3),
      }
   );

   double metricReturn1 = test1.TestCalcInteractionStrength({ 0, 1 });

   TestInteraction test2 = TestInteraction(
      OutputType_BinaryClassification,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, 0, FloatTickIncrementTest(2)),
         TestSample({ 0, 1 }, 1, 2),
         TestSample({ 1, 0 }, 1, 2),
         TestSample({ 1, 1 }, 0, 2),
      }
   );

   double metricReturn2 = test2.TestCalcInteractionStrength({ 0, 1 });

   TestInteraction test3 = TestInteraction(
      OutputType_BinaryClassification,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, 0, 0.125),
         TestSample({ 0, 1 }, 1, 0.125),
         TestSample({ 1, 0 }, 1, 0.125),
         TestSample({ 1, 1 }, 0, 0.125),
      }
   );

   double metricReturn3 = test3.TestCalcInteractionStrength({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
   CHECK_APPROX(metricReturn1, metricReturn3);
}

TEST_CASE("weights are proportional, interaction, multiclass") {
   TestInteraction test1 = TestInteraction(
      3,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, 0, FloatTickIncrementTest(0.3)),
         TestSample({ 0, 1 }, 1, 0.3),
         TestSample({ 1, 0 }, 2, 0.3),
         TestSample({ 1, 1 }, 0, 0.3),
      }
   );

   double metricReturn1 = test1.TestCalcInteractionStrength({ 0, 1 });

   TestInteraction test2 = TestInteraction(
      3,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, 0, FloatTickIncrementTest(2)),
         TestSample({ 0, 1 }, 1, 2),
         TestSample({ 1, 0 }, 2, 2),
         TestSample({ 1, 1 }, 0, 2),
      }
   );

   double metricReturn2 = test2.TestCalcInteractionStrength({ 0, 1 });

   TestInteraction test3 = TestInteraction(
      3,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, 0, 0.125),
         TestSample({ 0, 1 }, 1, 0.125),
         TestSample({ 1, 0 }, 2, 0.125),
         TestSample({ 1, 1 }, 0, 0.125),
      }
   );

   double metricReturn3 = test3.TestCalcInteractionStrength({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
   CHECK_APPROX(metricReturn1, metricReturn3);
}

TEST_CASE("weights totals equivalence, interaction, regression") {
   TestInteraction test1 = TestInteraction(
      OutputType_Regression,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, 10.1, 0.15),
         TestSample({ 0, 0 }, 10.1, 0.15),
         TestSample({ 0, 1 }, 20.2, 0.3),
         TestSample({ 1, 0 }, 30.3, 0.3),
         TestSample({ 1, 1 }, 40.4, 0.3),
      }
   );

   double metricReturn1 = test1.TestCalcInteractionStrength({ 0, 1 });

   TestInteraction test2 = TestInteraction(
      OutputType_Regression,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, 10.1, 2),
         TestSample({ 0, 1 }, 20.2, 2),
         TestSample({ 1, 0 }, 30.3, 1),
         TestSample({ 1, 0 }, 30.3, 1),
         TestSample({ 1, 1 }, 40.4, 2),
      }
   );

   double metricReturn2 = test2.TestCalcInteractionStrength({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
}

TEST_CASE("weights totals equivalence, interaction, binary") {
   TestInteraction test1 = TestInteraction(
      OutputType_BinaryClassification,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, 0, 0.3),
         TestSample({ 0, 1 }, 1, 0.15),
         TestSample({ 0, 1 }, 1, 0.15),
         TestSample({ 1, 0 }, 1, 0.3),
         TestSample({ 1, 1 }, 0, 0.3),
      }
   );

   double metricReturn1 = test1.TestCalcInteractionStrength({ 0, 1 });

   TestInteraction test2 = TestInteraction(
      OutputType_BinaryClassification,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, 0, 2),
         TestSample({ 0, 1 }, 1, 2),
         TestSample({ 1, 0 }, 1, 2),
         TestSample({ 1, 1 }, 0, 1),
         TestSample({ 1, 1 }, 0, 1),
      }
   );

   double metricReturn2 = test2.TestCalcInteractionStrength({ 0, 1 });

   CHECK_APPROX(metricReturn1, metricReturn2);
}

TEST_CASE("weights totals equivalence, interaction, multiclass") {
   TestInteraction test1 = TestInteraction(
      3,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, 0, 0.3),
         TestSample({ 0, 1 }, 1, 0.15),
         TestSample({ 0, 1 }, 1, 0.15),
         TestSample({ 1, 0 }, 2, 0.3),
         TestSample({ 1, 1 }, 0, 0.3),
      }
   );

   double metricReturn1 = test1.TestCalcInteractionStrength({ 0, 1 });

   TestInteraction test2 = TestInteraction(
      3,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, 0, 1),
         TestSample({ 0, 0 }, 0, 1),
         TestSample({ 0, 1 }, 1, 2),
         TestSample({ 1, 0 }, 2, 2),
         TestSample({ 1, 1 }, 0, 2),
      }
   );

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

   TestInteraction test1 = TestInteraction(
      OutputType_Regression,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, (3.0 + 11.0), 24.25),
         TestSample({ 0, 1 }, (3.0 + 7.0), 21.5),
         TestSample({ 1, 0 }, (5.0 + 11.0), 8.125),
         TestSample({ 1, 1 }, (5.0 + 7.0), 11.625),
      }
   );

   double metricReturn = test1.TestCalcInteractionStrength({ 0, 1 }, CalcInteractionFlags_Pure);

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

   TestInteraction test1 = TestInteraction(
      OutputType_Regression,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, -16.0, 2.5),
         TestSample({ 0, 1 }, 2.0, 20),
         TestSample({ 1, 0 }, 32.0, 1.25),
         TestSample({ 1, 1 }, -8.0, 5),
      }
   );

   double metricReturn1 = test1.TestCalcInteractionStrength({ 0, 1 }, CalcInteractionFlags_Pure);

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

   TestInteraction test2 = TestInteraction(
      OutputType_Regression,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, -16.0 + (3.0 + 11.0), 2.5),
         TestSample({ 0, 1 }, 2.0 + (3.0 + 7.0), 20),
         TestSample({ 1, 0 }, 32.0 + (5.0 + 11.0), 1.25),
         TestSample({ 1, 1 }, -8.0 + (5.0 + 7.0), 5),
      }
   );

   double metricReturn2 = test2.TestCalcInteractionStrength({ 0, 1 }, CalcInteractionFlags_Pure);

   CHECK_APPROX(metricReturn1, metricReturn2);
}

TEST_CASE("compare boosting gain to interaction strength, which should be identical") {
   // we use the same algorithm to calculate interaction strength (gain) and during boosting (gain again)
   // so we would expect them to generate the same response

   TestInteraction test1 = TestInteraction(
      OutputType_Regression,
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, 3, 232.24),
         TestSample({ 0, 1 }, 11, 12.124),
         TestSample({ 1, 0 }, 5, 85.1254),
         TestSample({ 1, 1 }, 7, 1.355),
      }
   );

   const double interactionStrength = test1.TestCalcInteractionStrength({ 0, 1 }, CalcInteractionFlags_EnableNewton);

   // we have a 2x2 matrix for boosting, which means there is only 1 cut point and it is known
   // so the gain should be from going from a singularity to the 4 quadrants

   TestBoost test2 = TestBoost(
      OutputType_Regression,
      { FeatureTest(2), FeatureTest(2) },
      { { 0, 1 } },
      {
         TestSample({ 0, 0 }, 3, 232.24),
         TestSample({ 0, 1 }, 11, 12.124),
         TestSample({ 1, 0 }, 5, 85.1254),
         TestSample({ 1, 1 }, 7, 1.355),
      },
      {}
   );
   const double gainAvg = test2.Boost(0).gainAvg;

   CHECK_APPROX(interactionStrength, gainAvg);
}

TEST_CASE("tweedie, interaction") {
   TestInteraction test = TestInteraction(
      OutputType_Regression, 
      { FeatureTest(2), FeatureTest(2) },
      {
         TestSample({ 0, 0 }, 10),
         TestSample({ 0, 1 }, 11),
         TestSample({ 1, 0 }, 13),
         TestSample({ 1, 1 }, 12)
      },
      k_testCreateInteractionFlags_Default,
      "tweedie_deviance:variance_power=1.3"
   );

   double metricReturn = test.TestCalcInteractionStrength({ 0, 1 });
   CHECK_APPROX(metricReturn, 1.25);
}

