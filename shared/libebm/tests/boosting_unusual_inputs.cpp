// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch_test.hpp"

#include "libebm.h"
#include "libebm_test.hpp"

static constexpr TestPriority k_filePriority = TestPriority::BoostingUnusualInputs;

TEST_CASE("intercept boosting, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression,
         {},
         {},
         {
               TestSample({}, 5),
               TestSample({}, 7),
         },
         {TestSample({}, 6)});

   double validationMetric;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      validationMetric = test.Boost(-1).validationMetric;
   }
   CHECK_APPROX(validationMetric, 6.7095237708845920e-08);
}

TEST_CASE("intercept boosting, boosting, binary") {
   TestBoost test = TestBoost(Task_BinaryClassification,
         {},
         {},
         {
               TestSample({}, 1),
               TestSample({}, 1),
         },
         {TestSample({}, 1)});

   double validationMetric;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      validationMetric = test.Boost(-1).validationMetric;
   }
   CHECK_APPROX(validationMetric, 2.2621439908125978e-05);
}

TEST_CASE("intercept boosting, boosting, multiclass") {
   TestBoost test = TestBoost(3,
         {},
         {},
         {
               TestSample({}, 1),
               TestSample({}, 1),
         },
         {TestSample({}, 1)});

   double validationMetric;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      validationMetric = test.Boost(-1).validationMetric;
   }
   CHECK_APPROX(validationMetric, 1.3530110746819210e-06);
}

TEST_CASE("zero learning rate, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression, {}, {{}}, {TestSample({}, 10)}, {TestSample({}, 12)});

   double validationMetric = double{std::numeric_limits<double>::quiet_NaN()};
   double termScore = double{std::numeric_limits<double>::quiet_NaN()};
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm, TermBoostFlags_Default, 0).validationMetric;
         CHECK(144 == validationMetric);
         termScore = test.GetCurrentTermScore(iTerm, {}, 0);
         CHECK(0 == termScore);

         termScore = test.GetBestTermScore(iTerm, {}, 0);
         CHECK(0 == termScore);
      }
   }
}

TEST_CASE("zero learning rate, boosting, binary") {
   TestBoost test = TestBoost(Task_BinaryClassification,
         {},
         {{}},
         {TestSample({}, 0)},
         {TestSample({}, 0)},
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         nullptr,
         0);

   double validationMetric = double{std::numeric_limits<double>::quiet_NaN()};
   double termScore = double{std::numeric_limits<double>::quiet_NaN()};
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm, TermBoostFlags_Default, 0).validationMetric;
         CHECK_APPROX_TOLERANCE(validationMetric, 0.69314718055994529, double{1e-1});
         termScore = test.GetCurrentTermScore(iTerm, {}, 0);
         CHECK(0 == termScore);
         termScore = test.GetCurrentTermScore(iTerm, {}, 1);
         CHECK(0 == termScore);

         termScore = test.GetBestTermScore(iTerm, {}, 0);
         CHECK(0 == termScore);
         termScore = test.GetBestTermScore(iTerm, {}, 1);
         CHECK(0 == termScore);
      }
   }
}

TEST_CASE("zero learning rate, boosting, multiclass") {
   TestBoost test = TestBoost(3, {}, {{}}, {TestSample({}, 0)}, {TestSample({}, 0)});

   double validationMetric = double{std::numeric_limits<double>::quiet_NaN()};
   double termScore = double{std::numeric_limits<double>::quiet_NaN()};
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm, TermBoostFlags_Default, 0).validationMetric;
         CHECK_APPROX_TOLERANCE(validationMetric, 1.0986122886681098, double{1e-1});
         termScore = test.GetCurrentTermScore(iTerm, {}, 0);
         CHECK(0 == termScore);
         termScore = test.GetCurrentTermScore(iTerm, {}, 1);
         CHECK(0 == termScore);
         termScore = test.GetCurrentTermScore(iTerm, {}, 2);
         CHECK(0 == termScore);

         termScore = test.GetBestTermScore(iTerm, {}, 0);
         CHECK(0 == termScore);
         termScore = test.GetBestTermScore(iTerm, {}, 1);
         CHECK(0 == termScore);
         termScore = test.GetBestTermScore(iTerm, {}, 2);
         CHECK(0 == termScore);
      }
   }
}

TEST_CASE("negative learning rate, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression, {}, {{}}, {TestSample({}, 10)}, {TestSample({}, 12)});

   double validationMetric = double{std::numeric_limits<double>::quiet_NaN()};
   double termScore = double{std::numeric_limits<double>::quiet_NaN()};
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm, TermBoostFlags_Default, -k_learningRateDefault).validationMetric;
         if(0 == iTerm && 0 == iEpoch) {
            CHECK_APPROX(validationMetric, 146.41);
            termScore = test.GetCurrentTermScore(iTerm, {}, 0);
            CHECK_APPROX(termScore, -0.1);
         }
         if(0 == iTerm && 1 == iEpoch) {
            CHECK_APPROX(validationMetric, 148.864401);
            termScore = test.GetCurrentTermScore(iTerm, {}, 0);
            CHECK_APPROX(termScore, -0.201);
         }
      }
   }
   CHECK_APPROX(validationMetric, 43929458875.235196700295656826033);
   termScore = test.GetCurrentTermScore(0, {}, 0);
   CHECK_APPROX(termScore, -209581.55637813677);
}

TEST_CASE("negative learning rate, boosting, binary") {
   TestBoost test = TestBoost(Task_BinaryClassification,
         {},
         {{}},
         {TestSample({}, 0)},
         {TestSample({}, 0)},
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         nullptr,
         0);

   double validationMetric = double{std::numeric_limits<double>::quiet_NaN()};
   double termScore = double{std::numeric_limits<double>::quiet_NaN()};
   for(int iEpoch = 0; iEpoch < 50; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm, TermBoostFlags_Default, -k_learningRateDefault).validationMetric;
         if(0 == iTerm && 0 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 0.70319717972663420, double{1e-1});
            termScore = test.GetCurrentTermScore(iTerm, {}, 0);
            CHECK(0 == termScore);
            termScore = test.GetCurrentTermScore(iTerm, {}, 1);
            CHECK_APPROX_TOLERANCE(termScore, 0.02, double{1.5e-1});
         }
         if(0 == iTerm && 1 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 0.71345019889199235, double{1e-1});
            termScore = test.GetCurrentTermScore(iTerm, {}, 0);
            CHECK(0 == termScore);
            termScore = test.GetCurrentTermScore(iTerm, {}, 1);
            CHECK_APPROX_TOLERANCE(termScore, 0.040202013400267564, double{1.5e-1});
         }
      }
   }

   CHECK_APPROX_TOLERANCE(validationMetric, 1.7158914513238979, double{1e-1});
   termScore = test.GetCurrentTermScore(0, {}, 0);
   CHECK(0 == termScore);
   termScore = test.GetCurrentTermScore(0, {}, 1);
   CHECK_APPROX_TOLERANCE(termScore, 1.5176802847035755, double{1e-2});
}

TEST_CASE("negative learning rate, boosting, multiclass") {
   TestBoost test = TestBoost(3, {}, {{}}, {TestSample({}, 0)}, {TestSample({}, 0)});

   double validationMetric = double{std::numeric_limits<double>::quiet_NaN()};
   double termScore = double{std::numeric_limits<double>::quiet_NaN()};
   for(int iEpoch = 0; iEpoch < 20; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm, TermBoostFlags_Default, -k_learningRateDefault).validationMetric;
         if(0 == iTerm && 0 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 1.1288361512023379, double{1e-1});
            const double zeroLogit = test.GetCurrentTermScore(iTerm, {}, 0);
            termScore = test.GetCurrentTermScore(iTerm, {}, 1) - zeroLogit;
            CHECK_APPROX(termScore, 0.03);
            termScore = test.GetCurrentTermScore(iTerm, {}, 2) - zeroLogit;
            CHECK_APPROX(termScore, 0.03);
         }
         if(0 == iTerm && 1 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 1.1602122411839852, double{1e-1});
            const double zeroLogit = test.GetCurrentTermScore(iTerm, {}, 0);
            termScore = test.GetCurrentTermScore(iTerm, {}, 1) - zeroLogit;
            CHECK_APPROX_TOLERANCE(termScore, 0.060338810750421350, double{1e-2});
            termScore = test.GetCurrentTermScore(iTerm, {}, 2) - zeroLogit;
            CHECK_APPROX_TOLERANCE(termScore, 0.060338810750421350, double{1e-2});
         }
      }
   }
   CHECK_APPROX_TOLERANCE(validationMetric, 1.5713119506835938, double{1e-1});
   const double zeroLogit1 = test.GetCurrentTermScore(0, {}, 0);
   termScore = test.GetCurrentTermScore(0, {}, 1) - zeroLogit1;
   CHECK_APPROX_TOLERANCE(termScore, 0.70734588332837767, double{1e-1});
   termScore = test.GetCurrentTermScore(0, {}, 2) - zeroLogit1;
   CHECK_APPROX_TOLERANCE(termScore, 0.70734588332837767, double{1e-1});
}

TEST_CASE("zero minSamplesLeaf, boosting, regression") {
   // TODO : call test.Boost many more times in a loop, and verify the output remains the same as previous runs
   // TODO : add classification binary and multiclass versions of this

   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 10),
               TestSample({1}, 10),
         },
         {TestSample({1}, 12)});

   double validationMetric = test.Boost(0, TermBoostFlags_Default, k_learningRateDefault, 0).validationMetric;
   CHECK_APPROX(validationMetric, 141.61);
   double termScore;
   termScore = test.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore, 0.1);
   CHECK_APPROX(termScore, test.GetCurrentTermScore(0, {1}, 0));
}

TEST_CASE("leave one potential cut uncut, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(3)},
         {{0}},
         {
               TestSample({0}, 10),
               TestSample({1}, 11),
               TestSample({2}, 12),
         },
         {TestSample({1}, 12)});

   double validationMetric = test.Boost(0,
                                       TermBoostFlags_MissingLow,
                                       k_learningRateDefault,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       k_minCategorySamplesDefault,
                                       k_categoricalSmoothingDefault,
                                       k_maxCategoricalThresholdDefault,
                                       k_categoricalInclusionPercentDefault,
                                       {2})
                                   .validationMetric;
   CHECK_APPROX(validationMetric, 141.25322499999999);
   double termScore;
   termScore = test.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore, 0.1);
   termScore = test.GetCurrentTermScore(0, {1}, 0);
   CHECK_APPROX(termScore, 0.115);
}

TEST_CASE("weights are proportional, boosting, regression") {
   TestBoost test1 = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 10, FloatTickIncrementTest(0.3)),
               TestSample({1}, 10, 0.3),
         },
         {TestSample({1}, 12, FloatTickIncrementTest(0.3)), TestSample({1}, 12, 0.3)});

   double validationMetric1 = test1.Boost(0).validationMetric;
   double termScore1;
   termScore1 = test1.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore1, test1.GetCurrentTermScore(0, {1}, 0));

   TestBoost test2 = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 10, FloatTickIncrementTest(2)),
               TestSample({1}, 10, 2),
         },
         {TestSample({1}, 12, FloatTickIncrementTest(2)), TestSample({1}, 12, 2)});

   double validationMetric2 = test2.Boost(0).validationMetric;
   double termScore2;
   termScore2 = test2.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore2, test2.GetCurrentTermScore(0, {1}, 0));

   TestBoost test3 = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 10, 0.125),
               TestSample({1}, 10, 0.125),
         },
         {TestSample({1}, 12, 0.125), TestSample({1}, 12, 0.125)});

   double validationMetric3 = test3.Boost(0).validationMetric;
   double termScore3;
   termScore3 = test3.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore3, test3.GetCurrentTermScore(0, {1}, 0));

   CHECK_APPROX(validationMetric1, validationMetric2);
   CHECK_APPROX(validationMetric1, validationMetric3);
   CHECK_APPROX(termScore1, termScore2);
   CHECK_APPROX(termScore1, termScore3);
}

TEST_CASE("weights are proportional, boosting, binary") {
   TestBoost test1 = TestBoost(Task_BinaryClassification,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 0, FloatTickIncrementTest(0.3)),
               TestSample({1}, 0, 0.3),
         },
         {TestSample({1}, 0, FloatTickIncrementTest(0.3)), TestSample({1}, 0, 0.3)});

   double validationMetric1 = test1.Boost(0).validationMetric;
   double termScore1;
   termScore1 = test1.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore1, test1.GetCurrentTermScore(0, {1}, 0));

   TestBoost test2 = TestBoost(Task_BinaryClassification,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 1, FloatTickIncrementTest(2)),
               TestSample({1}, 1, 2),
         },
         {TestSample({1}, 1, FloatTickIncrementTest(2)), TestSample({1}, 1, 2)});

   double validationMetric2 = test2.Boost(0).validationMetric;
   double termScore2;
   termScore2 = test2.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore2, test2.GetCurrentTermScore(0, {1}, 0));

   TestBoost test3 = TestBoost(Task_BinaryClassification,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 0, 0.125),
               TestSample({1}, 0, 0.125),
         },
         {TestSample({1}, 0, 0.125), TestSample({1}, 0, 0.125)});

   double validationMetric3 = test3.Boost(0).validationMetric;
   double termScore3;
   termScore3 = test3.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore3, test3.GetCurrentTermScore(0, {1}, 0));

   CHECK_APPROX(validationMetric1, validationMetric2);
   CHECK_APPROX(validationMetric1, validationMetric3);
   CHECK_APPROX(termScore1, termScore2);
   CHECK_APPROX(termScore1, termScore3);
}

TEST_CASE("weights are proportional, boosting, multiclass") {
   TestBoost test1 = TestBoost(3,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 0, FloatTickIncrementTest(0.3)),
               TestSample({1}, 0, 0.3),
         },
         {TestSample({1}, 0, FloatTickIncrementTest(0.3)), TestSample({1}, 0, 0.3)});

   double validationMetric1 = test1.Boost(0).validationMetric;
   double termScore1;
   termScore1 = test1.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore1, test1.GetCurrentTermScore(0, {1}, 0));

   TestBoost test2 = TestBoost(3,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 1, FloatTickIncrementTest(2)),
               TestSample({1}, 1, 2),
         },
         {TestSample({1}, 1, FloatTickIncrementTest(2)), TestSample({1}, 1, 2)});

   double validationMetric2 = test2.Boost(0).validationMetric;
   double termScore2;
   termScore2 = test2.GetCurrentTermScore(0, {0}, 1);
   CHECK_APPROX(termScore2, test2.GetCurrentTermScore(0, {1}, 1));

   TestBoost test3 = TestBoost(3,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 2, 0.125),
               TestSample({1}, 2, 0.125),
         },
         {TestSample({1}, 2, 0.125), TestSample({1}, 2, 0.125)});

   double validationMetric3 = test3.Boost(0).validationMetric;
   double termScore3;
   termScore3 = test3.GetCurrentTermScore(0, {0}, 2);
   CHECK_APPROX(termScore3, test3.GetCurrentTermScore(0, {1}, 2));

   CHECK_APPROX(validationMetric1, validationMetric2);
   CHECK_APPROX(validationMetric1, validationMetric3);
   CHECK_APPROX(termScore1, termScore2);
   CHECK_APPROX(termScore1, termScore3);
}

TEST_CASE("weights totals equivalence, boosting, regression") {
   TestBoost test1 = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 10, 0.15),
               TestSample({0}, 10, 0.15),
               TestSample({1}, 12, 1.2),
         },
         {TestSample({0}, 10, 0.6), TestSample({1}, 12, 0.3)});

   double validationMetric1 = test1.Boost(0).validationMetric;
   double termScore1;
   termScore1 = test1.GetCurrentTermScore(0, {0}, 0);

   TestBoost test2 = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 10, 0.5),
               TestSample({1}, 12, 2),
         },
         {TestSample({0}, 10, 1), TestSample({0}, 10, 1), TestSample({1}, 12, 1)});

   double validationMetric2 = test2.Boost(0).validationMetric;
   double termScore2;
   termScore2 = test2.GetCurrentTermScore(0, {0}, 0);

   CHECK_APPROX(validationMetric1, validationMetric2);
   CHECK_APPROX(termScore1, termScore2);
}

TEST_CASE("weights totals equivalence, boosting, binary") {
   TestBoost test1 = TestBoost(Task_BinaryClassification,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 0, 0.15),
               TestSample({0}, 0, 0.15),
               TestSample({1}, 1, 1.2),
         },
         {TestSample({0}, 0, 0.6), TestSample({1}, 1, 0.3)});

   double validationMetric1 = test1.Boost(0).validationMetric;
   double termScore1;
   termScore1 = test1.GetCurrentTermScore(0, {0}, 1);

   TestBoost test2 = TestBoost(Task_BinaryClassification,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 0, 0.5),
               TestSample({1}, 1, 2),
         },
         {TestSample({0}, 0, 1), TestSample({0}, 0, 1), TestSample({1}, 1, 1)});

   double validationMetric2 = test2.Boost(0).validationMetric;
   double termScore2;
   termScore2 = test2.GetCurrentTermScore(0, {0}, 1);

   CHECK_APPROX(validationMetric1, validationMetric2);
   CHECK_APPROX(termScore1, termScore2);
}

TEST_CASE("weights totals equivalence, boosting, multiclass") {
   TestBoost test1 = TestBoost(3,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 0, 0.15),
               TestSample({0}, 0, 0.15),
               TestSample({1}, 2, 1.2),
         },
         {TestSample({0}, 0, 0.6), TestSample({1}, 2, 0.3)});

   double validationMetric1 = test1.Boost(0).validationMetric;
   double termScore1;
   termScore1 = test1.GetCurrentTermScore(0, {0}, 1);

   TestBoost test2 = TestBoost(3,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 0, 0.5),
               TestSample({1}, 2, 2),
         },
         {TestSample({0}, 0, 1), TestSample({0}, 0, 1), TestSample({1}, 2, 1)});

   double validationMetric2 = test2.Boost(0).validationMetric;
   double termScore2;
   termScore2 = test2.GetCurrentTermScore(0, {0}, 1);

   CHECK_APPROX(validationMetric1, validationMetric2);
   CHECK_APPROX(termScore1, termScore2);
}

TEST_CASE("one leavesMax, boosting, regression") {
   // TODO : add classification binary and multiclass versions of this

   static const std::vector<IntEbm> k_leavesMax = {IntEbm{1}};

   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 10),
               TestSample({1}, 10),
         },
         {TestSample({1}, 12)});

   double validationMetric = test.Boost(0,
                                       TermBoostFlags_Default,
                                       k_learningRateDefault,
                                       k_minSamplesLeafDefault,
                                       k_minHessianDefault,
                                       k_regAlphaDefault,
                                       k_regLambdaDefault,
                                       k_maxDeltaStepDefault,
                                       k_minCategorySamplesDefault,
                                       k_categoricalSmoothingDefault,
                                       k_maxCategoricalThresholdDefault,
                                       k_categoricalInclusionPercentDefault,
                                       k_leavesMax)
                                   .validationMetric;
   CHECK_APPROX(validationMetric, 141.61);
   double termScore;
   termScore = test.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore, 0.1);
   CHECK_APPROX(termScore, test.GetCurrentTermScore(0, {1}, 0));
}

TEST_CASE("mono-classification") {
   TestBoost test = TestBoost(Task_MonoClassification,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 0),
               TestSample({1}, 0),
         },
         {
               TestSample({1}, 0),
               TestSample({0}, 0),
         });

   ErrorEbm error;

   double avgGain;
   error = GenerateTermUpdate(nullptr,
         test.GetBoosterHandle(),
         0,
         TermBoostFlags_Default,
         k_learningRateDefault,
         k_minSamplesLeafDefault,
         k_minHessianDefault,
         k_regAlphaDefault,
         k_regLambdaDefault,
         k_maxDeltaStepDefault,
         k_minCategorySamplesDefault,
         k_categoricalSmoothingDefault,
         k_maxCategoricalThresholdDefault,
         k_categoricalInclusionPercentDefault,
         &k_leavesMaxDefault[0],
         nullptr,
         &avgGain);
   CHECK(Error_None == error);
   CHECK(0 == avgGain);

   IntEbm countSplits = 1; // since we have 2 bins we could have 1 split (except we don't)
   error = GetTermUpdateSplits(test.GetBoosterHandle(), 0, &countSplits, nullptr);
   CHECK(Error_None == error);
   CHECK(0 == countSplits);

   error = GetTermUpdate(test.GetBoosterHandle(), nullptr);
   CHECK(Error_None == error);

   error = SetTermUpdate(test.GetBoosterHandle(), 0, nullptr);
   CHECK(Error_None == error);

   double avgValidationMetric;
   error = ApplyTermUpdate(test.GetBoosterHandle(), &avgValidationMetric);
   CHECK(Error_None == error);
   CHECK(std::numeric_limits<double>::infinity() == avgValidationMetric);

   error = GetBestTermScores(test.GetBoosterHandle(), 0, nullptr);
   CHECK(Error_None == error);

   error = GetCurrentTermScores(test.GetBoosterHandle(), 0, nullptr);
   CHECK(Error_None == error);
}

TEST_CASE("Zero training samples, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression, {FeatureTest(2)}, {{0}}, {}, {TestSample({1}, 12)});

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      double validationMetric = test.Boost(0).validationMetric;
      CHECK_APPROX(validationMetric, 144);
      double termScore;
      termScore = test.GetCurrentTermScore(0, {0}, 0);
      CHECK(0 == termScore);
      CHECK_APPROX(termScore, test.GetCurrentTermScore(0, {1}, 0));
   }
}

TEST_CASE("Zero training samples, boosting, binary") {
   TestBoost test = TestBoost(Task_BinaryClassification,
         {FeatureTest(2)},
         {{0}},
         {},
         {TestSample({1}, 0)},
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         nullptr,
         0);

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      double validationMetric = test.Boost(0).validationMetric;
      CHECK_APPROX_TOLERANCE(validationMetric, 0.69314718055994529, double{1e-1});
      double termScore;
      termScore = test.GetCurrentTermScore(0, {0}, 0);
      CHECK(0 == termScore);
      CHECK_APPROX(termScore, test.GetCurrentTermScore(0, {1}, 0));

      termScore = test.GetCurrentTermScore(0, {0}, 1);
      CHECK(0 == termScore);
      CHECK_APPROX(termScore, test.GetCurrentTermScore(0, {1}, 1));
   }
}

TEST_CASE("Zero training samples, boosting, multiclass") {
   TestBoost test = TestBoost(3, {FeatureTest(2)}, {{0}}, {}, {TestSample({1}, 0)});

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      double validationMetric = test.Boost(0).validationMetric;
      CHECK_APPROX_TOLERANCE(validationMetric, 1.0986122886681098, double{1e-1});
      double termScore;

      termScore = test.GetCurrentTermScore(0, {0}, 0);
      CHECK(0 == termScore);
      CHECK_APPROX(termScore, test.GetCurrentTermScore(0, {1}, 0));
      termScore = test.GetCurrentTermScore(0, {0}, 1);
      CHECK(0 == termScore);
      CHECK_APPROX(termScore, test.GetCurrentTermScore(0, {1}, 1));
      termScore = test.GetCurrentTermScore(0, {0}, 2);
      CHECK(0 == termScore);
      CHECK_APPROX(termScore, test.GetCurrentTermScore(0, {1}, 2));
   }
}

TEST_CASE("Zero validation samples, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression, {FeatureTest(2)}, {{0}}, {TestSample({1}, 10)}, {});

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      double validationMetric = test.Boost(0).validationMetric;
      CHECK(0 == validationMetric);
      // the current term will continue to update, even though we have no way of evaluating it
      double termScore;
      termScore = test.GetCurrentTermScore(0, {0}, 0);
      if(0 == iEpoch) {
         CHECK_APPROX(termScore, 0.1);
      }
      if(1 == iEpoch) {
         CHECK_APPROX(termScore, 0.199);
      }
      CHECK_APPROX(termScore, test.GetCurrentTermScore(0, {1}, 0));

      CHECK_APPROX(test.GetCurrentTermScore(0, {0}, 0), test.GetBestTermScore(0, {0}, 0));
      CHECK_APPROX(test.GetCurrentTermScore(0, {1}, 0), test.GetBestTermScore(0, {1}, 0));
   }
}

TEST_CASE("Zero validation samples, boosting, binary") {
   TestBoost test = TestBoost(Task_BinaryClassification,
         {FeatureTest(2)},
         {{0}},
         {TestSample({1}, 0)},
         {},
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         nullptr,
         0);

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      double validationMetric = test.Boost(0).validationMetric;
      CHECK(0 == validationMetric);
      // the current term will continue to update, even though we have no way of evaluating it
      double termScore;

      termScore = test.GetCurrentTermScore(0, {0}, 0);
      CHECK(0 == termScore);
      CHECK_APPROX(termScore, test.GetCurrentTermScore(0, {1}, 0));

      termScore = test.GetCurrentTermScore(0, {0}, 1);
      if(0 == iEpoch) {
         CHECK_APPROX_TOLERANCE(termScore, -0.02, double{1.5e-1});
      }
      if(1 == iEpoch) {
         CHECK_APPROX_TOLERANCE(termScore, -0.039801986733067563, double{1e-1});
      }
      CHECK_APPROX(termScore, test.GetCurrentTermScore(0, {1}, 1));

      CHECK_APPROX(test.GetCurrentTermScore(0, {0}, 0), test.GetBestTermScore(0, {0}, 0));
      CHECK_APPROX(test.GetCurrentTermScore(0, {1}, 0), test.GetBestTermScore(0, {1}, 0));

      CHECK_APPROX(test.GetCurrentTermScore(0, {0}, 1), test.GetBestTermScore(0, {0}, 1));
      CHECK_APPROX(test.GetCurrentTermScore(0, {1}, 1), test.GetBestTermScore(0, {1}, 1));
   }
}

TEST_CASE("Zero validation samples, boosting, multiclass") {
   TestBoost test = TestBoost(3, {FeatureTest(2)}, {{0}}, {TestSample({1}, 0)}, {});

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      double validationMetric = test.Boost(0).validationMetric;
      CHECK(0 == validationMetric);
      // the current term will continue to update, even though we have no way of evaluating it
      double termScore;
      if(0 == iEpoch) {
         const double zeroLogit = test.GetCurrentTermScore(0, {0}, 0);
         CHECK_APPROX(zeroLogit, test.GetCurrentTermScore(0, {1}, 0));
         termScore = test.GetCurrentTermScore(0, {0}, 1) - zeroLogit;
         CHECK_APPROX(termScore, -0.03);
         CHECK_APPROX(termScore, test.GetCurrentTermScore(0, {1}, 1) - zeroLogit);
         termScore = test.GetCurrentTermScore(0, {0}, 2) - zeroLogit;
         CHECK_APPROX(termScore, -0.03);
         CHECK_APPROX(termScore, test.GetCurrentTermScore(0, {1}, 2) - zeroLogit);
      }
      if(1 == iEpoch) {
         const double zeroLogit = test.GetCurrentTermScore(0, {0}, 0);
         CHECK_APPROX(zeroLogit, test.GetCurrentTermScore(0, {1}, 0));
         termScore = test.GetCurrentTermScore(0, {0}, 1) - zeroLogit;
         CHECK_APPROX_TOLERANCE(termScore, -0.059670291487725124, double{1e-2});
         CHECK_APPROX(termScore, test.GetCurrentTermScore(0, {1}, 1) - zeroLogit);
         termScore = test.GetCurrentTermScore(0, {0}, 2) - zeroLogit;
         CHECK_APPROX_TOLERANCE(termScore, -0.059670291487725124, double{1e-2});
         CHECK_APPROX(termScore, test.GetCurrentTermScore(0, {1}, 2) - zeroLogit);
      }
      CHECK_APPROX(test.GetCurrentTermScore(0, {0}, 0), test.GetBestTermScore(0, {0}, 0));
      CHECK_APPROX(test.GetCurrentTermScore(0, {0}, 1), test.GetBestTermScore(0, {0}, 1));
      CHECK_APPROX(test.GetCurrentTermScore(0, {0}, 2), test.GetBestTermScore(0, {0}, 2));
      CHECK_APPROX(test.GetCurrentTermScore(0, {1}, 0), test.GetBestTermScore(0, {1}, 0));
      CHECK_APPROX(test.GetCurrentTermScore(0, {1}, 1), test.GetBestTermScore(0, {1}, 1));
      CHECK_APPROX(test.GetCurrentTermScore(0, {1}, 2), test.GetBestTermScore(0, {1}, 2));
   }
}

TEST_CASE("features with 0 states, boosting") {
   // for there to be zero states, there can't be an training data or testing data since then those would be required to
   // have a value for the state
   TestBoost test = TestBoost(Task_Regression, {FeatureTest(2, false, false)}, {{0}}, {}, {});

   double validationMetric = test.Boost(0).validationMetric;
   CHECK(std::numeric_limits<double>::infinity() == validationMetric);

   double termScores[1];

   // we're not sure what we'd get back since we aren't allowed to access it, so don't do anything with the return
   // value.  We just want to make sure calling to get the term doesn't crash
   termScores[0] = 9.99;
   test.GetBestTermScoresRaw(0, termScores);
   CHECK(9.99 == termScores[0]); // the term is a tensor with zero values since one of the dimensions is non-existant
   termScores[0] = 9.99;
   test.GetCurrentTermScoresRaw(0, termScores);
   CHECK(9.99 == termScores[0]); // the term is a tensor with zero values since one of the dimensions is non-existant
}

TEST_CASE("features with 1 state in various positions, boosting") {
   TestBoost test0 = TestBoost(Task_Regression,
         {FeatureTest(2, true, false), FeatureTest(2), FeatureTest(2)},
         {{0}, {1}, {2}},
         {TestSample({0, 1, 1}, 10)},
         {TestSample({0, 1, 1}, 12)});

   TestBoost test1 = TestBoost(Task_Regression,
         {FeatureTest(2), FeatureTest(2, true, false), FeatureTest(2)},
         {{0}, {1}, {2}},
         {TestSample({1, 0, 1}, 10)},
         {TestSample({1, 0, 1}, 12)});

   TestBoost test2 = TestBoost(Task_Regression,
         {FeatureTest(2), FeatureTest(2), FeatureTest(2, true, false)},
         {{0}, {1}, {2}},
         {TestSample({1, 1, 0}, 10)},
         {TestSample({1, 1, 0}, 12)});

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      double validationMetric00 = test0.Boost(0).validationMetric;
      double validationMetric10 = test1.Boost(1).validationMetric;
      CHECK_APPROX(validationMetric00, validationMetric10);
      double validationMetric20 = test2.Boost(2).validationMetric;
      CHECK_APPROX(validationMetric00, validationMetric20);

      double validationMetric01 = test0.Boost(1).validationMetric;
      double validationMetric11 = test1.Boost(2).validationMetric;
      CHECK_APPROX(validationMetric01, validationMetric11);
      double validationMetric21 = test2.Boost(0).validationMetric;
      CHECK_APPROX(validationMetric01, validationMetric21);

      double validationMetric02 = test0.Boost(2).validationMetric;
      double validationMetric12 = test1.Boost(0).validationMetric;
      CHECK_APPROX(validationMetric02, validationMetric12);
      double validationMetric22 = test2.Boost(1).validationMetric;
      CHECK_APPROX(validationMetric02, validationMetric22);

      double termScore000 = test0.GetCurrentTermScore(0, {0}, 0);
      double termScore010 = test0.GetCurrentTermScore(1, {0}, 0);
      double termScore011 = test0.GetCurrentTermScore(1, {1}, 0);
      double termScore020 = test0.GetCurrentTermScore(2, {0}, 0);
      double termScore021 = test0.GetCurrentTermScore(2, {1}, 0);

      double termScore110 = test1.GetCurrentTermScore(1, {0}, 0);
      double termScore120 = test1.GetCurrentTermScore(2, {0}, 0);
      double termScore121 = test1.GetCurrentTermScore(2, {1}, 0);
      double termScore100 = test1.GetCurrentTermScore(0, {0}, 0);
      double termScore101 = test1.GetCurrentTermScore(0, {1}, 0);
      CHECK_APPROX(termScore110, termScore000);
      CHECK_APPROX(termScore120, termScore010);
      CHECK_APPROX(termScore121, termScore011);
      CHECK_APPROX(termScore100, termScore020);
      CHECK_APPROX(termScore101, termScore021);

      double termScore220 = test2.GetCurrentTermScore(2, {0}, 0);
      double termScore200 = test2.GetCurrentTermScore(0, {0}, 0);
      double termScore201 = test2.GetCurrentTermScore(0, {1}, 0);
      double termScore210 = test2.GetCurrentTermScore(1, {0}, 0);
      double termScore211 = test2.GetCurrentTermScore(1, {1}, 0);
      CHECK_APPROX(termScore220, termScore000);
      CHECK_APPROX(termScore200, termScore010);
      CHECK_APPROX(termScore201, termScore011);
      CHECK_APPROX(termScore210, termScore020);
      CHECK_APPROX(termScore211, termScore021);
   }
}

TEST_CASE("zero terms, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression, {}, {}, {TestSample({}, 10)}, {TestSample({}, 12)});

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just
                           // crash/assert if there was a problem
   // boosting isn't legal since we'd need to specify a term index
}

TEST_CASE("zero terms, boosting, binary") {
   TestBoost test = TestBoost(Task_BinaryClassification,
         {},
         {},
         {TestSample({}, 1)},
         {TestSample({}, 1)},
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         nullptr,
         0);

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just
                           // crash/assert if there was a problem
   // boosting isn't legal since we'd need to specify a term index
}

TEST_CASE("zero terms, boosting, multiclass") {
   TestBoost test = TestBoost(3, {}, {}, {TestSample({}, 2)}, {TestSample({}, 2)});

   UNUSED(testCaseHidden); // this is a hidden parameter from TEST_CASE, but we don't test anything here.. we would just
                           // crash/assert if there was a problem
   // boosting isn't legal since we'd need to specify a term index
}

TEST_CASE("Term with zero features, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression, {}, {{}}, {TestSample({}, 10)}, {TestSample({}, 12)});

   double validationMetric = double{std::numeric_limits<double>::quiet_NaN()};
   double termScore = double{std::numeric_limits<double>::quiet_NaN()};
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm).validationMetric;
         if(0 == iTerm && 0 == iEpoch) {
            CHECK_APPROX(validationMetric, 141.61);
            termScore = test.GetCurrentTermScore(iTerm, {}, 0);
            CHECK_APPROX(termScore, 0.1);
         }
         if(0 == iTerm && 1 == iEpoch) {
            CHECK_APPROX(validationMetric, 139.263601);
            termScore = test.GetCurrentTermScore(iTerm, {}, 0);
            CHECK_APPROX(termScore, 0.199);
         }
      }
   }
   CHECK_APPROX(validationMetric, 4.001727036272099502004735302456);
   termScore = test.GetCurrentTermScore(0, {}, 0);
   CHECK_APPROX(termScore, 9.9995682875258822);
}

TEST_CASE("Term with zero features, boosting, binary") {
   TestBoost test = TestBoost(Task_BinaryClassification,
         {},
         {{}},
         {TestSample({}, 0)},
         {TestSample({}, 0)},
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         nullptr,
         0);

   double validationMetric = double{std::numeric_limits<double>::quiet_NaN()};
   double termScore = double{std::numeric_limits<double>::quiet_NaN()};
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm).validationMetric;
         if(0 == iTerm && 0 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 0.68319717972663419, double{1e-1});
            termScore = test.GetCurrentTermScore(iTerm, {}, 0);
            CHECK(0 == termScore);
            termScore = test.GetCurrentTermScore(iTerm, {}, 1);
            CHECK_APPROX_TOLERANCE(termScore, -0.02, double{1e-1});
         }
         if(0 == iTerm && 1 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 0.67344419889200957, double{1e-1});
            termScore = test.GetCurrentTermScore(iTerm, {}, 0);
            CHECK(0 == termScore);
            termScore = test.GetCurrentTermScore(iTerm, {}, 1);
            CHECK_APPROX_TOLERANCE(termScore, -0.039801986733067563, double{1e-1});
         }
      }
   }
   CHECK_APPROX_TOLERANCE(validationMetric, 2.2621439908125974e-05, double{1e+1});
   termScore = test.GetCurrentTermScore(0, {}, 0);
   CHECK(0 == termScore);
   termScore = test.GetCurrentTermScore(0, {}, 1);
   CHECK_APPROX_TOLERANCE(termScore, -10.696601122148364, double{1e-2});
}

TEST_CASE("Term with zero features, boosting, multiclass") {
   TestBoost test = TestBoost(3,
         {},
         {{}},
         {
               TestSample({}, 0),
               TestSample({}, 1),
               TestSample({}, 2),
               TestSample({}, 2),
         },
         {TestSample({}, 0)});

   double validationMetric = double{std::numeric_limits<double>::quiet_NaN()};
   double termScore = double{std::numeric_limits<double>::quiet_NaN()};
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm).validationMetric;
         if(0 == iTerm && 0 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 1.0688384008227103, double{1e-1});
            double zeroLogit = test.GetCurrentTermScore(iTerm, {}, 0);
            termScore = test.GetCurrentTermScore(iTerm, {}, 1) - zeroLogit;
            CHECK_APPROX(termScore, 0.0);
            termScore = test.GetCurrentTermScore(iTerm, {}, 2) - zeroLogit;
            CHECK_APPROX(termScore, 0.0074999999999999997);
         }
         if(0 == iTerm && 1 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 1.0401627411809615, double{1e-1});
            double zeroLogit = test.GetCurrentTermScore(iTerm, {}, 0);
            termScore = test.GetCurrentTermScore(iTerm, {}, 1) - zeroLogit;
            CHECK_APPROX(termScore, 0.0);
            termScore = test.GetCurrentTermScore(iTerm, {}, 2) - zeroLogit;
            CHECK_APPROX(termScore, 0.014915718091257543);
         }
      }
   }
   CHECK_APPROX_TOLERANCE(validationMetric, 1.3864288330078125, double{1e-1});
   double zeroLogit1 = test.GetCurrentTermScore(0, {}, 0);
   termScore = test.GetCurrentTermScore(0, {}, 1) - zeroLogit1;
   CHECK_APPROX(termScore, 0.0);
   termScore = test.GetCurrentTermScore(0, {}, 2) - zeroLogit1;
   CHECK_APPROX(termScore, 0.69314934225893399);
}

TEST_CASE("Term with one feature with one or two states is the exact same as zero terms, boosting, regression") {
   TestBoost testZeroDimensions = TestBoost(Task_Regression, {}, {{}}, {TestSample({}, 10)}, {TestSample({}, 12)});

   TestBoost testOneState =
         TestBoost(Task_Regression, {FeatureTest(2, true, false)}, {{0}}, {TestSample({0}, 10)}, {TestSample({0}, 12)});

   TestBoost testTwoStates =
         TestBoost(Task_Regression, {FeatureTest(2)}, {{0}}, {TestSample({1}, 10)}, {TestSample({1}, 12)});

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroDimensions.GetCountTerms() == testOneState.GetCountTerms());
      assert(testZeroDimensions.GetCountTerms() == testTwoStates.GetCountTerms());
      for(size_t iTerm = 0; iTerm < testZeroDimensions.GetCountTerms(); ++iTerm) {
         double validationMetricZeroDimensions = testZeroDimensions.Boost(iTerm).validationMetric;
         double validationMetricOneState = testOneState.Boost(iTerm).validationMetric;
         CHECK_APPROX(validationMetricZeroDimensions, validationMetricOneState);
         double validationMetricTwoStates = testTwoStates.Boost(iTerm).validationMetric;
         CHECK_APPROX(validationMetricZeroDimensions, validationMetricTwoStates);

         double termScoreZeroDimensions = testZeroDimensions.GetCurrentTermScore(iTerm, {}, 0);
         double termScoreOneState = testOneState.GetCurrentTermScore(iTerm, {0}, 0);
         CHECK_APPROX(termScoreZeroDimensions, termScoreOneState);
         double termScoreTwoStates = testTwoStates.GetCurrentTermScore(iTerm, {1}, 0);
         CHECK_APPROX(termScoreZeroDimensions, termScoreTwoStates);
      }
   }
}

TEST_CASE("Term with one feature with one or two states is the exact same as zero terms, boosting, binary") {
   TestBoost testZeroDimensions = TestBoost(Task_BinaryClassification,
         {},
         {{}},
         {TestSample({}, 0)},
         {TestSample({}, 0)},
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         nullptr,
         0);

   TestBoost testOneState = TestBoost(Task_BinaryClassification,
         {FeatureTest(2, true, false)},
         {{0}},
         {TestSample({0}, 0)},
         {TestSample({0}, 0)},
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         nullptr,
         0);

   TestBoost testTwoStates = TestBoost(Task_BinaryClassification,
         {FeatureTest(2)},
         {{0}},
         {TestSample({1}, 0)},
         {TestSample({1}, 0)},
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         nullptr,
         0);

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroDimensions.GetCountTerms() == testOneState.GetCountTerms());
      assert(testZeroDimensions.GetCountTerms() == testTwoStates.GetCountTerms());
      for(size_t iTerm = 0; iTerm < testZeroDimensions.GetCountTerms(); ++iTerm) {
         double validationMetricZeroDimensions = testZeroDimensions.Boost(iTerm).validationMetric;
         double validationMetricOneState = testOneState.Boost(iTerm).validationMetric;
         CHECK_APPROX(validationMetricZeroDimensions, validationMetricOneState);
         double validationMetricTwoStates = testTwoStates.Boost(iTerm).validationMetric;
         CHECK_APPROX(validationMetricZeroDimensions, validationMetricTwoStates);

         double termScoreZeroDimensions0 = testZeroDimensions.GetCurrentTermScore(iTerm, {}, 0);
         double termScoreOneState0 = testOneState.GetCurrentTermScore(iTerm, {0}, 0);
         CHECK_APPROX(termScoreZeroDimensions0, termScoreOneState0);
         double termScoreTwoStates0 = testTwoStates.GetCurrentTermScore(iTerm, {1}, 0);
         CHECK_APPROX(termScoreZeroDimensions0, termScoreTwoStates0);

         double termScoreZeroDimensions1 = testZeroDimensions.GetCurrentTermScore(iTerm, {}, 1);
         double termScoreOneState1 = testOneState.GetCurrentTermScore(iTerm, {0}, 1);
         CHECK_APPROX(termScoreZeroDimensions1, termScoreOneState1);
         double termScoreTwoStates1 = testTwoStates.GetCurrentTermScore(iTerm, {1}, 1);
         CHECK_APPROX(termScoreZeroDimensions1, termScoreTwoStates1);
      }
   }
}

TEST_CASE("Term with one feature with one or two states is the exact same as zero terms, boosting, multiclass") {
   TestBoost testZeroDimensions = TestBoost(3, {}, {{}}, {TestSample({}, 0)}, {TestSample({}, 0)});

   TestBoost testOneState =
         TestBoost(3, {FeatureTest(2, true, false)}, {{0}}, {TestSample({0}, 0)}, {TestSample({0}, 0)});

   TestBoost testTwoStates = TestBoost(3, {FeatureTest(2)}, {{0}}, {TestSample({1}, 0)}, {TestSample({1}, 0)});

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(testZeroDimensions.GetCountTerms() == testOneState.GetCountTerms());
      assert(testZeroDimensions.GetCountTerms() == testTwoStates.GetCountTerms());
      for(size_t iTerm = 0; iTerm < testZeroDimensions.GetCountTerms(); ++iTerm) {
         double validationMetricZeroDimensions = testZeroDimensions.Boost(iTerm).validationMetric;
         double validationMetricOneState = testOneState.Boost(iTerm).validationMetric;
         CHECK_APPROX(validationMetricZeroDimensions, validationMetricOneState);
         double validationMetricTwoStates = testTwoStates.Boost(iTerm).validationMetric;
         CHECK_APPROX(validationMetricZeroDimensions, validationMetricTwoStates);

         double termScoreZeroDimensions0 = testZeroDimensions.GetCurrentTermScore(iTerm, {}, 0);
         double termScoreOneState0 = testOneState.GetCurrentTermScore(iTerm, {0}, 0);
         CHECK_APPROX(termScoreZeroDimensions0, termScoreOneState0);
         double termScoreTwoStates0 = testTwoStates.GetCurrentTermScore(iTerm, {1}, 0);
         CHECK_APPROX(termScoreZeroDimensions0, termScoreTwoStates0);

         double termScoreZeroDimensions1 = testZeroDimensions.GetCurrentTermScore(iTerm, {}, 1);
         double termScoreOneState1 = testOneState.GetCurrentTermScore(iTerm, {0}, 1);
         CHECK_APPROX(termScoreZeroDimensions1, termScoreOneState1);
         double termScoreTwoStates1 = testTwoStates.GetCurrentTermScore(iTerm, {1}, 1);
         CHECK_APPROX(termScoreZeroDimensions1, termScoreTwoStates1);

         double termScoreZeroDimensions2 = testZeroDimensions.GetCurrentTermScore(iTerm, {}, 2);
         double termScoreOneState2 = testOneState.GetCurrentTermScore(iTerm, {0}, 2);
         CHECK_APPROX(termScoreZeroDimensions2, termScoreOneState2);
         double termScoreTwoStates2 = testTwoStates.GetCurrentTermScore(iTerm, {1}, 2);
         CHECK_APPROX(termScoreZeroDimensions2, termScoreTwoStates2);
      }
   }
}

TEST_CASE("2 dimensional with two splits in each dimension, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(3), FeatureTest(3)},
         {{0, 1}},
         {
               TestSample({0, 0}, 1),
               TestSample({0, 1}, 2),
               TestSample({0, 2}, 3),
               TestSample({1, 0}, 4),
               TestSample({1, 1}, 5),
               TestSample({1, 2}, 6),
               TestSample({2, 0}, 7),
               TestSample({2, 1}, 8),
               TestSample({2, 2}, 9),
         },
         {TestSample({0, 0}, 10)});

   double validationMetric = test.Boost(0).validationMetric;
   CHECK_APPROX(validationMetric, 99.500624999999999);
}

#ifdef NEVER
// TODO: restore this test
TEST_CASE("3 dimensional term with one dimension reduced in different ways, boosting, regression") {
   TestBoost test0 = TestBoost(Task_Regression,
         {FeatureTest(2, true, false), FeatureTest(2), FeatureTest(2)},
         {{0, 1, 2}},
         {
               TestSample({0, 0, 0}, 9),
               TestSample({0, 1, 0}, 10),
               TestSample({0, 0, 1}, 11),
               TestSample({0, 1, 1}, 12),
         },
         {TestSample({0, 1, 0}, 12)});

   TestBoost test1 = TestBoost(Task_Regression,
         {FeatureTest(2), FeatureTest(2, true, false), FeatureTest(2)},
         {{0, 1, 2}},
         {
               TestSample({0, 0, 0}, 9),
               TestSample({0, 0, 1}, 10),
               TestSample({1, 0, 0}, 11),
               TestSample({1, 0, 1}, 12),
         },
         {TestSample({0, 0, 1}, 12)});

   TestBoost test2 = TestBoost(Task_Regression,
         {FeatureTest(2), FeatureTest(2), FeatureTest(2, true, false)},
         {{0, 1, 2}},
         {
               TestSample({0, 0, 0}, 9),
               TestSample({1, 0, 0}, 10),
               TestSample({0, 1, 0}, 11),
               TestSample({1, 1, 0}, 12),
         },
         {TestSample({1, 0, 0}, 12)});

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      assert(test0.GetCountTerms() == test1.GetCountTerms());
      assert(test0.GetCountTerms() == test2.GetCountTerms());
      for(size_t iTerm = 0; iTerm < test0.GetCountTerms(); ++iTerm) {
         double validationMetric0 = test0.Boost(iTerm).validationMetric;
         double validationMetric1 = test1.Boost(iTerm).validationMetric;
         CHECK_APPROX(validationMetric0, validationMetric1);
         double validationMetric2 = test2.Boost(iTerm).validationMetric;
         CHECK_APPROX(validationMetric0, validationMetric2);

         double termScore01 = test0.GetCurrentTermScore(iTerm, {0, 0, 0}, 0);
         double termScore02 = test0.GetCurrentTermScore(iTerm, {1, 0, 0}, 0);
         double termScore03 = test0.GetCurrentTermScore(iTerm, {0, 1, 0}, 0);
         double termScore04 = test0.GetCurrentTermScore(iTerm, {1, 1, 0}, 0);

         double termScore11 = test1.GetCurrentTermScore(iTerm, {0, 0, 0}, 0);
         double termScore12 = test1.GetCurrentTermScore(iTerm, {0, 0, 1}, 0);
         double termScore13 = test1.GetCurrentTermScore(iTerm, {1, 0, 0}, 0);
         double termScore14 = test1.GetCurrentTermScore(iTerm, {1, 0, 1}, 0);
         CHECK_APPROX(termScore11, termScore01);
         CHECK_APPROX(termScore12, termScore02);
         CHECK_APPROX(termScore13, termScore03);
         CHECK_APPROX(termScore14, termScore04);

         double termScore21 = test2.GetCurrentTermScore(iTerm, {0, 0, 0}, 0);
         double termScore22 = test2.GetCurrentTermScore(iTerm, {0, 1, 0}, 0);
         double termScore23 = test2.GetCurrentTermScore(iTerm, {0, 0, 1}, 0);
         double termScore24 = test2.GetCurrentTermScore(iTerm, {0, 1, 1}, 0);
         CHECK_APPROX(termScore21, termScore01);
         CHECK_APPROX(termScore22, termScore02);
         CHECK_APPROX(termScore23, termScore03);
         CHECK_APPROX(termScore24, termScore04);
      }
   }
}
#endif // NEVER

TEST_CASE("Random splitting with 3 features, boosting, multiclass") {
   static const std::vector<IntEbm> k_leavesMax = {IntEbm{3}};

   TestBoost test = TestBoost(3,
         {FeatureTest(4)},
         {{0}},
         {TestSample({0}, 0), TestSample({1}, 1), TestSample({2}, 1), TestSample({3}, 2)},
         {TestSample({1}, 0)});

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         double validationMetric = test.Boost(iTerm,
                                             TermBoostFlags_RandomSplits,
                                             k_learningRateDefault,
                                             k_minSamplesLeafDefault,
                                             k_minHessianDefault,
                                             k_regAlphaDefault,
                                             k_regLambdaDefault,
                                             k_maxDeltaStepDefault,
                                             k_minCategorySamplesDefault,
                                             k_categoricalSmoothingDefault,
                                             k_maxCategoricalThresholdDefault,
                                             k_categoricalInclusionPercentDefault,
                                             k_leavesMax)
                                         .validationMetric;
         if(0 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 1.0340957641601563f, double{1e-1});

            double zeroLogit = test.GetCurrentTermScore(iTerm, {1}, 0);

            double termScore1 = test.GetCurrentTermScore(iTerm, {1}, 1) - zeroLogit;
            CHECK_APPROX(termScore1, 0.0f);

            double termScore2 = test.GetCurrentTermScore(iTerm, {1}, 2) - zeroLogit;
            CHECK_APPROX(termScore2, -0.015f);
         }
      }
   }
}

TEST_CASE("Random splitting with 3 features, boosting, multiclass, sums") {
   static const std::vector<IntEbm> k_leavesMax = {IntEbm{3}};

   TestBoost test = TestBoost(3,
         {FeatureTest(4)},
         {{0}},
         {TestSample({0}, 0), TestSample({1}, 1), TestSample({2}, 1), TestSample({3}, 2)},
         {TestSample({1}, 0)});

   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         double validationMetric = test.Boost(iTerm,
                                             TermBoostFlags_RandomSplits | TermBoostFlags_GradientSums,
                                             k_learningRateDefault,
                                             k_minSamplesLeafDefault,
                                             k_minHessianDefault,
                                             k_regAlphaDefault,
                                             k_regLambdaDefault,
                                             k_maxDeltaStepDefault,
                                             k_minCategorySamplesDefault,
                                             k_categoricalSmoothingDefault,
                                             k_maxCategoricalThresholdDefault,
                                             k_categoricalInclusionPercentDefault,
                                             k_leavesMax)
                                         .validationMetric;
         if(0 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 1.0986122886681098, double{1e-1});

            // we set our update to zero since this is for sums so we need to set it to something

            double termScore0 = test.GetCurrentTermScore(iTerm, {1}, 0);
            CHECK_APPROX(termScore0, 0.0f);

            double termScore1 = test.GetCurrentTermScore(iTerm, {1}, 1);
            CHECK_APPROX(termScore1, 0.0f);

            double termScore2 = test.GetCurrentTermScore(iTerm, {1}, 2);
            CHECK_APPROX(termScore2, 0.0f);
         }
      }
   }
}

TEST_CASE("Random splitting, tripple with one dimension missing, multiclass") {
   static constexpr IntEbm cStates = 7;
   static const std::vector<IntEbm> k_leavesMax = {IntEbm{3}, IntEbm{3}, IntEbm{3}};

   std::vector<TestSample> samples;
   for(IntEbm i0 = 0; i0 < cStates; ++i0) {
      for(IntEbm i2 = 0; i2 < cStates; ++i2) {
         // create a few zero spaces where we have no data
         if(i0 != i2) {
            if(i0 < i2) {
               samples.push_back(TestSample({i0, 0, i2}, 1));
            } else {
               samples.push_back(TestSample({i0, 0, i2}, 2));
            }
         }
      }
   }

   TestBoost test = TestBoost(3,
         {FeatureTest(cStates), FeatureTest(2, true, false), FeatureTest(cStates)},
         {{0, 1, 2}},
         samples,
         samples // evaluate on the train set
   );

   double validationMetric = double{0};
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm,
                                      TermBoostFlags_RandomSplits,
                                      k_learningRateDefault,
                                      1,
                                      k_minHessianDefault,
                                      k_regAlphaDefault,
                                      k_regLambdaDefault,
                                      k_maxDeltaStepDefault,
                                      k_minCategorySamplesDefault,
                                      k_categoricalSmoothingDefault,
                                      k_maxCategoricalThresholdDefault,
                                      k_categoricalInclusionPercentDefault,
                                      k_leavesMax)
                                  .validationMetric;
      }
   }

   CHECK(validationMetric <= 0.00068252183319557282 * 1.3);

   for(IntEbm i0 = 0; i0 < cStates; ++i0) {
      for(IntEbm i2 = 0; i2 < cStates; ++i2) {
#if false
         std::cout << std::endl;
         std::cout << i0 << ' ' << '0' << ' ' << i2 << std::endl;

         double termScore0 = test.GetCurrentTermScore(0, { static_cast<size_t>(i0), static_cast<size_t>(0), static_cast<size_t>(i2) }, 0);
         std::cout << termScore0 << std::endl;

         double termScore1 = test.GetCurrentTermScore(0, { static_cast<size_t>(i0), static_cast<size_t>(0), static_cast<size_t>(i2) }, 1);
         std::cout << termScore1 << std::endl;

         double termScore2 = test.GetCurrentTermScore(0, { static_cast<size_t>(i0), static_cast<size_t>(0), static_cast<size_t>(i2) }, 2);
         std::cout << termScore2 << std::endl;
#endif
      }
   }
}

TEST_CASE("Random splitting, pure tripples, multiclass") {
   static constexpr IntEbm cStates = 7;
   static const std::vector<IntEbm> k_leavesMax = {IntEbm{3}, IntEbm{3}, IntEbm{3}};

   std::vector<TestSample> samples;
   for(IntEbm i0 = 0; i0 < cStates; ++i0) {
      for(IntEbm i1 = 0; i1 < cStates; ++i1) {
         for(IntEbm i2 = 0; i2 < cStates; ++i2) {
            if(i0 == i1 && i0 == i2) {
               samples.push_back(TestSample({i0, i1, i2}, 0));
            } else if(i0 < i1) {
               samples.push_back(TestSample({i0, i1, i2}, 1));
            } else {
               samples.push_back(TestSample({i0, i1, i2}, 2));
            }
         }
      }
   }

   TestBoost test = TestBoost(3,
         {FeatureTest(cStates), FeatureTest(cStates), FeatureTest(cStates)},
         {{0, 1, 2}},
         samples,
         samples // evaluate on the train set
   );

   double validationMetric = double{0};
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm,
                                      TermBoostFlags_RandomSplits,
                                      k_learningRateDefault,
                                      1,
                                      k_minHessianDefault,
                                      k_regAlphaDefault,
                                      k_regLambdaDefault,
                                      k_maxDeltaStepDefault,
                                      k_minCategorySamplesDefault,
                                      k_categoricalSmoothingDefault,
                                      k_maxCategoricalThresholdDefault,
                                      k_categoricalInclusionPercentDefault,
                                      k_leavesMax)
                                  .validationMetric;
      }
   }
   CHECK(validationMetric <= 0.024179781143588853 * 1.4);

   for(IntEbm i0 = 0; i0 < cStates; ++i0) {
      for(IntEbm i1 = 0; i1 < cStates; ++i1) {
         for(IntEbm i2 = 0; i2 < cStates; ++i2) {
#if false
            std::cout << std::endl;
            std::cout << i0 << ' ' << i1 << ' ' << i2 << std::endl;

            double termScore0 = test.GetCurrentTermScore(0, { static_cast<size_t>(i0), static_cast<size_t>(i1), static_cast<size_t>(i2) }, 0);
            std::cout << termScore0 << std::endl;

            double termScore1 = test.GetCurrentTermScore(0, { static_cast<size_t>(i0), static_cast<size_t>(i1), static_cast<size_t>(i2) }, 1);
            std::cout << termScore1 << std::endl;

            double termScore2 = test.GetCurrentTermScore(0, { static_cast<size_t>(i0), static_cast<size_t>(i1), static_cast<size_t>(i2) }, 2);
            std::cout << termScore2 << std::endl;
#endif
         }
      }
   }
}

TEST_CASE("Random splitting, pure tripples, regression") {
   static constexpr IntEbm cStates = 7;
   static const std::vector<IntEbm> k_leavesMax = {IntEbm{3}, IntEbm{3}, IntEbm{3}};

   std::vector<TestSample> samples;
   for(IntEbm i0 = 0; i0 < cStates; ++i0) {
      for(IntEbm i1 = 0; i1 < cStates; ++i1) {
         for(IntEbm i2 = 0; i2 < cStates; ++i2) {
            if(i0 == i1 && i0 == i2) {
               samples.push_back(TestSample({i0, i1, i2}, -10));
            } else if(i0 < i1) {
               samples.push_back(TestSample({i0, i1, i2}, 1));
            } else {
               samples.push_back(TestSample({i0, i1, i2}, 2));
            }
         }
      }
   }

   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(cStates), FeatureTest(cStates), FeatureTest(cStates)},
         {{0, 1, 2}},
         samples,
         samples // evaluate on the train set
   );

   double validationMetric = double{0};
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm,
                                      TermBoostFlags_RandomSplits,
                                      k_learningRateDefault,
                                      1,
                                      k_minHessianDefault,
                                      k_regAlphaDefault,
                                      k_regLambdaDefault,
                                      k_maxDeltaStepDefault,
                                      k_minCategorySamplesDefault,
                                      k_categoricalSmoothingDefault,
                                      k_maxCategoricalThresholdDefault,
                                      k_categoricalInclusionPercentDefault,
                                      k_leavesMax)
                                  .validationMetric;
      }
   }

   CHECK_APPROX(validationMetric, 1.4542426709976266);

   for(IntEbm i0 = 0; i0 < cStates; ++i0) {
      for(IntEbm i1 = 0; i1 < cStates; ++i1) {
         for(IntEbm i2 = 0; i2 < cStates; ++i2) {
#if false
            std::cout << std::endl;
            std::cout << i0 << ' ' << i1 << ' ' << i2 << std::endl;

            double termScore0 = test.GetCurrentTermScore(0, { static_cast<size_t>(i0), static_cast<size_t>(i1), static_cast<size_t>(i2) }, 0);
            std::cout << termScore0 << std::endl;
#endif
         }
      }
   }
}

TEST_CASE("Random splitting, pure tripples, only 1 leaf, multiclass") {
   static constexpr IntEbm k_cStates = 7;
   static constexpr IntEbm k_minSamplesLeaf = 1;
   static constexpr double k_minHessian = k_minHessianDefault;
   static const std::vector<IntEbm> k_leavesMax = {IntEbm{1}, IntEbm{1}, IntEbm{1}};

   std::vector<TestSample> samples;
   for(IntEbm i0 = 0; i0 < k_cStates; ++i0) {
      for(IntEbm i1 = 0; i1 < k_cStates; ++i1) {
         for(IntEbm i2 = 0; i2 < k_cStates; ++i2) {
            if(i0 == i1 && i0 == i2) {
               samples.push_back(TestSample({i0, i1, i2}, 0));
            } else if(i0 < i1) {
               samples.push_back(TestSample({i0, i1, i2}, 1));
            } else {
               samples.push_back(TestSample({i0, i1, i2}, 2));
            }
         }
      }
   }

   TestBoost test = TestBoost(3,
         {FeatureTest(k_cStates), FeatureTest(k_cStates), FeatureTest(k_cStates)},
         {{0, 1, 2}},
         samples,
         samples // evaluate on the train set
   );

   double validationMetric = double{0};
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm,
                                      TermBoostFlags_RandomSplits,
                                      k_learningRateDefault,
                                      k_minSamplesLeaf,
                                      k_minHessian,
                                      k_regAlphaDefault,
                                      k_regLambdaDefault,
                                      k_maxDeltaStepDefault,
                                      k_minCategorySamplesDefault,
                                      k_categoricalSmoothingDefault,
                                      k_maxCategoricalThresholdDefault,
                                      k_categoricalInclusionPercentDefault,
                                      k_leavesMax)
                                  .validationMetric;
      }
   }

   // it can't really benefit from splitting since we only allow the boosting rounds to have 1 leaf
   CHECK(validationMetric <= 0.73616339235889672 * 1.1);
   CHECK(0.73616339235889672 / 1.1 <= validationMetric);

   for(IntEbm i0 = 0; i0 < k_cStates; ++i0) {
      for(IntEbm i1 = 0; i1 < k_cStates; ++i1) {
         for(IntEbm i2 = 0; i2 < k_cStates; ++i2) {
#if false
            std::cout << std::endl;
            std::cout << i0 << ' ' << i1 << ' ' << i2 << std::endl;

            double termScore0 = test.GetCurrentTermScore(0, { static_cast<size_t>(i0), static_cast<size_t>(i1), static_cast<size_t>(i2) }, 0);
            std::cout << termScore0 << std::endl;

            double termScore1 = test.GetCurrentTermScore(0, { static_cast<size_t>(i0), static_cast<size_t>(i1), static_cast<size_t>(i2) }, 1);
            std::cout << termScore1 << std::endl;

            double termScore2 = test.GetCurrentTermScore(0, { static_cast<size_t>(i0), static_cast<size_t>(i1), static_cast<size_t>(i2) }, 2);
            std::cout << termScore2 << std::endl;
#endif
         }
      }
   }
}

TEST_CASE("Random splitting, no splits, binary, sums") {
   static const std::vector<IntEbm> k_leavesMax = {IntEbm{3}};

   TestBoost test = TestBoost(Task_BinaryClassification,
         {FeatureTest(2, true, false)},
         {{0}},
         {
               TestSample({0}, 0),
               TestSample({0}, 0),
               TestSample({0}, 1),
               TestSample({0}, 1),
               TestSample({0}, 1),
         },
         {TestSample({0}, 0)});

   double validationMetric = 0;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm,
                                      TermBoostFlags_RandomSplits | TermBoostFlags_GradientSums,
                                      k_learningRateDefault,
                                      k_minSamplesLeafDefault,
                                      k_minHessianDefault,
                                      k_regAlphaDefault,
                                      k_regLambdaDefault,
                                      k_maxDeltaStepDefault,
                                      k_minCategorySamplesDefault,
                                      k_categoricalSmoothingDefault,
                                      k_maxCategoricalThresholdDefault,
                                      k_categoricalInclusionPercentDefault,
                                      k_leavesMax)
                                  .validationMetric;
         if(0 == iEpoch) {
            CHECK_APPROX_TOLERANCE(validationMetric, 0.69314718055994529, double{1e-1});

            // we set our update to zero since we're getting the sum

            double termScore0 = test.GetCurrentTermScore(iTerm, {0}, 0);
            CHECK(0 == termScore0);

            double termScore1 = test.GetCurrentTermScore(iTerm, {0}, 1);
            CHECK(0 == termScore1);
         }
      }
   }

   // we're generating updates from gradient sums, which isn't good, so we expect a bad result
   CHECK_APPROX_TOLERANCE(validationMetric, 0.69314718055994529, double{1e-1});
}

TEST_CASE("zero gain, boosting, regression") {
   // construct a case where there should be zero gain and test that we get zero.

   // we start with a singular bin that has 5 cases, and split it to two bins with 2 and 3 cases respectively.
   // We can arbitrarily set the gradient totals to 4 and 7, and then calculate what the

   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 10.75, 1.5),
               TestSample({1}, 10.75, 2.25),
         },
         {});

   double gainAvg = test.Boost(0, TermBoostFlags_Default, k_learningRateDefault, 0).gainAvg;
   CHECK(0 <= gainAvg && gainAvg < 0.0000001);
}

TEST_CASE("pair and main gain identical, boosting, regression") {
   // if we have a scenario where boosting has gain in the split in one dimension, but the gain
   // for the split on both sides into the pair are zero, then the gain from the pair boosting
   // should be identical to the gain from the main if we were to combine the pairs into mains

   TestBoost test1 = TestBoost(Task_Regression,
         {FeatureTest(2), FeatureTest(2)},
         {{0, 1}},
         {
               TestSample({0, 0}, 10.75, 1.5),
               TestSample({0, 1}, 10.75, 2.25),
               TestSample({1, 0}, 11.25, 3.25),
               TestSample({1, 1}, 11.25, 4.5),
         },
         {});

   const double gainAvg1 = test1.Boost(0).gainAvg;

   TestBoost test2 = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 10.75, 1.5 + 2.25),
               TestSample({1}, 11.25, 3.25 + 4.5),
         },
         {});

   double gainAvg2 = test2.Boost(0).gainAvg;

   CHECK_APPROX(gainAvg1, gainAvg2);
}

TEST_CASE("tweedie, boosting") {
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(2, true, false)},
         {{0}},
         {TestSample({0}, 10)},
         {TestSample({0}, 12)},
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         "tweedie_deviance:variance_power=1.3");

   double validationMetric = double{std::numeric_limits<double>::quiet_NaN()};
   double termScore = double{std::numeric_limits<double>::quiet_NaN()};
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm, TermBoostFlags_Default).validationMetric;
      }
   }

   CHECK_APPROX(validationMetric, 54.414769150086464);

   termScore = test.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore, 2.3025076860047466);
}

TEST_CASE("purified boosting of impure input, regression") {
   // We give the booster a dataset with only impurity and ask it to purify the model
   // which results in no update each iteration.

   TestBoost testPure = TestBoost(Task_Regression,
         {FeatureTest(2), FeatureTest(2)},
         {{0, 1}},
         {
               TestSample({0, 0}, 9),
               TestSample({0, 1}, 10),
               TestSample({1, 0}, 11),
               TestSample({1, 1}, 12),
         },
         {TestSample({0, 1}, 12)});

   for(int iEpoch = 0; iEpoch < 10; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < testPure.GetCountTerms(); ++iTerm) {
         double validationMetric0 = testPure.Boost(iTerm, TermBoostFlags_PurifyUpdate).validationMetric;
         CHECK_APPROX(validationMetric0, 144.0);

         double termScore01 = testPure.GetCurrentTermScore(iTerm, {0, 0}, 0);
         double termScore02 = testPure.GetCurrentTermScore(iTerm, {1, 0}, 0);
         double termScore03 = testPure.GetCurrentTermScore(iTerm, {0, 1}, 0);
         double termScore04 = testPure.GetCurrentTermScore(iTerm, {1, 1}, 0);

         CHECK_APPROX(termScore01, 0.0);
         CHECK_APPROX(termScore02, 0.0);
         CHECK_APPROX(termScore03, 0.0);
         CHECK_APPROX(termScore04, 0.0);
      }
   }
}

TEST_CASE("purified boosting of impure input, multiclass") {
   // We give the booster a dataset with only impurity and ask it to purify the model
   // which results in no update each iteration.

   TestBoost testPure = TestBoost(3,
         {FeatureTest(2), FeatureTest(2)},
         {{0, 1}},
         {
               TestSample({0, 0}, 2),
               TestSample({0, 1}, 2),
               TestSample({1, 0}, 0),
               TestSample({1, 1}, 0),
               TestSample({0, 0}, 1),
               TestSample({0, 1}, 0),
               TestSample({1, 0}, 1),
               TestSample({1, 1}, 0),
         },
         {TestSample({0, 1}, 1)});

   for(int iEpoch = 0; iEpoch < 10; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < testPure.GetCountTerms(); ++iTerm) {
         double validationMetric0 = testPure.Boost(iTerm, TermBoostFlags_PurifyUpdate).validationMetric;
         CHECK_APPROX(validationMetric0, 1.0986122886681096);

         double termScore01 = testPure.GetCurrentTermScore(iTerm, {0, 0}, 0);
         double termScore02 = testPure.GetCurrentTermScore(iTerm, {1, 0}, 0);
         double termScore03 = testPure.GetCurrentTermScore(iTerm, {0, 1}, 0);
         double termScore04 = testPure.GetCurrentTermScore(iTerm, {1, 1}, 0);

         CHECK(-0.00001 < termScore01 && termScore01 < 0.00001);
         CHECK(-0.00001 < termScore02 && termScore01 < 0.00001);
         CHECK(-0.00001 < termScore03 && termScore01 < 0.00001);
         CHECK(-0.00001 < termScore04 && termScore01 < 0.00001);

         termScore01 = testPure.GetCurrentTermScore(iTerm, {0, 0}, 1);
         termScore02 = testPure.GetCurrentTermScore(iTerm, {1, 0}, 1);
         termScore03 = testPure.GetCurrentTermScore(iTerm, {0, 1}, 1);
         termScore04 = testPure.GetCurrentTermScore(iTerm, {1, 1}, 1);

         CHECK(-0.00001 < termScore01 && termScore01 < 0.00001);
         CHECK(-0.00001 < termScore02 && termScore01 < 0.00001);
         CHECK(-0.00001 < termScore03 && termScore01 < 0.00001);
         CHECK(-0.00001 < termScore04 && termScore01 < 0.00001);

         termScore01 = testPure.GetCurrentTermScore(iTerm, {0, 0}, 2);
         termScore02 = testPure.GetCurrentTermScore(iTerm, {1, 0}, 2);
         termScore03 = testPure.GetCurrentTermScore(iTerm, {0, 1}, 2);
         termScore04 = testPure.GetCurrentTermScore(iTerm, {1, 1}, 2);

         CHECK(-0.00001 < termScore01 && termScore01 < 0.00001);
         CHECK(-0.00001 < termScore02 && termScore01 < 0.00001);
         CHECK(-0.00001 < termScore03 && termScore01 < 0.00001);
         CHECK(-0.00001 < termScore04 && termScore01 < 0.00001);
      }
   }
}

#ifdef NEVER
// TODO: restore this test
TEST_CASE("purified boosting and impure boosting identical for pure input, regression") {
   TestBoost testPure = TestBoost(Task_Regression,
         {FeatureTest(2), FeatureTest(2)},
         {{0, 1}},
         {
               TestSample({0, 0}, 3.5),
               TestSample({0, 1}, -3.5),
               TestSample({1, 0}, -3.5),
               TestSample({1, 1}, 3.5),
         },
         {TestSample({0, 1}, 12)});

   TestBoost testImpure = TestBoost(Task_Regression,
         {FeatureTest(2), FeatureTest(2)},
         {{0, 1}},
         {
               TestSample({0, 0}, 3.5),
               TestSample({0, 1}, -3.5),
               TestSample({1, 0}, -3.5),
               TestSample({1, 1}, 3.5),
         },
         {TestSample({0, 1}, 12)});

   for(int iEpoch = 0; iEpoch < 10; ++iEpoch) {
      assert(testPure.GetCountTerms() == testImpure.GetCountTerms());
      for(size_t iTerm = 0; iTerm < testPure.GetCountTerms(); ++iTerm) {
         double validationMetric0 = testPure.Boost(iTerm, TermBoostFlags_PurifyUpdate).validationMetric;
         double validationMetric1 = testImpure.Boost(iTerm, TermBoostFlags_Default).validationMetric;
         CHECK_APPROX(validationMetric0, validationMetric1);

         double termScore01 = testPure.GetCurrentTermScore(iTerm, {0, 0}, 0);
         double termScore02 = testPure.GetCurrentTermScore(iTerm, {1, 0}, 0);
         double termScore03 = testPure.GetCurrentTermScore(iTerm, {0, 1}, 0);
         double termScore04 = testPure.GetCurrentTermScore(iTerm, {1, 1}, 0);

         double termScore11 = testImpure.GetCurrentTermScore(iTerm, {0, 0}, 0);
         double termScore12 = testImpure.GetCurrentTermScore(iTerm, {1, 0}, 0);
         double termScore13 = testImpure.GetCurrentTermScore(iTerm, {0, 1}, 0);
         double termScore14 = testImpure.GetCurrentTermScore(iTerm, {1, 1}, 0);

         CHECK_APPROX(termScore11, termScore01);
         CHECK_APPROX(termScore12, termScore02);
         CHECK_APPROX(termScore13, termScore03);
         CHECK_APPROX(termScore14, termScore04);
      }
   }
}

TEST_CASE("purified boosting and impure boosting identical for pure input, multiclass") {
   TestBoost testPure = TestBoost(3,
         {FeatureTest(2), FeatureTest(2)},
         {{0, 1}},
         {
               TestSample({0, 0}, 1),
               TestSample({0, 0}, 1),
               TestSample({0, 0}, 0),
               TestSample({0, 0}, 0),
               TestSample({0, 0}, 0),
               TestSample({0, 0}, 2),

               TestSample({0, 1}, 1),
               TestSample({0, 1}, 1),
               TestSample({0, 1}, 0),
               TestSample({0, 1}, 2),
               TestSample({0, 1}, 2),
               TestSample({0, 1}, 2),

               TestSample({1, 0}, 1),
               TestSample({1, 0}, 1),
               TestSample({1, 0}, 0),
               TestSample({1, 0}, 2),
               TestSample({1, 0}, 2),
               TestSample({1, 0}, 2),

               TestSample({1, 1}, 1),
               TestSample({1, 1}, 1),
               TestSample({1, 1}, 0),
               TestSample({1, 1}, 0),
               TestSample({1, 1}, 0),
               TestSample({1, 1}, 2),
         },
         {TestSample({0, 1}, 1)},
         0,
         CreateBoosterFlags_Default,
         AccelerationFlags_NONE);

   TestBoost testImpure = TestBoost(3,
         {FeatureTest(2), FeatureTest(2)},
         {{0, 1}},
         {
               TestSample({0, 0}, 1),
               TestSample({0, 0}, 1),
               TestSample({0, 0}, 0),
               TestSample({0, 0}, 0),
               TestSample({0, 0}, 0),
               TestSample({0, 0}, 2),

               TestSample({0, 1}, 1),
               TestSample({0, 1}, 1),
               TestSample({0, 1}, 0),
               TestSample({0, 1}, 2),
               TestSample({0, 1}, 2),
               TestSample({0, 1}, 2),

               TestSample({1, 0}, 1),
               TestSample({1, 0}, 1),
               TestSample({1, 0}, 0),
               TestSample({1, 0}, 2),
               TestSample({1, 0}, 2),
               TestSample({1, 0}, 2),

               TestSample({1, 1}, 1),
               TestSample({1, 1}, 1),
               TestSample({1, 1}, 0),
               TestSample({1, 1}, 0),
               TestSample({1, 1}, 0),
               TestSample({1, 1}, 2),
         },
         {TestSample({0, 1}, 1)},
         0,
         CreateBoosterFlags_Default,
         AccelerationFlags_NONE);

   for(int iEpoch = 0; iEpoch < 10; ++iEpoch) {
      assert(testPure.GetCountTerms() == testImpure.GetCountTerms());
      for(size_t iTerm = 0; iTerm < testPure.GetCountTerms(); ++iTerm) {
         double validationMetric0 = testPure
                                          .Boost(iTerm,
                                                TermBoostFlags_PurifyUpdate | TermBoostFlags_DisableNewtonGain |
                                                      TermBoostFlags_DisableNewtonUpdate)
                                          .validationMetric;
         double validationMetric1 =
               testImpure.Boost(iTerm, TermBoostFlags_DisableNewtonGain | TermBoostFlags_DisableNewtonUpdate)
                     .validationMetric;
         CHECK_APPROX_TOLERANCE(validationMetric0, validationMetric1, 1e-2);

         double termScore01_0 = testPure.GetCurrentTermScore(iTerm, {0, 0}, 0);
         double termScore02_0 = testPure.GetCurrentTermScore(iTerm, {1, 0}, 0);
         double termScore03_0 = testPure.GetCurrentTermScore(iTerm, {0, 1}, 0);
         double termScore04_0 = testPure.GetCurrentTermScore(iTerm, {1, 1}, 0);

         double termScore11_0 = testImpure.GetCurrentTermScore(iTerm, {0, 0}, 0);
         double termScore12_0 = testImpure.GetCurrentTermScore(iTerm, {1, 0}, 0);
         double termScore13_0 = testImpure.GetCurrentTermScore(iTerm, {0, 1}, 0);
         double termScore14_0 = testImpure.GetCurrentTermScore(iTerm, {1, 1}, 0);

         CHECK_APPROX(termScore11_0, termScore01_0);
         CHECK_APPROX(termScore12_0, termScore02_0);
         CHECK_APPROX(termScore13_0, termScore03_0);
         CHECK_APPROX(termScore14_0, termScore04_0);

         double termScore01_1 = testPure.GetCurrentTermScore(iTerm, {0, 0}, 1);
         double termScore02_1 = testPure.GetCurrentTermScore(iTerm, {1, 0}, 1);
         double termScore03_1 = testPure.GetCurrentTermScore(iTerm, {0, 1}, 1);
         double termScore04_1 = testPure.GetCurrentTermScore(iTerm, {1, 1}, 1);

         double termScore11_1 = testImpure.GetCurrentTermScore(iTerm, {0, 0}, 1);
         double termScore12_1 = testImpure.GetCurrentTermScore(iTerm, {1, 0}, 1);
         double termScore13_1 = testImpure.GetCurrentTermScore(iTerm, {0, 1}, 1);
         double termScore14_1 = testImpure.GetCurrentTermScore(iTerm, {1, 1}, 1);

         CHECK(-0.00001 < termScore01_1 && termScore01_1 < 0.00001);
         CHECK(-0.00001 < termScore02_1 && termScore02_1 < 0.00001);
         CHECK(-0.00001 < termScore03_1 && termScore03_1 < 0.00001);
         CHECK(-0.00001 < termScore04_1 && termScore04_1 < 0.00001);

         CHECK(-0.00001 < termScore11_1 && termScore11_1 < 0.00001);
         CHECK(-0.00001 < termScore12_1 && termScore12_1 < 0.00001);
         CHECK(-0.00001 < termScore13_1 && termScore13_1 < 0.00001);
         CHECK(-0.00001 < termScore14_1 && termScore14_1 < 0.00001);

         double termScore01_2 = testPure.GetCurrentTermScore(iTerm, {0, 0}, 2);
         double termScore02_2 = testPure.GetCurrentTermScore(iTerm, {1, 0}, 2);
         double termScore03_2 = testPure.GetCurrentTermScore(iTerm, {0, 1}, 2);
         double termScore04_2 = testPure.GetCurrentTermScore(iTerm, {1, 1}, 2);

         double termScore11_2 = testImpure.GetCurrentTermScore(iTerm, {0, 0}, 2);
         double termScore12_2 = testImpure.GetCurrentTermScore(iTerm, {1, 0}, 2);
         double termScore13_2 = testImpure.GetCurrentTermScore(iTerm, {0, 1}, 2);
         double termScore14_2 = testImpure.GetCurrentTermScore(iTerm, {1, 1}, 2);

         CHECK_APPROX(termScore11_2, termScore01_2);
         CHECK_APPROX(termScore12_2, termScore02_2);
         CHECK_APPROX(termScore13_2, termScore03_2);
         CHECK_APPROX(termScore14_2, termScore04_2);
      }
   }
}
#endif // NEVER

TEST_CASE("purified boosting and impure boosting different for impure input, regression") {
   TestBoost testPure = TestBoost(Task_Regression,
         {FeatureTest(2), FeatureTest(2)},
         {{0, 1}},
         {
               TestSample({0, 0}, 2.0),
               TestSample({0, 1}, -3.5),
               TestSample({1, 0}, -3.5),
               TestSample({1, 1}, 3.5),
         },
         {TestSample({0, 1}, 12)});

   TestBoost testImpure = TestBoost(Task_Regression,
         {FeatureTest(2), FeatureTest(2)},
         {{0, 1}},
         {
               TestSample({0, 0}, 2.0),
               TestSample({0, 1}, -3.5),
               TestSample({1, 0}, -3.5),
               TestSample({1, 1}, 3.5),
         },
         {TestSample({0, 1}, 12)});

   for(int iEpoch = 0; iEpoch < 10; ++iEpoch) {
      assert(testPure.GetCountTerms() == testImpure.GetCountTerms());
      for(size_t iTerm = 0; iTerm < testPure.GetCountTerms(); ++iTerm) {
         double validationMetric0 = testPure.Boost(iTerm, TermBoostFlags_PurifyUpdate).validationMetric;
         double validationMetric1 = testImpure.Boost(iTerm, TermBoostFlags_Default).validationMetric;
         CHECK(validationMetric0 != validationMetric1);

         double termScore01 = testPure.GetCurrentTermScore(iTerm, {0, 0}, 0);
         double termScore02 = testPure.GetCurrentTermScore(iTerm, {1, 0}, 0);
         double termScore03 = testPure.GetCurrentTermScore(iTerm, {0, 1}, 0);
         double termScore04 = testPure.GetCurrentTermScore(iTerm, {1, 1}, 0);

         double termScore11 = testImpure.GetCurrentTermScore(iTerm, {0, 0}, 0);
         double termScore12 = testImpure.GetCurrentTermScore(iTerm, {1, 0}, 0);
         double termScore13 = testImpure.GetCurrentTermScore(iTerm, {0, 1}, 0);
         double termScore14 = testImpure.GetCurrentTermScore(iTerm, {1, 1}, 0);

         CHECK(termScore11 != termScore01);
         CHECK(termScore12 != termScore02);
         CHECK(termScore13 != termScore03);
         CHECK(termScore14 != termScore04);
      }
   }
}

TEST_CASE("lossguide, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(5, true, true, false), FeatureTest(5, true, true, true)},
         {{0}, {1}},
         {
               TestSample({0, 0}, 21),
               TestSample({1, 1}, 10),
               TestSample({2, 2}, 20),
               TestSample({3, 3}, 30),
               TestSample({4, 4}, 29),
         },
         {TestSample({0, 0}, 20.5)});

   // boost continuous missing lossguide
   double validationMetric = test.Boost(0, TermBoostFlags_Default).validationMetric;
   CHECK_APPROX(validationMetric, 411.88702500000005);

   // boost nominal
   validationMetric = test.Boost(1, TermBoostFlags_Default).validationMetric;
   CHECK_APPROX(validationMetric, 403.48957770250007);

   // boost continuous missing lossguide
   validationMetric = test.Boost(0, TermBoostFlags_Default).validationMetric;
   CHECK_APPROX(validationMetric, 395.45814649077033);

   double termScore;
   termScore = test.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore, 0.40592050000000002);
}

TEST_CASE("missing separate continuous with non-missing data, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(4, false, false, false)},
         {{0}},
         {
               TestSample({1}, 20.0),
               TestSample({2}, 30.0),
         },
         {TestSample({1}, 20.0)});

   // boost continuous missing separate
   double validationMetric = test.Boost(0, TermBoostFlags_MissingSeparate, 1.0).validationMetric;
   CHECK_APPROX(validationMetric, 0.0);

   double termScore;
   termScore = test.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore, 20.0);
   termScore = test.GetCurrentTermScore(0, {1}, 0);
   CHECK_APPROX(termScore, 20.0);
   termScore = test.GetCurrentTermScore(0, {2}, 0);
   CHECK_APPROX(termScore, 30.0);
}

TEST_CASE("missing separate continuous, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(4, true, false, false)},
         {{0}},
         {
               TestSample({0}, 10.0),
               TestSample({1}, 20.0),
               TestSample({2}, 30.0),
         },
         {TestSample({1}, 20.0)});

   // boost continuous missing separate
   double validationMetric = test.Boost(0, TermBoostFlags_MissingSeparate, 1.0).validationMetric;
   CHECK_APPROX(validationMetric, 0.0);

   double termScore;
   termScore = test.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore, 10.0);
   termScore = test.GetCurrentTermScore(0, {1}, 0);
   CHECK_APPROX(termScore, 20.0);
   termScore = test.GetCurrentTermScore(0, {2}, 0);
   CHECK_APPROX(termScore, 30.0);
}

TEST_CASE("missing category nominal, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(4, true, false, true)},
         {{0}},
         {
               TestSample({0}, 10.0),
               TestSample({1}, 20.0),
               TestSample({2}, 30.0),
         },
         {TestSample({1}, 20.0)});

   // boost nominal missing separate
   double validationMetric = test.Boost(0, TermBoostFlags_Default, 1.0).validationMetric;
   CHECK_APPROX(validationMetric, 0.0);

   double termScore;
   termScore = test.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore, 10.0);
   termScore = test.GetCurrentTermScore(0, {1}, 0);
   CHECK_APPROX(termScore, 20.0);
   termScore = test.GetCurrentTermScore(0, {2}, 0);
   CHECK_APPROX(termScore, 30.0);
}

TEST_CASE("missing + unseen continuous, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(2, true, true, false)},
         {{0}},
         {
               TestSample({0}, 10.0),
               TestSample({1}, 20.0),
         },
         {TestSample({1}, 20.0)});

   // boost nominal missing separate
   double validationMetric = test.Boost(0, TermBoostFlags_Default, 1.0).validationMetric;
   CHECK_APPROX(validationMetric, 0.0);

   double termScore;
   termScore = test.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore, 10.0);
   termScore = test.GetCurrentTermScore(0, {1}, 0);
   CHECK_APPROX(termScore, 20.0);
}

TEST_CASE("missing + unseen nominal, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(2, true, true, true)},
         {{0}},
         {
               TestSample({0}, 10.0),
               TestSample({1}, 20.0),
         },
         {TestSample({1}, 20.0)});

   // boost nominal missing separate
   double validationMetric = test.Boost(0, TermBoostFlags_Default, 1.0).validationMetric;
   CHECK_APPROX(validationMetric, 0.0);

   double termScore;
   termScore = test.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore, 10.0);
   termScore = test.GetCurrentTermScore(0, {1}, 0);
   CHECK_APPROX(termScore, 20.0);
}

TEST_CASE("missing category nominal, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(3, true, false, true)},
         {{0}},
         {
               TestSample({0}, 10.0),
               TestSample({1}, 20.0),
         },
         {TestSample({1}, 20.0)});

   // boost nominal missing separate
   double validationMetric = test.Boost(0, TermBoostFlags_Default, 1.0).validationMetric;
   CHECK_APPROX(validationMetric, 0.0);

   double termScore;
   termScore = test.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore, 10.0);
   termScore = test.GetCurrentTermScore(0, {1}, 0);
   CHECK_APPROX(termScore, 20.0);
   termScore = test.GetCurrentTermScore(0, {2}, 0);
   CHECK_APPROX(termScore, 20.0);
}

TEST_CASE("missing category nominal, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(3, false, true, true)},
         {{0}},
         {
               TestSample({1}, 20.0),
               TestSample({2}, 30.0),
         },
         {TestSample({1}, 20.0)});

   // boost nominal missing separate
   double validationMetric = test.Boost(0, TermBoostFlags_Default, 1.0).validationMetric;
   CHECK_APPROX(validationMetric, 0.0);

   double termScore;
   termScore = test.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore, 20.0);
   termScore = test.GetCurrentTermScore(0, {1}, 0);
   CHECK_APPROX(termScore, 20.0);
   termScore = test.GetCurrentTermScore(0, {2}, 0);
   CHECK_APPROX(termScore, 30.0);
}

TEST_CASE("missing category nominal, boosting, regression") {
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(3, true, true, true)},
         {{0}},
         {
               TestSample({0}, 10.0),
               TestSample({1}, 20.0),
               TestSample({2}, 30.0),
         },
         {TestSample({2}, 30.0)});

   // boost nominal missing separate
   double validationMetric = test.Boost(0, TermBoostFlags_Default, 1.0).validationMetric;
   CHECK_APPROX(validationMetric, 0.0);

   double termScore;
   termScore = test.GetCurrentTermScore(0, {0}, 0);
   CHECK_APPROX(termScore, 10.0);
   termScore = test.GetCurrentTermScore(0, {1}, 0);
   CHECK_APPROX(termScore, 20.0);
   termScore = test.GetCurrentTermScore(0, {2}, 0);
   CHECK_APPROX(termScore, 30.0);
}

static double RandomizedTesting(const AccelerationFlags acceleration) {
   const IntEbm cTrainSamples = 211; // have some non-SIMD residuals
   const IntEbm cValidationSamples = 101; // have some non-SIMD residuals
   const size_t cRounds = 200;

   auto rng = MakeRng(0);
   const std::vector<FeatureTest> features = {
         FeatureTest(10, false, false, false),
         FeatureTest(10, false, false, true),
         FeatureTest(10, false, true, false),
         FeatureTest(10, false, true, true),
         FeatureTest(10, true, false, false),
         FeatureTest(10, true, false, true),
         FeatureTest(10, true, true, false),
         FeatureTest(10, true, true, true),
   };
   auto terms = MakeMains(features);
   terms.push_back({0, 0});
   if(2 <= features.size()) {
      terms.push_back({0, 1});
      terms.push_back({1, 0});
   }
   if(3 <= features.size()) {
      // terms.push_back({0, 1, 2}); // TODO: enable when fast enough
   }
   if(4 <= features.size()) {
      // terms.push_back({0, 1, 2, 3}); // TODO: enable when fast enough
   }

   std::vector<IntEbm> boostFlagsAny{TermBoostFlags_PurifyGain,
         TermBoostFlags_DisableNewtonGain,
         TermBoostFlags_DisableCategorical,
         TermBoostFlags_PurifyUpdate,
         // TermBoostFlags_GradientSums, // does not return a metric
         TermBoostFlags_DisableNewtonUpdate,
         TermBoostFlags_RandomSplits};
   std::vector<IntEbm> boostFlagsChoose{
         TermBoostFlags_Default, TermBoostFlags_MissingLow, TermBoostFlags_MissingHigh, TermBoostFlags_MissingSeparate};

   double validationMetric = 1.0;
   for(IntEbm classesCount = Task_Regression; classesCount < 5; ++classesCount) {
      if(classesCount != Task_Regression && classesCount < 1) {
         continue;
      }
      const auto train = MakeRandomDataset(rng, classesCount, cTrainSamples, features);
      const auto validation = MakeRandomDataset(rng, classesCount, cValidationSamples, features);
      for(IntEbm innerBagCount = 0; innerBagCount < 3; ++innerBagCount) {
         TestBoost test = TestBoost(classesCount,
               features,
               terms,
               train,
               validation,
               innerBagCount,
               k_testCreateBoosterFlags_Default,
               acceleration);

         double validationMetricIteration = 0.0;
         for(size_t iRound = 0; iRound < cRounds; ++iRound) {
            for(IntEbm iTerm = 0; iTerm < static_cast<IntEbm>(terms.size()); ++iTerm) {
               const IntEbm cRealBins = features[terms[iTerm][0]].CountRealBins();
               const IntEbm cDimensions = terms[iTerm].size();

               const TermBoostFlags boostFlags =
                     static_cast<TermBoostFlags>(ChooseAny(rng, boostFlagsAny) | ChooseFrom(rng, boostFlagsChoose));

               const double learningRate = 0.015625;
               const IntEbm minSamplesLeaf = TestRand(rng, 5) + 1;
               const double minHessian = 0 == TestRand(rng, 5) ? 0.015625 : 0.0;
               const double regAlpha = 0 == TestRand(rng, 5) ? 0.015625 : 0.0;
               const double regLambda = 0 == TestRand(rng, 5) ? 0.015625 : 0.0;
               const double maxDeltaStep = 0 == TestRand(rng, 5) ? 1.0 : 0.0;
               const IntEbm minCategorySamples = TestRand(rng, 100);
               const double categoricalSmoothing = 10.0;
               const IntEbm maxCategoricalThreshold = 1 + TestRand(rng, cRealBins + 1);
               const double categoricalInclusionPercent = 0 == TestRand(rng, 2) ? 0.75 : 1.0;

               // we allow 1 cut more than the number of bins to test excessive leaves.
               const IntEbm cLeaves = 1 + TestRand(rng, cRealBins + 1);
               const std::vector<IntEbm> leaves(cDimensions, cLeaves);
               const MonotoneDirection direction =
                     0 == TestRand(rng, 5) ? static_cast<MonotoneDirection>(TestRand(rng, 2) * 2 - 1) : 0;
               const std::vector<MonotoneDirection> monotonicity(cDimensions, direction);

               validationMetricIteration = test.Boost(iTerm,
                                                     boostFlags,
                                                     learningRate,
                                                     minSamplesLeaf,
                                                     minHessian,
                                                     regAlpha,
                                                     regLambda,
                                                     maxDeltaStep,
                                                     minCategorySamples,
                                                     categoricalSmoothing,
                                                     maxCategoricalThreshold,
                                                     categoricalInclusionPercent,
                                                     leaves,
                                                     monotonicity)
                                                 .validationMetric;
            }
         }
         if(classesCount == 1) {
            if(std::numeric_limits<double>::infinity() != validationMetricIteration) {
               return -std::numeric_limits<double>::infinity();
            }
         } else {
            validationMetric *= validationMetricIteration;
         }
      }
   }
   return validationMetric;
}

TEST_CASE("stress test, boosting") {
   const double expected = 12286777380857.959;

   double validationMetricExact = RandomizedTesting(AccelerationFlags_NONE);
   CHECK(validationMetricExact == expected);

   double validationMetricSIMD = RandomizedTesting(AccelerationFlags_ALL);
   CHECK_APPROX_TOLERANCE(validationMetricSIMD, expected, 1e-2);
}
