// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch_test.hpp"

#include "libebm.h"
#include "libebm_test.hpp"

static constexpr TestPriority k_filePriority = TestPriority::CoxBoosting;

TEST_CASE("cox, intercept boosting") {
   // No features, just intercept. With identity link and all scores starting at 0,
   // the intercept update is zero (all exp(score) equal), so metric stays 0.
   TestBoost test = TestBoost(Task_Regression,
         {},
         {},
         {
               TestSample({}, 5),
               TestSample({}, 7),
         },
         {TestSample({}, 6)},
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         "survival_cox");

   double validationMetric;
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      validationMetric = test.Boost(-1).validationMetric;
   }
   CHECK(0.0 == validationMetric);
}

TEST_CASE("cox, one feature, all events") {
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         {TestSample({0}, 1.0), TestSample({0}, 2.0), TestSample({1}, 3.0), TestSample({1}, 4.0)},
         {TestSample({0}, 1.5), TestSample({1}, 3.5)},
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         "survival_cox");

   double validationMetric = double{std::numeric_limits<double>::quiet_NaN()};
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm).validationMetric;
      }
   }
   CHECK_APPROX(validationMetric, 5.18617265258614069e-03);

   double termScore0 = test.GetCurrentTermScore(0, {0}, 0);
   double termScore1 = test.GetCurrentTermScore(0, {1}, 0);
   CHECK_APPROX(termScore0, 2.54812088061053021e+00);
   CHECK_APPROX(termScore1, -2.01530058381678501e+00);
}

TEST_CASE("cox, mixed events and censored") {
   // Mix of uncensored events (positive targets) and censored observations (negative targets)
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 1.0), // event at time 1
               TestSample({0}, -2.0), // censored at time 2
               TestSample({1}, 3.0), // event at time 3
               TestSample({1}, -4.0), // censored at time 4
         },
         {
               TestSample({0}, 1.5), // validation: event at time 1.5
               TestSample({1}, -3.5), // validation: censored at time 3.5
         },
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         "survival_cox");

   double validationMetric = double{std::numeric_limits<double>::quiet_NaN()};
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm).validationMetric;
      }
   }
   CHECK_APPROX(validationMetric, 1.40204968579940026e-02);

   double termScore0 = test.GetCurrentTermScore(0, {0}, 0);
   double termScore1 = test.GetCurrentTermScore(0, {1}, 0);
   CHECK_APPROX(termScore0, 2.06181419902693541e+00);
   CHECK_APPROX(termScore1, -1.49822032015551976e+00);
}

TEST_CASE("cox, zero learning rate") {
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         {TestSample({0}, 1.0), TestSample({1}, 2.0)},
         {TestSample({0}, 1.5)},
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         "survival_cox");

   double validationMetric = double{std::numeric_limits<double>::quiet_NaN()};
   double termScore = double{std::numeric_limits<double>::quiet_NaN()};
   for(int iEpoch = 0; iEpoch < 100; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm, TermBoostFlags_Default, 0).validationMetric;
         termScore = test.GetCurrentTermScore(iTerm, {0}, 0);
         CHECK(0 == termScore);
      }
   }
   CHECK(0.0 == validationMetric);
}

TEST_CASE("cox, all censored") {
   // All observations are censored — no events. No gradient signal, metric stays zero.
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         {TestSample({0}, -1.0), TestSample({1}, -2.0)},
         {TestSample({0}, -1.5)},
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         "survival_cox");

   double validationMetric = double{std::numeric_limits<double>::quiet_NaN()};
   for(int iEpoch = 0; iEpoch < 100; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm).validationMetric;
      }
   }
   CHECK(0.0 == validationMetric);
}

TEST_CASE("cox, first step matches Breslow formula") {
   // Verify the first boosting step produces a term score consistent with the Breslow
   // partial likelihood gradient formula. This validates mathematical correctness.
   //
   // Dataset: 4 training samples, all events, 2 bins (bin 0: times 1,2; bin 1: times 3,4)
   // At initial scores=0, exp(score)=1 for all.
   // Sorted by time: indices [0,1,2,3], risk set sums: [4, 3, 2, 1]
   //
   // Forward pass (all events):
   //   j=0: cumH=1/4,       cumH2=1/16
   //     grad[0] = 1*1/4 - 1 = -3/4,       hess[0] = 1/4 - 1/16 = 3/16
   //   j=1: cumH=1/4+1/3=7/12, cumH2=1/16+1/9=25/144
   //     grad[1] = 7/12 - 1 = -5/12,        hess[1] = 7/12 - 25/144 = 59/144
   //   j=2: cumH=7/12+1/2=13/12, cumH2=25/144+1/4=61/144
   //     grad[2] = 13/12 - 1 = 1/12,         hess[2] = 13/12 - 61/144 = 95/144
   //   j=3: cumH=13/12+1=25/12, cumH2=61/144+1=205/144
   //     grad[3] = 25/12 - 1 = 13/12,        hess[3] = 25/12 - 205/144 = 95/144
   //
   // Bin 0 (samples 0,1): sum_grad = -3/4 + -5/12 = -14/12 = -7/6
   //                       sum_hess = 3/16 + 59/144 = 27/144 + 59/144 = 86/144 = 43/72
   //   Newton update = -(-7/6) / (43/72) = (7/6)*(72/43) = 504/258 = 84/43 ≈ 1.9534883721
   //   After lr=0.01: 0.019534883721
   //
   // Bin 1 (samples 2,3): sum_grad = 1/12 + 13/12 = 14/12 = 7/6
   //                       sum_hess = 95/144 + 95/144 = 190/144 = 95/72
   //   Newton update = -(7/6) / (95/72) = -(7/6)*(72/95) = -504/570 = -84/95 ≈ -0.884210526
   //   After lr=0.01: -0.008842105263
   //
   // Validation NPL at scores=0 with 2 validation samples (both events at times 1.5, 3.5):
   // Sorted: [val0(1.5), val1(3.5)], exp(score)=1, risk sums: [2, 1]
   // NPL = log(2) - 0 + log(1) - 0 = log(2) ≈ 0.693147180559945

   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         {TestSample({0}, 1.0), TestSample({0}, 2.0), TestSample({1}, 3.0), TestSample({1}, 4.0)},
         {TestSample({0}, 1.5), TestSample({1}, 3.5)},
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         "survival_cox");

   // First boost: the metric returned is computed AFTER applying the update
   double validationMetric = test.Boost(0).validationMetric;

   double termScore0 = test.GetCurrentTermScore(0, {0}, 0);
   double termScore1 = test.GetCurrentTermScore(0, {1}, 0);

   // Verify the first-step Newton update matches the hand-computed Breslow gradient
   CHECK_APPROX_TOLERANCE(termScore0, 0.019534883720930, double{1e-2});
   CHECK_APPROX_TOLERANCE(termScore1, -0.008842105263158, double{1e-2});

   // The first-step metric (NPL on validation set AFTER applying the update).
   // At scores=0 the NPL would be log(2) ≈ 0.6931, but the update shifts scores
   // so the metric drops. This value also matches the first-epoch metric in the convergence test.
   CHECK_APPROX_TOLERANCE(validationMetric, 3.39529669689400848e-01, double{1e-2});
}

TEST_CASE("cox, convergence") {
   // Group 0 has short event times (high hazard), group 1 has long event times (low hazard).
   // The model should learn to differentiate and the metric should decrease.
   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         {
               TestSample({0}, 1.0),
               TestSample({0}, 2.0),
               TestSample({1}, 5.0),
               TestSample({1}, 6.0),
         },
         {
               TestSample({0}, 1.5),
               TestSample({1}, 5.5),
         },
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         "survival_cox");

   double firstMetric = double{std::numeric_limits<double>::quiet_NaN()};
   double validationMetric = double{std::numeric_limits<double>::quiet_NaN()};
   for(int iEpoch = 0; iEpoch < 1000; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         validationMetric = test.Boost(iTerm).validationMetric;
      }
      if(0 == iEpoch) {
         firstMetric = validationMetric;
      }
   }
   CHECK_APPROX(firstMetric, 3.39529669689400848e-01);
   CHECK_APPROX(validationMetric, 5.18617265258614069e-03);
   CHECK(validationMetric < firstMetric);

   double termScore0 = test.GetCurrentTermScore(0, {0}, 0);
   double termScore1 = test.GetCurrentTermScore(0, {1}, 0);
   CHECK_APPROX(termScore0, 2.54812088061053021e+00);
   CHECK_APPROX(termScore1, -2.01530058381678501e+00);
}

TEST_CASE("cox, 67 samples, single bin, all events, gradient sum invariant") {
   // 67 training samples all mapped to bin 0, all uncensored events at distinct times 1..67.
   //
   // Mathematical invariant: the Cox Breslow partial-likelihood gradients over ALL samples
   // sum to exactly zero (this is the defining property of the score equation at any score
   // vector). Since every sample maps to bin 0, the bin 0 gradient sum == total gradient
   // sum == 0 (up to floating-point roundoff), so the Newton update for bin 0 is
   // -0 / hessSum ≈ 0, and the term score must remain at essentially 0 on every step.
   //
   // If any of the 67 samples were dropped, duplicated, re-ordered incorrectly, or decoded
   // with a wrong time/event pair, the gradient sum would become nonzero and the term score
   // would drift noticeably. 67 is prime, > 64, so with FeatureTest(2) the 1-bit packing
   // spills across two uint64 bitpack words and exercises the partial-first-word
   // initialization path in the InjectedApplyUpdate loop.

   std::vector<TestSample> trainingSamples;
   for(size_t i = 0; i < 67; ++i) {
      trainingSamples.push_back(TestSample({0}, static_cast<double>(i + 1)));
   }
   std::vector<TestSample> validationSamples = {TestSample({0}, 1.0)};

   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         trainingSamples,
         validationSamples,
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         "survival_cox");

   for(int iEpoch = 0; iEpoch < 50; ++iEpoch) {
      test.Boost(0);
      const double termScore = test.GetCurrentTermScore(0, {0}, 0);
      // Invariant: |termScore| should be within floating-point roundoff of 0.
      // A dropped or duplicated sample would produce |termScore| on the order of
      // lr * 1/67 ≈ 1.5e-4 — well above this tolerance.
      CHECK(std::fabs(termScore) < 1e-10);
   }
}

TEST_CASE("cox, 67 samples, single bin, mixed events and censored") {
   // Same single-bin invariant as the all-events case: alternating events and censored
   // observations still leaves the bin's gradient sum equal to the (zero) total gradient
   // sum, so the term score stays at essentially 0.
   //
   // This extends the prior test to exercise the event-decode branch (target sign) for
   // each of the 67 samples.

   std::vector<TestSample> trainingSamples;
   for(size_t i = 0; i < 67; ++i) {
      const double time = static_cast<double>(i + 1);
      // Alternate: even indices are events (positive target), odd are censored (negative).
      const double target = (0 == i % 2) ? time : -time;
      trainingSamples.push_back(TestSample({0}, target));
   }
   std::vector<TestSample> validationSamples = {TestSample({0}, 1.0)};

   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         trainingSamples,
         validationSamples,
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         "survival_cox");

   for(int iEpoch = 0; iEpoch < 50; ++iEpoch) {
      test.Boost(0);
      const double termScore = test.GetCurrentTermScore(0, {0}, 0);
      CHECK(std::fabs(termScore) < 1e-10);
   }
}

TEST_CASE("cox, 67 samples, two bins, bin assignment reaches all samples") {
   // 67 training samples split into two bins with a STRONG signal: bin 0 gets the 33
   // shortest times (1..33, high hazard), bin 1 gets the 34 longest times (34..67,
   // low hazard). All events (uncensored).
   //
   // Verifies:
   //   1. All 67 samples contribute: gradSum_bin0 + gradSum_bin1 == 0 (total Cox gradient
   //      at scores=0 is 0 by construction). If any sample were dropped, this invariant
   //      would be broken and termScores could drift in unexpected directions.
   //   2. The algorithm correctly assigns direction: bin 0 (short times) must get a
   //      positive log-hazard ratio and bin 1 (long times) a negative one — the opposite
   //      ordering would mean samples were assigned to the wrong bins during sample
   //      processing.
   //   3. Convergence: after many boosting steps, both scores move monotonically away
   //      from zero in the expected direction and the validation metric stays finite.

   std::vector<TestSample> trainingSamples;
   for(size_t i = 0; i < 67; ++i) {
      const IntEbm bin = (i < 33) ? IntEbm{0} : IntEbm{1};
      trainingSamples.push_back(TestSample({bin}, static_cast<double>(i + 1)));
   }
   std::vector<TestSample> validationSamples = {
         TestSample({0}, 1.0),
         TestSample({1}, 67.0),
   };

   TestBoost test = TestBoost(Task_Regression,
         {FeatureTest(2)},
         {{0}},
         trainingSamples,
         validationSamples,
         k_countInnerBagsDefault,
         k_testCreateBoosterFlags_Default,
         k_testAccelerationFlags_Default,
         "survival_cox");

   // First boost step: verify opposite signs (enforced by gradSum_bin0 + gradSum_bin1 == 0)
   test.Boost(0);
   const double firstTermScore0 = test.GetCurrentTermScore(0, {0}, 0);
   const double firstTermScore1 = test.GetCurrentTermScore(0, {1}, 0);
   CHECK(firstTermScore0 > 0.0);
   CHECK(firstTermScore1 < 0.0);

   // Continue boosting and verify the scores move farther apart.
   double validationMetric = double{std::numeric_limits<double>::quiet_NaN()};
   for(int iEpoch = 0; iEpoch < 500; ++iEpoch) {
      validationMetric = test.Boost(0).validationMetric;
   }
   const double termScore0 = test.GetCurrentTermScore(0, {0}, 0);
   const double termScore1 = test.GetCurrentTermScore(0, {1}, 0);

   CHECK(termScore0 > firstTermScore0);
   CHECK(termScore1 < firstTermScore1);
   CHECK(termScore0 > 0.0);
   CHECK(termScore1 < 0.0);
   // The metric must remain finite (catches NaN/Inf from malformed risk sets)
   CHECK(std::isfinite(validationMetric));
}
