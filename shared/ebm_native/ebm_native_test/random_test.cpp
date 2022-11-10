// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"
#include "RandomStreamTest.hpp"

static constexpr TestPriority k_filePriority = TestPriority::RandomNumbers;

TEST_CASE("InitRNG, 0") {
   IntEbm cBytes = MeasureRNG();
   std::vector<unsigned char> rng(static_cast<size_t>(cBytes));
   InitRNG(0, &rng[0]);
   CHECK(true); // just check if it crashes
}

TEST_CASE("InitRNG, 2147483647") {
   IntEbm cBytes = MeasureRNG();
   std::vector<unsigned char> rng(static_cast<size_t>(cBytes));
   InitRNG(2147483647, &rng[0]);
   CHECK(true); // just check if it crashes
}

TEST_CASE("InitRNG, -2147483648") {
   IntEbm cBytes = MeasureRNG();
   std::vector<unsigned char> rng(static_cast<size_t>(cBytes));
   InitRNG(-2147483648, &rng[0]);
   CHECK(true); // just check if it crashes
}

TEST_CASE("CopyRNG, 1") {
   IntEbm cBytes = MeasureRNG();
   std::vector<unsigned char> rng(static_cast<size_t>(cBytes));
   InitRNG(1, &rng[0]);
   std::vector<unsigned char> rngCopy(static_cast<size_t>(cBytes));
   CopyRNG(&rng[0], &rngCopy[0]);
   CHECK(true); // just check if it crashes
}

TEST_CASE("BranchRNG, -1") {
   IntEbm cBytes = MeasureRNG();
   std::vector<unsigned char> rng(static_cast<size_t>(cBytes));
   InitRNG(-1, &rng[0]);
   std::vector<unsigned char> rngBranch(static_cast<size_t>(cBytes));
   BranchRNG(&rng[0], &rngBranch[0]);
   CHECK(true); // just check if it crashes
}

TEST_CASE("GenerateSeed, 99") {
   IntEbm cBytes = MeasureRNG();
   std::vector<unsigned char> rng(static_cast<size_t>(cBytes));
   InitRNG(99, &rng[0]);
   SeedEbm seed;
   GenerateSeed(&rng[0], &seed);
   CHECK(-1237406560 == seed);
}

TEST_CASE("SampleWithoutReplacementStratified, 0 samples") {
   static constexpr size_t cClasses = 2;

   ErrorEbm error;

   error = SampleWithoutReplacementStratified(
      nullptr,
      cClasses,
      0,
      0,
      nullptr,
      nullptr
   );
   CHECK(Error_None == error);
}

TEST_CASE("SampleWithoutReplacementStratified, 0 training samples") {
   static constexpr size_t cSamples = 2;
   static constexpr size_t cClasses = 2;

   ErrorEbm error;

   IntEbm targets[cSamples] = { 0 };
   BagEbm sampleCounts[cSamples];

   error = SampleWithoutReplacementStratified(
      nullptr,
      cClasses,
      0,
      2,
      targets,
      sampleCounts
   );
   CHECK(Error_None == error);

   CHECK(BagEbm { -1 } == sampleCounts[0]);
   CHECK(BagEbm { -1 } == sampleCounts[1]);
}

TEST_CASE("SampleWithoutReplacementStratified, 0 validation samples") {
   static constexpr size_t cSamples = 2;
   static constexpr size_t cClasses = 2;

   ErrorEbm error;

   IntEbm targets[cSamples] = { 1 };
   BagEbm sampleCounts[cSamples];

   error = SampleWithoutReplacementStratified(
      nullptr,
      cClasses,
      2,
      0,
      targets,
      sampleCounts
   );
   CHECK(Error_None == error);

   CHECK(BagEbm { 1 } == sampleCounts[0]);
   CHECK(BagEbm { 1 } == sampleCounts[1]);
}

TEST_CASE("SampleWithoutReplacementStratified, monoclassification") {
   static constexpr size_t cSamples = 2;
   static constexpr size_t cClasses = 1;

   ErrorEbm error;

   IntEbm targets[cSamples] = { 0 };
   BagEbm sampleCounts[cSamples];

   error = SampleWithoutReplacementStratified(
      nullptr,
      cClasses,
      1,
      1,
      targets,
      sampleCounts
   );
   CHECK(Error_None == error);

   CHECK(BagEbm { -1 } == sampleCounts[0] || BagEbm { 1 } == sampleCounts[0]);
   CHECK(BagEbm { -1 } == sampleCounts[1] || BagEbm { 1 } == sampleCounts[1]);

   CHECK(BagEbm { 0 } == sampleCounts[0] + sampleCounts[1]);
}

TEST_CASE("SampleWithoutReplacementStratified, stress test") {
   static constexpr size_t cSamples = 500;
   static constexpr size_t cClasses = 10;

   ErrorEbm error;

   RandomStreamTest randomStream(k_seed);
   if(!randomStream.IsSuccess()) {
      exit(1);
   }

   SeedEbm seed = k_seed;

   for(IntEbm iRun = 0; iRun < 10000; ++iRun) {
      IntEbm targets[cSamples] = { 0 };
      BagEbm sampleCounts[cSamples];

      size_t classCount[cClasses] = { 0 };
      size_t trainingCount[cClasses] = { 0 };
      size_t valCount[cClasses] = { 0 };

      size_t cRandomSamples = randomStream.Next(cSamples + 1);
      size_t cClassSize = randomStream.Next(cClasses) + 1;
      size_t cTrainingSamples = randomStream.Next(cRandomSamples + size_t { 1 });
      EBM_ASSERT(0 <= cTrainingSamples && cTrainingSamples <= cSamples);
      size_t cValidationSamples = cRandomSamples - cTrainingSamples;
      EBM_ASSERT(0 <= cValidationSamples && cValidationSamples <= cSamples);

      ++seed;

      for(size_t iSample = 0; iSample < cRandomSamples; ++iSample) {
         size_t iRandom = randomStream.Next(cClassSize);
         IntEbm targetClass = static_cast<IntEbm>(iRandom);
         targets[iSample] = targetClass;
         ++classCount[targetClass];
      }

      std::vector<unsigned char> rng(static_cast<size_t>(MeasureRNG()));
      InitRNG(k_seed, &rng[0]);

      error = SampleWithoutReplacementStratified(
         &rng[0],
         cClassSize,
         cTrainingSamples,
         cValidationSamples,
         targets,
         sampleCounts
      );
      CHECK(Error_None == error);

      // Check the overall correct number of training/validation samples have been returned
      size_t cTrainingSamplesVerified = 0;
      size_t cValidationSamplesVerified = 0;
      for(size_t i = 0; i < cRandomSamples; ++i) {
         const IntEbm targetClass = targets[i];
         const BagEbm val = sampleCounts[i];
         CHECK(-1 == val || 1 == val);
         if(val == 1) {
            ++cTrainingSamplesVerified;
            ++trainingCount[targetClass];
         }
         if(val == -1) {
            ++cValidationSamplesVerified;
            ++valCount[targetClass];
         }
      }
      CHECK(cTrainingSamplesVerified == cTrainingSamples);
      CHECK(cValidationSamplesVerified == cValidationSamples);

      // This stratified sampling algorithm guarantees:
      // (1) Either the train/validation counts work out perfectly for each class -or- there is at 
      //     least one class with a count above the ideal training count and at least one class with
      //     a training count below the ideal count,
      // (2) Given a sufficient amount of training samples, if a class has only one sample, it 
      //     should go to training,
      // (3) Given a sufficient amount of training samples, if a class only has two samples, one 
      //     should go to train and one should go to test,
      // (4) If a class has enough samples to hit the target train/validation count, its actual
      //     train/validation count should be no more than one away from the ideal count. 

      const double idealTrainSplit = static_cast<double>(cTrainingSamples) / (cTrainingSamples + cValidationSamples);

      // Should (4) be tested?
      bool checkProportions = true;
      for(size_t iClass = 0; iClass < cClassSize; ++iClass) {
         const double cTrainingPerClass = idealTrainSplit * classCount[iClass];
         const double cValidationPerClass = (1 - idealTrainSplit) * classCount[iClass];
         if(cTrainingPerClass < 1 || cValidationPerClass < 1) {
            checkProportions = false;
         }
      }

      size_t cLower = 0;
      size_t cHigher = 0;

      for(size_t iClass = 0; iClass < cClassSize; ++iClass) {
         CHECK(trainingCount[iClass] + valCount[iClass] == classCount[iClass]);

         if(classCount[iClass] == 0) {
            continue;
         }

         const double actualTrainSplit = trainingCount[iClass] / static_cast<double>(classCount[iClass]);

         cHigher = (idealTrainSplit <= actualTrainSplit) ? cHigher + 1 : cHigher;
         cLower = (idealTrainSplit >= actualTrainSplit) ? cLower + 1 : cLower;

         if(cClassSize < cTrainingSamples) {
            // Test (2)
            if(classCount[iClass] == 1) {
               CHECK(trainingCount[iClass] == 1 && valCount[iClass] == 0);
            }
            // Test (3)
            else if(cClassSize < cValidationSamples && classCount[iClass] == 2) {
               CHECK(trainingCount[iClass] == 1 && valCount[iClass] == 1);
            }
         }

         // Test (4)
         // Note: never more than 1 off
         if(checkProportions) {
            const double cTrainIdeal = classCount[iClass] * idealTrainSplit;
            const double cValIdeal = classCount[iClass] * (1 - idealTrainSplit);

            if(idealTrainSplit > actualTrainSplit) {
               CHECK(static_cast<size_t>(std::floor(cTrainIdeal)) == trainingCount[iClass]);
               CHECK(static_cast<size_t>(std::ceil(cValIdeal)) == valCount[iClass]);
            } else if(idealTrainSplit < actualTrainSplit) {
               CHECK(static_cast<size_t>(std::ceil(cTrainIdeal)) == trainingCount[iClass]);
               CHECK(static_cast<size_t>(std::floor(cValIdeal)) == valCount[iClass]);
            } else {
               CHECK_APPROX(static_cast<double>(trainingCount[iClass]), cTrainIdeal);
               CHECK_APPROX(static_cast<double>(valCount[iClass]), cValIdeal);
            }
         }
      }

      // Test (1)
      CHECK((cLower > 0 && cHigher > 0) || (cRandomSamples == 0));
   }
}

TEST_CASE("SampleWithoutReplacement, stress test") {
   ErrorEbm error;

   for(int iBool = 0; iBool < 2; ++iBool) {
      static constexpr size_t cSamples = 1000;
      BagEbm samples[cSamples];

      RandomStreamTest randomStream(k_seed);
      if(!randomStream.IsSuccess()) {
         exit(1);
      }

      std::vector<unsigned char> rng(static_cast<size_t>(MeasureRNG()));
      InitRNG(k_seed, &rng[0]);

      for(IntEbm iRun = 0; iRun < 10000; ++iRun) {
         size_t cRandomSamples = randomStream.Next(cSamples + 1);
         size_t cTrainingSamples = randomStream.Next(cRandomSamples + size_t { 1 });
         size_t cValidationSamples = cRandomSamples - cTrainingSamples;

         error = SampleWithoutReplacement(
            0 == iBool ? nullptr : &rng[0],
            static_cast<IntEbm>(cTrainingSamples),
            static_cast<IntEbm>(cValidationSamples),
            samples
         );
         CHECK(Error_None == error);

         size_t cTrainingSamplesVerified = 0;
         size_t cValidationSamplesVerified = 0;
         for(size_t i = 0; i < cRandomSamples; ++i) {
            const BagEbm val = samples[i];
            CHECK(BagEbm { -1 } == val || BagEbm { 1 } == val);
            if(BagEbm { 0 } < val) {
               ++cTrainingSamplesVerified;
            }
            if(val < BagEbm { 0 }) {
               ++cValidationSamplesVerified;
            }
         }
         CHECK(cTrainingSamplesVerified == cTrainingSamples);
         CHECK(cValidationSamplesVerified == cValidationSamples);
         CHECK(cTrainingSamplesVerified + cValidationSamplesVerified == cRandomSamples);
      }
   }
}

TEST_CASE("test random number generator equivalency") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddTerms({ { 0 } });

   std::vector<TestSample> samples;
   for(int i = 0; i < 1000; ++i) {
      samples.push_back(TestSample({ 0 == (i * 7) % 3 }, i % 2));
   }

   test.AddTrainingSamples(samples);
   test.AddValidationSamples({ TestSample({ 0 }, 0), TestSample({ 1 }, 1) });

   test.InitializeBoosting(2);

   for(int iEpoch = 0; iEpoch < 100; ++iEpoch) {
      for(size_t iTerm = 0; iTerm < test.GetCountTerms(); ++iTerm) {
         test.Boost(iTerm);
      }
   }

   double termScore = test.GetCurrentTermScore(0, { 0 }, 0);
   // this is meant to be an exact check for this value.  We are testing here if we can generate identical results
   // accross different OSes and C/C++ libraries.  We specificed 2 inner samples, which will use the random generator
   // and if there are any differences between environments then this will catch those

   CHECK_APPROX(termScore, 0.32364558747317862);
}

TEST_CASE("GenerateGaussianRandom") {
   static constexpr int cIterations = 1000;

   std::vector<unsigned char> rng(static_cast<size_t>(MeasureRNG()));
   InitRNG(k_seed, &rng[0]);

   for(int iBool = 0; iBool < 2; ++iBool) {
      size_t cNegative = 0;

      double avg = 0;
      double avgAbs = 0;

      double stddev = 10.0;
      for(int i = 0; i < cIterations; ++i) {
         double result;
         GenerateGaussianRandom(0 == iBool ? nullptr : &rng[0], stddev, 1, &result);
         if(result < 0) {
            ++cNegative;
         }
         avg += result;
         avgAbs += std::abs(result);
      }
      avg /= cIterations;
      avgAbs /= cIterations;

      // use better tests and improve these bounds
      CHECK(std::abs(avg) <= 1.5);
      CHECK(6.5 <= avgAbs && avgAbs <= 9.5);
      CHECK(300 <= cNegative && cNegative <= 700);
   }
}
