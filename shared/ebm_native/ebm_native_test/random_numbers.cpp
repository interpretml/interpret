// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"
#include "RandomStreamTest.hpp"

static const TestPriority k_filePriority = TestPriority::RandomNumbers;

TEST_CASE("GenerateSeed, 0 0") {
   SeedEbmType ret = GenerateSeed(0, 0);
   CHECK(1557540150 == ret);
}

TEST_CASE("GenerateSeed, 1 3 (it gives us a negative return value)") {
   SeedEbmType ret = GenerateSeed(1, 3);
   CHECK(-1784761967 == ret);
}

TEST_CASE("GenerateSeed, -1 0") {
   SeedEbmType ret = GenerateSeed(-1, 0);
   CHECK(237524772 == ret);
}

TEST_CASE("GenerateSeed, max") {
   SeedEbmType ret = GenerateSeed(std::numeric_limits<SeedEbmType>::max(), 0);
   CHECK(1266972904 == ret);
}

TEST_CASE("GenerateSeed, lowest") {
   SeedEbmType ret = GenerateSeed(std::numeric_limits<SeedEbmType>::lowest(), 0);
   CHECK(879100963 == ret);
}

TEST_CASE("SampleWithoutReplacementStratified, stress test") {
   constexpr size_t cSamples = 500;
   constexpr size_t cClasses = 10;

   ErrorEbmType error;

   IntEbmType targets[cSamples];
   BagEbmType sampleCounts[cSamples];
   size_t trainingCount[cClasses];
   size_t valCount[cClasses];
   size_t classCount[cClasses];

   RandomStreamTest randomStream(k_seed);
   if(!randomStream.IsSuccess()) {
      exit(1);
   }

   SeedEbmType seed = k_seed;

   for(IntEbmType iRun = 0; iRun < 10000; ++iRun) {
      size_t cRandomSamples = randomStream.Next(cSamples + 1);
      size_t cClassSize = randomStream.Next(cClasses) + 1;
      size_t cTrainingSamples = randomStream.Next(cRandomSamples + size_t { 1 });
      size_t cValidationSamples = cRandomSamples - cTrainingSamples;

      memset(trainingCount, 0, sizeof(trainingCount));
      memset(valCount, 0, sizeof(valCount));
      memset(classCount, 0, sizeof(classCount));

      ++seed;

      for(size_t iSample = 0; iSample < cRandomSamples; ++iSample) {
         size_t iRandom = randomStream.Next(cClassSize);
         IntEbmType targetClass = static_cast<IntEbmType>(iRandom);
         targets[iSample] = targetClass;
         ++classCount[targetClass];
      }

      error = SampleWithoutReplacementStratified(
         EBM_TRUE,
         seed,
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
         const IntEbmType targetClass = targets[i];
         const BagEbmType val = sampleCounts[i];
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
   ErrorEbmType error;

   for(int iBool = 0; iBool < 2; ++iBool) {
      constexpr size_t cSamples = 1000;
      BagEbmType samples[cSamples];

      RandomStreamTest randomStream(k_seed);
      if(!randomStream.IsSuccess()) {
         exit(1);
      }

      SeedEbmType seed = k_seed;

      for(IntEbmType iRun = 0; iRun < 10000; ++iRun) {
         size_t cRandomSamples = randomStream.Next(cSamples + 1);
         size_t cTrainingSamples = randomStream.Next(cRandomSamples + size_t { 1 });
         size_t cValidationSamples = cRandomSamples - cTrainingSamples;

         ++seed;
         error = SampleWithoutReplacement(
            static_cast<BoolEbmType>(iBool), 
            seed,
            static_cast<IntEbmType>(cTrainingSamples),
            static_cast<IntEbmType>(cValidationSamples),
            samples
         );
         CHECK(Error_None == error);

         size_t cTrainingSamplesVerified = 0;
         size_t cValidationSamplesVerified = 0;
         for(size_t i = 0; i < cRandomSamples; ++i) {
            const BagEbmType val = samples[i];
            CHECK(-1 == val || 1 == val);
            if(0 < val) {
               ++cTrainingSamplesVerified;
            }
            if(val < 0) {
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

   CHECK_APPROX(termScore, 0.31169469451667819);
}

TEST_CASE("GenerateGaussianRandom") {
   constexpr int cIterations = 1000;
   constexpr int offset = 0;

   for(int iBool = 0; iBool < 2; ++iBool) {
      size_t cNegative = 0;

      double avg = 0;
      double avgAbs = 0;

      double stddev = 10.0;
      for(int i = 0; i < cIterations; ++i) {
         double result;
         GenerateGaussianRandom(static_cast<BoolEbmType>(iBool), static_cast<SeedEbmType>(i + offset), stddev, 1, &result);
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
