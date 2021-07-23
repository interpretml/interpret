// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"
#include "RandomStreamTest.hpp"

static const TestPriority k_filePriority = TestPriority::RandomNumbers;

TEST_CASE("GenerateRandomNumber, 0 0") {
   SeedEbmType ret = GenerateRandomNumber(0, 0);
   CHECK(1557540150 == ret);
}

TEST_CASE("GenerateRandomNumber, 1 3 (it gives us a negative return value)") {
   SeedEbmType ret = GenerateRandomNumber(1, 3);
   CHECK(-1784761967 == ret);
}

TEST_CASE("GenerateRandomNumber, -1 0") {
   SeedEbmType ret = GenerateRandomNumber(-1, 0);
   CHECK(237524772 == ret);
}

TEST_CASE("GenerateRandomNumber, max") {
   SeedEbmType ret = GenerateRandomNumber(std::numeric_limits<SeedEbmType>::max(), 0);
   CHECK(1266972904 == ret);
}

TEST_CASE("GenerateRandomNumber, lowest") {
   SeedEbmType ret = GenerateRandomNumber(std::numeric_limits<SeedEbmType>::lowest(), 0);
   CHECK(879100963 == ret);
}

TEST_CASE("StratifiedSamplingWithoutReplacement, stress test") {
   constexpr size_t cSamples = 500;
   IntEbmType targets[cSamples];
   IntEbmType sampleCounts[cSamples];
   constexpr size_t cClasses = 10;
   size_t trainingCount[cClasses];
   size_t valCount[cClasses];
   size_t classCount[cClasses];

   RandomStreamTest randomStream(k_randomSeed);
   if (!randomStream.IsSuccess()) {
      exit(1);
   }

   SeedEbmType randomSeed = k_randomSeed;

   for (IntEbmType iRun = 0; iRun < 10000; ++iRun) {
      size_t cRandomSamples = randomStream.Next(cSamples + 1);
      size_t cClassSize = randomStream.Next(cClasses) + 1;
      size_t cTrainingSamples = randomStream.Next(cRandomSamples + size_t{ 1 });
      size_t cValidationSamples = cRandomSamples - cTrainingSamples;

      memset(trainingCount, 0, sizeof(trainingCount));
      memset(valCount, 0, sizeof(valCount));
      memset(classCount, 0, sizeof(classCount));

      ++randomSeed;

      for (size_t iSample = 0; iSample < cRandomSamples; ++iSample) {
         size_t iRandom = randomStream.Next(cClassSize);
         IntEbmType targetClass = static_cast<IntEbmType>(iRandom);
         targets[iSample] = targetClass;
         ++classCount[targetClass];
      }

      const ErrorEbmType error = StratifiedSamplingWithoutReplacement(
         randomSeed,
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
      for (size_t i = 0; i < cRandomSamples; ++i) {
         const IntEbmType targetClass = targets[i];
         const IntEbmType val = sampleCounts[i];
         CHECK(-1 == val || 1 == val);
         if (val == 1) {
            ++cTrainingSamplesVerified;
            ++trainingCount[targetClass];
         }
         if (val == -1) {
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
      for (size_t iClass = 0; iClass < cClassSize; ++iClass) {
         const double cTrainingPerClass = idealTrainSplit * classCount[iClass];
         const double cValidationPerClass = (1 - idealTrainSplit) * classCount[iClass];
         if (cTrainingPerClass < 1 || cValidationPerClass < 1) {
            checkProportions = false;
         }
      }

      size_t cLower = 0;
      size_t cHigher = 0;

      for (size_t iClass = 0; iClass < cClassSize; ++iClass) {
         CHECK(trainingCount[iClass] + valCount[iClass] == classCount[iClass]);

         if (classCount[iClass] == 0) {
            continue;
         }

         const double actualTrainSplit = trainingCount[iClass] / static_cast<double>(classCount[iClass]);

         cHigher = (idealTrainSplit <= actualTrainSplit) ? ++cHigher : cHigher;
         cLower = (idealTrainSplit >= actualTrainSplit) ? ++cLower : cLower;
         
         if (cClassSize < cTrainingSamples) {
            // Test (2)
            if (classCount[iClass] == 1) {
               CHECK(trainingCount[iClass] == 1 && valCount[iClass] == 0);
            }
            // Test (3)
            else if (cClassSize < cValidationSamples && classCount[iClass] == 2) {
               CHECK(trainingCount[iClass] == 1 && valCount[iClass] == 1);
            }
         }

         // Test (4)
         // Note: never more than 1 off
         if (checkProportions) {
            const double cTrainIdeal = classCount[iClass] * idealTrainSplit;
            const double cValIdeal = classCount[iClass] * (1 - idealTrainSplit);

            if (idealTrainSplit > actualTrainSplit) {
               CHECK(static_cast<size_t>(std::floor(cTrainIdeal)) == trainingCount[iClass]);
               CHECK(static_cast<size_t>(std::ceil(cValIdeal)) == valCount[iClass]);
            }
            else if (idealTrainSplit < actualTrainSplit) {
               CHECK(static_cast<size_t>(std::ceil(cTrainIdeal)) == trainingCount[iClass]);
               CHECK(static_cast<size_t>(std::floor(cValIdeal)) == valCount[iClass]);
            }
            else {
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
   constexpr size_t cSamples = 1000;
   IntEbmType samples[cSamples];

   RandomStreamTest randomStream(k_randomSeed);
   if(!randomStream.IsSuccess()) {
      exit(1);
   }

   SeedEbmType randomSeed = k_randomSeed;

   for(IntEbmType iRun = 0; iRun < 10000; ++iRun) {
      size_t cRandomSamples = randomStream.Next(cSamples + 1);
      size_t cTrainingSamples = randomStream.Next(cRandomSamples + size_t { 1 });
      size_t cValidationSamples = cRandomSamples - cTrainingSamples;

      ++randomSeed;
      SampleWithoutReplacement(
         randomSeed,
         static_cast<IntEbmType>(cTrainingSamples),
         static_cast<IntEbmType>(cValidationSamples),
         samples
      );

      size_t cTrainingSamplesVerified = 0;
      size_t cValidationSamplesVerified = 0;
      for(size_t i = 0; i < cRandomSamples; ++i) {
         const IntEbmType val = samples[i];
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

TEST_CASE("test random number generator equivalency") {
   TestApi test = TestApi(k_learningTypeRegression);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureGroups({ { 0 } });

   std::vector<TestSample> samples;
   for(int i = 0; i < 1000; ++i) {
      samples.push_back(TestSample({ 0 == (i * 7) % 3 }, i % 2));
   }

   test.AddTrainingSamples(samples);
   test.AddValidationSamples({ TestSample({ 0 }, 0), TestSample({ 1 }, 1) });

   test.InitializeBoosting(2);

   for(int iEpoch = 0; iEpoch < 100; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         test.Boost(iFeatureGroup);
      }
   }

   FloatEbmType modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 0);
   // this is meant to be an exact check for this value.  We are testing here if we can generate identical results
   // accross different OSes and C/C++ libraries.  We specificed 2 inner samples, which will use the random generator
   // and if there are any differences between environments then this will catch those

   CHECK_APPROX(modelValue, 0.31169469451667819);
}

