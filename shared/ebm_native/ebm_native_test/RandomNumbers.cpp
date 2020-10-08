// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderEbmNativeTest.h"

#include "ebm_native.h"
#include "EbmNativeTest.h"
#include "RandomStreamTest.h"

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

TEST_CASE("SamplingWithoutReplacement, stress test") {
   constexpr size_t cSamples = 1000;
   IntEbmType samples[cSamples];

   RandomStreamTest randomStream(k_randomSeed);
   if(!randomStream.IsSuccess()) {
      exit(1);
   }

   SeedEbmType randomSeed = k_randomSeed;
   SeedEbmType stageRandomizationMix = SeedEbmType { 34298572 };

   for(IntEbmType iRun = 0; iRun < 10000; ++iRun) {
      size_t cRandomSamples = randomStream.Next(cSamples + 1);
      size_t cIncluded = randomStream.Next(cRandomSamples + size_t { 1 });

      randomSeed = GenerateRandomNumber(randomSeed, stageRandomizationMix);

      SamplingWithoutReplacement(
         randomSeed,
         static_cast<IntEbmType>(cIncluded),
         static_cast<IntEbmType>(cRandomSamples),
         samples
      );

      size_t cIncludedVerified = 0;
      for(size_t i = 0; i < cRandomSamples; ++i) {
         const IntEbmType val = samples[i];
         CHECK(EBM_FALSE == val || EBM_TRUE == val);
         if(EBM_TRUE == val) {
            ++cIncludedVerified;
         }
      }
      CHECK(cIncludedVerified == cIncluded);
   }
}

TEST_CASE("test random number generator equivalency") {
   TestApi test = TestApi(2);
   test.AddFeatures({ FeatureTest(2) });
   test.AddFeatureGroups({ { 0 } });

   std::vector<ClassificationSample> samples;
   for(int i = 0; i < 1000; ++i) {
      samples.push_back(ClassificationSample(i % 2, { 0 == (i * 7) % 3 }));
   }

   test.AddTrainingSamples(samples);
   test.AddValidationSamples({ ClassificationSample(0, { 0 }), ClassificationSample(1, { 1 }) });

   test.InitializeBoosting(2);

   for(int iEpoch = 0; iEpoch < 100; ++iEpoch) {
      for(size_t iFeatureGroup = 0; iFeatureGroup < test.GetFeatureGroupsCount(); ++iFeatureGroup) {
         test.Boost(iFeatureGroup);
      }
   }

   FloatEbmType modelValue = test.GetCurrentModelPredictorScore(0, { 0 }, 1);
   // this is meant to be an exact check for this value.  We are testing here if we can generate identical results
   // accross different OSes and C/C++ libraries.  We specificed 2 inner samples, which will use the random generator
   // and if there are any differences between environments then this will catch those
   CHECK_APPROX(modelValue, 0.0057459461127468267);
}

