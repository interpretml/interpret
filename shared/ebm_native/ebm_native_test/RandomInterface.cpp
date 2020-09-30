// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderEbmNativeTest.h"

#include "ebm_native.h"
#include "EbmNativeTest.h"
#include "RandomStreamTest.h"

static const TestPriority k_filePriority = TestPriority::RandomInterface;

TEST_CASE("GenerateRandomNumber, 0") {
   IntEbmType ret = GenerateRandomNumber(0);
   CHECK(6689584010395485840 == ret);
}

TEST_CASE("GenerateRandomNumber, 2 (it gives us a negative return value)") {
   IntEbmType ret = GenerateRandomNumber(2);
   CHECK(-7665494278583557961 == ret);
}

TEST_CASE("GenerateRandomNumber, -1") {
   IntEbmType ret = GenerateRandomNumber(-1);
   CHECK(1020161130959650823 == ret);
}

TEST_CASE("GenerateRandomNumber, max") {
   IntEbmType ret = GenerateRandomNumber(std::numeric_limits<IntEbmType>::max());
   CHECK(1413621321926136738 == ret);
}

TEST_CASE("GenerateRandomNumber, lowest") {
   IntEbmType ret = GenerateRandomNumber(std::numeric_limits<IntEbmType>::lowest());
   CHECK(3582286234797358975 == ret);
}

TEST_CASE("SamplingWithoutReplacement, stress test") {
   constexpr size_t cSamples = 1000;
   IntEbmType samples[cSamples];

   RandomStreamTest randomStream(k_randomSeed);
   if(!randomStream.IsSuccess()) {
      exit(1);
   }

   IntEbmType randomSeed = k_randomSeed;

   for(IntEbmType iRun = 0; iRun < 10000; ++iRun) {
      IntEbmType cRandomSamples = randomStream.Next(cSamples + 1);
      IntEbmType cIncluded = randomStream.Next(cRandomSamples + 1);

      randomSeed = GenerateRandomNumber(randomSeed);

      SamplingWithoutReplacement(
         randomSeed,
         cIncluded,
         cRandomSamples,
         samples
      );

      size_t cIncludedVerified = 0;
      for(size_t i = 0; i < static_cast<size_t>(cRandomSamples); ++i) {
         const IntEbmType val = samples[i];
         CHECK(EBM_FALSE == val || EBM_TRUE == val);
         if(EBM_TRUE == val) {
            ++cIncludedVerified;
         }
      }
      CHECK(cIncludedVerified == static_cast<size_t>(cIncluded));
   }
}



