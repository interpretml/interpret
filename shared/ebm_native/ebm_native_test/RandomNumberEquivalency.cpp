// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeaderEbmNativeTest.h"

#include "ebm_native.h"
#include "EbmNativeTest.h"

static const TestPriority k_filePriority = TestPriority::RandomNumberEquivalency;

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
   CHECK_APPROX(modelValue, -0.021981997067385354);
}

