// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"

static const TestPriority k_filePriority = TestPriority::DataSetShared;

TEST_CASE("data_set_shared, zero features, zero samples, regression") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   part = SizeDataSetHeader(0);
   CHECK(0 < part);
   sum += part;

   part = SizeRegressionTargets(0, nullptr);
   CHECK(0 < part);
   sum += part;

   std::vector<char> buffer(sum + 1, 77);
   buffer[sum] = 99;

   error = FillDataSetHeader(0, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillRegressionTargets(0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[sum]);
}

TEST_CASE("data_set_shared, zero features, three samples, regression") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   FloatEbmType targets[] { 0.3, 0.2, 0.1 };
   constexpr IntEbmType k_cSamples = sizeof(targets) / sizeof(targets[0]);

   part = SizeDataSetHeader(0);
   CHECK(0 < part);
   sum += part;

   part = SizeRegressionTargets(k_cSamples, targets);
   CHECK(0 < part);
   sum += part;

   std::vector<char> buffer(sum + 1, 77);
   buffer[sum] = 99;

   error = FillDataSetHeader(0, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillRegressionTargets(k_cSamples, targets, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[sum]);
}


TEST_CASE("data_set_shared, two features, zero samples, regression") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   part = SizeDataSetHeader(2);
   CHECK(0 < part);
   sum += part;

   part = SizeDataSetFeature(EBM_FALSE, 0, 0, nullptr);
   CHECK(0 < part);
   sum += part;

   part = SizeDataSetFeature(EBM_FALSE, 0, 0, nullptr);
   CHECK(0 < part);
   sum += part;

   part = SizeRegressionTargets(0, nullptr);
   CHECK(0 < part);
   sum += part;

   std::vector<char> buffer(sum + 1, 77);
   buffer[sum] = 99;

   error = FillDataSetHeader(2, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillDataSetFeature(EBM_FALSE, 0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillDataSetFeature(EBM_FALSE, 0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillRegressionTargets(0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[sum]);
}

TEST_CASE("data_set_shared, two features, 3 samples, regression") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;
   IntEbmType binnedData[] { 2, 1, 0 };
   constexpr IntEbmType k_cSamples = sizeof(binnedData) / sizeof(binnedData[0]);
   FloatEbmType targets[k_cSamples] { 0.3, 0.2, 0.1 };

   part = SizeDataSetHeader(2);
   CHECK(0 < part);
   sum += part;

   part = SizeDataSetFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0]);
   CHECK(0 < part);
   sum += part;

   part = SizeDataSetFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0]);
   CHECK(0 < part);
   sum += part;

   part = SizeRegressionTargets(k_cSamples, &targets[0]);
   CHECK(0 < part);
   sum += part;

   std::vector<char> buffer(sum + 1, 77);
   buffer[sum] = 99;

   error = FillDataSetHeader(2, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillDataSetFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillDataSetFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillRegressionTargets(k_cSamples, &targets[0], sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[sum]);
}

TEST_CASE("data_set_shared, zero features, zero samples, classification") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   part = SizeDataSetHeader(0);
   CHECK(0 < part);
   sum += part;

   part = SizeClassificationTargets(0, 0, nullptr);
   CHECK(0 < part);
   sum += part;

   std::vector<char> buffer(sum + 1, 77);
   buffer[sum] = 99;

   error = FillDataSetHeader(0, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillClassificationTargets(0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[sum]);
}

TEST_CASE("data_set_shared, zero features, three samples, classification") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   IntEbmType targets[] { 2, 1, 0 };
   constexpr IntEbmType k_cSamples = sizeof(targets) / sizeof(targets[0]);

   part = SizeDataSetHeader(0);
   CHECK(0 < part);
   sum += part;

   part = SizeClassificationTargets(3, k_cSamples, targets);
   CHECK(0 < part);
   sum += part;

   std::vector<char> buffer(sum + 1, 77);
   buffer[sum] = 99;

   error = FillDataSetHeader(0, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillClassificationTargets(3, k_cSamples, targets, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[sum]);
}

TEST_CASE("data_set_shared, two features, zero samples, classification") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   part = SizeDataSetHeader(2);
   CHECK(0 < part);
   sum += part;

   part = SizeDataSetFeature(EBM_FALSE, 0, 0, nullptr);
   CHECK(0 < part);
   sum += part;

   part = SizeDataSetFeature(EBM_FALSE, 0, 0, nullptr);
   CHECK(0 < part);
   sum += part;

   part = SizeClassificationTargets(0, 0, nullptr);
   CHECK(0 < part);
   sum += part;

   std::vector<char> buffer(sum + 1, 77);
   buffer[sum] = 99;

   error = FillDataSetHeader(2, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillDataSetFeature(EBM_FALSE, 0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillDataSetFeature(EBM_FALSE, 0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillClassificationTargets(0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[sum]);
}

TEST_CASE("data_set_shared, two features, 3 samples, classification") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;
   IntEbmType binnedData[] { 2, 1, 0 };
   constexpr IntEbmType k_cSamples = sizeof(binnedData) / sizeof(binnedData[0]);
   IntEbmType targets[k_cSamples] { 2, 1, 0 };

   part = SizeDataSetHeader(2);
   CHECK(0 < part);
   sum += part;

   part = SizeDataSetFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0]);
   CHECK(0 < part);
   sum += part;

   part = SizeDataSetFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0]);
   CHECK(0 < part);
   sum += part;

   part = SizeClassificationTargets(3, k_cSamples, &targets[0]);
   CHECK(0 < part);
   sum += part;

   std::vector<char> buffer(sum + 1, 77);
   buffer[sum] = 99;

   error = FillDataSetHeader(2, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillDataSetFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillDataSetFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillClassificationTargets(3, k_cSamples, &targets[0], sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[sum]);
}
