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

   part = SizeDataSetHeader(0, 0, 1);
   CHECK(0 <= part);
   sum += part;

   part = SizeRegressionTarget(0, nullptr);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(0, 0, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillRegressionTarget(0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("data_set_shared, zero features, three samples, regression") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   constexpr IntEbmType k_cSamples = 3;
   FloatEbmType targets[k_cSamples] { 0.3, 0.2, 0.1 };

   part = SizeDataSetHeader(0, 0, 1);
   CHECK(0 <= part);
   sum += part;

   part = SizeRegressionTarget(k_cSamples, targets);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(0, 0, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillRegressionTarget(k_cSamples, targets, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}


TEST_CASE("data_set_shared, two features, zero samples, regression") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   part = SizeDataSetHeader(2, 0, 1);
   CHECK(0 <= part);
   sum += part;

   part = SizeFeature(EBM_FALSE, 0, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = SizeFeature(EBM_FALSE, 0, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = SizeRegressionTarget(0, nullptr);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(2, 0, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(EBM_FALSE, 0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(EBM_FALSE, 0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillRegressionTarget(0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("data_set_shared, two features, 3 samples, regression") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;
   constexpr IntEbmType k_cSamples = 3;
   IntEbmType binnedData[k_cSamples] { 2, 1, 0 };
   FloatEbmType targets[k_cSamples] { 0.3, 0.2, 0.1 };

   part = SizeDataSetHeader(2, 0, 1);
   CHECK(0 <= part);
   sum += part;

   part = SizeFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0]);
   CHECK(0 <= part);
   sum += part;

   part = SizeFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0]);
   CHECK(0 <= part);
   sum += part;

   part = SizeRegressionTarget(k_cSamples, &targets[0]);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(2, 0, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillRegressionTarget(k_cSamples, &targets[0], sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("data_set_shared, zero features, zero samples, classification") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   part = SizeDataSetHeader(0, 0, 1);
   CHECK(0 <= part);
   sum += part;

   part = SizeClassificationTarget(0, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(0, 0, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillClassificationTarget(0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("data_set_shared, zero features, three samples, classification") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   constexpr IntEbmType k_cSamples = 3;
   IntEbmType targets[k_cSamples] { 2, 1, 0 };

   part = SizeDataSetHeader(0, 0, 1);
   CHECK(0 <= part);
   sum += part;

   part = SizeClassificationTarget(3, k_cSamples, targets);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(0, 0, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillClassificationTarget(3, k_cSamples, targets, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("data_set_shared, two features, zero samples, classification") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   part = SizeDataSetHeader(2, 0, 1);
   CHECK(0 <= part);
   sum += part;

   part = SizeFeature(EBM_FALSE, 0, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = SizeFeature(EBM_FALSE, 0, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = SizeClassificationTarget(0, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(2, 0, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(EBM_FALSE, 0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(EBM_FALSE, 0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillClassificationTarget(0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("data_set_shared, two features, 3 samples, classification") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;
   constexpr IntEbmType k_cSamples = 3;
   IntEbmType binnedData[k_cSamples] { 2, 1, 0 };
   IntEbmType targets[k_cSamples] { 2, 1, 0 };

   part = SizeDataSetHeader(2, 0, 1);
   CHECK(0 <= part);
   sum += part;

   part = SizeFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0]);
   CHECK(0 <= part);
   sum += part;

   part = SizeFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0]);
   CHECK(0 <= part);
   sum += part;

   part = SizeClassificationTarget(3, k_cSamples, &targets[0]);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(2, 0, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillClassificationTarget(3, k_cSamples, &targets[0], sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}


// weights 

TEST_CASE("data_set_shared, zero features, zero samples, weights, regression") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   part = SizeDataSetHeader(0, 1, 1);
   CHECK(0 <= part);
   sum += part;

   part = SizeWeight(0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = SizeRegressionTarget(0, nullptr);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(0, 1, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillWeight(0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillRegressionTarget(0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("data_set_shared, zero features, three samples, weights, regression") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   constexpr IntEbmType k_cSamples = 3;
   FloatEbmType weights[k_cSamples] { 0.31, 0.21, 0.11 };
   FloatEbmType targets[k_cSamples] { 0.3, 0.2, 0.1 };

   part = SizeDataSetHeader(0, 1, 1);
   CHECK(0 <= part);
   sum += part;

   part = SizeWeight(k_cSamples, weights);
   CHECK(0 <= part);
   sum += part;

   part = SizeRegressionTarget(k_cSamples, targets);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(0, 1, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillWeight(k_cSamples, weights, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillRegressionTarget(k_cSamples, targets, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}


TEST_CASE("data_set_shared, two features, zero samples, weights, regression") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   part = SizeDataSetHeader(2, 1, 1);
   CHECK(0 <= part);
   sum += part;

   part = SizeWeight(0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = SizeFeature(EBM_FALSE, 0, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = SizeFeature(EBM_FALSE, 0, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = SizeRegressionTarget(0, nullptr);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(2, 1, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(EBM_FALSE, 0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(EBM_FALSE, 0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillWeight(0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillRegressionTarget(0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("data_set_shared, two features, 3 samples, weights, regression") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;
   constexpr IntEbmType k_cSamples = 3;
   IntEbmType binnedData[k_cSamples] { 2, 1, 0 };
   FloatEbmType weights[k_cSamples] { 0.31, 0.21, 0.11 };
   FloatEbmType targets[k_cSamples] { 0.3, 0.2, 0.1 };

   part = SizeDataSetHeader(2, 1, 1);
   CHECK(0 <= part);
   sum += part;

   part = SizeFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0]);
   CHECK(0 <= part);
   sum += part;

   part = SizeFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0]);
   CHECK(0 <= part);
   sum += part;

   part = SizeWeight(k_cSamples, weights);
   CHECK(0 <= part);
   sum += part;

   part = SizeRegressionTarget(k_cSamples, &targets[0]);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(2, 1, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillWeight(k_cSamples, weights, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillRegressionTarget(k_cSamples, &targets[0], sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("data_set_shared, zero features, zero samples, weights, classification") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   part = SizeDataSetHeader(0, 1, 1);
   CHECK(0 <= part);
   sum += part;

   part = SizeWeight(0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = SizeClassificationTarget(0, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(0, 1, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillWeight(0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillClassificationTarget(0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("data_set_shared, zero features, three samples, weights, classification") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   constexpr IntEbmType k_cSamples = 3;
   FloatEbmType weights[k_cSamples] { 0.31, 0.21, 0.11 };
   IntEbmType targets[k_cSamples] { 2, 1, 0 };

   part = SizeDataSetHeader(0, 1, 1);
   CHECK(0 <= part);
   sum += part;

   part = SizeWeight(k_cSamples, weights);
   CHECK(0 <= part);
   sum += part;

   part = SizeClassificationTarget(3, k_cSamples, targets);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(0, 1, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillWeight(k_cSamples, weights, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillClassificationTarget(3, k_cSamples, targets, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("data_set_shared, two features, zero samples, weights, classification") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;

   part = SizeDataSetHeader(2, 1, 1);
   CHECK(0 <= part);
   sum += part;

   part = SizeFeature(EBM_FALSE, 0, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = SizeFeature(EBM_FALSE, 0, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = SizeWeight(0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = SizeClassificationTarget(0, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(2, 1, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(EBM_FALSE, 0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(EBM_FALSE, 0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillWeight(0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillClassificationTarget(0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("data_set_shared, two features, 3 samples, weights, classification") {
   IntEbmType sum = 0;
   IntEbmType part;
   ErrorEbmType error;
   constexpr IntEbmType k_cSamples = 3;
   IntEbmType binnedData[k_cSamples] { 2, 1, 0 };
   FloatEbmType weights[k_cSamples] { 0.31, 0.21, 0.11 };
   IntEbmType targets[k_cSamples] { 2, 1, 0 };

   part = SizeDataSetHeader(2, 1, 1);
   CHECK(0 <= part);
   sum += part;

   part = SizeFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0]);
   CHECK(0 <= part);
   sum += part;

   part = SizeFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0]);
   CHECK(0 <= part);
   sum += part;

   part = SizeWeight(k_cSamples, weights);
   CHECK(0 <= part);
   sum += part;

   part = SizeClassificationTarget(3, k_cSamples, &targets[0]);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(2, 1, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(EBM_FALSE, 3, k_cSamples, &binnedData[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillWeight(k_cSamples, weights, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillClassificationTarget(3, k_cSamples, &targets[0], sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}
