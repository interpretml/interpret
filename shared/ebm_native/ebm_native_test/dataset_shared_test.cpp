// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_test.hpp"

#include "ebm_native.h"
#include "ebm_native_test.hpp"

static constexpr TestPriority k_filePriority = TestPriority::DataSetShared;

TEST_CASE("dataset_shared, zero features, zero samples, regression") {
   IntEbm sum = 0;
   IntEbm part;
   ErrorEbm error;

   part = MeasureDataSetHeader(0, 0, 1);
   CHECK(0 <= part);
   sum += part;

   part = MeasureRegressionTarget(0, nullptr);
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

TEST_CASE("dataset_shared, zero features, three samples, regression") {
   IntEbm sum = 0;
   IntEbm part;
   ErrorEbm error;

   static constexpr IntEbm k_cSamples = 3;
   double targets[k_cSamples] { 0.3, 0.2, 0.1 };

   part = MeasureDataSetHeader(0, 0, 1);
   CHECK(0 <= part);
   sum += part;

   part = MeasureRegressionTarget(k_cSamples, targets);
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


TEST_CASE("dataset_shared, two features, zero samples, regression") {
   IntEbm sum = 0;
   IntEbm part;
   ErrorEbm error;

   part = MeasureDataSetHeader(2, 0, 1);
   CHECK(0 <= part);
   sum += part;

   part = MeasureFeature(0, EBM_TRUE, EBM_TRUE, EBM_FALSE, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = MeasureFeature(0, EBM_TRUE, EBM_TRUE, EBM_FALSE, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = MeasureRegressionTarget(0, nullptr);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(2, 0, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(0, EBM_TRUE, EBM_TRUE, EBM_FALSE, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(0, EBM_TRUE, EBM_TRUE, EBM_FALSE, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillRegressionTarget(0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("dataset_shared, two features, 3 samples, regression") {
   IntEbm sum = 0;
   IntEbm part;
   ErrorEbm error;
   static constexpr IntEbm k_cSamples = 3;
   IntEbm binIndexes[k_cSamples] { 2, 1, 0 };
   double targets[k_cSamples] { 0.3, 0.2, 0.1 };

   part = MeasureDataSetHeader(2, 0, 1);
   CHECK(0 <= part);
   sum += part;

   part = MeasureFeature(3, EBM_TRUE, EBM_TRUE, EBM_FALSE, k_cSamples, &binIndexes[0]);
   CHECK(0 <= part);
   sum += part;

   part = MeasureFeature(3, EBM_TRUE, EBM_TRUE, EBM_FALSE, k_cSamples, &binIndexes[0]);
   CHECK(0 <= part);
   sum += part;

   part = MeasureRegressionTarget(k_cSamples, &targets[0]);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(2, 0, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(3, EBM_TRUE, EBM_TRUE, EBM_FALSE, k_cSamples, &binIndexes[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(3, EBM_TRUE, EBM_TRUE, EBM_FALSE, k_cSamples, &binIndexes[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillRegressionTarget(k_cSamples, &targets[0], sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("dataset_shared, zero features, zero samples, classification") {
   IntEbm sum = 0;
   IntEbm part;
   ErrorEbm error;

   part = MeasureDataSetHeader(0, 0, 1);
   CHECK(0 <= part);
   sum += part;

   part = MeasureClassificationTarget(0, 0, nullptr);
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

TEST_CASE("dataset_shared, zero features, three samples, classification") {
   IntEbm sum = 0;
   IntEbm part;
   ErrorEbm error;

   static constexpr IntEbm k_cSamples = 3;
   IntEbm targets[k_cSamples] { 2, 1, 0 };

   part = MeasureDataSetHeader(0, 0, 1);
   CHECK(0 <= part);
   sum += part;

   part = MeasureClassificationTarget(3, k_cSamples, targets);
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

TEST_CASE("dataset_shared, two features, zero samples, classification") {
   IntEbm sum = 0;
   IntEbm part;
   ErrorEbm error;

   part = MeasureDataSetHeader(2, 0, 1);
   CHECK(0 <= part);
   sum += part;

   part = MeasureFeature(0, EBM_TRUE, EBM_TRUE, EBM_FALSE, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = MeasureFeature(0, EBM_TRUE, EBM_TRUE, EBM_FALSE, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = MeasureClassificationTarget(0, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(2, 0, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(0, EBM_TRUE, EBM_TRUE, EBM_FALSE, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(0, EBM_TRUE, EBM_TRUE, EBM_FALSE, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillClassificationTarget(0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("dataset_shared, two features, 3 samples, classification") {
   IntEbm sum = 0;
   IntEbm part;
   ErrorEbm error;
   static constexpr IntEbm k_cSamples = 3;
   IntEbm binIndexes[k_cSamples] { 2, 1, 0 };
   IntEbm targets[k_cSamples] { 2, 1, 0 };

   part = MeasureDataSetHeader(2, 0, 1);
   CHECK(0 <= part);
   sum += part;

   part = MeasureFeature(3, EBM_TRUE, EBM_TRUE, EBM_FALSE, k_cSamples, &binIndexes[0]);
   CHECK(0 <= part);
   sum += part;

   part = MeasureFeature(3, EBM_TRUE, EBM_TRUE, EBM_FALSE, k_cSamples, &binIndexes[0]);
   CHECK(0 <= part);
   sum += part;

   part = MeasureClassificationTarget(3, k_cSamples, &targets[0]);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(2, 0, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(3, EBM_TRUE, EBM_TRUE, EBM_FALSE, k_cSamples, &binIndexes[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(3, EBM_TRUE, EBM_TRUE, EBM_FALSE, k_cSamples, &binIndexes[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillClassificationTarget(3, k_cSamples, &targets[0], sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}


// weights 

TEST_CASE("dataset_shared, zero features, zero samples, weights, regression") {
   IntEbm sum = 0;
   IntEbm part;
   ErrorEbm error;

   part = MeasureDataSetHeader(0, 1, 1);
   CHECK(0 <= part);
   sum += part;

   part = MeasureWeight(0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = MeasureRegressionTarget(0, nullptr);
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

TEST_CASE("dataset_shared, zero features, three samples, weights, regression") {
   IntEbm sum = 0;
   IntEbm part;
   ErrorEbm error;

   static constexpr IntEbm k_cSamples = 3;
   double weights[k_cSamples] { 0.31, 0.21, 0.11 };
   double targets[k_cSamples] { 0.3, 0.2, 0.1 };

   part = MeasureDataSetHeader(0, 1, 1);
   CHECK(0 <= part);
   sum += part;

   part = MeasureWeight(k_cSamples, weights);
   CHECK(0 <= part);
   sum += part;

   part = MeasureRegressionTarget(k_cSamples, targets);
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


TEST_CASE("dataset_shared, two features, zero samples, weights, regression") {
   IntEbm sum = 0;
   IntEbm part;
   ErrorEbm error;

   part = MeasureDataSetHeader(2, 1, 1);
   CHECK(0 <= part);
   sum += part;

   part = MeasureWeight(0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = MeasureFeature(0, EBM_TRUE, EBM_TRUE, EBM_FALSE, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = MeasureFeature(0, EBM_TRUE, EBM_TRUE, EBM_FALSE, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = MeasureRegressionTarget(0, nullptr);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(2, 1, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(0, EBM_TRUE, EBM_TRUE, EBM_FALSE, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(0, EBM_TRUE, EBM_TRUE, EBM_FALSE, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillWeight(0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillRegressionTarget(0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("dataset_shared, two features, 3 samples, weights, regression") {
   IntEbm sum = 0;
   IntEbm part;
   ErrorEbm error;
   static constexpr IntEbm k_cSamples = 3;
   IntEbm binIndexes[k_cSamples] { 2, 1, 0 };
   double weights[k_cSamples] { 0.31, 0.21, 0.11 };
   double targets[k_cSamples] { 0.3, 0.2, 0.1 };

   part = MeasureDataSetHeader(2, 1, 1);
   CHECK(0 <= part);
   sum += part;

   part = MeasureFeature(3, EBM_TRUE, EBM_TRUE, EBM_FALSE, k_cSamples, &binIndexes[0]);
   CHECK(0 <= part);
   sum += part;

   part = MeasureFeature(3, EBM_TRUE, EBM_TRUE, EBM_FALSE, k_cSamples, &binIndexes[0]);
   CHECK(0 <= part);
   sum += part;

   part = MeasureWeight(k_cSamples, weights);
   CHECK(0 <= part);
   sum += part;

   part = MeasureRegressionTarget(k_cSamples, &targets[0]);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(2, 1, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(3, EBM_TRUE, EBM_TRUE, EBM_FALSE, k_cSamples, &binIndexes[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(3, EBM_TRUE, EBM_TRUE, EBM_FALSE, k_cSamples, &binIndexes[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillWeight(k_cSamples, weights, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillRegressionTarget(k_cSamples, &targets[0], sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("dataset_shared, zero features, zero samples, weights, classification") {
   IntEbm sum = 0;
   IntEbm part;
   ErrorEbm error;

   part = MeasureDataSetHeader(0, 1, 1);
   CHECK(0 <= part);
   sum += part;

   part = MeasureWeight(0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = MeasureClassificationTarget(0, 0, nullptr);
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

TEST_CASE("dataset_shared, zero features, three samples, weights, classification") {
   IntEbm sum = 0;
   IntEbm part;
   ErrorEbm error;

   static constexpr IntEbm k_cSamples = 3;
   double weights[k_cSamples] { 0.31, 0.21, 0.11 };
   IntEbm targets[k_cSamples] { 2, 1, 0 };

   part = MeasureDataSetHeader(0, 1, 1);
   CHECK(0 <= part);
   sum += part;

   part = MeasureWeight(k_cSamples, weights);
   CHECK(0 <= part);
   sum += part;

   part = MeasureClassificationTarget(3, k_cSamples, targets);
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

TEST_CASE("dataset_shared, two features, zero samples, weights, classification") {
   IntEbm sum = 0;
   IntEbm part;
   ErrorEbm error;

   part = MeasureDataSetHeader(2, 1, 1);
   CHECK(0 <= part);
   sum += part;

   part = MeasureFeature(0, EBM_TRUE, EBM_TRUE, EBM_FALSE, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = MeasureFeature(0, EBM_TRUE, EBM_TRUE, EBM_FALSE, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = MeasureWeight(0, nullptr);
   CHECK(0 <= part);
   sum += part;

   part = MeasureClassificationTarget(0, 0, nullptr);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(2, 1, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(0, EBM_TRUE, EBM_TRUE, EBM_FALSE, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(0, EBM_TRUE, EBM_TRUE, EBM_FALSE, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillWeight(0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillClassificationTarget(0, 0, nullptr, sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}

TEST_CASE("dataset_shared, two features, 3 samples, weights, classification") {
   IntEbm sum = 0;
   IntEbm part;
   ErrorEbm error;
   static constexpr IntEbm k_cSamples = 3;
   IntEbm binIndexes[k_cSamples] { 2, 1, 0 };
   double weights[k_cSamples] { 0.31, 0.21, 0.11 };
   IntEbm targets[k_cSamples] { 2, 1, 0 };

   part = MeasureDataSetHeader(2, 1, 1);
   CHECK(0 <= part);
   sum += part;

   part = MeasureFeature(3, EBM_TRUE, EBM_TRUE, EBM_FALSE, k_cSamples, &binIndexes[0]);
   CHECK(0 <= part);
   sum += part;

   part = MeasureFeature(3, EBM_TRUE, EBM_TRUE, EBM_FALSE, k_cSamples, &binIndexes[0]);
   CHECK(0 <= part);
   sum += part;

   part = MeasureWeight(k_cSamples, weights);
   CHECK(0 <= part);
   sum += part;

   part = MeasureClassificationTarget(3, k_cSamples, &targets[0]);
   CHECK(0 <= part);
   sum += part;

   std::vector<char> buffer(static_cast<size_t>(sum) + 1, 77);
   buffer[static_cast<size_t>(sum)] = 99;

   error = FillDataSetHeader(2, 1, 1, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(3, EBM_TRUE, EBM_TRUE, EBM_FALSE, k_cSamples, &binIndexes[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillFeature(3, EBM_TRUE, EBM_TRUE, EBM_FALSE, k_cSamples, &binIndexes[0], sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillWeight(k_cSamples, weights, sum, &buffer[0]);
   CHECK(Error_None == error);
   error = FillClassificationTarget(3, k_cSamples, &targets[0], sum, &buffer[0]);
   CHECK(Error_None == error);

   CHECK(99 == buffer[static_cast<size_t>(sum)]);
}
