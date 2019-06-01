// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef PREDICTION_STATISTICS_H
#define PREDICTION_STATISTICS_H

#include <type_traits> // std::is_pod
#include <assert.h>

#include "ebmcore.h" // FractionalDataType
#include "EbmInternal.h" // TML_INLINE

template<bool bRegression>
class PredictionStatistics;

template<>
struct PredictionStatistics<false> final {
   // classification version of the PredictionStatistics class

   FractionalDataType sumResidualError;
   // TODO: for single features, we probably want to just do a single pass of the data and collect our sumDenominator during that sweep.  This is probably also true for pairs since calculating pair sums can be done fairly efficiently, but for tripples and higher dimensions we might be better off calculating JUST the sumResidualError which is the only thing required for choosing splits and we could then do a second pass of the data to find the denominators once we know the splits.  Tripples and higher dimensions tend to re-add/subtract the same cells many times over which is why it might be better there.  Test these theories out on large datasets
   FractionalDataType sumDenominator;

   TML_INLINE FractionalDataType GetSumDenominator() const {
      return sumDenominator;
   }
   TML_INLINE void SetSumDenominator(FractionalDataType sumDenominatorSet) {
      sumDenominator = sumDenominatorSet;
   }
   TML_INLINE void Add(const PredictionStatistics<false> & other) {
      sumResidualError += other.sumResidualError;
      sumDenominator += other.sumDenominator;
   }
   TML_INLINE void Subtract(const PredictionStatistics<false> & other) {
      sumResidualError -= other.sumResidualError;
      sumDenominator -= other.sumDenominator;
   }
   TML_INLINE void Copy(const PredictionStatistics<false> & other) {
      sumResidualError = other.sumResidualError;
      sumDenominator = other.sumDenominator;
   }
   TML_INLINE void AssertZero() {
      assert(0 == sumResidualError);
      assert(0 == sumDenominator);
   }
};

template<>
struct PredictionStatistics<true> final {
   // regression version of the PredictionStatistics class

   FractionalDataType sumResidualError;

   TML_INLINE FractionalDataType GetSumDenominator() const {
      assert(false); // this should never be called, but the compiler seems to want it to exist
      return static_cast<FractionalDataType>(0);
   }
   TML_INLINE void SetSumDenominator(FractionalDataType sumDenominator) {
      assert(false); // this should never be called, but the compiler seems to want it to exist
   }
   TML_INLINE void Add(const PredictionStatistics<true> & other) {
      sumResidualError += other.sumResidualError;
   }
   TML_INLINE void Subtract(const PredictionStatistics<true> & other) {
      sumResidualError -= other.sumResidualError;
   }
   TML_INLINE void Copy(const PredictionStatistics<true> & other) {
      sumResidualError = other.sumResidualError;
   }
   TML_INLINE void AssertZero() {
      assert(0 == sumResidualError);
   }
};

static_assert(std::is_pod<PredictionStatistics<false>>::value, "PredictionStatistics will be more efficient as a POD as we make potentially large arrays of them!");
static_assert(std::is_pod<PredictionStatistics<true>>::value, "PredictionStatistics will be more efficient as a POD as we make potentially large arrays of them!");

#endif // PREDICTION_STATISTICS_H
