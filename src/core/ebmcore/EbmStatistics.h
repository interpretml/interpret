// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef STATISTICS_H
#define STATISTICS_H

#include <assert.h>
#include <cmath> // log, exp, sqrt, etc.  Use cmath instead of math.h so that we get type overloading for these functions for seemless float/double useage
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // TML_INLINE
#include "Logging.h" // EBM_ASSERT & LOG

TML_INLINE FractionalDataType ComputeNodeSplittingScore(const FractionalDataType sumResidualError, const size_t cCases) {
   // TODO: after we eliminate bin compression, we should be checking to see if cCases is zero before divding by it.. Instead of doing that outside this function, we can move all instances of checking for zero into this function
   EBM_ASSERT(0 < cCases); // we purge bins that have case counts of zero, so cCases should never be zero
   return sumResidualError / cCases * sumResidualError;
}

WARNING_PUSH
WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO

TML_INLINE FractionalDataType ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(const FractionalDataType sumResidualError, const FractionalDataType sumDenominator) {
   if(LIKELY(static_cast<FractionalDataType>(0) != sumDenominator)) {
      // this is a very predictable branch, so we'd prefer this to be an actual branch rather than an unpredictable one
      return sumResidualError / sumDenominator;
   } else {
      return static_cast<FractionalDataType>(0);
   }
}

WARNING_POP

TML_INLINE FractionalDataType ComputeSmallChangeInRegressionPredictionForOneSegment(const FractionalDataType sumResidualError, const size_t cCases) {
   // TODO: check again if we can ever have a zero here
   // TODO: after we eliminate bin compression, we should be checking to see if cCases is zero before divding by it.. Instead of doing that outside this function, we can move all instances of checking for zero into this function
   EBM_ASSERT(0 != cCases);
   return sumResidualError / cCases;
}

TML_INLINE FractionalDataType ComputeRegressionResidualError(const FractionalDataType predictionScore, const FractionalDataType actualValue) {
   const FractionalDataType result = actualValue - predictionScore;
   return result;
}

TML_INLINE FractionalDataType ComputeRegressionResidualError(const FractionalDataType value) {
   // this function is here to document where we're calculating regression, like ComputeClassificationResidualErrorBinaryclass below.  It doesn't do anything, but it serves as an indication that the calculation would be placed here if we changed it in the future
   return value;
}

TML_INLINE FractionalDataType ComputeClassificationResidualErrorBinaryclass(const FractionalDataType trainingLogOddsPrediction, const StorageDataTypeCore binnedActualValue) {
   EBM_ASSERT(0 == binnedActualValue || 1 == binnedActualValue);

   // this function outputs 0 if we perfectly predict the target with 100% certainty.  To do so, trainingLogOddsPrediction would need to be either infinity or -infinity
   // this function outputs 1 if actual value was 1 but we incorrectly predicted with 100% certainty that it was 0 by having trainingLogOddsPrediction be -infinity
   // this function outputs -1 if actual value was 0 but we incorrectly predicted with 100% certainty that it was 1 by having trainingLogOddsPrediction be infinity
   //
   // this function outputs 0.5 if actual value was 1 but we were 50%/50% by having trainingLogOddsPrediction be 0
   // this function outputs -0.5 if actual value was 0 but we were 50%/50% by having trainingLogOddsPrediction be 0

   // TODO : this assembly is using branches instead of conditional execution.  Try finding something that will work without any branching, like multiplication, even if it's slower without branch misses

   // TODO: instead of negating trainingLogOddsPrediction and then conditionally assigning it, can we multiply it with the result of the first conditional -1 vs 1
   return (UNPREDICTABLE(0 == binnedActualValue) ? -1 : 1) / (1 + std::exp(UNPREDICTABLE(0 == binnedActualValue) ? -trainingLogOddsPrediction : trainingLogOddsPrediction)); // exp will return the same type that it is given, either float or double
}

// if trainingLogOddsPrediction is zero (so, 50%/50% odds), then we can call this function
TML_INLINE FractionalDataType ComputeClassificationResidualErrorBinaryclass(const StorageDataTypeCore binnedActualValue) {
   EBM_ASSERT(0 == binnedActualValue || 1 == binnedActualValue);
   const FractionalDataType result = UNPREDICTABLE(0 == binnedActualValue) ? -0.5 : 0.5;
   EBM_ASSERT(ComputeClassificationResidualErrorBinaryclass(0, binnedActualValue) == result);
   return result;
}

TML_INLINE FractionalDataType ComputeClassificationResidualErrorMulticlass(const FractionalDataType sumExp, const FractionalDataType trainingLogWeight, const StorageDataTypeCore binnedActualValue, const StorageDataTypeCore iVector) {
   // TODO: is it better to use the non-branching conditional below, or is it better to assign all the items the negation case and then AFTERWARDS adding one to the single case that is equal to iVector 
   const FractionalDataType yi = UNPREDICTABLE(iVector == binnedActualValue) ? static_cast<FractionalDataType>(1) : static_cast<FractionalDataType>(0);
   const FractionalDataType ret = yi - std::exp(trainingLogWeight) / sumExp;
   return ret;
}

// if trainingLogWeight is zero, we can call this simpler function
TML_INLINE FractionalDataType ComputeClassificationResidualErrorMulticlass(const bool isMatch, const FractionalDataType sumExp) {
   const FractionalDataType yi = UNPREDICTABLE(isMatch) ? static_cast<FractionalDataType>(1) : static_cast<FractionalDataType>(0);
   const FractionalDataType ret = yi - static_cast<FractionalDataType>(1) / sumExp;

   EBM_ASSERT(!isMatch || ComputeClassificationResidualErrorMulticlass(sumExp, 0, 1, 1) == ret);
   EBM_ASSERT(isMatch || ComputeClassificationResidualErrorMulticlass(sumExp, 0, 1, 2) == ret);

   return ret;
}

// if trainingLogWeight is zero, we can call this simpler function
TML_INLINE FractionalDataType ComputeClassificationResidualErrorMulticlass(const StorageDataTypeCore binnedActualValue, const StorageDataTypeCore iVector, const FractionalDataType matchValue, const FractionalDataType nonMatchValue) {
   // TODO: is it better to use the non-branching conditional below, or is it better to assign all the items the negation case and then AFTERWARDS adding one to the single case that is equal to iVector 
   const FractionalDataType ret = UNPREDICTABLE(iVector == binnedActualValue) ? matchValue : nonMatchValue;
   return ret;
}

TML_INLINE FractionalDataType ComputeClassificationSingleCaseLogLossBinaryclass(const FractionalDataType validationLogOddsPrediction, const StorageDataTypeCore binnedActualValue) {
   EBM_ASSERT(0 == binnedActualValue || 1 == binnedActualValue);

   // TODO: also try log1p and I guess (exp1p?) for accuracy and performance
   // TODO: the calls to log and exp have loops and conditional statements.  Suposedly the assembly FYL2X is slower than the C++ log/exp functions.  Look into this more.  We might end up sorting our input data by the target to avoid this if we can't find a non-branching solution because branch prediction will be important here
   // https://stackoverflow.com/questions/45785705/logarithm-in-c-and-assembly

   return std::log(1 + std::exp(UNPREDICTABLE(0 == binnedActualValue) ? validationLogOddsPrediction : -validationLogOddsPrediction)); // log & exp will return the same type that it is given, either float or double
}

TML_INLINE FractionalDataType ComputeClassificationSingleCaseLogLossMulticlass(const FractionalDataType sumExp, const FractionalDataType * const aValidationLogWeight, const StorageDataTypeCore binnedActualValue) {
   // TODO: is there any way to avoid doing the negation below, like changing sumExp or what we store in memory?
   return -std::log(std::exp(aValidationLogWeight[binnedActualValue]) / sumExp);
}

#endif // STATISTICS_H