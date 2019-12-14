// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef EBM_STATISTICS_H
#define EBM_STATISTICS_H

#include <cmath> // log, exp, sqrt, etc.  Use cmath instead of math.h so that we get type overloading for these functions for seemless float/double useage
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG

class EbmStatistics final {
   EBM_INLINE EbmStatistics() {
      // DON'T allow anyone to make this static class
   }

public:

   EBM_INLINE static FractionalDataType ComputeNewtonRaphsonStep(const FractionalDataType residualError) {
      // !!! IMPORTANT: Newton-Raphson step, as illustrated in Friedman's original paper (https://statweb.stanford.edu/~jhf/ftp/trebst.pdf, page 9). Note that they are using t * (2 - t) since they have a 2 in their objective
      const FractionalDataType absResidualError = std::abs(residualError); // abs will return the same type that it is given, either float or double
      return absResidualError * (FractionalDataType { 1 } - absResidualError);
   }

   EBM_INLINE static FractionalDataType ComputeNodeSplittingScore(const FractionalDataType sumResidualError, const size_t cInstances) {
      // !!! IMPORTANT: This gain function used to determine splits is equivalent to minimizing sum of squared error SSE, which can be seen following the derivation of Equation #7 in Ping Li's paper -> https://arxiv.org/pdf/1203.3491.pdf

      // TODO: we're using this node splitting score for both classification and regression.  It is designed to minimize MSE, so should we also then use it for classification, and what about the possibility of using Newton-Raphson step in the gain?

      return UNPREDICTABLE(0 == cInstances) ? FractionalDataType { 0 } : sumResidualError / cInstances * sumResidualError;
   }

   WARNING_PUSH
   WARNING_DISABLE_POTENTIAL_DIVIDE_BY_ZERO
   EBM_INLINE static FractionalDataType ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(const FractionalDataType sumResidualError, const FractionalDataType sumDenominator) {
      if(LIKELY(FractionalDataType { 0 } != sumDenominator)) {
         // this is a very predictable branch, so we'd prefer this to be an actual branch rather than an unpredictable one
         return sumResidualError / sumDenominator;
      } else {
         return FractionalDataType { 0 };
      }
   }
   WARNING_POP

   EBM_INLINE static FractionalDataType ComputeSmallChangeInRegressionPredictionForOneSegment(const FractionalDataType sumResidualError, const FractionalDataType cInstances) {
      // TODO: check again if we can ever have a zero here
      // TODO: after we eliminate bin compression, we should be checking to see if cInstances is zero before divding by it.. Instead of doing that outside this function, we can move all instances of checking for zero into this function
      EBM_ASSERT(0 != cInstances);
      return sumResidualError / cInstances;
   }

   EBM_INLINE static FractionalDataType ComputeRegressionResidualError(const FractionalDataType predictionScore, const FractionalDataType actualValue) {
      const FractionalDataType result = actualValue - predictionScore;
      return result;
   }

   EBM_INLINE static FractionalDataType ComputeRegressionResidualError(const FractionalDataType value) {
      // this function is here to document where we're calculating regression, like ComputeClassificationResidualErrorBinaryclass below.  It doesn't do anything, but it serves as an indication that the calculation would be placed here if we changed it in the future
      return value;
   }

   EBM_INLINE static FractionalDataType ComputeClassificationResidualErrorBinaryclass(const FractionalDataType trainingLogOddsPrediction, const StorageDataTypeCore binnedActualValue) {
      EBM_ASSERT(0 == binnedActualValue || 1 == binnedActualValue);

      // this function outputs 0 if we perfectly predict the target with 100% certainty.  To do so, trainingLogOddsPrediction would need to be either infinity or -infinity
      // this function outputs 1 if actual value was 1 but we incorrectly predicted with 100% certainty that it was 0 by having trainingLogOddsPrediction be -infinity
      // this function outputs -1 if actual value was 0 but we incorrectly predicted with 100% certainty that it was 1 by having trainingLogOddsPrediction be infinity
      //
      // this function outputs 0.5 if actual value was 1 but we were 50%/50% by having trainingLogOddsPrediction be 0
      // this function outputs -0.5 if actual value was 0 but we were 50%/50% by having trainingLogOddsPrediction be 0

      // TODO : In the future we'll sort our data by the target value, so we'll know ahead of time if 0 == binnedActualValue.  We expect 0 to be the default target, so we should flip the value of trainingLogOddsPrediction so that we don't need to negate it for the default 0 case
      const FractionalDataType ret = (UNPREDICTABLE(0 == binnedActualValue) ? FractionalDataType { -1 } : FractionalDataType { 1 }) / (FractionalDataType { 1 } + EbmExp(UNPREDICTABLE(0 == binnedActualValue) ? -trainingLogOddsPrediction : trainingLogOddsPrediction)); // exp will return the same type that it is given, either float or double
#ifndef NDEBUG
      const FractionalDataType retDebug = ComputeClassificationResidualErrorMulticlass(1 + EbmExp(trainingLogOddsPrediction), trainingLogOddsPrediction, binnedActualValue, 1);
      EBM_ASSERT(std::isinf(ret) || std::isinf(retDebug) || std::isnan(ret) || std::isnan(retDebug) || std::abs(retDebug - ret) < FractionalDataType { 0.0000001 });
#endif // NDEBUG
      return ret;
   }

   // if trainingLogOddsPrediction is zero (so, 50%/50% odds), then we can call this function
   EBM_INLINE static FractionalDataType ComputeClassificationResidualErrorBinaryclass(const StorageDataTypeCore binnedActualValue) {
      EBM_ASSERT(0 == binnedActualValue || 1 == binnedActualValue);
      const FractionalDataType result = UNPREDICTABLE(0 == binnedActualValue) ? FractionalDataType { -0.5 } : FractionalDataType { 0.5 };
      EBM_ASSERT(ComputeClassificationResidualErrorBinaryclass(0, binnedActualValue) == result);
      return result;
   }

   EBM_INLINE static FractionalDataType ComputeClassificationResidualErrorMulticlass(const FractionalDataType sumExp, const FractionalDataType trainingLogWeight, const StorageDataTypeCore binnedActualValue, const StorageDataTypeCore iVector) {
      // TODO: is it better to use the non-branching conditional below, or is it better to assign all the items the negation case and then AFTERWARDS adding one to the single case that is equal to iVector 
      const FractionalDataType yi = UNPREDICTABLE(iVector == binnedActualValue) ? FractionalDataType { 1 } : FractionalDataType { 0 };
      const FractionalDataType ret = yi - EbmExp(trainingLogWeight) / sumExp;
      return ret;
   }

   // if trainingLogWeight is zero, we can call this simpler function
   EBM_INLINE static FractionalDataType ComputeClassificationResidualErrorMulticlass(const bool isMatch, const FractionalDataType sumExp) {
      const FractionalDataType yi = UNPREDICTABLE(isMatch) ? FractionalDataType { 1 } : FractionalDataType { 0 };
      const FractionalDataType ret = yi - FractionalDataType { 1 } / sumExp;

      EBM_ASSERT(!isMatch || ComputeClassificationResidualErrorMulticlass(sumExp, 0, 1, 1) == ret);
      EBM_ASSERT(isMatch || ComputeClassificationResidualErrorMulticlass(sumExp, 0, 1, 2) == ret);

      return ret;
   }

   // if trainingLogWeight is zero, we can call this simpler function
   EBM_INLINE static FractionalDataType ComputeClassificationResidualErrorMulticlass(const StorageDataTypeCore binnedActualValue, const StorageDataTypeCore iVector, const FractionalDataType matchValue, const FractionalDataType nonMatchValue) {
      // TODO: is it better to use the non-branching conditional below, or is it better to assign all the items the negation case and then AFTERWARDS adding one to the single case that is equal to iVector 
      const FractionalDataType ret = UNPREDICTABLE(iVector == binnedActualValue) ? matchValue : nonMatchValue;
      return ret;
   }

   EBM_INLINE static FractionalDataType ComputeClassificationSingleInstanceLogLossBinaryclass(const FractionalDataType validationLogOddsPrediction, const StorageDataTypeCore binnedActualValue) {
      // we are confirmed to get the same log loss value as scikit-learn for binary and multiclass classification
      EBM_ASSERT(0 == binnedActualValue || 1 == binnedActualValue);

      // TODO: also try log1p and I guess (exp1p?) for accuracy and performance
      // TODO: the calls to log and exp have loops and conditional statements.  Suposedly the assembly FYL2X is slower than the C++ log/exp functions.  Look into this more.  We might end up sorting our input data by the target to avoid this if we can't find a non-branching solution because branch prediction will be important here
      // https://stackoverflow.com/questions/45785705/logarithm-in-c-and-assembly

      const FractionalDataType ret = EbmLog(FractionalDataType { 1 } + EbmExp(UNPREDICTABLE(0 == binnedActualValue) ? validationLogOddsPrediction : -validationLogOddsPrediction)); // log & exp will return the same type that it is given, either float or double

#ifndef NDEBUG
      FractionalDataType scores[2];
      scores[0] = 0;
      scores[1] = validationLogOddsPrediction;
      const FractionalDataType retDebug = EbmStatistics::ComputeClassificationSingleInstanceLogLossMulticlass(1 + EbmExp(validationLogOddsPrediction), scores, binnedActualValue);
      EBM_ASSERT(std::isinf(ret) || std::isinf(retDebug) || std::abs(retDebug - ret) < FractionalDataType { 0.0000001 });
#endif // NDEBUG

      return ret;
   }

   EBM_INLINE static FractionalDataType ComputeClassificationSingleInstanceLogLossMulticlass(const FractionalDataType sumExp, const FractionalDataType * const aValidationLogWeight, const StorageDataTypeCore binnedActualValue) {
      // we are confirmed to get the same log loss value as scikit-learn for binary and multiclass classification
      return EbmLog(sumExp / EbmExp(aValidationLogWeight[binnedActualValue]));
   }

   EBM_INLINE static FractionalDataType ComputeRegressionSingleInstanceMeanSquaredError(const FractionalDataType residualError) {
      // we are confirmed to get the same mean squared error value as scikit-learn for regression
      return residualError * residualError;
   }
};

#endif // EBM_STATISTICS_H