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

   EBM_INLINE static FractionalDataType ComputeNodeSplittingScore(const FractionalDataType sumResidualError, const FractionalDataType cInstances) {
      // !!! IMPORTANT: This gain function used to determine splits is equivalent to minimizing sum of squared error SSE, which can be seen following the derivation of Equation #7 in Ping Li's paper -> https://arxiv.org/pdf/1203.3491.pdf

      // TODO: we're using this node splitting score for both classification and regression.  It is designed to minimize MSE, so should we also then use it for classification?  What about the possibility of using Newton-Raphson step in the gain?

#ifdef LEGACY_COMPATIBILITY
      return LIKELY(FractionalDataType { 0 } != cInstances) ? sumResidualError / cInstances * sumResidualError : FractionalDataType { 0 };
#else // LEGACY_COMPATIBILITY
      EBM_ASSERT(0 < cInstances); // we shouldn't be making splits with children with less than 1 instance
      return sumResidualError / cInstances * sumResidualError;
#endif // LEGACY_COMPATIBILITY
   }

   EBM_INLINE static FractionalDataType ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(const FractionalDataType sumResidualError, const FractionalDataType sumDenominator) {
#ifdef LEGACY_COMPATIBILITY
      return LIKELY(FractionalDataType { 0 } != sumDenominator) ? sumResidualError / sumDenominator : FractionalDataType { 0 };
#else // LEGACY_COMPATIBILITY
      // sumDenominator should really never be zero since we never allow childrent instances to be less than zero, and we special case 0 instances in the dataset.
      // This means that we should only hit zero for the sumDenominator when we have very few instances AND ComputeNewtonRaphsonStep computed zero, which should only happen if absResidualError
      // is either very close to zero or very close to 1, which implies 100% certainty in one direction.  If we have a dataset that has very few cases for one
      // feature value, then we might see this, but we can also get NaN or infinities if we have extremely high values for something, like having SIZE_T_MAX - 1 instances
      // or other such extreme situations, so we should probably detect results that lead to NaN or +- infinities as results and refuse to allow updates in such extreme
      // conditions.  We can handle this in the caller though since they'll get back a NaN or inifinity in the log loss or MSE on the test set, or maybe we should
      // just report NaN or infnity back to the user anyways, since it's a true reflection of the result.
      // We can handle situations where the resulting update has a NaN or +-infinity or if adding the update to our model feature tensors leads to a NaN or +-infinity, but we can't easily
      // check to see if we'll get a NaN or +-infinity in any validation instance, which can happen if we have two extreme features that individually don't overflow but together do
      // So, we'll have to accept that it's possible for NaN or +- infinity to occur, and we'll just detect those conditions and treat them as really extreme outcomes
      // Also, once we add in our small non-observed outcome values (based on # of instances) then we'll be more protected against extreme outcomes.
      // 
      // TODO : after we've generated our small tensor update, check the values to see if any are NaN or +- inf and make the update zero if so.  Also, check to see if we added the small tensor update to our existing model tensor if we'd get any NaN or +- inf, and zero the update if that's true also
      //
      // TODO : In the future we'll add a little bit of uncertainty floating point value to the residual error, because you can never get 100% certainty for any prediction
      //        If you only see 100,000 cases, then you can't really make statements beyond that something happens less than arround 1/100,000 of the time
      return sumResidualError / sumDenominator;
#endif // LEGACY_COMPATIBILITY
   }

   EBM_INLINE static FractionalDataType ComputeSmallChangeInRegressionPredictionForOneSegment(const FractionalDataType sumResidualError, const FractionalDataType cInstances) {
#ifdef LEGACY_COMPATIBILITY
      return LIKELY(FractionalDataType { 0 } != cInstances) ? sumResidualError / cInstances : FractionalDataType { 0 };
#else // LEGACY_COMPATIBILITY
      EBM_ASSERT(0 < cInstances); // we shouldn't be making splits with children with less than 1 instance
      return sumResidualError / cInstances;
#endif // LEGACY_COMPATIBILITY
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