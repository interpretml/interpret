// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef INITIALIZE_RESIDUALS_H
#define INITIALIZE_RESIDUALS_H

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "EbmStatistics.h"
#include "Logging.h" // EBM_ASSERT & LOG

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
static void InitializeResiduals(const size_t cInstances, const void * const aTargetData, const FloatEbmType * const aPredictorScores, FloatEbmType * pResidualError, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses) {
   LOG_0(TraceLevelInfo, "Entered InitializeResiduals");

   // TODO : review this function to see if iZeroResidual was set to a valid index, does that affect the number of items in pPredictorScores (I assume so), and does it affect any calculations below like sumExp += std::exp(predictionScore) and the equivalent.  Should we use cVectorLength or runtimeLearningTypeOrCountTargetClasses for some of the addition
   // TODO : !!! re-examine the idea of zeroing one of the residuals with iZeroResidual after we have the ability to test large numbers of datasets
   EBM_ASSERT(0 < cInstances);
   EBM_ASSERT(nullptr != aTargetData);
   EBM_ASSERT(nullptr != pResidualError);

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses);
   EBM_ASSERT(0 < cVectorLength);
   EBM_ASSERT(!IsMultiplyError(cVectorLength, cInstances)); // if we couldn't multiply these then we should not have been able to allocate pResidualError before calling this function
   const size_t cVectoredItems = cVectorLength * cInstances;
   EBM_ASSERT(!IsMultiplyError(cVectoredItems, sizeof(pResidualError[0]))); // if we couldn't multiply these then we should not have been able to allocate pResidualError before calling this function
   const FloatEbmType * const pResidualErrorEnd = pResidualError + cVectoredItems;

   if(nullptr == aPredictorScores) {
      // TODO: do we really need to handle the case where pPredictorScores is null? In the future, we'll probably initialize our data with the intercept, in which case we'll always have existing predictions
      if(IsRegression(compilerLearningTypeOrCountTargetClasses)) {
         // calling ComputeResidualErrorRegressionInit(predictionScore, data) with predictionScore as zero gives just data, so we can memcopy these values

         // if there is a NaN or +-infinity value in aTargetData, we'll probably be exiting pretty quickly since we'll get NaN values everywhere.  We can just pass the
         // bad input on here and we'll detect it later
         // TODO: NaN target values essentially mean missing, so we should be filtering those instances out, but our caller should do that so that we don't need to do the work here per outer bag.  Our job in C++ is just not to crash or return inexplicable values.
         memcpy(pResidualError, aTargetData, cInstances * sizeof(pResidualError[0]));
#ifndef NDEBUG
         const FloatEbmType * pTargetData = static_cast<const FloatEbmType *>(aTargetData);
         do {
            const FloatEbmType data = *pTargetData;
            // TODO : our caller should handle NaN *pTargetData values, which means that the target is missing, which means we should delete that instance from the input data

            // if data is NaN, we pass this along and NaN propagation will ensure that we stop boosting immediately.  
            // There is no need to check it here since we already have graceful detection later for other reasons.
            const FloatEbmType predictionScore = 0;
            const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorRegressionInit(predictionScore, data);
            EBM_ASSERT(std::abs(*pResidualError - residualError) <= k_epsilonResidualError);
            ++pTargetData;
            ++pResidualError;
         } while(pResidualErrorEnd != pResidualError);
#endif // NDEBUG
      } else {
         EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));

         const IntEbmType * pTargetData = static_cast<const IntEbmType *>(aTargetData);

         const FloatEbmType matchValue = EbmStatistics::ComputeResidualErrorMulticlassInitZero(true, runtimeLearningTypeOrCountTargetClasses);
         const FloatEbmType nonMatchValue = EbmStatistics::ComputeResidualErrorMulticlassInitZero(false, runtimeLearningTypeOrCountTargetClasses);

         EBM_ASSERT((IsNumberConvertable<StorageDataType, size_t>(cVectorLength)));
         const StorageDataType cVectorLengthStorage = static_cast<StorageDataType>(cVectorLength);

         do {
            const IntEbmType targetOriginal = *pTargetData;
            EBM_ASSERT(0 <= targetOriginal);
            EBM_ASSERT((IsNumberConvertable<StorageDataType, IntEbmType>(targetOriginal))); // if we can't fit it, then we should increase our StorageDataType size!
            const StorageDataType target = static_cast<StorageDataType>(targetOriginal);
            EBM_ASSERT((IsNumberConvertable<StorageDataType, ptrdiff_t>(runtimeLearningTypeOrCountTargetClasses)));
            EBM_ASSERT(target < static_cast<StorageDataType>(runtimeLearningTypeOrCountTargetClasses));

            if(IsBinaryClassification(compilerLearningTypeOrCountTargetClasses)) {
               const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorBinaryClassificationInitZero(target);
               *pResidualError = residualError;
               ++pResidualError;
            } else {
               for(StorageDataType iVector = 0; iVector < cVectorLengthStorage; ++iVector) {
                  const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorMulticlassInitZero(target, iVector, matchValue, nonMatchValue);
                  EBM_ASSERT(!std::isnan(residualError)); // without a aPredictorScores value, the logits are all zero, and we can't really overflow the exps
                  EBM_ASSERT(std::abs(EbmStatistics::ComputeResidualErrorMulticlass(static_cast<FloatEbmType>(runtimeLearningTypeOrCountTargetClasses), FloatEbmType { 0 }, target, iVector) - residualError) <= k_epsilonResidualError);
                  *pResidualError = residualError;
                  ++pResidualError;
               }
               // TODO: this works as a way to remove one parameter, but it obviously insn't as efficient as omitting the parameter
               // 
               // this works out in the math as making the first model vector parameter equal to zero, which in turn removes one degree of freedom
               // from the model vector parameters.  Since the model vector weights need to be normalized to sum to a probabilty of 100%, we can set the first
               // one to the constant 1 (0 in log space) and force the other parameters to adjust to that scale which fixes them to a single valid set of values
               // insted of allowing them to be scaled.  
               // Probability = exp(T1 + I1) / [exp(T1 + I1) + exp(T2 + I2) + exp(T3 + I3)] => we can add a constant inside each exp(..) term, which will be multiplication outside the exp(..), which
               // means the numerator and denominator are multiplied by the same constant, which cancels eachother out.  We can thus set exp(T2 + I2) to exp(0) and adjust the other terms
               constexpr bool bZeroingResiduals = 0 <= k_iZeroResidual;
               if(bZeroingResiduals) {
                  pResidualError[k_iZeroResidual - static_cast<ptrdiff_t>(cVectorLength)] = 0;
               }
            }
            ++pTargetData;
         } while(pResidualErrorEnd != pResidualError);
      }
   } else {
      const FloatEbmType * pPredictorScores = aPredictorScores;
      if(IsRegression(compilerLearningTypeOrCountTargetClasses)) {
         const FloatEbmType * pTargetData = static_cast<const FloatEbmType *>(aTargetData);
         do {
            // TODO : our caller should handle NaN *pTargetData values, which means that the target is missing, which means we should delete that instance from the input data

            // if data is NaN, we pass this along and NaN propagation will ensure that we stop boosting immediately.
            // There is no need to check it here since we already have graceful detection later for other reasons.

            const FloatEbmType data = *pTargetData;
            // TODO: NaN target values essentially mean missing, so we should be filtering those instances out, but our caller should do that so that we don't need to do the work here per outer bag.  Our job in C++ is just not to crash or return inexplicable values.
            const FloatEbmType predictionScore = *pPredictorScores;
            const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorRegressionInit(predictionScore, data);
            *pResidualError = residualError;
            ++pTargetData;
            ++pPredictorScores;
            ++pResidualError;
         } while(pResidualErrorEnd != pResidualError);
      } else {
         EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));

         const IntEbmType * pTargetData = static_cast<const IntEbmType *>(aTargetData);

         EBM_ASSERT((IsNumberConvertable<StorageDataType, size_t>(cVectorLength)));
         const StorageDataType cVectorLengthStorage = static_cast<StorageDataType>(cVectorLength);

         do {
            const IntEbmType targetOriginal = *pTargetData;
            EBM_ASSERT(0 <= targetOriginal);
            EBM_ASSERT((IsNumberConvertable<StorageDataType, IntEbmType>(targetOriginal))); // if we can't fit it, then we should increase our StorageDataType size!
            const StorageDataType target = static_cast<StorageDataType>(targetOriginal);
            EBM_ASSERT((IsNumberConvertable<StorageDataType, ptrdiff_t>(runtimeLearningTypeOrCountTargetClasses)));
            EBM_ASSERT(target < static_cast<StorageDataType>(runtimeLearningTypeOrCountTargetClasses));
            if(IsBinaryClassification(compilerLearningTypeOrCountTargetClasses)) {
               const FloatEbmType predictionScore = *pPredictorScores;
               const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorBinaryClassification(predictionScore, target);
               *pResidualError = residualError;
               ++pPredictorScores;
               ++pResidualError;
            } else {
               FloatEbmType sumExp = 0;
               // TODO : eventually eliminate this subtract variable once we've decided how to handle removing one logit
               const FloatEbmType subtract = 0 <= k_iZeroClassificationLogitAtInitialize ? pPredictorScores[k_iZeroClassificationLogitAtInitialize] : 0;

               for(StorageDataType iVector = 0; iVector < cVectorLengthStorage; ++iVector) {
                  const FloatEbmType predictionScore = *pPredictorScores - subtract;
                  sumExp += EbmExp(predictionScore);
                  ++pPredictorScores;
               }

               // go back to the start so that we can iterate again
               pPredictorScores -= cVectorLengthStorage;

               for(StorageDataType iVector = 0; iVector < cVectorLengthStorage; ++iVector) {
                  const FloatEbmType predictionScore = *pPredictorScores - subtract;
                  // TODO : we're calculating exp(predictionScore) above, and then again in ComputeClassificationResidualErrorMulticlass.  exp(..) is expensive so we should just do it once instead and store the result in a small memory array here
                  const FloatEbmType residualError = EbmStatistics::ComputeResidualErrorMulticlass(sumExp, predictionScore, target, iVector);
                  *pResidualError = residualError;
                  ++pPredictorScores;
                  ++pResidualError;
               }
               // TODO: this works as a way to remove one parameter, but it obviously insn't as efficient as omitting the parameter
               // 
               // this works out in the math as making the first model vector parameter equal to zero, which in turn removes one degree of freedom
               // from the model vector parameters.  Since the model vector weights need to be normalized to sum to a probabilty of 100%, we can set the first
               // one to the constant 1 (0 in log space) and force the other parameters to adjust to that scale which fixes them to a single valid set of values
               // insted of allowing them to be scaled.  
               // Probability = exp(T1 + I1) / [exp(T1 + I1) + exp(T2 + I2) + exp(T3 + I3)] => we can add a constant inside each exp(..) term, which will be multiplication outside the exp(..), which
               // means the numerator and denominator are multiplied by the same constant, which cancels eachother out.  We can thus set exp(T2 + I2) to exp(0) and adjust the other terms
               constexpr bool bZeroingResiduals = 0 <= k_iZeroResidual;
               if(bZeroingResiduals) {
                  pResidualError[k_iZeroResidual - static_cast<ptrdiff_t>(cVectorLengthStorage)] = 0;
               }
            }
            ++pTargetData;
         } while(pResidualErrorEnd != pResidualError);
      }
   }
   LOG_0(TraceLevelInfo, "Exited InitializeResiduals");
}

#endif // INITIALIZE_RESIDUALS_H
