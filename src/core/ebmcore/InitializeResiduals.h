// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef INITIALIZE_RESIDUALS_H
#define INITIALIZE_RESIDUALS_H

#include <assert.h>
#include <stddef.h> // size_t, ptrdiff_t

#include "ebmcore.h"
#include "EbmStatistics.h"
#include "Logging.h" // EBM_ASSERT & LOG

// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
template<ptrdiff_t countCompilerClassificationTargetStates>
static void InitializeResiduals(const size_t cCases, const void * const aTargetData, const FractionalDataType * const aPredictionScores, FractionalDataType * pResidualError, const size_t cTargetStates, const int iZeroResidual) {
   LOG(TraceLevelInfo, "Entered InitializeResiduals");

   // TODO : review this function to see if iZeroResidual was set to a valid index, does that affect the number of items in pPredictionScores (I assume so), and does it affect any calculations below like sumExp += std::exp(predictionScore) and the equivalent.  Should we use cVectorLength or cTargetStates for some of the addition
   // TODO : !!! re-examine the idea of zeroing one of the residuals with iZeroResidual.  Do we get exact equivalent results if we initialize them the correct way.  Try debugging this by first doing a binary as multiclass (2 == cVectorLength) and seeing if our algorithm is re-startable (do 2 cycles and then try doing 1 cycle and exiting then re-creating it with aPredictionScore values and doing a 2nd cycle and see if it gives the same results).  It would be a huge win to be able to consitently eliminate one residual value!).  Maybe try construcing a super-simple dataset with 10 cases and 1 attribute and see how it behaves
   EBM_ASSERT(0 < cCases);
   EBM_ASSERT(nullptr != aTargetData);
   EBM_ASSERT(nullptr != pResidualError);

   const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
   EBM_ASSERT(0 < cVectorLength);
   EBM_ASSERT(!IsMultiplyError(cVectorLength, cCases)); // if we couldn't multiply these then we should not have been able to allocate pResidualError before calling this function
   const size_t cVectoredItems = cVectorLength * cCases;
   EBM_ASSERT(!IsMultiplyError(cVectoredItems, sizeof(pResidualError[0]))); // if we couldn't multiply these then we should not have been able to allocate pResidualError before calling this function
   const FractionalDataType * const pResidualErrorEnd = pResidualError + cVectoredItems;

   if(nullptr == aPredictionScores) {
      // TODO: do we really need to handle the case where pPredictionScores is null? In the future, we'll probably initialize our data with the intercept, in which case we'll always have existing predictions
      if(IsRegression(countCompilerClassificationTargetStates)) {
         // calling ComputeRegressionResidualError(predictionScore, data) with predictionScore as zero gives just data, so we can memcopy these values
         memcpy(pResidualError, aTargetData, cCases * sizeof(pResidualError[0]));
#ifndef NDEBUG
         const FractionalDataType * pTargetData = static_cast<const FractionalDataType *>(aTargetData);
         do {
            const FractionalDataType data = *pTargetData;
            EBM_ASSERT(!std::isnan(data));
            EBM_ASSERT(!std::isinf(data));
            const FractionalDataType predictionScore = 0;
            const FractionalDataType residualError = ComputeRegressionResidualError(predictionScore, data);
            EBM_ASSERT(*pResidualError == residualError);
            ++pTargetData;
            ++pResidualError;
         } while(pResidualErrorEnd != pResidualError);
#endif // NDEBUG
      } else {
         EBM_ASSERT(IsClassification(countCompilerClassificationTargetStates));

         const IntegerDataType * pTargetData = static_cast<const IntegerDataType *>(aTargetData);

         const FractionalDataType matchValue = ComputeClassificationResidualErrorMulticlass(true, static_cast<FractionalDataType>(cVectorLength));
         const FractionalDataType nonMatchValue = ComputeClassificationResidualErrorMulticlass(false, static_cast<FractionalDataType>(cVectorLength));

         EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, size_t>(cVectorLength)));
         const StorageDataTypeCore cVectorLengthStorage = static_cast<StorageDataTypeCore>(cVectorLength);

         do {
            const IntegerDataType dataOriginal = *pTargetData;
            EBM_ASSERT(0 <= dataOriginal);
            EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, IntegerDataType>(dataOriginal))); // if we can't fit it, then we should increase our StorageDataTypeCore size!
            const StorageDataTypeCore data = static_cast<StorageDataTypeCore>(dataOriginal);
            EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, size_t>(cTargetStates)));
            EBM_ASSERT(data < static_cast<StorageDataTypeCore>(cTargetStates));

            if(IsBinaryClassification(countCompilerClassificationTargetStates)) {
               const FractionalDataType residualError = ComputeClassificationResidualErrorBinaryclass(data);
               *pResidualError = residualError;
               ++pResidualError;
            } else {
               for(StorageDataTypeCore iVector = 0; iVector < cVectorLengthStorage; ++iVector) {
                  const FractionalDataType residualError = ComputeClassificationResidualErrorMulticlass(data, iVector, matchValue, nonMatchValue);
                  EBM_ASSERT(ComputeClassificationResidualErrorMulticlass(static_cast<FractionalDataType>(cVectorLength), 0, data, iVector) == residualError);
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
               if(0 <= iZeroResidual) {
                  pResidualError[static_cast<ptrdiff_t>(iZeroResidual) - static_cast<ptrdiff_t>(cVectorLength)] = 0;
               }
            }
            ++pTargetData;
         } while(pResidualErrorEnd != pResidualError);
      }
   } else {
      const FractionalDataType * pPredictionScores = aPredictionScores;
      if(IsRegression(countCompilerClassificationTargetStates)) {
         const FractionalDataType * pTargetData = static_cast<const FractionalDataType *>(aTargetData);
         do {
            const FractionalDataType data = *pTargetData;
            EBM_ASSERT(!std::isnan(data));
            EBM_ASSERT(!std::isinf(data));
            const FractionalDataType predictionScore = *pPredictionScores;
            const FractionalDataType residualError = ComputeRegressionResidualError(predictionScore, data);
            *pResidualError = residualError;
            ++pTargetData;
            ++pPredictionScores;
            ++pResidualError;
         } while(pResidualErrorEnd != pResidualError);
      } else {
         EBM_ASSERT(IsClassification(countCompilerClassificationTargetStates));

         const IntegerDataType * pTargetData = static_cast<const IntegerDataType *>(aTargetData);

         EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, size_t>(cVectorLength)));
         const StorageDataTypeCore cVectorLengthStorage = static_cast<StorageDataTypeCore>(cVectorLength);

         do {
            const IntegerDataType dataOriginal = *pTargetData;
            EBM_ASSERT(0 <= dataOriginal);
            EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, IntegerDataType>(dataOriginal))); // if we can't fit it, then we should increase our StorageDataTypeCore size!
            const StorageDataTypeCore data = static_cast<StorageDataTypeCore>(dataOriginal);
            EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, size_t>(cTargetStates)));
            EBM_ASSERT(data < static_cast<StorageDataTypeCore>(cTargetStates));

            if(IsBinaryClassification(countCompilerClassificationTargetStates)) {
               const FractionalDataType predictionScore = *pPredictionScores;
               const FractionalDataType residualError = ComputeClassificationResidualErrorBinaryclass(predictionScore, data);
               *pResidualError = residualError;
               ++pPredictionScores;
               ++pResidualError;
            } else {
               FractionalDataType sumExp = 0;
               for(StorageDataTypeCore iVector = 0; iVector < cVectorLengthStorage; ++iVector) {
                  const FractionalDataType predictionScore = *pPredictionScores;
                  sumExp += std::exp(predictionScore);
                  ++pPredictionScores;
               }

               // go back to the start so that we can iterate again
               pPredictionScores -= cVectorLengthStorage;

               for(StorageDataTypeCore iVector = 0; iVector < cVectorLengthStorage; ++iVector) {
                  const FractionalDataType predictionScore = *pPredictionScores;
                  const FractionalDataType residualError = ComputeClassificationResidualErrorMulticlass(sumExp, predictionScore, data, iVector);
                  *pResidualError = residualError;
                  ++pPredictionScores;
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
               if(0 <= iZeroResidual) {
                  pResidualError[static_cast<ptrdiff_t>(iZeroResidual) - static_cast<ptrdiff_t>(cVectorLengthStorage)] = 0;
               }
            }
            ++pTargetData;
         } while(pResidualErrorEnd != pResidualError);
      }
   }
   LOG(TraceLevelInfo, "Exited InitializeResiduals");
}

TML_INLINE static void InitializeResidualsFlat(const bool bRegression, const size_t cCases, const void * const aTargetData, const FractionalDataType * const aPredictionScores, FractionalDataType * pResidualError, const size_t cTargetStates, const int iZeroResidual) {
   if(bRegression) {
      InitializeResiduals<k_Regression>(cCases, aTargetData, aPredictionScores, pResidualError, 0, iZeroResidual);
   } else {
      if(2 == cTargetStates) {
         InitializeResiduals<2>(cCases, aTargetData, aPredictionScores, pResidualError, 2, iZeroResidual);
      } else {
         InitializeResiduals<k_DynamicClassification>(cCases, aTargetData, aPredictionScores, pResidualError, cTargetStates, iZeroResidual);
      }
   }
}

#endif // INITIALIZE_RESIDUALS_H
