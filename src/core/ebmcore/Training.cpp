// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <string.h> // memset
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebmcore.h"

#include "EbmInternal.h"
// very independent includes
#include "RandomStream.h"
#include "SegmentedRegion.h"
#include "EbmStatistics.h"
// this depends on TreeNode pointers, but doesn't require the full definition of TreeNode
#include "CachedThreadResources.h"
// attribute includes
#include "AttributeInternal.h"
#include "AttributeSet.h"
// AttributeCombination.h might in the future depend on AttributeSetInternalCore.h
#include "AttributeCombinationInternal.h"
// dataset depends on attributes
#include "DataSetByAttributeCombination.h"
// samples is somewhat independent from datasets, but relies on an indirect coupling with them
#include "SamplingWithReplacement.h"
// TreeNode depends on almost everything
#include "SingleDimensionalTraining.h"
#include "MultiDimensionalTraining.h"

static void DeleteSegmentsCore(const size_t cAttributeCombinations, SegmentedRegionCore<ActiveDataType, FractionalDataType> ** apSegmentedRegions) {
   assert(0 < cAttributeCombinations);
   if(UNLIKELY(nullptr != apSegmentedRegions)) {
      SegmentedRegionCore<ActiveDataType, FractionalDataType> ** ppSegmentedRegions = apSegmentedRegions;
      const SegmentedRegionCore<ActiveDataType, FractionalDataType> * const * const ppSegmentedRegionsEnd = &apSegmentedRegions[cAttributeCombinations];
      do {
         SegmentedRegionCore<ActiveDataType, FractionalDataType>::Free(*ppSegmentedRegions);
         ++ppSegmentedRegions;
      } while(ppSegmentedRegionsEnd != ppSegmentedRegions);
      delete[] apSegmentedRegions;
   }
}

static SegmentedRegionCore<ActiveDataType, FractionalDataType> ** InitializeSegmentsCore(const size_t cAttributeCombinations, const AttributeCombinationCore * const * const apAttributeCombinations, const size_t cVectorLength) {
   SegmentedRegionCore<ActiveDataType, FractionalDataType> ** apSegmentedRegions;
   apSegmentedRegions = new (std::nothrow) SegmentedRegionCore<ActiveDataType, FractionalDataType> *[cAttributeCombinations];
   if(UNLIKELY(nullptr == apSegmentedRegions)) {
      return nullptr;
   }
   memset(apSegmentedRegions, 0, sizeof(*apSegmentedRegions) * cAttributeCombinations); // this needs to be done immediately after allocation otherwise we might attempt to free random garbage on an error

   SegmentedRegionCore<ActiveDataType, FractionalDataType> ** ppSegmentedRegions = apSegmentedRegions;
   for(size_t iAttributeCombination = 0; iAttributeCombination < cAttributeCombinations; ++iAttributeCombination) {
      const AttributeCombinationCore * const pAttributeCombination = apAttributeCombinations[iAttributeCombination];
      SegmentedRegionCore<ActiveDataType, FractionalDataType> * pSegmentedRegions = SegmentedRegionCore<ActiveDataType, FractionalDataType>::Allocate(pAttributeCombination->m_cAttributes, cVectorLength);
      if(UNLIKELY(nullptr == pSegmentedRegions)) {
         DeleteSegmentsCore(cAttributeCombinations, apSegmentedRegions);
         return nullptr;
      }

      // TODO optimize the next few lines
      // TODO there might be a nicer way to expand this at allocation time (fill with zeros is easier)
      // we want to return a pointer to our interior state in the GetCurrentModel and GetBestModel functions.  For simplicity we don't transmit the divions, so we need to expand our SegmentedRegion before returning
      // the easiest way to ensure that the SegmentedRegion is expanded is to start it off expanded, and then we don't have to check later since anything merged into an expanded SegmentedRegion will itself be expanded
      size_t acDivisionIntegersEnd[k_cDimensionsMax];
      for(size_t iDimension = 0; iDimension < pAttributeCombination->m_cAttributes; ++iDimension) {
         acDivisionIntegersEnd[iDimension] = pAttributeCombination->m_AttributeCombinationEntry[iDimension].m_pAttribute->m_cStates;
      }
      if(pSegmentedRegions->Expand(acDivisionIntegersEnd)) {
         DeleteSegmentsCore(cAttributeCombinations, apSegmentedRegions);
         return nullptr;
      }

      *ppSegmentedRegions = pSegmentedRegions;
      ++ppSegmentedRegions;
   }
   return apSegmentedRegions;
}

// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
template<ptrdiff_t countCompilerClassificationTargetStates>
static void InitializeErrorCore(const size_t cCases, const void * const aTargetData, const FractionalDataType * pPredictionScores, FractionalDataType * pResidualError, const size_t cTargetStates, const int iZeroResidual) {
   const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
   const FractionalDataType * const pResidualErrorEnd = pResidualError + cVectorLength * cCases;

   if(nullptr == pPredictionScores) {
      if(IsRegression(countCompilerClassificationTargetStates)) {
         // calling ComputeRegressionResidualError(predictionScore, data) with predictionScore as zero gives just data, so we can memcopy these values
         memcpy(pResidualError, aTargetData, cCases * sizeof(pResidualError[0]));
#ifndef NDEBUG
         const FractionalDataType * pTargetData = static_cast<const FractionalDataType *>(aTargetData);
         do {
            FractionalDataType data = *pTargetData;
            FractionalDataType predictionScore = 0;
            FractionalDataType residualError = ComputeRegressionResidualError(predictionScore, data);
            assert(*pResidualError == residualError);
            ++pTargetData;
            ++pResidualError;
         } while(pResidualErrorEnd != pResidualError);
#endif // NDEBUG
      } else {
         assert(IsClassification(countCompilerClassificationTargetStates));

         const StorageDataTypeCore * pTargetData = static_cast<const StorageDataTypeCore *>(aTargetData);

         const FractionalDataType matchValue = ComputeClassificationResidualErrorMulticlass(true, static_cast<FractionalDataType>(cVectorLength));
         const FractionalDataType nonMatchValue = ComputeClassificationResidualErrorMulticlass(false, static_cast<FractionalDataType>(cVectorLength));

         assert((IsNumberConvertable<StorageDataTypeCore, size_t>(cVectorLength)));
         const StorageDataTypeCore cVectorLengthStorage = static_cast<StorageDataTypeCore>(cVectorLength);

         do {
            const StorageDataTypeCore data = *pTargetData;
            if(IsBinaryClassification(countCompilerClassificationTargetStates)) {
               const FractionalDataType residualError = ComputeClassificationResidualErrorBinaryclass(data);
               *pResidualError = residualError;
               ++pResidualError;
            } else {
               for(StorageDataTypeCore iVector = 0; iVector < cVectorLengthStorage; ++iVector) {
                  const FractionalDataType residualError = ComputeClassificationResidualErrorMulticlass(data, iVector, matchValue, nonMatchValue);
                  assert(ComputeClassificationResidualErrorMulticlass(static_cast<FractionalDataType>(cVectorLength), 0, data, iVector) == residualError);
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
      if(IsRegression(countCompilerClassificationTargetStates)) {
         const FractionalDataType * pTargetData = static_cast<const FractionalDataType *>(aTargetData);
         do {
            const FractionalDataType data = *pTargetData;
            const FractionalDataType predictionScore = *pPredictionScores;
            const FractionalDataType residualError = ComputeRegressionResidualError(predictionScore, data);
            *pResidualError = residualError;
            ++pTargetData;
            ++pPredictionScores;
            ++pResidualError;
         } while(pResidualErrorEnd != pResidualError);
      } else {
         assert(IsClassification(countCompilerClassificationTargetStates));

         const StorageDataTypeCore * pTargetData = static_cast<const StorageDataTypeCore *>(aTargetData);

         assert((IsNumberConvertable<StorageDataTypeCore, size_t>(cVectorLength)));
         const StorageDataTypeCore cVectorLengthStorage = static_cast<StorageDataTypeCore>(cVectorLength);

         do {
            StorageDataTypeCore data = *pTargetData;
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
                  FractionalDataType predictionScore = *pPredictionScores;
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
}

// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
template<unsigned int cInputBits, unsigned int cTargetBits, ptrdiff_t countCompilerClassificationTargetStates>
static void TrainingSetTargetAttributeLoop(const AttributeCombinationCore * const pAttributeCombination, DataSetAttributeCombination * const pTrainingSet, SegmentedRegionCore<ActiveDataType, FractionalDataType> * const pSmallChangeToModel, const size_t cTargetStates, int iZeroResidual) {
   const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
   const size_t cItemsPerBitPackDataUnit = pAttributeCombination->m_cItemsPerBitPackDataUnit;
   const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackDataUnit);
   const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);

   const size_t cCases = pTrainingSet->GetCountCases();
   assert(0 < cCases);

   const StorageDataTypeCore * pInputData = pTrainingSet->GetDataPointer(pAttributeCombination);
   FractionalDataType * pResidualError = pTrainingSet->GetResidualPointer();
   const FractionalDataType * const pResidualErrorLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete = pResidualError + cVectorLength * (static_cast<ptrdiff_t>(cCases) - cItemsPerBitPackDataUnit);

   if(IsRegression(countCompilerClassificationTargetStates)) {
      size_t cItemsRemaining;
      while(pResidualError < pResidualErrorLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete) {
         cItemsRemaining = cItemsPerBitPackDataUnit;
         // TODO : jumping back into this loop and changing cItemsRemaining to a dynamic value that isn't compile time determinable
         // causes this function to NOT be optimized as much as it could if we had two separate loops.  We're just trying this out for now though
      one_last_loop_regression:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            const size_t iBin = maskBits & iBinCombined;
            const FractionalDataType * pValues = pSmallChangeToModel->GetValueDirect(iBin);

            const FractionalDataType smallChangeToPrediction = pValues[0];
            // this will apply a small fix to our existing TrainingPredictionScores, either positive or negative, whichever is needed
            const FractionalDataType residualError = ComputeRegressionResidualError(*pResidualError - smallChangeToPrediction);
            *pResidualError = residualError;
            ++pResidualError;

            iBinCombined >>= cBitsPerItemMax;
            // TODO : try replacing cItemsRemaining with a pResidualErrorInnerLoopEnd which eliminates one subtact operation, but might make it harder for the compiler to optimize the loop away
            --cItemsRemaining;
         } while(0 != cItemsRemaining);
      }
      const FractionalDataType * const pResidualErrorEnd = pResidualErrorLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete + cVectorLength * cItemsPerBitPackDataUnit;
      if(pResidualError < pResidualErrorEnd) {
         // first time through?
         assert(0 == (pResidualErrorEnd - pResidualError) % cVectorLength);
         cItemsRemaining = (pResidualErrorEnd - pResidualError) / cVectorLength;
         assert(0 < cItemsRemaining);
         assert(cItemsRemaining <= cItemsPerBitPackDataUnit);
         goto one_last_loop_regression;
      }
      assert(pResidualError == pResidualErrorEnd); // after our second iteration we should have finished everything!
   } else {
      FractionalDataType * pTrainingPredictionScores = pTrainingSet->GetPredictionScores();
      assert(IsClassification(countCompilerClassificationTargetStates));
      const StorageDataTypeCore * pTargetData = static_cast<const StorageDataTypeCore *>(pTrainingSet->GetTargetDataPointer());

      size_t cItemsRemaining;

      while(pResidualError < pResidualErrorLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete) {
         cItemsRemaining = cItemsPerBitPackDataUnit;
         // TODO : jumping back into this loop and changing cItemsRemaining to a dynamic value that isn't compile time determinable
         // causes this function to NOT be optimized as much as it could if we had two separate loops.  We're just trying this out for now though
      one_last_loop_classification:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            StorageDataTypeCore targetData = *pTargetData;

            const size_t iBin = maskBits & iBinCombined;
            const FractionalDataType * pValues = pSmallChangeToModel->GetValueDirect(iBin);

            if(IsBinaryClassification(countCompilerClassificationTargetStates)) {
               const FractionalDataType smallChangeToPredictionScores = pValues[0];
               // this will apply a small fix to our existing TrainingPredictionScores, either positive or negative, whichever is needed
               const FractionalDataType trainingPredictionScore = *pTrainingPredictionScores + smallChangeToPredictionScores;
               *pTrainingPredictionScores = trainingPredictionScore;
               const FractionalDataType residualError = ComputeClassificationResidualErrorBinaryclass(trainingPredictionScore, targetData);
               *pResidualError = residualError;
               ++pResidualError;
            } else {
               FractionalDataType sumExp = 0;
               size_t iVector1 = 0;
               do {
                  const FractionalDataType smallChangeToPredictionScores = pValues[iVector1];
                  // this will apply a small fix to our existing TrainingPredictionScores, either positive or negative, whichever is needed
                  const FractionalDataType trainingPredictionScores = pTrainingPredictionScores[iVector1] + smallChangeToPredictionScores;
                  pTrainingPredictionScores[iVector1] = trainingPredictionScores;
                  sumExp += std::exp(trainingPredictionScores);
                  ++iVector1;
               } while(iVector1 < cVectorLength);

               assert((IsNumberConvertable<StorageDataTypeCore, size_t>(cVectorLength)));
               const StorageDataTypeCore cVectorLengthStorage = static_cast<StorageDataTypeCore>(cVectorLength);
               StorageDataTypeCore iVector2 = 0;
               do {
                  const FractionalDataType residualError = ComputeClassificationResidualErrorMulticlass(sumExp, pTrainingPredictionScores[iVector2], targetData, iVector2);
                  *pResidualError = residualError;
                  ++pResidualError;
                  ++iVector2;
               } while(iVector2 < cVectorLengthStorage);
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
            pTrainingPredictionScores += cVectorLength;
            ++pTargetData;

            iBinCombined >>= cBitsPerItemMax;
            // TODO : try replacing cItemsRemaining with a pResidualErrorInnerLoopEnd which eliminates one subtact operation, but might make it harder for the compiler to optimize the loop away
            --cItemsRemaining;
         } while(0 != cItemsRemaining);
      }
      const FractionalDataType * const pResidualErrorEnd = pResidualErrorLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete + cVectorLength * cItemsPerBitPackDataUnit;
      if(pResidualError < pResidualErrorEnd) {
         // first time through?
         assert(0 == (pResidualErrorEnd - pResidualError) % cVectorLength);
         cItemsRemaining = (pResidualErrorEnd - pResidualError) / cVectorLength;
         assert(0 < cItemsRemaining);
         assert(cItemsRemaining <= cItemsPerBitPackDataUnit);
         goto one_last_loop_classification;
      }
      assert(pResidualError == pResidualErrorEnd); // after our second iteration we should have finished everything!
   }
}

// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
template<unsigned int cInputBits, ptrdiff_t countCompilerClassificationTargetStates>
static void TrainingSetInputAttributeLoop(const AttributeCombinationCore * const pAttributeCombination, DataSetAttributeCombination * const pTrainingSet, SegmentedRegionCore<ActiveDataType, FractionalDataType> * const pSmallChangeToModel, const size_t cTargetStates, int iZeroResidual) {
   if(cTargetStates <= 1 << 1) {
      TrainingSetTargetAttributeLoop<cInputBits, 1, countCompilerClassificationTargetStates>(pAttributeCombination, pTrainingSet, pSmallChangeToModel, cTargetStates, iZeroResidual);
   } else if(cTargetStates <= 1 << 2) {
      TrainingSetTargetAttributeLoop<cInputBits, 2, countCompilerClassificationTargetStates>(pAttributeCombination, pTrainingSet, pSmallChangeToModel, cTargetStates, iZeroResidual);
   } else if(cTargetStates <= 1 << 4) {
      TrainingSetTargetAttributeLoop<cInputBits, 4, countCompilerClassificationTargetStates>(pAttributeCombination, pTrainingSet, pSmallChangeToModel, cTargetStates, iZeroResidual);
   } else if(cTargetStates <= 1 << 8) {
      TrainingSetTargetAttributeLoop<cInputBits, 8, countCompilerClassificationTargetStates>(pAttributeCombination, pTrainingSet, pSmallChangeToModel, cTargetStates, iZeroResidual);
   } else if(cTargetStates <= 1 << 16) {
      TrainingSetTargetAttributeLoop<cInputBits, 16, countCompilerClassificationTargetStates>(pAttributeCombination, pTrainingSet, pSmallChangeToModel, cTargetStates, iZeroResidual);
   } else if(cTargetStates <= static_cast<uint64_t>(1) << 32) {
      // if this is a 32 bit system, then m_cStates can't be 0x100000000 or above, because we would have checked that when converting the 64 bit numbers into size_t, and m_cStates will be promoted to a 64 bit number for the above comparison
      // if this is a 64 bit system, then this comparison is fine

      // TODO : perhaps we should change m_cStates into m_iStateMax so that we don't need to do the above promotion to 64 bits.. we can make it <= 0xFFFFFFFF.  Write a function to fill the lowest bits with ones for any number of bits

      TrainingSetTargetAttributeLoop<cInputBits, 32, countCompilerClassificationTargetStates>(pAttributeCombination, pTrainingSet, pSmallChangeToModel, cTargetStates, iZeroResidual);
   } else {
      // our interface doesn't allow more than 64 bits, so even if size_t was bigger then we don't need to examine higher
      static_assert(63 == CountBitsRequiredPositiveMax<IntegerDataType>(), "");
      TrainingSetTargetAttributeLoop<cInputBits, 64, countCompilerClassificationTargetStates>(pAttributeCombination, pTrainingSet, pSmallChangeToModel, cTargetStates, iZeroResidual);
   }
}

// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
template<unsigned int cInputBits, unsigned int cTargetBits, ptrdiff_t countCompilerClassificationTargetStates>
static FractionalDataType ValidationSetTargetAttributeLoop(const AttributeCombinationCore * const pAttributeCombination, DataSetAttributeCombination * const pValidationSet, SegmentedRegionCore<ActiveDataType, FractionalDataType> * const pSmallChangeToModel, const size_t cTargetStates) {
   const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
   const size_t cItemsPerBitPackDataUnit = pAttributeCombination->m_cItemsPerBitPackDataUnit;
   const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackDataUnit);
   const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);

   const size_t cCases = pValidationSet->GetCountCases();
   assert(0 < cCases);

   const StorageDataTypeCore * pInputData = pValidationSet->GetDataPointer(pAttributeCombination);

   if(IsRegression(countCompilerClassificationTargetStates)) {
      FractionalDataType * pResidualError = pValidationSet->GetResidualPointer();
      const FractionalDataType * const pResidualErrorLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete = pResidualError + cVectorLength * (static_cast<ptrdiff_t>(cCases) - cItemsPerBitPackDataUnit);

      FractionalDataType rootMeanSquareError = 0;
      size_t cItemsRemaining;
      while(pResidualError < pResidualErrorLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete) {
         cItemsRemaining = cItemsPerBitPackDataUnit;
         // TODO : jumping back into this loop and changing cItemsRemaining to a dynamic value that isn't compile time determinable
         // causes this function to NOT be optimized as much as it could if we had two separate loops.  We're just trying this out for now though
      one_last_loop_regression:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            const size_t iBin = maskBits & iBinCombined;
            const FractionalDataType * pValues = pSmallChangeToModel->GetValueDirect(iBin);

            const FractionalDataType smallChangeToPrediction = pValues[0];
            // this will apply a small fix to our existing ValidationPredictionScores, either positive or negative, whichever is needed
            const FractionalDataType residualError = ComputeRegressionResidualError(*pResidualError - smallChangeToPrediction);
            rootMeanSquareError += residualError * residualError;
            *pResidualError = residualError;
            ++pResidualError;

            iBinCombined >>= cBitsPerItemMax;
            // TODO : try replacing cItemsRemaining with a pResidualErrorInnerLoopEnd which eliminates one subtact operation, but might make it harder for the compiler to optimize the loop away
            --cItemsRemaining;
         } while(0 != cItemsRemaining);
      }
      const FractionalDataType * const pResidualErrorEnd = pResidualErrorLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete + cVectorLength * cItemsPerBitPackDataUnit;
      if(pResidualError < pResidualErrorEnd) {
         // first time through?
         assert(0 == (pResidualErrorEnd - pResidualError) % cVectorLength);
         cItemsRemaining = (pResidualErrorEnd - pResidualError) / cVectorLength;
         assert(0 < cItemsRemaining);
         assert(cItemsRemaining <= cItemsPerBitPackDataUnit);
         goto one_last_loop_regression;
      }
      assert(pResidualError == pResidualErrorEnd); // after our second iteration we should have finished everything!

      rootMeanSquareError /= pValidationSet->GetCountCases();
      return sqrt(rootMeanSquareError);
   } else {
      FractionalDataType * pValidationPredictionScores = pValidationSet->GetPredictionScores();
      assert(IsClassification(countCompilerClassificationTargetStates));
      const StorageDataTypeCore * pTargetData = static_cast<const StorageDataTypeCore *>(pValidationSet->GetTargetDataPointer());

      size_t cItemsRemaining;

      const FractionalDataType * const pValidationPredictionScoresLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete = pValidationPredictionScores + cVectorLength * (static_cast<ptrdiff_t>(cCases) - cItemsPerBitPackDataUnit);

      FractionalDataType sumLogLoss = 0;
      while(pValidationPredictionScores < pValidationPredictionScoresLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete) {
         cItemsRemaining = cItemsPerBitPackDataUnit;
         // TODO : jumping back into this loop and changing cItemsRemaining to a dynamic value that isn't compile time determinable
         // causes this function to NOT be optimized as much as it could if we had two separate loops.  We're just trying this out for now though
      one_last_loop_classification:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            StorageDataTypeCore targetData = *pTargetData;

            const size_t iBin = maskBits & iBinCombined;
            const FractionalDataType * pValues = pSmallChangeToModel->GetValueDirect(iBin);

            if(IsBinaryClassification(countCompilerClassificationTargetStates)) {
               const FractionalDataType smallChangeToPredictionScores = pValues[0];
               // this will apply a small fix to our existing ValidationPredictionScores, either positive or negative, whichever is needed
               const FractionalDataType validationPredictionScores = *pValidationPredictionScores + smallChangeToPredictionScores;
               *pValidationPredictionScores = validationPredictionScores;
               sumLogLoss += ComputeClassificationSingleCaseLogLossBinaryclass(validationPredictionScores, targetData);
               ++pValidationPredictionScores;
            } else {
               FractionalDataType sumExp = 0;
               size_t iVector = 0;
               do {
                  const FractionalDataType smallChangeToPredictionScores = pValues[iVector];
                  // this will apply a small fix to our existing validationPredictionScores, either positive or negative, whichever is needed

                  // TODO : this is no longer a prediction for multiclass.  It is a weight.  Change all instances of this naming. -> validationLogWeight
                  const FractionalDataType validationPredictionScores = *pValidationPredictionScores + smallChangeToPredictionScores;
                  *pValidationPredictionScores = validationPredictionScores;
                  sumExp += std::exp(validationPredictionScores);
                  ++pValidationPredictionScores;

                  // TODO : consider replacing iVector with pValidationPredictionScoresInnerEnd
                  ++iVector;
               } while(iVector < cVectorLength);
               sumLogLoss += ComputeClassificationSingleCaseLogLossMulticlass(sumExp, pValidationPredictionScores - cVectorLength, targetData);
            }
            ++pTargetData;

            iBinCombined >>= cBitsPerItemMax;
            // TODO : try replacing cItemsRemaining with a pResidualErrorInnerLoopEnd which eliminates one subtact operation, but might make it harder for the compiler to optimize the loop away
            --cItemsRemaining;
         } while(0 != cItemsRemaining);
      }

      const FractionalDataType * const pValidationPredictionScoresEnd = pValidationPredictionScoresLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete + cVectorLength * cItemsPerBitPackDataUnit;
      if(pValidationPredictionScores < pValidationPredictionScoresEnd) {
         // first time through?
         assert(0 == (pValidationPredictionScoresEnd - pValidationPredictionScores) % cVectorLength);
         cItemsRemaining = (pValidationPredictionScoresEnd - pValidationPredictionScores) / cVectorLength;
         assert(0 < cItemsRemaining);
         assert(cItemsRemaining <= cItemsPerBitPackDataUnit);
         goto one_last_loop_classification;
      }
      assert(pValidationPredictionScores == pValidationPredictionScoresEnd); // after our second iteration we should have finished everything!

      return sumLogLoss;
   }
}

// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
template<unsigned int cInputBits, ptrdiff_t countCompilerClassificationTargetStates>
static FractionalDataType ValidationSetInputAttributeLoop(const AttributeCombinationCore * const pAttributeCombination, DataSetAttributeCombination * const pValidationSet, SegmentedRegionCore<ActiveDataType, FractionalDataType> * const pSmallChangeToModel, const size_t cTargetStates) {
   if(cTargetStates <= 1 << 1) {
      return ValidationSetTargetAttributeLoop<cInputBits, 1, countCompilerClassificationTargetStates>(pAttributeCombination, pValidationSet, pSmallChangeToModel, cTargetStates);
   } else if(cTargetStates <= 1 << 2) {
      return ValidationSetTargetAttributeLoop<cInputBits, 2, countCompilerClassificationTargetStates>(pAttributeCombination, pValidationSet, pSmallChangeToModel, cTargetStates);
   } else if(cTargetStates <= 1 << 4) {
      return ValidationSetTargetAttributeLoop<cInputBits, 4, countCompilerClassificationTargetStates>(pAttributeCombination, pValidationSet, pSmallChangeToModel, cTargetStates);
   } else if(cTargetStates <= 1 << 8) {
      return ValidationSetTargetAttributeLoop<cInputBits, 8, countCompilerClassificationTargetStates>(pAttributeCombination, pValidationSet, pSmallChangeToModel, cTargetStates);
   } else if(cTargetStates <= 1 << 16) {
      return ValidationSetTargetAttributeLoop<cInputBits, 16, countCompilerClassificationTargetStates>(pAttributeCombination, pValidationSet, pSmallChangeToModel, cTargetStates);
   } else if(cTargetStates <= static_cast<uint64_t>(1) << 32) {
      // if this is a 32 bit system, then m_cStates can't be 0x100000000 or above, because we would have checked that when converting the 64 bit numbers into size_t, and m_cStates will be promoted to a 64 bit number for the above comparison
      // if this is a 64 bit system, then this comparison is fine

      // TODO : perhaps we should change m_cStates into m_iStateMax so that we don't need to do the above promotion to 64 bits.. we can make it <= 0xFFFFFFFF.  Write a function to fill the lowest bits with ones for any number of bits

      return ValidationSetTargetAttributeLoop<cInputBits, 32, countCompilerClassificationTargetStates>(pAttributeCombination, pValidationSet, pSmallChangeToModel, cTargetStates);
   } else {
      // our interface doesn't allow more than 64 bits, so even if size_t was bigger then we don't need to examine higher
      static_assert(63 == CountBitsRequiredPositiveMax<IntegerDataType>(), "");
      return ValidationSetTargetAttributeLoop<cInputBits, 64, countCompilerClassificationTargetStates>(pAttributeCombination, pValidationSet, pSmallChangeToModel, cTargetStates);
   }
}

// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
template<ptrdiff_t countCompilerClassificationTargetStates>
static bool GenerateModelLoop(SegmentedRegionCore<ActiveDataType, FractionalDataType> * const pSmallChangeToModelAccumulated, CachedTrainingThreadResources<IsRegression(countCompilerClassificationTargetStates)> * const pCachedThreadResources, const SamplingMethod * const * const apSamplingSets, const AttributeCombinationCore * const pAttributeCombination, const size_t cTreeSplitsMax, const size_t cCasesRequiredForSplitParentMin, SegmentedRegionCore<ActiveDataType, FractionalDataType> ** const apCurrentModel, SegmentedRegionCore<ActiveDataType, FractionalDataType> * const pSmallChangeToModelOverwrite, const size_t cAttributeCombinations, const size_t cSamplingSetsAfterZero, size_t iCurrentModel, const FractionalDataType learningRate, DataSetAttributeCombination * const pValidationSet, const size_t cTargetStates, FractionalDataType * pModelMetric) {
   size_t cDimensions = pAttributeCombination->m_cAttributes;

   pSmallChangeToModelAccumulated->SetCountDimensions(cDimensions);
   pSmallChangeToModelAccumulated->Reset();

   pSmallChangeToModelOverwrite->SetCountDimensions(cDimensions);

   for(size_t iSamplingSet = 0; iSamplingSet < cSamplingSetsAfterZero; ++iSamplingSet) {
      if(1 == pAttributeCombination->m_cAttributes) {
         if(TrainSingleDimensional<countCompilerClassificationTargetStates>(pCachedThreadResources, apSamplingSets[iSamplingSet], pAttributeCombination, cTreeSplitsMax, cCasesRequiredForSplitParentMin, pSmallChangeToModelOverwrite, cTargetStates)) {
            return true;
         }
      } else {
         if(TrainMultiDimensional<countCompilerClassificationTargetStates, 0>(pCachedThreadResources, apSamplingSets[iSamplingSet], pAttributeCombination, pSmallChangeToModelOverwrite, cTargetStates)) {
            return true;
         }
      }
      // GetThreadByteBuffer1 is overwritten inside the function above, so we need to obtain it here instead of higher
      void * pThreadBuffer = pCachedThreadResources->GetThreadByteBuffer1(pSmallChangeToModelAccumulated->GetStackMemorySizeBytes());
      if(UNLIKELY(nullptr == pThreadBuffer)) {
         return true;
      }
      // TODO : when we thread this code, let's have each thread take a lock and update the combined line segment.  They'll each do it while the others are working, so there should be no blocking and our final result won't require adding by the main thread
      if(pSmallChangeToModelAccumulated->Add(*pSmallChangeToModelOverwrite, pThreadBuffer)) {
         return true;
      }
   }

   // we need to divide by the number of sampling sets that we constructed this from.
   // We also need to slow down our growth so that the more relevant Attributes get a chance to grow first so we multiply by a user defined learning rate

#ifdef TREAT_BINARY_AS_MULTICLASS
   if(2 == GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates)) {
      // in the specific case of us simulating multiclass classification on a binary attribute, we want to divide the learning rate by 2 since doubling the classes effectively doubles the learning rate
      pSmallChangeToModelAccumulated->Multiply(learningRate / cSamplingSetsAfterZero / 2);
   } else {
      pSmallChangeToModelAccumulated->Multiply(learningRate / cSamplingSetsAfterZero);
   }
#else
   pSmallChangeToModelAccumulated->Multiply(learningRate / cSamplingSetsAfterZero);
#endif

   size_t acDivisionIntegersEnd[k_cDimensionsMax];
   for(size_t iDimension = 0; iDimension < cDimensions; ++iDimension) {
      acDivisionIntegersEnd[iDimension] = pAttributeCombination->m_AttributeCombinationEntry[iDimension].m_pAttribute->m_cStates;
   }
   if(pSmallChangeToModelAccumulated->Expand(acDivisionIntegersEnd)) {
      return true;
   }

   SegmentedRegionCore<ActiveDataType, FractionalDataType> * const pSegmentedRegion = apCurrentModel[iCurrentModel % cAttributeCombinations];
   void * pThreadBuffer = pCachedThreadResources->GetThreadByteBuffer1(pSegmentedRegion->GetStackMemorySizeBytes());
   if(UNLIKELY(nullptr == pThreadBuffer)) {
      return true;
   }
   if(pSegmentedRegion->Add(*pSmallChangeToModelAccumulated, pThreadBuffer)) {
      return true;
   }

   // TODO : move the target bits branch inside TrainingSetInputAttributeLoop to here outside instead of the attribute combination.  The target # of bits is extremely predictable and so we get to only process one sub branch of code below that.  If we do attribute combinations here then we have to keep in instruction cache a whole bunch of options
   *pModelMetric = ValidationSetInputAttributeLoop<1, countCompilerClassificationTargetStates>(pAttributeCombination, pValidationSet, pSmallChangeToModelAccumulated, cTargetStates);

   return false;
}

// TODO: can I put the SamplingWithoutReplacement bit into the data if I separate my data out by sample?

// TODO: review everything from here on down in this file
union CachedThreadResourcesUnion {
   CachedTrainingThreadResources<true> regression;
   CachedTrainingThreadResources<false> classification;

   CachedThreadResourcesUnion(const bool bRegression, const size_t cVectorLength) {
      if(bRegression) {
         // member classes inside a union requre explicit call to constructor
         new(&regression) CachedTrainingThreadResources<true>(cVectorLength);
      } else {
         // member classes inside a union requre explicit call to constructor
         new(&classification) CachedTrainingThreadResources<false>(cVectorLength);
      }
   }

   ~CachedThreadResourcesUnion() {
      // we don't have enough information here to delete this object, so we do it from our caller
   }
};

// TODO: rename this TmlTrainingState
class TmlState {
public:
   const bool m_bRegression;
   const size_t m_cTargetStates;

   const size_t m_cAttributeCombinations;
   AttributeCombinationCore ** const m_apAttributeCombinations;

   DataSetAttributeCombination * m_pTrainingSet;
   DataSetAttributeCombination * m_pValidationSet;

   const size_t m_cSamplingSets;

   SamplingMethod ** m_apSamplingSets;
   SegmentedRegionCore<ActiveDataType, FractionalDataType> ** m_apCurrentModel;
   SegmentedRegionCore<ActiveDataType, FractionalDataType> ** m_apBestModel;

   FractionalDataType m_bestModelMetric;

   SegmentedRegionCore<ActiveDataType, FractionalDataType> * const m_pSmallChangeToModelOverwriteSingleSamplingSet;
   SegmentedRegionCore<ActiveDataType, FractionalDataType> * const m_pSmallChangeToModelAccumulatedFromSamplingSets;

   // TODO : right now we need to keep these arround and separate but we can eliminate them in the future... and we already know the number of attributes at startup since that's done outside our core module, so we can just allocate the correct number of them.  And combine them for both training and validation since they both use the same parameters.  For now we need to keep these arround so that our Attributes aren't deleted
   AttributeSetInternalCore * m_pAttributeSet;

   CachedThreadResourcesUnion m_cachedThreadResourcesUnion;

   TmlState(const bool bRegression, const size_t cTargetStates, const size_t cAttributeCombinations, const size_t cSamplingSets)
      : m_bRegression(bRegression)
      , m_cTargetStates(cTargetStates)
      , m_cAttributeCombinations(cAttributeCombinations)
      , m_apAttributeCombinations(new (std::nothrow) AttributeCombinationCore *[cAttributeCombinations])
      , m_pTrainingSet(nullptr)
      , m_pValidationSet(nullptr)
      , m_cSamplingSets(cSamplingSets)
      , m_apSamplingSets(nullptr)
      , m_apCurrentModel(nullptr)
      , m_apBestModel(nullptr)
      , m_bestModelMetric(std::numeric_limits<FractionalDataType>::infinity())
      , m_pSmallChangeToModelOverwriteSingleSamplingSet(SegmentedRegionCore<ActiveDataType, FractionalDataType>::Allocate(k_cDimensionsMax, GetVectorLengthFlatCore(cTargetStates)))
      , m_pSmallChangeToModelAccumulatedFromSamplingSets(SegmentedRegionCore<ActiveDataType, FractionalDataType>::Allocate(k_cDimensionsMax, GetVectorLengthFlatCore(cTargetStates)))
      , m_pAttributeSet(nullptr)
      , m_cachedThreadResourcesUnion(bRegression, GetVectorLengthFlatCore(cTargetStates)) {
      // we need to set this to zero otherwise our destructor will attempt to free garbage memory pointers if we prematurely call the destructor
      memset(m_apAttributeCombinations, 0, sizeof(*m_apAttributeCombinations) * cAttributeCombinations);
   }

   ~TmlState() {
      if(m_bRegression) {
         // member classes inside a union requre explicit call to destructor
         m_cachedThreadResourcesUnion.regression.~CachedTrainingThreadResources();
      } else {
         // member classes inside a union requre explicit call to destructor
         m_cachedThreadResourcesUnion.classification.~CachedTrainingThreadResources();
      }

      SamplingWithReplacement::FreeSamplingSets(m_cSamplingSets, m_apSamplingSets);

      delete m_pTrainingSet;
      delete m_pValidationSet;

      AttributeCombinationCore::FreeAttributeCombinations(m_cAttributeCombinations, m_apAttributeCombinations);

      delete m_pAttributeSet;

      DeleteSegmentsCore(m_cAttributeCombinations, m_apCurrentModel);
      DeleteSegmentsCore(m_cAttributeCombinations, m_apBestModel);
      SegmentedRegionCore<ActiveDataType, FractionalDataType>::Free(m_pSmallChangeToModelOverwriteSingleSamplingSet);
      SegmentedRegionCore<ActiveDataType, FractionalDataType>::Free(m_pSmallChangeToModelAccumulatedFromSamplingSets);
   }

   bool Initialize(const IntegerDataType randomSeed, const size_t cAttributes, const EbmAttribute * const aAttributes, const EbmAttributeCombination * const aAttributeCombinations, const IntegerDataType * attributeCombinationIndexes, const size_t cTargetStates, const size_t cTrainingCases, const void * const aTrainingTargets, const IntegerDataType * const aTrainingData, const FractionalDataType * const aTrainingPredictionScores, const size_t cValidationCases, const void * const aValidationTargets, const IntegerDataType * const aValidationData, const FractionalDataType * const aValidationPredictionScores) {
      try {
         if (m_bRegression) {
            if (m_cachedThreadResourcesUnion.regression.IsError()) {
               return true;
            }
         } else {
            if (m_cachedThreadResourcesUnion.classification.IsError()) {
               return true;
            }
         }

         assert(nullptr == m_pAttributeSet);
         m_pAttributeSet = new (std::nothrow) AttributeSetInternalCore();
         if (nullptr == m_pAttributeSet) {
            return true;
         }

         assert(!IsMultiplyError(cAttributes, sizeof(*aAttributes))); // if this overflows then our caller should not have been able to allocate the array
         const EbmAttribute * pAttributeInitialize = aAttributes;
         const EbmAttribute * const pAttributeEnd = &aAttributes[cAttributes];
         assert(pAttributeInitialize < pAttributeEnd);
         size_t iAttributeInitialize = 0;
         do {
            static_assert(AttributeTypeCore::OrdinalCore == static_cast<AttributeTypeCore>(AttributeTypeOrdinal), "AttributeTypeCore::OrdinalCore must have the same value as AttributeTypeOrdinal");
            static_assert(AttributeTypeCore::NominalCore == static_cast<AttributeTypeCore>(AttributeTypeNominal), "AttributeTypeCore::NominalCore must have the same value as AttributeTypeNominal");
            assert(AttributeTypeOrdinal == pAttributeInitialize->attributeType || AttributeTypeNominal == pAttributeInitialize->attributeType);
            AttributeTypeCore attributeTypeCore = static_cast<AttributeTypeCore>(pAttributeInitialize->attributeType);

            IntegerDataType countStates = pAttributeInitialize->countStates;
            assert(2 <= countStates);
            if (!IsNumberConvertable<size_t, IntegerDataType>(countStates)) {
               return true;
            }
            size_t cStates = static_cast<size_t>(countStates);

            assert(0 == pAttributeInitialize->hasMissing || 1 == pAttributeInitialize->hasMissing);
            bool bMissing = 0 != pAttributeInitialize->hasMissing;

            AttributeInternalCore * pAttribute = m_pAttributeSet->AddAttribute(cStates, iAttributeInitialize, attributeTypeCore, bMissing);
            if (nullptr == pAttribute) {
               return true;
            }

            assert(0 == pAttributeInitialize->hasMissing); // TODO : implement this, then remove this assert
            assert(AttributeTypeOrdinal == pAttributeInitialize->attributeType); // TODO : implement this, then remove this assert

            ++iAttributeInitialize;
            ++pAttributeInitialize;
         } while (pAttributeEnd != pAttributeInitialize);

         size_t cVectorLength = GetVectorLengthFlatCore(cTargetStates);

         if (UNLIKELY(nullptr == m_apAttributeCombinations)) {
            return true;
         }

         const IntegerDataType * pAttributeCombinationIndex = attributeCombinationIndexes;
         for (size_t iAttributeCombination = 0; iAttributeCombination < m_cAttributeCombinations; ++iAttributeCombination) {
            const EbmAttributeCombination * pAttributeCombinationInterop = &aAttributeCombinations[iAttributeCombination];

            IntegerDataType countAttributesInCombination = pAttributeCombinationInterop->countAttributesInCombination;
            assert(1 <= countAttributesInCombination);
            if (!IsNumberConvertable<size_t, IntegerDataType>(countAttributesInCombination)) {
               return true;
            }
            size_t cAttributesInCombination = static_cast<size_t>(countAttributesInCombination);
            if (k_cDimensionsMax < cAttributesInCombination) {
               // if we try to run with more than k_cDimensionsMax we'll exceed our memory capacity, so let's exit here instead
               return true;
            }

            AttributeCombinationCore * pAttributeCombination = AttributeCombinationCore::Allocate(cAttributesInCombination, iAttributeCombination);
            if (nullptr == pAttributeCombination) {
               return true;
            }

            size_t cTensorStates = 1;
            for (size_t iAttributeInCombination = 0; iAttributeInCombination < cAttributesInCombination; ++iAttributeInCombination) {
               const IntegerDataType indexAttributeInterop = *pAttributeCombinationIndex;
               assert(0 <= indexAttributeInterop);
               ++pAttributeCombinationIndex;

               if (!IsNumberConvertable<size_t, IntegerDataType>(indexAttributeInterop)) {
                  return true;
               }
               const size_t iAttributeForCombination = static_cast<size_t>(indexAttributeInterop);
               assert(iAttributeForCombination < cAttributes);
               AttributeInternalCore * const pInputAttribute = m_pAttributeSet->m_inputAttributes[iAttributeForCombination];
               pAttributeCombination->m_AttributeCombinationEntry[iAttributeInCombination].m_pAttribute = pInputAttribute;
               cTensorStates *= pInputAttribute->m_cStates;
            }
            size_t cBitsRequiredMin = CountBitsRequiredCore(cTensorStates);
            pAttributeCombination->m_cItemsPerBitPackDataUnit = GetCountItemsBitPacked(cBitsRequiredMin);

            m_apAttributeCombinations[iAttributeCombination] = pAttributeCombination;
         }

         if (UNLIKELY(nullptr == m_pSmallChangeToModelOverwriteSingleSamplingSet)) {
            return true;
         }

         if (UNLIKELY(nullptr == m_pSmallChangeToModelAccumulatedFromSamplingSets)) {
            return true;
         }

         m_pTrainingSet = new (std::nothrow) DataSetAttributeCombination(true, !m_bRegression, !m_bRegression, m_cAttributeCombinations, m_apAttributeCombinations, cTrainingCases, aTrainingData, aTrainingTargets, aTrainingPredictionScores, cVectorLength);
         if (nullptr == m_pTrainingSet || m_pTrainingSet->IsError()) {
            return true;
         }

         m_pValidationSet = new (std::nothrow) DataSetAttributeCombination(m_bRegression, !m_bRegression, !m_bRegression, m_cAttributeCombinations, m_apAttributeCombinations, cValidationCases, aValidationData, aValidationTargets, aValidationPredictionScores, cVectorLength);
         if (nullptr == m_pValidationSet || m_pValidationSet->IsError()) {
            return true;
         }


         RandomStream randomStream(randomSeed);

         assert(nullptr == m_apSamplingSets);
         m_apSamplingSets = SamplingWithReplacement::GenerateSamplingSets(&randomStream, m_pTrainingSet, m_cSamplingSets);
         if (UNLIKELY(nullptr == m_apSamplingSets)) {
            return true;
         }

         assert(nullptr == m_apCurrentModel);
         m_apCurrentModel = InitializeSegmentsCore(m_cAttributeCombinations, m_apAttributeCombinations, cVectorLength);
         if (nullptr == m_apCurrentModel) {
            return true;
         }
         assert(nullptr == m_apBestModel);
         m_apBestModel = InitializeSegmentsCore(m_cAttributeCombinations, m_apAttributeCombinations, cVectorLength);
         if (nullptr == m_apBestModel) {
            return true;
         }

         if (m_bRegression) {
            InitializeErrorCore<k_Regression>(cTrainingCases, aTrainingTargets, aTrainingPredictionScores, m_pTrainingSet->GetResidualPointer(), 0, k_iZeroResidual);
            InitializeErrorCore<k_Regression>(cValidationCases, aValidationTargets, aValidationPredictionScores, m_pValidationSet->GetResidualPointer(), 0, k_iZeroResidual);
         } else {
            if (2 == cTargetStates) {
               InitializeErrorCore<2>(cTrainingCases, m_pTrainingSet->GetTargetDataPointer(), aTrainingPredictionScores, m_pTrainingSet->GetResidualPointer(), cTargetStates, k_iZeroResidual);
            } else {
               InitializeErrorCore<k_DynamicClassification>(cTrainingCases, m_pTrainingSet->GetTargetDataPointer(), aTrainingPredictionScores, m_pTrainingSet->GetResidualPointer(), cTargetStates, k_iZeroResidual);
            }
         }
         return false;
      } catch (...) {
         // this is here to catch exceptions from RandomStream randomStream(randomSeed), but it could also catch errors if we put any other C++ types in here later
         return true;
      }
   }
};

#ifndef NDEBUG
void CheckTargets(const size_t cTargetStates, const size_t cCases, const void * const aTargets) {
   if(0 == cTargetStates) {
      // regression!

      const FractionalDataType * pTarget = static_cast<const FractionalDataType *>(aTargets);
      const FractionalDataType * const pTargetEnd = pTarget + cCases;
      do {
         const FractionalDataType data = *pTarget;
         assert(!std::isnan(data));
         assert(!std::isinf(data));
         ++pTarget;
      } while(pTargetEnd != pTarget);
   } else {
      // classification

      const IntegerDataType * pTarget = static_cast<const IntegerDataType *>(aTargets);
      const IntegerDataType * const pTargetEnd = pTarget + cCases;
      do {
         const IntegerDataType data = *pTarget;
         assert(0 <= data);
         assert((IsNumberConvertable<size_t, IntegerDataType>(data))); // data must be lower than cTargetStates and cTargetStates fits into a size_t which we checked earlier
         assert(static_cast<size_t>(data) < cTargetStates);
         ++pTarget;
      } while(pTargetEnd != pTarget);
   }
}
#endif

// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
TmlState * AllocateCore(bool bRegression, IntegerDataType randomSeed, IntegerDataType countAttributes, const EbmAttribute * attributes, IntegerDataType countAttributeCombinations, const EbmAttributeCombination * attributeCombinations, const IntegerDataType * attributeCombinationIndexes, IntegerDataType countTargetStates, IntegerDataType countTrainingCases, const void * trainingTargets, const IntegerDataType * trainingData, const FractionalDataType * trainingPredictionScores, IntegerDataType countValidationCases, const void * validationTargets, const IntegerDataType * validationData, const FractionalDataType * validationPredictionScores, IntegerDataType countInnerBags) {
   // bRegression is set in our program, so our caller can't pass in invalid data
   // randomSeed can be any value
   assert(1 <= countAttributes);
   assert(nullptr != attributes);
   assert(1 <= countAttributeCombinations);
   assert(nullptr != attributeCombinations);
   assert(nullptr != attributeCombinationIndexes);
   assert(bRegression || 2 <= countTargetStates);
   assert(1 <= countTrainingCases);
   assert(nullptr != trainingTargets);
   assert(nullptr != trainingData);
   // trainingPredictionScores can be null
   assert(1 <= countValidationCases);
   assert(nullptr != validationTargets);
   assert(nullptr != validationData);
   // validationPredictionScores can be null
   assert(0 <= countInnerBags); // 0 means use the full set (good value).  1 means make a single bag (this is useless but allowed for comparison purposes).  2+ are good numbers of bag

   if (!IsNumberConvertable<size_t, IntegerDataType>(countAttributes)) {
      return nullptr;
   }
   if (!IsNumberConvertable<size_t, IntegerDataType>(countAttributeCombinations)) {
      return nullptr;
   }
   if (!IsNumberConvertable<size_t, IntegerDataType>(countTargetStates)) {
      return nullptr;
   }
   if (!IsNumberConvertable<size_t, IntegerDataType>(countTrainingCases)) {
      return nullptr;
   }
   if (!IsNumberConvertable<size_t, IntegerDataType>(countValidationCases)) {
      return nullptr;
   }
   if (!IsNumberConvertable<size_t, IntegerDataType>(countInnerBags)) {
      return nullptr;
   }

   size_t cAttributes = static_cast<size_t>(countAttributes);
   size_t cAttributeCombinations = static_cast<size_t>(countAttributeCombinations);
   size_t cTargetStates = static_cast<size_t>(countTargetStates);
   size_t cTrainingCases = static_cast<size_t>(countTrainingCases);
   size_t cValidationCases = static_cast<size_t>(countValidationCases);
   size_t cInnerBags = static_cast<size_t>(countInnerBags);

   size_t cVectorLength = GetVectorLengthFlatCore(cTargetStates);

   if (IsMultiplyError(cVectorLength, cTrainingCases)) {
      return nullptr;
   }
   if (IsMultiplyError(cVectorLength, cValidationCases)) {
      return nullptr;
   }

#ifndef NDEBUG
   CheckTargets(cTargetStates, cTrainingCases, trainingTargets);
   CheckTargets(cTargetStates, cValidationCases, validationTargets);
#endif

   TmlState * const pTmlState = new (std::nothrow) TmlState(bRegression, cTargetStates, cAttributeCombinations, cInnerBags);
   if (UNLIKELY(nullptr == pTmlState)) {
      return nullptr;
   }
   if (UNLIKELY(pTmlState->Initialize(randomSeed, cAttributes, attributes, attributeCombinations, attributeCombinationIndexes, cTargetStates, cTrainingCases, trainingTargets, trainingData, trainingPredictionScores, cValidationCases, validationTargets, validationData, validationPredictionScores))) {
      delete pTmlState;
      return nullptr;
   }
   return pTmlState;
}

EBMCORE_IMPORT_EXPORT PEbmTraining EBMCORE_CALLING_CONVENTION InitializeTrainingRegression(IntegerDataType randomSeed, IntegerDataType countAttributes, const EbmAttribute * attributes, IntegerDataType countAttributeCombinations, const EbmAttributeCombination * attributeCombinations, const IntegerDataType * attributeCombinationIndexes, IntegerDataType countTrainingCases, const FractionalDataType * trainingTargets, const IntegerDataType * trainingData, const FractionalDataType * trainingPredictionScores, IntegerDataType countValidationCases, const FractionalDataType * validationTargets, const IntegerDataType * validationData, const FractionalDataType * validationPredictionScores, IntegerDataType countInnerBags) {
   LOG(TraceLevelInfo, "Entered InitializeTrainingRegression");
   PEbmTraining pEbmTraining = reinterpret_cast<PEbmTraining>(AllocateCore(true, randomSeed, countAttributes, attributes, countAttributeCombinations, attributeCombinations, attributeCombinationIndexes, 0, countTrainingCases, trainingTargets, trainingData, trainingPredictionScores, countValidationCases, validationTargets, validationData, validationPredictionScores, countInnerBags));
   LOG(TraceLevelInfo, "Exited InitializeTrainingRegression");
   return pEbmTraining;
}

EBMCORE_IMPORT_EXPORT PEbmTraining EBMCORE_CALLING_CONVENTION InitializeTrainingClassification(IntegerDataType randomSeed, IntegerDataType countAttributes, const EbmAttribute * attributes, IntegerDataType countAttributeCombinations, const EbmAttributeCombination * attributeCombinations, const IntegerDataType * attributeCombinationIndexes, IntegerDataType countTargetStates, IntegerDataType countTrainingCases, const IntegerDataType * trainingTargets, const IntegerDataType * trainingData, const FractionalDataType * trainingPredictionScores, IntegerDataType countValidationCases, const IntegerDataType * validationTargets, const IntegerDataType * validationData, const FractionalDataType * validationPredictionScores, IntegerDataType countInnerBags) {
   LOG(TraceLevelInfo, "Entered InitializeTrainingClassification");
   PEbmTraining pEbmTraining = reinterpret_cast<PEbmTraining>(AllocateCore(false, randomSeed, countAttributes, attributes, countAttributeCombinations, attributeCombinations, attributeCombinationIndexes, countTargetStates, countTrainingCases, trainingTargets, trainingData, trainingPredictionScores, countValidationCases, validationTargets, validationData, validationPredictionScores, countInnerBags));
   LOG(TraceLevelInfo, "Exited InitializeTrainingClassification");
   return pEbmTraining;
}

template<bool bRegression>
TML_INLINE CachedTrainingThreadResources<bRegression> * GetCachedThreadResources(TmlState * pTmlState);
template<>
TML_INLINE CachedTrainingThreadResources<false> * GetCachedThreadResources<false>(TmlState * pTmlState) {
   return &pTmlState->m_cachedThreadResourcesUnion.classification;
}
template<>
TML_INLINE CachedTrainingThreadResources<true> * GetCachedThreadResources<true>(TmlState * pTmlState) {
   return &pTmlState->m_cachedThreadResourcesUnion.regression;
}

template<ptrdiff_t countCompilerClassificationTargetStates>
static IntegerDataType TrainingStepPerTargetStates(TmlState * const pTmlState, const size_t iAttributeCombination, const FractionalDataType learningRate, const size_t cTreeSplitsMax, const size_t cCasesRequiredForSplitParentMin, const FractionalDataType * const aTrainingWeights, const FractionalDataType * const aValidationWeights, FractionalDataType * const pValidationMetricReturn) {
   const size_t cSamplingSetsAfterZero = 0 == pTmlState->m_cSamplingSets ? 1 : pTmlState->m_cSamplingSets;

   const AttributeCombinationCore * const pAttributeCombination = pTmlState->m_apAttributeCombinations[iAttributeCombination];

   FractionalDataType modelMetric;
   // TODO : can I extract the stuff below into a function that can be templated?
   if(GenerateModelLoop<countCompilerClassificationTargetStates>(pTmlState->m_pSmallChangeToModelAccumulatedFromSamplingSets, GetCachedThreadResources<IsRegression(countCompilerClassificationTargetStates)>(pTmlState), pTmlState->m_apSamplingSets, pAttributeCombination, cTreeSplitsMax, cCasesRequiredForSplitParentMin, pTmlState->m_apCurrentModel, pTmlState->m_pSmallChangeToModelOverwriteSingleSamplingSet, pTmlState->m_cAttributeCombinations, cSamplingSetsAfterZero, iAttributeCombination, learningRate, pTmlState->m_pValidationSet, pTmlState->m_cTargetStates, &modelMetric)) {
      return 1;
   }

   // modelMetric is either logloss (classification) or rmse (regression).  In either case we want to minimize it.
   if(LIKELY(modelMetric < pTmlState->m_bestModelMetric)) {
      // we keep on improving, so this is more likely than not, and we'll exit if it becomes negative a lot
      pTmlState->m_bestModelMetric = modelMetric;

      // TODO : in the future don't copy over all SegmentedRegions.  We only need to copy the ones that changed, which we can detect if we use a linked list and array lookup for the same data structure
      size_t iModel = 0;
      size_t iModelEnd = pTmlState->m_cAttributeCombinations;
      do {
         //const AttributeCombinationCore * const pAttributeCombinationUpdate = pTmlState->m_apAttributeCombinations[iModel];
         if(pTmlState->m_apBestModel[iModel]->Copy(*pTmlState->m_apCurrentModel[iModel])) {
            return 1;
         }
         ++iModel;
      } while(iModel != iModelEnd);
   }

   //pTmlState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Expand();

   // TODO : move the target bits branch inside TrainingSetInputAttributeLoop to here outside instead of the attribute combination.  The target # of bits is extremely predictable and so we get to only process one sub branch of code below that.  If we do attribute combinations here then we have to keep in instruction cache a whole bunch of options
   TrainingSetInputAttributeLoop<1, countCompilerClassificationTargetStates>(pAttributeCombination, pTmlState->m_pTrainingSet, pTmlState->m_pSmallChangeToModelAccumulatedFromSamplingSets, pTmlState->m_cTargetStates, k_iZeroResidual);

   *pValidationMetricReturn = modelMetric;
   return 0;
}

template<ptrdiff_t iPossibleCompilerOptimizedTargetStates>
TML_INLINE IntegerDataType CompilerRecursiveTrainingStep(const size_t cRuntimeTargetStates, TmlState * const pTmlState, const size_t iAttributeCombination, const FractionalDataType learningRate, const size_t cTreeSplitsMax, const size_t cCasesRequiredForSplitParentMin, const FractionalDataType * const aTrainingWeights, const FractionalDataType * const aValidationWeights, FractionalDataType * const pValidationMetricReturn) {
   assert(IsClassification(iPossibleCompilerOptimizedTargetStates));
   if(iPossibleCompilerOptimizedTargetStates == cRuntimeTargetStates) {
      assert(cRuntimeTargetStates <= k_cCompilerOptimizedTargetStatesMax);
      return TrainingStepPerTargetStates<iPossibleCompilerOptimizedTargetStates>(pTmlState, iAttributeCombination, learningRate, cTreeSplitsMax, cCasesRequiredForSplitParentMin, aTrainingWeights, aValidationWeights, pValidationMetricReturn);
   } else {
      return CompilerRecursiveTrainingStep<iPossibleCompilerOptimizedTargetStates + 1>(cRuntimeTargetStates, pTmlState, iAttributeCombination, learningRate, cTreeSplitsMax, cCasesRequiredForSplitParentMin, aTrainingWeights, aValidationWeights, pValidationMetricReturn);
   }
}
template<>
TML_INLINE IntegerDataType CompilerRecursiveTrainingStep<k_cCompilerOptimizedTargetStatesMax + 1>(const size_t cRuntimeTargetStates, TmlState * const pTmlState, const size_t iAttributeCombination, const FractionalDataType learningRate, const size_t cTreeSplitsMax, const size_t cCasesRequiredForSplitParentMin, const FractionalDataType * const aTrainingWeights, const FractionalDataType * const aValidationWeights, FractionalDataType * const pValidationMetricReturn) {
   // it is logically possible, but uninteresting to have a classification with 1 target state, so let our runtime system handle those unlikley and uninteresting cases
   assert(k_cCompilerOptimizedTargetStatesMax < cRuntimeTargetStates || 1 == cRuntimeTargetStates);
   return TrainingStepPerTargetStates<k_DynamicClassification>(pTmlState, iAttributeCombination, learningRate, cTreeSplitsMax, cCasesRequiredForSplitParentMin, aTrainingWeights, aValidationWeights, pValidationMetricReturn);
}

EBMCORE_IMPORT_EXPORT IntegerDataType EBMCORE_CALLING_CONVENTION TrainingStep(PEbmTraining ebmTraining, IntegerDataType indexAttributeCombination, FractionalDataType learningRate, IntegerDataType countTreeSplitsMax, IntegerDataType countCasesRequiredForSplitParentMin, const FractionalDataType* trainingWeights, const FractionalDataType* validationWeights, FractionalDataType* validationMetricReturn) {
   TmlState * pTmlState = reinterpret_cast<TmlState *>(ebmTraining);
   assert(nullptr != pTmlState);
   assert(0 <= indexAttributeCombination);
   assert((IsNumberConvertable<size_t, IntegerDataType>(indexAttributeCombination))); // we wouldn't have allowed the creation of an attribute set larger than size_t
   size_t iAttributeCombination = static_cast<size_t>(indexAttributeCombination);
   assert(iAttributeCombination < pTmlState->m_cAttributeCombinations);

   assert(!std::isnan(learningRate));
   assert(!std::isinf(learningRate));
   assert(0 < learningRate);

   assert(1 <= countTreeSplitsMax);
   size_t cTreeSplitsMax = static_cast<size_t>(countTreeSplitsMax);
   if(!IsNumberConvertable<size_t, IntegerDataType>(countTreeSplitsMax)) {
      // we can never exceed a size_t number of splits, so let's just set it to the maximum if we were going to overflow because it will generate the same results as if we used the true number
      cTreeSplitsMax = std::numeric_limits<size_t>::max();
   }

   assert(2 <= countCasesRequiredForSplitParentMin);
   size_t cCasesRequiredForSplitParentMin = static_cast<size_t>(countCasesRequiredForSplitParentMin);
   if(!IsNumberConvertable<size_t, IntegerDataType>(countCasesRequiredForSplitParentMin)) {
      // we can never exceed a size_t number of cases, so let's just set it to the maximum if we were going to overflow because it will generate the same results as if we used the true number
      cCasesRequiredForSplitParentMin = std::numeric_limits<size_t>::max();
   }

   assert(nullptr == trainingWeights); // TODO : implement this later
   assert(nullptr == validationWeights); // TODO : implement this later
   assert(nullptr != validationMetricReturn);

   if(pTmlState->m_bRegression) {
      return TrainingStepPerTargetStates<k_Regression>(pTmlState, iAttributeCombination, learningRate, cTreeSplitsMax, cCasesRequiredForSplitParentMin, trainingWeights, validationWeights, validationMetricReturn);
   } else {
      const size_t cTargetStates = pTmlState->m_cTargetStates;
      return CompilerRecursiveTrainingStep<2>(cTargetStates, pTmlState, iAttributeCombination, learningRate, cTreeSplitsMax, cCasesRequiredForSplitParentMin, trainingWeights, validationWeights, validationMetricReturn);
   }
}

EBMCORE_IMPORT_EXPORT FractionalDataType * EBMCORE_CALLING_CONVENTION GetCurrentModel(PEbmTraining ebmTraining, IntegerDataType indexAttributeCombination) {
   TmlState * pTmlState = reinterpret_cast<TmlState *>(ebmTraining);
   assert(nullptr != pTmlState);
   assert(0 <= indexAttributeCombination);
   assert((IsNumberConvertable<size_t, IntegerDataType>(indexAttributeCombination))); // we wouldn't have allowed the creation of an attribute set larger than size_t
   size_t iAttributeCombination = static_cast<size_t>(indexAttributeCombination);
   assert(iAttributeCombination < pTmlState->m_cAttributeCombinations);

   SegmentedRegionCore<ActiveDataType, FractionalDataType> * pCurrentModel = pTmlState->m_apCurrentModel[iAttributeCombination];
   assert(pCurrentModel->m_bExpanded); // the model should have been expanded at startup
   return pCurrentModel->GetValuePointer();
}

EBMCORE_IMPORT_EXPORT FractionalDataType * EBMCORE_CALLING_CONVENTION GetBestModel(PEbmTraining ebmTraining, IntegerDataType indexAttributeCombination) {
   TmlState * pTmlState = reinterpret_cast<TmlState *>(ebmTraining);
   assert(nullptr != pTmlState);
   assert(0 <= indexAttributeCombination);
   assert((IsNumberConvertable<size_t, IntegerDataType>(indexAttributeCombination))); // we wouldn't have allowed the creation of an attribute set larger than size_t
   size_t iAttributeCombination = static_cast<size_t>(indexAttributeCombination);
   assert(iAttributeCombination < pTmlState->m_cAttributeCombinations);

   SegmentedRegionCore<ActiveDataType, FractionalDataType> * pBestModel = pTmlState->m_apBestModel[iAttributeCombination];
   assert(pBestModel->m_bExpanded); // the model should have been expanded at startup
   return pBestModel->GetValuePointer();
}

EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION CancelTraining(PEbmTraining ebmTraining) {
   TmlState * pTmlState = reinterpret_cast<TmlState *>(ebmTraining);
   assert(nullptr != pTmlState);
}

EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION FreeTraining(PEbmTraining ebmTraining) {
   TmlState * pTmlState = reinterpret_cast<TmlState *>(ebmTraining);
   assert(nullptr != pTmlState);
   delete pTmlState;
}

