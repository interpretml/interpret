// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <string.h> // memset
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebmcore.h"

#include "Logging.h"
#include "EbmInternal.h"
// very independent includes
#include "Logging.h" // EBM_ASSERT & LOG
#include "InitializeResiduals.h"
#include "RandomStream.h"
#include "SegmentedTensor.h"
#include "EbmStatistics.h"
// this depends on TreeNode pointers, but doesn't require the full definition of TreeNode
#include "CachedThreadResources.h"
// feature includes
#include "Feature.h"
// FeatureCombination.h depends on FeatureInternal.h
#include "FeatureCombination.h"
// dataset depends on features
#include "DataSetByFeatureCombination.h"
// samples is somewhat independent from datasets, but relies on an indirect coupling with them
#include "SamplingWithReplacement.h"
// TreeNode depends on almost everything
#include "DimensionSingle.h"
#include "DimensionMultiple.h"

static void DeleteSegmentsCore(const size_t cFeatureCombinations, SegmentedRegionCore<ActiveDataType, FractionalDataType> ** const apSegmentedRegions) {
   LOG(TraceLevelInfo, "Entered DeleteSegmentsCore");

   if(UNLIKELY(nullptr != apSegmentedRegions)) {
      EBM_ASSERT(0 < cFeatureCombinations);
      SegmentedRegionCore<ActiveDataType, FractionalDataType> ** ppSegmentedRegions = apSegmentedRegions;
      const SegmentedRegionCore<ActiveDataType, FractionalDataType> * const * const ppSegmentedRegionsEnd = &apSegmentedRegions[cFeatureCombinations];
      do {
         SegmentedRegionCore<ActiveDataType, FractionalDataType>::Free(*ppSegmentedRegions);
         ++ppSegmentedRegions;
      } while(ppSegmentedRegionsEnd != ppSegmentedRegions);
      delete[] apSegmentedRegions;
   }
   LOG(TraceLevelInfo, "Exited DeleteSegmentsCore");
}

static SegmentedRegionCore<ActiveDataType, FractionalDataType> ** InitializeSegmentsCore(const size_t cFeatureCombinations, const FeatureCombinationCore * const * const apFeatureCombinations, const size_t cVectorLength) {
   LOG(TraceLevelInfo, "Entered InitializeSegmentsCore");

   EBM_ASSERT(0 < cFeatureCombinations);
   EBM_ASSERT(nullptr != apFeatureCombinations);
   EBM_ASSERT(1 <= cVectorLength);

   SegmentedRegionCore<ActiveDataType, FractionalDataType> ** const apSegmentedRegions = new (std::nothrow) SegmentedRegionCore<ActiveDataType, FractionalDataType> *[cFeatureCombinations];
   if(UNLIKELY(nullptr == apSegmentedRegions)) {
      LOG(TraceLevelWarning, "WARNING InitializeSegmentsCore nullptr == apSegmentedRegions");
      return nullptr;
   }
   memset(apSegmentedRegions, 0, sizeof(*apSegmentedRegions) * cFeatureCombinations); // this needs to be done immediately after allocation otherwise we might attempt to free random garbage on an error

   SegmentedRegionCore<ActiveDataType, FractionalDataType> ** ppSegmentedRegions = apSegmentedRegions;
   for(size_t iFeatureCombination = 0; iFeatureCombination < cFeatureCombinations; ++iFeatureCombination) {
      const FeatureCombinationCore * const pFeatureCombination = apFeatureCombinations[iFeatureCombination];
      SegmentedRegionCore<ActiveDataType, FractionalDataType> * const pSegmentedRegions = SegmentedRegionCore<ActiveDataType, FractionalDataType>::Allocate(pFeatureCombination->m_cFeatures, cVectorLength);
      if(UNLIKELY(nullptr == pSegmentedRegions)) {
         LOG(TraceLevelWarning, "WARNING InitializeSegmentsCore nullptr == pSegmentedRegions");
         DeleteSegmentsCore(cFeatureCombinations, apSegmentedRegions);
         return nullptr;
      }

      if(0 == pFeatureCombination->m_cFeatures) {
         // if there are zero dimensions, then we have a tensor with 1 item, and we're already expanded
         pSegmentedRegions->m_bExpanded = true;
      } else {
         // if our segmented region has no dimensions, then it's already a fully expanded with 1 bin

         // TODO optimize the next few lines
         // TODO there might be a nicer way to expand this at allocation time (fill with zeros is easier)
         // we want to return a pointer to our interior state in the GetCurrentModelFeatureCombination and GetBestModelFeatureCombination functions.  For simplicity we don't transmit the divions, so we need to expand our SegmentedRegion before returning
         // the easiest way to ensure that the SegmentedRegion is expanded is to start it off expanded, and then we don't have to check later since anything merged into an expanded SegmentedRegion will itself be expanded
         size_t acDivisionIntegersEnd[k_cDimensionsMax];
         size_t iDimension = 0;
         do {
            acDivisionIntegersEnd[iDimension] = pFeatureCombination->m_FeatureCombinationEntry[iDimension].m_pFeature->m_cStates;
            ++iDimension;
         } while(iDimension < pFeatureCombination->m_cFeatures);

         if(pSegmentedRegions->Expand(acDivisionIntegersEnd)) {
            LOG(TraceLevelWarning, "WARNING InitializeSegmentsCore pSegmentedRegions->Expand(acDivisionIntegersEnd)");
            DeleteSegmentsCore(cFeatureCombinations, apSegmentedRegions);
            return nullptr;
         }
      }

      *ppSegmentedRegions = pSegmentedRegions;
      ++ppSegmentedRegions;
   }

   LOG(TraceLevelInfo, "Exited InitializeSegmentsCore");
   return apSegmentedRegions;
}

// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
template<unsigned int cInputBits, unsigned int cTargetBits, ptrdiff_t countCompilerClassificationTargetStates>
static void TrainingSetTargetFeatureLoop(const FeatureCombinationCore * const pFeatureCombination, DataSetFeatureCombination * const pTrainingSet, const FractionalDataType * const aModelUpdateTensor, const size_t cTargetStates) {
   LOG(TraceLevelVerbose, "Entered TrainingSetTargetFeatureLoop");

   const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
   const size_t cCases = pTrainingSet->GetCountCases();
   EBM_ASSERT(0 < cCases);

   if(0 == pFeatureCombination->m_cFeatures) {
      FractionalDataType * pResidualError = pTrainingSet->GetResidualPointer();
      const FractionalDataType * const pResidualErrorEnd = pResidualError + cVectorLength * cCases;
      if(IsRegression(countCompilerClassificationTargetStates)) {
         const FractionalDataType smallChangeToPrediction = aModelUpdateTensor[0];
         while(pResidualErrorEnd != pResidualError) {
            // this will apply a small fix to our existing TrainingPredictionScores, either positive or negative, whichever is needed
            const FractionalDataType residualError = EbmStatistics::ComputeRegressionResidualError(*pResidualError - smallChangeToPrediction);
            *pResidualError = residualError;
            ++pResidualError;
         }
      } else {
         EBM_ASSERT(IsClassification(countCompilerClassificationTargetStates));
         FractionalDataType * pTrainingPredictionScores = pTrainingSet->GetPredictionScores();
         const StorageDataTypeCore * pTargetData = pTrainingSet->GetTargetDataPointer();
         if(IsBinaryClassification(countCompilerClassificationTargetStates)) {
            const FractionalDataType smallChangeToPredictionScores = aModelUpdateTensor[0];
            while(pResidualErrorEnd != pResidualError) {
               StorageDataTypeCore targetData = *pTargetData;
               // TODO : because there is only one bin for a zero feature feature combination, we can move the fetch of smallChangeToPredictionScores outside of our loop so that the code doesn't have this dereference each loop
               // this will apply a small fix to our existing TrainingPredictionScores, either positive or negative, whichever is needed
               const FractionalDataType trainingPredictionScore = *pTrainingPredictionScores + smallChangeToPredictionScores;
               *pTrainingPredictionScores = trainingPredictionScore;
               const FractionalDataType residualError = EbmStatistics::ComputeClassificationResidualErrorBinaryclass(trainingPredictionScore, targetData);
               *pResidualError = residualError;
               ++pResidualError;
               ++pTrainingPredictionScores;
               ++pTargetData;
            }
         } else {
            const FractionalDataType * pValues = aModelUpdateTensor;
            while(pResidualErrorEnd != pResidualError) {
               StorageDataTypeCore targetData = *pTargetData;
               FractionalDataType sumExp = 0;
               size_t iVector1 = 0;
               do {
                  // TODO : because there is only one bin for a zero feature feature combination, we could move these values to the stack where the copmiler could reason about their visibility and optimize small arrays into registers
                  const FractionalDataType smallChangeToPredictionScores = pValues[iVector1];
                  // this will apply a small fix to our existing TrainingPredictionScores, either positive or negative, whichever is needed
                  const FractionalDataType trainingPredictionScores = pTrainingPredictionScores[iVector1] + smallChangeToPredictionScores;
                  pTrainingPredictionScores[iVector1] = trainingPredictionScores;
                  sumExp += std::exp(trainingPredictionScores);
                  ++iVector1;
               } while(iVector1 < cVectorLength);

               EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, size_t>(cVectorLength)));
               const StorageDataTypeCore cVectorLengthStorage = static_cast<StorageDataTypeCore>(cVectorLength);
               StorageDataTypeCore iVector2 = 0;
               do {
                  // TODO : we're calculating exp(predictionScore) above, and then again in ComputeClassificationResidualErrorMulticlass.  exp(..) is expensive so we should just do it once instead and store the result in a small memory array here
                  const FractionalDataType residualError = EbmStatistics::ComputeClassificationResidualErrorMulticlass(sumExp, pTrainingPredictionScores[iVector2], targetData, iVector2);
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
               constexpr bool bZeroingResiduals = 0 <= k_iZeroResidual;
               if(bZeroingResiduals) {
                  pResidualError[k_iZeroResidual - static_cast<ptrdiff_t>(cVectorLength)] = 0;
               }
               pTrainingPredictionScores += cVectorLength;
               ++pTargetData;
            }
         }
      }
      LOG(TraceLevelVerbose, "Exited TrainingSetTargetFeatureLoop - Zero dimensions");
      return;
   }

   const size_t cItemsPerBitPackDataUnit = pFeatureCombination->m_cItemsPerBitPackDataUnit;
   const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackDataUnit);
   const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);

   const StorageDataTypeCore * pInputData = pTrainingSet->GetDataPointer(pFeatureCombination);
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
            const FractionalDataType smallChangeToPrediction = aModelUpdateTensor[iBin * cVectorLength];
            // this will apply a small fix to our existing TrainingPredictionScores, either positive or negative, whichever is needed
            const FractionalDataType residualError = EbmStatistics::ComputeRegressionResidualError(*pResidualError - smallChangeToPrediction);
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
         EBM_ASSERT(0 == (pResidualErrorEnd - pResidualError) % cVectorLength);
         cItemsRemaining = (pResidualErrorEnd - pResidualError) / cVectorLength;
         EBM_ASSERT(0 < cItemsRemaining);
         EBM_ASSERT(cItemsRemaining <= cItemsPerBitPackDataUnit);
         goto one_last_loop_regression;
      }
      EBM_ASSERT(pResidualError == pResidualErrorEnd); // after our second iteration we should have finished everything!
   } else {
      EBM_ASSERT(IsClassification(countCompilerClassificationTargetStates));
      FractionalDataType * pTrainingPredictionScores = pTrainingSet->GetPredictionScores();
      const StorageDataTypeCore * pTargetData = pTrainingSet->GetTargetDataPointer();

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
            const FractionalDataType * pValues = &aModelUpdateTensor[iBin * cVectorLength];

            if(IsBinaryClassification(countCompilerClassificationTargetStates)) {
               const FractionalDataType smallChangeToPredictionScores = pValues[0];
               // this will apply a small fix to our existing TrainingPredictionScores, either positive or negative, whichever is needed
               const FractionalDataType trainingPredictionScore = *pTrainingPredictionScores + smallChangeToPredictionScores;
               *pTrainingPredictionScores = trainingPredictionScore;
               const FractionalDataType residualError = EbmStatistics::ComputeClassificationResidualErrorBinaryclass(trainingPredictionScore, targetData);
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

               EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, size_t>(cVectorLength)));
               const StorageDataTypeCore cVectorLengthStorage = static_cast<StorageDataTypeCore>(cVectorLength);
               StorageDataTypeCore iVector2 = 0;
               do {
                  // TODO : we're calculating exp(predictionScore) above, and then again in ComputeClassificationResidualErrorMulticlass.  exp(..) is expensive so we should just do it once instead and store the result in a small memory array here
                  const FractionalDataType residualError = EbmStatistics::ComputeClassificationResidualErrorMulticlass(sumExp, pTrainingPredictionScores[iVector2], targetData, iVector2);
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
               constexpr bool bZeroingResiduals = 0 <= k_iZeroResidual;
               if(bZeroingResiduals) {
                  pResidualError[k_iZeroResidual - static_cast<ptrdiff_t>(cVectorLength)] = 0;
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
         EBM_ASSERT(0 == (pResidualErrorEnd - pResidualError) % cVectorLength);
         cItemsRemaining = (pResidualErrorEnd - pResidualError) / cVectorLength;
         EBM_ASSERT(0 < cItemsRemaining);
         EBM_ASSERT(cItemsRemaining <= cItemsPerBitPackDataUnit);
         goto one_last_loop_classification;
      }
      EBM_ASSERT(pResidualError == pResidualErrorEnd); // after our second iteration we should have finished everything!
   }
   LOG(TraceLevelVerbose, "Exited TrainingSetTargetFeatureLoop");
}

// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
template<unsigned int cInputBits, ptrdiff_t countCompilerClassificationTargetStates>
static void TrainingSetInputFeatureLoop(const FeatureCombinationCore * const pFeatureCombination, DataSetFeatureCombination * const pTrainingSet, const FractionalDataType * const aModelUpdateTensor, const size_t cTargetStates) {
   if(cTargetStates <= 1 << 1) {
      TrainingSetTargetFeatureLoop<cInputBits, 1, countCompilerClassificationTargetStates>(pFeatureCombination, pTrainingSet, aModelUpdateTensor, cTargetStates);
   } else if(cTargetStates <= 1 << 2) {
      TrainingSetTargetFeatureLoop<cInputBits, 2, countCompilerClassificationTargetStates>(pFeatureCombination, pTrainingSet, aModelUpdateTensor, cTargetStates);
   } else if(cTargetStates <= 1 << 4) {
      TrainingSetTargetFeatureLoop<cInputBits, 4, countCompilerClassificationTargetStates>(pFeatureCombination, pTrainingSet, aModelUpdateTensor, cTargetStates);
   } else if(cTargetStates <= 1 << 8) {
      TrainingSetTargetFeatureLoop<cInputBits, 8, countCompilerClassificationTargetStates>(pFeatureCombination, pTrainingSet, aModelUpdateTensor, cTargetStates);
   } else if(cTargetStates <= 1 << 16) {
      TrainingSetTargetFeatureLoop<cInputBits, 16, countCompilerClassificationTargetStates>(pFeatureCombination, pTrainingSet, aModelUpdateTensor, cTargetStates);
   } else if(static_cast<uint64_t>(cTargetStates) <= uint64_t { 1 } << 32) {
      // if this is a 32 bit system, then m_cStates can't be 0x100000000 or above, because we would have checked that when converting the 64 bit numbers into size_t, and m_cStates will be promoted to a 64 bit number for the above comparison
      // if this is a 64 bit system, then this comparison is fine

      // TODO : perhaps we should change m_cStates into m_iStateMax so that we don't need to do the above promotion to 64 bits.. we can make it <= 0xFFFFFFFF.  Write a function to fill the lowest bits with ones for any number of bits

      TrainingSetTargetFeatureLoop<cInputBits, 32, countCompilerClassificationTargetStates>(pFeatureCombination, pTrainingSet, aModelUpdateTensor, cTargetStates);
   } else {
      // our interface doesn't allow more than 64 bits, so even if size_t was bigger then we don't need to examine higher
      static_assert(63 == CountBitsRequiredPositiveMax<IntegerDataType>(), "");
      TrainingSetTargetFeatureLoop<cInputBits, 64, countCompilerClassificationTargetStates>(pFeatureCombination, pTrainingSet, aModelUpdateTensor, cTargetStates);
   }
}

// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
template<unsigned int cInputBits, unsigned int cTargetBits, ptrdiff_t countCompilerClassificationTargetStates>
static FractionalDataType ValidationSetTargetFeatureLoop(const FeatureCombinationCore * const pFeatureCombination, DataSetFeatureCombination * const pValidationSet, const FractionalDataType * const aModelUpdateTensor, const size_t cTargetStates) {
   LOG(TraceLevelVerbose, "Entering ValidationSetTargetFeatureLoop");

   const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
   const size_t cCases = pValidationSet->GetCountCases();
   EBM_ASSERT(0 < cCases);

   if(0 == pFeatureCombination->m_cFeatures) {
      if(IsRegression(countCompilerClassificationTargetStates)) {
         FractionalDataType * pResidualError = pValidationSet->GetResidualPointer();
         const FractionalDataType * const pResidualErrorEnd = pResidualError + cCases;

         const FractionalDataType smallChangeToPrediction = aModelUpdateTensor[0];

         FractionalDataType rootMeanSquareError = 0;
         while(pResidualErrorEnd != pResidualError) {
            // this will apply a small fix to our existing ValidationPredictionScores, either positive or negative, whichever is needed
            const FractionalDataType residualError = EbmStatistics::ComputeRegressionResidualError(*pResidualError - smallChangeToPrediction);
            rootMeanSquareError += residualError * residualError;
            *pResidualError = residualError;
            ++pResidualError;
         }

         rootMeanSquareError /= pValidationSet->GetCountCases();
         LOG(TraceLevelVerbose, "Exited ValidationSetTargetFeatureLoop - Zero dimensions");
         return sqrt(rootMeanSquareError);
      } else {
         EBM_ASSERT(IsClassification(countCompilerClassificationTargetStates));
         FractionalDataType * pValidationPredictionScores = pValidationSet->GetPredictionScores();
         const StorageDataTypeCore * pTargetData = pValidationSet->GetTargetDataPointer();

         const FractionalDataType * const pValidationPredictionEnd = pValidationPredictionScores + cVectorLength * cCases;

         FractionalDataType sumLogLoss = 0;
         if(IsBinaryClassification(countCompilerClassificationTargetStates)) {
            const FractionalDataType smallChangeToPredictionScores = aModelUpdateTensor[0];
            while(pValidationPredictionEnd != pValidationPredictionScores) {
               StorageDataTypeCore targetData = *pTargetData;
               // this will apply a small fix to our existing ValidationPredictionScores, either positive or negative, whichever is needed
               const FractionalDataType validationPredictionScores = *pValidationPredictionScores + smallChangeToPredictionScores;
               *pValidationPredictionScores = validationPredictionScores;
               sumLogLoss += EbmStatistics::ComputeClassificationSingleCaseLogLossBinaryclass(validationPredictionScores, targetData);
               ++pValidationPredictionScores;
               ++pTargetData;
            }
         } else {
            const FractionalDataType * pValues = aModelUpdateTensor;
            while(pValidationPredictionEnd != pValidationPredictionScores) {
               StorageDataTypeCore targetData = *pTargetData;
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
               // TODO: store the result of std::exp above for the index that we care about above since exp(..) is going to be expensive and probably even more expensive than an unconditional branch
               sumLogLoss += EbmStatistics::ComputeClassificationSingleCaseLogLossMulticlass(sumExp, pValidationPredictionScores - cVectorLength, targetData);
               ++pTargetData;
            }
         }
         LOG(TraceLevelVerbose, "Exited ValidationSetTargetFeatureLoop - Zero dimensions");
         return sumLogLoss;
      }
      EBM_ASSERT(false);
   }

   const size_t cItemsPerBitPackDataUnit = pFeatureCombination->m_cItemsPerBitPackDataUnit;
   const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackDataUnit);
   const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);
   const StorageDataTypeCore * pInputData = pValidationSet->GetDataPointer(pFeatureCombination);

   if(IsRegression(countCompilerClassificationTargetStates)) {
      FractionalDataType * pResidualError = pValidationSet->GetResidualPointer();
      const FractionalDataType * const pResidualErrorLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete = pResidualError + (static_cast<ptrdiff_t>(cCases) - cItemsPerBitPackDataUnit);

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
            const FractionalDataType smallChangeToPrediction = aModelUpdateTensor[iBin * cVectorLength];
            // this will apply a small fix to our existing ValidationPredictionScores, either positive or negative, whichever is needed
            const FractionalDataType residualError = EbmStatistics::ComputeRegressionResidualError(*pResidualError - smallChangeToPrediction);
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
         EBM_ASSERT(0 == (pResidualErrorEnd - pResidualError) % cVectorLength);
         cItemsRemaining = (pResidualErrorEnd - pResidualError) / cVectorLength;
         EBM_ASSERT(0 < cItemsRemaining);
         EBM_ASSERT(cItemsRemaining <= cItemsPerBitPackDataUnit);
         goto one_last_loop_regression;
      }
      EBM_ASSERT(pResidualError == pResidualErrorEnd); // after our second iteration we should have finished everything!

      rootMeanSquareError /= pValidationSet->GetCountCases();
      LOG(TraceLevelVerbose, "Exited ValidationSetTargetFeatureLoop");
      return sqrt(rootMeanSquareError);
   } else {
      EBM_ASSERT(IsClassification(countCompilerClassificationTargetStates));
      FractionalDataType * pValidationPredictionScores = pValidationSet->GetPredictionScores();
      const StorageDataTypeCore * pTargetData = pValidationSet->GetTargetDataPointer();

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
            const FractionalDataType * pValues = &aModelUpdateTensor[iBin * cVectorLength];

            if(IsBinaryClassification(countCompilerClassificationTargetStates)) {
               const FractionalDataType smallChangeToPredictionScores = pValues[0];
               // this will apply a small fix to our existing ValidationPredictionScores, either positive or negative, whichever is needed
               const FractionalDataType validationPredictionScores = *pValidationPredictionScores + smallChangeToPredictionScores;
               *pValidationPredictionScores = validationPredictionScores;
               sumLogLoss += EbmStatistics::ComputeClassificationSingleCaseLogLossBinaryclass(validationPredictionScores, targetData);
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
               // TODO: store the result of std::exp above for the index that we care about above since exp(..) is going to be expensive and probably even more expensive than an unconditional branch
               sumLogLoss += EbmStatistics::ComputeClassificationSingleCaseLogLossMulticlass(sumExp, pValidationPredictionScores - cVectorLength, targetData);
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
         EBM_ASSERT(0 == (pValidationPredictionScoresEnd - pValidationPredictionScores) % cVectorLength);
         cItemsRemaining = (pValidationPredictionScoresEnd - pValidationPredictionScores) / cVectorLength;
         EBM_ASSERT(0 < cItemsRemaining);
         EBM_ASSERT(cItemsRemaining <= cItemsPerBitPackDataUnit);
         goto one_last_loop_classification;
      }
      EBM_ASSERT(pValidationPredictionScores == pValidationPredictionScoresEnd); // after our second iteration we should have finished everything!

      LOG(TraceLevelVerbose, "Exited ValidationSetTargetFeatureLoop");
      return sumLogLoss;
   }
}

// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
template<unsigned int cInputBits, ptrdiff_t countCompilerClassificationTargetStates>
static FractionalDataType ValidationSetInputFeatureLoop(const FeatureCombinationCore * const pFeatureCombination, DataSetFeatureCombination * const pValidationSet, const FractionalDataType * const aModelUpdateTensor, const size_t cTargetStates) {
   if(cTargetStates <= 1 << 1) {
      return ValidationSetTargetFeatureLoop<cInputBits, 1, countCompilerClassificationTargetStates>(pFeatureCombination, pValidationSet, aModelUpdateTensor, cTargetStates);
   } else if(cTargetStates <= 1 << 2) {
      return ValidationSetTargetFeatureLoop<cInputBits, 2, countCompilerClassificationTargetStates>(pFeatureCombination, pValidationSet, aModelUpdateTensor, cTargetStates);
   } else if(cTargetStates <= 1 << 4) {
      return ValidationSetTargetFeatureLoop<cInputBits, 4, countCompilerClassificationTargetStates>(pFeatureCombination, pValidationSet, aModelUpdateTensor, cTargetStates);
   } else if(cTargetStates <= 1 << 8) {
      return ValidationSetTargetFeatureLoop<cInputBits, 8, countCompilerClassificationTargetStates>(pFeatureCombination, pValidationSet, aModelUpdateTensor, cTargetStates);
   } else if(cTargetStates <= 1 << 16) {
      return ValidationSetTargetFeatureLoop<cInputBits, 16, countCompilerClassificationTargetStates>(pFeatureCombination, pValidationSet, aModelUpdateTensor, cTargetStates);
   } else if(static_cast<uint64_t>(cTargetStates) <= uint64_t { 1 } << 32) {
      // if this is a 32 bit system, then m_cStates can't be 0x100000000 or above, because we would have checked that when converting the 64 bit numbers into size_t, and m_cStates will be promoted to a 64 bit number for the above comparison
      // if this is a 64 bit system, then this comparison is fine

      // TODO : perhaps we should change m_cStates into m_iStateMax so that we don't need to do the above promotion to 64 bits.. we can make it <= 0xFFFFFFFF.  Write a function to fill the lowest bits with ones for any number of bits

      return ValidationSetTargetFeatureLoop<cInputBits, 32, countCompilerClassificationTargetStates>(pFeatureCombination, pValidationSet, aModelUpdateTensor, cTargetStates);
   } else {
      // our interface doesn't allow more than 64 bits, so even if size_t was bigger then we don't need to examine higher
      static_assert(63 == CountBitsRequiredPositiveMax<IntegerDataType>(), "");
      return ValidationSetTargetFeatureLoop<cInputBits, 64, countCompilerClassificationTargetStates>(pFeatureCombination, pValidationSet, aModelUpdateTensor, cTargetStates);
   }
}

union CachedThreadResourcesUnion {
   CachedTrainingThreadResources<true> regression;
   CachedTrainingThreadResources<false> classification;

   CachedThreadResourcesUnion(const bool bRegression, const size_t cVectorLength) {
      LOG(TraceLevelInfo, "Entered CachedThreadResourcesUnion: bRegression=%u, cVectorLength=%zu", static_cast<unsigned int>(bRegression), cVectorLength);
      if(bRegression) {
         // member classes inside a union requre explicit call to constructor
         new(&regression) CachedTrainingThreadResources<true>(cVectorLength);
      } else {
         // member classes inside a union requre explicit call to constructor
         new(&classification) CachedTrainingThreadResources<false>(cVectorLength);
      }
      LOG(TraceLevelInfo, "Exited CachedThreadResourcesUnion");
   }

   ~CachedThreadResourcesUnion() {
      // TODO: figure out why this is being called, and if that is bad!
      //LOG(TraceLevelError, "ERROR ~CachedThreadResourcesUnion called.  It's union destructors should be called explicitly");

      // we don't have enough information here to delete this object, so we do it from our caller
      // we still need this destructor for a technicality that it might be called
      // if there were an excpetion generated in the initializer list which it is constructed in
      // but we have been careful to ensure that the class we are including it in doesn't thow exceptions in the
      // initializer list
   }
};

// TODO: rename this EbmTrainingState
class TmlState {
public:
   const bool m_bRegression;
   const size_t m_cTargetStates;

   const size_t m_cFeatureCombinations;
   FeatureCombinationCore ** const m_apFeatureCombinations;

   // TODO : can we internalize these so that they are not pointers and are therefore subsumed into our class
   DataSetFeatureCombination * m_pTrainingSet;
   DataSetFeatureCombination * m_pValidationSet;

   const size_t m_cSamplingSets;

   SamplingMethod ** m_apSamplingSets;
   SegmentedRegionCore<ActiveDataType, FractionalDataType> ** m_apCurrentModel;
   SegmentedRegionCore<ActiveDataType, FractionalDataType> ** m_apBestModel;

   FractionalDataType m_bestModelMetric;

   SegmentedRegionCore<ActiveDataType, FractionalDataType> * const m_pSmallChangeToModelOverwriteSingleSamplingSet;
   SegmentedRegionCore<ActiveDataType, FractionalDataType> * const m_pSmallChangeToModelAccumulatedFromSamplingSets;

   const size_t m_cFeatures;
   // TODO : in the future, we can allocate this inside a function so that even the objects inside are const
   FeatureInternalCore * const m_aFeatures;

   CachedThreadResourcesUnion m_cachedThreadResourcesUnion;

   TmlState(const bool bRegression, const size_t cTargetStates, const size_t cFeatures, const size_t cFeatureCombinations, const size_t cSamplingSets)
      : m_bRegression(bRegression)
      , m_cTargetStates(cTargetStates)
      , m_cFeatureCombinations(cFeatureCombinations)
      , m_apFeatureCombinations(0 == cFeatureCombinations ? nullptr : FeatureCombinationCore::AllocateFeatureCombinations(cFeatureCombinations))
      , m_pTrainingSet(nullptr)
      , m_pValidationSet(nullptr)
      , m_cSamplingSets(cSamplingSets)
      , m_apSamplingSets(nullptr)
      , m_apCurrentModel(nullptr)
      , m_apBestModel(nullptr)
      , m_bestModelMetric(FractionalDataType { std::numeric_limits<FractionalDataType>::infinity() })
      , m_pSmallChangeToModelOverwriteSingleSamplingSet(SegmentedRegionCore<ActiveDataType, FractionalDataType>::Allocate(k_cDimensionsMax, GetVectorLengthFlatCore(cTargetStates)))
      , m_pSmallChangeToModelAccumulatedFromSamplingSets(SegmentedRegionCore<ActiveDataType, FractionalDataType>::Allocate(k_cDimensionsMax, GetVectorLengthFlatCore(cTargetStates)))
      , m_cFeatures(cFeatures)
      , m_aFeatures(0 == cFeatures || IsMultiplyError(sizeof(FeatureInternalCore), cFeatures) ? nullptr : static_cast<FeatureInternalCore *>(malloc(sizeof(FeatureInternalCore) * cFeatures)))
      // we catch any errors in the constructor, so this should not be able to throw
      , m_cachedThreadResourcesUnion(bRegression, GetVectorLengthFlatCore(cTargetStates)) {
   }
   
   ~TmlState() {
      LOG(TraceLevelInfo, "Entered ~EbmTrainingState");

      if(m_bRegression) {
         // member classes inside a union requre explicit call to destructor
         LOG(TraceLevelInfo, "~EbmTrainingState identified as regression type");
         m_cachedThreadResourcesUnion.regression.~CachedTrainingThreadResources();
      } else {
         // member classes inside a union requre explicit call to destructor
         LOG(TraceLevelInfo, "~EbmTrainingState identified as classification type");
         m_cachedThreadResourcesUnion.classification.~CachedTrainingThreadResources();
      }

      SamplingWithReplacement::FreeSamplingSets(m_cSamplingSets, m_apSamplingSets);

      delete m_pTrainingSet;
      delete m_pValidationSet;

      FeatureCombinationCore::FreeFeatureCombinations(m_cFeatureCombinations, m_apFeatureCombinations);

      free(m_aFeatures);

      DeleteSegmentsCore(m_cFeatureCombinations, m_apCurrentModel);
      DeleteSegmentsCore(m_cFeatureCombinations, m_apBestModel);
      SegmentedRegionCore<ActiveDataType, FractionalDataType>::Free(m_pSmallChangeToModelOverwriteSingleSamplingSet);
      SegmentedRegionCore<ActiveDataType, FractionalDataType>::Free(m_pSmallChangeToModelAccumulatedFromSamplingSets);

      LOG(TraceLevelInfo, "Exited ~EbmTrainingState");
   }

   bool Initialize(const IntegerDataType randomSeed, const EbmCoreFeature * const aFeatures, const EbmCoreFeatureCombination * const aFeatureCombinations, const IntegerDataType * featureCombinationIndexes, const size_t cTrainingCases, const void * const aTrainingTargets, const IntegerDataType * const aTrainingData, const FractionalDataType * const aTrainingPredictionScores, const size_t cValidationCases, const void * const aValidationTargets, const IntegerDataType * const aValidationData, const FractionalDataType * const aValidationPredictionScores) {
      LOG(TraceLevelInfo, "Entered EbmTrainingState::Initialize");
      try {
         if(m_bRegression) {
            if(m_cachedThreadResourcesUnion.regression.IsError()) {
               LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize m_cachedThreadResourcesUnion.regression.IsError()");
               return true;
            }
         } else {
            if(m_cachedThreadResourcesUnion.classification.IsError()) {
               LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize m_cachedThreadResourcesUnion.classification.IsError()");
               return true;
            }
         }

         if(0 != m_cFeatures && nullptr == m_aFeatures) {
            LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize 0 != m_cFeatures && nullptr == m_aFeatures");
            return true;
         }

         if(UNLIKELY(0 != m_cFeatureCombinations && nullptr == m_apFeatureCombinations)) {
            LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize 0 != m_cFeatureCombinations && nullptr == m_apFeatureCombinations");
            return true;
         }

         if(UNLIKELY(nullptr == m_pSmallChangeToModelOverwriteSingleSamplingSet)) {
            LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize nullptr == m_pSmallChangeToModelOverwriteSingleSamplingSet");
            return true;
         }

         if(UNLIKELY(nullptr == m_pSmallChangeToModelAccumulatedFromSamplingSets)) {
            LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize nullptr == m_pSmallChangeToModelAccumulatedFromSamplingSets");
            return true;
         }

         LOG(TraceLevelInfo, "EbmTrainingState::Initialize starting feature processing");
         if(0 != m_cFeatures) {
            EBM_ASSERT(!IsMultiplyError(m_cFeatures, sizeof(*aFeatures))); // if this overflows then our caller should not have been able to allocate the array
            const EbmCoreFeature * pFeatureInitialize = aFeatures;
            const EbmCoreFeature * const pFeatureEnd = &aFeatures[m_cFeatures];
            EBM_ASSERT(pFeatureInitialize < pFeatureEnd);
            size_t iFeatureInitialize = 0;
            do {
               static_assert(FeatureTypeCore::OrdinalCore == static_cast<FeatureTypeCore>(FeatureTypeOrdinal), "FeatureTypeCore::OrdinalCore must have the same value as FeatureTypeOrdinal");
               static_assert(FeatureTypeCore::NominalCore == static_cast<FeatureTypeCore>(FeatureTypeNominal), "FeatureTypeCore::NominalCore must have the same value as FeatureTypeNominal");
               EBM_ASSERT(FeatureTypeOrdinal == pFeatureInitialize->featureType || FeatureTypeNominal == pFeatureInitialize->featureType);
               FeatureTypeCore featureTypeCore = static_cast<FeatureTypeCore>(pFeatureInitialize->featureType);

               IntegerDataType countStates = pFeatureInitialize->countBins;
               EBM_ASSERT(0 <= countStates); // we can handle 1 == cStates or 0 == cStates even though that's a degenerate case that shouldn't be trained on (dimensions with 1 state don't contribute anything since they always have the same value).  0 cases could only occur if there were zero training and zero validation cases since the features would require a value, even if it was 0
               if(!IsNumberConvertable<size_t, IntegerDataType>(countStates)) {
                  LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize !IsNumberConvertable<size_t, IntegerDataType>(countStates)");
                  return true;
               }
               size_t cStates = static_cast<size_t>(countStates);
               if(cStates <= 1) {
                  EBM_ASSERT(0 != cStates || 0 == cTrainingCases && 0 == cValidationCases);
                  LOG(TraceLevelInfo, "INFO EbmTrainingState::Initialize feature with 0/1 values");
               }

               EBM_ASSERT(0 == pFeatureInitialize->hasMissing || 1 == pFeatureInitialize->hasMissing);
               bool bMissing = 0 != pFeatureInitialize->hasMissing;

               // this is an in-place new, so there is no new memory allocated, and we already knew where it was going, so we don't need the resulting pointer returned
               new (&m_aFeatures[iFeatureInitialize]) FeatureInternalCore(cStates, iFeatureInitialize, featureTypeCore, bMissing);
               // we don't allocate memory and our constructor doesn't have errors, so we shouldn't have an error here

               EBM_ASSERT(0 == pFeatureInitialize->hasMissing); // TODO : implement this, then remove this assert
               EBM_ASSERT(FeatureTypeOrdinal == pFeatureInitialize->featureType); // TODO : implement this, then remove this assert

               ++iFeatureInitialize;
               ++pFeatureInitialize;
            } while(pFeatureEnd != pFeatureInitialize);
         }
         LOG(TraceLevelInfo, "EbmTrainingState::Initialize done feature processing");

         LOG(TraceLevelInfo, "EbmTrainingState::Initialize starting feature combination processing");
         if(0 != m_cFeatureCombinations) {
            const IntegerDataType * pFeatureCombinationIndex = featureCombinationIndexes;
            size_t iFeatureCombination = 0;
            do {
               const EbmCoreFeatureCombination * const pFeatureCombinationInterop = &aFeatureCombinations[iFeatureCombination];

               IntegerDataType countFeaturesInCombination = pFeatureCombinationInterop->countFeaturesInCombination;
               EBM_ASSERT(0 <= countFeaturesInCombination);
               if(!IsNumberConvertable<size_t, IntegerDataType>(countFeaturesInCombination)) {
                  LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize !IsNumberConvertable<size_t, IntegerDataType>(countFeaturesInCombination)");
                  return true;
               }
               size_t cFeaturesInCombination = static_cast<size_t>(countFeaturesInCombination);
               size_t cSignificantFeaturesInCombination = 0;
               const IntegerDataType * const pFeatureCombinationIndexEnd = pFeatureCombinationIndex + cFeaturesInCombination;
               if(UNLIKELY(0 == cFeaturesInCombination)) {
                  LOG(TraceLevelInfo, "INFO EbmTrainingState::Initialize empty feature combination");
               } else {
                  EBM_ASSERT(nullptr != featureCombinationIndexes);
                  const IntegerDataType * pFeatureCombinationIndexTemp = pFeatureCombinationIndex;
                  do {
                     const IntegerDataType indexFeatureInterop = *pFeatureCombinationIndexTemp;
                     EBM_ASSERT(0 <= indexFeatureInterop);
                     if(!IsNumberConvertable<size_t, IntegerDataType>(indexFeatureInterop)) {
                        LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize !IsNumberConvertable<size_t, IntegerDataType>(indexFeatureInterop)");
                        return true;
                     }
                     const size_t iFeatureForCombination = static_cast<size_t>(indexFeatureInterop);
                     EBM_ASSERT(iFeatureForCombination < m_cFeatures);
                     FeatureInternalCore * const pInputFeature = &m_aFeatures[iFeatureForCombination];
                     if(LIKELY(1 < pInputFeature->m_cStates)) {
                        // if we have only 1 state, then we can eliminate the feature from consideration since the resulting tensor loses one dimension but is otherwise indistinquishable from the original data
                        ++cSignificantFeaturesInCombination;
                     } else {
                        LOG(TraceLevelInfo, "INFO EbmTrainingState::Initialize feature combination with no useful features");
                     }
                     ++pFeatureCombinationIndexTemp;
                  } while(pFeatureCombinationIndexEnd != pFeatureCombinationIndexTemp);

                  // TODO : we can allow more dimensions, if some of the dimensions have only 1 state
                  if(k_cDimensionsMax < cSignificantFeaturesInCombination) {
                     // if we try to run with more than k_cDimensionsMax we'll exceed our memory capacity, so let's exit here instead
                     LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize k_cDimensionsMax < cSignificantFeaturesInCombination");
                     return true;
                  }
               }

               FeatureCombinationCore * pFeatureCombination = FeatureCombinationCore::Allocate(cSignificantFeaturesInCombination, iFeatureCombination);
               if(nullptr == pFeatureCombination) {
                  LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize nullptr == pFeatureCombination");
                  return true;
               }
               // assign our pointer directly to our array right now so that we can't loose the memory if we decide to exit due to an error below
               m_apFeatureCombinations[iFeatureCombination] = pFeatureCombination;

               if(LIKELY(0 == cSignificantFeaturesInCombination)) {
                  // move our index forward to the next feature.  
                  // We won't be executing the loop below that would otherwise increment it by the number of features in this feature combination
                  pFeatureCombinationIndex = pFeatureCombinationIndexEnd;
               } else {
                  EBM_ASSERT(nullptr != featureCombinationIndexes);
                  size_t cTensorStates = 1;
                  FeatureCombinationCore::FeatureCombinationEntry * pFeatureCombinationEntry = &pFeatureCombination->m_FeatureCombinationEntry[0];
                  do {
                     const IntegerDataType indexFeatureInterop = *pFeatureCombinationIndex;
                     EBM_ASSERT(0 <= indexFeatureInterop);
                     EBM_ASSERT((IsNumberConvertable<size_t, IntegerDataType>(indexFeatureInterop))); // this was checked above
                     const size_t iFeatureForCombination = static_cast<size_t>(indexFeatureInterop);
                     EBM_ASSERT(iFeatureForCombination < m_cFeatures);
                     const FeatureInternalCore * const pInputFeature = &m_aFeatures[iFeatureForCombination];
                     const size_t cStates = pInputFeature->m_cStates;
                     if(LIKELY(1 < cStates)) {
                        // if we have only 1 state, then we can eliminate the feature from consideration since the resulting tensor loses one dimension but is otherwise indistinquishable from the original data
                        pFeatureCombinationEntry->m_pFeature = pInputFeature;
                        ++pFeatureCombinationEntry;
                        if(IsMultiplyError(cTensorStates, cStates)) {
                           // if this overflows, we definetly won't be able to allocate it
                           LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize IsMultiplyError(cTensorStates, cStates)");
                           return true;
                        }
                        cTensorStates *= cStates;
                     }
                     ++pFeatureCombinationIndex;
                  } while(pFeatureCombinationIndexEnd != pFeatureCombinationIndex);
                  // if cSignificantFeaturesInCombination is zero, don't both initializing pFeatureCombination->m_cItemsPerBitPackDataUnit
                  const size_t cBitsRequiredMin = CountBitsRequiredCore(cTensorStates - 1);
                  pFeatureCombination->m_cItemsPerBitPackDataUnit = GetCountItemsBitPacked(cBitsRequiredMin);
               }
               ++iFeatureCombination;
            } while(iFeatureCombination < m_cFeatureCombinations);
         }
         LOG(TraceLevelInfo, "EbmTrainingState::Initialize finished feature combination processing");

         size_t cVectorLength = GetVectorLengthFlatCore(m_cTargetStates);

         LOG(TraceLevelInfo, "Entered DataSetFeatureCombination for m_pTrainingSet");
         if(0 != cTrainingCases) {
            m_pTrainingSet = new (std::nothrow) DataSetFeatureCombination(true, !m_bRegression, !m_bRegression, m_cFeatureCombinations, m_apFeatureCombinations, cTrainingCases, aTrainingData, aTrainingTargets, aTrainingPredictionScores, cVectorLength);
            if(nullptr == m_pTrainingSet || m_pTrainingSet->IsError()) {
               LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize nullptr == m_pTrainingSet || m_pTrainingSet->IsError()");
               return true;
            }
         }
         LOG(TraceLevelInfo, "Exited DataSetFeatureCombination for m_pTrainingSet %p", static_cast<void *>(m_pTrainingSet));

         LOG(TraceLevelInfo, "Entered DataSetFeatureCombination for m_pValidationSet");
         if(0 != cValidationCases) {
            m_pValidationSet = new (std::nothrow) DataSetFeatureCombination(m_bRegression, !m_bRegression, !m_bRegression, m_cFeatureCombinations, m_apFeatureCombinations, cValidationCases, aValidationData, aValidationTargets, aValidationPredictionScores, cVectorLength);
            if(nullptr == m_pValidationSet || m_pValidationSet->IsError()) {
               LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize nullptr == m_pValidationSet || m_pValidationSet->IsError()");
               return true;
            }
         }
         LOG(TraceLevelInfo, "Exited DataSetFeatureCombination for m_pValidationSet %p", static_cast<void *>(m_pValidationSet));

         RandomStream randomStream(randomSeed);

         EBM_ASSERT(nullptr == m_apSamplingSets);
         if(0 != cTrainingCases) {
            m_apSamplingSets = SamplingWithReplacement::GenerateSamplingSets(&randomStream, m_pTrainingSet, m_cSamplingSets);
            if(UNLIKELY(nullptr == m_apSamplingSets)) {
               LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize nullptr == m_apSamplingSets");
               return true;
            }
         }

         EBM_ASSERT(nullptr == m_apCurrentModel);
         EBM_ASSERT(nullptr == m_apBestModel);
         if(0 != m_cFeatureCombinations && (m_bRegression || 2 <= m_cTargetStates)) {
            m_apCurrentModel = InitializeSegmentsCore(m_cFeatureCombinations, m_apFeatureCombinations, cVectorLength);
            if(nullptr == m_apCurrentModel) {
               LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize nullptr == m_apCurrentModel");
               return true;
            }
            m_apBestModel = InitializeSegmentsCore(m_cFeatureCombinations, m_apFeatureCombinations, cVectorLength);
            if(nullptr == m_apBestModel) {
               LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize nullptr == m_apBestModel");
               return true;
            }
         }

         if(m_bRegression) {
            if(0 != cTrainingCases) {
               InitializeResiduals<k_Regression>(cTrainingCases, aTrainingTargets, aTrainingPredictionScores, m_pTrainingSet->GetResidualPointer(), 0);
            }
            if(0 != cValidationCases) {
               InitializeResiduals<k_Regression>(cValidationCases, aValidationTargets, aValidationPredictionScores, m_pValidationSet->GetResidualPointer(), 0);
            }
         } else {
            if(2 == m_cTargetStates) {
               if(0 != cTrainingCases) {
                  InitializeResiduals<2>(cTrainingCases, aTrainingTargets, aTrainingPredictionScores, m_pTrainingSet->GetResidualPointer(), m_cTargetStates);
               }
            } else {
               if(0 != cTrainingCases) {
                  InitializeResiduals<k_DynamicClassification>(cTrainingCases, aTrainingTargets, aTrainingPredictionScores, m_pTrainingSet->GetResidualPointer(), m_cTargetStates);
               }
            }
         }
         
         LOG(TraceLevelInfo, "Exited EbmTrainingState::Initialize");
         return false;
      } catch (...) {
         // this is here to catch exceptions from RandomStream randomStream(randomSeed), but it could also catch errors if we put any other C++ types in here later
         LOG(TraceLevelWarning, "WARNING EbmTrainingState::Initialize exception");
         return true;
      }
   }
};

#ifndef NDEBUG
void CheckTargets(const size_t cTargetStates, const size_t cCases, const void * const aTargets) {
   if(0 != cCases) {
      if(0 == cTargetStates) {
         // regression!

         const FractionalDataType * pTarget = static_cast<const FractionalDataType *>(aTargets);
         const FractionalDataType * const pTargetEnd = pTarget + cCases;
         do {
            const FractionalDataType data = *pTarget;
            EBM_ASSERT(!std::isnan(data));
            EBM_ASSERT(!std::isinf(data));
            ++pTarget;
         } while(pTargetEnd != pTarget);
      } else {
         // classification

         const IntegerDataType * pTarget = static_cast<const IntegerDataType *>(aTargets);
         const IntegerDataType * const pTargetEnd = pTarget + cCases;
         do {
            const IntegerDataType data = *pTarget;
            EBM_ASSERT(0 <= data);
            EBM_ASSERT((IsNumberConvertable<size_t, IntegerDataType>(data))); // data must be lower than cTargetStates and cTargetStates fits into a size_t which we checked earlier
            EBM_ASSERT(static_cast<size_t>(data) < cTargetStates);
            ++pTarget;
         } while(pTargetEnd != pTarget);
      }
   }
}
#endif // NDEBUG

// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
TmlState * AllocateCore(bool bRegression, IntegerDataType randomSeed, IntegerDataType countFeatures, const EbmCoreFeature * features, IntegerDataType countFeatureCombinations, const EbmCoreFeatureCombination * featureCombinations, const IntegerDataType * featureCombinationIndexes, IntegerDataType countTargetStates, IntegerDataType countTrainingCases, const void * trainingTargets, const IntegerDataType * trainingData, const FractionalDataType * trainingPredictionScores, IntegerDataType countValidationCases, const void * validationTargets, const IntegerDataType * validationData, const FractionalDataType * validationPredictionScores, IntegerDataType countInnerBags) {
   // randomSeed can be any value
   EBM_ASSERT(0 <= countFeatures);
   EBM_ASSERT(0 == countFeatures || nullptr != features);
   EBM_ASSERT(0 <= countFeatureCombinations);
   EBM_ASSERT(0 == countFeatureCombinations || nullptr != featureCombinations);
   // featureCombinationIndexes -> it's legal for featureCombinationIndexes to be nullptr if there are no features indexed by our featureCombinations.  FeatureCombinations can have zero features, so it could be legal for this to be null even if there are featureCombinations
   EBM_ASSERT(bRegression && 0 == countTargetStates || !bRegression && (1 <= countTargetStates || 0 == countTargetStates && 0 == countTrainingCases && 0 == countValidationCases));
   EBM_ASSERT(0 <= countTrainingCases);
   EBM_ASSERT(0 == countTrainingCases || nullptr != trainingTargets);
   EBM_ASSERT(0 == countTrainingCases || 0 == countFeatures || nullptr != trainingData);
   // trainingPredictionScores can be null
   EBM_ASSERT(0 <= countValidationCases); // TODO: change this to make it possible to be 0 if the user doesn't want a validation set
   EBM_ASSERT(0 == countValidationCases || nullptr != validationTargets); // TODO: change this to make it possible to have no validation set
   EBM_ASSERT(0 == countValidationCases || 0 == countFeatures || nullptr != validationData); // TODO: change this to make it possible to have no validation set
   // validationPredictionScores can be null
   EBM_ASSERT(0 <= countInnerBags); // 0 means use the full set (good value).  1 means make a single bag (this is useless but allowed for comparison purposes).  2+ are good numbers of bag

   if(!IsNumberConvertable<size_t, IntegerDataType>(countFeatures)) {
      LOG(TraceLevelWarning, "WARNING AllocateCore !IsNumberConvertable<size_t, IntegerDataType>(countFeatures)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntegerDataType>(countFeatureCombinations)) {
      LOG(TraceLevelWarning, "WARNING AllocateCore !IsNumberConvertable<size_t, IntegerDataType>(countFeatureCombinations)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntegerDataType>(countTargetStates)) {
      LOG(TraceLevelWarning, "WARNING AllocateCore !IsNumberConvertable<size_t, IntegerDataType>(countTargetStates)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntegerDataType>(countTrainingCases)) {
      LOG(TraceLevelWarning, "WARNING AllocateCore !IsNumberConvertable<size_t, IntegerDataType>(countTrainingCases)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntegerDataType>(countValidationCases)) {
      LOG(TraceLevelWarning, "WARNING AllocateCore !IsNumberConvertable<size_t, IntegerDataType>(countValidationCases)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntegerDataType>(countInnerBags)) {
      LOG(TraceLevelWarning, "WARNING AllocateCore !IsNumberConvertable<size_t, IntegerDataType>(countInnerBags)");
      return nullptr;
   }

   size_t cFeatures = static_cast<size_t>(countFeatures);
   size_t cFeatureCombinations = static_cast<size_t>(countFeatureCombinations);
   size_t cTargetStates = static_cast<size_t>(countTargetStates);
   size_t cTrainingCases = static_cast<size_t>(countTrainingCases);
   size_t cValidationCases = static_cast<size_t>(countValidationCases);
   size_t cInnerBags = static_cast<size_t>(countInnerBags);

   size_t cVectorLength = GetVectorLengthFlatCore(cTargetStates);

   if(IsMultiplyError(cVectorLength, cTrainingCases)) {
      LOG(TraceLevelWarning, "WARNING AllocateCore IsMultiplyError(cVectorLength, cTrainingCases)");
      return nullptr;
   }
   if(IsMultiplyError(cVectorLength, cValidationCases)) {
      LOG(TraceLevelWarning, "WARNING AllocateCore IsMultiplyError(cVectorLength, cValidationCases)");
      return nullptr;
   }

#ifndef NDEBUG
   CheckTargets(cTargetStates, cTrainingCases, trainingTargets);
   CheckTargets(cTargetStates, cValidationCases, validationTargets);
#endif // NDEBUG

   LOG(TraceLevelInfo, "Entered EbmTrainingState");
   TmlState * const pTmlState = new (std::nothrow) TmlState(bRegression, cTargetStates, cFeatures, cFeatureCombinations, cInnerBags);
   LOG(TraceLevelInfo, "Exited EbmTrainingState %p", static_cast<void *>(pTmlState));
   if(UNLIKELY(nullptr == pTmlState)) {
      LOG(TraceLevelWarning, "WARNING AllocateCore nullptr == pTmlState");
      return nullptr;
   }
   if(UNLIKELY(pTmlState->Initialize(randomSeed, features, featureCombinations, featureCombinationIndexes, cTrainingCases, trainingTargets, trainingData, trainingPredictionScores, cValidationCases, validationTargets, validationData, validationPredictionScores))) {
      LOG(TraceLevelWarning, "WARNING AllocateCore pTmlState->Initialize");
      delete pTmlState;
      return nullptr;
   }
   return pTmlState;
}

EBMCORE_IMPORT_EXPORT PEbmTraining EBMCORE_CALLING_CONVENTION InitializeTrainingRegression(IntegerDataType randomSeed, IntegerDataType countFeatures, const EbmCoreFeature * features, IntegerDataType countFeatureCombinations, const EbmCoreFeatureCombination * featureCombinations, const IntegerDataType * featureCombinationIndexes, IntegerDataType countTrainingCases, const FractionalDataType * trainingTargets, const IntegerDataType * trainingData, const FractionalDataType * trainingPredictionScores, IntegerDataType countValidationCases, const FractionalDataType * validationTargets, const IntegerDataType * validationData, const FractionalDataType * validationPredictionScores, IntegerDataType countInnerBags) {
   LOG(TraceLevelInfo, "Entered InitializeTrainingRegression: randomSeed=%" IntegerDataTypePrintf ", countFeatures=%" IntegerDataTypePrintf ", features=%p, countFeatureCombinations=%" IntegerDataTypePrintf ", featureCombinations=%p, featureCombinationIndexes=%p, countTrainingCases=%" IntegerDataTypePrintf ", trainingTargets=%p, trainingData=%p, trainingPredictionScores=%p, countValidationCases=%" IntegerDataTypePrintf ", validationTargets=%p, validationData=%p, validationPredictionScores=%p, countInnerBags=%" IntegerDataTypePrintf, randomSeed, countFeatures, static_cast<const void *>(features), countFeatureCombinations, static_cast<const void *>(featureCombinations), static_cast<const void *>(featureCombinationIndexes), countTrainingCases, static_cast<const void *>(trainingTargets), static_cast<const void *>(trainingData), static_cast<const void *>(trainingPredictionScores), countValidationCases, static_cast<const void *>(validationTargets), static_cast<const void *>(validationData), static_cast<const void *>(validationPredictionScores), countInnerBags);
   PEbmTraining pEbmTraining = reinterpret_cast<PEbmTraining>(AllocateCore(true, randomSeed, countFeatures, features, countFeatureCombinations, featureCombinations, featureCombinationIndexes, 0, countTrainingCases, trainingTargets, trainingData, trainingPredictionScores, countValidationCases, validationTargets, validationData, validationPredictionScores, countInnerBags));
   LOG(TraceLevelInfo, "Exited InitializeTrainingRegression %p", static_cast<void *>(pEbmTraining));
   return pEbmTraining;
}

EBMCORE_IMPORT_EXPORT PEbmTraining EBMCORE_CALLING_CONVENTION InitializeTrainingClassification(IntegerDataType randomSeed, IntegerDataType countFeatures, const EbmCoreFeature * features, IntegerDataType countFeatureCombinations, const EbmCoreFeatureCombination * featureCombinations, const IntegerDataType * featureCombinationIndexes, IntegerDataType countTargetStates, IntegerDataType countTrainingCases, const IntegerDataType * trainingTargets, const IntegerDataType * trainingData, const FractionalDataType * trainingPredictionScores, IntegerDataType countValidationCases, const IntegerDataType * validationTargets, const IntegerDataType * validationData, const FractionalDataType * validationPredictionScores, IntegerDataType countInnerBags) {
   LOG(TraceLevelInfo, "Entered InitializeTrainingClassification: randomSeed=%" IntegerDataTypePrintf ", countFeatures=%" IntegerDataTypePrintf ", features=%p, countFeatureCombinations=%" IntegerDataTypePrintf ", featureCombinations=%p, featureCombinationIndexes=%p, countTargetStates=%" IntegerDataTypePrintf ", countTrainingCases=%" IntegerDataTypePrintf ", trainingTargets=%p, trainingData=%p, trainingPredictionScores=%p, countValidationCases=%" IntegerDataTypePrintf ", validationTargets=%p, validationData=%p, validationPredictionScores=%p, countInnerBags=%" IntegerDataTypePrintf, randomSeed, countFeatures, static_cast<const void *>(features), countFeatureCombinations, static_cast<const void *>(featureCombinations), static_cast<const void *>(featureCombinationIndexes), countTargetStates, countTrainingCases, static_cast<const void *>(trainingTargets), static_cast<const void *>(trainingData), static_cast<const void *>(trainingPredictionScores), countValidationCases, static_cast<const void *>(validationTargets), static_cast<const void *>(validationData), static_cast<const void *>(validationPredictionScores), countInnerBags);
   PEbmTraining pEbmTraining = reinterpret_cast<PEbmTraining>(AllocateCore(false, randomSeed, countFeatures, features, countFeatureCombinations, featureCombinations, featureCombinationIndexes, countTargetStates, countTrainingCases, trainingTargets, trainingData, trainingPredictionScores, countValidationCases, validationTargets, validationData, validationPredictionScores, countInnerBags));
   LOG(TraceLevelInfo, "Exited InitializeTrainingClassification %p", static_cast<void *>(pEbmTraining));
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

// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
template<ptrdiff_t countCompilerClassificationTargetStates>
static FractionalDataType * GenerateModelFeatureCombinationUpdatePerTargetStates(TmlState * const pTmlState, const size_t iFeatureCombination, const FractionalDataType learningRate, const size_t cTreeSplitsMax, const size_t cCasesRequiredForSplitParentMin, const FractionalDataType * const aTrainingWeights, const FractionalDataType * const aValidationWeights, FractionalDataType * const pGainReturn) {
   // TODO remove this after we use aTrainingWeights and aValidationWeights into the GenerateModelFeatureCombinationUpdatePerTargetStates function
   UNUSED(aTrainingWeights);
   UNUSED(aValidationWeights);

   LOG(TraceLevelVerbose, "Entered GenerateModelFeatureCombinationUpdatePerTargetStates");

   if(nullptr != pGainReturn) {
      *pGainReturn = 0; // always set this, even on errors.  We might as well do it here at the top
   }

   const size_t cSamplingSetsAfterZero = (0 == pTmlState->m_cSamplingSets) ? 1 : pTmlState->m_cSamplingSets;
   CachedTrainingThreadResources<IsRegression(countCompilerClassificationTargetStates)> * const pCachedThreadResources = GetCachedThreadResources<IsRegression(countCompilerClassificationTargetStates)>(pTmlState);
   const FeatureCombinationCore * const pFeatureCombination = pTmlState->m_apFeatureCombinations[iFeatureCombination];
   const size_t cDimensions = pFeatureCombination->m_cFeatures;

   pTmlState->m_pSmallChangeToModelAccumulatedFromSamplingSets->SetCountDimensions(cDimensions);
   pTmlState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Reset();

   // if pTmlState->m_apSamplingSets is nullptr, then we should have zero training cases
   // we can't be partially constructed here since then we wouldn't have returned our state pointer to our caller
   EBM_ASSERT(!pTmlState->m_apSamplingSets == !pTmlState->m_pTrainingSet); // m_pTrainingSet and m_apSamplingSets should be the same null-ness in that they should either both be null or both be non-null (although different non-null values)
   FractionalDataType totalGain = 0;
   if(nullptr != pTmlState->m_apSamplingSets) {
      pTmlState->m_pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDimensions(cDimensions);

      for(size_t iSamplingSet = 0; iSamplingSet < cSamplingSetsAfterZero; ++iSamplingSet) {
         FractionalDataType gain = 0;
         if(0 == pFeatureCombination->m_cFeatures) {
            if(TrainZeroDimensional<countCompilerClassificationTargetStates>(pCachedThreadResources, pTmlState->m_apSamplingSets[iSamplingSet], pTmlState->m_pSmallChangeToModelOverwriteSingleSamplingSet, pTmlState->m_cTargetStates)) {
               return nullptr;
            }
         } else if(1 == pFeatureCombination->m_cFeatures) {
            if(TrainSingleDimensional<countCompilerClassificationTargetStates>(pCachedThreadResources, pTmlState->m_apSamplingSets[iSamplingSet], pFeatureCombination, cTreeSplitsMax, cCasesRequiredForSplitParentMin, pTmlState->m_pSmallChangeToModelOverwriteSingleSamplingSet, &gain, pTmlState->m_cTargetStates)) {
               return nullptr;
            }
         } else {
            if(TrainMultiDimensional<countCompilerClassificationTargetStates, 0>(pCachedThreadResources, pTmlState->m_apSamplingSets[iSamplingSet], pFeatureCombination, pTmlState->m_pSmallChangeToModelOverwriteSingleSamplingSet, pTmlState->m_cTargetStates)) {
               return nullptr;
            }
         }
         totalGain += gain;
         // TODO : when we thread this code, let's have each thread take a lock and update the combined line segment.  They'll each do it while the others are working, so there should be no blocking and our final result won't require adding by the main thread
         if(pTmlState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Add(*pTmlState->m_pSmallChangeToModelOverwriteSingleSamplingSet)) {
            return nullptr;
         }
      }
      totalGain /= static_cast<FractionalDataType>(cSamplingSetsAfterZero);

      LOG(TraceLevelVerbose, "GenerateModelFeatureCombinationUpdatePerTargetStates done sampling set loop");

      // we need to divide by the number of sampling sets that we constructed this from.
      // We also need to slow down our growth so that the more relevant Features get a chance to grow first so we multiply by a user defined learning rate
      if(IsClassification(countCompilerClassificationTargetStates)) {
#ifdef EXPAND_BINARY_LOGITS
         constexpr bool bExpandBinaryLogits = true;
#else // EXPAND_BINARY_LOGITS
         constexpr bool bExpandBinaryLogits = false;
#endif // EXPAND_BINARY_LOGITS

         //if(0 <= k_iZeroResidual || 2 == pTmlState->m_cTargetStates && bExpandBinaryLogits) {
         //   EBM_ASSERT(2 <= pTmlState->m_cTargetStates);
         //   // TODO : for classification with residual zeroing, is our learning rate essentially being inflated as pTmlState->m_cTargetStates goes up?  If so, maybe we should divide by pTmlState->m_cTargetStates here to keep learning rates as equivalent as possible..  Actually, I think the real solution here is that 
         //   pTmlState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Multiply(learningRate / cSamplingSetsAfterZero * (pTmlState->m_cTargetStates - 1) / pTmlState->m_cTargetStates);
         //} else {
         //   // TODO : for classification, is our learning rate essentially being inflated as pTmlState->m_cTargetStates goes up?  If so, maybe we should divide by pTmlState->m_cTargetStates here to keep learning rates equivalent as possible
         //   pTmlState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Multiply(learningRate / cSamplingSetsAfterZero);
         //}

         constexpr bool bDividing = bExpandBinaryLogits && 2 == countCompilerClassificationTargetStates;
         if(bDividing) {
            pTmlState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Multiply(learningRate / cSamplingSetsAfterZero / 2);
         } else {
            pTmlState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Multiply(learningRate / cSamplingSetsAfterZero);
         }
      } else {
         pTmlState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Multiply(learningRate / cSamplingSetsAfterZero);
      }
   }

   if(0 != cDimensions) {
      // pTmlState->m_pSmallChangeToModelAccumulatedFromSamplingSets was reset above, so it isn't expanded.  We want to expand it before calling ValidationSetInputFeatureLoop so that we can more efficiently lookup the results by index rather than do a binary search
      size_t acDivisionIntegersEnd[k_cDimensionsMax];
      size_t iDimension = 0;
      do {
         acDivisionIntegersEnd[iDimension] = pFeatureCombination->m_FeatureCombinationEntry[iDimension].m_pFeature->m_cStates;
         ++iDimension;
      } while(iDimension < cDimensions);
      if(pTmlState->m_pSmallChangeToModelAccumulatedFromSamplingSets->Expand(acDivisionIntegersEnd)) {
         return nullptr;
      }
   }

   if(nullptr != pGainReturn) {
      *pGainReturn = totalGain;
   }

   LOG(TraceLevelVerbose, "Exited GenerateModelFeatureCombinationUpdatePerTargetStates");
   return pTmlState->m_pSmallChangeToModelAccumulatedFromSamplingSets->m_aValues;
}

template<ptrdiff_t iPossibleCompilerOptimizedTargetStates>
TML_INLINE FractionalDataType * CompilerRecursiveGenerateModelFeatureCombinationUpdate(const size_t cRuntimeTargetStates, TmlState * const pTmlState, const size_t iFeatureCombination, const FractionalDataType learningRate, const size_t cTreeSplitsMax, const size_t cCasesRequiredForSplitParentMin, const FractionalDataType * const aTrainingWeights, const FractionalDataType * const aValidationWeights, FractionalDataType * const pGainReturn) {
   EBM_ASSERT(IsClassification(iPossibleCompilerOptimizedTargetStates));
   if(iPossibleCompilerOptimizedTargetStates == cRuntimeTargetStates) {
      EBM_ASSERT(cRuntimeTargetStates <= k_cCompilerOptimizedTargetStatesMax);
      return GenerateModelFeatureCombinationUpdatePerTargetStates<iPossibleCompilerOptimizedTargetStates>(pTmlState, iFeatureCombination, learningRate, cTreeSplitsMax, cCasesRequiredForSplitParentMin, aTrainingWeights, aValidationWeights, pGainReturn);
   } else {
      return CompilerRecursiveGenerateModelFeatureCombinationUpdate<iPossibleCompilerOptimizedTargetStates + 1>(cRuntimeTargetStates, pTmlState, iFeatureCombination, learningRate, cTreeSplitsMax, cCasesRequiredForSplitParentMin, aTrainingWeights, aValidationWeights, pGainReturn);
   }
}

template<>
TML_INLINE FractionalDataType * CompilerRecursiveGenerateModelFeatureCombinationUpdate<k_cCompilerOptimizedTargetStatesMax + 1>(const size_t cRuntimeTargetStates, TmlState * const pTmlState, const size_t iFeatureCombination, const FractionalDataType learningRate, const size_t cTreeSplitsMax, const size_t cCasesRequiredForSplitParentMin, const FractionalDataType * const aTrainingWeights, const FractionalDataType * const aValidationWeights, FractionalDataType * const pGainReturn) {
   UNUSED(cRuntimeTargetStates);
   // it is logically possible, but uninteresting to have a classification with 1 target state, so let our runtime system handle those unlikley and uninteresting cases
   EBM_ASSERT(k_cCompilerOptimizedTargetStatesMax < cRuntimeTargetStates);
   return GenerateModelFeatureCombinationUpdatePerTargetStates<k_DynamicClassification>(pTmlState, iFeatureCombination, learningRate, cTreeSplitsMax, cCasesRequiredForSplitParentMin, aTrainingWeights, aValidationWeights, pGainReturn);
}

// we made this a global because if we had put this variable inside the EbmTrainingState object, then we would need to dereference that before getting the count.  By making this global we can send a log message incase a bad EbmTrainingState object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more times than desired, but we can live with that
static unsigned int g_cLogGenerateModelFeatureCombinationUpdateParametersMessages = 10;

// TODO : we can make GenerateModelFeatureCombinationUpdate callable by multiple threads so that this step could be parallelized before making a decision and applying one of the updates.  Right now we're accessing scratch space in the pTmlState object, but we can move that to a thread resident object.  Do do this, we would need to have our caller allocate our tensor, but that is a manageable operation
EBMCORE_IMPORT_EXPORT FractionalDataType * EBMCORE_CALLING_CONVENTION GenerateModelFeatureCombinationUpdate(PEbmTraining ebmTraining, IntegerDataType indexFeatureCombination, FractionalDataType learningRate, IntegerDataType countTreeSplitsMax, IntegerDataType countCasesRequiredForSplitParentMin, const FractionalDataType * trainingWeights, const FractionalDataType * validationWeights, FractionalDataType * gainReturn) {
   LOG_COUNTED(&g_cLogGenerateModelFeatureCombinationUpdateParametersMessages, TraceLevelInfo, TraceLevelVerbose, "GenerateModelFeatureCombinationUpdate parameters: ebmTraining=%p, indexFeatureCombination=%" IntegerDataTypePrintf ", learningRate=%" FractionalDataTypePrintf ", countTreeSplitsMax=%" IntegerDataTypePrintf ", countCasesRequiredForSplitParentMin=%" IntegerDataTypePrintf ", trainingWeights=%p, validationWeights=%p, gainReturn=%p", static_cast<void *>(ebmTraining), indexFeatureCombination, learningRate, countTreeSplitsMax, countCasesRequiredForSplitParentMin, static_cast<const void *>(trainingWeights), static_cast<const void *>(validationWeights), static_cast<void *>(gainReturn));

   TmlState * pTmlState = reinterpret_cast<TmlState *>(ebmTraining);
   EBM_ASSERT(nullptr != pTmlState);

   EBM_ASSERT(0 <= indexFeatureCombination);
   EBM_ASSERT((IsNumberConvertable<size_t, IntegerDataType>(indexFeatureCombination))); // we wouldn't have allowed the creation of an feature set larger than size_t
   size_t iFeatureCombination = static_cast<size_t>(indexFeatureCombination);
   EBM_ASSERT(iFeatureCombination < pTmlState->m_cFeatureCombinations);
   EBM_ASSERT(nullptr != pTmlState->m_apFeatureCombinations); // this is true because 0 < pTmlState->m_cFeatureCombinations since our caller needs to pass in a valid indexFeatureCombination to this function

   LOG_COUNTED(&pTmlState->m_apFeatureCombinations[iFeatureCombination]->m_cLogEnterGenerateModelFeatureCombinationUpdateMessages, TraceLevelInfo, TraceLevelVerbose, "Entered GenerateModelFeatureCombinationUpdate");

   EBM_ASSERT(!std::isnan(learningRate));
   EBM_ASSERT(!std::isinf(learningRate));

   EBM_ASSERT(0 <= countTreeSplitsMax);
   size_t cTreeSplitsMax = static_cast<size_t>(countTreeSplitsMax);
   if(!IsNumberConvertable<size_t, IntegerDataType>(countTreeSplitsMax)) {
      // we can never exceed a size_t number of splits, so let's just set it to the maximum if we were going to overflow because it will generate the same results as if we used the true number
      cTreeSplitsMax = std::numeric_limits<size_t>::max();
   }

   EBM_ASSERT(0 <= countCasesRequiredForSplitParentMin); // if there is 1 case, then it can't be split, but we accept this input from our user
   size_t cCasesRequiredForSplitParentMin = static_cast<size_t>(countCasesRequiredForSplitParentMin);
   if(!IsNumberConvertable<size_t, IntegerDataType>(countCasesRequiredForSplitParentMin)) {
      // we can never exceed a size_t number of cases, so let's just set it to the maximum if we were going to overflow because it will generate the same results as if we used the true number
      cCasesRequiredForSplitParentMin = std::numeric_limits<size_t>::max();
   }

   EBM_ASSERT(nullptr == trainingWeights); // TODO : implement this later
   EBM_ASSERT(nullptr == validationWeights); // TODO : implement this later
   // validationMetricReturn can be nullptr

   FractionalDataType * aModelUpdateTensor;
   if(pTmlState->m_bRegression) {
      aModelUpdateTensor = GenerateModelFeatureCombinationUpdatePerTargetStates<k_Regression>(pTmlState, iFeatureCombination, learningRate, cTreeSplitsMax, cCasesRequiredForSplitParentMin, trainingWeights, validationWeights, gainReturn);
   } else {
      const size_t cTargetStates = pTmlState->m_cTargetStates;
      if(cTargetStates <= 1) {
         // if there is only 1 target state for classification, then we can predict the output with 100% accuracy.  The model is a tensor with zero length array logits, which means for our representation that we have zero items in the array total.
         // since we can predit the output with 100% accuracy, our gain will be 0.
         if(nullptr != gainReturn) {
            *gainReturn = 0;
         }
         LOG(TraceLevelWarning, "WARNING GenerateModelFeatureCombinationUpdate cTargetStates <= 1");
         return nullptr;
      }
      aModelUpdateTensor = CompilerRecursiveGenerateModelFeatureCombinationUpdate<2>(cTargetStates, pTmlState, iFeatureCombination, learningRate, cTreeSplitsMax, cCasesRequiredForSplitParentMin, trainingWeights, validationWeights, gainReturn);
   }

   if(nullptr != gainReturn) {
      EBM_ASSERT(*gainReturn <= 0.000000001);
      LOG_COUNTED(&pTmlState->m_apFeatureCombinations[iFeatureCombination]->m_cLogExitGenerateModelFeatureCombinationUpdateMessages, TraceLevelInfo, TraceLevelVerbose, "Exited GenerateModelFeatureCombinationUpdate %" FractionalDataTypePrintf, *gainReturn);
   } else {
      LOG_COUNTED(&pTmlState->m_apFeatureCombinations[iFeatureCombination]->m_cLogExitGenerateModelFeatureCombinationUpdateMessages, TraceLevelInfo, TraceLevelVerbose, "Exited GenerateModelFeatureCombinationUpdate no gain");
   }
   if(nullptr == aModelUpdateTensor) {
      LOG(TraceLevelWarning, "WARNING GenerateModelFeatureCombinationUpdate returned nullptr");
   }
   return aModelUpdateTensor;
}

// a*PredictionScores = logOdds for binary classification
// a*PredictionScores = logWeights for multiclass classification
// a*PredictionScores = predictedValue for regression
template<ptrdiff_t countCompilerClassificationTargetStates>
static IntegerDataType ApplyModelFeatureCombinationUpdatePerTargetStates(TmlState * const pTmlState, const size_t iFeatureCombination, const FractionalDataType * const aModelUpdateTensor, FractionalDataType * const pValidationMetricReturn) {
   LOG(TraceLevelVerbose, "Entered ApplyModelFeatureCombinationUpdatePerTargetStates");

   EBM_ASSERT(nullptr != pTmlState->m_apCurrentModel); // m_apCurrentModel can be null if there are no featureCombinations (but we have an feature combination index), or if the target has 1 or 0 states (which we check before calling this function), so it shouldn't be possible to be null
   EBM_ASSERT(nullptr != pTmlState->m_apBestModel); // m_apCurrentModel can be null if there are no featureCombinations (but we have an feature combination index), or if the target has 1 or 0 states (which we check before calling this function), so it shouldn't be possible to be null
   EBM_ASSERT(nullptr != aModelUpdateTensor); // aModelUpdateTensor is checked for nullptr before calling this function   

   pTmlState->m_apCurrentModel[iFeatureCombination]->AddExpanded(aModelUpdateTensor);

   const FeatureCombinationCore * const pFeatureCombination = pTmlState->m_apFeatureCombinations[iFeatureCombination];

   // if the count of training cases is zero, then pTmlState->m_pTrainingSet will be nullptr
   if(nullptr != pTmlState->m_pTrainingSet) {
      // TODO : move the target bits branch inside TrainingSetInputFeatureLoop to here outside instead of the feature combination.  The target # of bits is extremely predictable and so we get to only process one sub branch of code below that.  If we do feature combinations here then we have to keep in instruction cache a whole bunch of options
      TrainingSetInputFeatureLoop<1, countCompilerClassificationTargetStates>(pFeatureCombination, pTmlState->m_pTrainingSet, aModelUpdateTensor, pTmlState->m_cTargetStates);
   }

   FractionalDataType modelMetric = 0;
   if(nullptr != pTmlState->m_pValidationSet) {
      // if there is no validation set, it's pretty hard to know what the metric we'll get for our validation set
      // we could in theory return anything from zero to infinity or possibly, NaN (probably legally the best), but we return 0 here
      // because we want to kick our caller out of any loop it might be calling us in.  Infinity and NaN are odd values that might cause problems in
      // a caller that isn't expecting those values, so 0 is the safest option, and our caller can avoid the situation entirely by not calling
      // us with zero count validation sets

      // if the count of validation set is zero, then pTmlState->m_pValidationSet will be nullptr
      // if the count of training cases is zero, don't update the best model (it will stay as all zeros), and we don't need to update our non-existant training set either
      // C++ doesn't define what happens when you compare NaN to annother number.  It probably follows IEEE 754, but it isn't guaranteed, so let's check for zero cases in the validation set this better way   https://stackoverflow.com/questions/31225264/what-is-the-result-of-comparing-a-number-with-nan

      // TODO : move the target bits branch inside TrainingSetInputFeatureLoop to here outside instead of the feature combination.  The target # of bits is extremely predictable and so we get to only process one sub branch of code below that.  If we do feature combinations here then we have to keep in instruction cache a whole bunch of options

      modelMetric = ValidationSetInputFeatureLoop<1, countCompilerClassificationTargetStates>(pFeatureCombination, pTmlState->m_pValidationSet, aModelUpdateTensor, pTmlState->m_cTargetStates);

      // modelMetric is either logloss (classification) or rmse (regression).  In either case we want to minimize it.
      if(LIKELY(modelMetric < pTmlState->m_bestModelMetric)) {
         // we keep on improving, so this is more likely than not, and we'll exit if it becomes negative a lot
         pTmlState->m_bestModelMetric = modelMetric;

         // TODO : in the future don't copy over all SegmentedRegions.  We only need to copy the ones that changed, which we can detect if we use a linked list and array lookup for the same data structure
         size_t iModel = 0;
         size_t iModelEnd = pTmlState->m_cFeatureCombinations;
         do {
            if(pTmlState->m_apBestModel[iModel]->Copy(*pTmlState->m_apCurrentModel[iModel])) {
               if(nullptr != pValidationMetricReturn) {
                  *pValidationMetricReturn = 0; // on error set it to something instead of random bits
               }
               LOG(TraceLevelVerbose, "Exited ApplyModelFeatureCombinationUpdatePerTargetStates with memory allocation error in copy");
               return 1;
            }
            ++iModel;
         } while(iModel != iModelEnd);
      }
   }
   if(nullptr != pValidationMetricReturn) {
      *pValidationMetricReturn = modelMetric;
   }

   LOG(TraceLevelVerbose, "Exited ApplyModelFeatureCombinationUpdatePerTargetStates");
   return 0;
}

template<ptrdiff_t iPossibleCompilerOptimizedTargetStates>
TML_INLINE IntegerDataType CompilerRecursiveApplyModelFeatureCombinationUpdate(const size_t cRuntimeTargetStates, TmlState * const pTmlState, const size_t iFeatureCombination, const FractionalDataType * const aModelUpdateTensor, FractionalDataType * const pValidationMetricReturn) {
   EBM_ASSERT(IsClassification(iPossibleCompilerOptimizedTargetStates));
   if(iPossibleCompilerOptimizedTargetStates == cRuntimeTargetStates) {
      EBM_ASSERT(cRuntimeTargetStates <= k_cCompilerOptimizedTargetStatesMax);
      return ApplyModelFeatureCombinationUpdatePerTargetStates<iPossibleCompilerOptimizedTargetStates>(pTmlState, iFeatureCombination, aModelUpdateTensor, pValidationMetricReturn);
   } else {
      return CompilerRecursiveApplyModelFeatureCombinationUpdate<iPossibleCompilerOptimizedTargetStates + 1>(cRuntimeTargetStates, pTmlState, iFeatureCombination, aModelUpdateTensor, pValidationMetricReturn);
   }
}

template<>
TML_INLINE IntegerDataType CompilerRecursiveApplyModelFeatureCombinationUpdate<k_cCompilerOptimizedTargetStatesMax + 1>(const size_t cRuntimeTargetStates, TmlState * const pTmlState, const size_t iFeatureCombination, const FractionalDataType * const aModelUpdateTensor, FractionalDataType * const pValidationMetricReturn) {
   UNUSED(cRuntimeTargetStates);
   // it is logically possible, but uninteresting to have a classification with 1 target state, so let our runtime system handle those unlikley and uninteresting cases
   EBM_ASSERT(k_cCompilerOptimizedTargetStatesMax < cRuntimeTargetStates);
   return ApplyModelFeatureCombinationUpdatePerTargetStates<k_DynamicClassification>(pTmlState, iFeatureCombination, aModelUpdateTensor, pValidationMetricReturn);
}

// we made this a global because if we had put this variable inside the EbmTrainingState object, then we would need to dereference that before getting the count.  By making this global we can send a log message incase a bad EbmTrainingState object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more times than desired, but we can live with that
static unsigned int g_cLogApplyModelFeatureCombinationUpdateParametersMessages = 10;

EBMCORE_IMPORT_EXPORT IntegerDataType EBMCORE_CALLING_CONVENTION ApplyModelFeatureCombinationUpdate(PEbmTraining ebmTraining, IntegerDataType indexFeatureCombination, const FractionalDataType * modelUpdateTensor, FractionalDataType * validationMetricReturn) {
   LOG_COUNTED(&g_cLogApplyModelFeatureCombinationUpdateParametersMessages, TraceLevelInfo, TraceLevelVerbose, "ApplyModelFeatureCombinationUpdate parameters: ebmTraining=%p, indexFeatureCombination=%" IntegerDataTypePrintf ", modelUpdateTensor=%p, validationMetricReturn=%p", static_cast<void *>(ebmTraining), indexFeatureCombination, static_cast<const void *>(modelUpdateTensor), static_cast<void *>(validationMetricReturn));

   TmlState * pTmlState = reinterpret_cast<TmlState *>(ebmTraining);
   EBM_ASSERT(nullptr != pTmlState);

   EBM_ASSERT(0 <= indexFeatureCombination);
   EBM_ASSERT((IsNumberConvertable<size_t, IntegerDataType>(indexFeatureCombination))); // we wouldn't have allowed the creation of an feature set larger than size_t
   size_t iFeatureCombination = static_cast<size_t>(indexFeatureCombination);
   EBM_ASSERT(iFeatureCombination < pTmlState->m_cFeatureCombinations);
   EBM_ASSERT(nullptr != pTmlState->m_apFeatureCombinations); // this is true because 0 < pTmlState->m_cFeatureCombinations since our caller needs to pass in a valid indexFeatureCombination to this function

   LOG_COUNTED(&pTmlState->m_apFeatureCombinations[iFeatureCombination]->m_cLogEnterApplyModelFeatureCombinationUpdateMessages, TraceLevelInfo, TraceLevelVerbose, "Entered ApplyModelFeatureCombinationUpdate");

   // modelUpdateTensor can be nullptr (then nothing gets updated)
   // validationMetricReturn can be nullptr

   if(nullptr == modelUpdateTensor) {
      if(nullptr != validationMetricReturn) {
         *validationMetricReturn = 0;
      }
      LOG_COUNTED(&pTmlState->m_apFeatureCombinations[iFeatureCombination]->m_cLogExitApplyModelFeatureCombinationUpdateMessages, TraceLevelInfo, TraceLevelVerbose, "Exited ApplyModelFeatureCombinationUpdate from null modelUpdateTensor");
      return 0;
   }

   IntegerDataType ret;
   if(pTmlState->m_bRegression) {
      ret = ApplyModelFeatureCombinationUpdatePerTargetStates<k_Regression>(pTmlState, iFeatureCombination, modelUpdateTensor, validationMetricReturn);
   } else {
      const size_t cTargetStates = pTmlState->m_cTargetStates;
      if(cTargetStates <= 1) {
         // if there is only 1 target state for classification, then we can predict the output with 100% accuracy.  The model is a tensor with zero length array logits, which means for our representation that we have zero items in the array total.
         // since we can predit the output with 100% accuracy, our log loss is 0.
         if(nullptr != validationMetricReturn) {
            *validationMetricReturn = 0;
         }
         LOG_COUNTED(&pTmlState->m_apFeatureCombinations[iFeatureCombination]->m_cLogExitApplyModelFeatureCombinationUpdateMessages, TraceLevelInfo, TraceLevelVerbose, "Exited ApplyModelFeatureCombinationUpdate from cTargetStates <= 1");
         return 0;
      }
      ret = CompilerRecursiveApplyModelFeatureCombinationUpdate<2>(cTargetStates, pTmlState, iFeatureCombination, modelUpdateTensor, validationMetricReturn);
   }
   if(0 != ret) {
      LOG(TraceLevelWarning, "WARNING ApplyModelFeatureCombinationUpdate returned %" IntegerDataTypePrintf, ret);
   }
   if(nullptr != validationMetricReturn) {
      EBM_ASSERT(0 <= *validationMetricReturn); // both log loss and RMSE need to be above zero
      LOG_COUNTED(&pTmlState->m_apFeatureCombinations[iFeatureCombination]->m_cLogExitApplyModelFeatureCombinationUpdateMessages, TraceLevelInfo, TraceLevelVerbose, "Exited ApplyModelFeatureCombinationUpdate %" FractionalDataTypePrintf, *validationMetricReturn);
   } else {
      LOG_COUNTED(&pTmlState->m_apFeatureCombinations[iFeatureCombination]->m_cLogExitApplyModelFeatureCombinationUpdateMessages, TraceLevelInfo, TraceLevelVerbose, "Exited ApplyModelFeatureCombinationUpdate.  No validation pointer.");
   }
   return ret;
}

EBMCORE_IMPORT_EXPORT IntegerDataType EBMCORE_CALLING_CONVENTION TrainingStep(PEbmTraining ebmTraining, IntegerDataType indexFeatureCombination, FractionalDataType learningRate, IntegerDataType countTreeSplitsMax, IntegerDataType countCasesRequiredForSplitParentMin, const FractionalDataType * trainingWeights, const FractionalDataType * validationWeights, FractionalDataType * validationMetricReturn) {
   TmlState * pTmlState = reinterpret_cast<TmlState *>(ebmTraining);
   EBM_ASSERT(nullptr != pTmlState);

   if(!pTmlState->m_bRegression) {
      // we need to special handle this case because if we call GenerateModelUpdate, we'll get back a nullptr for the model (since there is no model) and we'll return 1 from this function.  We'd like to return 0 (success) here, so we handle it ourselves
      const size_t cTargetStates = pTmlState->m_cTargetStates;
      if(cTargetStates <= 1) {
         // if there is only 1 target state for classification, then we can predict the output with 100% accuracy.  The model is a tensor with zero length array logits, which means for our representation that we have zero items in the array total.
         // since we can predit the output with 100% accuracy, our gain will be 0.
         if(nullptr != validationMetricReturn) {
            *validationMetricReturn = 0;
         }
         LOG(TraceLevelWarning, "WARNING TrainingStep cTargetStates <= 1");
         return 0;
      }
   }

   FractionalDataType gain; // we toss this value, but we still need to get it
   FractionalDataType * pModelUpdateTensor = GenerateModelFeatureCombinationUpdate(ebmTraining, indexFeatureCombination, learningRate, countTreeSplitsMax, countCasesRequiredForSplitParentMin, trainingWeights, validationWeights, &gain);
   if(nullptr == pModelUpdateTensor) {
      EBM_ASSERT(nullptr == validationMetricReturn || 0 == *validationMetricReturn); // rely on GenerateModelUpdate to set the validationMetricReturn to zero on error
      return 1;
   }
   return ApplyModelFeatureCombinationUpdate(ebmTraining, indexFeatureCombination, pModelUpdateTensor, validationMetricReturn);
}

EBMCORE_IMPORT_EXPORT FractionalDataType * EBMCORE_CALLING_CONVENTION GetCurrentModelFeatureCombination(PEbmTraining ebmTraining, IntegerDataType indexFeatureCombination) {
   LOG(TraceLevelInfo, "Entered GetCurrentModelFeatureCombination: ebmTraining=%p, indexFeatureCombination=%" IntegerDataTypePrintf, static_cast<void *>(ebmTraining), indexFeatureCombination);

   TmlState * pTmlState = reinterpret_cast<TmlState *>(ebmTraining);
   EBM_ASSERT(nullptr != pTmlState);
   EBM_ASSERT(0 <= indexFeatureCombination);
   EBM_ASSERT((IsNumberConvertable<size_t, IntegerDataType>(indexFeatureCombination))); // we wouldn't have allowed the creation of an feature set larger than size_t
   size_t iFeatureCombination = static_cast<size_t>(indexFeatureCombination);
   EBM_ASSERT(iFeatureCombination < pTmlState->m_cFeatureCombinations);

   if(nullptr == pTmlState->m_apCurrentModel) {
      // if pTmlState->m_apCurrentModel is nullptr, then either:
      //    1) m_cFeatureCombinations was 0, in which case this function would have undefined behavior since the caller needs to indicate a valid indexFeatureCombination, which is impossible, so we can do anything we like, include the below actions.
      //    2) m_cTargetStates was either 1 or 0 (and the learning type is classification), which is legal, which we need to handle here
      // for classification, if there is only 1 possible target state, then the probability of that state is 100%.  If there were logits in this model, they'd all be infinity, but you could alternatively think of this model as having zero logits, since the number of logits can be one less than the number of target cases.  A model with zero logits is empty, and has zero items.  We want to return a tensor with 0 items in it, so we could either return a pointer to some random memory that can't be accessed, or we can return nullptr.  We return a nullptr in the hopes that our caller will either handle it or throw a nicer exception.
      return nullptr;
   }

   SegmentedRegionCore<ActiveDataType, FractionalDataType> * pCurrentModel = pTmlState->m_apCurrentModel[iFeatureCombination];
   EBM_ASSERT(pCurrentModel->m_bExpanded); // the model should have been expanded at startup
   FractionalDataType * pRet = pCurrentModel->GetValuePointer();

   LOG(TraceLevelInfo, "Exited GetCurrentModelFeatureCombination %p", static_cast<void *>(pRet));
   return pRet;
}

EBMCORE_IMPORT_EXPORT FractionalDataType * EBMCORE_CALLING_CONVENTION GetBestModelFeatureCombination(PEbmTraining ebmTraining, IntegerDataType indexFeatureCombination) {
   LOG(TraceLevelInfo, "Entered GetBestModelFeatureCombination: ebmTraining=%p, indexFeatureCombination=%" IntegerDataTypePrintf, static_cast<void *>(ebmTraining), indexFeatureCombination);

   TmlState * pTmlState = reinterpret_cast<TmlState *>(ebmTraining);
   EBM_ASSERT(nullptr != pTmlState);
   EBM_ASSERT(0 <= indexFeatureCombination);
   EBM_ASSERT((IsNumberConvertable<size_t, IntegerDataType>(indexFeatureCombination))); // we wouldn't have allowed the creation of an feature set larger than size_t
   size_t iFeatureCombination = static_cast<size_t>(indexFeatureCombination);
   EBM_ASSERT(iFeatureCombination < pTmlState->m_cFeatureCombinations);

   if(nullptr == pTmlState->m_apBestModel) {
      // if pTmlState->m_apBestModel is nullptr, then either:
      //    1) m_cFeatureCombinations was 0, in which case this function would have undefined behavior since the caller needs to indicate a valid indexFeatureCombination, which is impossible, so we can do anything we like, include the below actions.
      //    2) m_cTargetStates was either 1 or 0 (and the learning type is classification), which is legal, which we need to handle here
      // for classification, if there is only 1 possible target state, then the probability of that state is 100%.  If there were logits in this model, they'd all be infinity, but you could alternatively think of this model as having zero logits, since the number of logits can be one less than the number of target cases.  A model with zero logits is empty, and has zero items.  We want to return a tensor with 0 items in it, so we could either return a pointer to some random memory that can't be accessed, or we can return nullptr.  We return a nullptr in the hopes that our caller will either handle it or throw a nicer exception.
      return nullptr;
   }

   SegmentedRegionCore<ActiveDataType, FractionalDataType> * pBestModel = pTmlState->m_apBestModel[iFeatureCombination];
   EBM_ASSERT(pBestModel->m_bExpanded); // the model should have been expanded at startup
   FractionalDataType * pRet = pBestModel->GetValuePointer();

   LOG(TraceLevelInfo, "Exited GetBestModelFeatureCombination %p", static_cast<void *>(pRet));
   return pRet;
}

EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION FreeTraining(PEbmTraining ebmTraining) {
   LOG(TraceLevelInfo, "Entered FreeTraining: ebmTraining=%p", static_cast<void *>(ebmTraining));
   TmlState * pTmlState = reinterpret_cast<TmlState *>(ebmTraining);
   EBM_ASSERT(nullptr != pTmlState);
   delete pTmlState;
   LOG(TraceLevelInfo, "Exited FreeTraining");
}
