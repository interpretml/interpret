// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // FeatureType
#include "Logging.h" // EBM_ASSERT & LOG
#include "Feature.h"
#include "FeatureGroup.h"
#include "DataSetBoosting.h"

EBM_INLINE static FloatEbmType * ConstructResidualErrors(const size_t cInstances, const size_t cVectorLength) {
   LOG_0(TraceLevelInfo, "Entered DataSetByFeatureCombination::ConstructResidualErrors");

   EBM_ASSERT(1 <= cInstances);
   EBM_ASSERT(1 <= cVectorLength);

   if(IsMultiplyError(cInstances, cVectorLength)) {
      LOG_0(TraceLevelWarning, "WARNING DataSetByFeatureCombination::ConstructResidualErrors IsMultiplyError(cInstances, cVectorLength)");
      return nullptr;
   }

   const size_t cElements = cInstances * cVectorLength;
   FloatEbmType * aResidualErrors = EbmMalloc<FloatEbmType>(cElements);

   LOG_0(TraceLevelInfo, "Exited DataSetByFeatureCombination::ConstructResidualErrors");
   return aResidualErrors;
}

INLINE_RELEASE static FloatEbmType * ConstructPredictorScores(
   const size_t cInstances, 
   const size_t cVectorLength, 
   const FloatEbmType * const aPredictorScoresFrom
) {
   LOG_0(TraceLevelInfo, "Entered DataSetByFeatureCombination::ConstructPredictorScores");

   EBM_ASSERT(0 < cInstances);
   EBM_ASSERT(0 < cVectorLength);
   EBM_ASSERT(nullptr != aPredictorScoresFrom);

   if(IsMultiplyError(cInstances, cVectorLength)) {
      LOG_0(TraceLevelWarning, "WARNING DataSetByFeatureCombination::ConstructPredictorScores IsMultiplyError(cInstances, cVectorLength)");
      return nullptr;
   }

   const size_t cElements = cInstances * cVectorLength;
   FloatEbmType * const aPredictorScoresTo = EbmMalloc<FloatEbmType>(cElements);
   if(nullptr == aPredictorScoresTo) {
      LOG_0(TraceLevelWarning, "WARNING DataSetByFeatureCombination::ConstructPredictorScores nullptr == aPredictorScoresTo");
      return nullptr;
   }

   const size_t cBytes = sizeof(FloatEbmType) * cElements;
   // if there are any NaN or +- infinity values we should just propagate them and exit during boosting
   memcpy(aPredictorScoresTo, aPredictorScoresFrom, cBytes);
   constexpr bool bZeroingLogits = 0 <= k_iZeroClassificationLogitAtInitialize;
   if(bZeroingLogits) {
      // TODO : integrate this subtraction into the copy instead of doing it afterwards
      FloatEbmType * pScore = aPredictorScoresTo;
      const FloatEbmType * const pScoreExteriorEnd = pScore + cVectorLength * cInstances;
      do {
         FloatEbmType scoreShift = pScore[k_iZeroClassificationLogitAtInitialize];
         const FloatEbmType * const pScoreInteriorEnd = pScore + cVectorLength;
         do {
            *pScore -= scoreShift;
            ++pScore;
         } while(pScoreInteriorEnd != pScore);
      } while(pScoreExteriorEnd != pScore);
   }

   LOG_0(TraceLevelInfo, "Exited DataSetByFeatureCombination::ConstructPredictorScores");
   return aPredictorScoresTo;
}

EBM_INLINE static StorageDataType * ConstructTargetData(
   const size_t cInstances, 
   const IntEbmType * const aTargets, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
) {
   LOG_0(TraceLevelInfo, "Entered DataSetByFeatureCombination::ConstructTargetData");

   EBM_ASSERT(0 < cInstances);
   EBM_ASSERT(nullptr != aTargets);
   EBM_ASSERT(1 <= runtimeLearningTypeOrCountTargetClasses); // this should be classification
   const size_t countTargetClasses = static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses);

   StorageDataType * const aTargetData = EbmMalloc<StorageDataType>(cInstances);
   if(nullptr == aTargetData) {
      LOG_0(TraceLevelWarning, "WARNING nullptr == aTargetData");
      return nullptr;
   }

   const IntEbmType * pTargetFrom = aTargets;
   const IntEbmType * const pTargetFromEnd = aTargets + cInstances;
   StorageDataType * pTargetTo = aTargetData;
   do {
      const IntEbmType data = *pTargetFrom;
      if(data < 0) {
         LOG_0(TraceLevelError, "ERROR DataSetByFeatureCombination::ConstructTargetData target value cannot be negative");
         free(aTargetData);
         return nullptr;
      }
      if(!IsNumberConvertable<StorageDataType, IntEbmType>(data)) {
         // this shouldn't be possible since we previously checked that we could convert our target,
         // so if this is failing then we'll be larger than the maximum number of classes
         LOG_0(TraceLevelError, "ERROR DataSetByFeatureCombination::ConstructTargetData data target too big to reference memory");
         free(aTargetData);
         return nullptr;
      }
      if(!IsNumberConvertable<size_t, IntEbmType>(data)) {
         // this shouldn't be possible since we previously checked that we could convert our target,
         // so if this is failing then we'll be larger than the maximum number of classes
         LOG_0(TraceLevelError, "ERROR DataSetByFeatureCombination::ConstructTargetData data target too big to reference memory");
         free(aTargetData);
         return nullptr;
      }
      const StorageDataType iData = static_cast<StorageDataType>(data);
      if(countTargetClasses <= static_cast<size_t>(iData)) {
         LOG_0(TraceLevelError, "ERROR DataSetByFeatureCombination::ConstructTargetData target value larger than number of classes");
         free(aTargetData);
         return nullptr;
      }
      *pTargetTo = iData;
      ++pTargetTo;
      ++pTargetFrom;
   } while(pTargetFromEnd != pTargetFrom);

   LOG_0(TraceLevelInfo, "Exited DataSetByFeatureCombination::ConstructTargetData");
   return aTargetData;
}

struct InputDataPointerAndCountBins {
   const IntEbmType * m_pInputData;
   size_t m_cBins;
};

EBM_INLINE static StorageDataType * * ConstructInputData(
   const size_t cFeatureCombinations, 
   const FeatureCombination * const * const apFeatureCombination, 
   const size_t cInstances, 
   const IntEbmType * const aInputDataFrom
) {
   LOG_0(TraceLevelInfo, "Entered DataSetByFeatureCombination::ConstructInputData");

   EBM_ASSERT(0 < cFeatureCombinations);
   EBM_ASSERT(nullptr != apFeatureCombination);
   EBM_ASSERT(0 < cInstances);
   // aInputDataFrom can be nullptr EVEN if 0 < cFeatureCombinations && 0 < cInstances IF the featureCombinations are all empty, 
   // which makes none of them refer to features, so the aInputDataFrom pointer isn't necessary

   StorageDataType ** const aaInputDataTo = EbmMalloc<StorageDataType *>(cFeatureCombinations);
   if(nullptr == aaInputDataTo) {
      LOG_0(TraceLevelWarning, "WARNING DataSetByFeatureCombination::ConstructInputData nullptr == aaInputDataTo");
      return nullptr;
   }

   StorageDataType ** paInputDataTo = aaInputDataTo;
   const FeatureCombination * const * ppFeatureCombination = apFeatureCombination;
   const FeatureCombination * const * const ppFeatureCombinationEnd = apFeatureCombination + cFeatureCombinations;
   do {
      const FeatureCombination * const pFeatureCombination = *ppFeatureCombination;
      EBM_ASSERT(nullptr != pFeatureCombination);
      const size_t cFeatures = pFeatureCombination->GetCountFeatures();
      if(0 == cFeatures) {
         *paInputDataTo = nullptr; // free will skip over these later
         ++paInputDataTo;
      } else {
         const size_t cItemsPerBitPackedDataUnit = pFeatureCombination->GetCountItemsPerBitPackedDataUnit();
         // for a 32/64 bit storage item, we can't have more than 32/64 bit packed items stored
         EBM_ASSERT(cItemsPerBitPackedDataUnit <= CountBitsRequiredPositiveMax<StorageDataType>());
         const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackedDataUnit);
         // if we have 1 item, it can't be larger than the number of bits of storage
         EBM_ASSERT(cBitsPerItemMax <= CountBitsRequiredPositiveMax<StorageDataType>());

         EBM_ASSERT(0 < cInstances);
         const size_t cDataUnits = (cInstances - 1) / cItemsPerBitPackedDataUnit + 1; // this can't overflow or underflow

         StorageDataType * pInputDataTo = EbmMalloc<StorageDataType>(cDataUnits);
         if(nullptr == pInputDataTo) {
            LOG_0(TraceLevelWarning, "WARNING DataSetByFeatureCombination::ConstructInputData nullptr == pInputDataTo");
            goto free_all;
         }
         *paInputDataTo = pInputDataTo;
         ++paInputDataTo;

         const size_t cBytesData = sizeof(StorageDataType) * cDataUnits;
         // stop on the last item in our array AND then do one special last loop with less or equal iterations to the normal loop
         const StorageDataType * const pInputDataToLast = 
            reinterpret_cast<const StorageDataType *>(reinterpret_cast<const char *>(pInputDataTo) + cBytesData) - 1;
         EBM_ASSERT(pInputDataTo <= pInputDataToLast); // we have 1 item or more, and therefore the last one can't be before the first item

         EBM_ASSERT(nullptr != aInputDataFrom);

         const FeatureCombinationEntry * pFeatureCombinationEntry = pFeatureCombination->GetFeatureCombinationEntries();
         InputDataPointerAndCountBins dimensionInfo[k_cDimensionsMax];
         InputDataPointerAndCountBins * pDimensionInfo = &dimensionInfo[0];
         EBM_ASSERT(0 < cFeatures);
         const InputDataPointerAndCountBins * const pDimensionInfoEnd = &dimensionInfo[cFeatures];
         do {
            const Feature * const pFeature = pFeatureCombinationEntry->m_pFeature;
            pDimensionInfo->m_pInputData = &aInputDataFrom[pFeature->GetIndexFeatureData() * cInstances];
            pDimensionInfo->m_cBins = pFeature->GetCountBins();
            ++pFeatureCombinationEntry;
            ++pDimensionInfo;
         } while(pDimensionInfoEnd != pDimensionInfo);

         // THIS IS NOT A CONSTANT FOR A REASON.. WE CHANGE IT ON OUR LAST ITERATION
         // if we ever template this function on cItemsPerBitPackedDataUnit, then we'd want
         // to make this a constant so that the compiler could reason about it an eliminate loops
         // as it is, it isn't a constant, so the compiler would not be able to figure out that most
         // of the time it is a constant
         size_t shiftEnd = cBitsPerItemMax * cItemsPerBitPackedDataUnit;
         while(pInputDataTo < pInputDataToLast) /* do the last iteration AFTER we re-enter this loop through the goto label! */ {
         one_last_loop:;
            EBM_ASSERT(shiftEnd <= CountBitsRequiredPositiveMax<StorageDataType>());

            size_t bits = 0;
            size_t shift = 0;
            do {
               size_t tensorMultiple = 1;
               size_t tensorIndex = 0;
               pDimensionInfo = &dimensionInfo[0];
               do {
                  const IntEbmType * pInputData = pDimensionInfo->m_pInputData;
                  const IntEbmType inputData = *pInputData;
                  pDimensionInfo->m_pInputData = pInputData + 1;
                  if(inputData < 0) {
                     LOG_0(TraceLevelError, "ERROR DataSetByFeatureCombination::ConstructInputData inputData value cannot be negative");
                     goto free_all;
                  }
                  if(!IsNumberConvertable<size_t, IntEbmType>(inputData)) {
                     LOG_0(TraceLevelError, "ERROR DataSetByFeatureCombination::ConstructInputData inputData value too big to reference memory");
                     goto free_all;
                  }
                  const size_t iData = static_cast<size_t>(inputData);

                  if(pDimensionInfo->m_cBins <= iData) {
                     LOG_0(TraceLevelError, "ERROR DataSetByFeatureCombination::ConstructInputData iData value must be less than the number of bins");
                     goto free_all;
                  }
                  // we check for overflows during FeatureCombination construction, but let's check here again
                  EBM_ASSERT(!IsMultiplyError(tensorMultiple, pDimensionInfo->m_cBins));

                  // this can't overflow if the multiplication below doesn't overflow, and we checked for that above
                  tensorIndex += tensorMultiple * iData;
                  tensorMultiple *= pDimensionInfo->m_cBins;

                  ++pDimensionInfo;
               } while(pDimensionInfoEnd != pDimensionInfo);
               // put our first item in the least significant bits.  We do this so that later when
               // unpacking the indexes, we can just AND our mask with the bitfield to get the index and in subsequent loops
               // we can just shift down.  This eliminates one extra shift that we'd otherwise need to make if the first
               // item was in the MSB
               EBM_ASSERT(shift < CountBitsRequiredPositiveMax<StorageDataType>());
               bits |= tensorIndex << shift;
               shift += cBitsPerItemMax;
            } while(shiftEnd != shift);
            EBM_ASSERT((IsNumberConvertable<StorageDataType, size_t>(bits)));
            *pInputDataTo = static_cast<StorageDataType>(bits);
            ++pInputDataTo;
         }

         if(pInputDataTo == pInputDataToLast) {
            // if this is the first time we've exited the loop, then re-enter it to do our last loop, but reduce the number of times we do the inner loop
            shiftEnd = cBitsPerItemMax * ((cInstances - 1) % cItemsPerBitPackedDataUnit + 1);
            goto one_last_loop;
         }
      }
      ++ppFeatureCombination;
   } while(ppFeatureCombinationEnd != ppFeatureCombination);

   LOG_0(TraceLevelInfo, "Exited DataSetByFeatureCombination::ConstructInputData");
   return aaInputDataTo;

free_all:
   while(aaInputDataTo != paInputDataTo) {
      --paInputDataTo;
      free(*paInputDataTo);
   }
   free(aaInputDataTo);
   return nullptr;
}

bool DataSetByFeatureCombination::Initialize(
   const bool bAllocateResidualErrors, 
   const bool bAllocatePredictorScores, 
   const bool bAllocateTargetData, 
   const size_t cFeatureCombinations, 
   const FeatureCombination * const * const apFeatureCombination, 
   const size_t cInstances, 
   const IntEbmType * const aInputDataFrom, 
   const void * const aTargets, 
   const FloatEbmType * const aPredictorScoresFrom, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
) {
   EBM_ASSERT(nullptr == m_aResidualErrors);
   EBM_ASSERT(nullptr == m_aPredictorScores);
   EBM_ASSERT(nullptr == m_aTargetData);
   EBM_ASSERT(nullptr == m_aaInputData);

   LOG_0(TraceLevelInfo, "Entered DataSetByFeatureCombination::Initialize");
   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   if(0 != cInstances) {
      FloatEbmType * aResidualErrors = nullptr;
      if(bAllocateResidualErrors) {
         aResidualErrors = ConstructResidualErrors(cInstances, cVectorLength);
         if(nullptr == aResidualErrors) {
            LOG_0(TraceLevelWarning, "WARNING Exited DataSetByFeatureCombination::Initialize nullptr == aResidualErrors");
            return true;
         }
      }
      FloatEbmType * aPredictorScores = nullptr;
      if(bAllocatePredictorScores) {
         aPredictorScores = ConstructPredictorScores(cInstances, cVectorLength, aPredictorScoresFrom);
         if(nullptr == aPredictorScores) {
            free(aResidualErrors);
            LOG_0(TraceLevelWarning, "WARNING Exited DataSetByFeatureCombination::Initialize nullptr == aPredictorScores");
            return true;
         }
      }
      StorageDataType * aTargetData = nullptr;
      if(bAllocateTargetData) {
         aTargetData = ConstructTargetData(cInstances, static_cast<const IntEbmType *>(aTargets), runtimeLearningTypeOrCountTargetClasses);
         if(nullptr == aTargetData) {
            free(aResidualErrors);
            free(aPredictorScores);
            LOG_0(TraceLevelWarning, "WARNING Exited DataSetByFeatureCombination::Initialize nullptr == aTargetData");
            return true;
         }
      }
      StorageDataType ** aaInputData = nullptr;
      if(0 != cFeatureCombinations) {
         aaInputData = ConstructInputData(cFeatureCombinations, apFeatureCombination, cInstances, aInputDataFrom);
         if(nullptr == aaInputData) {
            free(aResidualErrors);
            free(aPredictorScores);
            free(aTargetData);
            LOG_0(TraceLevelWarning, "WARNING Exited DataSetByFeatureCombination::Initialize nullptr == aaInputData");
            return true;
         }
      }

      m_aResidualErrors = aResidualErrors;
      m_aPredictorScores = aPredictorScores;
      m_aTargetData = aTargetData;
      m_aaInputData = aaInputData;
      m_cInstances = cInstances;
      m_cFeatureCombinations = cFeatureCombinations;
   }

   LOG_0(TraceLevelInfo, "Exited DataSetByFeatureCombination::Initialize");

   return false;
}

WARNING_PUSH
WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
void DataSetByFeatureCombination::Destruct() {
   LOG_0(TraceLevelInfo, "Entered DataSetByFeatureCombination::Destruct");

   free(m_aResidualErrors);
   free(m_aPredictorScores);
   free(m_aTargetData);

   if(nullptr != m_aaInputData) {
      EBM_ASSERT(0 < m_cFeatureCombinations);
      StorageDataType * * paInputData = m_aaInputData;
      const StorageDataType * const * const paInputDataEnd = m_aaInputData + m_cFeatureCombinations;
      do {
         free(*paInputData);
         ++paInputData;
      } while(paInputDataEnd != paInputData);
      free(m_aaInputData);
   }

   LOG_0(TraceLevelInfo, "Exited DataSetByFeatureCombination::Destruct");
}
WARNING_POP
