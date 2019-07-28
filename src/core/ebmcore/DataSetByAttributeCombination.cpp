// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <assert.h>
#include <string.h> // memset
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebmcore.h" // FractionalDataType
#include "EbmInternal.h" // AttributeTypeCore
#include "Logging.h" // EBM_ASSERT & LOG
#include "AttributeInternal.h"
#include "AttributeCombinationInternal.h"
#include "DataSetByAttributeCombination.h"

#define INVALID_POINTER (reinterpret_cast<void *>(~static_cast<size_t>(0)))

TML_INLINE static FractionalDataType * ConstructResidualErrors(const size_t cCases, const size_t cVectorLength) {
   LOG(TraceLevelInfo, "Entered DataSetAttributeCombination::ConstructResidualErrors");

   EBM_ASSERT(1 <= cCases);
   EBM_ASSERT(1 <= cVectorLength);

   if(IsMultiplyError(cCases, cVectorLength)) {
      LOG(TraceLevelWarning, "WARNING DataSetAttributeCombination::ConstructResidualErrors IsMultiplyError(cCases, cVectorLength)");
      return nullptr;
   }

   const size_t cElements = cCases * cVectorLength;

   if(IsMultiplyError(sizeof(FractionalDataType), cElements)) {
      LOG(TraceLevelWarning, "WARNING DataSetAttributeCombination::ConstructResidualErrors IsMultiplyError(sizeof(FractionalDataType), cElements)");
      return nullptr;
   }

   const size_t cBytes = sizeof(FractionalDataType) * cElements;
   FractionalDataType * aResidualErrors = static_cast<FractionalDataType *>(malloc(cBytes));

   LOG(TraceLevelInfo, "Exited DataSetAttributeCombination::ConstructResidualErrors");
   return aResidualErrors;
}

TML_INLINE static FractionalDataType * ConstructPredictionScores(const size_t cCases, const size_t cVectorLength, const FractionalDataType * const aPredictionScoresFrom) {
   LOG(TraceLevelInfo, "Entered DataSetAttributeCombination::ConstructPredictionScores");

   EBM_ASSERT(0 < cCases);
   EBM_ASSERT(0 < cVectorLength);

   if(IsMultiplyError(cCases, cVectorLength)) {
      LOG(TraceLevelWarning, "WARNING DataSetAttributeCombination::ConstructPredictionScores IsMultiplyError(cCases, cVectorLength)");
      return nullptr;
   }

   const size_t cElements = cCases * cVectorLength;

   if(IsMultiplyError(sizeof(FractionalDataType), cElements)) {
      LOG(TraceLevelWarning, "WARNING DataSetAttributeCombination::ConstructPredictionScores IsMultiplyError(sizeof(FractionalDataType), cElements)");
      return nullptr;
   }

   const size_t cBytes = sizeof(FractionalDataType) * cElements;
   FractionalDataType * const aPredictionScoresTo = static_cast<FractionalDataType *>(malloc(cBytes));
   if(nullptr == aPredictionScoresTo) {
      LOG(TraceLevelWarning, "WARNING DataSetAttributeCombination::ConstructPredictionScores nullptr == aPredictionScoresTo");
      return nullptr;
   }

   if(nullptr == aPredictionScoresFrom) {
      memset(aPredictionScoresTo, 0, cBytes);
   } else {
      memcpy(aPredictionScoresTo, aPredictionScoresFrom, cBytes);
      constexpr bool bZeroingLogits = 0 <= k_iZeroClassificationLogitAtInitialize;
      if(bZeroingLogits) {
         // TODO : integrate this subtraction into the copy instead of doing it afterwards
         FractionalDataType * pScore = aPredictionScoresTo;
         const FractionalDataType * const pScoreExteriorEnd = pScore + cVectorLength * cCases;
         do {
            FractionalDataType scoreShift = pScore[k_iZeroClassificationLogitAtInitialize];
            const FractionalDataType * const pScoreInteriorEnd = pScore + cVectorLength;
            do {
               *pScore -= scoreShift;
               ++pScore;
            } while(pScoreInteriorEnd != pScore);
         } while(pScoreExteriorEnd != pScore);
      }
   }

   LOG(TraceLevelInfo, "Exited DataSetAttributeCombination::ConstructPredictionScores");
   return aPredictionScoresTo;
}

TML_INLINE static const StorageDataTypeCore * ConstructTargetData(const size_t cCases, const IntegerDataType * const aTargets) {
   LOG(TraceLevelInfo, "Entered DataSetAttributeCombination::ConstructTargetData");

   EBM_ASSERT(0 < cCases);
   EBM_ASSERT(nullptr != aTargets);

   if(IsMultiplyError(sizeof(StorageDataTypeCore), cCases)) {
      LOG(TraceLevelWarning, "WARNING DataSetAttributeCombination::ConstructTargetData");
      return nullptr;
   }
   const size_t cTargetArrayBytes = sizeof(StorageDataTypeCore) * cCases;
   StorageDataTypeCore * const aTargetData = static_cast<StorageDataTypeCore *>(malloc(cTargetArrayBytes));
   if(nullptr == aTargetData) {
      LOG(TraceLevelWarning, "WARNING nullptr == aTargetData");
      return nullptr;
   }

   const IntegerDataType * pTargetFrom = aTargets;
   const IntegerDataType * const pTargetFromEnd = aTargets + cCases;
   StorageDataTypeCore * pTargetTo = aTargetData;
   do {
      const IntegerDataType data = *pTargetFrom;
      EBM_ASSERT(0 <= data);
      EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, IntegerDataType>(data)));
      // we can't check the upper range of our target here since we don't have that information, so we have a function at the allocation entry point that checks it there.  See CheckTargets(..)
      *pTargetTo = static_cast<StorageDataTypeCore>(data);
      ++pTargetTo;
      ++pTargetFrom;
   } while(pTargetFromEnd != pTargetFrom);

   LOG(TraceLevelInfo, "Exited DataSetAttributeCombination::ConstructTargetData");
   return aTargetData;
}

struct InputDataPointerAndCountStates {
   const IntegerDataType * m_pInputData;
   size_t m_cStates;
};

TML_INLINE static const StorageDataTypeCore * const * ConstructInputData(const size_t cAttributeCombinations, const AttributeCombinationCore * const * const apAttributeCombination, const size_t cCases, const IntegerDataType * const aInputDataFrom) {
   LOG(TraceLevelInfo, "Entered DataSetAttributeCombination::ConstructInputData");

   EBM_ASSERT(0 < cAttributeCombinations);
   EBM_ASSERT(nullptr != apAttributeCombination);
   EBM_ASSERT(0 < cCases);
   EBM_ASSERT(nullptr != aInputDataFrom);

   if(IsMultiplyError(sizeof(void *), cAttributeCombinations)) {
      LOG(TraceLevelWarning, "WARNING DataSetAttributeCombination::ConstructInputData IsMultiplyError(sizeof(void *), cAttributeCombinations)");
      return nullptr;
   }
   const size_t cBytesMemoryArray = sizeof(void *) * cAttributeCombinations;
   StorageDataTypeCore ** const aaInputDataTo = static_cast<StorageDataTypeCore * *>(malloc(cBytesMemoryArray));
   if(nullptr == aaInputDataTo) {
      LOG(TraceLevelWarning, "WARNING DataSetAttributeCombination::ConstructInputData nullptr == aaInputDataTo");
      return nullptr;
   }

   StorageDataTypeCore ** paInputDataTo = aaInputDataTo;
   const AttributeCombinationCore * const * ppAttributeCombination = apAttributeCombination;
   const AttributeCombinationCore * const * const ppAttributeCombinationEnd = apAttributeCombination + cAttributeCombinations;
   do {
      const AttributeCombinationCore * const pAttributeCombination = *ppAttributeCombination;
      EBM_ASSERT(nullptr != pAttributeCombination);
      const size_t cAttributes = pAttributeCombination->m_cAttributes;
      if(0 == cAttributes) {
         *paInputDataTo = nullptr; // free will skip over these later
      } else {
         const size_t cItemsPerBitPackDataUnit = pAttributeCombination->m_cItemsPerBitPackDataUnit;
         EBM_ASSERT(cItemsPerBitPackDataUnit <= CountBitsRequiredPositiveMax<StorageDataTypeCore>()); // for a 32/64 bit storage item, we can't have more than 32/64 bit packed items stored
         const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackDataUnit);
         EBM_ASSERT(cBitsPerItemMax <= CountBitsRequiredPositiveMax<StorageDataTypeCore>()); // if we have 1 item, it can't be larger than the number of bits of storage

         EBM_ASSERT(0 < cCases);
         const size_t cDataUnits = (cCases - 1) / cItemsPerBitPackDataUnit + 1; // this can't overflow or underflow

         if(IsMultiplyError(sizeof(StorageDataTypeCore), cDataUnits)) {
            LOG(TraceLevelWarning, "WARNING DataSetAttributeCombination::ConstructInputData IsMultiplyError(sizeof(StorageDataTypeCore), cDataUnits)");
            goto free_all;
         }
         const size_t cBytesData = sizeof(StorageDataTypeCore) * cDataUnits;
         StorageDataTypeCore * pInputDataTo = static_cast<StorageDataTypeCore *>(malloc(cBytesData));
         if(nullptr == pInputDataTo) {
            LOG(TraceLevelWarning, "WARNING DataSetAttributeCombination::ConstructInputData nullptr == pInputDataTo");
            goto free_all;
         }
         *paInputDataTo = pInputDataTo;
         ++paInputDataTo;

         // stop on the last item in our array AND then do one special last loop with less or equal iterations to the normal loop
         const StorageDataTypeCore * const pInputDataToLast = reinterpret_cast<const StorageDataTypeCore *>(reinterpret_cast<const char *>(pInputDataTo) + cBytesData) - 1;
         EBM_ASSERT(pInputDataTo <= pInputDataToLast); // we have 1 item or more, and therefore the last one can't be before the first item

         const AttributeCombinationCore::AttributeCombinationEntry * pAttributeCombinationEntry = &pAttributeCombination->m_AttributeCombinationEntry[0];
         InputDataPointerAndCountStates dimensionInfo[k_cDimensionsMax];
         InputDataPointerAndCountStates * pDimensionInfo = &dimensionInfo[0];
         EBM_ASSERT(0 < cAttributes);
         const InputDataPointerAndCountStates * const pDimensionInfoEnd = &dimensionInfo[cAttributes];
         do {
            const AttributeInternalCore * const pAttribute = pAttributeCombinationEntry->m_pAttribute;
            pDimensionInfo->m_pInputData = &aInputDataFrom[pAttribute->m_iAttributeData * cCases];
            pDimensionInfo->m_cStates = pAttribute->m_cStates;
            ++pAttributeCombinationEntry;
            ++pDimensionInfo;
         } while(pDimensionInfoEnd != pDimensionInfo);

         // THIS IS NOT A CONSTANT FOR A REASON.. WE CHANGE IT ON OUR LAST ITERATION
         // if we ever template this function on cItemsPerBitPackDataUnit, then we'd want
         // to make this a constant so that the compiler could reason about it an eliminate loops
         // as it is, it isn't a constant, so the compiler would not be able to figure out that most
         // of the time it is a constant
         size_t shiftEnd = cBitsPerItemMax * cItemsPerBitPackDataUnit;
         while(pInputDataTo < pInputDataToLast) /* do the last iteration AFTER we re-enter this loop through the goto label! */ {
         one_last_loop:;
            EBM_ASSERT(shiftEnd <= CountBitsRequiredPositiveMax<StorageDataTypeCore>());

            size_t bits = 0;
            size_t shift = 0;
            do {
               size_t tensorMultiple = 1;
               size_t tensorIndex = 0;
               pDimensionInfo = &dimensionInfo[0];
               do {
                  const IntegerDataType * pInputData = pDimensionInfo->m_pInputData;
                  const IntegerDataType inputData = *pInputData;
                  pDimensionInfo->m_pInputData = pInputData + 1;

                  EBM_ASSERT(0 <= inputData);
                  EBM_ASSERT((IsNumberConvertable<size_t, IntegerDataType>(inputData))); // data must be lower than cTargetStates and cTargetStates fits into a size_t which we checked earlier
                  EBM_ASSERT(static_cast<size_t>(inputData) < pDimensionInfo->m_cStates);
                  EBM_ASSERT(!IsMultiplyError(tensorMultiple, pDimensionInfo->m_cStates)); // we check for overflows during AttributeCombination construction, but let's check here again

                  tensorIndex += tensorMultiple * static_cast<size_t>(inputData); // this can't overflow if the multiplication below doesn't overflow, and we checked for that above
                  tensorMultiple *= pDimensionInfo->m_cStates;

                  ++pDimensionInfo;
               } while(pDimensionInfoEnd != pDimensionInfo);
               // put our first item in the least significant bits.  We do this so that later when
               // unpacking the indexes, we can just AND our mask with the bitfield to get the index and in subsequent loops
               // we can just shift down.  This eliminates one extra shift that we'd otherwise need to make if the first
               // item was in the MSB
               EBM_ASSERT(shift < CountBitsRequiredPositiveMax<StorageDataTypeCore>());
               bits |= tensorIndex << shift;
               shift += cBitsPerItemMax;
            } while(shiftEnd != shift);
            EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, size_t>(bits)));
            *pInputDataTo = static_cast<StorageDataTypeCore>(bits);
            ++pInputDataTo;
         }

         if(pInputDataTo == pInputDataToLast) {
            // if this is the first time we've exited the loop, then re-enter it to do our last loop, but reduce the number of times we do the inner loop
            shiftEnd = cBitsPerItemMax * ((cCases - 1) % cItemsPerBitPackDataUnit + 1);
            goto one_last_loop;
         }
      }
      ++ppAttributeCombination;
   } while(ppAttributeCombinationEnd != ppAttributeCombination);

   LOG(TraceLevelInfo, "Exited DataSetAttributeCombination::ConstructInputData");
   return aaInputDataTo;

free_all:
   while(aaInputDataTo != paInputDataTo) {
      --paInputDataTo;
      free(*paInputDataTo);
   }
   free(aaInputDataTo);
   return nullptr;
}

DataSetAttributeCombination::DataSetAttributeCombination(const bool bAllocateResidualErrors, const bool bAllocatePredictionScores, const bool bAllocateTargetData, const size_t cAttributeCombinations, const AttributeCombinationCore * const * const apAttributeCombination, const size_t cCases, const IntegerDataType * const aInputDataFrom, const void * const aTargets, const FractionalDataType * const aPredictionScoresFrom, const size_t cVectorLength)
   : m_aResidualErrors(bAllocateResidualErrors ? ConstructResidualErrors(cCases, cVectorLength) : static_cast<FractionalDataType *>(INVALID_POINTER))
   , m_aPredictionScores(bAllocatePredictionScores ? ConstructPredictionScores(cCases, cVectorLength, aPredictionScoresFrom) : static_cast<FractionalDataType *>(INVALID_POINTER))
   , m_aTargetData(bAllocateTargetData ? ConstructTargetData(cCases, static_cast<const IntegerDataType *>(aTargets)) : static_cast<const StorageDataTypeCore *>(INVALID_POINTER))
   , m_aaInputData(ConstructInputData(cAttributeCombinations, apAttributeCombination, cCases, aInputDataFrom))
   , m_cCases(cCases)
   , m_cAttributeCombinations(cAttributeCombinations) {

   EBM_ASSERT(0 < cCases);
   EBM_ASSERT(0 < cAttributeCombinations);
}

DataSetAttributeCombination::~DataSetAttributeCombination() {
   LOG(TraceLevelInfo, "Entered ~DataSetAttributeCombination");

   if(INVALID_POINTER != m_aResidualErrors) {
      free(m_aResidualErrors);
   }
   if(INVALID_POINTER != m_aPredictionScores) {
      free(m_aPredictionScores);
   }
   if(INVALID_POINTER != m_aTargetData) {
      free(const_cast<StorageDataTypeCore *>(m_aTargetData));
   }
   if(nullptr != m_aaInputData) {
      EBM_ASSERT(0 < m_cAttributeCombinations);
      const StorageDataTypeCore * const * paInputData = m_aaInputData;
      const StorageDataTypeCore * const * const paInputDataEnd = m_aaInputData + m_cAttributeCombinations;
      do {
         free(const_cast<StorageDataTypeCore *>(*paInputData));
         ++paInputData;
      } while(paInputDataEnd != paInputData);
      free(const_cast<StorageDataTypeCore **>(m_aaInputData));
   }

   LOG(TraceLevelInfo, "Exited ~DataSetAttributeCombination");
}
