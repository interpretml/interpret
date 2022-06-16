// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "data_set_shared.hpp"
#include "Feature.hpp"
#include "FeatureGroup.hpp"
#include "DataSetBoosting.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

INLINE_RELEASE_UNTEMPLATED static FloatFast * ConstructGradientsAndHessians(const bool bAllocateHessians, const size_t cSamples, const size_t cVectorLength) {
   LOG_0(TraceLevelInfo, "Entered ConstructGradientsAndHessians");

   EBM_ASSERT(1 <= cSamples);
   EBM_ASSERT(1 <= cVectorLength);

   const size_t cStorageItems = bAllocateHessians ? 2 : 1;
   if(IsMultiplyError(cVectorLength, cStorageItems, cSamples)) {
      LOG_0(TraceLevelWarning, "WARNING ConstructGradientsAndHessians IsMultiplyError(cVectorLength, cStorageItems, cSamples)");
      return nullptr;
   }
   const size_t cElements = cVectorLength * cStorageItems * cSamples;

   FloatFast * aGradientsAndHessians = EbmMalloc<FloatFast>(cElements);

   LOG_0(TraceLevelInfo, "Exited ConstructGradientsAndHessians");
   return aGradientsAndHessians;
}

INLINE_RELEASE_UNTEMPLATED static FloatFast * ConstructPredictorScores(
   const size_t cVectorLength,
   const BagEbmType direction,
   const BagEbmType * const aBag,
   const double * const aPredictorScoresFrom,
   const size_t cSetSamples
) {
   LOG_0(TraceLevelInfo, "Entered DataSetBoosting::ConstructPredictorScores");

   EBM_ASSERT(0 < cVectorLength);
   EBM_ASSERT(BagEbmType { -1 } == direction || BagEbmType { 1 } == direction);
   EBM_ASSERT(0 < cSetSamples);

   if(IsMultiplyError(cVectorLength, cSetSamples)) {
      LOG_0(TraceLevelWarning, "WARNING DataSetBoosting::ConstructPredictorScores IsMultiplyError(cVectorLength, cSetSamples)");
      return nullptr;
   }

   const size_t cElements = cVectorLength * cSetSamples;
   FloatFast * const aPredictorScoresTo = EbmMalloc<FloatFast>(cElements);
   if(nullptr == aPredictorScoresTo) {
      LOG_0(TraceLevelWarning, "WARNING DataSetBoosting::ConstructPredictorScores nullptr == aPredictorScoresTo");
      return nullptr;
   }

   const size_t cBytesPerItem = sizeof(*aPredictorScoresTo) * cVectorLength;

   const BagEbmType * pBag = aBag;
   FloatFast * pPredictorScoresTo = aPredictorScoresTo;
   const FloatFast * const pPredictorScoresToEnd = aPredictorScoresTo + cVectorLength * cSetSamples;
   const double * pPredictorScoresFrom = aPredictorScoresFrom;
   const bool isLoopTraining = BagEbmType { 0 } < direction;
   do {
      BagEbmType countBagged = 1;
      if(nullptr != pBag) {
         countBagged = *pBag;
         ++pBag;
      }
      if(BagEbmType { 0 } != countBagged) {
         const bool isItemTraining = BagEbmType { 0 } < countBagged;
         if(isLoopTraining == isItemTraining) {
            do {
               EBM_ASSERT(pPredictorScoresTo < aPredictorScoresTo + cVectorLength * cSetSamples);
               if(nullptr == pPredictorScoresFrom) {
                  static_assert(std::numeric_limits<FloatFast>::is_iec559, "IEEE 754 guarantees zeros means a zero float");
                  memset(pPredictorScoresTo, 0, cBytesPerItem);
               } else {
                  static_assert(sizeof(*pPredictorScoresTo) == sizeof(*pPredictorScoresFrom), "float mismatch");
                  memcpy(pPredictorScoresTo, pPredictorScoresFrom, cBytesPerItem);
               }
               pPredictorScoresTo += cVectorLength;
               countBagged -= direction;
            } while(BagEbmType { 0 } != countBagged);
         }
         if(nullptr != pPredictorScoresFrom) {
            pPredictorScoresFrom += cVectorLength;
         }
      }
   } while(pPredictorScoresToEnd != pPredictorScoresTo);

#ifdef ZERO_FIRST_MULTICLASS_LOGIT

   // TODO: move this into the loop above
   if(2 <= cVectorLength) {
      FloatFast * pScore = aPredictorScoresTo;
      const FloatFast * const pScoreExteriorEnd = pScore + cVectorLength * cSetSamples;
      do {
         FloatFast scoreShift = pScore[0];
         const FloatFast * const pScoreInteriorEnd = pScore + cVectorLength;
         do {
            *pScore -= scoreShift;
            ++pScore;
         } while(pScoreInteriorEnd != pScore);
      } while(pScoreExteriorEnd != pScore);
   }

#endif // ZERO_FIRST_MULTICLASS_LOGIT

   LOG_0(TraceLevelInfo, "Exited DataSetBoosting::ConstructPredictorScores");
   return aPredictorScoresTo;
}

INLINE_RELEASE_UNTEMPLATED static StorageDataType * ConstructTargetData(
   const unsigned char * const pDataSetShared,
   const BagEbmType direction,
   const BagEbmType * const aBag,
   const size_t cSetSamples
) {
   LOG_0(TraceLevelInfo, "Entered DataSetBoosting::ConstructTargetData");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbmType { -1 } == direction || BagEbmType { 1 } == direction);
   EBM_ASSERT(0 < cSetSamples);

   ptrdiff_t runtimeLearningTypeOrCountTargetClasses;
   const void * const aTargets = GetDataSetSharedTarget(pDataSetShared, 0, &runtimeLearningTypeOrCountTargetClasses);
   EBM_ASSERT(1 <= runtimeLearningTypeOrCountTargetClasses); // this should be classification, and 0 < cSetSamples
   EBM_ASSERT(nullptr != aTargets);

   const size_t countTargetClasses = static_cast<size_t>(runtimeLearningTypeOrCountTargetClasses);

   StorageDataType * const aTargetData = EbmMalloc<StorageDataType>(cSetSamples);
   if(nullptr == aTargetData) {
      LOG_0(TraceLevelWarning, "WARNING nullptr == aTargetData");
      return nullptr;
   }

   const BagEbmType * pBag = aBag;
   const SharedStorageDataType * pTargetFrom = static_cast<const SharedStorageDataType *>(aTargets);
   StorageDataType * pTargetTo = aTargetData;
   StorageDataType * pTargetToEnd = aTargetData + cSetSamples;
   const bool isLoopTraining = BagEbmType { 0 } < direction;
   do {
      BagEbmType countBagged = 1;
      if(nullptr != pBag) {
         countBagged = *pBag;
         ++pBag;
      }
      if(BagEbmType { 0 } != countBagged) {
         const bool isItemTraining = BagEbmType { 0 } < countBagged;
         if(isLoopTraining == isItemTraining) {
            const SharedStorageDataType data = *pTargetFrom;
            EBM_ASSERT(!IsConvertError<size_t>(data));
            if(IsConvertError<StorageDataType>(data)) {
               // this shouldn't be possible since we previously checked that we could convert our target,
               // so if this is failing then we'll be larger than the maximum number of classes
               LOG_0(TraceLevelError, "ERROR DataSetBoosting::ConstructTargetData data target too big to reference memory");
               free(aTargetData);
               return nullptr;
            }
            const StorageDataType iData = static_cast<StorageDataType>(data);
            if(countTargetClasses <= static_cast<size_t>(iData)) {
               LOG_0(TraceLevelError, "ERROR DataSetBoosting::ConstructTargetData target value larger than number of classes");
               free(aTargetData);
               return nullptr;
            }
            do {
               EBM_ASSERT(pTargetTo < aTargetData + cSetSamples);
               *pTargetTo = iData;
               ++pTargetTo;
               countBagged -= direction;
            } while(BagEbmType { 0 } != countBagged);
         }
      }
      ++pTargetFrom;
   } while(pTargetToEnd != pTargetTo);

   LOG_0(TraceLevelInfo, "Exited DataSetBoosting::ConstructTargetData");
   return aTargetData;
}

struct InputDataPointerAndCountBins {

   InputDataPointerAndCountBins() = default; // preserve our POD status
   ~InputDataPointerAndCountBins() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   const SharedStorageDataType * m_pInputData;
   size_t m_cBins;
};
static_assert(std::is_standard_layout<InputDataPointerAndCountBins>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<InputDataPointerAndCountBins>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<InputDataPointerAndCountBins>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

INLINE_RELEASE_UNTEMPLATED static StorageDataType * * ConstructInputData(
   const unsigned char * const pDataSetShared,
   const BagEbmType direction,
   const BagEbmType * const aBag,
   const size_t cSetSamples,
   const size_t cFeatureGroups,
   const FeatureGroup * const * const apFeatureGroup
) {
   LOG_0(TraceLevelInfo, "Entered DataSetBoosting::ConstructInputData");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbmType { -1 } == direction || BagEbmType { 1 } == direction);
   EBM_ASSERT(0 < cSetSamples);
   EBM_ASSERT(0 < cFeatureGroups);
   EBM_ASSERT(nullptr != apFeatureGroup);

   StorageDataType ** const aaInputDataTo = EbmMalloc<StorageDataType *>(cFeatureGroups);
   if(nullptr == aaInputDataTo) {
      LOG_0(TraceLevelWarning, "WARNING DataSetBoosting::ConstructInputData nullptr == aaInputDataTo");
      return nullptr;
   }

   const bool isLoopTraining = BagEbmType { 0 } < direction;

   StorageDataType ** paInputDataTo = aaInputDataTo;
   const FeatureGroup * const * ppFeatureGroup = apFeatureGroup;
   const FeatureGroup * const * const ppFeatureGroupEnd = apFeatureGroup + cFeatureGroups;
   do {
      const FeatureGroup * const pFeatureGroup = *ppFeatureGroup;
      EBM_ASSERT(nullptr != pFeatureGroup);
      if(0 == pFeatureGroup->GetCountSignificantDimensions()) {
         *paInputDataTo = nullptr; // free will skip over these later
         ++paInputDataTo;
      } else {
         EBM_ASSERT(1 <= pFeatureGroup->GetBitPack());
         const size_t cItemsPerBitPack = static_cast<size_t>(pFeatureGroup->GetBitPack());
         // for a 32/64 bit storage item, we can't have more than 32/64 bit packed items stored
         EBM_ASSERT(cItemsPerBitPack <= CountBitsRequiredPositiveMax<StorageDataType>());
         const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPack);
         // if we have 1 item, it can't be larger than the number of bits of storage
         EBM_ASSERT(cBitsPerItemMax <= CountBitsRequiredPositiveMax<StorageDataType>());

         EBM_ASSERT(0 < cSetSamples);
         const size_t cDataUnits = (cSetSamples - 1) / cItemsPerBitPack + 1; // this can't overflow or underflow

         StorageDataType * pInputDataTo = EbmMalloc<StorageDataType>(cDataUnits);
         if(nullptr == pInputDataTo) {
            LOG_0(TraceLevelWarning, "WARNING DataSetBoosting::ConstructInputData nullptr == pInputDataTo");
            goto free_all;
         }
         *paInputDataTo = pInputDataTo;
         ++paInputDataTo;

         // stop on the last item in our array AND then do one special last loop with less or equal iterations to the normal loop
         const StorageDataType * const pInputDataToLast = pInputDataTo + cDataUnits - 1;
         EBM_ASSERT(pInputDataTo <= pInputDataToLast); // we have 1 item or more, and therefore the last one can't be before the first item

         const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
         EBM_ASSERT(1 <= pFeatureGroup->GetCountDimensions());
         const FeatureGroupEntry * const pFeatureGroupEntryEnd = pFeatureGroupEntry + pFeatureGroup->GetCountDimensions();

         InputDataPointerAndCountBins dimensionInfo[k_cDimensionsMax];
         InputDataPointerAndCountBins * pDimensionInfoInit = &dimensionInfo[0];
         do {
            const Feature * const pFeature = pFeatureGroupEntry->m_pFeature;
            const size_t cBins = pFeature->GetCountBins();
            EBM_ASSERT(size_t { 1 } <= cBins); // we don't construct datasets on empty training sets
            if(size_t { 1 } < cBins) {
               size_t cBinsUnused;
               bool bMissing;
               bool bUnknown;
               bool bNominal;
               bool bSparse;
               SharedStorageDataType defaultValueSparse;
               size_t cNonDefaultsSparse;
               const void * pInputDataFrom = GetDataSetSharedFeature(
                  pDataSetShared,
                  pFeature->GetIndexFeatureData(),
                  &cBinsUnused,
                  &bMissing,
                  &bUnknown,
                  &bNominal,
                  &bSparse,
                  &defaultValueSparse,
                  &cNonDefaultsSparse
               );
               EBM_ASSERT(nullptr != pInputDataFrom);
               EBM_ASSERT(cBinsUnused == cBins);
               EBM_ASSERT(!bSparse); // we don't support sparse yet

               pDimensionInfoInit->m_pInputData = static_cast<const SharedStorageDataType *>(pInputDataFrom);
               pDimensionInfoInit->m_cBins = cBins;
               ++pDimensionInfoInit;
            }
            ++pFeatureGroupEntry;
         } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
         EBM_ASSERT(pDimensionInfoInit == &dimensionInfo[pFeatureGroup->GetCountSignificantDimensions()]);

         const BagEbmType * pBag = aBag;
         BagEbmType countBagged = 0;
         size_t tensorIndex = 0;

         size_t shiftEnd = cBitsPerItemMax * cItemsPerBitPack;
         while(pInputDataTo < pInputDataToLast) /* do the last iteration AFTER we re-enter this loop through the goto label! */ {
         one_last_loop:;
            EBM_ASSERT(shiftEnd <= CountBitsRequiredPositiveMax<StorageDataType>());

            StorageDataType bits = 0;
            size_t shift = 0;
            do {
               if(BagEbmType { 0 } == countBagged) {
                  while(true) {
                     tensorIndex = 0;
                     size_t tensorMultiple = 1;
                     InputDataPointerAndCountBins * pDimensionInfo = &dimensionInfo[0];
                     do {
                        const SharedStorageDataType * pInputData = pDimensionInfo->m_pInputData;
                        const SharedStorageDataType inputData = *pInputData;
                        pDimensionInfo->m_pInputData = pInputData + 1;
                        EBM_ASSERT(!IsConvertError<size_t>(inputData));
                        const size_t iData = static_cast<size_t>(inputData);

                        if(pDimensionInfo->m_cBins <= iData) {
                           // TODO: I think this check has been moved to constructing the shared dataset
                           LOG_0(TraceLevelError, "ERROR DataSetBoosting::ConstructInputData iData value must be less than the number of bins");
                           goto free_all;
                        }
                        // we check for overflows during FeatureGroup construction, but let's check here again
                        EBM_ASSERT(!IsMultiplyError(tensorMultiple, pDimensionInfo->m_cBins));

                        // this can't overflow if the multiplication below doesn't overflow, and we checked for that above
                        tensorIndex += tensorMultiple * iData;
                        tensorMultiple *= pDimensionInfo->m_cBins;

                        ++pDimensionInfo;
                     } while(pDimensionInfoInit != pDimensionInfo);

                     countBagged = 1;
                     if(nullptr != pBag) {
                        countBagged = *pBag;
                        ++pBag;
                     }
                     if(BagEbmType { 0 } != countBagged) {
                        const bool isItemTraining = BagEbmType { 0 } < countBagged;
                        if(isLoopTraining == isItemTraining) {
                           break;
                        }
                     }
                  }
               }
               EBM_ASSERT(0 != countBagged);
               EBM_ASSERT(0 < countBagged && 0 < direction || countBagged < 0 && direction < 0);
               countBagged -= direction;

               // put our first item in the least significant bits.  We do this so that later when
               // unpacking the indexes, we can just AND our mask with the bitfield to get the index and in subsequent loops
               // we can just shift down.  This eliminates one extra shift that we'd otherwise need to make if the first
               // item was in the MSB
               EBM_ASSERT(shift < CountBitsRequiredPositiveMax<StorageDataType>());
               // the tensor index needs to fit in memory, but concivably StorageDataType does not
               EBM_ASSERT(!IsConvertError<StorageDataType>(tensorIndex)); // this was checked when determining packing
               bits |= static_cast<StorageDataType>(tensorIndex) << shift;
               shift += cBitsPerItemMax;
            } while(shiftEnd != shift);
            *pInputDataTo = bits;
            ++pInputDataTo;
         }

         if(pInputDataTo == pInputDataToLast) {
            // if this is the first time we've exited the loop, then re-enter it to do our last loop, but reduce the number of times we do the inner loop
            shiftEnd = cBitsPerItemMax * ((cSetSamples - 1) % cItemsPerBitPack + 1);
            goto one_last_loop;
         }

         EBM_ASSERT(0 == countBagged);
      }
      ++ppFeatureGroup;
   } while(ppFeatureGroupEnd != ppFeatureGroup);

   LOG_0(TraceLevelInfo, "Exited DataSetBoosting::ConstructInputData");
   return aaInputDataTo;

free_all:
   while(aaInputDataTo != paInputDataTo) {
      --paInputDataTo;
      free(*paInputDataTo);
   }
   free(aaInputDataTo);
   return nullptr;
}

ErrorEbmType DataSetBoosting::Initialize(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const bool bAllocateGradients,
   const bool bAllocateHessians,
   const bool bAllocatePredictorScores,
   const bool bAllocateTargetData,
   const unsigned char * const pDataSetShared,
   const BagEbmType direction,
   const BagEbmType * const aBag,
   const double * const aPredictorScores,
   const size_t cSetSamples,
   const size_t cFeatureGroups,
   const FeatureGroup * const * const apFeatureGroup
) {
   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbmType { -1 } == direction || BagEbmType { 1 } == direction);

   EBM_ASSERT(nullptr == m_aGradientsAndHessians);
   EBM_ASSERT(nullptr == m_aPredictorScores);
   EBM_ASSERT(nullptr == m_aTargetData);
   EBM_ASSERT(nullptr == m_aaInputData);

   LOG_0(TraceLevelInfo, "Entered DataSetBoosting::Initialize");
   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   if(0 != cSetSamples) {
      if(bAllocateGradients) {
         FloatFast * aGradientsAndHessians = ConstructGradientsAndHessians(bAllocateHessians, cSetSamples, cVectorLength);
         if(nullptr == aGradientsAndHessians) {
            LOG_0(TraceLevelWarning, "WARNING Exited DataSetBoosting::Initialize nullptr == aGradientsAndHessians");
            return Error_OutOfMemory;
         }
         m_aGradientsAndHessians = aGradientsAndHessians;
      } else {
         EBM_ASSERT(!bAllocateHessians);
      }
      if(bAllocatePredictorScores) {
         FloatFast * const aPredictorScoresBag = ConstructPredictorScores(
            cVectorLength, 
            direction, 
            aBag, 
            aPredictorScores,
            cSetSamples
         );
         if(nullptr == aPredictorScoresBag) {
            LOG_0(TraceLevelWarning, "WARNING Exited DataSetBoosting::Initialize nullptr == aPredictorScoresBag");
            return Error_OutOfMemory;
         }
         m_aPredictorScores = aPredictorScoresBag;
      }
      if(bAllocateTargetData) {
         StorageDataType * const aTargetData = ConstructTargetData(
            pDataSetShared,
            direction,
            aBag,
            cSetSamples
         );
         if(nullptr == aTargetData) {
            LOG_0(TraceLevelWarning, "WARNING Exited DataSetBoosting::Initialize nullptr == aTargetData");
            return Error_OutOfMemory;
         }
         m_aTargetData = aTargetData;
      }
      if(0 != cFeatureGroups) {
         StorageDataType ** const aaInputData = ConstructInputData(
            pDataSetShared,
            direction,
            aBag,
            cSetSamples,
            cFeatureGroups,
            apFeatureGroup
         );
         if(nullptr == aaInputData) {
            LOG_0(TraceLevelWarning, "WARNING Exited DataSetBoosting::Initialize nullptr == aaInputData");
            return Error_OutOfMemory;
         }
         m_aaInputData = aaInputData;
      }
      m_cSamples = cSetSamples;
      m_cFeatureGroups = cFeatureGroups;
   }

   LOG_0(TraceLevelInfo, "Exited DataSetBoosting::Initialize");

   return Error_None;
}

WARNING_PUSH
WARNING_DISABLE_USING_UNINITIALIZED_MEMORY
void DataSetBoosting::Destruct() {
   LOG_0(TraceLevelInfo, "Entered DataSetBoosting::Destruct");

   free(m_aGradientsAndHessians);
   free(m_aPredictorScores);
   free(m_aTargetData);

   if(nullptr != m_aaInputData) {
      EBM_ASSERT(0 < m_cFeatureGroups);
      StorageDataType * * paInputData = m_aaInputData;
      const StorageDataType * const * const paInputDataEnd = m_aaInputData + m_cFeatureGroups;
      do {
         free(*paInputData);
         ++paInputData;
      } while(paInputDataEnd != paInputData);
      free(m_aaInputData);
   }

   LOG_0(TraceLevelInfo, "Exited DataSetBoosting::Destruct");
}
WARNING_POP

} // DEFINED_ZONE_NAME
