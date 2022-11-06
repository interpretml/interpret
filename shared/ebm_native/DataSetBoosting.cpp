// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "common_cpp.hpp" // INLINE_RELEASE_UNTEMPLATED
#include "bridge_cpp.hpp" // GetCountScores

#include "Feature.hpp" // Feature
#include "Term.hpp" // Term
#include "dataset_shared.hpp" // SharedStorageDataType
#include "DataSetBoosting.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

INLINE_RELEASE_UNTEMPLATED static FloatFast * ConstructGradientsAndHessians(const bool bAllocateHessians, const size_t cSamples, const size_t cScores) {
   LOG_0(Trace_Info, "Entered ConstructGradientsAndHessians");

   EBM_ASSERT(1 <= cSamples);
   EBM_ASSERT(1 <= cScores);

   const size_t cStorageItems = bAllocateHessians ? size_t { 2 } : size_t { 1 };
   if(IsMultiplyError(sizeof(FloatFast), cScores, cStorageItems, cSamples)) {
      LOG_0(Trace_Warning, "WARNING ConstructGradientsAndHessians IsMultiplyError(sizeof(FloatFast), cScores, cStorageItems, cSamples)");
      return nullptr;
   }
   const size_t cBytesGradientsAndHessians = sizeof(FloatFast) * cScores * cStorageItems * cSamples;
   ANALYSIS_ASSERT(0 != cBytesGradientsAndHessians);

   FloatFast * const aGradientsAndHessians = static_cast<FloatFast *>(malloc(cBytesGradientsAndHessians));

   LOG_0(Trace_Info, "Exited ConstructGradientsAndHessians");
   return aGradientsAndHessians;
}

WARNING_PUSH
// NOTE: This warning seems to be flagged by the DEBUG 32 bit build
WARNING_BUFFER_OVERRUN
INLINE_RELEASE_UNTEMPLATED static FloatFast * ConstructSampleScores(
   const size_t cScores,
   const BagEbm direction,
   const BagEbm * const aBag,
   const double * const aInitScores,
   const size_t cSetSamples
) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::ConstructSampleScores");

   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(1 <= cSetSamples);

   if(IsMultiplyError(sizeof(FloatFast), cScores, cSetSamples)) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructSampleScores IsMultiplyError(sizeof(FloatFast), cScores, cSetSamples)");
      return nullptr;
   }
   const size_t cElements = cScores * cSetSamples;

   FloatFast * const aSampleScores = static_cast<FloatFast *>(malloc(sizeof(FloatFast) * cElements));
   if(nullptr == aSampleScores) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructSampleScores nullptr == aSampleScores");
      return nullptr;
   }

   if(nullptr == aInitScores) {
      EBM_ASSERT(BagEbm { 0 } < direction); // if aBag is nullptr then we have no validation samples

      static_assert(std::numeric_limits<FloatFast>::is_iec559, "IEEE 754 guarantees zeros means a zero float");
      memset(aSampleScores, 0, sizeof(FloatFast) * cElements);
   } else {
      const size_t cBytesPerItem = sizeof(*aSampleScores) * cScores;

      const BagEbm * pSampleReplication = aBag;
      FloatFast * pSampleScore = aSampleScores;
      const FloatFast * const pSampleScoresEnd = &aSampleScores[cElements];
      const double * pInitScore = aInitScores;
      const bool isLoopTraining = BagEbm { 0 } < direction;
      do {
         BagEbm replication = 1;
         if(nullptr != pSampleReplication) {
            bool isItemTraining;
            do {
               do {
                  replication = *pSampleReplication;
                  ++pSampleReplication;
               } while(BagEbm { 0 } == replication);
               isItemTraining = BagEbm { 0 } < replication;
               pInitScore += cScores;
            } while(isLoopTraining != isItemTraining);
            pInitScore -= cScores;
         }
         do {
            EBM_ASSERT(pSampleScore < pSampleScoresEnd);

            static_assert(sizeof(*pSampleScore) == sizeof(*pInitScore), "float mismatch");
            memcpy(pSampleScore, pInitScore, cBytesPerItem);

            pSampleScore += cScores;
            replication -= direction;
         } while(BagEbm { 0 } != replication);
         pInitScore += cScores;
      } while(pSampleScoresEnd != pSampleScore);
   }

   LOG_0(Trace_Info, "Exited DataSetBoosting::ConstructSampleScores");
   return aSampleScores;
}
WARNING_POP

INLINE_RELEASE_UNTEMPLATED static StorageDataType * ConstructTargetData(
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const BagEbm * const aBag,
   const size_t cSetSamples
) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::ConstructTargetData");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(1 <= cSetSamples);

   ptrdiff_t cClasses;
   const void * const aTargets = GetDataSetSharedTarget(pDataSetShared, 0, &cClasses);
   EBM_ASSERT(1 <= cClasses); // this should be classification, and 0 < cSetSamples
   EBM_ASSERT(nullptr != aTargets);

   const size_t countClasses = static_cast<size_t>(cClasses);

   if(IsMultiplyError(sizeof(StorageDataType), cSetSamples)) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructTargetData IsMultiplyError(sizeof(StorageDataType), cSetSamples)");
      return nullptr;
   }
   StorageDataType * const aTargetData = static_cast<StorageDataType *>(malloc(sizeof(StorageDataType) * cSetSamples));
   if(nullptr == aTargetData) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructTargetData nullptr == aTargetData");
      return nullptr;
   }

   const BagEbm * pSampleReplication = aBag;
   const SharedStorageDataType * pTargetFrom = static_cast<const SharedStorageDataType *>(aTargets);
   StorageDataType * pTargetTo = aTargetData;
   StorageDataType * pTargetToEnd = aTargetData + cSetSamples;
   const bool isLoopTraining = BagEbm { 0 } < direction;
   EBM_ASSERT(nullptr != aBag || isLoopTraining); // if aBag is nullptr then we have no validation samples
   do {
      BagEbm replication = 1;
      if(nullptr != pSampleReplication) {
         bool isItemTraining;
         do {
            do {
               replication = *pSampleReplication;
               ++pSampleReplication;
               ++pTargetFrom;
            } while(BagEbm { 0 } == replication);
            isItemTraining = BagEbm { 0 } < replication;
         } while(isLoopTraining != isItemTraining);
         --pTargetFrom;
      }
      const SharedStorageDataType data = *pTargetFrom;
      ++pTargetFrom;
      EBM_ASSERT(!IsConvertError<size_t>(data));
      if(IsConvertError<StorageDataType>(data)) {
         // this shouldn't be possible since we previously checked that we could convert our target,
         // so if this is failing then we'll be larger than the maximum number of classes
         LOG_0(Trace_Error, "ERROR DataSetBoosting::ConstructTargetData data target too big to reference memory");
         free(aTargetData);
         return nullptr;
      }
      const StorageDataType iData = static_cast<StorageDataType>(data);
      if(countClasses <= static_cast<size_t>(iData)) {
         LOG_0(Trace_Error, "ERROR DataSetBoosting::ConstructTargetData target value larger than number of classes");
         free(aTargetData);
         return nullptr;
      }
      do {
         EBM_ASSERT(pTargetTo < aTargetData + cSetSamples);
         *pTargetTo = iData;
         ++pTargetTo;
         replication -= direction;
      } while(BagEbm { 0 } != replication);
   } while(pTargetToEnd != pTargetTo);

   LOG_0(Trace_Info, "Exited DataSetBoosting::ConstructTargetData");
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

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
INLINE_RELEASE_UNTEMPLATED static StorageDataType * * ConstructInputData(
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const BagEbm * const aBag,
   const size_t cSetSamples,
   const IntEbm * const aiTermFeatures,
   const size_t cTerms,
   const Term * const * const apTerms
) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::ConstructInputData");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(1 <= cSetSamples);
   EBM_ASSERT(1 <= cTerms);
   EBM_ASSERT(nullptr != apTerms);

   if(IsMultiplyError(sizeof(StorageDataType *), cTerms)) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructInputData IsMultiplyError(sizeof(StorageDataType *), cTerms)");
      return nullptr;
   }
   StorageDataType ** const aaInputDataTo = static_cast<StorageDataType **>(malloc(sizeof(StorageDataType *) * cTerms));
   if(nullptr == aaInputDataTo) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructInputData nullptr == aaInputDataTo");
      return nullptr;
   }

   const bool isLoopTraining = BagEbm { 0 } < direction;

   const IntEbm * piTermFeatures = aiTermFeatures;

   StorageDataType ** paInputDataTo = aaInputDataTo;
   const Term * const * ppTerm = apTerms;
   const Term * const * const ppTermsEnd = apTerms + cTerms;
   do {
      const Term * const pTerm = *ppTerm;
      EBM_ASSERT(nullptr != pTerm);
      if(0 == pTerm->GetCountRealDimensions()) {
         piTermFeatures += pTerm->GetCountDimensions();
         *paInputDataTo = nullptr; // free will skip over these later
         ++paInputDataTo;
      } else {
         EBM_ASSERT(1 <= pTerm->GetTermBitPack());
         const size_t cItemsPerBitPackTo = static_cast<size_t>(pTerm->GetTermBitPack());
         // for a 32/64 bit storage item, we can't have more than 32/64 bit packed items stored
         EBM_ASSERT(cItemsPerBitPackTo <= k_cBitsForStorageType);
         const size_t cBitsPerItemMaxTo = GetCountBits<StorageDataType>(cItemsPerBitPackTo);
         // if we have 1 item, it can't be larger than the number of bits of storage
         EBM_ASSERT(cBitsPerItemMaxTo <= k_cBitsForStorageType);

         EBM_ASSERT(1 <= cSetSamples);
         const size_t cDataUnitsTo = (cSetSamples - 1) / cItemsPerBitPackTo + 1; // this can't overflow or underflow

         if(IsMultiplyError(sizeof(StorageDataType), cDataUnitsTo)) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructInputData IsMultiplyError(sizeof(StorageDataType), cDataUnitsTo)");
            goto free_all;
         }
         StorageDataType * pInputDataTo = static_cast<StorageDataType *>(malloc(sizeof(StorageDataType) * cDataUnitsTo));
         if(nullptr == pInputDataTo) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructInputData nullptr == pInputDataTo");
            goto free_all;
         }
         *paInputDataTo = pInputDataTo;
         ++paInputDataTo;

         const StorageDataType * const pInputDataToEnd = pInputDataTo + cDataUnitsTo;

         const FeatureBoosting * const * ppFeature = pTerm->GetFeatures();
         EBM_ASSERT(1 <= pTerm->GetCountDimensions());
         const FeatureBoosting * const * const ppFeaturesEnd = &ppFeature[pTerm->GetCountDimensions()];

         InputDataPointerAndCountBins dimensionInfo[k_cDimensionsMax];
         InputDataPointerAndCountBins * pDimensionInfoInit = &dimensionInfo[0];
         do {
            const FeatureBoosting * const pFeature = *ppFeature;
            const size_t cBins = pFeature->GetCountBins();
            EBM_ASSERT(size_t { 1 } <= cBins); // we don't construct datasets on empty training sets
            if(size_t { 1 } < cBins) {
               const IntEbm indexFeature = *piTermFeatures;
               EBM_ASSERT(!IsConvertError<size_t>(indexFeature)); // we converted it previously
               const size_t iFeature = static_cast<size_t>(indexFeature);

               size_t cBinsUnused;
               bool bMissing;
               bool bUnknown;
               bool bNominal;
               bool bSparse;
               SharedStorageDataType defaultValSparse;
               size_t cNonDefaultsSparse;
               const void * pInputDataFrom = GetDataSetSharedFeature(
                  pDataSetShared,
                  iFeature,
                  &cBinsUnused,
                  &bMissing,
                  &bUnknown,
                  &bNominal,
                  &bSparse,
                  &defaultValSparse,
                  &cNonDefaultsSparse
               );
               EBM_ASSERT(nullptr != pInputDataFrom);
               EBM_ASSERT(cBinsUnused == cBins);
               EBM_ASSERT(!bSparse); // we don't support sparse yet

               pDimensionInfoInit->m_pInputData = static_cast<const SharedStorageDataType *>(pInputDataFrom);
               pDimensionInfoInit->m_cBins = cBins;
               ++pDimensionInfoInit;
            }
            ++piTermFeatures;
            ++ppFeature;
         } while(ppFeaturesEnd != ppFeature);
         EBM_ASSERT(pDimensionInfoInit == &dimensionInfo[pTerm->GetCountRealDimensions()]);

         EBM_ASSERT(nullptr != aBag || isLoopTraining); // if aBag is nullptr then we have no validation samples
         const BagEbm * pSampleReplication = aBag;
         BagEbm replication = 0;
         StorageDataType iTensor;

         ptrdiff_t cShiftTo = static_cast<ptrdiff_t>((cSetSamples - 1) % cItemsPerBitPackTo * cBitsPerItemMaxTo);
         const ptrdiff_t cShiftResetTo = static_cast<ptrdiff_t>((cItemsPerBitPackTo - 1) * cBitsPerItemMaxTo);

         do {
            StorageDataType bits = 0;
            do {
               if(BagEbm { 0 } == replication) {
                  replication = 1;
                  if(nullptr != pSampleReplication) {
                     const BagEbm * pSampleReplicationOriginal = pSampleReplication;
                     bool isItemTraining;
                     do {
                        do {
                           replication = *pSampleReplication;
                           ++pSampleReplication;
                        } while(BagEbm { 0 } == replication);
                        isItemTraining = BagEbm { 0 } < replication;
                     } while(isLoopTraining != isItemTraining);
                     const size_t cAdvances = pSampleReplication - pSampleReplicationOriginal - 1;
                     if(0 != cAdvances) {
                        InputDataPointerAndCountBins * pDimensionInfo = &dimensionInfo[0];
                        do {
                           const SharedStorageDataType * pInputData = pDimensionInfo->m_pInputData;
                           pDimensionInfo->m_pInputData = pInputData + cAdvances;
                           ++pDimensionInfo;
                        } while(pDimensionInfoInit != pDimensionInfo);
                     }
                  }

                  size_t tensorIndex = 0;
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
                        LOG_0(Trace_Error, "ERROR DataSetBoosting::ConstructInputData iData value must be less than the number of bins");
                        goto free_all;
                     }
                     // we check for overflows during Term construction, but let's check here again
                     EBM_ASSERT(!IsMultiplyError(tensorMultiple, pDimensionInfo->m_cBins));

                     // this can't overflow if the multiplication below doesn't overflow, and we checked for that above
                     tensorIndex += tensorMultiple * iData;
                     tensorMultiple *= pDimensionInfo->m_cBins;

                     ++pDimensionInfo;
                  } while(pDimensionInfoInit != pDimensionInfo);

                  EBM_ASSERT(!IsConvertError<StorageDataType>(tensorIndex)); // this was checked when determining packing
                  iTensor = static_cast<StorageDataType>(tensorIndex);
               }

               EBM_ASSERT(0 != replication);
               EBM_ASSERT(0 < replication && 0 < direction || replication < 0 && direction < 0);
               replication -= direction;

               EBM_ASSERT(0 <= cShiftTo);
               EBM_ASSERT(static_cast<size_t>(cShiftTo) < k_cBitsForStorageType);
               // the tensor index needs to fit in memory, but concivably StorageDataType does not
               bits |= iTensor << cShiftTo;
               cShiftTo -= cBitsPerItemMaxTo;
            } while(ptrdiff_t { 0 } <= cShiftTo);
            cShiftTo = cShiftResetTo;
            *pInputDataTo = bits;
            ++pInputDataTo;
         } while(pInputDataToEnd != pInputDataTo);
         EBM_ASSERT(0 == replication);
      }
      ++ppTerm;
   } while(ppTermsEnd != ppTerm);

   LOG_0(Trace_Info, "Exited DataSetBoosting::ConstructInputData");
   return aaInputDataTo;

free_all:
   while(aaInputDataTo != paInputDataTo) {
      --paInputDataTo;
      free(*paInputDataTo);
   }
   free(aaInputDataTo);
   return nullptr;
}
WARNING_POP

ErrorEbm DataSetBoosting::Initialize(
   const ptrdiff_t cClasses,
   const bool bAllocateGradients,
   const bool bAllocateHessians,
   const bool bAllocateSampleScores,
   const bool bAllocateTargetData,
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const BagEbm * const aBag,
   const double * const aInitScores,
   const size_t cSetSamples,
   const IntEbm * const aiTermFeatures,
   const size_t cTerms,
   const Term * const * const apTerms
) {
   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);

   EBM_ASSERT(nullptr == m_aGradientsAndHessians);
   EBM_ASSERT(nullptr == m_aSampleScores);
   EBM_ASSERT(nullptr == m_aTargetData);
   EBM_ASSERT(nullptr == m_aaInputData);

   LOG_0(Trace_Info, "Entered DataSetBoosting::Initialize");

   if(0 != cSetSamples) {
      const size_t cScores = GetCountScores(cClasses);

      if(bAllocateGradients) {
         // if there are 0 or 1 classes, then with reduction there should be zero scores and the caller should disable
         EBM_ASSERT(0 != cClasses);
         EBM_ASSERT(1 != cClasses);

         FloatFast * aGradientsAndHessians = ConstructGradientsAndHessians(bAllocateHessians, cSetSamples, cScores);
         if(nullptr == aGradientsAndHessians) {
            LOG_0(Trace_Warning, "WARNING Exited DataSetBoosting::Initialize nullptr == aGradientsAndHessians");
            return Error_OutOfMemory;
         }
         m_aGradientsAndHessians = aGradientsAndHessians;
      } else {
         EBM_ASSERT(!bAllocateHessians);
      }
      if(bAllocateSampleScores) {
         // if there are 0 or 1 classes, then with reduction there should be zero scores and the caller should disable
         EBM_ASSERT(0 != cClasses);
         EBM_ASSERT(1 != cClasses);

         FloatFast * const aSampleScores = ConstructSampleScores(
            cScores, 
            direction, 
            aBag, 
            aInitScores,
            cSetSamples
         );
         if(nullptr == aSampleScores) {
            LOG_0(Trace_Warning, "WARNING Exited DataSetBoosting::Initialize nullptr == aSampleScores");
            return Error_OutOfMemory;
         }
         m_aSampleScores = aSampleScores;
      }
      if(bAllocateTargetData) {
         StorageDataType * const aTargetData = ConstructTargetData(
            pDataSetShared,
            direction,
            aBag,
            cSetSamples
         );
         if(nullptr == aTargetData) {
            LOG_0(Trace_Warning, "WARNING Exited DataSetBoosting::Initialize nullptr == aTargetData");
            return Error_OutOfMemory;
         }
         m_aTargetData = aTargetData;
      }
      if(0 != cTerms) {
         StorageDataType ** const aaInputData = ConstructInputData(
            pDataSetShared,
            direction,
            aBag,
            cSetSamples,
            aiTermFeatures,
            cTerms,
            apTerms
         );
         if(nullptr == aaInputData) {
            LOG_0(Trace_Warning, "WARNING Exited DataSetBoosting::Initialize nullptr == aaInputData");
            return Error_OutOfMemory;
         }
         m_aaInputData = aaInputData;
         m_cTerms = cTerms; // only needed if nullptr != m_aaInputData
      }
      m_cSamples = cSetSamples;
   }

   LOG_0(Trace_Info, "Exited DataSetBoosting::Initialize");

   return Error_None;
}

void DataSetBoosting::Destruct() {
   LOG_0(Trace_Info, "Entered DataSetBoosting::Destruct");

   free(m_aGradientsAndHessians);
   free(m_aSampleScores);
   free(m_aTargetData);

   if(nullptr != m_aaInputData) {
      EBM_ASSERT(1 <= m_cTerms);
      StorageDataType * * paInputData = m_aaInputData;
      const StorageDataType * const * const paInputDataEnd = m_aaInputData + m_cTerms;
      do {
         free(*paInputData);
         ++paInputData;
      } while(paInputDataEnd != paInputData);
      free(m_aaInputData);
   }

   LOG_0(Trace_Info, "Exited DataSetBoosting::Destruct");
}

} // DEFINED_ZONE_NAME
