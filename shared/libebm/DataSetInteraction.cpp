// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "common_cpp.hpp" // INLINE_RELEASE_UNTEMPLATED

#include "ebm_internal.hpp" // AddPositiveFloatsSafe

#include "Feature.hpp"
#include "dataset_shared.hpp" // SharedStorageDataType
#include "DataSetInteraction.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern bool CheckWeightsEqual(
   const BagEbm direction,
   const BagEbm * const aBag,
   const FloatFast * pWeights,
   const size_t cSetSamples
);

ErrorEbm DataSetInteraction::InitGradHess(
   const ObjectiveWrapper * const pObjective,
   const size_t cScores,
   const bool bAllocateHessians
) {
   LOG_0(Trace_Info, "Entered DataSetInteraction::InitGradHess");

   UNUSED(pObjective);

   EBM_ASSERT(nullptr != pObjective);
   EBM_ASSERT(1 <= cScores);

   const size_t cStorageItems = bAllocateHessians ? size_t { 2 } : size_t { 1 };

   EBM_ASSERT(sizeof(FloatFast) == pObjective->cFloatBytes); // TODO: add this check elsewhere that FloatFast is used
   if(IsMultiplyError(sizeof(FloatFast) * cStorageItems, cScores)) {
      LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitGradHess IsMultiplyError(sizeof(FloatFast) * cStorageItems, cScores)");
      return Error_OutOfMemory;
   }
   const size_t cElementBytes = sizeof(FloatFast) * cStorageItems * cScores;

   DataSubsetInteraction * pSubset = m_aSubsets;
   const DataSubsetInteraction * const pSubsetsEnd = pSubset + m_cSubsets;
   do {
      const size_t cSubsetSamples = pSubset->m_cSamples;
      EBM_ASSERT(1 <= cSubsetSamples);

      if(IsMultiplyError(cElementBytes, cSubsetSamples)) {
         LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitGradHess IsMultiplyError(cElementBytes, cSubsetSamples)");
         return Error_OutOfMemory;
      }
      const size_t cBytesGradHess = cElementBytes * cSubsetSamples;
      ANALYSIS_ASSERT(0 != cBytesGradHess);

      FloatFast * const aGradHess = static_cast<FloatFast *>(AlignedAlloc(cBytesGradHess));
      if(nullptr == aGradHess) {
         LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitGradHess nullptr == aGradHess");
         return Error_OutOfMemory;
      }
      pSubset->m_aGradHess = aGradHess;

      ++pSubset;
   } while(pSubsetsEnd != pSubset);

   LOG_0(Trace_Info, "Exited DataSetInteraction::InitGradHess");
   return Error_None;
}

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
ErrorEbm DataSetInteraction::InitFeatureData(
   const unsigned char * const pDataSetShared,
   const size_t cSharedSamples,
   const BagEbm * const aBag,
   const size_t cFeatures
) {
   LOG_0(Trace_Info, "Entered DataSetInteraction::InitFeatureData");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(1 <= cSharedSamples);
   EBM_ASSERT(1 <= cFeatures);

   EBM_ASSERT(nullptr != m_aSubsets);
   EBM_ASSERT(1 <= m_cSubsets);
   const DataSubsetInteraction * const pSubsetsEnd = m_aSubsets + m_cSubsets;

   size_t iFeature = 0;
   do {
      bool bMissing;
      bool bUnknown;
      bool bNominal;
      bool bSparse;
      SharedStorageDataType countBins;
      SharedStorageDataType defaultValSparse;
      size_t cNonDefaultsSparse;
      const void * aFeatureDataFrom = GetDataSetSharedFeature(
         pDataSetShared,
         iFeature,
         &bMissing,
         &bUnknown,
         &bNominal,
         &bSparse,
         &countBins,
         &defaultValSparse,
         &cNonDefaultsSparse
      );
      EBM_ASSERT(nullptr != aFeatureDataFrom);
      EBM_ASSERT(!bSparse); // we don't support sparse yet

      EBM_ASSERT(!IsConvertError<size_t>(countBins)); // checked in a previous call to GetDataSetSharedFeature
      const size_t cBins = static_cast<size_t>(countBins);

      if(size_t { 1 } < cBins) {
         // we don't need any bits to store 1 bin since it's always going to be the only bin available, and also 
         // we return 0.0 on interactions whenever we find a feature with 1 bin before further processing
      
         if(IsConvertError<StorageDataType>(cBins - size_t { 1 })) {
            // if we check this here, we can be guaranteed that any feature data will convert to StorageDataType
            // since the shared datastructure would not allow data items equal or greater than cBins
            LOG_0(Trace_Error, "ERROR DataSetInteraction::InitFeatureData IsConvertError<StorageDataType>(cBins - 1)");
            return Error_OutOfMemory;
         }

         const size_t cBitsRequiredMin = CountBitsRequired(cBins - size_t { 1 });
         EBM_ASSERT(1 <= cBitsRequiredMin);
         EBM_ASSERT(cBitsRequiredMin <= k_cBitsForSharedStorageType); // comes from shared data set
         EBM_ASSERT(cBitsRequiredMin <= k_cBitsForSizeT); // since cBins fits into size_t (previous call to GetDataSetSharedFeature)
         EBM_ASSERT(cBitsRequiredMin <= k_cBitsForStorageType); // since cBins - 1 (above) fits into StorageDataType

         const size_t cItemsPerBitPackFrom = GetCountItemsBitPacked<SharedStorageDataType>(cBitsRequiredMin);
         EBM_ASSERT(1 <= cItemsPerBitPackFrom);
         EBM_ASSERT(cItemsPerBitPackFrom <= k_cBitsForSharedStorageType);

         const size_t cBitsPerItemMaxFrom = GetCountBits<SharedStorageDataType>(cItemsPerBitPackFrom);
         EBM_ASSERT(1 <= cBitsPerItemMaxFrom);
         EBM_ASSERT(cBitsPerItemMaxFrom <= k_cBitsForSharedStorageType);

         // we can only guarantee that cBitsPerItemMaxFrom is less than or equal to k_cBitsForSharedStorageType
         // so we need to construct our mask in that type, but afterwards we can convert it to a 
         // StorageDataType since we know the ultimate answer must fit into that. If in theory 
         // SharedStorageDataType were allowed to be a billion, then the mask could be 65 bits while the end
         // result would be forced to be 64 bits or less since we use the maximum number of bits per item possible
         const StorageDataType maskBitsFrom = static_cast<StorageDataType>(MakeLowMask<SharedStorageDataType>(cBitsPerItemMaxFrom));

         ptrdiff_t iShiftFrom = static_cast<ptrdiff_t>((cSharedSamples - size_t { 1 }) % cItemsPerBitPackFrom);

         const BagEbm * pSampleReplication = aBag;
         const SharedStorageDataType * pFeatureDataFrom = static_cast<const SharedStorageDataType *>(aFeatureDataFrom);

         BagEbm replication = 0;
         StorageDataType featureData;

         DataSubsetInteraction * pSubset = m_aSubsets;
         do {
            const size_t cItemsPerBitPackTo = GetCountItemsBitPacked<StorageDataType>(cBitsRequiredMin);
            EBM_ASSERT(1 <= cItemsPerBitPackTo);
            EBM_ASSERT(cItemsPerBitPackTo <= k_cBitsForStorageType);

            const size_t cBitsPerItemMaxTo = GetCountBits<StorageDataType>(cItemsPerBitPackTo);
            EBM_ASSERT(1 <= cBitsPerItemMaxTo);
            EBM_ASSERT(cBitsPerItemMaxTo <= k_cBitsForStorageType);

            const size_t cSubsetSamples = pSubset->GetCountSamples();
            EBM_ASSERT(1 <= cSubsetSamples);
            const size_t cDataUnitsTo = (cSubsetSamples - size_t { 1 }) / cItemsPerBitPackTo + size_t { 1 }; // this can't overflow or underflow

            if(IsMultiplyError(sizeof(StorageDataType), cDataUnitsTo)) {
               LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitFeatureData IsMultiplyError(sizeof(StorageDataType), cDataUnitsTo)");
               return Error_OutOfMemory;
            }
            StorageDataType * pFeatureDataTo = static_cast<StorageDataType *>(AlignedAlloc(sizeof(StorageDataType) * cDataUnitsTo));
            if(nullptr == pFeatureDataTo) {
               LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitFeatureData nullptr == pFeatureDataTo");
               return Error_OutOfMemory;
            }
            pSubset->m_aaFeatureData[iFeature] = pFeatureDataTo;

            const StorageDataType * const pFeatureDataToEnd = &pFeatureDataTo[cDataUnitsTo];


            ptrdiff_t cShiftTo = static_cast<ptrdiff_t>((cSubsetSamples - 1) % cItemsPerBitPackTo * cBitsPerItemMaxTo);
            const ptrdiff_t cShiftResetTo = static_cast<ptrdiff_t>((cItemsPerBitPackTo - 1) * cBitsPerItemMaxTo);
            do {
               StorageDataType bitsTo = 0;
               do {
                  if(BagEbm { 0 } == replication) {
                     replication = 1;
                     if(nullptr != pSampleReplication) {
                        const BagEbm * pSampleReplicationOriginal = pSampleReplication;
                        do {
                           replication = *pSampleReplication;
                           ++pSampleReplication;
                        } while(replication <= BagEbm { 0 });
                        const size_t cAdvances = pSampleReplication - pSampleReplicationOriginal - 1;

                        size_t cCompleteAdvanced = cAdvances / cItemsPerBitPackFrom;
                        iShiftFrom -= static_cast<ptrdiff_t>(cAdvances % cItemsPerBitPackFrom);
                        if(iShiftFrom < ptrdiff_t { 0 }) {
                           ++cCompleteAdvanced;
                           iShiftFrom += cItemsPerBitPackFrom;
                        }
                        pFeatureDataFrom += cCompleteAdvanced;
                     }
                     EBM_ASSERT(0 <= iShiftFrom);
                     EBM_ASSERT(static_cast<size_t>(iShiftFrom * cBitsPerItemMaxFrom) < k_cBitsForSharedStorageType);

                     const SharedStorageDataType bitsFrom = *pFeatureDataFrom;
                     featureData = static_cast<StorageDataType>(bitsFrom >> (iShiftFrom * cBitsPerItemMaxFrom)) & maskBitsFrom;
                     EBM_ASSERT(static_cast<size_t>(featureData) < cBins);
                     --iShiftFrom;
                     if(iShiftFrom < ptrdiff_t { 0 }) {
                        ++pFeatureDataFrom;
                        iShiftFrom += cItemsPerBitPackFrom;
                     }
                  }

                  EBM_ASSERT(1 <= replication);
                  --replication;

                  EBM_ASSERT(0 <= cShiftTo);
                  EBM_ASSERT(static_cast<size_t>(cShiftTo) < k_cBitsForStorageType);
                  bitsTo |= featureData << cShiftTo;
                  cShiftTo -= cBitsPerItemMaxTo;
               } while(ptrdiff_t { 0 } <= cShiftTo);
               cShiftTo = cShiftResetTo;
               *pFeatureDataTo = bitsTo;
               ++pFeatureDataTo;
            } while(pFeatureDataToEnd != pFeatureDataTo);

            ++pSubset;
         } while(pSubsetsEnd != pSubset);
         EBM_ASSERT(0 == replication);
      }
      ++iFeature;
   } while(cFeatures != iFeature);

   LOG_0(Trace_Info, "Exited DataSetInteraction::InitFeatureData");
   return Error_None;
}
WARNING_POP

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
ErrorEbm DataSetInteraction::InitWeights(
   const unsigned char * const pDataSetShared,
   const BagEbm * const aBag,
   const size_t cSetSamples
) {
   LOG_0(Trace_Info, "Entered DataSetInteraction::InitWeights");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(1 <= cSetSamples);

   const FloatFast * pWeightFrom = GetDataSetSharedWeight(pDataSetShared, 0);
   EBM_ASSERT(nullptr != pWeightFrom);
   if(CheckWeightsEqual(BagEbm { 1 }, aBag, pWeightFrom, cSetSamples)) {
      LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitWeights all weights identical, so ignoring weights");
      return Error_None;
   }

   EBM_ASSERT(nullptr != m_aSubsets);
   EBM_ASSERT(1 <= m_cSubsets);
   DataSubsetInteraction * pSubset = m_aSubsets;
   const DataSubsetInteraction * const pSubsetsEnd = pSubset + m_cSubsets;

   const BagEbm * pSampleReplication = aBag;

   double totalWeight = 0.0;

   BagEbm replication = 0;
   FloatFast weight;
   do {
      const size_t cSubsetSamples = pSubset->m_cSamples;
      EBM_ASSERT(1 <= cSubsetSamples);

      if(IsMultiplyError(sizeof(FloatFast), cSubsetSamples)) {
         LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitWeights IsMultiplyError(sizeof(FloatFast), cSubsetSamples)");
         return Error_OutOfMemory;
      }
      FloatFast * pWeightTo = static_cast<FloatFast *>(AlignedAlloc(sizeof(FloatFast) * cSubsetSamples));
      if(nullptr == pWeightTo) {
         LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitWeights nullptr == pWeightTo");
         return Error_OutOfMemory;
      }
      pSubset->m_aWeights = pWeightTo;
      const FloatFast * const pWeightsToEnd = pWeightTo + cSubsetSamples;
      do {
         if(BagEbm { 0 } == replication) {
            replication = 1;
            if(nullptr != pSampleReplication) {
               do {
                  replication = *pSampleReplication;
                  ++pSampleReplication;
                  ++pWeightFrom;
               } while(replication <= BagEbm { 0 });
               --pWeightFrom;
            }
            weight = *pWeightFrom;
            ++pWeightFrom;
         }
         *pWeightTo = weight;
         ++pWeightTo;

         --replication;
      } while(pWeightsToEnd != pWeightTo);

      totalWeight += AddPositiveFloatsSafe<double>(cSubsetSamples, pSubset->m_aWeights);

      ++pSubset;
   } while(pSubsetsEnd != pSubset);
   EBM_ASSERT(0 == replication);

   if(std::isnan(totalWeight) || std::isinf(totalWeight) || totalWeight < std::numeric_limits<double>::min()) {
      LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitWeights std::isnan(totalWeight) || std::isinf(totalWeight) || totalWeight < std::numeric_limits<double>::min()");
      return Error_UserParamVal;
   }

   m_weightTotal = totalWeight;

   LOG_0(Trace_Info, "Exited DataSetInteraction::InitWeights");
   return Error_None;
}
WARNING_POP

void DataSubsetInteraction::DestructDataSubsetInteraction(const size_t cFeatures) {
   LOG_0(Trace_Info, "Entered DataSubsetInteraction::DestructDataSubsetInteraction");

   AlignedFree(m_aWeights);
   StorageDataType ** paFeatureData = m_aaFeatureData;
   if(nullptr != paFeatureData) {
      EBM_ASSERT(1 <= cFeatures);
      const StorageDataType * const * const paFeatureDataEnd = paFeatureData + cFeatures;
      do {
         AlignedFree(*paFeatureData);
         ++paFeatureData;
      } while(paFeatureDataEnd != paFeatureData);
      free(m_aaFeatureData);
   }
   AlignedFree(m_aGradHess);

   LOG_0(Trace_Info, "Exited DataSubsetInteraction::DestructDataSubsetInteraction");
}

ErrorEbm DataSetInteraction::InitDataSetInteraction(
   const ObjectiveWrapper * const pObjective,
   const size_t cSubsetItemsMax,
   const size_t cScores,
   const bool bAllocateHessians,
   const unsigned char * const pDataSetShared,
   const size_t cSharedSamples,
   const BagEbm * const aBag,
   const size_t cSetSamples,
   const size_t cWeights,
   const size_t cFeatures
) {
   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(nullptr != pDataSetShared);

   EBM_ASSERT(0 == m_cSamples);
   EBM_ASSERT(0 == m_cSubsets);
   EBM_ASSERT(nullptr == m_aSubsets);
   EBM_ASSERT(0.0 == m_weightTotal);

   LOG_0(Trace_Info, "Entered DataSetInteraction::InitDataSetInteraction");

   ErrorEbm error;

   if(0 != cSetSamples) {
      m_cSamples = cSetSamples;

      if(IsMultiplyError(sizeof(StorageDataType *), cFeatures)) {
         LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitDataSetInteraction IsMultiplyError(sizeof(StorageDataType *), cFeatures)");
         return Error_OutOfMemory;
      }

      const size_t cSubsets = (cSetSamples - size_t { 1 }) / cSubsetItemsMax + size_t { 1 };

      if(IsMultiplyError(sizeof(DataSubsetInteraction), cSubsets)) {
         LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitDataSetInteraction IsMultiplyError(sizeof(DataSubsetInteraction), cSubsets)");
         return Error_OutOfMemory;
      }
      DataSubsetInteraction * pSubset = static_cast<DataSubsetInteraction *>(malloc(sizeof(DataSubsetInteraction) * cSubsets));
      if(nullptr == pSubset) {
         LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitDataSetInteraction nullptr == pSubset");
         return Error_OutOfMemory;
      }
      m_aSubsets = pSubset;
      m_cSubsets = cSubsets;

      EBM_ASSERT(1 <= cSubsets);
      const DataSubsetInteraction * const pSubsetsEnd = pSubset + cSubsets;

      DataSubsetInteraction * pSubsetInit = pSubset;
      do {
         pSubsetInit->SafeInitDataSubsetInteraction();
         ++pSubsetInit;
      } while(pSubsetsEnd != pSubsetInit);

      size_t cSetSamplesRemaining = cSetSamples;
      do {
         const size_t cSubsetSamples = cSetSamplesRemaining <= cSubsetItemsMax ? cSetSamplesRemaining : cSubsetItemsMax;
         EBM_ASSERT(1 <= cSubsetSamples);
         pSubset->m_cSamples = cSubsetSamples;

         cSetSamplesRemaining -= cSubsetItemsMax; // this will overflow on last loop, but that's ok

         if(0 != cFeatures) {
            StorageDataType ** paData = static_cast<StorageDataType **>(malloc(sizeof(StorageDataType *) * cFeatures));
            if(nullptr == paData) {
               LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitDataSetInteraction nullptr == paData");
               return Error_OutOfMemory;
            }
            pSubset->m_aaFeatureData = paData;

            const StorageDataType * const * const paDataEnd = paData + cFeatures;
            do {
               *paData = nullptr;
               ++paData;
            } while(paDataEnd != paData);
         }

         ++pSubset;
      } while(pSubsetsEnd != pSubset);

      error = InitGradHess(
         pObjective,
         cScores,
         bAllocateHessians
      );
      if(Error_None != error) {
         return error;
      }

      if(0 != cFeatures) {
         error = InitFeatureData(
            pDataSetShared,
            cSharedSamples,
            aBag,
            cFeatures
         );
         if(Error_None != error) {
            return error;
         }
      }

      m_weightTotal = static_cast<double>(cSetSamples); // this is the default if there are no weights
      if(0 != cWeights) {
         error = InitWeights(
            pDataSetShared,
            aBag,
            cSetSamples
         );
         if(Error_None != error) {
            return error;
         }
      }
   }

   LOG_0(Trace_Info, "Exited DataSetInteraction::InitDataSetInteraction");
   return Error_None;
}

void DataSetInteraction::DestructDataSetInteraction(const size_t cFeatures) {
   LOG_0(Trace_Info, "Entered DataSetInteraction::DestructDataSetInteraction");

   DataSubsetInteraction * pSubset = m_aSubsets;
   if(nullptr != pSubset) {
      EBM_ASSERT(1 <= m_cSubsets);
      const DataSubsetInteraction * const pSubsetsEnd = pSubset + m_cSubsets;
      do {
         pSubset->DestructDataSubsetInteraction(cFeatures);
         ++pSubset;
      } while(pSubsetsEnd != pSubset);
      free(m_aSubsets);
   }

   LOG_0(Trace_Info, "Exited DataSetInteraction::DestructDataSetInteraction");
}

} // DEFINED_ZONE_NAME
