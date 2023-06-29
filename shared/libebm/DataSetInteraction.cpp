// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_internal.hpp" // SafeConvertFloat
#include "dataset_shared.hpp" // SharedStorageDataType
#include "DataSetInteraction.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

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

ErrorEbm DataSetInteraction::InitGradHess(
   const bool bAllocateHessians,
   const size_t cScores
) {
   LOG_0(Trace_Info, "Entered DataSetInteraction::InitGradHess");

   EBM_ASSERT(1 <= cScores);

   size_t cTotalScores = cScores;
   if(bAllocateHessians) {
      if(IsMultiplyError(size_t { 2 }, cTotalScores)) {
         LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitGradHess IsMultiplyError(size_t { 2 }, cTotalScores)");
         return Error_OutOfMemory;
      }
      cTotalScores = size_t { 2 } * cTotalScores;
   }

   DataSubsetInteraction * pSubset = m_aSubsets;
   const DataSubsetInteraction * const pSubsetsEnd = pSubset + m_cSubsets;
   do {
      const size_t cSubsetSamples = pSubset->m_cSamples;
      EBM_ASSERT(1 <= cSubsetSamples);

      EBM_ASSERT(nullptr != pSubset->m_pObjective);
      EBM_ASSERT(sizeof(FloatFast) == pSubset->m_pObjective->m_cFloatBytes); // TODO: add this check elsewhere that FloatFast is used
      if(IsMultiplyError(sizeof(FloatFast), cTotalScores, cSubsetSamples)) {
         LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitGradHess IsMultiplyError(sizeof(FloatFast), cTotalScores, cSubsetSamples)");
         return Error_OutOfMemory;
      }
      const size_t cBytesGradHess = sizeof(FloatFast) * cTotalScores * cSubsetSamples;
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

         const SharedStorageDataType * pFeatureDataFrom = static_cast<const SharedStorageDataType *>(aFeatureDataFrom);
         const BagEbm * pSampleReplication = aBag;
         BagEbm replication = 0;
         StorageDataType iFeatureBin;

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

            // this can't overflow or underflow
            const size_t cDataUnitsTo = (cSubsetSamples - size_t { 1 }) / cItemsPerBitPackTo + size_t { 1 };

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
                           iShiftFrom += cItemsPerBitPackFrom;
                           ++cCompleteAdvanced;
                        }
                        pFeatureDataFrom += cCompleteAdvanced;
                     }

                     const SharedStorageDataType bitsFrom = *pFeatureDataFrom;

                     EBM_ASSERT(0 <= iShiftFrom);
                     EBM_ASSERT(static_cast<size_t>(iShiftFrom) * cBitsPerItemMaxFrom < k_cBitsForSharedStorageType);
                     iFeatureBin = static_cast<StorageDataType>(bitsFrom >> 
                        (static_cast<size_t>(iShiftFrom) * cBitsPerItemMaxFrom)) & maskBitsFrom;

                     EBM_ASSERT(!IsConvertError<size_t>(iFeatureBin));
                     EBM_ASSERT(static_cast<size_t>(iFeatureBin) < cBins);

                     --iShiftFrom;
                     if(iShiftFrom < ptrdiff_t { 0 }) {
                        EBM_ASSERT(ptrdiff_t { -1 } == iShiftFrom);
                        iShiftFrom += cItemsPerBitPackFrom;
                        ++pFeatureDataFrom;
                     }
                  }

                  EBM_ASSERT(1 <= replication);
                  --replication;

                  EBM_ASSERT(0 <= cShiftTo);
                  EBM_ASSERT(static_cast<size_t>(cShiftTo) < k_cBitsForStorageType);
                  bitsTo |= iFeatureBin << cShiftTo;
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
   const BagEbm * const aBag
) {
   LOG_0(Trace_Info, "Entered DataSetInteraction::InitWeights");

   EBM_ASSERT(nullptr != pDataSetShared);

   const FloatFast * pWeightFrom = GetDataSetSharedWeight(pDataSetShared, 0);
   EBM_ASSERT(nullptr != pWeightFrom);

   EBM_ASSERT(nullptr != m_aSubsets);
   EBM_ASSERT(1 <= m_cSubsets);
   DataSubsetInteraction * pSubset = m_aSubsets;
   const DataSubsetInteraction * const pSubsetsEnd = pSubset + m_cSubsets;

   const BagEbm * pSampleReplication = aBag;

   double totalWeight = 0.0;

   BagEbm replication = 0;
   double weight;
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
      // add the weights in 2 stages to preserve precision
      double subsetWeight = 0.0;
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

            weight = SafeConvertFloat<double>(*pWeightFrom);
            ++pWeightFrom;

            // these were checked when creating the shared dataset
            EBM_ASSERT(!std::isnan(weight));
            EBM_ASSERT(!std::isinf(weight));
            EBM_ASSERT(double { std::numeric_limits<float>::min() } <= weight);
            EBM_ASSERT(weight <= double { std::numeric_limits<float>::max() });
         }

         subsetWeight += weight;

         *pWeightTo = SafeConvertFloat<FloatFast>(weight);
         ++pWeightTo;

         --replication;
      } while(pWeightsToEnd != pWeightTo);

      totalWeight += subsetWeight;

      ++pSubset;
   } while(pSubsetsEnd != pSubset);
   EBM_ASSERT(0 == replication);

   EBM_ASSERT(!std::isnan(totalWeight));
   EBM_ASSERT(std::numeric_limits<double>::min() <= totalWeight);

   if(std::isinf(totalWeight)) {
      LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitWeights std::isinf(totalWeight)");
      return Error_UserParamVal;
   }

   m_weightTotal = totalWeight;

   LOG_0(Trace_Info, "Exited DataSetInteraction::InitWeights");
   return Error_None;
}
WARNING_POP

ErrorEbm DataSetInteraction::InitDataSetInteraction(
   const bool bAllocateHessians,
   const size_t cScores,
   const size_t cSubsetItemsMax,
   const ObjectiveWrapper * const pObjectiveCpu,
   const ObjectiveWrapper * const pObjectiveSIMD,
   const unsigned char * const pDataSetShared,
   const size_t cSharedSamples,
   const BagEbm * const aBag,
   const size_t cIncludedSamples,
   const size_t cWeights,
   const size_t cFeatures
) {
   LOG_0(Trace_Info, "Entered DataSetInteraction::InitDataSetInteraction");

   ErrorEbm error;

   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(1 <= cSubsetItemsMax);
   EBM_ASSERT(nullptr != pObjectiveCpu);
   EBM_ASSERT(nullptr != pObjectiveCpu->m_pObjective); // the objective for the CPU zone cannot be null unlike SIMD
   EBM_ASSERT(nullptr != pObjectiveSIMD);
   EBM_ASSERT(nullptr != pDataSetShared);

   EBM_ASSERT(0 == m_cSamples);
   EBM_ASSERT(0 == m_cSubsets);
   EBM_ASSERT(nullptr == m_aSubsets);
   EBM_ASSERT(0.0 == m_weightTotal);

   if(0 != cIncludedSamples) {
      EBM_ASSERT(1 <= cSharedSamples);

      m_cSamples = cIncludedSamples;

      EBM_ASSERT(1 == pObjectiveCpu->m_cSIMDPack);
      EBM_ASSERT(nullptr == pObjectiveSIMD->m_pObjective && 0 == pObjectiveSIMD->m_cSIMDPack ||
         nullptr != pObjectiveSIMD->m_pObjective && 2 <= pObjectiveSIMD->m_cSIMDPack);
      const size_t cSIMDPack = pObjectiveSIMD->m_cSIMDPack;

      size_t cSubsets = 0;
      size_t cIncludedSamplesRemainingInit = cIncludedSamples;
      do {
         size_t cSubsetSamples = EbmMin(cIncludedSamplesRemainingInit, cSubsetItemsMax);

         if(size_t { 0 } == cSIMDPack || cSubsetSamples < cSIMDPack) {
            // these remaing items cannot be processed with the SIMD compute, so they go into the CPU compute
         } else {
            // drop any items which cannot fit into the SIMD pack
            cSubsetSamples = cSubsetSamples - cSubsetSamples % cSIMDPack;
         }
         ++cSubsets;
         EBM_ASSERT(1 <= cSubsetSamples);
         EBM_ASSERT(cSubsetSamples <= cIncludedSamplesRemainingInit);
         cIncludedSamplesRemainingInit -= cSubsetSamples;
      } while(size_t { 0 } != cIncludedSamplesRemainingInit);
      EBM_ASSERT(1 <= cSubsets);

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

      const DataSubsetInteraction * const pSubsetsEnd = pSubset + cSubsets;

      DataSubsetInteraction * pSubsetInit = pSubset;
      do {
         pSubsetInit->SafeInitDataSubsetInteraction();
         ++pSubsetInit;
      } while(pSubsetsEnd != pSubsetInit);

      size_t cIncludedSamplesRemaining = cIncludedSamples;
      do {
         EBM_ASSERT(1 <= cIncludedSamplesRemaining);

         size_t cSubsetSamples = EbmMin(cIncludedSamplesRemaining, cSubsetItemsMax);

         if(size_t { 0 } == cSIMDPack || cSubsetSamples < cSIMDPack) {
            // these remaing items cannot be processed with the SIMD compute, so they go into the CPU compute
            pSubset->m_pObjective = pObjectiveCpu;
         } else {
            // drop any items which cannot fit into the SIMD pack
            cSubsetSamples = cSubsetSamples - cSubsetSamples % cSIMDPack;
            pSubset->m_pObjective = pObjectiveSIMD;
         }
         EBM_ASSERT(nullptr != pSubset->m_pObjective->m_pObjective);
         EBM_ASSERT(1 <= cSubsetSamples);
         EBM_ASSERT(0 == cSubsetSamples % pSubset->m_pObjective->m_cSIMDPack);
         EBM_ASSERT(cSubsetSamples <= cIncludedSamplesRemaining);
         cIncludedSamplesRemaining -= cSubsetSamples;

         pSubset->m_cSamples = cSubsetSamples;

         if(0 != cFeatures) {
            // TODO: add this check elsewhere that StorageDataType is used
            EBM_ASSERT(sizeof(StorageDataType) == pSubset->m_pObjective->m_cUIntBytes);
            if(IsMultiplyError(sizeof(StorageDataType *), cFeatures)) {
               LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitDataSetInteraction IsMultiplyError(sizeof(StorageDataType *), cFeatures)");
               return Error_OutOfMemory;
            }
            StorageDataType ** paFeatureData = static_cast<StorageDataType **>(malloc(sizeof(StorageDataType *) * cFeatures));
            if(nullptr == paFeatureData) {
               LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitDataSetInteraction nullptr == paData");
               return Error_OutOfMemory;
            }
            pSubset->m_aaFeatureData = paFeatureData;

            const StorageDataType * const * const paFeatureDataEnd = paFeatureData + cFeatures;
            do {
               *paFeatureData = nullptr;
               ++paFeatureData;
            } while(paFeatureDataEnd != paFeatureData);
         }

         ++pSubset;
      } while(pSubsetsEnd != pSubset);
      EBM_ASSERT(0 == cIncludedSamplesRemaining);

      error = InitGradHess(bAllocateHessians, cScores);
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

      m_weightTotal = static_cast<double>(cIncludedSamples); // this is the default if there are no weights
      if(0 != cWeights) {
         error = InitWeights(
            pDataSetShared,
            aBag
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
