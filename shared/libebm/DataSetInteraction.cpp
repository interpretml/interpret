// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_internal.hpp"
#include "dataset_shared.hpp" // UIntShared
#include "DataSetInteraction.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

void DataSubsetInteraction::DestructDataSubsetInteraction(const size_t cFeatures) {
   LOG_0(Trace_Info, "Entered DataSubsetInteraction::DestructDataSubsetInteraction");

   AlignedFree(m_aWeights);

   void ** paFeatureData = m_aaFeatureData;
   if(nullptr != paFeatureData) {
      EBM_ASSERT(1 <= cFeatures);
      const void * const * const paFeatureDataEnd = paFeatureData + cFeatures;
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
      cTotalScores = cTotalScores << 1;
   }

   DataSubsetInteraction * pSubset = m_aSubsets;
   const DataSubsetInteraction * const pSubsetsEnd = pSubset + m_cSubsets;
   do {
      const size_t cSubsetSamples = pSubset->m_cSamples;
      EBM_ASSERT(1 <= cSubsetSamples);

      EBM_ASSERT(nullptr != pSubset->m_pObjective);
      if(IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cTotalScores, cSubsetSamples)) {
         LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitGradHess IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cTotalScores, cSubsetSamples)");
         return Error_OutOfMemory;
      }
      const size_t cBytesGradHess = pSubset->m_pObjective->m_cFloatBytes * cTotalScores * cSubsetSamples;
      ANALYSIS_ASSERT(0 != cBytesGradHess);

      void * const aGradHess = AlignedAlloc(cBytesGradHess);
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
      UIntShared countBins;
      UIntShared defaultValSparse;
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
      
         const int cBitsRequiredMin = CountBitsRequired(cBins - size_t { 1 });
         EBM_ASSERT(1 <= cBitsRequiredMin);
         EBM_ASSERT(cBitsRequiredMin <= COUNT_BITS(UIntShared)); // comes from shared data set
         EBM_ASSERT(cBitsRequiredMin <= COUNT_BITS(size_t)); // since cBins fits into size_t (previous call to GetDataSetSharedFeature)

         const int cItemsPerBitPackFrom = GetCountItemsBitPacked<UIntShared>(cBitsRequiredMin);
         EBM_ASSERT(1 <= cItemsPerBitPackFrom);
         EBM_ASSERT(cItemsPerBitPackFrom <= COUNT_BITS(UIntShared));

         const int cBitsPerItemMaxFrom = GetCountBits<UIntShared>(cItemsPerBitPackFrom);
         EBM_ASSERT(1 <= cBitsPerItemMaxFrom);
         EBM_ASSERT(cBitsPerItemMaxFrom <= COUNT_BITS(UIntShared));

         // we can only guarantee that cBitsPerItemMaxFrom is less than or equal to COUNT_BITS(UIntShared)
         // so we need to construct our mask in that type, but afterwards we can convert it to the 
         // zone type since we know the ultimate answer must fit into that. If in theory 
         // UIntShared were allowed to be a billion, then the mask could be 65 bits while the end
         // result would be forced to be 64 bits or less since we use the maximum number of bits per item possible
         const UIntShared maskBitsFrom = MakeLowMask<UIntShared>(cBitsPerItemMaxFrom);

         int iShiftFrom = static_cast<int>((cSharedSamples - size_t { 1 }) % static_cast<size_t>(cItemsPerBitPackFrom));

         const UIntShared * pFeatureDataFrom = static_cast<const UIntShared *>(aFeatureDataFrom);
         const BagEbm * pSampleReplication = aBag;
         BagEbm replication = 0;
         UIntShared iFeatureBin;

         DataSubsetInteraction * pSubset = m_aSubsets;
         do {
            const int cItemsPerBitPackTo =
               GetCountItemsBitPacked(cBitsRequiredMin, pSubset->GetObjectiveWrapper()->m_cUIntBytes);
            EBM_ASSERT(1 <= cItemsPerBitPackTo);

            const int cBitsPerItemMaxTo = GetCountBits(cItemsPerBitPackTo, pSubset->GetObjectiveWrapper()->m_cUIntBytes);
            EBM_ASSERT(1 <= cBitsPerItemMaxTo);

            const size_t cSIMDPack = pSubset->GetObjectiveWrapper()->m_cSIMDPack;
            EBM_ASSERT(1 <= cSIMDPack);

            const size_t cSubsetSamples = pSubset->GetCountSamples();
            EBM_ASSERT(1 <= cSubsetSamples);
            EBM_ASSERT(0 == cSubsetSamples % cSIMDPack);

            const size_t cParallelSamples = cSubsetSamples / cSIMDPack;
            EBM_ASSERT(1 <= cParallelSamples);

            // this can't overflow or underflow
            const size_t cParallelDataUnitsTo = (cParallelSamples - size_t { 1 }) / static_cast<size_t>(cItemsPerBitPackTo) + size_t { 1 };
            const size_t cDataUnitsTo = cParallelDataUnitsTo * cSIMDPack;

            if(IsMultiplyError(pSubset->GetObjectiveWrapper()->m_cUIntBytes, cDataUnitsTo)) {
               LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitFeatureData IsMultiplyError(pSubset->GetObjectiveWrapper()->m_cUIntBytes, cDataUnitsTo)");
               return Error_OutOfMemory;
            }
            const size_t cBytes = pSubset->GetObjectiveWrapper()->m_cUIntBytes * cDataUnitsTo;
            void * pFeatureDataTo = AlignedAlloc(cBytes);
            if(nullptr == pFeatureDataTo) {
               LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitFeatureData nullptr == pFeatureDataTo");
               return Error_OutOfMemory;
            }
            pSubset->m_aaFeatureData[iFeature] = pFeatureDataTo;
            const void * const pFeatureDataToEnd = IndexByte(pFeatureDataTo, cBytes);

            memset(pFeatureDataTo, 0, cBytes);

            ANALYSIS_ASSERT(0 != cItemsPerBitPackTo);
            int cShiftTo = static_cast<int>((cParallelSamples - size_t { 1 }) % static_cast<size_t>(cItemsPerBitPackTo)) * cBitsPerItemMaxTo;
            const int cShiftResetTo = (cItemsPerBitPackTo - 1) * cBitsPerItemMaxTo;
            do {
               do {
                  size_t iPartition = 0;
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

                           size_t cCompleteAdvanced = cAdvances / static_cast<size_t>(cItemsPerBitPackFrom);
                           iShiftFrom -= static_cast<int>(cAdvances % static_cast<size_t>(cItemsPerBitPackFrom));
                           if(iShiftFrom < 0) {
                              iShiftFrom += cItemsPerBitPackFrom;
                              EBM_ASSERT(0 <= iShiftFrom);
                              ++cCompleteAdvanced;
                           }
                           pFeatureDataFrom += cCompleteAdvanced;
                        }

                        const UIntShared bitsFrom = *pFeatureDataFrom;

                        EBM_ASSERT(0 <= iShiftFrom);
                        EBM_ASSERT(iShiftFrom * cBitsPerItemMaxFrom < COUNT_BITS(UIntShared));
                        iFeatureBin = (bitsFrom >> (iShiftFrom * cBitsPerItemMaxFrom)) & maskBitsFrom;

                        EBM_ASSERT(!IsConvertError<size_t>(iFeatureBin));
                        EBM_ASSERT(static_cast<size_t>(iFeatureBin) < cBins);

                        --iShiftFrom;
                        if(iShiftFrom < 0) {
                           EBM_ASSERT(-1 == iShiftFrom);
                           iShiftFrom += cItemsPerBitPackFrom;
                           ++pFeatureDataFrom;
                        }
                     }

                     EBM_ASSERT(1 <= replication);
                     --replication;

                     EBM_ASSERT(0 <= cShiftTo);
                     if(sizeof(UIntBig) == pSubset->m_pObjective->m_cUIntBytes) {
                        *(reinterpret_cast<UIntBig *>(pFeatureDataTo) + iPartition) |= static_cast<UIntBig>(iFeatureBin) << cShiftTo;
                     } else {
                        EBM_ASSERT(sizeof(UIntSmall) == pSubset->m_pObjective->m_cUIntBytes);
                        *(reinterpret_cast<UIntSmall *>(pFeatureDataTo) + iPartition) |= static_cast<UIntSmall>(iFeatureBin) << cShiftTo;
                     }

                     ++iPartition;
                  } while(cSIMDPack != iPartition);
                  cShiftTo -= cBitsPerItemMaxTo;
               } while(0 <= cShiftTo);
               cShiftTo = cShiftResetTo;

               pFeatureDataTo = IndexByte(pFeatureDataTo, pSubset->m_pObjective->m_cUIntBytes * cSIMDPack);
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

   const FloatShared * pWeightFrom = GetDataSetSharedWeight(pDataSetShared, 0);
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

      if(IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cSubsetSamples)) {
         LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitWeights IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cSubsetSamples)");
         return Error_OutOfMemory;
      }
      const size_t cBytes = pSubset->m_pObjective->m_cFloatBytes * cSubsetSamples;
      void * pWeightTo = AlignedAlloc(cBytes);
      if(nullptr == pWeightTo) {
         LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitWeights nullptr == pWeightTo");
         return Error_OutOfMemory;
      }
      pSubset->m_aWeights = pWeightTo;

      const void * const pWeightsToEnd = IndexByte(pWeightTo, cBytes);
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

            weight = static_cast<double>(*pWeightFrom);
            ++pWeightFrom;

            // these were checked when creating the shared dataset
            EBM_ASSERT(!std::isnan(weight));
            EBM_ASSERT(!std::isinf(weight));
            EBM_ASSERT(static_cast<double>(std::numeric_limits<float>::min()) <= weight);
            EBM_ASSERT(weight <= static_cast<double>(std::numeric_limits<float>::max()));
         }

         subsetWeight += weight;

         if(sizeof(FloatBig) == pSubset->m_pObjective->m_cFloatBytes) {
            *reinterpret_cast<FloatBig *>(pWeightTo) = static_cast<FloatBig>(weight);
         } else {
            EBM_ASSERT(sizeof(FloatSmall) == pSubset->m_pObjective->m_cFloatBytes);
            *reinterpret_cast<FloatSmall *>(pWeightTo) = static_cast<FloatSmall>(weight);
         }
         pWeightTo = IndexByte(pWeightTo, pSubset->m_pObjective->m_cFloatBytes);

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
            if(IsMultiplyError(sizeof(void *), cFeatures)) {
               LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitDataSetInteraction IsMultiplyError(sizeof(void *), cFeatures)");
               return Error_OutOfMemory;
            }
            void ** paFeatureData = static_cast<void **>(malloc(sizeof(void *) * cFeatures));
            if(nullptr == paFeatureData) {
               LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitDataSetInteraction nullptr == paData");
               return Error_OutOfMemory;
            }
            pSubset->m_aaFeatureData = paFeatureData;

            const void * const * const paFeatureDataEnd = paFeatureData + cFeatures;
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
