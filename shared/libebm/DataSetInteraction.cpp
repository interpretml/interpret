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

extern ErrorEbm ExtractWeights(
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const BagEbm * const aBag,
   const size_t cSetSamples,
   FloatFast ** ppWeightsOut
);

INLINE_RELEASE_UNTEMPLATED static ErrorEbm ConstructGradientsAndHessians(
   const size_t cScores,
   const bool bAllocateHessians,
   const size_t cSetSamples,
   FloatFast ** paGradientsAndHessiansOut
) {
   LOG_0(Trace_Info, "Entered ConstructGradientsAndHessians");

   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(1 <= cSetSamples);
   EBM_ASSERT(nullptr != paGradientsAndHessiansOut);
   EBM_ASSERT(nullptr == *paGradientsAndHessiansOut);

   const size_t cStorageItems = bAllocateHessians ? size_t { 2 } : size_t { 1 };
   if(IsMultiplyError(sizeof(FloatFast), cScores, cStorageItems, cSetSamples)) {
      LOG_0(Trace_Warning, "WARNING ConstructGradientsAndHessians IsMultiplyError(sizeof(FloatFast), cScores, cStorageItems, cSamples)");
      return Error_OutOfMemory;
   }
   const size_t cBytesGradientsAndHessians = sizeof(FloatFast) * cScores * cStorageItems * cSetSamples;
   ANALYSIS_ASSERT(0 != cBytesGradientsAndHessians);

   FloatFast * const aGradientsAndHessians = static_cast<FloatFast *>(AlignedAlloc(cBytesGradientsAndHessians));
   if(UNLIKELY(nullptr == aGradientsAndHessians)) {
      LOG_0(Trace_Warning, "WARNING ConstructGradientsAndHessians nullptr == aGradientsAndHessians");
      return Error_OutOfMemory;
   }
   *paGradientsAndHessiansOut = aGradientsAndHessians; // transfer ownership for future deletion

   LOG_0(Trace_Info, "Exited ConstructGradientsAndHessians");
   return Error_None;
}

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
ErrorEbm DataSetInteraction::InitializeInputData(
   const unsigned char * const pDataSetShared,
   const size_t cSharedSamples,
   const BagEbm * const aBag,
   const size_t cFeatures
) {
   LOG_0(Trace_Info, "Entered DataSetInteraction::InitializeInputData");

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
      const void * aInputDataFrom = GetDataSetSharedFeature(
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
      EBM_ASSERT(nullptr != aInputDataFrom);
      EBM_ASSERT(!bSparse); // we don't support sparse yet

      EBM_ASSERT(!IsConvertError<size_t>(countBins)); // checked in a previous call to GetDataSetSharedFeature
      const size_t cBins = static_cast<size_t>(countBins);

      if(size_t { 1 } < cBins) {
         // we don't need any bits to store 1 bin since it's always going to be the only bin available, and also 
         // we return 0.0 on interactions whenever we find a feature with 1 bin before further processing
      
         if(IsConvertError<StorageDataType>(cBins - size_t { 1 })) {
            // if we check this here, we can be guaranteed that any inputData will convert to StorageDataType
            // since the shared datastructure would not allow data items equal or greater than cBins
            LOG_0(Trace_Error, "ERROR DataSetInteraction::InitializeInputData IsConvertError<StorageDataType>(cBins - 1)");
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
         const SharedStorageDataType * pInputDataFrom = static_cast<const SharedStorageDataType *>(aInputDataFrom);

         BagEbm replication = 0;
         StorageDataType inputData;

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
               LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitializeInputData IsMultiplyError(sizeof(StorageDataType), cDataUnitsTo)");
               return Error_OutOfMemory;
            }
            StorageDataType * pInputDataTo = static_cast<StorageDataType *>(AlignedAlloc(sizeof(StorageDataType) * cDataUnitsTo));
            if(nullptr == pInputDataTo) {
               LOG_0(Trace_Warning, "WARNING DataSetInteraction::InitializeInputData nullptr == pInputDataTo");
               return Error_OutOfMemory;
            }
            pSubset->m_aaInputData[iFeature] = pInputDataTo;

            const StorageDataType * const pInputDataToEnd = &pInputDataTo[cDataUnitsTo];


            ptrdiff_t cShiftTo = static_cast<ptrdiff_t>((cSubsetSamples - 1) % cItemsPerBitPackTo * cBitsPerItemMaxTo);
            const ptrdiff_t cShiftResetTo = static_cast<ptrdiff_t>((cItemsPerBitPackTo - 1) * cBitsPerItemMaxTo);
            do {
               StorageDataType bits = 0;
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
                        pInputDataFrom += cCompleteAdvanced;
                     }
                     EBM_ASSERT(0 <= iShiftFrom);
                     EBM_ASSERT(static_cast<size_t>(iShiftFrom * cBitsPerItemMaxFrom) < k_cBitsForSharedStorageType);

                     const SharedStorageDataType dataFrom = *pInputDataFrom;
                     inputData = static_cast<StorageDataType>(dataFrom >> (iShiftFrom * cBitsPerItemMaxFrom)) & maskBitsFrom;
                     EBM_ASSERT(static_cast<size_t>(inputData) < cBins);
                     --iShiftFrom;
                     if(iShiftFrom < ptrdiff_t { 0 }) {
                        ++pInputDataFrom;
                        iShiftFrom += cItemsPerBitPackFrom;
                     }
                  }

                  EBM_ASSERT(1 <= replication);
                  --replication;

                  EBM_ASSERT(0 <= cShiftTo);
                  EBM_ASSERT(static_cast<size_t>(cShiftTo) < k_cBitsForStorageType);
                  bits |= inputData << cShiftTo;
                  cShiftTo -= cBitsPerItemMaxTo;
               } while(ptrdiff_t { 0 } <= cShiftTo);
               cShiftTo = cShiftResetTo;
               *pInputDataTo = bits;
               ++pInputDataTo;
            } while(pInputDataToEnd != pInputDataTo);

            ++pSubset;
         } while(pSubsetsEnd != pSubset);
         EBM_ASSERT(0 == replication);
      }
      ++iFeature;
   } while(cFeatures != iFeature);

   LOG_0(Trace_Info, "Exited DataSetInteraction::InitializeInputData");
   return Error_None;
}
WARNING_POP

void DataSubsetInteraction::Destruct(const size_t cFeatures) {
   LOG_0(Trace_Info, "Entered DataSubsetInteraction::Destruct");

   AlignedFree(m_aGradientsAndHessians);
   AlignedFree(m_aWeights);
   if(nullptr != m_aaInputData) {
      EBM_ASSERT(1 <= cFeatures);
      StorageDataType ** paInputData = m_aaInputData;
      const StorageDataType * const * const paInputDataEnd = m_aaInputData + cFeatures;
      do {
         AlignedFree(*paInputData);
         ++paInputData;
      } while(paInputDataEnd != paInputData);
      free(m_aaInputData);
   }

   LOG_0(Trace_Info, "Exited DataSubsetInteraction::Destruct");
}

ErrorEbm DataSubsetInteraction::Initialize(
   const size_t cScores,
   const bool bAllocateHessians,
   const unsigned char * const pDataSetShared,
   const BagEbm * const aBag,
   const size_t cSetSamples,
   const size_t cWeights
) {
   EBM_ASSERT(nullptr != pDataSetShared);

   EBM_ASSERT(nullptr == m_aGradientsAndHessians); // we expect to start with zeroed values
   EBM_ASSERT(nullptr == m_aaInputData); // we expect to start with zeroed values
   EBM_ASSERT(0 == m_cSamples); // we expect to start with zeroed values

   LOG_0(Trace_Info, "Entered DataSubsetInteraction::Initialize");

   ErrorEbm error;

   if(0 != cSetSamples) {
      // if cSamples is zero, then we don't need to allocate anything since we won't use them anyways

      EBM_ASSERT(nullptr == m_aWeights);
      m_weightTotal = static_cast<double>(cSetSamples);
      if(0 != cWeights) {
         error = ExtractWeights(
            pDataSetShared,
            BagEbm { 1 },
            aBag,
            cSetSamples,
            &m_aWeights
         );
         if(Error_None != error) {
            // error already logged
            return error;
         }
         if(nullptr != m_aWeights) {
            const double total = AddPositiveFloatsSafe<double>(cSetSamples, m_aWeights);
            if(std::isnan(total) || std::isinf(total) || total < std::numeric_limits<double>::min()) {
               LOG_0(Trace_Warning, "WARNING DataSubsetInteraction::Initialize std::isnan(total) || std::isinf(total) || total < std::numeric_limits<double>::min()");
               return Error_UserParamVal;
            }
            // if they were all zero then we'd ignore the weights param.  If there are negative numbers it might add
            // to zero though so check it after checking for negative
            EBM_ASSERT(0 != total);
            m_weightTotal = total;
         }
      }

      error = ConstructGradientsAndHessians(
         cScores,
         bAllocateHessians,
         cSetSamples,
         &m_aGradientsAndHessians
      );
      if(Error_None != error) {
         // we should have already logged the failure
         return error;
      }

      m_cSamples = cSetSamples;
   }

   LOG_0(Trace_Info, "Exited DataSubsetInteraction::Initialize");
   return Error_None;
}


ErrorEbm DataSetInteraction::Initialize(
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

   LOG_0(Trace_Info, "Entered DataSetInteraction::Initialize");

   ErrorEbm error;

   if(0 != cSetSamples) {
      m_cSamples = cSetSamples;

      if(IsMultiplyError(sizeof(StorageDataType *), cFeatures)) {
         LOG_0(Trace_Warning, "WARNING DataSetInteraction::Initialize IsMultiplyError(sizeof(StorageDataType *), cFeatures)");
         return Error_OutOfMemory;
      }





      // TODO: allow more than 1 subset
      size_t cSubsets = 1;



      if(IsMultiplyError(sizeof(DataSubsetInteraction), cSubsets)) {
         LOG_0(Trace_Warning, "WARNING DataSetInteraction::Initialize IsMultiplyError(sizeof(DataSubsetInteraction), cSubsets)");
         return Error_OutOfMemory;
      }
      DataSubsetInteraction * pSubset = static_cast<DataSubsetInteraction *>(malloc(sizeof(DataSubsetInteraction) * cSubsets));
      if(nullptr == pSubset) {
         LOG_0(Trace_Warning, "WARNING DataSetInteraction::Initialize nullptr == pSubset");
         return Error_OutOfMemory;
      }
      m_aSubsets = pSubset;
      m_cSubsets = cSubsets;

      EBM_ASSERT(1 <= cSubsets);
      const DataSubsetInteraction * const pSubsetsEnd = pSubset + cSubsets;

      DataSubsetInteraction * pSubsetInit = pSubset;
      do {
         pSubsetInit->InitializeUnfailing();
         ++pSubsetInit;
      } while(pSubsetsEnd != pSubsetInit);

      double totalWeight = 0.0;



      { // while loop
         // TODO: allow more than 1 subset (this entire block needs to change)
         error = pSubset->Initialize(
            cScores,
            bAllocateHessians,
            pDataSetShared,
            aBag,
            cSetSamples,
            cWeights
         );
         if(Error_None != error) {
            return error;
         }

         if(0 != cFeatures) {
            StorageDataType ** paData = static_cast<StorageDataType **>(malloc(sizeof(StorageDataType *) * cFeatures));
            if(nullptr == paData) {
               LOG_0(Trace_Warning, "WARNING DataSetInteraction::Initialize nullptr == paData");
               return Error_OutOfMemory;
            }
            pSubset->m_aaInputData = paData;

            const StorageDataType * const * const paDataEnd = paData + cFeatures;
            do {
               *paData = nullptr;
               ++paData;
            } while(paDataEnd != paData);
         }



         // TODO: does this work here or are we adding zeros?
         totalWeight += pSubset->GetWeightTotal();
      }
      m_weightTotal = totalWeight;








      if(0 != cFeatures) {
         error = InitializeInputData(
            pDataSetShared,
            cSharedSamples,
            aBag,
            cFeatures
         );
         if(Error_None != error) {
            return error;
         }
      }
   }

   LOG_0(Trace_Info, "Exited DataSetInteraction::Initialize");
   return Error_None;
}

void DataSetInteraction::Destruct(const size_t cFeatures) {
   LOG_0(Trace_Info, "Entered DataSetInteraction::Destruct");

   DataSubsetInteraction * pSubset = m_aSubsets;
   if(nullptr != pSubset) {
      EBM_ASSERT(1 <= m_cSubsets);
      const DataSubsetInteraction * const pSubsetsEnd = pSubset + m_cSubsets;
      do {
         pSubset->Destruct(cFeatures);
         ++pSubset;
      } while(pSubsetsEnd != pSubset);
      free(m_aSubsets);
   }

   LOG_0(Trace_Info, "Exited DataSetInteraction::Destruct");
}

} // DEFINED_ZONE_NAME
