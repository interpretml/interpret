// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "common_cpp.hpp" // INLINE_RELEASE_UNTEMPLATED

#include "ebm_internal.hpp" // SafeConvertFloat, AddPositiveFloatsSafe
#include "RandomDeterministic.hpp" // RandomDeterministic
#include "RandomNondeterministic.hpp" // RandomNondeterministic
#include "Feature.hpp" // Feature
#include "Term.hpp" // Term
#include "dataset_shared.hpp" // SharedStorageDataType
#include "DataSetBoosting.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern bool CheckWeightsEqual(
   const BagEbm direction,
   const BagEbm * const aBag,
   const FloatFast * pWeights,
   const size_t cIncludedSamples
);

void DataSubsetBoosting::DestructDataSubsetBoosting(const size_t cTerms, const size_t cInnerBags) {
   LOG_0(Trace_Info, "Entered DataSubsetBoosting::DestructDataSubsetBoosting");

   InnerBag::FreeInnerBags(cInnerBags, m_aInnerBags);

   StorageDataType ** paTermData = m_aaTermData;
   if(nullptr != paTermData) {
      EBM_ASSERT(1 <= cTerms);
      const StorageDataType * const * const paTermDataEnd = paTermData + cTerms;
      do {
         AlignedFree(*paTermData);
         ++paTermData;
      } while(paTermDataEnd != paTermData);
      free(m_aaTermData);
   }

   AlignedFree(m_aTargetData);
   AlignedFree(m_aSampleScores);
   AlignedFree(m_aGradHess);

   LOG_0(Trace_Info, "Exited DataSubsetBoosting::DestructDataSubsetBoosting");
}


ErrorEbm DataSetBoosting::InitGradHess(
   const bool bAllocateHessians,
   const size_t cScores,
   const ObjectiveWrapper * const pObjective
) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::InitGradHess");

   UNUSED(pObjective);

   EBM_ASSERT(1 <= cScores);

   const size_t cStorageItems = bAllocateHessians ? size_t { 2 } : size_t { 1 };

   EBM_ASSERT(sizeof(FloatFast) == pObjective->cFloatBytes); // TODO: add this check elsewhere that FloatFast is used
   if(IsMultiplyError(sizeof(FloatFast) * cStorageItems, cScores)) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitGradHess IsMultiplyError(sizeof(FloatFast) * cStorageItems, cScores)");
      return Error_OutOfMemory;
   }
   const size_t cElementBytes = sizeof(FloatFast) * cStorageItems * cScores;

   DataSubsetBoosting * pSubset = m_aSubsets;
   const DataSubsetBoosting * const pSubsetsEnd = pSubset + m_cSubsets;
   do {
      const size_t cSubsetSamples = pSubset->m_cSamples;
      EBM_ASSERT(1 <= cSubsetSamples);

      if(IsMultiplyError(cElementBytes, cSubsetSamples)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitGradHess IsMultiplyError(cElementBytes, cSubsetSamples)");
         return Error_OutOfMemory;
      }
      const size_t cBytesGradHess = cElementBytes * cSubsetSamples;
      ANALYSIS_ASSERT(0 != cBytesGradHess);

      FloatFast * const aGradHess = static_cast<FloatFast *>(AlignedAlloc(cBytesGradHess));
      if(nullptr == aGradHess) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitGradHess nullptr == aGradHess");
         return Error_OutOfMemory;
      }
      pSubset->m_aGradHess = aGradHess;

      ++pSubset;
   } while(pSubsetsEnd != pSubset);

   LOG_0(Trace_Info, "Exited DataSetBoosting::InitGradHess");
   return Error_None;
}

WARNING_PUSH
// NOTE: This warning seems to be flagged by the DEBUG 32 bit build
WARNING_BUFFER_OVERRUN
ErrorEbm DataSetBoosting::InitSampleScores(
   const size_t cScores,
   const BagEbm direction,
   const BagEbm * const aBag,
   const double * const aInitScores
) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::InitSampleScores");

   DataSubsetBoosting * pSubset = m_aSubsets;
   const DataSubsetBoosting * const pSubsetsEnd = pSubset + m_cSubsets;

   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(nullptr != aBag || BagEbm { 1 } == direction);  // if aBag is nullptr then we have no validation samples

   if(IsMultiplyError(sizeof(FloatFast), cScores)) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitSampleScores IsMultiplyError(sizeof(FloatFast), cScores)");
      return Error_OutOfMemory;
   }
   const size_t cBytesOneElement = sizeof(FloatFast) * cScores;

   if(nullptr == aInitScores) {
      static_assert(std::numeric_limits<FloatFast>::is_iec559, "IEEE 754 guarantees zeros means a zero float");
      do {
         const size_t cSubsetSamples = pSubset->m_cSamples;
         EBM_ASSERT(1 <= cSubsetSamples);

         if(IsMultiplyError(cBytesOneElement, cSubsetSamples)) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitSampleScores IsMultiplyError(cBytesOneElement, cSubsetSamples)");
            return Error_OutOfMemory;
         }
         const size_t cBytes = cBytesOneElement * cSubsetSamples;
         ANALYSIS_ASSERT(0 != cBytes);
         FloatFast * pSampleScore = static_cast<FloatFast *>(AlignedAlloc(cBytes));
         if(nullptr == pSampleScore) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitSampleScores nullptr == pSampleScore");
            return Error_OutOfMemory;
         }
         pSubset->m_aSampleScores = pSampleScore;

         memset(pSampleScore, 0, cBytes);

         ++pSubset;
      } while(pSubsetsEnd != pSubset);
   } else {
      const size_t cSubsetSamplesInit = pSubset->m_cSamples;
      EBM_ASSERT(1 <= cSubsetSamplesInit);

      if(IsMultiplyError(cBytesOneElement, cSubsetSamplesInit)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitSampleScores IsMultiplyError(cBytesOneElement, cSubsetSamplesInit)");
         return Error_OutOfMemory;
      }
      const size_t cBytesInit = cBytesOneElement * cSubsetSamplesInit;
      ANALYSIS_ASSERT(0 != cBytesInit);
      FloatFast * pSampleScore = static_cast<FloatFast *>(AlignedAlloc(cBytesInit));
      if(nullptr == pSampleScore) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitSampleScores nullptr == pSampleScore");
         return Error_OutOfMemory;
      }
      pSubset->m_aSampleScores = pSampleScore;
      const FloatFast * pSampleScoresEnd = IndexByte(pSampleScore, cBytesInit);


      const BagEbm * pSampleReplication = aBag;
      const double * pInitScore = aInitScores;
      const bool isLoopValidation = direction < BagEbm { 0 };
      while(true) {
         BagEbm replication = 1;
         if(nullptr != pSampleReplication) {
            bool isItemValidation;
            do {
               do {
                  replication = *pSampleReplication;
                  ++pSampleReplication;
               } while(BagEbm { 0 } == replication);
               isItemValidation = replication < BagEbm { 0 };
               pInitScore += cScores;
            } while(isLoopValidation != isItemValidation);
            pInitScore -= cScores;
         }

         const double * const pFromEnd = &pInitScore[cScores];
         do {
            const double * pFrom = pInitScore;
            do {
               *pSampleScore = SafeConvertFloat<FloatFast>(*pFrom);
               ++pSampleScore;
               ++pFrom;
            } while(pFromEnd != pFrom);

            if(pSampleScoresEnd == pSampleScore) {
               ++pSubset;
               if(pSubsetsEnd == pSubset) {
                  EBM_ASSERT(replication == direction);
                  LOG_0(Trace_Info, "Exited DataSetBoosting::InitSampleScores");
                  return Error_None;
               }

               const size_t cSubsetSamples = pSubset->m_cSamples;
               EBM_ASSERT(1 <= cSubsetSamples);

               if(IsMultiplyError(cBytesOneElement, cSubsetSamples)) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitSampleScores IsMultiplyError(cBytesOneElement, cSubsetSamples)");
                  return Error_OutOfMemory;
               }
               const size_t cBytes = cBytesOneElement * cSubsetSamples;
               pSampleScore = static_cast<FloatFast *>(AlignedAlloc(cBytes));
               if(nullptr == pSampleScore) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitSampleScores nullptr == pSampleScore");
                  return Error_OutOfMemory;
               }
               pSubset->m_aSampleScores = pSampleScore;
               pSampleScoresEnd = IndexByte(pSampleScore, cBytes);
            }

            replication -= direction;
         } while(BagEbm { 0 } != replication);
         pInitScore += cScores;
      }
   }

   LOG_0(Trace_Info, "Exited DataSetBoosting::InitSampleScores");
   return Error_None;
}
WARNING_POP


WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
ErrorEbm DataSetBoosting::InitTargetData(
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const BagEbm * const aBag
) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::InitTargetData");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);

   ptrdiff_t cClasses;
   const void * const aTargets = GetDataSetSharedTarget(pDataSetShared, 0, &cClasses);
   EBM_ASSERT(nullptr != aTargets); // we previously called GetDataSetSharedTarget and got back non-null result

   EBM_ASSERT(nullptr != m_aSubsets);
   EBM_ASSERT(1 <= m_cSubsets);
   DataSubsetBoosting * pSubset = m_aSubsets;
   const DataSubsetBoosting * const pSubsetsEnd = pSubset + m_cSubsets;

   const BagEbm * pSampleReplication = aBag;
   const bool isLoopValidation = direction < BagEbm { 0 };
   EBM_ASSERT(nullptr != aBag || !isLoopValidation); // if aBag is nullptr then we have no validation samples

   BagEbm replication = 0;
   if(IsClassification(cClasses)) {
      const size_t countClasses = static_cast<size_t>(cClasses);
      const SharedStorageDataType * pTargetFrom = static_cast<const SharedStorageDataType *>(aTargets);
      StorageDataType iData;
      do {
         const size_t cSubsetSamples = pSubset->m_cSamples;
         EBM_ASSERT(1 <= cSubsetSamples);

         if(IsMultiplyError(sizeof(StorageDataType), cSubsetSamples)) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitTargetData IsMultiplyError(sizeof(StorageDataType), cSubsetSamples)");
            return Error_OutOfMemory;
         }
         StorageDataType * pTargetTo = static_cast<StorageDataType *>(AlignedAlloc(sizeof(StorageDataType) * cSubsetSamples));
         if(nullptr == pTargetTo) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitTargetData nullptr == pTargetTo");
            return Error_OutOfMemory;
         }
         pSubset->m_aTargetData = pTargetTo;
         const StorageDataType * const pTargetToEnd = pTargetTo + cSubsetSamples;
         do {
            if(BagEbm { 0 } == replication) {
               replication = 1;
               if(nullptr != pSampleReplication) {
                  bool isItemValidation;
                  do {
                     do {
                        replication = *pSampleReplication;
                        ++pSampleReplication;
                        ++pTargetFrom;
                     } while(BagEbm { 0 } == replication);
                     isItemValidation = replication < BagEbm { 0 };
                  } while(isLoopValidation != isItemValidation);
                  --pTargetFrom;
               }
               const SharedStorageDataType data = *pTargetFrom;
               ++pTargetFrom;
               EBM_ASSERT(!IsConvertError<size_t>(data));
               if(IsConvertError<StorageDataType>(data)) {
                  // this shouldn't be possible since we previously checked that we could convert our target,
                  // so if this is failing then we'll be larger than the maximum number of classes
                  LOG_0(Trace_Error, "ERROR DataSetBoosting::InitTargetData data target too big to reference memory");
                  return Error_UnexpectedInternal;
               }
               iData = static_cast<StorageDataType>(data);
               if(countClasses <= static_cast<size_t>(iData)) {
                  LOG_0(Trace_Error, "ERROR DataSetBoosting::InitTargetData target value larger than number of classes");
                  return Error_UnexpectedInternal;
               }
            }
            *pTargetTo = iData;
            ++pTargetTo;

            replication -= direction;
         } while(pTargetToEnd != pTargetTo);
         
         ++pSubset;
      } while(pSubsetsEnd != pSubset);
   } else {
      const FloatFast * pTargetFrom = static_cast<const FloatFast *>(aTargets);
      FloatFast data;
      do {
         const size_t cSubsetSamples = pSubset->m_cSamples;
         EBM_ASSERT(1 <= cSubsetSamples);

         if(IsMultiplyError(sizeof(FloatFast), cSubsetSamples)) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitTargetData IsMultiplyError(sizeof(FloatFast), cSubsetSamples)");
            return Error_OutOfMemory;
         }
         FloatFast * pTargetTo = static_cast<FloatFast *>(AlignedAlloc(sizeof(FloatFast) * cSubsetSamples));
         if(nullptr == pTargetTo) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitTargetData nullptr == pTargetTo");
            return Error_OutOfMemory;
         }
         pSubset->m_aTargetData = pTargetTo;
         const FloatFast * const pTargetToEnd = pTargetTo + cSubsetSamples;
         do {
            if(BagEbm { 0 } == replication) {
               replication = 1;
               if(nullptr != pSampleReplication) {
                  bool isItemValidation;
                  do {
                     do {
                        replication = *pSampleReplication;
                        ++pSampleReplication;
                        ++pTargetFrom;
                     } while(BagEbm { 0 } == replication);
                     isItemValidation = replication < BagEbm { 0 };
                  } while(isLoopValidation != isItemValidation);
                  --pTargetFrom;
               }
               data = *pTargetFrom;
               ++pTargetFrom;
            }
            *pTargetTo = data;
            ++pTargetTo;

            replication -= direction;
         } while(pTargetToEnd != pTargetTo);

         ++pSubset;
      } while(pSubsetsEnd != pSubset);
   }
   EBM_ASSERT(0 == replication);
   LOG_0(Trace_Info, "Exited DataSetBoosting::InitTargetData");
   return Error_None;
}
WARNING_POP

struct FeatureDimension {
   FeatureDimension() = default; // preserve our POD status
   ~FeatureDimension() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   size_t m_cItemsPerBitPackFrom;
   size_t m_cBitsPerItemMaxFrom;
   size_t m_maskBitsFrom;
   ptrdiff_t m_iShiftFrom;

   const SharedStorageDataType * m_pFeatureDataFrom;
   size_t m_cBins;
};
static_assert(std::is_standard_layout<FeatureDimension>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<FeatureDimension>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
ErrorEbm DataSetBoosting::InitTermData(
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const size_t cSharedSamples,
   const BagEbm * const aBag,
   const size_t cTerms,
   const Term * const * const apTerms,
   const IntEbm * const aiTermFeatures
) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::InitTermData");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(1 <= cSharedSamples);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(1 <= cTerms);
   EBM_ASSERT(nullptr != apTerms);

   const DataSubsetBoosting * const pSubsetsEnd = m_aSubsets + m_cSubsets;

   const bool isLoopValidation = direction < BagEbm { 0 };
   const IntEbm * piTermFeature = aiTermFeatures;
   size_t iTerm = 0;
   do {
      const Term * const pTerm = apTerms[iTerm];
      EBM_ASSERT(nullptr != pTerm);
      if(0 == pTerm->GetCountRealDimensions()) {
         // we need to check if there are zero dimensions since if there are then piTermFeatures could be nullptr
         if(0 != pTerm->GetCountDimensions()) {
            EBM_ASSERT(nullptr != piTermFeature); // we would have exited when constructing the terms if nullptr
            piTermFeature += pTerm->GetCountDimensions();
         }
      } else {
         const TermFeature * pTermFeature = pTerm->GetTermFeatures();
         EBM_ASSERT(1 <= pTerm->GetCountDimensions());
         const TermFeature * const pTermFeaturesEnd = &pTermFeature[pTerm->GetCountDimensions()];

         FeatureDimension dimensionInfo[k_cDimensionsMax];
         FeatureDimension * pDimensionInfoInit = dimensionInfo;
         do {
            const FeatureBoosting * const pFeature = pTermFeature->m_pFeature;
            const size_t cBins = pFeature->GetCountBins();
            EBM_ASSERT(size_t { 1 } <= cBins); // we don't construct datasets on empty training sets
            if(size_t { 1 } < cBins) {
               const IntEbm indexFeature = *piTermFeature;
               EBM_ASSERT(!IsConvertError<size_t>(indexFeature)); // we converted it previously
               const size_t iFeature = static_cast<size_t>(indexFeature);

               bool bMissing;
               bool bUnknown;
               bool bNominal;
               bool bSparse;
               SharedStorageDataType cBinsUnused;
               SharedStorageDataType defaultValSparse;
               size_t cNonDefaultsSparse;
               const void * pFeatureDataFrom = GetDataSetSharedFeature(
                  pDataSetShared,
                  iFeature,
                  &bMissing,
                  &bUnknown,
                  &bNominal,
                  &bSparse,
                  &cBinsUnused,
                  &defaultValSparse,
                  &cNonDefaultsSparse
               );
               EBM_ASSERT(nullptr != pFeatureDataFrom);
               EBM_ASSERT(!IsConvertError<size_t>(cBinsUnused)); // since we previously extracted cBins and checked
               EBM_ASSERT(static_cast<size_t>(cBinsUnused) == cBins);
               EBM_ASSERT(!bSparse); // we don't support sparse yet

               pDimensionInfoInit->m_pFeatureDataFrom = static_cast<const SharedStorageDataType *>(pFeatureDataFrom);
               pDimensionInfoInit->m_cBins = cBins;

               const size_t cBitsRequiredMin = CountBitsRequired(cBins - size_t { 1 });
               EBM_ASSERT(1 <= cBitsRequiredMin);
               EBM_ASSERT(cBitsRequiredMin <= k_cBitsForSharedStorageType); // comes from shared data set
               EBM_ASSERT(cBitsRequiredMin <= k_cBitsForSizeT); // since cBins fits into size_t (previous call to GetDataSetSharedFeature)

               const size_t cItemsPerBitPackFrom = GetCountItemsBitPacked<SharedStorageDataType>(cBitsRequiredMin);
               EBM_ASSERT(1 <= cItemsPerBitPackFrom);
               EBM_ASSERT(cItemsPerBitPackFrom <= k_cBitsForSharedStorageType);

               const size_t cBitsPerItemMaxFrom = GetCountBits<SharedStorageDataType>(cItemsPerBitPackFrom);
               EBM_ASSERT(1 <= cBitsPerItemMaxFrom);
               EBM_ASSERT(cBitsPerItemMaxFrom <= k_cBitsForSharedStorageType);

               // we can only guarantee that cBitsPerItemMaxFrom is less than or equal to k_cBitsForSharedStorageType
               // so we need to construct our mask in that type, but afterwards we can convert it to a 
               // size_t since we know the ultimate answer must fit into that since cBins fits into a size_t. If in theory 
               // SharedStorageDataType were allowed to be a billion bits, then the mask could be 65 bits while the end
               // result would be forced to be 64 bits or less since we use the maximum number of bits per item possible
               const size_t maskBitsFrom = static_cast<size_t>(MakeLowMask<SharedStorageDataType>(cBitsPerItemMaxFrom));

               pDimensionInfoInit->m_cItemsPerBitPackFrom = cItemsPerBitPackFrom;
               pDimensionInfoInit->m_cBitsPerItemMaxFrom = cBitsPerItemMaxFrom;
               pDimensionInfoInit->m_maskBitsFrom = maskBitsFrom;
               pDimensionInfoInit->m_iShiftFrom = static_cast<ptrdiff_t>((cSharedSamples - 1) % cItemsPerBitPackFrom);

               ++pDimensionInfoInit;
            }
            ++piTermFeature;
            ++pTermFeature;
         } while(pTermFeaturesEnd != pTermFeature);
         EBM_ASSERT(pDimensionInfoInit == &dimensionInfo[pTerm->GetCountRealDimensions()]);

         EBM_ASSERT(nullptr != aBag || !isLoopValidation); // if aBag is nullptr then we have no validation samples
         const BagEbm * pSampleReplication = aBag;
         BagEbm replication = 0;
         StorageDataType iTensor;

         DataSubsetBoosting * pSubset = m_aSubsets;
         do {
            EBM_ASSERT(1 <= pTerm->GetTermBitPack());
            const size_t cItemsPerBitPackTo = static_cast<size_t>(pTerm->GetTermBitPack());
            EBM_ASSERT(1 <= cItemsPerBitPackTo);
            EBM_ASSERT(cItemsPerBitPackTo <= k_cBitsForStorageType);

            const size_t cBitsPerItemMaxTo = GetCountBits<StorageDataType>(cItemsPerBitPackTo);
            EBM_ASSERT(1 <= cBitsPerItemMaxTo);
            EBM_ASSERT(cBitsPerItemMaxTo <= k_cBitsForStorageType);

            const size_t cSubsetSamples = pSubset->m_cSamples;
            EBM_ASSERT(1 <= cSubsetSamples);

            const size_t cDataUnitsTo = (cSubsetSamples - 1) / cItemsPerBitPackTo + 1; // this can't overflow or underflow

            if(IsMultiplyError(sizeof(StorageDataType), cDataUnitsTo)) {
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitTermData IsMultiplyError(sizeof(StorageDataType), cDataUnitsTo)");
               return Error_OutOfMemory;
            }
            StorageDataType * pTermDataTo = static_cast<StorageDataType *>(AlignedAlloc(sizeof(StorageDataType) * cDataUnitsTo));
            if(nullptr == pTermDataTo) {
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitTermData nullptr == pTermDataTo");
               return Error_OutOfMemory;
            }
            pSubset->m_aaTermData[iTerm] = pTermDataTo;

            const StorageDataType * const pTermDataToEnd = pTermDataTo + cDataUnitsTo;

            ptrdiff_t cShiftTo = static_cast<ptrdiff_t>((cSubsetSamples - 1) % cItemsPerBitPackTo * cBitsPerItemMaxTo);
            const ptrdiff_t cShiftResetTo = static_cast<ptrdiff_t>((cItemsPerBitPackTo - 1) * cBitsPerItemMaxTo);

            do {
               StorageDataType bitsTo = 0;
               do {
                  if(BagEbm { 0 } == replication) {
                     replication = 1;
                     if(nullptr != pSampleReplication) {
                        const BagEbm * pSampleReplicationOriginal = pSampleReplication;
                        bool isItemValidation;
                        do {
                           do {
                              replication = *pSampleReplication;
                              ++pSampleReplication;
                           } while(BagEbm { 0 } == replication);
                           isItemValidation = replication < BagEbm { 0 };
                        } while(isLoopValidation != isItemValidation);
                        const size_t cAdvances = pSampleReplication - pSampleReplicationOriginal - 1;
                        if(0 != cAdvances) {
                           FeatureDimension * pDimensionInfo = dimensionInfo;
                           do {
                              const size_t cItemsPerBitPackFrom = pDimensionInfo->m_cItemsPerBitPackFrom;
                              size_t cCompleteAdvanced = cAdvances / cItemsPerBitPackFrom;
                              ptrdiff_t iShiftFrom = pDimensionInfo->m_iShiftFrom;
                              iShiftFrom -= static_cast<ptrdiff_t>(cAdvances % cItemsPerBitPackFrom);
                              pDimensionInfo->m_iShiftFrom = iShiftFrom;
                              if(iShiftFrom < ptrdiff_t { 0 }) {
                                 pDimensionInfo->m_iShiftFrom = iShiftFrom + cItemsPerBitPackFrom;
                                 ++cCompleteAdvanced;
                              }
                              pDimensionInfo->m_pFeatureDataFrom += cCompleteAdvanced;

                              ++pDimensionInfo;
                           } while(pDimensionInfoInit != pDimensionInfo);
                        }
                     }

                     size_t tensorIndex = 0;
                     size_t tensorMultiple = 1;
                     FeatureDimension * pDimensionInfo = dimensionInfo;
                     do {
                        const SharedStorageDataType * const pFeatureDataFrom = pDimensionInfo->m_pFeatureDataFrom;
                        const SharedStorageDataType bitsFrom = *pFeatureDataFrom;
                        ptrdiff_t iShiftFrom = pDimensionInfo->m_iShiftFrom;
                        EBM_ASSERT(0 <= iShiftFrom);
                        EBM_ASSERT(iShiftFrom * pDimensionInfo->m_cBitsPerItemMaxFrom < k_cBitsForSharedStorageType);
                        const size_t iBin = static_cast<size_t>(bitsFrom >>
                           (iShiftFrom * pDimensionInfo->m_cBitsPerItemMaxFrom)) &
                           pDimensionInfo->m_maskBitsFrom;

                        // we check our dataSet when we get the header, and cBins has been checked to fit into size_t
                        EBM_ASSERT(iBin < pDimensionInfo->m_cBins);

                        --iShiftFrom;
                        pDimensionInfo->m_iShiftFrom = iShiftFrom;
                        if(iShiftFrom < ptrdiff_t { 0 }) {
                           EBM_ASSERT(ptrdiff_t { -1 } == iShiftFrom);
                           pDimensionInfo->m_iShiftFrom = iShiftFrom + pDimensionInfo->m_cItemsPerBitPackFrom;
                           pDimensionInfo->m_pFeatureDataFrom = pFeatureDataFrom + 1;
                        }

                        // we check for overflows during Term construction, but let's check here again
                        EBM_ASSERT(!IsMultiplyError(tensorMultiple, pDimensionInfo->m_cBins));

                        // this can't overflow if the multiplication below doesn't overflow, and we checked for that above
                        tensorIndex += tensorMultiple * iBin;
                        tensorMultiple *= pDimensionInfo->m_cBins;

                        ++pDimensionInfo;
                     } while(pDimensionInfoInit != pDimensionInfo);

                     // during term construction we check that the maximum tensor index fits into StorageDataType
                     EBM_ASSERT(!IsConvertError<StorageDataType>(tensorIndex));
                     iTensor = static_cast<StorageDataType>(tensorIndex);
                  }

                  EBM_ASSERT(0 != replication);
                  EBM_ASSERT(0 < replication && 0 < direction || replication < 0 && direction < 0);
                  replication -= direction;

                  EBM_ASSERT(0 <= cShiftTo);
                  EBM_ASSERT(static_cast<size_t>(cShiftTo) < k_cBitsForStorageType);
                  // the tensor index needs to fit in memory, but concivably StorageDataType does not
                  bitsTo |= iTensor << cShiftTo;
                  cShiftTo -= cBitsPerItemMaxTo;
               } while(ptrdiff_t { 0 } <= cShiftTo);
               cShiftTo = cShiftResetTo;
               *pTermDataTo = bitsTo;
               ++pTermDataTo;
            } while(pTermDataToEnd != pTermDataTo);

            ++pSubset;
         } while(pSubsetsEnd != pSubset);
         EBM_ASSERT(0 == replication);
      }
      ++iTerm;
   } while(cTerms != iTerm);

   LOG_0(Trace_Info, "Exited DataSetBoosting::InitTermData");
   return Error_None;
}
WARNING_POP


ErrorEbm DataSetBoosting::InitBags(
   void * const rng,
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const BagEbm * const aBag,
   const size_t cInnerBags,
   const size_t cWeights
) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::InitBags");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);

   const size_t cIncludedSamples = m_cSamples;
   EBM_ASSERT(1 <= cIncludedSamples);

   const size_t cInnerBagsAfterZero = size_t { 0 } == cInnerBags ? size_t { 1 } : cInnerBags;

   if(IsMultiplyError(sizeof(double), cInnerBagsAfterZero)) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags IsMultiplyError(sizeof(double), cInnerBagsAfterZero))");
      return Error_OutOfMemory;
   }
   double * pBagWeightTotals = static_cast<double *>(malloc(sizeof(double) * cInnerBagsAfterZero));
   if(nullptr == pBagWeightTotals) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == pBagWeightTotals");
      return Error_OutOfMemory;
   }
   m_aBagWeightTotals = pBagWeightTotals;

   // the compiler understands the internal state of this RNG and can locate its internal state into CPU registers
   RandomDeterministic cpuRng;
   size_t * aOccurrencesFrom = nullptr;
   if(size_t { 0 } != cInnerBags) {
      if(nullptr == rng) {
         // Inner bags are not used when building a differentially private model, so
         // we can use low-quality non-determinism.  Generate a non-deterministic seed
         uint64_t seed;
         try {
            RandomNondeterministic<uint64_t> randomGenerator;
            seed = randomGenerator.Next(std::numeric_limits<uint64_t>::max());
         } catch(const std::bad_alloc &) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags Out of memory in std::random_device");
            return Error_OutOfMemory;
         } catch(...) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags Unknown error in std::random_device");
            return Error_UnexpectedInternal;
         }
         cpuRng.Initialize(seed);
      } else {
         const RandomDeterministic * const pRng = reinterpret_cast<RandomDeterministic *>(rng);
         cpuRng.Initialize(*pRng); // move the RNG from memory into CPU registers
      }

      if(IsMultiplyError(sizeof(size_t), cIncludedSamples)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags IsMultiplyError(sizeof(size_t), cIncludedSamples)");
         return Error_OutOfMemory;
      }
      aOccurrencesFrom = static_cast<size_t *>(malloc(sizeof(size_t) * cIncludedSamples));
      if(nullptr == aOccurrencesFrom) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == aCountOccurrences");
         return Error_OutOfMemory;
      }
   }

   const FloatFast * aWeightsFrom = nullptr;
   if(size_t { 0 } != cWeights) {
      aWeightsFrom = GetDataSetSharedWeight(pDataSetShared, 0);
      EBM_ASSERT(nullptr != aWeightsFrom);
      if(CheckWeightsEqual(direction, aBag, aWeightsFrom, cIncludedSamples)) {
         aWeightsFrom = nullptr;
      }
   }

   const bool isLoopValidation = direction < BagEbm { 0 };
   EBM_ASSERT(nullptr != aBag || !isLoopValidation); // if aBag is nullptr then we have no validation samples

   EBM_ASSERT(1 <= m_cSubsets);
   const DataSubsetBoosting * const pSubsetsEnd = m_aSubsets + m_cSubsets;

   size_t iBag = 0;
   do {
      if(nullptr != aOccurrencesFrom) {
         memset(aOccurrencesFrom, 0, sizeof(*aOccurrencesFrom) * cIncludedSamples);

         size_t cSamplesRemaining = cIncludedSamples;
         do {
            const size_t iSample = cpuRng.NextFast(cIncludedSamples);
            ++aOccurrencesFrom[iSample];
            --cSamplesRemaining;
         } while(size_t { 0 } != cSamplesRemaining);
      }

      double total;
      if(nullptr == aWeightsFrom) {
         total = static_cast<double>(cIncludedSamples);
         if(nullptr != aOccurrencesFrom) {
            const size_t * pOccurrencesFrom = aOccurrencesFrom;
            DataSubsetBoosting * pSubset = m_aSubsets;
            do {
               EBM_ASSERT(nullptr != pSubset->m_aInnerBags);
               InnerBag * const pInnerBag = &pSubset->m_aInnerBags[iBag];

               const size_t cSubsetSamples = pSubset->GetCountSamples();
               EBM_ASSERT(1 <= cSubsetSamples);

               if(IsMultiplyError(sizeof(FloatFast), cSubsetSamples)) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags IsMultiplyError(sizeof(FloatFast), cSubsetSamples)");
                  free(aOccurrencesFrom);
                  return Error_OutOfMemory;
               }
               FloatFast * pWeightTo = static_cast<FloatFast *>(AlignedAlloc(sizeof(FloatFast) * cSubsetSamples));
               if(nullptr == pWeightTo) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == pWeightsInternal");
                  free(aOccurrencesFrom);
                  return Error_OutOfMemory;
               }
               pInnerBag->m_aWeights = pWeightTo;

               EBM_ASSERT(cSubsetSamples <= cIncludedSamples);

               size_t * pOccurrencesTo = static_cast<size_t *>(AlignedAlloc(sizeof(size_t) * cSubsetSamples));
               if(nullptr == pOccurrencesTo) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == pOccurrences");
                  free(aOccurrencesFrom);
                  return Error_OutOfMemory;
               }
               pInnerBag->m_aCountOccurrences = pOccurrencesTo;

               const FloatFast * const pWeightsToEnd = pWeightTo + cSubsetSamples;
               do {
                  const size_t cOccurrences = *pOccurrencesFrom;
                  *pOccurrencesTo = cOccurrences;
                  *pWeightTo = static_cast<FloatFast>(cOccurrences);

                  ++pOccurrencesFrom;
                  ++pOccurrencesTo;
                  ++pWeightTo;
               } while(pWeightsToEnd != pWeightTo);

               ++pSubset;
            } while(pSubsetsEnd != pSubset);
         }
      } else {
         const BagEbm * pSampleReplication = aBag;
         const size_t * pOccurrencesFrom = aOccurrencesFrom;
         const FloatFast * pWeightFrom = aWeightsFrom;
         DataSubsetBoosting * pSubset = m_aSubsets;
         total = 0.0;
            
         EBM_ASSERT(nullptr != pSubset->m_aInnerBags);
         InnerBag * pInnerBag = &pSubset->m_aInnerBags[iBag];

         const size_t cSubsetSamplesInit = pSubset->GetCountSamples();
         EBM_ASSERT(1 <= cSubsetSamplesInit);

         if(IsMultiplyError(sizeof(FloatFast), cSubsetSamplesInit)) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags IsMultiplyError(sizeof(FloatFast), cSubsetSamplesInit)");
            free(aOccurrencesFrom);
            return Error_OutOfMemory;
         }
         FloatFast * pWeightTo = static_cast<FloatFast *>(AlignedAlloc(sizeof(FloatFast) * cSubsetSamplesInit));
         if(nullptr == pWeightTo) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == pWeightsInternal");
            free(aOccurrencesFrom);
            return Error_OutOfMemory;
         }
         pInnerBag->m_aWeights = pWeightTo;

         size_t * pOccurrencesTo = nullptr;
         if(nullptr != pOccurrencesFrom) {
            EBM_ASSERT(cSubsetSamplesInit <= cIncludedSamples);
            pOccurrencesTo = static_cast<size_t *>(AlignedAlloc(sizeof(size_t) * cSubsetSamplesInit));
            if(nullptr == pOccurrencesTo) {
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == aCountOccurrences");
               free(aOccurrencesFrom);
               return Error_OutOfMemory;
            }
            pInnerBag->m_aCountOccurrences = pOccurrencesTo;
         }

         const FloatFast * pWeightsToEnd = pWeightTo + cSubsetSamplesInit;

         while(true) {
            BagEbm replication = 1;
            if(nullptr != pSampleReplication) {
               bool isItemValidation;
               do {
                  do {
                     replication = *pSampleReplication;
                     ++pSampleReplication;
                     ++pWeightFrom;
                  } while(BagEbm { 0 } == replication);
                  isItemValidation = replication < BagEbm { 0 };
               } while(isLoopValidation != isItemValidation);
               --pWeightFrom;
            }

            const FloatFast weight = *pWeightFrom;
            ++pWeightFrom;

            do {
               FloatFast result = weight;
               if(nullptr != pOccurrencesFrom) {
                  const size_t cOccurrences = *pOccurrencesFrom;
                  ++pOccurrencesFrom;

                  *pOccurrencesTo = cOccurrences;
                  ++pOccurrencesTo;

                  result *= static_cast<FloatFast>(cOccurrences);
               }

               *pWeightTo = result;
               ++pWeightTo;

               if(pWeightsToEnd == pWeightTo) {
                  total += AddPositiveFloatsSafe<double>(pSubset->GetCountSamples(), pInnerBag->m_aWeights);

                  ++pSubset;
                  if(pSubsetsEnd == pSubset) {
                     EBM_ASSERT(replication == direction);
                     goto next_bag;
                  }

                  EBM_ASSERT(nullptr != pSubset->m_aInnerBags);
                  pInnerBag = &pSubset->m_aInnerBags[iBag];

                  const size_t cSubsetSamples = pSubset->GetCountSamples();
                  EBM_ASSERT(1 <= cSubsetSamples);

                  if(IsMultiplyError(sizeof(FloatFast), cSubsetSamples)) {
                     LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags IsMultiplyError(sizeof(FloatFast), cSubsetSamples)");
                     free(aOccurrencesFrom);
                     return Error_OutOfMemory;
                  }
                  pWeightTo = static_cast<FloatFast *>(AlignedAlloc(sizeof(FloatFast) * cSubsetSamples));
                  if(nullptr == pWeightTo) {
                     LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == pWeightsInternal");
                     free(aOccurrencesFrom);
                     return Error_OutOfMemory;
                  }
                  pInnerBag->m_aWeights = pWeightTo;

                  if(nullptr != pOccurrencesFrom) {
                     EBM_ASSERT(cSubsetSamples <= cIncludedSamples);
                     pOccurrencesTo = static_cast<size_t *>(AlignedAlloc(sizeof(size_t) * cSubsetSamples));
                     if(nullptr == pOccurrencesTo) {
                        LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == aCountOccurrences");
                        free(aOccurrencesFrom);
                        return Error_OutOfMemory;
                     }
                     pInnerBag->m_aCountOccurrences = pOccurrencesTo;
                  }

                  pWeightsToEnd = pWeightTo + cSubsetSamples;
               }

               replication -= direction;
            } while(BagEbm { 0 } != replication);
         }
      next_bag:;

         if(std::isnan(total) || std::isinf(total) || total < std::numeric_limits<double>::min()) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags std::isnan(total) || std::isinf(total) || total < std::numeric_limits<double>::min()");
            free(aOccurrencesFrom);
            return Error_UserParamVal;
         }
      }

      *pBagWeightTotals = total;
      ++pBagWeightTotals;

      ++iBag;
   } while(cInnerBagsAfterZero != iBag);

   if(nullptr != aOccurrencesFrom) {
      if(nullptr != rng) {
         RandomDeterministic * pRng = reinterpret_cast<RandomDeterministic *>(rng);
         pRng->Initialize(cpuRng); // move the RNG from memory into CPU registers
      }
   }

   free(aOccurrencesFrom);

   LOG_0(Trace_Info, "Exited DataSetBoosting::InitBags");

   return Error_None;
}

ErrorEbm DataSetBoosting::InitDataSetBoosting(
   const bool bAllocateGradients,
   const bool bAllocateHessians,
   const bool bAllocateSampleScores,
   const bool bAllocateTargetData,
   void * const rng,
   const size_t cScores,
   const size_t cSubsetItemsMax,
   const ObjectiveWrapper * const pObjective,
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const size_t cSharedSamples,
   const BagEbm * const aBag,
   const double * const aInitScores,
   const size_t cIncludedSamples,
   const size_t cInnerBags,
   const size_t cWeights,
   const size_t cTerms,
   const Term * const * const apTerms,
   const IntEbm * const aiTermFeatures
) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::InitDataSetBoosting");

   ErrorEbm error;

   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(nullptr == m_aSubsets);
   EBM_ASSERT(0 == m_cSubsets);
   EBM_ASSERT(0 == m_cSamples);

   if(0 != cIncludedSamples) {
      m_cSamples = cIncludedSamples;

      // TODO: add this check elsewhere that StorageDataType is used
      EBM_ASSERT(sizeof(StorageDataType) == pObjective->cUIntBytes);
      if(IsMultiplyError(sizeof(StorageDataType *), cTerms)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitDataSetBoosting IsMultiplyError(sizeof(StorageDataType *), cTerms)");
         return Error_OutOfMemory;
      }

      const size_t cSubsets = (cIncludedSamples - size_t { 1 }) / cSubsetItemsMax + size_t { 1 };

      if(IsMultiplyError(sizeof(DataSubsetBoosting), cSubsets)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitDataSetBoosting IsMultiplyError(sizeof(DataSubsetBoosting), cSubsets)");
         return Error_OutOfMemory;
      }
      DataSubsetBoosting * pSubset = static_cast<DataSubsetBoosting *>(malloc(sizeof(DataSubsetBoosting) * cSubsets));
      if(nullptr == pSubset) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitDataSetBoosting nullptr == pSubset");
         return Error_OutOfMemory;
      }
      m_aSubsets = pSubset;
      m_cSubsets = cSubsets;

      EBM_ASSERT(1 <= cSubsets);
      const DataSubsetBoosting * const pSubsetsEnd = pSubset + cSubsets;

      DataSubsetBoosting * pSubsetInit = pSubset;
      do {
         pSubsetInit->SafeInitDataSubsetBoosting();
         ++pSubsetInit;
      } while(pSubsetsEnd != pSubsetInit);

      size_t cIncludedSamplesRemaining = cIncludedSamples;
      do {
         const size_t cSubsetSamples = cIncludedSamplesRemaining <= cSubsetItemsMax ? cIncludedSamplesRemaining : cSubsetItemsMax;
         EBM_ASSERT(1 <= cSubsetSamples);
         pSubset->m_cSamples = cSubsetSamples;

         cIncludedSamplesRemaining -= cSubsetItemsMax; // this will overflow on last loop, but that's ok

         if(0 != cTerms) {
            StorageDataType ** paData = static_cast<StorageDataType **>(malloc(sizeof(StorageDataType *) * cTerms));
            if(nullptr == paData) {
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitDataSetBoosting nullptr == paData");
               return Error_OutOfMemory;
            }

            pSubset->m_aaTermData = paData;

            const StorageDataType * const * const paDataEnd = paData + cTerms;
            do {
               *paData = nullptr;
               ++paData;
            } while(paDataEnd != paData);
         }

         InnerBag * const aInnerBags = InnerBag::AllocateInnerBags(cInnerBags);
         if(nullptr == aInnerBags) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitDataSetBoosting nullptr == aInnerBags");
            return Error_OutOfMemory;
         }
         pSubset->m_aInnerBags = aInnerBags;

         ++pSubset;
      } while(pSubsetsEnd != pSubset);

      if(bAllocateGradients) {
         error = InitGradHess(bAllocateHessians, cScores, pObjective);
         if(Error_None != error) {
            return error;
         }
      } else {
         EBM_ASSERT(!bAllocateHessians);
      }

      if(bAllocateSampleScores) {
         error = InitSampleScores(
            cScores,
            direction,
            aBag,
            aInitScores
         );
         if(Error_None != error) {
            return error;
         }
      }

      if(bAllocateTargetData) {
         error = InitTargetData(
            pDataSetShared,
            direction,
            aBag
         );
         if(Error_None != error) {
            return error;
         }
      }

      if(0 != cTerms) {
         error = InitTermData(
            pDataSetShared,
            direction,
            cSharedSamples,
            aBag,
            cTerms,
            apTerms,
            aiTermFeatures
         );
         if(Error_None != error) {
            return error;
         }
      }

      error = InitBags(
         rng,
         pDataSetShared,
         direction,
         aBag,
         cInnerBags,
         cWeights
      );
      if(Error_None != error) {
         return error;
      }
   }

   LOG_0(Trace_Info, "Exited DataSetBoosting::InitDataSetBoosting");

   return Error_None;
}

void DataSetBoosting::DestructDataSetBoosting(const size_t cTerms, const size_t cInnerBags) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::DestructDataSetBoosting");

   free(m_aBagWeightTotals);

   DataSubsetBoosting * pSubset = m_aSubsets;
   if(nullptr != pSubset) {
      EBM_ASSERT(1 <= m_cSubsets);
      const DataSubsetBoosting * const pSubsetsEnd = pSubset + m_cSubsets;
      do {
         pSubset->DestructDataSubsetBoosting(cTerms, cInnerBags);
         ++pSubset;
      } while(pSubsetsEnd != pSubset);
      free(m_aSubsets);
   }

   LOG_0(Trace_Info, "Exited DataSetBoosting::DestructDataSetBoosting");
}

} // DEFINED_ZONE_NAME
