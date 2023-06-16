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
   const size_t cSetSamples
);

ErrorEbm DataSetBoosting::ConstructGradientsAndHessians(const bool bAllocateHessians, const size_t cScores) {
   LOG_0(Trace_Info, "Entered ConstructGradientsAndHessians");

   EBM_ASSERT(1 <= cScores);

   const size_t cStorageItems = bAllocateHessians ? size_t { 2 } : size_t { 1 };
   if(IsMultiplyError(sizeof(FloatFast) * cStorageItems, cScores)) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructGradientsAndHessians IsMultiplyError(sizeof(FloatFast) * cStorageItems, cScores)");
      return Error_OutOfMemory;
   }
   const size_t cElementBytes = sizeof(FloatFast) * cStorageItems * cScores;

   DataSubsetBoosting * pSubset = m_aSubsets;
   const DataSubsetBoosting * const pSubsetsEnd = pSubset + m_cSubsets;
   do {
      const size_t cSubsetSamples = pSubset->m_cSamples;
      EBM_ASSERT(1 <= cSubsetSamples);

      if(IsMultiplyError(cElementBytes, cSubsetSamples)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructGradientsAndHessians IsMultiplyError(cElementBytes, cSubsetSamples)");
         return Error_OutOfMemory;
      }
      const size_t cBytesGradientsAndHessians = cElementBytes * cSubsetSamples;
      ANALYSIS_ASSERT(0 != cBytesGradientsAndHessians);

      FloatFast * const aGradientsAndHessians = static_cast<FloatFast *>(malloc(cBytesGradientsAndHessians));
      if(nullptr == aGradientsAndHessians) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructGradientsAndHessians nullptr == aGradientsAndHessians");
         return Error_OutOfMemory;
      }
      pSubset->m_aGradientsAndHessians = aGradientsAndHessians;

      ++pSubset;
   } while(pSubsetsEnd != pSubset);

   LOG_0(Trace_Info, "Exited ConstructGradientsAndHessians");
   return Error_None;
}

WARNING_PUSH
// NOTE: This warning seems to be flagged by the DEBUG 32 bit build
WARNING_BUFFER_OVERRUN
ErrorEbm DataSetBoosting::ConstructSampleScores(
   const size_t cScores,
   const BagEbm direction,
   const BagEbm * const aBag,
   const double * const aInitScores
) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::ConstructSampleScores");

   DataSubsetBoosting * pSubset = m_aSubsets;
   const DataSubsetBoosting * const pSubsetsEnd = pSubset + m_cSubsets;

   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(nullptr != aBag || BagEbm { 1 } == direction);  // if aBag is nullptr then we have no validation samples

   if(IsMultiplyError(sizeof(FloatFast), cScores)) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructSampleScores IsMultiplyError(sizeof(FloatFast), cScores)");
      return Error_OutOfMemory;
   }
   const size_t cBytesOneElement = sizeof(FloatFast) * cScores;

   if(nullptr == aInitScores) {
      static_assert(std::numeric_limits<FloatFast>::is_iec559, "IEEE 754 guarantees zeros means a zero float");
      do {
         const size_t cSubsetSamples = pSubset->m_cSamples;
         EBM_ASSERT(1 <= cSubsetSamples);

         if(IsMultiplyError(cBytesOneElement, cSubsetSamples)) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructSampleScores IsMultiplyError(cBytesOneElement, cSubsetSamples)");
            return Error_OutOfMemory;
         }
         const size_t cBytes = cBytesOneElement * cSubsetSamples;
         ANALYSIS_ASSERT(0 != cBytes);
         FloatFast * pSampleScore = static_cast<FloatFast *>(malloc(cBytes));
         if(nullptr == pSampleScore) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructSampleScores nullptr == pSampleScore");
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
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructSampleScores IsMultiplyError(cBytesOneElement, cSubsetSamplesInit)");
         return Error_OutOfMemory;
      }
      const size_t cBytesInit = cBytesOneElement * cSubsetSamplesInit;
      ANALYSIS_ASSERT(0 != cBytesInit);
      FloatFast * pSampleScore = static_cast<FloatFast *>(malloc(cBytesInit));
      if(nullptr == pSampleScore) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructSampleScores nullptr == pSampleScore");
         return Error_OutOfMemory;
      }
      pSubset->m_aSampleScores = pSampleScore;
      const FloatFast * pSampleScoresEnd = IndexByte(pSampleScore, cBytesInit);


      const BagEbm * pSampleReplication = aBag;
      const double * pInitScore = aInitScores;
      const bool isLoopTraining = BagEbm { 0 } < direction;
      while(true) {
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
                  LOG_0(Trace_Info, "Exited DataSetBoosting::ConstructTargetData");
                  return Error_None;
               }

               const size_t cSubsetSamples = pSubset->m_cSamples;
               EBM_ASSERT(1 <= cSubsetSamples);

               if(IsMultiplyError(cBytesOneElement, cSubsetSamples)) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructSampleScores IsMultiplyError(cBytesOneElement, cSubsetSamples)");
                  return Error_OutOfMemory;
               }
               const size_t cBytes = cBytesOneElement * cSubsetSamples;
               pSampleScore = static_cast<FloatFast *>(malloc(cBytes));
               if(nullptr == pSampleScore) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructSampleScores nullptr == pSampleScore");
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

   LOG_0(Trace_Info, "Exited DataSetBoosting::ConstructSampleScores");
   return Error_None;
}
WARNING_POP


ErrorEbm DataSetBoosting::ConstructTargetData(
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const BagEbm * const aBag
) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::ConstructTargetData");

   DataSubsetBoosting * pSubset = m_aSubsets;
   const DataSubsetBoosting * const pSubsetsEnd = pSubset + m_cSubsets;
   const size_t cSubsetSamplesInit = pSubset->m_cSamples;
   EBM_ASSERT(1 <= cSubsetSamplesInit);

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);

   ptrdiff_t cClasses;
   const void * const aTargets = GetDataSetSharedTarget(pDataSetShared, 0, &cClasses);
   EBM_ASSERT(nullptr != aTargets); // we previously called GetDataSetSharedTarget and got back non-null result

   const BagEbm * pSampleReplication = aBag;
   const bool isLoopTraining = BagEbm { 0 } < direction;
   EBM_ASSERT(nullptr != aBag || isLoopTraining); // if aBag is nullptr then we have no validation samples

   if(IsClassification(cClasses)) {
      const size_t countClasses = static_cast<size_t>(cClasses);
      const SharedStorageDataType * pTargetFrom = static_cast<const SharedStorageDataType *>(aTargets);


      if(IsMultiplyError(sizeof(StorageDataType), cSubsetSamplesInit)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructTargetData IsMultiplyError(sizeof(StorageDataType), cSubsetSamplesInit)");
         return Error_OutOfMemory;
      }
      StorageDataType * pTargetTo = static_cast<StorageDataType *>(malloc(sizeof(StorageDataType) * cSubsetSamplesInit));
      if(nullptr == pTargetTo) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructTargetData nullptr == pTargetTo");
         return Error_OutOfMemory;
      }
      pSubset->m_aTargetData = pTargetTo;
      const StorageDataType * pTargetToEnd = pTargetTo + cSubsetSamplesInit;

      while(true) {
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
            return Error_UnexpectedInternal;
         }
         const StorageDataType iData = static_cast<StorageDataType>(data);
         if(countClasses <= static_cast<size_t>(iData)) {
            LOG_0(Trace_Error, "ERROR DataSetBoosting::ConstructTargetData target value larger than number of classes");
            return Error_UnexpectedInternal;
         }
         do {
            *pTargetTo = iData;
            ++pTargetTo;

            if(pTargetToEnd == pTargetTo) {
               ++pSubset;
               if(pSubsetsEnd == pSubset) {
                  EBM_ASSERT(replication == direction);
                  LOG_0(Trace_Info, "Exited DataSetBoosting::ConstructTargetData");
                  return Error_None;
               }

               const size_t cSubsetSamples = pSubset->m_cSamples;
               EBM_ASSERT(1 <= cSubsetSamples);

               if(IsMultiplyError(sizeof(StorageDataType), cSubsetSamples)) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructTargetData IsMultiplyError(sizeof(StorageDataType), cSubsetSamples)");
                  return Error_OutOfMemory;
               }
               pTargetTo = static_cast<StorageDataType *>(malloc(sizeof(StorageDataType) * cSubsetSamples));
               if(nullptr == pTargetTo) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructTargetData nullptr == pTargetTo");
                  return Error_OutOfMemory;
               }
               pSubset->m_aTargetData = pTargetTo;
               pTargetToEnd = pTargetTo + cSubsetSamples;
            }

            replication -= direction;
         } while(BagEbm { 0 } != replication);
      }
   } else {
      const FloatFast * pTargetFrom = static_cast<const FloatFast *>(aTargets);


      if(IsMultiplyError(sizeof(FloatFast), cSubsetSamplesInit)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructTargetData IsMultiplyError(sizeof(FloatFast), cSubsetSamplesInit)");
         return Error_OutOfMemory;
      }
      FloatFast * pTargetTo = static_cast<FloatFast *>(malloc(sizeof(FloatFast) * cSubsetSamplesInit));
      if(nullptr == pTargetTo) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructTargetData nullptr == pTargetTo");
         return Error_OutOfMemory;
      }
      pSubset->m_aTargetData = pTargetTo;
      const FloatFast * pTargetToEnd = pTargetTo + cSubsetSamplesInit;

      while(true) {
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
         const FloatFast data = *pTargetFrom;
         ++pTargetFrom;

         // TODO : our caller should handle NaN *pTargetFrom values, which means that the target is missing, which means we should delete that sample 
         //   from the input data

         // if data is NaN, we pass this along and NaN propagation will ensure that we stop boosting immediately.
         // There is no need to check it here since we already have graceful detection later for other reasons.

         // TODO: NaN target values essentially mean missing, so we should be filtering those samples out, but our caller should do that so 
         //   that we don't need to do the work here per outer bag.  Our job in C++ is just not to crash or return inexplicable values.

         do {
            *pTargetTo = data;
            ++pTargetTo;

            if(pTargetToEnd == pTargetTo) {
               ++pSubset;
               if(pSubsetsEnd == pSubset) {
                  EBM_ASSERT(replication == direction);
                  LOG_0(Trace_Info, "Exited DataSetBoosting::ConstructTargetData");
                  return Error_None;
               }

               const size_t cSubsetSamples = pSubset->m_cSamples;
               EBM_ASSERT(1 <= cSubsetSamples);

               if(IsMultiplyError(sizeof(FloatFast), cSubsetSamples)) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructTargetData IsMultiplyError(sizeof(FloatFast), cSubsetSamples)");
                  return Error_OutOfMemory;
               }
               pTargetTo = static_cast<FloatFast *>(malloc(sizeof(FloatFast) * cSubsetSamples));
               if(nullptr == pTargetTo) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructTargetData nullptr == pTargetTo");
                  return Error_OutOfMemory;
               }
               pSubset->m_aTargetData = pTargetTo;
               pTargetToEnd = pTargetTo + cSubsetSamples;
            }

            replication -= direction;
         } while(BagEbm { 0 } != replication);
      }
   }
}

struct InputDataPointerAndCountBins {

   InputDataPointerAndCountBins() = default; // preserve our POD status
   ~InputDataPointerAndCountBins() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   size_t m_cItemsPerBitPackFrom;
   size_t m_cBitsPerItemMaxFrom;
   size_t m_maskBitsFrom;
   ptrdiff_t m_iShiftFrom;

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
ErrorEbm DataSetBoosting::ConstructInputData(
   const unsigned char * const pDataSetShared,
   const size_t cSharedSamples,
   const BagEbm direction,
   const BagEbm * const aBag,
   const IntEbm * const aiTermFeatures,
   const size_t cTerms,
   const Term * const * const apTerms
) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::ConstructInputData");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(1 <= cTerms);
   EBM_ASSERT(nullptr != apTerms);

   const DataSubsetBoosting * const pSubsetsEnd = m_aSubsets + m_cSubsets;

   const bool isLoopTraining = BagEbm { 0 } < direction;
   const IntEbm * piTermFeatures = aiTermFeatures;
   size_t iTerm = 0;
   do {
      const Term * const pTerm = apTerms[iTerm];
      EBM_ASSERT(nullptr != pTerm);
      if(0 == pTerm->GetCountRealDimensions()) {
         piTermFeatures += pTerm->GetCountDimensions();
      } else {
         const TermFeature * pTermFeature = pTerm->GetTermFeatures();
         EBM_ASSERT(1 <= pTerm->GetCountDimensions());
         const TermFeature * const pTermFeaturesEnd = &pTermFeature[pTerm->GetCountDimensions()];

         InputDataPointerAndCountBins dimensionInfo[k_cDimensionsMax];
         InputDataPointerAndCountBins * pDimensionInfoInit = &dimensionInfo[0];
         do {
            const FeatureBoosting * const pFeature = pTermFeature->m_pFeature;
            const size_t cBins = pFeature->GetCountBins();
            EBM_ASSERT(size_t { 1 } <= cBins); // we don't construct datasets on empty training sets
            if(size_t { 1 } < cBins) {
               const IntEbm indexFeature = *piTermFeatures;
               EBM_ASSERT(!IsConvertError<size_t>(indexFeature)); // we converted it previously
               const size_t iFeature = static_cast<size_t>(indexFeature);

               bool bMissing;
               bool bUnknown;
               bool bNominal;
               bool bSparse;
               SharedStorageDataType cBinsUnused;
               SharedStorageDataType defaultValSparse;
               size_t cNonDefaultsSparse;
               const void * pInputDataFrom = GetDataSetSharedFeature(
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
               EBM_ASSERT(nullptr != pInputDataFrom);
               EBM_ASSERT(!IsConvertError<size_t>(cBinsUnused)); // since we previously extracted cBins and checked
               EBM_ASSERT(static_cast<size_t>(cBinsUnused) == cBins);
               EBM_ASSERT(!bSparse); // we don't support sparse yet

               pDimensionInfoInit->m_pInputData = static_cast<const SharedStorageDataType *>(pInputDataFrom);
               pDimensionInfoInit->m_cBins = cBins;

               const size_t cBitsRequiredMin = CountBitsRequired(cBins - size_t { 1 });
               EBM_ASSERT(1 <= cBitsRequiredMin);
               EBM_ASSERT(cBitsRequiredMin <= k_cBitsForSharedStorageType); // comes from shared data set
               EBM_ASSERT(cBitsRequiredMin <= k_cBitsForSizeT); // since cBins fits into size_t (previous call to GetDataSetSharedFeature)
               // we previously calculated the tensor bin count, and with that determined that the total number of
               // bins minus one (the maximum tensor index) would fit into a StorageDataType. Since any particular
               // dimensional index must be less than the multiple of all of them, we know that the number of bits
               // will fit into a StorageDataType
               EBM_ASSERT(cBitsRequiredMin <= k_cBitsForStorageType);

               const size_t cItemsPerBitPackFrom = GetCountItemsBitPacked<SharedStorageDataType>(cBitsRequiredMin);
               EBM_ASSERT(1 <= cItemsPerBitPackFrom);
               EBM_ASSERT(cItemsPerBitPackFrom <= k_cBitsForSharedStorageType);

               const size_t cBitsPerItemMaxFrom = GetCountBits<SharedStorageDataType>(cItemsPerBitPackFrom);
               EBM_ASSERT(1 <= cBitsPerItemMaxFrom);
               EBM_ASSERT(cBitsPerItemMaxFrom <= k_cBitsForSharedStorageType);

               // we can only guarantee that cBitsPerItemMaxFrom is less than or equal to k_cBitsForSharedStorageType
               // so we need to construct our mask in that type, but afterwards we can convert it to a 
               // StorageDataType since we know the ultimate answer must fit into that. If in theory 
               // SharedStorageDataType were allowed to be a billion bits, then the mask could be 65 bits while the end
               // result would be forced to be 64 bits or less since we use the maximum number of bits per item possible
               const size_t maskBitsFrom = static_cast<size_t>(MakeLowMask<SharedStorageDataType>(cBitsPerItemMaxFrom));

               pDimensionInfoInit->m_cItemsPerBitPackFrom = cItemsPerBitPackFrom;
               pDimensionInfoInit->m_cBitsPerItemMaxFrom = cBitsPerItemMaxFrom;
               pDimensionInfoInit->m_maskBitsFrom = maskBitsFrom;
               pDimensionInfoInit->m_iShiftFrom = static_cast<ptrdiff_t>((cSharedSamples - 1) % cItemsPerBitPackFrom);

               ++pDimensionInfoInit;
            }
            ++piTermFeatures;
            ++pTermFeature;
         } while(pTermFeaturesEnd != pTermFeature);
         EBM_ASSERT(pDimensionInfoInit == &dimensionInfo[pTerm->GetCountRealDimensions()]);

         EBM_ASSERT(nullptr != aBag || isLoopTraining); // if aBag is nullptr then we have no validation samples
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
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructInputData IsMultiplyError(sizeof(StorageDataType), cDataUnitsTo)");
               return Error_OutOfMemory;
            }
            StorageDataType * pInputDataTo = static_cast<StorageDataType *>(malloc(sizeof(StorageDataType) * cDataUnitsTo));
            if(nullptr == pInputDataTo) {
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::ConstructInputData nullptr == pInputDataTo");
               return Error_OutOfMemory;
            }
            pSubset->m_aaInputData[iTerm] = pInputDataTo;

            const StorageDataType * const pInputDataToEnd = pInputDataTo + cDataUnitsTo;

            ptrdiff_t cShiftTo = static_cast<ptrdiff_t>((cSubsetSamples - 1) % cItemsPerBitPackTo * cBitsPerItemMaxTo);
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
                              size_t cCompleteAdvanced = cAdvances / pDimensionInfo->m_cItemsPerBitPackFrom;
                              pDimensionInfo->m_iShiftFrom -= static_cast<ptrdiff_t>(cAdvances % pDimensionInfo->m_cItemsPerBitPackFrom);
                              if(pDimensionInfo->m_iShiftFrom < ptrdiff_t { 0 }) {
                                 ++cCompleteAdvanced;
                                 pDimensionInfo->m_iShiftFrom += pDimensionInfo->m_cItemsPerBitPackFrom;
                              }
                              pDimensionInfo->m_pInputData += cCompleteAdvanced;

                              ++pDimensionInfo;
                           } while(pDimensionInfoInit != pDimensionInfo);
                        }
                     }

                     size_t tensorIndex = 0;
                     size_t tensorMultiple = 1;
                     InputDataPointerAndCountBins * pDimensionInfo = &dimensionInfo[0];
                     do {
                        const SharedStorageDataType indexDataCombined = *pDimensionInfo->m_pInputData;
                        EBM_ASSERT(pDimensionInfo->m_iShiftFrom * pDimensionInfo->m_cBitsPerItemMaxFrom < k_cBitsForSharedStorageType);
                        const size_t iData = static_cast<size_t>(indexDataCombined >>
                           (pDimensionInfo->m_iShiftFrom * pDimensionInfo->m_cBitsPerItemMaxFrom)) &
                           pDimensionInfo->m_maskBitsFrom;

                        // we check our dataSet when we get the header, and cBins has been checked to fit into size_t
                        EBM_ASSERT(iData < pDimensionInfo->m_cBins);

                        pDimensionInfo->m_iShiftFrom -= 1;
                        if(pDimensionInfo->m_iShiftFrom < ptrdiff_t { 0 }) {
                           pDimensionInfo->m_pInputData += 1;
                           pDimensionInfo->m_iShiftFrom += pDimensionInfo->m_cItemsPerBitPackFrom;
                        }

                        // we check for overflows during Term construction, but let's check here again
                        EBM_ASSERT(!IsMultiplyError(tensorMultiple, pDimensionInfo->m_cBins));

                        // this can't overflow if the multiplication below doesn't overflow, and we checked for that above
                        tensorIndex += tensorMultiple * iData;
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
                  bits |= iTensor << cShiftTo;
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
      ++iTerm;
   } while(cTerms != iTerm);

   LOG_0(Trace_Info, "Exited DataSetBoosting::ConstructInputData");
   return Error_None;
}
WARNING_POP

void DataSubsetBoosting::Destruct(const size_t cTerms, const size_t cInnerBags) {
   LOG_0(Trace_Info, "Entered DataSubsetBoosting::Destruct");

   InnerBag::FreeInnerBags(cInnerBags, m_aInnerBags);

   free(m_aGradientsAndHessians);
   free(m_aSampleScores);
   free(m_aTargetData);

   if(nullptr != m_aaInputData) {
      EBM_ASSERT(1 <= cTerms);
      StorageDataType * * paInputData = m_aaInputData;
      const StorageDataType * const * const paInputDataEnd = m_aaInputData + cTerms;
      do {
         free(*paInputData);
         ++paInputData;
      } while(paInputDataEnd != paInputData);
      free(m_aaInputData);
   }

   LOG_0(Trace_Info, "Exited DataSubsetBoosting::Destruct");
}


ErrorEbm DataSetBoosting::InitializeBags(
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const BagEbm * const aBag,
   void * const rng,
   const size_t cInnerBags,
   const size_t cWeights
) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::InitializeBags");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);

   const size_t cSetSamples = m_cSamples;
   EBM_ASSERT(1 <= cSetSamples);

   const size_t cInnerBagsAfterZero = size_t { 0 } == cInnerBags ? size_t { 1 } : cInnerBags;

   if(IsMultiplyError(sizeof(double), cInnerBagsAfterZero)) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitializeBags IsMultiplyError(sizeof(double), cInnerBagsAfterZero))");
      return Error_OutOfMemory;
   }
   double * pBagWeightTotals = static_cast<double *>(malloc(sizeof(double) * cInnerBagsAfterZero));
   if(nullptr == pBagWeightTotals) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitializeBags nullptr == pBagWeightTotals");
      return Error_OutOfMemory;
   }
   m_aBagWeightTotals = pBagWeightTotals;

   // the compiler understands the internal state of this RNG and can locate its internal state into CPU registers
   RandomDeterministic cpuRng;
   size_t * aCountOccurrences = nullptr;
   if(size_t { 0 } != cInnerBags) {
      if(nullptr == rng) {
         // Inner bags are not used when building a differentially private model, so
         // we can use low-quality non-determinism.  Generate a non-deterministic seed
         uint64_t seed;
         try {
            RandomNondeterministic<uint64_t> randomGenerator;
            seed = randomGenerator.Next(std::numeric_limits<uint64_t>::max());
         } catch(const std::bad_alloc &) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitializeBags Out of memory in std::random_device");
            return Error_OutOfMemory;
         } catch(...) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitializeBags Unknown error in std::random_device");
            return Error_UnexpectedInternal;
         }
         cpuRng.Initialize(seed);
      } else {
         const RandomDeterministic * const pRng = reinterpret_cast<RandomDeterministic *>(rng);
         cpuRng.Initialize(*pRng); // move the RNG from memory into CPU registers
      }

      if(IsMultiplyError(sizeof(size_t), cSetSamples)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitializeBags IsMultiplyError(sizeof(size_t), cSetSamples)");
         return Error_OutOfMemory;
      }
      aCountOccurrences = static_cast<size_t *>(malloc(sizeof(size_t) * cSetSamples));
      if(nullptr == aCountOccurrences) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitializeBags nullptr == aCountOccurrences");
         return Error_OutOfMemory;
      }
   }

   const FloatFast * aWeights = nullptr;
   if(size_t { 0 } != cWeights) {
      aWeights = GetDataSetSharedWeight(pDataSetShared, 0);
      EBM_ASSERT(nullptr != aWeights);
      if(CheckWeightsEqual(direction, aBag, aWeights, cSetSamples)) {
         aWeights = nullptr;
      }
   }

   const bool isLoopTraining = BagEbm { 0 } < direction;
   EBM_ASSERT(nullptr != aBag || isLoopTraining); // if aBag is nullptr then we have no validation samples

   EBM_ASSERT(1 <= m_cSubsets);
   const DataSubsetBoosting * const pSubsetsEnd = m_aSubsets + m_cSubsets;

   size_t iBag = 0;
   do {
      if(nullptr != aCountOccurrences) {
         memset(aCountOccurrences, 0, sizeof(*aCountOccurrences) * cSetSamples);

         size_t iSample = 0;
         do {
            const size_t iCountOccurrences = cpuRng.NextFast(cSetSamples);
            ++aCountOccurrences[iCountOccurrences];
            ++iSample;
         } while(cSetSamples != iSample);
      }

      double total;
      if(nullptr == aWeights) {
         total = static_cast<double>(cSetSamples);
         if(nullptr != aCountOccurrences) {
            const size_t * pCountOccurrences = aCountOccurrences;
            DataSubsetBoosting * pSubset = m_aSubsets;
            do {
               EBM_ASSERT(nullptr != pSubset->m_aInnerBags);
               InnerBag * const pInnerBag = &pSubset->m_aInnerBags[iBag];

               const size_t cSubsetSamples = pSubset->GetCountSamples();
               EBM_ASSERT(1 <= cSubsetSamples);

               if(IsMultiplyError(sizeof(FloatFast), cSubsetSamples)) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitializeBags IsMultiplyError(sizeof(FloatFast), cSubsetSamples)");
                  free(aCountOccurrences);
                  return Error_OutOfMemory;
               }
               FloatFast * pWeightsInternal = static_cast<FloatFast *>(malloc(sizeof(FloatFast) * cSubsetSamples));
               if(nullptr == pWeightsInternal) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitializeBags nullptr == pWeightsInternal");
                  free(aCountOccurrences);
                  return Error_OutOfMemory;
               }
               pInnerBag->m_aWeights = pWeightsInternal;

               EBM_ASSERT(cSubsetSamples <= cSetSamples);

               size_t * pOccurrences = static_cast<size_t *>(malloc(sizeof(size_t) * cSubsetSamples));
               if(nullptr == pOccurrences) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitializeBags nullptr == pOccurrences");
                  free(aCountOccurrences);
                  return Error_OutOfMemory;
               }
               pInnerBag->m_aCountOccurrences = pOccurrences;

               const FloatFast * const pWeightsInternalEnd = pWeightsInternal + cSubsetSamples;
               do {
                  const size_t cOccurrences = *pCountOccurrences;
                  *pOccurrences = cOccurrences;
                  *pWeightsInternal = static_cast<FloatFast>(cOccurrences);

                  ++pCountOccurrences;
                  ++pOccurrences;
                  ++pWeightsInternal;
               } while(pWeightsInternalEnd != pWeightsInternal);

               ++pSubset;
            } while(pSubsetsEnd != pSubset);
         }
      } else {
         const BagEbm * pSampleReplication = aBag;
         const size_t * pCountOccurrences = aCountOccurrences;
         const FloatFast * pWeight = aWeights;
         DataSubsetBoosting * pSubset = m_aSubsets;
         total = 0.0;
            
         EBM_ASSERT(nullptr != pSubset->m_aInnerBags);
         InnerBag * pInnerBag = &pSubset->m_aInnerBags[iBag];

         const size_t cSubsetSamplesInit = pSubset->GetCountSamples();
         EBM_ASSERT(1 <= cSubsetSamplesInit);

         if(IsMultiplyError(sizeof(FloatFast), cSubsetSamplesInit)) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitializeBags IsMultiplyError(sizeof(FloatFast), cSubsetSamplesInit)");
            free(aCountOccurrences);
            return Error_OutOfMemory;
         }
         FloatFast * pWeightsInternal = static_cast<FloatFast *>(malloc(sizeof(FloatFast) * cSubsetSamplesInit));
         if(nullptr == pWeightsInternal) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitializeBags nullptr == pWeightsInternal");
            free(aCountOccurrences);
            return Error_OutOfMemory;
         }
         pInnerBag->m_aWeights = pWeightsInternal;

         size_t * pOccurrences = nullptr;
         if(nullptr != pCountOccurrences) {
            EBM_ASSERT(cSubsetSamplesInit <= cSetSamples);
            pOccurrences = static_cast<size_t *>(malloc(sizeof(size_t) * cSubsetSamplesInit));
            if(nullptr == pOccurrences) {
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitializeBags nullptr == aCountOccurrences");
               free(aCountOccurrences);
               return Error_OutOfMemory;
            }
            pInnerBag->m_aCountOccurrences = pOccurrences;
         }

         const FloatFast * pWeightsInternalEnd = pWeightsInternal + cSubsetSamplesInit;

         while(true) {
            BagEbm replication = 1;
            if(nullptr != pSampleReplication) {
               bool isItemTraining;
               do {
                  do {
                     replication = *pSampleReplication;
                     ++pSampleReplication;
                     ++pWeight;
                  } while(BagEbm { 0 } == replication);
                  isItemTraining = BagEbm { 0 } < replication;
               } while(isLoopTraining != isItemTraining);
               --pWeight;
            }

            const FloatFast weight = *pWeight;
            ++pWeight;

            do {
               FloatFast result = weight;
               if(nullptr != pCountOccurrences) {
                  const size_t cOccurrences = *pCountOccurrences;
                  ++pCountOccurrences;

                  *pOccurrences = cOccurrences;
                  ++pOccurrences;

                  result *= static_cast<FloatFast>(cOccurrences);
               }

               *pWeightsInternal = result;
               ++pWeightsInternal;

               if(pWeightsInternalEnd == pWeightsInternal) {
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
                     LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitializeBags IsMultiplyError(sizeof(FloatFast), cSubsetSamples)");
                     free(aCountOccurrences);
                     return Error_OutOfMemory;
                  }
                  pWeightsInternal = static_cast<FloatFast *>(malloc(sizeof(FloatFast) * cSubsetSamples));
                  if(nullptr == pWeightsInternal) {
                     LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitializeBags nullptr == pWeightsInternal");
                     free(aCountOccurrences);
                     return Error_OutOfMemory;
                  }
                  pInnerBag->m_aWeights = pWeightsInternal;

                  if(nullptr != pCountOccurrences) {
                     EBM_ASSERT(cSubsetSamples <= cSetSamples);
                     pOccurrences = static_cast<size_t *>(malloc(sizeof(size_t) * cSubsetSamples));
                     if(nullptr == pOccurrences) {
                        LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitializeBags nullptr == aCountOccurrences");
                        free(aCountOccurrences);
                        return Error_OutOfMemory;
                     }
                     pInnerBag->m_aCountOccurrences = pOccurrences;
                  }

                  pWeightsInternalEnd = pWeightsInternal + cSubsetSamples;
               }

               replication -= direction;
            } while(BagEbm { 0 } != replication);
         }
      next_bag:;

         if(std::isnan(total) || std::isinf(total) || total < std::numeric_limits<double>::min()) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitializeBags std::isnan(total) || std::isinf(total) || total < std::numeric_limits<double>::min()");
            free(aCountOccurrences);
            return Error_UserParamVal;
         }
      }

      *pBagWeightTotals = total;
      ++pBagWeightTotals;

      ++iBag;
   } while(cInnerBagsAfterZero != iBag);

   if(nullptr != aCountOccurrences) {
      if(nullptr != rng) {
         RandomDeterministic * pRng = reinterpret_cast<RandomDeterministic *>(rng);
         pRng->Initialize(cpuRng); // move the RNG from memory into CPU registers
      }
   }

   free(aCountOccurrences);

   LOG_0(Trace_Info, "Exited DataSetBoosting::InitializeBags");

   return Error_None;
}

ErrorEbm DataSetBoosting::Initialize(
   const size_t cSubsetItemsMax,
   const size_t cScores,
   const bool bAllocateGradients,
   const bool bAllocateHessians,
   const bool bAllocateSampleScores,
   const bool bAllocateTargetData,
   const unsigned char * const pDataSetShared,
   const size_t cSharedSamples,
   const BagEbm direction,
   const BagEbm * const aBag,
   const double * const aInitScores,
   const size_t cSetSamples,
   void * const rng,
   const size_t cInnerBags,
   const size_t cWeights,
   const IntEbm * const aiTermFeatures,
   const size_t cTerms,
   const Term * const * const apTerms
) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::Initialize");

   ErrorEbm error;

   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(nullptr == m_aSubsets);
   EBM_ASSERT(0 == m_cSubsets);
   EBM_ASSERT(0 == m_cSamples);

   if(0 != cSetSamples) {
      m_cSamples = cSetSamples;

      if(IsMultiplyError(sizeof(StorageDataType *), cTerms)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::Initialize IsMultiplyError(sizeof(StorageDataType *), cTerms)");
         return Error_OutOfMemory;
      }

      const size_t cSubsets = (cSetSamples - size_t { 1 }) / cSubsetItemsMax + size_t { 1 };

      if(IsMultiplyError(sizeof(DataSubsetBoosting), cSubsets)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::Initialize IsMultiplyError(sizeof(DataSubsetBoosting), cSubsets)");
         return Error_OutOfMemory;
      }
      DataSubsetBoosting * aSubsets = static_cast<DataSubsetBoosting *>(malloc(sizeof(DataSubsetBoosting) * cSubsets));
      if(nullptr == aSubsets) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::Initialize nullptr == aSubsets");
         return Error_OutOfMemory;
      }
      m_aSubsets = aSubsets;
      m_cSubsets = cSubsets;

      EBM_ASSERT(1 <= cSubsets);
      const DataSubsetBoosting * const pSubsetsEnd = aSubsets + cSubsets;

      DataSubsetBoosting * pSubsetUnfailingInit = aSubsets;
      do {
         pSubsetUnfailingInit->InitializeUnfailing();
         ++pSubsetUnfailingInit;
      } while(pSubsetsEnd != pSubsetUnfailingInit);

      size_t cSetSamplesRemaining = cSetSamples;
      DataSubsetBoosting * pSubsetInit = aSubsets;
      do {
         const size_t cSubsetSamples = cSetSamplesRemaining <= cSubsetItemsMax ? cSetSamplesRemaining : cSubsetItemsMax;
         EBM_ASSERT(1 <= cSubsetSamples);
         pSubsetInit->m_cSamples = cSubsetSamples;

         cSetSamplesRemaining -= cSubsetItemsMax; // this will overflow on last loop, but that's ok

         if(0 != cTerms) {
            StorageDataType ** paData = static_cast<StorageDataType **>(malloc(sizeof(StorageDataType *) * cTerms));
            if(nullptr == paData) {
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::Initialize nullptr == paData");
               return Error_OutOfMemory;
            }

            pSubsetInit->m_aaInputData = paData;

            const StorageDataType * const * const paDataEnd = paData + cTerms;
            do {
               *paData = nullptr;
               ++paData;
            } while(paDataEnd != paData);
         }

         InnerBag * const aInnerBags = InnerBag::AllocateInnerBags(cInnerBags);
         if(nullptr == aInnerBags) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::Initialize nullptr == aInnerBags");
            return Error_OutOfMemory;
         }
         pSubsetInit->m_aInnerBags = aInnerBags;

         ++pSubsetInit;
      } while(pSubsetsEnd != pSubsetInit);

      if(bAllocateGradients) {
         error = ConstructGradientsAndHessians(bAllocateHessians, cScores);
         if(Error_None != error) {
            return error;
         }
      } else {
         EBM_ASSERT(!bAllocateHessians);
      }

      if(bAllocateSampleScores) {
         error = ConstructSampleScores(
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
         error = ConstructTargetData(
            pDataSetShared,
            direction,
            aBag
         );
         if(Error_None != error) {
            return error;
         }
      }

      if(0 != cTerms) {
         error = ConstructInputData(
            pDataSetShared,
            cSharedSamples,
            direction,
            aBag,
            aiTermFeatures,
            cTerms,
            apTerms
         );
         if(Error_None != error) {
            return error;
         }
      }

      error = InitializeBags(
         pDataSetShared,
         direction,
         aBag,
         rng,
         cInnerBags,
         cWeights
      );
      if(Error_None != error) {
         return error;
      }
   }

   LOG_0(Trace_Info, "Exited DataSetBoosting::Initialize");

   return Error_None;
}

void DataSetBoosting::Destruct(const size_t cTerms, const size_t cInnerBags) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::Destruct");

   DataSubsetBoosting * pSubset = m_aSubsets;
   if(nullptr != pSubset) {
      EBM_ASSERT(1 <= m_cSubsets);
      const DataSubsetBoosting * const pSubsetsEnd = pSubset + m_cSubsets;
      do {
         pSubset->Destruct(cTerms, cInnerBags);
         ++pSubset;
      } while(pSubsetsEnd != pSubset);
      free(m_aSubsets);
   }

   free(m_aBagWeightTotals);

   LOG_0(Trace_Info, "Exited DataSetBoosting::Destruct");
}

} // DEFINED_ZONE_NAME
