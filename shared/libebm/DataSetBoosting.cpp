// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_internal.hpp" // SafeConvertFloat
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

void DataSubsetBoosting::DestructDataSubsetBoosting(const size_t cTerms, const size_t cInnerBags) {
   LOG_0(Trace_Info, "Entered DataSubsetBoosting::DestructDataSubsetBoosting");

   InnerBag::FreeInnerBags(cInnerBags, m_aInnerBags);

   void ** paTermData = m_aaTermData;
   if(nullptr != paTermData) {
      EBM_ASSERT(1 <= cTerms);
      const void * const * const paTermDataEnd = paTermData + cTerms;
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
   const size_t cScores
) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::InitGradHess");

   EBM_ASSERT(1 <= cScores);

   size_t cTotalScores = cScores;
   if(bAllocateHessians) {
      if(IsMultiplyError(size_t { 2 }, cTotalScores)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitGradHess IsMultiplyError(size_t { 2 }, cTotalScores)");
         return Error_OutOfMemory;
      }
      cTotalScores = size_t { 2 } * cTotalScores;
   }

   DataSubsetBoosting * pSubset = m_aSubsets;
   const DataSubsetBoosting * const pSubsetsEnd = pSubset + m_cSubsets;
   do {
      const size_t cSubsetSamples = pSubset->m_cSamples;
      EBM_ASSERT(1 <= cSubsetSamples);

      EBM_ASSERT(nullptr != pSubset->m_pObjective);
      if(IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cTotalScores, cSubsetSamples)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitGradHess IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cTotalScores, cSubsetSamples)");
         return Error_OutOfMemory;
      }
      const size_t cBytesGradHess = pSubset->m_pObjective->m_cFloatBytes * cTotalScores * cSubsetSamples;
      ANALYSIS_ASSERT(0 != cBytesGradHess);

      void * const aGradHess = AlignedAlloc(cBytesGradHess);
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
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
WARNING_DISABLE_UNINITIALIZED_LOCAL_POINTER
ErrorEbm DataSetBoosting::InitSampleScores(
   const size_t cScores,
   const BagEbm direction,
   const BagEbm * const aBag,
   const double * const aInitScores
) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::InitSampleScores");

   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(nullptr != aBag || BagEbm { 1 } == direction);  // if aBag is nullptr then we have no validation samples

   DataSubsetBoosting * pSubset = m_aSubsets;
   const DataSubsetBoosting * const pSubsetsEnd = pSubset + m_cSubsets;

   if(nullptr == aInitScores) {
      static_assert(std::numeric_limits<double>::is_iec559, "IEEE 754 guarantees zeros means a zero float");
      do {
         const size_t cSubsetSamples = pSubset->m_cSamples;
         EBM_ASSERT(1 <= cSubsetSamples);

         if(IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cScores, cSubsetSamples)) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitSampleScores IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cScores, cSubsetSamples)");
            return Error_OutOfMemory;
         }
         const size_t cBytes = pSubset->m_pObjective->m_cFloatBytes * cScores * cSubsetSamples;
         ANALYSIS_ASSERT(0 != cBytes);
         void * pSampleScore = AlignedAlloc(cBytes);
         if(nullptr == pSampleScore) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitSampleScores nullptr == pSampleScore");
            return Error_OutOfMemory;
         }
         pSubset->m_aSampleScores = pSampleScore;

         memset(pSampleScore, 0, cBytes);

         ++pSubset;
      } while(pSubsetsEnd != pSubset);
   } else {
      const BagEbm * pSampleReplication = aBag;
      const double * pInitScore;
      const double * pFromEnd = aInitScores;
      const bool isLoopValidation = direction < BagEbm { 0 };
      BagEbm replication = 0;
      do {
         const size_t cSubsetSamples = pSubset->m_cSamples;
         EBM_ASSERT(1 <= cSubsetSamples);

         if(IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cScores, cSubsetSamples)) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitSampleScores IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cScores, cSubsetSamples)");
            return Error_OutOfMemory;
         }
         const size_t cBytes = pSubset->m_pObjective->m_cFloatBytes * cScores * cSubsetSamples;
         ANALYSIS_ASSERT(0 != cBytes);
         void * pSampleScore = AlignedAlloc(cBytes);
         if(nullptr == pSampleScore) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitSampleScores nullptr == pSampleScore");
            return Error_OutOfMemory;
         }
         pSubset->m_aSampleScores = pSampleScore;
         const void * pSampleScoresEnd = IndexByte(pSampleScore, cBytes);

         do {
            if(BagEbm { 0 } == replication) {
               pInitScore = pFromEnd;
               replication = 1;
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
               pFromEnd = &pInitScore[cScores];
            }

            const double * pFrom = pInitScore;
            do {
               if(sizeof(Float_Small) == pSubset->m_pObjective->m_cFloatBytes) {
                  *reinterpret_cast<Float_Small *>(pSampleScore) = SafeConvertFloat<Float_Small>(*pFrom);
               } else {
                  EBM_ASSERT(sizeof(Float_Big) == pSubset->m_pObjective->m_cFloatBytes);
                  *reinterpret_cast<Float_Big *>(pSampleScore) = SafeConvertFloat<Float_Big>(*pFrom);
               }
               pSampleScore = IndexByte(pSampleScore, pSubset->m_pObjective->m_cFloatBytes);
               ++pFrom;
            } while(pFromEnd != pFrom);

            replication -= direction;
         } while(pSampleScoresEnd != pSampleScore);

         ++pSubset;
      } while(pSubsetsEnd != pSubset);
      EBM_ASSERT(0 == replication);
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
      const SharedStorageDataType * pTargetFrom = static_cast<const SharedStorageDataType *>(aTargets);
      SharedStorageDataType iData;
      do {
         const size_t cSubsetSamples = pSubset->m_cSamples;
         EBM_ASSERT(1 <= cSubsetSamples);

         if(IsMultiplyError(pSubset->m_pObjective->m_cUIntBytes, cSubsetSamples)) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitTargetData IsMultiplyError(pSubset->m_pObjective->m_cUIntBytes, cSubsetSamples)");
            return Error_OutOfMemory;
         }
         const size_t cBytes = pSubset->m_pObjective->m_cUIntBytes * cSubsetSamples;
         void * pTargetTo = AlignedAlloc(cBytes);
         if(nullptr == pTargetTo) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitTargetData nullptr == pTargetTo");
            return Error_OutOfMemory;
         }
         pSubset->m_aTargetData = pTargetTo;
         const void * const pTargetToEnd = IndexByte(pTargetTo, cBytes);
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
               iData = *pTargetFrom;
               ++pTargetFrom;

#ifndef NDEBUG
               // this was checked when creating the shared dataset
               EBM_ASSERT(iData < static_cast<SharedStorageDataType>(cClasses));
               EBM_ASSERT(!IsConvertError<size_t>(iData)); // since cClasses came from size_t
               if(sizeof(UInt_Small) == pSubset->m_pObjective->m_cUIntBytes) {
                  // we checked earlier that cClasses - 1 would fit into UInt_Small
                  EBM_ASSERT(!IsConvertError<UInt_Small>(iData));
               } else {
                  // we checked earlier that cClasses - 1 would fit into UInt_Big
                  EBM_ASSERT(sizeof(UInt_Big) == pSubset->m_pObjective->m_cFloatBytes);
                  EBM_ASSERT(!IsConvertError<UInt_Big>(iData));
               }
#endif // NDEBUG
            }
            if(sizeof(UInt_Small) == pSubset->m_pObjective->m_cUIntBytes) {
               *reinterpret_cast<UInt_Small *>(pTargetTo) = static_cast<UInt_Small>(iData);
            } else {
               EBM_ASSERT(sizeof(UInt_Big) == pSubset->m_pObjective->m_cUIntBytes);
               *reinterpret_cast<UInt_Big *>(pTargetTo) = static_cast<UInt_Big>(iData);
            }
            pTargetTo = IndexByte(pTargetTo, pSubset->m_pObjective->m_cUIntBytes);

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

         if(IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cSubsetSamples)) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitTargetData IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cSubsetSamples)");
            return Error_OutOfMemory;
         }
         const size_t cBytes = pSubset->m_pObjective->m_cFloatBytes * cSubsetSamples;
         void * pTargetTo = AlignedAlloc(cBytes);
         if(nullptr == pTargetTo) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitTargetData nullptr == pTargetTo");
            return Error_OutOfMemory;
         }
         pSubset->m_aTargetData = pTargetTo;
         const void * const pTargetToEnd = IndexByte(pTargetTo, cBytes);
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
            if(sizeof(Float_Small) == pSubset->m_pObjective->m_cFloatBytes) {
               *reinterpret_cast<Float_Small *>(pTargetTo) = SafeConvertFloat<Float_Small>(data);
            } else {
               EBM_ASSERT(sizeof(Float_Big) == pSubset->m_pObjective->m_cFloatBytes);
               *reinterpret_cast<Float_Big *>(pTargetTo) = SafeConvertFloat<Float_Big>(data);
            }
            pTargetTo = IndexByte(pTargetTo, pSubset->m_pObjective->m_cFloatBytes);

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
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);
   EBM_ASSERT(1 <= cSharedSamples);
   EBM_ASSERT(1 <= cTerms);
   EBM_ASSERT(nullptr != apTerms);

   EBM_ASSERT(nullptr != m_aSubsets);
   EBM_ASSERT(1 <= m_cSubsets);
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
               EBM_ASSERT(!bSparse); // we don't support sparse yet

               EBM_ASSERT(!IsConvertError<size_t>(cBinsUnused)); // since we previously extracted cBins and checked
               EBM_ASSERT(static_cast<size_t>(cBinsUnused) == cBins);

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
         UIntExceed iTensor;

         DataSubsetBoosting * pSubset = m_aSubsets;
         do {
            const unsigned int cItemsPerBitPackTo =
               GetCountItemsBitPacked(pTerm->GetBitsRequiredMin(), static_cast<unsigned int>(pSubset->GetObjectiveWrapper()->m_cUIntBytes));
            EBM_ASSERT(1 <= cItemsPerBitPackTo);

            const unsigned int cBitsPerItemMaxTo = GetCountBits(cItemsPerBitPackTo, static_cast<unsigned int>(pSubset->GetObjectiveWrapper()->m_cUIntBytes));
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
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitTermData IsMultiplyError(pSubset->GetObjectiveWrapper()->m_cUIntBytes, cDataUnitsTo)");
               return Error_OutOfMemory;
            }
            const size_t cBytes = pSubset->GetObjectiveWrapper()->m_cUIntBytes * cDataUnitsTo;
            void * pTermDataTo = AlignedAlloc(cBytes);
            if(nullptr == pTermDataTo) {
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitTermData nullptr == pTermDataTo");
               return Error_OutOfMemory;
            }
            pSubset->m_aaTermData[iTerm] = pTermDataTo;
            const void * const pTermDataToEnd = IndexByte(pTermDataTo, cBytes);

            memset(pTermDataTo, 0, cBytes);

            int cShiftTo = static_cast<int>((cParallelSamples - size_t { 1 }) % static_cast<size_t>(cItemsPerBitPackTo)) * cBitsPerItemMaxTo;
            const unsigned int cShiftResetTo = (cItemsPerBitPackTo - 1) * cBitsPerItemMaxTo;
            do {
               do {
                  size_t iPartition = 0;
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
                           EBM_ASSERT(static_cast<size_t>(iShiftFrom) * pDimensionInfo->m_cBitsPerItemMaxFrom < k_cBitsForSharedStorageType);
                           const size_t iFeatureBin = static_cast<size_t>(bitsFrom >>
                              (static_cast<size_t>(iShiftFrom) * pDimensionInfo->m_cBitsPerItemMaxFrom)) &
                              pDimensionInfo->m_maskBitsFrom;

                           // we check our dataSet when we get the header, and cBins has been checked to fit into size_t
                           EBM_ASSERT(iFeatureBin < pDimensionInfo->m_cBins);

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
                           tensorIndex += tensorMultiple * iFeatureBin;
                           tensorMultiple *= pDimensionInfo->m_cBins;

                           ++pDimensionInfo;
                        } while(pDimensionInfoInit != pDimensionInfo);

                        EBM_ASSERT(tensorIndex < pTerm->GetCountTensorBins());
                        // during term construction we checked that the maximum tensor index fits into the destination storage
                        iTensor = static_cast<UIntExceed>(tensorIndex);
                     }

                     EBM_ASSERT(0 != replication);
                     EBM_ASSERT(0 < replication && 0 < direction || replication < 0 && direction < 0);
                     replication -= direction;

                     EBM_ASSERT(0 <= cShiftTo);
                     // the tensor index needs to fit in memory, but concivably bitsTo does not
                     const UIntExceed bitsTo = iTensor << cShiftTo;

                     if(sizeof(UInt_Small) == pSubset->m_pObjective->m_cUIntBytes) {
                        *(reinterpret_cast<UInt_Small *>(pTermDataTo) + iPartition) |= static_cast<UInt_Small>(bitsTo);
                     } else {
                        EBM_ASSERT(sizeof(UInt_Big) == pSubset->m_pObjective->m_cUIntBytes);
                        *(reinterpret_cast<UInt_Big *>(pTermDataTo) + iPartition) |= static_cast<UInt_Big>(bitsTo);
                     }

                     ++iPartition;
                  } while(cSIMDPack != iPartition);
                  cShiftTo -= cBitsPerItemMaxTo;
               } while(ptrdiff_t { 0 } <= cShiftTo);
               cShiftTo = cShiftResetTo;

               pTermDataTo = IndexByte(pTermDataTo, pSubset->m_pObjective->m_cUIntBytes * cSIMDPack);
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

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
WARNING_DISABLE_UNINITIALIZED_LOCAL_POINTER
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
   uint8_t * aOccurrencesFrom = nullptr;
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

      if(IsMultiplyError(sizeof(uint8_t), cIncludedSamples)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags IsMultiplyError(sizeof(uint8_t), cIncludedSamples)");
         return Error_OutOfMemory;
      }
      aOccurrencesFrom = static_cast<uint8_t *>(malloc(sizeof(uint8_t) * cIncludedSamples));
      if(nullptr == aOccurrencesFrom) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == aCountOccurrences");
         return Error_OutOfMemory;
      }
   }

   const FloatFast * aWeightsFrom = nullptr;
   if(size_t { 0 } != cWeights) {
      aWeightsFrom = GetDataSetSharedWeight(pDataSetShared, 0);
      EBM_ASSERT(nullptr != aWeightsFrom);
   }

   const bool isLoopValidation = direction < BagEbm { 0 };
   EBM_ASSERT(nullptr != aBag || !isLoopValidation); // if aBag is nullptr then we have no validation samples

   EBM_ASSERT(nullptr != m_aSubsets);
   EBM_ASSERT(1 <= m_cSubsets);
   const DataSubsetBoosting * const pSubsetsEnd = m_aSubsets + m_cSubsets;

   size_t iBag = 0;
   do {
      if(nullptr != aOccurrencesFrom) {
         memset(aOccurrencesFrom, 0, sizeof(*aOccurrencesFrom) * cIncludedSamples);

         size_t cSamplesRemaining = cIncludedSamples;
         do {
            const size_t iSample = cpuRng.NextFast(cIncludedSamples);
            const uint8_t existing = aOccurrencesFrom[iSample];
            if(std::numeric_limits<uint8_t>::max() == existing) {
               // it should be essentially impossible for sampling with replacement to get to 255 items in the bin
               // but check it anyways..
               continue;
            }
            aOccurrencesFrom[iSample] = existing + uint8_t { 1 };
            --cSamplesRemaining;
         } while(size_t { 0 } != cSamplesRemaining);
      }

      double totalWeight;
      if(nullptr == aWeightsFrom) {
         totalWeight = static_cast<double>(cIncludedSamples);
         if(nullptr != aOccurrencesFrom) {
            const uint8_t * pOccurrencesFrom = aOccurrencesFrom;
            DataSubsetBoosting * pSubset = m_aSubsets;
            do {
               EBM_ASSERT(nullptr != pSubset->m_aInnerBags);
               InnerBag * const pInnerBag = &pSubset->m_aInnerBags[iBag];

               const size_t cSubsetSamples = pSubset->GetCountSamples();
               EBM_ASSERT(1 <= cSubsetSamples);

               if(IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cSubsetSamples)) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cSubsetSamples)");
                  free(aOccurrencesFrom);
                  return Error_OutOfMemory;
               }
               const size_t cBytes = pSubset->m_pObjective->m_cFloatBytes * cSubsetSamples;
               void * pWeightTo = AlignedAlloc(cBytes);
               if(nullptr == pWeightTo) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == pWeightsInternal");
                  free(aOccurrencesFrom);
                  return Error_OutOfMemory;
               }
               pInnerBag->m_aWeights = pWeightTo;

               EBM_ASSERT(cSubsetSamples <= cIncludedSamples);

               uint8_t * pOccurrencesTo = static_cast<uint8_t *>(AlignedAlloc(sizeof(uint8_t) * cSubsetSamples));
               if(nullptr == pOccurrencesTo) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == pOccurrences");
                  free(aOccurrencesFrom);
                  return Error_OutOfMemory;
               }
               pInnerBag->m_aCountOccurrences = pOccurrencesTo;

               const void * const pWeightsToEnd = IndexByte(pWeightTo, cBytes);
               do {
                  const uint8_t cOccurrences = *pOccurrencesFrom;
                  *pOccurrencesTo = cOccurrences;

                  if(sizeof(Float_Small) == pSubset->m_pObjective->m_cFloatBytes) {
                     *reinterpret_cast<Float_Small *>(pWeightTo) = static_cast<Float_Small>(cOccurrences);
                  } else {
                     EBM_ASSERT(sizeof(Float_Big) == pSubset->m_pObjective->m_cFloatBytes);
                     *reinterpret_cast<Float_Big *>(pWeightTo) = static_cast<Float_Big>(cOccurrences);
                  }
                  pWeightTo = IndexByte(pWeightTo, pSubset->m_pObjective->m_cFloatBytes);

                  ++pOccurrencesFrom;
                  ++pOccurrencesTo;
               } while(pWeightsToEnd != pWeightTo);

               ++pSubset;
            } while(pSubsetsEnd != pSubset);
         }
      } else {
         const uint8_t * pOccurrencesFrom = aOccurrencesFrom;
         const BagEbm * pSampleReplication = aBag;
         const FloatFast * pWeightFrom = aWeightsFrom;
         DataSubsetBoosting * pSubset = m_aSubsets;
         totalWeight = 0.0;

         BagEbm replication = 0;
         double weight;
         do {
            const size_t cSubsetSamples = pSubset->GetCountSamples();
            EBM_ASSERT(1 <= cSubsetSamples);

            if(IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cSubsetSamples)) {
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cSubsetSamples)");
               free(aOccurrencesFrom);
               return Error_OutOfMemory;
            }
            const size_t cBytes = pSubset->m_pObjective->m_cFloatBytes * cSubsetSamples;
            void * pWeightTo = AlignedAlloc(cBytes);
            if(nullptr == pWeightTo) {
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == pWeightTo");
               free(aOccurrencesFrom);
               return Error_OutOfMemory;
            }
            EBM_ASSERT(nullptr != pSubset->m_aInnerBags);
            InnerBag * pInnerBag = &pSubset->m_aInnerBags[iBag];
            pInnerBag->m_aWeights = pWeightTo;

            uint8_t * pOccurrencesTo;
            if(nullptr != pOccurrencesFrom) {
               EBM_ASSERT(cSubsetSamples <= cIncludedSamples);
               pOccurrencesTo = static_cast<uint8_t *>(AlignedAlloc(sizeof(uint8_t) * cSubsetSamples));
               if(nullptr == pOccurrencesTo) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == aCountOccurrences");
                  free(aOccurrencesFrom);
                  return Error_OutOfMemory;
               }
               pInnerBag->m_aCountOccurrences = pOccurrencesTo;
            }

            const void * const pWeightsToEnd = IndexByte(pWeightTo, cBytes);

            // add the weights in 2 stages to preserve precision
            double subsetWeight = 0.0;
            do {
               if(BagEbm { 0 } == replication) {
                  replication = 1;
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

                  weight = SafeConvertFloat<double>(*pWeightFrom);
                  ++pWeightFrom;

                  // these were checked when creating the shared dataset
                  EBM_ASSERT(!std::isnan(weight));
                  EBM_ASSERT(!std::isinf(weight));
                  EBM_ASSERT(static_cast<double>(std::numeric_limits<float>::min()) <= weight);
                  EBM_ASSERT(weight <= static_cast<double>(std::numeric_limits<float>::max()));
               }

               double result = weight;
               if(nullptr != pOccurrencesFrom) {
                  EBM_ASSERT(nullptr != pOccurrencesTo);
                  const uint8_t cOccurrences = *pOccurrencesFrom;
                  ++pOccurrencesFrom;

                  *pOccurrencesTo = cOccurrences;
                  ++pOccurrencesTo;

                  result *= static_cast<double>(cOccurrences);
               }

               subsetWeight += result;

               if(sizeof(Float_Small) == pSubset->m_pObjective->m_cFloatBytes) {
                  *reinterpret_cast<Float_Small *>(pWeightTo) = SafeConvertFloat<Float_Small>(result);
               } else {
                  EBM_ASSERT(sizeof(Float_Big) == pSubset->m_pObjective->m_cFloatBytes);
                  *reinterpret_cast<Float_Big *>(pWeightTo) = SafeConvertFloat<Float_Big>(result);
               }
               pWeightTo = IndexByte(pWeightTo, pSubset->m_pObjective->m_cFloatBytes);

               replication -= direction;
            } while(pWeightsToEnd != pWeightTo);

            totalWeight += subsetWeight;

            ++pSubset;
         } while(pSubsetsEnd != pSubset);
         EBM_ASSERT(0 == replication);

         EBM_ASSERT(!std::isnan(totalWeight));
         EBM_ASSERT(std::numeric_limits<double>::min() <= totalWeight);

         if(std::isinf(totalWeight)) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags std::isinf(total)");
            free(aOccurrencesFrom);
            return Error_UserParamVal;
         }
      }

      *pBagWeightTotals = totalWeight;
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
WARNING_POP

ErrorEbm DataSetBoosting::InitDataSetBoosting(
   const bool bAllocateGradients,
   const bool bAllocateHessians,
   const bool bAllocateSampleScores,
   const bool bAllocateTargetData,
   void * const rng,
   const size_t cScores,
   const size_t cSubsetItemsMax,
   const ObjectiveWrapper * const pObjectiveCpu,
   const ObjectiveWrapper * const pObjectiveSIMD,
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
   EBM_ASSERT(1 <= cSubsetItemsMax);
   EBM_ASSERT(nullptr != pObjectiveCpu);
   EBM_ASSERT(nullptr != pObjectiveCpu->m_pObjective);
   EBM_ASSERT(nullptr != pObjectiveSIMD);
   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbm { -1 } == direction || BagEbm { 1 } == direction);

   EBM_ASSERT(0 == m_cSamples);
   EBM_ASSERT(0 == m_cSubsets);
   EBM_ASSERT(nullptr == m_aSubsets);
   EBM_ASSERT(nullptr == m_aBagWeightTotals);

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

      const DataSubsetBoosting * const pSubsetsEnd = pSubset + cSubsets;

      DataSubsetBoosting * pSubsetInit = pSubset;
      do {
         pSubsetInit->SafeInitDataSubsetBoosting();
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

         if(0 != cTerms) {
            if(IsMultiplyError(sizeof(void *), cTerms)) {
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitDataSetBoosting IsMultiplyError(sizeof(void *), cTerms)");
               return Error_OutOfMemory;
            }
            void ** paTermData = static_cast<void **>(malloc(sizeof(void *) * cTerms));
            if(nullptr == paTermData) {
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitDataSetBoosting nullptr == paTermData");
               return Error_OutOfMemory;
            }
            pSubset->m_aaTermData = paTermData;

            const void * const * const paTermDataEnd = paTermData + cTerms;
            do {
               *paTermData = nullptr;
               ++paTermData;
            } while(paTermDataEnd != paTermData);
         }

         InnerBag * const aInnerBags = InnerBag::AllocateInnerBags(cInnerBags);
         if(nullptr == aInnerBags) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitDataSetBoosting nullptr == aInnerBags");
            return Error_OutOfMemory;
         }
         pSubset->m_aInnerBags = aInnerBags;

         ++pSubset;
      } while(pSubsetsEnd != pSubset);
      EBM_ASSERT(0 == cIncludedSamplesRemaining);

      if(bAllocateGradients) {
         error = InitGradHess(bAllocateHessians, cScores);
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
