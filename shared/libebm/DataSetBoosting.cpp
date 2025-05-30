// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#define ZONE_main
#include "zones.h"

#include "ebm_internal.hpp"
#include "RandomDeterministic.hpp" // RandomDeterministic
#include "RandomNondeterministic.hpp" // RandomNondeterministic
#include "Feature.hpp" // Feature
#include "Term.hpp" // Term
#include "dataset_shared.hpp" // UIntShared
#include "DataSetBoosting.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

void DataSubsetBoosting::DestructDataSubsetBoosting(const size_t cTerms, const size_t cInnerBags) {
   LOG_0(Trace_Info, "Entered DataSubsetBoosting::DestructDataSubsetBoosting");

   SubsetInnerBag::FreeSubsetInnerBags(cInnerBags, m_aSubsetInnerBags);

   void** paTermData = m_aaTermData;
   if(nullptr != paTermData) {
      EBM_ASSERT(1 <= cTerms);
      const void* const* const paTermDataEnd = paTermData + cTerms;
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

ErrorEbm DataSetBoosting::InitGradHess(const bool bAllocateHessians, const size_t cScores) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::InitGradHess");

   EBM_ASSERT(1 <= cScores);

   size_t cTotalScores = cScores;
   if(bAllocateHessians) {
      if(IsMultiplyError(size_t{2}, cTotalScores)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitGradHess IsMultiplyError(size_t { 2 }, cTotalScores)");
         return Error_OutOfMemory;
      }
      cTotalScores = cTotalScores << 1;
   }

   EBM_ASSERT(nullptr != m_aSubsets);
   EBM_ASSERT(1 <= m_cSubsets);
   DataSubsetBoosting* pSubset = m_aSubsets;
   const DataSubsetBoosting* const pSubsetsEnd = pSubset + m_cSubsets;
   do {
      const size_t cSubsetSamples = pSubset->m_cSamples;
      EBM_ASSERT(1 <= cSubsetSamples);

      EBM_ASSERT(nullptr != pSubset->m_pObjective);
      if(IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cTotalScores, cSubsetSamples)) {
         LOG_0(Trace_Warning,
               "WARNING DataSetBoosting::InitGradHess IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, "
               "cTotalScores, cSubsetSamples)");
         return Error_OutOfMemory;
      }
      const size_t cBytesGradHess = pSubset->m_pObjective->m_cFloatBytes * cTotalScores * cSubsetSamples;
      ANALYSIS_ASSERT(0 != cBytesGradHess);

      void* const aGradHess = AlignedAlloc(cBytesGradHess);
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
ErrorEbm DataSetBoosting::InitSampleScores(const size_t cScores,
      const double* const aIntercept,
      const BagEbm direction,
      const BagEbm* const aBag,
      const double* const aInitScores) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::InitSampleScores");

   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(BagEbm{-1} == direction || BagEbm{1} == direction);
   EBM_ASSERT(nullptr != aBag || BagEbm{1} == direction); // if aBag is nullptr then we have no validation samples

   DataSubsetBoosting* pSubset = m_aSubsets;
   EBM_ASSERT(nullptr != pSubset);
   EBM_ASSERT(1 <= m_cSubsets);
   const DataSubsetBoosting* const pSubsetsEnd = pSubset + m_cSubsets;

   const BagEbm* pSampleReplication = aBag;
   const double* pInitScore = nullptr;
   const double* pFromEnd = aInitScores;
   const bool isLoopValidation = direction < BagEbm{0};
   BagEbm replication = 0;
   do {
      const size_t cSubsetSamples = pSubset->m_cSamples;
      EBM_ASSERT(1 <= cSubsetSamples);

      const size_t cSIMDPack = pSubset->GetObjectiveWrapper()->m_cSIMDPack;
      EBM_ASSERT(1 <= cSIMDPack);
      EBM_ASSERT(0 == cSubsetSamples % cSIMDPack);

      if(IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cScores, cSubsetSamples)) {
         LOG_0(Trace_Warning,
               "WARNING DataSetBoosting::InitSampleScores IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, "
               "cScores, cSubsetSamples)");
         return Error_OutOfMemory;
      }
      const size_t cBytes = pSubset->m_pObjective->m_cFloatBytes * cScores * cSubsetSamples;
      ANALYSIS_ASSERT(0 != cBytes);
      void* pSampleScore = AlignedAlloc(cBytes);
      if(nullptr == pSampleScore) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitSampleScores nullptr == pSampleScore");
         return Error_OutOfMemory;
      }
      pSubset->m_aSampleScores = pSampleScore;
      const void* pSampleScoresEnd = IndexByte(pSampleScore, cBytes);

      do {
         size_t iPartition = 0;
         do {
            if(BagEbm{0} == replication) {
               replication = 1;
               size_t cAdvance = cScores;
               if(nullptr != pSampleReplication) {
                  cAdvance = 0;
                  bool isItemValidation;
                  do {
                     do {
                        replication = *pSampleReplication;
                        ++pSampleReplication;
                     } while(BagEbm{0} == replication);
                     isItemValidation = replication < BagEbm{0};
                     cAdvance += cScores;
                  } while(isLoopValidation != isItemValidation);
               }
               if(nullptr != pFromEnd) {
                  pFromEnd += cAdvance;
                  pInitScore = pFromEnd - cScores;
               }
            }

            size_t iScore = 0;
            do {
               double score = 0.0;
               if(nullptr != aIntercept) {
                  score = aIntercept[iScore];
               }
               if(nullptr != pInitScore) {
                  score += pInitScore[iScore];
               }

               if(sizeof(FloatBig) == pSubset->m_pObjective->m_cFloatBytes) {
                  reinterpret_cast<FloatBig*>(pSampleScore)[iScore * cSIMDPack + iPartition] =
                        static_cast<FloatBig>(score);
               } else {
                  EBM_ASSERT(sizeof(FloatSmall) == pSubset->m_pObjective->m_cFloatBytes);
                  reinterpret_cast<FloatSmall*>(pSampleScore)[iScore * cSIMDPack + iPartition] =
                        static_cast<FloatSmall>(score);
               }

               ++iScore;
            } while(cScores != iScore);

            replication -= direction;

            ++iPartition;
         } while(cSIMDPack != iPartition);
         pSampleScore = IndexByte(pSampleScore, cScores * pSubset->GetObjectiveWrapper()->m_cFloatBytes * cSIMDPack);
      } while(pSampleScoresEnd != pSampleScore);

      ++pSubset;
   } while(pSubsetsEnd != pSubset);
   EBM_ASSERT(0 == replication);

   LOG_0(Trace_Info, "Exited DataSetBoosting::InitSampleScores");
   return Error_None;
}
WARNING_POP

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
ErrorEbm DataSetBoosting::InitTargetData(
      const unsigned char* const pDataSetShared, const BagEbm direction, const BagEbm* const aBag) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::InitTargetData");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbm{-1} == direction || BagEbm{1} == direction);

   ptrdiff_t cClasses;
   const void* const aTargets = GetDataSetSharedTarget(pDataSetShared, 0, &cClasses);
   EBM_ASSERT(nullptr != aTargets); // we previously called GetDataSetSharedTarget and got back non-null result

   EBM_ASSERT(nullptr != m_aSubsets);
   EBM_ASSERT(1 <= m_cSubsets);
   DataSubsetBoosting* pSubset = m_aSubsets;
   const DataSubsetBoosting* const pSubsetsEnd = pSubset + m_cSubsets;

   const BagEbm* pSampleReplication = aBag;
   const bool isLoopValidation = direction < BagEbm{0};
   EBM_ASSERT(nullptr != aBag || !isLoopValidation); // if aBag is nullptr then we have no validation samples

   BagEbm replication = 0;
   if(ptrdiff_t{Task_GeneralClassification} <= cClasses) {
      const UIntShared* pTargetFrom = static_cast<const UIntShared*>(aTargets);
      UIntShared iData;
      do {
         const size_t cSubsetSamples = pSubset->m_cSamples;
         EBM_ASSERT(1 <= cSubsetSamples);

         if(IsMultiplyError(pSubset->m_pObjective->m_cUIntBytes, cSubsetSamples)) {
            LOG_0(Trace_Warning,
                  "WARNING DataSetBoosting::InitTargetData IsMultiplyError(pSubset->m_pObjective->m_cUIntBytes, "
                  "cSubsetSamples)");
            return Error_OutOfMemory;
         }
         const size_t cBytes = pSubset->m_pObjective->m_cUIntBytes * cSubsetSamples;
         void* pTargetTo = AlignedAlloc(cBytes);
         if(nullptr == pTargetTo) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitTargetData nullptr == pTargetTo");
            return Error_OutOfMemory;
         }
         pSubset->m_aTargetData = pTargetTo;
         const void* const pTargetToEnd = IndexByte(pTargetTo, cBytes);
         do {
            if(BagEbm{0} == replication) {
               replication = 1;
               if(nullptr != pSampleReplication) {
                  bool isItemValidation;
                  do {
                     do {
                        replication = *pSampleReplication;
                        ++pSampleReplication;
                        ++pTargetFrom;
                     } while(BagEbm{0} == replication);
                     isItemValidation = replication < BagEbm{0};
                  } while(isLoopValidation != isItemValidation);
                  --pTargetFrom;
               }
               iData = *pTargetFrom;
               ++pTargetFrom;

#ifndef NDEBUG
               // this was checked when creating the shared dataset
               EBM_ASSERT(iData < static_cast<UIntShared>(cClasses));
               EBM_ASSERT(!IsConvertError<size_t>(iData)); // since cClasses came from size_t
               if(sizeof(UIntBig) == pSubset->m_pObjective->m_cUIntBytes) {
                  // we checked earlier that cClasses - 1 would fit into UIntBig
                  EBM_ASSERT(!IsConvertError<UIntBig>(iData));
               } else {
                  EBM_ASSERT(sizeof(UIntSmall) == pSubset->m_pObjective->m_cUIntBytes);
                  // we checked earlier that cClasses - 1 would fit into UIntSmall
                  EBM_ASSERT(!IsConvertError<UIntSmall>(iData));
               }
#endif // NDEBUG
            }
            if(sizeof(UIntBig) == pSubset->m_pObjective->m_cUIntBytes) {
               *reinterpret_cast<UIntBig*>(pTargetTo) = static_cast<UIntBig>(iData);
            } else {
               EBM_ASSERT(sizeof(UIntSmall) == pSubset->m_pObjective->m_cUIntBytes);
               *reinterpret_cast<UIntSmall*>(pTargetTo) = static_cast<UIntSmall>(iData);
            }
            pTargetTo = IndexByte(pTargetTo, pSubset->m_pObjective->m_cUIntBytes);

            replication -= direction;
         } while(pTargetToEnd != pTargetTo);

         ++pSubset;
      } while(pSubsetsEnd != pSubset);
      EBM_ASSERT(0 == replication);
   } else {
      const FloatShared* pTargetFrom = static_cast<const FloatShared*>(aTargets);
      FloatShared data;
      do {
         const size_t cSubsetSamples = pSubset->m_cSamples;
         EBM_ASSERT(1 <= cSubsetSamples);

         if(IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cSubsetSamples)) {
            LOG_0(Trace_Warning,
                  "WARNING DataSetBoosting::InitTargetData IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, "
                  "cSubsetSamples)");
            return Error_OutOfMemory;
         }
         const size_t cBytes = pSubset->m_pObjective->m_cFloatBytes * cSubsetSamples;
         void* pTargetTo = AlignedAlloc(cBytes);
         if(nullptr == pTargetTo) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitTargetData nullptr == pTargetTo");
            return Error_OutOfMemory;
         }
         pSubset->m_aTargetData = pTargetTo;
         const void* const pTargetToEnd = IndexByte(pTargetTo, cBytes);
         do {
            if(BagEbm{0} == replication) {
               replication = 1;
               if(nullptr != pSampleReplication) {
                  bool isItemValidation;
                  do {
                     do {
                        replication = *pSampleReplication;
                        ++pSampleReplication;
                        ++pTargetFrom;
                     } while(BagEbm{0} == replication);
                     isItemValidation = replication < BagEbm{0};
                  } while(isLoopValidation != isItemValidation);
                  --pTargetFrom;
               }
               data = *pTargetFrom;
               ++pTargetFrom;
            }
            if(sizeof(FloatBig) == pSubset->m_pObjective->m_cFloatBytes) {
               *reinterpret_cast<FloatBig*>(pTargetTo) = static_cast<FloatBig>(data);
            } else {
               EBM_ASSERT(sizeof(FloatSmall) == pSubset->m_pObjective->m_cFloatBytes);
               *reinterpret_cast<FloatSmall*>(pTargetTo) = static_cast<FloatSmall>(data);
            }
            pTargetTo = IndexByte(pTargetTo, pSubset->m_pObjective->m_cFloatBytes);

            replication -= direction;
         } while(pTargetToEnd != pTargetTo);

         ++pSubset;
      } while(pSubsetsEnd != pSubset);
      EBM_ASSERT(0 == replication);
   }
   LOG_0(Trace_Info, "Exited DataSetBoosting::InitTargetData");
   return Error_None;
}
WARNING_POP

struct FeatureDimension {
   FeatureDimension() = default; // preserve our POD status
   ~FeatureDimension() = default; // preserve our POD status
   void* operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete(void*) = delete; // we only use malloc/free in this library

   const UIntShared* m_pFeatureDataFrom;
   size_t m_maskBitsFrom;
   size_t m_cBins;
   int m_cItemsPerBitPackFrom;
   int m_cBitsPerItemMaxFrom;
   int m_iShiftFrom;
};
static_assert(std::is_standard_layout<FeatureDimension>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<FeatureDimension>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
ErrorEbm DataSetBoosting::InitTermData(const unsigned char* const pDataSetShared,
      const BagEbm direction,
      const size_t cSharedSamples,
      const BagEbm* const aBag,
      const size_t cTerms,
      const Term* const* const apTerms,
      const IntEbm* const aiTermFeatures) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::InitTermData");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbm{-1} == direction || BagEbm{1} == direction);
   EBM_ASSERT(1 <= cSharedSamples);
   EBM_ASSERT(1 <= cTerms);
   EBM_ASSERT(nullptr != apTerms);

   EBM_ASSERT(nullptr != m_aSubsets);
   EBM_ASSERT(1 <= m_cSubsets);
   const DataSubsetBoosting* const pSubsetsEnd = m_aSubsets + m_cSubsets;

   const bool isLoopValidation = direction < BagEbm{0};
   EBM_ASSERT(nullptr != aBag || !isLoopValidation); // if aBag is nullptr then we have no validation samples
   const IntEbm* piTermFeature = aiTermFeatures;
   size_t iTerm = 0;
   do {
      const Term* const pTerm = apTerms[iTerm];
      EBM_ASSERT(nullptr != pTerm);
      if(0 == pTerm->GetCountRealDimensions()) {
         // we need to check if there are zero dimensions since if there are then piTermFeatures could be nullptr
         if(0 != pTerm->GetCountDimensions()) {
            EBM_ASSERT(nullptr != piTermFeature); // we would have exited when constructing the terms if nullptr
            piTermFeature += pTerm->GetCountDimensions();
         }
      } else {
         const TermFeature* pTermFeature = pTerm->GetTermFeatures();
         EBM_ASSERT(1 <= pTerm->GetCountDimensions());
         const TermFeature* const pTermFeaturesEnd = &pTermFeature[pTerm->GetCountDimensions()];

         FeatureDimension dimensionInfo[k_cDimensionsMax];
         FeatureDimension* pDimensionInfoInit = dimensionInfo;
         do {
            const FeatureBoosting* const pFeature = pTermFeature->m_pFeature;
            const size_t cBins = pFeature->GetCountBins();
            EBM_ASSERT(size_t{1} <= cBins); // we don't construct datasets on empty training sets
            if(size_t{1} < cBins) {
               const IntEbm indexFeature = *piTermFeature;
               EBM_ASSERT(!IsConvertError<size_t>(indexFeature)); // we converted it previously
               const size_t iFeature = static_cast<size_t>(indexFeature);

               bool bMissing;
               bool bUnseen;
               bool bNominal;
               bool bSparse;
               UIntShared cBinsUnused;
               UIntShared defaultValSparse;
               size_t cNonDefaultsSparse;
               const void* pFeatureDataFrom = GetDataSetSharedFeature(pDataSetShared,
                     iFeature,
                     &bMissing,
                     &bUnseen,
                     &bNominal,
                     &bSparse,
                     &cBinsUnused,
                     &defaultValSparse,
                     &cNonDefaultsSparse);
               EBM_ASSERT(nullptr != pFeatureDataFrom);
               EBM_ASSERT(!bSparse); // we don't support sparse yet

               EBM_ASSERT(!IsConvertError<size_t>(cBinsUnused)); // since we previously extracted cBins and checked
               EBM_ASSERT(static_cast<size_t>(cBinsUnused) == cBins);

               pDimensionInfoInit->m_pFeatureDataFrom = static_cast<const UIntShared*>(pFeatureDataFrom);
               pDimensionInfoInit->m_cBins = cBins;

               const int cBitsRequiredMin = CountBitsRequired(cBins - size_t{1});
               EBM_ASSERT(1 <= cBitsRequiredMin);
               EBM_ASSERT(cBitsRequiredMin <= COUNT_BITS(UIntShared)); // comes from shared data set
               // since cBins fits into size_t (previous call to GetDataSetSharedFeature)
               EBM_ASSERT(cBitsRequiredMin <= COUNT_BITS(size_t));

               const int cItemsPerBitPackFrom = GetCountItemsBitPacked<UIntShared>(cBitsRequiredMin);
               EBM_ASSERT(1 <= cItemsPerBitPackFrom);
               EBM_ASSERT(cItemsPerBitPackFrom <= COUNT_BITS(UIntShared));

               const int cBitsPerItemMaxFrom = GetCountBits<UIntShared>(cItemsPerBitPackFrom);
               EBM_ASSERT(1 <= cBitsPerItemMaxFrom);
               EBM_ASSERT(cBitsPerItemMaxFrom <= COUNT_BITS(UIntShared));

               // we can only guarantee that cBitsPerItemMaxFrom is less than or equal to COUNT_BITS(UIntShared)
               // so we need to construct our mask in that type, but afterwards we can convert it to a
               // size_t since we know the ultimate answer must fit into that since cBins fits into a size_t. If in
               // theory UIntShared were allowed to be a billion bits, then the mask could be 65 bits while the end
               // result would be forced to be 64 bits or less since we use the maximum number of bits per item possible
               const size_t maskBitsFrom = static_cast<size_t>(MakeLowMask<UIntShared>(cBitsPerItemMaxFrom));

               pDimensionInfoInit->m_cItemsPerBitPackFrom = cItemsPerBitPackFrom;
               pDimensionInfoInit->m_cBitsPerItemMaxFrom = cBitsPerItemMaxFrom;
               pDimensionInfoInit->m_maskBitsFrom = maskBitsFrom;
               pDimensionInfoInit->m_iShiftFrom =
                     static_cast<int>((cSharedSamples - size_t{1}) % static_cast<size_t>(cItemsPerBitPackFrom));

               ++pDimensionInfoInit;
            }
            ++piTermFeature;
            ++pTermFeature;
         } while(pTermFeaturesEnd != pTermFeature);
         EBM_ASSERT(pDimensionInfoInit == &dimensionInfo[pTerm->GetCountRealDimensions()]);

         const BagEbm* pSampleReplication = aBag;
         BagEbm replication = 0;
         size_t iTensor;

         DataSubsetBoosting* pSubset = m_aSubsets;
         do {
            EBM_ASSERT(1 <= pTerm->GetBitsRequiredMin());
            const int cItemsPerBitPackTo =
                  GetCountItemsBitPacked(pTerm->GetBitsRequiredMin(), pSubset->GetObjectiveWrapper()->m_cUIntBytes);
            EBM_ASSERT(1 <= cItemsPerBitPackTo);
            ANALYSIS_ASSERT(0 != cItemsPerBitPackTo);

            const int cBitsPerItemMaxTo =
                  GetCountBits(cItemsPerBitPackTo, pSubset->GetObjectiveWrapper()->m_cUIntBytes);
            EBM_ASSERT(1 <= cBitsPerItemMaxTo);

            const size_t cSIMDPack = pSubset->GetObjectiveWrapper()->m_cSIMDPack;
            EBM_ASSERT(1 <= cSIMDPack);

            const size_t cSubsetSamples = pSubset->GetCountSamples();
            EBM_ASSERT(1 <= cSubsetSamples);
            EBM_ASSERT(0 == cSubsetSamples % cSIMDPack);

            size_t cParallelSamples = cSubsetSamples / cSIMDPack;
            EBM_ASSERT(1 <= cParallelSamples);

            // the last bit position is wasted and set to zero to improve prefetching
            size_t cParallelDataUnitsTo = cParallelSamples / static_cast<size_t>(cItemsPerBitPackTo);
            if(IsAddError(cParallelDataUnitsTo, size_t{1})) {
               LOG_0(Trace_Warning,
                     "WARNING DataSetBoosting::InitTermData IsAddError(cParallelDataUnitsTo, size_t{1})");
               return Error_OutOfMemory;
            }
            ++cParallelDataUnitsTo;

            if(IsMultiplyError(pSubset->GetObjectiveWrapper()->m_cUIntBytes, cParallelDataUnitsTo, cSIMDPack)) {
               LOG_0(Trace_Warning,
                     "WARNING DataSetBoosting::InitTermData "
                     "IsMultiplyError(pSubset->GetObjectiveWrapper()->m_cUIntBytes, cParallelDataUnitsTo, cSIMDPack)");
               return Error_OutOfMemory;
            }
            const size_t cBytes = pSubset->GetObjectiveWrapper()->m_cUIntBytes * cParallelDataUnitsTo * cSIMDPack;
            void* pTermDataTo = AlignedAlloc(cBytes);
            if(nullptr == pTermDataTo) {
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitTermData nullptr == pTermDataTo");
               return Error_OutOfMemory;
            }
            EBM_ASSERT(nullptr != pSubset->m_aaTermData);
            pSubset->m_aaTermData[iTerm] = pTermDataTo;

            memset(pTermDataTo, 0, cBytes);

            // we always leave the last bit slot empty (with zeros) for prefetch optimization
            int cShiftTo =
                  static_cast<int>(cParallelSamples % static_cast<size_t>(cItemsPerBitPackTo)) * cBitsPerItemMaxTo;
            const int cShiftResetTo = (cItemsPerBitPackTo - 1) * cBitsPerItemMaxTo;
            while(true) {
               do {
                  size_t iPartition = 0;
                  do {
                     if(BagEbm{0} == replication) {
                        replication = 1;
                        if(nullptr != pSampleReplication) {
                           const BagEbm* pSampleReplicationOriginal = pSampleReplication;
                           bool isItemValidation;
                           do {
                              do {
                                 replication = *pSampleReplication;
                                 ++pSampleReplication;
                              } while(BagEbm{0} == replication);
                              isItemValidation = replication < BagEbm{0};
                           } while(isLoopValidation != isItemValidation);
                           const size_t cAdvances = pSampleReplication - pSampleReplicationOriginal - 1;
                           if(0 != cAdvances) {
                              FeatureDimension* pDimensionInfo = dimensionInfo;
                              do {
                                 const int cItemsPerBitPackFrom = pDimensionInfo->m_cItemsPerBitPackFrom;
                                 size_t cCompleteAdvanced = cAdvances / static_cast<size_t>(cItemsPerBitPackFrom);
                                 int iShiftFrom = pDimensionInfo->m_iShiftFrom;
                                 EBM_ASSERT(0 <= iShiftFrom);
                                 iShiftFrom -= static_cast<int>(cAdvances % static_cast<size_t>(cItemsPerBitPackFrom));
                                 pDimensionInfo->m_iShiftFrom = iShiftFrom;
                                 if(iShiftFrom < 0) {
                                    pDimensionInfo->m_iShiftFrom = iShiftFrom + cItemsPerBitPackFrom;
                                    EBM_ASSERT(0 <= pDimensionInfo->m_iShiftFrom);
                                    ++cCompleteAdvanced;
                                 }
                                 pDimensionInfo->m_pFeatureDataFrom += cCompleteAdvanced;

                                 ++pDimensionInfo;
                              } while(pDimensionInfoInit != pDimensionInfo);
                           }
                        }

                        iTensor = 0;
                        size_t tensorMultiple = 1;
                        FeatureDimension* pDimensionInfo = dimensionInfo;
                        do {
                           const UIntShared* const pFeatureDataFrom = pDimensionInfo->m_pFeatureDataFrom;
                           const UIntShared bitsFrom = *pFeatureDataFrom;

                           int iShiftFrom = pDimensionInfo->m_iShiftFrom;
                           EBM_ASSERT(0 <= iShiftFrom);
                           EBM_ASSERT(iShiftFrom * pDimensionInfo->m_cBitsPerItemMaxFrom < COUNT_BITS(UIntShared));
                           const size_t iFeatureBin =
                                 static_cast<size_t>(bitsFrom >> (iShiftFrom * pDimensionInfo->m_cBitsPerItemMaxFrom)) &
                                 pDimensionInfo->m_maskBitsFrom;

                           // we check our dataSet when we get the header, and cBins has been checked to fit into size_t
                           EBM_ASSERT(iFeatureBin < pDimensionInfo->m_cBins);

                           --iShiftFrom;
                           pDimensionInfo->m_iShiftFrom = iShiftFrom;
                           if(iShiftFrom < 0) {
                              EBM_ASSERT(-1 == iShiftFrom);
                              pDimensionInfo->m_iShiftFrom = iShiftFrom + pDimensionInfo->m_cItemsPerBitPackFrom;
                              pDimensionInfo->m_pFeatureDataFrom = pFeatureDataFrom + 1;
                           }

                           // we check for overflows during Term construction, but let's check here again
                           EBM_ASSERT(!IsMultiplyError(tensorMultiple, pDimensionInfo->m_cBins));

                           // this can't overflow if the multiplication below doesn't overflow, and we checked for that
                           // above
                           iTensor += tensorMultiple * iFeatureBin;
                           tensorMultiple *= pDimensionInfo->m_cBins;

                           ++pDimensionInfo;
                        } while(pDimensionInfoInit != pDimensionInfo);

                        EBM_ASSERT(iTensor < pTerm->GetCountTensorBins());
                     }

                     EBM_ASSERT(0 != replication);
                     EBM_ASSERT(0 < replication && 0 < direction || replication < 0 && direction < 0);
                     replication -= direction;

                     EBM_ASSERT(0 <= cShiftTo);
                     if(sizeof(UIntBig) == pSubset->m_pObjective->m_cUIntBytes) {
                        *(reinterpret_cast<UIntBig*>(pTermDataTo) + iPartition) |= static_cast<UIntBig>(iTensor)
                              << cShiftTo;
                     } else {
                        EBM_ASSERT(sizeof(UIntSmall) == pSubset->m_pObjective->m_cUIntBytes);
                        *(reinterpret_cast<UIntSmall*>(pTermDataTo) + iPartition) |= static_cast<UIntSmall>(iTensor)
                              << cShiftTo;
                     }

                     ++iPartition;
                  } while(cSIMDPack != iPartition);

                  --cParallelSamples;
                  if(0 == cParallelSamples) {
                     // we always leave the last bit slot empty (with zeros) for prefetch optimization
                     goto done_subset;
                  }

                  cShiftTo -= cBitsPerItemMaxTo;
               } while(int{0} <= cShiftTo);
               cShiftTo = cShiftResetTo;

               pTermDataTo = IndexByte(pTermDataTo, pSubset->m_pObjective->m_cUIntBytes * cSIMDPack);
            }
         done_subset:

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
ErrorEbm DataSetBoosting::CopyWeights(
      const unsigned char* const pDataSetShared, const BagEbm direction, const BagEbm* const aBag) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::CopyWeights");

   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbm{-1} == direction || BagEbm{1} == direction);
   EBM_ASSERT(1 <= m_cSamples);

   const FloatShared* pWeightFrom = GetDataSetSharedWeight(pDataSetShared, 0);
   EBM_ASSERT(nullptr != pWeightFrom);

   const bool isLoopValidation = direction < BagEbm{0};
   EBM_ASSERT(nullptr != aBag || !isLoopValidation); // if aBag is nullptr then we have no validation samples

   const BagEbm* pSampleReplication = aBag;

   BagEbm replication = 0;
   FloatShared weight;
   if(IsMultiplyError(sizeof(FloatShared), m_cSamples)) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::CopyWeights IsMultiplyError(sizeof(FloatShared), m_cSamples)");
      return Error_OutOfMemory;
   }
   FloatShared* pWeightTo = reinterpret_cast<FloatShared*>(malloc(sizeof(FloatShared) * m_cSamples));
   if(nullptr == pWeightTo) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::CopyWeights nullptr == pWeightTo");
      return Error_OutOfMemory;
   }
   m_aOriginalWeights = pWeightTo;

   const FloatShared* const pWeightsToEnd = &pWeightTo[m_cSamples];
   do {
      if(BagEbm{0} == replication) {
         replication = 1;
         if(nullptr != pSampleReplication) {
            bool isItemValidation;
            do {
               do {
                  replication = *pSampleReplication;
                  ++pSampleReplication;
                  ++pWeightFrom;
               } while(BagEbm{0} == replication);
               isItemValidation = replication < BagEbm{0};
            } while(isLoopValidation != isItemValidation);
            --pWeightFrom;
         }

         weight = *pWeightFrom;
         ++pWeightFrom;

         // these were checked when creating the shared dataset
         EBM_ASSERT(!std::isnan(weight));
         EBM_ASSERT(!std::isinf(weight));
         EBM_ASSERT(FloatShared{0} < weight);
      }

      *pWeightTo = weight;
      ++pWeightTo;

      replication -= direction;
   } while(pWeightsToEnd != pWeightTo);
   EBM_ASSERT(0 == replication);

   LOG_0(Trace_Info, "Exited DataSetBoosting::CopyWeights");
   return Error_None;
}
WARNING_POP

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
WARNING_DISABLE_UNINITIALIZED_LOCAL_POINTER
ErrorEbm DataSetBoosting::InitBags(const bool bAllocateCachedTensors,
      void* const rng,
      const size_t cInnerBags,
      const size_t cTerms,
      const Term* const* const apTerms) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::InitBags");

   const size_t cIncludedSamples = m_cSamples;
   EBM_ASSERT(1 <= cIncludedSamples);

   if(IsMultiplyError(sizeof(TermInnerBag), cTerms)) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags IsMultiplyError(sizeof(TermInnerBag), cTerms)");
      return Error_OutOfMemory;
   }
   const size_t cTermInnerBagBytes = sizeof(TermInnerBag) * cTerms;

   DataSetInnerBag* pDataSetInnerBag = DataSetInnerBag::AllocateDataSetInnerBags(cInnerBags);
   if(nullptr == pDataSetInnerBag) {
      LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == pDataSetInnerBag");
      return Error_OutOfMemory;
   }
   m_aDataSetInnerBags = pDataSetInnerBag;

   // the compiler understands the internal state of this RNG and can locate its internal state into CPU registers
   RandomDeterministic cpuRng;
   uint8_t* aOccurrencesFrom = nullptr;
   if(size_t{0} != cInnerBags) {
      if(nullptr == rng) {
         // Inner bags are not used when building a differentially private model, so
         // we can use low-quality non-determinism.  Generate a non-deterministic seed
         uint64_t seed;
         try {
            RandomNondeterministic<uint64_t> randomGenerator;
            seed = randomGenerator.Next(std::numeric_limits<uint64_t>::max());
         } catch(const std::bad_alloc&) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags Out of memory in std::random_device");
            return Error_OutOfMemory;
         } catch(...) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags Unknown error in std::random_device");
            return Error_UnexpectedInternal;
         }
         cpuRng.Initialize(seed);
      } else {
         const RandomDeterministic* const pRng = reinterpret_cast<RandomDeterministic*>(rng);
         cpuRng.Initialize(*pRng); // move the RNG from memory into CPU registers
      }

      if(IsMultiplyError(sizeof(uint8_t), cIncludedSamples)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags IsMultiplyError(sizeof(uint8_t), cIncludedSamples)");
         return Error_OutOfMemory;
      }
      aOccurrencesFrom = static_cast<uint8_t*>(malloc(sizeof(uint8_t) * cIncludedSamples));
      if(nullptr == aOccurrencesFrom) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == aCountOccurrences");
         return Error_OutOfMemory;
      }
   }

   EBM_ASSERT(nullptr != m_aSubsets);
   EBM_ASSERT(1 <= m_cSubsets);
   const DataSubsetBoosting* const pSubsetsEnd = m_aSubsets + m_cSubsets;

   const size_t cInnerBagsAfterZero = size_t{0} == cInnerBags ? size_t{1} : cInnerBags;
   size_t iBag = 0;
   do {
      EBM_ASSERT(nullptr == pDataSetInnerBag->m_aTermInnerBags);
      if(size_t{0} != cTerms && bAllocateCachedTensors) {
         EBM_ASSERT(nullptr != apTerms);
         TermInnerBag* pTermInnerBag = static_cast<TermInnerBag*>(malloc(cTermInnerBagBytes));
         if(nullptr == pTermInnerBag) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == aTermInnerBag");
            free(aOccurrencesFrom);
            return Error_OutOfMemory;
         }
         pDataSetInnerBag->m_aTermInnerBags = pTermInnerBag;

         const TermInnerBag* const pTermInnerBagEnd = IndexByte(pTermInnerBag, cTermInnerBagBytes);
         do {
            pTermInnerBag->m_aCounts = nullptr;
            pTermInnerBag->m_aWeights = nullptr;
            ++pTermInnerBag;
         } while(pTermInnerBagEnd != pTermInnerBag);

         const Term* const* ppTerm = apTerms;
         pTermInnerBag = pDataSetInnerBag->m_aTermInnerBags;
         do {
            const Term* const pTerm = *ppTerm;
            ++ppTerm;
            const size_t cBins = pTerm->GetCountTensorBins();
            if(size_t{1} != cBins) {
               if(IsMultiplyError(EbmMax(sizeof(UIntMain), sizeof(FloatPrecomp)), cBins)) {
                  LOG_0(Trace_Warning,
                        "WARNING DataSetBoosting::InitBags IsMultiplyError(EbmMax(sizeof(UIntMain), "
                        "sizeof(FloatPrecomp)), cBins)");
                  free(aOccurrencesFrom);
                  return Error_OutOfMemory;
               }
               const size_t cBytesCounts = sizeof(UIntMain) * cBins;
               UIntMain* aBinCounts = static_cast<UIntMain*>(AlignedAlloc(cBytesCounts));
               if(nullptr == aBinCounts) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == aBinCounts");
                  free(aOccurrencesFrom);
                  return Error_OutOfMemory;
               }
               pTermInnerBag->m_aCounts = aBinCounts;

               const size_t cBytesWeights = sizeof(FloatPrecomp) * cBins;
               FloatPrecomp* aBinWeights = static_cast<FloatPrecomp*>(AlignedAlloc(cBytesWeights));
               if(nullptr == aBinWeights) {
                  LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == aBinWeights");
                  free(aOccurrencesFrom);
                  return Error_OutOfMemory;
               }
               pTermInnerBag->m_aWeights = aBinWeights;

               memset(aBinCounts, 0, cBytesCounts);
               memset(aBinWeights, 0, cBytesWeights);
            }
            ++pTermInnerBag;
         } while(pTermInnerBagEnd != pTermInnerBag);
      }

      if(nullptr != aOccurrencesFrom) {
         EBM_ASSERT(size_t{0} != cInnerBags);
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
            aOccurrencesFrom[iSample] = existing + uint8_t{1};
            --cSamplesRemaining;
         } while(size_t{0} != cSamplesRemaining);
      }

      const FloatShared* pWeightFrom = m_aOriginalWeights;
      const uint8_t* pOccurrencesFrom;
      DataSubsetBoosting* pSubset;
      double totalWeight;
      if(nullptr != aOccurrencesFrom || nullptr != pWeightFrom) {
         totalWeight = 0.0;
         pOccurrencesFrom = aOccurrencesFrom;
         pSubset = m_aSubsets;
         do {
            EBM_ASSERT(nullptr != pSubset->m_aSubsetInnerBags);
            SubsetInnerBag* pSubsetInnerBag = &pSubset->m_aSubsetInnerBags[iBag];
            size_t cSubsetSamples = pSubset->GetCountSamples();
            EBM_ASSERT(1 <= cSubsetSamples);

            // add the weights in 2 stages to preserve precision
            double subsetWeight = 0.0;

            if(IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, cSubsetSamples)) {
               LOG_0(Trace_Warning,
                     "WARNING DataSetBoosting::InitBags IsMultiplyError(pSubset->m_pObjective->m_cFloatBytes, "
                     "cSubsetSamples)");
               free(aOccurrencesFrom);
               return Error_OutOfMemory;
            }
            size_t cBytes = pSubset->m_pObjective->m_cFloatBytes * cSubsetSamples;
            void* pWeightTo = AlignedAlloc(cBytes);
            if(nullptr == pWeightTo) {
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags nullptr == pWeightTo");
               free(aOccurrencesFrom);
               return Error_OutOfMemory;
            }
            pSubsetInnerBag->m_aWeights = pWeightTo;

            const void* const pWeightToEnd = IndexByte(pWeightTo, cBytes);
            do {
               double weight = double{1};
               if(nullptr != pOccurrencesFrom) {
                  EBM_ASSERT(size_t{0} != cInnerBags);
                  const uint8_t cOccurrences = *pOccurrencesFrom;
                  ++pOccurrencesFrom;
                  weight = static_cast<double>(cOccurrences);
               }

               if(nullptr != pWeightFrom) {
                  const double weightChange = static_cast<double>(*pWeightFrom);

                  // these were checked when creating the shared dataset
                  EBM_ASSERT(!std::isnan(weightChange));
                  EBM_ASSERT(!std::isinf(weightChange));
                  EBM_ASSERT(static_cast<double>(std::numeric_limits<float>::min()) <= weightChange);
                  EBM_ASSERT(weightChange <= static_cast<double>(std::numeric_limits<float>::max()));

                  weight *= weightChange;
                  ++pWeightFrom;

                  subsetWeight += weight;
               }

               if(sizeof(FloatBig) == pSubset->m_pObjective->m_cFloatBytes) {
                  *reinterpret_cast<FloatBig*>(pWeightTo) = static_cast<FloatBig>(weight);
               } else {
                  EBM_ASSERT(sizeof(FloatSmall) == pSubset->m_pObjective->m_cFloatBytes);
                  *reinterpret_cast<FloatSmall*>(pWeightTo) = static_cast<FloatSmall>(weight);
               }
               pWeightTo = IndexByte(pWeightTo, pSubset->m_pObjective->m_cFloatBytes);
            } while(pWeightToEnd != pWeightTo);

            totalWeight += subsetWeight;

            ++pSubset;
         } while(pSubsetsEnd != pSubset);
      }
      if(nullptr == pWeightFrom) {
         // use this more accurate non-floating point version if we can
         totalWeight = static_cast<double>(cIncludedSamples);
      }

      EBM_ASSERT(!std::isnan(totalWeight));
      EBM_ASSERT(std::numeric_limits<double>::min() <= totalWeight);

      if(std::isinf(totalWeight)) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitBags std::isinf(total)");
         free(aOccurrencesFrom);
         return Error_UserParamVal;
      }

      pDataSetInnerBag->m_totalWeight = totalWeight;
      pDataSetInnerBag->m_totalCount = cIncludedSamples;

      TermInnerBag* const aTermInnerBag = pDataSetInnerBag->m_aTermInnerBags;
      if(nullptr != aTermInnerBag) {
         EBM_ASSERT(1 <= cTerms);
         EBM_ASSERT(nullptr != apTerms);

         size_t iTerm = 0;
         do {
            const Term* const pTerm = apTerms[iTerm];

            if(1 != pTerm->GetCountTensorBins()) {
               TermInnerBag* const pTermInnerBag = &aTermInnerBag[iTerm];
               UIntMain* const aCounts = pTermInnerBag->GetCounts();
               FloatPrecomp* const aWeights = pTermInnerBag->GetWeights();

               pWeightFrom = m_aOriginalWeights;
               pOccurrencesFrom = aOccurrencesFrom;
               pSubset = m_aSubsets;
               do {
                  EBM_ASSERT(1 <= pTerm->GetBitsRequiredMin());
                  const int cItemsPerBitPack = GetCountItemsBitPacked(
                        pTerm->GetBitsRequiredMin(), pSubset->GetObjectiveWrapper()->m_cUIntBytes);
                  EBM_ASSERT(1 <= cItemsPerBitPack);
                  ANALYSIS_ASSERT(0 != cItemsPerBitPack);

                  const int cBitsPerItemMax =
                        GetCountBits(cItemsPerBitPack, pSubset->GetObjectiveWrapper()->m_cUIntBytes);
                  EBM_ASSERT(1 <= cBitsPerItemMax);

                  const size_t cSIMDPack = pSubset->GetObjectiveWrapper()->m_cSIMDPack;
                  EBM_ASSERT(1 <= cSIMDPack);

                  const size_t cSubsetSamples = pSubset->GetCountSamples();
                  EBM_ASSERT(1 <= cSubsetSamples);
                  EBM_ASSERT(0 == cSubsetSamples % cSIMDPack);

                  size_t cParallelSamples = cSubsetSamples / cSIMDPack;
                  EBM_ASSERT(1 <= cParallelSamples);

                  size_t maskBits;
                  if(sizeof(UIntBig) == pSubset->m_pObjective->m_cUIntBytes) {
                     maskBits = static_cast<size_t>(MakeLowMask<UIntBig>(cBitsPerItemMax));
                  } else {
                     EBM_ASSERT(sizeof(UIntSmall) == pSubset->m_pObjective->m_cUIntBytes);
                     maskBits = static_cast<size_t>(MakeLowMask<UIntSmall>(cBitsPerItemMax));
                  }

                  void* pTermData = pSubset->m_aaTermData[iTerm];

                  const int cShiftReset = (cItemsPerBitPack - 1) * cBitsPerItemMax;
                  int cShift =
                        static_cast<int>(cParallelSamples % static_cast<size_t>(cItemsPerBitPack)) * cBitsPerItemMax;

                  while(true) {
                     do {
                        size_t iPartition = 0;
                        do {
                           size_t iTensor;

                           EBM_ASSERT(0 <= cShift);
                           if(sizeof(UIntBig) == pSubset->m_pObjective->m_cUIntBytes) {
                              iTensor = maskBits &
                                    static_cast<size_t>(
                                          *(reinterpret_cast<UIntBig*>(pTermData) + iPartition) >> cShift);
                           } else {
                              EBM_ASSERT(sizeof(UIntSmall) == pSubset->m_pObjective->m_cUIntBytes);
                              iTensor = maskBits &
                                    static_cast<size_t>(
                                          *(reinterpret_cast<UIntSmall*>(pTermData) + iPartition) >> cShift);
                           }
                           EBM_ASSERT(iTensor < pTerm->GetCountTensorBins());

                           double weight = double{1};
                           if(nullptr != pWeightFrom) {
                              weight = static_cast<double>(*pWeightFrom);
                              ++pWeightFrom;
                           }

                           uint8_t cOccurrences = 1;
                           if(nullptr != pOccurrencesFrom) {
                              EBM_ASSERT(size_t{0} != cInnerBags);
                              cOccurrences = *pOccurrencesFrom;
                              ++pOccurrencesFrom;
                              weight *= static_cast<double>(cOccurrences);
                           }

                           if(nullptr != aCounts) {
                              aCounts[iTensor] += cOccurrences;
                           }

                           if(nullptr != aWeights) {
                              aWeights[iTensor] += weight;
                           }

                           ++iPartition;
                        } while(cSIMDPack != iPartition);

                        --cParallelSamples;
                        if(0 == cParallelSamples) {
                           // we always leave the last slot empty (with zeros)
                           goto done_subset;
                        }

                        cShift -= cBitsPerItemMax;
                     } while(0 <= cShift);
                     cShift = cShiftReset;

                     pTermData = IndexByte(pTermData, pSubset->m_pObjective->m_cUIntBytes * cSIMDPack);
                  }
               done_subset:

                  ++pSubset;
               } while(pSubsetsEnd != pSubset);
            }
            ++iTerm;
         } while(cTerms != iTerm);
      }
      ++pDataSetInnerBag;
      ++iBag;
   } while(cInnerBagsAfterZero != iBag);

   if(nullptr != aOccurrencesFrom) {
      EBM_ASSERT(size_t{0} != cInnerBags);
      free(aOccurrencesFrom);
      if(nullptr != rng) {
         RandomDeterministic* pRng = reinterpret_cast<RandomDeterministic*>(rng);
         pRng->Initialize(cpuRng); // move the RNG from CPU registers back into memory
      }
   }

   LOG_0(Trace_Info, "Exited DataSetBoosting::InitBags");
   return Error_None;
}
WARNING_POP

ErrorEbm DataSetBoosting::InitDataSetBoosting(const bool bAllocateGradients,
      const bool bAllocateHessians,
      const bool bAllocateSampleScores,
      const bool bAllocateTargetData,
      const bool bAllocateCachedTensors,
      void* const rng,
      const size_t cScores,
      const size_t cSubsetItemsMax,
      const ObjectiveWrapper* const pObjectiveCpu,
      const ObjectiveWrapper* const pObjectiveSIMD,
      const unsigned char* const pDataSetShared,
      const double* const aIntercept,
      const BagEbm direction,
      const size_t cSharedSamples,
      const BagEbm* const aBag,
      const double* const aInitScores,
      const size_t cIncludedSamples,
      const size_t cInnerBags,
      const size_t cWeights,
      const size_t cTerms,
      const Term* const* const apTerms,
      const IntEbm* const aiTermFeatures) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::InitDataSetBoosting");

   ErrorEbm error;

   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(1 <= cSubsetItemsMax);
   EBM_ASSERT(nullptr != pObjectiveCpu);
   EBM_ASSERT(nullptr != pObjectiveCpu->m_pObjective); // the objective for the CPU zone cannot be null unlike SIMD
   EBM_ASSERT(nullptr != pObjectiveSIMD);
   EBM_ASSERT(nullptr != pDataSetShared);
   EBM_ASSERT(BagEbm{-1} == direction || BagEbm{1} == direction);

   EBM_ASSERT(0 == m_cSamples);
   EBM_ASSERT(0 == m_cSubsets);
   EBM_ASSERT(nullptr == m_aSubsets);
   EBM_ASSERT(nullptr == m_aDataSetInnerBags);
   EBM_ASSERT(nullptr == m_aOriginalWeights);

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

         if(size_t{0} == cSIMDPack || cSubsetSamples < cSIMDPack) {
            // these remaing items cannot be processed with the SIMD compute, so they go into the CPU compute
         } else {
            // drop any items which cannot fit into the SIMD pack
            cSubsetSamples = cSubsetSamples - cSubsetSamples % cSIMDPack;
         }
         ++cSubsets;
         EBM_ASSERT(1 <= cSubsetSamples);
         EBM_ASSERT(cSubsetSamples <= cIncludedSamplesRemainingInit);
         cIncludedSamplesRemainingInit -= cSubsetSamples;
      } while(size_t{0} != cIncludedSamplesRemainingInit);
      EBM_ASSERT(1 <= cSubsets);

      if(IsMultiplyError(sizeof(DataSubsetBoosting), cSubsets)) {
         LOG_0(Trace_Warning,
               "WARNING DataSetBoosting::InitDataSetBoosting IsMultiplyError(sizeof(DataSubsetBoosting), cSubsets)");
         return Error_OutOfMemory;
      }
      DataSubsetBoosting* pSubset = static_cast<DataSubsetBoosting*>(malloc(sizeof(DataSubsetBoosting) * cSubsets));
      if(nullptr == pSubset) {
         LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitDataSetBoosting nullptr == pSubset");
         return Error_OutOfMemory;
      }
      m_aSubsets = pSubset;
      m_cSubsets = cSubsets;

      const DataSubsetBoosting* const pSubsetsEnd = pSubset + cSubsets;

      DataSubsetBoosting* pSubsetInit = pSubset;
      do {
         pSubsetInit->SafeInitDataSubsetBoosting();
         ++pSubsetInit;
      } while(pSubsetsEnd != pSubsetInit);

      size_t cIncludedSamplesRemaining = cIncludedSamples;
      do {
         EBM_ASSERT(1 <= cIncludedSamplesRemaining);

         size_t cSubsetSamples = EbmMin(cIncludedSamplesRemaining, cSubsetItemsMax);

         if(size_t{0} == cSIMDPack || cSubsetSamples < cSIMDPack) {
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

         if(size_t{0} != cTerms) {
            if(IsMultiplyError(sizeof(void*), cTerms)) {
               LOG_0(Trace_Warning,
                     "WARNING DataSetBoosting::InitDataSetBoosting IsMultiplyError(sizeof(void *), cTerms)");
               return Error_OutOfMemory;
            }
            void** paTermData = static_cast<void**>(malloc(sizeof(void*) * cTerms));
            if(nullptr == paTermData) {
               LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitDataSetBoosting nullptr == paTermData");
               return Error_OutOfMemory;
            }
            pSubset->m_aaTermData = paTermData;

            const void* const* const paTermDataEnd = paTermData + cTerms;
            do {
               *paTermData = nullptr;
               ++paTermData;
            } while(paTermDataEnd != paTermData);
         }

         SubsetInnerBag* const aSubsetInnerBags = SubsetInnerBag::AllocateSubsetInnerBags(cInnerBags);
         if(nullptr == aSubsetInnerBags) {
            LOG_0(Trace_Warning, "WARNING DataSetBoosting::InitDataSetBoosting nullptr == aSubsetInnerBags");
            return Error_OutOfMemory;
         }
         pSubset->m_aSubsetInnerBags = aSubsetInnerBags;

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
         error = InitSampleScores(cScores, aIntercept, direction, aBag, aInitScores);
         if(Error_None != error) {
            return error;
         }
      }

      if(bAllocateTargetData) {
         error = InitTargetData(pDataSetShared, direction, aBag);
         if(Error_None != error) {
            return error;
         }
      }

      if(0 != cTerms) {
         error = InitTermData(pDataSetShared, direction, cSharedSamples, aBag, cTerms, apTerms, aiTermFeatures);
         if(Error_None != error) {
            return error;
         }
      }

      if(size_t{0} != cWeights) {
         error = CopyWeights(pDataSetShared, direction, aBag);
         if(Error_None != error) {
            return error;
         }
      }

      error = InitBags(bAllocateCachedTensors, rng, cInnerBags, cTerms, apTerms);
      if(Error_None != error) {
         return error;
      }
   }

   LOG_0(Trace_Info, "Exited DataSetBoosting::InitDataSetBoosting");
   return Error_None;
}

void DataSetBoosting::DestructDataSetBoosting(const size_t cTerms, const size_t cInnerBags) {
   LOG_0(Trace_Info, "Entered DataSetBoosting::DestructDataSetBoosting");

   DataSetInnerBag::FreeDataSetInnerBags(cInnerBags, m_aDataSetInnerBags, cTerms);
   free(m_aOriginalWeights);

   DataSubsetBoosting* pSubset = m_aSubsets;
   if(nullptr != pSubset) {
      EBM_ASSERT(1 <= m_cSubsets);
      const DataSubsetBoosting* const pSubsetsEnd = pSubset + m_cSubsets;
      do {
         pSubset->DestructDataSubsetBoosting(cTerms, cInnerBags);
         ++pSubset;
      } while(pSubsetsEnd != pSubset);
      free(m_aSubsets);
   }

   LOG_0(Trace_Info, "Exited DataSetBoosting::DestructDataSetBoosting");
}

} // namespace DEFINED_ZONE_NAME
