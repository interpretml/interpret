// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "CompressibleTensor.hpp"
#include "ebm_stats.hpp"
// feature includes
#include "Feature.hpp"
// FeatureGroup.h depends on FeatureInternal.h
#include "FeatureGroup.hpp"
// dataset depends on features
#include "DataSetBoosting.hpp"
// samples is somewhat independent from datasets, but relies on an indirect coupling with them
#include "SamplingSet.hpp"
#include "HistogramTargetEntry.hpp"
#include "HistogramBucket.hpp"

#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern void BinBoosting(
   BoosterShell * const pBoosterShell,
   const FeatureGroup * const pFeatureGroup,
   const SamplingSet * const pTrainingSet
);

extern void SumHistogramBuckets(
   BoosterShell * const pBoosterShell,
   const size_t cHistogramBuckets
#ifndef NDEBUG
   , const size_t cSamplesTotal
   , const FloatEbmType weightTotal
#endif // NDEBUG
);

extern void TensorTotalsBuild(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const FeatureGroup * const pFeatureGroup,
   HistogramBucketBase * pBucketAuxiliaryBuildZone,
   HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
   , HistogramBucketBase * const aHistogramBucketsDebugCopy
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
);

extern ErrorEbmType PartitionOneDimensionalBoosting(
   BoosterShell * const pBoosterShell,
   const size_t cHistogramBuckets,
   const size_t cSamplesTotal,
   const FloatEbmType weightTotal,
   const size_t iDimension,
   const size_t cSamplesRequiredForChildSplitMin,
   const size_t cLeavesMax,
   FloatEbmType * const pTotalGain
);

extern ErrorEbmType PartitionTwoDimensionalBoosting(
   BoosterShell * const pBoosterShell,
   const FeatureGroup * const pFeatureGroup,
   const size_t cSamplesRequiredForChildSplitMin,
   HistogramBucketBase * pAuxiliaryBucketZone,
   FloatEbmType * const pTotalGain
#ifndef NDEBUG
   , const HistogramBucketBase * const aHistogramBucketsDebugCopy
#endif // NDEBUG
);

extern ErrorEbmType PartitionRandomBoosting(
   BoosterShell * const pBoosterShell,
   const FeatureGroup * const pFeatureGroup,
   const GenerateUpdateOptionsType options,
   const IntEbmType * const aLeavesMax,
   FloatEbmType * const pTotalGain
);

static ErrorEbmType BoostZeroDimensional(
   BoosterShell * const pBoosterShell, 
   const SamplingSet * const pTrainingSet,
   const GenerateUpdateOptionsType options
) {
   LOG_0(TraceLevelVerbose, "Entered BoostZeroDimensional");

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   if(GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength) || 
      GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength)) 
   {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "WARNING BoostZeroDimensional GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength) || GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength)");
      return Error_OutOfMemory;
   }
   const size_t cBytesPerHistogramBucketFast = GetHistogramBucketSize<FloatEbmType>(bClassification, cVectorLength);

   HistogramBucketBase * const pHistogramBucketFast = pBoosterShell->GetHistogramBucketBaseFast(cBytesPerHistogramBucketFast);
   if(UNLIKELY(nullptr == pHistogramBucketFast)) {
      // already logged
      return Error_OutOfMemory;
   }
   pHistogramBucketFast->Zero(cBytesPerHistogramBucketFast);

#ifndef NDEBUG
   pBoosterShell->SetHistogramBucketsEndDebugFast(reinterpret_cast<unsigned char *>(pHistogramBucketFast) + cBytesPerHistogramBucketFast);
#endif // NDEBUG

   BinBoosting(
      pBoosterShell,
      nullptr,
      pTrainingSet
   );

   const size_t cBytesPerHistogramBucketBig = GetHistogramBucketSize<FloatEbmType>(bClassification, cVectorLength);

   HistogramBucketBase * const pHistogramBucketBig = pBoosterShell->GetHistogramBucketBaseBig(cBytesPerHistogramBucketBig);
   if(UNLIKELY(nullptr == pHistogramBucketBig)) {
      // already logged
      return Error_OutOfMemory;
   }

#ifndef NDEBUG
   pBoosterShell->SetHistogramBucketsEndDebugBig(reinterpret_cast<unsigned char *>(pHistogramBucketBig) + cBytesPerHistogramBucketBig);
#endif // NDEBUG

   // TODO: put this into it's own function that converts our fast floats to big floats
   static_assert(sizeof(FloatEbmType) == sizeof(FloatEbmType), "float mismatch");
   EBM_ASSERT(cBytesPerHistogramBucketFast == cBytesPerHistogramBucketBig); // until we switch fast to float datatypes
   memcpy(pHistogramBucketBig, pHistogramBucketFast, cBytesPerHistogramBucketFast);


   // TODO: we can exit here back to python to allow caller modification to our histograms


   CompressibleTensor * const pSmallChangeToModelOverwriteSingleSamplingSet = 
      pBoosterShell->GetOverwritableModelUpdate();
   FloatEbmType * aValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
   if(bClassification) {
      const auto * const pHistogramBucketLocal = pHistogramBucketBig->GetHistogramBucket<FloatEbmType, true>();
      const auto * const aSumHistogramTargetEntry = pHistogramBucketLocal->GetHistogramTargetEntry();
      if(0 != (GenerateUpdateOptions_GradientSums & options)) {
         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            const FloatEbmType update = EbmStats::ComputeSinglePartitionUpdateGradientSum(aSumHistogramTargetEntry[iVector].m_sumGradients);

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
            // Hmmm.. for DP we need the sum, which means that we can't zero one of the class numbers as we
            // could with one of the logits in multiclass.
#endif // ZERO_FIRST_MULTICLASS_LOGIT

            aValues[iVector] = update;
         }
      } else {

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
         FloatEbmType zeroLogit = FloatEbmType { 0 };
#endif // ZERO_FIRST_MULTICLASS_LOGIT

         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            FloatEbmType update = EbmStats::ComputeSinglePartitionUpdate(
               aSumHistogramTargetEntry[iVector].m_sumGradients,
               aSumHistogramTargetEntry[iVector].GetSumHessians()
            );

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
            if(IsMulticlass(runtimeLearningTypeOrCountTargetClasses)) {
               if(size_t { 0 } == iVector) {
                  zeroLogit = update;
               }
               update -= zeroLogit;
            }
#endif // ZERO_FIRST_MULTICLASS_LOGIT

            aValues[iVector] = update;
         }
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      const auto * const pHistogramBucketLocal = pHistogramBucketBig->GetHistogramBucket<FloatEbmType, false>();
      const auto * const aSumHistogramTargetEntry = pHistogramBucketLocal->GetHistogramTargetEntry();
      if(0 != (GenerateUpdateOptions_GradientSums & options)) {
         const FloatEbmType smallChangeToModel = EbmStats::ComputeSinglePartitionUpdateGradientSum(aSumHistogramTargetEntry[0].m_sumGradients);
         aValues[0] = smallChangeToModel;
      } else {
         const FloatEbmType smallChangeToModel = EbmStats::ComputeSinglePartitionUpdate(
            aSumHistogramTargetEntry[0].m_sumGradients,
            pHistogramBucketLocal->GetWeightInBucket()
         );
         aValues[0] = smallChangeToModel;
      }
   }

   LOG_0(TraceLevelVerbose, "Exited BoostZeroDimensional");
   return Error_None;
}

static ErrorEbmType BoostSingleDimensional(
   BoosterShell * const pBoosterShell,
   const FeatureGroup * const pFeatureGroup,
   const size_t cHistogramBuckets,
   const SamplingSet * const pTrainingSet,
   const size_t iDimension,
   const size_t cSamplesRequiredForChildSplitMin,
   const IntEbmType countLeavesMax,
   FloatEbmType * const pTotalGain
) {
   ErrorEbmType error;

   LOG_0(TraceLevelVerbose, "Entered BoostSingleDimensional");

   EBM_ASSERT(1 == pFeatureGroup->GetCountSignificantDimensions());

   EBM_ASSERT(IntEbmType { 2 } <= countLeavesMax); // otherwise we would have called BoostZeroDimensional
   size_t cLeavesMax = static_cast<size_t>(countLeavesMax);
   if(IsConvertError<size_t>(countLeavesMax)) {
      // we can never exceed a size_t number of leaves, so let's just set it to the maximum if we were going to overflow because it will generate 
      // the same results as if we used the true number
      cLeavesMax = std::numeric_limits<size_t>::max();
   }

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);
   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   if(GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength) ||
      GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength)) 
   {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "WARNING BoostSingleDimensional GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength) || GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength)");
      return Error_OutOfMemory;
   }

   const size_t cBytesPerHistogramBucketFast = GetHistogramBucketSize<FloatEbmType>(bClassification, cVectorLength);
   if(IsMultiplyError(cBytesPerHistogramBucketFast, cHistogramBuckets)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "WARNING BoostSingleDimensional IsMultiplyError(cBytesPerHistogramBucketFast, cHistogramBuckets)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBufferFast = cBytesPerHistogramBucketFast * cHistogramBuckets;

   HistogramBucketBase * const aHistogramBucketsFast = pBoosterShell->GetHistogramBucketBaseFast(cBytesBufferFast);
   if(UNLIKELY(nullptr == aHistogramBucketsFast)) {
      // already logged
      return Error_OutOfMemory;
   }
   aHistogramBucketsFast->Zero(cBytesPerHistogramBucketFast, cHistogramBuckets);

#ifndef NDEBUG
   pBoosterShell->SetHistogramBucketsEndDebugFast(reinterpret_cast<unsigned char *>(aHistogramBucketsFast) + cBytesBufferFast);
#endif // NDEBUG

   BinBoosting(
      pBoosterShell,
      pFeatureGroup,
      pTrainingSet
   );

   const size_t cBytesPerHistogramBucketBig = GetHistogramBucketSize<FloatEbmType>(bClassification, cVectorLength);
   if(IsMultiplyError(cBytesPerHistogramBucketBig, cHistogramBuckets)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "WARNING BoostSingleDimensional IsMultiplyError(cBytesPerHistogramBucketBig, cHistogramBuckets)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBufferBig = cBytesPerHistogramBucketBig * cHistogramBuckets;

   HistogramBucketBase * const aHistogramBucketsBig = pBoosterShell->GetHistogramBucketBaseBig(cBytesBufferBig);
   if(UNLIKELY(nullptr == aHistogramBucketsBig)) {
      // already logged
      return Error_OutOfMemory;
   }

#ifndef NDEBUG
   pBoosterShell->SetHistogramBucketsEndDebugBig(reinterpret_cast<unsigned char *>(aHistogramBucketsBig) + cBytesBufferBig);
#endif // NDEBUG

   // TODO: put this into it's own function that converts our fast floats to big floats
   static_assert(sizeof(FloatEbmType) == sizeof(FloatEbmType), "float mismatch");
   EBM_ASSERT(cBytesBufferFast == cBytesBufferBig); // until we switch fast to float datatypes
   memcpy(aHistogramBucketsBig, aHistogramBucketsFast, cBytesBufferFast);


   // TODO: we can exit here back to python to allow caller modification to our histograms


   HistogramTargetEntryBase * const aSumHistogramTargetEntry = pBoosterShell->GetSumHistogramTargetEntryArray();
   const size_t cBytesPerHistogramTargetEntry = GetHistogramTargetEntrySize<FloatEbmType>(bClassification);
   aSumHistogramTargetEntry->Zero(cBytesPerHistogramTargetEntry, cVectorLength);

   SumHistogramBuckets(
      pBoosterShell,
      cHistogramBuckets
#ifndef NDEBUG
      , pTrainingSet->GetTotalCountSampleOccurrences()
      , pTrainingSet->GetWeightTotal()
#endif // NDEBUG
   );

   const size_t cSamplesTotal = pTrainingSet->GetTotalCountSampleOccurrences();
   EBM_ASSERT(1 <= cSamplesTotal);
   const FloatEbmType weightTotal = pTrainingSet->GetWeightTotal();

   error = PartitionOneDimensionalBoosting(
      pBoosterShell,
      cHistogramBuckets,
      cSamplesTotal,
      weightTotal,
      iDimension,
      cSamplesRequiredForChildSplitMin,
      cLeavesMax, 
      pTotalGain
   );

   LOG_0(TraceLevelVerbose, "Exited BoostSingleDimensional");
   return error;
}

// TODO: for higher dimensional spaces, we need to add/subtract individual cells alot and the hessian isn't required (yet) in order to make decisions about
//   where to split.  For dimensions higher than 2, we might want to copy the tensor to a new tensor AFTER binning that keeps only the gradients and then 
//    go back to our original tensor after splits to determine the hessian
static ErrorEbmType BoostMultiDimensional(
   BoosterShell * const pBoosterShell,
   const FeatureGroup * const pFeatureGroup,
   const SamplingSet * const pTrainingSet,
   const size_t cSamplesRequiredForChildSplitMin,
   FloatEbmType * const pTotalGain
) {
   LOG_0(TraceLevelVerbose, "Entered BoostMultiDimensional");

   EBM_ASSERT(2 <= pFeatureGroup->GetCountDimensions());
   EBM_ASSERT(2 <= pFeatureGroup->GetCountSignificantDimensions());

   ErrorEbmType error;

   size_t cAuxillaryBucketsForBuildFastTotals = 0;
   size_t cTotalBucketsMainSpace = 1;

   const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
   const FeatureGroupEntry * const pFeatureGroupEntryEnd = pFeatureGroupEntry + pFeatureGroup->GetCountDimensions();
   do {
      const size_t cBins = pFeatureGroupEntry->m_pFeature->GetCountBins();
      EBM_ASSERT(size_t { 1 } <= cBins); // we don't boost on empty training sets
      if(size_t { 1 } < cBins) {
         // if this wasn't true then we'd have to check IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace) at runtime
         EBM_ASSERT(cAuxillaryBucketsForBuildFastTotals < cTotalBucketsMainSpace);
         // since cBins must be 2 or more, cAuxillaryBucketsForBuildFastTotals must grow slower than cTotalBucketsMainSpace, and we checked at 
         // allocation that cTotalBucketsMainSpace would not overflow
         EBM_ASSERT(!IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace));
         cAuxillaryBucketsForBuildFastTotals += cTotalBucketsMainSpace;
         // we check for simple multiplication overflow from m_cBins in pBoosterCore->Initialize when we unpack featureGroupsFeatureIndexes
         EBM_ASSERT(!IsMultiplyError(cTotalBucketsMainSpace, cBins));
         cTotalBucketsMainSpace *= cBins;
         // if this wasn't true then we'd have to check IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace) at runtime
         EBM_ASSERT(cAuxillaryBucketsForBuildFastTotals < cTotalBucketsMainSpace);
      }
      ++pFeatureGroupEntry;
   } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);
   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   if(GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength) || 
      GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength)) 
   {
      LOG_0(
         TraceLevelWarning,
         "WARNING BoostMultiDimensional GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength) || GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength)"
      );
      return Error_OutOfMemory;
   }
   const size_t cBytesPerHistogramBucketFast = GetHistogramBucketSize<FloatEbmType>(bClassification, cVectorLength);
   if(IsMultiplyError(cBytesPerHistogramBucketFast, cTotalBucketsMainSpace)) {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional IsMultiplyError(cBytesPerHistogramBucketFast, cTotalBucketsMainSpace)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBufferFast = cBytesPerHistogramBucketFast * cTotalBucketsMainSpace;

   // we don't need to free this!  It's tracked and reused by pBoosterShell
   HistogramBucketBase * const aHistogramBucketsFast = pBoosterShell->GetHistogramBucketBaseFast(cBytesBufferFast);
   if(UNLIKELY(nullptr == aHistogramBucketsFast)) {
      // already logged
      return Error_OutOfMemory;
   }
   aHistogramBucketsFast->Zero(cBytesPerHistogramBucketFast, cTotalBucketsMainSpace);

#ifndef NDEBUG
   pBoosterShell->SetHistogramBucketsEndDebugFast(reinterpret_cast<unsigned char *>(aHistogramBucketsFast) + cBytesBufferFast);
#endif // NDEBUG

   BinBoosting(
      pBoosterShell,
      pFeatureGroup,
      pTrainingSet
   );

   // we need to reserve 4 PAST the pointer we pass into SweepMultiDimensional!!!!.  We pass in index 20 at max, so we need 24
   const size_t cAuxillaryBucketsForSplitting = 24;
   const size_t cAuxillaryBuckets =
      cAuxillaryBucketsForBuildFastTotals < cAuxillaryBucketsForSplitting ? cAuxillaryBucketsForSplitting : cAuxillaryBucketsForBuildFastTotals;
   if(IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)");
      return Error_OutOfMemory;
   }
   const size_t cTotalBucketsBig = cTotalBucketsMainSpace + cAuxillaryBuckets;

   const size_t cBytesPerHistogramBucketBig = GetHistogramBucketSize<FloatEbmType>(bClassification, cVectorLength);
   if(IsMultiplyError(cBytesPerHistogramBucketBig, cTotalBucketsBig)) {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional IsMultiplyError(cBytesPerHistogramBucketBig, cTotalBucketsBig)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBufferBig = cBytesPerHistogramBucketBig * cTotalBucketsBig;

   // we don't need to free this!  It's tracked and reused by pBoosterShell
   HistogramBucketBase * const aHistogramBucketsBig = pBoosterShell->GetHistogramBucketBaseBig(cBytesBufferBig);
   if(UNLIKELY(nullptr == aHistogramBucketsBig)) {
      // already logged
      return Error_OutOfMemory;
   }

#ifndef NDEBUG
   const unsigned char * const aHistogramBucketsEndDebugBig = reinterpret_cast<unsigned char *>(aHistogramBucketsBig) + cBytesBufferBig;
   pBoosterShell->SetHistogramBucketsEndDebugBig(aHistogramBucketsEndDebugBig);
#endif // NDEBUG

   // TODO: put this into it's own function that converts our fast floats to big floats
   static_assert(sizeof(FloatEbmType) == sizeof(FloatEbmType), "float mismatch");
   memcpy(aHistogramBucketsBig, aHistogramBucketsFast, cBytesBufferFast);


   // we also need to zero the top end buckets above the binned histograms we've already generated
   aHistogramBucketsBig->Zero(cBytesPerHistogramBucketBig, cAuxillaryBuckets, cTotalBucketsMainSpace);


   // TODO: we can exit here back to python to allow caller modification to our histograms


#ifndef NDEBUG
   // make a copy of the original binned buckets for debugging purposes

   HistogramBucketBase * const aHistogramBucketsDebugCopy =
      EbmMalloc<HistogramBucketBase>(cTotalBucketsMainSpace, cBytesPerHistogramBucketBig);
   if(nullptr != aHistogramBucketsDebugCopy) {
      // if we can't allocate, don't fail.. just stop checking
      const size_t cBytesBufferDebug = cTotalBucketsMainSpace * cBytesPerHistogramBucketBig;
      memcpy(aHistogramBucketsDebugCopy, aHistogramBucketsBig, cBytesBufferDebug);
   }
#endif // NDEBUG

   HistogramBucketBase * pAuxiliaryBucketZone = GetHistogramBucketByIndex(
      cBytesPerHistogramBucketBig,
      aHistogramBucketsBig,
      cTotalBucketsMainSpace
   );

   TensorTotalsBuild(
      runtimeLearningTypeOrCountTargetClasses,
      pFeatureGroup,
      pAuxiliaryBucketZone,
      aHistogramBucketsBig
#ifndef NDEBUG
      , aHistogramBucketsDebugCopy
      , aHistogramBucketsEndDebugBig
#endif // NDEBUG
   );

   //permutation0
   //gain_permute0
   //  divs0
   //  gain0
   //    divs00
   //    gain00
   //      divs000
   //      gain000
   //      divs001
   //      gain001
   //    divs01
   //    gain01
   //      divs010
   //      gain010
   //      divs011
   //      gain011
   //  divs1
   //  gain1
   //    divs10
   //    gain10
   //      divs100
   //      gain100
   //      divs101
   //      gain101
   //    divs11
   //    gain11
   //      divs110
   //      gain110
   //      divs111
   //      gain111
   //---------------------------
   //permutation1
   //gain_permute1
   //  divs0
   //  gain0
   //    divs00
   //    gain00
   //      divs000
   //      gain000
   //      divs001
   //      gain001
   //    divs01
   //    gain01
   //      divs010
   //      gain010
   //      divs011
   //      gain011
   //  divs1
   //  gain1
   //    divs10
   //    gain10
   //      divs100
   //      gain100
   //      divs101
   //      gain101
   //    divs11
   //    gain11
   //      divs110
   //      gain110
   //      divs111
   //      gain111       *


   //size_t aiDimensionPermutation[k_cDimensionsMax];
   //for(unsigned int iDimensionInitialize = 0; iDimensionInitialize < cDimensions; ++iDimensionInitialize) {
   //   aiDimensionPermutation[iDimensionInitialize] = iDimensionInitialize;
   //}
   //size_t aiDimensionPermutationBest[k_cDimensionsMax];

   // DO this is a fixed length that we should make variable!
   //size_t aDOSplits[1000000];
   //size_t aDOSplitsBest[1000000];

   //do {
   //   size_t aiDimensions[k_cDimensionsMax];
   //   memset(aiDimensions, 0, sizeof(aiDimensions[0]) * cDimensions));
   //   while(true) {


   //      EBM_ASSERT(0 == iDimension);
   //      while(true) {
   //         ++aiDimension[iDimension];
   //         if(aiDimension[iDimension] != 
   //               pFeatureGroups->GetFeatureGroupEntries()[aiDimensionPermutation[iDimension]].m_pFeature->m_cBins) {
   //            break;
   //         }
   //         aiDimension[iDimension] = 0;
   //         ++iDimension;
   //         if(iDimension == cDimensions) {
   //            goto move_next_permutation;
   //         }
   //      }
   //   }
   //   move_next_permutation:
   //} while(std::next_permutation(aiDimensionPermutation, &aiDimensionPermutation[cDimensions]));

   if(2 == pFeatureGroup->GetCountSignificantDimensions()) {
      error = PartitionTwoDimensionalBoosting(
         pBoosterShell,
         pFeatureGroup,
         cSamplesRequiredForChildSplitMin,
         pAuxiliaryBucketZone,
         pTotalGain
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
#endif // NDEBUG
      );
      if(Error_None != error) {
#ifndef NDEBUG
         free(aHistogramBucketsDebugCopy);
#endif // NDEBUG

         LOG_0(TraceLevelVerbose, "Exited BoostMultiDimensional with Error code");

         return error;
      }

      EBM_ASSERT(!std::isnan(*pTotalGain));
      EBM_ASSERT(FloatEbmType { 0 } <= *pTotalGain);
   } else {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional 2 != pFeatureGroup->GetCountSignificantFeatures()");

      // TODO: eventually handle this in our caller and this function can specialize in handling just 2 dimensional
      //       then we can replace this branch with an assert
#ifndef NDEBUG
      EBM_ASSERT(false);
      free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
      return Error_UnexpectedInternal;
   }

#ifndef NDEBUG
   free(aHistogramBucketsDebugCopy);
#endif // NDEBUG

   LOG_0(TraceLevelVerbose, "Exited BoostMultiDimensional");
   return Error_None;
}

static ErrorEbmType BoostRandom(
   BoosterShell * const pBoosterShell,
   const FeatureGroup * const pFeatureGroup,
   const SamplingSet * const pTrainingSet,
   const GenerateUpdateOptionsType options,
   const IntEbmType * const aLeavesMax,
   FloatEbmType * const pTotalGain
) {
   // THIS RANDOM SPLIT FUNCTION IS PRIMARILY USED FOR DIFFERENTIAL PRIVACY EBMs

   LOG_0(TraceLevelVerbose, "Entered BoostRandom");

   ErrorEbmType error;

   const size_t cDimensions = pFeatureGroup->GetCountDimensions();
   EBM_ASSERT(1 <= cDimensions);

   size_t cTotalBuckets = 1;
   for(size_t iDimension = 0; iDimension < cDimensions; ++iDimension) {
      const size_t cBins = pFeatureGroup->GetFeatureGroupEntries()[iDimension].m_pFeature->GetCountBins();
      EBM_ASSERT(size_t { 1 } <= cBins); // we don't boost on empty training sets
      // we check for simple multiplication overflow from m_cBins in BoosterCore::Initialize when we unpack featureGroupsFeatureIndexes
      EBM_ASSERT(!IsMultiplyError(cTotalBuckets, cBins));
      cTotalBuckets *= cBins;
   }

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);
   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   if(GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength) ||
      GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength))
   {
      LOG_0(
         TraceLevelWarning,
         "WARNING BoostRandom GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength) || GetHistogramBucketSizeOverflow<FloatEbmType>(bClassification, cVectorLength)"
      );
      return Error_OutOfMemory;
   }
   const size_t cBytesPerHistogramBucketFast = GetHistogramBucketSize<FloatEbmType>(bClassification, cVectorLength);
   if(IsMultiplyError(cBytesPerHistogramBucketFast, cTotalBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING BoostRandom IsMultiplyError(cBytesPerHistogramBucketFast, cTotalBuckets)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBufferFast = cBytesPerHistogramBucketFast * cTotalBuckets;

   // we don't need to free this!  It's tracked and reused by pBoosterShell
   HistogramBucketBase * const aHistogramBucketsFast = pBoosterShell->GetHistogramBucketBaseFast(cBytesBufferFast);
   if(UNLIKELY(nullptr == aHistogramBucketsFast)) {
      // already logged
      return Error_OutOfMemory;
   }
   aHistogramBucketsFast->Zero(cBytesPerHistogramBucketFast, cTotalBuckets);

#ifndef NDEBUG
   pBoosterShell->SetHistogramBucketsEndDebugFast(reinterpret_cast<unsigned char *>(aHistogramBucketsFast) + cBytesBufferFast);
#endif // NDEBUG

   BinBoosting(
      pBoosterShell,
      pFeatureGroup,
      pTrainingSet
   );

   const size_t cBytesPerHistogramBucketBig = GetHistogramBucketSize<FloatEbmType>(bClassification, cVectorLength);
   if(IsMultiplyError(cBytesPerHistogramBucketBig, cTotalBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING BoostRandom IsMultiplyError(cBytesPerHistogramBucketBig, cTotalBuckets)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBufferBig = cBytesPerHistogramBucketBig * cTotalBuckets;

   // we don't need to free this!  It's tracked and reused by pBoosterShell
   HistogramBucketBase * const aHistogramBucketsBig = pBoosterShell->GetHistogramBucketBaseBig(cBytesBufferBig);
   if(UNLIKELY(nullptr == aHistogramBucketsBig)) {
      // already logged
      return Error_OutOfMemory;
   }

#ifndef NDEBUG
   pBoosterShell->SetHistogramBucketsEndDebugBig(reinterpret_cast<unsigned char *>(aHistogramBucketsBig) + cBytesBufferBig);
#endif // NDEBUG

   // TODO: put this into it's own function that converts our fast floats to big floats
   static_assert(sizeof(FloatEbmType) == sizeof(FloatEbmType), "float mismatch");
   EBM_ASSERT(cBytesBufferFast == cBytesBufferBig); // until we switch fast to float datatypes
   memcpy(aHistogramBucketsBig, aHistogramBucketsFast, cBytesBufferFast);


   // TODO: we can exit here back to python to allow caller modification to our histograms


   error = PartitionRandomBoosting(
      pBoosterShell,
      pFeatureGroup,
      options,
      aLeavesMax,
      pTotalGain
   );
   if(Error_None != error) {
      LOG_0(TraceLevelVerbose, "Exited BoostRandom with Error code");
      return error;
   }

   EBM_ASSERT(!std::isnan(*pTotalGain));
   EBM_ASSERT(FloatEbmType { 0 } <= *pTotalGain);

   LOG_0(TraceLevelVerbose, "Exited BoostRandom");
   return Error_None;
}

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
static ErrorEbmType GenerateModelUpdateInternal(
   BoosterShell * const pBoosterShell,
   const size_t iFeatureGroup,
   const GenerateUpdateOptionsType options,
   const double learningRate,
   const size_t cSamplesRequiredForChildSplitMin,
   const IntEbmType * const aLeavesMax, 
   double * const pGainAvgOut
) {
   ErrorEbmType error;

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelVerbose, "Entered GenerateModelUpdateInternal");

   const size_t cSamplingSetsAfterZero = (0 == pBoosterCore->GetCountSamplingSets()) ? 1 : pBoosterCore->GetCountSamplingSets();
   const FeatureGroup * const pFeatureGroup = pBoosterCore->GetFeatureGroups()[iFeatureGroup];
   const size_t cSignificantDimensions = pFeatureGroup->GetCountSignificantDimensions();
   const size_t cDimensions = pFeatureGroup->GetCountDimensions();

   // TODO: we can probably eliminate lastDimensionLeavesMax and cSignificantBinCount and just fetch them from iDimensionImportant afterwards
   IntEbmType lastDimensionLeavesMax = IntEbmType { 0 };
   // this initialization isn't required, but this variable ends up touching a lot of downstream state
   // and g++ seems to warn about all of that usage, even in other downstream functions!
   size_t cSignificantBinCount = size_t { 0 };
   size_t iDimensionImportant = 0;
   if(nullptr == aLeavesMax) {
      LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdateInternal aLeavesMax was null, so there won't be any splits");
   } else {
      if(0 != cSignificantDimensions) {
         size_t iDimensionInit = 0;
         const IntEbmType * pLeavesMax = aLeavesMax;
         const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
         EBM_ASSERT(1 <= cDimensions);
         const FeatureGroupEntry * const pFeatureGroupEntryEnd = pFeatureGroupEntry + cDimensions;
         do {
            const Feature * pFeature = pFeatureGroupEntry->m_pFeature;
            const size_t cBins = pFeature->GetCountBins();
            if(size_t { 1 } < cBins) {
               EBM_ASSERT(size_t { 2 } <= cSignificantDimensions || IntEbmType { 0 } == lastDimensionLeavesMax);

               iDimensionImportant = iDimensionInit;
               cSignificantBinCount = cBins;
               EBM_ASSERT(nullptr != pLeavesMax);
               const IntEbmType countLeavesMax = *pLeavesMax;
               if(countLeavesMax <= IntEbmType { 1 }) {
                  LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdateInternal countLeavesMax is 1 or less.");
               } else {
                  // keep iteration even once we find this so that we output logs for any bins of 1
                  lastDimensionLeavesMax = countLeavesMax;
               }
            }
            ++iDimensionInit;
            ++pLeavesMax;
            ++pFeatureGroupEntry;
         } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
         
         EBM_ASSERT(size_t { 2 } <= cSignificantBinCount);
      }
   }

   pBoosterShell->GetAccumulatedModelUpdate()->SetCountDimensions(cDimensions);
   pBoosterShell->GetAccumulatedModelUpdate()->Reset();

   // if pBoosterCore->m_apSamplingSets is nullptr, then we should have zero training samples
   // we can't be partially constructed here since then we wouldn't have returned our state pointer to our caller

   double gainAvgOut = 0.0;
   const SamplingSet * const * ppSamplingSet = pBoosterCore->GetSamplingSets();
   if(nullptr != ppSamplingSet) {
      pBoosterShell->GetOverwritableModelUpdate()->SetCountDimensions(cDimensions);
      // if we have ignored dimensions, set the splits count to zero!
      // we only need to do this once instead of per-loop since any dimensions with 1 bin 
      // are going to remain having 0 splits.
      pBoosterShell->GetOverwritableModelUpdate()->Reset();

      EBM_ASSERT(1 <= cSamplingSetsAfterZero);
      const SamplingSet * const * const ppSamplingSetEnd = &ppSamplingSet[cSamplingSetsAfterZero];
      const FloatEbmType invertedSampleCount = FloatEbmType { 1 } / cSamplingSetsAfterZero;
      FloatEbmType gainAvg = FloatEbmType { 0 };
      do {
         const SamplingSet * const pSamplingSet = *ppSamplingSet;
         if(UNLIKELY(IntEbmType { 0 } == lastDimensionLeavesMax)) {
            LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdateInternal boosting zero dimensional");
            error = BoostZeroDimensional(pBoosterShell, pSamplingSet, options);
            if(Error_None != error) {
               if(LIKELY(nullptr != pGainAvgOut)) {
                  *pGainAvgOut = double { 0 };
               }
               return error;
            }
         } else {
            FloatEbmType gain;
            if(0 != (GenerateUpdateOptions_RandomSplits & options) || 2 < cSignificantDimensions) {
               if(size_t { 1 } != cSamplesRequiredForChildSplitMin) {
                  LOG_0(TraceLevelWarning,
                     "WARNING GenerateModelUpdateInternal cSamplesRequiredForChildSplitMin is ignored when doing random splitting"
                  );
               }
               // THIS RANDOM SPLIT OPTION IS PRIMARILY USED FOR DIFFERENTIAL PRIVACY EBMs

               error = BoostRandom(
                  pBoosterShell,
                  pFeatureGroup,
                  pSamplingSet,
                  options,
                  aLeavesMax,
                  &gain
               );
               if(Error_None != error) {
                  if(LIKELY(nullptr != pGainAvgOut)) {
                     *pGainAvgOut = double { 0 };
                  }
                  return error;
               }
            } else if(1 == cSignificantDimensions) {
               EBM_ASSERT(nullptr != aLeavesMax); // otherwise we'd use BoostZeroDimensional above
               EBM_ASSERT(IntEbmType { 2 } <= lastDimensionLeavesMax); // otherwise we'd use BoostZeroDimensional above
               EBM_ASSERT(size_t { 2 } <= cSignificantBinCount); // otherwise we'd use BoostZeroDimensional above

               error = BoostSingleDimensional(
                  pBoosterShell,
                  pFeatureGroup,
                  cSignificantBinCount,
                  pSamplingSet,
                  iDimensionImportant,
                  cSamplesRequiredForChildSplitMin,
                  lastDimensionLeavesMax,
                  &gain
               );
               if(Error_None != error) {
                  if(LIKELY(nullptr != pGainAvgOut)) {
                     *pGainAvgOut = double { 0 };
                  }
                  return error;
               }
            } else {
               error = BoostMultiDimensional(
                  pBoosterShell,
                  pFeatureGroup,
                  pSamplingSet,
                  cSamplesRequiredForChildSplitMin,
                  &gain
               );
               if(Error_None != error) {
                  if(LIKELY(nullptr != pGainAvgOut)) {
                     *pGainAvgOut = double { 0 };
                  }
                  return error;
               }
            }

            // gain should be +inf if there was an overflow in our callees
            EBM_ASSERT(!std::isnan(gain));
            EBM_ASSERT(FloatEbmType { 0 } <= gain);

            const FloatEbmType weightTotal = pSamplingSet->GetWeightTotal();
            EBM_ASSERT(FloatEbmType { 0 } < weightTotal); // if all are zeros we assume there are no weights and use the count

            // this could re-promote gain to be +inf again if weightTotal < 1.0
            // do the sample count inversion here in case adding all the avgeraged gains pushes us into +inf
            EBM_ASSERT(invertedSampleCount <= FloatEbmType { 1 });
            gain = gain * invertedSampleCount / weightTotal;
            gainAvg += gain;
            EBM_ASSERT(!std::isnan(gainAvg));
            EBM_ASSERT(FloatEbmType { 0 } <= gainAvg);
         }

         // TODO : when we thread this code, let's have each thread take a lock and update the combined line segment.  They'll each do it while the 
         // others are working, so there should be no blocking and our final result won't require adding by the main thread
         error = pBoosterShell->GetAccumulatedModelUpdate()->Add(*pBoosterShell->GetOverwritableModelUpdate());
         if(Error_None != error) {
            if(LIKELY(nullptr != pGainAvgOut)) {
               *pGainAvgOut = double { 0 };
            }
            return error;
         }
         ++ppSamplingSet;
      } while(ppSamplingSetEnd != ppSamplingSet);

      // gainAvg is +inf on overflow. It cannot be NaN, but check for that anyways since it's free
      EBM_ASSERT(!std::isnan(gainAvg));
      EBM_ASSERT(FloatEbmType { 0 } <= gainAvg);

      gainAvgOut = static_cast<double>(gainAvg);
      if(UNLIKELY(/* NaN */ !LIKELY(gainAvg <= std::numeric_limits<FloatEbmType>::max()))) {
         // this also checks for NaN since NaN < anything is FALSE

         // indicate an error/overflow with -inf similar to interaction strength.
         // Making it -inf gives it the worst ranking possible and avoids the weirdness of NaN

         // it is possible that some of our inner bags overflowed but others did not
         // in some boosting we allow both an update and an overflow.  We indicate the overflow
         // to the caller via a negative gain, but we pass through any update and let the caller
         // decide if they want to stop boosting at that point or continue.
         // So, if there is an update do not reset it here

         gainAvgOut = k_illegalGainDouble;
      } else {
         EBM_ASSERT(!std::isnan(gainAvg));
         EBM_ASSERT(!std::isinf(gainAvg));
         EBM_ASSERT(FloatEbmType { 0 } <= gainAvg);
      }

      LOG_0(TraceLevelVerbose, "GenerateModelUpdatePerTargetClasses done sampling set loop");

      FloatEbmType learningRateFloat = static_cast<FloatEbmType>(learningRate);
      static_assert(sizeof(learningRateFloat) == sizeof(learningRate), "float mismatch");

      bool bBad;
      // we need to divide by the number of sampling sets that we constructed this from.
      // We also need to slow down our growth so that the more relevant Features get a chance to grow first so we multiply by a user defined learning rate
      if(bClassification) {
#ifdef EXPAND_BINARY_LOGITS
         constexpr bool bExpandBinaryLogits = true;
#else // EXPAND_BINARY_LOGITS
         constexpr bool bExpandBinaryLogits = false;
#endif // EXPAND_BINARY_LOGITS

         //if(0 <= k_iZeroLogit || ptrdiff_t { 2 } == pBoosterCore->m_runtimeLearningTypeOrCountTargetClasses && bExpandBinaryLogits) {
         //   EBM_ASSERT(ptrdiff_t { 2 } <= pBoosterCore->m_runtimeLearningTypeOrCountTargetClasses);
         //   // TODO : for classification with logit zeroing, is our learning rate essentially being inflated as 
         //       pBoosterCore->m_runtimeLearningTypeOrCountTargetClasses goes up?  If so, maybe we should divide by 
         //       pBoosterCore->m_runtimeLearningTypeOrCountTargetClasses here to keep learning rates as equivalent as possible..  
         //       Actually, I think the real solution here is that 
         //   pBoosterCore->m_pSmallChangeToModelAccumulatedFromSamplingSets->Multiply(
         //      learningRateFloat * invertedSampleCount * (pBoosterCore->m_runtimeLearningTypeOrCountTargetClasses - 1) / 
         //      pBoosterCore->m_runtimeLearningTypeOrCountTargetClasses
         //   );
         //} else {
         //   // TODO : for classification, is our learning rate essentially being inflated as 
         //        pBoosterCore->m_runtimeLearningTypeOrCountTargetClasses goes up?  If so, maybe we should divide by 
         //        pBoosterCore->m_runtimeLearningTypeOrCountTargetClasses here to keep learning rates equivalent as possible
         //   pBoosterCore->m_pSmallChangeToModelAccumulatedFromSamplingSets->Multiply(learningRateFloat * invertedSampleCount);
         //}

         // TODO: When NewtonBoosting is enabled, we need to multiply our rate by (K - 1)/K (see above), per:
         // https://arxiv.org/pdf/1810.09092v2.pdf (forumla 5) and also the 
         // Ping Li paper (algorithm #1, line 5, (K - 1) / K )
         // https://arxiv.org/pdf/1006.5051.pdf

         const bool bDividing = bExpandBinaryLogits && ptrdiff_t { 2 } == runtimeLearningTypeOrCountTargetClasses;
         if(bDividing) {
            bBad = pBoosterShell->GetAccumulatedModelUpdate()->MultiplyAndCheckForIssues(learningRateFloat * invertedSampleCount * FloatEbmType { 0.5 });
         } else {
            bBad = pBoosterShell->GetAccumulatedModelUpdate()->MultiplyAndCheckForIssues(learningRateFloat * invertedSampleCount);
         }
      } else {
         bBad = pBoosterShell->GetAccumulatedModelUpdate()->MultiplyAndCheckForIssues(learningRateFloat * invertedSampleCount);
      }

      if(UNLIKELY(bBad)) {
         // our update contains a NaN or -inf or +inf and we cannot tollerate a model that does this, so destroy it

         pBoosterShell->GetAccumulatedModelUpdate()->SetCountDimensions(cDimensions);
         pBoosterShell->GetAccumulatedModelUpdate()->Reset();

         // also, signal to our caller that an overflow occured with a negative gain
         gainAvgOut = k_illegalGainDouble;
      }
   }

   pBoosterShell->SetFeatureGroupIndex(iFeatureGroup);

   EBM_ASSERT(!std::isnan(gainAvgOut));
   EBM_ASSERT(std::numeric_limits<double>::infinity() != gainAvgOut);
   EBM_ASSERT(k_illegalGainDouble == gainAvgOut || double { 0 } <= gainAvgOut);

   if(nullptr != pGainAvgOut) {
      *pGainAvgOut = gainAvgOut;
   }

   LOG_0(TraceLevelVerbose, "Exited GenerateModelUpdatePerTargetClasses");
   return Error_None;
}

// we made this a global because if we had put this variable inside the BoosterCore object, then we would need to dereference that before getting 
// the count.  By making this global we can send a log message incase a bad BoosterCore object is sent into us we only decrease the count if the 
// count is non-zero, so at worst if there is a race condition then we'll output this log message more times than desired, but we can live with that
static int g_cLogGenerateModelUpdateParametersMessages = 10;

// TODO : change this so that our caller allocates the memory that contains the update, but this is complicated in various ways
//        we don't want to just copy the internal tensor into the memory region that our caller provides, and we want to work with
//        compressed representations of the CompressibleTensor object while we're building it, so we'll work within the memory the caller
//        provides, but that means we'll potentially need more memory than the full tensor, and we'll need to put some header info
//        at the start, so the caller can't treat this memory as a pure tensor.
//        So:
//          1) provide a function that returns the maximum memory needed.  A smart caller will call this once on each feature_group, 
//             choose the max and allocate it once
//          2) return a compressed complete CompressibleTensor to the caller inside an opaque memory region 
//             (return the exact size that we require to the caller for copying)
//          3) if caller wants a simplified tensor, then they call a separate function that expands the tensor 
//             and returns a pointer to the memory inside the opaque object
//          4) ApplyModelUpdate will take an opaque CompressibleTensor, and expand it if needed
//        The benefit of returning a compressed object is that we don't have to do the work of expanding it if the caller decides not to use it 
//        (which might happen in greedy algorithms)
//        The other benefit of returning a compressed object is that our caller can store/copy it faster
//        The other benefit of returning a compressed object is that it can be copied from process to process faster
//        Lastly, with the memory allocated by our caller, we can call GenerateModelUpdate in parallel on multiple feature_groups.  
//        Right now you can't call it in parallel since we're updating our internal single tensor

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION GenerateModelUpdate(
   BoosterHandle boosterHandle,
   IntEbmType indexFeatureGroup,
   GenerateUpdateOptionsType options,
   double learningRate,
   IntEbmType countSamplesRequiredForChildSplitMin,
   const IntEbmType * leavesMax,
   double * avgGainOut
) {
   LOG_COUNTED_N(
      &g_cLogGenerateModelUpdateParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "GenerateModelUpdate: "
      "boosterHandle=%p, "
      "indexFeatureGroup=%" IntEbmTypePrintf ", "
      "options=0x%" UGenerateUpdateOptionsTypePrintf ", "
      "learningRate=%le, "
      "countSamplesRequiredForChildSplitMin=%" IntEbmTypePrintf ", "
      "leavesMax=%p, "
      "avgGainOut=%p"
      ,
      static_cast<void *>(boosterHandle),
      indexFeatureGroup,
      static_cast<UGenerateUpdateOptionsType>(options), // signed to unsigned conversion is defined behavior in C++
      learningRate,
      countSamplesRequiredForChildSplitMin,
      static_cast<const void *>(leavesMax),
      static_cast<void *>(avgGainOut)
   );

   ErrorEbmType error;

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      if(LIKELY(nullptr != avgGainOut)) {
         *avgGainOut = double { 0 };
      }
      // already logged
      return Error_IllegalParamValue;
   }

   // set this to illegal so if we exit with an error we have an invalid index
   pBoosterShell->SetFeatureGroupIndex(BoosterShell::k_illegalFeatureGroupIndex);

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(nullptr != pBoosterCore);

   if(indexFeatureGroup < 0) {
      if(LIKELY(nullptr != avgGainOut)) {
         *avgGainOut = double { 0 };
      }
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate indexFeatureGroup must be positive");
      return Error_IllegalParamValue;
   }
   if(IsConvertError<size_t>(indexFeatureGroup)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      if(LIKELY(nullptr != avgGainOut)) {
         *avgGainOut = double { 0 };
      }
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate indexFeatureGroup is too high to index");
      return Error_IllegalParamValue;
   }
   size_t iFeatureGroup = static_cast<size_t>(indexFeatureGroup);
   if(pBoosterCore->GetCountFeatureGroups() <= iFeatureGroup) {
      if(LIKELY(nullptr != avgGainOut)) {
         *avgGainOut = double { 0 };
      }
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate indexFeatureGroup above the number of feature groups that we have");
      return Error_IllegalParamValue;
   }
   // this is true because 0 < pBoosterCore->m_cFeatureGroups since our caller needs to pass in a valid indexFeatureGroup to this function
   EBM_ASSERT(nullptr != pBoosterCore->GetFeatureGroups());

   LOG_COUNTED_0(
      pBoosterCore->GetFeatureGroups()[iFeatureGroup]->GetPointerCountLogEnterGenerateModelUpdateMessages(),
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered GenerateModelUpdate"
   );

   // TODO : test if our GenerateUpdateOptionsType options flags only include flags that we use

   if(std::isnan(learningRate)) {
      LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdate learningRate is NaN");
   } else if(std::numeric_limits<double>::infinity() == learningRate) {
      LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdate learningRate is +infinity");
   } else if(double { std::numeric_limits<FloatEbmType>::max() } < learningRate) {
      LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdate learningRate is +infinity in float32");
   } else if(0.0 == learningRate) {
      LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdate learningRate is zero");
   } else if(learningRate < double { 0 }) {
      LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdate learningRate is negative");
   }

   size_t cSamplesRequiredForChildSplitMin = size_t { 1 }; // this is the min value
   if(IntEbmType { 1 } <= countSamplesRequiredForChildSplitMin) {
      cSamplesRequiredForChildSplitMin = static_cast<size_t>(countSamplesRequiredForChildSplitMin);
      if(IsConvertError<size_t>(countSamplesRequiredForChildSplitMin)) {
         // we can never exceed a size_t number of samples, so let's just set it to the maximum if we were going to overflow because it will generate 
         // the same results as if we used the true number
         cSamplesRequiredForChildSplitMin = std::numeric_limits<size_t>::max();
      }
   } else {
      LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdate countSamplesRequiredForChildSplitMin can't be less than 1.  Adjusting to 1.");
   }

   // leavesMax is handled in GenerateModelUpdateInternal

   // avgGainOut can be nullptr

   if(ptrdiff_t { 0 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses() || ptrdiff_t { 1 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses()) {
      // if there is only 1 target class for classification, then we can predict the output with 100% accuracy.  The model is a tensor with zero 
      // length array logits, which means for our representation that we have zero items in the array total.
      // since we can predit the output with 100% accuracy, our gain will be 0.
      if(LIKELY(nullptr != avgGainOut)) {
         *avgGainOut = double { 0 };
      }
      pBoosterShell->SetFeatureGroupIndex(iFeatureGroup);

      LOG_0(
         TraceLevelWarning,
         "WARNING GenerateModelUpdate pBoosterCore->m_runtimeLearningTypeOrCountTargetClasses <= ptrdiff_t { 1 }"
      );
      return Error_None;
   }

   error = GenerateModelUpdateInternal(
      pBoosterShell,
      iFeatureGroup,
      options,
      learningRate,
      cSamplesRequiredForChildSplitMin,
      leavesMax,
      avgGainOut
   );
   if(Error_None != error) {
      LOG_N(TraceLevelWarning, "WARNING GenerateModelUpdate: return=%" ErrorEbmTypePrintf, error);
      if(LIKELY(nullptr != avgGainOut)) {
         *avgGainOut = double { 0 };
      }
      return error;
   }

   if(nullptr != avgGainOut) {
      EBM_ASSERT(!std::isnan(*avgGainOut)); // NaNs can happen, but we should have edited those before here
      EBM_ASSERT(!std::isinf(*avgGainOut)); // infinities can happen, but we should have edited those before here
      // no epsilon required.  We make it zero if the value is less than zero for floating point instability reasons
      EBM_ASSERT(double { 0 } <= *avgGainOut);
      LOG_COUNTED_N(
         pBoosterCore->GetFeatureGroups()[iFeatureGroup]->GetPointerCountLogExitGenerateModelUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited GenerateModelUpdate: "
         "*avgGainOut=%le"
         ,
         *avgGainOut
      );
   } else {
      LOG_COUNTED_0(
         pBoosterCore->GetFeatureGroups()[iFeatureGroup]->GetPointerCountLogExitGenerateModelUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited GenerateModelUpdate"
      );
   }
   return Error_None;
}

} // DEFINED_ZONE_NAME
