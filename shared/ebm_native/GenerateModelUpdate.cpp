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

#include "TensorTotalsSum.hpp"

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

extern ErrorEbmType PartitionOneDimensionalBoosting(
   BoosterShell * const pBoosterShell,
   const size_t cHistogramBuckets,
   const size_t cSamplesTotal,
   const FloatEbmType weightTotal,
   const size_t cSamplesRequiredForChildSplitMin,
   const size_t cLeavesMax,
   FloatEbmType * const pTotalGain
);

extern ErrorEbmType PartitionTwoDimensionalBoosting(
   BoosterShell * const pBoosterShell,
   const FeatureGroup * const pFeatureGroup,
   const size_t cSamplesRequiredForChildSplitMin,
   HistogramBucketBase * pAuxiliaryBucketZone,
   HistogramBucketBase * const pTotal,
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
   if(GetHistogramBucketSizeOverflow(bClassification, cVectorLength)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)");
      return Error_OutOfMemory;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);

   HistogramBucketBase * const pHistogramBucket = pBoosterShell->GetHistogramBucketBase(cBytesPerHistogramBucket);
   if(UNLIKELY(nullptr == pHistogramBucket)) {
      // already logged
      return Error_OutOfMemory;
   }

   if(bClassification) {
      pHistogramBucket->GetHistogramBucket<true>()->Zero(cVectorLength);
   } else {
      pHistogramBucket->GetHistogramBucket<false>()->Zero(cVectorLength);
   }

#ifndef NDEBUG
   pBoosterShell->SetHistogramBucketsEndDebug(reinterpret_cast<unsigned char *>(pHistogramBucket) + cBytesPerHistogramBucket);
#endif // NDEBUG

   BinBoosting(
      pBoosterShell,
      nullptr,
      pTrainingSet
   );

   CompressibleTensor * const pSmallChangeToModelOverwriteSingleSamplingSet = 
      pBoosterShell->GetOverwritableModelUpdate();
   FloatEbmType * aValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
   if(bClassification) {
      const HistogramBucket<true> * const pHistogramBucketLocal = pHistogramBucket->GetHistogramBucket<true>();
      const HistogramTargetEntry<true> * const aSumHistogramTargetEntry =
         pHistogramBucketLocal->GetHistogramTargetEntry();
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
      const HistogramBucket<false> * const pHistogramBucketLocal = pHistogramBucket->GetHistogramBucket<false>();
      const HistogramTargetEntry<false> * const aSumHistogramTargetEntry =
         pHistogramBucketLocal->GetHistogramTargetEntry();
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
   if(GetHistogramBucketSizeOverflow(bClassification, cVectorLength)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "WARNING GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)");
      return Error_OutOfMemory;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);
   if(IsMultiplyError(cBytesPerHistogramBucket, cHistogramBuckets)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "WARNING IsMultiplyError(cBytesPerHistogramBucket, cHistogramBuckets)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBuffer = cBytesPerHistogramBucket * cHistogramBuckets;

   HistogramBucketBase * const aHistogramBuckets = pBoosterShell->GetHistogramBucketBase(cBytesBuffer);
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      // already logged
      return Error_OutOfMemory;
   }

   HistogramTargetEntryBase * const aSumHistogramTargetEntry =
      pBoosterShell->GetSumHistogramTargetEntryArray();

   if(bClassification) {
      HistogramBucket<true> * const aHistogramBucketsLocal = aHistogramBuckets->GetHistogramBucket<true>();
      for(size_t i = 0; i < cHistogramBuckets; ++i) {
         HistogramBucket<true> * const pHistogramBucket =
            GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBucketsLocal, i);
         pHistogramBucket->Zero(cVectorLength);
      }

      HistogramTargetEntry<true> * const aSumHistogramTargetEntryLocal = aSumHistogramTargetEntry->GetHistogramTargetEntry<true>();
      for(size_t i = 0; i < cVectorLength; ++i) {
         aSumHistogramTargetEntryLocal[i].Zero();
      }
   } else {
      HistogramBucket<false> * const aHistogramBucketsLocal = aHistogramBuckets->GetHistogramBucket<false>();
      for(size_t i = 0; i < cHistogramBuckets; ++i) {
         HistogramBucket<false> * const pHistogramBucket =
            GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBucketsLocal, i);
         pHistogramBucket->Zero(cVectorLength);
      }

      HistogramTargetEntry<false> * const aSumHistogramTargetEntryLocal = aSumHistogramTargetEntry->GetHistogramTargetEntry<false>();
      for(size_t i = 0; i < cVectorLength; ++i) {
         aSumHistogramTargetEntryLocal[i].Zero();
      }
   }

#ifndef NDEBUG
   pBoosterShell->SetHistogramBucketsEndDebug(reinterpret_cast<unsigned char *>(aHistogramBuckets) + cBytesBuffer);
#endif // NDEBUG

   BinBoosting(
      pBoosterShell,
      pFeatureGroup,
      pTrainingSet
   );

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
   // we need to reserve 4 PAST the pointer we pass into SweepMultiDimensional!!!!.  We pass in index 20 at max, so we need 24
   const size_t cAuxillaryBucketsForSplitting = 24;
   const size_t cAuxillaryBuckets =
      cAuxillaryBucketsForBuildFastTotals < cAuxillaryBucketsForSplitting ? cAuxillaryBucketsForSplitting : cAuxillaryBucketsForBuildFastTotals;
   if(IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)");
      return Error_OutOfMemory;
   }
   const size_t cTotalBuckets = cTotalBucketsMainSpace + cAuxillaryBuckets;

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);
   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
   if(GetHistogramBucketSizeOverflow(bClassification, cVectorLength)) {
      LOG_0(
         TraceLevelWarning,
         "WARNING BoostMultiDimensional GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)"
      );
      return Error_OutOfMemory;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);
   if(IsMultiplyError(cBytesPerHistogramBucket, cTotalBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional IsMultiplyError(cBytesPerHistogramBucket, cTotalBuckets)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBuffer = cBytesPerHistogramBucket * cTotalBuckets;

   // we don't need to free this!  It's tracked and reused by pBoosterShell
   HistogramBucketBase * const aHistogramBuckets = pBoosterShell->GetHistogramBucketBase(cBytesBuffer);
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      // already logged
      return Error_OutOfMemory;
   }

   if(bClassification) {
      HistogramBucket<true> * const aHistogramBucketsLocal = aHistogramBuckets->GetHistogramBucket<true>();
      for(size_t i = 0; i < cTotalBuckets; ++i) {
         HistogramBucket<true> * const pHistogramBucket =
            GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBucketsLocal, i);
         pHistogramBucket->Zero(cVectorLength);
      }
   } else {
      HistogramBucket<false> * const aHistogramBucketsLocal = aHistogramBuckets->GetHistogramBucket<false>();
      for(size_t i = 0; i < cTotalBuckets; ++i) {
         HistogramBucket<false> * const pHistogramBucket =
            GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBucketsLocal, i);
         pHistogramBucket->Zero(cVectorLength);
      }
   }

   HistogramBucketBase * pAuxiliaryBucketZone = GetHistogramBucketByIndex(
      cBytesPerHistogramBucket,
      aHistogramBuckets,
      cTotalBucketsMainSpace
   );

#ifndef NDEBUG
   const unsigned char * const aHistogramBucketsEndDebug = reinterpret_cast<unsigned char *>(aHistogramBuckets) + cBytesBuffer;
   pBoosterShell->SetHistogramBucketsEndDebug(aHistogramBucketsEndDebug);
#endif // NDEBUG

   BinBoosting(
      pBoosterShell,
      pFeatureGroup,
      pTrainingSet
   );

#ifndef NDEBUG
   // make a copy of the original binned buckets for debugging purposes

   HistogramBucketBase * const aHistogramBucketsDebugCopy =
      EbmMalloc<HistogramBucketBase>(cTotalBucketsMainSpace, cBytesPerHistogramBucket);
   if(nullptr != aHistogramBucketsDebugCopy) {
      // if we can't allocate, don't fail.. just stop checking
      const size_t cBytesBufferDebug = cTotalBucketsMainSpace * cBytesPerHistogramBucket;
      memcpy(aHistogramBucketsDebugCopy, aHistogramBuckets, cBytesBufferDebug);
   }
#endif // NDEBUG

   TensorTotalsBuild(
      runtimeLearningTypeOrCountTargetClasses,
      pFeatureGroup,
      pAuxiliaryBucketZone,
      aHistogramBuckets
#ifndef NDEBUG
      , aHistogramBucketsDebugCopy
      , aHistogramBucketsEndDebug
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
      HistogramBucketBase * const pTotal = GetHistogramBucketByIndex(
         cBytesPerHistogramBucket,
         aHistogramBuckets,
         cTotalBucketsMainSpace - 1
      );

      error = PartitionTwoDimensionalBoosting(
         pBoosterShell,
         pFeatureGroup,
         cSamplesRequiredForChildSplitMin,
         pAuxiliaryBucketZone,
         pTotal,
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

      // gain can be -infinity for regression in a super-super-super-rare condition.  
      // See notes above regarding "gain = bestSplittingScore - splittingScoreParent"

      // within a set, no split should make our model worse.  It might in our validation set, but not within the training set
      EBM_ASSERT(std::isnan(*pTotalGain) || (!bClassification) && std::isinf(*pTotalGain) ||
         k_epsilonNegativeGainAllowed <= *pTotalGain);
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
   if(GetHistogramBucketSizeOverflow(bClassification, cVectorLength)) {
      LOG_0(
         TraceLevelWarning,
         "WARNING BoostRandom GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)"
      );
      return Error_OutOfMemory;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);
   if(IsMultiplyError(cBytesPerHistogramBucket, cTotalBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING BoostRandom IsMultiplyError(cBytesPerHistogramBucket, cTotalBuckets)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBuffer = cBytesPerHistogramBucket * cTotalBuckets;

   // we don't need to free this!  It's tracked and reused by pBoosterShell
   HistogramBucketBase * const aHistogramBuckets = pBoosterShell->GetHistogramBucketBase(cBytesBuffer);
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      // already logged
      return Error_OutOfMemory;
   }

   if(bClassification) {
      HistogramBucket<true> * const aHistogramBucketsLocal = aHistogramBuckets->GetHistogramBucket<true>();
      for(size_t i = 0; i < cTotalBuckets; ++i) {
         HistogramBucket<true> * const pHistogramBucket =
            GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBucketsLocal, i);
         pHistogramBucket->Zero(cVectorLength);
      }
   } else {
      HistogramBucket<false> * const aHistogramBucketsLocal = aHistogramBuckets->GetHistogramBucket<false>();
      for(size_t i = 0; i < cTotalBuckets; ++i) {
         HistogramBucket<false> * const pHistogramBucket =
            GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBucketsLocal, i);
         pHistogramBucket->Zero(cVectorLength);
      }
   }

#ifndef NDEBUG
   pBoosterShell->SetHistogramBucketsEndDebug(reinterpret_cast<unsigned char *>(aHistogramBuckets) + cBytesBuffer);
#endif // NDEBUG

   BinBoosting(
      pBoosterShell,
      pFeatureGroup,
      pTrainingSet
   );

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

   // gain can be -infinity for regression in a super-super-super-rare condition.  
   // See notes above regarding "gain = bestSplittingScore - splittingScoreParent"

   // within a set, no split should make our model worse.  It might in our validation set, but not within the training set
   EBM_ASSERT(std::isnan(*pTotalGain) || (!bClassification) && std::isinf(*pTotalGain) ||
      k_epsilonNegativeGainAllowed <= *pTotalGain);

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
   const FloatEbmType learningRate,
   const size_t cSamplesRequiredForChildSplitMin,
   const IntEbmType * const aLeavesMax, 
   FloatEbmType * const pGainReturn
) {
   ErrorEbmType error;

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelVerbose, "Entered GenerateModelUpdateInternal");

   const size_t cSamplingSetsAfterZero = (0 == pBoosterCore->GetCountSamplingSets()) ? 1 : pBoosterCore->GetCountSamplingSets();
   const FeatureGroup * const pFeatureGroup = pBoosterCore->GetFeatureGroups()[iFeatureGroup];
   const size_t cSignificantDimensions = pFeatureGroup->GetCountSignificantDimensions();

   IntEbmType lastDimensionLeavesMax = IntEbmType { 0 };
   // this initialization isn't required, but this variable ends up touching a lot of downstream state
   // and g++ seems to warn about all of that usage, even in other downstream functions!
   size_t cSignificantBinCount = size_t { 0 };
   if(nullptr == aLeavesMax) {
      LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdateInternal aLeavesMax was null, so there won't be any splits");
   } else {
      if(0 != cSignificantDimensions) {
         const IntEbmType * pLeavesMax = aLeavesMax;
         const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
         EBM_ASSERT(1 <= pFeatureGroup->GetCountDimensions());
         const FeatureGroupEntry * const pFeatureGroupEntryEnd = pFeatureGroupEntry + pFeatureGroup->GetCountDimensions();
         do {
            const Feature * pFeature = pFeatureGroupEntry->m_pFeature;
            const size_t cBins = pFeature->GetCountBins();
            if(size_t { 1 } < cBins) {
               EBM_ASSERT(size_t { 2 } <= cSignificantDimensions || IntEbmType { 0 } == lastDimensionLeavesMax);

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
            ++pLeavesMax;
            ++pFeatureGroupEntry;
         } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
         
         EBM_ASSERT(size_t { 2 } <= cSignificantBinCount);
      }
   }

   pBoosterShell->GetAccumulatedModelUpdate()->SetCountDimensions(cSignificantDimensions);
   pBoosterShell->GetAccumulatedModelUpdate()->Reset();

   // if pBoosterCore->m_apSamplingSets is nullptr, then we should have zero training samples
   // we can't be partially constructed here since then we wouldn't have returned our state pointer to our caller

   FloatEbmType totalGain = FloatEbmType { 0 };
   if(nullptr != pBoosterCore->GetSamplingSets()) {
      pBoosterShell->GetOverwritableModelUpdate()->SetCountDimensions(cSignificantDimensions);

      for(size_t iSamplingSet = 0; iSamplingSet < cSamplingSetsAfterZero; ++iSamplingSet) {
         FloatEbmType gain;
         if(UNLIKELY(IntEbmType { 0 } == lastDimensionLeavesMax)) {
            LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdateInternal boosting zero dimensional");
            error =
               BoostZeroDimensional(pBoosterShell, pBoosterCore->GetSamplingSets()[iSamplingSet], options);
            if(Error_None != error) {
               if(LIKELY(nullptr != pGainReturn)) {
                  *pGainReturn = FloatEbmType { 0 };
               }
               return error;
            }
            gain = FloatEbmType { 0 };
         } else if(0 != (GenerateUpdateOptions_RandomSplits & options) || 2 < cSignificantDimensions) {
            if(size_t { 1 } != cSamplesRequiredForChildSplitMin) {
               LOG_0(TraceLevelWarning, 
                  "WARNING GenerateModelUpdateInternal cSamplesRequiredForChildSplitMin is ignored when doing random splitting"
               );
            }
            // THIS RANDOM SPLIT OPTION IS PRIMARILY USED FOR DIFFERENTIAL PRIVACY EBMs

            error = BoostRandom(
               pBoosterShell,
               pFeatureGroup,
               pBoosterCore->GetSamplingSets()[iSamplingSet],
               options,
               aLeavesMax,
               &gain
            );
            if(Error_None != error) {
               if(LIKELY(nullptr != pGainReturn)) {
                  *pGainReturn = FloatEbmType { 0 };
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
               pBoosterCore->GetSamplingSets()[iSamplingSet],
               cSamplesRequiredForChildSplitMin,
               lastDimensionLeavesMax,
               &gain
            );
            if(Error_None != error) {
               if(LIKELY(nullptr != pGainReturn)) {
                  *pGainReturn = FloatEbmType { 0 };
               }
               return error;
            }
         } else {
            error = BoostMultiDimensional(
               pBoosterShell,
               pFeatureGroup,
               pBoosterCore->GetSamplingSets()[iSamplingSet],
               cSamplesRequiredForChildSplitMin,
               &gain
            );
            if(Error_None != error) {
               if(LIKELY(nullptr != pGainReturn)) {
                  *pGainReturn = FloatEbmType { 0 };
               }
               return error;
            }
         }
         // regression can be -infinity or slightly negative in extremely rare circumstances.  
         // See ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint for details, and the equivalent interaction function
         EBM_ASSERT(std::isnan(gain) || (!bClassification) && std::isinf(gain) || k_epsilonNegativeGainAllowed <= gain); // we previously normalized to 0
         totalGain += gain;
         // TODO : when we thread this code, let's have each thread take a lock and update the combined line segment.  They'll each do it while the 
         // others are working, so there should be no blocking and our final result won't require adding by the main thread
         error = pBoosterShell->GetAccumulatedModelUpdate()->Add(*pBoosterShell->GetOverwritableModelUpdate());
         if(Error_None != error) {
            if(LIKELY(nullptr != pGainReturn)) {
               *pGainReturn = FloatEbmType { 0 };
            }
            return error;
         }
      }
      totalGain /= static_cast<FloatEbmType>(cSamplingSetsAfterZero);
      // regression can be -infinity or slightly negative in extremely rare circumstances.  
      // See ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint for details, and the equivalent interaction function
      EBM_ASSERT(std::isnan(totalGain) || (!bClassification) && std::isinf(totalGain) || k_epsilonNegativeGainAllowed <= totalGain);

      LOG_0(TraceLevelVerbose, "GenerateModelUpdatePerTargetClasses done sampling set loop");

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
         //      learningRate / cSamplingSetsAfterZero * (pBoosterCore->m_runtimeLearningTypeOrCountTargetClasses - 1) / 
         //      pBoosterCore->m_runtimeLearningTypeOrCountTargetClasses
         //   );
         //} else {
         //   // TODO : for classification, is our learning rate essentially being inflated as 
         //        pBoosterCore->m_runtimeLearningTypeOrCountTargetClasses goes up?  If so, maybe we should divide by 
         //        pBoosterCore->m_runtimeLearningTypeOrCountTargetClasses here to keep learning rates equivalent as possible
         //   pBoosterCore->m_pSmallChangeToModelAccumulatedFromSamplingSets->Multiply(learningRate / cSamplingSetsAfterZero);
         //}

         // TODO: When NewtonBoosting is enabled, we need to multiply our rate by (K - 1)/K (see above), per:
         // https://arxiv.org/pdf/1810.09092v2.pdf (forumla 5) and also the 
         // Ping Li paper (algorithm #1, line 5, (K - 1) / K )
         // https://arxiv.org/pdf/1006.5051.pdf

         const bool bDividing = bExpandBinaryLogits && ptrdiff_t { 2 } == runtimeLearningTypeOrCountTargetClasses;
         if(bDividing) {
            bBad = pBoosterShell->GetAccumulatedModelUpdate()->MultiplyAndCheckForIssues(learningRate / cSamplingSetsAfterZero / 2);
         } else {
            bBad = pBoosterShell->GetAccumulatedModelUpdate()->MultiplyAndCheckForIssues(learningRate / cSamplingSetsAfterZero);
         }
      } else {
         bBad = pBoosterShell->GetAccumulatedModelUpdate()->MultiplyAndCheckForIssues(learningRate / cSamplingSetsAfterZero);
      }

      // handle the case where totalGain is either +infinity or -infinity (very rare, see above), or NaN
      // don't use std::inf because with some compiler flags on some compilers that isn't reliable
      if(UNLIKELY(
         UNLIKELY(bBad) || 
         UNLIKELY(std::isnan(totalGain)) || 
         UNLIKELY(totalGain <= std::numeric_limits<FloatEbmType>::lowest()) ||
         UNLIKELY(std::numeric_limits<FloatEbmType>::max() <= totalGain)
      )) {
         pBoosterShell->GetAccumulatedModelUpdate()->SetCountDimensions(cSignificantDimensions);
         pBoosterShell->GetAccumulatedModelUpdate()->Reset();
         // declare there is no gain, so that our caller will think there is no benefit in splitting us, which there isn't since we're zeroed.
         totalGain = FloatEbmType { 0 };
      } else if(UNLIKELY(totalGain < FloatEbmType { 0 })) {
         totalGain = FloatEbmType { 0 };
      }
   }

   pBoosterShell->SetFeatureGroupIndex(iFeatureGroup);

   if(nullptr != pGainReturn) {
      *pGainReturn = totalGain;
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
   FloatEbmType learningRate,
   IntEbmType countSamplesRequiredForChildSplitMin,
   const IntEbmType * leavesMax,
   FloatEbmType * gainOut
) {
   LOG_COUNTED_N(
      &g_cLogGenerateModelUpdateParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "GenerateModelUpdate: "
      "boosterHandle=%p, "
      "indexFeatureGroup=%" IntEbmTypePrintf ", "
      "options=0x%" UGenerateUpdateOptionsTypePrintf ", "
      "learningRate=%" FloatEbmTypePrintf ", "
      "countSamplesRequiredForChildSplitMin=%" IntEbmTypePrintf ", "
      "leavesMax=%p, "
      "gainOut=%p"
      ,
      static_cast<void *>(boosterHandle),
      indexFeatureGroup,
      static_cast<UGenerateUpdateOptionsType>(options), // signed to unsigned conversion is defined behavior in C++
      learningRate,
      countSamplesRequiredForChildSplitMin,
      static_cast<const void *>(leavesMax),
      static_cast<void *>(gainOut)
   );

   ErrorEbmType error;

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromBoosterHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      if(LIKELY(nullptr != gainOut)) {
         *gainOut = FloatEbmType { 0 };
      }
      // already logged
      return Error_IllegalParamValue;
   }

   // set this to illegal so if we exit with an error we have an invalid index
   pBoosterShell->SetFeatureGroupIndex(BoosterShell::k_illegalFeatureGroupIndex);

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(nullptr != pBoosterCore);

   if(indexFeatureGroup < 0) {
      if(LIKELY(nullptr != gainOut)) {
         *gainOut = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate indexFeatureGroup must be positive");
      return Error_IllegalParamValue;
   }
   if(IsConvertError<size_t>(indexFeatureGroup)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      if(LIKELY(nullptr != gainOut)) {
         *gainOut = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate indexFeatureGroup is too high to index");
      return Error_IllegalParamValue;
   }
   size_t iFeatureGroup = static_cast<size_t>(indexFeatureGroup);
   if(pBoosterCore->GetCountFeatureGroups() <= iFeatureGroup) {
      if(LIKELY(nullptr != gainOut)) {
         *gainOut = FloatEbmType { 0 };
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
   } else if(std::isinf(learningRate)) {
      LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdate learningRate is infinity");
   } else if(0 == learningRate) {
      LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdate learningRate is zero");
   } else if(learningRate < FloatEbmType { 0 }) {
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

   // gainOut can be nullptr

   if(ptrdiff_t { 0 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses() || ptrdiff_t { 1 } == pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses()) {
      // if there is only 1 target class for classification, then we can predict the output with 100% accuracy.  The model is a tensor with zero 
      // length array logits, which means for our representation that we have zero items in the array total.
      // since we can predit the output with 100% accuracy, our gain will be 0.
      if(LIKELY(nullptr != gainOut)) {
         *gainOut = FloatEbmType { 0 };
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
      gainOut
   );

   if(nullptr != gainOut) {
      EBM_ASSERT(!std::isnan(*gainOut)); // NaNs can happen, but we should have edited those before here
      EBM_ASSERT(!std::isinf(*gainOut)); // infinities can happen, but we should have edited those before here
      // no epsilon required.  We make it zero if the value is less than zero for floating point instability reasons
      EBM_ASSERT(FloatEbmType { 0 } <= *gainOut);
      LOG_COUNTED_N(
         pBoosterCore->GetFeatureGroups()[iFeatureGroup]->GetPointerCountLogExitGenerateModelUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited GenerateModelUpdate %" FloatEbmTypePrintf,
         *gainOut
      );
   } else {
      LOG_COUNTED_0(
         pBoosterCore->GetFeatureGroups()[iFeatureGroup]->GetPointerCountLogExitGenerateModelUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited GenerateModelUpdate no gain"
      );
   }
   return error;
}

} // DEFINED_ZONE_NAME
