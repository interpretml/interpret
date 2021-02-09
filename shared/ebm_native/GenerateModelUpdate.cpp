// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebm_native.h"
#include "EbmInternal.h"
// very independent includes
#include "Logging.h" // EBM_ASSERT & LOG
#include "SegmentedTensor.h"
#include "EbmStats.h"
// feature includes
#include "FeatureAtomic.h"
// FeatureGroup.h depends on FeatureInternal.h
#include "FeatureGroup.h"
// dataset depends on features
#include "DataSetBoosting.h"
// samples is somewhat independent from datasets, but relies on an indirect coupling with them
#include "SamplingSet.h"
#include "HistogramTargetEntry.h"
#include "HistogramBucket.h"

#include "Booster.h"
#include "ThreadStateBoosting.h"

#include "TensorTotalsSum.h"

extern void BinBoosting(
   ThreadStateBoosting * const pThreadStateBoosting,
   const FeatureGroup * const pFeatureGroup,
   const SamplingSet * const pTrainingSet
);

extern void SumHistogramBuckets(
   ThreadStateBoosting * const pThreadStateBoosting,
   const size_t cHistogramBuckets
#ifndef NDEBUG
   , const size_t cSamplesTotal
#endif // NDEBUG
);

extern bool GrowDecisionTree(
   ThreadStateBoosting * const pThreadStateBoosting,
   const size_t cHistogramBuckets,
   const size_t cSamplesTotal,
   const size_t cSamplesRequiredForChildSplitMin,
   const size_t cLeavesMax,
   FloatEbmType * const pTotalGain
);

extern bool FindBestBoostingSplitPairs(
   ThreadStateBoosting * const pThreadStateBoosting,
   const FeatureGroup * const pFeatureGroup,
   const size_t cSamplesRequiredForChildSplitMin,
   HistogramBucketBase * pAuxiliaryBucketZone,
   HistogramBucketBase * const pTotal,
   FloatEbmType * const pTotalGain
#ifndef NDEBUG
   , const HistogramBucketBase * const aHistogramBucketsDebugCopy
#endif // NDEBUG
);

extern bool CutRandom(
   ThreadStateBoosting * const pThreadStateBoosting,
   const FeatureGroup * const pFeatureGroup,
   const GenerateUpdateOptionsType options,
   const IntEbmType * const aLeavesMax,
   FloatEbmType * const pTotalGain
);

static bool BoostZeroDimensional(
   ThreadStateBoosting * const pThreadStateBoosting, 
   const SamplingSet * const pTrainingSet,
   const GenerateUpdateOptionsType options
) {
   LOG_0(TraceLevelVerbose, "Entered BoostZeroDimensional");

   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
   if(GetHistogramBucketSizeOverflow(bClassification, cVectorLength)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)");
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);

   HistogramBucketBase * const pHistogramBucket =
      pThreadStateBoosting->GetHistogramBucketBase(cBytesPerHistogramBucket);

   if(UNLIKELY(nullptr == pHistogramBucket)) {
      LOG_0(TraceLevelWarning, "WARNING nullptr == pHistogramBucket");
      return true;
   }

   if(bClassification) {
      pHistogramBucket->GetHistogramBucket<true>()->Zero(cVectorLength);
   } else {
      pHistogramBucket->GetHistogramBucket<false>()->Zero(cVectorLength);
   }

#ifndef NDEBUG
   pThreadStateBoosting->SetHistogramBucketsEndDebug(reinterpret_cast<unsigned char *>(pHistogramBucket) + cBytesPerHistogramBucket);
#endif // NDEBUG

   BinBoosting(
      pThreadStateBoosting,
      nullptr,
      pTrainingSet
   );

   SegmentedTensor * const pSmallChangeToModelOverwriteSingleSamplingSet = 
      pThreadStateBoosting->GetSmallChangeToModelOverwriteSingleSamplingSet();
   FloatEbmType * aValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
   if(bClassification) {
      const HistogramBucket<true> * const pHistogramBucketLocal = pHistogramBucket->GetHistogramBucket<true>();
      const HistogramBucketVectorEntry<true> * const aSumHistogramBucketVectorEntry =
         pHistogramBucketLocal->GetHistogramBucketVectorEntry();
      if(0 != (GenerateUpdateOptions_GradientSums & options)) {
         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            const FloatEbmType smallChangeToModel = aSumHistogramBucketVectorEntry[iVector].m_sumResidualError;
            aValues[iVector] = smallChangeToModel;
         }
      } else {
         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            const FloatEbmType smallChangeToModel = EbmStats::ComputeSmallChangeForOneSegmentClassificationLogOdds(
               aSumHistogramBucketVectorEntry[iVector].m_sumResidualError,
               aSumHistogramBucketVectorEntry[iVector].GetSumDenominator()
            );
            aValues[iVector] = smallChangeToModel;
         }
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      const HistogramBucket<false> * const pHistogramBucketLocal = pHistogramBucket->GetHistogramBucket<false>();
      const HistogramBucketVectorEntry<false> * const aSumHistogramBucketVectorEntry =
         pHistogramBucketLocal->GetHistogramBucketVectorEntry();
      if(0 != (GenerateUpdateOptions_GradientSums & options)) {
         const FloatEbmType smallChangeToModel = aSumHistogramBucketVectorEntry[0].m_sumResidualError;
         aValues[0] = smallChangeToModel;
      } else {
         const FloatEbmType smallChangeToModel = EbmStats::ComputeSmallChangeForOneSegmentRegression(
            aSumHistogramBucketVectorEntry[0].m_sumResidualError,
            static_cast<FloatEbmType>(pHistogramBucketLocal->GetCountSamplesInBucket())
         );
         aValues[0] = smallChangeToModel;
      }
   }

   LOG_0(TraceLevelVerbose, "Exited BoostZeroDimensional");
   return false;
}

static bool BoostSingleDimensional(
   ThreadStateBoosting * const pThreadStateBoosting,
   const FeatureGroup * const pFeatureGroup,
   const size_t cHistogramBuckets,
   const SamplingSet * const pTrainingSet,
   const size_t cSamplesRequiredForChildSplitMin,
   const IntEbmType countLeavesMax,
   FloatEbmType * const pTotalGain
) {
   LOG_0(TraceLevelVerbose, "Entered BoostSingleDimensional");

   EBM_ASSERT(1 == pFeatureGroup->GetCountSignificantFeatures());

   EBM_ASSERT(IntEbmType { 2 } <= countLeavesMax); // otherwise we would have called BoostZeroDimensional
   size_t cLeavesMax = static_cast<size_t>(countLeavesMax);
   if(!IsNumberConvertable<size_t>(countLeavesMax)) {
      // we can never exceed a size_t number of leaves, so let's just set it to the maximum if we were going to overflow because it will generate 
      // the same results as if we used the true number
      cLeavesMax = std::numeric_limits<size_t>::max();
   }

   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);
   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
   if(GetHistogramBucketSizeOverflow(bClassification, cVectorLength)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "WARNING GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)");
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);
   if(IsMultiplyError(cHistogramBuckets, cBytesPerHistogramBucket)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "WARNING IsMultiplyError(cHistogramBuckets, cBytesPerHistogramBucket)");
      return true;
   }
   const size_t cBytesBuffer = cHistogramBuckets * cBytesPerHistogramBucket;

   HistogramBucketBase * const aHistogramBuckets = pThreadStateBoosting->GetHistogramBucketBase(cBytesBuffer);
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING BoostSingleDimensional nullptr == aHistogramBuckets");
      return true;
   }

   HistogramBucketVectorEntryBase * const aSumHistogramBucketVectorEntry =
      pThreadStateBoosting->GetSumHistogramBucketVectorEntryArray();

   if(bClassification) {
      HistogramBucket<true> * const aHistogramBucketsLocal = aHistogramBuckets->GetHistogramBucket<true>();
      for(size_t i = 0; i < cHistogramBuckets; ++i) {
         HistogramBucket<true> * const pHistogramBucket =
            GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBucketsLocal, i);
         pHistogramBucket->Zero(cVectorLength);
      }

      HistogramBucketVectorEntry<true> * const aSumHistogramBucketVectorEntryLocal = aSumHistogramBucketVectorEntry->GetHistogramBucketVectorEntry<true>();
      for(size_t i = 0; i < cVectorLength; ++i) {
         aSumHistogramBucketVectorEntryLocal[i].Zero();
      }
   } else {
      HistogramBucket<false> * const aHistogramBucketsLocal = aHistogramBuckets->GetHistogramBucket<false>();
      for(size_t i = 0; i < cHistogramBuckets; ++i) {
         HistogramBucket<false> * const pHistogramBucket =
            GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBucketsLocal, i);
         pHistogramBucket->Zero(cVectorLength);
      }

      HistogramBucketVectorEntry<false> * const aSumHistogramBucketVectorEntryLocal = aSumHistogramBucketVectorEntry->GetHistogramBucketVectorEntry<false>();
      for(size_t i = 0; i < cVectorLength; ++i) {
         aSumHistogramBucketVectorEntryLocal[i].Zero();
      }
   }

#ifndef NDEBUG
   pThreadStateBoosting->SetHistogramBucketsEndDebug(reinterpret_cast<unsigned char *>(aHistogramBuckets) + cBytesBuffer);
#endif // NDEBUG

   BinBoosting(
      pThreadStateBoosting,
      pFeatureGroup,
      pTrainingSet
   );

   SumHistogramBuckets(
      pThreadStateBoosting,
      cHistogramBuckets
#ifndef NDEBUG
      , pTrainingSet->GetTotalCountSampleOccurrences()
#endif // NDEBUG
   );

   const size_t cSamplesTotal = pTrainingSet->GetTotalCountSampleOccurrences();
   EBM_ASSERT(1 <= cSamplesTotal);

   bool bRet = GrowDecisionTree(
      pThreadStateBoosting,
      cHistogramBuckets,
      cSamplesTotal,
      cSamplesRequiredForChildSplitMin,
      cLeavesMax, 
      pTotalGain
   );

   LOG_0(TraceLevelVerbose, "Exited BoostSingleDimensional");
   return bRet;
}

// TODO: for higher dimensional spaces, we need to add/subtract individual cells alot and the denominator isn't required in order to make decisions about
//   where to cut.  For dimensions higher than 2, we might want to copy the tensor to a new tensor AFTER binning that keeps only the residuals and then 
//    go back to our original tensor after splits to determine the denominator
static bool BoostMultiDimensional(
   ThreadStateBoosting * const pThreadStateBoosting,
   const FeatureGroup * const pFeatureGroup,
   const SamplingSet * const pTrainingSet,
   const size_t cSamplesRequiredForChildSplitMin,
   FloatEbmType * const pTotalGain
) {
   LOG_0(TraceLevelVerbose, "Entered BoostMultiDimensional");

   EBM_ASSERT(2 <= pFeatureGroup->GetCountFeatures());
   EBM_ASSERT(2 <= pFeatureGroup->GetCountSignificantFeatures());

   size_t cAuxillaryBucketsForBuildFastTotals = 0;
   size_t cTotalBucketsMainSpace = 1;

   const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
   const FeatureGroupEntry * const pFeatureGroupEntryEnd = pFeatureGroupEntry + pFeatureGroup->GetCountFeatures();
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
         // we check for simple multiplication overflow from m_cBins in Booster->Initialize when we unpack featureGroupsFeatureIndexes
         EBM_ASSERT(!IsMultiplyError(cTotalBucketsMainSpace, cBins));
         cTotalBucketsMainSpace *= cBins;
         // if this wasn't true then we'd have to check IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace) at runtime
         EBM_ASSERT(cAuxillaryBucketsForBuildFastTotals < cTotalBucketsMainSpace);
      }
      ++pFeatureGroupEntry;
   } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
   // we need to reserve 4 PAST the pointer we pass into SweepMultiDiemensional!!!!.  We pass in index 20 at max, so we need 24
   const size_t cAuxillaryBucketsForSplitting = 24;
   const size_t cAuxillaryBuckets =
      cAuxillaryBucketsForBuildFastTotals < cAuxillaryBucketsForSplitting ? cAuxillaryBucketsForSplitting : cAuxillaryBucketsForBuildFastTotals;
   if(IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)");
      return true;
   }
   const size_t cTotalBuckets = cTotalBucketsMainSpace + cAuxillaryBuckets;

   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);
   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
   if(GetHistogramBucketSizeOverflow(bClassification, cVectorLength)) {
      LOG_0(
         TraceLevelWarning,
         "WARNING BoostMultiDimensional GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)"
      );
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);
   if(IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)) {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)");
      return true;
   }
   const size_t cBytesBuffer = cTotalBuckets * cBytesPerHistogramBucket;

   // we don't need to free this!  It's tracked and reused by pThreadStateBoosting
   HistogramBucketBase * const aHistogramBuckets = pThreadStateBoosting->GetHistogramBucketBase(cBytesBuffer);
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional nullptr == aHistogramBuckets");
      return true;
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
   pThreadStateBoosting->SetHistogramBucketsEndDebug(aHistogramBucketsEndDebug);
#endif // NDEBUG

   BinBoosting(
      pThreadStateBoosting,
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

   if(2 == pFeatureGroup->GetCountSignificantFeatures()) {
      HistogramBucketBase * const pTotal = GetHistogramBucketByIndex(
         cBytesPerHistogramBucket,
         aHistogramBuckets,
         cTotalBucketsMainSpace - 1
      );

      bool bError = FindBestBoostingSplitPairs(
         pThreadStateBoosting,
         pFeatureGroup,
         cSamplesRequiredForChildSplitMin,
         pAuxiliaryBucketZone,
         pTotal,
         pTotalGain
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
#endif // NDEBUG
      );
      if(bError) {
#ifndef NDEBUG
         free(aHistogramBucketsDebugCopy);
#endif // NDEBUG

         LOG_0(TraceLevelVerbose, "Exited BoostMultiDimensional with Error code");

         return true;
      }

      // gain can be -infinity for regression in a super-super-super-rare condition.  
      // See notes above regarding "gain = bestSplittingScore - splittingScoreParent"

      // within a set, no split should make our model worse.  It might in our validation set, but not within the training set
      EBM_ASSERT(std::isnan(*pTotalGain) || (!bClassification) && std::isinf(*pTotalGain) ||
         k_epsilonNegativeGainAllowed <= *pTotalGain);
   } else {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional 2 != pFeatureGroup->GetCountSignificantFeatures()");

      // TODO: handle this better
#ifndef NDEBUG
      EBM_ASSERT(false);
      free(aHistogramBucketsDebugCopy);
#endif // NDEBUG
      return true;
   }

#ifndef NDEBUG
   free(aHistogramBucketsDebugCopy);
#endif // NDEBUG

   LOG_0(TraceLevelVerbose, "Exited BoostMultiDimensional");
   return false;
}

static bool BoostRandom(
   ThreadStateBoosting * const pThreadStateBoosting,
   const FeatureGroup * const pFeatureGroup,
   const SamplingSet * const pTrainingSet,
   const GenerateUpdateOptionsType options,
   const IntEbmType * const aLeavesMax,
   FloatEbmType * const pTotalGain
) {
   // THIS RANDOM CUT FUNCTION IS PRIMARILY USED FOR DIFFERENTIAL PRIVACY EBMs

   LOG_0(TraceLevelVerbose, "Entered BoostRandom");

   const size_t cDimensions = pFeatureGroup->GetCountFeatures();
   EBM_ASSERT(1 <= cDimensions);

   size_t cTotalBuckets = 1;
   for(size_t iDimension = 0; iDimension < cDimensions; ++iDimension) {
      const size_t cBins = pFeatureGroup->GetFeatureGroupEntries()[iDimension].m_pFeature->GetCountBins();
      EBM_ASSERT(size_t { 1 } <= cBins); // we don't boost on empty training sets
      // we check for simple multiplication overflow from m_cBins in Booster::Initialize when we unpack featureGroupsFeatureIndexes
      EBM_ASSERT(!IsMultiplyError(cTotalBuckets, cBins));
      cTotalBuckets *= cBins;
   }

   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);
   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
   if(GetHistogramBucketSizeOverflow(bClassification, cVectorLength)) {
      LOG_0(
         TraceLevelWarning,
         "WARNING BoostRandom GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)"
      );
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);
   if(IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)) {
      LOG_0(TraceLevelWarning, "WARNING BoostRandom IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)");
      return true;
   }
   const size_t cBytesBuffer = cTotalBuckets * cBytesPerHistogramBucket;

   // we don't need to free this!  It's tracked and reused by pThreadStateBoosting
   HistogramBucketBase * const aHistogramBuckets = pThreadStateBoosting->GetHistogramBucketBase(cBytesBuffer);
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING BoostRandom nullptr == aHistogramBuckets");
      return true;
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
   pThreadStateBoosting->SetHistogramBucketsEndDebug(reinterpret_cast<unsigned char *>(aHistogramBuckets) + cBytesBuffer);
#endif // NDEBUG

   BinBoosting(
      pThreadStateBoosting,
      pFeatureGroup,
      pTrainingSet
   );

   bool bError = CutRandom(
      pThreadStateBoosting,
      pFeatureGroup,
      options,
      aLeavesMax,
      pTotalGain
   );
   if(bError) {
      LOG_0(TraceLevelVerbose, "Exited BoostRandom with Error code");
      return true;
   }

   // gain can be -infinity for regression in a super-super-super-rare condition.  
   // See notes above regarding "gain = bestSplittingScore - splittingScoreParent"

   // within a set, no split should make our model worse.  It might in our validation set, but not within the training set
   EBM_ASSERT(std::isnan(*pTotalGain) || (!bClassification) && std::isinf(*pTotalGain) ||
      k_epsilonNegativeGainAllowed <= *pTotalGain);

   LOG_0(TraceLevelVerbose, "Exited BoostRandom");
   return false;
}

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
static IntEbmType GenerateModelUpdateInternal(
   ThreadStateBoosting * const pThreadStateBoosting,
   const size_t iFeatureGroup,
   const GenerateUpdateOptionsType options,
   const FloatEbmType learningRate,
   const size_t cSamplesRequiredForChildSplitMin,
   const IntEbmType * const aLeavesMax, 
   FloatEbmType * const pGainReturn
) {
   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelVerbose, "Entered GenerateModelUpdateInternal");

   const size_t cSamplingSetsAfterZero = (0 == pBooster->GetCountSamplingSets()) ? 1 : pBooster->GetCountSamplingSets();
   const FeatureGroup * const pFeatureGroup = pBooster->GetFeatureGroups()[iFeatureGroup];
   const size_t cSignificantDimensions = pFeatureGroup->GetCountSignificantFeatures();

   IntEbmType lastDimensionLeavesMax = IntEbmType { 0 };
   size_t cSignificantBinCount;
   if(nullptr == aLeavesMax) {
      LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdateInternal aLeavesMax was null, so there won't be any splits");
   } else {
      if(0 != cSignificantDimensions) {
         const IntEbmType * pLeavesMax = aLeavesMax;
         const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
         EBM_ASSERT(1 <= pFeatureGroup->GetCountFeatures());
         const FeatureGroupEntry * const pFeatureGroupEntryEnd = pFeatureGroupEntry + pFeatureGroup->GetCountFeatures();
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

   pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->SetCountDimensions(cSignificantDimensions);
   pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->Reset();

   // if pBooster->m_apSamplingSets is nullptr, then we should have zero training samples
   // we can't be partially constructed here since then we wouldn't have returned our state pointer to our caller

   FloatEbmType totalGain = FloatEbmType { 0 };
   if(nullptr != pBooster->GetSamplingSets()) {
      pThreadStateBoosting->GetSmallChangeToModelOverwriteSingleSamplingSet()->SetCountDimensions(cSignificantDimensions);

      for(size_t iSamplingSet = 0; iSamplingSet < cSamplingSetsAfterZero; ++iSamplingSet) {
         FloatEbmType gain;
         if(UNLIKELY(IntEbmType { 0 } == lastDimensionLeavesMax)) {
            LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdateInternal boosting zero dimensional");
            if(BoostZeroDimensional(pThreadStateBoosting, pBooster->GetSamplingSets()[iSamplingSet], options)) {
               if(LIKELY(nullptr != pGainReturn)) {
                  *pGainReturn = FloatEbmType { 0 };
               }
               return IntEbmType { 1 };
            }
            gain = FloatEbmType { 0 };
         } else if(0 != (GenerateUpdateOptions_RandomSplits & options)) {
            if(size_t { 1 } != cSamplesRequiredForChildSplitMin) {
               LOG_0(TraceLevelWarning, 
                  "WARNING GenerateModelUpdateInternal cSamplesRequiredForChildSplitMin is ignored when doing random splitting"
               );
            }
            // THIS RANDOM CUT OPTION IS PRIMARILY USED FOR DIFFERENTIAL PRIVACY EBMs
            if(BoostRandom(
               pThreadStateBoosting,
               pFeatureGroup,
               pBooster->GetSamplingSets()[iSamplingSet],
               options,
               aLeavesMax, 
               &gain
            )) {
               if(LIKELY(nullptr != pGainReturn)) {
                  *pGainReturn = FloatEbmType { 0 };
               }
               return IntEbmType { 1 };
            }
         } else if(1 == cSignificantDimensions) {
            EBM_ASSERT(nullptr != aLeavesMax); // otherwise we'd use BoostZeroDimensional above
            EBM_ASSERT(size_t { 2 } <= lastDimensionLeavesMax); // otherwise we'd use BoostZeroDimensional above
            EBM_ASSERT(size_t { 2 } <= cSignificantBinCount); // otherwise we'd use BoostZeroDimensional above
            if(BoostSingleDimensional(
               pThreadStateBoosting,
               pFeatureGroup,
               cSignificantBinCount,
               pBooster->GetSamplingSets()[iSamplingSet],
               cSamplesRequiredForChildSplitMin,
               lastDimensionLeavesMax,
               &gain
            )) {
               if(LIKELY(nullptr != pGainReturn)) {
                  *pGainReturn = FloatEbmType { 0 };
               }
               return IntEbmType { 1 };
            }
         } else {
            if(BoostMultiDimensional(
               pThreadStateBoosting,
               pFeatureGroup,
               pBooster->GetSamplingSets()[iSamplingSet],
               cSamplesRequiredForChildSplitMin,
               &gain
            )) {
               if(LIKELY(nullptr != pGainReturn)) {
                  *pGainReturn = FloatEbmType { 0 };
               }
               return IntEbmType { 1 };
            }
         }
         // regression can be -infinity or slightly negative in extremely rare circumstances.  
         // See ExamineNodeForPossibleFutureSplittingAndDetermineBestSplitPoint for details, and the equivalent interaction function
         EBM_ASSERT(std::isnan(gain) || (!bClassification) && std::isinf(gain) || k_epsilonNegativeGainAllowed <= gain); // we previously normalized to 0
         totalGain += gain;
         // TODO : when we thread this code, let's have each thread take a lock and update the combined line segment.  They'll each do it while the 
         // others are working, so there should be no blocking and our final result won't require adding by the main thread
         if(pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->Add(*pThreadStateBoosting->GetSmallChangeToModelOverwriteSingleSamplingSet())) {
            if(LIKELY(nullptr != pGainReturn)) {
               *pGainReturn = FloatEbmType { 0 };
            }
            return IntEbmType { 1 };
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

         //if(0 <= k_iZeroResidual || ptrdiff_t { 2 } == pBooster->m_runtimeLearningTypeOrCountTargetClasses && bExpandBinaryLogits) {
         //   EBM_ASSERT(ptrdiff_t { 2 } <= pBooster->m_runtimeLearningTypeOrCountTargetClasses);
         //   // TODO : for classification with residual zeroing, is our learning rate essentially being inflated as 
         //       pBooster->m_runtimeLearningTypeOrCountTargetClasses goes up?  If so, maybe we should divide by 
         //       pBooster->m_runtimeLearningTypeOrCountTargetClasses here to keep learning rates as equivalent as possible..  
         //       Actually, I think the real solution here is that 
         //   pBooster->m_pSmallChangeToModelAccumulatedFromSamplingSets->Multiply(
         //      learningRate / cSamplingSetsAfterZero * (pBooster->m_runtimeLearningTypeOrCountTargetClasses - 1) / 
         //      pBooster->m_runtimeLearningTypeOrCountTargetClasses
         //   );
         //} else {
         //   // TODO : for classification, is our learning rate essentially being inflated as 
         //        pBooster->m_runtimeLearningTypeOrCountTargetClasses goes up?  If so, maybe we should divide by 
         //        pBooster->m_runtimeLearningTypeOrCountTargetClasses here to keep learning rates equivalent as possible
         //   pBooster->m_pSmallChangeToModelAccumulatedFromSamplingSets->Multiply(learningRate / cSamplingSetsAfterZero);
         //}

         // TODO: When NewtonBoosting is enabled, we need to multiply our rate by (K - 1)/K (see above), per:
         // https://arxiv.org/pdf/1810.09092v2.pdf (forumla 5) and also the 
         // Ping Li paper (algorithm #1, line 5, (K - 1) / K )
         // https://arxiv.org/pdf/1006.5051.pdf

         const bool bDividing = bExpandBinaryLogits && ptrdiff_t { 2 } == runtimeLearningTypeOrCountTargetClasses;
         if(bDividing) {
            bBad = pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->MultiplyAndCheckForIssues(learningRate / cSamplingSetsAfterZero / 2);
         } else {
            bBad = pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->MultiplyAndCheckForIssues(learningRate / cSamplingSetsAfterZero);
         }
      } else {
         bBad = pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->MultiplyAndCheckForIssues(learningRate / cSamplingSetsAfterZero);
      }

      // handle the case where totalGain is either +infinity or -infinity (very rare, see above), or NaN
      // don't use std::inf because with some compiler flags on some compilers that isn't reliable
      if(UNLIKELY(
         UNLIKELY(bBad) || 
         UNLIKELY(std::isnan(totalGain)) || 
         UNLIKELY(totalGain <= std::numeric_limits<FloatEbmType>::lowest()) ||
         UNLIKELY(std::numeric_limits<FloatEbmType>::max() <= totalGain)
      )) {
         pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->SetCountDimensions(cSignificantDimensions);
         pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->Reset();
         // declare there is no gain, so that our caller will think there is no benefit in splitting us, which there isn't since we're zeroed.
         totalGain = FloatEbmType { 0 };
      } else if(UNLIKELY(totalGain < FloatEbmType { 0 })) {
         totalGain = FloatEbmType { 0 };
      }
   }

   pThreadStateBoosting->SetFeatureGroupIndex(iFeatureGroup);

   if(nullptr != pGainReturn) {
      *pGainReturn = totalGain;
   }

   LOG_0(TraceLevelVerbose, "Exited GenerateModelUpdatePerTargetClasses");
   return IntEbmType { 0 };
}

WARNING_POP

// we made this a global because if we had put this variable inside the Booster object, then we would need to dereference that before getting 
// the count.  By making this global we can send a log message incase a bad Booster object is sent into us we only decrease the count if the 
// count is non-zero, so at worst if there is a race condition then we'll output this log message more times than desired, but we can live with that
static int g_cLogGenerateModelUpdateParametersMessages = 10;

// TODO : change this so that our caller allocates the memory that contains the update, but this is complicated in various ways
//        we don't want to just copy the internal tensor into the memory region that our caller provides, and we want to work with
//        compressed representations of the SegmentedTensor object while we're building it, so we'll work within the memory the caller
//        provides, but that means we'll potentially need more memory than the full tensor, and we'll need to put some header info
//        at the start, so the caller can't treat this memory as a pure tensor.
//        So:
//          1) provide a function that returns the maximum memory needed.  A smart caller will call this once on each feature_group, 
//             choose the max and allocate it once
//          2) return a compressed complete SegmentedTensor to the caller inside an opaque memory region 
//             (return the exact size that we require to the caller for copying)
//          3) if caller wants a simplified tensor, then they call a separate function that expands the tensor 
//             and returns a pointer to the memory inside the opaque object
//          4) ApplyModelUpdate will take an opaque SegmentedTensor, and expand it if needed
//        The benefit of returning a compressed object is that we don't have to do the work of expanding it if the caller decides not to use it 
//        (which might happen in greedy algorithms)
//        The other benefit of returning a compressed object is that our caller can store/copy it faster
//        The other benefit of returning a compressed object is that it can be copied from process to process faster
//        Lastly, with the memory allocated by our caller, we can call GenerateModelUpdate in parallel on multiple feature_groups.  
//        Right now you can't call it in parallel since we're updating our internal single tensor

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateModelUpdate(
   ThreadStateBoostingHandle threadStateBoostingHandle,
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
      "threadStateBoostingHandle=%p, "
      "indexFeatureGroup=%" IntEbmTypePrintf ", "
      "options=0x%" UGenerateUpdateOptionsTypePrintf ", "
      "learningRate=%" FloatEbmTypePrintf ", "
      "countSamplesRequiredForChildSplitMin=%" IntEbmTypePrintf ", "
      "leavesMax=%p, "
      "gainOut=%p"
      ,
      static_cast<void *>(threadStateBoostingHandle),
      indexFeatureGroup,
      static_cast<UGenerateUpdateOptionsType>(options), // signed to unsigned conversion is defined behavior in C++
      learningRate,
      countSamplesRequiredForChildSplitMin,
      static_cast<const void *>(leavesMax),
      static_cast<void *>(gainOut)
   );

   ThreadStateBoosting * const pThreadStateBoosting = reinterpret_cast<ThreadStateBoosting *>(threadStateBoostingHandle);
   if(nullptr == pThreadStateBoosting) {
      if(LIKELY(nullptr != gainOut)) {
         *gainOut = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate threadStateBoosting cannot be nullptr");
      return IntEbmType { 1 };
   }

   // set this to illegal so if we exit with an error we have an invalid index
   pThreadStateBoosting->SetFeatureGroupIndex(ThreadStateBoosting::k_illegalFeatureGroupIndex);

   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   EBM_ASSERT(nullptr != pBooster);

   if(indexFeatureGroup < 0) {
      if(LIKELY(nullptr != gainOut)) {
         *gainOut = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate indexFeatureGroup must be positive");
      return IntEbmType { 1 };
   }
   if(!IsNumberConvertable<size_t>(indexFeatureGroup)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      if(LIKELY(nullptr != gainOut)) {
         *gainOut = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate indexFeatureGroup is too high to index");
      return IntEbmType { 1 };
   }
   size_t iFeatureGroup = static_cast<size_t>(indexFeatureGroup);
   if(pBooster->GetCountFeatureGroups() <= iFeatureGroup) {
      if(LIKELY(nullptr != gainOut)) {
         *gainOut = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate indexFeatureGroup above the number of feature groups that we have");
      return IntEbmType { 1 };
   }
   // this is true because 0 < pBooster->m_cFeatureGroups since our caller needs to pass in a valid indexFeatureGroup to this function
   EBM_ASSERT(nullptr != pBooster->GetFeatureGroups());

   LOG_COUNTED_0(
      pBooster->GetFeatureGroups()[iFeatureGroup]->GetPointerCountLogEnterGenerateModelUpdateMessages(),
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
      if(!IsNumberConvertable<size_t>(countSamplesRequiredForChildSplitMin)) {
         // we can never exceed a size_t number of samples, so let's just set it to the maximum if we were going to overflow because it will generate 
         // the same results as if we used the true number
         cSamplesRequiredForChildSplitMin = std::numeric_limits<size_t>::max();
      }
   } else {
      LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdate countSamplesRequiredForChildSplitMin can't be less than 1.  Adjusting to 1.");
   }

   // leavesMax is handled in GenerateModelUpdateInternal

   // gainOut can be nullptr

   if(ptrdiff_t { 0 } == pBooster->GetRuntimeLearningTypeOrCountTargetClasses() || ptrdiff_t { 1 } == pBooster->GetRuntimeLearningTypeOrCountTargetClasses()) {
      // if there is only 1 target class for classification, then we can predict the output with 100% accuracy.  The model is a tensor with zero 
      // length array logits, which means for our representation that we have zero items in the array total.
      // since we can predit the output with 100% accuracy, our gain will be 0.
      if(LIKELY(nullptr != gainOut)) {
         *gainOut = FloatEbmType { 0 };
      }
      pThreadStateBoosting->SetFeatureGroupIndex(iFeatureGroup);

      LOG_0(
         TraceLevelWarning,
         "WARNING GenerateModelUpdate pBooster->m_runtimeLearningTypeOrCountTargetClasses <= ptrdiff_t { 1 }"
      );
      return IntEbmType { 0 };
   }

   const IntEbmType ret = GenerateModelUpdateInternal(
      pThreadStateBoosting,
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
         pBooster->GetFeatureGroups()[iFeatureGroup]->GetPointerCountLogExitGenerateModelUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited GenerateModelUpdate %" FloatEbmTypePrintf,
         *gainOut
      );
   } else {
      LOG_COUNTED_0(
         pBooster->GetFeatureGroups()[iFeatureGroup]->GetPointerCountLogExitGenerateModelUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited GenerateModelUpdate no gain"
      );
   }
   return ret;
}

