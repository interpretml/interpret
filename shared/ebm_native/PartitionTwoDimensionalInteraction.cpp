// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "ebm_stats.hpp"

#include "Feature.hpp"
#include "FeatureGroup.hpp"

#include "HistogramTargetEntry.hpp"
#include "HistogramBucket.hpp"

#include "InteractionCore.hpp"

#include "TensorTotalsSum.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class PartitionTwoDimensionalInteractionInternal final {
public:

   PartitionTwoDimensionalInteractionInternal() = delete; // this is a static class.  Do not construct

   static FloatEbmType Func(
      InteractionCore * const pInteractionCore,
      const FeatureGroup * const pFeatureGroup,
      const InteractionOptionsType options,
      const size_t cSamplesRequiredForChildSplitMin,
      HistogramBucketBase * pAuxiliaryBucketZoneBase,
      HistogramBucketBase * const aHistogramBucketsBase
#ifndef NDEBUG
      , const HistogramBucketBase * const aHistogramBucketsDebugCopyBase
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pAuxiliaryBucketZone =
         pAuxiliaryBucketZoneBase->GetHistogramBucket<bClassification>();

      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets =
         aHistogramBucketsBase->GetHistogramBucket<bClassification>();

#ifndef NDEBUG
      const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucketsDebugCopy =
         aHistogramBucketsDebugCopyBase->GetHistogramBucket<bClassification>();
#endif // NDEBUG

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses()
      );

      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);

      HistogramBucket<bClassification> * pTotalsLowLow =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 0);
      HistogramBucket<bClassification> * pTotalsLowHigh =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 1);
      HistogramBucket<bClassification> * pTotalsHighLow =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 2);
      HistogramBucket<bClassification> * pTotalsHighHigh =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pAuxiliaryBucketZone, 3);

      // for interactions we return an interaction score of 0 if any of the dimensions are useless
      EBM_ASSERT(2 == pFeatureGroup->GetCountDimensions());
      EBM_ASSERT(2 == pFeatureGroup->GetCountSignificantDimensions());

      // we return an interaction score of 0 if any features are useless before calling here
      EBM_ASSERT(pFeatureGroup->GetCountDimensions() == pFeatureGroup->GetCountSignificantDimensions());
      const size_t cBinsDimension1 = pFeatureGroup->GetFeatureGroupEntries()[0].m_pFeature->GetCountBins();
      const size_t cBinsDimension2 = pFeatureGroup->GetFeatureGroupEntries()[1].m_pFeature->GetCountBins();

      // any pair with a feature with 1 cBins returns an interaction score of 0
      EBM_ASSERT(2 <= cBinsDimension1);
      EBM_ASSERT(2 <= cBinsDimension2);

      EBM_ASSERT(0 < cSamplesRequiredForChildSplitMin);

      // if a negative value were to occur, then it would be due to numeric instability, so clip it to zero here
      FloatEbmType bestSplittingScore = FloatEbmType { 0 };

      size_t aiStart[k_cDimensionsMax];

      EBM_ASSERT(1 < cBinsDimension1);
      size_t iBin1 = 0;
      do {
         aiStart[0] = iBin1;
         EBM_ASSERT(1 < cBinsDimension2);
         size_t iBin2 = 0;
         do {
            aiStart[1] = iBin2;

            EBM_ASSERT(2 == pFeatureGroup->GetCountSignificantDimensions()); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
            TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, 2>(
               learningTypeOrCountTargetClasses,
               pFeatureGroup,
               aHistogramBuckets,
               aiStart,
               0x00,
               pTotalsLowLow
#ifndef NDEBUG
               , aHistogramBucketsDebugCopy
               , aHistogramBucketsEndDebug
#endif // NDEBUG
               );
            if(LIKELY(cSamplesRequiredForChildSplitMin <= pTotalsLowLow->GetCountSamplesInBucket())) {
               EBM_ASSERT(2 == pFeatureGroup->GetCountSignificantDimensions()); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
               TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, 2>(
                  learningTypeOrCountTargetClasses,
                  pFeatureGroup,
                  aHistogramBuckets,
                  aiStart,
                  0x02,
                  pTotalsLowHigh
#ifndef NDEBUG
                  , aHistogramBucketsDebugCopy
                  , aHistogramBucketsEndDebug
#endif // NDEBUG
                  );
               if(LIKELY(cSamplesRequiredForChildSplitMin <= pTotalsLowHigh->GetCountSamplesInBucket())) {
                  EBM_ASSERT(2 == pFeatureGroup->GetCountSignificantDimensions()); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
                  TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, 2>(
                     learningTypeOrCountTargetClasses,
                     pFeatureGroup,
                     aHistogramBuckets,
                     aiStart,
                     0x01,
                     pTotalsHighLow
#ifndef NDEBUG
                     , aHistogramBucketsDebugCopy
                     , aHistogramBucketsEndDebug
#endif // NDEBUG
                     );
                  if(LIKELY(cSamplesRequiredForChildSplitMin <= pTotalsHighLow->GetCountSamplesInBucket())) {
                     EBM_ASSERT(2 == pFeatureGroup->GetCountSignificantDimensions()); // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
                     TensorTotalsSum<compilerLearningTypeOrCountTargetClasses, 2>(
                        learningTypeOrCountTargetClasses,
                        pFeatureGroup,
                        aHistogramBuckets,
                        aiStart,
                        0x03,
                        pTotalsHighHigh
#ifndef NDEBUG
                        , aHistogramBucketsDebugCopy
                        , aHistogramBucketsEndDebug
#endif // NDEBUG
                        );
                     if(LIKELY(cSamplesRequiredForChildSplitMin <= pTotalsHighHigh->GetCountSamplesInBucket())) {
                        FloatEbmType splittingScore = 0;

                        FloatEbmType cLowLowWeightInBucket = pTotalsLowLow->GetWeightInBucket();
                        FloatEbmType cLowHighWeightInBucket = pTotalsLowHigh->GetWeightInBucket();
                        FloatEbmType cHighLowWeightInBucket = pTotalsHighLow->GetWeightInBucket();
                        FloatEbmType cHighHighWeightInBucket = pTotalsHighHigh->GetWeightInBucket();

                        HistogramTargetEntry<bClassification> * const pHistogramTargetEntryTotalsLowLow =
                           pTotalsLowLow->GetHistogramTargetEntry();
                        HistogramTargetEntry<bClassification> * const pHistogramTargetEntryTotalsLowHigh =
                           pTotalsLowHigh->GetHistogramTargetEntry();
                        HistogramTargetEntry<bClassification> * const pHistogramTargetEntryTotalsHighLow =
                           pTotalsHighLow->GetHistogramTargetEntry();
                        HistogramTargetEntry<bClassification> * const pHistogramTargetEntryTotalsHighHigh =
                           pTotalsHighHigh->GetHistogramTargetEntry();

                        for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
                           // TODO : we can make this faster by doing the division in ComputeSinglePartitionGain after we add all the numerators 
                           // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

                           constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;

                           // use l = low, h = high, n = numerator, d = denominator

                           const FloatEbmType lln = pHistogramTargetEntryTotalsLowLow[iVector].m_sumGradients;
                           const FloatEbmType lld = bUseLogitBoost ?
                              pHistogramTargetEntryTotalsLowLow[iVector].GetSumHessians() : cLowLowWeightInBucket;

                           const FloatEbmType lhn = pHistogramTargetEntryTotalsLowHigh[iVector].m_sumGradients;
                           const FloatEbmType lhd = bUseLogitBoost ?
                              pHistogramTargetEntryTotalsLowHigh[iVector].GetSumHessians() : cLowHighWeightInBucket;

                           const FloatEbmType hln = pHistogramTargetEntryTotalsHighLow[iVector].m_sumGradients;
                           const FloatEbmType hld = bUseLogitBoost ?
                              pHistogramTargetEntryTotalsHighLow[iVector].GetSumHessians() : cHighLowWeightInBucket;

                           const FloatEbmType hhn = pHistogramTargetEntryTotalsHighHigh[iVector].m_sumGradients;
                           const FloatEbmType hhd = bUseLogitBoost ?
                              pHistogramTargetEntryTotalsHighHigh[iVector].GetSumHessians() : cHighHighWeightInBucket;

                           if(0 != (InteractionOptions_Pure & options)) {
                              // purified gain

                              // if any of the denominators (weights) are zero then the purified gain tends 
                              // towards zero.  Handle it here to avoid division by zero
                              if(FloatEbmType { 0 } != lld && FloatEbmType { 0 } != lhd && 
                                 FloatEbmType { 0 } != hld && FloatEbmType { 0 } != hhd) {

                                 // TODO: instead of checking the denominators for zero above, can we do it earlier?
                                 // If we're using hessians then we'd need it here, but we aren't using them yet

                                 // calculate what the full updates would be for non-purified:
                                 // u = update (non-purified)
                                 const FloatEbmType llu = lln / lld;
                                 const FloatEbmType lhu = lhn / lhd;
                                 const FloatEbmType hlu = hln / hld;
                                 const FloatEbmType hhu = hhn / hhd;

                                 // purified numerator (positive for ll & hh equations, negative for lh and hl)
                                 const FloatEbmType n = llu - lhu - hlu + hhu;

                                 // p = purified update
                                 const FloatEbmType llp = n / (FloatEbmType { 1 } + lld / lhd + lld / hld + lld / hhd);
                                 const FloatEbmType lhp = n / (FloatEbmType { -1 } - lhd / lld - lhd / hld - lhd / hhd);
                                 const FloatEbmType hlp = n / (FloatEbmType { -1 } - hld / lld - hld / lhd - hld / hhd);
                                 const FloatEbmType hhp = n / (FloatEbmType { 1 } + hhd / lld + hhd / lhd + hhd / hld);

#ifndef NDEBUG
                                 // r = reconsituted gradient sum (after purification)
                                 const FloatEbmType llr = llp * lld;
                                 const FloatEbmType lhr = lhp * lhd;
                                 const FloatEbmType hlr = hlp * hld;
                                 const FloatEbmType hhr = hhp * hhd;

                                 // purification means summing any direction gives us zero
                                 EBM_ASSERT(std::abs(llr + lhr) < 0.001);
                                 EBM_ASSERT(std::abs(lhr + hhr) < 0.001);
                                 EBM_ASSERT(std::abs(hhr + hlr) < 0.001);
                                 EBM_ASSERT(std::abs(hlr + llr) < 0.001);

                                 // if all of these are zero, then the sum of all of them should be zero,
                                 // which means our RSS of the combined bin should be zero, which means we do not
                                 // need to subtract the parent RSS to get our gain.
                                 EBM_ASSERT(std::abs(llr + lhr + hlr + hhr) < 0.001);
#endif // NDEBUG

                                 // this is another way to calculate our variance gain
                                 // g = gain (normally gain would require subtacting from the parent RSS, but that's zero)
                                 const FloatEbmType llg = llp * llp * lld;
                                 const FloatEbmType lhg = lhp * lhp * lhd;
                                 const FloatEbmType hlg = hlp * hlp * hld;
                                 const FloatEbmType hhg = hhp * hhp * hhd;

                                 EBM_ASSERT(std::isnan(llg) || FloatEbmType { 0 } <= llg);
                                 EBM_ASSERT(std::isnan(lhg) || FloatEbmType { 0 } <= lhg);
                                 EBM_ASSERT(std::isnan(hlg) || FloatEbmType { 0 } <= hlg);
                                 EBM_ASSERT(std::isnan(hhg) || FloatEbmType { 0 } <= hhg);

                                 splittingScore += llg;
                                 splittingScore += lhg;
                                 splittingScore += hlg;
                                 splittingScore += hhg;
                              }
                           } else {
                              // non-purified gain

                              const FloatEbmType splittingScoreUpdate1 = EbmStats::ComputeSinglePartitionGain(lln, lld);
                              EBM_ASSERT(std::isnan(splittingScoreUpdate1) || FloatEbmType { 0 } <= splittingScoreUpdate1);
                              splittingScore += splittingScoreUpdate1;

                              const FloatEbmType splittingScoreUpdate2 = EbmStats::ComputeSinglePartitionGain(lhn, lhd);
                              EBM_ASSERT(std::isnan(splittingScoreUpdate2) || FloatEbmType { 0 } <= splittingScoreUpdate2);
                              splittingScore += splittingScoreUpdate2;

                              const FloatEbmType splittingScoreUpdate3 = EbmStats::ComputeSinglePartitionGain(hln, hld);
                              EBM_ASSERT(std::isnan(splittingScoreUpdate3) || FloatEbmType { 0 } <= splittingScoreUpdate3);
                              splittingScore += splittingScoreUpdate3;

                              const FloatEbmType splittingScoreUpdate4 = EbmStats::ComputeSinglePartitionGain(hhn, hhd);
                              EBM_ASSERT(std::isnan(splittingScoreUpdate4) || FloatEbmType { 0 } <= splittingScoreUpdate4);
                              splittingScore += splittingScoreUpdate4;
                           }
                        }
                        EBM_ASSERT(std::isnan(splittingScore) || FloatEbmType { 0 } <= splittingScore); // sumations of positive numbers should be positive

                        // If we get a NaN result, we'd like to propagate it by making bestSplittingScore NaN.  
                        // The rules for NaN values say that non equality comparisons are all false so, 
                        // let's flip this comparison such that it should be true for NaN values.
                        if(UNLIKELY(! /* NaN */ LIKELY(splittingScore <= bestSplittingScore))) {
                           bestSplittingScore = splittingScore;
                        } else {
                           EBM_ASSERT(!std::isnan(splittingScore));
                        }
                     }
                  }
               }
            }
            ++iBin2;
         } while(iBin2 < cBinsDimension2 - 1);
         ++iBin1;
      } while(iBin1 < cBinsDimension1 - 1);

      // we start from zero, so bestSplittingScore can't be negative here
      EBM_ASSERT(std::isnan(bestSplittingScore) || FloatEbmType { 0 } <= bestSplittingScore);

      if(0 == (InteractionOptions_Pure & options)) {
         // if we are detecting impure interaction then so far we've only calculated the first part of the RSS 
         // calculation but we still need to subtract the RSS of the non-split case and subtract it to have
         // gain. All the splits we've analyzed so far though had the same non-split RSS, so we subtract it here
         // instead of inside the loop.

#ifndef NDEBUG
         // if no legal splits were found, then bestSplittingScore will be zero.  In theory we should
         // therefore not subtract the base RSS, but doing so does no harm since we later set any
         // negative interaction score to zero in the caller of this function.  Due to that we don't
         // need to check here, since any value we subtract from zero will lead to a negative number and
         // then will be zeroed by our caller
         // BUT, for debugging purposes, check here for that condition so that we can check for illegal negative gain.
         bool isOriginallyZero = FloatEbmType { 0 } == bestSplittingScore;
#endif // NDEBUG

         // the bucket before the pAuxiliaryBucketZoneBase is the last summation bucket of aHistogramBucketsBase, 
         // which contains the totals of all buckets
         const HistogramBucket<bClassification> * const pTotal =
            reinterpret_cast<const HistogramBucket<bClassification> *>(
               reinterpret_cast<const char *>(pAuxiliaryBucketZoneBase) - cBytesPerHistogramBucket);

         const FloatEbmType cTotalWeightInBucket = pTotal->GetWeightInBucket();

         const HistogramTargetEntry<bClassification> * const pHistogramTargetEntryTotal =
            pTotal->GetHistogramTargetEntry();

         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            // TODO : we can make this faster by doing the division in ComputeSinglePartitionGain after we add all the numerators 
            // (but only do this after we've determined the best node splitting score for classification, and the NewtonRaphsonStep for gain

            constexpr bool bUseLogitBoost = k_bUseLogitboost && bClassification;
            const FloatEbmType splittingScoreUpdate = EbmStats::ComputeSinglePartitionGain(
               pHistogramTargetEntryTotal[iVector].m_sumGradients,
               bUseLogitBoost ? pHistogramTargetEntryTotal[iVector].GetSumHessians() : cTotalWeightInBucket
            );
            EBM_ASSERT(std::isnan(splittingScoreUpdate) || FloatEbmType { 0 } <= splittingScoreUpdate);
            bestSplittingScore -= splittingScoreUpdate;
         }

         // gain should be positive, or NaN, BUT it can be slightly negative due to floating point noise
         // it could also be -inf if the total bucket overflows, but the parts did not.  In that case we've
         // reached -inf due to numeric instability, but we should eventually return zero in this case.
         // We fix this up though in our caller.
         // bestSplittingScore can also be substantially negative if we didn't find any legal cuts and 
         // then we subtracted the base RSS here from zero
         EBM_ASSERT(std::isnan(bestSplittingScore) ||
            isOriginallyZero ||
            -std::numeric_limits<FloatEbmType>::infinity() == bestSplittingScore ||
            k_epsilonNegativeGainAllowed <= bestSplittingScore);
      }

      // we clean up bestSplittingScore in the caller, since this function is templated and created many times

      const DataSetInteraction * const pDataSet = pInteractionCore->GetDataSetInteraction();
      EBM_ASSERT(nullptr != pDataSet);
      EBM_ASSERT(FloatEbmType { 0 } < pDataSet->GetWeightTotal()); // if all are zeros we assume there are no weights and use the count
      return bestSplittingScore / pDataSet->GetWeightTotal();
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class PartitionTwoDimensionalInteractionTarget final {
public:

   PartitionTwoDimensionalInteractionTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static FloatEbmType Func(
      InteractionCore * const pInteractionCore,
      const FeatureGroup * const pFeatureGroup,
      const InteractionOptionsType options,
      const size_t cSamplesRequiredForChildSplitMin,
      HistogramBucketBase * pAuxiliaryBucketZone,
      HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
      , const HistogramBucketBase * const aHistogramBucketsDebugCopy
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         return PartitionTwoDimensionalInteractionInternal<compilerLearningTypeOrCountTargetClassesPossible>::Func(
            pInteractionCore,
            pFeatureGroup,
            options,
            cSamplesRequiredForChildSplitMin,
            pAuxiliaryBucketZone,
            aHistogramBuckets
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      } else {
         return PartitionTwoDimensionalInteractionTarget<compilerLearningTypeOrCountTargetClassesPossible + 1>::Func(
            pInteractionCore,
            pFeatureGroup,
            options,
            cSamplesRequiredForChildSplitMin,
            pAuxiliaryBucketZone,
            aHistogramBuckets
#ifndef NDEBUG
            , aHistogramBucketsDebugCopy
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<>
class PartitionTwoDimensionalInteractionTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   PartitionTwoDimensionalInteractionTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static FloatEbmType Func(
      InteractionCore * const pInteractionCore,
      const FeatureGroup * const pFeatureGroup,
      const InteractionOptionsType options,
      const size_t cSamplesRequiredForChildSplitMin,
      HistogramBucketBase * pAuxiliaryBucketZone,
      HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
      , const HistogramBucketBase * const aHistogramBucketsDebugCopy
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses());

      return PartitionTwoDimensionalInteractionInternal<k_dynamicClassification>::Func(
         pInteractionCore,
         pFeatureGroup,
         options,
         cSamplesRequiredForChildSplitMin,
         pAuxiliaryBucketZone,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
};

extern FloatEbmType PartitionTwoDimensionalInteraction(
   InteractionCore * const pInteractionCore,
   const FeatureGroup * const pFeatureGroup,
   const InteractionOptionsType options,
   const size_t cSamplesRequiredForChildSplitMin,
   HistogramBucketBase * pAuxiliaryBucketZone,
   HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
   , const HistogramBucketBase * const aHistogramBucketsDebugCopy
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses();

   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      return PartitionTwoDimensionalInteractionTarget<2>::Func(
         pInteractionCore,
         pFeatureGroup,
         options,
         cSamplesRequiredForChildSplitMin,
         pAuxiliaryBucketZone,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      return PartitionTwoDimensionalInteractionInternal<k_regression>::Func(
         pInteractionCore,
         pFeatureGroup,
         options,
         cSamplesRequiredForChildSplitMin,
         pAuxiliaryBucketZone,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
}

} // DEFINED_ZONE_NAME
