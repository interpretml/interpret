// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "EbmInternal.h"

#include "FeatureAtomic.h"
#include "FeatureGroup.h"

#include "HistogramTargetEntry.h"
#include "HistogramBucket.h"

#include "Booster.h"
#include "ThreadStateBoosting.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class SumHistogramBucketsInternal final {
public:

   SumHistogramBucketsInternal() = delete; // this is a static class.  Do not construct

   static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const size_t cHistogramBuckets
#ifndef NDEBUG
      , const size_t cSamplesTotal
#endif // NDEBUG
   ) {
      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();

      HistogramTargetEntryBase * const aSumHistogramTargetEntryBase =
         pThreadStateBoosting->GetSumHistogramTargetEntryArray();

      HistogramBucketBase * const aHistogramBucketsBase = pThreadStateBoosting->GetHistogramBucketBase();

      const HistogramBucket<bClassification> * const aHistogramBuckets = 
         aHistogramBucketsBase->GetHistogramBucket<bClassification>();
      HistogramTargetEntry<bClassification> * const aSumHistogramTargetEntry = 
         aSumHistogramTargetEntryBase->GetHistogramTargetEntry<bClassification>();

      EBM_ASSERT(2 <= cHistogramBuckets); // we pre-filter out features with only one bucket

#ifndef NDEBUG
      size_t cSamplesTotalDebug = 0;
#endif // NDEBUG

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      EBM_ASSERT(!GetHistogramBucketSizeOverflow(bClassification, cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);

      const HistogramBucket<bClassification> * pCopyFrom = aHistogramBuckets;
      const HistogramBucket<bClassification> * pCopyFromEnd =
         GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, aHistogramBuckets, cHistogramBuckets);

      // we do a lot more work in the GrowDecisionTree function per binned bucket entry, so if we can compress it by any amount, then it will probably be a win
      // for binned bucket arrays that have a small set of labels, this loop will be fast and result in no movements.  For binned bucket arrays that are long 
      // and have many different labels, we are more likley to find bins with zero items, and that's where we get a win by compressing it down to just the 
      // non-zero binned buckets, even though this requires one more member variable in the binned bucket array
      do {
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pCopyFrom, pThreadStateBoosting->GetHistogramBucketsEndDebug());
#ifndef NDEBUG
         cSamplesTotalDebug += pCopyFrom->GetCountSamplesInBucket();
#endif // NDEBUG

         const HistogramTargetEntry<bClassification> * pHistogramTargetEntry = 
            pCopyFrom->GetHistogramTargetEntry();

         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            // when building a tree, we start from one end and sweep to the other.  In order to caluculate
            // gain on both sides, we need the sum on both sides, which means when starting from one end
            // we need to know the sum of everything on the other side, so we need to calculate this sum
            // somewhere.  If we have a continuous value and bin it such that many samples are in the same bin
            // then it makes sense to calculate the total of all bins after generating the histograms of the bins
            // since then we just need to sum N bins (where N is the number of bins) vs the # of samples.
            // There is one case though where we might want to calculate the sum while looping the samples,
            // and that is if almost all bins have either 0 or 1 samples, which would happen if we didn't bin at all
            // beforehand.  We'll still want this per-bin sumation though since it's unlikley that all data
            // will be continuous in an ML problem.
            aSumHistogramTargetEntry[iVector].Add(pHistogramTargetEntry[iVector]);
         }

         pCopyFrom = GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pCopyFrom, 1);
      } while(pCopyFromEnd != pCopyFrom);
      EBM_ASSERT(0 == (reinterpret_cast<const char *>(pCopyFrom) - reinterpret_cast<const char *>(aHistogramBuckets)) % cBytesPerHistogramBucket);

      EBM_ASSERT(cSamplesTotal == cSamplesTotalDebug);
   }
};

extern void SumHistogramBuckets(
   ThreadStateBoosting * const pThreadStateBoosting,
   const size_t cHistogramBuckets
#ifndef NDEBUG
   , const size_t cSamplesTotal
#endif // NDEBUG
) {
   LOG_0(TraceLevelVerbose, "Entered SumHistogramBuckets");

   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();

   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      if(IsBinaryClassification(runtimeLearningTypeOrCountTargetClasses)) {
         SumHistogramBucketsInternal<2>::Func(
            pThreadStateBoosting,
            cHistogramBuckets
#ifndef NDEBUG
            , cSamplesTotal
#endif // NDEBUG
         );
      } else {
         SumHistogramBucketsInternal<k_dynamicClassification>::Func(
            pThreadStateBoosting,
            cHistogramBuckets
#ifndef NDEBUG
            , cSamplesTotal
#endif // NDEBUG
         );
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      SumHistogramBucketsInternal<k_regression>::Func(
         pThreadStateBoosting,
         cHistogramBuckets
#ifndef NDEBUG
         , cSamplesTotal
#endif // NDEBUG
      );
   }

   LOG_0(TraceLevelVerbose, "Exited SumHistogramBuckets");
}

} // DEFINED_ZONE_NAME
