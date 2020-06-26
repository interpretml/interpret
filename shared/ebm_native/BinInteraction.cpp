// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG

#include "EbmStatisticUtils.h"

#include "Feature.h"
#include "FeatureGroup.h"
#include "DataSetInteraction.h"

#include "InteractionDetection.h"

#include "HistogramTargetEntry.h"
#include "HistogramBucket.h"

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerCountDimensions>
class BinInteractionInternal {
public:
   static void Func(
      EbmInteractionState * const pEbmInteractionState,
      const FeatureCombination * const pFeatureCombination,
      HistogramBucketBase * const aHistogramBucketBase
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      LOG_0(TraceLevelVerbose, "Entered BinDataSetInteraction");

      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets = 
         aHistogramBucketBase->GetHistogramBucket<bClassification>();

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pEbmInteractionState->GetRuntimeLearningTypeOrCountTargetClasses();

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      EBM_ASSERT(!GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bClassification>(cVectorLength);

      const DataSetByFeature * const pDataSet = pEbmInteractionState->GetDataSetByFeature();
      const FloatEbmType * pResidualError = pDataSet->GetResidualPointer();
      const FloatEbmType * const pResidualErrorEnd = pResidualError + cVectorLength * pDataSet->GetCountInstances();

      size_t cFeatures = pFeatureCombination->GetCountFeatures();
      EBM_ASSERT(1 <= cFeatures); // for interactions, we just return 0 for interactions with zero features
      for(size_t iInstance = 0; pResidualErrorEnd != pResidualError; ++iInstance) {
         // this loop gets about twice as slow if you add a single unpredictable branching if statement based on count, even if you still access all the memory
         // in complete sequential order, so we'll probably want to use non-branching instructions for any solution like conditional selection or multiplication
         // this loop gets about 3 times slower if you use a bad pseudo random number generator like rand(), although it might be better if you inlined rand().
         // this loop gets about 10 times slower if you use a proper pseudo random number generator like std::default_random_engine
         // taking all the above together, it seems unlikley we'll use a method of separating sets via single pass randomized set splitting.  Even if count is 
         // stored in memory if shouldn't increase the time spent fetching it by 2 times, unless our bottleneck when threading is overwhelmingly memory pressure 
         // related, and even then we could store the count for a single bit aleviating the memory pressure greatly, if we use the right sampling method 

         // TODO : try using a sampling method with non-repeating instances, and put the count into a bit.  Then unwind that loop either at the byte level 
         //   (8 times) or the uint64_t level.  This can be done without branching and doesn't require random number generators

         // TODO : we can elminate the inner vector loop for regression at least, and also if we add a templated bool for binary class.  Propegate this change 
         //   to all places that we loop on the vector

         size_t cBuckets = 1;
         size_t iBucket = 0;
         size_t iDimension = 0;
         do {
            const Feature * const pInputFeature = pFeatureCombination->GetFeatureCombinationEntries()[iDimension].m_pFeature;
            const size_t cBins = pInputFeature->GetCountBins();
            const StorageDataType * pInputData = pDataSet->GetInputDataPointer(pInputFeature);
            pInputData += iInstance;
            StorageDataType iBinOriginal = *pInputData;
            EBM_ASSERT((IsNumberConvertable<size_t, StorageDataType>(iBinOriginal)));
            size_t iBin = static_cast<size_t>(iBinOriginal);
            EBM_ASSERT(iBin < cBins);
            iBucket += cBuckets * iBin;
            cBuckets *= cBins;
            ++iDimension;
         } while(iDimension < cFeatures);

         HistogramBucket<bClassification> * pHistogramBucketEntry =
            GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, aHistogramBuckets, iBucket);
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucketEntry, aHistogramBucketsEndDebug);
         pHistogramBucketEntry->m_cInstancesInBucket += 1;
         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            const FloatEbmType residualError = *pResidualError;
            // residualError could be NaN
            // for classification, residualError can be anything from -1 to +1 (it cannot be infinity!)
            // for regression, residualError can be anything from +infinity or -infinity
            ArrayToPointer(pHistogramBucketEntry->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError += residualError;
            // m_sumResidualError could be NaN, or anything from +infinity or -infinity in the case of regression
            if(bClassification) {
               EBM_ASSERT(
                  std::isnan(residualError) ||
                  !std::isinf(residualError) && FloatEbmType {
                  -1
               } - k_epsilonResidualError <= residualError && residualError <= FloatEbmType { 1 }
                  );

               // TODO : this code gets executed for each SamplingSet set.  I could probably execute it once and then all the SamplingSet
               //   sets would have this value, but I would need to store the computation in a new memory place, and it might make more sense to calculate this 
               //   values in the CPU rather than put more pressure on memory.  I think controlling this should be done in a MACRO and we should use a class to 
               //   hold the residualError and this computation from that value and then comment out the computation if not necssary and access it through an 
               //   accessor so that we can make the change entirely via macro
               const FloatEbmType denominator = EbmStatistics::ComputeNewtonRaphsonStep(residualError);
               EBM_ASSERT(
                  std::isnan(denominator) ||
                  !std::isinf(denominator) && -k_epsilonResidualError <= denominator && denominator <= FloatEbmType { 0.25 }
               ); // since any one denominatory is limited to -1 <= denominator <= 1, the sum must be representable by a 64 bit number, 

               const FloatEbmType oldDenominator = ArrayToPointer(pHistogramBucketEntry->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator();
               // since any one denominatory is limited to -1 <= denominator <= 1, the sum must be representable by a 64 bit number, 
               EBM_ASSERT(std::isnan(oldDenominator) || !std::isinf(oldDenominator) && -k_epsilonResidualError <= oldDenominator);
               const FloatEbmType newDenominator = oldDenominator + denominator;
               // since any one denominatory is limited to -1 <= denominator <= 1, the sum must be representable by a 64 bit number, 
               EBM_ASSERT(std::isnan(newDenominator) || !std::isinf(newDenominator) && -k_epsilonResidualError <= newDenominator);
               // which will always be representable by a float or double, so we can't overflow to inifinity or -infinity
               ArrayToPointer(pHistogramBucketEntry->m_aHistogramBucketVectorEntry)[iVector].SetSumDenominator(newDenominator);
            }
            ++pResidualError;
         }
      }
      LOG_0(TraceLevelVerbose, "Exited BinDataSetInteraction");
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerCountDimensionsPossible>
class BinInteractionDimensions {
public:
   EBM_INLINE static void Func(
      EbmInteractionState * const pEbmInteractionState,
      const FeatureCombination * const pFeatureCombination,
      HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      static_assert(2 <= compilerCountDimensionsPossible, "can't have less than 2 dimensions for interactions");
      static_assert(compilerCountDimensionsPossible <= k_cDimensionsMax, "can't have more than the max dimensions");

      const size_t runtimeCountDimensions = pFeatureCombination->GetCountFeatures();

      EBM_ASSERT(2 <= runtimeCountDimensions);
      EBM_ASSERT(runtimeCountDimensions <= k_cDimensionsMax);
      if(compilerCountDimensionsPossible == runtimeCountDimensions) {
         BinInteractionInternal<compilerLearningTypeOrCountTargetClasses, compilerCountDimensionsPossible>::Func(
            pEbmInteractionState,
            pFeatureCombination,
            aHistogramBuckets
#ifndef NDEBUG
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      } else {
         BinInteractionDimensions<compilerLearningTypeOrCountTargetClasses, compilerCountDimensionsPossible + 1>::Func(
            pEbmInteractionState,
            pFeatureCombination,
            aHistogramBuckets
#ifndef NDEBUG
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class BinInteractionDimensions<compilerLearningTypeOrCountTargetClasses, k_cCompilerOptimizedCountDimensionsMax + 1> {
public:
   EBM_INLINE static void Func(
      EbmInteractionState * const pEbmInteractionState,
      const FeatureCombination * const pFeatureCombination,
      HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      EBM_ASSERT(2 <= pFeatureCombination->GetCountFeatures());
      EBM_ASSERT(pFeatureCombination->GetCountFeatures() <= k_cDimensionsMax);
      BinInteractionInternal<compilerLearningTypeOrCountTargetClasses, k_DynamicDimensions>::Func(
         pEbmInteractionState,
         pFeatureCombination,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class BinInteractionTarget {
public:
   EBM_INLINE static void Func(
      EbmInteractionState * const pEbmInteractionState,
      const FeatureCombination * const pFeatureCombination,
      HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pEbmInteractionState->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         BinInteractionDimensions<compilerLearningTypeOrCountTargetClassesPossible, 2>::Func(
            pEbmInteractionState,
            pFeatureCombination,
            aHistogramBuckets
#ifndef NDEBUG
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      } else {
         BinInteractionTarget<compilerLearningTypeOrCountTargetClassesPossible + 1>::Func(
            pEbmInteractionState,
            pFeatureCombination,
            aHistogramBuckets
#ifndef NDEBUG
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<>
class BinInteractionTarget<k_cCompilerOptimizedTargetClassesMax + 1> {
public:
   EBM_INLINE static void Func(
      EbmInteractionState * const pEbmInteractionState,
      const FeatureCombination * const pFeatureCombination,
      HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pEbmInteractionState->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pEbmInteractionState->GetRuntimeLearningTypeOrCountTargetClasses());

      BinInteractionDimensions<k_DynamicClassification, 2>::Func(
         pEbmInteractionState,
         pFeatureCombination,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
};

extern void BinInteraction(
   EbmInteractionState * const pEbmInteractionState,
   const FeatureCombination * const pFeatureCombination,
   HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pEbmInteractionState->GetRuntimeLearningTypeOrCountTargetClasses();

   if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
      BinInteractionTarget<2>::Func(
         pEbmInteractionState,
         pFeatureCombination,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      BinInteractionDimensions<k_Regression, 2>::Func(
         pEbmInteractionState,
         pFeatureCombination,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
}
