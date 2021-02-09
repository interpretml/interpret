// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG

#include "EbmStats.h"

#include "FeatureAtomic.h"
#include "FeatureGroup.h"
#include "DataFrameBoosting.h"

#include "Booster.h"
#include "ThreadStateBoosting.h"

#include "HistogramTargetEntry.h"
#include "HistogramBucket.h"

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class BinBoostingZeroDimensions final {
public:

   BinBoostingZeroDimensions() = delete; // this is a static class.  Do not construct

   static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const SamplingSet * const pTrainingSet
   ) {
      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      LOG_0(TraceLevelVerbose, "Entered BinBoostingZeroDimensions");

      HistogramBucketBase * const pHistogramBucketBase = pThreadStateBoosting->GetHistogramBucketBase();
      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pHistogramBucketEntry =
         pHistogramBucketBase->GetHistogramBucket<bClassification>();

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
      EBM_ASSERT(!GetHistogramBucketSizeOverflow(bClassification, cVectorLength)); // we're accessing allocated memory

      const size_t cSamples = pTrainingSet->GetDataFrameBoosting()->GetCountSamples();
      EBM_ASSERT(0 < cSamples);

      const size_t * pCountOccurrences = pTrainingSet->GetCountOccurrences();
      const FloatEbmType * pResidualError = pTrainingSet->GetDataFrameBoosting()->GetResidualPointer();
      // this shouldn't overflow since we're accessing existing memory
      const FloatEbmType * const pResidualErrorEnd = pResidualError + cVectorLength * cSamples;

      HistogramTargetEntry<bClassification> * const pHistogramTargetEntry =
         pHistogramBucketEntry->GetHistogramTargetEntry();
      do {
         // this loop gets about twice as slow if you add a single unpredictable branching if statement based on count, even if you still access all the memory
         //   in complete sequential order, so we'll probably want to use non-branching instructions for any solution like conditional selection or multiplication
         // this loop gets about 3 times slower if you use a bad pseudo random number generator like rand(), although it might be better if you inlined rand().
         // this loop gets about 10 times slower if you use a proper pseudo random number generator like std::default_random_engine
         // taking all the above together, it seems unlikley we'll use a method of separating sets via single pass randomized set splitting.  Even if count is 
         //   stored in memory if shouldn't increase the time spent fetching it by 2 times, unless our bottleneck when threading is overwhelmingly memory 
         //   pressure related, and even then we could store the count for a single bit aleviating the memory pressure greatly, if we use the right 
         //   sampling method 

         // TODO : try using a sampling method with non-repeating samples, and put the count into a bit.  Then unwind that loop either at the byte level 
         //   (8 times) or the uint64_t level.  This can be done without branching and doesn't require random number generators

         const size_t cOccurences = *pCountOccurrences;
         ++pCountOccurrences;
         pHistogramBucketEntry->SetCountSamplesInBucket(pHistogramBucketEntry->GetCountSamplesInBucket() + cOccurences);
         const FloatEbmType cFloatOccurences = static_cast<FloatEbmType>(cOccurences);

         size_t iVector = 0;

#ifndef NDEBUG
#ifdef EXPAND_BINARY_LOGITS
         constexpr bool bExpandBinaryLogits = true;
#else // EXPAND_BINARY_LOGITS
         constexpr bool bExpandBinaryLogits = false;
#endif // EXPAND_BINARY_LOGITS
         FloatEbmType residualTotalDebug = 0;
#endif // NDEBUG
         do {
            const FloatEbmType residualError = *pResidualError;
            EBM_ASSERT(!bClassification ||
               ptrdiff_t { 2 } == runtimeLearningTypeOrCountTargetClasses && !bExpandBinaryLogits ||
               static_cast<ptrdiff_t>(iVector) != k_iZeroResidual || 0 == residualError);
#ifndef NDEBUG
            residualTotalDebug += residualError;
#endif // NDEBUG
            pHistogramTargetEntry[iVector].m_sumResidualError += cFloatOccurences * residualError;
            if(bClassification) {
               // TODO : this code gets executed for each SamplingSet set.  I could probably execute it once and then all the 
               //   SamplingSet sets would have this value, but I would need to store the computation in a new memory place, and it might make 
               //   more sense to calculate this values in the CPU rather than put more pressure on memory.  I think controlling this should be done in a 
               //   MACRO and we should use a class to hold the residualError and this computation from that value and then comment out the computation if 
               //   not necssary and access it through an accessor so that we can make the change entirely via macro
               const FloatEbmType denominator = EbmStats::ComputeNewtonRaphsonStep(residualError);
               pHistogramTargetEntry[iVector].SetSumDenominator(pHistogramTargetEntry[iVector].GetSumDenominator() + cFloatOccurences * denominator);
            }
            ++pResidualError;
            ++iVector;
            // if we use this specific format where (iVector < cVectorLength) then the compiler collapses alway the loop for small cVectorLength values
            // if we make this (iVector != cVectorLength) then the loop is not collapsed
            // the compiler seems to not mind if we make this a for loop or do loop in terms of collapsing away the loop
         } while(iVector < cVectorLength);

         EBM_ASSERT(
            !bClassification ||
            ptrdiff_t { 2 } == runtimeLearningTypeOrCountTargetClasses && !bExpandBinaryLogits ||
            0 <= k_iZeroResidual ||
            std::isnan(residualTotalDebug) ||
            -k_epsilonResidualError < residualTotalDebug && residualTotalDebug < k_epsilonResidualError
         );
      } while(pResidualErrorEnd != pResidualError);
      LOG_0(TraceLevelVerbose, "Exited BinBoostingZeroDimensions");
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class BinBoostingZeroDimensionsTarget final {
public:

   BinBoostingZeroDimensionsTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const SamplingSet * const pTrainingSet
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         BinBoostingZeroDimensions<compilerLearningTypeOrCountTargetClassesPossible>::Func(
            pThreadStateBoosting,
            pTrainingSet
         );
      } else {
         BinBoostingZeroDimensionsTarget<compilerLearningTypeOrCountTargetClassesPossible + 1>::Func(
            pThreadStateBoosting,
            pTrainingSet
         );
      }
   }
};

template<>
class BinBoostingZeroDimensionsTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   BinBoostingZeroDimensionsTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const SamplingSet * const pTrainingSet
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses());

      BinBoostingZeroDimensions<k_dynamicClassification>::Func(
         pThreadStateBoosting,
         pTrainingSet
      );
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerCountItemsPerBitPackedDataUnit>
class BinBoostingInternal final {
public:

   BinBoostingInternal() = delete; // this is a static class.  Do not construct

   static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      const SamplingSet * const pTrainingSet
   ) {
      constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

      LOG_0(TraceLevelVerbose, "Entered BinBoostingInternal");

      HistogramBucketBase * const aHistogramBucketBase = pThreadStateBoosting->GetHistogramBucketBase();
      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets =
         aHistogramBucketBase->GetHistogramBucket<bClassification>();

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();

      const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
         compilerLearningTypeOrCountTargetClasses,
         runtimeLearningTypeOrCountTargetClasses
      );
      const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);

      const size_t cItemsPerBitPackedDataUnit = GET_COUNT_ITEMS_PER_BIT_PACKED_DATA_UNIT(
         compilerCountItemsPerBitPackedDataUnit,
         pFeatureGroup->GetCountItemsPerBitPackedDataUnit()
      );
      EBM_ASSERT(1 <= cItemsPerBitPackedDataUnit);
      EBM_ASSERT(cItemsPerBitPackedDataUnit <= k_cBitsForStorageType);
      const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackedDataUnit);
      EBM_ASSERT(1 <= cBitsPerItemMax);
      EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
      const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);
      EBM_ASSERT(!GetHistogramBucketSizeOverflow(bClassification, cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);

      const size_t cSamples = pTrainingSet->GetDataFrameBoosting()->GetCountSamples();
      EBM_ASSERT(0 < cSamples);

      const size_t * pCountOccurrences = pTrainingSet->GetCountOccurrences();
      const StorageDataType * pInputData = pTrainingSet->GetDataFrameBoosting()->GetInputDataPointer(pFeatureGroup);
      const FloatEbmType * pResidualError = pTrainingSet->GetDataFrameBoosting()->GetResidualPointer();

      // this shouldn't overflow since we're accessing existing memory
      const FloatEbmType * const pResidualErrorTrueEnd = pResidualError + cVectorLength * cSamples;
      const FloatEbmType * pResidualErrorExit = pResidualErrorTrueEnd;
      size_t cItemsRemaining = cSamples;
      if(cSamples <= cItemsPerBitPackedDataUnit) {
         goto one_last_loop;
      }
      pResidualErrorExit = pResidualErrorTrueEnd - cVectorLength * ((cSamples - 1) % cItemsPerBitPackedDataUnit + 1);
      EBM_ASSERT(pResidualError < pResidualErrorExit);
      EBM_ASSERT(pResidualErrorExit < pResidualErrorTrueEnd);

      do {
         // this loop gets about twice as slow if you add a single unpredictable branching if statement based on count, even if you still access all the memory
         // in complete sequential order, so we'll probably want to use non-branching instructions for any solution like conditional selection or multiplication
         // this loop gets about 3 times slower if you use a bad pseudo random number generator like rand(), although it might be better if you inlined rand().
         // this loop gets about 10 times slower if you use a proper pseudo random number generator like std::default_random_engine
         // taking all the above together, it seems unlikley we'll use a method of separating sets via single pass randomized set splitting.  Even if count is 
         // stored in memory if shouldn't increase the time spent fetching it by 2 times, unless our bottleneck when threading is overwhelmingly memory pressure
         // related, and even then we could store the count for a single bit aleviating the memory pressure greatly, if we use the right sampling method 

         // TODO : try using a sampling method with non-repeating samples, and put the count into a bit.  Then unwind that loop either at the byte level 
         //   (8 times) or the uint64_t level.  This can be done without branching and doesn't require random number generators

         cItemsRemaining = cItemsPerBitPackedDataUnit;
         // TODO : jumping back into this loop and changing cItemsRemaining to a dynamic value that isn't compile time determinable
         // causes this function to NOT be optimized as much as it could if we had two separate loops.  We're just trying this out for now though
      one_last_loop:;
         // we store the already multiplied dimensional value in *pInputData
         size_t iTensorBinCombined = static_cast<size_t>(*pInputData);
         ++pInputData;
         do {
            const size_t iTensorBin = maskBits & iTensorBinCombined;

            HistogramBucket<bClassification> * const pHistogramBucketEntry = GetHistogramBucketByIndex(
               cBytesPerHistogramBucket,
               aHistogramBuckets,
               iTensorBin
            );

            ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucketEntry, pThreadStateBoosting->GetHistogramBucketsEndDebug());
            const size_t cOccurences = *pCountOccurrences;
            ++pCountOccurrences;
            pHistogramBucketEntry->SetCountSamplesInBucket(pHistogramBucketEntry->GetCountSamplesInBucket() + cOccurences);
            const FloatEbmType cFloatOccurences = static_cast<FloatEbmType>(cOccurences);
            HistogramTargetEntry<bClassification> * pHistogramTargetEntry = 
               pHistogramBucketEntry->GetHistogramTargetEntry();

            size_t iVector = 0;

#ifndef NDEBUG
#ifdef EXPAND_BINARY_LOGITS
            constexpr bool bExpandBinaryLogits = true;
#else // EXPAND_BINARY_LOGITS
            constexpr bool bExpandBinaryLogits = false;
#endif // EXPAND_BINARY_LOGITS
            FloatEbmType residualTotalDebug = 0;
#endif // NDEBUG
            do {
               const FloatEbmType residualError = *pResidualError;
               EBM_ASSERT(
                  !bClassification ||
                  ptrdiff_t { 2 } == runtimeLearningTypeOrCountTargetClasses && !bExpandBinaryLogits ||
                  static_cast<ptrdiff_t>(iVector) != k_iZeroResidual ||
                  0 == residualError
               );
#ifndef NDEBUG
               residualTotalDebug += residualError;
#endif // NDEBUG
               pHistogramTargetEntry[iVector].m_sumResidualError += cFloatOccurences * residualError;
               if(bClassification) {
                  // TODO : this code gets executed for each SamplingSet set.  I could probably execute it once and then all the
                  //   SamplingSet sets would have this value, but I would need to store the computation in a new memory place, and it might 
                  //   make more sense to calculate this values in the CPU rather than put more pressure on memory.  I think controlling this should be 
                  //   done in a MACRO and we should use a class to hold the residualError and this computation from that value and then comment out the 
                  //   computation if not necssary and access it through an accessor so that we can make the change entirely via macro
                  const FloatEbmType denominator = EbmStats::ComputeNewtonRaphsonStep(residualError);
                  pHistogramTargetEntry[iVector].SetSumDenominator(
                     pHistogramTargetEntry[iVector].GetSumDenominator() + cFloatOccurences * denominator
                  );
               }
               ++pResidualError;
               ++iVector;
               // if we use this specific format where (iVector < cVectorLength) then the compiler collapses alway the loop for small cVectorLength values
               // if we make this (iVector != cVectorLength) then the loop is not collapsed
               // the compiler seems to not mind if we make this a for loop or do loop in terms of collapsing away the loop
            } while(iVector < cVectorLength);

            EBM_ASSERT(
               !bClassification ||
               ptrdiff_t { 2 } == runtimeLearningTypeOrCountTargetClasses && !bExpandBinaryLogits ||
               0 <= k_iZeroResidual ||
               -k_epsilonResidualError < residualTotalDebug && residualTotalDebug < k_epsilonResidualError
            );

            iTensorBinCombined >>= cBitsPerItemMax;
            // TODO : try replacing cItemsRemaining with a pResidualErrorInnerLoopEnd which eliminates one subtact operation, but might make it harder for 
            //   the compiler to optimize the loop away
            --cItemsRemaining;
         } while(0 != cItemsRemaining);
      } while(pResidualErrorExit != pResidualError);

      // first time through?
      if(pResidualErrorTrueEnd != pResidualError) {
         LOG_0(TraceLevelVerbose, "Handling last BinBoostingInternal loop");

         EBM_ASSERT(0 == (pResidualErrorTrueEnd - pResidualError) % cVectorLength);
         cItemsRemaining = (pResidualErrorTrueEnd - pResidualError) / cVectorLength;
         EBM_ASSERT(0 < cItemsRemaining);
         EBM_ASSERT(cItemsRemaining <= cItemsPerBitPackedDataUnit);

         pResidualErrorExit = pResidualErrorTrueEnd;

         goto one_last_loop;
      }

      LOG_0(TraceLevelVerbose, "Exited BinBoostingInternal");
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class BinBoostingNormalTarget final {
public:

   BinBoostingNormalTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      const SamplingSet * const pTrainingSet
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         BinBoostingInternal<compilerLearningTypeOrCountTargetClassesPossible, k_cItemsPerBitPackedDataUnitDynamic>::Func(
            pThreadStateBoosting,
            pFeatureGroup,
            pTrainingSet
         );
      } else {
         BinBoostingNormalTarget<compilerLearningTypeOrCountTargetClassesPossible + 1>::Func(
            pThreadStateBoosting,
            pFeatureGroup,
            pTrainingSet
         );
      }
   }
};

template<>
class BinBoostingNormalTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   BinBoostingNormalTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      const SamplingSet * const pTrainingSet
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses());

      BinBoostingInternal<k_dynamicClassification, k_cItemsPerBitPackedDataUnitDynamic>::Func(
         pThreadStateBoosting,
         pFeatureGroup,
         pTrainingSet
      );
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t compilerCountItemsPerBitPackedDataUnitPossible>
class BinBoostingSIMDPacking final {
public:

   BinBoostingSIMDPacking() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      const SamplingSet * const pTrainingSet
   ) {
      const size_t runtimeCountItemsPerBitPackedDataUnit = pFeatureGroup->GetCountItemsPerBitPackedDataUnit();

      EBM_ASSERT(1 <= runtimeCountItemsPerBitPackedDataUnit);
      EBM_ASSERT(runtimeCountItemsPerBitPackedDataUnit <= k_cBitsForStorageType);
      static_assert(compilerCountItemsPerBitPackedDataUnitPossible <= k_cBitsForStorageType, "We can't have this many items in a data pack.");
      if(compilerCountItemsPerBitPackedDataUnitPossible == runtimeCountItemsPerBitPackedDataUnit) {
         BinBoostingInternal<compilerLearningTypeOrCountTargetClasses, compilerCountItemsPerBitPackedDataUnitPossible>::Func(
            pThreadStateBoosting,
            pFeatureGroup,
            pTrainingSet
         );
      } else {
         BinBoostingSIMDPacking<
            compilerLearningTypeOrCountTargetClasses,
            GetNextCountItemsBitPacked(compilerCountItemsPerBitPackedDataUnitPossible)
         >::Func(
            pThreadStateBoosting,
            pFeatureGroup,
            pTrainingSet
         );
      }
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class BinBoostingSIMDPacking<compilerLearningTypeOrCountTargetClasses, k_cItemsPerBitPackedDataUnitDynamic> final {
public:

   BinBoostingSIMDPacking() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      const SamplingSet * const pTrainingSet
   ) {
      EBM_ASSERT(1 <= pFeatureGroup->GetCountItemsPerBitPackedDataUnit());
      EBM_ASSERT(pFeatureGroup->GetCountItemsPerBitPackedDataUnit() <= k_cBitsForStorageType);
      BinBoostingInternal<compilerLearningTypeOrCountTargetClasses, k_cItemsPerBitPackedDataUnitDynamic>::Func(
         pThreadStateBoosting,
         pFeatureGroup,
         pTrainingSet
      );
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClassesPossible>
class BinBoostingSIMDTarget final {
public:

   BinBoostingSIMDTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      const SamplingSet * const pTrainingSet
   ) {
      static_assert(IsClassification(compilerLearningTypeOrCountTargetClassesPossible), "compilerLearningTypeOrCountTargetClassesPossible needs to be a classification");
      static_assert(compilerLearningTypeOrCountTargetClassesPossible <= k_cCompilerOptimizedTargetClassesMax, "We can't have this many items in a data pack.");

      Booster * const pBooster = pThreadStateBoosting->GetBooster();
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
      EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);

      if(compilerLearningTypeOrCountTargetClassesPossible == runtimeLearningTypeOrCountTargetClasses) {
         BinBoostingSIMDPacking<
            compilerLearningTypeOrCountTargetClassesPossible,
            k_cItemsPerBitPackedDataUnitMax
         >::Func(
            pThreadStateBoosting,
            pFeatureGroup,
            pTrainingSet
         );
      } else {
         BinBoostingSIMDTarget<compilerLearningTypeOrCountTargetClassesPossible + 1>::Func(
            pThreadStateBoosting,
            pFeatureGroup,
            pTrainingSet
         );
      }
   }
};

template<>
class BinBoostingSIMDTarget<k_cCompilerOptimizedTargetClassesMax + 1> final {
public:

   BinBoostingSIMDTarget() = delete; // this is a static class.  Do not construct

   INLINE_ALWAYS static void Func(
      ThreadStateBoosting * const pThreadStateBoosting,
      const FeatureGroup * const pFeatureGroup,
      const SamplingSet * const pTrainingSet
   ) {
      static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");

      EBM_ASSERT(IsClassification(pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses()));
      EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < pThreadStateBoosting->GetBooster()->GetRuntimeLearningTypeOrCountTargetClasses());

      BinBoostingSIMDPacking<k_dynamicClassification, k_cItemsPerBitPackedDataUnitMax>::Func(
         pThreadStateBoosting,
         pFeatureGroup,
         pTrainingSet
      );
   }
};

extern void BinBoosting(
   ThreadStateBoosting * const pThreadStateBoosting,
   const FeatureGroup * const pFeatureGroup,
   const SamplingSet * const pTrainingSet
) {
   LOG_0(TraceLevelVerbose, "Entered BinBoosting");

   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();

   if(nullptr == pFeatureGroup) {
      if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
         BinBoostingZeroDimensionsTarget<2>::Func(
            pThreadStateBoosting,
            pTrainingSet
         );
      } else {
         EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
         BinBoostingZeroDimensions<k_regression>::Func(
            pThreadStateBoosting,
            pTrainingSet
         );
      }
   } else {
      EBM_ASSERT(1 <= pFeatureGroup->GetCountSignificantFeatures());
      if(k_bUseSIMD) {
         // TODO : enable SIMD(AVX-512) to work

         // 64 - do 8 at a time and unroll the loop 8 times.  These are bool features and are common.  Put the unrolled inner loop into a function
         // 32 - do 8 at a time and unroll the loop 4 times.  These are bool features and are common.  Put the unrolled inner loop into a function
         // 21 - do 8 at a time and unroll the loop 3 times (ignore the last 3 with a mask)
         // 16 - do 8 at a time and unroll the loop 2 times.  These are bool features and are common.  Put the unrolled inner loop into a function
         // 12 - do 8 of them, shift the low 4 upwards and then load the next 12 and take the top 4, repeat.
         // 10 - just drop this down to packing 8 together
         // 9 - just drop this down to packing 8 together
         // 8 - do all 8 at a time without an inner loop.  This is one of the most common values.  256 binned values
         // 7,6,5,4,3,2,1 - use a mask to exclude the non-used conditions and process them like the 8.  These are rare since they require more than 256 values

         if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
            BinBoostingSIMDTarget<2>::Func(
               pThreadStateBoosting,
               pFeatureGroup,
               pTrainingSet
            );
         } else {
            EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
            BinBoostingSIMDPacking<k_regression, k_cItemsPerBitPackedDataUnitMax>::Func(
               pThreadStateBoosting,
               pFeatureGroup,
               pTrainingSet
            );
         }
      } else {
         // there isn't much benefit in eliminating the loop that unpacks a data unit unless we're also unpacking that to SIMD code
         // Our default packing structure is to bin continuous values to 256 values, and we have 64 bit packing structures, so we usually
         // have more than 8 values per memory fetch.  Eliminating the inner loop for multiclass is valuable since we can have low numbers like 3 class,
         // 4 class, etc, but by the time we get to 8 loops with exp inside and a lot of other instructures we should worry that our code expansion
         // will exceed the L1 instruction cache size.  With SIMD we do 8 times the work in the same number of instructions so these are lesser issues

         if(IsClassification(runtimeLearningTypeOrCountTargetClasses)) {
            BinBoostingNormalTarget<2>::Func(
               pThreadStateBoosting,
               pFeatureGroup,
               pTrainingSet
            );
         } else {
            EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
            BinBoostingInternal<k_regression, k_cItemsPerBitPackedDataUnitDynamic>::Func(
               pThreadStateBoosting,
               pFeatureGroup,
               pTrainingSet
            );
         }
      }
   }

   LOG_0(TraceLevelVerbose, "Exited BinBoosting");
}
