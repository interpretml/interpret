// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef HISTOGRAM_BUCKET_H
#define HISTOGRAM_BUCKET_H

#include <type_traits> // std::is_standard_layout
#include <string.h> // memset
#include <stddef.h> // size_t, ptrdiff_t
#include <cmath> // abs

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "HistogramBucketVectorEntry.h"
#include "CachedThreadResources.h"
#include "Feature.h"
#include "FeatureCombination.h"
#include "DataSetByFeatureCombination.h"
#include "DataSetByFeature.h"
#include "SamplingWithReplacement.h"

// we don't need to handle multi-dimensional inputs with more than 64 bits total
// the rational is that we need to bin this data, and our binning memory will be N1*N1*...*N(D-1)*N(D)
// So, even for binary input featuers, we would have 2^64 bins, and that would take more memory than a 64 bit machine can have
// Similarily, we need a huge amount of memory in order to bin any data with a combined total of more than 64 bits.
// The worst case I can think of is where we have 3 bins, requiring 2 bit each, and we overflowed at 33 dimensions
// in that bad case, we would have 3^33 bins * 8 bytes minimum per bin = 44472484532444184 bytes, which would take 56 bits to express
// Nobody is every going to build a machine with more than 64 bits, since you need a non-trivial volume of material assuming bits require
// more than several atoms to store.
// we can just return an out of memory error if someone requests a set of features that would sum to more than 64 bits
// we DO need to check for this error condition though, since it's not impossible for someone to request this kind of thing.
// any dimensions with only 1 bin don't count since you would just be multiplying by 1 for each such dimension

template<bool bClassification>
struct HistogramBucket;

template<bool bClassification>
EBM_INLINE bool GetHistogramBucketSizeOverflow(const size_t cVectorLength) {
   return IsMultiplyError(
      sizeof(HistogramBucketVectorEntry<bClassification>), cVectorLength) ? 
      true : 
      IsAddError(
         sizeof(HistogramBucket<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>), 
         sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength
      ) ? true : false;
}
template<bool bClassification>
EBM_INLINE size_t GetHistogramBucketSize(const size_t cVectorLength) {
   return sizeof(HistogramBucket<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>) + 
      sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength;
}
template<bool bClassification>
EBM_INLINE HistogramBucket<bClassification> * GetHistogramBucketByIndex(
   const size_t cBytesPerHistogramBucket, 
   HistogramBucket<bClassification> * const aHistogramBuckets, 
   const size_t iBin
) {
   // TODO : remove the use of this function anywhere performant by making the tensor calculation start with the # of bytes per histogram bucket, 
   // therefore eliminating the need to do the multiplication at the end when finding the index
   return reinterpret_cast<HistogramBucket<bClassification> *>(reinterpret_cast<char *>(aHistogramBuckets) + iBin * cBytesPerHistogramBucket);
}
template<bool bClassification>
EBM_INLINE const HistogramBucket<bClassification> * GetHistogramBucketByIndex(
   const size_t cBytesPerHistogramBucket, 
   const HistogramBucket<bClassification> * const aHistogramBuckets, 
   const size_t iBin
) {
   // TODO : remove the use of this function anywhere performant by making the tensor calculation start with the # of bytes per histogram bucket, 
   //   therefore eliminating the need to do the multiplication at the end when finding the index
   return reinterpret_cast<const HistogramBucket<bClassification> *>(reinterpret_cast<const char *>(aHistogramBuckets) + iBin * cBytesPerHistogramBucket);
}

// keep this as a MACRO so that we don't materialize any of the parameters on non-debug builds
#define ASSERT_BINNED_BUCKET_OK(MACRO_cBytesPerHistogramBucket, MACRO_pHistogramBucket, MACRO_aHistogramBucketsEnd) \
   (EBM_ASSERT(reinterpret_cast<const char *>(MACRO_pHistogramBucket) + static_cast<size_t>(MACRO_cBytesPerHistogramBucket) <= \
      reinterpret_cast<const char *>(MACRO_aHistogramBucketsEnd)))

struct HistogramBucketBase {
   // TODO: is HistogramBucketBase used?  I created it meaning for us to have a common point in classes where bClassification 
   //       wasn't needed but we needed pointers to these

   // empty base optimization is REQUIRED by the C++11 standard for StandardLayoutType objects, so this struct will use 0 bytes in our derived class
   // https://en.cppreference.com/w/cpp/language/ebo
};
static_assert(std::is_standard_layout<HistogramBucketBase>::value,
   "HistogramBucket uses the struct hack, so it needs to be standard layout class otherwise we can't depend on the layout!");

template<bool bClassification>
struct HistogramBucket final : public HistogramBucketBase {
public:

   size_t m_cInstancesInBucket;

   // use the "struct hack" since Flexible array member method is not available in C++
   // aHistogramBucketVectorEntry must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   HistogramBucketVectorEntry<bClassification> m_aHistogramBucketVectorEntry[1];

   EBM_INLINE void Add(const HistogramBucket<bClassification> & other, const size_t cVectorLength) {
      m_cInstancesInBucket += other.m_cInstancesInBucket;
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         ARRAY_TO_POINTER(m_aHistogramBucketVectorEntry)[iVector].Add(ARRAY_TO_POINTER_CONST(other.m_aHistogramBucketVectorEntry)[iVector]);
      }
   }

   EBM_INLINE void Subtract(const HistogramBucket<bClassification> & other, const size_t cVectorLength) {
      m_cInstancesInBucket -= other.m_cInstancesInBucket;
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         ARRAY_TO_POINTER(m_aHistogramBucketVectorEntry)[iVector].Subtract(ARRAY_TO_POINTER_CONST(other.m_aHistogramBucketVectorEntry)[iVector]);
      }
   }

   EBM_INLINE void Copy(const HistogramBucket<bClassification> & other, const size_t cVectorLength) {
      EBM_ASSERT(!GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bClassification>(cVectorLength);
      memcpy(this, &other, cBytesPerHistogramBucket);
   }

   EBM_INLINE void Zero(const size_t cVectorLength) {
      EBM_ASSERT(!GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bClassification>(cVectorLength);
      memset(this, 0, cBytesPerHistogramBucket);
   }

   EBM_INLINE void AssertZero(const size_t cVectorLength) const {
      UNUSED(cVectorLength);
#ifndef NDEBUG
      EBM_ASSERT(0 == m_cInstancesInBucket);
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         ARRAY_TO_POINTER_CONST(m_aHistogramBucketVectorEntry)[iVector].AssertZero();
      }
#endif // NDEBUG
   }
};
static_assert(std::is_standard_layout<HistogramBucket<false>>::value && std::is_standard_layout<HistogramBucket<true>>::value, 
   "HistogramBucket uses the struct hack, so it needs to be standard layout class otherwise we can't depend on the layout!");

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
void BinDataSetTrainingZeroDimensions(
   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pHistogramBucketEntry, 
   const SamplingMethod * const pTrainingSet, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
) {
   constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelVerbose, "Entered BinDataSetTrainingZeroDimensions");

   const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
      compilerLearningTypeOrCountTargetClasses,
      runtimeLearningTypeOrCountTargetClasses
   );
   const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)); // we're accessing allocated memory

   const size_t cInstances = pTrainingSet->m_pOriginDataSet->GetCountInstances();
   EBM_ASSERT(0 < cInstances);

   const SamplingWithReplacement * const pSamplingWithReplacement = static_cast<const SamplingWithReplacement *>(pTrainingSet);
   const size_t * pCountOccurrences = pSamplingWithReplacement->m_aCountOccurrences;
   const FloatEbmType * pResidualError = pSamplingWithReplacement->m_pOriginDataSet->GetResidualPointer();
   // this shouldn't overflow since we're accessing existing memory
   const FloatEbmType * const pResidualErrorEnd = pResidualError + cVectorLength * cInstances;

   HistogramBucketVectorEntry<bClassification> * const pHistogramBucketVectorEntry =
      ARRAY_TO_POINTER(pHistogramBucketEntry->m_aHistogramBucketVectorEntry);
   do {
      // this loop gets about twice as slow if you add a single unpredictable branching if statement based on count, even if you still access all the memory
      //   in complete sequential order, so we'll probably want to use non-branching instructions for any solution like conditional selection or multiplication
      // this loop gets about 3 times slower if you use a bad pseudo random number generator like rand(), although it might be better if you inlined rand().
      // this loop gets about 10 times slower if you use a proper pseudo random number generator like std::default_random_engine
      // taking all the above together, it seems unlikley we'll use a method of separating sets via single pass randomized set splitting.  Even if count is 
      //   stored in memory if shouldn't increase the time spent fetching it by 2 times, unless our bottleneck when threading is overwhelmingly memory 
      //   pressure related, and even then we could store the count for a single bit aleviating the memory pressure greatly, if we use the right 
      //   sampling method 

      // TODO : try using a sampling method with non-repeating instances, and put the count into a bit.  Then unwind that loop either at the byte level 
      //   (8 times) or the uint64_t level.  This can be done without branching and doesn't require random number generators

      const size_t cOccurences = *pCountOccurrences;
      ++pCountOccurrences;
      pHistogramBucketEntry->m_cInstancesInBucket += cOccurences;
      const FloatEbmType cFloatOccurences = static_cast<FloatEbmType>(cOccurences);

#ifndef NDEBUG
#ifdef EXPAND_BINARY_LOGITS
      constexpr bool bExpandBinaryLogits = true;
#else // EXPAND_BINARY_LOGITS
      constexpr bool bExpandBinaryLogits = false;
#endif // EXPAND_BINARY_LOGITS
      FloatEbmType residualTotalDebug = 0;
#endif // NDEBUG
      size_t iVector = 0;
      do {
         const FloatEbmType residualError = *pResidualError;
         EBM_ASSERT(!bClassification ||
            ptrdiff_t { 2 } == runtimeLearningTypeOrCountTargetClasses && !bExpandBinaryLogits || 
            static_cast<ptrdiff_t>(iVector) != k_iZeroResidual || 0 == residualError);
#ifndef NDEBUG
         residualTotalDebug += residualError;
#endif // NDEBUG
         pHistogramBucketVectorEntry[iVector].m_sumResidualError += cFloatOccurences * residualError;
         if(bClassification) {
            // TODO : this code gets executed for each SamplingWithReplacement set.  I could probably execute it once and then all the 
            //   SamplingWithReplacement sets would have this value, but I would need to store the computation in a new memory place, and it might make 
            //   more sense to calculate this values in the CPU rather than put more pressure on memory.  I think controlling this should be done in a 
            //   MACRO and we should use a class to hold the residualError and this computation from that value and then comment out the computation if 
            //   not necssary and access it through an accessor so that we can make the change entirely via macro
            const FloatEbmType denominator = EbmStatistics::ComputeNewtonRaphsonStep(residualError);
            pHistogramBucketVectorEntry[iVector].SetSumDenominator(pHistogramBucketVectorEntry[iVector].GetSumDenominator() + cFloatOccurences * denominator);
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
   LOG_0(TraceLevelVerbose, "Exited BinDataSetTrainingZeroDimensions");
}

// TODO : remove cCompilerDimensions since we don't need it anymore, and replace it with a more useful number like the number of cItemsPerBitPackedDataUnit
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t cCompilerDimensions>
void BinDataSetTraining(HistogramBucket<IsClassification(
   compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, 
   const FeatureCombination * const pFeatureCombination, 
   const SamplingMethod * const pTrainingSet, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelVerbose, "Entered BinDataSetTraining");

   EBM_ASSERT(cCompilerDimensions == pFeatureCombination->m_cFeatures);
   static_assert(1 <= cCompilerDimensions, "cCompilerDimensions must be 1 or greater");

   const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
      compilerLearningTypeOrCountTargetClasses,
      runtimeLearningTypeOrCountTargetClasses
   );
   const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
   const size_t cItemsPerBitPackedDataUnit = pFeatureCombination->m_cItemsPerBitPackedDataUnit;
   EBM_ASSERT(1 <= cItemsPerBitPackedDataUnit);
   EBM_ASSERT(cItemsPerBitPackedDataUnit <= k_cBitsForStorageType);
   const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackedDataUnit);
   EBM_ASSERT(1 <= cBitsPerItemMax);
   EBM_ASSERT(cBitsPerItemMax <= k_cBitsForStorageType);
   const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bClassification>(cVectorLength);

   const size_t cInstances = pTrainingSet->m_pOriginDataSet->GetCountInstances();
   EBM_ASSERT(0 < cInstances);

   const SamplingWithReplacement * const pSamplingWithReplacement = static_cast<const SamplingWithReplacement *>(pTrainingSet);
   const size_t * pCountOccurrences = pSamplingWithReplacement->m_aCountOccurrences;
   const StorageDataType * pInputData = pSamplingWithReplacement->m_pOriginDataSet->GetInputDataPointer(pFeatureCombination);
   const FloatEbmType * pResidualError = pSamplingWithReplacement->m_pOriginDataSet->GetResidualPointer();

   // this shouldn't overflow since we're accessing existing memory
   const FloatEbmType * const pResidualErrorTrueEnd = pResidualError + cVectorLength * cInstances;
   const FloatEbmType * pResidualErrorExit = pResidualErrorTrueEnd;
   size_t cItemsRemaining = cInstances;
   if(cInstances <= cItemsPerBitPackedDataUnit) {
      goto one_last_loop;
   }
   pResidualErrorExit = pResidualErrorTrueEnd - cVectorLength * ((cInstances - 1) % cItemsPerBitPackedDataUnit + 1);
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

      // TODO : try using a sampling method with non-repeating instances, and put the count into a bit.  Then unwind that loop either at the byte level 
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

         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucketEntry, aHistogramBucketsEndDebug);
         const size_t cOccurences = *pCountOccurrences;
         ++pCountOccurrences;
         pHistogramBucketEntry->m_cInstancesInBucket += cOccurences;
         const FloatEbmType cFloatOccurences = static_cast<FloatEbmType>(cOccurences);
         HistogramBucketVectorEntry<bClassification> * pHistogramBucketVectorEntry = ARRAY_TO_POINTER(
            pHistogramBucketEntry->m_aHistogramBucketVectorEntry
         );
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
            pHistogramBucketVectorEntry[iVector].m_sumResidualError += cFloatOccurences * residualError;
            if(bClassification) {
               // TODO : this code gets executed for each SamplingWithReplacement set.  I could probably execute it once and then all the
               //   SamplingWithReplacement sets would have this value, but I would need to store the computation in a new memory place, and it might 
               //   make more sense to calculate this values in the CPU rather than put more pressure on memory.  I think controlling this should be 
               //   done in a MACRO and we should use a class to hold the residualError and this computation from that value and then comment out the 
               //   computation if not necssary and access it through an accessor so that we can make the change entirely via macro
               const FloatEbmType denominator = EbmStatistics::ComputeNewtonRaphsonStep(residualError);
               pHistogramBucketVectorEntry[iVector].SetSumDenominator(
                  pHistogramBucketVectorEntry[iVector].GetSumDenominator() + cFloatOccurences * denominator
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
      LOG_0(TraceLevelVerbose, "Handling last BinDataSetTraining loop");

      EBM_ASSERT(0 == (pResidualErrorTrueEnd - pResidualError) % cVectorLength);
      cItemsRemaining = (pResidualErrorTrueEnd - pResidualError) / cVectorLength;
      EBM_ASSERT(0 < cItemsRemaining);
      EBM_ASSERT(cItemsRemaining <= cItemsPerBitPackedDataUnit);

      pResidualErrorExit = pResidualErrorTrueEnd;

      goto one_last_loop;
   }

   LOG_0(TraceLevelVerbose, "Exited BinDataSetTraining");
}

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t cCompilerDimensions>
class RecursiveBinDataSetTraining {
   // C++ does not allow partial function specialization, so we need to use these cumbersome inline static class functions to do partial
   //   function specialization
public:
   EBM_INLINE static void Recursive(
      const size_t cRuntimeDimensions, 
      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, 
      const FeatureCombination * const pFeatureCombination, 
      const SamplingMethod * const pTrainingSet, 
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      EBM_ASSERT(cRuntimeDimensions < k_cDimensionsMax);
      static_assert(
         cCompilerDimensions < k_cDimensionsMax, 
         "cCompilerDimensions must be less than or equal to k_cDimensionsMax.  This line only handles the less than part, but we handle the equals "
         "in a partial specialization template.");
      if(cCompilerDimensions == cRuntimeDimensions) {
         BinDataSetTraining<compilerLearningTypeOrCountTargetClasses, cCompilerDimensions>(
            aHistogramBuckets, 
            pFeatureCombination, 
            pTrainingSet, 
            runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      } else {
         RecursiveBinDataSetTraining<compilerLearningTypeOrCountTargetClasses, 1 + cCompilerDimensions>::Recursive(
            cRuntimeDimensions, 
            aHistogramBuckets, 
            pFeatureCombination, 
            pTrainingSet, 
            runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
class RecursiveBinDataSetTraining<compilerLearningTypeOrCountTargetClasses, k_cDimensionsMax> {
   // C++ does not allow partial function specialization, so we need to use these cumbersome inline static class functions to do partial function specialization
public:
   EBM_INLINE static void Recursive(
      const size_t cRuntimeDimensions, 
      HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, 
      const FeatureCombination * const pFeatureCombination, 
      const SamplingMethod * const pTrainingSet, 
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      UNUSED(cRuntimeDimensions);
      EBM_ASSERT(k_cDimensionsMax == cRuntimeDimensions);
      BinDataSetTraining<compilerLearningTypeOrCountTargetClasses, k_cDimensionsMax>(
         aHistogramBuckets, 
         pFeatureCombination, 
         pTrainingSet, 
         runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
};

// TODO: make the number of dimensions (pFeatureCombination->m_cFeatures) a template parameter so that we don't have to have the inner loop that is 
//   very bad for performance.  Since the data will be stored contiguously and have the same length in the future, we can just loop based on the 
//   number of dimensions, so we might as well have a couple of different values
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
void BinDataSetInteraction(HistogramBucket<IsClassification(
   compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, 
   const FeatureCombination * const pFeatureCombination, 
   const DataSetByFeature * const pDataSet, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelVerbose, "Entered BinDataSetInteraction");

   const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
      compilerLearningTypeOrCountTargetClasses,
      runtimeLearningTypeOrCountTargetClasses
   );
   const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bClassification>(cVectorLength);

   const FloatEbmType * pResidualError = pDataSet->GetResidualPointer();
   const FloatEbmType * const pResidualErrorEnd = pResidualError + cVectorLength * pDataSet->GetCountInstances();

   size_t cFeatures = pFeatureCombination->m_cFeatures;
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
         const Feature * const pInputFeature = ARRAY_TO_POINTER_CONST(pFeatureCombination->m_FeatureCombinationEntry)[iDimension].m_pFeature;
         const size_t cBins = pInputFeature->m_cBins;
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
         ARRAY_TO_POINTER(pHistogramBucketEntry->m_aHistogramBucketVectorEntry)[iVector].m_sumResidualError += residualError;
         // m_sumResidualError could be NaN, or anything from +infinity or -infinity in the case of regression
         if(bClassification) {
            EBM_ASSERT(
               std::isnan(residualError) || 
               !std::isinf(residualError) && FloatEbmType { -1 } - k_epsilonResidualError <= residualError && residualError <= FloatEbmType { 1 }
            );

            // TODO : this code gets executed for each SamplingWithReplacement set.  I could probably execute it once and then all the SamplingWithReplacement
            //   sets would have this value, but I would need to store the computation in a new memory place, and it might make more sense to calculate this 
            //   values in the CPU rather than put more pressure on memory.  I think controlling this should be done in a MACRO and we should use a class to 
            //   hold the residualError and this computation from that value and then comment out the computation if not necssary and access it through an 
            //   accessor so that we can make the change entirely via macro
            const FloatEbmType denominator = EbmStatistics::ComputeNewtonRaphsonStep(residualError);
            EBM_ASSERT(
               std::isnan(denominator) || 
               !std::isinf(denominator) && -k_epsilonResidualError <= denominator && denominator <= FloatEbmType { 0.25 }
            ); // since any one denominatory is limited to -1 <= denominator <= 1, the sum must be representable by a 64 bit number, 

            const FloatEbmType oldDenominator = ARRAY_TO_POINTER(pHistogramBucketEntry->m_aHistogramBucketVectorEntry)[iVector].GetSumDenominator();
            // since any one denominatory is limited to -1 <= denominator <= 1, the sum must be representable by a 64 bit number, 
            EBM_ASSERT(std::isnan(oldDenominator) || !std::isinf(oldDenominator) && -k_epsilonResidualError <= oldDenominator);
            const FloatEbmType newDenominator = oldDenominator + denominator;
            // since any one denominatory is limited to -1 <= denominator <= 1, the sum must be representable by a 64 bit number, 
            EBM_ASSERT(std::isnan(newDenominator) || !std::isinf(newDenominator) && -k_epsilonResidualError <= newDenominator);
            // which will always be representable by a float or double, so we can't overflow to inifinity or -infinity
            ARRAY_TO_POINTER(pHistogramBucketEntry->m_aHistogramBucketVectorEntry)[iVector].SetSumDenominator(newDenominator);
         }
         ++pResidualError;
      }
   }
   LOG_0(TraceLevelVerbose, "Exited BinDataSetInteraction");
}

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
size_t SumHistogramBuckets(
   const SamplingMethod * const pTrainingSet, 
   const size_t cHistogramBuckets, 
   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, 
   HistogramBucketVectorEntry<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aSumHistogramBucketVectorEntry, 
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   constexpr bool bClassification = IsClassification(compilerLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelVerbose, "Entered SumHistogramBuckets");

   EBM_ASSERT(2 <= cHistogramBuckets); // we pre-filter out features with only one bucket

#ifndef NDEBUG
   size_t cInstancesTotalDebug = 0;
#endif // NDEBUG

   const ptrdiff_t learningTypeOrCountTargetClasses = GET_LEARNING_TYPE_OR_COUNT_TARGET_CLASSES(
      compilerLearningTypeOrCountTargetClasses,
      runtimeLearningTypeOrCountTargetClasses
   );
   const size_t cVectorLength = GetVectorLength(learningTypeOrCountTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bClassification>(cVectorLength);

   HistogramBucket<bClassification> * pCopyFrom = aHistogramBuckets;
   HistogramBucket<bClassification> * pCopyFromEnd =
      GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, aHistogramBuckets, cHistogramBuckets);

   // we do a lot more work in the GrowDecisionTree function per binned bucket entry, so if we can compress it by any amount, then it will probably be a win
   // for binned bucket arrays that have a small set of labels, this loop will be fast and result in no movements.  For binned bucket arrays that are long 
   // and have many different labels, we are more likley to find bins with zero items, and that's where we get a win by compressing it down to just the 
   // non-zero binned buckets, even though this requires one more member variable in the binned bucket array
   do {
      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pCopyFrom, aHistogramBucketsEndDebug);
#ifndef NDEBUG
      cInstancesTotalDebug += pCopyFrom->m_cInstancesInBucket;
#endif // NDEBUG
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         // when building a tree, we start from one end and sweep to the other.  In order to caluculate
         // gain on both sides, we need the sum on both sides, which means when starting from one end
         // we need to know the sum of everything on the other side, so we need to calculate this sum
         // somewhere.  If we have a continuous value and bin it such that many instances are in the same bin
         // then it makes sense to calculate the total of all bins after generating the histograms of the bins
         // since then we just need to sum N bins (where N is the number of bins) vs the # of instances.
         // There is one case though where we might want to calculate the sum while looping the instances,
         // and that is if almost all bins have either 0 or 1 instances, which would happen if we didn't bin at all
         // beforehand.  We'll still want this per-bin sumation though since it's unlikley that all data
         // will be continuous in an ML problem.
         aSumHistogramBucketVectorEntry[iVector].Add(ARRAY_TO_POINTER(pCopyFrom->m_aHistogramBucketVectorEntry)[iVector]);
      }

      pCopyFrom = GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pCopyFrom, 1);
   } while(pCopyFromEnd != pCopyFrom);
   EBM_ASSERT(0 == (reinterpret_cast<char *>(pCopyFrom) - reinterpret_cast<char *>(aHistogramBuckets)) % cBytesPerHistogramBucket);

   const size_t cInstancesTotal = pTrainingSet->GetTotalCountInstanceOccurrences();
   EBM_ASSERT(cInstancesTotal == cInstancesTotalDebug);

   LOG_0(TraceLevelVerbose, "Exited SumHistogramBuckets");
   return cInstancesTotal;
}

#endif // HISTOGRAM_BUCKET_H
