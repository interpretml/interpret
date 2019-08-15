// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef BINNED_BUCKET_H
#define BINNED_BUCKET_H

#include <type_traits> // std::is_pod
#include <string.h> // memset
#include <stddef.h> // size_t, ptrdiff_t
#include <cmath> // abs

#include "ebmcore.h" // FractionalDataType
#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "HistogramBucketVectorEntry.h"
#include "CachedThreadResources.h"
#include "FeatureCore.h"
#include "FeatureCombinationCore.h"
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

template<bool bRegression>
class HistogramBucket;

template<bool bRegression>
EBM_INLINE bool GetHistogramBucketSizeOverflow(const size_t cVectorLength) {
   return IsMultiplyError(sizeof(HistogramBucketVectorEntry<bRegression>), cVectorLength) ? true : IsAddError(sizeof(HistogramBucket<bRegression>) - sizeof(HistogramBucketVectorEntry<bRegression>), sizeof(HistogramBucketVectorEntry<bRegression>) * cVectorLength) ? true : false;
}
template<bool bRegression>
EBM_INLINE size_t GetHistogramBucketSize(const size_t cVectorLength) {
   return sizeof(HistogramBucket<bRegression>) - sizeof(HistogramBucketVectorEntry<bRegression>) + sizeof(HistogramBucketVectorEntry<bRegression>) * cVectorLength;
}
template<bool bRegression>
EBM_INLINE HistogramBucket<bRegression> * GetHistogramBucketByIndex(const size_t cBytesPerHistogramBucket, HistogramBucket<bRegression> * const aHistogramBuckets, const ptrdiff_t iBin) {
   // TODO : remove the use of this function anywhere performant by making the tensor calculation start with the # of bytes per histogram bucket, therefore eliminating the need to do the multiplication at the end when finding the index
   return reinterpret_cast<HistogramBucket<bRegression> *>(reinterpret_cast<char *>(aHistogramBuckets) + iBin * static_cast<ptrdiff_t>(cBytesPerHistogramBucket));
}
template<bool bRegression>
EBM_INLINE const HistogramBucket<bRegression> * GetHistogramBucketByIndex(const size_t cBytesPerHistogramBucket, const HistogramBucket<bRegression> * const aHistogramBuckets, const ptrdiff_t iBin) {
   // TODO : remove the use of this function anywhere performant by making the tensor calculation start with the # of bytes per histogram bucket, therefore eliminating the need to do the multiplication at the end when finding the index
   return reinterpret_cast<const HistogramBucket<bRegression> *>(reinterpret_cast<const char *>(aHistogramBuckets) + iBin * static_cast<ptrdiff_t>(cBytesPerHistogramBucket));
}

// keep this as a MACRO so that we don't materialize any of the parameters on non-debug builds
#define ASSERT_BINNED_BUCKET_OK(MACRO_cBytesPerHistogramBucket, MACRO_aHistogramBuckets, MACRO_aHistogramBucketsEnd) (EBM_ASSERT(reinterpret_cast<const char *>(MACRO_aHistogramBuckets) + static_cast<size_t>(MACRO_cBytesPerHistogramBucket) <= reinterpret_cast<const char *>(MACRO_aHistogramBucketsEnd)))

template<bool bRegression>
class HistogramBucket final {
public:

   size_t cInstancesInBucket;
   // TODO : we really want to eliminate this bucketValue at some point.  When doing the mains, if we change our algorithm so that we don't compress the arrays afterwards then we don't need it as the index is == to the bucketValue.
   // The compresson step is actually really unnessary because we can pre-compress our data when we get it to ensure that there are no missing bin values.  The only way there could be a bin with an instance count of zero then
   // would be if a value is not in a particular sampling set, which should be quite rare.
   // even if we ended up keeping the bucket value, it may make sense to first build a non-compressed representation which is more compact and can fit into cache better while we stripe in our main instance data and then re-organize it to add the bucket afterwards, which we know from each bucket's index
   // if we remove this bucketValue then it will slightly change our results because buckets where there are zeros are ambiguous in terms of choosing a split point.  We should remove this value as late as possible so that we preserve our comparison data sets over a longer
   // period of time
   // We don't use it in the pairs at all since we can't compress those.  Even if we chose not to change the algorithm
   ActiveDataType bucketValue;
   HistogramBucketVectorEntry<bRegression> aHistogramBucketVectorEntry[1];

   template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
   EBM_INLINE void Add(const HistogramBucket<bRegression> & other, const size_t cTargetClasses) {
      static_assert(IsRegression(compilerLearningTypeOrCountTargetClasses) == bRegression, "regression types must match");
      cInstancesInBucket += other.cInstancesInBucket;
      const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         aHistogramBucketVectorEntry[iVector].Add(other.aHistogramBucketVectorEntry[iVector]);
      }
   }
   template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
   EBM_INLINE void Subtract(const HistogramBucket<bRegression> & other, const size_t cTargetClasses) {
      static_assert(IsRegression(compilerLearningTypeOrCountTargetClasses) == bRegression, "regression types must match");
      cInstancesInBucket -= other.cInstancesInBucket;
      const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         aHistogramBucketVectorEntry[iVector].Subtract(other.aHistogramBucketVectorEntry[iVector]);
      }
   }
   template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
   EBM_INLINE void Copy(const HistogramBucket<bRegression> & other, const size_t cTargetClasses) {
      static_assert(IsRegression(compilerLearningTypeOrCountTargetClasses) == bRegression, "regression types must match");
      const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
      EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bRegression>(cVectorLength);
      memcpy(this, &other, cBytesPerHistogramBucket);
   }

   template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
   EBM_INLINE void Zero(const size_t cTargetClasses) {
      static_assert(IsRegression(compilerLearningTypeOrCountTargetClasses) == bRegression, "regression types must match");
      const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
      EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bRegression>(cVectorLength);
      memset(this, 0, cBytesPerHistogramBucket);
   }

   template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
   EBM_INLINE void AssertZero(const size_t cTargetClasses) const {
      UNUSED(cTargetClasses);
      static_assert(IsRegression(compilerLearningTypeOrCountTargetClasses) == bRegression, "regression types must match");
#ifndef NDEBUG
      const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
      EBM_ASSERT(0 == cInstancesInBucket);
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         aHistogramBucketVectorEntry[iVector].AssertZero();
      }
#endif // NDEBUG
   }

   static_assert(std::is_pod<ActiveDataType>::value, "HistogramBucket will be more efficient as a POD as we make potentially large arrays of them!");
};

static_assert(std::is_pod<HistogramBucket<false>>::value, "HistogramBucket will be more efficient as a POD as we make potentially large arrays of them!");
static_assert(std::is_pod<HistogramBucket<true>>::value, "HistogramBucket will be more efficient as a POD as we make potentially large arrays of them!");

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
void BinDataSetTrainingZeroDimensions(HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pHistogramBucketEntry, const SamplingMethod * const pTrainingSet, const size_t cTargetClasses) {
   LOG(TraceLevelVerbose, "Entered BinDataSetTrainingZeroDimensions");

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory

   const size_t cInstances = pTrainingSet->m_pOriginDataSet->GetCountInstances();
   EBM_ASSERT(0 < cInstances);

   const SamplingWithReplacement * const pSamplingWithReplacement = static_cast<const SamplingWithReplacement *>(pTrainingSet);
   const size_t * pCountOccurrences = pSamplingWithReplacement->m_aCountOccurrences;
   const FractionalDataType * pResidualError = pSamplingWithReplacement->m_pOriginDataSet->GetResidualPointer();
   // this shouldn't overflow since we're accessing existing memory
   const FractionalDataType * const pResidualErrorEnd = pResidualError + cVectorLength * cInstances;

   HistogramBucketVectorEntry<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pHistogramBucketVectorEntry = &pHistogramBucketEntry->aHistogramBucketVectorEntry[0];
   while(pResidualErrorEnd != pResidualError) {
      // this loop gets about twice as slow if you add a single unpredictable branching if statement based on count, even if you still access all the memory in complete sequential order, so we'll probably want to use non-branching instructions for any solution like conditional selection or multiplication
      // this loop gets about 3 times slower if you use a bad pseudo random number generator like rand(), although it might be better if you inlined rand().
      // this loop gets about 10 times slower if you use a proper pseudo random number generator like std::default_random_engine
      // taking all the above together, it seems unlikley we'll use a method of separating sets via single pass randomized set splitting.  Even if count is stored in memory if shouldn't increase the time spent fetching it by 2 times, unless our bottleneck when threading is overwhelmingly memory pressure related, and even then we could store the count for a single bit aleviating the memory pressure greatly, if we use the right sampling method 

      // TODO : try using a sampling method with non-repeating instances, and put the count into a bit.  Then unwind that loop either at the byte level (8 times) or the uint64_t level.  This can be done without branching and doesn't require random number generators

      const size_t cOccurences = *pCountOccurrences;
      ++pCountOccurrences;
      pHistogramBucketEntry->cInstancesInBucket += cOccurences;
      const FractionalDataType cFloatOccurences = static_cast<FractionalDataType>(cOccurences);

#ifndef NDEBUG
#ifdef EXPAND_BINARY_LOGITS
      constexpr bool bExpandBinaryLogits = true;
#else // EXPAND_BINARY_LOGITS
      constexpr bool bExpandBinaryLogits = false;
#endif // EXPAND_BINARY_LOGITS
      FractionalDataType residualTotalDebug = 0;
#endif // NDEBUG
      size_t iVector = 0;
      do {
         const FractionalDataType residualError = *pResidualError;
         EBM_ASSERT(!IsClassification(compilerLearningTypeOrCountTargetClasses) || 2 == cTargetClasses && !bExpandBinaryLogits || static_cast<ptrdiff_t>(iVector) != k_iZeroResidual || 0 == residualError);
#ifndef NDEBUG
         residualTotalDebug += residualError;
#endif // NDEBUG
         pHistogramBucketVectorEntry[iVector].sumResidualError += cFloatOccurences * residualError;
         if(IsClassification(compilerLearningTypeOrCountTargetClasses)) {
            // TODO : this code gets executed for each SamplingWithReplacement set.  I could probably execute it once and then all the SamplingWithReplacement sets would have this value, but I would need to store the computation in a new memory place, and it might make more sense to calculate this values in the CPU rather than put more pressure on memory.  I think controlling this should be done in a MACRO and we should use a class to hold the residualError and this computation from that value and then comment out the computation if not necssary and access it through an accessor so that we can make the change entirely via macro
            const FractionalDataType absResidualError = std::abs(residualError); // abs will return the same type that it is given, either float or double
            pHistogramBucketVectorEntry[iVector].SetSumDenominator(pHistogramBucketVectorEntry[iVector].GetSumDenominator() + cFloatOccurences * (absResidualError * (1 - absResidualError)));
         }
         ++pResidualError;
         ++iVector;
         // if we use this specific format where (iVector < cVectorLength) then the compiler collapses alway the loop for small cVectorLength values
         // if we make this (iVector != cVectorLength) then the loop is not collapsed
         // the compiler seems to not mind if we make this a for loop or do loop in terms of collapsing away the loop
      } while(iVector < cVectorLength);

      EBM_ASSERT(!IsClassification(compilerLearningTypeOrCountTargetClasses) || 2 == cTargetClasses && !bExpandBinaryLogits || 0 <= k_iZeroResidual || -0.00000000001 < residualTotalDebug && residualTotalDebug < 0.00000000001);
   }
   LOG(TraceLevelVerbose, "Exited BinDataSetTrainingZeroDimensions");
}

// TODO : remove cCompilerDimensions since we don't need it anymore, and replace it with a more useful number like the number of cItemsPerBitPackDataUnit
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t cCompilerDimensions>
void BinDataSetTraining(HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, const FeatureCombinationCore * const pFeatureCombination, const SamplingMethod * const pTrainingSet, const size_t cTargetClasses
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   LOG(TraceLevelVerbose, "Entered BinDataSetTraining");

   EBM_ASSERT(cCompilerDimensions == pFeatureCombination->m_cFeatures);
   static_assert(1 <= cCompilerDimensions, "cCompilerDimensions must be 1 or greater");

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
   const size_t cItemsPerBitPackDataUnit = pFeatureCombination->m_cItemsPerBitPackDataUnit;
   const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackDataUnit);
   const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);

   const size_t cInstances = pTrainingSet->m_pOriginDataSet->GetCountInstances();
   EBM_ASSERT(0 < cInstances);

   const SamplingWithReplacement * const pSamplingWithReplacement = static_cast<const SamplingWithReplacement *>(pTrainingSet);
   const size_t * pCountOccurrences = pSamplingWithReplacement->m_aCountOccurrences;
   const StorageDataTypeCore * pInputData = pSamplingWithReplacement->m_pOriginDataSet->GetDataPointer(pFeatureCombination);
   const FractionalDataType * pResidualError = pSamplingWithReplacement->m_pOriginDataSet->GetResidualPointer();
   // this shouldn't overflow since we're accessing existing memory
   const FractionalDataType * const pResidualErrorLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete = pResidualError + cVectorLength * (static_cast<ptrdiff_t>(cInstances) - cItemsPerBitPackDataUnit);

   size_t cItemsRemaining;

   while(pResidualError < pResidualErrorLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete) {
      // this loop gets about twice as slow if you add a single unpredictable branching if statement based on count, even if you still access all the memory in complete sequential order, so we'll probably want to use non-branching instructions for any solution like conditional selection or multiplication
      // this loop gets about 3 times slower if you use a bad pseudo random number generator like rand(), although it might be better if you inlined rand().
      // this loop gets about 10 times slower if you use a proper pseudo random number generator like std::default_random_engine
      // taking all the above together, it seems unlikley we'll use a method of separating sets via single pass randomized set splitting.  Even if count is stored in memory if shouldn't increase the time spent fetching it by 2 times, unless our bottleneck when threading is overwhelmingly memory pressure related, and even then we could store the count for a single bit aleviating the memory pressure greatly, if we use the right sampling method 

      // TODO : try using a sampling method with non-repeating instances, and put the count into a bit.  Then unwind that loop either at the byte level (8 times) or the uint64_t level.  This can be done without branching and doesn't require random number generators

      cItemsRemaining = cItemsPerBitPackDataUnit;
      // TODO : jumping back into this loop and changing cItemsRemaining to a dynamic value that isn't compile time determinable
      // causes this function to NOT be optimized as much as it could if we had two separate loops.  We're just trying this out for now though
   one_last_loop:;
      // we store the already multiplied dimensional value in *pInputData
      size_t iTensorBinCombined = static_cast<size_t>(*pInputData);
      ++pInputData;
      do {
         const size_t iTensorBin = maskBits & iTensorBinCombined;

         HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const pHistogramBucketEntry = GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBuckets, iTensorBin);

         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucketEntry, aHistogramBucketsEndDebug);
         const size_t cOccurences = *pCountOccurrences;
         ++pCountOccurrences;
         pHistogramBucketEntry->cInstancesInBucket += cOccurences;
         const FractionalDataType cFloatOccurences = static_cast<FractionalDataType>(cOccurences);
         HistogramBucketVectorEntry<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pHistogramBucketVectorEntry = &pHistogramBucketEntry->aHistogramBucketVectorEntry[0];
         size_t iVector = 0;

#ifndef NDEBUG
#ifdef EXPAND_BINARY_LOGITS
         constexpr bool bExpandBinaryLogits = true;
#else // EXPAND_BINARY_LOGITS
         constexpr bool bExpandBinaryLogits = false;
#endif // EXPAND_BINARY_LOGITS
         FractionalDataType residualTotalDebug = 0;
#endif // NDEBUG
         do {
            const FractionalDataType residualError = *pResidualError;
            EBM_ASSERT(!IsClassification(compilerLearningTypeOrCountTargetClasses) || 2 == cTargetClasses && !bExpandBinaryLogits || static_cast<ptrdiff_t>(iVector) != k_iZeroResidual || 0 == residualError);
#ifndef NDEBUG
            residualTotalDebug += residualError;
#endif // NDEBUG
            pHistogramBucketVectorEntry[iVector].sumResidualError += cFloatOccurences * residualError;
            if(IsClassification(compilerLearningTypeOrCountTargetClasses)) {
               // TODO : this code gets executed for each SamplingWithReplacement set.  I could probably execute it once and then all the SamplingWithReplacement sets would have this value, but I would need to store the computation in a new memory place, and it might make more sense to calculate this values in the CPU rather than put more pressure on memory.  I think controlling this should be done in a MACRO and we should use a class to hold the residualError and this computation from that value and then comment out the computation if not necssary and access it through an accessor so that we can make the change entirely via macro
               const FractionalDataType absResidualError = std::abs(residualError); // abs will return the same type that it is given, either float or double
               pHistogramBucketVectorEntry[iVector].SetSumDenominator(pHistogramBucketVectorEntry[iVector].GetSumDenominator() + cFloatOccurences * (absResidualError * (1 - absResidualError)));
            }
            ++pResidualError;
            ++iVector;
            // if we use this specific format where (iVector < cVectorLength) then the compiler collapses alway the loop for small cVectorLength values
            // if we make this (iVector != cVectorLength) then the loop is not collapsed
            // the compiler seems to not mind if we make this a for loop or do loop in terms of collapsing away the loop
         } while(iVector < cVectorLength);

         EBM_ASSERT(!IsClassification(compilerLearningTypeOrCountTargetClasses) || 2 == cTargetClasses && !bExpandBinaryLogits || 0 <= k_iZeroResidual || -0.0000001 < residualTotalDebug && residualTotalDebug < 0.0000001);

         iTensorBinCombined >>= cBitsPerItemMax;
         // TODO : try replacing cItemsRemaining with a pResidualErrorInnerLoopEnd which eliminates one subtact operation, but might make it harder for the compiler to optimize the loop away
         --cItemsRemaining;
      } while(0 != cItemsRemaining);
   }
   const FractionalDataType * const pResidualErrorEnd = pResidualErrorLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete + cVectorLength * cItemsPerBitPackDataUnit;
   if(pResidualError < pResidualErrorEnd) {
      LOG(TraceLevelVerbose, "Handling last BinDataSetTraining loop");

      // first time through?
      EBM_ASSERT(0 == (pResidualErrorEnd - pResidualError) % cVectorLength);
      cItemsRemaining = (pResidualErrorEnd - pResidualError) / cVectorLength;
      EBM_ASSERT(0 < cItemsRemaining);
      EBM_ASSERT(cItemsRemaining <= cItemsPerBitPackDataUnit);
      goto one_last_loop;
   }
   EBM_ASSERT(pResidualError == pResidualErrorEnd); // after our second iteration we should have finished everything!

   LOG(TraceLevelVerbose, "Exited BinDataSetTraining");
}

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses, size_t cCompilerDimensions>
class RecursiveBinDataSetTraining {
   // C++ does not allow partial function specialization, so we need to use these cumbersome inline static class functions to do partial function specialization
public:
   EBM_INLINE static void Recursive(const size_t cRuntimeDimensions, HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, const FeatureCombinationCore * const pFeatureCombination, const SamplingMethod * const pTrainingSet, const size_t cTargetClasses
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      EBM_ASSERT(cRuntimeDimensions < k_cDimensionsMax);
      static_assert(cCompilerDimensions < k_cDimensionsMax, "cCompilerDimensions must be less than or equal to k_cDimensionsMax.  This line only handles the less than part, but we handle the equals in a partial specialization template.");
      if(cCompilerDimensions == cRuntimeDimensions) {
         BinDataSetTraining<compilerLearningTypeOrCountTargetClasses, cCompilerDimensions>(aHistogramBuckets, pFeatureCombination, pTrainingSet, cTargetClasses
#ifndef NDEBUG
            , aHistogramBucketsEndDebug
#endif // NDEBUG
         );
      } else {
         RecursiveBinDataSetTraining<compilerLearningTypeOrCountTargetClasses, 1 + cCompilerDimensions>::Recursive(cRuntimeDimensions, aHistogramBuckets, pFeatureCombination, pTrainingSet, cTargetClasses
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
   EBM_INLINE static void Recursive(const size_t cRuntimeDimensions, HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, const FeatureCombinationCore * const pFeatureCombination, const SamplingMethod * const pTrainingSet, const size_t cTargetClasses
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      UNUSED(cRuntimeDimensions);
      EBM_ASSERT(k_cDimensionsMax == cRuntimeDimensions);
      BinDataSetTraining<compilerLearningTypeOrCountTargetClasses, k_cDimensionsMax>(aHistogramBuckets, pFeatureCombination, pTrainingSet, cTargetClasses
#ifndef NDEBUG
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );
   }
};

// TODO: make the number of dimensions (pFeatureCombination->m_cFeatures) a template parameter so that we don't have to have the inner loop that is very bad for performance.  Since the data will be stored contiguously and have the same length in the future, we can just loop based on the number of dimensions, so we might as well have a couple of different values
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
void BinDataSetInteraction(HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, const FeatureCombinationCore * const pFeatureCombination, const DataSetByFeature * const pDataSet, const size_t cTargetClasses
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   LOG(TraceLevelVerbose, "Entered BinDataSetInteraction");

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);

   const FractionalDataType * pResidualError = pDataSet->GetResidualPointer();
   const FractionalDataType * const pResidualErrorEnd = pResidualError + cVectorLength * pDataSet->GetCountInstances();

   size_t cFeatures = pFeatureCombination->m_cFeatures;
   EBM_ASSERT(1 <= cFeatures); // for interactions, we just return 0 for interactions with zero features
   for(size_t iInstance = 0; pResidualErrorEnd != pResidualError; ++iInstance) {
      // this loop gets about twice as slow if you add a single unpredictable branching if statement based on count, even if you still access all the memory in complete sequential order, so we'll probably want to use non-branching instructions for any solution like conditional selection or multiplication
      // this loop gets about 3 times slower if you use a bad pseudo random number generator like rand(), although it might be better if you inlined rand().
      // this loop gets about 10 times slower if you use a proper pseudo random number generator like std::default_random_engine
      // taking all the above together, it seems unlikley we'll use a method of separating sets via single pass randomized set splitting.  Even if count is stored in memory if shouldn't increase the time spent fetching it by 2 times, unless our bottleneck when threading is overwhelmingly memory pressure related, and even then we could store the count for a single bit aleviating the memory pressure greatly, if we use the right sampling method 

      // TODO : try using a sampling method with non-repeating instances, and put the count into a bit.  Then unwind that loop either at the byte level (8 times) or the uint64_t level.  This can be done without branching and doesn't require random number generators

      // TODO : we can elminate the inner vector loop for regression at least, and also if we add a templated bool for binary class.  Propegate this change to all places that we loop on the vector

      size_t cBuckets = 1;
      size_t iBucket = 0;
      size_t iDimension = 0;
      do {
         const FeatureCore * const pInputFeature = pFeatureCombination->m_FeatureCombinationEntry[iDimension].m_pFeature;
         const size_t cBins = pInputFeature->m_cBins;
         const StorageDataTypeCore * pInputData = pDataSet->GetDataPointer(pInputFeature);
         pInputData += iInstance;
         StorageDataTypeCore iBinOriginal = *pInputData;
         EBM_ASSERT((IsNumberConvertable<size_t, StorageDataTypeCore>(iBinOriginal)));
         size_t iBin = static_cast<size_t>(iBinOriginal);
         EBM_ASSERT(iBin < cBins);
         iBucket += cBuckets * iBin;
         cBuckets *= cBins;
         ++iDimension;
      } while(iDimension < cFeatures);
 
      HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pHistogramBucketEntry = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aHistogramBuckets, iBucket);
      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucketEntry, aHistogramBucketsEndDebug);
      pHistogramBucketEntry->cInstancesInBucket += 1;
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         pHistogramBucketEntry->aHistogramBucketVectorEntry[iVector].sumResidualError += *pResidualError;
         if(IsClassification(compilerLearningTypeOrCountTargetClasses)) {
            // TODO : this code gets executed for each SamplingWithReplacement set.  I could probably execute it once and then all the SamplingWithReplacement sets would have this value, but I would need to store the computation in a new memory place, and it might make more sense to calculate this values in the CPU rather than put more pressure on memory.  I think controlling this should be done in a MACRO and we should use a class to hold the residualError and this computation from that value and then comment out the computation if not necssary and access it through an accessor so that we can make the change entirely via macro
            FractionalDataType absResidualError = std::abs(*pResidualError); // abs will return the same type that it is given, either float or double
            pHistogramBucketEntry->aHistogramBucketVectorEntry[iVector].SetSumDenominator(pHistogramBucketEntry->aHistogramBucketVectorEntry[iVector].GetSumDenominator() + absResidualError * (1 - absResidualError));
         }
         ++pResidualError;
      }
   }
   LOG(TraceLevelVerbose, "Exited BinDataSetInteraction");
}

// TODO: change our downstream code to not need this Compression.  This compression often won't do anything because most of the time every bin will have data, and if there is sparse data with lots of values then maybe we don't want to do a complete sweep of this data moving it arround anyways.  We only do a minimial # of splits anyways.  I can calculate the sums in the loop that builds the bins instead of here!
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
size_t CompressHistogramBuckets(const SamplingMethod * const pTrainingSet, const size_t cHistogramBuckets, HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets, size_t * const pcInstancesTotal, HistogramBucketVectorEntry<IsRegression(compilerLearningTypeOrCountTargetClasses)> * const aSumHistogramBucketVectorEntry, const size_t cTargetClasses
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   LOG(TraceLevelVerbose, "Entered CompressHistogramBuckets");

   EBM_ASSERT(1 <= cHistogramBuckets); // this function can handle 1 == cBins even though that's a degenerate case that shouldn't be trained on (dimensions with 1 bin don't contribute anything since they always have the same value)

#ifndef NDEBUG
   size_t cInstancesTotalDebug = 0;
#endif // NDEBUG

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, cTargetClasses);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);

   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pCopyFrom = aHistogramBuckets;
   HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pCopyFromEnd = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aHistogramBuckets, cHistogramBuckets);

   // we do a lot more work in the GrowDecisionTree function per binned bucket entry, so if we can compress it by any amount, then it will probably be a win
   // for binned bucket arrays that have a small set of labels, this loop will be fast and result in no movements.  For binned bucket arrays that are long and have many different labels, 
   // we are more likley to find bins with zero items, and that's where we get a win by compressing it down to just the non-zero binned buckets, even though this
   // requires one more member variable in the binned bucket array
   ActiveDataType iBucket = 0;
   do {
      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pCopyFrom, aHistogramBucketsEndDebug);
      if(UNLIKELY(0 == pCopyFrom->cInstancesInBucket)) {
         HistogramBucket<IsRegression(compilerLearningTypeOrCountTargetClasses)> * pCopyTo = pCopyFrom;
         goto skip_first_check;
         do {
            ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pCopyFrom, aHistogramBucketsEndDebug);
            if(LIKELY(0 != pCopyFrom->cInstancesInBucket)) {
#ifndef NDEBUG
               cInstancesTotalDebug += pCopyFrom->cInstancesInBucket;
#endif // NDEBUG
               ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pCopyTo, aHistogramBucketsEndDebug);
               memcpy(pCopyTo, pCopyFrom, cBytesPerHistogramBucket);

               for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
                  aSumHistogramBucketVectorEntry[iVector].Add(pCopyFrom->aHistogramBucketVectorEntry[iVector]);
               }

               pCopyTo->bucketValue = static_cast<ActiveDataType>(iBucket);
               pCopyTo = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pCopyTo, 1);
            }
            skip_first_check:
            ++iBucket;
            pCopyFrom = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pCopyFrom, 1);
         } while(pCopyFromEnd != pCopyFrom);
         // TODO: eliminate this extra variable copy by making our outer loop use pCopyTo which is equal to pCopyFrom in the outer loop
         pCopyFrom = pCopyTo;
         break;
      }
#ifndef NDEBUG
      cInstancesTotalDebug += pCopyFrom->cInstancesInBucket;
#endif // NDEBUG
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         aSumHistogramBucketVectorEntry[iVector].Add(pCopyFrom->aHistogramBucketVectorEntry[iVector]);
      }

      pCopyFrom->bucketValue = static_cast<ActiveDataType>(iBucket);

      ++iBucket;
      pCopyFrom = GetHistogramBucketByIndex<IsRegression(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, pCopyFrom, 1);
   } while(pCopyFromEnd != pCopyFrom);
   EBM_ASSERT(0 == (reinterpret_cast<char *>(pCopyFrom) - reinterpret_cast<char *>(aHistogramBuckets)) % cBytesPerHistogramBucket);
   size_t cFinalItems = (reinterpret_cast<char *>(pCopyFrom) - reinterpret_cast<char *>(aHistogramBuckets)) / cBytesPerHistogramBucket;

   const size_t cInstancesTotal = pTrainingSet->GetTotalCountInstanceOccurrences();
   EBM_ASSERT(cInstancesTotal == cInstancesTotalDebug);

   *pcInstancesTotal = cInstancesTotal;

   LOG(TraceLevelVerbose, "Exited CompressHistogramBuckets");
   return cFinalItems;
}

#endif // BINNED_BUCKET_H
