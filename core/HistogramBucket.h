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
#include "Feature.h"
#include "FeatureCombination.h"
#include "DataSetByFeatureCombination.h"
#include "DataSetByFeature.h"
#include "SamplingWithReplacement.h"

// we don't need to handle multi-dimensional inputs with more than 64 bits total
// the rational is that we need to bin this data, and our binning memory will be N1*N1*...*N(D-1)*N(D)
// So, even for binary input featuers, we would have 2^64 bins, and that would take more memory than a 64 bit machine can have
// Similarily, we need a huge amount of memory in order to bin any data with a combined total of more than 64 bits.
// The worst case I can think of is where we have 3 states, requiring 2 bit each, and we overflowed at 33 dimensions
// in that bad case, we would have 3^33 bins * 8 bytes minimum per bin = 44472484532444184 bytes, which would take 56 bits to express
// Nobody is every going to build a machine with more than 64 bits, since you need a non-trivial volume of material assuming bits require
// more than several atoms to store.
// we can just return an out of memory error if someone requests a set of features that would sum to more than 64 bits
// we DO need to check for this error condition though, since it's not impossible for someone to request this kind of thing.
// any dimensions with only 1 state don't count since you would just be multiplying by 1 for each such dimension

template<bool bRegression>
class BinnedBucket;

template<bool bRegression>
EBM_INLINE bool GetBinnedBucketSizeOverflow(const size_t cVectorLength) {
   return IsMultiplyError(sizeof(PredictionStatistics<bRegression>), cVectorLength) ? true : IsAddError(sizeof(BinnedBucket<bRegression>) - sizeof(PredictionStatistics<bRegression>), sizeof(PredictionStatistics<bRegression>) * cVectorLength) ? true : false;
}
template<bool bRegression>
EBM_INLINE size_t GetBinnedBucketSize(const size_t cVectorLength) {
   return sizeof(BinnedBucket<bRegression>) - sizeof(PredictionStatistics<bRegression>) + sizeof(PredictionStatistics<bRegression>) * cVectorLength;
}
template<bool bRegression>
EBM_INLINE BinnedBucket<bRegression> * GetBinnedBucketByIndex(const size_t cBytesPerBinnedBucket, BinnedBucket<bRegression> * const aBinnedBuckets, const ptrdiff_t index) {
   return reinterpret_cast<BinnedBucket<bRegression> *>(reinterpret_cast<char *>(aBinnedBuckets) + index * static_cast<ptrdiff_t>(cBytesPerBinnedBucket));
}
template<bool bRegression>
EBM_INLINE const BinnedBucket<bRegression> * GetBinnedBucketByIndex(const size_t cBytesPerBinnedBucket, const BinnedBucket<bRegression> * const aBinnedBuckets, const ptrdiff_t index) {
   return reinterpret_cast<const BinnedBucket<bRegression> *>(reinterpret_cast<const char *>(aBinnedBuckets) + index * static_cast<ptrdiff_t>(cBytesPerBinnedBucket));
}

// keep this as a MACRO so that we don't materialize any of the parameters on non-debug builds
#define ASSERT_BINNED_BUCKET_OK(MACRO_cBytesPerBinnedBucket, MACRO_aBinnedBuckets, MACRO_aBinnedBucketsEnd) (EBM_ASSERT(reinterpret_cast<const char *>(MACRO_aBinnedBuckets) + static_cast<size_t>(MACRO_cBytesPerBinnedBucket) <= reinterpret_cast<const char *>(MACRO_aBinnedBucketsEnd)))

template<bool bRegression>
class BinnedBucket final {
public:

   size_t cCasesInBucket;
   // TODO : we really want to eliminate this bucketValue at some point.  When doing the mains, if we change our algorithm so that we don't compress the arrays afterwards then we don't need it as the index is == to the bucketValue.
   // The compresson step is actually really unnessary because we can pre-compress our data when we get it to ensure that there are no missing bin values.  The only way there could be a bin with a case count of zero then
   // would be if a value is not in a particular sampling set, which should be quite rare.
   // even if we ended up keeping the bucket value, it may make sense to first build a non-compressed representation which is more compact and can fit into cache better while we stripe in our main case data and then re-organize it to add the bucket afterwards, which we know from each bucket's index
   // if we remove this bucketValue then it will slightly change our results because cases where there are zeros are ambiguous in terms of choosing a split point.  We should remove this value as late as possible so that we preserve our comparison data sets over a longer
   // period of time
   // We don't use it in the pairs at all since we can't compress those.  Even if we chose not to change the algorithm
   ActiveDataType bucketValue;
   PredictionStatistics<bRegression> aPredictionStatistics[1];

   template<ptrdiff_t countCompilerClassificationTargetStates>
   EBM_INLINE void Add(const BinnedBucket<bRegression> & other, const size_t cTargetStates) {
      static_assert(IsRegression(countCompilerClassificationTargetStates) == bRegression, "regression types must match");
      cCasesInBucket += other.cCasesInBucket;
      const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         aPredictionStatistics[iVector].Add(other.aPredictionStatistics[iVector]);
      }
   }
   template<ptrdiff_t countCompilerClassificationTargetStates>
   EBM_INLINE void Subtract(const BinnedBucket<bRegression> & other, const size_t cTargetStates) {
      static_assert(IsRegression(countCompilerClassificationTargetStates) == bRegression, "regression types must match");
      cCasesInBucket -= other.cCasesInBucket;
      const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         aPredictionStatistics[iVector].Subtract(other.aPredictionStatistics[iVector]);
      }
   }
   template<ptrdiff_t countCompilerClassificationTargetStates>
   EBM_INLINE void Copy(const BinnedBucket<bRegression> & other, const size_t cTargetStates) {
      static_assert(IsRegression(countCompilerClassificationTargetStates) == bRegression, "regression types must match");
      const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
      EBM_ASSERT(!GetBinnedBucketSizeOverflow<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerBinnedBucket = GetBinnedBucketSize<bRegression>(cVectorLength);
      memcpy(this, &other, cBytesPerBinnedBucket);
   }

   template<ptrdiff_t countCompilerClassificationTargetStates>
   EBM_INLINE void Zero(const size_t cTargetStates) {
      static_assert(IsRegression(countCompilerClassificationTargetStates) == bRegression, "regression types must match");
      const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
      EBM_ASSERT(!GetBinnedBucketSizeOverflow<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerBinnedBucket = GetBinnedBucketSize<bRegression>(cVectorLength);
      memset(this, 0, cBytesPerBinnedBucket);
   }

   template<ptrdiff_t countCompilerClassificationTargetStates>
   EBM_INLINE void AssertZero(const size_t cTargetStates) const {
      UNUSED(cTargetStates);
      static_assert(IsRegression(countCompilerClassificationTargetStates) == bRegression, "regression types must match");
#ifndef NDEBUG
      const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
      EBM_ASSERT(0 == cCasesInBucket);
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         aPredictionStatistics[iVector].AssertZero();
      }
#endif // NDEBUG
   }

   static_assert(std::is_pod<ActiveDataType>::value, "BinnedBucket will be more efficient as a POD as we make potentially large arrays of them!");
};

static_assert(std::is_pod<BinnedBucket<false>>::value, "BinnedBucket will be more efficient as a POD as we make potentially large arrays of them!");
static_assert(std::is_pod<BinnedBucket<true>>::value, "BinnedBucket will be more efficient as a POD as we make potentially large arrays of them!");

template<ptrdiff_t countCompilerClassificationTargetStates>
void BinDataSetTrainingZeroDimensions(BinnedBucket<IsRegression(countCompilerClassificationTargetStates)> * const pBinnedBucketEntry, const SamplingMethod * const pTrainingSet, const size_t cTargetStates) {
   LOG(TraceLevelVerbose, "Entered BinDataSetTrainingZeroDimensions");

   const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
   EBM_ASSERT(!GetBinnedBucketSizeOverflow<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength)); // we're accessing allocated memory

   const size_t cCases = pTrainingSet->m_pOriginDataSet->GetCountCases();
   EBM_ASSERT(0 < cCases);

   const SamplingWithReplacement * const pSamplingWithReplacement = static_cast<const SamplingWithReplacement *>(pTrainingSet);
   const size_t * pCountOccurrences = pSamplingWithReplacement->m_aCountOccurrences;
   const FractionalDataType * pResidualError = pSamplingWithReplacement->m_pOriginDataSet->GetResidualPointer();
   // this shouldn't overflow since we're accessing existing memory
   const FractionalDataType * const pResidualErrorEnd = pResidualError + cVectorLength * cCases;

   PredictionStatistics<IsRegression(countCompilerClassificationTargetStates)> * const pPredictionStatistics = &pBinnedBucketEntry->aPredictionStatistics[0];
   while(pResidualErrorEnd != pResidualError) {
      // this loop gets about twice as slow if you add a single unpredictable branching if statement based on count, even if you still access all the memory in complete sequential order, so we'll probably want to use non-branching instructions for any solution like conditional selection or multiplication
      // this loop gets about 3 times slower if you use a bad pseudo random number generator like rand(), although it might be better if you inlined rand().
      // this loop gets about 10 times slower if you use a proper pseudo random number generator like std::default_random_engine
      // taking all the above together, it seems unlikley we'll use a method of separating sets via single pass randomized set splitting.  Even if count is stored in memory if shouldn't increase the time spent fetching it by 2 times, unless our bottleneck when threading is overwhelmingly memory pressure related, and even then we could store the count for a single bit aleviating the memory pressure greatly, if we use the right sampling method 

      // TODO : try using a sampling method with non-repeating cases, and put the count into a bit.  Then unwind that loop either at the byte level (8 times) or the uint64_t level.  This can be done without branching and doesn't require random number generators

      const size_t cOccurences = *pCountOccurrences;
      ++pCountOccurrences;
      pBinnedBucketEntry->cCasesInBucket += cOccurences;
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
         EBM_ASSERT(!IsClassification(countCompilerClassificationTargetStates) || 2 == cTargetStates && !bExpandBinaryLogits || static_cast<ptrdiff_t>(iVector) != k_iZeroResidual || 0 == residualError);
#ifndef NDEBUG
         residualTotalDebug += residualError;
#endif // NDEBUG
         pPredictionStatistics[iVector].sumResidualError += cFloatOccurences * residualError;
         if(IsClassification(countCompilerClassificationTargetStates)) {
            // TODO : this code gets executed for each SamplingWithReplacement set.  I could probably execute it once and then all the SamplingWithReplacement sets would have this value, but I would need to store the computation in a new memory place, and it might make more sense to calculate this values in the CPU rather than put more pressure on memory.  I think controlling this should be done in a MACRO and we should use a class to hold the residualError and this computation from that value and then comment out the computation if not necssary and access it through an accessor so that we can make the change entirely via macro
            const FractionalDataType absResidualError = std::abs(residualError); // abs will return the same type that it is given, either float or double
            pPredictionStatistics[iVector].SetSumDenominator(pPredictionStatistics[iVector].GetSumDenominator() + cFloatOccurences * (absResidualError * (1 - absResidualError)));
         }
         ++pResidualError;
         ++iVector;
         // if we use this specific format where (iVector < cVectorLength) then the compiler collapses alway the loop for small cVectorLength values
         // if we make this (iVector != cVectorLength) then the loop is not collapsed
         // the compiler seems to not mind if we make this a for loop or do loop in terms of collapsing away the loop
      } while(iVector < cVectorLength);

      EBM_ASSERT(!IsClassification(countCompilerClassificationTargetStates) || 2 == cTargetStates && !bExpandBinaryLogits || 0 <= k_iZeroResidual || -0.00000000001 < residualTotalDebug && residualTotalDebug < 0.00000000001);
   }
   LOG(TraceLevelVerbose, "Exited BinDataSetTrainingZeroDimensions");
}

// TODO : remove cCompilerDimensions since we don't need it anymore, and replace it with a more useful number like the number of cItemsPerBitPackDataUnit
template<ptrdiff_t countCompilerClassificationTargetStates, size_t cCompilerDimensions>
void BinDataSetTraining(BinnedBucket<IsRegression(countCompilerClassificationTargetStates)> * const aBinnedBuckets, const FeatureCombination * const pFeatureCombination, const SamplingMethod * const pTrainingSet, const size_t cTargetStates
#ifndef NDEBUG
   , const unsigned char * const aBinnedBucketsEndDebug
#endif // NDEBUG
) {
   LOG(TraceLevelVerbose, "Entered BinDataSetTraining");

   EBM_ASSERT(cCompilerDimensions == pFeatureCombination->m_cFeatures);
   static_assert(1 <= cCompilerDimensions, "cCompilerDimensions must be 1 or greater");

   const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
   const size_t cItemsPerBitPackDataUnit = pFeatureCombination->m_cItemsPerBitPackDataUnit;
   const size_t cBitsPerItemMax = GetCountBits(cItemsPerBitPackDataUnit);
   const size_t maskBits = std::numeric_limits<size_t>::max() >> (k_cBitsForStorageType - cBitsPerItemMax);
   EBM_ASSERT(!GetBinnedBucketSizeOverflow<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerBinnedBucket = GetBinnedBucketSize<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength);

   const size_t cCases = pTrainingSet->m_pOriginDataSet->GetCountCases();
   EBM_ASSERT(0 < cCases);

   const SamplingWithReplacement * const pSamplingWithReplacement = static_cast<const SamplingWithReplacement *>(pTrainingSet);
   const size_t * pCountOccurrences = pSamplingWithReplacement->m_aCountOccurrences;
   const StorageDataTypeCore * pInputData = pSamplingWithReplacement->m_pOriginDataSet->GetDataPointer(pFeatureCombination);
   const FractionalDataType * pResidualError = pSamplingWithReplacement->m_pOriginDataSet->GetResidualPointer();
   // this shouldn't overflow since we're accessing existing memory
   const FractionalDataType * const pResidualErrorLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete = pResidualError + cVectorLength * (static_cast<ptrdiff_t>(cCases) - cItemsPerBitPackDataUnit);

   size_t cItemsRemaining;

   while(pResidualError < pResidualErrorLastItemWhereNextLoopCouldDoFullLoopOrLessAndComplete) {
      // this loop gets about twice as slow if you add a single unpredictable branching if statement based on count, even if you still access all the memory in complete sequential order, so we'll probably want to use non-branching instructions for any solution like conditional selection or multiplication
      // this loop gets about 3 times slower if you use a bad pseudo random number generator like rand(), although it might be better if you inlined rand().
      // this loop gets about 10 times slower if you use a proper pseudo random number generator like std::default_random_engine
      // taking all the above together, it seems unlikley we'll use a method of separating sets via single pass randomized set splitting.  Even if count is stored in memory if shouldn't increase the time spent fetching it by 2 times, unless our bottleneck when threading is overwhelmingly memory pressure related, and even then we could store the count for a single bit aleviating the memory pressure greatly, if we use the right sampling method 

      // TODO : try using a sampling method with non-repeating cases, and put the count into a bit.  Then unwind that loop either at the byte level (8 times) or the uint64_t level.  This can be done without branching and doesn't require random number generators

      cItemsRemaining = cItemsPerBitPackDataUnit;
      // TODO : jumping back into this loop and changing cItemsRemaining to a dynamic value that isn't compile time determinable
      // causes this function to NOT be optimized as much as it could if we had two separate loops.  We're just trying this out for now though
   one_last_loop:;
      // we store the already multiplied dimensional value in *pInputData
      size_t iBinCombined = static_cast<size_t>(*pInputData);
      ++pInputData;
      do {
         const size_t iBin = maskBits & iBinCombined;

         BinnedBucket<IsRegression(countCompilerClassificationTargetStates)> * const pBinnedBucketEntry = GetBinnedBucketByIndex(cBytesPerBinnedBucket, aBinnedBuckets, iBin);

         ASSERT_BINNED_BUCKET_OK(cBytesPerBinnedBucket, pBinnedBucketEntry, aBinnedBucketsEndDebug);
         const size_t cOccurences = *pCountOccurrences;
         ++pCountOccurrences;
         pBinnedBucketEntry->cCasesInBucket += cOccurences;
         const FractionalDataType cFloatOccurences = static_cast<FractionalDataType>(cOccurences);
         PredictionStatistics<IsRegression(countCompilerClassificationTargetStates)> * pPredictionStatistics = &pBinnedBucketEntry->aPredictionStatistics[0];
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
            EBM_ASSERT(!IsClassification(countCompilerClassificationTargetStates) || 2 == cTargetStates && !bExpandBinaryLogits || static_cast<ptrdiff_t>(iVector) != k_iZeroResidual || 0 == residualError);
#ifndef NDEBUG
            residualTotalDebug += residualError;
#endif // NDEBUG
            pPredictionStatistics[iVector].sumResidualError += cFloatOccurences * residualError;
            if(IsClassification(countCompilerClassificationTargetStates)) {
               // TODO : this code gets executed for each SamplingWithReplacement set.  I could probably execute it once and then all the SamplingWithReplacement sets would have this value, but I would need to store the computation in a new memory place, and it might make more sense to calculate this values in the CPU rather than put more pressure on memory.  I think controlling this should be done in a MACRO and we should use a class to hold the residualError and this computation from that value and then comment out the computation if not necssary and access it through an accessor so that we can make the change entirely via macro
               const FractionalDataType absResidualError = std::abs(residualError); // abs will return the same type that it is given, either float or double
               pPredictionStatistics[iVector].SetSumDenominator(pPredictionStatistics[iVector].GetSumDenominator() + cFloatOccurences * (absResidualError * (1 - absResidualError)));
            }
            ++pResidualError;
            ++iVector;
            // if we use this specific format where (iVector < cVectorLength) then the compiler collapses alway the loop for small cVectorLength values
            // if we make this (iVector != cVectorLength) then the loop is not collapsed
            // the compiler seems to not mind if we make this a for loop or do loop in terms of collapsing away the loop
         } while(iVector < cVectorLength);

         EBM_ASSERT(!IsClassification(countCompilerClassificationTargetStates) || 2 == cTargetStates && !bExpandBinaryLogits || 0 <= k_iZeroResidual || -0.0000001 < residualTotalDebug && residualTotalDebug < 0.0000001);

         iBinCombined >>= cBitsPerItemMax;
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

template<ptrdiff_t countCompilerClassificationTargetStates, size_t cCompilerDimensions>
class RecursiveBinDataSetTraining {
   // C++ does not allow partial function specialization, so we need to use these cumbersome inline static class functions to do partial function specialization
public:
   EBM_INLINE static void Recursive(const size_t cRuntimeDimensions, BinnedBucket<IsRegression(countCompilerClassificationTargetStates)> * const aBinnedBuckets, const FeatureCombination * const pFeatureCombination, const SamplingMethod * const pTrainingSet, const size_t cTargetStates
#ifndef NDEBUG
      , const unsigned char * const aBinnedBucketsEndDebug
#endif // NDEBUG
   ) {
      EBM_ASSERT(cRuntimeDimensions < k_cDimensionsMax);
      static_assert(cCompilerDimensions < k_cDimensionsMax, "cCompilerDimensions must be less than or equal to k_cDimensionsMax.  This line only handles the less than part, but we handle the equals in a partial specialization template.");
      if(cCompilerDimensions == cRuntimeDimensions) {
         BinDataSetTraining<countCompilerClassificationTargetStates, cCompilerDimensions>(aBinnedBuckets, pFeatureCombination, pTrainingSet, cTargetStates
#ifndef NDEBUG
            , aBinnedBucketsEndDebug
#endif // NDEBUG
         );
      } else {
         RecursiveBinDataSetTraining<countCompilerClassificationTargetStates, 1 + cCompilerDimensions>::Recursive(cRuntimeDimensions, aBinnedBuckets, pFeatureCombination, pTrainingSet, cTargetStates
#ifndef NDEBUG
            , aBinnedBucketsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<ptrdiff_t countCompilerClassificationTargetStates>
class RecursiveBinDataSetTraining<countCompilerClassificationTargetStates, k_cDimensionsMax> {
   // C++ does not allow partial function specialization, so we need to use these cumbersome inline static class functions to do partial function specialization
public:
   EBM_INLINE static void Recursive(const size_t cRuntimeDimensions, BinnedBucket<IsRegression(countCompilerClassificationTargetStates)> * const aBinnedBuckets, const FeatureCombination * const pFeatureCombination, const SamplingMethod * const pTrainingSet, const size_t cTargetStates
#ifndef NDEBUG
      , const unsigned char * const aBinnedBucketsEndDebug
#endif // NDEBUG
   ) {
      UNUSED(cRuntimeDimensions);
      EBM_ASSERT(k_cDimensionsMax == cRuntimeDimensions);
      BinDataSetTraining<countCompilerClassificationTargetStates, k_cDimensionsMax>(aBinnedBuckets, pFeatureCombination, pTrainingSet, cTargetStates
#ifndef NDEBUG
         , aBinnedBucketsEndDebug
#endif // NDEBUG
      );
   }
};

// TODO: make the number of dimensions (pFeatureCombination->m_cFeatures) a template parameter so that we don't have to have the inner loop that is very bad for performance.  Since the data will be stored contiguously and have the same length in the future, we can just loop based on the number of dimensions, so we might as well have a couple of different values
template<ptrdiff_t countCompilerClassificationTargetStates>
void BinDataSetInteraction(BinnedBucket<IsRegression(countCompilerClassificationTargetStates)> * const aBinnedBuckets, const FeatureCombination * const pFeatureCombination, const DataSetByFeature * const pDataSet, const size_t cTargetStates
#ifndef NDEBUG
   , const unsigned char * const aBinnedBucketsEndDebug
#endif // NDEBUG
) {
   LOG(TraceLevelVerbose, "Entered BinDataSetInteraction");

   const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
   EBM_ASSERT(!GetBinnedBucketSizeOverflow<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerBinnedBucket = GetBinnedBucketSize<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength);

   const FractionalDataType * pResidualError = pDataSet->GetResidualPointer();
   const FractionalDataType * const pResidualErrorEnd = pResidualError + cVectorLength * pDataSet->GetCountCases();

   size_t cFeatures = pFeatureCombination->m_cFeatures;
   EBM_ASSERT(1 <= cFeatures); // for interactions, we just return 0 for interactions with zero features
   for(size_t iCase = 0; pResidualErrorEnd != pResidualError; ++iCase) {
      // this loop gets about twice as slow if you add a single unpredictable branching if statement based on count, even if you still access all the memory in complete sequential order, so we'll probably want to use non-branching instructions for any solution like conditional selection or multiplication
      // this loop gets about 3 times slower if you use a bad pseudo random number generator like rand(), although it might be better if you inlined rand().
      // this loop gets about 10 times slower if you use a proper pseudo random number generator like std::default_random_engine
      // taking all the above together, it seems unlikley we'll use a method of separating sets via single pass randomized set splitting.  Even if count is stored in memory if shouldn't increase the time spent fetching it by 2 times, unless our bottleneck when threading is overwhelmingly memory pressure related, and even then we could store the count for a single bit aleviating the memory pressure greatly, if we use the right sampling method 

      // TODO : try using a sampling method with non-repeating cases, and put the count into a bit.  Then unwind that loop either at the byte level (8 times) or the uint64_t level.  This can be done without branching and doesn't require random number generators

      // TODO : we can elminate the inner vector loop for regression at least, and also if we add a templated bool for binary class.  Propegate this change to all places that we loop on the vector

      size_t cBuckets = 1;
      size_t iBucket = 0;
      size_t iDimension = 0;
      do {
         const Feature * const pInputFeature = pFeatureCombination->m_FeatureCombinationEntry[iDimension].m_pFeature;
         const size_t cStates = pInputFeature->m_cStates;
         const StorageDataTypeCore * pInputData = pDataSet->GetDataPointer(pInputFeature);
         pInputData += iCase;
         StorageDataTypeCore data = *pInputData;
         EBM_ASSERT((IsNumberConvertable<size_t, StorageDataTypeCore>(data)));
         size_t iState = static_cast<size_t>(data);
         EBM_ASSERT(iState < cStates);
         iBucket += cBuckets * iState;
         cBuckets *= cStates;
         ++iDimension;
      } while(iDimension < cFeatures);
 
      BinnedBucket<IsRegression(countCompilerClassificationTargetStates)> * pBinnedBucketEntry = GetBinnedBucketByIndex<IsRegression(countCompilerClassificationTargetStates)>(cBytesPerBinnedBucket, aBinnedBuckets, iBucket);
      ASSERT_BINNED_BUCKET_OK(cBytesPerBinnedBucket, pBinnedBucketEntry, aBinnedBucketsEndDebug);
      pBinnedBucketEntry->cCasesInBucket += 1;
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         pBinnedBucketEntry->aPredictionStatistics[iVector].sumResidualError += *pResidualError;
         if(IsClassification(countCompilerClassificationTargetStates)) {
            // TODO : this code gets executed for each SamplingWithReplacement set.  I could probably execute it once and then all the SamplingWithReplacement sets would have this value, but I would need to store the computation in a new memory place, and it might make more sense to calculate this values in the CPU rather than put more pressure on memory.  I think controlling this should be done in a MACRO and we should use a class to hold the residualError and this computation from that value and then comment out the computation if not necssary and access it through an accessor so that we can make the change entirely via macro
            FractionalDataType absResidualError = std::abs(*pResidualError); // abs will return the same type that it is given, either float or double
            pBinnedBucketEntry->aPredictionStatistics[iVector].SetSumDenominator(pBinnedBucketEntry->aPredictionStatistics[iVector].GetSumDenominator() + absResidualError * (1 - absResidualError));
         }
         ++pResidualError;
      }
   }
   LOG(TraceLevelVerbose, "Exited BinDataSetInteraction");
}

// TODO: change our downstream code to not need this Compression.  This compression often won't do anything because most of the time every bin will have data, and if there is sparse data with lots of values then maybe we don't want to do a complete sweep of this data moving it arround anyways.  We only do a minimial # of splits anyways.  I can calculate the sums in the loop that builds the bins instead of here!
template<ptrdiff_t countCompilerClassificationTargetStates>
size_t CompressBinnedBuckets(const SamplingMethod * const pTrainingSet, const size_t cBinnedBuckets, BinnedBucket<IsRegression(countCompilerClassificationTargetStates)> * const aBinnedBuckets, size_t * const pcCasesTotal, PredictionStatistics<IsRegression(countCompilerClassificationTargetStates)> * const aSumPredictionStatistics, const size_t cTargetStates
#ifndef NDEBUG
   , const unsigned char * const aBinnedBucketsEndDebug
#endif // NDEBUG
) {
   LOG(TraceLevelVerbose, "Entered CompressBinnedBuckets");

   EBM_ASSERT(1 <= cBinnedBuckets); // this function can handle 1 == cStates even though that's a degenerate case that shouldn't be trained on (dimensions with 1 state don't contribute anything since they always have the same value)

#ifndef NDEBUG
   size_t cCasesTotalDebug = 0;
#endif // NDEBUG

   const size_t cVectorLength = GET_VECTOR_LENGTH(countCompilerClassificationTargetStates, cTargetStates);
   EBM_ASSERT(!GetBinnedBucketSizeOverflow<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerBinnedBucket = GetBinnedBucketSize<IsRegression(countCompilerClassificationTargetStates)>(cVectorLength);

   BinnedBucket<IsRegression(countCompilerClassificationTargetStates)> * pCopyFrom = aBinnedBuckets;
   BinnedBucket<IsRegression(countCompilerClassificationTargetStates)> * pCopyFromEnd = GetBinnedBucketByIndex<IsRegression(countCompilerClassificationTargetStates)>(cBytesPerBinnedBucket, aBinnedBuckets, cBinnedBuckets);

   // we do a lot more work in the GrowDecisionTree function per binned bucket entry, so if we can compress it by any amount, then it will probably be a win
   // for binned bucket arrays that have a small set of labels, this loop will be fast and result in no movements.  For binned bucket arrays that are long and have many different labels, 
   // we are more likley to find bins with zero items, and that's where we get a win by compressing it down to just the non-zero binned buckets, even though this
   // requires one more member variable in the binned bucket array
   ActiveDataType iBucket = 0;
   do {
      ASSERT_BINNED_BUCKET_OK(cBytesPerBinnedBucket, pCopyFrom, aBinnedBucketsEndDebug);
      if(UNLIKELY(0 == pCopyFrom->cCasesInBucket)) {
         BinnedBucket<IsRegression(countCompilerClassificationTargetStates)> * pCopyTo = pCopyFrom;
         goto skip_first_check;
         do {
            ASSERT_BINNED_BUCKET_OK(cBytesPerBinnedBucket, pCopyFrom, aBinnedBucketsEndDebug);
            if(LIKELY(0 != pCopyFrom->cCasesInBucket)) {
#ifndef NDEBUG
               cCasesTotalDebug += pCopyFrom->cCasesInBucket;
#endif // NDEBUG
               ASSERT_BINNED_BUCKET_OK(cBytesPerBinnedBucket, pCopyTo, aBinnedBucketsEndDebug);
               memcpy(pCopyTo, pCopyFrom, cBytesPerBinnedBucket);

               for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
                  aSumPredictionStatistics[iVector].Add(pCopyFrom->aPredictionStatistics[iVector]);
               }

               pCopyTo->bucketValue = static_cast<ActiveDataType>(iBucket);
               pCopyTo = GetBinnedBucketByIndex<IsRegression(countCompilerClassificationTargetStates)>(cBytesPerBinnedBucket, pCopyTo, 1);
            }
            skip_first_check:
            ++iBucket;
            pCopyFrom = GetBinnedBucketByIndex<IsRegression(countCompilerClassificationTargetStates)>(cBytesPerBinnedBucket, pCopyFrom, 1);
         } while(pCopyFromEnd != pCopyFrom);
         // TODO: eliminate this extra variable copy by making our outer loop use pCopyTo which is equal to pCopyFrom in the outer loop
         pCopyFrom = pCopyTo;
         break;
      }
#ifndef NDEBUG
      cCasesTotalDebug += pCopyFrom->cCasesInBucket;
#endif // NDEBUG
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         aSumPredictionStatistics[iVector].Add(pCopyFrom->aPredictionStatistics[iVector]);
      }

      pCopyFrom->bucketValue = static_cast<ActiveDataType>(iBucket);

      ++iBucket;
      pCopyFrom = GetBinnedBucketByIndex<IsRegression(countCompilerClassificationTargetStates)>(cBytesPerBinnedBucket, pCopyFrom, 1);
   } while(pCopyFromEnd != pCopyFrom);
   EBM_ASSERT(0 == (reinterpret_cast<char *>(pCopyFrom) - reinterpret_cast<char *>(aBinnedBuckets)) % cBytesPerBinnedBucket);
   size_t cFinalItems = (reinterpret_cast<char *>(pCopyFrom) - reinterpret_cast<char *>(aBinnedBuckets)) / cBytesPerBinnedBucket;

   const size_t cCasesTotal = pTrainingSet->GetTotalCountCaseOccurrences();
   EBM_ASSERT(cCasesTotal == cCasesTotalDebug);

   *pcCasesTotal = cCasesTotal;

   LOG(TraceLevelVerbose, "Exited CompressBinnedBuckets");
   return cFinalItems;
}

#endif // BINNED_BUCKET_H
