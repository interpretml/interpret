// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef HISTOGRAM_BUCKET_VECTOR_ENTRY_H
#define HISTOGRAM_BUCKET_VECTOR_ENTRY_H

#include <type_traits> // std::is_standard_layout

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG

template<bool bClassification>
struct HistogramBucketVectorEntry;

struct HistogramBucketVectorEntryBase {
   HistogramBucketVectorEntryBase() = default; // preserve our POD status
   ~HistogramBucketVectorEntryBase() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   template<bool bClassification>
   INLINE_ALWAYS HistogramBucketVectorEntry<bClassification> * GetHistogramBucketVectorEntry() {
      return static_cast<HistogramBucketVectorEntry<bClassification> *>(this);
   }
   template<bool bClassification>
   INLINE_ALWAYS const HistogramBucketVectorEntry<bClassification> * GetHistogramBucketVectorEntry() const {
      return static_cast<const HistogramBucketVectorEntry<bClassification> *>(this);
   }
};
static_assert(std::is_standard_layout<HistogramBucketVectorEntryBase>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<HistogramBucketVectorEntryBase>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<HistogramBucketVectorEntryBase>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

template<>
struct HistogramBucketVectorEntry<true> final : HistogramBucketVectorEntryBase {

   HistogramBucketVectorEntry() = default; // preserve our POD status
   ~HistogramBucketVectorEntry() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   // classification version of the HistogramBucketVectorEntry class

   FloatEbmType m_sumResidualError;
   // TODO: for single features, we probably want to just do a single pass of the data and collect our m_sumDenominator during that sweep.  This is probably 
   //   also true for pairs since calculating pair sums can be done fairly efficiently, but for tripples and higher dimensions we might be better off 
   //   calculating JUST the sumResidualError which is the only thing required for choosing splits and we could then do a second pass of the data to 
   //   find the denominators once we know the splits.  Tripples and higher dimensions tend to re-add/subtract the same cells many times over which is 
   //   why it might be better there.  Test these theories out on large datasets
   FloatEbmType m_sumDenominator;

   INLINE_ALWAYS FloatEbmType GetSumDenominator() const {
      return m_sumDenominator;
   }
   INLINE_ALWAYS void SetSumDenominator(FloatEbmType sumDenominatorSet) {
      m_sumDenominator = sumDenominatorSet;
   }
   INLINE_ALWAYS void Add(const HistogramBucketVectorEntry<true> & other) {
      m_sumResidualError += other.m_sumResidualError;
      m_sumDenominator += other.m_sumDenominator;
   }
   INLINE_ALWAYS void Subtract(const HistogramBucketVectorEntry<true> & other) {
      m_sumResidualError -= other.m_sumResidualError;
      m_sumDenominator -= other.m_sumDenominator;
   }
   INLINE_ALWAYS void Copy(const HistogramBucketVectorEntry<true> & other) {
      m_sumResidualError = other.m_sumResidualError;
      m_sumDenominator = other.m_sumDenominator;
   }
   INLINE_ALWAYS void AssertZero() const {
      EBM_ASSERT(0 == m_sumResidualError);
      EBM_ASSERT(0 == m_sumDenominator);
   }
   INLINE_ALWAYS void Zero() {
      m_sumResidualError = FloatEbmType { 0 };
      m_sumDenominator = FloatEbmType { 0 };
   }
};
static_assert(std::is_standard_layout<HistogramBucketVectorEntry<true>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<HistogramBucketVectorEntry<true>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<HistogramBucketVectorEntry<true>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

template<>
struct HistogramBucketVectorEntry<false> final : HistogramBucketVectorEntryBase {
   // regression version of the HistogramBucketVectorEntry class

   HistogramBucketVectorEntry() = default; // preserve our POD status
   ~HistogramBucketVectorEntry() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   FloatEbmType m_sumResidualError;

   INLINE_ALWAYS FloatEbmType GetSumDenominator() const {
      EBM_ASSERT(false); // this should never be called, but the compiler seems to want it to exist
      return FloatEbmType { 0 };
   }
   INLINE_ALWAYS void SetSumDenominator(FloatEbmType sumDenominator) {
      UNUSED(sumDenominator);
      EBM_ASSERT(false); // this should never be called, but the compiler seems to want it to exist
   }
   INLINE_ALWAYS void Add(const HistogramBucketVectorEntry<false> & other) {
      m_sumResidualError += other.m_sumResidualError;
   }
   INLINE_ALWAYS void Subtract(const HistogramBucketVectorEntry<false> & other) {
      m_sumResidualError -= other.m_sumResidualError;
   }
   INLINE_ALWAYS void Copy(const HistogramBucketVectorEntry<false> & other) {
      m_sumResidualError = other.m_sumResidualError;
   }
   INLINE_ALWAYS void AssertZero() const {
      EBM_ASSERT(0 == m_sumResidualError);
   }
   INLINE_ALWAYS void Zero() {
      m_sumResidualError = FloatEbmType { 0 };
   }
};
static_assert(std::is_standard_layout<HistogramBucketVectorEntry<false>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<HistogramBucketVectorEntry<false>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<HistogramBucketVectorEntry<false>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");


#endif // HISTOGRAM_BUCKET_VECTOR_ENTRY_H
