// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef HISTOGRAM_TARGET_ENTRY_H
#define HISTOGRAM_TARGET_ENTRY_H

#include <type_traits> // std::is_standard_layout

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG

template<bool bClassification>
struct HistogramTargetEntry;

struct HistogramTargetEntryBase {
   HistogramTargetEntryBase() = default; // preserve our POD status
   ~HistogramTargetEntryBase() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   template<bool bClassification>
   INLINE_ALWAYS HistogramTargetEntry<bClassification> * GetHistogramTargetEntry() {
      return static_cast<HistogramTargetEntry<bClassification> *>(this);
   }
   template<bool bClassification>
   INLINE_ALWAYS const HistogramTargetEntry<bClassification> * GetHistogramTargetEntry() const {
      return static_cast<const HistogramTargetEntry<bClassification> *>(this);
   }
};
static_assert(std::is_standard_layout<HistogramTargetEntryBase>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<HistogramTargetEntryBase>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<HistogramTargetEntryBase>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

template<>
struct HistogramTargetEntry<true> final : HistogramTargetEntryBase {
   // classification version of the HistogramTargetEntry class

#ifndef __SUNPRO_CC

   // the Oracle Developer Studio compiler has what I think is a bug by making any class that includes 
   // HistogramTargetEntry fields turn into non-trivial classes, so exclude the Oracle compiler
   // from these protections

   HistogramTargetEntry() = default; // preserve our POD status
   ~HistogramTargetEntry() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

#endif // __SUNPRO_CC

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
   INLINE_ALWAYS void Add(const HistogramTargetEntry<true> & other) {
      m_sumResidualError += other.m_sumResidualError;
      m_sumDenominator += other.m_sumDenominator;
   }
   INLINE_ALWAYS void Subtract(const HistogramTargetEntry<true> & other) {
      m_sumResidualError -= other.m_sumResidualError;
      m_sumDenominator -= other.m_sumDenominator;
   }
   INLINE_ALWAYS void Copy(const HistogramTargetEntry<true> & other) {
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
static_assert(std::is_standard_layout<HistogramTargetEntry<true>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<HistogramTargetEntry<true>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<HistogramTargetEntry<true>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

template<>
struct HistogramTargetEntry<false> final : HistogramTargetEntryBase {
   // regression version of the HistogramTargetEntry class

#ifndef __SUNPRO_CC

   // the Oracle Developer Studio compiler has what I think is a bug by making any class that includes 
   // HistogramTargetEntry fields turn into non-trivial classes, so exclude the Oracle compiler
   // from these protections

   HistogramTargetEntry() = default; // preserve our POD status
   ~HistogramTargetEntry() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

#endif // __SUNPRO_CC

   FloatEbmType m_sumResidualError;

   INLINE_ALWAYS FloatEbmType GetSumDenominator() const {
      EBM_ASSERT(false); // this should never be called, but the compiler seems to want it to exist
      return FloatEbmType { 0 };
   }
   INLINE_ALWAYS void SetSumDenominator(FloatEbmType sumDenominator) {
      UNUSED(sumDenominator);
      EBM_ASSERT(false); // this should never be called, but the compiler seems to want it to exist
   }
   INLINE_ALWAYS void Add(const HistogramTargetEntry<false> & other) {
      m_sumResidualError += other.m_sumResidualError;
   }
   INLINE_ALWAYS void Subtract(const HistogramTargetEntry<false> & other) {
      m_sumResidualError -= other.m_sumResidualError;
   }
   INLINE_ALWAYS void Copy(const HistogramTargetEntry<false> & other) {
      m_sumResidualError = other.m_sumResidualError;
   }
   INLINE_ALWAYS void AssertZero() const {
      EBM_ASSERT(0 == m_sumResidualError);
   }
   INLINE_ALWAYS void Zero() {
      m_sumResidualError = FloatEbmType { 0 };
   }
};
static_assert(std::is_standard_layout<HistogramTargetEntry<false>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<HistogramTargetEntry<false>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<HistogramTargetEntry<false>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");


#endif // HISTOGRAM_TARGET_ENTRY_H
