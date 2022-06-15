// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef HISTOGRAM_TARGET_ENTRY_HPP
#define HISTOGRAM_TARGET_ENTRY_HPP

#include <type_traits> // std::is_standard_layout

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// TODO: use this in place of the hessian and the count of samples and the weight total.. we want to be able to convert everything to a float
//  to pass them on the network
template<typename TFloat>
union FloatAndInt {
   static_assert(std::is_same<float, TFloat>::value || std::is_same<double, TFloat>::value,
      "TFloat must be either float or double");
   typedef typename std::conditional<std::is_same<float, TFloat>::value, uint32_t, uint64_t>::type TUint;

   static_assert(sizeof(TFloat) == sizeof(TUint), "TFloat and TUint must be the same size");

   // these are paired to be the same size
   TFloat               m_float;
   TUint                m_uint;
};
static_assert(std::is_standard_layout<FloatAndInt<float>>::value && std::is_trivial<FloatAndInt<float>>::value,
   "This allows offsetof, memcpy, memset, the struct hack, inter-language, GPU or cross-machine");
static_assert(std::is_standard_layout<FloatAndInt<double>>::value && std::is_trivial<FloatAndInt<double>>::value,
   "This allows offsetof, memcpy, memset, the struct hack, inter-language, GPU or cross-machine");

static_assert(sizeof(FloatAndInt<float>) == sizeof(float), "FloatAndInt<float> and float must be the same size");
static_assert(sizeof(FloatAndInt<double>) == sizeof(double), "FloatAndInt<double> and double must be the same size");

template<typename TFloat, bool bClassification>
struct HistogramTargetEntry;

struct HistogramTargetEntryBase {
   HistogramTargetEntryBase() = default; // preserve our POD status
   ~HistogramTargetEntryBase() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   template<typename TFloat, bool bClassification>
   INLINE_ALWAYS HistogramTargetEntry<TFloat, bClassification> * GetHistogramTargetEntry() {
      return static_cast<HistogramTargetEntry<TFloat, bClassification> *>(this);
   }
   template<typename TFloat, bool bClassification>
   INLINE_ALWAYS const HistogramTargetEntry<TFloat, bClassification> * GetHistogramTargetEntry() const {
      return static_cast<const HistogramTargetEntry<TFloat, bClassification> *>(this);
   }

   INLINE_ALWAYS void Zero(const size_t cBytesPerItem, const size_t cItems = 1) {
      // The C standard guarantees that zeroing integer types is a zero, and IEEE-754 guarantees 
      // that zeroing a floating point is zero.  Our HistogramTargetEntry objects are POD and also only contain
      // floating point and unsigned integer types
      //
      // 6.2.6.2 Integer types -> 5. The values of any padding bits are unspecified.A valid (non - trap) 
      // object representation of a signed integer type where the sign bit is zero is a valid object 
      // representation of the corresponding unsigned type, and shall represent the same value.For any 
      // integer type, the object representation where all the bits are zero shall be a representation 
      // of the value zero in that type.

      static_assert(std::numeric_limits<float>::is_iec559, "memset of floats requires IEEE 754 to guarantee zeros");
      memset(this, 0, cItems * cBytesPerItem);
   }
};
static_assert(std::is_standard_layout<HistogramTargetEntryBase>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<HistogramTargetEntryBase>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<HistogramTargetEntryBase>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

template<typename TFloat>
struct HistogramTargetEntry<TFloat, true> final : HistogramTargetEntryBase {
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

   TFloat m_sumGradients;
   // TODO: for single features, we probably want to just do a single pass of the data and collect our m_sumHessians during that sweep.  This is probably 
   //   also true for pairs since calculating pair sums can be done fairly efficiently, but for tripples and higher dimensions we might be better off 
   //   calculating JUST the m_sumGradients which is the only thing required for choosing splits and we could then do a second pass of the data to 
   //   find the hessians once we know the splits.  Tripples and higher dimensions tend to re-add/subtract the same cells many times over which is 
   //   why it might be better there.  Test these theories out on large datasets
   TFloat m_sumHessians;

   // If we end up adding a 3rd derivative here, call it ThirdDerivative.  I like how Gradient and Hessian separate
   // nicely from eachother and match other package naming.  ThirdDerivative is nice since it's distinctly named
   // and easy to see rather than Derivative1, Derivative2, Derivative3, etc..

   INLINE_ALWAYS TFloat GetSumHessians() const {
      return m_sumHessians;
   }
   INLINE_ALWAYS void SetSumHessians(const TFloat sumHessians) {
      m_sumHessians = sumHessians;
   }
   INLINE_ALWAYS void Add(const HistogramTargetEntry<TFloat, true> & other) {
      m_sumGradients += other.m_sumGradients;
      m_sumHessians += other.m_sumHessians;
   }
   INLINE_ALWAYS void Subtract(const HistogramTargetEntry<TFloat, true> & other) {
      m_sumGradients -= other.m_sumGradients;
      m_sumHessians -= other.m_sumHessians;
   }
   INLINE_ALWAYS void Copy(const HistogramTargetEntry<TFloat, true> & other) {
      m_sumGradients = other.m_sumGradients;
      m_sumHessians = other.m_sumHessians;
   }
   INLINE_ALWAYS void AssertZero() const {
      EBM_ASSERT(0 == m_sumGradients);
      EBM_ASSERT(0 == m_sumHessians);
   }
};
static_assert(std::is_standard_layout<HistogramTargetEntry<double, true>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<HistogramTargetEntry<double, true>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<HistogramTargetEntry<double, true>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

static_assert(std::is_standard_layout<HistogramTargetEntry<float, true>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<HistogramTargetEntry<float, true>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<HistogramTargetEntry<float, true>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

template<typename TFloat>
struct HistogramTargetEntry<TFloat, false> final : HistogramTargetEntryBase {
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

   TFloat m_sumGradients;

   INLINE_ALWAYS TFloat GetSumHessians() const {
      EBM_ASSERT(false); // this should never be called, but the compiler seems to want it to exist
      return TFloat { 0 };
   }
   INLINE_ALWAYS void SetSumHessians(const TFloat sumHessians) {
      UNUSED(sumHessians);
      EBM_ASSERT(false); // this should never be called, but the compiler seems to want it to exist
   }
   INLINE_ALWAYS void Add(const HistogramTargetEntry<TFloat, false> & other) {
      m_sumGradients += other.m_sumGradients;
   }
   INLINE_ALWAYS void Subtract(const HistogramTargetEntry<TFloat, false> & other) {
      m_sumGradients -= other.m_sumGradients;
   }
   INLINE_ALWAYS void Copy(const HistogramTargetEntry<TFloat, false> & other) {
      m_sumGradients = other.m_sumGradients;
   }
   INLINE_ALWAYS void AssertZero() const {
      EBM_ASSERT(0 == m_sumGradients);
   }
};
static_assert(std::is_standard_layout<HistogramTargetEntry<double, false>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<HistogramTargetEntry<double, false>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<HistogramTargetEntry<double, false>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

static_assert(std::is_standard_layout<HistogramTargetEntry<float, false>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<HistogramTargetEntry<float, false>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<HistogramTargetEntry<float, false>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

template<typename TFloat>
INLINE_ALWAYS size_t GetHistogramTargetEntrySize(const bool bClassification) {
   if(bClassification) {
      return sizeof(HistogramTargetEntry<TFloat, true>);
   } else {
      return sizeof(HistogramTargetEntry<TFloat, false>);
   }
}

} // DEFINED_ZONE_NAME

#endif // HISTOGRAM_TARGET_ENTRY_HPP
