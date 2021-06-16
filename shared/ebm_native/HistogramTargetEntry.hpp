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
   typedef typename std::conditional<std::is_same<float, TFloat>::value, uint32_t, uint64_t>::type UIntEquivalentType;

   static_assert(sizeof(TFloat) == sizeof(UIntEquivalentType),
      "TFloat and IntEquivalentType must be the same size");

   // these are paired to be the same size
   TFloat               m_float;
   UIntEquivalentType   m_uint;
};
static_assert(std::is_standard_layout<FloatAndInt<float>>::value &&
   std::is_trivial<FloatAndInt<float>>::value,
   "This allows offsetof, memcpy, memset, the struct hack, inter-language, GPU or cross-machine");
static_assert(std::is_standard_layout<FloatAndInt<double>>::value &&
   std::is_trivial<FloatAndInt<double>>::value,
   "This allows offsetof, memcpy, memset, the struct hack, inter-language, GPU or cross-machine");

static_assert(sizeof(FloatAndInt<float>) == sizeof(float), "FloatAndInt<float> and float must be the same size");
static_assert(sizeof(FloatAndInt<double>) == sizeof(double), "FloatAndInt<double> and double must be the same size");

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

   FloatEbmType m_sumGradients;
   // TODO: for single features, we probably want to just do a single pass of the data and collect our m_sumHessians during that sweep.  This is probably 
   //   also true for pairs since calculating pair sums can be done fairly efficiently, but for tripples and higher dimensions we might be better off 
   //   calculating JUST the m_sumGradients which is the only thing required for choosing splits and we could then do a second pass of the data to 
   //   find the hessians once we know the splits.  Tripples and higher dimensions tend to re-add/subtract the same cells many times over which is 
   //   why it might be better there.  Test these theories out on large datasets
   FloatEbmType m_sumHessians;

   // If we end up adding a 3rd derivative here, call it ThirdDerivative.  I like how Gradient and Hessian separate
   // nicely from eachother and match other package naming.  ThirdDerivative is nice since it's distinctly named
   // and easy to see rather than Derivative1, Derivative2, Derivative3, etc..

   INLINE_ALWAYS FloatEbmType GetSumHessians() const {
      return m_sumHessians;
   }
   INLINE_ALWAYS void SetSumHessians(const FloatEbmType sumHessians) {
      m_sumHessians = sumHessians;
   }
   INLINE_ALWAYS void Add(const HistogramTargetEntry<true> & other) {
      m_sumGradients += other.m_sumGradients;
      m_sumHessians += other.m_sumHessians;
   }
   INLINE_ALWAYS void Subtract(const HistogramTargetEntry<true> & other) {
      m_sumGradients -= other.m_sumGradients;
      m_sumHessians -= other.m_sumHessians;
   }
   INLINE_ALWAYS void Copy(const HistogramTargetEntry<true> & other) {
      m_sumGradients = other.m_sumGradients;
      m_sumHessians = other.m_sumHessians;
   }
   INLINE_ALWAYS void AssertZero() const {
      EBM_ASSERT(0 == m_sumGradients);
      EBM_ASSERT(0 == m_sumHessians);
   }
   INLINE_ALWAYS void Zero() {
      m_sumGradients = FloatEbmType { 0 };
      m_sumHessians = FloatEbmType { 0 };
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

   FloatEbmType m_sumGradients;

   INLINE_ALWAYS FloatEbmType GetSumHessians() const {
      EBM_ASSERT(false); // this should never be called, but the compiler seems to want it to exist
      return FloatEbmType { 0 };
   }
   INLINE_ALWAYS void SetSumHessians(const FloatEbmType sumHessians) {
      UNUSED(sumHessians);
      EBM_ASSERT(false); // this should never be called, but the compiler seems to want it to exist
   }
   INLINE_ALWAYS void Add(const HistogramTargetEntry<false> & other) {
      m_sumGradients += other.m_sumGradients;
   }
   INLINE_ALWAYS void Subtract(const HistogramTargetEntry<false> & other) {
      m_sumGradients -= other.m_sumGradients;
   }
   INLINE_ALWAYS void Copy(const HistogramTargetEntry<false> & other) {
      m_sumGradients = other.m_sumGradients;
   }
   INLINE_ALWAYS void AssertZero() const {
      EBM_ASSERT(0 == m_sumGradients);
   }
   INLINE_ALWAYS void Zero() {
      m_sumGradients = FloatEbmType { 0 };
   }
};
static_assert(std::is_standard_layout<HistogramTargetEntry<false>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<HistogramTargetEntry<false>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<HistogramTargetEntry<false>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // HISTOGRAM_TARGET_ENTRY_HPP
