// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#if (defined(__clang__) || defined(__GNUC__) || defined(__SUNPRO_CC)) && defined(__x86_64__) || defined(_MSC_VER)

#include <cmath>
#include <type_traits>
#include <immintrin.h> // SIMD.  Do not include in precompiled_header_cpp.hpp!

#include "libebm.h"
#include "logging.h"
#include "common_c.h"
#include "bridge_c.h"
#include "zones.h"

#include "common_cpp.hpp"
#include "bridge_cpp.hpp"

#include "Registration.hpp"
#include "Objective.hpp"

#include "approximate_math.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct Sse_32_Float;

struct Sse_32_Int final {
   friend Sse_32_Float;
   friend inline Sse_32_Float IfEqual(const Sse_32_Int & cmp1, const Sse_32_Int & cmp2, const Sse_32_Float & trueVal, const Sse_32_Float & falseVal) noexcept;

   using T = uint32_t;
   using TPack = __m128i;
   static_assert(std::is_unsigned<T>::value, "T must be an unsigned integer type");
   static_assert(std::numeric_limits<T>::max() <= std::numeric_limits<UIntExceed>::max(), "UIntExceed must be able to hold a T");
   static constexpr bool bCpu = false;
   static constexpr int k_cSIMDShift = 2;
   static constexpr int k_cSIMDPack = 1 << k_cSIMDShift;

   WARNING_PUSH
   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   inline Sse_32_Int() noexcept {
   }
   WARNING_POP

   inline Sse_32_Int(const T val) noexcept : m_data(_mm_set1_epi32(static_cast<T>(val))) {
   }

   inline static Sse_32_Int Load(const T * const a) noexcept {
      EBM_ASSERT(IsAligned(a, sizeof(TPack)));
      return Sse_32_Int(_mm_load_si128(reinterpret_cast<const TPack *>(a)));
   }

   inline void Store(T * const a) const noexcept {
      EBM_ASSERT(IsAligned(a, sizeof(TPack)));
      _mm_store_si128(reinterpret_cast<TPack *>(a), m_data);
   }

   template<typename TFunc>
   friend inline Sse_32_Int ApplyFunction(const Sse_32_Int & val, const TFunc & func) noexcept {
      alignas(SIMD_BYTE_ALIGNMENT) T aTemp[k_cSIMDPack];
      val.Store(aTemp);

      // no loops because this will disable optimizations for loops in the caller
      aTemp[0] = func(aTemp[0]);
      aTemp[1] = func(aTemp[1]);
      aTemp[2] = func(aTemp[2]);
      aTemp[3] = func(aTemp[3]);

      return Load(aTemp);
   }

   inline static Sse_32_Int MakeIndexes() noexcept {
      return Sse_32_Int(_mm_set_epi32(3, 2, 1, 0));
   }

   inline Sse_32_Int operator+ (const Sse_32_Int & other) const noexcept {
      return Sse_32_Int(_mm_add_epi32(m_data, other.m_data));
   }

   inline Sse_32_Int & operator+= (const Sse_32_Int & other) noexcept {
      *this = (*this) + other;
      return *this;
   }

   inline Sse_32_Int operator* (const T & other) const noexcept {
      // we'd really like to use _mm_mullo_epi32 since that multiplies 32 bit integers (either signed or unsigned 
      // since the low 32 bit results are the same) in the SIMD register with another SIMD register, but 
      // it's only available in SSE4.1.

      return ApplyFunction(*this, [other](T x) { return x * other; });
   }

   inline Sse_32_Int & operator*= (const T & other) noexcept {
      *this = (*this) * other;
      return *this;
   }

   inline Sse_32_Int operator>> (int shift) const noexcept {
      return Sse_32_Int(_mm_srli_epi32(m_data, shift));
   }

   inline Sse_32_Int operator<< (int shift) const noexcept {
      return Sse_32_Int(_mm_slli_epi32(m_data, shift));
   }

   inline Sse_32_Int operator& (const Sse_32_Int & other) const noexcept {
      return Sse_32_Int(_mm_and_si128(other.m_data, m_data));
   }

private:
   inline Sse_32_Int(const TPack & data) noexcept : m_data(data) {
   }

   TPack m_data;
};
static_assert(std::is_standard_layout<Sse_32_Int>::value && std::is_trivially_copyable<Sse_32_Int>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

struct Sse_32_Float final {
   using T = float;
   using TPack = __m128;
   using TInt = Sse_32_Int;
   static_assert(sizeof(T) <= sizeof(FloatExceed), "FloatExceed must be able to hold a T");
   static constexpr bool bCpu = TInt::bCpu;
   static constexpr int k_cSIMDShift = TInt::k_cSIMDShift;
   static constexpr int k_cSIMDPack = TInt::k_cSIMDPack;

   WARNING_PUSH
   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   inline Sse_32_Float() noexcept {
   }
   WARNING_POP

   Sse_32_Float(const Sse_32_Float & other) noexcept = default; // preserve POD status
   Sse_32_Float & operator=(const Sse_32_Float &) noexcept = default; // preserve POD status

   inline Sse_32_Float(const double val) noexcept : m_data { _mm_set1_ps(static_cast<T>(val)) } {
   }
   inline Sse_32_Float(const float val) noexcept : m_data { _mm_set1_ps(static_cast<T>(val)) } {
   }
   inline Sse_32_Float(const int val) noexcept : m_data { _mm_set1_ps(static_cast<T>(val)) } {
   }

   inline Sse_32_Float & operator= (const double val) noexcept {
      m_data = _mm_set1_ps(static_cast<T>(val));
      return *this;
   }
   inline Sse_32_Float & operator= (const float val) noexcept {
      m_data = _mm_set1_ps(static_cast<T>(val));
      return *this;
   }
   inline Sse_32_Float & operator= (const int val) noexcept {
      m_data = _mm_set1_ps(static_cast<T>(val));
      return *this;
   }


   inline Sse_32_Float operator+() const noexcept {
      return *this;
   }

   inline Sse_32_Float operator-() const noexcept {
      return Sse_32_Float(_mm_xor_ps(m_data, _mm_set1_ps(-0.0f)));
   }


   inline Sse_32_Float operator+ (const Sse_32_Float & other) const noexcept {
      return Sse_32_Float(_mm_add_ps(m_data, other.m_data));
   }

   inline Sse_32_Float operator- (const Sse_32_Float & other) const noexcept {
      return Sse_32_Float(_mm_sub_ps(m_data, other.m_data));
   }

   inline Sse_32_Float operator* (const Sse_32_Float & other) const noexcept {
      return Sse_32_Float(_mm_mul_ps(m_data, other.m_data));
   }

   inline Sse_32_Float operator/ (const Sse_32_Float & other) const noexcept {
      return Sse_32_Float(_mm_div_ps(m_data, other.m_data));
   }

   inline Sse_32_Float & operator+= (const Sse_32_Float & other) noexcept {
      *this = (*this) + other;
      return *this;
   }

   inline Sse_32_Float & operator-= (const Sse_32_Float & other) noexcept {
      *this = (*this) - other;
      return *this;
   }

   inline Sse_32_Float & operator*= (const Sse_32_Float & other) noexcept {
      *this = (*this) * other;
      return *this;
   }

   inline Sse_32_Float & operator/= (const Sse_32_Float & other) noexcept {
      *this = (*this) / other;
      return *this;
   }


   friend inline Sse_32_Float operator+ (const double val, const Sse_32_Float & other) noexcept {
      return Sse_32_Float(val) + other;
   }

   friend inline Sse_32_Float operator- (const double val, const Sse_32_Float & other) noexcept {
      return Sse_32_Float(val) - other;
   }

   friend inline Sse_32_Float operator* (const double val, const Sse_32_Float & other) noexcept {
      return Sse_32_Float(val) * other;
   }

   friend inline Sse_32_Float operator/ (const double val, const Sse_32_Float & other) noexcept {
      return Sse_32_Float(val) / other;
   }

   inline static Sse_32_Float Load(const T * const a) noexcept {
      EBM_ASSERT(IsAligned(a, sizeof(TPack)));
      return Sse_32_Float(_mm_load_ps(a));
   }

   inline void Store(T * const a) const noexcept {
      EBM_ASSERT(IsAligned(a, sizeof(TPack)));
      _mm_store_ps(a, m_data);
   }

   inline static Sse_32_Float Load(const T * const a, const TInt i) noexcept {
      EBM_ASSERT(IsAligned(a, sizeof(TPack)));

      // TODO: in the future use _mm_i32gather_ps using a scale of sizeof(T)

      alignas(SIMD_BYTE_ALIGNMENT) TInt::T ints[k_cSIMDPack];
      alignas(SIMD_BYTE_ALIGNMENT) T floats[k_cSIMDPack];

      i.Store(ints);

      floats[0] = a[ints[0]];
      floats[1] = a[ints[1]];
      floats[2] = a[ints[2]];
      floats[3] = a[ints[3]];

      return Load(floats);
   }

   inline void Store(T * const a, const TInt i) noexcept {
      EBM_ASSERT(IsAligned(a, sizeof(TPack)));

      // TODO: in the future use _mm_i32scatter_ps using a scale of sizeof(T)

      alignas(SIMD_BYTE_ALIGNMENT) TInt::T ints[k_cSIMDPack];
      alignas(SIMD_BYTE_ALIGNMENT) T floats[k_cSIMDPack];

      i.Store(ints);
      Store(floats);

      a[ints[0]] = floats[0];
      a[ints[1]] = floats[1];
      a[ints[2]] = floats[2];
      a[ints[3]] = floats[3];
   }

   template<typename TFunc>
   friend inline Sse_32_Float ApplyFunction(const Sse_32_Float & val, const TFunc & func) noexcept {
      alignas(SIMD_BYTE_ALIGNMENT) T aTemp[k_cSIMDPack];
      val.Store(aTemp);

      // no loops because this will disable optimizations for loops in the caller
      aTemp[0] = func(aTemp[0]);
      aTemp[1] = func(aTemp[1]);
      aTemp[2] = func(aTemp[2]);
      aTemp[3] = func(aTemp[3]);

      return Load(aTemp);
   }

   friend inline Sse_32_Float IfGreater(const Sse_32_Float & cmp1, const Sse_32_Float & cmp2, const Sse_32_Float & trueVal, const Sse_32_Float & falseVal) noexcept {
      TPack mask = _mm_cmpgt_ps(cmp1.m_data, cmp2.m_data);
      TPack maskedTrue = _mm_and_ps(mask, trueVal.m_data);
      TPack maskedFalse = _mm_andnot_ps(mask, falseVal.m_data);
      return Sse_32_Float(_mm_or_ps(maskedTrue, maskedFalse));
   }

   friend inline Sse_32_Float IfLess(const Sse_32_Float & cmp1, const Sse_32_Float & cmp2, const Sse_32_Float & trueVal, const Sse_32_Float & falseVal) noexcept {
      TPack mask = _mm_cmplt_ps(cmp1.m_data, cmp2.m_data);
      TPack maskedTrue = _mm_and_ps(mask, trueVal.m_data);
      TPack maskedFalse = _mm_andnot_ps(mask, falseVal.m_data);
      return Sse_32_Float(_mm_or_ps(maskedTrue, maskedFalse));
   }

   friend inline Sse_32_Float IfEqual(const Sse_32_Int & cmp1, const Sse_32_Int & cmp2, const Sse_32_Float & trueVal, const Sse_32_Float & falseVal) noexcept {
      TPack mask = _mm_castsi128_ps(_mm_cmpeq_epi32(cmp1.m_data, cmp2.m_data));
      TPack maskedTrue = _mm_and_ps(mask, trueVal.m_data);
      TPack maskedFalse = _mm_andnot_ps(mask, falseVal.m_data);
      return Sse_32_Float(_mm_or_ps(maskedTrue, maskedFalse));
   }

   friend inline Sse_32_Float Abs(const Sse_32_Float & val) noexcept {
      return Sse_32_Float(_mm_and_ps(val.m_data, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF))));
   }

   friend inline Sse_32_Float Sqrt(const Sse_32_Float & val) noexcept {
      // TODO: make a fast approximation of this
      return Sse_32_Float(_mm_sqrt_ss(val.m_data));
   }

   friend inline Sse_32_Float Exp(const Sse_32_Float & val) noexcept {
      // TODO: make a fast approximation of this
      return ApplyFunction(val, [](T x) { return std::exp(x); });
   }

   friend inline Sse_32_Float Log(const Sse_32_Float & val) noexcept {
      // TODO: make a fast approximation of this
      return ApplyFunction(val, [](T x) { return std::log(x); });
   }

   friend inline T Sum(const Sse_32_Float & val) noexcept {
      // TODO: use _mm_hadd_ps for SSE3 and later

      TPack packed = val.m_data;
      packed = _mm_add_ps(packed, _mm_shuffle_ps(packed, packed, _MM_SHUFFLE(2, 3, 0, 1)));
      packed = _mm_add_ss(packed, _mm_shuffle_ps(packed, packed, _MM_SHUFFLE(1, 0, 3, 2)));
      return _mm_cvtss_f32(packed);
   }

   template<typename TObjective, size_t cCompilerScores, bool bKeepGradHess, bool bCalcMetric, bool bWeight, bool bHessian, ptrdiff_t cCompilerPack>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorApplyUpdate(const Objective * const pObjective, ApplyUpdateBridge * const pData) noexcept {
      // this allows us to switch execution onto GPU, FPGA, or other local computation
      RemoteApplyUpdate<TObjective, cCompilerScores, bKeepGradHess, bCalcMetric, bWeight, bHessian, cCompilerPack>(pObjective, pData);
      return Error_None;
   }

private:

   inline Sse_32_Float(const TPack & data) noexcept : m_data(data) {
   }

   TPack m_data;
};
static_assert(std::is_standard_layout<Sse_32_Float>::value && std::is_trivially_copyable<Sse_32_Float>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

// FIRST, define the RegisterObjective function that we'll be calling from our registrations.  This is a static 
// function, so we can have duplicate named functions in other files and they'll refer to different functions
template<template <typename> class TRegistrable, typename... Args>
INLINE_ALWAYS static std::shared_ptr<const Registration> RegisterObjective(const char * const sRegistrationName, const Args...args) {
   return Register<TRegistrable, Sse_32_Float>(sRegistrationName, args...);
}

// now include all our special objective registrations which will use the RegisterObjective function we defined above!
#include "objective_registrations.hpp"

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm CreateObjective_Sse_32(
   const Config * const pConfig,
   const char * const sObjective,
   const char * const sObjectiveEnd,
   ObjectiveWrapper * const pObjectiveWrapperOut
) {
   return Objective::CreateObjective(&RegisterObjectives, pConfig, sObjective, sObjectiveEnd, pObjectiveWrapperOut);
}

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm CreateMetric_Sse_32(
   const Config * const pConfig,
   const char * const sMetric,
   const char * const sMetricEnd
//   MetricWrapper * const pMetricWrapperOut,
) {
   UNUSED(pConfig);
   UNUSED(sMetric);
   UNUSED(sMetricEnd);

   return Error_UnexpectedInternal;
}

} // DEFINED_ZONE_NAME

#endif // architecture SSE2
