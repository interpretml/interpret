// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#if (defined(__clang__) || defined(__GNUC__) || defined(__SUNPRO_CC)) && defined(__x86_64__) || defined(_MSC_VER)

#include <cmath>
#include <immintrin.h> // SIMD.  Do not include in precompiled_header_cpp.hpp!

#include "libebm.h"
#include "logging.h"
#include "common_c.h"
#include "bridge_c.h"
#include "zones.h"

#include "common_cpp.hpp"
#include "bridge_cpp.hpp"

#include "Registration.hpp"
#include "Loss.hpp"

#include "approximate_math.hpp"
#include "compute_stats.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct Sse_32_Int final {
   static constexpr int cPack = 4;
   using T = uint32_t;

   inline Sse_32_Int(const T val) noexcept : m_data(_mm_set1_epi32(static_cast<T>(val))) {
   }

private:
   __m128i m_data;
};
static_assert(std::is_standard_layout<Sse_32_Int>::value && std::is_trivially_copyable<Sse_32_Int>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

struct Sse_32_Float final {
   static constexpr int cPack = 4;
   using T = float;
   using TInt = Sse_32_Int;

   WARNING_PUSH
   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   inline Sse_32_Float() noexcept {
   }
   WARNING_POP

   Sse_32_Float(const Sse_32_Float & other) noexcept = default; // preserve POD status
   Sse_32_Float & operator=(const Sse_32_Float &) noexcept = default; // preserve POD status

   inline Sse_32_Float(const double val) noexcept : m_data { _mm_set1_ps(static_cast<T>(val)) } {
   }

   inline Sse_32_Float & operator= (const double val) noexcept {
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

   inline void LoadAligned(const T * const a) noexcept {
      // WARNING: 'a' must be aligned memory with:    alignas(16) T a[cPack];
      m_data = _mm_load_ps(a);
   }

   inline void SaveAligned(T * const a) const noexcept {
      // WARNING: 'a' must be aligned memory with:    alignas(16) T a[cPack];
      _mm_store_ps(a, m_data);
   }

   template<typename Func>
   inline Sse_32_Float ApplyFunction(Func func) const noexcept {
      alignas(16) T aTemp[cPack];
      SaveAligned(aTemp);

      for(int i = 0; i < cPack; ++i) {
         aTemp[i] = func(aTemp[i]);
      }

      Sse_32_Float result;
      result.LoadAligned(aTemp);
      return result;
   }

   inline bool IsAnyEqual(const Sse_32_Float & other) const noexcept {
      return !!_mm_movemask_ps(_mm_cmpeq_ps(m_data, other.m_data));
   }

   inline bool IsAnyInf() const noexcept {
      return !!_mm_movemask_ps(_mm_cmpeq_ps(_mm_andnot_ps(_mm_set1_ps(T { -0.0 }), m_data), _mm_set1_ps(std::numeric_limits<T>::infinity())));
   }

   inline bool IsAnyNaN() const noexcept {
      // use the fact that a != a  always yields false, except when both are NaN in IEEE 754 where it's true
      return !!_mm_movemask_ps(_mm_cmpneq_ps(m_data, m_data));
   }

   inline Sse_32_Float Sqrt() const noexcept {
      // TODO: make a fast approximation of this
      return Sse_32_Float(_mm_sqrt_ss(m_data));
   }

   inline Sse_32_Float Exp() const noexcept {
      // TODO: make a fast approximation of this
      return ApplyFunction([](T x) { return std::exp(x); });
   }

   inline Sse_32_Float Log() const noexcept {
      // TODO: make a fast approximation of this
      return ApplyFunction([](T x) { return std::log(x); });
   }

   inline T Sum() const noexcept {
      // TODO: this could be written to be more efficient

      alignas(16) T aTemp[cPack];
      SaveAligned(aTemp);

      T sum = 0.0;
      for(int i = 0; i < cPack; ++i) {
         sum += aTemp[i];
      }
      return sum;
   }

   template<typename TLoss, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorApplyUpdate(const Loss * const pLoss, ApplyUpdateBridge * const pData) {
      // this allows us to switch execution onto GPU, FPGA, or other local computation
      RemoteApplyUpdate<TLoss, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pLoss, pData);
      return Error_None;
   }

private:

   inline Sse_32_Float(const __m128 & data) noexcept : m_data(data) {
   }

   __m128 m_data;
};
static_assert(std::is_standard_layout<Sse_32_Float>::value && std::is_trivially_copyable<Sse_32_Float>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

// FIRST, define the RegisterLoss function that we'll be calling from our registrations.  This is a static 
// function, so we can have duplicate named functions in other files and they'll refer to different functions
template<template <typename> class TRegistrable, typename... Args>
INLINE_ALWAYS static std::shared_ptr<const Registration> RegisterLoss(const char * const sRegistrationName, const Args...args) {
   return Register<TRegistrable, Sse_32_Float>(sRegistrationName, args...);
}

// now include all our special loss registrations which will use the RegisterLoss function we defined above!
#include "loss_registrations.hpp"

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm CreateLoss_Sse_32(
   const Config * const pConfig,
   const char * const sLoss,
   const char * const sLossEnd,
   LossWrapper * const pLossWrapperOut
) {
   return Loss::CreateLoss(&RegisterLosses, pConfig, sLoss, sLossEnd, pLossWrapperOut);
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
