// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <cmath> // exp, log
#include <limits> // numeric_limits
#include <type_traits> // is_unsigned
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
#include "compute_wrapper.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct Avx512_32_Float;

struct Avx512_32_Int final {
   friend Avx512_32_Float;
   friend inline Avx512_32_Float IfEqual(const Avx512_32_Int & cmp1, const Avx512_32_Int & cmp2, const Avx512_32_Float & trueVal, const Avx512_32_Float & falseVal) noexcept;

   using T = uint32_t;
   using TPack = __m512i;
   static_assert(std::is_unsigned<T>::value, "T must be an unsigned integer type");
   static_assert(std::numeric_limits<T>::max() <= std::numeric_limits<UIntExceed>::max(), "UIntExceed must be able to hold a T");
   static constexpr bool bCpu = false;
   static constexpr int k_cSIMDShift = 4;
   static constexpr int k_cSIMDPack = 1 << k_cSIMDShift;

   WARNING_PUSH
   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   inline Avx512_32_Int() noexcept {
   }
   WARNING_POP

   inline Avx512_32_Int(const T & val) noexcept : m_data(_mm512_set1_epi32(val)) {
   }

   inline static Avx512_32_Int Load(const T * const a) noexcept {
      return Avx512_32_Int(_mm512_load_si512(reinterpret_cast<const TPack *>(a)));
   }

   inline void Store(T * const a) const noexcept {
      _mm512_store_si512(reinterpret_cast<TPack *>(a), m_data);
   }

   inline static Avx512_32_Int LoadBytes(const uint8_t * const a) noexcept {
      const __m128i temp = _mm_load_si128(reinterpret_cast<const __m128i *>(a));
      return Avx512_32_Int(_mm512_cvtepu8_epi32(temp));
   }

   template<typename TFunc>
   static inline void Execute(const TFunc & func, const Avx512_32_Int & val0) noexcept {
      alignas(SIMD_BYTE_ALIGNMENT) T a0[k_cSIMDPack];
      val0.Store(a0);

      // no loops because this will disable optimizations for loops in the caller
      func(0, a0[0]);
      func(1, a0[1]);
      func(2, a0[2]);
      func(3, a0[3]);
      func(4, a0[4]);
      func(5, a0[5]);
      func(6, a0[6]);
      func(7, a0[7]);
      func(8, a0[8]);
      func(9, a0[9]);
      func(10, a0[10]);
      func(11, a0[11]);
      func(12, a0[12]);
      func(13, a0[13]);
      func(14, a0[14]);
      func(15, a0[15]);
   }

   inline static Avx512_32_Int MakeIndexes() noexcept {
      return Avx512_32_Int(_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
   }

   inline Avx512_32_Int operator+ (const Avx512_32_Int & other) const noexcept {
      return Avx512_32_Int(_mm512_add_epi32(m_data, other.m_data));
   }

   inline Avx512_32_Int operator* (const T & other) const noexcept {
      return Avx512_32_Int(_mm512_mullo_epi32(m_data, _mm512_set1_epi32(other)));
   }

   inline Avx512_32_Int operator>> (unsigned int shift) const noexcept {
      return Avx512_32_Int(_mm512_srli_epi32(m_data, shift));
   }

   inline Avx512_32_Int operator<< (unsigned int shift) const noexcept {
      return Avx512_32_Int(_mm512_slli_epi32(m_data, shift));
   }

   inline Avx512_32_Int operator& (const Avx512_32_Int & other) const noexcept {
      return Avx512_32_Int(_mm512_and_si512(other.m_data, m_data));
   }

private:
   inline Avx512_32_Int(const TPack & data) noexcept : m_data(data) {
   }

   TPack m_data;
};
static_assert(std::is_standard_layout<Avx512_32_Int>::value && std::is_trivially_copyable<Avx512_32_Int>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");


struct Avx512_32_Float final {
   using T = float;
   using TPack = __m512;
   using TInt = Avx512_32_Int;
   static_assert(sizeof(T) <= sizeof(Float_Big), "Float_Big must be able to hold a T");
   static constexpr bool bCpu = TInt::bCpu;
   static constexpr int k_cSIMDShift = TInt::k_cSIMDShift;
   static constexpr int k_cSIMDPack = TInt::k_cSIMDPack;

   WARNING_PUSH
   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   inline Avx512_32_Float() noexcept {
   }
   WARNING_POP

   inline Avx512_32_Float(const double val) noexcept : m_data { _mm512_set1_ps(static_cast<T>(val)) } {
   }
   inline Avx512_32_Float(const float val) noexcept : m_data { _mm512_set1_ps(static_cast<T>(val)) } {
   }
   inline Avx512_32_Float(const int val) noexcept : m_data { _mm512_set1_ps(static_cast<T>(val)) } {
   }


   inline Avx512_32_Float operator+() const noexcept {
      return *this;
   }

   inline Avx512_32_Float operator-() const noexcept {
      return Avx512_32_Float(_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(m_data), _mm512_set1_epi32(0x80000000))));
   }


   inline Avx512_32_Float operator+ (const Avx512_32_Float & other) const noexcept {
      return Avx512_32_Float(_mm512_add_ps(m_data, other.m_data));
   }

   inline Avx512_32_Float operator- (const Avx512_32_Float & other) const noexcept {
      return Avx512_32_Float(_mm512_sub_ps(m_data, other.m_data));
   }

   inline Avx512_32_Float operator* (const Avx512_32_Float & other) const noexcept {
      return Avx512_32_Float(_mm512_mul_ps(m_data, other.m_data));
   }

   inline Avx512_32_Float operator/ (const Avx512_32_Float & other) const noexcept {
      return Avx512_32_Float(_mm512_div_ps(m_data, other.m_data));
   }


   inline Avx512_32_Float & operator+= (const Avx512_32_Float & other) noexcept {
      *this = (*this) + other;
      return *this;
   }

   inline Avx512_32_Float & operator-= (const Avx512_32_Float & other) noexcept {
      *this = (*this) - other;
      return *this;
   }

   inline Avx512_32_Float & operator*= (const Avx512_32_Float & other) noexcept {
      *this = (*this) * other;
      return *this;
   }

   inline Avx512_32_Float & operator/= (const Avx512_32_Float & other) noexcept {
      *this = (*this) / other;
      return *this;
   }


   friend inline Avx512_32_Float operator+ (const double val, const Avx512_32_Float & other) noexcept {
      return Avx512_32_Float(val) + other;
   }

   friend inline Avx512_32_Float operator- (const double val, const Avx512_32_Float & other) noexcept {
      return Avx512_32_Float(val) - other;
   }

   friend inline Avx512_32_Float operator* (const double val, const Avx512_32_Float & other) noexcept {
      return Avx512_32_Float(val) * other;
   }

   friend inline Avx512_32_Float operator/ (const double val, const Avx512_32_Float & other) noexcept {
      return Avx512_32_Float(val) / other;
   }


   friend inline Avx512_32_Float operator+ (const float val, const Avx512_32_Float & other) noexcept {
      return Avx512_32_Float(val) + other;
   }

   friend inline Avx512_32_Float operator- (const float val, const Avx512_32_Float & other) noexcept {
      return Avx512_32_Float(val) - other;
   }

   friend inline Avx512_32_Float operator* (const float val, const Avx512_32_Float & other) noexcept {
      return Avx512_32_Float(val) * other;
   }

   friend inline Avx512_32_Float operator/ (const float val, const Avx512_32_Float & other) noexcept {
      return Avx512_32_Float(val) / other;
   }


   inline static Avx512_32_Float Load(const T * const a) noexcept {
      return Avx512_32_Float(_mm512_load_ps(a));
   }

   inline void Store(T * const a) const noexcept {
      _mm512_store_ps(a, m_data);
   }

   inline static Avx512_32_Float Load(const T * const a, const TInt i) noexcept {
      return Avx512_32_Float(_mm512_i32gather_ps(i.m_data, a, sizeof(a[0])));
   }

   inline void Store(T * const a, const TInt i) const noexcept {
      _mm512_i32scatter_ps(a, i.m_data, m_data, sizeof(a[0]));
   }

   template<typename TFunc>
   friend inline Avx512_32_Float ApplyFunc(const TFunc & func, const Avx512_32_Float & val) noexcept {

      alignas(SIMD_BYTE_ALIGNMENT) T aTemp[k_cSIMDPack];
      val.Store(aTemp);

      aTemp[0] = func(aTemp[0]);
      aTemp[1] = func(aTemp[1]);
      aTemp[2] = func(aTemp[2]);
      aTemp[3] = func(aTemp[3]);
      aTemp[4] = func(aTemp[4]);
      aTemp[5] = func(aTemp[5]);
      aTemp[6] = func(aTemp[6]);
      aTemp[7] = func(aTemp[7]);
      aTemp[8] = func(aTemp[8]);
      aTemp[9] = func(aTemp[9]);
      aTemp[10] = func(aTemp[10]);
      aTemp[11] = func(aTemp[11]);
      aTemp[12] = func(aTemp[12]);
      aTemp[13] = func(aTemp[13]);
      aTemp[14] = func(aTemp[14]);
      aTemp[15] = func(aTemp[15]);

      return Load(aTemp);
   }

   template<typename TFunc>
   static inline void Execute(const TFunc & func) noexcept {
      func(0);
      func(1);
      func(2);
      func(3);
      func(4);
      func(5);
      func(6);
      func(7);
      func(8);
      func(9);
      func(10);
      func(11);
      func(12);
      func(13);
      func(14);
      func(15);
   }

   template<typename TFunc>
   static inline void Execute(const TFunc & func, const Avx512_32_Float & val0) noexcept {
      alignas(SIMD_BYTE_ALIGNMENT) T a0[k_cSIMDPack];
      val0.Store(a0);

      func(0, a0[0]);
      func(1, a0[1]);
      func(2, a0[2]);
      func(3, a0[3]);
      func(4, a0[4]);
      func(5, a0[5]);
      func(6, a0[6]);
      func(7, a0[7]);
      func(8, a0[8]);
      func(9, a0[9]);
      func(10, a0[10]);
      func(11, a0[11]);
      func(12, a0[12]);
      func(13, a0[13]);
      func(14, a0[14]);
      func(15, a0[15]);
   }

   template<typename TFunc>
   static inline void Execute(const TFunc & func, const Avx512_32_Float & val0, const Avx512_32_Float & val1) noexcept {
      alignas(SIMD_BYTE_ALIGNMENT) T a0[k_cSIMDPack];
      val0.Store(a0);
      alignas(SIMD_BYTE_ALIGNMENT) T a1[k_cSIMDPack];
      val1.Store(a1);

      func(0, a0[0], a1[0]);
      func(1, a0[1], a1[1]);
      func(2, a0[2], a1[2]);
      func(3, a0[3], a1[3]);
      func(4, a0[4], a1[4]);
      func(5, a0[5], a1[5]);
      func(6, a0[6], a1[6]);
      func(7, a0[7], a1[7]);
      func(8, a0[8], a1[8]);
      func(9, a0[9], a1[9]);
      func(10, a0[10], a1[10]);
      func(11, a0[11], a1[11]);
      func(12, a0[12], a1[12]);
      func(13, a0[13], a1[13]);
      func(14, a0[14], a1[14]);
      func(15, a0[15], a1[15]);
   }

   friend inline Avx512_32_Float IfLess(const Avx512_32_Float & cmp1, const Avx512_32_Float & cmp2, const Avx512_32_Float & trueVal, const Avx512_32_Float & falseVal) noexcept {
      const __mmask16 mask = _mm512_cmp_ps_mask(cmp1.m_data, cmp2.m_data, _CMP_LT_OQ);
      return Avx512_32_Float(_mm512_mask_blend_ps(mask, falseVal.m_data, trueVal.m_data));
   }

   friend inline Avx512_32_Float IfEqual(const Avx512_32_Int & cmp1, const Avx512_32_Int & cmp2, const Avx512_32_Float & trueVal, const Avx512_32_Float & falseVal) noexcept {
      const __mmask16 mask = _mm512_cmpeq_epi32_mask(cmp1.m_data, cmp2.m_data);
      return Avx512_32_Float(_mm512_mask_blend_ps(mask, falseVal.m_data, trueVal.m_data));
   }

   friend inline Avx512_32_Float Abs(const Avx512_32_Float & val) noexcept {
      return Avx512_32_Float(_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(val.m_data), _mm512_set1_epi32(0x7FFFFFFF))));
   }

   friend inline Avx512_32_Float Reciprocal(const Avx512_32_Float & val) noexcept {
      return Avx512_32_Float(_mm512_rcp14_ps(val.m_data));
   }

   friend inline Avx512_32_Float FastApproxDivide(const Avx512_32_Float & dividend, const Avx512_32_Float & divisor) noexcept {
      return dividend * Reciprocal(divisor);
   }

   friend inline Avx512_32_Float Sqrt(const Avx512_32_Float & val) noexcept {
      return Avx512_32_Float(_mm512_sqrt_ps(val.m_data));
   }

   friend inline Avx512_32_Float Exp(const Avx512_32_Float & val) noexcept {
      return ApplyFunc([](T x) { return std::exp(x); }, val);
   }

   friend inline Avx512_32_Float Log(const Avx512_32_Float & val) noexcept {
      return ApplyFunc([](T x) { return std::log(x); }, val);
   }

   friend inline T Sum(const Avx512_32_Float & val) noexcept {
      return _mm512_reduce_add_ps(val.m_data);
   }


   template<typename TObjective, size_t cCompilerScores, bool bKeepGradHess, bool bCalcMetric, bool bWeight, bool bHessian, ptrdiff_t cCompilerPack>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorApplyUpdate(const Objective * const pObjective, ApplyUpdateBridge * const pData) noexcept {
      RemoteApplyUpdate<TObjective, cCompilerScores, bKeepGradHess, bCalcMetric, bWeight, bHessian, cCompilerPack>(pObjective, pData);
      return Error_None;
   }


   template<bool bHessian, size_t cCompilerScores, bool bWeight, bool bReplication, ptrdiff_t cCompilerPack>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorBinSumsBoosting(BinSumsBoostingBridge * const pParams) noexcept {
      RemoteBinSumsBoosting<Avx512_32_Float, bHessian, cCompilerScores, bWeight, bReplication, cCompilerPack>(pParams);
      return Error_None;
   }


   template<bool bHessian, size_t cCompilerScores, size_t cCompilerDimensions, bool bWeight>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorBinSumsInteraction(BinSumsInteractionBridge * const pParams) noexcept {
      RemoteBinSumsInteraction<Avx512_32_Float, bHessian, cCompilerScores, cCompilerDimensions, bWeight>(pParams);
      return Error_None;
   }


private:

   inline Avx512_32_Float(const TPack & data) noexcept : m_data(data) {
   }

   TPack m_data;
};
static_assert(std::is_standard_layout<Avx512_32_Float>::value && std::is_trivially_copyable<Avx512_32_Float>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

// FIRST, define the RegisterObjective function that we'll be calling from our registrations.  This is a static 
// function, so we can have duplicate named functions in other files and they'll refer to different functions
template<template <typename> class TRegistrable, typename... Args>
INLINE_ALWAYS static std::shared_ptr<const Registration> RegisterObjective(const char * const sRegistrationName, const Args...args) {
   return Register<TRegistrable, Avx512_32_Float>(sRegistrationName, args...);
}

// now include all our special objective registrations which will use the RegisterObjective function we defined above!
#include "objective_registrations.hpp"

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm CreateObjective_Avx512f_32(
   const Config * const pConfig,
   const char * const sObjective,
   const char * const sObjectiveEnd,
   ObjectiveWrapper * const pObjectiveWrapperOut
) {
   ErrorEbm error = ComputeWrapper<Avx512_32_Float>::FillWrapper(pObjectiveWrapperOut);
   if(Error_None != error) {
      return error;
   }
   return Objective::CreateObjective(&RegisterObjectives, pConfig, sObjective, sObjectiveEnd, pObjectiveWrapperOut);
}

} // DEFINED_ZONE_NAME
