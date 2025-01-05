// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifdef BRIDGE_AVX512F_32

#define _CRT_SECURE_NO_DEPRECATE

#include <cmath> // exp, log
#include <limits> // numeric_limits
#include <type_traits> // is_unsigned
#include <immintrin.h> // SIMD.  Do not include in pch.hpp!

#include "libebm.h"
#include "logging.h"
#include "unzoned.h"

#define ZONE_avx512f
#include "zones.h"

#include "bridge.h"
#include "common.hpp"
#include "bridge.hpp"

#include "Registration.hpp"
#include "Objective.hpp"

#include "math.hpp"
#include "approximate_math.hpp"
#include "compute_wrapper.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

static constexpr size_t k_cAlignment = 64;
struct alignas(k_cAlignment) Avx512f_32_Float;
struct alignas(k_cAlignment) Avx512f_32_Int;

template<bool bNegateInput = false,
      bool bNaNPossible = true,
      bool bUnderflowPossible = true,
      bool bOverflowPossible = true>
inline Avx512f_32_Float Exp(const Avx512f_32_Float& val) noexcept;
template<bool bNegateOutput = false,
      bool bNaNPossible = true,
      bool bNegativePossible = true,
      bool bZeroPossible = true,
      bool bPositiveInfinityPossible = true>
inline Avx512f_32_Float Log(const Avx512f_32_Float& val) noexcept;

// this is super-special and included inside the zone namespace
#include "objective_registrations.hpp"

struct alignas(k_cAlignment) Avx512f_32_Int final {
   friend Avx512f_32_Float;

   using T = uint32_t;
   using TPack = __m512i;
   static_assert(std::is_unsigned<T>::value, "T must be an unsigned integer type");
   static_assert(
         std::is_same<UIntBig, T>::value || std::is_same<UIntSmall, T>::value, "T must be either UIntBig or UIntSmall");
   static constexpr AccelerationFlags k_zone = AccelerationFlags_AVX512F;
   static constexpr int k_cSIMDShift = 4;
   static constexpr int k_cSIMDPack = 1 << k_cSIMDShift;
   static constexpr int k_cTypeShift = 2;
   static_assert(1 << k_cTypeShift == sizeof(T), "k_cTypeShift must be equivalent to the type size");

   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   inline Avx512f_32_Int() noexcept {}

   inline Avx512f_32_Int(const T& val) noexcept : m_data(_mm512_set1_epi32(val)) {}

   inline static Avx512f_32_Int Load(const T* const a) noexcept { return Avx512f_32_Int(_mm512_load_si512(a)); }

   inline void Store(T* const a) const noexcept { _mm512_store_si512(a, m_data); }

   inline static Avx512f_32_Int LoadBytes(const uint8_t* const a) noexcept {
      return Avx512f_32_Int(_mm512_cvtepu8_epi32(_mm_load_si128(reinterpret_cast<const __m128i*>(a))));
   }

   template<typename TFunc> static inline void Execute(const TFunc& func, const Avx512f_32_Int& val0) noexcept {
      alignas(k_cAlignment) T a0[k_cSIMDPack];
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

   inline static Avx512f_32_Int MakeIndexes() noexcept {
      return Avx512f_32_Int(_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
   }

   friend inline __mmask16 operator==(const Avx512f_32_Int& left, const Avx512f_32_Int& right) noexcept {
      return _mm512_cmpeq_epi32_mask(left.m_data, right.m_data);
   }

   inline Avx512f_32_Int operator+(const Avx512f_32_Int& other) const noexcept {
      return Avx512f_32_Int(_mm512_add_epi32(m_data, other.m_data));
   }

   inline Avx512f_32_Int operator-(const Avx512f_32_Int& other) const noexcept {
      return Avx512f_32_Int(_mm512_sub_epi32(m_data, other.m_data));
   }

   inline Avx512f_32_Int operator*(const T& other) const noexcept {
      return Avx512f_32_Int(_mm512_mullo_epi32(m_data, _mm512_set1_epi32(other)));
   }

   inline Avx512f_32_Int operator>>(int shift) const noexcept {
      return Avx512f_32_Int(_mm512_srli_epi32(m_data, shift));
   }

   inline Avx512f_32_Int operator<<(int shift) const noexcept {
      return Avx512f_32_Int(_mm512_slli_epi32(m_data, shift));
   }

   inline Avx512f_32_Int operator&(const Avx512f_32_Int& other) const noexcept {
      return Avx512f_32_Int(_mm512_and_si512(m_data, other.m_data));
   }

   inline Avx512f_32_Int operator|(const Avx512f_32_Int& other) const noexcept {
      return Avx512f_32_Int(_mm512_or_si512(m_data, other.m_data));
   }

   friend inline Avx512f_32_Int IfThenElse(
         const __mmask16& cmp, const Avx512f_32_Int& trueVal, const Avx512f_32_Int& falseVal) noexcept {
      return Avx512f_32_Int(_mm512_castps_si512(
            _mm512_mask_blend_ps(cmp, _mm512_castsi512_ps(falseVal.m_data), _mm512_castsi512_ps(trueVal.m_data))));
   }

   friend inline Avx512f_32_Int IfAdd(
         const __mmask16& cmp, const Avx512f_32_Int& base, const Avx512f_32_Int& addend) noexcept {
      return Avx512f_32_Int(_mm512_mask_add_epi32(base.m_data, cmp, base.m_data, addend.m_data));
   }

   friend inline Avx512f_32_Int PermuteForInterleaf(const Avx512f_32_Int& val) noexcept {
      // this function permutes the values into positions that the Interleaf function expects
      // but for any SIMD implementation the positions can be variable as long as they work together

      // TODO: we might be able to move this operation to where we store the packed indexes so that
      // it doesn't need to execute in the tight loop

      return Avx512f_32_Int(_mm512_permutexvar_epi32(
            _mm512_setr_epi32(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15), val.m_data));
   }

 private:
   inline Avx512f_32_Int(const TPack& data) noexcept : m_data(data) {}

   TPack m_data;
};
static_assert(std::is_standard_layout<Avx512f_32_Int>::value && std::is_trivially_copyable<Avx512f_32_Int>::value,
      "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

struct alignas(k_cAlignment) Avx512f_32_Float final {
   template<bool bNegateInput, bool bNaNPossible, bool bUnderflowPossible, bool bOverflowPossible>
   friend Avx512f_32_Float Exp(const Avx512f_32_Float& val) noexcept;
   template<bool bNegateOutput,
         bool bNaNPossible,
         bool bNegativePossible,
         bool bZeroPossible,
         bool bPositiveInfinityPossible>
   friend Avx512f_32_Float Log(const Avx512f_32_Float& val) noexcept;

   using T = float;
   using TPack = __m512;
   using TInt = Avx512f_32_Int;
   static_assert(std::is_same<FloatBig, T>::value || std::is_same<FloatSmall, T>::value,
         "T must be either FloatBig or FloatSmall");
   static constexpr AccelerationFlags k_zone = TInt::k_zone;
   static constexpr int k_cSIMDShift = TInt::k_cSIMDShift;
   static constexpr int k_cSIMDPack = TInt::k_cSIMDPack;
   static constexpr int k_cTypeShift = TInt::k_cTypeShift;
   static_assert(1 << k_cTypeShift == sizeof(T), "k_cTypeShift must be equivalent to the type size");

   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   inline Avx512f_32_Float() noexcept {}

   inline Avx512f_32_Float(const double val) noexcept : m_data(_mm512_set1_ps(static_cast<T>(val))) {}
   inline Avx512f_32_Float(const float val) noexcept : m_data(_mm512_set1_ps(static_cast<T>(val))) {}
   inline Avx512f_32_Float(const int val) noexcept : m_data(_mm512_set1_ps(static_cast<T>(val))) {}
   explicit Avx512f_32_Float(const Avx512f_32_Int& val) : m_data(_mm512_cvtepi32_ps(val.m_data)) {}

   inline Avx512f_32_Float operator+() const noexcept { return *this; }

   inline Avx512f_32_Float operator-() const noexcept {
      return Avx512f_32_Float(
            _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(m_data), _mm512_set1_epi32(0x80000000))));
   }

   inline Avx512f_32_Float operator+(const Avx512f_32_Float& other) const noexcept {
      return Avx512f_32_Float(_mm512_add_ps(m_data, other.m_data));
   }

   inline Avx512f_32_Float operator-(const Avx512f_32_Float& other) const noexcept {
      return Avx512f_32_Float(_mm512_sub_ps(m_data, other.m_data));
   }

   inline Avx512f_32_Float operator*(const Avx512f_32_Float& other) const noexcept {
      return Avx512f_32_Float(_mm512_mul_ps(m_data, other.m_data));
   }

   inline Avx512f_32_Float operator/(const Avx512f_32_Float& other) const noexcept {
      return Avx512f_32_Float(_mm512_div_ps(m_data, other.m_data));
   }

   inline Avx512f_32_Float& operator+=(const Avx512f_32_Float& other) noexcept {
      *this = (*this) + other;
      return *this;
   }

   inline Avx512f_32_Float& operator-=(const Avx512f_32_Float& other) noexcept {
      *this = (*this) - other;
      return *this;
   }

   inline Avx512f_32_Float& operator*=(const Avx512f_32_Float& other) noexcept {
      *this = (*this) * other;
      return *this;
   }

   inline Avx512f_32_Float& operator/=(const Avx512f_32_Float& other) noexcept {
      *this = (*this) / other;
      return *this;
   }

   friend inline Avx512f_32_Float operator+(const double val, const Avx512f_32_Float& other) noexcept {
      return Avx512f_32_Float(val) + other;
   }

   friend inline Avx512f_32_Float operator-(const double val, const Avx512f_32_Float& other) noexcept {
      return Avx512f_32_Float(val) - other;
   }

   friend inline Avx512f_32_Float operator*(const double val, const Avx512f_32_Float& other) noexcept {
      return Avx512f_32_Float(val) * other;
   }

   friend inline Avx512f_32_Float operator/(const double val, const Avx512f_32_Float& other) noexcept {
      return Avx512f_32_Float(val) / other;
   }

   friend inline Avx512f_32_Float operator+(const float val, const Avx512f_32_Float& other) noexcept {
      return Avx512f_32_Float(val) + other;
   }

   friend inline Avx512f_32_Float operator-(const float val, const Avx512f_32_Float& other) noexcept {
      return Avx512f_32_Float(val) - other;
   }

   friend inline Avx512f_32_Float operator*(const float val, const Avx512f_32_Float& other) noexcept {
      return Avx512f_32_Float(val) * other;
   }

   friend inline Avx512f_32_Float operator/(const float val, const Avx512f_32_Float& other) noexcept {
      return Avx512f_32_Float(val) / other;
   }

   friend inline __mmask16 operator==(const Avx512f_32_Float& left, const Avx512f_32_Float& right) noexcept {
      return _mm512_cmp_ps_mask(left.m_data, right.m_data, _CMP_EQ_OQ);
   }

   friend inline __mmask16 operator<(const Avx512f_32_Float& left, const Avx512f_32_Float& right) noexcept {
      return _mm512_cmp_ps_mask(left.m_data, right.m_data, _CMP_LT_OQ);
   }

   friend inline __mmask16 operator<=(const Avx512f_32_Float& left, const Avx512f_32_Float& right) noexcept {
      return _mm512_cmp_ps_mask(left.m_data, right.m_data, _CMP_LE_OQ);
   }

   inline static Avx512f_32_Float Load(const T* const a) noexcept { return Avx512f_32_Float(_mm512_load_ps(a)); }

   inline void Store(T* const a) const noexcept { _mm512_store_ps(a, m_data); }

   template<int cShift = k_cTypeShift> inline static Avx512f_32_Float Load(const T* const a, const TInt& i) noexcept {
      // i is treated as signed, so we should only use the lower 31 bits otherwise we'll read from memory before a
      static_assert(
            0 == cShift || 1 == cShift || 2 == cShift || 3 == cShift, "_mm512_i32gather_ps allows certain shift sizes");
      return Avx512f_32_Float(_mm512_i32gather_ps(i.m_data, a, 1 << cShift));
   }

   template<int cShift>
   inline static void DoubleLoad(
         const T* const a, const Avx512f_32_Int& i, Avx512f_32_Float& ret1, Avx512f_32_Float& ret2) noexcept {
      // i is treated as signed, so we should only use the lower 31 bits otherwise we'll read from memory before a
      static_assert(0 == cShift || 1 == cShift || 2 == cShift || 3 == cShift,
            "_mm256_i32gather_epi64 allows certain shift sizes");
      const __m256i i1 = _mm512_castsi512_si256(i.m_data);
      // we're purposely using the 64-bit double version of this because we want to fetch the gradient
      // and hessian together in one operation
      ret1 = Avx512f_32_Float(
            _mm512_castpd_ps(_mm512_i32gather_pd(i1, reinterpret_cast<const double*>(a), 1 << cShift)));
      const __m256i i2 = _mm256_castpd_si256(_mm512_extractf64x4_pd(_mm512_castsi512_pd(i.m_data), 1));
      ret2 = Avx512f_32_Float(
            _mm512_castpd_ps(_mm512_i32gather_pd(i2, reinterpret_cast<const double*>(a), 1 << cShift)));
   }

   template<int cShift = k_cTypeShift> inline void Store(T* const a, const TInt& i) const noexcept {
      // i is treated as signed, so we should only use the lower 31 bits otherwise we'll read from memory before a
      static_assert(0 == cShift || 1 == cShift || 2 == cShift || 3 == cShift,
            "_mm512_i32scatter_ps allows certain shift sizes");
      _mm512_i32scatter_ps(a, i.m_data, m_data, 1 << cShift);
   }

   template<int cShift>
   inline static void DoubleStore(
         T* const a, const TInt& i, const Avx512f_32_Float& val1, const Avx512f_32_Float& val2) noexcept {
      // i is treated as signed, so we should only use the lower 31 bits otherwise we'll read from memory before a

      static_assert(0 == cShift || 1 == cShift || 2 == cShift || 3 == cShift,
            "_mm512_i32scatter_pd allows certain shift sizes");

      const __m256i i1 = _mm512_castsi512_si256(i.m_data);
      _mm512_i32scatter_pd(a, i1, _mm512_castps_pd(val1.m_data), 1 << cShift);
      const __m256i i2 = _mm256_castpd_si256(_mm512_extractf64x4_pd(_mm512_castsi512_pd(i.m_data), 1));
      _mm512_i32scatter_pd(a, i2, _mm512_castps_pd(val2.m_data), 1 << cShift);
   }

   inline static void Interleaf(const Avx512f_32_Float& val0,
         const Avx512f_32_Float& val1,
         Avx512f_32_Float& ret0,
         Avx512f_32_Float& ret1) noexcept {
      // this function permutes the values into positions that the PermuteForInterleaf function expects
      // but for any SIMD implementation, the positions can be variable as long as they work together
      ret0 = Avx512f_32_Float(_mm512_unpacklo_ps(val0.m_data, val1.m_data));
      ret1 = Avx512f_32_Float(_mm512_unpackhi_ps(val0.m_data, val1.m_data));
   }

   template<typename TFunc>
   friend inline Avx512f_32_Float ApplyFunc(const TFunc& func, const Avx512f_32_Float& val) noexcept {
      alignas(k_cAlignment) T aTemp[k_cSIMDPack];
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

   template<typename TFunc> static inline void Execute(const TFunc& func) noexcept {
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

   template<typename TFunc> static inline void Execute(const TFunc& func, const Avx512f_32_Float& val0) noexcept {
      alignas(k_cAlignment) T a0[k_cSIMDPack];
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
   static inline void Execute(const TFunc& func, const Avx512f_32_Float& val0, const Avx512f_32_Float& val1) noexcept {
      alignas(k_cAlignment) T a0[k_cSIMDPack];
      val0.Store(a0);
      alignas(k_cAlignment) T a1[k_cSIMDPack];
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

   template<typename TFunc>
   static inline void Execute(const TFunc& func, const Avx512f_32_Int& val0, const Avx512f_32_Float& val1) noexcept {
      alignas(k_cAlignment) TInt::T a0[k_cSIMDPack];
      val0.Store(a0);
      alignas(k_cAlignment) T a1[k_cSIMDPack];
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

   template<typename TFunc>
   static inline void Execute(const TFunc& func,
         const Avx512f_32_Int& val0,
         const Avx512f_32_Float& val1,
         const Avx512f_32_Float& val2) noexcept {
      alignas(k_cAlignment) TInt::T a0[k_cSIMDPack];
      val0.Store(a0);
      alignas(k_cAlignment) T a1[k_cSIMDPack];
      val1.Store(a1);
      alignas(k_cAlignment) T a2[k_cSIMDPack];
      val2.Store(a2);

      func(0, a0[0], a1[0], a2[0]);
      func(1, a0[1], a1[1], a2[1]);
      func(2, a0[2], a1[2], a2[2]);
      func(3, a0[3], a1[3], a2[3]);
      func(4, a0[4], a1[4], a2[4]);
      func(5, a0[5], a1[5], a2[5]);
      func(6, a0[6], a1[6], a2[6]);
      func(7, a0[7], a1[7], a2[7]);
      func(8, a0[8], a1[8], a2[8]);
      func(9, a0[9], a1[9], a2[9]);
      func(10, a0[10], a1[10], a2[10]);
      func(11, a0[11], a1[11], a2[11]);
      func(12, a0[12], a1[12], a2[12]);
      func(13, a0[13], a1[13], a2[13]);
      func(14, a0[14], a1[14], a2[14]);
      func(15, a0[15], a1[15], a2[15]);
   }

   template<typename TFunc>
   static inline void Execute(const TFunc& func,
         const Avx512f_32_Int& val0,
         const Avx512f_32_Float& val1,
         const Avx512f_32_Float& val2,
         const Avx512f_32_Float& val3) noexcept {
      alignas(k_cAlignment) TInt::T a0[k_cSIMDPack];
      val0.Store(a0);
      alignas(k_cAlignment) T a1[k_cSIMDPack];
      val1.Store(a1);
      alignas(k_cAlignment) T a2[k_cSIMDPack];
      val2.Store(a2);
      alignas(k_cAlignment) T a3[k_cSIMDPack];
      val3.Store(a3);

      func(0, a0[0], a1[0], a2[0], a3[0]);
      func(1, a0[1], a1[1], a2[1], a3[1]);
      func(2, a0[2], a1[2], a2[2], a3[2]);
      func(3, a0[3], a1[3], a2[3], a3[3]);
      func(4, a0[4], a1[4], a2[4], a3[4]);
      func(5, a0[5], a1[5], a2[5], a3[5]);
      func(6, a0[6], a1[6], a2[6], a3[6]);
      func(7, a0[7], a1[7], a2[7], a3[7]);
      func(8, a0[8], a1[8], a2[8], a3[8]);
      func(9, a0[9], a1[9], a2[9], a3[9]);
      func(10, a0[10], a1[10], a2[10], a3[10]);
      func(11, a0[11], a1[11], a2[11], a3[11]);
      func(12, a0[12], a1[12], a2[12], a3[12]);
      func(13, a0[13], a1[13], a2[13], a3[13]);
      func(14, a0[14], a1[14], a2[14], a3[14]);
      func(15, a0[15], a1[15], a2[15], a3[15]);
   }

   template<typename TFunc>
   static inline void Execute(const TFunc& func,
         const Avx512f_32_Int& val0,
         const Avx512f_32_Int& val1,
         const Avx512f_32_Float& val2,
         const Avx512f_32_Float& val3) noexcept {
      alignas(k_cAlignment) TInt::T a0[k_cSIMDPack];
      val0.Store(a0);
      alignas(k_cAlignment) TInt::T a1[k_cSIMDPack];
      val1.Store(a1);
      alignas(k_cAlignment) T a2[k_cSIMDPack];
      val2.Store(a2);
      alignas(k_cAlignment) T a3[k_cSIMDPack];
      val3.Store(a3);

      func(0, a0[0], a1[0], a2[0], a3[0]);
      func(1, a0[1], a1[1], a2[1], a3[1]);
      func(2, a0[2], a1[2], a2[2], a3[2]);
      func(3, a0[3], a1[3], a2[3], a3[3]);
      func(4, a0[4], a1[4], a2[4], a3[4]);
      func(5, a0[5], a1[5], a2[5], a3[5]);
      func(6, a0[6], a1[6], a2[6], a3[6]);
      func(7, a0[7], a1[7], a2[7], a3[7]);
      func(8, a0[8], a1[8], a2[8], a3[8]);
      func(9, a0[9], a1[9], a2[9], a3[9]);
      func(10, a0[10], a1[10], a2[10], a3[10]);
      func(11, a0[11], a1[11], a2[11], a3[11]);
      func(12, a0[12], a1[12], a2[12], a3[12]);
      func(13, a0[13], a1[13], a2[13], a3[13]);
      func(14, a0[14], a1[14], a2[14], a3[14]);
      func(15, a0[15], a1[15], a2[15], a3[15]);
   }

   template<typename TFunc>
   static inline void Execute(const TFunc& func,
         const Avx512f_32_Int& val0,
         const Avx512f_32_Int& val1,
         const Avx512f_32_Float& val2,
         const Avx512f_32_Float& val3,
         const Avx512f_32_Float& val4) noexcept {
      alignas(k_cAlignment) TInt::T a0[k_cSIMDPack];
      val0.Store(a0);
      alignas(k_cAlignment) TInt::T a1[k_cSIMDPack];
      val1.Store(a1);
      alignas(k_cAlignment) T a2[k_cSIMDPack];
      val2.Store(a2);
      alignas(k_cAlignment) T a3[k_cSIMDPack];
      val3.Store(a3);
      alignas(k_cAlignment) T a4[k_cSIMDPack];
      val4.Store(a4);

      func(0, a0[0], a1[0], a2[0], a3[0], a4[0]);
      func(1, a0[1], a1[1], a2[1], a3[1], a4[1]);
      func(2, a0[2], a1[2], a2[2], a3[2], a4[2]);
      func(3, a0[3], a1[3], a2[3], a3[3], a4[3]);
      func(4, a0[4], a1[4], a2[4], a3[4], a4[4]);
      func(5, a0[5], a1[5], a2[5], a3[5], a4[5]);
      func(6, a0[6], a1[6], a2[6], a3[6], a4[6]);
      func(7, a0[7], a1[7], a2[7], a3[7], a4[7]);
      func(8, a0[8], a1[8], a2[8], a3[8], a4[8]);
      func(9, a0[9], a1[9], a2[9], a3[9], a4[9]);
      func(10, a0[10], a1[10], a2[10], a3[10], a4[10]);
      func(11, a0[11], a1[11], a2[11], a3[11], a4[11]);
      func(12, a0[12], a1[12], a2[12], a3[12], a4[12]);
      func(13, a0[13], a1[13], a2[13], a3[13], a4[13]);
      func(14, a0[14], a1[14], a2[14], a3[14], a4[14]);
      func(15, a0[15], a1[15], a2[15], a3[15], a4[15]);
   }

   friend inline Avx512f_32_Float IfThenElse(
         const __mmask16& cmp, const Avx512f_32_Float& trueVal, const Avx512f_32_Float& falseVal) noexcept {
      return Avx512f_32_Float(_mm512_mask_blend_ps(cmp, falseVal.m_data, trueVal.m_data));
   }

   friend inline Avx512f_32_Float IfAdd(
         const __mmask16& cmp, const Avx512f_32_Float& base, const Avx512f_32_Float& addend) noexcept {
      return Avx512f_32_Float(_mm512_mask_add_ps(base.m_data, cmp, base.m_data, addend.m_data));
   }

   friend inline __mmask16 IsNaN(const Avx512f_32_Float& cmp) noexcept {
      return _mm512_cmp_ps_mask(cmp.m_data, cmp.m_data, _CMP_UNORD_Q);
   }

   static inline Avx512f_32_Int ReinterpretInt(const Avx512f_32_Float& val) noexcept {
      return Avx512f_32_Int(_mm512_castps_si512(val.m_data));
   }

   static inline Avx512f_32_Float ReinterpretFloat(const Avx512f_32_Int& val) noexcept {
      return Avx512f_32_Float(_mm512_castsi512_ps(val.m_data));
   }

   friend inline Avx512f_32_Float Round(const Avx512f_32_Float& val) noexcept {
      return Avx512f_32_Float(_mm512_roundscale_ps(val.m_data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
   }

   friend inline Avx512f_32_Float Abs(const Avx512f_32_Float& val) noexcept {
      return Avx512f_32_Float(
            _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(val.m_data), _mm512_set1_epi32(0x7FFFFFFF))));
   }

   friend inline Avx512f_32_Float FastApproxReciprocal(const Avx512f_32_Float& val) noexcept {
#ifdef FAST_DIVISION
      return Avx512f_32_Float(_mm512_rcp14_ps(val.m_data));
#else // FAST_DIVISION
      return Avx512f_32_Float(1.0) / val;
#endif // FAST_DIVISION
   }

   friend inline Avx512f_32_Float FastApproxDivide(
         const Avx512f_32_Float& dividend, const Avx512f_32_Float& divisor) noexcept {
#ifdef FAST_DIVISION
      return dividend * FastApproxReciprocal(divisor);
#else // FAST_DIVISION
      return dividend / divisor;
#endif // FAST_DIVISION
   }

   friend inline Avx512f_32_Float FusedMultiplyAdd(
         const Avx512f_32_Float& mul1, const Avx512f_32_Float& mul2, const Avx512f_32_Float& add) noexcept {
      return Avx512f_32_Float(_mm512_fmadd_ps(mul1.m_data, mul2.m_data, add.m_data));
   }

   friend inline Avx512f_32_Float FusedNegateMultiplyAdd(
         const Avx512f_32_Float& mul1, const Avx512f_32_Float& mul2, const Avx512f_32_Float& add) noexcept {
      // equivalent to: -(mul1 * mul2) + add
      return Avx512f_32_Float(_mm512_fnmadd_ps(mul1.m_data, mul2.m_data, add.m_data));
   }

   friend inline Avx512f_32_Float FusedMultiplySubtract(
         const Avx512f_32_Float& mul1, const Avx512f_32_Float& mul2, const Avx512f_32_Float& subtract) noexcept {
      // equivalent to: mul1 * mul2 - subtract
      return Avx512f_32_Float(_mm512_fmsub_ps(mul1.m_data, mul2.m_data, subtract.m_data));
   }

   friend inline Avx512f_32_Float Sqrt(const Avx512f_32_Float& val) noexcept {
      return Avx512f_32_Float(_mm512_sqrt_ps(val.m_data));
   }

   template<bool bUseApprox,
         bool bNegateInput = false,
         bool bNaNPossible = true,
         bool bUnderflowPossible = true,
         bool bOverflowPossible = true,
         bool bSpecialCaseZero = false,
         typename std::enable_if<!bUseApprox, int>::type = 0>
   static inline Avx512f_32_Float ApproxExp(const Avx512f_32_Float& val,
         const int32_t addExpSchraudolphTerm = k_expTermZeroMeanErrorForSoftmaxWithZeroedLogit) noexcept {
      UNUSED(addExpSchraudolphTerm);
      return Exp<bNegateInput, bNaNPossible, bUnderflowPossible, bOverflowPossible>(val);
   }

   template<bool bUseApprox,
         bool bNegateInput = false,
         bool bNaNPossible = true,
         bool bUnderflowPossible = true,
         bool bOverflowPossible = true,
         bool bSpecialCaseZero = false,
         typename std::enable_if<bUseApprox, int>::type = 0>
   static inline Avx512f_32_Float ApproxExp(const Avx512f_32_Float& val,
         const int32_t addExpSchraudolphTerm = k_expTermZeroMeanErrorForSoftmaxWithZeroedLogit) noexcept {
      // This code will make no sense until you read the Nicol N. Schraudolph paper:
      // https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.4508&rep=rep1&type=pdf
      // and also see approximate_math.hpp
      static constexpr float signedExpMultiple = bNegateInput ? -k_expMultiple : k_expMultiple;
#ifdef EXP_INT_SIMD
      const __m512 product = (val * signedExpMultiple).m_data;
      const __m512i retInt = _mm512_add_epi32(_mm512_cvttps_epi32(product), _mm512_set1_epi32(addExpSchraudolphTerm));
#else // EXP_INT_SIMD
      const __m512 retFloat = FusedMultiplyAdd(val, signedExpMultiple, static_cast<T>(addExpSchraudolphTerm)).m_data;
      const __m512i retInt = _mm512_cvttps_epi32(retFloat);
#endif // EXP_INT_SIMD
      Avx512f_32_Float result = Avx512f_32_Float(_mm512_castsi512_ps(retInt));
      if(bSpecialCaseZero) {
         result = IfThenElse(0.0 == val, 1.0, result);
      }
      if(bOverflowPossible) {
         if(bNegateInput) {
            result = IfThenElse(val < static_cast<T>(-k_expOverflowPoint), std::numeric_limits<T>::infinity(), result);
         } else {
            result = IfThenElse(static_cast<T>(k_expOverflowPoint) < val, std::numeric_limits<T>::infinity(), result);
         }
      }
      if(bUnderflowPossible) {
         if(bNegateInput) {
            result = IfThenElse(static_cast<T>(-k_expUnderflowPoint) < val, 0.0, result);
         } else {
            result = IfThenElse(val < static_cast<T>(k_expUnderflowPoint), 0.0, result);
         }
      }
      if(bNaNPossible) {
         result = IfThenElse(IsNaN(val), val, result);
      }
      return result;
   }

   template<bool bUseApprox,
         bool bNegateOutput = false,
         bool bNaNPossible = true,
         bool bNegativePossible = true,
         bool bZeroPossible = true, // if false, positive zero returns a big negative number, negative zero returns a
                                    // big positive number
         bool bPositiveInfinityPossible = true, // if false, +inf returns a big positive number.  If val can be a
                                                // double that is above the largest representable float, then setting
                                                // this is necessary to avoid undefined behavior
         typename std::enable_if<!bUseApprox, int>::type = 0>
   static inline Avx512f_32_Float ApproxLog(
         const Avx512f_32_Float& val, const float addLogSchraudolphTerm = k_logTermLowerBoundInputCloseToOne) noexcept {
      UNUSED(addLogSchraudolphTerm);
      return Log<bNegateOutput, bNaNPossible, bNegativePossible, bZeroPossible, bPositiveInfinityPossible>(val);
   }

   template<bool bUseApprox,
         bool bNegateOutput = false,
         bool bNaNPossible = true,
         bool bNegativePossible = true,
         bool bZeroPossible = true, // if false, positive zero returns a big negative number, negative zero returns a
                                    // big positive number
         bool bPositiveInfinityPossible = true, // if false, +inf returns a big positive number.  If val can be a
                                                // double that is above the largest representable float, then setting
                                                // this is necessary to avoid undefined behavior
         typename std::enable_if<bUseApprox, int>::type = 0>
   static inline Avx512f_32_Float ApproxLog(
         const Avx512f_32_Float& val, const float addLogSchraudolphTerm = k_logTermLowerBoundInputCloseToOne) noexcept {
      // This code will make no sense until you read the Nicol N. Schraudolph paper:
      // https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.4508&rep=rep1&type=pdf
      // and also see approximate_math.hpp
      const __m512i retInt = _mm512_castps_si512(val.m_data);
      Avx512f_32_Float result = Avx512f_32_Float(_mm512_cvtepi32_ps(retInt));
      if(bNaNPossible) {
         if(bPositiveInfinityPossible) {
            result = IfThenElse(val < std::numeric_limits<T>::infinity(), result, val);
         } else {
            result = IfThenElse(IsNaN(val), val, result);
         }
      } else {
         if(bPositiveInfinityPossible) {
            result = IfThenElse(std::numeric_limits<T>::infinity() == val, val, result);
         }
      }
      if(bNegateOutput) {
         result = FusedMultiplyAdd(result, -k_logMultiple, -addLogSchraudolphTerm);
      } else {
         result = FusedMultiplyAdd(result, k_logMultiple, addLogSchraudolphTerm);
      }
      if(bZeroPossible) {
         result = IfThenElse(val < std::numeric_limits<T>::min(),
               bNegateOutput ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity(),
               result);
      }
      if(bNegativePossible) {
         result = IfThenElse(val < T{0}, std::numeric_limits<T>::quiet_NaN(), result);
      }
      return result;
   }

   friend inline T Sum(const Avx512f_32_Float& val) noexcept { return _mm512_reduce_add_ps(val.m_data); }

   template<typename TObjective,
         bool bCollapsed,
         bool bValidation,
         bool bWeight,
         bool bHessian,
         bool bUseApprox,
         size_t cCompilerScores>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorApplyUpdate(
         const Objective* const pObjective, ApplyUpdateBridge* const pData) noexcept {
      RemoteApplyUpdate<TObjective, bCollapsed, bValidation, bWeight, bHessian, bUseApprox, cCompilerScores>(
            pObjective, pData);
      return Error_None;
   }

   template<bool bHessian, bool bWeight, bool bCollapsed, size_t cCompilerScores, bool bParallel>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorBinSumsBoosting(BinSumsBoostingBridge* const pParams) noexcept {
      RemoteBinSumsBoosting<Avx512f_32_Float, bHessian, bWeight, bCollapsed, cCompilerScores, bParallel>(pParams);
      return Error_None;
   }

   template<bool bHessian, bool bWeight, size_t cCompilerScores, size_t cCompilerDimensions>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorBinSumsInteraction(
         BinSumsInteractionBridge* const pParams) noexcept {
      RemoteBinSumsInteraction<Avx512f_32_Float, bHessian, bWeight, cCompilerScores, cCompilerDimensions>(pParams);
      return Error_None;
   }

 private:
   inline Avx512f_32_Float(const TPack& data) noexcept : m_data(data) {}

   TPack m_data;
};
static_assert(std::is_standard_layout<Avx512f_32_Float>::value && std::is_trivially_copyable<Avx512f_32_Float>::value,
      "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

template<bool bNegateInput, bool bNaNPossible, bool bUnderflowPossible, bool bOverflowPossible>
inline Avx512f_32_Float Exp(const Avx512f_32_Float& val) noexcept {
   return Exp32<Avx512f_32_Float, bNegateInput, bNaNPossible, bUnderflowPossible, bOverflowPossible>(val);
}

template<bool bNegateOutput,
      bool bNaNPossible,
      bool bNegativePossible,
      bool bZeroPossible,
      bool bPositiveInfinityPossible>
inline Avx512f_32_Float Log(const Avx512f_32_Float& val) noexcept {
   return Log32<Avx512f_32_Float,
         bNegateOutput,
         bNaNPossible,
         bNegativePossible,
         bZeroPossible,
         bPositiveInfinityPossible>(val);
}

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm ApplyUpdate_Avx512f_32(
      const ObjectiveWrapper* const pObjectiveWrapper, ApplyUpdateBridge* const pData) {
   const Objective* const pObjective = static_cast<const Objective*>(pObjectiveWrapper->m_pObjective);
   const APPLY_UPDATE_CPP pApplyUpdateCpp =
         (static_cast<FunctionPointersCpp*>(pObjectiveWrapper->m_pFunctionPointersCpp))->m_pApplyUpdateCpp;

   // all our memory should be aligned. It is required by SIMD for correctness or performance
   EBM_ASSERT(IsAligned(pData->m_aMulticlassMidwayTemp));
   EBM_ASSERT(IsAligned(pData->m_aUpdateTensorScores));
   EBM_ASSERT(IsAligned(pData->m_aPacked));
   EBM_ASSERT(IsAligned(pData->m_aTargets));
   EBM_ASSERT(IsAligned(pData->m_aWeights));
   EBM_ASSERT(IsAligned(pData->m_aSampleScores));
   EBM_ASSERT(IsAligned(pData->m_aGradientsAndHessians));

   return (*pApplyUpdateCpp)(pObjective, pData);
}

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm BinSumsBoosting_Avx512f_32(
      const ObjectiveWrapper* const pObjectiveWrapper, BinSumsBoostingBridge* const pParams) {
   const BIN_SUMS_BOOSTING_CPP pBinSumsBoostingCpp =
         (static_cast<FunctionPointersCpp*>(pObjectiveWrapper->m_pFunctionPointersCpp))->m_pBinSumsBoostingCpp;

   // all our memory should be aligned. It is required by SIMD for correctness or performance
   EBM_ASSERT(IsAligned(pParams->m_aGradientsAndHessians));
   EBM_ASSERT(IsAligned(pParams->m_aWeights));
   EBM_ASSERT(IsAligned(pParams->m_aPacked));
   EBM_ASSERT(IsAligned(pParams->m_aFastBins));

   return (*pBinSumsBoostingCpp)(pParams);
}

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm BinSumsInteraction_Avx512f_32(
      const ObjectiveWrapper* const pObjectiveWrapper, BinSumsInteractionBridge* const pParams) {
   const BIN_SUMS_INTERACTION_CPP pBinSumsInteractionCpp =
         (static_cast<FunctionPointersCpp*>(pObjectiveWrapper->m_pFunctionPointersCpp))->m_pBinSumsInteractionCpp;

#ifndef NDEBUG
   // all our memory should be aligned. It is required by SIMD for correctness or performance
   EBM_ASSERT(IsAligned(pParams->m_aGradientsAndHessians));
   EBM_ASSERT(IsAligned(pParams->m_aWeights));
   EBM_ASSERT(IsAligned(pParams->m_aFastBins));
   for(size_t iDebug = 0; iDebug < pParams->m_cRuntimeRealDimensions; ++iDebug) {
      EBM_ASSERT(IsAligned(pParams->m_aaPacked[iDebug]));
   }
#endif // NDEBUG

   return (*pBinSumsInteractionCpp)(pParams);
}

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm CreateObjective_Avx512f_32(const Config* const pConfig,
      const char* const sObjective,
      const char* const sObjectiveEnd,
      ObjectiveWrapper* const pObjectiveWrapperOut) {
   pObjectiveWrapperOut->m_pApplyUpdateC = ApplyUpdate_Avx512f_32;
   pObjectiveWrapperOut->m_pBinSumsBoostingC = BinSumsBoosting_Avx512f_32;
   pObjectiveWrapperOut->m_pBinSumsInteractionC = BinSumsInteraction_Avx512f_32;
   ErrorEbm error = ComputeWrapper<Avx512f_32_Float>::FillWrapper(pObjectiveWrapperOut);
   if(Error_None != error) {
      return error;
   }
   return Objective::CreateObjective<Avx512f_32_Float>(pConfig, sObjective, sObjectiveEnd, pObjectiveWrapperOut);
}

} // namespace DEFINED_ZONE_NAME

#endif // BRIDGE_AVX512F_32
