// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifdef BRIDGE_AVX2_32

#define _CRT_SECURE_NO_DEPRECATE

#include <cmath> // exp, log
#include <limits> // numeric_limits
#include <type_traits> // is_unsigned
#include <immintrin.h> // SIMD.  Do not include in pch.hpp!

#include "libebm.h"
#include "logging.h"
#include "unzoned.h"

#define ZONE_avx2
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

static constexpr size_t k_cAlignment = 32;
struct alignas(k_cAlignment) Avx2_32_Float;
struct alignas(k_cAlignment) Avx2_32_Int;

template<bool bNegateInput = false,
      bool bNaNPossible = true,
      bool bUnderflowPossible = true,
      bool bOverflowPossible = true>
inline Avx2_32_Float Exp(const Avx2_32_Float& val) noexcept;
template<bool bNegateOutput = false,
      bool bNaNPossible = true,
      bool bNegativePossible = true,
      bool bZeroPossible = true,
      bool bPositiveInfinityPossible = true>
inline Avx2_32_Float Log(const Avx2_32_Float& val) noexcept;

// this is super-special and included inside the zone namespace
#include "objective_registrations.hpp"

struct alignas(k_cAlignment) Avx2_32_Int final {
   friend Avx2_32_Float;

   using T = uint32_t;
   using TPack = __m256i;
   static_assert(std::is_unsigned<T>::value, "T must be an unsigned integer type");
   static_assert(
         std::is_same<UIntBig, T>::value || std::is_same<UIntSmall, T>::value, "T must be either UIntBig or UIntSmall");
   static constexpr AccelerationFlags k_zone = AccelerationFlags_AVX2;
   static constexpr int k_cSIMDShift = 3;
   static constexpr int k_cSIMDPack = 1 << k_cSIMDShift;
   static constexpr int k_cTypeShift = 2;
   static_assert(1 << k_cTypeShift == sizeof(T), "k_cTypeShift must be equivalent to the type size");

   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   inline Avx2_32_Int() noexcept {}

   inline Avx2_32_Int(const T& val) noexcept : m_data(_mm256_set1_epi32(val)) {}

   inline static Avx2_32_Int Load(const T* const a) noexcept {
      return Avx2_32_Int(_mm256_load_si256(reinterpret_cast<const TPack*>(a)));
   }

   inline void Store(T* const a) const noexcept { _mm256_store_si256(reinterpret_cast<TPack*>(a), m_data); }

   inline static Avx2_32_Int LoadBytes(const uint8_t* const a) noexcept {
      return Avx2_32_Int(_mm256_cvtepu8_epi32(_mm_loadu_si64(a)));
   }

   template<typename TFunc> static inline void Execute(const TFunc& func, const Avx2_32_Int& val0) noexcept {
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
   }

   inline static Avx2_32_Int MakeIndexes() noexcept { return Avx2_32_Int(_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0)); }

   inline Avx2_32_Int operator~() const noexcept {
      return Avx2_32_Int(_mm256_xor_si256(m_data, _mm256_set1_epi32(-1)));
   }

   friend inline Avx2_32_Int operator==(const Avx2_32_Int& left, const Avx2_32_Int& right) noexcept {
      return Avx2_32_Int(_mm256_cmpeq_epi32(left.m_data, right.m_data));
   }

   inline Avx2_32_Int operator+(const Avx2_32_Int& other) const noexcept {
      return Avx2_32_Int(_mm256_add_epi32(m_data, other.m_data));
   }

   inline Avx2_32_Int operator-(const Avx2_32_Int& other) const noexcept {
      return Avx2_32_Int(_mm256_sub_epi32(m_data, other.m_data));
   }

   inline Avx2_32_Int operator*(const T& other) const noexcept {
      return Avx2_32_Int(_mm256_mullo_epi32(m_data, _mm256_set1_epi32(other)));
   }

   inline Avx2_32_Int operator>>(int shift) const noexcept { return Avx2_32_Int(_mm256_srli_epi32(m_data, shift)); }

   inline Avx2_32_Int operator<<(int shift) const noexcept { return Avx2_32_Int(_mm256_slli_epi32(m_data, shift)); }

   inline Avx2_32_Int operator&(const Avx2_32_Int& other) const noexcept {
      return Avx2_32_Int(_mm256_and_si256(m_data, other.m_data));
   }

   inline Avx2_32_Int operator|(const Avx2_32_Int& other) const noexcept {
      return Avx2_32_Int(_mm256_or_si256(m_data, other.m_data));
   }

   friend inline Avx2_32_Int IfThenElse(
         const Avx2_32_Int& cmp, const Avx2_32_Int& trueVal, const Avx2_32_Int& falseVal) noexcept {
      return Avx2_32_Int(_mm256_blendv_epi8(falseVal.m_data, trueVal.m_data, cmp.m_data));
   }

   friend inline Avx2_32_Int IfAdd(
         const Avx2_32_Int& cmp, const Avx2_32_Int& base, const Avx2_32_Int& addend) noexcept {
      return base + (cmp & addend);
   }

   friend inline Avx2_32_Int PermuteForInterleaf(const Avx2_32_Int& val) noexcept {
      // this function permutes the values into positions that the Interleaf function expects
      // but for any SIMD implementation the positions can be variable as long as they work together

      // TODO: we might be able to move this operation to where we store the packed indexes so that
      // it doesn't need to execute in the tight loop

      return Avx2_32_Int(_mm256_permutevar8x32_epi32(val.m_data, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7)));
   }

 private:
   inline Avx2_32_Int(const TPack& data) noexcept : m_data(data) {}

   TPack m_data;
};
static_assert(std::is_standard_layout<Avx2_32_Int>::value && std::is_trivially_copyable<Avx2_32_Int>::value,
      "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

struct alignas(k_cAlignment) Avx2_32_Float final {
   template<bool bNegateInput, bool bNaNPossible, bool bUnderflowPossible, bool bOverflowPossible>
   friend Avx2_32_Float Exp(const Avx2_32_Float& val) noexcept;
   template<bool bNegateOutput,
         bool bNaNPossible,
         bool bNegativePossible,
         bool bZeroPossible,
         bool bPositiveInfinityPossible>
   friend Avx2_32_Float Log(const Avx2_32_Float& val) noexcept;

   using T = float;
   using TPack = __m256;
   using TInt = Avx2_32_Int;
   static_assert(std::is_same<FloatBig, T>::value || std::is_same<FloatSmall, T>::value,
         "T must be either FloatBig or FloatSmall");
   static constexpr AccelerationFlags k_zone = TInt::k_zone;
   static constexpr int k_cSIMDShift = TInt::k_cSIMDShift;
   static constexpr int k_cSIMDPack = TInt::k_cSIMDPack;
   static constexpr int k_cTypeShift = TInt::k_cTypeShift;
   static_assert(1 << k_cTypeShift == sizeof(T), "k_cTypeShift must be equivalent to the type size");

   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   inline Avx2_32_Float() noexcept {}

   inline Avx2_32_Float(const double val) noexcept : m_data(_mm256_set1_ps(static_cast<T>(val))) {}
   inline Avx2_32_Float(const float val) noexcept : m_data(_mm256_set1_ps(static_cast<T>(val))) {}
   inline Avx2_32_Float(const int val) noexcept : m_data(_mm256_set1_ps(static_cast<T>(val))) {}
   explicit Avx2_32_Float(const Avx2_32_Int& val) : m_data(_mm256_cvtepi32_ps(val.m_data)) {}

   inline Avx2_32_Float operator+() const noexcept { return *this; }

   inline Avx2_32_Float operator-() const noexcept {
      return Avx2_32_Float(
            _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(m_data), _mm256_set1_epi32(0x80000000))));
   }

   inline Avx2_32_Float operator+(const Avx2_32_Float& other) const noexcept {
      return Avx2_32_Float(_mm256_add_ps(m_data, other.m_data));
   }

   inline Avx2_32_Float operator-(const Avx2_32_Float& other) const noexcept {
      return Avx2_32_Float(_mm256_sub_ps(m_data, other.m_data));
   }

   inline Avx2_32_Float operator*(const Avx2_32_Float& other) const noexcept {
      return Avx2_32_Float(_mm256_mul_ps(m_data, other.m_data));
   }

   inline Avx2_32_Float operator/(const Avx2_32_Float& other) const noexcept {
      return Avx2_32_Float(_mm256_div_ps(m_data, other.m_data));
   }

   inline Avx2_32_Float& operator+=(const Avx2_32_Float& other) noexcept {
      *this = (*this) + other;
      return *this;
   }

   inline Avx2_32_Float& operator-=(const Avx2_32_Float& other) noexcept {
      *this = (*this) - other;
      return *this;
   }

   inline Avx2_32_Float& operator*=(const Avx2_32_Float& other) noexcept {
      *this = (*this) * other;
      return *this;
   }

   inline Avx2_32_Float& operator/=(const Avx2_32_Float& other) noexcept {
      *this = (*this) / other;
      return *this;
   }

   friend inline Avx2_32_Float operator+(const double val, const Avx2_32_Float& other) noexcept {
      return Avx2_32_Float(val) + other;
   }

   friend inline Avx2_32_Float operator-(const double val, const Avx2_32_Float& other) noexcept {
      return Avx2_32_Float(val) - other;
   }

   friend inline Avx2_32_Float operator*(const double val, const Avx2_32_Float& other) noexcept {
      return Avx2_32_Float(val) * other;
   }

   friend inline Avx2_32_Float operator/(const double val, const Avx2_32_Float& other) noexcept {
      return Avx2_32_Float(val) / other;
   }

   friend inline Avx2_32_Float operator+(const float val, const Avx2_32_Float& other) noexcept {
      return Avx2_32_Float(val) + other;
   }

   friend inline Avx2_32_Float operator-(const float val, const Avx2_32_Float& other) noexcept {
      return Avx2_32_Float(val) - other;
   }

   friend inline Avx2_32_Float operator*(const float val, const Avx2_32_Float& other) noexcept {
      return Avx2_32_Float(val) * other;
   }

   friend inline Avx2_32_Float operator/(const float val, const Avx2_32_Float& other) noexcept {
      return Avx2_32_Float(val) / other;
   }

   friend inline Avx2_32_Int operator==(const Avx2_32_Float& left, const Avx2_32_Float& right) noexcept {
      return ReinterpretInt(Avx2_32_Float(_mm256_cmp_ps(left.m_data, right.m_data, _CMP_EQ_OQ)));
   }

   friend inline Avx2_32_Int operator<(const Avx2_32_Float& left, const Avx2_32_Float& right) noexcept {
      return ReinterpretInt(Avx2_32_Float(_mm256_cmp_ps(left.m_data, right.m_data, _CMP_LT_OQ)));
   }

   friend inline Avx2_32_Int operator<=(const Avx2_32_Float& left, const Avx2_32_Float& right) noexcept {
      return ReinterpretInt(Avx2_32_Float(_mm256_cmp_ps(left.m_data, right.m_data, _CMP_LE_OQ)));
   }

   inline static Avx2_32_Float Load(const T* const a) noexcept { return Avx2_32_Float(_mm256_load_ps(a)); }

   inline void Store(T* const a) const noexcept { _mm256_store_ps(a, m_data); }

   template<int cShift = k_cTypeShift> inline static Avx2_32_Float Load(const T* const a, const TInt& i) noexcept {
      // i is treated as signed, so we should only use the lower 31 bits otherwise we'll read from memory before a
      static_assert(
            0 == cShift || 1 == cShift || 2 == cShift || 3 == cShift, "_mm256_i32gather_ps allows certain shift sizes");
      return Avx2_32_Float(_mm256_i32gather_ps(a, i.m_data, 1 << cShift));
   }

   template<int cShift>
   inline static void DoubleLoad(
         const T* const a, const Avx2_32_Int& i, Avx2_32_Float& ret1, Avx2_32_Float& ret2) noexcept {
      // i is treated as signed, so we should only use the lower 31 bits otherwise we'll read from memory before a
      static_assert(0 == cShift || 1 == cShift || 2 == cShift || 3 == cShift,
            "_mm256_i32gather_epi64 allows certain shift sizes");
      const __m128i i1 = _mm256_castsi256_si128(i.m_data);
      // we're purposely using the 64-bit double version of this because we want to fetch the gradient
      // and hessian together in one operation
      ret1 = Avx2_32_Float(_mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(a), i1, 1 << cShift)));
      const __m128i i2 = _mm256_extracti128_si256(i.m_data, 1);
      ret2 = Avx2_32_Float(_mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(a), i2, 1 << cShift)));
   }

   template<int cShift = k_cTypeShift> inline void Store(T* const a, const TInt& i) const noexcept {
      alignas(k_cAlignment) TInt::T ints[k_cSIMDPack];
      alignas(k_cAlignment) T floats[k_cSIMDPack];

      i.Store(ints);
      Store(floats);

      // if we shifted ints[] without converting to size_t first the compiler cannot
      // use the built in index shifting because ints could be 32 bits and shifting
      // right would chop off some bits, but when converted to size_t first then
      // that isn't an issue so the compiler can optimize the shift away and incorporate
      // it into the store assembly instruction
      *IndexByte(a, static_cast<size_t>(ints[0]) << cShift) = floats[0];
      *IndexByte(a, static_cast<size_t>(ints[1]) << cShift) = floats[1];
      *IndexByte(a, static_cast<size_t>(ints[2]) << cShift) = floats[2];
      *IndexByte(a, static_cast<size_t>(ints[3]) << cShift) = floats[3];
      *IndexByte(a, static_cast<size_t>(ints[4]) << cShift) = floats[4];
      *IndexByte(a, static_cast<size_t>(ints[5]) << cShift) = floats[5];
      *IndexByte(a, static_cast<size_t>(ints[6]) << cShift) = floats[6];
      *IndexByte(a, static_cast<size_t>(ints[7]) << cShift) = floats[7];
   }

   template<int cShift>
   inline static void DoubleStore(
         T* const a, const TInt& i, const Avx2_32_Float& val1, const Avx2_32_Float& val2) noexcept {
      // i is treated as signed, so we should only use the lower 31 bits otherwise we'll read from memory before a

      alignas(k_cAlignment) TInt::T ints[k_cSIMDPack];
      alignas(k_cAlignment) uint64_t floats1[k_cSIMDPack >> 1];
      alignas(k_cAlignment) uint64_t floats2[k_cSIMDPack >> 1];

      i.Store(ints);
      val1.Store(reinterpret_cast<T*>(floats1));
      val2.Store(reinterpret_cast<T*>(floats2));

      // if we shifted ints[] without converting to size_t first the compiler cannot
      // use the built in index shifting because ints could be 32 bits and shifting
      // right would chop off some bits, but when converted to size_t first then
      // that isn't an issue so the compiler can optimize the shift away and incorporate
      // it into the store assembly instruction
      *IndexByte(reinterpret_cast<uint64_t*>(a), static_cast<size_t>(ints[0]) << cShift) = floats1[0];
      *IndexByte(reinterpret_cast<uint64_t*>(a), static_cast<size_t>(ints[1]) << cShift) = floats1[1];
      *IndexByte(reinterpret_cast<uint64_t*>(a), static_cast<size_t>(ints[2]) << cShift) = floats1[2];
      *IndexByte(reinterpret_cast<uint64_t*>(a), static_cast<size_t>(ints[3]) << cShift) = floats1[3];

      *IndexByte(reinterpret_cast<uint64_t*>(a), static_cast<size_t>(ints[4]) << cShift) = floats2[0];
      *IndexByte(reinterpret_cast<uint64_t*>(a), static_cast<size_t>(ints[5]) << cShift) = floats2[1];
      *IndexByte(reinterpret_cast<uint64_t*>(a), static_cast<size_t>(ints[6]) << cShift) = floats2[2];
      *IndexByte(reinterpret_cast<uint64_t*>(a), static_cast<size_t>(ints[7]) << cShift) = floats2[3];
   }

   inline static void Interleaf(
         const Avx2_32_Float& val0, const Avx2_32_Float& val1, Avx2_32_Float& ret0, Avx2_32_Float& ret1) noexcept {
      // this function permutes the values into positions that the PermuteForInterleaf function expects
      // but for any SIMD implementation, the positions can be variable as long as they work together
      ret0 = Avx2_32_Float(_mm256_unpacklo_ps(val0.m_data, val1.m_data));
      ret1 = Avx2_32_Float(_mm256_unpackhi_ps(val0.m_data, val1.m_data));
   }

   template<typename TFunc>
   friend inline Avx2_32_Float ApplyFunc(const TFunc& func, const Avx2_32_Float& val) noexcept {
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
   }

   template<typename TFunc> static inline void Execute(const TFunc& func, const Avx2_32_Float& val0) noexcept {
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
   }

   template<typename TFunc>
   static inline void Execute(const TFunc& func, const Avx2_32_Float& val0, const Avx2_32_Float& val1) noexcept {
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
   }

   template<typename TFunc>
   static inline void Execute(const TFunc& func, const Avx2_32_Int& val0, const Avx2_32_Float& val1) noexcept {
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
   }

   template<typename TFunc>
   static inline void Execute(
         const TFunc& func, const Avx2_32_Int& val0, const Avx2_32_Float& val1, const Avx2_32_Float& val2) noexcept {
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
   }

   template<typename TFunc>
   static inline void Execute(const TFunc& func,
         const Avx2_32_Int& val0,
         const Avx2_32_Float& val1,
         const Avx2_32_Float& val2,
         const Avx2_32_Float& val3) noexcept {
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
   }

   template<typename TFunc>
   static inline void Execute(const TFunc& func,
         const Avx2_32_Int& val0,
         const Avx2_32_Int& val1,
         const Avx2_32_Float& val2,
         const Avx2_32_Float& val3) noexcept {
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
   }

   template<typename TFunc>
   static inline void Execute(const TFunc& func,
         const Avx2_32_Int& val0,
         const Avx2_32_Int& val1,
         const Avx2_32_Float& val2,
         const Avx2_32_Float& val3,
         const Avx2_32_Float& val4) noexcept {
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
   }

   friend inline Avx2_32_Float IfThenElse(
         const Avx2_32_Int& cmp, const Avx2_32_Float& trueVal, const Avx2_32_Float& falseVal) noexcept {
      return Avx2_32_Float(_mm256_blendv_ps(falseVal.m_data, trueVal.m_data, ReinterpretFloat(cmp).m_data));
   }

   friend inline Avx2_32_Float IfAdd(
         const Avx2_32_Int& cmp, const Avx2_32_Float& base, const Avx2_32_Float& addend) noexcept {
      return base + ReinterpretFloat(cmp & ReinterpretInt(addend));
   }

   friend inline Avx2_32_Int IsNaN(const Avx2_32_Float& cmp) noexcept {
      return ReinterpretInt(Avx2_32_Float(_mm256_cmp_ps(cmp.m_data, cmp.m_data, _CMP_UNORD_Q)));
   }

   static inline Avx2_32_Int ReinterpretInt(const Avx2_32_Float& val) noexcept {
      return Avx2_32_Int(_mm256_castps_si256(val.m_data));
   }

   static inline Avx2_32_Float ReinterpretFloat(const Avx2_32_Int& val) noexcept {
      return Avx2_32_Float(_mm256_castsi256_ps(val.m_data));
   }

   friend inline Avx2_32_Float Round(const Avx2_32_Float& val) noexcept {
      return Avx2_32_Float(_mm256_round_ps(val.m_data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
   }

   friend inline Avx2_32_Float Abs(const Avx2_32_Float& val) noexcept {
      return Avx2_32_Float(_mm256_and_ps(val.m_data, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF))));
   }

   friend inline Avx2_32_Float FastApproxReciprocal(const Avx2_32_Float& val) noexcept {
#ifdef FAST_DIVISION
      return Avx2_32_Float(_mm256_rcp_ps(val.m_data));
#else // FAST_DIVISION
      return Avx2_32_Float(1.0) / val;
#endif // FAST_DIVISION
   }

   friend inline Avx2_32_Float FastApproxDivide(const Avx2_32_Float& dividend, const Avx2_32_Float& divisor) noexcept {
#ifdef FAST_DIVISION
      return dividend * FastApproxReciprocal(divisor);
#else // FAST_DIVISION
      return dividend / divisor;
#endif // FAST_DIVISION
   }

   friend inline Avx2_32_Float FusedMultiplyAdd(
         const Avx2_32_Float& mul1, const Avx2_32_Float& mul2, const Avx2_32_Float& add) noexcept {
      // equivalent to: mul1 * mul2 + add
      return Avx2_32_Float(_mm256_fmadd_ps(mul1.m_data, mul2.m_data, add.m_data));
   }

   friend inline Avx2_32_Float FusedNegateMultiplyAdd(
         const Avx2_32_Float& mul1, const Avx2_32_Float& mul2, const Avx2_32_Float& add) noexcept {
      // equivalent to: -(mul1 * mul2) + add
      return Avx2_32_Float(_mm256_fnmadd_ps(mul1.m_data, mul2.m_data, add.m_data));
   }

   friend inline Avx2_32_Float FusedMultiplySubtract(
         const Avx2_32_Float& mul1, const Avx2_32_Float& mul2, const Avx2_32_Float& subtract) noexcept {
      // equivalent to: mul1 * mul2 - subtract
      return Avx2_32_Float(_mm256_fmsub_ps(mul1.m_data, mul2.m_data, subtract.m_data));
   }

   friend inline Avx2_32_Float Sqrt(const Avx2_32_Float& val) noexcept {
      return Avx2_32_Float(_mm256_sqrt_ps(val.m_data));
   }

   template<bool bUseApprox,
         bool bNegateInput = false,
         bool bNaNPossible = true,
         bool bUnderflowPossible = true,
         bool bOverflowPossible = true,
         bool bSpecialCaseZero = false,
         typename std::enable_if<!bUseApprox, int>::type = 0>
   static inline Avx2_32_Float ApproxExp(const Avx2_32_Float& val,
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
   static inline Avx2_32_Float ApproxExp(const Avx2_32_Float& val,
         const int32_t addExpSchraudolphTerm = k_expTermZeroMeanErrorForSoftmaxWithZeroedLogit) noexcept {
      // This code will make no sense until you read the Nicol N. Schraudolph paper:
      // https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.4508&rep=rep1&type=pdf
      // and also see approximate_math.hpp
      static constexpr float signedExpMultiple = bNegateInput ? -k_expMultiple : k_expMultiple;
#ifdef EXP_INT_SIMD
      const __m256 product = (val * signedExpMultiple).m_data;
      const __m256i retInt = _mm256_add_epi32(_mm256_cvttps_epi32(product), _mm256_set1_epi32(addExpSchraudolphTerm));
#else // EXP_INT_SIMD
      const __m256 retFloat = FusedMultiplyAdd(val, signedExpMultiple, static_cast<T>(addExpSchraudolphTerm)).m_data;
      const __m256i retInt = _mm256_cvttps_epi32(retFloat);
#endif // EXP_INT_SIMD
      Avx2_32_Float result = Avx2_32_Float(_mm256_castsi256_ps(retInt));
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
   static inline Avx2_32_Float ApproxLog(
         const Avx2_32_Float& val, const float addLogSchraudolphTerm = k_logTermLowerBoundInputCloseToOne) noexcept {
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
   static inline Avx2_32_Float ApproxLog(
         const Avx2_32_Float& val, const float addLogSchraudolphTerm = k_logTermLowerBoundInputCloseToOne) noexcept {
      // This code will make no sense until you read the Nicol N. Schraudolph paper:
      // https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.4508&rep=rep1&type=pdf
      // and also see approximate_math.hpp
      const __m256i retInt = _mm256_castps_si256(val.m_data);
      Avx2_32_Float result = Avx2_32_Float(_mm256_cvtepi32_ps(retInt));
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

   friend inline T Sum(const Avx2_32_Float& val) noexcept {
      const __m128 vlow = _mm256_castps256_ps128(val.m_data);
      const __m128 vhigh = _mm256_extractf128_ps(val.m_data, 1);
      const __m128 sum = _mm_add_ps(vlow, vhigh);
      const __m128 sum1 = _mm_hadd_ps(sum, sum);
      const __m128 sum2 = _mm_hadd_ps(sum1, sum1);
      return _mm_cvtss_f32(sum2);
   }

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
      RemoteBinSumsBoosting<Avx2_32_Float, bHessian, bWeight, bCollapsed, cCompilerScores, bParallel>(pParams);
      return Error_None;
   }

   template<bool bHessian, bool bWeight, size_t cCompilerScores, size_t cCompilerDimensions>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorBinSumsInteraction(
         BinSumsInteractionBridge* const pParams) noexcept {
      RemoteBinSumsInteraction<Avx2_32_Float, bHessian, bWeight, cCompilerScores, cCompilerDimensions>(pParams);
      return Error_None;
   }

 private:
   inline Avx2_32_Float(const TPack& data) noexcept : m_data(data) {}

   TPack m_data;
};
static_assert(std::is_standard_layout<Avx2_32_Float>::value && std::is_trivially_copyable<Avx2_32_Float>::value,
      "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

template<bool bNegateInput, bool bNaNPossible, bool bUnderflowPossible, bool bOverflowPossible>
inline Avx2_32_Float Exp(const Avx2_32_Float& val) noexcept {
   return Exp32<Avx2_32_Float, bNegateInput, bNaNPossible, bUnderflowPossible, bOverflowPossible>(val);
}

template<bool bNegateOutput,
      bool bNaNPossible,
      bool bNegativePossible,
      bool bZeroPossible,
      bool bPositiveInfinityPossible>
inline Avx2_32_Float Log(const Avx2_32_Float& val) noexcept {
   return Log32<Avx2_32_Float,
         bNegateOutput,
         bNaNPossible,
         bNegativePossible,
         bZeroPossible,
         bPositiveInfinityPossible>(val);
}

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm ApplyUpdate_Avx2_32(
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

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm BinSumsBoosting_Avx2_32(
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

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm BinSumsInteraction_Avx2_32(
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

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm CreateObjective_Avx2_32(const Config* const pConfig,
      const char* const sObjective,
      const char* const sObjectiveEnd,
      ObjectiveWrapper* const pObjectiveWrapperOut) {
   pObjectiveWrapperOut->m_pApplyUpdateC = ApplyUpdate_Avx2_32;
   pObjectiveWrapperOut->m_pBinSumsBoostingC = BinSumsBoosting_Avx2_32;
   pObjectiveWrapperOut->m_pBinSumsInteractionC = BinSumsInteraction_Avx2_32;
   ErrorEbm error = ComputeWrapper<Avx2_32_Float>::FillWrapper(pObjectiveWrapperOut);
   if(Error_None != error) {
      return error;
   }
   return Objective::CreateObjective<Avx2_32_Float>(pConfig, sObjective, sObjectiveEnd, pObjectiveWrapperOut);
}

} // namespace DEFINED_ZONE_NAME

#endif // BRIDGE_AVX2_32