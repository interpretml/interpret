// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#ifdef NEVER

// BEFORE we do anything with out include files, ensure we have the correct SIMD compilation flags set

// TODO: un-disable this file
#define DISABLE_THIS_FILE

// TODO: turn on separate compiler flags for c++, clang++ and also Visual Studio (through the file options!!)

//#if defined(__clang__) // compiler type (clang++)
//#pragma clang attribute push (__attribute__((target("msse2"))), apply_to=function)
//#elif defined(__GNUC__) // compiler type (g++)
//#pragma GCC target("msse2")
//#elif defined(__SUNPRO_CC) // compiler type (Oracle Developer Studio)
//#define DISABLE_THIS_FILE
//#elif defined(_MSC_VER) // compiler type (Microsoft Visual Studio compiler)
//#else  // compiler type
//#error compiler not recognized
//#endif // compiler type

#ifndef DISABLE_THIS_FILE


#include <stddef.h> // size_t, ptrdiff_t
#include <immintrin.h> // SIMD.  Do not include in PrecompiledHeader.h!

// TODO: put EVERYTHING in this zone into a single anonymous namespace (including *.cpp files which we'll include!)
//namespace {

#include "ebm_native.h"
#include "EbmInternal.h"
// very independent includes
#include "logging.h" // EBM_ASSERT & LOG
#include "Registration.h"

#include "Registrable.h"
#include "Loss.h"

// helpful SSE info: https://software.intel.com/sites/landingpage/IntrinsicsGuide/

struct Simd32Sse2 final {
   typedef float Unpacked;
   typedef __m128 Packed;

private:

   Packed m_data;

   INLINE_ALWAYS Simd32Sse2(const Packed & data) noexcept : m_data(data) {
   }

public:

   INLINE_ALWAYS Simd32Sse2() noexcept : m_data(_mm_undefined_ps()) {
   }

   INLINE_ALWAYS Simd32Sse2(const Simd32Sse2 & data) noexcept : m_data(data.m_data) {
   }

   INLINE_ALWAYS Simd32Sse2(const float data) noexcept : m_data(_mm_set1_ps(static_cast<Unpacked>(data))) {
   }

   INLINE_ALWAYS Simd32Sse2(const double data) noexcept : m_data(_mm_set1_ps(static_cast<Unpacked>(data))) {
   }

   INLINE_ALWAYS Simd32Sse2(const int data) noexcept : m_data(_mm_set1_ps(static_cast<Unpacked>(data))) {
   }

   INLINE_ALWAYS Simd32Sse2 operator+ (const Simd32Sse2 & other) const noexcept {
      return Simd32Sse2(_mm_add_ps(m_data, other.m_data));
   }

   INLINE_ALWAYS Simd32Sse2 operator- (const Simd32Sse2 & other) const noexcept {
      return Simd32Sse2(_mm_sub_ps(m_data, other.m_data));
   }

   INLINE_ALWAYS Simd32Sse2 operator* (const Simd32Sse2 & other) const noexcept {
      return Simd32Sse2(_mm_mul_ps(m_data, other.m_data));
   }

   INLINE_ALWAYS Simd32Sse2 operator/ (const Simd32Sse2 & other) const noexcept {
      return Simd32Sse2(_mm_div_ps(m_data, other.m_data));
   }

   INLINE_ALWAYS bool IsAnyEqual(const Simd32Sse2 & other) const noexcept {
      return !!_mm_movemask_ps(_mm_cmpeq_ps(m_data, other.m_data));
   }

   INLINE_ALWAYS bool IsAnyInf() const noexcept {
      return !!_mm_movemask_ps(_mm_cmpeq_ps(_mm_andnot_ps(_mm_set1_ps(Unpacked { -0.0 }), m_data), _mm_set1_ps(std::numeric_limits<Unpacked>::infinity())));
   }

   INLINE_ALWAYS bool IsAnyNaN() const noexcept {
      // use the fact that a != a  always yields false, except when both are NaN in IEEE 754 where it's true
      return !!_mm_movemask_ps(_mm_cmpneq_ps(m_data, m_data));
   }

   INLINE_ALWAYS Simd32Sse2 Sqrt() const noexcept {
      // TODO: consider making a fast approximation of this
      return Simd32Sse2(_mm_sqrt_ss(m_data));
   }
};

// !!! IMPORTANT: This file is compiled with SSE2 enhanced instruction set enabled !!!
// In Visual Studio this is done by right clicking this filename -> properties -> C/C++ -> Code Generation -> 
//   Enable Enhanced Instruction Set -> SSE2.
// In g++/clang++ we use pragmas

// FIRST, define the RegisterLoss function that we'll be calling from our registrations.  This is a static 
// function, so we can have duplicate named functions in other files and they'll refer to different functions
template<template <typename> class TRegistrable, typename... Args>
static INLINE_ALWAYS std::shared_ptr<const Registration> RegisterLoss(const char * const sRegistrationName, const Args...args) {
   return Register<TRegistrable, Simd32Sse2>(sRegistrationName, args...);
}

// now include all our special loss registrations which will use the RegisterLoss function we defined above!
#include "LossRegistrations.h"

extern const std::vector<std::shared_ptr<const Registration>> RegisterLosses32Sse2() {
   return RegisterLosses();
}

// TODO: I'm not sure if this is needed.  
//#pragma clang attribute pop


//} // namespace


#else // DISABLE THIS FILE

//extern const std::vector<std::shared_ptr<const Registration>> RegisterLosses32Sse2() {
//   // signal that this compilation doesn't handle this type
//   return std::vector<std::shared_ptr<const Registration>>();
//}

#endif // DISABLE_THIS_FILE

#endif // NEVER
