// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <cmath>
#include <immintrin.h> // SIMD.  Do not include in PrecompiledHeader.h!

#include "ebm_native.h"
#include "logging.h"
#include "common_c.h"
#include "bridge_c.h"
#include "zones.h"

#include "common_cpp.hpp"
#include "bridge_cpp.hpp"

#include "Registrable.hpp"
#include "Registration.hpp"
#include "Loss.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct Sse_32_Operators final {
   typedef float Unpacked;
   typedef __m128 Packed;

private:

   Packed m_data;

   INLINE_ALWAYS Sse_32_Operators(const Packed & data) noexcept : m_data(data) {
   }

public:

   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   INLINE_ALWAYS Sse_32_Operators() noexcept {
   }

   INLINE_ALWAYS Sse_32_Operators(const Sse_32_Operators & data) noexcept : m_data(data.m_data) {
   }

   INLINE_ALWAYS Sse_32_Operators(const float data) noexcept : m_data(_mm_set1_ps(static_cast<Unpacked>(data))) {
   }

   INLINE_ALWAYS Sse_32_Operators(const double data) noexcept : m_data(_mm_set1_ps(static_cast<Unpacked>(data))) {
   }

   INLINE_ALWAYS Sse_32_Operators(const int data) noexcept : m_data(_mm_set1_ps(static_cast<Unpacked>(data))) {
   }

   INLINE_ALWAYS Sse_32_Operators operator+ (const Sse_32_Operators & other) const noexcept {
      return Sse_32_Operators(_mm_add_ps(m_data, other.m_data));
   }

   INLINE_ALWAYS Sse_32_Operators operator- (const Sse_32_Operators & other) const noexcept {
      return Sse_32_Operators(_mm_sub_ps(m_data, other.m_data));
   }

   INLINE_ALWAYS Sse_32_Operators operator* (const Sse_32_Operators & other) const noexcept {
      return Sse_32_Operators(_mm_mul_ps(m_data, other.m_data));
   }

   INLINE_ALWAYS Sse_32_Operators operator/ (const Sse_32_Operators & other) const noexcept {
      return Sse_32_Operators(_mm_div_ps(m_data, other.m_data));
   }

   INLINE_ALWAYS bool IsAnyEqual(const Sse_32_Operators & other) const noexcept {
      return !!_mm_movemask_ps(_mm_cmpeq_ps(m_data, other.m_data));
   }

   INLINE_ALWAYS bool IsAnyInf() const noexcept {
      return !!_mm_movemask_ps(_mm_cmpeq_ps(_mm_andnot_ps(_mm_set1_ps(Unpacked { -0.0 }), m_data), _mm_set1_ps(std::numeric_limits<Unpacked>::infinity())));
   }

   INLINE_ALWAYS bool IsAnyNaN() const noexcept {
      // use the fact that a != a  always yields false, except when both are NaN in IEEE 754 where it's true
      return !!_mm_movemask_ps(_mm_cmpneq_ps(m_data, m_data));
   }

   INLINE_ALWAYS Sse_32_Operators Sqrt() const noexcept {
      // TODO: consider making a fast approximation of this
      return Sse_32_Operators(_mm_sqrt_ss(m_data));
   }
};

// FIRST, define the RegisterLoss function that we'll be calling from our registrations.  This is a static 
// function, so we can have duplicate named functions in other files and they'll refer to different functions
template<template <typename> class TRegistrable, typename... Args>
static INLINE_ALWAYS std::shared_ptr<const Registration> RegisterLoss(const char * const sRegistrationName, const Args...args) {
   return Register<TRegistrable, Sse_32_Operators>(sRegistrationName, args...);
}

// now include all our special loss registrations which will use the RegisterLoss function we defined above!
#include "loss_registrations.hpp"

INTERNAL_IMPORT_EXPORT_BODY ErrorEbmType CreateLoss_Sse_32(
   const size_t cOutputs,
   const char * const sLoss,
   const char * const sLossEnd,
   const void ** const ppLossOut
) {
   return Loss::CreateLoss(&RegisterLosses, cOutputs, sLoss, sLossEnd, ppLossOut);
}

INTERNAL_IMPORT_EXPORT_BODY ErrorEbmType CreateMetric_Sse_32(
   const size_t cOutputs,
   const char * const sLoss,
   const char * const sLossEnd,
   const void ** const ppLossOut
) {
   UNUSED(cOutputs);
   UNUSED(sLoss);
   UNUSED(sLossEnd);
   UNUSED(ppLossOut);

   return Error_UnknownInternalError;
}

} // DEFINED_ZONE_NAME
