// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#if (defined(__clang__) || defined(__GNUC__) || defined(__SUNPRO_CC)) && defined(__x86_64__) || defined(_MSC_VER)

#include <cmath>
#include <immintrin.h> // SIMD.  Do not include in precompiled_header_cpp.hpp!

#include "ebm_native.h"
#include "logging.h"
#include "common_c.h"
#include "bridge_c.h"
#include "zones.h"

#include "common_cpp.hpp"
#include "bridge_cpp.hpp"

#include "Registration.hpp"
#include "Loss.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct Sse_32_Operators final {
   static constexpr size_t countPackedItems = 4; // the number of Unpacked items in a Packed structure
   typedef float Unpacked;
   typedef __m128 Packed;

private:

   Packed m_data;

   INLINE_ALWAYS Sse_32_Operators(const Packed & data) noexcept : m_data(data) {
   }

public:

   WARNING_PUSH
   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   INLINE_ALWAYS Sse_32_Operators() noexcept {
   }
   WARNING_POP

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

   template<template <typename, typename, ptrdiff_t, ptrdiff_t, bool> class TExecute, typename TLoss, typename TFloat, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian>
   INLINE_RELEASE_TEMPLATED static ErrorEbm ApplyTraining(const Loss * const pLoss, ApplyTrainingData * const pData) {
      // this allows us to switch execution onto GPU, FPGA, or other local computation
      ExecuteApplyTraining<TExecute, TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian>(
         pLoss,
         pData->m_cRuntimeScores,
         pData->m_cRuntimePack
      );
      return Error_None;
   }

   template<template <typename, typename, ptrdiff_t, ptrdiff_t, bool> class TExecute, typename TLoss, typename TFloat, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian>
   INLINE_RELEASE_TEMPLATED static ErrorEbm ApplyValidation(const Loss * const pLoss, ApplyValidationData * const pData) {
      // this allows us to switch execution onto GPU, FPGA, or other local computation
      ExecuteApplyValidation<TExecute, TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian>(
         pLoss,
         pData->m_cRuntimeScores,
         pData->m_cRuntimePack,
         &pData->m_metricOut
      );
      return Error_None;
   }
};
static_assert(std::is_standard_layout<Sse_32_Operators>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");
#if !(defined(__GNUC__) && __GNUC__ < 5)
static_assert(std::is_trivially_copyable<Sse_32_Operators>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");
#endif // !(defined(__GNUC__) && __GNUC__ < 5)

// FIRST, define the RegisterLoss function that we'll be calling from our registrations.  This is a static 
// function, so we can have duplicate named functions in other files and they'll refer to different functions
template<template <typename> class TRegistrable, typename... Args>
INLINE_ALWAYS static std::shared_ptr<const Registration> RegisterLoss(const char * const sRegistrationName, const Args...args) {
   return Register<TRegistrable, Sse_32_Operators>(sRegistrationName, args...);
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
