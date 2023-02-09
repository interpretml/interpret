// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <cmath>

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

struct Cpu_64_Operators final {
   static constexpr size_t countPackedItems = 1; // the number of Unpacked items in a Packed structure
   typedef double Unpacked;
   typedef double Packed;

private:

   Packed m_data;

public:

   WARNING_PUSH
   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   INLINE_ALWAYS Cpu_64_Operators() noexcept {
   }
   WARNING_POP

   INLINE_ALWAYS Cpu_64_Operators(const float data) noexcept : m_data(static_cast<Unpacked>(data)) {
   }

   INLINE_ALWAYS Cpu_64_Operators(const double data) noexcept : m_data(static_cast<Unpacked>(data)) {
   }

   INLINE_ALWAYS Cpu_64_Operators(const int data) noexcept : m_data(static_cast<Unpacked>(data)) {
   }

   INLINE_ALWAYS Cpu_64_Operators operator+ (const Cpu_64_Operators & other) const noexcept {
      return Cpu_64_Operators(m_data + other.m_data);
   }

   INLINE_ALWAYS Cpu_64_Operators operator- (const Cpu_64_Operators & other) const noexcept {
      return Cpu_64_Operators(m_data - other.m_data);
   }

   INLINE_ALWAYS Cpu_64_Operators operator* (const Cpu_64_Operators & other) const noexcept {
      return Cpu_64_Operators(m_data * other.m_data);
   }

   INLINE_ALWAYS Cpu_64_Operators operator/ (const Cpu_64_Operators & other) const noexcept {
      return Cpu_64_Operators(m_data / other.m_data);
   }

   INLINE_ALWAYS bool IsAnyEqual(const Cpu_64_Operators & other) const noexcept {
      return m_data == other.m_data;
   }

   INLINE_ALWAYS bool IsAnyInf() const noexcept {
      return std::isinf(m_data);
   }

   INLINE_ALWAYS bool IsAnyNaN() const noexcept {
      return std::isnan(m_data);
   }

   INLINE_ALWAYS Cpu_64_Operators Sqrt() const noexcept {
      return Cpu_64_Operators(std::sqrt(m_data));
   }

   INLINE_ALWAYS Unpacked GetUnpacked(const size_t indexPack) const noexcept {
      UNUSED(indexPack);
      return m_data; // we only have 1 packed item
   }

   INLINE_ALWAYS void SetUnpacked(const size_t indexPack, const Unpacked data) noexcept {
      UNUSED(indexPack);
      m_data = data; // we only have 1 packed item
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
static_assert(std::is_standard_layout<Cpu_64_Operators>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");
#if !(defined(__GNUC__) && __GNUC__ < 5)
static_assert(std::is_trivially_copyable<Cpu_64_Operators>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");
#endif // !(defined(__GNUC__) && __GNUC__ < 5)

// FIRST, define the RegisterLoss function that we'll be calling from our registrations.  This is a static 
// function, so we can have duplicate named functions in other files and they'll refer to different functions
template<template <typename> class TRegistrable, typename... Args>
INLINE_ALWAYS static std::shared_ptr<const Registration> RegisterLoss(const char * const sRegistrationName, const Args...args) {
   return Register<TRegistrable, Cpu_64_Operators>(sRegistrationName, args...);
}

// now include all our special loss registrations which will use the RegisterLoss function we defined above!
#include "loss_registrations.hpp"

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm CreateLoss_Cpu_64(
   const Config * const pConfig,
   const char * const sLoss,
   const char * const sLossEnd,
   LossWrapper * const pLossWrapperOut
) {
   return Loss::CreateLoss(&RegisterLosses, pConfig, sLoss, sLossEnd, pLossWrapperOut);
}

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm CreateMetric_Cpu_64(
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
