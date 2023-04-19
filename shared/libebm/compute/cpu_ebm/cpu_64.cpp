// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <cmath>

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

struct Cpu_64_Int final {
   static constexpr int cPack = 1;
   using T = uint64_t;

   inline Cpu_64_Int(const uint64_t val) noexcept : m_data(val) {
   }

private:
   uint64_t m_data;
};
static_assert(std::is_standard_layout<Cpu_64_Int>::value && std::is_trivially_copyable<Cpu_64_Int>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

struct Cpu_64_Float final {
   static constexpr int cPack = 1;
   using T = double;
   using TInt = Cpu_64_Int;

   WARNING_PUSH
   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   inline Cpu_64_Float() noexcept {
   }
   WARNING_POP

   Cpu_64_Float(const Cpu_64_Float & other) noexcept = default; // preserve POD status
   Cpu_64_Float & operator=(const Cpu_64_Float &) noexcept = default; // preserve POD status

   inline Cpu_64_Float(const double val) noexcept : m_data { val } {
   }

   inline Cpu_64_Float & operator= (const double val) noexcept {
      m_data = val;
      return *this;
   }


   inline Cpu_64_Float operator+() const noexcept {
      return *this;
   }

   inline Cpu_64_Float operator-() const noexcept {
      return Cpu_64_Float(-m_data);
   }


   inline Cpu_64_Float operator+ (const Cpu_64_Float & other) const noexcept {
      return Cpu_64_Float(m_data + other.m_data);
   }

   inline Cpu_64_Float operator- (const Cpu_64_Float & other) const noexcept {
      return Cpu_64_Float(m_data - other.m_data);
   }

   inline Cpu_64_Float operator* (const Cpu_64_Float & other) const noexcept {
      return Cpu_64_Float(m_data * other.m_data);
   }

   inline Cpu_64_Float operator/ (const Cpu_64_Float & other) const noexcept {
      return Cpu_64_Float(m_data / other.m_data);
   }

   inline Cpu_64_Float & operator+= (const Cpu_64_Float & other) noexcept {
      *this = (*this) + other;
      return *this;
   }

   inline Cpu_64_Float & operator-= (const Cpu_64_Float & other) noexcept {
      *this = (*this) - other;
      return *this;
   }

   inline Cpu_64_Float & operator*= (const Cpu_64_Float & other) noexcept {
      *this = (*this) * other;
      return *this;
   }

   inline Cpu_64_Float & operator/= (const Cpu_64_Float & other) noexcept {
      *this = (*this) / other;
      return *this;
   }


   friend inline Cpu_64_Float operator+ (const double val, const Cpu_64_Float & other) noexcept {
      return Cpu_64_Float(val) + other;
   }

   friend inline Cpu_64_Float operator- (const double val, const Cpu_64_Float & other) noexcept {
      return Cpu_64_Float(val) - other;
   }

   friend inline Cpu_64_Float operator* (const double val, const Cpu_64_Float & other) noexcept {
      return Cpu_64_Float(val) * other;
   }

   friend inline Cpu_64_Float operator/ (const double val, const Cpu_64_Float & other) noexcept {
      return Cpu_64_Float(val) / other;
   }

   inline void LoadAligned(const T * const a) noexcept {
      m_data = *a;
   }

   inline void SaveAligned(T * const a) const noexcept {
      *a = m_data;
   }

   template<typename TFunc>
   friend inline Cpu_64_Float ApplyFunction(const Cpu_64_Float & val, const TFunc & func) noexcept {
      // this function is more useful for a SIMD operator where it applies func() to all packed items
      return Cpu_64_Float(func(val.m_data));
   }

   friend inline Cpu_64_Float IfGreater(const Cpu_64_Float & cmp1, const Cpu_64_Float & cmp2, const Cpu_64_Float & trueVal, const Cpu_64_Float & falseVal) noexcept {
      return cmp1.m_data > cmp2.m_data ? trueVal : falseVal;
   }

   friend inline Cpu_64_Float IfLess(const Cpu_64_Float & cmp1, const Cpu_64_Float & cmp2, const Cpu_64_Float & trueVal, const Cpu_64_Float & falseVal) noexcept {
      return cmp1.m_data < cmp2.m_data ? trueVal : falseVal;
   }

   friend inline Cpu_64_Float Sqrt(const Cpu_64_Float & val) noexcept {
      return Cpu_64_Float(std::sqrt(val.m_data));
   }

   friend inline Cpu_64_Float Exp(const Cpu_64_Float & val) noexcept {
      return Cpu_64_Float(std::exp(val.m_data));
   }

   friend inline Cpu_64_Float Log(const Cpu_64_Float & val) noexcept {
      return Cpu_64_Float(std::log(val.m_data));
   }

   friend inline T Sum(const Cpu_64_Float & val) noexcept {
      return val.m_data;
   }

   template<typename TLoss, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorApplyUpdate(const Loss * const pLoss, ApplyUpdateBridge * const pData) noexcept {
      // this allows us to switch execution onto GPU, FPGA, or other local computation
      RemoteApplyUpdate<TLoss, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight>(pLoss, pData);
      return Error_None;
   }

private:

   double m_data;
};
static_assert(std::is_standard_layout<Cpu_64_Float>::value && std::is_trivially_copyable<Cpu_64_Float>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");


// FIRST, define the RegisterLoss function that we'll be calling from our registrations.  This is a static 
// function, so we can have duplicate named functions in other files and they'll refer to different functions
template<template <typename> class TRegistrable, typename... Args>
INLINE_ALWAYS static std::shared_ptr<const Registration> RegisterLoss(const char * const sRegistrationName, const Args...args) {
   return Register<TRegistrable, Cpu_64_Float>(sRegistrationName, args...);
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
