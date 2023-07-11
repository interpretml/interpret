// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <cmath>
#include <type_traits>

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

struct Cpu_64_Float;

struct Cpu_64_Int final {
   friend Cpu_64_Float;
   friend inline Cpu_64_Float IfEqual(const Cpu_64_Int & cmp1, const Cpu_64_Int & cmp2, const Cpu_64_Float & trueVal, const Cpu_64_Float & falseVal) noexcept;

   using T = uint64_t;
   using TPack = uint64_t;
   static_assert(std::is_unsigned<T>::value, "T must be an unsigned integer type");
   static_assert(std::numeric_limits<T>::max() <= std::numeric_limits<UIntExceed>::max(), "UIntExceed must be able to hold a T");
   static constexpr bool bCpu = true;
   static constexpr int k_cSIMDShift = 0;
   static constexpr int k_cSIMDPack = 1 << k_cSIMDShift;

   WARNING_PUSH
   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   inline Cpu_64_Int() noexcept {
   }
   WARNING_POP

   inline Cpu_64_Int(const T val) noexcept : m_data(val) {
   }

   inline static Cpu_64_Int Load(const T * const a) noexcept {
      return Cpu_64_Int(*a);
   }

   inline void Store(T * const a) const noexcept {
      *a = m_data;
   }

   inline static Cpu_64_Int LoadBytes(const uint8_t * const a) noexcept {
      return Cpu_64_Int(*a);
   }

   template<typename TFunc, typename... TArgs>
   static inline void Execute(const TFunc & func, const TArgs&... args) noexcept {
      func(0, (args.m_data)...);
   }

   inline static Cpu_64_Int MakeIndexes() noexcept {
      return Cpu_64_Int(0);
   }

   inline Cpu_64_Int operator+ (const Cpu_64_Int & other) const noexcept {
      return Cpu_64_Int(m_data + other.m_data);
   }

   inline Cpu_64_Int & operator+= (const Cpu_64_Int & other) noexcept {
      *this = (*this) + other;
      return *this;
   }

   inline Cpu_64_Int operator* (const Cpu_64_Int & other) const noexcept {
      return Cpu_64_Int(m_data * other.m_data);
   }

   inline Cpu_64_Int & operator*= (const Cpu_64_Int & other) noexcept {
      *this = (*this) * other;
      return *this;
   }

   inline Cpu_64_Int operator>> (int shift) const noexcept {
      return Cpu_64_Int(m_data >> shift);
   }

   inline Cpu_64_Int operator<< (int shift) const noexcept {
      return Cpu_64_Int(m_data << shift);
   }

   inline Cpu_64_Int operator& (const Cpu_64_Int & other) const noexcept {
      return Cpu_64_Int(other.m_data & m_data);
   }

private:
   TPack m_data;
};
static_assert(std::is_standard_layout<Cpu_64_Int>::value && std::is_trivially_copyable<Cpu_64_Int>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

struct Cpu_64_Float final {
   using T = double;
   using TPack = double;
   using TInt = Cpu_64_Int;
   static_assert(sizeof(T) <= sizeof(Float_Big), "Float_Big must be able to hold a T");
   static constexpr bool bCpu = TInt::bCpu;
   static constexpr int k_cSIMDShift = TInt::k_cSIMDShift;
   static constexpr int k_cSIMDPack = TInt::k_cSIMDPack;

   WARNING_PUSH
   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   inline Cpu_64_Float() noexcept {
   }
   WARNING_POP

   Cpu_64_Float(const Cpu_64_Float & other) noexcept = default; // preserve POD status
   Cpu_64_Float & operator=(const Cpu_64_Float &) noexcept = default; // preserve POD status

   inline Cpu_64_Float(const double val) noexcept : m_data { static_cast<T>(val) } {
   }
   inline Cpu_64_Float(const float val) noexcept : m_data { static_cast<T>(val) } {
   }
   inline Cpu_64_Float(const int val) noexcept : m_data { static_cast<T>(val) } {
   }

   inline Cpu_64_Float & operator= (const double val) noexcept {
      m_data = static_cast<T>(val);
      return *this;
   }
   inline Cpu_64_Float & operator= (const float val) noexcept {
      m_data = static_cast<T>(val);
      return *this;
   }
   inline Cpu_64_Float & operator= (const int val) noexcept {
      m_data = static_cast<T>(val);
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

   inline static Cpu_64_Float Load(const T * const a) noexcept {
      return Cpu_64_Float(*a);
   }

   inline void Store(T * const a) const noexcept {
      *a = m_data;
   }

   inline static Cpu_64_Float Load(const T * const a, const TInt i) noexcept {
      return Cpu_64_Float(a[i.m_data]);
   }

   inline void Store(T * const a, const TInt i) noexcept {
      a[i.m_data] = m_data;
   }

   template<typename TFunc>
   friend inline Cpu_64_Float ApplyFunction(const Cpu_64_Float & val, const TFunc & func) noexcept {
      // this function is more useful for a SIMD operator where it applies func() to all packed items
      return Cpu_64_Float(func(val.m_data));
   }

   template<typename TFunc, typename... TArgs>
   static inline void Execute(const TFunc & func, const TArgs&... args) noexcept {
      func(0, (args.m_data)...);
   }

   friend inline Cpu_64_Float IfGreater(const Cpu_64_Float & cmp1, const Cpu_64_Float & cmp2, const Cpu_64_Float & trueVal, const Cpu_64_Float & falseVal) noexcept {
      return cmp1.m_data > cmp2.m_data ? trueVal : falseVal;
   }

   friend inline Cpu_64_Float IfLess(const Cpu_64_Float & cmp1, const Cpu_64_Float & cmp2, const Cpu_64_Float & trueVal, const Cpu_64_Float & falseVal) noexcept {
      return cmp1.m_data < cmp2.m_data ? trueVal : falseVal;
   }

   friend inline Cpu_64_Float IfEqual(const Cpu_64_Int & cmp1, const Cpu_64_Int & cmp2, const Cpu_64_Float & trueVal, const Cpu_64_Float & falseVal) noexcept {
      return Cpu_64_Float(cmp1.m_data == cmp2.m_data ? trueVal : falseVal);
   }

   friend inline Cpu_64_Float Abs(const Cpu_64_Float & val) noexcept {
      return Cpu_64_Float(std::abs(val.m_data));
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

   template<typename TObjective, size_t cCompilerScores, bool bKeepGradHess, bool bCalcMetric, bool bWeight, bool bHessian, ptrdiff_t cCompilerPack>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorApplyUpdate(const Objective * const pObjective, ApplyUpdateBridge * const pData) noexcept {
      // this allows us to switch execution onto GPU, FPGA, or other local computation
      RemoteApplyUpdate<TObjective, cCompilerScores, bKeepGradHess, bCalcMetric, bWeight, bHessian, cCompilerPack>(pObjective, pData);
      return Error_None;
   }

private:

   TPack m_data;
};
static_assert(std::is_standard_layout<Cpu_64_Float>::value && std::is_trivially_copyable<Cpu_64_Float>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");


// FIRST, define the RegisterObjective function that we'll be calling from our registrations.  This is a static 
// function, so we can have duplicate named functions in other files and they'll refer to different functions
template<template <typename> class TRegistrable, typename... Args>
INLINE_ALWAYS static std::shared_ptr<const Registration> RegisterObjective(const char * const sRegistrationName, const Args...args) {
   return Register<TRegistrable, Cpu_64_Float>(sRegistrationName, args...);
}

// now include all our special objective registrations which will use the RegisterObjective function we defined above!
#include "objective_registrations.hpp"

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm CreateObjective_Cpu_64(
   const Config * const pConfig,
   const char * const sObjective,
   const char * const sObjectiveEnd,
   ObjectiveWrapper * const pObjectiveWrapperOut
) {
   ErrorEbm error = ComputeWrapper<Cpu_64_Float>::FillWrapper(pObjectiveWrapperOut);
   if(Error_None != error) {
      return error;
   }
   error = Objective::CreateObjective(&RegisterObjectives, pConfig, sObjective, sObjectiveEnd, pObjectiveWrapperOut);
   if(Error_None != error) {
      return error;
   }
   return Error_None;
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
