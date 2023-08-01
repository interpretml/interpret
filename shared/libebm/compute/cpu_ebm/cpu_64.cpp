// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <cmath> // exp, log
#include <limits> // numeric_limits
#include <type_traits> // is_unsigned

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
   static_assert(std::is_same<UIntBig, T>::value || std::is_same<UIntSmall, T>::value,
      "T must be either UIntBig or UIntSmall");
   static constexpr bool k_bCpu = true;
   static constexpr int k_cSIMDShift = 0;
   static constexpr int k_cSIMDPack = 1 << k_cSIMDShift;

   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   inline Cpu_64_Int() noexcept {
   }

   inline Cpu_64_Int(const T & val) noexcept : m_data(val) {
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
   static inline void Execute(const TFunc & func, const TArgs &... args) noexcept {
      func(0, (args.m_data)...);
   }

   inline static Cpu_64_Int MakeIndexes() noexcept {
      return Cpu_64_Int(0);
   }

   inline Cpu_64_Int operator+ (const Cpu_64_Int & other) const noexcept {
      return Cpu_64_Int(m_data + other.m_data);
   }

   inline Cpu_64_Int operator* (const T & other) const noexcept {
      return Cpu_64_Int(m_data * other);
   }

   inline Cpu_64_Int operator>> (int shift) const noexcept {
      return Cpu_64_Int(m_data >> shift);
   }

   inline Cpu_64_Int operator<< (int shift) const noexcept {
      return Cpu_64_Int(m_data << shift);
   }

   inline Cpu_64_Int operator& (const Cpu_64_Int & other) const noexcept {
      return Cpu_64_Int(m_data & other.m_data);
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
   static_assert(std::is_same<FloatBig, T>::value || std::is_same<FloatSmall, T>::value,
      "T must be either FloatBig or FloatSmall");
   static constexpr bool k_bCpu = TInt::k_bCpu;
   static constexpr int k_cSIMDShift = TInt::k_cSIMDShift;
   static constexpr int k_cSIMDPack = TInt::k_cSIMDPack;

   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   inline Cpu_64_Float() noexcept {
   }

   inline Cpu_64_Float(const double val) noexcept : m_data(static_cast<T>(val)) {
   }
   inline Cpu_64_Float(const float val) noexcept : m_data(static_cast<T>(val)) {
   }
   inline Cpu_64_Float(const int val) noexcept : m_data(static_cast<T>(val)) {
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


   friend inline Cpu_64_Float operator+ (const float val, const Cpu_64_Float & other) noexcept {
      return Cpu_64_Float(val) + other;
   }

   friend inline Cpu_64_Float operator- (const float val, const Cpu_64_Float & other) noexcept {
      return Cpu_64_Float(val) - other;
   }

   friend inline Cpu_64_Float operator* (const float val, const Cpu_64_Float & other) noexcept {
      return Cpu_64_Float(val) * other;
   }

   friend inline Cpu_64_Float operator/ (const float val, const Cpu_64_Float & other) noexcept {
      return Cpu_64_Float(val) / other;
   }


   inline static Cpu_64_Float Load(const T * const a) noexcept {
      return Cpu_64_Float(*a);
   }

   inline void Store(T * const a) const noexcept {
      *a = m_data;
   }

   inline static Cpu_64_Float Load(const T * const a, const TInt & i) noexcept {
      return Cpu_64_Float(a[i.m_data]);
   }

   inline void Store(T * const a, const TInt & i) const noexcept {
      a[i.m_data] = m_data;
   }

   template<typename TFunc>
   friend inline Cpu_64_Float ApplyFunc(const TFunc & func, const Cpu_64_Float & val) noexcept {
      return Cpu_64_Float(func(val.m_data));
   }

   template<typename TFunc, typename... TArgs>
   static inline void Execute(const TFunc & func, const TArgs &... args) noexcept {
      func(0, (args.m_data)...);
   }

   friend inline Cpu_64_Float IfLess(const Cpu_64_Float & cmp1, const Cpu_64_Float & cmp2, const Cpu_64_Float & trueVal, const Cpu_64_Float & falseVal) noexcept {
      return cmp1.m_data < cmp2.m_data ? trueVal : falseVal;
   }

   friend inline Cpu_64_Float IfEqual(const Cpu_64_Float & cmp1, const Cpu_64_Float & cmp2, const Cpu_64_Float & trueVal, const Cpu_64_Float & falseVal) noexcept {
      return cmp1.m_data == cmp2.m_data ? trueVal : falseVal;
   }

   friend inline Cpu_64_Float IfNaN(const Cpu_64_Float & cmp, const Cpu_64_Float & trueVal, const Cpu_64_Float & falseVal) noexcept {
      return std::isnan(cmp.m_data) ? trueVal : falseVal;
   }

   friend inline Cpu_64_Float IfEqual(const Cpu_64_Int & cmp1, const Cpu_64_Int & cmp2, const Cpu_64_Float & trueVal, const Cpu_64_Float & falseVal) noexcept {
      return cmp1.m_data == cmp2.m_data ? trueVal : falseVal;
   }

   friend inline Cpu_64_Float Abs(const Cpu_64_Float & val) noexcept {
      return Cpu_64_Float(std::abs(val.m_data));
   }

   friend inline Cpu_64_Float FastApproxReciprocal(const Cpu_64_Float & val) noexcept {
      return Cpu_64_Float(T { 1.0 } / val.m_data);
   }

   friend inline Cpu_64_Float FastApproxDivide(const Cpu_64_Float & dividend, const Cpu_64_Float & divisor) noexcept {
      return Cpu_64_Float(dividend.m_data / divisor.m_data);
   }

   friend inline Cpu_64_Float FusedMultiplyAdd(const Cpu_64_Float & mul1, const Cpu_64_Float & mul2, const Cpu_64_Float & add) noexcept {
      return mul1 * mul2 + add;
   }

   friend inline Cpu_64_Float FusedNegateMultiplyAdd(const Cpu_64_Float & mul1, const Cpu_64_Float & mul2, const Cpu_64_Float & add) noexcept {
      return add - mul1 * mul2;
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

   template<
      bool bNegateInput = false,
      bool bNaNPossible = true,
      bool bUnderflowPossible = true,
      bool bOverflowPossible = true,
      bool bSpecialCaseZero = false
   >
   static inline Cpu_64_Float ApproxExp(
      const Cpu_64_Float & val, 
      const int32_t addExpSchraudolphTerm = k_expTermZeroMeanErrorForSoftmaxWithZeroedLogit
   ) noexcept {
#ifdef FAST_LOG
      // TODO: we might want different constants for binary classification and multiclass. See notes in approximate_math.hpp
      return Cpu_64_Float(ExpApproxSchraudolph<
         bNegateInput, bNaNPossible, bUnderflowPossible, bOverflowPossible, bSpecialCaseZero
      >(val.m_data, addExpSchraudolphTerm));
#else // FAST_LOG
      return Exp(bNegateInput ? -val : val);
#endif // FAST_LOG
   }

   template<
      bool bNegateOutput = false,
      bool bNaNPossible = true,
      bool bNegativePossible = false,
      bool bZeroPossible = false, // if false, positive zero returns a big negative number, negative zero returns a big positive number
      bool bPositiveInfinityPossible = false // if false, +inf returns a big positive number.  If val can be a double that is above the largest representable float, then setting this is necessary to avoid undefined behavior
   >
   static inline Cpu_64_Float ApproxLog(
      const Cpu_64_Float & val, 
      const float addLogSchraudolphTerm = k_logTermLowerBoundInputCloseToOne
   ) noexcept {
#ifdef FAST_LOG
      return Cpu_64_Float(LogApproxSchraudolph<
         bNegateOutput, bNaNPossible, bNegativePossible, bZeroPossible, bPositiveInfinityPossible
      >(val.m_data, addLogSchraudolphTerm));
#else // FAST_LOG
      const Cpu_64_Float ret = Log(val);
      return bNegateOutput ? -ret : ret;
#endif // FAST_LOG
   }

   friend inline T Sum(const Cpu_64_Float & val) noexcept {
      return val.m_data;
   }


   template<typename TObjective, size_t cCompilerScores, bool bValidation, bool bWeight, bool bHessian, int cCompilerPack>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorApplyUpdate(const Objective * const pObjective, ApplyUpdateBridge * const pData) noexcept {
      RemoteApplyUpdate<TObjective, cCompilerScores, bValidation, bWeight, bHessian, cCompilerPack>(pObjective, pData);
      return Error_None;
   }


   template<bool bHessian, size_t cCompilerScores, bool bWeight, bool bReplication, int cCompilerPack>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorBinSumsBoosting(BinSumsBoostingBridge * const pParams) noexcept {
      RemoteBinSumsBoosting<Cpu_64_Float, bHessian, cCompilerScores, bWeight, bReplication, cCompilerPack>(pParams);
      return Error_None;
   }


   template<bool bHessian, size_t cCompilerScores, size_t cCompilerDimensions, bool bWeight>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorBinSumsInteraction(BinSumsInteractionBridge * const pParams) noexcept {
      RemoteBinSumsInteraction<Cpu_64_Float, bHessian, cCompilerScores, cCompilerDimensions, bWeight>(pParams);
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
INLINE_ALWAYS static std::shared_ptr<const Registration> RegisterObjective(const char * const sRegistrationName, const Args &... args) {
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
   return Objective::CreateObjective(&RegisterObjectives, pConfig, sObjective, sObjectiveEnd, pObjectiveWrapperOut);
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
