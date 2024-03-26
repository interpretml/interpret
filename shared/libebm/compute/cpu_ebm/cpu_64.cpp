// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#define _CRT_SECURE_NO_DEPRECATE

#include <cmath> // exp, log
#include <limits> // numeric_limits
#include <type_traits> // is_unsigned

#include "libebm.h"
#include "logging.h"
#include "unzoned.h"

#define ZONE_cpu
#include "zones.h"

#include "bridge.h"
#include "common.hpp"
#include "bridge.hpp"

#include "Registration.hpp"
#include "Objective.hpp"

#include "approximate_math.hpp"
#include "compute_wrapper.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// this is super-special and included inside the zone namespace
#include "objective_registrations.hpp"

struct Cpu_64_Float;

struct Cpu_64_Int final {
   friend Cpu_64_Float;
   friend inline Cpu_64_Float IfEqual(const Cpu_64_Int& cmp1,
         const Cpu_64_Int& cmp2,
         const Cpu_64_Float& trueVal,
         const Cpu_64_Float& falseVal) noexcept;

   using T = uint64_t;
   using TPack = uint64_t;
   static_assert(std::is_unsigned<T>::value, "T must be an unsigned integer type");
   static_assert(
         std::is_same<UIntBig, T>::value || std::is_same<UIntSmall, T>::value, "T must be either UIntBig or UIntSmall");
   static constexpr AccelerationFlags k_zone = AccelerationFlags_NONE;
   static constexpr int k_cSIMDShift = 0;
   static constexpr int k_cSIMDPack = 1 << k_cSIMDShift;

   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   inline Cpu_64_Int() noexcept {}

   inline Cpu_64_Int(const T& val) noexcept : m_data(val) {}

   inline static Cpu_64_Int Load(const T* const a) noexcept { return Cpu_64_Int(*a); }

   inline void Store(T* const a) const noexcept { *a = m_data; }

   inline static Cpu_64_Int LoadBytes(const uint8_t* const a) noexcept { return Cpu_64_Int(*a); }

   template<typename TFunc, typename... TArgs>
   static inline void Execute(const TFunc& func, const TArgs&... args) noexcept {
      func(0, (args.m_data)...);
   }

   inline static Cpu_64_Int MakeIndexes() noexcept { return Cpu_64_Int(0); }

   inline Cpu_64_Int operator+(const Cpu_64_Int& other) const noexcept { return Cpu_64_Int(m_data + other.m_data); }

   inline Cpu_64_Int operator*(const T& other) const noexcept { return Cpu_64_Int(m_data * other); }

   inline Cpu_64_Int operator>>(int shift) const noexcept { return Cpu_64_Int(m_data >> shift); }

   inline Cpu_64_Int operator<<(int shift) const noexcept { return Cpu_64_Int(m_data << shift); }

   inline Cpu_64_Int operator&(const Cpu_64_Int& other) const noexcept { return Cpu_64_Int(m_data & other.m_data); }

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
   static constexpr AccelerationFlags k_zone = TInt::k_zone;
   static constexpr int k_cSIMDShift = TInt::k_cSIMDShift;
   static constexpr int k_cSIMDPack = TInt::k_cSIMDPack;

   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   inline Cpu_64_Float() noexcept {}

   inline Cpu_64_Float(const double val) noexcept : m_data(static_cast<T>(val)) {}
   inline Cpu_64_Float(const float val) noexcept : m_data(static_cast<T>(val)) {}
   inline Cpu_64_Float(const int val) noexcept : m_data(static_cast<T>(val)) {}

   inline Cpu_64_Float operator+() const noexcept { return *this; }

   inline Cpu_64_Float operator-() const noexcept { return Cpu_64_Float(-m_data); }

   inline Cpu_64_Float operator+(const Cpu_64_Float& other) const noexcept {
      return Cpu_64_Float(m_data + other.m_data);
   }

   inline Cpu_64_Float operator-(const Cpu_64_Float& other) const noexcept {
      return Cpu_64_Float(m_data - other.m_data);
   }

   inline Cpu_64_Float operator*(const Cpu_64_Float& other) const noexcept {
      return Cpu_64_Float(m_data * other.m_data);
   }

   inline Cpu_64_Float operator/(const Cpu_64_Float& other) const noexcept {
      return Cpu_64_Float(m_data / other.m_data);
   }

   inline Cpu_64_Float& operator+=(const Cpu_64_Float& other) noexcept {
      *this = (*this) + other;
      return *this;
   }

   inline Cpu_64_Float& operator-=(const Cpu_64_Float& other) noexcept {
      *this = (*this) - other;
      return *this;
   }

   inline Cpu_64_Float& operator*=(const Cpu_64_Float& other) noexcept {
      *this = (*this) * other;
      return *this;
   }

   inline Cpu_64_Float& operator/=(const Cpu_64_Float& other) noexcept {
      *this = (*this) / other;
      return *this;
   }

   friend inline Cpu_64_Float operator+(const double val, const Cpu_64_Float& other) noexcept {
      return Cpu_64_Float(val) + other;
   }

   friend inline Cpu_64_Float operator-(const double val, const Cpu_64_Float& other) noexcept {
      return Cpu_64_Float(val) - other;
   }

   friend inline Cpu_64_Float operator*(const double val, const Cpu_64_Float& other) noexcept {
      return Cpu_64_Float(val) * other;
   }

   friend inline Cpu_64_Float operator/(const double val, const Cpu_64_Float& other) noexcept {
      return Cpu_64_Float(val) / other;
   }

   friend inline Cpu_64_Float operator+(const float val, const Cpu_64_Float& other) noexcept {
      return Cpu_64_Float(val) + other;
   }

   friend inline Cpu_64_Float operator-(const float val, const Cpu_64_Float& other) noexcept {
      return Cpu_64_Float(val) - other;
   }

   friend inline Cpu_64_Float operator*(const float val, const Cpu_64_Float& other) noexcept {
      return Cpu_64_Float(val) * other;
   }

   friend inline Cpu_64_Float operator/(const float val, const Cpu_64_Float& other) noexcept {
      return Cpu_64_Float(val) / other;
   }

   inline static Cpu_64_Float Load(const T* const a) noexcept { return Cpu_64_Float(*a); }

   inline void Store(T* const a) const noexcept { *a = m_data; }

   inline static Cpu_64_Float Load(const T* const a, const TInt& i) noexcept { return Cpu_64_Float(a[i.m_data]); }

   inline void Store(T* const a, const TInt& i) const noexcept { a[i.m_data] = m_data; }

   template<typename TFunc> friend inline Cpu_64_Float ApplyFunc(const TFunc& func, const Cpu_64_Float& val) noexcept {
      return Cpu_64_Float(func(val.m_data));
   }

   template<typename TFunc, typename... TArgs>
   static inline void Execute(const TFunc& func, const TArgs&... args) noexcept {
      func(0, (args.m_data)...);
   }

   friend inline Cpu_64_Float IfLess(const Cpu_64_Float& cmp1,
         const Cpu_64_Float& cmp2,
         const Cpu_64_Float& trueVal,
         const Cpu_64_Float& falseVal) noexcept {
      return cmp1.m_data < cmp2.m_data ? trueVal : falseVal;
   }

   friend inline Cpu_64_Float IfEqual(const Cpu_64_Float& cmp1,
         const Cpu_64_Float& cmp2,
         const Cpu_64_Float& trueVal,
         const Cpu_64_Float& falseVal) noexcept {
      return cmp1.m_data == cmp2.m_data ? trueVal : falseVal;
   }

   friend inline Cpu_64_Float IfNaN(
         const Cpu_64_Float& cmp, const Cpu_64_Float& trueVal, const Cpu_64_Float& falseVal) noexcept {
      return std::isnan(cmp.m_data) ? trueVal : falseVal;
   }

   friend inline Cpu_64_Float IfEqual(const Cpu_64_Int& cmp1,
         const Cpu_64_Int& cmp2,
         const Cpu_64_Float& trueVal,
         const Cpu_64_Float& falseVal) noexcept {
      return cmp1.m_data == cmp2.m_data ? trueVal : falseVal;
   }

   friend inline Cpu_64_Float Abs(const Cpu_64_Float& val) noexcept { return Cpu_64_Float(std::abs(val.m_data)); }

   friend inline Cpu_64_Float FastApproxReciprocal(const Cpu_64_Float& val) noexcept {
      return Cpu_64_Float(T{1.0} / val.m_data);
   }

   friend inline Cpu_64_Float FastApproxDivide(const Cpu_64_Float& dividend, const Cpu_64_Float& divisor) noexcept {
      return Cpu_64_Float(dividend.m_data / divisor.m_data);
   }

   friend inline Cpu_64_Float FusedMultiplyAdd(
         const Cpu_64_Float& mul1, const Cpu_64_Float& mul2, const Cpu_64_Float& add) noexcept {
      return mul1 * mul2 + add;
   }

   friend inline Cpu_64_Float FusedNegateMultiplyAdd(
         const Cpu_64_Float& mul1, const Cpu_64_Float& mul2, const Cpu_64_Float& add) noexcept {
      return add - mul1 * mul2;
   }

   friend inline Cpu_64_Float Sqrt(const Cpu_64_Float& val) noexcept { return Cpu_64_Float(std::sqrt(val.m_data)); }

   friend inline Cpu_64_Float Exp(const Cpu_64_Float& val) noexcept { return Cpu_64_Float(std::exp(val.m_data)); }

   friend inline Cpu_64_Float Log(const Cpu_64_Float& val) noexcept { return Cpu_64_Float(std::log(val.m_data)); }

   template<bool bDisableApprox,
         bool bNegateInput = false,
         bool bNaNPossible = true,
         bool bUnderflowPossible = true,
         bool bOverflowPossible = true,
         bool bSpecialCaseZero = false,
         typename std::enable_if<bDisableApprox, int>::type = 0>
   static inline Cpu_64_Float ApproxExp(const Cpu_64_Float& val,
         const int32_t addExpSchraudolphTerm = k_expTermZeroMeanErrorForSoftmaxWithZeroedLogit) noexcept {
      UNUSED(addExpSchraudolphTerm);
      return Exp(bNegateInput ? -val : val);
   }

   template<bool bDisableApprox,
         bool bNegateInput = false,
         bool bNaNPossible = true,
         bool bUnderflowPossible = true,
         bool bOverflowPossible = true,
         bool bSpecialCaseZero = false,
         typename std::enable_if<!bDisableApprox, int>::type = 0>
   static inline Cpu_64_Float ApproxExp(const Cpu_64_Float& val,
         const int32_t addExpSchraudolphTerm = k_expTermZeroMeanErrorForSoftmaxWithZeroedLogit) noexcept {
      // TODO: we might want different constants for binary classification and multiclass. See notes in
      // approximate_math.hpp
      return Cpu_64_Float(
            ExpApproxSchraudolph<bNegateInput, bNaNPossible, bUnderflowPossible, bOverflowPossible, bSpecialCaseZero>(
                  val.m_data, addExpSchraudolphTerm));
   }

   template<bool bDisableApprox,
         bool bNegateOutput = false,
         bool bNaNPossible = true,
         bool bNegativePossible = false,
         bool bZeroPossible = false, // if false, positive zero returns a big negative number, negative zero returns a
                                     // big positive number
         bool bPositiveInfinityPossible = false, // if false, +inf returns a big positive number.  If val can be a
                                                 // double that is above the largest representable float, then setting
                                                 // this is necessary to avoid undefined behavior
         typename std::enable_if<bDisableApprox, int>::type = 0>
   static inline Cpu_64_Float ApproxLog(
         const Cpu_64_Float& val, const float addLogSchraudolphTerm = k_logTermLowerBoundInputCloseToOne) noexcept {
      UNUSED(addLogSchraudolphTerm);
      Cpu_64_Float ret = Log(val);
      return bNegateOutput ? -ret : ret;
   }

   template<bool bDisableApprox,
         bool bNegateOutput = false,
         bool bNaNPossible = true,
         bool bNegativePossible = false,
         bool bZeroPossible = false, // if false, positive zero returns a big negative number, negative zero returns a
                                     // big positive number
         bool bPositiveInfinityPossible = false, // if false, +inf returns a big positive number.  If val can be a
                                                 // double that is above the largest representable float, then setting
                                                 // this is necessary to avoid undefined behavior
         typename std::enable_if<!bDisableApprox, int>::type = 0>
   static inline Cpu_64_Float ApproxLog(
         const Cpu_64_Float& val, const float addLogSchraudolphTerm = k_logTermLowerBoundInputCloseToOne) noexcept {
      return Cpu_64_Float(LogApproxSchraudolph<bNegateOutput,
            bNaNPossible,
            bNegativePossible,
            bZeroPossible,
            bPositiveInfinityPossible>(val.m_data, addLogSchraudolphTerm));
   }

   friend inline T Sum(const Cpu_64_Float& val) noexcept { return val.m_data; }

   template<typename TObjective,
         bool bValidation,
         bool bWeight,
         bool bHessian,
         bool bDisableApprox,
         size_t cCompilerScores,
         int cCompilerPack>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorApplyUpdate(
         const Objective* const pObjective, ApplyUpdateBridge* const pData) noexcept {
      RemoteApplyUpdate<TObjective, bValidation, bWeight, bHessian, bDisableApprox, cCompilerScores, cCompilerPack>(
            pObjective, pData);
      return Error_None;
   }

   template<bool bHessian, bool bWeight, size_t cCompilerScores>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorBinSumsBoosting(BinSumsBoostingBridge* const pParams) noexcept {
      RemoteBinSumsBoosting<Cpu_64_Float, bHessian, bWeight, cCompilerScores>(pParams);
      return Error_None;
   }

   template<bool bHessian, bool bWeight, size_t cCompilerScores, size_t cCompilerDimensions>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorBinSumsInteraction(
         BinSumsInteractionBridge* const pParams) noexcept {
      RemoteBinSumsInteraction<Cpu_64_Float, bHessian, bWeight, cCompilerScores, cCompilerDimensions>(pParams);
      return Error_None;
   }

 private:
   TPack m_data;
};
static_assert(std::is_standard_layout<Cpu_64_Float>::value && std::is_trivially_copyable<Cpu_64_Float>::value,
      "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm ApplyUpdate_Cpu_64(
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

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm BinSumsBoosting_Cpu_64(
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

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm BinSumsInteraction_Cpu_64(
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

INTERNAL_IMPORT_EXPORT_BODY double FinishMetricC(
      const ObjectiveWrapper* const pObjectiveWrapper, const double metricSum) {
   const Objective* const pObjective = static_cast<const Objective*>(pObjectiveWrapper->m_pObjective);
   const FINISH_METRIC_CPP pFinishMetricCpp =
         (static_cast<const FunctionPointersCpp*>(pObjectiveWrapper->m_pFunctionPointersCpp))->m_pFinishMetricCpp;
   return (*pFinishMetricCpp)(pObjective, metricSum);
}

INTERNAL_IMPORT_EXPORT_BODY BoolEbm CheckTargetsC(
      const ObjectiveWrapper* const pObjectiveWrapper, const size_t c, const void* const aTargets) {
   EBM_ASSERT(nullptr != pObjectiveWrapper);
   EBM_ASSERT(nullptr != aTargets);
   const Objective* const pObjective = static_cast<const Objective*>(pObjectiveWrapper->m_pObjective);
   EBM_ASSERT(nullptr != pObjective);
   const CHECK_TARGETS_CPP pCheckTargetsCpp =
         (static_cast<const FunctionPointersCpp*>(pObjectiveWrapper->m_pFunctionPointersCpp))->m_pCheckTargetsCpp;
   EBM_ASSERT(nullptr != pCheckTargetsCpp);
   return (*pCheckTargetsCpp)(pObjective, c, aTargets);
}

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm CreateObjective_Cpu_64(const Config* const pConfig,
      const char* const sObjective,
      const char* const sObjectiveEnd,
      ObjectiveWrapper* const pObjectiveWrapperOut) {
   pObjectiveWrapperOut->m_pApplyUpdateC = ApplyUpdate_Cpu_64;
   pObjectiveWrapperOut->m_pBinSumsBoostingC = BinSumsBoosting_Cpu_64;
   pObjectiveWrapperOut->m_pBinSumsInteractionC = BinSumsInteraction_Cpu_64;
   ErrorEbm error = ComputeWrapper<Cpu_64_Float>::FillWrapper(pObjectiveWrapperOut);
   if(Error_None != error) {
      return error;
   }
   return Objective::CreateObjective<Cpu_64_Float>(pConfig, sObjective, sObjectiveEnd, pObjectiveWrapperOut);
}

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm CreateMetric_Cpu_64(
      const Config* const pConfig, const char* const sMetric, const char* const sMetricEnd
      //   MetricWrapper * const pMetricWrapperOut,
) {
   UNUSED(pConfig);
   UNUSED(sMetric);
   UNUSED(sMetricEnd);

   return Error_UnexpectedInternal;
}

} // namespace DEFINED_ZONE_NAME
