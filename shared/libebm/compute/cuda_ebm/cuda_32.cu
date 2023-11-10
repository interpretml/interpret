// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <limits> // numeric_limits
#include <type_traits>

#include "libebm.h"
#include "logging.h"
#include "unzoned.h"

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

template <typename TObjective>
GPU_GLOBAL void TestGpuAdd(const Objective * const pObjective, const int * const pVal1, const int * const pVal2, int * const pResult) {
   TObjective * const pObjectiveSpecific = static_cast<TObjective *>(pObjective);
   const size_t iGpuThread = threadIdx.x;
//   pResult[iGpuThread] = static_cast<int>(static_cast<float>(pObjectiveSpecific->CalculateGradient(static_cast<float>(pVal1[iGpuThread]), static_cast<float>(pVal2[iGpuThread]))));
}

struct Cuda_32_Float;

struct Cuda_32_Int final {
   friend Cuda_32_Float;
   GPU_BOTH friend inline Cuda_32_Float IfEqual(const Cuda_32_Int & cmp1, const Cuda_32_Int & cmp2, const Cuda_32_Float & trueVal, const Cuda_32_Float & falseVal) noexcept;

   using T = uint32_t;
   using TPack = uint32_t;
   static_assert(std::is_unsigned<T>::value, "T must be an unsigned integer type");
   static_assert(std::is_same<UIntBig, T>::value || std::is_same<UIntSmall, T>::value,
      "T must be either UIntBig or UIntSmall");
   static constexpr bool k_bCpu = false;
   static constexpr int k_cSIMDShift = 0;
   static constexpr int k_cSIMDPack = 1 << k_cSIMDShift;

   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   GPU_BOTH inline Cuda_32_Int() noexcept {
   }

   GPU_BOTH inline Cuda_32_Int(const T & val) noexcept : m_data(val) {
   }

   GPU_BOTH inline static Cuda_32_Int Load(const T * const a) noexcept {
      return Cuda_32_Int(*a);
   }

   GPU_BOTH inline void Store(T * const a) const noexcept {
      *a = m_data;
   }

   GPU_BOTH inline static Cuda_32_Int LoadBytes(const uint8_t * const a) noexcept {
      return Cuda_32_Int(*a);
   }

   template<typename TFunc, typename... TArgs>
   GPU_BOTH static inline void Execute(const TFunc & func, const TArgs &... args) noexcept {
      func(0, (args.m_data)...);
   }

   GPU_BOTH inline static Cuda_32_Int MakeIndexes() noexcept {
      return Cuda_32_Int(0);
   }

   GPU_BOTH inline Cuda_32_Int operator+ (const Cuda_32_Int & other) const noexcept {
      return Cuda_32_Int(m_data + other.m_data);
   }

   GPU_BOTH inline Cuda_32_Int operator* (const T & other) const noexcept {
      return Cuda_32_Int(m_data * other);
   }

   GPU_BOTH inline Cuda_32_Int operator>> (int shift) const noexcept {
      return Cuda_32_Int(m_data >> shift);
   }

   GPU_BOTH inline Cuda_32_Int operator<< (int shift) const noexcept {
      return Cuda_32_Int(m_data << shift);
   }

   GPU_BOTH inline Cuda_32_Int operator& (const Cuda_32_Int & other) const noexcept {
      return Cuda_32_Int(m_data & other.m_data);
   }

private:
   TPack m_data;
};
static_assert(std::is_standard_layout<Cuda_32_Int>::value && std::is_trivially_copyable<Cuda_32_Int>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");


struct Cuda_32_Float final {
   // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE
   // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE

   using T = float;
   using TPack = float;
   using TInt = Cuda_32_Int;
   static_assert(std::is_same<FloatBig, T>::value || std::is_same<FloatSmall, T>::value,
      "T must be either FloatBig or FloatSmall");
   static constexpr bool k_bCpu = TInt::k_bCpu;
   static constexpr int k_cSIMDShift = TInt::k_cSIMDShift;
   static constexpr int k_cSIMDPack = TInt::k_cSIMDPack;

   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   GPU_BOTH inline Cuda_32_Float() noexcept {
   }

   GPU_BOTH inline Cuda_32_Float(const double val) noexcept : m_data(static_cast<T>(val)) {
   }
   GPU_BOTH inline Cuda_32_Float(const float val) noexcept : m_data(static_cast<T>(val)) {
   }
   GPU_BOTH inline Cuda_32_Float(const int val) noexcept : m_data(static_cast<T>(val)) {
   }


   GPU_BOTH inline Cuda_32_Float operator+() const noexcept {
      return *this;
   }

   GPU_BOTH inline Cuda_32_Float operator-() const noexcept {
      return Cuda_32_Float(-m_data);
   }


   GPU_BOTH inline Cuda_32_Float operator+ (const Cuda_32_Float & other) const noexcept {
      return Cuda_32_Float(m_data + other.m_data);
   }

   GPU_BOTH inline Cuda_32_Float operator- (const Cuda_32_Float & other) const noexcept {
      return Cuda_32_Float(m_data - other.m_data);
   }

   GPU_BOTH inline Cuda_32_Float operator* (const Cuda_32_Float & other) const noexcept {
      return Cuda_32_Float(m_data * other.m_data);
   }

   GPU_BOTH inline Cuda_32_Float operator/ (const Cuda_32_Float & other) const noexcept {
      return Cuda_32_Float(m_data / other.m_data);
   }


   GPU_BOTH inline Cuda_32_Float & operator+= (const Cuda_32_Float & other) noexcept {
      *this = (*this) + other;
      return *this;
   }

   GPU_BOTH inline Cuda_32_Float & operator-= (const Cuda_32_Float & other) noexcept {
      *this = (*this) - other;
      return *this;
   }

   GPU_BOTH inline Cuda_32_Float & operator*= (const Cuda_32_Float & other) noexcept {
      *this = (*this) * other;
      return *this;
   }

   GPU_BOTH inline Cuda_32_Float & operator/= (const Cuda_32_Float & other) noexcept {
      *this = (*this) / other;
      return *this;
   }


   GPU_BOTH friend inline Cuda_32_Float operator+ (const double val, const Cuda_32_Float & other) noexcept {
      return Cuda_32_Float(val) + other;
   }

   GPU_BOTH friend inline Cuda_32_Float operator- (const double val, const Cuda_32_Float & other) noexcept {
      return Cuda_32_Float(val) - other;
   }

   GPU_BOTH friend inline Cuda_32_Float operator* (const double val, const Cuda_32_Float & other) noexcept {
      return Cuda_32_Float(val) * other;
   }

   GPU_BOTH friend inline Cuda_32_Float operator/ (const double val, const Cuda_32_Float & other) noexcept {
      return Cuda_32_Float(val) / other;
   }


   GPU_BOTH friend inline Cuda_32_Float operator+ (const float val, const Cuda_32_Float & other) noexcept {
      return Cuda_32_Float(val) + other;
   }

   GPU_BOTH friend inline Cuda_32_Float operator- (const float val, const Cuda_32_Float & other) noexcept {
      return Cuda_32_Float(val) - other;
   }

   GPU_BOTH friend inline Cuda_32_Float operator* (const float val, const Cuda_32_Float & other) noexcept {
      return Cuda_32_Float(val) * other;
   }

   GPU_BOTH friend inline Cuda_32_Float operator/ (const float val, const Cuda_32_Float & other) noexcept {
      return Cuda_32_Float(val) / other;
   }


   GPU_BOTH inline static Cuda_32_Float Load(const T * const a) noexcept {
      return Cuda_32_Float(*a);
   }

   GPU_BOTH inline void Store(T * const a) const noexcept {
      *a = m_data;
   }

   GPU_BOTH inline static Cuda_32_Float Load(const T * const a, const TInt & i) noexcept {
      return Cuda_32_Float(a[i.m_data]);
   }

   GPU_BOTH inline void Store(T * const a, const TInt & i) const noexcept {
      a[i.m_data] = m_data;
   }

   template<typename TFunc>
   GPU_BOTH friend inline Cuda_32_Float ApplyFunc(const TFunc & func, const Cuda_32_Float & val) noexcept {
      return Cuda_32_Float(func(val.m_data));
   }

   template<typename TFunc, typename... TArgs>
   GPU_BOTH static inline void Execute(const TFunc & func, const TArgs &... args) noexcept {
      func(0, (args.m_data)...);
   }

   GPU_BOTH friend inline Cuda_32_Float IfLess(const Cuda_32_Float & cmp1, const Cuda_32_Float & cmp2, const Cuda_32_Float & trueVal, const Cuda_32_Float & falseVal) noexcept {
      return cmp1.m_data < cmp2.m_data ? trueVal : falseVal;
   }

   GPU_BOTH friend inline Cuda_32_Float IfEqual(const Cuda_32_Float & cmp1, const Cuda_32_Float & cmp2, const Cuda_32_Float & trueVal, const Cuda_32_Float & falseVal) noexcept {
      return cmp1.m_data == cmp2.m_data ? trueVal : falseVal;
   }

   GPU_BOTH friend inline Cuda_32_Float IfNaN(const Cuda_32_Float & cmp, const Cuda_32_Float & trueVal, const Cuda_32_Float & falseVal) noexcept {
      return isnan(cmp.m_data) ? trueVal : falseVal;
   }

   GPU_BOTH friend inline Cuda_32_Float IfEqual(const Cuda_32_Int & cmp1, const Cuda_32_Int & cmp2, const Cuda_32_Float & trueVal, const Cuda_32_Float & falseVal) noexcept {
      return cmp1.m_data == cmp2.m_data ? trueVal : falseVal;
   }

   GPU_BOTH friend inline Cuda_32_Float Abs(const Cuda_32_Float & val) noexcept {
      return Cuda_32_Float(fabsf(val.m_data));
   }

   GPU_BOTH friend inline Cuda_32_Float FastApproxReciprocal(const Cuda_32_Float & val) noexcept {
      return Cuda_32_Float(T { 1.0 } / val.m_data);
   }

   GPU_BOTH friend inline Cuda_32_Float FastApproxDivide(const Cuda_32_Float & dividend, const Cuda_32_Float & divisor) noexcept {
      return Cuda_32_Float(dividend.m_data / divisor.m_data);
   }

   GPU_BOTH friend inline Cuda_32_Float FusedMultiplyAdd(const Cuda_32_Float & mul1, const Cuda_32_Float & mul2, const Cuda_32_Float & add) noexcept {
      return mul1 * mul2 + add;
   }

   GPU_BOTH friend inline Cuda_32_Float FusedNegateMultiplyAdd(const Cuda_32_Float & mul1, const Cuda_32_Float & mul2, const Cuda_32_Float & add) noexcept {
      return add - mul1 * mul2;
   }

   GPU_BOTH friend inline Cuda_32_Float Sqrt(const Cuda_32_Float & val) noexcept {
      return Cuda_32_Float(sqrtf(val.m_data));
   }

   GPU_BOTH friend inline Cuda_32_Float Exp(const Cuda_32_Float & val) noexcept {
      return Cuda_32_Float(expf(val.m_data));
   }

   GPU_BOTH friend inline Cuda_32_Float Log(const Cuda_32_Float & val) noexcept {
      return Cuda_32_Float(logf(val.m_data));
   }

   template<
      bool bNegateInput = false,
      bool bNaNPossible = true,
      bool bUnderflowPossible = true,
      bool bOverflowPossible = true,
      bool bSpecialCaseZero = false
   >
   GPU_DEVICE static inline Cuda_32_Float ApproxExp(
      const Cuda_32_Float & val, 
      const int32_t addExpSchraudolphTerm = k_expTermZeroMeanErrorForSoftmaxWithZeroedLogit
   ) noexcept {
#ifdef FAST_LOG
      // TODO: we might want different constants for binary classification and multiclass. See notes in approximate_math.hpp
      return Cuda_32_Float(ExpApproxSchraudolph<
         bNegateInput, bNaNPossible, bUnderflowPossible, bOverflowPossible, bSpecialCaseZero
      >(val.m_data, addExpSchraudolphTerm));
#else // FAST_LOG
      UNUSED(addExpSchraudolphTerm);
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
   GPU_DEVICE static inline Cuda_32_Float ApproxLog(
      const Cuda_32_Float & val, 
      const float addLogSchraudolphTerm = k_logTermLowerBoundInputCloseToOne
   ) noexcept {
#ifdef FAST_LOG
      return Cuda_32_Float(LogApproxSchraudolph<
         bNegateOutput, bNaNPossible, bNegativePossible, bZeroPossible, bPositiveInfinityPossible
      >(val.m_data, addLogSchraudolphTerm));
#else // FAST_LOG
      UNUSED(addLogSchraudolphTerm);
      const Cuda_32_Float ret = Log(val);
      return bNegateOutput ? -ret : ret;
#endif // FAST_LOG
   }

   GPU_BOTH friend inline T Sum(const Cuda_32_Float & val) noexcept {
      return val.m_data;
   }


   template<typename TObjective, size_t cCompilerScores, bool bValidation, bool bWeight, bool bHessian, int cCompilerPack>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorApplyUpdate(const Objective * const pObjective, ApplyUpdateBridge * const pData) noexcept {
      // TODO: currently we're duplicating all this code to interface with the GPU for each and every templated
      //       set of options. We should instead call a single non-templated function to handle most of this where
      //       the code can be shared accross instanciations, and then make the final call to RemoteApplyUpdate from 
      //       this function

      static constexpr size_t k_cItems = 5;

      bool bExitError = true;

      const int aVal1[k_cItems] = { 5, 4, 3, 2, 1 };
      const int aVal2[k_cItems] = { 100, 200, 300, 400, 500 };
      int aResult[k_cItems];

      static_assert(std::is_standard_layout<TObjective>::value && std::is_trivially_copyable<TObjective>::value,
         "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

      int * aDeviceVal1 = nullptr;
      int * aDeviceVal2 = nullptr;
      int * aDeviceResult = nullptr;
      void * pDeviceObjective = nullptr;
      cudaError_t error;

      error = cudaSetDevice(0);
      if(cudaSuccess != error) {
         goto exit_error;
      }

      error = cudaMalloc((void **)&aDeviceVal1, k_cItems * sizeof(int));
      if(cudaSuccess != error) {
         goto exit_error;
      }

      error = cudaMalloc((void **)&aDeviceVal2, k_cItems * sizeof(int));
      if(cudaSuccess != error) {
         goto exit_error;
      }

      error = cudaMalloc((void **)&aDeviceResult, k_cItems * sizeof(int));
      if(cudaSuccess != error) {
         goto exit_error;
      }

      if(!std::is_empty<TObjective>::value) {
         error = cudaMalloc((void **)&pDeviceObjective, sizeof(TObjective));
         if(cudaSuccess != error) {
            goto exit_error;
         }
         error = cudaMemcpy(pDeviceObjective, pObjective, sizeof(TObjective), cudaMemcpyHostToDevice);
         if(cudaSuccess != error) {
            goto exit_error;
         }
      }

      error = cudaMemcpy(aDeviceVal1, aVal1, k_cItems * sizeof(int), cudaMemcpyHostToDevice);
      if(cudaSuccess != error) {
         goto exit_error;
      }

      error = cudaMemcpy(aDeviceVal2, aVal2, k_cItems * sizeof(int), cudaMemcpyHostToDevice);
      if(cudaSuccess != error) {
         goto exit_error;
      }

      TestGpuAdd<TObjective><<<1, k_cItems>>>(static_cast<Objective *>(pDeviceObjective), aDeviceVal1, aDeviceVal2, aDeviceResult);
      RemoteApplyUpdate<TObjective, cCompilerScores, bValidation, bWeight, bHessian, cCompilerPack><<<1, k_cItems>>>(pObjective, pData);

      error = cudaGetLastError();
      if(cudaSuccess != error) {
         goto exit_error;
      }

      error = cudaDeviceSynchronize();
      if(cudaSuccess != error) {
         goto exit_error;
      }

      error = cudaMemcpy(aResult, aDeviceResult, k_cItems * sizeof(int), cudaMemcpyDeviceToHost);
      if(cudaSuccess != error) {
         goto exit_error;
      }

      bExitError = false;

   exit_error:

      bool bExitHard = false;

      if(nullptr != pDeviceObjective) {
         error = cudaFree(pDeviceObjective);
         if(cudaSuccess != error) {
            bExitHard = true;
         }
      }

      if(nullptr != aDeviceResult) {
         error = cudaFree(aDeviceResult);
         if(cudaSuccess != error) {
            bExitHard = true;
         }
      }

      if(nullptr != aDeviceVal2) {
         error = cudaFree(aDeviceVal2);
         if(cudaSuccess != error) {
            bExitHard = true;
         }
      }

      if(nullptr != aDeviceVal1) {
         error = cudaFree(aDeviceVal1);
         if(cudaSuccess != error) {
            bExitHard = true;
         }
      }

      if(bExitHard) {
         bExitError = true;

         // not much to do with the error if we fail cudaDeviceReset after failing cudaFree
         error = cudaDeviceReset();
      }

      return bExitError ? Error_UnexpectedInternal : Error_None;
   }




   template<bool bHessian, size_t cCompilerScores, bool bWeight, bool bReplication, int cCompilerPack>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorBinSumsBoosting(BinSumsBoostingBridge * const pParams) noexcept {
      // TODO: move memory to the GPU and return errors
      static constexpr size_t k_cItems = 5;
      RemoteBinSumsBoosting<Cuda_32_Float, bHessian, cCompilerScores, bWeight, bReplication, cCompilerPack><<<1, k_cItems>>>(pParams);
      return Error_None;
   }


   template<bool bHessian, size_t cCompilerScores, size_t cCompilerDimensions, bool bWeight>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorBinSumsInteraction(BinSumsInteractionBridge * const pParams) noexcept {
      // TODO: move memory to the GPU and return errors
      static constexpr size_t k_cItems = 5;
      RemoteBinSumsInteraction<Cuda_32_Float, bHessian, cCompilerScores, cCompilerDimensions, bWeight><<<1, k_cItems>>>(pParams);
      return Error_None;
   }



private:

   TPack m_data;
};
static_assert(std::is_standard_layout<Cuda_32_Float>::value && std::is_trivially_copyable<Cuda_32_Float>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");


INTERNAL_IMPORT_EXPORT_BODY ErrorEbm CreateObjective_Cuda_32(
   const Config * const pConfig,
   const char * const sObjective,
   const char * const sObjectiveEnd,
   ObjectiveWrapper * const pObjectiveWrapperOut
) {
   pObjectiveWrapperOut->m_pApplyUpdateC = ApplyUpdate_Cuda_32;
   pObjectiveWrapperOut->m_pBinSumsBoostingC = BinSumsBoosting_Cuda_32;
   pObjectiveWrapperOut->m_pBinSumsInteractionC = BinSumsInteraction_Cuda_32;
   ErrorEbm error = ComputeWrapper<Cuda_32_Float>::FillWrapper(pObjectiveWrapperOut);
   if(Error_None != error) {
      return error;
   }
   return Objective::CreateObjective<Cuda_32_Float>(pConfig, sObjective, sObjectiveEnd, pObjectiveWrapperOut);
}

} // DEFINED_ZONE_NAME
