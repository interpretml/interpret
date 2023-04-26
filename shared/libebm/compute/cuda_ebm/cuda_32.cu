// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
#include "compute_stats.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template <typename TObjective>
GPU_GLOBAL void TestGpuAdd(const Objective * const pObjective, const int * const pVal1, const int * const pVal2, int * const pResult) {
   TObjective * const pObjectiveSpecific = static_cast<TObjective *>(pObjective);
   const size_t iGpuThread = threadIdx.x;
//   pResult[iGpuThread] = static_cast<int>(static_cast<float>(pObjectiveSpecific->CalculateGradient(static_cast<float>(pVal1[iGpuThread]), static_cast<float>(pVal2[iGpuThread]))));
}

struct Cuda_32_Int final {
   static constexpr int cPack = 1;
   using T = uint32_t;

   GPU_BOTH inline Cuda_32_Int(const uint32_t val) noexcept : m_data(val) {
   }

private:
   uint32_t m_data;
};
static_assert(std::is_standard_layout<Cuda_32_Int>::value && std::is_trivially_copyable<Cuda_32_Int>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");


struct Cuda_32_Float final {
   // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE
   // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE

   static constexpr bool bCpu = false;
   static constexpr int cPack = 1;
   using T = float;
   using TInt = Cuda_32_Int;

   WARNING_PUSH
   ATTRIBUTE_WARNING_DISABLE_UNINITIALIZED_MEMBER
   GPU_BOTH inline Cuda_32_Float() noexcept {
   }
   WARNING_POP

   Cuda_32_Float(const Cuda_32_Float & other) noexcept = default; // preserve POD status
   Cuda_32_Float & operator=(const Cuda_32_Float &) noexcept = default; // preserve POD status

   GPU_BOTH inline Cuda_32_Float(const double val) noexcept : m_data { static_cast<T>(val) } {
   }

   GPU_BOTH inline Cuda_32_Float & operator= (const double val) noexcept {
      m_data = static_cast<T>(val);
      return *this;
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

   GPU_BOTH inline void LoadAligned(const T * const a) noexcept {
      m_data = *a;
   }

   GPU_BOTH inline void SaveAligned(T * const a) const noexcept {
      *a = m_data;
   }

   template<typename TFunc>
   GPU_BOTH friend inline Cuda_32_Float ApplyFunction(const Cuda_32_Float & val, const TFunc & func) noexcept {
      // this function is more useful for a SIMD operator where it applies func() to all packed items
      return Cuda_32_Float(func(val.m_data));
   }

   GPU_BOTH friend inline Cuda_32_Float IfGreater(const Cuda_32_Float & cmp1, const Cuda_32_Float & cmp2, const Cuda_32_Float & trueVal, const Cuda_32_Float & falseVal) noexcept {
      return cmp1.m_data > cmp2.m_data ? trueVal : falseVal;
   }

   GPU_BOTH friend inline Cuda_32_Float IfLess(const Cuda_32_Float & cmp1, const Cuda_32_Float & cmp2, const Cuda_32_Float & trueVal, const Cuda_32_Float & falseVal) noexcept {
      return cmp1.m_data < cmp2.m_data ? trueVal : falseVal;
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

   GPU_BOTH friend inline T Sum(const Cuda_32_Float & val) noexcept {
      return val.m_data;
   }


   template<typename TObjective, size_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian, bool bKeepGradHess, bool bCalcMetric, bool bWeight>
   INLINE_RELEASE_TEMPLATED static ErrorEbm OperatorApplyUpdate(const Objective * const pObjective, ApplyUpdateBridge * const pData) noexcept {
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
      RemoteApplyUpdate<TObjective, cCompilerScores, cCompilerPack, bHessian, bKeepGradHess, bCalcMetric, bWeight><<<1, k_cItems>>>(pObjective, pData);

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

private:

   float m_data;
};
static_assert(std::is_standard_layout<Cuda_32_Float>::value && std::is_trivially_copyable<Cuda_32_Float>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

// FIRST, define the RegisterObjective function that we'll be calling from our registrations.  This is a static 
// function, so we can have duplicate named functions in other files and they'll refer to different functions
template<template <typename> class TRegistrable, typename... Args>
INLINE_ALWAYS static std::shared_ptr<const Registration> RegisterObjective(const char * const sRegistrationName, const Args...args) {
   return Register<TRegistrable, Cuda_32_Float>(sRegistrationName, args...);
}

// now include all our special objective registrations which will use the RegisterObjective function we defined above!
#include "objective_registrations.hpp"

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm CreateObjective_Cuda_32(
   const Config * const pConfig,
   const char * const sObjective,
   const char * const sObjectiveEnd,
   ObjectiveWrapper * const pObjectiveWrapperOut
) {
   return Objective::CreateObjective(&RegisterObjectives, pConfig, sObjective, sObjectiveEnd, pObjectiveWrapperOut);
}

} // DEFINED_ZONE_NAME
