// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <type_traits>

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

template <typename TLoss>
GPU_GLOBAL void TestGpuAdd(const Loss * const pLoss, const int * const pVal1, const int * const pVal2, int * const pResult) {
   TLoss * const pLossSpecific = static_cast<TLoss *>(pLoss);
   const size_t iGpuThread = threadIdx.x;
   pResult[iGpuThread] = static_cast<int>(static_cast<float>(pLossSpecific->CalculateGradient(static_cast<float>(pVal1[iGpuThread]), static_cast<float>(pVal2[iGpuThread]))));
}

struct Cuda_32_Operators final {
   // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE
   // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE

   static constexpr size_t countPackedItems = 1; // the number of Unpacked items in a Packed structure
   typedef float Unpacked;
   typedef float Packed;

private:

   Packed m_data;

public:

   GPU_BOTH INLINE_ALWAYS Cuda_32_Operators() noexcept {
   }

   GPU_BOTH INLINE_ALWAYS Cuda_32_Operators(const float data) noexcept : m_data(static_cast<Unpacked>(data)) {
   }

   GPU_BOTH INLINE_ALWAYS Cuda_32_Operators(const double data) noexcept : m_data(static_cast<Unpacked>(data)) {
   }

   GPU_BOTH INLINE_ALWAYS Cuda_32_Operators(const int data) noexcept : m_data(static_cast<Unpacked>(data)) {
   }

   GPU_BOTH INLINE_ALWAYS Cuda_32_Operators operator+ (const Cuda_32_Operators & other) const noexcept {
      return Cuda_32_Operators(m_data + other.m_data);
   }

   GPU_BOTH INLINE_ALWAYS Cuda_32_Operators operator- (const Cuda_32_Operators & other) const noexcept {
      return Cuda_32_Operators(m_data - other.m_data);
   }

   GPU_BOTH INLINE_ALWAYS Cuda_32_Operators operator* (const Cuda_32_Operators & other) const noexcept {
      return Cuda_32_Operators(m_data * other.m_data);
   }

   GPU_BOTH INLINE_ALWAYS Cuda_32_Operators operator/ (const Cuda_32_Operators & other) const noexcept {
      return Cuda_32_Operators(m_data / other.m_data);
   }

   GPU_BOTH INLINE_ALWAYS bool IsAnyEqual(const Cuda_32_Operators & other) const noexcept {
      return m_data == other.m_data;
   }

   GPU_BOTH INLINE_ALWAYS operator float() const noexcept {
      return m_data;
   }

   GPU_BOTH INLINE_ALWAYS operator double() const noexcept {
      return m_data;
   }

   GPU_BOTH INLINE_ALWAYS bool IsAnyInf() const noexcept {
      return isinf(m_data);
   }

   GPU_BOTH INLINE_ALWAYS bool IsAnyNaN() const noexcept {
      return isnan(m_data);
   }

   GPU_BOTH INLINE_ALWAYS Cuda_32_Operators Sqrt() const noexcept {
      return Cuda_32_Operators(sqrtf(m_data));
   }

   template<template <typename, typename, ptrdiff_t, ptrdiff_t, bool> class TExecute, typename TLoss, typename TFloat, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian>
   INLINE_RELEASE_TEMPLATED static ErrorEbm ApplyTraining(const Loss * const pLoss, ApplyTrainingData * const pData) noexcept {
      static constexpr size_t k_cItems = 5;

      bool bExitError = true;

      const int aVal1[k_cItems] = { 5, 4, 3, 2, 1 };
      const int aVal2[k_cItems] = { 100, 200, 300, 400, 500 };
      int aResult[k_cItems];

      static_assert(std::is_standard_layout<TLoss>::value &&
         std::is_trivially_copyable<TLoss>::value,
         "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

      int * aDeviceVal1 = nullptr;
      int * aDeviceVal2 = nullptr;
      int * aDeviceResult = nullptr;
      void * pDeviceLoss = nullptr;
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

      if(!std::is_empty<TLoss>::value) {
         error = cudaMalloc((void **)&pDeviceLoss, sizeof(TLoss));
         if(cudaSuccess != error) {
            goto exit_error;
         }
         error = cudaMemcpy(pDeviceLoss, pLoss, sizeof(TLoss), cudaMemcpyHostToDevice);
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

      TestGpuAdd<TLoss><<<1, k_cItems>>>(static_cast<Loss *>(pDeviceLoss), aDeviceVal1, aDeviceVal2, aDeviceResult);
      ExecuteApplyTraining<TExecute, TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian><<<1, k_cItems>>>(
         pLoss,
         pData->m_cRuntimeScores,
         pData->m_cRuntimePack
      );

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

      if(nullptr != pDeviceLoss) {
         error = cudaFree(pDeviceLoss);
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

   template<template <typename, typename, ptrdiff_t, ptrdiff_t, bool> class TExecute, typename TLoss, typename TFloat, ptrdiff_t cCompilerScores, ptrdiff_t cCompilerPack, bool bHessian>
   INLINE_RELEASE_TEMPLATED static ErrorEbm ApplyValidation(const Loss * const pLoss, ApplyValidationData * const pData) noexcept {
      // this allows us to switch execution onto GPU, FPGA, or other local computation

      // TODO: use something other than <<<1, 1>>>
      ExecuteApplyValidation<TExecute, TLoss, TFloat, cCompilerScores, cCompilerPack, bHessian><<<1, 1>>>(
         pLoss,
         pData->m_cRuntimeScores,
         pData->m_cRuntimePack,
         nullptr
      );
      return Error_None;
   }
};
static_assert(std::is_standard_layout<Cuda_32_Operators>::value &&
   std::is_trivially_copyable<Cuda_32_Operators>::value,
   "This allows offsetof, memcpy, memset, inter-language, GPU and cross-machine use where needed");

// FIRST, define the RegisterLoss function that we'll be calling from our registrations.  This is a static 
// function, so we can have duplicate named functions in other files and they'll refer to different functions
template<template <typename> class TRegistrable, typename... Args>
INLINE_ALWAYS static std::shared_ptr<const Registration> RegisterLoss(const char * const sRegistrationName, const Args...args) {
   return Register<TRegistrable, Cuda_32_Operators>(sRegistrationName, args...);
}

// now include all our special loss registrations which will use the RegisterLoss function we defined above!
#include "loss_registrations.hpp"

INTERNAL_IMPORT_EXPORT_BODY ErrorEbm CreateLoss_Cuda_32(
   const Config * const pConfig,
   const char * const sLoss,
   const char * const sLossEnd,
   LossWrapper * const pLossWrapperOut
) {
   return Loss::CreateLoss(&RegisterLosses, pConfig, sLoss, sLossEnd, pLossWrapperOut);
}

} // DEFINED_ZONE_NAME
