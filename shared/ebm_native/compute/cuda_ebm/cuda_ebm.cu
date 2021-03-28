
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <type_traits>

struct TestLoss {
   float m_multiple;

   TestLoss(float multiple) {
      m_multiple = multiple;
   }

   __device__ float CalculateGradient(float target, float prediction) {
      return target * m_multiple + prediction * 100;
   }

   // if the loss function doesn't have a second derivative, then delete the CalculateHessian function.
   __device__ float CalculateHessian(float target, float prediction) {
      return target * 10 + prediction * 100;
   }
};

template <typename TLoss>
__global__ void TestGpuAdd(TLoss * pLoss, const int * const pVal1, const int * const pVal2, int * const pResult) {
   const size_t iGpuThread = threadIdx.x;
   pResult[iGpuThread] = pLoss->CalculateGradient(pVal1[iGpuThread], pVal2[iGpuThread]);
}

constexpr size_t k_cItems = 5;

bool TestCuda() {
   bool bExitError = true;

   const int aVal1[k_cItems] = { 5, 4, 3, 2, 1 };
   const int aVal2[k_cItems] = { 100, 200, 300, 400, 500 };
   int aResult[k_cItems];
   memset(aResult, 0, sizeof(aResult));

   // TODO: unfortunately, I think this means our Loss classes need to be standard_layout and trivially copyable
   // which means no virtual function  :(.  We can use function pointers instead though, even though that's kind
   // of uggly, but at least those will be hidden from the Loss class writer.  In Registration.hpp after
   // calling new TRegistrable... we still have the specific loss type after that call, so we can take a pointer
   // to a function that we inject via the loss MACRO.  Dirty, but it'll get the job done.

   static_assert(std::is_standard_layout<TestLoss>::value,
      "Our Loss type must be a standard layout struct to be inserted into the GPU");
   static_assert(std::is_trivially_copyable<TestLoss>::value,
      "Our Loss type must be a trivial struct to be inserted into the GPU");

   TestLoss loss(9);

   int * aDeviceVal1 = nullptr;
   int * aDeviceVal2 = nullptr;
   int * aDeviceResult = nullptr;
   TestLoss * pDeviceLoss = nullptr;
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

   error = cudaMalloc((void **)&pDeviceLoss, sizeof(TestLoss));
   if(cudaSuccess != error) {
      goto exit_error;
   }

   error = cudaMemcpy(aDeviceVal1, aVal1, k_cItems * sizeof(int), cudaMemcpyHostToDevice);
   if(cudaSuccess != error) {
      goto exit_error;
   }

   error = cudaMemcpy(aDeviceVal2, aVal2, k_cItems * sizeof(int), cudaMemcpyHostToDevice);
   if(cudaSuccess != error) {
      goto exit_error;
   }

   error = cudaMemcpy(pDeviceLoss, &loss, sizeof(TestLoss), cudaMemcpyHostToDevice);
   if(cudaSuccess != error) {
      goto exit_error;
   }

   TestGpuAdd<TestLoss><<<1, k_cItems>>>(pDeviceLoss, aDeviceVal1, aDeviceVal2, aDeviceResult);

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

   return bExitError;
}

