
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ class Addme {

};

__global__ void TestGpuAdd(const int * const pVal1, const int * const pVal2, int * const pResult) {
   Addme();
   const size_t iGpuThread = threadIdx.x;
   pResult[iGpuThread] = pVal1[iGpuThread] + pVal2[iGpuThread];
}

constexpr size_t k_cItems = 5;

bool TestCuda() {
   bool bExitError = true;

   const int aVal1[k_cItems] = { 5, 4, 3, 2, 1 };
   const int aVal2[k_cItems] = { 100, 200, 300, 400, 500 };
   int aResult[k_cItems];
   memset(aResult, 0, sizeof(aResult));

   int * aDeviceVal1 = nullptr;
   int * aDeviceVal2 = nullptr;
   int * aDeviceResult = nullptr;
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

   error = cudaMemcpy(aDeviceVal1, aVal1, k_cItems * sizeof(int), cudaMemcpyHostToDevice);
   if(cudaSuccess != error) {
      goto exit_error;
   }

   error = cudaMemcpy(aDeviceVal2, aVal2, k_cItems * sizeof(int), cudaMemcpyHostToDevice);
   if(cudaSuccess != error) {
      goto exit_error;
   }

   TestGpuAdd<<<1, k_cItems>>>(aDeviceVal1, aDeviceVal2, aDeviceResult);

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

