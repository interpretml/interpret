// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef ZONES_H
#define ZONES_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if defined(ZONE_main)
#define DEFINED_ZONE_NAME      NAMESPACE_MAIN
#elif defined(ZONE_cpu)
#define DEFINED_ZONE_NAME      NAMESPACE_COMPUTE_CPU
#elif defined(ZONE_avx512)
#define DEFINED_ZONE_NAME      NAMESPACE_COMPUTE_AVX512
#elif defined(ZONE_cuda)
#define DEFINED_ZONE_NAME      NAMESPACE_COMPUTE_CUDA
#elif defined(ZONE_cuda_missing)
#define DEFINED_ZONE_NAME      NAMESPACE_COMPUTE_CUDA_MISSING
#else
#error ZONE not recognized
#endif

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // ZONES_H
