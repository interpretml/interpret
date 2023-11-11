// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef ZONES_H
#define ZONES_H

#include <stdint.h> // uint32_t

#ifdef __cplusplus
extern "C" {
#else // __cplusplus
#endif // __cplusplus

#if defined(ZONE_main)
#define DEFINED_ZONE_NAME      NAMESPACE_MAIN
#elif defined(ZONE_cpu)
#define DEFINED_ZONE_NAME      NAMESPACE_CPU
#elif defined(ZONE_avx2)
#define DEFINED_ZONE_NAME      NAMESPACE_AVX2
#elif defined(ZONE_avx512f)
#define DEFINED_ZONE_NAME      NAMESPACE_AVX512F
#elif defined(ZONE_cuda)
#define DEFINED_ZONE_NAME      NAMESPACE_CUDA
#else
#error ZONE not recognized
#endif

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // ZONES_H
