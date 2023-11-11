// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef ZONES_H
#define ZONES_H

#include <stdint.h> // uint32_t

#ifdef __cplusplus
extern "C" {
#define STATIC_CAST(type, val)  (static_cast<type>(val))
#else // __cplusplus
#define STATIC_CAST(type, val)  ((type)(val))
#endif // __cplusplus

typedef uint32_t ZoneEbm;

#define ZONE_CAST(val)  (STATIC_CAST(ZoneEbm, (val)))

#define Z_CPU           (ZONE_CAST(0x1))
#define Z_NVIDIA        (ZONE_CAST(0x2))
#define Z_AVX2          (ZONE_CAST(0x4))
#define Z_AVX512F       (ZONE_CAST(0x8))

#define Z_INTEL_SIMD    (Z_AVX2 | Z_AVX512F)
#define Z_SIMD          (Z_INTEL_SIMD)
#define Z_GPU           (Z_NVIDIA)
#define Z_ALL           (ZONE_CAST(~ZONE_CAST(0)))

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
