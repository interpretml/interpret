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
#elif defined(ZONE_no_cuda)
#define DEFINED_ZONE_NAME      NAMESPACE_COMPUTE_NO_CUDA
#elif defined(ZONE_R)
#define DEFINED_ZONE_NAME      NAMESPACE_R
#else
#error ZONE not recognized
#endif

// we need to nest twice to get the defined value into the token pasted operation:
// https://stackoverflow.com/questions/1597007/creating-c-macro-with-and-line-token-concatenation-with-positioning-macr
#define MAKE_ZONED_C_FUNCTION_NAME_INTERNAL1(__zone_name, __function_name) __zone_name ## _ ## __function_name
#define MAKE_ZONED_C_FUNCTION_NAME_INTERNAL2(__zone_name, __function_name) MAKE_ZONED_C_FUNCTION_NAME_INTERNAL1(__zone_name, __function_name)
#define MAKE_ZONED_C_FUNCTION_NAME(__function_name) MAKE_ZONED_C_FUNCTION_NAME_INTERNAL2(DEFINED_ZONE_NAME, __function_name)

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // ZONES_H
