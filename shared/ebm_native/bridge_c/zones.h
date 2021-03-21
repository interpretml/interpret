// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef ZONES_H
#define ZONES_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if defined(MAIN_ZONE)
#define DEFINED_ZONE_NAME      NAMESPACE_MAIN
#elif defined(CPU_ZONE)
#define DEFINED_ZONE_NAME      NAMESPACE_COMPUTE_CPU
#elif defined(AVX512_ZONE)
#define DEFINED_ZONE_NAME      NAMESPACE_COMPUTE_AVX512
#else
#error ZONE not recognized
#endif

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // ZONES_H
