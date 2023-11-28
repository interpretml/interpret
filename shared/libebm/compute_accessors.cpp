// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#if defined(BRIDGE_AVX512F_32) || defined(BRIDGE_AVX2_32)
#define INTEL_SIMD
#endif

#ifdef INTEL_SIMD

#ifdef _MSC_VER
#include <intrin.h>
#else // compiler type
// clang or gcc
#include <x86intrin.h>
#endif // compiler type

#endif // INTEL_SIMD

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "unzoned.h"

#include "zones.h"
#include "bridge.h" // CreateObjective_*
#include "common.hpp" // INLINE_RELEASE_UNTEMPLATED

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

#ifdef INTEL_SIMD

inline static void cpuid(int cpuInfo[4], int function_id, int subfunction_id = 0) {
   // from: https://github.com/vectorclass/version2/blob/9d324e13457cf67b44be04f49a4a0036bb188a89/instrset.h#L217
#if defined (_MSC_VER)
   __cpuidex(cpuInfo, function_id, subfunction_id);
#elif defined(__GNUC__) || defined(__clang__)
   int a, b, c, d;
   __asm("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(function_id), "c"(subfunction_id) : );
   cpuInfo[0] = a;
   cpuInfo[1] = b;
   cpuInfo[2] = c;
   cpuInfo[3] = d;
#else
   // Some other platform. Try masm/intel syntax assembly
   __asm {
      mov eax, function_id
      mov ecx, subfunction_id
      cpuid;
      mov esi, cpuInfo
      mov[esi], eax
      mov[esi + 4], ebx
      mov[esi + 8], ecx
      mov[esi + 12], edx
   }
#endif
}

inline static uint64_t xgetbv(int xcr) {
   // from: https://github.com/vectorclass/version2/blob/9d324e13457cf67b44be04f49a4a0036bb188a89/instrset_detect.cpp#L22C1-L48C1
#if (defined (_MSC_FULL_VER) && _MSC_FULL_VER >= 160040000) || (defined (__INTEL_COMPILER) && __INTEL_COMPILER >= 1200)
   return uint64_t(_xgetbv(xcr));
#elif defined(__GNUC__) ||  defined (__clang__)
   uint32_t a, d;
   __asm("xgetbv" : "=a"(a), "=d"(d) : "c"(xcr) : );
   return a | (uint64_t(d) << 32);
#else
   // Some other platform. Try masm/intel syntax assembly
   uint32_t a, d;
   __asm {
      mov ecx, xcr
      _emit 0x0f
      _emit 0x01
      _emit 0xd0; // xgetbv
      mov a, eax
      mov d, edx
   }
   return a | (uint64_t(d) << 32);
#endif
}

static int DetectInstructionset() {
   // from: https://github.com/vectorclass/version2/blob/9d324e13457cf67b44be04f49a4a0036bb188a89/instrset_detect.cpp#L63
   int instructionSet = 0;
   int abcd[4] = { 0, 0, 0, 0 };
   cpuid(abcd, 0);
   if(abcd[0] == 0) return instructionSet;
   cpuid(abcd, 1);
   if((abcd[3] & (1 << 0)) == 0) return instructionSet;
   if((abcd[3] & (1 << 23)) == 0) return instructionSet;
   if((abcd[3] & (1 << 15)) == 0) return instructionSet;
   if((abcd[3] & (1 << 24)) == 0) return instructionSet;
   if((abcd[3] & (1 << 25)) == 0) return instructionSet;
   instructionSet = 1;
   if((abcd[3] & (1 << 26)) == 0) return instructionSet;
   instructionSet = 2;
   if((abcd[2] & (1 << 0)) == 0) return instructionSet;
   instructionSet = 3;
   if((abcd[2] & (1 << 9)) == 0) return instructionSet;
   instructionSet = 4;
   if((abcd[2] & (1 << 19)) == 0) return instructionSet;
   instructionSet = 5;
   if((abcd[2] & (1 << 23)) == 0) return instructionSet;
   if((abcd[2] & (1 << 20)) == 0) return instructionSet;
   instructionSet = 6;
   if((abcd[2] & (1 << 27)) == 0) return instructionSet;
   if((xgetbv(0) & 6) != 6)       return instructionSet;
   if((abcd[2] & (1 << 28)) == 0) return instructionSet;
   instructionSet = 7;
   cpuid(abcd, 7);
   if((abcd[1] & (1 << 5)) == 0) return instructionSet;
   instructionSet = 8;
   if((abcd[1] & (1 << 16)) == 0) return instructionSet;
   cpuid(abcd, 0xD);
   if((abcd[0] & 0x60) != 0x60)   return instructionSet;
   instructionSet = 9;
   cpuid(abcd, 7);
   if((abcd[1] & (1 << 31)) == 0) return instructionSet;
   if((abcd[1] & 0x40020000) != 0x40020000) return instructionSet;
   instructionSet = 10;
   return instructionSet;
}

static bool IsFMA3() {
   // only call this if 7 <= DetectInstructionset(), which stands for AVX
   // Since we limit ourselves to AVX2 and above, this might be consdidered superfluous since all processors released 
   // with AVX2 also support FMA3, but call it anyways for extra insurance since it's the correct thing to do.
   int abcd[4];
   cpuid(abcd, 1);
   return 0 != (abcd[2] & (1 << 12));
}

#endif // INTEL_SIMD

extern ErrorEbm GetObjective(
   const Config * const pConfig,
   const char * sObjective,
   const AccelerationFlags acceleration,
   ObjectiveWrapper * const pCpuObjectiveWrapperOut,
   ObjectiveWrapper * const pSIMDObjectiveWrapperOut
) noexcept {
   EBM_ASSERT(nullptr != pConfig);
   EBM_ASSERT(nullptr != pCpuObjectiveWrapperOut);
   EBM_ASSERT(nullptr == pCpuObjectiveWrapperOut->m_pObjective);
   EBM_ASSERT(nullptr == pCpuObjectiveWrapperOut->m_pFunctionPointersCpp);

   EBM_ASSERT(nullptr != pSIMDObjectiveWrapperOut || acceleration == AccelerationFlags_NONE);
   EBM_ASSERT(nullptr == pSIMDObjectiveWrapperOut || nullptr == pSIMDObjectiveWrapperOut->m_pObjective);
   EBM_ASSERT(nullptr == pSIMDObjectiveWrapperOut || nullptr == pSIMDObjectiveWrapperOut->m_pFunctionPointersCpp);

   if(nullptr == sObjective) {
      return Error_ObjectiveUnknown;
   }
   sObjective = SkipWhitespace(sObjective);
   if('\0' == *sObjective) {
      return Error_ObjectiveUnknown;
   }

   const char * const sObjectiveEnd = sObjective + strlen(sObjective);

   ErrorEbm error;

   error = CreateObjective_Cpu_64(pConfig, sObjective, sObjectiveEnd, pCpuObjectiveWrapperOut);
   if(Error_None != error) {
      return error;
   }

   const AccelerationFlags zones = static_cast<AccelerationFlags>(pCpuObjectiveWrapperOut->m_zones & acceleration);

   // when compiled with only CPU these variables are not used
   UNUSED(zones);
   UNUSED(pSIMDObjectiveWrapperOut);

   do {
#ifdef BRIDGE_AVX512F_32
      if(0 != (AccelerationFlags_AVX512F & zones)) {
         LOG_0(Trace_Info, "INFO GetObjective checking for AVX512F compatibility");
         EBM_ASSERT(nullptr != pSIMDObjectiveWrapperOut);
         if(9 <= DetectInstructionset()) {
            LOG_0(Trace_Info, "INFO GetObjective creating AVX512F SIMD Objective");
            error = CreateObjective_Avx512f_32(pConfig, sObjective, sObjectiveEnd, pSIMDObjectiveWrapperOut);
            if(Error_None != error) {
               return error;
            }
            break;
         }
      }
#endif // BRIDGE_AVX512F_32

#ifdef BRIDGE_AVX2_32
      if(0 != (AccelerationFlags_AVX2 & zones)) {
         LOG_0(Trace_Info, "INFO GetObjective checking for AVX2 compatibility");
         EBM_ASSERT(nullptr != pSIMDObjectiveWrapperOut);
         if(8 <= DetectInstructionset() && IsFMA3()) {
            LOG_0(Trace_Info, "INFO GetObjective creating AVX2 SIMD Objective");
            error = CreateObjective_Avx2_32(pConfig, sObjective, sObjectiveEnd, pSIMDObjectiveWrapperOut);
            if(Error_None != error) {
               return error;
            }
            break;
         }
      }
#endif // BRIDGE_AVX2_32

      LOG_0(Trace_Info, "INFO GetObjective no SIMD option found");
   } while(false);

   return Error_None;
}

#ifdef NEVER
// TODO: eventually enable metrics
INLINE_RELEASE_UNTEMPLATED static ErrorEbm GetMetrics(
   const Config * const pConfig,
   const char * sMetric
//   MetricWrapper * const aMetricWrapperOut
) noexcept {
   EBM_ASSERT(nullptr != pConfig);
   //EBM_ASSERT(nullptr != pMetricWrapperOut);
   //aMetricWrapperOut->m_pMetric = nullptr;
   //aMetricWrapperOut->m_pFunctionPointersCpp = nullptr;

   if(nullptr == sMetric) {
      // it's legal to have no metrics
      return Error_None;
   }
   while(true) {
      sMetric = SkipWhitespace(sMetric);
      const char * sMetricEnd = strchr(sMetric, k_registrationSeparator);
      if(nullptr == sMetricEnd) {
         // find the null terminator then
         sMetricEnd = sMetric + strlen(sMetric);
      }
      if(sMetricEnd != sMetric) {
         // we allow empty registrations like ",,,something_legal,,,  something_else  , " since the intent is clear
         ErrorEbm error;

         error = CreateMetric_Cpu_64(pConfig, sMetric, sMetricEnd);
         if(Error_None != error) {
            return error;
         }

         // TODO: for now let's return after we find the first metric, but in the future we'll want to return
         //       some kind of list of them
         return error;
      }
      if('\0' == *sMetricEnd) {
         return Error_None;
      }
      EBM_ASSERT(k_registrationSeparator == *sMetricEnd);

      sMetric = sMetricEnd + 1;
   }
}
#endif // NEVER

} // DEFINED_ZONE_NAME
