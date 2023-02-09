// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// This WrapFunc.cpp file is ONLY included in builds where it is needed
//
// We want our library to be loadable into as many Linux distributions as possible.  It seems that on Linux
// the C runtime libraries (libm.so and libc.so) can't be statically linked, or at least it's not advisable
// we therefore want to ensure that we only use older versions of the functions in GLIBC
//
// If we run the command: objdump -T lib_ebm_native_linux_x64.so | grep GLIBC_
// we see the following entry:
// 0000000000000000      DF *UND*  0000000000000000  GLIBC_2.14  memcpy
// We can also see this information in the "Version References" section of the disassembly of lib_ebm_native*.s that 
// build.sh optionally generates. We put that file into the artifacts of the Azure build pipeline.
// 
// The version 2.14 for memcpy has been a problem for some of our CentOS users.
// Unfortunately, this function is called by the libgcc OR libstdc++ libraries which we've statically linked in.
// We can demand an older version of this function here using the ".symver" below, and then we can make the compiler
// substitute our wrapper function below for other calls throughout our library, which then calls the older function.
// The wrapper function also needs to be included in the gcc/g++ command line in build.sh using "-Wl,--wrap=memcpy"
// 
// It is possible to get a list of function versions available to link to with the following command:
// objdump -T /lib/x86_64-linux-gnu/libc.so.6    OR   objdump -T /lib/x86_64-linux-gnu/libm.so.6
//
// More info:
// https://stackoverflow.com/questions/8823267/linking-against-older-symbol-version-in-a-so-file
// https://stackoverflow.com/questions/36461555/is-it-possible-to-statically-link-libstdc-and-wrap-memcpy
//

#include <string.h>
#include <math.h>

#if defined(__x86_64__)
// 64 bit x64

__asm__(".symver memcpy, memcpy@GLIBC_2.2.5");
__asm__(".symver exp, exp@GLIBC_2.2.5");
__asm__(".symver log, log@GLIBC_2.2.5");
__asm__(".symver log2, log2@GLIBC_2.2.5");
__asm__(".symver pow, pow@GLIBC_2.2.5");

extern "C" {
   void * __wrap_memcpy(void * dest, const void * src, size_t n) {
      return memcpy(dest, src, n);
   }
   double __wrap_exp(double x) {
      return exp(x);
   }
   double __wrap_log(double x) {
      return log(x);
   }
   double __wrap_log2(double x) {
      return log2(x);
   }
   double __wrap_pow(double base, double exponent) {
      return pow(base, exponent);
   }
}

#elif defined(__i386__)
// 32 bit x86

// no GLIBC substitutions

#else
#error unrecognized GCC architecture
#endif

