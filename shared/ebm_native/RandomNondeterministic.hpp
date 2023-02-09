// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef RANDOM_NONDETERMINISTIC_HPP
#define RANDOM_NONDETERMINISTIC_HPP

#include <random>

#include "ebm_native.h" // SeedEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // INLINE_ALWAYS
#include "zones.h"

#include "common_cpp.hpp" // CountBitsRequiredPositiveMax

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<typename T>
class RandomNondeterministic final {
   static_assert(!std::is_signed<T>::value, "T must be an unsigned type");
   static_assert(0 == std::numeric_limits<T>::min(), "T must have a min value of 0");

   T m_randomRemainingMax;
   T m_randomRemaining;

   // The C++ standard does not give us a guarantee that random_device is non-deterministic in all implementations
   // We need to check on each platform whether std::random_device comes from a non-deterministic high quality source.
   // The platforms we run on, and have checked are:
   //   Windows => Microsoft documents "the values produced are non-deterministic and cryptographically secure"
   //     https://docs.microsoft.com/en-us/cpp/standard-library/random-device-class?view=msvc-170
   //   GCC, Linux, Intel => Uses Intel RDRAND instruction
   //     From source code & https://www.pcg-random.org/posts/cpps-random_device.html
   //   Clang, Mac => Uses /dev/urandom (newer versions use successive Mac OS APIs that provide the equivalent)
   //     From source code & https://www.pcg-random.org/posts/cpps-random_device.html
   //
   // MinGW used to have a bad implementation, but that was fixed in 2019.  We do not use MinGW currently.
   //   https://sourceforge.net/p/mingw-w64/bugs/338/
   //   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85494
   // 
   // There is no reason to check random_device.entropy since it is unreliable across implementations
   //
   std::random_device m_generator;

   INLINE_ALWAYS T Shift(const T val, const size_t shift) {
      // putting this shift in a function avoids a compiler warning
      return val << shift;
   }

public:

   INLINE_ALWAYS RandomNondeterministic() :
      m_randomRemainingMax(0),
      m_randomRemaining(0) {
   }

   INLINE_ALWAYS T Next() {
      static constexpr size_t k_bitsT = CountBitsRequiredPositiveMax<T>();
      static constexpr size_t k_bitsRandom = CountBitsRequiredPositiveMax<unsigned int>();

      static_assert(MaxFromCountBits<T>(k_bitsT) == std::numeric_limits<T>::max(), "T max must be all 1s");
      static_assert(MaxFromCountBits<unsigned int>(k_bitsRandom) == std::numeric_limits<unsigned int>::max(),
         "unsigned int max must be all 1s");

      static_assert(0 == std::random_device::min(), "std::random_device::min() must be zero");
      static_assert(MaxFromCountBits<unsigned int>(k_bitsRandom) == std::random_device::max(),
         "std::random_device::max() must be the max for unsigned int");

      T ret = static_cast<T>(m_generator());
      size_t count = (k_bitsT + k_bitsRandom - 1) / k_bitsRandom - 1;
      while(0 != count) {
         // if k_bitsT == k_bitsRandom then the compiler should optimize this out
         ret = Shift(ret, k_bitsRandom) | static_cast<T>(m_generator());
         --count;
      }
      return ret;
   }

   INLINE_ALWAYS T Next(const T max) {
      if(std::numeric_limits<T>::max() == max) {
         return Next();
      }
      const T maxPlusOne = max + T { 1 };

      T randomRemainingMax = m_randomRemainingMax;
      T randomRemaining = m_randomRemaining;
      EBM_ASSERT(randomRemaining <= randomRemainingMax);
      while(true) {
         if(max <= randomRemainingMax) {
            randomRemainingMax = (randomRemainingMax - max) / maxPlusOne;
            const T legalMax = randomRemainingMax * maxPlusOne + max;
            if(randomRemaining <= legalMax) {
               break;
            }
         }

         randomRemaining = Next();
         randomRemainingMax = std::numeric_limits<T>::max();
      }
      const T ret = randomRemaining % maxPlusOne;
      EBM_ASSERT(randomRemaining / maxPlusOne <= randomRemainingMax);
      m_randomRemainingMax = randomRemainingMax;
      m_randomRemaining = randomRemaining / maxPlusOne;

      return ret;
   }

   INLINE_ALWAYS T NextFast(const T maxPlusOne) {
      EBM_ASSERT(T { 1 } <= maxPlusOne);
      return Next(maxPlusOne - T { 1 });
   }

   INLINE_ALWAYS typename std::make_signed<T>::type NextNegative() {
      return TwosComplementConvert(Next());
   }
};

} // DEFINED_ZONE_NAME

#endif // RANDOM_NONDETERMINISTIC_HPP
