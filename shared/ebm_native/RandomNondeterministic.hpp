// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef RANDOM_NONDETERMINISTIC_HPP
#define RANDOM_NONDETERMINISTIC_HPP

#include <random>

#include "zones.h"
#include "common_cpp.hpp"
#include "ebm_internal.hpp" // INLINE_ALWAYS

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

   INLINE_ALWAYS SeedEbm NextSeed() {
      // TODO: I could probably generalize this to make any negative number type

      static_assert(std::numeric_limits<SeedEbm>::lowest() < SeedEbm { 0 }, "SeedEbm must be signed");
      static_assert(std::is_same<USeedEbm, T>::value, "T must be USeedEbm");

      // this is meant to result in a positive value that is of the negation of 
      // std::numeric_limits<SeedEbm>::lowest(), so -std::numeric_limits<SeedEbm>::lowest().
      // but the pitfall is that for numbers expressed in twos complement, there is one more
      // negative number than there are positive numbers, so we subtract one (adding to a negated number), then add 
      // one to keep the numbers in bounds.  If the compiler is using some non-twos complement
      // representation, then we'll get a compile error in the static_asserts below or in the initialization
      // of USeedEbm below
      constexpr USeedEbm negativeOfLowest =
         USeedEbm { -(std::numeric_limits<SeedEbm>::lowest() + SeedEbm { 1 }) } + USeedEbm { 1 };

      static_assert(USeedEbm { std::numeric_limits<SeedEbm>::max() } == negativeOfLowest - USeedEbm { 1 }, 
         "max must == lowestInUnsigned - 1");

      const USeedEbm randomNumber = Next();
      // adding negativeOfLowest and then adding lowest are a no-op as far as affecting the value of randomNumber
      // but since adding randomNumber + negativeOfLowest (two unsigned values) is legal in C++, and since we'll
      // always end up with a value that can be expressed as an SeedEbm after that addition we don't have
      // and undefined behavior here.  The compiler should be smart enough to eliminate this operation.
      const SeedEbm ret = randomNumber < negativeOfLowest ? static_cast<SeedEbm>(randomNumber) :
         static_cast<SeedEbm>(randomNumber + negativeOfLowest) + std::numeric_limits<SeedEbm>::lowest();

      return ret;
   }
};

} // DEFINED_ZONE_NAME

#endif // RANDOM_NONDETERMINISTIC_HPP
