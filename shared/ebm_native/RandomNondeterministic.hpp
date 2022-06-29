// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef RANDOM_NONDETERMINISTIC_HPP
#define RANDOM_NONDETERMINISTIC_HPP

#include <random>

#include "zones.h"
#include "ebm_internal.hpp" // INLINE_ALWAYS

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class RandomNondeterministic final {

   unsigned int m_randomRemainingMax;
   unsigned int m_randomRemaining;

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

public:

   INLINE_ALWAYS RandomNondeterministic() :
      m_randomRemainingMax(0),
      m_randomRemaining(0) {
   }

   INLINE_ALWAYS unsigned int Next() {
      return m_generator();
   }

   INLINE_ALWAYS unsigned int Next(const unsigned int maxInclusive) {
      static_assert(0 == std::random_device::min(), "std::random_device::min() must be zero");
      static_assert(std::numeric_limits<decltype(maxInclusive)>::max() == std::random_device::max(), 
         "std::random_device::max() must be the maximum of the maxInclusive type");

      if(std::random_device::max() == maxInclusive) {
         return m_generator();
      }
      const unsigned int maxExclusive = maxInclusive + 1;

      unsigned int randomRemainingMax = m_randomRemainingMax;
      unsigned int randomRemaining = m_randomRemaining;
      EBM_ASSERT(randomRemaining <= randomRemainingMax);
      while(true) {
         if(maxInclusive <= randomRemainingMax) {
            randomRemainingMax = (randomRemainingMax - maxInclusive) / maxExclusive;
            const unsigned int legalMax = randomRemainingMax * maxExclusive + maxInclusive;
            if(randomRemaining <= legalMax) {
               break;
            }
         }

         randomRemaining = m_generator();
         randomRemainingMax = std::random_device::max();
      }
      const unsigned int ret = randomRemaining % maxExclusive;
      EBM_ASSERT(randomRemaining / maxExclusive <= randomRemainingMax);
      m_randomRemainingMax = randomRemainingMax;
      m_randomRemaining = randomRemaining / maxExclusive;

      return ret;
   }
};

} // DEFINED_ZONE_NAME

#endif // RANDOM_NONDETERMINISTIC_HPP
