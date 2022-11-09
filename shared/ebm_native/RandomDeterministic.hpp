// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef RANDOM_DETERMINISTIC_HPP
#define RANDOM_DETERMINISTIC_HPP

#include <inttypes.h> // uint64_t, uint_fast64_t, uint32_t, uint_fast32_t
#include <stddef.h> // size_t, ptrdiff_t
#include <type_traits>

#include "ebm_native.h" // SeedEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // INLINE_ALWAYS
#include "zones.h"

#include "common_cpp.hpp" // CountBitsRequiredPositiveMax

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class RandomDeterministic final {
   // If the RandomDeterministic object is stored inside a class/struct, and used inside a hotspot loop, to get the best 
   // performance copy this structure to the stack before using it, and then copy it back to the struct/class 
   // after looping on it.  Copying it to the stack allows the internal state to be kept inside CPU
   // registers since no aliased pointers can point to the stack version unless we explicitly create a pointer to it.
   // Use references instead of pointers if we want to pass the RandomDeterministic to called functions to avoid the 
   // aliasing issues. In contrast, if we used main memory to hold our state, the compiler can't know if any pointers 
   // used are pointing to the main memory version of our internal state for sure if there are char or void * pointers
   // involved, or if aliasing assumptions are disallowed.

   // use the RNG method described in https://arxiv.org/abs/1704.00358
   // there is also the related counter based version that might be useful: https://arxiv.org/abs/2004.06278v2
   // but this Middle Square Weyl algorithm looks like it would be faster than the counter based version if all the 
   // state were put into registers inside of a loop.  I don't know how the authors found the reverse.

   // NOTE: the Middle Square Weyl Sequence paper makes claims of it being a cryptographically secure RNG.
   // We don't care about this from a general sense and all we want is something that's at least
   // pseudo random for tie-breaking and bagging and it having passed BigCrush probably means it's good enough
   // in this respect.

   // NOTE: this random number generator has a mathematically proven cycle of 2^64 numbers, after which it will
   // exactly repeat the same sequence. On a 3GHz machine, if we generated a random number every clock cycle 
   // it would take 195 years to do this full cycle, so in practice this isn't a problem for us.  Even with multiple 
   // threads it isn't a problem since we'll give each of those threads their own seed, which means they each have 
   // their own 2^64 cycles that are different from eachother.  We only get cycling problems if the computation
   // happens sequentially.  In some far future when computers are much faster, we might add some detection code 
   // that detects a cycle and increments to the next valid internal seed.  Then we'd have 2^118 unique random 
   // 64 bit numbers, according to the paper.

   uint64_t m_state1;
   uint64_t m_state2;
   uint64_t m_stateSeedConst;

   INLINE_ALWAYS uint_fast32_t Rand32() {
      // if this gets properly optimized, it gets converted into 4 machine instructions: imulq, iaddq, iaddq, and rorq
      m_state1 *= m_state1;
      m_state2 += m_stateSeedConst;
      m_state1 += m_state2;
      m_state1 = (m_state1 >> 32) | (m_state1 << 32); // this should get optimized to a single rorq instruction
      // chop it to 32 bits if it was given to us with more bits
      return static_cast<uint_fast32_t>(static_cast<uint32_t>(m_state1));
   }

   static uint_fast64_t GetOneTimePadConversion(uint_fast64_t seed);

public:

   RandomDeterministic() = default; // preserve our POD status
   ~RandomDeterministic() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   void Initialize(const uint64_t seed);

   INLINE_ALWAYS void Initialize(const SeedEbm seed) {
      // the C++ standard guarantees that the unsigned result of this 
      // conversion is the two's complement of the seed if seed is negative
      Initialize(static_cast<uint64_t>(static_cast<USeedEbm>(seed)));
   }

   INLINE_ALWAYS void Initialize(const RandomDeterministic & other) {
      m_state1 = other.m_state1;
      m_state2 = other.m_state2;
      m_stateSeedConst = other.m_stateSeedConst;
   }

   template<typename T>
   INLINE_ALWAYS typename std::enable_if<std::is_unsigned<T>::value && std::numeric_limits<uint32_t>::max() < std::numeric_limits<T>::max(), T>::type NextFast(const T maxPlusOne) {
      EBM_ASSERT(T { 1 } <= maxPlusOne);

      // let's say that we are given maxPlusOneConverted == 7.  In that case we take our 32 bit
      // number and take modulo 7, giving us random numbers 0, 1, 2, 3, 4, 5, and 6.
      // but there's a problem if rand is 4294967295 (the maximum 32 bit number) because 
      // 4294967295 % 7 == 3, which means there are more opportunities to return 0, 1, 2, 3 than there are
      // to return 4, 5, or 6.  To avoid this, we throw our any random numbers equal or above
      // 4294967292.  Calculating the exact number 4294967292 is expensive though, so we instead find the
      // highest number that the rounded down multiple can be before we reject it.  In the case of 7,
      // any rounded down muliple that is higher than (UINT32_MAX + 1) - 7 will lead to imbalanced
      // random number generation, since the next higher up muliple of 7 will overflow.  (UINT32_MAX + 1)
      // overlfows to 0, but since underflow and overlfow of unsigned numbers is legal in C++, we can use
      // this fact, and the fact that maxPlusOneConverted needs to be at least 1 to generate the
      // low bound that we need.

      if(T { std::numeric_limits<uint32_t>::max() } < maxPlusOne) {
         const T max = maxPlusOne - T { 1 };
         T rand;
         T randMult;
         do {
            T maxContent = T { std::numeric_limits<uint32_t>::max() };
            rand = static_cast<T>(Rand32());
            while(maxContent < max) {
               maxContent = (maxContent << 32) | T { std::numeric_limits<uint32_t>::max() };
               rand = (rand << 32) | static_cast<T>(Rand32());
            }
            const T randDivided = rand / maxPlusOne;
            randMult = randDivided * maxPlusOne;
         } while(UNLIKELY(T { 0 } - maxPlusOne < randMult));
         EBM_ASSERT(randMult <= rand);
         return rand - randMult;
      } else {
         const uint32_t maxPlusOneConverted = static_cast<uint32_t>(maxPlusOne);
         uint32_t rand;
         uint32_t randMult;
         do {
            rand = static_cast<uint32_t>(Rand32());
            const uint32_t randDivided = rand / maxPlusOneConverted;
            randMult = randDivided * maxPlusOneConverted;
         } while(UNLIKELY(uint32_t { 0 } - maxPlusOneConverted < randMult));
         EBM_ASSERT(randMult <= rand);
         return static_cast<T>(rand - randMult);
      }
   }

   template<typename T>
   INLINE_ALWAYS typename std::enable_if<std::is_unsigned<T>::value && std::numeric_limits<T>::max() <= std::numeric_limits<uint32_t>::max(), T>::type NextFast(const T maxPlusOne) {
      EBM_ASSERT(T { 1 } <= maxPlusOne);

      // let's say that we are given maxPlusOneConverted == 7.  In that case we take our 32 bit
      // number and take modulo 7, giving us random numbers 0, 1, 2, 3, 4, 5, and 6.
      // but there's a problem if rand is 4294967295 (the maximum 32 bit number) because 
      // 4294967295 % 7 == 3, which means there are more opportunities to return 0, 1, 2, 3 than there are
      // to return 4, 5, or 6.  To avoid this, we throw our any random numbers equal or above
      // 4294967292.  Calculating the exact number 4294967292 is expensive though, so we instead find the
      // highest number that the rounded down multiple can be before we reject it.  In the case of 7,
      // any rounded down muliple that is higher than (UINT32_MAX + 1) - 7 will lead to imbalanced
      // random number generation, since the next higher up muliple of 7 will overflow.  (UINT32_MAX + 1)
      // overlfows to 0, but since underflow and overlfow of unsigned numbers is legal in C++, we can use
      // this fact, and the fact that maxPlusOneConverted needs to be at least 1 to generate the
      // low bound that we need.

      const uint32_t maxPlusOneConverted = static_cast<uint32_t>(maxPlusOne);
      uint32_t rand;
      uint32_t randMult;
      do {
         rand = static_cast<uint32_t>(Rand32());
         const uint32_t randDivided = rand / maxPlusOneConverted;
         randMult = randDivided * maxPlusOneConverted;
      } while(UNLIKELY(uint32_t { 0 } - maxPlusOneConverted < randMult));
      EBM_ASSERT(randMult <= rand);
      return static_cast<T>(rand - randMult);
   }

   template<typename T>
   INLINE_ALWAYS typename std::enable_if<std::is_unsigned<T>::value && std::numeric_limits<uint32_t>::max() < std::numeric_limits<T>::max(), T>::type Next(const T max) {
      if(std::numeric_limits<T>::max() == max) {
         static constexpr size_t k_bitsT = CountBitsRequiredPositiveMax<T>();
         static_assert(MaxFromCountBits<T>(k_bitsT) == std::numeric_limits<T>::max(), "T max must be all 1s");
         size_t count = (k_bitsT + 31) / 32 - 1;
         EBM_ASSERT(1 <= count);
         T rand = static_cast<T>(Rand32());
         do {
            rand = (rand << 32) | static_cast<T>(Rand32());
            --count;
         } while(0 != count);
         return rand;
      }
      return NextFast(max + T { 1 });
   }

   template<typename T>
   INLINE_ALWAYS typename std::enable_if<std::is_unsigned<T>::value && std::numeric_limits<T>::max() <= std::numeric_limits<uint32_t>::max(), T>::type Next(const T max) {
      if(std::numeric_limits<T>::max() == max) {
         return static_cast<T>(Rand32());
      }
      return NextFast(max + T { 1 });
   }

   template<typename T>
   INLINE_ALWAYS typename std::enable_if<std::is_unsigned<T>::value && !std::is_same<T, bool>::value, T>::type Next() {
      return Next(std::numeric_limits<T>::max()); // inlining will pick the best direct branch and eliminate the rest
   }

   template<typename T>
   INLINE_ALWAYS typename std::enable_if<std::is_signed<T>::value, T>::type Next() {
      static_assert(is_twos_complement<T>::value, "we only support twos complement negative numbers");

      return TwosComplementConvert(Next<typename std::make_unsigned<T>::type>());
   }

   template<typename T>
   INLINE_ALWAYS typename std::enable_if<std::is_same<T, bool>::value, T>::type Next() {
      return uint_fast32_t { 0 } != (uint_fast32_t { 1 } & Rand32());
   }
};
static_assert(std::is_standard_layout<RandomDeterministic>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<RandomDeterministic>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<RandomDeterministic>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // RANDOM_DETERMINISTIC_HPP