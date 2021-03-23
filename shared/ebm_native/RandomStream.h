// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef RANDOM_STREAM_H
#define RANDOM_STREAM_H

#include <inttypes.h> // uint32_t, uint_fast64_t
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // IntEbmType
#include "EbmInternal.h" // INLINE_ALWAYS
#include "logging.h"

// We expose our random number generator, so to ensure that we get
// different random number streams from what our caller will get, we want to mix in these values
// in case our caller happens to mistakenly forget to provide a mix in value
constexpr uint64_t k_quantileRandomizationMix = uint64_t { 5744215463699302938u };
constexpr uint64_t k_boosterRandomizationMix = uint64_t { 9397611943394063143u };
constexpr uint64_t k_samplingWithoutReplacementRandomizationMix = uint64_t { 10077040353197036781u };

class RandomStream final {
   // If the RandomStream object is stored inside a class/struct, and used inside a hotspot loop, to get the best 
   // performance copy this structure to the stack before using it, and then copy it back to the struct/class 
   // after looping on it.  Copying it to the stack allows the internal state to be kept inside CPU
   // registers since no aliased pointers can point to the stack version unless we explicitly create a pointer to it.
   // Use references instead of pointers if we want to pass the RandomStream to called functions to avoid the 
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

   // If we had memcopied RandomStream cross machine we would want these to be uint64_t instead of uint_fast64_t
   // but we only copy seeds cross machine, so we can leave them as uint_fast64_t

   uint_fast64_t m_state1;
   uint_fast64_t m_state2;
   uint_fast64_t m_stateSeedConst;

   static uint_fast64_t GetOneTimePadConversion(uint_fast64_t seed);

   // TODO: it might be possible to make the 64 bit version a bit faster, either by having 2 independent streams
   //       so that the multiplication can be done in parallel (although this would consume more registers), or
   //       if we combine the two 32 bit functions which might eliminate some of the shifting
   INLINE_ALWAYS uint_fast32_t Rand32() {
      // if this gets properly optimized, it gets converted into 4 machine instructions: imulq, iaddq, iaddq, and rorq
      m_state1 *= m_state1;
      m_state2 += m_stateSeedConst;
      m_state1 += m_state2;
      const uint64_t state1Shiftable = static_cast<uint64_t>(m_state1);
      // this should get optimized to a single rorq assembly instruction
      const uint64_t result = (state1Shiftable >> 32) | (state1Shiftable << 32);
      m_state1 = static_cast<uint_fast64_t>(result);
      // chop it to 32 bits if it was given to us with more bits
      return static_cast<uint_fast32_t>(static_cast<uint32_t>(result));
   }

   INLINE_ALWAYS uint_fast64_t Rand64() {
      const uint_fast64_t top = static_cast<uint_fast64_t>(Rand32());
      const uint_fast64_t bottom = static_cast<uint_fast64_t>(Rand32());
      return (top << 32) | bottom;
   }

   void Initialize(const uint64_t seed);

public:

   RandomStream() = default; // preserve our POD status
   ~RandomStream() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS void InitializeSigned(const SeedEbmType seed, const SeedEbmType stageRandomizationMix) {
      // the C++ standard guarantees that the unsigned result of this 
      // conversion is 2^64 + seed if seed is negative
      Initialize(static_cast<uint64_t>(seed) ^ static_cast<uint64_t>(stageRandomizationMix));
   }

   INLINE_ALWAYS void InitializeUnsigned(const SeedEbmType seed, const uint64_t stageRandomizationMix) {
      // the C++ standard guarantees that the unsigned result of this 
      // conversion is 2^64 + seed if seed is negative
      Initialize(static_cast<uint64_t>(seed) ^ stageRandomizationMix);
   }

   INLINE_ALWAYS void Initialize(const RandomStream & other) {
      m_state1 = other.m_state1;
      m_state2 = other.m_state2;
      m_stateSeedConst = other.m_stateSeedConst;
   }

   INLINE_ALWAYS SeedEbmType NextSeed() {
      static_assert(std::numeric_limits<SeedEbmType>::lowest() < SeedEbmType { 0 },
         "SeedEbmType must be signed");

      // this is meant to result in a positive value that is of the negation of 
      // std::numeric_limits<SeedEbmType>::lowest(), so -std::numeric_limits<SeedEbmType>::lowest().
      // but the pitfall is that for numbers expressed in twos complement, there is one more
      // negative number than there are positive numbers, so we subtract one (adding to a negated number), then add 
      // one to keep the numbers in bounds.  If the compiler is using some non-twos complement
      // representation, then we'll get a compile error in the static_asserts below or in the initialization
      // of uint32_t below
      constexpr uint32_t negativeOfLowest = 
         uint32_t { -(std::numeric_limits<SeedEbmType>::lowest() + SeedEbmType { 1 }) } + uint32_t { 1 };

      static_assert(uint32_t { std::numeric_limits<SeedEbmType>::max() } ==
         negativeOfLowest - uint32_t { 1 }, "max must == lowestInUnsigned - 1");

      const uint32_t randomNumber = static_cast<uint32_t>(Rand32());
      // adding negativeOfLowest and then adding lowest are a no-op as far as affecting the value of randomNumber
      // but since adding randomNumber + negativeOfLowest (two unsigned values) is legal in C++, and since we'll
      // always end up with a value that can be expressed as an SeedEbmType after that addition we don't have
      // and undefined behavior here.  The compiler should be smart enough to eliminate this operation.
      const SeedEbmType ret = randomNumber < negativeOfLowest ? static_cast<SeedEbmType>(randomNumber) :
         static_cast<SeedEbmType>(randomNumber + negativeOfLowest) + std::numeric_limits<SeedEbmType>::lowest();

      return ret;
   }

   INLINE_ALWAYS bool Next() {
      return uint_fast32_t { 0 } != (uint_fast32_t { 1 } & Rand32());
   }

   INLINE_ALWAYS size_t Next(const size_t maxValueExclusive) {
      EBM_ASSERT(size_t { 1 } <= maxValueExclusive);

      // let's say that we are given maxValueExclusiveConverted == 7.  In that case we take our 32 bit
      // number and take modulo 7, giving us random numbers 0, 1, 2, 3, 4, 5, and 6.
      // but there's a problem if rand is 4294967295 (the maximum 32 bit number) because 
      // 4294967295 % 7 == 3, which means there are more opportunities to return 0, 1, 2, 3 than there are
      // to return 4, 5, or 6.  To avoid this, we throw our any random numbers equal or above
      // 4294967292.  Calculating the exact number 4294967292 is expensive though, so we instead find the
      // highest number that the rounded down multiple can be before we reject it.  In the case of 7,
      // any rounded down muliple that is higher than (UINT32_MAX + 1) - 7 will lead to imbalanced
      // random number generation, since the next higher up muliple of 7 will overflow.  (UINT32_MAX + 1)
      // overlfows to 0, but since underflow and overlfow of unsigned numbers is legal in C++, we can use
      // this fact, and the fact that maxValueExclusiveConverted needs to be at least 1 to generate the
      // low bound that we need.

#if !defined(UINT32_MAX) || !defined(SIZE_MAX) || UINT32_MAX < SIZE_MAX
      static_assert(std::numeric_limits<size_t>::max() <= std::numeric_limits<uint64_t>::max(),
         "we must be able to at least generate a real random size_t value");

      if(size_t { std::numeric_limits<uint32_t>::max() } < maxValueExclusive) {
         const uint64_t maxValueExclusiveConverted = static_cast<uint64_t>(maxValueExclusive);
         uint64_t rand;
         uint64_t randMult;
         do {
            rand = static_cast<uint64_t>(Rand64());
            const uint64_t randDivided = rand / maxValueExclusiveConverted;
            randMult = randDivided * maxValueExclusiveConverted;
         } while(UNLIKELY(uint64_t { 0 } - maxValueExclusiveConverted < randMult));
         EBM_ASSERT(randMult <= rand);
         return rand - randMult;
      } else 
#endif // UINT32_MAX < SIZE_MAX
      {
         const uint32_t maxValueExclusiveConverted = static_cast<uint32_t>(maxValueExclusive);
         uint32_t rand;
         uint32_t randMult;
         do {
            rand = static_cast<uint32_t>(Rand32());
            const uint32_t randDivided = rand / maxValueExclusiveConverted;
            randMult = randDivided * maxValueExclusiveConverted;
         } while(UNLIKELY(uint32_t { 0 } - maxValueExclusiveConverted < randMult));
         EBM_ASSERT(randMult <= rand);
         return rand - randMult;
      }
   }
};
static_assert(std::is_standard_layout<RandomStream>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<RandomStream>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<RandomStream>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

#endif // RANDOM_STREAM_H