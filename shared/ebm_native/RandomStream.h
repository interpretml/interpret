// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef RANDOM_STREAM_H
#define RANDOM_STREAM_H

#include <inttypes.h> // uint32_t, uint_fast64_t
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // IntEbmType
#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h"

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
   // their own 2^64 cycle that are different from eachother.  We only get cycling problems if the computation
   // happens sequentially.  In some far future when computers are much faster, we might add some detection code 
   // that detects a cycle and increments to the next valid internal seed.  Then we'd have 2^118 unique random 
   // 64 bit numbers, according to the paper.

   // If we memcopied RandomStream cross machine we would want these to be uint64_t instead of uint_fast64_t
   // but we only copy seeds cross machine, so we can leave them as uint_fast64_t

   uint_fast64_t m_state1;
   uint_fast64_t m_state2;
   uint_fast64_t m_stateSeedConst;

   uint_fast64_t m_randomRemainingMax;
   uint_fast64_t m_randomRemaining;

   static const uint_fast64_t k_oneTimePadRandomSeed[64];

   uint_fast64_t GetOneTimePadConversion(uint_fast64_t seed);
   void InitializeInternal(const uint64_t seed);

   // TODO: it might be possible to make the 64 bit version a bit faster, either by having 2 independent streams
   //       so that the multiplication can be done in parallel (although this would consume more registers), or
   //       if we combine the two 32 bit functions which might eliminate some of the shifting
   EBM_INLINE uint_fast32_t Rand32() {
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

   EBM_INLINE uint_fast64_t Rand64() {
      const uint_fast64_t top = static_cast<uint_fast64_t>(Rand32());
      const uint_fast64_t bottom = static_cast<uint_fast64_t>(Rand32());
      return (top << 32) | bottom;
   }

public:

   EBM_INLINE void Initialize(const RandomStream & other) {
      m_state1 = other.m_state1;
      m_state2 = other.m_state2;
      m_stateSeedConst = other.m_stateSeedConst;
      m_randomRemainingMax = other.m_randomRemainingMax;
      m_randomRemaining = other.m_randomRemaining;
   }

   EBM_INLINE void Initialize(const uint64_t seed) {
      // this function may seem odd to have given that it doesn't do anything visibly useful over InitializeInternal, 
      // but it does have a purpose.  Our InitializeInternal function is a non-inlined function so that there's 
      // only one copy of the code made within the process for space savings, but that creates a problem in 
      // that the compiler can't reason about the internals of that function since it just knows a pointer 
      // goes to an opaque function.  This function creates a new copy of RandomStream, initializes it, 
      // then steals it's internals.  The compiler can then know that our internals haven't leaked 
      // anywhere through pointers and it is then free to optimize the memory into CPU registers
      // inside loops and then write out the result back to our class afterwards.
      RandomStream internalRandom;
      internalRandom.InitializeInternal(seed);

      m_state1 = internalRandom.m_state1;
      m_state2 = internalRandom.m_state2;
      m_stateSeedConst = internalRandom.m_stateSeedConst;
      m_randomRemainingMax = uint_fast64_t { 0 };
      m_randomRemaining = uint_fast64_t { 0 };
   }

   EBM_INLINE void Initialize(const IntEbmType seed) {
      // the C++ standard guarantees that the unsigned result of this 
      // conversion is 2^64 + seed if seed is negative
      Initialize(static_cast<uint64_t>(seed));
   }

   EBM_INLINE bool NextBit() {
      // TODO : If there is a use case for getting single bits, implement this by directly striping bits off 
      //        randomRemaining and randomRemainingMax
      EBM_ASSERT(false);
      return false;
   }

   EBM_INLINE size_t Next(const size_t maxValueExclusive) {
      static_assert(std::numeric_limits<size_t>::max() <= std::numeric_limits<uint64_t>::max(), 
         "we must be able to at least generate a real random size_t value");

      const uint_fast64_t maxValueExclusiveConverted = static_cast<uint_fast64_t>(maxValueExclusive);

      // let's say our user requests a value of 6 exclusive, so we have 6 possible random values (0,1,2,3,4,5)
      // let's say our randomRemainingMax is 6 or 7.  If randomRemaining is 6 or 7, we don't want to use 6%6 or 7%6 and re-use 0 or 1, since then 0 and 1 
      // are twice as likley as 0-5.  So if randomRemaining is equal to 6 or above, we want to re-generate the random numbers and odd case arrises if 
      // randomRemainingMax == 5, because then there is no illegal value for randomRemaining.  We can optionally regenerate the random number since it's 
      // balanced anyways if randomRemainingMax == 4, then we want to re-generate it since we can't express 5
      //
      // per above, we could instead use randomRemainingMax + 1, but then we'd overflow if randomRemainingMax was std::numeric_limits<uint64_t>::max()
      // when randomRemainingMax + 1 == maxValueExclusiveConverted,or in fact whenever (randomRemainingMax + 1) % maxValueExclusiveConverted == 0 then we 
      // have a special case where no random value is bad since our random value is perfectly divisible by maxValueExclusiveConverted.  We can re-generate 
      // it though, and that's the easiest thing to do in that case, so avoid an additional check by regenerating
      uint_fast64_t randomRemainingTempMax = m_randomRemainingMax;
      uint_fast64_t randomRemainingTemp = m_randomRemaining;
      while(true) {
         // TODO : given that our random number generator is 4 assembly instructions, and given that division
         //        is still very expensive, even for integers, perhaps we should eliminate this code that attepts
         //        to maximize the benefit that we get from random bits.  Perf test generating a new random number
         //        each time.  It might even be possible to eliminate both divisions if we were willing to throw
         //        away some results if we divided by a power of 2 and then multiplied to find where the throw
         //        away point was for a particular random number
         randomRemainingTempMax /= maxValueExclusiveConverted;
         const uint_fast64_t randomRemainingIllegal = randomRemainingTempMax * maxValueExclusiveConverted;
         if(LIKELY(randomRemainingTemp < randomRemainingIllegal)) {
            break;
         }
         randomRemainingTemp = Rand64();
         randomRemainingTempMax = static_cast<uint_fast64_t>(std::numeric_limits<uint64_t>::max());
      }
      // if 0 == randomRemainingMax then we would have stayed in the loop since randomRemaining could not be less than zero
      EBM_ASSERT(0 < randomRemainingTempMax);
      m_randomRemainingMax = randomRemainingTempMax - 1; // the top range can no longer be fairly represented
      const uint_fast64_t ret = randomRemainingTemp % maxValueExclusiveConverted;
      m_randomRemaining = randomRemainingTemp / maxValueExclusiveConverted;

      return static_cast<size_t>(ret);
   }
};
static_assert(std::is_standard_layout<RandomStream>::value,
   "we might want to copy this internal state cross process in the future");

#endif // RANDOM_STREAM_H