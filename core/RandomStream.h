// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef RANDOM_STREAM_H
#define RANDOM_STREAM_H

#include <inttypes.h> // uint32_t seed
#include <random>
#include <stddef.h> // size_t, ptrdiff_t

#include "ebmcore.h" // IntegerDataType
#include "EbmInternal.h" // EBM_INLINE

class RandomStream final {
   static constexpr uint_fast64_t k_max = std::mt19937_64::max();
   static constexpr uint_fast64_t k_min = std::mt19937_64::min();

   bool m_bSuccess; // make it zero the error just in case someone introduces an initialization bug such that this doesn't set set.  The default will be an error then

   // uniform_int_distribution isn't guaranteed to be cross platform compatible, in fact it isn't between Windows/Mac/Linux
   uint_fast64_t randomRemainingMax;
   uint_fast64_t randomRemaining;

   // THIS SHOULD ALWAYS BE THE LAST ITEM IN THIS STRUCTURE.  C++ guarantees that constructions initialize data members in the order that they are declared
   // since this class can potentially throw an exception in the constructor, we leave it last so that we are guaranteed that the rest of our object has been initialized

#ifdef LEGACY_COMPATIBILITY
   std::default_random_engine m_randomGenerator;
#else // LEGACY_COMPATIBILITY
   // use std::mt19937_64 for cross platform random number identical results
   std::mt19937_64 m_randomGenerator;
#endif // LEGACY_COMPATIBILITY

public:
   // in case you were wondering, this odd syntax of putting a try outside the function is called "Function try blocks" and it's the best way of handling exception in initialization
   RandomStream(const IntegerDataType seed) try
      : m_bSuccess(false)
      , randomRemainingMax(0)
      , randomRemaining(0)
#ifdef LEGACY_COMPATIBILITY
      , m_randomGenerator(static_cast<unsigned int>(seed)) {
#else // LEGACY_COMPATIBILITY
      , m_randomGenerator(static_cast<uint_fast64_t>(static_cast<uint64_t>(seed))) {
#endif // LEGACY_COMPATIBILITY
      // an unfortunate thing about function exception handling is that accessing non-static data from the catch block gives undefined behavior
      // so, we can't set m_bSuccess to false if an error occurs, so instead we set it to false in the static initialization
      // C++ guarantees that initialization will occur in the order the variables are declared (not in the order of initialization)
      // but since we put m_bSuccess at the top, if an exception occurs then our m_bSuccess will be left as false since it won't call the 
      // initializer which sets it to true
      // https://en.cppreference.com/w/cpp/language/function-try-block
      m_bSuccess = true;
   } catch(...) {
      // the only reason we should potentially find outselves here is if there was an exception thrown during construction
      // C++ exceptions are suposed to be thrown by value and caught by reference, so it shouldn't be a pointer, and we shouldn't leak memory
   }

#ifdef LEGACY_COMPATIBILITY
   EBM_INLINE size_t Next(const size_t maxValueExclusive) {
      // anyone calling this function should wrap it in an try/catch.  We're not wrapping it here because if this is being called in a loop we'd rather
      // move the try/catch overhead to ourside that loop

      // initializing uniform_int_distribution doesn't have official nothrow properties, but a random number generator should not have to throw
      std::uniform_int_distribution<size_t> distribution(size_t { 0 }, maxValueExclusive - 1);
      return distribution(m_randomGenerator);
   }
#else // LEGACY_COMPATIBILITY
   EBM_INLINE size_t Next(const size_t maxValueExclusive) {
      // std::uniform_int_distribution doesn't give cross platform identical results, so roll our own (it's more efficient too I think, or at least not worse)
      static_assert(std::numeric_limits<size_t>::max() <= k_max - k_min, "k_max - k_min isn't large enough to encompass size_t");
      static_assert(std::numeric_limits<size_t>::max() <= std::numeric_limits<uint_fast64_t>::max(), "uint_fast64_t isn't large enough to encompass size_t");
      const uint_fast64_t maxValueExclusiveConverted = static_cast<uint_fast64_t>(maxValueExclusive);

      // let's say our user requests a value of 6 exclusive, so we have 6 possible random values (0,1,2,3,4,5)
      // let's say our randomRemainingMax is 6 or 7.  If randomRemaining is 6 or 7, we don't want to use 6%6 or 7%6 and re-use 0 or 1, since then 0 and 1 are twice as likley as 0-5.
      // so if randomRemaining is equal to 6 or above, we want to re-generate the random numbers
      // and odd case arrises if randomRemainingMax == 5, because then there is no illegal value for randomRemaining.  We can optionally regenerate the random number since it's balanced anyways
      // if randomRemainingMax == 4, then we want to re-generate it since we can't express 5
      //
      // per above, we could instead use randomRemainingMax + 1, but then we'd overflow if randomRemainingMax was std::numeric_limits<uint_fast64_t>::max()
      // when randomRemainingMax + 1 == maxValueExclusiveConverted,or in fact whenever (randomRemainingMax + 1) % maxValueExclusiveConverted == 0 then we have a special case where no random value is bad
      // since our random value is perfectly divisible by maxValueExclusiveConverted.  We can re-generate it though, and that's the easiest thing to do in that case, so avoid an additional check by regenerating
      uint_fast64_t randomRemainingTempMax = randomRemainingMax;
      uint_fast64_t randomRemainingTemp = randomRemaining;
      while(true) {
         randomRemainingTempMax /= maxValueExclusiveConverted;
         const uint_fast64_t randomRemainingIllegal = randomRemainingTempMax * maxValueExclusiveConverted;
         if(randomRemainingTemp < randomRemainingIllegal) {
            break;
         }
         // TODO : consider changing this to use the AES instruction set, which would ensure compatibility between languages and it would only take 2-3 clock cycles (although we'd still probably need to div [can we multiply instead] which is expensive).
         // 
         // this ridiculous round bracket operator overload of m_randomGenerator gets new random bits
         randomRemainingTemp = m_randomGenerator() - k_min;
         randomRemainingTempMax = k_max - k_min;
      }
      EBM_ASSERT(0 < randomRemainingTempMax); // if 0 == randomRemainingMax then we would have stayed in the loop since randomRemaining could not be less than zero
      randomRemainingMax = randomRemainingTempMax - 1; // the top range can no longer be fairly represented
      const uint_fast64_t ret = randomRemainingTemp % maxValueExclusiveConverted;
      randomRemaining = randomRemainingTemp / maxValueExclusiveConverted;

      return static_cast<size_t>(ret);
   }
#endif // LEGACY_COMPATIBILITY

   EBM_INLINE bool IsSuccess() const {
      return m_bSuccess;
   }
};

#endif // RANDOM_STREAM_H