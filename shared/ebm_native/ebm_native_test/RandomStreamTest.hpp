// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef RANDOM_STREAM_TEST_HPP
#define RANDOM_STREAM_TEST_HPP

#include <inttypes.h> // uint32_t
#include <random>
#include <stddef.h> // size_t, ptrdiff_t
#include <assert.h>

class RandomStreamTest final {
   static constexpr uint_fast64_t k_max = std::mt19937_64::max();
   static constexpr uint_fast64_t k_min = std::mt19937_64::min();

   // make it zero the error just in case someone introduces an initialization bug such that this doesn't set set.  The default will be an error then
   bool m_bSuccess;

   // uniform_int_distribution isn't guaranteed to be cross platform compatible, in fact it isn't between Windows/Mac/Linux
   uint_fast64_t randomRemainingMax;
   uint_fast64_t randomRemaining;

   // THIS SHOULD ALWAYS BE THE LAST ITEM IN THIS STRUCTURE.  C++ guarantees that constructions initialize data members in the order that they are declared
   // since this class can potentially throw an exception in the constructor, we leave it last so that we are guaranteed that the rest of our object 
   // has been initialized
   // we won't be able to generate perfeclty identical results between platforms given that true floating point determinism would required
   // that we implement our own floating poing processing in software, rather than use hardware, and that would be too slow.
   // More Details: https://randomascii.wordpress.com/2013/07/16/floating-point-determinism/
   // And: https://randomascii.wordpress.com/2012/03/21/intermediate-floating-point-precision/
   // but we can get very close down to just a few decimals, which for a lot of datasets probably allows 
   // us to at least have identical splits in most cases, and if we round we can probably get identical floating point numbers displayed as well
   // exp and log are both difficult to reproduct accross implementations as exact calculations often require hundreds of bits of precision, 
   // and doing this better is an active area of reserach 

   // use std::mt19937_64 for cross platform random number identical results for the random number generator
   std::mt19937_64 m_randomGenerator;

public:
   // in case you were wondering, this odd syntax of putting a try outside the function is called "Function try blocks" and it's the best way of handling 
   // exception in initialization
   RandomStreamTest(const IntEbm seed) try
      : m_bSuccess(false)
      , randomRemainingMax(0)
      , randomRemaining(0)
      , m_randomGenerator(static_cast<uint_fast64_t>(static_cast<uint64_t>(seed))) {
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

   inline size_t Next(const size_t maxValExclusive) {
      // std::uniform_int_distribution doesn't give cross platform identical results, so roll our own (it's more efficient too I think, or at least not worse)
      static_assert(std::numeric_limits<size_t>::max() <= k_max - k_min, "k_max - k_min isn't large enough to encompass size_t");
      static_assert(std::numeric_limits<size_t>::max() <= std::numeric_limits<uint_fast64_t>::max(), "uint_fast64_t isn't large enough to encompass size_t");
      const uint_fast64_t maxValExclusiveConverted = static_cast<uint_fast64_t>(maxValExclusive);

      // let's say our user requests a value of 6 exclusive, so we have 6 possible random values (0,1,2,3,4,5)
      // let's say our randomRemainingMax is 6 or 7.  If randomRemaining is 6 or 7, we don't want to use 6%6 or 7%6 and re-use 0 or 1, since then 0 and 1 
      // are twice as likley as 0-5.  So if randomRemaining is equal to 6 or above, we want to re-generate the random numbers and odd case arrises if 
      // randomRemainingMax == 5, because then there is no illegal value for randomRemaining.  We can optionally regenerate the random number since it's 
      // balanced anyways if randomRemainingMax == 4, then we want to re-generate it since we can't express 5
      //
      // per above, we could instead use randomRemainingMax + 1, but then we'd overflow if randomRemainingMax was std::numeric_limits<uint_fast64_t>::max()
      // when randomRemainingMax + 1 == maxValExclusiveConverted,or in fact whenever (randomRemainingMax + 1) % maxValExclusiveConverted == 0 then we 
      // have a special case where no random value is bad since our random value is perfectly divisible by maxValExclusiveConverted.  We can re-generate 
      // it though, and that's the easiest thing to do in that case, so avoid an additional check by regenerating
      uint_fast64_t randomRemainingTempMax = randomRemainingMax;
      uint_fast64_t randomRemainingTemp = randomRemaining;
      while(true) {
         randomRemainingTempMax /= maxValExclusiveConverted;
         const uint_fast64_t randomRemainingIllegal = randomRemainingTempMax * maxValExclusiveConverted;
         if(randomRemainingTemp < randomRemainingIllegal) {
            break;
         }
         // TODO : consider changing this to use the AES instruction set, which would ensure compatibility between languages and it would only 
         //   take 2-3 clock cycles (although we'd still probably need to div [can we multiply instead] which is expensive).
         // 
         // this ridiculous round bracket operator overload of m_randomGenerator gets new random bits
         randomRemainingTemp = m_randomGenerator() - k_min;
         randomRemainingTempMax = k_max - k_min;
      }
      // if 0 == randomRemainingMax then we would have stayed in the loop since randomRemaining could not be less than zero
      assert(0 < randomRemainingTempMax);
      randomRemainingMax = randomRemainingTempMax - 1; // the top range can no longer be fairly represented
      const uint_fast64_t ret = randomRemainingTemp % maxValExclusiveConverted;
      randomRemaining = randomRemainingTemp / maxValExclusiveConverted;

      return static_cast<size_t>(ret);
   }

   inline bool IsSuccess() const {
      return m_bSuccess;
   }
};

#endif // RANDOM_STREAM_TEST_HPP