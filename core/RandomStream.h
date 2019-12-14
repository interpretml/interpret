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
   bool m_bSuccess; // make it zero the error just in case someone introduces an initialization bug such that this doesn't set set.  The default will be an error then

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
      , m_randomGenerator(static_cast<unsigned int>(seed)) {
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

   EBM_INLINE size_t Next(const size_t maxValueInclusive) {
      // anyone calling this function should wrap it in an try/catch.  We're not wrapping it here because if this is being called in a loop we'd rather
      // move the try/catch overhead to ourside that loop

      // TODO : change this to use the AES instruction set, which would ensure compatibility between languages and it would only take 2-3 clock cycles (although we'd still probably need to div [can we multiply instead] which is expensive).

      // TODO: uniform_int_distribution suposedly doesn't return cross platform identical results, so we should roll our own someday.  Check first that we actually need to do this
      //       https://stackoverflow.com/questions/40361041/achieve-same-random-number-sequence-on-different-os-with-same-seed

      // initializing uniform_int_distribution doesn't have official nothrow properties, but a random number generator should not have to throw
      std::uniform_int_distribution<size_t> distribution(size_t { 0 }, maxValueInclusive);
      return distribution(m_randomGenerator);
   }

   EBM_INLINE bool IsSuccess() const {
      return m_bSuccess;
   }
};

#endif // RANDOM_STREAM_H