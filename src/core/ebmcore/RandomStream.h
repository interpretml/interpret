// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef RANDOM_STREAM_H
#define RANDOM_STREAM_H

#include <inttypes.h> // uint32_t seed
#include <random>
#include <stddef.h> // size_t, ptrdiff_t

#include "ebmcore.h" // IntegerDataType
#include "EbmInternal.h" // TML_INLINE

class RandomStream final {
   std::default_random_engine m_randomGenerator;

public:
   TML_INLINE RandomStream(IntegerDataType seed)
      : m_randomGenerator((unsigned int)seed) { // initializing default_random_engine doesn't have official nothrow properties, but a random number generator should not have to throw
                                  // TODO : someday we might need our own random generator which can repeat the same numbers at a later date for instance for re-starting a previous training run.  Use the Mersenne Twister algorithm.  Also, if we need cross language random synchronization, return uint64_t instead of size_t since size_t is platform dependent
   }

   TML_INLINE size_t Next(const size_t minValueInclusive, const size_t maxValueInclusive) {
      // initializing uniform_int_distribution doesn't have official nothrow properties, but a random number generator should not have to throw
      std::uniform_int_distribution<size_t> distribution(minValueInclusive, maxValueInclusive);
      return distribution(m_randomGenerator);
   }

   TML_INLINE size_t Next(const size_t maxValueInclusive) {
      return Next(0, maxValueInclusive);
   }
};

#endif // RANDOM_STREAM_H