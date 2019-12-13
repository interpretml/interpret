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
   bool m_bError;

   // TODO: change from std::default_random_engine to std::mt19937_64 m_randomGenerator for cross platform random number identical results
   // TODO: uniform_int_distribution suposedly doesn't return cross platform identical results, so we should roll our own someday
   //       https://stackoverflow.com/questions/40361041/achieve-same-random-number-sequence-on-different-os-with-same-seed
   std::default_random_engine m_randomGenerator;

public:
   RandomStream(const IntegerDataType seed) try
      : m_bError(false)
      , m_randomGenerator(static_cast<unsigned int>(seed)) {
      // TODO : change this to use the AES instruction set, which would ensure compatibility between languages and it would only take 2-3 clock cycles (although we'd still probably need to div [can we multiply instead] which is expensive).
   } catch(...) {
      m_bError = true;
   }

   EBM_INLINE size_t Next(const size_t maxValueInclusive) {
      // initializing uniform_int_distribution doesn't have official nothrow properties, but a random number generator should not have to throw
      std::uniform_int_distribution<size_t> distribution(size_t { 0 }, maxValueInclusive);
      return distribution(m_randomGenerator);
   }

   EBM_INLINE bool IsError() const {
      return m_bError;
   }
};

#endif // RANDOM_STREAM_H