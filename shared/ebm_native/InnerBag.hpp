// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef INNER_BAG_HPP
#define INNER_BAG_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "common_c.h" // FloatFast
#include "zones.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class RandomDeterministic;

class InnerBag final {
   // Sampling with replacement is the more theoretically correct method of sampling, but it has the drawback that 
   // we need to keep a count of the number of times each sample is selected in the dataset.  
   // Sampling without replacement would require 1 bit per case, so it can be faster.

   // TODO : make this a struct of FractionalType and size_t counts and use MACROS to have either size_t or 
   // FractionalType or both, and perf how this changes things.  We don't get a benefit anywhere by storing 
   // the raw data in both formats since it is never converted anyways, but this count is!
   size_t * m_aCountOccurrences;
   FloatFast * m_aWeights;
   FloatBig m_weightTotal;

   // we take owernship of the aWeights array

   static ErrorEbm GenerateSingleInnerBag(
      void * const rng,
      const size_t cSamples,
      const FloatFast * const aWeights,
      InnerBag ** const ppOut
   );
   static InnerBag * GenerateFlatInnerBag(
      const size_t cSamples,
      const FloatFast * const aWeights
   );
   void Free();
   void InitializeUnfailing();

public:

   InnerBag() = default; // preserve our POD status
   ~InnerBag() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   const size_t * GetCountOccurrences() const {
      return m_aCountOccurrences;
   }
   const FloatFast * GetWeights() const {
      return m_aWeights;
   }
   FloatBig GetWeightTotal() const {
      return m_weightTotal;
   }

   static ErrorEbm GenerateInnerBags(
      void * const rng,
      const size_t cSamples,
      const FloatFast * const aWeights,
      const size_t cInnerBags,
      InnerBag *** const papOut
   );
   static void FreeInnerBags(const size_t cInnerBags, InnerBag ** const apInnerBags);
};
static_assert(std::is_standard_layout<InnerBag>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<InnerBag>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<InnerBag>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // INNER_BAG_HPP
