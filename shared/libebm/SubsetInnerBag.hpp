// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef SUBSET_INNER_BAG_HPP
#define SUBSET_INNER_BAG_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "unzoned.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct DataSetBoosting;

struct SubsetInnerBag final {
   friend DataSetBoosting;

   SubsetInnerBag() = default; // preserve our POD status
   ~SubsetInnerBag() = default; // preserve our POD status
   void* operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete(void*) = delete; // we only use malloc/free in this library

   static SubsetInnerBag* AllocateSubsetInnerBags(const size_t cInnerBags);
   static void FreeSubsetInnerBags(const size_t cInnerBags, SubsetInnerBag* const aSubsetInnerBags);

   inline const void* GetWeights() const { return m_aWeights; }

 private:
   // Sampling with replacement is the more theoretically correct method of sampling, but it has the drawback that
   // we need to keep a count of the number of times each sample is selected in the dataset.
   // Sampling without replacement would require 1 bit or byte per case, so it can be faster and
   // we wouldn't need to use float weights. We could use a branchless comparison to get either 0.0 or the
   // gradient or hessian

   void* m_aWeights;
};
static_assert(std::is_standard_layout<SubsetInnerBag>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<SubsetInnerBag>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");

} // namespace DEFINED_ZONE_NAME

#endif // SUBSET_INNER_BAG_HPP
