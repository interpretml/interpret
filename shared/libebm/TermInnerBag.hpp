// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TERM_INNER_BAG_HPP
#define TERM_INNER_BAG_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "unzoned.h"

#include "ebm_internal.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct DataSetBoosting;
class Term;

struct TermInnerBag final {
   friend DataSetBoosting;

   TermInnerBag() = default; // preserve our POD status
   ~TermInnerBag() = default; // preserve our POD status
   void* operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete(void*) = delete; // we only use malloc/free in this library

   static TermInnerBag** AllocateTermInnerBags(const size_t cTerms);
   static ErrorEbm InitTermInnerBags(const size_t cTerms,
         const Term* const* const apTerms,
         TermInnerBag** const aaTermInnerBags,
         const size_t cInnerBags);
   static void FreeTermInnerBags(const size_t cTerms, TermInnerBag** const aaTermInnerBags, const size_t cInnerBags);

   static inline const UIntMain* GetCounts(const bool bCollapsed,
         const size_t iTerm,
         const size_t iBag,
         const TermInnerBag* const* const aaTermInnerBags) {
      return bCollapsed ? &aaTermInnerBags[iTerm][iBag].collapsedCount :
                                        aaTermInnerBags[iTerm][iBag].m_aCounts;
   }
   static inline UIntMain* GetCounts(
         const bool bCollapsed, const size_t iTerm, const size_t iBag, TermInnerBag** const aaTermInnerBags) {
      return bCollapsed ? &aaTermInnerBags[iTerm][iBag].collapsedCount :
                                        aaTermInnerBags[iTerm][iBag].m_aCounts;
   }
   static inline const FloatPrecomp* GetWeights(const bool bCollapsed,
         const size_t iTerm,
         const size_t iBag,
         const TermInnerBag* const* const aaTermInnerBags) {
      return bCollapsed ? &aaTermInnerBags[iTerm][iBag].collapsedWeight :
                                        aaTermInnerBags[iTerm][iBag].m_aWeights;
   }
   static inline FloatPrecomp* GetWeights(
         const bool bCollapsed, const size_t iTerm, const size_t iBag, TermInnerBag** const aaTermInnerBags) {
      return bCollapsed ? &aaTermInnerBags[iTerm][iBag].collapsedWeight :
                                        aaTermInnerBags[iTerm][iBag].m_aWeights;
   }

 private:
   UIntMain collapsedCount;
   FloatPrecomp collapsedWeight;

   UIntMain* m_aCounts;
   FloatPrecomp* m_aWeights;
};
static_assert(std::is_standard_layout<TermInnerBag>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<TermInnerBag>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");

} // namespace DEFINED_ZONE_NAME

#endif // TERM_INNER_BAG_HPP
