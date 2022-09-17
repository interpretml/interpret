// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TERM_HPP
#define TERM_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "Feature.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class Term final {
   ptrdiff_t m_cItemsPerBitPack;
   size_t m_cDimensions;
   size_t m_cRealDimensions;
   size_t m_cTensorBins;
   size_t m_cAuxillaryBins;
   int m_cLogEnterGenerateTermUpdateMessages;
   int m_cLogExitGenerateTermUpdateMessages;
   int m_cLogEnterApplyTermUpdateMessages;
   int m_cLogExitApplyTermUpdateMessages;

   // IMPORTANT: m_apFeature must be in the last position for the struct hack and this must be standard layout
   const Feature * m_apFeature[k_cDimensionsMax];

public:

   Term() = default; // preserve our POD status
   ~Term() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS constexpr static size_t GetTermCountBytes(const size_t cFeatures) noexcept {
      return offsetof(Term, m_apFeature) + sizeof(Term::m_apFeature[0]) * cFeatures;
   }

   INLINE_ALWAYS static void Free(Term * const pTerm) noexcept {
      free(pTerm);
   }

   INLINE_ALWAYS void Initialize(const size_t cFeatures) noexcept {
      m_cDimensions = cFeatures;
      m_cLogEnterGenerateTermUpdateMessages = 2;
      m_cLogExitGenerateTermUpdateMessages = 2;
      m_cLogEnterApplyTermUpdateMessages = 2;
      m_cLogExitApplyTermUpdateMessages = 2;
   }

   static Term * Allocate(const size_t cFeatures) noexcept;
   static Term ** AllocateTerms(const size_t cTerms) noexcept;
   static void FreeTerms(const size_t cTerms, Term ** apTerms) noexcept;

   INLINE_ALWAYS void SetBitPack(const ptrdiff_t cItemsPerBitPack) noexcept {
      EBM_ASSERT(k_cItemsPerBitPackDynamic2 != cItemsPerBitPack);
      m_cItemsPerBitPack = cItemsPerBitPack;
   }

   INLINE_ALWAYS ptrdiff_t GetBitPack() const noexcept {
      // don't check the legal value for m_cItemsPerBitPack here since we call this function from a huge
      // number of templates.  We check this value when SetBitPack is called
      return m_cItemsPerBitPack;
   }

   INLINE_ALWAYS size_t GetCountDimensions() const noexcept {
      EBM_ASSERT(m_cRealDimensions <= m_cDimensions);
      return m_cDimensions;
   }

   INLINE_ALWAYS size_t GetCountRealDimensions() const noexcept {
      EBM_ASSERT(m_cRealDimensions <= m_cDimensions);
      return m_cRealDimensions;
   }

   INLINE_ALWAYS void SetCountRealDimensions(const size_t cRealDimensions) noexcept {
      m_cRealDimensions = cRealDimensions;
   }

   INLINE_ALWAYS size_t GetCountTensorBins() const noexcept {
      return m_cTensorBins;
   }

   INLINE_ALWAYS void SetCountTensorBins(const size_t cTensorBins) noexcept {
      m_cTensorBins = cTensorBins;
   }

   INLINE_ALWAYS size_t GetCountAuxillaryBins() const noexcept {
      return m_cAuxillaryBins;
   }

   INLINE_ALWAYS void SetCountAuxillaryBins(const size_t cAuxillaryBins) noexcept {
      m_cAuxillaryBins = cAuxillaryBins;
   }

   INLINE_ALWAYS const Feature * const * GetFeatures() const noexcept {
      return ArrayToPointer(m_apFeature);
   }
   INLINE_ALWAYS const Feature ** GetFeatures() noexcept {
      return ArrayToPointer(m_apFeature);
   }

   INLINE_ALWAYS int * GetPointerCountLogEnterGenerateTermUpdateMessages() noexcept {
      return &m_cLogEnterGenerateTermUpdateMessages;
   }

   INLINE_ALWAYS int * GetPointerCountLogExitGenerateTermUpdateMessages() noexcept {
      return &m_cLogExitGenerateTermUpdateMessages;
   }

   INLINE_ALWAYS int * GetPointerCountLogEnterApplyTermUpdateMessages() noexcept {
      return &m_cLogEnterApplyTermUpdateMessages;
   }

   INLINE_ALWAYS int * GetPointerCountLogExitApplyTermUpdateMessages() noexcept {
      return &m_cLogExitApplyTermUpdateMessages;
   }
};
static_assert(std::is_standard_layout<Term>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<Term>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<Term>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // TERM_HPP
