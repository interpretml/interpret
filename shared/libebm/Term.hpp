// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TERM_HPP
#define TERM_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h" // EBM_ASSERT

#include "common.hpp" // k_cDimensionsMax
#include "bridge.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class FeatureBoosting;

struct TermFeature {
   const FeatureBoosting* m_pFeature;
   size_t m_cStride;
   size_t m_iTranspose;
};

class Term final {
   size_t m_cDimensions;
   size_t m_cRealDimensions;
   size_t m_cTensorBins;
   size_t m_cAuxillaryBins;
   int m_cBitsRequiredMin;
   int m_cLogEnterGenerateTermUpdateMessages;
   int m_cLogExitGenerateTermUpdateMessages;
   int m_cLogEnterApplyTermUpdateMessages;
   int m_cLogExitApplyTermUpdateMessages;

   // IMPORTANT: m_apFeature must be in the last position for the struct hack and this must be standard layout
   TermFeature m_aTermFeatures[k_cDimensionsMax];

 public:
   Term() = default; // preserve our POD status
   ~Term() = default; // preserve our POD status
   void* operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete(void*) = delete; // we only use malloc/free in this library

   inline static size_t GetTermCountBytes(const size_t cDimensions) noexcept {
      return offsetof(Term, m_aTermFeatures) + sizeof(Term::m_aTermFeatures[0]) * cDimensions;
   }

   inline static void Free(Term* const pTerm) noexcept { free(pTerm); }

   inline void Initialize(const size_t cDimensions) noexcept {
      m_cDimensions = cDimensions;
      m_cLogEnterGenerateTermUpdateMessages = 2;
      m_cLogExitGenerateTermUpdateMessages = 2;
      m_cLogEnterApplyTermUpdateMessages = 2;
      m_cLogExitApplyTermUpdateMessages = 2;
   }

   static Term* Allocate(const size_t cDimensions) noexcept;
   static Term** AllocateTerms(const size_t cTerms) noexcept;
   static void FreeTerms(const size_t cTerms, Term** apTerms) noexcept;

   inline void SetBitsRequiredMin(const int cBitsRequiredMin) noexcept { m_cBitsRequiredMin = cBitsRequiredMin; }

   inline int GetBitsRequiredMin() const noexcept { return m_cBitsRequiredMin; }

   inline size_t GetCountDimensions() const noexcept {
      EBM_ASSERT(m_cRealDimensions <= m_cDimensions);
      return m_cDimensions;
   }

   inline size_t GetCountRealDimensions() const noexcept {
      EBM_ASSERT(m_cRealDimensions <= m_cDimensions);
      return m_cRealDimensions;
   }

   inline void SetCountRealDimensions(const size_t cRealDimensions) noexcept { m_cRealDimensions = cRealDimensions; }

   inline size_t GetCountTensorBins() const noexcept { return m_cTensorBins; }

   inline void SetCountTensorBins(const size_t cTensorBins) noexcept { m_cTensorBins = cTensorBins; }

   inline size_t GetCountAuxillaryBins() const noexcept { return m_cAuxillaryBins; }

   inline void SetCountAuxillaryBins(const size_t cAuxillaryBins) noexcept { m_cAuxillaryBins = cAuxillaryBins; }

   inline const TermFeature* GetTermFeatures() const noexcept { return ArrayToPointer(m_aTermFeatures); }
   inline TermFeature* GetTermFeatures() noexcept { return ArrayToPointer(m_aTermFeatures); }

   inline int* GetPointerCountLogEnterGenerateTermUpdateMessages() noexcept {
      return &m_cLogEnterGenerateTermUpdateMessages;
   }

   inline int* GetPointerCountLogExitGenerateTermUpdateMessages() noexcept {
      return &m_cLogExitGenerateTermUpdateMessages;
   }

   inline int* GetPointerCountLogEnterApplyTermUpdateMessages() noexcept { return &m_cLogEnterApplyTermUpdateMessages; }

   inline int* GetPointerCountLogExitApplyTermUpdateMessages() noexcept { return &m_cLogExitApplyTermUpdateMessages; }
};
static_assert(std::is_standard_layout<Term>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(
      std::is_trivial<Term>::value, "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<Term>::value, "We use a lot of C constructs, so disallow non-POD types in general");

} // namespace DEFINED_ZONE_NAME

#endif // TERM_HPP
