// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef HISTOGRAM_BUCKET_HPP
#define HISTOGRAM_BUCKET_HPP

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t
#include <cmath> // abs
#include <string.h> // memcpy

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "GradientPair.hpp"
#include "Feature.hpp"
#include "Term.hpp"
#include "DataSetBoosting.hpp"
#include "DataSetInteraction.hpp"
#include "InnerBag.hpp"

#include "BoosterCore.hpp"
#include "InteractionCore.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<typename TFloat, bool bClassification>
struct Bin;

struct BinBase {
   BinBase() = default; // preserve our POD status
   ~BinBase() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   template<typename TFloat, bool bClassification>
   INLINE_ALWAYS Bin<TFloat, bClassification> * Specialize() {
      return static_cast<Bin<TFloat, bClassification> *>(this);
   }
   template<typename TFloat, bool bClassification>
   INLINE_ALWAYS const Bin<TFloat, bClassification> * Specialize() const {
      return static_cast<const Bin<TFloat, bClassification> *>(this);
   }

   INLINE_ALWAYS void Zero(const size_t cBytesPerBin, const size_t cBins = 1, const size_t iBin = 0) {
      // The C standard guarantees that zeroing integer types is a zero, and IEEE-754 guarantees 
      // that zeroing a floating point is zero.  Our Bin objects are POD and also only contain
      // floating point and unsigned integer types
      //
      // 6.2.6.2 Integer types -> 5. The values of any padding bits are unspecified.A valid (non - trap) 
      // object representation of a signed integer type where the sign bit is zero is a valid object 
      // representation of the corresponding unsigned type, and shall represent the same value.For any 
      // integer type, the object representation where all the bits are zero shall be a representation 
      // of the value zero in that type.

      static_assert(std::numeric_limits<float>::is_iec559, "memset of floats requires IEEE 754 to guarantee zeros");
      memset(reinterpret_cast<char *>(this) + iBin * cBytesPerBin, 0, cBins * cBytesPerBin);
   }
};
static_assert(std::is_standard_layout<BinBase>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<BinBase>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<BinBase>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

template<typename TFloat, bool bClassification>
struct Bin final : BinBase {
private:

   size_t m_cSamples;
   TFloat m_weight;

   // use the "struct hack" since Flexible array member method is not available in C++
   // m_aGradientPairs must be the last item in this struct
   // AND this class must be "is_standard_layout" since otherwise we can't guarantee that this item is placed at the bottom
   // standard layout classes have some additional odd restrictions like all the member data must be in a single class 
   // (either the parent or child) if the class is derrived
   GradientPair<TFloat, bClassification> m_aGradientPairs[1];

public:

   Bin() = default; // preserve our POD status
   ~Bin() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS size_t GetCountSamples() const {
      return m_cSamples;
   }
   INLINE_ALWAYS void SetCountSamples(const size_t cSamples) {
      m_cSamples = cSamples;
   }

   INLINE_ALWAYS TFloat GetWeight() const {
      return m_weight;
   }
   INLINE_ALWAYS void SetWeight(const TFloat weight) {
      m_weight = weight;
   }

   INLINE_ALWAYS const GradientPair<TFloat, bClassification> * GetGradientPairs() const {
      return ArrayToPointer(m_aGradientPairs);
   }
   INLINE_ALWAYS GradientPair<TFloat, bClassification> * GetGradientPairs() {
      return ArrayToPointer(m_aGradientPairs);
   }

   INLINE_ALWAYS void Add(const Bin<TFloat, bClassification> & other, const size_t cScores) {
      m_cSamples += other.m_cSamples;
      m_weight += other.m_weight;

      auto * aGradientPairs = GetGradientPairs();

      const auto * aOtherGradientPairs = other.GetGradientPairs();

      for(size_t iScore = 0; iScore < cScores; ++iScore) {
         aGradientPairs[iScore].Add(aOtherGradientPairs[iScore]);
      }
   }

   INLINE_ALWAYS void Subtract(const Bin<TFloat, bClassification> & other, const size_t cScores) {
      m_cSamples -= other.m_cSamples;
      m_weight -= other.m_weight;

      auto * aGradientPairs = GetGradientPairs();
      const auto * aOtherGradientPairs = other.GetGradientPairs();

      for(size_t iScore = 0; iScore < cScores; ++iScore) {
         aGradientPairs[iScore].Subtract(aOtherGradientPairs[iScore]);
      }
   }

   INLINE_ALWAYS void Copy(const Bin<TFloat, bClassification> & other, const size_t cScores) {
      const size_t cBytesPerBin = sizeof(Bin) - sizeof(m_aGradientPairs) + sizeof(m_aGradientPairs[0]) * cScores;

      memcpy(this, &other, cBytesPerBin);
   }

   INLINE_ALWAYS void AssertZero(const size_t cScores) const {
      UNUSED(cScores);
#ifndef NDEBUG
      EBM_ASSERT(0 == m_cSamples);
      EBM_ASSERT(0 == m_weight);

      const auto * aGradientPairs = GetGradientPairs();
      for(size_t iScore = 0; iScore < cScores; ++iScore) {
         aGradientPairs[iScore].AssertZero();
      }
#endif // NDEBUG
   }
};
static_assert(std::is_standard_layout<Bin<double, true>>::value && std::is_standard_layout<Bin<double, false>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<Bin<double, true>>::value && std::is_trivial<Bin<double, false>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<Bin<double, true>>::value && std::is_pod<Bin<double, false>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

static_assert(std::is_standard_layout<Bin<float, true>>::value && std::is_standard_layout<Bin<float, false>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<Bin<float, true>>::value && std::is_trivial<Bin<float, false>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<Bin<float, true>>::value && std::is_pod<Bin<float, false>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

template<typename TFloat>
INLINE_ALWAYS bool IsOverflowBinSize(const bool bClassification, const size_t cScores) {
   const size_t cBytesPerGradientPair = GetGradientPairSize<TFloat>(bClassification);

   if(UNLIKELY(IsMultiplyError(cBytesPerGradientPair, cScores))) {
      return true;
   }

   size_t cBytesBinComponent;
   if(bClassification) {
      cBytesBinComponent = sizeof(Bin<TFloat, true>);
   } else {
      cBytesBinComponent = sizeof(Bin<TFloat, false>);
   }
   cBytesBinComponent -= cBytesPerGradientPair;

   if(UNLIKELY(IsAddError(cBytesBinComponent, cBytesPerGradientPair * cScores))) {
      return true;
   }

   return false;
}

template<typename TFloat>
INLINE_ALWAYS size_t GetBinSize(const bool bClassification, const size_t cScores) {
   // TODO: someday try out bin sizes that are a power of two.  This would allow us to use a shift when using bins
   //       instead of using multiplications.  In that version return the number of bits to shift here to make it easy
   //       to get either the shift required for indexing OR the number of bytes (shift 1 << num_bits)

   const size_t cBytesPerGradientPair = GetGradientPairSize<TFloat>(bClassification);

   size_t cBytesBinComponent;
   if(bClassification) {
      cBytesBinComponent = sizeof(Bin<TFloat, true>);
   } else {
      cBytesBinComponent = sizeof(Bin<TFloat, false>);
   }
   cBytesBinComponent -= cBytesPerGradientPair;

   return cBytesBinComponent + cBytesPerGradientPair * cScores;
}

template<typename TFloat, bool bClassification>
INLINE_ALWAYS Bin<TFloat, bClassification> * IndexBin(
   Bin<TFloat, bClassification> * const aBins,
   const size_t iByte
) {
   return reinterpret_cast<Bin<TFloat, bClassification> *>(reinterpret_cast<char *>(aBins) + iByte);
}

template<typename TFloat, bool bClassification>
INLINE_ALWAYS const Bin<TFloat, bClassification> * IndexBin(
   const Bin<TFloat, bClassification> * const aBins,
   const size_t iByte
) {
   return reinterpret_cast<const Bin<TFloat, bClassification> *>(reinterpret_cast<const char *>(aBins) + iByte);
}

INLINE_ALWAYS BinBase * IndexBin(
   BinBase * const aBins,
   const size_t iByte
) {
   return reinterpret_cast<BinBase *>(reinterpret_cast<char *>(aBins) + iByte);
}

INLINE_ALWAYS const BinBase * IndexBin(
   const BinBase * const aBins,
   const size_t iByte
) {
   return reinterpret_cast<const BinBase *>(reinterpret_cast<const char *>(aBins) + iByte);
}

// keep this as a MACRO so that we don't materialize any of the parameters on non-debug builds
#define ASSERT_BIN_OK(MACRO_cBytesPerBin, MACRO_pBin, MACRO_pBinsEnd) \
   (EBM_ASSERT(reinterpret_cast<const char *>(MACRO_pBin) + static_cast<size_t>(MACRO_cBytesPerBin) <= \
      reinterpret_cast<const char *>(MACRO_pBinsEnd)))

} // DEFINED_ZONE_NAME

#endif // HISTOGRAM_BUCKET_HPP
