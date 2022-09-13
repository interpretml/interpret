// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef BIN_HPP
#define BIN_HPP

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t
#include <cmath> // abs
#include <string.h> // memcpy

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "common_cpp.hpp"
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

   INLINE_ALWAYS void ZeroMem(const size_t cBytesPerBin, const size_t cBins = 1, const size_t iBin = 0) {
      // The C standard guarantees that memset to 0 on integer types is a zero, and IEEE-754 guarantees 
      // that mem zeroing a floating point is zero.  Our Bin objects are POD and also only contain floating point
      // and unsigned integer types, so memset is legal. We do not use pointers which would be implementation defined.
      //
      // 6.2.6.2 Integer types -> 5. The values of any padding bits are unspecified.A valid (non - trap) 
      // object representation of a signed integer type where the sign bit is zero is a valid object 
      // representation of the corresponding unsigned type, and shall represent the same value.For any 
      // integer type, the object representation where all the bits are zero shall be a representation 
      // of the value zero in that type.

      static_assert(std::numeric_limits<float>::is_iec559, "memset of floats requires IEEE 754 to guarantee zeros");
      memset(reinterpret_cast<char *>(this) + iBin * cBytesPerBin, 0, cBytesPerBin * cBins);
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

   // IMPORTANT: m_aGradientPairs must be in the last position for the struct hack and this must be standard layout
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

      auto * const aThisGradientPairs = GetGradientPairs();
      const auto * const aOtherGradientPairs = other.GetGradientPairs();

      EBM_ASSERT(1 <= cScores);
      size_t iScore = 0;
      do {
         aThisGradientPairs[iScore] += aOtherGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);
   }

   INLINE_ALWAYS void Subtract(const Bin<TFloat, bClassification> & other, const size_t cScores) {
      m_cSamples -= other.m_cSamples;
      m_weight -= other.m_weight;

      auto * const aThisGradientPairs = GetGradientPairs();
      const auto * const aOtherGradientPairs = other.GetGradientPairs();

      EBM_ASSERT(1 <= cScores);
      size_t iScore = 0;
      do {
         aThisGradientPairs[iScore] -= aOtherGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);
   }

   INLINE_ALWAYS void Copy(const Bin<TFloat, bClassification> & other, const size_t cScores) {
      m_cSamples = other.m_cSamples;
      m_weight = other.m_weight;

      auto * const aThisGradientPairs = GetGradientPairs();
      const auto * const aOtherGradientPairs = other.GetGradientPairs();

      EBM_ASSERT(1 <= cScores);
      size_t iScore = 0;
      do {
         aThisGradientPairs[iScore] = aOtherGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);
   }

   INLINE_ALWAYS void AddTo(
      size_t & cSamplesInOut, 
      TFloat & weightInOut,
      GradientPair<TFloat, bClassification> * const aGradientPairsInOut,
      const size_t cScores
   ) const {
      cSamplesInOut += m_cSamples;
      weightInOut += m_weight;

      const auto * const aThisGradientPairs = GetGradientPairs();

      EBM_ASSERT(1 <= cScores);
      size_t iScore = 0;
      do {
         aGradientPairsInOut[iScore] += aThisGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);
   }

   INLINE_ALWAYS void SubtractTo(
      size_t & cSamplesInOut,
      TFloat & weightInOut,
      GradientPair<TFloat, bClassification> * const aGradientPairsInOut,
      const size_t cScores
   ) const {
      cSamplesInOut -= m_cSamples;
      weightInOut -= m_weight;

      const auto * const aThisGradientPairs = GetGradientPairs();

      EBM_ASSERT(1 <= cScores);
      size_t iScore = 0;
      do {
         aGradientPairsInOut[iScore] -= aThisGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);
   }

   INLINE_ALWAYS void CopyTo(
      size_t & cSamplesInOut,
      TFloat & weightInOut,
      GradientPair<TFloat, bClassification> * const aGradientPairsInOut,
      const size_t cScores
   ) const {
      cSamplesInOut = m_cSamples;
      weightInOut = m_weight;

      const auto * const aThisGradientPairs = GetGradientPairs();

      EBM_ASSERT(1 <= cScores);
      size_t iScore = 0;
      do {
         aGradientPairsInOut[iScore] = aThisGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);
   }


   INLINE_ALWAYS void AddFrom(
      const size_t & cSamples,
      const TFloat & weight,
      const GradientPair<TFloat, bClassification> * const aGradientPairs,
      const size_t cScores
   ) {
      m_cSamples += cSamples;
      m_weight += weight;

      auto * const aThisGradientPairs = GetGradientPairs();

      EBM_ASSERT(1 <= cScores);
      size_t iScore = 0;
      do {
         aThisGradientPairs[iScore] += aGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);
   }

   INLINE_ALWAYS void SubtractFrom(
      const size_t & cSamples,
      const TFloat & weight,
      const GradientPair<TFloat, bClassification> * const aGradientPairs,
      const size_t cScores
   ) {
      m_cSamples -= cSamples;
      m_weight -= weight;

      auto * const aThisGradientPairs = GetGradientPairs();

      EBM_ASSERT(1 <= cScores);
      size_t iScore = 0;
      do {
         aThisGradientPairs[iScore] -= aGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);
   }

   INLINE_ALWAYS void CopyFrom(
      const size_t & cSamples,
      const TFloat & weight,
      const GradientPair<TFloat, bClassification> * const aGradientPairs,
      const size_t cScores
   ) {
      m_cSamples = cSamples;
      m_weight = weight;

      auto * const aThisGradientPairs = GetGradientPairs();

      EBM_ASSERT(1 <= cScores);
      size_t iScore = 0;
      do {
         aThisGradientPairs[iScore] = aGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);
   }

   INLINE_ALWAYS bool IsBinClose(
      const size_t & cSamples,
      const TFloat & weight,
      const GradientPair<TFloat, bClassification> * const aGradientPairs,
      const size_t cScores
   ) const {
      if(cSamples != m_cSamples) {
         return false;
      }
      if(!IsClose(weight, m_weight)) {
         return false;
      }

      const auto * const aThisGradientPairs = GetGradientPairs();

      EBM_ASSERT(1 <= cScores);
      size_t iScore = 0;
      do {
         if(!aThisGradientPairs[iScore].IsGradientsClose(aGradientPairs[iScore]))             {
            return false;
         }
         ++iScore;
      } while(cScores != iScore);
   }

   INLINE_ALWAYS void AssertZero(const size_t cScores) const {
      UNUSED(cScores);
#ifndef NDEBUG
      EBM_ASSERT(0 == m_cSamples);
      EBM_ASSERT(0 == m_weight);

      const auto * const aThisGradientPairs = GetGradientPairs();

      EBM_ASSERT(1 <= cScores);
      size_t iScore = 0;
      do {
         aThisGradientPairs[iScore].AssertZero();
         ++iScore;
      } while(cScores != iScore);
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

INLINE_ALWAYS BinBase * IndexBin(BinBase * const aBins, const size_t iByte) {
   return reinterpret_cast<BinBase *>(reinterpret_cast<char *>(aBins) + iByte);
}

INLINE_ALWAYS const BinBase * IndexBin(const BinBase * const aBins, const size_t iByte) {
   return reinterpret_cast<const BinBase *>(reinterpret_cast<const char *>(aBins) + iByte);
}

template<typename TFloat, bool bClassification>
INLINE_ALWAYS const Bin<TFloat, bClassification> * NegativeIndexBin(
   const Bin<TFloat, bClassification> * const aBins,
   const size_t iByte
) {
   return reinterpret_cast<const Bin<TFloat, bClassification> *>(reinterpret_cast<const char *>(aBins) - iByte);
}

template<typename TFloat, bool bClassification>
INLINE_ALWAYS Bin<TFloat, bClassification> * NegativeIndexBin(
   Bin<TFloat, bClassification> * const aBins,
   const size_t iByte
) {
   return reinterpret_cast<Bin<TFloat, bClassification> *>(reinterpret_cast<char *>(aBins) - iByte);
}

template<typename TFloat, bool bClassification>
INLINE_ALWAYS size_t CountBins(
   const Bin<TFloat, bClassification> * const pBinLow,
   const Bin<TFloat, bClassification> * const pBinHigh,
   const size_t cBytesPerBin
) {
   EBM_ASSERT(pBinLow <= pBinHigh);
   const size_t cBytesDiff = reinterpret_cast<const char *>(pBinHigh) - reinterpret_cast<const char *>(pBinLow);
   EBM_ASSERT(0 == cBytesDiff % cBytesPerBin);
   return cBytesDiff / cBytesPerBin;
}


// keep this as a MACRO so that we don't materialize any of the parameters on non-debug builds
#define ASSERT_BIN_OK(MACRO_cBytesPerBin, MACRO_pBin, MACRO_pBinsEnd) \
   (EBM_ASSERT(reinterpret_cast<const char *>(MACRO_pBin) + static_cast<size_t>(MACRO_cBytesPerBin) <= \
      reinterpret_cast<const char *>(MACRO_pBinsEnd)))

} // DEFINED_ZONE_NAME

#endif // BIN_HPP
