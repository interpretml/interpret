// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef BIN_HPP
#define BIN_HPP

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t
#include <cmath> // abs
#include <string.h> // memcpy

#include "logging.h" // EBM_ASSERT
#include "common_c.h" // UNUSED
#include "zones.h"

#include "GradientPair.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<typename TFloat, bool bClassification, size_t cCompilerScores = 1>
struct Bin;

struct BinBase {
   BinBase() = default; // preserve our POD status
   ~BinBase() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   template<typename TFloat, bool bClassification, size_t cCompilerScores = 1>
   inline Bin<TFloat, bClassification, cCompilerScores> * Specialize() {
      return static_cast<Bin<TFloat, bClassification, cCompilerScores> *>(this);
   }
   template<typename TFloat, bool bClassification, size_t cCompilerScores = 1>
   inline const Bin<TFloat, bClassification, cCompilerScores> * Specialize() const {
      return static_cast<const Bin<TFloat, bClassification, cCompilerScores> *>(this);
   }

   inline void ZeroMem(const size_t cBytesPerBin, const size_t cBins = 1, const size_t iBin = 0) {
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
      memset(IndexByte(this, iBin * cBytesPerBin), 0, cBytesPerBin * cBins);
   }
};
static_assert(std::is_standard_layout<BinBase>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<BinBase>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<BinBase>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");


template<typename TFloat>
static bool IsOverflowBinSize(const bool bClassification, const size_t cScores);
template<typename TFloat>
static size_t GetBinSize(const bool bClassification, const size_t cScores);

template<typename TFloat, bool bClassification, size_t cCompilerScores>
struct Bin final : BinBase {
   template<typename> friend bool IsOverflowBinSize(const bool, const size_t);
   template<typename> friend size_t GetBinSize(const bool, const size_t);

private:

   size_t m_cSamples;
   TFloat m_weight;

   // IMPORTANT: m_aGradientPairs must be in the last position for the struct hack and this must be standard layout
   GradientPair<TFloat, bClassification> m_aGradientPairs[cCompilerScores];

public:

   Bin() = default; // preserve our POD status
   ~Bin() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   inline size_t GetCountSamples() const {
      return m_cSamples;
   }
   inline void SetCountSamples(const size_t cSamples) {
      m_cSamples = cSamples;
   }

   inline TFloat GetWeight() const {
      return m_weight;
   }
   inline void SetWeight(const TFloat weight) {
      m_weight = weight;
   }

   inline const GradientPair<TFloat, bClassification> * GetGradientPairs() const {
      return ArrayToPointer(m_aGradientPairs);
   }
   inline GradientPair<TFloat, bClassification> * GetGradientPairs() {
      return ArrayToPointer(m_aGradientPairs);
   }

   inline const Bin<TFloat, bClassification, 1> * Downgrade() const {
      return reinterpret_cast<const Bin<TFloat, bClassification, 1> *>(this);
   }
   inline Bin<TFloat, bClassification, 1> * Downgrade() {
      return reinterpret_cast<Bin<TFloat, bClassification, 1> *>(this);
   }

   inline void Add(
      const size_t cScores,
      const Bin & other,
      const GradientPair<TFloat, bClassification> * const aOtherGradientPairs,
      GradientPair<TFloat, bClassification> * const aThisGradientPairs
   ) {
      EBM_ASSERT(1 == cCompilerScores || cScores == cCompilerScores);
      EBM_ASSERT(cScores != cCompilerScores || aOtherGradientPairs == other.GetGradientPairs());
      EBM_ASSERT(cScores != cCompilerScores || aThisGradientPairs == GetGradientPairs());

      m_cSamples += other.m_cSamples;
      m_weight += other.m_weight;

      EBM_ASSERT(1 <= cScores);
      size_t iScore = 0;
      do {
         aThisGradientPairs[iScore] += aOtherGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);
   }
   inline void Add(
      const size_t cScores,
      const Bin & other,
      const GradientPair<TFloat, bClassification> * const aOtherGradientPairs
   ) {
      Add(cScores, other, aOtherGradientPairs, GetGradientPairs());
   }
   inline void Add(const size_t cScores, const Bin & other) {
      Add(cScores, other, other.GetGradientPairs(), GetGradientPairs());
   }

   inline void Subtract(
      const size_t cScores,
      const Bin & other,
      const GradientPair<TFloat, bClassification> * const aOtherGradientPairs,
      GradientPair<TFloat, bClassification> * const aThisGradientPairs
   ) {
      EBM_ASSERT(1 == cCompilerScores || cScores == cCompilerScores);
      EBM_ASSERT(cScores != cCompilerScores || aOtherGradientPairs == other.GetGradientPairs());
      EBM_ASSERT(cScores != cCompilerScores || aThisGradientPairs == GetGradientPairs());

      m_cSamples -= other.m_cSamples;
      m_weight -= other.m_weight;

      EBM_ASSERT(1 <= cScores);
      size_t iScore = 0;
      do {
         aThisGradientPairs[iScore] -= aOtherGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);
   }
   inline void Subtract(
      const size_t cScores,
      const Bin & other,
      const GradientPair<TFloat, bClassification> * const aOtherGradientPairs
   ) {
      Subtract(cScores, other, aOtherGradientPairs, GetGradientPairs());
   }
   inline void Subtract(const size_t cScores, const Bin & other) {
      Subtract(cScores, other, other.GetGradientPairs(), GetGradientPairs());
   }

   inline void Copy(
      const size_t cScores,
      const Bin & other,
      const GradientPair<TFloat, bClassification> * const aOtherGradientPairs,
      GradientPair<TFloat, bClassification> * const aThisGradientPairs
   ) {
      EBM_ASSERT(1 == cCompilerScores || cScores == cCompilerScores);
      EBM_ASSERT(cScores != cCompilerScores || aOtherGradientPairs == other.GetGradientPairs());
      EBM_ASSERT(cScores != cCompilerScores || aThisGradientPairs == GetGradientPairs());

      m_cSamples = other.m_cSamples;
      m_weight = other.m_weight;

      EBM_ASSERT(1 <= cScores);
      size_t iScore = 0;
      do {
         aThisGradientPairs[iScore] = aOtherGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);
   }
   inline void Copy(
      const size_t cScores,
      const Bin & other,
      const GradientPair<TFloat, bClassification> * const aOtherGradientPairs
   ) {
      Copy(cScores, other, aOtherGradientPairs, GetGradientPairs());
   }
   inline void Copy(const size_t cScores, const Bin & other) {
      Copy(cScores, other, other.GetGradientPairs(), GetGradientPairs());
   }

   inline void Zero(
      const size_t cScores,
      GradientPair<TFloat, bClassification> * const aThisGradientPairs
   ) {
      EBM_ASSERT(1 == cCompilerScores || cScores == cCompilerScores);
      EBM_ASSERT(cScores != cCompilerScores || aThisGradientPairs == GetGradientPairs());

      m_cSamples = 0;
      m_weight = 0;
      ZeroGradientPairs(aThisGradientPairs, cScores);
   }
   inline void Zero(const size_t cScores) {
      Zero(cScores, GetGradientPairs());
   }

   inline void AssertZero(
      const size_t cScores,
      const GradientPair<TFloat, bClassification> * const aThisGradientPairs
   ) const {
      UNUSED(cScores);
      UNUSED(aThisGradientPairs);
#ifndef NDEBUG
      EBM_ASSERT(1 == cCompilerScores || cScores == cCompilerScores);
      EBM_ASSERT(cScores != cCompilerScores || aThisGradientPairs == GetGradientPairs());

      EBM_ASSERT(0 == m_cSamples);
      EBM_ASSERT(0 == m_weight);

      EBM_ASSERT(1 <= cScores);
      size_t iScore = 0;
      do {
         aThisGradientPairs[iScore].AssertZero();
         ++iScore;
      } while(cScores != iScore);
#endif // NDEBUG
   }
   inline void AssertZero(const size_t cScores) const {
      AssertZero(cScores, GetGradientPairs());
   }
};
static_assert(std::is_standard_layout<Bin<float, true>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<Bin<float, true>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<Bin<float, true>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

static_assert(std::is_standard_layout<Bin<double, false>>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<Bin<double, false>>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<Bin<double, false>>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

template<typename TFloat>
inline static bool IsOverflowBinSize(const bool bClassification, const size_t cScores) {
   const size_t cBytesPerGradientPair = GetGradientPairSize<TFloat>(bClassification);

   if(UNLIKELY(IsMultiplyError(cBytesPerGradientPair, cScores))) {
      return true;
   }

   size_t cBytesBinComponent;
   if(bClassification) {
      typedef Bin<TFloat, true> OffsetType;
      cBytesBinComponent = offsetof(OffsetType, m_aGradientPairs);
   } else {
      typedef Bin<TFloat, false> OffsetType;
      cBytesBinComponent = offsetof(OffsetType, m_aGradientPairs);
   }

   if(UNLIKELY(IsAddError(cBytesBinComponent, cBytesPerGradientPair * cScores))) {
      return true;
   }

   return false;
}

template<typename TFloat>
inline static size_t GetBinSize(const bool bClassification, const size_t cScores) {
   // TODO: someday try out bin sizes that are a power of two.  This would allow us to use a shift when using bins
   //       instead of using multiplications.  In that version return the number of bits to shift here to make it easy
   //       to get either the shift required for indexing OR the number of bytes (shift 1 << num_bits)

   const size_t cBytesPerGradientPair = GetGradientPairSize<TFloat>(bClassification);

   size_t cBytesBinComponent;
   if(bClassification) {
      typedef Bin<TFloat, true> OffsetType;
      cBytesBinComponent = offsetof(OffsetType, m_aGradientPairs);
   } else {
      typedef Bin<TFloat, false> OffsetType;
      cBytesBinComponent = offsetof(OffsetType, m_aGradientPairs);
   }

   return cBytesBinComponent + cBytesPerGradientPair * cScores;
}

template<typename TFloat, bool bClassification, size_t cCompilerScores>
inline static Bin<TFloat, bClassification, cCompilerScores> * IndexBin(
   Bin<TFloat, bClassification, cCompilerScores> * const aBins,
   const size_t iByte
) {
   return IndexByte(aBins, iByte);
}

template<typename TFloat, bool bClassification, size_t cCompilerScores>
inline static const Bin<TFloat, bClassification, cCompilerScores> * IndexBin(
   const Bin<TFloat, bClassification, cCompilerScores> * const aBins,
   const size_t iByte
) {
   return IndexByte(aBins, iByte);
}

inline static BinBase * IndexBin(BinBase * const aBins, const size_t iByte) {
   return IndexByte(aBins, iByte);
}

inline static const BinBase * IndexBin(const BinBase * const aBins, const size_t iByte) {
   return IndexByte(aBins, iByte);
}

template<typename TFloat, bool bClassification, size_t cCompilerScores>
inline static const Bin<TFloat, bClassification, cCompilerScores> * NegativeIndexBin(
   const Bin<TFloat, bClassification, cCompilerScores> * const aBins,
   const size_t iByte
) {
   return NegativeIndexByte(aBins, iByte);
}

template<typename TFloat, bool bClassification, size_t cCompilerScores>
inline static Bin<TFloat, bClassification, cCompilerScores> * NegativeIndexBin(
   Bin<TFloat, bClassification, cCompilerScores> * const aBins,
   const size_t iByte
) {
   return NegativeIndexByte(aBins, iByte);
}

template<typename TFloat, bool bClassification, size_t cCompilerScores>
inline static size_t CountBins(
   const Bin<TFloat, bClassification, cCompilerScores> * const pBinHigh,
   const Bin<TFloat, bClassification, cCompilerScores> * const pBinLow,
   const size_t cBytesPerBin
) {
   const size_t cBytesDiff = CountBytes(pBinHigh, pBinLow);
   EBM_ASSERT(0 == cBytesDiff % cBytesPerBin);
   return cBytesDiff / cBytesPerBin;
}


// keep this as a MACRO so that we don't materialize any of the parameters on non-debug builds
#define ASSERT_BIN_OK(MACRO_cBytesPerBin, MACRO_pBin, MACRO_pBinsEnd) \
   (EBM_ASSERT(reinterpret_cast<const BinBase *>(reinterpret_cast<const char *>(MACRO_pBin) + \
      static_cast<size_t>(MACRO_cBytesPerBin)) <= (MACRO_pBinsEnd)))

} // DEFINED_ZONE_NAME

#endif // BIN_HPP
