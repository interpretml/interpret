// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef BIN_HPP
#define BIN_HPP

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t
#include <cmath> // abs
#include <string.h> // memcpy

#include "logging.h" // EBM_ASSERT
#include "unzoned.h" // UNUSED

#include "common.hpp"
#include "GradientPair.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<typename TFloat, typename TUInt, bool bHessian, size_t cCompilerScores = 1> struct Bin;

struct BinBase {
   BinBase() = default; // preserve our POD status
   ~BinBase() = default; // preserve our POD status
   void* operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete(void*) = delete; // we only use malloc/free in this library

   template<typename TFloat, typename TUInt, bool bHessian, size_t cCompilerScores = 1>
   GPU_BOTH inline Bin<TFloat, TUInt, bHessian, cCompilerScores>* Specialize() {
      return static_cast<Bin<TFloat, TUInt, bHessian, cCompilerScores>*>(this);
   }
   template<typename TFloat, typename TUInt, bool bHessian, size_t cCompilerScores = 1>
   GPU_BOTH inline const Bin<TFloat, TUInt, bHessian, cCompilerScores>* Specialize() const {
      return static_cast<const Bin<TFloat, TUInt, bHessian, cCompilerScores>*>(this);
   }

   GPU_BOTH inline void ZeroMem(const size_t cBytesPerBin, const size_t cBins = 1, const size_t iBin = 0) {
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
static_assert(
      std::is_trivial<BinBase>::value, "We use memcpy in several places, so disallow non-trivial types in general");

template<typename TFloat, typename TUInt> static bool IsOverflowBinSize(const bool bHessian, const size_t cScores);
template<typename TFloat, typename TUInt>
GPU_BOTH inline constexpr static size_t GetBinSize(const bool bHessian, const size_t cScores);

template<typename TFloat, typename TUInt, bool bHessian, size_t cCompilerScores> struct Bin final : BinBase {
   // TODO: Use the type std::nullptr_t for TUInt to indicate that the m_cSamples field should be dropped
   //       and add a bool bWeight template parameter to indicate if weight should be kept

   friend void ConvertAddBin(const size_t,
         const bool,
         const size_t,
         const bool,
         const bool,
         const void* const,
         const uint64_t* const,
         const double* const,
         const bool,
         const bool,
         void* const);
   template<typename, typename> friend bool IsOverflowBinSize(const bool, const size_t);
   template<typename, typename> GPU_BOTH friend inline constexpr size_t GetBinSize(const bool, const size_t);

   static_assert(std::is_floating_point<TFloat>::value, "TFloat must be a float type");
   static_assert(std::is_integral<TUInt>::value, "TUInt must be an integer type");
   static_assert(std::is_unsigned<TUInt>::value, "TUInt must be unsigned");

 private:
   TUInt m_cSamples;
   TFloat m_weight;

   // IMPORTANT: m_aGradientPairs must be in the last position for the struct hack and this must be standard layout
   GradientPair<TFloat, bHessian> m_aGradientPairs[cCompilerScores];

 public:
   Bin() = default; // preserve our POD status
   ~Bin() = default; // preserve our POD status
   void* operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete(void*) = delete; // we only use malloc/free in this library

   GPU_BOTH inline TUInt GetCountSamples() const { return m_cSamples; }
   GPU_BOTH inline void SetCountSamples(const TUInt cSamples) { m_cSamples = cSamples; }

   GPU_BOTH inline TFloat GetWeight() const { return m_weight; }
   GPU_BOTH inline void SetWeight(const TFloat weight) { m_weight = weight; }

   GPU_BOTH inline const GradientPair<TFloat, bHessian>* GetGradientPairs() const {
      return ArrayToPointer(m_aGradientPairs);
   }
   GPU_BOTH inline GradientPair<TFloat, bHessian>* GetGradientPairs() { return ArrayToPointer(m_aGradientPairs); }

   GPU_BOTH inline const Bin<TFloat, TUInt, bHessian, 1>* Downgrade() const {
      return reinterpret_cast<const Bin<TFloat, TUInt, bHessian, 1>*>(this);
   }
   GPU_BOTH inline Bin<TFloat, TUInt, bHessian, 1>* Downgrade() {
      return reinterpret_cast<Bin<TFloat, TUInt, bHessian, 1>*>(this);
   }

   GPU_BOTH inline void Add(const size_t cScores,
         const Bin& other,
         const GradientPair<TFloat, bHessian>* const aOtherGradientPairs,
         GradientPair<TFloat, bHessian>* const aThisGradientPairs) {
#ifndef GPU_COMPILE
      EBM_ASSERT(1 == cCompilerScores || cScores == cCompilerScores);
      EBM_ASSERT(cScores != cCompilerScores || aOtherGradientPairs == other.GetGradientPairs());
      EBM_ASSERT(cScores != cCompilerScores || aThisGradientPairs == GetGradientPairs());
      EBM_ASSERT(1 <= cScores);
#endif // GPU_COMPILE
      m_cSamples += other.m_cSamples;
      m_weight += other.m_weight;

      size_t iScore = 0;
      do {
         aThisGradientPairs[iScore] += aOtherGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);
   }
   GPU_BOTH inline void Add(
         const size_t cScores, const Bin& other, const GradientPair<TFloat, bHessian>* const aOtherGradientPairs) {
      Add(cScores, other, aOtherGradientPairs, GetGradientPairs());
   }
   GPU_BOTH inline void Add(const size_t cScores, const Bin& other) {
      Add(cScores, other, other.GetGradientPairs(), GetGradientPairs());
   }

   GPU_BOTH inline void Subtract(const size_t cScores,
         const Bin& other,
         const GradientPair<TFloat, bHessian>* const aOtherGradientPairs,
         GradientPair<TFloat, bHessian>* const aThisGradientPairs) {
#ifndef GPU_COMPILE
      EBM_ASSERT(1 == cCompilerScores || cScores == cCompilerScores);
      EBM_ASSERT(cScores != cCompilerScores || aOtherGradientPairs == other.GetGradientPairs());
      EBM_ASSERT(cScores != cCompilerScores || aThisGradientPairs == GetGradientPairs());
      EBM_ASSERT(1 <= cScores);
#endif // GPU_COMPILE
      m_cSamples -= other.m_cSamples;
      m_weight -= other.m_weight;

      size_t iScore = 0;
      do {
         aThisGradientPairs[iScore] -= aOtherGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);
   }
   GPU_BOTH inline void Subtract(
         const size_t cScores, const Bin& other, const GradientPair<TFloat, bHessian>* const aOtherGradientPairs) {
      Subtract(cScores, other, aOtherGradientPairs, GetGradientPairs());
   }
   GPU_BOTH inline void Subtract(const size_t cScores, const Bin& other) {
      Subtract(cScores, other, other.GetGradientPairs(), GetGradientPairs());
   }

   GPU_BOTH inline void Copy(const size_t cScores,
         const Bin& other,
         const GradientPair<TFloat, bHessian>* const aOtherGradientPairs,
         GradientPair<TFloat, bHessian>* const aThisGradientPairs) {
#ifndef GPU_COMPILE
      EBM_ASSERT(1 == cCompilerScores || cScores == cCompilerScores);
      EBM_ASSERT(cScores != cCompilerScores || aOtherGradientPairs == other.GetGradientPairs());
      EBM_ASSERT(cScores != cCompilerScores || aThisGradientPairs == GetGradientPairs());
      EBM_ASSERT(1 <= cScores);
#endif // GPU_COMPILE

      m_cSamples = other.m_cSamples;
      m_weight = other.m_weight;

      size_t iScore = 0;
      do {
         aThisGradientPairs[iScore] = aOtherGradientPairs[iScore];
         ++iScore;
      } while(cScores != iScore);
   }
   GPU_BOTH inline void Copy(
         const size_t cScores, const Bin& other, const GradientPair<TFloat, bHessian>* const aOtherGradientPairs) {
      Copy(cScores, other, aOtherGradientPairs, GetGradientPairs());
   }
   GPU_BOTH inline void Copy(const size_t cScores, const Bin& other) {
      Copy(cScores, other, other.GetGradientPairs(), GetGradientPairs());
   }

   GPU_BOTH inline void Zero(const size_t cScores, GradientPair<TFloat, bHessian>* const aThisGradientPairs) {
#ifndef GPU_COMPILE
      EBM_ASSERT(1 == cCompilerScores || cScores == cCompilerScores);
      EBM_ASSERT(cScores != cCompilerScores || aThisGradientPairs == GetGradientPairs());
#endif // GPU_COMPILE

      m_cSamples = 0;
      m_weight = 0;
      ZeroGradientPairs(aThisGradientPairs, cScores);
   }
   GPU_BOTH inline void Zero(const size_t cScores) { Zero(cScores, GetGradientPairs()); }

   GPU_BOTH inline void AssertZero(
         const size_t cScores, const GradientPair<TFloat, bHessian>* const aThisGradientPairs) const {
      UNUSED(cScores);
      UNUSED(aThisGradientPairs);
#ifndef GPU_COMPILE
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
#endif // GPU_COMPILE
   }
   GPU_BOTH inline void AssertZero(const size_t cScores) const { AssertZero(cScores, GetGradientPairs()); }
};
static_assert(std::is_standard_layout<Bin<float, uint32_t, true>>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<Bin<float, uint32_t, true>>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");

static_assert(std::is_standard_layout<Bin<double, uint64_t, false>>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<Bin<double, uint64_t, false>>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");

template<typename TFloat, typename TUInt>
inline static bool IsOverflowBinSize(const bool bHessian, const size_t cScores) {
   const size_t cBytesPerGradientPair = GetGradientPairSize<TFloat>(bHessian);

   if(UNLIKELY(IsMultiplyError(cBytesPerGradientPair, cScores))) {
      return true;
   }

   size_t cBytesBinComponent;
   if(bHessian) {
      typedef Bin<TFloat, TUInt, true> OffsetType;
      cBytesBinComponent = offsetof(OffsetType, m_aGradientPairs);
   } else {
      typedef Bin<TFloat, TUInt, false> OffsetType;
      cBytesBinComponent = offsetof(OffsetType, m_aGradientPairs);
   }

   if(UNLIKELY(IsAddError(cBytesBinComponent, cBytesPerGradientPair * cScores))) {
      return true;
   }

   return false;
}

template<typename TFloat, typename TUInt>
GPU_BOTH inline constexpr static size_t GetBinSize(const bool bHessian, const size_t cScores) {
   typedef Bin<TFloat, TUInt, true> OffsetTypeHt;
   typedef Bin<TFloat, TUInt, false> OffsetTypeHf;

   // TODO: someday try out bin sizes that are a power of two.  This would allow us to use a shift when using bins
   //       instead of using multiplications.  In that version return the number of bits to shift here to make it easy
   //       to get either the shift required for indexing OR the number of bytes (shift 1 << num_bits)

   return (bHessian ? offsetof(OffsetTypeHt, m_aGradientPairs) : offsetof(OffsetTypeHf, m_aGradientPairs)) +
         GetGradientPairSize<TFloat>(bHessian) * cScores;
}

template<typename TFloat, typename TUInt, bool bHessian, size_t cCompilerScores>
GPU_BOTH inline static Bin<TFloat, TUInt, bHessian, cCompilerScores>* IndexBin(
      Bin<TFloat, TUInt, bHessian, cCompilerScores>* const aBins, const size_t iByte) {
   return IndexByte(aBins, iByte);
}

template<typename TFloat, typename TUInt, bool bHessian, size_t cCompilerScores>
GPU_BOTH inline static const Bin<TFloat, TUInt, bHessian, cCompilerScores>* IndexBin(
      const Bin<TFloat, TUInt, bHessian, cCompilerScores>* const aBins, const size_t iByte) {
   return IndexByte(aBins, iByte);
}

GPU_BOTH inline static BinBase* IndexBin(BinBase* const aBins, const size_t iByte) { return IndexByte(aBins, iByte); }

GPU_BOTH inline static const BinBase* IndexBin(const BinBase* const aBins, const size_t iByte) {
   return IndexByte(aBins, iByte);
}

template<typename TFloat, typename TUInt, bool bHessian, size_t cCompilerScores>
GPU_BOTH inline static const Bin<TFloat, TUInt, bHessian, cCompilerScores>* NegativeIndexBin(
      const Bin<TFloat, TUInt, bHessian, cCompilerScores>* const aBins, const size_t iByte) {
   return NegativeIndexByte(aBins, iByte);
}

template<typename TFloat, typename TUInt, bool bHessian, size_t cCompilerScores>
GPU_BOTH inline static Bin<TFloat, TUInt, bHessian, cCompilerScores>* NegativeIndexBin(
      Bin<TFloat, TUInt, bHessian, cCompilerScores>* const aBins, const size_t iByte) {
   return NegativeIndexByte(aBins, iByte);
}

template<typename TFloat, typename TUInt, bool bHessian, size_t cCompilerScores>
inline static size_t CountBins(const Bin<TFloat, TUInt, bHessian, cCompilerScores>* const pBinHigh,
      const Bin<TFloat, TUInt, bHessian, cCompilerScores>* const pBinLow,
      const size_t cBytesPerBin) {
   const size_t cBytesDiff = CountBytes(pBinHigh, pBinLow);
#ifndef GPU_COMPILE
   EBM_ASSERT(0 == cBytesDiff % cBytesPerBin);
#endif // GPU_COMPILE
   return cBytesDiff / cBytesPerBin;
}

// keep this as a MACRO so that we don't materialize any of the parameters on non-debug builds
#define ASSERT_BIN_OK(MACRO_cBytesPerBin, MACRO_pBin, MACRO_pBinsEnd)                                                  \
   (EBM_ASSERT(reinterpret_cast<const BinBase*>(reinterpret_cast<const char*>(MACRO_pBin) +                            \
                     static_cast<size_t>(MACRO_cBytesPerBin)) <= (MACRO_pBinsEnd)))

} // namespace DEFINED_ZONE_NAME

#endif // BIN_HPP
