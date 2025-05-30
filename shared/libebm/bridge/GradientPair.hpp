// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef GRADIENT_PAIR_HPP
#define GRADIENT_PAIR_HPP

#include <type_traits> // std::is_standard_layout

#include "logging.h" // EBM_ASSERT
#include "unzoned.h" // UNUSED

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<typename TFloat, bool bHessian> struct GradientPair;

template<typename TFloat> struct GradientPair<TFloat, true> final {
   // classification version of the GradientPair class

#ifndef __SUNPRO_CC

   // the Oracle Developer Studio compiler has what I think is a bug by making any class that includes
   // GradientPair fields turn into non-trivial classes, so exclude the Oracle compiler
   // from these protections

   GradientPair() = default; // preserve our POD status
   ~GradientPair() = default; // preserve our POD status
   void* operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete(void*) = delete; // we only use malloc/free in this library

#endif // __SUNPRO_CC

   TFloat m_sumGradients;
   // TODO: for single features, we probably want to just do a single pass of the data and collect our m_sumHessians
   // during that sweep.  This is probably
   //   also true for pairs since calculating pair sums can be done fairly efficiently, but for tripples and higher
   //   dimensions we might be better off calculating JUST the m_sumGradients which is the only thing required for
   //   choosing splits and we could then do a second pass of the data to find the hessians once we know the splits.
   //   Tripples and higher dimensions tend to re-add/subtract the same cells many times over which is why it might be
   //   better there.  Test these theories out on large datasets
   TFloat m_sumHessians;

   // If we end up adding a 3rd derivative here, call it ThirdDerivative.  I like how Gradient and Hessian separate
   // nicely from eachother and match other package naming.  ThirdDerivative is nice since it's distinctly named
   // and easy to see rather than Derivative1, Derivative2, Derivative3, etc..

   GPU_BOTH inline TFloat GetHess() const { return m_sumHessians; }
   GPU_BOTH inline void SetHess(const TFloat sumHessians) { m_sumHessians = sumHessians; }
   GPU_BOTH inline GradientPair& operator+=(const GradientPair& other) {
      m_sumGradients += other.m_sumGradients;
      m_sumHessians += other.m_sumHessians;
      return *this;
   }
   GPU_BOTH inline GradientPair& operator-=(const GradientPair& other) {
      m_sumGradients -= other.m_sumGradients;
      m_sumHessians -= other.m_sumHessians;
      return *this;
   }
   GPU_BOTH inline bool IsGradientsClose(const GradientPair& other) const {
      return IsClose(m_sumGradients, other.m_sumGradients) && IsClose(m_sumHessians, other.m_sumHessians);
   }
   GPU_BOTH inline void Zero() {
      m_sumGradients = 0;
      m_sumHessians = 0;
   }
   GPU_BOTH inline void AssertZero() const {
#ifndef GPU_COMPILE
      EBM_ASSERT(0 == m_sumGradients);
      EBM_ASSERT(0 == m_sumHessians);
#endif // GPU_COMPILE
   }
};
static_assert(std::is_standard_layout<GradientPair<double, true>>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<GradientPair<double, true>>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");

static_assert(std::is_standard_layout<GradientPair<float, true>>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<GradientPair<float, true>>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");

template<typename TFloat> struct GradientPair<TFloat, false> final {
   // regression version of the GradientPair class

#ifndef __SUNPRO_CC

   // the Oracle Developer Studio compiler has what I think is a bug by making any class that includes
   // GradientPair fields turn into non-trivial classes, so exclude the Oracle compiler
   // from these protections

   GradientPair() = default; // preserve our POD status
   ~GradientPair() = default; // preserve our POD status
   void* operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete(void*) = delete; // we only use malloc/free in this library

#endif // __SUNPRO_CC

   TFloat m_sumGradients;

   GPU_BOTH inline TFloat GetHess() const {
#ifndef GPU_COMPILE
      EBM_ASSERT(false); // this should never be called, but the compiler seems to want it to exist
#endif // GPU_COMPILE
      return TFloat{0};
   }
   GPU_BOTH inline void SetHess(const TFloat sumHessians) {
      UNUSED(sumHessians);
#ifndef GPU_COMPILE
      EBM_ASSERT(false); // this should never be called, but the compiler seems to want it to exist
#endif // GPU_COMPILE
   }
   GPU_BOTH inline GradientPair& operator+=(const GradientPair& other) {
      m_sumGradients += other.m_sumGradients;
      return *this;
   }
   GPU_BOTH inline GradientPair& operator-=(const GradientPair& other) {
      m_sumGradients -= other.m_sumGradients;
      return *this;
   }
   GPU_BOTH inline bool IsGradientsClose(const GradientPair& other) const {
      return IsClose(m_sumGradients, other.m_sumGradients);
   }
   GPU_BOTH inline void Zero() { m_sumGradients = 0; }
   GPU_BOTH inline void AssertZero() const {
#ifndef GPU_COMPILE
      EBM_ASSERT(0 == m_sumGradients);
#endif // GPU_COMPILE
   }
};
static_assert(std::is_standard_layout<GradientPair<double, false>>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<GradientPair<double, false>>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");

static_assert(std::is_standard_layout<GradientPair<float, false>>::value,
      "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<GradientPair<float, false>>::value,
      "We use memcpy in several places, so disallow non-trivial types in general");

template<typename TFloat, bool bHessian>
GPU_BOTH inline static void ZeroGradientPairs(
      GradientPair<TFloat, bHessian>* const aGradientPairs, const size_t cScores) {
#ifndef GPU_COMPILE
   EBM_ASSERT(1 <= cScores);
#endif // GPU_COMPILE
   size_t iScore = 0;
   do {
      aGradientPairs[iScore].Zero();
      ++iScore;
   } while(cScores != iScore);
}

template<typename TFloat> GPU_BOTH inline constexpr static size_t GetGradientPairSize(const bool bHessian) {
   return bHessian ? sizeof(GradientPair<TFloat, true>) : sizeof(GradientPair<TFloat, false>);
}

} // namespace DEFINED_ZONE_NAME

#endif // GRADIENT_PAIR_HPP
