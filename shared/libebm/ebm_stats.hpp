// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef EBM_STATS_HPP
#define EBM_STATS_HPP

#include <cmath>

#include "logging.h" // EBM_ASSERT
#include "unzoned.h" // INLINE_ALWAYS, LIKELY, UNLIKELY

#include "ebm_internal.hpp" // FloatCalc

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

static_assert(std::numeric_limits<double>::is_iec559,
      "If IEEE 754 isn't guaranteed, then we can't use or compare infinity values in any standard way. "
      "We can't even guarantee that infinity exists as a concept.");

static constexpr FloatCalc k_gainMin = 0;

// - most compilers have flags that give unspecified behavior when either NaN or +-infinity are generated,
//   or propagated, or in conditional statements involving them.  Our code can't tolerate these switches being
//   enabled since we can sometimes generate these values, and in general we need to accept NaN values from our caller.
// - g++ makes std::isnan fail to work when the compiler flags indicate no NaN values are present, so we don't use
//   "-ffast-math".  -fno-finite-math-only would solve this, but we still want NaN comparisons to work outside of
//   just std::isnan for NaN propagation.
// - given our need to allow NaN and +-infinity values, we can then also use NaN and infinity propagation to avoid
//   checking individual samples for overflows, and we propagate the overflows to the final result which we check.
//   We also use some of the odd properties of NaN comparison that rely on a high degree of compliance with IEEE-754
// - most compilers have switches that allow the compiler to re-order math operations.  We've found issues with
//   these switches, so our code needs to be compiled with strict adherence to floating point operations.  Once
//   we found a bug where the compiler was optimizing "log(x) + log(y)" into "log(x * y)", which then overflowed.
//
// IEEE 754 special value operations:
//   denormal values don't get rounded to zero
//   NaN values get propagated in operations
//   anything compared to a NaN, is false, except for "a != a" (where a is NaN), which is true!
//        Take note that "a == a" is FALSE for NaN values!!!
//   log(-ANYTHING) = NaN
//   +infinity + -infinity = NaN
//   -infinity + +infinity = NaN
//   -infinity - -infinity = NaN
//   +infinity - +infinity = NaN
//   +-infinity * +-0 = NaN
//   +-0 * +-infinity = NaN
//   +-0 / +-0 = NaN
//   +-infinity / +-infinity = NaN
//   +infinity + +infinity = +infinity
//   +infinity - -infinity = +infinity
//   -infinity + -infinity = -infinity
//   -infinity - +infinity = -infinity
//   +-infinity * +-infinity = +-infinity
//   +-FINITE * +-infinity = +-infinity
//   +-NON_ZERO / +-0 = +-infinity
//   +-FINITE / +-infinity = +-0
//
// - useful resources:
//   - comparing floats -> https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
//   - details on float representations -> https://www.volkerschatz.com/science/float.html
//   - things that are guaranteed and not -> https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html

INLINE_ALWAYS static FloatCalc ApplyL1(const FloatCalc sumGradient, const FloatCalc regAlpha) {
   EBM_ASSERT(0 <= regAlpha);
   FloatCalc regularizedSumGradient = std::fabs(sumGradient) - regAlpha;
   regularizedSumGradient = UNLIKELY(regularizedSumGradient < FloatCalc{0}) ? FloatCalc{0} : regularizedSumGradient;
   regularizedSumGradient =
         UNPREDICTABLE(sumGradient < FloatCalc{0}) ? -regularizedSumGradient : regularizedSumGradient;
   return regularizedSumGradient;
}

INLINE_ALWAYS static FloatCalc ApplyL2(const FloatCalc sumHessian, const FloatCalc regLambda) {
   EBM_ASSERT(0 <= regLambda);
   return sumHessian + regLambda;
}

INLINE_ALWAYS static FloatCalc CalcGradientUpdate(const FloatCalc sumGradient) {
   // for differentially private EBMs we return the gradients. It is not really the update.
   return sumGradient;
}

template<bool bCheckHessian>
INLINE_ALWAYS static FloatCalc CalcNegUpdate(const FloatCalc sumGradient,
      const FloatCalc sumHessian,
      const FloatCalc regAlpha,
      const FloatCalc regLambda,
      const FloatCalc deltaStepMax) {
   // a loss function with negative hessians would be unstable
   EBM_ASSERT(std::isnan(sumHessian) || FloatCalc{0} <= sumHessian);
   EBM_ASSERT(FloatCalc{0} < deltaStepMax);

   if(bCheckHessian) {
      if(UNLIKELY(sumHessian < std::numeric_limits<FloatCalc>::min())) {
         return FloatCalc{0};
      }
   }
   FloatCalc ret = ApplyL1(sumGradient, regAlpha) / ApplyL2(sumHessian, regLambda);
   if(UNLIKELY(std::fabs(ret) > deltaStepMax)) {
      ret = UNPREDICTABLE(ret < FloatCalc{0}) ? -deltaStepMax : deltaStepMax;
   }
   return ret;
}

INLINE_ALWAYS static FloatCalc CalcPartialGainFromUpdate(const FloatCalc sumGradient,
      const FloatCalc sumHessian,
      const FloatCalc negUpdate,
      const FloatCalc regAlpha,
      const FloatCalc regLambda) {
   // a loss function with negative hessians would be unstable
   EBM_ASSERT(std::isnan(sumHessian) || FloatCalc{0} <= sumHessian);

   const FloatCalc partialGain =
         negUpdate * (ApplyL1(sumGradient, regAlpha) * FloatCalc{2} - negUpdate * ApplyL2(sumHessian, regLambda));

   return partialGain;
}

INLINE_ALWAYS static FloatCalc CalcPartialGain(const FloatCalc sumGradient,
      const FloatCalc sumHessian,
      const FloatCalc regAlpha,
      const FloatCalc regLambda,
      const FloatCalc deltaStepMax) {
   // This gain function used to determine splits is equivalent to minimizing sum of squared error SSE, which
   // can be seen following the derivation of Equation #7 in Ping Li's paper -> https://arxiv.org/pdf/1203.3491.pdf

   // a loss function with negative hessians would be unstable
   EBM_ASSERT(std::isnan(sumHessian) || FloatCalc{0} <= sumHessian);
   EBM_ASSERT(FloatCalc{0} < deltaStepMax);

   FloatCalc partialGain;
   if(UNLIKELY(std::numeric_limits<FloatCalc>::infinity() != deltaStepMax)) {
      const FloatCalc negUpdate = CalcNegUpdate<false>(sumGradient, sumHessian, regAlpha, regLambda, deltaStepMax);
      partialGain = CalcPartialGainFromUpdate(sumGradient, sumHessian, negUpdate, regAlpha, regLambda);
   } else {
      const FloatCalc regularizedSumGradient = ApplyL1(sumGradient, regAlpha);
      partialGain = regularizedSumGradient / ApplyL2(sumHessian, regLambda) * regularizedSumGradient;

      EBM_ASSERT(std::isnan(partialGain) ||
            IsApproxEqual(partialGain,
                  CalcPartialGainFromUpdate(sumGradient,
                        sumHessian,
                        CalcNegUpdate<false>(sumGradient, sumHessian, regAlpha, regLambda, deltaStepMax),
                        regAlpha,
                        regLambda)));
   }
   // This function does not allow flipping the sign of the update, unlike
   // CalcPartialGainFromUpdate, so it should be impossible to get a negative partial gain.
   EBM_ASSERT(std::isnan(partialGain) || FloatCalc{0} <= partialGain);

   return partialGain;
}

} // namespace DEFINED_ZONE_NAME

#endif // EBM_STATS_HPP