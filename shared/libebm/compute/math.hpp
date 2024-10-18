// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef MATH_HPP
#define MATH_HPP

#include "libebm.h"
#include "logging.h" // EBM_ASSERT
#include "unzoned.h" // INLINE_ALWAYS

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<typename TFloat> static INLINE_ALWAYS TFloat Mantissa32(const TFloat& val) noexcept {
   return TFloat::ReinterpretFloat(
         (TFloat::ReinterpretInt(val) & typename TFloat::TInt{0x007FFFFF}) | typename TFloat::TInt{0x3F000000});
}

template<typename TFloat> static INLINE_ALWAYS typename TFloat::TInt Exponent32(const TFloat& val) noexcept {
   return ((TFloat::ReinterpretInt(val) << 1) >> 24) - typename TFloat::TInt{0x7F};
}

template<typename TFloat> static INLINE_ALWAYS TFloat Mantissa64(const TFloat& val) noexcept {
   return TFloat::ReinterpretFloat((TFloat::ReinterpretInt(val) & typename TFloat::TInt{0x000FFFFFFFFFFFFFll}) |
         typename TFloat::TInt{0x3FE0000000000000ll});
}

template<typename TFloat> static INLINE_ALWAYS TFloat Exponent64(const TFloat& val) noexcept {
   return TFloat::ReinterpretFloat(
                ((TFloat::ReinterpretInt(val) >> 52) | TFloat::ReinterpretInt(TFloat{4503599627370496.0}))) -
         TFloat{4503599627370496.0 + 1023.0};
}

template<typename TFloat> static INLINE_ALWAYS TFloat Power32(const TFloat val) {
   return TFloat::ReinterpretFloat(TFloat::ReinterpretInt(val + TFloat{8388608 + 127}) << 23);
}

template<typename TFloat> static INLINE_ALWAYS TFloat Power64(const TFloat val) {
   return TFloat::ReinterpretFloat(
         TFloat::ReinterpretInt(val + TFloat{int64_t{4503599627370496} + int64_t{1023}}) << 52);
}

template<typename TFloat>
static INLINE_ALWAYS TFloat Polynomial32(const TFloat x,
      const TFloat c0,
      const TFloat c1,
      const TFloat c2,
      const TFloat c3,
      const TFloat c4,
      const TFloat c5) {
   TFloat x2 = x * x;
   TFloat x4 = x2 * x2;
   return FusedMultiplyAdd(FusedMultiplyAdd(c3, x, c2),
         x2,
         FusedMultiplyAdd(FusedMultiplyAdd(c5, x, c4), x4, FusedMultiplyAdd(c1, x, c0)));
}

template<typename TFloat>
static INLINE_ALWAYS TFloat Polynomial32(const TFloat x,
      const TFloat c0,
      const TFloat c1,
      const TFloat c2,
      const TFloat c3,
      const TFloat c4,
      const TFloat c5,
      const TFloat c6,
      const TFloat c7,
      const TFloat c8) {
   TFloat x2 = x * x;
   TFloat x4 = x2 * x2;
   TFloat x8 = x4 * x4;
   return FusedMultiplyAdd(FusedMultiplyAdd(FusedMultiplyAdd(c7, x, c6), x2, FusedMultiplyAdd(c5, x, c4)),
         x4,
         FusedMultiplyAdd(FusedMultiplyAdd(c3, x, c2), x2, FusedMultiplyAdd(c1, x, c0) + c8 * x8));
}

template<typename TFloat>
static INLINE_ALWAYS TFloat Polynomial64(const TFloat x,
      const TFloat c2,
      const TFloat c3,
      const TFloat c4,
      const TFloat c5,
      const TFloat c6,
      const TFloat c7,
      const TFloat c8,
      const TFloat c9,
      const TFloat c10,
      const TFloat c11,
      const TFloat c12,
      const TFloat c13) {
   TFloat x2 = x * x;
   TFloat x4 = x2 * x2;
   TFloat x8 = x4 * x4;
   return FusedMultiplyAdd(FusedMultiplyAdd(FusedMultiplyAdd(c13, x, c12),
                                 x4,
                                 FusedMultiplyAdd(FusedMultiplyAdd(c11, x, c10), x2, FusedMultiplyAdd(c9, x, c8))),
         x8,
         FusedMultiplyAdd(FusedMultiplyAdd(FusedMultiplyAdd(c7, x, c6), x2, FusedMultiplyAdd(c5, x, c4)),
               x4,
               FusedMultiplyAdd(FusedMultiplyAdd(c3, x, c2), x2, x)));
}

template<typename TFloat>
static INLINE_ALWAYS TFloat Polynomial64(const TFloat x,
      const TFloat c0,
      const TFloat c1,
      const TFloat c2,
      const TFloat c3,
      const TFloat c4,
      const TFloat c5) {
   TFloat x2 = x * x;
   TFloat x4 = x2 * x2;
   return FusedMultiplyAdd(FusedMultiplyAdd(c3, x, c2),
         x2,
         FusedMultiplyAdd(FusedMultiplyAdd(c5, x, c4), x4, FusedMultiplyAdd(c1, x, c0)));
}

template<typename TFloat>
static INLINE_ALWAYS TFloat Polynomial64(
      const TFloat x, const TFloat c0, const TFloat c1, const TFloat c2, const TFloat c3, const TFloat c4) {
   TFloat x2 = x * x;
   TFloat x4 = x2 * x2;
   return FusedMultiplyAdd(FusedMultiplyAdd(c3, x, c2), x2, FusedMultiplyAdd(c4 + x, x4, FusedMultiplyAdd(c1, x, c0)));
}

template<typename TFloat,
      bool bNegateInput = false,
      bool bNaNPossible = true,
      bool bUnderflowPossible = true,
      bool bOverflowPossible = true>
static INLINE_ALWAYS TFloat Exp32(const TFloat val) {
   // algorithm comes from:
   // https://github.com/vectorclass/version2/blob/f4617df57e17efcd754f5bbe0ec87883e0ed9ce6/vectormath_exp.h#L501

   // k_expUnderflow is set to a value that prevents us from returning a denormal number.
   static constexpr float k_expUnderflow = -87.25f; // this is exactly representable in IEEE 754
   static constexpr float k_expOverflow = 87.25f; // this is exactly representable in IEEE 754

   TFloat rounded;
   TFloat x;
   if (bNegateInput) {
      rounded = Round(val * TFloat{-1.44269504088896340736f});
      x = FusedMultiplySubtract(rounded, TFloat{-0.693359375f}, val);
   } else {
      rounded = Round(val * TFloat{1.44269504088896340736f});
      x = FusedMultiplyAdd(rounded, TFloat{-0.693359375f}, val);
   }
   x = FusedNegateMultiplyAdd(rounded, TFloat{-2.12194440e-4f}, x);

   const TFloat x2 = x * x;
   TFloat ret = Polynomial32(x,
         TFloat{1} / TFloat{2},
         TFloat{1} / TFloat{6},
         TFloat{1} / TFloat{24},
         TFloat{1} / TFloat{120},
         TFloat{1} / TFloat{720},
         TFloat{1} / TFloat{5040});
   ret = FusedMultiplyAdd(ret, x2, x);

   const TFloat rounded2 = Power32(rounded);

   ret = (ret + TFloat{1}) * rounded2;

   if(bOverflowPossible) {
      if(bNegateInput) {
         ret = IfThenElse(val < TFloat{-k_expOverflow},
               std::numeric_limits<typename TFloat::T>::infinity(),
               ret);
      } else {
         ret = IfThenElse(TFloat{k_expOverflow} < val,
               std::numeric_limits<typename TFloat::T>::infinity(),
               ret);
      }
   }
   if(bUnderflowPossible) {
      if(bNegateInput) {
         ret = IfThenElse(TFloat{-k_expUnderflow} < val, TFloat{0}, ret);
      } else {
         ret = IfThenElse(val < TFloat{k_expUnderflow}, TFloat{0}, ret);
      }
   }
   if(bNaNPossible) {
      ret = IfThenElse(IsNaN(val), val, ret);
   }

#ifndef NDEBUG
   TFloat::Execute(
         [](int, typename TFloat::T orig, typename TFloat::T ret) {
            EBM_ASSERT(IsApproxEqual(std::exp(orig), ret, typename TFloat::T{1e-6}));
         },
         bNegateInput ? -val : val,
         ret);
#endif // NDEBUG

   return ret;
}

template<typename TFloat,
      bool bNegateOutput = false,
      bool bNaNPossible = true,
      bool bNegativePossible = true,
      bool bZeroPossible = true,
      bool bPositiveInfinityPossible = true>
static INLINE_ALWAYS TFloat Log32(const TFloat& val) noexcept {
   // algorithm comes from:
   // https://github.com/vectorclass/version2/blob/f4617df57e17efcd754f5bbe0ec87883e0ed9ce6/vectormath_exp.h#L1147

   TFloat x = Mantissa32(val);
   typename TFloat::TInt exponent = Exponent32(val);

   const auto comparison = x <= TFloat{float{1.41421356237309504880} * 0.5f};
   x = IfAdd(comparison, x, x);
   exponent = IfAdd(~comparison, exponent, typename TFloat::TInt{1});

   TFloat exponentFloat = TFloat(exponent);

   x += TFloat{-1};

   TFloat ret = Polynomial32(x,
         TFloat{3.3333331174E-1f},
         TFloat{-2.4999993993E-1f},
         TFloat{2.0000714765E-1f},
         TFloat{-1.6668057665E-1f},
         TFloat{1.4249322787E-1f},
         TFloat{-1.2420140846E-1f},
         TFloat{1.1676998740E-1f},
         TFloat{-1.1514610310E-1f},
         TFloat{7.0376836292E-2f});
   TFloat x2 = x * x;
   ret *= x2 * x;

   ret = FusedMultiplyAdd(exponentFloat, TFloat{-2.12194440E-4f}, ret);
   ret += FusedNegateMultiplyAdd(x2, TFloat{0.5f}, x);

   if(bNegateOutput) {
      ret = FusedMultiplySubtract(exponentFloat, TFloat{-0.693359375f}, ret);
   } else {
      ret = FusedMultiplyAdd(exponentFloat, TFloat{0.693359375f}, ret);
   }

   if(bZeroPossible) {
      ret = IfThenElse(val < std::numeric_limits<typename TFloat::T>::min(),
            bNegateOutput ? std::numeric_limits<typename TFloat::T>::infinity() :
                            -std::numeric_limits<typename TFloat::T>::infinity(),
            ret);
   }
   if(bNegativePossible) {
      ret = IfThenElse(val < TFloat{0}, std::numeric_limits<typename TFloat::T>::quiet_NaN(), ret);
   }
   if(bNaNPossible) {
      if(bPositiveInfinityPossible) {
         ret = IfThenElse(val < std::numeric_limits<typename TFloat::T>::infinity(), ret, bNegateOutput ? -val : val);
      } else {
         ret = IfThenElse(IsNaN(val), val, ret);
      }
   } else {
      if(bPositiveInfinityPossible) {
         ret = IfThenElse(std::numeric_limits<typename TFloat::T>::infinity() == val,
               bNegateOutput ? -std::numeric_limits<typename TFloat::T>::infinity() :
                               std::numeric_limits<typename TFloat::T>::infinity(),
               ret);
      }
   }

#ifndef NDEBUG
   TFloat::Execute(
         [](int, typename TFloat::T orig, typename TFloat::T ret) {
            EBM_ASSERT(IsApproxEqual(std::log(orig), ret, typename TFloat::T{1e-6}));
         },
         val,
         bNegateOutput ? -ret : ret);
#endif // NDEBUG

   return ret;
}

template<typename TFloat,
      bool bNegateInput = false,
      bool bNaNPossible = true,
      bool bUnderflowPossible = true,
      bool bOverflowPossible = true>
static INLINE_ALWAYS TFloat Exp64(const TFloat val) {
   // algorithm comes from:
   // https://github.com/vectorclass/version2/blob/f4617df57e17efcd754f5bbe0ec87883e0ed9ce6/vectormath_exp.h#L327

   // k_expUnderflow is set to a value that prevents us from returning a denormal number.
   static constexpr double k_expUnderflow = -708.25; // this is exactly representable in IEEE 754
   static constexpr double k_expOverflow = 708.25; // this is exactly representable in IEEE 754

   TFloat rounded;
   TFloat x;
   if(bNegateInput) {
      rounded = Round(val * TFloat{-1.44269504088896340736});
      x = FusedMultiplySubtract(rounded, TFloat{-0.693145751953125}, val);
   } else {
      rounded = Round(val * TFloat{1.44269504088896340736});
      x = FusedMultiplyAdd(rounded, TFloat{-0.693145751953125}, val);
   }
   x = FusedNegateMultiplyAdd(rounded, TFloat{1.42860682030941723212E-6}, x);

   TFloat ret = Polynomial64(x,
         TFloat{1} / TFloat{2},
         TFloat{1} / TFloat{6},
         TFloat{1} / TFloat{24},
         TFloat{1} / TFloat{120},
         TFloat{1} / TFloat{720},
         TFloat{1} / TFloat{5040},
         TFloat{1} / TFloat{40320},
         TFloat{1} / TFloat{362880},
         TFloat{1} / TFloat{3628800},
         TFloat{1} / TFloat{39916800},
         TFloat{1} / TFloat{479001600},
         TFloat{1} / TFloat{int64_t{6227020800}});

   const TFloat rounded2 = Power64(rounded);

   ret = (ret + TFloat{1}) * rounded2;

   if(bOverflowPossible) {
      if(bNegateInput) {
         ret = IfThenElse(val < TFloat{-k_expOverflow},
               std::numeric_limits<typename TFloat::T>::infinity(),
               ret);
      } else {
         ret = IfThenElse(TFloat{k_expOverflow} < val,
               std::numeric_limits<typename TFloat::T>::infinity(),
               ret);
      }
   }
   if(bUnderflowPossible) {
      if(bNegateInput) {
         ret = IfThenElse(TFloat{-k_expUnderflow} < val, TFloat{0}, ret);
      } else {
         ret = IfThenElse(val < TFloat{k_expUnderflow}, TFloat{0}, ret);
      }
   }
   if(bNaNPossible) {
      ret = IfThenElse(IsNaN(val), val, ret);
   }

#ifndef NDEBUG
   TFloat::Execute(
         [](int, typename TFloat::T orig, typename TFloat::T ret) {
            EBM_ASSERT(IsApproxEqual(std::exp(orig), ret, typename TFloat::T{1e-12}));
         },
         bNegateInput ? -val : val,
         ret);
#endif // NDEBUG

   return ret;
}

template<typename TFloat,
      bool bNegateOutput = false,
      bool bNaNPossible = true,
      bool bNegativePossible = true,
      bool bZeroPossible = true,
      bool bPositiveInfinityPossible = true>
static INLINE_ALWAYS TFloat Log64(const TFloat& val) noexcept {
   // algorithm comes from:
   // https://github.com/vectorclass/version2/blob/f4617df57e17efcd754f5bbe0ec87883e0ed9ce6/vectormath_exp.h#L1048

   TFloat x = Mantissa64(val);
   TFloat exponent = Exponent64(val);

   const auto comparison = x <= TFloat{1.41421356237309504880 * 0.5};
   x = IfAdd(comparison, x, x);
   exponent = IfAdd(~comparison, exponent, TFloat{1});

   x += TFloat{-1};

   TFloat poly1 = Polynomial64(x,
         TFloat{7.70838733755885391666E0},
         TFloat{1.79368678507819816313E1},
         TFloat{1.44989225341610930846E1},
         TFloat{4.70579119878881725854E0},
         TFloat{4.97494994976747001425E-1},
         TFloat{1.01875663804580931796E-4});
   TFloat x2 = x * x;
   poly1 *= x * x2;
   TFloat poly2 = Polynomial64(x,
         TFloat{2.31251620126765340583E1},
         TFloat{7.11544750618563894466E1},
         TFloat{8.29875266912776603211E1},
         TFloat{4.52279145837532221105E1},
         TFloat{1.12873587189167450590E1});
   TFloat ret = poly1 / poly2;

   ret = FusedMultiplyAdd(exponent, TFloat{-2.121944400546905827679E-4}, ret);
   ret += FusedNegateMultiplyAdd(x2, TFloat{0.5}, x);

   if(bNegateOutput) {
      ret = FusedMultiplySubtract(exponent, TFloat{-0.693359375}, ret);
   } else {
      ret = FusedMultiplyAdd(exponent, TFloat{0.693359375}, ret);
   }

   if(bZeroPossible) {
      ret = IfThenElse(val < std::numeric_limits<typename TFloat::T>::min(),
            bNegateOutput ? std::numeric_limits<typename TFloat::T>::infinity() :
                            -std::numeric_limits<typename TFloat::T>::infinity(),
            ret);
   }
   if(bNegativePossible) {
      ret = IfThenElse(val < TFloat{0}, std::numeric_limits<typename TFloat::T>::quiet_NaN(), ret);
   }
   if(bNaNPossible) {
      if(bPositiveInfinityPossible) {
         ret = IfThenElse(val < std::numeric_limits<typename TFloat::T>::infinity(), ret, bNegateOutput ? -val : val);
      } else {
         ret = IfThenElse(IsNaN(val), val, ret);
      }
   } else {
      if(bPositiveInfinityPossible) {
         ret = IfThenElse(std::numeric_limits<typename TFloat::T>::infinity() == val,
               bNegateOutput ? -std::numeric_limits<typename TFloat::T>::infinity() :
                               std::numeric_limits<typename TFloat::T>::infinity(),
               ret);
      }
   }

#ifndef NDEBUG
   TFloat::Execute(
         [](int, typename TFloat::T orig, typename TFloat::T ret) {
            EBM_ASSERT(IsApproxEqual(std::log(orig), ret, typename TFloat::T{1e-12}));
         },
         val,
         bNegateOutput ? -ret : ret);
#endif // NDEBUG

   return ret;
}

} // namespace DEFINED_ZONE_NAME

#endif // REGISTRATION_HPP
