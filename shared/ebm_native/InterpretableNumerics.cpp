// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // std::numeric_limits
#include <string.h> // strchr, memmove

#include "ebm_native.h"
#include "EbmInternal.h"
#include "logging.h" // EBM_ASSERT & LOG

constexpr FloatEbmType k_percentageDeviationFromEndpointForInterpretableNumbers = FloatEbmType { 0.25 };

static_assert(FLOAT_EBM_MAX == std::numeric_limits<FloatEbmType>::max(), "FLOAT_EBM_MAX mismatch");
static_assert(FLOAT_EBM_LOWEST == std::numeric_limits<FloatEbmType>::lowest(), "FLOAT_EBM_LOWEST mismatch");
static_assert(FLOAT_EBM_MIN == std::numeric_limits<FloatEbmType>::min(), "FLOAT_EBM_MIN mismatch");
// FLOAT_EBM_DENORM_MIN isn't included in g++'s float.h, even though it's a C11 construct
//static_assert(FLOAT_EBM_DENORM_MIN == std::numeric_limits<FloatEbmType>::denorm_min(), "FLOAT_EBM_DENORM_MIN mismatch");
static_assert(FLOAT_EBM_POSITIVE_INF == std::numeric_limits<FloatEbmType>::infinity(), "FLOAT_EBM_POSITIVE_INF mismatch");
static_assert(FLOAT_EBM_NEGATIVE_INF == -std::numeric_limits<FloatEbmType>::infinity(), "FLOAT_EBM_NEGATIVE_INF mismatch");
#ifndef __clang__ // compiler type (clang++)
// clang's static checker seems to dislike this comparison and says it's not an integral comparison, but it is!
static_assert(FLOAT_EBM_NAN != FLOAT_EBM_NAN, "FLOAT_EBM_NAN mismatch"); // a != a is only true for NaN
#endif

INLINE_ALWAYS constexpr static size_t CountBase10CharactersAbs(int n) noexcept {
   // this works for negative numbers too
   return int { 0 } == n / int { 10 } ? size_t { 1 } : size_t { 1 } + CountBase10CharactersAbs(n / int { 10 });
}

// According to the C++ documentation, std::numeric_limits<FloatEbmType>::max_digits10 - 1 digits 
// are required after the period in +9.1234567890123456e-301 notation, so for a double, the values would be 
// 17 == std::numeric_limits<FloatEbmType>::max_digits10, and printf format specifier "%.16e"
constexpr size_t k_cDigitsAfterPeriod = size_t { std::numeric_limits<FloatEbmType>::max_digits10 } - size_t { 1 };

// Unfortunately, min_exponent10 doesn't seem to include subnormal numbers, so although it's the true
// minimum exponent in terms of the floating point exponential representation, it isn't the true minimum exponent 
// when considering numbers converted into text.  To counter this, we add 1 extra digit.  For double numbers
// the largest exponent (+308), the smallest exponent for normal (-308), and the smallest exponent for subnormal (-324) 
// all have 3 digits, but in the more general scenario we might go from N to N+1 digits, but I think
// it's really unlikely to go from N to N+2, since in the simplest case that would be a factor of 10 in the 
// exponential term (if the low number was almost N and the high number was just a bit above N+2), and 
// subnormal numbers shouldn't increase the exponent by that much ever.
constexpr size_t k_cExponentMaxTextDigits = CountBase10CharactersAbs(std::numeric_limits<FloatEbmType>::max_exponent10);
constexpr size_t k_cExponentMinTextDigits = CountBase10CharactersAbs(std::numeric_limits<FloatEbmType>::min_exponent10) + size_t { 1 };
constexpr size_t k_cExponentTextDigits =
k_cExponentMaxTextDigits < k_cExponentMinTextDigits ? k_cExponentMinTextDigits : k_cExponentMaxTextDigits;

// we have a function that ensures our output is exactly in the format that we require.  That format is:
// "+9.1234567890123456e-301" (this is when 16 == cDigitsAfterPeriod, the value for doubles)
// the exponential term can have some variation.  It can be any number of digits and the '+' isn't required
// our text float handling code handles these conditions without requiring modification.
// 3 characters for "+9."
// cDigitsAfterPeriod characters for the mantissa text
// 2 characters for "e-"
// cExponentTextDigits characters for the exponent text
// 1 character for null terminator
constexpr size_t k_iExp = size_t { 3 } + k_cDigitsAfterPeriod;
constexpr size_t k_cCharsFloatPrint = k_iExp + size_t { 2 } + k_cExponentTextDigits + size_t { 1 };

extern FloatEbmType ArithmeticMean(
   const FloatEbmType low,
   const FloatEbmType high
) noexcept {
   // nan values represent missing, and are filtered out from our data prior to discretization
   EBM_ASSERT(!std::isnan(low));
   EBM_ASSERT(!std::isnan(high));

   // -infinity is converted to std::numeric_limits<FloatEbmType>::lowest() and 
   // +infinity is converted to std::numeric_limits<FloatEbmType>::max() in our data prior to discretization
   EBM_ASSERT(!std::isinf(low));
   EBM_ASSERT(!std::isinf(high));

   EBM_ASSERT(low < high); // if two numbers were equal, we wouldn't put a cut point between them

   static_assert(std::numeric_limits<FloatEbmType>::is_iec559,
      "IEEE 754 gives us certain guarantees for floating point results that we use below");

   // this multiplication before addition format avoid overflows/underflows at the cost of a little more work.
   // IEEE 754 guarantees that 0.5 is representable as 2^(-1), so it has an exact representation.
   // IEEE 754 guarantees that division and addition give exactly rounded results.  Since we're multiplying by 0.5,
   // the internal representation will have the same mantissa, but will decrement the exponent, unless it underflows
   // to zero, or we have a subnormal number, which also works for reasons described below.
   // Fundamentally, the average can be equal to low if high is one epsilon tick above low.  If low is zero and 
   // high is the smallest number, then both numbers divided by two are zero and the average is zero.  
   // If low is the smallest number and high is one tick above that, low will go to zero on the division, but 
   // high will become the smallest number since it uses powers of two, so the avg is again the 
   // low value in this case.
   FloatEbmType avg = low * FloatEbmType { 0.5 } + high * FloatEbmType { 0.5 };

   EBM_ASSERT(!std::isnan(avg)); // in no reasonable implementation should this result in NaN

   // in theory, EBM_ASSERT(!std::isinf(avg)); should be ok, but there are bad IEEE 754 implementations that might
   // do the addition before the multiplication, which could result in overflow if done that way.

   // these should be correct in IEEE 754, even with floating point inexactness, due to "correct rounding"
   // in theory, EBM_ASSERT(low <= avg); AND EBM_ASSERT(avg < high); should be ok, but there are bad IEEE 754 
   // implementations that might incorrectly implement "correct rounding", like the Intel x87 instructions

   // if our result is equal to low, then high should be guaranteed to be the next highest floating point number
   // in theory, EBM_ASSERT(low < avg || low == avg && std::nextafter(low, high) == high); should be ok, but
   // this depends on "correct rounding" which isn't true of all compilers

   if(UNLIKELY(avg <= low)) {
      // This check is required to handle the case where high is one epsilon higher than low, which means the average 
      // could be low (the average could also be higher than low, but we don't need to handle that)
      // In that case, our only option is to make our cut equal to high, since we use lower bound inclusive semantics
      //
      // this check has the added benefit that if we have a compiler/platform that isn't truely IEEE 754 compliant,
      // which is sadly common due to double rounding and other issues, then we'd return high, 
      // which is a legal value for us to cut on, and if we have values this close, it's appropriate to just return
      // high instead of doing a more exhaustive examination
      //
      // this check has the added advantage of checking for -infinity
      avg = high;
   }
   if(UNLIKELY(high < avg)) {
      // because so many compilers claim to be IEEE 754, but are not, we have this fallback to prevent us
      // from crashing due to unexpected outputs.  high is a legal value to return since we use lower bound 
      // inclusivity.  I don't see how, even in a bad compiler/platform, we'd get a NaN result I'm not including it
      // here.  Some non-compliant platforms might get to +-infinity if they do the addition first then multiply
      // so that's one possibility to be wary about
      //
      // this check has the added advantage of checking for +infinity
      avg = high;
   }
   return avg;
}

INLINE_RELEASE_UNTEMPLATED static FloatEbmType GeometricMeanPositives(const FloatEbmType low, const FloatEbmType high) noexcept {
   // nan values represent missing, and are filtered out from our data prior to discretization
   EBM_ASSERT(!std::isnan(low));
   EBM_ASSERT(!std::isnan(high));

   // -infinity is converted to min_float and +infinity is converted to max_float in our data prior to discretization
   EBM_ASSERT(!std::isinf(low));
   EBM_ASSERT(!std::isinf(high));

   // we handle zeros outside of this function
   EBM_ASSERT(FloatEbmType { 0 } < low);
   EBM_ASSERT(FloatEbmType { 0 } < high);

   EBM_ASSERT(low < high);

   // in a reasonable world, with both low and high being non-zero, non-nan, non-infinity, and 
   // positive values before calling log, log should return a non-overflowing or non-underflowing 
   // value since all floating point values from -min to +max for floats give us reasonable log values.  
   // Since our logs should average to a number that is between them, the exp value should result in a value 
   // between them in almost all cases, so it shouldn't overflow or underflow either.  BUT, with floating
   // point jitter, we might get any of these scenarios.  This is a real corner case that we can presume
   // is very very very rare.

   FloatEbmType result = std::exp((std::log(low) + std::log(high)) * FloatEbmType {
      0.5
   });

   // IEEE 754 doesn't give us a lot of guarantees about log and exp.  They don't have have "correct rounding"
   // guarantees, unlike basic operators, so we could obtain results outside of our bounds, or perhaps
   // even overflow or underflow in a way that would lead to infinities.  I can't think of a way to get NaN
   // but who knows what's happening inside log, which would get NaN for zero and in a bad implementation
   // perhaps might return that for subnormal floats.
   //
   // If our result is not between low and high, then low and high should be very close and we can use the
   // arithmatic mean.  In the spirit of not trusting log and exp, we'll check for bad outputs and 
   // switch to arithmatic mean.  In the case that we have nan or +-infinity, we aren't guaranteed that
   // low and high are close, so we can't really use an approach were we move small epsilon values in 
   // our floats, so the artithmetic mean is really our only viable falllback in that case.
   //
   // Even in the fully compliant IEEE 754 case, result could be equal to low, so we do need to handle that
   // since we can't return the low value given we use lower bound inclusivity for cut points

   // checking the bounds also checks for +-infinity
   if(std::isnan(result) || result <= low || high < result) {
      result = ArithmeticMean(low, high);
   }
   return result;
}

static bool FloatToString(const FloatEbmType val, char * const str) noexcept {
   EBM_ASSERT(!std::isnan(val));
   EBM_ASSERT(!std::isinf(val));
   EBM_ASSERT(FloatEbmType { 0 } <= val);
   EBM_ASSERT(nullptr != str);

   // NOTE: str must be a buffer with k_cCharsFloatPrint characters available 

   // the C++ standard is pretty good about harmonizing the "e" format.  There is some openess to what happens
   // in the exponent (2 or 3 digits with or without the leading sign character, etc).  If there is ever any
   // implementation observed that differs, this function should convert all formats to a common standard that
   // we use for string manipulation to find interpretable cut points, so we need all strings to have a common format

   // snprintf says to use the buffer size for the "n" term, but in alternate unicode versions it says # of characters
   // with the null terminator as one of the characters, so a string of 5 characters plus a null terminator would be 6.
   // For char strings, the number of bytes and the number of characters is the same.  I use number of characters for 
   // future-proofing the n term to unicode versions, so n-1 characters other than the null terminator can fill 
   // the buffer.  According to the docs, snprintf returns the number of characters that would have been written MINUS 
   // the null terminator.

   constexpr static char g_pPrintfForRoundTrip[] = "%+.*" FloatEbmTypePrintf;

   const int cCharsWithoutNullTerminator = snprintf(
      str,
      k_cCharsFloatPrint,
      g_pPrintfForRoundTrip,
      int { k_cDigitsAfterPeriod },
      val
   );
   if(cCharsWithoutNullTerminator < int { k_iExp + size_t { 2 } } ||
      int { k_cCharsFloatPrint } <= cCharsWithoutNullTerminator) {
      // cCharsWithoutNullTerminator < iExp + 2 checks for both negative values returned and strings that are too short
      // we need the 'e' and at least one digit, so +2 is legal, and anything less is illegal
      return true;
   }
   char ch;
   ch = str[0];
   if('+' != ch) {
      return true;
   }
   ch = str[1];
   if(ch < '0' || '9' < ch) {
      return true;
   }
   ch = str[2];
   if('.' != ch) {
      return true;
   }
   char * pch = &str[3];
   char * pE = &str[k_iExp];
   do {
      ch = *pch;
      if(ch < '0' || '9' < ch) {
         return true;
      }
      ++pch;
   } while(pch != pE);
   ch = *pch;
   if('e' != ch && 'E' != ch) {
      return true;
   }

   // use strtol instead of atol in case we have a bad input.  atol has undefined behavior if the
   // number isn't representable as an int.  strtol returns a 0 with bad inputs, or LONG_MAX, or LONG_MIN, 
   // on overflow or underflow.  The C++ standard makes clear though that on error strtol sets endptr
   // equal to str, so we can use that

   ++pch;
   char * endptr = pch; // set it to the error value so that even if the function doesn't set it we get an error
   // we use endptr to detect failure, so ignore the return value.  Use the (void)! trick to eliminate WARNINGs
   (void)! strtol(pch, &endptr, 10);
   if(endptr <= pch) {
      return true;
   }
   return false;
}

INLINE_RELEASE_UNTEMPLATED static long GetExponent(const char * const str) noexcept {
   // we previously checked that this converted to a long in FloatToString
   return strtol(&str[k_iExp + size_t { 1 }], nullptr, int { 10 });
}

static FloatEbmType StringToFloatWithFixup(
   const char * const str,
   const size_t iIdenticalCharsRequired
) noexcept {
   char strRehydrate[k_cCharsFloatPrint];

   // we only convert str values that we've verified to conform, OR chopped versions of these which we know to be legal
   // If the chopped representations underflow (possible on chopping to lower) or 
   // overflow (possible when we increment from the lower chopped value), then strtod gives 
   // us enough information to convert these

   static_assert(std::is_same<FloatEbmType, double>::value,
      "FloatEbmType must be double, otherwise use something other than strtod");

   // the documentation says that if we have an underflow or overflow, strtod returns us +-HUGE_VAL, which is
   // +-infinity for at least some implementations.  We can't really take a ratio from those numbers, so convert
   // this to the lowest and max values

   // TODO: switch over to using our better ConvertStringToFloat function now!
   FloatEbmType ret = strtod(str, nullptr);

   // this is a check for -infinity/-HUGE_VAL, without the -infinity value since some compilers make that illegal
   // even so far as to make isinf always FALSE with some compiler flags
   // include the equals case so that the compiler is less likely to optimize that out
   ret = ret <= std::numeric_limits<FloatEbmType>::lowest() ? std::numeric_limits<FloatEbmType>::lowest() : ret;
   // this is a check for +infinity/HUGE_VAL, without the +infinity value since some compilers make that illegal
   // even so far as to make isinf always FALSE with some compiler flags
   // include the equals case so that the compiler is less likely to optimize that out
   ret = std::numeric_limits<FloatEbmType>::max() <= ret ? std::numeric_limits<FloatEbmType>::max() : ret;

   if(FloatToString(ret, strRehydrate)) {
      return ret;
   }

   if(0 == memcmp(str, strRehydrate, iIdenticalCharsRequired * sizeof(*str))) {
      return ret;
   }

   EBM_ASSERT('+' == str[0]);

   // according to the C++ docs, nextafter won't exceed the to parameter, so we don't have to worry about this
   // generating infinities
   ret = std::nextafter(ret, std::numeric_limits<FloatEbmType>::max());

   return ret;
}

static bool StringToFloatChopped(
   const char * const pStr,
   size_t iTruncateMantissaTextDigitsAfterFirstDigit,
   FloatEbmType * const pLowChopOut,
   FloatEbmType * const pHighChopOut
) noexcept {
   // the lowChopOut returned can be equal to highChopOut if pStr is an overflow

   // when iTruncateMantissaTextDigitsAfterFirstDigit is zero we chop anything after the first digit, so 
   // 3.456789*10^4 -> 3*10^4 when iTruncateMantissaTextDigitsAfterFirstDigit == 0
   // 3.456789*10^4 -> 3.4*10^4 when iTruncateMantissaTextDigitsAfterFirstDigit == 1

   EBM_ASSERT(nullptr != pStr);
   EBM_ASSERT('+' == pStr[0]);
   // don't pass us a non-truncated string, since we should handle anything that gets to that level differently
   EBM_ASSERT(iTruncateMantissaTextDigitsAfterFirstDigit < k_cDigitsAfterPeriod);

   char strTruncated[k_cCharsFloatPrint];

   // eg: "+9.1234567890123456e-301"
   size_t iTruncateTextAfter = size_t { 0 } == iTruncateMantissaTextDigitsAfterFirstDigit ?
      size_t { 2 } : iTruncateMantissaTextDigitsAfterFirstDigit + size_t { 3 };

   memcpy(strTruncated, pStr, iTruncateTextAfter * sizeof(*pStr));
   strcpy_NO_WARNINGS(&strTruncated[iTruncateTextAfter], &pStr[k_iExp]);

   if(PREDICTABLE(nullptr != pLowChopOut)) {
      *pLowChopOut = StringToFloatWithFixup(strTruncated, iTruncateTextAfter);
   }
   if(PREDICTABLE(nullptr != pHighChopOut)) {
      char * pDigit = &strTruncated[iTruncateTextAfter - size_t { 1 }];
      char ch;
      if(size_t { 2 } == iTruncateTextAfter) {
         goto start_at_top;
      }
      while(true) {
         ch = *pDigit;
         if('.' == ch) {
            --pDigit;
         start_at_top:;
            EBM_ASSERT(strTruncated + size_t { 1 } == pDigit);
            ch = *pDigit;
            if('9' == ch) {
               // oh, great.  now we need to increment our exponential
               int exponent = GetExponent(pStr) + int { 1 };
               *pDigit = '1';
               *(pDigit + size_t { 1 }) = 'e';

               constexpr static char g_pPrintfLongInt[] = "%+d";
               // for the size -> one for the '+' or '-' sign, k_cExponentTextDigits for the digits, 1 for null terminator
               int cCharsWithoutNullTerminator = snprintf(
                  pDigit + size_t { 2 },
                  size_t { 1 } + k_cExponentTextDigits + size_t { 1 },
                  g_pPrintfLongInt,
                  exponent
               );
               if(cCharsWithoutNullTerminator <= int { 1 } ||
                  int { size_t { 1 } + k_cExponentTextDigits } < cCharsWithoutNullTerminator) {
                  return true;
               }
               // we don't have all those '9' characters anymore to check.  we just need the 1
               iTruncateTextAfter = size_t { 2 };
            } else {
               EBM_ASSERT('0' <= ch && ch <= '8');
               *pDigit = ch + char { 1 };
            }
            break;
         } else if('9' == ch) {
            *pDigit = '0';
            --pDigit;
         } else {
            EBM_ASSERT('0' <= ch && ch <= '8');
            *pDigit = ch + char { 1 };
            break;
         }
      }
      *pHighChopOut = StringToFloatWithFixup(strTruncated, iTruncateTextAfter);
   }
   return false;
}

extern FloatEbmType GetInterpretableCutPointFloat(
   FloatEbmType low,
   FloatEbmType high
) noexcept {
   // TODO : add logs or asserts here when we find a condition we didn't think was possible, but that occurs

   // nan values represent missing, and are filtered out from our data prior to discretization
   EBM_ASSERT(!std::isnan(low));
   EBM_ASSERT(!std::isnan(high));

   // -infinity is converted to std::numeric_limits<FloatEbmType>::lowest() and 
   // +infinity is converted to std::numeric_limits<FloatEbmType>::max() in our data prior to discretization
   EBM_ASSERT(!std::isinf(low));
   EBM_ASSERT(!std::isinf(high));

   EBM_ASSERT(low < high); // if two numbers were equal, we wouldn't put a cut point between them
   EBM_ASSERT(low < std::numeric_limits<FloatEbmType>::max());
   EBM_ASSERT(std::numeric_limits<FloatEbmType>::lowest() < high);

   // if our numbers pass the asserts above, all combinations of low and high values can get a legal cut point, 
   // since we can always return the high value given that our binning is lower bound inclusive

   FloatEbmType lowChop;
   FloatEbmType highChop;
   char strAvg[k_cCharsFloatPrint];
   FloatEbmType ret;

   bool bNegative = false;
   if(UNLIKELY(low <= FloatEbmType { 0 })) {
      if(UNLIKELY(FloatEbmType { 0 } == low)) {
         EBM_ASSERT(FloatEbmType { 0 } < high);
         // half of any number should give us something with sufficient distance.  For instance probably the worse
         // number would be something like 1.999999999999*10^1 where the division by two might round up to
         // 1.000000000000*10^1.  In that case though, we'll find that 1*10^1 is closest to the average, and we'll
         // choose that instead of the much farther away 2.000*10^1

         const FloatEbmType avg = high * FloatEbmType { 0.5 };
         EBM_ASSERT(!std::isnan(avg));
         EBM_ASSERT(!std::isinf(avg));
         EBM_ASSERT(FloatEbmType { 0 } <= avg);
         ret = high;
         // check for underflow
         if(LIKELY(FloatEbmType { 0 } != avg)) {
            ret = avg;
            if(LIKELY(!FloatToString(ret, strAvg)) && LIKELY(!StringToFloatChopped(strAvg, 0, &lowChop, &highChop))) {
               EBM_ASSERT(!std::isnan(lowChop));
               EBM_ASSERT(!std::isinf(lowChop));
               // it's possible we could have chopped off digits such that we round down to zero
               EBM_ASSERT(FloatEbmType { 0 } <= lowChop);
               EBM_ASSERT(lowChop <= ret);
               // check for underflow from digit chopping.  If this happens avg/high must be pretty close to zero
               if(LIKELY(FloatEbmType { 0 } != lowChop)) {
                  EBM_ASSERT(!std::isnan(highChop));
                  EBM_ASSERT(!std::isinf(highChop));
                  EBM_ASSERT(FloatEbmType { 0 } < highChop);
                  EBM_ASSERT(ret <= highChop);

                  const FloatEbmType highDistance = highChop - ret;
                  EBM_ASSERT(!std::isnan(highDistance));
                  EBM_ASSERT(!std::isinf(highDistance));
                  EBM_ASSERT(FloatEbmType { 0 } <= highDistance);
                  const FloatEbmType lowDistance = ret - lowChop;
                  EBM_ASSERT(!std::isnan(lowDistance));
                  EBM_ASSERT(!std::isinf(lowDistance));
                  EBM_ASSERT(FloatEbmType { 0 } <= lowDistance);

                  ret = UNPREDICTABLE(highDistance <= lowDistance) ? highChop : lowChop;
               }
            }
         }

         EBM_ASSERT(!std::isnan(ret));
         EBM_ASSERT(!std::isinf(ret));
         EBM_ASSERT(low < ret);
         EBM_ASSERT(ret <= high);

         return ret;
      }

      if(UNLIKELY(FloatEbmType { 0 } <= high)) {
         // if low is negative and high is zero or positive, a natural cut point is zero.  Also, this solves the issue
         // that we can't take the geometric mean of mixed positive/negative numbers.  This works since we use 
         // lower bound inclusivity, so a cut point of 0 will include the number 0 in the upper bin.  Normally we try 
         // to avoid putting a cut directly on one of the numbers, but in the case of zero it seems appropriate.
         ret = FloatEbmType { 0 };
         if(UNLIKELY(FloatEbmType { 0 } == high)) {
            // half of any number should give us something with sufficient distance.  For instance probably the worse
            // number would be something like 1.999999999999*10^1 where the division by two might round up to
            // 1.000000000000*10^1.  In that case though, we'll find that 1*10^1 is closest to the average, and we'll
            // choose that instead of the much farther away 2.000*10^1

            ret = low * FloatEbmType { -0.5 };
            EBM_ASSERT(!std::isnan(ret));
            EBM_ASSERT(!std::isinf(ret));
            EBM_ASSERT(FloatEbmType { 0 } <= ret);

            if(LIKELY(!FloatToString(ret, strAvg)) && LIKELY(!StringToFloatChopped(strAvg, 0, &lowChop, &highChop))) {
               EBM_ASSERT(!std::isnan(lowChop));
               EBM_ASSERT(!std::isinf(lowChop));
               // it's possible we could have chopped off digits such that we round down to zero
               EBM_ASSERT(FloatEbmType { 0 } <= lowChop);
               EBM_ASSERT(lowChop <= ret);

               EBM_ASSERT(!std::isnan(highChop));
               EBM_ASSERT(!std::isinf(highChop));
               EBM_ASSERT(FloatEbmType { 0 } < highChop);
               EBM_ASSERT(ret <= highChop);

               const FloatEbmType highDistance = highChop - ret;
               EBM_ASSERT(!std::isnan(highDistance));
               EBM_ASSERT(!std::isinf(highDistance));
               EBM_ASSERT(FloatEbmType { 0 } <= highDistance);
               const FloatEbmType lowDistance = ret - lowChop;
               EBM_ASSERT(!std::isnan(lowDistance));
               EBM_ASSERT(!std::isinf(lowDistance));
               EBM_ASSERT(FloatEbmType { 0 } <= lowDistance);

               ret = UNPREDICTABLE(highDistance <= lowDistance) ? highChop : lowChop;
            }
            ret = -ret;
         }

         EBM_ASSERT(!std::isnan(ret));
         EBM_ASSERT(!std::isinf(ret));
         EBM_ASSERT(low < ret);
         EBM_ASSERT(ret <= high);

         return ret;
      }

      const FloatEbmType tmpLow = low;
      low = -high;
      high = -tmpLow;
      bNegative = true;
   } else {
      EBM_ASSERT(FloatEbmType { 0 } < high);
   }

   EBM_ASSERT(FloatEbmType { 0 } < low);
   EBM_ASSERT(FloatEbmType { 0 } < high);
   EBM_ASSERT(low < high);
   EBM_ASSERT(low < std::numeric_limits<FloatEbmType>::max());
   EBM_ASSERT(high <= std::numeric_limits<FloatEbmType>::max());

   // divide by high since it's guaranteed to be bigger than low, so we can't blow up to infinity
   const FloatEbmType ratio = low / high;
   EBM_ASSERT(!std::isnan(ratio));
   EBM_ASSERT(!std::isinf(ratio));
   EBM_ASSERT(ratio <= FloatEbmType { 1 });
   EBM_ASSERT(FloatEbmType { 0 } <= ratio);

   // don't transition on a perfect 1000 ratio from arithmetic to geometric mean since many of our numbers
   // are probably going to be whole numbers and we don't want floating point inexactness to dictate the
   // transition, so choose a number just slightly lower than 1000, in this case 996.18959224497322090157279627358
   if(ratio < FloatEbmType { 0.001003824982498 }) {
      ret = GeometricMeanPositives(low, high);
      EBM_ASSERT(!std::isnan(ret));
      EBM_ASSERT(!std::isinf(ret));
      EBM_ASSERT(low < ret);
      EBM_ASSERT(ret <= high);

      if(LIKELY(LIKELY(!FloatToString(ret, strAvg)) &&
         LIKELY(!StringToFloatChopped(strAvg, size_t { 0 }, &lowChop, &highChop)))) {
         // avg / low == high / avg (approximately) since it's the geometric mean
         // the lowChop or highChop side that is closest to the average will be farthest away
         // from it's corresponding low/high value
         // since we don't want infinties, we divide the smaller number by the bigger one
         // the smallest number means it has the longest distance from the low/high value, hense it's closer
         // to the average

         EBM_ASSERT(low < lowChop);
         const FloatEbmType lowRatio = low / lowChop;
         EBM_ASSERT(!std::isnan(lowRatio));
         EBM_ASSERT(!std::isinf(lowRatio));
         EBM_ASSERT(lowRatio <= FloatEbmType { 1 });
         EBM_ASSERT(FloatEbmType { 0 } <= lowRatio);

         EBM_ASSERT(highChop < high);
         const FloatEbmType highRatio = highChop / high;
         EBM_ASSERT(!std::isnan(highRatio));
         EBM_ASSERT(!std::isinf(highRatio));
         EBM_ASSERT(highRatio <= FloatEbmType { 1 });
         EBM_ASSERT(FloatEbmType { 0 } <= highRatio);

         ret = UNPREDICTABLE(lowRatio <= highRatio) ? lowChop : highChop;
      }
   } else {
      ret = ArithmeticMean(low, high);
      EBM_ASSERT(!std::isnan(ret));
      EBM_ASSERT(!std::isinf(ret));
      EBM_ASSERT(low < ret);
      EBM_ASSERT(ret <= high);

      char strLow[k_cCharsFloatPrint];
      char strHigh[k_cCharsFloatPrint];
      if(LIKELY(LIKELY(!FloatToString(low, strLow)) &&
         LIKELY(!FloatToString(high, strHigh)) && LIKELY(!FloatToString(ret, strAvg)))) {
         size_t iTruncateMantissa = size_t { 0 };
         do {
            FloatEbmType lowHigh;
            FloatEbmType avgLow;
            FloatEbmType avgHigh;
            FloatEbmType highLow;

            if(UNLIKELY(StringToFloatChopped(strLow, iTruncateMantissa, nullptr, &lowHigh))) {
               break;
            }
            if(UNLIKELY(StringToFloatChopped(strAvg, iTruncateMantissa, &avgLow, &avgHigh))) {
               break;
            }
            if(UNLIKELY(StringToFloatChopped(strHigh, iTruncateMantissa, &highLow, nullptr))) {
               break;
            }

            if(lowHigh < avgLow && avgLow < highLow && low < avgLow && avgLow <= high) {
               // avgLow is a possibility
               if(lowHigh < avgHigh && avgHigh < highLow && low < avgHigh && avgHigh <= high) {
                  // avgHigh is a possibility
                  const FloatEbmType lowDistanceToAverage = ret - avgLow;
                  const FloatEbmType highDistanceToAverage = avgHigh - ret;
                  EBM_ASSERT(-0.000001 < lowDistanceToAverage);
                  EBM_ASSERT(-0.000001 < highDistanceToAverage);
                  if(UNPREDICTABLE(highDistanceToAverage < lowDistanceToAverage)) {
                     ret = avgHigh;
                     break;
                  }
               }
               ret = avgLow;
               break;
            } else {
               if(lowHigh < avgHigh && avgHigh < highLow && low < avgHigh && avgHigh <= high) {
                  // avgHigh works!
                  ret = avgHigh;
                  break;
               }
            }

            ++iTruncateMantissa;
         } while(k_cDigitsAfterPeriod != iTruncateMantissa);
      }
   }
   if(PREDICTABLE(bNegative)) {
      ret = -ret;
   }
   return ret;
}

extern FloatEbmType GetInterpretableEndpoint(
   const FloatEbmType center,
   const FloatEbmType movementFromEnds
) noexcept {
   // TODO : add logs or asserts here when we find a condition we didn't think was possible, but that occurs

   // center can be -infinity OR +infinity
   // movementFromEnds can be +infinity

   EBM_ASSERT(!std::isnan(center));
   EBM_ASSERT(!std::isnan(movementFromEnds));
   EBM_ASSERT(FloatEbmType { 0 } <= movementFromEnds);

   FloatEbmType ret = center;
   // if the center is +-infinity then we'll always be farter away than the end cut points which can't be +-infinity
   // so return +-infinity so that our alternative cut point is rejected
   if(LIKELY(!std::isinf(ret))) {
      // we use movementFromEnds to compute center, so if movementFromEnd was an infinity, then center would be
      // an infinity value.  We filter out infinity values for center above though, so movementFromEnds can't be
      // infinity here, even though the 
      EBM_ASSERT(!std::isinf(movementFromEnds));

      const FloatEbmType distance = k_percentageDeviationFromEndpointForInterpretableNumbers * movementFromEnds;
      EBM_ASSERT(!std::isnan(distance));
      EBM_ASSERT(!std::isinf(distance));
      EBM_ASSERT(FloatEbmType { 0 } <= distance);

      bool bNegative = false;
      if(PREDICTABLE(ret < FloatEbmType { 0 })) {
         ret = -ret;
         bNegative = true;
      }

      const FloatEbmType lowBound = ret - distance;
      EBM_ASSERT(!std::isnan(lowBound));
      // lowBound can be a negative number, but can't be +-infinity because we subtract from a positive number
      // and we use IEEE 754
      EBM_ASSERT(!std::isinf(lowBound));

      const FloatEbmType highBound = ret + distance;
      // highBound can be +infinity, but can't be negative
      EBM_ASSERT(!std::isnan(highBound));
      EBM_ASSERT(FloatEbmType { 0 } <= highBound);

      char str[k_cCharsFloatPrint];
      if(LIKELY(!FloatToString(ret, str))) {
         size_t iTruncateMantissa = size_t { 0 };
         do {
            FloatEbmType lowChop;
            FloatEbmType highChop;

            if(UNLIKELY(StringToFloatChopped(str, iTruncateMantissa, &lowChop, &highChop))) {
               break;
            }

            // these comparisons works even if lowBound is negative or highBound is +infinity
            EBM_ASSERT(!std::isinf(lowChop));
            EBM_ASSERT(!std::isinf(highChop));
            if(lowBound <= lowChop && lowChop <= highBound) {
               // lowChop is a possibility
               if(lowBound <= highChop && highChop <= highBound) {
                  // highChop is a possibility
                  const FloatEbmType lowDistanceToAverage = ret - lowChop;
                  const FloatEbmType highDistanceToAverage = highChop - ret;
                  EBM_ASSERT(-0.000001 < lowDistanceToAverage);
                  EBM_ASSERT(-0.000001 < highDistanceToAverage);
                  if(UNPREDICTABLE(highDistanceToAverage < lowDistanceToAverage)) {
                     ret = highChop;
                     break;
                  }
               }
               ret = lowChop;
               break;
            } else {
               if(lowBound <= highChop && highChop <= highBound) {
                  // highChop works!
                  ret = highChop;
                  break;
               }
            }

            ++iTruncateMantissa;
         } while(k_cDigitsAfterPeriod != iTruncateMantissa);
      }
      if(bNegative) {
         ret = -ret;
      }
   }
   return ret;
}

extern size_t RemoveMissingValuesAndReplaceInfinities(
   size_t cSamples,
   FloatEbmType * const aValues,
   FloatEbmType * const pMinNonInfinityValueOut,
   IntEbmType * const pCountNegativeInfinityOut,
   FloatEbmType * const pMaxNonInfinityValueOut,
   IntEbmType * const pCountPositiveInfinityOut
) noexcept {
   EBM_ASSERT(size_t { 1 } <= cSamples);
   EBM_ASSERT(nullptr != aValues);
   EBM_ASSERT(nullptr != pMinNonInfinityValueOut);
   EBM_ASSERT(nullptr != pCountNegativeInfinityOut);
   EBM_ASSERT(nullptr != pMaxNonInfinityValueOut);
   EBM_ASSERT(nullptr != pCountPositiveInfinityOut);

   // In most cases we believe that for graphing the caller should only need the bin cuts that we'll eventually
   // return, and they'll want to position the graph to include the first and last cuts, and have a little bit of 
   // space both above and below those cuts.  In most cases they shouldn't need the non-infinity min/max values or know
   // whether or not there is +-infinity in the data, BUT on the margins of choosing graphing it might be useful.
   // For example, if the first cut was at 0.1 it might be reasonable to think that the low boundary should be 0,
   // and that would be reasonable if the lowest true value was 0.01, but if the lowest value was actually -0.1,
   // then we might want to instead make our graph start at -1.  Likewise, knowing if there were +-infinity 
   // values in the data probably won't affect the bounds shown, but perhaps the graphing code might want to 
   // somehow indicate the existance of +-infinity values.  The user might write custom graphing code, so we should
   // just return all this information and let the user choose what they want.

   // we really don't want to have cut points that are either -infinity or +infinity because these values are 
   // problematic for serialization, cross language compatibility, human understantability, graphing, etc.
   // In some cases though, +-infinity might carry some information that we do want to capture.  In almost all
   // cases though we can put a cut point between -infinity and the smallest value or +infinity and the largest
   // value.  One corner case is if our data has both max_float and +infinity values.  Our binning uses
   // lower inclusive bounds, so a cut value of max_float will include both max_float and +infinity, so if
   // our algorithm decides to put a cut there we'd be in trouble.  We don't want to make the cut +infinity
   // since that violates our no infinity cut point rule above.  A good compromise is to turn +infinity
   // into max_float.  If we do it here, our cutting algorithm won't need to deal with the odd case of indicating
   // a cut and removing it later.  In theory we could separate -infinity and min_float, since a cut value of
   // min_float would separate the two, but we convert -infinity to min_float here for symmetry with the positive
   // case and for simplicity.

   // when +-infinity values and min_float/max_float values are present, they usually don't represent real values,
   // since it's exceedingly unlikley that min_float or max_float represents a natural value that just happened
   // to not overflow.  When picking our cut points later between values, we should care more about the highest
   // or lowest value that is not min_float/max_float/+-infinity.  So, we convert +-infinity to min_float/max_float
   // here and disregard that value when choosing bin cut points.  We put the bin cut closer to the other value
   // A good point to put the cut is the value that has the same exponent, but increments the top value, so for
   // example, (7.84222e22, +infinity) should have a bin cut value of 8e22).

   // all of this infrastructure gives the user back the maximum amount of information possible, while also avoiding
   // +-infinity values in either the cut points, or the min/max values, which is good since serialization of
   // +-infinity isn't very standardized accross languages.  It's a problem in JSON especially.

   FloatEbmType minNonInfinityValue = std::numeric_limits<FloatEbmType>::max();
   size_t cNegativeInfinity = size_t { 0 };
   FloatEbmType maxNonInfinityValue = std::numeric_limits<FloatEbmType>::lowest();
   size_t cPositiveInfinity = size_t { 0 };

   FloatEbmType * pCopyFrom = aValues;
   const FloatEbmType * const pValuesEnd = aValues + cSamples;
   do {
      FloatEbmType val = *pCopyFrom;
      if(UNLIKELY(std::isnan(val))) {
         FloatEbmType * pCopyTo = pCopyFrom;
         goto skip_val;
         do {
            val = *pCopyFrom;
            if(PREDICTABLE(!std::isnan(val))) {
               if(PREDICTABLE(std::numeric_limits<FloatEbmType>::infinity() == val)) {
                  val = std::numeric_limits<FloatEbmType>::max();
                  ++cPositiveInfinity;
               } else if(PREDICTABLE(-std::numeric_limits<FloatEbmType>::infinity() == val)) {
                  val = std::numeric_limits<FloatEbmType>::lowest();
                  ++cNegativeInfinity;
               } else {
                  maxNonInfinityValue = UNPREDICTABLE(maxNonInfinityValue < val) ? val : maxNonInfinityValue;
                  minNonInfinityValue = UNPREDICTABLE(val < minNonInfinityValue) ? val : minNonInfinityValue;
               }
               *pCopyTo = val;
               ++pCopyTo;
            }
         skip_val:
            ++pCopyFrom;
         } while(LIKELY(pValuesEnd != pCopyFrom));
         const size_t cSamplesWithoutMissing = pCopyTo - aValues;
         EBM_ASSERT(cSamplesWithoutMissing < cSamples);

         cSamples = cSamplesWithoutMissing;
         break;
      }
      if(PREDICTABLE(std::numeric_limits<FloatEbmType>::infinity() == val)) {
         *pCopyFrom = std::numeric_limits<FloatEbmType>::max();
         ++cPositiveInfinity;
      } else if(PREDICTABLE(-std::numeric_limits<FloatEbmType>::infinity() == val)) {
         *pCopyFrom = std::numeric_limits<FloatEbmType>::lowest();
         ++cNegativeInfinity;
      } else {
         maxNonInfinityValue = UNPREDICTABLE(maxNonInfinityValue < val) ? val : maxNonInfinityValue;
         minNonInfinityValue = UNPREDICTABLE(val < minNonInfinityValue) ? val : minNonInfinityValue;
      }
      ++pCopyFrom;
   } while(LIKELY(pValuesEnd != pCopyFrom));

   if(UNLIKELY(cNegativeInfinity + cPositiveInfinity == cSamples)) {
      // all values were special values (missing, +infinity, -infinity), so make our min/max both zero
      maxNonInfinityValue = FloatEbmType { 0 };
      minNonInfinityValue = FloatEbmType { 0 };
   }

   *pMinNonInfinityValueOut = minNonInfinityValue;
   // this can't overflow since we got our cSamples from an IntEbmType, and we can't have more infinities than that
   *pCountNegativeInfinityOut = static_cast<IntEbmType>(cNegativeInfinity);
   *pMaxNonInfinityValueOut = maxNonInfinityValue;
   // this can't overflow since we got our cSamples from an IntEbmType, and we can't have more infinities than that
   *pCountPositiveInfinityOut = static_cast<IntEbmType>(cPositiveInfinity);

   return cSamples;
}

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION SuggestGraphBounds(
   IntEbmType countCuts,
   FloatEbmType lowestCut,
   FloatEbmType highestCut,
   FloatEbmType minValue,
   FloatEbmType maxValue,
   FloatEbmType * lowGraphBoundOut,
   FloatEbmType * highGraphBoundOut
) {
   if(nullptr == lowGraphBoundOut) {
      LOG_0(TraceLevelWarning, "ERROR SuggestGraphBounds nullptr == lowGraphBoundOut");
      return;
   }
   if(nullptr == highGraphBoundOut) {
      LOG_0(TraceLevelWarning, "ERROR SuggestGraphBounds nullptr == highGraphBoundOut");
      return;
   }

   FloatEbmType lowGraphBound;
   FloatEbmType highGraphBound;
   if(countCuts <= IntEbmType { 0 }) {
      if(countCuts < IntEbmType { 0 }) {
         LOG_0(TraceLevelWarning, "ERROR SuggestGraphBounds countCuts < IntEbmType { 0 }");
         lowGraphBound = FloatEbmType { 0 };
         highGraphBound = FloatEbmType { 0 };
      } else {
      use_just_min_max:;

         if(std::isnan(minValue) || -std::numeric_limits<FloatEbmType>::infinity() == minValue) {
            minValue = std::numeric_limits<FloatEbmType>::lowest();
         } else if(std::numeric_limits<FloatEbmType>::infinity() == minValue) {
            minValue = std::numeric_limits<FloatEbmType>::max();
         }

         if(std::isnan(maxValue) || std::numeric_limits<FloatEbmType>::infinity() == maxValue) {
            maxValue = std::numeric_limits<FloatEbmType>::max();
         } else if(-std::numeric_limits<FloatEbmType>::infinity() == maxValue) {
            maxValue = std::numeric_limits<FloatEbmType>::lowest();
         }

         if(maxValue < minValue) {
            // silly caller, these should be reversed
            LOG_0(TraceLevelError, "ERROR SuggestGraphBounds maxValue < minValue");
            const FloatEbmType tmp = minValue;
            minValue = maxValue;
            maxValue = tmp;
         }

         // if countCuts was zero, we can ignore lowestCut and highestCut since they have no meaning.
         // Since there are no cuts, don't turn these into interpretable values.  We show 
         // the most information on the graph by showing the true bounds on the graph.
         lowGraphBound = minValue;
         highGraphBound = maxValue;
      }
   } else if(IntEbmType { 1 } == countCuts) {
      if(std::isnan(lowestCut)) {
         LOG_0(TraceLevelWarning, "WARNING SuggestGraphBounds std::isnan(lowestCut)");
         goto use_just_min_max;
      }
      if(std::isnan(highestCut)) {
         LOG_0(TraceLevelWarning, "WARNING SuggestGraphBounds std::isnan(highestCut)");
         goto use_just_min_max;
      }
      if(lowestCut != highestCut) {
         // if there's only one cut then the lowestCut should be the same as the highestCut.  If they aren't
         // the same, then there's a serious problem and we can't really know which to trust, so ignore them
         LOG_0(TraceLevelError, 
            "ERROR SuggestGraphBounds when 1 == countCuts, then lowestCut and highestCut should be identical");
         goto use_just_min_max;
      }
      const FloatEbmType center = lowestCut;
      if(maxValue < minValue) {
         // silly caller, these should be reversed
         LOG_0(TraceLevelError, "ERROR SuggestGraphBounds maxValue < minValue");
         const FloatEbmType tmp = minValue;
         minValue = maxValue;
         maxValue = tmp;
      }
      if(maxValue < center || center < minValue) {
         LOG_0(TraceLevelError, "ERROR SuggestGraphBounds maxValue < center || center < minValue");
         // hmmm.. the center value isn't between highGraphBound and highGraphBound, so they don't really make sense. Ignore them
         // and just use the full range (which also alerts the user to something odd).
         lowGraphBound = std::numeric_limits<FloatEbmType>::lowest();
         highGraphBound = std::numeric_limits<FloatEbmType>::max();
      } else {
         EBM_ASSERT(-std::numeric_limits<FloatEbmType>::infinity() != maxValue);
         EBM_ASSERT(std::numeric_limits<FloatEbmType>::infinity() != minValue);

         if(-std::numeric_limits<FloatEbmType>::infinity() == minValue) {
            minValue = std::numeric_limits<FloatEbmType>::lowest();
         }
         if(std::numeric_limits<FloatEbmType>::infinity() == maxValue) {
            maxValue = std::numeric_limits<FloatEbmType>::max();
         }

         // since there's just 1 cut, don't turn these into interpretable values.  We show 
         // the most information on the graph by showing the true bounds on the graph.
         const FloatEbmType lowDiff = center - minValue;
         EBM_ASSERT(!std::isnan(lowDiff));
         EBM_ASSERT(FloatEbmType { 0 } <= lowDiff);
         const FloatEbmType highDiff = maxValue - center;
         EBM_ASSERT(!std::isnan(highDiff));
         EBM_ASSERT(FloatEbmType { 0 } <= highDiff);

         if(std::isinf(lowDiff) || std::isinf(highDiff)) {
            // the range is so big that once we extend it both sides will be at infinity, so put them both at the max
            lowGraphBound = std::numeric_limits<FloatEbmType>::lowest();
            highGraphBound = std::numeric_limits<FloatEbmType>::max();
         } else {
            if(lowDiff < highDiff) {
               highGraphBound = maxValue;
               // highGraphBound is farther from our single cut than the lowGraphBound, so let's make it symmetric to the highGraphBound
               // so that lowGraphBound can be shown on the graph
               lowGraphBound = center - highDiff;
               EBM_ASSERT(!std::isnan(lowGraphBound));
               if(-std::numeric_limits<FloatEbmType>::infinity() == lowGraphBound) {
                  lowGraphBound = std::numeric_limits<FloatEbmType>::lowest();
               }
            } else {
               lowGraphBound = minValue;
               // lowGraphBound is farther from our single cut than the highGraphBound, so let's make it symmetric to the lowGraphBound
               // so that highGraphBound can be shown on the graph
               highGraphBound = center + lowDiff;
               EBM_ASSERT(!std::isnan(highGraphBound));
               if(std::numeric_limits<FloatEbmType>::infinity() == highGraphBound) {
                  highGraphBound = std::numeric_limits<FloatEbmType>::max();
               }
            }
         }
      }
   } else {
      // ignore the min and max value.  We have complexity in our graph with multiple cuts so we want to focus on that
      // part.  

      if(std::isnan(lowestCut) || -std::numeric_limits<FloatEbmType>::infinity() == lowestCut) {
         lowestCut = std::numeric_limits<FloatEbmType>::lowest();
      } else if(std::numeric_limits<FloatEbmType>::infinity() == lowestCut) {
         lowestCut = std::numeric_limits<FloatEbmType>::max();
      }

      if(std::isnan(highestCut) || std::numeric_limits<FloatEbmType>::infinity() == highestCut) {
         highestCut = std::numeric_limits<FloatEbmType>::max();
      } else if(-std::numeric_limits<FloatEbmType>::infinity() == highestCut) {
         highestCut = std::numeric_limits<FloatEbmType>::lowest();
      }

      if(highestCut < lowestCut) {
         // silly caller, these should be reversed
         LOG_0(TraceLevelError, "ERROR SuggestGraphBounds highestCut < lowestCut");
         const FloatEbmType tmp = lowestCut;
         lowestCut = highestCut;
         highestCut = tmp;
      }

      const FloatEbmType scaleMin = highestCut - lowestCut;
      // scaleMin can be +infinity if highestCut is max and lowestCut is lowest.  We can handle it.
      EBM_ASSERT(!std::isnan(scaleMin));
      // IEEE 754 (which we static_assert) won't allow the subtraction of two unequal numbers to be non-zero
      EBM_ASSERT(FloatEbmType { 0 } <= scaleMin);

      // limit the amount of dillution allowed for the tails by capping the relevant cCutPointRet value
      // to 1/32, which means we leave about 3% of the visible area to tail bounds (1.5% on the left and
      // 1.5% on the right)

      const size_t cCutsLimited = static_cast<size_t>(IntEbmType { 32 } < countCuts ? IntEbmType { 32 } : countCuts);

      // the leftmost and rightmost cuts can legally be right outside of the bounds between scaleHighLow and
      // scaleLowHigh, so we subtract these two cuts, leaving us the number of ranges between the two end
      // points.  Half a range on the bottom, N - 1 ranges in the middle, and half a range on the top
      // Dividing by that number of ranges gives us the average range width.  We don't want to get the final
      // cut though from the previous inner cut.  We want to move outwards from the scaleHighLow and
      // scaleLowHigh values, which should be half a cut inwards (not exactly but in spirit), so we
      // divide by two, which is the same as multiplying the divisor by 2, which is the right shift below
      EBM_ASSERT(IntEbmType { 2 } <= countCuts); // if there's just one cut then suggest bounds surrounding it
      const size_t denominator = (cCutsLimited - size_t { 1 }) << 1;
      EBM_ASSERT(size_t { 0 } < denominator);
      const FloatEbmType movementFromEnds = scaleMin / static_cast<FloatEbmType>(denominator);
      // movementFromEnds can be +infinity if scaleMin is infinity. We can handle it.
      EBM_ASSERT(!std::isnan(movementFromEnds));
      EBM_ASSERT(FloatEbmType { 0 } <= movementFromEnds);

      lowestCut = lowestCut - movementFromEnds;
      // lowestCut can be -infinity if movementFromEnds is +infinity.  We can handle it.
      EBM_ASSERT(!std::isnan(lowestCut));
      EBM_ASSERT(lowestCut <= std::numeric_limits<FloatEbmType>::max());
      // GetInterpretableEndpoint can accept -infinity, but it'll return -infinity in that case
      lowGraphBound = GetInterpretableEndpoint(lowestCut, movementFromEnds);
      // lowGraphBound can legally be -infinity and we handle this scenario below
      if(-std::numeric_limits<FloatEbmType>::infinity() == lowGraphBound) {
         lowGraphBound = std::numeric_limits<FloatEbmType>::lowest();
      }

      highestCut = highestCut + movementFromEnds;
      // highestCut can be +infinity if movementFromEnds is +infinity.  We can handle it.
      EBM_ASSERT(!std::isnan(highestCut));
      EBM_ASSERT(std::numeric_limits<FloatEbmType>::lowest() <= highestCut);
      // GetInterpretableEndpoint can accept infinity, but it'll return infinity in that case
      highGraphBound = GetInterpretableEndpoint(highestCut, movementFromEnds);
      // highGraphBound can legally be +infinity and we handle this scenario below
      if(std::numeric_limits<FloatEbmType>::infinity() == highGraphBound) {
         highGraphBound = std::numeric_limits<FloatEbmType>::max();
      }
   }
   *lowGraphBoundOut = lowGraphBound;
   *highGraphBoundOut = highGraphBound;
}
