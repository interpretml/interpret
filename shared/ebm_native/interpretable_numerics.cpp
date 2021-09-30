// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // std::numeric_limits
#include <string.h> // strchr, memmove

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

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

extern FloatEbmType ArithmeticMean(const FloatEbmType low, const FloatEbmType high) noexcept {
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

INLINE_RELEASE_UNTEMPLATED static FloatEbmType GeometricMeanPositives(
   const FloatEbmType low, 
   const FloatEbmType high
) noexcept {
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

   FloatEbmType result = std::exp((std::log(low) + std::log(high)) * FloatEbmType { 0.5 });

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

static bool FloatToFullString(const FloatEbmType val, char * const str) noexcept {
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

#if 0

// TODO: this entire section below!

// We have a problem in that converting from a float to a string has many possible legal outputs.
// IEEE-754 requires that if we output 17 to 20 digits for a double that the result can be converted from the
// string back to the original double, but no guarantees are made that 16 digits or 21 digits will work,
// and some language implementations output shorter strings like 0.25 which have exact representations in IEEE-754
// or strings like 0.1 when 0.1 converts back to the original float representation in their representation
// There are odd corner cases that rely on the type of rounding (IEEE-754 requires bankers' rounding for strings
// I believe).  There are legals and illegal ways to format IEEE-754 in text specifically:
// ISO 6093:1985 -> https://www.titanwolf.org/Network/q/4d680399-6711-4742-9900-74a42ad9f5d7/y
// 
// We desire cross-language identical results, so when we have cut points or categorical strings
// of floats we want these to be identical between languages for both converting to strings and when strings
// are converted back to floats.  This also applies to serialization to JSON and other text.  To support this
// we implement our own converters that are guararanteed to be identical between languages.  
// 
// The gold standard for float/string conversion is this: http://www.netlib.org/fp/dtoa.c
// It is used in python: https://github.com/python/cpython/blob/main/Python/dtoa.c
// Microsoft Edge for Android uses it: https://www.microsoft.com/en-us/legal/products/notices/msedgeandroid
// Java uses a port of this made to the Java language
// Other languages also use this code, but not any C++ built-in libraries yet.
// Some languages don't round trip properly with less than 17 digits.
// Some languages are buggy and don't correctly round when outputting 17 digits.
//
// This implementation will progressively shorten the string until it reaches the point where the conversion
// back won't be identical which is a stronger guarantee than IEEE-754.
// 
// Outputting 17 digits should be guaranteed to have a unique conversion back to float, but when shortening
// to 16 digits there are oddities where incrementing the 16th digit upwards yields a good result but
// chopping the 17th digit doesn't work.  I assume there are numbers where both chopping the 17th digit and
// either moving the 16th digit up or keeping it the same yield the same result, so we need a consistent
// policy with regards to the 16th digit.  For the 15th digit we can always chop since there are no numbers
// where moving the 15th digit up yields the same number. One example is:
// 2e-44 which is 5.684341886080801486968994140625e-14.  Rounded to 15 digits (5.68434188608080e-14) doesn't work.
// Rounding down to 16 digits down doesn't work (5.684341886080801e-14), but rounding up to (5.684341886080802e-14) does work.
// as described in: https://www.exploringbinary.com/the-shortest-decimal-string-that-round-trips-may-not-be-the-nearest/

// for cut points we should always use float64 values since that gives us the best resolution, and our caller
// could have float64 values or float32 values and cuts points that are float64 can work on both of them
// Also, float64 is the most cross-language compatible format, and in JSON it's the only option.

// for scores we should always use float64 values.  Unlike cut points, we'll be using float32 internally within
// the booster, BUT we only need to turn scores into text for serialization to JSON, and JSON only supports
// float64 values, so we need to output that.  We should get 100% reproducibility by turning float32 scores
// into float64, then text, then back to float64, then back to float32, so this is fine.  The only thing
// we loose is a bit of simplicity since our JSON scores will have more digits, but we don't have the equivalent
// of humanized cuts anyways, so the scores will have as many decimals as we get via boosting anywyas

// lastly, since there are no integers in JSON (everything is a double), we should eliminate the difference
// between float64 and integers when numbers can be represented as unique integers.  WE should convert the float
// 4.0 therefore to "4" for any number that meets the criteria: "floor(x) == x && abs(x) <= SAFE_FLOAT64_AS_INT_MAX"

extern IntEbmType GetCountCharactersPerFloat() {
   // for calling FloatsToStrings the caller needs to allocate this many bytes per float in the string buffer
   // after every float is either a space separator or a null-terminator
   return k_cCharsFloatPrint;
}

extern ErrorEbmType FloatsToString(IntEbmType count, const double * values, char * str) {
   // TODO: implement this:
   // 
   // This code takes an array of floats and converts them to a single string separated by spaces and a null-terminator
   // at the end
}

extern ErrorEbmType StringToFloats(const char * str, double * values) {
   // TODO: implement this:
   //
   // This code takes a single string with the floats separated by spaces and a null-terminator at the end
   // and converts these into an array of floats.  The caller had better be carefull in allocating, but they
   // should know how many float values they put into the string so they should know how many values they'll
   // get back and therefore how big to make the buffer
}

#endif

INLINE_RELEASE_UNTEMPLATED static long GetExponent(const char * const str) noexcept {
   // we previously checked that this converted to a long in FloatToFullString
   return strtol(&str[k_iExp + size_t { 1 }], nullptr, int { 10 });
}

static FloatEbmType StringToFloatWithFixup(
   const char * const str,
   const size_t iIdenticalCharsRequired
) noexcept {
   // TODO: this is misguided... python shortens floating point numbers to the shorted string when printing numbers
   // and using nextafter to get to all zeros makes python, and other languages that do the same have a bunch of
   // zeros and a long string.  So, if 1.6999999999999994 rounds to 1.7 nicely in python, don't use nextafter
   // below to convert it to 1.7000000000000009 since then python will need to output the long string
   // to avoid ambiguity since 1.7 will convert to 1.6999999999999994 and not 1.7000000000000009
   // see: https://www.exploringbinary.com/the-shortest-decimal-string-that-round-trips-may-not-be-the-nearest/

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

   if(FloatToFullString(ret, strRehydrate)) {
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
            if(LIKELY(!FloatToFullString(ret, strAvg)) && LIKELY(!StringToFloatChopped(strAvg, 0, &lowChop, &highChop))) {
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

            if(LIKELY(!FloatToFullString(ret, strAvg)) && LIKELY(!StringToFloatChopped(strAvg, 0, &lowChop, &highChop))) {
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

      if(LIKELY(LIKELY(!FloatToFullString(ret, strAvg)) &&
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
      if(LIKELY(LIKELY(!FloatToFullString(low, strLow)) &&
         LIKELY(!FloatToFullString(high, strHigh)) && LIKELY(!FloatToFullString(ret, strAvg)))) {
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
      if(LIKELY(!FloatToFullString(ret, str))) {
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

extern size_t RemoveMissingValuesAndReplaceInfinities(const size_t cSamples, FloatEbmType * const aValues) noexcept {
   EBM_ASSERT(size_t { 1 } <= cSamples);
   EBM_ASSERT(nullptr != aValues);

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

   FloatEbmType * pCopyFrom = aValues;
   FloatEbmType * pCopyTo = aValues;
   const FloatEbmType * const pValuesEnd = aValues + cSamples;
   do {
      FloatEbmType val = *pCopyFrom;
      if(PREDICTABLE(!std::isnan(val))) {
         val = UNPREDICTABLE(std::numeric_limits<FloatEbmType>::infinity() == val) ? 
            std::numeric_limits<FloatEbmType>::max() : val;
         val = UNPREDICTABLE(-std::numeric_limits<FloatEbmType>::infinity() == val) ? 
            std::numeric_limits<FloatEbmType>::lowest() : val;
         *pCopyTo = val;
         ++pCopyTo;
      }
      ++pCopyFrom;
   } while(LIKELY(pValuesEnd != pCopyFrom));
   const size_t cSamplesWithoutMissing = pCopyTo - aValues;
   EBM_ASSERT(cSamplesWithoutMissing <= cSamples);
   return cSamplesWithoutMissing;
}

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION SuggestGraphBounds(
   IntEbmType countCuts,
   FloatEbmType lowestCut,
   FloatEbmType highestCut,
   FloatEbmType minValue,
   FloatEbmType maxValue,
   FloatEbmType * lowGraphBoundOut,
   FloatEbmType * highGraphBoundOut
) {
   // There are a lot of complexities in choosing the graphing bounds.  Let's start from the beginning:
   // - cuts occur on floating point values.  We need to make a choice whether features that are the exact value 
   //   of the cut point go into the upper or lower bounds
   // - we choose lower bound inclusivity so that if a cut is at 5, then the numbers 5.0 and 5.1 will be in the same bound
   // - unfortunately, this means that -1.0 is NOT in the same bounds as -1.1, but negative numbers should be rarer
   //   and in general we shouldn't be putting cuts on whole numbers anyways.  Cuts at 0.5, 1.5, etc are better
   //   and this is where the algorithm will tend to put cuts anyways for whole numbers.
   // - if we wanted -1.0 to be in the same bin as -1.1 AND also have +1.0 be in the same bin as +1.1 there is
   //   an alternative of making 0 it's own bin and using lower bound inclusive for positives, and upper bound 
   //   inclusive for negatives.  We do however want other people to be able to write EBM evaluators and this kind
   //   of oddity is just a bit too odd and non-standard.  It's also hard to optimize this in the binary search.
   // - With lower bound inclusivity, no cut should ever be -inf, since -inf will be included in the upper bound
   //   and nothing can be lower than -inf (NaN means missing)
   // - In theory, +inf cut points might have a use to separate max and +inf feature values, but it's kind of weird 
   //   to disallow cuts at -inf but allow them at +inf, so we declare +inf to be an illegal cut point as well.  
   //   The consequence of this is that the highest legal cut point at max can't separate max from +inf values, 
   //   but we can live with that.  lowest can however separate -inf from lowest, but that's just the consequence 
   //   of our decisions above.
   // - NaN cut points don't make any sense, so in conclusion cuts can be any normal floating point number.
   // - To preserve interpretability, our graphs should initially be displayed over the range from the min value in
   //   the data to the maximum value in the data.  This is a strong indicator of issues in the data if the range
   //   is too wide, and the user can rapidly zoom into the main area if they need to see more detail.  We could
   //   later add some kind of button to zoom to a point that keeps the outer most bounds in view as an option. 
   // - Annother important aspect is that all bins and their scores should be visible in principle on the graphs.
   //   We can't prevent a bin though from being vanishingly small such that it wouldn't be effectively visible, but
   //   it should at least be within the observable range on the graphs.
   // - Under normal circumstances when the user doesn't edit the graphs or provide custom cut points we can offer
   //   an absolute guarantee that when the graphing view goes from the min to the max value that it includes all cut 
   //   points IF the following are true:
   //   - the user has not stripped the min and max value information from the model.  It would be nice to allow 
   //     the user to do this as an option though if they want to for privacy reasons since the min and max are 
   //     potentially egregious violations of privacy (eg: the max net worth in the dataset is $100,000,000,000).  
   //   - The end user doesn't edit the graphs after the fact.  Our automatic binning never put cuts automatically 
   //     above the max or below the min, however if the user later edits graphs they could want to put cuts outside 
   //     of the min/max range.  If the user wants to put a range between 999 and 1001 with a value of +10 and the 
   //     max value in the natural data was 100, then we'll have a cut point outside of the normally displayed 
   //     min -> max range.  The scenario of wanting a previously unseen bin might happen if the data changes after
   //     training and the user wants to correct the model to handle these cases (eg: a sensor fails and the user wants
   //     to give all values scores of 0 in some range where they were previously +3.1), so I believe we should support
   //     these kinds of scenarios of adding bins outside the natural data range.
   //   - The end user didn't supply user defined cut points upfront before fitting.  If the user supplies user 
   //     defined cut points of 1,2,3 and no data is larger than 2.5, then the 3 value cut is above the max
   //   - There is a corner case where the max value is for example 10 and there is a lot of data at 10, but also
   //     there is data one floating point tick below 10 (9.9999999 as a close example).  Since we use lower bound
   //     inclusivity, the cut point will be at 10, and the max value will be at 10, so if our graph goes from the
   //     min value to 10, then the score in the bin ABOVE 10 isn't visible on the graph.  The worse case of this
   //     would occur if one third of the data was at the max float minus one float tick, one third of the data
   //     was at the max, and one third of the data was at +inf.  The only valid cut would be at max, with 1/3 of the
   //     data on the left and 2/3 of the data on the right and the graph bounds ending at max value with no 
   //     possibility to show the upper score bin unless the graph shows beyond the max value and in fact shows 
   //     beyond the max float value.
   //   - There is an odd case, but one that'll happen with regularity on real data where all the feature values
   //     are the same.  For instance a sensor that has never worked and always reports 0.  In this case the
   //     min and max value are both 0, and there are no cuts (for non-editied or pre-specified cuts), but the
   //     interesting aspect is that the range has zero mass since (0 - 0) = 0 and thus the graph doesn't have
   //     a range that is representable as a 2D graph.  I would recommend putting the single value on a graph with
   //     one tick at the specific value and showing no other tick marks on the graph leaving the impression that
   //     the graph has infinite resolution, which it does.
   //   - To handle the scenario where the max and the highest next data value is max minus one tick AND to handle
   //     the more likely scenario where there is only one value, Slicer needs to be able to handle zero width
   //     regions where the upper and lower bound is the same number.  Other than these two special cases though
   //     since cut points should always increase, it should not be possible for the lower and upper bound to be
   //     identical
   // - we could disallow Slicer from having zero width slices (low bound and high bound identical) if we were 
   //   willing to do the following:
   //   - if all the data is identical (eg: all values are 5.0), then we could choose some arbitrary zone to graph
   //     by showing for instance 4.9999999 to 5.0000001 which would be purposely narrow to show that there is
   //     only 1 value in the data, and then we have the width on the graph to show the score in the "bin"
   //   - if we encounter the scenario where the top value and next lower value are separated by a float tick,
   //     we can move the graph bound outwards a bit.  This violates our rule that the graph should initially show
   //     from min to max, but this is an exceptional circumstance.
   //   - alternatively, we could simply remove zero width bins at the top and just accept that this is a truely
   //     exceptional scenario that just doesn't get graphed.  In this case since we want to avoid the zero width
   //     zone in the programming interface it's not just a UI change but we need to filter it out when genereating
   //     the explanation
   //   - If we choose to increment up one tick, since we can't increment up from max, we need to disallow cuts on 
   //     exactly max, so we should therefore throw exceptions if the user specifies max as a cut point and force 
   //     cuts to not choose max on automated cut selection
   //   - disallowing NaN and infinities for cut points is expected, but I think many users would be surprised to
   //     recieve an exception if they specify a cut point at max.  For this reason, I prefer allowing zero
   //     width intervals in Slicer and detecting handling this scenario in the graphs.  I also don't like removing
   //     one Slicer zone if the end slices are zero width since this will be surprizing behaviour to anyone
   //     using our interface and even if they don't get an exception it could lead to a crash bug or worse.
   // - In general, I think that we should allow the user to edit graphs and add bins beyond the original graphing
   //   range because there are valid reasons why the user might want to do this:
   //   - adding special casing after the fact, like if a sensor stops working and the returned value of 1000 should
   //     be ignored, yet this particular odd value never appears in the training data.
   //   - the user will get a surprising to them exception if we disallow editing outside of the min/max range.  Model
   //     evaluation could work during initial model building, but then fail later in a model building pipeline if one 
   //     of the upper bins is rare.  For example if it's rare for data to be above 1000, but the user still wants 
   //     to special case edit models to handle that differently, if in a production environment they just happen to 
   //     get a dataset that doesn't have values above 1000 in it.  I think it shouldn't fail since then the user 
   //     needs to carefully check their data for all kinds of rare exceptional events which they won't know about 
   //     beforehand unless they read our code very carefully to know all the failure cases.  A warning would be a 
   //     far nicer option in these circumstances, which we have enough information to do.
   // - if we allow the user to edit graphs to put cut points beyond the min/max value, then we need to break one of
   //   our two cardinal rules above in those exceptoinal cirumstances where the user edits features:
   //   - If we choose to continue to show the graph between the min and max then
   //     there will be cuts and scores not visible on our graphs, and the data inside Slicer will look very odd
   //     with the last bin having negative width where the lower bound would be the last cut and the upper bound
   //     be the maximum, which is now lower than the upper bound cut
   //   - if we choose to expand the graph viewing area beyond the min and max value, then we can now show the entire
   //     range of valid scores.  We need to though expand BEYOND the outer cut points in this case because there
   //     is a range that extends from the highest cut point until infinity and from the lowest cut point to negative
   //     infinity and those score values won't be shown if our graph terminates at the lowest and highest cut points
   // - Of these two bad options, probably expanding the view of the graph is the least worse since it at least 
   //   notifies the user of an odd/unusual situation with that feature and it allows them to see all values which
   //   wouldn't be possible, and it avoids the issues of really super odd negative width ranges in Slicer.
   // - Under this scenario there are 3 types of values:
   //   - cut points
   //   - min/max value of the data (storing this as is is nice to keep in the model file, and maybe indicate on the graphs)
   //   - the graph low and grpah high values.  We can use these values in Slicer intervals.  The graph low and graph high values
   //     can be made to guarantee that they are always beyond the outside cut points AND either equal to or beyond
   //     the min/max values
   // - slicer should only include the cut points and the graph bounds in the ranges (never the min or max values!).  
   //   We should keep the true min and max values as separate data fields outside of the Slicer ranges since 
   //   otherwise we'll get negative range widths in some bad cases
   // - We don't need to store the graph min and graph max in the model.  We can generate these when we generate
   //   an explanation object since they derive precicely from the cut points and min/max values which we do store
   // - if the user chooses to remove the min/max value for privacy reasons (or any other reason), then we'd use
   //   the same algorithm of putting the graph view just a bit outside of the range of the lower and upper cut points
   // - There is one more wrinkle however.  What if the min is -inf or the max is +inf:
   //   - infinity isn't a JSON compatible value, so we should avoid serializing it as JSON
   //   - it's impossible to make a graph that starts on +-infinity since then all other features are 1/+-inf and
   //     then you can't even zoom into them, even in theory since your range is infinite to begin with
   //   - our options are:
   //     - make +inf as max and -inf as lowest
   //     - write out non-JSON compliant +-inf values and special case the graphs
   //     - store the non-infinity min and non-infinity max AND also keep bools to know if the true min and max were
   //       inf values.  (we should store this as -1/0/+1 for each since the max can be -inf if all data values
   //       are -inf OR if all data values are +inf then the min can be +inf!
   //     - I like the latter option since it might be nice for debugging to know what the min/max values were other
   //       than the +-inf values
   //   - If we choose the latter though of storing the non-infinity min/max, then we need to be careful when we
   //     we choose automatic cut points.  If we replaced +inf with max, and -inf with min for the purposes of
   //     cut point calculation, then we can have a cut point that is outside the min/max range for non-edited and
   //     non-pre-specified cut points.  We can avoid this scenario though by not allowing the automatic cut point 
   //     algorithm to select cut points between the non-infinity min/max values and min/lowest.  We essentially 
   //     ignore the +inf and -inf values beside using their counts.  This does however resolve another kind of 
   //     issue that we can get huge unnatural values if we attempt to put cuts between the 
   //     non-inf min/max and +-inf or max/lowest
   //   - as a nicety, we can avoid the scenario where the upper cut point and the max are the same value for the
   //     automatic binning code by disallowing a cut at the max value.  So if the data consisted of:
   //     4.999999, 5.000000, 5.000001 (assuming these are 1 float tick apart), then 5.000001 would be the max value
   //     and the cutting algorithm would tend to put a cut at 5.000001 in order to separate 5.000000 from 5.000001
   //     but then we'd get a zero width Slicer interval and we wouldn't be able to show the score value for 
   //     5.000001 and beyond since our graph ends at 5.000001, but if we disallow an automatic cut there then
   //     the only cut allowed here would be at 5.000000 with our graph going from 4.999999 to 5.000001, thus
   //     we'd show some region for both the lower and upper bin (exactly 1 float point tick's worth).
   // - consider two clarifying scenarios:
   //   - you have values between -5 and +10 and a single +inf value
   //     - obviously the cut should be below +10, but what should the max be?  We can't really show a range
   //       of +inf since graphs don't work that way.  We could show max_float but that's confusing to end users
   //       and brittle if the number is changed between float/double
   //     - probably using 10 is the right max here.. what else makes sense?
   //   - you have values between -5 and +10 and 1/100 of the data is +inf
   //     - if we set the max to +inf then we can't really make a graph and we can't write to JSON
   //     - if we set the maxValue to max_float then people won't really undrestand why the graph goes unitl 
   //       3.402823466e+38
   //     - if we set the upper cut below 10, then we get to keep it less than the max, but then +10 will be bunched
   //       with +inf and that seems like a bad clusting
   //     - PROBABLY, choosing something like 1000 times the range of the data beyond the max is the right thing
   //       to do.  If someone said the data ranged from 0 to 1 with +inf values too, but then you get a value 
   //       at 10,000, you might wonder if it should go into the 0 to 1 bin or the +inf bin.
   // - since we can't graph to +-inf anyways, we might want to include a field to indicate if there are infinities
   //   at either the min or the max, but it gets complicated if all values are +inf for instance then the min
   //   value is +inf and that makes it confusing.  You'd need to have -1/0/+1 to cover all the -inf/no_inf/+inf cases
   //   so I tend to think it's not worth preserving the information that there are +inf or -inf values in the dataset
   // - generally, max_float and min_float are problematic to graph or reason about.  If we set the bounds to
   //   min_float to max_float then the distance between them goes to infinity since max_float - (min_float) = +inf
   //   Also, people don't generally know that max_float is 3.402823466e+38 so it just confuses them.
   //   Also, if the user tries to zoom past max_float, then the graphing software might fail, and we do want to
   //   allow people to use graphing software other than ours.

//-we want to get the max and min non - infinity values to write in JSON and to graph
//- BUT, if there are a lot of + inf or -inf values we have to decide where to put cut points if automatic cutting is selected
//- the graph is going to go from min to max then it's tempting to want to put the cut points slightly below the max and slightly above the min
//by bunching the max and min values with the + inf or -inf values, but + inf or -inf might be truely special scenarios so
//we'd rather keep them in their own bins if there is enough data for them
//so we'd rather put the cut point above max and below min to separate the max and min
//- but where to put the cut points above max or min.In theory they have a huge range(to infinity!)
//but we can't graph infinity anyways, and making a graph go to 10^38 seems excessive
//- if I asked a human where they'd put a cut if they had data going from 1 to 10 and then +inf and -inf, I think a reasonable answer is that
//if a new value poped in that was above 100 or 1000 times larger than the previous max it would be ambiguous if that should be binned with
//the + inf / -inf values or with the min / max values.So, let's say we go with 100 times larger, that would mean we'd have a cut point at 10 * 100
//= 1000 and -1000 (because our range is
//


   if(nullptr == lowGraphBoundOut) {
      LOG_0(TraceLevelError, "ERROR SuggestGraphBounds nullptr == lowGraphBoundOut");
      return Error_IllegalParamValue;
   }
   if(nullptr == highGraphBoundOut) {
      LOG_0(TraceLevelError, "ERROR SuggestGraphBounds nullptr == highGraphBoundOut");
      return Error_IllegalParamValue;
   }
   if(maxValue < minValue) {
      // silly caller, these should be reversed.  If either or both are NaN this won't execute, which is good
      LOG_0(TraceLevelError, "ERROR SuggestGraphBounds maxValue < minValue");
      *lowGraphBoundOut = 0;
      *highGraphBoundOut = 0;
      return Error_IllegalParamValue;
   }

   if(countCuts <= IntEbmType { 0 }) {
      if(countCuts < IntEbmType { 0 }) {
         LOG_0(TraceLevelError, "ERROR SuggestGraphBounds countCuts < IntEbmType { 0 }");
         *lowGraphBoundOut = 0;
         *highGraphBoundOut = 0;
         return Error_IllegalParamValue;
      }
      // countCuts was zero, so the only information we have to go on are the minValue and maxValue..
      if(std::isnan(minValue)) {
         if(std::isnan(maxValue)) {
            // no cuts and min and max are both unknown, let's return 0 -> 0 since 
            // going from lowest_float -> max_float leads to overflows when you subtract and makes graphing hard
            // and most people don't know what lowest_float and max_float are anyways, and the range also changes
            // depending on if you're using floats or doubles

            *lowGraphBoundOut = 0;
            *highGraphBoundOut = 0;
            return Error_None;
         } 

         // no min value, but we do have a max value?? Ok, well, since we only have one value let's return that
         if(std::isinf(maxValue)) {
            // you can't graph +inf, so return 0 -> 0
            *lowGraphBoundOut = 0;
            *highGraphBoundOut = 0;
            return Error_None;
         }

         *lowGraphBoundOut = maxValue;
         *highGraphBoundOut = maxValue;
         return Error_None;
      } else if(std::isnan(maxValue)) {
         // no max value, but we do have a min value?? Ok, well, since we only have one value let's return that
         if(std::isinf(minValue)) {
            // you can't graph -inf, so return 0 -> 0
            *lowGraphBoundOut = 0;
            *highGraphBoundOut = 0;
            return Error_None;
         }

         *lowGraphBoundOut = minValue;
         *highGraphBoundOut = minValue;
         return Error_None;
      }

      // great, both the min and max are known.  We still don't want to use +-inf values if they are present
      // for the graph bounds.  Normally we woudn't return +-inf, but this is a field which the user might
      // modify, so let's handle +inf or -inf values

      if(std::isinf(minValue)) {
         if(std::isinf(maxValue)) {
            // both are inf values.  You can't graph +-inf, so return 0 -> 0
            *lowGraphBoundOut = 0;
            *highGraphBoundOut = 0;
            return Error_None;
         }
         // minValue is an infinity but maxValue isn't, so let's return that
         *lowGraphBoundOut = maxValue;
         *highGraphBoundOut = maxValue;
         return Error_None;
      } else if(std::isinf(maxValue)) {
         // maxValue is an infinity but minValue isn't, so let's return that
         *lowGraphBoundOut = minValue;
         *highGraphBoundOut = minValue;
         return Error_None;
      }

      *lowGraphBoundOut = minValue;
      *highGraphBoundOut = maxValue;
      return Error_None;
   }

   if(std::isnan(lowestCut) || std::isinf(lowestCut) || std::isnan(highestCut) || std::isinf(highestCut)) {
      LOG_0(TraceLevelError, "ERROR SuggestGraphBounds std::isnan(lowestCut) || std::isinf(lowestCut) || std::isnan(highestCut) || std::isinf(highestCut)");
      *lowGraphBoundOut = 0;
      *highGraphBoundOut = 0;
      return Error_IllegalParamValue;
   }

   // we're going to be checking lowestCut and highestCut, so we should check that they have valid values
   if(IntEbmType { 1 } == countCuts) {
      if(lowestCut != highestCut) {
         LOG_0(TraceLevelError,
            "ERROR SuggestGraphBounds when 1 == countCuts, then lowestCut and highestCut should be identical");
         *lowGraphBoundOut = 0;
         *highGraphBoundOut = 0;
         return Error_IllegalParamValue;
      }
   } else {
      if(highestCut <= lowestCut) {
         LOG_0(TraceLevelError,
            "ERROR SuggestGraphBounds highestCut <= lowestCut");
         *lowGraphBoundOut = 0;
         *highGraphBoundOut = 0;
         return Error_IllegalParamValue;
      }
   }

   bool bExpandLower;
   FloatEbmType lowGraphBound;
   if(std::isnan(minValue)) {
      // the user removed the min value from the model so we need to use the available info, which is the lowestCut
      lowGraphBound = lowestCut;
      bExpandLower = true;
   } else if(-std::numeric_limits<FloatEbmType>::infinity() == minValue) {
      // we can't graph -inf, and don't use lowest since that can lead to graph issues too
      lowGraphBound = lowestCut;
      bExpandLower = true;
   } else {
      if(lowestCut <= minValue) {
         // the model has been edited or supplied with non-data derived cut points
         // our automatic binning code should disallow cuts on the exact min value
         // if equal and we don't expand lower, then there won't be any place on the graph to see the lowest bin score
         lowGraphBound = lowestCut;
         bExpandLower = true;
      } else {
         lowGraphBound = minValue;
         bExpandLower = false;
      }
   }
   EBM_ASSERT(!std::isnan(lowGraphBound));
   EBM_ASSERT(!std::isinf(lowGraphBound));

   bool bExpandHigher;
   FloatEbmType highGraphBound;
   if(std::isnan(maxValue)) {
      // the user removed the max value from the model so we need to use the available info, which is the highestCut
      highGraphBound = highestCut;
      bExpandHigher = true;
   } else if(std::numeric_limits<FloatEbmType>::infinity() == maxValue) {
      highGraphBound = highestCut;
      bExpandHigher = true;
   } else {
      if(maxValue <= highestCut) {
         // the model has been edited or supplied with non-data derived cut points
         // our automatic binning code should disallow cuts on the exact max value
         // if equal and we don't expand higher, then there won't be any place on the graph to see the highest bin score
         highGraphBound = highestCut;
         bExpandHigher = true;
      } else {
         highGraphBound = maxValue;
         bExpandHigher = false;
      }
   }
   EBM_ASSERT(!std::isnan(highGraphBound));
   EBM_ASSERT(!std::isinf(highGraphBound));

   if(lowGraphBound == highGraphBound) {
      // we handled zero cuts above, and if there were two cuts they'd have to have unique increasing values
      // so the only way we can have the low and high graph bounds the same is if we have one cut and both the
      // minValue and maxValue are the same as that cut (otherwise we'd create some space), or they are missing (NaN)
      EBM_ASSERT(IntEbmType { 1 } == countCuts);
      EBM_ASSERT(std::isnan(minValue) || minValue == highGraphBound);
      EBM_ASSERT(std::isnan(maxValue) || maxValue == lowGraphBound);

      // if the regular binning code was kept and the min/max value wasn't removed from the model, then we should
      // not be able to get here, since minValue == maxValue can only happen if there is only one value, and if there
      // is only one value we would never create cut points, so the cut points or min/max have been user edited
      // we can therefore put our bounds outside of the original min/max values.  We'll create a visible bin on the
      // lower side and higher side

      // it's possible that this creates zero sized regions for Slicer if lowGraphBound/highGraphBound was lowest_float
      // or max_float, but as we've covered above, zero width regions should be legal for user defined binning
      *lowGraphBoundOut = std::nextafter(lowGraphBound, std::numeric_limits<FloatEbmType>::lowest());
      *highGraphBoundOut = std::nextafter(highGraphBound, std::numeric_limits<FloatEbmType>::max());
      return Error_None;
   }

   EBM_ASSERT(lowGraphBound < highGraphBound);

   const FloatEbmType scaleMin = highGraphBound - lowGraphBound;
   // scaleMin can be +infinity if highestCut is max and lowestCut is lowest.  We can handle it.
   EBM_ASSERT(!std::isnan(scaleMin));
   // IEEE 754 (which we static_assert) won't allow the subtraction of two unequal numbers to be non-zero
   EBM_ASSERT(FloatEbmType { 0 } < scaleMin);

   // limit the amount of dillution allowed for the tails by capping the relevant cCutPointRet value
   // to 1/32, which means we leave about 3% of the visible area to tail bounds (1.5% on the left and
   // 1.5% on the right)

   const size_t cCutsLimited = static_cast<size_t>(IntEbmType { 32 } < countCuts ? IntEbmType { 32 } : countCuts);

   EBM_ASSERT(size_t { 1 } <= cCutsLimited);
   const size_t denominator = cCutsLimited << 1;
   EBM_ASSERT(size_t { 2 } <= denominator);
   const FloatEbmType movementFromEnds = scaleMin / static_cast<FloatEbmType>(denominator);
   // movementFromEnds can be +infinity if scaleMin is infinity. We can handle it.  It could also underflow to zero
   EBM_ASSERT(!std::isnan(movementFromEnds));
   EBM_ASSERT(FloatEbmType { 0 } <= movementFromEnds);

   if(bExpandLower) {
      lowGraphBound = lowGraphBound - movementFromEnds;
      // lowGraphBound can be -infinity if movementFromEnds is +infinity.  We can handle it.
      EBM_ASSERT(!std::isnan(lowGraphBound));
      EBM_ASSERT(lowGraphBound <= std::numeric_limits<FloatEbmType>::max());
      // GetInterpretableEndpoint can accept -infinity, but it'll return -infinity in that case
      lowGraphBound = GetInterpretableEndpoint(lowGraphBound, movementFromEnds);
      // lowGraphBound can legally be -infinity and we handle this scenario below
      if(-std::numeric_limits<FloatEbmType>::infinity() == lowGraphBound) {
         // in this case the real data has huge magnitudes, so returning lowest is the best solution
         lowGraphBound = std::numeric_limits<FloatEbmType>::lowest();
      }
   }

   if(bExpandHigher) {
      highGraphBound = highGraphBound + movementFromEnds;
      // highGraphBound can be +infinity if movementFromEnds is +infinity.  We can handle it.
      EBM_ASSERT(!std::isnan(highGraphBound));
      EBM_ASSERT(std::numeric_limits<FloatEbmType>::lowest() <= highGraphBound);
      // GetInterpretableEndpoint can accept infinity, but it'll return infinity in that case
      highGraphBound = GetInterpretableEndpoint(highGraphBound, movementFromEnds);
      // highGraphBound can legally be +infinity and we handle this scenario below
      if(std::numeric_limits<FloatEbmType>::infinity() == highGraphBound) {
         // in this case the real data has huge magnitudes, so returning max_float is the best solution
         highGraphBound = std::numeric_limits<FloatEbmType>::max();
      }
   }

   *lowGraphBoundOut = lowGraphBound;
   *highGraphBoundOut = highGraphBound;
   return Error_None;
}

static size_t CountNormal(const size_t cSamples, const double * const aFeatureValues) {
   EBM_ASSERT(1 <= cSamples);
   EBM_ASSERT(nullptr != aFeatureValues);

   size_t cNormal = 0;
   const double * pFeatureValue = aFeatureValues;
   const double * const featureValuesEnd = aFeatureValues + cSamples;
   do {
      const double val = *pFeatureValue;
      if(!std::isnan(val) && !std::isinf(val)) {
         ++cNormal;
      }
      ++pFeatureValue;
   } while(featureValuesEnd != pFeatureValue);
   return cNormal;
}

static double Stddev(const size_t cSamples, const double * const aFeatureValues, const size_t cNormal) {
   EBM_ASSERT(2 <= cSamples);
   EBM_ASSERT(2 <= cNormal);
   EBM_ASSERT(nullptr != aFeatureValues);

   // use Welford's method to calculate stddev
   // https://stackoverflow.com/questions/895929/how-do-i-determine-the-standard-deviation-stddev-of-a-set-of-values
   // https://www.johndcook.com/blog/standard_deviation/

   double m = 0;
   double s = 0;
   size_t k = 0;
   const double multFactor = double { 1 } / static_cast<double>(cNormal);
   const double * pFeatureValue = aFeatureValues;
   const double * const featureValuesEnd = aFeatureValues + cSamples;
   do {
      const double val = *pFeatureValue;
      if(!std::isnan(val) && !std::isinf(val)) {
         ++k;
         const double numerator = val - m;
         m += numerator / static_cast<double>(k);
         s += multFactor * numerator * (val - m);
      }
      ++pFeatureValue;
   } while(featureValuesEnd != pFeatureValue);
   EBM_ASSERT(k == cNormal);
   s = std::sqrt(s);
   return s;
}

static double Mean(const size_t cSamples, const double * const aFeatureValues, const size_t cNormal) {
   EBM_ASSERT(1 <= cSamples);
   EBM_ASSERT(1 <= cNormal);
   EBM_ASSERT(nullptr != aFeatureValues);

   double sum = 0;
   const double * pFeatureValue = aFeatureValues;
   const double * const featureValuesEnd = aFeatureValues + cSamples;
   do {
      const double val = *pFeatureValue;
      if(!std::isnan(val) && !std::isinf(val)) {
         sum += val;
      }
      ++pFeatureValue;
   } while(featureValuesEnd != pFeatureValue);

   EBM_ASSERT(!std::isnan(sum));

   const double cNormalDouble = static_cast<double>(cNormal);
   if(!std::isinf(sum)) {
      return sum / cNormalDouble;
   }

   // ok, maybe we overflowed. Try again but this time divide as we go. This is less accurate and slower, but whatever
   const double cNormalDoubleInv = double { 1 } / cNormalDouble;
   double mean = 0;
   pFeatureValue = aFeatureValues;
   do {
      const double val = *pFeatureValue;
      if(!std::isnan(val) && !std::isinf(val)) {
         mean += val * cNormalDoubleInv;
      }
      ++pFeatureValue;
   } while(featureValuesEnd != pFeatureValue);
   return mean;
}

// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterGetHistogramCutCountParametersMessages = 25;
static int g_cLogExitGetHistogramCutCountParametersMessages = 25;

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GetHistogramCutCount(
   IntEbmType countSamples,
   const double * featureValues,
   IntEbmType strategy
) {
   UNUSED(strategy);

   LOG_COUNTED_N(
      &g_cLogEnterGetHistogramCutCountParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered GetHistogramCutCount: "
      "countSamples=%" IntEbmTypePrintf ", "
      "featureValues=%p, "
      "strategy=%" IntEbmTypePrintf
      ,
      countSamples,
      static_cast<const void *>(featureValues),
      strategy
   );

   if(UNLIKELY(countSamples <= 0)) {
      if(UNLIKELY(countSamples < 0)) {
         LOG_0(TraceLevelWarning, "WARNING GetHistogramCutCount countSamples < 0");
      }
      return 0;
   }
   if(UNLIKELY(IsConvertError<size_t>(countSamples))) {
      LOG_0(TraceLevelWarning, "WARNING GetHistogramCutCount IsConvertError<size_t>(countSamples)");
      return 0;
   }
   const size_t cSamples = static_cast<size_t>(countSamples);
   const size_t cNormal = CountNormal(cSamples, featureValues);

   IntEbmType ret = 0;
   if(size_t { 3 } <= cNormal) {
      const double stddev = Stddev(cSamples, featureValues, cNormal);
      if(double { 0 } < stddev) {
         const double mean = Mean(cSamples, featureValues, cNormal);
         const double cNormalDouble = static_cast<double>(cNormal);
         const double cNormalCubicRootDouble = std::cbrt(cNormalDouble);
         const double multFactor = double { 1 } / cNormalCubicRootDouble / stddev;

         double g1 = 0;
         const double * pFeatureValue = featureValues;
         const double * const featureValuesEnd = featureValues + cSamples;
         do {
            const double val = *pFeatureValue;
            if(!std::isnan(val) && !std::isinf(val)) {
               const double interior = (val - mean) * multFactor;
               g1 += interior * interior * interior;
            }
            ++pFeatureValue;
         } while(featureValuesEnd != pFeatureValue);
         g1 = std::abs(g1);

         const double denom = std::sqrt(double { 6 } * (cNormalDouble - double { 2 }) / ((cNormalDouble + double { 1 }) * (cNormalDouble + double { 3 })));
         const double countSturgesBins = double { 1 } + std::log2(cNormalDouble);
         double countBins = countSturgesBins + std::log2(double { 1 } + g1 / denom);
         countBins = std::ceil(countBins);
         if(std::isnan(countBins) || std::isinf(countBins)) {
            // use Sturges' formula if we have a numeracy issue with our data. countSturgesBins pretty much can't fail
            countBins = std::ceil(countSturgesBins);
         }
         ret = double { FLOAT64_TO_INT_MAX } < countBins ? IntEbmType { FLOAT64_TO_INT_MAX } : static_cast<IntEbmType>(countBins);
         EBM_ASSERT(1 <= ret); // since our formula started from 1 and added
         --ret; // # of cuts is one less than the number of bins
      }
   }

   LOG_COUNTED_N(
      &g_cLogExitGetHistogramCutCountParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Exited GetHistogramCutCount: "
      "ret=%" IntEbmTypePrintf
      ,
      ret
   );

   return ret;
}

} // DEFINED_ZONE_NAME
