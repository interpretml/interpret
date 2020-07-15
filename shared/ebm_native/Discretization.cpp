// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // std::numeric_limits
#include <algorithm> // std::sort
#include <cmath> // std::round
#include <vector> // std::vector (used in std::priority_queue)
#include <queue> // std::priority_queue
#include <stdio.h> // snprintf
#include <set> // std::set
#include <string.h> // strchr, memmove

#include "ebm_native.h"
#include "EbmInternal.h"
#include "Logging.h" // EBM_ASSERT & LOG
#include "RandomStream.h"

constexpr size_t k_SplitExploreDistance = 20;

// keep k_SplitDeleted as -1 since we use this property to make comparisons faster below
constexpr ptrdiff_t k_SplitDeleted = ptrdiff_t { -1 };
constexpr ptrdiff_t k_SplitHigher = ptrdiff_t { -2 };
constexpr ptrdiff_t k_SplitLower = ptrdiff_t { -3 };

constexpr unsigned int k_MiddleSplittingRange = 0x0;
constexpr unsigned int k_FirstSplittingRange = 0x1;
constexpr unsigned int k_LastSplittingRange = 0x2;

struct NeighbourJump final {

   NeighbourJump() = default; // preserve our POD status
   ~NeighbourJump() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   size_t      m_iStartCur;
   size_t      m_iStartNext;
};
static_assert(std::is_standard_layout<NeighbourJump>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<NeighbourJump>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<NeighbourJump>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

struct SplittingRange final {

   SplittingRange() = default; // preserve our POD status
   ~SplittingRange() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   // we divide the space into long segments of unsplittable equal values separated by spaces where we can put
   // splits, which we call SplittingRanges.  SplittingRanges can have zero or more items.  If they have zero
   // splittable items, than the SplittingRange is just there to separate two unsplittable ranges on both sides.
   // The first and last SplittingRanges are special in that they can either have a long range of unsplittable
   // values on the tail end, or not.  If they have a tail consisting of a long range of unsplitable values, then
   // we'll definetly want to have a split point within the tail SplittingRange, but if there is no unsplitable
   // range on the tail end, then having splits within that range is more optional.
   // 
   // If the first few or last few values in the list are unequal, and followed by an unsplittable range, then
   // we put the unequal values into the unsplittable ranger IF there are not enough of them to create a split based
   // on our countMinimumInstancesPerBin value.
   // Example: If countMinimumInstancesPerBin == 3 and the avg bin size is 5, and the list is 
   // (1, 2, 3, 3, 3, 3, 3 | 4, 5, 6 | 7, 7, 7, 7, 7, 8, 9) -> then the only splittable range is (4, 5, 6)

   size_t         m_cSplittableItems; // this can be zero
   FloatEbmType * m_pSplittableValuesStart;

   size_t         m_cUnsplittablePriorItems;
   size_t         m_cUnsplittableSubsequentItems;

   size_t         m_cUnsplittableEitherSideMax;
   size_t         m_cUnsplittableEitherSideMin;

   size_t         m_uniqueRandom;

   size_t         m_cSplitsAssigned;

   FloatEbmType   m_avgRangeWidthAfterAddingOneSplit;

   unsigned int   m_flags;
};
static_assert(std::is_standard_layout<SplittingRange>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<SplittingRange>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<SplittingRange>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

struct SplitPoint final {
   SplitPoint() = default; // preserve our POD status
   ~SplitPoint() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   bool           m_bDeleted; // TODO: replace this with m_pSplitLowerBoundary == nullptr
   bool           m_bSplit; // TODO: replace this with m_pSplitHigherBoundary == nullptr

   FloatEbmType   m_priority;
   size_t         m_uniqueRandom;

   size_t         m_iVal;
   FloatEbmType   m_iValAspirationalFloat;

   ptrdiff_t      m_cSplitMoveThis;

   size_t         m_cSplitLowerBoundary;
   SplitPoint *   m_pSplitLowerBoundary;

   size_t         m_cSplitHigherBoundary;
   SplitPoint *   m_pSplitHigherBoundary;

   INLINE_ALWAYS void SetDeleted(bool bDeleted) {
      m_bDeleted = bDeleted;
   }
   INLINE_ALWAYS bool IsDeleted() {
      return m_bDeleted;
   }
   INLINE_ALWAYS void SetSplit(bool bSplit) {
      m_bSplit = bSplit;
   }
   INLINE_ALWAYS bool IsSplit() {
      return m_bSplit;
   }
};
static_assert(std::is_standard_layout<SplitPoint>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<SplitPoint>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<SplitPoint>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

class CompareSplitPoint final {
public:
   // TODO : check how efficient this is.  Is there a faster way to to this
   INLINE_ALWAYS bool operator() (const SplitPoint * const & lhs, const SplitPoint * const & rhs) const {
      if(UNLIKELY(lhs->m_priority == rhs->m_priority)) {
         return UNPREDICTABLE(lhs->m_uniqueRandom <= rhs->m_uniqueRandom);
      } else {
         return UNPREDICTABLE(lhs->m_priority <= rhs->m_priority);
      }
   }
};

INLINE_ALWAYS size_t CalculateRangesLeft(const FloatEbmType iVal, const FloatEbmType cVals, const size_t cRanges) {
   // our goal is to, as much as possible, avoid having small ranges at the end.  We don't care as much
   // about having long ranges so much as small range since small ranges allow the boosting algorithm to overfit
   // more easily.  This function takes a 

   EBM_ASSERT(2 <= cRanges); // we require there to be at least one range on the left and one range on the right
   EBM_ASSERT(0 <= iVal);
   EBM_ASSERT(iVal <= cVals);
   // provided FloatEbmType is a double, this shouldn't be able to overflow even if we're on a 128 bit computer
   // if FloatEbmType was a float we might be in trouble for extrememly large ranges and iVal values
   //
   // even with numeric instability, we shouldn't end up with a terrible result here since we only get numeric
   // issues if the number of ranges is huge, and we clip on both the low and high ranges below to handle issues
   // where rounding pushes us a bit over the numeric limits
   size_t cLeft = static_cast<size_t>(static_cast<FloatEbmType>(cRanges + 1) * iVal / cVals);
   cLeft = std::max(size_t { 1 }, cLeft); // don't allow zero ranges on the low side
   cLeft = std::min(cLeft, cRanges - 1); // don't allow zero ranges on the high side
   return cLeft;
}

constexpr static char g_pPrintfForRoundTrip[] = "%+.*" FloatEbmTypePrintf;
constexpr static char g_pPrintfLongInt[] = "%ld";
static FloatEbmType FindClean1eFloat(
   const int cCharsFloatPrint,
   char * const pStr,
   const FloatEbmType low, 
   const FloatEbmType high, 
   FloatEbmType val
) {
   // we know that we are very close to 1e[something].  For positive exponents, we have a whole number,
   // which for smaller values is guaranteed to be exact, but for decimal numbers they will all be inexact
   // we could therefore be either "+9.99999999999999999e+299" or "+1.00000000000000000e+300"
   // we just need to check that the number starts with a 1 to be sure that we're the latter

   constexpr int cMantissaTextDigits = std::numeric_limits<FloatEbmType>::max_digits10;
   unsigned int cIterationsRemaining = 100;
   do {
      if(high <= val) {
         // oh no.  how did this happen.  Oh well, just return the high value, which is guaranteed 
         // to split low and high
         break;
      }
      const int cCharsWithoutNullTerminator = snprintf(
         pStr,
         cCharsFloatPrint,
         g_pPrintfForRoundTrip,
         cMantissaTextDigits,
         val
      );
      if(cCharsFloatPrint <= cCharsWithoutNullTerminator) {
         break;
      }
      if(0 == cCharsWithoutNullTerminator) {
         // check this before trying to access the 2nd item in the array
         break;
      }
      if('1' == pStr[1]) {
         // do one last check to verify for sure that we're above val in the end!
         val = low < val ? val : high;
         return val;
      }

      val = std::nextafter(val, std::numeric_limits<FloatEbmType>::max());
      --cIterationsRemaining;
   } while(0 != cIterationsRemaining);
   return high;
}
// checked
INLINE_RELEASE static FloatEbmType GeometricMeanSameSign(const FloatEbmType val1, const FloatEbmType val2) {
   EBM_ASSERT(val1 < 0 && val2 < 0 || 0 <= val1 && 0 <= val2);
   FloatEbmType result = val1 * val2;
   if(UNLIKELY(std::isinf(result))) {
      if(PREDICTABLE(val1 < 0)) {
         result = -std::exp((std::log(-val1) + std::log(-val2)) * FloatEbmType { 0.5 });
      } else {
         result = std::exp((std::log(val1) + std::log(val2)) * FloatEbmType { 0.5 });
      }
   } else {
      result = std::sqrt(result);
      if(PREDICTABLE(val1 < 0)) {
         result = -result;
      }
   }
   return result;
}

// checked
INLINE_ALWAYS constexpr static int CountBase10CharactersAbs(int n) {
   // this works for negative numbers too
   return int { 0 } == n / int { 10 } ? int { 1 } : int { 1 } + CountBase10CharactersAbs(n / int { 10 });
}

// checked
INLINE_ALWAYS constexpr static long MaxReprsentation(int cDigits) {
   return int { 1 } == cDigits ? long { 9 } : long { 10 } * MaxReprsentation(cDigits - int { 1 }) + long { 9 };
}

INLINE_RELEASE static FloatEbmType GetInterpretableCutPointFloat(const FloatEbmType low, const FloatEbmType high) {
   EBM_ASSERT(low < high);
   EBM_ASSERT(!std::isnan(low));
   EBM_ASSERT(!std::isinf(low));
   EBM_ASSERT(!std::isnan(high));
   EBM_ASSERT(!std::isinf(high));

   if(low < FloatEbmType { 0 } && FloatEbmType { 0 } <= high) {
      // if low is negative and high is positive, a natural cut point is zero.  Also, this solves the issue
      // that we can't take the geometric mean of mixed positive/negative numbers.
      return FloatEbmType { 0 };
   }

   // We want to handle widly different exponentials, so the average of 1e10 and 1e20 is 1e15, not 1e20 minus some 
   // small epsilon, so we use the geometric mean instead of the arithmetic mean.
   //
   // Because of floating point inexactness, geometricMean is NOT GUARANTEED 
   // to be (low < geometricMean && geometricMean <= high).  We generally don't return the geometric mean though,
   // so don't check it here.
   const FloatEbmType geometricMean = GeometricMeanSameSign(low, high);

   constexpr int cMantissaTextDigits = std::numeric_limits<FloatEbmType>::max_digits10;

   // Unfortunately, min_exponent10 doesn't seem to include denormal/subnormal numbers, so although it's the true
   // minimum exponent in terms of the floating point exponential representations, it isn't the true minimum exponent 
   // when considering numbers converted to text.  To counter this, we add 1 extra character.  For double numbers, 
   // we're 3 digits in either case, but in the more general scenario we might go from N to N+1 digits, but I think
   // it's really unlikely to go from N to N+2, since in the simplest case that would be a factor of 10 
   // (if the low number was almost N and the high number was just a bit above N+2), and subnormal numbers 
   // shouldn't increase the exponent by that much ever.

   constexpr int cExponentMaxTextDigits = CountBase10CharactersAbs(std::numeric_limits<FloatEbmType>::max_exponent10);
   constexpr int cExponentMinTextDigits = CountBase10CharactersAbs(std::numeric_limits<FloatEbmType>::min_exponent10);
   constexpr int cExponentTextDigits = 
      1 + cExponentMaxTextDigits < cExponentMinTextDigits ? cExponentMinTextDigits : cExponentMaxTextDigits;
       
   // example: "+9.12345678901234567e+300" (this is when 17 == cMantissaTextDigits, the value for doubles)
   // 3 characters for "+9."
   // cMantissaTextDigits characters for the mantissa text
   // 2 characters for "e+"
   // cExponentTextDigits characters for the exponent text
   // 1 characters for null terminator
   constexpr int cCharsFloatPrint = 3 + cMantissaTextDigits + 2 + cExponentTextDigits + 1;
   char str0[cCharsFloatPrint];
   char str1[cCharsFloatPrint];

   // I don't trust that snprintf has 100% guaranteed formats.  Let's trust, but verify the results, 
   // including indexes of characters like the "e" character

   // snprintf says to use the buffer size for the "n" term, but in alternate unicode versions it says # of characters
   // with the null terminator as one of the characters, so a string of 5 characters plus a null terminator would be 6.
   // For char strings, the number of bytes and the number of characters is the same.  I use number of characters for 
   // future-proofing the n term to unicode versions, so n-1 characters other than the null terminator can fill 
   // the buffer.  According to the docs, snprintf returns the number of characters that would have been written MINUS 
   // the null terminator.
   const int cLowCharsWithoutNullTerminator = snprintf(
      str0, 
      cCharsFloatPrint, 
      g_pPrintfForRoundTrip, 
      cMantissaTextDigits, 
      low
   );
   if(0 <= cLowCharsWithoutNullTerminator && cLowCharsWithoutNullTerminator < cCharsFloatPrint) {
      const int cHighCharsWithoutNullTerminator = snprintf(
         str1, 
         cCharsFloatPrint, 
         g_pPrintfForRoundTrip, 
         cMantissaTextDigits, 
         high
      );
      if(0 <= cHighCharsWithoutNullTerminator && cHighCharsWithoutNullTerminator < cCharsFloatPrint) {
         const char * pLowEChar = strchr(str0, 'e');
         if(nullptr == pLowEChar) {
            EBM_ASSERT(false); // we should be getting lower case 'e', but don't trust sprintf
            pLowEChar = strchr(str0, 'E');
         }
         if(nullptr != pLowEChar) {
            const char * pHighEChar = strchr(str1, 'e');
            if(nullptr == pHighEChar) {
               EBM_ASSERT(false); // we should be getting lower case 'e', but don't trust sprintf
               pHighEChar = strchr(str1, 'E');
            }
            if(nullptr != pHighEChar) {
               // use strtol instead of atoi incase we have a bad input.  atoi has undefined behavior if the
               // number isn't representable as an int.  strtol returns a 0 with bad inputs, or LONG_MAX, or LONG_MIN, 
               // which we handle by checking that our final output is within the range between low and high.
               const long lowExp = strtol(pLowEChar + 1, nullptr, 10);
               const long highExp = strtol(pHighEChar + 1, nullptr, 10);
               // strtol can return LONG_MAX, or LONG_MIN on errors.  We need to cleanse these away since they would
               // exceed the length of our print string
               constexpr long maxText = MaxReprsentation(cExponentTextDigits);
               // assert on this above, but don't trust our sprintf in release either
               if(-maxText <= lowExp && lowExp <= maxText && -maxText <= highExp && highExp <= maxText) {
                  const long double lowLongDouble = static_cast<long double>(low);
                  const long double highLongDouble = static_cast<long double>(high);
                  if(lowExp != highExp) {
                     EBM_ASSERT(lowExp < highExp);

                     str0[0] = '1';
                     str0[1] = 'e';

                     const long lowAvgExp = (lowExp + highExp) >> 1;
                     EBM_ASSERT(lowExp <= lowAvgExp);
                     EBM_ASSERT(lowAvgExp < highExp);
                     const long highAvgExp = lowAvgExp + 1;
                     EBM_ASSERT(lowExp < highAvgExp);
                     EBM_ASSERT(highAvgExp <= highExp);

                     // do the high avg exp first since it's guaranteed to exist and be between the low and high
                     // values, unlike the low avg exp which can be below the low value
                     const int cHighAvgExpWithoutNullTerminator = snprintf(
                        &str0[2],
                        cCharsFloatPrint - 2,
                        g_pPrintfLongInt,
                        highAvgExp
                     );
                     if(0 <= cHighAvgExpWithoutNullTerminator && cHighAvgExpWithoutNullTerminator < cCharsFloatPrint - 2) {
                        // unless something unexpected happens in our framework, str0 should be a valid 
                        // FloatEbmType value, which means it should also be a valid long double value
                        // so we shouldn't get a return of 0 for errors
                        //
                        // highAvgExp <= highExp, so e1HIGH is literally the smallest number that can be represented
                        // with the same exponent as high, so we shouldn't get back an overflow result, but check it
                        // anyways because of floating point jitter

                        // lowExp < highAvgExp, so e1HIGH should be larger than low, but check it
                        // anyways because of floating point jitter

                        const long double highExpLongDouble = strtold(str0, nullptr);

                        if(lowExp + 1 == highExp) {
                           EBM_ASSERT(lowAvgExp == lowExp);
                           // 1eLOW can't be above low since it's literally the lowest value with the same exponent
                           // as our low value.  So, skip all the low value computations

                        only_high_exp:
                           if(lowLongDouble < highExpLongDouble && highExpLongDouble <= highLongDouble) {
                              // we know that highExpLongDouble can be converted to FloatEbmType since it's
                              // between valid FloatEbmTypes, our low and high values.
                              const FloatEbmType highExpFloat = static_cast<FloatEbmType>(highExpLongDouble);
                              return FindClean1eFloat(cCharsFloatPrint, str0, low, high, highExpFloat);
                           } else {
                              // fallthrough case.  Floating point numbers are inexact, so perhaps if they are 
                              // separated by 1 epsilon or something like that and/or the text conversion isn't exact, 
                              // we could get a case where this might happen
                           }
                        } else {
                           const int cLowAvgExpWithoutNullTerminator = snprintf(
                              &str0[2],
                              cCharsFloatPrint - 2,
                              g_pPrintfLongInt,
                              lowAvgExp
                           );
                           if(0 <= cLowAvgExpWithoutNullTerminator && cLowAvgExpWithoutNullTerminator < cCharsFloatPrint - 2) {
                              EBM_ASSERT(lowExp < lowAvgExp);
                              EBM_ASSERT(lowAvgExp < highExp);

                              // unless something unexpected happens in our framework, str0 should be a valid 
                              // FloatEbmType value, which means it should also be a valid long double value
                              // so we shouldn't get a return of 0 for errors
                              //
                              // lowAvgExp is above lowExp and below lowAvgExp, which are both valid FloatEbmTypes
                              // so str0 must contain a valid number that is convertable to FloatEbmTypes
                              // but check this anyways incase there is floating point jitter

                              const long double lowExpLongDouble = strtold(str0, nullptr);

                              if(lowLongDouble < lowExpLongDouble && lowExpLongDouble <= highLongDouble) {
                                 // We know that lowExpLongDouble can be converted now to FloatEbmType since it's
                                 // between valid our low and high FloatEbmType values.
                                 const FloatEbmType lowExpFloat = static_cast<FloatEbmType>(lowExpLongDouble);
                                 if(lowLongDouble < highExpLongDouble && highExpLongDouble <= highLongDouble) {
                                    // we know that highExpLongDouble can be converted now to FloatEbmType since it's
                                    // between valid FloatEbmType, our low and high values.
                                    const FloatEbmType highExpFloat = static_cast<FloatEbmType>(highExpLongDouble);

                                    // take the one that is closest to the geometric mean
                                    //
                                    // we want to compare in terms of exponential distance, so instead of subtacting,
                                    // divide these.  Flip them so that the geometricMean is at the bottom of the low
                                    // one because it's expected to be bigger than the lowExpFloat (the lowest of all
                                    // 3 numbers)
                                    const FloatEbmType lowRatio = lowExpFloat / geometricMean;
                                    const FloatEbmType highRatio = geometricMean / highExpFloat;
                                    // we flipped them, so higher numbers (closer to 1) are bad.  We want small numbers
                                    if(lowRatio < highRatio) {
                                       return FindClean1eFloat(cCharsFloatPrint, str0, low, high, lowExpFloat);
                                    } else {
                                       return FindClean1eFloat(cCharsFloatPrint, str0, low, high, highExpFloat);
                                    }
                                 } else {
                                    return FindClean1eFloat(cCharsFloatPrint, str0, low, high, lowExpFloat);
                                 }
                              } else {
                                 goto only_high_exp;
                              }
                           } else {
                              EBM_ASSERT(false); // this shouldn't happen, but don't trust sprintf
                           }
                        }
                     } else {
                        EBM_ASSERT(false); // this shouldn't happen, but don't trust sprintf
                     }
                  } else {
                     EBM_ASSERT('+' == str0[0] || '-' == str0[0]);
                     EBM_ASSERT('+' == str1[0] || '-' == str1[0]);
                     EBM_ASSERT(str0[0] == str1[0]);

                     // there should somewhere be an 'e" or 'E' character, otherwise we wouldn't have gotten here,
                     // so there must at least be 1 character
                     size_t iChar = 1;
                     // we shouldn't really need to take the min value, but I don't trust floating point number text
                     const size_t iCharEnd = std::min(pLowEChar - str0, pHighEChar - str1);
                     // handle the virtually impossible case of the string starting with 'e' by using iChar < iCharEnd
                     while(LIKELY(iChar < iCharEnd)) {
                        // "+9.1234 5 678901234567e+300" (low)
                        // "+9.1234 6 546545454545e+300" (high)
                        if(UNLIKELY(str0[iChar] != str1[iChar])) {
                           // we know our low value is lower, so this digit should be lower
                           EBM_ASSERT(str0[iChar] < str1[iChar]);
                           // nothing is bigger than '9' for a single digit, so the low value can't be '9'
                           EBM_ASSERT('9' != str0[iChar]);
                           char * pDiffChar = str0 + iChar;
                           memmove(
                              pDiffChar + 1,
                              pLowEChar,
                              static_cast<size_t>(cLowCharsWithoutNullTerminator) - (pLowEChar - str0) + 1
                           );

                           const char charEnd = str1[iChar];
                           char curChar = *pDiffChar;
                           FloatEbmType ret = FloatEbmType { 0 }; // this value should never be used
                           FloatEbmType bestRatio = std::numeric_limits<FloatEbmType>::lowest();
                           char bestChar = 0;
                           do {
                              // start by incrementing the char, since if we chop off trailing digits we won't
                              // end up with a number higher than the low value
                              ++curChar;
                              *pDiffChar = curChar;
                              const long double valLongDouble = strtold(str0, nullptr);
                              if(lowLongDouble < valLongDouble && valLongDouble <= highLongDouble) {
                                 // we know that valLongDouble can be converted to FloatEbmType since it's
                                 // between valid FloatEbmTypes, our low and high values.
                                 const FloatEbmType val = static_cast<FloatEbmType>(valLongDouble);
                                 const FloatEbmType ratio = 
                                    geometricMean < val ? geometricMean / val: val / geometricMean;
                                 EBM_ASSERT(ratio <= FloatEbmType { 1 });
                                 if(bestRatio < ratio) {
                                    bestRatio = ratio;
                                    bestChar = curChar;
                                    ret = val;
                                 }
                              }
                           } while(charEnd != curChar);
                           if(std::numeric_limits<FloatEbmType>::max() != bestRatio) {
                              // once we have our value, try converting it with printf to ensure that it gives 0000s 
                              // at the end (where the text will match up), instead of 9999s.  If we get this, then 
                              // increment the floating point with integer math until it works.

                              // restore str0 to the best string available
                              *pDiffChar = bestChar;

                              unsigned int cIterationsRemaining = 100;
                              do {
                                 int cCheckCharsWithoutNullTerminator = snprintf(
                                    str1,
                                    cCharsFloatPrint,
                                    g_pPrintfForRoundTrip,
                                    cMantissaTextDigits,
                                    ret
                                 );
                                 if(cCheckCharsWithoutNullTerminator < 0 || 
                                    cCharsFloatPrint <= cCheckCharsWithoutNullTerminator) 
                                 {
                                    break;
                                 }
                                 size_t iFindChar = 0;
                                 while(true) {
                                    if(LIKELY(iChar < iFindChar)) {
                                       // all seems good.  We examined up until what was the changing char
                                       return ret;
                                    }
                                    if(str0[iFindChar] != str1[iFindChar]) {
                                       break;
                                    }
                                    ++iFindChar;
                                 }
                                 ret = std::nextafter(ret, std::numeric_limits<FloatEbmType>::max());
                                 --cIterationsRemaining;
                              } while(0 != cIterationsRemaining);
                           }
                           break; // this shouldn't happen, but who knows with floats
                        }
                        ++iChar;
                     }
                     // we should have seen a difference somehwere since our low should be lower than our high,
                     // and we used enough digits for a "round trip" guarantee, but whatever.  Just fall through
                     // and handle it like other close numbers where we just take the geometric mean
                     EBM_ASSERT(false); // this shouldn't happen, but don't trust sprintf
                  }
               } else {
                  EBM_ASSERT(false); // this shouldn't happen, but don't trust sprintf
               }
            } else {
               EBM_ASSERT(false); // this shouldn't happen, but don't trust sprintf
            }
         } else {
            EBM_ASSERT(false); // this shouldn't happen, but don't trust sprintf
         }
      } else {
         EBM_ASSERT(false); // this shouldn't happen, but don't trust sprintf
      }
   } else {
      EBM_ASSERT(false); // this shouldn't happen, but don't trust sprintf
   }
   // something failed, probably due to floating point inexactness.  Let's first try and see if the 
   // geometric mean will work
   if(low < geometricMean && geometricMean <= high) {
      return geometricMean;
   }

   // For interpretability reasons, our digitization puts numbers that are exactly equal to the cut point into the 
   // higher bin. This keeps 2 in the (2, 3] bin if the cut point is 2, so that 2 is lumped in with 2.2, 2.9, etc
   // 
   // We should never reall get to this point in the code, except perhaps in exceptionally contrived cases, like 
   // perahps if two floating poing numbers were separated by 1 epsilon.
   return high;
}

INLINE_RELEASE static void IronSplits() {
   // - TODO: POST-HEALING
   //   - after fitting these, we might want to jigger the final results.  We would do this by finding the smallest 
   //     section and trying to expand it either way.  Each side we'd push it only enough to make things better.
   //     If we find that we can make a push that improves things, then we take that.  We'd need a priority queue to 
   //     indicate the smallest sections
   //

   // TODO: here we should try to even out our final result in case there are large scale differences in size
   //       that we can address by pushing our existing cuts arround by small amounts
}

#ifdef NEVER
constexpr unsigned int cNeighbourExploreDistanceMax = 5;
constexpr size_t illegalIndex = std::numeric_limits<size_t>::max();

INLINE_RELEASE static void BuildNeighbourhoodPlan(
   const size_t cMinimumInstancesPerBin,
   const size_t iValuesStart,
   const size_t cSplittableItems,
   const NeighbourJump * const aNeighbourJumps,

   const size_t iValLower,
   const FloatEbmType iValAspirationalLowerFloat,
   const size_t cRangesLower,

   const size_t iValHigher,
   const FloatEbmType iValAspirationalHigherFloat,
   const size_t cRangesHigher,

   SplitPoint * const pSplitCur
) {
   EBM_ASSERT(1 <= cMinimumInstancesPerBin);
   EBM_ASSERT(nullptr != aNeighbourJumps);

   EBM_ASSERT(!pSplitCur->IsDeleted());
   EBM_ASSERT(!pSplitCur->IsSplit());

   EBM_ASSERT(1 <= cRangesLower);
   EBM_ASSERT(illegalIndex == iValLower || static_cast<FloatEbmType>(iValLower) == iValAspirationalLowerFloat);

   EBM_ASSERT(1 <= cRangesHigher);
   EBM_ASSERT(illegalIndex == iValHigher || static_cast<FloatEbmType>(iValHigher) == iValAspirationalHigherFloat);

   const FloatEbmType cMinimumInstancesPerBinFloat = static_cast<FloatEbmType>(cMinimumInstancesPerBin);

   FloatEbmType totalDistance;
   if(illegalIndex != iValLower && illegalIndex != iValHigher) {
      totalDistance = static_cast<FloatEbmType>(iValHigher - iValLower);
   } else {
      totalDistance = iValAspirationalHigherFloat - iValAspirationalLowerFloat;
   }
   size_t cRanges = cRangesLower + cRangesHigher;
   const FloatEbmType iValAspirationalRelativeFloat = totalDistance / static_cast<FloatEbmType>(cRanges);

   const FloatEbmType iLandingValFloat = iValAspirationalLowerFloat + iValAspirationalRelativeFloat;
   EBM_ASSERT(FloatEbmType { 0 } <= iLandingValFloat);
   size_t iLandingVal = static_cast<size_t>(iLandingValFloat);
   if(UNLIKELY(cSplittableItems <= iLandingVal)) {
      // handle the very very unlikely situation where m_iAspirationalFloat rounds up to 
      // cSplittableItems due to floating point issues
      iLandingVal = cSplittableItems - 1;
   }
   const NeighbourJump * const pNeighbourJump = &aNeighbourJumps[iValuesStart + iLandingVal];
   EBM_ASSERT(iValuesStart <= pNeighbourJump->m_iStartCur);
   EBM_ASSERT(iValuesStart < pNeighbourJump->m_iStartNext);
   EBM_ASSERT(pNeighbourJump->m_iStartCur < pNeighbourJump->m_iStartNext);
   EBM_ASSERT(pNeighbourJump->m_iStartNext <= iValuesStart + cSplittableItems);
   EBM_ASSERT(pNeighbourJump->m_iStartCur < iValuesStart + cSplittableItems);

   const size_t iValLowChoice = aNeighbourJumps[iValuesStart + iLandingVal].m_iStartCur - iValuesStart;
   const size_t iValHighChoice = aNeighbourJumps[iValuesStart + iLandingVal].m_iStartNext - iValuesStart;

   const size_t cRangesLowerLower = CalculateRangesLeft(iValLowChoice, totalDistance, cRanges);
   const size_t cRangesLowerHigher = cRanges - cRangesLowerLower;

   bool bCanSplitLow = true;
   if(illegalIndex == iValLower) {
      if(static_cast<FloatEbmType>(iValLowChoice) - iValAspirationalLowerFloat < cMinimumInstancesPerBinFloat) {
         bCanSplitLow = false;
      }
   } else {
      if(iValLowChoice - iValLower < cMinimumInstancesPerBin) {
         bCanSplitLow = false;
      }
   }

   FloatEbmType avgLengthLowBest;
   size_t cModifyDirectionLowBest = 0;
   if(bCanSplitLow) {
      FloatEbmType distanceLowFloat;
      FloatEbmType distanceHighFloat;

      if(illegalIndex == iValLower) {
         distanceLowFloat = static_cast<FloatEbmType>(iValLowChoice) - iValAspirationalLowerFloat;
      } else {
         distanceLowFloat = static_cast<FloatEbmType>(iValLowChoice - iValLower);
      }

      if(illegalIndex == iValHigher) {
         distanceHighFloat = static_cast<FloatEbmType>(iValAspirationalHigherFloat)
            - static_cast<FloatEbmType>(iValLowChoice);
      } else {
         distanceHighFloat = static_cast<FloatEbmType>(iValHigher - iValLowChoice);
      }

      int choices = static_cast<int>((unsigned int { 1 } << (unsigned int { 1 } + cNeighbourExploreDistanceMax)) 
         - unsigned int { 1 });
      do {
         // use the bits to go left or right


         --choices;
      } while(0 <= choices);

   }












   const size_t cRangesHigherLower = CalculateRangesLeft(iValHighChoice, totalDistance, cRanges);
   const size_t cRangesHigherHigher = cRanges - cRangesHigherLower;

















}

INLINE_RELEASE static size_t SplitSegment(
   std::set<SplitPoint *, CompareSplitPoint> * const pBestSplitPoints,

   const size_t cMinimumInstancesPerBin,

   const size_t iValuesStart,
   const size_t cSplittableItems,
   const NeighbourJump * const aNeighbourJumps,

   // prior to calling this function, the caller must remove the endpoints if any
   const size_t cCENTERSplitsAssigned,
   // for efficiency we include space for the end point cuts even if they don't exist
   SplitPoint * const aSplitsWithENDPOINTS
) {
   // TODO: someday, for performance, it might make sense to use a non-allocating tree, like:
   //       https://github.com/attractivechaos/klib/blob/master/kavl.h

   // TODO: try to use integer math for indexes instead of floating point numbers which introduce inexactness, and they break down above 2^52 where
   //   // individual integers can no longer be represented by a double
   //   // use integer math and integer based fractions (this is also slighly faster)



   // this function assumes that there will either be splits on either sides of the ranges OR that we're 
   // at the end of a range.  We don't handle the special case of there only being 1 split in a range that needs
   // to be chosen.  That's a special case handled elsewhere

   EBM_ASSERT(nullptr != pBestSplitPoints);
   EBM_ASSERT(pBestSplitPoints->empty());

   EBM_ASSERT(1 <= cMinimumInstancesPerBin);
   // we need to be able to put down at least one split not at the edges
   EBM_ASSERT(2 <= cSplittableItems / cMinimumInstancesPerBin);
   EBM_ASSERT(nullptr != aNeighbourJumps);
   EBM_ASSERT(1 <= cCENTERSplitsAssigned);
   EBM_ASSERT(cCENTERSplitsAssigned < cSplittableItems / cMinimumInstancesPerBin);
   EBM_ASSERT(nullptr != aSplitsWithENDPOINTS);

   const size_t iTop = cCENTERSplitsAssigned + 1;

   aSplitsWithENDPOINTS[0].m_iValActualized = 0;
   aSplitsWithENDPOINTS[iTop].m_iValActualized = cSplittableItems;

   const FloatEbmType stepInit = static_cast<FloatEbmType>(cSplittableItems) / static_cast<FloatEbmType>(iTop);
   for(size_t i = 1; i < iTop; ++i) {
      aSplitsWithENDPOINTS[i].m_iValActualized = k_SplitDeleted;
      aSplitsWithENDPOINTS[i].m_iValAspirationalFloat = i * stepInit;
   }

   size_t iBehindSplitPointInit = 0;
   size_t iAheadSplitPointInit = 1 + k_SplitExploreDistance < iTop ? 1 + k_SplitExploreDistance : iTop;

   for(size_t i = 1; i < iTop; ++i) {
      const size_t iBehindDist = i - iBehindSplitPointInit;
      const size_t iAheadDist = iAheadSplitPointInit - i;
      const SplitPoint * const pSplitBehind = &aSplitsWithENDPOINTS[iBehindSplitPointInit];
      const SplitPoint * const pSplitAhead = &aSplitsWithENDPOINTS[iAheadSplitPointInit];

      const ptrdiff_t iActualBehind = pSplitBehind->m_iValActualized;
      const FloatEbmType iBehindValue = iActualBehind < 0 ?
         pSplitBehind->m_iValAspirationalFloat :
         static_cast<FloatEbmType>(iActualBehind);

      const ptrdiff_t iActualAhead = pSplitAhead->m_iValActualized;
      const FloatEbmType iAheadValue = iActualAhead < 0 ?
         pSplitAhead->m_iValAspirationalFloat :
         static_cast<FloatEbmType>(iActualAhead);

      SplitPoint * const pSplitCur = &aSplitsWithENDPOINTS[i];

      // we generated this floating point index, so it shouldn't be possible to overflow into a size_t
      const size_t iLanding = static_cast<size_t>(pSplitCur->m_iValAspirationalFloat);

      const size_t iLowChoice = aNeighbourJumps[iValuesStart + iLanding].m_iStartCur;
      const size_t iHighChoice = aNeighbourJumps[iValuesStart + iLanding].m_iStartNext;

      const FloatEbmType avgLengthBehind = (iLowChoice - iBehindValue) / iBehindDist;
      const FloatEbmType avgLengthAhead = (iAheadValue - iHighChoice) / iAheadDist;

      const bool bBehind = avgLengthBehind < avgLengthAhead;

      // we want to settle the most obvious ones first.  The most obvious ones are the ones that are super
      // good on one side, but super bad on the other.  If both sides are easy, or both sides are hard, then
      // let's leave those for later

      pSplitCur->m_priority = bBehind ? avgLengthAhead - avgLengthBehind : avgLengthBehind - avgLengthAhead;
      pSplitCur->m_iValActualized = bBehind ? k_SplitLower : k_SplitHigher;

      pBestSplitPoints->insert(pSplitCur);
   }

   const FloatEbmType cMinimumInstancesPerBinFloat = static_cast<FloatEbmType>(cMinimumInstancesPerBin);

   do {
      // We've located our desired split points previously.  Sometimes those desired split points
      // are placed in the bulk of a long run of identical values and we have to decide if we'll be putting
      // the split at the start or the end of those long run of identical values.
      //
      // Before this function in the call stack, we do some expensive exploration of the hardest split point placement
      // decisions that we need to make.  We do a full exploration of both the lower and higher placements of the long 
      // runs, but that grows at O(2^N), so we need to limit this full exploration to just a few choices
      //
      // This function being lower in the stack needs to decide whether to place the split at the lower or higher
      // position without the benefit of an in-depth exploration of both options across all possible other cut points 
      // (local exploration is still ok though).  We need to choose one side and live with that decision, so we 
      // look at all our potential splits, and we greedily pick out the split that is 
      // really nice on one side, but really bad on the other, and we keep greedily picking splits this way until they 
      // are all selected.  We use a priority queue to efficiently find the most important split at any given time.
      // Now we have an O(N * log(N)) algorithm in principal, but it's still a bit worse than that.
      //
      // After we decide whether to put the split at the start or end of a run, we're actualizing the location of 
      // the split and we'll be changing the size of the runs to our left and right since they'll either have
      // actualized or desired split points, or the immutable ends as neighbours.  We'd prefer to spread out the
      // movement between our desired and actual split points into all our potential neighbours instead of to the 
      // immediately bordering ranges.  Ideally, we'd like to spread out and re-calculate all other split points 
      // until we reach the immovable boundaries of an already decided split, or the ends, after we've decided 
      // on each split. So, if we had 255 splits, we'd choose one, then re-calculate the split points of the remaining 
      // 254, but that is clearly bad computationally, since then our algorithm would be O(N^2 * log(N)).  For low 
      // numbers like 255 it might be fine, but our user could choose much larger numbers of splits, and then 
      // it would become intractable.
      //
      // Instead of re-calculating all remaining 255 split points though, maybe we can instead choose a window 
      // of influence.  So, if our influence window was set to 50 split points, then even if we had to move one 
      // split point by a large amount of almost a complete split range, we'd only impact the neighboring 50 ranges 
      // by 2% (1/50).
      //
      // After we choose whether to go to the start or the end, we then choose an anchor point 50 to the 
      // left and another one 50 to the right, unless we hit a materialized split point, or the end, which we can't 
      // move.  All the 50 items to the left and right either grow a bit smaller, or a bit bigger anchored to 
      // the influence region ends.
      //
      // After calculating the new sizes of the ranges and the new desired split points, we can then remove
      // the 50-ish items on both sides from our priority queue, which in fact needs to be a tree so that we can
      // remove items that aren't just the lowest value, and we can re-add them to the tree with their new
      // recalculated priority score.
      //
      // But the scores of the desired split points outside of our window have changed slightly too!  
      // The 50th, 51st, 52nd, etc, items to the left and the right weren't moved, but they can still "see" split 
      // points that are within our influence window of desired split points that we changed, so their priorty 
      // scores need to change.  Once we get to twice the window size though, the items beyond that can't be 
      // affected, so we only need to update items within a four time range of our window size, 
      // two on the left and two on the right.
      //
      // Initially we place all our desired split points equidistant from each other within a splitting range.  
      // After one split has been actualized though, we find that there are different distances between desired 
      // split point, which is a consequence of our choosing a tractable algorithm that uses windows of influence. 
      // Let's say our next split point that we extract from the priority queue was at the previous influence boundary
      // split point.  In this case, the ranges to the left and right are sized differently.  Let's say that our influence
      // windows for this new split point are 50 in both directions.  We have two options for choosing where to
      // actualize our split point.  We could equally divide up the space between our influence window boundaries
      // such that our actualized split point is as close to the center as possible, BUT then we'd be radically
      // departing from the priority that we inserted ourselves into the priority queue with.  We also couldn't
      // have inserted ourselves into the priority queue with equidistant ranges on our sides, since then we'd need
      // to cascade the range lengths all the way to the ends, thus giving up the non N^2 nature of of our window
      // of influence algorithm.  I think we need to proportionally move our desired split point to the actual split
      // point such that we don't radically change the location beyond the edges of the range that we fall inside
      // if we alllowed more radical departures in location then our priority queue would loose meaning.
      // 
      // By using proportional adjustment we keep our proposed split point within the range that it fell under 
      // when we generated a priority score, since ottherwise we'd randomize the scores too much and be pulling 
      // out bad choices.  We therefore
      // need to keep the proportions of the ranges relative to their sizes at any given time.  If we re-computed
      // where the split point should be based on a new 50 item window, then it could very well be placed
      // well outside of the range that it was originally suposed to be inside.  It seems like getting that far out
      // of the priority queue score would be bad, so we preseve the relative lengths of the ranges and always
      // find ourselves falling into the range we were considering.
      //
      // If we have our window set to larger than the number of splits, then we'll effectively be re-doing all
      // the splits, which might be ok for small N.  In that case all the splits would always have the same width
      // In our modified world, we get divergence over time, but since we're limiiting our change to a small
      // percentage, we shouldn't get too far out of whack.  Also, we'll quickly put down actualized splitting points such
      // that afer a few we'll probably find ourselves close to a previous split point and we'll proceed by
      // updating all the priority scores exactly since we'll hit the already decided splits before the 
      // influence window length
      //
      // Our priority score will probably depend on the smallest range in our range window, but then we'd need to
      // maintain a sliding window that knows the smallest range within it.  This requries a tree to hold
      // the lengths, and we'd maintain the window as we move, so if we were moving to the left, we'd add one
      // item to the window from the left, and remove the item to the right.  But if the item to the right is
      // the lowest value, we'd need to scan the remaining items, unless we use a tree to keep track.
      // It's probably reasonably effective though to just use the average smallest range that we calculate from 
      // the right side minus the left divided by the number of ranges.  It isn't perfect, but it shouldn't get too
      // bad, and the complexity is lower so it would allow us to do our calculations faster, and therefore allows
      // more exploratory forays in our caller above before we hit time limits.

      // initially, we have pre-calculated which direction each split should go, and we've calculated how many
      // cut points should move between our right and left sides, and we also previously calculated a priority for
      // making decisions. When we pull one potential split point off the queue, we need to nuke all our decisions
      // within the 50 item window on both sides (or until we hit an imovable boundary) and then we need to
      // recalculate for each split which way it should go and what it's priority is
      //
      // At this point we're re-doing our cuts within the 50 item cut window and we need to decide two things:
      //   1) calculate the direction we'd go for each new cut point, and how many cuts we'd move from our right 
      //      and left to the other side
      //   2) Calculate the priority of making the decions
      //
      // For each range we can go either left or right and we need to choose.  We should try and make decions
      // based on our local neighbourhood, so what we can do is start by assuming we go left, and then chop
      // up the spaces to our left boundary equally.  We can then know our desired step distance so we can go and
      // examine what our close neighbours look like within a smaller window.  We can try going left and right for each
      // of our close neighbours and see if we'll need to make hard decisions.  We can try out the combinations by
      // using a 32 bit number (provided we're exploring less than 2^32 options) and then let each bit represent
      // going left or right for a position at the index.  We can then increment the number and re-simulate various
      // nearby options until we find the one that has the best outcome (the one with the biggest smallest range with
      // the least dropage of cuts).  This represents a possible/reasonable outcome.  We know our neighbours will
      // do the same.  Even though they won't end up with the same cut points they'll at least have an available reasonable
      // choice if we make one available to them.
      //
      // Ok, so each cut point we've examined our neighbours and selected a right/left decison that we can live with
      // for ourselves.  Each cut point does this independently.  We can then maybe do an analysis to see if our
      // ideas for the neighbours match up with theirs and do some jiggering if the outcome within a window is bad
      //
      // Ok, so now we've computed our aspirational cut points, and decided where we'd go for each cut point if we
      // were forced to select a cut point now.  We now need to calculate the PRIORITY for all our cut points
      // 
      // Initially we have a lot of options when deciding cuts.  As time goes on, we get less options.  We want our
      // first cut to minimize the danger that later cuts will force us into a bad position.  

      // When calculating the pririty of a cut point a couple of things come to mind:
      //   - if we have a large open space of many un-materialized cuts, an aspiration cut in the middle that falls 
      //     into a big range is not a threat yet since we can deftly avoid it by changing by small amounts the
      //     aspirational cuts on both ends, BUT if someone puts down a ham-fisted cut right down next to it then
      //     we'll have a problem
      //   - even if the cuts in the center are good, a cut in the center next to a large range of equal values could
      //     create a problem for us easily
      //   - our cut materializer needs to be smart and examine the local space before it finalizes a cut, so that
      //     we avoid the largest risks around putting down ham-fisted cuts.  So, with this combination we can relax
      //     and not worry about the asipirational cuts in the middle of a large un-materialized section, whether
      //     they fall onto a currently bad cut or not
      //   - the asipirational cuts near the boundaries of materialized cuts are the most problematic, especially
      //     if they currently happen to be hard decision cuts.  We probably want to make the hard decisions early
      //     when we have the most flexibility to address them
      //   - so, our algorithm will tend to first materialize the cuts near existing boundaries and move inwards as
      //     spaces that were previously not problems become more constrained
      //   - we might or might not want to include results from our decisions about where we'll put cuts.  For
      //     instance, let's say a potential cut point has one good option and one terrible option.  We may want
      //     to materialize the good option so that a neighbour cut doesn't force us to take the terrible option
      //     But this issue is reduced if before we materialize cuts we do an exploration of of the local space to
      //     avoid hurting neighbours.
      //   - in general, because cuts tend to disrupt many other aspirational cuts, we should probably weigh the exact
      //     cut plan less and concentrate of making the most disruptive cuts first.  We might find that our carefully
      //     crafted plan for future cuts is irrelevant and we no longer even make cuts on ranges that we thought
      //     were important previously.
      //   - by choosing the most disruptive cuts first, we'll probably get to a point quickly were most of our
      //     remaining potential cuts are non-conrovertial.  All the hard decisions will be made early and we'll be
      //     left with the cuts that jiggle the remaining cuts less.
      //
      // If we do a full local exploration of where we're going to do our cuts for any single cut, then we can
      // do a better job at calculating the priority, since we'll know how many cuts will be moved from right to left
      //
      // When doing a local exploration, examine going right left on each N segments to each side (2^N explorations)
      // and then re-calculate the average cut point length on the remaining open space beyond the last cut
      //
      // When we're making a cut decision, we recalculate the aspirational position and intended direction of N cuts
      // within a window, but we make NO changes to aspiration cuts OR intended direction outside of that window
      // since those changes might change the priority, and we want to limit the number of cuts we modify
      // we still need to travel 2 * N cuts from our position.  The first N are ones we change, and the next N can
      // see the changes that we made within our window

      // If we push aspirational cuts from our left to right, we don't change the window bounds when that happens
      // because if we did, then there would be no bounds on where we can 100% guarantee that no changes will affect
      // outside regions
      //
      // Ok, so our last step should be to re-calcluate all the priorities.  We do this since we might have a
      // non-linear step in between where we harmonize through a non-sinlge-individual cut process where cuts will
      // happen.  Our non-single-individual cut process should happen after a single pass where we place the cuts
      // without detailed information about where are neighbours want to cut (although we can examine distances in that
      // round
      //
      // I think we pretty much need to pre-calcluate which direction we're going to split the split point BEFORE
      // calculating the priority, since we can't otherwise decide whether to use the best or worst case left/right
      // decision.  If you have a hard to decide small range of values where a cut fall in the middle near one of the
      // tails, then we shouldn't choose the inwards decision for priority calculation if that leaves us with too small
      // a range for putting a split based on our minimum length, so I think this means we need to calculate the
      // splits that we would acualize before calculating priority
      //
      // If for our priority score, we wanted to first materialize the items with the highest tidal disruption, then
      // we need to know which direction we'll be going from an aspirational split point.  We can choose the best
      // case or worst case side but at the time we're computing it we have the same amount of info that we'll have
      // in the future if we decide to split one of them, so we can compute which direction we'll prefer at this
      // point and we can use either the high cost minus the low cost if we think we want to first materialize the
      // options that don't screw us, or if we want to first work on the splits with the highest tidal disruption
      // after we've carefullly decided which way we'll split
      // 
      // So, our process is:
      //    1) Pull a high priority item from the queue (which has a pre-calculated direction to split and all other
      //       splitting decisions already calculated beforehand)
      //    1) execute our pre-determined cut placement AND move cuts from one side to the other if called for in our pre-plan
      //    2) re-calculate the aspiration cut points and for each of those do a first pass combination exploration
      //       to choose the best materialied cut point based on just ourselves
      //    3) Re-pass through our semi-materialized cuts points and jiggle them as necessary against their neighbours
      //       since the "view of the world" is different for each cut point and they don't match perfectly even if
      //       they are often close.
      //    4) Pass from the center to the outer-outer boundary (twice the boundary distance), and remove cuts from
      //       our priority queue, then calculate our new priority which is based on the squared change in all
      //       aspirational cut point (either real or just assuming equal splitting after the cut)
      //       And re-add them with our newly calculated priority, which can examine any cuts within
      //       the N item window at any point (but won't change them)

      auto iterator = pBestSplitPoints->begin();
      SplitPoint * const pSplitBest = *iterator;

      EBM_ASSERT(!pSplitBest->IsDeleted());
      EBM_ASSERT(!pSplitBest->IsSplit());

      EBM_ASSERT(pSplitBest->m_pSplitLowerBoundary < pSplitBest->m_pSplitMoveThis);
      EBM_ASSERT(pSplitBest->m_pSplitMoveThis < pSplitBest->m_pSplitHigherBoundary);
      EBM_ASSERT(1 <= pSplitBest->m_cSplitLowerBoundary);
      EBM_ASSERT(1 <= pSplitBest->m_cSplitHigherBoundary);
      EBM_ASSERT(pSplitBest->m_pSplitLowerBoundary <= pSplitBest->m_pSplitMoveThis - pSplitBest->m_cSplitLowerBoundary);
      EBM_ASSERT(pSplitBest->m_pSplitMoveThis + pSplitBest->m_cSplitHigherBoundary <= pSplitBest->m_pSplitHigherBoundary);

      // TODO: erase all of them
      //pBestSplitPoints->erase(iterator);

      SplitPoint * const pSplitMoveThis = pSplitBest->m_pSplitMoveThis;
      const size_t iVal = pSplitBest->m_iVal;

      pSplitMoveThis->m_iVal = iVal;
      pSplitMoveThis->SetDeleted(false);
      pSplitMoveThis->SetSplit(true);

      const FloatEbmType iValFloat = static_cast<FloatEbmType>(iVal);

      const size_t cSplitLowerBoundary = pSplitBest->m_cSplitLowerBoundary;
      SplitPoint * pSplitLowerAspirationalCur = pSplitMoveThis - 1;
      SplitPoint * const pSplitLowerBoundary = pSplitBest->m_pSplitLowerBoundary;
      EBM_ASSERT(!pSplitLowerBoundary->IsDeleted());
      const size_t cSplitLowerEnd = cSplitLowerBoundary - 1;
      if(LIKELY(0 < cSplitLowerEnd)) {
         const FloatEbmType iValLowerFloat = pSplitLowerBoundary->IsSplit() ?
            static_cast<FloatEbmType>(pSplitLowerBoundary->m_iVal) :
            pSplitLowerBoundary->m_iValAspirationalFloat;

         const FloatEbmType lowerDistancePerSplit = iValLowerFloat / cSplitLowerBoundary;
         size_t cLowerCompleted = 0;
         do {
            pSplitLowerAspirationalCur->SetDeleted(false);
            ++cLowerCompleted;
            pSplitLowerAspirationalCur->m_iValAspirationalFloat =
               iValFloat - lowerDistancePerSplit * static_cast<FloatEbmType>(cLowerCompleted);
            --pSplitLowerAspirationalCur;
         } while(LIKELY(cLowerCompleted != cSplitLowerEnd));
      }
      while(UNLIKELY(pSplitLowerAspirationalCur != pSplitLowerBoundary)) {
         pSplitLowerAspirationalCur->SetDeleted(true);
         --pSplitLowerAspirationalCur;
      }




      //////////////// GOOD BELOW

      constexpr size_t illegalIndex = std::numeric_limits<size_t>::max();

      SplitPoint * pSplitLowerLower = pSplitBest->m_pSplitLowerStartWindow;
      EBM_ASSERT(!pSplitLowerLower->IsDeleted());
      size_t cSplitLowerLower = pSplitBest->m_cSplitLowerStartWindow;
      size_t iValLowerLower = PREDICTABLE(pSplitLowerLower->IsSplit()) ? pSplitLowerLower->m_iVal : illegalIndex;

      size_t iValLowerHigher = iVal;
      size_t cSplitLowerHigher = 0;
      SplitPoint * pSplitLowerHigher = pSplitMoveThis;

      SplitPoint * pSplitLowerCur = pSplitMoveThis;
      SplitPoint * const pSplitLowerHigherEnd = pSplitBest->m_pSplitLowerBoundary;

      while(true) {
         EBM_ASSERT(!pSplitLowerLower->IsDeleted());
         if(PREDICTABLE(illegalIndex == iValLowerLower)) {
            EBM_ASSERT(!pSplitLowerLower->IsSplit());
            do {
               --pSplitLowerLower;
            } while(UNLIKELY(pSplitLowerLower->IsDeleted()));
            if(UNLIKELY(pSplitLowerLower->IsSplit())) {
               iValLowerLower = pSplitLowerLower->m_iVal;
            }
         } else {
            EBM_ASSERT(pSplitLowerLower->IsSplit());
            --cSplitLowerLower;
            if(UNLIKELY(0 == cSplitLowerLower)) {
               // our center has reached the end of the split range or an unmovable actualized split
#ifndef NDEBUG
               SplitPoint * pSplitDebug = pSplitLowerCur;
               do {
                  --pSplitDebug;
               } while(UNLIKELY(pSplitDebug->IsDeleted()));
               EBM_ASSERT(pSplitDebug == pSplitLowerLower); // we should have exited on 0 == cSplitLowerLower beforehand
#endif // NDEBUG
               break;
            }
         }

         EBM_ASSERT(!pSplitLowerHigher->IsDeleted());
         if(PREDICTABLE(k_SplitExploreDistance == cSplitLowerHigher)) {
            do {
               --pSplitLowerHigher;
            } while(UNLIKELY(pSplitLowerHigher->IsDeleted()));
            EBM_ASSERT(!pSplitLowerHigher->IsSplit());

            iValLowerHigher = illegalIndex;

            if(UNLIKELY(pSplitLowerHigher == pSplitLowerHigherEnd)) {
               // the item at pSplitLowerHigherEnd is the limit of our change and it didn't change itself,
               // so everything beyond this point doesn't need to be updated, so exit the loop
               break;
            }
         } else {
            EBM_ASSERT(pSplitLowerHigher->IsSplit());
            EBM_ASSERT(illegalIndex == iValLowerHigher);
            ++cSplitLowerHigher;
         }

         do {
            --pSplitLowerCur;
         } while(UNLIKELY(pSplitLowerCur->IsDeleted()));
         EBM_ASSERT(!pSplitLowerCur->IsSplit()); // we should have exited on 0 == cSplitLowerLower beforehand

         const size_t iValLanding = static_cast<size_t>(pSplitLowerCur->m_iValAspirationalFloat);
         const size_t iValLowChoice = aNeighbourJumps[iValuesStart + iValLanding].m_iStartCur - iValuesStart;
         const size_t iValHighChoice = aNeighbourJumps[iValuesStart + iValLanding].m_iStartNext - iValuesStart;

         bool bCanSplitLow = true;
         if(illegalIndex == iValLowerLower) {
            EBM_ASSERT(!pSplitLowerLower->IsSplit());
            if(static_cast<FloatEbmType>(iValLowChoice) - pSplitLowerLower->m_iValAspirationalFloat < 
               cMinimumInstancesPerBinFloat) 
            {
               bCanSplitLow = false;
            }
         } else {
            EBM_ASSERT(pSplitLowerLower->IsSplit());
            if(iValLowChoice - iValLowerLower < cMinimumInstancesPerBin) {
               bCanSplitLow = false;
            }
         }
         FloatEbmType avgLengthLowBest;
         size_t cModifyDirectionLowBest = 0;
         if(bCanSplitLow) {
            FloatEbmType distanceLowLowFloat;
            FloatEbmType distanceLowHighFloat;

            if(illegalIndex == iValLowerLower) {
               EBM_ASSERT(!pSplitLowerLower->IsSplit());
               distanceLowLowFloat = static_cast<FloatEbmType>(iValLowChoice) - pSplitLowerLower->m_iValAspirationalFloat;
            } else {
               EBM_ASSERT(pSplitLowerLower->IsSplit());
               distanceLowLowFloat = static_cast<FloatEbmType>(iValLowChoice - iValLowerLower);
            }

            if(illegalIndex == iValLowerHigher) {
               EBM_ASSERT(!pSplitLowerHigher->IsSplit());
               distanceLowHighFloat = static_cast<FloatEbmType>(pSplitLowerHigher->m_iValAspirationalFloat)
                  - static_cast<FloatEbmType>(iValLowChoice);
            } else {
               EBM_ASSERT(pSplitLowerHigher->IsSplit());
               distanceLowHighFloat = static_cast<FloatEbmType>(iValLowerHigher - iValLowChoice);
            }

            // we dominate our cost by the smallest range which we try to maximize
            avgLengthLowBest = std::min(distanceLowLowFloat / static_cast<FloatEbmType>(cSplitLowerLower),
               distanceLowHighFloat / static_cast<FloatEbmType>(cSplitLowerHigher));
            
            ptrdiff_t cModifyDirection;

            cModifyDirection = ptrdiff_t { -1 };
            while(true) {
               const ptrdiff_t denominatorLow = static_cast<ptrdiff_t>(cSplitLowerLower) + cModifyDirection;
               if(ptrdiff_t { 0 } == denominatorLow) {
                  break;
               }
               const ptrdiff_t denominatorHigh = static_cast<ptrdiff_t>(cSplitLowerHigher) - cModifyDirection;

               const FloatEbmType avgLengthLow = distanceLowLowFloat / static_cast<FloatEbmType>(denominatorLow);
               const FloatEbmType avgLengthHigh = distanceLowHighFloat / static_cast<FloatEbmType>(denominatorHigh);

               // we dominate our cost by the smallest range which we try to maximize
               const FloatEbmType avgLength = std::min(avgLengthLow, avgLengthHigh);

               if(avgLength < avgLengthLowBest) {
                  // if things didn't improve, don't continue
                  break;
               }
               --cModifyDirection;
               EBM_ASSERT(ptrdiff_t { 0 } != cModifyDirection);

               avgLengthLowBest = avgLength;
               cModifyDirectionLowBest = cModifyDirection;
            }
         }

         //////////////// GOOD ABOVE


         const FloatEbmType iValHighChoiceFloat = static_cast<FloatEbmType>(iValHighChoice);


         while(true) {
            move counts from higher to lower since we moved lower(don't bother checking the opposite direction)


               const FloatEbmType avgLengthBehindHigh = (iHighChoice - iFloatLeft) / iSplitDistanceLeft;
            const FloatEbmType avgLengthAheadHigh = (iFloatRight - iValHighChoice) / iSplitDistanceRight;
            // we dominate our cost by the smallest range which we try to maximize
            const FloatEbmType avgLengthHigh = avgLengthBehindHigh < avgLengthAheadHigh ?
               avgLengthBehindHigh : avgLengthAheadHigh;
         }

         // TODO : verify that we can in fact go low or high with enough separation to the absolute edges to make a cut
         // because as it stands if this range is selected next it WILL be split that way

         const bool bGoLow = avgLengthHigh < avgLengthLow;
         const FloatEbmType priority = std::abs(avgLengthLow - avgLengthHigh);

         // we want our priority to be as high as possible because it means that it would be stupid to make one of
         // the available choices, and we have high certainty that choosing the lower cost side is better
         // if both sides are low cost, or both sides are high cost, we might as well continue splitting and
         // push these decisions until a later time when we've decided the more obvious choices

         pBestSplitPoints->erase(&aSplitsWithENDPOINTS[iSplitCenter]);
         aSplitsWithENDPOINTS[iSplitCenter].m_iVal = bGoLow ? k_SplitLower : k_SplitHigher;
         aSplitsWithENDPOINTS[iSplitCenter].m_cutPriority = priority;
         pBestSplitPoints->insert(&aSplitsWithENDPOINTS[iSplitCenter]);
      }






























      // even with floating point inexactness we shouldn't be able to get a number below zero
      const FloatEbmType iAspirationalValFloatCur = pSplitCur->m_iValAspirationalFloat;
      EBM_ASSERT(FloatEbmType { 0 } <= iAspirationalValFloatCur);
      size_t iLandingVal = static_cast<size_t>(iAspirationalValFloatCur); // round down to find the zone
      if(UNLIKELY(cSplittableItems <= iLandingVal)) {
         // handle the very very unlikely situation where m_iAspirationalFloat rounds up to 
         // cSplittableItems due to floating point issues
         iLandingVal = cSplittableItems - 1;
      }
      const NeighbourJump * const pNeighbourJump = &aNeighbourJumps[iValuesStart + iLandingVal];
      EBM_ASSERT(iValuesStart <= pNeighbourJump->m_iStartCur);
      EBM_ASSERT(iValuesStart < pNeighbourJump->m_iStartNext);
      EBM_ASSERT(pNeighbourJump->m_iStartCur < pNeighbourJump->m_iStartNext);
      EBM_ASSERT(pNeighbourJump->m_iStartNext <= iValuesStart + cSplittableItems);
      EBM_ASSERT(pNeighbourJump->m_iStartCur < iValuesStart + cSplittableItems);

      EBM_ASSERT(k_SplitHigher == pSplitCur->m_iValActualized || k_SplitLower == pSplitCur->m_iValActualized);
      const size_t iActualizedValAbsoluteCur = *(UNPREDICTABLE(k_SplitHigher == pSplitCur->m_iValActualized) ?
         &pNeighbourJump->m_iStartNext : &pNeighbourJump->m_iStartCur);
      EBM_ASSERT(iValuesStart <= iActualizedValAbsoluteCur);
      const size_t iActualizedValCur = iActualizedValAbsoluteCur - iValuesStart;
      pSplitCur->m_iValActualized = iActualizedValCur;

      //const size_t iSplitCur = pSplitCur - aSplitsWithENDPOINTS;









      //struct SplitPoint final {
      //   // TODO can we wrap some of this into a union so that we can overlay pre-set and post-seting of the actual index

      //   SplitPoint() = default; // preserve our POD status
      //   ~SplitPoint() = default; // preserve our POD status
      //   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
      //   void operator delete (void *) = delete; // we only use malloc/free in this library

      //   FloatEbmType   m_cutPriority;
      //   size_t         m_uniqueRandom;

      //   ptrdiff_t      m_iValActualized;
      //   // TODO: see if we can turn m_iAspirationalFloat into an integer or set of integers (numerator/denominator) to avoid floating point weirdness
      //   FloatEbmType   m_iValAspirationalFloat;

      //   ptrdiff_t      m_cSplitMoveThis;
      //   SplitPoint * m_pSplitMoveThis;

      //   size_t         m_cSplitLowerBoundary;
      //   SplitPoint * m_pSplitLowerBoundary;

      //   size_t         m_cSplitHigherBoundary;
      //   SplitPoint * m_pSplitHigherBoundary;
      //};


      //size_t         cSplitLowerBoundary = pSplitCur->m_cSplitLowerBoundary;
      SplitPoint * pSplitLowerBoundary = pSplitCur->m_pSplitLowerBoundary;
      ptrdiff_t iValActualizedLower = pSplitLowerBoundary->m_iValActualized;
      FloatEbmType iValLowerFloat = 0 <= iValActualizedLower ? static_cast<FloatEbmType>(iValActualizedLower) :
         pSplitLowerBoundary->m_iValAspirationalFloat;

      //size_t         cSplitHigherBoundary = pSplitCur->m_cSplitHigherBoundary;
      SplitPoint * pSplitHigherBoundary = pSplitCur->m_pSplitHigherBoundary;
      ptrdiff_t iValActualizedHigher = pSplitHigherBoundary->m_iValActualized;
      FloatEbmType iValHigherFloat = 0 <= iValActualizedHigher ? static_cast<FloatEbmType>(iValActualizedHigher) :
         pSplitHigherBoundary->m_iValAspirationalFloat;






      static_assert(1 <= k_SplitExploreDistance, "k_SplitExploreDistance can't be zero");
      // if this underflows we don't care because underflow isn't undefined behavior, and we have a guard item
      size_t cSplitWindowRemainingEarlier = k_SplitExploreDistance;
      size_t iSplitEarlier = iSplitCur;
      FloatEbmType iValEarlierFloat;
      while(true) {
         // TODO: this loop can be elimiminated by storing all the state inside SplitPoint since we need to determine
         //       it anyways when we compute the cost of splitting on each potential SplitPoint.

         --iSplitEarlier;
         // we inserted guard items at the start and end, so we are guaranteed to exit before falling off the sides
         const ptrdiff_t iActualizedValMoving = aSplitsWithENDPOINTS[iSplitEarlier].m_iValActualized;
         if(UNLIKELY(ptrdiff_t { -1 } < iActualizedValMoving)) {
            // checking against -1 is a clever way to eliminate one compare instruction since we can then use
            // the equality flag below without comparing again

            // preserve the accuracy of this value for future use
            --cSplitWindowRemainingEarlier;

            // this check needs to occur first since m_iAspirationalFloat is invalid if 0 <= iActualMoving
            iValEarlierFloat = static_cast<FloatEbmType>(iActualizedValMoving);
            break;
         }
         // k_SplitDeleted is -1, so the comparion to k_SplitDeleted is faster and uses the CMP instruction from above
         if(LIKELY(k_SplitDeleted != iActualizedValMoving)) {
            --cSplitWindowRemainingEarlier;
            EBM_ASSERT(k_SplitExploreDistance - cSplitWindowRemainingEarlier <= iSplitCur - iSplitEarlier);
            if(UNLIKELY(0 == cSplitWindowRemainingEarlier)) {
               iValEarlierFloat = aSplitsWithENDPOINTS[iSplitEarlier].m_iValAspirationalFloat;
               break;
            }
         }
      }

      //CHANGE THE ALGORITHM HERE.  WE WILL INSTEAD NUKE ALL OUR EXISTING PROPOSED CUTS AND RE-DO THEM WITHIN OUR WINDOW
      //BECAUSE WE WANT TO DECIDE HERE IF WE'LL MOVE SOME CUTS FROM ONE SIDE TO THE OTHER BASED ON 

      const FloatEbmType scaleDenomerator = iAspirationalFloatCur - iEarlierFloat;
      if(UNLIKELY(scaleDenomerator <= 0)) {
         // this might be possible in extreme situations where we have exceedingly large indexes that cause
         // floating point inexactness

         // TODO: handle this situation
         EBM_ASSERT(false); // assert false for now even though we think this might be possible in theory
      }
      const FloatEbmType iActualFloat = static_cast<FloatEbmType>(iActualizedValCur);

      // we multiply the size of any segments earlier by this percentage to find their new aspirational lengths
      const FloatEbmType scaleEarlier = (iActualFloat - iValEarlierFloat) / scaleDenomerator;
      if(UNLIKELY(scaleEarlier <= 0)) {
         // in theory it's probably possible to construct an adversarial dataset that keeps moving
         // split points to one side to the point where the remaining ranges are larger than the difference
         // between splits points in the distilled region that now has unnaturally small ranges.
         // Taken to the extreme, you could even make a large window smaller than the allowed number of
         // items in one range, so we need to handle this in theory
         //
         // Also, this could end up as zero or even negative due to floating point inexactness, probably only
         // if there were huge numbers of items

         // TODO: handle this situation
         EBM_ASSERT(false); // assert false for now even though we think this might be possible in theory
      }

      size_t iSplitMoving = iSplitCur;
      FloatEbmType iNewFloatPrev = iActualFloat;
      while(true) {
         do {
            --iSplitMoving;
         } while(UNLIKELY(k_SplitDeleted == aSplitsWithENDPOINTS[iSplitMoving].m_iValActualized));

         if(UNLIKELY(iSplitEarlier == iSplitMoving)) {
            if(UNLIKELY(iNewFloatPrev - iValEarlierFloat < cMinimumInstancesPerBin)) {
               // our split zones are too small.  Eliminate one or more regions, and re-do everything.
               break;
            }
            goto done_earlier_jiggle;
         }

         EBM_ASSERT(aSplitsWithENDPOINTS[iSplitMoving].m_iValActualized < 0);

         const FloatEbmType iAspirationalFloatMoving = aSplitsWithENDPOINTS[iSplitMoving].m_iValAspirationalFloat;
         const FloatEbmType distance = iAspirationalValFloatCur - iAspirationalFloatMoving;
         const FloatEbmType distanceScaled = distance * scaleEarlier;
         const FloatEbmType iNewFloatMoving = iActualFloat - distanceScaled;

         if(UNLIKELY(iNewFloatPrev - iNewFloatMoving < cMinimumInstancesPerBin)) {
            // we've passed the point of no return.  Eliminate one or more regions
            break;
         }

         iNewFloatPrev = iNewFloatMoving;

         // note: this doesn't affect our priority number which we aren't changing here
         aSplitsWithENDPOINTS[iSplitMoving].m_iValAspirationalFloat = iNewFloatMoving;
      }

      // TODO : handle this case of having a range that's too small

      // for now allow a range that is less than cMinimumInstancesPerBin in length.  We haven't materialized
      // it here, so there's a chance things will ultimately resolve with this bin being greater than
      // cMinimumInstancesPerBin.  Even if we end up with a bin less than cMinimumInstancesPerBin after
      // materializing, we try a post-process step that might create enough space for this range to expand
      // into a large enough size

   done_earlier_jiggle:;

      size_t iSplitCenter = iSplitCur;
      size_t iSplitEdgeLeft = iSplitEarlier;
      size_t iSplitEdgeRight = iSplitCur;
      size_t iSplitDistanceLeft = k_SplitExploreDistance - cSplitWindowRemainingEarlier;
      size_t iSplitDistanceRight = 0;
      FloatEbmType iFloatLeft = iValEarlierFloat;
      FloatEbmType iFloatRight = iActualFloat;
      ptrdiff_t iActualLeft = aSplitsWithENDPOINTS[iSplitEdgeLeft].m_iValActualized;
      while(true) {
         if(PREDICTABLE(iActualLeft < 0)) {
            EBM_ASSERT(k_SplitDeleted != iActualLeft);
            do {
               --iSplitEdgeLeft;
               iActualLeft = aSplitsWithENDPOINTS[iSplitEdgeLeft].m_iValActualized;
            } while(UNLIKELY(k_SplitDeleted == iActualLeft));
            if(LIKELY(iActualLeft <= ptrdiff_t { -1 })) {
               // the above is a clever way to check if it's below zero.  k_SplitDeleted is -1, 
               // so we don't need the CMP assembly instruction
               iFloatLeft = aSplitsWithENDPOINTS[iSplitEdgeLeft].m_iValAspirationalFloat;
            } else {
               iFloatLeft = static_cast<FloatEbmType>(iActualLeft);
            }
         } else {
            --iSplitDistanceLeft;
            if(UNLIKELY(0 == iSplitDistanceLeft)) {
               // our center has reached the end of the split range or an unmovable actualized split
               break;
            }
         }

         if(PREDICTABLE(k_SplitExploreDistance == iSplitDistanceRight)) {
            do {
               --iSplitEdgeRight;
            } while(UNLIKELY(k_SplitDeleted == aSplitsWithENDPOINTS[iSplitEdgeRight].m_iValActualized));
            if(UNLIKELY(iSplitEarlier == iSplitEdgeRight)) {
               // the item at iSplitEarlier was our anchor and didn't itself move, so if that's our right edge
               // then we're done since it never changed the priority within our window from item iSplitCenter
               break;
            }
            iFloatRight = static_cast<FloatEbmType>(aSplitsWithENDPOINTS[iSplitEdgeRight].m_iValAspirationalFloat);
         } else {
            ++iSplitDistanceRight;
         }

         do {
            --iSplitCenter;
         } while(UNLIKELY(k_SplitDeleted == aSplitsWithENDPOINTS[iSplitCenter].m_iValActualized));

         const size_t iChoiceLanding = static_cast<size_t>(aSplitsWithENDPOINTS[iSplitCenter].m_iValAspirationalFloat);
         const size_t iLowChoice = aNeighbourJumps[iValuesStart + iChoiceLanding].m_iStartCur - iValuesStart;
         const size_t iHighChoice = aNeighbourJumps[iValuesStart + iChoiceLanding].m_iStartNext - iValuesStart;

         const FloatEbmType avgLengthBehindLow = (iLowChoice - iFloatLeft) / iSplitDistanceLeft;
         const FloatEbmType avgLengthAheadLow = (iFloatRight - iLowChoice) / iSplitDistanceRight;
         // we dominate our cost by the smallest range which we try to maximize
         const FloatEbmType avgLengthLow = avgLengthBehindLow < avgLengthAheadLow ?
            avgLengthBehindLow : avgLengthAheadLow;

         const FloatEbmType avgLengthBehindHigh = (iHighChoice - iFloatLeft) / iSplitDistanceLeft;
         const FloatEbmType avgLengthAheadHigh = (iFloatRight - iHighChoice) / iSplitDistanceRight;
         // we dominate our cost by the smallest range which we try to maximize
         const FloatEbmType avgLengthHigh = avgLengthBehindHigh < avgLengthAheadHigh ?
            avgLengthBehindHigh : avgLengthAheadHigh;

         // TODO : verify that we can in fact go low or high with enough separation to the absolute edges to make a cut
         // because as it stands if this range is selected next it WILL be split that way

         const bool bGoLow = avgLengthHigh < avgLengthLow;
         const FloatEbmType priority = std::abs(avgLengthLow - avgLengthHigh);

         // we want our priority to be as high as possible because it means that it would be stupid to make one of
         // the available choices, and we have high certainty that choosing the lower cost side is better
         // if both sides are low cost, or both sides are high cost, we might as well continue splitting and
         // push these decisions until a later time when we've decided the more obvious choices

         pBestSplitPoints->erase(&aSplitsWithENDPOINTS[iSplitCenter]);
         aSplitsWithENDPOINTS[iSplitCenter].m_iValActualized = bGoLow ? k_SplitLower : k_SplitHigher;
         aSplitsWithENDPOINTS[iSplitCenter].m_priority = priority;
         pBestSplitPoints->insert(&aSplitsWithENDPOINTS[iSplitCenter]);
      }

      // TODO: also handle the right hand side

   } while(!pBestSplitPoints->empty());

   IronSplits();

   return 0;
}
#endif // NEVER

INLINE_RELEASE static size_t TreeSearchSplitSegment(
   std::set<SplitPoint *, CompareSplitPoint> * pBestSplitPoints,

   const size_t cMinimumInstancesPerBin,

   const size_t iValuesStart,
   const size_t cSplittableItems,
   const NeighbourJump * const aNeighbourJumps,

   // prior to calling this function, the caller must remove the endpoints if any
   const size_t cCENTERSplitsAssigned,
   // for efficiency we include space for the end point cuts even if they don't exist
   SplitPoint * const aSplitsWithENDPOINTS
) {
   UNUSED(pBestSplitPoints);
   UNUSED(cMinimumInstancesPerBin);
   UNUSED(iValuesStart);
   UNUSED(cSplittableItems);
   UNUSED(aNeighbourJumps);
   UNUSED(cCENTERSplitsAssigned);
   UNUSED(aSplitsWithENDPOINTS);



   // - TODO: EXPLORING BOTH SIDES
   //   - first strategy is to divide the region into floating point divisions, and find the single worst split where going both left or right is bad (using floating point distance)
   //   - then go left and go right, re - divide the entire set base on the left choice and the right choice
   //   - but this grows at 2 ^ N, so we need annother one that makes a decision without going down two paths.It needs to check the left and right and make a decion on the spot
   //
   //   - we can design an algorithm that divides into 255 and chooses the worst one and then does a complete fit on either direction.Best fit is recorded
   //     then we re-do all 254 other cuts on BOTH sides.We can only do a set number of these, so after 8 levels we'd have 256 attempts.  That might be acceptable
   //   - the algorithm that we have below plays it safe since it needs to live with it's decions.  This more spectlative algorithm above can be more
   //     risky since it plays both directions a bad play won't undermine it.  As such, we should try and chose the worst decion without regard to position
   //     so in other words, try to choose the range that we have a drop point in in the middle where we need to move the most to get away from the 
   //     best drops.  We can also try going left, going right, OR not choosing.  Don't traverse down the NO choice path, so we add 50% load, but we don't grow at 3^N, and we'll 
   //     also explore the no choice at the root option
   //

   //constexpr size_t k_SplitExploreDepth = 8;
   //constexpr size_t k_SplitExplorations = size_t { 1 } << k_SplitExploreDepth;

//   return SplitSegment(pBestSplitPoints, cMinimumInstancesPerBin, iValuesStart, cSplittableItems, aNeighbourJumps,
//      cCENTERSplitsAssigned, aSplitsWithENDPOINTS);

   return 999999;
}

INLINE_RELEASE static size_t TradeSplitSegment(
   std::set<SplitPoint *, CompareSplitPoint> * pBestSplitPoints,

   const size_t cMinimumInstancesPerBin,

   const size_t iValuesStart,
   const size_t cSplittableItems,
   const NeighbourJump * const aNeighbourJumps,

   // prior to calling this function, the caller must remove the endpoints if any
   const size_t cCENTERSplitsAssigned,
   // for efficiency we include space for the end point cuts even if they don't exist
   SplitPoint * const aSplitsWithENDPOINTS
) {
   // - TODO: ABOVE THE COUNT FIXING
   //   - we can examine what it would look like to have 1 more cut and 1 less cut that our original choice
   //   - then we can try and sort the best to worst subtraction and addition, and then try and swap the best subtraction with the best addition and repeat
   //   - calculate the maximum number of splits based on the minimum bunch size.  we should be able to do this by
   //     doing a single pass where we make every range the minimum
   //   - then we can loop from our current cuts to the maximum and stop when we hit the maximum (perahps there are long 
   //     ranges that prevent good spits)
   //   - if we want to get a specific number of cuts, we can ask for that many, but we might get less back as a result
   //     (never more).  We can try to increment the number of items that we ask for and see if we end up with the right
   //     number.  It might be bad though if we continually do +1 because it might be intractable if there are a lot
   //     of splits.  Perahps we want to use a binary algorithm where we do +1, +2, +4, +8, and if we exceed then
   //     do binary descent between 4 and 8 until we get our exact number.

   return TreeSearchSplitSegment(pBestSplitPoints, cMinimumInstancesPerBin, iValuesStart, cSplittableItems, aNeighbourJumps,
      cCENTERSplitsAssigned, aSplitsWithENDPOINTS);
}

// TODO: make this "INLINE_RELEASE static"
size_t StuffSplitsIntoSplittingRanges(
   const size_t cSplittingRanges,
   SplittingRange * const aSplittingRange,
   const size_t cMinimumInstancesPerBin,
   size_t cRemainingSplits
) {
   // generally, having small bins with insufficient data to cover the base rate is more damaging
   // than the lost opportunity from not cutting big bins.  So, what we want to avoid is having
   // small bins.  So, create a heap and insert the average bin size AFTER we would add a new cut
   // don't insert any SplittingRanges that cannot legally be cut (so, it's guaranteed to only have
   // cuttable items).  Now pick off the SplittingRange that has the largest average AFTER adding a cut
   // and then add the cut, re-calculate the new average, and re-insert into the heap.  Continue
   // until there are no items in the heap because they've all exhausted the possibilities of cuts
   // OR until we run out of cuts to dole out.

   class CompareSplittingRange final {
   public:
      INLINE_ALWAYS bool operator() (
         const SplittingRange * const & lhs,
         const SplittingRange * const & rhs
         ) const {
         return lhs->m_avgRangeWidthAfterAddingOneSplit == rhs->m_avgRangeWidthAfterAddingOneSplit ?
            (lhs->m_uniqueRandom <= rhs->m_uniqueRandom) :
            (lhs->m_avgRangeWidthAfterAddingOneSplit <= rhs->m_avgRangeWidthAfterAddingOneSplit);
      }
   };

   EBM_ASSERT(1 <= cSplittingRanges);
   EBM_ASSERT(nullptr != aSplittingRange);

   if(0 != cRemainingSplits) {
      const FloatEbmType cMinimumInstancesPerBinFloat = static_cast<FloatEbmType>(cMinimumInstancesPerBin);
      try {
         std::priority_queue<SplittingRange *, std::vector<SplittingRange *>, CompareSplittingRange> queue;

         SplittingRange * pSplittingRangeInit = aSplittingRange;
         const SplittingRange * const pSplittingRangeEnd = aSplittingRange + cSplittingRanges;
         do {
            // we're adding one to the splits assigned here because the # of ranges is -1 from the # of splits
            size_t newProposedRanges = pSplittingRangeInit->m_cSplitsAssigned;
            if(0 == pSplittingRangeInit->m_cUnsplittableEitherSideMin) {
               // our first and last SplittingRanges can either have a long range of equal items on their tail ends
               // or nothing.  If there is a long range of equal items, then we'll be placing one cut at the tail
               // end, otherwise we have an implicit cut there and we don't need to use one of our cuts.  It's
               // like getting a free cut, so increase the number of ranges by one if we don't need one cut at the tail
               // side

               ++newProposedRanges;
               if(0 == pSplittingRangeInit->m_cUnsplittableEitherSideMax) {
                  // if there's just one range and there are no long ranges on either end, then one split will create
                  // two ranges, so add 1 more.

                  ++newProposedRanges;
               }
            }
            const FloatEbmType avgRangeWidthAfterAddingOneSplit =
               static_cast<FloatEbmType>(pSplittingRangeInit->m_cSplittableItems) / static_cast<FloatEbmType>(newProposedRanges);

            if(cMinimumInstancesPerBinFloat <= avgRangeWidthAfterAddingOneSplit) {
               pSplittingRangeInit->m_avgRangeWidthAfterAddingOneSplit = avgRangeWidthAfterAddingOneSplit;
               queue.push(pSplittingRangeInit);
            }

            ++pSplittingRangeInit;
         } while(pSplittingRangeEnd != pSplittingRangeInit);

         while(!queue.empty()) {
            SplittingRange * pSplittingRangeAdd = queue.top();
            queue.pop();

            // by virtue of being in the queue, we know that we meet the minimum length requirement
            size_t newProposedRanges = pSplittingRangeAdd->m_cSplitsAssigned + 1;

            // we're adding one to the splits assigned here because the # of ranges is -1 from the # of splits
            pSplittingRangeAdd->m_cSplitsAssigned = newProposedRanges;

            --cRemainingSplits;
            if(0 == cRemainingSplits) {
               break;
            }

            if(0 == pSplittingRangeAdd->m_cUnsplittableEitherSideMin) {
               // our first and last SplittingRanges can either have a long range of equal items on their tail ends
               // or nothing.  If there is a long range of equal items, then we'll be placing one cut at the tail
               // end, otherwise we have an implicit cut there and we don't need to use one of our cuts.  It's
               // like getting a free cut, so increase the number of ranges by one if we don't need one cut at the tail
               // side

               ++newProposedRanges;
               if(0 == pSplittingRangeInit->m_cUnsplittableEitherSideMax) {
                  // if there's just one range and there are no long ranges on either end, then one split will create
                  // two ranges, so add 1 more.

                  ++newProposedRanges;
               }
            }
            const FloatEbmType avgRangeWidthAfterAddingOneSplit =
               static_cast<FloatEbmType>(pSplittingRangeAdd->m_cSplittableItems) / static_cast<FloatEbmType>(newProposedRanges);

            if(cMinimumInstancesPerBinFloat <= avgRangeWidthAfterAddingOneSplit) {
               pSplittingRangeAdd->m_avgRangeWidthAfterAddingOneSplit = avgRangeWidthAfterAddingOneSplit;
               queue.push(pSplittingRangeAdd);
            }
         }
      } catch(...) {
         LOG_0(TraceLevelWarning, "WARNING StuffSplitsIntoSplittingRanges exception");
      }
   }
   return cRemainingSplits;
}

INLINE_RELEASE static size_t FillSplittingRangeRemaining(
   const size_t cSplittingRanges,
   SplittingRange * const aSplittingRange
) {
   EBM_ASSERT(1 <= cSplittingRanges);
   EBM_ASSERT(nullptr != aSplittingRange);

   SplittingRange * pSplittingRange = aSplittingRange;
   const SplittingRange * const pSplittingRangeEnd = pSplittingRange + cSplittingRanges;
   do {
      const size_t cUnsplittablePriorItems = pSplittingRange->m_cUnsplittablePriorItems;
      const size_t cUnsplittableSubsequentItems = pSplittingRange->m_cUnsplittableSubsequentItems;

      pSplittingRange->m_cUnsplittableEitherSideMax = std::max(cUnsplittablePriorItems, cUnsplittableSubsequentItems);
      pSplittingRange->m_cUnsplittableEitherSideMin = std::min(cUnsplittablePriorItems, cUnsplittableSubsequentItems);

      pSplittingRange->m_flags = k_MiddleSplittingRange;
      pSplittingRange->m_cSplitsAssigned = 1;

      ++pSplittingRange;
   } while(pSplittingRangeEnd != pSplittingRange);

   size_t cConsumedSplittingRanges = cSplittingRanges;
   if(1 == cSplittingRanges) {
      aSplittingRange[0].m_flags = k_FirstSplittingRange | k_LastSplittingRange;
      // might as well assign a split to the only SplittingRange.  We'll be stuffing it as full as it can get soon
      EBM_ASSERT(1 == aSplittingRange[0].m_cSplitsAssigned);
   } else {
      aSplittingRange[0].m_flags = k_FirstSplittingRange;
      if(0 == aSplittingRange[0].m_cUnsplittablePriorItems) {
         aSplittingRange[0].m_cSplitsAssigned = 0;
         --cConsumedSplittingRanges;
      }

      --pSplittingRange; // go back to the last one
      pSplittingRange->m_flags = k_LastSplittingRange;
      if(0 == pSplittingRange->m_cUnsplittableSubsequentItems) {
         pSplittingRange->m_cSplitsAssigned = 0;
         --cConsumedSplittingRanges;
      }
   }
   return cConsumedSplittingRanges;
}

INLINE_RELEASE static void FillSplittingRangeNeighbours(
   const size_t cInstances,
   FloatEbmType * const aSingleFeatureValues,
   const size_t cSplittingRanges,
   SplittingRange * const aSplittingRange
) {
   EBM_ASSERT(1 <= cInstances);
   EBM_ASSERT(nullptr != aSingleFeatureValues);
   EBM_ASSERT(1 <= cSplittingRanges);
   EBM_ASSERT(nullptr != aSplittingRange);

   SplittingRange * pSplittingRange = aSplittingRange;
   size_t cUnsplittablePriorItems = pSplittingRange->m_pSplittableValuesStart - aSingleFeatureValues;
   const FloatEbmType * const aSingleFeatureValuesEnd = aSingleFeatureValues + cInstances;
   if(1 != cSplittingRanges) {
      const SplittingRange * const pSplittingRangeLast = pSplittingRange + cSplittingRanges - 1; // exit without doing the last one
      do {
         const size_t cUnsplittableSubsequentItems =
            (pSplittingRange + 1)->m_pSplittableValuesStart - pSplittingRange->m_pSplittableValuesStart - pSplittingRange->m_cSplittableItems;

         pSplittingRange->m_cUnsplittablePriorItems = cUnsplittablePriorItems;
         pSplittingRange->m_cUnsplittableSubsequentItems = cUnsplittableSubsequentItems;

         cUnsplittablePriorItems = cUnsplittableSubsequentItems;
         ++pSplittingRange;
      } while(pSplittingRangeLast != pSplittingRange);
   }
   const size_t cUnsplittableSubsequentItems =
      aSingleFeatureValuesEnd - pSplittingRange->m_pSplittableValuesStart - pSplittingRange->m_cSplittableItems;

   pSplittingRange->m_cUnsplittablePriorItems = cUnsplittablePriorItems;
   pSplittingRange->m_cUnsplittableSubsequentItems = cUnsplittableSubsequentItems;
}

INLINE_RELEASE static void FillSplittingRangeBasics(
   const size_t cInstances,
   FloatEbmType * const aSingleFeatureValues,
   const size_t avgLength,
   const size_t cMinimumInstancesPerBin,
   const size_t cSplittingRanges,
   SplittingRange * const aSplittingRange
) {
   EBM_ASSERT(1 <= cInstances);
   EBM_ASSERT(nullptr != aSingleFeatureValues);
   EBM_ASSERT(1 <= avgLength);
   EBM_ASSERT(1 <= cMinimumInstancesPerBin);
   EBM_ASSERT(1 <= cSplittingRanges);
   EBM_ASSERT(nullptr != aSplittingRange);

   FloatEbmType rangeValue = *aSingleFeatureValues;
   FloatEbmType * pSplittableValuesStart = aSingleFeatureValues;
   const FloatEbmType * pStartEqualRange = aSingleFeatureValues;
   FloatEbmType * pScan = aSingleFeatureValues + 1;
   const FloatEbmType * const pValuesEnd = aSingleFeatureValues + cInstances;

   SplittingRange * pSplittingRange = aSplittingRange;
   while(pValuesEnd != pScan) {
      const FloatEbmType val = *pScan;
      if(val != rangeValue) {
         size_t cEqualRangeItems = pScan - pStartEqualRange;
         if(avgLength <= cEqualRangeItems) {
            if(aSingleFeatureValues != pSplittableValuesStart || cMinimumInstancesPerBin <= static_cast<size_t>(pStartEqualRange - pSplittableValuesStart)) {
               EBM_ASSERT(pSplittingRange < aSplittingRange + cSplittingRanges);
               pSplittingRange->m_pSplittableValuesStart = pSplittableValuesStart;
               pSplittingRange->m_cSplittableItems = pStartEqualRange - pSplittableValuesStart;
               ++pSplittingRange;
            }
            pSplittableValuesStart = pScan;
         }
         rangeValue = val;
         pStartEqualRange = pScan;
      }
      ++pScan;
   }
   if(pSplittingRange != aSplittingRange + cSplittingRanges) {
      // we're not done, so we have one more to go.. this last one
      EBM_ASSERT(pSplittingRange == aSplittingRange + cSplittingRanges - 1);
      EBM_ASSERT(pSplittableValuesStart < pValuesEnd);
      pSplittingRange->m_pSplittableValuesStart = pSplittableValuesStart;
      EBM_ASSERT(pStartEqualRange < pValuesEnd);
      const size_t cEqualRangeItems = pValuesEnd - pStartEqualRange;
      const FloatEbmType * const pSplittableRangeEnd = avgLength <= cEqualRangeItems ? pStartEqualRange : pValuesEnd;
      pSplittingRange->m_cSplittableItems = pSplittableRangeEnd - pSplittableValuesStart;
   }
}

// verified
INLINE_RELEASE static void FillSplittingRangeRandom(
   RandomStream * const pRandomStream,
   const size_t cSplittingRanges,
   SplittingRange * const aSplittingRange
) {
   EBM_ASSERT(1 <= cSplittingRanges);
   EBM_ASSERT(nullptr != aSplittingRange);

   size_t index = 0;
   SplittingRange * pSplittingRange = aSplittingRange;
   const SplittingRange * const pSplittingRangeEnd = pSplittingRange + cSplittingRanges;
   do {
      pSplittingRange->m_uniqueRandom = index;
      ++index;
      ++pSplittingRange;
   } while(pSplittingRangeEnd != pSplittingRange);

   // the last index doesn't need to be swapped, since there is nothing to swap it with
   const size_t cVisitSplittingRanges = cSplittingRanges - 1;
   for(size_t i = 0; LIKELY(i < cVisitSplittingRanges); ++i) {
      const size_t cPossibleSwapLocations = cSplittingRanges - i;
      EBM_ASSERT(1 <= cPossibleSwapLocations);
      // for randomness, we need to be able to swap with ourselves, so iSwap can be 0 
      // and in that case we'll swap with ourselves
      const size_t iSwap = pRandomStream->Next(cPossibleSwapLocations);
      const size_t uniqueRandomTmp = aSplittingRange[i].m_uniqueRandom;
      aSplittingRange[i].m_uniqueRandom = aSplittingRange[i + iSwap].m_uniqueRandom;
      aSplittingRange[i + iSwap].m_uniqueRandom = uniqueRandomTmp;
   }
}

INLINE_RELEASE static void FillSplittingRangePointers(
   const size_t cSplittingRanges,
   SplittingRange ** const apSplittingRange,
   SplittingRange * const aSplittingRange
) {
   EBM_ASSERT(1 <= cSplittingRanges);
   EBM_ASSERT(nullptr != apSplittingRange);
   EBM_ASSERT(nullptr != aSplittingRange);

   SplittingRange ** ppSplittingRange = apSplittingRange;
   const SplittingRange * const * const apSplittingRangeEnd = apSplittingRange + cSplittingRanges;
   SplittingRange * pSplittingRange = aSplittingRange;
   do {
      *ppSplittingRange = pSplittingRange;
      ++pSplittingRange;
      ++ppSplittingRange;
   } while(apSplittingRangeEnd != ppSplittingRange);
}

// verified
INLINE_RELEASE static void FillSplitPointRandom(
   RandomStream * const pRandomStream,
   const size_t cSplitPoints,
   SplitPoint * const aSplitPoints
) {
   EBM_ASSERT(1 <= cSplitPoints); // 1 can happen if there is only one splitting range
   EBM_ASSERT(nullptr != aSplitPoints);

   size_t index = 0;
   SplitPoint * pSplitPoint = aSplitPoints;
   const SplitPoint * const pSplitPointEnd = pSplitPoint + cSplitPoints;
   do {
      pSplitPoint->m_uniqueRandom = index;
      ++index;
      ++pSplitPoint;
   } while(pSplitPointEnd != pSplitPoint);

   // the last index doesn't need to be swapped, since there is nothing to swap it with
   const size_t cVisitSplitPoints = cSplitPoints - 1;
   for(size_t i = 0; LIKELY(i < cVisitSplitPoints); ++i) {
      const size_t cPossibleSwapLocations = cSplitPoints - i;
      EBM_ASSERT(1 <= cPossibleSwapLocations);
      // for randomness, we need to be able to swap with ourselves, so iSwap can be 0 
      // and in that case we'll swap with ourselves
      const size_t iSwap = pRandomStream->Next(cPossibleSwapLocations);
      const size_t uniqueRandomTmp = aSplitPoints[i].m_uniqueRandom;
      aSplitPoints[i].m_uniqueRandom = aSplitPoints[i + iSwap].m_uniqueRandom;
      aSplitPoints[i + iSwap].m_uniqueRandom = uniqueRandomTmp;
   }
}

// verified
INLINE_RELEASE static NeighbourJump * ConstructJumps(const size_t cInstances, const FloatEbmType * const aValues) {
   // TODO test this
   EBM_ASSERT(0 < cInstances);
   EBM_ASSERT(nullptr != aValues);

   NeighbourJump * const aNeighbourJump = EbmMalloc<NeighbourJump>(cInstances);
   if(nullptr == aNeighbourJump) {
      return nullptr;
   }

   FloatEbmType valNext = aValues[0];
   const FloatEbmType * pValue = aValues;
   const FloatEbmType * const pValueEnd = aValues + cInstances;

   size_t iStartCur = 0;
   NeighbourJump * pNeighbourJump = aNeighbourJump;
   do {
      const FloatEbmType valCur = valNext;
      do {
         ++pValue;
         if(UNLIKELY(pValueEnd == pValue)) {
            break;
         }
         valNext = *pValue;
      } while(PREDICTABLE(valNext == valCur));

      const size_t iStartNext = pValue - aValues;
      const size_t cItems = iStartNext - iStartCur;

      const NeighbourJump * const pNeighbourJumpEnd = pNeighbourJump + cItems;
      do {
         pNeighbourJump->m_iStartCur = iStartCur;
         pNeighbourJump->m_iStartNext = iStartNext;
         ++pNeighbourJump;
      } while(pNeighbourJumpEnd != pNeighbourJump);

      iStartCur = iStartNext;
   } while(LIKELY(pValueEnd != pValue));

   return aNeighbourJump;
}

INLINE_RELEASE static size_t CountSplittingRanges(
   const size_t cInstances,
   const FloatEbmType * const aSingleFeatureValues,
   const size_t avgLength,
   const size_t cMinimumInstancesPerBin
) {
   EBM_ASSERT(1 <= cInstances);
   EBM_ASSERT(nullptr != aSingleFeatureValues);
   EBM_ASSERT(1 <= avgLength);
   EBM_ASSERT(1 <= cMinimumInstancesPerBin);

   if(cInstances < (cMinimumInstancesPerBin << 1)) {
      // we can't make any cuts if we have less than 2 * cMinimumInstancesPerBin instances, 
      // since we need at least cMinimumInstancesPerBin instances on either side of the cut point
      return 0;
   }
   FloatEbmType rangeValue = *aSingleFeatureValues;
   const FloatEbmType * pSplittableValuesStart = aSingleFeatureValues;
   const FloatEbmType * pStartEqualRange = aSingleFeatureValues;
   const FloatEbmType * pScan = aSingleFeatureValues + 1;
   const FloatEbmType * const pValuesEnd = aSingleFeatureValues + cInstances;
   size_t cSplittingRanges = 0;
   while(pValuesEnd != pScan) {
      const FloatEbmType val = *pScan;
      if(val != rangeValue) {
         size_t cEqualRangeItems = pScan - pStartEqualRange;
         if(avgLength <= cEqualRangeItems) {
            if(aSingleFeatureValues != pSplittableValuesStart || cMinimumInstancesPerBin <= static_cast<size_t>(pStartEqualRange - pSplittableValuesStart)) {
               ++cSplittingRanges;
            }
            pSplittableValuesStart = pScan;
         }
         rangeValue = val;
         pStartEqualRange = pScan;
      }
      ++pScan;
   }
   if(aSingleFeatureValues == pSplittableValuesStart) {
      EBM_ASSERT(0 == cSplittingRanges);

      // we're still on the first splitting range.  We need to make sure that there is at least one possible cut
      // if we require 3 items for a cut, a problematic range like 0 1 3 3 4 5 could look ok, but we can't cut it in the middle!
      const FloatEbmType * pCheckForSplitPoint = aSingleFeatureValues + cMinimumInstancesPerBin;
      EBM_ASSERT(pCheckForSplitPoint <= pValuesEnd);
      const FloatEbmType * pCheckForSplitPointLast = pValuesEnd - cMinimumInstancesPerBin;
      EBM_ASSERT(aSingleFeatureValues <= pCheckForSplitPointLast);
      EBM_ASSERT(aSingleFeatureValues < pCheckForSplitPoint);
      FloatEbmType checkValue = *(pCheckForSplitPoint - 1);
      while(pCheckForSplitPoint <= pCheckForSplitPointLast) {
         if(checkValue != *pCheckForSplitPoint) {
            return 1;
         }
         ++pCheckForSplitPoint;
      }
      // there's no possible place to split, so return
      return 0;
   } else {
      const size_t cItemsLast = static_cast<size_t>(pValuesEnd - pSplittableValuesStart);
      if(cMinimumInstancesPerBin <= cItemsLast) {
         ++cSplittingRanges;
      }
      return cSplittingRanges;
   }
}

INLINE_RELEASE static size_t GetAvgLength(const size_t cInstances, const size_t cMaximumBins, const size_t cMinimumInstancesPerBin) {
   EBM_ASSERT(size_t { 1 } <= cInstances);
   EBM_ASSERT(size_t { 2 } <= cMaximumBins); // if there is just one bin, then you can't have splits, so we exit earlier
   EBM_ASSERT(size_t { 1 } <= cMinimumInstancesPerBin);

   // SplittingRanges are ranges of numbers that we have the guaranteed option of making at least one split within.
   // if there is only one SplittingRange, then we have no choice other than make cuts within the one SplittingRange that we're given
   // if there are multiple SplittingRanges, then every SplittingRanges borders at least one long range of equal values which are unsplittable.
   // cuts are a limited resource, so we want to spend them wisely.  If we have N cuts to give out, we'll first want to ensure that we get a cut
   // within each possible SplittingRange, since these things always border long ranges of unsplittable values.
   //
   // BUT, what happens if we have N SplittingRange, but only N-1 cuts to give out.  In that case we would have to make difficult decisions about where
   // to put the cuts
   //
   // To avoid the bad scenario of having to figure out which SplittingRange won't get a cut, we instead ensure that we can never have more SplittingRanges
   // than we have cuts.  This way every SplittingRanges is guaranteed to have at least 1 cut.
   // 
   // If our avgLength is the ceiling of cInstances / cMaximumBins, then we get this guarantee
   // but std::ceil works on floating point numbers, and it is inexact, especially if cInstances is above the point where floating point numbers can't
   // represent all integer values anymore (above 2^52)
   // so, instead of taking the std::ceil, we take the floor instead by just converting it to size_t, then we increment the avgLength until we
   // get our guarantee using integer math.  This gives us a true guarantee that we'll have sufficient cuts to give each SplittingRange at least one cut

   // Example of a bad situation if we took the rounded average of cInstances / cMaximumBins:
   // 20 == cInstances, 9 == cMaximumBins (so 8 cuts).  20 / 9 = 2.22222222222.  std::round(2.222222222) = 2.  So avgLength would be 2 if we rounded 20 / 9
   // but if our data is:
   // 0,0|1,1|2,2|3,3|4,4|5,5|6,6|7,7|8,8|9,9
   // then we get 9 SplittingRanges, but we only have 8 cuts to distribute.  And then we get to somehow choose which SplittingRange gets 0 cuts.
   // a better choice would have been to make avgLength 3 instead, so the ceiling.  Then we'd be guaranteed to have 8 or less SplittingRanges

   // our algorithm has the option of not putting cut points in the first and last SplittingRanges, since they could be cMinimumInstancesPerBin long
   // and have a long set of equal values only on one side, which means that a cut there isn't absolutely required.  We still need to take the ceiling
   // for the avgLength though since it's possible to create arbitrarily high number of missing bins.  We have a test that creates 3 missing bins, thereby
   // testing for the case that we don't give the first and last SplittingRanges an initial cut.  In this case, we're still missing a cut for one of the
   // long ranges that we can't fullfil.

   size_t avgLength = static_cast<size_t>(static_cast<FloatEbmType>(cInstances) / static_cast<FloatEbmType>(cMaximumBins));
   avgLength = UNPREDICTABLE(avgLength < cMinimumInstancesPerBin) ? cMinimumInstancesPerBin : avgLength;
   while(true) {
      if(UNLIKELY(IsMultiplyError(avgLength, cMaximumBins))) {
         // cInstances isn't an overflow (we checked when we entered), so if we've reached an overflow in the multiplication, 
         // then our multiplication result must be larger than cInstances, even though we can't perform it, so we're good
         break;
      }
      if(PREDICTABLE(cInstances <= avgLength * cMaximumBins)) {
         break;
      }
      ++avgLength;
   }
   return avgLength;
}

INLINE_RELEASE static size_t PossiblyRemoveBinForMissing(const bool bMissing, const IntEbmType countMaximumBins) {
   EBM_ASSERT(IntEbmType { 2 } <= countMaximumBins);
   size_t cMaximumBins = static_cast<size_t>(countMaximumBins);
   if(PREDICTABLE(bMissing)) {
      // if there is a missing value, then we use 0 for the missing value bin, and bump up all other values by 1.  This creates a semi-problem
      // if the number of bins was specified as a power of two like 256, because we now have 257 possible values, and instead of consuming 8
      // bits per value, we're consuming 9.  If we're told to have a maximum of a power of two bins though, in most cases it won't hurt to
      // have one less bin so that we consume less data.  Our countMaximumBins is just a maximum afterall, so we can choose to have less bins.
      // BUT, if the user requests 8 bins or less, then don't reduce the number of bins since then we'll be changing the bin size significantly

      size_t cBits = (~size_t { 0 }) ^ ((~size_t { 0 }) >> 1);
      do {
         // if cMaximumBins is a power of two equal to or greater than 16, then reduce the number of bins (it's a maximum after all) to one less so that
         // it's more compressible.  If we have 256 bins, we really want 255 bins and 0 to be the missing value, using 256 values and 1 byte of storage
         // some powers of two aren't compressible, like 2^34, which needs to fit into a 64 bit storage, but we don't want to take a dependency
         // on the size of the storage system, which is system dependent, so we just exclude all powers of two
         if(UNLIKELY(cBits == cMaximumBins)) {
            --cMaximumBins;
            break;
         }
         cBits >>= 1;
         // don't allow shrinkage below 16 bins (8 is the first power of two below 16).  By the time we reach 8 bins, we don't want to reduce this
         // by a complete bin.  We can just use an extra bit for the missing bin
         // if we had shrunk down to 7 bits for non-missing, we would have been able to fit in 21 items per data item instead of 16 for 64 bit systems
      } while(UNLIKELY(0x8 != cBits));
   }
   return cMaximumBins;
}

INLINE_RELEASE static size_t RemoveMissingValues(const size_t cInstances, FloatEbmType * const aValues) {
   FloatEbmType * pCopyFrom = aValues;
   const FloatEbmType * const pValuesEnd = aValues + cInstances;
   do {
      FloatEbmType val = *pCopyFrom;
      if(UNLIKELY(std::isnan(val))) {
         FloatEbmType * pCopyTo = pCopyFrom;
         goto skip_val;
         do {
            val = *pCopyFrom;
            if(PREDICTABLE(!std::isnan(val))) {
               *pCopyTo = val;
               ++pCopyTo;
            }
         skip_val:
            ++pCopyFrom;
         } while(LIKELY(pValuesEnd != pCopyFrom));
         const size_t cInstancesWithoutMissing = pCopyTo - aValues;
         EBM_ASSERT(cInstancesWithoutMissing < cInstances);
         return cInstancesWithoutMissing;
      }
      ++pCopyFrom;
   } while(LIKELY(pValuesEnd != pCopyFrom));
   return cInstances;
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateQuantileCutPoints(
   IntEbmType randomSeed,
   IntEbmType countInstances,
   FloatEbmType * singleFeatureValues,
   IntEbmType countMaximumBins,
   IntEbmType countMinimumInstancesPerBin,
   FloatEbmType * cutPointsLowerBoundInclusive,
   IntEbmType * countCutPoints,
   IntEbmType * isMissing,
   FloatEbmType * minValue,
   FloatEbmType * maxValue
) {
   EBM_ASSERT(0 <= countInstances);
   EBM_ASSERT(0 == countInstances || nullptr != singleFeatureValues);
   EBM_ASSERT(0 <= countMaximumBins);
   EBM_ASSERT(0 == countInstances || 0 < countMaximumBins); // countMaximumBins can only be zero if there are no instances, because otherwise you need a bin
   EBM_ASSERT(0 <= countMinimumInstancesPerBin);
   EBM_ASSERT(0 == countInstances || countMaximumBins <= 1 || nullptr != cutPointsLowerBoundInclusive);
   EBM_ASSERT(nullptr != countCutPoints);
   EBM_ASSERT(nullptr != isMissing);
   EBM_ASSERT(nullptr != minValue);
   EBM_ASSERT(nullptr != maxValue);

   // TODO: 
   //   - we shouldn't take a randomization seed from the user.. choosing the split points isn't that critical to have
   //       variability for.  We can do things like hashing the data, etc to choose random values, and we should REALLY
   //       try to not use randomness, instead using things like index position, etc for that
   //       One option would be to hash the value in a cell and use the hash.  it will be randomly distributed in direction!
   //   - we can't be 100% invariant to the direction the data is presented to us for binning, but we can be 99.99999% sure
   //     by doing a combination of:
   //       1) rounding, when not falling on the center value of a split
   //       2) if we fall on a center value, then prefer the inward direction
   //       3) if we happen to be at the index in the exact center, we can first use neighbouring splits and continue
   //          all the way to both tail ends
   //       4) if all the splits are the same from the center, we can use the values given to us to choose a direction
   //       5) if all of these things fail, we can use a random number

   LOG_N(TraceLevelInfo, "Entered GenerateQuantileCutPoints: randomSeed=%" IntEbmTypePrintf ", countInstances=%" IntEbmTypePrintf 
      ", singleFeatureValues=%p, countMaximumBins=%" IntEbmTypePrintf ", countMinimumInstancesPerBin=%" IntEbmTypePrintf 
      ", cutPointsLowerBoundInclusive=%p, countCutPoints=%p, isMissing=%p, minValue=%p, maxValue=%p", 
      randomSeed, 
      countInstances, 
      static_cast<void *>(singleFeatureValues), 
      countMaximumBins, 
      countMinimumInstancesPerBin, 
      static_cast<void *>(cutPointsLowerBoundInclusive), 
      static_cast<void *>(countCutPoints),
      static_cast<void *>(isMissing),
      static_cast<void *>(minValue),
      static_cast<void *>(maxValue)
   );

   if(!IsNumberConvertable<size_t, IntEbmType>(countInstances)) {
      LOG_0(TraceLevelWarning, "WARNING GenerateQuantileCutPoints !IsNumberConvertable<size_t, IntEbmType>(countInstances)");
      return 1;
   }

   if(!IsNumberConvertable<size_t, IntEbmType>(countMaximumBins)) {
      LOG_0(TraceLevelWarning, "WARNING GenerateQuantileCutPoints !IsNumberConvertable<size_t, IntEbmType>(countMaximumBins)");
      return 1;
   }

   if(!IsNumberConvertable<size_t, IntEbmType>(countMinimumInstancesPerBin)) {
      LOG_0(TraceLevelWarning, "WARNING GenerateQuantileCutPoints !IsNumberConvertable<size_t, IntEbmType>(countMinimumInstancesPerBin)");
      return 1;
   }

   const size_t cInstancesIncludingMissingValues = static_cast<size_t>(countInstances);

   if(0 == cInstancesIncludingMissingValues) {
      *countCutPoints = 0;
      *isMissing = EBM_FALSE;
      *minValue = 0;
      *maxValue = 0;
   } else {
      const size_t cInstances = RemoveMissingValues(cInstancesIncludingMissingValues, singleFeatureValues);

      const bool bMissing = cInstancesIncludingMissingValues != cInstances;
      *isMissing = bMissing ? EBM_TRUE : EBM_FALSE;

      if(0 == cInstances) {
         *countCutPoints = 0;
         *minValue = 0;
         *maxValue = 0;
      } else {
         FloatEbmType * const pValuesEnd = singleFeatureValues + cInstances;
         std::sort(singleFeatureValues, pValuesEnd);
         *minValue = singleFeatureValues[0];
         *maxValue = pValuesEnd[-1];
         if(countMaximumBins <= 1) {
            // if there is only 1 bin, then there can be no cut points, and no point doing any more work here
            *countCutPoints = 0;
         } else {
            const size_t cMinimumInstancesPerBin =
               countMinimumInstancesPerBin <= IntEbmType { 0 } ? size_t { 1 } : static_cast<size_t>(countMinimumInstancesPerBin);
            const size_t cMaximumBins = PossiblyRemoveBinForMissing(bMissing, countMaximumBins);
            EBM_ASSERT(2 <= cMaximumBins); // if we had just one bin then there would be no cuts and we should have exited above
            const size_t avgLength = GetAvgLength(cInstances, cMaximumBins, cMinimumInstancesPerBin);
            EBM_ASSERT(1 <= avgLength);
            const size_t cSplittingRanges = CountSplittingRanges(cInstances, singleFeatureValues, avgLength, cMinimumInstancesPerBin);
            // we GUARANTEE that each SplittingRange can have at least one cut by choosing an avgLength sufficiently long to ensure this property
            EBM_ASSERT(cSplittingRanges < cMaximumBins);
            if(0 == cSplittingRanges) {
               *countCutPoints = 0;
            } else {
               NeighbourJump * const aNeighbourJumps = ConstructJumps(cInstances, singleFeatureValues);
               if(nullptr == aNeighbourJumps) {
                  goto exit_error;
               }

               // TODO: limit cMaximumBins to a reasonable number based on the number of instances.
               //       if the user passes us the maximum size_t number, we shouldn't try and allocate
               //       that much memory

               // sometimes cut points will move between SplittingRanges, so we won't know an accurate
               // number of cut points, but we can be sure that we won't exceed the total number of cut points
               // so allocate the same number each time.  Hopefully we'll get back the same memory range each time
               // to avoid memory fragmentation.
               const size_t cSplitPointsMax = cMaximumBins - 1;

               // TODO: review if we still require these extra split point endpoints or not
               const size_t cSplitPointsWithEndpointsMax = cSplitPointsMax + 2; // include storage for the end points
               SplitPoint * const aSplitPoints = EbmMalloc<SplitPoint>(cSplitPointsWithEndpointsMax);

               if(nullptr == aSplitPoints) {
                  free(aNeighbourJumps);
                  goto exit_error;
               }

               RandomStream randomStream;
               randomStream.Initialize(randomSeed);

               // do this just once and reuse the random numbers
               FillSplitPointRandom(&randomStream, cSplitPointsWithEndpointsMax, aSplitPoints);

               const size_t cBytesCombined = sizeof(SplittingRange) + sizeof(SplittingRange *);
               if(IsMultiplyError(cSplittingRanges, cBytesCombined)) {
                  free(aSplitPoints);
                  free(aNeighbourJumps);
                  goto exit_error;
               }
               // use the same memory allocation for both the Junction items and the pointers to the junctions that we'll use for sorting
               SplittingRange ** const apSplittingRange = static_cast<SplittingRange **>(EbmMalloc<void>(cSplittingRanges * cBytesCombined));
               if(nullptr == apSplittingRange) {
                  free(aSplitPoints);
                  free(aNeighbourJumps);
                  goto exit_error;
               }
               SplittingRange * const aSplittingRange = reinterpret_cast<SplittingRange *>(apSplittingRange + cSplittingRanges);

               FillSplittingRangePointers(cSplittingRanges, apSplittingRange, aSplittingRange);
               FillSplittingRangeRandom(&randomStream, cSplittingRanges, aSplittingRange);

               FillSplittingRangeBasics(cInstances, singleFeatureValues, avgLength, cMinimumInstancesPerBin, cSplittingRanges, aSplittingRange);
               FillSplittingRangeNeighbours(cInstances, singleFeatureValues, cSplittingRanges, aSplittingRange);

#ifdef NEVER

               const size_t cUsedSplits = FillSplittingRangeRemaining(cSplittingRanges, aSplittingRange);

               size_t cCutsRemaining = cMaximumBins - 1 - cUsedSplits;
               cCutsRemaining = StuffSplitsIntoSplittingRanges(
                  cSplittingRanges,
                  aSplittingRange,
                  cMinimumInstancesPerBin,
                  cCutsRemaining
               );

               for(size_t i = 0; i < cSplittingRanges; ++i) {
                  size_t cCENTERSplitsAssigned = aSplittingRange[i].m_cSplitsAssigned;
                  if(0 == aSplittingRange[i].m_cUnsplittableEitherSideMin) {
                     // our first and last SplittingRanges can either have a long range of equal items on their tail ends
                     // or nothing.  If there is a long range of equal items, then we'll be placing one cut at the tail
                     // end, otherwise we have an implicit cut there and we don't need to use one of our cuts.  It's
                     // like getting a free cut, so increase the number of ranges by one if we don't need one cut at the tail
                     // side

                     ++cCENTERSplitsAssigned;
                     if(0 == aSplittingRange[i].m_cUnsplittableEitherSideMax) {
                        // if there's just one range and there are no long ranges on either end, then one split will create
                        // two ranges, so add 1 more.

                        ++cCENTERSplitsAssigned;
                     }
                  }
                  if(3 <= cCENTERSplitsAssigned) {
                     // take our the end splits
                     cCENTERSplitsAssigned -= 2;

                     std::set<SplitPoint *, CompareSplitPoint> bestSplitPoints;

                     // TODO : don't ignore the return value of TradeSplitSegment
                     TradeSplitSegment(
                        &bestSplitPoints,
                        cMinimumInstancesPerBin,
                        aSplittingRange[i].m_pSplittableValuesStart - singleFeatureValues,
                        aSplittingRange[i].m_cSplittableItems,
                        aNeighbourJumps,
                        cCENTERSplitsAssigned,
                        // for efficiency we include space for the end point cuts even if they don't exist
                        aSplitPoints
                     );
                  } else {
                     //EBM_ASSERT(false); // the condition of 1 split needs to be handled!
                  }
               }

#endif // NEVER



               //GetInterpretableCutPointFloat(0, 1);
               //GetInterpretableCutPointFloat(11, 12);
               //GetInterpretableCutPointFloat(345.33545, 3453.3745);
               //GetInterpretableCutPointFloat(0.000034533545, 0.0034533545);








               free(apSplittingRange); // both the junctions and the pointers to the junctions are in the same memory allocation

               // first let's tackle the short ranges between big ranges (or at the tails) where we know there will be a split to separate the big ranges to either
               // side, but the short range isn't big enough to split.  In otherwords, there are less than cMinimumInstancesPerBin items
               // we start with the biggest long ranges and essentially try to push whatever mass there is away from them and continue down the list

               *countCutPoints = 0;

               free(aSplitPoints);
               free(aNeighbourJumps);
            }
         }
      }
   }
   LOG_N(TraceLevelInfo, "Exited GenerateQuantileCutPoints countCutPoints=%" IntEbmTypePrintf ", isMissing=%" IntEbmTypePrintf,
      *countCutPoints,
      *isMissing
   );
   return 0;

exit_error:;
   LOG_N(TraceLevelWarning, "WARNING GenerateQuantileCutPoints returned %" IntEbmTypePrintf, 1);
   return 1;
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateImprovedEqualWidthCutPoints(
   IntEbmType countInstances,
   FloatEbmType * singleFeatureValues,
   IntEbmType countMaximumBins,
   FloatEbmType * cutPointsLowerBoundInclusive,
   IntEbmType * countCutPoints,
   IntEbmType * isMissing,
   FloatEbmType * minValue,
   FloatEbmType * maxValue
) {
   UNUSED(countInstances);
   UNUSED(singleFeatureValues);
   UNUSED(countMaximumBins);
   UNUSED(cutPointsLowerBoundInclusive);
   UNUSED(countCutPoints);
   UNUSED(isMissing);
   UNUSED(minValue);
   UNUSED(maxValue);

   // TODO: IMPLEMENT

   return 0;
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateEqualWidthCutPoints(
   IntEbmType countInstances,
   FloatEbmType * singleFeatureValues,
   IntEbmType countMaximumBins,
   FloatEbmType * cutPointsLowerBoundInclusive,
   IntEbmType * countCutPoints,
   IntEbmType * isMissing,
   FloatEbmType * minValue,
   FloatEbmType * maxValue
) {
   UNUSED(countInstances);
   UNUSED(singleFeatureValues);
   UNUSED(countMaximumBins);
   UNUSED(cutPointsLowerBoundInclusive);
   UNUSED(countCutPoints);
   UNUSED(isMissing);
   UNUSED(minValue);
   UNUSED(maxValue);

   // TODO: IMPLEMENT

   return 0;
}

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION Discretize(
   IntEbmType countCutPoints,
   const FloatEbmType * cutPointsLowerBoundInclusive,
   IntEbmType countInstances,
   const FloatEbmType * singleFeatureValues,
   IntEbmType * singleFeatureDiscretized
) {
   EBM_ASSERT(0 <= countCutPoints);
   EBM_ASSERT((IsNumberConvertable<size_t, IntEbmType>(countCutPoints))); // this needs to point to real memory, otherwise it's invalid
   EBM_ASSERT(0 == countInstances || 0 == countCutPoints || nullptr != cutPointsLowerBoundInclusive);
   EBM_ASSERT(0 <= countInstances);
   EBM_ASSERT((IsNumberConvertable<size_t, IntEbmType>(countInstances))); // this needs to point to real memory, otherwise it's invalid
   EBM_ASSERT(0 == countInstances || nullptr != singleFeatureValues);
   EBM_ASSERT(0 == countInstances || nullptr != singleFeatureDiscretized);

   if(IntEbmType { 0 } < countInstances) {
      const size_t cCutPoints = static_cast<size_t>(countCutPoints);
#ifndef NDEBUG
      for(size_t iDebug = 1; iDebug < cCutPoints; ++iDebug) {
         EBM_ASSERT(cutPointsLowerBoundInclusive[iDebug - 1] < cutPointsLowerBoundInclusive[iDebug]);
      }
# endif // NDEBUG
      const size_t cInstances = static_cast<size_t>(countInstances);
      const FloatEbmType * pValue = singleFeatureValues;
      const FloatEbmType * const pValueEnd = singleFeatureValues + cInstances;
      IntEbmType * pDiscretized = singleFeatureDiscretized;

      if(size_t { 0 } == cCutPoints) {
         do {
            const FloatEbmType val = *pValue;
            const IntEbmType result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 1 } : IntEbmType { 0 };
            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
      } else {
         const ptrdiff_t missingVal = static_cast<ptrdiff_t>(cCutPoints + size_t { 1 });
         const ptrdiff_t highStart = static_cast<ptrdiff_t>(cCutPoints - size_t { 1 });
         do {
            const FloatEbmType val = *pValue;
            ptrdiff_t middle = missingVal;
            if(!std::isnan(val)) {
               ptrdiff_t high = highStart;
               ptrdiff_t low = 0;
               FloatEbmType midVal;
               do {
                  middle = (low + high) >> 1;
                  EBM_ASSERT(ptrdiff_t { 0 } <= middle && static_cast<size_t>(middle) < cCutPoints);
                  midVal = cutPointsLowerBoundInclusive[static_cast<size_t>(middle)];
                  high = UNPREDICTABLE(midVal <= val) ? high : middle - ptrdiff_t { 1 };
                  low = UNPREDICTABLE(midVal <= val) ? middle + ptrdiff_t { 1 } : low;
               } while(LIKELY(low <= high));
               middle = UNPREDICTABLE(midVal <= val) ? middle + ptrdiff_t { 1 } : middle;
               EBM_ASSERT(ptrdiff_t { 0 } <= middle && middle <= static_cast<ptrdiff_t>(cCutPoints));
            }
            *pDiscretized = static_cast<IntEbmType>(middle);
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
      }
   }
}
