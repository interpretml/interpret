// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

//#define LOG_SUPERVERBOSE_DISCRETIZATION_ORDERED
//#define LOG_SUPERVERBOSE_DISCRETIZATION_UNORDERED

// TODO: use noexcept throughout our codebase (exception extern "C" functions) !  The compiler can optimize functions better if it knows there are no exceptions
// TODO: review all the C++ library calls, including things like std::abs and verify that none of them throw exceptions, otherwise use the C versions that provide this guarantee
// TODO: after we've found our cuts, generate the best interpretable cut points, then move 1% backwards and forwards
//       to pick the cut points with the lowest numbers of digits that are closest to the original cut points.  
//       Moving 1% either way should be acceptable.  Make it a parameter that can be used from internal python code 
//       but we shouldn't export this to the end user since it has limited usefullness.  We can do this efficiently
//       by moving to the 1% end points directly, calculating the best cut text cut points between the 1% up and
//       1% down boundaries.  Then we'll know how many digits we need.  The problem is that the number retunred
//       will be the numeric mid-point, but not the low digit text median.  So, we find out if our new point is
//       above or below our previous text number, then we increment or decrement our number by one at the least
//       significant bit, so our result has the same number of digits as the best one, but it is closest to the
//       best bin divisor in terms of # of samples that go into each of our low and high side bins

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

// TODO: Next steps:
// - separate this into different files: Discretization.cpp, Binning.cpp, and InterpretableNumerics.cpp
// 1) Ok, once we're sure the algorithm isn't completely off base, let's add automated tests for almost every path we
//    can put a breakpoint on below or that we can think is important
// 2) Do a complete review top to bottom review of this entire file.  Which is just the GenerateQuantileBinCuts 
//    system, and the the Discretize function.  Don't expand our already complex functionality unless necessary
// 3) Implement GenerateWinsorizedBinCuts and GenerateUniformBinCuts
// 4) expose everything in python and clean up the preprocessor stuff there and make the cut points per
//    additive_term all work, look at how this changes the visualization objects, and continue along the path of 
//    implementing the python changes we agreed on including generational binning, etc..
// 5) Run tests against the 200 datasets to see if we degraded performance in any detectable way
// 6) Put out a version in python with all of these changes.  Wait a few months
// 7) Come back later and improve on this algorithm per the TODOs in this file

               //   - add log notes to the interpretable cut points on errors
               //   - REVIEW ALL THE CUT POINTS IN TESTS..DO they all look good and consistent

// python binning types
//quantile
//quantile_humanized
//uniform
//winsorized



// Some general definitions:
//  - uncuttable range - a long contiguous series of feature values after sorting that have the same value, 
//    and are therefore not separable by binning.  In order for us to consider the range uncuttable, the number of
//    identical values in the range needs to be longer than the average number of values in a bin.  Example: if
//    we are given 15 bins max, and we have 150 values, then an uncuttable range needs to be 10 values at minimum
//  - CuttingRange - a contiguous series of values after sorting that we can attempt to find BinCuts within
//    because there are no long series of uncuttable values within the CuttingRange.
//  - CutPoint - the places where we cut one bin to annother
//  - cutPoint - the value we assign to a CutPoint that separates one bin from annother.  Example:
//    if we had the values [1, 2, 3, 4] and one CutPoint, a reasonable cutPoint would be 2.5.
//  - cut range - the values between two CutPoint

// TODO: increase the k_cutExploreDistance, and also increase our testing length to compensate
constexpr size_t k_cutExploreDistance = 20;
constexpr FloatEbmType k_percentageDeviationFromEndpointForInterpretableNumbers = FloatEbmType { 0.25 };
// 1073741824 is 2^30.  Using a power of two with no detail in the mantissa might help multiplication
constexpr FloatEbmType tweakIncrement = std::numeric_limits<FloatEbmType>::epsilon() * FloatEbmType { 1073741824 };


INLINE_ALWAYS constexpr static FloatEbmType GetTweakingMultiplePositive(const size_t iTweak) noexcept {
   return FloatEbmType { 1 } + tweakIncrement * static_cast<FloatEbmType>(iTweak);
}
INLINE_ALWAYS constexpr static FloatEbmType GetTweakingMultipleNegative(const size_t iTweak) noexcept {
   return FloatEbmType { 1 } - tweakIncrement * static_cast<FloatEbmType>(iTweak);
}

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

constexpr ptrdiff_t k_movementDoneCut = std::numeric_limits<ptrdiff_t>::lowest();
constexpr FloatEbmType k_priorityNoCutsPossible = std::numeric_limits<FloatEbmType>::lowest();
constexpr size_t k_valNotLegal = std::numeric_limits<size_t>::max();

constexpr FloatEbmType k_illegalAvgCuttableRangeWidthAfterAddingOneCut = std::numeric_limits<FloatEbmType>::lowest();

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

struct CutPoint final {
   CutPoint() = default; // preserve our POD status
   ~CutPoint() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   CutPoint *   m_pPrev;
   CutPoint *   m_pNext;

   // after cutting, we set m_iValAspirationalFloat to m_iVal to avoid needing if statements in some places
   FloatEbmType   m_iValAspirationalFloat;

   size_t         m_iVal;
   // m_cPredeterminedMovementOnCut is a valid number until we cut it.  After cutting we don't 
   // need a movement value, so we set it to k_cutValue and use it to detect whether this CutPoint was cut
   ptrdiff_t      m_cPredeterminedMovementOnCut;

   // the higher the m_priority, the more likely it is that it'll be chosen to cut
   FloatEbmType   m_priority;

   // the higher the m_uniqueTiebreaker, the more likely it is that it'll be chosen to cut (after considering priority)
   // the tiebreakers are ordered with symmetry in mind such that items are ranked first by distance to the end
   // points and secondly by a random number generator.  The randomness only comes into play to break ties when
   // comparing two BinCuts that have the same distance to their endpoints
   size_t         m_uniqueTiebreaker;

   INLINE_ALWAYS void SetCut() noexcept {
      m_cPredeterminedMovementOnCut = k_movementDoneCut;
   }
   INLINE_ALWAYS bool IsCut() const noexcept {
      return k_movementDoneCut == m_cPredeterminedMovementOnCut;
   }
};
static_assert(std::is_standard_layout<CutPoint>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<CutPoint>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<CutPoint>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

struct CuttingRange final {

   // we divide the space into long segments of uncuttable equal values separated by spaces where we can put
   // cuts, which we call CuttingRanges.  CuttingRanges can have zero or more items.  If they have zero
   // cuttable items, then the CuttingRange is just there to separate two uncuttable ranges on both sides.
   // The first and last CuttingRanges are special in that they can either have a long range of uncuttable
   // values on the tail end, or not.  If they have a tail consisting of a long range of uncutable values, then
   // we'll definetly want to have a cut point within the tail CuttingRange, but if there is no uncutable
   // range on the tail end, then having cuts within that range is more optional.
   // 
   // If the first few or last few values are unequal, and followed by an uncuttable range, then
   // we put the unequal values into the uncuttable range IF there are not enough of them to create a cut based
   // on our countSamplesPerBinMin value.
   // Example: If countSamplesPerBinMin == 3 and the avg bin size is 5, and the list is 
   // (1, 2, 3, 3, 3, 3, 3 | 4, 5, 6 | 7, 7, 7, 7, 7, 8, 9) -> then the only cuttable range is (4, 5, 6)

   CuttingRange() = default; // preserve our POD status
   ~CuttingRange() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   // TODO: m_cUncuttableHighValues is redundant in that the next higher cutting range has the same value.
   //       we can eliminate it here and allocate an extra "offset(m_cUncuttableHighValues) + sizeof(size_t)"
   // in the array for the extra one at the top
   size_t         m_cUncuttableHighValues;
   size_t         m_cUncuttableLowValues;

   // this can be zero if we're sandwitched between two uncuttable ranges, eg: 0, 0, 0, <CuttingRange here> 1, 1, 1
   size_t         m_cCuttableValues;
   FloatEbmType * m_pCuttableValuesFirst;

   size_t         m_uniqueTiebreaker;

   size_t         m_cRangesAssigned;

   FloatEbmType   m_avgCuttableRangeWidthAfterAddingOneCut;
   size_t         m_cRangesMax;
};
static_assert(std::is_standard_layout<CuttingRange>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<CuttingRange>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<CuttingRange>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

class CompareCuttingRange final {
public:
   INLINE_ALWAYS bool operator() (const CuttingRange * const & lhs, const CuttingRange * const & rhs) const noexcept {
      if(UNLIKELY(rhs->m_avgCuttableRangeWidthAfterAddingOneCut == lhs->m_avgCuttableRangeWidthAfterAddingOneCut)) {
         // NEVER check for exact equality (as a precondition is ok), since then we'd violate the weak ordering rule
         // https://medium.com/@shiansu/strict-weak-ordering-and-the-c-stl-f7dcfa4d4e07
         return UNPREDICTABLE(rhs->m_uniqueTiebreaker < lhs->m_uniqueTiebreaker);
      } else {
         // NEVER check for exact equality (as a precondition is ok), since then we'd violate the weak ordering rule
         // https://medium.com/@shiansu/strict-weak-ordering-and-the-c-stl-f7dcfa4d4e07
         return UNPREDICTABLE(rhs->m_avgCuttableRangeWidthAfterAddingOneCut < lhs->m_avgCuttableRangeWidthAfterAddingOneCut);
      }
   }
};

class CompareCutPoint final {
public:
   // TODO : check how efficient this is.  Is there a faster way to to this
   INLINE_ALWAYS bool operator() (const CutPoint * const & lhs, const CutPoint * const & rhs) const noexcept {
      if(UNLIKELY(rhs->m_priority == lhs->m_priority)) {
         // NEVER check for exact equality (as a precondition is ok), since then we'd violate the weak ordering rule
         // https://medium.com/@shiansu/strict-weak-ordering-and-the-c-stl-f7dcfa4d4e07
         return UNPREDICTABLE(rhs->m_uniqueTiebreaker < lhs->m_uniqueTiebreaker);
      } else {
         // NEVER check for exact equality (as a precondition is ok), since then we'd violate the weak ordering rule
         // https://medium.com/@shiansu/strict-weak-ordering-and-the-c-stl-f7dcfa4d4e07
         return UNPREDICTABLE(rhs->m_priority < lhs->m_priority);
      }
   }
};

INLINE_RELEASE_UNTEMPLATED size_t CalculateRangesMaximizeMin(
   const FloatEbmType sideDistance, 
   const FloatEbmType totalDistance, 
   const size_t cRanges,
   const size_t cRangesSideOriginal
) noexcept {
   // our goal is to, as much as possible, avoid having small ranges at the end.  We don't care as much
   // about having long ranges so much as small range since small ranges allow the boosting algorithm to overfit
   // more easily.

   EBM_ASSERT(2 <= cRanges); // we require there to be at least one range on the left and one range on the right
   EBM_ASSERT(0 <= sideDistance);
   EBM_ASSERT(sideDistance <= totalDistance);
   // provided FloatEbmType is a double, this shouldn't be able to overflow even if we're on a 128 bit computer
   // if FloatEbmType was a float we might be in trouble for extrememly large ranges and iVal values
   //
   // even with numeric instability, we shouldn't end up with a terrible result here since we only get numeric
   // issues if the number of ranges is huge, and we clip on both the low and high ranges below to handle issues
   // where rounding pushes us a bit over the numeric limits
   const size_t cRangesPlusOne = cRanges + size_t { 1 };
   const FloatEbmType result = static_cast<FloatEbmType>(cRangesPlusOne) * sideDistance / totalDistance;
   size_t cSide = static_cast<size_t>(result);
   cSide = std::max(size_t { 1 }, cSide); // don't allow zero ranges on the low side
   cSide = std::min(cSide, cRanges - 1); // don't allow zero ranges on the high side

#ifndef NDEBUG

   const FloatEbmType avg = std::min(sideDistance / cSide, (totalDistance - sideDistance) / (cRanges - cSide));
   if(2 <= cSide) {
      const size_t denominator1 = cSide - size_t { 1 };
      const size_t denominator2 = cRanges - cSide + size_t { 1 };
      const FloatEbmType avgOther = std::min(sideDistance / static_cast<FloatEbmType>(denominator1), (totalDistance - sideDistance) / static_cast<FloatEbmType>(denominator2));
      EBM_ASSERT(avgOther <= avg * 1.00001);
   }

   if(2 <= cRanges - cSide) {
      const size_t denominator1 = cSide + size_t { 1 };
      const size_t denominator2 = cRanges - cSide - size_t { 1 };
      const FloatEbmType avgOther = std::min(sideDistance / static_cast<FloatEbmType>(denominator1), (totalDistance - sideDistance) / static_cast<FloatEbmType>(denominator2));
      EBM_ASSERT(avgOther <= avg * 1.00001);
   }

#endif

   if(UNLIKELY(cSide != cRangesSideOriginal)) {
      // sometimes, "cRangesPlusOne * sideDistance == totalDistance" and when that happens we can get a situation
      // where symmetry breaks down as we round up when the numbers are in one orientation and round down (since
      // they are reversed) in the opposite direction.  By adding a slight bias towards keeping the original 
      // number of ranges we can avoid divergence on exact matches

      // move slighly in the direction towards zero to put us off exact integers and move us in a helpful direction
      const FloatEbmType multiple = UNPREDICTABLE(cSide < cRangesSideOriginal) ? 
         GetTweakingMultiplePositive(1) : GetTweakingMultipleNegative(1);

      cSide = static_cast<size_t>(result * multiple);

      // I don't see how our new cSide could be outside of boundaries since cRangesSideOriginal would need
      // to be 2 to even consider a 1 in the new range, and then it'd have to actually be less than 1.
      // Same thing on the top end, we'd have to skip over an entire range
      // But maybe, under some extreme floating point ranges, it might be possible, so keep these checks for now
      cSide = std::max(size_t { 1 }, cSide); // don't allow zero ranges on the low side
      cSide = std::min(cSide, cRanges - 1); // don't allow zero ranges on the high side
   }
   EBM_ASSERT(0 < cSide);
   EBM_ASSERT(cSide < cRanges);

   return cSide;
}

INLINE_RELEASE_UNTEMPLATED static FloatEbmType ArithmeticMean(
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
      int { k_cCharsFloatPrint } <= cCharsWithoutNullTerminator) 
   {
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
   strtol(pch, &endptr, 10);
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
   strcpy(&strTruncated[iTruncateTextAfter], &pStr[k_iExp]);

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
                  int { size_t { 1 } + k_cExponentTextDigits } < cCharsWithoutNullTerminator) 
               {
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

static FloatEbmType GetInterpretableCutPointFloat(
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
         LIKELY(!StringToFloatChopped(strAvg, size_t { 0 }, &lowChop, &highChop)))) 
      {
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
         LIKELY(!FloatToString(high, strHigh)) && LIKELY(!FloatToString(ret, strAvg)))) 
      {
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

static FloatEbmType GetInterpretableEndpoint(
   const FloatEbmType center,
   const FloatEbmType movementFromEnds
) noexcept {
   // TODO : add logs or asserts here when we find a condition we didn't think was possible, but that occurs

   EBM_ASSERT(!std::isnan(center));
   EBM_ASSERT(!std::isnan(movementFromEnds));
   EBM_ASSERT(!std::isinf(movementFromEnds));
   EBM_ASSERT(FloatEbmType { 0 } <= movementFromEnds);

   const FloatEbmType distance = k_percentageDeviationFromEndpointForInterpretableNumbers * movementFromEnds;

   FloatEbmType ret = center;
   // if the center is +-infinity then we'll always be farter away than the end cut points which can't be +-infinity
   // so return +-infinity so that our alternative cut point is rejected
   if(LIKELY(!std::isinf(ret))) {
      bool bNegative = false;
      if(PREDICTABLE(center < FloatEbmType { 0 })) {
         ret = -ret;
         bNegative = true;
      }

      const FloatEbmType lowBound = ret - distance;
      // lowBound can be a negative number, but can't be +-infinity
      EBM_ASSERT(!std::isnan(lowBound));
      EBM_ASSERT(!std::isinf(lowBound));

      const FloatEbmType highBound = ret + distance;
      // highBound can be +infinity, but can't be negative
      EBM_ASSERT(!std::isnan(highBound));
      EBM_ASSERT(0 <= highBound);

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

INLINE_RELEASE_UNTEMPLATED static void IronCuts() noexcept {
   // - TODO: POST-HEALING
   //   Our cutting algorithm is greedy and some of the early decisions might not have been optimal.  
   //   We can try and improve things after we're done by looking at small one by one movements that try and
   //   reduce the square error, or some other metric.  Here are some ideas:
   //   - we could find the smallest section and trying to expand it either way and slide the smallness on either
   //     side until we find a solution that improves on our old one.  Each side we'd push it only enough to make 
   //     things better. If we find that we can make a push that improves things, then we take that.  We'd need a 
   //     priority queue to indicate the smallest sections, or we could iteratively sweep the array (from both
   //     sides simultaneously to keep them invariant to ordering)
   //   - we could try pushing inwards from the outer regions.  Take a window size that we think is good and 
   //     try pushing inwards simultaneously from both sides such that no window is smaller than that
   //     size and keep examining the result for best squared error fit while it's happening.  We might end up
   //     squeezing smaller sized ranges to the center, which might actually be good since overfitting is probably
   //     happening more on the edges.  We should evalute this method on a large number of datasets, since we might
   //     get better results than our squared error from average length might imply
   //   - we might try making a sliding window of 5 cuts.  Delete the 5 cuts in between two boundaries and try
   //     5! * 2^5 (examine all orderings of cuts and all left/right choices).  Move from both edges simultaneously
   //     to the center and repeat several times.  This has the advantage that all examinations will have their
   //     endpoints fixed while being examined, but the end points will themselves be examined as the window
   //     moves along
}

static FloatEbmType CalculatePriority(
   const FloatEbmType iValLowerFloat,
   const FloatEbmType iValHigherFloat,
   const CutPoint * const pCutCur
) noexcept {
   EBM_ASSERT(!pCutCur->IsCut());

   // TODO: It's tempting to want to materialize cuts if both of it's neighbours are materialized, since our 
   // boundaries won't change.  In the future though we might someday move counts of ranges arround, and perhaps 
   // a cut point will be moved into our range before we make our actual cut.  We should probably therefore give 
   // a priority of zero to any CutPoint that has materialized cut points to either side so that it doesn't 
   // get materialized until the end.  For the same reason we probably want to significantly reduce the priority
   // of range with 2 aspirational cuts, since we already uderstand them well.  We don't want to make the priority
   // zero though since we want the algorithm to choose which of the 2 cuts should be chosen.  Perhaps we should
   // just multiply the priority by a tiny value for 1,2,3 cut ranges so that the algorithm favors deciding the
   // larger ones first and then settle these cuts that we have the power to exmaine combinatorially in our
   // BuildNeighbourhoodPlan function

   // if the m_iVal value was set to k_illegalIndex, then there are no legal cuts, 
   // so leave it with the most terrible possible priority
   FloatEbmType priority = k_priorityNoCutsPossible;
   if(LIKELY(k_valNotLegal != pCutCur->m_iVal)) {
      // TODO: This calculation doesn't take into account that we can trade our cut points with neighbours
      // with m_cPredeterminedMovementOnCut.  For an example, see test:
      // GenerateQuantileBinCuts, left+unsplitable+splitable+unsplitable+splitable
      // I'm not sure if this is bad or not.  In general, if we're swapping cut points, we're probably moving
      // pretty far, but I think if we're swaping cut points then we probably do in fact want to add priority
      // to those potential cut points since they are shuffling cut points around and we want to ensure that this
      // can still happen.  We migth even want to increase the priority of such even by first sorting on the
      // absolute value of m_cPredeterminedMovementOnCut, then by the priority.

      // TODO : these are not guaranteed due to floating point inexactness.  We should detect this scenario
      //        For now, we don't need to worry about violations of these.  It would take truely huge datasets
      //        to reach the big number required where changes in the floating point numbers exceeded integers
      EBM_ASSERT(iValLowerFloat < pCutCur->m_iVal); // it would violate cSamplesPerBinMin if these were equal
      EBM_ASSERT(iValLowerFloat < pCutCur->m_iValAspirationalFloat);
      EBM_ASSERT(pCutCur->m_iVal < iValHigherFloat); // it would violate cSamplesPerBinMin if these were equal
      EBM_ASSERT(pCutCur->m_iValAspirationalFloat < iValHigherFloat);

      // this metric considers proportional movement to be on the equality boundary.  So, if we've moved from
      // an aspirational value of 10 down to 5, that's equivalent in priority to a movement from 10 to 20.
      // the other option which might be considered is to measure the absolute movement, so movement from
      // 10 to 5 would be the same as movement from 10 to 15, but compression or expansion by 50% in either direction
      // is probably the right way to think about it since compressing small ranges is more damaging, and this metric
      // values movement towards the smaller end more.
      FloatEbmType priorityLow;
      FloatEbmType priorityHigh;
      if(pCutCur->m_iVal < pCutCur->m_iValAspirationalFloat) {
         priorityLow = (pCutCur->m_iValAspirationalFloat - iValLowerFloat) / (pCutCur->m_iVal - iValLowerFloat);
         priorityHigh = (iValHigherFloat - pCutCur->m_iVal) / (iValHigherFloat - pCutCur->m_iValAspirationalFloat);
      } else {
         priorityLow = (pCutCur->m_iVal - iValLowerFloat) / (pCutCur->m_iValAspirationalFloat - iValLowerFloat);
         priorityHigh = (iValHigherFloat - pCutCur->m_iValAspirationalFloat) / (iValHigherFloat - pCutCur->m_iVal);
      }

      // TODO : these are not guaranteed due to floating point inexactness.  We should detect this scenario
      EBM_ASSERT(FloatEbmType { 1 } <= priorityLow);
      EBM_ASSERT(FloatEbmType { 1 } <= priorityHigh);

      // TODO: evaluate max here instead as well

      // We could alternatively take the max, but multiplying these takes both sides into account in a nice way.
      // This does have the unfortunate effect of weighing the center cuts a bit higher than if we took the max, but 
      // it also has the nice property that it adds more information into the decision and therefore should have 
      // less close tiebreaker decisions
      //priority = std::max(priorityLow, priorityHigh);

      priority = priorityLow * priorityHigh;
      EBM_ASSERT(FloatEbmType { 1 } <= priority);

      // initiallly the space is divided into equal length ranges, so there are usually a lot of collisions
      // in priority from potential cuts on opposite sides of the value array.  We have a tiebreaker inside
      // our CutPoint to handle exact matches, but frequently due to floating point inexactness we find that
      // the priority isn't exactly equivalent from the top and bottom.  We generate a very small multiple which
      // is very very close to 1 in most cases, but it is enough to separate similar numbers.  If we have huge numbers
      // of potential cuts, then we might exceed 1 by a lot, but neighbouring cuts all have very similar values
      // and only differ by a small amount.  If we have such huge numbers of cuts, we probably want to focus
      // on the ends anyways, and our tiebreaker selection algorithm will put the higher priority numbers at
      // the tail ends, which is perfect
      
      priority *= GetTweakingMultiplePositive(pCutCur->m_uniqueTiebreaker);
   }

#ifdef LOG_SUPERVERBOSE_DISCRETIZATION_UNORDERED
   LOG_N(TraceLevelVerbose, "Prioritized CutPoint: %zu, %zu, %" FloatEbmTypePrintf ", %td, %" FloatEbmTypePrintf,
      pCutCur->m_uniqueTiebreaker,
      pCutCur->m_iVal,
      pCutCur->m_iValAspirationalFloat,
      pCutCur->m_cPredeterminedMovementOnCut,
      priority
   );
#endif // LOG_SUPERVERBOSE_DISCRETIZATION_UNORDERED

   return priority;
}

static void BuildNeighbourhoodPlan(
   const size_t cSamples,
   const bool bSymmetryReversal,
   const size_t cSamplesPerBinMin,
   const size_t iValuesStart,
   const size_t cCuttableItems,
   const NeighbourJump * const aNeighbourJumps,

   const size_t cRangesLow,
   const size_t iValLow,
   const FloatEbmType iValAspirationalLowFloat,

   const size_t cRangesHigh,
   const size_t iValHigh,
   const FloatEbmType iValAspirationalHighFloat,

   // m_iValAspirationalFloat and m_uniqueTiebreaker are the only values in pCurCut that are pre-initialized
   CutPoint * const pCutCur
) noexcept {

   EBM_ASSERT(1 <= cSamplesPerBinMin);
   EBM_ASSERT(2 <= cCuttableItems); // this is the min if cSamplesPerBinMin is 1 (the min for cSamplesPerBinMin)
   EBM_ASSERT(2 * cSamplesPerBinMin <= cCuttableItems);
   EBM_ASSERT(nullptr != aNeighbourJumps);

   EBM_ASSERT(1 <= cRangesLow);
   EBM_ASSERT(1 <= cRangesHigh);

   EBM_ASSERT(k_valNotLegal == iValLow || (iValAspirationalLowFloat * FloatEbmType { 0.9999 } <=
      static_cast<FloatEbmType>(iValLow) && static_cast<FloatEbmType>(iValLow) <=
      iValAspirationalLowFloat * FloatEbmType { 1.0001 }));

   EBM_ASSERT(k_valNotLegal == iValHigh || (iValAspirationalHighFloat * FloatEbmType { 0.9999 } <=
      static_cast<FloatEbmType>(iValHigh) && static_cast<FloatEbmType>(iValHigh) <=
      iValAspirationalHighFloat * FloatEbmType { 1.0001 }));

   EBM_ASSERT(iValAspirationalLowFloat < iValAspirationalHighFloat * FloatEbmType { 1.0001 });

   EBM_ASSERT(nullptr != pCutCur);

   // normally m_iValAspirationalFloat shouldn't get much smaller than cSamplesPerBinMin, although we don't
   // prevent our aspirational cuts from breaking the cSamplesPerBinMin barrier since the ultimate cut might
   // end up on the far side.  There's a huge gulf though from starting at cSamplesPerBinMin to the minimum
   // floating point, so much so that it should never get to zero
   EBM_ASSERT(FloatEbmType { 0 } < pCutCur->m_iValAspirationalFloat);
   EBM_ASSERT(pCutCur->m_iValAspirationalFloat <= static_cast<FloatEbmType>(cCuttableItems) * FloatEbmType { 1.0001 });

   // Before making any cuts, we examine each potential cut AS IF we were going to cut it, and we determine
   // which direction we would go in that instance.  After making all these future decisions for each aspirational
   // cut, our priority queue picks the decision that looks the hardest, and that'll make the most chaotic damage 
   // to our future plans.  We'll need to make those decisions someday anyways, and materializing those early gives
   // us more wiggle room to course correct as we greedily progress in our decision making.
   // 
   // The priority queue which determines the order that we materialize cuts tends to force us to make the hard 
   // decisions early. The hardest decisisions tend to be at the tail ends of CuttableRanges, since at center
   // of a long CuttingRange we can move future aspirational cuts long distances without affecting the average 
   // size of the remaining cuts much since we have lots of options for graudually steering the other cuts in a
   // direction afterwards.  At the tail ends though, one of our sides is fixed and unmovable, so we have to place our 
   // cuts with that restriction, and if our aspirational cut is in the middle of a long range of equal values we 
   // have to choose whether to make smaller ranges on the tail end side or on the open space side.  Usually we would 
   // choose to avoid small cut ranges on the constrained side, but there are cases where we shouldn't, like for 
   // instance if we're choosing a cut 2 ranges out from a tail end. Perhaps by choosing the inner cut we get a 
   // perfect cut between our tail and the cut we're materializing, so in reality we should probably explore our close 
   // neighbourhood a bit to ensure that a choice we're making has good downstream options.
   // example: if we have 0, 0, 0, 0 | 1, 1, 1, 1 | 2, 2 * 2, 2, 3 | 4, 5, 6 ... and our average cut size is 5, 
   // we might want to put the cut between the 1 and 2 instead of the 2 and 3 because we can nicely cut the 0s and 1s
   // afterwards. To make these choices we should examine our near zone neighbourhood and develop a contingency
   // plan for cuts on either side of the range we're making.  If a cut that we're going to materialize isn't
   // within a long range of equal values, then we can relax because the priority queue will only try to decide these
   // after all the hard decisions are already made, and by the time we reach there we're only moving small amounts
   // so future decisions tend to move our aspirational cut points by small amounts and not generate surprising
   // new hard to decide cuts.  As we progress in materializing cuts, if a cut becomes progressively harder, eventually
   // the prioritizing algorithm will notice and force us to materialize that cut. Given we cannot undo our choices,
   // once materialized, we should try to choose the safest options whenever we can't see past a certain horizon of
   // future choices.  If a materiazlied boundary is within our neighbourhood examination window, that's great, but
   // we also need to make choices when one or both sides are completely open.  In that case we don't know where
   // the future will materialize cuts on the open border side, so we should look at our aspirational cut at our
   // border and generate a certain number of reasonable points that could be possible (like 10 options).  Then
   // we should calculate what choices we'd make to each of those 10 possible destinations and then pick either
   // the worst potential one, or maybe the 2nd worst one (our prioritizing algoritm migh avoid the worst pick).

   // TODO: generate 4-20 potential landing points if either side of our window beyond the 5 or so neighbourhood
   //       cuts and develop a plan to getting to each of them, and then pick the worst (or 2nd worst) per above

   // TODO: We're currently NOT examining our near neighbourhood for where we'd put our neighbouring cuts.  We need
   //       to solve this, and here are some options:
   //       - the most computationally intensive one would be a mini version of our main cutting algorithm where
   //         we cut the neighbourhood into 5 aspirational cut points and we then materialize them one at a time
   //         This has the advantage that cuts will reposition themselves based on the others in our mini-neighbourhood
   //         The problem is that we have 5! orderings on which order to pick the mini-aspirational cuts, and 2 
   //         directions per choice, so we get 5! * 2 & 5 possible options (3,840) to examine.  That might be doable
   //       - we can divide the 5 cut neighbourhood into equally sized ranges and examine what happens if we go
   //         low/high on each cut.  This means we have only 2 ^ 5 options (32 options), but if one of the ranges
   //         puts one of our cuts in a bad place, it won't reposition the others
   //       - we can decide the cuts one at a time.  Start from our main low/high aspirational side, then decide
   //         the cut N/M items to the left or right, then keep going.  We can also reverse the decision making and
   //         go from our endpoint.  This might work in a lot of cases.  This method only requres 5 choices (or twice
   //         if we start from either end.  This might work in many cases
   //       - we could take an alternate approach here and look at N lower and N higher points based on our ideal 
   //         division width, and get the square distance between the ideal cut points and their nearest real 
   //         cuttable points.  This doesn't build an exact plan, but it's probably easier (I think we should probably
   //         try the other ideas above out first).
   //
   // TODO: we also want to try tweaking the number of cuts on either side.  Perhaps we determined at first that
   //       5 cuts on the lower side is ideal for minimizing the minimim cut range size, but when we go and examine
   //       that side maybe we have 4 perfect cutting points, but 5 doesn't fit.  We should try to go 2 up or 2 down
   //       or keep going until we get worse results.  I favor going 2 up and 2 down at minimum since it isn't
   //       guaranteed to have a linear improvement rate, and maybe go to 3, 4, etc.. if 2 improves things.  Perhaps
   //       we want to go one direction until we don't see an improvement for 2 successive changes
   // 
   // TODO: our priority queue priority should probably somewhat incorporate how good or bad our neighbourhood options
   //       are.  If we have an open ended side, and we have one great option, but all the other options are terrible
   //       then we probably want to raise our priority so that we get to go first and materialize our only good option
   //       this should probably be a secondary consideration to the global priority though since the global priority
   //       measures how much downstream chances a option changes, which is important to minimize.  We could probably
   //       consider returning a multiple here that we would muliply our global priority by.  Say we could change
   //       the global priority by a factor of 2 either lower or higher.
   //
   // TODO: IF there are no hard decisions to make near the tail ends of the windows, we might want to switch tracts
   //       and consider how we're going to thread the needle within the interior space.  We might use a completely
   //       different algorithm that slides a window of 5 cut points along the values and finds windows that are
   //       really bad that we'd like to avoid, and comparatively good cut points that we wnat to aim for.  This is
   //       very likely a secondary consideration to make after we've made the hard choices at the tail end
   //       and our priority queue might already avoid bad outcomes in many cases anyways.  Let's see if we can find
   //       a bad outcome that needs to be solved.

   const size_t cRanges = cRangesLow + cRangesHigh; // after this the compiler can forget cRangesHigh

   // often we'll find that pCutCur->m_iValAspirationalFloat will be an exact integer because we're being slotted
   // between two other exact integers.  When pCutCur->m_iValAspirationalFloat happens to be exactly the integerized
   // version, then we find that when we access aNeighbourJumps we'll get different ending up points on flipped
   // symmetric input data.  This means that we'll flip the side we jump to afterwards.  By adding a tiny bit of
   // noise that correlated to the input data direction, we can ensure that in the vast vast majority of cases
   // that we don't fall on an exact integer boundary anymore and therefore this problem goes away from a practical
   // point of view. 
   //   
   // using m_iValAspirationalFloat without tweaking it a bit is problematic due to the fact that often we're
   // dividing up a space with an integer number of items by an integer number of cuts which leaves us with an integer
   // m_iValAspirationalFloat.  In that case, if we're doing a symmetric reversal of the input data, we land on the
   // same integer in both directions.  In that case our resulting iStartNext is different and we sometimes get
   // different results due to the fact that the m_uniqueTiebreaker value will be different when we process it in
   // one direction or the other

   // Ideally, we'd have a second set of random numbers here that we wouldn't share with the priority
   // tweaking function that uses m_uniqueTiebreaker, but practically speaking this shouldn't make any difference

   const bool bLocalSymmetryReversal = (0 != (size_t { 1 } & pCutCur->m_uniqueTiebreaker)) != bSymmetryReversal;
   const FloatEbmType smallTweak = bLocalSymmetryReversal ? GetTweakingMultiplePositive(1) : GetTweakingMultipleNegative(1);

   size_t iValAspirationalCur = static_cast<size_t>(smallTweak * pCutCur->m_iValAspirationalFloat);
   if(UNLIKELY(cCuttableItems <= iValAspirationalCur)) {
      // handle the very very unlikely situation where m_iAspirationalFloat rounds up to 
      // cCuttableItems due to floating point issues
      iValAspirationalCur = cCuttableItems - 1;
   }

   const NeighbourJump * const pNeighbourJump = &aNeighbourJumps[iValuesStart + iValAspirationalCur];

   const size_t iStartCur = pNeighbourJump->m_iStartCur;
   const size_t iStartNext = pNeighbourJump->m_iStartNext;

   EBM_ASSERT(iStartCur < iStartNext);
   EBM_ASSERT(iValuesStart <= iStartCur); // since iValAspirationalCur can't be negative
   EBM_ASSERT(iValuesStart <= iStartNext); // since iValAspirationalCur can't be negative

   // it shouldn't be possible to have iValAspirationalCur even close to zero, since normally the lowest value
   // would be 1, and we have a lot of resultion in floating point numbers near zero, and we always calculate
   // these values starting from the low value and adding up (since there's more resolution in low numbers)
   // on the upper end though there are failure cases where if we had sufficiently huge numbers we might
   // find that we got back a Neighbour Jump above our legal range due to floating point inexactness when rounding
   // up.  We check for this condition below though

   const ptrdiff_t iValLowChoice = 
      static_cast<ptrdiff_t>(iStartCur) - static_cast<ptrdiff_t>(iValuesStart);
   const ptrdiff_t iValHighChoice = 
      static_cast<ptrdiff_t>(iStartNext) - static_cast<ptrdiff_t>(iValuesStart);

   FloatEbmType totalDistance;
   FloatEbmType distanceLowLowFloat;
   FloatEbmType distanceHighLowFloat;
   bool bCanCutLow;
   bool bCanCutHigh;

   const ptrdiff_t lowHighBound = iValLowChoice + static_cast<ptrdiff_t>(cSamplesPerBinMin);
   const ptrdiff_t highHighBound = iValHighChoice + static_cast<ptrdiff_t>(cSamplesPerBinMin);
   if(UNLIKELY(k_valNotLegal == iValLow)) {
      // we always start from the low index because for floating points the low numbers have more resolution
      totalDistance = iValAspirationalHighFloat - iValAspirationalLowFloat;
      distanceLowLowFloat = static_cast<FloatEbmType>(iValLowChoice) - iValAspirationalLowFloat;
      distanceHighLowFloat = static_cast<FloatEbmType>(iValHighChoice) - iValAspirationalLowFloat;

      const ptrdiff_t lowLowBoundPtrdiff = iValLowChoice - static_cast<ptrdiff_t>(cSamplesPerBinMin);
      const FloatEbmType lowLowBoundFloat = static_cast<FloatEbmType>(lowLowBoundPtrdiff);
      const ptrdiff_t highLowBoundPtrdiff = iValHighChoice - static_cast<ptrdiff_t>(cSamplesPerBinMin);
      const FloatEbmType highLowBoundFloat = static_cast<FloatEbmType>(highLowBoundPtrdiff);
      if(UNLIKELY(k_valNotLegal == iValHigh)) {
         // given all our indexes and counts refer to an existing array with more than 4 bytes, 
         // they should not be able to overflow when adding any of these numbers

         // since we always start from the low end, I don't think we can ever get a number less than zero, which
         // is a preceise floating point value also.
         EBM_ASSERT(FloatEbmType { 0 } <= iValAspirationalLowFloat);

         // check our soft bounds and hard bounds (to avoid floating point issues)
         bCanCutLow = LIKELY(LIKELY(iValAspirationalLowFloat <= lowLowBoundFloat) &&
            LIKELY(static_cast<FloatEbmType>(lowHighBound) <= iValAspirationalHighFloat) &&
            LIKELY(lowHighBound <= static_cast<ptrdiff_t>(cCuttableItems)));

         bCanCutHigh = LIKELY(LIKELY(static_cast<FloatEbmType>(highHighBound) <= iValAspirationalHighFloat) &&
            LIKELY(iValAspirationalLowFloat <= highLowBoundFloat) &&
            LIKELY(highHighBound <= static_cast<ptrdiff_t>(cCuttableItems)));
      } else {
         // given all our indexes and counts refer to an existing array with more than 4 bytes, 
         // they should not be able to overflow when adding any of these numbers

         EBM_ASSERT(iValHigh <= cCuttableItems);
         // since we always start from the low end, I don't think we can ever get a number less than zero, which
         // is a preceise floating point value also.
         EBM_ASSERT(FloatEbmType { 0 } <= iValAspirationalLowFloat);

         // check our soft bounds and hard bounds (to avoid floating point issues)
         bCanCutLow = LIKELY(LIKELY(iValAspirationalLowFloat <= lowLowBoundFloat) &&
            LIKELY(lowHighBound <= static_cast<ptrdiff_t>(iValHigh)));

         bCanCutHigh = LIKELY(LIKELY(highHighBound <= static_cast<ptrdiff_t>(iValHigh)) &&
            LIKELY(iValAspirationalLowFloat <= highLowBoundFloat));
      }
   } else {
      // even though our lower boundary is materialized, if there are a huge number of items, then we can reach 
      // a region where doubles can't represent integer numbers and we could concivably be outside our
      // bounds for floating point inexact reasons.
      //
      // iValLowChoice, iValHighChoice, and iValLow values should be convertible to a ptrdiff_t, since they refer to indexes
      // for data structures much larger than 2 bytes, so we should have room for the negatives here
      const ptrdiff_t distanceLowPtrdiffT = iValLowChoice - static_cast<ptrdiff_t>(iValLow);
      const ptrdiff_t distanceHighPtrdiffT = iValHighChoice - static_cast<ptrdiff_t>(iValLow);
      distanceLowLowFloat = static_cast<FloatEbmType>(distanceLowPtrdiffT);
      distanceHighLowFloat = static_cast<FloatEbmType>(distanceHighPtrdiffT);
      if(UNLIKELY(k_valNotLegal == iValHigh)) {
         totalDistance = iValAspirationalHighFloat - iValAspirationalLowFloat;

         // given all our indexes and counts refer to an existing array with more than 4 bytes, 
         // they should not be able to overflow when adding any of these numbers

         // check our soft bounds and hard bounds (to avoid floating point issues)
         bCanCutLow = LIKELY(LIKELY(static_cast<ptrdiff_t>(cSamplesPerBinMin) <= distanceLowPtrdiffT) &&
            LIKELY(static_cast<FloatEbmType>(lowHighBound) <= iValAspirationalHighFloat) &&
            LIKELY(lowHighBound <= static_cast<ptrdiff_t>(cCuttableItems)));

         bCanCutHigh = LIKELY(LIKELY(static_cast<FloatEbmType>(highHighBound) <= iValAspirationalHighFloat) &&
            LIKELY(static_cast<ptrdiff_t>(cSamplesPerBinMin) <= distanceHighPtrdiffT) &&
            LIKELY(highHighBound <= static_cast<ptrdiff_t>(cCuttableItems)));
      } else {
         // reduce floating point noise when we have have exact distances
         totalDistance = static_cast<FloatEbmType>(iValHigh - iValLow);

         // given all our indexes and counts refer to an existing array with more than 4 bytes, 
         // they should not be able to overflow when adding any of these numbers

         EBM_ASSERT(iValHigh <= cCuttableItems);

         bCanCutLow = LIKELY(LIKELY(static_cast<ptrdiff_t>(cSamplesPerBinMin) <= distanceLowPtrdiffT) &&
            LIKELY(lowHighBound <= static_cast<ptrdiff_t>(iValHigh)));

         bCanCutHigh = LIKELY(LIKELY(highHighBound <= static_cast<ptrdiff_t>(iValHigh)) &&
            LIKELY(static_cast<ptrdiff_t>(cSamplesPerBinMin) <= distanceHighPtrdiffT));
      }
   }

   constexpr FloatEbmType k_badScore = std::numeric_limits<FloatEbmType>::lowest();

   FloatEbmType scoreHigh;
   ptrdiff_t transferRangesHigh;

   if(LIKELY(bCanCutHigh)) {
      {
         const size_t cRangesHighLow = CalculateRangesMaximizeMin(distanceHighLowFloat, totalDistance, cRanges, cRangesLow);
         EBM_ASSERT(1 <= cRangesHighLow);
         EBM_ASSERT(cRangesHighLow < cRanges);
         const size_t cRangesHighHigh = cRanges - cRangesHighLow;
         EBM_ASSERT(1 <= cRangesHighHigh);

         FloatEbmType distanceHigh;
         if(UNLIKELY(k_valNotLegal == iValHigh)) {
            distanceHigh = iValAspirationalHighFloat - static_cast<FloatEbmType>(iValHighChoice);
         } else {
            const ptrdiff_t distanceHighPtrdiff = static_cast<ptrdiff_t>(iValHigh) - iValHighChoice;
            distanceHigh = static_cast<FloatEbmType>(distanceHighPtrdiff);
         }
         const FloatEbmType avgLengthHighHigh = distanceHigh / cRangesHighHigh;
         const FloatEbmType avgLengthHighLow = distanceHighLowFloat / cRangesHighLow;

         scoreHigh = std::min(avgLengthHighLow, avgLengthHighHigh);
         transferRangesHigh = static_cast<ptrdiff_t>(cRangesHighLow) - static_cast<ptrdiff_t>(cRangesLow);
      }

      FloatEbmType scoreLow;
      ptrdiff_t transferRangesLow;

      if(LIKELY(bCanCutLow)) {

      do_low:;

         const size_t cRangesLowLow = CalculateRangesMaximizeMin(distanceLowLowFloat, totalDistance, cRanges, cRangesLow);
         EBM_ASSERT(1 <= cRangesLowLow);
         EBM_ASSERT(cRangesLowLow < cRanges);
         const size_t cRangesLowHigh = cRanges - cRangesLowLow;
         EBM_ASSERT(1 <= cRangesLowHigh);

         FloatEbmType distanceHigh;
         if(UNLIKELY(k_valNotLegal == iValHigh)) {
            distanceHigh = iValAspirationalHighFloat - static_cast<FloatEbmType>(iValLowChoice);
         } else {
            const ptrdiff_t distanceHighPtrdiff = static_cast<ptrdiff_t>(iValHigh) - iValLowChoice;
            distanceHigh = static_cast<FloatEbmType>(distanceHighPtrdiff);
         }
         const FloatEbmType avgLengthLowHigh = distanceHigh / cRangesLowHigh;
         const FloatEbmType avgLengthLowLow = distanceLowLowFloat / cRangesLowLow;

         scoreLow = std::min(avgLengthLowLow, avgLengthLowHigh);
         transferRangesLow = static_cast<ptrdiff_t>(cRangesLowLow) - static_cast<ptrdiff_t>(cRangesLow);
      } else {
         scoreLow = k_badScore;
         transferRangesLow = 0;
      }

      EBM_ASSERT(k_badScore != scoreHigh || k_badScore != scoreLow);

      if(UNPREDICTABLE(scoreHigh < scoreLow * GetTweakingMultipleNegative(1))) {
         pCutCur->m_iVal = static_cast<size_t>(iValLowChoice);
         pCutCur->m_cPredeterminedMovementOnCut = transferRangesLow;
      } else if(LIKELY(scoreLow < scoreHigh * GetTweakingMultipleNegative(1))) {
         pCutCur->m_iVal = static_cast<size_t>(iValHighChoice);
         pCutCur->m_cPredeterminedMovementOnCut = transferRangesHigh;
      } else {
         ptrdiff_t transferRangesLowAbs = transferRangesLow;
         if(UNLIKELY(transferRangesLowAbs < ptrdiff_t { 0 })) {
            transferRangesLowAbs = -transferRangesLowAbs;
         }
         ptrdiff_t transferRangesHighAbs = transferRangesHigh;
         if(UNLIKELY(transferRangesHighAbs < ptrdiff_t { 0 })) {
            transferRangesHighAbs = -transferRangesHighAbs;
         }
         if(transferRangesLowAbs != transferRangesHighAbs) {
            // very very occasionally we get a situation where the priorities are equal because our aspirational
            // cut lands exactly on the optimal cut AND thus our other side cuts are different in a reversed symmetry
            // scenario (aNeighbourJumps is reversed but has the same index).  And if all of that is the case, AND
            // also the other sided cut happens to be exactly on the ideal separation boundary, then we
            // can see that transfering the cut point allows us to change the number of ranges.  To guard against
            // this we select the cut that minimizes the transfer, which is a good goal in itself
            if(UNPREDICTABLE(transferRangesLowAbs < transferRangesHighAbs)) {
               pCutCur->m_iVal = static_cast<size_t>(iValLowChoice);
               pCutCur->m_cPredeterminedMovementOnCut = transferRangesLow;
            } else {
               EBM_ASSERT(transferRangesHighAbs < transferRangesLowAbs);
               pCutCur->m_iVal = static_cast<size_t>(iValHighChoice);
               pCutCur->m_cPredeterminedMovementOnCut = transferRangesHigh;
            }
         } else {
            // next, let's try to the edges of our full array
            const size_t cDistanceLow = iStartCur;
            EBM_ASSERT(iStartNext <= cSamples);
            const size_t cDistanceHigh = cSamples - iStartNext;
            if(UNPREDICTABLE(cDistanceHigh < cDistanceLow)) {
               pCutCur->m_iVal = static_cast<size_t>(iValLowChoice);
               pCutCur->m_cPredeterminedMovementOnCut = transferRangesLow;
            } else if(LIKELY(cDistanceLow < cDistanceHigh)) {
               pCutCur->m_iVal = static_cast<size_t>(iValHighChoice);
               pCutCur->m_cPredeterminedMovementOnCut = transferRangesHigh;
            } else {
               // we're at the center of the entire array. Our final fallback is to resort to our symmetric determination
               if(UNPREDICTABLE(bLocalSymmetryReversal)) {
                  pCutCur->m_iVal = static_cast<size_t>(iValLowChoice);
                  pCutCur->m_cPredeterminedMovementOnCut = transferRangesLow;
               } else {
                  pCutCur->m_iVal = static_cast<size_t>(iValHighChoice);
                  pCutCur->m_cPredeterminedMovementOnCut = transferRangesHigh;
               }
            }
         }
      }

#ifdef LOG_SUPERVERBOSE_DISCRETIZATION_UNORDERED
      LOG_N(TraceLevelVerbose, "Plan CutPoint: %zu, %zu, %" FloatEbmTypePrintf ", %td, %" FloatEbmTypePrintf ", %" FloatEbmTypePrintf,
         pCutCur->m_uniqueTiebreaker,
         pCutCur->m_iVal,
         pCutCur->m_iValAspirationalFloat,
         pCutCur->m_cPredeterminedMovementOnCut,
         scoreLow, 
         scoreHigh
      );
#endif // LOG_SUPERVERBOSE_DISCRETIZATION_UNORDERED

   } else if(LIKELY(bCanCutLow)) {
      scoreHigh = k_badScore;
      transferRangesHigh = ptrdiff_t { 0 };

      goto do_low;

   } else {
      // can't cut either high or low, so exit indicating we're at an impossible cut
      pCutCur->m_iVal = k_valNotLegal;
      pCutCur->m_cPredeterminedMovementOnCut = 0; // set this to indicate that we aren't cut

#ifdef LOG_SUPERVERBOSE_DISCRETIZATION_UNORDERED
      LOG_0(TraceLevelVerbose, "Plan CutPoint: DENIED");
#endif // LOG_SUPERVERBOSE_DISCRETIZATION_UNORDERED
   }
   EBM_ASSERT(!pCutCur->IsCut());
}

static bool CutCuttingRange(
   std::set<CutPoint *, CompareCutPoint> * const pBestBinCuts,

   const size_t cSamples,
   const bool bSymmetryReversal,
   const size_t cSamplesPerBinMin,

   const size_t iValuesStart,
   const size_t cCuttableItems,
   const NeighbourJump * const aNeighbourJumps
) noexcept {
   EBM_ASSERT(nullptr != pBestBinCuts);

   EBM_ASSERT(2 <= cSamples); // we wouldn't be cutting this if there weren't two potential bins

   EBM_ASSERT(1 <= cSamplesPerBinMin);

   // we need to be able to put down at least one cut not at the edges
   EBM_ASSERT(2 <= cCuttableItems); // we wouldn't be cutting this if there weren't two potential bins
   EBM_ASSERT(2 <= cCuttableItems / cSamplesPerBinMin);
   EBM_ASSERT(cCuttableItems <= cSamples);
   EBM_ASSERT(nullptr != aNeighbourJumps);

   // TODO: someday, for performance, it might make sense to use a non-allocating tree, like:
   //       https://github.com/attractivechaos/klib/blob/master/kavl.h

   try {
      while(!pBestBinCuts->empty()) {
         // We've located our desired cut points previously.  Sometimes those desired cut points
         // are placed in the bulk of a long run of identical values and we have to decide if we'll be putting
         // the cut at the start or the end of those long runs of identical values.
         //
         // Before this function in the call stack, we do some expensive exploration of the hardest cut point placement
         // decisions that we need to make.  We do a full exploration of both the lower and higher placements of the long 
         // runs, but that grows at O(2^N), so we need to limit this full exploration to just a few choices
         //
         // This function being lower in the stack needs to decide whether to place the cut at the lower or higher
         // position without the benefit of an in-depth exploration of both options across all possible other cut points 
         // (local exploration is still ok though).  We need to choose one side and live with that decision, so we 
         // look at all our potential cuts, and we greedily pick out the cut that is 
         // really nice on one side, but really bad on the other, and we keep greedily picking cuts this way until they 
         // are all selected.  We use a priority queue to efficiently find the most important cut at any given time.
         // Now we have an O(N * log(N)) algorithm in principal, but it's still a bit worse than that.
         //
         // After we decide whether to put the cut at the start or end of a run, we're actualizing the location of 
         // the cut and we'll be changing the size of the runs to our left and right since they'll either have
         // actualized or desired cut points, or the immutable ends as neighbours.  We'd prefer to spread out the
         // movement between our desired and actual cut points into all our potential neighbours instead of to the 
         // immediately bordering ranges.  Ideally, we'd like to spread out and re-calculate all other cut points 
         // until we reach the immovable boundaries of an already decided cut, or the ends, after we've decided 
         // on each cut. So, if we had 255 cuts, we'd choose one, then re-calculate the cut points of the remaining 
         // 254, but that is clearly bad computationally, since then our algorithm would be O(N^2 * log(N)).  For low 
         // numbers like 255 it might be fine, but our user could choose much larger numbers of cuts, and then 
         // it would become intractable.
         //
         // Instead of re-calculating all remaining 255 cut points though, we instead choose a window 
         // of influence.  So, if our influence window was set to 50 cut points, then even if we had to move one 
         // cut point by a large amount of almost a complete cut range, we'd only impact the neighboring 50 ranges 
         // by 2% (1/50).
         //
         // After we choose whether to go to the start or the end, we then choose an anchor point 50 to the 
         // left and another one 50 to the right, unless we hit a materialized cut point, or the end, which we can't 
         // move.  All the 50 items to the left and right either grow a bit smaller, or a bit bigger anchored to 
         // the influence region ends.
         //
         // After calculating the new sizes of the ranges and the new desired cut points, we can then remove
         // the 50-ish items on both sides from our priority queue, which in fact needs to be a tree so that we can
         // remove items that aren't just the lowest value, and we can re-add them to the tree with their new
         // recalculated priority score.
         //
         // But the scores of the desired cut points outside of our window have changed slightly too!  
         // The 50th, 51st, 52nd, etc, items to the left and the right weren't moved, but they can still "see" cut 
         // points that are within our influence window of desired cut points that we changed, so their priorty 
         // scores need to change.  Once we get to twice the window size though, the items beyond that can't be 
         // affected, so we only need to update items within a four time range of our window size, 
         // two on the left and two on the right.
         //
         // If we have our window set to larger than the number of cuts, then we'll effectively be re-doing all
         // the cuts, which might be ok for small N.  In that case all the cuts would always have the same width
         // In our modified world, we get divergence over time, but since we're limiting our change to a small
         // percentage, we shouldn't get too far out of whack.  Also, we'll quickly put down actualized cutting points such
         // that afer a few we'll probably find ourselves close to a previous cut point and we'll proceed by
         // updating all the priority scores exactly since we'll hit the already decided cuts before the 
         // influence window length
         //
         // initially, we have pre-calculated which direction each cut should go, and we've calculated how many
         // cut points should move between our right and left sides, and we also previously calculated a priority for
         // making decisions. When we pull one potential cut point off the queue, we need to nuke all our decisions
         // within the 50 item window on both sides (or until we hit an imovable boundary) and then we need to
         // recalculate for each cut which way it should go and what it's priority is
         //
         // At this point we're re-doing our cuts within the 50 item cut window and we need to decide two things:
         //   1) calculate the direction we'd go for each new cut point, and how many cuts we'd move from our right 
         //      and left to the other side
         //   2) Calculate the priority of making the decision
         //
         // If we do a full local exploration of where we're going to do our cuts for any single cut, then we can
         // do a better job at calculating the priority, since we'll know how many cuts will be moved from right to left
         //
         // When doing a local exploration, examine going right left on each N segments to each side
         //
         // So, our process is:
         //    1) Pull a high priority item from the queue (which has a pre-calculated direction to cut and all other
         //       cutting decisions already calculated beforehand)
         //    1) execute our pre-determined cut placement AND move cuts from one side to the other if called for in our pre-plan
         //    2) re-calculate the aspirational cut points and for each of those do a first pass combination exploration
         //       to choose the best materialied cut point based on just ourselves
         //    3) Re-pass through our semi-materialized cuts points and jiggle them as necessary against their neighbours
         //       since the "view of the world" is different for each cut point and they don't match perfectly even if
         //       they are often close.
         //    4) Pass from the center to the outer-outer boundary (twice the boundary distance), and remove cuts from
         //       our priority queue, then calculate our new priority which is based on the squared change in all
         //       aspirational cut point (either real or just assuming equal cutting after the cut)
         //       And re-add them with our newly calculated priority, which can examine any cuts within
         //       the N item window at any point (but won't change them)

         // all fields of our pCutBest should have been filled with initialized data previously
         CutPoint * const pCutBest = *pBestBinCuts->begin();

#ifdef LOG_SUPERVERBOSE_DISCRETIZATION_ORDERED
         LOG_N(TraceLevelVerbose, "Dequeue CutPoint: %zu, %zu, %" FloatEbmTypePrintf ", %td, %" FloatEbmTypePrintf,
            pCutBest->m_uniqueTiebreaker,
            pCutBest->m_iVal,
            pCutBest->m_iValAspirationalFloat,
            pCutBest->m_cPredeterminedMovementOnCut,
            pCutBest->m_priority
         );
#endif // LOG_SUPERVERBOSE_DISCRETIZATION_ORDERED

         EBM_ASSERT(nullptr != pCutBest->m_pPrev);
         EBM_ASSERT(nullptr != pCutBest->m_pNext);
         EBM_ASSERT(!pCutBest->IsCut()); // this checks m_cPredeterminedMovementOnCut

         // we can't move past our outer boundaries
         EBM_ASSERT(-ptrdiff_t { k_cutExploreDistance } < pCutBest->m_cPredeterminedMovementOnCut &&
            pCutBest->m_cPredeterminedMovementOnCut < ptrdiff_t { k_cutExploreDistance });

         EBM_ASSERT(!std::isnan(pCutBest->m_iValAspirationalFloat));
         EBM_ASSERT(!std::isinf(pCutBest->m_iValAspirationalFloat));
         EBM_ASSERT(FloatEbmType { 0 } < pCutBest->m_iValAspirationalFloat);

         EBM_ASSERT(!std::isnan(pCutBest->m_priority));
         EBM_ASSERT(!std::isinf(pCutBest->m_priority));

         const size_t iVal = pCutBest->m_iVal; // preserve the location of our cut in case we end up moving
         if(k_valNotLegal == iVal) {
            // k_valNoCutsPossible means there are no legal cuts, and also that all the remaining items 
            // in the queue are also uncuttable, so exit.
            EBM_ASSERT(k_priorityNoCutsPossible == pCutBest->m_priority);
            break;
         }
         EBM_ASSERT(FloatEbmType { 0 } <= pCutBest->m_priority);

         // find our visibility window region
         CutPoint * pCutLowModificationExclusiveBoundary = pCutBest;
         size_t cRangesLowModification = k_cutExploreDistance;
         ptrdiff_t cPredeterminedMovementOnCutLowLow;
         do {
            pCutLowModificationExclusiveBoundary = pCutLowModificationExclusiveBoundary->m_pPrev;
            cPredeterminedMovementOnCutLowLow = pCutLowModificationExclusiveBoundary->m_cPredeterminedMovementOnCut;
            --cRangesLowModification;
         } while(LIKELY(LIKELY(k_movementDoneCut != cPredeterminedMovementOnCutLowLow) && 
            LIKELY(size_t { 0 } != cRangesLowModification)));

         cRangesLowModification = k_cutExploreDistance - cRangesLowModification;
         EBM_ASSERT(1 <= cRangesLowModification);
         EBM_ASSERT(cRangesLowModification <= k_cutExploreDistance);
         EBM_ASSERT(-pCutBest->m_cPredeterminedMovementOnCut < static_cast<ptrdiff_t>(cRangesLowModification));

         EBM_ASSERT(FloatEbmType { 0 } <= pCutLowModificationExclusiveBoundary->m_iValAspirationalFloat);

         // this should be exact, since we would have set it like this
         EBM_ASSERT(!pCutLowModificationExclusiveBoundary->IsCut() || pCutLowModificationExclusiveBoundary->m_iValAspirationalFloat == static_cast<FloatEbmType>(pCutLowModificationExclusiveBoundary->m_iVal));
         EBM_ASSERT(pCutLowModificationExclusiveBoundary->m_iVal <= pCutBest->m_iVal);
         EBM_ASSERT(pCutLowModificationExclusiveBoundary->m_iValAspirationalFloat < pCutBest->m_iValAspirationalFloat);

         CutPoint * pCutHighModificationExclusiveBoundary = pCutBest;
         size_t cRangesHighModification = k_cutExploreDistance;
         ptrdiff_t cPredeterminedMovementOnCutHighHigh;
         do {
            pCutHighModificationExclusiveBoundary = pCutHighModificationExclusiveBoundary->m_pNext;
            cPredeterminedMovementOnCutHighHigh = pCutHighModificationExclusiveBoundary->m_cPredeterminedMovementOnCut;
            --cRangesHighModification;
         } while(LIKELY(LIKELY(k_movementDoneCut != cPredeterminedMovementOnCutHighHigh) && 
            LIKELY(size_t { 0 } != cRangesHighModification)));

         cRangesHighModification = k_cutExploreDistance - cRangesHighModification;
         EBM_ASSERT(1 <= cRangesHighModification);
         EBM_ASSERT(cRangesHighModification <= k_cutExploreDistance);
         EBM_ASSERT(pCutBest->m_cPredeterminedMovementOnCut < static_cast<ptrdiff_t>(cRangesHighModification));

         EBM_ASSERT(FloatEbmType { 0 } < pCutHighModificationExclusiveBoundary->m_iValAspirationalFloat);

         // this should be exact, since we would have set it like this
         EBM_ASSERT(!pCutHighModificationExclusiveBoundary->IsCut() || pCutHighModificationExclusiveBoundary->m_iValAspirationalFloat == static_cast<FloatEbmType>(pCutHighModificationExclusiveBoundary->m_iVal));
         EBM_ASSERT(pCutBest->m_iVal <= pCutHighModificationExclusiveBoundary->m_iVal);
         EBM_ASSERT(pCutBest->m_iValAspirationalFloat < pCutHighModificationExclusiveBoundary->m_iValAspirationalFloat);

         // we're allowed to move cuts between our sides before cutting, so let's find our new home
         ptrdiff_t cPredeterminedMovementOnCut = pCutBest->m_cPredeterminedMovementOnCut;

         CutPoint * pCutCur = pCutBest;

         CutPoint * pCutLowLowVisibilityInclusiveBoundary = pCutLowModificationExclusiveBoundary;
         size_t cRangesLowLowPlan = cRangesLowModification;

         CutPoint * pCutHighHighVisibilityInclusiveBoundary = pCutHighModificationExclusiveBoundary;
         size_t cRangesHighHighPlan = cRangesHighModification;

         cRangesLowModification = 
            static_cast<size_t>(static_cast<ptrdiff_t>(cRangesLowModification) + cPredeterminedMovementOnCut);
         cRangesHighModification = 
            static_cast<size_t>(static_cast<ptrdiff_t>(cRangesHighModification) - cPredeterminedMovementOnCut);

         EBM_ASSERT(size_t { 1 } <= cRangesLowModification);
         EBM_ASSERT(size_t { 1 } <= cRangesHighModification);

         if(UNLIKELY(ptrdiff_t { 0 } != cPredeterminedMovementOnCut)) {

            // If we push cuts either left or right, we don't change the window bounds within which we modify
            // the aspirational cuts, because if we did, then there would be no bounds on where we can 
            // 100% guarantee that no changes will affect outside regions
            // we do however keep track of our visibility bounds since we'll use that when computing
            // the plan and the priority of our cuts since those can observe other aspirational cuts
            // outside of the bounds which we modify the aspirational cuts

            if(UNPREDICTABLE(cPredeterminedMovementOnCut < ptrdiff_t { 0 })) {
               do {
                  pCutCur = pCutCur->m_pPrev;
                  EBM_ASSERT(!pCutCur->IsCut());

                  if(k_movementDoneCut != cPredeterminedMovementOnCutLowLow) {
                     pCutLowLowVisibilityInclusiveBoundary = pCutLowLowVisibilityInclusiveBoundary->m_pPrev;
                     cPredeterminedMovementOnCutLowLow = pCutLowLowVisibilityInclusiveBoundary->m_cPredeterminedMovementOnCut;
                  } else {
                     // we've hit a cut boundary which we can't move, so we get closer to it
                     EBM_ASSERT(2 <= cRangesLowLowPlan);
                     --cRangesLowLowPlan;
                  }
                  EBM_ASSERT((k_movementDoneCut == cPredeterminedMovementOnCutLowLow) == pCutLowLowVisibilityInclusiveBoundary->IsCut());

                  if(k_cutExploreDistance == cRangesHighHighPlan) {
                     pCutHighHighVisibilityInclusiveBoundary = pCutHighHighVisibilityInclusiveBoundary->m_pPrev;
                     EBM_ASSERT(!pCutHighHighVisibilityInclusiveBoundary->IsCut());
                  } else {
                     EBM_ASSERT(pCutHighHighVisibilityInclusiveBoundary->IsCut());
                     ++cRangesHighHighPlan;
                  }

                  ++cPredeterminedMovementOnCut;
               } while(UNLIKELY(ptrdiff_t { 0 } != cPredeterminedMovementOnCut));
               cPredeterminedMovementOnCutHighHigh = pCutHighHighVisibilityInclusiveBoundary->m_cPredeterminedMovementOnCut;
            } else {
               do {
                  pCutCur = pCutCur->m_pNext;
                  EBM_ASSERT(!pCutCur->IsCut());

                  if(k_cutExploreDistance == cRangesLowLowPlan) {
                     pCutLowLowVisibilityInclusiveBoundary = pCutLowLowVisibilityInclusiveBoundary->m_pNext;
                     EBM_ASSERT(!pCutLowLowVisibilityInclusiveBoundary->IsCut());
                  } else {
                     EBM_ASSERT(pCutLowLowVisibilityInclusiveBoundary->IsCut());
                     ++cRangesLowLowPlan;
                  }

                  if(k_movementDoneCut != cPredeterminedMovementOnCutHighHigh) {
                     pCutHighHighVisibilityInclusiveBoundary = pCutHighHighVisibilityInclusiveBoundary->m_pNext;
                     cPredeterminedMovementOnCutHighHigh = pCutHighHighVisibilityInclusiveBoundary->m_cPredeterminedMovementOnCut;
                  } else {
                     // we've hit a cut boundary which we can't move, so we get closer to it
                     EBM_ASSERT(2 <= cRangesHighHighPlan);
                     --cRangesHighHighPlan;
                  }
                  EBM_ASSERT((k_movementDoneCut == cPredeterminedMovementOnCutHighHigh) == pCutHighHighVisibilityInclusiveBoundary->IsCut());

                  --cPredeterminedMovementOnCut;
               } while(UNLIKELY(ptrdiff_t { 0 } != cPredeterminedMovementOnCut));
               cPredeterminedMovementOnCutLowLow = pCutLowLowVisibilityInclusiveBoundary->m_cPredeterminedMovementOnCut;
            }
         }

         EBM_ASSERT(size_t { 1 } <= cRangesLowLowPlan);
         EBM_ASSERT(size_t { 1 } <= cRangesHighHighPlan);

         EBM_ASSERT(!pCutCur->IsCut());

         pCutCur->SetCut();

         const FloatEbmType iValFloat = static_cast<FloatEbmType>(iVal);

         pCutCur->m_iValAspirationalFloat = iValFloat;
         pCutCur->m_iVal = iVal;

         // TODO: We've just finished materializing the cut based on the plan we developed earlier.  We'll
         // now go and re-do our aspirational cut plan for all the aspirational cuts within our visibility windows
         // on each side.  Before we do that though, we can do a quick check to find out what the maximum number of 
         // cuts we could place is between our new materialized cut and our visibility windows.  If it's not possible
         // even in theory to place 20 cuts on our low side, then our aspirational plans shouldn't even consider that
         // Also, if we delete/move aspirational cuts early, there's a higher chance that we'll be able to re-use them
         // in a good place.  See the DetermineRangesMax(...) function on how to do this.
         // There's actually two subtle issues here that we need to handle differently:
         //  1) we need to determine if we should delete any cuts on our left or right.  To do this go from our
         //     materialized cut and jump by cSamplesPerBinMin using aNeighbourJumps until we hit the aspirational
         //     or materialized window.  If the window is aspirational we can either use the edge that's within the
         //     aspirational window or the one right outside if we want more certainty, but in either case it's not a
         //     100% guarantee since we might not even cut on the range that our aspirational cut falls on.  I lean towards
         //     using our inner window and pruning early so that we get the cuts to useful place early.  The alternate
         //     viewpoint is to only trim when we hit a materialized boundary before our visibility window.
         //     since cuts could eventually be pushed into the open ended range potentially, but in general we should
         //     think that if cuts can't be used within our range for a long distance we'd want to reallocate them
         //     even if they could be pushed since it changes the density within our visibility window and
         //     if we pushed them to a point outside then we'be be increasing the density there, so better to change
         //     the densities in a more controlled way beforehand
         //  2) We want to know how many potential cuts there are on each side of each aspirational cut that we're
         //     we're considering.  Since we're processing like 20-50 of these, we can slide the value window
         //     with cSamplesPerBinMin as we slide the visiblility windows to the left or right
         //
         // We only need to know if we have more than k_cutExploreDistance items, since we can't have more than
         // that number of cuts until our border.  We can allocate a fixed size array on the stack with 
         // k_cutExploreDistance size_t indexes and fill these from the lower and higher sides, and then if we
         // cross any of those indexes we know if we've gone below a limit.


         FloatEbmType stepPoint = pCutLowModificationExclusiveBoundary->m_iValAspirationalFloat;
         FloatEbmType stepLength;
         if(pCutLowModificationExclusiveBoundary->IsCut()) {
            EBM_ASSERT(pCutLowModificationExclusiveBoundary->m_iVal < iVal);
            stepLength = static_cast<FloatEbmType>(iVal - pCutLowModificationExclusiveBoundary->m_iVal);
         } else {
            EBM_ASSERT(stepPoint < iValFloat);
            stepLength = iValFloat - stepPoint;
         }
         stepLength /= static_cast<FloatEbmType>(cRangesLowModification);

         CutPoint * pCutAspirational = pCutCur;
         while(LIKELY(size_t { 0 } != --cRangesLowModification)) {
            pCutAspirational = pCutAspirational->m_pPrev;
            const FloatEbmType iValAspirationalFloat = stepPoint + 
               stepLength * static_cast<FloatEbmType>(cRangesLowModification);
            pCutAspirational->m_iValAspirationalFloat = iValAspirationalFloat;
         }

         stepPoint = iValFloat;
         if(pCutHighModificationExclusiveBoundary->IsCut()) {
            EBM_ASSERT(iVal < pCutHighModificationExclusiveBoundary->m_iVal);
            stepLength = static_cast<FloatEbmType>(pCutHighModificationExclusiveBoundary->m_iVal - iVal);
         } else {
            EBM_ASSERT(iValFloat < pCutHighModificationExclusiveBoundary->m_iValAspirationalFloat);
            stepLength = pCutHighModificationExclusiveBoundary->m_iValAspirationalFloat - stepPoint;
         }
         stepLength /= static_cast<FloatEbmType>(cRangesHighModification);

         pCutAspirational = pCutHighModificationExclusiveBoundary;
         while(size_t { 0 } != --cRangesHighModification) {
            pCutAspirational = pCutAspirational->m_pPrev;
            const FloatEbmType iValAspirationalFloat = stepPoint + 
               stepLength * static_cast<FloatEbmType>(cRangesHighModification);
            pCutAspirational->m_iValAspirationalFloat = iValAspirationalFloat;
         }

         CutPoint * pCutLowLowPlanInclusiveBoundary = pCutLowLowVisibilityInclusiveBoundary;
         CutPoint * pCutLowHighPlanInclusiveBoundary = pCutCur;
         size_t cRangesLowHighPlan = size_t { 0 };

         size_t iValLowLowPlan = LIKELY(k_movementDoneCut == cPredeterminedMovementOnCutLowLow) ? 
            pCutLowLowPlanInclusiveBoundary->m_iVal : k_valNotLegal;
         size_t iValLowHighPlan = iVal;

         CutPoint * pCutLowPlanCur = pCutCur;

         while(true) {
            if(UNLIKELY(k_valNotLegal == iValLowLowPlan)) {
               EBM_ASSERT(!pCutLowLowPlanInclusiveBoundary->IsCut());
               pCutLowLowPlanInclusiveBoundary = pCutLowLowPlanInclusiveBoundary->m_pPrev;
               if(UNLIKELY(pCutLowLowPlanInclusiveBoundary->IsCut())) {
                  iValLowLowPlan = pCutLowLowPlanInclusiveBoundary->m_iVal;
               }
            } else {
               EBM_ASSERT(pCutLowLowPlanInclusiveBoundary->IsCut());
               --cRangesLowLowPlan;
               if(UNLIKELY(0 == cRangesLowLowPlan)) {
                  // we've reached the hard boundary of a materialized cut
                  break;
               }
            }

            if(UNLIKELY(k_cutExploreDistance == cRangesLowHighPlan)) {
               pCutLowHighPlanInclusiveBoundary = pCutLowHighPlanInclusiveBoundary->m_pPrev;
               EBM_ASSERT(!pCutLowHighPlanInclusiveBoundary->IsCut());
               iValLowHighPlan = k_valNotLegal;
               if(UNLIKELY(pCutLowHighPlanInclusiveBoundary == pCutLowModificationExclusiveBoundary)) {
                  // we've reached the boundary of where we changed the aspirational cuts, so no changes should
                  // occur beyond this point
                  break;
               }
            } else {
               EBM_ASSERT(pCutLowHighPlanInclusiveBoundary->IsCut());
               EBM_ASSERT(k_valNotLegal != iValLowHighPlan);
               ++cRangesLowHighPlan;
            }

            pCutLowPlanCur = pCutLowPlanCur->m_pPrev;
            EBM_ASSERT(!pCutLowPlanCur->IsCut()); // we should have exited on 0 == cRangesLowLowPlan beforehand

            BuildNeighbourhoodPlan(
               cSamples,
               bSymmetryReversal,

               cSamplesPerBinMin,
               iValuesStart,
               cCuttableItems,
               aNeighbourJumps,

               cRangesLowLowPlan,
               iValLowLowPlan,
               pCutLowLowPlanInclusiveBoundary->m_iValAspirationalFloat,

               cRangesLowHighPlan,
               iValLowHighPlan,
               pCutLowHighPlanInclusiveBoundary->m_iValAspirationalFloat,

               pCutLowPlanCur
            );
         }

         CutPoint * pCutHighHighPlanInclusiveBoundary = pCutHighHighVisibilityInclusiveBoundary;
         CutPoint * pCutHighLowPlanInclusiveBoundary = pCutCur;
         size_t cRangesHighLowPlan = size_t { 0 };

         size_t iValHighHighPlan = LIKELY(k_movementDoneCut == cPredeterminedMovementOnCutHighHigh) ? 
            pCutHighHighPlanInclusiveBoundary->m_iVal : k_valNotLegal;
         size_t iValHighLowPlan = iVal;

         CutPoint * pCutHighPlanCur = pCutCur;

         while(true) {
            if(UNLIKELY(k_valNotLegal == iValHighHighPlan)) {
               EBM_ASSERT(!pCutHighHighPlanInclusiveBoundary->IsCut());
               pCutHighHighPlanInclusiveBoundary = pCutHighHighPlanInclusiveBoundary->m_pNext;
               if(UNLIKELY(pCutHighHighPlanInclusiveBoundary->IsCut())) {
                  iValHighHighPlan = pCutHighHighPlanInclusiveBoundary->m_iVal;
               }
            } else {
               EBM_ASSERT(pCutHighHighPlanInclusiveBoundary->IsCut());
               --cRangesHighHighPlan;
               if(UNLIKELY(0 == cRangesHighHighPlan)) {
                  // we've reached the hard boundary of a materialized cut
                  break;
               }
            }

            if(UNLIKELY(k_cutExploreDistance == cRangesHighLowPlan)) {
               pCutHighLowPlanInclusiveBoundary = pCutHighLowPlanInclusiveBoundary->m_pNext;
               EBM_ASSERT(!pCutHighLowPlanInclusiveBoundary->IsCut());
               iValHighLowPlan = k_valNotLegal;
               if(UNLIKELY(pCutHighLowPlanInclusiveBoundary == pCutHighModificationExclusiveBoundary)) {
                  // we've reached the boundary of where we changed the aspirational cuts, so no changes should
                  // occur beyond this point
                  break;
               }
            } else {
               EBM_ASSERT(pCutHighLowPlanInclusiveBoundary->IsCut());
               EBM_ASSERT(k_valNotLegal != iValHighLowPlan);
               ++cRangesHighLowPlan;
            }

            pCutHighPlanCur = pCutHighPlanCur->m_pNext;
            EBM_ASSERT(!pCutHighPlanCur->IsCut()); // we should have exited on 0 == cRangesHighHighPlan beforehand

            BuildNeighbourhoodPlan(
               cSamples,
               bSymmetryReversal,

               cSamplesPerBinMin,
               iValuesStart,
               cCuttableItems,
               aNeighbourJumps,

               cRangesHighLowPlan,
               iValHighLowPlan,
               pCutHighLowPlanInclusiveBoundary->m_iValAspirationalFloat,

               cRangesHighHighPlan,
               iValHighHighPlan,
               pCutHighHighPlanInclusiveBoundary->m_iValAspirationalFloat,

               pCutHighPlanCur
            );
         }

         // TODO: For each cut point we've examined our neighbourhood and selected a right/left decison that we can live with
         // for that cut point independent of all the other ones.  We can then maybe do an analysis to see if our
         // ideas for the neighbours match up with theirs and do some jiggering if the outcome within a window is bad
         // this allows us to see a bigger area, so we have to be careful that changes don't cascade beyond our visibility
         // window.  Perhaps we allow changes to ASPIRATIONAL cuts within our hard change boundary, but don't
         // change things outside of this window.

         EBM_ASSERT(pBestBinCuts->end() != pBestBinCuts->find(pCutCur));
         pBestBinCuts->erase(pCutCur);

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
         //     create a problem for us easily (so we should include metrics on the goodness of cuts)
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
         //
         //
         // TODO: CONSIDER (very tentatively) incorporating how bad it would be if we were forced to choose the worse side to cut on
         // if that's a bad scenario, we should probably try increasing our priority for our aspirational cut point
         // since we want that one to be materialized first. 
         // There are a lot of metrics we might use.  Three ideas:
         //   1) Look at how bad the best solution for any particular cut is.. if it's bad it's probably because the
         //      alternatives were worse
         //   2) Look at how bad the worst solution for any particular cut is.. we don't want to be forced to take the
         //      worst
         //   3) * take the aspirational cut, take the best matrialized cut, calculate what percentage we need to
         //      stretch from either boundary (the low boundary and the high boundary).  Take the one that has the highest
         //      percentage stretch
         //
         // I like #3 (it's the one we have implemented now), because after we choose each cut everything 
         // within the visibility windows gets re-shuffed.  We might not even fall on some of the problematic 
         // ranges anymore.  Choosing the cuts with the highest "tension" causes
         // us to decide the longest ranges that are the closest to one of our existing imovable boundaries thus
         // we're nailing down the ones that'll cause the most movement first while we have the most room, and it also
         // captures the idea that these are bad ones that need to be selected.  It'll tend to try deciding cuts
         // near our existing edge boundaries first instead of the ones in the center.  This is good since the ones at
         // the boundaries are more critical.  As we materialize cuts we'll get closer to the center and those will start
         // to want attention

         CutPoint * pCutLowLowPriorityInclusiveBoundary = pCutLowLowVisibilityInclusiveBoundary;
         CutPoint * pCutLowHighPriorityInclusiveBoundary = pCutCur;
         size_t cRangesLowHighPriority = 0;
         CutPoint * pCutLowPriorityCur = pCutCur;

         while(true) {
            pCutLowPriorityCur = pCutLowPriorityCur->m_pPrev;
            if(PREDICTABLE(k_movementDoneCut != cPredeterminedMovementOnCutLowLow)) {
               EBM_ASSERT(!pCutLowLowPriorityInclusiveBoundary->IsCut());
               pCutLowLowPriorityInclusiveBoundary = pCutLowLowPriorityInclusiveBoundary->m_pPrev;
               cPredeterminedMovementOnCutLowLow = pCutLowLowPriorityInclusiveBoundary->m_cPredeterminedMovementOnCut;
            } else {
               EBM_ASSERT(pCutLowLowPriorityInclusiveBoundary->IsCut());
               if(UNLIKELY(pCutLowPriorityCur == pCutLowLowPriorityInclusiveBoundary)) {
                  EBM_ASSERT(pCutLowPriorityCur->IsCut());
                  break;
               }
            }
            EBM_ASSERT(!pCutLowPriorityCur->IsCut());

            if(PREDICTABLE(k_cutExploreDistance == cRangesLowHighPriority)) {
               pCutLowHighPriorityInclusiveBoundary = pCutLowHighPriorityInclusiveBoundary->m_pPrev;
               EBM_ASSERT(!pCutLowHighPriorityInclusiveBoundary->IsCut());
               if(UNLIKELY(pCutLowHighPriorityInclusiveBoundary == pCutLowModificationExclusiveBoundary)) {

#ifndef NDEBUG

                  FloatEbmType debugPriority = CalculatePriority(
                     pCutLowLowPriorityInclusiveBoundary->m_iValAspirationalFloat,
                     pCutLowHighPriorityInclusiveBoundary->m_iValAspirationalFloat,
                     pCutLowPriorityCur
                  );

                  // these should be calculated via the same pathway, so should be identical
                  EBM_ASSERT(debugPriority == pCutLowPriorityCur->m_priority);

#endif // NDEBUG

                  break;
               }
            } else {
               EBM_ASSERT(pCutLowHighPriorityInclusiveBoundary->IsCut());
               ++cRangesLowHighPriority;
            }

            EBM_ASSERT(pBestBinCuts->end() != pBestBinCuts->find(pCutLowPriorityCur));
            pBestBinCuts->erase(pCutLowPriorityCur);

            pCutLowPriorityCur->m_priority = CalculatePriority(
               pCutLowLowPriorityInclusiveBoundary->m_iValAspirationalFloat,
               pCutLowHighPriorityInclusiveBoundary->m_iValAspirationalFloat,
               pCutLowPriorityCur
            );

            pBestBinCuts->insert(pCutLowPriorityCur);
         }

         CutPoint * pCutHighHighPriorityInclusiveBoundary = pCutHighHighVisibilityInclusiveBoundary;
         CutPoint * pCutHighLowPriorityInclusiveBoundary = pCutCur;
         size_t cRangesHighLowPriority = 0;
         CutPoint * pCutHighPriorityCur = pCutCur;

         while(true) {
            pCutHighPriorityCur = pCutHighPriorityCur->m_pNext;
            if(PREDICTABLE(k_movementDoneCut != cPredeterminedMovementOnCutHighHigh)) {
               EBM_ASSERT(!pCutHighHighPriorityInclusiveBoundary->IsCut());
               pCutHighHighPriorityInclusiveBoundary = pCutHighHighPriorityInclusiveBoundary->m_pNext;
               cPredeterminedMovementOnCutHighHigh = pCutHighHighPriorityInclusiveBoundary->m_cPredeterminedMovementOnCut;
            } else {
               EBM_ASSERT(pCutHighHighPriorityInclusiveBoundary->IsCut());
               if(UNLIKELY(pCutHighPriorityCur == pCutHighHighPriorityInclusiveBoundary)) {
                  EBM_ASSERT(pCutHighPriorityCur->IsCut());
                  break;
               }
            }
            EBM_ASSERT(!pCutHighPriorityCur->IsCut());

            if(PREDICTABLE(k_cutExploreDistance == cRangesHighLowPriority)) {
               pCutHighLowPriorityInclusiveBoundary = pCutHighLowPriorityInclusiveBoundary->m_pNext;
               EBM_ASSERT(!pCutHighLowPriorityInclusiveBoundary->IsCut());
               if(UNLIKELY(pCutHighLowPriorityInclusiveBoundary == pCutHighModificationExclusiveBoundary)) {

#ifndef NDEBUG

                  FloatEbmType debugPriority = CalculatePriority(
                     pCutHighLowPriorityInclusiveBoundary->m_iValAspirationalFloat,
                     pCutHighHighPriorityInclusiveBoundary->m_iValAspirationalFloat,
                     pCutHighPriorityCur
                  );

                  // these should be calculated via the same pathway, so should be identical
                  EBM_ASSERT(debugPriority == pCutHighPriorityCur->m_priority);

#endif // NDEBUG

                  break;
               }
            } else {
               EBM_ASSERT(pCutHighLowPriorityInclusiveBoundary->IsCut());
               ++cRangesHighLowPriority;
            }

            EBM_ASSERT(pBestBinCuts->end() != pBestBinCuts->find(pCutHighPriorityCur));
            pBestBinCuts->erase(pCutHighPriorityCur);

            pCutHighPriorityCur->m_priority = CalculatePriority(
               pCutHighLowPriorityInclusiveBoundary->m_iValAspirationalFloat,
               pCutHighHighPriorityInclusiveBoundary->m_iValAspirationalFloat,
               pCutHighPriorityCur
            );

            pBestBinCuts->insert(pCutHighPriorityCur);
         }
      }
   } catch(...) {
      LOG_0(TraceLevelWarning, "WARNING CutSegment exception");
      return true;
   }

   IronCuts();

   return false;
}

static bool TreeSearchCutSegment(
   std::set<CutPoint *, CompareCutPoint> * pBestBinCuts,

   const size_t cSamples,
   const bool bSymmetryReversal,
   const size_t cSamplesPerBinMin,

   const size_t iValuesStart,
   const size_t cCuttableItems,
   const NeighbourJump * const aNeighbourJumps,

   const size_t cRanges,
   // for efficiency we include space for the end point cuts even if they don't exist
   CutPoint * const aCutsWithENDPOINTS
) noexcept {
   try {
      EBM_ASSERT(nullptr != pBestBinCuts);
      EBM_ASSERT(pBestBinCuts->empty());

      EBM_ASSERT(2 <= cSamples); // we need at least 2 to split, otherwise we'd have exited before calling here
      EBM_ASSERT(1 <= cSamplesPerBinMin);

      EBM_ASSERT(nullptr != aNeighbourJumps);

      EBM_ASSERT(2 <= cRanges);
      EBM_ASSERT(cSamplesPerBinMin <= cCuttableItems / cRanges);
      EBM_ASSERT(nullptr != aCutsWithENDPOINTS);

      // - TODO: EXPLORING BOTH SIDES
      //   - this function calls CutSegment, which greedily materializes cuts, so when it's unsure about a cut
      //     it needs to be conservative and pick the least likley cut to cause problems down the road
      //   - at this higher level, we can try cutting both low AND high AND skip the cut.  We use CutCuttingRange to
      //     do the full exploration of both options and then we pick the better one.
      //   - we can also explore N steps in the future to pick the best first step, then delete the worst 1st step
      //     and keep all the work we did along the choice that we made (the remaining 128 options) then we can pick
      //     the best step from all those 128 options and continue this way.  Since we do a complete recalculation
      //     of all the BinCuts we can only do this several times, but it allows us to have 2 levels of fallback

      //   - we can design an algorithm that divides into 255 and chooses the worst one and then does a complete fit on either direction.Best fit is recorded
      //     then we re-do all 254 other cuts on BOTH sides.  We can only do a set number of these, so after 8 levels we'd have 256 attempts.  That might be acceptable
      //   - the algorithm that we have below plays it safe since it needs to live with it's decions.  This more spectlative algorithm above can be more
      //     risky since it plays both directions a bad play won't undermine it.  As such, we should try and chose the worst decion without regard to position
      //     so in other words, try to choose the range that we have a drop point in in the middle where we need to move the most to get away from the 
      //     best drops.  We can also try going left, going right, OR not choosing.  Don't traverse down the NO choice path, so we add 50% load, but we don't grow at 3^N, and we'll 
      //     also explore the no choice at the root option
      //

      //constexpr size_t k_CutExploreDepth = 8;
      //constexpr size_t k_CutExplorations = size_t { 1 } << k_CutExploreDepth;

      CutPoint * pCutCur = &aCutsWithENDPOINTS[0];
      CutPoint * pCutNext = &aCutsWithENDPOINTS[1];

      pCutCur->m_pNext = pCutNext;
      pCutCur->SetCut();
      pCutCur->m_iValAspirationalFloat = FloatEbmType { 0 };
      pCutCur->m_iVal = size_t { 0 };

      const FloatEbmType stepInit = static_cast<FloatEbmType>(cCuttableItems) / static_cast<FloatEbmType>(cRanges);
      EBM_ASSERT(cSamplesPerBinMin <= 1.00001 * stepInit);

      const FloatEbmType cCuttableItemsFloat = static_cast<FloatEbmType>(cCuttableItems);
      size_t iCutCur = 1;
      size_t iValLow = size_t { 0 };
      FloatEbmType iValAspirationalLowFloat = FloatEbmType { 0 };
      size_t cRangesHigh = k_cutExploreDistance;
      size_t iValHigh = k_valNotLegal;
      do {
         pCutNext->m_pPrev = pCutCur;
         pCutCur = pCutNext;
         ++pCutNext;
         pCutCur->m_pNext = pCutNext;

         size_t cRangesLow;
         const ptrdiff_t iRangeLow = 
            static_cast<ptrdiff_t>(iCutCur) - static_cast<ptrdiff_t>(k_cutExploreDistance);
         if(UNLIKELY(iRangeLow <= ptrdiff_t { 0 })) {
            cRangesLow = iCutCur;
            EBM_ASSERT(size_t { 0 } == iValLow);
            EBM_ASSERT(FloatEbmType { 0 } == iValAspirationalLowFloat);
         } else {
            cRangesLow = k_cutExploreDistance;
            iValLow = k_valNotLegal;
            iValAspirationalLowFloat = stepInit * static_cast<FloatEbmType>(static_cast<size_t>(iRangeLow));
         }

         FloatEbmType iValAspirationalHighFloat;
         size_t iRangeHigh = iCutCur + k_cutExploreDistance;
         if(UNLIKELY(cRanges <= iRangeHigh)) {
            cRangesHigh = cRanges - iCutCur;
            iValHigh = cCuttableItems;
            iValAspirationalHighFloat = cCuttableItemsFloat;
         } else {
            EBM_ASSERT(k_cutExploreDistance == cRangesHigh);
            EBM_ASSERT(k_valNotLegal == iValHigh);
            iValAspirationalHighFloat = stepInit * static_cast<FloatEbmType>(iRangeHigh);
         }

         const FloatEbmType iValAspirationalCurFloat = stepInit * static_cast<FloatEbmType>(iCutCur);
         pCutCur->m_iValAspirationalFloat = iValAspirationalCurFloat;

         EBM_ASSERT(pCutCur->m_uniqueTiebreaker < cRanges);

         BuildNeighbourhoodPlan(
            cSamples,
            bSymmetryReversal,
            cSamplesPerBinMin,
            iValuesStart,
            cCuttableItems,
            aNeighbourJumps,
            cRangesLow,
            iValLow,
            iValAspirationalLowFloat,
            cRangesHigh,
            iValHigh,
            iValAspirationalHighFloat,
            pCutCur
         );
         ++iCutCur;
      } while(iCutCur < cRanges);

      pCutNext->m_pPrev = pCutCur;
      pCutNext->m_pNext = nullptr;
      pCutNext->SetCut();
      pCutNext->m_iValAspirationalFloat = cCuttableItemsFloat;
      pCutNext->m_iVal = cCuttableItems;


      // now calculate priorities
      CutPoint * pCutLow = &aCutsWithENDPOINTS[0];
      CutPoint * pCutCenter = &aCutsWithENDPOINTS[1];
      const size_t iRangeHigh = cRanges <= size_t { 1 } + k_cutExploreDistance ? 
         cRanges : size_t { 1 } + k_cutExploreDistance;
      CutPoint * pCutHigh = &aCutsWithENDPOINTS[iRangeHigh];

#ifndef NDEBUG

      EBM_ASSERT(aCutsWithENDPOINTS[0].m_pNext == pCutCenter); // this will fail if we remove items above in the future
      CutPoint * pCutDebug = pCutCenter;
      for(size_t cDebugRemaining = k_cutExploreDistance; nullptr != pCutDebug->m_pNext && 0 < cDebugRemaining ; 
         --cDebugRemaining) 
      {
         pCutDebug = pCutDebug->m_pNext;
      }
      // this will fail if we remove items above in the future
      EBM_ASSERT(pCutDebug == pCutHigh);

#endif // NDEBUG

      size_t cLowRanges = 1;
      do {
         // in the future we might write code above that removes BinCuts, which if it were true could mean no legal cuts
         EBM_ASSERT(nullptr != pCutCenter->m_pNext);
         EBM_ASSERT(pCutLow < pCutCenter);
         EBM_ASSERT(pCutCenter < pCutHigh);

         pCutCenter->m_priority = CalculatePriority(
            pCutLow->m_iValAspirationalFloat,
            pCutHigh->m_iValAspirationalFloat,
            pCutCenter
         );

         EBM_ASSERT(!pCutCenter->IsCut());
         pBestBinCuts->insert(pCutCenter);

         if(UNLIKELY(k_cutExploreDistance != cLowRanges)) {
            ++cLowRanges;
         } else {
            pCutLow = pCutLow->m_pNext;
         }

         if(UNLIKELY(pCutNext != pCutHigh)) {
            pCutHigh = pCutHigh->m_pNext;
         }

         pCutCenter = pCutCenter->m_pNext;
      } while(pCutNext != pCutCenter);
   } catch(...) {
      LOG_0(TraceLevelWarning, "WARNING TreeSearchCutSegment exception");
      return true;
   }

   return CutCuttingRange(
      pBestBinCuts,
      cSamples,
      bSymmetryReversal,
      cSamplesPerBinMin,
      iValuesStart,
      cCuttableItems,
      aNeighbourJumps
   );
}

INLINE_RELEASE_UNTEMPLATED static bool TradeCutSegment(
   std::set<CutPoint *, CompareCutPoint> * const pBestBinCuts,

   const size_t cSamples,
   const bool bSymmetryReversal,
   const size_t cSamplesPerBinMin,

   const size_t iValuesStart,
   const size_t cCuttableItems,
   const NeighbourJump * const aNeighbourJumps,

   const size_t cRanges,
   // for efficiency we include space for the end point cuts even if they don't exist
   CutPoint * const aCutsWithENDPOINTS
) noexcept {
   // - TODO:
   //   - we can examine what it would look like to have 1 more cut and 1 less cut that our original choice
   //     then we can try and sort the best to worst subtraction and addition, and then try and swap the best subtraction with the best addition and repeat
   //   - calculate the maximum number of cuts based on the minimum bunch size.  we should be able to do this by
   //     doing a single pass where we make every range the minimum
   //   - then we can loop from our current cuts to the maximum and stop when we hit the maximum (perahps there are long 
   //     ranges that prevent good spits)
   //   - if we want to get a specific number of cuts, we can ask for that many, but we might get less back as a result
   //     (never more).  We can try to increment the number of items that we ask for and see if we end up with the right
   //     number.  It might be bad though if we continually do +1 because it might be intractable if there are a lot
   //     of cuts.  Perahps we want to use a binary algorithm where we do +1, +2, +4, +8, and if we exceed then
   //     do binary descent between 4 and 8 until we get our exact number.

   return TreeSearchCutSegment(
      pBestBinCuts, 
      cSamples,
      bSymmetryReversal,
      cSamplesPerBinMin,
      iValuesStart, 
      cCuttableItems, 
      aNeighbourJumps,
      cRanges, 
      aCutsWithENDPOINTS
   );
}

INLINE_RELEASE_UNTEMPLATED static size_t DetermineRangesMax(
   const size_t cSamplesInSubset,
   const FloatEbmType * const pValues,
   const size_t cSamplesPerBinMin
) noexcept {
   EBM_ASSERT(1 <= cSamplesInSubset);
   EBM_ASSERT(nullptr != pValues);
   EBM_ASSERT(1 <= cSamplesPerBinMin);
   EBM_ASSERT(cSamplesPerBinMin <= cSamplesInSubset);

   if(size_t { 1 } == cSamplesInSubset) {
      EBM_ASSERT(size_t { 1 } == cSamplesPerBinMin);
      return size_t { 1 };
   }

   FloatEbmType valPrev = pValues[0];
   const FloatEbmType * pValueStartRange = pValues;
   const FloatEbmType * pValue = pValues + 1;
   const FloatEbmType * const pValueEnd = pValues + cSamplesInSubset;

   size_t cRanges = 0;
   size_t cItems;
   do {
      EBM_ASSERT(pValue < pValueEnd);
      FloatEbmType valCur = *pValue;
      if(valCur != valPrev) {
         cItems = pValue - pValueStartRange;
         if(cSamplesPerBinMin <= cItems) {
            ++cRanges;
            pValueStartRange = pValue;
         }
         valPrev = valCur;
      }
      ++pValue;
   } while(pValueEnd != pValue);
   cItems = pValue - pValueStartRange;
   if(cSamplesPerBinMin <= cItems) {
      ++cRanges;
   }

#ifndef NDEBUG

   // try it in reverse using different code
   size_t iDebugCur = cSamplesInSubset - 1;
   size_t iDebugStartEqual = iDebugCur;
   size_t iDebugStartRange = iDebugCur;
   size_t cDebugRanges = 0;
   while(0 != iDebugCur) {
      --iDebugCur;
      if(pValues[iDebugCur] != pValues[iDebugStartEqual]) {
         if(cSamplesPerBinMin <= iDebugStartRange - iDebugCur) {
            ++cDebugRanges;
            iDebugStartRange = iDebugCur;
         }
         iDebugStartEqual = iDebugCur;
      }
   }
   if(cSamplesPerBinMin <= iDebugStartRange + 1) {
      ++cDebugRanges;
   }
   EBM_ASSERT(cDebugRanges == cRanges);

#endif

   return cRanges;
}

static bool AddCutToRanges(
   std::set<CuttingRange *, CompareCuttingRange> & queue
) {
   EBM_ASSERT(!queue.empty());

   auto iterator = queue.begin();
   CuttingRange * const pCuttingRangeAdd = *iterator;
   if(k_illegalAvgCuttableRangeWidthAfterAddingOneCut == pCuttingRangeAdd->m_avgCuttableRangeWidthAfterAddingOneCut) {
      // nothing remaining in the queue can accept new cuts
      return true;
   }
   queue.erase(iterator);

   // this is how many ranges we were assigned before deciding that this range would recieve a new cut
   const size_t cRangesPrev = pCuttingRangeAdd->m_cRangesAssigned;
   // now that this range has been decided as the receiver of a new cut, it now has this many ranges
   const size_t cRangesCur = cRangesPrev + size_t { 1 };
   pCuttingRangeAdd->m_cRangesAssigned = cRangesCur;

   FloatEbmType avgRangeWidthAfterAddingOneCut = k_illegalAvgCuttableRangeWidthAfterAddingOneCut;
   EBM_ASSERT(cRangesCur <= pCuttingRangeAdd->m_cRangesMax);
   if(LIKELY(pCuttingRangeAdd->m_cRangesMax != cRangesCur)) {
      // we need to re-add our CuttingRange back into the priority queue.  If we were assigned a new range
      // again, this is how many we'd be at
      const size_t cRangesNext = cRangesCur + size_t { 1 };
      const size_t cCuttableItems = pCuttingRangeAdd->m_cCuttableValues;

      avgRangeWidthAfterAddingOneCut =
         static_cast<FloatEbmType>(cCuttableItems) / static_cast<FloatEbmType>(cRangesNext);

      // don't muliply by GetTweakingMultiple, since avgRangeWidthAfterAddingOneCut is derrived from
      // size_t values, it should have exactly the same value when cCuttableItems and cRangesNext
      // are the same, so we should then get to compare on m_uniqueTiebreaker after seeing the exact
      // floating point equality.  Also, unlike the CutPoint priority value, we don't want to affect
      // m_avgCuttableRangeWidthAfterAddingOneCut since even distant regions shouldn't have divergent
      // priorities, unlike for BinCuts
   }
   pCuttingRangeAdd->m_avgCuttableRangeWidthAfterAddingOneCut = avgRangeWidthAfterAddingOneCut;
   queue.insert(pCuttingRangeAdd);
   return false;
}

static void StuffCutsIntoCuttingRanges(
   std::set<CuttingRange *, CompareCuttingRange> & queue,
   const size_t cCuttingRanges,
   CuttingRange * const aCuttingRange,
   const size_t cSamplesPerBinMin,
   const size_t cCutsAssignable
) {
   EBM_ASSERT(1 <= cCuttingRanges);
   EBM_ASSERT(nullptr != aCuttingRange);
   EBM_ASSERT(1 <= cSamplesPerBinMin);
   EBM_ASSERT(1 <= cCutsAssignable);
   // we add 2 here, which should be legal since we allocate sentinel nodes when allocating cutting ranges
   EBM_ASSERT(cCuttingRanges <= cCutsAssignable + 2);

   // generally, having small bins with insufficient data is more dangerous for overfitting
   // than the lost opportunity from not cutting big bins down.  So, what we want to avoid is having
   // small bins.  So, create a heap and insert the average bin size AFTER we would add a new cut
   // don't insert any CuttingRanges that cannot legally be cut (so, it's guaranteed to only have
   // cuttable items).  Now pick off the CuttingRange that has the largest average AFTER adding a cut
   // and then add the cut, re-calculate the new average, and re-insert into the heap.  Continue
   // until there are no items in the heap because they've all exhausted the possibilities of cuts
   // OR until we run out of cuts to dole out.

   CuttingRange * pCuttingRangeInit = aCuttingRange;
   const CuttingRange * const pCuttingRangeEnd = aCuttingRange + cCuttingRanges;
   do {
      // when dividing the values into uncutable and cutable ranges, we chose the minimum number of equal values
      // that needed to exist before a range was considered uncutable (inside GetUncuttableRangeLengthMin)
      // That value was chosen very carefully such that we could guarantee that every cuttable range that was
      // surrounded by uncuttable ranges on both sides would get a cut.  The only cuttable ranges that are not
      // guaranteed to get an explicit cut are the first and last ranges, and only if they don't have uncuttable
      // ranges between the cutable range and the first or last values.  Of course, we would want any cutable
      // range surrounded on both sides by uncutable ranges to have a cut, if only just to separate the uncuttable
      // ranges.  The interesting thing about the first and last cuttable ranges is that they can form a a range 
      // with just one cut since the end of the array provides an implicit cut.
      // 
      // So, our initial starting point is that every cuttable range will achieve a range if we give it one
      // more cut than what it has currenlty.  If we add 1 cut, we'll get 1 range in each of the CuttingRanges.
      // Another way to say this is that each CuttingRange currently has 0 ranges, so our initial state is
      // to set m_cRangesAssigned to zero, and insert each CuttingRange into the priority queue with the priority
      // based on the forward looking future that would occur if we added one cut, which would lead to 1 full range.
      //
      // One oddity is the case where we have one cuttable range and no uncuttable ranges.  In that case we
      // essentially have two implicit cuts, and therefore a full range even without an explicit cut assigned.
      // The zero cut scenario isn't possible, since we filter those cases out before calling this function, 
      // so we are always guaranteed that adding the first cut will succeed in this scenario, so we just let 
      // our existing logic run and add the first cut instead of special casing that handling.

      pCuttingRangeInit->m_cRangesAssigned = size_t { 0 };

      // we want to add any cutting range into the priority queue with what the result would be if we added one
      // cut, but since all of our ranges currently have one cut (zero ranges), adding one cut will get us a full
      // range, and we don't need to divide our m_cCuttableValues by 1 for the 1 range.  We just need to check
      // that it has enough samples per bin
      const size_t cCuttableItems = pCuttingRangeInit->m_cCuttableValues;
      size_t cRangesMax = 0;
      FloatEbmType avgRangeWidthAfterAddingOneCut = k_illegalAvgCuttableRangeWidthAfterAddingOneCut;
      if(LIKELY(cSamplesPerBinMin <= cCuttableItems)) {
         // don't muliply by GetTweakingMultiple, since avgRangeWidthAfterAddingOneCut is derrived from
         // size_t values, it should have exactly the same value when cCuttableItems and newProposedRanges
         // are the same, so we should then get to compare on m_uniqueTiebreaker after seeing the exact
         // floating point equality.  Also, unlike the CutPoint priority value, we don't want to affect
         // m_avgCuttableRangeWidthAfterAddingOneCut since even distant regions shouldn't have divergent
         // priorities, unlike for BinCuts
         avgRangeWidthAfterAddingOneCut = static_cast<FloatEbmType>(cCuttableItems);

         cRangesMax = DetermineRangesMax(
            cCuttableItems, 
            pCuttingRangeInit->m_pCuttableValuesFirst, 
            cSamplesPerBinMin
         );

         EBM_ASSERT(1 <= cRangesMax);
      }
      pCuttingRangeInit->m_avgCuttableRangeWidthAfterAddingOneCut = avgRangeWidthAfterAddingOneCut;
      pCuttingRangeInit->m_cRangesMax = cRangesMax;
      queue.insert(pCuttingRangeInit);

      ++pCuttingRangeInit;
   } while(LIKELY(pCuttingRangeEnd != pCuttingRangeInit));

   size_t cRemainingCuts = cCutsAssignable;
   if(UNLIKELY(0 == aCuttingRange[0].m_cUncuttableLowValues)) {
      // if our tail end is a pure tail with no uncuttable range on it's side, then we can get a range with just
      // one cut since the end of the values provides us an implicit cut.  If our tail end is an uncutable range,
      // then we need to put cuts on both ends to get a single range, so we don't get an implicit cut
      // add one to our remaining cuts to account for the implicit cut that we get at the start
      ++cRemainingCuts;
   }

   if(UNLIKELY(0 == (pCuttingRangeEnd - 1)->m_cUncuttableHighValues)) {
      // if our tail end is a pure tail with no uncuttable range on it's side, then we can get a range with just
      // one cut since the end of the values provides us an implicit cut.  If our tail end is an uncutable range,
      // then we need to put cuts on both ends to get a single range, so we don't get an implicit cut
      // add one to our remaining cuts to account for the implicit cut that we get at the end
      ++cRemainingCuts;
   }

   EBM_ASSERT(cCuttingRanges <= cRemainingCuts);
   cRemainingCuts -= cCuttingRanges;
   // the queue can initially be empty if all the ranges are too short to make them cSamplesPerBinMin
   while(LIKELY(0 != cRemainingCuts)) {
      if(AddCutToRanges(queue)) {
         break;
      }
      --cRemainingCuts;
   }
}

INLINE_RELEASE_UNTEMPLATED static void FillCuttingRangeNeighbours(
   const size_t cSamples,
   FloatEbmType * const aSingleFeatureValues,
   const size_t cCuttingRanges,
   CuttingRange * const aCuttingRange
) noexcept {
   EBM_ASSERT(2 <= cSamples); // if there wern't 2 samples we couldn't have any bins and we'd exit earliers
   EBM_ASSERT(nullptr != aSingleFeatureValues);
   EBM_ASSERT(1 <= cCuttingRanges);
   EBM_ASSERT(nullptr != aCuttingRange);

   CuttingRange * pCuttingRange = aCuttingRange;
   size_t cUncuttablePriorItems = pCuttingRange->m_pCuttableValuesFirst - aSingleFeatureValues;
   const FloatEbmType * const aSingleFeatureValuesEnd = aSingleFeatureValues + cSamples;
   const size_t cCuttingRangesMinusOne = cCuttingRanges - 1;
   if(PREDICTABLE(0 != cCuttingRangesMinusOne)) {
      // exit without doing the last one
      const CuttingRange * const pCuttingRangeLast = pCuttingRange + cCuttingRangesMinusOne;
      do {
         const size_t cUncuttableSubsequentItems = (pCuttingRange + 1)->m_pCuttableValuesFirst - 
            pCuttingRange->m_pCuttableValuesFirst - pCuttingRange->m_cCuttableValues;

         // TODO : eliminate this function after we've eliminated m_cUncuttableHighValues and wrap this functionality
         // into FillCuttingRangeBasics?

         pCuttingRange->m_cUncuttableLowValues = cUncuttablePriorItems;
         pCuttingRange->m_cUncuttableHighValues = cUncuttableSubsequentItems;

         cUncuttablePriorItems = cUncuttableSubsequentItems;
         ++pCuttingRange;
      } while(LIKELY(pCuttingRangeLast != pCuttingRange));
   }
   const size_t cUncuttableSubsequentItems =
      aSingleFeatureValuesEnd - pCuttingRange->m_pCuttableValuesFirst - pCuttingRange->m_cCuttableValues;

   pCuttingRange->m_cUncuttableLowValues = cUncuttablePriorItems;
   pCuttingRange->m_cUncuttableHighValues = cUncuttableSubsequentItems;
}

INLINE_RELEASE_UNTEMPLATED static void FillCuttingRangeBasics(
   const size_t cSamples,
   FloatEbmType * const aSingleFeatureValues,
   const size_t cUncuttableRangeLengthMin,
   const size_t cSamplesPerBinMin,
   const size_t cCuttingRanges,
   CuttingRange * const aCuttingRange
) noexcept {
   EBM_ASSERT(2 <= cSamples); // we would have exited earlier unless there were 2 bins
   EBM_ASSERT(nullptr != aSingleFeatureValues);
   EBM_ASSERT(1 <= cUncuttableRangeLengthMin);
   EBM_ASSERT(1 <= cSamplesPerBinMin);
   EBM_ASSERT(1 <= cCuttingRanges);
   EBM_ASSERT(nullptr != aCuttingRange);

   FloatEbmType rangeValue = *aSingleFeatureValues;
   FloatEbmType * pCuttableValuesStart = aSingleFeatureValues;
   const FloatEbmType * pStartEqualRange = aSingleFeatureValues;
   FloatEbmType * pScan = aSingleFeatureValues + 1;
   const FloatEbmType * const pValuesEnd = aSingleFeatureValues + cSamples;

   CuttingRange * pCuttingRange = aCuttingRange;
   do {
      const FloatEbmType val = *pScan;
      if(PREDICTABLE(val != rangeValue)) {
         size_t cEqualRangeItems = pScan - pStartEqualRange;
         if(PREDICTABLE(cUncuttableRangeLengthMin <= cEqualRangeItems)) {
            if(PREDICTABLE(
               PREDICTABLE(cSamplesPerBinMin <= static_cast<size_t>(pStartEqualRange - pCuttableValuesStart)) ||
               UNLIKELY(aSingleFeatureValues != pCuttableValuesStart))) 
            {
               EBM_ASSERT(pCuttingRange < aCuttingRange + cCuttingRanges);
               pCuttingRange->m_pCuttableValuesFirst = pCuttableValuesStart;
               pCuttingRange->m_cCuttableValues = pStartEqualRange - pCuttableValuesStart;
               ++pCuttingRange;
            }
            pCuttableValuesStart = pScan;
         }
         rangeValue = val;
         pStartEqualRange = pScan;
      }
      ++pScan;
   } while(LIKELY(pValuesEnd != pScan));
   if(LIKELY(pCuttingRange != aCuttingRange + cCuttingRanges)) {
      // we're not done, so we have one more to go.. this last one
      EBM_ASSERT(pCuttingRange == aCuttingRange + cCuttingRanges - 1);
      EBM_ASSERT(pCuttableValuesStart < pValuesEnd);
      pCuttingRange->m_pCuttableValuesFirst = pCuttableValuesStart;
      EBM_ASSERT(pStartEqualRange < pValuesEnd);
      const size_t cEqualRangeItems = pValuesEnd - pStartEqualRange;
      const FloatEbmType * const pCuttableRangeEnd = cUncuttableRangeLengthMin <= cEqualRangeItems ? 
         pStartEqualRange : pValuesEnd;
      pCuttingRange->m_cCuttableValues = pCuttableRangeEnd - pCuttableValuesStart;
   }
}

template<typename T>
static void FillTiebreakers(
   const bool bSymmetryReversal,
   RandomStream * const pRandomStream,
   const size_t cItems,
   T * const aItems
) noexcept {

   // Occasionally there will be ties in our priority calculation. We need a repeatable method to break
   // those ties so that our outputs are repetable in a consistent cross platform way.  We also want to have symmetry
   // such that if we reversed the order of our input values, we'd get the same cuts.  This isn't possible
   // in 100% of all cases, but we can get this property in 99.99999% of cases.
   
   // Before calling this function, we first detect a consistent starting side by looking for patterns which should
   // be the most invariant to transformations of the values (we first look for ranges of equal values).
   // Then we use a repeatable random number generator which will order our tiebreakers in a consistent way 
   // relative from the side we've chosen as our starting point based on the detected symmetry.
   
   // We add some consistent/repeatable noise to our priority for splitting to combat floating point inexactnes issues.
   // We therefore want our tiebreakers to roughly also follow a priority order.  Since in general, all things being 
   // equal, we prefer our initial cuts to be at the ends, we want the biggest numbers at the ends and smaller 
   // values at the center.  This will only have a practical effect when the number of samples is huge, but when
   // that happens neighbours have roughly the same priority skews, so it tends not to change the cuts in local
   // regions.  This will likekly bias cuts towards the tail ends, but that's generally what we want anyways.

   // a nice ancillary property of m_uniqueTiebreaker is that we can use it to detect distance from the center by 
   // shifting by 1, or we can get a consistent random bit which is symmetric consistent by ANDing with 1.
   // In order to preserve the distance when shifting by 1, we need the last value for odd numbers of items to
   // be 1 or 0, and the last two central items to be 1 or 0 for even numbers of items

   // If we have an odd number of items, the central item needs to be either 0 or 1 randomly to preserve the random
   // last bit property that we've created.  BUT, it's critical that this last center random bit is NOT dependent
   // on the symmetry detected.  If we get a 1 going from right to left we need to get a 1 going from left to right
   // Example: "3 1 0 2 4" when reversed is "4 2 0 1 3".  So if a transform is applies that reverses the symmetry
   // then we'll see the 3 first, then the 1, then the 0, then the 2, then the 4 consistently (or the reverse). 
   // If the center values was flipped, then you'd see a 1 in the opposite direction, which would screw up our
   // symmetry
   
   EBM_ASSERT(nullptr != pRandomStream);
   EBM_ASSERT(size_t { 1 } <= cItems);
   EBM_ASSERT(nullptr != aItems);

   // this conversion to a signed number should be ok since cItems is allocated memory with more than 
   // 2 bytes, so we should have room for the negative values.
   ptrdiff_t tiebreaker = static_cast<ptrdiff_t>((cItems - size_t { 1 }) | size_t { 1 });

   EBM_ASSERT(ptrdiff_t { 1 } <= tiebreaker);
   EBM_ASSERT(size_t { 1 } == static_cast<size_t>(tiebreaker) % 2); // we should always have an odd tiebreaker to start

   T * pLow = aItems;
   T * pHigh = aItems + cItems - size_t { 1 };
   do {
      // bSymmetryReversal helps us ensure symmetry because we pick true or false based on a fingerprint of the original 
      // values so if the values are flipped in a transform, then we'll flip bSymmetryReversal and get the same 
      // cuts mirror on the opposite sides from the ends
      const bool bRandom = pRandomStream->Next() != bSymmetryReversal; // this is an XOR for bools

      const ptrdiff_t tiebreakerMinusOne = tiebreaker - ptrdiff_t { 1 };
      const ptrdiff_t tiebreaker1 = UNPREDICTABLE(bRandom) ? tiebreaker : tiebreakerMinusOne;
      const ptrdiff_t tiebreaker2 = UNPREDICTABLE(bRandom) ? tiebreakerMinusOne : tiebreaker;

      EBM_ASSERT(ptrdiff_t { 0 } <= tiebreaker1);
      EBM_ASSERT(ptrdiff_t { 0 } <= tiebreaker2);
      EBM_ASSERT(pLow <= pHigh);

      // if we have an even number of items, the last write will be aliased (both pointers will point to the same
      // location), and we'll write either a 0 or 1 in that location selected randomly.
      pLow->m_uniqueTiebreaker = static_cast<size_t>(tiebreaker1);
      pHigh->m_uniqueTiebreaker = static_cast<size_t>(tiebreaker2);

      ++pLow;
      // this would be undefined behavior if we ended up pointing to a memory location before the 
      // beginning of our allocation, BUT we have allocated sentinal cut points on the top and bottom ends
      // so we won't wander outside our legal allocation window with this decrement
      --pHigh;

      tiebreaker -= ptrdiff_t { 2 };
   } while(ptrdiff_t { 0 } < tiebreaker);

   EBM_ASSERT(ptrdiff_t { -1 } == tiebreaker);

   EBM_ASSERT(size_t { 0 } == cItems % size_t { 2 } || pHigh + 1 == pLow - 1);
   EBM_ASSERT(size_t { 0 } != cItems % size_t { 2 } || pHigh + 1 == pLow);

   EBM_ASSERT(size_t { 0 } == cItems % size_t { 2 } || (
      size_t { 0 } == (aItems + cItems / 2)->m_uniqueTiebreaker ||
      size_t { 1 } == (aItems + cItems / 2)->m_uniqueTiebreaker
      ));
   EBM_ASSERT(size_t { 0 } != cItems % size_t { 2 } || (
      size_t { 0 } == (aItems + cItems / 2 - 1)->m_uniqueTiebreaker &&
      size_t { 1 } == (aItems + cItems / 2)->m_uniqueTiebreaker ||
      size_t { 1 } == (aItems + cItems / 2 - 1)->m_uniqueTiebreaker &&
      size_t { 0 } == (aItems + cItems / 2)->m_uniqueTiebreaker
   ));

   T * const pCenter = pHigh + size_t { 1 };
   if(pCenter != pLow) {
      // we had an odd number of items.  We will have either a 1 or 0 in the last center tiebreaker, but we
      // want the last central bit to only be random and not change when our symmetry changes.

      EBM_ASSERT(size_t { 1 } == cItems % size_t { 2 });
      EBM_ASSERT(&aItems[cItems >> 1] == pCenter);
      EBM_ASSERT(pCenter == pLow - size_t { 1 });

      // undo the application of bSymmetryReversal to the center value (see reasons at top of function)
      pCenter->m_uniqueTiebreaker ^= bSymmetryReversal ? size_t { 1 } : size_t { 0 };
   }
}

INLINE_RELEASE_UNTEMPLATED static bool DetermineSymmetricDirection(
   const size_t cSamples,
   const FloatEbmType * const aSingleFeatureValues
) noexcept {
   EBM_ASSERT(size_t { 2 } <= cSamples); // if we don't have enough samples to generate 2 bins we exit earlier
   EBM_ASSERT(nullptr != aSingleFeatureValues);

   const FloatEbmType * const pTop = aSingleFeatureValues + cSamples - size_t { 1 };

   const FloatEbmType * pLow = aSingleFeatureValues;
   const FloatEbmType * pHigh = pTop;
   FloatEbmType lowPrev = *pLow;
   FloatEbmType highPrev = *pHigh;

   // first try and see if we can differentiate by having identical values next to eachother.  Identical values
   // should be invariant to the transform, except in the rare case that two values get mapped to the same
   // value, but if we get that, then our data is probably different enough that our user shouldn't expect symmetry

   ++pLow;
   do {
      --pHigh;

      // surprisingly, this works if 2 == cSamples, since low becomes high, and the reverse and they should
      // have identical agreement or disagreement, so we don't need to check above
      EBM_ASSERT(size_t { 2 } == cSamples && pLow == 1 + pHigh || pLow <= pHigh);

      // pLow and pHigh can be aliased, and that's ok.  If that happens we want to compare to the previous values anyways
      const FloatEbmType lowCur = *pLow;
      const FloatEbmType highCur = *pHigh;

      const bool lowIdentical = lowPrev == lowCur;
      const bool highIdentical = highPrev == highCur;

      if(UNLIKELY(lowIdentical != highIdentical)) {
         // if cSamples was 2, then they should be symetric in terms of identicality since they are the same numbers
         EBM_ASSERT(size_t { 3 } <= cSamples);
         return highIdentical;
      }

      lowPrev = lowCur;
      highPrev = highCur;

      // increment pLow here so that we can compare it against high to determine if we should exit
      // pLow can legally increase to the location one past the end of the array in C++.
      // IMPORTANT: WE CANNOT DECREMENT pHigh HERE because then if cSamples == 2 WE WOULD MOVE TO A LOCATION
      // BEFORE THE BEGINNING OF THE ARRAY, and that's undefined behavior.
      ++pLow;
   } while(LIKELY(pLow < pHigh));

   // ok, we weren't able to find any differences in identical value spacing.  It's very very very likely that
   // every value is unique, otherwise the data would have to be very symmetric for some reason.
   // To figure out our symmetric direction we need to use the values in the data now.  It's hard to envision
   // a way to always determine consistent direction when the user could transform the data in the following ways:
   //   - take the negative
   //   - shift the values by addition
   //   - 1/X
   //   - log(X)
   //   - combinations of these
   //   - an infinite number of other possible transforms
   //
   // But, one thing we can usually bet on is that values at the center of the array are more likley to be closer 
   // together in value than the values at the extreme ends.  If we start from the center and look for differences
   // in value then we're likely to get a more consistent result than using the wild swings in value at the tails.
   // We use the difference between the center and the values at each point and terminate when one side or the
   // other diverges from the center by more than a noise like amount.

   if(size_t { 2 } < cSamples) {
      pHigh = aSingleFeatureValues + (cSamples >> 1);
      pLow = pHigh - (size_t { 1 } & (cSamples - size_t { 1 }));

      // pHigh and pLow might be aliased pointers.  If we have an odd number of values then there is one center
      const FloatEbmType valCenterHigh = *pHigh;
      const FloatEbmType valCenterLow = *pLow;

      do {
         ++pHigh;
         --pLow;

         const FloatEbmType valHigh = *pHigh;
         const FloatEbmType valLow = *pLow;

         const FloatEbmType distanceHigh = valHigh - valCenterHigh;
         const FloatEbmType distanceLow = valCenterLow - valLow;

         // our distance high and distance low might be integers, so don't use base 10 decimals which might
         // lead to collisions
         if(distanceLow < distanceHigh * FloatEbmType { 0.9999248297572194127975024574325 }) {
            return true;
         }
         if(distanceHigh < distanceLow * FloatEbmType { 0.9999248297572194127975024574325 }) {
            return false;
         }
      } while(aSingleFeatureValues != pLow);
   }
   pLow = aSingleFeatureValues;
   pHigh = pTop;
   do {
      EBM_ASSERT(pLow < pHigh);

      const FloatEbmType lowCur = std::abs(*pLow);
      const FloatEbmType highCur = std::abs(*pHigh);

      if(UNLIKELY(lowCur != highCur)) {
         return lowCur < highCur;
      }

      ++pLow;
      --pHigh;
      // we can exit if they are equal, since their absolute value would be equal
   } while(LIKELY(pLow < pHigh));

   // if all our values are identical, then we shouldn't have gotten any cuts, so we shouldn't have gotten here
   EBM_ASSERT(*aSingleFeatureValues < *pTop);

   // the data is perfectly symmetric centered arround zero.  Something like "-2 -1 0 1 2" would do this.  There is
   // no way for us to tell when it's been reversed even in theory.  Let's return a consistent value of false.  
   // This value is XORed with a random value later anyways, so there's no direction bias in returning either 
   // true or false here.
   return false;
}

INLINE_RELEASE_UNTEMPLATED static void ConstructJumps(
   const size_t cSamples, 
   const FloatEbmType * const aValues, 
   NeighbourJump * const aNeighbourJump
) noexcept {
   EBM_ASSERT(1 <= cSamples);
   EBM_ASSERT(nullptr != aValues);
   EBM_ASSERT(nullptr != aNeighbourJump);

   FloatEbmType valNext = aValues[0];
   const FloatEbmType * pValue = aValues;
   const FloatEbmType * const pValueEnd = aValues + cSamples;

   size_t iStartCur = 0;
   NeighbourJump * pNeighbourJump = aNeighbourJump;
   while(true) {
      const FloatEbmType valCur = valNext;
      do {
         ++pValue;
         if(UNLIKELY(pValueEnd == pValue)) {
            const size_t iStartNext = pValue - aValues;
            const NeighbourJump * const pNeighbourJumpEnd = aNeighbourJump + iStartNext;
            do {
               pNeighbourJump->m_iStartCur = iStartCur;
               pNeighbourJump->m_iStartNext = iStartNext;
               ++pNeighbourJump;
            } while(PREDICTABLE(pNeighbourJumpEnd != pNeighbourJump));

            return;
         }
         valNext = *pValue;
      } while(PREDICTABLE(valNext == valCur));

      const size_t iStartNext = pValue - aValues;
      const NeighbourJump * const pNeighbourJumpEnd = aNeighbourJump + iStartNext;
      do {
         pNeighbourJump->m_iStartCur = iStartCur;
         pNeighbourJump->m_iStartNext = iStartNext;
         ++pNeighbourJump;
      } while(PREDICTABLE(pNeighbourJumpEnd != pNeighbourJump));

      iStartCur = iStartNext;
   }
}

INLINE_RELEASE_UNTEMPLATED static size_t CountCuttingRanges(
   const size_t cSamples,
   const FloatEbmType * const aSingleFeatureValues,
   const size_t cUncuttableRangeLengthMin,
   const size_t cSamplesPerBinMin
) noexcept {
   EBM_ASSERT(size_t { 2 } <= cSamples); // if we don't have enough samples to generate 2 bins we exit earlier
   EBM_ASSERT(nullptr != aSingleFeatureValues);
   EBM_ASSERT(size_t { 1 } <= cUncuttableRangeLengthMin);
   EBM_ASSERT(size_t { 1 } <= cSamplesPerBinMin);
   EBM_ASSERT(cSamplesPerBinMin <= cSamples / size_t { 2 }); // we exit earlier if we don't have enough samples for 2 bins

   FloatEbmType rangeValue = *aSingleFeatureValues;
   const FloatEbmType * pCuttableValuesStart = aSingleFeatureValues;
   const FloatEbmType * pStartEqualRange = aSingleFeatureValues;
   const FloatEbmType * pScan = aSingleFeatureValues + 1;
   const FloatEbmType * const pValuesEnd = aSingleFeatureValues + cSamples;
   size_t cCuttingRanges = 0;
   EBM_ASSERT(pValuesEnd != pScan); // because 2 <= cSamples
   do {
      const FloatEbmType val = *pScan;
      if(PREDICTABLE(val != rangeValue)) {
         const size_t cEqualRangeItems = pScan - pStartEqualRange;
         if(cUncuttableRangeLengthMin <= cEqualRangeItems) {
            if(aSingleFeatureValues != pCuttableValuesStart || cSamplesPerBinMin <= static_cast<size_t>(pStartEqualRange - pCuttableValuesStart)) {
               ++cCuttingRanges;
            }
            pCuttableValuesStart = pScan;
         }
         rangeValue = val;
         pStartEqualRange = pScan;
      }
      ++pScan;
   } while(LIKELY(pValuesEnd != pScan));
   if(aSingleFeatureValues == pCuttableValuesStart) {
      EBM_ASSERT(0 == cCuttingRanges);

      // we're still on the first cutting range.  We need to make sure that there is at least one possible cut
      // if we require 3 items for a cut, a problematic range like 0 1 3 3 4 5 could look ok, but we can't cut it in the middle!

      const FloatEbmType * pLow = aSingleFeatureValues + cSamplesPerBinMin - 1;
      EBM_ASSERT(pLow < pValuesEnd);
      const FloatEbmType * pHigh = pValuesEnd - cSamplesPerBinMin;
      EBM_ASSERT(aSingleFeatureValues <= pHigh);
      EBM_ASSERT(pLow < pHigh);

      // if they are equal, then there are no values between them where we could cut.  
      // If unequal, there's a cut somewhere
      return UNPREDICTABLE(*pLow == *pHigh) ? size_t { 0 } : size_t { 1 };
   } else {
      const size_t cItemsLast = static_cast<size_t>(pValuesEnd - pCuttableValuesStart);
      if(cSamplesPerBinMin <= cItemsLast) {
         ++cCuttingRanges;
      }
      return cCuttingRanges;
   }
}

INLINE_RELEASE_UNTEMPLATED static size_t GetUncuttableRangeLengthMin(
   const size_t cSamples, 
   const size_t cBinsMax, 
   const size_t cSamplesPerBinMin
) noexcept {

   // !! IMPORTANT !!
   // This function returns an uncuttable range minimum length that is sufficiently long that it can GUARANTEE 
   // that all interior CuttingRanges can be assigned 1 cut.  There is no guarantee that either the first or last 
   // CuttingRange will get a cut though, unless there is an uncuttable range on both sides of the first or last
   // CuttingRange, which can happen if the value sequence either starts or ends with an uncuttable range.
   //
   // The best way to describe what this function does is via an example.  Let's say that we had 100 samples,
   // and 10 bins.  The ideal data would allow us to create 10 bins of 10 items each.  This function would return
   // the value 10 because that's the ideal range length for this hypothetical example.  If there is a range of
   // 10 identical values or more (say 15), that range is GUARANTEED to be segmented by an aspirational cut point
   // or be perfectly bound by two cut points.  In this example, a range of 15 identical values will always 
   // contain an aspirational cut point, so we will treat these long ranges differently and place explicit cut
   // points on their ends.
   //
   // Let's change the example and say that there were 101 samples, but still 10 bins. Our aspirational cut points
   // will now be on fractional numbers.  Ranges are now 10.1 items each.  We are no longer guaranteed that a
   // range of identical values 10 long will have an aspirational cut point within it, so we need to take the
   // ceiling when getting the average, so that we set our minimum uncuttable range length to 11 in this example.
   // 
   // We get one more very important GUARANTEE by taking the ceiling of the average length.  We get a guarantee
   // that we can always assign 1 cut to each interior CuttingRange.  When we partition our data, our data
   // can start and end with a cuttable range or an uncutable range.  It can also have cuttable interior ranges
   // separated by uncuttable ranges.  The general form looks like this:
   // optional_cutable | uncutable | cutable | uncutable | cutable | uncutable | optional_cutable
   // The cutable interior ranges are always bounded by uncutable ranges on both sides.  The interior cutable
   // ranges can have a length of zero.  It would be very very bad if one of these interior cutable ranges
   // was not able to get a cut, since then we'd need to choose two uncutable ranges, and an cutable range
   // to join together.  By taking the ceiling of the average length, we avoid this potential misshap.  In most cases
   // we'll also still have cut points to give to the first and last cutable ranges (if they exist), but
   // those are less critical since they can be small down to even just 1 value if cSamplesPerBinMin is 1.
   // Also, even if the first or last tail cutable range is longer it isn't as catetrophic to not get a cut since
   // we'll at most then be combining a full uncutable range with the cutable range values, and if we need to
   // make this kind of hard decision, it's best to make a bigger range at the tail ends to avoid overfitting there.
   // Also, if the two tail uncutable ranges sum to the length of a complete average range, then we'll get at least
   // one cut that we can assign to the larger of the two.
   //
   // Ok, here is why we get a guarantee that our interior cuttable ranges will get a cut.  If we discard the
   // first and last cutable ranges, then the most efficient packing method for a dataset that starts and ends
   // with an uncutable range is to have equal spaced uncutable ranges separated by zero length uncutable
   // ranges, so in the example above of 100 samples with 10 bins, this would be something like having:
   // 10 zeros, 10 ones, 10 twos, 10 threes, 10 fours, 10 fives, 10 sixes, 10 sevens, 10 eights, and 10 nines.
   // With a minimum uncutable range length of 10, there is no way to have more than 10 bins with this dataset.
   // 
   // Example of a bad situation if we took the rounded average of cSamples / cBinsMax:
   // 20 == cSamples, 9 == cBinsMax (so 8 cuts).  20 / 9 = 2.22222222222.  std::round(2.222222222) = 2.  
   // So cUncuttableRangeLengthMin would be 2 if we rounded 20 / 9  , but if our data is:
   // 0,0|1,1|2,2|3,3|4,4|5,5|6,6|7,7|8,8|9,9
   // then we get 9 CuttingRanges, but we only have 8 cuts to distribute.  And then we get to somehow choose 
   // which CuttingRange gets 0 cuts. A better choice would have been to make cUncuttableRangeLengthMin 3 
   // instead, so the ceiling.  Then we'd be guaranteed to have 8 or less CuttingRanges
   //

   EBM_ASSERT(size_t { 2 } <= cSamples); // if we don't have enough samples to generate 2 bins we exit earlier
   EBM_ASSERT(size_t { 2 } <= cBinsMax); // if there is just one bin, then you can't have cuts, so we exit earlier
   EBM_ASSERT(size_t { 1 } <= cSamplesPerBinMin);
   EBM_ASSERT(cSamplesPerBinMin <= cSamples / size_t { 2 }); // we exit earlier if we don't have enough samples for 2 bins

   size_t cUncuttableRangeLengthMin = (cSamples - size_t { 1 }) / cBinsMax + size_t { 1 }; // get the ceil value
   cUncuttableRangeLengthMin = UNPREDICTABLE(cUncuttableRangeLengthMin < cSamplesPerBinMin) ? 
      cSamplesPerBinMin : cUncuttableRangeLengthMin;

   EBM_ASSERT(size_t { 1 } <= cUncuttableRangeLengthMin);

   return cUncuttableRangeLengthMin;
}

INLINE_RELEASE_UNTEMPLATED static size_t PossiblyRemoveCutForMissing(
   const bool bMissingPresent,
   size_t cBinCutsMax
) noexcept {
   if(PREDICTABLE(bMissingPresent)) {
      // if there is a missing value, then we use 0 for the missing value bin, and bump up all other values by 1.  This creates a semi-problem
      // if the number of bins was specified as a power of two like 256, because we now have 257 possible values, and instead of consuming 8
      // bits per value, we're consuming 9.  If we're told to have a maximum of a power of two bins though, in most cases it won't hurt to
      // have one less bin so that we consume less data.  Our countBinsMax is just a maximum afterall, so we can choose to have less bins.
      // BUT, if the user requests 8 bins or less, then don't reduce the number of bins since then we'll be changing the bin size significantly

      size_t cCuts = ~size_t { 0 };
      do {
         // if cBinsMax is a power of two equal to or greater than 16, then reduce the number of bins (it's a maximum after all) to one less so that
         // it's more compressible.  If we have 256 bins, we really want 255 bins and 0 to be the missing value, using 256 values and 1 byte of storage
         // some powers of two aren't compressible, like 2^34, which needs to fit into a 64 bit storage, but we don't want to take a dependency
         // on the size of the storage system, which is system dependent, so we just exclude all powers of two
         if(UNLIKELY(cCuts == cBinCutsMax)) {
            --cBinCutsMax;
            break;
         }
         cCuts >>= 1;
         // don't allow shrinkage below 16 bins (8 is the first power of two below 16, which is 7 cuts).  By the time we reach 8 bins, we don't want to reduce this
         // by a complete bin.  We can just use an extra bit for the missing bin
         // if we had shrunk down to 7 bits for non-missing, we would have been able to fit in 21 items per data item instead of 16 for 64 bit systems
      } while(UNLIKELY(0x7 != cCuts));
   }
   return cBinCutsMax;
}

INLINE_RELEASE_UNTEMPLATED static size_t RemoveMissingValuesAndReplaceInfinities(
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
               if(PREDICTABLE(std::numeric_limits<FloatEbmType>::max() < val)) {
                  val = std::numeric_limits<FloatEbmType>::max();
                  ++cPositiveInfinity;
               } else if(PREDICTABLE(val < std::numeric_limits<FloatEbmType>::lowest())) {
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
      if(PREDICTABLE(std::numeric_limits<FloatEbmType>::max() < val)) {
         *pCopyFrom = std::numeric_limits<FloatEbmType>::max();
         ++cPositiveInfinity;
      } else if(PREDICTABLE(val < std::numeric_limits<FloatEbmType>::lowest())) {
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

// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterGenerateQuantileBinCutsParametersMessages = 25;
static int g_cLogExitGenerateQuantileBinCutsParametersMessages = 25;

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateQuantileBinCuts(
   IntEbmType countSamples,
   FloatEbmType * featureValues,
   IntEbmType countSamplesPerBinMin,
   IntEbmType isHumanized,
   IntEbmType randomSeed,
   IntEbmType * countBinCutsInOut,
   FloatEbmType * binCutsLowerBoundInclusiveOut,
   IntEbmType * countMissingValuesOut,
   FloatEbmType * minNonInfinityValueOut,
   IntEbmType * countNegativeInfinityOut,
   FloatEbmType * maxNonInfinityValueOut,
   IntEbmType * countPositiveInfinityOut
) {
   // TODO: 
   //   - we shouldn't use randomness unless impossible to do otherwise.  choosing the cut points isn't that critical to have
   //       variability for.  We can do things like hashing the data, etc to choose random values, and we should REALLY
   //       try to not use randomness, instead using things like index position, etc for that
   //       One option would be to hash the value in a cell and use the hash.  it will be randomly distributed in direction!
   //   - we can't be 100% invariant to the direction the data is presented to us for binning, but we can be 99.99999% sure
   //     by doing a combination of:
   //       1) rounding, when not falling on the center value of a cut
   //       2) if we fall on a center value, then prefer the inward direction
   //       3) if we happen to be at the index in the exact center, we can first use neighbouring cuts and continue
   //          all the way to both tail ends
   //       4) if all the cuts are the same from the center, we can use the values given to us to choose a direction
   //       5) if all of these things fail, we can use a random number

   LOG_COUNTED_N(
      &g_cLogEnterGenerateQuantileBinCutsParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered GenerateQuantileBinCuts: "
      "countSamples=%" IntEbmTypePrintf ", "
      "featureValues=%p, "
      "countSamplesPerBinMin=%" IntEbmTypePrintf ", "
      "isHumanized=%s, "
      "randomSeed=%" IntEbmTypePrintf ", "
      "countBinCutsInOut=%p, "
      "binCutsLowerBoundInclusiveOut=%p, "
      "countMissingValuesOut=%p, "
      "minNonInfinityValueOut=%p, "
      "countNegativeInfinityOut=%p, "
      "maxNonInfinityValueOut=%p, "
      "countPositiveInfinityOut=%p"
      ,
      countSamples,
      static_cast<void *>(featureValues),
      countSamplesPerBinMin,
      ObtainTruth(isHumanized),
      randomSeed,
      static_cast<void *>(countBinCutsInOut),
      static_cast<void *>(binCutsLowerBoundInclusiveOut),
      static_cast<void *>(countMissingValuesOut),
      static_cast<void *>(minNonInfinityValueOut),
      static_cast<void *>(countNegativeInfinityOut),
      static_cast<void *>(maxNonInfinityValueOut),
      static_cast<void *>(countPositiveInfinityOut)
   );

   IntEbmType countBinCutsRet;
   IntEbmType countMissingValuesRet;
   FloatEbmType minNonInfinityValueRet;
   IntEbmType countNegativeInfinityRet;
   FloatEbmType maxNonInfinityValueRet;
   IntEbmType countPositiveInfinityRet;
   IntEbmType ret;

   // if there is only 1 bin, then there can be no cut points, and no point doing any more work here
   if(UNLIKELY(nullptr == countBinCutsInOut)) {
      LOG_0(TraceLevelError, "ERROR GenerateQuantileBinCuts nullptr == countBinCutsInOut");
      countBinCutsRet = IntEbmType { 0 };
      countMissingValuesRet = IntEbmType { 0 };
      minNonInfinityValueRet = FloatEbmType { 0 };
      countNegativeInfinityRet = IntEbmType { 0 };
      maxNonInfinityValueRet = FloatEbmType { 0 };
      countPositiveInfinityRet = IntEbmType { 0 };
      ret = IntEbmType { 1 };
   } else {
      if(UNLIKELY(countSamples <= IntEbmType { 0 })) {
         countBinCutsRet = IntEbmType { 0 };
         countMissingValuesRet = IntEbmType { 0 };
         minNonInfinityValueRet = FloatEbmType { 0 };
         countNegativeInfinityRet = IntEbmType { 0 };
         maxNonInfinityValueRet = FloatEbmType { 0 };
         countPositiveInfinityRet = IntEbmType { 0 };
         ret = IntEbmType { 0 };
         if(UNLIKELY(countSamples < IntEbmType { 0 })) {
            LOG_0(TraceLevelError, "ERROR GenerateQuantileBinCuts countSamples < IntEbmType { 0 }");
            ret = IntEbmType { 1 };
         }
      } else {
         if(UNLIKELY(nullptr == featureValues)) {
            LOG_0(TraceLevelError, "ERROR GenerateQuantileBinCuts nullptr == featureValues");

            countBinCutsRet = IntEbmType { 0 };
            countMissingValuesRet = IntEbmType { 0 };
            minNonInfinityValueRet = FloatEbmType { 0 };
            countNegativeInfinityRet = IntEbmType { 0 };
            maxNonInfinityValueRet = FloatEbmType { 0 };
            countPositiveInfinityRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }

         if(UNLIKELY(!IsNumberConvertable<size_t>(countSamples))) {
            LOG_0(TraceLevelWarning, "WARNING GenerateQuantileBinCuts !IsNumberConvertable<size_t>(countSamples)");

            countBinCutsRet = IntEbmType { 0 };
            countMissingValuesRet = IntEbmType { 0 };
            minNonInfinityValueRet = FloatEbmType { 0 };
            countNegativeInfinityRet = IntEbmType { 0 };
            maxNonInfinityValueRet = FloatEbmType { 0 };
            countPositiveInfinityRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }

         const size_t cSamplesIncludingMissingValues = static_cast<size_t>(countSamples);

         if(UNLIKELY(IsMultiplyError(sizeof(*featureValues), cSamplesIncludingMissingValues))) {
            LOG_0(TraceLevelError, "ERROR GenerateQuantileBinCuts countSamples was too large to fit into featureValues");

            countBinCutsRet = IntEbmType { 0 };
            countMissingValuesRet = IntEbmType { 0 };
            minNonInfinityValueRet = FloatEbmType { 0 };
            countNegativeInfinityRet = IntEbmType { 0 };
            maxNonInfinityValueRet = FloatEbmType { 0 };
            countPositiveInfinityRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }

         const size_t cSamples = RemoveMissingValuesAndReplaceInfinities(
            cSamplesIncludingMissingValues, 
            featureValues,
            &minNonInfinityValueRet,
            &countNegativeInfinityRet,
            &maxNonInfinityValueRet,
            &countPositiveInfinityRet
         );

         EBM_ASSERT(cSamples <= cSamplesIncludingMissingValues);
         const size_t cMissingValues = cSamplesIncludingMissingValues - cSamples;
         // this is guaranteed to work since the number of missing values can't exceed the number of original
         // samples, and samples came to us as an IntEbmType
         EBM_ASSERT(IsNumberConvertable<IntEbmType>(cMissingValues));
         countMissingValuesRet = static_cast<IntEbmType>(cMissingValues);

         if(UNLIKELY(size_t { 0 } == cSamples)) {
            countBinCutsRet = IntEbmType { 0 };
            EBM_ASSERT(FloatEbmType { 0 } == minNonInfinityValueRet);
            EBM_ASSERT(IntEbmType { 0 } == countNegativeInfinityRet);
            EBM_ASSERT(FloatEbmType { 0 } == maxNonInfinityValueRet);
            EBM_ASSERT(IntEbmType { 0 } == countPositiveInfinityRet);
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }

         EBM_ASSERT(nullptr != countBinCutsInOut);
         const IntEbmType countBinCuts = *countBinCutsInOut;

         if(UNLIKELY(countBinCuts <= IntEbmType { 0 })) {
            countBinCutsRet = IntEbmType { 0 };
            ret = IntEbmType { 0 };
            if(UNLIKELY(countBinCuts < IntEbmType { 0 })) {
               LOG_0(TraceLevelError, "ERROR GenerateQuantileBinCuts countBinCuts can't be negative.");
               ret = IntEbmType { 1 };
            }
            goto exit_with_log;
         }
         
         if(UNLIKELY(nullptr == binCutsLowerBoundInclusiveOut)) {
            // if we have a potential bin cut, then binCutsLowerBoundInclusiveOut shouldn't be nullptr
            LOG_0(TraceLevelError, "ERROR GenerateQuantileBinCuts nullptr == binCutsLowerBoundInclusiveOut");

            countBinCutsRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };

            goto exit_with_log;
         }

         if(UNLIKELY(countSamplesPerBinMin <= IntEbmType { 0 })) {
            LOG_0(TraceLevelWarning,
               "WARNING GenerateQuantileBinCuts countSamplesPerBinMin shouldn't be zero or negative.  Setting to 1");

            countSamplesPerBinMin = IntEbmType { 1 };
         }

         EBM_ASSERT(IsNumberConvertable<IntEbmType>(cSamples)); // since it came from an IntEbmType originally
         if(UNLIKELY(static_cast<IntEbmType>(cSamples >> 1) < countSamplesPerBinMin)) {
            // each bin needs at least countSamplesPerBinMin samples, so we need two sets of countSamplesPerBinMin
            // in order to make any cuts.  Anything less and we should just return now.
            // We also use this as a comparison to ensure that countSamplesPerBinMin is convertible to a size_t

            countBinCutsRet = IntEbmType { 0 };
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }

         // countSamplesPerBinMin is convertible to size_t since countSamplesPerBinMin <= (cSamples >> 1)
         EBM_ASSERT(IsNumberConvertable<size_t>(countSamplesPerBinMin));
         const size_t cSamplesPerBinMin = static_cast<size_t>(countSamplesPerBinMin);

         // In theory, we could constrain our cBinsMaxInitialInitial value a bit more by taking our value array
         // and attempting to jump by the minimum each time.  Then if there was a long run of equal values we'd
         // be able to limit the number of cuts, but then the algorithm is going to need to be pretty smart later
         // on when it finds the long run and needs to compress the available cuts back down into the cutable regions
         // it's probably better to just place a lot of asiprational cuts at the minimum separation and trim them
         // as we go on so.  In that case we'd be hard pressed to misallocate cuts since they'll almost always
         // alrady be cSamplesPerBinMin apart in the regions that are cutable.
         const size_t cBinsMaxInitialInitial = cSamples / cSamplesPerBinMin;

         // otherwise we'd have failed the check "static_cast<IntEbmType>(cSamples >> 1) < countSamplesPerBinMin"
         EBM_ASSERT(size_t { 2 } <= cBinsMaxInitialInitial);
         const size_t cBinCutsMaxInitialInitial = cBinsMaxInitialInitial - size_t { 1 };

         // cSamples fit into an IntEbmType, and since cBinCutsMaxInitialInitial is less than cSamples, 
         // we should be able to convert it back to an IntEbmType
         EBM_ASSERT(cBinCutsMaxInitialInitial < cSamples);
         EBM_ASSERT(IsNumberConvertable<IntEbmType>(cBinCutsMaxInitialInitial));
         const size_t cBinCutsMaxInitial = static_cast<IntEbmType>(cBinCutsMaxInitialInitial) < countBinCuts ?
            cBinCutsMaxInitialInitial : static_cast<size_t>(countBinCuts);

         const size_t cBinCutsMax = PossiblyRemoveCutForMissing(
            IntEbmType { 0 } != countMissingValuesRet, 
            cBinCutsMaxInitial
         );
         EBM_ASSERT(size_t { 1 } <= cBinCutsMax); // we won't eliminate to less than 1, and we had at least 1 before

         std::sort(featureValues, featureValues + cSamples);

         EBM_ASSERT(cBinCutsMax < cSamples); // so we can add 1 to cBinCutsMax safely
         const size_t cUncuttableRangeLengthMin = 
            GetUncuttableRangeLengthMin(cSamples, cBinCutsMax + 1, cSamplesPerBinMin);
         EBM_ASSERT(size_t { 1 } <= cUncuttableRangeLengthMin);

         const size_t cCuttingRanges = CountCuttingRanges(
            cSamples, 
            featureValues, 
            cUncuttableRangeLengthMin, 
            cSamplesPerBinMin
         );
         // we GUARANTEE that each interior CuttingRange can have at least one cut by choosing an 
         // cUncuttableRangeLengthMin sufficiently long to ensure this property.  The first and last cutable
         // ranges, if they exist, can be quite small, so we can trade 1 long uncutable range for 2 cutable
         // ranges at the tail ends, so we can get 1 more cut than the maximum number of cuts given to us
         // but not 2 more.  cBinCutsMax + size_t { 1 } can't overflow since cBinCutsMax < cSamples , and
         // cSamples is a size_t
         EBM_ASSERT(cCuttingRanges <= cBinCutsMax + size_t { 1 });
         if(UNLIKELY(size_t { 0 } == cCuttingRanges)) {
            countBinCutsRet = IntEbmType { 0 };
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }

         if(UNLIKELY(IsMultiplyError(cSamples, sizeof(NeighbourJump)))) {
            LOG_0(TraceLevelWarning, "WARNING GenerateQuantileBinCuts IsMultiplyError(cSamples, sizeof(NeighbourJump))");
            countBinCutsRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }
         const size_t cBytesNeighbourJumps = cSamples * sizeof(NeighbourJump);

         if(UNLIKELY(IsMultiplyError(cBinCutsMax, sizeof(FloatEbmType *)))) {
            LOG_0(TraceLevelWarning, "WARNING GenerateQuantileBinCuts IsMultiplyError(cBinCutsMax, sizeof(FloatEbmType *))");
            countBinCutsRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }
         const size_t cBytesValueCutPointers = cBinCutsMax * sizeof(FloatEbmType *);

         // we limit the cBinCutsMax to no more than cSamples - 1.  cSamples can't be anywhere close to
         // the maximum size_t though since the caller must have allocated cSamples floats in featureValues, and
         // there are no float types that are 1 byte, and we checked that this didn't overflow, so we should be good
         // to add 2 to the cBinCutsMax value
         EBM_ASSERT(cBinCutsMax <= std::numeric_limits<size_t>::max() - size_t { 2 });
         // include storage for the end points
         const size_t cBinCutsWithEndpointsMax = cBinCutsMax + size_t { 2 };
         if(UNLIKELY(IsMultiplyError(cBinCutsWithEndpointsMax, sizeof(CutPoint)))) {
            LOG_0(TraceLevelWarning, "WARNING GenerateQuantileBinCuts IsMultiplyError(cBinCutsWithEndpointsMax, sizeof(CutPoint))");
            countBinCutsRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }
         const size_t cBytesBinCuts = cBinCutsWithEndpointsMax * sizeof(CutPoint);

         if(UNLIKELY(IsMultiplyError(cCuttingRanges, sizeof(CuttingRange)))) {
            LOG_0(TraceLevelWarning, "WARNING GenerateQuantileBinCuts IsMultiplyError(cCuttingRanges, sizeof(CuttingRange))");
            countBinCutsRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }
         const size_t cBytesCuttingRanges = cCuttingRanges * sizeof(CuttingRange);


         const size_t cBytesToNeighbourJump = size_t { 0 };
         const size_t cBytesToValueCutPointers = cBytesToNeighbourJump + cBytesNeighbourJumps;

         if(UNLIKELY(IsAddError(cBytesToValueCutPointers, cBytesValueCutPointers))) {
            LOG_0(TraceLevelWarning, "WARNING GenerateQuantileBinCuts IsAddError(cBytesToValueCutPointers, cBytesValueCutPointers))");
            countBinCutsRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }
         const size_t cBytesToBinCuts = cBytesToValueCutPointers + cBytesValueCutPointers;

         if(UNLIKELY(IsAddError(cBytesToBinCuts, cBytesBinCuts))) {
            LOG_0(TraceLevelWarning, "WARNING GenerateQuantileBinCuts IsAddError(cBytesToBinCuts, cBytesBinCuts))");
            countBinCutsRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }
         const size_t cBytesToCuttingRange = cBytesToBinCuts + cBytesBinCuts;

         if(UNLIKELY(IsAddError(cBytesToCuttingRange, cBytesCuttingRanges))) {
            LOG_0(TraceLevelWarning, "WARNING GenerateQuantileBinCuts IsAddError(cBytesToCuttingRange, cBytesCuttingRanges))");
            countBinCutsRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }
         const size_t cBytesToEnd = cBytesToCuttingRange + cBytesCuttingRanges;

         char * const pMem = static_cast<char *>(malloc(cBytesToEnd));
         if(UNLIKELY(nullptr == pMem)) {
            LOG_0(TraceLevelWarning, "WARNING GenerateQuantileBinCuts nullptr == pMem");
            countBinCutsRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }

         NeighbourJump * const aNeighbourJumps = reinterpret_cast<NeighbourJump *>(pMem + cBytesToNeighbourJump);
         const FloatEbmType ** const apValueCutTops = reinterpret_cast<const FloatEbmType **>(pMem + cBytesToValueCutPointers);
         CutPoint * const aBinCuts = reinterpret_cast<CutPoint *>(pMem + cBytesToBinCuts);
         CuttingRange * const aCuttingRange = reinterpret_cast<CuttingRange *>(pMem + cBytesToCuttingRange);

         ConstructJumps(cSamples, featureValues, aNeighbourJumps);

         // we always XOR (with != for bools) a random number with bSymmetryReversal, so there is no need to
         // XOR bSymmetryReversal with a random number here
         const bool bSymmetryReversal = DetermineSymmetricDirection(cSamples, featureValues);

         RandomStream randomStream;
         randomStream.Initialize(randomSeed);

         FillTiebreakers(bSymmetryReversal, &randomStream, cCuttingRanges, aCuttingRange);

         FillCuttingRangeBasics(cSamples, featureValues, cUncuttableRangeLengthMin, cSamplesPerBinMin, cCuttingRanges, aCuttingRange);
         FillCuttingRangeNeighbours(cSamples, featureValues, cCuttingRanges, aCuttingRange);

         const FloatEbmType ** ppValueCutTop = apValueCutTops;
         try {
            std::set<CuttingRange *, CompareCuttingRange> priorityQueue;
            StuffCutsIntoCuttingRanges(
               priorityQueue,
               cCuttingRanges,
               aCuttingRange,
               cSamplesPerBinMin,
               cBinCutsMax
            );
            do {
               EBM_ASSERT(!priorityQueue.empty());
               // remove the item that is the worst CuttingRange for us to add a new cut to.  We'll keep
               // the cutting ranges that are closest to the threshold for adding new cuts in the queue so that
               // if we can't use all our cuts, we can move the cuts to the next best choice
               auto iterator = prev(priorityQueue.end());
               CuttingRange * const pCuttingRange = *iterator;
               priorityQueue.erase(iterator);

               const size_t cRanges = pCuttingRange->m_cRangesAssigned;

#ifdef LOG_SUPERVERBOSE_DISCRETIZATION_ORDERED
               LOG_N(TraceLevelVerbose, "Dequque CuttingRange: %zu, %zu, %zu, %zu, %zu, %zu, %zu, %" FloatEbmTypePrintf,
                  pCuttingRange->m_uniqueTiebreaker,
                  pCuttingRange->m_cRangesAssigned,
                  pCuttingRange->m_cCuttableValues,
                  static_cast<size_t>(pCuttingRange->m_pCuttableValuesFirst - featureValues),
                  pCuttingRange->m_cUncuttableHighValues,
                  pCuttingRange->m_cUncuttableLowValues,
                  pCuttingRange->m_cRangesMax,
                  pCuttingRange->m_avgCuttableRangeWidthAfterAddingOneCut
               );
#endif // LOG_SUPERVERBOSE_DISCRETIZATION_ORDERED

               if(PREDICTABLE(size_t { 1 } < cRanges)) {
                  // we have cuts on our ends, either explicit or implicit at the tail ends that don't have unsplitable
                  // ranges on the tails, and at least one cut in our center, so we have to make decisions
                  std::set<CutPoint *, CompareCutPoint> bestBinCuts;

#ifdef NEVER
                  // TODO : in the future fill this priority queue with the average length within our
                  //        visibility window AFTER a new cut would be added.  We calculate this value per
                  //        CutPoint and we do it at the same time we're calculating the cut priority, which
                  //        is good since we'll already have the visibility windows calculated and all that.
                  //        One wrinkle is that we want to be able to insert a cut into a range that no longer
                  //        has any internal cuts.  So for instance if we had a range from 50 to 100 with
                  //        materialized cuts on both 50 and 100, and no allocated cuts between them, in
                  //        the future if cuts become plentiful, then we want to create a new cut between
                  //        those materialized cuts.  I believe the best way to handle this is to check
                  //        when materializing a cut if both our lower and higher cut points are aspirational
                  //        or materialized.  If they are both materialized, then insert our new materialized
                  //        cut into the open space priority queue AND the cut to the left (which represents)
                  //        the lower range.  Or if that's too complicated then take the maximum min from both
                  //        our sides and insert ourselves with that.  We can always examine the left and right
                  //        on extraction to determine which side we should go to.
                  //        Inisde CalculateRangesMaximizeMin, we might notice that one of our sides doesn't
                  //        work very well with a certain number of cuts.  We should speculatively move
                  //        one of our cuts from that side to a new set of ranges (encoded as BinCuts)
                  //        We still do the low/high cut number optimization with our left and right windows
                  //        when planning since it's more efficient, and no changes should leak information
                  //        outside those windows otherwise it would become an N^2 algorithm.
                  //        We use our doubly linked list to move non-materialized cut points long distances
                  //        from one part of the cutting range to annother if necessary.
                  //        We should also use the doubly linked list to delete BinCuts that we can't use
                  //        if there is no place to put them

                  std::set<CutPoint *, CompareCutPoint> fillTheVoids;
#endif // NEVER

                  FillTiebreakers(bSymmetryReversal, &randomStream, cRanges - size_t { 1 }, aBinCuts + 1);
                  if(TradeCutSegment(
                     &bestBinCuts,
                     cSamples,
                     bSymmetryReversal,
                     cSamplesPerBinMin,
                     pCuttingRange->m_pCuttableValuesFirst - featureValues,
                     pCuttingRange->m_cCuttableValues,
                     aNeighbourJumps,
                     cRanges,
                     // for efficiency we include space for the end point cuts even if they don't exist
                     aBinCuts
                  )) {
                     // any error messages should have been written to the log inside TradeCutSegment

                     free(pMem);

                     countBinCutsRet = IntEbmType { 0 };
                     ret = IntEbmType { 1 };
                     goto exit_with_log;
                  }

                  const FloatEbmType * const pCuttableValuesStart = pCuttingRange->m_pCuttableValuesFirst;

                  if(0 != pCuttingRange->m_cUncuttableLowValues) {
                     // if it's zero then it's an implicit cut and we shouldn't put one there, 
                     // otherwise put in the cut
                     const FloatEbmType * const pCut = pCuttableValuesStart;
                     EBM_ASSERT(featureValues < pCut);
                     EBM_ASSERT(pCut < featureValues + countSamples);
                     *ppValueCutTop = pCut;
                     ++ppValueCutTop;
                  }

                  const CutPoint * pCutPoint = aBinCuts->m_pNext;
                  const CutPoint * pNext = pCutPoint->m_pNext;
                  while(LIKELY(nullptr != pNext)) {
                     const size_t iVal = pCutPoint->m_iVal;
                     if(LIKELY(k_valNotLegal != iVal)) {
                        const FloatEbmType * const pCut = pCuttableValuesStart + iVal;
                        EBM_ASSERT(featureValues < pCut);
                        EBM_ASSERT(pCut < featureValues + countSamples);
                        EBM_ASSERT(pCuttingRange->m_pCuttableValuesFirst < pCut);
                        EBM_ASSERT(pCut < pCuttingRange->m_pCuttableValuesFirst + pCuttingRange->m_cCuttableValues);
                        *ppValueCutTop = pCut;
                        ++ppValueCutTop;
                     }
                     pCutPoint = pNext;
                     pNext = pCutPoint->m_pNext;
                  }

                  if(0 != pCuttingRange->m_cUncuttableHighValues) {
                     // if it's zero then it's an implicit cut and we shouldn't put one there, 
                     // otherwise put in the cut
                     const FloatEbmType * const pCut =
                        pCuttableValuesStart + pCuttingRange->m_cCuttableValues;
                     EBM_ASSERT(featureValues < pCut);
                     EBM_ASSERT(pCut < featureValues + countSamples);
                     *ppValueCutTop = pCut;
                     ++ppValueCutTop;
                  }
               } else if(PREDICTABLE(size_t { 1 } == cRanges)) {
                  // we have cuts on both our ends (either explicit or implicit), so
                  // we don't have to make any hard decisions, but we do have to be careful of the scenarios
                  // where some of our cuts are implicit

                  if(0 != pCuttingRange->m_cUncuttableLowValues) {
                     // if it's zero then it's an implicit cut and we shouldn't put one there, 
                     // otherwise put in the cut
                     const FloatEbmType * const pCut = pCuttingRange->m_pCuttableValuesFirst;
                     EBM_ASSERT(featureValues < pCut);
                     EBM_ASSERT(pCut < featureValues + countSamples);
                     *ppValueCutTop = pCut;
                     ++ppValueCutTop;
                  }
                  if(0 != pCuttingRange->m_cUncuttableHighValues) {
                     // if it's zero then it's an implicit cut and we shouldn't put one there, 
                     // otherwise put in the cut
                     const FloatEbmType * const pCut =
                        pCuttingRange->m_pCuttableValuesFirst + pCuttingRange->m_cCuttableValues;
                     EBM_ASSERT(featureValues < pCut);
                     EBM_ASSERT(pCut < featureValues + countSamples);
                     *ppValueCutTop = pCut;
                     ++ppValueCutTop;
                  }
               } else {
                  EBM_ASSERT(0 == cRanges);
                  // we have only 1 cut to place, and no cuts on our boundaries, so we need to figure out
                  // where in our range to place it, taking into consideration that we might have neighbours on our
                  // sides that could be large

                  // if we had implicit cuts on both ends and zero assigned cuts, we'd have 1 range and would
                  // be handled above
                  EBM_ASSERT(0 != pCuttingRange->m_cUncuttableLowValues || 0 != pCuttingRange->m_cUncuttableHighValues);

                  // if one side or the other was an implicit cut, then we have zero cuts left after
                  // the implicit cut is accounted for, so do nothing
                  if(LIKELY(LIKELY(0 != pCuttingRange->m_cUncuttableLowValues) && 
                     LIKELY(0 != pCuttingRange->m_cUncuttableHighValues))) {
                     // even though we could reduce our squared error length more, it probably makes sense to 
                     // include a little bit of our available numbers on one long range and the other, so let's put
                     // the cut in the middle and only make the low/high decision to settle long-ish ranges
                     // in the center

                     const FloatEbmType * pCut = pCuttingRange->m_pCuttableValuesFirst;
                     const size_t cCuttableItems = pCuttingRange->m_cCuttableValues;
                        
                     const size_t iRangeFirst = pCuttingRange->m_pCuttableValuesFirst - featureValues;
                     const size_t iCenterOfRange = iRangeFirst + (cCuttableItems >> 1);

                     // unlike in BuildNeighbourhoodPlan, we don't need to worry about the scenario that
                     // a jumping range falls on the exact iCenterOfRange value, since for our purposes here
                     // if we have a perfect answer that is perfectly in the center, then we always select that
                     // one since we have no exclusion criteria here.  We never will seriously consider the 
                     // iStartNext value if iStartCur is a perfectly centered match.
                     // So we don't need to inject some randomness here, unlike in BuildNeighbourhoodPlan

                     const NeighbourJump * const pNeighbourJump = &aNeighbourJumps[iCenterOfRange];

                     const size_t iStartCur = pNeighbourJump->m_iStartCur;
                     const size_t iStartNext = pNeighbourJump->m_iStartNext;

                     const ptrdiff_t cDistanceLow1 = static_cast<ptrdiff_t>(iStartCur - iRangeFirst);
                     EBM_ASSERT(ptrdiff_t { 0 } <= cDistanceLow1);
                     EBM_ASSERT(cDistanceLow1 <= static_cast<ptrdiff_t>(cCuttableItems >> 1));
                     // cDistanceHigh1 can be negative if cCuttableItems is zero since then iStartNext
                     // will reflect the boundary of the point after the unsplittable range above
                     // our cut point, but since our cDistanceLow1 will be zero, it'll work out without
                     // a special check
                     const ptrdiff_t cDistanceHigh1 = static_cast<ptrdiff_t>(iRangeFirst + cCuttableItems) 
                        - static_cast<ptrdiff_t>(iStartNext);
                     EBM_ASSERT(cDistanceHigh1 <= static_cast<ptrdiff_t>(cCuttableItems >> 1));
                     EBM_ASSERT(size_t { 1 } == cCuttableItems % size_t { 2 } ||
                        cDistanceHigh1 < static_cast<ptrdiff_t>(cCuttableItems >> 1));

                     size_t iResult = UNPREDICTABLE(cDistanceHigh1 < cDistanceLow1) ? iStartCur : iStartNext;
                     if(UNLIKELY(cDistanceHigh1 == cDistanceLow1)) {
                        // per above, we can't get the situation where iCenterOfRange is the perfect center
                        // past our if check above for cDistanceHigh1 == cDistanceLow1
                        EBM_ASSERT(static_cast<size_t>(cDistanceLow1) * size_t { 2 } != cCuttableItems);

                        // we're equidistant to both edges.  Next try to see which is closer to the outer
                        // edge if we include the uncuttable ranges beyond
                        const size_t cDistanceLow2 = pCuttingRange->m_cUncuttableLowValues;
                        const size_t cDistanceHigh2 = pCuttingRange->m_cUncuttableHighValues;
                        iResult = UNPREDICTABLE(cDistanceHigh2 < cDistanceLow2) ? iStartCur : iStartNext;
                        if(UNLIKELY(cDistanceHigh2 == cDistanceLow2)) {
                           // next, let's try to the edges of our full array
                           const size_t cDistanceLow3 = iStartCur;
                           const size_t cDistanceHigh3 = cSamples - iStartNext;
                           iResult = UNPREDICTABLE(cDistanceHigh3 < cDistanceLow3) ? iStartCur : iStartNext;
                           if(UNLIKELY(cDistanceHigh3 == cDistanceLow3)) {
                              // wow, we're at the center of the entire array AND the center of the outer
                              // unsplittable ranges, AND the center of the splitable ranges.  Our final fallback
                              // is to resort to our symmetric determination (PLUS randomness)

                              bool bLocalSymmetryReversal = randomStream.Next() != bSymmetryReversal;
                              iResult = UNPREDICTABLE(bLocalSymmetryReversal) ? iStartCur : iStartNext;
                           }
                        }
                     }
                     pCut = featureValues + iResult;
                     EBM_ASSERT(featureValues < pCut);
                     *ppValueCutTop = pCut;
                     ++ppValueCutTop;
                  }
               }
            } while(!priorityQueue.empty());
         } catch(...) {
            LOG_0(TraceLevelWarning, "WARNING GenerateQuantileBinCuts exception");

            free(pMem);

            countBinCutsRet = IntEbmType { 0 };
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }

         EBM_ASSERT(apValueCutTops <= ppValueCutTop);
         const size_t cBinCutsRet = ppValueCutTop - apValueCutTops;

         // it's possible, although extremely unlikely, that due to floating point issues that should only
         // occur with huge double indexes, we were not able to find the legal cut point, so check for zero
         if(LIKELY(size_t { 0 } != cBinCutsRet)) {
            // the pointers are guaranteed to be in same order as the cut values
            std::sort(apValueCutTops, ppValueCutTop);

            FloatEbmType * pBinCutsLowerBoundInclusive = binCutsLowerBoundInclusiveOut;
            const FloatEbmType * const * ppValueCutTop2 = apValueCutTops;

            if(EBM_FALSE == isHumanized) {
               do {
                  const FloatEbmType * const pCut = *ppValueCutTop2;
                  EBM_ASSERT(featureValues < pCut);
                  EBM_ASSERT(pCut < featureValues + cSamples);
                  const FloatEbmType valHigh = *pCut;
                  EBM_ASSERT(!std::isnan(valHigh));
                  EBM_ASSERT(!std::isinf(valHigh));
                  const FloatEbmType valLow = *(pCut - size_t { 1 });
                  EBM_ASSERT(!std::isnan(valLow));
                  EBM_ASSERT(!std::isinf(valLow));
                  const FloatEbmType cut = ArithmeticMean(valLow, valHigh);
                  EBM_ASSERT(binCutsLowerBoundInclusiveOut == pBinCutsLowerBoundInclusive || *(pBinCutsLowerBoundInclusive - size_t { 1 }) < cut);
                  *pBinCutsLowerBoundInclusive = cut;
                  ++pBinCutsLowerBoundInclusive;
                  ++ppValueCutTop2;
               } while(ppValueCutTop != ppValueCutTop2);
            } else {
               do {
                  const FloatEbmType * const pCut = *ppValueCutTop2;
                  EBM_ASSERT(featureValues < pCut);
                  EBM_ASSERT(pCut < featureValues + cSamples);
                  const FloatEbmType valHigh = *pCut;
                  EBM_ASSERT(!std::isnan(valHigh));
                  EBM_ASSERT(!std::isinf(valHigh));
                  const FloatEbmType valLow = *(pCut - size_t { 1 });
                  EBM_ASSERT(!std::isnan(valLow));
                  EBM_ASSERT(!std::isinf(valLow));
                  const FloatEbmType cut = GetInterpretableCutPointFloat(valLow, valHigh);
                  EBM_ASSERT(binCutsLowerBoundInclusiveOut == pBinCutsLowerBoundInclusive || *(pBinCutsLowerBoundInclusive - size_t { 1 }) < cut);
                  *pBinCutsLowerBoundInclusive = cut;
                  ++pBinCutsLowerBoundInclusive;
                  ++ppValueCutTop2;
               } while(ppValueCutTop != ppValueCutTop2);

               // if you have 1 cut point, then you get a graph with some mass on the left, some mass on the right
               // and the cut point, and that's great.  We don't need to improve on that.  Our one cut points provides
               // the most information possible and it's displayable on a graph.
               // eg: "0.01 0.01 | 100 100" -> put the cut point at 1 and we can show both logit sides without
               // indicating the min/max values of 0.001 and 1000
               //
               // if you have 2 cut points, then the graph will have 3 regions, and we can scale the graph so that
               // 1/3 of the mass in on the left, 1/3 is in the scaled center, and 1/3 is on the right.  Whatever cuts
               // we get provide the most amount of information possible, and it's graphable.
               // eg: "0.01 0.01 | 1 1 | 100 100" -> put the cut points at 0.1 and 10 and the graph can range
               // from 0.1 to 10 with some space on the tails to show the logits for the "-infinity -> 0.1" bin
               // and the "10 -> +infinity" bin.
               //
               // if we have 3 cut points, then we could get into graphing issues if one of the ranges was so big
               // that it dwarfed the other two in size.  We can't do anything about this if one of the interior
               // ranges is huge, but often times the huge range is at the extreme ends of the graph and if the
               // value on the interior side is smaller then we have some ability to pick the cut point.
               // eg: "1 1 | 2 2 | 3 3 | infinity infinity".  The cut points can legally be:
               //         1.5   2.5   3.5
               // but if the values were instead:
               // eg: "1 1 | 2 2 | 3 3 | 3.2 3.2".  The cut points can't exceed 3.2, so we'd use:
               //         1.5   2.5   3.1
               //
               // Our algorithm finds 3.5 and 3.1 and picks the minimum, and the same on the low side, but there we
               // take the maximum.
               //
               // In the above example, our graph must at minimum show the data from 2 -> 3, and in fact we'll want
               // to not put our cuts right outside 2 and 3, so we want to move a reasonable distance away from those
               // ends to the 1.5 and 3.5 positions to put our cuts, and since the "-infinity -> 1.5" bin and
               // "3.5 -> +infinity" bins have logits, we also want some space on the graph to show those logits
               // so we probably want our graph to show something like the space 0 -> 5, although this can be
               // chosen by the graphing function.
               //
               // It's tempting to want to use the interior cuts to determine the outer cuts:
               // eg: "-infinity -infinity | 2 2 | 3 3 | 4 4 | +infinity +infinity"
               //                         1.5   2.5   3.5   4.5
               // We might want to use 2.5 and 3.5 to determine that the cuts progress with distnaces of 1, and
               // extrapolate 2.5 - 1 = 1.5 and 3.5 + 1 = 4.5, but we can't really do that because we might instead have
               // something like this where the extrapolation will put us below the highLow value
               // eg: "-infinity -infinity | 2 2 | 3 3 | 9 9 | +infinity +infinity"
               // So we need to use the 9 value and extend from there.
               //
               // In the examples above, we've chosen point values, but we could easily have the following situation:
               // 0.6 1.4 | 1.6 2.4 | 2.6 3.4 | 3.6 4.4 | 4.6 5.4
               //        1.5       2.5       3.5       4.5
               // which illustrates that in general the cut points can be very close to their neighbouring values.
               // so in the examples farther above we had a spacing of 0.5 units from the interior values to the
               // exterior cuts (1.5 -> 2) and (3 -> 3.5), but here we have separations of 0.1 (1.4 -> 1.5) and
               // "4.4 -> 4.5".  
               // 
               // We're only choosing to override the averaged cut value when the outer value is a huge way off
               // so we probably want to be conservative about how much we're override this and not put the
               // new cut point too close to our lowHigh or highLow values.  If we start from a pointalism point
               // of view that all the interior values are bunched onto discrete values like "2 2", and we assume
               // half of the distance between a value and it's cut occurs on the lower and higher side, it gives
               // us a kind of worse case reasonable scenario to deal with.  So starting from:
               // "-infinity -infinity | 2 2 | 3 3 | 4 4 | +infinity +infinity"
               //                     1.5   2.5   3.5   4.5
               // We get the minimum graph range by taking the 4 and the 2 and substracting for 2.
               // Then we assume that half of the bin on the upper side of the 2 is within that range and
               // the lower side of the 4 is within that range, and we know that there is a range bounding 3,
               // so we have 0.5 + 1 + 0.5 ranges total = 2.
               // So our cut density is 2 / 2 = 1 cut per range.
               // and we extend by half a bin downwards from the 2, which gives (2 - 1 / 2) = 1.5
               // and we extend by half a bin upwards from the 4, which gives (4 + 1 / 2) = 4.5

               if(LIKELY(size_t { 3 } <= cBinCutsRet)) {
                  const FloatEbmType * const pScaleHighHigh = *(ppValueCutTop - size_t { 1 });
                  EBM_ASSERT(featureValues + size_t { 2 } < pScaleHighHigh);
                  EBM_ASSERT(pScaleHighHigh < featureValues + cSamples);
                  const FloatEbmType * const pScaleHighLow = pScaleHighHigh - size_t { 1 };
                  EBM_ASSERT(featureValues + size_t { 1 } < pScaleHighLow);
                  EBM_ASSERT(pScaleHighLow < featureValues + cSamples - size_t { 1 });
                  const FloatEbmType scaleHighLow = *pScaleHighLow;
                  EBM_ASSERT(!std::isnan(scaleHighLow));
                  EBM_ASSERT(!std::isinf(scaleHighLow));
                  const FloatEbmType * pScaleLowHigh = *apValueCutTops;
                  EBM_ASSERT(featureValues < pScaleLowHigh);
                  EBM_ASSERT(pScaleLowHigh < featureValues + cSamples - size_t { 2 });
                  const FloatEbmType scaleLowHigh = *pScaleLowHigh;
                  EBM_ASSERT(!std::isnan(scaleLowHigh));
                  EBM_ASSERT(!std::isinf(scaleLowHigh));
                  EBM_ASSERT(scaleLowHigh < scaleHighLow);
                  // this is the inescapable scale of our graph, from the value right above the lowest cut to the value 
                  // right below the highest cut
                  const FloatEbmType scaleMin = scaleHighLow - scaleLowHigh;
                  EBM_ASSERT(!std::isnan(scaleMin));
                  EBM_ASSERT(!std::isinf(scaleMin));
                  // IEEE 754 (which we static_assert) won't allow the subtraction of two unequal numbers to be non-zero
                  EBM_ASSERT(FloatEbmType { 0 } < scaleMin);

                  // limit the amount of dillution allowed for the tails by capping the relevant cCutPointRet value
                  // to 1/32, which means we leave about 3% of the visible area to tail bounds (1.5% on the left and
                  // 1.5% on the right)

                  const size_t cBinCutsLimited = size_t { 32 } < cBinCutsRet ? size_t { 32 } : cBinCutsRet;

                  // the leftmost and rightmost cuts can legally be right outside of the bounds between scaleHighLow and
                  // scaleLowHigh, so we subtract these two cuts, leaving us the number of ranges between the two end
                  // points.  Half a range on the bottom, N - 1 ranges in the middle, and half a range on the top
                  // Dividing by that number of ranges gives us the average range width.  We don't want to get the final
                  // cut though from the previous inner cut.  We want to move outwards from the scaleHighLow and
                  // scaleLowHigh values, which should be half a cut inwards (not exactly but in spirit), so we
                  // divide by two, which is the same as multiplying the divisor by 2, which is the right shift below
                  const size_t denominator = (cBinCutsLimited - size_t { 2 }) << 1;
                  EBM_ASSERT(size_t { 0 } < denominator);
                  const FloatEbmType movementFromEnds = scaleMin / static_cast<FloatEbmType>(denominator);

                  EBM_ASSERT(!std::isnan(movementFromEnds));
                  EBM_ASSERT(!std::isinf(movementFromEnds));
                  EBM_ASSERT(FloatEbmType { 0 } <= movementFromEnds);

                  const FloatEbmType lowCutFullPrecisionMin = scaleLowHigh - movementFromEnds;
                  EBM_ASSERT(!std::isnan(lowCutFullPrecisionMin));
                  EBM_ASSERT(lowCutFullPrecisionMin < std::numeric_limits<FloatEbmType>::max());
                  const FloatEbmType highCutFullPrecisionMax = scaleHighLow + movementFromEnds;
                  EBM_ASSERT(!std::isnan(highCutFullPrecisionMax));
                  EBM_ASSERT(std::numeric_limits<FloatEbmType>::lowest() < highCutFullPrecisionMax);

                  const FloatEbmType lowCutMin = GetInterpretableEndpoint(lowCutFullPrecisionMin, movementFromEnds);
                  const FloatEbmType highCutMax = GetInterpretableEndpoint(highCutFullPrecisionMax, movementFromEnds);

                  const FloatEbmType lowCutExisting = *binCutsLowerBoundInclusiveOut;
                  EBM_ASSERT(!std::isnan(lowCutExisting));
                  EBM_ASSERT(!std::isinf(lowCutExisting));
                  const FloatEbmType highCutExisting = *(pBinCutsLowerBoundInclusive - size_t { 1 });
                  EBM_ASSERT(!std::isnan(highCutExisting));
                  EBM_ASSERT(!std::isinf(highCutExisting));

                  if(lowCutExisting < lowCutMin) {
                     // lowCutMin can legally be -infinity, but then we wouldn't get here then
                     EBM_ASSERT(!std::isnan(lowCutMin));
                     EBM_ASSERT(!std::isinf(lowCutMin));
                     *binCutsLowerBoundInclusiveOut = lowCutMin;
                  }
                  if(highCutMax < highCutExisting) {
                     // highCutMax can legally be +infinity, but then we wouldn't get here then
                     EBM_ASSERT(!std::isnan(highCutMax));
                     EBM_ASSERT(!std::isinf(highCutMax));
                     *(pBinCutsLowerBoundInclusive - size_t { 1 }) = highCutMax;
                  }
               }
            }
         }

         // this conversion is guaranteed to work since the number of cut points can't exceed the number our user
         // specified, and that value came to us as an IntEbmType
         countBinCutsRet = static_cast<IntEbmType>(cBinCutsRet);
         EBM_ASSERT(countBinCutsRet <= countBinCuts);

         free(pMem);

         ret = IntEbmType { 0 };
      }

   exit_with_log:;

      EBM_ASSERT(nullptr != countBinCutsInOut);
      *countBinCutsInOut = countBinCutsRet;
   }

   if(LIKELY(nullptr != countMissingValuesOut)) {
      *countMissingValuesOut = countMissingValuesRet;
   }
   if(LIKELY(nullptr != minNonInfinityValueOut)) {
      *minNonInfinityValueOut = minNonInfinityValueRet;
   }
   if(LIKELY(nullptr != countNegativeInfinityOut)) {
      *countNegativeInfinityOut = countNegativeInfinityRet;
   }
   if(LIKELY(nullptr != maxNonInfinityValueOut)) {
      *maxNonInfinityValueOut = maxNonInfinityValueRet;
   }
   if(LIKELY(nullptr != countPositiveInfinityOut)) {
      *countPositiveInfinityOut = countPositiveInfinityRet;
   }

   LOG_COUNTED_N(
      &g_cLogExitGenerateQuantileBinCutsParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Exited GenerateQuantileBinCuts: "
      "countBinCuts=%" IntEbmTypePrintf ", "
      "countMissingValues=%" IntEbmTypePrintf ", "
      "minNonInfinityValue=%" FloatEbmTypePrintf ", "
      "countNegativeInfinity=%" IntEbmTypePrintf ", "
      "maxNonInfinityValue=%" FloatEbmTypePrintf ", "
      "countPositiveInfinity=%" IntEbmTypePrintf ", "
      "return=%" IntEbmTypePrintf
      ,
      countBinCutsRet,
      countMissingValuesRet,
      minNonInfinityValueRet,
      countNegativeInfinityRet,
      maxNonInfinityValueRet,
      countPositiveInfinityRet,
      ret
   );

   return ret;
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateWinsorizedBinCuts(
   IntEbmType countSamples,
   FloatEbmType * featureValues,
   IntEbmType randomSeed,
   IntEbmType * countBinCutsInOut,
   FloatEbmType * binCutsLowerBoundInclusiveOut,
   IntEbmType * countMissingValuesOut,
   FloatEbmType * minNonInfinityValueOut,
   IntEbmType * countNegativeInfinityOut,
   FloatEbmType * maxNonInfinityValueOut,
   IntEbmType * countPositiveInfinityOut
) {
   UNUSED(countSamples);
   UNUSED(featureValues);
   UNUSED(randomSeed);
   UNUSED(countBinCutsInOut);
   UNUSED(binCutsLowerBoundInclusiveOut);
   UNUSED(countMissingValuesOut);
   UNUSED(minNonInfinityValueOut);
   UNUSED(countNegativeInfinityOut);
   UNUSED(maxNonInfinityValueOut);
   UNUSED(countPositiveInfinityOut);

   // TODO: IMPLEMENT

   return IntEbmType { 1 };
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GenerateUniformBinCuts(
   IntEbmType countSamples,
   FloatEbmType * featureValues,
   IntEbmType randomSeed,
   IntEbmType * countBinCutsInOut,
   FloatEbmType * binCutsLowerBoundInclusiveOut,
   IntEbmType * countMissingValuesOut,
   FloatEbmType * minNonInfinityValueOut,
   IntEbmType * countNegativeInfinityOut,
   FloatEbmType * maxNonInfinityValueOut,
   IntEbmType * countPositiveInfinityOut
) {
   UNUSED(countSamples);
   UNUSED(featureValues);
   UNUSED(randomSeed);
   UNUSED(countBinCutsInOut);
   UNUSED(binCutsLowerBoundInclusiveOut);
   UNUSED(countMissingValuesOut);
   UNUSED(minNonInfinityValueOut);
   UNUSED(countNegativeInfinityOut);
   UNUSED(maxNonInfinityValueOut);
   UNUSED(countPositiveInfinityOut);

   // TODO: IMPLEMENT

   return IntEbmType { 1 };
}

// Plan:
//   - when making predictions, in the great majority of cases, we should serially determine the logits of each
//     sample per feature and then later add those logits.  It's tempting to want to process more than one feature
//     at a time, but that's a red-hearing:
//     - data typically gets passed to us as C ordered data, so feature0 and feature1 are in adjacent memory
//       cells, and sample0 and sample1 are distant.  It's less costly to read the data per feature for our pure input
//       data.  It wouldn't do us much good though if we striped just two features at a time, so we'd want to
//       process all N features in order to take advantage of this property.  But if you do that, then we'd need
//       to do binary searches on a single sample for a single feature, then fetch into cache the next feature's
//       cut "definition".  The cost of constantly bringing into L1 cache the cut points and logits for each feature
//       would entail more memory movement than either processing the matrix out of order or transposing it beforehand
//     - it's tempting to then consider striping just 2 features or some limited subset.  We get limited speed benefits
//       when processing two features at a time since at best it halves the time to access the matrix, but we still
//       then need to keep two cut point arrays that we do unpredictable branches on and it potentially pushes some
//       of our cut point and logit arrays out from L1 cache into L2 or beyond
//     - we get benefits by having special case algorithms based on the number of cut points (see below where we
//       do linear searches for small numbers of cut points, and pad cut point arrays for slightly larger numbers of
//       cut points).  And it's hard to see how we could combine these together and say have a special loop to handle
//       when one feature has 3 cut points, and the other has 50 cut points
//     - one of the benefits of doing 2 features at once would be that we could add the logits together and write
//       the sum to memory instead of writing both logits and later reading those again and summing them and writing
//       them back to memory, but since we'd be doing this with correcly ordered memory, we'd be able to stream
//       the reads and the writes such that they'd take approx 1 clock cycle each, so in reality we don't gain much
//       from combining the logits at binary search time
//     - in theory we might gain something if we had two single cut features because we could load the 2 values we're
//       cutting into 2 registers, then have the cut points in 2 persistent registers, and have 4 registers for the
//       logit results.  We can overwrite one of the two registers loaded with the sum of the resulting logits.  
//       That's a total of 8 registers.  For 2 cuts, we'd need 2 for loading, 4 for cuts, 6 for logits, so 12 registers
//       Which is also doable.  Beyond that, we'd need to use or access memory when combining processing for 2 features
//       and I think it would be better to pay the streaming to memory cost than to fetch somewhat unpredictably
//       the cut points or logits
//     - even if we did write special case code for handling two binary features, it won't help us if the matrix the
//       user passes us doesn't put the binary features adjacent to eachother.  We can't re-arrange the columsn for
//       less than the cost of partial transposes, so we'd rather just go with partial transposes
//     - doing a partial striped transpose is 32% faster in my tests than reading 2 columns at once, so we'd be
//       better off transposing the two columns than process them.  This is because we are limited to reading just
//       two values efficiently at a time, rather than reading a full stripe efficiently.
//   - we can get data from the user as fortran ordered.  If it comes to us fortran ordered
//     then great, because our accessing that data per feature is very efficient (approx 1 clock cycle per read)
//   - we can get data from the user as C ordered (this is more common).  We could read the matrix in poor memory
//     order, but then we're still loading in a complete cache line at a time.  It makes more sense to read in data
//     in a stripe and transpose it that way.  I did some perfs, and reading stripes of 64 doubles was fastest
//     We pay the cost of having 64 write streams, but our reads are very fast.  That's the break even point though
//   - transposing the complete matrix would double our memory requirements.  Since transposing is fastest with 64
//     doubles though, we can extract and transpose our original C ordered data in 64 feature groupings
//   - we can use SIMD easily enough by loading the next 2/4/8 doubles at a time and re-using the same cut definition
//     within a single processor
//   - we can use threading efficiently in one of two ways.  We can subdivide the samples up by the number of CPUs
//     and have each CPU process those ranges.  This allows all the CPUs to utilize the same cut point definitions
//     but they have smaller batches.  Alternatively, we can give each CPU one feature and have it load the cut
//     point and logit definitions into it's L1 cache which isn't likely to be shared.  If some of the cut points
//     or logits need to be in L2 though, there might be bad contention.
//   - hyper-threads would probably benefit from having the same cut points and logits since both hyper-threads share
//     the L1 cahce, so the "best" solution is probably use thread afinity to keep CPUs working on the same feature
//     and dividing up the samples between the hyper-threads, but then benefit from larger batch sizes by putting
//     different features on different CPUs
//   - the exact threading solution will probably depend on exact numbers of samples and threads and machine 
//     architecture
//   - whether dividing the work by samples or features or a mix, if we make multiple calls into our discritize
//     function, we would want to preserve our threads since they are costly to make, so we'd want to have a
//     thread allocation object that we'd free after discretization
//   - for fortran ordered arrays, the user might as well pass us the entire array and we'll process it directly
//   - for C ordered data, either the 64 stride transpose happens in our higher level caller, or they just pass
//     us the C ordered data, and we do the partial transposes inside C++ from the badly ordered original data
//   - in the entire dataset gets passed to us, then we don't need a thread allocation object since we just do it once
//   - if the original array is in pandas, it seems to be stored internally as a numpy array if the datatypes are all
//     the same, so we can pass that direclty into our function
//   - if the original array is in pandas, and consists of strings or integers or anything heterogenious, then
//     the data appears to be fortran ordered.  In that case we'd like to pass the data in that bare format
//   - but we're not sure that pandas stores these as 2-D matricies or multiple 1-D arrays.  If the ladder, then
//     we either need to process it one array at a time, or copy the data together.
//   - handling strings can either be done with python vectorized functions or in cython (try pure python first)
//   - after our per-feature logit arrays have been written, we can load in several at a time and add them together
//     and write out the result, and we can parallelize that operation until all the logits have been added
//   - SIMD reads and writes are better on certain boundaries.  We don't control the data passed to us from the user
//     so we might want to read the first few instances with a special binary search function and then start
//     on the SIMD on a memory aligned boundary, then also use the special binary search function for the last few
//   - one complication is that for pairs we need to have both feature in memory to evaluate.  If the pairs are
//     not in the same stripe we need to preserve them until they are.  In most cases we can probably just hold the
//     features we need or organize which stripes we load at which times, but in the worst case we may want
//     to re-discretize some features, or in the worst case discretize all features (preserving in a compressed 
//     format?).  This really needs to be threshed out.
//
//   - Table of matrix access speeds (for summing cells in a matrix):
//       bad_order = 7.43432
//       stride_1 = 7.27575
//       stride_2 = 4.08857
//       stride_16384 = 0.431882
//       transpose_1 = 10.4326
//       transpose_2 = 6.49787
//       transpose_4 = 4.54615
//       transpose_8 = 3.42918
//       transpose_16 = 3.04755
//       transpose_32 = 2.80757
//       transpose_64 = 2.75464
//       transpose_128 = 2.79845
//       transpose_256 = 2.8748
//       transpose_512 = 2.96725
//       transpose_1024 = 3.17072
//       transpose_2048 = 6.04042
//       transpose_4096 = 6.1348
//       transpose_8192 = 6.26907
//       transpose_16384 = 7.73406

// don't bother using a lock here.  We don't care if an extra log message is written out due to thread parallism
static int g_cLogEnterDiscretizeParametersMessages = 25;
static int g_cLogExitDiscretizeParametersMessages = 25;

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION Discretize(
   IntEbmType countSamples,
   const FloatEbmType * featureValues,
   IntEbmType countBinCuts,
   const FloatEbmType * binCutsLowerBoundInclusive,
   IntEbmType * discretizedOut
) {
   LOG_COUNTED_N(
      &g_cLogEnterDiscretizeParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered Discretize: "
      "countSamples=%" IntEbmTypePrintf ", "
      "featureValues=%p, "
      "countBinCuts=%" IntEbmTypePrintf ", "
      "binCutsLowerBoundInclusive=%p, "
      "discretizedOut=%p"
      ,
      countSamples,
      static_cast<const void *>(featureValues),
      countBinCuts,
      static_cast<const void *>(binCutsLowerBoundInclusive),
      static_cast<void *>(discretizedOut)
   );

   IntEbmType ret;
   if(UNLIKELY(countSamples <= IntEbmType { 0 })) {
      if(UNLIKELY(countSamples < IntEbmType { 0 })) {
         LOG_0(TraceLevelError, "ERROR Discretize countSamples cannot be negative");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      } else {
         EBM_ASSERT(IntEbmType { 0 } == countSamples);
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }
   } else {
      if(UNLIKELY(!IsNumberConvertable<size_t>(countSamples))) {
         // this needs to point to real memory, otherwise it's invalid
         LOG_0(TraceLevelError, "ERROR Discretize countSamples was too large to fit into memory");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      const size_t cSamples = static_cast<size_t>(countSamples);

      if(IsMultiplyError(sizeof(*featureValues), cSamples)) {
         LOG_0(TraceLevelError, "ERROR Discretize countSamples was too large to fit into featureValues");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      if(IsMultiplyError(sizeof(*discretizedOut), cSamples)) {
         LOG_0(TraceLevelError, "ERROR Discretize countSamples was too large to fit into discretizedOut");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      if(UNLIKELY(nullptr == featureValues)) {
         LOG_0(TraceLevelError, "ERROR Discretize featureValues cannot be null");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      if(UNLIKELY(nullptr == discretizedOut)) {
         LOG_0(TraceLevelError, "ERROR Discretize discretizedOut cannot be null");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      const FloatEbmType * pValue = featureValues;
      const FloatEbmType * const pValueEnd = featureValues + cSamples;
      IntEbmType * pDiscretized = discretizedOut;

      if(UNLIKELY(countBinCuts <= IntEbmType { 0 })) {
         if(UNLIKELY(countBinCuts < IntEbmType { 0 })) {
            LOG_0(TraceLevelError, "ERROR Discretize countBinCuts cannot be negative");
            ret = IntEbmType { 1 };
            goto exit_with_log;
         }
         EBM_ASSERT(IntEbmType { 0 } == countBinCuts);

         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 1 } : IntEbmType { 0 };
            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      if(UNLIKELY(nullptr == binCutsLowerBoundInclusive)) {
         LOG_0(TraceLevelError, "ERROR Discretize binCutsLowerBoundInclusive cannot be null");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

#ifndef NDEBUG
      if(IsNumberConvertable<size_t>(countBinCuts)) {
         const size_t cBinCuts = static_cast<size_t>(countBinCuts);
         size_t iDebug = 0;
         while(true) {
            EBM_ASSERT(!std::isnan(binCutsLowerBoundInclusive[iDebug]));
            EBM_ASSERT(!std::isinf(binCutsLowerBoundInclusive[iDebug]));

            size_t iDebugInc = iDebug + 1;
            if(cBinCuts <= iDebugInc) {
               break;
            }
            // if the values aren't increasing, we won't crash, but we'll return non-sensical bins.  That's a tollerable
            // failure though given that this check might be expensive if cBinCuts was large compared to cSamples
            EBM_ASSERT(binCutsLowerBoundInclusive[iDebug] < binCutsLowerBoundInclusive[iDebugInc]);
            iDebug = iDebugInc;
         }
      }
# endif // NDEBUG

      if(PREDICTABLE(IntEbmType { 1 } == countBinCuts)) {
         const FloatEbmType cut0 = binCutsLowerBoundInclusive[0];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 1 } : IntEbmType { 0 };
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 2 } : result;

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 2 } == countBinCuts)) {
         const FloatEbmType cut0 = binCutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = binCutsLowerBoundInclusive[1];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 1 } : IntEbmType { 0 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 2 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 3 } : result;

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 3 } == countBinCuts)) {
         const FloatEbmType cut0 = binCutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = binCutsLowerBoundInclusive[1];
         const FloatEbmType cut2 = binCutsLowerBoundInclusive[2];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 1 } : IntEbmType { 0 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 2 } : result;
            result = UNPREDICTABLE(cut2 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 4 } : result;

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 4 } == countBinCuts)) {
         const FloatEbmType cut0 = binCutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = binCutsLowerBoundInclusive[1];
         const FloatEbmType cut2 = binCutsLowerBoundInclusive[2];
         const FloatEbmType cut3 = binCutsLowerBoundInclusive[3];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 1 } : IntEbmType { 0 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 2 } : result;
            result = UNPREDICTABLE(cut2 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(cut3 <= val) ? IntEbmType { 4 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 5 } : result;

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 5 } == countBinCuts)) {
         const FloatEbmType cut0 = binCutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = binCutsLowerBoundInclusive[1];
         const FloatEbmType cut2 = binCutsLowerBoundInclusive[2];
         const FloatEbmType cut3 = binCutsLowerBoundInclusive[3];
         const FloatEbmType cut4 = binCutsLowerBoundInclusive[4];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 1 } : IntEbmType { 0 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 2 } : result;
            result = UNPREDICTABLE(cut2 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(cut3 <= val) ? IntEbmType { 4 } : result;
            result = UNPREDICTABLE(cut4 <= val) ? IntEbmType { 5 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 6 } : result;

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 6 } == countBinCuts)) {
         const FloatEbmType cut0 = binCutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = binCutsLowerBoundInclusive[1];
         const FloatEbmType cut2 = binCutsLowerBoundInclusive[2];
         const FloatEbmType cut3 = binCutsLowerBoundInclusive[3];
         const FloatEbmType cut4 = binCutsLowerBoundInclusive[4];
         const FloatEbmType cut5 = binCutsLowerBoundInclusive[5];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 1 } : IntEbmType { 0 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 2 } : result;
            result = UNPREDICTABLE(cut2 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(cut3 <= val) ? IntEbmType { 4 } : result;
            result = UNPREDICTABLE(cut4 <= val) ? IntEbmType { 5 } : result;
            result = UNPREDICTABLE(cut5 <= val) ? IntEbmType { 6 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 7 } : result;

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 7 } == countBinCuts)) {
         // for digitization during training this all fits into registers, but if we evaluate the logits at the same
         // time for mains it doesn't quite fit into 16 registers.  We have 1 value that we load from memory to
         // process, 7 cut points, 8 logits bin cut logits (for prediction), and 1 logit for the missing value bin
         // (for prediction), which is 17 registers.  We need to load some things from memory therefore.  We can
         // do the final load for the missing bin logit after the comparison operator of isnan, so we could probably
         // just load 1 value, otherwise we'd need to load 2 values since loading them would overwrite one of them

         // TODO: since we can't fit everything into registers, perhaps we'd get better performance doing a full
         // binary search using memory, thus minimizing the number of comparisons?

         const FloatEbmType cut0 = binCutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = binCutsLowerBoundInclusive[1];
         const FloatEbmType cut2 = binCutsLowerBoundInclusive[2];
         const FloatEbmType cut3 = binCutsLowerBoundInclusive[3];
         const FloatEbmType cut4 = binCutsLowerBoundInclusive[4];
         const FloatEbmType cut5 = binCutsLowerBoundInclusive[5];
         const FloatEbmType cut6 = binCutsLowerBoundInclusive[6];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 1 } : IntEbmType { 0 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 2 } : result;
            result = UNPREDICTABLE(cut2 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(cut3 <= val) ? IntEbmType { 4 } : result;
            result = UNPREDICTABLE(cut4 <= val) ? IntEbmType { 5 } : result;
            result = UNPREDICTABLE(cut5 <= val) ? IntEbmType { 6 } : result;
            result = UNPREDICTABLE(cut6 <= val) ? IntEbmType { 7 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 8 } : result;

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      if(PREDICTABLE(IntEbmType { 8 } == countBinCuts)) {
         // TODO: for 8 specifically, we could probably do a special case of binary search with everything in
         //       registers.  It would take just 3 comparisons then instead of 8.  7 is less benefit since we'd then 
         //       still have 3 comparisons required plus one to handle the padding case at the end, so we'd have 
         //       4 total and more register moves.  8 is probably the only sweet spot special case.

         const FloatEbmType cut0 = binCutsLowerBoundInclusive[0];
         const FloatEbmType cut1 = binCutsLowerBoundInclusive[1];
         const FloatEbmType cut2 = binCutsLowerBoundInclusive[2];
         const FloatEbmType cut3 = binCutsLowerBoundInclusive[3];
         const FloatEbmType cut4 = binCutsLowerBoundInclusive[4];
         const FloatEbmType cut5 = binCutsLowerBoundInclusive[5];
         const FloatEbmType cut6 = binCutsLowerBoundInclusive[6];
         const FloatEbmType cut7 = binCutsLowerBoundInclusive[7];
         do {
            const FloatEbmType val = *pValue;
            IntEbmType result;

            result = UNPREDICTABLE(cut0 <= val) ? IntEbmType { 1 } : IntEbmType { 0 };
            result = UNPREDICTABLE(cut1 <= val) ? IntEbmType { 2 } : result;
            result = UNPREDICTABLE(cut2 <= val) ? IntEbmType { 3 } : result;
            result = UNPREDICTABLE(cut3 <= val) ? IntEbmType { 4 } : result;
            result = UNPREDICTABLE(cut4 <= val) ? IntEbmType { 5 } : result;
            result = UNPREDICTABLE(cut5 <= val) ? IntEbmType { 6 } : result;
            result = UNPREDICTABLE(cut6 <= val) ? IntEbmType { 7 } : result;
            result = UNPREDICTABLE(cut7 <= val) ? IntEbmType { 8 } : result;
            result = UNPREDICTABLE(std::isnan(val)) ? IntEbmType { 9 } : result;

            *pDiscretized = result;
            ++pDiscretized;
            ++pValue;
         } while(LIKELY(pValueEnd != pValue));
         ret = IntEbmType { 0 };
         goto exit_with_log;
      }

      FloatEbmType binCutsLowerBoundInclusiveCopy[1023];
      if(PREDICTABLE(countBinCuts <= IntEbmType { 15 })) {
         constexpr size_t cPower = 16;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(binCutsLowerBoundInclusiveCopy) /
               sizeof(binCutsLowerBoundInclusiveCopy[0]), "binCutsLowerBoundInclusiveCopy buffer not large enough");

            const size_t cBinCuts = static_cast<size_t>(countBinCuts);
            const size_t cSkip = cPower - 1 - cBinCuts;

            for(size_t i = 0; i < cSkip; ++i) {
               binCutsLowerBoundInclusiveCopy[i] = -std::numeric_limits<FloatEbmType>::infinity();
            }

            char * const pBaseValid = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy + cSkip);
            memcpy(pBaseValid, binCutsLowerBoundInclusive, sizeof(*binCutsLowerBoundInclusive) * cBinCuts);

            const size_t missingVal = cBinCuts + size_t { 1 };
            const FloatEbmType firstComparison = binCutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               size_t result = (pResult - pBaseValid) / sizeof(FloatEbmType);
               result = UNPREDICTABLE(std::isnan(val)) ? missingVal : result;

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countBinCuts <= IntEbmType { 31 })) {
         constexpr size_t cPower = 32;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(binCutsLowerBoundInclusiveCopy) /
               sizeof(binCutsLowerBoundInclusiveCopy[0]), "binCutsLowerBoundInclusiveCopy buffer not large enough");

            const size_t cBinCuts = static_cast<size_t>(countBinCuts);
            const size_t cSkip = cPower - 1 - cBinCuts;

            for(size_t i = 0; i < cSkip; ++i) {
               binCutsLowerBoundInclusiveCopy[i] = -std::numeric_limits<FloatEbmType>::infinity();
            }

            char * const pBaseValid = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy + cSkip);
            memcpy(pBaseValid, binCutsLowerBoundInclusive, sizeof(*binCutsLowerBoundInclusive) * cBinCuts);

            const size_t missingVal = cBinCuts + size_t { 1 };
            const FloatEbmType firstComparison = binCutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               size_t result = (pResult - pBaseValid) / sizeof(FloatEbmType);
               result = UNPREDICTABLE(std::isnan(val)) ? missingVal : result;

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countBinCuts <= IntEbmType { 63 })) {
         constexpr size_t cPower = 64;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(binCutsLowerBoundInclusiveCopy) /
               sizeof(binCutsLowerBoundInclusiveCopy[0]), "binCutsLowerBoundInclusiveCopy buffer not large enough");

            const size_t cBinCuts = static_cast<size_t>(countBinCuts);
            const size_t cSkip = cPower - 1 - cBinCuts;

            for(size_t i = 0; i < cSkip; ++i) {
               binCutsLowerBoundInclusiveCopy[i] = -std::numeric_limits<FloatEbmType>::infinity();
            }

            char * const pBaseValid = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy + cSkip);
            memcpy(pBaseValid, binCutsLowerBoundInclusive, sizeof(*binCutsLowerBoundInclusive) * cBinCuts);

            const size_t missingVal = cBinCuts + size_t { 1 };
            const FloatEbmType firstComparison = binCutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 15 } * sizeof(FloatEbmType)) <= val) ? size_t { 16 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               size_t result = (pResult - pBaseValid) / sizeof(FloatEbmType);
               result = UNPREDICTABLE(std::isnan(val)) ? missingVal : result;

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countBinCuts <= IntEbmType { 127 })) {
         constexpr size_t cPower = 128;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(binCutsLowerBoundInclusiveCopy) /
               sizeof(binCutsLowerBoundInclusiveCopy[0]), "binCutsLowerBoundInclusiveCopy buffer not large enough");

            const size_t cBinCuts = static_cast<size_t>(countBinCuts);
            const size_t cSkip = cPower - 1 - cBinCuts;

            for(size_t i = 0; i < cSkip; ++i) {
               binCutsLowerBoundInclusiveCopy[i] = -std::numeric_limits<FloatEbmType>::infinity();
            }

            char * const pBaseValid = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy + cSkip);
            memcpy(pBaseValid, binCutsLowerBoundInclusive, sizeof(*binCutsLowerBoundInclusive) * cBinCuts);

            const size_t missingVal = cBinCuts + size_t { 1 };
            const FloatEbmType firstComparison = binCutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 31 } * sizeof(FloatEbmType)) <= val) ? size_t { 32 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 15 } * sizeof(FloatEbmType)) <= val) ? size_t { 16 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               size_t result = (pResult - pBaseValid) / sizeof(FloatEbmType);
               result = UNPREDICTABLE(std::isnan(val)) ? missingVal : result;

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countBinCuts <= IntEbmType { 255 })) {
         constexpr size_t cPower = 256;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(binCutsLowerBoundInclusiveCopy) /
               sizeof(binCutsLowerBoundInclusiveCopy[0]), "binCutsLowerBoundInclusiveCopy buffer not large enough");

            const size_t cBinCuts = static_cast<size_t>(countBinCuts);
            const size_t cSkip = cPower - 1 - cBinCuts;

            for(size_t i = 0; i < cSkip; ++i) {
               binCutsLowerBoundInclusiveCopy[i] = -std::numeric_limits<FloatEbmType>::infinity();
            }

            char * const pBaseValid = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy + cSkip);
            memcpy(pBaseValid, binCutsLowerBoundInclusive, sizeof(*binCutsLowerBoundInclusive) * cBinCuts);

            const size_t missingVal = cBinCuts + size_t { 1 };
            const FloatEbmType firstComparison = binCutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 63 } * sizeof(FloatEbmType)) <= val) ? size_t { 64 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 31 } * sizeof(FloatEbmType)) <= val) ? size_t { 32 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 15 } * sizeof(FloatEbmType)) <= val) ? size_t { 16 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               size_t result = (pResult - pBaseValid) / sizeof(FloatEbmType);
               result = UNPREDICTABLE(std::isnan(val)) ? missingVal : result;

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countBinCuts <= IntEbmType { 511 })) {
         constexpr size_t cPower = 512;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 <= sizeof(binCutsLowerBoundInclusiveCopy) /
               sizeof(binCutsLowerBoundInclusiveCopy[0]), "binCutsLowerBoundInclusiveCopy buffer not large enough");

            const size_t cBinCuts = static_cast<size_t>(countBinCuts);
            const size_t cSkip = cPower - 1 - cBinCuts;

            for(size_t i = 0; i < cSkip; ++i) {
               binCutsLowerBoundInclusiveCopy[i] = -std::numeric_limits<FloatEbmType>::infinity();
            }

            char * const pBaseValid = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy + cSkip);
            memcpy(pBaseValid, binCutsLowerBoundInclusive, sizeof(*binCutsLowerBoundInclusive) * cBinCuts);

            const size_t missingVal = cBinCuts + size_t { 1 };
            const FloatEbmType firstComparison = binCutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 127 } * sizeof(FloatEbmType)) <= val) ? size_t { 128 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 63 } * sizeof(FloatEbmType)) <= val) ? size_t { 64 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 31 } * sizeof(FloatEbmType)) <= val) ? size_t { 32 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 15 } * sizeof(FloatEbmType)) <= val) ? size_t { 16 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               size_t result = (pResult - pBaseValid) / sizeof(FloatEbmType);
               result = UNPREDICTABLE(std::isnan(val)) ? missingVal : result;

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }
      } else if(PREDICTABLE(countBinCuts <= IntEbmType { 1023 })) {
         constexpr size_t cPower = 1024;
         if(cPower * 4 <= cSamples) {
            static_assert(cPower - 1 == sizeof(binCutsLowerBoundInclusiveCopy) /
               sizeof(binCutsLowerBoundInclusiveCopy[0]), "binCutsLowerBoundInclusiveCopy buffer not large enough");

            const size_t cBinCuts = static_cast<size_t>(countBinCuts);
            const size_t cSkip = cPower - 1 - cBinCuts;

            for(size_t i = 0; i < cSkip; ++i) {
               binCutsLowerBoundInclusiveCopy[i] = -std::numeric_limits<FloatEbmType>::infinity();
            }

            char * const pBaseValid = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy + cSkip);
            memcpy(pBaseValid, binCutsLowerBoundInclusive, sizeof(*binCutsLowerBoundInclusive) * cBinCuts);

            const size_t missingVal = cBinCuts + size_t { 1 };
            const FloatEbmType firstComparison = binCutsLowerBoundInclusiveCopy[cPower / 2 - 1];
            do {
               const FloatEbmType val = *pValue;
               char * pResult = reinterpret_cast<char *>(binCutsLowerBoundInclusiveCopy);

               pResult += UNPREDICTABLE(firstComparison <= val) ? size_t { cPower / 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 255 } * sizeof(FloatEbmType)) <= val) ? size_t { 256 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 127 } * sizeof(FloatEbmType)) <= val) ? size_t { 128 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 63 } * sizeof(FloatEbmType)) <= val) ? size_t { 64 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 31 } * sizeof(FloatEbmType)) <= val) ? size_t { 32 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 15 } * sizeof(FloatEbmType)) <= val) ? size_t { 16 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 7 } * sizeof(FloatEbmType)) <= val) ? size_t { 8 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 3 } * sizeof(FloatEbmType)) <= val) ? size_t { 4 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult + size_t { 1 } * sizeof(FloatEbmType)) <= val) ? size_t { 2 } * sizeof(FloatEbmType) : size_t { 0 };
               pResult += UNPREDICTABLE(*reinterpret_cast<FloatEbmType *>(pResult) <= val) ? size_t { 1 } * sizeof(FloatEbmType) : size_t { 0 };

               size_t result = (pResult - pBaseValid) / sizeof(FloatEbmType);
               result = UNPREDICTABLE(std::isnan(val)) ? missingVal : result;

               *pDiscretized = static_cast<IntEbmType>(result);
               ++pDiscretized;
               ++pValue;
            } while(LIKELY(pValueEnd != pValue));
            ret = IntEbmType { 0 };
            goto exit_with_log;
         }
      }

      if(UNLIKELY(std::numeric_limits<IntEbmType>::max() == countBinCuts)) {
         // we convert back to IntEbmType when we return, and if countBinCuts is at the limit, then we don't
         // have any value to indicate missing
         LOG_0(TraceLevelError,
            "ERROR Discretize countBinCuts was too large to allow for a missing value placeholder");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      if(UNLIKELY(!IsNumberConvertable<size_t>(countBinCuts))) {
         // this needs to point to real memory, otherwise it's invalid
         LOG_0(TraceLevelError, "ERROR Discretize countBinCuts was too large to fit into memory");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      const size_t cBinCuts = static_cast<size_t>(countBinCuts);

      if(IsMultiplyError(sizeof(*binCutsLowerBoundInclusive), cBinCuts)) {
         LOG_0(TraceLevelError,
            "ERROR Discretize countBinCuts was too large to fit into binCutsLowerBoundInclusive");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      if(UNLIKELY(std::numeric_limits<size_t>::max() == cBinCuts)) {
         // we add 1 to cBinCuts as our missing value, so this addition must succeed
         LOG_0(TraceLevelError,
            "ERROR Discretize countBinCuts was too large to allow for a missing value placeholder");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      if(UNLIKELY(size_t { std::numeric_limits<ptrdiff_t>::max() } < cBinCuts)) {
         // the low value can increase until it's equal to cBinCuts, so cBinCuts must be expressable as a ptrdiff_t
         LOG_0(TraceLevelError,
            "ERROR Discretize countBinCuts was too large to allow for the binary search comparison");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      if(UNLIKELY(std::numeric_limits<size_t>::max() / size_t { 2 } + size_t { 1 } < cBinCuts)) {
         // our first operation towards getting the mid-point is to add the size_t low and size_t high, and that can't 
         // overflow, so check that the maximum high added to the maximum low (which is the high) don't exceed that value
         LOG_0(TraceLevelError,
            "ERROR Discretize countBinCuts was too large to allow for the binary search add");
         ret = IntEbmType { 1 };
         goto exit_with_log;
      }

      EBM_ASSERT(cBinCuts < std::numeric_limits<size_t>::max());
      const size_t missingVal = cBinCuts + size_t { 1 };
      EBM_ASSERT(size_t { 1 } <= cBinCuts);
      EBM_ASSERT(cBinCuts - size_t { 1 } <= size_t { std::numeric_limits<ptrdiff_t>::max() });
      const ptrdiff_t highStart = static_cast<ptrdiff_t>(cBinCuts - size_t { 1 });

      // if we're going to runroll our first loop, then we need to ensure that there's a next loop after the first
      // unrolled loop, otherwise we would need to check if we were done before the first real loop iteration.
      // To ensure we have 2 original loop iterations, we need 1 cut in the center, 1 cut above, and 1 cut below, so 3
      EBM_ASSERT(size_t { 3 } <= cBinCuts);
      const size_t firstMiddle = static_cast<size_t>(highStart) >> 1;
      EBM_ASSERT(firstMiddle < cBinCuts);
      const FloatEbmType firstMidVal = binCutsLowerBoundInclusive[firstMiddle];
      const ptrdiff_t firstMidLow = static_cast<ptrdiff_t>(firstMiddle) + ptrdiff_t { 1 };
      const ptrdiff_t firstMidHigh = static_cast<ptrdiff_t>(firstMiddle) - ptrdiff_t { 1 };

      do {
         const FloatEbmType val = *pValue;
         size_t middle = missingVal;
         if(PREDICTABLE(!std::isnan(val))) {
            ptrdiff_t high = UNPREDICTABLE(firstMidVal <= val) ? highStart : firstMidHigh;
            ptrdiff_t low = UNPREDICTABLE(firstMidVal <= val) ? firstMidLow : ptrdiff_t { 0 };
            FloatEbmType midVal;
            do {
               EBM_ASSERT(ptrdiff_t { 0 } <= low && static_cast<size_t>(low) < cBinCuts);
               EBM_ASSERT(ptrdiff_t { 0 } <= high && static_cast<size_t>(high) < cBinCuts);
               EBM_ASSERT(low <= high);
               // low is equal or lower than high, so summing them can't exceed 2 * high, and after division it
               // can't be higher than high, so middle can't overflow ptrdiff_t after the division since high
               // is already a ptrdiff_t.  Generally the maximum positive value of a ptrdiff_t can be doubled 
               // when converted to a size_t, although that isn't guaranteed.  A more correct statement is that
               // the following must be false (which we check above):
               // "std::numeric_limits<size_t>::max() / 2 < cBinCuts - 1"
               EBM_ASSERT(!IsAddError(static_cast<size_t>(low), static_cast<size_t>(high)));
               middle = (static_cast<size_t>(low) + static_cast<size_t>(high)) >> 1;
               EBM_ASSERT(middle <= static_cast<size_t>(high));
               EBM_ASSERT(middle < cBinCuts);
               midVal = binCutsLowerBoundInclusive[middle];
               EBM_ASSERT(middle < size_t { std::numeric_limits<ptrdiff_t>::max() });
               low = UNPREDICTABLE(midVal <= val) ? static_cast<ptrdiff_t>(middle) + ptrdiff_t { 1 } : low;
               EBM_ASSERT(ptrdiff_t { 0 } <= low && static_cast<size_t>(low) <= cBinCuts);
               high = UNPREDICTABLE(midVal <= val) ? high : static_cast<ptrdiff_t>(middle) - ptrdiff_t { 1 };
               EBM_ASSERT(ptrdiff_t { -1 } <= high && high <= highStart);

               // high can become -1 in some cases, so it needs to be ptrdiff_t.  It's tempting to try and change
               // this code and use the Hermann Bottenbruch version that checks for low != high in the loop comparison
               // since then we wouldn't have negative values and we could use size_t, but unfortunately that version
               // has a check at the end where we'd need to fetch binCutsLowerBoundInclusive[low] after exiting the 
               // loop, so this version we have here is faster given that we only need to compare to a value that
               // we've already fetched from memory.  Also, this version makes slightly faster progress since
               // it does middle + 1 AND middle - 1 instead of just middle - 1, so it often eliminates one loop
               // iteration.  In practice this version will always work since no floating point type is less than 4
               // bytes, so we shouldn't have difficulty expressing any indexes with ptrdiff_t, and our indexes
               // for accessing memory are always size_t, so those should always work.
            } while(LIKELY(low <= high));
            EBM_ASSERT(size_t { 0 } <= middle && middle < cBinCuts);
            middle = UNPREDICTABLE(midVal <= val) ? middle + size_t { 1 } : middle;
            EBM_ASSERT(size_t { 0 } <= middle && middle <= cBinCuts);
         }
         EBM_ASSERT(IsNumberConvertable<IntEbmType>(middle));
         *pDiscretized = static_cast<IntEbmType>(middle);
         ++pDiscretized;
         ++pValue;
      } while(LIKELY(pValueEnd != pValue));
      ret = IntEbmType { 0 };
   }

exit_with_log:;

   LOG_COUNTED_N(
      &g_cLogExitDiscretizeParametersMessages, 
      TraceLevelInfo, 
      TraceLevelVerbose, 
      "Exited Discretize: "
      "return=%" IntEbmTypePrintf
      ,
      ret
   );

   return ret;
}

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION SuggestGraphBounds(
   IntEbmType countBinCuts,
   FloatEbmType * binCutsLowerBoundInclusive,
   FloatEbmType minValue,
   FloatEbmType maxValue,
   FloatEbmType * lowBoundOut,
   FloatEbmType * highBoundOut
) {
   UNUSED(countBinCuts);
   UNUSED(binCutsLowerBoundInclusive);
   UNUSED(minValue);
   UNUSED(maxValue);
   UNUSED(lowBoundOut);
   UNUSED(highBoundOut);

   // TODO : COMPLETE

   return IntEbmType { 1 };
}
