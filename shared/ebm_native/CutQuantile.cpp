// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // std::numeric_limits
#include <algorithm> // std::sort
#include <cmath> // std::round
#include <vector> // std::vector (used in std::priority_queue)
#include <queue> // std::priority_queue
#include <set> // std::set
#include <string.h> // strchr, memmove

#include "ebm_native.h" // EBM_API_BODY
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // LIKELY
#include "zones.h"

#include "common_cpp.hpp" // IsConvertError

#include "RandomDeterministic.hpp"

// TODO: check this file for how we handle subnormal numbers.  NEVER RETURN SUBNORMALS!

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

//#define LOG_SUPERVERBOSE_DISCRETIZATION_ORDERED
//#define LOG_SUPERVERBOSE_DISCRETIZATION_UNORDERED

// TODO: increase the k_cutExploreDistance, and also increase our testing length to compensate
static constexpr size_t k_cutExploreDistance = 20;
// 1073741824 is 2^30.  Using a power of two with no detail in the mantissa might help multiplication
static constexpr double tweakIncrement = std::numeric_limits<double>::epsilon() * double { 1073741824 };

// Some general definitions:
//  - uncuttable range - a long contiguous series of feature values after sorting that have the same value, 
//    and are therefore not separable by binning.  In order for us to consider the range uncuttable, the number of
//    identical values in the range needs to be longer than the average number of values in a bin.  Example: if
//    we are given 15 bins max, and we have 150 values, then an uncuttable range needs to be 10 values at minimum
//  - CuttingRange - a contiguous series of values after sorting that we can attempt to find Cuts within
//    because there are no long series of uncuttable values within the CuttingRange.
//  - CutPoint - the places where we cut one bin to annother
//  - cutPoint - the value we assign to a CutPoint that separates one bin from annother.  Example:
//    if we had the values [1, 2, 3, 4] and one CutPoint, a reasonable cutPoint would be 2.5.
//  - cut range - the values between two CutPoint

extern double ArithmeticMean(
   const double low,
   const double high
) noexcept;

extern double GetInterpretableCutPointFloat(
   double low,
   double high
) noexcept;

extern double GetInterpretableEndpoint(
   const double center,
   const double movementFromEnds
) noexcept;

extern size_t RemoveMissingValsAndReplaceInfinities(const size_t cSamples, double * const aVals) noexcept;

INLINE_ALWAYS constexpr static double GetTweakingMultiplePositive(const size_t iTweak) noexcept {
   return double { 1 } + tweakIncrement * static_cast<double>(iTweak);
}
INLINE_ALWAYS constexpr static double GetTweakingMultipleNegative(const size_t iTweak) noexcept {
   return double { 1 } - tweakIncrement * static_cast<double>(iTweak);
}

static constexpr ptrdiff_t k_movementDoneCut = std::numeric_limits<ptrdiff_t>::lowest();
static constexpr double k_priorityNoCutsPossible = std::numeric_limits<double>::lowest();
static constexpr size_t k_valNotLegal = std::numeric_limits<size_t>::max();

static constexpr double k_illegalAvgCuttableRangeWidthAfterAddingOneCut = std::numeric_limits<double>::lowest();

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
   double   m_iValAspirationalFloat;

   size_t         m_iVal;
   // m_cPredeterminedMovementOnCut is a valid number until we cut it.  After cutting we don't 
   // need a movement value, so we set it to k_cutValue and use it to detect whether this CutPoint was cut
   ptrdiff_t      m_cPredeterminedMovementOnCut;

   // the higher the m_priority, the more likely it is that it'll be chosen to cut
   double   m_priority;

   // the higher the m_uniqueTiebreaker, the more likely it is that it'll be chosen to cut (after considering priority)
   // the tiebreakers are ordered with symmetry in mind such that items are ranked first by distance to the end
   // points and secondly by a random number generator.  The randomness only comes into play to break ties when
   // comparing two Cuts that have the same distance to their endpoints
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
   // on our minSamplesBin value.
   // Example: If minSamplesBin == 3 and the avg bin size is 5, and the list is 
   // (1, 2, 3, 3, 3, 3, 3 | 4, 5, 6 | 7, 7, 7, 7, 7, 8, 9) -> then the only cuttable range is (4, 5, 6)

   CuttingRange() = default; // preserve our POD status
   ~CuttingRange() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   // TODO: m_cUncuttableHighVals is redundant in that the next higher cutting range has the same value.
   //       we can eliminate it here and allocate an extra "offset(m_cUncuttableHighVals) + sizeof(size_t)"
   // in the array for the extra one at the top
   size_t         m_cUncuttableHighVals;
   size_t         m_cUncuttableLowVals;

   // this can be zero if we're sandwitched between two uncuttable ranges, eg: 0, 0, 0, <CuttingRange here> 1, 1, 1
   size_t         m_cCuttableVals;
   double * m_pCuttableValsFirst;

   size_t         m_uniqueTiebreaker;

   size_t         m_cRangesAssigned;

   double   m_avgCuttableRangeWidthAfterAddingOneCut;
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
      // NEVER check for exact equality (as a precondition is ok), since then we'd violate the weak ordering rule
      // https://medium.com/@shiansu/strict-weak-ordering-and-the-c-stl-f7dcfa4d4e07

      if(UNLIKELY(rhs->m_avgCuttableRangeWidthAfterAddingOneCut == lhs->m_avgCuttableRangeWidthAfterAddingOneCut)) {
         return UNPREDICTABLE(rhs->m_uniqueTiebreaker < lhs->m_uniqueTiebreaker);
      } else {
         return UNPREDICTABLE(rhs->m_avgCuttableRangeWidthAfterAddingOneCut < lhs->m_avgCuttableRangeWidthAfterAddingOneCut);
      }
   }
};

class CompareCutPoint final {
public:
   INLINE_ALWAYS bool operator() (const CutPoint * const & lhs, const CutPoint * const & rhs) const noexcept {
      // NEVER check for exact equality (as a precondition is ok), since then we'd violate the weak ordering rule
      // https://medium.com/@shiansu/strict-weak-ordering-and-the-c-stl-f7dcfa4d4e07

      if(UNLIKELY(rhs->m_priority == lhs->m_priority)) {
         return UNPREDICTABLE(rhs->m_uniqueTiebreaker < lhs->m_uniqueTiebreaker);
      } else {
         return UNPREDICTABLE(rhs->m_priority < lhs->m_priority);
      }
   }
};

INLINE_RELEASE_UNTEMPLATED static size_t CalculateRangesMaximizeMin(
   const double sideDistance, 
   const double totalDistance, 
   const size_t cRanges,
   const size_t cRangesSideOriginal
) noexcept {
   // our goal is to, as much as possible, avoid having small ranges at the end.  We don't care as much
   // about having long ranges so much as small range since small ranges allow the boosting algorithm to overfit
   // more easily.

   EBM_ASSERT(2 <= cRanges); // we require there to be at least one range on the left and one range on the right
   EBM_ASSERT(0 <= sideDistance);
   EBM_ASSERT(sideDistance <= totalDistance);
   // This shouldn't be able to overflow even if we're on a 128 bit computer
   //
   // even with numeric instability, we shouldn't end up with a terrible result here since we only get numeric
   // issues if the number of ranges is huge, and we clip on both the low and high ranges below to handle issues
   // where rounding pushes us a bit over the numeric limits
   const size_t cRangesPlusOne = cRanges + size_t { 1 };
   const double result = static_cast<double>(cRangesPlusOne) * sideDistance / totalDistance;
   size_t cSide = static_cast<size_t>(result);
   cSide = std::max(size_t { 1 }, cSide); // don't allow zero ranges on the low side
   cSide = std::min(cSide, cRanges - 1); // don't allow zero ranges on the high side

#ifndef NDEBUG

   const double avg = std::min(sideDistance / cSide, (totalDistance - sideDistance) / (cRanges - cSide));
   if(2 <= cSide) {
      const size_t denominator1 = cSide - size_t { 1 };
      const size_t denominator2 = cRanges - cSide + size_t { 1 };
      const double avgOther = std::min(sideDistance / static_cast<double>(denominator1), (totalDistance - sideDistance) / static_cast<double>(denominator2));
      EBM_ASSERT(avgOther <= avg * 1.00001);
   }

   if(2 <= cRanges - cSide) {
      const size_t denominator1 = cSide + size_t { 1 };
      const size_t denominator2 = cRanges - cSide - size_t { 1 };
      const double avgOther = std::min(sideDistance / static_cast<double>(denominator1), (totalDistance - sideDistance) / static_cast<double>(denominator2));
      EBM_ASSERT(avgOther <= avg * 1.00001);
   }

#endif

   if(UNLIKELY(cSide != cRangesSideOriginal)) {
      // sometimes, "cRangesPlusOne * sideDistance == totalDistance" and when that happens we can get a situation
      // where symmetry breaks down as we round up when the numbers are in one orientation and round down (since
      // they are reversed) in the opposite direction.  By adding a slight bias towards keeping the original 
      // number of ranges we can avoid divergence on exact matches

      // move slighly in the direction towards zero to put us off exact integers and move us in a helpful direction
      const double multiple = UNPREDICTABLE(cSide < cRangesSideOriginal) ? 
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

static double CalculatePriority(
   const double iValLowerFloat,
   const double iValHigherFloat,
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
   double priority = k_priorityNoCutsPossible;
   if(LIKELY(k_valNotLegal != pCutCur->m_iVal)) {
      // TODO: This calculation doesn't take into account that we can trade our cut points with neighbours
      // with m_cPredeterminedMovementOnCut.  For an example, see test:
      // CutQuantile, left+uncuttable+cuttable+uncuttable+cuttable
      // I'm not sure if this is bad or not.  In general, if we're swapping cut points, we're probably moving
      // pretty far, but I think if we're swaping cut points then we probably do in fact want to add priority
      // to those potential cut points since they are shuffling cut points around and we want to ensure that this
      // can still happen.  We migth even want to increase the priority of such even by first sorting on the
      // absolute value of m_cPredeterminedMovementOnCut, then by the priority.

      // TODO : these are not guaranteed due to floating point inexactness.  We should detect this scenario
      //        For now, we don't need to worry about violations of these.  It would take truely huge datasets
      //        to reach the big number required where changes in the floating point numbers exceeded integers
      EBM_ASSERT(iValLowerFloat < pCutCur->m_iVal); // it would violate cSamplesBinMin if these were equal
      EBM_ASSERT(iValLowerFloat < pCutCur->m_iValAspirationalFloat);
      EBM_ASSERT(pCutCur->m_iVal < iValHigherFloat); // it would violate cSamplesBinMin if these were equal
      EBM_ASSERT(pCutCur->m_iValAspirationalFloat < iValHigherFloat);

      // this metric considers proportional movement to be on the equality boundary.  So, if we've moved from
      // an aspirational value of 10 down to 5, that's equivalent in priority to a movement from 10 to 20.
      // the other option which might be considered is to measure the absolute movement, so movement from
      // 10 to 5 would be the same as movement from 10 to 15, but compression or expansion by 50% in either direction
      // is probably the right way to think about it since compressing small ranges is more damaging, and this metric
      // values movement towards the smaller end more.
      double priorityLow;
      double priorityHigh;
      if(pCutCur->m_iVal < pCutCur->m_iValAspirationalFloat) {
         priorityLow = (pCutCur->m_iValAspirationalFloat - iValLowerFloat) / (pCutCur->m_iVal - iValLowerFloat);
         priorityHigh = (iValHigherFloat - pCutCur->m_iVal) / (iValHigherFloat - pCutCur->m_iValAspirationalFloat);
      } else {
         priorityLow = (pCutCur->m_iVal - iValLowerFloat) / (pCutCur->m_iValAspirationalFloat - iValLowerFloat);
         priorityHigh = (iValHigherFloat - pCutCur->m_iValAspirationalFloat) / (iValHigherFloat - pCutCur->m_iVal);
      }

      // TODO : these are not guaranteed due to floating point inexactness.  We should detect this scenario
      EBM_ASSERT(double { 1 } <= priorityLow);
      EBM_ASSERT(double { 1 } <= priorityHigh);

      // TODO: evaluate max here instead as well

      // We could alternatively take the max, but multiplying these takes both sides into account in a nice way.
      // This does have the unfortunate effect of weighing the center cuts a bit higher than if we took the max, but 
      // it also has the nice property that it adds more information into the decision and therefore should have 
      // less close tiebreaker decisions
      //priority = std::max(priorityLow, priorityHigh);

      priority = priorityLow * priorityHigh;
      EBM_ASSERT(double { 1 } <= priority);

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
   LOG_N(Trace_Verbose, "Prioritized CutPoint: %zu, %zu, %le, %td, %le",
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
   const size_t cSamplesBinMin,
   const size_t iValStart,
   const size_t cCuttableItems,
   const NeighbourJump * const aNeighbourJumps,

   const size_t cRangesLow,
   const size_t iValLow,
   const double iValAspirationalLowFloat,

   const size_t cRangesHigh,
   const size_t iValHigh,
   const double iValAspirationalHighFloat,

   // m_iValAspirationalFloat and m_uniqueTiebreaker are the only values in pCurCut that are pre-initialized
   CutPoint * const pCutCur
) noexcept {

   EBM_ASSERT(1 <= cSamplesBinMin);
   EBM_ASSERT(2 <= cCuttableItems); // this is the min if cSamplesBinMin is 1 (the min for cSamplesBinMin)
   EBM_ASSERT(2 * cSamplesBinMin <= cCuttableItems);
   EBM_ASSERT(nullptr != aNeighbourJumps);

   EBM_ASSERT(1 <= cRangesLow);
   EBM_ASSERT(1 <= cRangesHigh);

   EBM_ASSERT(k_valNotLegal == iValLow || (iValAspirationalLowFloat * double { 0.9999 } <=
      static_cast<double>(iValLow) && static_cast<double>(iValLow) <=
      iValAspirationalLowFloat * double { 1.0001 }));

   EBM_ASSERT(k_valNotLegal == iValHigh || (iValAspirationalHighFloat * double { 0.9999 } <=
      static_cast<double>(iValHigh) && static_cast<double>(iValHigh) <=
      iValAspirationalHighFloat * double { 1.0001 }));

   EBM_ASSERT(iValAspirationalLowFloat < iValAspirationalHighFloat * double { 1.0001 });

   EBM_ASSERT(nullptr != pCutCur);

   // normally m_iValAspirationalFloat shouldn't get much smaller than cSamplesBinMin, although we don't
   // prevent our aspirational cuts from breaking the cSamplesBinMin barrier since the ultimate cut might
   // end up on the far side.  There's a huge gulf though from starting at cSamplesBinMin to the minimum
   // floating point, so much so that it should never get to zero
   EBM_ASSERT(double { 0 } < pCutCur->m_iValAspirationalFloat);
   EBM_ASSERT(pCutCur->m_iValAspirationalFloat <= static_cast<double>(cCuttableItems) * double { 1.0001 });

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
   //         width, and get the square distance between the ideal cut points and their nearest real 
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
   const double smallTweak = bLocalSymmetryReversal ? GetTweakingMultiplePositive(1) : GetTweakingMultipleNegative(1);

   size_t iValAspirationalCur = static_cast<size_t>(smallTweak * pCutCur->m_iValAspirationalFloat);
   if(UNLIKELY(cCuttableItems <= iValAspirationalCur)) {
      // handle the very very unlikely situation where m_iAspirationalFloat rounds up to 
      // cCuttableItems due to floating point issues
      iValAspirationalCur = cCuttableItems - 1;
   }

   const NeighbourJump * const pNeighbourJump = &aNeighbourJumps[iValStart + iValAspirationalCur];

   const size_t iStartCur = pNeighbourJump->m_iStartCur;
   const size_t iStartNext = pNeighbourJump->m_iStartNext;

   EBM_ASSERT(iStartCur < iStartNext);
   EBM_ASSERT(iValStart <= iStartCur); // since iValAspirationalCur can't be negative
   EBM_ASSERT(iValStart <= iStartNext); // since iValAspirationalCur can't be negative

   // it shouldn't be possible to have iValAspirationalCur even close to zero, since normally the lowest value
   // would be 1, and we have a lot of resultion in floating point numbers near zero, and we always calculate
   // these values starting from the low value and adding up (since there's more resolution in low numbers)
   // on the upper end though there are failure cases where if we had sufficiently huge numbers we might
   // find that we got back a Neighbour Jump above our legal range due to floating point inexactness when rounding
   // up.  We check for this condition below though

   const ptrdiff_t iValLowChoice = 
      static_cast<ptrdiff_t>(iStartCur) - static_cast<ptrdiff_t>(iValStart);
   const ptrdiff_t iValHighChoice = 
      static_cast<ptrdiff_t>(iStartNext) - static_cast<ptrdiff_t>(iValStart);

   double totalDistance;
   double distanceLowLowFloat;
   double distanceHighLowFloat;
   bool bCanCutLow;
   bool bCanCutHigh;

   const ptrdiff_t lowHighBound = iValLowChoice + static_cast<ptrdiff_t>(cSamplesBinMin);
   const ptrdiff_t highHighBound = iValHighChoice + static_cast<ptrdiff_t>(cSamplesBinMin);
   if(UNLIKELY(k_valNotLegal == iValLow)) {
      // we always start from the low index because for floating points the low numbers have more resolution
      totalDistance = iValAspirationalHighFloat - iValAspirationalLowFloat;
      distanceLowLowFloat = static_cast<double>(iValLowChoice) - iValAspirationalLowFloat;
      distanceHighLowFloat = static_cast<double>(iValHighChoice) - iValAspirationalLowFloat;

      const ptrdiff_t lowLowBoundPtrdiff = iValLowChoice - static_cast<ptrdiff_t>(cSamplesBinMin);
      const double lowLowBoundFloat = static_cast<double>(lowLowBoundPtrdiff);
      const ptrdiff_t highLowBoundPtrdiff = iValHighChoice - static_cast<ptrdiff_t>(cSamplesBinMin);
      const double highLowBoundFloat = static_cast<double>(highLowBoundPtrdiff);
      if(UNLIKELY(k_valNotLegal == iValHigh)) {
         // given all our indexes and counts refer to an existing array with more than 4 bytes, 
         // they should not be able to overflow when adding any of these numbers

         // since we always start from the low end, I don't think we can ever get a number less than zero, which
         // is a preceise floating point value also.
         EBM_ASSERT(double { 0 } <= iValAspirationalLowFloat);

         // check our soft bounds and hard bounds (to avoid floating point issues)
         bCanCutLow = LIKELY(LIKELY(iValAspirationalLowFloat <= lowLowBoundFloat) &&
            LIKELY(static_cast<double>(lowHighBound) <= iValAspirationalHighFloat) &&
            LIKELY(lowHighBound <= static_cast<ptrdiff_t>(cCuttableItems)));

         bCanCutHigh = LIKELY(LIKELY(static_cast<double>(highHighBound) <= iValAspirationalHighFloat) &&
            LIKELY(iValAspirationalLowFloat <= highLowBoundFloat) &&
            LIKELY(highHighBound <= static_cast<ptrdiff_t>(cCuttableItems)));
      } else {
         // given all our indexes and counts refer to an existing array with more than 4 bytes, 
         // they should not be able to overflow when adding any of these numbers

         EBM_ASSERT(iValHigh <= cCuttableItems);
         // since we always start from the low end, I don't think we can ever get a number less than zero, which
         // is a preceise floating point value also.
         EBM_ASSERT(double { 0 } <= iValAspirationalLowFloat);

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
      distanceLowLowFloat = static_cast<double>(distanceLowPtrdiffT);
      distanceHighLowFloat = static_cast<double>(distanceHighPtrdiffT);
      if(UNLIKELY(k_valNotLegal == iValHigh)) {
         totalDistance = iValAspirationalHighFloat - iValAspirationalLowFloat;

         // given all our indexes and counts refer to an existing array with more than 4 bytes, 
         // they should not be able to overflow when adding any of these numbers

         // check our soft bounds and hard bounds (to avoid floating point issues)
         bCanCutLow = LIKELY(LIKELY(static_cast<ptrdiff_t>(cSamplesBinMin) <= distanceLowPtrdiffT) &&
            LIKELY(static_cast<double>(lowHighBound) <= iValAspirationalHighFloat) &&
            LIKELY(lowHighBound <= static_cast<ptrdiff_t>(cCuttableItems)));

         bCanCutHigh = LIKELY(LIKELY(static_cast<double>(highHighBound) <= iValAspirationalHighFloat) &&
            LIKELY(static_cast<ptrdiff_t>(cSamplesBinMin) <= distanceHighPtrdiffT) &&
            LIKELY(highHighBound <= static_cast<ptrdiff_t>(cCuttableItems)));
      } else {
         // reduce floating point noise when we have have exact distances
         totalDistance = static_cast<double>(iValHigh - iValLow);

         // given all our indexes and counts refer to an existing array with more than 4 bytes, 
         // they should not be able to overflow when adding any of these numbers

         EBM_ASSERT(iValHigh <= cCuttableItems);

         bCanCutLow = LIKELY(LIKELY(static_cast<ptrdiff_t>(cSamplesBinMin) <= distanceLowPtrdiffT) &&
            LIKELY(lowHighBound <= static_cast<ptrdiff_t>(iValHigh)));

         bCanCutHigh = LIKELY(LIKELY(highHighBound <= static_cast<ptrdiff_t>(iValHigh)) &&
            LIKELY(static_cast<ptrdiff_t>(cSamplesBinMin) <= distanceHighPtrdiffT));
      }
   }

   static constexpr double k_badScore = std::numeric_limits<double>::lowest();

   double scoreHigh;
   ptrdiff_t transferRangesHigh;

   if(LIKELY(bCanCutHigh)) {
      {
         const size_t cRangesHighLow = CalculateRangesMaximizeMin(distanceHighLowFloat, totalDistance, cRanges, cRangesLow);
         EBM_ASSERT(1 <= cRangesHighLow);
         EBM_ASSERT(cRangesHighLow < cRanges);
         const size_t cRangesHighHigh = cRanges - cRangesHighLow;
         EBM_ASSERT(1 <= cRangesHighHigh);

         double distanceHigh;
         if(UNLIKELY(k_valNotLegal == iValHigh)) {
            distanceHigh = iValAspirationalHighFloat - static_cast<double>(iValHighChoice);
         } else {
            const ptrdiff_t distanceHighPtrdiff = static_cast<ptrdiff_t>(iValHigh) - iValHighChoice;
            distanceHigh = static_cast<double>(distanceHighPtrdiff);
         }
         const double avgLengthHighHigh = distanceHigh / cRangesHighHigh;
         const double avgLengthHighLow = distanceHighLowFloat / cRangesHighLow;

         scoreHigh = std::min(avgLengthHighLow, avgLengthHighHigh);
         transferRangesHigh = static_cast<ptrdiff_t>(cRangesHighLow) - static_cast<ptrdiff_t>(cRangesLow);
      }

      double scoreLow;
      ptrdiff_t transferRangesLow;

      if(LIKELY(bCanCutLow)) {

      do_low:;

         const size_t cRangesLowLow = CalculateRangesMaximizeMin(distanceLowLowFloat, totalDistance, cRanges, cRangesLow);
         EBM_ASSERT(1 <= cRangesLowLow);
         EBM_ASSERT(cRangesLowLow < cRanges);
         const size_t cRangesLowHigh = cRanges - cRangesLowLow;
         EBM_ASSERT(1 <= cRangesLowHigh);

         double distanceHigh;
         if(UNLIKELY(k_valNotLegal == iValHigh)) {
            distanceHigh = iValAspirationalHighFloat - static_cast<double>(iValLowChoice);
         } else {
            const ptrdiff_t distanceHighPtrdiff = static_cast<ptrdiff_t>(iValHigh) - iValLowChoice;
            distanceHigh = static_cast<double>(distanceHighPtrdiff);
         }
         const double avgLengthLowHigh = distanceHigh / cRangesLowHigh;
         const double avgLengthLowLow = distanceLowLowFloat / cRangesLowLow;

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
      LOG_N(Trace_Verbose, "Plan CutPoint: %zu, %zu, %le, %td, %le" ", %le",
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
      LOG_0(Trace_Verbose, "Plan CutPoint: DENIED");
#endif // LOG_SUPERVERBOSE_DISCRETIZATION_UNORDERED
   }
   EBM_ASSERT(!pCutCur->IsCut());
}

static ErrorEbm CutCuttingRange(
   std::set<CutPoint *, CompareCutPoint> * const pBestCuts,

   const size_t cSamples,
   const bool bSymmetryReversal,
   const size_t cSamplesBinMin,

   const size_t iValStart,
   const size_t cCuttableItems,
   const NeighbourJump * const aNeighbourJumps
) noexcept {
   EBM_ASSERT(nullptr != pBestCuts);

   EBM_ASSERT(2 <= cSamples); // we wouldn't be cutting this if there weren't two potential bins

   EBM_ASSERT(1 <= cSamplesBinMin);

   // we need to be able to put down at least one cut not at the edges
   EBM_ASSERT(2 <= cCuttableItems); // we wouldn't be cutting this if there weren't two potential bins
   EBM_ASSERT(2 <= cCuttableItems / cSamplesBinMin);
   EBM_ASSERT(cCuttableItems <= cSamples);
   EBM_ASSERT(nullptr != aNeighbourJumps);

   // TODO: someday, for performance, it might make sense to use a non-allocating tree, like:
   //       https://github.com/attractivechaos/klib/blob/master/kavl.h

   try {
      while(!pBestCuts->empty()) {
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
         CutPoint * const pCutBest = *pBestCuts->begin();

#ifdef LOG_SUPERVERBOSE_DISCRETIZATION_ORDERED
         LOG_N(Trace_Verbose, "Dequeue CutPoint: %zu, %zu, %le, %td, %le",
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
         EBM_ASSERT(double { 0 } < pCutBest->m_iValAspirationalFloat);

         EBM_ASSERT(!std::isnan(pCutBest->m_priority));
         EBM_ASSERT(!std::isinf(pCutBest->m_priority));

         const size_t iVal = pCutBest->m_iVal; // preserve the location of our cut in case we end up moving
         if(k_valNotLegal == iVal) {
            // k_valNoCutsPossible means there are no legal cuts, and also that all the remaining items 
            // in the queue are also uncuttable, so exit.
            EBM_ASSERT(k_priorityNoCutsPossible == pCutBest->m_priority);
            break;
         }
         EBM_ASSERT(double { 0 } <= pCutBest->m_priority);

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

         EBM_ASSERT(double { 0 } <= pCutLowModificationExclusiveBoundary->m_iValAspirationalFloat);

         // this should be exact, since we would have set it like this
         EBM_ASSERT(!pCutLowModificationExclusiveBoundary->IsCut() || pCutLowModificationExclusiveBoundary->m_iValAspirationalFloat == static_cast<double>(pCutLowModificationExclusiveBoundary->m_iVal));
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

         EBM_ASSERT(double { 0 } < pCutHighModificationExclusiveBoundary->m_iValAspirationalFloat);

         // this should be exact, since we would have set it like this
         EBM_ASSERT(!pCutHighModificationExclusiveBoundary->IsCut() || pCutHighModificationExclusiveBoundary->m_iValAspirationalFloat == static_cast<double>(pCutHighModificationExclusiveBoundary->m_iVal));
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

         const double iValFloat = static_cast<double>(iVal);

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
         //     materialized cut and jump by cSamplesBinMin using aNeighbourJumps until we hit the aspirational
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
         //     with cSamplesBinMin as we slide the visiblility windows to the left or right
         //
         // We only need to know if we have more than k_cutExploreDistance items, since we can't have more than
         // that number of cuts until our border.  We can allocate a fixed size array on the stack with 
         // k_cutExploreDistance size_t indexes and fill these from the lower and higher sides, and then if we
         // cross any of those indexes we know if we've gone below a limit.


         double stepPoint = pCutLowModificationExclusiveBoundary->m_iValAspirationalFloat;
         double stepLength;
         if(pCutLowModificationExclusiveBoundary->IsCut()) {
            EBM_ASSERT(pCutLowModificationExclusiveBoundary->m_iVal < iVal);
            stepLength = static_cast<double>(iVal - pCutLowModificationExclusiveBoundary->m_iVal);
         } else {
            EBM_ASSERT(stepPoint < iValFloat);
            stepLength = iValFloat - stepPoint;
         }
         stepLength /= static_cast<double>(cRangesLowModification);

         CutPoint * pCutAspirational = pCutCur;
         while(LIKELY(size_t { 0 } != --cRangesLowModification)) {
            pCutAspirational = pCutAspirational->m_pPrev;
            const double iValAspirationalFloat = stepPoint + 
               stepLength * static_cast<double>(cRangesLowModification);
            pCutAspirational->m_iValAspirationalFloat = iValAspirationalFloat;
         }

         stepPoint = iValFloat;
         if(pCutHighModificationExclusiveBoundary->IsCut()) {
            EBM_ASSERT(iVal < pCutHighModificationExclusiveBoundary->m_iVal);
            stepLength = static_cast<double>(pCutHighModificationExclusiveBoundary->m_iVal - iVal);
         } else {
            EBM_ASSERT(iValFloat < pCutHighModificationExclusiveBoundary->m_iValAspirationalFloat);
            stepLength = pCutHighModificationExclusiveBoundary->m_iValAspirationalFloat - stepPoint;
         }
         stepLength /= static_cast<double>(cRangesHighModification);

         pCutAspirational = pCutHighModificationExclusiveBoundary;
         while(size_t { 0 } != --cRangesHighModification) {
            pCutAspirational = pCutAspirational->m_pPrev;
            const double iValAspirationalFloat = stepPoint + 
               stepLength * static_cast<double>(cRangesHighModification);
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

               cSamplesBinMin,
               iValStart,
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

               cSamplesBinMin,
               iValStart,
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

         EBM_ASSERT(pBestCuts->end() != pBestCuts->find(pCutCur));
         pBestCuts->erase(pCutCur);

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

                  double debugPriority = CalculatePriority(
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

            EBM_ASSERT(pBestCuts->end() != pBestCuts->find(pCutLowPriorityCur));
            pBestCuts->erase(pCutLowPriorityCur);

            pCutLowPriorityCur->m_priority = CalculatePriority(
               pCutLowLowPriorityInclusiveBoundary->m_iValAspirationalFloat,
               pCutLowHighPriorityInclusiveBoundary->m_iValAspirationalFloat,
               pCutLowPriorityCur
            );

            pBestCuts->insert(pCutLowPriorityCur);
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

                  double debugPriority = CalculatePriority(
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

            EBM_ASSERT(pBestCuts->end() != pBestCuts->find(pCutHighPriorityCur));
            pBestCuts->erase(pCutHighPriorityCur);

            pCutHighPriorityCur->m_priority = CalculatePriority(
               pCutHighLowPriorityInclusiveBoundary->m_iValAspirationalFloat,
               pCutHighHighPriorityInclusiveBoundary->m_iValAspirationalFloat,
               pCutHighPriorityCur
            );

            pBestCuts->insert(pCutHighPriorityCur);
         }
      }
   } catch(const std::bad_alloc &) {
      LOG_0(Trace_Warning, "WARNING CutSegment out of memory");
      return Error_OutOfMemory;
   } catch(...) {
      LOG_0(Trace_Warning, "WARNING CutSegment exception");
      return Error_UnexpectedInternal;
   }

   IronCuts();

   return Error_None;
}

static ErrorEbm TreeSearchCutSegment(
   std::set<CutPoint *, CompareCutPoint> * pBestCuts,

   const size_t cSamples,
   const bool bSymmetryReversal,
   const size_t cSamplesBinMin,

   const size_t iValStart,
   const size_t cCuttableItems,
   const NeighbourJump * const aNeighbourJumps,

   const size_t cRanges,
   // for efficiency we include space for the end point cuts even if they don't exist
   CutPoint * const aCutsWithENDPOINTS
) noexcept {
   try {
      EBM_ASSERT(nullptr != pBestCuts);
      EBM_ASSERT(pBestCuts->empty());

      EBM_ASSERT(2 <= cSamples); // we need at least 2 to cut, otherwise we'd have exited before calling here
      EBM_ASSERT(1 <= cSamplesBinMin);

      EBM_ASSERT(nullptr != aNeighbourJumps);

      EBM_ASSERT(2 <= cRanges);
      EBM_ASSERT(cSamplesBinMin <= cCuttableItems / cRanges);
      EBM_ASSERT(nullptr != aCutsWithENDPOINTS);

      // - TODO: EXPLORING BOTH SIDES
      //   - this function calls CutSegment, which greedily materializes cuts, so when it's unsure about a cut
      //     it needs to be conservative and pick the least likley cut to cause problems down the road
      //   - at this higher level, we can try cutting both low AND high AND skip the cut.  We use CutCuttingRange to
      //     do the full exploration of both options and then we pick the better one.
      //   - we can also explore N steps in the future to pick the best first step, then delete the worst 1st step
      //     and keep all the work we did along the choice that we made (the remaining 128 options) then we can pick
      //     the best step from all those 128 options and continue this way.  Since we do a complete recalculation
      //     of all the Cuts we can only do this several times, but it allows us to have 2 levels of fallback

      //   - we can design an algorithm that divides into 255 and chooses the worst one and then does a complete fit on either direction.Best fit is recorded
      //     then we re-do all 254 other cuts on BOTH sides.  We can only do a set number of these, so after 8 levels we'd have 256 attempts.  That might be acceptable
      //   - the algorithm that we have below plays it safe since it needs to live with it's decions.  This more spectlative algorithm above can be more
      //     risky since it plays both directions a bad play won't undermine it.  As such, we should try and chose the worst decion without regard to position
      //     so in other words, try to choose the range that we have a drop point in in the middle where we need to move the most to get away from the 
      //     best drops.  We can also try going left, going right, OR not choosing.  Don't traverse down the NO choice path, so we add 50% load, but we don't grow at 3^N, and we'll 
      //     also explore the no choice at the root option
      //

      //static constexpr size_t k_CutExploreDepth = 8;
      //static constexpr size_t k_CutExplorations = size_t { 1 } << k_CutExploreDepth;

      CutPoint * pCutCur = &aCutsWithENDPOINTS[0];
      CutPoint * pCutNext = &aCutsWithENDPOINTS[1];

      pCutCur->m_pNext = pCutNext;
      pCutCur->SetCut();
      pCutCur->m_iValAspirationalFloat = double { 0 };
      pCutCur->m_iVal = size_t { 0 };

      const double stepInit = static_cast<double>(cCuttableItems) / static_cast<double>(cRanges);
      EBM_ASSERT(cSamplesBinMin <= 1.00001 * stepInit);

      const double cCuttableItemsFloat = static_cast<double>(cCuttableItems);
      size_t iCutCur = 1;
      size_t iValLow = size_t { 0 };
      double iValAspirationalLowFloat = double { 0 };
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
            EBM_ASSERT(double { 0 } == iValAspirationalLowFloat);
         } else {
            cRangesLow = k_cutExploreDistance;
            iValLow = k_valNotLegal;
            iValAspirationalLowFloat = stepInit * static_cast<double>(static_cast<size_t>(iRangeLow));
         }

         double iValAspirationalHighFloat;
         size_t iRangeHigh = iCutCur + k_cutExploreDistance;
         if(UNLIKELY(cRanges <= iRangeHigh)) {
            cRangesHigh = cRanges - iCutCur;
            iValHigh = cCuttableItems;
            iValAspirationalHighFloat = cCuttableItemsFloat;
         } else {
            EBM_ASSERT(k_cutExploreDistance == cRangesHigh);
            EBM_ASSERT(k_valNotLegal == iValHigh);
            iValAspirationalHighFloat = stepInit * static_cast<double>(iRangeHigh);
         }

         const double iValAspirationalCurFloat = stepInit * static_cast<double>(iCutCur);
         pCutCur->m_iValAspirationalFloat = iValAspirationalCurFloat;

         EBM_ASSERT(pCutCur->m_uniqueTiebreaker < cRanges);

         BuildNeighbourhoodPlan(
            cSamples,
            bSymmetryReversal,
            cSamplesBinMin,
            iValStart,
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
         // in the future we might write code above that removes Cuts, which if it were true could mean no legal cuts
         EBM_ASSERT(nullptr != pCutCenter->m_pNext);
         EBM_ASSERT(pCutLow < pCutCenter);
         EBM_ASSERT(pCutCenter < pCutHigh);

         pCutCenter->m_priority = CalculatePriority(
            pCutLow->m_iValAspirationalFloat,
            pCutHigh->m_iValAspirationalFloat,
            pCutCenter
         );

         EBM_ASSERT(!pCutCenter->IsCut());
         pBestCuts->insert(pCutCenter);

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
   } catch(const std::bad_alloc &) {
      LOG_0(Trace_Warning, "WARNING TreeSearchCutSegment out of memory exception");
      return Error_OutOfMemory;
   } catch(...) {
      LOG_0(Trace_Warning, "WARNING TreeSearchCutSegment exception");
      return Error_UnexpectedInternal;
   }

   return CutCuttingRange(
      pBestCuts,
      cSamples,
      bSymmetryReversal,
      cSamplesBinMin,
      iValStart,
      cCuttableItems,
      aNeighbourJumps
   );
}

INLINE_RELEASE_UNTEMPLATED static ErrorEbm TradeCutSegment(
   std::set<CutPoint *, CompareCutPoint> * const pBestCuts,

   const size_t cSamples,
   const bool bSymmetryReversal,
   const size_t cSamplesBinMin,

   const size_t iValStart,
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
      pBestCuts, 
      cSamples,
      bSymmetryReversal,
      cSamplesBinMin,
      iValStart, 
      cCuttableItems, 
      aNeighbourJumps,
      cRanges, 
      aCutsWithENDPOINTS
   );
}

INLINE_RELEASE_UNTEMPLATED static size_t DetermineRangesMax(
   const size_t cSamplesInSubset,
   const double * const aVals,
   const size_t cSamplesBinMin
) noexcept {
   EBM_ASSERT(1 <= cSamplesInSubset);
   EBM_ASSERT(nullptr != aVals);
   EBM_ASSERT(1 <= cSamplesBinMin);
   EBM_ASSERT(cSamplesBinMin <= cSamplesInSubset);

   if(size_t { 1 } == cSamplesInSubset) {
      EBM_ASSERT(size_t { 1 } == cSamplesBinMin);
      return size_t { 1 };
   }

   double valPrev = aVals[0];
   const double * pValStartRange = aVals;
   const double * pVal = aVals + 1;
   const double * const pValsEnd = aVals + cSamplesInSubset;

   size_t cRanges = 0;
   size_t cItems;
   do {
      EBM_ASSERT(pVal < pValsEnd);
      double valCur = *pVal;
      if(valCur != valPrev) {
         cItems = pVal - pValStartRange;
         if(cSamplesBinMin <= cItems) {
            ++cRanges;
            pValStartRange = pVal;
         }
         valPrev = valCur;
      }
      ++pVal;
   } while(pValsEnd != pVal);
   cItems = pVal - pValStartRange;
   if(cSamplesBinMin <= cItems) {
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
      if(aVals[iDebugCur] != aVals[iDebugStartEqual]) {
         if(cSamplesBinMin <= iDebugStartRange - iDebugCur) {
            ++cDebugRanges;
            iDebugStartRange = iDebugCur;
         }
         iDebugStartEqual = iDebugCur;
      }
   }
   if(cSamplesBinMin <= iDebugStartRange + 1) {
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

   double avgRangeWidthAfterAddingOneCut = k_illegalAvgCuttableRangeWidthAfterAddingOneCut;
   EBM_ASSERT(cRangesCur <= pCuttingRangeAdd->m_cRangesMax);
   if(LIKELY(pCuttingRangeAdd->m_cRangesMax != cRangesCur)) {
      // we need to re-add our CuttingRange back into the priority queue.  If we were assigned a new range
      // again, this is how many we'd be at
      const size_t cRangesNext = cRangesCur + size_t { 1 };
      const size_t cCuttableItems = pCuttingRangeAdd->m_cCuttableVals;

      avgRangeWidthAfterAddingOneCut =
         static_cast<double>(cCuttableItems) / static_cast<double>(cRangesNext);

      // don't muliply by GetTweakingMultiple, since avgRangeWidthAfterAddingOneCut is derrived from
      // size_t values, it should have exactly the same value when cCuttableItems and cRangesNext
      // are the same, so we should then get to compare on m_uniqueTiebreaker after seeing the exact
      // floating point equality.  Also, unlike the CutPoint priority value, we don't want to affect
      // m_avgCuttableRangeWidthAfterAddingOneCut since even distant regions shouldn't have divergent
      // priorities, unlike for Cuts
   }
   pCuttingRangeAdd->m_avgCuttableRangeWidthAfterAddingOneCut = avgRangeWidthAfterAddingOneCut;
   queue.insert(pCuttingRangeAdd);
   return false;
}

static void StuffCutsIntoCuttingRanges(
   std::set<CuttingRange *, CompareCuttingRange> & queue,
   const size_t cCuttingRanges,
   CuttingRange * const aCuttingRange,
   const size_t cSamplesBinMin,
   const size_t cCutsAssignable
) {
   EBM_ASSERT(1 <= cCuttingRanges);
   EBM_ASSERT(nullptr != aCuttingRange);
   EBM_ASSERT(1 <= cSamplesBinMin);
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
      // range, and we don't need to divide our m_cCuttableVals by 1 for the 1 range.  We just need to check
      // that it has enough samples per bin
      const size_t cCuttableItems = pCuttingRangeInit->m_cCuttableVals;
      size_t cRangesMax = 0;
      double avgRangeWidthAfterAddingOneCut = k_illegalAvgCuttableRangeWidthAfterAddingOneCut;
      if(LIKELY(cSamplesBinMin <= cCuttableItems)) {
         // don't muliply by GetTweakingMultiple, since avgRangeWidthAfterAddingOneCut is derrived from
         // size_t values, it should have exactly the same value when cCuttableItems and newProposedRanges
         // are the same, so we should then get to compare on m_uniqueTiebreaker after seeing the exact
         // floating point equality.  Also, unlike the CutPoint priority value, we don't want to affect
         // m_avgCuttableRangeWidthAfterAddingOneCut since even distant regions shouldn't have divergent
         // priorities, unlike for Cuts
         avgRangeWidthAfterAddingOneCut = static_cast<double>(cCuttableItems);

         cRangesMax = DetermineRangesMax(
            cCuttableItems, 
            pCuttingRangeInit->m_pCuttableValsFirst, 
            cSamplesBinMin
         );

         EBM_ASSERT(1 <= cRangesMax);
      }
      pCuttingRangeInit->m_avgCuttableRangeWidthAfterAddingOneCut = avgRangeWidthAfterAddingOneCut;
      pCuttingRangeInit->m_cRangesMax = cRangesMax;
      queue.insert(pCuttingRangeInit);

      ++pCuttingRangeInit;
   } while(LIKELY(pCuttingRangeEnd != pCuttingRangeInit));

   size_t cRemainingCuts = cCutsAssignable;
   if(UNLIKELY(0 == aCuttingRange[0].m_cUncuttableLowVals)) {
      // if our tail end is a pure tail with no uncuttable range on it's side, then we can get a range with just
      // one cut since the end of the values provides us an implicit cut.  If our tail end is an uncutable range,
      // then we need to put cuts on both ends to get a single range, so we don't get an implicit cut
      // add one to our remaining cuts to account for the implicit cut that we get at the start
      ++cRemainingCuts;
   }

   if(UNLIKELY(0 == (pCuttingRangeEnd - 1)->m_cUncuttableHighVals)) {
      // if our tail end is a pure tail with no uncuttable range on it's side, then we can get a range with just
      // one cut since the end of the values provides us an implicit cut.  If our tail end is an uncutable range,
      // then we need to put cuts on both ends to get a single range, so we don't get an implicit cut
      // add one to our remaining cuts to account for the implicit cut that we get at the end
      ++cRemainingCuts;
   }

   EBM_ASSERT(cCuttingRanges <= cRemainingCuts);
   cRemainingCuts -= cCuttingRanges;
   // the queue can initially be empty if all the ranges are too short to make them cSamplesBinMin
   while(LIKELY(0 != cRemainingCuts)) {
      if(AddCutToRanges(queue)) {
         break;
      }
      --cRemainingCuts;
   }
}

INLINE_RELEASE_UNTEMPLATED static void FillCuttingRangeNeighbours(
   const size_t cSamples,
   double * const aFeatureVals,
   const size_t cCuttingRanges,
   CuttingRange * const aCuttingRange
) noexcept {
   EBM_ASSERT(2 <= cSamples); // if there wern't 2 samples we couldn't have any bins and we'd exit earliers
   EBM_ASSERT(nullptr != aFeatureVals);
   EBM_ASSERT(1 <= cCuttingRanges);
   EBM_ASSERT(nullptr != aCuttingRange);

   CuttingRange * pCuttingRange = aCuttingRange;
   size_t cUncuttablePriorItems = pCuttingRange->m_pCuttableValsFirst - aFeatureVals;
   const double * const aFeatureValsEnd = aFeatureVals + cSamples;
   const size_t cCuttingRangesMinusOne = cCuttingRanges - 1;
   if(PREDICTABLE(0 != cCuttingRangesMinusOne)) {
      // exit without doing the last one
      const CuttingRange * const pCuttingRangeLast = pCuttingRange + cCuttingRangesMinusOne;
      do {
         const size_t cUncuttableSubsequentItems = (pCuttingRange + 1)->m_pCuttableValsFirst - 
            pCuttingRange->m_pCuttableValsFirst - pCuttingRange->m_cCuttableVals;

         // TODO : eliminate this function after we've eliminated m_cUncuttableHighVals and wrap this functionality
         // into FillCuttingRangeBasics?

         pCuttingRange->m_cUncuttableLowVals = cUncuttablePriorItems;
         pCuttingRange->m_cUncuttableHighVals = cUncuttableSubsequentItems;

         cUncuttablePriorItems = cUncuttableSubsequentItems;
         ++pCuttingRange;
      } while(LIKELY(pCuttingRangeLast != pCuttingRange));
   }
   const size_t cUncuttableSubsequentItems =
      aFeatureValsEnd - pCuttingRange->m_pCuttableValsFirst - pCuttingRange->m_cCuttableVals;

   pCuttingRange->m_cUncuttableLowVals = cUncuttablePriorItems;
   pCuttingRange->m_cUncuttableHighVals = cUncuttableSubsequentItems;
}

INLINE_RELEASE_UNTEMPLATED static void FillCuttingRangeBasics(
   const size_t cSamples,
   double * const aFeatureVals,
   const size_t cUncuttableRangeLengthMin,
   const size_t cSamplesBinMin,
   const size_t cCuttingRanges,
   CuttingRange * const aCuttingRange
) noexcept {
   EBM_ASSERT(2 <= cSamples); // we would have exited earlier unless there were 2 bins
   EBM_ASSERT(nullptr != aFeatureVals);
   EBM_ASSERT(1 <= cUncuttableRangeLengthMin);
   EBM_ASSERT(1 <= cSamplesBinMin);
   EBM_ASSERT(1 <= cCuttingRanges);
   EBM_ASSERT(nullptr != aCuttingRange);

   double rangeVal = *aFeatureVals;
   double * pCuttableValsStart = aFeatureVals;
   const double * pStartEqualRange = aFeatureVals;
   double * pScan = aFeatureVals + 1;
   const double * const pValsEnd = aFeatureVals + cSamples;

   CuttingRange * pCuttingRange = aCuttingRange;
   do {
      const double val = *pScan;
      if(PREDICTABLE(val != rangeVal)) {
         size_t cEqualRangeItems = pScan - pStartEqualRange;
         if(PREDICTABLE(cUncuttableRangeLengthMin <= cEqualRangeItems)) {
            if(PREDICTABLE(
               PREDICTABLE(cSamplesBinMin <= static_cast<size_t>(pStartEqualRange - pCuttableValsStart)) ||
               UNLIKELY(aFeatureVals != pCuttableValsStart))) 
            {
               EBM_ASSERT(pCuttingRange < aCuttingRange + cCuttingRanges);
               pCuttingRange->m_pCuttableValsFirst = pCuttableValsStart;
               pCuttingRange->m_cCuttableVals = pStartEqualRange - pCuttableValsStart;
               ++pCuttingRange;
            }
            pCuttableValsStart = pScan;
         }
         rangeVal = val;
         pStartEqualRange = pScan;
      }
      ++pScan;
   } while(LIKELY(pValsEnd != pScan));
   if(LIKELY(pCuttingRange != aCuttingRange + cCuttingRanges)) {
      // we're not done, so we have one more to go.. this last one
      EBM_ASSERT(pCuttingRange == aCuttingRange + cCuttingRanges - 1);
      EBM_ASSERT(pCuttableValsStart < pValsEnd);
      pCuttingRange->m_pCuttableValsFirst = pCuttableValsStart;
      EBM_ASSERT(pStartEqualRange < pValsEnd);
      const size_t cEqualRangeItems = pValsEnd - pStartEqualRange;
      const double * const pCuttableRangeEnd = cUncuttableRangeLengthMin <= cEqualRangeItems ? 
         pStartEqualRange : pValsEnd;
      pCuttingRange->m_cCuttableVals = pCuttableRangeEnd - pCuttableValsStart;
   }
}

template<typename T>
static void FillTiebreakers(
   const bool bSymmetryReversal,
   RandomDeterministic * const pRng,
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
   
   // We add some consistent/repeatable noise to our priority for cutting to combat floating point inexactnes issues.
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
   
   EBM_ASSERT(nullptr != pRng);
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
      const bool bRandom = pRng->Next<bool>() != bSymmetryReversal; // this is an XOR for bools

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
   const double * const aFeatureVals
) noexcept {
   EBM_ASSERT(size_t { 2 } <= cSamples); // if we don't have enough samples to generate 2 bins we exit earlier
   EBM_ASSERT(nullptr != aFeatureVals);

   const double * const pTop = aFeatureVals + cSamples - size_t { 1 };

   const double * pLow = aFeatureVals;
   const double * pHigh = pTop;
   double lowPrev = *pLow;
   double highPrev = *pHigh;

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
      const double lowCur = *pLow;
      const double highCur = *pHigh;

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
      pHigh = aFeatureVals + (cSamples >> 1);
      pLow = pHigh - (size_t { 1 } & (cSamples - size_t { 1 }));

      // pHigh and pLow might be aliased pointers.  If we have an odd number of values then there is one center
      const double valCenterHigh = *pHigh;
      const double valCenterLow = *pLow;

      do {
         ++pHigh;
         --pLow;

         const double valHigh = *pHigh;
         const double valLow = *pLow;

         const double distanceHigh = valHigh - valCenterHigh;
         const double distanceLow = valCenterLow - valLow;

         // our distance high and distance low might be integers, so don't use base 10 decimals which might
         // lead to collisions
         if(distanceLow < distanceHigh * double { 0.9999248297572194127975024574325 }) {
            return true;
         }
         if(distanceHigh < distanceLow * double { 0.9999248297572194127975024574325 }) {
            return false;
         }
      } while(aFeatureVals != pLow);
   }
   pLow = aFeatureVals;
   pHigh = pTop;
   do {
      EBM_ASSERT(pLow < pHigh);

      const double lowCur = std::abs(*pLow);
      const double highCur = std::abs(*pHigh);

      if(UNLIKELY(lowCur != highCur)) {
         return lowCur < highCur;
      }

      ++pLow;
      --pHigh;
      // we can exit if they are equal, since their absolute value would be equal
   } while(LIKELY(pLow < pHigh));

   // if all our values are identical, then we shouldn't have gotten any cuts, so we shouldn't have gotten here
   EBM_ASSERT(*aFeatureVals < *pTop);

   // the data is perfectly symmetric centered arround zero.  Something like "-2 -1 0 1 2" would do this.  There is
   // no way for us to tell when it's been reversed even in theory.  Let's return a consistent value of false.  
   // This value is XORed with a random value later anyways, so there's no direction bias in returning either 
   // true or false here.
   return false;
}

INLINE_RELEASE_UNTEMPLATED static void ConstructJumps(
   const size_t cSamples, 
   const double * const aVals, 
   NeighbourJump * const aNeighbourJump
) noexcept {
   EBM_ASSERT(1 <= cSamples);
   EBM_ASSERT(nullptr != aVals);
   EBM_ASSERT(nullptr != aNeighbourJump);

   double valNext = aVals[0];
   const double * pVal = aVals;
   const double * const pValsEnd = aVals + cSamples;

   size_t iStartCur = 0;
   NeighbourJump * pNeighbourJump = aNeighbourJump;
   while(true) {
      const double valCur = valNext;
      do {
         ++pVal;
         if(UNLIKELY(pValsEnd == pVal)) {
            const size_t iStartNext = pVal - aVals;
            EBM_ASSERT(iStartNext == cSamples);
            const NeighbourJump * const pNeighbourJumpEnd = aNeighbourJump + iStartNext;
            do {
               pNeighbourJump->m_iStartCur = iStartCur;
               pNeighbourJump->m_iStartNext = iStartNext;
               ++pNeighbourJump;
            } while(PREDICTABLE(pNeighbourJumpEnd != pNeighbourJump));

            EBM_ASSERT(aNeighbourJump + cSamples == pNeighbourJump);

            // The Clang static analyzer seems to not understand that ConstructJumps fully initializes the
            // aNeighbourJumps buffer.  If I put a memset(aNeighbourJumps, 0, cBytesNeighbourJumps);
            // then the warning is resolved.  Given ConstructJumps fully initializes this buffer this seems
            // to be a spurious static analysis warning.  ConstructJumps does have odd processing logic that I 
            // could see the compiler having a difficult time analyzing.
            StopClangAnalysis();

            return;
         }
         valNext = *pVal;
      } while(PREDICTABLE(valNext == valCur));

      const size_t iStartNext = pVal - aVals;
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
   const double * const aFeatureVals,
   const size_t cUncuttableRangeLengthMin,
   const size_t cSamplesBinMin
) noexcept {
   EBM_ASSERT(size_t { 2 } <= cSamples); // if we don't have enough samples to generate 2 bins we exit earlier
   EBM_ASSERT(nullptr != aFeatureVals);
   EBM_ASSERT(size_t { 1 } <= cUncuttableRangeLengthMin);
   EBM_ASSERT(size_t { 1 } <= cSamplesBinMin);
   EBM_ASSERT(cSamplesBinMin <= cSamples / size_t { 2 }); // we exit earlier if we don't have enough samples for 2 bins

   double rangeVal = *aFeatureVals;
   const double * pCuttableValsStart = aFeatureVals;
   const double * pStartEqualRange = aFeatureVals;
   const double * pScan = aFeatureVals + 1;
   const double * const pValsEnd = aFeatureVals + cSamples;
   size_t cCuttingRanges = 0;
   EBM_ASSERT(pValsEnd != pScan); // because 2 <= cSamples
   do {
      const double val = *pScan;
      if(PREDICTABLE(val != rangeVal)) {
         const size_t cEqualRangeItems = pScan - pStartEqualRange;
         if(cUncuttableRangeLengthMin <= cEqualRangeItems) {
            if(aFeatureVals != pCuttableValsStart || cSamplesBinMin <= static_cast<size_t>(pStartEqualRange - pCuttableValsStart)) {
               ++cCuttingRanges;
            }
            pCuttableValsStart = pScan;
         }
         rangeVal = val;
         pStartEqualRange = pScan;
      }
      ++pScan;
   } while(LIKELY(pValsEnd != pScan));
   if(aFeatureVals == pCuttableValsStart) {
      EBM_ASSERT(0 == cCuttingRanges);

      // we're still on the first cutting range.  We need to make sure that there is at least one possible cut
      // if we require 3 items for a cut, a problematic range like 0 1 3 3 4 5 could look ok, but we can't cut it in the middle!

      const double * pLow = aFeatureVals + cSamplesBinMin - 1;
      EBM_ASSERT(pLow < pValsEnd);
      const double * pHigh = pValsEnd - cSamplesBinMin;
      EBM_ASSERT(aFeatureVals <= pHigh);
      EBM_ASSERT(pLow < pHigh);

      // if they are equal, then there are no values between them where we could cut.  
      // If unequal, there's a cut somewhere
      return UNPREDICTABLE(*pLow == *pHigh) ? size_t { 0 } : size_t { 1 };
   } else {
      const size_t cItemsLast = static_cast<size_t>(pValsEnd - pCuttableValsStart);
      if(cSamplesBinMin <= cItemsLast) {
         ++cCuttingRanges;
      }
      return cCuttingRanges;
   }
}

INLINE_RELEASE_UNTEMPLATED static size_t GetUncuttableRangeLengthMin(
   const size_t cSamples, 
   const size_t cBinsMax, 
   const size_t cSamplesBinMin
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
   // those are less critical since they can be small down to even just 1 value if cSamplesBinMin is 1.
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
   EBM_ASSERT(size_t { 1 } <= cSamplesBinMin);
   EBM_ASSERT(cSamplesBinMin <= cSamples / size_t { 2 }); // we exit earlier if we don't have enough samples for 2 bins

   size_t cUncuttableRangeLengthMin = (cSamples - size_t { 1 }) / cBinsMax + size_t { 1 }; // get the ceil value
   cUncuttableRangeLengthMin = UNPREDICTABLE(cUncuttableRangeLengthMin < cSamplesBinMin) ? 
      cSamplesBinMin : cUncuttableRangeLengthMin;

   EBM_ASSERT(size_t { 1 } <= cUncuttableRangeLengthMin);

   return cUncuttableRangeLengthMin;
}

// we don't care if an extra log message is outputted due to the non-atomic nature of the decrement to this value
static int g_cLogEnterCutQuantile = 25;
static int g_cLogExitCutQuantile = 25;

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION CutQuantile(
   IntEbm countSamples,
   const double * featureVals,
   IntEbm minSamplesBin,
   BoolEbm isRounded,
   IntEbm * countCutsInOut,
   double * cutsLowerBoundInclusiveOut
) {
   // don't expose this random seed.  It's used to settle tiebreakers and will only make 
   // marginal changes to where the cuts are placed.  Exposing it just means we need to 
   // use the same value in every language that we support, and any preprocessors then need to
   // take a random number to be useful, which would be odd for a preprocessor.
   static constexpr uint64_t seed = 9397611943394063143u;

   LOG_COUNTED_N(
      &g_cLogEnterCutQuantile,
      Trace_Info,
      Trace_Verbose,
      "Entered CutQuantile: "
      "countSamples=%" IntEbmPrintf ", "
      "featureVals=%p, "
      "minSamplesBin=%" IntEbmPrintf ", "
      "isRounded=%s, "
      "countCutsInOut=%p, "
      "cutsLowerBoundInclusiveOut=%p"
      ,
      countSamples,
      static_cast<const void *>(featureVals),
      minSamplesBin,
      ObtainTruth(isRounded),
      static_cast<void *>(countCutsInOut),
      static_cast<void *>(cutsLowerBoundInclusiveOut)
   );

   ErrorEbm error;

   IntEbm countCutsRet;

   if(UNLIKELY(nullptr == countCutsInOut)) {
      LOG_0(Trace_Error, "ERROR CutQuantile nullptr == countCutsInOut");
      countCutsRet = IntEbm { 0 };
      error = Error_IllegalParamVal;
   } else {
      if(UNLIKELY(countSamples <= IntEbm { 1 })) {
         // can't cut 1 sample
         countCutsRet = IntEbm { 0 };
         error = Error_None;
         if(UNLIKELY(countSamples < IntEbm { 0 })) {
            LOG_0(Trace_Error, "ERROR CutQuantile countSamples < IntEbm { 0 }");
            error = Error_IllegalParamVal;
         }
      } else {
         if(UNLIKELY(nullptr == featureVals)) {
            LOG_0(Trace_Error, "ERROR CutQuantile nullptr == featureVals");

            countCutsRet = IntEbm { 0 };
            error = Error_IllegalParamVal;
            goto exit_with_log;
         }

         if(UNLIKELY(IsConvertError<size_t>(countSamples))) {
            LOG_0(Trace_Warning, "WARNING CutQuantile IsConvertError<size_t>(countSamples)");

            countCutsRet = IntEbm { 0 };
            error = Error_IllegalParamVal;
            goto exit_with_log;
         }

         const size_t cSamplesIncludingMissingVals = static_cast<size_t>(countSamples);


         if(IsMultiplyError(sizeof(double), cSamplesIncludingMissingVals)) {
            LOG_0(Trace_Error, "ERROR CutQuantile IsMultiplyError(sizeof(double), cSamplesIncludingMissingVals)");

            countCutsRet = IntEbm { 0 };
            error = Error_OutOfMemory;
            goto exit_with_log;
         }
         const size_t cBytesFeatureVals = sizeof(double) * cSamplesIncludingMissingVals;
         double * const aFeatureVals = static_cast<double *>(malloc(cBytesFeatureVals));
         if(UNLIKELY(nullptr == aFeatureVals)) {
            LOG_0(Trace_Error, "ERROR CutQuantile nullptr == aFeatureVals");

            countCutsRet = IntEbm { 0 };
            error = Error_OutOfMemory;
            goto exit_with_log;
         }
         memcpy(aFeatureVals, featureVals, cBytesFeatureVals);

         // if there are +infinity values in the data we won't be able to separate them
         // from max_float values without having a cut at infinity since we use lower bound inclusivity
         // so we disallow +infinity values by turning them into max_float.  For symmetry we do the same on
         // the -infinity side turning those into lowest_float.  
         const size_t cSamples = RemoveMissingValsAndReplaceInfinities(cSamplesIncludingMissingVals, aFeatureVals);

         EBM_ASSERT(cSamples <= cSamplesIncludingMissingVals);

         if(UNLIKELY(cSamples <= size_t { 1 })) {
            free(aFeatureVals);
            // we can't really cut 0 or 1 samples.  Now that we know our min, max, etc values, we can exit
            // or if there was only 1 non-missing value
            countCutsRet = IntEbm { 0 };
            error = Error_None;
            goto exit_with_log;
         }

         EBM_ASSERT(nullptr != countCutsInOut);
         const IntEbm countCuts = *countCutsInOut;

         if(UNLIKELY(countCuts <= IntEbm { 0 })) {
            free(aFeatureVals);
            countCutsRet = IntEbm { 0 };
            error = Error_None;
            if(UNLIKELY(countCuts < IntEbm { 0 })) {
               LOG_0(Trace_Error, "ERROR CutQuantile countCuts can't be negative.");
               error = Error_IllegalParamVal;
            }
            goto exit_with_log;
         }
         
         if(UNLIKELY(nullptr == cutsLowerBoundInclusiveOut)) {
            // if we have a potential bin cut, then cutsLowerBoundInclusiveOut shouldn't be nullptr
            LOG_0(Trace_Error, "ERROR CutQuantile nullptr == cutsLowerBoundInclusiveOut");

            free(aFeatureVals);
            countCutsRet = IntEbm { 0 };
            error = Error_IllegalParamVal;

            goto exit_with_log;
         }

         if(UNLIKELY(minSamplesBin <= IntEbm { 0 })) {
            LOG_0(Trace_Warning,
               "WARNING CutQuantile minSamplesBin shouldn't be zero or negative.  Setting to 1");

            minSamplesBin = IntEbm { 1 };
         }

         EBM_ASSERT(!IsConvertError<IntEbm>(cSamples)); // since it came from an IntEbm originally
         if(UNLIKELY(static_cast<IntEbm>(cSamples >> 1) < minSamplesBin)) {
            // each bin needs at least minSamplesBin samples, so we need two sets of minSamplesBin
            // in order to make any cuts.  Anything less and we should just return now.
            // We also use this as a comparison to ensure that minSamplesBin is convertible to a size_t

            free(aFeatureVals);
            countCutsRet = IntEbm { 0 };
            error = Error_None;
            goto exit_with_log;
         }

         // minSamplesBin is convertible to size_t since minSamplesBin <= (cSamples >> 1)
         EBM_ASSERT(!IsConvertError<size_t>(minSamplesBin));
         const size_t cSamplesBinMin = static_cast<size_t>(minSamplesBin);

         // In theory, we could constrain our cBinsMaxInitial value a bit more by taking our value array
         // and attempting to jump by the minimum each time.  Then if there was a long run of equal values we'd
         // be able to limit the number of cuts, but then the algorithm is going to need to be pretty smart later
         // on when it finds the long run and needs to compress the available cuts back down into the cutable regions
         // it's probably better to just place a lot of asiprational cuts at the minimum separation and trim them
         // as we go on so.  In that case we'd be hard pressed to misallocate cuts since they'll almost always
         // alrady be cSamplesBinMin apart in the regions that are cutable.
         const size_t cBinsMaxInitial = cSamples / cSamplesBinMin;

         // otherwise we'd have failed the check "static_cast<IntEbm>(cSamples >> 1) < minSamplesBin"
         EBM_ASSERT(size_t { 2 } <= cBinsMaxInitial);
         const size_t cCutsMaxInitial = cBinsMaxInitial - size_t { 1 };

         // cSamples fit into an IntEbm, and since cCutsMaxInitial is less than cSamples, 
         // we should be able to convert it back to an IntEbm
         EBM_ASSERT(cCutsMaxInitial < cSamples);
         EBM_ASSERT(!IsConvertError<IntEbm>(cCutsMaxInitial));
         const size_t cCutsMax = static_cast<IntEbm>(cCutsMaxInitial) < countCuts ?
            cCutsMaxInitial : static_cast<size_t>(countCuts);

         EBM_ASSERT(size_t { 1 } <= cCutsMax); // we won't eliminate to less than 1, and we had at least 1 before

         // we need to be able to index both the cutsLowerBoundInclusiveOut AND we also allocate an array
         // of pointers below of double * to index into aFeatureVals 
         if(UNLIKELY(IsMultiplyError(std::max(sizeof(*cutsLowerBoundInclusiveOut), sizeof(double *)), cCutsMax))) {
            LOG_0(Trace_Warning, "WARNING CutQuantile IsMultiplyError(std::max(sizeof(*cutsLowerBoundInclusiveOut), sizeof(double *)), cCutsMax)");
            free(aFeatureVals);
            countCutsRet = IntEbm { 0 };
            error = Error_OutOfMemory;
            goto exit_with_log;
         }

         std::sort(aFeatureVals, aFeatureVals + cSamples);

         EBM_ASSERT(cCutsMax < cSamples); // so we can add 1 to cCutsMax safely
         const size_t cUncuttableRangeLengthMin = 
            GetUncuttableRangeLengthMin(cSamples, cCutsMax + size_t { 1 }, cSamplesBinMin);
         EBM_ASSERT(size_t { 1 } <= cUncuttableRangeLengthMin);

         const size_t cCuttingRanges = CountCuttingRanges(
            cSamples, 
            aFeatureVals,
            cUncuttableRangeLengthMin, 
            cSamplesBinMin
         );
         // we GUARANTEE that each interior CuttingRange can have at least one cut by choosing an 
         // cUncuttableRangeLengthMin sufficiently long to ensure this property.  The first and last cutable
         // ranges, if they exist, can be quite small, so we can trade 1 long uncutable range for 2 cutable
         // ranges at the tail ends, so we can get 1 more cut than the maximum number of cuts given to us
         // but not 2 more.  cCutsMax + size_t { 1 } can't overflow since cCutsMax < cSamples , and
         // cSamples is a size_t
         EBM_ASSERT(cCuttingRanges <= cCutsMax + size_t { 1 });
         if(UNLIKELY(size_t { 0 } == cCuttingRanges)) {
            free(aFeatureVals);
            countCutsRet = IntEbm { 0 };
            error = Error_None;
            goto exit_with_log;
         }

         if(UNLIKELY(IsMultiplyError(sizeof(NeighbourJump), cSamples))) {
            LOG_0(Trace_Warning, "WARNING CutQuantile IsMultiplyError(sizeof(NeighbourJump), cSamples)");
            free(aFeatureVals);
            countCutsRet = IntEbm { 0 };
            error = Error_OutOfMemory;
            goto exit_with_log;
         }
         const size_t cBytesNeighbourJumps = sizeof(NeighbourJump) * cSamples;

         // we checked that this multiplication wouldn't overflow above
         EBM_ASSERT(!IsMultiplyError(sizeof(double *), cCutsMax));
         const size_t cBytesValCutPointers = sizeof(double *) * cCutsMax;

         // we limit the cCutsMax to no more than cSamples - 1.  cSamples can't be anywhere close to
         // the maximum size_t though since the caller must have allocated cSamples floats in aFeatureVals, and
         // there are no float types that are 1 byte, and we checked that this didn't overflow, so we should be good
         // to add 2 to the cCutsMax value
         EBM_ASSERT(cCutsMax <= std::numeric_limits<size_t>::max() - size_t { 2 });
         // include storage for the end points
         const size_t cCutsWithEndpointsMax = cCutsMax + size_t { 2 };
         if(UNLIKELY(IsMultiplyError(sizeof(CutPoint), cCutsWithEndpointsMax))) {
            LOG_0(Trace_Warning, "WARNING CutQuantile IsMultiplyError(sizeof(CutPoint), cCutsWithEndpointsMax)");
            free(aFeatureVals);
            countCutsRet = IntEbm { 0 };
            error = Error_OutOfMemory;
            goto exit_with_log;
         }
         const size_t cBytesCuts = sizeof(CutPoint) * cCutsWithEndpointsMax;

         if(UNLIKELY(IsMultiplyError(sizeof(CuttingRange), cCuttingRanges))) {
            LOG_0(Trace_Warning, "WARNING CutQuantile IsMultiplyError(sizeof(CuttingRange), cCuttingRanges)");
            free(aFeatureVals);
            countCutsRet = IntEbm { 0 };
            error = Error_OutOfMemory;
            goto exit_with_log;
         }
         const size_t cBytesCuttingRanges = sizeof(CuttingRange) * cCuttingRanges;


         const size_t cBytesToNeighbourJump = size_t { 0 };
         const size_t cBytesToValCutPointers = cBytesToNeighbourJump + cBytesNeighbourJumps;

         if(UNLIKELY(IsAddError(cBytesToValCutPointers, cBytesValCutPointers))) {
            LOG_0(Trace_Warning, "WARNING CutQuantile IsAddError(cBytesToValCutPointers, cBytesValCutPointers))");
            free(aFeatureVals);
            countCutsRet = IntEbm { 0 };
            error = Error_OutOfMemory;
            goto exit_with_log;
         }
         const size_t cBytesToCuts = cBytesToValCutPointers + cBytesValCutPointers;

         if(UNLIKELY(IsAddError(cBytesToCuts, cBytesCuts))) {
            LOG_0(Trace_Warning, "WARNING CutQuantile IsAddError(cBytesToCuts, cBytesCuts))");
            free(aFeatureVals);
            countCutsRet = IntEbm { 0 };
            error = Error_OutOfMemory;
            goto exit_with_log;
         }
         const size_t cBytesToCuttingRange = cBytesToCuts + cBytesCuts;

         if(UNLIKELY(IsAddError(cBytesToCuttingRange, cBytesCuttingRanges))) {
            LOG_0(Trace_Warning, "WARNING CutQuantile IsAddError(cBytesToCuttingRange, cBytesCuttingRanges))");
            free(aFeatureVals);
            countCutsRet = IntEbm { 0 };
            error = Error_OutOfMemory;
            goto exit_with_log;
         }
         const size_t cBytesToEnd = cBytesToCuttingRange + cBytesCuttingRanges;

         char * const pMem = static_cast<char *>(malloc(cBytesToEnd));
         if(UNLIKELY(nullptr == pMem)) {
            LOG_0(Trace_Warning, "WARNING CutQuantile nullptr == pMem");
            free(aFeatureVals);
            countCutsRet = IntEbm { 0 };
            error = Error_OutOfMemory;
            goto exit_with_log;
         }

         NeighbourJump * const aNeighbourJumps = reinterpret_cast<NeighbourJump *>(pMem + cBytesToNeighbourJump);
         const double ** const apValCutTops = reinterpret_cast<const double **>(pMem + cBytesToValCutPointers);
         CutPoint * const aCuts = reinterpret_cast<CutPoint *>(pMem + cBytesToCuts);
         CuttingRange * const aCuttingRange = reinterpret_cast<CuttingRange *>(pMem + cBytesToCuttingRange);

         ConstructJumps(cSamples, aFeatureVals, aNeighbourJumps);

         // we always XOR (with != for bools) a random number with bSymmetryReversal, so there is no need to
         // XOR bSymmetryReversal with a random number here
         const bool bSymmetryReversal = DetermineSymmetricDirection(cSamples, aFeatureVals);

         RandomDeterministic rng;
         rng.Initialize(seed);

         FillTiebreakers(bSymmetryReversal, &rng, cCuttingRanges, aCuttingRange);

         FillCuttingRangeBasics(cSamples, aFeatureVals, cUncuttableRangeLengthMin, cSamplesBinMin, cCuttingRanges, aCuttingRange);
         FillCuttingRangeNeighbours(cSamples, aFeatureVals, cCuttingRanges, aCuttingRange);

         const double ** ppValCutTop = apValCutTops;
         try {
            std::set<CuttingRange *, CompareCuttingRange> priorityQueue;
            StuffCutsIntoCuttingRanges(
               priorityQueue,
               cCuttingRanges,
               aCuttingRange,
               cSamplesBinMin,
               cCutsMax
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
               LOG_N(Trace_Verbose, "Dequque CuttingRange: %zu, %zu, %zu, %zu, %zu, %zu, %zu, %le",
                  pCuttingRange->m_uniqueTiebreaker,
                  pCuttingRange->m_cRangesAssigned,
                  pCuttingRange->m_cCuttableVals,
                  static_cast<size_t>(pCuttingRange->m_pCuttableValsFirst - aFeatureVals),
                  pCuttingRange->m_cUncuttableHighVals,
                  pCuttingRange->m_cUncuttableLowVals,
                  pCuttingRange->m_cRangesMax,
                  pCuttingRange->m_avgCuttableRangeWidthAfterAddingOneCut
               );
#endif // LOG_SUPERVERBOSE_DISCRETIZATION_ORDERED

               if(PREDICTABLE(size_t { 1 } < cRanges)) {
                  // we have cuts on our ends, either explicit or implicit at the tail ends that don't have uncuttable
                  // ranges on the tails, and at least one cut in our center, so we have to make decisions
                  std::set<CutPoint *, CompareCutPoint> bestCuts;

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
                  //        one of our cuts from that side to a new set of ranges (encoded as Cuts)
                  //        We still do the low/high cut number optimization with our left and right windows
                  //        when planning since it's more efficient, and no changes should leak information
                  //        outside those windows otherwise it would become an N^2 algorithm.
                  //        We use our doubly linked list to move non-materialized cut points long distances
                  //        from one part of the cutting range to annother if necessary.
                  //        We should also use the doubly linked list to delete Cuts that we can't use
                  //        if there is no place to put them

                  std::set<CutPoint *, CompareCutPoint> fillTheVoids;
#endif // NEVER

                  FillTiebreakers(bSymmetryReversal, &rng, cRanges - size_t { 1 }, aCuts + 1);

                  error = TradeCutSegment(
                     &bestCuts,
                     cSamples,
                     bSymmetryReversal,
                     cSamplesBinMin,
                     pCuttingRange->m_pCuttableValsFirst - aFeatureVals,
                     pCuttingRange->m_cCuttableVals,
                     aNeighbourJumps,
                     cRanges,
                     // for efficiency we include space for the end point cuts even if they don't exist
                     aCuts
                  );
                  if(Error_None != error) {
                     // any error messages should have been written to the log inside TradeCutSegment

                     free(pMem);
                     free(aFeatureVals);

                     countCutsRet = IntEbm { 0 };
                     goto exit_with_log;
                  }

                  const double * const pCuttableValsStart = pCuttingRange->m_pCuttableValsFirst;

                  if(0 != pCuttingRange->m_cUncuttableLowVals) {
                     // if it's zero then it's an implicit cut and we shouldn't put one there, 
                     // otherwise put in the cut
                     const double * const pCut = pCuttableValsStart;
                     EBM_ASSERT(aFeatureVals < pCut);
                     EBM_ASSERT(pCut < aFeatureVals + countSamples);
                     *ppValCutTop = pCut;
                     ++ppValCutTop;
                  }

                  const CutPoint * pCutPoint = aCuts->m_pNext;
                  const CutPoint * pNext = pCutPoint->m_pNext;
                  while(LIKELY(nullptr != pNext)) {
                     const size_t iVal = pCutPoint->m_iVal;
                     if(LIKELY(k_valNotLegal != iVal)) {
                        const double * const pCut = pCuttableValsStart + iVal;
                        EBM_ASSERT(aFeatureVals < pCut);
                        EBM_ASSERT(pCut < aFeatureVals + countSamples);
                        EBM_ASSERT(pCuttingRange->m_pCuttableValsFirst < pCut);
                        EBM_ASSERT(pCut < pCuttingRange->m_pCuttableValsFirst + pCuttingRange->m_cCuttableVals);
                        *ppValCutTop = pCut;
                        ++ppValCutTop;
                     }
                     pCutPoint = pNext;
                     pNext = pCutPoint->m_pNext;
                  }

                  if(0 != pCuttingRange->m_cUncuttableHighVals) {
                     // if it's zero then it's an implicit cut and we shouldn't put one there, 
                     // otherwise put in the cut
                     const double * const pCut =
                        pCuttableValsStart + pCuttingRange->m_cCuttableVals;
                     EBM_ASSERT(aFeatureVals < pCut);
                     EBM_ASSERT(pCut < aFeatureVals + countSamples);
                     *ppValCutTop = pCut;
                     ++ppValCutTop;
                  }
               } else if(PREDICTABLE(size_t { 1 } == cRanges)) {
                  // we have cuts on both our ends (either explicit or implicit), so
                  // we don't have to make any hard decisions, but we do have to be careful of the scenarios
                  // where some of our cuts are implicit

                  if(0 != pCuttingRange->m_cUncuttableLowVals) {
                     // if it's zero then it's an implicit cut and we shouldn't put one there, 
                     // otherwise put in the cut
                     const double * const pCut = pCuttingRange->m_pCuttableValsFirst;
                     EBM_ASSERT(aFeatureVals < pCut);
                     EBM_ASSERT(pCut < aFeatureVals + countSamples);
                     *ppValCutTop = pCut;
                     ++ppValCutTop;
                  }
                  if(0 != pCuttingRange->m_cUncuttableHighVals) {
                     // if it's zero then it's an implicit cut and we shouldn't put one there, 
                     // otherwise put in the cut
                     const double * const pCut =
                        pCuttingRange->m_pCuttableValsFirst + pCuttingRange->m_cCuttableVals;
                     EBM_ASSERT(aFeatureVals < pCut);
                     EBM_ASSERT(pCut < aFeatureVals + countSamples);
                     *ppValCutTop = pCut;
                     ++ppValCutTop;
                  }
               } else {
                  EBM_ASSERT(0 == cRanges);
                  // we have only 1 cut to place, and no cuts on our boundaries, so we need to figure out
                  // where in our range to place it, taking into consideration that we might have neighbours on our
                  // sides that could be large

                  // if we had implicit cuts on both ends and zero assigned cuts, we'd have 1 range and would
                  // be handled above
                  EBM_ASSERT(0 != pCuttingRange->m_cUncuttableLowVals || 0 != pCuttingRange->m_cUncuttableHighVals);

                  // if one side or the other was an implicit cut, then we have zero cuts left after
                  // the implicit cut is accounted for, so do nothing
                  if(LIKELY(LIKELY(0 != pCuttingRange->m_cUncuttableLowVals) && 
                     LIKELY(0 != pCuttingRange->m_cUncuttableHighVals))) {
                     // even though we could reduce our squared error length more, it probably makes sense to 
                     // include a little bit of our available numbers on one long range and the other, so let's put
                     // the cut in the middle and only make the low/high decision to settle long-ish ranges
                     // in the center

                     const size_t cCuttableItems = pCuttingRange->m_cCuttableVals;
                        
                     const size_t iRangeFirst = pCuttingRange->m_pCuttableValsFirst - aFeatureVals;
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
                     // will reflect the boundary of the point after the uncuttable range above
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
                        const size_t cDistanceLow2 = pCuttingRange->m_cUncuttableLowVals;
                        const size_t cDistanceHigh2 = pCuttingRange->m_cUncuttableHighVals;
                        iResult = UNPREDICTABLE(cDistanceHigh2 < cDistanceLow2) ? iStartCur : iStartNext;
                        if(UNLIKELY(cDistanceHigh2 == cDistanceLow2)) {
                           // next, let's try to the edges of our full array
                           const size_t cDistanceLow3 = iStartCur;
                           const size_t cDistanceHigh3 = cSamples - iStartNext;
                           iResult = UNPREDICTABLE(cDistanceHigh3 < cDistanceLow3) ? iStartCur : iStartNext;
                           if(UNLIKELY(cDistanceHigh3 == cDistanceLow3)) {
                              // wow, we're at the center of the entire array AND the center of the outer
                              // uncuttable ranges, AND the center of the cutable ranges.  Our final fallback
                              // is to resort to our symmetric determination (PLUS randomness)

                              bool bLocalSymmetryReversal = rng.Next<bool>() != bSymmetryReversal;
                              iResult = UNPREDICTABLE(bLocalSymmetryReversal) ? iStartCur : iStartNext;
                           }
                        }
                     }
                     const double * pCut = aFeatureVals + iResult;
                     EBM_ASSERT(aFeatureVals < pCut);
                     *ppValCutTop = pCut;
                     ++ppValCutTop;
                  }
               }
            } while(!priorityQueue.empty());
         } catch(const std::bad_alloc &) {
            LOG_0(Trace_Warning, "WARNING CutQuantile out of memory");

            free(pMem);
            free(aFeatureVals);

            countCutsRet = IntEbm { 0 };
            error = Error_OutOfMemory;
            goto exit_with_log;
         } catch(...) {
            LOG_0(Trace_Warning, "WARNING CutQuantile exception");

            free(pMem);
            free(aFeatureVals);

            countCutsRet = IntEbm { 0 };
            error = Error_UnexpectedInternal;
            goto exit_with_log;
         }

         EBM_ASSERT(apValCutTops <= ppValCutTop);
         const size_t cCutsRet = ppValCutTop - apValCutTops;

         // it's possible, although extremely unlikely, that due to floating point issues that should only
         // occur with huge double indexes, we were not able to find the legal cut point, so check for zero
         if(LIKELY(size_t { 0 } != cCutsRet)) {
            // the pointers are guaranteed to be in same order as the cut values
            std::sort(apValCutTops, ppValCutTop);

            double * pCutsLowerBoundInclusive = cutsLowerBoundInclusiveOut;
            const double * const * ppValCutTop2 = apValCutTops;

            if(EBM_FALSE == isRounded) {
               do {
                  const double * const pCut = *ppValCutTop2;
                  EBM_ASSERT(aFeatureVals < pCut);
                  EBM_ASSERT(pCut < aFeatureVals + cSamples);
                  const double valHigh = *pCut;
                  EBM_ASSERT(!std::isnan(valHigh));
                  EBM_ASSERT(!std::isinf(valHigh));
                  const double valLow = *(pCut - size_t { 1 });
                  EBM_ASSERT(!std::isnan(valLow));
                  EBM_ASSERT(!std::isinf(valLow));
                  const double cut = ArithmeticMean(valLow, valHigh);
                  EBM_ASSERT(cutsLowerBoundInclusiveOut == pCutsLowerBoundInclusive || *(pCutsLowerBoundInclusive - size_t { 1 }) < cut);
                  *pCutsLowerBoundInclusive = cut;
                  ++pCutsLowerBoundInclusive;
                  ++ppValCutTop2;
               } while(ppValCutTop != ppValCutTop2);
            } else {
               do {
                  const double * const pCut = *ppValCutTop2;
                  EBM_ASSERT(aFeatureVals < pCut);
                  EBM_ASSERT(pCut < aFeatureVals + cSamples);
                  const double valHigh = *pCut;
                  EBM_ASSERT(!std::isnan(valHigh));
                  EBM_ASSERT(!std::isinf(valHigh));
                  const double valLow = *(pCut - size_t { 1 });
                  EBM_ASSERT(!std::isnan(valLow));
                  EBM_ASSERT(!std::isinf(valLow));
                  const double cut = GetInterpretableCutPointFloat(valLow, valHigh);
                  EBM_ASSERT(cutsLowerBoundInclusiveOut == pCutsLowerBoundInclusive || *(pCutsLowerBoundInclusive - size_t { 1 }) < cut);
                  *pCutsLowerBoundInclusive = cut;
                  ++pCutsLowerBoundInclusive;
                  ++ppValCutTop2;
               } while(ppValCutTop != ppValCutTop2);

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

               if(LIKELY(size_t { 3 } <= cCutsRet)) {
                  const double * const pScaleHighHigh = *(ppValCutTop - size_t { 1 });
                  EBM_ASSERT(aFeatureVals + size_t { 2 } < pScaleHighHigh);
                  EBM_ASSERT(pScaleHighHigh < aFeatureVals + cSamples);
                  const double * const pScaleHighLow = pScaleHighHigh - size_t { 1 };
                  EBM_ASSERT(aFeatureVals + size_t { 1 } < pScaleHighLow);
                  EBM_ASSERT(pScaleHighLow < aFeatureVals + cSamples - size_t { 1 });
                  const double scaleHighLow = *pScaleHighLow;
                  EBM_ASSERT(!std::isnan(scaleHighLow));
                  EBM_ASSERT(!std::isinf(scaleHighLow));
                  const double * pScaleLowHigh = *apValCutTops;
                  EBM_ASSERT(aFeatureVals < pScaleLowHigh);
                  EBM_ASSERT(pScaleLowHigh < aFeatureVals + cSamples - size_t { 2 });
                  const double scaleLowHigh = *pScaleLowHigh;
                  EBM_ASSERT(!std::isnan(scaleLowHigh));
                  EBM_ASSERT(!std::isinf(scaleLowHigh));
                  EBM_ASSERT(scaleLowHigh < scaleHighLow);
                  // this is the inescapable scale of our graph, from the value right above the lowest cut to the value 
                  // right below the highest cut

                  const double scaleMin = scaleHighLow - scaleLowHigh;
                  // scaleMin can be +infinity if scaleHighLow is max and scaleLowHigh is lowest.  We can handle it.
                  EBM_ASSERT(!std::isnan(scaleMin));
                  // IEEE 754 (which we static_assert) won't allow the subtraction of two unequal numbers to be non-zero
                  EBM_ASSERT(double { 0 } < scaleMin);

                  // limit the amount of dillution allowed for the tails by capping the relevant cCutPointRet value
                  // to 1/32, which means we leave about 3% of the visible area to tail bounds (1.5% on the left and
                  // 1.5% on the right)

                  const size_t cCutsLimited = size_t { 32 } < cCutsRet ? size_t { 32 } : cCutsRet;

                  // the leftmost and rightmost cuts can legally be right outside of the bounds between scaleHighLow and
                  // scaleLowHigh, so we subtract these two cuts, leaving us the number of ranges between the two end
                  // points.  Half a range on the bottom, N - 1 ranges in the middle, and half a range on the top
                  // Dividing by that number of ranges gives us the average range width.  We don't want to get the final
                  // cut though from the previous inner cut.  We want to move outwards from the scaleHighLow and
                  // scaleLowHigh values, which should be half a cut inwards (not exactly but in spirit), so we
                  // divide by two, which is the same as multiplying the divisor by 2, which is the right shift below
                  EBM_ASSERT(IntEbm { 3 } <= countCuts);
                  const size_t denominator = (cCutsLimited - size_t { 2 }) << 1;
                  EBM_ASSERT(size_t { 0 } < denominator);
                  const double movementFromEnds = scaleMin / static_cast<double>(denominator);
                  // movementFromEnds can be +infinity if scaleMin is infinity. We can handle it.
                  EBM_ASSERT(!std::isnan(movementFromEnds));
                  EBM_ASSERT(double { 0 } <= movementFromEnds); // underflow is possible

                  const double lowCutFullPrecisionMin = scaleLowHigh - movementFromEnds;
                  // lowCutFullPrecisionMin can be -infinity if movementFromEnds is +infinity.  We can handle it.
                  EBM_ASSERT(!std::isnan(lowCutFullPrecisionMin));
                  EBM_ASSERT(lowCutFullPrecisionMin < std::numeric_limits<double>::max());
                  // GetInterpretableEndpoint can accept -infinity, but it'll return -infinity in that case
                  const double lowCutMin = GetInterpretableEndpoint(lowCutFullPrecisionMin, movementFromEnds);
                  // lowCutMin can legally be -infinity and we handle this scenario below

                  const double lowCutExisting = *cutsLowerBoundInclusiveOut;
                  EBM_ASSERT(!std::isnan(lowCutExisting));
                  EBM_ASSERT(!std::isinf(lowCutExisting));

                  if(lowCutExisting < lowCutMin) {
                     // lowCutMin can legally be -infinity, but then we wouldn't get here then
                     EBM_ASSERT(!std::isnan(lowCutMin));
                     EBM_ASSERT(!std::isinf(lowCutMin));
                     *cutsLowerBoundInclusiveOut = lowCutMin;
                  }

                  const double highCutFullPrecisionMax = scaleHighLow + movementFromEnds;
                  // highCutFullPrecisionMax can be +infinity if movementFromEnds is +infinity.  We can handle it.
                  EBM_ASSERT(!std::isnan(highCutFullPrecisionMax));
                  EBM_ASSERT(std::numeric_limits<double>::lowest() < highCutFullPrecisionMax);
                  // GetInterpretableEndpoint can accept infinity, but it'll return infinity in that case
                  const double highCutMax = GetInterpretableEndpoint(highCutFullPrecisionMax, movementFromEnds);
                  // highCutMax can legally be +infinity and we handle this scenario below

                  const double highCutExisting = *(pCutsLowerBoundInclusive - size_t { 1 });
                  EBM_ASSERT(!std::isnan(highCutExisting));
                  EBM_ASSERT(!std::isinf(highCutExisting));

                  if(highCutMax < highCutExisting) {
                     // highCutMax can legally be +infinity, but then we wouldn't get here then
                     EBM_ASSERT(!std::isnan(highCutMax));
                     EBM_ASSERT(!std::isinf(highCutMax));
                     *(pCutsLowerBoundInclusive - size_t { 1 }) = highCutMax;
                  }
               }
            }
         }

         // this conversion is guaranteed to work since the number of cut points can't exceed the number our user
         // specified, and that value came to us as an IntEbm
         countCutsRet = static_cast<IntEbm>(cCutsRet);
         EBM_ASSERT(countCutsRet <= countCuts);

         free(pMem);
         free(aFeatureVals);

         error = Error_None;
      }

   exit_with_log:;

      EBM_ASSERT(nullptr != countCutsInOut);
      *countCutsInOut = countCutsRet;
   }

   LOG_COUNTED_N(
      &g_cLogExitCutQuantile,
      Trace_Info,
      Trace_Verbose,
      "Exited CutQuantile: "
      "countCuts=%" IntEbmPrintf ", "
      "return=%" ErrorEbmPrintf
      ,
      countCutsRet,
      error
   );

   return error;
}

} // DEFINED_ZONE_NAME
