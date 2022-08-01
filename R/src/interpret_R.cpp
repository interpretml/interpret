// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include <cmath> // std::isnan, std::isinf
#include <limits> // std::numeric_limits
#include <cstring> // memcpy, strcmp
#include <algorithm> // std::min, std::max

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "common_cpp.hpp"
#include "ebm_internal.hpp"

#include "BoosterCore.hpp"
#include "InteractionCore.hpp"
#include "BoosterShell.hpp"

#include <Rinternals.h>
#include <R_ext/Visibility.h>

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// when R compiles this library, on some systems it can generate a "NOTE installed size is.." meaning the C++ compiled into a library produces too big a
// library.  We would want to disable the -g flag (with -g0), but according to this, it's not possible currently:
// https://stat.ethz.ch/pipermail/r-devel/2016-October/073273.html

// TODO: switch logging to use the R logging infrastructure when invoked from R, BUT calling error or warning will generate longjumps, which 
//   bypass the regular return mechanisms.  We need to use R_tryCatch (which is older than R_UnwindProtect) to not leak memory that we allocate 
//   before calling the R error or warning functions

INLINE_ALWAYS bool IsSingleDoubleVector(const SEXP sexp) {
   if(REALSXP != TYPEOF(sexp)) {
      return false;
   }
   if(R_xlen_t { 1 } != xlength(sexp)) {
      return false;
   }
   return true;
}

INLINE_ALWAYS bool IsSingleIntVector(const SEXP sexp) {
   if(INTSXP != TYPEOF(sexp)) {
      return false;
   }
   if(R_xlen_t { 1 } != xlength(sexp)) {
      return false;
   }
   return true;
}

INLINE_ALWAYS bool IsSingleBoolVector(const SEXP sexp) {
   if(LGLSXP != TYPEOF(sexp)) {
      return false;
   }
   if(R_xlen_t { 1 } != xlength(sexp)) {
      return false;
   }
   return true;
}

INLINE_ALWAYS bool IsDoubleToIntEbmIndexValid(const double val) {
   if(std::isnan(val)) {
      return false;
   }
   static_assert(std::numeric_limits<double>::is_iec559, "we need is_iec559 to know that comparisons to infinity and -infinity to normal numbers work");
   if(val < double { 0 }) {
      return false;
   }
   double maxValid = std::min(static_cast<double>(std::numeric_limits<size_t>::max()),
      std::min(double { R_XLEN_T_MAX }, static_cast<double>(std::numeric_limits<IntEbm>::max())));
   if(maxValid < val) {
      return false;
   }
   return true;
}

void BoostingFinalizer(SEXP boosterHandleWrapped) {
   EBM_ASSERT(nullptr != boosterHandleWrapped); // shouldn't be possible
   if(EXTPTRSXP == TYPEOF(boosterHandleWrapped)) {
      const BoosterHandle boosterHandle = static_cast<BoosterHandle>(R_ExternalPtrAddr(boosterHandleWrapped));
      if(nullptr != boosterHandle) {
         FreeBooster(boosterHandle);
         R_ClearExternalPtr(boosterHandleWrapped);
      }
   }
}

void InteractionFinalizer(SEXP interactionHandleWrapped) {
   EBM_ASSERT(nullptr != interactionHandleWrapped); // shouldn't be possible
   if(EXTPTRSXP == TYPEOF(interactionHandleWrapped)) {
      const InteractionHandle interactionHandle = static_cast<InteractionHandle>(R_ExternalPtrAddr(interactionHandleWrapped));
      if(nullptr != interactionHandle) {
         FreeInteractionDetector(interactionHandle);
         R_ClearExternalPtr(interactionHandleWrapped);
      }
   }
}

size_t CountTotalDimensions(const size_t cTerms, const IntEbm * const acTermDimensions) {
   EBM_ASSERT(nullptr != acTermDimensions);

   size_t cTotalDimensions = size_t { 0 };
   if(0 != cTerms) {
      const IntEbm * pcTermDimensions = acTermDimensions;
      const IntEbm * const pcTermDimensionsEnd = acTermDimensions + cTerms;
      do {
         const IntEbm countDimensions = *pcTermDimensions;
         if(IsConvertError<size_t>(countDimensions)) {
            LOG_0(Trace_Error, "ERROR CountTotalDimensions IsConvertError<size_t>(countDimensions)");
            return SIZE_MAX;
         }
         const size_t cDimensions = static_cast<size_t>(countDimensions);
         if(IsAddError(cTotalDimensions, cDimensions)) {
            LOG_0(Trace_Error, "ERROR CountTotalDimensions IsAddError(cTotalDimensions, cDimensions)");
            return SIZE_MAX;
         }
         cTotalDimensions += cDimensions;
         ++pcTermDimensions;
      } while(pcTermDimensionsEnd != pcTermDimensions);
   }
   return cTotalDimensions;
}

bool ConvertLogicalsToBools(const SEXP items, size_t * const pcItems, const BoolEbm ** const pRet) {
   EBM_ASSERT(nullptr != items);
   EBM_ASSERT(nullptr != pcItems);
   EBM_ASSERT(nullptr != pRet);
   if(LGLSXP != TYPEOF(items)) {
      LOG_0(Trace_Error, "ERROR ConvertLogicalsToBools LGLSXP != TYPEOF(items)");
      return true;
   }
   const R_xlen_t countItemsR = xlength(items);
   if(IsConvertError<size_t>(countItemsR)) {
      LOG_0(Trace_Error, "ERROR ConvertLogicalsToBools IsConvertError<size_t>(countItemsR)");
      return true;
   }
   const size_t cItems = static_cast<size_t>(countItemsR);
   if(IsConvertError<IntEbm>(cItems)) {
      LOG_0(Trace_Error, "ERROR ConvertLogicalsToBools IsConvertError<IntEbm>(cItems)");
      return true;
   }
   *pcItems = cItems;

   BoolEbm * aItems = nullptr;
   if(0 != cItems) {
      aItems = reinterpret_cast<BoolEbm *>(R_alloc(cItems, static_cast<int>(sizeof(BoolEbm))));
      EBM_ASSERT(nullptr != aItems); // R_alloc doesn't return nullptr, so we don't need to check aItems
      BoolEbm * pItem = aItems;
      const BoolEbm * const pItemEnd = aItems + cItems;
      const int * pOriginal = LOGICAL(items);
      do {
         const Rboolean val = static_cast<Rboolean>(*pOriginal);
         *pItem = Rboolean::FALSE != val ? EBM_TRUE : EBM_FALSE;
         ++pOriginal;
         ++pItem;
      } while(pItemEnd != pItem);
   }
   *pRet = aItems;
   return false;
}

bool ConvertDoublesToIndexes(const SEXP items, size_t * const pcItems, const IntEbm ** const pRet) {
   EBM_ASSERT(nullptr != items);
   EBM_ASSERT(nullptr != pcItems);
   EBM_ASSERT(nullptr != pRet);
   if(REALSXP != TYPEOF(items)) {
      LOG_0(Trace_Error, "ERROR ConvertDoublesToIndexes REALSXP != TYPEOF(items)");
      return true;
   }
   const R_xlen_t countItemsR = xlength(items);
   if(IsConvertError<size_t>(countItemsR)) {
      LOG_0(Trace_Error, "ERROR ConvertDoublesToIndexes IsConvertError<size_t>(countItemsR)");
      return true;
   }
   const size_t cItems = static_cast<size_t>(countItemsR);
   if(IsConvertError<IntEbm>(cItems)) {
      LOG_0(Trace_Error, "ERROR ConvertDoublesToIndexes IsConvertError<IntEbm>(cItems)");
      return true;
   }
   *pcItems = cItems;

   IntEbm * aItems = nullptr;
   if(0 != cItems) {
      aItems = reinterpret_cast<IntEbm *>(R_alloc(cItems, static_cast<int>(sizeof(IntEbm))));
      EBM_ASSERT(nullptr != aItems); // R_alloc doesn't return nullptr, so we don't need to check aItems
      IntEbm * pItem = aItems;
      const IntEbm * const pItemEnd = aItems + cItems;
      const double * pOriginal = REAL(items);
      do {
         const double val = *pOriginal;
         if(!IsDoubleToIntEbmIndexValid(val)) {
            LOG_0(Trace_Error, "ERROR ConvertDoublesToIndexes !IsDoubleToIntEbmIndexValid(val)");
            return true;
         }
         *pItem = static_cast<IntEbm>(val);
         ++pOriginal;
         ++pItem;
      } while(pItemEnd != pItem);
   }
   *pRet = aItems;
   return false;
}

IntEbm CountDoubles(const SEXP items) {
   EBM_ASSERT(nullptr != items);
   if(REALSXP != TYPEOF(items)) {
      LOG_0(Trace_Error, "ERROR CountDoubles REALSXP != TYPEOF(items)");
      return IntEbm { -1 };
   }
   const R_xlen_t countItemsR = xlength(items);
   if(IsConvertErrorDual<size_t, IntEbm>(countItemsR)) {
      LOG_0(Trace_Error, "ERROR CountDoubles IsConvertErrorDual<size_t, IntEbm>(countItemsR)");
      return IntEbm { -1 };
   }
   return static_cast<IntEbm>(countItemsR);
}

SEXP GenerateSeed_R(
   SEXP seed,
   SEXP randomMix
) {
   EBM_ASSERT(nullptr != seed);
   EBM_ASSERT(nullptr != randomMix);

   if(!IsSingleIntVector(seed)) {
      LOG_0(Trace_Error, "ERROR GenerateSeed_R !IsSingleIntVector(seed)");
      return R_NilValue;
   }
   const SeedEbm seedLocal = INTEGER(seed)[0];

   if(!IsSingleIntVector(randomMix)) {
      LOG_0(Trace_Error, "ERROR GenerateSeed_R !IsSingleIntVector(randomMix)");
      return R_NilValue;
   }
   const SeedEbm randomMixLocal = INTEGER(randomMix)[0];

   const SeedEbm retSeed = GenerateSeed(seedLocal, randomMixLocal);

   SEXP ret = PROTECT(allocVector(INTSXP, R_xlen_t { 1 }));
   INTEGER(ret)[0] = retSeed;
   UNPROTECT(1);
   return ret;
}

SEXP CutQuantile_R(
   SEXP featureVals,
   SEXP minSamplesBin,
   SEXP isRounded,
   SEXP countCuts
) {
   EBM_ASSERT(nullptr != featureVals);
   EBM_ASSERT(nullptr != minSamplesBin);
   EBM_ASSERT(nullptr != isRounded);
   EBM_ASSERT(nullptr != countCuts);

   ErrorEbm error;

   const IntEbm countSamples = CountDoubles(featureVals);
   if(countSamples < 0) {
      // we've already logged any errors
      return R_NilValue;
   }
   const double * const aFeatureVals = REAL(featureVals);

   if(!IsSingleDoubleVector(minSamplesBin)) {
      LOG_0(Trace_Error, "ERROR CutQuantile_R !IsSingleDoubleVector(minSamplesBin)");
      return R_NilValue;
   }
   const double minSamplesBinDouble = REAL(minSamplesBin)[0];
   if(!IsDoubleToIntEbmIndexValid(minSamplesBinDouble)) {
      LOG_0(Trace_Error, "ERROR CutQuantile_R !IsDoubleToIntEbmIndexValid(minSamplesBinDouble)");
      return R_NilValue;
   }
   const IntEbm minSamplesBinIntEbm = static_cast<IntEbm>(minSamplesBinDouble);

   if(!IsSingleBoolVector(isRounded)) {
      LOG_0(Trace_Error, "ERROR CutQuantile_R !IsSingleBoolVector(isRounded)");
      return R_NilValue;
   }

   const Rboolean isRoundedR = static_cast<Rboolean>(LOGICAL(isRounded)[0]);
   if(Rboolean::FALSE != isRoundedR && Rboolean::TRUE != isRoundedR) {
      LOG_0(Trace_Error, "ERROR CutQuantile_R Rboolean::FALSE != isRoundedR && Rboolean::TRUE != isRoundedR");
      return R_NilValue;
   }
   const bool bRounded = Rboolean::FALSE != isRoundedR;

   if(!IsSingleDoubleVector(countCuts)) {
      LOG_0(Trace_Error, "ERROR CutQuantile_R !IsSingleDoubleVector(countCuts)");
      return R_NilValue;
   }
   const double countCutsDouble = REAL(countCuts)[0];
   if(!IsDoubleToIntEbmIndexValid(countCutsDouble)) {
      LOG_0(Trace_Error, "ERROR CutQuantile_R !IsDoubleToIntEbmIndexValid(countCutsDouble)");
      return R_NilValue;
   }
   IntEbm countCutsIntEbm = static_cast<IntEbm>(countCutsDouble);
   EBM_ASSERT(!IsConvertError<size_t>(countCutsIntEbm)); // IsDoubleToIntEbmIndexValid checks this

   // TODO: we should allocate the buffer that we're doing to return here directly
   double * const aCutsLowerBoundInclusive = reinterpret_cast<double *>(
      R_alloc(static_cast<size_t>(countCutsIntEbm), static_cast<int>(sizeof(double))));
   EBM_ASSERT(nullptr != aCutsLowerBoundInclusive); // R_alloc doesn't return nullptr, so we don't need to check aItems

   error = CutQuantile(
      countSamples,
      aFeatureVals,
      minSamplesBinIntEbm,
      bRounded ? EBM_TRUE : EBM_FALSE,
      &countCutsIntEbm,
      aCutsLowerBoundInclusive,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr
   );

   if(Error_None != error) {
      return R_NilValue;
   }

   if(IsConvertError<R_xlen_t>(countCutsIntEbm)) {
      return R_NilValue;
   }
   if(IsConvertError<size_t>(countCutsIntEbm)) {
      return R_NilValue;
   }
   const SEXP ret = PROTECT(allocVector(REALSXP, static_cast<R_xlen_t>(countCutsIntEbm)));

   const size_t cCutsIntEbm = static_cast<size_t>(countCutsIntEbm);

   // we've allocated this memory, so it should be reachable, so these numbers should multiply
   EBM_ASSERT(!IsMultiplyError(sizeof(*aCutsLowerBoundInclusive), cCutsIntEbm));

   if(0 != cCutsIntEbm) {
      double * pRet = REAL(ret);
      const double * pCutsLowerBoundInclusive = aCutsLowerBoundInclusive;
      const double * const pCutsLowerBoundInclusiveEnd = aCutsLowerBoundInclusive + cCutsIntEbm;
      do {
         *pRet = static_cast<double>(*pCutsLowerBoundInclusive);
         ++pRet;
         ++pCutsLowerBoundInclusive;
      } while(pCutsLowerBoundInclusiveEnd != pCutsLowerBoundInclusive);
   }

   UNPROTECT(1);
   return ret;
}

SEXP BinFeature_R(
   SEXP featureVals,
   SEXP cutsLowerBoundInclusive,
   SEXP binIndexesOut
) {
   EBM_ASSERT(nullptr != featureVals);
   EBM_ASSERT(nullptr != cutsLowerBoundInclusive);
   EBM_ASSERT(nullptr != binIndexesOut);

   const IntEbm countSamples = CountDoubles(featureVals);
   if(countSamples < 0) {
      // we've already logged any errors
      return R_NilValue;
   }
   const size_t cSamples = static_cast<size_t>(countSamples);
   const double * const aFeatureVals = REAL(featureVals);

   const IntEbm countCuts = CountDoubles(cutsLowerBoundInclusive);
   if(countCuts < 0) {
         // we've already logged any errors
      return R_NilValue;
   }
   const double * const aCutsLowerBoundInclusive = REAL(cutsLowerBoundInclusive);

   if(REALSXP != TYPEOF(binIndexesOut)) {
      LOG_0(Trace_Error, "ERROR BinFeature_R REALSXP != TYPEOF(binIndexesOut)");
      return R_NilValue;
   }
   const R_xlen_t countBinIndexesOutR = xlength(binIndexesOut);
   if(IsConvertError<size_t>(countBinIndexesOutR)) {
      LOG_0(Trace_Error, "ERROR BinFeature_R IsConvertError<size_t>(countBinIndexesOutR)");
      return R_NilValue;
   }
   const size_t cBinIndexesOut = static_cast<size_t>(countBinIndexesOutR);
   if(cSamples != cBinIndexesOut) {
      LOG_0(Trace_Error, "ERROR BinFeature_R cSamples != cBinIndexesOut");
      return R_NilValue;
   }

   if(0 != cSamples) {
      IntEbm * const aiBins = 
         reinterpret_cast<IntEbm *>(R_alloc(cSamples, static_cast<int>(sizeof(IntEbm))));
      EBM_ASSERT(nullptr != aiBins); // this can't be nullptr since R_alloc uses R error handling

      if(Error_None != BinFeature(
         countSamples,
         aFeatureVals,
         countCuts,
         aCutsLowerBoundInclusive,
         aiBins
      )) {
         // we've already logged any errors
         return R_NilValue;
      }

      double * pBinIndexesOut = REAL(binIndexesOut);
      const IntEbm * piBin = aiBins;
      const IntEbm * const piBinsEnd = aiBins + cSamples;
      do {
         const IntEbm iBin = *piBin;
         *pBinIndexesOut = static_cast<double>(iBin);
         ++pBinIndexesOut;
         ++piBin;
      } while(piBinsEnd != piBin);
   }

   // this return isn't useful beyond that it's not R_NilValue, which would signify error
   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = static_cast<double>(cSamples);
   UNPROTECT(1);
   return ret;
}

SEXP SampleWithoutReplacement_R(
   SEXP seed,
   SEXP countTrainingSamples,
   SEXP countValidationSamples,
   SEXP bagOut
) {
   EBM_ASSERT(nullptr != seed);
   EBM_ASSERT(nullptr != countTrainingSamples);
   EBM_ASSERT(nullptr != countValidationSamples);
   EBM_ASSERT(nullptr != bagOut);

   ErrorEbm error;

   if(!IsSingleIntVector(seed)) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement_R !IsSingleIntVector(seed)");
      return R_NilValue;
   }
   const SeedEbm seedLocal = INTEGER(seed)[0];

   if(!IsSingleDoubleVector(countTrainingSamples)) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement_R !IsSingleDoubleVector(countTrainingSamples)");
      return R_NilValue;
   }
   const double countTrainingSamplesDouble = REAL(countTrainingSamples)[0];
   if(!IsDoubleToIntEbmIndexValid(countTrainingSamplesDouble)) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement_R !IsDoubleToIntEbmIndexValid(countTrainingSamplesDouble)");
      return R_NilValue;
   }
   const IntEbm countTrainingSamplesIntEbm = static_cast<IntEbm>(countTrainingSamplesDouble);
   EBM_ASSERT(!IsConvertError<size_t>(countTrainingSamplesIntEbm)); // IsDoubleToIntEbmIndexValid checks this

   if(!IsSingleDoubleVector(countValidationSamples)) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement_R !IsSingleDoubleVector(countValidationSamples)");
      return R_NilValue;
   }
   const double countValidationSamplesDouble = REAL(countValidationSamples)[0];
   if(!IsDoubleToIntEbmIndexValid(countValidationSamplesDouble)) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement_R !IsDoubleToIntEbmIndexValid(countValidationSamplesDouble)");
      return R_NilValue;
   }
   IntEbm countValidationSamplesIntEbm = static_cast<IntEbm>(countValidationSamplesDouble);
   EBM_ASSERT(!IsConvertError<size_t>(countValidationSamplesIntEbm)); // IsDoubleToIntEbmIndexValid checks this

   if(INTSXP != TYPEOF(bagOut)) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement_R INTSXP != TYPEOF(bagOut)");
      return R_NilValue;
   }
   const R_xlen_t countSamples = xlength(bagOut);
   if(IsConvertError<size_t>(countSamples)) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement_R IsConvertError<size_t>(countSamples)");
      return R_NilValue;
   }
   const size_t cSamples = static_cast<size_t>(countSamples);
   if(static_cast<size_t>(countTrainingSamplesIntEbm) + static_cast<size_t>(countValidationSamplesIntEbm) != cSamples) {
      LOG_0(Trace_Error, "ERROR SampleWithoutReplacement_R static_cast<size_t>(countTrainingSamplesIntEbm) + static_cast<size_t>(countValidationSamplesIntEbm) != cSamples");
      return R_NilValue;
   }

   if(0 != cSamples) {
      BagEbm * const aBag =
         reinterpret_cast<BagEbm *>(R_alloc(cSamples, static_cast<int>(sizeof(BagEbm))));
      EBM_ASSERT(nullptr != aBag); // this can't be nullptr since R_alloc uses R error handling

      error = SampleWithoutReplacement(
         seedLocal,
         countTrainingSamplesIntEbm,
         countValidationSamplesIntEbm,
         aBag
      );

      if(Error_None != error) {
         return R_NilValue;
      }

      int32_t * pSampleReplicationOut = INT(bagOut);
      const BagEbm * pSampleReplication = aBag;
      const BagEbm * const pSampleReplicationEnd = aBag + cSamples;
      do {
         const BagEbm replication = *pSampleReplication;
         *pSampleReplicationOut = static_cast<int32_t>(replication);
         ++pSampleReplicationOut;
         ++pSampleReplication;
      } while(pSampleReplicationEnd != pSampleReplication);
   }

   // this return isn't useful beyond that it's not R_NilValue, which would signify error
   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = static_cast<double>(cSamples);
   UNPROTECT(1);
   return ret;
}

SEXP CreateClassificationBooster_R(
   SEXP seed,
   SEXP countClasses,
   SEXP featuresCategorical,
   SEXP featuresBinCount,
   SEXP dimensionCounts,
   SEXP featureIndexes,
   SEXP trainingBinIndexes,
   SEXP trainingTargets,
   SEXP trainingWeights,
   SEXP trainingInitScores,
   SEXP validationBinIndexes,
   SEXP validationTargets,
   SEXP validationWeights,
   SEXP validationInitScores,
   SEXP countInnerBags
) {
   EBM_ASSERT(nullptr != seed);
   EBM_ASSERT(nullptr != countClasses);
   EBM_ASSERT(nullptr != featuresCategorical);
   EBM_ASSERT(nullptr != featuresBinCount);
   EBM_ASSERT(nullptr != dimensionCounts);
   EBM_ASSERT(nullptr != featureIndexes);
   EBM_ASSERT(nullptr != trainingBinIndexes);
   EBM_ASSERT(nullptr != trainingTargets);
   EBM_ASSERT(nullptr != trainingWeights);
   EBM_ASSERT(nullptr != trainingInitScores);
   EBM_ASSERT(nullptr != validationBinIndexes);
   EBM_ASSERT(nullptr != validationTargets);
   EBM_ASSERT(nullptr != validationWeights);
   EBM_ASSERT(nullptr != validationInitScores);
   EBM_ASSERT(nullptr != countInnerBags);

   ErrorEbm error;

   if(!IsSingleIntVector(seed)) {
      LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R !IsSingleIntVector(seed)");
      return R_NilValue;
   }
   const SeedEbm seedLocal = INTEGER(seed)[0];

   if(!IsSingleDoubleVector(countClasses)) {
      LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R !IsSingleDoubleVector(countClasses)");
      return R_NilValue;
   }
   double countClassesDouble = REAL(countClasses)[0];
   if(!IsDoubleToIntEbmIndexValid(countClassesDouble)) {
      LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R !IsDoubleToIntEbmIndexValid(countClassesDouble)");
      return R_NilValue;
   }
   EBM_ASSERT(!IsConvertError<size_t>(countClassesDouble)); // IsDoubleToIntEbmIndexValid checks this
   const size_t cClasses = static_cast<size_t>(countClassesDouble);
   if(IsConvertError<ptrdiff_t>(cClasses)) {
      LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R IsConvertError<ptrdiff_t>(cClasses)");
      return R_NilValue;
   }
   const size_t cScores = GetCountScores(static_cast<ptrdiff_t>(cClasses));

   size_t cFeatures;
   const BoolEbm * aFeaturesCategorical;
   if(ConvertLogicalsToBools(featuresCategorical, &cFeatures, &aFeaturesCategorical)) {
      // we've already logged any errors
      return R_NilValue;
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbm countFeatures = static_cast<IntEbm>(cFeatures);

   size_t cFeaturesFromBinCount;
   const IntEbm * aFeaturesBinCount;
   if(ConvertDoublesToIndexes(featuresBinCount, &cFeaturesFromBinCount, &aFeaturesBinCount)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cFeatures != cFeaturesFromBinCount) {
      LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R cFeatures != cFeaturesFromBinCount");
      return R_NilValue;
   }

   size_t cTerms;
   const IntEbm * acTermDimensions;
   if(ConvertDoublesToIndexes(dimensionCounts, &cTerms, &acTermDimensions)) {
      // we've already logged any errors
      return R_NilValue;
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbm countTerms = static_cast<IntEbm>(cTerms);

   const size_t cTotalDimensionsCheck = CountTotalDimensions(cTerms, acTermDimensions);
   if(SIZE_MAX == cTotalDimensionsCheck) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cTotalDimensionsActual;
   const IntEbm * aiTermFeatures;
   if(ConvertDoublesToIndexes(featureIndexes, &cTotalDimensionsActual, &aiTermFeatures)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cTotalDimensionsActual != cTotalDimensionsCheck) {
      LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R cTotalDimensionsActual != cTotalDimensionsCheck");
      return R_NilValue;
   }

   size_t cTrainingBinIndexes;
   const IntEbm * aTrainingBinIndexes;
   if(ConvertDoublesToIndexes(trainingBinIndexes, &cTrainingBinIndexes, &aTrainingBinIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cTrainingSamples;
   const IntEbm * aTrainingTargets;
   if(ConvertDoublesToIndexes(trainingTargets, &cTrainingSamples, &aTrainingTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbm countTrainingSamples = static_cast<IntEbm>(cTrainingSamples);
   if(IsMultiplyError(cTrainingSamples, cFeatures)) {
      LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R IsMultiplyError(cTrainingSamples, cFeatures)");
      return R_NilValue;
   }
   if(cTrainingSamples * cFeatures != cTrainingBinIndexes) {
      LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R cTrainingSamples * cFeatures != cTrainingBinIndexes");
      return R_NilValue;
   }

   const IntEbm countTrainingInitScores = CountDoubles(trainingInitScores);
   if(countTrainingInitScores < 0) {
      // we've already logged any errors
      return R_NilValue;
   }
   const size_t cTrainingInitScores = static_cast<size_t>(countTrainingInitScores);
   if(IsMultiplyError(cScores, cTrainingSamples)) {
      LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R IsMultiplyError(cScores, cTrainingSamples)");
      return R_NilValue;
   }
   if(cScores * cTrainingSamples != cTrainingInitScores) {
      LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R cScores * cTrainingSamples != cTrainingInitScores");
      return R_NilValue;
   }
   const double * const aTrainingInitScores = REAL(trainingInitScores);

   size_t cValidationBinIndexes;
   const IntEbm * aValidationBinIndexes;
   if(ConvertDoublesToIndexes(validationBinIndexes, &cValidationBinIndexes, &aValidationBinIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cValidationSamples;
   const IntEbm * aValidationTargets;
   if(ConvertDoublesToIndexes(validationTargets, &cValidationSamples, &aValidationTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbm countValidationSamples = static_cast<IntEbm>(cValidationSamples);

   if(IsMultiplyError(cValidationSamples, cFeatures)) {
      LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R IsMultiplyError(cValidationSamples, cFeatures)");
      return R_NilValue;
   }
   if(cValidationSamples * cFeatures != cValidationBinIndexes) {
      LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R cValidationSamples * cFeatures != cValidationBinIndexes");
      return R_NilValue;
   }

   const IntEbm countValidationInitScores = CountDoubles(validationInitScores);
   if(countValidationInitScores < 0) {
      // we've already logged any errors
      return R_NilValue;
   }
   const size_t cValidationInitScores = static_cast<size_t>(countValidationInitScores);
   if(IsMultiplyError(cScores, cValidationSamples)) {
      LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R IsMultiplyError(cScores, cValidationSamples)");
      return R_NilValue;
   }
   if(cScores * cValidationSamples != cValidationInitScores) {
      LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R cScores * cValidationSamples != cValidationInitScores");
      return R_NilValue;
   }
   const double * const aValidationInitScores = REAL(validationInitScores);

   if(!IsSingleIntVector(countInnerBags)) {
      LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R !IsSingleIntVector(countInnerBags)");
      return R_NilValue;
   }
   int countInnerBagsInt = INTEGER(countInnerBags)[0];
   if(IsConvertError<IntEbm>(countInnerBagsInt)) {
      LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R IsConvertError<IntEbm>(countInnerBagsInt)");
      return nullptr;
   }
   IntEbm countInnerBagsLocal = static_cast<IntEbm>(countInnerBagsInt);

   double * pTrainingWeights = nullptr;
   double * pValidationWeights = nullptr;
   if(NILSXP != TYPEOF(trainingWeights) || NILSXP != TYPEOF(validationWeights)) {
      if(REALSXP != TYPEOF(trainingWeights)) {
         LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R REALSXP != TYPEOF(trainingWeights)");
         return R_NilValue;
      }
      R_xlen_t trainingWeightsLength = xlength(trainingWeights);
      if(IsConvertError<size_t>(trainingWeightsLength)) {
         LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R IsConvertError<size_t>(trainingWeightsLength)");
         return R_NilValue;
      }
      size_t cTrainingWeights = static_cast<size_t>(trainingWeightsLength);
      if(cTrainingWeights != cTrainingSamples) {
         LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R cTrainingWeights != cTrainingSamples");
         return R_NilValue;
      }
      pTrainingWeights = REAL(trainingWeights);

      if(REALSXP != TYPEOF(validationWeights)) {
         LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R REALSXP != TYPEOF(validationWeights)");
         return R_NilValue;
      }
      R_xlen_t validationWeightsLength = xlength(validationWeights);
      if(IsConvertError<size_t>(validationWeightsLength)) {
         LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R IsConvertError<size_t>(validationWeightsLength)");
         return R_NilValue;
      }
      size_t cValidationWeights = static_cast<size_t>(validationWeightsLength);
      if(cValidationWeights != cValidationSamples) {
         LOG_0(Trace_Error, "ERROR CreateClassificationBooster_R cValidationWeights != cValidationSamples");
         return R_NilValue;
      }
      pValidationWeights = REAL(validationWeights);
   }

   BoosterHandle boosterHandle;
   error = CreateClassificationBooster(
      seedLocal,
      static_cast<IntEbm>(cClasses),
      countFeatures, 
      aFeaturesCategorical,
      aFeaturesBinCount,
      countTerms, 
      acTermDimensions,
      aiTermFeatures,
      countTrainingSamples, 
      aTrainingBinIndexes, 
      aTrainingTargets, 
      pTrainingWeights,
      aTrainingInitScores,
      countValidationSamples, 
      aValidationBinIndexes, 
      aValidationTargets, 
      pValidationWeights,
      aValidationInitScores,
      countInnerBagsLocal, 
      nullptr,
      &boosterHandle
   );

   if(Error_None != error || nullptr == boosterHandle) {
      return R_NilValue;
   }

   SEXP boosterHandleWrapped = R_MakeExternalPtr(static_cast<void *>(boosterHandle), R_NilValue, R_NilValue); // makes an EXTPTRSXP
   PROTECT(boosterHandleWrapped);

   R_RegisterCFinalizerEx(boosterHandleWrapped, &BoostingFinalizer, Rboolean::TRUE);

   UNPROTECT(1);
   return boosterHandleWrapped;
}

SEXP CreateRegressionBooster_R(
   SEXP seed,
   SEXP featuresCategorical,
   SEXP featuresBinCount,
   SEXP dimensionCounts,
   SEXP featureIndexes,
   SEXP trainingBinIndexes,
   SEXP trainingTargets,
   SEXP trainingWeights,
   SEXP trainingInitScores,
   SEXP validationBinIndexes,
   SEXP validationTargets,
   SEXP validationWeights,
   SEXP validationInitScores,
   SEXP countInnerBags
) {
   EBM_ASSERT(nullptr != seed);
   EBM_ASSERT(nullptr != featuresCategorical);
   EBM_ASSERT(nullptr != featuresBinCount);
   EBM_ASSERT(nullptr != dimensionCounts);
   EBM_ASSERT(nullptr != featureIndexes);
   EBM_ASSERT(nullptr != trainingBinIndexes);
   EBM_ASSERT(nullptr != trainingTargets);
   EBM_ASSERT(nullptr != trainingWeights);
   EBM_ASSERT(nullptr != trainingInitScores);
   EBM_ASSERT(nullptr != validationBinIndexes);
   EBM_ASSERT(nullptr != validationTargets);
   EBM_ASSERT(nullptr != validationWeights);
   EBM_ASSERT(nullptr != validationInitScores);
   EBM_ASSERT(nullptr != countInnerBags);

   ErrorEbm error;

   if(!IsSingleIntVector(seed)) {
      LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R !IsSingleIntVector(seed)");
      return R_NilValue;
   }
   const SeedEbm seedLocal = INTEGER(seed)[0];

   size_t cFeatures;
   const BoolEbm * aFeaturesCategorical;
   if(ConvertLogicalsToBools(featuresCategorical, &cFeatures, &aFeaturesCategorical)) {
      // we've already logged any errors
      return R_NilValue;
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbm countFeatures = static_cast<IntEbm>(cFeatures);

   size_t cFeaturesFromBinCount;
   const IntEbm * aFeaturesBinCount;
   if(ConvertDoublesToIndexes(featuresBinCount, &cFeaturesFromBinCount, &aFeaturesBinCount)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cFeatures != cFeaturesFromBinCount) {
      LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R cFeatures != cFeaturesFromBinCount");
      return R_NilValue;
   }

   size_t cTerms;
   const IntEbm * acTermDimensions;
   if(ConvertDoublesToIndexes(dimensionCounts, &cTerms, &acTermDimensions)) {
      // we've already logged any errors
      return R_NilValue;
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbm countTerms = static_cast<IntEbm>(cTerms);

   const size_t cTotalDimensionsCheck = CountTotalDimensions(cTerms, acTermDimensions);
   if(SIZE_MAX == cTotalDimensionsCheck) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cTotalDimensionsActual;
   const IntEbm * aiTermFeatures;
   if(ConvertDoublesToIndexes(featureIndexes, &cTotalDimensionsActual, &aiTermFeatures)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cTotalDimensionsActual != cTotalDimensionsCheck) {
      LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R cTotalDimensionsActual != cTotalDimensionsCheck");
      return R_NilValue;
   }

   size_t cTrainingBinIndexes;
   const IntEbm * aTrainingBinIndexes;
   if(ConvertDoublesToIndexes(trainingBinIndexes, &cTrainingBinIndexes, &aTrainingBinIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }

   const IntEbm countTrainingSamples = CountDoubles(trainingTargets);
   if(countTrainingSamples < 0) {
      // we've already logged any errors
      return R_NilValue;
   }
   size_t cTrainingSamples = static_cast<size_t>(countTrainingSamples);
   if(IsMultiplyError(cTrainingSamples, cFeatures)) {
      LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R IsMultiplyError(cTrainingSamples, cFeatures)");
      return R_NilValue;
   }
   if(cTrainingSamples * cFeatures != cTrainingBinIndexes) {
      LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R cTrainingSamples * cFeatures != cTrainingBinIndexes");
      return R_NilValue;
   }
   const double * const aTrainingTargets = REAL(trainingTargets);

   const IntEbm countTrainingInitScores = CountDoubles(trainingInitScores);
   if(countTrainingInitScores < 0) {
      // we've already logged any errors
      return R_NilValue;
   }
   size_t cTrainingInitScores = static_cast<size_t>(countTrainingInitScores);
   if(cTrainingSamples != cTrainingInitScores) {
      LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R cTrainingSamples != cTrainingInitScores");
      return R_NilValue;
   }
   const double * const aTrainingInitScores = REAL(trainingInitScores);

   size_t cValidationBinIndexes;
   const IntEbm * aValidationBinIndexes;
   if(ConvertDoublesToIndexes(validationBinIndexes, &cValidationBinIndexes, &aValidationBinIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }

   const IntEbm countValidationSamples = CountDoubles(validationTargets);
   if(countValidationSamples < 0) {
      // we've already logged any errors
      return R_NilValue;
   }
   size_t cValidationSamples = static_cast<size_t>(countValidationSamples);
   if(IsMultiplyError(cValidationSamples, cFeatures)) {
      LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R IsMultiplyError(cValidationSamples, cFeatures)");
      return R_NilValue;
   }
   if(cValidationSamples * cFeatures != cValidationBinIndexes) {
      LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R cValidationSamples * cFeatures != cValidationBinIndexes");
      return R_NilValue;
   }
   const double * const aValidationTargets = REAL(validationTargets);

   const IntEbm countValidationInitScores = CountDoubles(validationInitScores);
   if(countValidationInitScores < 0) {
      // we've already logged any errors
      return R_NilValue;
   }
   size_t cValidationInitScores = static_cast<size_t>(countValidationInitScores);
   if(cValidationSamples != cValidationInitScores) {
      LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R cValidationSamples != cValidationInitScores");
      return R_NilValue;
   }
   const double * const aValidationInitScores = REAL(validationInitScores);

   if(!IsSingleIntVector(countInnerBags)) {
      LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R !IsSingleIntVector(countInnerBags)");
      return R_NilValue;
   }
   int countInnerBagsInt = INTEGER(countInnerBags)[0];
   if(IsConvertError<IntEbm>(countInnerBagsInt)) {
      LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R IsConvertError<IntEbm>(countInnerBagsInt)");
      return nullptr;
   }
   IntEbm countInnerBagsLocal = static_cast<IntEbm>(countInnerBagsInt);

   double * pTrainingWeights = nullptr;
   double * pValidationWeights = nullptr;
   if(NILSXP != TYPEOF(trainingWeights) || NILSXP != TYPEOF(validationWeights)) {
      if(REALSXP != TYPEOF(trainingWeights)) {
         LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R REALSXP != TYPEOF(trainingWeights)");
         return R_NilValue;
      }
      R_xlen_t trainingWeightsLength = xlength(trainingWeights);
      if(IsConvertError<size_t>(trainingWeightsLength)) {
         LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R IsConvertError<size_t>(trainingWeightsLength)");
         return R_NilValue;
      }
      size_t cTrainingWeights = static_cast<size_t>(trainingWeightsLength);
      if(cTrainingWeights != cTrainingSamples) {
         LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R cTrainingWeights != cTrainingSamples");
         return R_NilValue;
      }
      pTrainingWeights = REAL(trainingWeights);

      if(REALSXP != TYPEOF(validationWeights)) {
         LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R REALSXP != TYPEOF(validationWeights)");
         return R_NilValue;
      }
      R_xlen_t validationWeightsLength = xlength(validationWeights);
      if(IsConvertError<size_t>(validationWeightsLength)) {
         LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R IsConvertError<size_t>(validationWeightsLength)");
         return R_NilValue;
      }
      size_t cValidationWeights = static_cast<size_t>(validationWeightsLength);
      if(cValidationWeights != cValidationSamples) {
         LOG_0(Trace_Error, "ERROR CreateRegressionBooster_R cValidationWeights != cValidationSamples");
         return R_NilValue;
      }
      pValidationWeights = REAL(validationWeights);
   }

   BoosterHandle boosterHandle;
   error = CreateRegressionBooster(
      seedLocal,
      countFeatures,
      aFeaturesCategorical,
      aFeaturesBinCount,
      countTerms,
      acTermDimensions,
      aiTermFeatures,
      countTrainingSamples, 
      aTrainingBinIndexes, 
      aTrainingTargets, 
      pTrainingWeights, 
      aTrainingInitScores,
      countValidationSamples, 
      aValidationBinIndexes, 
      aValidationTargets, 
      pValidationWeights, 
      aValidationInitScores,
      countInnerBagsLocal, 
      nullptr,
      &boosterHandle
   );
   if(Error_None != error || nullptr == boosterHandle) {
      return R_NilValue;
   }

   SEXP boosterHandleWrapped = R_MakeExternalPtr(static_cast<void *>(boosterHandle), R_NilValue, R_NilValue); // makes an EXTPTRSXP
   PROTECT(boosterHandleWrapped);

   R_RegisterCFinalizerEx(boosterHandleWrapped, &BoostingFinalizer, Rboolean::TRUE);

   UNPROTECT(1);
   return boosterHandleWrapped;
}

SEXP GenerateTermUpdate_R(
   SEXP boosterHandleWrapped,
   SEXP indexTerm,
   SEXP learningRate,
   SEXP minSamplesLeaf,
   SEXP leavesMax
) {
   EBM_ASSERT(nullptr != boosterHandleWrapped);
   EBM_ASSERT(nullptr != indexTerm);
   EBM_ASSERT(nullptr != learningRate);
   EBM_ASSERT(nullptr != minSamplesLeaf);
   EBM_ASSERT(nullptr != leavesMax);

   ErrorEbm error;

   if(EXTPTRSXP != TYPEOF(boosterHandleWrapped)) {
      LOG_0(Trace_Error, "ERROR GenerateTermUpdate_R EXTPTRSXP != TYPEOF(boosterHandleWrapped)");
      return R_NilValue;
   }
   const BoosterHandle boosterHandle = static_cast<BoosterHandle>(R_ExternalPtrAddr(boosterHandleWrapped));
   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return R_NilValue;
   }

   if(!IsSingleDoubleVector(indexTerm)) {
      LOG_0(Trace_Error, "ERROR GenerateTermUpdate_R !IsSingleDoubleVector(indexTerm)");
      return R_NilValue;
   }
   double doubleIndex = REAL(indexTerm)[0];
   if(!IsDoubleToIntEbmIndexValid(doubleIndex)) {
      LOG_0(Trace_Error, "ERROR GenerateTermUpdate_R !IsDoubleToIntEbmIndexValid(doubleIndex)");
      return R_NilValue;
   }
   const size_t iTerm = static_cast<size_t>(doubleIndex);

   if(!IsSingleDoubleVector(learningRate)) {
      LOG_0(Trace_Error, "ERROR GenerateTermUpdate_R !IsSingleDoubleVector(learningRate)");
      return R_NilValue;
   }
   const double learningRateLocal = REAL(learningRate)[0];

   if(!IsSingleDoubleVector(minSamplesLeaf)) {
      LOG_0(Trace_Error, "ERROR GenerateTermUpdate_R !IsSingleDoubleVector(minSamplesLeaf)");
      return R_NilValue;
   }
   double doubleMinSamplesLeaf = REAL(minSamplesLeaf)[0];
   IntEbm minSamplesLeafEbm;
   static_assert(std::numeric_limits<double>::is_iec559, "we need is_iec559 to know that comparisons to infinity and -infinity to normal numbers work");
   if(std::isnan(doubleMinSamplesLeaf) || FLOAT64_TO_INT64_MAX < doubleMinSamplesLeaf) {
      LOG_0(Trace_Warning, "WARNING GenerateTermUpdate_R minSamplesLeaf overflow");
      minSamplesLeafEbm = FLOAT64_TO_INT64_MAX;
   } else if(doubleMinSamplesLeaf < IntEbm { 1 }) {
      LOG_0(Trace_Warning, "WARNING GenerateTermUpdate_R minSamplesLeaf can't be less than 1. Adjusting to 1.");
      minSamplesLeafEbm = 1;
   } else {
      minSamplesLeafEbm = static_cast<IntEbm>(doubleMinSamplesLeaf);
   }

   size_t cDimensions;
   const IntEbm * aLeavesMax;
   if(ConvertDoublesToIndexes(leavesMax, &cDimensions, &aLeavesMax)) {
      LOG_0(Trace_Error, "ERROR GenerateTermUpdate_R ConvertDoublesToIndexes(leavesMax, &cDimensions, &aLeavesMax)");
      return R_NilValue;
   }
   if(pBoosterShell->GetBoosterCore()->GetCountTerms() <= iTerm) {
      LOG_0(Trace_Error, "ERROR GenerateTermUpdate_R pBoosterShell->GetBoosterCore()->GetCountTerms() <= iTerm");
      return R_NilValue;
   }
   if(cDimensions < pBoosterShell->GetBoosterCore()->GetTerms()[iTerm]->GetCountDimensions()) {
      LOG_0(Trace_Error, "ERROR GenerateTermUpdate_R cDimensions < pBoosterShell->GetBoosterCore()->GetTerms()[iTerm]->GetCountDimensions()");
      return R_NilValue;
   }

   double avgGain;

   error = GenerateTermUpdate(
      boosterHandle,
      static_cast<IntEbm>(iTerm),
      BoostFlags_Default,
      learningRateLocal,
      minSamplesLeafEbm,
      aLeavesMax,
      &avgGain
   );
   if(Error_None != error) {
      LOG_0(Trace_Warning, "WARNING GenerateTermUpdate_R BoostingStep returned error code");
      return R_NilValue;
   }

   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = static_cast<double>(avgGain);
   UNPROTECT(1);
   return ret;
}

SEXP ApplyTermUpdate_R(
   SEXP boosterHandleWrapped
) {
   EBM_ASSERT(nullptr != boosterHandleWrapped);

   ErrorEbm error;

   if(EXTPTRSXP != TYPEOF(boosterHandleWrapped)) {
      LOG_0(Trace_Error, "ERROR ApplyTermUpdate_R EXTPTRSXP != TYPEOF(boosterHandleWrapped)");
      return R_NilValue;
   }
   const BoosterHandle boosterHandle = static_cast<BoosterHandle>(R_ExternalPtrAddr(boosterHandleWrapped));
   // we don't use boosterHandle in this function, so let ApplyTermUpdate check if it's null or invalid

   double validationMetric;
   error = ApplyTermUpdate(boosterHandle, &validationMetric);
   if(Error_None != error) {
      LOG_0(Trace_Warning, "WARNING ApplyTermUpdate_R ApplyTermUpdate returned error code");
      return R_NilValue;
   }

   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = static_cast<double>(validationMetricOut);
   UNPROTECT(1);
   return ret;
}

SEXP GetBestTermScores_R(
   SEXP boosterHandleWrapped,
   SEXP indexTerm
) {
   EBM_ASSERT(nullptr != boosterHandleWrapped); // shouldn't be possible
   EBM_ASSERT(nullptr != indexTerm); // shouldn't be possible

   ErrorEbm error;

   if(EXTPTRSXP != TYPEOF(boosterHandleWrapped)) {
      LOG_0(Trace_Error, "ERROR GetBestTermScores_R EXTPTRSXP != TYPEOF(boosterHandleWrapped)");
      return R_NilValue;
   }
   const BoosterHandle boosterHandle = static_cast<BoosterHandle>(R_ExternalPtrAddr(boosterHandleWrapped));
   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return R_NilValue;
   }
   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();

   if(!IsSingleDoubleVector(indexTerm)) {
      LOG_0(Trace_Error, "ERROR GetBestTermScores_R !IsSingleDoubleVector(indexTerm)");
      return R_NilValue;
   }
   const double doubleIndex = REAL(indexTerm)[0];
   if(!IsDoubleToIntEbmIndexValid(doubleIndex)) {
      LOG_0(Trace_Error, "ERROR GetBestTermScores_R !IsDoubleToIntEbmIndexValid(doubleIndex)");
      return R_NilValue;
   }
   const size_t iTerm = static_cast<size_t>(doubleIndex);
   // we check that iTerm can be converted to size_t in IsDoubleToIntEbmIndexValid
   if(pBoosterCore->GetCountTerms() <= iTerm) {
      LOG_0(Trace_Error, "ERROR GetBestTermScores_R pBoosterCore->GetCountTerms() <= iTerm");
      return R_NilValue;
   }

   size_t cTensorScores = GetCountScores(pBoosterCore->GetCountClasses());
   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];
   const size_t cDimensions = pTerm->GetCountDimensions();
   if(0 != cDimensions) {
      const TermEntry * pTermEntry = pTerm->GetTermEntries();
      const TermEntry * const pTermEntriesEnd = &pTermEntry[cDimensions];
      do {
         const size_t cBins = pTermEntry->m_pFeature->GetCountBins();
         EBM_ASSERT(!IsMultiplyError(cTensorScores, cBins)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
         cTensorScores *= cBins;
         ++pTermEntry;
      } while(pTermEntriesEnd != pTermEntry);
   }
   if(IsConvertError<R_xlen_t>(cTensorScores)) {
      return R_NilValue;
   }
   SEXP ret = PROTECT(allocVector(REALSXP, static_cast<R_xlen_t>(cTensorScores)));
   EBM_ASSERT(!IsMultiplyError(sizeof(double), cTensorScores)); // we've allocated this memory, so it should be reachable, so these numbers should multiply

   error = GetBestTermScores(boosterHandle, static_cast<IntEbm>(iTerm), REAL(ret));

   UNPROTECT(1);

   if(Error_None != error) {
      LOG_0(Trace_Warning, "WARNING GetBestTermScores_R IntEbm { 0 } != error");
      return R_NilValue;
   }
   return ret;
}

SEXP GetCurrentTermScores_R(
   SEXP boosterHandleWrapped,
   SEXP indexTerm
) {
   EBM_ASSERT(nullptr != boosterHandleWrapped); // shouldn't be possible
   EBM_ASSERT(nullptr != indexTerm); // shouldn't be possible

   ErrorEbm error;

   if(EXTPTRSXP != TYPEOF(boosterHandleWrapped)) {
      LOG_0(Trace_Error, "ERROR GetCurrentTermScores_R EXTPTRSXP != TYPEOF(boosterHandleWrapped)");
      return R_NilValue;
   }
   const BoosterHandle boosterHandle = static_cast<BoosterHandle>(R_ExternalPtrAddr(boosterHandleWrapped));
   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return R_NilValue;
   }
   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();

   if(!IsSingleDoubleVector(indexTerm)) {
      LOG_0(Trace_Error, "ERROR GetCurrentTermScores_R !IsSingleDoubleVector(indexTerm)");
      return R_NilValue;
   }
   const double doubleIndex = REAL(indexTerm)[0];
   if(!IsDoubleToIntEbmIndexValid(doubleIndex)) {
      LOG_0(Trace_Error, "ERROR GetCurrentTermScores_R !IsDoubleToIntEbmIndexValid(doubleIndex)");
      return R_NilValue;
   }
   const size_t iTerm = static_cast<size_t>(doubleIndex);
   // we check that iTerm can be converted to size_t in IsDoubleToIntEbmIndexValid
   if(pBoosterCore->GetCountTerms() <= iTerm) {
      LOG_0(Trace_Error, "ERROR GetCurrentTermScores_R pBoosterCore->GetCountTerms() <= iTerm");
      return R_NilValue;
   }

   size_t cTensorScores = GetCountScores(pBoosterCore->GetCountClasses());
   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];
   const size_t cDimensions = pTerm->GetCountDimensions();
   if(0 != cDimensions) {
      const TermEntry * pTermEntry = pTerm->GetTermEntries();
      const TermEntry * const pTermEntriesEnd = &pTermEntry[cDimensions];
      do {
         const size_t cBins = pTermEntry->m_pFeature->GetCountBins();
         EBM_ASSERT(!IsMultiplyError(cTensorScores, cBins)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
         cTensorScores *= cBins;
         ++pTermEntry;
      } while(pTermEntriesEnd != pTermEntry);
   }
   if(IsConvertError<R_xlen_t>(cTensorScores)) {
      return R_NilValue;
   }
   SEXP ret = PROTECT(allocVector(REALSXP, static_cast<R_xlen_t>(cTensorScores)));
   EBM_ASSERT(!IsMultiplyError(sizeof(double), cTensorScores)); // we've allocated this memory, so it should be reachable, so these numbers should multiply

   error = GetCurrentTermScores(boosterHandle, static_cast<IntEbm>(iTerm), REAL(ret));

   UNPROTECT(1);

   if(Error_None != error) {
      LOG_0(Trace_Warning, "WARNING GetCurrentTermScores_R IntEbm { 0 } != error");
      return R_NilValue;
   }
   return ret;
}

SEXP FreeBooster_R(
   SEXP boosterHandleWrapped
) {
   BoostingFinalizer(boosterHandleWrapped);
   return R_NilValue;
}


SEXP CreateClassificationInteractionDetector_R(
   SEXP countClasses,
   SEXP featuresCategorical,
   SEXP featuresBinCount,
   SEXP binIndexes,
   SEXP targets,
   SEXP weights,
   SEXP initScores
) {
   EBM_ASSERT(nullptr != countClasses);
   EBM_ASSERT(nullptr != featuresCategorical);
   EBM_ASSERT(nullptr != featuresBinCount);
   EBM_ASSERT(nullptr != binIndexes);
   EBM_ASSERT(nullptr != targets);
   EBM_ASSERT(nullptr != weights);
   EBM_ASSERT(nullptr != initScores);

   ErrorEbm error;

   if(!IsSingleDoubleVector(countClasses)) {
      LOG_0(Trace_Error, "ERROR CreateClassificationInteractionDetector_R !IsSingleDoubleVector(countClasses)");
      return R_NilValue;
   }
   double countClassesDouble = REAL(countClasses)[0];
   if(!IsDoubleToIntEbmIndexValid(countClassesDouble)) {
      LOG_0(Trace_Error, "ERROR CreateClassificationInteractionDetector_R !IsDoubleToIntEbmIndexValid(countClassesDouble)");
      return R_NilValue;
   }
   const size_t cClasses = static_cast<size_t>(countClassesDouble);
   if(IsConvertError<ptrdiff_t>(cClasses)) {
      LOG_0(Trace_Error, "ERROR CreateClassificationInteractionDetector_R IsConvertError<ptrdiff_t>(cClasses)");
      return R_NilValue;
   }
   const size_t cScores = GetCountScores(static_cast<ptrdiff_t>(cClasses));

   size_t cFeatures;
   const BoolEbm * aFeaturesCategorical;
   if(ConvertLogicalsToBools(featuresCategorical, &cFeatures, &aFeaturesCategorical)) {
      // we've already logged any errors
      return R_NilValue;
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbm countFeatures = static_cast<IntEbm>(cFeatures);

   size_t cFeaturesFromBinCount;
   const IntEbm * aFeaturesBinCount;
   if(ConvertDoublesToIndexes(featuresBinCount, &cFeaturesFromBinCount, &aFeaturesBinCount)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cFeatures != cFeaturesFromBinCount) {
      LOG_0(Trace_Error, "ERROR CreateClassificationInteractionDetector_R cFeatures != cFeaturesFromBinCount");
      return R_NilValue;
   }

   size_t cBinIndexes;
   const IntEbm * aBinIndexes;
   if(ConvertDoublesToIndexes(binIndexes, &cBinIndexes, &aBinIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cSamples;
   const IntEbm * aTargets;
   if(ConvertDoublesToIndexes(targets, &cSamples, &aTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbm countSamples = static_cast<IntEbm>(cSamples);

   if(IsMultiplyError(cSamples, cFeatures)) {
      LOG_0(Trace_Error, "ERROR CreateClassificationInteractionDetector_R IsMultiplyError(cSamples, cFeatures)");
      return R_NilValue;
   }
   if(cSamples * cFeatures != cBinIndexes) {
      LOG_0(Trace_Error, "ERROR CreateClassificationInteractionDetector_R cSamples * cFeatures != cBinIndexes");
      return R_NilValue;
   }

   const IntEbm countInitScores = CountDoubles(initScores);
   if(countInitScores < 0) {
      // we've already logged any errors
      return R_NilValue;
   }
   size_t cInitScores = static_cast<size_t>(countInitScores);
   if(IsMultiplyError(cScores, cSamples)) {
      LOG_0(Trace_Error, "ERROR CreateClassificationInteractionDetector_R IsMultiplyError(cScores, cSamples)");
      return R_NilValue;
   }
   if(cScores * cSamples != cInitScores) {
      LOG_0(Trace_Error, "ERROR CreateClassificationInteractionDetector_R cScores * cSamples != cInitScores");
      return R_NilValue;
   }
   const double * const aInitScores = REAL(initScores);

   double * pWeights = nullptr;
   if(NILSXP != TYPEOF(weights)) {
      if(REALSXP != TYPEOF(weights)) {
         LOG_0(Trace_Error, "ERROR CreateClassificationInteractionDetector_R REALSXP != TYPEOF(weights)");
         return R_NilValue;
      }
      const R_xlen_t weightsLength = xlength(weights);
      if(IsConvertError<size_t>(weightsLength)) {
         LOG_0(Trace_Error, "ERROR CreateClassificationInteractionDetector_R IsConvertError<size_t>(weightsLength)");
         return R_NilValue;
      }
      const size_t cWeights = static_cast<size_t>(weightsLength);
      if(cWeights != cSamples) {
         LOG_0(Trace_Error, "ERROR CreateClassificationInteractionDetector_R cWeights != cSamples");
         return R_NilValue;
      }
      pWeights = REAL(weights);
   }

   InteractionHandle interactionHandle;
   error = CreateClassificationInteractionDetector(
      static_cast<IntEbm>(cClasses),
      countFeatures,
      aFeaturesCategorical,
      aFeaturesBinCount,
      countSamples,
      aBinIndexes,
      aTargets,
      pWeights,
      aInitScores,
      nullptr,
      &interactionHandle
   );

   if(Error_None != error || nullptr == interactionHandle) {
      return R_NilValue;
   }

   SEXP interactionHandleWrapped = R_MakeExternalPtr(static_cast<void *>(interactionHandle), R_NilValue, R_NilValue); // makes an EXTPTRSXP
   PROTECT(interactionHandleWrapped);

   R_RegisterCFinalizerEx(interactionHandleWrapped, &InteractionFinalizer, Rboolean::TRUE);

   UNPROTECT(1);
   return interactionHandleWrapped;
}

SEXP CreateRegressionInteractionDetector_R(
   SEXP featuresCategorical,
   SEXP featuresBinCount,
   SEXP binIndexes,
   SEXP targets,
   SEXP weights,
   SEXP initScores
) {
   EBM_ASSERT(nullptr != featuresCategorical);
   EBM_ASSERT(nullptr != featuresBinCount);
   EBM_ASSERT(nullptr != binIndexes);
   EBM_ASSERT(nullptr != targets);
   EBM_ASSERT(nullptr != weights);
   EBM_ASSERT(nullptr != initScores);

   ErrorEbm error;

   size_t cFeatures;
   const BoolEbm * aFeaturesCategorical;
   if(ConvertLogicalsToBools(featuresCategorical, &cFeatures, &aFeaturesCategorical)) {
      // we've already logged any errors
      return R_NilValue;
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbm countFeatures = static_cast<IntEbm>(cFeatures);

   size_t cFeaturesFromBinCount;
   const IntEbm * aFeaturesBinCount;
   if(ConvertDoublesToIndexes(featuresBinCount, &cFeaturesFromBinCount, &aFeaturesBinCount)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cFeatures != cFeaturesFromBinCount) {
      LOG_0(Trace_Error, "ERROR CreateRegressionInteractionDetector_R cFeatures != cFeaturesFromBinCount");
      return R_NilValue;
   }

   size_t cBinIndexes;
   const IntEbm * aBinIndexes;
   if(ConvertDoublesToIndexes(binIndexes, &cBinIndexes, &aBinIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }

   const IntEbm countSamples = CountDoubles(targets);
   if(countSamples < 0) {
      // we've already logged any errors
      return R_NilValue;
   }
   size_t cSamples = static_cast<size_t>(countSamples);
   if(IsMultiplyError(cSamples, cFeatures)) {
      LOG_0(Trace_Error, "ERROR CreateRegressionInteractionDetector_R IsMultiplyError(cSamples, cFeatures)");
      return R_NilValue;
   }
   if(cSamples * cFeatures != cBinIndexes) {
      LOG_0(Trace_Error, "ERROR CreateRegressionInteractionDetector_R cSamples * cFeatures != cBinIndexes");
      return R_NilValue;
   }
   const double * const aTargets = REAL(targets);

   const IntEbm countInitScores = CountDoubles(initScores);
   if(countInitScores < 0) {
      // we've already logged any errors
      return R_NilValue;
   }
   size_t cInitScores = static_cast<size_t>(countInitScores);
   if(cSamples != cInitScores) {
      LOG_0(Trace_Error, "ERROR CreateRegressionInteractionDetector_R cSamples != cInitScores");
      return R_NilValue;
   }
   const double * const aInitScores = REAL(initScores);

   double * pWeights = nullptr;
   if(NILSXP != TYPEOF(weights)) {
      if(REALSXP != TYPEOF(weights)) {
         LOG_0(Trace_Error, "ERROR CreateRegressionInteractionDetector_R REALSXP != TYPEOF(weights)");
         return R_NilValue;
      }
      const R_xlen_t weightsLength = xlength(weights);
      if(IsConvertError<size_t>(weightsLength)) {
         LOG_0(Trace_Error, "ERROR CreateRegressionInteractionDetector_R IsConvertError<size_t>(weightsLength)");
         return R_NilValue;
      }
      const size_t cWeights = static_cast<size_t>(weightsLength);
      if(cWeights != cSamples) {
         LOG_0(Trace_Error, "ERROR CreateRegressionInteractionDetector_R cWeights != cSamples");
         return R_NilValue;
      }
      pWeights = REAL(weights);
   }

   InteractionHandle interactionHandle;
   error = CreateRegressionInteractionDetector(
      countFeatures, 
      aFeaturesCategorical,
      aFeaturesBinCount,
      countSamples,
      aBinIndexes, 
      aTargets, 
      pWeights,
      aInitScores,
      nullptr,
      &interactionHandle
   );

   if(Error_None != error || nullptr == interactionHandle) {
      return R_NilValue;
   }

   SEXP interactionHandleWrapped = R_MakeExternalPtr(static_cast<void *>(interactionHandle), R_NilValue, R_NilValue); // makes an EXTPTRSXP
   PROTECT(interactionHandleWrapped);

   R_RegisterCFinalizerEx(interactionHandleWrapped, &InteractionFinalizer, Rboolean::TRUE);

   UNPROTECT(1);
   return interactionHandleWrapped;
}

SEXP CalcInteractionStrength_R(
   SEXP interactionHandleWrapped,
   SEXP featureIndexes,
   SEXP minSamplesLeaf
) {
   EBM_ASSERT(nullptr != interactionHandleWrapped); // shouldn't be possible
   EBM_ASSERT(nullptr != featureIndexes); // shouldn't be possible
   EBM_ASSERT(nullptr != minSamplesLeaf);

   if(EXTPTRSXP != TYPEOF(interactionHandleWrapped)) {
      LOG_0(Trace_Error, "ERROR CalcInteractionStrength_R EXTPTRSXP != TYPEOF(interactionHandleWrapped)");
      return R_NilValue;
   }
   const InteractionHandle interactionHandle = static_cast<InteractionHandle>(R_ExternalPtrAddr(interactionHandleWrapped));
   if(nullptr == interactionHandle) {
      LOG_0(Trace_Error, "ERROR CalcInteractionStrength_R nullptr == interactionHandle");
      return R_NilValue;
   }

   size_t cDimensions;
   const IntEbm * aFeatureIndexes;
   if(ConvertDoublesToIndexes(featureIndexes, &cDimensions, &aFeatureIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbm countDimensions = static_cast<IntEbm>(cDimensions);

   if(!IsSingleDoubleVector(minSamplesLeaf)) {
      LOG_0(Trace_Error, "ERROR CalcInteractionStrength_R !IsSingleDoubleVector(minSamplesLeaf)");
      return R_NilValue;
   }
   double doubleMinSamplesLeaf = REAL(minSamplesLeaf)[0];
   IntEbm minSamplesLeafEbm;
   static_assert(std::numeric_limits<double>::is_iec559, "we need is_iec559 to know that comparisons to infinity and -infinity to normal numbers work");
   if(std::isnan(doubleMinSamplesLeaf) || FLOAT64_TO_INT64_MAX < doubleMinSamplesLeaf) {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrength_R minSamplesLeaf overflow");
      minSamplesLeafEbm = FLOAT64_TO_INT64_MAX;
   } else if(doubleMinSamplesLeaf < IntEbm { 1 }) {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrength_R minSamplesLeaf can't be less than 1. Adjusting to 1.");
      minSamplesLeafEbm = 1;
   } else {
      minSamplesLeafEbm = static_cast<IntEbm>(doubleMinSamplesLeaf);
   }

   double avgInteractionStrength;
   if(Error_None != CalcInteractionStrength(interactionHandle, countDimensions, aFeatureIndexes, minSamplesLeafEbm, &avgInteractionStrength)) {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrength_R CalcInteractionStrength returned error code");
      return R_NilValue;
   }

   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = static_cast<double>(avgInteractionStrength);
   UNPROTECT(1);
   return ret;
}

SEXP FreeInteractionDetector_R(
   SEXP interactionHandleWrapped
) {
   InteractionFinalizer(interactionHandleWrapped);
   return R_NilValue;
}

static const R_CallMethodDef g_exposedFunctions[] = {
   { "GenerateSeed_R", (DL_FUNC)&GenerateSeed_R, 2 },
   { "CutQuantile_R", (DL_FUNC)&CutQuantile_R, 4 },
   { "BinFeature_R", (DL_FUNC)&BinFeature_R, 3 },
   { "SampleWithoutReplacement_R", (DL_FUNC)&SampleWithoutReplacement_R, 4 },
   { "CreateClassificationBooster_R", (DL_FUNC)&CreateClassificationBooster_R, 15 },
   { "CreateRegressionBooster_R", (DL_FUNC)&CreateRegressionBooster_R, 14 },
   { "GenerateTermUpdate_R", (DL_FUNC)&GenerateTermUpdate_R, 5 },
   { "ApplyTermUpdate_R", (DL_FUNC)&ApplyTermUpdate_R, 1 },
   { "GetBestTermScores_R", (DL_FUNC)&GetBestTermScores_R, 2 },
   { "GetCurrentTermScores_R", (DL_FUNC)& GetCurrentTermScores_R, 2 },
   { "FreeBooster_R", (DL_FUNC)& FreeBooster_R, 1 },
   { "CreateClassificationInteractionDetector_R", (DL_FUNC)&CreateClassificationInteractionDetector_R, 7 },
   { "CreateRegressionInteractionDetector_R", (DL_FUNC)&CreateRegressionInteractionDetector_R, 6 },
   { "CalcInteractionStrength_R", (DL_FUNC)&CalcInteractionStrength_R, 3 },
   { "FreeInteractionDetector_R", (DL_FUNC)&FreeInteractionDetector_R, 1 },
   { NULL, NULL, 0 }
};

} // DEFINED_ZONE_NAME

extern "C" {
   void attribute_visible R_init_interpret(DllInfo * info) {
      R_registerRoutines(info, NULL, DEFINED_ZONE_NAME::g_exposedFunctions, NULL, NULL);
      R_useDynamicSymbols(info, FALSE);
      R_forceSymbols(info, TRUE);
   }
} // extern "C"
