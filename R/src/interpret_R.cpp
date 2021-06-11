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

INLINE_ALWAYS FloatEbmType ConvertToEbmFloat(const double val) {
   // TODO : change this so that it doesn't execute anything if EbmFloat is a double
   if(val < double { std::numeric_limits<FloatEbmType>::lowest() }) {
      // if val is NaN it won't go here
      return -std::numeric_limits<FloatEbmType>::infinity();
   } else if(double { std::numeric_limits<FloatEbmType>::max() } < val) {
      // if val is NaN it won't go here
      return std::numeric_limits<FloatEbmType>::infinity();
   } else {
      // this works for NaN
      return static_cast<FloatEbmType>(val);
   }
}

INLINE_ALWAYS bool IsDoubleToIntEbmTypeIndexValid(const double val) {
   if(std::isnan(val)) {
      return false;
   }
   static_assert(std::numeric_limits<double>::is_iec559, "we need is_iec559 to know that comparisons to infinity and -infinity to normal numbers work");
   if(val < double { 0 }) {
      return false;
   }
   double maxValid = std::min(static_cast<double>(std::numeric_limits<size_t>::max()),
      std::min(double { R_XLEN_T_MAX }, static_cast<double>(std::numeric_limits<IntEbmType>::max())));
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

void ThreadStateBoostingFinalizer(SEXP threadStateBoostingHandleWrapped) {
   EBM_ASSERT(nullptr != threadStateBoostingHandleWrapped); // shouldn't be possible
   if(EXTPTRSXP == TYPEOF(threadStateBoostingHandleWrapped)) {
      const ThreadStateBoostingHandle threadStateBoostingHandle = static_cast<ThreadStateBoostingHandle>(R_ExternalPtrAddr(threadStateBoostingHandleWrapped));
      if(nullptr != threadStateBoostingHandle) {
         FreeThreadStateBoosting(threadStateBoostingHandle);
         R_ClearExternalPtr(threadStateBoostingHandleWrapped);
      }
   }
}

void InteractionFinalizer(SEXP interactionDetectorHandleWrapped) {
   EBM_ASSERT(nullptr != interactionDetectorHandleWrapped); // shouldn't be possible
   if(EXTPTRSXP == TYPEOF(interactionDetectorHandleWrapped)) {
      const InteractionDetectorHandle interactionDetectorHandle = static_cast<InteractionDetectorHandle>(R_ExternalPtrAddr(interactionDetectorHandleWrapped));
      if(nullptr != interactionDetectorHandle) {
         FreeInteractionDetector(interactionDetectorHandle);
         R_ClearExternalPtr(interactionDetectorHandleWrapped);
      }
   }
}

size_t CountFeatureGroupsFeatureIndexes(const size_t cFeatureGroups, const IntEbmType * const aFeatureGroupsDimensionCount) {
   EBM_ASSERT(nullptr != aFeatureGroupsDimensionCount);

   size_t cFeatureGroupsFeatureIndexes = size_t { 0 };
   if(0 != cFeatureGroups) {
      const IntEbmType * pFeatureGroupDimensionCount = aFeatureGroupsDimensionCount;
      const IntEbmType * const pFeatureGroupDimensionCountEnd = aFeatureGroupsDimensionCount + cFeatureGroups;
      do {
         const IntEbmType countDimensions = *pFeatureGroupDimensionCount;
         if(!IsNumberConvertable<size_t>(countDimensions)) {
            LOG_0(TraceLevelError, "ERROR CountFeatureGroupsFeatureIndexes !IsNumberConvertable<size_t>(countDimensions)");
            return SIZE_MAX;
         }
         const size_t cDimensions = static_cast<size_t>(countDimensions);
         if(IsAddError(cFeatureGroupsFeatureIndexes, cDimensions)) {
            LOG_0(TraceLevelError, "ERROR CountFeatureGroupsFeatureIndexes IsAddError(cFeatureGroupsFeatureIndexes, cDimensions)");
            return SIZE_MAX;
         }
         cFeatureGroupsFeatureIndexes += cDimensions;
         ++pFeatureGroupDimensionCount;
      } while(pFeatureGroupDimensionCountEnd != pFeatureGroupDimensionCount);
   }
   return cFeatureGroupsFeatureIndexes;
}

bool ConvertLogicalsToBools(const SEXP items, size_t * const pcItems, const BoolEbmType ** const pRet) {
   EBM_ASSERT(nullptr != items);
   EBM_ASSERT(nullptr != pcItems);
   EBM_ASSERT(nullptr != pRet);
   if(LGLSXP != TYPEOF(items)) {
      LOG_0(TraceLevelError, "ERROR ConvertLogicalsToBools LGLSXP != TYPEOF(items)");
      return true;
   }
   const R_xlen_t countItemsR = xlength(items);
   if(!IsNumberConvertable<size_t>(countItemsR)) {
      LOG_0(TraceLevelError, "ERROR ConvertLogicalsToBools !IsNumberConvertable<size_t>(countItemsR)");
      return true;
   }
   const size_t cItems = static_cast<size_t>(countItemsR);
   if(!IsNumberConvertable<IntEbmType>(cItems)) {
      LOG_0(TraceLevelError, "ERROR ConvertLogicalsToBools !IsNumberConvertable<IntEbmType>(cItems)");
      return true;
   }
   *pcItems = cItems;

   BoolEbmType * aItems = nullptr;
   if(0 != cItems) {
      aItems = reinterpret_cast<BoolEbmType *>(R_alloc(cItems, static_cast<int>(sizeof(BoolEbmType))));
      EBM_ASSERT(nullptr != aItems); // R_alloc doesn't return nullptr, so we don't need to check aItems
      BoolEbmType * pItem = aItems;
      const BoolEbmType * const pItemEnd = aItems + cItems;
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

bool ConvertDoublesToIndexes(const SEXP items, size_t * const pcItems, const IntEbmType ** const pRet) {
   EBM_ASSERT(nullptr != items);
   EBM_ASSERT(nullptr != pcItems);
   EBM_ASSERT(nullptr != pRet);
   if(REALSXP != TYPEOF(items)) {
      LOG_0(TraceLevelError, "ERROR ConvertDoublesToIndexes REALSXP != TYPEOF(items)");
      return true;
   }
   const R_xlen_t countItemsR = xlength(items);
   if(!IsNumberConvertable<size_t>(countItemsR)) {
      LOG_0(TraceLevelError, "ERROR ConvertDoublesToIndexes !IsNumberConvertable<size_t>(countItemsR)");
      return true;
   }
   const size_t cItems = static_cast<size_t>(countItemsR);
   if(!IsNumberConvertable<IntEbmType>(cItems)) {
      LOG_0(TraceLevelError, "ERROR ConvertDoublesToIndexes !IsNumberConvertable<IntEbmType>(cItems)");
      return true;
   }
   *pcItems = cItems;

   IntEbmType * aItems = nullptr;
   if(0 != cItems) {
      aItems = reinterpret_cast<IntEbmType *>(R_alloc(cItems, static_cast<int>(sizeof(IntEbmType))));
      EBM_ASSERT(nullptr != aItems); // R_alloc doesn't return nullptr, so we don't need to check aItems
      IntEbmType * pItem = aItems;
      const IntEbmType * const pItemEnd = aItems + cItems;
      const double * pOriginal = REAL(items);
      do {
         const double val = *pOriginal;
         if(!IsDoubleToIntEbmTypeIndexValid(val)) {
            LOG_0(TraceLevelError, "ERROR ConvertDoublesToIndexes !IsDoubleToIntEbmTypeIndexValid(val)");
            return true;
         }
         *pItem = static_cast<IntEbmType>(val);
         ++pOriginal;
         ++pItem;
      } while(pItemEnd != pItem);
   }
   *pRet = aItems;
   return false;
}

bool ConvertDoublesToEbmFloats(const SEXP items, size_t * const pcItems, const FloatEbmType ** const pRet) {
   EBM_ASSERT(nullptr != items);
   EBM_ASSERT(nullptr != pcItems);
   EBM_ASSERT(nullptr != pRet);
   if(REALSXP != TYPEOF(items)) {
      LOG_0(TraceLevelError, "ERROR ConvertDoublesToEbmFloats REALSXP != TYPEOF(items)");
      return true;
   }
   const R_xlen_t countItemsR = xlength(items);
   if(!IsNumberConvertable<size_t>(countItemsR)) {
      LOG_0(TraceLevelError, "ERROR ConvertDoublesToEbmFloats !IsNumberConvertable<size_t>(countItemsR)");
      return true;
   }
   const size_t cItems = static_cast<size_t>(countItemsR);
   if(!IsNumberConvertable<IntEbmType>(cItems)) {
      LOG_0(TraceLevelError, "ERROR ConvertDoublesToEbmFloats !IsNumberConvertable<IntEbmType>(cItems)");
      return true;
   }
   *pcItems = cItems;

   FloatEbmType * aItems = nullptr;
   if(0 != cItems) {
      aItems = reinterpret_cast<FloatEbmType *>(R_alloc(cItems, static_cast<int>(sizeof(FloatEbmType))));
      EBM_ASSERT(nullptr != aItems); // R_alloc doesn't return nullptr, so we don't need to check aItems
      FloatEbmType * pItem = aItems;
      const FloatEbmType * const pItemEnd = aItems + cItems;
      const double * pOriginal = REAL(items);
      do {
         const double val = *pOriginal;
         *pItem = ConvertToEbmFloat(val);
         ++pOriginal;
         ++pItem;
      } while(pItemEnd != pItem);
   }
   *pRet = aItems;
   return false;
}

SEXP GenerateRandomNumber_R(
   SEXP randomSeed,
   SEXP stageRandomizationMix
) {
   EBM_ASSERT(nullptr != randomSeed);
   EBM_ASSERT(nullptr != stageRandomizationMix);

   if(!IsSingleIntVector(randomSeed)) {
      LOG_0(TraceLevelError, "ERROR GenerateRandomNumber_R !IsSingleIntVector(randomSeed)");
      return R_NilValue;
   }
   const SeedEbmType randomSeedLocal = INTEGER(randomSeed)[0];

   if(!IsSingleIntVector(stageRandomizationMix)) {
      LOG_0(TraceLevelError, "ERROR GenerateRandomNumber_R !IsSingleIntVector(stageRandomizationMix)");
      return R_NilValue;
   }
   const SeedEbmType stageRandomizationMixLocal = INTEGER(stageRandomizationMix)[0];

   const SeedEbmType retSeed = GenerateRandomNumber(randomSeedLocal, stageRandomizationMixLocal);

   SEXP ret = PROTECT(allocVector(INTSXP, R_xlen_t { 1 }));
   INTEGER(ret)[0] = retSeed;
   UNPROTECT(1);
   return ret;
}

SEXP CutQuantile_R(
   SEXP featureValues,
   SEXP countSamplesPerBinMin,
   SEXP isHumanized,
   SEXP countCuts
) {
   EBM_ASSERT(nullptr != featureValues);
   EBM_ASSERT(nullptr != countSamplesPerBinMin);
   EBM_ASSERT(nullptr != isHumanized);
   EBM_ASSERT(nullptr != countCuts);

   size_t cFeatureValues;
   const FloatEbmType * aFeatureValues;
   if(ConvertDoublesToEbmFloats(featureValues, &cFeatureValues, &aFeatureValues)) {
      // we've already logged any errors
      return R_NilValue;
   }
   EBM_ASSERT(IsNumberConvertable<IntEbmType>(cFeatureValues)); // ConvertDoublesToEbmFloats checks this

   if(!IsSingleDoubleVector(countSamplesPerBinMin)) {
      LOG_0(TraceLevelError, "ERROR CutQuantile_R !IsSingleDoubleVector(countSamplesPerBinMin)");
      return R_NilValue;
   }
   const double countSamplesPerBinMinDouble = REAL(countSamplesPerBinMin)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(countSamplesPerBinMinDouble)) {
      LOG_0(TraceLevelError, "ERROR CutQuantile_R !IsDoubleToIntEbmTypeIndexValid(countSamplesPerBinMinDouble)");
      return R_NilValue;
   }
   const IntEbmType countSamplesPerBinMinIntEbmType = static_cast<IntEbmType>(countSamplesPerBinMinDouble);

   if(!IsSingleBoolVector(isHumanized)) {
      LOG_0(TraceLevelError, "ERROR CutQuantile_R !IsSingleBoolVector(isHumanized)");
      return R_NilValue;
   }

   const Rboolean isHumanizedR = static_cast<Rboolean>(LOGICAL(isHumanized)[0]);
   if(Rboolean::FALSE != isHumanizedR && Rboolean::TRUE != isHumanizedR) {
      LOG_0(TraceLevelError, "ERROR CutQuantile_R Rboolean::FALSE != isHumanizedR && Rboolean::TRUE != isHumanizedR");
      return R_NilValue;
   }
   const bool bHumanized = Rboolean::FALSE != isHumanizedR;

   if(!IsSingleDoubleVector(countCuts)) {
      LOG_0(TraceLevelError, "ERROR CutQuantile_R !IsSingleDoubleVector(countCuts)");
      return R_NilValue;
   }
   const double countCutsDouble = REAL(countCuts)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(countCutsDouble)) {
      LOG_0(TraceLevelError, "ERROR CutQuantile_R !IsDoubleToIntEbmTypeIndexValid(countCutsDouble)");
      return R_NilValue;
   }
   IntEbmType countCutsIntEbmType = static_cast<IntEbmType>(countCutsDouble);
   EBM_ASSERT(IsNumberConvertable<size_t>(countCutsIntEbmType)); // IsDoubleToIntEbmTypeIndexValid checks this

   FloatEbmType * const aCutsLowerBoundInclusive = reinterpret_cast<FloatEbmType *>(
      R_alloc(static_cast<size_t>(countCutsIntEbmType), static_cast<int>(sizeof(FloatEbmType))));
   EBM_ASSERT(nullptr != aCutsLowerBoundInclusive); // R_alloc doesn't return nullptr, so we don't need to check aItems

   const IntEbmType error = CutQuantile(
      static_cast<IntEbmType>(cFeatureValues),
      aFeatureValues,
      countSamplesPerBinMinIntEbmType,
      bHumanized ? EBM_TRUE : EBM_FALSE,
      &countCutsIntEbmType,
      aCutsLowerBoundInclusive,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr
   );

   if(0 != error) {
      return R_NilValue;
   }

   if(!IsNumberConvertable<R_xlen_t>(countCutsIntEbmType)) {
      return R_NilValue;
   }
   if(!IsNumberConvertable<size_t>(countCutsIntEbmType)) {
      return R_NilValue;
   }
   const SEXP ret = PROTECT(allocVector(REALSXP, static_cast<R_xlen_t>(countCutsIntEbmType)));

   const size_t cCutsIntEbmType = static_cast<size_t>(countCutsIntEbmType);

   // we've allocated this memory, so it should be reachable, so these numbers should multiply
   EBM_ASSERT(!IsMultiplyError(sizeof(*aCutsLowerBoundInclusive), cCutsIntEbmType));

   if(0 != cCutsIntEbmType) {
      double * pRet = REAL(ret);
      const FloatEbmType * pCutsLowerBoundInclusive = aCutsLowerBoundInclusive;
      const FloatEbmType * const pCutsLowerBoundInclusiveEnd = aCutsLowerBoundInclusive + cCutsIntEbmType;
      do {
         *pRet = static_cast<double>(*pCutsLowerBoundInclusive);
         ++pRet;
         ++pCutsLowerBoundInclusive;
      } while(pCutsLowerBoundInclusiveEnd != pCutsLowerBoundInclusive);
   }

   UNPROTECT(1);
   return ret;
}

SEXP Discretize_R(
   SEXP featureValues,
   SEXP cutsLowerBoundInclusive,
   SEXP discretizedOut
) {
   EBM_ASSERT(nullptr != featureValues);
   EBM_ASSERT(nullptr != cutsLowerBoundInclusive);
   EBM_ASSERT(nullptr != discretizedOut);

   size_t cFeatureValues;
   const FloatEbmType * aFeatureValues;
   if(ConvertDoublesToEbmFloats(featureValues, &cFeatureValues, &aFeatureValues)) {
      // we've already logged any errors
      return R_NilValue;
   }
   EBM_ASSERT(IsNumberConvertable<IntEbmType>(cFeatureValues)); // ConvertDoublesToEbmFloats checks this

   size_t cCuts;
   const FloatEbmType * aCutsLowerBoundInclusive;
   if(ConvertDoublesToEbmFloats(cutsLowerBoundInclusive, &cCuts, &aCutsLowerBoundInclusive)) {
      // we've already logged any errors
      return R_NilValue;
   }
   EBM_ASSERT(IsNumberConvertable<IntEbmType>(cCuts)); // ConvertDoublesToEbmFloats checks this

   if(REALSXP != TYPEOF(discretizedOut)) {
      LOG_0(TraceLevelError, "ERROR Discretize_R REALSXP != TYPEOF(discretizedOut)");
      return R_NilValue;
   }
   const R_xlen_t countDiscretizedOutR = xlength(discretizedOut);
   if(!IsNumberConvertable<size_t>(countDiscretizedOutR)) {
      LOG_0(TraceLevelError, "ERROR Discretize_R !IsNumberConvertable<size_t>(countDiscretizedOutR)");
      return R_NilValue;
   }
   const size_t cDiscretizedOut = static_cast<size_t>(countDiscretizedOutR);
   if(cFeatureValues != cDiscretizedOut) {
      LOG_0(TraceLevelError, "ERROR Discretize_R cFeatureValues != cDiscretizedOut");
      return R_NilValue;
   }

   if(0 != cFeatureValues) {
      IntEbmType * const aDiscretized = 
         reinterpret_cast<IntEbmType *>(R_alloc(cFeatureValues, static_cast<int>(sizeof(IntEbmType))));
      EBM_ASSERT(nullptr != aDiscretized); // this can't be nullptr since R_alloc uses R error handling

      if(0 != Discretize(
         static_cast<IntEbmType>(cFeatureValues),
         aFeatureValues,
         static_cast<IntEbmType>(cCuts),
         aCutsLowerBoundInclusive,
         aDiscretized
      )) {
         // we've already logged any errors
         return R_NilValue;
      }

      double * pDiscretizedOut = REAL(discretizedOut);
      const IntEbmType * pDiscretized = aDiscretized;
      const IntEbmType * const pDiscretizedEnd = aDiscretized + cFeatureValues;
      do {
         const IntEbmType val = *pDiscretized;
         *pDiscretizedOut = static_cast<double>(val);
         ++pDiscretizedOut;
         ++pDiscretized;
      } while(pDiscretizedEnd != pDiscretized);
   }

   // this return isn't useful beyond that it's not R_NilValue, which would signify error
   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = static_cast<double>(cFeatureValues);
   UNPROTECT(1);
   return ret;
}

SEXP SampleWithoutReplacement_R(
   SEXP randomSeed,
   SEXP countTrainingSamples,
   SEXP countValidationSamples,
   SEXP sampleCountsOut
) {
   EBM_ASSERT(nullptr != randomSeed);
   EBM_ASSERT(nullptr != countTrainingSamples);
   EBM_ASSERT(nullptr != countValidationSamples);
   EBM_ASSERT(nullptr != sampleCountsOut);

   if(!IsSingleIntVector(randomSeed)) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement_R !IsSingleIntVector(randomSeed)");
      return R_NilValue;
   }
   const SeedEbmType randomSeedLocal = INTEGER(randomSeed)[0];

   if(!IsSingleDoubleVector(countTrainingSamples)) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement_R !IsSingleDoubleVector(countTrainingSamples)");
      return R_NilValue;
   }
   const double countTrainingSamplesDouble = REAL(countTrainingSamples)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(countTrainingSamplesDouble)) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement_R !IsDoubleToIntEbmTypeIndexValid(countTrainingSamplesDouble)");
      return R_NilValue;
   }
   const IntEbmType countTrainingSamplesIntEbmType = static_cast<IntEbmType>(countTrainingSamplesDouble);
   EBM_ASSERT(IsNumberConvertable<size_t>(countTrainingSamplesIntEbmType)); // IsDoubleToIntEbmTypeIndexValid checks this

   if(!IsSingleDoubleVector(countValidationSamples)) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement_R !IsSingleDoubleVector(countValidationSamples)");
      return R_NilValue;
   }
   const double countValidationSamplesDouble = REAL(countValidationSamples)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(countValidationSamplesDouble)) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement_R !IsDoubleToIntEbmTypeIndexValid(countValidationSamplesDouble)");
      return R_NilValue;
   }
   IntEbmType countValidationSamplesIntEbmType = static_cast<IntEbmType>(countValidationSamplesDouble);
   EBM_ASSERT(IsNumberConvertable<size_t>(countValidationSamplesIntEbmType)); // IsDoubleToIntEbmTypeIndexValid checks this

   if(REALSXP != TYPEOF(sampleCountsOut)) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement_R REALSXP != TYPEOF(sampleCountsOut)");
      return R_NilValue;
   }
   const R_xlen_t sampleCountsOutR = xlength(sampleCountsOut);
   if(!IsNumberConvertable<size_t>(sampleCountsOutR)) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement_R !IsNumberConvertable<size_t>(sampleCountsOutR)");
      return R_NilValue;
   }
   const size_t cSampleCountsOut = static_cast<size_t>(sampleCountsOutR);
   if(static_cast<size_t>(countTrainingSamplesIntEbmType) + static_cast<size_t>(countValidationSamplesIntEbmType) != cSampleCountsOut) {
      LOG_0(TraceLevelError, "ERROR SampleWithoutReplacement_R static_cast<size_t>(countTrainingSamplesIntEbmType) + static_cast<size_t>(countValidationSamplesIntEbmType) != cSampleCountsOut");
      return R_NilValue;
   }

   if(0 != cSampleCountsOut) {
      IntEbmType * const aSampleCounts =
         reinterpret_cast<IntEbmType *>(R_alloc(cSampleCountsOut, static_cast<int>(sizeof(IntEbmType))));
      EBM_ASSERT(nullptr != aSampleCounts); // this can't be nullptr since R_alloc uses R error handling

      SampleWithoutReplacement(
         randomSeedLocal,
         countTrainingSamplesIntEbmType,
         countValidationSamplesIntEbmType,
         aSampleCounts
      );

      double * pSampleCountsOut = REAL(sampleCountsOut);
      const IntEbmType * pSampleCounts = aSampleCounts;
      const IntEbmType * const pSampleCountsEnd = aSampleCounts + cSampleCountsOut;
      do {
         const IntEbmType val = *pSampleCounts;
         *pSampleCountsOut = static_cast<double>(val);
         ++pSampleCountsOut;
         ++pSampleCounts;
      } while(pSampleCountsEnd != pSampleCounts);
   }

   // this return isn't useful beyond that it's not R_NilValue, which would signify error
   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = static_cast<double>(cSampleCountsOut);
   UNPROTECT(1);
   return ret;
}

SEXP CreateClassificationBooster_R(
   SEXP randomSeed,
   SEXP countTargetClasses,
   SEXP featuresCategorical,
   SEXP featuresBinCount,
   SEXP featureGroupsDimensionCount,
   SEXP featureGroupsFeatureIndexes,
   SEXP trainingBinnedData,
   SEXP trainingTargets,
   SEXP trainingWeights,
   SEXP trainingPredictorScores,
   SEXP validationBinnedData,
   SEXP validationTargets,
   SEXP validationWeights,
   SEXP validationPredictorScores,
   SEXP countInnerBags
) {
   EBM_ASSERT(nullptr != randomSeed);
   EBM_ASSERT(nullptr != countTargetClasses);
   EBM_ASSERT(nullptr != featuresCategorical);
   EBM_ASSERT(nullptr != featuresBinCount);
   EBM_ASSERT(nullptr != featureGroupsDimensionCount);
   EBM_ASSERT(nullptr != featureGroupsFeatureIndexes);
   EBM_ASSERT(nullptr != trainingBinnedData);
   EBM_ASSERT(nullptr != trainingTargets);
   EBM_ASSERT(nullptr != trainingWeights);
   EBM_ASSERT(nullptr != trainingPredictorScores);
   EBM_ASSERT(nullptr != validationBinnedData);
   EBM_ASSERT(nullptr != validationTargets);
   EBM_ASSERT(nullptr != validationWeights);
   EBM_ASSERT(nullptr != validationPredictorScores);
   EBM_ASSERT(nullptr != countInnerBags);

   if(!IsSingleIntVector(randomSeed)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R !IsSingleIntVector(randomSeed)");
      return R_NilValue;
   }
   const SeedEbmType randomSeedLocal = INTEGER(randomSeed)[0];

   if(!IsSingleDoubleVector(countTargetClasses)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R !IsSingleDoubleVector(countTargetClasses)");
      return R_NilValue;
   }
   double countTargetClassesDouble = REAL(countTargetClasses)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(countTargetClassesDouble)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R !IsDoubleToIntEbmTypeIndexValid(countTargetClassesDouble)");
      return R_NilValue;
   }
   EBM_ASSERT(IsNumberConvertable<size_t>(countTargetClassesDouble)); // IsDoubleToIntEbmTypeIndexValid checks this
   const size_t cTargetClasses = static_cast<size_t>(countTargetClassesDouble);
   if(!IsNumberConvertable<ptrdiff_t>(cTargetClasses)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R !IsNumberConvertable<ptrdiff_t>(cTargetClasses)");
      return R_NilValue;
   }
   const size_t cVectorLength = GetVectorLength(static_cast<ptrdiff_t>(cTargetClasses));

   size_t cFeatures;
   const BoolEbmType * aFeaturesCategorical;
   if(ConvertLogicalsToBools(featuresCategorical, &cFeatures, &aFeaturesCategorical)) {
      // we've already logged any errors
      return R_NilValue;
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbmType countFeatures = static_cast<IntEbmType>(cFeatures);

   size_t cFeaturesFromBinCount;
   const IntEbmType * aFeaturesBinCount;
   if(ConvertDoublesToIndexes(featuresBinCount, &cFeaturesFromBinCount, &aFeaturesBinCount)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cFeatures != cFeaturesFromBinCount) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R cFeatures != cFeaturesFromBinCount");
      return R_NilValue;
   }

   size_t cFeatureGroups;
   const IntEbmType * aFeatureGroupsDimensionCount;
   if(ConvertDoublesToIndexes(featureGroupsDimensionCount, &cFeatureGroups, &aFeatureGroupsDimensionCount)) {
      // we've already logged any errors
      return R_NilValue;
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbmType countFeatureGroups = static_cast<IntEbmType>(cFeatureGroups);

   const size_t cFeatureGroupsFeatureIndexesCheck = CountFeatureGroupsFeatureIndexes(cFeatureGroups, aFeatureGroupsDimensionCount);
   if(SIZE_MAX == cFeatureGroupsFeatureIndexesCheck) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cFeatureGroupsFeatureIndexesActual;
   const IntEbmType * aFeatureGroupsFeatureIndexes;
   if(ConvertDoublesToIndexes(featureGroupsFeatureIndexes, &cFeatureGroupsFeatureIndexesActual, &aFeatureGroupsFeatureIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cFeatureGroupsFeatureIndexesActual != cFeatureGroupsFeatureIndexesCheck) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R cFeatureGroupsFeatureIndexesActual != cFeatureGroupsFeatureIndexesCheck");
      return R_NilValue;
   }

   size_t cTrainingBinnedData;
   const IntEbmType * aTrainingBinnedData;
   if(ConvertDoublesToIndexes(trainingBinnedData, &cTrainingBinnedData, &aTrainingBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cTrainingSamples;
   const IntEbmType * aTrainingTargets;
   if(ConvertDoublesToIndexes(trainingTargets, &cTrainingSamples, &aTrainingTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countTrainingSamples = static_cast<IntEbmType>(cTrainingSamples);

   if(IsMultiplyError(cTrainingSamples, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R IsMultiplyError(cTrainingSamples, cFeatures)");
      return R_NilValue;
   }
   if(cTrainingSamples * cFeatures != cTrainingBinnedData) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R cTrainingSamples * cFeatures != cTrainingBinnedData");
      return R_NilValue;
   }

   size_t cTrainingPredictorScores;
   const FloatEbmType * aTrainingPredictorScores;
   if(ConvertDoublesToEbmFloats(trainingPredictorScores, &cTrainingPredictorScores, &aTrainingPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(IsMultiplyError(cTrainingSamples, cVectorLength)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R IsMultiplyError(cTrainingSamples, cVectorLength)");
      return R_NilValue;
   }
   if(cVectorLength * cTrainingSamples != cTrainingPredictorScores) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R cVectorLength * cTrainingSamples != cTrainingPredictorScores");
      return R_NilValue;
   }

   size_t cValidationBinnedData;
   const IntEbmType * aValidationBinnedData;
   if(ConvertDoublesToIndexes(validationBinnedData, &cValidationBinnedData, &aValidationBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cValidationSamples;
   const IntEbmType * aValidationTargets;
   if(ConvertDoublesToIndexes(validationTargets, &cValidationSamples, &aValidationTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countValidationSamples = static_cast<IntEbmType>(cValidationSamples);

   if(IsMultiplyError(cValidationSamples, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R IsMultiplyError(cValidationSamples, cFeatures)");
      return R_NilValue;
   }
   if(cValidationSamples * cFeatures != cValidationBinnedData) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R cValidationSamples * cFeatures != cValidationBinnedData");
      return R_NilValue;
   }

   size_t cValidationPredictorScores;
   const FloatEbmType * aValidationPredictorScores;
   if(ConvertDoublesToEbmFloats(validationPredictorScores, &cValidationPredictorScores, &aValidationPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(IsMultiplyError(cValidationSamples, cVectorLength)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R IsMultiplyError(cValidationSamples, cVectorLength)");
      return R_NilValue;
   }
   if(cVectorLength * cValidationSamples != cValidationPredictorScores) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R cVectorLength * cValidationSamples != cValidationPredictorScores");
      return R_NilValue;
   }

   if(!IsSingleIntVector(countInnerBags)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R !IsSingleIntVector(countInnerBags)");
      return R_NilValue;
   }
   int countInnerBagsInt = INTEGER(countInnerBags)[0];
   if(!IsNumberConvertable<IntEbmType>(countInnerBagsInt)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R !IsNumberConvertable<IntEbmType>(countInnerBagsInt)");
      return nullptr;
   }
   IntEbmType countInnerBagsLocal = static_cast<IntEbmType>(countInnerBagsInt);

   double * pTrainingWeights = nullptr;
   double * pValidationWeights = nullptr;
   if(NILSXP != TYPEOF(trainingWeights) || NILSXP != TYPEOF(validationWeights)) {
      if(REALSXP != TYPEOF(trainingWeights)) {
         LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R REALSXP != TYPEOF(trainingWeights)");
         return R_NilValue;
      }
      R_xlen_t trainingWeightsLength = xlength(trainingWeights);
      if(!IsNumberConvertable<size_t>(trainingWeightsLength)) {
         LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R !IsNumberConvertable<size_t>(trainingWeightsLength)");
         return R_NilValue;
      }
      size_t cTrainingWeights = static_cast<size_t>(trainingWeightsLength);
      if(cTrainingWeights != cTrainingSamples) {
         LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R cTrainingWeights != cTrainingSamples");
         return R_NilValue;
      }
      pTrainingWeights = REAL(trainingWeights);

      if(REALSXP != TYPEOF(validationWeights)) {
         LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R REALSXP != TYPEOF(validationWeights)");
         return R_NilValue;
      }
      R_xlen_t validationWeightsLength = xlength(validationWeights);
      if(!IsNumberConvertable<size_t>(validationWeightsLength)) {
         LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R !IsNumberConvertable<size_t>(validationWeightsLength)");
         return R_NilValue;
      }
      size_t cValidationWeights = static_cast<size_t>(validationWeightsLength);
      if(cValidationWeights != cValidationSamples) {
         LOG_0(TraceLevelError, "ERROR CreateClassificationBooster_R cValidationWeights != cValidationSamples");
         return R_NilValue;
      }
      pValidationWeights = REAL(validationWeights);
   }

   const BoosterHandle boosterHandle = CreateClassificationBooster(
      randomSeedLocal,
      static_cast<IntEbmType>(cTargetClasses),
      countFeatures, 
      aFeaturesCategorical,
      aFeaturesBinCount,
      countFeatureGroups, 
      aFeatureGroupsDimensionCount,
      aFeatureGroupsFeatureIndexes,
      countTrainingSamples, 
      aTrainingBinnedData, 
      aTrainingTargets, 
      pTrainingWeights,
      aTrainingPredictorScores,
      countValidationSamples, 
      aValidationBinnedData, 
      aValidationTargets, 
      pValidationWeights,
      aValidationPredictorScores,
      countInnerBagsLocal, 
      nullptr
   );

   if(nullptr == boosterHandle) {
      return R_NilValue;
   }

   SEXP boosterHandleWrapped = R_MakeExternalPtr(static_cast<void *>(boosterHandle), R_NilValue, R_NilValue); // makes an EXTPTRSXP
   PROTECT(boosterHandleWrapped);

   R_RegisterCFinalizerEx(boosterHandleWrapped, &BoostingFinalizer, Rboolean::TRUE);

   UNPROTECT(1);
   return boosterHandleWrapped;
}

SEXP CreateRegressionBooster_R(
   SEXP randomSeed,
   SEXP featuresCategorical,
   SEXP featuresBinCount,
   SEXP featureGroupsDimensionCount,
   SEXP featureGroupsFeatureIndexes,
   SEXP trainingBinnedData,
   SEXP trainingTargets,
   SEXP trainingWeights,
   SEXP trainingPredictorScores,
   SEXP validationBinnedData,
   SEXP validationTargets,
   SEXP validationWeights,
   SEXP validationPredictorScores,
   SEXP countInnerBags
) {
   EBM_ASSERT(nullptr != randomSeed);
   EBM_ASSERT(nullptr != featuresCategorical);
   EBM_ASSERT(nullptr != featuresBinCount);
   EBM_ASSERT(nullptr != featureGroupsDimensionCount);
   EBM_ASSERT(nullptr != featureGroupsFeatureIndexes);
   EBM_ASSERT(nullptr != trainingBinnedData);
   EBM_ASSERT(nullptr != trainingTargets);
   EBM_ASSERT(nullptr != trainingWeights);
   EBM_ASSERT(nullptr != trainingPredictorScores);
   EBM_ASSERT(nullptr != validationBinnedData);
   EBM_ASSERT(nullptr != validationTargets);
   EBM_ASSERT(nullptr != validationWeights);
   EBM_ASSERT(nullptr != validationPredictorScores);
   EBM_ASSERT(nullptr != countInnerBags);

   if(!IsSingleIntVector(randomSeed)) {
      LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R !IsSingleIntVector(randomSeed)");
      return R_NilValue;
   }
   const SeedEbmType randomSeedLocal = INTEGER(randomSeed)[0];

   size_t cFeatures;
   const BoolEbmType * aFeaturesCategorical;
   if(ConvertLogicalsToBools(featuresCategorical, &cFeatures, &aFeaturesCategorical)) {
      // we've already logged any errors
      return R_NilValue;
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbmType countFeatures = static_cast<IntEbmType>(cFeatures);

   size_t cFeaturesFromBinCount;
   const IntEbmType * aFeaturesBinCount;
   if(ConvertDoublesToIndexes(featuresBinCount, &cFeaturesFromBinCount, &aFeaturesBinCount)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cFeatures != cFeaturesFromBinCount) {
      LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R cFeatures != cFeaturesFromBinCount");
      return R_NilValue;
   }

   size_t cFeatureGroups;
   const IntEbmType * aFeatureGroupsDimensionCount;
   if(ConvertDoublesToIndexes(featureGroupsDimensionCount, &cFeatureGroups, &aFeatureGroupsDimensionCount)) {
      // we've already logged any errors
      return R_NilValue;
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbmType countFeatureGroups = static_cast<IntEbmType>(cFeatureGroups);

   const size_t cFeatureGroupsFeatureIndexesCheck = CountFeatureGroupsFeatureIndexes(cFeatureGroups, aFeatureGroupsDimensionCount);
   if(SIZE_MAX == cFeatureGroupsFeatureIndexesCheck) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cFeatureGroupsFeatureIndexesActual;
   const IntEbmType * aFeatureGroupsFeatureIndexes;
   if(ConvertDoublesToIndexes(featureGroupsFeatureIndexes, &cFeatureGroupsFeatureIndexesActual, &aFeatureGroupsFeatureIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cFeatureGroupsFeatureIndexesActual != cFeatureGroupsFeatureIndexesCheck) {
      LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R cFeatureGroupsFeatureIndexesActual != cFeatureGroupsFeatureIndexesCheck");
      return R_NilValue;
   }

   size_t cTrainingBinnedData;
   const IntEbmType * aTrainingBinnedData;
   if(ConvertDoublesToIndexes(trainingBinnedData, &cTrainingBinnedData, &aTrainingBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cTrainingSamples;
   const FloatEbmType * aTrainingTargets;
   if(ConvertDoublesToEbmFloats(trainingTargets, &cTrainingSamples, &aTrainingTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countTrainingSamples = static_cast<IntEbmType>(cTrainingSamples);

   if(IsMultiplyError(cTrainingSamples, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R IsMultiplyError(cTrainingSamples, cFeatures)");
      return R_NilValue;
   }
   if(cTrainingSamples * cFeatures != cTrainingBinnedData) {
      LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R cTrainingSamples * cFeatures != cTrainingBinnedData");
      return R_NilValue;
   }

   size_t cTrainingPredictorScores;
   const FloatEbmType * aTrainingPredictorScores;
   if(ConvertDoublesToEbmFloats(trainingPredictorScores, &cTrainingPredictorScores, &aTrainingPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cTrainingSamples != cTrainingPredictorScores) {
      LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R cTrainingSamples != cTrainingPredictorScores");
      return R_NilValue;
   }

   size_t cValidationBinnedData;
   const IntEbmType * aValidationBinnedData;
   if(ConvertDoublesToIndexes(validationBinnedData, &cValidationBinnedData, &aValidationBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cValidationSamples;
   const FloatEbmType * aValidationTargets;
   if(ConvertDoublesToEbmFloats(validationTargets, &cValidationSamples, &aValidationTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countValidationSamples = static_cast<IntEbmType>(cValidationSamples);

   if(IsMultiplyError(cValidationSamples, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R IsMultiplyError(cValidationSamples, cFeatures)");
      return R_NilValue;
   }
   if(cValidationSamples * cFeatures != cValidationBinnedData) {
      LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R cValidationSamples * cFeatures != cValidationBinnedData");
      return R_NilValue;
   }

   size_t cValidationPredictorScores;
   const FloatEbmType * aValidationPredictorScores;
   if(ConvertDoublesToEbmFloats(validationPredictorScores, &cValidationPredictorScores, &aValidationPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cValidationSamples != cValidationPredictorScores) {
      LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R cValidationSamples != cValidationPredictorScores");
      return R_NilValue;
   }

   if(!IsSingleIntVector(countInnerBags)) {
      LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R !IsSingleIntVector(countInnerBags)");
      return R_NilValue;
   }
   int countInnerBagsInt = INTEGER(countInnerBags)[0];
   if(!IsNumberConvertable<IntEbmType>(countInnerBagsInt)) {
      LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R !IsNumberConvertable<IntEbmType>(countInnerBagsInt)");
      return nullptr;
   }
   IntEbmType countInnerBagsLocal = static_cast<IntEbmType>(countInnerBagsInt);

   double * pTrainingWeights = nullptr;
   double * pValidationWeights = nullptr;
   if(NILSXP != TYPEOF(trainingWeights) || NILSXP != TYPEOF(validationWeights)) {
      if(REALSXP != TYPEOF(trainingWeights)) {
         LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R REALSXP != TYPEOF(trainingWeights)");
         return R_NilValue;
      }
      R_xlen_t trainingWeightsLength = xlength(trainingWeights);
      if(!IsNumberConvertable<size_t>(trainingWeightsLength)) {
         LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R !IsNumberConvertable<size_t>(trainingWeightsLength)");
         return R_NilValue;
      }
      size_t cTrainingWeights = static_cast<size_t>(trainingWeightsLength);
      if(cTrainingWeights != cTrainingSamples) {
         LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R cTrainingWeights != cTrainingSamples");
         return R_NilValue;
      }
      pTrainingWeights = REAL(trainingWeights);

      if(REALSXP != TYPEOF(validationWeights)) {
         LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R REALSXP != TYPEOF(validationWeights)");
         return R_NilValue;
      }
      R_xlen_t validationWeightsLength = xlength(validationWeights);
      if(!IsNumberConvertable<size_t>(validationWeightsLength)) {
         LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R !IsNumberConvertable<size_t>(validationWeightsLength)");
         return R_NilValue;
      }
      size_t cValidationWeights = static_cast<size_t>(validationWeightsLength);
      if(cValidationWeights != cValidationSamples) {
         LOG_0(TraceLevelError, "ERROR CreateRegressionBooster_R cValidationWeights != cValidationSamples");
         return R_NilValue;
      }
      pValidationWeights = REAL(validationWeights);
   }

   const BoosterHandle boosterHandle = CreateRegressionBooster(
      randomSeedLocal,
      countFeatures,
      aFeaturesCategorical,
      aFeaturesBinCount,
      countFeatureGroups,
      aFeatureGroupsDimensionCount,
      aFeatureGroupsFeatureIndexes,
      countTrainingSamples, 
      aTrainingBinnedData, 
      aTrainingTargets, 
      pTrainingWeights, 
      aTrainingPredictorScores,
      countValidationSamples, 
      aValidationBinnedData, 
      aValidationTargets, 
      pValidationWeights, 
      aValidationPredictorScores,
      countInnerBagsLocal, 
      nullptr
   );

   if(nullptr == boosterHandle) {
      return R_NilValue;
   }

   SEXP boosterHandleWrapped = R_MakeExternalPtr(static_cast<void *>(boosterHandle), R_NilValue, R_NilValue); // makes an EXTPTRSXP
   PROTECT(boosterHandleWrapped);

   R_RegisterCFinalizerEx(boosterHandleWrapped, &BoostingFinalizer, Rboolean::TRUE);

   UNPROTECT(1);
   return boosterHandleWrapped;
}

SEXP GenerateModelUpdate_R(
   SEXP threadStateBoostingHandleWrapped,
   SEXP indexFeatureGroup,
   SEXP learningRate,
   SEXP countSamplesRequiredForChildSplitMin,
   SEXP leavesMax
) {
   EBM_ASSERT(nullptr != threadStateBoostingHandleWrapped);
   EBM_ASSERT(nullptr != indexFeatureGroup);
   EBM_ASSERT(nullptr != learningRate);
   EBM_ASSERT(nullptr != countSamplesRequiredForChildSplitMin);
   EBM_ASSERT(nullptr != leavesMax);

   if(EXTPTRSXP != TYPEOF(threadStateBoostingHandleWrapped)) {
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate_R EXTPTRSXP != TYPEOF(threadStateBoostingHandleWrapped)");
      return R_NilValue;
   }
   BoosterShell * pBoosterShell = static_cast<BoosterShell *>(R_ExternalPtrAddr(threadStateBoostingHandleWrapped));
   if(nullptr == pBoosterShell) {
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate_R nullptr == pBoosterShell");
      return R_NilValue;
   }
   if(!pBoosterShell->IsValid()) {
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate_R !pBoosterShell->IsValid()");
      return R_NilValue;
   }

   if(!IsSingleDoubleVector(indexFeatureGroup)) {
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate_R !IsSingleDoubleVector(indexFeatureGroup)");
      return R_NilValue;
   }
   double doubleIndex = REAL(indexFeatureGroup)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(doubleIndex)) {
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate_R !IsDoubleToIntEbmTypeIndexValid(doubleIndex)");
      return R_NilValue;
   }
   const size_t iFeatureGroup = static_cast<size_t>(doubleIndex);

   if(!IsSingleDoubleVector(learningRate)) {
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate_R !IsSingleDoubleVector(learningRate)");
      return R_NilValue;
   }
   const FloatEbmType learningRateLocal = ConvertToEbmFloat(REAL(learningRate)[0]);

   if(!IsSingleDoubleVector(countSamplesRequiredForChildSplitMin)) {
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate_R !IsSingleDoubleVector(countSamplesRequiredForChildSplitMin)");
      return R_NilValue;
   }
   double doubleCountSamplesRequiredForChildSplitMin = REAL(countSamplesRequiredForChildSplitMin)[0];
   IntEbmType countEbmSamplesRequiredForChildSplitMin;
   static_assert(std::numeric_limits<double>::is_iec559, "we need is_iec559 to know that comparisons to infinity and -infinity to normal numbers work");
   if(std::isnan(doubleCountSamplesRequiredForChildSplitMin) ||
      static_cast<double>(std::numeric_limits<IntEbmType>::max()) < doubleCountSamplesRequiredForChildSplitMin
      ) {
      LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdate_R countSamplesRequiredForChildSplitMin overflow");
      countEbmSamplesRequiredForChildSplitMin = std::numeric_limits<IntEbmType>::max();
   } else if(doubleCountSamplesRequiredForChildSplitMin < static_cast<double>(std::numeric_limits<IntEbmType>::lowest())) {
      LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdate_R countSamplesRequiredForChildSplitMin underflow");
      countEbmSamplesRequiredForChildSplitMin = std::numeric_limits<IntEbmType>::lowest();
   } else {
      countEbmSamplesRequiredForChildSplitMin = static_cast<IntEbmType>(doubleCountSamplesRequiredForChildSplitMin);
   }

   size_t cDimensions;
   const IntEbmType * aLeavesMax;
   if(ConvertDoublesToIndexes(leavesMax, &cDimensions, &aLeavesMax)) {
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate_R ConvertDoublesToIndexes(leavesMax, &cDimensions, &aLeavesMax)");
      return R_NilValue;
   }
   if(pBoosterShell->GetBoosterCore()->GetCountFeatureGroups() <= iFeatureGroup) {
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate_R pBoosterShell->GetBoosterCore()->GetCountFeatureGroups() <= iFeatureGroup");
      return R_NilValue;
   }
   if(cDimensions < pBoosterShell->GetBoosterCore()->GetFeatureGroups()[iFeatureGroup]->GetCountDimensions()) {
      LOG_0(TraceLevelError, "ERROR GenerateModelUpdate_R cDimensions < pBoosterShell->GetBoosterCore()->GetFeatureGroups()[iFeatureGroup]->GetCountDimensions()");
      return R_NilValue;
   }

   FloatEbmType gainOut;
   if(0 != GenerateModelUpdate(
      reinterpret_cast<ThreadStateBoostingHandle>(pBoosterShell),
      static_cast<IntEbmType>(iFeatureGroup),
      GenerateUpdateOptions_Default,
      learningRateLocal,
      countEbmSamplesRequiredForChildSplitMin,
      aLeavesMax,
      &gainOut
   )) {
      LOG_0(TraceLevelWarning, "WARNING GenerateModelUpdate_R BoostingStep returned error code");
      return R_NilValue;
   }

   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = static_cast<double>(gainOut);
   UNPROTECT(1);
   return ret;
}

SEXP ApplyModelUpdate_R(
   SEXP threadStateBoostingHandleWrapped
) {
   EBM_ASSERT(nullptr != threadStateBoostingHandleWrapped);

   if(EXTPTRSXP != TYPEOF(threadStateBoostingHandleWrapped)) {
      LOG_0(TraceLevelError, "ERROR ApplyModelUpdate_R EXTPTRSXP != TYPEOF(threadStateBoostingHandleWrapped)");
      return R_NilValue;
   }
   BoosterShell * pBoosterShell = static_cast<BoosterShell *>(R_ExternalPtrAddr(threadStateBoostingHandleWrapped));
   if(nullptr == pBoosterShell) {
      LOG_0(TraceLevelError, "ERROR ApplyModelUpdate_R nullptr == pBoosterShell");
      return R_NilValue;
   }

   FloatEbmType validationMetricOut;
   if(0 != ApplyModelUpdate(
      reinterpret_cast<ThreadStateBoostingHandle>(pBoosterShell),
      &validationMetricOut
   )) {
      LOG_0(TraceLevelWarning, "WARNING ApplyModelUpdate_R ApplyModelUpdate returned error code");
      return R_NilValue;
   }

   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = static_cast<double>(validationMetricOut);
   UNPROTECT(1);
   return ret;
}

SEXP GetBestModelFeatureGroup_R(
   SEXP boosterHandleWrapped,
   SEXP indexFeatureGroup
) {
   EBM_ASSERT(nullptr != boosterHandleWrapped); // shouldn't be possible
   EBM_ASSERT(nullptr != indexFeatureGroup); // shouldn't be possible

   if(EXTPTRSXP != TYPEOF(boosterHandleWrapped)) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup_R EXTPTRSXP != TYPEOF(boosterHandleWrapped)");
      return R_NilValue;
   }
   BoosterCore * const pBoosterCore = static_cast<BoosterCore *>(R_ExternalPtrAddr(boosterHandleWrapped));
   if(nullptr == pBoosterCore) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup_R nullptr == pBoosterCore");
      return R_NilValue;
   }

   if(!IsSingleDoubleVector(indexFeatureGroup)) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup_R !IsSingleDoubleVector(indexFeatureGroup)");
      return R_NilValue;
   }
   const double doubleIndex = REAL(indexFeatureGroup)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(doubleIndex)) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup_R !IsDoubleToIntEbmTypeIndexValid(doubleIndex)");
      return R_NilValue;
   }
   const size_t iFeatureGroup = static_cast<size_t>(doubleIndex);
   // we check that iFeatureGroup can be converted to size_t in IsDoubleToIntEbmTypeIndexValid
   if(pBoosterCore->GetCountFeatureGroups() <= iFeatureGroup) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup_R pBoosterCore->GetCountFeatureGroups() <= iFeatureGroup");
      return R_NilValue;
   }

   size_t cValues = GetVectorLength(pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses());
   const FeatureGroup * const pFeatureGroup = pBoosterCore->GetFeatureGroups()[iFeatureGroup];
   const size_t cDimensions = pFeatureGroup->GetCountDimensions();
   if(0 != cDimensions) {
      const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
      const FeatureGroupEntry * const pFeatureGroupEntryEnd = &pFeatureGroupEntry[cDimensions];
      do {
         const size_t cBins = pFeatureGroupEntry->m_pFeature->GetCountBins();
         EBM_ASSERT(!IsMultiplyError(cBins, cValues)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
         cValues *= cBins;
         ++pFeatureGroupEntry;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
   }
   if(!IsNumberConvertable<R_xlen_t>(cValues)) {
      return R_NilValue;
   }
   SEXP ret = PROTECT(allocVector(REALSXP, static_cast<R_xlen_t>(cValues)));
   EBM_ASSERT(!IsMultiplyError(sizeof(double), cValues)); // we've allocated this memory, so it should be reachable, so these numbers should multiply

   const IntEbmType error = GetBestModelFeatureGroup(reinterpret_cast<BoosterHandle>(pBoosterCore), static_cast<IntEbmType>(iFeatureGroup), REAL(ret));

   UNPROTECT(1);

   if(IntEbmType { 0 } != error) {
      LOG_0(TraceLevelWarning, "WARNING GetBestModelFeatureGroup_R IntEbmType { 0 } != error");
      return R_NilValue;
   }
   return ret;
}

SEXP GetCurrentModelFeatureGroup_R(
   SEXP boosterHandleWrapped,
   SEXP indexFeatureGroup
) {
   EBM_ASSERT(nullptr != boosterHandleWrapped); // shouldn't be possible
   EBM_ASSERT(nullptr != indexFeatureGroup); // shouldn't be possible

   if(EXTPTRSXP != TYPEOF(boosterHandleWrapped)) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup_R EXTPTRSXP != TYPEOF(boosterHandleWrapped)");
      return R_NilValue;
   }
   BoosterCore * const pBoosterCore = static_cast<BoosterCore *>(R_ExternalPtrAddr(boosterHandleWrapped));
   if(nullptr == pBoosterCore) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup_R nullptr == pBoosterCore");
      return R_NilValue;
   }

   if(!IsSingleDoubleVector(indexFeatureGroup)) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup_R !IsSingleDoubleVector(indexFeatureGroup)");
      return R_NilValue;
   }
   const double doubleIndex = REAL(indexFeatureGroup)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(doubleIndex)) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup_R !IsDoubleToIntEbmTypeIndexValid(doubleIndex)");
      return R_NilValue;
   }
   const size_t iFeatureGroup = static_cast<size_t>(doubleIndex);
   // we check that iFeatureGroup can be converted to size_t in IsDoubleToIntEbmTypeIndexValid
   if(pBoosterCore->GetCountFeatureGroups() <= iFeatureGroup) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup_R pBoosterCore->GetCountFeatureGroups() <= iFeatureGroup");
      return R_NilValue;
   }

   size_t cValues = GetVectorLength(pBoosterCore->GetRuntimeLearningTypeOrCountTargetClasses());
   const FeatureGroup * const pFeatureGroup = pBoosterCore->GetFeatureGroups()[iFeatureGroup];
   const size_t cDimensions = pFeatureGroup->GetCountDimensions();
   if(0 != cDimensions) {
      const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
      const FeatureGroupEntry * const pFeatureGroupEntryEnd = &pFeatureGroupEntry[cDimensions];
      do {
         const size_t cBins = pFeatureGroupEntry->m_pFeature->GetCountBins();
         EBM_ASSERT(!IsMultiplyError(cBins, cValues)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
         cValues *= cBins;
         ++pFeatureGroupEntry;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
   }
   if(!IsNumberConvertable<R_xlen_t>(cValues)) {
      return R_NilValue;
   }
   SEXP ret = PROTECT(allocVector(REALSXP, static_cast<R_xlen_t>(cValues)));
   EBM_ASSERT(!IsMultiplyError(sizeof(double), cValues)); // we've allocated this memory, so it should be reachable, so these numbers should multiply

   const IntEbmType error = GetCurrentModelFeatureGroup(reinterpret_cast<BoosterHandle>(pBoosterCore), static_cast<IntEbmType>(iFeatureGroup), REAL(ret));

   UNPROTECT(1);

   if(IntEbmType { 0 } != error) {
      LOG_0(TraceLevelWarning, "WARNING GetCurrentModelFeatureGroup_R IntEbmType { 0 } != error");
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

SEXP CreateThreadStateBoosting_R(
   SEXP boosterHandleWrapped
) {
   EBM_ASSERT(nullptr != boosterHandleWrapped); // shouldn't be possible

   if(EXTPTRSXP != TYPEOF(boosterHandleWrapped)) {
      LOG_0(TraceLevelError, "ERROR CreateThreadStateBoosting_R EXTPTRSXP != TYPEOF(boosterHandleWrapped)");
      return R_NilValue;
   }
   const BoosterHandle boosterHandle = static_cast<BoosterHandle>(R_ExternalPtrAddr(boosterHandleWrapped));
   if(nullptr == boosterHandle) {
      LOG_0(TraceLevelError, "ERROR CreateThreadStateBoosting_R nullptr == boosterHandle");
      return R_NilValue;
   }

   const ThreadStateBoostingHandle threadStateBoostingHandle = CreateThreadStateBoosting(boosterHandle);
   if(nullptr == threadStateBoostingHandle) {
      return R_NilValue;
   }

   SEXP threadStateBoostingHandleWrapped = R_MakeExternalPtr(static_cast<void *>(threadStateBoostingHandle), R_NilValue, R_NilValue); // makes an EXTPTRSXP
   PROTECT(threadStateBoostingHandleWrapped);

   R_RegisterCFinalizerEx(threadStateBoostingHandleWrapped, &ThreadStateBoostingFinalizer, Rboolean::TRUE);

   UNPROTECT(1);
   return threadStateBoostingHandleWrapped;
}

SEXP FreeThreadStateBoosting_R(
   SEXP threadStateBoostingHandleWrapped
) {
   ThreadStateBoostingFinalizer(threadStateBoostingHandleWrapped);
   return R_NilValue;
}


SEXP CreateClassificationInteractionDetector_R(
   SEXP countTargetClasses,
   SEXP featuresCategorical,
   SEXP featuresBinCount,
   SEXP binnedData,
   SEXP targets,
   SEXP weights,
   SEXP predictorScores
) {
   EBM_ASSERT(nullptr != countTargetClasses);
   EBM_ASSERT(nullptr != featuresCategorical);
   EBM_ASSERT(nullptr != featuresBinCount);
   EBM_ASSERT(nullptr != binnedData);
   EBM_ASSERT(nullptr != targets);
   EBM_ASSERT(nullptr != weights);
   EBM_ASSERT(nullptr != predictorScores);

   if(!IsSingleDoubleVector(countTargetClasses)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationInteractionDetector_R !IsSingleDoubleVector(countTargetClasses)");
      return R_NilValue;
   }
   double countTargetClassesDouble = REAL(countTargetClasses)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(countTargetClassesDouble)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationInteractionDetector_R !IsDoubleToIntEbmTypeIndexValid(countTargetClassesDouble)");
      return R_NilValue;
   }
   const size_t cTargetClasses = static_cast<size_t>(countTargetClassesDouble);
   if(!IsNumberConvertable<ptrdiff_t>(cTargetClasses)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationInteractionDetector_R !IsNumberConvertable<ptrdiff_t>(cTargetClasses)");
      return R_NilValue;
   }
   const size_t cVectorLength = GetVectorLength(static_cast<ptrdiff_t>(cTargetClasses));

   size_t cFeatures;
   const BoolEbmType * aFeaturesCategorical;
   if(ConvertLogicalsToBools(featuresCategorical, &cFeatures, &aFeaturesCategorical)) {
      // we've already logged any errors
      return R_NilValue;
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbmType countFeatures = static_cast<IntEbmType>(cFeatures);

   size_t cFeaturesFromBinCount;
   const IntEbmType * aFeaturesBinCount;
   if(ConvertDoublesToIndexes(featuresBinCount, &cFeaturesFromBinCount, &aFeaturesBinCount)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cFeatures != cFeaturesFromBinCount) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationInteractionDetector_R cFeatures != cFeaturesFromBinCount");
      return R_NilValue;
   }

   size_t cBinnedData;
   const IntEbmType * aBinnedData;
   if(ConvertDoublesToIndexes(binnedData, &cBinnedData, &aBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cSamples;
   const IntEbmType * aTargets;
   if(ConvertDoublesToIndexes(targets, &cSamples, &aTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countSamples = static_cast<IntEbmType>(cSamples);

   if(IsMultiplyError(cSamples, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationInteractionDetector_R IsMultiplyError(cSamples, cFeatures)");
      return R_NilValue;
   }
   if(cSamples * cFeatures != cBinnedData) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationInteractionDetector_R cSamples * cFeatures != cBinnedData");
      return R_NilValue;
   }

   size_t cPredictorScores;
   const FloatEbmType * aPredictorScores;
   if(ConvertDoublesToEbmFloats(predictorScores, &cPredictorScores, &aPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(IsMultiplyError(cSamples, cVectorLength)) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationInteractionDetector_R IsMultiplyError(cSamples, cVectorLength)");
      return R_NilValue;
   }
   if(cVectorLength * cSamples != cPredictorScores) {
      LOG_0(TraceLevelError, "ERROR CreateClassificationInteractionDetector_R cVectorLength * cSamples != cPredictorScores");
      return R_NilValue;
   }

   double * pWeights = nullptr;
   if(NILSXP != TYPEOF(weights)) {
      if(REALSXP != TYPEOF(weights)) {
         LOG_0(TraceLevelError, "ERROR CreateClassificationInteractionDetector_R REALSXP != TYPEOF(weights)");
         return R_NilValue;
      }
      const R_xlen_t weightsLength = xlength(weights);
      if(!IsNumberConvertable<size_t>(weightsLength)) {
         LOG_0(TraceLevelError, "ERROR CreateClassificationInteractionDetector_R !IsNumberConvertable<size_t>(weightsLength)");
         return R_NilValue;
      }
      const size_t cWeights = static_cast<size_t>(weightsLength);
      if(cWeights != cSamples) {
         LOG_0(TraceLevelError, "ERROR CreateClassificationInteractionDetector_R cWeights != cSamples");
         return R_NilValue;
      }
      pWeights = REAL(weights);
   }

   const InteractionDetectorHandle interactionDetectorHandle = CreateClassificationInteractionDetector(
      static_cast<IntEbmType>(cTargetClasses),
      countFeatures,
      aFeaturesCategorical,
      aFeaturesBinCount,
      countSamples,
      aBinnedData,
      aTargets,
      pWeights,
      aPredictorScores,
      nullptr
   );

   if(nullptr == interactionDetectorHandle) {
      return R_NilValue;
   }

   SEXP interactionDetectorHandleWrapped = R_MakeExternalPtr(static_cast<void *>(interactionDetectorHandle), R_NilValue, R_NilValue); // makes an EXTPTRSXP
   PROTECT(interactionDetectorHandleWrapped);

   R_RegisterCFinalizerEx(interactionDetectorHandleWrapped, &InteractionFinalizer, Rboolean::TRUE);

   UNPROTECT(1);
   return interactionDetectorHandleWrapped;
}

SEXP CreateRegressionInteractionDetector_R(
   SEXP featuresCategorical,
   SEXP featuresBinCount,
   SEXP binnedData,
   SEXP targets,
   SEXP weights,
   SEXP predictorScores
) {
   EBM_ASSERT(nullptr != featuresCategorical);
   EBM_ASSERT(nullptr != featuresBinCount);
   EBM_ASSERT(nullptr != binnedData);
   EBM_ASSERT(nullptr != targets);
   EBM_ASSERT(nullptr != weights);
   EBM_ASSERT(nullptr != predictorScores);

   size_t cFeatures;
   const BoolEbmType * aFeaturesCategorical;
   if(ConvertLogicalsToBools(featuresCategorical, &cFeatures, &aFeaturesCategorical)) {
      // we've already logged any errors
      return R_NilValue;
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbmType countFeatures = static_cast<IntEbmType>(cFeatures);

   size_t cFeaturesFromBinCount;
   const IntEbmType * aFeaturesBinCount;
   if(ConvertDoublesToIndexes(featuresBinCount, &cFeaturesFromBinCount, &aFeaturesBinCount)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cFeatures != cFeaturesFromBinCount) {
      LOG_0(TraceLevelError, "ERROR CreateRegressionInteractionDetector_R cFeatures != cFeaturesFromBinCount");
      return R_NilValue;
   }

   size_t cBinnedData;
   const IntEbmType * aBinnedData;
   if(ConvertDoublesToIndexes(binnedData, &cBinnedData, &aBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cSamples;
   const FloatEbmType * aTargets;
   if(ConvertDoublesToEbmFloats(targets, &cSamples, &aTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countSamples = static_cast<IntEbmType>(cSamples);

   if(IsMultiplyError(cSamples, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR CreateRegressionInteractionDetector_R IsMultiplyError(cSamples, cFeatures)");
      return R_NilValue;
   }
   if(cSamples * cFeatures != cBinnedData) {
      LOG_0(TraceLevelError, "ERROR CreateRegressionInteractionDetector_R cSamples * cFeatures != cBinnedData");
      return R_NilValue;
   }

   size_t cPredictorScores;
   const FloatEbmType * aPredictorScores;
   if(ConvertDoublesToEbmFloats(predictorScores, &cPredictorScores, &aPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cSamples != cPredictorScores) {
      LOG_0(TraceLevelError, "ERROR CreateRegressionInteractionDetector_R cSamples != cPredictorScores");
      return R_NilValue;
   }

   double * pWeights = nullptr;
   if(NILSXP != TYPEOF(weights)) {
      if(REALSXP != TYPEOF(weights)) {
         LOG_0(TraceLevelError, "ERROR CreateRegressionInteractionDetector_R REALSXP != TYPEOF(weights)");
         return R_NilValue;
      }
      const R_xlen_t weightsLength = xlength(weights);
      if(!IsNumberConvertable<size_t>(weightsLength)) {
         LOG_0(TraceLevelError, "ERROR CreateRegressionInteractionDetector_R !IsNumberConvertable<size_t>(weightsLength)");
         return R_NilValue;
      }
      const size_t cWeights = static_cast<size_t>(weightsLength);
      if(cWeights != cSamples) {
         LOG_0(TraceLevelError, "ERROR CreateRegressionInteractionDetector_R cWeights != cSamples");
         return R_NilValue;
      }
      pWeights = REAL(weights);
   }

   const InteractionDetectorHandle interactionDetectorHandle = CreateRegressionInteractionDetector(
      countFeatures, 
      aFeaturesCategorical,
      aFeaturesBinCount,
      countSamples,
      aBinnedData, 
      aTargets, 
      pWeights,
      aPredictorScores,
      nullptr
   );

   if(nullptr == interactionDetectorHandle) {
      return R_NilValue;
   }
   
   SEXP interactionDetectorHandleWrapped = R_MakeExternalPtr(static_cast<void *>(interactionDetectorHandle), R_NilValue, R_NilValue); // makes an EXTPTRSXP
   PROTECT(interactionDetectorHandleWrapped);

   R_RegisterCFinalizerEx(interactionDetectorHandleWrapped, &InteractionFinalizer, Rboolean::TRUE);

   UNPROTECT(1);
   return interactionDetectorHandleWrapped;
}

SEXP CalculateInteractionScore_R(
   SEXP interactionDetectorHandleWrapped,
   SEXP featureIndexes,
   SEXP countSamplesRequiredForChildSplitMin
) {
   EBM_ASSERT(nullptr != interactionDetectorHandleWrapped); // shouldn't be possible
   EBM_ASSERT(nullptr != featureIndexes); // shouldn't be possible
   EBM_ASSERT(nullptr != countSamplesRequiredForChildSplitMin);

   if(EXTPTRSXP != TYPEOF(interactionDetectorHandleWrapped)) {
      LOG_0(TraceLevelError, "ERROR CalculateInteractionScore_R EXTPTRSXP != TYPEOF(interactionDetectorHandleWrapped)");
      return R_NilValue;
   }
   InteractionCore * pInteractionCore = static_cast<InteractionCore *>(R_ExternalPtrAddr(interactionDetectorHandleWrapped));
   if(nullptr == pInteractionCore) {
      LOG_0(TraceLevelError, "ERROR CalculateInteractionScore_R nullptr == pInteractionCore");
      return R_NilValue;
   }

   size_t cDimensions;
   const IntEbmType * aFeatureIndexes;
   if(ConvertDoublesToIndexes(featureIndexes, &cDimensions, &aFeatureIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countDimensions = static_cast<IntEbmType>(cDimensions);

   if(!IsSingleDoubleVector(countSamplesRequiredForChildSplitMin)) {
      LOG_0(TraceLevelError, "ERROR CalculateInteractionScore_R !IsSingleDoubleVector(countSamplesRequiredForChildSplitMin)");
      return R_NilValue;
   }
   double doubleCountSamplesRequiredForChildSplitMin = REAL(countSamplesRequiredForChildSplitMin)[0];
   IntEbmType countEbmSamplesRequiredForChildSplitMin;
   static_assert(std::numeric_limits<double>::is_iec559, "we need is_iec559 to know that comparisons to infinity and -infinity to normal numbers work");
   if(std::isnan(doubleCountSamplesRequiredForChildSplitMin) ||
      static_cast<double>(std::numeric_limits<IntEbmType>::max()) < doubleCountSamplesRequiredForChildSplitMin
   ) {
      LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScore_R countSamplesRequiredForChildSplitMin overflow");
      countEbmSamplesRequiredForChildSplitMin = std::numeric_limits<IntEbmType>::max();
   } else if(doubleCountSamplesRequiredForChildSplitMin < static_cast<double>(std::numeric_limits<IntEbmType>::lowest())) {
      LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScore_R countSamplesRequiredForChildSplitMin underflow");
      countEbmSamplesRequiredForChildSplitMin = std::numeric_limits<IntEbmType>::lowest();
   } else {
      countEbmSamplesRequiredForChildSplitMin = static_cast<IntEbmType>(doubleCountSamplesRequiredForChildSplitMin);
   }

   FloatEbmType interactionScoreOut;
   if(0 != CalculateInteractionScore(reinterpret_cast<InteractionDetectorHandle>(pInteractionCore), countDimensions, aFeatureIndexes, countEbmSamplesRequiredForChildSplitMin, &interactionScoreOut)) {
      LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScore_R CalculateInteractionScore returned error code");
      return R_NilValue;
   }

   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = static_cast<double>(interactionScoreOut);
   UNPROTECT(1);
   return ret;
}

SEXP FreeInteractionDetector_R(
   SEXP interactionDetectorHandleWrapped
) {
   InteractionFinalizer(interactionDetectorHandleWrapped);
   return R_NilValue;
}

static const R_CallMethodDef g_exposedFunctions[] = {
   { "GenerateRandomNumber_R", (DL_FUNC)&GenerateRandomNumber_R, 2 },
   { "CutQuantile_R", (DL_FUNC)&CutQuantile_R, 4 },
   { "Discretize_R", (DL_FUNC)&Discretize_R, 3 },
   { "SampleWithoutReplacement_R", (DL_FUNC)&SampleWithoutReplacement_R, 4 },
   { "CreateClassificationBooster_R", (DL_FUNC)&CreateClassificationBooster_R, 15 },
   { "CreateRegressionBooster_R", (DL_FUNC)&CreateRegressionBooster_R, 14 },
   { "GenerateModelUpdate_R", (DL_FUNC)&GenerateModelUpdate_R, 5 },
   { "ApplyModelUpdate_R", (DL_FUNC)&ApplyModelUpdate_R, 1 },
   { "GetBestModelFeatureGroup_R", (DL_FUNC)&GetBestModelFeatureGroup_R, 2 },
   { "GetCurrentModelFeatureGroup_R", (DL_FUNC)& GetCurrentModelFeatureGroup_R, 2 },
   { "FreeBooster_R", (DL_FUNC)& FreeBooster_R, 1 },
   { "CreateThreadStateBoosting_R", (DL_FUNC)&CreateThreadStateBoosting_R, 1 },
   { "FreeThreadStateBoosting_R", (DL_FUNC)&FreeThreadStateBoosting_R, 1 },
   { "CreateClassificationInteractionDetector_R", (DL_FUNC)&CreateClassificationInteractionDetector_R, 7 },
   { "CreateRegressionInteractionDetector_R", (DL_FUNC)&CreateRegressionInteractionDetector_R, 6 },
   { "CalculateInteractionScore_R", (DL_FUNC)&CalculateInteractionScore_R, 3 },
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
