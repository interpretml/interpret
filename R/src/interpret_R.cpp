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

// when R compiles this library, on some systems it can generate a "NOTE installed size is.." meaning the C++ 
// compiled into a library produces too big a library.  
// We would want to disable the -g flag (with -g0), but according to this, it's not possible currently:
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

INLINE_ALWAYS BoolEbm ConvertBool(const SEXP sexp) {
   if(LGLSXP != TYPEOF(sexp)) {
      error("ConvertBool LGLSXP != TYPEOF(sexp)");
   }
   if(R_xlen_t { 1 } != xlength(sexp)) {
      error("ConvertBool R_xlen_t { 1 } != xlength(sexp)");
   }
   const Rboolean val = static_cast<Rboolean>(LOGICAL(sexp)[0]);
   if(Rboolean::FALSE == val) {
      return EBM_FALSE;
   }
   if(Rboolean::TRUE == val) {
      return EBM_TRUE;
   }
   error("ConvertBool val not a bool");
}

INLINE_ALWAYS bool IsDoubleToIntEbmIndexValid(const double val) {
   if(std::isnan(val)) {
      return false;
   }
   if(val < double { 0 }) {
      return false;
   }
   const double maxValid = std::min(std::min(double { R_XLEN_T_MAX }, double { SAFE_FLOAT64_AS_INT64_MAX }),
      std::min(static_cast<double>(std::numeric_limits<size_t>::max()), 
      static_cast<double>(std::numeric_limits<IntEbm>::max())));
   if(maxValid < val) {
      return false;
   }
   return true;
}

void DataSetFinalizer(SEXP dataSetHandleWrapped) {
   EBM_ASSERT(nullptr != dataSetHandleWrapped); // shouldn't be possible
   if(EXTPTRSXP == TYPEOF(dataSetHandleWrapped)) {
      void * const dataSetHandle = R_ExternalPtrAddr(dataSetHandleWrapped);
      if(nullptr != dataSetHandle) {
         R_ClearExternalPtr(dataSetHandleWrapped);
         free(dataSetHandle);
      }
   }
}

void BoostingFinalizer(SEXP boosterHandleWrapped) {
   EBM_ASSERT(nullptr != boosterHandleWrapped); // shouldn't be possible
   if(EXTPTRSXP == TYPEOF(boosterHandleWrapped)) {
      const BoosterHandle boosterHandle = static_cast<BoosterHandle>(R_ExternalPtrAddr(boosterHandleWrapped));
      if(nullptr != boosterHandle) {
         R_ClearExternalPtr(boosterHandleWrapped);
         FreeBooster(boosterHandle);
      }
   }
}

void InteractionFinalizer(SEXP interactionHandleWrapped) {
   EBM_ASSERT(nullptr != interactionHandleWrapped); // shouldn't be possible
   if(EXTPTRSXP == TYPEOF(interactionHandleWrapped)) {
      const InteractionHandle interactionHandle = static_cast<InteractionHandle>(R_ExternalPtrAddr(interactionHandleWrapped));
      if(nullptr != interactionHandle) {
         R_ClearExternalPtr(interactionHandleWrapped);
         FreeInteractionDetector(interactionHandle);
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

SEXP GenerateSeed_R(SEXP seed, SEXP randomMix) {
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

SEXP CutQuantile_R(SEXP featureVals, SEXP minSamplesBin, SEXP isRounded, SEXP countCuts) {
   EBM_ASSERT(nullptr != featureVals);
   EBM_ASSERT(nullptr != minSamplesBin);
   EBM_ASSERT(nullptr != isRounded);
   EBM_ASSERT(nullptr != countCuts);

   ErrorEbm err;

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

   BoolEbm bRounded = ConvertBool(isRounded);

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

   err = CutQuantile(
      countSamples,
      aFeatureVals,
      minSamplesBinIntEbm,
      bRounded,
      &countCutsIntEbm,
      aCutsLowerBoundInclusive
   );

   if(Error_None != err) {
      return R_NilValue;
   }

   if(IsConvertErrorDual<R_xlen_t, size_t>(countCutsIntEbm)) {
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

SEXP BinFeature_R(SEXP featureVals, SEXP cutsLowerBoundInclusive, SEXP binIndexesOut) {
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
      IntEbm * const aiBins = reinterpret_cast<IntEbm *>(R_alloc(cSamples, static_cast<int>(sizeof(IntEbm))));
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

SEXP MeasureDataSetHeader_R(
   SEXP countFeatures,
   SEXP countWeights,
   SEXP countTargets
) {
   EBM_ASSERT(nullptr != countFeatures);
   EBM_ASSERT(nullptr != countWeights);
   EBM_ASSERT(nullptr != countTargets);

   if(!IsSingleDoubleVector(countFeatures)) {
      error("MeasureDataSetHeader_R !IsSingleDoubleVector(countFeatures)");
   }
   const double countFeaturesDouble = REAL(countFeatures)[0];
   if(!IsDoubleToIntEbmIndexValid(countFeaturesDouble)) {
      error("MeasureDataSetHeader_R !IsDoubleToIntEbmIndexValid(countFeaturesDouble)");
   }
   const IntEbm countFeaturesIntEbm = static_cast<IntEbm>(countFeaturesDouble);

   if(!IsSingleDoubleVector(countWeights)) {
      error("MeasureDataSetHeader_R !IsSingleDoubleVector(countWeights)");
   }
   const double countWeightsDouble = REAL(countWeights)[0];
   if(!IsDoubleToIntEbmIndexValid(countWeightsDouble)) {
      error("MeasureDataSetHeader_R !IsDoubleToIntEbmIndexValid(countWeightsDouble)");
   }
   const IntEbm countWeightsIntEbm = static_cast<IntEbm>(countWeightsDouble);

   if(!IsSingleDoubleVector(countTargets)) {
      error("!IsSingleDoubleVector(countTargets)");
   }
   const double countTargetsDouble = REAL(countTargets)[0];
   if(!IsDoubleToIntEbmIndexValid(countTargetsDouble)) {
      error("MeasureDataSetHeader_R !IsDoubleToIntEbmIndexValid(countTargetsDouble)");
   }
   const IntEbm countTargetsIntEbm = static_cast<IntEbm>(countTargetsDouble);

   const IntEbm countBytes = MeasureDataSetHeader(
      countFeaturesIntEbm,
      countWeightsIntEbm,
      countTargetsIntEbm
   );
   if(countBytes < 0) {
      error("MeasureDataSetHeader_R MeasureDataSetHeader returned error code: %" ErrorEbmPrintf, static_cast<ErrorEbm>(countBytes));
   }
   if(SAFE_FLOAT64_AS_INT64_MAX < countBytes) {
      error("MeasureDataSetHeader_R SAFE_FLOAT64_AS_INT64_MAX < countBytes");
   }

   const SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = static_cast<double>(countBytes);
   UNPROTECT(1);
   return ret;
}

SEXP MeasureFeature_R(
   SEXP countBins,
   SEXP isMissing,
   SEXP isUnknown,
   SEXP isNominal,
   SEXP binIndexes
) {
   EBM_ASSERT(nullptr != countBins);
   EBM_ASSERT(nullptr != isMissing);
   EBM_ASSERT(nullptr != isUnknown);
   EBM_ASSERT(nullptr != isNominal);
   EBM_ASSERT(nullptr != binIndexes);

   if(!IsSingleDoubleVector(countBins)) {
      error("MeasureFeature_R !IsSingleDoubleVector(countBins)");
   }
   const double countBinsDouble = REAL(countBins)[0];
   if(!IsDoubleToIntEbmIndexValid(countBinsDouble)) {
      error("MeasureFeature_R !IsDoubleToIntEbmIndexValid(countBinsDouble)");
   }
   const IntEbm countBinsIntEbm = static_cast<IntEbm>(countBinsDouble);

   BoolEbm bMissing = ConvertBool(isMissing);
   BoolEbm bUnknown = ConvertBool(isUnknown);
   BoolEbm bNominal = ConvertBool(isNominal);

   size_t cSamples;
   const IntEbm * aiBins;
   if(ConvertDoublesToIndexes(binIndexes, &cSamples, &aiBins)) {
      error("MeasureFeature_R ConvertDoublesToIndexes(binIndexes, &cSamples, &aiBins)");
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbm countSamples = static_cast<IntEbm>(cSamples);

   const IntEbm countBytes = MeasureFeature(
      countBinsIntEbm,
      bMissing,
      bUnknown,
      bNominal,
      countSamples,
      aiBins
   );
   if(countBytes < 0) {
      error("MeasureFeature_R MeasureFeature returned error code: %" ErrorEbmPrintf, static_cast<ErrorEbm>(countBytes));
   }
   if(SAFE_FLOAT64_AS_INT64_MAX < countBytes) {
      error("MeasureFeature_R SAFE_FLOAT64_AS_INT64_MAX < countBytes");
   }

   const SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = static_cast<double>(countBytes);
   UNPROTECT(1);
   return ret;
}

SEXP MeasureClassificationTarget_R(
   SEXP countClasses,
   SEXP targets
) {
   EBM_ASSERT(nullptr != countClasses);
   EBM_ASSERT(nullptr != targets);

   if(!IsSingleDoubleVector(countClasses)) {
      error("MeasureClassificationTarget_R !IsSingleDoubleVector(countClasses)");
   }
   const double countClassesDouble = REAL(countClasses)[0];
   if(!IsDoubleToIntEbmIndexValid(countClassesDouble)) {
      error("MeasureClassificationTarget_R !IsDoubleToIntEbmIndexValid(countClassesDouble)");
   }
   const IntEbm countClassesIntEbm = static_cast<IntEbm>(countClassesDouble);

   size_t cSamples;
   const IntEbm * aTargets;
   if(ConvertDoublesToIndexes(targets, &cSamples, &aTargets)) {
      error("MeasureClassificationTarget_R ConvertDoublesToIndexes(targets, &cSamples, &aTargets)");
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbm countSamples = static_cast<IntEbm>(cSamples);

   const IntEbm countBytes = MeasureClassificationTarget(
      countClassesIntEbm,
      countSamples,
      aTargets
   );
   if(countBytes < 0) {
      error("MeasureClassificationTarget_R MeasureClassificationTarget returned error code: %" ErrorEbmPrintf, static_cast<ErrorEbm>(countBytes));
   }
   if(SAFE_FLOAT64_AS_INT64_MAX < countBytes) {
      error("MeasureClassificationTarget_R SAFE_FLOAT64_AS_INT64_MAX < countBytes");
   }

   const SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = static_cast<double>(countBytes);
   UNPROTECT(1);
   return ret;
}

SEXP CreateDataSet_R(SEXP countBytes) {
   EBM_ASSERT(nullptr != countBytes);

   if(!IsSingleDoubleVector(countBytes)) {
      error("CreateDataSet_R !IsSingleDoubleVector(countBytes)");
   }
   const double countBytesDouble = REAL(countBytes)[0];
   if(!IsDoubleToIntEbmIndexValid(countBytesDouble)) {
      error("CreateDataSet_R !IsDoubleToIntEbmIndexValid(countBytesDouble)");
   }
   const size_t cBytes = static_cast<size_t>(countBytesDouble);

   void * dataSetHandle = malloc(cBytes);

   SEXP dataSetHandleWrapped = R_MakeExternalPtr(dataSetHandle, R_NilValue, R_NilValue); // makes an EXTPTRSXP
   PROTECT(dataSetHandleWrapped);

   R_RegisterCFinalizerEx(dataSetHandleWrapped, &DataSetFinalizer, Rboolean::TRUE);

   UNPROTECT(1);
   return dataSetHandleWrapped;
}

SEXP FreeDataSet_R(SEXP dataSetHandleWrapped) {
   EBM_ASSERT(nullptr != dataSetHandleWrapped);

   DataSetFinalizer(dataSetHandleWrapped);
   return R_NilValue;
}

SEXP FillDataSetHeader_R(
   SEXP countFeatures,
   SEXP countWeights,
   SEXP countTargets,
   SEXP countBytesAllocated,
   SEXP fillMemWrapped
) {
   EBM_ASSERT(nullptr != countFeatures);
   EBM_ASSERT(nullptr != countWeights);
   EBM_ASSERT(nullptr != countTargets);
   EBM_ASSERT(nullptr != countBytesAllocated);
   EBM_ASSERT(nullptr != fillMemWrapped);

   if(!IsSingleDoubleVector(countFeatures)) {
      error("FillDataSetHeader_R !IsSingleDoubleVector(countFeatures)");
   }
   const double countFeaturesDouble = REAL(countFeatures)[0];
   if(!IsDoubleToIntEbmIndexValid(countFeaturesDouble)) {
      error("FillDataSetHeader_R !IsDoubleToIntEbmIndexValid(countFeaturesDouble)");
   }
   const IntEbm countFeaturesIntEbm = static_cast<IntEbm>(countFeaturesDouble);

   if(!IsSingleDoubleVector(countWeights)) {
      error("FillDataSetHeader_R !IsSingleDoubleVector(countWeights)");
   }
   const double countWeightsDouble = REAL(countWeights)[0];
   if(!IsDoubleToIntEbmIndexValid(countWeightsDouble)) {
      error("FillDataSetHeader_R !IsDoubleToIntEbmIndexValid(countWeightsDouble)");
   }
   const IntEbm countWeightsIntEbm = static_cast<IntEbm>(countWeightsDouble);

   if(!IsSingleDoubleVector(countTargets)) {
      error("FillDataSetHeader_R !IsSingleDoubleVector(countTargets)");
   }
   const double countTargetsDouble = REAL(countTargets)[0];
   if(!IsDoubleToIntEbmIndexValid(countTargetsDouble)) {
      error("FillDataSetHeader_R !IsDoubleToIntEbmIndexValid(countTargetsDouble)");
   }
   const IntEbm countTargetsIntEbm = static_cast<IntEbm>(countTargetsDouble);

   if(!IsSingleDoubleVector(countBytesAllocated)) {
      error("FillDataSetHeader_R !IsSingleDoubleVector(countBytesAllocated)");
   }
   const double countBytesAllocatedDouble = REAL(countBytesAllocated)[0];
   if(!IsDoubleToIntEbmIndexValid(countBytesAllocatedDouble)) {
      error("FillDataSetHeader_R !IsDoubleToIntEbmIndexValid(countBytesAllocatedDouble)");
   }
   const IntEbm countBytesAllocatedIntEbm = static_cast<IntEbm>(countBytesAllocatedDouble);

   if(EXTPTRSXP != TYPEOF(fillMemWrapped)) {
      error("FillDataSetHeader_R EXTPTRSXP != TYPEOF(fillMemWrapped)");
   }
   void * const pDataset = R_ExternalPtrAddr(fillMemWrapped);

   const ErrorEbm err = FillDataSetHeader(
      countFeaturesIntEbm,
      countWeightsIntEbm,
      countTargetsIntEbm,
      countBytesAllocatedIntEbm,
      pDataset
   );
   if(Error_None != err) {
      error("FillDataSetHeader_R FillDataSetHeader returned error code: %" ErrorEbmPrintf, err);
   }

   return R_NilValue;
}

SEXP FillFeature_R(
   SEXP countBins,
   SEXP isMissing,
   SEXP isUnknown,
   SEXP isNominal,
   SEXP binIndexes,
   SEXP countBytesAllocated,
   SEXP fillMemWrapped
) {
   EBM_ASSERT(nullptr != countBins);
   EBM_ASSERT(nullptr != isMissing);
   EBM_ASSERT(nullptr != isUnknown);
   EBM_ASSERT(nullptr != isNominal);
   EBM_ASSERT(nullptr != binIndexes);
   EBM_ASSERT(nullptr != countBytesAllocated);
   EBM_ASSERT(nullptr != fillMemWrapped);

   if(!IsSingleDoubleVector(countBins)) {
      error("FillFeature_R !IsSingleDoubleVector(countBins)");
   }
   const double countBinsDouble = REAL(countBins)[0];
   if(!IsDoubleToIntEbmIndexValid(countBinsDouble)) {
      error("FillFeature_R !IsDoubleToIntEbmIndexValid(countBinsDouble)");
   }
   const IntEbm countBinsIntEbm = static_cast<IntEbm>(countBinsDouble);

   BoolEbm bMissing = ConvertBool(isMissing);
   BoolEbm bUnknown = ConvertBool(isUnknown);
   BoolEbm bNominal = ConvertBool(isNominal);

   size_t cSamples;
   const IntEbm * aiBins;
   if(ConvertDoublesToIndexes(binIndexes, &cSamples, &aiBins)) {
      error("FillFeature_R ConvertDoublesToIndexes(binIndexes, &cSamples, &aiBins)");
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbm countSamples = static_cast<IntEbm>(cSamples);

   if(!IsSingleDoubleVector(countBytesAllocated)) {
      error("FillFeature_R !IsSingleDoubleVector(countBytesAllocated)");
   }
   const double countBytesAllocatedDouble = REAL(countBytesAllocated)[0];
   if(!IsDoubleToIntEbmIndexValid(countBytesAllocatedDouble)) {
      error("FillFeature_R !IsDoubleToIntEbmIndexValid(countBytesAllocatedDouble)");
   }
   const IntEbm countBytesAllocatedIntEbm = static_cast<IntEbm>(countBytesAllocatedDouble);

   if(EXTPTRSXP != TYPEOF(fillMemWrapped)) {
      error("FillFeature_R EXTPTRSXP != TYPEOF(fillMemWrapped)");
   }
   void * const pDataset = R_ExternalPtrAddr(fillMemWrapped);

   const ErrorEbm err = FillFeature(
      countBinsIntEbm,
      bMissing,
      bUnknown,
      bNominal,
      countSamples,
      aiBins,
      countBytesAllocatedIntEbm,
      pDataset
   );
   if(Error_None != err) {
      error("FillFeature_R FillFeature returned error code: %" ErrorEbmPrintf, err);
   }

   return R_NilValue;
}

SEXP FillClassificationTarget_R(SEXP countClasses, SEXP targets, SEXP countBytesAllocated, SEXP fillMemWrapped) {
   EBM_ASSERT(nullptr != countClasses);
   EBM_ASSERT(nullptr != targets);
   EBM_ASSERT(nullptr != countBytesAllocated);
   EBM_ASSERT(nullptr != fillMemWrapped);

   if(!IsSingleDoubleVector(countClasses)) {
      error("FillClassificationTarget_R !IsSingleDoubleVector(countClasses)");
   }
   const double countClassesDouble = REAL(countClasses)[0];
   if(!IsDoubleToIntEbmIndexValid(countClassesDouble)) {
      error("FillClassificationTarget_R !IsDoubleToIntEbmIndexValid(countClassesDouble)");
   }
   const IntEbm countClassesIntEbm = static_cast<IntEbm>(countClassesDouble);

   size_t cSamples;
   const IntEbm * aTargets;
   if(ConvertDoublesToIndexes(targets, &cSamples, &aTargets)) {
      error("FillClassificationTarget_R ConvertDoublesToIndexes(targets, &cSamples, &aTargets)");
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbm countSamples = static_cast<IntEbm>(cSamples);

   if(!IsSingleDoubleVector(countBytesAllocated)) {
      error("FillClassificationTarget_R !IsSingleDoubleVector(countBytesAllocated)");
   }
   const double countBytesAllocatedDouble = REAL(countBytesAllocated)[0];
   if(!IsDoubleToIntEbmIndexValid(countBytesAllocatedDouble)) {
      error("FillClassificationTarget_R !IsDoubleToIntEbmIndexValid(countBytesAllocatedDouble)");
   }
   const IntEbm countBytesAllocatedIntEbm = static_cast<IntEbm>(countBytesAllocatedDouble);

   if(EXTPTRSXP != TYPEOF(fillMemWrapped)) {
      error("FillClassificationTarget_R EXTPTRSXP != TYPEOF(fillMemWrapped)");
   }
   void * const pDataset = R_ExternalPtrAddr(fillMemWrapped);

   const ErrorEbm err = FillClassificationTarget(
      countClassesIntEbm,
      countSamples,
      aTargets,
      countBytesAllocatedIntEbm,
      pDataset
   );
   if(Error_None != err) {
      error("FillClassificationTarget_R FillClassificationTarget returned error code: %" ErrorEbmPrintf, err);
   }

   return R_NilValue;
}

SEXP SampleWithoutReplacement_R(SEXP seed, SEXP countTrainingSamples, SEXP countValidationSamples, SEXP bagOut) {
   EBM_ASSERT(nullptr != seed);
   EBM_ASSERT(nullptr != countTrainingSamples);
   EBM_ASSERT(nullptr != countValidationSamples);
   EBM_ASSERT(nullptr != bagOut);

   ErrorEbm err;

   BoolEbm bDeterministic;
   SeedEbm seedLocal;
   if(NILSXP == TYPEOF(seed)) {
      seedLocal = 0;
      bDeterministic = EBM_FALSE;
   } else {
      if(!IsSingleIntVector(seed)) {
         LOG_0(Trace_Error, "ERROR SampleWithoutReplacement_R !IsSingleIntVector(seed)");
         return R_NilValue;
      }
      seedLocal = INTEGER(seed)[0];
      bDeterministic = EBM_TRUE;
   }

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
      BagEbm * const aBag = reinterpret_cast<BagEbm *>(R_alloc(cSamples, static_cast<int>(sizeof(BagEbm))));
      EBM_ASSERT(nullptr != aBag); // this can't be nullptr since R_alloc uses R error handling

      err = SampleWithoutReplacement(
         bDeterministic,
         seedLocal,
         countTrainingSamplesIntEbm,
         countValidationSamplesIntEbm,
         aBag
      );
      if(Error_None != err) {
         return R_NilValue;
      }

      int32_t * pSampleReplicationOut = INTEGER(bagOut);
      const BagEbm * pSampleReplication = aBag;
      const BagEbm * const pSampleReplicationEnd = aBag + cSamples;
      do {
         const BagEbm replication = *pSampleReplication;
         if(IsConvertError<int32_t>(replication)) {
            error("SampleWithoutReplacement_R IsConvertError<int32_t>(replication)");
         }
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

SEXP CreateBooster_R(
   SEXP seed,
   SEXP dataSetWrapped,
   SEXP bag,
   SEXP initScores,
   SEXP dimensionCounts,
   SEXP featureIndexes,
   SEXP countInnerBags
) {
   EBM_ASSERT(nullptr != seed);
   EBM_ASSERT(nullptr != dataSetWrapped);
   EBM_ASSERT(nullptr != bag);
   EBM_ASSERT(nullptr != initScores);
   EBM_ASSERT(nullptr != dimensionCounts);
   EBM_ASSERT(nullptr != featureIndexes);
   EBM_ASSERT(nullptr != countInnerBags);

   ErrorEbm err;

   BoolEbm bDeterministic;
   SeedEbm seedLocal;
   if(NILSXP == TYPEOF(seed)) {
      seedLocal = 0;
      bDeterministic = EBM_FALSE;
   } else {
      if(!IsSingleIntVector(seed)) {
         error("CreateBooster_R !IsSingleIntVector(seed)");
      }
      seedLocal = INTEGER(seed)[0];
      bDeterministic = EBM_TRUE;
   }

   if(EXTPTRSXP != TYPEOF(dataSetWrapped)) {
      error("CreateBooster_R EXTPTRSXP != TYPEOF(dataSetWrapped)");
   }
   const void * pDataSet = R_ExternalPtrAddr(dataSetWrapped);

   IntEbm countSamples;
   IntEbm unused1;
   IntEbm unused2;
   IntEbm unused3;

   err = ExtractDataSetHeader(pDataSet, &countSamples, &unused1, &unused2, &unused3);
   if(Error_None != err) {
      error("CreateBooster_R ExtractDataSetHeader failed with error code: %" ErrorEbmPrintf, err);
   }
   const size_t cSamples = static_cast<size_t>(countSamples); // we trust our internal code that this is convertible

   BagEbm * aBag = nullptr;
   size_t cExpectedInitScores = cSamples;
   if(NILSXP != TYPEOF(bag)) {
      if(INTSXP != TYPEOF(bag)) {
         error("CreateBooster_R INTSXP != TYPEOF(bag)");
      }
      const R_xlen_t countSamplesVerify = xlength(bag);
      if(IsConvertError<size_t>(countSamplesVerify)) {
         error("CreateBooster_R IsConvertError<size_t>(countSamplesVerify)");
      }
      const size_t cSamplesVerify = static_cast<size_t>(countSamplesVerify);
      if(cSamples != cSamplesVerify) {
         error("CreateBooster_R cSamples != cSamplesVerify");
      }

      aBag = reinterpret_cast<BagEbm *>(R_alloc(cSamples, static_cast<int>(sizeof(BagEbm))));
      EBM_ASSERT(nullptr != aBag); // this can't be nullptr since R_alloc uses R error handling

      cExpectedInitScores = 0;

      const int32_t * pSampleReplicationR = INTEGER(bag);
      BagEbm * pSampleReplication = aBag;
      const BagEbm * const pSampleReplicationEnd = aBag + cSamples;
      do {
         const int32_t replication = *pSampleReplicationR;
         if(IsConvertError<BagEbm>(replication)) {
            error("CreateBooster_R IsConvertError<BagEbm>(replication)");
         }
         if(0 != replication) {
            ++cExpectedInitScores;
         }
         *pSampleReplication = static_cast<BagEbm>(replication);
         ++pSampleReplicationR;
         ++pSampleReplication;
      } while(pSampleReplicationEnd != pSampleReplication);
   }

   const double * aInitScores = nullptr;
   if(NILSXP != TYPEOF(initScores)) {
      const IntEbm countInitScores = CountDoubles(initScores);
      if(countInitScores < 0) {
         // we've already logged any errors
         error("CreateBooster_R countInitScores < 0");
      }
      size_t cInitScores = static_cast<size_t>(countInitScores);
      if(cInitScores != cExpectedInitScores) {
         error("CreateBooster_R cInitScores != cExpectedInitScores");
      }
      aInitScores = REAL(initScores);
   }

   size_t cTerms;
   const IntEbm * acTermDimensions;
   if(ConvertDoublesToIndexes(dimensionCounts, &cTerms, &acTermDimensions)) {
      // we've already logged any errors
      error("CreateBooster_R ConvertDoublesToIndexes(dimensionCounts, &cTerms, &acTermDimensions)");
   }
   // the validity of this conversion was checked in ConvertDoublesToIndexes(...)
   const IntEbm countTerms = static_cast<IntEbm>(cTerms);

   const size_t cTotalDimensionsCheck = CountTotalDimensions(cTerms, acTermDimensions);
   if(SIZE_MAX == cTotalDimensionsCheck) {
      // we've already logged any errors
      error("CreateBooster_R SIZE_MAX == cTotalDimensionsCheck");
   }

   size_t cTotalDimensionsActual;
   const IntEbm * aiTermFeatures;
   if(ConvertDoublesToIndexes(featureIndexes, &cTotalDimensionsActual, &aiTermFeatures)) {
      // we've already logged any errors
      error("CreateBooster_R ConvertDoublesToIndexes(featureIndexes, &cTotalDimensionsActual, &aiTermFeatures)");
   }
   if(cTotalDimensionsActual != cTotalDimensionsCheck) {
      error("CreateBooster_R cTotalDimensionsActual != cTotalDimensionsCheck");
   }

   if(!IsSingleDoubleVector(countInnerBags)) {
      error("CreateBooster_R !IsSingleDoubleVector(countInnerBags)");
   }
   const double countInnerBagsDouble = REAL(countInnerBags)[0];
   if(!IsDoubleToIntEbmIndexValid(countInnerBagsDouble)) {
      error("CreateBooster_R !IsDoubleToIntEbmIndexValid(countInnerBagsDouble)");
   }
   const IntEbm countInnerBagsIntEbm = static_cast<IntEbm>(countInnerBagsDouble);

   BoosterHandle boosterHandle;
   err = CreateBooster(
      bDeterministic,
      seedLocal,
      pDataSet,
      aBag,
      aInitScores,
      countTerms,
      acTermDimensions,
      aiTermFeatures,
      countInnerBagsIntEbm,
      nullptr,
      &boosterHandle
   );
   if(Error_None != err || nullptr == boosterHandle) {
      error("CreateBooster_R error in call to CreateBooster: %" ErrorEbmPrintf, err);
   }

   SEXP boosterHandleWrapped = R_MakeExternalPtr(static_cast<void *>(boosterHandle), R_NilValue, R_NilValue); // makes an EXTPTRSXP
   PROTECT(boosterHandleWrapped);

   R_RegisterCFinalizerEx(boosterHandleWrapped, &BoostingFinalizer, Rboolean::TRUE);

   UNPROTECT(1);
   return boosterHandleWrapped;
}

SEXP FreeBooster_R(SEXP boosterHandleWrapped) {
   BoostingFinalizer(boosterHandleWrapped);
   return R_NilValue;
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

   ErrorEbm err;

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

   err = GenerateTermUpdate(
      boosterHandle,
      static_cast<IntEbm>(iTerm),
      BoostFlags_Default,
      learningRateLocal,
      minSamplesLeafEbm,
      aLeavesMax,
      &avgGain
   );
   if(Error_None != err) {
      LOG_0(Trace_Warning, "WARNING GenerateTermUpdate_R BoostingStep returned error code");
      return R_NilValue;
   }

   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = static_cast<double>(avgGain);
   UNPROTECT(1);
   return ret;
}

SEXP ApplyTermUpdate_R(SEXP boosterHandleWrapped) {
   EBM_ASSERT(nullptr != boosterHandleWrapped);

   ErrorEbm err;

   if(EXTPTRSXP != TYPEOF(boosterHandleWrapped)) {
      LOG_0(Trace_Error, "ERROR ApplyTermUpdate_R EXTPTRSXP != TYPEOF(boosterHandleWrapped)");
      return R_NilValue;
   }
   const BoosterHandle boosterHandle = static_cast<BoosterHandle>(R_ExternalPtrAddr(boosterHandleWrapped));
   // we don't use boosterHandle in this function, so let ApplyTermUpdate check if it's null or invalid

   double validationMetric;
   err = ApplyTermUpdate(boosterHandle, &validationMetric);
   if(Error_None != err) {
      LOG_0(Trace_Warning, "WARNING ApplyTermUpdate_R ApplyTermUpdate returned error code");
      return R_NilValue;
   }

   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = static_cast<double>(validationMetric);
   UNPROTECT(1);
   return ret;
}

SEXP GetBestTermScores_R(SEXP boosterHandleWrapped, SEXP indexTerm) {
   EBM_ASSERT(nullptr != boosterHandleWrapped); // shouldn't be possible
   EBM_ASSERT(nullptr != indexTerm); // shouldn't be possible

   ErrorEbm err;

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

   err = GetBestTermScores(boosterHandle, static_cast<IntEbm>(iTerm), REAL(ret));

   UNPROTECT(1);

   if(Error_None != err) {
      LOG_0(Trace_Warning, "WARNING GetBestTermScores_R IntEbm { 0 } != err");
      return R_NilValue;
   }
   return ret;
}

SEXP GetCurrentTermScores_R(SEXP boosterHandleWrapped, SEXP indexTerm) {
   EBM_ASSERT(nullptr != boosterHandleWrapped); // shouldn't be possible
   EBM_ASSERT(nullptr != indexTerm); // shouldn't be possible

   ErrorEbm err;

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

   err = GetCurrentTermScores(boosterHandle, static_cast<IntEbm>(iTerm), REAL(ret));

   UNPROTECT(1);

   if(Error_None != err) {
      LOG_0(Trace_Warning, "WARNING GetCurrentTermScores_R IntEbm { 0 } != err");
      return R_NilValue;
   }
   return ret;
}

SEXP CreateInteractionDetector_R(SEXP dataSetWrapped, SEXP bag, SEXP initScores) {
   EBM_ASSERT(nullptr != dataSetWrapped);
   EBM_ASSERT(nullptr != bag);
   EBM_ASSERT(nullptr != initScores);

   ErrorEbm err;

   if(EXTPTRSXP != TYPEOF(dataSetWrapped)) {
      error("CreateInteractionDetector_R EXTPTRSXP != TYPEOF(dataSetWrapped)");
   }
   const void * pDataSet = R_ExternalPtrAddr(dataSetWrapped);

   IntEbm countSamples;
   IntEbm unused1;
   IntEbm unused2;
   IntEbm unused3;

   err = ExtractDataSetHeader(pDataSet, &countSamples, &unused1, &unused2, &unused3);
   if(Error_None != err) {
      error("CreateInteractionDetector_R ExtractDataSetHeader failed with error code: %" ErrorEbmPrintf, err);
   }
   const size_t cSamples = static_cast<size_t>(countSamples); // we trust our internal code that this is convertible

   BagEbm * aBag = nullptr;
   size_t cExpectedInitScores = cSamples;
   if(NILSXP != TYPEOF(bag)) {
      if(INTSXP != TYPEOF(bag)) {
         error("CreateInteractionDetector_R INTSXP != TYPEOF(bag)");
      }
      const R_xlen_t countSamplesVerify = xlength(bag);
      if(IsConvertError<size_t>(countSamplesVerify)) {
         error("CreateInteractionDetector_R IsConvertError<size_t>(countSamplesVerify)");
      }
      const size_t cSamplesVerify = static_cast<size_t>(countSamplesVerify);
      if(cSamples != cSamplesVerify) {
         error("CreateInteractionDetector_R cSamples != cSamplesVerify");
      }

      aBag = reinterpret_cast<BagEbm *>(R_alloc(cSamples, static_cast<int>(sizeof(BagEbm))));
      EBM_ASSERT(nullptr != aBag); // this can't be nullptr since R_alloc uses R error handling

      cExpectedInitScores = 0;

      const int32_t * pSampleReplicationR = INTEGER(bag);
      BagEbm * pSampleReplication = aBag;
      const BagEbm * const pSampleReplicationEnd = aBag + cSamples;
      do {
         const int32_t replication = *pSampleReplicationR;
         if(IsConvertError<BagEbm>(replication)) {
            error("CreateInteractionDetector_R IsConvertError<BagEbm>(replication)");
         }
         if(0 != replication) {
            ++cExpectedInitScores;
         }
         *pSampleReplication = static_cast<BagEbm>(replication);
         ++pSampleReplicationR;
         ++pSampleReplication;
      } while(pSampleReplicationEnd != pSampleReplication);
   }

   const double * aInitScores = nullptr;
   if(NILSXP != TYPEOF(initScores)) {
      const IntEbm countInitScores = CountDoubles(initScores);
      if(countInitScores < 0) {
         // we've already logged any errors
         error("CreateInteractionDetector_R countInitScores < 0");
      }
      size_t cInitScores = static_cast<size_t>(countInitScores);
      if(cInitScores != cExpectedInitScores) {
         error("CreateInteractionDetector_R cInitScores != cExpectedInitScores");
      }
      aInitScores = REAL(initScores);
   }

   InteractionHandle interactionHandle;
   err = CreateInteractionDetector(
      pDataSet,
      aBag,
      aInitScores,
      nullptr,
      &interactionHandle
   );
   if(Error_None != err || nullptr == interactionHandle) {
      error("CreateInteractionDetector_R error in call to CreateInteractionDetector: %" ErrorEbmPrintf, err);
   }

   SEXP interactionHandleWrapped = R_MakeExternalPtr(static_cast<void *>(interactionHandle), R_NilValue, R_NilValue); // makes an EXTPTRSXP
   PROTECT(interactionHandleWrapped);

   R_RegisterCFinalizerEx(interactionHandleWrapped, &InteractionFinalizer, Rboolean::TRUE);

   UNPROTECT(1);
   return interactionHandleWrapped;
}

SEXP FreeInteractionDetector_R(SEXP interactionHandleWrapped) {
   InteractionFinalizer(interactionHandleWrapped);
   return R_NilValue;
}

SEXP CalcInteractionStrength_R(SEXP interactionHandleWrapped, SEXP featureIndexes, SEXP minSamplesLeaf) {
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
   if(Error_None != CalcInteractionStrength(interactionHandle, countDimensions, aFeatureIndexes, InteractionFlags_Default, minSamplesLeafEbm, &avgInteractionStrength)) {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrength_R CalcInteractionStrength returned error code");
      return R_NilValue;
   }

   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = static_cast<double>(avgInteractionStrength);
   UNPROTECT(1);
   return ret;
}

static const R_CallMethodDef g_exposedFunctions[] = {
   { "GenerateSeed_R", (DL_FUNC)&GenerateSeed_R, 2 },
   { "CutQuantile_R", (DL_FUNC)&CutQuantile_R, 4 },
   { "BinFeature_R", (DL_FUNC)&BinFeature_R, 3 },
   { "MeasureDataSetHeader_R", (DL_FUNC)&MeasureDataSetHeader_R, 3 },
   { "MeasureFeature_R", (DL_FUNC)&MeasureFeature_R, 5 },
   { "MeasureClassificationTarget_R", (DL_FUNC)&MeasureClassificationTarget_R, 2 },
   { "CreateDataSet_R", (DL_FUNC)&CreateDataSet_R, 1 },
   { "FreeDataSet_R", (DL_FUNC)&FreeDataSet_R, 1 },
   { "FillDataSetHeader_R", (DL_FUNC)&FillDataSetHeader_R, 5 },
   { "FillFeature_R", (DL_FUNC)&FillFeature_R, 7 },
   { "FillClassificationTarget_R", (DL_FUNC)&FillClassificationTarget_R, 4 },
   { "SampleWithoutReplacement_R", (DL_FUNC)&SampleWithoutReplacement_R, 4 },
   { "CreateBooster_R", (DL_FUNC)&CreateBooster_R, 7 },
   { "FreeBooster_R", (DL_FUNC)&FreeBooster_R, 1 },
   { "GenerateTermUpdate_R", (DL_FUNC)&GenerateTermUpdate_R, 5 },
   { "ApplyTermUpdate_R", (DL_FUNC)&ApplyTermUpdate_R, 1 },
   { "GetBestTermScores_R", (DL_FUNC)&GetBestTermScores_R, 2 },
   { "GetCurrentTermScores_R", (DL_FUNC)&GetCurrentTermScores_R, 2 },
   { "CreateInteractionDetector_R", (DL_FUNC)&CreateInteractionDetector_R, 3 },
   { "FreeInteractionDetector_R", (DL_FUNC)&FreeInteractionDetector_R, 1 },
   { "CalcInteractionStrength_R", (DL_FUNC)&CalcInteractionStrength_R, 3 },
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
