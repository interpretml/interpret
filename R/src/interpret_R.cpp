// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include <cmath> // std::isnan, std::isinf
#include <limits> // std::numeric_limits
#include <cstring> // memcpy, strcmp
#include <algorithm> // std::min, std::max

#include "libebm.h"
#include "logging.h"
#include "zones.h"

#include "common.hpp"

#include "ebm_internal.hpp"
#include "Feature.hpp"
#include "Term.hpp"
#include "BoosterCore.hpp"
#include "BoosterShell.hpp"
#include "InteractionCore.hpp"

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



INLINE_ALWAYS static double ConvertDouble(const SEXP sexp) {
   if(REALSXP != TYPEOF(sexp)) {
      error("ConvertDouble REALSXP != TYPEOF(sexp)");
   }
   if(R_xlen_t { 1 } != xlength(sexp)) {
      error("ConvertDouble R_xlen_t { 1 } != xlength(sexp)");
   }
   return REAL(sexp)[0];
}

INLINE_ALWAYS static IntEbm ConvertIndex(double index) {
   if(std::isnan(index)) {
      error("ConvertIndex std::isnan(index)");
   }
   if(index < 0) {
      error("ConvertIndex index < 0");
   }
   static constexpr double maxValid = EbmMin(
      double { R_XLEN_T_MAX }, 
      double { SAFE_FLOAT64_AS_INT64_MAX },
      static_cast<double>(std::numeric_limits<size_t>::max()),
      static_cast<double>(std::numeric_limits<IntEbm>::max()), 
      static_cast<double>(std::numeric_limits<R_xlen_t>::max())
   );
   if(maxValid < index) {
      error("ConvertIndex maxValid < index");
   }
   return static_cast<IntEbm>(index);
}

INLINE_ALWAYS static IntEbm ConvertIndex(const SEXP sexp) {
   return ConvertIndex(ConvertDouble(sexp));
}

INLINE_ALWAYS static IntEbm ConvertIndexApprox(double index) {
   if(std::isnan(index)) {
      error("ConvertIndexApprox std::isnan(index)");
   }
   static constexpr double minValid = EbmMax(
      double { -FLOAT64_TO_INT64_MAX },
      static_cast<double>(std::numeric_limits<IntEbm>::lowest())
   );
   if(index < minValid) {
      return minValid;
   }
   static constexpr double maxValid = EbmMin(
      double { FLOAT64_TO_INT64_MAX },
      static_cast<double>(std::numeric_limits<IntEbm>::max())
   );
   if(maxValid < index) {
      return maxValid;
   }
   return static_cast<IntEbm>(index);
}

INLINE_ALWAYS static IntEbm ConvertIndexApprox(const SEXP sexp) {
   return ConvertIndexApprox(ConvertDouble(sexp));
}

INLINE_ALWAYS static IntEbm ConvertInt(const SEXP sexp) {
   if(INTSXP != TYPEOF(sexp)) {
      error("ConvertInt INTSXP != TYPEOF(sexp)");
   }
   if(R_xlen_t { 1 } != xlength(sexp)) {
      error("ConvertInt R_xlen_t { 1 } != xlength(sexp)");
   }
   return INTEGER(sexp)[0];
}

INLINE_ALWAYS static BoolEbm ConvertBool(const SEXP sexp) {
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

static IntEbm CountInts(const SEXP a) {
   EBM_ASSERT(nullptr != a);
   if(INTSXP != TYPEOF(a)) {
      error("CountInts INTSXP != TYPEOF(a)");
   }
   const R_xlen_t c = xlength(a);
   if(IsConvertError<size_t>(c) || IsConvertError<IntEbm>(c)) {
      error("CountInts IsConvertError<size_t>(c) || IsConvertError<IntEbm>(c)");
   }
   return static_cast<IntEbm>(c);
}

static IntEbm CountDoubles(const SEXP a) {
   EBM_ASSERT(nullptr != a);
   if(REALSXP != TYPEOF(a)) {
      error("CountDoubles REALSXP != TYPEOF(a)");
   }
   const R_xlen_t c = xlength(a);
   if(IsConvertError<size_t>(c) || IsConvertError<IntEbm>(c)) {
      error("CountDoubles IsConvertError<size_t>(c) || IsConvertError<IntEbm>(c)");
   }
   return static_cast<IntEbm>(c);
}

static const IntEbm * ConvertDoublesToIndexes(const IntEbm c, const SEXP a) {
   EBM_ASSERT(0 <= c);
   EBM_ASSERT(nullptr != a);
   if(REALSXP != TYPEOF(a)) {
      error("ConvertDoublesToIndexes REALSXP != TYPEOF(a)");
   }
   IntEbm * aTo = nullptr;
   if(0 < c) {
      aTo = reinterpret_cast<IntEbm *>(R_alloc(c, static_cast<int>(sizeof(IntEbm))));
      EBM_ASSERT(nullptr != aTo); // R_alloc doesn't return nullptr, so we don't need to check aItems
      IntEbm * pTo = aTo;
      const IntEbm * const pToEnd = aTo + c;
      const double * pFrom = REAL(a);
      do {
         const double val = *pFrom;
         *pTo = ConvertIndex(val);
         ++pFrom;
         ++pTo;
      } while(pToEnd != pTo);
   }
   return aTo;
}

static IntEbm CountTotalDimensions(const size_t cTerms, const IntEbm * const acTermDimensions) {
   EBM_ASSERT(nullptr != acTermDimensions);

   size_t cTotalDimensions = size_t { 0 };
   if(0 != cTerms) {
      const IntEbm * pcTermDimensions = acTermDimensions;
      const IntEbm * const pcTermDimensionsEnd = acTermDimensions + cTerms;
      do {
         const IntEbm countDimensions = *pcTermDimensions;
         if(IsConvertError<size_t>(countDimensions)) {
            error("CountTotalDimensions IsConvertError<size_t>(countDimensions)");
         }
         const size_t cDimensions = static_cast<size_t>(countDimensions);
         if(IsAddError(cTotalDimensions, cDimensions)) {
            error("CountTotalDimensions IsAddError(cTotalDimensions, cDimensions)");
         }
         cTotalDimensions += cDimensions;
         ++pcTermDimensions;
      } while(pcTermDimensionsEnd != pcTermDimensions);
      if(IsConvertError<IntEbm>(cTotalDimensions)) {
         error("CountTotalDimensions IsConvertError<IntEbm>(cTotalDimensions)");
      }
   }
   return static_cast<IntEbm>(cTotalDimensions);
}

static void RngFinalizer(SEXP rngHandleWrapped) {
   EBM_ASSERT(nullptr != rngHandleWrapped); // shouldn't be possible
   if(EXTPTRSXP == TYPEOF(rngHandleWrapped)) {
      void * const rngHandle = R_ExternalPtrAddr(rngHandleWrapped);
      if(nullptr != rngHandle) {
         R_ClearExternalPtr(rngHandleWrapped);
         free(rngHandle);
      }
   }
}

static void DataSetFinalizer(SEXP dataSetHandleWrapped) {
   EBM_ASSERT(nullptr != dataSetHandleWrapped); // shouldn't be possible
   if(EXTPTRSXP == TYPEOF(dataSetHandleWrapped)) {
      void * const dataSetHandle = R_ExternalPtrAddr(dataSetHandleWrapped);
      if(nullptr != dataSetHandle) {
         R_ClearExternalPtr(dataSetHandleWrapped);
         free(dataSetHandle);
      }
   }
}

static void BoostingFinalizer(SEXP boosterHandleWrapped) {
   EBM_ASSERT(nullptr != boosterHandleWrapped); // shouldn't be possible
   if(EXTPTRSXP == TYPEOF(boosterHandleWrapped)) {
      const BoosterHandle boosterHandle = static_cast<BoosterHandle>(R_ExternalPtrAddr(boosterHandleWrapped));
      if(nullptr != boosterHandle) {
         R_ClearExternalPtr(boosterHandleWrapped);
         FreeBooster(boosterHandle);
      }
   }
}

static void InteractionFinalizer(SEXP interactionHandleWrapped) {
   EBM_ASSERT(nullptr != interactionHandleWrapped); // shouldn't be possible
   if(EXTPTRSXP == TYPEOF(interactionHandleWrapped)) {
      const InteractionHandle interactionHandle = static_cast<InteractionHandle>(R_ExternalPtrAddr(interactionHandleWrapped));
      if(nullptr != interactionHandle) {
         R_ClearExternalPtr(interactionHandleWrapped);
         FreeInteractionDetector(interactionHandle);
      }
   }
}

SEXP CreateRNG_R(SEXP seed) {
   EBM_ASSERT(nullptr != seed);

   const SeedEbm seedLocal = ConvertInt(seed);

   void * const rngHandle = malloc(static_cast<size_t>(MeasureRNG()));

   InitRNG(seedLocal, rngHandle);

   SEXP rngHandleWrapped = R_MakeExternalPtr(rngHandle, R_NilValue, R_NilValue); // makes an EXTPTRSXP
   PROTECT(rngHandleWrapped);
   
   R_RegisterCFinalizerEx(rngHandleWrapped, &RngFinalizer, Rboolean::TRUE);
   
   UNPROTECT(1);
   return rngHandleWrapped;
}

SEXP CutQuantile_R(SEXP featureVals, SEXP minSamplesBin, SEXP isRounded, SEXP countCuts) {
   EBM_ASSERT(nullptr != featureVals);
   EBM_ASSERT(nullptr != minSamplesBin);
   EBM_ASSERT(nullptr != isRounded);
   EBM_ASSERT(nullptr != countCuts);

   const IntEbm countSamples = CountDoubles(featureVals);
   const double * const aFeatureVals = REAL(featureVals);

   const IntEbm samplesBinMin = ConvertIndexApprox(minSamplesBin);

   BoolEbm bRounded = ConvertBool(isRounded);

   IntEbm cCuts = ConvertIndex(countCuts);

   // TODO: we should allocate the buffer that we're doing to return here directly
   double * const aCutsLowerBoundInclusive = reinterpret_cast<double *>(
      R_alloc(static_cast<size_t>(cCuts), static_cast<int>(sizeof(double))));
   EBM_ASSERT(nullptr != aCutsLowerBoundInclusive); // R_alloc doesn't return nullptr, so we don't need to check aItems

   const ErrorEbm err = CutQuantile(
      countSamples,
      aFeatureVals,
      samplesBinMin,
      bRounded,
      &cCuts,
      aCutsLowerBoundInclusive
   );
   if(Error_None != err) {
      error("CutQuantile returned error code: %" ErrorEbmPrintf, err);
   }

   const SEXP ret = PROTECT(allocVector(REALSXP, static_cast<R_xlen_t>(cCuts)));

   // we've allocated this memory, so it should be reachable, so these numbers should multiply
   EBM_ASSERT(!IsMultiplyError(sizeof(*aCutsLowerBoundInclusive), static_cast<size_t>(cCuts)));

   if(0 != cCuts) {
      double * pRet = REAL(ret);
      const double * pCutsLowerBoundInclusive = aCutsLowerBoundInclusive;
      const double * const pCutsLowerBoundInclusiveEnd = aCutsLowerBoundInclusive + static_cast<size_t>(cCuts);
      do {
         *pRet = *pCutsLowerBoundInclusive;
         ++pRet;
         ++pCutsLowerBoundInclusive;
      } while(pCutsLowerBoundInclusiveEnd != pCutsLowerBoundInclusive);
   }

   UNPROTECT(1);
   return ret;
}

SEXP Discretize_R(SEXP featureVals, SEXP cutsLowerBoundInclusive, SEXP binIndexesOut) {
   EBM_ASSERT(nullptr != featureVals);
   EBM_ASSERT(nullptr != cutsLowerBoundInclusive);
   EBM_ASSERT(nullptr != binIndexesOut);

   const IntEbm cSamples = CountDoubles(featureVals);
   const double * const aFeatureVals = REAL(featureVals);

   const IntEbm cCuts = CountDoubles(cutsLowerBoundInclusive);
   if(SAFE_FLOAT64_AS_INT64_MAX - 2 < cCuts) {
      // if the number of cuts is low enough, we don't need to check if the bin indexes below exceed our safe float max
      // the highest bin index is +2 from the number of cuts, although the # of bins is 1 higher but we're ok with that
      error("Discretize_R SAFE_FLOAT64_AS_INT64_MAX - 2 < cCuts");
   }
   const double * const aCutsLowerBoundInclusive = REAL(cutsLowerBoundInclusive);

   const IntEbm cBinIndexesOut = CountDoubles(binIndexesOut);
   if(cSamples != cBinIndexesOut) {
      error("Discretize_R cSamples != cBinIndexesOut");
   }

   if(0 != cSamples) {
      IntEbm * const aiBins = 
         reinterpret_cast<IntEbm *>(R_alloc(static_cast<size_t>(cSamples), static_cast<int>(sizeof(IntEbm))));
      EBM_ASSERT(nullptr != aiBins); // this can't be nullptr since R_alloc uses R error handling

      const ErrorEbm err = Discretize(cSamples, aFeatureVals, cCuts, aCutsLowerBoundInclusive, aiBins);
      if(Error_None != err) {
         error("Discretize returned error code: %" ErrorEbmPrintf, err);
      }

      double * pBinIndexesOut = REAL(binIndexesOut);
      const IntEbm * piBin = aiBins;
      const IntEbm * const piBinsEnd = aiBins + static_cast<size_t>(cSamples);
      do {
         const IntEbm iBin = *piBin;
         EBM_ASSERT(iBin <= SAFE_FLOAT64_AS_INT64_MAX); // we checked the number of cuts above
         *pBinIndexesOut = static_cast<double>(iBin);
         ++pBinIndexesOut;
         ++piBin;
      } while(piBinsEnd != piBin);
   }
   return R_NilValue;
}

SEXP MeasureDataSetHeader_R(SEXP countFeatures, SEXP countWeights, SEXP countTargets) {
   EBM_ASSERT(nullptr != countFeatures);
   EBM_ASSERT(nullptr != countWeights);
   EBM_ASSERT(nullptr != countTargets);

   const IntEbm cFeatures = ConvertIndex(countFeatures);
   const IntEbm cWeights = ConvertIndex(countWeights);
   const IntEbm cTargets = ConvertIndex(countTargets);

   const IntEbm countBytes = MeasureDataSetHeader(cFeatures, cWeights, cTargets);
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

SEXP MeasureFeature_R(SEXP countBins, SEXP isMissing, SEXP isUnknown, SEXP isNominal, SEXP binIndexes) {
   EBM_ASSERT(nullptr != countBins);
   EBM_ASSERT(nullptr != isMissing);
   EBM_ASSERT(nullptr != isUnknown);
   EBM_ASSERT(nullptr != isNominal);
   EBM_ASSERT(nullptr != binIndexes);

   const IntEbm cBins = ConvertIndex(countBins);
   BoolEbm bMissing = ConvertBool(isMissing);
   BoolEbm bUnknown = ConvertBool(isUnknown);
   BoolEbm bNominal = ConvertBool(isNominal);

   const IntEbm cSamples = CountDoubles(binIndexes);
   const IntEbm * const aiBins = ConvertDoublesToIndexes(cSamples, binIndexes);

   const IntEbm countBytes = MeasureFeature(
      cBins,
      bMissing,
      bUnknown,
      bNominal,
      cSamples,
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

SEXP MeasureClassificationTarget_R(SEXP countClasses, SEXP targets) {
   EBM_ASSERT(nullptr != countClasses);
   EBM_ASSERT(nullptr != targets);

   const IntEbm cClasses = ConvertIndex(countClasses);

   const IntEbm cSamples = CountDoubles(targets);
   const IntEbm * const aTargets = ConvertDoublesToIndexes(cSamples, targets);

   const IntEbm countBytes = MeasureClassificationTarget(
      cClasses,
      cSamples,
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

   const IntEbm cBytes = ConvertIndex(countBytes);

   void * const dataSetHandle = malloc(static_cast<size_t>(cBytes));

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

   const IntEbm cFeatures = ConvertIndex(countFeatures);
   const IntEbm cWeights = ConvertIndex(countWeights);
   const IntEbm cTargets = ConvertIndex(countTargets);
   const IntEbm cBytesAllocated = ConvertIndex(countBytesAllocated);

   if(EXTPTRSXP != TYPEOF(fillMemWrapped)) {
      error("FillDataSetHeader_R EXTPTRSXP != TYPEOF(fillMemWrapped)");
   }
   void * const pDataset = R_ExternalPtrAddr(fillMemWrapped);

   const ErrorEbm err = FillDataSetHeader(
      cFeatures,
      cWeights,
      cTargets,
      cBytesAllocated,
      pDataset
   );
   if(Error_None != err) {
      error("FillDataSetHeader returned error code: %" ErrorEbmPrintf, err);
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

   const IntEbm cBins = ConvertIndex(countBins);
   BoolEbm bMissing = ConvertBool(isMissing);
   BoolEbm bUnknown = ConvertBool(isUnknown);
   BoolEbm bNominal = ConvertBool(isNominal);

   const IntEbm cSamples = CountDoubles(binIndexes);
   const IntEbm * const aiBins = ConvertDoublesToIndexes(cSamples, binIndexes);

   const IntEbm cBytesAllocated = ConvertIndex(countBytesAllocated);

   if(EXTPTRSXP != TYPEOF(fillMemWrapped)) {
      error("FillFeature_R EXTPTRSXP != TYPEOF(fillMemWrapped)");
   }
   void * const pDataset = R_ExternalPtrAddr(fillMemWrapped);

   const ErrorEbm err = FillFeature(
      cBins,
      bMissing,
      bUnknown,
      bNominal,
      cSamples,
      aiBins,
      cBytesAllocated,
      pDataset
   );
   if(Error_None != err) {
      error("FillFeature returned error code: %" ErrorEbmPrintf, err);
   }

   return R_NilValue;
}

SEXP FillClassificationTarget_R(SEXP countClasses, SEXP targets, SEXP countBytesAllocated, SEXP fillMemWrapped) {
   EBM_ASSERT(nullptr != countClasses);
   EBM_ASSERT(nullptr != targets);
   EBM_ASSERT(nullptr != countBytesAllocated);
   EBM_ASSERT(nullptr != fillMemWrapped);

   const IntEbm cClasses = ConvertIndex(countClasses);

   const IntEbm cSamples = CountDoubles(targets);
   const IntEbm * const aTargets = ConvertDoublesToIndexes(cSamples, targets);

   const IntEbm cBytesAllocated = ConvertIndex(countBytesAllocated);

   if(EXTPTRSXP != TYPEOF(fillMemWrapped)) {
      error("FillClassificationTarget_R EXTPTRSXP != TYPEOF(fillMemWrapped)");
   }
   void * const pDataset = R_ExternalPtrAddr(fillMemWrapped);

   const ErrorEbm err = FillClassificationTarget(
      cClasses,
      cSamples,
      aTargets,
      cBytesAllocated,
      pDataset
   );
   if(Error_None != err) {
      error("FillClassificationTarget returned error code: %" ErrorEbmPrintf, err);
   }

   return R_NilValue;
}

SEXP SampleWithoutReplacement_R(SEXP rng, SEXP countTrainingSamples, SEXP countValidationSamples, SEXP bagOut) {
   EBM_ASSERT(nullptr != rng);
   EBM_ASSERT(nullptr != countTrainingSamples);
   EBM_ASSERT(nullptr != countValidationSamples);
   EBM_ASSERT(nullptr != bagOut);

   void * pRng = nullptr;
   if(NILSXP != TYPEOF(rng)) {
      if(EXTPTRSXP != TYPEOF(rng)) {
         error("SampleWithoutReplacement_R EXTPTRSXP != TYPEOF(rng)");
      }
      pRng = R_ExternalPtrAddr(rng);
   }

   const IntEbm cTrainingSamples = ConvertIndex(countTrainingSamples);
   const IntEbm cValidationSamples = ConvertIndex(countValidationSamples);
   if(IsAddError(static_cast<size_t>(cTrainingSamples), static_cast<size_t>(cValidationSamples))) {
      error("SampleWithoutReplacement_R IsAddError(static_cast<size_t>(cTrainingSamples), static_cast<size_t>(cValidationSamples))");
   }

   const size_t cSamples = static_cast<size_t>(CountInts(bagOut));

   if(static_cast<size_t>(cTrainingSamples) + static_cast<size_t>(cValidationSamples) != cSamples) {
      error("SampleWithoutReplacement_R static_cast<size_t>(cTrainingSamples) + static_cast<size_t>(cValidationSamples) != cSamples");
   }

   if(0 != cSamples) {
      BagEbm * const aBag = 
         reinterpret_cast<BagEbm *>(R_alloc(cSamples, static_cast<int>(sizeof(BagEbm))));
      EBM_ASSERT(nullptr != aBag); // this can't be nullptr since R_alloc uses R error handling

      const ErrorEbm err = SampleWithoutReplacement(
         pRng,
         cTrainingSamples,
         cValidationSamples,
         aBag
      );
      if(Error_None != err) {
         error("SampleWithoutReplacementFillDataSetHeader returned error code: %" ErrorEbmPrintf, err);
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
   return R_NilValue;
}

SEXP CreateBooster_R(
   SEXP rng,
   SEXP dataSetWrapped,
   SEXP bag,
   SEXP initScores,
   SEXP dimensionCounts,
   SEXP featureIndexes,
   SEXP countInnerBags
) {
   EBM_ASSERT(nullptr != rng);
   EBM_ASSERT(nullptr != dataSetWrapped);
   EBM_ASSERT(nullptr != bag);
   EBM_ASSERT(nullptr != initScores);
   EBM_ASSERT(nullptr != dimensionCounts);
   EBM_ASSERT(nullptr != featureIndexes);
   EBM_ASSERT(nullptr != countInnerBags);

   ErrorEbm err;

   void * pRng = nullptr;
   if(NILSXP != TYPEOF(rng)) {
      if(EXTPTRSXP != TYPEOF(rng)) {
         error("CreateBooster_R EXTPTRSXP != TYPEOF(rng)");
      }
      pRng = R_ExternalPtrAddr(rng);
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
      error("ExtractDataSetHeader returned error code: %" ErrorEbmPrintf, err);
   }
   const size_t cSamples = static_cast<size_t>(countSamples); // we trust our internal code that this is convertible

   BagEbm * aBag = nullptr;
   size_t cExpectedInitScores = cSamples;
   if(NILSXP != TYPEOF(bag)) {
      const size_t cSamplesVerify = static_cast<size_t>(CountInts(bag));
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
      size_t cInitScores = static_cast<size_t>(countInitScores);
      if(cInitScores != cExpectedInitScores) {
         error("CreateBooster_R cInitScores != cExpectedInitScores");
      }
      aInitScores = REAL(initScores);
   }

   const IntEbm cTerms = CountDoubles(dimensionCounts);
   const IntEbm * const acTermDimensions = ConvertDoublesToIndexes(cTerms, dimensionCounts);
   const IntEbm cTotalDimensionsCheck = CountTotalDimensions(static_cast<size_t>(cTerms), acTermDimensions);

   const IntEbm cTotalDimensionsActual = CountDoubles(featureIndexes);
   if(cTotalDimensionsActual != cTotalDimensionsCheck) {
      error("CreateBooster_R cTotalDimensionsActual != cTotalDimensionsCheck");
   }
   const IntEbm * const aiTermFeatures = ConvertDoublesToIndexes(cTotalDimensionsActual, featureIndexes);

   const IntEbm cInnerBags = ConvertIndex(countInnerBags);

   BoosterHandle boosterHandle;
   err = CreateBooster(
      pRng,
      pDataSet,
      aBag,
      aInitScores,
      cTerms,
      acTermDimensions,
      aiTermFeatures,
      cInnerBags,
      CreateBoosterFlags_Default,
      ComputeFlags_Default,
      "log_loss",
      nullptr,
      &boosterHandle
   );
   if(Error_None != err || nullptr == boosterHandle) {
      error("CreateBooster returned error code: %" ErrorEbmPrintf, err);
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
   SEXP rng,
   SEXP boosterHandleWrapped,
   SEXP indexTerm,
   SEXP learningRate,
   SEXP minSamplesLeaf,
   SEXP leavesMax
) {
   EBM_ASSERT(nullptr != rng);
   EBM_ASSERT(nullptr != boosterHandleWrapped);
   EBM_ASSERT(nullptr != indexTerm);
   EBM_ASSERT(nullptr != learningRate);
   EBM_ASSERT(nullptr != minSamplesLeaf);
   EBM_ASSERT(nullptr != leavesMax);

   void * pRng = nullptr;
   if(NILSXP != TYPEOF(rng)) {
      if(EXTPTRSXP != TYPEOF(rng)) {
         error("GenerateTermUpdate_R EXTPTRSXP != TYPEOF(rng)");
      }
      pRng = R_ExternalPtrAddr(rng);
   }

   if(EXTPTRSXP != TYPEOF(boosterHandleWrapped)) {
      error("GenerateTermUpdate_R EXTPTRSXP != TYPEOF(boosterHandleWrapped)");
   }
   const BoosterHandle boosterHandle = static_cast<BoosterHandle>(R_ExternalPtrAddr(boosterHandleWrapped));
   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      error("GenerateTermUpdate_R nullptr == pBoosterShell");
   }

   const IntEbm iTerm = ConvertIndex(indexTerm);

   const double learningRateLocal = ConvertDouble(learningRate);

   const IntEbm samplesLeafMin = ConvertIndexApprox(minSamplesLeaf);

   const IntEbm cDimensions = CountDoubles(leavesMax);
   const IntEbm * const aLeavesMax = ConvertDoublesToIndexes(cDimensions, leavesMax);
   if(pBoosterShell->GetBoosterCore()->GetCountTerms() <= static_cast<size_t>(iTerm)) {
      error("GenerateTermUpdate_R pBoosterShell->GetBoosterCore()->GetCountTerms() <= static_cast<size_t>(iTerm)");
   }
   if(static_cast<size_t>(cDimensions) < pBoosterShell->GetBoosterCore()->GetTerms()[static_cast<size_t>(iTerm)]->GetCountDimensions()) {
      error("GenerateTermUpdate_R static_cast<size_t>(cDimensions) < pBoosterShell->GetBoosterCore()->GetTerms()[static_cast<size_t>(iTerm)]->GetCountDimensions()");
   }

   double avgGain;

   const ErrorEbm err = GenerateTermUpdate(
      pRng,
      boosterHandle,
      iTerm,
      TermBoostFlags_Default,
      learningRateLocal,
      samplesLeafMin,
      aLeavesMax,
      &avgGain
   );
   if(Error_None != err) {
      error("GenerateTermUpdate returned error code: %" ErrorEbmPrintf, err);
   }

   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = avgGain;
   UNPROTECT(1);
   return ret;
}

SEXP ApplyTermUpdate_R(SEXP boosterHandleWrapped) {
   EBM_ASSERT(nullptr != boosterHandleWrapped);

   if(EXTPTRSXP != TYPEOF(boosterHandleWrapped)) {
      error("ApplyTermUpdate_R EXTPTRSXP != TYPEOF(boosterHandleWrapped)");
   }
   const BoosterHandle boosterHandle = static_cast<BoosterHandle>(R_ExternalPtrAddr(boosterHandleWrapped));
   // we don't use boosterHandle in this function, so let ApplyTermUpdate check if it's null or invalid

   double avgValidationMetric;
   const ErrorEbm err = ApplyTermUpdate(boosterHandle, &avgValidationMetric);
   if(Error_None != err) {
      error("ApplyTermUpdate returned error code: %" ErrorEbmPrintf, err);
   }

   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = avgValidationMetric;
   UNPROTECT(1);
   return ret;
}

SEXP GetBestTermScores_R(SEXP boosterHandleWrapped, SEXP indexTerm) {
   EBM_ASSERT(nullptr != boosterHandleWrapped); // shouldn't be possible
   EBM_ASSERT(nullptr != indexTerm); // shouldn't be possible

   if(EXTPTRSXP != TYPEOF(boosterHandleWrapped)) {
      error("GetBestTermScores_R EXTPTRSXP != TYPEOF(boosterHandleWrapped)");
   }
   const BoosterHandle boosterHandle = static_cast<BoosterHandle>(R_ExternalPtrAddr(boosterHandleWrapped));
   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      error("GetBestTermScores_R nullptr == pBoosterShell");
   }
   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();

   const IntEbm iTerm = ConvertIndex(indexTerm);

   if(pBoosterCore->GetCountTerms() <= static_cast<size_t>(iTerm)) {
      error("GetBestTermScores_R pBoosterCore->GetCountTerms() <= static_cast<size_t>(iTerm)");
   }

   size_t cTensorScores = pBoosterCore->GetCountScores();
   if(size_t { 0 } != cTensorScores) {
      const Term * const pTerm = pBoosterCore->GetTerms()[static_cast<size_t>(iTerm)];
      const size_t cDimensions = pTerm->GetCountDimensions();
      if(0 != cDimensions) {
         const TermFeature * pTermFeature = pTerm->GetTermFeatures();
         const TermFeature * const pTermFeaturesEnd = &pTermFeature[cDimensions];
         do {
            const FeatureBoosting * const pFeature = pTermFeature->m_pFeature;
            const size_t cBins = pFeature->GetCountBins();
            EBM_ASSERT(!IsMultiplyError(cTensorScores, cBins)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
            cTensorScores *= cBins;
            ++pTermFeature;
         } while(pTermFeaturesEnd != pTermFeature);
      }
      if(IsConvertError<R_xlen_t>(cTensorScores)) {
         error("GetBestTermScores_R IsConvertError<R_xlen_t>(cTensorScores)");
      }
   }
   SEXP ret = PROTECT(allocVector(REALSXP, static_cast<R_xlen_t>(cTensorScores)));
   EBM_ASSERT(!IsMultiplyError(sizeof(double), cTensorScores)); // we've allocated this memory, so it should be reachable, so these numbers should multiply

   const ErrorEbm err = GetBestTermScores(boosterHandle, iTerm, REAL(ret));

   UNPROTECT(1);

   if(Error_None != err) {
      error("GetBestTermScores returned error code: %" ErrorEbmPrintf, err);
   }
   return ret;
}

SEXP GetCurrentTermScores_R(SEXP boosterHandleWrapped, SEXP indexTerm) {
   EBM_ASSERT(nullptr != boosterHandleWrapped); // shouldn't be possible
   EBM_ASSERT(nullptr != indexTerm); // shouldn't be possible

   if(EXTPTRSXP != TYPEOF(boosterHandleWrapped)) {
      error("GetCurrentTermScores_R EXTPTRSXP != TYPEOF(boosterHandleWrapped)");
   }
   const BoosterHandle boosterHandle = static_cast<BoosterHandle>(R_ExternalPtrAddr(boosterHandleWrapped));
   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      error("GetCurrentTermScores_R nullptr == pBoosterShell");
   }
   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();

   const IntEbm iTerm = ConvertIndex(indexTerm);

   if(pBoosterCore->GetCountTerms() <= static_cast<size_t>(iTerm)) {
      error("GetCurrentTermScores_R pBoosterCore->GetCountTerms() <= static_cast<size_t>(iTerm)");
   }

   size_t cTensorScores = pBoosterCore->GetCountScores();
   if(ptrdiff_t { 0 } != cTensorScores) {
      const Term * const pTerm = pBoosterCore->GetTerms()[static_cast<size_t>(iTerm)];
      const size_t cDimensions = pTerm->GetCountDimensions();
      if(0 != cDimensions) {
         const TermFeature * pTermFeature = pTerm->GetTermFeatures();
         const TermFeature * const pTermFeaturesEnd = &pTermFeature[cDimensions];
         do {
            const FeatureBoosting * const pFeature = pTermFeature->m_pFeature;
            const size_t cBins = pFeature->GetCountBins();
            EBM_ASSERT(!IsMultiplyError(cTensorScores, cBins)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
            cTensorScores *= cBins;
            ++pTermFeature;
         } while(pTermFeaturesEnd != pTermFeature);
      }
      if(IsConvertError<R_xlen_t>(cTensorScores)) {
         error("GetCurrentTermScores_R IsConvertError<R_xlen_t>(cTensorScores)");
      }
   }
   SEXP ret = PROTECT(allocVector(REALSXP, static_cast<R_xlen_t>(cTensorScores)));
   EBM_ASSERT(!IsMultiplyError(sizeof(double), cTensorScores)); // we've allocated this memory, so it should be reachable, so these numbers should multiply

   const ErrorEbm err = GetCurrentTermScores(boosterHandle, iTerm, REAL(ret));

   UNPROTECT(1);

   if(Error_None != err) {
      error("GetCurrentTermScores returned error code: %" ErrorEbmPrintf, err);
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
      error("ExtractDataSetHeader returned error code: %" ErrorEbmPrintf, err);
   }
   const size_t cSamples = static_cast<size_t>(countSamples); // we trust our internal code that this is convertible

   BagEbm * aBag = nullptr;
   size_t cExpectedInitScores = cSamples;
   if(NILSXP != TYPEOF(bag)) {
      const size_t cSamplesVerify = static_cast<size_t>(CountInts(bag));
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
      CreateInteractionFlags_Default,
      ComputeFlags_Default,
      "log_loss",
      nullptr,
      &interactionHandle
   );
   if(Error_None != err || nullptr == interactionHandle) {
      error("CreateInteractionDetector returned error code: %" ErrorEbmPrintf, err);
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

SEXP CalcInteractionStrength_R(SEXP interactionHandleWrapped, SEXP featureIndexes, SEXP maxCardinality, SEXP minSamplesLeaf) {
   EBM_ASSERT(nullptr != interactionHandleWrapped); // shouldn't be possible
   EBM_ASSERT(nullptr != featureIndexes); // shouldn't be possible
   EBM_ASSERT(nullptr != maxCardinality);
   EBM_ASSERT(nullptr != minSamplesLeaf);

   if(EXTPTRSXP != TYPEOF(interactionHandleWrapped)) {
      error("CalcInteractionStrength_R EXTPTRSXP != TYPEOF(interactionHandleWrapped)");
   }
   const InteractionHandle interactionHandle = static_cast<InteractionHandle>(R_ExternalPtrAddr(interactionHandleWrapped));
   if(nullptr == interactionHandle) {
      error("CalcInteractionStrength_R nullptr == interactionHandle");
   }

   const IntEbm cDimensions = CountDoubles(featureIndexes);
   const IntEbm * const aFeatureIndexes = ConvertDoublesToIndexes(cDimensions, featureIndexes);

   const IntEbm cardinalityMax = ConvertIndexApprox(maxCardinality);
   const IntEbm samplesLeafMin = ConvertIndexApprox(minSamplesLeaf);

   double avgInteractionStrength;
   const ErrorEbm err = CalcInteractionStrength(
      interactionHandle, 
      cDimensions, 
      aFeatureIndexes, 
      CalcInteractionFlags_Default, 
      cardinalityMax, 
      samplesLeafMin, 
      &avgInteractionStrength
   );
   if(Error_None != err) {
      error("CalcInteractionStrength returned error code: %" ErrorEbmPrintf, err);
   }

   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = avgInteractionStrength;
   UNPROTECT(1);
   return ret;
}

static const R_CallMethodDef g_exposedFunctions[] = {
   { "CreateRNG_R", (DL_FUNC)&CreateRNG_R, 1 },
   { "CutQuantile_R", (DL_FUNC)&CutQuantile_R, 4 },
   { "Discretize_R", (DL_FUNC)&Discretize_R, 3 },
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
   { "GenerateTermUpdate_R", (DL_FUNC)&GenerateTermUpdate_R, 6 },
   { "ApplyTermUpdate_R", (DL_FUNC)&ApplyTermUpdate_R, 1 },
   { "GetBestTermScores_R", (DL_FUNC)&GetBestTermScores_R, 2 },
   { "GetCurrentTermScores_R", (DL_FUNC)&GetCurrentTermScores_R, 2 },
   { "CreateInteractionDetector_R", (DL_FUNC)&CreateInteractionDetector_R, 3 },
   { "FreeInteractionDetector_R", (DL_FUNC)&FreeInteractionDetector_R, 1 },
   { "CalcInteractionStrength_R", (DL_FUNC)&CalcInteractionStrength_R, 4 },
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
