// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include <cmath> // std::isnan, std::isinf

#include "ebm_native.h"
#include "EbmBoostingState.h"
#include "EbmInteractionState.h"

#include <Rinternals.h>
#include <R_ext/Visibility.h>

// when R compiles this library, on some systems it can generate a "NOTE installed size is.." meaning the C++ compiled into a library produces too big a
// library.  We would want to disable the -g flag (with -g0), but according to this, it's not possible currently:
// https://stat.ethz.ch/pipermail/r-devel/2016-October/073273.html

// TODO: switch logging to use the R logging infrastructure when invoked from R, BUT calling error or warning will generate longjumps, which 
//   bypass the regular return mechanisms.  We need to use R_tryCatch (which is older than R_UnwindProtect) to not leak memory that we allocate 
//   before calling the R error or warning functions

EBM_INLINE bool IsSingleDoubleVector(const SEXP sexp) {
   if(REALSXP != TYPEOF(sexp)) {
      return false;
   }
   if(R_xlen_t { 1 } != xlength(sexp)) {
      return false;
   }
   return true;
}

EBM_INLINE bool IsSingleIntVector(const SEXP sexp) {
   if(INTSXP != TYPEOF(sexp)) {
      return false;
   }
   if(R_xlen_t { 1 } != xlength(sexp)) {
      return false;
   }
   return true;
}

EBM_INLINE bool IsDoubleToIntEbmTypeIndexValid(const double val) {
   if(std::isnan(val)) {
      return false;
   }
   static_assert(std::numeric_limits<double>::is_iec559, "we need is_iec559 to know that comparisons to infinity and -infinity to normal numbers work");
   if(val < double { 0 }) {
      return false;
   }
   if(std::min(
      static_cast<double>(std::numeric_limits<size_t>::max()), 
      std::min(double { R_XLEN_T_MAX }, static_cast<double>(std::numeric_limits<IntEbmType>::max()))) < val
   ) {
      return false;
   }
   return true;
}

void BoostingFinalizer(SEXP boostingRPointer) {
   EBM_ASSERT(nullptr != boostingRPointer); // shouldn't be possible
   if(EXTPTRSXP == TYPEOF(boostingRPointer)) {
      PEbmBoosting pEbmBoosting = static_cast<PEbmBoosting>(R_ExternalPtrAddr(boostingRPointer));
      if(nullptr != pEbmBoosting) {
         FreeBoosting(pEbmBoosting);
         R_ClearExternalPtr(boostingRPointer);
      }
   }
}

void InteractionFinalizer(SEXP interactionRPointer) {
   EBM_ASSERT(nullptr != interactionRPointer); // shouldn't be possible
   if(EXTPTRSXP == TYPEOF(interactionRPointer)) {
      PEbmInteraction pInteraction = static_cast<PEbmInteraction>(R_ExternalPtrAddr(interactionRPointer));
      if(nullptr != pInteraction) {
         FreeInteraction(pInteraction);
         R_ClearExternalPtr(interactionRPointer);
      }
   }
}

EbmNativeFeature * ConvertFeatures(const SEXP features, size_t * const pcFeatures) {
   if(VECSXP != TYPEOF(features)) {
      LOG_0(TraceLevelError, "ERROR ConvertFeatures VECSXP != TYPEOF(features)");
      return nullptr;
   }
   const R_xlen_t countFeaturesR = xlength(features);
   if(!IsNumberConvertable<size_t, R_xlen_t>(countFeaturesR)) {
      LOG_0(TraceLevelError, "ERROR ConvertFeatures !IsNumberConvertable<size_t, R_xlen_t>(countFeaturesR)");
      return nullptr;
   }
   const size_t cFeatures = static_cast<size_t>(countFeaturesR);
   if(!IsNumberConvertable<IntEbmType, size_t>(cFeatures)) {
      LOG_0(TraceLevelError, "ERROR ConvertFeatures !IsNumberConvertable<IntEbmType, size_t>(cFeatures)");
      return nullptr;
   }
   *pcFeatures = cFeatures;

   EbmNativeFeature * const aFeatures = reinterpret_cast<EbmNativeFeature *>(R_alloc(cFeatures, static_cast<int>(sizeof(EbmNativeFeature))));
   // R_alloc doesn't return nullptr, so we don't need to check aFeatures
   EbmNativeFeature * pFeature = aFeatures;
   for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
      const SEXP oneFeature = VECTOR_ELT(features, iFeature);
      EBM_ASSERT(nullptr != oneFeature);
      if(VECSXP != TYPEOF(oneFeature)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatures VECSXP != TYPEOF(oneFeature)");
         return nullptr;
      }
      constexpr size_t cItems = 3;
      if(R_xlen_t { cItems } != xlength(oneFeature)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatures R_xlen_t { cItems } != xlength(oneFeature)");
         return nullptr;
      }
      const SEXP fieldNames = getAttrib(oneFeature, R_NamesSymbol);
      EBM_ASSERT(nullptr != fieldNames);
      if(STRSXP != TYPEOF(fieldNames)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatures STRSXP != TYPEOF(fieldNames)");
         return nullptr;
      }
      if(R_xlen_t { cItems } != xlength(fieldNames)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatures R_xlen_t { cItems } != xlength(fieldNames)");
         return nullptr;
      }

      bool bCountBinsFound = false;
      bool bHasMissingFound = false;
      bool bFeatureTypeFound = false;
      for(size_t iName = 0; iName < cItems; ++iName) {
         const SEXP nameR = STRING_ELT(fieldNames, iName);
         if(CHARSXP != TYPEOF(nameR)) {
            LOG_0(TraceLevelError, "ERROR ConvertFeatures CHARSXP != TYPEOF(nameR)");
            return nullptr;
         }
         const char * pName = CHAR(nameR);
         if(0 == strcmp("n_bins", pName)) {
            if(bCountBinsFound) {
               LOG_0(TraceLevelError, "ERROR ConvertFeatures bCountBinsFound");
               return nullptr;
            }

            SEXP val = VECTOR_ELT(oneFeature, iName);
            if(REALSXP != TYPEOF(val)) {
               LOG_0(TraceLevelError, "ERROR ConvertFeatures REALSXP != TYPEOF(value)");
               return nullptr;
            }
            if(1 != xlength(val)) {
               LOG_0(TraceLevelError, "ERROR ConvertFeatures 1 != xlength(val)");
               return nullptr;
            }

            double countBinsDouble = REAL(val)[0];
            if(!IsDoubleToIntEbmTypeIndexValid(countBinsDouble)) {
               LOG_0(TraceLevelError, "ERROR ConvertFeatures !IsDoubleToIntEbmTypeIndexValid(countBinsDouble)");
               return nullptr;
            }
            pFeature->countBins = static_cast<IntEbmType>(countBinsDouble);
            bCountBinsFound = true;
         } else if(0 == strcmp("has_missing", pName)) {
            if(bHasMissingFound) {
               LOG_0(TraceLevelError, "ERROR ConvertFeatures bHasMissingFound");
               return nullptr;
            }

            SEXP val = VECTOR_ELT(oneFeature, iName);
            if(LGLSXP != TYPEOF(val)) {
               LOG_0(TraceLevelError, "ERROR ConvertFeatures LGLSXP != TYPEOF(value)");
               return nullptr;
            }
            if(1 != xlength(val)) {
               LOG_0(TraceLevelError, "ERROR ConvertFeatures 1 != xlength(val)");
               return nullptr;
            }

            int hasMissing = LOGICAL(val)[0];
            pFeature->hasMissing = 0 != hasMissing ? 1 : 0;
            bHasMissingFound = true;
         } else if(0 == strcmp("feature_type", pName)) {
            if(bFeatureTypeFound) {
               LOG_0(TraceLevelError, "ERROR ConvertFeatures bFeatureTypeFound");
               return nullptr;
            }

            SEXP val = VECTOR_ELT(oneFeature, iName);

            if(STRSXP != TYPEOF(val)) {
               LOG_0(TraceLevelError, "ERROR ConvertFeatures STRSXP != TYPEOF(value)");
               return nullptr;
            }
            if(1 != xlength(val)) {
               LOG_0(TraceLevelError, "ERROR ConvertFeatures 1 != xlength(val)");
               return nullptr;
            }

            const SEXP featureTypeR = STRING_ELT(val, 0);
            if(CHARSXP != TYPEOF(featureTypeR)) {
               LOG_0(TraceLevelError, "ERROR ConvertFeatures CHARSXP != TYPEOF(featureTypeR)");
               return nullptr;
            }

            bFeatureTypeFound = true;
            const char * pFeatureType = CHAR(featureTypeR);
            if(0 == strcmp("ordinal", pFeatureType)) {
               pFeature->featureType = FeatureTypeOrdinal;
            } else if(0 == strcmp("nominal", pFeatureType)) {
               pFeature->featureType = FeatureTypeNominal;
            } else {
               LOG_0(TraceLevelError, "ERROR ConvertFeatures unrecognized pFeatureType");
               return nullptr;
            }
         } else {
            LOG_0(TraceLevelError, "ERROR ConvertFeatures unrecognized pName");
            return nullptr;
         }
      }
      if(!bCountBinsFound) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatures !bCountBinsFound");
         return nullptr;
      }
      if(!bHasMissingFound) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatures !bHasMissingFound");
         return nullptr;
      }
      if(!bFeatureTypeFound) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatures !bFeatureTypeFound");
         return nullptr;
      }
      ++pFeature;
   }
   return aFeatures;
}

EbmNativeFeatureGroup * ConvertFeatureGroups(const SEXP featureGroups, size_t * const pcFeatureGroups) {
   if(VECSXP != TYPEOF(featureGroups)) {
      LOG_0(TraceLevelError, "ERROR ConvertFeatureGroups VECSXP != TYPEOF(featureGroups)");
      return nullptr;
   }

   const R_xlen_t countFeatureGroupsR = xlength(featureGroups);
   if(!IsNumberConvertable<size_t, R_xlen_t>(countFeatureGroupsR)) {
      LOG_0(TraceLevelError, "ERROR ConvertFeatureGroups !IsNumberConvertable<size_t, R_xlen_t>(countFeatureGroupsR)");
      return nullptr;
   }
   const size_t cFeatureGroups = static_cast<size_t>(countFeatureGroupsR);
   if(!IsNumberConvertable<IntEbmType, size_t>(cFeatureGroups)) {
      LOG_0(TraceLevelError, "ERROR ConvertFeatureGroups !IsNumberConvertable<IntEbmType, size_t>(cFeatureGroups)");
      return nullptr;
   }
   *pcFeatureGroups = cFeatureGroups;

   EbmNativeFeatureGroup * const aFeatureGroups = reinterpret_cast<EbmNativeFeatureGroup *>(
      R_alloc(cFeatureGroups, static_cast<int>(sizeof(EbmNativeFeatureGroup))));
   // R_alloc doesn't return nullptr, so we don't need to check aFeatureGroups
   EbmNativeFeatureGroup * pFeatureGroup = aFeatureGroups;
   for(size_t iFeatureGroup = 0; iFeatureGroup < cFeatureGroups; ++iFeatureGroup) {
      const SEXP oneFeatureGroup = VECTOR_ELT(featureGroups, iFeatureGroup);
      EBM_ASSERT(nullptr != oneFeatureGroup);
      if(VECSXP != TYPEOF(oneFeatureGroup)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureGroups VECSXP != TYPEOF(oneFeatureGroup)");
         return nullptr;
      }

      constexpr size_t cItems = 1;
      if(R_xlen_t { cItems } != xlength(oneFeatureGroup)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureGroups R_xlen_t { cItems } != xlength(oneFeatureGroup)");
         return nullptr;
      }
      const SEXP fieldNames = getAttrib(oneFeatureGroup, R_NamesSymbol);
      EBM_ASSERT(nullptr != fieldNames);
      if(STRSXP != TYPEOF(fieldNames)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureGroups STRSXP != TYPEOF(fieldNames)");
         return nullptr;
      }
      if(R_xlen_t { cItems } != xlength(fieldNames)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureGroups R_xlen_t { cItems } != xlength(fieldNames)");
         return nullptr;
      }

      const SEXP nameR = STRING_ELT(fieldNames, 0);
      if(CHARSXP != TYPEOF(nameR)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureGroups CHARSXP != TYPEOF(nameR)");
         return nullptr;
      }
      const char * pName = CHAR(nameR);
      if(0 != strcmp("n_features", pName)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureGroups 0 != strcmp(\"n_features\", pName");
         return nullptr;
      }

      SEXP val = VECTOR_ELT(oneFeatureGroup, 0);
      if(REALSXP != TYPEOF(val)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureGroups REALSXP != TYPEOF(value)");
         return nullptr;
      }
      if(1 != xlength(val)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureGroups 1 != xlength(val)");
         return nullptr;
      }

      double countFeaturesInGroupDouble = REAL(val)[0];
      if(!IsDoubleToIntEbmTypeIndexValid(countFeaturesInGroupDouble)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureGroups !IsDoubleToIntEbmTypeIndexValid(countFeaturesInGroupDouble)");
         return nullptr;
      }
      pFeatureGroup->countFeaturesInGroup = static_cast<IntEbmType>(countFeaturesInGroupDouble);

      ++pFeatureGroup;
   }
   return aFeatureGroups;
}

size_t CountFeatureGroupsIndexes(const size_t cFeatureGroups, const EbmNativeFeatureGroup * const aFeatureGroups) {
   size_t cFeatureGroupsIndexes = 0;
   if(0 != cFeatureGroups) {
      const EbmNativeFeatureGroup * pFeatureGroup = aFeatureGroups;
      const EbmNativeFeatureGroup * const pFeatureGroupEnd = aFeatureGroups + cFeatureGroups;
      do {
         const IntEbmType countFeaturesInGroup = pFeatureGroup->countFeaturesInGroup;
         if(!IsNumberConvertable<size_t, IntEbmType>(countFeaturesInGroup)) {
            LOG_0(TraceLevelError, "ERROR CountFeatureGroupsIndexes !IsNumberConvertable<size_t, IntEbmType>(countFeaturesInGroup)");
            return SIZE_MAX;
         }
         const size_t cFeaturesInGroup = static_cast<size_t>(countFeaturesInGroup);
         if(IsAddError(cFeatureGroupsIndexes, cFeaturesInGroup)) {
            LOG_0(TraceLevelError, "ERROR CountFeatureGroupsIndexes IsAddError(cFeatureGroupsIndexes, cFeaturesInGroup)");
            return SIZE_MAX;
         }
         cFeatureGroupsIndexes += cFeaturesInGroup;
         ++pFeatureGroup;
      } while(pFeatureGroupEnd != pFeatureGroup);
   }
   return cFeatureGroupsIndexes;
}

bool ConvertDoublesToIndexes(const SEXP items, size_t * const pcItems, const IntEbmType * * const pRet) {
   EBM_ASSERT(nullptr != items);
   EBM_ASSERT(nullptr != pcItems);
   if(REALSXP != TYPEOF(items)) {
      LOG_0(TraceLevelError, "ERROR ConvertDoublesToIndexes REALSXP != TYPEOF(items)");
      return true;
   }
   const R_xlen_t countItemsR = xlength(items);
   if(!IsNumberConvertable<size_t, R_xlen_t>(countItemsR)) {
      LOG_0(TraceLevelError, "ERROR ConvertDoublesToIndexes !IsNumberConvertable<size_t, R_xlen_t>(countItemsR)");
      return true;
   }
   const size_t cItems = static_cast<size_t>(countItemsR);
   if(!IsNumberConvertable<IntEbmType, size_t>(cItems)) {
      LOG_0(TraceLevelError, "ERROR ConvertDoublesToIndexes !IsNumberConvertable<IntEbmType, size_t>(cItems)");
      return true;
   }
   *pcItems = cItems;

   IntEbmType * aItems = nullptr;
   if(0 != cItems) {
      aItems = reinterpret_cast<IntEbmType *>(R_alloc(cItems, static_cast<int>(sizeof(IntEbmType))));
      // R_alloc doesn't return nullptr, so we don't need to check aItems
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
         ++pItem;
         ++pOriginal;
      } while(pItemEnd != pItem);
   }
   *pRet = aItems;
   return false;
}

bool ConvertDoublesToDoubles(const SEXP items, size_t * const pcItems, const FloatEbmType * * const pRet) {
   EBM_ASSERT(nullptr != items);
   EBM_ASSERT(nullptr != pcItems);
   if(REALSXP != TYPEOF(items)) {
      LOG_0(TraceLevelError, "ERROR ConvertDoublesToDoubles REALSXP != TYPEOF(items)");
      return true;
   }
   const R_xlen_t countItemsR = xlength(items);
   if(!IsNumberConvertable<size_t, R_xlen_t>(countItemsR)) {
      LOG_0(TraceLevelError, "ERROR ConvertDoublesToDoubles !IsNumberConvertable<size_t, R_xlen_t>(countItemsR)");
      return true;
   }
   const size_t cItems = static_cast<size_t>(countItemsR);
   if(!IsNumberConvertable<IntEbmType, size_t>(cItems)) {
      LOG_0(TraceLevelError, "ERROR ConvertDoublesToDoubles !IsNumberConvertable<IntEbmType, size_t>(cItems)");
      return true;
   }
   *pcItems = cItems;

   FloatEbmType * aItems = nullptr;
   if(0 != cItems) {
      aItems = reinterpret_cast<FloatEbmType *>(R_alloc(cItems, static_cast<int>(sizeof(FloatEbmType))));
      // R_alloc doesn't return nullptr, so we don't need to check aItems
      FloatEbmType * pItem = aItems;
      const FloatEbmType * const pItemEnd = aItems + cItems;
      const double * pOriginal = REAL(items);
      do {
         const double val = *pOriginal;
         *pItem = static_cast<FloatEbmType>(val);
         ++pItem;
         ++pOriginal;
      } while(pItemEnd != pItem);
   }
   *pRet = aItems;
   return false;
}

SEXP InitializeBoostingClassification_R(
   SEXP countTargetClasses,
   SEXP features,
   SEXP featureGroups,
   SEXP featureGroupIndexes,
   SEXP trainingBinnedData,
   SEXP trainingTargets,
   SEXP trainingPredictorScores,
   SEXP validationBinnedData,
   SEXP validationTargets,
   SEXP validationPredictorScores,
   SEXP countInnerBags,
   SEXP randomSeed
) {
   EBM_ASSERT(nullptr != countTargetClasses);
   EBM_ASSERT(nullptr != features);
   EBM_ASSERT(nullptr != featureGroups);
   EBM_ASSERT(nullptr != featureGroupIndexes);
   EBM_ASSERT(nullptr != trainingBinnedData);
   EBM_ASSERT(nullptr != trainingTargets);
   EBM_ASSERT(nullptr != trainingPredictorScores);
   EBM_ASSERT(nullptr != validationBinnedData);
   EBM_ASSERT(nullptr != validationTargets);
   EBM_ASSERT(nullptr != validationPredictorScores);
   EBM_ASSERT(nullptr != countInnerBags);
   EBM_ASSERT(nullptr != randomSeed);

   if(!IsSingleDoubleVector(countTargetClasses)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R !IsSingleDoubleVector(countTargetClasses)");
      return R_NilValue;
   }
   double countTargetClassesDouble = REAL(countTargetClasses)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(countTargetClassesDouble)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R !IsDoubleToIntEbmTypeIndexValid(countTargetClassesDouble)");
      return R_NilValue;
   }
   const size_t cTargetClasses = static_cast<size_t>(countTargetClassesDouble);
   if(!IsNumberConvertable<ptrdiff_t, size_t>(cTargetClasses)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R !IsNumberConvertable<ptrdiff_t, size_t>(cTargetClasses)");
      return R_NilValue;
   }
   const size_t cVectorLength = GetVectorLengthFlat(static_cast<ptrdiff_t>(cTargetClasses));

   size_t cFeatures;
   EbmNativeFeature * const aFeatures = ConvertFeatures(features, &cFeatures);
   if(nullptr == aFeatures) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countFeatures = static_cast<IntEbmType>(cFeatures); // the validity of this conversion was checked in ConvertFeatures(...)

   size_t cFeatureGroups;
   EbmNativeFeatureGroup * const aFeatureGroups = ConvertFeatureGroups(featureGroups, &cFeatureGroups);
   if(nullptr == aFeatureGroups) {
      // we've already logged any errors
      return R_NilValue;
   }
   // the validity of this conversion was checked in ConvertFeatureGroups(...)
   const IntEbmType countFeatureGroups = static_cast<IntEbmType>(cFeatureGroups);

   const size_t cFeatureGroupsIndexesCheck = CountFeatureGroupsIndexes(cFeatureGroups, aFeatureGroups);
   if(SIZE_MAX == cFeatureGroupsIndexesCheck) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cFeatureGroupsIndexesActual;
   const IntEbmType * aFeatureGroupIndexes;
   if(ConvertDoublesToIndexes(featureGroupIndexes, &cFeatureGroupsIndexesActual, &aFeatureGroupIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cFeatureGroupsIndexesActual != cFeatureGroupsIndexesCheck) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R cFeatureGroupsIndexesActual != cFeatureGroupsIndexesCheck");
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
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R IsMultiplyError(cTrainingSamples, cFeatures)");
      return R_NilValue;
   }
   if(cTrainingSamples * cFeatures != cTrainingBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R cTrainingSamples * cFeatures != cTrainingBinnedData");
      return R_NilValue;
   }

   const FloatEbmType * aTrainingPredictorScores = nullptr;
   size_t cTrainingPredictorScores;
   if(ConvertDoublesToDoubles(trainingPredictorScores, &cTrainingPredictorScores, &aTrainingPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(IsMultiplyError(cTrainingSamples, cVectorLength)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R IsMultiplyError(cTrainingSamples, cVectorLength)");
      return R_NilValue;
   }
   if(cVectorLength * cTrainingSamples != cTrainingPredictorScores) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R cVectorLength * cTrainingSamples != cTrainingPredictorScores");
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
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R IsMultiplyError(cValidationSamples, cFeatures)");
      return R_NilValue;
   }
   if(cValidationSamples * cFeatures != cValidationBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R cValidationSamples * cFeatures != cValidationBinnedData");
      return R_NilValue;
   }

   const FloatEbmType * aValidationPredictorScores = nullptr;
   size_t cValidationPredictorScores;
   if(ConvertDoublesToDoubles(validationPredictorScores, &cValidationPredictorScores, &aValidationPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(IsMultiplyError(cValidationSamples, cVectorLength)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R IsMultiplyError(cValidationSamples, cVectorLength)");
      return R_NilValue;
   }
   if(cVectorLength * cValidationSamples != cValidationPredictorScores) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R cVectorLength * cValidationSamples != cValidationPredictorScores");
      return R_NilValue;
   }

   if(!IsSingleIntVector(countInnerBags)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R !IsSingleIntVector(countInnerBags)");
      return R_NilValue;
   }
   int countInnerBagsInt = INTEGER(countInnerBags)[0];
   if(!IsNumberConvertable<IntEbmType, int>(countInnerBagsInt)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R !IsNumberConvertable<IntEbmType, int>(countInnerBagsInt)");
      return nullptr;
   }
   IntEbmType countInnerBagsLocal = static_cast<IntEbmType>(countInnerBagsInt);

   if(!IsSingleIntVector(randomSeed)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R !IsSingleIntVector(randomSeed)");
      return R_NilValue;
   }
   // we don't care if the seed is clipped or doesn't fit, or whatever.  
   // Casting to unsigned avoids undefined behavior issues with casting between signed values.  
   const IntEbmType randomSeedLocal = static_cast<IntEbmType>(static_cast<unsigned int>(INTEGER(randomSeed)[0]));

   PEbmBoosting pEbmBoosting = InitializeBoostingClassification(
      static_cast<IntEbmType>(cTargetClasses), 
      countFeatures, 
      aFeatures, 
      countFeatureGroups, 
      aFeatureGroups, 
      aFeatureGroupIndexes, 
      countTrainingSamples, 
      aTrainingBinnedData, 
      aTrainingTargets, 
      aTrainingPredictorScores, 
      countValidationSamples, 
      aValidationBinnedData, 
      aValidationTargets, 
      aValidationPredictorScores, 
      countInnerBagsLocal, 
      randomSeedLocal
   );

   if(nullptr == pEbmBoosting) {
      return R_NilValue;
   } else {
      SEXP boostingRPointer = R_MakeExternalPtr(static_cast<void *>(pEbmBoosting), R_NilValue, R_NilValue); // makes an EXTPTRSXP
      PROTECT(boostingRPointer);

      R_RegisterCFinalizerEx(boostingRPointer, &BoostingFinalizer, Rboolean::TRUE);

      UNPROTECT(1);
      return boostingRPointer;
   }
}

SEXP InitializeBoostingRegression_R(
   SEXP features,
   SEXP featureGroups,
   SEXP featureGroupIndexes,
   SEXP trainingBinnedData,
   SEXP trainingTargets,
   SEXP trainingPredictorScores,
   SEXP validationBinnedData,
   SEXP validationTargets,
   SEXP validationPredictorScores,
   SEXP countInnerBags,
   SEXP randomSeed
) {
   EBM_ASSERT(nullptr != features);
   EBM_ASSERT(nullptr != featureGroups);
   EBM_ASSERT(nullptr != featureGroupIndexes);
   EBM_ASSERT(nullptr != trainingBinnedData);
   EBM_ASSERT(nullptr != trainingTargets);
   EBM_ASSERT(nullptr != trainingPredictorScores);
   EBM_ASSERT(nullptr != validationBinnedData);
   EBM_ASSERT(nullptr != validationTargets);
   EBM_ASSERT(nullptr != validationPredictorScores);
   EBM_ASSERT(nullptr != countInnerBags);
   EBM_ASSERT(nullptr != randomSeed);

   size_t cFeatures;
   EbmNativeFeature * const aFeatures = ConvertFeatures(features, &cFeatures);
   if(nullptr == aFeatures) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countFeatures = static_cast<IntEbmType>(cFeatures); // the validity of this conversion was checked in ConvertFeatures(...)

   size_t cFeatureGroups;
   EbmNativeFeatureGroup * const aFeatureGroups = ConvertFeatureGroups(featureGroups, &cFeatureGroups);
   if(nullptr == aFeatureGroups) {
      // we've already logged any errors
      return R_NilValue;
   }
   // the validity of this conversion was checked in ConvertFeatureGroups(...)
   const IntEbmType countFeatureGroups = static_cast<IntEbmType>(cFeatureGroups);

   const size_t cFeatureGroupsIndexesCheck = CountFeatureGroupsIndexes(cFeatureGroups, aFeatureGroups);
   if(SIZE_MAX == cFeatureGroupsIndexesCheck) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cFeatureGroupsIndexesActual;
   const IntEbmType * aFeatureGroupIndexes;
   if(ConvertDoublesToIndexes(featureGroupIndexes, &cFeatureGroupsIndexesActual, &aFeatureGroupIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cFeatureGroupsIndexesActual != cFeatureGroupsIndexesCheck) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R cFeatureGroupsIndexesActual != cFeatureGroupsIndexesCheck");
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
   if(ConvertDoublesToDoubles(trainingTargets, &cTrainingSamples, &aTrainingTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countTrainingSamples = static_cast<IntEbmType>(cTrainingSamples);

   if(IsMultiplyError(cTrainingSamples, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R IsMultiplyError(cTrainingSamples, cFeatures)");
      return R_NilValue;
   }
   if(cTrainingSamples * cFeatures != cTrainingBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R cTrainingSamples * cFeatures != cTrainingBinnedData");
      return R_NilValue;
   }

   const FloatEbmType * aTrainingPredictorScores = nullptr;
   size_t cTrainingPredictorScores;
   if(ConvertDoublesToDoubles(trainingPredictorScores, &cTrainingPredictorScores, &aTrainingPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cTrainingSamples != cTrainingPredictorScores) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R cTrainingSamples != cTrainingPredictorScores");
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
   if(ConvertDoublesToDoubles(validationTargets, &cValidationSamples, &aValidationTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countValidationSamples = static_cast<IntEbmType>(cValidationSamples);

   if(IsMultiplyError(cValidationSamples, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R IsMultiplyError(cValidationSamples, cFeatures)");
      return R_NilValue;
   }
   if(cValidationSamples * cFeatures != cValidationBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R cValidationSamples * cFeatures != cValidationBinnedData");
      return R_NilValue;
   }

   const FloatEbmType * aValidationPredictorScores = nullptr;
   size_t cValidationPredictorScores;
   if(ConvertDoublesToDoubles(validationPredictorScores, &cValidationPredictorScores, &aValidationPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cValidationSamples != cValidationPredictorScores) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R cValidationSamples != cValidationPredictorScores");
      return R_NilValue;
   }

   if(!IsSingleIntVector(countInnerBags)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R !IsSingleIntVector(countInnerBags)");
      return R_NilValue;
   }
   int countInnerBagsInt = INTEGER(countInnerBags)[0];
   if(!IsNumberConvertable<IntEbmType, int>(countInnerBagsInt)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R !IsNumberConvertable<IntEbmType, int>(countInnerBagsInt)");
      return nullptr;
   }
   IntEbmType countInnerBagsLocal = static_cast<IntEbmType>(countInnerBagsInt);

   if(!IsSingleIntVector(randomSeed)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R !IsSingleIntVector(randomSeed)");
      return R_NilValue;
   }
   // we don't care if the seed is clipped or doesn't fit, or whatever.  
   // Casting to unsigned avoids undefined behavior issues with casting between signed values.  
   const IntEbmType randomSeedLocal = static_cast<IntEbmType>(static_cast<unsigned int>(INTEGER(randomSeed)[0]));

   PEbmBoosting pEbmBoosting = InitializeBoostingRegression(
      countFeatures, 
      aFeatures, 
      countFeatureGroups, 
      aFeatureGroups, 
      aFeatureGroupIndexes, 
      countTrainingSamples, 
      aTrainingBinnedData, 
      aTrainingTargets, 
      aTrainingPredictorScores, 
      countValidationSamples, 
      aValidationBinnedData, 
      aValidationTargets, 
      aValidationPredictorScores, 
      countInnerBagsLocal, 
      randomSeedLocal
   );

   if(nullptr == pEbmBoosting) {
      return R_NilValue;
   } else {
      SEXP boostingRPointer = R_MakeExternalPtr(static_cast<void *>(pEbmBoosting), R_NilValue, R_NilValue); // makes an EXTPTRSXP
      PROTECT(boostingRPointer);

      R_RegisterCFinalizerEx(boostingRPointer, &BoostingFinalizer, Rboolean::TRUE);

      UNPROTECT(1);
      return boostingRPointer;
   }
}

SEXP BoostingStep_R(
   SEXP ebmBoosting,
   SEXP indexFeatureGroup,
   SEXP learningRate,
   SEXP countTreeSplitsMax,
   SEXP countSamplesRequiredForChildSplitMin,
   SEXP trainingWeights,
   SEXP validationWeights
) {
   EBM_ASSERT(nullptr != ebmBoosting);
   EBM_ASSERT(nullptr != indexFeatureGroup);
   EBM_ASSERT(nullptr != learningRate);
   EBM_ASSERT(nullptr != countTreeSplitsMax);
   EBM_ASSERT(nullptr != countSamplesRequiredForChildSplitMin);
   EBM_ASSERT(nullptr != trainingWeights);
   EBM_ASSERT(nullptr != validationWeights);

   if(EXTPTRSXP != TYPEOF(ebmBoosting)) {
      LOG_0(TraceLevelError, "ERROR BoostingStep_R EXTPTRSXP != TYPEOF(ebmBoosting)");
      return R_NilValue;
   }
   EbmBoostingState * pEbmBoosting = static_cast<EbmBoostingState *>(R_ExternalPtrAddr(ebmBoosting));
   if(nullptr == pEbmBoosting) {
      LOG_0(TraceLevelError, "ERROR BoostingStep_R nullptr == pEbmBoosting");
      return R_NilValue;
   }

   if(!IsSingleDoubleVector(indexFeatureGroup)) {
      LOG_0(TraceLevelError, "ERROR BoostingStep_R !IsSingleDoubleVector(indexFeatureGroup)");
      return R_NilValue;
   }
   double doubleIndex = REAL(indexFeatureGroup)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(doubleIndex)) {
      LOG_0(TraceLevelError, "ERROR BoostingStep_R !IsDoubleToIntEbmTypeIndexValid(doubleIndex)");
      return R_NilValue;
   }
   IntEbmType iFeatureGroup = static_cast<IntEbmType>(doubleIndex);

   if(!IsSingleDoubleVector(learningRate)) {
      LOG_0(TraceLevelError, "ERROR BoostingStep_R !IsSingleDoubleVector(learningRate)");
      return R_NilValue;
   }
   double learningRateLocal = REAL(learningRate)[0];

   if(!IsSingleDoubleVector(countTreeSplitsMax)) {
      LOG_0(TraceLevelError, "ERROR BoostingStep_R !IsSingleDoubleVector(countTreeSplitsMax)");
      return R_NilValue;
   }
   double doubleCountTreeSplitsMax = REAL(countTreeSplitsMax)[0];
   IntEbmType cTreeSplitsMax;
   static_assert(std::numeric_limits<double>::is_iec559, "we need is_iec559 to know that comparisons to infinity and -infinity to normal numbers work");
   if(std::isnan(doubleCountTreeSplitsMax) || static_cast<double>(std::numeric_limits<IntEbmType>::max()) < doubleCountTreeSplitsMax) {
      LOG_0(TraceLevelWarning, "WARNING BoostingStep_R countTreeSplitsMax overflow");
      cTreeSplitsMax = std::numeric_limits<IntEbmType>::max();
   } else if(doubleCountTreeSplitsMax < static_cast<double>(std::numeric_limits<IntEbmType>::lowest())) {
      LOG_0(TraceLevelWarning, "WARNING BoostingStep_R countTreeSplitsMax underflow");
      cTreeSplitsMax = std::numeric_limits<IntEbmType>::lowest();
   } else {
      cTreeSplitsMax = static_cast<IntEbmType>(doubleCountTreeSplitsMax);
   }

   if(!IsSingleDoubleVector(countSamplesRequiredForChildSplitMin)) {
      LOG_0(TraceLevelError, "ERROR BoostingStep_R !IsSingleDoubleVector(countSamplesRequiredForChildSplitMin)");
      return R_NilValue;
   }
   double doubleCountSamplesRequiredForChildSplitMin = REAL(countSamplesRequiredForChildSplitMin)[0];
   IntEbmType cSamplesRequiredForChildSplitMin;
   static_assert(std::numeric_limits<double>::is_iec559, "we need is_iec559 to know that comparisons to infinity and -infinity to normal numbers work");
   if(std::isnan(doubleCountSamplesRequiredForChildSplitMin) || 
      static_cast<double>(std::numeric_limits<IntEbmType>::max()) < doubleCountSamplesRequiredForChildSplitMin
   ) {
      LOG_0(TraceLevelWarning, "WARNING BoostingStep_R countSamplesRequiredForChildSplitMin overflow");
      cSamplesRequiredForChildSplitMin = std::numeric_limits<IntEbmType>::max();
   } else if(doubleCountSamplesRequiredForChildSplitMin < static_cast<double>(std::numeric_limits<IntEbmType>::lowest())) {
      LOG_0(TraceLevelWarning, "WARNING BoostingStep_R countSamplesRequiredForChildSplitMin underflow");
      cSamplesRequiredForChildSplitMin = std::numeric_limits<IntEbmType>::lowest();
   } else {
      cSamplesRequiredForChildSplitMin = static_cast<IntEbmType>(doubleCountSamplesRequiredForChildSplitMin);
   }

   double * pTrainingWeights = nullptr;
   double * pValidationWeights = nullptr;
   if(NILSXP != TYPEOF(trainingWeights) || NILSXP != TYPEOF(validationWeights)) {
      if(REALSXP != TYPEOF(trainingWeights)) {
         LOG_0(TraceLevelError, "ERROR BoostingStep_R REALSXP != TYPEOF(trainingWeights)");
         return R_NilValue;
      }
      R_xlen_t trainingWeightsLength = xlength(trainingWeights);
      if(!IsNumberConvertable<size_t, R_xlen_t>(trainingWeightsLength)) {
         LOG_0(TraceLevelError, "ERROR BoostingStep_R !IsNumberConvertable<size_t, R_xlen_t>(trainingWeightsLength)");
         return R_NilValue;
      }
      size_t cTrainingWeights = static_cast<size_t>(trainingWeightsLength);
      if(cTrainingWeights != pEbmBoosting->m_pTrainingSet->GetCountSamples()) {
         LOG_0(TraceLevelError, "ERROR BoostingStep_R cTrainingWeights != pEbmBoosting->m_pTrainingSet->GetCountSamples()");
         return R_NilValue;
      }
      pTrainingWeights = REAL(trainingWeights);

      if(REALSXP != TYPEOF(validationWeights)) {
         LOG_0(TraceLevelError, "ERROR BoostingStep_R REALSXP != TYPEOF(validationWeights)");
         return R_NilValue;
      }
      R_xlen_t validationWeightsLength = xlength(validationWeights);
      if(!IsNumberConvertable<size_t, R_xlen_t>(validationWeightsLength)) {
         LOG_0(TraceLevelError, "ERROR BoostingStep_R !IsNumberConvertable<size_t, R_xlen_t>(validationWeightsLength)");
         return R_NilValue;
      }
      size_t cValidationWeights = static_cast<size_t>(validationWeightsLength);
      if(cValidationWeights != pEbmBoosting->m_pValidationSet->GetCountSamples()) {
         LOG_0(TraceLevelError, "ERROR BoostingStep_R cValidationWeights != pEbmBoosting->m_pValidationSet->GetCountSamples()");
         return R_NilValue;
      }
      pValidationWeights = REAL(validationWeights);
   }

   FloatEbmType validationMetricReturn;
   if(0 != BoostingStep(
      reinterpret_cast<PEbmBoosting>(pEbmBoosting), 
      iFeatureGroup, 
      learningRateLocal, 
      cTreeSplitsMax, 
      cSamplesRequiredForChildSplitMin, 
      pTrainingWeights, 
      pValidationWeights, 
      &validationMetricReturn
   )) {
      LOG_0(TraceLevelWarning, "WARNING BoostingStep_R BoostingStep returned error code");
      return R_NilValue;
   }

   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = validationMetricReturn;
   UNPROTECT(1);
   return ret;
}

SEXP GetBestModelFeatureGroup_R(
   SEXP ebmBoosting,
   SEXP indexFeatureGroup
) {
   EBM_ASSERT(nullptr != ebmBoosting); // shouldn't be possible
   EBM_ASSERT(nullptr != indexFeatureGroup); // shouldn't be possible

   if(EXTPTRSXP != TYPEOF(ebmBoosting)) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup_R EXTPTRSXP != TYPEOF(ebmBoosting)");
      return R_NilValue;
   }
   EbmBoostingState * pEbmBoosting = static_cast<EbmBoostingState *>(R_ExternalPtrAddr(ebmBoosting));
   if(nullptr == pEbmBoosting) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup_R nullptr == pEbmBoosting");
      return R_NilValue;
   }

   if(!IsSingleDoubleVector(indexFeatureGroup)) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup_R !IsSingleDoubleVector(indexFeatureGroup)");
      return R_NilValue;
   }
   double doubleIndex = REAL(indexFeatureGroup)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(doubleIndex)) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup_R !IsDoubleToIntEbmTypeIndexValid(doubleIndex)");
      return R_NilValue;
   }
   IntEbmType iFeatureGroup = static_cast<IntEbmType>(doubleIndex);
   // we check that iFeatureGroup can be converted to size_t in IsDoubleToIntEbmTypeIndexValid
   if(pEbmBoosting->m_cFeatureGroups <= static_cast<size_t>(iFeatureGroup)) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureGroup_R pEbmBoosting->m_cFeatureGroups <= static_cast<size_t>(iFeatureGroup)");
      return R_NilValue;
   }

   FloatEbmType * pModelFeatureGroupTensor = GetBestModelFeatureGroup(reinterpret_cast<PEbmBoosting>(pEbmBoosting), iFeatureGroup);
   if(nullptr == pModelFeatureGroupTensor) {
      // if nullptr == pModelFeatureGroupTensor then either:
      //    1) m_cFeatureGroups was 0, in which case this function would have undefined behavior since the caller needs to indicate a valid 
      //       indexFeatureGroup, which is impossible, so we can do anything we like, include the below actions.
      //    2) m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0 (and the learning type is classification), 
      //       which is legal, which we need to handle here
      SEXP ret = allocVector(REALSXP, R_xlen_t { 0 });
      LOG_0(TraceLevelWarning, "WARNING GetBestModelFeatureGroup_R nullptr == pModelFeatureGroupTensor");
      return ret;
   }
   size_t cValues = GetVectorLengthFlat(pEbmBoosting->m_runtimeLearningTypeOrCountTargetClasses);
   const FeatureGroup * const pFeatureGroup = pEbmBoosting->m_apFeatureGroups[static_cast<size_t>(iFeatureGroup)];
   const size_t cFeatures = pFeatureGroup->m_cFeatures;
   if(0 != cFeatures) {
      const FeatureGroup::FeatureGroupEntry * pFeatureGroupEntry = &pFeatureGroup->m_FeatureGroupEntry[0];
      const FeatureGroup::FeatureGroupEntry * const pFeatureGroupEntryEnd = &pFeatureGroupEntry[cFeatures];
      do {
         const size_t cBins = pFeatureGroupEntry->m_pFeature->m_cBins;
         EBM_ASSERT(!IsMultiplyError(cBins, cValues)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
         cValues *= cBins;
         ++pFeatureGroupEntry;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
   }
   if(!IsNumberConvertable<R_xlen_t, size_t>(cValues)) {
      return R_NilValue;
   }
   SEXP ret = PROTECT(allocVector(REALSXP, static_cast<R_xlen_t>(cValues)));
   EBM_ASSERT(!IsMultiplyError(sizeof(double), cValues)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
   memcpy(REAL(ret), pModelFeatureGroupTensor, sizeof(double) * cValues);
   UNPROTECT(1);
   return ret;
}

SEXP GetCurrentModelFeatureGroup_R(
   SEXP ebmBoosting,
   SEXP indexFeatureGroup
) {
   EBM_ASSERT(nullptr != ebmBoosting); // shouldn't be possible
   EBM_ASSERT(nullptr != indexFeatureGroup); // shouldn't be possible

   if(EXTPTRSXP != TYPEOF(ebmBoosting)) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup_R EXTPTRSXP != TYPEOF(ebmBoosting)");
      return R_NilValue;
   }
   EbmBoostingState * pEbmBoosting = static_cast<EbmBoostingState *>(R_ExternalPtrAddr(ebmBoosting));
   if(nullptr == pEbmBoosting) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup_R nullptr == pEbmBoosting");
      return R_NilValue;
   }

   if(!IsSingleDoubleVector(indexFeatureGroup)) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup_R !IsSingleDoubleVector(indexFeatureGroup)");
      return R_NilValue;
   }
   double doubleIndex = REAL(indexFeatureGroup)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(doubleIndex)) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup_R !IsDoubleToIntEbmTypeIndexValid(doubleIndex)");
      return R_NilValue;
   }
   IntEbmType iFeatureGroup = static_cast<IntEbmType>(doubleIndex);
   // we check that iFeatureGroup can be converted to size_t in IsDoubleToIntEbmTypeIndexValid
   if(pEbmBoosting->m_cFeatureGroups <= static_cast<size_t>(iFeatureGroup)) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureGroup_R pEbmBoosting->m_cFeatureGroups <= static_cast<size_t>(iFeatureGroup)");
      return R_NilValue;
   }

   FloatEbmType * pModelFeatureGroupTensor = GetCurrentModelFeatureGroup(reinterpret_cast<PEbmBoosting>(pEbmBoosting), iFeatureGroup);
   if(nullptr == pModelFeatureGroupTensor) {
      // if nullptr == pModelFeatureGroupTensor then either:
      //    1) m_cFeatureGroups was 0, in which case this function would have undefined behavior since the caller needs to indicate a valid 
      //       indexFeatureGroup, which is impossible, so we can do anything we like, include the below actions.
      //    2) m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0 (and the learning type is classification), which is legal, 
      //       which we need to handle here
      SEXP ret = allocVector(REALSXP, R_xlen_t { 0 });
      LOG_0(TraceLevelWarning, "WARNING GetCurrentModelFeatureGroup_R nullptr == pModelFeatureGroupTensor");
      return ret;
   }
   size_t cValues = GetVectorLengthFlat(pEbmBoosting->m_runtimeLearningTypeOrCountTargetClasses);
   const FeatureGroup * const pFeatureGroup = pEbmBoosting->m_apFeatureGroups[static_cast<size_t>(iFeatureGroup)];
   const size_t cFeatures = pFeatureGroup->m_cFeatures;
   if(0 != cFeatures) {
      const FeatureGroup::FeatureGroupEntry * pFeatureGroupEntry = &pFeatureGroup->m_FeatureGroupEntry[0];
      const FeatureGroup::FeatureGroupEntry * const pFeatureGroupEntryEnd = &pFeatureGroupEntry[cFeatures];
      do {
         const size_t cBins = pFeatureGroupEntry->m_pFeature->m_cBins;
         EBM_ASSERT(!IsMultiplyError(cBins, cValues)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
         cValues *= cBins;
         ++pFeatureGroupEntry;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
   }
   if(!IsNumberConvertable<R_xlen_t, size_t>(cValues)) {
      return R_NilValue;
   }
   SEXP ret = PROTECT(allocVector(REALSXP, static_cast<R_xlen_t>(cValues)));
   EBM_ASSERT(!IsMultiplyError(sizeof(double), cValues)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
   memcpy(REAL(ret), pModelFeatureGroupTensor, sizeof(double) * cValues);
   UNPROTECT(1);
   return ret;
}

SEXP FreeBoosting_R(
   SEXP ebmBoosting
) {
   BoostingFinalizer(ebmBoosting);
   return R_NilValue;
}


SEXP InitializeInteractionClassification_R(
   SEXP countTargetClasses,
   SEXP features,
   SEXP binnedData,
   SEXP targets,
   SEXP predictorScores
) {
   EBM_ASSERT(nullptr != countTargetClasses);
   EBM_ASSERT(nullptr != features);
   EBM_ASSERT(nullptr != binnedData);
   EBM_ASSERT(nullptr != targets);
   EBM_ASSERT(nullptr != predictorScores);

   if(!IsSingleDoubleVector(countTargetClasses)) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification_R !IsSingleDoubleVector(countTargetClasses)");
      return R_NilValue;
   }
   double countTargetClassesDouble = REAL(countTargetClasses)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(countTargetClassesDouble)) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification_R !IsDoubleToIntEbmTypeIndexValid(countTargetClassesDouble)");
      return R_NilValue;
   }
   const size_t cTargetClasses = static_cast<size_t>(countTargetClassesDouble);
   if(!IsNumberConvertable<ptrdiff_t, size_t>(cTargetClasses)) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification_R !IsNumberConvertable<ptrdiff_t, size_t>(cTargetClasses)");
      return R_NilValue;
   }
   const size_t cVectorLength = GetVectorLengthFlat(static_cast<ptrdiff_t>(cTargetClasses));

   size_t cFeatures;
   EbmNativeFeature * const aFeatures = ConvertFeatures(features, &cFeatures);
   if(nullptr == aFeatures) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countFeatures = static_cast<IntEbmType>(cFeatures); // the validity of this conversion was checked in ConvertFeatures(...)

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
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification_R IsMultiplyError(cSamples, cFeatures)");
      return R_NilValue;
   }
   if(cSamples * cFeatures != cBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification_R cSamples * cFeatures != cBinnedData");
      return R_NilValue;
   }

   const FloatEbmType * aPredictorScores = nullptr;
   size_t cPredictorScores;
   if(ConvertDoublesToDoubles(predictorScores, &cPredictorScores, &aPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(IsMultiplyError(cSamples, cVectorLength)) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification_R IsMultiplyError(cSamples, cVectorLength)");
      return R_NilValue;
   }
   if(cVectorLength * cSamples != cPredictorScores) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification_R cVectorLength * cSamples != cPredictorScores");
      return R_NilValue;
   }

   PEbmInteraction pEbmInteraction = InitializeInteractionClassification(
      static_cast<IntEbmType>(cTargetClasses), 
      countFeatures, 
      aFeatures, 
      countSamples, 
      aBinnedData, 
      aTargets, 
      aPredictorScores
   );

   if(nullptr == pEbmInteraction) {
      return R_NilValue;
   } else {
      SEXP interactionRPointer = R_MakeExternalPtr(static_cast<void *>(pEbmInteraction), R_NilValue, R_NilValue); // makes an EXTPTRSXP
      PROTECT(interactionRPointer);

      R_RegisterCFinalizerEx(interactionRPointer, &InteractionFinalizer, Rboolean::TRUE);

      UNPROTECT(1);
      return interactionRPointer;
   }
}

SEXP InitializeInteractionRegression_R(
   SEXP features,
   SEXP binnedData,
   SEXP targets,
   SEXP predictorScores
) {
   EBM_ASSERT(nullptr != features);
   EBM_ASSERT(nullptr != binnedData);
   EBM_ASSERT(nullptr != targets);
   EBM_ASSERT(nullptr != predictorScores);

   size_t cFeatures;
   EbmNativeFeature * const aFeatures = ConvertFeatures(features, &cFeatures);
   if(nullptr == aFeatures) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countFeatures = static_cast<IntEbmType>(cFeatures); // the validity of this conversion was checked in ConvertFeatures(...)

   size_t cBinnedData;
   const IntEbmType * aBinnedData;
   if(ConvertDoublesToIndexes(binnedData, &cBinnedData, &aBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cSamples;
   const FloatEbmType * aTargets;
   if(ConvertDoublesToDoubles(targets, &cSamples, &aTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countSamples = static_cast<IntEbmType>(cSamples);

   if(IsMultiplyError(cSamples, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionRegression_R IsMultiplyError(cSamples, cFeatures)");
      return R_NilValue;
   }
   if(cSamples * cFeatures != cBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionRegression_R cSamples * cFeatures != cBinnedData");
      return R_NilValue;
   }

   const FloatEbmType * aPredictorScores = nullptr;
   size_t cPredictorScores;
   if(ConvertDoublesToDoubles(predictorScores, &cPredictorScores, &aPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cSamples != cPredictorScores) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionRegression_R cSamples != cPredictorScores");
      return R_NilValue;
   }

   PEbmInteraction pEbmInteraction = InitializeInteractionRegression(countFeatures, aFeatures, countSamples, aBinnedData, aTargets, aPredictorScores);

   if(nullptr == pEbmInteraction) {
      return R_NilValue;
   } else {
      SEXP interactionRPointer = R_MakeExternalPtr(static_cast<void *>(pEbmInteraction), R_NilValue, R_NilValue); // makes an EXTPTRSXP
      PROTECT(interactionRPointer);

      R_RegisterCFinalizerEx(interactionRPointer, &InteractionFinalizer, Rboolean::TRUE);

      UNPROTECT(1);
      return interactionRPointer;
   }
}

SEXP GetInteractionScore_R(
   SEXP ebmInteraction,
   SEXP featureIndexes
   SEXP countSamplesRequiredForChildSplitMin,
   ) {
   EBM_ASSERT(nullptr != ebmInteraction); // shouldn't be possible
   EBM_ASSERT(nullptr != featureIndexes); // shouldn't be possible
   EBM_ASSERT(nullptr != countSamplesRequiredForChildSplitMin);

   if(EXTPTRSXP != TYPEOF(ebmInteraction)) {
      LOG_0(TraceLevelError, "ERROR GetInteractionScore_R EXTPTRSXP != TYPEOF(ebmInteraction)");
      return R_NilValue;
   }
   EbmInteractionState * pEbmInteraction = static_cast<EbmInteractionState *>(R_ExternalPtrAddr(ebmInteraction));
   if(nullptr == pEbmInteraction) {
      LOG_0(TraceLevelError, "ERROR GetInteractionScore_R nullptr == pEbmInteraction");
      return R_NilValue;
   }

   size_t cFeaturesInGroup;
   const IntEbmType * aFeatureIndexes;
   if(ConvertDoublesToIndexes(featureIndexes, &cFeaturesInGroup, &aFeatureIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }
   IntEbmType countFeaturesInGroup = static_cast<IntEbmType>(cFeaturesInGroup);

   if(!IsSingleDoubleVector(countSamplesRequiredForChildSplitMin)) {
      LOG_0(TraceLevelError, "ERROR GetInteractionScore_R !IsSingleDoubleVector(countSamplesRequiredForChildSplitMin)");
      return R_NilValue;
   }
   double doubleCountSamplesRequiredForChildSplitMin = REAL(countSamplesRequiredForChildSplitMin)[0];
   IntEbmType cSamplesRequiredForChildSplitMin;
   static_assert(std::numeric_limits<double>::is_iec559, "we need is_iec559 to know that comparisons to infinity and -infinity to normal numbers work");
   if(std::isnan(doubleCountSamplesRequiredForChildSplitMin) ||
      static_cast<double>(std::numeric_limits<IntEbmType>::max()) < doubleCountSamplesRequiredForChildSplitMin
      ) {
      LOG_0(TraceLevelWarning, "WARNING GetInteractionScore_R countSamplesRequiredForChildSplitMin overflow");
      cSamplesRequiredForChildSplitMin = std::numeric_limits<IntEbmType>::max();
   } else if(doubleCountSamplesRequiredForChildSplitMin < static_cast<double>(std::numeric_limits<IntEbmType>::lowest())) {
      LOG_0(TraceLevelWarning, "WARNING GetInteractionScore_R countSamplesRequiredForChildSplitMin underflow");
      cSamplesRequiredForChildSplitMin = std::numeric_limits<IntEbmType>::lowest();
   } else {
      cSamplesRequiredForChildSplitMin = static_cast<IntEbmType>(doubleCountSamplesRequiredForChildSplitMin);
   }

   FloatEbmType interactionScoreReturn;
   if(0 != GetInteractionScore(reinterpret_cast<PEbmInteraction>(pEbmInteraction), countFeaturesInGroup, aFeatureIndexes, cSamplesRequiredForChildSplitMin, &interactionScoreReturn)) {
      LOG_0(TraceLevelWarning, "WARNING GetInteractionScore_R GetInteractionScore returned error code");
      return R_NilValue;
   }

   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = interactionScoreReturn;
   UNPROTECT(1);
   return ret;
}

SEXP FreeInteraction_R(
   SEXP ebmInteraction
) {
   InteractionFinalizer(ebmInteraction);
   return R_NilValue;
}

static const R_CallMethodDef g_exposedFunctions[] = {
   { "InitializeBoostingClassification_R", (DL_FUNC)&InitializeBoostingClassification_R, 12 },
   { "InitializeBoostingRegression_R", (DL_FUNC)& InitializeBoostingRegression_R, 11 },
   { "BoostingStep_R", (DL_FUNC)& BoostingStep_R, 7 },
   { "GetBestModelFeatureGroup_R", (DL_FUNC)&GetBestModelFeatureGroup_R, 2 },
   { "GetCurrentModelFeatureGroup_R", (DL_FUNC)& GetCurrentModelFeatureGroup_R, 2 },
   { "FreeBoosting_R", (DL_FUNC)& FreeBoosting_R, 1 },
   { "InitializeInteractionClassification_R", (DL_FUNC)&InitializeInteractionClassification_R, 5 },
   { "InitializeInteractionRegression_R", (DL_FUNC)& InitializeInteractionRegression_R, 4 },
   { "GetInteractionScore_R", (DL_FUNC)& GetInteractionScore_R, 3 },
   { "FreeInteraction_R", (DL_FUNC)& FreeInteraction_R, 1 },
   { NULL, NULL, 0 }
};

extern "C" {
   void attribute_visible R_init_interpret(DllInfo * info) {
      R_registerRoutines(info, NULL, g_exposedFunctions, NULL, NULL);
      R_useDynamicSymbols(info, FALSE);
      R_forceSymbols(info, TRUE);
   }
} // extern "C"
