// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include <cmath> // std::isnan, std::isinf

#include "ebm_native.h"
#include "EbmBoostingState.h"
#include "EbmInteractionState.h"

#include <Rinternals.h>
#include <R_ext/Visibility.h>

// TODO: switch logging to use the R logging infrastructure when invoked from R, BUT calling error or warning will generate longjumps, which bypass the regular return mechanisms.  We need to use R_tryCatch (which is older than R_UnwindProtect) to not leak memory that we allocate before calling the R error or warning functions

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
   if(std::min(static_cast<double>(std::numeric_limits<size_t>::max()), std::min(double { R_XLEN_T_MAX }, static_cast<double>(std::numeric_limits<IntEbmType>::max()))) < val) {
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

EbmNativeFeatureCombination * ConvertFeatureCombinations(const SEXP featureCombinations, size_t * const pcFeatureCombinations) {
   if(VECSXP != TYPEOF(featureCombinations)) {
      LOG_0(TraceLevelError, "ERROR ConvertFeatureCombinations VECSXP != TYPEOF(featureCombinations)");
      return nullptr;
   }

   const R_xlen_t countFeatureCombinationsR = xlength(featureCombinations);
   if(!IsNumberConvertable<size_t, R_xlen_t>(countFeatureCombinationsR)) {
      LOG_0(TraceLevelError, "ERROR ConvertFeatureCombinations !IsNumberConvertable<size_t, R_xlen_t>(countFeatureCombinationsR)");
      return nullptr;
   }
   const size_t cFeatureCombinations = static_cast<size_t>(countFeatureCombinationsR);
   if(!IsNumberConvertable<IntEbmType, size_t>(cFeatureCombinations)) {
      LOG_0(TraceLevelError, "ERROR ConvertFeatureCombinations !IsNumberConvertable<IntEbmType, size_t>(cFeatureCombinations)");
      return nullptr;
   }
   *pcFeatureCombinations = cFeatureCombinations;

   EbmNativeFeatureCombination * const aFeatureCombinations = reinterpret_cast<EbmNativeFeatureCombination *>(R_alloc(cFeatureCombinations, static_cast<int>(sizeof(EbmNativeFeatureCombination))));
   // R_alloc doesn't return nullptr, so we don't need to check aFeatureCombinations
   EbmNativeFeatureCombination * pFeatureCombination = aFeatureCombinations;
   for(size_t iFeatureCombination = 0; iFeatureCombination < cFeatureCombinations; ++iFeatureCombination) {
      const SEXP oneFeatureCombination = VECTOR_ELT(featureCombinations, iFeatureCombination);
      EBM_ASSERT(nullptr != oneFeatureCombination);
      if(VECSXP != TYPEOF(oneFeatureCombination)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureCombinations VECSXP != TYPEOF(oneFeatureCombination)");
         return nullptr;
      }

      constexpr size_t cItems = 1;
      if(R_xlen_t { cItems } != xlength(oneFeatureCombination)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureCombinations R_xlen_t { cItems } != xlength(oneFeatureCombination)");
         return nullptr;
      }
      const SEXP fieldNames = getAttrib(oneFeatureCombination, R_NamesSymbol);
      EBM_ASSERT(nullptr != fieldNames);
      if(STRSXP != TYPEOF(fieldNames)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureCombinations STRSXP != TYPEOF(fieldNames)");
         return nullptr;
      }
      if(R_xlen_t { cItems } != xlength(fieldNames)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureCombinations R_xlen_t { cItems } != xlength(fieldNames)");
         return nullptr;
      }

      const SEXP nameR = STRING_ELT(fieldNames, 0);
      if(CHARSXP != TYPEOF(nameR)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureCombinations CHARSXP != TYPEOF(nameR)");
         return nullptr;
      }
      const char * pName = CHAR(nameR);
      if(0 != strcmp("n_features", pName)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureCombinations 0 != strcmp(\"n_features\", pName");
         return nullptr;
      }

      SEXP val = VECTOR_ELT(oneFeatureCombination, 0);
      if(REALSXP != TYPEOF(val)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureCombinations REALSXP != TYPEOF(value)");
         return nullptr;
      }
      if(1 != xlength(val)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureCombinations 1 != xlength(val)");
         return nullptr;
      }

      double countFeaturesInCombinationDouble = REAL(val)[0];
      if(!IsDoubleToIntEbmTypeIndexValid(countFeaturesInCombinationDouble)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureCombinations !IsDoubleToIntEbmTypeIndexValid(countFeaturesInCombinationDouble)");
         return nullptr;
      }
      pFeatureCombination->countFeaturesInCombination = static_cast<IntEbmType>(countFeaturesInCombinationDouble);

      ++pFeatureCombination;
   }
   return aFeatureCombinations;
}

size_t CountFeatureCombinationsIndexes(const size_t cFeatureCombinations, const EbmNativeFeatureCombination * const aFeatureCombinations) {
   size_t cFeatureCombinationsIndexes = 0;
   if(0 != cFeatureCombinations) {
      const EbmNativeFeatureCombination * pFeatureCombination = aFeatureCombinations;
      const EbmNativeFeatureCombination * const pFeatureCombinationEnd = aFeatureCombinations + cFeatureCombinations;
      do {
         const IntEbmType countFeaturesInCombination = pFeatureCombination->countFeaturesInCombination;
         if(!IsNumberConvertable<size_t, IntEbmType>(countFeaturesInCombination)) {
            LOG_0(TraceLevelError, "ERROR CountFeatureCombinationsIndexes !IsNumberConvertable<size_t, IntEbmType>(countFeaturesInCombination)");
            return SIZE_MAX;
         }
         const size_t cFeaturesInCombination = static_cast<size_t>(countFeaturesInCombination);
         if(IsAddError(cFeatureCombinationsIndexes, cFeaturesInCombination)) {
            LOG_0(TraceLevelError, "ERROR CountFeatureCombinationsIndexes IsAddError(cFeatureCombinationsIndexes, cFeaturesInCombination)");
            return SIZE_MAX;
         }
         cFeatureCombinationsIndexes += cFeaturesInCombination;
         ++pFeatureCombination;
      } while(pFeatureCombinationEnd != pFeatureCombination);
   }
   return cFeatureCombinationsIndexes;
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
   SEXP featureCombinations,
   SEXP featureCombinationIndexes,
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
   EBM_ASSERT(nullptr != featureCombinations);
   EBM_ASSERT(nullptr != featureCombinationIndexes);
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

   size_t cFeatureCombinations;
   EbmNativeFeatureCombination * const aFeatureCombinations = ConvertFeatureCombinations(featureCombinations, &cFeatureCombinations);
   if(nullptr == aFeatureCombinations) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countFeatureCombinations = static_cast<IntEbmType>(cFeatureCombinations); // the validity of this conversion was checked in ConvertFeatureCombinations(...)

   const size_t cFeatureCombinationsIndexesCheck = CountFeatureCombinationsIndexes(cFeatureCombinations, aFeatureCombinations);
   if(SIZE_MAX == cFeatureCombinationsIndexesCheck) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cFeatureCombinationsIndexesActual;
   const IntEbmType * aFeatureCombinationIndexes;
   if(ConvertDoublesToIndexes(featureCombinationIndexes, &cFeatureCombinationsIndexesActual, &aFeatureCombinationIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cFeatureCombinationsIndexesActual != cFeatureCombinationsIndexesCheck) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R cFeatureCombinationsIndexesActual != cFeatureCombinationsIndexesCheck");
      return R_NilValue;
   }

   size_t cTrainingBinnedData;
   const IntEbmType * aTrainingBinnedData;
   if(ConvertDoublesToIndexes(trainingBinnedData, &cTrainingBinnedData, &aTrainingBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cTrainingInstances;
   const IntEbmType * aTrainingTargets;
   if(ConvertDoublesToIndexes(trainingTargets, &cTrainingInstances, &aTrainingTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countTrainingInstances = static_cast<IntEbmType>(cTrainingInstances);

   if(IsMultiplyError(cTrainingInstances, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R IsMultiplyError(cTrainingInstances, cFeatures)");
      return R_NilValue;
   }
   if(cTrainingInstances * cFeatures != cTrainingBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R cTrainingInstances * cFeatures != cTrainingBinnedData");
      return R_NilValue;
   }

   const FloatEbmType * aTrainingPredictorScores = nullptr;
   size_t cTrainingPredictorScores;
   if(ConvertDoublesToDoubles(trainingPredictorScores, &cTrainingPredictorScores, &aTrainingPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(IsMultiplyError(cTrainingInstances, cVectorLength)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R IsMultiplyError(cTrainingInstances, cVectorLength)");
      return R_NilValue;
   }
   if(cVectorLength * cTrainingInstances != cTrainingPredictorScores) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R cVectorLength * cTrainingInstances != cTrainingPredictorScores");
      return R_NilValue;
   }

   size_t cValidationBinnedData;
   const IntEbmType * aValidationBinnedData;
   if(ConvertDoublesToIndexes(validationBinnedData, &cValidationBinnedData, &aValidationBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cValidationInstances;
   const IntEbmType * aValidationTargets;
   if(ConvertDoublesToIndexes(validationTargets, &cValidationInstances, &aValidationTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countValidationInstances = static_cast<IntEbmType>(cValidationInstances);

   if(IsMultiplyError(cValidationInstances, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R IsMultiplyError(cValidationInstances, cFeatures)");
      return R_NilValue;
   }
   if(cValidationInstances * cFeatures != cValidationBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R cValidationInstances * cFeatures != cValidationBinnedData");
      return R_NilValue;
   }

   const FloatEbmType * aValidationPredictorScores = nullptr;
   size_t cValidationPredictorScores;
   if(ConvertDoublesToDoubles(validationPredictorScores, &cValidationPredictorScores, &aValidationPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(IsMultiplyError(cValidationInstances, cVectorLength)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R IsMultiplyError(cValidationInstances, cVectorLength)");
      return R_NilValue;
   }
   if(cVectorLength * cValidationInstances != cValidationPredictorScores) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R cVectorLength * cValidationInstances != cValidationPredictorScores");
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
   // we don't care if the seed is clipped or doesn't fit, or whatever.  Casting to unsigned avoids undefined behavior issues with casting between signed values.  
   const IntEbmType randomSeedLocal = static_cast<IntEbmType>(static_cast<unsigned int>(INTEGER(randomSeed)[0]));

   PEbmBoosting pEbmBoosting = InitializeBoostingClassification(static_cast<IntEbmType>(cTargetClasses), countFeatures, aFeatures, countFeatureCombinations, aFeatureCombinations, aFeatureCombinationIndexes, countTrainingInstances, aTrainingBinnedData, aTrainingTargets, aTrainingPredictorScores, countValidationInstances, aValidationBinnedData, aValidationTargets, aValidationPredictorScores, countInnerBagsLocal, randomSeedLocal);

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
   SEXP featureCombinations,
   SEXP featureCombinationIndexes,
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
   EBM_ASSERT(nullptr != featureCombinations);
   EBM_ASSERT(nullptr != featureCombinationIndexes);
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

   size_t cFeatureCombinations;
   EbmNativeFeatureCombination * const aFeatureCombinations = ConvertFeatureCombinations(featureCombinations, &cFeatureCombinations);
   if(nullptr == aFeatureCombinations) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countFeatureCombinations = static_cast<IntEbmType>(cFeatureCombinations); // the validity of this conversion was checked in ConvertFeatureCombinations(...)

   const size_t cFeatureCombinationsIndexesCheck = CountFeatureCombinationsIndexes(cFeatureCombinations, aFeatureCombinations);
   if(SIZE_MAX == cFeatureCombinationsIndexesCheck) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cFeatureCombinationsIndexesActual;
   const IntEbmType * aFeatureCombinationIndexes;
   if(ConvertDoublesToIndexes(featureCombinationIndexes, &cFeatureCombinationsIndexesActual, &aFeatureCombinationIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cFeatureCombinationsIndexesActual != cFeatureCombinationsIndexesCheck) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R cFeatureCombinationsIndexesActual != cFeatureCombinationsIndexesCheck");
      return R_NilValue;
   }

   size_t cTrainingBinnedData;
   const IntEbmType * aTrainingBinnedData;
   if(ConvertDoublesToIndexes(trainingBinnedData, &cTrainingBinnedData, &aTrainingBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cTrainingInstances;
   const FloatEbmType * aTrainingTargets;
   if(ConvertDoublesToDoubles(trainingTargets, &cTrainingInstances, &aTrainingTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countTrainingInstances = static_cast<IntEbmType>(cTrainingInstances);

   if(IsMultiplyError(cTrainingInstances, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R IsMultiplyError(cTrainingInstances, cFeatures)");
      return R_NilValue;
   }
   if(cTrainingInstances * cFeatures != cTrainingBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R cTrainingInstances * cFeatures != cTrainingBinnedData");
      return R_NilValue;
   }

   const FloatEbmType * aTrainingPredictorScores = nullptr;
   size_t cTrainingPredictorScores;
   if(ConvertDoublesToDoubles(trainingPredictorScores, &cTrainingPredictorScores, &aTrainingPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cTrainingInstances != cTrainingPredictorScores) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R cTrainingInstances != cTrainingPredictorScores");
      return R_NilValue;
   }

   size_t cValidationBinnedData;
   const IntEbmType * aValidationBinnedData;
   if(ConvertDoublesToIndexes(validationBinnedData, &cValidationBinnedData, &aValidationBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cValidationInstances;
   const FloatEbmType * aValidationTargets;
   if(ConvertDoublesToDoubles(validationTargets, &cValidationInstances, &aValidationTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countValidationInstances = static_cast<IntEbmType>(cValidationInstances);

   if(IsMultiplyError(cValidationInstances, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R IsMultiplyError(cValidationInstances, cFeatures)");
      return R_NilValue;
   }
   if(cValidationInstances * cFeatures != cValidationBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R cValidationInstances * cFeatures != cValidationBinnedData");
      return R_NilValue;
   }

   const FloatEbmType * aValidationPredictorScores = nullptr;
   size_t cValidationPredictorScores;
   if(ConvertDoublesToDoubles(validationPredictorScores, &cValidationPredictorScores, &aValidationPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cValidationInstances != cValidationPredictorScores) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R cValidationInstances != cValidationPredictorScores");
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
   // we don't care if the seed is clipped or doesn't fit, or whatever.  Casting to unsigned avoids undefined behavior issues with casting between signed values.  
   const IntEbmType randomSeedLocal = static_cast<IntEbmType>(static_cast<unsigned int>(INTEGER(randomSeed)[0]));

   PEbmBoosting pEbmBoosting = InitializeBoostingRegression(countFeatures, aFeatures, countFeatureCombinations, aFeatureCombinations, aFeatureCombinationIndexes, countTrainingInstances, aTrainingBinnedData, aTrainingTargets, aTrainingPredictorScores, countValidationInstances, aValidationBinnedData, aValidationTargets, aValidationPredictorScores, countInnerBagsLocal, randomSeedLocal);

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
   SEXP indexFeatureCombination,
   SEXP learningRate,
   SEXP countTreeSplitsMax,
   SEXP countInstancesRequiredForParentSplitMin,
   SEXP trainingWeights,
   SEXP validationWeights
) {
   EBM_ASSERT(nullptr != ebmBoosting);
   EBM_ASSERT(nullptr != indexFeatureCombination);
   EBM_ASSERT(nullptr != learningRate);
   EBM_ASSERT(nullptr != countTreeSplitsMax);
   EBM_ASSERT(nullptr != countInstancesRequiredForParentSplitMin);
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

   if(!IsSingleDoubleVector(indexFeatureCombination)) {
      LOG_0(TraceLevelError, "ERROR BoostingStep_R !IsSingleDoubleVector(indexFeatureCombination)");
      return R_NilValue;
   }
   double doubleIndex = REAL(indexFeatureCombination)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(doubleIndex)) {
      LOG_0(TraceLevelError, "ERROR BoostingStep_R !IsDoubleToIntEbmTypeIndexValid(doubleIndex)");
      return R_NilValue;
   }
   IntEbmType iFeatureCombination = static_cast<IntEbmType>(doubleIndex);

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

   if(!IsSingleDoubleVector(countInstancesRequiredForParentSplitMin)) {
      LOG_0(TraceLevelError, "ERROR BoostingStep_R !IsSingleDoubleVector(countInstancesRequiredForParentSplitMin)");
      return R_NilValue;
   }
   double doubleCountInstancesRequiredForParentSplitMin = REAL(countInstancesRequiredForParentSplitMin)[0];
   IntEbmType cInstancesRequiredForParentSplitMin;
   static_assert(std::numeric_limits<double>::is_iec559, "we need is_iec559 to know that comparisons to infinity and -infinity to normal numbers work");
   if(std::isnan(doubleCountInstancesRequiredForParentSplitMin) || static_cast<double>(std::numeric_limits<IntEbmType>::max()) < doubleCountInstancesRequiredForParentSplitMin) {
      LOG_0(TraceLevelWarning, "WARNING BoostingStep_R countInstancesRequiredForParentSplitMin overflow");
      cInstancesRequiredForParentSplitMin = std::numeric_limits<IntEbmType>::max();
   } else if(doubleCountInstancesRequiredForParentSplitMin < static_cast<double>(std::numeric_limits<IntEbmType>::lowest())) {
      LOG_0(TraceLevelWarning, "WARNING BoostingStep_R countInstancesRequiredForParentSplitMin underflow");
      cInstancesRequiredForParentSplitMin = std::numeric_limits<IntEbmType>::lowest();
   } else {
      cInstancesRequiredForParentSplitMin = static_cast<IntEbmType>(doubleCountInstancesRequiredForParentSplitMin);
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
      if(cTrainingWeights != pEbmBoosting->m_pTrainingSet->GetCountInstances()) {
         LOG_0(TraceLevelError, "ERROR BoostingStep_R cTrainingWeights != pEbmBoosting->m_pTrainingSet->GetCountInstances()");
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
      if(cValidationWeights != pEbmBoosting->m_pValidationSet->GetCountInstances()) {
         LOG_0(TraceLevelError, "ERROR BoostingStep_R cValidationWeights != pEbmBoosting->m_pValidationSet->GetCountInstances()");
         return R_NilValue;
      }
      pValidationWeights = REAL(validationWeights);
   }

   FloatEbmType validationMetricReturn;
   if(0 != BoostingStep(reinterpret_cast<PEbmBoosting>(pEbmBoosting), iFeatureCombination, learningRateLocal, cTreeSplitsMax, cInstancesRequiredForParentSplitMin, pTrainingWeights, pValidationWeights, &validationMetricReturn)) {
      LOG_0(TraceLevelWarning, "WARNING BoostingStep_R BoostingStep returned error code");
      return R_NilValue;
   }

   SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
   REAL(ret)[0] = validationMetricReturn;
   UNPROTECT(1);
   return ret;
}

SEXP GetBestModelFeatureCombination_R(
   SEXP ebmBoosting,
   SEXP indexFeatureCombination
) {
   EBM_ASSERT(nullptr != ebmBoosting); // shouldn't be possible
   EBM_ASSERT(nullptr != indexFeatureCombination); // shouldn't be possible

   if(EXTPTRSXP != TYPEOF(ebmBoosting)) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureCombination_R EXTPTRSXP != TYPEOF(ebmBoosting)");
      return R_NilValue;
   }
   EbmBoostingState * pEbmBoosting = static_cast<EbmBoostingState *>(R_ExternalPtrAddr(ebmBoosting));
   if(nullptr == pEbmBoosting) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureCombination_R nullptr == pEbmBoosting");
      return R_NilValue;
   }

   if(!IsSingleDoubleVector(indexFeatureCombination)) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureCombination_R !IsSingleDoubleVector(indexFeatureCombination)");
      return R_NilValue;
   }
   double doubleIndex = REAL(indexFeatureCombination)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(doubleIndex)) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureCombination_R !IsDoubleToIntEbmTypeIndexValid(doubleIndex)");
      return R_NilValue;
   }
   IntEbmType iFeatureCombination = static_cast<IntEbmType>(doubleIndex);
   // we check that iFeatureCombination can be converted to size_t in IsDoubleToIntEbmTypeIndexValid
   if(pEbmBoosting->m_cFeatureCombinations <= static_cast<size_t>(iFeatureCombination)) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureCombination_R pEbmBoosting->m_cFeatureCombinations <= static_cast<size_t>(iFeatureCombination)");
      return R_NilValue;
   }

   FloatEbmType * pModelFeatureCombinationTensor = GetBestModelFeatureCombination(reinterpret_cast<PEbmBoosting>(pEbmBoosting), iFeatureCombination);
   if(nullptr == pModelFeatureCombinationTensor) {
      // if nullptr == pModelFeatureCombinationTensor then either:
      //    1) m_cFeatureCombinations was 0, in which case this function would have undefined behavior since the caller needs to indicate a valid indexFeatureCombination, which is impossible, so we can do anything we like, include the below actions.
      //    2) m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0 (and the learning type is classification), which is legal, which we need to handle here
      SEXP ret = allocVector(REALSXP, R_xlen_t { 0 });
      LOG_0(TraceLevelWarning, "WARNING GetBestModelFeatureCombination_R nullptr == pModelFeatureCombinationTensor");
      return ret;
   }
   size_t cValues = GetVectorLengthFlat(pEbmBoosting->m_runtimeLearningTypeOrCountTargetClasses);
   const FeatureCombination * const pFeatureCombination = pEbmBoosting->m_apFeatureCombinations[static_cast<size_t>(iFeatureCombination)];
   const size_t cFeatures = pFeatureCombination->m_cFeatures;
   if(0 != cFeatures) {
      const FeatureCombination::FeatureCombinationEntry * pFeatureCombinationEntry = &pFeatureCombination->m_FeatureCombinationEntry[0];
      const FeatureCombination::FeatureCombinationEntry * const pFeatureCombinationEntryEnd = &pFeatureCombinationEntry[cFeatures];
      do {
         const size_t cBins = pFeatureCombinationEntry->m_pFeature->m_cBins;
         EBM_ASSERT(!IsMultiplyError(cBins, cValues)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
         cValues *= cBins;
         ++pFeatureCombinationEntry;
      } while(pFeatureCombinationEntryEnd != pFeatureCombinationEntry);
   }
   if(!IsNumberConvertable<R_xlen_t, size_t>(cValues)) {
      return R_NilValue;
   }
   SEXP ret = PROTECT(allocVector(REALSXP, static_cast<R_xlen_t>(cValues)));
   EBM_ASSERT(!IsMultiplyError(sizeof(double), cValues)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
   memcpy(REAL(ret), pModelFeatureCombinationTensor, sizeof(double) * cValues);
   UNPROTECT(1);
   return ret;
}

SEXP GetCurrentModelFeatureCombination_R(
   SEXP ebmBoosting,
   SEXP indexFeatureCombination
) {
   EBM_ASSERT(nullptr != ebmBoosting); // shouldn't be possible
   EBM_ASSERT(nullptr != indexFeatureCombination); // shouldn't be possible

   if(EXTPTRSXP != TYPEOF(ebmBoosting)) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureCombination_R EXTPTRSXP != TYPEOF(ebmBoosting)");
      return R_NilValue;
   }
   EbmBoostingState * pEbmBoosting = static_cast<EbmBoostingState *>(R_ExternalPtrAddr(ebmBoosting));
   if(nullptr == pEbmBoosting) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureCombination_R nullptr == pEbmBoosting");
      return R_NilValue;
   }

   if(!IsSingleDoubleVector(indexFeatureCombination)) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureCombination_R !IsSingleDoubleVector(indexFeatureCombination)");
      return R_NilValue;
   }
   double doubleIndex = REAL(indexFeatureCombination)[0];
   if(!IsDoubleToIntEbmTypeIndexValid(doubleIndex)) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureCombination_R !IsDoubleToIntEbmTypeIndexValid(doubleIndex)");
      return R_NilValue;
   }
   IntEbmType iFeatureCombination = static_cast<IntEbmType>(doubleIndex);
   // we check that iFeatureCombination can be converted to size_t in IsDoubleToIntEbmTypeIndexValid
   if(pEbmBoosting->m_cFeatureCombinations <= static_cast<size_t>(iFeatureCombination)) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureCombination_R pEbmBoosting->m_cFeatureCombinations <= static_cast<size_t>(iFeatureCombination)");
      return R_NilValue;
   }

   FloatEbmType * pModelFeatureCombinationTensor = GetCurrentModelFeatureCombination(reinterpret_cast<PEbmBoosting>(pEbmBoosting), iFeatureCombination);
   if(nullptr == pModelFeatureCombinationTensor) {
      // if nullptr == pModelFeatureCombinationTensor then either:
      //    1) m_cFeatureCombinations was 0, in which case this function would have undefined behavior since the caller needs to indicate a valid indexFeatureCombination, which is impossible, so we can do anything we like, include the below actions.
      //    2) m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0 (and the learning type is classification), which is legal, which we need to handle here
      SEXP ret = allocVector(REALSXP, R_xlen_t { 0 });
      LOG_0(TraceLevelWarning, "WARNING GetCurrentModelFeatureCombination_R nullptr == pModelFeatureCombinationTensor");
      return ret;
   }
   size_t cValues = GetVectorLengthFlat(pEbmBoosting->m_runtimeLearningTypeOrCountTargetClasses);
   const FeatureCombination * const pFeatureCombination = pEbmBoosting->m_apFeatureCombinations[static_cast<size_t>(iFeatureCombination)];
   const size_t cFeatures = pFeatureCombination->m_cFeatures;
   if(0 != cFeatures) {
      const FeatureCombination::FeatureCombinationEntry * pFeatureCombinationEntry = &pFeatureCombination->m_FeatureCombinationEntry[0];
      const FeatureCombination::FeatureCombinationEntry * const pFeatureCombinationEntryEnd = &pFeatureCombinationEntry[cFeatures];
      do {
         const size_t cBins = pFeatureCombinationEntry->m_pFeature->m_cBins;
         EBM_ASSERT(!IsMultiplyError(cBins, cValues)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
         cValues *= cBins;
         ++pFeatureCombinationEntry;
      } while(pFeatureCombinationEntryEnd != pFeatureCombinationEntry);
   }
   if(!IsNumberConvertable<R_xlen_t, size_t>(cValues)) {
      return R_NilValue;
   }
   SEXP ret = PROTECT(allocVector(REALSXP, static_cast<R_xlen_t>(cValues)));
   EBM_ASSERT(!IsMultiplyError(sizeof(double), cValues)); // we've allocated this memory, so it should be reachable, so these numbers should multiply
   memcpy(REAL(ret), pModelFeatureCombinationTensor, sizeof(double) * cValues);
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

   size_t cInstances;
   const IntEbmType * aTargets;
   if(ConvertDoublesToIndexes(targets, &cInstances, &aTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countInstances = static_cast<IntEbmType>(cInstances);

   if(IsMultiplyError(cInstances, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification_R IsMultiplyError(cInstances, cFeatures)");
      return R_NilValue;
   }
   if(cInstances * cFeatures != cBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification_R cInstances * cFeatures != cBinnedData");
      return R_NilValue;
   }

   const FloatEbmType * aPredictorScores = nullptr;
   size_t cPredictorScores;
   if(ConvertDoublesToDoubles(predictorScores, &cPredictorScores, &aPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(IsMultiplyError(cInstances, cVectorLength)) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification_R IsMultiplyError(cInstances, cVectorLength)");
      return R_NilValue;
   }
   if(cVectorLength * cInstances != cPredictorScores) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification_R cVectorLength * cInstances != cPredictorScores");
      return R_NilValue;
   }

   PEbmInteraction pEbmInteraction = InitializeInteractionClassification(static_cast<IntEbmType>(cTargetClasses), countFeatures, aFeatures, countInstances, aBinnedData, aTargets, aPredictorScores);

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

   size_t cInstances;
   const FloatEbmType * aTargets;
   if(ConvertDoublesToDoubles(targets, &cInstances, &aTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntEbmType countInstances = static_cast<IntEbmType>(cInstances);

   if(IsMultiplyError(cInstances, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionRegression_R IsMultiplyError(cInstances, cFeatures)");
      return R_NilValue;
   }
   if(cInstances * cFeatures != cBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionRegression_R cInstances * cFeatures != cBinnedData");
      return R_NilValue;
   }

   const FloatEbmType * aPredictorScores = nullptr;
   size_t cPredictorScores;
   if(ConvertDoublesToDoubles(predictorScores, &cPredictorScores, &aPredictorScores)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cInstances != cPredictorScores) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionRegression_R cInstances != cPredictorScores");
      return R_NilValue;
   }

   PEbmInteraction pEbmInteraction = InitializeInteractionRegression(countFeatures, aFeatures, countInstances, aBinnedData, aTargets, aPredictorScores);

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
) {
   EBM_ASSERT(nullptr != ebmInteraction); // shouldn't be possible
   EBM_ASSERT(nullptr != featureIndexes); // shouldn't be possible

   if(EXTPTRSXP != TYPEOF(ebmInteraction)) {
      LOG_0(TraceLevelError, "ERROR GetInteractionScore_R EXTPTRSXP != TYPEOF(ebmInteraction)");
      return R_NilValue;
   }
   EbmInteractionState * pEbmInteraction = static_cast<EbmInteractionState *>(R_ExternalPtrAddr(ebmInteraction));
   if(nullptr == pEbmInteraction) {
      LOG_0(TraceLevelError, "ERROR GetInteractionScore_R nullptr == pEbmInteraction");
      return R_NilValue;
   }

   size_t cFeaturesInCombination;
   const IntEbmType * aFeatureIndexes;
   if(ConvertDoublesToIndexes(featureIndexes, &cFeaturesInCombination, &aFeatureIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }
   IntEbmType countFeaturesInCombination = static_cast<IntEbmType>(cFeaturesInCombination);

   FloatEbmType interactionScoreReturn;
   if(0 != GetInteractionScore(reinterpret_cast<PEbmInteraction>(pEbmInteraction), countFeaturesInCombination, aFeatureIndexes, &interactionScoreReturn)) {
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
   { "GetBestModelFeatureCombination_R", (DL_FUNC)&GetBestModelFeatureCombination_R, 2 },
   { "GetCurrentModelFeatureCombination_R", (DL_FUNC)& GetCurrentModelFeatureCombination_R, 2 },
   { "FreeBoosting_R", (DL_FUNC)& FreeBoosting_R, 1 },
   { "InitializeInteractionClassification_R", (DL_FUNC)&InitializeInteractionClassification_R, 5 },
   { "InitializeInteractionRegression_R", (DL_FUNC)& InitializeInteractionRegression_R, 4 },
   { "GetInteractionScore_R", (DL_FUNC)& GetInteractionScore_R, 2 },
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
