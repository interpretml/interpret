// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include <cmath> // std::isnan, std::isinf

#include "ebmcore.h"
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

EBM_INLINE bool IsDoubleToIntegerDataTypeIndexValid(const double val) {
   if(std::isnan(val)) {
      return false;
   }
   static_assert(std::numeric_limits<double>::is_iec559, "we need is_iec559 to know that comparisons to infinity and -infinity to normal numbers work");
   if(val < double { 0 }) {
      return false;
   }
   if(std::min(static_cast<double>(std::numeric_limits<size_t>::max()), std::min(double { R_XLEN_T_MAX }, static_cast<double>(std::numeric_limits<IntegerDataType>::max()))) < val) {
      return false;
   }
   return true;
}

void TrainingFinalizer(SEXP trainingRPointer) {
   EBM_ASSERT(nullptr != trainingRPointer); // shouldn't be possible
   if(EXTPTRSXP == TYPEOF(trainingRPointer)) {
      PEbmBoosting pTraining = static_cast<PEbmBoosting>(R_ExternalPtrAddr(trainingRPointer));
      if(nullptr != pTraining) {
         FreeBoosting(pTraining);
         R_ClearExternalPtr(trainingRPointer);
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

EbmCoreFeature * ConvertFeatures(const SEXP features, size_t * const pcFeatures) {
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
   if(!IsNumberConvertable<IntegerDataType, size_t>(cFeatures)) {
      LOG_0(TraceLevelError, "ERROR ConvertFeatures !IsNumberConvertable<IntegerDataType, size_t>(cFeatures)");
      return nullptr;
   }
   *pcFeatures = cFeatures;

   EbmCoreFeature * const aFeatures = reinterpret_cast<EbmCoreFeature *>(R_alloc(cFeatures, static_cast<int>(sizeof(EbmCoreFeature))));
   // R_alloc doesn't return nullptr, so we don't need to check aFeatures
   EbmCoreFeature * pFeature = aFeatures;
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
         if(0 == strcmp("count_bins", pName)) {
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
            if(!IsDoubleToIntegerDataTypeIndexValid(countBinsDouble)) {
               LOG_0(TraceLevelError, "ERROR ConvertFeatures !IsDoubleToIntegerDataTypeIndexValid(countBinsDouble)");
               return nullptr;
            }
            pFeature->countBins = static_cast<IntegerDataType>(countBinsDouble);
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

EbmCoreFeatureCombination * ConvertFeatureCombinations(const SEXP featureCombinations, size_t * const pcFeatureCombinations) {
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
   if(!IsNumberConvertable<IntegerDataType, size_t>(cFeatureCombinations)) {
      LOG_0(TraceLevelError, "ERROR ConvertFeatureCombinations !IsNumberConvertable<IntegerDataType, size_t>(cFeatureCombinations)");
      return nullptr;
   }
   *pcFeatureCombinations = cFeatureCombinations;

   EbmCoreFeatureCombination * const aFeatureCombinations = reinterpret_cast<EbmCoreFeatureCombination *>(R_alloc(cFeatureCombinations, static_cast<int>(sizeof(EbmCoreFeatureCombination))));
   // R_alloc doesn't return nullptr, so we don't need to check aFeatureCombinations
   EbmCoreFeatureCombination * pFeatureCombination = aFeatureCombinations;
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
      if(0 != strcmp("count_features_in_combination", pName)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureCombinations 0 != strcmp(\"count_features_in_combination\", pName");
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
      if(!IsDoubleToIntegerDataTypeIndexValid(countFeaturesInCombinationDouble)) {
         LOG_0(TraceLevelError, "ERROR ConvertFeatureCombinations !IsDoubleToIntegerDataTypeIndexValid(countFeaturesInCombinationDouble)");
         return nullptr;
      }
      pFeatureCombination->countFeaturesInCombination = static_cast<IntegerDataType>(countFeaturesInCombinationDouble);

      ++pFeatureCombination;
   }
   return aFeatureCombinations;
}

size_t CountFeatureCombinationsIndexes(const size_t cFeatureCombinations, const EbmCoreFeatureCombination * const aFeatureCombinations) {
   size_t cFeatureCombinationsIndexes = 0;
   if(0 != cFeatureCombinations) {
      const EbmCoreFeatureCombination * pFeatureCombination = aFeatureCombinations;
      const EbmCoreFeatureCombination * const pFeatureCombinationEnd = aFeatureCombinations + cFeatureCombinations;
      do {
         const IntegerDataType countFeaturesInCombination = pFeatureCombination->countFeaturesInCombination;
         if(!IsNumberConvertable<size_t, IntegerDataType>(countFeaturesInCombination)) {
            LOG_0(TraceLevelError, "ERROR CountFeatureCombinationsIndexes !IsNumberConvertable<size_t, IntegerDataType>(countFeaturesInCombination)");
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

bool ConvertDoublesToIndexes(const SEXP items, size_t * const pcItems, const IntegerDataType * * const pRet) {
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
   if(!IsNumberConvertable<IntegerDataType, size_t>(cItems)) {
      LOG_0(TraceLevelError, "ERROR ConvertDoublesToIndexes !IsNumberConvertable<IntegerDataType, size_t>(cItems)");
      return true;
   }
   *pcItems = cItems;

   IntegerDataType * aItems = nullptr;
   if(0 != cItems) {
      aItems = reinterpret_cast<IntegerDataType *>(R_alloc(cItems, static_cast<int>(sizeof(IntegerDataType))));
      // R_alloc doesn't return nullptr, so we don't need to check aItems
      IntegerDataType * pItem = aItems;
      const IntegerDataType * const pItemEnd = aItems + cItems;
      const double * pOriginal = REAL(items);
      do {
         const double val = *pOriginal;
         if(!IsDoubleToIntegerDataTypeIndexValid(val)) {
            LOG_0(TraceLevelError, "ERROR ConvertDoublesToIndexes !IsDoubleToIntegerDataTypeIndexValid(val)");
            return true;
         }
         *pItem = static_cast<IntegerDataType>(val);
         ++pItem;
         ++pOriginal;
      } while(pItemEnd != pItem);
   }
   *pRet = aItems;
   return false;
}

bool ConvertDoublesToDoubles(const SEXP items, size_t * const pcItems, const FractionalDataType * * const pRet) {
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
   if(!IsNumberConvertable<IntegerDataType, size_t>(cItems)) {
      LOG_0(TraceLevelError, "ERROR ConvertDoublesToDoubles !IsNumberConvertable<IntegerDataType, size_t>(cItems)");
      return true;
   }
   *pcItems = cItems;

   FractionalDataType * aItems = nullptr;
   if(0 != cItems) {
      aItems = reinterpret_cast<FractionalDataType *>(R_alloc(cItems, static_cast<int>(sizeof(FractionalDataType))));
      // R_alloc doesn't return nullptr, so we don't need to check aItems
      FractionalDataType * pItem = aItems;
      const FractionalDataType * const pItemEnd = aItems + cItems;
      const double * pOriginal = REAL(items);
      do {
         const double val = *pOriginal;
         *pItem = static_cast<FractionalDataType>(val);
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
   if(!IsDoubleToIntegerDataTypeIndexValid(countTargetClassesDouble)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R !IsDoubleToIntegerDataTypeIndexValid(countTargetClassesDouble)");
      return R_NilValue;
   }
   const size_t cTargetClasses = static_cast<size_t>(countTargetClassesDouble);
   if(!IsNumberConvertable<ptrdiff_t, size_t>(cTargetClasses)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R !IsNumberConvertable<ptrdiff_t, size_t>(cTargetClasses)");
      return R_NilValue;
   }
   const size_t cVectorLength = GetVectorLengthFlatCore(static_cast<ptrdiff_t>(cTargetClasses));

   size_t cFeatures;
   EbmCoreFeature * const aFeatures = ConvertFeatures(features, &cFeatures);
   if(nullptr == aFeatures) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntegerDataType countFeatures = static_cast<IntegerDataType>(cFeatures); // the validity of this conversion was checked in ConvertFeatures(...)

   size_t cFeatureCombinations;
   EbmCoreFeatureCombination * const aFeatureCombinations = ConvertFeatureCombinations(featureCombinations, &cFeatureCombinations);
   if(nullptr == aFeatureCombinations) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntegerDataType countFeatureCombinations = static_cast<IntegerDataType>(cFeatureCombinations); // the validity of this conversion was checked in ConvertFeatureCombinations(...)

   const size_t cFeatureCombinationsIndexesCheck = CountFeatureCombinationsIndexes(cFeatureCombinations, aFeatureCombinations);
   if(SIZE_MAX == cFeatureCombinationsIndexesCheck) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cFeatureCombinationsIndexesActual;
   const IntegerDataType * aFeatureCombinationIndexes;
   if(ConvertDoublesToIndexes(featureCombinationIndexes, &cFeatureCombinationsIndexesActual, &aFeatureCombinationIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cFeatureCombinationsIndexesActual != cFeatureCombinationsIndexesCheck) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R cFeatureCombinationsIndexesActual != cFeatureCombinationsIndexesCheck");
      return R_NilValue;
   }

   size_t cTrainingBinnedData;
   const IntegerDataType * aTrainingBinnedData;
   if(ConvertDoublesToIndexes(trainingBinnedData, &cTrainingBinnedData, &aTrainingBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cTrainingInstances;
   const IntegerDataType * aTrainingTargets;
   if(ConvertDoublesToIndexes(trainingTargets, &cTrainingInstances, &aTrainingTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntegerDataType countTrainingInstances = static_cast<IntegerDataType>(cTrainingInstances);

   if(IsMultiplyError(cTrainingInstances, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R IsMultiplyError(cTrainingInstances, cFeatures)");
      return R_NilValue;
   }
   if(cTrainingInstances * cFeatures != cTrainingBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R cTrainingInstances * cFeatures != cTrainingBinnedData");
      return R_NilValue;
   }

   const FractionalDataType * aTrainingPredictorScores = nullptr;
   if(NILSXP != TYPEOF(trainingPredictorScores)) {
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
   }

   size_t cValidationBinnedData;
   const IntegerDataType * aValidationBinnedData;
   if(ConvertDoublesToIndexes(validationBinnedData, &cValidationBinnedData, &aValidationBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cValidationInstances;
   const IntegerDataType * aValidationTargets;
   if(ConvertDoublesToIndexes(validationTargets, &cValidationInstances, &aValidationTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntegerDataType countValidationInstances = static_cast<IntegerDataType>(cValidationInstances);

   if(IsMultiplyError(cValidationInstances, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R IsMultiplyError(cValidationInstances, cFeatures)");
      return R_NilValue;
   }
   if(cValidationInstances * cFeatures != cValidationBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R cValidationInstances * cFeatures != cValidationBinnedData");
      return R_NilValue;
   }

   const FractionalDataType * aValidationPredictorScores = nullptr;
   if(NILSXP != TYPEOF(validationPredictorScores)) {
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
   }

   if(!IsSingleIntVector(countInnerBags)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R !IsSingleIntVector(countInnerBags)");
      return R_NilValue;
   }
   int countInnerBagsInt = INTEGER(countInnerBags)[0];
   if(!IsNumberConvertable<IntegerDataType, int>(countInnerBagsInt)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R !IsNumberConvertable<IntegerDataType, int>(countInnerBagsInt)");
      return nullptr;
   }
   IntegerDataType countInnerBagsLocal = static_cast<IntegerDataType>(countInnerBagsInt);

   if(!IsSingleIntVector(randomSeed)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingClassification_R !IsSingleIntVector(randomSeed)");
      return R_NilValue;
   }
   // we don't care if the seed is clipped or doesn't fit, or whatever.  Casting to unsigned avoids undefined behavior issues with casting between signed values.  
   const IntegerDataType randomSeedLocal = static_cast<IntegerDataType>(static_cast<unsigned int>(INTEGER(randomSeed)[0]));

   PEbmBoosting pEbmBoosting = InitializeBoostingClassification(static_cast<IntegerDataType>(cTargetClasses), countFeatures, aFeatures, countFeatureCombinations, aFeatureCombinations, aFeatureCombinationIndexes, countTrainingInstances, aTrainingBinnedData, aTrainingTargets, aTrainingPredictorScores, countValidationInstances, aValidationBinnedData, aValidationTargets, aValidationPredictorScores, countInnerBagsLocal, randomSeedLocal);

   if(nullptr == pEbmBoosting) {
      return R_NilValue;
   } else {
      SEXP trainingRPointer = R_MakeExternalPtr(static_cast<void *>(pEbmBoosting), R_NilValue, R_NilValue); // makes an EXTPTRSXP
      PROTECT(trainingRPointer);

      R_RegisterCFinalizerEx(trainingRPointer, &TrainingFinalizer, Rboolean::TRUE);

      UNPROTECT(1);
      return trainingRPointer;
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
   EbmCoreFeature * const aFeatures = ConvertFeatures(features, &cFeatures);
   if(nullptr == aFeatures) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntegerDataType countFeatures = static_cast<IntegerDataType>(cFeatures); // the validity of this conversion was checked in ConvertFeatures(...)

   size_t cFeatureCombinations;
   EbmCoreFeatureCombination * const aFeatureCombinations = ConvertFeatureCombinations(featureCombinations, &cFeatureCombinations);
   if(nullptr == aFeatureCombinations) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntegerDataType countFeatureCombinations = static_cast<IntegerDataType>(cFeatureCombinations); // the validity of this conversion was checked in ConvertFeatureCombinations(...)

   const size_t cFeatureCombinationsIndexesCheck = CountFeatureCombinationsIndexes(cFeatureCombinations, aFeatureCombinations);
   if(SIZE_MAX == cFeatureCombinationsIndexesCheck) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cFeatureCombinationsIndexesActual;
   const IntegerDataType * aFeatureCombinationIndexes;
   if(ConvertDoublesToIndexes(featureCombinationIndexes, &cFeatureCombinationsIndexesActual, &aFeatureCombinationIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }
   if(cFeatureCombinationsIndexesActual != cFeatureCombinationsIndexesCheck) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R cFeatureCombinationsIndexesActual != cFeatureCombinationsIndexesCheck");
      return R_NilValue;
   }

   size_t cTrainingBinnedData;
   const IntegerDataType * aTrainingBinnedData;
   if(ConvertDoublesToIndexes(trainingBinnedData, &cTrainingBinnedData, &aTrainingBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cTrainingInstances;
   const FractionalDataType * aTrainingTargets;
   if(ConvertDoublesToDoubles(trainingTargets, &cTrainingInstances, &aTrainingTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntegerDataType countTrainingInstances = static_cast<IntegerDataType>(cTrainingInstances);

   if(IsMultiplyError(cTrainingInstances, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R IsMultiplyError(cTrainingInstances, cFeatures)");
      return R_NilValue;
   }
   if(cTrainingInstances * cFeatures != cTrainingBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R cTrainingInstances * cFeatures != cTrainingBinnedData");
      return R_NilValue;
   }

   const FractionalDataType * aTrainingPredictorScores = nullptr;
   if(NILSXP != TYPEOF(trainingPredictorScores)) {
      size_t cTrainingPredictorScores;
      if(ConvertDoublesToDoubles(trainingPredictorScores, &cTrainingPredictorScores, &aTrainingPredictorScores)) {
         // we've already logged any errors
         return R_NilValue;
      }
      if(cTrainingInstances != cTrainingPredictorScores) {
         LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R cTrainingInstances != cTrainingPredictorScores");
         return R_NilValue;
      }
   }

   size_t cValidationBinnedData;
   const IntegerDataType * aValidationBinnedData;
   if(ConvertDoublesToIndexes(validationBinnedData, &cValidationBinnedData, &aValidationBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cValidationInstances;
   const FractionalDataType * aValidationTargets;
   if(ConvertDoublesToDoubles(validationTargets, &cValidationInstances, &aValidationTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntegerDataType countValidationInstances = static_cast<IntegerDataType>(cValidationInstances);

   if(IsMultiplyError(cValidationInstances, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R IsMultiplyError(cValidationInstances, cFeatures)");
      return R_NilValue;
   }
   if(cValidationInstances * cFeatures != cValidationBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R cValidationInstances * cFeatures != cValidationBinnedData");
      return R_NilValue;
   }

   const FractionalDataType * aValidationPredictorScores = nullptr;
   if(NILSXP != TYPEOF(validationPredictorScores)) {
      size_t cValidationPredictorScores;
      if(ConvertDoublesToDoubles(validationPredictorScores, &cValidationPredictorScores, &aValidationPredictorScores)) {
         // we've already logged any errors
         return R_NilValue;
      }
      if(cValidationInstances != cValidationPredictorScores) {
         LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R cValidationInstances != cValidationPredictorScores");
         return R_NilValue;
      }
   }

   if(!IsSingleIntVector(countInnerBags)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R !IsSingleIntVector(countInnerBags)");
      return R_NilValue;
   }
   int countInnerBagsInt = INTEGER(countInnerBags)[0];
   if(!IsNumberConvertable<IntegerDataType, int>(countInnerBagsInt)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R !IsNumberConvertable<IntegerDataType, int>(countInnerBagsInt)");
      return nullptr;
   }
   IntegerDataType countInnerBagsLocal = static_cast<IntegerDataType>(countInnerBagsInt);

   if(!IsSingleIntVector(randomSeed)) {
      LOG_0(TraceLevelError, "ERROR InitializeBoostingRegression_R !IsSingleIntVector(randomSeed)");
      return R_NilValue;
   }
   // we don't care if the seed is clipped or doesn't fit, or whatever.  Casting to unsigned avoids undefined behavior issues with casting between signed values.  
   const IntegerDataType randomSeedLocal = static_cast<IntegerDataType>(static_cast<unsigned int>(INTEGER(randomSeed)[0]));

   PEbmBoosting pEbmBoosting = InitializeBoostingRegression(countFeatures, aFeatures, countFeatureCombinations, aFeatureCombinations, aFeatureCombinationIndexes, countTrainingInstances, aTrainingBinnedData, aTrainingTargets, aTrainingPredictorScores, countValidationInstances, aValidationBinnedData, aValidationTargets, aValidationPredictorScores, countInnerBagsLocal, randomSeedLocal);

   if(nullptr == pEbmBoosting) {
      return R_NilValue;
   } else {
      SEXP trainingRPointer = R_MakeExternalPtr(static_cast<void *>(pEbmBoosting), R_NilValue, R_NilValue); // makes an EXTPTRSXP
      PROTECT(trainingRPointer);

      R_RegisterCFinalizerEx(trainingRPointer, &TrainingFinalizer, Rboolean::TRUE);

      UNPROTECT(1);
      return trainingRPointer;
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
   if(!IsDoubleToIntegerDataTypeIndexValid(doubleIndex)) {
      LOG_0(TraceLevelError, "ERROR BoostingStep_R !IsDoubleToIntegerDataTypeIndexValid(doubleIndex)");
      return R_NilValue;
   }
   IntegerDataType iFeatureCombination = static_cast<IntegerDataType>(doubleIndex);

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
   IntegerDataType cTreeSplitsMax;
   static_assert(std::numeric_limits<double>::is_iec559, "we need is_iec559 to know that comparisons to infinity and -infinity to normal numbers work");
   if(std::isnan(doubleCountTreeSplitsMax) || static_cast<double>(std::numeric_limits<IntegerDataType>::max()) < doubleCountTreeSplitsMax) {
      LOG_0(TraceLevelWarning, "WARNING BoostingStep_R countTreeSplitsMax overflow");
      cTreeSplitsMax = std::numeric_limits<IntegerDataType>::max();
   } else if(doubleCountTreeSplitsMax < static_cast<double>(std::numeric_limits<IntegerDataType>::lowest())) {
      LOG_0(TraceLevelWarning, "WARNING BoostingStep_R countTreeSplitsMax underflow");
      cTreeSplitsMax = std::numeric_limits<IntegerDataType>::lowest();
   } else {
      cTreeSplitsMax = static_cast<IntegerDataType>(doubleCountTreeSplitsMax);
   }

   if(!IsSingleDoubleVector(countInstancesRequiredForParentSplitMin)) {
      LOG_0(TraceLevelError, "ERROR BoostingStep_R !IsSingleDoubleVector(countInstancesRequiredForParentSplitMin)");
      return R_NilValue;
   }
   double doubleCountInstancesRequiredForParentSplitMin = REAL(countInstancesRequiredForParentSplitMin)[0];
   IntegerDataType cInstancesRequiredForParentSplitMin;
   static_assert(std::numeric_limits<double>::is_iec559, "we need is_iec559 to know that comparisons to infinity and -infinity to normal numbers work");
   if(std::isnan(doubleCountInstancesRequiredForParentSplitMin) || static_cast<double>(std::numeric_limits<IntegerDataType>::max()) < doubleCountInstancesRequiredForParentSplitMin) {
      LOG_0(TraceLevelWarning, "WARNING BoostingStep_R countInstancesRequiredForParentSplitMin overflow");
      cInstancesRequiredForParentSplitMin = std::numeric_limits<IntegerDataType>::max();
   } else if(doubleCountInstancesRequiredForParentSplitMin < static_cast<double>(std::numeric_limits<IntegerDataType>::lowest())) {
      LOG_0(TraceLevelWarning, "WARNING BoostingStep_R countInstancesRequiredForParentSplitMin underflow");
      cInstancesRequiredForParentSplitMin = std::numeric_limits<IntegerDataType>::lowest();
   } else {
      cInstancesRequiredForParentSplitMin = static_cast<IntegerDataType>(doubleCountInstancesRequiredForParentSplitMin);
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

   FractionalDataType validationMetricReturn;
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
   if(!IsDoubleToIntegerDataTypeIndexValid(doubleIndex)) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureCombination_R !IsDoubleToIntegerDataTypeIndexValid(doubleIndex)");
      return R_NilValue;
   }
   IntegerDataType iFeatureCombination = static_cast<IntegerDataType>(doubleIndex);
   // we check that iFeatureCombination can be converted to size_t in IsDoubleToIntegerDataTypeIndexValid
   if(pEbmBoosting->m_cFeatureCombinations <= static_cast<size_t>(iFeatureCombination)) {
      LOG_0(TraceLevelError, "ERROR GetBestModelFeatureCombination_R pEbmBoosting->m_cFeatureCombinations <= static_cast<size_t>(iFeatureCombination)");
      return R_NilValue;
   }

   FractionalDataType * pModelFeatureCombinationTensor = GetBestModelFeatureCombination(reinterpret_cast<PEbmBoosting>(pEbmBoosting), iFeatureCombination);
   if(nullptr == pModelFeatureCombinationTensor) {
      // if nullptr == pModelFeatureCombinationTensor then either:
      //    1) m_cFeatureCombinations was 0, in which case this function would have undefined behavior since the caller needs to indicate a valid indexFeatureCombination, which is impossible, so we can do anything we like, include the below actions.
      //    2) m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0 (and the learning type is classification), which is legal, which we need to handle here
      SEXP ret = allocVector(REALSXP, R_xlen_t { 0 });
      LOG_0(TraceLevelWarning, "WARNING GetBestModelFeatureCombination_R nullptr == pModelFeatureCombinationTensor");
      return ret;
   }
   size_t cValues = GetVectorLengthFlatCore(pEbmBoosting->m_runtimeLearningTypeOrCountTargetClasses);
   const FeatureCombinationCore * const pFeatureCombinationCore = pEbmBoosting->m_apFeatureCombinations[static_cast<size_t>(iFeatureCombination)];
   const size_t cFeatures = pFeatureCombinationCore->m_cFeatures;
   if(0 != cFeatures) {
      const FeatureCombinationCore::FeatureCombinationEntry * pFeatureCombinationEntry = &pFeatureCombinationCore->m_FeatureCombinationEntry[0];
      const FeatureCombinationCore::FeatureCombinationEntry * const pFeatureCombinationEntryEnd = &pFeatureCombinationEntry[cFeatures];
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
   if(!IsDoubleToIntegerDataTypeIndexValid(doubleIndex)) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureCombination_R !IsDoubleToIntegerDataTypeIndexValid(doubleIndex)");
      return R_NilValue;
   }
   IntegerDataType iFeatureCombination = static_cast<IntegerDataType>(doubleIndex);
   // we check that iFeatureCombination can be converted to size_t in IsDoubleToIntegerDataTypeIndexValid
   if(pEbmBoosting->m_cFeatureCombinations <= static_cast<size_t>(iFeatureCombination)) {
      LOG_0(TraceLevelError, "ERROR GetCurrentModelFeatureCombination_R pEbmBoosting->m_cFeatureCombinations <= static_cast<size_t>(iFeatureCombination)");
      return R_NilValue;
   }

   FractionalDataType * pModelFeatureCombinationTensor = GetCurrentModelFeatureCombination(reinterpret_cast<PEbmBoosting>(pEbmBoosting), iFeatureCombination);
   if(nullptr == pModelFeatureCombinationTensor) {
      // if nullptr == pModelFeatureCombinationTensor then either:
      //    1) m_cFeatureCombinations was 0, in which case this function would have undefined behavior since the caller needs to indicate a valid indexFeatureCombination, which is impossible, so we can do anything we like, include the below actions.
      //    2) m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0 (and the learning type is classification), which is legal, which we need to handle here
      SEXP ret = allocVector(REALSXP, R_xlen_t { 0 });
      LOG_0(TraceLevelWarning, "WARNING GetCurrentModelFeatureCombination_R nullptr == pModelFeatureCombinationTensor");
      return ret;
   }
   size_t cValues = GetVectorLengthFlatCore(pEbmBoosting->m_runtimeLearningTypeOrCountTargetClasses);
   const FeatureCombinationCore * const pFeatureCombinationCore = pEbmBoosting->m_apFeatureCombinations[static_cast<size_t>(iFeatureCombination)];
   const size_t cFeatures = pFeatureCombinationCore->m_cFeatures;
   if(0 != cFeatures) {
      const FeatureCombinationCore::FeatureCombinationEntry * pFeatureCombinationEntry = &pFeatureCombinationCore->m_FeatureCombinationEntry[0];
      const FeatureCombinationCore::FeatureCombinationEntry * const pFeatureCombinationEntryEnd = &pFeatureCombinationEntry[cFeatures];
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
   TrainingFinalizer(ebmBoosting);
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
   if(!IsDoubleToIntegerDataTypeIndexValid(countTargetClassesDouble)) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification_R !IsDoubleToIntegerDataTypeIndexValid(countTargetClassesDouble)");
      return R_NilValue;
   }
   const size_t cTargetClasses = static_cast<size_t>(countTargetClassesDouble);
   if(!IsNumberConvertable<ptrdiff_t, size_t>(cTargetClasses)) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification_R !IsNumberConvertable<ptrdiff_t, size_t>(cTargetClasses)");
      return R_NilValue;
   }
   const size_t cVectorLength = GetVectorLengthFlatCore(static_cast<ptrdiff_t>(cTargetClasses));

   size_t cFeatures;
   EbmCoreFeature * const aFeatures = ConvertFeatures(features, &cFeatures);
   if(nullptr == aFeatures) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntegerDataType countFeatures = static_cast<IntegerDataType>(cFeatures); // the validity of this conversion was checked in ConvertFeatures(...)

   size_t cBinnedData;
   const IntegerDataType * aBinnedData;
   if(ConvertDoublesToIndexes(binnedData, &cBinnedData, &aBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cInstances;
   const IntegerDataType * aTargets;
   if(ConvertDoublesToIndexes(targets, &cInstances, &aTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntegerDataType countInstances = static_cast<IntegerDataType>(cInstances);

   if(IsMultiplyError(cInstances, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification_R IsMultiplyError(cInstances, cFeatures)");
      return R_NilValue;
   }
   if(cInstances * cFeatures != cBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionClassification_R cInstances * cFeatures != cBinnedData");
      return R_NilValue;
   }

   const FractionalDataType * aPredictorScores = nullptr;
   if(NILSXP != TYPEOF(predictorScores)) {
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
   }

   PEbmInteraction pEbmInteraction = InitializeInteractionClassification(static_cast<IntegerDataType>(cTargetClasses), countFeatures, aFeatures, countInstances, aBinnedData, aTargets, aPredictorScores);

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
   EbmCoreFeature * const aFeatures = ConvertFeatures(features, &cFeatures);
   if(nullptr == aFeatures) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntegerDataType countFeatures = static_cast<IntegerDataType>(cFeatures); // the validity of this conversion was checked in ConvertFeatures(...)

   size_t cBinnedData;
   const IntegerDataType * aBinnedData;
   if(ConvertDoublesToIndexes(binnedData, &cBinnedData, &aBinnedData)) {
      // we've already logged any errors
      return R_NilValue;
   }

   size_t cInstances;
   const FractionalDataType * aTargets;
   if(ConvertDoublesToDoubles(targets, &cInstances, &aTargets)) {
      // we've already logged any errors
      return R_NilValue;
   }
   const IntegerDataType countInstances = static_cast<IntegerDataType>(cInstances);

   if(IsMultiplyError(cInstances, cFeatures)) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionRegression_R IsMultiplyError(cInstances, cFeatures)");
      return R_NilValue;
   }
   if(cInstances * cFeatures != cBinnedData) {
      LOG_0(TraceLevelError, "ERROR InitializeInteractionRegression_R cInstances * cFeatures != cBinnedData");
      return R_NilValue;
   }

   const FractionalDataType * aPredictorScores = nullptr;
   if(NILSXP != TYPEOF(predictorScores)) {
      size_t cPredictorScores;
      if(ConvertDoublesToDoubles(predictorScores, &cPredictorScores, &aPredictorScores)) {
         // we've already logged any errors
         return R_NilValue;
      }
      if(cInstances != cPredictorScores) {
         LOG_0(TraceLevelError, "ERROR InitializeInteractionRegression_R cInstances != cPredictorScores");
         return R_NilValue;
      }
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
   const IntegerDataType * aFeatureIndexes;
   if(ConvertDoublesToIndexes(featureIndexes, &cFeaturesInCombination, &aFeatureIndexes)) {
      // we've already logged any errors
      return R_NilValue;
   }
   IntegerDataType countFeaturesInCombination = static_cast<IntegerDataType>(cFeaturesInCombination);

   FractionalDataType interactionScoreReturn;
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
