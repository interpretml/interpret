// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include <cmath> // std::isnan, std::isinf

#include <Rinternals.h>

#include "ebmcore.h"
#include "EbmTrainingState.h"
#include "EbmInteractionState.h"

// TODO: remove visibility to internal functions that don't need visibiliy -> https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Controlling-visibility
// TODO: Improve calling speed (see section 5.4.1 Speed considerations) https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Registering-native-routines
// TODO: switch logging to use the R logging infrastructure when invoked from R, BUT calling error or warning will generate longjumps, which bypass the regular return mechanisms.  We need to use R_tryCatch (which is older than R_UnwindProtect) to not leak memory that we allocate before calling the R error or warning functions
// todo: use our define in the compilation scrips in the larger program to determine if we're being compiled in R in our larger program (or USING_R, which might be set by R's compilation scripts and therefore be always defined at compilation time)

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
      PEbmTraining pTraining = static_cast<PEbmTraining>(R_ExternalPtrAddr(trainingRPointer));
      if(nullptr != pTraining) {
         FreeTraining(pTraining);
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
      LOG(TraceLevelError, "ERROR ConvertFeatures VECSXP != TYPEOF(features)");
      return nullptr;
   }
   const R_xlen_t countFeaturesR = xlength(features);
   if(!IsNumberConvertable<size_t, R_xlen_t>(countFeaturesR)) {
      LOG(TraceLevelError, "ERROR ConvertFeatures !IsNumberConvertable<size_t, R_xlen_t>(countFeaturesR)");
      return nullptr;
   }
   const size_t cFeatures = static_cast<size_t>(countFeaturesR);
   if(!IsNumberConvertable<IntegerDataType, size_t>(cFeatures)) {
      LOG(TraceLevelError, "ERROR ConvertFeatures !IsNumberConvertable<IntegerDataType, size_t>(cFeatures)");
      return nullptr;
   }
   *pcFeatures = cFeatures;

   EbmCoreFeature * const aFeatures = reinterpret_cast<EbmCoreFeature *>(R_alloc(cFeatures, static_cast<int>(sizeof(EbmCoreFeature))));
   EbmCoreFeature * pFeature = aFeatures;
   for(size_t iFeature = 0; iFeature < cFeatures; ++iFeature) {
      const SEXP oneFeature = VECTOR_ELT(features, iFeature);
      EBM_ASSERT(nullptr != oneFeature);
      if(VECSXP != TYPEOF(oneFeature)) {
         LOG(TraceLevelError, "ERROR ConvertFeatures VECSXP != TYPEOF(oneFeature)");
         return nullptr;
      }
      constexpr size_t cItems = 3;
      if(R_xlen_t { cItems } != xlength(oneFeature)) {
         LOG(TraceLevelError, "ERROR ConvertFeatures R_xlen_t { cItems } != xlength(oneFeature)");
         return nullptr;
      }
      const SEXP fieldNames = getAttrib(oneFeature, R_NamesSymbol);
      EBM_ASSERT(nullptr != fieldNames);
      if(STRSXP != TYPEOF(fieldNames)) {
         LOG(TraceLevelError, "ERROR ConvertFeatures STRSXP != TYPEOF(fieldNames)");
         return nullptr;
      }
      if(R_xlen_t { cItems } != xlength(fieldNames)) {
         LOG(TraceLevelError, "ERROR ConvertFeatures R_xlen_t { cItems } != xlength(fieldNames)");
         return nullptr;
      }

      bool bCountBinsFound = false;
      bool bHasMissingFound = false;
      bool bFeatureTypeFound = false;
      for(size_t iName = 0; iName < cItems; ++iName) {
         const SEXP nameR = STRING_ELT(fieldNames, iName);
         if(CHARSXP != TYPEOF(nameR)) {
            LOG(TraceLevelError, "ERROR ConvertFeatures CHARSXP != TYPEOF(nameR)");
            return nullptr;
         }
         const char * pName = CHAR(nameR);
         if(0 == strcmp("count_bins", pName)) {
            if(bCountBinsFound) {
               LOG(TraceLevelError, "ERROR ConvertFeatures bCountBinsFound");
               return nullptr;
            }

            SEXP val = VECTOR_ELT(oneFeature, iName);
            if(REALSXP != TYPEOF(val)) {
               LOG(TraceLevelError, "ERROR ConvertFeatures REALSXP != TYPEOF(value)");
               return nullptr;
            }
            if(1 != xlength(val)) {
               LOG(TraceLevelError, "ERROR ConvertFeatures 1 != xlength(val)");
               return nullptr;
            }

            double countBinsDouble = REAL(val)[0];
            if(!IsDoubleToIntegerDataTypeIndexValid(countBinsDouble)) {
               LOG(TraceLevelError, "ERROR ConvertFeatures !IsDoubleToIntegerDataTypeIndexValid(countBinsDouble)");
               return nullptr;
            }
            pFeature->countBins = static_cast<IntegerDataType>(countBinsDouble);
            bCountBinsFound = true;
         } else if(0 == strcmp("has_missing", pName)) {
            if(bHasMissingFound) {
               LOG(TraceLevelError, "ERROR ConvertFeatures bHasMissingFound");
               return nullptr;
            }

            SEXP val = VECTOR_ELT(oneFeature, iName);
            if(LGLSXP != TYPEOF(val)) {
               LOG(TraceLevelError, "ERROR ConvertFeatures LGLSXP != TYPEOF(value)");
               return nullptr;
            }
            if(1 != xlength(val)) {
               LOG(TraceLevelError, "ERROR ConvertFeatures 1 != xlength(val)");
               return nullptr;
            }

            int hasMissing = LOGICAL(val)[0];
            pFeature->hasMissing = 0 != hasMissing ? 1 : 0;
            bHasMissingFound = true;
         } else if(0 == strcmp("feature_type", pName)) {
            if(bFeatureTypeFound) {
               LOG(TraceLevelError, "ERROR ConvertFeatures bFeatureTypeFound");
               return nullptr;
            }

            SEXP val = VECTOR_ELT(oneFeature, iName);

            if(STRSXP != TYPEOF(val)) {
               LOG(TraceLevelError, "ERROR ConvertFeatures STRSXP != TYPEOF(value)");
               return nullptr;
            }
            if(1 != xlength(val)) {
               LOG(TraceLevelError, "ERROR ConvertFeatures 1 != xlength(val)");
               return nullptr;
            }

            const SEXP featureTypeR = STRING_ELT(val, 0);
            if(CHARSXP != TYPEOF(featureTypeR)) {
               LOG(TraceLevelError, "ERROR ConvertFeatures CHARSXP != TYPEOF(featureTypeR)");
               return nullptr;
            }

            bFeatureTypeFound = true;
            const char * pFeatureType = CHAR(featureTypeR);
            if(0 == strcmp("ordinal", pFeatureType)) {
               pFeature->featureType = FeatureTypeOrdinal;
            } else if(0 == strcmp("nominal", pFeatureType)) {
               pFeature->featureType = FeatureTypeNominal;
            } else {
               LOG(TraceLevelError, "ERROR ConvertFeatures unrecognized pFeatureType");
               return nullptr;
            }
         } else {
            LOG(TraceLevelError, "ERROR ConvertFeatures unrecognized pName");
            return nullptr;
         }
      }
      if(!bCountBinsFound) {
         LOG(TraceLevelError, "ERROR ConvertFeatures !bCountBinsFound");
         return nullptr;
      }
      if(!bHasMissingFound) {
         LOG(TraceLevelError, "ERROR ConvertFeatures !bHasMissingFound");
         return nullptr;
      }
      if(!bFeatureTypeFound) {
         LOG(TraceLevelError, "ERROR ConvertFeatures !bFeatureTypeFound");
         return nullptr;
      }
      ++pFeature;
   }
   return aFeatures;
}

EbmCoreFeatureCombination * ConvertFeatureCombinations(const SEXP featureCombinations, size_t * const pcFeatureCombinations) {
   if(VECSXP != TYPEOF(featureCombinations)) {
      LOG(TraceLevelError, "ERROR ConvertFeatureCombinations VECSXP != TYPEOF(featureCombinations)");
      return nullptr;
   }

   const R_xlen_t countFeatureCombinationsR = xlength(featureCombinations);
   if(!IsNumberConvertable<size_t, R_xlen_t>(countFeatureCombinationsR)) {
      LOG(TraceLevelError, "ERROR ConvertFeatureCombinations !IsNumberConvertable<size_t, R_xlen_t>(countFeatureCombinationsR)");
      return nullptr;
   }
   const size_t cFeatureCombinations = static_cast<size_t>(countFeatureCombinationsR);
   if(!IsNumberConvertable<IntegerDataType, size_t>(cFeatureCombinations)) {
      LOG(TraceLevelError, "ERROR ConvertFeatureCombinations !IsNumberConvertable<IntegerDataType, size_t>(cFeatureCombinations)");
      return nullptr;
   }
   *pcFeatureCombinations = cFeatureCombinations;

   EbmCoreFeatureCombination * const aFeatureCombinations = reinterpret_cast<EbmCoreFeatureCombination *>(R_alloc(cFeatureCombinations, static_cast<int>(sizeof(EbmCoreFeatureCombination))));
   EbmCoreFeatureCombination * pFeatureCombination = aFeatureCombinations;
   for(size_t iFeatureCombination = 0; iFeatureCombination < cFeatureCombinations; ++iFeatureCombination) {
      const SEXP oneFeatureCombination = VECTOR_ELT(featureCombinations, iFeatureCombination);
      EBM_ASSERT(nullptr != oneFeatureCombination);
      if(VECSXP != TYPEOF(oneFeatureCombination)) {
         LOG(TraceLevelError, "ERROR ConvertFeatureCombinations VECSXP != TYPEOF(oneFeatureCombination)");
         return nullptr;
      }

      constexpr size_t cItems = 1;
      if(R_xlen_t { cItems } != xlength(oneFeatureCombination)) {
         LOG(TraceLevelError, "ERROR ConvertFeatureCombinations R_xlen_t { cItems } != xlength(oneFeatureCombination)");
         return nullptr;
      }
      const SEXP fieldNames = getAttrib(oneFeatureCombination, R_NamesSymbol);
      EBM_ASSERT(nullptr != fieldNames);
      if(STRSXP != TYPEOF(fieldNames)) {
         LOG(TraceLevelError, "ERROR ConvertFeatureCombinations STRSXP != TYPEOF(fieldNames)");
         return nullptr;
      }
      if(R_xlen_t { cItems } != xlength(fieldNames)) {
         LOG(TraceLevelError, "ERROR ConvertFeatureCombinations R_xlen_t { cItems } != xlength(fieldNames)");
         return nullptr;
      }

      const SEXP nameR = STRING_ELT(fieldNames, 0);
      if(CHARSXP != TYPEOF(nameR)) {
         LOG(TraceLevelError, "ERROR ConvertFeatureCombinations CHARSXP != TYPEOF(nameR)");
         return nullptr;
      }
      const char * pName = CHAR(nameR);
      if(0 != strcmp("count_features_in_combination", pName)) {
         LOG(TraceLevelError, "ERROR ConvertFeatureCombinations 0 != strcmp(\"count_features_in_combination\", pName");
         return nullptr;
      }

      SEXP val = VECTOR_ELT(oneFeatureCombination, 0);
      if(REALSXP != TYPEOF(val)) {
         LOG(TraceLevelError, "ERROR ConvertFeatureCombinations REALSXP != TYPEOF(value)");
         return nullptr;
      }
      if(1 != xlength(val)) {
         LOG(TraceLevelError, "ERROR ConvertFeatureCombinations 1 != xlength(val)");
         return nullptr;
      }

      double countFeaturesInCombinationDouble = REAL(val)[0];
      if(!IsDoubleToIntegerDataTypeIndexValid(countFeaturesInCombinationDouble)) {
         LOG(TraceLevelError, "ERROR ConvertFeatureCombinations !IsDoubleToIntegerDataTypeIndexValid(countFeaturesInCombinationDouble)");
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
            LOG(TraceLevelError, "ERROR CountFeatureCombinationsIndexes !IsNumberConvertable<size_t, IntegerDataType>(countFeaturesInCombination)");
            return SIZE_MAX;
         }
         const size_t cFeaturesInCombination = static_cast<size_t>(countFeaturesInCombination);
         if(IsAddError(cFeatureCombinationsIndexes, cFeaturesInCombination)) {
            LOG(TraceLevelError, "ERROR CountFeatureCombinationsIndexes IsAddError(cFeatureCombinationsIndexes, cFeaturesInCombination)");
            return SIZE_MAX;
         }
         cFeatureCombinationsIndexes += cFeaturesInCombination;
         ++pFeatureCombination;
      } while(pFeatureCombinationEnd != pFeatureCombination);
   }
   return cFeatureCombinationsIndexes;
}

IntegerDataType * ConvertDoublesToIndexes(const SEXP items, size_t * const pcItems) {
   EBM_ASSERT(nullptr != items);
   EBM_ASSERT(nullptr != pcItems);
   if(REALSXP != TYPEOF(items)) {
      LOG(TraceLevelError, "ERROR ConvertDoublesToIndexes REALSXP != TYPEOF(items)");
      return nullptr;
   }
   const R_xlen_t countItemsR = xlength(items);
   if(!IsNumberConvertable<size_t, R_xlen_t>(countItemsR)) {
      LOG(TraceLevelError, "ERROR ConvertDoublesToIndexes !IsNumberConvertable<size_t, R_xlen_t>(countItemsR)");
      return nullptr;
   }
   const size_t cItems = static_cast<size_t>(countItemsR);
   if(!IsNumberConvertable<IntegerDataType, size_t>(cItems)) {
      LOG(TraceLevelError, "ERROR ConvertDoublesToIndexes !IsNumberConvertable<IntegerDataType, size_t>(cItems)");
      return nullptr;
   }
   *pcItems = cItems;

   IntegerDataType * aItems = static_cast<IntegerDataType *>(INVALID_POINTER);
   if(0 != cItems) {
      aItems = reinterpret_cast<IntegerDataType *>(R_alloc(cItems, static_cast<int>(sizeof(IntegerDataType))));
      IntegerDataType * pItem = aItems;
      const IntegerDataType * const pItemEnd = aItems + cItems;
      const double * pOriginal = REAL(items);
      do {
         const double val = *pOriginal;
         if(!IsDoubleToIntegerDataTypeIndexValid(val)) {
            LOG(TraceLevelError, "ERROR ConvertDoublesToIndexes !IsDoubleToIntegerDataTypeIndexValid(val)");
            return nullptr;
         }
         *pItem = static_cast<IntegerDataType>(val);
         ++pItem;
         ++pOriginal;
      } while(pItemEnd != pItem);
   }
   return aItems;
}

FractionalDataType * ConvertDoublesToDoubles(const SEXP items, size_t * const pcItems) {
   EBM_ASSERT(nullptr != items);
   EBM_ASSERT(nullptr != pcItems);
   if(REALSXP != TYPEOF(items)) {
      LOG(TraceLevelError, "ERROR ConvertDoublesToIndexes REALSXP != TYPEOF(items)");
      return nullptr;
   }
   const R_xlen_t countItemsR = xlength(items);
   if(!IsNumberConvertable<size_t, R_xlen_t>(countItemsR)) {
      LOG(TraceLevelError, "ERROR ConvertDoublesToIndexes !IsNumberConvertable<size_t, R_xlen_t>(countItemsR)");
      return nullptr;
   }
   const size_t cItems = static_cast<size_t>(countItemsR);
   if(!IsNumberConvertable<IntegerDataType, size_t>(cItems)) {
      LOG(TraceLevelError, "ERROR ConvertDoublesToIndexes !IsNumberConvertable<IntegerDataType, size_t>(cItems)");
      return nullptr;
   }
   *pcItems = cItems;

   FractionalDataType * aItems = static_cast<FractionalDataType *>(INVALID_POINTER);
   if(0 != cItems) {
      aItems = reinterpret_cast<FractionalDataType *>(R_alloc(cItems, static_cast<int>(sizeof(FractionalDataType))));
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
   return aItems;
}

extern "C" {

   SEXP InitializeTrainingRegression_R(
      SEXP randomSeed,
      SEXP features,
      SEXP featureCombinations,
      SEXP featureCombinationIndexes,
      SEXP trainingTargets,
      SEXP trainingBinnedData,
      SEXP trainingPredictorScores,
      SEXP validationTargets,
      SEXP validationBinnedData,
      SEXP validationPredictorScores,
      SEXP countInnerBags
   ) {
      EBM_ASSERT(nullptr != randomSeed);
      EBM_ASSERT(nullptr != features);
      EBM_ASSERT(nullptr != featureCombinations);
      EBM_ASSERT(nullptr != featureCombinationIndexes);
      EBM_ASSERT(nullptr != trainingTargets);
      EBM_ASSERT(nullptr != trainingBinnedData);
      EBM_ASSERT(nullptr != trainingPredictorScores);
      EBM_ASSERT(nullptr != validationTargets);
      EBM_ASSERT(nullptr != validationBinnedData);
      EBM_ASSERT(nullptr != validationPredictorScores);
      EBM_ASSERT(nullptr != countInnerBags);

      if(!IsSingleIntVector(randomSeed)) {
         LOG(TraceLevelError, "ERROR InitializeTrainingRegression_R !IsSingleIntVector(randomSeed)");
         return R_NilValue;
      }
      // we don't care if the seed is clipped or doesn't fit, or whatever.  Casting to unsigned avoids undefined behavior issues with casting between signed values.  
      const IntegerDataType randomSeedLocal = static_cast<IntegerDataType>(static_cast<unsigned int>(INTEGER(randomSeed)[0]));

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
      IntegerDataType * const aFeatureCombinationIndexes = ConvertDoublesToIndexes(featureCombinationIndexes, &cFeatureCombinationsIndexesActual);
      if(nullptr == aFeatureCombinationIndexes) {
         // we've already logged any errors
         return R_NilValue;
      }
      if(cFeatureCombinationsIndexesActual != cFeatureCombinationsIndexesCheck) {
         LOG(TraceLevelError, "ERROR InitializeTrainingRegression_R cFeatureCombinationsIndexesActual != cFeatureCombinationsIndexesCheck");
         return R_NilValue;
      }

      size_t cTrainingInstances;
      FractionalDataType * const aTrainingTargets = ConvertDoublesToDoubles(trainingTargets, &cTrainingInstances);
      if(nullptr == aTrainingTargets) {
         // we've already logged any errors
         return R_NilValue;
      }
      const IntegerDataType countTrainingTargets = static_cast<IntegerDataType>(cTrainingInstances);

      size_t cTrainingBinnedData;
      IntegerDataType * const aTrainingBinnedData = ConvertDoublesToIndexes(trainingBinnedData, &cTrainingBinnedData);
      if(nullptr == aTrainingBinnedData) {
         // we've already logged any errors
         return R_NilValue;
      }
      if(IsMultiplyError(cTrainingInstances, cFeatures)) {
         LOG(TraceLevelError, "ERROR InitializeTrainingRegression_R IsMultiplyError(cTrainingInstances, cFeatures)");
         return R_NilValue;
      }
      if(cTrainingInstances * cFeatures != cTrainingBinnedData) {
         LOG(TraceLevelError, "ERROR InitializeTrainingRegression_R cTrainingInstances * cFeatures != cTrainingBinnedData");
         return R_NilValue;
      }

      FractionalDataType * aTrainingPredictorScores = nullptr;
      if(NILSXP != TYPEOF(trainingPredictorScores)) {
         size_t cTrainingPredictorScores;
         aTrainingPredictorScores = ConvertDoublesToDoubles(trainingPredictorScores, &cTrainingPredictorScores);
         if(nullptr == aTrainingPredictorScores) {
            // we've already logged any errors
            return R_NilValue;
         }
         if(cTrainingInstances != cTrainingPredictorScores) {
            LOG(TraceLevelError, "ERROR InitializeTrainingRegression_R cTrainingInstances != cTrainingPredictorScores");
            return R_NilValue;
         }
      }

      size_t cValidationInstances;
      FractionalDataType * const aValidationTargets = ConvertDoublesToDoubles(validationTargets, &cValidationInstances);
      if(nullptr == aValidationTargets) {
         // we've already logged any errors
         return R_NilValue;
      }
      const IntegerDataType countValidationTargets = static_cast<IntegerDataType>(cValidationInstances);

      size_t cValidationBinnedData;
      IntegerDataType * const aValidationBinnedData = ConvertDoublesToIndexes(validationBinnedData, &cValidationBinnedData);
      if(nullptr == aValidationBinnedData) {
         // we've already logged any errors
         return R_NilValue;
      }
      if(IsMultiplyError(cValidationInstances, cFeatures)) {
         LOG(TraceLevelError, "ERROR InitializeTrainingRegression_R IsMultiplyError(cValidationInstances, cFeatures)");
         return R_NilValue;
      }
      if(cValidationInstances * cFeatures != cValidationBinnedData) {
         LOG(TraceLevelError, "ERROR InitializeTrainingRegression_R cValidationInstances * cFeatures != cValidationBinnedData");
         return R_NilValue;
      }

      FractionalDataType * aValidationPredictorScores = nullptr;
      if(NILSXP != TYPEOF(validationPredictorScores)) {
         size_t cValidationPredictorScores;
         aValidationPredictorScores = ConvertDoublesToDoubles(validationPredictorScores, &cValidationPredictorScores);
         if(nullptr == aValidationPredictorScores) {
            // we've already logged any errors
            return R_NilValue;
         }
         if(cValidationInstances != cValidationPredictorScores) {
            LOG(TraceLevelError, "ERROR InitializeTrainingRegression_R cValidationInstances != cValidationPredictorScores");
            return R_NilValue;
         }
      }

      if(!IsSingleIntVector(countInnerBags)) {
         LOG(TraceLevelError, "ERROR InitializeTrainingRegression_R !IsSingleIntVector(countInnerBags)");
         return R_NilValue;
      }
      int countInnerBagsInt = INTEGER(countInnerBags)[0];
      if(!IsNumberConvertable<IntegerDataType, int>(countInnerBagsInt)) {
         LOG(TraceLevelError, "ERROR InitializeTrainingRegression_R !IsNumberConvertable<IntegerDataType, int>(countInnerBagsInt)");
         return nullptr;
      }
      IntegerDataType countInnerBagsLocal = static_cast<IntegerDataType>(countInnerBagsInt);

      PEbmTraining pEbmTraining = InitializeTrainingRegression(randomSeedLocal, countFeatures, aFeatures, countFeatureCombinations, aFeatureCombinations, aFeatureCombinationIndexes, countTrainingTargets, aTrainingTargets, aTrainingBinnedData, aTrainingPredictorScores, countValidationTargets, aValidationTargets, aValidationBinnedData, aValidationPredictorScores, countInnerBagsLocal);

      if(nullptr == pEbmTraining) {
         return R_NilValue;
      } else {
         SEXP trainingRPointer = R_MakeExternalPtr(static_cast<void *>(pEbmTraining), R_NilValue, R_NilValue); // makes an EXTPTRSXP
         PROTECT(trainingRPointer);

         R_RegisterCFinalizerEx(trainingRPointer, &TrainingFinalizer, Rboolean::TRUE);

         UNPROTECT(1);
         return trainingRPointer;
      }
   }

   SEXP InitializeTrainingClassification_R(
      SEXP randomSeed,
      SEXP features,
      SEXP featureCombinations,
      SEXP featureCombinationIndexes,
      SEXP countTargetClasses,
      SEXP trainingTargets,
      SEXP trainingBinnedData,
      SEXP trainingPredictorScores,
      SEXP validationTargets,
      SEXP validationBinnedData,
      SEXP validationPredictorScores,
      SEXP countInnerBags
   ) {
      EBM_ASSERT(nullptr != randomSeed);
      EBM_ASSERT(nullptr != features);
      EBM_ASSERT(nullptr != featureCombinations);
      EBM_ASSERT(nullptr != featureCombinationIndexes);
      EBM_ASSERT(nullptr != countTargetClasses);
      EBM_ASSERT(nullptr != trainingTargets);
      EBM_ASSERT(nullptr != trainingBinnedData);
      EBM_ASSERT(nullptr != trainingPredictorScores);
      EBM_ASSERT(nullptr != validationTargets);
      EBM_ASSERT(nullptr != validationBinnedData);
      EBM_ASSERT(nullptr != validationPredictorScores);
      EBM_ASSERT(nullptr != countInnerBags);

      if(!IsSingleIntVector(randomSeed)) {
         LOG(TraceLevelError, "ERROR InitializeTrainingClassification_R !IsSingleIntVector(randomSeed)");
         return R_NilValue;
      }
      // we don't care if the seed is clipped or doesn't fit, or whatever.  Casting to unsigned avoids undefined behavior issues with casting between signed values.  
      const IntegerDataType randomSeedLocal = static_cast<IntegerDataType>(static_cast<unsigned int>(INTEGER(randomSeed)[0]));

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
      IntegerDataType * const aFeatureCombinationIndexes = ConvertDoublesToIndexes(featureCombinationIndexes, &cFeatureCombinationsIndexesActual);
      if(nullptr == aFeatureCombinationIndexes) {
         // we've already logged any errors
         return R_NilValue;
      }
      if(cFeatureCombinationsIndexesActual != cFeatureCombinationsIndexesCheck) {
         LOG(TraceLevelError, "ERROR InitializeTrainingClassification_R cFeatureCombinationsIndexesActual != cFeatureCombinationsIndexesCheck");
         return R_NilValue;
      }

      if(!IsSingleDoubleVector(countTargetClasses)) {
         LOG(TraceLevelError, "ERROR InitializeTrainingClassification_R !IsSingleDoubleVector(countTargetClasses)");
         return R_NilValue;
      }
      double countTargetClassesDouble = REAL(countTargetClasses)[0];
      if(!IsDoubleToIntegerDataTypeIndexValid(countTargetClassesDouble)) {
         LOG(TraceLevelError, "ERROR InitializeTrainingClassification_R !IsDoubleToIntegerDataTypeIndexValid(countTargetClassesDouble)");
         return R_NilValue;
      }
      const size_t cTargetClasses = static_cast<size_t>(countTargetClassesDouble);
      if(!IsNumberConvertable<ptrdiff_t, size_t>(cTargetClasses)) {
         LOG(TraceLevelError, "ERROR InitializeInteractionClassification_R !IsNumberConvertable<ptrdiff_t, size_t>(cTargetClasses)");
         return R_NilValue;
      }
      const size_t cVectorLength = GetVectorLengthFlatCore(static_cast<ptrdiff_t>(cTargetClasses));

      size_t cTrainingInstances;
      IntegerDataType * const aTrainingTargets = ConvertDoublesToIndexes(trainingTargets, &cTrainingInstances);
      if(nullptr == aTrainingTargets) {
         // we've already logged any errors
         return R_NilValue;
      }
      const IntegerDataType countTrainingTargets = static_cast<IntegerDataType>(cTrainingInstances);

      size_t cTrainingBinnedData;
      IntegerDataType * const aTrainingBinnedData = ConvertDoublesToIndexes(trainingBinnedData, &cTrainingBinnedData);
      if(nullptr == aTrainingBinnedData) {
         // we've already logged any errors
         return R_NilValue;
      }
      if(IsMultiplyError(cTrainingInstances, cFeatures)) {
         LOG(TraceLevelError, "ERROR InitializeTrainingClassification_R IsMultiplyError(cTrainingInstances, cFeatures)");
         return R_NilValue;
      }
      if(cTrainingInstances * cFeatures != cTrainingBinnedData) {
         LOG(TraceLevelError, "ERROR InitializeTrainingClassification_R cTrainingInstances * cFeatures != cTrainingBinnedData");
         return R_NilValue;
      }

      FractionalDataType * aTrainingPredictorScores = nullptr;
      if(NILSXP != TYPEOF(trainingPredictorScores)) {
         size_t cTrainingPredictorScores;
         aTrainingPredictorScores = ConvertDoublesToDoubles(trainingPredictorScores, &cTrainingPredictorScores);
         if(nullptr == aTrainingPredictorScores) {
            // we've already logged any errors
            return R_NilValue;
         }
         if(IsMultiplyError(cTrainingInstances, cVectorLength)) {
            LOG(TraceLevelError, "ERROR InitializeTrainingClassification_R IsMultiplyError(cTrainingInstances, cVectorLength)");
            return R_NilValue;
         }
         if(cVectorLength * cTrainingInstances != cTrainingPredictorScores) {
            LOG(TraceLevelError, "ERROR InitializeTrainingClassification_R cVectorLength * cTrainingInstances != cTrainingPredictorScores");
            return R_NilValue;
         }
      }

      size_t cValidationInstances;
      IntegerDataType * const aValidationTargets = ConvertDoublesToIndexes(validationTargets, &cValidationInstances);
      if(nullptr == aValidationTargets) {
         // we've already logged any errors
         return R_NilValue;
      }
      const IntegerDataType countValidationTargets = static_cast<IntegerDataType>(cValidationInstances);

      size_t cValidationBinnedData;
      IntegerDataType * const aValidationBinnedData = ConvertDoublesToIndexes(validationBinnedData, &cValidationBinnedData);
      if(nullptr == aValidationBinnedData) {
         // we've already logged any errors
         return R_NilValue;
      }
      if(IsMultiplyError(cValidationInstances, cFeatures)) {
         LOG(TraceLevelError, "ERROR InitializeTrainingClassification_R IsMultiplyError(cValidationInstances, cFeatures)");
         return R_NilValue;
      }
      if(cValidationInstances * cFeatures != cValidationBinnedData) {
         LOG(TraceLevelError, "ERROR InitializeTrainingClassification_R cValidationInstances * cFeatures != cValidationBinnedData");
         return R_NilValue;
      }

      FractionalDataType * aValidationPredictorScores = nullptr;
      if(NILSXP != TYPEOF(validationPredictorScores)) {
         size_t cValidationPredictorScores;
         aValidationPredictorScores = ConvertDoublesToDoubles(validationPredictorScores, &cValidationPredictorScores);
         if(nullptr == aValidationPredictorScores) {
            // we've already logged any errors
            return R_NilValue;
         }
         if(IsMultiplyError(cValidationInstances, cVectorLength)) {
            LOG(TraceLevelError, "ERROR InitializeTrainingClassification_R IsMultiplyError(cValidationInstances, cVectorLength)");
            return R_NilValue;
         }
         if(cVectorLength * cValidationInstances != cValidationPredictorScores) {
            LOG(TraceLevelError, "ERROR InitializeTrainingClassification_R cVectorLength * cValidationInstances != cValidationPredictorScores");
            return R_NilValue;
         }
      }

      if(!IsSingleIntVector(countInnerBags)) {
         LOG(TraceLevelError, "ERROR InitializeTrainingClassification_R !IsSingleIntVector(countInnerBags)");
         return R_NilValue;
      }
      int countInnerBagsInt = INTEGER(countInnerBags)[0];
      if(!IsNumberConvertable<IntegerDataType, int>(countInnerBagsInt)) {
         LOG(TraceLevelError, "ERROR InitializeTrainingClassification_R !IsNumberConvertable<IntegerDataType, int>(countInnerBagsInt)");
         return nullptr;
      }
      IntegerDataType countInnerBagsLocal = static_cast<IntegerDataType>(countInnerBagsInt);

      PEbmTraining pEbmTraining = InitializeTrainingClassification(randomSeedLocal, countFeatures, aFeatures, countFeatureCombinations, aFeatureCombinations, aFeatureCombinationIndexes, static_cast<IntegerDataType>(cTargetClasses), countTrainingTargets, aTrainingTargets, aTrainingBinnedData, aTrainingPredictorScores, countValidationTargets, aValidationTargets, aValidationBinnedData, aValidationPredictorScores, countInnerBagsLocal);

      if(nullptr == pEbmTraining) {
         return R_NilValue;
      } else {
         SEXP trainingRPointer = R_MakeExternalPtr(static_cast<void *>(pEbmTraining), R_NilValue, R_NilValue); // makes an EXTPTRSXP
         PROTECT(trainingRPointer);

         R_RegisterCFinalizerEx(trainingRPointer, &TrainingFinalizer, Rboolean::TRUE);

         UNPROTECT(1);
         return trainingRPointer;
      }
   }

   SEXP TrainingStep_R(
      SEXP ebmTraining,
      SEXP indexFeatureCombination,
      SEXP learningRate,
      SEXP countTreeSplitsMax,
      SEXP countInstancesRequiredForParentSplitMin,
      SEXP trainingWeights,
      SEXP validationWeights
   ) {
      EBM_ASSERT(nullptr != ebmTraining);
      EBM_ASSERT(nullptr != indexFeatureCombination);
      EBM_ASSERT(nullptr != learningRate);
      EBM_ASSERT(nullptr != countTreeSplitsMax);
      EBM_ASSERT(nullptr != countInstancesRequiredForParentSplitMin);
      EBM_ASSERT(nullptr != trainingWeights);
      EBM_ASSERT(nullptr != validationWeights);

      if(EXTPTRSXP != TYPEOF(ebmTraining)) {
         LOG(TraceLevelError, "ERROR TrainingStep_R EXTPTRSXP != TYPEOF(ebmTraining)");
         return R_NilValue;
      }
      EbmTrainingState * pEbmTraining = static_cast<EbmTrainingState *>(R_ExternalPtrAddr(ebmTraining));
      if(nullptr == pEbmTraining) {
         LOG(TraceLevelError, "ERROR TrainingStep_R nullptr == pEbmTraining");
         return R_NilValue;
      }

      if(!IsSingleDoubleVector(indexFeatureCombination)) {
         LOG(TraceLevelError, "ERROR TrainingStep_R !IsSingleDoubleVector(indexFeatureCombination)");
         return R_NilValue;
      }
      double doubleIndex = REAL(indexFeatureCombination)[0];
      if(!IsDoubleToIntegerDataTypeIndexValid(doubleIndex)) {
         LOG(TraceLevelError, "ERROR TrainingStep_R !IsDoubleToIntegerDataTypeIndexValid(doubleIndex)");
         return R_NilValue;
      }
      IntegerDataType iFeatureCombination = static_cast<IntegerDataType>(doubleIndex);

      if(!IsSingleDoubleVector(learningRate)) {
         LOG(TraceLevelError, "ERROR TrainingStep_R !IsSingleDoubleVector(learningRate)");
         return R_NilValue;
      }
      double learningRateLocal = REAL(learningRate)[0];

      if(!IsSingleDoubleVector(countTreeSplitsMax)) {
         LOG(TraceLevelError, "ERROR TrainingStep_R !IsSingleDoubleVector(countTreeSplitsMax)");
         return R_NilValue;
      }
      double doubleCountTreeSplitsMax = REAL(countTreeSplitsMax)[0];
      IntegerDataType cTreeSplitsMax;
      static_assert(std::numeric_limits<double>::is_iec559, "we need is_iec559 to know that comparisons to infinity and -infinity to normal numbers work");
      if(std::isnan(doubleCountTreeSplitsMax) || static_cast<double>(std::numeric_limits<IntegerDataType>::max()) < doubleCountTreeSplitsMax) {
         LOG(TraceLevelWarning, "WARNING TrainingStep_R countTreeSplitsMax overflow");
         cTreeSplitsMax = std::numeric_limits<IntegerDataType>::max();
      } else if(doubleCountTreeSplitsMax < static_cast<double>(std::numeric_limits<IntegerDataType>::lowest())) {
         LOG(TraceLevelWarning, "WARNING TrainingStep_R countTreeSplitsMax underflow");
         cTreeSplitsMax = std::numeric_limits<IntegerDataType>::lowest();
      } else {
         cTreeSplitsMax = static_cast<IntegerDataType>(doubleCountTreeSplitsMax);
      }

      if(!IsSingleDoubleVector(countInstancesRequiredForParentSplitMin)) {
         LOG(TraceLevelError, "ERROR TrainingStep_R !IsSingleDoubleVector(countInstancesRequiredForParentSplitMin)");
         return R_NilValue;
      }
      double doubleCountInstancesRequiredForParentSplitMin = REAL(countInstancesRequiredForParentSplitMin)[0];
      IntegerDataType cInstancesRequiredForParentSplitMin;
      static_assert(std::numeric_limits<double>::is_iec559, "we need is_iec559 to know that comparisons to infinity and -infinity to normal numbers work");
      if(std::isnan(doubleCountInstancesRequiredForParentSplitMin) || static_cast<double>(std::numeric_limits<IntegerDataType>::max()) < doubleCountInstancesRequiredForParentSplitMin) {
         LOG(TraceLevelWarning, "WARNING TrainingStep_R countInstancesRequiredForParentSplitMin overflow");
         cInstancesRequiredForParentSplitMin = std::numeric_limits<IntegerDataType>::max();
      } else if(doubleCountInstancesRequiredForParentSplitMin < static_cast<double>(std::numeric_limits<IntegerDataType>::lowest())) {
         LOG(TraceLevelWarning, "WARNING TrainingStep_R countInstancesRequiredForParentSplitMin underflow");
         cInstancesRequiredForParentSplitMin = std::numeric_limits<IntegerDataType>::lowest();
      } else {
         cInstancesRequiredForParentSplitMin = static_cast<IntegerDataType>(doubleCountInstancesRequiredForParentSplitMin);
      }

      double * pTrainingWeights = nullptr;
      double * pValidationWeights = nullptr;
      if(NILSXP != TYPEOF(trainingWeights) || NILSXP != TYPEOF(validationWeights)) {
         if(REALSXP != TYPEOF(trainingWeights)) {
            LOG(TraceLevelError, "ERROR TrainingStep_R REALSXP != TYPEOF(trainingWeights)");
            return R_NilValue;
         }
         R_xlen_t trainingWeightsLength = xlength(trainingWeights);
         if(!IsNumberConvertable<size_t, R_xlen_t>(trainingWeightsLength)) {
            LOG(TraceLevelError, "ERROR TrainingStep_R !IsNumberConvertable<size_t, R_xlen_t>(trainingWeightsLength)");
            return R_NilValue;
         }
         size_t cTrainingWeights = static_cast<size_t>(trainingWeightsLength);
         if(cTrainingWeights != pEbmTraining->m_pTrainingSet->GetCountInstances()) {
            LOG(TraceLevelError, "ERROR TrainingStep_R cTrainingWeights != pEbmTraining->m_pTrainingSet->GetCountInstances()");
            return R_NilValue;
         }
         pTrainingWeights = REAL(trainingWeights);

         if(REALSXP != TYPEOF(validationWeights)) {
            LOG(TraceLevelError, "ERROR TrainingStep_R REALSXP != TYPEOF(validationWeights)");
            return R_NilValue;
         }
         R_xlen_t validationWeightsLength = xlength(validationWeights);
         if(!IsNumberConvertable<size_t, R_xlen_t>(validationWeightsLength)) {
            LOG(TraceLevelError, "ERROR TrainingStep_R !IsNumberConvertable<size_t, R_xlen_t>(validationWeightsLength)");
            return R_NilValue;
         }
         size_t cValidationWeights = static_cast<size_t>(validationWeightsLength);
         if(cValidationWeights != pEbmTraining->m_pValidationSet->GetCountInstances()) {
            LOG(TraceLevelError, "ERROR TrainingStep_R cValidationWeights != pEbmTraining->m_pValidationSet->GetCountInstances()");
            return R_NilValue;
         }
         pValidationWeights = REAL(validationWeights);
      }

      FractionalDataType validationMetricReturn;
      if(0 != TrainingStep(reinterpret_cast<PEbmTraining>(pEbmTraining), iFeatureCombination, learningRateLocal, cTreeSplitsMax, cInstancesRequiredForParentSplitMin, pTrainingWeights, pValidationWeights, &validationMetricReturn)) {
         LOG(TraceLevelWarning, "WARNING TrainingStep_R TrainingStep returned error code");
         return R_NilValue;
      }

      SEXP ret = PROTECT(allocVector(REALSXP, R_xlen_t { 1 }));
      REAL(ret)[0] = validationMetricReturn;
      UNPROTECT(1);
      return ret;
   }

   SEXP GetCurrentModelFeatureCombination_R(
      SEXP ebmTraining,
      SEXP indexFeatureCombination
   ) {
      EBM_ASSERT(nullptr != ebmTraining); // shouldn't be possible
      EBM_ASSERT(nullptr != indexFeatureCombination); // shouldn't be possible

      if(EXTPTRSXP != TYPEOF(ebmTraining)) {
         LOG(TraceLevelError, "ERROR GetCurrentModelFeatureCombination_R EXTPTRSXP != TYPEOF(ebmTraining)");
         return R_NilValue;
      }
      EbmTrainingState * pEbmTraining = static_cast<EbmTrainingState *>(R_ExternalPtrAddr(ebmTraining));
      if(nullptr == pEbmTraining) {
         LOG(TraceLevelError, "ERROR GetCurrentModelFeatureCombination_R nullptr == pEbmTraining");
         return R_NilValue;
      }

      if(!IsSingleDoubleVector(indexFeatureCombination)) {
         LOG(TraceLevelError, "ERROR GetCurrentModelFeatureCombination_R !IsSingleDoubleVector(indexFeatureCombination)");
         return R_NilValue;
      }
      double doubleIndex = REAL(indexFeatureCombination)[0];
      if(!IsDoubleToIntegerDataTypeIndexValid(doubleIndex)) {
         LOG(TraceLevelError, "ERROR GetCurrentModelFeatureCombination_R !IsDoubleToIntegerDataTypeIndexValid(doubleIndex)");
         return R_NilValue;
      }
      IntegerDataType iFeatureCombination = static_cast<IntegerDataType>(doubleIndex);
      // we check if iFeatureCombination can fit into a size_t in IsDoubleToIntegerDataTypeIndexValid
      if(pEbmTraining->m_cFeatureCombinations <= static_cast<size_t>(iFeatureCombination)) {
         LOG(TraceLevelError, "ERROR GetCurrentModelFeatureCombination_R pEbmTraining->m_cFeatureCombinations <= static_cast<size_t>(iFeatureCombination)");
         return R_NilValue;
      }

      FractionalDataType * pModelFeatureCombinationTensor = GetCurrentModelFeatureCombination(reinterpret_cast<PEbmTraining>(pEbmTraining), iFeatureCombination);
      if(nullptr == pModelFeatureCombinationTensor) {
         // if nullptr == pModelFeatureCombinationTensor then either:
         //    1) m_cFeatureCombinations was 0, in which case this function would have undefined behavior since the caller needs to indicate a valid indexFeatureCombination, which is impossible, so we can do anything we like, include the below actions.
         //    2) m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0 (and the learning type is classification), which is legal, which we need to handle here
         SEXP ret = allocVector(REALSXP, R_xlen_t { 0 });
         LOG(TraceLevelWarning, "WARNING GetCurrentModelFeatureCombination_R nullptr == pModelFeatureCombinationTensor");
         return ret;
      }
      size_t cValues = GetVectorLengthFlatCore(pEbmTraining->m_runtimeLearningTypeOrCountTargetClasses);
      const FeatureCombinationCore * const pFeatureCombinationCore = pEbmTraining->m_apFeatureCombinations[static_cast<size_t>(iFeatureCombination)];
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

   SEXP GetBestModelFeatureCombination_R(
      SEXP ebmTraining,
      SEXP indexFeatureCombination
   ) {
      EBM_ASSERT(nullptr != ebmTraining); // shouldn't be possible
      EBM_ASSERT(nullptr != indexFeatureCombination); // shouldn't be possible

      if(EXTPTRSXP != TYPEOF(ebmTraining)) {
         LOG(TraceLevelError, "ERROR GetBestModelFeatureCombination_R EXTPTRSXP != TYPEOF(ebmTraining)");
         return R_NilValue;
      }
      EbmTrainingState * pEbmTraining = static_cast<EbmTrainingState *>(R_ExternalPtrAddr(ebmTraining));
      if(nullptr == pEbmTraining) {
         LOG(TraceLevelError, "ERROR GetBestModelFeatureCombination_R nullptr == pEbmTraining");
         return R_NilValue;
      }

      if(!IsSingleDoubleVector(indexFeatureCombination)) {
         LOG(TraceLevelError, "ERROR GetBestModelFeatureCombination_R !IsSingleDoubleVector(indexFeatureCombination)");
         return R_NilValue;
      }
      double doubleIndex = REAL(indexFeatureCombination)[0];
      if(!IsDoubleToIntegerDataTypeIndexValid(doubleIndex)) {
         LOG(TraceLevelError, "ERROR GetBestModelFeatureCombination_R !IsDoubleToIntegerDataTypeIndexValid(doubleIndex)");
         return R_NilValue;
      }
      IntegerDataType iFeatureCombination = static_cast<IntegerDataType>(doubleIndex);
      // we check that iFeatureCombination can be converted to size_t in IsDoubleToIntegerDataTypeIndexValid
      if(pEbmTraining->m_cFeatureCombinations <= static_cast<size_t>(iFeatureCombination)) {
         LOG(TraceLevelError, "ERROR GetBestModelFeatureCombination_R pEbmTraining->m_cFeatureCombinations <= static_cast<size_t>(iFeatureCombination)");
         return R_NilValue;
      }

      FractionalDataType * pModelFeatureCombinationTensor = GetBestModelFeatureCombination(reinterpret_cast<PEbmTraining>(pEbmTraining), iFeatureCombination);
      if(nullptr == pModelFeatureCombinationTensor) {
         // if nullptr == pModelFeatureCombinationTensor then either:
         //    1) m_cFeatureCombinations was 0, in which case this function would have undefined behavior since the caller needs to indicate a valid indexFeatureCombination, which is impossible, so we can do anything we like, include the below actions.
         //    2) m_runtimeLearningTypeOrCountTargetClasses was either 1 or 0 (and the learning type is classification), which is legal, which we need to handle here
         SEXP ret = allocVector(REALSXP, R_xlen_t { 0 });
         LOG(TraceLevelWarning, "WARNING GetBestModelFeatureCombination_R nullptr == pModelFeatureCombinationTensor");
         return ret;
      }
      size_t cValues = GetVectorLengthFlatCore(pEbmTraining->m_runtimeLearningTypeOrCountTargetClasses);
      const FeatureCombinationCore * const pFeatureCombinationCore = pEbmTraining->m_apFeatureCombinations[static_cast<size_t>(iFeatureCombination)];
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

   SEXP FreeTraining_R(
      SEXP ebmTraining
   ) {
      TrainingFinalizer(ebmTraining);
      return R_NilValue;
   }


   SEXP InitializeInteractionRegression_R(
      SEXP features,
      SEXP targets,
      SEXP binnedData,
      SEXP predictorScores
   ) {
      EBM_ASSERT(nullptr != features);
      EBM_ASSERT(nullptr != targets);
      EBM_ASSERT(nullptr != binnedData);
      EBM_ASSERT(nullptr != predictorScores);

      size_t cFeatures;
      EbmCoreFeature * const aFeatures = ConvertFeatures(features, &cFeatures);
      if(nullptr == aFeatures) {
         // we've already logged any errors
         return R_NilValue;
      }
      const IntegerDataType countFeatures = static_cast<IntegerDataType>(cFeatures); // the validity of this conversion was checked in ConvertFeatures(...)

      size_t cInstances;
      FractionalDataType * const aTargets = ConvertDoublesToDoubles(targets, &cInstances);
      if(nullptr == aTargets) {
         // we've already logged any errors
         return R_NilValue;
      }
      const IntegerDataType countInstances = static_cast<IntegerDataType>(cInstances);

      size_t cBinnedData;
      IntegerDataType * const aBinnedData = ConvertDoublesToIndexes(binnedData, &cBinnedData);
      if(nullptr == aBinnedData) {
         // we've already logged any errors
         return R_NilValue;
      }
      if(IsMultiplyError(cInstances, cFeatures)) {
         LOG(TraceLevelError, "ERROR InitializeInteractionRegression_R IsMultiplyError(cInstances, cFeatures)");
         return R_NilValue;
      }
      if(cInstances * cFeatures != cBinnedData) {
         LOG(TraceLevelError, "ERROR InitializeInteractionRegression_R cInstances * cFeatures != cBinnedData");
         return R_NilValue;
      }

      FractionalDataType * aPredictorScores = nullptr;
      if(NILSXP != TYPEOF(predictorScores)) {
         size_t cPredictorScores;
         aPredictorScores = ConvertDoublesToDoubles(predictorScores, &cPredictorScores);
         if(nullptr == aPredictorScores) {
            // we've already logged any errors
            return R_NilValue;
         }
         if(cInstances != cPredictorScores) {
            LOG(TraceLevelError, "ERROR InitializeInteractionRegression_R cInstances != cPredictorScores");
            return R_NilValue;
         }
      }

      PEbmInteraction pEbmInteraction = InitializeInteractionRegression(countFeatures, aFeatures, countInstances, aTargets, aBinnedData, aPredictorScores);

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

   SEXP InitializeInteractionClassification_R(
      SEXP features,
      SEXP countTargetClasses,
      SEXP targets,
      SEXP binnedData,
      SEXP predictorScores
   ) {
      EBM_ASSERT(nullptr != features);
      EBM_ASSERT(nullptr != countTargetClasses);
      EBM_ASSERT(nullptr != targets);
      EBM_ASSERT(nullptr != binnedData);
      EBM_ASSERT(nullptr != predictorScores);

      size_t cFeatures;
      EbmCoreFeature * const aFeatures = ConvertFeatures(features, &cFeatures);
      if(nullptr == aFeatures) {
         // we've already logged any errors
         return R_NilValue;
      }
      const IntegerDataType countFeatures = static_cast<IntegerDataType>(cFeatures); // the validity of this conversion was checked in ConvertFeatures(...)

      if(!IsSingleDoubleVector(countTargetClasses)) {
         LOG(TraceLevelError, "ERROR InitializeInteractionClassification_R !IsSingleDoubleVector(countTargetClasses)");
         return R_NilValue;
      }
      double countTargetClassesDouble = REAL(countTargetClasses)[0];
      if(!IsDoubleToIntegerDataTypeIndexValid(countTargetClassesDouble)) {
         LOG(TraceLevelError, "ERROR InitializeInteractionClassification_R !IsDoubleToIntegerDataTypeIndexValid(countTargetClassesDouble)");
         return R_NilValue;
      }
      const size_t cTargetClasses = static_cast<size_t>(countTargetClassesDouble);
      if(!IsNumberConvertable<ptrdiff_t, size_t>(cTargetClasses)) {
         LOG(TraceLevelError, "ERROR InitializeInteractionClassification_R !IsNumberConvertable<ptrdiff_t, size_t>(cTargetClasses)");
         return R_NilValue;
      }
      const size_t cVectorLength = GetVectorLengthFlatCore(static_cast<ptrdiff_t>(cTargetClasses));

      size_t cInstances;
      IntegerDataType * const aTargets = ConvertDoublesToIndexes(targets, &cInstances);
      if(nullptr == aTargets) {
         // we've already logged any errors
         return R_NilValue;
      }
      const IntegerDataType countInstances = static_cast<IntegerDataType>(cInstances);

      size_t cBinnedData;
      IntegerDataType * const aBinnedData = ConvertDoublesToIndexes(binnedData, &cBinnedData);
      if(nullptr == aBinnedData) {
         // we've already logged any errors
         return R_NilValue;
      }
      if(IsMultiplyError(cInstances, cFeatures)) {
         LOG(TraceLevelError, "ERROR InitializeInteractionClassification_R IsMultiplyError(cInstances, cFeatures)");
         return R_NilValue;
      }
      if(cInstances * cFeatures != cBinnedData) {
         LOG(TraceLevelError, "ERROR InitializeInteractionClassification_R cInstances * cFeatures != cBinnedData");
         return R_NilValue;
      }

      FractionalDataType * aPredictorScores = nullptr;
      if(NILSXP != TYPEOF(predictorScores)) {
         size_t cPredictorScores;
         aPredictorScores = ConvertDoublesToDoubles(predictorScores, &cPredictorScores);
         if(nullptr == aPredictorScores) {
            // we've already logged any errors
            return R_NilValue;
         }
         if(IsMultiplyError(cInstances, cVectorLength)) {
            LOG(TraceLevelError, "ERROR InitializeInteractionClassification_R IsMultiplyError(cInstances, cVectorLength)");
            return R_NilValue;
         }
         if(cVectorLength * cInstances != cPredictorScores) {
            LOG(TraceLevelError, "ERROR InitializeInteractionClassification_R cVectorLength * cInstances != cPredictorScores");
            return R_NilValue;
         }
      }

      PEbmInteraction pEbmInteraction = InitializeInteractionClassification(countFeatures, aFeatures, static_cast<IntegerDataType>(cTargetClasses), countInstances, aTargets, aBinnedData, aPredictorScores);

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
         LOG(TraceLevelError, "ERROR GetInteractionScore_R EXTPTRSXP != TYPEOF(ebmInteraction)");
         return R_NilValue;
      }
      EbmInteractionState * pEbmInteraction = static_cast<EbmInteractionState *>(R_ExternalPtrAddr(ebmInteraction));
      if(nullptr == pEbmInteraction) {
         LOG(TraceLevelError, "ERROR GetInteractionScore_R nullptr == pEbmInteraction");
         return R_NilValue;
      }

      size_t cFeaturesInCombination;
      IntegerDataType * const aFeatureIndexes = ConvertDoublesToIndexes(featureIndexes, &cFeaturesInCombination);
      if(nullptr == aFeatureIndexes) {
         // we've already logged any errors
         return R_NilValue;
      }
      IntegerDataType countFeaturesInCombination = static_cast<IntegerDataType>(cFeaturesInCombination);

      FractionalDataType interactionScoreReturn;
      if(0 != GetInteractionScore(reinterpret_cast<PEbmInteraction>(pEbmInteraction), countFeaturesInCombination, aFeatureIndexes, &interactionScoreReturn)) {
         LOG(TraceLevelWarning, "WARNING GetInteractionScore_R GetInteractionScore returned error code");
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
      { "InitializeTrainingRegression_R", (DL_FUNC)& InitializeTrainingRegression_R, 11 },
      { "InitializeTrainingClassification_R", (DL_FUNC)& InitializeTrainingClassification_R, 12 },
      { "TrainingStep_R", (DL_FUNC)& TrainingStep_R, 7 },
      { "GetCurrentModelFeatureCombination_R", (DL_FUNC)& GetCurrentModelFeatureCombination_R, 2 },
      { "GetBestModelFeatureCombination_R", (DL_FUNC)& GetBestModelFeatureCombination_R, 2 },
      { "FreeTraining_R", (DL_FUNC)& FreeTraining_R, 1 },
      { "InitializeInteractionRegression_R", (DL_FUNC)& InitializeInteractionRegression_R, 4 },
      { "InitializeInteractionClassification_R", (DL_FUNC)& InitializeInteractionClassification_R, 5 },
      { "GetInteractionScore_R", (DL_FUNC)& GetInteractionScore_R, 2 },
      { "FreeInteraction_R", (DL_FUNC)& FreeInteraction_R, 1 },
      { NULL, NULL, 0 }
   };

#ifdef _WIN32
   __declspec(dllexport)
#endif  // _WIN32
   void R_init_interpret(DllInfo * dllInfo) {
      R_registerRoutines(dllInfo, NULL, g_exposedFunctions, NULL, NULL);
      R_useDynamicSymbols(dllInfo, FALSE);
   }
}
