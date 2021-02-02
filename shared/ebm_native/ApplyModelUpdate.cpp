// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"

#include "EbmInternal.h"
// very independent includes
#include "Logging.h" // EBM_ASSERT & LOG

// FeatureGroup.h depends on FeatureInternal.h
#include "FeatureGroup.h"

#include "Booster.h"
#include "ThreadStateBoosting.h"

extern void ApplyModelUpdateTraining(
   ThreadStateBoosting * const pThreadStateBoosting,
   const FeatureGroup * const pFeatureGroup,
   const FloatEbmType * const aModelFeatureGroupUpdateTensor
);

extern FloatEbmType ApplyModelUpdateValidation(
   Booster * const pBooster,
   const FeatureGroup * const pFeatureGroup,
   const FloatEbmType * const aModelFeatureGroupUpdateTensor
);

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
static IntEbmType ApplyModelUpdateInternal(
   ThreadStateBoosting * const pThreadStateBoosting,
   const size_t iFeatureGroup,
   FloatEbmType * const pValidationMetricReturn
) {
   LOG_0(TraceLevelVerbose, "Entered ApplyModelUpdateInternal");

   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   const FeatureGroup * const pFeatureGroup = pBooster->GetFeatureGroups()[iFeatureGroup];
   const size_t cDimensions = pFeatureGroup->GetCountFeatures();

   if(0 != cDimensions) {
      const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
      size_t acDivisionIntegersEnd[k_cDimensionsMax];
      size_t iDimension = 0;
      do {
         acDivisionIntegersEnd[iDimension] = pFeatureGroupEntry[iDimension].m_pFeature->GetCountBins();
         ++iDimension;
      } while(iDimension < cDimensions);
      if(pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->Expand(acDivisionIntegersEnd)) {
         if(nullptr != pValidationMetricReturn) {
            *pValidationMetricReturn = FloatEbmType { 0 };
         }
         return IntEbmType { 1 };
      }
   }

   // m_apCurrentModel can be null if there are no featureGroups (but we have an feature group index), 
   // or if the target has 1 or 0 classes (which we check before calling this function), so it shouldn't be possible to be null
   EBM_ASSERT(nullptr != pBooster->GetCurrentModel());
   // m_apCurrentModel can be null if there are no featureGroups (but we have an feature group index), 
   // or if the target has 1 or 0 classes (which we check before calling this function), so it shouldn't be possible to be null
   EBM_ASSERT(nullptr != pBooster->GetBestModel());

   const FloatEbmType * const aModelFeatureGroupUpdateTensor = pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->GetValuePointer();
   EBM_ASSERT(nullptr != aModelFeatureGroupUpdateTensor);

   // our caller can give us one of these bad types of inputs:
   //  1) NaN values
   //  2) +-infinity
   //  3) numbers that are fine, but when added to our existing model overflow to +-infinity
   // Our caller should really just not pass us the first two, but it's hard for our caller to protect against giving us values that won't overflow
   // so we should have some reasonable way to handle them.  If we were meant to overflow, logits or regression values at the maximum/minimum values
   // of doubles should be so close to infinity that it won't matter, and then you can at least graph them without overflowing to special values
   // We have the same problem when we go to make changes to the individual sample updates, but there we can have two graphs that combined push towards
   // an overflow to +-infinity.  We just ignore those overflows, because checking for them would add branches that we don't want, and we can just
   // propagate +-infinity and NaN values to the point where we get a metric and that should cause our client to stop boosting when our metric
   // overlfows and gets converted to the maximum value which will mean the metric won't be changing or improving after that.
   // This is an acceptable compromise.  We protect our models since the user might want to extract them AFTER we overlfow our measurment metric
   // so we don't want to overflow the values to NaN or +-infinity there, and it's very cheap for us to check for overflows when applying the model
   pBooster->GetCurrentModel()[iFeatureGroup]->AddExpandedWithBadValueProtection(aModelFeatureGroupUpdateTensor);

   if(0 != pBooster->GetTrainingSet()->GetCountSamples()) {
      ApplyModelUpdateTraining(
         pThreadStateBoosting,
         pFeatureGroup,
         aModelFeatureGroupUpdateTensor
      );
   }

   FloatEbmType modelMetric = FloatEbmType { 0 };
   if(0 != pBooster->GetValidationSet()->GetCountSamples()) {
      // if there is no validation set, it's pretty hard to know what the metric we'll get for our validation set
      // we could in theory return anything from zero to infinity or possibly, NaN (probably legally the best), but we return 0 here
      // because we want to kick our caller out of any loop it might be calling us in.  Infinity and NaN are odd values that might cause problems in
      // a caller that isn't expecting those values, so 0 is the safest option, and our caller can avoid the situation entirely by not calling
      // us with zero count validation sets

      // if the count of training samples is zero, don't update the best model (it will stay as all zeros), and we don't need to update our 
      // non-existant training set either C++ doesn't define what happens when you compare NaN to annother number.  It probably follows IEEE 754, 
      // but it isn't guaranteed, so let's check for zero samples in the validation set this better way
      // https://stackoverflow.com/questions/31225264/what-is-the-result-of-comparing-a-number-with-nan

      modelMetric = ApplyModelUpdateValidation(
         pBooster,
         pFeatureGroup,
         aModelFeatureGroupUpdateTensor
      );

      EBM_ASSERT(!std::isnan(modelMetric)); // NaNs can happen, but we should have converted them
      EBM_ASSERT(!std::isinf(modelMetric)); // +infinity can happen, but we should have converted it
      // both log loss and RMSE need to be above zero.  If we got a negative number due to floating point 
      // instability we should have previously converted it to zero.
      EBM_ASSERT(FloatEbmType { 0 } <= modelMetric);

      // modelMetric is either logloss (classification) or mean squared error (mse) (regression).  In either case we want to minimize it.
      if(LIKELY(modelMetric < pBooster->GetBestModelMetric())) {
         // we keep on improving, so this is more likely than not, and we'll exit if it becomes negative a lot
         pBooster->SetBestModelMetric(modelMetric);

         // TODO : in the future don't copy over all SegmentedTensors.  We only need to copy the ones that changed, which we can detect if we 
         // use a linked list and array lookup for the same data structure
         size_t iModel = 0;
         size_t iModelEnd = pBooster->GetCountFeatureGroups();
         do {
            if(pBooster->GetBestModel()[iModel]->Copy(*pBooster->GetCurrentModel()[iModel])) {
               if(nullptr != pValidationMetricReturn) {
                  *pValidationMetricReturn = FloatEbmType { 0 };
               }
               LOG_0(TraceLevelVerbose, "Exited ApplyModelUpdateInternal with memory allocation error in copy");
               return 1;
            }
            ++iModel;
         } while(iModel != iModelEnd);
      }
   }
   if(nullptr != pValidationMetricReturn) {
      *pValidationMetricReturn = modelMetric;
   }

   LOG_0(TraceLevelVerbose, "Exited ApplyModelUpdateInternal");
   return 0;
}

// we made this a global because if we had put this variable inside the Booster object, then we would need to dereference that before 
// getting the count.  By making this global we can send a log message incase a bad Booster object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more 
// times than desired, but we can live with that
static int g_cLogApplyModelUpdateParametersMessages = 10;

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION ApplyModelUpdate(
   ThreadStateBoostingHandle threadStateBoostingHandle,
   FloatEbmType * validationMetricOut
) {
   LOG_COUNTED_N(
      &g_cLogApplyModelUpdateParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "ApplyModelUpdate: "
      "threadStateBoostingHandle=%p, "
      "validationMetricOut=%p"
      ,
      static_cast<void *>(threadStateBoostingHandle),
      static_cast<void *>(validationMetricOut)
   );

   ThreadStateBoosting * pThreadStateBoosting = reinterpret_cast<ThreadStateBoosting *>(threadStateBoostingHandle);
   if(nullptr == pThreadStateBoosting) {
      if(LIKELY(nullptr != validationMetricOut)) {
         *validationMetricOut = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR ApplyModelUpdate threadStateBoosting cannot be nullptr");
      return 1;
   }

   const size_t iFeatureGroup = pThreadStateBoosting->GetFeatureGroupIndex();
   if(ThreadStateBoosting::k_illegalFeatureGroupIndex == iFeatureGroup) {
      if(LIKELY(nullptr != validationMetricOut)) {
         *validationMetricOut = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR ApplyModelUpdate bad internal state.  No FeatureGroupIndex set");
      return 1;
   }
   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   EBM_ASSERT(nullptr != pBooster);
   EBM_ASSERT(iFeatureGroup < pBooster->GetCountFeatureGroups());
   EBM_ASSERT(nullptr != pBooster->GetFeatureGroups());

   LOG_COUNTED_0(
      pBooster->GetFeatureGroups()[iFeatureGroup]->GetPointerCountLogEnterApplyModelUpdateMessages(),
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered ApplyModelUpdate"
   );

   if(ptrdiff_t { 0 } == pBooster->GetRuntimeLearningTypeOrCountTargetClasses() || ptrdiff_t { 1 } == pBooster->GetRuntimeLearningTypeOrCountTargetClasses()) {
      // if there is only 1 target class for classification, then we can predict the output with 100% accuracy.  The model is a tensor with zero 
      // length array logits, which means for our representation that we have zero items in the array total.
      // since we can predit the output with 100% accuracy, our log loss is 0.
      if(nullptr != validationMetricOut) {
         *validationMetricOut = 0;
      }
      pThreadStateBoosting->SetFeatureGroupIndex(ThreadStateBoosting::k_illegalFeatureGroupIndex);
      LOG_COUNTED_0(
         pBooster->GetFeatureGroups()[iFeatureGroup]->GetPointerCountLogExitApplyModelUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited ApplyModelUpdate from runtimeLearningTypeOrCountTargetClasses <= 1"
      );
      return 0;
   }

   IntEbmType ret = ApplyModelUpdateInternal(
      pThreadStateBoosting,
      iFeatureGroup,
      validationMetricOut
   );
   if(0 != ret) {
      LOG_N(TraceLevelWarning, "WARNING ApplyModelUpdate returned %" IntEbmTypePrintf, ret);
   }

   pThreadStateBoosting->SetFeatureGroupIndex(ThreadStateBoosting::k_illegalFeatureGroupIndex);

   if(nullptr != validationMetricOut) {
      EBM_ASSERT(!std::isnan(*validationMetricOut)); // NaNs can happen, but we should have edited those before here
      EBM_ASSERT(!std::isinf(*validationMetricOut)); // infinities can happen, but we should have edited those before here
      // both log loss and RMSE need to be above zero.  We previously zero any values below zero, which can happen due to floating point instability.
      EBM_ASSERT(FloatEbmType { 0 } <= *validationMetricOut);
      LOG_COUNTED_N(
         pBooster->GetFeatureGroups()[iFeatureGroup]->GetPointerCountLogExitApplyModelUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited ApplyModelUpdate %" FloatEbmTypePrintf, *validationMetricOut
      );
   } else {
      LOG_COUNTED_0(
         pBooster->GetFeatureGroups()[iFeatureGroup]->GetPointerCountLogExitApplyModelUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited ApplyModelUpdate.  No validation pointer."
      );
   }
   return ret;
}

// we made this a global because if we had put this variable inside the Booster object, then we would need to dereference that before 
// getting the count.  By making this global we can send a log message incase a bad Booster object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more 
// times than desired, but we can live with that
static int g_cLogGetModelUpdateCutsParametersMessages = 10;

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GetModelUpdateCuts(
   ThreadStateBoostingHandle threadStateBoostingHandle,
   IntEbmType indexDimension,
   IntEbmType * countCutsInOut,
   IntEbmType * cutIndexesOut
) {
   LOG_COUNTED_N(
      &g_cLogGetModelUpdateCutsParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "GetModelUpdateCuts: "
      "threadStateBoostingHandle=%p, "
      "indexDimension=%" IntEbmTypePrintf ", "
      "countCutsInOut=%p"
      "cutIndexesOut=%p"
      ,
      static_cast<void *>(threadStateBoostingHandle),
      indexDimension, 
      static_cast<void *>(countCutsInOut),
      static_cast<void *>(cutIndexesOut)
   );

   if(nullptr == countCutsInOut) {
      LOG_0(TraceLevelError, "ERROR GetModelUpdateCuts countCutsInOut cannot be nullptr");
      return IntEbmType { 1 };
   }

   if(nullptr == cutIndexesOut) {
      *countCutsInOut = IntEbmType { 0 };
      LOG_0(TraceLevelError, "ERROR GetModelUpdateCuts cutIndexesOut cannot be nullptr");
      return IntEbmType { 1 };
   }

   ThreadStateBoosting * const pThreadStateBoosting = reinterpret_cast<ThreadStateBoosting *>(threadStateBoostingHandle);
   if(nullptr == pThreadStateBoosting) {
      *countCutsInOut = IntEbmType { 0 };
      LOG_0(TraceLevelError, "ERROR GetModelUpdateCuts threadStateBoosting cannot be nullptr");
      return IntEbmType { 1 };
   }

   const size_t iFeatureGroup = pThreadStateBoosting->GetFeatureGroupIndex();
   if(ThreadStateBoosting::k_illegalFeatureGroupIndex == iFeatureGroup) {
      *countCutsInOut = IntEbmType { 0 };
      LOG_0(TraceLevelError, "ERROR GetModelUpdateCuts bad internal state.  No FeatureGroupIndex set");
      return IntEbmType { 1 };
   }
   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   EBM_ASSERT(nullptr != pBooster);
   EBM_ASSERT(iFeatureGroup < pBooster->GetCountFeatureGroups());
   EBM_ASSERT(nullptr != pBooster->GetFeatureGroups());
   const FeatureGroup * const pFeatureGroup = pBooster->GetFeatureGroups()[iFeatureGroup];

   if(indexDimension < 0) {
      *countCutsInOut = IntEbmType { 0 };
      LOG_0(TraceLevelError, "ERROR GetModelUpdateCuts indexDimension must be positive");
      return IntEbmType { 1 };
   }
   if(!IsNumberConvertable<size_t>(indexDimension)) {
      *countCutsInOut = IntEbmType { 0 };
      LOG_0(TraceLevelError, "ERROR GetModelUpdateCuts indexDimension is too high to index");
      return IntEbmType { 1 };
   }
   const size_t iDimension = static_cast<size_t>(indexDimension);
   if(pFeatureGroup->GetCountFeatures() <= iDimension) {
      *countCutsInOut = IntEbmType { 0 };
      LOG_0(TraceLevelError, "ERROR GetModelUpdateCuts indexDimension above the number of dimensions that we have");
      return IntEbmType { 1 };
   }

   // cBins started from IntEbmType, so we should be able to convert back safely
   const size_t cBins = pFeatureGroup->GetFeatureGroupEntries()[iDimension].m_pFeature->GetCountBins();
   if(*countCutsInOut != static_cast<IntEbmType>(cBins - size_t { 1 })) {
      *countCutsInOut = IntEbmType { 0 };
      LOG_0(TraceLevelError, "ERROR GetModelUpdateCuts bad cut array length");
      return IntEbmType { 1 };
   }

   const size_t cCuts = pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->GetCountDivisions(iDimension);
   EBM_ASSERT(cCuts < cBins);
   const ActiveDataType * const aCutIndexes = pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->GetDivisionPointer(iDimension);

   // TODO: handle this better where we handle mismatches in index types
   static_assert(sizeof(*cutIndexesOut) == sizeof(*aCutIndexes), "not same type for cuts");
   memcpy(cutIndexesOut, aCutIndexes, sizeof(*aCutIndexes) * cCuts);

   EBM_ASSERT(IsNumberConvertable<IntEbmType>(cCuts)); // cCuts originally came from an IntEbmType

   *countCutsInOut = static_cast<IntEbmType>(cCuts);
   return IntEbmType { 0 };
}

// we made this a global because if we had put this variable inside the Booster object, then we would need to dereference that before 
// getting the count.  By making this global we can send a log message incase a bad Booster object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more 
// times than desired, but we can live with that
static int g_cLogGetModelUpdateExpandedParametersMessages = 10;

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION GetModelUpdateExpanded(
   ThreadStateBoostingHandle threadStateBoostingHandle,
   FloatEbmType * modelFeatureGroupUpdateTensor
) {
   LOG_COUNTED_N(
      &g_cLogGetModelUpdateExpandedParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "GetModelUpdateExpanded: "
      "threadStateBoostingHandle=%p, "
      "modelFeatureGroupUpdateTensor=%p",
      static_cast<void *>(threadStateBoostingHandle),
      static_cast<void *>(modelFeatureGroupUpdateTensor)
   );

   ThreadStateBoosting * const pThreadStateBoosting = reinterpret_cast<ThreadStateBoosting *>(threadStateBoostingHandle);
   if(nullptr == pThreadStateBoosting) {
      LOG_0(TraceLevelError, "ERROR GetModelUpdateExpanded threadStateBoosting cannot be nullptr");
      return IntEbmType { 1 };
   }

   const size_t iFeatureGroup = pThreadStateBoosting->GetFeatureGroupIndex();
   if(ThreadStateBoosting::k_illegalFeatureGroupIndex == iFeatureGroup) {
      LOG_0(TraceLevelError, "ERROR GetModelUpdateExpanded bad internal state.  No FeatureGroupIndex set");
      return 1;
   }
   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   EBM_ASSERT(nullptr != pBooster);
   EBM_ASSERT(iFeatureGroup < pBooster->GetCountFeatureGroups());
   EBM_ASSERT(nullptr != pBooster->GetFeatureGroups());

   if(ptrdiff_t { 0 } == pBooster->GetRuntimeLearningTypeOrCountTargetClasses() || ptrdiff_t { 1 } == pBooster->GetRuntimeLearningTypeOrCountTargetClasses()) {
      return IntEbmType { 0 };
   }

   const FeatureGroup * const pFeatureGroup = pBooster->GetFeatureGroups()[iFeatureGroup];
   const size_t cDimensions = pFeatureGroup->GetCountFeatures();
   size_t cValues = GetVectorLength(pBooster->GetRuntimeLearningTypeOrCountTargetClasses());
   if(0 != cDimensions) {
      const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
      const FeatureGroupEntry * const pFeatureGroupEntryEnd = &pFeatureGroupEntry[cDimensions];

      size_t acDivisionIntegersEnd[k_cDimensionsMax];
      size_t * pcDivisionIntegersEnd = acDivisionIntegersEnd;
      do {
         const size_t cBins = pFeatureGroupEntry->m_pFeature->GetCountBins();
         // we've allocated this memory, so it should be reachable, so these numbers should multiply
         EBM_ASSERT(!IsMultiplyError(cBins, cValues));
         cValues *= cBins;
         *pcDivisionIntegersEnd = cBins;
         ++pcDivisionIntegersEnd;
         ++pFeatureGroupEntry;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
      if(pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->Expand(acDivisionIntegersEnd)) {
         return IntEbmType { 1 };
      }
   }
   const FloatEbmType * const pValues = pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->GetValuePointer();
   // we've allocated this memory, so it should be reachable, so these numbers should multiply
   EBM_ASSERT(!IsMultiplyError(sizeof(*pValues), cValues));
   memcpy(modelFeatureGroupUpdateTensor, pValues, sizeof(*pValues) * cValues);
   return IntEbmType { 0 };
}

// we made this a global because if we had put this variable inside the Booster object, then we would need to dereference that before 
// getting the count.  By making this global we can send a log message incase a bad Booster object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more 
// times than desired, but we can live with that
static int g_cLogSetModelUpdateExpandedParametersMessages = 10;

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION SetModelUpdateExpanded(
   ThreadStateBoostingHandle threadStateBoostingHandle,
   IntEbmType indexFeatureGroup,
   FloatEbmType * modelFeatureGroupUpdateTensor
) {
   LOG_COUNTED_N(
      &g_cLogSetModelUpdateExpandedParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "SetModelUpdateExpanded: "
      "threadStateBoostingHandle=%p, "
      "indexFeatureGroup=%" IntEbmTypePrintf ", "
      "modelFeatureGroupUpdateTensor=%p",
      static_cast<void *>(threadStateBoostingHandle),
      indexFeatureGroup,
      static_cast<void *>(modelFeatureGroupUpdateTensor)
   );

   ThreadStateBoosting * const pThreadStateBoosting = reinterpret_cast<ThreadStateBoosting *>(threadStateBoostingHandle);
   if(nullptr == pThreadStateBoosting) {
      LOG_0(TraceLevelError, "ERROR SetModelUpdateExpanded threadStateBoosting cannot be nullptr");
      return IntEbmType { 1 };
   }

   Booster * const pBooster = pThreadStateBoosting->GetBooster();
   EBM_ASSERT(nullptr != pBooster);

   if(indexFeatureGroup < 0) {
      pThreadStateBoosting->SetFeatureGroupIndex(ThreadStateBoosting::k_illegalFeatureGroupIndex);
      LOG_0(TraceLevelError, "ERROR SetModelUpdateExpanded indexFeatureGroup must be positive");
      return IntEbmType { 1 };
   }
   if(!IsNumberConvertable<size_t>(indexFeatureGroup)) {
      pThreadStateBoosting->SetFeatureGroupIndex(ThreadStateBoosting::k_illegalFeatureGroupIndex);
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(TraceLevelError, "ERROR SetModelUpdateExpanded indexFeatureGroup is too high to index");
      return IntEbmType { 1 };
   }
   const size_t iFeatureGroup = static_cast<size_t>(indexFeatureGroup);
   if(pBooster->GetCountFeatureGroups() <= iFeatureGroup) {
      pThreadStateBoosting->SetFeatureGroupIndex(ThreadStateBoosting::k_illegalFeatureGroupIndex);
      LOG_0(TraceLevelError, "ERROR SetModelUpdateExpanded indexFeatureGroup above the number of feature groups that we have");
      return IntEbmType { 1 };
   }
   // this is true because 0 < pBooster->m_cFeatureGroups since our caller needs to pass in a valid indexFeatureGroup to this function
   EBM_ASSERT(nullptr != pBooster->GetFeatureGroups());

   if(ptrdiff_t { 0 } == pBooster->GetRuntimeLearningTypeOrCountTargetClasses() || ptrdiff_t { 1 } == pBooster->GetRuntimeLearningTypeOrCountTargetClasses()) {
      pThreadStateBoosting->SetFeatureGroupIndex(iFeatureGroup);
      return IntEbmType { 0 };
   }

   const FeatureGroup * const pFeatureGroup = pBooster->GetFeatureGroups()[iFeatureGroup];
   const size_t cDimensions = pFeatureGroup->GetCountFeatures();
   size_t cValues = GetVectorLength(pBooster->GetRuntimeLearningTypeOrCountTargetClasses());
   if(0 != cDimensions) {
      const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
      const FeatureGroupEntry * const pFeatureGroupEntryEnd = &pFeatureGroupEntry[cDimensions];

      size_t acDivisionIntegersEnd[k_cDimensionsMax];
      size_t * pcDivisionIntegersEnd = acDivisionIntegersEnd;
      do {
         const size_t cBins = pFeatureGroupEntry->m_pFeature->GetCountBins();
         // we've allocated this memory, so it should be reachable, so these numbers should multiply
         EBM_ASSERT(!IsMultiplyError(cBins, cValues));
         cValues *= cBins;
         *pcDivisionIntegersEnd = cBins;
         ++pcDivisionIntegersEnd;
         ++pFeatureGroupEntry;
      } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);
      if(pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->Expand(acDivisionIntegersEnd)) {
         pThreadStateBoosting->SetFeatureGroupIndex(ThreadStateBoosting::k_illegalFeatureGroupIndex);
         return IntEbmType { 1 };
      }
   }
   FloatEbmType * const pValues = pThreadStateBoosting->GetSmallChangeToModelAccumulatedFromSamplingSets()->GetValuePointer();
   // we've allocated this memory, so it should be reachable, so these numbers should multiply
   EBM_ASSERT(!IsMultiplyError(sizeof(*pValues), cValues));
   memcpy(pValues, modelFeatureGroupUpdateTensor, sizeof(*pValues) * cValues);

   pThreadStateBoosting->SetFeatureGroupIndex(iFeatureGroup);

   return IntEbmType { 0 };
}
