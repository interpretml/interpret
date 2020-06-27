// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"

#include "EbmInternal.h"
// very independent includes
#include "Logging.h" // EBM_ASSERT & LOG

// FeatureCombination.h depends on FeatureInternal.h
#include "FeatureGroup.h"

#include "Booster.h"

void ApplyModelUpdateTraining(
   EbmBoostingState * const pEbmBoostingState,
   const FeatureCombination * const pFeatureCombination,
   const FloatEbmType * const aModelFeatureCombinationUpdateTensor,
   const bool bUseSIMD
);

FloatEbmType ApplyModelUpdateValidation(
   EbmBoostingState * const pEbmBoostingState,
   const FeatureCombination * const pFeatureCombination,
   const FloatEbmType * const aModelFeatureCombinationUpdateTensor,
   const bool bUseSIMD
);

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
static IntEbmType ApplyModelFeatureCombinationUpdateInternal(
   EbmBoostingState * const pEbmBoostingState,
   const size_t iFeatureCombination,
   const FloatEbmType * const aModelFeatureCombinationUpdateTensor,
   FloatEbmType * const pValidationMetricReturn
) {
   LOG_0(TraceLevelVerbose, "Entered ApplyModelFeatureCombinationUpdateInternal");

   // m_apCurrentModel can be null if there are no featureCombinations (but we have an feature combination index), 
   // or if the target has 1 or 0 classes (which we check before calling this function), so it shouldn't be possible to be null
   EBM_ASSERT(nullptr != pEbmBoostingState->GetCurrentModel());
   // m_apCurrentModel can be null if there are no featureCombinations (but we have an feature combination index), 
   // or if the target has 1 or 0 classes (which we check before calling this function), so it shouldn't be possible to be null
   EBM_ASSERT(nullptr != pEbmBoostingState->GetBestModel());
   EBM_ASSERT(nullptr != aModelFeatureCombinationUpdateTensor); // aModelFeatureCombinationUpdateTensor is checked for nullptr before calling this function   

   // our caller can give us one of these bad types of inputs:
   //  1) NaN values
   //  2) +-infinity
   //  3) numbers that are fine, but when added to our existing model overflow to +-infinity
   // Our caller should really just not pass us the first two, but it's hard for our caller to protect against giving us values that won't overflow
   // so we should have some reasonable way to handle them.  If we were meant to overflow, logits or regression values at the maximum/minimum values
   // of doubles should be so close to infinity that it won't matter, and then you can at least graph them without overflowing to special values
   // We have the same problem when we go to make changes to the individual instance updates, but there we can have two graphs that combined push towards
   // an overflow to +-infinity.  We just ignore those overflows, because checking for them would add branches that we don't want, and we can just
   // propagate +-infinity and NaN values to the point where we get a metric and that should cause our client to stop boosting when our metric
   // overlfows and gets converted to the maximum value which will mean the metric won't be changing or improving after that.
   // This is an acceptable compromise.  We protect our models since the user might want to extract them AFTER we overlfow our measurment metric
   // so we don't want to overflow the values to NaN or +-infinity there, and it's very cheap for us to check for overflows when applying the model
   pEbmBoostingState->GetCurrentModel()[iFeatureCombination]->AddExpandedWithBadValueProtection(aModelFeatureCombinationUpdateTensor);

   const FeatureCombination * const pFeatureCombination = pEbmBoostingState->GetFeatureCombinations()[iFeatureCombination];

   if(0 != pEbmBoostingState->GetTrainingSet()->GetCountInstances()) {
      ApplyModelUpdateTraining(
         pEbmBoostingState,
         pFeatureCombination,
         aModelFeatureCombinationUpdateTensor,
         false
      );
   }

   FloatEbmType modelMetric = FloatEbmType { 0 };
   if(0 != pEbmBoostingState->GetValidationSet()->GetCountInstances()) {
      // if there is no validation set, it's pretty hard to know what the metric we'll get for our validation set
      // we could in theory return anything from zero to infinity or possibly, NaN (probably legally the best), but we return 0 here
      // because we want to kick our caller out of any loop it might be calling us in.  Infinity and NaN are odd values that might cause problems in
      // a caller that isn't expecting those values, so 0 is the safest option, and our caller can avoid the situation entirely by not calling
      // us with zero count validation sets

      // if the count of training instances is zero, don't update the best model (it will stay as all zeros), and we don't need to update our 
      // non-existant training set either C++ doesn't define what happens when you compare NaN to annother number.  It probably follows IEEE 754, 
      // but it isn't guaranteed, so let's check for zero instances in the validation set this better way
      // https://stackoverflow.com/questions/31225264/what-is-the-result-of-comparing-a-number-with-nan

      modelMetric = ApplyModelUpdateValidation(
         pEbmBoostingState,
         pFeatureCombination,
         aModelFeatureCombinationUpdateTensor,
         false
      );

      EBM_ASSERT(!std::isnan(modelMetric)); // NaNs can happen, but we should have converted them
      EBM_ASSERT(!std::isinf(modelMetric)); // +infinity can happen, but we should have converted it
      // both log loss and RMSE need to be above zero.  If we got a negative number due to floating point 
      // instability we should have previously converted it to zero.
      EBM_ASSERT(FloatEbmType { 0 } <= modelMetric);

      // modelMetric is either logloss (classification) or mean squared error (mse) (regression).  In either case we want to minimize it.
      if(LIKELY(modelMetric < pEbmBoostingState->GetBestModelMetric())) {
         // we keep on improving, so this is more likely than not, and we'll exit if it becomes negative a lot
         pEbmBoostingState->SetBestModelMetric(modelMetric);

         // TODO : in the future don't copy over all SegmentedTensors.  We only need to copy the ones that changed, which we can detect if we 
         // use a linked list and array lookup for the same data structure
         size_t iModel = 0;
         size_t iModelEnd = pEbmBoostingState->GetCountFeatureCombinations();
         do {
            if(pEbmBoostingState->GetBestModel()[iModel]->Copy(*pEbmBoostingState->GetCurrentModel()[iModel])) {
               if(nullptr != pValidationMetricReturn) {
                  *pValidationMetricReturn = FloatEbmType { 0 }; // on error set it to something instead of random bits
               }
               LOG_0(TraceLevelVerbose, "Exited ApplyModelFeatureCombinationUpdateInternal with memory allocation error in copy");
               return 1;
            }
            ++iModel;
         } while(iModel != iModelEnd);
      }
   }
   if(nullptr != pValidationMetricReturn) {
      *pValidationMetricReturn = modelMetric;
   }

   LOG_0(TraceLevelVerbose, "Exited ApplyModelFeatureCombinationUpdateInternal");
   return 0;
}

// we made this a global because if we had put this variable inside the EbmBoostingState object, then we would need to dereference that before 
// getting the count.  By making this global we can send a log message incase a bad EbmBoostingState object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more 
// times than desired, but we can live with that
static unsigned int g_cLogApplyModelFeatureCombinationUpdateParametersMessages = 10;

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION ApplyModelFeatureCombinationUpdate(
   PEbmBoosting ebmBoosting,
   IntEbmType indexFeatureCombination,
   const FloatEbmType * modelFeatureCombinationUpdateTensor,
   FloatEbmType * validationMetricReturn
) {
   LOG_COUNTED_N(
      &g_cLogApplyModelFeatureCombinationUpdateParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "ApplyModelFeatureCombinationUpdate parameters: ebmBoosting=%p, indexFeatureCombination=%" IntEbmTypePrintf
      ", modelFeatureCombinationUpdateTensor=%p, validationMetricReturn=%p",
      static_cast<void *>(ebmBoosting),
      indexFeatureCombination,
      static_cast<const void *>(modelFeatureCombinationUpdateTensor),
      static_cast<void *>(validationMetricReturn)
   );

   EbmBoostingState * pEbmBoostingState = reinterpret_cast<EbmBoostingState *>(ebmBoosting);
   if(nullptr == pEbmBoostingState) {
      if(LIKELY(nullptr != validationMetricReturn)) {
         *validationMetricReturn = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR ApplyModelFeatureCombinationUpdate ebmBoosting cannot be nullptr");
      return 1;
   }
   if(indexFeatureCombination < 0) {
      if(LIKELY(nullptr != validationMetricReturn)) {
         *validationMetricReturn = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR ApplyModelFeatureCombinationUpdate indexFeatureCombination must be positive");
      return 1;
   }
   if(!IsNumberConvertable<size_t, IntEbmType>(indexFeatureCombination)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      if(LIKELY(nullptr != validationMetricReturn)) {
         *validationMetricReturn = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR ApplyModelFeatureCombinationUpdate indexFeatureCombination is too high to index");
      return 1;
   }
   size_t iFeatureCombination = static_cast<size_t>(indexFeatureCombination);
   if(pEbmBoostingState->GetCountFeatureCombinations() <= iFeatureCombination) {
      if(LIKELY(nullptr != validationMetricReturn)) {
         *validationMetricReturn = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR ApplyModelFeatureCombinationUpdate indexFeatureCombination above the number of feature groups that we have");
      return 1;
   }
   // this is true because 0 < pEbmBoostingState->m_cFeatureCombinations since our caller needs to pass in a valid indexFeatureCombination to this function
   EBM_ASSERT(nullptr != pEbmBoostingState->GetFeatureCombinations());

   LOG_COUNTED_0(
      pEbmBoostingState->GetFeatureCombinations()[iFeatureCombination]->GetPointerCountLogEnterApplyModelFeatureCombinationUpdateMessages(),
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered ApplyModelFeatureCombinationUpdate"
   );
   if(nullptr == modelFeatureCombinationUpdateTensor) {
      // modelFeatureCombinationUpdateTensor can be nullptr (then nothing gets updated).  This could happen for
      // if there was only 1 class, meaning we would be 100% confident in the outcome and no tensor would be retunred
      // since we can eliminate one class, and if there's only 1 class then we eliminate all logits
      if(nullptr != validationMetricReturn) {
         *validationMetricReturn = FloatEbmType { 0 };
      }
      LOG_COUNTED_0(
         pEbmBoostingState->GetFeatureCombinations()[iFeatureCombination]->GetPointerCountLogExitApplyModelFeatureCombinationUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited ApplyModelFeatureCombinationUpdate from null modelFeatureCombinationUpdateTensor"
      );
      return 0;
   }

   // TODO: check if GetRuntimeLearningTypeOrCountTargetClasses can be zero?
   if(ptrdiff_t { 0 } == pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses() || ptrdiff_t { 1 } == pEbmBoostingState->GetRuntimeLearningTypeOrCountTargetClasses()) {
      // if there is only 1 target class for classification, then we can predict the output with 100% accuracy.  The model is a tensor with zero 
      // length array logits, which means for our representation that we have zero items in the array total.
      // since we can predit the output with 100% accuracy, our log loss is 0.
      if(nullptr != validationMetricReturn) {
         *validationMetricReturn = 0;
      }
      LOG_COUNTED_0(
         pEbmBoostingState->GetFeatureCombinations()[iFeatureCombination]->GetPointerCountLogExitApplyModelFeatureCombinationUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited ApplyModelFeatureCombinationUpdate from runtimeLearningTypeOrCountTargetClasses <= 1"
      );
      return 0;
   }

   IntEbmType ret = ApplyModelFeatureCombinationUpdateInternal(
      pEbmBoostingState,
      iFeatureCombination,
      modelFeatureCombinationUpdateTensor,
      validationMetricReturn
   );
   if(0 != ret) {
      LOG_N(TraceLevelWarning, "WARNING ApplyModelFeatureCombinationUpdate returned %" IntEbmTypePrintf, ret);
   }

   if(nullptr != validationMetricReturn) {
      EBM_ASSERT(!std::isnan(*validationMetricReturn)); // NaNs can happen, but we should have edited those before here
      EBM_ASSERT(!std::isinf(*validationMetricReturn)); // infinities can happen, but we should have edited those before here
      // both log loss and RMSE need to be above zero.  We previously zero any values below zero, which can happen due to floating point instability.
      EBM_ASSERT(FloatEbmType { 0 } <= *validationMetricReturn);
      LOG_COUNTED_N(
         pEbmBoostingState->GetFeatureCombinations()[iFeatureCombination]->GetPointerCountLogExitApplyModelFeatureCombinationUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited ApplyModelFeatureCombinationUpdate %" FloatEbmTypePrintf, *validationMetricReturn
      );
   } else {
      LOG_COUNTED_0(
         pEbmBoostingState->GetFeatureCombinations()[iFeatureCombination]->GetPointerCountLogExitApplyModelFeatureCombinationUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited ApplyModelFeatureCombinationUpdate.  No validation pointer."
      );
   }
   return ret;
}
