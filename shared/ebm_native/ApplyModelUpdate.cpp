// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

// FeatureGroup.hpp depends on FeatureInternal.h
#include "FeatureGroup.hpp"

#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern void ApplyTermUpdateTraining(
   BoosterShell * const pBoosterShell,
   const Term * const pTerm
);

extern double ApplyTermUpdateValidation(
   BoosterShell * const pBoosterShell,
   const Term * const pTerm
);

static ErrorEbmType ApplyTermUpdateInternal(
   BoosterShell * const pBoosterShell,
   double * const pValidationMetricReturn
) {
   LOG_0(TraceLevelVerbose, "Entered ApplyTermUpdateInternal");

   ErrorEbmType error;

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const size_t iTerm = pBoosterShell->GetTermIndex();
   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];

   error = pBoosterShell->GetTermUpdate()->Expand(pTerm);
   if(Error_None != error) {
      if(nullptr != pValidationMetricReturn) {
         *pValidationMetricReturn = double { 0 };
      }
      return error;
   }

   // m_apCurrentTermTensors can be null if there are no terms (but we have an feature group index), 
   // or if the target has 1 or 0 classes (which we check before calling this function), so it shouldn't be possible to be null
   EBM_ASSERT(nullptr != pBoosterCore->GetCurrentModel());
   // m_apCurrentTermTensors can be null if there are no terms (but we have an feature group index), 
   // or if the target has 1 or 0 classes (which we check before calling this function), so it shouldn't be possible to be null
   EBM_ASSERT(nullptr != pBoosterCore->GetBestModel());

   const FloatFast * const aUpdateScores = pBoosterShell->GetTermUpdate()->GetTensorScoresPointer();

   // our caller can give us one of these bad types of inputs:
   //  1) NaN values
   //  2) +-infinity
   //  3) numbers that are fine, but when added to our existing term scores overflow to +-infinity
   // Our caller should really just not pass us the first two, but it's hard for our caller to protect against giving us values that won't overflow
   // so we should have some reasonable way to handle them.  If we were meant to overflow, logits or regression values at the maximum/minimum values
   // of doubles should be so close to infinity that it won't matter, and then you can at least graph them without overflowing to special values
   // We have the same problem when we go to make changes to the individual sample updates, but there we can have two graphs that combined push towards
   // an overflow to +-infinity.  We just ignore those overflows, because checking for them would add branches that we don't want, and we can just
   // propagate +-infinity and NaN values to the point where we get a metric and that should cause our client to stop boosting when our metric
   // overlfows and gets converted to the maximum value which will mean the metric won't be changing or improving after that.
   // This is an acceptable compromise.  We protect our term scores since the user might want to extract them AFTER we overlfow our measurment metric
   // so we don't want to overflow the values to NaN or +-infinity there, and it's very cheap for us to check for overflows when applying the term score updates
   pBoosterCore->GetCurrentModel()[iTerm]->AddExpandedWithBadValueProtection(aUpdateScores);

   if(0 != pBoosterCore->GetTrainingSet()->GetCountSamples()) {
      ApplyTermUpdateTraining(pBoosterShell, pTerm);
   }

   double modelMetric = 0.0;
   if(0 != pBoosterCore->GetValidationSet()->GetCountSamples()) {
      // if there is no validation set, it's pretty hard to know what the metric we'll get for our validation set
      // we could in theory return anything from zero to infinity or possibly, NaN (probably legally the best), but we return 0 here
      // because we want to kick our caller out of any loop it might be calling us in.  Infinity and NaN are odd values that might cause problems in
      // a caller that isn't expecting those values, so 0 is the safest option, and our caller can avoid the situation entirely by not calling
      // us with zero count validation sets

      // if the count of training samples is zero, don't update the best term scores (it will stay as all zeros), and we don't need to update our 
      // non-existant training set either C++ doesn't define what happens when you compare NaN to annother number.  It probably follows IEEE 754, 
      // but it isn't guaranteed, so let's check for zero samples in the validation set this better way
      // https://stackoverflow.com/questions/31225264/what-is-the-result-of-comparing-a-number-with-nan

      modelMetric = ApplyTermUpdateValidation(pBoosterShell, pTerm);

      EBM_ASSERT(!std::isnan(modelMetric)); // NaNs can happen, but we should have converted them
      EBM_ASSERT(!std::isinf(modelMetric)); // +infinity can happen, but we should have converted it
      // both log loss and RMSE need to be above zero.  If we got a negative number due to floating point 
      // instability we should have previously converted it to zero.
      EBM_ASSERT(0.0 <= modelMetric);

      // modelMetric is either logloss (classification) or mean squared error (mse) (regression).  In either case we want to minimize it.
      if(LIKELY(modelMetric < pBoosterCore->GetBestModelMetric())) {
         // we keep on improving, so this is more likely than not, and we'll exit if it becomes negative a lot
         pBoosterCore->SetBestModelMetric(modelMetric);

         // TODO : in the future don't copy over all Tensors.  We only need to copy the ones that changed, which we can detect if we 
         // use a linked list and array lookup for the same data structure
         size_t iTermCopy = 0;
         size_t iTermCopyEnd = pBoosterCore->GetCountTerms();
         do {
            error = pBoosterCore->GetBestModel()[iTermCopy]->Copy(*pBoosterCore->GetCurrentModel()[iTermCopy]);
            if(Error_None != error) {
               if(nullptr != pValidationMetricReturn) {
                  *pValidationMetricReturn = double { 0 };
               }
               LOG_0(TraceLevelVerbose, "Exited ApplyTermUpdateInternal with memory allocation error in copy");
               return error;
            }
            ++iTermCopy;
         } while(iTermCopy != iTermCopyEnd);
      }
   }
   if(nullptr != pValidationMetricReturn) {
      *pValidationMetricReturn = modelMetric;
   }

   LOG_0(TraceLevelVerbose, "Exited ApplyTermUpdateInternal");
   return Error_None;
}

// we made this a global because if we had put this variable inside the BoosterCore object, then we would need to dereference that before 
// getting the count.  By making this global we can send a log message incase a bad BoosterCore object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more 
// times than desired, but we can live with that
static int g_cLogApplyTermUpdateParametersMessages = 10;

// TODO: validationMetricOut should be an average
EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION ApplyTermUpdate(
   BoosterHandle boosterHandle,
   double * validationMetricOut
) {
   LOG_COUNTED_N(
      &g_cLogApplyTermUpdateParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "ApplyTermUpdate: "
      "boosterHandle=%p, "
      "validationMetricOut=%p"
      ,
      static_cast<void *>(boosterHandle),
      static_cast<void *>(validationMetricOut)
   );

   ErrorEbmType error;

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      if(LIKELY(nullptr != validationMetricOut)) {
         *validationMetricOut = 0.0;
      }
      // already logged
      return Error_IllegalParamValue;
   }

   const size_t iTerm = pBoosterShell->GetTermIndex();
   if(BoosterShell::k_illegalTermIndex == iTerm) {
      if(LIKELY(nullptr != validationMetricOut)) {
         *validationMetricOut = 0.0;
      }
      LOG_0(TraceLevelError, "ERROR ApplyTermUpdate bad internal state.  No Term index set");
      return Error_IllegalParamValue;
   }
   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(nullptr != pBoosterCore);
   EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());

   LOG_COUNTED_0(
      pBoosterCore->GetTerms()[iTerm]->GetPointerCountLogEnterApplyTermUpdateMessages(),
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered ApplyTermUpdate"
   );

   if(ptrdiff_t { 0 } == pBoosterCore->GetCountClasses() || ptrdiff_t { 1 } == pBoosterCore->GetCountClasses()) {
      // if there is only 1 target class for classification, then we can predict the output with 100% accuracy.  The term scores are a tensor with zero 
      // length array logits, which means for our representation that we have zero items in the array total.
      // since we can predit the output with 100% accuracy, our log loss is 0.
      if(nullptr != validationMetricOut) {
         *validationMetricOut = 0.0;
      }
      pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);
      LOG_COUNTED_0(
         pBoosterCore->GetTerms()[iTerm]->GetPointerCountLogExitApplyTermUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited ApplyTermUpdate. cClasses <= 1"
      );
      return Error_None;
   }

   error = ApplyTermUpdateInternal(
      pBoosterShell,
      validationMetricOut
   );

   pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);

   if(Error_None != error) {
      LOG_N(TraceLevelWarning, "WARNING ApplyTermUpdate: return=%" ErrorEbmTypePrintf, error);
   }

   if(nullptr != validationMetricOut) {
      EBM_ASSERT(!std::isnan(*validationMetricOut)); // NaNs can happen, but we should have edited those before here
      EBM_ASSERT(!std::isinf(*validationMetricOut)); // infinities can happen, but we should have edited those before here
      // both log loss and RMSE need to be above zero.  We previously zero any values below zero, which can happen due to floating point instability.
      EBM_ASSERT(0.0 <= *validationMetricOut);
      LOG_COUNTED_N(
         pBoosterCore->GetTerms()[iTerm]->GetPointerCountLogExitApplyTermUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited ApplyTermUpdate: "
         "*validationMetricOut=%le"
         , 
         *validationMetricOut
      );
   } else {
      LOG_COUNTED_0(
         pBoosterCore->GetTerms()[iTerm]->GetPointerCountLogExitApplyTermUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited ApplyTermUpdate"
      );
   }
   return error;
}

// we made this a global because if we had put this variable inside the BoosterCore object, then we would need to dereference that before 
// getting the count.  By making this global we can send a log message incase a bad BoosterCore object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more 
// times than desired, but we can live with that
static int g_cLogGetTermUpdateSplitsParametersMessages = 10;

EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION GetTermUpdateSplits(
   BoosterHandle boosterHandle,
   IntEbmType indexDimension,
   IntEbmType * countSplitsInOut,
   IntEbmType * splitIndexesOut
) {
   LOG_COUNTED_N(
      &g_cLogGetTermUpdateSplitsParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "GetTermUpdateSplits: "
      "boosterHandle=%p, "
      "indexDimension=%" IntEbmTypePrintf ", "
      "countSplitsInOut=%p"
      "splitIndexesOut=%p"
      ,
      static_cast<void *>(boosterHandle),
      indexDimension, 
      static_cast<void *>(countSplitsInOut),
      static_cast<void *>(splitIndexesOut)
   );

   if(nullptr == countSplitsInOut) {
      LOG_0(TraceLevelError, "ERROR GetTermUpdateSplits countSplitsInOut cannot be nullptr");
      return Error_IllegalParamValue;
   }

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      *countSplitsInOut = IntEbmType { 0 };
      // already logged
      return Error_IllegalParamValue;
   }

   const size_t iTerm = pBoosterShell->GetTermIndex();
   if(BoosterShell::k_illegalTermIndex == iTerm) {
      *countSplitsInOut = IntEbmType { 0 };
      LOG_0(TraceLevelError, "ERROR GetTermUpdateSplits bad internal state.  No Term index set");
      return Error_IllegalParamValue;
   }
   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(nullptr != pBoosterCore);
   EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());
   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];

   if(indexDimension < 0) {
      *countSplitsInOut = IntEbmType { 0 };
      LOG_0(TraceLevelError, "ERROR GetTermUpdateSplits indexDimension must be positive");
      return Error_IllegalParamValue;
   }
   if(static_cast<IntEbmType>(pTerm->GetCountDimensions()) <= indexDimension) {
      *countSplitsInOut = IntEbmType { 0 };
      LOG_0(TraceLevelError, "ERROR GetTermUpdateSplits indexDimension above the number of dimensions that we have");
      return Error_IllegalParamValue;
   }
   const size_t iDimension = static_cast<size_t>(indexDimension);

   const size_t cBins = pTerm->GetTermEntries()[iDimension].m_pFeature->GetCountBins();
   // cBins started from IntEbmType, so we should be able to convert back safely
   if(*countSplitsInOut != static_cast<IntEbmType>(cBins - size_t { 1 })) {
      *countSplitsInOut = IntEbmType { 0 };
      LOG_0(TraceLevelError, "ERROR GetTermUpdateSplits bad split array length");
      return Error_IllegalParamValue;
   }

   const size_t cSplits = pBoosterShell->GetTermUpdate()->GetCountSplits(iDimension);
   EBM_ASSERT(cSplits < cBins);
   if(0 != cSplits) {
      if(nullptr == splitIndexesOut) {
         *countSplitsInOut = IntEbmType { 0 };
         LOG_0(TraceLevelError, "ERROR GetTermUpdateSplits splitIndexesOut cannot be nullptr");
         return Error_IllegalParamValue;
      }

      const ActiveDataType * pSplitIndexesFrom = pBoosterShell->GetTermUpdate()->GetSplitPointer(iDimension);
      IntEbmType * pSplitIndexesTo = splitIndexesOut;
      IntEbmType * pSplitIndexesToEnd = splitIndexesOut + cSplits;
      do {
         const ActiveDataType indexSplit = *pSplitIndexesFrom;
         ++pSplitIndexesFrom;

         EBM_ASSERT(!IsConvertError<IntEbmType>(indexSplit)); // the total count works so the index should too
         *pSplitIndexesTo = static_cast<IntEbmType>(indexSplit);

         ++pSplitIndexesTo;
      } while(pSplitIndexesToEnd != pSplitIndexesTo);
   }

   EBM_ASSERT(!IsConvertError<IntEbmType>(cSplits)); // cSplits originally came from an IntEbmType

   *countSplitsInOut = static_cast<IntEbmType>(cSplits);
   return Error_None;
}

// we made this a global because if we had put this variable inside the BoosterCore object, then we would need to dereference that before 
// getting the count.  By making this global we can send a log message incase a bad BoosterCore object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more 
// times than desired, but we can live with that
static int g_cLogGetTermUpdateParametersMessages = 10;

EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION GetTermUpdate(
   BoosterHandle boosterHandle,
   double * updateScoresTensorOut
) {
   LOG_COUNTED_N(
      &g_cLogGetTermUpdateParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "GetTermUpdate: "
      "boosterHandle=%p, "
      "updateScoresTensorOut=%p"
      ,
      static_cast<void *>(boosterHandle),
      static_cast<void *>(updateScoresTensorOut)
   );

   ErrorEbmType error;

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamValue;
   }

   const size_t iTerm = pBoosterShell->GetTermIndex();
   if(BoosterShell::k_illegalTermIndex == iTerm) {
      LOG_0(TraceLevelError, "ERROR GetTermUpdate bad internal state.  No Term index set");
      return Error_IllegalParamValue; // technically we're in an illegal state, but why split hairs
   }
   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(nullptr != pBoosterCore);
   EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());

   if(ptrdiff_t { 0 } == pBoosterCore->GetCountClasses() || ptrdiff_t { 1 } == pBoosterCore->GetCountClasses()) {
      return Error_None;
   }

   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];
   error = pBoosterShell->GetTermUpdate()->Expand(pTerm);
   if(Error_None != error) {
      return error;
   }

   const size_t cDimensions = pTerm->GetCountDimensions();
   size_t cTensorScores = GetCountScores(pBoosterCore->GetCountClasses());
   if(0 != cDimensions) {
      const TermEntry * pTermEntry = pTerm->GetTermEntries();
      const TermEntry * const pTermEntriesEnd = &pTermEntry[cDimensions];
      do {
         const size_t cBins = pTermEntry->m_pFeature->GetCountBins();
         // we've allocated this memory, so it should be reachable, so these numbers should multiply
         EBM_ASSERT(!IsMultiplyError(cTensorScores, cBins));
         cTensorScores *= cBins;
         ++pTermEntry;
      } while(pTermEntriesEnd != pTermEntry);
   }
   const FloatFast * const aUpdateScores = pBoosterShell->GetTermUpdate()->GetTensorScoresPointer();
   // we've allocated this memory, so it should be reachable, so these numbers should multiply
   EBM_ASSERT(!IsMultiplyError(sizeof(*updateScoresTensorOut), cTensorScores));
   EBM_ASSERT(!IsMultiplyError(sizeof(*aUpdateScores), cTensorScores));
   static_assert(sizeof(*updateScoresTensorOut) == sizeof(*aUpdateScores), "float mismatch");
   memcpy(updateScoresTensorOut, aUpdateScores, sizeof(*aUpdateScores) * cTensorScores);
   return Error_None;
}

// we made this a global because if we had put this variable inside the BoosterCore object, then we would need to dereference that before 
// getting the count.  By making this global we can send a log message incase a bad BoosterCore object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more 
// times than desired, but we can live with that
static int g_cLogSetTermUpdateParametersMessages = 10;

EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION SetTermUpdate(
   BoosterHandle boosterHandle,
   IntEbmType indexTerm,
   double * updateScoresTensor
) {
   LOG_COUNTED_N(
      &g_cLogSetTermUpdateParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "SetTermUpdate: "
      "boosterHandle=%p, "
      "indexTerm=%" IntEbmTypePrintf ", "
      "updateScoresTensor=%p"
      ,
      static_cast<void *>(boosterHandle),
      indexTerm,
      static_cast<void *>(updateScoresTensor)
   );

   ErrorEbmType error;

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamValue;
   }

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(nullptr != pBoosterCore);

   if(indexTerm < 0) {
      pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);
      LOG_0(TraceLevelError, "ERROR SetTermUpdate indexTerm must be positive");
      return Error_IllegalParamValue;
   }
   if(IsConvertError<size_t>(indexTerm)) {
      pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(TraceLevelError, "ERROR SetTermUpdate indexTerm is too high to index");
      return Error_IllegalParamValue;
   }
   const size_t iTerm = static_cast<size_t>(indexTerm);
   if(pBoosterCore->GetCountTerms() <= iTerm) {
      pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);
      LOG_0(TraceLevelError, "ERROR SetTermUpdate indexTerm above the number of feature groups that we have");
      return Error_IllegalParamValue;
   }
   // pBoosterCore->GetTerms() can be null if 0 == pBoosterCore->m_cTerms, but we checked that condition above
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());

   if(ptrdiff_t { 0 } == pBoosterCore->GetCountClasses() || ptrdiff_t { 1 } == pBoosterCore->GetCountClasses()) {
      pBoosterShell->SetTermIndex(iTerm);
      return Error_None;
   }

   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];
   error = pBoosterShell->GetTermUpdate()->Expand(pTerm);
   if(Error_None != error) {
      // already logged
      pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);
      return error;
   }

   const size_t cDimensions = pTerm->GetCountDimensions();
   const size_t cScores = GetCountScores(pBoosterCore->GetCountClasses());
   size_t cTensorScores = cScores;
   if(0 != cDimensions) {
      const TermEntry * pTermEntry = pTerm->GetTermEntries();
      const TermEntry * const pTermEntriesEnd = &pTermEntry[cDimensions];
      do {
         const size_t cBins = pTermEntry->m_pFeature->GetCountBins();
         // we've allocated this memory, so it should be reachable, so these numbers should multiply
         EBM_ASSERT(!IsMultiplyError(cTensorScores, cBins));
         cTensorScores *= cBins;
         ++pTermEntry;
      } while(pTermEntriesEnd != pTermEntry);
   }
   FloatFast * const aUpdateScores = pBoosterShell->GetTermUpdate()->GetTensorScoresPointer();
   EBM_ASSERT(!IsMultiplyError(sizeof(*aUpdateScores), cTensorScores));
   EBM_ASSERT(!IsMultiplyError(sizeof(*updateScoresTensor), cTensorScores));
   static_assert(sizeof(*updateScoresTensor) == sizeof(*aUpdateScores), "float mismatch");
   memcpy(aUpdateScores, updateScoresTensor, sizeof(*aUpdateScores) * cTensorScores);

#ifdef ZERO_FIRST_MULTICLASS_LOGIT

   if(2 <= cScores) {
      FloatFast * pUpdateScore = aUpdateScores;
      const FloatFast * const pExteriorEnd = pUpdateScore + cTensorScores;
      do {
         FloatFast shiftScore = pUpdateScore[0];
         const FloatFast * const pInteriorEnd = pUpdateScore + cScores;
         do {
            *pUpdateScore -= shiftScore;
            ++pUpdateScore;
         } while(pInteriorEnd != pUpdateScore);
      } while(pExteriorEnd != pUpdateScore);
   }

#endif // ZERO_FIRST_MULTICLASS_LOGIT

   pBoosterShell->SetTermIndex(iTerm);

   return Error_None;
}

} // DEFINED_ZONE_NAME
