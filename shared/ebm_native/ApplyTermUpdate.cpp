// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "ebm_native.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "zones.h"

#include "Feature.hpp"
#include "Term.hpp"
#include "Tensor.hpp"
#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern ErrorEbm ApplyUpdate(ApplyUpdateBridge * const pData);

// we made this a global because if we had put this variable inside the BoosterCore object, then we would need to dereference that before 
// getting the count.  By making this global we can send a log message incase a bad BoosterCore object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more 
// times than desired, but we can live with that
static int g_cLogApplyTermUpdate = 10;

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION ApplyTermUpdate(
   BoosterHandle boosterHandle,
   double * avgValidationMetricOut
) {
   ErrorEbm error;

   LOG_COUNTED_N(
      &g_cLogApplyTermUpdate,
      Trace_Info,
      Trace_Verbose,
      "ApplyTermUpdate: "
      "boosterHandle=%p, "
      "avgValidationMetricOut=%p"
      ,
      static_cast<void *>(boosterHandle),
      static_cast<void *>(avgValidationMetricOut)
   );

   if(LIKELY(nullptr != avgValidationMetricOut)) {
      // returning +inf means that boosting won't consider this to be an improvement.  After a few cycles
      // it should exit with the last model that was good if the error was ignored (it shouldn't be ignored though)
      *avgValidationMetricOut = std::numeric_limits<double>::infinity();
   }

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamVal;
   }

   const size_t iTerm = pBoosterShell->GetTermIndex();
   if(BoosterShell::k_illegalTermIndex == iTerm) {
      LOG_0(Trace_Error, "ERROR ApplyTermUpdate bad internal state.  No Term index set");
      return Error_IllegalParamVal;
   }

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(nullptr != pBoosterCore);
   EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());

   pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);

   Term * const pTerm = pBoosterCore->GetTerms()[iTerm];

   LOG_COUNTED_0(
      pTerm->GetPointerCountLogEnterApplyTermUpdateMessages(),
      Trace_Info,
      Trace_Verbose,
      "Entered ApplyTermUpdate"
   );

   if(ptrdiff_t { 0 } == pBoosterCore->GetCountClasses() || ptrdiff_t { 1 } == pBoosterCore->GetCountClasses()) {
      EBM_ASSERT(nullptr == pBoosterShell->GetTermUpdate());
      EBM_ASSERT(nullptr == pBoosterCore->GetCurrentModel());

      // if there are 0 classes, then there must be zero samples, but our caller can still specify 0 != cBins below
      EBM_ASSERT(ptrdiff_t { 0 } != pBoosterCore->GetCountClasses() || 0 == pBoosterCore->GetTrainingSet()->GetCountSamples());
      EBM_ASSERT(ptrdiff_t { 0 } != pBoosterCore->GetCountClasses() || 0 == pBoosterCore->GetValidationSet()->GetCountSamples());

      // if there is only 1 target class for classification, then we can predict the output with 100% accuracy.  
      // The term scores are a tensor with zero length array logits, which means for our representation that we 
      // have zero items in the array total. Since we can predit the output with 100% accuracy, our log loss is 0.
      if(nullptr != avgValidationMetricOut) {
         *avgValidationMetricOut = 0.0;
      }
      LOG_COUNTED_0(
         pTerm->GetPointerCountLogExitApplyTermUpdateMessages(),
         Trace_Info,
         Trace_Verbose,
         "Exited ApplyTermUpdate. cClasses <= 1"
      );
      return Error_None;
   }
   EBM_ASSERT(nullptr != pBoosterShell->GetTermUpdate());
   EBM_ASSERT(nullptr != pBoosterCore->GetCurrentModel());

   if(size_t { 0 } == pTerm->GetCountTensorBins()) {
      EBM_ASSERT(nullptr == pBoosterCore->GetCurrentModel()[iTerm]);
      
      // if GetCountTensorBins is 0, then pBoosterShell->GetTermUpdate() does not contain valid data

      // if we have zero samples and one of the dimensions has 0 bins then there is no tensor, so return now
      if(nullptr != avgValidationMetricOut) {
         *avgValidationMetricOut = 0.0;
      }
      LOG_COUNTED_0(
         pTerm->GetPointerCountLogExitApplyTermUpdateMessages(),
         Trace_Info,
         Trace_Verbose,
         "Exited ApplyTermUpdate. dimension with a feature that has 0 bins"
      );
      return Error_None;
   }
   EBM_ASSERT(nullptr != pBoosterCore->GetCurrentModel()[iTerm]);

   error = pBoosterShell->GetTermUpdate()->Expand(pTerm);
   if(Error_None != error) {
      return error;
   }

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
      ApplyUpdateBridge data;
      data.m_cClasses = pBoosterCore->GetCountClasses();
      data.m_cPack = pTerm->GetTermBitPack();
      data.m_bCalcMetric = false;
      data.m_aMulticlassMidwayTemp = pBoosterShell->GetMulticlassMidwayTemp();
      data.m_aUpdateTensorScores = pBoosterShell->GetTermUpdate()->GetTensorScoresPointer();
      data.m_cSamples = pBoosterCore->GetTrainingSet()->GetCountSamples();
      data.m_aPacked = pBoosterCore->GetTrainingSet()->GetInputDataPointer(iTerm);
      data.m_aTargets = pBoosterCore->GetTrainingSet()->GetTargetDataPointer();
      data.m_aWeights = nullptr;
      data.m_aSampleScores = pBoosterCore->GetTrainingSet()->GetSampleScores();
      data.m_aGradientsAndHessians = pBoosterCore->GetTrainingSet()->GetGradientsAndHessiansPointer();
      error = ApplyUpdate(&data);
      if(Error_None != error) {
         return error;
      }
   }

   double validationMetricAvg = 0.0;
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

      ApplyUpdateBridge data;
      data.m_cClasses = pBoosterCore->GetCountClasses();
      data.m_cPack = pTerm->GetTermBitPack();
      data.m_bCalcMetric = true;
      data.m_aMulticlassMidwayTemp = pBoosterShell->GetMulticlassMidwayTemp();
      data.m_aUpdateTensorScores = pBoosterShell->GetTermUpdate()->GetTensorScoresPointer();
      data.m_cSamples = pBoosterCore->GetValidationSet()->GetCountSamples();
      data.m_aPacked = pBoosterCore->GetValidationSet()->GetInputDataPointer(iTerm);
      data.m_aTargets = pBoosterCore->GetValidationSet()->GetTargetDataPointer();
      data.m_aWeights = pBoosterCore->GetValidationWeights();
      data.m_aSampleScores = pBoosterCore->GetValidationSet()->GetSampleScores();
      data.m_aGradientsAndHessians = pBoosterCore->GetValidationSet()->GetGradientsAndHessiansPointer();
      error = ApplyUpdate(&data);
      if(Error_None != error) {
         return error;
      }
      validationMetricAvg = data.m_metricOut;

      EBM_ASSERT(!std::isnan(validationMetricAvg)); // NaNs can happen, but we should have cleaned them up
      EBM_ASSERT(0.0 <= validationMetricAvg); // negatives can happen, but we should have cleaned them up

      const double totalWeight = static_cast<double>(pBoosterCore->GetValidationWeightTotal());
      EBM_ASSERT(!std::isnan(totalWeight));
      EBM_ASSERT(!std::isinf(totalWeight));
      EBM_ASSERT(0 < totalWeight);
      validationMetricAvg /= totalWeight; // if totalWeight < 1.0 then this can overflow to +inf

      EBM_ASSERT(!std::isnan(validationMetricAvg)); // NaNs can happen, but we should have cleaned them up
      EBM_ASSERT(0.0 <= validationMetricAvg); // negatives can happen, but we should have cleaned them up

      // validationMetricAvg is either logloss (classification) or mean squared error (mse) (regression).  
      // In either case we want to minimize it.
      if(LIKELY(validationMetricAvg < pBoosterCore->GetBestModelMetric())) {
         // we keep on improving, so this is more likely than not, and we'll exit if it becomes negative a lot
         pBoosterCore->SetBestModelMetric(validationMetricAvg);

         // TODO : in the future don't copy over all Tensors.  We only need to copy the ones that changed, which we can detect if we 
         // use a linked list and array lookup for the same data structure
         size_t iTermCopy = 0;
         size_t iTermCopyEnd = pBoosterCore->GetCountTerms();
         do {
            if(nullptr != pBoosterCore->GetCurrentModel()[iTermCopy]) {
               EBM_ASSERT(nullptr != pBoosterCore->GetBestModel()[iTermCopy]);
               error = pBoosterCore->GetBestModel()[iTermCopy]->Copy(*pBoosterCore->GetCurrentModel()[iTermCopy]);
               if(Error_None != error) {
                  LOG_0(Trace_Verbose, "Exited ApplyTermUpdateInternal with memory allocation error in copy");
                  return error;
               }
            } else {
               EBM_ASSERT(nullptr == pBoosterCore->GetBestModel()[iTermCopy]);
            }
            ++iTermCopy;
         } while(iTermCopy != iTermCopyEnd);
      }
   }
   
   if(nullptr != avgValidationMetricOut) {
      *avgValidationMetricOut = validationMetricAvg;
   }

   LOG_COUNTED_N(
      pTerm->GetPointerCountLogExitApplyTermUpdateMessages(),
      Trace_Info,
      Trace_Verbose,
      "Exited ApplyTermUpdate: "
      "validationMetricAvg=%le"
      , 
      validationMetricAvg
   );

   return Error_None;
}

// we made this a global because if we had put this variable inside the BoosterCore object, then we would need to dereference that before 
// getting the count.  By making this global we can send a log message incase a bad BoosterCore object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more 
// times than desired, but we can live with that
static int g_cLogGetTermUpdateSplits = 10;

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION GetTermUpdateSplits(
   BoosterHandle boosterHandle,
   IntEbm indexDimension,
   IntEbm * countSplitsInOut,
   IntEbm * splitIndexesOut
) {
   LOG_COUNTED_N(
      &g_cLogGetTermUpdateSplits,
      Trace_Info,
      Trace_Verbose,
      "GetTermUpdateSplits: "
      "boosterHandle=%p, "
      "indexDimension=%" IntEbmPrintf ", "
      "countSplitsInOut=%p"
      "splitIndexesOut=%p"
      ,
      static_cast<void *>(boosterHandle),
      indexDimension, 
      static_cast<void *>(countSplitsInOut),
      static_cast<void *>(splitIndexesOut)
   );

   if(nullptr == countSplitsInOut) {
      LOG_0(Trace_Error, "ERROR GetTermUpdateSplits countSplitsInOut cannot be nullptr");
      return Error_IllegalParamVal;
   }

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      *countSplitsInOut = IntEbm { 0 };
      // already logged
      return Error_IllegalParamVal;
   }

   const size_t iTerm = pBoosterShell->GetTermIndex();
   if(BoosterShell::k_illegalTermIndex == iTerm) {
      *countSplitsInOut = IntEbm { 0 };
      LOG_0(Trace_Error, "ERROR GetTermUpdateSplits bad internal state.  No Term index set");
      return Error_IllegalParamVal;
   }
   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(nullptr != pBoosterCore);
   EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());
   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];

   if(indexDimension < 0) {
      *countSplitsInOut = IntEbm { 0 };
      LOG_0(Trace_Error, "ERROR GetTermUpdateSplits indexDimension must be positive");
      return Error_IllegalParamVal;
   }
   if(static_cast<IntEbm>(pTerm->GetCountDimensions()) <= indexDimension) {
      *countSplitsInOut = IntEbm { 0 };
      LOG_0(Trace_Error, "ERROR GetTermUpdateSplits indexDimension above the number of dimensions that we have");
      return Error_IllegalParamVal;
   }
   const size_t iDimension = static_cast<size_t>(indexDimension);

   size_t cBins = pTerm->GetFeatures()[iDimension]->GetCountBins();
   cBins = size_t { 0 } == cBins ? size_t { 1 } : cBins; // for our purposes here, 0 bins means 0 splits

   // cBins started from IntEbm, so we should be able to convert back safely
   if(*countSplitsInOut != static_cast<IntEbm>(cBins - size_t { 1 })) {
      *countSplitsInOut = IntEbm { 0 };
      LOG_0(Trace_Error, "ERROR GetTermUpdateSplits bad split array length");
      return Error_IllegalParamVal;
   }

   if(ptrdiff_t { 0 } == pBoosterCore->GetCountClasses() || ptrdiff_t { 1 } == pBoosterCore->GetCountClasses()) {
      EBM_ASSERT(nullptr == pBoosterShell->GetTermUpdate());

      // if there are 0 classes, then there must be zero samples, but our caller can still specify 0 != cBins below
      EBM_ASSERT(ptrdiff_t { 0 } != pBoosterCore->GetCountClasses() || 0 == pBoosterCore->GetTrainingSet()->GetCountSamples());
      EBM_ASSERT(ptrdiff_t { 0 } != pBoosterCore->GetCountClasses() || 0 == pBoosterCore->GetValidationSet()->GetCountSamples());

      // if we have 0 or 1 classes then there is no tensor, so return now
      *countSplitsInOut = 0;
      LOG_0(Trace_Warning, "ERROR GetTermUpdateSplits ptrdiff_t { 0 } == pBoosterCore->GetCountClasses() || ptrdiff_t { 1 } == pBoosterCore->GetCountClasses()");
      return Error_None;
   }
   EBM_ASSERT(nullptr != pBoosterShell->GetTermUpdate());

   if(size_t { 0 } == pTerm->GetCountTensorBins()) {
      // if we have zero samples and one of the dimensions has 0 bins then there is no tensor, so return now

      // if GetCountTensorBins is 0, then pBoosterShell->GetTermUpdate() does not contain valid data

      *countSplitsInOut = 0;
      LOG_0(Trace_Warning, "ERROR GetTermUpdateSplits size_t { 0 } == pTerm->GetCountTensorBins()");
      return Error_None;
   }

   const size_t cSplits = pBoosterShell->GetTermUpdate()->GetCountSplits(iDimension);
   EBM_ASSERT(cSplits < cBins);
   if(0 != cSplits) {
      if(nullptr == splitIndexesOut) {
         *countSplitsInOut = IntEbm { 0 };
         LOG_0(Trace_Error, "ERROR GetTermUpdateSplits splitIndexesOut cannot be nullptr");
         return Error_IllegalParamVal;
      }

      const ActiveDataType * pSplitIndexesFrom = pBoosterShell->GetTermUpdate()->GetSplitPointer(iDimension);
      IntEbm * pSplitIndexesTo = splitIndexesOut;
      IntEbm * pSplitIndexesToEnd = splitIndexesOut + cSplits;
      do {
         const ActiveDataType indexSplit = *pSplitIndexesFrom;
         ++pSplitIndexesFrom;

         EBM_ASSERT(!IsConvertError<IntEbm>(indexSplit)); // the total count works so the index should too
         *pSplitIndexesTo = static_cast<IntEbm>(indexSplit);

         ++pSplitIndexesTo;
      } while(pSplitIndexesToEnd != pSplitIndexesTo);
   }
   EBM_ASSERT(!IsConvertError<IntEbm>(cSplits)); // cSplits originally came from an IntEbm
   *countSplitsInOut = static_cast<IntEbm>(cSplits);
   return Error_None;
}

// we made this a global because if we had put this variable inside the BoosterCore object, then we would need to dereference that before 
// getting the count.  By making this global we can send a log message incase a bad BoosterCore object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more 
// times than desired, but we can live with that
static int g_cLogGetTermUpdate = 10;

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION GetTermUpdate(
   BoosterHandle boosterHandle,
   double * updateScoresTensorOut
) {
   LOG_COUNTED_N(
      &g_cLogGetTermUpdate,
      Trace_Info,
      Trace_Verbose,
      "GetTermUpdate: "
      "boosterHandle=%p, "
      "updateScoresTensorOut=%p"
      ,
      static_cast<void *>(boosterHandle),
      static_cast<void *>(updateScoresTensorOut)
   );

   ErrorEbm error;

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamVal;
   }

   const size_t iTerm = pBoosterShell->GetTermIndex();
   if(BoosterShell::k_illegalTermIndex == iTerm) {
      LOG_0(Trace_Error, "ERROR GetTermUpdate bad internal state.  No Term index set");
      return Error_IllegalParamVal; // technically we're in an illegal state, but why split hairs
   }
   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(nullptr != pBoosterCore);
   EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());

   if(ptrdiff_t { 0 } == pBoosterCore->GetCountClasses() || ptrdiff_t { 1 } == pBoosterCore->GetCountClasses()) {
      EBM_ASSERT(nullptr == pBoosterShell->GetTermUpdate());

      // if there are 0 classes, then there must be zero samples, but our caller can still specify 0 != cBins below
      EBM_ASSERT(ptrdiff_t { 0 } != pBoosterCore->GetCountClasses() || 0 == pBoosterCore->GetTrainingSet()->GetCountSamples());
      EBM_ASSERT(ptrdiff_t { 0 } != pBoosterCore->GetCountClasses() || 0 == pBoosterCore->GetValidationSet()->GetCountSamples());

      return Error_None;
   }
   EBM_ASSERT(nullptr != pBoosterShell->GetTermUpdate());

   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];

   size_t cTensorScores = pTerm->GetCountTensorBins();
   if(size_t { 0 } == cTensorScores) {
      // if we have zero samples and one of the dimensions has 0 bins then there is no tensor, so return now

      // if GetCountTensorBins is 0, then pBoosterShell->GetTermUpdate() does not contain valid data

      return Error_None;
   }

   error = pBoosterShell->GetTermUpdate()->Expand(pTerm);
   if(Error_None != error) {
      return error;
   }

   EBM_ASSERT(!IsMultiplyError(cTensorScores, GetCountScores(pBoosterCore->GetCountClasses())));
   cTensorScores *= GetCountScores(pBoosterCore->GetCountClasses());

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
static int g_cLogSetTermUpdate = 10;

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION SetTermUpdate(
   BoosterHandle boosterHandle,
   IntEbm indexTerm,
   const double * updateScoresTensor
) {
   LOG_COUNTED_N(
      &g_cLogSetTermUpdate,
      Trace_Info,
      Trace_Verbose,
      "SetTermUpdate: "
      "boosterHandle=%p, "
      "indexTerm=%" IntEbmPrintf ", "
      "updateScoresTensor=%p"
      ,
      static_cast<void *>(boosterHandle),
      indexTerm,
      static_cast<const void *>(updateScoresTensor)
   );

   ErrorEbm error;

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamVal;
   }

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(nullptr != pBoosterCore);

   if(indexTerm < 0) {
      pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);
      LOG_0(Trace_Error, "ERROR SetTermUpdate indexTerm must be positive");
      return Error_IllegalParamVal;
   }
   if(IsConvertError<size_t>(indexTerm)) {
      pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);
      // we wouldn't have allowed the creation of an feature set larger than size_t
      LOG_0(Trace_Error, "ERROR SetTermUpdate indexTerm is too high to index");
      return Error_IllegalParamVal;
   }
   const size_t iTerm = static_cast<size_t>(indexTerm);
   if(pBoosterCore->GetCountTerms() <= iTerm) {
      pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);
      LOG_0(Trace_Error, "ERROR SetTermUpdate indexTerm above the number of feature groups that we have");
      return Error_IllegalParamVal;
   }

   if(ptrdiff_t { 0 } == pBoosterCore->GetCountClasses() || ptrdiff_t { 1 } == pBoosterCore->GetCountClasses()) {
      EBM_ASSERT(nullptr == pBoosterShell->GetTermUpdate());

      // if there are 0 classes, then there must be zero samples, but our caller can still specify 0 != cBins below
      EBM_ASSERT(ptrdiff_t { 0 } != pBoosterCore->GetCountClasses() || 0 == pBoosterCore->GetTrainingSet()->GetCountSamples());
      EBM_ASSERT(ptrdiff_t { 0 } != pBoosterCore->GetCountClasses() || 0 == pBoosterCore->GetValidationSet()->GetCountSamples());

      pBoosterShell->SetTermIndex(iTerm);
      return Error_None;
   }
   EBM_ASSERT(nullptr != pBoosterShell->GetTermUpdate());

   // pBoosterCore->GetTerms() can be null if 0 == pBoosterCore->m_cTerms, but we checked that condition above
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());
   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];

   size_t cTensorScores = pTerm->GetCountTensorBins();
   if(size_t { 0 } == cTensorScores) {
      // if we have zero samples and one of the dimensions has 0 bins then there is no tensor, so return now

      // if GetCountTensorBins is 0, then we leave pBoosterShell->GetTermUpdate() with invalid data since
      // out Tensor class does not support tensors of zero elements

      pBoosterShell->SetTermIndex(iTerm);
      return Error_None;
   }

   pBoosterShell->GetTermUpdate()->SetCountDimensions(pTerm->GetCountDimensions());
   pBoosterShell->GetTermUpdate()->Reset();

   error = pBoosterShell->GetTermUpdate()->Expand(pTerm);
   if(Error_None != error) {
      // already logged
      pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);
      return error;
   }

   EBM_ASSERT(!IsMultiplyError(cTensorScores, GetCountScores(pBoosterCore->GetCountClasses())));
   cTensorScores *= GetCountScores(pBoosterCore->GetCountClasses());

   FloatFast * const aUpdateScores = pBoosterShell->GetTermUpdate()->GetTensorScoresPointer();
   EBM_ASSERT(!IsMultiplyError(sizeof(*aUpdateScores), cTensorScores));
   EBM_ASSERT(!IsMultiplyError(sizeof(*updateScoresTensor), cTensorScores));
   static_assert(sizeof(*updateScoresTensor) == sizeof(*aUpdateScores), "float mismatch");
   memcpy(aUpdateScores, updateScoresTensor, sizeof(*aUpdateScores) * cTensorScores);

   pBoosterShell->SetTermIndex(iTerm);

   return Error_None;
}

} // DEFINED_ZONE_NAME
