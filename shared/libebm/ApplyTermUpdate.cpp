// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT

#define ZONE_main
#include "zones.h"

#include "Feature.hpp"
#include "Term.hpp"
#include "Transpose.hpp"
#include "Tensor.hpp"
#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

// we made this a global because if we had put this variable inside the BoosterCore object, then we would need to
// dereference that before getting the count.  By making this global we can send a log message incase a bad BoosterCore
// object is sent into us we only decrease the count if the count is non-zero, so at worst if there is a race condition
// then we'll output this log message more times than desired, but we can live with that
static int g_cLogApplyTermUpdate = 10;

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION ApplyTermUpdate(
      BoosterHandle boosterHandle, double* avgValidationMetricOut) {
   ErrorEbm error;

   LOG_COUNTED_N(&g_cLogApplyTermUpdate,
         Trace_Info,
         Trace_Verbose,
         "ApplyTermUpdate: "
         "boosterHandle=%p, "
         "avgValidationMetricOut=%p",
         static_cast<void*>(boosterHandle),
         static_cast<void*>(avgValidationMetricOut));

   if(LIKELY(nullptr != avgValidationMetricOut)) {
      // returning +inf means that boosting won't consider this to be an improvement.  After a few cycles
      // it should exit with the last model that was good if the error was ignored (it shouldn't be ignored though)
      *avgValidationMetricOut = std::numeric_limits<double>::infinity();
   }

   BoosterShell* const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamVal;
   }

   const size_t iTerm = pBoosterShell->GetTermIndex();
   if(BoosterShell::k_illegalTermIndex == iTerm) {
      LOG_0(Trace_Error, "ERROR ApplyTermUpdate bad internal state.  No Term index set");
      return Error_IllegalParamVal;
   }

   BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(nullptr != pBoosterCore);

   pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);

   FloatScore* aUpdateScores;

   Term* pTerm;
   size_t cTensorBins;
   if(BoosterShell::k_interceptTermIndex == iTerm) {
      LOG_0(Trace_Info, "Entered ApplyTermUpdate");

      if(size_t{0} == pBoosterCore->GetCountScores()) {
         // if there is only 1 target class for classification, then we can predict the output with 100% accuracy.
         // The term scores are a tensor with zero length array logits, which means for our representation that we
         // have zero items in the array total. Since we can predit the output with 100% accuracy, our log loss is 0.
         // Leave the avgValidationMetricOut value as +inf though to avoid special casing here without calling the
         // metric.
         LOG_0(Trace_Info, "Exited ApplyTermUpdate. cClasses <= 1");
         return Error_None;
      }

      EBM_ASSERT(nullptr != pBoosterShell->GetTermUpdate());
      aUpdateScores = pBoosterShell->GetTermUpdate()->GetTensorScoresPointer();
      EBM_ASSERT(nullptr != aUpdateScores);

      pTerm = nullptr;
      cTensorBins = 1;
   } else {
      EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
      EBM_ASSERT(nullptr != pBoosterCore->GetTerms());

      pTerm = pBoosterCore->GetTerms()[iTerm];

      LOG_COUNTED_0(pTerm->GetPointerCountLogEnterApplyTermUpdateMessages(),
            Trace_Info,
            Trace_Verbose,
            "Entered ApplyTermUpdate");

      if(size_t{0} == pBoosterCore->GetCountScores()) {
         // if there is only 1 target class for classification, then we can predict the output with 100% accuracy.
         // The term scores are a tensor with zero length array logits, which means for our representation that we
         // have zero items in the array total. Since we can predit the output with 100% accuracy, our log loss is 0.
         // Leave the avgValidationMetricOut value as +inf though to avoid special casing here without calling the
         // metric.
         LOG_COUNTED_0(pTerm->GetPointerCountLogExitApplyTermUpdateMessages(),
               Trace_Info,
               Trace_Verbose,
               "Exited ApplyTermUpdate. cClasses <= 1");
         return Error_None;
      }

      cTensorBins = pTerm->GetCountTensorBins();
      if(size_t{0} == cTensorBins) {
         LOG_COUNTED_0(pTerm->GetPointerCountLogExitApplyTermUpdateMessages(),
               Trace_Info,
               Trace_Verbose,
               "Exited ApplyTermUpdate. dimension with a feature that has 0 bins");
         return Error_None;
      }

      EBM_ASSERT(nullptr != pBoosterCore->GetCurrentModel());
      EBM_ASSERT(nullptr != pBoosterCore->GetBestModel());
      EBM_ASSERT(nullptr != pBoosterCore->GetCurrentModel()[iTerm]);
      EBM_ASSERT(nullptr != pBoosterCore->GetBestModel()[iTerm]);

      error = pBoosterShell->GetTermUpdate()->Expand(pTerm);
      if(Error_None != error) {
         return error;
      }

      EBM_ASSERT(nullptr != pBoosterShell->GetTermUpdate());
      aUpdateScores = pBoosterShell->GetTermUpdate()->GetTensorScoresPointer();
      EBM_ASSERT(nullptr != aUpdateScores);

      // our caller can give us one of these bad types of inputs:
      //  1) NaN values
      //  2) +-infinity
      //  3) numbers that are fine, but when added to our existing term scores overflow to +-infinity
      // Our caller should really just not pass us the first two, but it's hard for our caller to protect against giving
      // us values that won't overflow so we should have some reasonable way to handle them.  If we were meant to
      // overflow, logits or regression values at the maximum/minimum values of doubles should be so close to infinity
      // that it won't matter, and then you can at least graph them without overflowing to special values We have the
      // same problem when we go to make changes to the individual sample updates, but there we can have two graphs that
      // combined push towards an overflow to +-infinity.  We just ignore those overflows, because checking for them
      // would add branches that we don't want, and we can just propagate +-infinity and NaN values to the point where
      // we get a metric and that should cause our client to stop boosting when our metric overlfows and gets converted
      // to the maximum value which will mean the metric won't be changing or improving after that. This is an
      // acceptable compromise.  We protect our term scores since the user might want to extract them AFTER we overlfow
      // our measurment metric so we don't want to overflow the values to NaN or +-infinity there, and it's very cheap
      // for us to check for overflows when applying the term score updates
      pBoosterCore->GetCurrentModel()[iTerm]->AddExpandedWithBadValueProtection(aUpdateScores);
   }

   double validationMetricAvg = 0.0;

   static_assert(std::is_same<FloatBig, FloatScore>::value || std::is_same<FloatSmall, FloatScore>::value,
         "FloatScore must be either FloatBig or FloatSmall");
   size_t cFloatSize = sizeof(aUpdateScores[0]);
   bool bIgnored = false;
   while(true) {
      if(0 != pBoosterCore->GetTrainingSet()->GetCountSamples()) {
         EBM_ASSERT(1 <= pBoosterCore->GetTrainingSet()->GetCountSubsets());

         DataSubsetBoosting* pSubset = pBoosterCore->GetTrainingSet()->GetSubsets();
         EBM_ASSERT(nullptr != pSubset);
         const DataSubsetBoosting* const pSubsetsEnd = pSubset + pBoosterCore->GetTrainingSet()->GetCountSubsets();
         do {
            if(pSubset->GetObjectiveWrapper()->m_cFloatBytes != cFloatSize) {
               bIgnored = true;
            } else {
               ApplyUpdateBridge data;

               data.m_cPack = k_cItemsPerBitPackUndefined;
               data.m_aPacked = nullptr;
               if(BoosterShell::k_interceptTermIndex != iTerm) {
                  data.m_aPacked = pSubset->GetTermData(iTerm);
                  if(0 != pTerm->GetBitsRequiredMin()) {
                     data.m_cPack = GetCountItemsBitPacked(
                           pTerm->GetBitsRequiredMin(), pSubset->GetObjectiveWrapper()->m_cUIntBytes);
                  }
               }

               data.m_cScores = pBoosterCore->GetCountScores();
               data.m_bHessianNeeded = pBoosterCore->IsHessian() ? EBM_TRUE : EBM_FALSE;
               data.m_bUseApprox = pBoosterCore->IsUseApprox();
               data.m_bValidation = EBM_FALSE;
               data.m_aMulticlassMidwayTemp = pBoosterShell->GetMulticlassMidwayTemp();
               data.m_aUpdateTensorScores = aUpdateScores;
               data.m_cSamples = pSubset->GetCountSamples();
               data.m_aTargets = pSubset->GetTargetData();
               data.m_aWeights = nullptr;
               data.m_aSampleScores = pSubset->GetSampleScores();
               data.m_aGradientsAndHessians = pSubset->GetGradHess();
               data.m_metricOut = 0.0;
               error = pSubset->ObjectiveApplyUpdate(&data);
               if(Error_None != error) {
                  return error;
               }
            }
            ++pSubset;
         } while(pSubsetsEnd != pSubset);
      }

      if(0 != pBoosterCore->GetValidationSet()->GetCountSamples()) {
         EBM_ASSERT(1 <= pBoosterCore->GetValidationSet()->GetCountSubsets());

         DataSubsetBoosting* pSubset = pBoosterCore->GetValidationSet()->GetSubsets();
         EBM_ASSERT(nullptr != pSubset);
         const DataSubsetBoosting* const pSubsetsEnd = pSubset + pBoosterCore->GetValidationSet()->GetCountSubsets();
         do {
            if(pSubset->GetObjectiveWrapper()->m_cFloatBytes != cFloatSize) {
               bIgnored = true;
            } else {
               // if there is no validation set, it's pretty hard to know what the metric we'll get for our validation
               // set we could in theory return anything from zero to infinity or possibly, NaN (probably legally the
               // best), but we return 0 here because we want to kick our caller out of any loop it might be calling us
               // in.  Infinity and NaN are odd values that might cause problems in a caller that isn't expecting those
               // values, so 0 is the safest option, and our caller can avoid the situation entirely by not calling us
               // with zero count validation sets

               // if the count of training samples is zero, don't update the best term scores (it will stay as all
               // zeros), and we don't need to update our non-existant training set either C++ doesn't define what
               // happens when you compare NaN to annother number.  It probably follows IEEE 754, but it isn't
               // guaranteed, so let's check for zero samples in the validation set this better way
               // https://stackoverflow.com/questions/31225264/what-is-the-result-of-comparing-a-number-with-nan

               ApplyUpdateBridge data;

               data.m_cPack = k_cItemsPerBitPackUndefined;
               data.m_aPacked = nullptr;
               if(BoosterShell::k_interceptTermIndex != iTerm) {
                  data.m_aPacked = pSubset->GetTermData(iTerm);
                  if(0 != pTerm->GetBitsRequiredMin()) {
                     data.m_cPack = GetCountItemsBitPacked(
                           pTerm->GetBitsRequiredMin(), pSubset->GetObjectiveWrapper()->m_cUIntBytes);
                  }
               }

               data.m_cScores = pBoosterCore->GetCountScores();
               // for the validation set we're calculating the metric and updating the scores, but we don't use
               // the gradients, except for the special case of RMSE where the gradients are also the error
               data.m_bHessianNeeded = EBM_FALSE;
               data.m_bUseApprox = pBoosterCore->IsUseApprox();
               data.m_bValidation = EBM_TRUE;
               data.m_aMulticlassMidwayTemp = pBoosterShell->GetMulticlassMidwayTemp();
               data.m_aUpdateTensorScores = aUpdateScores;
               data.m_cSamples = pSubset->GetCountSamples();
               data.m_aTargets = pSubset->GetTargetData();
               data.m_aWeights = pSubset->GetSubsetInnerBag(0)->GetWeights();
               data.m_aSampleScores = pSubset->GetSampleScores();
               data.m_aGradientsAndHessians = pSubset->GetGradHess();
               data.m_metricOut = 0.0;
               error = pSubset->ObjectiveApplyUpdate(&data);
               if(Error_None != error) {
                  return error;
               }
               validationMetricAvg += data.m_metricOut;
            }
            ++pSubset;
         } while(pSubsetsEnd != pSubset);
      }
      if(!bIgnored) {
         break;
      }
      // we support having our updates as float64 with float64 or float32 compute zone values
      // or we support having our updates as float32 with float32 compute zone values
      // but we do not support having float32 updates with float64 compute zone values
      EBM_ASSERT(sizeof(FloatBig) == sizeof(FloatScore));
      if(sizeof(FloatSmall) == cFloatSize) {
         break;
      }

      EBM_ASSERT(sizeof(FloatBig) == cFloatSize);
      static_assert(sizeof(FloatSmall) < sizeof(FloatBig),
            "we reuse the memory below and since we overwrite memory the FloatBig needs to be larger or equal to "
            "FloatSmall, but also we use the size to differentiate so they can't be equal either");

      cFloatSize = sizeof(FloatSmall);

      // these need to be void * to avoid breaking the C++ aliasing rules
      void* pUpdateSmall = aUpdateScores;
      void* pUpdateBig = aUpdateScores;
      const void* const pUpdateBigEnd = IndexByte(
            reinterpret_cast<void*>(aUpdateScores), sizeof(FloatBig) * pBoosterCore->GetCountScores() * cTensorBins);
      do {
         *reinterpret_cast<FloatSmall*>(pUpdateSmall) =
               static_cast<FloatSmall>(*reinterpret_cast<FloatBig*>(pUpdateBig));
         pUpdateBig = IndexByte(pUpdateBig, sizeof(FloatBig));
         pUpdateSmall = IndexByte(pUpdateSmall, sizeof(FloatSmall));
      } while(pUpdateBigEnd != pUpdateBig);
   }

   if(0 != pBoosterCore->GetValidationSet()->GetCountSamples()) {
      validationMetricAvg = pBoosterCore->FinishMetric(validationMetricAvg);

      if(EBM_FALSE != pBoosterCore->MaximizeMetric()) {
         // make it so that we always return values such that the caller wants to minimize them. If the caller
         // wants more information they can determine if they should negate the values we return them.
         validationMetricAvg = -validationMetricAvg;
      }

      EBM_ASSERT(!std::isnan(validationMetricAvg)); // NaNs can happen, but we should have cleaned them up

      // The validation set has no inner bags, and index 0 always contains the weight of the complete validation set
      const double totalWeight = pBoosterCore->GetValidationSet()->GetBagWeightTotal(0);
      EBM_ASSERT(!std::isnan(totalWeight));
      EBM_ASSERT(!std::isinf(totalWeight));
      EBM_ASSERT(0.0 < totalWeight);
      validationMetricAvg /= totalWeight; // if totalWeight < 1.0 then this can overflow to +inf

      EBM_ASSERT(!std::isnan(validationMetricAvg)); // NaNs can happen, but we should have cleaned them up
   }

   if(LIKELY(validationMetricAvg <= pBoosterCore->GetBestModelMetric())) {
      pBoosterCore->SetBestModelMetric(validationMetricAvg);

      if(BoosterShell::k_interceptTermIndex != iTerm) {
         // TODO: We're doing a lot more work here than necessary.  Typically in the early phases we improve
         // on each boosting step, and in that case we should only need to copy over the term's tensor that
         // we just improved on since all the other ones are up to date.  Later though we'll get into a stage
         // where some of the terms will improve on the metric but others won't. At that point if we see
         // two terms not improve the stopping metric, then we see one that does, we'd need to copy over the
         // last 3 terms to maintain consistency.  That requires that we keep track of the terms we boosted
         // on since the last improvement.  This can get even more interesting if we do more than a full boosting
         // round where a term might have been boosted on a few times.  In that case we only need to overwrite
         // it once.  We can do this by keeping a set that holds the terms that have been bosted on since the last
         // improvement and then we would overwrite only those whenever we see an improvement. Instead of a set though
         // we could instead maintan a reversed linked list of terms that we've boosted on using a flat array
         // with 1 pointer entry for each term.  If a term is already in the linked list there is no need to add it
         // again.  This way we can avoid a sweep of the entire list of terms on each boosting round.
         // Or just use an array of bools (1 per term) and sweep the array of bools each time. Since it's co-located
         // doing the sweep will be fast and resetting to zero is easy with memset

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

   if(BoosterShell::k_interceptTermIndex == iTerm) {
      LOG_N(Trace_Info, "Exited ApplyTermUpdate: validationMetricAvg=%le", validationMetricAvg);
   } else {
      LOG_COUNTED_N(pTerm->GetPointerCountLogExitApplyTermUpdateMessages(),
            Trace_Info,
            Trace_Verbose,
            "Exited ApplyTermUpdate: "
            "validationMetricAvg=%le",
            validationMetricAvg);
   }

   return Error_None;
}

// we made this a global because if we had put this variable inside the BoosterCore object, then we would need to
// dereference that before getting the count.  By making this global we can send a log message incase a bad BoosterCore
// object is sent into us we only decrease the count if the count is non-zero, so at worst if there is a race condition
// then we'll output this log message more times than desired, but we can live with that
static int g_cLogGetTermUpdateSplits = 10;

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION GetTermUpdateSplits(
      BoosterHandle boosterHandle, IntEbm indexDimension, IntEbm* countSplitsInOut, IntEbm* splitsOut) {
   LOG_COUNTED_N(&g_cLogGetTermUpdateSplits,
         Trace_Info,
         Trace_Verbose,
         "GetTermUpdateSplits: "
         "boosterHandle=%p, "
         "indexDimension=%" IntEbmPrintf ", "
         "countSplitsInOut=%p, "
         "splitsOut=%p",
         static_cast<void*>(boosterHandle),
         indexDimension,
         static_cast<void*>(countSplitsInOut),
         static_cast<void*>(splitsOut));

   if(nullptr == countSplitsInOut) {
      LOG_0(Trace_Error, "ERROR GetTermUpdateSplits countSplitsInOut cannot be nullptr");
      return Error_IllegalParamVal;
   }

   BoosterShell* const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      *countSplitsInOut = IntEbm{0};
      // already logged
      return Error_IllegalParamVal;
   }

   const size_t iTerm = pBoosterShell->GetTermIndex();
   if(BoosterShell::k_illegalTermIndex == iTerm) {
      *countSplitsInOut = IntEbm{0};
      LOG_0(Trace_Error, "ERROR GetTermUpdateSplits bad internal state.  No Term index set");
      return Error_IllegalParamVal;
   }

   if(BoosterShell::k_interceptTermIndex == iTerm) {
      const IntEbm countSplits = *countSplitsInOut;
      *countSplitsInOut = IntEbm{0};

      if(IntEbm{0} != countSplits) {
         LOG_0(Trace_Error, "ERROR GetTermUpdateSplits bad split array length");
         return Error_IllegalParamVal;
      }

      LOG_0(Trace_Warning, "WARNING GetTermUpdateSplits BoosterShell::k_interceptTermIndex == iTerm");
      return Error_None;
   }

   BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(nullptr != pBoosterCore);
   EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());
   const Term* const pTerm = pBoosterCore->GetTerms()[iTerm];

   if(indexDimension < 0) {
      *countSplitsInOut = IntEbm{0};
      LOG_0(Trace_Error, "ERROR GetTermUpdateSplits indexDimension must be positive");
      return Error_IllegalParamVal;
   }
   if(static_cast<IntEbm>(pTerm->GetCountDimensions()) <= indexDimension) {
      *countSplitsInOut = IntEbm{0};
      LOG_0(Trace_Error, "ERROR GetTermUpdateSplits indexDimension above the number of dimensions that we have");
      return Error_IllegalParamVal;
   }
   const size_t iDimension = static_cast<size_t>(indexDimension);

   size_t cBins = pTerm->GetTermFeatures()[iDimension].m_pFeature->GetCountBins();
   const bool bMissing = pTerm->GetTermFeatures()[iDimension].m_pFeature->IsMissing();
   const bool bUnseen = pTerm->GetTermFeatures()[iDimension].m_pFeature->IsUnseen();
   cBins += bMissing ? size_t{0} : size_t{1};
   cBins += bUnseen ? size_t{0} : size_t{1};
   cBins = size_t{0} == cBins ? size_t{1} : cBins; // for our purposes here, 0 bins means 0 splits

   // cBins started from IntEbm, so we should be able to convert back safely
   if(*countSplitsInOut != static_cast<IntEbm>(cBins - size_t{1})) {
      *countSplitsInOut = IntEbm{0};
      LOG_0(Trace_Error, "ERROR GetTermUpdateSplits bad split array length");
      return Error_IllegalParamVal;
   }

   if(size_t{0} == pBoosterCore->GetCountScores()) {
      // if we have 0 or 1 classes then there is no tensor, so return now
      *countSplitsInOut = 0;
      LOG_0(Trace_Warning, "WARNING GetTermUpdateSplits size_t { 0 } == pBoosterCore->GetCountScores()");
      return Error_None;
   }

   if(size_t{0} == pTerm->GetCountTensorBins()) {
      // if we have zero samples and one of the dimensions has 0 bins then there is no tensor, so return now

      // if GetCountTensorBins is 0, then pBoosterShell->GetTermUpdate() does not contain valid data

      *countSplitsInOut = 0;
      LOG_0(Trace_Warning, "WARNING GetTermUpdateSplits size_t { 0 } == pTerm->GetCountTensorBins()");
      return Error_None;
   }

   EBM_ASSERT(nullptr != pBoosterShell->GetTermUpdate());
   const size_t cSplits = pBoosterShell->GetTermUpdate()->GetCountSlices(iDimension) - 1;
   EBM_ASSERT(cSplits < cBins);
   if(0 != cSplits) {
      if(nullptr == splitsOut) {
         *countSplitsInOut = IntEbm{0};
         LOG_0(Trace_Error, "ERROR GetTermUpdateSplits splitsOut cannot be nullptr");
         return Error_IllegalParamVal;
      }

      const UIntSplit indexEdgeAdd = bMissing ? UIntSplit{0} : UIntSplit{1};

      const UIntSplit* pFrom = pBoosterShell->GetTermUpdate()->GetSplitPointer(iDimension);
      IntEbm* pTo = splitsOut;
      IntEbm* pToEnd = splitsOut + cSplits;
      do {
         // if the missing bin was eliminated, we need to increment our split indexes
         const UIntSplit indexEdge = *pFrom + indexEdgeAdd;
         ++pFrom;

         // the caller passed us a countBins value as an IntEbm, and the UIntSplit value should be equal or less
         EBM_ASSERT(!IsConvertError<IntEbm>(indexEdge));
         *pTo = static_cast<IntEbm>(indexEdge);

         ++pTo;
      } while(pToEnd != pTo);
   }
   EBM_ASSERT(!IsConvertError<IntEbm>(cSplits)); // cSplits originally came from an IntEbm
   *countSplitsInOut = static_cast<IntEbm>(cSplits);
   return Error_None;
}

// we made this a global because if we had put this variable inside the BoosterCore object, then we would need to
// dereference that before getting the count.  By making this global we can send a log message incase a bad BoosterCore
// object is sent into us we only decrease the count if the count is non-zero, so at worst if there is a race condition
// then we'll output this log message more times than desired, but we can live with that
static int g_cLogGetTermUpdate = 10;

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION GetTermUpdate(BoosterHandle boosterHandle, double* updateScoresTensorOut) {
   LOG_COUNTED_N(&g_cLogGetTermUpdate,
         Trace_Info,
         Trace_Verbose,
         "GetTermUpdate: "
         "boosterHandle=%p, "
         "updateScoresTensorOut=%p",
         static_cast<void*>(boosterHandle),
         static_cast<void*>(updateScoresTensorOut));

   ErrorEbm error;

   BoosterShell* const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamVal;
   }

   const size_t iTerm = pBoosterShell->GetTermIndex();
   if(BoosterShell::k_illegalTermIndex == iTerm) {
      LOG_0(Trace_Error, "ERROR GetTermUpdate bad internal state.  No Term index set");
      return Error_IllegalParamVal; // technically we're in an illegal state, but why split hairs
   }
   BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(nullptr != pBoosterCore);

   if(size_t{0} == pBoosterCore->GetCountScores()) {
      return Error_None;
   }

   const Term* pTerm = nullptr;
   size_t cTensorScores = 1;
   if(BoosterShell::k_interceptTermIndex != iTerm) {
      EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
      EBM_ASSERT(nullptr != pBoosterCore->GetTerms());
      pTerm = pBoosterCore->GetTerms()[iTerm];

      cTensorScores = pTerm->GetCountTensorBins();
      if(size_t{0} == cTensorScores) {
         // If we have zero samples and one of the dimensions has 0 bins then there is no tensor, so return now
         // In theory it might be better to zero out the caller's tensor cells (2 ^ n_dimensions), but this condition
         // is almost an error already, so don't try reading/writing memory. We just define this situation as
         // having a zero sized tensor result. The caller can zero their own memory if they want it zero

         // if GetCountTensorBins is 0, then pBoosterShell->GetTermUpdate() does not contain valid data

         LOG_0(Trace_Warning, "WARNING GetTermUpdate size_t { 0 } == cTensorScores");
         return Error_None;
      }

      EBM_ASSERT(nullptr != pBoosterShell->GetTermUpdate());
      error = pBoosterShell->GetTermUpdate()->Expand(pTerm);
      if(Error_None != error) {
         return error;
      }
   }

   EBM_ASSERT(nullptr != pBoosterShell->GetTermUpdate());
   FloatScore* const aUpdateScores = pBoosterShell->GetTermUpdate()->GetTensorScoresPointer();
   EBM_ASSERT(nullptr != aUpdateScores);
   Transpose<true>(pTerm, pBoosterCore->GetCountScores(), updateScoresTensorOut, aUpdateScores);

   return Error_None;
}

// we made this a global because if we had put this variable inside the BoosterCore object, then we would need to
// dereference that before getting the count.  By making this global we can send a log message incase a bad BoosterCore
// object is sent into us we only decrease the count if the count is non-zero, so at worst if there is a race condition
// then we'll output this log message more times than desired, but we can live with that
static int g_cLogSetTermUpdate = 10;

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION SetTermUpdate(
      BoosterHandle boosterHandle, IntEbm indexTerm, const double* updateScoresTensor) {
   LOG_COUNTED_N(&g_cLogSetTermUpdate,
         Trace_Info,
         Trace_Verbose,
         "SetTermUpdate: "
         "boosterHandle=%p, "
         "indexTerm=%" IntEbmPrintf ", "
         "updateScoresTensor=%p",
         static_cast<void*>(boosterHandle),
         indexTerm,
         static_cast<const void*>(updateScoresTensor));

   ErrorEbm error;

   BoosterShell* const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamVal;
   }

   BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(nullptr != pBoosterCore);

   size_t iTerm;
   Term* pTerm;
   size_t cDimensions;
   if(indexTerm < IntEbm{0}) {
      iTerm = BoosterShell::k_interceptTermIndex;
      pTerm = nullptr;
      cDimensions = 0;
      if(indexTerm != IntEbm{-1}) {
         LOG_0(Trace_Error, "ERROR SetTermUpdate indexTerm must be positive or -1");
         pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);
         return Error_IllegalParamVal;
      }
   } else {
      if(static_cast<IntEbm>(pBoosterCore->GetCountTerms()) <= indexTerm) {
         LOG_0(Trace_Error, "ERROR SetTermUpdate indexTerm above the number of terms that we have");
         pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);
         return Error_IllegalParamVal;
      }
      iTerm = static_cast<size_t>(indexTerm);

      // this is true because 0 < pBoosterCore->m_cTerms since our caller needs to pass in a valid indexTerm to this
      // function
      EBM_ASSERT(nullptr != pBoosterCore->GetTerms());
      pTerm = pBoosterCore->GetTerms()[iTerm];
      EBM_ASSERT(nullptr != pTerm);

      cDimensions = pTerm->GetCountDimensions();
   }

   if(size_t{0} == pBoosterCore->GetCountScores()) {
      pBoosterShell->SetTermIndex(iTerm);
      return Error_None;
   }

   if(nullptr != pTerm) {
      size_t cTensorScores = pTerm->GetCountTensorBins();
      if(size_t{0} == cTensorScores) {
         // If we have zero samples and one of the dimensions has 0 bins then there is no tensor, so return now

         // if GetCountTensorBins is 0, then pBoosterShell->GetTermUpdate() does not contain valid data

         LOG_0(Trace_Warning, "WARNING SetTermUpdate size_t { 0 } == cTensorScores");

         pBoosterShell->SetTermIndex(iTerm);
         return Error_None;
      }
   }

   EBM_ASSERT(nullptr != pBoosterShell->GetTermUpdate());
   pBoosterShell->GetTermUpdate()->SetCountDimensions(cDimensions);
   pBoosterShell->GetTermUpdate()->Reset();

   if(nullptr != pTerm) {
      error = pBoosterShell->GetTermUpdate()->Expand(pTerm);
      if(Error_None != error) {
         // already logged
         pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);
         return error;
      }
   }

   FloatScore* const aUpdateScores = pBoosterShell->GetTermUpdate()->GetTensorScoresPointer();
   // *updateScoresTensor is const, but Transpose can go either way.  When bCopyToIncrement is false like it
   // is below, then Transpose will treat updateScoresTensor as const
   Transpose<false>(pTerm, pBoosterCore->GetCountScores(), const_cast<double*>(updateScoresTensor), aUpdateScores);

   pBoosterShell->SetTermIndex(iTerm);

   return Error_None;
}

} // namespace DEFINED_ZONE_NAME
