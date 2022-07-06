// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <string.h> // memcpy

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

// feature includes
#include "Feature.hpp"
#include "FeatureGroup.hpp"
// dataset depends on features
#include "DataSetInteraction.hpp"
#include "InteractionShell.hpp"

#include "InteractionCore.hpp"

#include "TensorTotalsSum.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern void BinInteraction(InteractionShell * const pInteractionShell, const Term * const pTerm);

extern void TensorTotalsBuild(
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
   const Term * const pTerm,
   HistogramBucketBase * pBucketAuxiliaryBuildZone,
   HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
   , HistogramBucketBase * const aHistogramBucketsDebugCopy
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
);

extern double PartitionTwoDimensionalInteraction(
   InteractionCore * const pInteractionCore,
   const Term * const pTerm,
   const InteractionOptionsType options,
   const size_t cSamplesRequiredForChildSplitMin,
   HistogramBucketBase * pAuxiliaryBucketZone,
   HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
   , const HistogramBucketBase * const aHistogramBucketsDebugCopy
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
);

static ErrorEbmType CalcInteractionStrengthInternal(
   InteractionShell * const pInteractionShell,
   InteractionCore * const pInteractionCore,
   const Term * const pTerm,
   const InteractionOptionsType options,
   const size_t cSamplesRequiredForChildSplitMin,
   double * const pInteractionStrengthAvgOut
) {
   // TODO : we NEVER use the hessian term (currently) in HistogramTargetEntry when calculating interaction scores, but we're spending time calculating 
   // it, and it's taking up precious memory.  We should eliminate the hessian term HERE in our datastructures OR we should think whether we can 
   // use the hessian as part of the gain function!!!

   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelVerbose, "Entered CalcInteractionStrengthInternal");

   // situations with 0 dimensions should have been filtered out before this function was called (but still inside the C++)
   EBM_ASSERT(1 <= pTerm->GetCountDimensions());
   EBM_ASSERT(1 <= pTerm->GetCountSignificantDimensions());
   EBM_ASSERT(pTerm->GetCountDimensions() == pTerm->GetCountSignificantDimensions());

   size_t cAuxillaryBucketsForBuildFastTotals = 0;
   size_t cTotalBucketsMainSpace = 1;
   const TermEntry * pTermEntry = pTerm->GetTermEntries();
   const TermEntry * const pTermEntriesEnd = pTermEntry + pTerm->GetCountDimensions();
   do {
      const size_t cBins = pTermEntry->m_pFeature->GetCountBins();
      // situations with 1 bin should have been filtered out before this function was called (but still inside the C++)
      // our tensor code strips out features with 1 bin, and we'd need to do that here too if cBins was 1
      EBM_ASSERT(size_t { 2 } <= cBins);
      // if cBins could be 1, then we'd need to check at runtime for overflow of cAuxillaryBucketsForBuildFastTotals
      // if this wasn't true then we'd have to check IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace) at runtime
      EBM_ASSERT(cAuxillaryBucketsForBuildFastTotals < cTotalBucketsMainSpace);
      // since cBins must be 2 or more, cAuxillaryBucketsForBuildFastTotals must grow slower than cTotalBucketsMainSpace, and we checked at allocation 
      // that cTotalBucketsMainSpace would not overflow
      EBM_ASSERT(!IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace));
      // this can overflow, but if it does then we're guaranteed to catch the overflow via the multiplication check below
      cAuxillaryBucketsForBuildFastTotals += cTotalBucketsMainSpace;
      if(IsMultiplyError(cTotalBucketsMainSpace, cBins)) {
         // unlike in the boosting code where we check at allocation time if the tensor created overflows on multiplication
         // we don't know what group of features our caller will give us for calculating the interaction scores,
         // so we need to check if our caller gave us a tensor that overflows multiplication
         LOG_0(TraceLevelWarning, "WARNING CalcInteractionStrengthInternal IsMultiplyError(cTotalBucketsMainSpace, cBins)");
         return Error_OutOfMemory;
      }
      cTotalBucketsMainSpace *= cBins;
      // if this wasn't true then we'd have to check IsAddError(cAuxillaryBucketsForBuildFastTotals, cTotalBucketsMainSpace) at runtime
      EBM_ASSERT(cAuxillaryBucketsForBuildFastTotals < cTotalBucketsMainSpace);

      ++pTermEntry;
   } while(pTermEntriesEnd != pTermEntry);

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   if(GetHistogramBucketSizeOverflow<FloatFast>(bClassification, cVectorLength) || 
      GetHistogramBucketSizeOverflow<FloatBig>(bClassification, cVectorLength)) 
   {
      LOG_0(
         TraceLevelWarning,
         "WARNING CalcInteractionStrengthInternal GetHistogramBucketSizeOverflow overflow"
      );
      return Error_OutOfMemory;
   }
   const size_t cBytesPerHistogramBucketFast = GetHistogramBucketSize<FloatFast>(bClassification, cVectorLength);
   if(IsMultiplyError(cBytesPerHistogramBucketFast, cTotalBucketsMainSpace)) {
      LOG_0(TraceLevelWarning, "WARNING CalcInteractionStrengthInternal IsMultiplyError(cBytesPerHistogramBucket, cTotalBucketsMainSpace)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBufferFast = cBytesPerHistogramBucketFast * cTotalBucketsMainSpace;

   // this doesn't need to be freed since it's tracked and re-used by the class InteractionShell
   HistogramBucketBase * const aHistogramBucketsFast = pInteractionShell->GetHistogramBucketBaseFast(cBytesBufferFast);
   if(UNLIKELY(nullptr == aHistogramBucketsFast)) {
      // already logged
      return Error_OutOfMemory;
   }
   aHistogramBucketsFast->Zero(cBytesPerHistogramBucketFast, cTotalBucketsMainSpace);

#ifndef NDEBUG
   const unsigned char * const aHistogramBucketsEndDebugFast = reinterpret_cast<unsigned char *>(aHistogramBucketsFast) + cBytesBufferFast;
   pInteractionShell->SetHistogramBucketsEndDebugFast(aHistogramBucketsEndDebugFast);
#endif // NDEBUG

   BinInteraction(pInteractionShell, pTerm);

   const size_t cAuxillaryBucketsForSplitting = 4;
   const size_t cAuxillaryBuckets =
      cAuxillaryBucketsForBuildFastTotals < cAuxillaryBucketsForSplitting ? cAuxillaryBucketsForSplitting : cAuxillaryBucketsForBuildFastTotals;
   if(IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING CalcInteractionStrengthInternal IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)");
      return Error_OutOfMemory;
   }

   const size_t cTotalBucketsBig = cTotalBucketsMainSpace + cAuxillaryBuckets;

   const size_t cBytesPerHistogramBucketBig = GetHistogramBucketSize<FloatBig>(bClassification, cVectorLength);
   if(IsMultiplyError(cBytesPerHistogramBucketBig, cTotalBucketsBig)) {
      LOG_0(TraceLevelWarning, "WARNING CalcInteractionStrengthInternal IsMultiplyError(cBytesPerHistogramBucket, cTotalBucketsBig)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBufferBig = cBytesPerHistogramBucketBig * cTotalBucketsBig;

   HistogramBucketBase * const aHistogramBucketsBig = pInteractionShell->GetHistogramBucketBaseBig(cBytesBufferBig);
   if(UNLIKELY(nullptr == aHistogramBucketsBig)) {
      // already logged
      return Error_OutOfMemory;
   }
   aHistogramBucketsBig->Zero(cBytesPerHistogramBucketBig, cAuxillaryBuckets, cTotalBucketsMainSpace);

#ifndef NDEBUG
   const unsigned char * const aHistogramBucketsEndDebugBig = reinterpret_cast<unsigned char *>(aHistogramBucketsBig) + cBytesBufferBig;
#endif // NDEBUG

   // TODO: put this into it's own function that converts our fast floats to big floats
   static_assert(sizeof(FloatBig) == sizeof(FloatFast), "float mismatch");
   memcpy(aHistogramBucketsBig, aHistogramBucketsFast, cBytesBufferFast);


   // TODO: we can exit here back to python to allow caller modification to our histograms


#ifndef NDEBUG
   // make a copy of the original binned buckets for debugging purposes
   HistogramBucketBase * const aHistogramBucketsDebugCopy =
      EbmMalloc<HistogramBucketBase>(cTotalBucketsMainSpace, cBytesPerHistogramBucketBig);
   if(nullptr != aHistogramBucketsDebugCopy) {
      // if we can't allocate, don't fail.. just stop checking
      const size_t cBytesBufferDebug = cTotalBucketsMainSpace * cBytesPerHistogramBucketBig;
      memcpy(aHistogramBucketsDebugCopy, aHistogramBucketsBig, cBytesBufferDebug);
   }
#endif // NDEBUG

   HistogramBucketBase * pAuxiliaryBucketZone =
      GetHistogramBucketByIndex(cBytesPerHistogramBucketBig, aHistogramBucketsBig, cTotalBucketsMainSpace);

   TensorTotalsBuild(
      runtimeLearningTypeOrCountTargetClasses,
      pTerm,
      pAuxiliaryBucketZone,
      aHistogramBucketsBig
#ifndef NDEBUG
      , aHistogramBucketsDebugCopy
      , aHistogramBucketsEndDebugBig
#endif // NDEBUG
   );

   if(2 == pTerm->GetCountSignificantDimensions()) {
      LOG_0(TraceLevelVerbose, "CalcInteractionStrengthInternal Starting bin sweep loop");

      double bestGain = PartitionTwoDimensionalInteraction(
         pInteractionCore,
         pTerm,
         options,
         cSamplesRequiredForChildSplitMin,
         pAuxiliaryBucketZone,
         aHistogramBucketsBig
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebugBig
#endif // NDEBUG
      );

      if(nullptr != pInteractionStrengthAvgOut) {
         // if totalWeight < 1 then bestGain could overflow to +inf, so do the division first
         const DataSetInteraction * const pDataSet = pInteractionCore->GetDataSetInteraction();
         EBM_ASSERT(nullptr != pDataSet);
         const double totalWeight = static_cast<double>(pDataSet->GetWeightTotal());
         EBM_ASSERT(0 < totalWeight); // if all are zeros we assume there are no weights and use the count
         bestGain /= totalWeight;

         if(UNLIKELY(/* NaN */ !LIKELY(bestGain <= std::numeric_limits<double>::max()))) {
            // We simplify our caller's handling by returning -lowest as our error indicator. -lowest will sort to being the
            // least important item, which is good, but it also signals an overflow without the weirness of NaNs.
            EBM_ASSERT(std::isnan(bestGain) || std::numeric_limits<double>::infinity() == bestGain);
            bestGain = k_illegalGainDouble;
         } else if(UNLIKELY(bestGain < 0)) {
            // gain can't mathematically be legally negative, but it can be here in the following situations:
            //   1) for impure interaction gain we subtract the parent partial gain, and there can be floating point
            //      noise that makes this slightly negative
            //   2) for impure interaction gain we subtract the parent partial gain, but if there were no legal cuts
            //      then the partial gain before subtracting the parent partial gain was zero and we then get a 
            //      substantially negative value.  In this case we should not have subtracted the parent partial gain
            //      since we had never even calculated the 4 quadrant partial gain, but we handle this scenario 
            //      here instead of inside the templated function.

            EBM_ASSERT(!std::isnan(bestGain));
            EBM_ASSERT(std::numeric_limits<double>::infinity() != bestGain);
            bestGain = std::numeric_limits<double>::lowest() <= bestGain ? 0.0 : k_illegalGainDouble;
         } else {
            EBM_ASSERT(!std::isnan(bestGain));
            EBM_ASSERT(!std::isinf(bestGain));
         }
         *pInteractionStrengthAvgOut = bestGain;
      }
   } else {
      EBM_ASSERT(false); // we only support pairs currently
      LOG_0(TraceLevelWarning, "WARNING CalcInteractionStrengthInternal 2 != pTerm->GetCountSignificantDimensions()");

      // TODO: handle this better
      if(nullptr != pInteractionStrengthAvgOut) {
         // for now, just return any interactions that have other than 2 dimensions as -inf, 
         // which means they won't be considered but indicates they were not handled
         *pInteractionStrengthAvgOut = k_illegalGainDouble;
      }
   }

#ifndef NDEBUG
   free(aHistogramBucketsDebugCopy);
#endif // NDEBUG

   LOG_0(TraceLevelVerbose, "Exited CalcInteractionStrengthInternal");
   return Error_None;
}

// there is a race condition for decrementing this variable, but if a thread loses the 
// race then it just doesn't get decremented as quickly, which we can live with
static int g_cLogCalcInteractionStrengthParametersMessages = 10;

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CalcInteractionStrength(
   InteractionHandle interactionHandle,
   IntEbmType countDimensions,
   const IntEbmType * featureIndexes,
   InteractionOptionsType options,
   IntEbmType countSamplesRequiredForChildSplitMin,
   double * avgInteractionStrengthOut
) {
   LOG_COUNTED_N(
      &g_cLogCalcInteractionStrengthParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "CalcInteractionStrength: "
      "interactionHandle=%p, "
      "countDimensions=%" IntEbmTypePrintf ", "
      "featureIndexes=%p, "
      "options=0x%" UInteractionOptionsTypePrintf ", "
      "countSamplesRequiredForChildSplitMin=%" IntEbmTypePrintf ", "
      "avgInteractionStrengthOut=%p"
      ,
      static_cast<void *>(interactionHandle),
      countDimensions,
      static_cast<const void *>(featureIndexes),
      static_cast<UInteractionOptionsType>(options), // signed to unsigned conversion is defined behavior in C++
      countSamplesRequiredForChildSplitMin,
      static_cast<void *>(avgInteractionStrengthOut)
   );

   if(LIKELY(nullptr != avgInteractionStrengthOut)) {
      *avgInteractionStrengthOut = k_illegalGainDouble;
   }

   ErrorEbmType error;

   InteractionShell * const pInteractionShell = InteractionShell::GetInteractionShellFromHandle(interactionHandle);
   if(nullptr == pInteractionShell) {
      // already logged
      return Error_IllegalParamValue;
   }
   LOG_COUNTED_0(
      pInteractionShell->GetPointerCountLogEnterMessages(), 
      TraceLevelInfo, 
      TraceLevelVerbose, 
      "Entered CalcInteractionStrength"
   );

   if(0 != ((~static_cast<UInteractionOptionsType>(InteractionOptions_Pure)) &
      static_cast<UInteractionOptionsType>(options))) {
      LOG_0(TraceLevelError, "ERROR CalcInteractionStrength options contains unknown flags. Ignoring extras.");
   }

   size_t cSamplesRequiredForChildSplitMin = size_t { 1 }; // this is the min value
   if(IntEbmType { 1 } <= countSamplesRequiredForChildSplitMin) {
      cSamplesRequiredForChildSplitMin = static_cast<size_t>(countSamplesRequiredForChildSplitMin);
      if(IsConvertError<size_t>(countSamplesRequiredForChildSplitMin)) {
         // we can never exceed a size_t number of samples, so let's just set it to the maximum if we were going to 
         // overflow because it will generate the same results as if we used the true number
         cSamplesRequiredForChildSplitMin = std::numeric_limits<size_t>::max();
      }
   } else {
      LOG_0(TraceLevelWarning, "WARNING CalcInteractionStrength countSamplesRequiredForChildSplitMin can't be less than 1. Adjusting to 1.");
   }

   if(countDimensions <= IntEbmType { 0 }) {
      if(IntEbmType { 0 } == countDimensions) {
         LOG_0(TraceLevelInfo, "INFO CalcInteractionStrength empty feature list");
         if(LIKELY(nullptr != avgInteractionStrengthOut)) {
            *avgInteractionStrengthOut = 0.0;
         }
         return Error_None;
      } else {
         LOG_0(TraceLevelError, "ERROR CalcInteractionStrength countDimensions must be positive");
         return Error_IllegalParamValue;
      }
   }
   if(nullptr == featureIndexes) {
      LOG_0(TraceLevelError, "ERROR CalcInteractionStrength featureIndexes cannot be nullptr if 0 < countDimensions");
      return Error_IllegalParamValue;
   }
   if(IntEbmType { k_cDimensionsMax } < countDimensions) {
      LOG_0(TraceLevelWarning, "WARNING CalcInteractionStrength countDimensions too large and would cause out of memory condition");
      return Error_OutOfMemory;
   }
   size_t cDimensions = static_cast<size_t>(countDimensions);

   Term term;
   TermEntry * pTermEntry = term.GetTermEntries();
   InteractionCore * const pInteractionCore = pInteractionShell->GetInteractionCore();
   const Feature * const aFeatures = pInteractionCore->GetFeatures();
   const IntEbmType * piFeature = featureIndexes;
   const IntEbmType * const piFeaturesEnd = featureIndexes + cDimensions;
   size_t cTensorBins = 1;
   do {
      // TODO: merge this loop with the one below inside the internal function

      const IntEbmType indexFeature = *piFeature;
      if(indexFeature < IntEbmType { 0 }) {
         LOG_0(TraceLevelError, "ERROR CalcInteractionStrength featureIndexes value cannot be negative");
         return Error_IllegalParamValue;
      }
      if(static_cast<IntEbmType>(pInteractionCore->GetCountFeatures()) <= indexFeature) {
         LOG_0(TraceLevelError, "ERROR CalcInteractionStrength featureIndexes value must be less than the number of features");
         return Error_IllegalParamValue;
      }
      const size_t iFeature = static_cast<size_t>(indexFeature);
      const Feature * const pFeature = &aFeatures[iFeature];
      const size_t cBins = pFeature->GetCountBins();
      if(cBins <= size_t { 1 }) {
         LOG_0(TraceLevelInfo, "INFO CalcInteractionStrength feature group contains a feature with only 1 bin");
         if(nullptr != avgInteractionStrengthOut) {
            *avgInteractionStrengthOut = double { 0 };
         }
         return Error_None;
      }
      if(IsMultiplyError(cTensorBins, cBins)) {
         LOG_0(TraceLevelWarning, "WARNING CalcInteractionStrength IsMultiplyError(cTensorBins, cBins)");
         return Error_OutOfMemory;
      }
      cTensorBins *= cBins;

      pTermEntry->m_pFeature = pFeature;
      ++pTermEntry;

      ++piFeature;
   } while(piFeaturesEnd != piFeature);
   term.Initialize(cDimensions, 0);
   term.SetCountTensorBins(cTensorBins);
   term.SetCountSignificantFeatures(cDimensions); // if we get past the loop below this will be true

   if(size_t { 0 } == pInteractionCore->GetDataSetInteraction()->GetCountSamples()) {
      // if there are zero samples, there isn't much basis to say whether there are interactions, so just return zero
      LOG_0(TraceLevelInfo, "INFO CalcInteractionStrength zero samples");
      if(nullptr != avgInteractionStrengthOut) {
         *avgInteractionStrengthOut = double { 0 };
      }
      return Error_None;
   }
   // GetRuntimeLearningTypeOrCountTargetClasses cannot be zero if there is 1 or more samples
   EBM_ASSERT(ptrdiff_t { 0 } != pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses());

   if(ptrdiff_t { 1 } == pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses()) {
      LOG_0(TraceLevelInfo, "INFO CalcInteractionStrength target with 1 class perfectly predicts the target");
      if(nullptr != avgInteractionStrengthOut) {
         *avgInteractionStrengthOut = double { 0 };
      }
      return Error_None;
   }

   // TODO: remove the pInteractionCore object here.  pInteractionShell contains pInteractionCore
   error = CalcInteractionStrengthInternal(
      pInteractionShell,
      pInteractionCore,
      &term,
      options,
      cSamplesRequiredForChildSplitMin,
      avgInteractionStrengthOut
   );
   if(Error_None != error) {
      LOG_N(TraceLevelWarning, "WARNING CalcInteractionStrength: return=%" ErrorEbmTypePrintf, error);
      return error;
   }

   if(nullptr != avgInteractionStrengthOut) {
      EBM_ASSERT(k_illegalGainDouble == *avgInteractionStrengthOut || double { 0 } <= *avgInteractionStrengthOut);
      LOG_COUNTED_N(
         pInteractionShell->GetPointerCountLogExitMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited CalcInteractionStrength: "
         "*avgInteractionStrengthOut=%le"
         , 
         *avgInteractionStrengthOut
      );
   } else {
      LOG_COUNTED_0(
         pInteractionShell->GetPointerCountLogExitMessages(),
         TraceLevelInfo, 
         TraceLevelVerbose, 
         "Exited CalcInteractionStrength"
      );
   }
   return Error_None;
}

} // DEFINED_ZONE_NAME
