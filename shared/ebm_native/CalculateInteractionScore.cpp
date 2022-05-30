// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

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

extern void BinInteraction(
   InteractionCore * const pInteractionCore,
   const FeatureGroup * const pFeatureGroup,
   HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
);

extern FloatEbmType PartitionTwoDimensionalInteraction(
   InteractionCore * const pInteractionCore,
   const FeatureGroup * const pFeatureGroup,
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
   const FeatureGroup * const pFeatureGroup,
   const InteractionOptionsType options,
   const size_t cSamplesRequiredForChildSplitMin,
   FloatEbmType * const pInteractionStrengthAvgOut
) {
   // TODO : we NEVER use the hessian term (currently) in HistogramTargetEntry when calculating interaction scores, but we're spending time calculating 
   // it, and it's taking up precious memory.  We should eliminate the hessian term HERE in our datastructures OR we should think whether we can 
   // use the hessian as part of the gain function!!!

   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelVerbose, "Entered CalcInteractionStrengthInternal");

   // situations with 0 dimensions should have been filtered out before this function was called (but still inside the C++)
   EBM_ASSERT(1 <= pFeatureGroup->GetCountDimensions());
   EBM_ASSERT(1 <= pFeatureGroup->GetCountSignificantDimensions());
   EBM_ASSERT(pFeatureGroup->GetCountDimensions() == pFeatureGroup->GetCountSignificantDimensions());

   size_t cAuxillaryBucketsForBuildFastTotals = 0;
   size_t cTotalBucketsMainSpace = 1;
   const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
   const FeatureGroupEntry * const pFeatureGroupEntryEnd = pFeatureGroupEntry + pFeatureGroup->GetCountDimensions();
   do {
      const size_t cBins = pFeatureGroupEntry->m_pFeature->GetCountBins();
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

      ++pFeatureGroupEntry;
   } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);

   const size_t cAuxillaryBucketsForSplitting = 4;
   const size_t cAuxillaryBuckets =
      cAuxillaryBucketsForBuildFastTotals < cAuxillaryBucketsForSplitting ? cAuxillaryBucketsForSplitting : cAuxillaryBucketsForBuildFastTotals;
   if(IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING CalcInteractionStrengthInternal IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)");
      return Error_OutOfMemory;
   }
   const size_t cTotalBuckets = cTotalBucketsMainSpace + cAuxillaryBuckets;

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   if(GetHistogramBucketSizeOverflow(bClassification, cVectorLength)) {
      LOG_0(
         TraceLevelWarning,
         "WARNING CalcInteractionStrengthInternal GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)"
      );
      return Error_OutOfMemory;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);
   if(IsMultiplyError(cBytesPerHistogramBucket, cTotalBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING CalcInteractionStrengthInternal IsMultiplyError(cBytesPerHistogramBucket, cTotalBuckets)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBuffer = cBytesPerHistogramBucket * cTotalBuckets;

   // this doesn't need to be freed since it's tracked and re-used by the class InteractionShell
   HistogramBucketBase * const aHistogramBuckets = pInteractionShell->GetHistogramBucketBase(cBytesBuffer);
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      // already logged
      return Error_OutOfMemory;
   }

   if(bClassification) {
      HistogramBucket<true> * const aHistogramBucketsLocal = aHistogramBuckets->GetHistogramBucket<true>();
      for(size_t i = 0; i < cTotalBuckets; ++i) {
         HistogramBucket<true> * const pHistogramBucket =
            GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBucketsLocal, i);
         pHistogramBucket->Zero(cVectorLength);
      }
   } else {
      HistogramBucket<false> * const aHistogramBucketsLocal = aHistogramBuckets->GetHistogramBucket<false>();
      for(size_t i = 0; i < cTotalBuckets; ++i) {
         HistogramBucket<false> * const pHistogramBucket =
            GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBucketsLocal, i);
         pHistogramBucket->Zero(cVectorLength);
      }
   }

   HistogramBucketBase * pAuxiliaryBucketZone =
      GetHistogramBucketByIndex(cBytesPerHistogramBucket, aHistogramBuckets, cTotalBucketsMainSpace);

#ifndef NDEBUG
   const unsigned char * const aHistogramBucketsEndDebug = reinterpret_cast<unsigned char *>(aHistogramBuckets) + cBytesBuffer;
#endif // NDEBUG

   BinInteraction(
      pInteractionCore,
      pFeatureGroup,
      aHistogramBuckets
#ifndef NDEBUG
      , aHistogramBucketsEndDebug
#endif // NDEBUG
   );

#ifndef NDEBUG
   // make a copy of the original binned buckets for debugging purposes
   HistogramBucketBase * const aHistogramBucketsDebugCopy =
      EbmMalloc<HistogramBucketBase>(cTotalBucketsMainSpace, cBytesPerHistogramBucket);
   if(nullptr != aHistogramBucketsDebugCopy) {
      // if we can't allocate, don't fail.. just stop checking
      const size_t cBytesBufferDebug = cTotalBucketsMainSpace * cBytesPerHistogramBucket;
      memcpy(aHistogramBucketsDebugCopy, aHistogramBuckets, cBytesBufferDebug);
   }
#endif // NDEBUG

   TensorTotalsBuild(
      runtimeLearningTypeOrCountTargetClasses,
      pFeatureGroup,
      pAuxiliaryBucketZone,
      aHistogramBuckets
#ifndef NDEBUG
      , aHistogramBucketsDebugCopy
      , aHistogramBucketsEndDebug
#endif // NDEBUG
   );

   if(2 == pFeatureGroup->GetCountSignificantDimensions()) {
      LOG_0(TraceLevelVerbose, "CalcInteractionStrengthInternal Starting bin sweep loop");

      FloatEbmType bestGain = PartitionTwoDimensionalInteraction(
         pInteractionCore,
         pFeatureGroup,
         options,
         cSamplesRequiredForChildSplitMin,
         pAuxiliaryBucketZone,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );

      if(nullptr != pInteractionStrengthAvgOut) {
         // if totalWeight < 1 then bestGain could overflow to +inf, so do the division first
         const DataSetInteraction * const pDataSet = pInteractionCore->GetDataSetInteraction();
         EBM_ASSERT(nullptr != pDataSet);
         const FloatEbmType totalWeight = pDataSet->GetWeightTotal();
         EBM_ASSERT(FloatEbmType { 0 } < totalWeight); // if all are zeros we assume there are no weights and use the count
         bestGain /= totalWeight;

         if(UNLIKELY(/* NaN */ !LIKELY(bestGain <= std::numeric_limits<FloatEbmType>::max()))) {
            // We simplify our caller's handling by returning -lowest as our error indicator. -lowest will sort to being the
            // least important item, which is good, but it also signals an overflow without the weirness of NaNs.
            EBM_ASSERT(std::isnan(bestGain) || std::numeric_limits<FloatEbmType>::infinity() == bestGain);
            bestGain = k_illegalGain;
         } else if(UNLIKELY(bestGain < FloatEbmType { 0 })) {
            // gain can't mathematically be legally negative, but it can be here in the following situations:
            //   1) for impure interaction gain we subtract the parent partial gain, and there can be floating point
            //      noise that makes this slightly negative
            //   2) for impure interaction gain we subtract the parent partial gain, but if there were no legal cuts
            //      then the partial gain before subtracting the parent partial gain was zero and we then get a 
            //      substantially negative value.  In this case we should not have subtracted the parent partial gain
            //      since we had never even calculated the 4 quadrant partial gain, but we handle this scenario 
            //      here instead of inside the templated function.

            EBM_ASSERT(!std::isnan(bestGain));
            EBM_ASSERT(std::numeric_limits<FloatEbmType>::infinity() != bestGain);
            bestGain = std::numeric_limits<FloatEbmType>::lowest() <= bestGain ? FloatEbmType { 0 } : k_illegalGain;
         } else {
            EBM_ASSERT(!std::isnan(bestGain));
            EBM_ASSERT(!std::isinf(bestGain));
         }
         *pInteractionStrengthAvgOut = bestGain;
      }
   } else {
      EBM_ASSERT(false); // we only support pairs currently
      LOG_0(TraceLevelWarning, "WARNING CalcInteractionStrengthInternal 2 != pFeatureGroup->GetCountSignificantDimensions()");

      // TODO: handle this better
      if(nullptr != pInteractionStrengthAvgOut) {
         // for now, just return any interactions that have other than 2 dimensions as -inf, 
         // which means they won't be considered but indicates they were not handled
         *pInteractionStrengthAvgOut = k_illegalGain;
      }
   }

#ifndef NDEBUG
   free(aHistogramBucketsDebugCopy);
#endif // NDEBUG

   LOG_0(TraceLevelVerbose, "Exited CalcInteractionStrengthInternal");
   return Error_None;
}

// we made this a global because if we had put this variable inside the InteractionCore object, then we would need to dereference that before getting 
// the count.  By making this global we can send a log message incase a bad InteractionCore object is sent into us we only decrease the count if the 
// count is non-zero, so at worst if there is a race condition then we'll output this log message more times than desired, but we can live with that
static int g_cLogCalcInteractionStrengthParametersMessages = 10;

EBM_NATIVE_IMPORT_EXPORT_BODY ErrorEbmType EBM_NATIVE_CALLING_CONVENTION CalcInteractionStrength(
   InteractionHandle interactionHandle,
   IntEbmType countDimensions,
   const IntEbmType * featureIndexes,
   InteractionOptionsType options,
   IntEbmType countSamplesRequiredForChildSplitMin,
   FloatEbmType * avgInteractionStrengthOut
) {
   LOG_COUNTED_N(
      &g_cLogCalcInteractionStrengthParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "CalcInteractionStrength parameters: interactionHandle=%p, countDimensions=%" IntEbmTypePrintf ", featureIndexes=%p, options=0x%" UInteractionOptionsTypePrintf ", countSamplesRequiredForChildSplitMin=%" IntEbmTypePrintf ", avgInteractionStrengthOut=%p",
      static_cast<void *>(interactionHandle),
      countDimensions,
      static_cast<const void *>(featureIndexes),
      static_cast<UInteractionOptionsType>(options), // signed to unsigned conversion is defined behavior in C++
      countSamplesRequiredForChildSplitMin,
      static_cast<void *>(avgInteractionStrengthOut)
   );

   ErrorEbmType error;

   InteractionShell * const pInteractionShell = InteractionShell::GetInteractionShellFromInteractionHandle(interactionHandle);
   if(nullptr == pInteractionShell) {
      if(LIKELY(nullptr != avgInteractionStrengthOut)) {
         *avgInteractionStrengthOut = FloatEbmType { 0 };
      }
      // already logged
      return Error_IllegalParamValue;
   }
   InteractionCore * const pInteractionCore = pInteractionShell->GetInteractionCore();

   LOG_COUNTED_0(pInteractionCore->GetPointerCountLogEnterMessages(), TraceLevelInfo, TraceLevelVerbose, "Entered CalcInteractionStrength");

   if(countDimensions <= IntEbmType { 0 }) {
      if(LIKELY(nullptr != avgInteractionStrengthOut)) {
         *avgInteractionStrengthOut = FloatEbmType { 0 };
      }
      if(IntEbmType { 0 } == countDimensions) {
         LOG_0(TraceLevelInfo, "INFO CalcInteractionStrength empty feature group");
         return Error_None;
      } else {
         LOG_0(TraceLevelError, "ERROR CalcInteractionStrength countDimensions must be positive");
         return Error_IllegalParamValue;
      }
   }
   if(nullptr == featureIndexes) {
      if(LIKELY(nullptr != avgInteractionStrengthOut)) {
         *avgInteractionStrengthOut = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR CalcInteractionStrength featureIndexes cannot be nullptr if 0 < countDimensions");
      return Error_IllegalParamValue;
   }
   if(IntEbmType { k_cDimensionsMax } < countDimensions) {
      if(LIKELY(nullptr != avgInteractionStrengthOut)) {
         *avgInteractionStrengthOut = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelWarning, "WARNING CalcInteractionStrength countDimensions too large and would cause out of memory condition");
      return Error_OutOfMemory;
   }
   size_t cDimensions = static_cast<size_t>(countDimensions);
   if(0 == pInteractionCore->GetDataSetInteraction()->GetCountSamples()) {
      // if there are zero samples, there isn't much basis to say whether there are interactions, so just return zero
      LOG_0(TraceLevelInfo, "INFO CalcInteractionStrength zero samples");
      if(nullptr != avgInteractionStrengthOut) {
         // we return the lowest value possible for the interaction score, but we don't return an error since we handle it even though we'd prefer our 
         // caler be smarter about this condition
         *avgInteractionStrengthOut = 0;
      }
      return Error_None;
   }

   // TODO : test if our InteractionOptionsType options flags only include flags that we use

   size_t cSamplesRequiredForChildSplitMin = size_t { 1 }; // this is the min value
   if(IntEbmType { 1 } <= countSamplesRequiredForChildSplitMin) {
      cSamplesRequiredForChildSplitMin = static_cast<size_t>(countSamplesRequiredForChildSplitMin);
      if(IsConvertError<size_t>(countSamplesRequiredForChildSplitMin)) {
         // we can never exceed a size_t number of samples, so let's just set it to the maximum if we were going to overflow because it will generate 
         // the same results as if we used the true number
         cSamplesRequiredForChildSplitMin = std::numeric_limits<size_t>::max();
      }
   } else {
      LOG_0(TraceLevelWarning, "WARNING CalcInteractionStrength countSamplesRequiredForChildSplitMin can't be less than 1.  Adjusting to 1.");
   }

   const Feature * const aFeatures = pInteractionCore->GetFeatures();
   const IntEbmType * pFeatureIndexes = featureIndexes;
   const IntEbmType * const pFeatureIndexesEnd = featureIndexes + cDimensions;

   do {
      const IntEbmType indexFeatureInterop = *pFeatureIndexes;
      if(indexFeatureInterop < 0) {
         if(LIKELY(nullptr != avgInteractionStrengthOut)) {
            *avgInteractionStrengthOut = FloatEbmType { 0 };
         }
         LOG_0(TraceLevelError, "ERROR CalcInteractionStrength featureIndexes value cannot be negative");
         return Error_IllegalParamValue;
      }
      if(static_cast<IntEbmType>(pInteractionCore->GetCountFeatures()) <= indexFeatureInterop) {
         if(LIKELY(nullptr != avgInteractionStrengthOut)) {
            *avgInteractionStrengthOut = FloatEbmType { 0 };
         }
         LOG_0(TraceLevelError, "ERROR CalcInteractionStrength featureIndexes value must be less than the number of features");
         return Error_IllegalParamValue;
      }
      const size_t iFeature = static_cast<size_t>(indexFeatureInterop);
      const Feature * const pFeature = &aFeatures[iFeature];
      if(pFeature->GetCountBins() <= size_t { 1 }) {
         if(nullptr != avgInteractionStrengthOut) {
            // we return the lowest value possible for the interaction score, but we don't return an error since we handle it even though we'd prefer 
            // our caler be smarter about this condition
            *avgInteractionStrengthOut = FloatEbmType { 0 };
         }
         LOG_0(TraceLevelInfo, "INFO CalcInteractionStrength feature group contains a feature with only 1 bin");
         return Error_None;
      }
      ++pFeatureIndexes;
   } while(pFeatureIndexesEnd != pFeatureIndexes);

   // TODO: instead of putting the FeatureGroup into a character buffer, consider putting k_cDimensionsMax
   //       items in the array by default and dynamically allocate less if we need less, or use a template that
   //       allows specification of the number of items with a default of 1 (this would be the cleanest!)

   // put the pFeatureGroup object on the stack. We want to put it into a FeatureGroup object since we want to share code with boosting, 
   // which calls things like building the tensor totals (which is templated to be compiled many times)
   char FeatureGroupBuffer[FeatureGroup::GetFeatureGroupCountBytes(k_cDimensionsMax)];
   FeatureGroup * const pFeatureGroup = reinterpret_cast<FeatureGroup *>(&FeatureGroupBuffer);
   pFeatureGroup->Initialize(cDimensions, 0);
   pFeatureGroup->SetCountSignificantFeatures(cDimensions);

   pFeatureIndexes = featureIndexes; // restart from the start
   FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
   do {
      // TODO: move this into the loop above

      const IntEbmType indexFeatureInterop = *pFeatureIndexes;
      EBM_ASSERT(0 <= indexFeatureInterop);
      EBM_ASSERT(!IsConvertError<size_t>(indexFeatureInterop)); // we already checked indexFeatureInterop was good above
      size_t iFeature = static_cast<size_t>(indexFeatureInterop);
      EBM_ASSERT(iFeature < pInteractionCore->GetCountFeatures());
      const Feature * const pFeature = &aFeatures[iFeature];
      EBM_ASSERT(2 <= pFeature->GetCountBins()); // we should have filtered out anything with 1 bin above

      pFeatureGroupEntry->m_pFeature = pFeature;
      ++pFeatureGroupEntry;
      ++pFeatureIndexes;
   } while(pFeatureIndexesEnd != pFeatureIndexes);

   if(ptrdiff_t { 0 } == pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses() || ptrdiff_t { 1 } == pInteractionCore->GetRuntimeLearningTypeOrCountTargetClasses()) {
      LOG_0(TraceLevelInfo, "INFO CalcInteractionStrength target with 0/1 classes");
      if(nullptr != avgInteractionStrengthOut) {
         // if there is only 1 classification target, then we can predict the outcome with 100% accuracy and there is no need for logits or 
         // interactions or anything else.  We return 0 since interactions have no benefit
         *avgInteractionStrengthOut = FloatEbmType { 0 };
      }
      return Error_None;
   }

   // TODO: remove the pInteractionCore object here.  pInteractionShell contains pInteractionCore
   error = CalcInteractionStrengthInternal(
      pInteractionShell,
      pInteractionCore,
      pFeatureGroup,
      options,
      cSamplesRequiredForChildSplitMin,
      avgInteractionStrengthOut
   );
   if(Error_None != error) {
      LOG_N(TraceLevelWarning, "WARNING CalcInteractionStrength returned %" ErrorEbmTypePrintf, error);
   }

   if(nullptr != avgInteractionStrengthOut) {
      // if *avgInteractionStrengthOut was negative for floating point instability reasons, we zero it so that we don't return a negative number to our caller
      EBM_ASSERT(FloatEbmType { 0 } <= *avgInteractionStrengthOut);
      LOG_COUNTED_N(
         pInteractionCore->GetPointerCountLogExitMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited CalcInteractionStrength %" FloatEbmTypePrintf, *avgInteractionStrengthOut
      );
   } else {
      LOG_COUNTED_0(pInteractionCore->GetPointerCountLogExitMessages(), TraceLevelInfo, TraceLevelVerbose, "Exited CalcInteractionStrength");
   }
   return error;
}

} // DEFINED_ZONE_NAME
