// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebm_native.h"
#include "EbmInternal.h"
#include "Logging.h" // EBM_ASSERT & LOG
// feature includes
#include "FeatureAtomic.h"
#include "FeatureGroup.h"
// dataset depends on features
#include "DataFrameInteraction.h"
#include "ThreadStateInteraction.h"

#include "InteractionDetector.h"

#include "TensorTotalsSum.h"

extern void BinInteraction(
   InteractionDetector * const pInteractionDetector,
   const FeatureGroup * const pFeatureGroup,
   HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
);

extern FloatEbmType FindBestInteractionGainPairs(
   InteractionDetector * const pInteractionDetector,
   const FeatureGroup * const pFeatureGroup,
   const size_t cSamplesRequiredForChildSplitMin,
   HistogramBucketBase * pAuxiliaryBucketZone,
   HistogramBucketBase * const aHistogramBuckets
#ifndef NDEBUG
   , const HistogramBucketBase * const aHistogramBucketsDebugCopy
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
);

static bool CalculateInteractionScoreInternal(
   ThreadStateInteraction * const pThreadStateInteraction,
   InteractionDetector * const pInteractionDetector,
   const FeatureGroup * const pFeatureGroup,
   const size_t cSamplesRequiredForChildSplitMin,
   FloatEbmType * const pInteractionScoreReturn
) {
   // TODO : we NEVER use the denominator term in HistogramTargetEntry when calculating interaction scores, but we're spending time calculating 
   // it, and it's taking up precious memory.  We should eliminate the denominator term HERE in our datastructures OR we should think whether we can 
   // use the denominator as part of the gain function!!!

   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pInteractionDetector->GetRuntimeLearningTypeOrCountTargetClasses();
   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelVerbose, "Entered CalculateInteractionScoreInternal");

   // situations with 0 dimensions should have been filtered out before this function was called (but still inside the C++)
   EBM_ASSERT(1 <= pFeatureGroup->GetCountDimensions());
   EBM_ASSERT(1 <= pFeatureGroup->GetCountSignificantDimensions());
   EBM_ASSERT(pFeatureGroup->GetCountDimensions() == pFeatureGroup->GetCountSignificantDimensions());

   size_t cAuxillaryBucketsForBuildFastTotals = 0;
   size_t cTotalBucketsMainSpace = 1;
   const FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
   const FeatureGroupEntry * const pFeatureGroupEntryEnd = pFeatureGroupEntry + pFeatureGroup->GetCountDimensions();
   do {
      const size_t cBins = pFeatureGroupEntry->m_pFeatureAtomic->GetCountBins();
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
         LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScoreInternal IsMultiplyError(cTotalBucketsMainSpace, cBins)");
         return true;
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
      LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScoreInternal IsAddError(cTotalBucketsMainSpace, cAuxillaryBuckets)");
      return true;
   }
   const size_t cTotalBuckets = cTotalBucketsMainSpace + cAuxillaryBuckets;

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   if(GetHistogramBucketSizeOverflow(bClassification, cVectorLength)) {
      LOG_0(
         TraceLevelWarning,
         "WARNING CalculateInteractionScoreInternal GetHistogramBucketSizeOverflow<bClassification>(cVectorLength)"
      );
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize(bClassification, cVectorLength);
   if(IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)) {
      LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScoreInternal IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)");
      return true;
   }
   const size_t cBytesBuffer = cTotalBuckets * cBytesPerHistogramBucket;

   // this doesn't need to be freed since it's tracked and re-used by the class ThreadStateInteraction
   HistogramBucketBase * const aHistogramBuckets = pThreadStateInteraction->GetHistogramBucketBase(cBytesBuffer);
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScoreInternal nullptr == aHistogramBuckets");
      return true;
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
      pInteractionDetector,
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
      LOG_0(TraceLevelVerbose, "CalculateInteractionScoreInternal Starting bin sweep loop");

      FloatEbmType bestSplittingScore = FindBestInteractionGainPairs(
         pInteractionDetector,
         pFeatureGroup,
         cSamplesRequiredForChildSplitMin,
         pAuxiliaryBucketZone,
         aHistogramBuckets
#ifndef NDEBUG
         , aHistogramBucketsDebugCopy
         , aHistogramBucketsEndDebug
#endif // NDEBUG
      );

      LOG_0(TraceLevelVerbose, "CalculateInteractionScoreInternal Done bin sweep loop");

      if(nullptr != pInteractionScoreReturn) {
         // we started our score at zero, and didn't replace with anything lower, so it can't be below zero
         // if we collected a NaN value, then we kept it
         EBM_ASSERT(std::isnan(bestSplittingScore) || FloatEbmType { 0 } <= bestSplittingScore);
         EBM_ASSERT((!bClassification) || !std::isinf(bestSplittingScore));

         // if bestSplittingScore was NaN we make it zero so that it's not included.  If infinity, also don't include it since we overloaded something
         // even though bestSplittingScore shouldn't be +-infinity for classification, we check it for +-infinity 
         // here since it's most efficient to check that the exponential is all ones, which is the case only for +-infinity and NaN, but not others

         // comparing to max is a good way to check for +infinity without using infinity, which can be problematic on
         // some compilers with some compiler settings.  Using <= helps avoid optimization away because the compiler
         // might assume that nothing is larger than max if it thinks there's no +infinity

         if(UNLIKELY(UNLIKELY(std::isnan(bestSplittingScore)) || 
            UNLIKELY(std::numeric_limits<FloatEbmType>::max() <= bestSplittingScore))) {
            bestSplittingScore = FloatEbmType { 0 };
         }
         *pInteractionScoreReturn = bestSplittingScore;
      }
   } else {
      EBM_ASSERT(false); // we only support pairs currently
      LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScoreInternal 2 != pFeatureGroup->GetCountSignificantDimensions()");

      // TODO: handle this better
      if(nullptr != pInteractionScoreReturn) {
         // for now, just return any interactions that have other than 2 dimensions as zero, which means they won't be considered
         *pInteractionScoreReturn = FloatEbmType { 0 };
      }
   }

#ifndef NDEBUG
   free(aHistogramBucketsDebugCopy);
#endif // NDEBUG

   LOG_0(TraceLevelVerbose, "Exited CalculateInteractionScoreInternal");
   return false;
}

// we made this a global because if we had put this variable inside the InteractionDetector object, then we would need to dereference that before getting 
// the count.  By making this global we can send a log message incase a bad InteractionDetector object is sent into us we only decrease the count if the 
// count is non-zero, so at worst if there is a race condition then we'll output this log message more times than desired, but we can live with that
static int g_cLogCalculateInteractionScoreParametersMessages = 10;

EBM_NATIVE_IMPORT_EXPORT_BODY IntEbmType EBM_NATIVE_CALLING_CONVENTION CalculateInteractionScore(
   InteractionDetectorHandle interactionDetectorHandle,
   IntEbmType countDimensions,
   const IntEbmType * featureAtomicIndexes,
   IntEbmType countSamplesRequiredForChildSplitMin,
   FloatEbmType * interactionScoreOut
) {
   LOG_COUNTED_N(
      &g_cLogCalculateInteractionScoreParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "CalculateInteractionScore parameters: interactionDetectorHandle=%p, countDimensions=%" IntEbmTypePrintf ", featureAtomicIndexes=%p, countSamplesRequiredForChildSplitMin=%" IntEbmTypePrintf ", interactionScoreOut=%p",
      static_cast<void *>(interactionDetectorHandle),
      countDimensions,
      static_cast<const void *>(featureAtomicIndexes),
      countSamplesRequiredForChildSplitMin,
      static_cast<void *>(interactionScoreOut)
   );

   InteractionDetector * pInteractionDetector = reinterpret_cast<InteractionDetector *>(interactionDetectorHandle);
   if(nullptr == pInteractionDetector) {
      if(LIKELY(nullptr != interactionScoreOut)) {
         *interactionScoreOut = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR CalculateInteractionScore ebmInteraction cannot be nullptr");
      return 1;
   }

   LOG_COUNTED_0(pInteractionDetector->GetPointerCountLogEnterMessages(), TraceLevelInfo, TraceLevelVerbose, "Entered CalculateInteractionScore");

   if(countDimensions < 0) {
      if(LIKELY(nullptr != interactionScoreOut)) {
         *interactionScoreOut = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR CalculateInteractionScore countDimensions must be positive");
      return 1;
   }
   if(0 != countDimensions && nullptr == featureAtomicIndexes) {
      if(LIKELY(nullptr != interactionScoreOut)) {
         *interactionScoreOut = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR CalculateInteractionScore featureAtomicIndexes cannot be nullptr if 0 < countDimensions");
      return 1;
   }
   if(!IsNumberConvertable<size_t>(countDimensions)) {
      if(LIKELY(nullptr != interactionScoreOut)) {
         *interactionScoreOut = FloatEbmType { 0 };
      }
      LOG_0(TraceLevelError, "ERROR CalculateInteractionScore countDimensions too large to index");
      return 1;
   }
   size_t cDimensions = static_cast<size_t>(countDimensions);
   if(0 == cDimensions) {
      LOG_0(TraceLevelInfo, "INFO CalculateInteractionScore empty feature group");
      if(nullptr != interactionScoreOut) {
         // we return the lowest value possible for the interaction score, but we don't return an error since we handle it even though we'd prefer our 
         // caler be smarter about this condition
         *interactionScoreOut = FloatEbmType { 0 };
      }
      return 0;
   }
   if(0 == pInteractionDetector->GetDataFrameInteraction()->GetCountSamples()) {
      // if there are zero samples, there isn't much basis to say whether there are interactions, so just return zero
      LOG_0(TraceLevelInfo, "INFO CalculateInteractionScore zero samples");
      if(nullptr != interactionScoreOut) {
         // we return the lowest value possible for the interaction score, but we don't return an error since we handle it even though we'd prefer our 
         // caler be smarter about this condition
         *interactionScoreOut = 0;
      }
      return 0;
   }

   size_t cSamplesRequiredForChildSplitMin = size_t { 1 }; // this is the min value
   if(IntEbmType { 1 } <= countSamplesRequiredForChildSplitMin) {
      cSamplesRequiredForChildSplitMin = static_cast<size_t>(countSamplesRequiredForChildSplitMin);
      if(!IsNumberConvertable<size_t>(countSamplesRequiredForChildSplitMin)) {
         // we can never exceed a size_t number of samples, so let's just set it to the maximum if we were going to overflow because it will generate 
         // the same results as if we used the true number
         cSamplesRequiredForChildSplitMin = std::numeric_limits<size_t>::max();
      }
   } else {
      LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScore countSamplesRequiredForChildSplitMin can't be less than 1.  Adjusting to 1.");
   }

   const FeatureAtomic * const aFeatureAtomics = pInteractionDetector->GetFeatureAtomics();
   const IntEbmType * pFeatureAtomicIndexes = featureAtomicIndexes;
   const IntEbmType * const pFeatureAtomicIndexesEnd = featureAtomicIndexes + cDimensions;

   do {
      const IntEbmType indexFeatureAtomicInterop = *pFeatureAtomicIndexes;
      if(indexFeatureAtomicInterop < 0) {
         if(LIKELY(nullptr != interactionScoreOut)) {
            *interactionScoreOut = FloatEbmType { 0 };
         }
         LOG_0(TraceLevelError, "ERROR CalculateInteractionScore featureAtomicIndexes value cannot be negative");
         return 1;
      }
      if(!IsNumberConvertable<size_t>(indexFeatureAtomicInterop)) {
         if(LIKELY(nullptr != interactionScoreOut)) {
            *interactionScoreOut = FloatEbmType { 0 };
         }
         LOG_0(TraceLevelError, "ERROR CalculateInteractionScore featureAtomicIndexes value too big to reference memory");
         return 1;
      }
      const size_t iFeatureAtomic = static_cast<size_t>(indexFeatureAtomicInterop);
      if(pInteractionDetector->GetCountFeatureAtomics() <= iFeatureAtomic) {
         if(LIKELY(nullptr != interactionScoreOut)) {
            *interactionScoreOut = FloatEbmType { 0 };
         }
         LOG_0(TraceLevelError, "ERROR CalculateInteractionScore featureAtomicIndexes value must be less than the number of features");
         return 1;
      }
      const FeatureAtomic * const pFeatureAtomic = &aFeatureAtomics[iFeatureAtomic];
      if(pFeatureAtomic->GetCountBins() <= size_t { 1 }) {
         if(nullptr != interactionScoreOut) {
            // we return the lowest value possible for the interaction score, but we don't return an error since we handle it even though we'd prefer 
            // our caler be smarter about this condition
            *interactionScoreOut = FloatEbmType { 0 };
         }
         LOG_0(TraceLevelInfo, "INFO CalculateInteractionScore feature group contains a feature with only 1 bin");
         return IntEbmType { 0 };
      }
      ++pFeatureAtomicIndexes;
   } while(pFeatureAtomicIndexesEnd != pFeatureAtomicIndexes);

   if(k_cDimensionsMax < cDimensions) {
      // if we try to run with more than k_cDimensionsMax we'll exceed our memory capacity, so let's exit here instead
      LOG_0(TraceLevelWarning, "WARNING CalculateInteractionScore k_cDimensionsMax < cDimensions");
      return 1;
   }

   // TODO: instead of putting the FeatureGroup into a character buffer, consider putting k_cDimensionsMax
   //       items in the array by default and dynamically allocate less if we need less, or use a template that
   //       allows specification of the number of items with a default of 1 (this would be the cleanest!)

   // put the pFeatureGroup object on the stack. We want to put it into a FeatureGroup object since we want to share code with boosting, 
   // which calls things like building the tensor totals (which is templated to be compiled many times)
   char FeatureGroupBuffer[FeatureGroup::GetFeatureGroupCountBytes(k_cDimensionsMax)];
   FeatureGroup * const pFeatureGroup = reinterpret_cast<FeatureGroup *>(&FeatureGroupBuffer);
   pFeatureGroup->Initialize(cDimensions, 0);
   pFeatureGroup->SetCountSignificantFeatures(cDimensions);

   pFeatureAtomicIndexes = featureAtomicIndexes; // restart from the start
   FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
   do {
      // TODO: move this into the loop above

      const IntEbmType indexFeatureAtomicInterop = *pFeatureAtomicIndexes;
      EBM_ASSERT(0 <= indexFeatureAtomicInterop);
      EBM_ASSERT(IsNumberConvertable<size_t>(indexFeatureAtomicInterop)); // we already checked indexFeatureInterop was good above
      size_t iFeatureAtomic = static_cast<size_t>(indexFeatureAtomicInterop);
      EBM_ASSERT(iFeatureAtomic < pInteractionDetector->GetCountFeatureAtomics());
      const FeatureAtomic * const pFeatureAtomic = &aFeatureAtomics[iFeatureAtomic];
      EBM_ASSERT(2 <= pFeatureAtomic->GetCountBins()); // we should have filtered out anything with 1 bin above

      pFeatureGroupEntry->m_pFeatureAtomic = pFeatureAtomic;
      ++pFeatureGroupEntry;
      ++pFeatureAtomicIndexes;
   } while(pFeatureAtomicIndexesEnd != pFeatureAtomicIndexes);

   if(ptrdiff_t { 0 } == pInteractionDetector->GetRuntimeLearningTypeOrCountTargetClasses() || ptrdiff_t { 1 } == pInteractionDetector->GetRuntimeLearningTypeOrCountTargetClasses()) {
      LOG_0(TraceLevelInfo, "INFO CalculateInteractionScore target with 0/1 classes");
      if(nullptr != interactionScoreOut) {
         // if there is only 1 classification target, then we can predict the outcome with 100% accuracy and there is no need for logits or 
         // interactions or anything else.  We return 0 since interactions have no benefit
         *interactionScoreOut = FloatEbmType { 0 };
      }
      return 0;
   }

   // TODO : be smarter about our ThreadStateInteraction, otherwise why have it?
   ThreadStateInteraction * const pThreadStateInteraction = ThreadStateInteraction::Allocate();
   if(nullptr == pThreadStateInteraction) {
      return 1;
   }

   IntEbmType ret = CalculateInteractionScoreInternal(
      pThreadStateInteraction,
      pInteractionDetector,
      pFeatureGroup,
      cSamplesRequiredForChildSplitMin,
      interactionScoreOut
   );

   ThreadStateInteraction::Free(pThreadStateInteraction);

   if(0 != ret) {
      LOG_N(TraceLevelWarning, "WARNING CalculateInteractionScore returned %" IntEbmTypePrintf, ret);
   }

   if(nullptr != interactionScoreOut) {
      // if *interactionScoreOut was negative for floating point instability reasons, we zero it so that we don't return a negative number to our caller
      EBM_ASSERT(FloatEbmType { 0 } <= *interactionScoreOut);
      LOG_COUNTED_N(
         pInteractionDetector->GetPointerCountLogExitMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited CalculateInteractionScore %" FloatEbmTypePrintf, *interactionScoreOut
      );
   } else {
      LOG_COUNTED_0(pInteractionDetector->GetPointerCountLogExitMessages(), TraceLevelInfo, TraceLevelVerbose, "Exited CalculateInteractionScore");
   }
   return ret;
}
