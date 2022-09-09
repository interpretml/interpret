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
#include "Term.hpp"
// dataset depends on features
#include "DataSetInteraction.hpp"
#include "InteractionShell.hpp"

#include "InteractionCore.hpp"

#include "TensorTotalsSum.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern void BinSumsInteraction(
   InteractionShell * const pInteractionShell,
   const size_t cRealDimensions,
   const size_t * const aiFeatures,
   const size_t * const acBins
);

extern void TensorTotalsBuild(
   const ptrdiff_t cClasses,
   const size_t cRealDimensions,
   const size_t * const acBins,
   BinBase * aAuxiliaryBinsBase,
   BinBase * const aBinsBase
#ifndef NDEBUG
   , BinBase * const aDebugCopyBinsBase
   , const unsigned char * const pBinsEndDebug
#endif // NDEBUG
);

extern double PartitionTwoDimensionalInteraction(
   InteractionCore * const pInteractionCore,
   const size_t cRealDimensions,
   const size_t * const acBins,
   const InteractionFlags flags,
   const size_t cSamplesLeafMin,
   BinBase * aAuxiliaryBinsBase,
   BinBase * const aBinsBase
#ifndef NDEBUG
   , const BinBase * const aDebugCopyBinsBase
   , const unsigned char * const pBinsEndDebug
#endif // NDEBUG
);

// there is a race condition for decrementing this variable, but if a thread loses the 
// race then it just doesn't get decremented as quickly, which we can live with
static int g_cLogCalcInteractionStrength = 10;

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION CalcInteractionStrength(
   InteractionHandle interactionHandle,
   IntEbm countDimensions,
   const IntEbm * featureIndexes,
   InteractionFlags flags,
   IntEbm minSamplesLeaf,
   double * avgInteractionStrengthOut
) {
   LOG_COUNTED_N(
      &g_cLogCalcInteractionStrength,
      Trace_Info,
      Trace_Verbose,
      "CalcInteractionStrength: "
      "interactionHandle=%p, "
      "countDimensions=%" IntEbmPrintf ", "
      "featureIndexes=%p, "
      "flags=0x%" UInteractionFlagsPrintf ", "
      "minSamplesLeaf=%" IntEbmPrintf ", "
      "avgInteractionStrengthOut=%p"
      ,
      static_cast<void *>(interactionHandle),
      countDimensions,
      static_cast<const void *>(featureIndexes),
      static_cast<UInteractionFlags>(flags), // signed to unsigned conversion is defined behavior in C++
      minSamplesLeaf,
      static_cast<void *>(avgInteractionStrengthOut)
   );

   if(LIKELY(nullptr != avgInteractionStrengthOut)) {
      *avgInteractionStrengthOut = k_illegalGainDouble;
   }

   InteractionShell * const pInteractionShell = InteractionShell::GetInteractionShellFromHandle(interactionHandle);
   if(nullptr == pInteractionShell) {
      // already logged
      return Error_IllegalParamVal;
   }
   LOG_COUNTED_0(
      pInteractionShell->GetPointerCountLogEnterMessages(), 
      Trace_Info, 
      Trace_Verbose, 
      "Entered CalcInteractionStrength"
   );

   if(0 != (static_cast<UInteractionFlags>(flags) & ~(
      static_cast<UInteractionFlags>(InteractionFlags_Pure)
      ))) {
      LOG_0(Trace_Error, "ERROR CalcInteractionStrength flags contains unknown flags. Ignoring extras.");
   }

   size_t cSamplesLeafMin = size_t { 1 }; // this is the min value
   if(IntEbm { 1 } <= minSamplesLeaf) {
      cSamplesLeafMin = static_cast<size_t>(minSamplesLeaf);
      if(IsConvertError<size_t>(minSamplesLeaf)) {
         // we can never exceed a size_t number of samples, so let's just set it to the maximum if we were going to 
         // overflow because it will generate the same results as if we used the true number
         cSamplesLeafMin = std::numeric_limits<size_t>::max();
      }
   } else {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrength minSamplesLeaf can't be less than 1. Adjusting to 1.");
   }

   if(countDimensions <= IntEbm { 0 }) {
      if(IntEbm { 0 } == countDimensions) {
         LOG_0(Trace_Info, "INFO CalcInteractionStrength empty feature list");
         if(LIKELY(nullptr != avgInteractionStrengthOut)) {
            *avgInteractionStrengthOut = 0.0;
         }
         return Error_None;
      } else {
         LOG_0(Trace_Error, "ERROR CalcInteractionStrength countDimensions must be positive");
         return Error_IllegalParamVal;
      }
   }
   if(nullptr == featureIndexes) {
      LOG_0(Trace_Error, "ERROR CalcInteractionStrength featureIndexes cannot be nullptr if 0 < countDimensions");
      return Error_IllegalParamVal;
   }
   if(IntEbm { k_cDimensionsMax } < countDimensions) {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrength countDimensions too large and would cause out of memory condition");
      return Error_OutOfMemory;
   }
   size_t cDimensions = static_cast<size_t>(countDimensions);

   InteractionCore * const pInteractionCore = pInteractionShell->GetInteractionCore();

   // TODO : we NEVER use the hessian term (currently) in GradientPair when calculating interaction scores, but we're spending time calculating 
   // it, and it's taking up precious memory.  We should eliminate the hessian term HERE in our datastructures OR we should think whether we can 
   // use the hessian as part of the gain function!!!

   size_t aiFeatures[k_cDimensionsMax];
   size_t acBins[k_cDimensionsMax];

   const Feature * const aFeatures = pInteractionCore->GetFeatures();
   const IntEbm countFeatures = static_cast<IntEbm>(pInteractionCore->GetCountFeatures());

   // situations with 0 dimensions should have been filtered out before this function was called (but still inside the C++)
   EBM_ASSERT(1 <= cDimensions);

   size_t iDimension = 0;
   size_t cAuxillaryBinsForBuildFastTotals = 0;
   size_t cTensorBins = 1;
   do {
      const IntEbm indexFeature = featureIndexes[iDimension];
      if(indexFeature < IntEbm { 0 }) {
         LOG_0(Trace_Error, "ERROR CalcInteractionStrength featureIndexes value cannot be negative");
         return Error_IllegalParamVal;
      }
      if(countFeatures <= indexFeature) {
         LOG_0(Trace_Error, "ERROR CalcInteractionStrength featureIndexes value must be less than the number of features");
         return Error_IllegalParamVal;
      }
      const size_t iFeature = static_cast<size_t>(indexFeature);
      const Feature * const pFeature = &aFeatures[iFeature];
      aiFeatures[iDimension] = iFeature;

      const size_t cBins = pFeature->GetCountBins();
      if(cBins <= size_t { 1 }) {
         LOG_0(Trace_Info, "INFO CalcInteractionStrength feature group contains a feature with only 1 bin");
         if(nullptr != avgInteractionStrengthOut) {
            *avgInteractionStrengthOut = 0.0;
         }
         return Error_None;
      }
      acBins[iDimension] = cBins;

      // if cBins could be 1, then we'd need to check at runtime for overflow of cAuxillaryBinsForBuildFastTotals
      // if this wasn't true then we'd have to check IsAddError(cAuxillaryBinsForBuildFastTotals, cTensorBins) at runtime
      EBM_ASSERT(cAuxillaryBinsForBuildFastTotals < cTensorBins);
      // since cBins must be 2 or more, cAuxillaryBinsForBuildFastTotals must grow slower than cTensorBins, and we checked at allocation 
      // that cTensorBins would not overflow
      EBM_ASSERT(!IsAddError(cAuxillaryBinsForBuildFastTotals, cTensorBins));
      // this can overflow, but if it does then we're guaranteed to catch the overflow via the multiplication check below
      cAuxillaryBinsForBuildFastTotals += cTensorBins;
      if(IsMultiplyError(cTensorBins, cBins)) {
         // unlike in the boosting code where we check at allocation time if the tensor created overflows on multiplication
         // we don't know what group of features our caller will give us for calculating the interaction scores,
         // so we need to check if our caller gave us a tensor that overflows multiplication
         LOG_0(Trace_Warning, "WARNING CalcInteractionStrength IsMultiplyError(cTensorBins, cBins)");
         return Error_OutOfMemory;
      }
      cTensorBins *= cBins;
      // if this wasn't true then we'd have to check IsAddError(cAuxillaryBinsForBuildFastTotals, cTensorBins) at runtime
      EBM_ASSERT(cAuxillaryBinsForBuildFastTotals < cTensorBins);

      ++iDimension;
   } while(cDimensions != iDimension);

   if(size_t { 0 } == pInteractionCore->GetDataSetInteraction()->GetCountSamples()) {
      // if there are zero samples, there isn't much basis to say whether there are interactions, so just return zero
      LOG_0(Trace_Info, "INFO CalcInteractionStrength zero samples");
      if(nullptr != avgInteractionStrengthOut) {
         *avgInteractionStrengthOut = 0.0;
      }
      return Error_None;
   }

   const ptrdiff_t cClasses = pInteractionCore->GetCountClasses();

   // cClasses cannot be zero if there is 1 or more samples
   EBM_ASSERT(ptrdiff_t { 0 } != cClasses);

   if(ptrdiff_t { 1 } == cClasses) {
      LOG_0(Trace_Info, "INFO CalcInteractionStrength target with 1 class perfectly predicts the target");
      if(nullptr != avgInteractionStrengthOut) {
         *avgInteractionStrengthOut = 0.0;
      }
      return Error_None;
   }

   const bool bClassification = IsClassification(cClasses);
   const size_t cScores = GetCountScores(cClasses);

   EBM_ASSERT(!IsOverflowBinSize<FloatFast>(bClassification, cScores)); // checked in CreateInteractionDetector
   const size_t cBytesPerFastBin = GetBinSize<FloatFast>(bClassification, cScores);
   if(IsMultiplyError(cBytesPerFastBin, cTensorBins)) {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrength IsMultiplyError(cBytesPerBin, cTensorBins)");
      return Error_OutOfMemory;
   }

   // this doesn't need to be freed since it's tracked and re-used by the class InteractionShell
   BinBase * const aFastBins = pInteractionShell->GetInteractionFastBinsTemp(cBytesPerFastBin, cTensorBins);
   if(UNLIKELY(nullptr == aFastBins)) {
      // already logged
      return Error_OutOfMemory;
   }
   aFastBins->Zero(cBytesPerFastBin, cTensorBins);

#ifndef NDEBUG
   const unsigned char * const pDebugFastBinsEnd = 
      reinterpret_cast<unsigned char *>(aFastBins) + cBytesPerFastBin * cTensorBins;
   pInteractionShell->SetDebugFastBinsEnd(pDebugFastBinsEnd);
#endif // NDEBUG

   BinSumsInteraction(pInteractionShell, cDimensions, aiFeatures, acBins);

   constexpr size_t cAuxillaryBinsForSplitting = 4;
   const size_t cAuxillaryBins = EbmMax(cAuxillaryBinsForBuildFastTotals, cAuxillaryBinsForSplitting);
   
   if(IsAddError(cTensorBins, cAuxillaryBins)) {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrength IsAddError(cTensorBins, cAuxillaryBins)");
      return Error_OutOfMemory;
   }
   const size_t cTotalBigBins = cTensorBins + cAuxillaryBins;

   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // checked in CreateInteractionDetector
   const size_t cBytesPerBigBin = GetBinSize<FloatBig>(bClassification, cScores);
   if(IsMultiplyError(cBytesPerBigBin, cTotalBigBins)) {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrength IsMultiplyError(cBytesPerBin, cTotalBigBins)");
      return Error_OutOfMemory;
   }

   BinBase * const aBigBins = pInteractionShell->GetInteractionBigBins(cBytesPerBigBin, cTotalBigBins);
   if(UNLIKELY(nullptr == aBigBins)) {
      // already logged
      return Error_OutOfMemory;
   }

#ifndef NDEBUG
   const unsigned char * const pDebugBigBinsEnd = 
      reinterpret_cast<unsigned char *>(aBigBins) + cBytesPerBigBin * cTotalBigBins;
#endif // NDEBUG

   // TODO: put this into it's own function that converts our fast floats to big floats
   EBM_ASSERT(cBytesPerBigBin == cBytesPerFastBin);
   memcpy(aBigBins, aFastBins, cBytesPerFastBin * cTensorBins);



   // TODO: we can exit here back to python to allow caller modification to our bins



#ifndef NDEBUG
   // make a copy of the original bins for debugging purposes
   BinBase * const aDebugCopyBins = EbmMalloc<BinBase>(cTensorBins, cBytesPerBigBin);
   if(nullptr != aDebugCopyBins) {
      // if we can't allocate, don't fail.. just stop checking
      memcpy(aDebugCopyBins, aBigBins, cTensorBins * cBytesPerBigBin);
   }
#endif // NDEBUG

   BinBase * aAuxiliaryBins = IndexBin(aBigBins, cBytesPerBigBin * cTensorBins);
   aAuxiliaryBins->Zero(cBytesPerBigBin, cAuxillaryBins);

   TensorTotalsBuild(
      cClasses,
      cDimensions,
      acBins,
      aAuxiliaryBins,
      aBigBins
#ifndef NDEBUG
      , aDebugCopyBins
      , pDebugBigBinsEnd
#endif // NDEBUG
   );

   if(2 == cDimensions) {
      LOG_0(Trace_Verbose, "CalcInteractionStrength Starting bin sweep loop");

      double bestGain = PartitionTwoDimensionalInteraction(
         pInteractionCore,
         cDimensions,
         acBins,
         flags,
         cSamplesLeafMin,
         aAuxiliaryBins,
         aBigBins
#ifndef NDEBUG
         , aDebugCopyBins
         , pDebugBigBinsEnd
#endif // NDEBUG
      );

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

      if(nullptr != avgInteractionStrengthOut) {
         *avgInteractionStrengthOut = bestGain;
      }

      EBM_ASSERT(k_illegalGainDouble == bestGain || double { 0 } <= bestGain);
      LOG_COUNTED_N(
         pInteractionShell->GetPointerCountLogExitMessages(),
         Trace_Info,
         Trace_Verbose,
         "Exited CalcInteractionStrength: "
         "bestGain=%le"
         ,
         bestGain
      );
   } else {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrength We only support pairs for interaction detection currently");

      // TODO: handle interaction detection for higher dimensions

      // for now, just return any interactions that have other than 2 dimensions as k_illegalGainDouble, 
      // which means they won't be considered but indicates they were not handled
   }

#ifndef NDEBUG
   free(aDebugCopyBins);
#endif // NDEBUG

   return Error_None;
}

} // DEFINED_ZONE_NAME
