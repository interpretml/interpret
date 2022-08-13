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

extern void BinSumsInteraction(
   InteractionShell * const pInteractionShell,
   const size_t cSignificantDimensions,
   const size_t * const aiFeatures,
   const size_t * const acBins
);

extern void TensorTotalsBuild(
   const ptrdiff_t cClasses,
   const size_t cSignificantDimensions,
   const size_t * const acBins,
   BinBase * aAuxiliaryBinsBase,
   BinBase * const aBinsBase
#ifndef NDEBUG
   , BinBase * const aBinsBaseDebugCopy
   , const unsigned char * const pBinsEndDebug
#endif // NDEBUG
);

extern double PartitionTwoDimensionalInteraction(
   InteractionCore * const pInteractionCore,
   const size_t cSignificantDimensions,
   const size_t * const acBins,
   const InteractionFlags flags,
   const size_t cSamplesLeafMin,
   BinBase * aAuxiliaryBinsBase,
   BinBase * const aBinsBase
#ifndef NDEBUG
   , const BinBase * const aBinsBaseDebugCopy
   , const unsigned char * const pBinsEndDebug
#endif // NDEBUG
);

static ErrorEbm CalcInteractionStrengthInternal(
   InteractionShell * const pInteractionShell,
   const size_t cSignificantDimensions,
   const size_t * const aiFeatures,
   const size_t * const acBins,
   const InteractionFlags flags,
   const size_t cSamplesLeafMin,
   double * const pInteractionStrengthAvgOut
) {
   // TODO : we NEVER use the hessian term (currently) in GradientPair when calculating interaction scores, but we're spending time calculating 
   // it, and it's taking up precious memory.  We should eliminate the hessian term HERE in our datastructures OR we should think whether we can 
   // use the hessian as part of the gain function!!!

   InteractionCore * const pInteractionCore = pInteractionShell->GetInteractionCore();
   const ptrdiff_t cClasses = pInteractionCore->GetCountClasses();
   const bool bClassification = IsClassification(cClasses);

   LOG_0(Trace_Verbose, "Entered CalcInteractionStrengthInternal");

   // situations with 0 dimensions should have been filtered out before this function was called (but still inside the C++)
   EBM_ASSERT(1 <= cSignificantDimensions);

   const size_t * pcBins = acBins;
   const size_t * const pcBinsEnd = &acBins[cSignificantDimensions];

   size_t cAuxillaryBinsForBuildFastTotals = 0;
   size_t cTotalBinsMainSpace = 1;
   do {
      const size_t cBins = *pcBins;
      // situations with 1 bin should have been filtered out before this function was called (but still inside the C++)
      // our tensor code strips out features with 1 bin, and we'd need to do that here too if cBins was 1
      EBM_ASSERT(size_t { 2 } <= cBins);
      // if cBins could be 1, then we'd need to check at runtime for overflow of cAuxillaryBinsForBuildFastTotals
      // if this wasn't true then we'd have to check IsAddError(cAuxillaryBinsForBuildFastTotals, cTotalBinsMainSpace) at runtime
      EBM_ASSERT(cAuxillaryBinsForBuildFastTotals < cTotalBinsMainSpace);
      // since cBins must be 2 or more, cAuxillaryBinsForBuildFastTotals must grow slower than cTotalBinsMainSpace, and we checked at allocation 
      // that cTotalBinsMainSpace would not overflow
      EBM_ASSERT(!IsAddError(cAuxillaryBinsForBuildFastTotals, cTotalBinsMainSpace));
      // this can overflow, but if it does then we're guaranteed to catch the overflow via the multiplication check below
      cAuxillaryBinsForBuildFastTotals += cTotalBinsMainSpace;
      if(IsMultiplyError(cTotalBinsMainSpace, cBins)) {
         // unlike in the boosting code where we check at allocation time if the tensor created overflows on multiplication
         // we don't know what group of features our caller will give us for calculating the interaction scores,
         // so we need to check if our caller gave us a tensor that overflows multiplication
         LOG_0(Trace_Warning, "WARNING CalcInteractionStrengthInternal IsMultiplyError(cTotalBinsMainSpace, cBins)");
         return Error_OutOfMemory;
      }
      cTotalBinsMainSpace *= cBins;
      // if this wasn't true then we'd have to check IsAddError(cAuxillaryBinsForBuildFastTotals, cTotalBinsMainSpace) at runtime
      EBM_ASSERT(cAuxillaryBinsForBuildFastTotals < cTotalBinsMainSpace);

      ++pcBins;
   } while(pcBinsEnd != pcBins);

   const size_t cScores = GetCountScores(cClasses);

   if(IsOverflowBinSize<FloatFast>(bClassification, cScores) || 
      IsOverflowBinSize<FloatBig>(bClassification, cScores)) 
   {
      LOG_0(
         Trace_Warning,
         "WARNING CalcInteractionStrengthInternal IsOverflowBinSize overflow"
      );
      return Error_OutOfMemory;
   }
   const size_t cBytesPerBinFast = GetBinSize<FloatFast>(bClassification, cScores);
   if(IsMultiplyError(cBytesPerBinFast, cTotalBinsMainSpace)) {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrengthInternal IsMultiplyError(cBytesPerBin, cTotalBinsMainSpace)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBufferFast = cBytesPerBinFast * cTotalBinsMainSpace;

   // this doesn't need to be freed since it's tracked and re-used by the class InteractionShell
   BinBase * const aBinsFast = pInteractionShell->GetBinBaseFast(cBytesBufferFast);
   if(UNLIKELY(nullptr == aBinsFast)) {
      // already logged
      return Error_OutOfMemory;
   }
   aBinsFast->Zero(cBytesPerBinFast, cTotalBinsMainSpace);

#ifndef NDEBUG
   const unsigned char * const pBinsFastEndDebug = reinterpret_cast<unsigned char *>(aBinsFast) + cBytesBufferFast;
   pInteractionShell->SetBinsFastEndDebug(pBinsFastEndDebug);
#endif // NDEBUG

   BinSumsInteraction(pInteractionShell, cSignificantDimensions, aiFeatures, acBins);

   const size_t cAuxillaryBinsForSplitting = 4;
   const size_t cAuxillaryBins =
      cAuxillaryBinsForBuildFastTotals < cAuxillaryBinsForSplitting ? cAuxillaryBinsForSplitting : cAuxillaryBinsForBuildFastTotals;
   if(IsAddError(cTotalBinsMainSpace, cAuxillaryBins)) {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrengthInternal IsAddError(cTotalBinsMainSpace, cAuxillaryBins)");
      return Error_OutOfMemory;
   }

   const size_t cTotalBinsBig = cTotalBinsMainSpace + cAuxillaryBins;

   const size_t cBytesPerBinBig = GetBinSize<FloatBig>(bClassification, cScores);
   if(IsMultiplyError(cBytesPerBinBig, cTotalBinsBig)) {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrengthInternal IsMultiplyError(cBytesPerBin, cTotalBinsBig)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBufferBig = cBytesPerBinBig * cTotalBinsBig;

   BinBase * const aBinsBig = pInteractionShell->GetBinBaseBig(cBytesBufferBig);
   if(UNLIKELY(nullptr == aBinsBig)) {
      // already logged
      return Error_OutOfMemory;
   }
   aBinsBig->Zero(cBytesPerBinBig, cAuxillaryBins, cTotalBinsMainSpace);

#ifndef NDEBUG
   const unsigned char * const pBinsBigEndDebug = reinterpret_cast<unsigned char *>(aBinsBig) + cBytesBufferBig;
#endif // NDEBUG

   // TODO: put this into it's own function that converts our fast floats to big floats
   static_assert(sizeof(FloatBig) == sizeof(FloatFast), "float mismatch");
   memcpy(aBinsBig, aBinsFast, cBytesBufferFast);


   // TODO: we can exit here back to python to allow caller modification to our bins


#ifndef NDEBUG
   // make a copy of the original bins for debugging purposes
   BinBase * const aBinsDebugCopy =
      EbmMalloc<BinBase>(cTotalBinsMainSpace, cBytesPerBinBig);
   if(nullptr != aBinsDebugCopy) {
      // if we can't allocate, don't fail.. just stop checking
      const size_t cBytesBufferDebug = cTotalBinsMainSpace * cBytesPerBinBig;
      memcpy(aBinsDebugCopy, aBinsBig, cBytesBufferDebug);
   }
#endif // NDEBUG

   BinBase * aAuxiliaryBins =
      IndexBin(cBytesPerBinBig, aBinsBig, cTotalBinsMainSpace);

   TensorTotalsBuild(
      cClasses,
      cSignificantDimensions,
      acBins,
      aAuxiliaryBins,
      aBinsBig
#ifndef NDEBUG
      , aBinsDebugCopy
      , pBinsBigEndDebug
#endif // NDEBUG
   );

   if(2 == cSignificantDimensions) {
      LOG_0(Trace_Verbose, "CalcInteractionStrengthInternal Starting bin sweep loop");

      double bestGain = PartitionTwoDimensionalInteraction(
         pInteractionCore,
         cSignificantDimensions,
         acBins,
         flags,
         cSamplesLeafMin,
         aAuxiliaryBins,
         aBinsBig
#ifndef NDEBUG
         , aBinsDebugCopy
         , pBinsBigEndDebug
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
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrengthInternal 2 != cSignificantDimensions");

      // TODO: handle this better
      if(nullptr != pInteractionStrengthAvgOut) {
         // for now, just return any interactions that have other than 2 dimensions as -inf, 
         // which means they won't be considered but indicates they were not handled
         *pInteractionStrengthAvgOut = k_illegalGainDouble;
      }
   }

#ifndef NDEBUG
   free(aBinsDebugCopy);
#endif // NDEBUG

   LOG_0(Trace_Verbose, "Exited CalcInteractionStrengthInternal");
   return Error_None;
}

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

   ErrorEbm error;

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

   size_t aiFeatures[k_cDimensionsMax];
   size_t acBins[k_cDimensionsMax];

   InteractionCore * const pInteractionCore = pInteractionShell->GetInteractionCore();
   const Feature * const aFeatures = pInteractionCore->GetFeatures();
   size_t cTensorBins = 1;
   size_t iDimension = 0;
   do {
      // TODO: merge this loop with the one below inside the internal function

      const IntEbm indexFeature = featureIndexes[iDimension];
      if(indexFeature < IntEbm { 0 }) {
         LOG_0(Trace_Error, "ERROR CalcInteractionStrength featureIndexes value cannot be negative");
         return Error_IllegalParamVal;
      }
      if(static_cast<IntEbm>(pInteractionCore->GetCountFeatures()) <= indexFeature) {
         LOG_0(Trace_Error, "ERROR CalcInteractionStrength featureIndexes value must be less than the number of features");
         return Error_IllegalParamVal;
      }
      const size_t iFeature = static_cast<size_t>(indexFeature);
      const Feature * const pFeature = &aFeatures[iFeature];
      const size_t cBins = pFeature->GetCountBins();
      if(cBins <= size_t { 1 }) {
         LOG_0(Trace_Info, "INFO CalcInteractionStrength feature group contains a feature with only 1 bin");
         if(nullptr != avgInteractionStrengthOut) {
            *avgInteractionStrengthOut = double { 0 };
         }
         return Error_None;
      }
      if(IsMultiplyError(cTensorBins, cBins)) {
         LOG_0(Trace_Warning, "WARNING CalcInteractionStrength IsMultiplyError(cTensorBins, cBins)");
         return Error_OutOfMemory;
      }
      cTensorBins *= cBins; // TODO: Do I need this?  Maybe we just use it as a check for overflow, but then shouldn't we be calculating the total memory bytes?

      aiFeatures[iDimension] = iFeature;
      acBins[iDimension] = cBins;

      ++iDimension;
   } while(cDimensions != iDimension);

   if(size_t { 0 } == pInteractionCore->GetDataSetInteraction()->GetCountSamples()) {
      // if there are zero samples, there isn't much basis to say whether there are interactions, so just return zero
      LOG_0(Trace_Info, "INFO CalcInteractionStrength zero samples");
      if(nullptr != avgInteractionStrengthOut) {
         *avgInteractionStrengthOut = double { 0 };
      }
      return Error_None;
   }
   // GetCountClasses cannot be zero if there is 1 or more samples
   EBM_ASSERT(ptrdiff_t { 0 } != pInteractionCore->GetCountClasses());

   if(ptrdiff_t { 1 } == pInteractionCore->GetCountClasses()) {
      LOG_0(Trace_Info, "INFO CalcInteractionStrength target with 1 class perfectly predicts the target");
      if(nullptr != avgInteractionStrengthOut) {
         *avgInteractionStrengthOut = double { 0 };
      }
      return Error_None;
   }

   error = CalcInteractionStrengthInternal(
      pInteractionShell,
      cDimensions,
      aiFeatures,
      acBins,
      flags,
      cSamplesLeafMin,
      avgInteractionStrengthOut
   );
   if(Error_None != error) {
      LOG_N(Trace_Warning, "WARNING CalcInteractionStrength: return=%" ErrorEbmPrintf, error);
      return error;
   }

   if(nullptr != avgInteractionStrengthOut) {
      EBM_ASSERT(k_illegalGainDouble == *avgInteractionStrengthOut || double { 0 } <= *avgInteractionStrengthOut);
      LOG_COUNTED_N(
         pInteractionShell->GetPointerCountLogExitMessages(),
         Trace_Info,
         Trace_Verbose,
         "Exited CalcInteractionStrength: "
         "*avgInteractionStrengthOut=%le"
         , 
         *avgInteractionStrengthOut
      );
   } else {
      LOG_COUNTED_0(
         pInteractionShell->GetPointerCountLogExitMessages(),
         Trace_Info, 
         Trace_Verbose, 
         "Exited CalcInteractionStrength"
      );
   }
   return Error_None;
}

} // DEFINED_ZONE_NAME
