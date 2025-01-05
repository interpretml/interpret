// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <string.h> // memcpy

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "unzoned.h" // FloatMain

#define ZONE_main
#include "zones.h"

#include "Bin.hpp" // GetBinSize

#include "ebm_internal.hpp" // k_cDimensionsMax
#include "Feature.hpp"
#include "DataSetInteraction.hpp"
#include "Tensor.hpp"
#include "TreeNodeMulti.hpp"
#include "InteractionCore.hpp"
#include "InteractionShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern void ConvertAddBin(const size_t cScores,
      const bool bHessian,
      const size_t cBins,
      const bool bUInt64Src,
      const bool bDoubleSrc,
      const bool bCountSrc,
      const bool bWeightSrc,
      const void* const aSrc,
      const UIntMain* const aCounts,
      const FloatPrecomp* const aWeights,
      const bool bUInt64Dest,
      const bool bDoubleDest,
      void* const aAddDest);

extern void TensorTotalsBuild(const bool bHessian,
      const size_t cScores,
      const size_t cRealDimensions,
      const size_t* const acBins,
      BinBase* aAuxiliaryBinsBase,
      BinBase* const aBinsBase
#ifndef NDEBUG
      ,
      BinBase* const aDebugCopyBinsBase,
      const BinBase* const pBinsEndDebug
#endif // NDEBUG
);

extern double PartitionMultiDimensionalStraight(InteractionCore* const pInteractionCore,
      const size_t cRealDimensions,
      const size_t* const acBins,
      const CalcInteractionFlags flags,
      const size_t cSamplesLeafMin,
      const FloatCalc hessianMin,
      const FloatCalc regAlpha,
      const FloatCalc regLambda,
      const FloatCalc deltaStepMax,
      BinBase* aAuxiliaryBinsBase,
      BinBase* const aBinsBase
#ifndef NDEBUG
      ,
      const BinBase* const aDebugCopyBinsBase,
      const BinBase* const pBinsEndDebug
#endif // NDEBUG
);

extern ErrorEbm PartitionMultiDimensionalTree(const bool bHessian,
      const size_t cRuntimeScores,
      const size_t cDimensions,
      const size_t cRealDimensions,
      const TermBoostFlags flags,
      const size_t cSamplesLeafMin,
      const FloatCalc hessianMin,
      const FloatCalc regAlpha,
      const FloatCalc regLambda,
      const FloatCalc deltaStepMax,
      const BinBase* const aBinsBase,
      BinBase* const aAuxiliaryBinsBase,
      Tensor* const pInnerTermUpdate,
      void* const pRootTreeNodeBase,
      const size_t* const acBins,
      double* const aTensorWeights,
      double* const aTensorGrad,
      double* const aTensorHess,
      double* const pTotalGain,
      const size_t cPossibleSplits,
      void* const pTemp1
#ifndef NDEBUG
      ,
      const BinBase* const aDebugCopyBinsBase,
      const BinBase* const pBinsEndDebug
#endif // NDEBUG
);

// there is a race condition for decrementing this variable, but if a thread loses the
// race then it just doesn't get decremented as quickly, which we can live with
static int g_cLogCalcInteractionStrength = 10;

EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION CalcInteractionStrength(InteractionHandle interactionHandle,
      IntEbm countDimensions,
      const IntEbm* featureIndexes,
      CalcInteractionFlags flags,
      IntEbm maxCardinality,
      IntEbm minSamplesLeaf,
      double minHessian,
      double regAlpha,
      double regLambda,
      double maxDeltaStep,
      double* avgInteractionStrengthOut) {
   LOG_COUNTED_N(&g_cLogCalcInteractionStrength,
         Trace_Info,
         Trace_Verbose,
         "CalcInteractionStrength: "
         "interactionHandle=%p, "
         "countDimensions=%" IntEbmPrintf ", "
         "featureIndexes=%p, "
         "flags=0x%" UCalcInteractionFlagsPrintf ", "
         "maxCardinality=%" IntEbmPrintf ", "
         "minSamplesLeaf=%" IntEbmPrintf ", "
         "minHessian=%le, "
         "regAlpha=%le, "
         "regLambda=%le, "
         "maxDeltaStep=%le, "
         "avgInteractionStrengthOut=%p",
         static_cast<void*>(interactionHandle),
         countDimensions,
         static_cast<const void*>(featureIndexes),
         static_cast<UCalcInteractionFlags>(flags), // signed to unsigned conversion is defined behavior in C++
         maxCardinality,
         minSamplesLeaf,
         minHessian,
         regAlpha,
         regLambda,
         maxDeltaStep,
         static_cast<void*>(avgInteractionStrengthOut));

   ErrorEbm error;

   if(LIKELY(nullptr != avgInteractionStrengthOut)) {
      *avgInteractionStrengthOut = k_illegalGainDouble;
   }

   InteractionShell* const pInteractionShell = InteractionShell::GetInteractionShellFromHandle(interactionHandle);
   if(nullptr == pInteractionShell) {
      // already logged
      return Error_IllegalParamVal;
   }
   LOG_COUNTED_0(pInteractionShell->GetPointerCountLogEnterMessages(),
         Trace_Info,
         Trace_Verbose,
         "Entered CalcInteractionStrength");

   if(flags & ~(CalcInteractionFlags_DisableNewton | CalcInteractionFlags_Purify)) {
      LOG_0(Trace_Error, "ERROR CalcInteractionStrength flags contains unknown flags. Ignoring extras.");
   }

   size_t cCardinalityMax = std::numeric_limits<size_t>::max(); // set off by default
   if(IntEbm{0} <= maxCardinality) {
      if(IntEbm{0} != maxCardinality) {
         if(!IsConvertError<size_t>(maxCardinality)) {
            // we can never exceed a size_t number of samples, so let's just set it to the maximum if we were going to
            // overflow because it will generate the same results as if we used the true number
            cCardinalityMax = static_cast<size_t>(maxCardinality);
         }
      }
   } else {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrength maxCardinality can't be less than 0. Turning off.");
   }

   size_t cSamplesLeafMin = size_t{1}; // this is the min value
   if(IntEbm{1} <= minSamplesLeaf) {
      cSamplesLeafMin = static_cast<size_t>(minSamplesLeaf);
      if(IsConvertError<size_t>(minSamplesLeaf)) {
         // we can never exceed a size_t number of samples, so let's just set it to the maximum if we were going to
         // overflow because it will generate the same results as if we used the true number
         cSamplesLeafMin = std::numeric_limits<size_t>::max();
      }
   } else {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrength minSamplesLeaf can't be less than 1. Adjusting to 1.");
   }

   FloatCalc hessianMin = static_cast<FloatCalc>(minHessian);
   if(/* NaN */ !(std::numeric_limits<FloatCalc>::min() <= hessianMin)) {
      hessianMin = std::numeric_limits<FloatCalc>::min();
      if(/* NaN */ !(double{0} <= minHessian)) {
         LOG_0(Trace_Warning,
               "WARNING CalcInteractionStrength minHessian must be a positive number. Adjusting to minimum float");
      }
   }

   FloatCalc regAlphaCalc = static_cast<FloatCalc>(regAlpha);
   if(/* NaN */ !(FloatCalc{0} <= regAlphaCalc)) {
      regAlphaCalc = 0;
      LOG_0(Trace_Warning,
            "WARNING CalcInteractionStrength regAlpha must be a positive number or zero. Adjusting to 0.");
   }

   FloatCalc regLambdaCalc = static_cast<FloatCalc>(regLambda);
   if(/* NaN */ !(FloatCalc{0} <= regLambdaCalc)) {
      regLambdaCalc = 0;
      LOG_0(Trace_Warning,
            "WARNING CalcInteractionStrength regLambda must be a positive number or zero. Adjusting to 0.");
   }

   FloatCalc deltaStepMax = static_cast<FloatCalc>(maxDeltaStep);
   if(/* NaN */ !(double{0} < maxDeltaStep)) {
      // 0, negative numbers, and NaN mean turn off the max step. We use +inf to do this.
      deltaStepMax = std::numeric_limits<FloatCalc>::infinity();
   }

   if(countDimensions <= IntEbm{0}) {
      if(IntEbm{0} == countDimensions) {
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
   if(IntEbm{k_cDimensionsMax} < countDimensions) {
      LOG_0(Trace_Warning,
            "WARNING CalcInteractionStrength countDimensions too large and would cause out of memory condition");
      return Error_OutOfMemory;
   }
   size_t cDimensions = static_cast<size_t>(countDimensions);

   InteractionCore* const pInteractionCore = pInteractionShell->GetInteractionCore();

   const size_t cScores = pInteractionCore->GetCountScores();
   if(size_t{0} == cScores) {
      LOG_0(Trace_Info, "INFO CalcInteractionStrength target with 1 class perfectly predicts the target");
      if(nullptr != avgInteractionStrengthOut) {
         *avgInteractionStrengthOut = 0.0;
      }
      return Error_None;
   }

   const DataSetInteraction* const pDataSet = pInteractionCore->GetDataSetInteraction();
   EBM_ASSERT(nullptr != pDataSet);

   if(size_t{0} == pDataSet->GetCountSamples()) {
      // if there are zero samples, there isn't much basis to say whether there are interactions, so just return zero
      LOG_0(Trace_Info, "INFO CalcInteractionStrength zero samples");
      if(nullptr != avgInteractionStrengthOut) {
         *avgInteractionStrengthOut = 0.0;
      }
      return Error_None;
   }

   BinSumsInteractionBridge binSums;

   const FeatureInteraction* const aFeatures = pInteractionCore->GetFeatures();
   const IntEbm countFeatures = static_cast<IntEbm>(pInteractionCore->GetCountFeatures());

   // situations with 0 dimensions should have been filtered out before this function was called (but still inside the
   // C++)
   EBM_ASSERT(1 <= cDimensions);

   size_t iDimension = 0;
   size_t cAuxillaryBinsForBuildFastTotals = 0;
   size_t cTensorBins = 1;
   do {
      const IntEbm indexFeature = featureIndexes[iDimension];
      if(indexFeature < IntEbm{0}) {
         LOG_0(Trace_Error, "ERROR CalcInteractionStrength featureIndexes value cannot be negative");
         return Error_IllegalParamVal;
      }
      if(countFeatures <= indexFeature) {
         LOG_0(Trace_Error,
               "ERROR CalcInteractionStrength featureIndexes value must be less than the number of features");
         return Error_IllegalParamVal;
      }
      const size_t iFeature = static_cast<size_t>(indexFeature);

      const FeatureInteraction* const pFeature = &aFeatures[iFeature];

      const size_t cBins = pFeature->GetCountBins();
      if(UNLIKELY(cBins <= size_t{1})) {
         LOG_0(Trace_Info, "INFO CalcInteractionStrength term contains a feature with only 1 or 0 bins");
         if(nullptr != avgInteractionStrengthOut) {
            *avgInteractionStrengthOut = 0.0;
         }
         return Error_None;
      }
      binSums.m_acBins[iDimension] = cBins;

      // if cBins could be 1, then we'd need to check at runtime for overflow of cAuxillaryBinsForBuildFastTotals
      // if this wasn't true then we'd have to check IsAddError(cAuxillaryBinsForBuildFastTotals, cTensorBins) at
      // runtime
      EBM_ASSERT(0 == cTensorBins || cAuxillaryBinsForBuildFastTotals < cTensorBins);
      // since cBins must be 2 or more, cAuxillaryBinsForBuildFastTotals must grow slower than cTensorBins, and we
      // checked at allocation that cTensorBins would not overflow
      EBM_ASSERT(!IsAddError(cAuxillaryBinsForBuildFastTotals, cTensorBins));
      // this can overflow, but if it does then we're guaranteed to catch the overflow via the multiplication check
      // below
      cAuxillaryBinsForBuildFastTotals += cTensorBins;
      if(IsMultiplyError(cTensorBins, cBins)) {
         // unlike in the boosting code where we check at allocation time if the tensor created overflows on
         // multiplication we don't know what group of features our caller will give us for calculating the interaction
         // scores, so we need to check if our caller gave us a tensor that overflows multiplication if we overflow
         // this, then we'd be above the cCardinalityMax value, so set it to 0.0
         LOG_0(Trace_Info, "INFO CalcInteractionStrength IsMultiplyError(cTensorBins, cBins)");
         if(nullptr != avgInteractionStrengthOut) {
            *avgInteractionStrengthOut = 0.0;
         }
         return Error_None;
      }
      cTensorBins *= cBins;
      // if this wasn't true then we'd have to check IsAddError(cAuxillaryBinsForBuildFastTotals, cTensorBins) at
      // runtime
      EBM_ASSERT(0 == cTensorBins || cAuxillaryBinsForBuildFastTotals < cTensorBins);

      ++iDimension;
   } while(cDimensions != iDimension);

   if(cCardinalityMax < cTensorBins) {
      LOG_0(Trace_Info, "INFO CalcInteractionStrength cCardinalityMax < cTensorBins");
      if(nullptr != avgInteractionStrengthOut) {
         *avgInteractionStrengthOut = 0.0;
      }
      return Error_None;
   }

   static constexpr size_t cAuxillaryBinsForSplitting = 4;
   const size_t cAuxillaryBins = EbmMax(cAuxillaryBinsForBuildFastTotals, cAuxillaryBinsForSplitting);

   if(IsAddError(cTensorBins, cAuxillaryBins)) {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrength IsAddError(cTensorBins, cAuxillaryBins)");
      return Error_OutOfMemory;
   }
   const size_t cTotalMainBins = cTensorBins + cAuxillaryBins;

   const size_t cBytesPerMainBin = GetBinSize<FloatMain, UIntMain>(true, true, pInteractionCore->IsHessian(), cScores);
   if(IsMultiplyError(cBytesPerMainBin, cTotalMainBins)) {
      LOG_0(Trace_Warning, "WARNING CalcInteractionStrength IsMultiplyError(cBytesPerBin, cTotalMainBins)");
      return Error_OutOfMemory;
   }

   BinBase* const aMainBins = pInteractionShell->GetInteractionMainBins(cBytesPerMainBin, cTotalMainBins);
   if(UNLIKELY(nullptr == aMainBins)) {
      // already logged
      return Error_OutOfMemory;
   }

#ifndef NDEBUG
   const auto* const pDebugMainBinsEnd = IndexBin(aMainBins, cBytesPerMainBin * cTotalMainBins);
#endif // NDEBUG

   memset(aMainBins, 0, cBytesPerMainBin * cTensorBins);

   const bool bHessian = pInteractionCore->IsHessian();

   EBM_ASSERT(1 <= pInteractionCore->GetDataSetInteraction()->GetCountSubsets());
   DataSubsetInteraction* pSubset = pInteractionCore->GetDataSetInteraction()->GetSubsets();
   const DataSubsetInteraction* const pSubsetsEnd =
         pSubset + pInteractionCore->GetDataSetInteraction()->GetCountSubsets();
   do {
      size_t cBytesPerFastBin;
      if(sizeof(UIntBig) == pSubset->GetObjectiveWrapper()->m_cUIntBytes) {
         if(sizeof(FloatBig) == pSubset->GetObjectiveWrapper()->m_cFloatBytes) {
            cBytesPerFastBin = GetBinSize<FloatBig, UIntBig>(true, true, bHessian, cScores);
         } else {
            EBM_ASSERT(sizeof(FloatSmall) == pSubset->GetObjectiveWrapper()->m_cFloatBytes);
            cBytesPerFastBin = GetBinSize<FloatSmall, UIntBig>(true, true, bHessian, cScores);
         }
      } else {
         EBM_ASSERT(sizeof(UIntSmall) == pSubset->GetObjectiveWrapper()->m_cUIntBytes);
         if(sizeof(FloatBig) == pSubset->GetObjectiveWrapper()->m_cFloatBytes) {
            cBytesPerFastBin = GetBinSize<FloatBig, UIntSmall>(true, true, bHessian, cScores);
         } else {
            EBM_ASSERT(sizeof(FloatSmall) == pSubset->GetObjectiveWrapper()->m_cFloatBytes);
            cBytesPerFastBin = GetBinSize<FloatSmall, UIntSmall>(true, true, bHessian, cScores);
         }
      }
      if(IsMultiplyError(cBytesPerFastBin, cTensorBins)) {
         LOG_0(Trace_Warning, "WARNING CalcInteractionStrength IsMultiplyError(cBytesPerBin, cTensorBins)");
         return Error_OutOfMemory;
      }

      // this doesn't need to be freed since it's tracked and re-used by the class InteractionShell
      BinBase* const aFastBins = pInteractionShell->GetInteractionFastBinsTemp(cBytesPerFastBin * cTensorBins);
      if(UNLIKELY(nullptr == aFastBins)) {
         // already logged
         return Error_OutOfMemory;
      }

      aFastBins->ZeroMem(cBytesPerFastBin, cTensorBins);

#ifndef NDEBUG
      binSums.m_pDebugFastBinsEnd = IndexBin(aFastBins, cBytesPerFastBin * cTensorBins);
#endif // NDEBUG

      size_t iDimensionLoop = 0;
      do {
         const IntEbm indexFeature = featureIndexes[iDimensionLoop];
         const size_t iFeature = static_cast<size_t>(indexFeature);
         const FeatureInteraction* const pFeature = &aFeatures[iFeature];

         binSums.m_aaPacked[iDimensionLoop] = pSubset->GetFeatureData(iFeature);

         EBM_ASSERT(1 <= pFeature->GetBitsRequiredMin());
         binSums.m_acItemsPerBitPack[iDimensionLoop] =
               GetCountItemsBitPacked(pFeature->GetBitsRequiredMin(), pSubset->GetObjectiveWrapper()->m_cUIntBytes);

         ++iDimensionLoop;
      } while(cDimensions != iDimensionLoop);

      binSums.m_cRuntimeRealDimensions = cDimensions;

      binSums.m_bHessian = pInteractionCore->IsHessian() ? EBM_TRUE : EBM_FALSE;
      binSums.m_cScores = cScores;

      binSums.m_cSamples = pSubset->GetCountSamples();
      binSums.m_aGradientsAndHessians = pSubset->GetGradHess();
      binSums.m_aWeights = pSubset->GetWeights();

      binSums.m_aFastBins = aFastBins;

      error = pSubset->BinSumsInteraction(&binSums);
      if(Error_None != error) {
         return error;
      }

      ConvertAddBin(cScores,
            pInteractionCore->IsHessian(),
            cTensorBins,
            sizeof(UIntBig) == pSubset->GetObjectiveWrapper()->m_cUIntBytes,
            sizeof(FloatBig) == pSubset->GetObjectiveWrapper()->m_cFloatBytes,
            true,
            true,
            aFastBins,
            nullptr,
            nullptr,
            std::is_same<UIntMain, uint64_t>::value,
            std::is_same<FloatMain, double>::value,
            aMainBins);

      ++pSubset;
   } while(pSubsetsEnd != pSubset);

   // TODO: we can exit here back to python to allow caller modification to our bins

#ifndef NDEBUG
   // make a copy of the original bins for debugging purposes

   BinBase* aDebugCopyBins = nullptr;
   if(!IsMultiplyError(cBytesPerMainBin, cTensorBins)) {
      ANALYSIS_ASSERT(0 != cBytesPerMainBin);
      ANALYSIS_ASSERT(1 <= cTensorBins);
      aDebugCopyBins = static_cast<BinBase*>(malloc(cBytesPerMainBin * cTensorBins));
      if(nullptr != aDebugCopyBins) {
         // if we can't allocate, don't fail.. just stop checking
         memcpy(aDebugCopyBins, aMainBins, cTensorBins * cBytesPerMainBin);
      }
   }
#endif // NDEBUG

   BinBase* aAuxiliaryBins = IndexBin(aMainBins, cBytesPerMainBin * cTensorBins);
   aAuxiliaryBins->ZeroMem(cBytesPerMainBin, cAuxillaryBins);

   TensorTotalsBuild(pInteractionCore->IsHessian(),
         cScores,
         cDimensions,
         binSums.m_acBins,
         aAuxiliaryBins,
         aMainBins
#ifndef NDEBUG
         ,
         aDebugCopyBins,
         pDebugMainBinsEnd
#endif // NDEBUG
   );

   double bestGain;
   if(2 == cDimensions) {
      LOG_0(Trace_Verbose, "CalcInteractionStrength Starting bin sweep loop");

      bestGain = PartitionMultiDimensionalStraight(pInteractionCore,
            cDimensions,
            binSums.m_acBins,
            flags,
            cSamplesLeafMin,
            hessianMin,
            regAlphaCalc,
            regLambdaCalc,
            deltaStepMax,
            aAuxiliaryBins,
            aMainBins
#ifndef NDEBUG
            ,
            aDebugCopyBins,
            pDebugMainBinsEnd
#endif // NDEBUG
      );
   } else {
      size_t cPossibleSplits;
      if(IsOverflowBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores)) {
         // TODO: move this to init
         return Error_OutOfMemory;
      }

      if(IsOverflowTreeNodeMultiSize(bHessian, cScores)) {
         // TODO: move this to init
         return Error_OutOfMemory;
      }

      cPossibleSplits = 0;

      size_t cBytes = 1;

      size_t* pcBins = binSums.m_acBins;
      size_t* pcBinsEnd = binSums.m_acBins + cDimensions;
      do {
         const size_t cBins = *pcBins;
         EBM_ASSERT(size_t{2} <= cBins);
         const size_t cSplits = cBins - 1;
         if(IsAddError(cPossibleSplits, cSplits)) {
            return Error_OutOfMemory;
         }
         cPossibleSplits += cSplits;
         if(IsMultiplyError(cBins, cBytes)) {
            return Error_OutOfMemory;
         }
         cBytes *= cBins;
         ++pcBins;
      } while(pcBinsEnd != pcBins);

      // For pairs, this calculates the exact max number of splits. For higher dimensions
      // the max number of splits will be less, but it should be close enough.
      // Each bin gets a tree node to record the gradient totals, and each split gets a TreeNode
      // during construction. Each split contains a minimum of 1 bin on each side, so we have
      // cBins - 1 potential splits.

      if(IsAddError(cBytes, cBytes - 1)) {
         return Error_OutOfMemory;
      }
      cBytes = cBytes + cBytes - 1;

      const size_t cBytesTreeNodeMulti = GetTreeNodeMultiSize(bHessian, cScores);

      if(IsMultiplyError(cBytesTreeNodeMulti, cBytes)) {
         return Error_OutOfMemory;
      }
      cBytes *= cBytesTreeNodeMulti;

      const size_t cBytesBest = cBytesTreeNodeMulti * (size_t{1} + (cDimensions << 1));
      EBM_ASSERT(cBytesBest <= cBytes);

      // double it because we during the multi-dimensional sweep we need the best and we need the current
      if(IsAddError(cBytesBest, cBytesBest)) {
         return Error_OutOfMemory;
      }
      const size_t cBytesSweep = cBytesBest + cBytesBest;

      cBytes = EbmMax(cBytes, cBytesSweep);

      double* aWeights = nullptr;
      double* pGradient = nullptr;
      double* pHessian = nullptr;
      void* pTreeNodesTemp = nullptr;
      void* pTemp1 = nullptr;

      if(0 != (CalcInteractionFlags_Purify & flags)) {
         // allocate the biggest tensor that is possible to split into

         // TODO: cache this memory allocation so that we don't do it each time

         if(IsAddError(size_t{1}, cScores)) {
            return Error_OutOfMemory;
         }
         size_t cItems = 1 + cScores;
         const bool bUseLogitBoost = bHessian && !(CalcInteractionFlags_DisableNewton & flags);
         if(bUseLogitBoost) {
            if(IsAddError(cScores, cItems)) {
               return Error_OutOfMemory;
            }
            cItems += cScores;
         }
         if(IsMultiplyError(sizeof(double), cItems, cTensorBins)) {
            return Error_OutOfMemory;
         }
         aWeights = static_cast<double*>(malloc(sizeof(double) * cItems * cTensorBins));
         if(nullptr == aWeights) {
            return Error_OutOfMemory;
         }
         pGradient = aWeights + cTensorBins;
         if(bUseLogitBoost) {
            pHessian = pGradient + cTensorBins * cScores;
         }
      }

      pTreeNodesTemp = malloc(cBytes);
      if(nullptr == pTreeNodesTemp) {
         free(aWeights);
         return Error_OutOfMemory;
      }

      pTemp1 = malloc(cPossibleSplits * sizeof(unsigned char));
      if(nullptr == pTemp1) {
         free(pTreeNodesTemp);
         free(aWeights);
         return Error_OutOfMemory;
      }

      Tensor* const pInnerTermUpdate = Tensor::Allocate(k_cDimensionsMax, cScores);
      if(nullptr == pInnerTermUpdate) {
         free(pTemp1);
         free(pTreeNodesTemp);
         free(aWeights);
         return Error_OutOfMemory;
      }

      error = PartitionMultiDimensionalTree(bHessian,
            cScores,
            cDimensions,
            cDimensions,
            flags,
            cSamplesLeafMin,
            hessianMin,
            regAlpha,
            regLambda,
            deltaStepMax,
            aMainBins,
            aAuxiliaryBins,
            pInnerTermUpdate,
            pTreeNodesTemp,
            binSums.m_acBins,
            aWeights,
            pGradient,
            pHessian,
            &bestGain,
            cPossibleSplits,
            pTemp1
#ifndef NDEBUG
            ,
            aDebugCopyBins,
            pDebugMainBinsEnd
#endif // NDEBUG
      );

      Tensor::Free(pInnerTermUpdate);
      free(pTemp1);
      free(pTreeNodesTemp);
      free(aWeights);

      if(Error_None != error) {
#ifndef NDEBUG
         free(aDebugCopyBins);
#endif // NDEBUG

         LOG_0(Trace_Verbose, "Exited BoostMultiDimensional with Error code");

         return error;
      }
      EBM_ASSERT(!std::isnan(bestGain));
      EBM_ASSERT(0 == bestGain || std::numeric_limits<FloatCalc>::min() <= bestGain);
   }

#ifndef NDEBUG
   free(aDebugCopyBins);
#endif // NDEBUG

   // if totalWeight < 1 then bestGain could overflow to +inf, so do the division first
   const double totalWeight = pDataSet->GetWeightTotal();
   EBM_ASSERT(0 < totalWeight); // if all are zeros we assume there are no weights and use the count
   bestGain /= totalWeight;
   if(CalcInteractionFlags_DisableNewton & flags) {
      bestGain *= pInteractionCore->GainAdjustmentGradientBoosting();
   } else {
      bestGain /= pInteractionCore->HessianConstant();
      bestGain *= pInteractionCore->GainAdjustmentHessianBoosting();
   }
   const double gradientConstant = pInteractionCore->GradientConstant();
   bestGain *= gradientConstant;
   bestGain *= gradientConstant;

   if(UNLIKELY(/* NaN */ !LIKELY(bestGain <= std::numeric_limits<double>::max()))) {
      // We simplify our caller's handling by returning -lowest as our overflow indicator. -lowest will sort to being
      // the least important item, which is good, but it also signals an overflow without the weirness of NaNs.
      EBM_ASSERT(std::isnan(bestGain) || std::numeric_limits<double>::infinity() == bestGain);
      bestGain = k_illegalGainDouble;
   } else if(UNLIKELY(bestGain < std::numeric_limits<FloatCalc>::min())) {
      // gain can't mathematically be legally negative, but it can be here in the following situations:
      //   1) for impure interaction gain we subtract the parent partial gain, and there can be floating point
      //      noise that makes this slightly negative
      //   2) for impure interaction gain we subtract the parent partial gain, but if there were no legal cuts
      //      then the partial gain before subtracting the parent partial gain was zero and we then get a
      //      substantially negative value.  In this case we should not have subtracted the parent partial gain
      //      since we had never even calculated the 4 quadrant partial gain, but we handle this scenario
      //      here instead of inside the templated function.

      EBM_ASSERT(!std::isnan(bestGain));
      // make bestGain k_illegalGainDouble if it's -infinity, otherwise make it zero
      bestGain = std::numeric_limits<double>::lowest() <= bestGain ? 0.0 : k_illegalGainDouble;
   }

   EBM_ASSERT(!std::isnan(bestGain));
   EBM_ASSERT(!std::isinf(bestGain));
   EBM_ASSERT(k_illegalGainDouble == bestGain || 0.0 == bestGain || std::numeric_limits<FloatCalc>::min() <= bestGain);

   if(nullptr != avgInteractionStrengthOut) {
      *avgInteractionStrengthOut = bestGain;
   }

   LOG_COUNTED_N(pInteractionShell->GetPointerCountLogExitMessages(),
         Trace_Info,
         Trace_Verbose,
         "Exited CalcInteractionStrength: "
         "bestGain=%le",
         bestGain);

   return Error_None;
}

} // namespace DEFINED_ZONE_NAME
