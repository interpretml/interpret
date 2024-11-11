// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "unzoned.h" // LIKELY

#define ZONE_main
#include "zones.h"

#include "GradientPair.hpp"
#include "Bin.hpp"

#include "ebm_stats.hpp"
#include "Feature.hpp"
#include "Term.hpp"
#include "Tensor.hpp"
#include "TensorTotalsSum.hpp"
#include "TreeNode.hpp"
#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<bool bHessian, size_t cCompilerScores, size_t cCompilerDimensions>
static FloatCalc SweepMultiDimensional(const size_t cRuntimeScores,
      const size_t cRuntimeRealDimensions,
      const TermBoostFlags flags,
      const size_t* const aiPoint,
      const size_t* const acBins,
      const size_t directionVectorLow,
      const size_t iDimensionSweep,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const aBins,
      const size_t cSamplesLeafMin,
      const FloatCalc hessianMin,
      const FloatCalc regAlpha,
      const FloatCalc regLambda,
      const FloatCalc deltaStepMax,
      Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const pBinBestAndTemp,
      size_t* const piBestSplit
#ifndef NDEBUG
      ,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const aDebugCopyBins,
      const BinBase* const pBinsEndDebug
#endif // NDEBUG
) {
   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, cRuntimeScores);
   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);

   const size_t cRealDimensions = GET_COUNT_DIMENSIONS(cCompilerDimensions, cRuntimeRealDimensions);
   EBM_ASSERT(1 <= cRealDimensions); // for interactions, we just return 0 for interactions with zero features
   EBM_ASSERT(iDimensionSweep < cRealDimensions);
   EBM_ASSERT(0 == (directionVectorLow & (size_t{1} << iDimensionSweep)));

   TensorSumDimension aDimensions[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
   size_t directionDestroy = directionVectorLow;
   size_t iDimensionInit = 0;
   do {
      aDimensions[iDimensionInit].m_cBins = acBins[iDimensionInit];
      if(0 != (size_t{1} & directionDestroy)) {
         aDimensions[iDimensionInit].m_iLow = aiPoint[iDimensionInit] + 1;
         aDimensions[iDimensionInit].m_iHigh = acBins[iDimensionInit];
      } else {
         aDimensions[iDimensionInit].m_iLow = 0;
         aDimensions[iDimensionInit].m_iHigh = aiPoint[iDimensionInit] + 1;
      }
      directionDestroy >>= 1;
      ++iDimensionInit;
   } while(cRealDimensions != iDimensionInit);

   const size_t cSweepCuts = aDimensions[iDimensionSweep].m_cBins - 1;
   EBM_ASSERT(1 <= cSweepCuts); // dimensions with 1 bin are removed earlier

   size_t iBestSplit = 0;

   auto* const p_DO_NOT_USE_DIRECTLY_Low = IndexBin(pBinBestAndTemp, cBytesPerBin * 2);
   ASSERT_BIN_OK(cBytesPerBin, p_DO_NOT_USE_DIRECTLY_Low, pBinsEndDebug);
   auto* const p_DO_NOT_USE_DIRECTLY_High = IndexBin(pBinBestAndTemp, cBytesPerBin * 3);
   ASSERT_BIN_OK(cBytesPerBin, p_DO_NOT_USE_DIRECTLY_High, pBinsEndDebug);

   Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)> binLow;
   Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)> binHigh;

   // if we know how many scores there are, use the memory on the stack where the compiler can optimize access
   static constexpr bool bUseStackMemory = k_dynamicScores != cCompilerScores;
   auto* const aGradientPairsLow =
         bUseStackMemory ? binLow.GetGradientPairs() : p_DO_NOT_USE_DIRECTLY_Low->GetGradientPairs();
   auto* const aGradientPairsHigh =
         bUseStackMemory ? binHigh.GetGradientPairs() : p_DO_NOT_USE_DIRECTLY_High->GetGradientPairs();

   EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= hessianMin);

   const bool bUseLogitBoost = bHessian && !(TermBoostFlags_DisableNewtonGain & flags);

   // our TensorTotalsSum needs to be templated as dynamic if we want to have something other than 2 dimensions
   EBM_ASSERT(2 == cRealDimensions);

   FloatCalc bestGain = k_illegalGainFloat;
   size_t iBin = 0;
   do {
      aDimensions[iDimensionSweep].m_iLow = 0;
      aDimensions[iDimensionSweep].m_iHigh = iBin + 1;
      TensorTotalsSum<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
            cRealDimensions,
            aDimensions,
            aBins,
            binLow,
            aGradientPairsLow
#ifndef NDEBUG
            ,
            aDebugCopyBins,
            pBinsEndDebug
#endif // NDEBUG
      );
      if(binLow.GetCountSamples() < cSamplesLeafMin) {
         goto next;
      }

      aDimensions[iDimensionSweep].m_iLow = iBin + 1;
      aDimensions[iDimensionSweep].m_iHigh = aDimensions[iDimensionSweep].m_cBins;
      TensorTotalsSum<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
            cRealDimensions,
            aDimensions,
            aBins,
            binHigh,
            aGradientPairsHigh
#ifndef NDEBUG
            ,
            aDebugCopyBins,
            pBinsEndDebug
#endif // NDEBUG
      );
      if(binHigh.GetCountSamples() < cSamplesLeafMin) {
         goto next;
      }

      // if (0 != (TermBoostFlags_PurifyGain & flags)) {
      //  TODO: At this point we have the bin sums histogram for the tensor, so we can purify the future update
      //  for the cuts we're currently evaluating before calculating the gain. This should give us a more accurate gain
      //  calculation for the purified update. We need to construct the entire tensor here before purifying.
      //  We already calculate purified gain as an option during interaction detection, since the
      //  interaction metric we use is the gain calculation.
      //  See: Use of CalcInteractionFlags_Purify in PartitionTwoDimensionalInteraction.cpp
      //}

      {
         FloatCalc gain = 0;
         EBM_ASSERT(0 < binLow.GetCountSamples());
         EBM_ASSERT(0 < binHigh.GetCountSamples());

         EBM_ASSERT(1 <= cScores);
         size_t iScore = 0;
         do {
            FloatCalc hessianLow;
            if(bUseLogitBoost) {
               hessianLow = static_cast<FloatCalc>(aGradientPairsLow[iScore].GetHess());
            } else {
               hessianLow = static_cast<FloatCalc>(binLow.GetWeight());
            }
            if(hessianLow < hessianMin) {
               goto next;
            }

            FloatCalc hessianHigh;
            if(bUseLogitBoost) {
               hessianHigh = static_cast<FloatCalc>(aGradientPairsHigh[iScore].GetHess());
            } else {
               hessianHigh = static_cast<FloatCalc>(binHigh.GetWeight());
            }
            if(hessianHigh < hessianMin) {
               goto next;
            }

            const FloatCalc gain1 = CalcPartialGain(static_cast<FloatCalc>(aGradientPairsLow[iScore].m_sumGradients),
                  hessianLow,
                  regAlpha,
                  regLambda,
                  deltaStepMax);
            EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);
            gain += gain1;

            const FloatCalc gain2 = CalcPartialGain(static_cast<FloatCalc>(aGradientPairsHigh[iScore].m_sumGradients),
                  hessianHigh,
                  regAlpha,
                  regLambda,
                  deltaStepMax);
            EBM_ASSERT(std::isnan(gain2) || 0 <= gain2);
            gain += gain2;

            ++iScore;
         } while(cScores != iScore);
         EBM_ASSERT(std::isnan(gain) || 0 <= gain); // sumation of positive numbers should be positive

         if(UNLIKELY(/* NaN */ !LIKELY(gain <= bestGain))) {
            // propagate NaNs

            bestGain = gain;
            iBestSplit = iBin;

            auto* const pTotalsLowOut = IndexBin(pBinBestAndTemp, cBytesPerBin * 0);
            ASSERT_BIN_OK(cBytesPerBin, pTotalsLowOut, pBinsEndDebug);

            pTotalsLowOut->Copy(cScores, binLow, aGradientPairsLow);

            auto* const pTotalsHighOut = IndexBin(pBinBestAndTemp, cBytesPerBin * 1);
            ASSERT_BIN_OK(cBytesPerBin, pTotalsHighOut, pBinsEndDebug);

            pTotalsHighOut->Copy(cScores, binHigh, aGradientPairsHigh);
         } else {
            EBM_ASSERT(!std::isnan(gain));
         }
      }

   next:;

      ++iBin;
   } while(cSweepCuts != iBin);
   *piBestSplit = iBestSplit;

   EBM_ASSERT(std::isnan(bestGain) || k_illegalGainFloat == bestGain || FloatCalc{0} <= bestGain);
   return bestGain;
}

template<bool bHessian, size_t cCompilerScores> class PartitionTwoDimensionalBoostingInternal final {
 public:
   PartitionTwoDimensionalBoostingInternal() = delete; // this is a static class.  Do not construct

   WARNING_PUSH
   WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BoosterShell* const pBoosterShell,
         const TermBoostFlags flags,
         const Term* const pTerm,
         const size_t* const acBins,
         const size_t cSamplesLeafMin,
         const FloatCalc hessianMin,
         const FloatCalc regAlpha,
         const FloatCalc regLambda,
         const FloatCalc deltaStepMax,
         BinBase* const aAuxiliaryBinsBase,
         double* const aTensorWeights,
         double* const aTensorGrad,
         double* const aTensorHess,
         double* const pTotalGain,
         const size_t cPossibleSplits,
         unsigned char** const aaSplits
#ifndef NDEBUG
         ,
         const BinBase* const aDebugCopyBinsBase
#endif // NDEBUG
   ) {
      static constexpr size_t cCompilerDimensions = 2;
      static constexpr size_t cRealDimensions = cCompilerDimensions;
      const size_t cDimensions = pTerm->GetCountDimensions();

      ErrorEbm error;
      BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();

      auto* const aBins =
            pBoosterShell->GetBoostingMainBins()
                  ->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();
      Tensor* const pInnerTermUpdate = pBoosterShell->GetInnerTermUpdate();

      const size_t cRuntimeScores = pBoosterCore->GetCountScores();
      const size_t cScores = GET_COUNT_SCORES(cCompilerScores, cRuntimeScores);
      const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);
      const size_t cBytesTreeNodeMulti = GetTreeNodeMultiSize(bHessian, cScores);

      auto* const pRootTreeNode = reinterpret_cast<TreeNodeMulti<bHessian, GetArrayScores(cCompilerScores)>*>(
            pBoosterShell->GetTreeNodeMultiTemp());
      auto* const pLowTreeNode = IndexTreeNodeMulti(pRootTreeNode, cBytesTreeNodeMulti * 1);
      auto* const pHighTreeNode = IndexTreeNodeMulti(pRootTreeNode, cBytesTreeNodeMulti * 2);
      auto* const pLowLowTreeNode = IndexTreeNodeMulti(pRootTreeNode, cBytesTreeNodeMulti * 3);
      auto* const pLowHighTreeNode = IndexTreeNodeMulti(pRootTreeNode, cBytesTreeNodeMulti * 4);
      auto* const pHighLowTreeNode = IndexTreeNodeMulti(pRootTreeNode, cBytesTreeNodeMulti * 5);
      auto* const pHighHighTreeNode = IndexTreeNodeMulti(pRootTreeNode, cBytesTreeNodeMulti * 6);
      const auto* const pTreeNodeEnd = IndexTreeNodeMulti(pRootTreeNode, cBytesTreeNodeMulti * 7);

      const bool bUseLogitBoost = bHessian && !(TermBoostFlags_DisableNewtonGain & flags);

      auto* const aAuxiliaryBins =
            aAuxiliaryBinsBase
                  ->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();

      TensorSumDimension
            aDimensions[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];

#ifndef NDEBUG
      const auto* const aDebugCopyBins =
            aDebugCopyBinsBase
                  ->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();
#endif // NDEBUG

      size_t aiStart[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
      // technically this assignment to zero might not be needed, but if we left it uninitialized, then we would later
      // be copying an unitialized memory location into another unitialized memory location which the static clang
      // analysis does not like and which seems might be problematic in some compilers even though not technically
      // undefined behavior according to the standard
      // https://stackoverflow.com/questions/11962457/why-is-using-an-uninitialized-variable-undefined-behavior
      aiStart[1] = 0;

      EBM_ASSERT(2 == pTerm->GetCountRealDimensions());
      EBM_ASSERT(2 <= pTerm->GetCountDimensions());
      size_t iDimensionLoop = 0;
      size_t iDimension1 = 0;
      size_t iDimension2 = 0;
      size_t cBinsDimension1 = 0;
      size_t cBinsDimension2 = 0;
      const TermFeature* pTermFeature = pTerm->GetTermFeatures();
      const TermFeature* const pTermFeaturesEnd = &pTermFeature[cDimensions];
      do {
         const FeatureBoosting* const pFeature = pTermFeature->m_pFeature;
         const size_t cBins = pFeature->GetCountBins();
         EBM_ASSERT(size_t{1} <= cBins); // we don't boost on empty training sets
         if(size_t{1} < cBins) {
            EBM_ASSERT(0 == cBinsDimension2);
            if(0 == cBinsDimension1) {
               iDimension1 = iDimensionLoop;
               cBinsDimension1 = cBins;
               aDimensions[0].m_cBins = cBins;
            } else {
               iDimension2 = iDimensionLoop;
               cBinsDimension2 = cBins;
               aDimensions[1].m_cBins = cBins;
            }
         }
         ++iDimensionLoop;
         ++pTermFeature;
      } while(pTermFeaturesEnd != pTermFeature);
      EBM_ASSERT(2 <= cBinsDimension1);
      EBM_ASSERT(2 <= cBinsDimension2);

      FloatCalc bestGain = k_illegalGainFloat;

      // TODO: put this somewhere safer than at the top of the array
      // and also, we can reduce our auxillary space
      auto* const pTempScratch = IndexBin(aAuxiliaryBins, cBytesPerBin * 0);

      Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)> binTemp;

      // if we know how many scores there are, use the memory on the stack where the compiler can optimize access
      static constexpr bool bUseStackMemory = k_dynamicScores != cCompilerScores;
      auto* const aGradientPairsTemp = bUseStackMemory ? binTemp.GetGradientPairs() : pTempScratch->GetGradientPairs();

      EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= hessianMin);

      LOG_0(Trace_Verbose, "PartitionTwoDimensionalBoostingInternal Starting FIRST bin sweep loop");
      size_t iBin1 = 0;
      do {
         aiStart[0] = iBin1;
         size_t splitSecond1LowBest;
         auto* pTotals2LowLowBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 1);
         auto* pTotals2LowHighBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 2);
         const FloatCalc gain1 = SweepMultiDimensional<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
               cRealDimensions,
               flags,
               aiStart,
               acBins,
               0x0,
               1,
               aBins,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               pTotals2LowLowBest,
               &splitSecond1LowBest
#ifndef NDEBUG
               ,
               aDebugCopyBins,
               pBoosterShell->GetDebugMainBinsEnd()
#endif // NDEBUG
         );

         if(LIKELY(/* NaN */ !UNLIKELY(gain1 < FloatCalc{0}))) {
            EBM_ASSERT(std::isnan(gain1) || FloatCalc{0} <= gain1);

            size_t splitSecond1HighBest;
            auto* pTotals2HighLowBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 5);
            auto* pTotals2HighHighBest = IndexBin(aAuxiliaryBins, cBytesPerBin * 6);
            const FloatCalc gain2 = SweepMultiDimensional<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
                  cRealDimensions,
                  flags,
                  aiStart,
                  acBins,
                  0x1,
                  1,
                  aBins,
                  cSamplesLeafMin,
                  hessianMin,
                  regAlpha,
                  regLambda,
                  deltaStepMax,
                  pTotals2HighLowBest,
                  &splitSecond1HighBest
#ifndef NDEBUG
                  ,
                  aDebugCopyBins,
                  pBoosterShell->GetDebugMainBinsEnd()
#endif // NDEBUG
            );

            if(LIKELY(/* NaN */ !UNLIKELY(gain2 < FloatCalc{0}))) {
               EBM_ASSERT(std::isnan(gain2) || FloatCalc{0} <= gain2);

               const FloatCalc gain = gain1 + gain2;
               if(UNLIKELY(/* NaN */ !LIKELY(gain <= bestGain))) {
                  // propagate NaNs

                  EBM_ASSERT(std::isnan(gain) || FloatCalc{0} <= gain);

                  bestGain = gain;

                  pRootTreeNode->SetSplitGain(0.0);
                  pRootTreeNode->SetDimensionIndex(0);
                  pRootTreeNode->SetSplitIndex(iBin1);
                  pRootTreeNode->SetParent(nullptr);
                  pRootTreeNode->SplitNode();
                  pRootTreeNode->SetChildren(pLowTreeNode);

                  pLowTreeNode->SetSplitGain(0.0);
                  pLowTreeNode->SetDimensionIndex(1);
                  pLowTreeNode->SetSplitIndex(splitSecond1LowBest);
                  pLowTreeNode->SetParent(pRootTreeNode);
                  pLowTreeNode->SplitNode();
                  pLowTreeNode->SetChildren(pLowLowTreeNode);

                  pHighTreeNode->SetSplitGain(0.0);
                  pHighTreeNode->SetDimensionIndex(1);
                  pHighTreeNode->SetSplitIndex(splitSecond1HighBest);
                  pHighTreeNode->SetParent(pRootTreeNode);
                  pHighTreeNode->SplitNode();
                  pHighTreeNode->SetChildren(pHighLowTreeNode);

                  pLowLowTreeNode->SetSplitGain(0.0);
                  pLowLowTreeNode->SetParent(pLowTreeNode);
                  memcpy(pLowLowTreeNode->GetBin(), pTotals2LowLowBest, cBytesPerBin);

                  pLowHighTreeNode->SetSplitGain(0.0);
                  pLowHighTreeNode->SetParent(pLowTreeNode);
                  memcpy(pLowHighTreeNode->GetBin(), pTotals2LowHighBest, cBytesPerBin);

                  pHighLowTreeNode->SetSplitGain(0.0);
                  pHighLowTreeNode->SetParent(pHighTreeNode);
                  memcpy(pHighLowTreeNode->GetBin(), pTotals2HighLowBest, cBytesPerBin);

                  pHighHighTreeNode->SetSplitGain(0.0);
                  pHighHighTreeNode->SetParent(pHighTreeNode);
                  memcpy(pHighHighTreeNode->GetBin(), pTotals2HighHighBest, cBytesPerBin);
               } else {
                  EBM_ASSERT(!std::isnan(gain));
               }
            } else {
               EBM_ASSERT(!std::isnan(gain2));
               EBM_ASSERT(k_illegalGainFloat == gain2);
            }
         } else {
            EBM_ASSERT(!std::isnan(gain1));
            EBM_ASSERT(k_illegalGainFloat == gain1);
         }
         ++iBin1;
      } while(iBin1 < cBinsDimension1 - 1);

      LOG_0(Trace_Verbose, "PartitionTwoDimensionalBoostingInternal Starting SECOND bin sweep loop");
      size_t iBin2 = 0;
      do {
         aiStart[1] = iBin2;
         size_t splitSecond2LowBest;
         auto* pTotals1LowLowBestInner = IndexBin(aAuxiliaryBins, cBytesPerBin * 9);
         auto* pTotals1LowHighBestInner = IndexBin(aAuxiliaryBins, cBytesPerBin * 10);
         const FloatCalc gain1 = SweepMultiDimensional<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
               cRealDimensions,
               flags,
               aiStart,
               acBins,
               0x0,
               0,
               aBins,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               pTotals1LowLowBestInner,
               &splitSecond2LowBest
#ifndef NDEBUG
               ,
               aDebugCopyBins,
               pBoosterShell->GetDebugMainBinsEnd()
#endif // NDEBUG
         );

         if(LIKELY(/* NaN */ !UNLIKELY(gain1 < FloatCalc{0}))) {
            EBM_ASSERT(std::isnan(gain1) || FloatCalc{0} <= gain1);

            size_t splitSecond2HighBest;
            auto* pTotals1HighLowBestInner = IndexBin(aAuxiliaryBins, cBytesPerBin * 13);
            auto* pTotals1HighHighBestInner = IndexBin(aAuxiliaryBins, cBytesPerBin * 14);
            const FloatCalc gain2 = SweepMultiDimensional<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
                  cRealDimensions,
                  flags,
                  aiStart,
                  acBins,
                  0x2,
                  0,
                  aBins,
                  cSamplesLeafMin,
                  hessianMin,
                  regAlpha,
                  regLambda,
                  deltaStepMax,
                  pTotals1HighLowBestInner,
                  &splitSecond2HighBest
#ifndef NDEBUG
                  ,
                  aDebugCopyBins,
                  pBoosterShell->GetDebugMainBinsEnd()
#endif // NDEBUG
            );

            if(LIKELY(/* NaN */ !UNLIKELY(gain2 < FloatCalc{0}))) {
               EBM_ASSERT(std::isnan(gain2) || FloatCalc{0} <= gain2);

               const FloatCalc gain = gain1 + gain2;
               if(UNLIKELY(/* NaN */ !LIKELY(gain <= bestGain))) {
                  // propagate NaNs

                  EBM_ASSERT(std::isnan(gain) || 0 <= gain);

                  bestGain = gain;

                  pRootTreeNode->SetSplitGain(0.0);
                  pRootTreeNode->SetDimensionIndex(1);
                  pRootTreeNode->SetSplitIndex(iBin2);
                  pRootTreeNode->SetParent(nullptr);
                  pRootTreeNode->SplitNode();
                  pRootTreeNode->SetChildren(pLowTreeNode);

                  pLowTreeNode->SetSplitGain(0.0);
                  pLowTreeNode->SetDimensionIndex(0);
                  pLowTreeNode->SetSplitIndex(splitSecond2LowBest);
                  pLowTreeNode->SetParent(pRootTreeNode);
                  pLowTreeNode->SplitNode();
                  pLowTreeNode->SetChildren(pLowLowTreeNode);

                  pHighTreeNode->SetSplitGain(0.0);
                  pHighTreeNode->SetDimensionIndex(0);
                  pHighTreeNode->SetSplitIndex(splitSecond2HighBest);
                  pHighTreeNode->SetParent(pRootTreeNode);
                  pHighTreeNode->SplitNode();
                  pHighTreeNode->SetChildren(pHighLowTreeNode);

                  pLowLowTreeNode->SetSplitGain(0.0);
                  pLowLowTreeNode->SetParent(pLowTreeNode);
                  memcpy(pLowLowTreeNode->GetBin(), pTotals1LowLowBestInner, cBytesPerBin);

                  pLowHighTreeNode->SetSplitGain(0.0);
                  pLowHighTreeNode->SetParent(pLowTreeNode);
                  memcpy(pLowHighTreeNode->GetBin(), pTotals1LowHighBestInner, cBytesPerBin);

                  pHighLowTreeNode->SetSplitGain(0.0);
                  pHighLowTreeNode->SetParent(pHighTreeNode);
                  memcpy(pHighLowTreeNode->GetBin(), pTotals1HighLowBestInner, cBytesPerBin);

                  pHighHighTreeNode->SetSplitGain(0.0);
                  pHighHighTreeNode->SetParent(pHighTreeNode);
                  memcpy(pHighHighTreeNode->GetBin(), pTotals1HighHighBestInner, cBytesPerBin);
               } else {
                  EBM_ASSERT(!std::isnan(gain));
               }
            } else {
               EBM_ASSERT(!std::isnan(gain2));
               EBM_ASSERT(k_illegalGainFloat == gain2);
            }
         } else {
            EBM_ASSERT(!std::isnan(gain1));
            EBM_ASSERT(k_illegalGainFloat == gain1);
         }
         ++iBin2;
      } while(iBin2 < cBinsDimension2 - 1);
      LOG_0(Trace_Verbose, "PartitionTwoDimensionalBoostingInternal Done sweep loops");

      EBM_ASSERT(std::isnan(bestGain) || k_illegalGainFloat == bestGain || FloatCalc{0} <= bestGain);

      // the bin before the aAuxiliaryBins is the last summation bin of aBinsBase,
      // which contains the totals of all bins
      const auto* const pTotal = NegativeIndexBin(aAuxiliaryBins, cBytesPerBin);

      ASSERT_BIN_OK(cBytesPerBin, pTotal, pBoosterShell->GetDebugMainBinsEnd());

      const auto* const pGradientPairTotal = pTotal->GetGradientPairs();

      const FloatMain weightAll = pTotal->GetWeight();
      EBM_ASSERT(0 < weightAll);

      const bool bUpdateWithHessian = bHessian && !(TermBoostFlags_DisableNewtonUpdate & flags);

      GradientPair<FloatMain, bHessian>* pTensorGradientPair = nullptr;

      *pTotalGain = 0;
      EBM_ASSERT(FloatCalc{0} <= k_gainMin);
      if(LIKELY(/* NaN */ !UNLIKELY(bestGain < k_gainMin))) {
         EBM_ASSERT(std::isnan(bestGain) || 0 <= bestGain);

         // signal that we've hit an overflow.  Use +inf here since our caller likes that and will flip to -inf
         *pTotalGain = std::numeric_limits<double>::infinity();
         if(LIKELY(/* NaN */ bestGain <= std::numeric_limits<FloatCalc>::max())) {
            EBM_ASSERT(!std::isnan(bestGain));
            EBM_ASSERT(0 <= bestGain);
            EBM_ASSERT(std::numeric_limits<FloatCalc>::infinity() != bestGain);

            // now subtract the parent partial gain
            for(size_t iScore = 0; iScore < cScores; ++iScore) {
               const FloatCalc hess =
                     static_cast<FloatCalc>(bUseLogitBoost ? pGradientPairTotal[iScore].GetHess() : weightAll);

               // we would not get there unless there was a legal cut, which requires that hessianMin <= hess
               EBM_ASSERT(hessianMin <= hess);

               const FloatCalc gain1 =
                     CalcPartialGain(static_cast<FloatCalc>(pGradientPairTotal[iScore].m_sumGradients),
                           hess,
                           regAlpha,
                           regLambda,
                           deltaStepMax);
               EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);
               bestGain -= gain1;
            }

            EBM_ASSERT(std::numeric_limits<FloatCalc>::infinity() != bestGain);
            EBM_ASSERT(std::isnan(bestGain) || -std::numeric_limits<FloatCalc>::infinity() == bestGain ||
                  k_epsilonNegativeGainAllowed <= bestGain);

            if(LIKELY(/* NaN */ std::numeric_limits<FloatCalc>::lowest() <= bestGain)) {
               EBM_ASSERT(!std::isnan(bestGain));
               EBM_ASSERT(!std::isinf(bestGain));
               EBM_ASSERT(k_epsilonNegativeGainAllowed <= bestGain);

               *pTotalGain = 0;
               if(LIKELY(k_gainMin <= bestGain)) {
                  *pTotalGain = static_cast<double>(bestGain);

                  size_t acSplits[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
                  memset(acSplits, 0, sizeof(acSplits[0]) * cRealDimensions);
                  memset(aaSplits[0], 0, cPossibleSplits * sizeof(*aaSplits[0]));
                  auto* pTreeNode = pRootTreeNode;
                  do {
                     if(pTreeNode->IsSplit()) {
                        const size_t iDimension = pTreeNode->GetDimensionIndex();
                        const size_t iSplit = pTreeNode->GetSplitIndex();
                        unsigned char* const aSplits = aaSplits[iDimension];
                        if(!aSplits[iSplit]) {
                           aSplits[iSplit] = 1;
                           ++acSplits[iDimension];
                        }
                     }
                     pTreeNode = IndexTreeNodeMulti(pTreeNode, cBytesTreeNodeMulti);
                  } while(pTreeNodeEnd != pTreeNode);

                  size_t cTensorCells = 1;
                  for(size_t iDimension = 0; iDimension < 2; ++iDimension) {
                     const size_t iRealDimension = 0 == iDimension ? iDimension1 : iDimension2;

                     const size_t cSplits = acSplits[iDimension];
                     const size_t cSlices = cSplits + 1;
                     error = pInnerTermUpdate->SetCountSlices(iRealDimension, cSlices);
                     if(Error_None != error) {
                        // already logged
                        return error;
                     }

                     cTensorCells *= cSlices;

                     UIntSplit* pSplits = pInnerTermUpdate->GetSplitPointer(iRealDimension);
                     EBM_ASSERT(1 <= cSplits);
                     UIntSplit* pSplitsLast = pSplits + (cSplits - 1);
                     size_t iSplit = 0;
                     unsigned char* const aSplits = aaSplits[iDimension];
                     while(true) {
                        if(aSplits[iSplit]) {
                           *pSplits = iSplit + 1;
                           if(pSplitsLast == pSplits) {
                              break;
                           }
                           ++pSplits;
                        }
                        ++iSplit;
                     }
                  }

                  error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * cTensorCells);
                  if(Error_None != error) {
                     // already logged
                     return error;
                  }

                  FloatScore* const aUpdateScores = pInnerTermUpdate->GetTensorScoresPointer();
                  FloatScore* pUpdateScores = aUpdateScores;

                  FloatScore* pTensorWeights = aTensorWeights;
                  FloatScore* pTensorGrad = aTensorGrad;
                  FloatScore* pTensorHess = aTensorHess;

                  UIntSplit* const aSplits1 = pInnerTermUpdate->GetSplitPointer(iDimension1);
                  UIntSplit* const aSplits2 = pInnerTermUpdate->GetSplitPointer(iDimension2);
                  const size_t cSplits1 = acSplits[0];
                  const size_t cSplits2 = acSplits[1];

                  size_t iSplit2 = 0;

                  aDimensions[1].m_iLow = 0;
                  aDimensions[1].m_iHigh = static_cast<size_t>(aSplits2[0]);
                  do {
                     aDimensions[0].m_iLow = 0;
                     aDimensions[0].m_iHigh = static_cast<size_t>(aSplits1[0]);

                     size_t iSplit1 = 0;
                     do {
                        pTreeNode = pRootTreeNode;
                        while(pTreeNode->IsSplit()) {
                           const size_t iDimension = pTreeNode->GetDimensionIndex();
                           const size_t iSplitTree = pTreeNode->GetSplitIndex();
                           const size_t iSplitTensor = aDimensions[iDimension].m_iLow;
                           pTreeNode = pTreeNode->GetChildren();
                           if(iSplitTree < iSplitTensor) {
                              pTreeNode = GetRightNode(pTreeNode, cBytesTreeNodeMulti);
                           } else {
                              pTreeNode = GetLeftNode(pTreeNode);
                           }
                        }

                        FloatCalc tensorHess;
                        if(nullptr != pTensorWeights || nullptr != pTensorHess || nullptr != pTensorGrad) {
                           ASSERT_BIN_OK(cBytesPerBin, pTempScratch, pBoosterShell->GetDebugMainBinsEnd());
                           TensorTotalsSum<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
                                 cRealDimensions,
                                 aDimensions,
                                 aBins,
                                 binTemp,
                                 aGradientPairsTemp
#ifndef NDEBUG
                                 ,
                                 aDebugCopyBins,
                                 pBoosterShell->GetDebugMainBinsEnd()
#endif // NDEBUG
                           );

                           pTensorGradientPair = aGradientPairsTemp;
                           tensorHess = static_cast<FloatCalc>(binTemp.GetWeight());
                           if(nullptr != pTensorWeights) {
                              *pTensorWeights = tensorHess;
                              ++pTensorWeights;
                           }
                        }

                        FloatCalc nodeHess = static_cast<FloatCalc>(pTreeNode->GetBin()->GetWeight());
                        auto* pGradientPair = pTreeNode->GetBin()->GetGradientPairs();
                        for(size_t iScore = 0; iScore < cScores; ++iScore) {
                           if(bUpdateWithHessian) {
                              nodeHess = static_cast<FloatCalc>(pGradientPair->GetHess());
                           }
                           if(nullptr != pTensorHess || nullptr != pTensorGrad) {
                              if(nullptr != pTensorHess) {
                                 if(bUseLogitBoost) {
                                    tensorHess = static_cast<FloatCalc>(pTensorGradientPair->GetHess());
                                 }
                                 *pTensorHess = tensorHess;
                                 ++pTensorHess;
                              }
                              if(nullptr != pTensorGrad) {
                                 *pTensorGrad = static_cast<FloatCalc>(pTensorGradientPair->m_sumGradients);
                                 ++pTensorGrad;
                              }
                              ++pTensorGradientPair;
                           }

                           FloatCalc prediction =
                                 -CalcNegUpdate<false>(static_cast<FloatCalc>(pGradientPair->m_sumGradients),
                                       nodeHess,
                                       regAlpha,
                                       regLambda,
                                       deltaStepMax);

                           *pUpdateScores = prediction;
                           ++pUpdateScores;
                           ++pGradientPair;
                        }

                        ++iSplit1;
                        EBM_ASSERT(acBins[0] == aDimensions[0].m_cBins);
                        aDimensions[0].m_iLow = aDimensions[0].m_iHigh;
                        aDimensions[0].m_iHigh =
                              iSplit1 < cSplits1 ? static_cast<size_t>(aSplits1[iSplit1]) : aDimensions[0].m_cBins;
                     } while(iSplit1 <= cSplits1);

                     ++iSplit2;
                     EBM_ASSERT(acBins[1] == aDimensions[1].m_cBins);
                     aDimensions[1].m_iLow = aDimensions[1].m_iHigh;
                     aDimensions[1].m_iHigh =
                           iSplit2 < cSplits2 ? static_cast<size_t>(aSplits2[iSplit2]) : aDimensions[1].m_cBins;
                  } while(iSplit2 <= cSplits2);

                  return Error_None;
               }
            } else {
               EBM_ASSERT(std::isnan(bestGain) || -std::numeric_limits<FloatCalc>::infinity() == bestGain);
            }
         } else {
            EBM_ASSERT(std::isnan(bestGain) || std::numeric_limits<FloatCalc>::infinity() == bestGain);
         }
      } else {
         EBM_ASSERT(!std::isnan(bestGain));
      }

      // there were no good splits found
#ifndef NDEBUG
      const ErrorEbm errorDebug1 =
#endif // NDEBUG
            pInnerTermUpdate->SetCountSlices(iDimension1, 1);
      // we can't fail since we're setting this to zero, so no allocations.  We don't in fact need the split array at
      // all
      EBM_ASSERT(Error_None == errorDebug1);

#ifndef NDEBUG
      const ErrorEbm errorDebug2 =
#endif // NDEBUG
            pInnerTermUpdate->SetCountSlices(iDimension2, 1);
      // we can't fail since we're setting this to zero, so no allocations.  We don't in fact need the split array at
      // all
      EBM_ASSERT(Error_None == errorDebug2);

      // we don't need to call pInnerTermUpdate->EnsureTensorScoreCapacity,
      // since our value capacity would be 1, which is pre-allocated

      if(nullptr != aTensorWeights) {
         *aTensorWeights = weightAll;
      }
      FloatScore* pTensorGrad = aTensorGrad;
      FloatScore* pTensorHess = aTensorHess;

      FloatScore* const aUpdateScores = pInnerTermUpdate->GetTensorScoresPointer();
      for(size_t iScore = 0; iScore < cScores; ++iScore) {
         FloatCalc update;
         FloatCalc weight;
         if(nullptr != pTensorGrad) {
            *pTensorGrad = static_cast<FloatCalc>(pGradientPairTotal[iScore].m_sumGradients);
            ++pTensorGrad;
         }
         if(nullptr != pTensorHess) {
            if(bUseLogitBoost) {
               weight = static_cast<FloatCalc>(pGradientPairTotal[iScore].GetHess());
            } else {
               weight = static_cast<FloatCalc>(weightAll);
            }
            *pTensorHess = weight;
            ++pTensorHess;
         }
         if(bUpdateWithHessian) {
            weight = static_cast<FloatCalc>(pGradientPairTotal[iScore].GetHess());
            update = -CalcNegUpdate<true>(static_cast<FloatCalc>(pGradientPairTotal[iScore].m_sumGradients),
                  weight,
                  regAlpha,
                  regLambda,
                  deltaStepMax);
         } else {
            weight = static_cast<FloatCalc>(weightAll);
            update = -CalcNegUpdate<true>(static_cast<FloatCalc>(pGradientPairTotal[iScore].m_sumGradients),
                  weight,
                  regAlpha,
                  regLambda,
                  deltaStepMax);
         }

         aUpdateScores[iScore] = static_cast<FloatScore>(update);
      }
      return Error_None;
   }
   WARNING_POP
};

template<bool bHessian, size_t cPossibleScores> class PartitionTwoDimensionalBoostingTarget final {
 public:
   PartitionTwoDimensionalBoostingTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BoosterShell* const pBoosterShell,
         const TermBoostFlags flags,
         const Term* const pTerm,
         const size_t* const acBins,
         const size_t cSamplesLeafMin,
         const FloatCalc hessianMin,
         const FloatCalc regAlpha,
         const FloatCalc regLambda,
         const FloatCalc deltaStepMax,
         BinBase* aAuxiliaryBinsBase,
         double* const aTensorWeights,
         double* const aTensorGrad,
         double* const aTensorHess,
         double* const pTotalGain,
         const size_t cPossibleSplits,
         unsigned char** const aaSplits
#ifndef NDEBUG
         ,
         const BinBase* const aDebugCopyBinsBase
#endif // NDEBUG
   ) {
      BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
      if(cPossibleScores == pBoosterCore->GetCountScores()) {
         return PartitionTwoDimensionalBoostingInternal<bHessian, cPossibleScores>::Func(pBoosterShell,
               flags,
               pTerm,
               acBins,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               aAuxiliaryBinsBase,
               aTensorWeights,
               aTensorGrad,
               aTensorHess,
               pTotalGain,
               cPossibleSplits,
               aaSplits
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase
#endif // NDEBUG
         );
      } else {
         return PartitionTwoDimensionalBoostingTarget<bHessian, cPossibleScores + 1>::Func(pBoosterShell,
               flags,
               pTerm,
               acBins,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               aAuxiliaryBinsBase,
               aTensorWeights,
               aTensorGrad,
               aTensorHess,
               pTotalGain,
               cPossibleSplits,
               aaSplits
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase
#endif // NDEBUG
         );
      }
   }
};

template<bool bHessian> class PartitionTwoDimensionalBoostingTarget<bHessian, k_cCompilerScoresMax + 1> final {
 public:
   PartitionTwoDimensionalBoostingTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(BoosterShell* const pBoosterShell,
         const TermBoostFlags flags,
         const Term* const pTerm,
         const size_t* const acBins,
         const size_t cSamplesLeafMin,
         const FloatCalc hessianMin,
         const FloatCalc regAlpha,
         const FloatCalc regLambda,
         const FloatCalc deltaStepMax,
         BinBase* aAuxiliaryBinsBase,
         double* const aTensorWeights,
         double* const aTensorGrad,
         double* const aTensorHess,
         double* const pTotalGain,
         const size_t cPossibleSplits,
         unsigned char** const aaSplits
#ifndef NDEBUG
         ,
         const BinBase* const aDebugCopyBinsBase
#endif // NDEBUG
   ) {
      return PartitionTwoDimensionalBoostingInternal<bHessian, k_dynamicScores>::Func(pBoosterShell,
            flags,
            pTerm,
            acBins,
            cSamplesLeafMin,
            hessianMin,
            regAlpha,
            regLambda,
            deltaStepMax,
            aAuxiliaryBinsBase,
            aTensorWeights,
            aTensorGrad,
            aTensorHess,
            pTotalGain,
            cPossibleSplits,
            aaSplits
#ifndef NDEBUG
            ,
            aDebugCopyBinsBase
#endif // NDEBUG
      );
   }
};

extern ErrorEbm PartitionTwoDimensionalBoosting(BoosterShell* const pBoosterShell,
      const TermBoostFlags flags,
      const Term* const pTerm,
      const size_t* const acBins,
      const size_t cSamplesLeafMin,
      const FloatCalc hessianMin,
      const FloatCalc regAlpha,
      const FloatCalc regLambda,
      const FloatCalc deltaStepMax,
      BinBase* aAuxiliaryBinsBase,
      double* const aTensorWeights,
      double* const aTensorGrad,
      double* const aTensorHess,
      double* const pTotalGain
#ifndef NDEBUG
      ,
      const BinBase* const aDebugCopyBinsBase
#endif // NDEBUG
) {
   BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   const size_t cRuntimeScores = pBoosterCore->GetCountScores();
   const bool bHessian = pBoosterCore->IsHessian();

   if(IsOverflowBinSize<FloatMain, UIntMain>(true, true, bHessian, cRuntimeScores)) {
      // TODO: move this to init
      return Error_OutOfMemory;
   }

   if(IsOverflowTreeNodeMultiSize(bHessian, cRuntimeScores)) {
      // TODO: move this to init
      return Error_OutOfMemory;
   }

   size_t cPossibleSplits = 0;

   size_t cBytes = 1;
   const size_t* pcBins = acBins;
   const size_t* const acBinsEnd = acBins + pTerm->GetCountRealDimensions();
   do {
      const size_t cBins = *pcBins;
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
   } while(acBinsEnd != pcBins);
   // For pairs, this calculates the exact max number of splits. For higher dimensions
   // the max number of splits will be less, but it should be close enough.
   // Each bin gets a tree node to record the gradient totals, and each split gets a TreeNode
   // during construction. Each split contains a minimum of 1 bin on each side, so we have
   // cBins - 1 potential splits.

   if(IsAddError(cBytes, cBytes - 1)) {
      return Error_OutOfMemory;
   }
   cBytes = cBytes + cBytes - 1;

   const size_t cBytesTreeNodeMulti = GetTreeNodeMultiSize(bHessian, cRuntimeScores);

   if(IsMultiplyError(cBytesTreeNodeMulti, cBytes)) {
      return Error_OutOfMemory;
   }
   cBytes *= cBytesTreeNodeMulti;

   ErrorEbm error = pBoosterShell->ReserveTreeNodesTemp(cBytes);
   if(Error_None != error) {
      return error;
   }

   error = pBoosterShell->ReserveTemp1(cPossibleSplits * sizeof(unsigned char));
   if(Error_None != error) {
      return error;
   }

   unsigned char* pSplits = static_cast<unsigned char*>(pBoosterShell->GetTemp1());
   unsigned char* aaSplits[k_cDimensionsMax];
   unsigned char** paSplits = aaSplits;

   pcBins = acBins;
   do {
      *paSplits = pSplits;
      const size_t cSplits = *pcBins - 1;
      pSplits += cSplits;
      ++paSplits;
      ++pcBins;
   } while(acBinsEnd != pcBins);

   EBM_ASSERT(1 <= cRuntimeScores);
   if(bHessian) {
      if(size_t{1} != cRuntimeScores) {
         // muticlass
         error = PartitionTwoDimensionalBoostingTarget<true, k_cCompilerScoresStart>::Func(pBoosterShell,
               flags,
               pTerm,
               acBins,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               aAuxiliaryBinsBase,
               aTensorWeights,
               aTensorGrad,
               aTensorHess,
               pTotalGain,
               cPossibleSplits,
               aaSplits
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase
#endif // NDEBUG
         );
      } else {
         error = PartitionTwoDimensionalBoostingInternal<true, k_oneScore>::Func(pBoosterShell,
               flags,
               pTerm,
               acBins,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               aAuxiliaryBinsBase,
               aTensorWeights,
               aTensorGrad,
               aTensorHess,
               pTotalGain,
               cPossibleSplits,
               aaSplits
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase
#endif // NDEBUG
         );
      }
   } else {
      if(size_t{1} != cRuntimeScores) {
         // Odd: gradient multiclass. Allow it, but do not optimize for it
         error = PartitionTwoDimensionalBoostingInternal<false, k_dynamicScores>::Func(pBoosterShell,
               flags,
               pTerm,
               acBins,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               aAuxiliaryBinsBase,
               aTensorWeights,
               aTensorGrad,
               aTensorHess,
               pTotalGain,
               cPossibleSplits,
               aaSplits
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase
#endif // NDEBUG
         );
      } else {
         error = PartitionTwoDimensionalBoostingInternal<false, k_oneScore>::Func(pBoosterShell,
               flags,
               pTerm,
               acBins,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               aAuxiliaryBinsBase,
               aTensorWeights,
               aTensorGrad,
               aTensorHess,
               pTotalGain,
               cPossibleSplits,
               aaSplits
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase
#endif // NDEBUG
         );
      }
   }
   return error;
}

} // namespace DEFINED_ZONE_NAME
