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
      static constexpr size_t cCompilerDimensions = k_dynamicDimensions;
      const size_t cRealDimensions = pTerm->GetCountRealDimensions();

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

      // each dimension requires 2 tree nodes, plus one for the last
      const size_t cBytesBest = cBytesTreeNodeMulti * (size_t{1} + (cRealDimensions << 1));
      auto* const pDeepTreeNode = IndexTreeNodeMulti(pRootTreeNode, cBytesBest);

      auto* const pLastTreeNode = IndexTreeNodeMulti(pDeepTreeNode, cBytesBest - (cBytesTreeNodeMulti << 1));
      auto* const pLastSplitTreeNode = NegativeIndexByte(pLastTreeNode, cBytesTreeNodeMulti);

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

      size_t iDimensionLoop = 0;
      size_t iDimInit = 0;
      const TermFeature* const aTermFeatures = pTerm->GetTermFeatures();
      size_t aiOriginalIndex[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
      size_t aiDim[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
      EBM_ASSERT(1 <= cRealDimensions);
      do {
         EBM_ASSERT(iDimensionLoop < pTerm->GetCountDimensions());
         const FeatureBoosting* const pFeature = aTermFeatures[iDimensionLoop].m_pFeature;
         const size_t cBins = pFeature->GetCountBins();
         EBM_ASSERT(size_t{1} <= cBins); // we don't boost on empty training sets
         if(size_t{1} < cBins) {
            aiOriginalIndex[iDimInit] = iDimensionLoop;
            aDimensions[iDimInit].m_cBins = cBins;
            aiDim[iDimInit] = iDimInit;
            ++iDimInit;
         }
         ++iDimensionLoop;
      } while(cRealDimensions != iDimInit);

      FloatCalc bestGain = k_illegalGainFloat;

      auto* const pTempScratch = aAuxiliaryBins;

      Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)> binTemp;

      // if we know how many scores there are, use the memory on the stack where the compiler can optimize access
      static constexpr bool bUseStackMemory = k_dynamicScores != cCompilerScores;
      auto* const aGradientPairsTemp = bUseStackMemory ? binTemp.GetGradientPairs() : pTempScratch->GetGradientPairs();

      EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= hessianMin);

      const TensorSumDimension* const pDimensionEnd = &aDimensions[cRealDimensions];

      // TODO: move this into the initialization above
      TreeNodeMulti<bHessian, GetArrayScores(cCompilerScores)>* pParentTreeNode = nullptr;
      auto* pTreeNode = pDeepTreeNode;
      auto* pHigh = IndexTreeNodeMulti(pTreeNode, cBytesTreeNodeMulti);
      EBM_ASSERT(1 <= cRealDimensions);
      do {
         pTreeNode->SplitNode();
         pTreeNode->SetSplitIndex(0);
         pTreeNode->SetParent(pParentTreeNode);
         pTreeNode->SetChildren(pHigh);

         pParentTreeNode = pTreeNode;
         pTreeNode = IndexTreeNodeMulti(pHigh, cBytesTreeNodeMulti);
         auto* const pNextHigh = IndexTreeNodeMulti(pTreeNode, cBytesTreeNodeMulti);

         // High child Node
         pHigh->SetSplitGain(0.0);
         pHigh->SetSplitIndex(0);
         pHigh->SetParent(pParentTreeNode);
         // set both high and low nodes to point to the same children. It isn't valid
         // if the node isn't split but this avoids having to continually swap them
         pHigh->SetChildren(pNextHigh);

         pHigh = pNextHigh;
      } while(pTreeNode <= pLastSplitTreeNode);

      // Low child node
      pTreeNode->SetSplitGain(0.0);
      pTreeNode->SetSplitIndex(0);
      pTreeNode->SetParent(pParentTreeNode);
      pTreeNode->SetChildren(pHigh); // we need to set it to something because we access this pointer below

      while(true) {
         pTreeNode = pLastSplitTreeNode;
         size_t iDim = 0;
         size_t iDimReordered;
         while(true) {
            EBM_ASSERT(pTreeNode->IsSplit());
            EBM_ASSERT(GetHighNode(pTreeNode->GetChildren())->GetParent() == pTreeNode);
            EBM_ASSERT(GetLowNode(pTreeNode->GetChildren(), cBytesTreeNodeMulti)->GetParent() == pTreeNode);
            EBM_ASSERT(GetHighNode(pTreeNode->GetChildren())->GetChildren() ==
                  GetLowNode(pTreeNode->GetChildren(), cBytesTreeNodeMulti)->GetChildren());
            EBM_ASSERT(!GetHighNode(pTreeNode->GetChildren())->IsSplit());
            EBM_ASSERT(size_t{0} != iDim || !GetLowNode(pTreeNode->GetChildren(), cBytesTreeNodeMulti)->IsSplit());
            EBM_ASSERT(size_t{0} == iDim || GetLowNode(pTreeNode->GetChildren(), cBytesTreeNodeMulti)->IsSplit());
            EBM_ASSERT(0 == pTreeNode->GetSplitIndex());

            iDimReordered = aiDim[iDim];

            auto* const pParent = pTreeNode->GetParent();
            if(nullptr == pParent) {
               break;
            }
            EBM_ASSERT(GetLowNode(pParent->GetChildren(), cBytesTreeNodeMulti) == pTreeNode);

            pTreeNode->SetDimensionIndex(iDimReordered);
            auto* const pHighSibling = NegativeIndexByte(pTreeNode, cBytesTreeNodeMulti);
            pHighSibling->SetDimensionIndex(iDimReordered);

            ++iDim;
            pTreeNode = pParent;
         }
         pTreeNode->SetDimensionIndex(iDimReordered);

         while(true) {
            while(true) {
               EBM_ASSERT(1 <= cRealDimensions);
               TensorSumDimension* pDimension = aDimensions;
               do {
                  pDimension->m_iLow = 0;
                  pDimension->m_iHigh = pDimension->m_cBins;
                  ++pDimension;
               } while(pDimensionEnd != pDimension);

               // TODO: We can optimize away some of these calls to TensorTotalsSum because some of the
               // tensors do not change in each tree cut. For example, if we had a primary cut on the 0th dimension
               // and one cut in the 1st dimension on the lower side of the 0th dimension cut, then if we move the
               // cut along the 1st dimension, the tensor sum on the opposite since is not changing.
               FloatCalc gain = 0.0;
               pTreeNode = pDeepTreeNode;
               TreeNodeMulti<bHessian, GetArrayScores(cCompilerScores)>* pNextTreeNode;
               do {
                  pNextTreeNode = nullptr;

                  EBM_ASSERT(pTreeNode->IsSplit());
                  const size_t iTreeDim = pTreeNode->GetDimensionIndex();
                  const size_t iSplit = pTreeNode->GetSplitIndex() + 1;
                  auto* const pChildren = pTreeNode->GetChildren();

                  auto* const pLowSum = GetLowNode(pChildren, cBytesTreeNodeMulti);
                  if(pLowSum->IsSplit()) {
                     pNextTreeNode = pLowSum;
                  } else {
                     aDimensions[iTreeDim].m_iLow = 0;
                     aDimensions[iTreeDim].m_iHigh = iSplit;

                     auto* const aGradientPairsLocal = pLowSum->GetBin()->GetGradientPairs();

                     TensorTotalsSum<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
                           cRealDimensions,
                           aDimensions,
                           aBins,
                           *pLowSum->GetBin(),
                           aGradientPairsLocal
#ifndef NDEBUG
                           ,
                           aDebugCopyBins,
                           pBoosterShell->GetDebugMainBinsEnd()
#endif // NDEBUG
                     );

                     if(pLowSum->GetBin()->GetCountSamples() < cSamplesLeafMin) {
                        goto next;
                     }

                     EBM_ASSERT(1 <= cScores);
                     size_t iScore = 0;
                     FloatCalc hessian = static_cast<FloatCalc>(pLowSum->GetBin()->GetWeight());
                     do {
                        if(bUseLogitBoost) {
                           hessian = static_cast<FloatCalc>(aGradientPairsLocal[iScore].GetHess());
                        }
                        if(hessian < hessianMin) {
                           goto next;
                        }

                        const FloatCalc gain1 =
                              CalcPartialGain(static_cast<FloatCalc>(aGradientPairsLocal[iScore].m_sumGradients),
                                    hessian,
                                    regAlpha,
                                    regLambda,
                                    deltaStepMax);
                        EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);
                        gain += gain1;

                        ++iScore;
                     } while(cScores != iScore);
                     EBM_ASSERT(std::isnan(gain) || 0 <= gain); // sumation of positive numbers should be positive
                  }

                  aDimensions[iTreeDim].m_iLow = iSplit;
                  aDimensions[iTreeDim].m_iHigh = aDimensions[iTreeDim].m_cBins;

                  auto* const pHighSum = GetHighNode(pChildren);
                  if(pHighSum->IsSplit()) {
                     EBM_ASSERT(nullptr == pNextTreeNode);
                     pNextTreeNode = pHighSum;
                  } else {
                     auto* const aGradientPairsLocal = pHighSum->GetBin()->GetGradientPairs();

                     TensorTotalsSum<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
                           cRealDimensions,
                           aDimensions,
                           aBins,
                           *pHighSum->GetBin(),
                           aGradientPairsLocal
#ifndef NDEBUG
                           ,
                           aDebugCopyBins,
                           pBoosterShell->GetDebugMainBinsEnd()
#endif // NDEBUG
                     );

                     if(pHighSum->GetBin()->GetCountSamples() < cSamplesLeafMin) {
                        goto next;
                     }

                     EBM_ASSERT(1 <= cScores);
                     FloatCalc hessian = static_cast<FloatCalc>(pHighSum->GetBin()->GetWeight());
                     size_t iScore = 0;
                     do {
                        if(bUseLogitBoost) {
                           hessian = static_cast<FloatCalc>(aGradientPairsLocal[iScore].GetHess());
                        }
                        if(hessian < hessianMin) {
                           goto next;
                        }

                        const FloatCalc gain1 =
                              CalcPartialGain(static_cast<FloatCalc>(aGradientPairsLocal[iScore].m_sumGradients),
                                    hessian,
                                    regAlpha,
                                    regLambda,
                                    deltaStepMax);
                        EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);
                        gain += gain1;

                        ++iScore;
                     } while(cScores != iScore);
                     EBM_ASSERT(std::isnan(gain) || 0 <= gain); // sumation of positive numbers should be positive

                     // for all descendents we restrict to the opposite side
                     aDimensions[iTreeDim].m_iLow = 0;
                     aDimensions[iTreeDim].m_iHigh = iSplit;
                  }

                  pTreeNode = pNextTreeNode;
               } while(nullptr != pTreeNode);

               if(UNLIKELY(/* NaN */ !LIKELY(gain <= bestGain))) {
                  // propagate NaNs
                  bestGain = gain;
                  memcpy(pRootTreeNode, pDeepTreeNode, cBytesBest);
               } else {
                  EBM_ASSERT(!std::isnan(gain));
               }

            next:;

               EBM_ASSERT(!pLastTreeNode->IsSplit());
               pTreeNode = pLastTreeNode->GetParent();
               EBM_ASSERT(nullptr != pTreeNode);
               while(true) {
                  EBM_ASSERT(pTreeNode->IsSplit());
                  const size_t iTreeDim = pTreeNode->GetDimensionIndex();
                  const size_t iSplit = pTreeNode->GetSplitIndex() + 1;
                  const size_t cBinsMinusOne = aDimensions[iTreeDim].m_cBins - 1;
                  EBM_ASSERT(1 <= cBinsMinusOne);
                  EBM_ASSERT(iSplit <= cBinsMinusOne);
                  pTreeNode->SetSplitIndex(iSplit);
                  if(iSplit != cBinsMinusOne) {
                     break;
                  }
                  pTreeNode->SetSplitIndex(0);
                  pTreeNode = pTreeNode->GetParent();
                  if(nullptr == pTreeNode) {
                     goto next_tree;
                  }
               }
            }
         next_tree:;

            EBM_ASSERT(!pLastTreeNode->IsSplit());
            pTreeNode = pLastTreeNode->GetParent();
            EBM_ASSERT(nullptr != pTreeNode);
            while(true) {
               EBM_ASSERT(pTreeNode->IsSplit());

               auto* const pParent = pTreeNode->GetParent();
               if(nullptr == pParent) {
                  goto done_tree;
               }

               auto* const pChildren = pParent->GetChildren();
               if(pTreeNode != pChildren) {
                  // move from low to high and we are done
                  auto* const pLowSwap = pTreeNode;
                  EBM_ASSERT(NegativeIndexByte(pTreeNode, cBytesTreeNodeMulti) == pChildren);
                  auto* const pHighSwap = pChildren;

                  EBM_ASSERT(0 == pHighSwap->GetSplitIndex());
                  EBM_ASSERT(pHighSwap->GetDimensionIndex() == pLowSwap->GetDimensionIndex());
                  pHighSwap->SplitNode();
                  EBM_ASSERT(pHighSwap->GetChildren() == pLowSwap->GetChildren());
                  EBM_ASSERT(pHighSwap->GetParent() == pLowSwap->GetParent());

                  pLowSwap->SetSplitGain(0.0);

                  auto* const pChildrenSwap = pLowSwap->GetChildren();
                  auto* const pLowChild = GetLowNode(pChildrenSwap, cBytesTreeNodeMulti);
                  auto* const pHighChild = GetHighNode(pChildrenSwap);

                  pLowChild->SetParent(pHighSwap);
                  pHighChild->SetParent(pHighSwap);

                  break;
               } else {
                  // move from high to low and continue
                  auto* const pHighSwap = pTreeNode;
                  auto* const pLowSwap = IndexByte(pTreeNode, cBytesTreeNodeMulti);

                  EBM_ASSERT(0 == pLowSwap->GetSplitIndex());
                  EBM_ASSERT(pLowSwap->GetDimensionIndex() == pHighSwap->GetDimensionIndex());
                  pLowSwap->SplitNode();
                  EBM_ASSERT(pLowSwap->GetChildren() == pHighSwap->GetChildren());
                  EBM_ASSERT(pLowSwap->GetParent() == pHighSwap->GetParent());

                  pHighSwap->SetSplitGain(0.0);

                  auto* const pChildrenSwap = pHighSwap->GetChildren();
                  auto* const pLowChild = GetLowNode(pChildrenSwap, cBytesTreeNodeMulti);
                  auto* const pHighChild = GetHighNode(pChildrenSwap);

                  pLowChild->SetParent(pLowSwap);
                  pHighChild->SetParent(pLowSwap);
               }

               pTreeNode = pParent;
            }
         }
      done_tree:;

         EBM_ASSERT(1 <= cRealDimensions);
         if(1 == cRealDimensions) {
            goto done;
         }
         size_t i = 1;
         while(aiDim[i] <= aiDim[i - 1]) {
            if(i == cRealDimensions - 1) {
               goto done;
            }
            ++i;
         }
         size_t j = 0;
         while(aiDim[j] >= aiDim[i]) {
            ++j;
         }

         size_t temp = aiDim[i];
         aiDim[i] = aiDim[j];
         aiDim[j] = temp;

         size_t start = 0;
         size_t end = i - 1;
         while(start < end) {
            temp = aiDim[start];
            aiDim[start] = aiDim[end];
            aiDim[end] = temp;
            ++start;
            --end;
         }
      }
   done:;

      auto* pCurTreeNode = pRootTreeNode;
      EBM_ASSERT(nullptr == pCurTreeNode->GetParent());
      while(true) {
         EBM_ASSERT(nullptr != pCurTreeNode->GetChildren());
         const size_t cBytesOffset1 =
               reinterpret_cast<char*>(pCurTreeNode->GetChildren()) - reinterpret_cast<char*>(pDeepTreeNode);
         TreeNodeMulti<bHessian, GetArrayScores(cCompilerScores)>* const pNode1 =
               IndexTreeNodeMulti(pRootTreeNode, cBytesOffset1);
         pCurTreeNode->SetChildren(pNode1);

         pCurTreeNode = IndexTreeNodeMulti(pCurTreeNode, cBytesTreeNodeMulti);
         if(pDeepTreeNode == pCurTreeNode) {
            break;
         }

         EBM_ASSERT(nullptr != pCurTreeNode->GetParent());
         const size_t cBytesOffset2 =
               reinterpret_cast<char*>(pCurTreeNode->GetParent()) - reinterpret_cast<char*>(pDeepTreeNode);
         TreeNodeMulti<bHessian, GetArrayScores(cCompilerScores)>* const pNode2 =
               IndexTreeNodeMulti(pRootTreeNode, cBytesOffset2);
         pCurTreeNode->SetParent(pNode2);
      }

      TreeNodeMulti<bHessian, GetArrayScores(cCompilerScores)>* const pTreeNodeEnd = pDeepTreeNode;

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
                  // TODO: we can improve this by moving 2 and not checking IsSplit
                  pTreeNode = pRootTreeNode;
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
                  EBM_ASSERT(1 <= cRealDimensions);
                  size_t iDimension = 0;
                  do {
                     const size_t iOriginalDimension = aiOriginalIndex[iDimension];

                     const size_t cSplits = acSplits[iDimension];
                     const size_t cSlices = cSplits + size_t{1};
                     error = pInnerTermUpdate->SetCountSlices(iOriginalDimension, cSlices);
                     if(Error_None != error) {
                        // already logged
                        return error;
                     }

                     cTensorCells *= cSlices;

                     UIntSplit* pSplits = pInnerTermUpdate->GetSplitPointer(iOriginalDimension);
                     EBM_ASSERT(1 <= cSplits);
                     UIntSplit* pSplitsLast = pSplits + (cSplits - size_t{1});
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
                     ++iDimension;
                  } while(cRealDimensions != iDimension);

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

                  size_t iDim = 0;
                  do {
                     const size_t cSplitFirst =
                           static_cast<size_t>(pInnerTermUpdate->GetSplitPointer(aiOriginalIndex[iDim])[0]);
                     aDimensions[iDim].m_iLow = 0;
                     aDimensions[iDim].m_iHigh = cSplitFirst;
                     ++iDim;
                  } while(cRealDimensions != iDim);

                  size_t aiSplits[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
                  memset(aiSplits, 0, sizeof(aiSplits));
                  while(true) {
                     pTreeNode = pRootTreeNode;
                     EBM_ASSERT(pTreeNode->IsSplit());
                     do {
                        const size_t iDimensionInternal = pTreeNode->GetDimensionIndex();
                        const size_t iSplitTree = pTreeNode->GetSplitIndex();
                        const size_t iSplitTensor = aDimensions[iDimensionInternal].m_iLow;
                        pTreeNode = pTreeNode->GetChildren();
                        if(iSplitTree < iSplitTensor) {
                           pTreeNode = GetHighNode(pTreeNode);
                        } else {
                           pTreeNode = GetLowNode(pTreeNode, cBytesTreeNodeMulti);
                        }
                     } while(pTreeNode->IsSplit());

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

                     iDim = 0;
                     while(true) {
                        const size_t iSplit = aiSplits[iDim] + size_t{1};
                        const size_t cSplits = acSplits[iDim];
                        if(iSplit <= cSplits) {
                           aDimensions[iDim].m_iLow = aDimensions[iDim].m_iHigh;
                           aDimensions[iDim].m_iHigh = cSplits == iSplit ?
                                 aDimensions[iDim].m_cBins :
                                 static_cast<size_t>(pInnerTermUpdate->GetSplitPointer(aiOriginalIndex[iDim])[iSplit]);
                           aiSplits[iDim] = iSplit;
                           break;
                        }
                        aDimensions[iDim].m_iLow = 0;
                        aDimensions[iDim].m_iHigh =
                              static_cast<size_t>(pInnerTermUpdate->GetSplitPointer(aiOriginalIndex[iDim])[0]);
                        aiSplits[iDim] = 0;

                        ++iDim;
                        if(cRealDimensions == iDim) {
                           goto done1;
                        }
                     }
                  }
               done1:;

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
      pInnerTermUpdate->Reset();

      // we don't need to call pInnerTermUpdate->EnsureTensorScoreCapacity,
      // since our value capacity would be 1, which is pre-allocated

      if(nullptr != aTensorWeights) {
         *aTensorWeights = weightAll;
      }
      FloatScore* pTensorGrad = aTensorGrad;
      FloatScore* pTensorHess = aTensorHess;

      FloatScore* const aUpdateScores = pInnerTermUpdate->GetTensorScoresPointer();
      FloatCalc weight1 = static_cast<FloatCalc>(weightAll);
      FloatCalc weight2 = static_cast<FloatCalc>(weightAll);
      for(size_t iScore = 0; iScore < cScores; ++iScore) {
         if(nullptr != pTensorGrad) {
            *pTensorGrad = static_cast<FloatCalc>(pGradientPairTotal[iScore].m_sumGradients);
            ++pTensorGrad;
         }
         if(nullptr != pTensorHess) {
            if(bUseLogitBoost) {
               weight1 = static_cast<FloatCalc>(pGradientPairTotal[iScore].GetHess());
            }
            *pTensorHess = weight1;
            ++pTensorHess;
         }
         if(bUpdateWithHessian) {
            weight2 = static_cast<FloatCalc>(pGradientPairTotal[iScore].GetHess());
         }
         const FloatCalc update =
               -CalcNegUpdate<true>(static_cast<FloatCalc>(pGradientPairTotal[iScore].m_sumGradients),
                     weight2,
                     regAlpha,
                     regLambda,
                     deltaStepMax);

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

   const size_t cBytesBest = cBytesTreeNodeMulti * (size_t{1} + (pTerm->GetCountRealDimensions() << 1));
   EBM_ASSERT(cBytesBest <= cBytes);

   // double it because we during the multi-dimensional sweep we need the best and we need the current
   if(IsAddError(cBytesBest, cBytesBest)) {
      return Error_OutOfMemory;
   }
   const size_t cBytesSweep = cBytesBest + cBytesBest;

   cBytes = EbmMax(cBytes, cBytesSweep);

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
