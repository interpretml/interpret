// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <type_traits> // std::is_standard_layout
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy
#include <vector>
#include <queue>
#include <algorithm> // std::sort

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT

#define ZONE_main
#include "zones.h"

#include "GradientPair.hpp"
#include "Bin.hpp"

#include "ebm_internal.hpp"
#include "RandomDeterministic.hpp"
#include "ebm_stats.hpp"
#include "Tensor.hpp"
#include "TreeNode.hpp"
#include "SplitPosition.hpp"
#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<bool bHessian, size_t cCompilerScores>
INLINE_RELEASE_TEMPLATED static void SumAllBins(BoosterShell* const pBoosterShell,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const pBinsEnd,
      const size_t cSamplesTotal,
      const FloatMain weightTotal,
      Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const pBinOut) {
   // these stay the same across boosting rounds, so we can calculate them once at init
   pBinOut->SetCountSamples(static_cast<UIntMain>(cSamplesTotal));
   pBinOut->SetWeight(weightTotal);

   BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pBoosterCore->GetCountScores());

   // if we know how many scores there are, use the memory on the stack where the compiler can optimize access
   GradientPair<FloatMain, bHessian> aSumGradHessLocal[GetArrayScores(cCompilerScores)];
   static constexpr bool bUseStackMemory = k_dynamicScores != cCompilerScores;
   auto* const aSumGradHess = bUseStackMemory ? aSumGradHessLocal : pBinOut->GetGradientPairs();

   ZeroGradientPairs(aSumGradHess, cScores);

   const auto* const aBins =
         pBoosterShell->GetBoostingMainBins()
               ->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();

#ifndef NDEBUG
   UIntMain cSamplesTotalDebug = 0;
   FloatMain weightTotalDebug = 0;
#endif // NDEBUG

   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);

   EBM_ASSERT(2 <= CountBins(pBinsEnd, aBins, cBytesPerBin)); // we pre-filter out features with only one bin

   const auto* pBin = aBins;
   do {
      ASSERT_BIN_OK(cBytesPerBin, pBin, pBoosterShell->GetDebugMainBinsEnd());
#ifndef NDEBUG
      cSamplesTotalDebug += pBin->GetCountSamples();
      weightTotalDebug += pBin->GetWeight();
#endif // NDEBUG

      const auto* aGradHess = pBin->GetGradientPairs();

      size_t iScore = 0;
      do {
         aSumGradHess[iScore] += aGradHess[iScore];
         ++iScore;
      } while(cScores != iScore);

      pBin = IndexBin(pBin, cBytesPerBin);
   } while(pBinsEnd != pBin);

   EBM_ASSERT(cSamplesTotal == static_cast<size_t>(cSamplesTotalDebug));
   EBM_ASSERT(weightTotalDebug * 0.999 <= weightTotal && weightTotal <= weightTotalDebug * 1.0001);

   if(bUseStackMemory) {
      // if we used registers to collect the gradients and hessians then copy them now to the bin memory

      auto* const aCopyToGradHess = pBinOut->GetGradientPairs();
      size_t iScoreCopy = 0;
      do {
         // do not use memset here so that the compiler can keep aSumGradHessLocal in registers
         aCopyToGradHess[iScoreCopy] = aSumGradHess[iScoreCopy];
         ++iScoreCopy;
      } while(cScores != iScoreCopy);
   }
}

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
// do not inline this.  Not inlining it makes fewer versions that can be called from the more templated functions
template<bool bHessian>
static ErrorEbm Flatten(BoosterShell* const pBoosterShell,
      const bool bNominal,
      const TermBoostFlags flags,
      const FloatCalc regAlpha,
      const FloatCalc regLambda,
      const FloatCalc deltaStepMax,
      const size_t iDimension,
      const Bin<FloatMain, UIntMain, true, true, bHessian>* const* const apBins,
      const Bin<FloatMain, UIntMain, true, true, bHessian>* const* const ppBinsEnd,
      const size_t cSlices) {
   LOG_0(Trace_Verbose, "Entered Flatten");

   EBM_ASSERT(nullptr != pBoosterShell);
   EBM_ASSERT(iDimension <= k_cDimensionsMax);
   EBM_ASSERT(apBins < ppBinsEnd);
   EBM_ASSERT(cSlices <= static_cast<size_t>(ppBinsEnd - apBins));
   EBM_ASSERT(!bNominal || cSlices == static_cast<size_t>(ppBinsEnd - apBins));

   ErrorEbm error;

   Tensor* const pInnerTermUpdate = pBoosterShell->GetInnerTermUpdate();

   error = pInnerTermUpdate->SetCountSlices(iDimension, cSlices);
   if(UNLIKELY(Error_None != error)) {
      // already logged
      return error;
   }

   const BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   const size_t cScores = pBoosterCore->GetCountScores();

   EBM_ASSERT(!IsMultiplyError(cScores, cSlices));
   error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * cSlices);
   if(UNLIKELY(Error_None != error)) {
      // already logged
      return error;
   }

   UIntSplit* pSplit = pInnerTermUpdate->GetSplitPointer(iDimension);

   const Bin<FloatMain, UIntMain, true, true, bHessian>* const* ppBinCur = nullptr;
   if(bNominal) {
      UIntSplit iSplit = 1;
      while(cSlices != iSplit) {
         pSplit[iSplit - 1] = iSplit;
         ++iSplit;
      }
      ppBinCur = reinterpret_cast<const Bin<FloatMain, UIntMain, true, true, bHessian>* const*>(apBins);
   }

   FloatScore* const aUpdateScore = pInnerTermUpdate->GetTensorScoresPointer();
   FloatScore* pUpdateScore = aUpdateScore;

   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);
   auto* const aBins = pBoosterShell->GetBoostingMainBins()->Specialize<FloatMain, UIntMain, true, true, bHessian>();

   EBM_ASSERT(!IsOverflowTreeNodeSize(bHessian, cScores)); // we're accessing allocated memory
   const size_t cBytesPerTreeNode = GetTreeNodeSize(bHessian, cScores);

   auto* const pRootTreeNode = pBoosterShell->GetTreeNodesTemp<bHessian>();
   auto* pTreeNode = pRootTreeNode;

   const bool bUpdateWithHessian = bHessian && !(TermBoostFlags_DisableNewtonUpdate & flags);

   TreeNode<bHessian>* pParent = nullptr;

   while(true) {
      if(UNPREDICTABLE(pTreeNode->AFTER_IsSplit())) {
#ifndef NDEBUG
         pTreeNode->SetDebugProgression(2);
#endif // NDEBUG

         pTreeNode->DECONSTRUCT_SetParent(pParent);
         pParent = pTreeNode;
         pTreeNode = GetLeftNode(pTreeNode->AFTER_GetChildren());
      } else {
         const void* pBinLastOrChildren = pTreeNode->DANGEROUS_GetBinLastOrChildren();
         // if the pointer points to the space within the bins, then the TreeNode could not be split
         // and this TreeNode never had children and we never wrote a pointer to the children in this memory
         if(pBinLastOrChildren < apBins || ppBinsEnd <= pBinLastOrChildren) {
            EBM_ASSERT(IndexTreeNode(pTreeNode, cBytesPerTreeNode) <= pBinLastOrChildren &&
                  pBinLastOrChildren <=
                        IndexTreeNode(pRootTreeNode, pBoosterCore->GetCountBytesTreeNodes() - cBytesPerTreeNode));

            // the node was examined and a gain calculated, so it has left and right children.
            // We can retrieve the split location by looking at where the right child would end its range
            const auto* const pRightChild = GetRightNode(pTreeNode->AFTER_GetChildren(), cBytesPerTreeNode);
            pBinLastOrChildren = pRightChild->BEFORE_GetBinLast();
         }
         const auto* const* const ppBinLast =
               reinterpret_cast<const Bin<FloatMain, UIntMain, true, true, bHessian>* const*>(pBinLastOrChildren);

         EBM_ASSERT(apBins <= ppBinLast);
         EBM_ASSERT(ppBinLast < ppBinsEnd);

         size_t iEdge;
         const auto* const aGradientPair = pTreeNode->GetBin()->GetGradientPairs();
         size_t iScore;
         if(nullptr != ppBinCur) {
            goto determine_bin;
         }

         // if bNominal, check the bin above and below for order
         EBM_ASSERT(apBins == ppBinLast || *(ppBinLast - 1) < *ppBinLast);
         EBM_ASSERT(ppBinLast == ppBinsEnd - 1 || *ppBinLast < *(ppBinLast + 1));

         iEdge = ppBinLast - apBins + 1;

         while(true) {
            iScore = 0;
            do {
               FloatCalc updateScore;
               if(bUpdateWithHessian) {
                  updateScore = -CalcNegUpdate<true>(static_cast<FloatCalc>(aGradientPair[iScore].m_sumGradients),
                        static_cast<FloatCalc>(aGradientPair[iScore].GetHess()),
                        regAlpha,
                        regLambda,
                        deltaStepMax);
               } else {
                  updateScore = -CalcNegUpdate<true>(static_cast<FloatCalc>(aGradientPair[iScore].m_sumGradients),
                        static_cast<FloatCalc>(pTreeNode->GetBin()->GetWeight()),
                        regAlpha,
                        regLambda,
                        deltaStepMax);
               }

               *pUpdateScore = static_cast<FloatScore>(updateScore);
               ++pUpdateScore;

               ++iScore;
            } while(cScores != iScore);

            if(nullptr == ppBinCur) {
               break;
            }
            ++ppBinCur;
            if(ppBinLast < ppBinCur) {
               break;
            }
         determine_bin:;
            const auto* const pBinCur = *ppBinCur;
            const size_t iBin = CountBins(pBinCur, aBins, cBytesPerBin);
            pUpdateScore = aUpdateScore + iBin * cScores;
         }

         pTreeNode = pParent;

         while(true) {
            if(nullptr == pTreeNode) {
               LOG_0(Trace_Verbose, "Exited Flatten");
               return Error_None;
            }
            auto* pChildren = pTreeNode->AFTER_GetChildren();
            if(nullptr != pChildren) {
               // we checked earlier that countBins could be converted to a UIntSplit
               if(nullptr == ppBinCur) {
                  EBM_ASSERT(!IsConvertError<UIntSplit>(iEdge));
                  *pSplit = static_cast<UIntSplit>(iEdge);
                  ++pSplit;
               }
               pParent = pTreeNode;
               pTreeNode->AFTER_SetChildren(nullptr);
               pTreeNode = GetRightNode(pChildren, cBytesPerTreeNode);
               break;
            } else {
               pTreeNode = pTreeNode->DECONSTRUCT_GetParent();
            }
         }
      }
   }
}
WARNING_POP

// TODO: it would be easy for us to implement a -1 lookback where we make the first split, find the second split,
// elimnate the first split and try
//   again on that side, then re-examine the second split again.  For mains this would be very quick we have found that
//   2-3 splits are optimimum. Probably 1 split isn't very good since with 2 splits we can localize a region of high
//   gain in the center somewhere

template<bool bHessian, size_t cCompilerScores>
static int FindBestSplitGain(RandomDeterministic* const pRng,
      BoosterShell* const pBoosterShell,
      const TermBoostFlags flags,
      TreeNode<bHessian, GetArrayScores(cCompilerScores)>* pTreeNode,
      TreeNode<bHessian, GetArrayScores(cCompilerScores)>* const pTreeNodeScratchSpace,
      const size_t cSamplesLeafMin,
      const FloatCalc hessianMin,
      const FloatCalc regAlpha,
      const FloatCalc regLambda,
      const FloatCalc deltaStepMax,
      const MonotoneDirection monotoneDirection) {

   LOG_N(Trace_Verbose,
         "Entered FindBestSplitGain: "
         "pRng=%p, "
         "pBoosterShell=%p, "
         "flags=0x%" UTermBoostFlagsPrintf ", "
         "pTreeNode=%p, "
         "pTreeNodeScratchSpace=%p, "
         "cSamplesLeafMin=%zu, "
         "hessianMin=%le, "
         "regAlpha=%le, "
         "regLambda=%le, "
         "deltaStepMax=%le, "
         "monotoneDirection=%" MonotoneDirectionPrintf,
         static_cast<void*>(pRng),
         static_cast<const void*>(pBoosterShell),
         static_cast<UTermBoostFlags>(flags), // signed to unsigned conversion is defined behavior in C++
         static_cast<void*>(pTreeNode),
         static_cast<void*>(pTreeNodeScratchSpace),
         cSamplesLeafMin,
         static_cast<double>(hessianMin),
         static_cast<double>(regAlpha),
         static_cast<double>(regLambda),
         static_cast<double>(deltaStepMax),
         monotoneDirection);

   if(!pTreeNode->BEFORE_IsSplittable()) {
#ifndef NDEBUG
      pTreeNode->SetDebugProgression(1);
#endif // NDEBUG

      pTreeNode->AFTER_RejectSplit();
      return 1;
   }

   // in the future we could traverse in both directions
   ptrdiff_t incDirectionBytes = static_cast<ptrdiff_t>(+sizeof(void*));

   BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pBoosterCore->GetCountScores());

   auto* const pLeftChild = GetLeftNode(pTreeNodeScratchSpace);
#ifndef NDEBUG
   pLeftChild->SetDebugProgression(0);
#endif // NDEBUG

   Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)> binParent;
   Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)> binInc;

   // if we know how many scores there are, use the memory on the stack where the compiler can optimize access
   static constexpr bool bUseStackMemory = k_dynamicScores != cCompilerScores;
   const auto* const aParentGradHess =
         bUseStackMemory ? binParent.GetGradientPairs() : pTreeNode->GetBin()->GetGradientPairs();
   // use use the pLeftChild since we already have a pointer to it, but it's just temp space
   auto* const aIncGradHess = bUseStackMemory ? binInc.GetGradientPairs() : pLeftChild->GetBin()->GetGradientPairs();
   if(bUseStackMemory) {
      binParent.Copy(cScores, *pTreeNode->GetBin());
   } else {
      binParent.SetCountSamples(pTreeNode->GetBin()->GetCountSamples());
      binParent.SetWeight(pTreeNode->GetBin()->GetWeight());
   }
   binInc.Zero(cScores, aIncGradHess);

   const auto* const* ppBinCur = pTreeNode->BEFORE_GetBinFirst();
   const auto* const* ppBinLast = pTreeNode->BEFORE_GetBinLast();

   pLeftChild->BEFORE_SetBinFirst(ppBinCur);

   EBM_ASSERT(!IsOverflowTreeNodeSize(bHessian, cScores)); // we're accessing allocated memory
   const size_t cBytesPerTreeNode = GetTreeNodeSize(bHessian, cScores);
   auto* const pRightChild = GetRightNode(pTreeNodeScratchSpace, cBytesPerTreeNode);

#ifndef NDEBUG
   pRightChild->SetDebugProgression(0);
#endif // NDEBUG
   pRightChild->BEFORE_SetBinLast(ppBinLast);

   MonotoneDirection monotoneAdjusted = monotoneDirection;
   if(incDirectionBytes < ptrdiff_t{0}) {
      const auto* const* const ppTmp = ppBinCur;
      ppBinCur = ppBinLast;
      ppBinLast = ppTmp;

      monotoneAdjusted = -monotoneAdjusted;
   }

   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);
   EBM_ASSERT(!IsOverflowSplitPositionSize(bHessian, cScores)); // we're accessing allocated memory
   const size_t cBytesPerSplitPosition = GetSplitPositionSize(bHessian, cScores);

   auto* pBestSplitsStart = pBoosterShell->GetSplitPositionsTemp<bHessian, GetArrayScores(cCompilerScores)>();
   auto* pBestSplitsCur = pBestSplitsStart;

   UIntMain cSamplesDec = binParent.GetCountSamples();

   const bool bUseLogitBoost = bHessian && !(TermBoostFlags_DisableNewtonGain & flags);
   const bool bUpdateWithHessian = bHessian && !(TermBoostFlags_DisableNewtonUpdate & flags);

   EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= k_gainMin);
   FloatCalc bestGain = k_gainMin; // it must at least be this, and maybe it needs to be more
   EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= hessianMin);
   EBM_ASSERT(ppBinLast != ppBinCur); // then we would be non-splitable and would have exited above
   do {
      const auto* const pBinCur = *ppBinCur;
      ASSERT_BIN_OK(cBytesPerBin, pBinCur, pBoosterShell->GetDebugMainBinsEnd());

      const UIntMain cSamplesChange = pBinCur->GetCountSamples();
      cSamplesDec -= cSamplesChange;
      if(UNLIKELY(cSamplesDec < cSamplesLeafMin)) {
         // we'll just keep subtracting if we continue, so there won't be any more splits, so we're done
         goto done;
      }
      binInc.SetCountSamples(binInc.GetCountSamples() + cSamplesChange);

      FloatMain hessIncOrig = binInc.GetWeight() + pBinCur->GetWeight();
      FloatCalc hessDec = static_cast<FloatCalc>(binParent.GetWeight() - hessIncOrig);

      binInc.SetWeight(hessIncOrig);
      FloatCalc hessInc = static_cast<FloatCalc>(hessIncOrig);

      FloatCalc hessDecUpdate = hessDec;
      FloatCalc hessIncUpdate = hessInc;

      const auto* const aBinGradHess = pBinCur->GetGradientPairs();
      bool bLegal = true;
      FloatCalc gain = 0;
      size_t iScore = 0;
      do {
         const FloatMain gradIncOrig = aIncGradHess[iScore].m_sumGradients + aBinGradHess[iScore].m_sumGradients;
         aIncGradHess[iScore].m_sumGradients = gradIncOrig;
         const FloatCalc gradDec = static_cast<FloatCalc>(aParentGradHess[iScore].m_sumGradients - gradIncOrig);

         if(bHessian) {
            const FloatMain newHessIncOrig = aIncGradHess[iScore].GetHess() + aBinGradHess[iScore].GetHess();
            aIncGradHess[iScore].SetHess(newHessIncOrig);
            const FloatCalc newHessDec = static_cast<FloatCalc>(aParentGradHess[iScore].GetHess() - newHessIncOrig);
            const FloatCalc newHessInc = static_cast<FloatCalc>(newHessIncOrig);
            if(bUseLogitBoost) {
               hessInc = newHessInc;
               hessDec = newHessDec;
            }
            if(bUpdateWithHessian) {
               hessIncUpdate = newHessInc;
               hessDecUpdate = newHessDec;
            }
         }
         if(UNLIKELY(hessDec < hessianMin)) {
            // we'll just keep subtracting if we continue, so there won't be any more splits, so we're done
            goto done;
         }
         if(UNLIKELY(hessInc < hessianMin)) {
            bLegal = false;
         }

         const FloatCalc gradInc = static_cast<FloatCalc>(gradIncOrig);

         if(MONOTONE_NONE != monotoneAdjusted) {
            const FloatCalc negUpdateDec =
                  CalcNegUpdate<true>(gradDec, hessDecUpdate, regAlpha, regLambda, deltaStepMax);
            const FloatCalc negUpdateInc =
                  CalcNegUpdate<true>(gradInc, hessIncUpdate, regAlpha, regLambda, deltaStepMax);
            if(MonotoneDirection{0} < monotoneAdjusted) {
               if(negUpdateInc < negUpdateDec) {
                  bLegal = false;
               }
            } else {
               EBM_ASSERT(monotoneAdjusted < MonotoneDirection{0});
               if(negUpdateDec < negUpdateInc) {
                  bLegal = false;
               }
            }
         }

         const FloatCalc gainDec = CalcPartialGain<false>(gradDec, hessDec, regAlpha, regLambda, deltaStepMax);
         EBM_ASSERT(!bLegal || std::isnan(gainDec) || 0 <= gainDec);
         gain += gainDec;

         // if bLegal was set to false, hessInc can be negative
         const FloatCalc gainInc = CalcPartialGain<true>(gradInc, hessInc, regAlpha, regLambda, deltaStepMax);
         EBM_ASSERT(!bLegal || std::isnan(gainInc) || 0 <= gainInc);
         gain += gainInc;

         ++iScore;
      } while(cScores != iScore);
      EBM_ASSERT(std::isnan(gain) || 0 <= gain);

      if(!bLegal || binInc.GetCountSamples() < cSamplesLeafMin) {
         goto next;
      }

      if(UNLIKELY(/* NaN */ !LIKELY(gain < bestGain))) {
         // propagate NaN values since we stop boosting when we see them

         pBestSplitsCur = UNPREDICTABLE(bestGain == gain) ? pBestSplitsCur : pBestSplitsStart;
         bestGain = gain;

         pBestSplitsCur->SetBinPosition(ppBinCur);
         pBestSplitsCur->SetIncDirectionBytes(incDirectionBytes);

         pBestSplitsCur->GetBinSum()->Copy(cScores, binInc, aIncGradHess);

         pBestSplitsCur = IndexSplitPosition(pBestSplitsCur, cBytesPerSplitPosition);
      } else {
         EBM_ASSERT(!std::isnan(gain));
      }

   next:;
      ppBinCur = IndexByte(ppBinCur, incDirectionBytes);
   } while(ppBinLast != ppBinCur);

done:;

   if(UNLIKELY(pBestSplitsStart == pBestSplitsCur)) {
      // no valid splits found
      EBM_ASSERT(k_gainMin == bestGain);

#ifndef NDEBUG
      pTreeNode->SetDebugProgression(1);
#endif // NDEBUG

      pTreeNode->AFTER_RejectSplit();
      return 1;
   }
   EBM_ASSERT(std::isnan(bestGain) || k_gainMin <= bestGain);

   if(UNLIKELY(/* NaN */ !LIKELY(bestGain <= std::numeric_limits<FloatCalc>::max()))) {
      // this tests for NaN and +inf

      // we need this test since the priority queue in the function that calls us cannot accept a NaN value
      // since we would break weak ordering with non-ordered NaN comparisons, thus create undefined behavior

#ifndef NDEBUG
      pTreeNode->SetDebugProgression(1);
#endif // NDEBUG

      pTreeNode->AFTER_RejectSplit();
      return -1; // exit boosting with overflow
   }

   FloatCalc hessOverwrite = static_cast<FloatCalc>(binParent.GetWeight());
   size_t iScoreParent = 0;
   do {
      const FloatCalc gradParent = static_cast<FloatCalc>(aParentGradHess[iScoreParent].m_sumGradients);
      if(bUseLogitBoost) {
         hessOverwrite = static_cast<FloatCalc>(aParentGradHess[iScoreParent].GetHess());
      }
      // we would not get here unless there was a split, so both sides must meet the minHessian reqirement
      EBM_ASSERT(hessianMin <= hessOverwrite);
      const FloatCalc gainParent = CalcPartialGain<false>(gradParent, hessOverwrite, regAlpha, regLambda, deltaStepMax);
      EBM_ASSERT(std::isnan(gainParent) || 0 <= gainParent);
      bestGain -= gainParent;

      ++iScoreParent;
   } while(cScores != iScoreParent);

   // bestGain could be -inf if the partial gain on the children reached a number close to +inf and then
   // the children were -inf due to floating point noise.
   EBM_ASSERT(std::isnan(bestGain) || -std::numeric_limits<FloatCalc>::infinity() == bestGain ||
         k_epsilonNegativeGainAllowed <= bestGain);
   EBM_ASSERT(std::numeric_limits<FloatCalc>::infinity() != bestGain);

   EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= k_gainMin);
   if(UNLIKELY(/* NaN */ !LIKELY(k_gainMin <= bestGain))) {
      // do not allow splits on gains that are too small
      // also filter out slightly negative numbers that can arrise from floating point noise

#ifndef NDEBUG
      pTreeNode->SetDebugProgression(1);
#endif // NDEBUG

      pTreeNode->AFTER_RejectSplit();

      // but if the parent partial gain overflowed to +inf and thus we got a -inf gain, then handle as an overflow
      return /* NaN */ std::numeric_limits<FloatCalc>::lowest() <= bestGain ? 1 : -1;
   }
   EBM_ASSERT(!std::isnan(bestGain));
   EBM_ASSERT(!std::isinf(bestGain));
   EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= bestGain);
   EBM_ASSERT(k_gainMin <= bestGain);

   const size_t cTies = CountSplitPositions(pBestSplitsCur, pBestSplitsStart, cBytesPerSplitPosition);
   if(UNLIKELY(1 < cTies)) {
      const size_t iRandom = pRng->NextFast(cTies);
      pBestSplitsStart = IndexSplitPosition(pBestSplitsStart, cBytesPerSplitPosition * iRandom);
   }

   const auto* const* const ppBestBinPosition = pBestSplitsStart->GetBinPosition();

   TreeNode<bHessian, GetArrayScores(cCompilerScores)>* pIncChild;
   TreeNode<bHessian, GetArrayScores(cCompilerScores)>* pDecChild;
   if(ptrdiff_t{0} <= pBestSplitsStart->GetIncDirectionBytes()) {
      pIncChild = pLeftChild;
      pDecChild = pRightChild;

      pLeftChild->BEFORE_SetBinLast(ppBestBinPosition);

      const auto* const ppBinFirst = ppBestBinPosition + 1;
      ASSERT_BIN_OK(cBytesPerBin, *ppBinFirst, pBoosterShell->GetDebugMainBinsEnd());
      pRightChild->BEFORE_SetBinFirst(ppBinFirst);
   } else {
      pIncChild = pRightChild;
      pDecChild = pLeftChild;

      pRightChild->BEFORE_SetBinFirst(ppBestBinPosition);

      const auto* const ppBinLeftLast = ppBestBinPosition - 1;
      ASSERT_BIN_OK(cBytesPerBin, *ppBinLeftLast, pBoosterShell->GetDebugMainBinsEnd());
      pLeftChild->BEFORE_SetBinLast(ppBinLeftLast);
   }

   memcpy(pIncChild->GetBin(), pBestSplitsStart->GetBinSum(), cBytesPerBin);

   pDecChild->GetBin()->SetCountSamples(binParent.GetCountSamples() - pBestSplitsStart->GetBinSum()->GetCountSamples());
   pDecChild->GetBin()->SetWeight(binParent.GetWeight() - pBestSplitsStart->GetBinSum()->GetWeight());

   auto* const aDecGradHess = pDecChild->GetBin()->GetGradientPairs();
   const auto* const aBestGradHess = pBestSplitsStart->GetBinSum()->GetGradientPairs();
   size_t iScoreCopy = 0;
   do {
      auto temp = aParentGradHess[iScoreCopy];
      temp -= aBestGradHess[iScoreCopy];
      aDecGradHess[iScoreCopy] = temp;

      ++iScoreCopy;
   } while(cScores != iScoreCopy);

   // IMPORTANT!! : We need to finish all our calls that use pTreeNode->m_UNION.m_beforeGainCalc BEFORE setting
   // anything in m_UNION.m_afterGainCalc as we do below this comment!
#ifndef NDEBUG
   pTreeNode->SetDebugProgression(1);
#endif // NDEBUG

   pTreeNode->AFTER_SetChildren(pTreeNodeScratchSpace);
   pTreeNode->AFTER_SetSplitGain(bestGain);

   LOG_N(Trace_Verbose, "Exited FindBestSplitGain: gain=%le", bestGain);

   return 0;
}

template<bool bHessian> class CompareNodeGain final {
 public:
   INLINE_ALWAYS bool operator()(
         const TreeNode<bHessian>* const& lhs, const TreeNode<bHessian>* const& rhs) const noexcept {
      // NEVER check for exact equality (as a precondition is ok), since then we'd violate the weak ordering rule
      // https://medium.com/@shiansu/strict-weak-ordering-and-the-c-stl-f7dcfa4d4e07
      return lhs->AFTER_GetSplitGain() < rhs->AFTER_GetSplitGain();
   }
};

template<bool bHessian, size_t cCompilerScores> class CompareBin final {
   bool m_bHessianRuntime;
   FloatCalc m_regAlpha;
   FloatCalc m_regLambda;
   FloatCalc m_deltaStepMax;

 public:
   INLINE_ALWAYS CompareBin(const bool bHessianRuntime,
         const FloatCalc regAlpha,
         const FloatCalc regLambda,
         const FloatCalc deltaStepMax) {
      m_bHessianRuntime = bHessianRuntime;
      m_regAlpha = regAlpha;
      m_regLambda = regLambda;
      m_deltaStepMax = deltaStepMax;
   }

   INLINE_ALWAYS bool operator()(
         const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>*& lhs,
         const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>*& rhs) const noexcept {
      // NEVER check for exact equality (as a precondition is ok), since then we'd violate the weak ordering rule
      // https://medium.com/@shiansu/strict-weak-ordering-and-the-c-stl-f7dcfa4d4e07

      const bool bUpdateWithHessian = bHessian && m_bHessianRuntime;

      const FloatCalc hess1 =
            static_cast<FloatCalc>(bUpdateWithHessian ? lhs->GetGradientPairs()[0].GetHess() : lhs->GetWeight());
      const FloatCalc val1 = CalcNegUpdate<true>(static_cast<FloatCalc>(lhs->GetGradientPairs()[0].m_sumGradients),
            hess1,
            m_regAlpha,
            m_regLambda,
            m_deltaStepMax);

      const FloatCalc hess2 =
            static_cast<FloatCalc>(bUpdateWithHessian ? rhs->GetGradientPairs()[0].GetHess() : rhs->GetWeight());
      const FloatCalc val2 = CalcNegUpdate<true>(static_cast<FloatCalc>(rhs->GetGradientPairs()[0].m_sumGradients),
            hess2,
            m_regAlpha,
            m_regLambda,
            m_deltaStepMax);

      if(val1 == val2) {
         return lhs < rhs;
      }
      return val1 < val2;
   }
};

template<bool bHessian, size_t cCompilerScores> class PartitionOneDimensionalBoostingInternal final {
 public:
   PartitionOneDimensionalBoostingInternal() = delete; // this is a static class.  Do not construct

   static ErrorEbm Func(RandomDeterministic* const pRng,
         BoosterShell* const pBoosterShell,
         bool bNominal,
         const TermBoostFlags flags,
         const size_t cBins,
         const size_t iDimension,
         const size_t cSamplesLeafMin,
         const FloatCalc hessianMin,
         const FloatCalc regAlpha,
         const FloatCalc regLambda,
         const FloatCalc deltaStepMax,
         const size_t cSplitsMax,
         const MonotoneDirection monotoneDirection,
         const size_t cSamplesTotal,
         const FloatMain weightTotal,
         double* const pTotalGain) {
      EBM_ASSERT(2 <= cBins); // filter these out at the start where we can handle this case easily
      EBM_ASSERT(1 <= cSplitsMax); // filter these out at the start where we can handle this case easily
      EBM_ASSERT(nullptr != pTotalGain);

      BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
      const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pBoosterCore->GetCountScores());
      const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);
      auto* const pRootTreeNode = pBoosterShell->GetTreeNodesTemp<bHessian, GetArrayScores(cCompilerScores)>();

      // we can only sort if there's a single sortable index, so 1 score value
      bNominal = 1 == cCompilerScores && bNominal && (0 == (TermBoostFlags_DisableCategorical & flags));

#ifndef NDEBUG
      pRootTreeNode->SetDebugProgression(0);
#endif // NDEBUG

      auto* const aBins =
            pBoosterShell->GetBoostingMainBins()
                  ->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();
      auto* const pBinsEnd = IndexBin(aBins, cBytesPerBin * cBins);

      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>** const apBins =
            reinterpret_cast<const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>**>(
                  pBinsEnd);
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>** const ppBinsEnd =
            apBins + cBins;

      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>** ppBin = apBins;
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* pBin = aBins;
      do {
         *ppBin = pBin;
         pBin = IndexBin(pBin, cBytesPerBin);
         ++ppBin;
      } while(ppBinsEnd != ppBin);

      if(bNominal) {
         std::sort(apBins,
               ppBinsEnd,
               CompareBin<bHessian, cCompilerScores>(
                     !(TermBoostFlags_DisableNewtonUpdate & flags), regAlpha, regLambda, deltaStepMax));
      }

      pRootTreeNode->BEFORE_SetBinFirst(apBins);
      pRootTreeNode->BEFORE_SetBinLast(ppBinsEnd - 1);
      ASSERT_BIN_OK(cBytesPerBin, *pRootTreeNode->BEFORE_GetBinLast(), pBoosterShell->GetDebugMainBinsEnd());

      SumAllBins<bHessian, cCompilerScores>(
            pBoosterShell, pBinsEnd, cSamplesTotal, weightTotal, pRootTreeNode->GetBin());

      EBM_ASSERT(!IsOverflowTreeNodeSize(bHessian, cScores));
      const size_t cBytesPerTreeNode = GetTreeNodeSize(bHessian, cScores);

      auto* pTreeNodeScratchSpace = IndexTreeNode(pRootTreeNode, cBytesPerTreeNode);

      int retFind = FindBestSplitGain<bHessian, cCompilerScores>(pRng,
            pBoosterShell,
            flags,
            pRootTreeNode,
            pTreeNodeScratchSpace,
            cSamplesLeafMin,
            hessianMin,
            regAlpha,
            regLambda,
            deltaStepMax,
            monotoneDirection);
      size_t cSplitsRemaining = cSplitsMax;
      FloatCalc totalGain = 0;
      if(UNLIKELY(0 != retFind)) {
         // there will be no splits at all
         if(UNLIKELY(retFind < 0)) {
            // Any negative return means there was an overflow. Signal the overflow by making gain infinite here.
            // We'll flip it later to a negative number to signal to the caller that there was an overflow.
            totalGain = std::numeric_limits<FloatCalc>::infinity();
         }
      } else {
         // our priority queue comparison function cannot handle NaN gains so we filter out before
         EBM_ASSERT(!std::isnan(pRootTreeNode->AFTER_GetSplitGain()));
         EBM_ASSERT(!std::isinf(pRootTreeNode->AFTER_GetSplitGain()));
         EBM_ASSERT(0 <= pRootTreeNode->AFTER_GetSplitGain());

         try {
            // TODO: someday see if we can replace this with an in-class priority queue that stores it's info inside
            //       the TreeNode datastructure
            std::priority_queue<TreeNode<bHessian>*, std::vector<TreeNode<bHessian>*>, CompareNodeGain<bHessian>>
                  nodeGainRanking;

            auto* pTreeNode = pRootTreeNode;

            // The root node used a left and right leaf, so reserve it here
            pTreeNodeScratchSpace = IndexTreeNode(pTreeNodeScratchSpace, cBytesPerTreeNode << 1);

            goto skip_first_push_pop;

            do {
               // there is no way to get the top and pop at the same time.. would be good to get a better queue, but our
               // code isn't bottlenecked by it
               pTreeNode = nodeGainRanking.top()->template Upgrade<GetArrayScores(cCompilerScores)>();
               // In theory we can have nodes with equal gain values here, but this is very very rare to occur in
               // practice We handle equal gain values in FindBestSplitGain because we can have zero instances in bins,
               // in which case it occurs, but those equivalent situations have been cleansed by the time we reach this
               // code, so the only realistic scenario where we might get equivalent gains is if we had an almost
               // symetric distribution samples bin distributions AND two tail ends that happen to have the same
               // statistics AND either this is our first split, or we've only made a single split in the center in the
               // case where there is symetry in the center Even if all of these things are true, after one non-symetric
               // split, we won't see that scenario anymore since the gradients won't be symetric anymore.  This is so
               // rare, and limited to one split, so we shouldn't bother to handle it since the complexity of doing so
               // outweights the benefits.
               nodeGainRanking.pop();

            skip_first_push_pop:

               // pTreeNode had the highest gain of all the available Nodes, so we will split it.

               // get the gain first, since calling AFTER_SplitNode destroys it
               const FloatCalc totalGainUpdate = pTreeNode->AFTER_GetSplitGain();
               EBM_ASSERT(!std::isnan(totalGainUpdate));
               EBM_ASSERT(!std::isinf(totalGainUpdate));
               EBM_ASSERT(0 <= totalGainUpdate);
               totalGain += totalGainUpdate;

               pTreeNode->AFTER_SplitNode();

               auto* const pLeftChild = GetLeftNode(pTreeNode->AFTER_GetChildren());

               retFind = FindBestSplitGain<bHessian, cCompilerScores>(pRng,
                     pBoosterShell,
                     flags,
                     pLeftChild,
                     pTreeNodeScratchSpace,
                     cSamplesLeafMin,
                     hessianMin,
                     regAlpha,
                     regLambda,
                     deltaStepMax,
                     monotoneDirection);
               // if FindBestSplitGain returned -1 to indicate an
               // overflow ignore it here. We successfully made a root node split, so we might as well continue
               // with the successful tree that we have which can make progress in boosting down the residuals
               if(0 == retFind) {
                  pTreeNodeScratchSpace = IndexTreeNode(pTreeNodeScratchSpace, cBytesPerTreeNode << 1);
                  // our priority queue comparison function cannot handle NaN gains so we filter out before
                  EBM_ASSERT(!std::isnan(pLeftChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(!std::isinf(pLeftChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(0 <= pLeftChild->AFTER_GetSplitGain());
                  nodeGainRanking.push(pLeftChild->Downgrade());
               }

               auto* const pRightChild = GetRightNode(pTreeNode->AFTER_GetChildren(), cBytesPerTreeNode);

               retFind = FindBestSplitGain<bHessian, cCompilerScores>(pRng,
                     pBoosterShell,
                     flags,
                     pRightChild,
                     pTreeNodeScratchSpace,
                     cSamplesLeafMin,
                     hessianMin,
                     regAlpha,
                     regLambda,
                     deltaStepMax,
                     monotoneDirection);
               // if FindBestSplitGain returned -1 to indicate an
               // overflow ignore it here. We successfully made a root node split, so we might as well continue
               // with the successful tree that we have which can make progress in boosting down the residuals
               if(0 == retFind) {
                  pTreeNodeScratchSpace = IndexTreeNode(pTreeNodeScratchSpace, cBytesPerTreeNode << 1);
                  // our priority queue comparison function cannot handle NaN gains so we filter out before
                  EBM_ASSERT(!std::isnan(pRightChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(!std::isinf(pRightChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(0 <= pRightChild->AFTER_GetSplitGain());
                  nodeGainRanking.push(pRightChild->Downgrade());
               }

               --cSplitsRemaining;
            } while(0 != cSplitsRemaining && UNLIKELY(!nodeGainRanking.empty()));

            EBM_ASSERT(!std::isnan(totalGain));
            EBM_ASSERT(0 <= totalGain);

            EBM_ASSERT(CountBytes(pTreeNodeScratchSpace, pRootTreeNode) <= pBoosterCore->GetCountBytesTreeNodes());
         } catch(const std::bad_alloc&) {
            // calling anything inside nodeGainRanking can throw exceptions
            LOG_0(Trace_Warning, "WARNING PartitionOneDimensionalBoosting out of memory exception");
            return Error_OutOfMemory;
         } catch(...) {
            // calling anything inside nodeGainRanking can throw exceptions
            LOG_0(Trace_Warning, "WARNING PartitionOneDimensionalBoosting exception");
            return Error_UnexpectedInternal;
         }
      }
      *pTotalGain = static_cast<double>(totalGain);
      const size_t cSlices = bNominal ? cBins : cSplitsMax - cSplitsRemaining + 1;
      return Flatten<bHessian>(pBoosterShell,
            bNominal,
            flags,
            regAlpha,
            regLambda,
            deltaStepMax,
            iDimension,
            reinterpret_cast<const Bin<FloatMain, UIntMain, true, true, bHessian>* const*>(apBins),
            reinterpret_cast<const Bin<FloatMain, UIntMain, true, true, bHessian>* const*>(ppBinsEnd),
            cSlices);
   }
};

extern ErrorEbm PartitionOneDimensionalBoosting(RandomDeterministic* const pRng,
      BoosterShell* const pBoosterShell,
      const bool bNominal,
      const TermBoostFlags flags,
      const size_t cBins,
      const size_t iDimension,
      const size_t cSamplesLeafMin,
      const FloatCalc hessianMin,
      const FloatCalc regAlpha,
      const FloatCalc regLambda,
      const FloatCalc deltaStepMax,
      const size_t cSplitsMax,
      const MonotoneDirection monotoneDirection,
      const size_t cSamplesTotal,
      const FloatMain weightTotal,
      double* const pTotalGain) {
   LOG_0(Trace_Verbose, "Entered PartitionOneDimensionalBoosting");

   ErrorEbm error;

   BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   const size_t cRuntimeScores = pBoosterCore->GetCountScores();

   EBM_ASSERT(1 <= cRuntimeScores);
   if(pBoosterCore->IsHessian()) {
      if(size_t{1} == cRuntimeScores) {
         error = PartitionOneDimensionalBoostingInternal<true, k_oneScore>::Func(pRng,
               pBoosterShell,
               bNominal,
               flags,
               cBins,
               iDimension,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               cSplitsMax,
               monotoneDirection,
               cSamplesTotal,
               weightTotal,
               pTotalGain);
      } else if(size_t{3} == cRuntimeScores) {
         // 3 classes
         error = PartitionOneDimensionalBoostingInternal<true, 3>::Func(pRng,
               pBoosterShell,
               bNominal,
               flags,
               cBins,
               iDimension,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               cSplitsMax,
               monotoneDirection,
               cSamplesTotal,
               weightTotal,
               pTotalGain);
      } else {
         // muticlass
         error = PartitionOneDimensionalBoostingInternal<true, k_dynamicScores>::Func(pRng,
               pBoosterShell,
               bNominal,
               flags,
               cBins,
               iDimension,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               cSplitsMax,
               monotoneDirection,
               cSamplesTotal,
               weightTotal,
               pTotalGain);
      }
   } else {
      if(size_t{1} == cRuntimeScores) {
         error = PartitionOneDimensionalBoostingInternal<false, k_oneScore>::Func(pRng,
               pBoosterShell,
               bNominal,
               flags,
               cBins,
               iDimension,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               cSplitsMax,
               monotoneDirection,
               cSamplesTotal,
               weightTotal,
               pTotalGain);
      } else {
         // Odd: gradient multiclass. Allow it, but do not optimize for it
         error = PartitionOneDimensionalBoostingInternal<false, k_dynamicScores>::Func(pRng,
               pBoosterShell,
               bNominal,
               flags,
               cBins,
               iDimension,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               cSplitsMax,
               monotoneDirection,
               cSamplesTotal,
               weightTotal,
               pTotalGain);
      }
   }

   LOG_0(Trace_Verbose, "Exited PartitionOneDimensionalBoosting");

   return error;
}

} // namespace DEFINED_ZONE_NAME
