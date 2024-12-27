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
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const aBins,
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
WARNING_DISABLE_UNINITIALIZED_LOCAL_POINTER
// do not inline this.  Not inlining it makes fewer versions that can be called from the more templated functions
template<bool bHessian>
static ErrorEbm Flatten(BoosterShell* const pBoosterShell,
      const bool bMissing,
      const bool bNominal,
      const TermBoostFlags flags,
      const FloatCalc regAlpha,
      const FloatCalc regLambda,
      const FloatCalc deltaStepMax,
      const size_t iDimension,
      const Bin<FloatMain, UIntMain, true, true, bHessian>* const* const apBins,
      const TreeNode<bHessian>* pMissingValueTreeNode,
      const size_t cSlices,
      const size_t cBins) {
   LOG_0(Trace_Verbose, "Entered Flatten");

   EBM_ASSERT(nullptr != pBoosterShell);
   EBM_ASSERT(iDimension <= k_cDimensionsMax);
   EBM_ASSERT(nullptr != apBins);
   EBM_ASSERT(1 <= cSlices);
   EBM_ASSERT(2 <= cBins);
   EBM_ASSERT(cSlices <= cBins);
   EBM_ASSERT(!bNominal || cSlices == cBins);

   ErrorEbm error;

#ifndef NDEBUG
   auto* const pRootTreeNodeDebug = pBoosterShell->GetTreeNodesTemp<bHessian>();
   size_t cSamplesExpectedDebug = static_cast<size_t>(pRootTreeNodeDebug->GetBin()->GetCountSamples());
   size_t cSamplesTotalDebug = 0;
#endif // NDEBUG

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

   FloatScore* const aUpdateScore = pInnerTermUpdate->GetTensorScoresPointer();
   FloatScore* pUpdateScore;

   auto* const aBins = pBoosterShell->GetBoostingMainBins()->Specialize<FloatMain, UIntMain, true, true, bHessian>();

   const Bin<FloatMain, UIntMain, true, true, bHessian>* pMissingBin = nullptr;
   const Bin<FloatMain, UIntMain, true, true, bHessian>* const* ppBinCur = nullptr;
   if(bNominal) {
      UIntSplit iSplit = 1;
      while(cSlices != iSplit) {
         pSplit[iSplit - 1] = iSplit;
         ++iSplit;
      }
      ppBinCur = reinterpret_cast<const Bin<FloatMain, UIntMain, true, true, bHessian>* const*>(apBins);
   } else {
      pUpdateScore = aUpdateScore;

      if(bMissing) {
         EBM_ASSERT(2 <= cSlices); // no cuts if there was only missing bin

         if(TermBoostFlags_MissingSeparate & flags) {
            pMissingBin = aBins;
         }

         // always put a split on the missing bin
         *pSplit = 1;
         ++pSplit;

         // pUpdateScore is overwritten later if bNominal
         pUpdateScore += cScores;
      }
   }

   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);

   EBM_ASSERT(!IsOverflowTreeNodeSize(bHessian, cScores)); // we're accessing allocated memory
   const size_t cBytesPerTreeNode = GetTreeNodeSize(bHessian, cScores);

   auto* const pRootTreeNode = pBoosterShell->GetTreeNodesTemp<bHessian>();
   auto* pTreeNode = pRootTreeNode;

   const bool bUpdateWithHessian = bHessian && !(TermBoostFlags_DisableNewtonUpdate & flags);

   TreeNode<bHessian>* pParent = nullptr;
   bool bDone = false;

   while(true) {
      if(UNPREDICTABLE(pTreeNode->AFTER_IsSplit())) {
         auto* const pLeftChild = pTreeNode->DECONSTRUCT_TraverseLeftAndMark(pParent);
         pParent = pTreeNode;
         pTreeNode = pLeftChild;
      } else {
         const Bin<FloatMain, UIntMain, true, true, bHessian>* const* ppBinLast;
         // if the pointer points to the space within the bins, then the TreeNode could not be split
         // and this TreeNode never had children and we never wrote a pointer to the children in this memory
         if(pTreeNode->AFTER_IsSplittable()) {
            auto* const pChildren = pTreeNode->AFTER_GetChildren();

            EBM_ASSERT(IndexTreeNode(pTreeNode, cBytesPerTreeNode) <= pChildren &&
                  pChildren <=
                        IndexTreeNode(pRootTreeNode, pBoosterCore->GetCountBytesTreeNodes() - cBytesPerTreeNode));

            if(pMissingValueTreeNode == GetLeftNode(pChildren)) {
               EBM_ASSERT(nullptr == pMissingBin);
               pMissingBin = pTreeNode->GetBin();
            }

            // the node was examined and a gain calculated, so it has left and right children.
            // We can retrieve the split location by looking at where the right child would end its range
            const auto* const pRightChild = GetRightNode(pChildren, cBytesPerTreeNode);
            ppBinLast = pRightChild->BEFORE_GetBinLast();

            if(pMissingValueTreeNode == pRightChild) {
               EBM_ASSERT(nullptr == pMissingBin);
               pMissingBin = pTreeNode->GetBin();
            }
         } else {
            ppBinLast = pTreeNode->BEFORE_GetBinLast();
            if(pMissingValueTreeNode == pTreeNode) {
               EBM_ASSERT(nullptr == pMissingBin);
               pMissingBin = pTreeNode->GetBin();
            }
         }

         EBM_ASSERT(apBins <= ppBinLast);
         EBM_ASSERT(ppBinLast < apBins +
                     (cBins -
                           (nullptr != pMissingValueTreeNode ||
                                             bMissing && !bNominal && (TermBoostFlags_MissingSeparate & flags) ?
                                       size_t{1} :
                                       size_t{0})));

#ifndef NDEBUG
         cSamplesTotalDebug += static_cast<size_t>(pTreeNode->GetBin()->GetCountSamples());
#endif // NDEBUG

         const auto* const pGradientPairEnd = pTreeNode->GetBin()->GetGradientPairs() + cScores;
         size_t iEdge;
         FloatScore hess;
         const GradientPair<FloatMain, bHessian>* pGradientPair;
         if(nullptr != ppBinCur) {
            goto determine_bin;
         }
         EBM_ASSERT(!bNominal);

         iEdge = ppBinLast - apBins + 1;

         while(true) { // not a real loop
            if(bMissing) {
               if(TermBoostFlags_MissingLow & flags) {
                  if(nullptr == pMissingBin) {
                     pMissingBin = pTreeNode->GetBin();
                  }
                  if(1 == iEdge) {
                     // this cut would isolate the missing bin, but we handle those scores separately
                     break;
                  }
               } else {
                  ++iEdge; // missing is at index 0 in the model, so we are offset by one
                  if(TermBoostFlags_MissingHigh & flags) {
                     pMissingBin = pTreeNode->GetBin();
                     EBM_ASSERT(iEdge <= cBins + 1);
                     if(bDone) {
                        // this cut would isolate the missing bin, but we handle those scores separately
                        goto done;
                     }
                  }
               }
            }

            while(true) {
               hess = static_cast<FloatCalc>(pTreeNode->GetBin()->GetWeight());
               pGradientPair = pTreeNode->GetBin()->GetGradientPairs();
               do {
                  if(bUpdateWithHessian) {
                     hess = static_cast<FloatCalc>(pGradientPair->GetHess());
                  }
                  const FloatCalc updateScore = -CalcNegUpdate<true>(
                        static_cast<FloatCalc>(pGradientPair->m_sumGradients), hess, regAlpha, regLambda, deltaStepMax);

                  EBM_ASSERT(pUpdateScore < aUpdateScore + cScores * cSlices);
                  *pUpdateScore = static_cast<FloatScore>(updateScore);
                  ++pUpdateScore;

                  ++pGradientPair;
               } while(pGradientPairEnd != pGradientPair);

               if(nullptr == ppBinCur) {
                  break;
               }
               EBM_ASSERT(bNominal);

               ++ppBinCur;
               if(ppBinLast < ppBinCur) {
                  break;
               }
            determine_bin:;
               const auto* const pBinCur = *ppBinCur;
               const size_t iBin = CountBins(pBinCur, aBins, cBytesPerBin);
               pUpdateScore = aUpdateScore + iBin * cScores;
            }

            break;
         }

         pTreeNode = pParent;

         while(true) {
            if(nullptr == pTreeNode) {
               goto done;
            }
            if(!pTreeNode->DECONSTRUCT_IsRightChildTraversal()) {
               // we checked earlier that countBins could be converted to a UIntSplit
               if(nullptr == ppBinCur) {
                  EBM_ASSERT(!bNominal);

                  while(true) { // not a real loop
                     if(bMissing) {
                        if(TermBoostFlags_MissingLow & flags) {
                           if(1 == iEdge) {
                              // this cut would isolate the missing bin, but missing already has a cut
                              break;
                           }
                        } else if(TermBoostFlags_MissingHigh & flags) {
                           EBM_ASSERT(iEdge <= cBins);
                           if(cBins == iEdge) {
                              // This cut would isolate the missing bin, but missing already has a cut.
                              // We still need to find the missing bin though in the tree, so continue.
                              bDone = true;
                              break;
                           }
                        }
                     }

                     EBM_ASSERT(!IsConvertError<UIntSplit>(iEdge));
                     EBM_ASSERT(pSplit < cSlices - 1 + pInnerTermUpdate->GetSplitPointer(iDimension));
                     *pSplit = static_cast<UIntSplit>(iEdge);
                     ++pSplit;

                     break;
                  }
               }
               pParent = pTreeNode;
               pTreeNode = pTreeNode->DECONSTRUCT_TraverseRightAndMark(cBytesPerTreeNode);
               break;
            } else {
               pTreeNode = pTreeNode->DECONSTRUCT_GetParent();
            }
         }
      }
   }

done:;

#ifndef NDEBUG
   EBM_ASSERT(cSamplesTotalDebug == cSamplesExpectedDebug);
   EBM_ASSERT(bNominal || pUpdateScore == aUpdateScore + cScores * cSlices);
   EBM_ASSERT(bNominal || pSplit == cSlices - 1 + pInnerTermUpdate->GetSplitPointer(iDimension));

   UIntSplit prevDebug = 0;
   for(size_t iDebug = 0; iDebug < cSlices - 1; ++iDebug) {
      UIntSplit curDebug = pInnerTermUpdate->GetSplitPointer(iDimension)[iDebug];
      EBM_ASSERT(prevDebug < curDebug);
      prevDebug = curDebug;
   }
   EBM_ASSERT(prevDebug < cBins);
#endif

   if(nullptr != pMissingBin) {
      EBM_ASSERT(bMissing);

      FloatScore hess = static_cast<FloatCalc>(pMissingBin->GetWeight());
      const auto* pGradientPair = pMissingBin->GetGradientPairs();
      const auto* const pGradientPairEnd = pGradientPair + cScores;
      FloatScore* pMissingUpdateScore = aUpdateScore;
      do {
         if(bUpdateWithHessian) {
            hess = static_cast<FloatCalc>(pGradientPair->GetHess());
         }
         FloatCalc updateScore = -CalcNegUpdate<true>(
               static_cast<FloatCalc>(pGradientPair->m_sumGradients), hess, regAlpha, regLambda, deltaStepMax);

         *pMissingUpdateScore = static_cast<FloatScore>(updateScore);
         ++pMissingUpdateScore;

         ++pGradientPair;
      } while(pGradientPairEnd != pGradientPair);
   } else {
      EBM_ASSERT(!bMissing || bNominal && (TermBoostFlags_MissingCategory & flags));
   }

   LOG_0(Trace_Verbose, "Exited Flatten");
   return Error_None;
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
      const MonotoneDirection monotoneDirection,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const pMissingBin,
      bool* pbMissingIsolated,
      const TreeNode<bHessian, GetArrayScores(cCompilerScores)>** const ppMissingValueTreeNode) {

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
         "monotoneDirection=%" MonotoneDirectionPrintf ", "
         "ppMissingValueTreeNode=%p",
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
         monotoneDirection,
         static_cast<const void*>(ppMissingValueTreeNode));
   EBM_ASSERT(nullptr != ppMissingValueTreeNode);

   const auto* const* ppBinCur = pTreeNode->BEFORE_GetBinFirst();
   const auto* const* ppBinLast = pTreeNode->BEFORE_GetBinLast();

   if(ppBinCur == ppBinLast) {
      // There is just one bin and therefore no splits
      pTreeNode->AFTER_RejectSplit();
      if(pMissingBin == *ppBinCur) {
         *pbMissingIsolated = true;
      }
      return 1;
   }

   const auto* const aBins =
         pBoosterShell->GetBoostingMainBins()
               ->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();

   // in the future we could traverse in both directions
   ptrdiff_t incDirectionBytes = static_cast<ptrdiff_t>(+sizeof(void*));

   BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pBoosterCore->GetCountScores());

   auto* const pLeftChild = GetLeftNode(pTreeNodeScratchSpace);
   pLeftChild->Init();

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

   UIntMain cSamplesDec = binParent.GetCountSamples();
   const auto* const pMissingValueTreeNode = *ppMissingValueTreeNode;
   if(pMissingValueTreeNode == pTreeNode) {
      // missing value is stored at the start of aBins
      binInc.Copy(cScores, *aBins, aBins->GetGradientPairs(), aIncGradHess);
      cSamplesDec -= aBins->GetCountSamples();
   } else {
      binInc.Zero(cScores, aIncGradHess);
   }

   pLeftChild->BEFORE_SetBinFirst(ppBinCur);

   EBM_ASSERT(!IsOverflowTreeNodeSize(bHessian, cScores)); // we're accessing allocated memory
   const size_t cBytesPerTreeNode = GetTreeNodeSize(bHessian, cScores);
   auto* const pRightChild = GetRightNode(pTreeNodeScratchSpace, cBytesPerTreeNode);
   pRightChild->Init();
   pRightChild->BEFORE_SetBinLast(ppBinLast);

   MonotoneDirection monotoneAdjusted = monotoneDirection;

   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);
   EBM_ASSERT(!IsOverflowSplitPositionSize(bHessian, cScores)); // we're accessing allocated memory
   const size_t cBytesPerSplitPosition = GetSplitPositionSize(bHessian, cScores);

   auto* pBestSplitsStart = pBoosterShell->GetSplitPositionsTemp<bHessian, GetArrayScores(cCompilerScores)>();
   auto* pBestSplitsCur = pBestSplitsStart;

   const bool bUseLogitBoost = bHessian && !(TermBoostFlags_DisableNewtonGain & flags);
   const bool bUpdateWithHessian = bHessian && !(TermBoostFlags_DisableNewtonUpdate & flags);

   EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= k_gainMin);
   FloatCalc bestGain = k_gainMin; // it must at least be this, and maybe it needs to be more
   EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= hessianMin);

   while(true) {
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

      if(pMissingValueTreeNode != pTreeNode || incDirectionBytes < ptrdiff_t{0}) {
         break;
      }

      // reverse direction
      incDirectionBytes = -incDirectionBytes;
      monotoneAdjusted = -monotoneAdjusted;
      ppBinCur = pTreeNode->BEFORE_GetBinLast();
      ppBinLast = pTreeNode->BEFORE_GetBinFirst();

      binInc.Copy(cScores, *aBins, aBins->GetGradientPairs(), aIncGradHess);
      cSamplesDec = binParent.GetCountSamples() - aBins->GetCountSamples();
   }

   if(UNLIKELY(pBestSplitsStart == pBestSplitsCur)) {
      // no valid splits found
      EBM_ASSERT(k_gainMin == bestGain);

      pTreeNode->AFTER_RejectSplit();
      return 1;
   }
   EBM_ASSERT(std::isnan(bestGain) || k_gainMin <= bestGain);

   if(UNLIKELY(/* NaN */ !LIKELY(bestGain <= std::numeric_limits<FloatCalc>::max()))) {
      // this tests for NaN and +inf

      // we need this test since the priority queue in the function that calls us cannot accept a NaN value
      // since we would break weak ordering with non-ordered NaN comparisons, thus create undefined behavior

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

   EBM_ASSERT(std::numeric_limits<FloatCalc>::infinity() != bestGain);
   EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= k_gainMin);
   if(UNLIKELY(/* NaN */ !LIKELY(k_gainMin <= bestGain))) {
      // do not allow splits on gains that are too small
      // also filter out slightly negative numbers that can arrise from floating point noise

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

   if(pMissingValueTreeNode == pTreeNode) {
      *ppMissingValueTreeNode = pIncChild;
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

   pTreeNode->AFTER_SetSplitGain(bestGain, pTreeNodeScratchSpace);

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
   FloatCalc m_categoricalSmoothing;

 public:
   INLINE_ALWAYS CompareBin(const bool bHessianRuntime, FloatCalc categoricalSmoothing) {
      m_bHessianRuntime = bHessianRuntime;
      m_categoricalSmoothing = categoricalSmoothing;
   }

   INLINE_ALWAYS bool operator()(
         const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>*& lhs,
         const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>*& rhs) const noexcept {
      // NEVER check for exact equality (as a precondition is ok), since then we'd violate the weak ordering rule
      // https://medium.com/@shiansu/strict-weak-ordering-and-the-c-stl-f7dcfa4d4e07

      const bool bUpdateWithHessian = bHessian && m_bHessianRuntime;

      const FloatCalc hess1 =
            static_cast<FloatCalc>(bUpdateWithHessian ? lhs->GetGradientPairs()[0].GetHess() : lhs->GetWeight());
      const FloatCalc val1 =
            static_cast<FloatCalc>(lhs->GetGradientPairs()[0].m_sumGradients) / (hess1 + m_categoricalSmoothing);

      const FloatCalc hess2 =
            static_cast<FloatCalc>(bUpdateWithHessian ? rhs->GetGradientPairs()[0].GetHess() : rhs->GetWeight());
      const FloatCalc val2 =
            static_cast<FloatCalc>(rhs->GetGradientPairs()[0].m_sumGradients) / (hess2 + m_categoricalSmoothing);

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
         bool bMissing,
         bool bUnseen,
         bool bNominal,
         const TermBoostFlags flags,
         const size_t cBins,
         const size_t iDimension,
         const size_t cSamplesLeafMin,
         const FloatCalc hessianMin,
         const FloatCalc regAlpha,
         const FloatCalc regLambda,
         const FloatCalc deltaStepMax,
         const FloatCalc categoricalSmoothing,
         const size_t categoricalThresholdMax,
         const FloatCalc categoricalInclusionPercent,
         const size_t cSplitsMax,
         const MonotoneDirection monotoneDirection,
         size_t cSamplesTotal,
         FloatMain weightTotal,
         double* const pTotalGain) {
      EBM_ASSERT(2 <= cBins); // filter these out at the start where we can handle this case easily
      EBM_ASSERT(1 <= cSplitsMax); // filter these out at the start where we can handle this case easily
      EBM_ASSERT(nullptr != pTotalGain);
      EBM_ASSERT(!bNominal || MONOTONE_NONE == monotoneDirection);

      // TODO: use all of these!
      UNUSED(bUnseen);
      UNUSED(categoricalSmoothing);
      UNUSED(categoricalThresholdMax);
      UNUSED(categoricalInclusionPercent);

      BoosterCore* const pBoosterCore = pBoosterShell->GetBoosterCore();
      const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pBoosterCore->GetCountScores());

      EBM_ASSERT(1 == cScores || MONOTONE_NONE == monotoneDirection);

      const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);
      auto* const pRootTreeNode = pBoosterShell->GetTreeNodesTemp<bHessian, GetArrayScores(cCompilerScores)>();
      pRootTreeNode->Init();

      // we can only sort if there's a single sortable index, so 1 score value
      bNominal = 1 == cCompilerScores && bNominal && (0 == (TermBoostFlags_DisableCategorical & flags));

      size_t cNormalBins = cBins;
      if(bMissing) {
         --cNormalBins;
      }
      if(bUnseen) {
         --cNormalBins;
      }
      if(cNormalBins <= 1) {
         // we could have just missing and unknown bins an no normal bins, or a normal bin plus missing and/or unseen.
         bMissing = false;
         bUnseen = false;
      }

      auto* const aBins =
            pBoosterShell->GetBoostingMainBins()
                  ->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();
      auto* pBinsEnd = IndexBin(aBins, cBytesPerBin * cBins);

      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>** const apBins =
            reinterpret_cast<const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>**>(
                  pBinsEnd);

      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>** ppBin = apBins;
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* pBin = aBins;

      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* pMissingBin = nullptr;
      bool bMissingIsolated = false;

      const TreeNode<bHessian, GetArrayScores(cCompilerScores)>* pMissingValueTreeNode = nullptr;

      const auto* aSumBins = aBins;
      if(bMissing) {
         if(bNominal) {
            if(TermBoostFlags_MissingCategory & flags) {
               // Nothing to do. Treat missing like any other category.
            } else {
               pMissingValueTreeNode = pRootTreeNode;
               // Skip the missing bin in the pointer to pointer mapping since it will not be part of the continuous
               // region.
               pBin = IndexBin(pBin, cBytesPerBin);
            }
         } else {
            if(TermBoostFlags_MissingLow & flags) {
               pMissingBin = pBin;
            } else if(TermBoostFlags_MissingHigh & flags) {
               pMissingBin = pBin;
               pBin = IndexBin(pBin, cBytesPerBin);
            } else if(TermBoostFlags_MissingSeparate & flags) {
               cSamplesTotal -= static_cast<size_t>(aSumBins->GetCountSamples());
               weightTotal -= aSumBins->GetWeight();
               aSumBins = IndexBin(aSumBins, cBytesPerBin);

               pBin = IndexBin(pBin, cBytesPerBin);
            } else {
               pMissingValueTreeNode = pRootTreeNode;
               // Skip the missing bin in the pointer to pointer mapping since it will not be part of the continuous
               // region.
               pBin = IndexBin(pBin, cBytesPerBin);
            }
         }
      }

      SumAllBins<bHessian, cCompilerScores>(
            pBoosterShell, aSumBins, pBinsEnd, cSamplesTotal, weightTotal, pRootTreeNode->GetBin());

      do {
         *ppBin = pBin;
         pBin = IndexBin(pBin, cBytesPerBin);
         ++ppBin;
      } while(pBinsEnd != pBin);

      if(bMissing && !bNominal && (TermBoostFlags_MissingHigh & flags)) {
         *ppBin = aBins;
         ++ppBin;
      }

      if(bNominal) {
         std::sort(apBins,
               ppBin,
               CompareBin<bHessian, cCompilerScores>(
                     !(TermBoostFlags_DisableNewtonUpdate & flags), categoricalSmoothing));
      }

      pRootTreeNode->BEFORE_SetBinFirst(apBins);
      pRootTreeNode->BEFORE_SetBinLast(ppBin - 1);
      ASSERT_BIN_OK(cBytesPerBin, *(ppBin - 1), pBoosterShell->GetDebugMainBinsEnd());

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
            monotoneDirection,
            pMissingBin,
            &bMissingIsolated,
            &pMissingValueTreeNode);
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
         EBM_ASSERT(std::numeric_limits<FloatMain>::min() <= pRootTreeNode->AFTER_GetSplitGain());

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
               EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= totalGainUpdate);
               totalGain += totalGainUpdate;

               auto* const pChildren = pTreeNode->AFTER_GetChildren();
               auto* const pLeftChild = GetLeftNode(pChildren);
               auto* const pRightChild = GetRightNode(pChildren, cBytesPerTreeNode);

               pTreeNode->AFTER_SplitNode();

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
                     monotoneDirection,
                     pMissingBin,
                     &bMissingIsolated,
                     &pMissingValueTreeNode);
               // if FindBestSplitGain returned -1 to indicate an
               // overflow ignore it here. We successfully made a root node split, so we might as well continue
               // with the successful tree that we have which can make progress in boosting down the residuals
               if(0 == retFind) {
                  pTreeNodeScratchSpace = IndexTreeNode(pTreeNodeScratchSpace, cBytesPerTreeNode << 1);
                  // our priority queue comparison function cannot handle NaN gains so we filter out before
                  EBM_ASSERT(!std::isnan(pLeftChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(!std::isinf(pLeftChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= pLeftChild->AFTER_GetSplitGain());
                  nodeGainRanking.push(pLeftChild->Downgrade());
               }

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
                     monotoneDirection,
                     pMissingBin,
                     &bMissingIsolated,
                     &pMissingValueTreeNode);
               // if FindBestSplitGain returned -1 to indicate an
               // overflow ignore it here. We successfully made a root node split, so we might as well continue
               // with the successful tree that we have which can make progress in boosting down the residuals
               if(0 == retFind) {
                  pTreeNodeScratchSpace = IndexTreeNode(pTreeNodeScratchSpace, cBytesPerTreeNode << 1);
                  // our priority queue comparison function cannot handle NaN gains so we filter out before
                  EBM_ASSERT(!std::isnan(pRightChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(!std::isinf(pRightChild->AFTER_GetSplitGain()));
                  EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= pRightChild->AFTER_GetSplitGain());
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

      size_t cSlices = cSplitsMax - cSplitsRemaining + 1;
      if(nullptr != pMissingValueTreeNode) {
         EBM_ASSERT(nullptr == pMissingBin);
         ++cSlices;
      } else if(!bNominal && bMissing && (TermBoostFlags_MissingSeparate & flags)) {
         ++cSlices; // missing is at index 0 but was unavailable to tree boosting, so add space for it
      } else if(nullptr != pMissingBin && !bMissingIsolated) {
         ++cSlices;
      }
      if(bNominal) {
         cSlices = cBins;
      }
      const ErrorEbm error = Flatten<bHessian>(pBoosterShell,
            bMissing,
            bNominal,
            flags,
            regAlpha,
            regLambda,
            deltaStepMax,
            iDimension,
            reinterpret_cast<const Bin<FloatMain, UIntMain, true, true, bHessian>* const*>(apBins),
            nullptr != pMissingValueTreeNode ? pMissingValueTreeNode->Downgrade() : nullptr,
            cSlices,
            cBins);

      EBM_ASSERT(
            error != Error_None || !bMissing || 2 <= pBoosterShell->GetInnerTermUpdate()->GetCountSlices(iDimension));
      EBM_ASSERT(
            error != Error_None || !bMissing || *pBoosterShell->GetInnerTermUpdate()->GetSplitPointer(iDimension) == 1);

      return error;
   }
};

extern ErrorEbm PartitionOneDimensionalBoosting(RandomDeterministic* const pRng,
      BoosterShell* const pBoosterShell,
      const bool bMissing,
      const bool bUnseen,
      const bool bNominal,
      const TermBoostFlags flags,
      const size_t cBins,
      const size_t iDimension,
      const size_t cSamplesLeafMin,
      const FloatCalc hessianMin,
      const FloatCalc regAlpha,
      const FloatCalc regLambda,
      const FloatCalc deltaStepMax,
      const FloatCalc categoricalSmoothing,
      const size_t categoricalThresholdMax,
      const FloatCalc categoricalInclusionPercent,
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
               bMissing,
               bUnseen,
               bNominal,
               flags,
               cBins,
               iDimension,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               categoricalSmoothing,
               categoricalThresholdMax,
               categoricalInclusionPercent,
               cSplitsMax,
               monotoneDirection,
               cSamplesTotal,
               weightTotal,
               pTotalGain);
      } else if(size_t{3} == cRuntimeScores) {
         // 3 classes
         error = PartitionOneDimensionalBoostingInternal<true, 3>::Func(pRng,
               pBoosterShell,
               bMissing,
               bUnseen,
               bNominal,
               flags,
               cBins,
               iDimension,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               categoricalSmoothing,
               categoricalThresholdMax,
               categoricalInclusionPercent,
               cSplitsMax,
               monotoneDirection,
               cSamplesTotal,
               weightTotal,
               pTotalGain);
      } else {
         // muticlass
         error = PartitionOneDimensionalBoostingInternal<true, k_dynamicScores>::Func(pRng,
               pBoosterShell,
               bMissing,
               bUnseen,
               bNominal,
               flags,
               cBins,
               iDimension,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               categoricalSmoothing,
               categoricalThresholdMax,
               categoricalInclusionPercent,
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
               bMissing,
               bUnseen,
               bNominal,
               flags,
               cBins,
               iDimension,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               categoricalSmoothing,
               categoricalThresholdMax,
               categoricalInclusionPercent,
               cSplitsMax,
               monotoneDirection,
               cSamplesTotal,
               weightTotal,
               pTotalGain);
      } else {
         // Odd: gradient multiclass. Allow it, but do not optimize for it
         error = PartitionOneDimensionalBoostingInternal<false, k_dynamicScores>::Func(pRng,
               pBoosterShell,
               bMissing,
               bUnseen,
               bNominal,
               flags,
               cBins,
               iDimension,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               categoricalSmoothing,
               categoricalThresholdMax,
               categoricalInclusionPercent,
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
