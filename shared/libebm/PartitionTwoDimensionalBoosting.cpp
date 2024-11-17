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

#include "RandomDeterministic.hpp"
#include "ebm_stats.hpp"
#include "Tensor.hpp"
#include "TensorTotalsSum.hpp"
#include "TreeNode.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern ErrorEbm PurifyInternal(const double tolerance,
      const size_t cScores,
      const size_t cTensorBins,
      const size_t cSurfaceBins,
      RandomDeterministic* const pRng,
      size_t* const aRandomize,
      const size_t* const aDimensionLengths,
      const double* const aWeights,
      double* const pScores,
      double* const pImpurities,
      double* const pIntercept);

template<bool bHessian, size_t cCompilerScores, size_t cCompilerDimensions>
INLINE_RELEASE_TEMPLATED static ErrorEbm MakeTensor(const size_t cRuntimeScores,
      const size_t cRuntimeRealDimensions,
      const TermBoostFlags flags,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const aBins,
      const FloatCalc regAlpha,
      const FloatCalc regLambda,
      const FloatCalc deltaStepMax,
      double* const aTensorWeights,
      double* const aTensorGrad,
      double* const aTensorHess,
      const size_t cPossibleSplits,
      unsigned char** const aaSplits,
      TreeNodeMulti<bHessian, GetArrayScores(cCompilerScores)>* const pRootTreeNode,
      const size_t* const aiOriginalIndex,
      TensorSumDimension* const aDimensions,
      Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const aAuxiliaryBins,
      Tensor* const pInnerTermUpdate
#ifndef NDEBUG
      ,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const aDebugCopyBins,
      const BinBase* const pBinsEndDebug
#endif // NDEBUG
) {
   ErrorEbm error;

   const bool bUseLogitBoost = bHessian && !(TermBoostFlags_DisableNewtonGain & flags);
   const bool bUpdateWithHessian = bHessian && !(TermBoostFlags_DisableNewtonUpdate & flags);

   const size_t cRealDimensions = GET_COUNT_DIMENSIONS(cCompilerDimensions, cRuntimeRealDimensions);
   EBM_ASSERT(1 <= cRealDimensions); // for interactions, we just return 0 for interactions with zero features

   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, cRuntimeScores);
#ifndef NDEBUG
   const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);
#endif // NDEBUG
   const size_t cBytesTreeNodeMulti = GetTreeNodeMultiSize(bHessian, cScores);

   const size_t cBytesBest = cBytesTreeNodeMulti * (size_t{1} + (cRealDimensions << 1));
   auto* const pTreeNodeEnd = IndexTreeNodeMulti(pRootTreeNode, cBytesBest);

   GradientPair<FloatMain, bHessian>* pTensorGradientPair = nullptr;

   auto* const pTempScratch = aAuxiliaryBins;

   Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)> binTemp;

   // if we know how many scores there are, use the memory on the stack where the compiler can optimize access
   static constexpr bool bUseStackMemory = k_dynamicScores != cCompilerScores;
   auto* const aGradientPairsTemp = bUseStackMemory ? binTemp.GetGradientPairs() : pTempScratch->GetGradientPairs();

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
      const size_t cSplitFirst = static_cast<size_t>(pInnerTermUpdate->GetSplitPointer(aiOriginalIndex[iDim])[0]);
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

      FloatCalc tensorHess = 0;
      if(nullptr != pTensorWeights || nullptr != pTensorHess || nullptr != pTensorGrad) {
         ASSERT_BIN_OK(cBytesPerBin, pTempScratch, pBinsEndDebug);
         TensorTotalsSum<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
               cRealDimensions,
               aDimensions,
               aBins,
               binTemp,
               aGradientPairsTemp
#ifndef NDEBUG
               ,
               aDebugCopyBins,
               pBinsEndDebug
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

         FloatCalc prediction = -CalcNegUpdate<false>(
               static_cast<FloatCalc>(pGradientPair->m_sumGradients), nodeHess, regAlpha, regLambda, deltaStepMax);

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
         aDimensions[iDim].m_iHigh = static_cast<size_t>(pInnerTermUpdate->GetSplitPointer(aiOriginalIndex[iDim])[0]);
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

template<bool bHessian, size_t cCompilerScores> class PartitionTwoDimensionalBoostingInternal final {
 public:
   PartitionTwoDimensionalBoostingInternal() = delete; // this is a static class.  Do not construct

   WARNING_PUSH
   WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(const size_t cRuntimeScores,
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
         unsigned char** const aaSplits
#ifndef NDEBUG
         ,
         const BinBase* const aDebugCopyBinsBase,
         const BinBase* const pBinsEndDebug
#endif // NDEBUG
   ) {
      static constexpr size_t cCompilerDimensions = k_dynamicDimensions;

      ErrorEbm error;

      auto* const aBins =
            aBinsBase->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();

      const size_t cScores = GET_COUNT_SCORES(cCompilerScores, cRuntimeScores);
      const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);
      const size_t cBytesTreeNodeMulti = GetTreeNodeMultiSize(bHessian, cScores);

      auto* const pRootTreeNode =
            reinterpret_cast<TreeNodeMulti<bHessian, GetArrayScores(cCompilerScores)>*>(pRootTreeNodeBase);

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

      TreeNodeMulti<bHessian, GetArrayScores(cCompilerScores)>* pParentTreeNode = nullptr;
      auto* pTreeNode = pDeepTreeNode;
      auto* pHigh = IndexTreeNodeMulti(pTreeNode, cBytesTreeNodeMulti);
      EBM_ASSERT(1 <= cRealDimensions);

      size_t iDimensionLoop = 0;
      size_t iDimInit = 0;
      size_t aiOriginalIndex[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
#ifndef NDEBUG
      size_t aiDEBUGDim[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
#endif // NDEBUG
      EBM_ASSERT(1 <= cRealDimensions);
      do {
         EBM_ASSERT(iDimensionLoop < cDimensions);
         const size_t cBins = acBins[iDimensionLoop];
         EBM_ASSERT(size_t{1} <= cBins); // we don't boost on empty training sets
         if(size_t{1} < cBins) {
            aiOriginalIndex[iDimInit] = iDimensionLoop;
            aDimensions[iDimInit].m_cBins = cBins;

#ifndef NDEBUG
            aiDEBUGDim[iDimInit] = cRealDimensions - 1 - iDimInit;
#endif // NDEBUG

            pTreeNode->SplitNode();
            pTreeNode->SetSplitIndex(0);
            pTreeNode->SetDimensionIndex(cRealDimensions - size_t{1} - iDimInit);
            pTreeNode->SetParent(pParentTreeNode);
            pTreeNode->SetChildren(pHigh);

            pParentTreeNode = pTreeNode;
            pTreeNode = IndexTreeNodeMulti(pHigh, cBytesTreeNodeMulti);
            auto* const pNextHigh = IndexTreeNodeMulti(pTreeNode, cBytesTreeNodeMulti);
            ++iDimInit;

            // High child Node
            pHigh->SetSplitGain(0.0);
            pHigh->SetSplitIndex(0);
            pHigh->SetDimensionIndex(cRealDimensions - size_t{1} - iDimInit);
            pHigh->SetParent(pParentTreeNode);
            // set both high and low nodes to point to the same children. It isn't valid
            // if the node isn't split but this avoids having to continually swap them
            pHigh->SetChildren(pNextHigh);

            pHigh = pNextHigh;
         }
         ++iDimensionLoop;
      } while(cRealDimensions != iDimInit);

      // Low child node
      pTreeNode->SetSplitGain(0.0);
      pTreeNode->SetParent(pParentTreeNode);
      pTreeNode->SetChildren(pHigh); // we need to set it to something because we access this pointer below

      // TODO: we should calculate the default partial gain before splitting anything. Once we have that
      // number we can calculate the minimum gain we need to reach k_gainMin after the cuts are made
      // which will allow us to avoid some work when the eventual gain will be less than our minimum
      FloatCalc bestGain = k_gainMin; // do not allow bad cuts that lead to negative gain

      EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= hessianMin);

      const TensorSumDimension* const pDimensionEnd = &aDimensions[cRealDimensions];

      while(true) {
         pTreeNode = pLastSplitTreeNode;
         while(true) {
            EBM_ASSERT(pTreeNode->IsSplit());
            EBM_ASSERT(GetHighNode(pTreeNode->GetChildren())->GetParent() == pTreeNode);
            EBM_ASSERT(GetLowNode(pTreeNode->GetChildren(), cBytesTreeNodeMulti)->GetParent() == pTreeNode);
            EBM_ASSERT(GetHighNode(pTreeNode->GetChildren())->GetChildren() ==
                  GetLowNode(pTreeNode->GetChildren(), cBytesTreeNodeMulti)->GetChildren());
            EBM_ASSERT(!GetHighNode(pTreeNode->GetChildren())->IsSplit());
            EBM_ASSERT(pTreeNode != pLastSplitTreeNode ||
                  !GetLowNode(pTreeNode->GetChildren(), cBytesTreeNodeMulti)->IsSplit());
            EBM_ASSERT(pTreeNode == pLastSplitTreeNode ||
                  GetLowNode(pTreeNode->GetChildren(), cBytesTreeNodeMulti)->IsSplit());
            EBM_ASSERT(0 == pTreeNode->GetSplitIndex());
            EBM_ASSERT(pTreeNode->GetDimensionIndex() ==
                  aiDEBUGDim[CountItems(pTreeNode, pDeepTreeNode, cBytesTreeNodeMulti << 1)]);

            if(pDeepTreeNode == pTreeNode) {
               EBM_ASSERT(nullptr == pTreeNode->GetParent());
               break;
            }

            EBM_ASSERT(GetLowNode(pTreeNode->GetParent()->GetChildren(), cBytesTreeNodeMulti) == pTreeNode);
            EBM_ASSERT(NegativeIndexByte(pTreeNode, cBytesTreeNodeMulti << 1) == pTreeNode->GetParent());
            EBM_ASSERT(NegativeIndexByte(pTreeNode, cBytesTreeNodeMulti) ==
                  GetHighNode(pTreeNode->GetParent()->GetChildren()));

            const size_t iDimCopy = pTreeNode->GetDimensionIndex();

            // this points to the high node now
            pTreeNode = NegativeIndexByte(pTreeNode, cBytesTreeNodeMulti);
            pTreeNode->SetDimensionIndex(iDimCopy);
            // this points to the parent now
            pTreeNode = NegativeIndexByte(pTreeNode, cBytesTreeNodeMulti);
         }

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
                           pBinsEndDebug
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
                           pBinsEndDebug
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

               if(0 != (TermBoostFlags_PurifyGain & flags)) {
                  // TODO: we're doing extra computation above when we calculate the unpurified gain so eliminate that

                  gain = 0.0;
                  error = MakeTensor<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
                        cRealDimensions,
                        flags,
                        aBins,
                        regAlpha,
                        regLambda,
                        deltaStepMax,
                        aTensorWeights,
                        aTensorGrad,
                        aTensorHess,
                        cPossibleSplits,
                        aaSplits,
                        pDeepTreeNode,
                        aiOriginalIndex,
                        aDimensions,
                        aAuxiliaryBins,
                        pInnerTermUpdate
#ifndef NDEBUG
                        ,
                        aDebugCopyBins,
                        pBinsEndDebug
#endif // NDEBUG
                  );
                  if(Error_None != error) {
                     return error;
                  }

                  // TODO: the code below is duplicated in GenerateTempUpdate, so we can probably
                  // put both of them into a function that we call from both places

                  Tensor* const pTensor = pInnerTermUpdate;

                  double* pGradient = aTensorGrad;
                  double* pHessian = aTensorHess;

                  size_t cTensorBinsPurify = 1;
                  size_t iDimension = 0;
                  do {
                     const size_t cBins = pTensor->GetCountSlices(iDimension);
                     cTensorBinsPurify *= cBins;
                     ++iDimension;
                  } while(cDimensions != iDimension);

                  size_t acPurifyBins[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax :
                                                                                   cCompilerDimensions];
                  size_t* pcPurifyBins = acPurifyBins;
                  size_t cSurfaceBinsTotal = 0;
                  iDimension = 0;
                  do {
                     const size_t cBins = pTensor->GetCountSlices(iDimension);
                     if(size_t{1} < cBins) {
                        *pcPurifyBins = cBins;
                        EBM_ASSERT(0 == cTensorBinsPurify % cBins);
                        const size_t cExcludeSurfaceBins = cTensorBinsPurify / cBins;
                        cSurfaceBinsTotal += cExcludeSurfaceBins;
                        ++pcPurifyBins;
                     }
                     ++iDimension;
                  } while(cDimensions != iDimension);

                  constexpr double tolerance =
                        0.0; // TODO: for now purify to the max, but test tolerances and profile them

                  // TODO: in the future try randomizing the purification order.  It probably doesn't make much
                  // difference
                  //       though if we're purifying to the 0.0 tolerance, and it might make things slower, although we
                  //       could see a speed increase if it allows us to use bigger tolerance values.

                  double* pScores = pTensor->GetTensorScoresPointer();
                  const double* const pScoreMulticlassEnd = &pScores[cScores];
                  do {
                     // ignore the return from PurifyInternal since we should check for NaN in the weights
                     // earlier and the checks in PurifyInternal are only for the stand-alone purification API
                     PurifyInternal(tolerance,
                           cScores,
                           cTensorBinsPurify,
                           cSurfaceBinsTotal,
                           nullptr,
                           nullptr,
                           acPurifyBins,
                           aTensorWeights,
                           pScores,
                           nullptr,
                           nullptr);
                     ++pScores;
                  } while(pScoreMulticlassEnd != pScores);

                  // When calculating purified gain, we do not subtract
                  // the parent since the pure partial gain is always zero.
                  double* pScore = pTensor->GetTensorScoresPointer();
                  EBM_ASSERT(!IsMultiplyError(cTensorBinsPurify, cScores)); // we have allocated it
                  double* pScoreEnd = pScore + cTensorBinsPurify * cScores;
                  double* pWeight = aTensorWeights;
                  do {
                     double hess = *pWeight;
                     for(size_t iScore = 0; iScore < cScores; ++iScore) {
                        double grad = *pGradient;
                        if(nullptr != pHessian) {
                           hess = *pHessian;
                           ++pHessian;
                        }
                        double update = *pScore;
                        gain += CalcPartialGainFromUpdate(grad, hess, -update, regAlpha, regLambda);
                        ++pGradient;
                        ++pScore;
                     }
                     ++pWeight;
                  } while(pScoreEnd != pScore);
               }

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

#ifndef NDEBUG
         const bool bDEBUGContinue = std::prev_permutation(aiDEBUGDim, &aiDEBUGDim[cRealDimensions]);
#endif // NDEBUG

         EBM_ASSERT(1 <= cRealDimensions);
         if(pDeepTreeNode == pLastSplitTreeNode) {
            EBM_ASSERT(!bDEBUGContinue);
            goto done;
         }
         auto* pNode2 = pLastSplitTreeNode;
         auto* pNode1 = NegativeIndexByte(pLastSplitTreeNode, cBytesTreeNodeMulti << 1);

         while(pNode1->GetDimensionIndex() <= pNode2->GetDimensionIndex()) {
            if(pNode1 == pDeepTreeNode) {
               EBM_ASSERT(!bDEBUGContinue);
               goto done;
            }
            pNode2 = pNode1;
            pNode1 = NegativeIndexByte(pNode1, cBytesTreeNodeMulti << 1);
         }
         EBM_ASSERT(bDEBUGContinue);

         auto* pNode3 = pLastSplitTreeNode;
         while(pNode3->GetDimensionIndex() >= pNode1->GetDimensionIndex()) {
            pNode3 = NegativeIndexByte(pNode3, cBytesTreeNodeMulti << 1);
         }

         size_t iDimTemp = pNode1->GetDimensionIndex();
         pNode1->SetDimensionIndex(pNode3->GetDimensionIndex());
         pNode3->SetDimensionIndex(iDimTemp);

         auto* pNode4 = pLastSplitTreeNode;
         while(pNode2 < pNode4) {
            iDimTemp = pNode4->GetDimensionIndex();
            pNode4->SetDimensionIndex(pNode2->GetDimensionIndex());
            pNode2->SetDimensionIndex(iDimTemp);
            pNode4 = NegativeIndexByte(pNode4, cBytesTreeNodeMulti << 1);
            pNode2 = IndexByte(pNode2, cBytesTreeNodeMulti << 1);
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

      EBM_ASSERT(std::isnan(bestGain) || k_illegalGainFloat == bestGain || FloatCalc{0} <= bestGain);

      // the bin before the aAuxiliaryBins is the last summation bin of aBinsBase,
      // which contains the totals of all bins
      const auto* const pTotal = NegativeIndexBin(aAuxiliaryBins, cBytesPerBin);

      ASSERT_BIN_OK(cBytesPerBin, pTotal, pBinsEndDebug);

      const auto* const pGradientPairTotal = pTotal->GetGradientPairs();

      const FloatMain weightAll = pTotal->GetWeight();
      EBM_ASSERT(0 < weightAll);

      const bool bUpdateWithHessian = bHessian && !(TermBoostFlags_DisableNewtonUpdate & flags);

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

            if(0 == (TermBoostFlags_PurifyGain & flags)) {
               // for purified, we don't subtract the parent since the parent's partial gain is zero when purified

               // TODO: we should move this computation to the top and then calculate how much gain we need
               // for the split in order to reach the k_gainMin value

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

                  error = MakeTensor<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
                        cRealDimensions,
                        flags,
                        aBins,
                        regAlpha,
                        regLambda,
                        deltaStepMax,
                        aTensorWeights,
                        aTensorGrad,
                        aTensorHess,
                        cPossibleSplits,
                        aaSplits,
                        pRootTreeNode,
                        aiOriginalIndex,
                        aDimensions,
                        aAuxiliaryBins,
                        pInnerTermUpdate
#ifndef NDEBUG
                        ,
                        aDebugCopyBins,
                        pBinsEndDebug
#endif // NDEBUG
                  );
                  if(Error_None != error) {
                     return error;
                  }

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

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(const size_t cRuntimeScores,
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
         unsigned char** const aaSplits
#ifndef NDEBUG
         ,
         const BinBase* const aDebugCopyBinsBase,
         const BinBase* const pBinsEndDebug
#endif // NDEBUG
   ) {
      if(cPossibleScores == cRuntimeScores) {
         return PartitionTwoDimensionalBoostingInternal<bHessian, cPossibleScores>::Func(cRuntimeScores,
               cDimensions,
               cRealDimensions,
               flags,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               aBinsBase,
               aAuxiliaryBinsBase,
               pInnerTermUpdate,
               pRootTreeNodeBase,
               acBins,
               aTensorWeights,
               aTensorGrad,
               aTensorHess,
               pTotalGain,
               cPossibleSplits,
               aaSplits
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase,
               pBinsEndDebug
#endif // NDEBUG
         );
      } else {
         return PartitionTwoDimensionalBoostingTarget<bHessian, cPossibleScores + 1>::Func(cRuntimeScores,
               cDimensions,
               cRealDimensions,
               flags,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               aBinsBase,
               aAuxiliaryBinsBase,
               pInnerTermUpdate,
               pRootTreeNodeBase,
               acBins,
               aTensorWeights,
               aTensorGrad,
               aTensorHess,
               pTotalGain,
               cPossibleSplits,
               aaSplits
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase,
               pBinsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<bool bHessian> class PartitionTwoDimensionalBoostingTarget<bHessian, k_cCompilerScoresMax + 1> final {
 public:
   PartitionTwoDimensionalBoostingTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(const size_t cRuntimeScores,
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
         unsigned char** const aaSplits
#ifndef NDEBUG
         ,
         const BinBase* const aDebugCopyBinsBase,
         const BinBase* const pBinsEndDebug
#endif // NDEBUG
   ) {
      return PartitionTwoDimensionalBoostingInternal<bHessian, k_dynamicScores>::Func(cRuntimeScores,
            cDimensions,
            cRealDimensions,
            flags,
            cSamplesLeafMin,
            hessianMin,
            regAlpha,
            regLambda,
            deltaStepMax,
            aBinsBase,
            aAuxiliaryBinsBase,
            pInnerTermUpdate,
            pRootTreeNodeBase,
            acBins,
            aTensorWeights,
            aTensorGrad,
            aTensorHess,
            pTotalGain,
            cPossibleSplits,
            aaSplits
#ifndef NDEBUG
            ,
            aDebugCopyBinsBase,
            pBinsEndDebug
#endif // NDEBUG
      );
   }
};

extern ErrorEbm PartitionTwoDimensionalBoosting(const bool bHessian,
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
) {
   ErrorEbm error;

   unsigned char* pSplits = static_cast<unsigned char*>(pTemp1);
   unsigned char* aaSplits[k_cDimensionsMax];
   unsigned char** paSplits = aaSplits;

   const size_t* pcBins = acBins;
   const size_t* const acBinsEnd = acBins + cDimensions;
   do {
      const size_t cSplits = *pcBins - 1;
      if(0 != cSplits) {
         *paSplits = pSplits;
         pSplits += cSplits;
         ++paSplits;
      }
      ++pcBins;
   } while(acBinsEnd != pcBins);

   EBM_ASSERT(1 <= cRuntimeScores);
   if(bHessian) {
      if(size_t{1} != cRuntimeScores) {
         // muticlass
         error = PartitionTwoDimensionalBoostingTarget<true, k_cCompilerScoresStart>::Func(cRuntimeScores,
               cDimensions,
               cRealDimensions,
               flags,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               aBinsBase,
               aAuxiliaryBinsBase,
               pInnerTermUpdate,
               pRootTreeNodeBase,
               acBins,
               aTensorWeights,
               aTensorGrad,
               aTensorHess,
               pTotalGain,
               cPossibleSplits,
               aaSplits
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase,
               pBinsEndDebug
#endif // NDEBUG
         );
      } else {
         error = PartitionTwoDimensionalBoostingInternal<true, k_oneScore>::Func(cRuntimeScores,
               cDimensions,
               cRealDimensions,
               flags,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               aBinsBase,
               aAuxiliaryBinsBase,
               pInnerTermUpdate,
               pRootTreeNodeBase,
               acBins,
               aTensorWeights,
               aTensorGrad,
               aTensorHess,
               pTotalGain,
               cPossibleSplits,
               aaSplits
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase,
               pBinsEndDebug
#endif // NDEBUG
         );
      }
   } else {
      if(size_t{1} != cRuntimeScores) {
         // Odd: gradient multiclass. Allow it, but do not optimize for it
         error = PartitionTwoDimensionalBoostingInternal<false, k_dynamicScores>::Func(cRuntimeScores,
               cDimensions,
               cRealDimensions,
               flags,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               aBinsBase,
               aAuxiliaryBinsBase,
               pInnerTermUpdate,
               pRootTreeNodeBase,
               acBins,
               aTensorWeights,
               aTensorGrad,
               aTensorHess,
               pTotalGain,
               cPossibleSplits,
               aaSplits
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase,
               pBinsEndDebug
#endif // NDEBUG
         );
      } else {
         error = PartitionTwoDimensionalBoostingInternal<false, k_oneScore>::Func(cRuntimeScores,
               cDimensions,
               cRealDimensions,
               flags,
               cSamplesLeafMin,
               hessianMin,
               regAlpha,
               regLambda,
               deltaStepMax,
               aBinsBase,
               aAuxiliaryBinsBase,
               pInnerTermUpdate,
               pRootTreeNodeBase,
               acBins,
               aTensorWeights,
               aTensorGrad,
               aTensorHess,
               pTotalGain,
               cPossibleSplits,
               aaSplits
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase,
               pBinsEndDebug
#endif // NDEBUG
         );
      }
   }
   return error;
}

} // namespace DEFINED_ZONE_NAME
