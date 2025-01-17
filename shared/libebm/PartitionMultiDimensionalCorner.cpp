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
      const FloatCalc regAlpha,
      const FloatCalc regLambda,
      const FloatCalc deltaStepMax,
      const size_t* const aiOriginalIndex,
      TensorSumDimension* const aDimensions,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const pTotalBin,
      const Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>* const pCornerBin,
      Tensor* const pInnerTermUpdate) {
   ErrorEbm error;

   const bool bUpdateWithHessian = bHessian && !(TermBoostFlags_DisableNewtonUpdate & flags);

   const size_t cRealDimensions = GET_COUNT_DIMENSIONS(cCompilerDimensions, cRuntimeRealDimensions);
   EBM_ASSERT(1 <= cRealDimensions); // for interactions, we just return 0 for interactions with zero features

   const size_t cScores = GET_COUNT_SCORES(cCompilerScores, cRuntimeScores);

   error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * (size_t{1} << cRealDimensions));
   if(Error_None != error) {
      // already logged
      return error;
   }
   FloatScore* const aUpdateScores = pInnerTermUpdate->GetTensorScoresPointer();
   FloatScore* pCornerScores = aUpdateScores;

   EBM_ASSERT(1 <= cRealDimensions);
   size_t iDimension = 0;
   size_t cTensorBytes = sizeof(*aUpdateScores);
   do {
      const size_t iOriginalDimension = aiOriginalIndex[iDimension];

      error = pInnerTermUpdate->SetCountSlices(iOriginalDimension, 2);
      if(Error_None != error) {
         // already logged
         return error;
      }

      UIntSplit* pSplits = pInnerTermUpdate->GetSplitPointer(iOriginalDimension);
      size_t iSplit = aDimensions[iDimension].m_iLow;
      if(0 == iSplit) {
         iSplit = aDimensions[iDimension].m_iHigh;
      } else {
         pCornerScores = IndexByte(pCornerScores, cTensorBytes);
      }
      *pSplits = iSplit;

      ++iDimension;
      cTensorBytes <<= 1;
   } while(cRealDimensions != iDimension);

   FloatScore* const pUpdateScoreEnd = aUpdateScores + cScores * (size_t{1} << cRealDimensions);
   FloatScore* pUpdateScore = aUpdateScores;
   do {
      if(pCornerScores == pUpdateScore) {
         FloatCalc cornerHess = static_cast<FloatCalc>(pCornerBin->GetWeight());
         auto* const aCornerGradPair = pCornerBin->GetGradientPairs();
         size_t iScore = 0;
         do {
            if(bUpdateWithHessian) {
               cornerHess = static_cast<FloatCalc>(aCornerGradPair[iScore].GetHess());
            }
            const FloatCalc prediction =
                  -CalcNegUpdate<false>(static_cast<FloatCalc>(aCornerGradPair[iScore].m_sumGradients),
                        cornerHess,
                        regAlpha,
                        regLambda,
                        deltaStepMax);
            *pUpdateScore = prediction;
            ++pUpdateScore;
            ++iScore;
         } while(cScores != iScore);
      } else {
         FloatCalc cornerHess = static_cast<FloatCalc>(pCornerBin->GetWeight());
         FloatCalc totalHess = static_cast<FloatCalc>(pTotalBin->GetWeight());
         auto* const aCornerGradPair = pCornerBin->GetGradientPairs();
         auto* const aTotalGradPair = pTotalBin->GetGradientPairs();
         size_t iScore = 0;
         do {
            if(bUpdateWithHessian) {
               cornerHess = static_cast<FloatCalc>(aCornerGradPair[iScore].GetHess());
               totalHess = static_cast<FloatCalc>(aTotalGradPair[iScore].GetHess());
            }
            EBM_ASSERT(cornerHess <= totalHess);
            const FloatCalc hess = totalHess - cornerHess;
            const FloatCalc grad = static_cast<FloatCalc>(
                  aTotalGradPair[iScore].m_sumGradients - aCornerGradPair[iScore].m_sumGradients);
            const FloatCalc prediction = -CalcNegUpdate<false>(grad, hess, regAlpha, regLambda, deltaStepMax);
            *pUpdateScore = prediction;
            ++pUpdateScore;
            ++iScore;
         } while(cScores != iScore);
      }
   } while(pUpdateScoreEnd != pUpdateScore);

   return Error_None;
}

template<bool bHessian, size_t cCompilerScores> class PartitionMultiDimensionalCornerInternal final {
 public:
   PartitionMultiDimensionalCornerInternal() = delete; // this is a static class.  Do not construct

   WARNING_PUSH
   WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(const size_t cRuntimeScores,
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
         const size_t* const acBins,
         double* const pTotalGain
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

      const bool bUseLogitBoost = bHessian && !(TermBoostFlags_DisableNewtonGain & flags);

      auto* const pTempBin =
            aAuxiliaryBinsBase
                  ->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();

#ifndef NDEBUG
      const auto* const aDebugCopyBins =
            aDebugCopyBinsBase
                  ->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();
#endif // NDEBUG

      EBM_ASSERT(1 <= cRealDimensions);

      // the bin before the aAuxiliaryBins is the last summation bin of aBinsBase,
      // which contains the totals of all bins
      const auto* const pTotal = NegativeIndexBin(pTempBin, cBytesPerBin);

      ASSERT_BIN_OK(cBytesPerBin, pTotal, pBinsEndDebug);

      const auto* const aGradientPairTotal = pTotal->GetGradientPairs();

      const FloatMain weightAll = pTotal->GetWeight();
      EBM_ASSERT(0 < weightAll);

      FloatCalc parentGain = 0;
      FloatCalc hess = static_cast<FloatCalc>(weightAll);
      for(size_t iScore = 0; iScore < cScores; ++iScore) {
         if(bUseLogitBoost) {
            hess = aGradientPairTotal[iScore].GetHess();
         }

         const FloatCalc gain1 =
               CalcPartialGain<true>(static_cast<FloatCalc>(aGradientPairTotal[iScore].m_sumGradients),
                     hess,
                     regAlpha,
                     regLambda,
                     deltaStepMax);
         EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);
         parentGain += gain1;
      }

      ptrdiff_t direction = static_cast<ptrdiff_t>(MakeLowMask<size_t>(static_cast<int>(cRealDimensions)));
      FloatCalc bestGain = -std::numeric_limits<double>::infinity();
      EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= hessianMin);

      TensorSumDimension
            aBestDimensions[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];

      size_t aiOriginalIndex[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
      {
         size_t iDimensionLoop = 0;
         size_t iDimInit = 0;
         EBM_ASSERT(1 <= cRealDimensions);
         do {
            const size_t cBins = acBins[iDimensionLoop];
            EBM_ASSERT(size_t{1} <= cBins); // we don't boost on empty training sets
            if(size_t{1} < cBins) {
               aiOriginalIndex[iDimInit] = iDimensionLoop;
               ++iDimInit;
            }
            ++iDimensionLoop;
         } while(cRealDimensions != iDimInit);
      }

      const size_t cTotalSamples = static_cast<size_t>(pTotal->GetCountSamples());

      TensorSumDimension
            aDimensions[k_dynamicDimensions == cCompilerDimensions ? k_cDimensionsMax : cCompilerDimensions];
      TensorSumDimension* pDimensionEnd = &aDimensions[cRealDimensions];

      do {
         ptrdiff_t directionDestroy = direction;
         size_t iDimensionLoop = 0;
         size_t iDimInit = 0;
         EBM_ASSERT(1 <= cRealDimensions);
         do {
            const size_t cBins = acBins[iDimensionLoop];
            EBM_ASSERT(size_t{1} <= cBins); // we don't boost on empty training sets
            if(size_t{1} < cBins) {
               aDimensions[iDimInit].m_cBins = cBins;
               if(0 != (directionDestroy & 1)) {
                  aDimensions[iDimInit].m_iLow = 0;
                  aDimensions[iDimInit].m_iHigh = 1;
               } else {
                  aDimensions[iDimInit].m_iLow = cBins - 1;
                  aDimensions[iDimInit].m_iHigh = cBins;
               }
               ++iDimInit;
               directionDestroy >>= 1;
            }
            ++iDimensionLoop;
         } while(cRealDimensions != iDimInit);
         EBM_ASSERT(0 == directionDestroy);

         while(true) {
            FloatCalc gain = 0;

            TensorTotalsSum<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
                  cRealDimensions,
                  aDimensions,
                  aBins,
                  *pTempBin,
                  pTempBin->GetGradientPairs()
#ifndef NDEBUG
                        ,
                  aDebugCopyBins,
                  pBinsEndDebug
#endif // NDEBUG
            );

            if(cSamplesLeafMin <= static_cast<size_t>(pTempBin->GetCountSamples())) {
               const size_t cSamplesOther = cTotalSamples - static_cast<size_t>(pTempBin->GetCountSamples());
               if(cSamplesLeafMin <= cSamplesOther) {
                  EBM_ASSERT(1 <= cScores);
                  size_t iScore = 0;
                  FloatCalc hessian = static_cast<FloatCalc>(pTempBin->GetWeight());
                  FloatCalc hessianOther = static_cast<FloatCalc>(weightAll);
                  auto* const aGradientPairsLocal = pTempBin->GetGradientPairs();
                  do {
                     if(bUseLogitBoost) {
                        const FloatMain hessOrig = aGradientPairsLocal[iScore].GetHess();
                        hessian = static_cast<FloatCalc>(hessOrig);
                        hessianOther = static_cast<FloatCalc>(aGradientPairTotal[iScore].GetHess() - hessOrig);
                     }
                     if(hessian < hessianMin) {
                        goto next;
                     }
                     if(hessianOther < hessianMin) {
                        goto next;
                     }
                     const FloatMain gradOrig = aGradientPairsLocal[iScore].m_sumGradients;
                     const FloatCalc grad = static_cast<FloatCalc>(gradOrig);
                     const FloatCalc gradOther =
                           static_cast<FloatCalc>(aGradientPairTotal[iScore].m_sumGradients - gradOrig);

                     const FloatCalc gain1 = CalcPartialGain<false>(grad, hessian, regAlpha, regLambda, deltaStepMax);
                     EBM_ASSERT(std::isnan(gain1) || 0 <= gain1);

                     const FloatCalc gain2 =
                           CalcPartialGain<false>(gradOther, hessianOther, regAlpha, regLambda, deltaStepMax);
                     EBM_ASSERT(std::isnan(gain2) || 0 <= gain2);

                     const FloatCalc gainChange = gain1 + gain2;
                     gain += gainChange;

                     ++iScore;
                  } while(cScores != iScore);
                  EBM_ASSERT(std::isnan(gain) || 0 <= gain); // sumation of positive numbers should be positive

                  if(UNLIKELY(/* NaN */ !LIKELY(gain <= bestGain))) {
                     // propagate NaNs
                     bestGain = gain;
                     memcpy(aBestDimensions, aDimensions, sizeof(*aDimensions) * cRealDimensions);
                  } else {
                     EBM_ASSERT(!std::isnan(gain));
                  }

               next:;
               }
            }

            ptrdiff_t directionDestroy2 = direction;
            TensorSumDimension* pDimension = aDimensions;
            while(true) {
               if(directionDestroy2 & 1) {
                  ++pDimension->m_iHigh;
                  if(LIKELY(pDimension->m_cBins != pDimension->m_iHigh)) {
                     break;
                  }
                  pDimension->m_iHigh = 1;
               } else {
                  --pDimension->m_iLow;
                  if(LIKELY(0 != pDimension->m_iLow)) {
                     break;
                  }
                  pDimension->m_iLow = pDimension->m_cBins - 1;
               }
               directionDestroy2 >>= 1;
               ++pDimension;
               if(UNLIKELY(pDimensionEnd == pDimension)) {
                  goto done_one;
               }
            }
         }
      done_one:;
         --direction;
      } while(0 <= direction);

      bestGain -= parentGain;

      *pTotalGain = 0;
      EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= k_gainMin);
      if(LIKELY(/* NaN */ !UNLIKELY(bestGain < k_gainMin))) {
         EBM_ASSERT(std::isnan(bestGain) || std::numeric_limits<FloatCalc>::min() <= bestGain);

         // signal that we've hit an overflow.  Use +inf here since our caller likes that and will flip to -inf
         *pTotalGain = std::numeric_limits<double>::infinity();
         if(LIKELY(/* NaN */ bestGain <= std::numeric_limits<FloatCalc>::max())) {
            EBM_ASSERT(!std::isnan(bestGain));
            EBM_ASSERT(std::numeric_limits<FloatCalc>::min() <= bestGain);
            EBM_ASSERT(std::numeric_limits<FloatCalc>::infinity() != bestGain);

            *pTotalGain = static_cast<double>(bestGain);

            TensorTotalsSum<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
                  cRealDimensions,
                  aBestDimensions,
                  aBins,
                  *pTempBin,
                  pTempBin->GetGradientPairs()
#ifndef NDEBUG
                        ,
                  aDebugCopyBins,
                  pBinsEndDebug
#endif // NDEBUG
            );

            error = MakeTensor<bHessian, cCompilerScores, cCompilerDimensions>(cScores,
                  cRealDimensions,
                  flags,
                  regAlpha,
                  regLambda,
                  deltaStepMax,
                  aiOriginalIndex,
                  aBestDimensions,
                  pTotal,
                  pTempBin,
                  pInnerTermUpdate);
            if(Error_None != error) {
               return error;
            }

            return Error_None;
         } else {
            EBM_ASSERT(std::isnan(bestGain) || std::numeric_limits<FloatCalc>::infinity() == bestGain);
         }
      }

      // there were no good splits found
      pInnerTermUpdate->Reset();

      // we don't need to call pInnerTermUpdate->EnsureTensorScoreCapacity,
      // since our value capacity would be 1, which is pre-allocated

      const bool bUpdateWithHessian = bHessian && !(TermBoostFlags_DisableNewtonUpdate & flags);

      FloatScore* const aUpdateScores = pInnerTermUpdate->GetTensorScoresPointer();
      FloatCalc weight2 = static_cast<FloatCalc>(weightAll);
      for(size_t iScore = 0; iScore < cScores; ++iScore) {
         if(bUpdateWithHessian) {
            weight2 = static_cast<FloatCalc>(aGradientPairTotal[iScore].GetHess());
         }
         const FloatCalc update =
               -CalcNegUpdate<true>(static_cast<FloatCalc>(aGradientPairTotal[iScore].m_sumGradients),
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

template<bool bHessian, size_t cPossibleScores> class PartitionMultiDimensionalCornerTarget final {
 public:
   PartitionMultiDimensionalCornerTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(const size_t cRuntimeScores,
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
         const size_t* const acBins,
         double* const pTotalGain
#ifndef NDEBUG
         ,
         const BinBase* const aDebugCopyBinsBase,
         const BinBase* const pBinsEndDebug
#endif // NDEBUG
   ) {
      if(cPossibleScores == cRuntimeScores) {
         return PartitionMultiDimensionalCornerInternal<bHessian, cPossibleScores>::Func(cRuntimeScores,
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
               acBins,
               pTotalGain
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase,
               pBinsEndDebug
#endif // NDEBUG
         );
      } else {
         return PartitionMultiDimensionalCornerTarget<bHessian, cPossibleScores + 1>::Func(cRuntimeScores,
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
               acBins,
               pTotalGain
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase,
               pBinsEndDebug
#endif // NDEBUG
         );
      }
   }
};

template<bool bHessian> class PartitionMultiDimensionalCornerTarget<bHessian, k_cCompilerScoresMax + 1> final {
 public:
   PartitionMultiDimensionalCornerTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(const size_t cRuntimeScores,
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
         const size_t* const acBins,
         double* const pTotalGain
#ifndef NDEBUG
         ,
         const BinBase* const aDebugCopyBinsBase,
         const BinBase* const pBinsEndDebug
#endif // NDEBUG
   ) {
      return PartitionMultiDimensionalCornerInternal<bHessian, k_dynamicScores>::Func(cRuntimeScores,
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
            acBins,
            pTotalGain
#ifndef NDEBUG
            ,
            aDebugCopyBinsBase,
            pBinsEndDebug
#endif // NDEBUG
      );
   }
};

extern ErrorEbm PartitionMultiDimensionalCorner(const bool bHessian,
      const size_t cRuntimeScores,
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
      const size_t* const acBins,
      double* const pTotalGain
#ifndef NDEBUG
      ,
      const BinBase* const aDebugCopyBinsBase,
      const BinBase* const pBinsEndDebug
#endif // NDEBUG
) {
   ErrorEbm error;

   EBM_ASSERT(1 <= cRuntimeScores);
   if(bHessian) {
      if(size_t{1} != cRuntimeScores) {
         // muticlass
         error = PartitionMultiDimensionalCornerTarget<true, k_cCompilerScoresStart>::Func(cRuntimeScores,
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
               acBins,
               pTotalGain
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase,
               pBinsEndDebug
#endif // NDEBUG
         );
      } else {
         error = PartitionMultiDimensionalCornerInternal<true, k_oneScore>::Func(cRuntimeScores,
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
               acBins,
               pTotalGain
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
         error = PartitionMultiDimensionalCornerInternal<false, k_dynamicScores>::Func(cRuntimeScores,
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
               acBins,
               pTotalGain
#ifndef NDEBUG
               ,
               aDebugCopyBinsBase,
               pBinsEndDebug
#endif // NDEBUG
         );
      } else {
         error = PartitionMultiDimensionalCornerInternal<false, k_oneScore>::Func(cRuntimeScores,
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
               acBins,
               pTotalGain
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
