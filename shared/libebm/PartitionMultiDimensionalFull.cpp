// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stddef.h> // size_t, ptrdiff_t

#include "logging.h"
#include "unzoned.h" // LIKELY

#define ZONE_main
#include "zones.h"

#include "GradientPair.hpp"
#include "Bin.hpp"

#include "ebm_internal.hpp"
#include "ebm_stats.hpp"
#include "InteractionCore.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<bool bHessian, size_t cCompilerScores> class PartitionMultiDimensionalFullInternal final {
 public:
   PartitionMultiDimensionalFullInternal() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(InteractionCore* const pInteractionCore,
         const size_t cTensorBins,
         const CalcInteractionFlags flags,
         const FloatCalc regAlpha,
         const FloatCalc regLambda,
         const FloatCalc deltaStepMax,
         BinBase* const aAuxiliaryBinsBase,
         BinBase* const aBinsBase) {
      auto* const aAuxiliaryBins =
            aAuxiliaryBinsBase
                  ->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();
      auto* const aBins =
            aBinsBase->Specialize<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)>();

      const size_t cScores = GET_COUNT_SCORES(cCompilerScores, pInteractionCore->GetCountScores());
      const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);

      Bin<FloatMain, UIntMain, true, true, bHessian, GetArrayScores(cCompilerScores)> totalBin;
      static constexpr bool bUseStackMemory = k_dynamicScores != cCompilerScores;
      auto* const aTotalGradPair = bUseStackMemory ? totalBin.GetGradientPairs() : aAuxiliaryBins->GetGradientPairs();

      const bool bUseLogitBoost = bHessian && !(CalcInteractionFlags_DisableNewton & flags);

      totalBin.SetCountSamples(0);
      totalBin.SetWeight(0);
      for(size_t iScore = 0; iScore < cScores; ++iScore) {
         aTotalGradPair[iScore].Zero();
      }

      FloatCalc gain = 0;

      auto* pBin = aBins;
      const auto* const pBinsEnd = IndexBin(aBins, cBytesPerBin * cTensorBins);
      do {
         totalBin.SetCountSamples(totalBin.GetCountSamples() + pBin->GetCountSamples());
         totalBin.SetWeight(totalBin.GetWeight() + pBin->GetWeight());
         FloatCalc hess = static_cast<FloatCalc>(pBin->GetWeight());
         auto* const aGradPair = pBin->GetGradientPairs();
         for(size_t iScore = 0; iScore < cScores; ++iScore) {
            aTotalGradPair[iScore] += aGradPair[iScore];
            const FloatCalc grad = static_cast<FloatCalc>(aGradPair[iScore].m_sumGradients);
            if(bUseLogitBoost) {
               hess = static_cast<FloatCalc>(aGradPair[iScore].GetHess());
            }
            gain += CalcPartialGain<true>(grad, hess, regAlpha, regLambda, deltaStepMax);
         }
         pBin = IndexBin(pBin, cBytesPerBin);
      } while(pBinsEnd != pBin);

      FloatCalc hessTotal = static_cast<FloatCalc>(totalBin.GetWeight());
      for(size_t iScore = 0; iScore < cScores; ++iScore) {
         const FloatCalc grad = static_cast<FloatCalc>(aTotalGradPair[iScore].m_sumGradients);
         if(bUseLogitBoost) {
            hessTotal = static_cast<FloatCalc>(aTotalGradPair[iScore].GetHess());
         }
         gain -= CalcPartialGain<true>(grad, hessTotal, regAlpha, regLambda, deltaStepMax);
      }

      return static_cast<double>(gain);
   }
};

template<bool bHessian, size_t cPossibleScores> class PartitionMultiDimensionalFullTarget final {
 public:
   PartitionMultiDimensionalFullTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(InteractionCore* const pInteractionCore,
         const size_t cTensorBins,
         const CalcInteractionFlags flags,
         const FloatCalc regAlpha,
         const FloatCalc regLambda,
         const FloatCalc deltaStepMax,
         BinBase* aAuxiliaryBinsBase,
         BinBase* const aBinsBase) {
      if(cPossibleScores == pInteractionCore->GetCountScores()) {
         return PartitionMultiDimensionalFullInternal<bHessian, cPossibleScores>::Func(
               pInteractionCore, cTensorBins, flags, regAlpha, regLambda, deltaStepMax, aAuxiliaryBinsBase, aBinsBase);
      } else {
         return PartitionMultiDimensionalFullTarget<bHessian, cPossibleScores + 1>::Func(
               pInteractionCore, cTensorBins, flags, regAlpha, regLambda, deltaStepMax, aAuxiliaryBinsBase, aBinsBase);
      }
   }
};

template<bool bHessian> class PartitionMultiDimensionalFullTarget<bHessian, k_cCompilerScoresMax + 1> final {
 public:
   PartitionMultiDimensionalFullTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static double Func(InteractionCore* const pInteractionCore,
         const size_t cTensorBins,
         const CalcInteractionFlags flags,
         const FloatCalc regAlpha,
         const FloatCalc regLambda,
         const FloatCalc deltaStepMax,
         BinBase* aAuxiliaryBinsBase,
         BinBase* const aBinsBase) {
      return PartitionMultiDimensionalFullInternal<bHessian, k_dynamicScores>::Func(
            pInteractionCore, cTensorBins, flags, regAlpha, regLambda, deltaStepMax, aAuxiliaryBinsBase, aBinsBase);
   }
};

extern double PartitionMultiDimensionalFull(InteractionCore* const pInteractionCore,
      const size_t cTensorBins,
      const CalcInteractionFlags flags,
      const FloatCalc regAlpha,
      const FloatCalc regLambda,
      const FloatCalc deltaStepMax,
      BinBase* aAuxiliaryBinsBase,
      BinBase* const aBinsBase) {
   const size_t cRuntimeScores = pInteractionCore->GetCountScores();

   EBM_ASSERT(1 <= cRuntimeScores);
   if(pInteractionCore->IsHessian()) {
      if(size_t{1} != cRuntimeScores) {
         // muticlass
         return PartitionMultiDimensionalFullTarget<true, k_cCompilerScoresStart>::Func(
               pInteractionCore, cTensorBins, flags, regAlpha, regLambda, deltaStepMax, aAuxiliaryBinsBase, aBinsBase);
      } else {
         return PartitionMultiDimensionalFullInternal<true, k_oneScore>::Func(
               pInteractionCore, cTensorBins, flags, regAlpha, regLambda, deltaStepMax, aAuxiliaryBinsBase, aBinsBase);
      }
   } else {
      if(size_t{1} != cRuntimeScores) {
         // Odd: gradient multiclass. Allow it, but do not optimize for it
         return PartitionMultiDimensionalFullInternal<false, k_dynamicScores>::Func(
               pInteractionCore, cTensorBins, flags, regAlpha, regLambda, deltaStepMax, aAuxiliaryBinsBase, aBinsBase);
      } else {
         return PartitionMultiDimensionalFullInternal<false, k_oneScore>::Func(
               pInteractionCore, cTensorBins, flags, regAlpha, regLambda, deltaStepMax, aAuxiliaryBinsBase, aBinsBase);
      }
   }
}

} // namespace DEFINED_ZONE_NAME
