// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <string.h> // memcpy

#include "ebm_native.h" // EBM_API_BODY
#include "logging.h" // EBM_ASSERT
#include "zones.h"

#include "ebm_internal.hpp"

#include "RandomDeterministic.hpp"
#include "RandomNondeterministic.hpp"
#include "ebm_stats.hpp"
#include "Feature.hpp"
#include "Term.hpp"
#include "InnerBag.hpp"
#include "Tensor.hpp"
#include "GradientPair.hpp"
#include "Bin.hpp"
#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern ErrorEbm BinSumsBoosting(BinSumsBoostingBridge * const pParams);

extern void TensorTotalsBuild(
   const ptrdiff_t cClasses,
   const size_t cRealDimensions,
   const size_t * const acBins,
   BinBase * aAuxiliaryBinsBase,
   BinBase * const aBinsBase
#ifndef NDEBUG
   , BinBase * const aDebugCopyBinsBase
   , const BinBase * const pBinsEndDebug
#endif // NDEBUG
);

extern ErrorEbm PartitionOneDimensionalBoosting(
   RandomDeterministic * const pRng,
   BoosterShell * const pBoosterShell,
   const size_t cBins,
   const size_t iDimension,
   const size_t cSamplesLeafMin,
   const size_t cSplitsMax,
   const size_t cSamplesTotal,
   const FloatBig weightTotal,
   double * const pTotalGain
);

extern ErrorEbm PartitionTwoDimensionalBoosting(
   BoosterShell * const pBoosterShell,
   const Term * const pTerm,
   const size_t * const acBins,
   const size_t cSamplesLeafMin,
   BinBase * aAuxiliaryBinsBase,
   double * const pTotalGain
#ifndef NDEBUG
   , const BinBase * const aDebugCopyBinsBase
#endif // NDEBUG
);

extern ErrorEbm PartitionRandomBoosting(
   RandomDeterministic * const pRng,
   BoosterShell * const pBoosterShell,
   const Term * const pTerm,
   const BoostFlags flags,
   const IntEbm * const aLeavesMax,
   double * const pTotalGain
);

static ErrorEbm BoostZeroDimensional(
   BoosterShell * const pBoosterShell, 
   const InnerBag * const pInnerBag,
   const BoostFlags flags
) {
   LOG_0(Trace_Verbose, "Entered BoostZeroDimensional");

   ErrorEbm error;

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t cClasses = pBoosterCore->GetCountClasses();
   const bool bClassification = IsClassification(cClasses);

   const size_t cScores = GetCountScores(cClasses);

   EBM_ASSERT(!IsOverflowBinSize<FloatFast>(bClassification, cScores)); // we check in CreateBooster
   const size_t cBytesPerFastBin = GetBinSize<FloatFast>(bClassification, cScores);

   BinBase * const pFastBin = pBoosterShell->GetBoostingFastBinsTemp();
   EBM_ASSERT(nullptr != pFastBin);

   pFastBin->ZeroMem(cBytesPerFastBin);

   BinSumsBoostingBridge params;
   params.m_cClasses = cClasses;
   params.m_cPack = k_cItemsPerBitPackNone;
   params.m_cSamples = pBoosterCore->GetTrainingSet()->GetCountSamples();
   params.m_aGradientsAndHessians = pBoosterCore->GetTrainingSet()->GetGradientsAndHessiansPointer();
   params.m_aWeights = pInnerBag->GetWeights();
   params.m_pCountOccurrences = pInnerBag->GetCountOccurrences();
   params.m_aFastBins = pBoosterShell->GetBoostingFastBinsTemp();
#ifndef NDEBUG
   params.m_pDebugFastBinsEnd = IndexBin(pFastBin, cBytesPerFastBin);
   params.m_totalWeightDebug = pInnerBag->GetWeightTotal();
#endif // NDEBUG
   error = BinSumsBoosting(&params);
   if(Error_None != error) {
      return error;
   }

   BinBase * const pBigBin = pBoosterShell->GetBoostingBigBins();
   EBM_ASSERT(nullptr != pBigBin);

#ifndef NDEBUG
   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we check in CreateBooster
   const size_t cBytesPerBigBin = GetBinSize<FloatBig>(bClassification, cScores);
   pBoosterShell->SetDebugBigBinsEnd(IndexBin(pBigBin, cBytesPerBigBin));
#endif // NDEBUG

   // TODO: put this into it's own function that converts our fast floats to big floats
   static_assert(sizeof(FloatBig) == sizeof(FloatFast), "float mismatch");
   EBM_ASSERT(cBytesPerFastBin == cBytesPerBigBin); // until we switch fast to float datatypes
   memcpy(pBigBin, pFastBin, cBytesPerFastBin);


   // TODO: we can exit here back to python to allow caller modification to our histograms


   Tensor * const pInnerTermUpdate = pBoosterShell->GetInnerTermUpdate();
   FloatFast * aUpdateScores = pInnerTermUpdate->GetTensorScoresPointer();
   if(bClassification) {
      const auto * const pBin = pBigBin->Specialize<FloatBig, true>();
      const auto * const aGradientPairs = pBin->GetGradientPairs();
      if(0 != (BoostFlags_GradientSums & flags)) {
         for(size_t iScore = 0; iScore < cScores; ++iScore) {
            const FloatBig updateScore = EbmStats::ComputeSinglePartitionUpdateGradientSum(aGradientPairs[iScore].m_sumGradients);
            aUpdateScores[iScore] = SafeConvertFloat<FloatFast>(updateScore);
         }
      } else {
         for(size_t iScore = 0; iScore < cScores; ++iScore) {
            FloatBig updateScore = EbmStats::ComputeSinglePartitionUpdate(
               aGradientPairs[iScore].m_sumGradients,
               aGradientPairs[iScore].GetHess()
            );
            aUpdateScores[iScore] = SafeConvertFloat<FloatFast>(updateScore);
         }
      }
   } else {
      EBM_ASSERT(IsRegression(cClasses));
      const auto * const pBin = pBigBin->Specialize<FloatBig, false>();
      const auto * const aGradientPairs = pBin->GetGradientPairs();
      if(0 != (BoostFlags_GradientSums & flags)) {
         const FloatBig updateScore = EbmStats::ComputeSinglePartitionUpdateGradientSum(aGradientPairs[0].m_sumGradients);
         aUpdateScores[0] = SafeConvertFloat<FloatFast>(updateScore);
      } else {
         const FloatBig updateScore = EbmStats::ComputeSinglePartitionUpdate(
            aGradientPairs[0].m_sumGradients,
            pBin->GetWeight()
         );
         aUpdateScores[0] = SafeConvertFloat<FloatFast>(updateScore);
      }
   }

   LOG_0(Trace_Verbose, "Exited BoostZeroDimensional");

   return Error_None;
}

static ErrorEbm BoostSingleDimensional(
   RandomDeterministic * const pRng,
   BoosterShell * const pBoosterShell,
   const size_t iTerm,
   const size_t cBins,
   const InnerBag * const pInnerBag,
   const size_t iDimension,
   const size_t cSamplesLeafMin,
   const IntEbm countLeavesMax,
   double * const pTotalGain
) {
   ErrorEbm error;

   LOG_0(Trace_Verbose, "Entered BoostSingleDimensional");

   EBM_ASSERT(IntEbm { 2 } <= countLeavesMax); // otherwise we would have called BoostZeroDimensional
   size_t cSplitsMax = static_cast<size_t>(countLeavesMax) - size_t { 1 };
   if(IsConvertError<size_t>(countLeavesMax)) {
      // we can never exceed a size_t number of leaves, so let's just set it to the maximum if we were going to overflow because it will generate 
      // the same results as if we used the true number
      cSplitsMax = std::numeric_limits<size_t>::max();
   }

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();

   EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
   EBM_ASSERT(1 == pBoosterCore->GetTerms()[iTerm]->GetCountRealDimensions());
   EBM_ASSERT(cBins == pBoosterCore->GetTerms()[iTerm]->GetCountTensorBins());
   EBM_ASSERT(0 == pBoosterCore->GetTerms()[iTerm]->GetCountAuxillaryBins());

   const ptrdiff_t cClasses = pBoosterCore->GetCountClasses();
   const bool bClassification = IsClassification(cClasses);
   const size_t cScores = GetCountScores(cClasses);

   EBM_ASSERT(!IsOverflowBinSize<FloatFast>(bClassification, cScores)); // we check in CreateBooster
   const size_t cBytesPerFastBin = GetBinSize<FloatFast>(bClassification, cScores);
   EBM_ASSERT(!IsMultiplyError(cBytesPerFastBin, cBins));

   BinBase * const aFastBins = pBoosterShell->GetBoostingFastBinsTemp();
   EBM_ASSERT(nullptr != aFastBins);

   aFastBins->ZeroMem(cBytesPerFastBin, cBins);

   BinSumsBoostingBridge params;
   params.m_cClasses = cClasses;
   params.m_cPack = pBoosterCore->GetTerms()[iTerm]->GetTermBitPack();
   params.m_cSamples = pBoosterCore->GetTrainingSet()->GetCountSamples();
   params.m_aGradientsAndHessians = pBoosterCore->GetTrainingSet()->GetGradientsAndHessiansPointer();
   params.m_aWeights = pInnerBag->GetWeights();
   params.m_pCountOccurrences = pInnerBag->GetCountOccurrences();
   params.m_aPacked = pBoosterCore->GetTrainingSet()->GetInputDataPointer(iTerm);
   params.m_aFastBins = pBoosterShell->GetBoostingFastBinsTemp();
#ifndef NDEBUG
   params.m_pDebugFastBinsEnd = IndexBin(aFastBins, cBytesPerFastBin * cBins);
   params.m_totalWeightDebug = pInnerBag->GetWeightTotal();
#endif // NDEBUG
   error = BinSumsBoosting(&params);
   if(Error_None != error) {
      return error;
   }

   BinBase * const aBigBins = pBoosterShell->GetBoostingBigBins();
   EBM_ASSERT(nullptr != aBigBins);

#ifndef NDEBUG
   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we check in CreateBooster 
   const size_t cBytesPerBigBin = GetBinSize<FloatBig>(bClassification, cScores);
   EBM_ASSERT(!IsMultiplyError(cBytesPerBigBin, cBins));
   pBoosterShell->SetDebugBigBinsEnd(IndexBin(aBigBins, cBytesPerBigBin * cBins));
#endif // NDEBUG

   // TODO: put this into it's own function that converts our fast floats to big floats
   static_assert(sizeof(FloatBig) == sizeof(FloatFast), "float mismatch");
   EBM_ASSERT(cBytesPerFastBin == cBytesPerBigBin); // until we switch fast to float datatypes
   memcpy(aBigBins, aFastBins, cBytesPerFastBin * cBins);


   // TODO: we can exit here back to python to allow caller modification to our histograms


   EBM_ASSERT(1 <= pBoosterCore->GetTrainingSet()->GetCountSamples());

   error = PartitionOneDimensionalBoosting(
      pRng,
      pBoosterShell,
      cBins,
      iDimension,
      cSamplesLeafMin,
      cSplitsMax,
      pBoosterCore->GetTrainingSet()->GetCountSamples(),
      pInnerBag->GetWeightTotal(),
      pTotalGain
   );

   LOG_0(Trace_Verbose, "Exited BoostSingleDimensional");
   return error;
}

// TODO: for higher dimensional spaces, we need to add/subtract individual cells alot and the hessian isn't required (yet) in order to make decisions about
//   where to split.  For dimensions higher than 2, we might want to copy the tensor to a new tensor AFTER binning that keeps only the gradients and then 
//    go back to our original tensor after splits to determine the hessian
static ErrorEbm BoostMultiDimensional(
   BoosterShell * const pBoosterShell,
   const size_t iTerm,
   const InnerBag * const pInnerBag,
   const size_t cSamplesLeafMin,
   double * const pTotalGain
) {
   LOG_0(Trace_Verbose, "Entered BoostMultiDimensional");

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];

   EBM_ASSERT(2 <= pTerm->GetCountDimensions());
   EBM_ASSERT(2 <= pTerm->GetCountRealDimensions());

   ErrorEbm error;

   const size_t cTensorBins = pTerm->GetCountTensorBins();
   EBM_ASSERT(1 <= cTensorBins);

   size_t acBins[k_cDimensionsMax];
   size_t * pcBins = acBins;

   const FeatureBoosting * const * ppFeature = pTerm->GetFeatures();
   const FeatureBoosting * const * const ppFeaturesEnd = &ppFeature[pTerm->GetCountDimensions()];
   do {
      const FeatureBoosting * pFeature = *ppFeature;
      const size_t cBins = pFeature->GetCountBins();
      EBM_ASSERT(size_t { 1 } <= cBins); // we don't boost on empty training sets
      if(size_t { 1 } < cBins) {
         *pcBins = cBins;
         ++pcBins;
      }
      ++ppFeature;
   } while(ppFeaturesEnd != ppFeature);

   const ptrdiff_t cClasses = pBoosterCore->GetCountClasses();
   const bool bClassification = IsClassification(cClasses);
   const size_t cScores = GetCountScores(cClasses);

   EBM_ASSERT(!IsOverflowBinSize<FloatFast>(bClassification, cScores)); // we check in CreateBooster 
   const size_t cBytesPerFastBin = GetBinSize<FloatFast>(bClassification, cScores);
   EBM_ASSERT(!IsMultiplyError(cBytesPerFastBin, cTensorBins));

   BinBase * const aFastBins = pBoosterShell->GetBoostingFastBinsTemp();
   EBM_ASSERT(nullptr != aFastBins);
   
   aFastBins->ZeroMem(cBytesPerFastBin, cTensorBins);

   BinSumsBoostingBridge params;
   params.m_cClasses = cClasses;
   params.m_cPack = pBoosterCore->GetTerms()[iTerm]->GetTermBitPack();
   params.m_cSamples = pBoosterCore->GetTrainingSet()->GetCountSamples();
   params.m_aGradientsAndHessians = pBoosterCore->GetTrainingSet()->GetGradientsAndHessiansPointer();
   params.m_aWeights = pInnerBag->GetWeights();
   params.m_pCountOccurrences = pInnerBag->GetCountOccurrences();
   params.m_aPacked = pBoosterCore->GetTrainingSet()->GetInputDataPointer(iTerm);
   params.m_aFastBins = pBoosterShell->GetBoostingFastBinsTemp();
#ifndef NDEBUG
   params.m_pDebugFastBinsEnd = IndexBin(aFastBins, cBytesPerFastBin * cTensorBins);
   params.m_totalWeightDebug = pInnerBag->GetWeightTotal();
#endif // NDEBUG
   error = BinSumsBoosting(&params);
   if(Error_None != error) {
      return error;
   }

   const size_t cAuxillaryBins = pTerm->GetCountAuxillaryBins();

   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we check in CreateBooster 
   const size_t cBytesPerBigBin = GetBinSize<FloatBig>(bClassification, cScores);


   // we don't need to free this!  It's tracked and reused by pBoosterShell
   BinBase * const aBigBins = pBoosterShell->GetBoostingBigBins();
   EBM_ASSERT(nullptr != aBigBins);

#ifndef NDEBUG
   EBM_ASSERT(!IsAddError(cTensorBins, cAuxillaryBins));
   EBM_ASSERT(!IsMultiplyError(cBytesPerBigBin, cTensorBins + cAuxillaryBins));
   const auto * const pDebugBigBinsEnd = IndexBin(aBigBins, cBytesPerBigBin * (cTensorBins + cAuxillaryBins));
   pBoosterShell->SetDebugBigBinsEnd(pDebugBigBinsEnd);
#endif // NDEBUG

   // TODO: put this into it's own function that converts our fast floats to big floats
   static_assert(sizeof(FloatBig) == sizeof(FloatFast), "float mismatch");
   memcpy(aBigBins, aFastBins, cBytesPerFastBin * cTensorBins);



   // TODO: we can exit here back to python to allow caller modification to our histograms



   // we also need to zero the auxillary bins
   aBigBins->ZeroMem(cBytesPerBigBin, cAuxillaryBins, cTensorBins);
 
#ifndef NDEBUG
   // make a copy of the original bins for debugging purposes

   BinBase * aDebugCopyBins = nullptr;
   if(!IsMultiplyError(cBytesPerBigBin, cTensorBins)) {
      ANALYSIS_ASSERT(0 != cBytesPerBigBin);
      aDebugCopyBins = static_cast<BinBase *>(malloc(cBytesPerBigBin * cTensorBins));
      if(nullptr != aDebugCopyBins) {
         // if we can't allocate, don't fail.. just stop checking
         memcpy(aDebugCopyBins, aBigBins, cBytesPerBigBin * cTensorBins);
      }
   }
#endif // NDEBUG

   BinBase * aAuxiliaryBins = IndexBin(aBigBins, cBytesPerBigBin * cTensorBins);

   TensorTotalsBuild(
      cClasses,
      pTerm->GetCountRealDimensions(),
      acBins,
      aAuxiliaryBins,
      aBigBins
#ifndef NDEBUG
      , aDebugCopyBins
      , pDebugBigBinsEnd
#endif // NDEBUG
   );

   //permutation0
   //gain_permute0
   //  divs0
   //  gain0
   //    divs00
   //    gain00
   //      divs000
   //      gain000
   //      divs001
   //      gain001
   //    divs01
   //    gain01
   //      divs010
   //      gain010
   //      divs011
   //      gain011
   //  divs1
   //  gain1
   //    divs10
   //    gain10
   //      divs100
   //      gain100
   //      divs101
   //      gain101
   //    divs11
   //    gain11
   //      divs110
   //      gain110
   //      divs111
   //      gain111
   //---------------------------
   //permutation1
   //gain_permute1
   //  divs0
   //  gain0
   //    divs00
   //    gain00
   //      divs000
   //      gain000
   //      divs001
   //      gain001
   //    divs01
   //    gain01
   //      divs010
   //      gain010
   //      divs011
   //      gain011
   //  divs1
   //  gain1
   //    divs10
   //    gain10
   //      divs100
   //      gain100
   //      divs101
   //      gain101
   //    divs11
   //    gain11
   //      divs110
   //      gain110
   //      divs111
   //      gain111       *


   //size_t aiDimensionPermutation[k_cDimensionsMax];
   //for(unsigned int iDimensionInitialize = 0; iDimensionInitialize < cDimensions; ++iDimensionInitialize) {
   //   aiDimensionPermutation[iDimensionInitialize] = iDimensionInitialize;
   //}
   //size_t aiDimensionPermutationBest[k_cDimensionsMax];

   // DO this is a fixed length that we should make variable!
   //size_t aDOSplits[1000000];
   //size_t aDOSplitsBest[1000000];

   //do {
   //   size_t aiDimensions[k_cDimensionsMax];
   //   memset(aiDimensions, 0, sizeof(aiDimensions[0]) * cDimensions));
   //   while(true) {


   //      EBM_ASSERT(0 == iDimension);
   //      while(true) {
   //         ++aiDimension[iDimension];
   //         if(aiDimension[iDimension] != 
   //               pTerms->GetFeatures()[aiDimensionPermutation[iDimension]].m_pFeature->m_cBins) {
   //            break;
   //         }
   //         aiDimension[iDimension] = 0;
   //         ++iDimension;
   //         if(iDimension == cDimensions) {
   //            goto move_next_permutation;
   //         }
   //      }
   //   }
   //   move_next_permutation:
   //} while(std::next_permutation(aiDimensionPermutation, &aiDimensionPermutation[cDimensions]));

   if(2 == pTerm->GetCountRealDimensions()) {
      error = PartitionTwoDimensionalBoosting(
         pBoosterShell,
         pTerm,
         acBins,
         cSamplesLeafMin,
         aAuxiliaryBins,
         pTotalGain
#ifndef NDEBUG
         , aDebugCopyBins
#endif // NDEBUG
      );
      if(Error_None != error) {
#ifndef NDEBUG
         free(aDebugCopyBins);
#endif // NDEBUG

         LOG_0(Trace_Verbose, "Exited BoostMultiDimensional with Error code");

         return error;
      }

      EBM_ASSERT(!std::isnan(*pTotalGain));
      EBM_ASSERT(0 <= *pTotalGain);
   } else {
      LOG_0(Trace_Warning, "WARNING BoostMultiDimensional 2 != pTerm->GetCountSignificantFeatures()");

      // TODO: eventually handle this in our caller and this function can specialize in handling just 2 dimensional
      //       then we can replace this branch with an assert
#ifndef NDEBUG
      EBM_ASSERT(false);
      free(aDebugCopyBins);
#endif // NDEBUG
      return Error_UnexpectedInternal;
   }

#ifndef NDEBUG
   free(aDebugCopyBins);
#endif // NDEBUG

   LOG_0(Trace_Verbose, "Exited BoostMultiDimensional");
   return Error_None;
}

static ErrorEbm BoostRandom(
   RandomDeterministic * const pRng,
   BoosterShell * const pBoosterShell,
   const size_t iTerm,
   const InnerBag * const pInnerBag,
   const BoostFlags flags,
   const IntEbm * const aLeavesMax,
   double * const pTotalGain
) {
   // THIS RANDOM SPLIT FUNCTION IS PRIMARILY USED FOR DIFFERENTIAL PRIVACY EBMs

   LOG_0(Trace_Verbose, "Entered BoostRandom");

   ErrorEbm error;

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(iTerm < pBoosterCore->GetCountTerms());
   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];

   const size_t cTotalBins = pTerm->GetCountTensorBins();
   EBM_ASSERT(1 <= cTotalBins);

   const ptrdiff_t cClasses = pBoosterCore->GetCountClasses();
   const bool bClassification = IsClassification(cClasses);
   const size_t cScores = GetCountScores(cClasses);

   EBM_ASSERT(!IsOverflowBinSize<FloatFast>(bClassification, cScores)); // we check in CreateBooster 
   const size_t cBytesPerFastBin = GetBinSize<FloatFast>(bClassification, cScores);

   BinBase * const aFastBins = pBoosterShell->GetBoostingFastBinsTemp();
   EBM_ASSERT(nullptr != aFastBins);

   EBM_ASSERT(!IsMultiplyError(cBytesPerFastBin, cTotalBins));
   aFastBins->ZeroMem(cBytesPerFastBin, cTotalBins);

   BinSumsBoostingBridge params;
   params.m_cClasses = cClasses;
   params.m_cPack = pBoosterCore->GetTerms()[iTerm]->GetTermBitPack();
   params.m_cSamples = pBoosterCore->GetTrainingSet()->GetCountSamples();
   params.m_aGradientsAndHessians = pBoosterCore->GetTrainingSet()->GetGradientsAndHessiansPointer();
   params.m_aWeights = pInnerBag->GetWeights();
   params.m_pCountOccurrences = pInnerBag->GetCountOccurrences();
   params.m_aPacked = pBoosterCore->GetTrainingSet()->GetInputDataPointer(iTerm);
   params.m_aFastBins = pBoosterShell->GetBoostingFastBinsTemp();
#ifndef NDEBUG
   params.m_pDebugFastBinsEnd = IndexBin(aFastBins, cBytesPerFastBin * cTotalBins);
   params.m_totalWeightDebug = pInnerBag->GetWeightTotal();
#endif // NDEBUG
   error = BinSumsBoosting(&params);
   if(Error_None != error) {
      return error;
   }

   BinBase * const aBigBins = pBoosterShell->GetBoostingBigBins();
   EBM_ASSERT(nullptr != aBigBins);

#ifndef NDEBUG
   EBM_ASSERT(!IsOverflowBinSize<FloatBig>(bClassification, cScores)); // we check in CreateBooster 
   const size_t cBytesPerBigBin = GetBinSize<FloatBig>(bClassification, cScores);
   EBM_ASSERT(!IsMultiplyError(cBytesPerBigBin, cTotalBins));
   pBoosterShell->SetDebugBigBinsEnd(IndexBin(aBigBins, cBytesPerBigBin * cTotalBins));
#endif // NDEBUG

   // TODO: put this into it's own function that converts our fast floats to big floats
   static_assert(sizeof(FloatBig) == sizeof(FloatFast), "float mismatch");
   EBM_ASSERT(cBytesPerFastBin == cBytesPerBigBin); // until we switch fast to float datatypes
   memcpy(aBigBins, aFastBins, cBytesPerFastBin * cTotalBins);


   // TODO: we can exit here back to python to allow caller modification to our histograms


   error = PartitionRandomBoosting(
      pRng,
      pBoosterShell,
      pTerm,
      flags,
      aLeavesMax,
      pTotalGain
   );
   if(Error_None != error) {
      LOG_0(Trace_Verbose, "Exited BoostRandom with Error code");
      return error;
   }

   EBM_ASSERT(!std::isnan(*pTotalGain));
   EBM_ASSERT(0 <= *pTotalGain);

   LOG_0(Trace_Verbose, "Exited BoostRandom");
   return Error_None;
}

// we made this a global because if we had put this variable inside the BoosterCore object, then we would need to dereference that before getting 
// the count.  By making this global we can send a log message incase a bad BoosterCore object is sent into us we only decrease the count if the 
// count is non-zero, so at worst if there is a race condition then we'll output this log message more times than desired, but we can live with that
static int g_cLogGenerateTermUpdate = 10;


EBM_API_BODY ErrorEbm EBM_CALLING_CONVENTION GenerateTermUpdate(
   void * rng,
   BoosterHandle boosterHandle,
   IntEbm indexTerm,
   BoostFlags flags,
   double learningRate,
   IntEbm minSamplesLeaf,
   const IntEbm * leavesMax,
   double * avgGainOut
) {
   ErrorEbm error;

   LOG_COUNTED_N(
      &g_cLogGenerateTermUpdate,
      Trace_Info,
      Trace_Verbose,
      "GenerateTermUpdate: "
      "rng=%p, "
      "boosterHandle=%p, "
      "indexTerm=%" IntEbmPrintf ", "
      "flags=0x%" UBoostFlagsPrintf ", "
      "learningRate=%le, "
      "minSamplesLeaf=%" IntEbmPrintf ", "
      "leavesMax=%p, "
      "avgGainOut=%p"
      ,
      rng,
      static_cast<void *>(boosterHandle),
      indexTerm,
      static_cast<UBoostFlags>(flags), // signed to unsigned conversion is defined behavior in C++
      learningRate,
      minSamplesLeaf,
      static_cast<const void *>(leavesMax),
      static_cast<void *>(avgGainOut)
   );

   if(LIKELY(nullptr != avgGainOut)) {
      *avgGainOut = k_illegalGainDouble;
   }

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      // already logged
      return Error_IllegalParamVal;
   }

   // set this to illegal so if we exit with an error we have an invalid index
   pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);

   if(indexTerm < 0) {
      LOG_0(Trace_Error, "ERROR GenerateTermUpdate indexTerm must be positive");
      return Error_IllegalParamVal;
   }

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(nullptr != pBoosterCore);

   if(static_cast<IntEbm>(pBoosterCore->GetCountTerms()) <= indexTerm) {
      LOG_0(Trace_Error, "ERROR GenerateTermUpdate indexTerm above the number of feature groups that we have");
      return Error_IllegalParamVal;
   }
   size_t iTerm = static_cast<size_t>(indexTerm);

   // this is true because 0 < pBoosterCore->m_cTerms since our caller needs to pass in a valid indexTerm to this function
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());
   Term * const pTerm = pBoosterCore->GetTerms()[iTerm];

   LOG_COUNTED_0(
      pTerm->GetPointerCountLogEnterGenerateTermUpdateMessages(),
      Trace_Info,
      Trace_Verbose,
      "Entered GenerateTermUpdate"
   );

   if(0 != (static_cast<UBoostFlags>(flags) & ~(
      static_cast<UBoostFlags>(BoostFlags_DisableNewtonGain) |
      static_cast<UBoostFlags>(BoostFlags_DisableNewtonUpdate) |
      static_cast<UBoostFlags>(BoostFlags_GradientSums) |
      static_cast<UBoostFlags>(BoostFlags_RandomSplits)
      ))) {
      LOG_0(Trace_Error, "ERROR GenerateTermUpdate flags contains unknown flags. Ignoring extras.");
   }

   if(std::isnan(learningRate)) {
      LOG_0(Trace_Warning, "WARNING GenerateTermUpdate learningRate is NaN");
   } else if(std::numeric_limits<double>::infinity() == learningRate) {
      LOG_0(Trace_Warning, "WARNING GenerateTermUpdate learningRate is +infinity");
   } else if(0.0 == learningRate) {
      LOG_0(Trace_Warning, "WARNING GenerateTermUpdate learningRate is zero");
   } else if(learningRate < double { 0 }) {
      LOG_0(Trace_Warning, "WARNING GenerateTermUpdate learningRate is negative");
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
      LOG_0(Trace_Warning, "WARNING GenerateTermUpdate minSamplesLeaf can't be less than 1.  Adjusting to 1.");
   }

   const ptrdiff_t cClasses = pBoosterCore->GetCountClasses();

   if(ptrdiff_t { 0 } == cClasses || ptrdiff_t { 1 } == cClasses) {
      EBM_ASSERT(nullptr == pBoosterShell->GetTermUpdate());
      EBM_ASSERT(nullptr == pBoosterShell->GetInnerTermUpdate());

      // if there are 0 classes, then there must be zero samples, but our caller can still specify 0 != cBins below
      EBM_ASSERT(ptrdiff_t { 0 } != pBoosterCore->GetCountClasses() || 0 == pBoosterCore->GetTrainingSet()->GetCountSamples());
      EBM_ASSERT(ptrdiff_t { 0 } != pBoosterCore->GetCountClasses() || 0 == pBoosterCore->GetValidationSet()->GetCountSamples());

      // if there is only 1 target class for classification, then we can predict the output with 100% accuracy.  
      // The term scores are a tensor with zero length array logits, which means for our representation that we have 
      // zero items in the array total. Since we can predit the output with 100% accuracy, our gain will be 0.
      if(LIKELY(nullptr != avgGainOut)) {
         *avgGainOut = 0.0;
      }
      pBoosterShell->SetTermIndex(iTerm);

      LOG_0(Trace_Warning, "WARNING GenerateTermUpdate ptrdiff_t { 0 } == cClasses || ptrdiff_t { 1 } == cClasses");
      return Error_None;
   }
   EBM_ASSERT(nullptr != pBoosterShell->GetTermUpdate());
   EBM_ASSERT(nullptr != pBoosterShell->GetInnerTermUpdate());

   if(size_t { 0 } == pTerm->GetCountTensorBins()) {
      // there are zero samples and 0 bins in one of the features in the dimensions, so the update tensor has 0 bins

      // if GetCountTensorBins is 0, then we leave pBoosterShell->GetTermUpdate() with invalid data since
      // out Tensor class does not support tensors of zero elements

      if(LIKELY(nullptr != avgGainOut)) {
         *avgGainOut = 0.0;
      }
      pBoosterShell->SetTermIndex(iTerm);

      LOG_0(Trace_Warning, "WARNING GenerateTermUpdate size_t { 0 } == pTerm->GetCountTensorBins()");
      return Error_None;
   }

   const size_t cInnerBagsAfterZero =
      size_t { 0 } == pBoosterCore->GetCountInnerBags() ? size_t { 1 } : pBoosterCore->GetCountInnerBags();
   const size_t cRealDimensions = pTerm->GetCountRealDimensions();
   const size_t cDimensions = pTerm->GetCountDimensions();

   // TODO: we can probably eliminate lastDimensionLeavesMax and cSignificantBinCount and just fetch them from iDimensionImportant afterwards
   IntEbm lastDimensionLeavesMax = IntEbm { 0 };
   // this initialization isn't required, but this variable ends up touching a lot of downstream state
   // and g++ seems to warn about all of that usage, even in other downstream functions!
   size_t cSignificantBinCount = size_t { 0 };
   size_t iDimensionImportant = 0;
   if(nullptr == leavesMax) {
      LOG_0(Trace_Warning, "WARNING GenerateTermUpdate leavesMax was null, so there won't be any splits");
   } else {
      if(0 != cRealDimensions) {
         size_t iDimensionInit = 0;
         const IntEbm * pLeavesMax = leavesMax;
         const FeatureBoosting * const * ppFeature = pTerm->GetFeatures();
         EBM_ASSERT(1 <= cDimensions);
         const FeatureBoosting * const * const ppFeaturesEnd = &ppFeature[cDimensions];
         do {
            const FeatureBoosting * const pFeature = *ppFeature;
            const size_t cBins = pFeature->GetCountBins();
            if(size_t { 1 } < cBins) {
               // if there is only 1 dimension then this is our first time here and lastDimensionLeavesMax must be zero
               EBM_ASSERT(size_t { 2 } <= cRealDimensions || IntEbm { 0 } == lastDimensionLeavesMax);

               iDimensionImportant = iDimensionInit;
               cSignificantBinCount = cBins;
               EBM_ASSERT(nullptr != pLeavesMax);
               const IntEbm countLeavesMax = *pLeavesMax;
               if(countLeavesMax <= IntEbm { 1 }) {
                  LOG_0(Trace_Warning, "WARNING GenerateTermUpdate countLeavesMax is 1 or less.");
               } else {
                  // keep iteration even once we find this so that we output logs for any bins of 1
                  lastDimensionLeavesMax = countLeavesMax;
               }
            }
            ++iDimensionInit;
            ++pLeavesMax;
            ++ppFeature;
         } while(ppFeaturesEnd != ppFeature);

         EBM_ASSERT(size_t { 2 } <= cSignificantBinCount);
      }
   }

   pBoosterShell->GetTermUpdate()->SetCountDimensions(cDimensions);
   pBoosterShell->GetTermUpdate()->Reset();

   // if pBoosterCore->m_apInnerBags is nullptr, then we should have zero training samples
   // we can't be partially constructed here since then we wouldn't have returned our state pointer to our caller

   double gainAvg = 0.0;
   const InnerBag * const * ppInnerBag = pBoosterCore->GetInnerBags();
   if(nullptr != ppInnerBag) {
      RandomDeterministic * pRng = reinterpret_cast<RandomDeterministic *>(rng);
      RandomDeterministic rngInternal;
      // TODO: move this code down into our called functions since we can happily pass down nullptr into there and then use the rng CPU register trick at the lowest function level
      if(nullptr == pRng) {
         // We use the RNG for two things during the boosting update, and none of them requires
         // a cryptographically secure random number generator. We use the RNG for:
         //   - Deciding ties in regular boosting, but we use random boosting in DP-EBMs, which doesn't have ties
         //   - Deciding split points during random boosting. The DP-EBM proof doesn't rely on the perfect 
         //     randomness of the chosen split points. It only relies on the fact that the splits are
         //     chosen independently of the data. We could allow an attacker to choose the split points,
         //     and privacy would be preserved provided the attacker was not able to look at the data when
         //     choosing the splits.
         //
         // Since we do not need high-quality non-determinism, generate a non-deterministic seed
         uint64_t seed;
         try {
            RandomNondeterministic<uint64_t> randomGenerator;
            seed = randomGenerator.Next(std::numeric_limits<uint64_t>::max());
         } catch(const std::bad_alloc &) {
            LOG_0(Trace_Warning, "WARNING GenerateTermUpdate Out of memory in std::random_device");
            return Error_OutOfMemory;
         } catch(...) {
            LOG_0(Trace_Warning, "WARNING GenerateTermUpdate Unknown error in std::random_device");
            return Error_UnexpectedInternal;
         }
         rngInternal.Initialize(seed);
         pRng = &rngInternal;
      }

      pBoosterShell->GetInnerTermUpdate()->SetCountDimensions(cDimensions);
      // if we have ignored dimensions, set the splits count to zero!
      // we only need to do this once instead of per-loop since any dimensions with 1 bin 
      // are going to remain having 0 splits.
      pBoosterShell->GetInnerTermUpdate()->Reset();

      EBM_ASSERT(1 <= cInnerBagsAfterZero);
      const InnerBag * const * const ppInnerBagsEnd = &ppInnerBag[cInnerBagsAfterZero];
      do {
         const InnerBag * const pInnerBag = *ppInnerBag;
         if(UNLIKELY(IntEbm { 0 } == lastDimensionLeavesMax)) {
            LOG_0(Trace_Warning, "WARNING GenerateTermUpdate boosting zero dimensional");
            error = BoostZeroDimensional(pBoosterShell, pInnerBag, flags);
            if(Error_None != error) {
               return error;
            }
         } else {
            double gain;
            if(0 != (BoostFlags_RandomSplits & flags) || 2 < cRealDimensions) {
               if(size_t { 1 } != cSamplesLeafMin) {
                  LOG_0(Trace_Warning,
                     "WARNING GenerateTermUpdate cSamplesLeafMin is ignored when doing random splitting"
                  );
               }
               // THIS RANDOM SPLIT OPTION IS PRIMARILY USED FOR DIFFERENTIAL PRIVACY EBMs

               error = BoostRandom(
                  pRng,
                  pBoosterShell,
                  iTerm,
                  pInnerBag,
                  flags,
                  leavesMax,
                  &gain
               );
               if(Error_None != error) {
                  return error;
               }
            } else if(1 == cRealDimensions) {
               EBM_ASSERT(nullptr != leavesMax); // otherwise we'd use BoostZeroDimensional above
               EBM_ASSERT(IntEbm { 2 } <= lastDimensionLeavesMax); // otherwise we'd use BoostZeroDimensional above
               EBM_ASSERT(size_t { 2 } <= cSignificantBinCount); // otherwise we'd use BoostZeroDimensional above

               error = BoostSingleDimensional(
                  pRng,
                  pBoosterShell,
                  iTerm,
                  cSignificantBinCount,
                  pInnerBag,
                  iDimensionImportant,
                  cSamplesLeafMin,
                  lastDimensionLeavesMax,
                  &gain
               );
               if(Error_None != error) {
                  return error;
               }
            } else {
               error = BoostMultiDimensional(
                  pBoosterShell,
                  iTerm,
                  pInnerBag,
                  cSamplesLeafMin,
                  &gain
               );
               if(Error_None != error) {
                  return error;
               }
            }

            // gain should be +inf if there was an overflow in our callees
            EBM_ASSERT(!std::isnan(gain));
            EBM_ASSERT(0 <= gain);

            const double weightTotal = static_cast<double>(pInnerBag->GetWeightTotal());
            EBM_ASSERT(0 < weightTotal); // if all are zeros we assume there are no weights and use the count

            // this could re-promote gain to be +inf again if weightTotal < 1.0
            // do the sample count inversion here in case adding all the avgeraged gains pushes us into +inf
            gain = gain / cInnerBagsAfterZero / weightTotal;
            gainAvg += gain;
            EBM_ASSERT(!std::isnan(gainAvg));
            EBM_ASSERT(0 <= gainAvg);
         }

         // TODO : when we thread this code, let's have each thread take a lock and update the combined line segment.  They'll each do it while the 
         // others are working, so there should be no blocking and our final result won't require adding by the main thread
         error = pBoosterShell->GetTermUpdate()->Add(*pBoosterShell->GetInnerTermUpdate());
         if(Error_None != error) {
            return error;
         }
         ++ppInnerBag;
      } while(ppInnerBagsEnd != ppInnerBag);

      // gainAvg is +inf on overflow. It cannot be NaN, but check for that anyways since it's free
      EBM_ASSERT(!std::isnan(gainAvg));
      EBM_ASSERT(0 <= gainAvg);

      if(UNLIKELY(/* NaN */ !LIKELY(gainAvg <= std::numeric_limits<double>::max()))) {
         // this also checks for NaN since NaN < anything is FALSE

         // indicate an error/overflow with -inf similar to interaction strength.
         // Making it -inf gives it the worst ranking possible and avoids the weirdness of NaN

         // it is possible that some of our inner bags overflowed but others did not
         // in some boosting we allow both an update and an overflow.  We indicate the overflow
         // to the caller via a negative gain, but we pass through any update and let the caller
         // decide if they want to stop boosting at that point or continue.
         // So, if there is an update do not reset it here

         gainAvg = k_illegalGainDouble;
      } else {
         EBM_ASSERT(!std::isnan(gainAvg));
         EBM_ASSERT(!std::isinf(gainAvg));
         EBM_ASSERT(0 <= gainAvg);
      }

      LOG_0(Trace_Verbose, "GenerateTermUpdate done sampling set loop");

      double multiple = 1.0; // TODO: get this from the loss function
      multiple /= cInnerBagsAfterZero;
      multiple *= learningRate;

      bool bBad;
      // we need to divide by the number of sampling sets that we constructed this from.
      // We also need to slow down our growth so that the more relevant Features get a chance to grow first so we multiply by a user defined learning rate
      bool bClassification = IsClassification(cClasses);
      if(bClassification) {
#ifdef EXPAND_BINARY_LOGITS
         static constexpr bool bExpandBinaryLogits = true;
#else // EXPAND_BINARY_LOGITS
         static constexpr bool bExpandBinaryLogits = false;
#endif // EXPAND_BINARY_LOGITS

         //if(0 <= k_iZeroLogit || ptrdiff_t { 2 } == pBoosterCore->m_cClasses && bExpandBinaryLogits) {
         //   EBM_ASSERT(ptrdiff_t { 2 } <= pBoosterCore->m_cClasses);
         //   // TODO : for classification with logit zeroing, is our learning rate essentially being inflated as 
         //       pBoosterCore->m_cClasses goes up?  If so, maybe we should divide by 
         //       pBoosterCore->m_cClasses here to keep learning rates as equivalent as possible..  
         //       Actually, I think the real solution here is that 
         //   pBoosterCore->m_pTermUpdate->Multiply(
         //      learningRateFloat / cInnerBagsAfterZero * (pBoosterCore->m_cClasses - 1) / 
         //      pBoosterCore->m_cClasses
         //   );
         //} else {
         //   // TODO : for classification, is our learning rate essentially being inflated as 
         //        pBoosterCore->m_cClasses goes up?  If so, maybe we should divide by 
         //        pBoosterCore->m_cClasses here to keep learning rates equivalent as possible
         //   pBoosterCore->m_pTermUpdate->Multiply(learningRateFloat / cInnerBagsAfterZero);
         //}

         // TODO: When NewtonBoosting is enabled, we need to multiply our rate by (K - 1)/K (see above), per:
         // https://arxiv.org/pdf/1810.09092v2.pdf (forumla 5) and also the 
         // Ping Li paper (algorithm #1, line 5, (K - 1) / K )
         // https://arxiv.org/pdf/1006.5051.pdf

         const bool bDividing = bExpandBinaryLogits && ptrdiff_t { 2 } == cClasses;
         if(bDividing) {
            bBad = pBoosterShell->GetTermUpdate()->MultiplyAndCheckForIssues(multiple * 0.5);
         } else {
            bBad = pBoosterShell->GetTermUpdate()->MultiplyAndCheckForIssues(multiple);
         }
      } else {
         bBad = pBoosterShell->GetTermUpdate()->MultiplyAndCheckForIssues(multiple);
      }

      if(UNLIKELY(bBad)) {
         // our update contains a NaN or -inf or +inf and we cannot tollerate a model that does this, so destroy it

         pBoosterShell->GetTermUpdate()->SetCountDimensions(cDimensions);
         pBoosterShell->GetTermUpdate()->Reset();

         // also, signal to our caller that an overflow occured with a negative gain
         gainAvg = k_illegalGainDouble;
      }
   }

   pBoosterShell->SetTermIndex(iTerm);

   EBM_ASSERT(!std::isnan(gainAvg));
   EBM_ASSERT(std::numeric_limits<double>::infinity() != gainAvg);
   EBM_ASSERT(k_illegalGainDouble == gainAvg || double { 0 } <= gainAvg);

   if(nullptr != avgGainOut) {
      *avgGainOut = gainAvg;
   }

   LOG_COUNTED_N(
      pTerm->GetPointerCountLogExitGenerateTermUpdateMessages(),
      Trace_Info,
      Trace_Verbose,
      "Exited GenerateTermUpdate: "
      "gainAvg=%le"
      ,
      gainAvg
   );

   return Error_None;
}

} // DEFINED_ZONE_NAME
