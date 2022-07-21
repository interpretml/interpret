// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <string.h> // memcpy

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "CompressibleTensor.hpp"
#include "ebm_stats.hpp"
// feature includes
#include "Feature.hpp"
// FeatureGroup.hpp depends on FeatureInternal.h
#include "FeatureGroup.hpp"
// dataset depends on features
#include "DataSetBoosting.hpp"
// samples is somewhat independent from datasets, but relies on an indirect coupling with them
#include "SamplingSet.hpp"
#include "HistogramTargetEntry.hpp"
#include "HistogramBucket.hpp"

#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern void BinSumsBoosting(
   BoosterShell * const pBoosterShell,
   const Term * const pTerm,
   const SamplingSet * const pTrainingSet
);

extern void SumAllBins(
   BoosterShell * const pBoosterShell,
   const size_t cBins
#ifndef NDEBUG
   , const size_t cSamplesTotal
   , const FloatBig weightTotal
#endif // NDEBUG
);

extern void TensorTotalsBuild(
   const ptrdiff_t cClasses,
   const Term * const pTerm,
   BinBase * aAuxiliaryBinsBase,
   BinBase * const aBinsBase
#ifndef NDEBUG
   , BinBase * const aBinsBaseDebugCopy
   , const unsigned char * const pBinsEndDebug
#endif // NDEBUG
);

extern ErrorEbmType PartitionOneDimensionalBoosting(
   BoosterShell * const pBoosterShell,
   const size_t cBins,
   const size_t cSamplesTotal,
   const FloatBig weightTotal,
   const size_t iDimension,
   const size_t cSamplesLeafMin,
   const size_t cLeavesMax,
   double * const pTotalGain
);

extern ErrorEbmType PartitionTwoDimensionalBoosting(
   BoosterShell * const pBoosterShell,
   const Term * const pTerm,
   const size_t cSamplesLeafMin,
   BinBase * aAuxiliaryBinsBase,
   double * const pTotalGain
#ifndef NDEBUG
   , const BinBase * const aBinsBaseDebugCopy
#endif // NDEBUG
);

extern ErrorEbmType PartitionRandomBoosting(
   BoosterShell * const pBoosterShell,
   const Term * const pTerm,
   const GenerateUpdateOptionsType options,
   const IntEbmType * const aLeavesMax,
   double * const pTotalGain
);

static ErrorEbmType BoostZeroDimensional(
   BoosterShell * const pBoosterShell, 
   const SamplingSet * const pTrainingSet,
   const GenerateUpdateOptionsType options
) {
   LOG_0(TraceLevelVerbose, "Entered BoostZeroDimensional");

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t cClasses = pBoosterCore->GetCountClasses();
   const bool bClassification = IsClassification(cClasses);

   const size_t cScores = GetCountScores(cClasses);

   if(IsOverflowBinSize<FloatFast>(bClassification, cScores) || IsOverflowBinSize<FloatBig>(bClassification, cScores)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "WARNING BoostZeroDimensional IsOverflowBinSize<FloatFast>(bClassification, cScores) || IsOverflowBinSize<FloatBig>(bClassification, cScores)");
      return Error_OutOfMemory;
   }
   const size_t cBytesPerBinFast = GetBinSize<FloatFast>(bClassification, cScores);

   BinBase * const pBinFast = pBoosterShell->GetBinBaseFast(cBytesPerBinFast);
   if(UNLIKELY(nullptr == pBinFast)) {
      // already logged
      return Error_OutOfMemory;
   }
   pBinFast->Zero(cBytesPerBinFast);

#ifndef NDEBUG
   pBoosterShell->SetBinsFastEndDebug(reinterpret_cast<unsigned char *>(pBinFast) + cBytesPerBinFast);
#endif // NDEBUG

   BinSumsBoosting(
      pBoosterShell,
      nullptr,
      pTrainingSet
   );

   const size_t cBytesPerBinBig = GetBinSize<FloatBig>(bClassification, cScores);

   BinBase * const pBinBig = pBoosterShell->GetBinBaseBig(cBytesPerBinBig);
   if(UNLIKELY(nullptr == pBinBig)) {
      // already logged
      return Error_OutOfMemory;
   }

#ifndef NDEBUG
   pBoosterShell->SetBinsBigEndDebug(reinterpret_cast<unsigned char *>(pBinBig) + cBytesPerBinBig);
#endif // NDEBUG

   // TODO: put this into it's own function that converts our fast floats to big floats
   static_assert(sizeof(FloatBig) == sizeof(FloatFast), "float mismatch");
   EBM_ASSERT(cBytesPerBinFast == cBytesPerBinBig); // until we switch fast to float datatypes
   memcpy(pBinBig, pBinFast, cBytesPerBinFast);


   // TODO: we can exit here back to python to allow caller modification to our histograms


   Tensor * const pInnerTermUpdate = pBoosterShell->GetInnerTermUpdate();
   FloatFast * aUpdateScores = pInnerTermUpdate->GetTensorScoresPointer();
   if(bClassification) {
      const auto * const pBin = pBinBig->Specialize<FloatBig, true>();
      const auto * const aGradientPairs = pBin->GetGradientPairs();
      if(0 != (GenerateUpdateOptions_GradientSums & options)) {
         for(size_t iScore = 0; iScore < cScores; ++iScore) {
            const FloatBig updateScore = EbmStats::ComputeSinglePartitionUpdateGradientSum(aGradientPairs[iScore].m_sumGradients);

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
            // Hmmm.. for DP we need the sum, which means that we can't zero one of the class numbers as we
            // could with one of the logits in multiclass.
#endif // ZERO_FIRST_MULTICLASS_LOGIT

            aUpdateScores[iScore] = SafeConvertFloat<FloatFast>(updateScore);
         }
      } else {

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
         FloatBig zeroLogit = 0;
#endif // ZERO_FIRST_MULTICLASS_LOGIT

         for(size_t iScore = 0; iScore < cScores; ++iScore) {
            FloatBig updateScore = EbmStats::ComputeSinglePartitionUpdate(
               aGradientPairs[iScore].m_sumGradients,
               aGradientPairs[iScore].GetSumHessians()
            );

#ifdef ZERO_FIRST_MULTICLASS_LOGIT
            if(IsMulticlass(cClasses)) {
               if(size_t { 0 } == iScore) {
                  zeroLogit = updateScore;
               }
               updateScore -= zeroLogit;
            }
#endif // ZERO_FIRST_MULTICLASS_LOGIT

            aUpdateScores[iScore] = SafeConvertFloat<FloatFast>(updateScore);
         }
      }
   } else {
      EBM_ASSERT(IsRegression(cClasses));
      const auto * const pBin = pBinBig->Specialize<FloatBig, false>();
      const auto * const aGradientPairs = pBin->GetGradientPairs();
      if(0 != (GenerateUpdateOptions_GradientSums & options)) {
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

   LOG_0(TraceLevelVerbose, "Exited BoostZeroDimensional");
   return Error_None;
}

static ErrorEbmType BoostSingleDimensional(
   BoosterShell * const pBoosterShell,
   const Term * const pTerm,
   const size_t cBins,
   const SamplingSet * const pTrainingSet,
   const size_t iDimension,
   const size_t cSamplesLeafMin,
   const IntEbmType countLeavesMax,
   double * const pTotalGain
) {
   ErrorEbmType error;

   LOG_0(TraceLevelVerbose, "Entered BoostSingleDimensional");

   EBM_ASSERT(1 == pTerm->GetCountSignificantDimensions());

   EBM_ASSERT(IntEbmType { 2 } <= countLeavesMax); // otherwise we would have called BoostZeroDimensional
   size_t cLeavesMax = static_cast<size_t>(countLeavesMax);
   if(IsConvertError<size_t>(countLeavesMax)) {
      // we can never exceed a size_t number of leaves, so let's just set it to the maximum if we were going to overflow because it will generate 
      // the same results as if we used the true number
      cLeavesMax = std::numeric_limits<size_t>::max();
   }

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t cClasses = pBoosterCore->GetCountClasses();
   const bool bClassification = IsClassification(cClasses);
   const size_t cScores = GetCountScores(cClasses);

   if(IsOverflowBinSize<FloatFast>(bClassification, cScores) ||
      IsOverflowBinSize<FloatBig>(bClassification, cScores)) 
   {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "WARNING BoostSingleDimensional IsOverflowBinSize<FloatFast>(bClassification, cScores) || IsOverflowBinSize<FloatBig>(bClassification, cScores)");
      return Error_OutOfMemory;
   }

   const size_t cBytesPerBinFast = GetBinSize<FloatFast>(bClassification, cScores);
   if(IsMultiplyError(cBytesPerBinFast, cBins)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "WARNING BoostSingleDimensional IsMultiplyError(cBytesPerBinFast, cBins)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBufferFast = cBytesPerBinFast * cBins;

   BinBase * const aBinsFast = pBoosterShell->GetBinBaseFast(cBytesBufferFast);
   if(UNLIKELY(nullptr == aBinsFast)) {
      // already logged
      return Error_OutOfMemory;
   }
   aBinsFast->Zero(cBytesPerBinFast, cBins);

#ifndef NDEBUG
   pBoosterShell->SetBinsFastEndDebug(reinterpret_cast<unsigned char *>(aBinsFast) + cBytesBufferFast);
#endif // NDEBUG

   BinSumsBoosting(
      pBoosterShell,
      pTerm,
      pTrainingSet
   );

   const size_t cBytesPerBinBig = GetBinSize<FloatBig>(bClassification, cScores);
   if(IsMultiplyError(cBytesPerBinBig, cBins)) {
      // TODO : move this to initialization where we execute it only once
      LOG_0(TraceLevelWarning, "WARNING BoostSingleDimensional IsMultiplyError(cBytesPerBinBig, cBins)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBufferBig = cBytesPerBinBig * cBins;

   BinBase * const aBinsBig = pBoosterShell->GetBinBaseBig(cBytesBufferBig);
   if(UNLIKELY(nullptr == aBinsBig)) {
      // already logged
      return Error_OutOfMemory;
   }

#ifndef NDEBUG
   pBoosterShell->SetBinsBigEndDebug(reinterpret_cast<unsigned char *>(aBinsBig) + cBytesBufferBig);
#endif // NDEBUG

   // TODO: put this into it's own function that converts our fast floats to big floats
   static_assert(sizeof(FloatBig) == sizeof(FloatFast), "float mismatch");
   EBM_ASSERT(cBytesBufferFast == cBytesBufferBig); // until we switch fast to float datatypes
   memcpy(aBinsBig, aBinsFast, cBytesBufferFast);


   // TODO: we can exit here back to python to allow caller modification to our histograms


   GradientPairBase * const aSumAllGradientPairs = pBoosterShell->GetSumAllGradientPairs();
   const size_t cBytesPerGradientPair = GetGradientPairSize<FloatBig>(bClassification);
   aSumAllGradientPairs->Zero(cBytesPerGradientPair, cScores);

   SumAllBins(
      pBoosterShell,
      cBins
#ifndef NDEBUG
      , pTrainingSet->GetTotalCountSampleOccurrences()
      , pTrainingSet->GetWeightTotal()
#endif // NDEBUG
   );

   const size_t cSamplesTotal = pTrainingSet->GetTotalCountSampleOccurrences();
   EBM_ASSERT(1 <= cSamplesTotal);
   const FloatBig weightTotal = pTrainingSet->GetWeightTotal();

   error = PartitionOneDimensionalBoosting(
      pBoosterShell,
      cBins,
      cSamplesTotal,
      weightTotal,
      iDimension,
      cSamplesLeafMin,
      cLeavesMax, 
      pTotalGain
   );

   LOG_0(TraceLevelVerbose, "Exited BoostSingleDimensional");
   return error;
}

// TODO: for higher dimensional spaces, we need to add/subtract individual cells alot and the hessian isn't required (yet) in order to make decisions about
//   where to split.  For dimensions higher than 2, we might want to copy the tensor to a new tensor AFTER binning that keeps only the gradients and then 
//    go back to our original tensor after splits to determine the hessian
static ErrorEbmType BoostMultiDimensional(
   BoosterShell * const pBoosterShell,
   const Term * const pTerm,
   const SamplingSet * const pTrainingSet,
   const size_t cSamplesLeafMin,
   double * const pTotalGain
) {
   LOG_0(TraceLevelVerbose, "Entered BoostMultiDimensional");

   EBM_ASSERT(2 <= pTerm->GetCountDimensions());
   EBM_ASSERT(2 <= pTerm->GetCountSignificantDimensions());

   ErrorEbmType error;

   size_t cAuxillaryBinsForBuildFastTotals = 0;
   size_t cTotalBinsMainSpace = 1;

   const TermEntry * pTermEntry = pTerm->GetTermEntries();
   const TermEntry * const pTermEntriesEnd = pTermEntry + pTerm->GetCountDimensions();
   do {
      const size_t cBins = pTermEntry->m_pFeature->GetCountBins();
      EBM_ASSERT(size_t { 1 } <= cBins); // we don't boost on empty training sets
      if(size_t { 1 } < cBins) {
         // if this wasn't true then we'd have to check IsAddError(cAuxillaryBinsForBuildFastTotals, cTotalBinsMainSpace) at runtime
         EBM_ASSERT(cAuxillaryBinsForBuildFastTotals < cTotalBinsMainSpace);
         // since cBins must be 2 or more, cAuxillaryBinsForBuildFastTotals must grow slower than cTotalBinsMainSpace, and we checked at 
         // allocation that cTotalBinsMainSpace would not overflow
         EBM_ASSERT(!IsAddError(cAuxillaryBinsForBuildFastTotals, cTotalBinsMainSpace));
         cAuxillaryBinsForBuildFastTotals += cTotalBinsMainSpace;
         // we check for simple multiplication overflow from m_cBins in pBoosterCore->Initialize when we unpack featureIndexes
         EBM_ASSERT(!IsMultiplyError(cTotalBinsMainSpace, cBins));
         cTotalBinsMainSpace *= cBins;
         // if this wasn't true then we'd have to check IsAddError(cAuxillaryBinsForBuildFastTotals, cTotalBinsMainSpace) at runtime
         EBM_ASSERT(cAuxillaryBinsForBuildFastTotals < cTotalBinsMainSpace);
      }
      ++pTermEntry;
   } while(pTermEntriesEnd != pTermEntry);

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t cClasses = pBoosterCore->GetCountClasses();
   const bool bClassification = IsClassification(cClasses);
   const size_t cScores = GetCountScores(cClasses);

   if(IsOverflowBinSize<FloatFast>(bClassification, cScores) || 
      IsOverflowBinSize<FloatBig>(bClassification, cScores)) 
   {
      LOG_0(
         TraceLevelWarning,
         "WARNING BoostMultiDimensional IsOverflowBinSize<FloatFast>(bClassification, cScores) || IsOverflowBinSize<FloatBig>(bClassification, cScores)"
      );
      return Error_OutOfMemory;
   }
   const size_t cBytesPerBinFast = GetBinSize<FloatFast>(bClassification, cScores);
   if(IsMultiplyError(cBytesPerBinFast, cTotalBinsMainSpace)) {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional IsMultiplyError(cBytesPerBinFast, cTotalBinsMainSpace)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBufferFast = cBytesPerBinFast * cTotalBinsMainSpace;

   // we don't need to free this!  It's tracked and reused by pBoosterShell
   BinBase * const aBinsFast = pBoosterShell->GetBinBaseFast(cBytesBufferFast);
   if(UNLIKELY(nullptr == aBinsFast)) {
      // already logged
      return Error_OutOfMemory;
   }
   aBinsFast->Zero(cBytesPerBinFast, cTotalBinsMainSpace);

#ifndef NDEBUG
   pBoosterShell->SetBinsFastEndDebug(reinterpret_cast<unsigned char *>(aBinsFast) + cBytesBufferFast);
#endif // NDEBUG

   BinSumsBoosting(
      pBoosterShell,
      pTerm,
      pTrainingSet
   );

   // we need to reserve 4 PAST the pointer we pass into SweepMultiDimensional!!!!.  We pass in index 20 at max, so we need 24
   const size_t cAuxillaryBinsForSplitting = 24;
   const size_t cAuxillaryBins =
      cAuxillaryBinsForBuildFastTotals < cAuxillaryBinsForSplitting ? cAuxillaryBinsForSplitting : cAuxillaryBinsForBuildFastTotals;
   if(IsAddError(cTotalBinsMainSpace, cAuxillaryBins)) {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional IsAddError(cTotalBinsMainSpace, cAuxillaryBins)");
      return Error_OutOfMemory;
   }
   const size_t cTotalBinsBig = cTotalBinsMainSpace + cAuxillaryBins;

   const size_t cBytesPerBinBig = GetBinSize<FloatBig>(bClassification, cScores);
   if(IsMultiplyError(cBytesPerBinBig, cTotalBinsBig)) {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional IsMultiplyError(cBytesPerBinBig, cTotalBinsBig)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBufferBig = cBytesPerBinBig * cTotalBinsBig;

   // we don't need to free this!  It's tracked and reused by pBoosterShell
   BinBase * const aBinsBig = pBoosterShell->GetBinBaseBig(cBytesBufferBig);
   if(UNLIKELY(nullptr == aBinsBig)) {
      // already logged
      return Error_OutOfMemory;
   }

#ifndef NDEBUG
   const unsigned char * const pBinsBigEndDebug = reinterpret_cast<unsigned char *>(aBinsBig) + cBytesBufferBig;
   pBoosterShell->SetBinsBigEndDebug(pBinsBigEndDebug);
#endif // NDEBUG

   // TODO: put this into it's own function that converts our fast floats to big floats
   static_assert(sizeof(FloatBig) == sizeof(FloatFast), "float mismatch");
   memcpy(aBinsBig, aBinsFast, cBytesBufferFast);


   // we also need to zero the auxillary bins
   aBinsBig->Zero(cBytesPerBinBig, cAuxillaryBins, cTotalBinsMainSpace);


   // TODO: we can exit here back to python to allow caller modification to our histograms


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

   BinBase * aAuxiliaryBins = IndexBin(
      cBytesPerBinBig,
      aBinsBig,
      cTotalBinsMainSpace
   );

   TensorTotalsBuild(
      cClasses,
      pTerm,
      aAuxiliaryBins,
      aBinsBig
#ifndef NDEBUG
      , aBinsDebugCopy
      , pBinsBigEndDebug
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
   //               pTerms->GetTermEntries()[aiDimensionPermutation[iDimension]].m_pFeature->m_cBins) {
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

   if(2 == pTerm->GetCountSignificantDimensions()) {
      error = PartitionTwoDimensionalBoosting(
         pBoosterShell,
         pTerm,
         cSamplesLeafMin,
         aAuxiliaryBins,
         pTotalGain
#ifndef NDEBUG
         , aBinsDebugCopy
#endif // NDEBUG
      );
      if(Error_None != error) {
#ifndef NDEBUG
         free(aBinsDebugCopy);
#endif // NDEBUG

         LOG_0(TraceLevelVerbose, "Exited BoostMultiDimensional with Error code");

         return error;
      }

      EBM_ASSERT(!std::isnan(*pTotalGain));
      EBM_ASSERT(0 <= *pTotalGain);
   } else {
      LOG_0(TraceLevelWarning, "WARNING BoostMultiDimensional 2 != pTerm->GetCountSignificantFeatures()");

      // TODO: eventually handle this in our caller and this function can specialize in handling just 2 dimensional
      //       then we can replace this branch with an assert
#ifndef NDEBUG
      EBM_ASSERT(false);
      free(aBinsDebugCopy);
#endif // NDEBUG
      return Error_UnexpectedInternal;
   }

#ifndef NDEBUG
   free(aBinsDebugCopy);
#endif // NDEBUG

   LOG_0(TraceLevelVerbose, "Exited BoostMultiDimensional");
   return Error_None;
}

static ErrorEbmType BoostRandom(
   BoosterShell * const pBoosterShell,
   const Term * const pTerm,
   const SamplingSet * const pTrainingSet,
   const GenerateUpdateOptionsType options,
   const IntEbmType * const aLeavesMax,
   double * const pTotalGain
) {
   // THIS RANDOM SPLIT FUNCTION IS PRIMARILY USED FOR DIFFERENTIAL PRIVACY EBMs

   LOG_0(TraceLevelVerbose, "Entered BoostRandom");

   ErrorEbmType error;

   const size_t cDimensions = pTerm->GetCountDimensions();
   EBM_ASSERT(1 <= cDimensions);

   size_t cTotalBins = 1;
   for(size_t iDimension = 0; iDimension < cDimensions; ++iDimension) {
      const size_t cBins = pTerm->GetTermEntries()[iDimension].m_pFeature->GetCountBins();
      EBM_ASSERT(size_t { 1 } <= cBins); // we don't boost on empty training sets
      // we check for simple multiplication overflow from m_cBins in BoosterCore::Initialize when we unpack featureIndexes
      EBM_ASSERT(!IsMultiplyError(cTotalBins, cBins));
      cTotalBins *= cBins;
   }

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t cClasses = pBoosterCore->GetCountClasses();
   const bool bClassification = IsClassification(cClasses);
   const size_t cScores = GetCountScores(cClasses);

   if(IsOverflowBinSize<FloatFast>(bClassification, cScores) ||
      IsOverflowBinSize<FloatBig>(bClassification, cScores))
   {
      LOG_0(
         TraceLevelWarning,
         "WARNING BoostRandom IsOverflowBinSize<FloatFast>(bClassification, cScores) || IsOverflowBinSize<FloatBig>(bClassification, cScores)"
      );
      return Error_OutOfMemory;
   }
   const size_t cBytesPerBinFast = GetBinSize<FloatFast>(bClassification, cScores);
   if(IsMultiplyError(cBytesPerBinFast, cTotalBins)) {
      LOG_0(TraceLevelWarning, "WARNING BoostRandom IsMultiplyError(cBytesPerBinFast, cTotalBins)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBufferFast = cBytesPerBinFast * cTotalBins;

   // we don't need to free this!  It's tracked and reused by pBoosterShell
   BinBase * const aBinsFast = pBoosterShell->GetBinBaseFast(cBytesBufferFast);
   if(UNLIKELY(nullptr == aBinsFast)) {
      // already logged
      return Error_OutOfMemory;
   }
   aBinsFast->Zero(cBytesPerBinFast, cTotalBins);

#ifndef NDEBUG
   pBoosterShell->SetBinsFastEndDebug(reinterpret_cast<unsigned char *>(aBinsFast) + cBytesBufferFast);
#endif // NDEBUG

   BinSumsBoosting(
      pBoosterShell,
      pTerm,
      pTrainingSet
   );

   const size_t cBytesPerBinBig = GetBinSize<FloatBig>(bClassification, cScores);
   if(IsMultiplyError(cBytesPerBinBig, cTotalBins)) {
      LOG_0(TraceLevelWarning, "WARNING BoostRandom IsMultiplyError(cBytesPerBinBig, cTotalBins)");
      return Error_OutOfMemory;
   }
   const size_t cBytesBufferBig = cBytesPerBinBig * cTotalBins;

   // we don't need to free this!  It's tracked and reused by pBoosterShell
   BinBase * const aBinsBig = pBoosterShell->GetBinBaseBig(cBytesBufferBig);
   if(UNLIKELY(nullptr == aBinsBig)) {
      // already logged
      return Error_OutOfMemory;
   }

#ifndef NDEBUG
   pBoosterShell->SetBinsBigEndDebug(reinterpret_cast<unsigned char *>(aBinsBig) + cBytesBufferBig);
#endif // NDEBUG

   // TODO: put this into it's own function that converts our fast floats to big floats
   static_assert(sizeof(FloatBig) == sizeof(FloatFast), "float mismatch");
   EBM_ASSERT(cBytesBufferFast == cBytesBufferBig); // until we switch fast to float datatypes
   memcpy(aBinsBig, aBinsFast, cBytesBufferFast);


   // TODO: we can exit here back to python to allow caller modification to our histograms


   error = PartitionRandomBoosting(
      pBoosterShell,
      pTerm,
      options,
      aLeavesMax,
      pTotalGain
   );
   if(Error_None != error) {
      LOG_0(TraceLevelVerbose, "Exited BoostRandom with Error code");
      return error;
   }

   EBM_ASSERT(!std::isnan(*pTotalGain));
   EBM_ASSERT(0 <= *pTotalGain);

   LOG_0(TraceLevelVerbose, "Exited BoostRandom");
   return Error_None;
}

static ErrorEbmType GenerateTermUpdateInternal(
   BoosterShell * const pBoosterShell,
   const size_t iTerm,
   const GenerateUpdateOptionsType options,
   const double learningRate,
   const size_t cSamplesLeafMin,
   const IntEbmType * const aLeavesMax, 
   double * const pGainAvgOut
) {
   ErrorEbmType error;

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const ptrdiff_t cClasses = pBoosterCore->GetCountClasses();
   const bool bClassification = IsClassification(cClasses);

   LOG_0(TraceLevelVerbose, "Entered GenerateTermUpdateInternal");

   const size_t cSamplingSetsAfterZero = (0 == pBoosterCore->GetCountSamplingSets()) ? 1 : pBoosterCore->GetCountSamplingSets();
   const Term * const pTerm = pBoosterCore->GetTerms()[iTerm];
   const size_t cSignificantDimensions = pTerm->GetCountSignificantDimensions();
   const size_t cDimensions = pTerm->GetCountDimensions();

   // TODO: we can probably eliminate lastDimensionLeavesMax and cSignificantBinCount and just fetch them from iDimensionImportant afterwards
   IntEbmType lastDimensionLeavesMax = IntEbmType { 0 };
   // this initialization isn't required, but this variable ends up touching a lot of downstream state
   // and g++ seems to warn about all of that usage, even in other downstream functions!
   size_t cSignificantBinCount = size_t { 0 };
   size_t iDimensionImportant = 0;
   if(nullptr == aLeavesMax) {
      LOG_0(TraceLevelWarning, "WARNING GenerateTermUpdateInternal aLeavesMax was null, so there won't be any splits");
   } else {
      if(0 != cSignificantDimensions) {
         size_t iDimensionInit = 0;
         const IntEbmType * pLeavesMax = aLeavesMax;
         const TermEntry * pTermEntry = pTerm->GetTermEntries();
         EBM_ASSERT(1 <= cDimensions);
         const TermEntry * const pTermEntriesEnd = pTermEntry + cDimensions;
         do {
            const Feature * pFeature = pTermEntry->m_pFeature;
            const size_t cBins = pFeature->GetCountBins();
            if(size_t { 1 } < cBins) {
               EBM_ASSERT(size_t { 2 } <= cSignificantDimensions || IntEbmType { 0 } == lastDimensionLeavesMax);

               iDimensionImportant = iDimensionInit;
               cSignificantBinCount = cBins;
               EBM_ASSERT(nullptr != pLeavesMax);
               const IntEbmType countLeavesMax = *pLeavesMax;
               if(countLeavesMax <= IntEbmType { 1 }) {
                  LOG_0(TraceLevelWarning, "WARNING GenerateTermUpdateInternal countLeavesMax is 1 or less.");
               } else {
                  // keep iteration even once we find this so that we output logs for any bins of 1
                  lastDimensionLeavesMax = countLeavesMax;
               }
            }
            ++iDimensionInit;
            ++pLeavesMax;
            ++pTermEntry;
         } while(pTermEntriesEnd != pTermEntry);
         
         EBM_ASSERT(size_t { 2 } <= cSignificantBinCount);
      }
   }

   pBoosterShell->GetTermUpdate()->SetCountDimensions(cDimensions);
   pBoosterShell->GetTermUpdate()->Reset();

   // if pBoosterCore->m_apSamplingSets is nullptr, then we should have zero training samples
   // we can't be partially constructed here since then we wouldn't have returned our state pointer to our caller

   double gainAvgOut = 0.0;
   const SamplingSet * const * ppSamplingSet = pBoosterCore->GetSamplingSets();
   if(nullptr != ppSamplingSet) {
      pBoosterShell->GetInnerTermUpdate()->SetCountDimensions(cDimensions);
      // if we have ignored dimensions, set the splits count to zero!
      // we only need to do this once instead of per-loop since any dimensions with 1 bin 
      // are going to remain having 0 splits.
      pBoosterShell->GetInnerTermUpdate()->Reset();

      EBM_ASSERT(1 <= cSamplingSetsAfterZero);
      const SamplingSet * const * const ppSamplingSetEnd = &ppSamplingSet[cSamplingSetsAfterZero];
      const double invertedSampleCount = 1.0 / cSamplingSetsAfterZero;
      double gainAvg = 0;
      do {
         const SamplingSet * const pSamplingSet = *ppSamplingSet;
         if(UNLIKELY(IntEbmType { 0 } == lastDimensionLeavesMax)) {
            LOG_0(TraceLevelWarning, "WARNING GenerateTermUpdateInternal boosting zero dimensional");
            error = BoostZeroDimensional(pBoosterShell, pSamplingSet, options);
            if(Error_None != error) {
               if(LIKELY(nullptr != pGainAvgOut)) {
                  *pGainAvgOut = double { 0 };
               }
               return error;
            }
         } else {
            double gain;
            if(0 != (GenerateUpdateOptions_RandomSplits & options) || 2 < cSignificantDimensions) {
               if(size_t { 1 } != cSamplesLeafMin) {
                  LOG_0(TraceLevelWarning,
                     "WARNING GenerateTermUpdateInternal cSamplesLeafMin is ignored when doing random splitting"
                  );
               }
               // THIS RANDOM SPLIT OPTION IS PRIMARILY USED FOR DIFFERENTIAL PRIVACY EBMs

               error = BoostRandom(
                  pBoosterShell,
                  pTerm,
                  pSamplingSet,
                  options,
                  aLeavesMax,
                  &gain
               );
               if(Error_None != error) {
                  if(LIKELY(nullptr != pGainAvgOut)) {
                     *pGainAvgOut = double { 0 };
                  }
                  return error;
               }
            } else if(1 == cSignificantDimensions) {
               EBM_ASSERT(nullptr != aLeavesMax); // otherwise we'd use BoostZeroDimensional above
               EBM_ASSERT(IntEbmType { 2 } <= lastDimensionLeavesMax); // otherwise we'd use BoostZeroDimensional above
               EBM_ASSERT(size_t { 2 } <= cSignificantBinCount); // otherwise we'd use BoostZeroDimensional above

               error = BoostSingleDimensional(
                  pBoosterShell,
                  pTerm,
                  cSignificantBinCount,
                  pSamplingSet,
                  iDimensionImportant,
                  cSamplesLeafMin,
                  lastDimensionLeavesMax,
                  &gain
               );
               if(Error_None != error) {
                  if(LIKELY(nullptr != pGainAvgOut)) {
                     *pGainAvgOut = double { 0 };
                  }
                  return error;
               }
            } else {
               error = BoostMultiDimensional(
                  pBoosterShell,
                  pTerm,
                  pSamplingSet,
                  cSamplesLeafMin,
                  &gain
               );
               if(Error_None != error) {
                  if(LIKELY(nullptr != pGainAvgOut)) {
                     *pGainAvgOut = double { 0 };
                  }
                  return error;
               }
            }

            // gain should be +inf if there was an overflow in our callees
            EBM_ASSERT(!std::isnan(gain));
            EBM_ASSERT(0 <= gain);

            const double weightTotal = static_cast<double>(pSamplingSet->GetWeightTotal());
            EBM_ASSERT(0 < weightTotal); // if all are zeros we assume there are no weights and use the count

            // this could re-promote gain to be +inf again if weightTotal < 1.0
            // do the sample count inversion here in case adding all the avgeraged gains pushes us into +inf
            EBM_ASSERT(invertedSampleCount <= 1);
            gain = gain * invertedSampleCount / weightTotal;
            gainAvg += gain;
            EBM_ASSERT(!std::isnan(gainAvg));
            EBM_ASSERT(0 <= gainAvg);
         }

         // TODO : when we thread this code, let's have each thread take a lock and update the combined line segment.  They'll each do it while the 
         // others are working, so there should be no blocking and our final result won't require adding by the main thread
         error = pBoosterShell->GetTermUpdate()->Add(*pBoosterShell->GetInnerTermUpdate());
         if(Error_None != error) {
            if(LIKELY(nullptr != pGainAvgOut)) {
               *pGainAvgOut = double { 0 };
            }
            return error;
         }
         ++ppSamplingSet;
      } while(ppSamplingSetEnd != ppSamplingSet);

      // gainAvg is +inf on overflow. It cannot be NaN, but check for that anyways since it's free
      EBM_ASSERT(!std::isnan(gainAvg));
      EBM_ASSERT(0 <= gainAvg);

      gainAvgOut = static_cast<double>(gainAvg);
      if(UNLIKELY(/* NaN */ !LIKELY(gainAvg <= std::numeric_limits<double>::max()))) {
         // this also checks for NaN since NaN < anything is FALSE

         // indicate an error/overflow with -inf similar to interaction strength.
         // Making it -inf gives it the worst ranking possible and avoids the weirdness of NaN

         // it is possible that some of our inner bags overflowed but others did not
         // in some boosting we allow both an update and an overflow.  We indicate the overflow
         // to the caller via a negative gain, but we pass through any update and let the caller
         // decide if they want to stop boosting at that point or continue.
         // So, if there is an update do not reset it here

         gainAvgOut = k_illegalGainDouble;
      } else {
         EBM_ASSERT(!std::isnan(gainAvg));
         EBM_ASSERT(!std::isinf(gainAvg));
         EBM_ASSERT(0 <= gainAvg);
      }

      LOG_0(TraceLevelVerbose, "GenerateTermUpdateInternal done sampling set loop");

      double multiple = 1.0; // TODO: get this from the loss function
      multiple /= cSamplingSetsAfterZero;
      multiple *= learningRate;

      bool bBad;
      // we need to divide by the number of sampling sets that we constructed this from.
      // We also need to slow down our growth so that the more relevant Features get a chance to grow first so we multiply by a user defined learning rate
      if(bClassification) {
#ifdef EXPAND_BINARY_LOGITS
         constexpr bool bExpandBinaryLogits = true;
#else // EXPAND_BINARY_LOGITS
         constexpr bool bExpandBinaryLogits = false;
#endif // EXPAND_BINARY_LOGITS

         //if(0 <= k_iZeroLogit || ptrdiff_t { 2 } == pBoosterCore->m_cClasses && bExpandBinaryLogits) {
         //   EBM_ASSERT(ptrdiff_t { 2 } <= pBoosterCore->m_cClasses);
         //   // TODO : for classification with logit zeroing, is our learning rate essentially being inflated as 
         //       pBoosterCore->m_cClasses goes up?  If so, maybe we should divide by 
         //       pBoosterCore->m_cClasses here to keep learning rates as equivalent as possible..  
         //       Actually, I think the real solution here is that 
         //   pBoosterCore->m_pTermUpdate->Multiply(
         //      learningRateFloat * invertedSampleCount * (pBoosterCore->m_cClasses - 1) / 
         //      pBoosterCore->m_cClasses
         //   );
         //} else {
         //   // TODO : for classification, is our learning rate essentially being inflated as 
         //        pBoosterCore->m_cClasses goes up?  If so, maybe we should divide by 
         //        pBoosterCore->m_cClasses here to keep learning rates equivalent as possible
         //   pBoosterCore->m_pTermUpdate->Multiply(learningRateFloat * invertedSampleCount);
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
         gainAvgOut = k_illegalGainDouble;
      }
   }

   pBoosterShell->SetTermIndex(iTerm);

   EBM_ASSERT(!std::isnan(gainAvgOut));
   EBM_ASSERT(std::numeric_limits<double>::infinity() != gainAvgOut);
   EBM_ASSERT(k_illegalGainDouble == gainAvgOut || double { 0 } <= gainAvgOut);

   if(nullptr != pGainAvgOut) {
      *pGainAvgOut = gainAvgOut;
   }

   LOG_0(TraceLevelVerbose, "Exited GenerateTermUpdateInternal");
   return Error_None;
}

// we made this a global because if we had put this variable inside the BoosterCore object, then we would need to dereference that before getting 
// the count.  By making this global we can send a log message incase a bad BoosterCore object is sent into us we only decrease the count if the 
// count is non-zero, so at worst if there is a race condition then we'll output this log message more times than desired, but we can live with that
static int g_cLogGenerateTermUpdateParametersMessages = 10;


EBM_API_BODY ErrorEbmType EBM_CALLING_CONVENTION GenerateTermUpdate(
   BoosterHandle boosterHandle,
   IntEbmType indexTerm,
   GenerateUpdateOptionsType options,
   double learningRate,
   IntEbmType minSamplesLeaf,
   const IntEbmType * leavesMax,
   double * avgGainOut
) {
   LOG_COUNTED_N(
      &g_cLogGenerateTermUpdateParametersMessages,
      TraceLevelInfo,
      TraceLevelVerbose,
      "GenerateTermUpdate: "
      "boosterHandle=%p, "
      "indexTerm=%" IntEbmTypePrintf ", "
      "options=0x%" UGenerateUpdateOptionsTypePrintf ", "
      "learningRate=%le, "
      "minSamplesLeaf=%" IntEbmTypePrintf ", "
      "leavesMax=%p, "
      "avgGainOut=%p"
      ,
      static_cast<void *>(boosterHandle),
      indexTerm,
      static_cast<UGenerateUpdateOptionsType>(options), // signed to unsigned conversion is defined behavior in C++
      learningRate,
      minSamplesLeaf,
      static_cast<const void *>(leavesMax),
      static_cast<void *>(avgGainOut)
   );

   ErrorEbmType error;

   BoosterShell * const pBoosterShell = BoosterShell::GetBoosterShellFromHandle(boosterHandle);
   if(nullptr == pBoosterShell) {
      if(LIKELY(nullptr != avgGainOut)) {
         *avgGainOut = double { 0 };
      }
      // already logged
      return Error_IllegalParamValue;
   }

   // set this to illegal so if we exit with an error we have an invalid index
   pBoosterShell->SetTermIndex(BoosterShell::k_illegalTermIndex);

   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   EBM_ASSERT(nullptr != pBoosterCore);

   if(indexTerm < 0) {
      if(LIKELY(nullptr != avgGainOut)) {
         *avgGainOut = double { 0 };
      }
      LOG_0(TraceLevelError, "ERROR GenerateTermUpdate indexTerm must be positive");
      return Error_IllegalParamValue;
   }
   if(IsConvertError<size_t>(indexTerm)) {
      // we wouldn't have allowed the creation of an feature set larger than size_t
      if(LIKELY(nullptr != avgGainOut)) {
         *avgGainOut = double { 0 };
      }
      LOG_0(TraceLevelError, "ERROR GenerateTermUpdate indexTerm is too high to index");
      return Error_IllegalParamValue;
   }
   size_t iTerm = static_cast<size_t>(indexTerm);
   if(pBoosterCore->GetCountTerms() <= iTerm) {
      if(LIKELY(nullptr != avgGainOut)) {
         *avgGainOut = double { 0 };
      }
      LOG_0(TraceLevelError, "ERROR GenerateTermUpdate indexTerm above the number of feature groups that we have");
      return Error_IllegalParamValue;
   }
   // this is true because 0 < pBoosterCore->m_cTerms since our caller needs to pass in a valid indexTerm to this function
   EBM_ASSERT(nullptr != pBoosterCore->GetTerms());

   LOG_COUNTED_0(
      pBoosterCore->GetTerms()[iTerm]->GetPointerCountLogEnterGenerateTermUpdateMessages(),
      TraceLevelInfo,
      TraceLevelVerbose,
      "Entered GenerateTermUpdate"
   );

   // TODO : test if our GenerateUpdateOptionsType options flags only include flags that we use

   if(std::isnan(learningRate)) {
      LOG_0(TraceLevelWarning, "WARNING GenerateTermUpdate learningRate is NaN");
   } else if(std::numeric_limits<double>::infinity() == learningRate) {
      LOG_0(TraceLevelWarning, "WARNING GenerateTermUpdate learningRate is +infinity");
   } else if(0.0 == learningRate) {
      LOG_0(TraceLevelWarning, "WARNING GenerateTermUpdate learningRate is zero");
   } else if(learningRate < double { 0 }) {
      LOG_0(TraceLevelWarning, "WARNING GenerateTermUpdate learningRate is negative");
   }

   size_t cSamplesLeafMin = size_t { 1 }; // this is the min value
   if(IntEbmType { 1 } <= minSamplesLeaf) {
      cSamplesLeafMin = static_cast<size_t>(minSamplesLeaf);
      if(IsConvertError<size_t>(minSamplesLeaf)) {
         // we can never exceed a size_t number of samples, so let's just set it to the maximum if we were going to overflow because it will generate 
         // the same results as if we used the true number
         cSamplesLeafMin = std::numeric_limits<size_t>::max();
      }
   } else {
      LOG_0(TraceLevelWarning, "WARNING GenerateTermUpdate minSamplesLeaf can't be less than 1.  Adjusting to 1.");
   }

   // leavesMax is handled in GenerateTermUpdateInternal

   // avgGainOut can be nullptr

   if(ptrdiff_t { 0 } == pBoosterCore->GetCountClasses() || ptrdiff_t { 1 } == pBoosterCore->GetCountClasses()) {
      // if there is only 1 target class for classification, then we can predict the output with 100% accuracy.  The term scores are a tensor with zero 
      // length array logits, which means for our representation that we have zero items in the array total.
      // since we can predit the output with 100% accuracy, our gain will be 0.
      if(LIKELY(nullptr != avgGainOut)) {
         *avgGainOut = double { 0 };
      }
      pBoosterShell->SetTermIndex(iTerm);

      LOG_0(
         TraceLevelWarning,
         "WARNING GenerateTermUpdate pBoosterCore->m_cClasses <= ptrdiff_t { 1 }"
      );
      return Error_None;
   }

   error = GenerateTermUpdateInternal(
      pBoosterShell,
      iTerm,
      options,
      learningRate,
      cSamplesLeafMin,
      leavesMax,
      avgGainOut
   );
   if(Error_None != error) {
      LOG_N(TraceLevelWarning, "WARNING GenerateTermUpdate: return=%" ErrorEbmTypePrintf, error);
      if(LIKELY(nullptr != avgGainOut)) {
         *avgGainOut = double { 0 };
      }
      return error;
   }

   if(nullptr != avgGainOut) {
      EBM_ASSERT(!std::isnan(*avgGainOut)); // NaNs can happen, but we should have edited those before here
      EBM_ASSERT(!std::isinf(*avgGainOut)); // infinities can happen, but we should have edited those before here
      // no epsilon required.  We make it zero if the value is less than zero for floating point instability reasons
      EBM_ASSERT(double { 0 } <= *avgGainOut);
      LOG_COUNTED_N(
         pBoosterCore->GetTerms()[iTerm]->GetPointerCountLogExitGenerateTermUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited GenerateTermUpdate: "
         "*avgGainOut=%le"
         ,
         *avgGainOut
      );
   } else {
      LOG_COUNTED_0(
         pBoosterCore->GetTerms()[iTerm]->GetPointerCountLogExitGenerateTermUpdateMessages(),
         TraceLevelInfo,
         TraceLevelVerbose,
         "Exited GenerateTermUpdate"
      );
   }
   return Error_None;
}

} // DEFINED_ZONE_NAME
