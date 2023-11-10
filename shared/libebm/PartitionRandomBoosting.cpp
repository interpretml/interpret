// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stddef.h> // size_t, ptrdiff_t
#include <algorithm> // sort

#include "libebm.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "unzoned.h" // LIKELY

#include "zones.h"
#include "GradientPair.hpp"
#include "Bin.hpp"

#include "RandomDeterministic.hpp"
#include "ebm_stats.hpp"
#include "Feature.hpp"
#include "Term.hpp"
#include "Tensor.hpp"
#include "BoosterCore.hpp"
#include "BoosterShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

template<bool bHessian, size_t cCompilerScores>
class PartitionRandomBoostingInternal final {
public:

   PartitionRandomBoostingInternal() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(
      RandomDeterministic * const pRng,
      BoosterShell * const pBoosterShell,
      const Term * const pTerm,
      const TermBoostFlags flags,
      const IntEbm * const aLeavesMax,
      double * const pTotalGain
   ) {
      // THIS RANDOM SPLIT FUNCTION IS PRIMARILY USED FOR DIFFERENTIAL PRIVACY EBMs

      // TODO: add a new random_rety option that will retry random splitting for N times and select the one with the best gain
      // TODO: accept the minimum number of items in a split and then refuse to allow the split if we violate it, or
      //       provide a soft trigger that generates 10 random ones and selects the one that violates the least
      //       maybe provide a flag to indicate if we want a hard or soft allowance.  We won't be splitting if we
      //       require a soft allowance and a lot of regions have zeros.
      // TODO: accept 0 == minSamplesLeaf as a minimum number of items so that we can always choose to allow a tensor split (for DP)
      // TODO: move most of this code out of this function into a non-templated place

      ErrorEbm error;
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();

      const size_t cScores = GET_COUNT_SCORES(cCompilerScores, GetCountScores(pBoosterCore->GetCountClasses()));
      const size_t cBytesPerBin = GetBinSize<FloatMain, UIntMain>(bHessian, cScores);

      auto * const aBins = pBoosterShell->GetBoostingMainBins()->Specialize<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)>();

      EBM_ASSERT(1 <= pTerm->GetCountRealDimensions());
      EBM_ASSERT(1 <= pTerm->GetCountDimensions());

      Tensor * const pInnerTermUpdate = pBoosterShell->GetInnerTermUpdate();

      const IntEbm * pLeavesMax1 = aLeavesMax;
      const TermFeature * pTermFeature1 = pTerm->GetTermFeatures();
      const TermFeature * const pTermFeaturesEnd = &pTermFeature1[pTerm->GetCountDimensions()];
      size_t cSlicesTotal = 0;
      size_t cSlicesPlusRandomMax = 0;
      size_t cCollapsedTensorCells = 1;
      do {
         size_t cLeavesMax;
         if(nullptr == pLeavesMax1) {
            cLeavesMax = size_t { 1 };
         } else {
            const IntEbm countLeavesMax = *pLeavesMax1;
            ++pLeavesMax1;
            if(countLeavesMax <= IntEbm { 1 }) {
               cLeavesMax = size_t { 1 };
            } else {
               cLeavesMax = static_cast<size_t>(countLeavesMax);
               if(IsConvertError<size_t>(countLeavesMax)) {
                  // we can never exceed a size_t number of leaves, so let's just set it to the maximum if we 
                  // were going to overflow because it will generate the same results as if we used the true number
                  cLeavesMax = std::numeric_limits<size_t>::max();
               }
            }
         }

         const FeatureBoosting * const pFeature = pTermFeature1->m_pFeature;
         const size_t cBins = pFeature->GetCountBins();
         EBM_ASSERT(size_t { 1 } <= cBins); // we don't boost on empty training sets
         const size_t cSlices = EbmMin(cLeavesMax, cBins);
         EBM_ASSERT(1 <= cSlices);

         const size_t cPossibleSplitLocations = cBins - size_t { 1 };
         if(size_t { 0 } != cPossibleSplitLocations) {
            // drop any dimensions with 1 bin since the tensor is the same without the extra dimension

            if(IsAddError(cSlicesTotal, cPossibleSplitLocations)) {
               LOG_0(Trace_Warning, "WARNING PartitionRandomBoostingInternal IsAddError(cSlicesTotal, cPossibleSplitLocations)");
               return Error_OutOfMemory;
            }
            const size_t cSlicesPlusRandom = cSlicesTotal + cPossibleSplitLocations;
            cSlicesPlusRandomMax = EbmMax(cSlicesPlusRandomMax, cSlicesPlusRandom);

            // our histogram is a tensor where we multiply the number of cells on each pass.  Addition of those 
            // same numbers can't be bigger than multiplication unless one of the dimensions is less than 2 wide.  
            // At 2, multiplication and addition would yield the same size.  All other numbers will be bigger for 
            // multiplication, so we can conclude that addition won't overflow since the multiplication didn't
            EBM_ASSERT(!IsAddError(cSlicesTotal, cSlices));
            cSlicesTotal += cSlices;

            EBM_ASSERT(!IsMultiplyError(cCollapsedTensorCells, cSlices)); // our allocated histogram is bigger
            cCollapsedTensorCells *= cSlices;
         }
         ++pTermFeature1;
      } while(pTermFeaturesEnd != pTermFeature1);

      // since we subtract 1 from cPossibleSplitLocations, we need to check that our final slice length isn't longer
      cSlicesPlusRandomMax = EbmMax(cSlicesPlusRandomMax, cSlicesTotal);

      if(IsMultiplyError(sizeof(size_t), cSlicesPlusRandomMax)) {
         LOG_0(Trace_Warning, "WARNING PartitionRandomBoostingInternal IsMultiplyError(sizeof(size_t), cSlicesPlusRandomMax)");
         return Error_OutOfMemory;
      }
      const size_t cBytesSlicesPlusRandom = sizeof(size_t) * cSlicesPlusRandomMax;

      error = pInnerTermUpdate->EnsureTensorScoreCapacity(cScores * cCollapsedTensorCells);
      if(UNLIKELY(Error_None != error)) {
         // already logged
         return error;
      }

      // our allocated histogram is bigger since it has more elements and the elements contain a size_t
      EBM_ASSERT(!IsMultiplyError(sizeof(size_t), cSlicesTotal));
      const size_t cBytesSlices = sizeof(size_t) * cSlicesTotal;

      // promote to bytes
      EBM_ASSERT(!IsMultiplyError(cBytesPerBin, cCollapsedTensorCells)); // our allocated histogram is bigger
      cCollapsedTensorCells *= cBytesPerBin;
      if(IsAddError(cBytesSlices, cCollapsedTensorCells)) {
         LOG_0(Trace_Warning, "WARNING PartitionRandomBoostingInternal IsAddError(cBytesSlices, cBytesCollapsedTensor1)");
         return Error_OutOfMemory;
      }

      // We previously handled conditions where a dimension had 0 bins in one of the features, so 
      // all dimensions should have had 1 bin and we set the number of leaves to 1 minimum, and
      // the cBytesPerBin value must be greater than 0, so cCollapsedTensorCells must be non-zero
      // The Clang static analyzer seems to not understand these things, so specify it here
      ANALYSIS_ASSERT(0 != cCollapsedTensorCells);

      const size_t cBytesSlicesAndCollapsedTensor = cBytesSlices + cCollapsedTensorCells;

      const size_t cBytesBuffer = EbmMax(cBytesSlicesAndCollapsedTensor, cBytesSlicesPlusRandom);

      // TODO: use GrowThreadByteBuffer2 for this, but first we need to change that to allocate void or bytes
      char * const pBuffer = static_cast<char *>(malloc(cBytesBuffer));
      if(UNLIKELY(nullptr == pBuffer)) {
         LOG_0(Trace_Warning, "WARNING PartitionRandomBoostingInternal nullptr == pBuffer");
         return Error_OutOfMemory;
      }
      size_t * const acItemsInNextSliceOrBytesInCurrentSlice = reinterpret_cast<size_t *>(pBuffer);

      const IntEbm * pLeavesMax2 = aLeavesMax;
      size_t * pcItemsInNextSliceOrBytesInCurrentSlice2 = acItemsInNextSliceOrBytesInCurrentSlice;
      const TermFeature * pTermFeature2 = pTerm->GetTermFeatures();
      do {
         size_t cTreeSplitsMax;
         if(nullptr == pLeavesMax2) {
            cTreeSplitsMax = size_t { 0 };
         } else {
            const IntEbm countLeavesMax = *pLeavesMax2;
            ++pLeavesMax2;
            if(countLeavesMax <= IntEbm { 1 }) {
               cTreeSplitsMax = size_t { 0 };
            } else {
               cTreeSplitsMax = static_cast<size_t>(countLeavesMax) - size_t { 1 };
               if(IsConvertError<size_t>(countLeavesMax)) {
                  // we can never exceed a size_t number of leaves, so let's just set it to the maximum if we 
                  // were going to overflow because it will generate the same results as if we used the true number
                  cTreeSplitsMax = std::numeric_limits<size_t>::max() - size_t { 1 };
               }
            }
         }

         const FeatureBoosting * const pFeature = pTermFeature2->m_pFeature;
         const size_t cBins = pFeature->GetCountBins();
         EBM_ASSERT(size_t { 1 } <= cBins); // we don't boost on empty training sets
         size_t cPossibleSplitLocations = cBins - size_t { 1 };
         if(size_t { 0 } != cPossibleSplitLocations) {
            // drop any dimensions with 1 bin since the tensor is the same without the extra dimension

            if(size_t { 0 } != cTreeSplitsMax) {
               size_t * pFillIndexes = pcItemsInNextSliceOrBytesInCurrentSlice2;
               size_t iEdge = cPossibleSplitLocations; // 1 means split between bin 0 and bin 1
               do {
                  *pFillIndexes = iEdge;
                  ++pFillIndexes;
                  --iEdge;
               } while(size_t { 0 } != iEdge);

               size_t * pOriginal = pcItemsInNextSliceOrBytesInCurrentSlice2;

               const size_t cSplits = EbmMin(cTreeSplitsMax, cPossibleSplitLocations);
               EBM_ASSERT(1 <= cSplits);
               const size_t * const pcItemsInNextSliceOrBytesInCurrentSliceEnd = pcItemsInNextSliceOrBytesInCurrentSlice2 + cSplits;
               do {
                  const size_t iRandom = pRng->NextFast(cPossibleSplitLocations);
                  size_t * const pRandomSwap = pcItemsInNextSliceOrBytesInCurrentSlice2 + iRandom;
                  const size_t temp = *pRandomSwap;
                  *pRandomSwap = *pcItemsInNextSliceOrBytesInCurrentSlice2;
                  *pcItemsInNextSliceOrBytesInCurrentSlice2 = temp;
                  --cPossibleSplitLocations;
                  ++pcItemsInNextSliceOrBytesInCurrentSlice2;
               } while(pcItemsInNextSliceOrBytesInCurrentSliceEnd != pcItemsInNextSliceOrBytesInCurrentSlice2);

               std::sort(pOriginal, pcItemsInNextSliceOrBytesInCurrentSlice2);
            }
            *pcItemsInNextSliceOrBytesInCurrentSlice2 = cBins; // index 1 past the last item
            ++pcItemsInNextSliceOrBytesInCurrentSlice2;
         }
         ++pTermFeature2;
      } while(pTermFeaturesEnd != pTermFeature2);

      const IntEbm * pLeavesMax3 = aLeavesMax;
      const size_t * pcBytesInSliceEnd;
      const TermFeature * pTermFeature3 = pTerm->GetTermFeatures();
      size_t * pcItemsInNextSliceOrBytesInCurrentSlice3 = acItemsInNextSliceOrBytesInCurrentSlice;
      size_t cBytesCollapsedTensor3;
      while(true) {
         EBM_ASSERT(pTermFeature3 < pTermFeaturesEnd);

         size_t cLeavesMax;
         if(nullptr == pLeavesMax3) {
            cLeavesMax = size_t { 1 };
         } else {
            const IntEbm countLeavesMax = *pLeavesMax3;
            ++pLeavesMax3;
            if(countLeavesMax <= IntEbm { 1 }) {
               cLeavesMax = size_t { 1 };
            } else {
               cLeavesMax = static_cast<size_t>(countLeavesMax);
               if(IsConvertError<size_t>(countLeavesMax)) {
                  // we can never exceed a size_t number of leaves, so let's just set it to the maximum if we 
                  // were going to overflow because it will generate the same results as if we used the true number
                  cLeavesMax = std::numeric_limits<size_t>::max();
               }
            }
         }

         // the first dimension is special.  we put byte until next item into it instead of counts remaining
         const FeatureBoosting * const pFeature = pTermFeature3->m_pFeature;
         ++pTermFeature3;
         const size_t cBins = pFeature->GetCountBins();
         EBM_ASSERT(size_t { 1 } <= cBins); // we don't boost on empty training sets
         if(size_t { 1 } < cBins) {
            // drop any dimensions with 1 bin since the tensor is the same without the extra dimension

            const size_t cFirstSlices = EbmMin(cLeavesMax, cBins);
            cBytesCollapsedTensor3 = cBytesPerBin * cFirstSlices;

            pcBytesInSliceEnd = acItemsInNextSliceOrBytesInCurrentSlice + cFirstSlices;
            size_t iPrev = size_t { 0 };
            do {
               // The Clang static analysis tool does not like our access here to the 
               // acItemsInNextSliceOrBytesInCurrentSlice buffer via the pcItemsInNextSliceOrBytesInCurrentSlice3
               // pointer. I think this is because we allocate the buffer to contain both the split information
               // and also the tensor information, and above this point we have only filled in the split information
               // which leaves the buffer only partly initialized as we use it here. If we put a 
               // memset(acItemsInNextSliceOrBytesInCurrentSlice, 0, cBytesBuffer) after allocation then
               // this analysis warning goes away which reinforces this suspicion.
               StopClangAnalysis();

               const size_t iCur = *pcItemsInNextSliceOrBytesInCurrentSlice3;
               EBM_ASSERT(iPrev < iCur);
               // turn these into bytes from the previous
               *pcItemsInNextSliceOrBytesInCurrentSlice3 = (iCur - iPrev) * cBytesPerBin;
               iPrev = iCur;
               ++pcItemsInNextSliceOrBytesInCurrentSlice3;
            } while(pcBytesInSliceEnd != pcItemsInNextSliceOrBytesInCurrentSlice3);

            // we found a non-eliminated dimension.  We treat the first dimension differently from others, so
            // if our first dimension is eliminated we need to keep looking until we find our first REAL dimension
            break;
         }
      }

      struct RandomSplitState {
         size_t         m_cItemsInSliceRemaining;
         size_t         m_cBytesSubtractResetCollapsedBin;

         const size_t * m_pcItemsInNextSlice;
         const size_t * m_pcItemsInNextSliceEnd;
      };
      RandomSplitState randomSplitState[k_cDimensionsMax - size_t { 1 }]; // the first dimension is special cased
      RandomSplitState * pStateInit = &randomSplitState[0];

      for(; pTermFeaturesEnd != pTermFeature3; ++pTermFeature3) {
         size_t cLeavesMax;
         if(nullptr == pLeavesMax3) {
            cLeavesMax = size_t { 1 };
         } else {
            const IntEbm countLeavesMax = *pLeavesMax3;
            ++pLeavesMax3;
            if(countLeavesMax <= IntEbm { 1 }) {
               cLeavesMax = size_t { 1 };
            } else {
               cLeavesMax = static_cast<size_t>(countLeavesMax);
               if(IsConvertError<size_t>(countLeavesMax)) {
                  // we can never exceed a size_t number of leaves, so let's just set it to the maximum if we 
                  // were going to overflow because it will generate the same results as if we used the true number
                  cLeavesMax = std::numeric_limits<size_t>::max();
               }
            }
         }

         const FeatureBoosting * const pFeature = pTermFeature3->m_pFeature;
         const size_t cBins = pFeature->GetCountBins();
         EBM_ASSERT(size_t { 1 } <= cBins); // we don't boost on empty training sets
         if(size_t { 1 } < cBins) {
            // drop any dimensions with 1 bin since the tensor is the same without the extra dimension

            size_t cSlices = EbmMin(cLeavesMax, cBins);

            pStateInit->m_cBytesSubtractResetCollapsedBin = cBytesCollapsedTensor3;

            EBM_ASSERT(!IsMultiplyError(cBytesCollapsedTensor3, cSlices)); // our allocated histogram is bigger
            cBytesCollapsedTensor3 *= cSlices;

            const size_t iFirst = *pcItemsInNextSliceOrBytesInCurrentSlice3;
            EBM_ASSERT(1 <= iFirst);
            pStateInit->m_cItemsInSliceRemaining = iFirst;
            pStateInit->m_pcItemsInNextSlice = pcItemsInNextSliceOrBytesInCurrentSlice3;

            size_t iPrev = iFirst;
            for(--cSlices; LIKELY(size_t { 0 } != cSlices); --cSlices) {
               size_t * const pCur = pcItemsInNextSliceOrBytesInCurrentSlice3 + size_t { 1 };
               const size_t iCur = *pCur;
               EBM_ASSERT(iPrev < iCur);
               *pcItemsInNextSliceOrBytesInCurrentSlice3 = iCur - iPrev;
               iPrev = iCur;
               pcItemsInNextSliceOrBytesInCurrentSlice3 = pCur;
            }
            *pcItemsInNextSliceOrBytesInCurrentSlice3 = iFirst;
            ++pcItemsInNextSliceOrBytesInCurrentSlice3;
            pStateInit->m_pcItemsInNextSliceEnd = pcItemsInNextSliceOrBytesInCurrentSlice3;
            ++pStateInit;
         }
      }

      // put the histograms right after our slice array
      auto * const aCollapsedBins =
         reinterpret_cast<Bin<FloatMain, UIntMain, bHessian, GetArrayScores(cCompilerScores)> *>(pcItemsInNextSliceOrBytesInCurrentSlice3);

      aCollapsedBins->ZeroMem(cBytesCollapsedTensor3);
      const auto * const pCollapsedBinEnd = IndexBin(aCollapsedBins, cBytesCollapsedTensor3);

      // we special case the first dimension, so drop it by subtracting
      EBM_ASSERT(&randomSplitState[pTerm->GetCountRealDimensions() - size_t { 1 }] == pStateInit);

      const auto * pBin = aBins;
      auto * pCollapsedBin1 = aCollapsedBins;

      {
      move_next_slice:;

         // for the first dimension, acItemsInNextSliceOrBytesInCurrentSlice contains the number of bytes to proceed 
         // until the next pBinSliceEnd point.  For the second dimension and higher, it contains a 
         // count of items for the NEXT slice.  The 0th element contains the count of items for the
         // 1st slice.  Yeah, it's pretty confusing, but it allows for some pretty compact code in this
         // super critical inner loop without overburdening the CPU registers when we execute the outer loop.
         const size_t * pcItemsInNextSliceOrBytesInCurrentSlice = acItemsInNextSliceOrBytesInCurrentSlice;
         do {
            const auto * const pBinSliceEnd = IndexBin(pBin, *pcItemsInNextSliceOrBytesInCurrentSlice);
            do {
               ASSERT_BIN_OK(cBytesPerBin, pBin, pBoosterShell->GetDebugMainBinsEnd());
               // TODO: add this first into a local Bin that can be put in registers then write it to
               // pCollapsedBin1 aferwards
               pCollapsedBin1->Add(cScores, *pBin);

               // we're walking through all bins, so just move to the next one in the flat array, 
               // with the knowledge that we'll figure out it's multi-dimenional index below
               pBin = IndexBin(pBin, cBytesPerBin);
            } while(LIKELY(pBinSliceEnd != pBin));

            pCollapsedBin1 = IndexBin(pCollapsedBin1, cBytesPerBin);

            ++pcItemsInNextSliceOrBytesInCurrentSlice;
         } while(PREDICTABLE(pcBytesInSliceEnd != pcItemsInNextSliceOrBytesInCurrentSlice));

         for(RandomSplitState * pState = randomSplitState; PREDICTABLE(pStateInit != pState); ++pState) {
            EBM_ASSERT(size_t { 1 } <= pState->m_cItemsInSliceRemaining);
            const size_t cItemsInSliceRemaining = pState->m_cItemsInSliceRemaining - size_t { 1 };
            if(LIKELY(size_t { 0 } != cItemsInSliceRemaining)) {
               // ideally, the compiler would move this to the location right above the first loop and it would
               // jump over it on the first loop, but I wasn't able to make the Visual Studio compiler do it

               pState->m_cItemsInSliceRemaining = cItemsInSliceRemaining;
               pCollapsedBin1 = NegativeIndexBin(pCollapsedBin1, pState->m_cBytesSubtractResetCollapsedBin);

               goto move_next_slice;
            }

            const size_t * pcItemsInNextSlice = pState->m_pcItemsInNextSlice;
            EBM_ASSERT(pcItemsInNextSliceOrBytesInCurrentSlice <= pcItemsInNextSlice);
            EBM_ASSERT(pcItemsInNextSlice < pState->m_pcItemsInNextSliceEnd);
            pState->m_cItemsInSliceRemaining = *pcItemsInNextSlice;
            ++pcItemsInNextSlice;
            // it would be legal for us to move this assignment into the if statement below, since if we don't
            // enter the if statement we overwrite the value that we just wrote, but writing it here allows the
            // compiler to emit a sinlge jne instruction to move to move_next_slice without using an extra
            // jmp instruction.  Typically we have 3 slices, so we avoid 2 jmp instructions for the cost of
            // 1 extra assignment that'll happen 1/3 of the time when m_pcItemsInNextSliceEnd == pcItemsInNextSlice
            // Something would have to be wrong for us to have less than 2 slices since then we'd be ignoring a
            // dimension, so even in the realistic worst case the 1 jmp instruction balances the extra mov instruction
            pState->m_pcItemsInNextSlice = pcItemsInNextSlice;
            if(UNPREDICTABLE(pState->m_pcItemsInNextSliceEnd != pcItemsInNextSlice)) {
               goto move_next_slice;
            }
            // the end of the previous dimension is the start of our current one
            pState->m_pcItemsInNextSlice = pcItemsInNextSliceOrBytesInCurrentSlice;
            pcItemsInNextSliceOrBytesInCurrentSlice = pcItemsInNextSlice;
         }
      }

      //TODO: retrieve the gain.  Always calculate the gain without respect to the parent and pick the best one
      //      Then, before exiting, on the last one we collapse the collapsed tensor even more into just a single
      //      bin from which we can calculate the parent and subtract the best child from the parent.
      
      //FloatCalc gain;
      //FloatCalc gainParent = 0;
      FloatCalc gain = 0;


      const TermFeature * pTermFeature4 = pTerm->GetTermFeatures();
      size_t iDimensionWrite = static_cast<size_t>(~size_t { 0 }); // this is -1, but without the compiler warning
      size_t cBinsWrite;
      do {
         const FeatureBoosting * const pFeature = pTermFeature4->m_pFeature;
         cBinsWrite = pFeature->GetCountBins();
         ++iDimensionWrite;
         ++pTermFeature4;
      } while(cBinsWrite <= size_t { 1 });

      EBM_ASSERT(acItemsInNextSliceOrBytesInCurrentSlice < pcBytesInSliceEnd);
      const size_t cFirstSlices = pcBytesInSliceEnd - acItemsInNextSliceOrBytesInCurrentSlice;
      // 3 items in the acItemsInNextSliceOrBytesInCurrentSlice means 2 splits and 
      // one last item to indicate the termination point
      error = pInnerTermUpdate->SetCountSlices(iDimensionWrite, cFirstSlices);
      if(UNLIKELY(Error_None != error)) {
         // already logged
         free(pBuffer);
         return error;
      }
      const size_t * pcBytesInSlice2 = acItemsInNextSliceOrBytesInCurrentSlice;
      if(LIKELY(size_t { 1 } < cFirstSlices)) {
         const size_t * const pcBytesInSliceLast = pcBytesInSliceEnd - size_t { 1 };
         UIntSplit * pSplitFirst = pInnerTermUpdate->GetSplitPointer(iDimensionWrite);
         size_t iEdgeFirst = 0;
         do {
            EBM_ASSERT(pcBytesInSlice2 < pcBytesInSliceLast);
            EBM_ASSERT(0 != *pcBytesInSlice2);
            EBM_ASSERT(0 == *pcBytesInSlice2 % cBytesPerBin);
            iEdgeFirst += *pcBytesInSlice2 / cBytesPerBin;
            // we checked earlier that countBins could be converted to a UIntSplit
            EBM_ASSERT(!IsConvertError<UIntSplit>(iEdgeFirst));
            *pSplitFirst = static_cast<UIntSplit>(iEdgeFirst);
            ++pSplitFirst;
            ++pcBytesInSlice2;
            // the last one is the distance to the end, which we don't include in the update
         } while(LIKELY(pcBytesInSliceLast != pcBytesInSlice2));
      }

      RandomSplitState * pState = randomSplitState;
      if(PREDICTABLE(pStateInit != pState)) {
         do {
            do {
               const FeatureBoosting * const pFeature = pTermFeature4->m_pFeature;
               cBinsWrite = pFeature->GetCountBins();
               ++iDimensionWrite;
               ++pTermFeature4;
            } while(cBinsWrite <= size_t { 1 });

            ++pcBytesInSlice2; // we have one less split than we have slices, so move to the next one

            const size_t * pcItemsInNextSliceEnd = pState->m_pcItemsInNextSliceEnd;
            error = pInnerTermUpdate->SetCountSlices(iDimensionWrite, pcItemsInNextSliceEnd - pcBytesInSlice2);
            if(Error_None != error) {
               // already logged
               free(pBuffer);
               return error;
            }
            const size_t * pcItemsInNextSliceLast = pcItemsInNextSliceEnd - size_t { 1 };
            if(pcItemsInNextSliceLast != pcBytesInSlice2) {
               UIntSplit * pSplit = pInnerTermUpdate->GetSplitPointer(iDimensionWrite);
               size_t iEdge2 = *pcItemsInNextSliceLast;
               // we checked earlier that countBins could be converted to a UIntSplit
               EBM_ASSERT(!IsConvertError<UIntSplit>(iEdge2));
               *pSplit = static_cast<UIntSplit>(iEdge2);
               --pcItemsInNextSliceLast;
               while(pcItemsInNextSliceLast != pcBytesInSlice2) {
                  iEdge2 += *pcBytesInSlice2;
                  ++pSplit;
                  // we checked earlier that countBins could be converted to a UIntSplit
                  EBM_ASSERT(!IsConvertError<UIntSplit>(iEdge2));
                  *pSplit = static_cast<UIntSplit>(iEdge2);
                  ++pcBytesInSlice2;
               }
               // increment it once more because our indexes are shifted such that the first one was the last item
               ++pcBytesInSlice2;
            }
            ++pState;
         } while(PREDICTABLE(pStateInit != pState));
      }

      FloatScore * pUpdateScore = pInnerTermUpdate->GetTensorScoresPointer();
      auto * pCollapsedBin2 = aCollapsedBins;

      if(0 != (TermBoostFlags_GradientSums & flags)) {
         do {
            auto * const pGradientPair = pCollapsedBin2->GetGradientPairs();

            for(size_t iScore = 0; iScore < cScores; ++iScore) {
               const FloatCalc updateScore = EbmStats::ComputeSinglePartitionUpdateGradientSum(static_cast<FloatCalc>(pGradientPair[iScore].m_sumGradients));
               *pUpdateScore = static_cast<FloatScore>(updateScore);
               ++pUpdateScore;
            }
            pCollapsedBin2 = IndexBin(pCollapsedBin2, cBytesPerBin);
         } while(pCollapsedBinEnd != pCollapsedBin2);
      } else {
         do {
            const auto cSamples = pCollapsedBin2->GetCountSamples();
            if(UNLIKELY(0 == cSamples)) {
               // TODO: this section can probably be eliminated since ComputeSinglePartitionUpdate now checks
               // for zero in the denominator, but I'm leaving it here to see how the removal of the 
               // GetCountSamples property works in the future in combination with the check on hessians

               // normally, we'd eliminate regions where the number of items was zero before putting down a split
               // but for random splits we can't know beforehand if there will be zero splits, so we need to check
               for(size_t iScore = 0; iScore < cScores; ++iScore) {
                  *pUpdateScore = 0;
                  ++pUpdateScore;
               }
            } else {
               auto * const pGradientPair = pCollapsedBin2->GetGradientPairs();
               for(size_t iScore = 0; iScore < cScores; ++iScore) {
                  FloatCalc updateScore;
                  if(bHessian) {
                     updateScore = EbmStats::ComputeSinglePartitionUpdate(
                        static_cast<FloatCalc>(pGradientPair[iScore].m_sumGradients),
                        static_cast<FloatCalc>(pGradientPair[iScore].GetHess())
                     );
                  } else {
                     updateScore = EbmStats::ComputeSinglePartitionUpdate(
                        static_cast<FloatCalc>(pGradientPair[iScore].m_sumGradients),
                        static_cast<FloatCalc>(pCollapsedBin2->GetWeight())
                     );
                  }
                  *pUpdateScore = static_cast<FloatScore>(updateScore);
                  ++pUpdateScore;
               }
            }
            pCollapsedBin2 = IndexBin(pCollapsedBin2, cBytesPerBin);
         } while(pCollapsedBinEnd != pCollapsedBin2);
      }

      free(pBuffer);
      *pTotalGain = static_cast<double>(gain);
      return Error_None;
   }
};

template<bool bHessian, size_t cPossibleScores>
class PartitionRandomBoostingTarget final {
public:

   PartitionRandomBoostingTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(
      RandomDeterministic * const pRng,
      BoosterShell * const pBoosterShell,
      const Term * const pTerm,
      const TermBoostFlags flags,
      const IntEbm * const aLeavesMax,
      double * const pTotalGain
   ) {
      BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
      if(cPossibleScores == GetCountScores(pBoosterCore->GetCountClasses())) {
         return PartitionRandomBoostingInternal<bHessian, cPossibleScores>::Func(
            pRng,
            pBoosterShell,
            pTerm,
            flags,
            aLeavesMax,
            pTotalGain
         );
      } else {
         return PartitionRandomBoostingTarget<bHessian, cPossibleScores + 1>::Func(
            pRng,
            pBoosterShell,
            pTerm,
            flags,
            aLeavesMax,
            pTotalGain
         );
      }
   }
};

template<bool bHessian>
class PartitionRandomBoostingTarget<bHessian, k_cCompilerScoresMax + 1> final {
public:

   PartitionRandomBoostingTarget() = delete; // this is a static class.  Do not construct

   INLINE_RELEASE_UNTEMPLATED static ErrorEbm Func(
      RandomDeterministic * const pRng,
      BoosterShell * const pBoosterShell,
      const Term * const pTerm,
      const TermBoostFlags flags,
      const IntEbm * const aLeavesMax,
      double * const pTotalGain
   ) {
      return PartitionRandomBoostingInternal<bHessian, k_dynamicScores>::Func(
         pRng,
         pBoosterShell,
         pTerm,
         flags,
         aLeavesMax,
         pTotalGain
      );
   }
};

extern ErrorEbm PartitionRandomBoosting(
   RandomDeterministic * const pRng,
   BoosterShell * const pBoosterShell,
   const Term * const pTerm,
   const TermBoostFlags flags,
   const IntEbm * const aLeavesMax,
   double * const pTotalGain
) {
   BoosterCore * const pBoosterCore = pBoosterShell->GetBoosterCore();
   const size_t cRuntimeScores = GetCountScores(pBoosterCore->GetCountClasses());

   EBM_ASSERT(1 <= cRuntimeScores);
   if(pBoosterCore->IsHessian()) {
      if(size_t { 1 } != cRuntimeScores) {
         // muticlass
         return PartitionRandomBoostingTarget<true, k_cCompilerScoresStart>::Func(
            pRng,
            pBoosterShell,
            pTerm,
            flags,
            aLeavesMax,
            pTotalGain
         );
      } else {
         return PartitionRandomBoostingInternal<true, k_oneScore>::Func(
            pRng,
            pBoosterShell,
            pTerm,
            flags,
            aLeavesMax,
            pTotalGain
         );
      }
   } else {
      if(size_t { 1 } != cRuntimeScores) {
         // Odd: gradient multiclass. Allow it, but do not optimize for it
         return PartitionRandomBoostingInternal<false, k_dynamicScores>::Func(
            pRng,
            pBoosterShell,
            pTerm,
            flags,
            aLeavesMax,
            pTotalGain
         );
      } else {
         return PartitionRandomBoostingInternal<false, k_oneScore>::Func(
            pRng,
            pBoosterShell,
            pTerm,
            flags,
            aLeavesMax,
            pTotalGain
         );
      }
   }
}

} // DEFINED_ZONE_NAME
