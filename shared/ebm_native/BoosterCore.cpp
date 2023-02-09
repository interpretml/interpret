// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <thread>


#include "logging.h" // EBM_ASSERT

#include "common_cpp.hpp" // IsConvertError, IsMultiplyError
#include "ebm_internal.hpp" // AddPositiveFloatsSafeBig

#include "dataset_shared.hpp" // GetDataSetSharedHeader
#include "Tensor.hpp" // Tensor
#include "Feature.hpp" // Feature
#include "Term.hpp" // Term
#include "InnerBag.hpp" // InnerBag

#include "Bin.hpp" // IsOverflowBinSize

#include "TreeNode.hpp" // IsOverflowTreeNodeSize
#include "SplitPosition.hpp" // IsOverflowSplitPositionSize

#include "BoosterCore.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class RandomDeterministic;

extern ErrorEbm ApplyUpdate(ApplyUpdateBridge * const pData);

extern ErrorEbm Unbag(
   const size_t cSamples,
   const BagEbm * const aBag,
   size_t * const pcTrainingSamplesOut,
   size_t * const pcValidationSamplesOut
);

extern ErrorEbm ExtractWeights(
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const BagEbm * const aBag,
   const size_t cSetSamples,
   FloatFast ** ppWeightsOut
);

void BoosterCore::DeleteTensors(const size_t cTerms, Tensor ** const apTensors) {
   LOG_0(Trace_Info, "Entered DeleteTensors");

   if(UNLIKELY(nullptr != apTensors)) {
      EBM_ASSERT(0 < cTerms);
      Tensor ** ppTensor = apTensors;
      const Tensor * const * const ppTensorsEnd = &apTensors[cTerms];
      do {
         Tensor::Free(*ppTensor);
         ++ppTensor;
      } while(ppTensorsEnd != ppTensor);
      free(apTensors);
   }
   LOG_0(Trace_Info, "Exited DeleteTensors");
}

ErrorEbm BoosterCore::InitializeTensors(
   const size_t cTerms, 
   const Term * const * const apTerms, 
   const size_t cScores,
   Tensor *** papTensorsOut)
{
   LOG_0(Trace_Info, "Entered InitializeTensors");

   EBM_ASSERT(1 <= cTerms);
   EBM_ASSERT(nullptr != apTerms);
   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(nullptr != papTensorsOut);
   EBM_ASSERT(nullptr == *papTensorsOut);

   ErrorEbm error;

   if(IsMultiplyError(sizeof(Tensor *), cTerms)) {
      LOG_0(Trace_Warning, "WARNING InitializeTensors IsMultiplyError(sizeof(Tensor *), cTerms)");
      return Error_OutOfMemory;
   }
   Tensor ** const apTensors = static_cast<Tensor **>(malloc(sizeof(Tensor *) * cTerms));
   if(UNLIKELY(nullptr == apTensors)) {
      LOG_0(Trace_Warning, "WARNING InitializeTensors nullptr == apTensors");
      return Error_OutOfMemory;
   }

   Tensor ** ppTensorInit = apTensors;
   const Tensor * const * const ppTensorsEnd = &apTensors[cTerms];
   do {
      *ppTensorInit = nullptr;
      ++ppTensorInit;
   } while(ppTensorsEnd != ppTensorInit);
   *papTensorsOut = apTensors; // transfer ownership for future deletion

   Tensor ** ppTensor = apTensors;
   const Term * const * ppTerm = apTerms;
   do {
      const Term * const pTerm = *ppTerm;
      if(size_t { 0 } != pTerm->GetCountTensorBins()) {
         // if there are any dimensions with features having 0 bins then do not allocate the tensor
         // since it will have 0 scores

         Tensor * const pTensors = Tensor::Allocate(pTerm->GetCountDimensions(), cScores);
         if(UNLIKELY(nullptr == pTensors)) {
            LOG_0(Trace_Warning, "WARNING InitializeTensors nullptr == pTensors");
            return Error_OutOfMemory;
         }
         *ppTensor = pTensors; // transfer ownership for future deletion

         error = pTensors->Expand(pTerm);
         if(Error_None != error) {
            // already logged
            return error;
         }
      }
      ++ppTerm;
      ++ppTensor;
   } while(ppTensorsEnd != ppTensor);

   LOG_0(Trace_Info, "Exited InitializeTensors");
   return Error_None;
}

BoosterCore::~BoosterCore() {
   // this only gets called after our reference count has been decremented to zero

   m_trainingSet.Destruct();
   m_validationSet.Destruct();

   InnerBag::FreeInnerBags(m_cInnerBags, m_apInnerBags);
   free(m_aValidationWeights);

   Term::FreeTerms(m_cTerms, m_apTerms);

   free(m_aFeatures);

   DeleteTensors(m_cTerms, m_apCurrentTermTensors);
   DeleteTensors(m_cTerms, m_apBestTermTensors);
};

void BoosterCore::Free(BoosterCore * const pBoosterCore) {
   LOG_0(Trace_Info, "Entered BoosterCore::Free");
   if(nullptr != pBoosterCore) {
      // for reference counting in general, a release is needed during the decrement and aquire is needed if freeing
      // https://www.boost.org/doc/libs/1_59_0/doc/html/atomic/usage_examples.html
      // We need to ensure that writes on this thread are not allowed to be re-ordered to a point below the 
      // decrement because if we happened to decrement to 2, and then get interrupted, and annother thread
      // decremented to 1 after us, we don't want our unclean writes to memory to be visible in the other thread
      // so we use memory_order_release on the decrement.
      if(size_t { 1 } == pBoosterCore->m_REFERENCE_COUNT.fetch_sub(1, std::memory_order_release)) {
         // we need to ensure that reads on this thread do not get reordered to a point before the decrement, otherwise
         // another thread might write some information, write the decrement to 2, then our thread decrements to 1
         // and then if we're allowed to read from data that occured before our decrement to 1 then we could have
         // stale data from before the other thread decrementing.  If our thread isn't freeing the memory though
         // we don't have to worry about staleness, so only use memory_order_acquire if we're going to delete the
         // object
         std::atomic_thread_fence(std::memory_order_acquire);
         LOG_0(Trace_Info, "INFO BoosterCore::Free deleting BoosterCore");
         delete pBoosterCore;
      }
   }
   LOG_0(Trace_Info, "Exited BoosterCore::Free");
}

//static int g_TODO_removeThisThreadTest = 0;
//void TODO_removeThisThreadTest() {
//   g_TODO_removeThisThreadTest = 1;
//}

ErrorEbm BoosterCore::Create(
   void * const rng,
   const size_t cTerms,
   const size_t cInnerBags,
   const double * const experimentalParams,
   const IntEbm * const acTermDimensions,
   const IntEbm * const aiTermFeatures, 
   const unsigned char * const pDataSetShared,
   const BagEbm * const aBag,
   const double * const aInitScores,
   BoosterCore ** const ppBoosterCoreOut
) {
   // experimentalParams isn't used by default.  It's meant to provide an easy way for python or other higher
   // level languages to pass EXPERIMENTAL temporary parameters easily to the C++ code.
   UNUSED(experimentalParams);

   LOG_0(Trace_Info, "Entered BoosterCore::Create");

   EBM_ASSERT(nullptr != ppBoosterCoreOut);
   EBM_ASSERT(nullptr == *ppBoosterCoreOut);
   EBM_ASSERT(nullptr != pDataSetShared);

   ErrorEbm error;

   //try {
   //   // TODO: eliminate this code I added to test that threads are available on the majority of our systems
   //   std::thread testThread(TODO_removeThisThreadTest);
   //   testThread.join();
   //   if(0 == g_TODO_removeThisThreadTest) {
   //      LOG_0(Trace_Warning, "WARNING BoosterCore::Create thread not started");
   //      return Error_UnexpectedInternal;
   //   }
   //} catch(const std::bad_alloc &) {
   //   LOG_0(Trace_Warning, "WARNING BoosterCore::Create thread start out of memory");
   //   return Error_OutOfMemory;
   //} catch(...) {
   //   // the C++ standard doesn't really seem to say what kind of exceptions we'd get for various errors, so
   //   // about the best we can do is catch(...) since the exact exceptions seem to be implementation specific
   //   LOG_0(Trace_Warning, "WARNING BoosterCore::Create thread start failed");
   //   return Error_ThreadStartFailed;
   //}
   //LOG_0(Trace_Info, "INFO BoosterCore::Create thread started");

   BoosterCore * pBoosterCore;
   try {
      pBoosterCore = new BoosterCore();
   } catch(const std::bad_alloc &) {
      LOG_0(Trace_Warning, "WARNING BoosterCore::Create Out of memory allocating BoosterCore");
      return Error_OutOfMemory;
   } catch(...) {
      LOG_0(Trace_Warning, "WARNING BoosterCore::Create Unknown error");
      return Error_UnexpectedInternal;
   }
   if(nullptr == pBoosterCore) {
      // this should be impossible since bad_alloc should have been thrown, but let's be untrusting
      LOG_0(Trace_Warning, "WARNING BoosterCore::Create nullptr == pBoosterCore");
      return Error_OutOfMemory;
   }
   // give ownership of our object back to the caller, even if there is a failure
   *ppBoosterCoreOut = pBoosterCore;

   SharedStorageDataType countSamples;
   size_t cFeatures;
   size_t cWeights;
   size_t cTargets;
   error = GetDataSetSharedHeader(pDataSetShared, &countSamples, &cFeatures, &cWeights, &cTargets);
   if(Error_None != error) {
      // already logged
      return error;
   }

   if(IsConvertError<size_t>(countSamples)) {
      LOG_0(Trace_Error, "ERROR BoosterCore::Create IsConvertError<size_t>(countSamples)");
      return Error_IllegalParamVal;
   }
   size_t cSamples = static_cast<size_t>(countSamples);

   if(size_t { 1 } < cWeights) {
      LOG_0(Trace_Warning, "WARNING BoosterCore::Create size_t { 1 } < cWeights");
      return Error_IllegalParamVal;
   }
   if(size_t { 1 } != cTargets) {
      LOG_0(Trace_Warning, "WARNING BoosterCore::Create 1 != cTargets");
      return Error_IllegalParamVal;
   }

   ptrdiff_t cClasses;
   if(nullptr == GetDataSetSharedTarget(pDataSetShared, 0, &cClasses)) {
      LOG_0(Trace_Warning, "WARNING BoosterCore::Create cClasses cannot fit into ptrdiff_t");
      return Error_IllegalParamVal;
   }

   size_t cTrainingSamples;
   size_t cValidationSamples;
   error = Unbag(cSamples, aBag, &cTrainingSamples, &cValidationSamples);
   if(Error_None != error) {
      // already logged
      return error;
   }

   LOG_0(Trace_Info, "BoosterCore::Create starting feature processing");
   if(0 != cFeatures) {
      pBoosterCore->m_cFeatures = cFeatures;

      if(IsMultiplyError(sizeof(FeatureBoosting), cFeatures)) {
         LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsMultiplyError(sizeof(Feature), cFeatures)");
         return Error_OutOfMemory;
      }
      FeatureBoosting * const aFeatures = static_cast<FeatureBoosting *>(malloc(sizeof(FeatureBoosting) * cFeatures));
      if(nullptr == aFeatures) {
         LOG_0(Trace_Warning, "WARNING BoosterCore::Create nullptr == aFeatures");
         return Error_OutOfMemory;
      }
      pBoosterCore->m_aFeatures = aFeatures;

      size_t iFeatureInitialize = size_t { 0 };
      do {
         bool bMissing;
         bool bUnknown;
         bool bNominal;
         bool bSparse;
         SharedStorageDataType countBins;
         SharedStorageDataType defaultValSparse;
         size_t cNonDefaultsSparse;
         GetDataSetSharedFeature(
            pDataSetShared,
            iFeatureInitialize,
            &bMissing,
            &bUnknown,
            &bNominal,
            &bSparse,
            &countBins,
            &defaultValSparse,
            &cNonDefaultsSparse
         );
         EBM_ASSERT(!bSparse); // we do not handle yet
         if(IsConvertError<size_t>(countBins)) {
            LOG_0(Trace_Error, "ERROR BoosterCore::Create IsConvertError<size_t>(countBins)");
            return Error_IllegalParamVal;
         }
         const size_t cBins = static_cast<size_t>(countBins);
         if(0 == cBins) {
            if(0 != cSamples) {
               LOG_0(Trace_Error, "ERROR BoosterCore::Create countBins cannot be zero if either 0 < cTrainingSamples OR 0 < cValidationSamples");
               return Error_IllegalParamVal;
            }

            // we can handle 0 == cBins even though that's a degenerate case that shouldn't be boosted on.  0 bins
            // can only occur if there were zero training and zero validation cases since the 
            // features would require a value, even if it was 0.
            LOG_0(Trace_Info, "INFO BoosterCore::Create feature with 0 values");
         } else if(1 == cBins) {
            // Dimensions with 1 bin don't contribute anything to the model since they always have the same value, but 
            // the user can specify interactions, so we handle them anyways in a consistent way by boosting on them
            LOG_0(Trace_Info, "INFO BoosterCore::Create feature with 1 value");
         }
         aFeatures[iFeatureInitialize].Initialize(cBins, bMissing, bUnknown, bNominal);

         ++iFeatureInitialize;
      } while(cFeatures != iFeatureInitialize);
   }
   LOG_0(Trace_Info, "BoosterCore::Create done feature processing");

   const bool bClassification = IsClassification(cClasses);

   EBM_ASSERT(nullptr == pBoosterCore->m_apCurrentTermTensors);
   EBM_ASSERT(nullptr == pBoosterCore->m_apBestTermTensors);

   LOG_0(Trace_Info, "BoosterCore::Create starting feature group processing");
   if(0 != cTerms) {
      pBoosterCore->m_cTerms = cTerms;
      pBoosterCore->m_apTerms = Term::AllocateTerms(cTerms);
      if(UNLIKELY(nullptr == pBoosterCore->m_apTerms)) {
         LOG_0(Trace_Warning, "WARNING BoosterCore::Create 0 != m_cTerms && nullptr == m_apTerms");
         return Error_OutOfMemory;
      }

      size_t cFastBinsMax = 0;
      size_t cBigBinsMax = 0;
      size_t cSingleDimensionBinsMax = 0;

      const IntEbm * piTermFeature = aiTermFeatures;
      size_t iTerm = 0;
      do {
         const IntEbm countDimensions = acTermDimensions[iTerm];
         if(countDimensions < IntEbm { 0 }) {
            LOG_0(Trace_Error, "ERROR BoosterCore::Create countDimensions cannot be negative");
            return Error_IllegalParamVal;
         }
         if(IntEbm { k_cDimensionsMax } < countDimensions) {
            LOG_0(Trace_Error, "WARNING BoosterCore::Create countDimensions too large and would cause out of memory condition");
            return Error_OutOfMemory;
         }
         const size_t cDimensions = static_cast<size_t>(countDimensions);
         Term * const pTerm = Term::Allocate(cDimensions);
         if(nullptr == pTerm) {
            LOG_0(Trace_Warning, "WARNING BoosterCore::Create nullptr == pTerm");
            return Error_OutOfMemory;
         }
         // assign our pointer directly to our array right now so that we can't loose the memory if we decide to exit due to an error below
         pBoosterCore->m_apTerms[iTerm] = pTerm;

         pTerm->SetCountAuxillaryBins(0); // we only use these for pairs, so otherwise it gets left as zero

         size_t cAuxillaryBinsForBuildFastTotals = 0;
         size_t cRealDimensions = 0;
         ptrdiff_t cItemsPerBitPack = k_cItemsPerBitPackNone;
         size_t cTensorBins = 1;
         if(UNLIKELY(0 == cDimensions)) {
            LOG_0(Trace_Info, "INFO BoosterCore::Create empty feature group");

            cFastBinsMax = EbmMax(cFastBinsMax, size_t { 1 });
            cBigBinsMax = EbmMax(cBigBinsMax, size_t { 1 });
         } else {
            if(nullptr == piTermFeature) {
               LOG_0(Trace_Error, "ERROR BoosterCore::Create aiTermFeatures is null when there are Terms with non-zero numbers of features");
               return Error_IllegalParamVal;
            }
            size_t cSingleDimensionBins = 0;
            const FeatureBoosting ** ppFeature = pTerm->GetFeatures();
            const FeatureBoosting * const * const ppFeaturesEnd = &ppFeature[cDimensions];
            do {
               const IntEbm indexFeature = *piTermFeature;
               if(indexFeature < 0) {
                  LOG_0(Trace_Error, "ERROR BoosterCore::Create aiTermFeatures value cannot be negative");
                  return Error_IllegalParamVal;
               }
               if(IsConvertError<size_t>(indexFeature)) {
                  LOG_0(Trace_Error, "ERROR BoosterCore::Create aiTermFeatures value too big to reference memory");
                  return Error_IllegalParamVal;
               }
               const size_t iFeature = static_cast<size_t>(indexFeature);

               if(cFeatures <= iFeature) {
                  LOG_0(Trace_Error, "ERROR BoosterCore::Create aiTermFeatures value must be less than the number of features");
                  return Error_IllegalParamVal;
               }

               EBM_ASSERT(1 <= cFeatures); // since our iFeature is valid and index 0 would mean cFeatures == 1
               EBM_ASSERT(nullptr != pBoosterCore->m_aFeatures);

               const FeatureBoosting * const pInputFeature = &pBoosterCore->m_aFeatures[iFeature];
               *ppFeature = pInputFeature;

               const size_t cBins = pInputFeature->GetCountBins();
               if(LIKELY(size_t { 1 } < cBins)) {
                  // if we have only 1 bin, then we can eliminate the feature from consideration since the resulting tensor loses one dimension but is 
                  // otherwise indistinquishable from the original data
                  ++cRealDimensions;

                  cSingleDimensionBins = cBins;
                  
                  if(IsMultiplyError(cTensorBins, cBins)) {
                     // if this overflows, we definetly won't be able to allocate it
                     LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsMultiplyError(cTensorStates, cBins)");
                     return Error_OutOfMemory;
                  }

                  // mathematically, cTensorBins grows faster than cAuxillaryBinsForBuildFastTotals
                  EBM_ASSERT(0 == cTensorBins || cAuxillaryBinsForBuildFastTotals < cTensorBins);

                  // since cBins must be 2 or more, cAuxillaryBinsForBuildFastTotals must grow slower than 
                  // cTensorBins, and we checked above that cTensorBins would not overflow
                  EBM_ASSERT(!IsAddError(cAuxillaryBinsForBuildFastTotals, cTensorBins));

                  cAuxillaryBinsForBuildFastTotals += cTensorBins;
               } else {
                  LOG_0(Trace_Info, "INFO BoosterCore::Create feature group with no useful features");
               }
               cTensorBins *= cBins;
               // same reasoning as above: cAuxillaryBinsForBuildFastTotals grows slower than cTensorBins
               EBM_ASSERT(0 == cTensorBins || cAuxillaryBinsForBuildFastTotals < cTensorBins);

               ++piTermFeature;
               ++ppFeature;
            } while(ppFeaturesEnd != ppFeature);

            if(LIKELY(size_t { 0 } != cTensorBins)) {
               cFastBinsMax = EbmMax(cFastBinsMax, cTensorBins);

               size_t cTotalBigBins = cTensorBins;
               if(LIKELY(size_t { 1 } != cTensorBins)) {
                  EBM_ASSERT(1 <= cRealDimensions);

                  const size_t iTensorBinMax = cTensorBins - size_t { 1 };

                  if(IsConvertError<StorageDataType>(iTensorBinMax)) {
                     LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsConvertError<StorageDataType>(iTensorBinMax)");
                     return Error_OutOfMemory;
                  }

                  const size_t cBitsRequiredMin = CountBitsRequired(iTensorBinMax);
                  EBM_ASSERT(1 <= cBitsRequiredMin); // 1 < cTensorBins otherwise we'd have filtered it out above
                  EBM_ASSERT(cBitsRequiredMin <= k_cBitsForSizeT);
                  EBM_ASSERT(cBitsRequiredMin <= k_cBitsForStorageType);

                  cItemsPerBitPack = static_cast<ptrdiff_t>(GetCountItemsBitPacked<StorageDataType>(cBitsRequiredMin));
                  EBM_ASSERT(ptrdiff_t { 1 } <= cItemsPerBitPack);
                  EBM_ASSERT(cItemsPerBitPack <= ptrdiff_t { k_cBitsForStorageType });

                  if(size_t { 1 } == cRealDimensions) {
                     cSingleDimensionBinsMax = EbmMax(cSingleDimensionBinsMax, cSingleDimensionBins);
                  } else {
                     // we only use AuxillaryBins for pairs.  We wouldn't use them for random pairs, but we
                     // don't know yet if the caller will set the random boosting flag on all pairs, so allocate it

                     // we need to reserve 4 PAST the pointer we pass into SweepMultiDimensional!!!!.  We pass in index 20 at max, so we need 24
                     static constexpr size_t cAuxillaryBinsForSplitting = 24;
                     const size_t cAuxillaryBins = EbmMax(cAuxillaryBinsForBuildFastTotals, cAuxillaryBinsForSplitting);
                     pTerm->SetCountAuxillaryBins(cAuxillaryBins);

                     if(IsAddError(cTensorBins, cAuxillaryBins)) {
                        LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsAddError(cTensorBins, cAuxillaryBins)");
                        return Error_OutOfMemory;
                     }
                     cTotalBigBins += cAuxillaryBins;
                  }
               } else {
                  EBM_ASSERT(0 == cRealDimensions);
               }
               cBigBinsMax = EbmMax(cBigBinsMax, cTotalBigBins);
            }
         }
         pTerm->SetCountRealDimensions(cRealDimensions);
         pTerm->SetBitPack(cItemsPerBitPack);
         pTerm->SetCountTensorBins(cTensorBins);

         ++iTerm;
      } while(iTerm < cTerms);

      if(ptrdiff_t { 0 } != cClasses && ptrdiff_t { 1 } != cClasses) {
         const size_t cScores = GetCountScores(cClasses);

         if(IsOverflowBinSize<FloatFast>(bClassification, cScores) || 
            IsOverflowBinSize<FloatBig>(bClassification, cScores))
         {
            LOG_0(Trace_Warning, "WARNING BoosterCore::Create bin size overflow");
            return Error_OutOfMemory;
         }

         const size_t cBytesPerFastBin = GetBinSize<FloatFast>(bClassification, cScores);
         if(IsMultiplyError(cBytesPerFastBin, cFastBinsMax)) {
            LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsMultiplyError(cBytesPerFastBin, cFastBinsMax)");
            return Error_OutOfMemory;
         }
         pBoosterCore->m_cBytesFastBins = cBytesPerFastBin * cFastBinsMax;

         const size_t cBytesPerBigBin = GetBinSize<FloatBig>(bClassification, cScores);
         if(IsMultiplyError(cBytesPerBigBin, cBigBinsMax)) {
            LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsMultiplyError(cBytesPerBigBin, cBigBinsMax)");
            return Error_OutOfMemory;
         }
         pBoosterCore->m_cBytesBigBins = cBytesPerBigBin * cBigBinsMax;

         if(0 != cSingleDimensionBinsMax) {
            if(IsOverflowTreeNodeSize(bClassification, cScores) ||
               IsOverflowSplitPositionSize(bClassification, cScores)) 
            {
               LOG_0(Trace_Warning, "WARNING BoosterCore::Create bin tracking size overflow");
               return Error_OutOfMemory;
            }

            const size_t cSingleDimensionSplitsMax = cSingleDimensionBinsMax - 1;
            const size_t cBytesPerSplitPosition = GetSplitPositionSize(bClassification, cScores);
            if(IsMultiplyError(cBytesPerSplitPosition, cSingleDimensionSplitsMax)) {
               LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsMultiplyError(cBytesPerSplitPosition, cSingleDimensionSplitsMax)");
               return Error_OutOfMemory;
            }
            // TODO : someday add equal gain multidimensional randomized picking.  I think for that we should generate
            //        random numbers as we find equal gains, so we won't need this memory if we do that
            pBoosterCore->m_cBytesSplitPositions = cBytesPerSplitPosition * cSingleDimensionSplitsMax;


            // If we have N bins, then we can have at most N - 1 splits.
            // At maximum if all splits are made, then we'll have a tree with N - 1 nodes.
            // Each node will contain a the total gradient sums of their left and right sides
            // Each of the N bins will also have a leaf in the tree, which will also consume a TreeNode structure
            // because each split needs to preserve the gradient sums of its left and right sides, which in this
            // case are individual bins.
            // So, in total we consume N + N - 1 TreeNodes
         
            if(IsAddError(cSingleDimensionSplitsMax, cSingleDimensionBinsMax)) {
               LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsAddError(cSingleDimensionSplitsMax, cSingleDimensionBinsMax)");
               return Error_OutOfMemory;
            }
            const size_t cTreeNodes = cSingleDimensionSplitsMax + cSingleDimensionBinsMax;

            const size_t cBytesPerTreeNode = GetTreeNodeSize(bClassification, cScores);
            if(IsMultiplyError(cBytesPerTreeNode, cTreeNodes)) {
               LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsMultiplyError(cBytesPerTreeNode, cTreeNodes)");
               return Error_OutOfMemory;
            }
            pBoosterCore->m_cBytesTreeNodes = cTreeNodes * cBytesPerTreeNode;
         } else {
            EBM_ASSERT(0 == pBoosterCore->m_cBytesSplitPositions);
            EBM_ASSERT(0 == pBoosterCore->m_cBytesTreeNodes);
         }

         error = InitializeTensors(cTerms, pBoosterCore->m_apTerms, cScores, &pBoosterCore->m_apCurrentTermTensors);
         if(Error_None != error) {
            LOG_0(Trace_Warning, "WARNING BoosterCore::Create nullptr == m_apCurrentTermTensors");
            return error;
         }
         error = InitializeTensors(cTerms, pBoosterCore->m_apTerms, cScores, &pBoosterCore->m_apBestTermTensors);
         if(Error_None != error) {
            LOG_0(Trace_Warning, "WARNING BoosterCore::Create nullptr == m_apBestTermTensors");
            return error;
         }
      }
   }
   LOG_0(Trace_Info, "BoosterCore::Create finished feature group processing");

   error = pBoosterCore->m_trainingSet.Initialize(
      cClasses,
      ptrdiff_t { 0 } != cClasses && ptrdiff_t { 1 } != cClasses, // regression, binary, multiclass
      ptrdiff_t { 1 } < cClasses, // binary, multiclass
      ptrdiff_t { 1 } < cClasses, // binary, multiclass
      bClassification,
      pDataSetShared,
      cSamples,
      BagEbm { 1 },
      aBag,
      aInitScores,
      cTrainingSamples,
      aiTermFeatures,
      cTerms,
      pBoosterCore->m_apTerms
   );
   if(Error_None != error) {
      LOG_0(Trace_Warning, "WARNING BoosterCore::Create m_trainingSet.Initialize");
      return error;
   }

   error = pBoosterCore->m_validationSet.Initialize(
      cClasses,
      !bClassification,
      false,
      ptrdiff_t { 1 } < cClasses, // binary, multiclass
      bClassification,
      pDataSetShared,
      cSamples,
      BagEbm { -1 },
      aBag,
      aInitScores,
      cValidationSamples,
      aiTermFeatures,
      cTerms,
      pBoosterCore->m_apTerms
   );
   if(Error_None != error) {
      LOG_0(Trace_Warning, "WARNING BoosterCore::Create m_validationSet.Initialize");
      return error;
   }

   EBM_ASSERT(nullptr == pBoosterCore->m_apInnerBags);
   if(0 != cTrainingSamples) {
      FloatFast * aWeights = nullptr;
      if(0 != cWeights) {
         error = ExtractWeights(
            pDataSetShared,
            BagEbm { 1 },
            aBag, 
            cTrainingSamples,
            &aWeights
         );
         if(Error_None != error) {
            // error already logged
            return error;
         }
      }
      pBoosterCore->m_cInnerBags = cInnerBags;
      // TODO: we could steal the aWeights in GenerateInnerBags for flat sampling sets
      error = InnerBag::GenerateInnerBags(
         rng,
         pBoosterCore->m_trainingSet.GetCountSamples(), 
         aWeights, 
         cInnerBags,
         &pBoosterCore->m_apInnerBags
      );
      free(aWeights);
      if(UNLIKELY(Error_None != error)) {
         // already logged
         return error;
      }
   }

   EBM_ASSERT(nullptr == pBoosterCore->m_aValidationWeights);
   pBoosterCore->m_validationWeightTotal = static_cast<FloatBig>(cValidationSamples);
   if(0 != cWeights && 0 != cValidationSamples) {
      error = ExtractWeights(
         pDataSetShared,
         BagEbm { -1 },
         aBag, 
         cValidationSamples,
         &pBoosterCore->m_aValidationWeights
      );
      if(Error_None != error) {
         // error already logged
         return error;
      }
      if(nullptr != pBoosterCore->m_aValidationWeights) {
         const FloatBig total = AddPositiveFloatsSafeBig(cValidationSamples, pBoosterCore->m_aValidationWeights);
         if(std::isnan(total) || std::isinf(total) || total <= 0) {
            LOG_0(Trace_Warning, "WARNING BoosterCore::Create std::isnan(total) || std::isinf(total) || total <= 0");
            return Error_UserParamVal;
         }
         // if they were all zero then we'd ignore the weights param.  If there are negative numbers it might add
         // to zero though so check it after checking for negative
         EBM_ASSERT(0 != total);
         pBoosterCore->m_validationWeightTotal = total;
      }
   }

   pBoosterCore->m_cClasses = cClasses;
   pBoosterCore->m_bestModelMetric = std::numeric_limits<double>::infinity();

   LOG_0(Trace_Info, "Exited BoosterCore::Create");
   return Error_None;
}

ErrorEbm BoosterCore::InitializeBoosterGradientsAndHessians(
   FloatFast * const aMulticlassMidwayTemp,
   FloatFast * const aUpdateScores
) {
#ifndef NDEBUG
   const size_t cScores = GetCountScores(GetCountClasses());
   // we should be initted to zero
   for(size_t iScore = 0; iScore < cScores; ++iScore) {
      EBM_ASSERT(0 == aUpdateScores[iScore]);
   }
#endif // NDEBUG

   ApplyUpdateBridge data;
   data.m_cClasses = GetCountClasses();
   data.m_cPack = k_cItemsPerBitPackNone;
   data.m_bCalcMetric = false;
   data.m_aMulticlassMidwayTemp = aMulticlassMidwayTemp;
   data.m_aUpdateTensorScores = aUpdateScores;
   data.m_cSamples = GetTrainingSet()->GetCountSamples();
   data.m_aPacked = nullptr;
   data.m_aTargets = GetTrainingSet()->GetTargetDataPointer();
   data.m_aWeights = nullptr;
   data.m_aSampleScores = GetTrainingSet()->GetSampleScores();
   data.m_aGradientsAndHessians = GetTrainingSet()->GetGradientsAndHessiansPointer();
   return ApplyUpdate(&data);
}

} // DEFINED_ZONE_NAME
