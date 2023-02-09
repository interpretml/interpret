// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "logging.h" // EBM_ASSERT

#include "bridge_cpp.hpp" // GetCountScores

#include "ebm_internal.hpp" // SafeConvertFloat

#include "Feature.hpp" // Feature
#include "dataset_shared.hpp" // GetDataSetSharedHeader
#include "Bin.hpp" // IsOverflowBinSize
#include "InteractionCore.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern ErrorEbm Unbag(
   const size_t cSamples,
   const BagEbm * const aBag,
   size_t * const pcTrainingSamplesOut,
   size_t * const pcValidationSamplesOut
);

extern ErrorEbm ApplyUpdate(ApplyUpdateBridge * const pData);

void InteractionCore::Free(InteractionCore * const pInteractionCore) {
   LOG_0(Trace_Info, "Entered InteractionCore::Free");

   if(nullptr != pInteractionCore) {
      // for reference counting in general, a release is needed during the decrement and aquire is needed if freeing
      // https://www.boost.org/doc/libs/1_59_0/doc/html/atomic/usage_examples.html
      // We need to ensure that writes on this thread are not allowed to be re-ordered to a point below the 
      // decrement because if we happened to decrement to 2, and then get interrupted, and annother thread
      // decremented to 1 after us, we don't want our unclean writes to memory to be visible in the other thread
      // so we use memory_order_release on the decrement.
      if(size_t { 1 } == pInteractionCore->m_REFERENCE_COUNT.fetch_sub(1, std::memory_order_release)) {
         // we need to ensure that reads on this thread do not get reordered to a point before the decrement, otherwise
         // another thread might write some information, write the decrement to 2, then our thread decrements to 1
         // and then if we're allowed to read from data that occured before our decrement to 1 then we could have
         // stale data from before the other thread decrementing.  If our thread isn't freeing the memory though
         // we don't have to worry about staleness, so only use memory_order_acquire if we're going to delete the
         // object
         std::atomic_thread_fence(std::memory_order_acquire);
         LOG_0(Trace_Info, "INFO InteractionCore::Free deleting InteractionCore");
         delete pInteractionCore;
      }
   }

   LOG_0(Trace_Info, "Exited InteractionCore::Free");
}

ErrorEbm InteractionCore::Create(
   const unsigned char * const pDataSetShared,
   const BagEbm * const aBag,
   const double * const experimentalParams,
   InteractionCore ** const ppInteractionCoreOut
) {
   // experimentalParams isn't used by default.  It's meant to provide an easy way for python or other higher
   // level languages to pass EXPERIMENTAL temporary parameters easily to the C++ code.
   UNUSED(experimentalParams);

   LOG_0(Trace_Info, "Entered InteractionCore::Allocate");

   EBM_ASSERT(nullptr != ppInteractionCoreOut);
   EBM_ASSERT(nullptr == *ppInteractionCoreOut);
   EBM_ASSERT(nullptr != pDataSetShared);

   ErrorEbm error;

   InteractionCore * pRet;
   try {
      pRet = new InteractionCore();
   } catch(const std::bad_alloc &) {
      LOG_0(Trace_Warning, "WARNING InteractionCore::Create Out of memory allocating InteractionCore");
      return Error_OutOfMemory;
   } catch(...) {
      LOG_0(Trace_Warning, "WARNING InteractionCore::Create Unknown error");
      return Error_UnexpectedInternal;
   }
   if(nullptr == pRet) {
      // this should be impossible since bad_alloc should have been thrown, but let's be untrusting
      LOG_0(Trace_Warning, "WARNING InteractionCore::Create nullptr == pInteractionCore");
      return Error_OutOfMemory;
   }
   // give ownership of our object back to the caller, even if there is a failure
   *ppInteractionCoreOut = pRet;

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
      LOG_0(Trace_Error, "ERROR InteractionCore::Create IsConvertError<size_t>(countSamples)");
      return Error_IllegalParamVal;
   }
   size_t cSamples = static_cast<size_t>(countSamples);

   if(size_t { 1 } < cWeights) {
      LOG_0(Trace_Warning, "WARNING InteractionCore::Create size_t { 1 } < cWeights");
      return Error_IllegalParamVal;
   }
   if(size_t { 1 } != cTargets) {
      LOG_0(Trace_Warning, "WARNING InteractionCore::Create 1 != cTargets");
      return Error_IllegalParamVal;
   }

   ptrdiff_t cClasses;
   if(nullptr == GetDataSetSharedTarget(pDataSetShared, 0, &cClasses)) {
      LOG_0(Trace_Warning, "WARNING InteractionCore::Create cClasses cannot fit into ptrdiff_t");
      return Error_IllegalParamVal;
   }

   pRet->m_cClasses = cClasses;

   size_t cTrainingSamples;
   size_t cValidationSamples;
   error = Unbag(cSamples, aBag, &cTrainingSamples, &cValidationSamples);
   if(Error_None != error) {
      // already logged
      return error;
   }

   const bool bClassification = IsClassification(cClasses);

   LOG_0(Trace_Info, "InteractionCore::Allocate starting feature processing");
   if(0 != cFeatures) {
      const size_t cScores = GetCountScores(cClasses);
      if(IsOverflowBinSize<FloatFast>(bClassification, cScores) || IsOverflowBinSize<FloatBig>(bClassification, cScores)) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::Create IsOverflowBinSize overflow");
         return Error_OutOfMemory;
      }

      if(IsMultiplyError(sizeof(FeatureInteraction), cFeatures)) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::Allocate IsMultiplyError(sizeof(Feature), cFeatures)");
         return Error_OutOfMemory;
      }
      pRet->m_cFeatures = cFeatures;
      FeatureInteraction * const aFeatures = 
         static_cast<FeatureInteraction *>(malloc(sizeof(FeatureInteraction) * cFeatures));
      if(nullptr == aFeatures) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::Allocate nullptr == aFeatures");
         return Error_OutOfMemory;
      }
      pRet->m_aFeatures = aFeatures;

      size_t iFeatureInitialize = 0;
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
         EBM_ASSERT(!bSparse); // not handled yet

         if(IsConvertError<size_t>(countBins)) {
            LOG_0(Trace_Error, "ERROR InteractionCore::Allocate IsConvertError<size_t>(countBins)");
            return Error_IllegalParamVal;
         }
         const size_t cBins = static_cast<size_t>(countBins);
         if(0 == cBins) {
            if(0 != cSamples) {
               LOG_0(Trace_Error, "ERROR InteractionCore::Allocate countBins cannot be zero if 0 < cSamples");
               return Error_IllegalParamVal;
            }
            // we can handle 0 == cBins even though that's a degenerate case that shouldn't be boosted on.  0 bins
            // can only occur if there were zero training and zero validation cases since the 
            // features would require a value, even if it was 0.
            LOG_0(Trace_Info, "INFO InteractionCore::Allocate feature with 0 values");
         } else if(1 == cBins) {
            // we can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on. 
            // Dimensions with 1 bin don't contribute anything since they always have the same value.
            LOG_0(Trace_Info, "INFO InteractionCore::Allocate feature with 1 value");
         } else {
            // can we fit iBin into a StorageDataType. We need to check this prior to calling Initialize
            if(IsConvertError<StorageDataType>(cBins - size_t { 1 })) {
               LOG_0(Trace_Warning, "WARNING InteractionCore::Allocate IsConvertError<StorageDataType>(cBins - size_t { 1 })");
               return Error_OutOfMemory;
            }
         }
         aFeatures[iFeatureInitialize].Initialize(cBins, bMissing, bUnknown, bNominal);

         ++iFeatureInitialize;
      } while(cFeatures != iFeatureInitialize);
   }
   LOG_0(Trace_Info, "InteractionCore::Allocate done feature processing");

   error = pRet->m_dataFrame.Initialize(
      ptrdiff_t { 0 } != cClasses && ptrdiff_t { 1 } != cClasses,  // regression, binary, multiclass
      ptrdiff_t { 1 } < cClasses,  // binary, multiclass
      pDataSetShared,
      cSamples,
      aBag,
      cTrainingSamples,
      cWeights,
      cFeatures
   );
   if(Error_None != error) {
      LOG_0(Trace_Warning, "WARNING InteractionCore::Allocate m_dataFrame.Initialize");
      return error;
   }

   LOG_0(Trace_Info, "Exited InteractionCore::Allocate");
   return Error_None;
}

ErrorEbm InteractionCore::InitializeInteractionGradientsAndHessians(
   const unsigned char * const pDataSetShared,
   const BagEbm * const aBag,
   const double * const aInitScores
) {
   if(!m_dataFrame.IsGradientsAndHessiansNull()) {
      size_t cSetSamples = m_dataFrame.GetCountSamples();
      EBM_ASSERT(1 <= cSetSamples); // if m_dataFrame.IsGradientsAndHessiansNull

      ptrdiff_t cClasses;
      const void * const aTargetsFrom = GetDataSetSharedTarget(pDataSetShared, 0, &cClasses);
      EBM_ASSERT(nullptr != aTargetsFrom); // we previously called GetDataSetSharedTarget and got back a non-null result
      EBM_ASSERT(0 != cClasses); // no gradients if 0 == cClasses
      EBM_ASSERT(1 != cClasses); // no gradients if 1 == cClasses
      EBM_ASSERT(IsClassification(cClasses));
      const size_t cScores = GetCountScores(cClasses);
      if(IsMultiplyError(sizeof(FloatFast), cScores, cSetSamples)) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians IsMultiplyError(sizeof(FloatFast), cScores, cSetSamples)");
         return Error_OutOfMemory;
      }
      const size_t cBytesScores = sizeof(FloatFast) * cScores;
      const size_t cBytesAllScores = cBytesScores * cSetSamples;

      if(IsMultiplyError(sizeof(StorageDataType), cSetSamples)) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians IsMultiplyError(sizeof(StorageDataType), cSetSamples)");
         return Error_OutOfMemory;
      }

      FloatFast * const aSampleScoreTo = static_cast<FloatFast *>(malloc(cBytesAllScores));
      if(UNLIKELY(nullptr == aSampleScoreTo)) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians nullptr == aSampleScoreTo");
         return Error_OutOfMemory;
      }

      StorageDataType * const aTargetsTo = static_cast<StorageDataType *>(malloc(sizeof(StorageDataType) * cSetSamples));
      if(UNLIKELY(nullptr == aTargetsTo)) {
         free(aSampleScoreTo);
         LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians nullptr == aTargetsTo");
         return Error_OutOfMemory;
      }

      FloatFast * const aUpdateScores = static_cast<FloatFast *>(malloc(cBytesScores));
      if(UNLIKELY(nullptr == aUpdateScores)) {
         free(aTargetsTo);
         free(aSampleScoreTo);
         LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians nullptr == aUpdateScores");
         return Error_OutOfMemory;
      }

      FloatFast * aMulticlassMidwayTemp = nullptr;
      if(IsMulticlass(cClasses)) {
         aMulticlassMidwayTemp = static_cast<FloatFast *>(malloc(cBytesScores));
         if(UNLIKELY(nullptr == aMulticlassMidwayTemp)) {
            free(aUpdateScores);
            free(aTargetsTo);
            free(aSampleScoreTo);
            LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians nullptr == aMulticlassMidwayTemp");
            return Error_OutOfMemory;
         }
      }

      memset(aUpdateScores, 0, cBytesScores);


      const BagEbm * pSampleReplication = aBag;
      const SharedStorageDataType * pTargetFrom = static_cast<const SharedStorageDataType *>(aTargetsFrom);
      StorageDataType * pTargetTo = aTargetsTo;
      const StorageDataType * const pTargetToEnd = &aTargetsTo[cSetSamples];

      if(nullptr == aInitScores) {
         // if aInitScores is nullptr then all initial scores are zero
         memset(aSampleScoreTo, 0, cBytesAllScores);

         do {
            BagEbm replication = 1;
            if(nullptr != pSampleReplication) {
               do {
                  replication = *pSampleReplication;
                  ++pSampleReplication;
                  ++pTargetFrom;
               } while(replication <= BagEbm { 0 });
               --pTargetFrom;
            }
            const SharedStorageDataType targetOriginal = *pTargetFrom;
            ++pTargetFrom; // target data is shared so unlike init scores we must keep them even if replication is zero

            // the shared data storage structure ensures that all target values are less than the number of classes
            // we also check that the number of classes can be converted to a ptrdiff_t and also a StorageDataType
            // so we do not need the runtime to check this
            EBM_ASSERT(targetOriginal < static_cast<SharedStorageDataType>(cClasses));
            // since cClasses must be below StorageDataType, it follows that..
            EBM_ASSERT(!IsConvertError<StorageDataType>(targetOriginal));
            const StorageDataType target = static_cast<StorageDataType>(targetOriginal);
            do {
               *pTargetTo = target;
               ++pTargetTo;

               --replication;
            } while(BagEbm { 0 } != replication);
         } while(pTargetToEnd != pTargetTo);
      } else {
         const double * pInitScoreFrom = aInitScores;
         FloatFast * pSampleScoreTo = aSampleScoreTo;
         do {
            BagEbm replication = 1;
            if(nullptr != pSampleReplication) {
               do {
                  do {
                     replication = *pSampleReplication;
                     ++pSampleReplication;
                     ++pTargetFrom;
                  } while(BagEbm { 0 } == replication);
                  pInitScoreFrom += cScores;
               } while(replication < BagEbm { 0 });
               --pTargetFrom;
               pInitScoreFrom -= cScores;
            }

            const SharedStorageDataType targetOriginal = *pTargetFrom;
            ++pTargetFrom; // target data is shared so unlike init scores we must keep them even if replication is zero

            // the shared data storage structure ensures that all target values are less than the number of classes
            // we also check that the number of classes can be converted to a ptrdiff_t and also a StorageDataType
            // so we do not need the runtime to check this
            EBM_ASSERT(targetOriginal < static_cast<SharedStorageDataType>(cClasses));
            // since cClasses must be below StorageDataType, it follows that..
            EBM_ASSERT(!IsConvertError<StorageDataType>(targetOriginal));
            const StorageDataType target = static_cast<StorageDataType>(targetOriginal);
            const double * pInitScoreFromEnd = pInitScoreFrom + cScores;
            do {
               *pTargetTo = target;
               ++pTargetTo;

               do {
                  *pSampleScoreTo = SafeConvertFloat<FloatFast>(*pInitScoreFrom);
                  ++pSampleScoreTo;
                  ++pInitScoreFrom;
               } while(pInitScoreFromEnd != pInitScoreFrom);
               pInitScoreFrom -= cScores; // in case replication is more than 1 and we do another loop

               --replication;
            } while(BagEbm { 0 } != replication);
            pInitScoreFrom += cScores;
         } while(pTargetToEnd != pTargetTo);
      }

      ApplyUpdateBridge data;
      data.m_cClasses = cClasses;
      data.m_cPack = k_cItemsPerBitPackNone;
      data.m_bCalcMetric = false;
      data.m_aMulticlassMidwayTemp = aMulticlassMidwayTemp;
      data.m_aUpdateTensorScores = aUpdateScores;
      data.m_cSamples = cSetSamples;
      data.m_aPacked = nullptr;
      data.m_aTargets = aTargetsTo;
      data.m_aWeights = m_dataFrame.GetWeights();
      data.m_aSampleScores = aSampleScoreTo;
      data.m_aGradientsAndHessians = m_dataFrame.GetGradientsAndHessiansPointer();
      // this is a kind of hack (a good one) where we are sending in an update of all zeros in order to 
      // reuse the same code that we use for boosting in order to generate our gradients and hessians
      const ErrorEbm error = ApplyUpdate(&data);

      free(aMulticlassMidwayTemp); // nullptr ok
      free(aUpdateScores);
      free(aTargetsTo);
      free(aSampleScoreTo);

      if(Error_None != error) {
         return error;
      }
   }
   return Error_None;
}

} // DEFINED_ZONE_NAME
