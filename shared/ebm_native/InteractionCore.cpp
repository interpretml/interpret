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

extern ErrorEbm ApplyTermUpdateValidation(
   const ptrdiff_t cRuntimeClasses,
   const ptrdiff_t runtimeBitPack,
   const bool bCalcMetric,
   FloatFast * const aMulticlassMidwayTemp,
   const FloatFast * const aUpdateScores,
   const size_t cSamples,
   const StorageDataType * const aInputData,
   const void * const aTargetData,
   const FloatFast * const aWeight,
   FloatFast * const aSampleScore,
   FloatFast * const aGradientAndHessian,
   double * const pMetricOut
);

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

   size_t cSamples = 0;
   size_t cFeatures = 0;
   size_t cWeights = 0;
   size_t cTargets = 0;
   error = GetDataSetSharedHeader(pDataSetShared, &cSamples, &cFeatures, &cWeights, &cTargets);
   if(Error_None != error) {
      // already logged
      return error;
   }
   if(size_t { 1 } < cWeights) {
      LOG_0(Trace_Warning, "WARNING InteractionCore::Create size_t { 1 } < cWeights");
      return Error_IllegalParamVal;
   }
   if(size_t { 1 } != cTargets) {
      LOG_0(Trace_Warning, "WARNING InteractionCore::Create 1 != cTargets");
      return Error_IllegalParamVal;
   }

   ptrdiff_t cClasses;
   GetDataSetSharedTarget(pDataSetShared, 0, &cClasses);

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

      if(IsMultiplyError(sizeof(Feature), cFeatures)) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::Allocate IsMultiplyError(sizeof(Feature), cFeatures)");
         return Error_OutOfMemory;
      }
      pRet->m_cFeatures = cFeatures;
      Feature * const aFeatures = static_cast<Feature *>(malloc(sizeof(Feature) * cFeatures));
      if(nullptr == aFeatures) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::Allocate nullptr == aFeatures");
         return Error_OutOfMemory;
      }
      pRet->m_aFeatures = aFeatures;

      size_t iFeatureInitialize = 0;
      do {
         size_t cBins;
         bool bMissing;
         bool bUnknown;
         bool bNominal;
         bool bSparse;
         SharedStorageDataType defaultValSparse;
         size_t cNonDefaultsSparse;
         GetDataSetSharedFeature(
            pDataSetShared,
            iFeatureInitialize,
            &cBins,
            &bMissing,
            &bUnknown,
            &bNominal,
            &bSparse,
            &defaultValSparse,
            &cNonDefaultsSparse
         );
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
      size_t cSamples = m_dataFrame.GetCountSamples();
      EBM_ASSERT(1 <= cSamples); // if m_dataFrame.IsGradientsAndHessiansNull

      ptrdiff_t cClasses;
      const void * const aTargetsFrom = GetDataSetSharedTarget(pDataSetShared, 0, &cClasses);
      EBM_ASSERT(nullptr != aTargetsFrom);
      EBM_ASSERT(0 != cClasses); // no gradients if 0 == cClasses
      EBM_ASSERT(1 != cClasses); // no gradients if 1 == cClasses
      EBM_ASSERT(IsClassification(cClasses));
      const size_t cScores = GetCountScores(cClasses);
      if(IsMultiplyError(sizeof(FloatFast), cScores, cSamples)) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians IsMultiplyError(sizeof(FloatFast), cScores, cSamples)");
         return Error_OutOfMemory;
      }
      const size_t cBytesScores = sizeof(FloatFast) * cScores;
      const size_t cBytesAllScores = cBytesScores * cSamples;

      if(IsMultiplyError(sizeof(StorageDataType), cSamples)) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians IsMultiplyError(sizeof(StorageDataType), cSamples)");
         return Error_OutOfMemory;
      }

      FloatFast * const aSampleScoreTo = static_cast<FloatFast *>(malloc(cBytesAllScores));
      if(UNLIKELY(nullptr == aSampleScoreTo)) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians nullptr == aSampleScoreTo");
         return Error_OutOfMemory;
      }

      StorageDataType * const aTargetsTo = static_cast<StorageDataType *>(malloc(sizeof(StorageDataType) * cSamples));
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

      if(nullptr == aInitScores) {
         // if aInitScores is nullptr then all initial scores are zero
         memset(aSampleScoreTo, 0, cBytesAllScores);
      }

      const BagEbm * pSampleReplication = aBag;
      const SharedStorageDataType * pTargetFrom = static_cast<const SharedStorageDataType *>(aTargetsFrom);
      StorageDataType * pTargetTo = aTargetsTo;
      const double * pInitScoreFrom = aInitScores;
      FloatFast * pSampleScoreTo = aSampleScoreTo;
      FloatFast * pSampleScoreToEnd = reinterpret_cast<FloatFast *>(reinterpret_cast<char *>(aSampleScoreTo) + cBytesAllScores);
      do {
         BagEbm replication = 1;
         if(nullptr != pSampleReplication) {
            replication = *pSampleReplication;
            ++pSampleReplication;
         }
         if(BagEbm { 0 } != replication) {
            if(BagEbm { 0 } < replication) {
               const SharedStorageDataType targetOriginal = *pTargetFrom;
               // the shared data storage structure ensures that all target values are less than the number of classes
               // we also check that the number of classes can be converted to a ptrdiff_t and also a StorageDataType
               // so we do not need the runtime to check this
               EBM_ASSERT(targetOriginal < static_cast<SharedStorageDataType>(cClasses));
               // since cClasses must be below StorageDataType, it follows that..
               EBM_ASSERT(!IsConvertError<StorageDataType>(targetOriginal));
               const StorageDataType target = static_cast<StorageDataType>(targetOriginal);
               const double * pInitScoreFromEnd = nullptr == pInitScoreFrom ? nullptr : pInitScoreFrom + cScores;
               do {
                  *pTargetTo = target;
                  ++pTargetTo;

                  if(nullptr != pInitScoreFrom) {
                     do {
                        *pSampleScoreTo = SafeConvertFloat<FloatFast>(*pInitScoreFrom);
                        ++pSampleScoreTo;
                        ++pInitScoreFrom;
                     } while(pInitScoreFromEnd != pInitScoreFrom);
                     pInitScoreFrom -= cScores; // in case replication is more than 1 and we do another loop
                  }

                  --replication;
               } while(BagEbm { 0 } != replication);
            }
            if(nullptr != pInitScoreFrom) {
               pInitScoreFrom += cScores;
            }
         }
         ++pTargetFrom; // target data is shared so unlike init scores we must keep them even if replication is zero
      } while(pSampleScoreToEnd != pSampleScoreTo);

      double unused;
      const ErrorEbm error = ApplyTermUpdateValidation(
         cClasses,
         k_cItemsPerBitPackNone,
         false,
         aMulticlassMidwayTemp,
         aUpdateScores,
         cSamples,
         nullptr,
         aTargetsTo,
         nullptr,
         aSampleScoreTo,
         m_dataFrame.GetGradientsAndHessiansPointer(),
         &unused
      );

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
