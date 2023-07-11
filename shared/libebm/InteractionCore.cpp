// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "logging.h" // EBM_ASSERT

#include "bridge_cpp.hpp" // GetCountScores
#include "Bin.hpp" // IsOverflowBinSize

#include "compute_accessors.hpp"
#include "ebm_internal.hpp" // SafeConvertFloat
#include "Feature.hpp" // Feature
#include "dataset_shared.hpp" // GetDataSetSharedHeader
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
   const size_t cSamples,
   const size_t cFeatures,
   const size_t cWeights,
   const BagEbm * const aBag,
   const CreateInteractionFlags flags,
   const char * const sObjective,
   const double * const experimentalParams,
   InteractionCore ** const ppInteractionCoreOut
) {
   // experimentalParams isn't used by default.  It's meant to provide an easy way for python or other higher
   // level languages to pass EXPERIMENTAL temporary parameters easily to the C++ code.
   UNUSED(experimentalParams);

   LOG_0(Trace_Info, "Entered InteractionCore::Create");

   EBM_ASSERT(nullptr != ppInteractionCoreOut);
   EBM_ASSERT(nullptr == *ppInteractionCoreOut);
   EBM_ASSERT(nullptr != pDataSetShared);

   ErrorEbm error;

   InteractionCore * pInteractionCore;
   try {
      pInteractionCore = new InteractionCore();
   } catch(const std::bad_alloc &) {
      LOG_0(Trace_Warning, "WARNING InteractionCore::Create Out of memory allocating InteractionCore");
      return Error_OutOfMemory;
   } catch(...) {
      LOG_0(Trace_Warning, "WARNING InteractionCore::Create Unknown error");
      return Error_UnexpectedInternal;
   }
   if(nullptr == pInteractionCore) {
      // this should be impossible since bad_alloc should have been thrown, but let's be untrusting
      LOG_0(Trace_Warning, "WARNING InteractionCore::Create nullptr == pInteractionCore");
      return Error_OutOfMemory;
   }
   // give ownership of our object back to the caller, even if there is a failure
   *ppInteractionCoreOut = pInteractionCore;

   size_t cTrainingSamples;
   size_t cValidationSamples;
   error = Unbag(cSamples, aBag, &cTrainingSamples, &cValidationSamples);
   if(Error_None != error) {
      // already logged
      return error;
   }

   LOG_0(Trace_Info, "InteractionCore::Create starting feature processing");
   if(0 != cFeatures) {
      if(IsMultiplyError(sizeof(FeatureInteraction), cFeatures)) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::Create IsMultiplyError(sizeof(Feature), cFeatures)");
         return Error_OutOfMemory;
      }
      pInteractionCore->m_cFeatures = cFeatures;
      FeatureInteraction * const aFeatures =
         static_cast<FeatureInteraction *>(malloc(sizeof(FeatureInteraction) * cFeatures));
      if(nullptr == aFeatures) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::Create nullptr == aFeatures");
         return Error_OutOfMemory;
      }
      pInteractionCore->m_aFeatures = aFeatures;

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
            LOG_0(Trace_Error, "ERROR InteractionCore::Create IsConvertError<size_t>(countBins)");
            return Error_IllegalParamVal;
         }
         const size_t cBins = static_cast<size_t>(countBins);
         if(0 == cBins) {
            if(0 != cSamples) {
               LOG_0(Trace_Error, "ERROR InteractionCore::Create countBins cannot be zero if 0 < cSamples");
               return Error_IllegalParamVal;
            }
            // we can handle 0 == cBins even though that's a degenerate case that shouldn't be boosted on.  0 bins
            // can only occur if there were zero training and zero validation cases since the 
            // features would require a value, even if it was 0.
            LOG_0(Trace_Info, "INFO InteractionCore::Create feature with 0 values");
         } else if(1 == cBins) {
            // we can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on. 
            // Dimensions with 1 bin don't contribute anything since they always have the same value.
            LOG_0(Trace_Info, "INFO InteractionCore::Create feature with 1 value");
         }
         aFeatures[iFeatureInitialize].Initialize(cBins, bMissing, bUnknown, bNominal);

         ++iFeatureInitialize;
      } while(cFeatures != iFeatureInitialize);
   }
   LOG_0(Trace_Info, "InteractionCore::Create done feature processing");


   ptrdiff_t cClasses;
   const void * const aTargets = GetDataSetSharedTarget(pDataSetShared, 0, &cClasses);
   if(nullptr == aTargets) {
      LOG_0(Trace_Warning, "WARNING InteractionCore::Create cClasses cannot fit into ptrdiff_t");
      return Error_IllegalParamVal;
   }
   pInteractionCore->m_cClasses = cClasses;

   if(ptrdiff_t { 0 } != cClasses && ptrdiff_t { 1 } != cClasses) {
      const size_t cScores = GetCountScores(cClasses);

      LOG_0(Trace_Info, "INFO InteractionCore::Create determining Objective");
      Config config;
      config.cOutputs = cScores;
      config.isDifferentialPrivacy = 0 != (CreateInteractionFlags_DifferentialPrivacy & flags) ? EBM_TRUE : EBM_FALSE;
      error = GetObjective(&config, sObjective, &pInteractionCore->m_objectiveCpu, &pInteractionCore->m_objectiveSIMD);
      if(Error_None != error) {
         // already logged
         return error;
      }
      LOG_0(Trace_Info, "INFO InteractionCore::Create Objective determined");

      const OutputType outputType = GetOutputType(pInteractionCore->m_objectiveCpu.m_linkFunction);
      if(IsClassification(cClasses)) {
         if(outputType < OutputType_GeneralClassification) {
            LOG_0(Trace_Error, "ERROR InteractionCore::Create mismatch in objective class model type");
            return Error_IllegalParamVal;
         }
      } else {
         if(OutputType_Regression != outputType) {
            LOG_0(Trace_Error, "ERROR InteractionCore::Create mismatch in objective class model type");
            return Error_IllegalParamVal;
         }
      }

      if(EBM_FALSE != pInteractionCore->CheckTargets(cSamples, aTargets)) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::Create invalid target value");
         return Error_ObjectiveIllegalTarget;
      }
      LOG_0(Trace_Info, "INFO InteractionCore::Create Targets verified");

      const bool bHessian = pInteractionCore->IsHessian();

      FeatureInteraction * pFeature = pInteractionCore->m_aFeatures;
      const FeatureInteraction * const pFeatureEnd = nullptr == pFeature ? nullptr : pFeature + cFeatures;
      if(sizeof(UInt_Small) == pInteractionCore->m_objectiveCpu.m_cUIntBytes) {
         if(sizeof(Float_Small) == pInteractionCore->m_objectiveCpu.m_cFloatBytes) {
            if(IsOverflowBinSize<Float_Small, UInt_Small>(bHessian, cScores)) {
               LOG_0(Trace_Warning, "WARNING InteractionCore::Create bin size overflow");
               return Error_OutOfMemory;
            }
         } else {
            EBM_ASSERT(sizeof(Float_Big) == pInteractionCore->m_objectiveCpu.m_cFloatBytes);
            if(IsOverflowBinSize<Float_Big, UInt_Small>(bHessian, cScores)) {
               LOG_0(Trace_Warning, "WARNING InteractionCore::Create bin size overflow");
               return Error_OutOfMemory;
            }
         }
         if(IsMulticlass(cClasses)) {
            // TODO: we currently index into the gradient array using the target, but the gradient array is also
            // layed out per-SIMD pack.  Once we sort the dataset by the target we'll be able to use non-random
            // indexing to fetch all the sample targets simultaneously, and we'll no longer need this indexing
            size_t cIndexes = static_cast<size_t>(cClasses);
            if(bHessian) {
               if(IsMultiplyError(size_t { 2 }, cIndexes)) {
                  LOG_0(Trace_Error, "ERROR InteractionCore::Create target indexes cannot fit into compute zone indexes");
                  return Error_IllegalParamVal;
               }
               cIndexes *= 2;
            }
            if(IsConvertError<UInt_Small>(cIndexes - size_t { 1 })) {
               LOG_0(Trace_Error, "ERROR InteractionCore::Create target indexes cannot fit into compute zone indexes");
               return Error_IllegalParamVal;
            }
         }
         for(; pFeatureEnd != pFeature; ++pFeature) {
            const size_t cBins = pFeature->GetCountBins();
            if(0 != cBins && IsConvertError<UInt_Small>(cBins - 1)) {
               LOG_0(Trace_Error, "ERROR InteractionCore::Create IsConvertError<UInt_Small>((*ppTerm)->GetCountTensorBins())");
               return Error_IllegalParamVal;
            }
         }
      } else {
         EBM_ASSERT(sizeof(UInt_Big) == pInteractionCore->m_objectiveCpu.m_cUIntBytes);
         if(sizeof(Float_Small) == pInteractionCore->m_objectiveCpu.m_cFloatBytes) {
            if(IsOverflowBinSize<Float_Small, UInt_Big>(bHessian, cScores)) {
               LOG_0(Trace_Warning, "WARNING InteractionCore::Create bin size overflow");
               return Error_OutOfMemory;
            }
         } else {
            EBM_ASSERT(sizeof(Float_Big) == pInteractionCore->m_objectiveCpu.m_cFloatBytes);
            if(IsOverflowBinSize<Float_Big, UInt_Big>(bHessian, cScores)) {
               LOG_0(Trace_Warning, "WARNING InteractionCore::Create bin size overflow");
               return Error_OutOfMemory;
            }
         }
         if(IsMulticlass(cClasses)) {
            // TODO: we currently index into the gradient array using the target, but the gradient array is also
            // layed out per-SIMD pack.  Once we sort the dataset by the target we'll be able to use non-random
            // indexing to fetch all the sample targets simultaneously, and we'll no longer need this indexing
            size_t cIndexes = static_cast<size_t>(cClasses);
            if(bHessian) {
               if(IsMultiplyError(size_t { 2 }, cIndexes)) {
                  LOG_0(Trace_Error, "ERROR InteractionCore::Create target indexes cannot fit into compute zone indexes");
                  return Error_IllegalParamVal;
               }
               cIndexes *= 2;
            }
            if(IsConvertError<UInt_Big>(cIndexes - size_t { 1 })) {
               LOG_0(Trace_Error, "ERROR InteractionCore::Create target indexes cannot fit into compute zone indexes");
               return Error_IllegalParamVal;
            }
         }
         for(; pFeatureEnd != pFeature; ++pFeature) {
            const size_t cBins = pFeature->GetCountBins();
            if(0 != cBins && IsConvertError<UInt_Big>(cBins - 1)) {
               LOG_0(Trace_Error, "ERROR InteractionCore::Create IsConvertError<UInt_Big>((*ppTerm)->GetCountTensorBins())");
               return Error_IllegalParamVal;
            }
         }
      }

      if(0 != pInteractionCore->m_objectiveSIMD.m_cUIntBytes) {
         bool bRemoveSIMD = false;
         while(true) {
            if(sizeof(UInt_Small) == pInteractionCore->m_objectiveSIMD.m_cUIntBytes) {
               if(sizeof(Float_Small) == pInteractionCore->m_objectiveSIMD.m_cFloatBytes) {
                  if(IsOverflowBinSize<Float_Small, UInt_Small>(bHessian, cScores)) {
                     bRemoveSIMD = true;
                     break;
                  }
               } else {
                  EBM_ASSERT(sizeof(Float_Big) == pInteractionCore->m_objectiveSIMD.m_cFloatBytes);
                  if(IsOverflowBinSize<Float_Big, UInt_Small>(bHessian, cScores)) {
                     bRemoveSIMD = true;
                     break;
                  }
               }
               if(IsMulticlass(cClasses)) {
                  // TODO: we currently index into the gradient array using the target, but the gradient array is also
                  // layed out per-SIMD pack.  Once we sort the dataset by the target we'll be able to use non-random
                  // indexing to fetch all the sample targets simultaneously, and we'll no longer need this indexing
                  size_t cIndexes = static_cast<size_t>(cClasses);
                  if(bHessian) {
                     if(IsMultiplyError(size_t { 2 }, cIndexes)) {
                        bRemoveSIMD = true;
                        break;
                     }
                     cIndexes *= 2;
                  }
                  if(IsMultiplyError(cIndexes, pInteractionCore->m_objectiveSIMD.m_cSIMDPack)) {
                     bRemoveSIMD = true;
                     break;
                  }
                  if(IsConvertError<UInt_Small>(cIndexes * pInteractionCore->m_objectiveSIMD.m_cSIMDPack - size_t { 1 })) {
                     bRemoveSIMD = true;
                     break;
                  }
               }
               for(; pFeatureEnd != pFeature; ++pFeature) {
                  const size_t cBins = pFeature->GetCountBins();
                  if(0 != cBins && IsConvertError<UInt_Small>(cBins - 1)) {
                     bRemoveSIMD = true;
                     break;
                  }
               }
            } else {
               EBM_ASSERT(sizeof(UInt_Big) == pInteractionCore->m_objectiveSIMD.m_cUIntBytes);
               if(sizeof(Float_Small) == pInteractionCore->m_objectiveSIMD.m_cFloatBytes) {
                  if(IsOverflowBinSize<Float_Small, UInt_Big>(bHessian, cScores)) {
                     bRemoveSIMD = true;
                     break;
                  }
               } else {
                  EBM_ASSERT(sizeof(Float_Big) == pInteractionCore->m_objectiveSIMD.m_cFloatBytes);
                  if(IsOverflowBinSize<Float_Big, UInt_Big>(bHessian, cScores)) {
                     bRemoveSIMD = true;
                     break;
                  }
               }
               if(IsMulticlass(cClasses)) {
                  // TODO: we currently index into the gradient array using the target, but the gradient array is also
                  // layed out per-SIMD pack.  Once we sort the dataset by the target we'll be able to use non-random
                  // indexing to fetch all the sample targets simultaneously, and we'll no longer need this indexing
                  size_t cIndexes = static_cast<size_t>(cClasses);
                  if(bHessian) {
                     if(IsMultiplyError(size_t { 2 }, cIndexes)) {
                        bRemoveSIMD = true;
                        break;
                     }
                     cIndexes *= 2;
                  }
                  if(IsMultiplyError(cIndexes, pInteractionCore->m_objectiveSIMD.m_cSIMDPack)) {
                     bRemoveSIMD = true;
                     break;
                  }
                  if(IsConvertError<UInt_Big>(cIndexes * pInteractionCore->m_objectiveSIMD.m_cSIMDPack - size_t { 1 })) {
                     bRemoveSIMD = true;
                     break;
                  }
               }
               for(; pFeatureEnd != pFeature; ++pFeature) {
                  const size_t cBins = pFeature->GetCountBins();
                  if(0 != cBins && IsConvertError<UInt_Big>(cBins - 1)) {
                     bRemoveSIMD = true;
                     break;
                  }
               }
            }
            break;
         }
         if(bRemoveSIMD) {
            FreeObjectiveWrapperInternals(&pInteractionCore->m_objectiveSIMD);
            InitializeObjectiveWrapperUnfailing(&pInteractionCore->m_objectiveSIMD);
         }
      }

      error = pInteractionCore->m_dataFrame.InitDataSetInteraction(
         bHessian,
         cScores,
         SIZE_MAX, // TODO: use k_cSubsetSamplesMax (and also use k_cSubsetSamplesMax everywhere else too)
         &pInteractionCore->m_objectiveCpu,
         &pInteractionCore->m_objectiveSIMD,
         pDataSetShared,
         cSamples,
         aBag,
         cTrainingSamples,
         cWeights,
         cFeatures
      );
      if(Error_None != error) {
         return error;
      }

      if(IsOverflowBinSize<FloatBig, StorageDataType>(bHessian, cScores)) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::Create IsOverflowBinSize overflow");
         return Error_OutOfMemory;
      }
   }

   LOG_0(Trace_Info, "Exited InteractionCore::Allocate");
   return Error_None;
}

WARNING_PUSH
WARNING_DISABLE_UNINITIALIZED_LOCAL_VARIABLE
ErrorEbm InteractionCore::InitializeInteractionGradientsAndHessians(
   const unsigned char * const pDataSetShared,
   const size_t cWeights,
   const BagEbm * const aBag,
   const double * const aInitScores
) {
   ErrorEbm error = Error_None;
   if(size_t { 0 } != m_dataFrame.GetCountSamples()) {
      ptrdiff_t cClasses;
      const void * const aTargetsFrom = GetDataSetSharedTarget(pDataSetShared, 0, &cClasses);
      EBM_ASSERT(nullptr != aTargetsFrom); // we previously called GetDataSetSharedTarget and got back a non-null result
      EBM_ASSERT(0 != cClasses); // no gradients if 0 == cClasses
      EBM_ASSERT(1 != cClasses); // no gradients if 1 == cClasses
      const size_t cScores = GetCountScores(cClasses);
      EBM_ASSERT(1 <= cScores);

      size_t cBytesScoresMax = 0;
      size_t cBytesAllScoresMax = 0;
      size_t cBytesTempMax = 0;
      size_t cBytesTargetMax = 0;
      DataSubsetInteraction * pSubsetInit = GetDataSetInteraction()->GetSubsets();
      EBM_ASSERT(1 <= GetDataSetInteraction()->GetCountSubsets());
      const DataSubsetInteraction * const pSubsetsEnd = pSubsetInit + GetDataSetInteraction()->GetCountSubsets();
      do {
         size_t cSamples = pSubsetInit->GetCountSamples();
         EBM_ASSERT(1 <= cSamples);

         if(IsMultiplyError(pSubsetInit->GetObjectiveWrapper()->m_cFloatBytes, cScores, 
            EbmMax(cSamples, pSubsetInit->GetObjectiveWrapper()->m_cSIMDPack))) 
         {
            LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians IsMultiplyError(pSubsetInit->GetObjectiveWrapper()->m_cFloatBytes, cScores, EbmMax(cSamples, pSubsetInit->GetObjectiveWrapper()->m_cSIMDPack))");
            return Error_OutOfMemory;
         }
         const size_t cBytesScores = pSubsetInit->GetObjectiveWrapper()->m_cFloatBytes * cScores;
         const size_t cBytesTemp = cBytesScores * pSubsetInit->GetObjectiveWrapper()->m_cSIMDPack;
         const size_t cBytesAllScores = cBytesScores * cSamples;

         cBytesScoresMax = EbmMax(cBytesScoresMax, cBytesScores);
         cBytesAllScoresMax = EbmMax(cBytesAllScoresMax, cBytesAllScores);
         cBytesTempMax = EbmMax(cBytesTempMax, cBytesTemp);

         if(IsClassification(cClasses)) {
            if(IsMultiplyError(pSubsetInit->GetObjectiveWrapper()->m_cUIntBytes, cSamples)) {
               LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians IsMultiplyError(pSubsetInit->GetObjectiveWrapper()->m_cUIntBytes, cSamples)");
               return Error_OutOfMemory;
            }
            const size_t cBytesTarget = pSubsetInit->GetObjectiveWrapper()->m_cUIntBytes * cSamples;
            cBytesTargetMax = EbmMax(cBytesTargetMax, cBytesTarget);
         } else {
            if(IsMultiplyError(pSubsetInit->GetObjectiveWrapper()->m_cFloatBytes, cSamples)) {
               LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians IsMultiplyError(pSubsetInit->GetObjectiveWrapper()->m_cFloatBytes, cSamples)");
               return Error_OutOfMemory;
            }
            const size_t cBytesTarget = pSubsetInit->GetObjectiveWrapper()->m_cFloatBytes * cSamples;
            cBytesTargetMax = EbmMax(cBytesTargetMax, cBytesTarget);
         }

         ++pSubsetInit;
      } while(pSubsetsEnd != pSubsetInit);

      ApplyUpdateBridge data;

      void * const aSampleScoreTo = AlignedAlloc(cBytesAllScoresMax);
      if(UNLIKELY(nullptr == aSampleScoreTo)) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians nullptr == aSampleScoreTo");
         return Error_OutOfMemory;
      }
      data.m_aSampleScores = aSampleScoreTo;

      void * const aUpdateScores = AlignedAlloc(cBytesScoresMax);
      if(UNLIKELY(nullptr == aUpdateScores)) {
         LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians nullptr == aUpdateScores");
         error = Error_OutOfMemory;
         goto free_sample_scores;
      }
      data.m_aUpdateTensorScores = aUpdateScores;

      memset(aUpdateScores, 0, cBytesScoresMax);

      data.m_aMulticlassMidwayTemp = nullptr;
      if(IsClassification(cClasses)) {
         void * const aTargetTo = AlignedAlloc(cBytesTargetMax);
         if(UNLIKELY(nullptr == aTargetTo)) {
            LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians nullptr == aTargetTo");
            error = Error_OutOfMemory;
            goto free_tensor_scores;
         }
         data.m_aTargets = aTargetTo;

         if(IsMulticlass(cClasses)) {
            void * const aMulticlassMidwayTemp = AlignedAlloc(cBytesTempMax);
            if(UNLIKELY(nullptr == aMulticlassMidwayTemp)) {
               LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians nullptr == aMulticlassMidwayTemp");
               error = Error_OutOfMemory;
               goto free_targets;
            }
            data.m_aMulticlassMidwayTemp = aMulticlassMidwayTemp;
         }

         const SharedStorageDataType * pTargetFrom = static_cast<const SharedStorageDataType *>(aTargetsFrom);

         const BagEbm * pSampleReplication = aBag;
         const double * pInitScoreFrom = aInitScores;
         BagEbm replication = 0;
         const double * pInitScoreFromOld = nullptr;
         SharedStorageDataType target;

         DataSubsetInteraction * pSubset = GetDataSetInteraction()->GetSubsets();
         do {
            EBM_ASSERT(1 <= pSubset->GetCountSamples());

            const size_t cSIMDPack = pSubset->GetObjectiveWrapper()->m_cSIMDPack;
            EBM_ASSERT(0 == pSubset->GetCountSamples() % cSIMDPack);

            void * pTargetTo = aTargetTo;
            void * pSampleScoreTo = aSampleScoreTo;
            const void * const pTargetToEnd = IndexByte(aTargetTo, pSubset->GetObjectiveWrapper()->m_cUIntBytes * pSubset->GetCountSamples());
            double initScore = 0;
            do {
               size_t iPartition = 0;
               do {
                  if(BagEbm { 0 } == replication) {
                     replication = 1;
                     size_t cAdvance = cScores;
                     if(nullptr != pSampleReplication) {
                        cAdvance = 0; // we'll add this now inside the loop below
                        do {
                           do {
                              replication = *pSampleReplication;
                              ++pSampleReplication;
                              ++pTargetFrom;
                           } while(BagEbm { 0 } == replication);
                           cAdvance += cScores;
                        } while(replication < BagEbm { 0 });
                        --pTargetFrom;
                     }

                     if(nullptr != pInitScoreFrom) {
                        pInitScoreFrom += cAdvance;
                        pInitScoreFromOld = pInitScoreFrom - cScores;
                     }

                     target = *pTargetFrom;
                     ++pTargetFrom; // target data is shared so unlike init scores we must keep them even if replication is zero

                     // the shared data storage structure ensures that all target values are less than the number of classes
                     // we also check that the number of classes can be converted to a ptrdiff_t and also a StorageDataType
                     // so we do not need the runtime to check this
                     EBM_ASSERT(target < static_cast<SharedStorageDataType>(cClasses));
                  }

                  if(sizeof(UInt_Small) == pSubset->GetObjectiveWrapper()->m_cUIntBytes) {
                     *reinterpret_cast<UInt_Small *>(pTargetTo) = static_cast<UInt_Small>(target);
                  } else {
                     EBM_ASSERT(sizeof(UInt_Big) == pSubset->GetObjectiveWrapper()->m_cUIntBytes);
                     *reinterpret_cast<UInt_Big *>(pTargetTo) = static_cast<UInt_Big>(target);
                  }
                  pTargetTo = IndexByte(pTargetTo, pSubset->GetObjectiveWrapper()->m_cUIntBytes);

                  const double * pInitScoreFromLoop = pInitScoreFromOld;
                  size_t iScore = 0;
                  do {
                     if(nullptr != pInitScoreFromLoop) {
                        initScore = *pInitScoreFromLoop;
                        ++pInitScoreFromLoop;
                     }

                     if(sizeof(Float_Small) == pSubset->GetObjectiveWrapper()->m_cFloatBytes) {
                        reinterpret_cast<Float_Small *>(pSampleScoreTo)[iScore * cSIMDPack + iPartition] = SafeConvertFloat<Float_Small>(initScore);
                     } else {
                        EBM_ASSERT(sizeof(Float_Big) == pSubset->GetObjectiveWrapper()->m_cFloatBytes);
                        reinterpret_cast<Float_Big *>(pSampleScoreTo)[iScore * cSIMDPack + iPartition] = SafeConvertFloat<Float_Big>(initScore);
                     }
                     ++iScore;
                  } while(cScores != iScore);
                  --replication;

                  ++iPartition;
               } while(cSIMDPack != iPartition);
               pSampleScoreTo = IndexByte(pSampleScoreTo, cScores * pSubset->GetObjectiveWrapper()->m_cFloatBytes * cSIMDPack);
            } while(pTargetToEnd != pTargetTo);

            data.m_cScores = cScores;
            data.m_cPack = k_cItemsPerBitPackNone;
            data.m_bHessianNeeded = IsHessian() ? EBM_TRUE : EBM_FALSE;
            data.m_bCalcMetric = EBM_FALSE;
            data.m_cSamples = pSubset->GetCountSamples();
            data.m_aPacked = nullptr;
            data.m_aWeights = nullptr;
            data.m_aGradientsAndHessians = pSubset->GetGradHess();
            // this is a kind of hack (a good one) where we are sending in an update of all zeros in order to 
            // reuse the same code that we use for boosting in order to generate our gradients and hessians
            error = pSubset->ObjectiveApplyUpdate(&data);
            if(Error_None != error) {
               goto free_temp;
            }
            ++pSubset;
         } while(pSubsetsEnd != pSubset);
         EBM_ASSERT(0 == replication);

      free_temp:
         AlignedFree(data.m_aMulticlassMidwayTemp); // nullptr ok
      } else {
         void * const aTargetTo = AlignedAlloc(cBytesTargetMax);
         if(UNLIKELY(nullptr == aTargetTo)) {
            LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians nullptr == aTargetTo");
            error = Error_OutOfMemory;
            goto free_tensor_scores;
         }
         data.m_aTargets = aTargetTo;

         const FloatFast * pTargetFrom = static_cast<const FloatFast *>(aTargetsFrom);

         const BagEbm * pSampleReplication = aBag;
         const double * pInitScoreFrom = aInitScores;
         BagEbm replication = 0;
         double initScore = 0;
         FloatFast target;

         DataSubsetInteraction * pSubset = GetDataSetInteraction()->GetSubsets();
         do {
            EBM_ASSERT(1 <= pSubset->GetCountSamples());
            void * pTargetTo = aTargetTo;
            void * pSampleScoreTo = aSampleScoreTo;
            const void * const pTargetToEnd = IndexByte(aTargetTo, pSubset->GetObjectiveWrapper()->m_cFloatBytes * pSubset->GetCountSamples());
            do {
               if(BagEbm { 0 } == replication) {
                  replication = 1;
                  size_t cAdvance = 1;
                  if(nullptr != pSampleReplication) {
                     cAdvance = 0; // we'll add this now inside the loop below
                     do {
                        do {
                           replication = *pSampleReplication;
                           ++pSampleReplication;
                           ++pTargetFrom;
                        } while(BagEbm { 0 } == replication);
                        ++cAdvance;
                     } while(replication < BagEbm { 0 });
                     --pTargetFrom;
                  }

                  if(nullptr != pInitScoreFrom) {
                     pInitScoreFrom += cAdvance;
                     initScore = pInitScoreFrom[-1];
                  }

                  target = *pTargetFrom;
                  ++pTargetFrom; // target data is shared so unlike init scores we must keep them even if replication is zero
               }

               if(sizeof(Float_Small) == pSubset->GetObjectiveWrapper()->m_cFloatBytes) {
                  *reinterpret_cast<Float_Small *>(pTargetTo) = static_cast<Float_Small>(target);
               } else {
                  EBM_ASSERT(sizeof(Float_Big) == pSubset->GetObjectiveWrapper()->m_cFloatBytes);
                  *reinterpret_cast<Float_Big *>(pTargetTo) = static_cast<Float_Big>(target);
               }
               pTargetTo = IndexByte(pTargetTo, pSubset->GetObjectiveWrapper()->m_cFloatBytes);

               if(sizeof(Float_Small) == pSubset->GetObjectiveWrapper()->m_cFloatBytes) {
                  *reinterpret_cast<Float_Small *>(pSampleScoreTo) = SafeConvertFloat<Float_Small>(initScore);
               } else {
                  EBM_ASSERT(sizeof(Float_Big) == pSubset->GetObjectiveWrapper()->m_cFloatBytes);
                  *reinterpret_cast<Float_Big *>(pSampleScoreTo) = SafeConvertFloat<Float_Big>(initScore);
               }
               pSampleScoreTo = IndexByte(pSampleScoreTo, pSubset->GetObjectiveWrapper()->m_cFloatBytes);

               --replication;

            } while(pTargetToEnd != pTargetTo);

            data.m_cScores = cScores;
            data.m_cPack = k_cItemsPerBitPackNone;
            data.m_bHessianNeeded = IsHessian() ? EBM_TRUE : EBM_FALSE;
            data.m_bCalcMetric = EBM_FALSE;
            data.m_cSamples = pSubset->GetCountSamples();
            data.m_aPacked = nullptr;
            data.m_aWeights = nullptr;
            data.m_aGradientsAndHessians = pSubset->GetGradHess();
            // this is a kind of hack (a good one) where we are sending in an update of all zeros in order to 
            // reuse the same code that we use for boosting in order to generate our gradients and hessians
            error = pSubset->ObjectiveApplyUpdate(&data);
            if(Error_None != error) {
               goto free_targets;
            }
            ++pSubset;
         } while(pSubsetsEnd != pSubset);
         EBM_ASSERT(0 == replication);
      }

   free_targets:
      AlignedFree(const_cast<void *>(data.m_aTargets));
   free_tensor_scores:
      AlignedFree(const_cast<void *>(data.m_aUpdateTensorScores));
   free_sample_scores:
      AlignedFree(data.m_aSampleScores);

      if(size_t { 0 } != cWeights) {
         // optimize by now multiplying the gradients and hessians by the weights. The gradients and hessians are constants
         // after we exit this function and we just bin the non-changing values after this. By multiplying here
         // we can avoid doing the multiplication each time we bin them.

         const FloatFast * pWeight = GetDataSetSharedWeight(pDataSetShared, 0);
         EBM_ASSERT(nullptr != pWeight);

         size_t cTotalScores = cScores;
         if(IsHessian()) {
            EBM_ASSERT(!IsMultiplyError(size_t { 2 }, cTotalScores)); // we are accessing allocated memory
            cTotalScores = size_t { 2 } * cTotalScores;
         }

         const BagEbm * pSampleReplication = aBag;
         BagEbm replication = 0;
         FloatFast weight;

         DataSubsetInteraction * pSubset = GetDataSetInteraction()->GetSubsets();
         do {
            const size_t cSIMDPack = pSubset->GetObjectiveWrapper()->m_cSIMDPack;

            EBM_ASSERT(1 <= pSubset->GetCountSamples());
            EBM_ASSERT(0 == pSubset->GetCountSamples() % cSIMDPack);

            void * pGradHess = pSubset->GetGradHess();
            const void * const pGradHessEnd = IndexByte(pGradHess, pSubset->GetObjectiveWrapper()->m_cFloatBytes * cTotalScores * pSubset->GetCountSamples());
            do {
               size_t iPartition = 0;
               do {
                  if(BagEbm { 0 } == replication) {
                     replication = 1;
                     if(nullptr != pSampleReplication) {
                        do {
                           replication = *pSampleReplication;
                           ++pSampleReplication;
                           ++pWeight;
                        } while(replication <= BagEbm { 0 });
                        --pWeight;
                     }
                     weight = *pWeight;
                     ++pWeight;
                  }
                  size_t iScore = 0;
                  if(sizeof(Float_Small) == pSubset->GetObjectiveWrapper()->m_cFloatBytes) {
                     const Float_Small weightConverted = SafeConvertFloat<Float_Small>(weight);
                     do {
                        reinterpret_cast<Float_Small *>(pGradHess)[iScore * cSIMDPack + iPartition] *= weightConverted;
                        ++iScore;
                     } while(cTotalScores != iScore);
                  } else {
                     EBM_ASSERT(sizeof(Float_Big) == pSubset->GetObjectiveWrapper()->m_cFloatBytes);
                     const Float_Big weightConverted = SafeConvertFloat<Float_Big>(weight);
                     do {
                        reinterpret_cast<Float_Big *>(pGradHess)[iScore * cSIMDPack + iPartition] *= weightConverted;
                        ++iScore;
                     } while(cTotalScores != iScore);
                  }
                  --replication;

                  ++iPartition;
               } while(cSIMDPack != iPartition);
               pGradHess = IndexByte(pGradHess, pSubset->GetObjectiveWrapper()->m_cFloatBytes * cTotalScores * cSIMDPack);
            } while(pGradHessEnd != pGradHess);

            ++pSubset;
         } while(pSubsetsEnd != pSubset);
         EBM_ASSERT(0 == replication);
      }
   }
   return error;
}
WARNING_POP

} // DEFINED_ZONE_NAME
