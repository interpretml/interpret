// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "logging.h" // EBM_ASSERT

#include "bridge.hpp" // ObjectiveWrapper
#include "Bin.hpp" // IsOverflowBinSize

#include "ebm_internal.hpp"
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

extern ErrorEbm GetObjective(
   const Config * const pConfig,
   const char * sObjective,
   const ComputeFlags disableCompute,
   ObjectiveWrapper * const pCpuObjectiveWrapperOut,
   ObjectiveWrapper * const pSIMDObjectiveWrapperOut
) noexcept;

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

template<typename TUInt>
static bool CheckInteractionRestrictionsInternal(
   const InteractionCore * const pInteractionCore,
   const ObjectiveWrapper * const pObjectiveWrapper,
   size_t cBinsMax
) {
   EBM_ASSERT(nullptr != pInteractionCore);
   EBM_ASSERT(nullptr != pObjectiveWrapper);
   EBM_ASSERT(1 <= pInteractionCore->GetCountFeatures());

   const size_t cScores = pInteractionCore->GetCountScores();
   const bool bHessian = EBM_FALSE != pObjectiveWrapper->m_bObjectiveHasHessian;

   // In BinSumsInteraction we calculate the BinSize value but keep it as a size_t, so the only requirement is
   // that the bin size is calculatable
   if(sizeof(FloatBig) == pObjectiveWrapper->m_cFloatBytes) {
      if(IsOverflowBinSize<FloatBig, TUInt>(bHessian, cScores)) {
         return true;
      }
   } else {
      EBM_ASSERT(sizeof(FloatSmall) == pObjectiveWrapper->m_cFloatBytes);
      if(IsOverflowBinSize<FloatSmall, TUInt>(bHessian, cScores)) {
         return true;
      }
   }

   EBM_ASSERT(1 <= cBinsMax); // since cBins can only be 0 if cSamples is 0, and we checked that
   if(IsConvertError<TUInt>(cBinsMax - 1)) {
      // In BinSumsInteractions we retrieve the binned feature index stored in an array of SIMDable integers, so 
      // the SIMDable integer needs to be large enough to hold the maximum feature index.
      //
      // In BinSumsInteractions there are no other restrictions becasue we have a code path that does the tensor
      // bin calculation in a non-SIMD size_t format. At interaction detection time we can alternatively select
      // a faster version of that function if the SIMD type has enough space to multiply the indexes to fit
      // the tensor and by the cBytesPerBin value.
      // 
      // Unlike in boosting, for interactions we use the objective code only to initialize, and for that
      // we do not pass in the binned feature index, but instead pass in k_cItemsPerBitPackNone, so we do not
      // need to restrict ourselves to positive numbers and we do not care about the number of bins there.

      return true;
   }

   if(size_t { 1 } != cScores) {
      // TODO: we currently index into the gradient array using the target, but the gradient array is also
      // layed out per-SIMD pack.  Once we sort the dataset by the target we'll be able to use non-random
      // indexing to fetch all the sample targets simultaneously, and we'll no longer need this indexing
      size_t cIndexes = cScores;
      if(bHessian) {
         if(IsMultiplyError(size_t { 2 }, cIndexes)) {
            return true;
         }
         cIndexes <<= 1;
      }
      if(IsMultiplyError(cIndexes, pObjectiveWrapper->m_cSIMDPack)) {
         return true;
      }
      // restriction from LogLossMulticlassObjective.hpp
      // we use the target value to index into the temp exp array and adjust the target gradient
      if(IsConvertError<typename std::make_signed<TUInt>::type>(cIndexes * pObjectiveWrapper->m_cSIMDPack - size_t { 1 })) {
         return true;
      }
   }

   return false;
}

static bool CheckInteractionRestrictions(
   const InteractionCore * const pInteractionCore,
   const ObjectiveWrapper * const pObjectiveWrapper,
   const size_t cBinsMax
) {
   EBM_ASSERT(nullptr != pObjectiveWrapper);
   if(sizeof(UIntBig) == pObjectiveWrapper->m_cUIntBytes) {
      return CheckInteractionRestrictionsInternal<UIntBig>(pInteractionCore, pObjectiveWrapper, cBinsMax);
   } else {
      EBM_ASSERT(sizeof(UIntSmall) == pObjectiveWrapper->m_cUIntBytes);
      return CheckInteractionRestrictionsInternal<UIntSmall>(pInteractionCore, pObjectiveWrapper, cBinsMax);
   }
}

ErrorEbm InteractionCore::Create(
   const unsigned char * const pDataSetShared,
   const size_t cSamples,
   const size_t cFeatures,
   const size_t cWeights,
   const BagEbm * const aBag,
   const CreateInteractionFlags flags,
   const ComputeFlags disableCompute,
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

   pInteractionCore->m_bDisableApprox = 0 != (CreateInteractionFlags_DisableApprox & flags) ? EBM_TRUE : EBM_FALSE;

   size_t cBinsMax = 0;

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
         UIntShared countBins;
         UIntShared defaultValSparse;
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

         cBinsMax = EbmMax(cBinsMax, cBins);

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

   if(ptrdiff_t { 0 } != cClasses && ptrdiff_t { 1 } != cClasses) {
      size_t cScores;
      if(0 != (CreateInteractionFlags_BinaryAsMulticlass & flags)) {
         cScores = cClasses < ptrdiff_t { 2 } ? size_t { 1 } : static_cast<size_t>(cClasses);
      } else {
         cScores = cClasses <= ptrdiff_t { 2 } ? size_t { 1 } : static_cast<size_t>(cClasses);
      }
      pInteractionCore->m_cScores = cScores;

      LOG_0(Trace_Info, "INFO InteractionCore::Create determining Objective");
      Config config;
      config.cOutputs = cScores;
      config.isDifferentialPrivacy = 0 != (CreateInteractionFlags_DifferentialPrivacy & flags) ? EBM_TRUE : EBM_FALSE;
      error = GetObjective(
         &config, 
         sObjective, 
         disableCompute,
         &pInteractionCore->m_objectiveCpu, 
         &pInteractionCore->m_objectiveSIMD
      );
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

      if(0 != cFeatures && 0 != cSamples) {
         if(EBM_FALSE != pInteractionCore->CheckTargets(cSamples, aTargets)) {
            LOG_0(Trace_Warning, "WARNING InteractionCore::Create invalid target value");
            return Error_ObjectiveIllegalTarget;
         }
         LOG_0(Trace_Info, "INFO InteractionCore::Create Targets verified");

         if(CheckInteractionRestrictions(pInteractionCore, &pInteractionCore->m_objectiveCpu, cBinsMax)) {
            LOG_0(Trace_Warning, "WARNING InteractionCore::Create cannot fit indexes in the cpu zone");
            return Error_IllegalParamVal;
         }
         if(0 != pInteractionCore->m_objectiveSIMD.m_cUIntBytes) {
            if(CheckInteractionRestrictions(pInteractionCore, &pInteractionCore->m_objectiveCpu, cBinsMax)) {
               FreeObjectiveWrapperInternals(&pInteractionCore->m_objectiveSIMD);
               InitializeObjectiveWrapperUnfailing(&pInteractionCore->m_objectiveSIMD);
            }
         }

         size_t cTrainingSamples;
         size_t cValidationSamples;
         error = Unbag(cSamples, aBag, &cTrainingSamples, &cValidationSamples);
         if(Error_None != error) {
            // already logged
            return error;
         }

         // if we have 32 bit floats or ints, then we need to break large datasets into smaller data subsets
         // because float32 values stop incrementing at 2^24 where the value 1 is below the threshold incrementing a float
         const bool bForceMultipleSubsets =
            sizeof(UIntSmall) == pInteractionCore->m_objectiveCpu.m_cUIntBytes ||
            sizeof(FloatSmall) == pInteractionCore->m_objectiveCpu.m_cFloatBytes ||
            sizeof(UIntSmall) == pInteractionCore->m_objectiveSIMD.m_cUIntBytes ||
            sizeof(FloatSmall) == pInteractionCore->m_objectiveSIMD.m_cFloatBytes;

         const bool bHessian = pInteractionCore->IsHessian();

         error = pInteractionCore->m_dataFrame.InitDataSetInteraction(
            bHessian,
            cScores,
            bForceMultipleSubsets ? k_cSubsetSamplesMax : SIZE_MAX,
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

         if(IsOverflowBinSize<FloatMain, UIntMain>(bHessian, cScores)) {
            LOG_0(Trace_Warning, "WARNING InteractionCore::Create IsOverflowBinSize overflow");
            return Error_OutOfMemory;
         }
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
      const size_t cScores = GetCountScores();
      EBM_ASSERT(1 <= cScores);

      size_t cBytesScoresMax = 0;
      size_t cBytesAllScoresMax = 0;
      size_t cBytesTempMax = 0;
      size_t cBytesTargetMax = 0;
      DataSubsetInteraction * pSubsetInit = GetDataSetInteraction()->GetSubsets();
      EBM_ASSERT(nullptr != pSubsetInit);
      EBM_ASSERT(1 <= GetDataSetInteraction()->GetCountSubsets());
      const DataSubsetInteraction * const pSubsetsEnd = pSubsetInit + GetDataSetInteraction()->GetCountSubsets();
      do {
         size_t cSamples = pSubsetInit->GetCountSamples();
         EBM_ASSERT(1 <= cSamples);

         EBM_ASSERT(0 == cSamples % pSubsetInit->GetObjectiveWrapper()->m_cSIMDPack);
         EBM_ASSERT(pSubsetInit->GetObjectiveWrapper()->m_cSIMDPack <= cSamples);
         if(IsMultiplyError(pSubsetInit->GetObjectiveWrapper()->m_cFloatBytes, cScores, cSamples)) {
            LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians IsMultiplyError(pSubsetInit->GetObjectiveWrapper()->m_cFloatBytes, cScores, cSamples)");
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

         if(size_t { 1 } != cScores) {
            void * const aMulticlassMidwayTemp = AlignedAlloc(cBytesTempMax);
            if(UNLIKELY(nullptr == aMulticlassMidwayTemp)) {
               LOG_0(Trace_Warning, "WARNING InteractionCore::InitializeInteractionGradientsAndHessians nullptr == aMulticlassMidwayTemp");
               error = Error_OutOfMemory;
               goto free_targets;
            }
            data.m_aMulticlassMidwayTemp = aMulticlassMidwayTemp;
         }

         const UIntShared * pTargetFrom = static_cast<const UIntShared *>(aTargetsFrom);

         const BagEbm * pSampleReplication = aBag;
         const double * pInitScoreFrom = aInitScores;
         BagEbm replication = 0;
         const double * pInitScoreFromOld = nullptr;
         UIntShared target;

         DataSubsetInteraction * pSubset = GetDataSetInteraction()->GetSubsets();
         do {
            EBM_ASSERT(1 <= pSubset->GetCountSamples());

            const size_t cSIMDPack = pSubset->GetObjectiveWrapper()->m_cSIMDPack;
            EBM_ASSERT(0 == pSubset->GetCountSamples() % cSIMDPack);

            void * pTargetTo = aTargetTo;
            void * pSampleScoreTo = aSampleScoreTo;
            const void * const pTargetToEnd = IndexByte(aTargetTo, pSubset->GetObjectiveWrapper()->m_cUIntBytes * pSubset->GetCountSamples());
            double initScore = 0.0;
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
                     // we also check that the number of classes can be converted to a ptrdiff_t and also a UIntMain
                     // so we do not need the runtime to check this
                     EBM_ASSERT(target < static_cast<UIntShared>(cClasses));
                  }

                  if(sizeof(UIntBig) == pSubset->GetObjectiveWrapper()->m_cUIntBytes) {
                     *reinterpret_cast<UIntBig *>(pTargetTo) = static_cast<UIntBig>(target);
                  } else {
                     EBM_ASSERT(sizeof(UIntSmall) == pSubset->GetObjectiveWrapper()->m_cUIntBytes);
                     *reinterpret_cast<UIntSmall *>(pTargetTo) = static_cast<UIntSmall>(target);
                  }
                  pTargetTo = IndexByte(pTargetTo, pSubset->GetObjectiveWrapper()->m_cUIntBytes);

                  size_t iScore = 0;
                  do {
                     if(nullptr != pInitScoreFromOld) {
                        initScore = pInitScoreFromOld[iScore];
                     }

                     if(sizeof(FloatBig) == pSubset->GetObjectiveWrapper()->m_cFloatBytes) {
                        reinterpret_cast<FloatBig *>(pSampleScoreTo)[iScore * cSIMDPack + iPartition] = static_cast<FloatBig>(initScore);
                     } else {
                        EBM_ASSERT(sizeof(FloatSmall) == pSubset->GetObjectiveWrapper()->m_cFloatBytes);
                        reinterpret_cast<FloatSmall *>(pSampleScoreTo)[iScore * cSIMDPack + iPartition] = static_cast<FloatSmall>(initScore);
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
            data.m_bDisableApprox = IsDisableApprox();
            data.m_bValidation = EBM_FALSE;
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

         const FloatShared * pTargetFrom = static_cast<const FloatShared *>(aTargetsFrom);

         const BagEbm * pSampleReplication = aBag;
         const double * pInitScoreFrom = aInitScores;
         BagEbm replication = 0;
         double initScore = 0.0;
         FloatShared target;

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

               if(sizeof(FloatBig) == pSubset->GetObjectiveWrapper()->m_cFloatBytes) {
                  *reinterpret_cast<FloatBig *>(pTargetTo) = static_cast<FloatBig>(target);
                  *reinterpret_cast<FloatBig *>(pSampleScoreTo) = static_cast<FloatBig>(initScore);
               } else {
                  EBM_ASSERT(sizeof(FloatSmall) == pSubset->GetObjectiveWrapper()->m_cFloatBytes);
                  *reinterpret_cast<FloatSmall *>(pTargetTo) = static_cast<FloatSmall>(target);
                  *reinterpret_cast<FloatSmall *>(pSampleScoreTo) = static_cast<FloatSmall>(initScore);
               }
               pTargetTo = IndexByte(pTargetTo, pSubset->GetObjectiveWrapper()->m_cFloatBytes);
               pSampleScoreTo = IndexByte(pSampleScoreTo, pSubset->GetObjectiveWrapper()->m_cFloatBytes);

               --replication;

            } while(pTargetToEnd != pTargetTo);

            EBM_ASSERT(1 == cScores);
            data.m_cScores = 1;
            data.m_cPack = k_cItemsPerBitPackNone;
            data.m_bHessianNeeded = IsHessian() ? EBM_TRUE : EBM_FALSE;
            data.m_bDisableApprox = IsDisableApprox();
            data.m_bValidation = EBM_FALSE;
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

         const FloatShared * pWeight = GetDataSetSharedWeight(pDataSetShared, 0);
         EBM_ASSERT(nullptr != pWeight);

         size_t cTotalScores = cScores;
         if(IsHessian()) {
            EBM_ASSERT(!IsMultiplyError(size_t { 2 }, cTotalScores)); // we are accessing allocated memory
            cTotalScores = cTotalScores << 1;
         }

         const BagEbm * pSampleReplication = aBag;
         BagEbm replication = 0;
         FloatShared weight;

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
                  if(sizeof(FloatBig) == pSubset->GetObjectiveWrapper()->m_cFloatBytes) {
                     const FloatBig weightConverted = static_cast<FloatBig>(weight);
                     do {
                        reinterpret_cast<FloatBig *>(pGradHess)[iScore * cSIMDPack + iPartition] *= weightConverted;
                        ++iScore;
                     } while(cTotalScores != iScore);
                  } else {
                     EBM_ASSERT(sizeof(FloatSmall) == pSubset->GetObjectiveWrapper()->m_cFloatBytes);
                     const FloatSmall weightConverted = static_cast<FloatSmall>(weight);
                     do {
                        reinterpret_cast<FloatSmall *>(pGradHess)[iScore * cSIMDPack + iPartition] *= weightConverted;
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
