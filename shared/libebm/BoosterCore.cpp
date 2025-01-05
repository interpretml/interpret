// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "pch.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <thread>

#include "logging.h" // EBM_ASSERT

#define ZONE_main
#include "zones.h"

#include "common.hpp" // IsConvertError, IsMultiplyError
#include "Bin.hpp" // IsOverflowBinSize

#include "ebm_internal.hpp"
#include "dataset_shared.hpp" // GetDataSetSharedHeader
#include "Tensor.hpp" // Tensor
#include "Feature.hpp" // Feature
#include "Term.hpp" // Term
#include "TreeNode.hpp" // IsOverflowTreeNodeSize
#include "SplitPosition.hpp" // IsOverflowSplitPositionSize
#include "BoosterCore.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class RandomDeterministic;

extern ErrorEbm Unbag(const size_t cSamples,
      const BagEbm* const aBag,
      size_t* const pcTrainingSamplesOut,
      size_t* const pcValidationSamplesOut);

NEVER_INLINE extern ErrorEbm GetObjective(const Config* const pConfig,
      const char* sObjective,
      const AccelerationFlags acceleration,
      ObjectiveWrapper* const pCpuObjectiveWrapperOut,
      ObjectiveWrapper* const pSIMDObjectiveWrapperOut) noexcept;

void BoosterCore::DeleteTensors(const size_t cTerms, Tensor** const apTensors) {
   LOG_0(Trace_Info, "Entered DeleteTensors");

   if(UNLIKELY(nullptr != apTensors)) {
      EBM_ASSERT(0 < cTerms);
      Tensor** ppTensor = apTensors;
      const Tensor* const* const ppTensorsEnd = &apTensors[cTerms];
      do {
         Tensor::Free(*ppTensor);
         ++ppTensor;
      } while(ppTensorsEnd != ppTensor);
      free(apTensors);
   }
   LOG_0(Trace_Info, "Exited DeleteTensors");
}

ErrorEbm BoosterCore::InitializeTensors(
      const size_t cTerms, const Term* const* const apTerms, const size_t cScores, Tensor*** papTensorsOut) {
   LOG_0(Trace_Info, "Entered InitializeTensors");

   EBM_ASSERT(1 <= cTerms);
   EBM_ASSERT(nullptr != apTerms);
   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(nullptr != papTensorsOut);
   EBM_ASSERT(nullptr == *papTensorsOut);

   ErrorEbm error;

   if(IsMultiplyError(sizeof(Tensor*), cTerms)) {
      LOG_0(Trace_Warning, "WARNING InitializeTensors IsMultiplyError(sizeof(Tensor *), cTerms)");
      return Error_OutOfMemory;
   }
   Tensor** const apTensors = static_cast<Tensor**>(malloc(sizeof(Tensor*) * cTerms));
   if(UNLIKELY(nullptr == apTensors)) {
      LOG_0(Trace_Warning, "WARNING InitializeTensors nullptr == apTensors");
      return Error_OutOfMemory;
   }

   Tensor** ppTensorInit = apTensors;
   const Tensor* const* const ppTensorsEnd = &apTensors[cTerms];
   do {
      *ppTensorInit = nullptr;
      ++ppTensorInit;
   } while(ppTensorsEnd != ppTensorInit);
   *papTensorsOut = apTensors; // transfer ownership for future deletion

   Tensor** ppTensor = apTensors;
   const Term* const* ppTerm = apTerms;
   do {
      const Term* const pTerm = *ppTerm;
      if(size_t{0} != pTerm->GetCountTensorBins()) {
         // if there are any dimensions with features having 0 bins then do not allocate the tensor
         // since it will have 0 scores

         Tensor* const pTensors = Tensor::Allocate(pTerm->GetCountDimensions(), cScores);
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

   m_trainingSet.DestructDataSetBoosting(m_cTerms, m_cInnerBags);
   m_validationSet.DestructDataSetBoosting(m_cTerms, 0);

   Term::FreeTerms(m_cTerms, m_apTerms);

   free(m_aFeatures);

   DeleteTensors(m_cTerms, m_apCurrentTermTensors);
   DeleteTensors(m_cTerms, m_apBestTermTensors);

   FreeObjectiveWrapperInternals(&m_objectiveCpu);
   FreeObjectiveWrapperInternals(&m_objectiveSIMD);
};

void BoosterCore::Free(BoosterCore* const pBoosterCore) {
   LOG_0(Trace_Info, "Entered BoosterCore::Free");
   if(nullptr != pBoosterCore) {
      // for reference counting in general, a release is needed during the decrement and aquire is needed if freeing
      // https://www.boost.org/doc/libs/1_59_0/doc/html/atomic/usage_examples.html
      // We need to ensure that writes on this thread are not allowed to be re-ordered to a point below the
      // decrement because if we happened to decrement to 2, and then get interrupted, and annother thread
      // decremented to 1 after us, we don't want our unclean writes to memory to be visible in the other thread
      // so we use memory_order_release on the decrement.
      if(size_t{1} == pBoosterCore->m_REFERENCE_COUNT.fetch_sub(1, std::memory_order_release)) {
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

template<typename TUInt>
static bool CheckBoosterRestrictionsInternal(const BoosterCore* const pBoosterCore,
      const ObjectiveWrapper* const pObjectiveWrapper,
      const size_t cTensorBinsMax) {
   EBM_ASSERT(nullptr != pBoosterCore);
   EBM_ASSERT(nullptr != pObjectiveWrapper);

   const size_t cScores = pBoosterCore->GetCountScores();

   if(IsConvertError<TUInt>(cScores)) {
      // restriction from LogLossMulticlassObjective.hpp
      // we cast the cScores value into a SIMD type before later multiplying.  If cScores was 1 + the max SIMD type
      // value then it could overflow back to 0 before the multiplication. Normally this would be very rare, but
      // we need to consider adversarial inputs to make this crash.

      return true;
   }

   const bool bHessian = EBM_FALSE != pObjectiveWrapper->m_bObjectiveHasHessian;

   // In BinSumsBoosting we calculate the BinSize value and put it into a SIMD pack, so it needs to fit
   size_t cBytes;
   if(sizeof(FloatBig) == pObjectiveWrapper->m_cFloatBytes) {
      if(IsOverflowBinSize<FloatBig, TUInt>(false, false, bHessian, cScores)) {
         return true;
      }
      cBytes = GetBinSize<FloatBig, TUInt>(false, false, bHessian, cScores);
   } else {
      EBM_ASSERT(sizeof(FloatSmall) == pObjectiveWrapper->m_cFloatBytes);
      if(IsOverflowBinSize<FloatSmall, TUInt>(false, false, bHessian, cScores)) {
         return true;
      }
      cBytes = GetBinSize<FloatSmall, TUInt>(false, false, bHessian, cScores);
   }

   if(size_t{0} != cTensorBinsMax) {
      // cTensorBins can only be 0 if cSamples or cTerms is 0
      if(IsMultiplyError(cTensorBinsMax, EbmMax(cBytes, cScores))) {
         return true;
      }
      if(IsConvertError<typename std::make_signed<TUInt>::type>(cTensorBinsMax * cScores - size_t{1})) {
         // In all objectives we take the binned feature index and use it to lookup the score update in the update
         // tensor. The lookup indexes are packed together in an array of SIMDable integers, so obviously the SIMDable
         // integer needs to be large enough to hold the maximum feature index.
         //
         // Additionally, we use a SIMD gather operations in the objectives to load from the score update tensor, which
         // use signed indexes, which means we need to restrict ourselves to the range of positive values.

         return true;
      }
      cBytes *= cTensorBinsMax;
      EBM_ASSERT(1 <= cBytes); // since cTensorBinsMax is non-zero
      if(IsConvertError<TUInt>(cBytes - 1)) {
         // In BinSumsBoosting we use the SIMD pack to hold an index to memory, so we need to be able to hold
         // the entire fast bin tensor
         return true;
      }
   }

   if(size_t{1} != cScores) {
      // TODO: we currently index into the gradient array using the target, but the gradient array is also
      // layed out per-SIMD pack.  Once we sort the dataset by the target we'll be able to use non-random
      // indexing to fetch all the sample targets simultaneously, and we'll no longer need this indexing
      size_t cIndexes = cScores;
      if(bHessian) {
         if(IsMultiplyError(size_t{2}, cIndexes)) {
            return true;
         }
         cIndexes <<= 1;
      }
      if(IsMultiplyError(cIndexes, pObjectiveWrapper->m_cSIMDPack)) {
         return true;
      }
      // restriction from LogLossMulticlassObjective.hpp
      // we use the target value to index into the temp exp array and adjust the target gradient
      if(IsConvertError<typename std::make_signed<TUInt>::type>(
               cIndexes * pObjectiveWrapper->m_cSIMDPack - size_t{1})) {
         return true;
      }
   }

   return false;
}

static bool CheckBoosterRestrictions(const BoosterCore* const pBoosterCore,
      const ObjectiveWrapper* const pObjectiveWrapper,
      const size_t cTensorBinsMax) {
   EBM_ASSERT(nullptr != pObjectiveWrapper);
   if(sizeof(UIntBig) == pObjectiveWrapper->m_cUIntBytes) {
      return CheckBoosterRestrictionsInternal<UIntBig>(pBoosterCore, pObjectiveWrapper, cTensorBinsMax);
   } else {
      EBM_ASSERT(sizeof(UIntSmall) == pObjectiveWrapper->m_cUIntBytes);
      return CheckBoosterRestrictionsInternal<UIntSmall>(pBoosterCore, pObjectiveWrapper, cTensorBinsMax);
   }
}

// static int g_TODO_removeThisThreadTest = 0;
// void TODO_removeThisThreadTest() {
//    g_TODO_removeThisThreadTest = 1;
// }

ErrorEbm BoosterCore::Create(void* const rng,
      const size_t cTerms,
      const size_t cInnerBags,
      const double* const experimentalParams,
      const IntEbm* const acTermDimensions,
      const IntEbm* const aiTermFeatures,
      const unsigned char* const pDataSetShared,
      const double* const aIntercept,
      const BagEbm* const aBag,
      const double* const aInitScores,
      const CreateBoosterFlags flags,
      const AccelerationFlags acceleration,
      const char* const sObjective,
      BoosterCore** const ppBoosterCoreOut) {
   // experimentalParams isn't used by default.  It's meant to provide an easy way for python or other higher
   // level languages to pass EXPERIMENTAL temporary parameters easily to the C++ code.
   UNUSED(experimentalParams);

   LOG_0(Trace_Info, "Entered BoosterCore::Create");

   EBM_ASSERT(nullptr != ppBoosterCoreOut);
   EBM_ASSERT(nullptr == *ppBoosterCoreOut);
   EBM_ASSERT(nullptr != pDataSetShared);

   ErrorEbm error;

   // try {
   //    // TODO: eliminate this code I added to test that threads are available on the majority of our systems
   //    std::thread testThread(TODO_removeThisThreadTest);
   //    testThread.join();
   //    if(0 == g_TODO_removeThisThreadTest) {
   //       LOG_0(Trace_Warning, "WARNING BoosterCore::Create thread not started");
   //       return Error_UnexpectedInternal;
   //    }
   // } catch(const std::bad_alloc &) {
   //    LOG_0(Trace_Warning, "WARNING BoosterCore::Create thread start out of memory");
   //    return Error_OutOfMemory;
   // } catch(...) {
   //    // the C++ standard doesn't really seem to say what kind of exceptions we'd get for various errors, so
   //    // about the best we can do is catch(...) since the exact exceptions seem to be implementation specific
   //    LOG_0(Trace_Warning, "WARNING BoosterCore::Create thread start failed");
   //    return Error_ThreadStartFailed;
   // }
   // LOG_0(Trace_Info, "INFO BoosterCore::Create thread started");

   BoosterCore* pBoosterCore;
   try {
      pBoosterCore = new BoosterCore();
   } catch(const std::bad_alloc&) {
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

   pBoosterCore->m_bUseApprox = CreateBoosterFlags_UseApprox & flags ? EBM_TRUE : EBM_FALSE;

   UIntShared countSamples;
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
   if(IsConvertError<UIntMain>(countSamples)) {
      LOG_0(Trace_Error, "ERROR BoosterCore::Create IsConvertError<UIntMain>(countSamples)");
      return Error_IllegalParamVal;
   }
   size_t cSamples = static_cast<size_t>(countSamples);

   if(size_t{1} < cWeights) {
      LOG_0(Trace_Warning, "WARNING BoosterCore::Create size_t { 1 } < cWeights");
      return Error_IllegalParamVal;
   }
   if(size_t{1} != cTargets) {
      LOG_0(Trace_Warning, "WARNING BoosterCore::Create 1 != cTargets");
      return Error_IllegalParamVal;
   }

   LOG_0(Trace_Info, "BoosterCore::Create starting feature processing");
   if(0 != cFeatures) {
      pBoosterCore->m_cFeatures = cFeatures;

      if(IsMultiplyError(sizeof(FeatureBoosting), cFeatures)) {
         LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsMultiplyError(sizeof(Feature), cFeatures)");
         return Error_OutOfMemory;
      }
      FeatureBoosting* const aFeatures = static_cast<FeatureBoosting*>(malloc(sizeof(FeatureBoosting) * cFeatures));
      if(nullptr == aFeatures) {
         LOG_0(Trace_Warning, "WARNING BoosterCore::Create nullptr == aFeatures");
         return Error_OutOfMemory;
      }
      pBoosterCore->m_aFeatures = aFeatures;

      size_t iFeatureInitialize = size_t{0};
      do {
         bool bMissing;
         bool bUnseen;
         bool bNominal;
         bool bSparse;
         UIntShared countBins;
         UIntShared defaultValSparse;
         size_t cNonDefaultsSparse;
         GetDataSetSharedFeature(pDataSetShared,
               iFeatureInitialize,
               &bMissing,
               &bUnseen,
               &bNominal,
               &bSparse,
               &countBins,
               &defaultValSparse,
               &cNonDefaultsSparse);
         EBM_ASSERT(!bSparse); // we do not handle yet
         if(IsConvertError<size_t>(countBins)) {
            LOG_0(Trace_Error, "ERROR BoosterCore::Create IsConvertError<size_t>(countBins)");
            return Error_IllegalParamVal;
         }
         if(IsConvertError<UIntSplit>(countBins)) {
            LOG_0(Trace_Error, "ERROR BoosterCore::Create IsConvertError<UIntSplit>(countBins)");
            return Error_IllegalParamVal;
         }
         const size_t cBins = static_cast<size_t>(countBins);
         if(size_t{0} == cBins) {
            if(size_t{0} != cSamples) {
               LOG_0(Trace_Error, "ERROR BoosterCore::Create countBins cannot be zero unless there are zero samples");
               return Error_IllegalParamVal;
            }

            // we can handle 0 == cBins even though that's a degenerate case that shouldn't be boosted on.  0 bins
            // can only occur if there were zero training and zero validation cases since the
            // features would require a value, even if it was 0.
            LOG_0(Trace_Info, "INFO BoosterCore::Create feature with 0 values");
         } else if(size_t{1} == cBins) {
            // Dimensions with 1 bin don't contribute anything to the model since they always have the same value, but
            // the user can specify interactions, so we handle them anyways in a consistent way by boosting on them
            LOG_0(Trace_Info, "INFO BoosterCore::Create feature with 1 value");
         }
         aFeatures[iFeatureInitialize].Initialize(cBins, bMissing, bUnseen, bNominal);

         ++iFeatureInitialize;
      } while(cFeatures != iFeatureInitialize);
   }
   LOG_0(Trace_Info, "BoosterCore::Create done feature processing");

   size_t cTensorBinsMax = 1;
   size_t cMainBinsMax = 1;
   size_t cSingleDimensionBinsMax = 0;

   LOG_0(Trace_Info, "BoosterCore::Create starting term processing");
   if(0 != cTerms) {
      pBoosterCore->m_cTerms = cTerms;
      pBoosterCore->m_apTerms = Term::AllocateTerms(cTerms);
      if(UNLIKELY(nullptr == pBoosterCore->m_apTerms)) {
         LOG_0(Trace_Warning, "WARNING BoosterCore::Create 0 != m_cTerms && nullptr == m_apTerms");
         return Error_OutOfMemory;
      }

      const IntEbm* piTermFeature = aiTermFeatures;
      size_t iTerm = 0;
      do {
         const IntEbm countDimensions = acTermDimensions[iTerm];
         if(countDimensions < IntEbm{0}) {
            LOG_0(Trace_Error, "ERROR BoosterCore::Create countDimensions cannot be negative");
            return Error_IllegalParamVal;
         }
         if(IntEbm{k_cDimensionsMax} < countDimensions) {
            LOG_0(Trace_Warning,
                  "WARNING BoosterCore::Create countDimensions too large and would cause out of memory condition");
            return Error_OutOfMemory;
         }
         const size_t cDimensions = static_cast<size_t>(countDimensions);
         Term* const pTerm = Term::Allocate(cDimensions);
         if(nullptr == pTerm) {
            LOG_0(Trace_Warning, "WARNING BoosterCore::Create nullptr == pTerm");
            return Error_OutOfMemory;
         }
         // assign our pointer directly to our array right now so that we can't loose the memory if we decide to exit
         // due to an error below
         pBoosterCore->m_apTerms[iTerm] = pTerm;

         pTerm->SetCountAuxillaryBins(0); // we only use these for pairs, so otherwise it gets left as zero

         size_t cAuxillaryBinsForBuildFastTotals = 0;
         size_t cRealDimensions = 0;
         int cBitsRequiredMin = 0;
         size_t cTensorBins = 1;
         if(UNLIKELY(0 == cDimensions)) {
            LOG_0(Trace_Info, "INFO BoosterCore::Create empty term");
         } else {
            if(nullptr == piTermFeature) {
               LOG_0(Trace_Error,
                     "ERROR BoosterCore::Create aiTermFeatures cannot be NULL when there are Terms with non-zero "
                     "numbers of features");
               return Error_IllegalParamVal;
            }
            size_t cSingleDimensionBins = 0;
            TermFeature* pTermFeature = pTerm->GetTermFeatures();
            const TermFeature* const pTermFeaturesEnd = &pTermFeature[cDimensions];
            // TODO: Ideally we would flip our input dimensions so that we're aligned with the output ordering
            //       and thus not need a transpose when transfering data to the caller. We're doing it this way
            //       for now to test the transpose ability and also to maintain the same results as before for
            //       comparison
            size_t iTranspose = cDimensions - 1;
            do {
               const IntEbm indexFeature = *piTermFeature;
               if(indexFeature < IntEbm{0}) {
                  LOG_0(Trace_Error, "ERROR BoosterCore::Create aiTermFeatures value cannot be negative");
                  return Error_IllegalParamVal;
               }
               if(IsConvertError<size_t>(indexFeature)) {
                  LOG_0(Trace_Error, "ERROR BoosterCore::Create aiTermFeatures value too big to reference memory");
                  return Error_IllegalParamVal;
               }
               const size_t iFeature = static_cast<size_t>(indexFeature);

               if(cFeatures <= iFeature) {
                  LOG_0(Trace_Error,
                        "ERROR BoosterCore::Create aiTermFeatures value must be less than the number of features");
                  return Error_IllegalParamVal;
               }

               EBM_ASSERT(1 <= cFeatures); // since our iFeature is valid and index 0 would mean cFeatures == 1
               EBM_ASSERT(nullptr != pBoosterCore->m_aFeatures);

               // Clang does not seems to understand that iFeature is bound to the legal
               // range of m_aFeatures through the check "cFeatures <= iFeature" above
               StopClangAnalysis();

               const FeatureBoosting* const pInputFeature = &pBoosterCore->m_aFeatures[iFeature];
               pTermFeature->m_pFeature = pInputFeature;
               pTermFeature->m_cStride = cTensorBins;
               pTermFeature->m_iTranspose = iTranspose; // TODO: no tranposition yet, but move it from python to C

               const size_t cBins = pInputFeature->GetCountBins();
               if(LIKELY(size_t{1} < cBins)) {
                  // if we have only 1 bin, then we can eliminate the feature from consideration since the resulting
                  // tensor loses one dimension but is otherwise indistinquishable from the original data
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
                  LOG_0(Trace_Info, "INFO BoosterCore::Create term with no useful features");
               }
               cTensorBins *= cBins;
               // same reasoning as above: cAuxillaryBinsForBuildFastTotals grows slower than cTensorBins
               EBM_ASSERT(0 == cTensorBins || cAuxillaryBinsForBuildFastTotals < cTensorBins);

               --iTranspose;
               ++piTermFeature;
               ++pTermFeature;
            } while(pTermFeaturesEnd != pTermFeature);

            cTensorBinsMax = EbmMax(cTensorBinsMax, cTensorBins);
            size_t cTotalMainBins = cTensorBins;
            if(LIKELY(size_t{1} < cTensorBins)) {
               EBM_ASSERT(1 <= cRealDimensions);

               cBitsRequiredMin = CountBitsRequired(cTensorBins - size_t{1});
               EBM_ASSERT(1 <= cBitsRequiredMin); // 1 < cTensorBins otherwise we'd have filtered it out above
               EBM_ASSERT(cBitsRequiredMin <= COUNT_BITS(size_t));

               if(size_t{1} == cRealDimensions) {
                  cSingleDimensionBinsMax = EbmMax(cSingleDimensionBinsMax, cSingleDimensionBins);
               } else {
                  // we only use AuxillaryBins for pairs.  We wouldn't use them for random pairs, but we
                  // don't know yet if the caller will set the random boosting flag on all pairs, so allocate it

                  // we need to reserve 4 PAST the pointer we pass into SweepMultiDimensional!!!!.  We pass in index 20
                  // at max, so we need 24
                  static constexpr size_t cAuxillaryBinsForSplitting = 24;
                  const size_t cAuxillaryBins = EbmMax(cAuxillaryBinsForBuildFastTotals, cAuxillaryBinsForSplitting);
                  pTerm->SetCountAuxillaryBins(cAuxillaryBins);

                  if(IsAddError(cTensorBins, cAuxillaryBins)) {
                     LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsAddError(cTensorBins, cAuxillaryBins)");
                     return Error_OutOfMemory;
                  }
                  cTotalMainBins += cAuxillaryBins;
               }
            } else {
               EBM_ASSERT(0 == cRealDimensions);
            }
            cMainBinsMax = EbmMax(cMainBinsMax, cTotalMainBins);
         }
         pTerm->SetCountRealDimensions(cRealDimensions);
         pTerm->SetBitsRequiredMin(cBitsRequiredMin);
         pTerm->SetCountTensorBins(cTensorBins);

         ++iTerm;
      } while(iTerm < cTerms);
   }
   LOG_0(Trace_Info, "BoosterCore::Create finished term processing");

   ptrdiff_t cClasses;
   const void* const aTargets = GetDataSetSharedTarget(pDataSetShared, 0, &cClasses);
   if(nullptr == aTargets) {
      LOG_0(Trace_Warning, "WARNING BoosterCore::Create cClasses cannot fit into ptrdiff_t");
      return Error_IllegalParamVal;
   }

   // having 1 class means that all predictions are perfect. In the C interface we reduce this into having 0 scores,
   // which means that we do not write anything to our upper level callers, and we don't need a bunch of things
   // since they have zero memory allocated to them. Having 0 classes means there are also 0 samples.
   if(ptrdiff_t{0} != cClasses && ptrdiff_t{1} != cClasses) {
      size_t cScores;
      if(CreateBoosterFlags_BinaryAsMulticlass & flags) {
         cScores = cClasses < ptrdiff_t{Task_BinaryClassification} ? size_t{1} : static_cast<size_t>(cClasses);
      } else {
         cScores = cClasses <= ptrdiff_t{Task_BinaryClassification} ? size_t{1} : static_cast<size_t>(cClasses);
      }
      pBoosterCore->m_cScores = cScores;

      LOG_0(Trace_Info, "INFO BoosterCore::Create determining Objective");
      Config config;
      config.cOutputs = cScores;
      config.isDifferentialPrivacy = CreateBoosterFlags_DifferentialPrivacy & flags ? EBM_TRUE : EBM_FALSE;
      error = GetObjective(
            &config, sObjective, acceleration, &pBoosterCore->m_objectiveCpu, &pBoosterCore->m_objectiveSIMD);
      if(Error_None != error) {
         // already logged
         return error;
      }
      LOG_0(Trace_Info, "INFO BoosterCore::Create Objective determined");

      const TaskEbm task = IdentifyTask(pBoosterCore->m_objectiveCpu.m_linkFunction);
      if(ptrdiff_t{Task_GeneralClassification} <= cClasses) {
         if(task < Task_GeneralClassification) {
            LOG_0(Trace_Error, "ERROR BoosterCore::Create mismatch in objective class model type");
            return Error_IllegalParamVal;
         }
      } else {
         if(Task_Regression != task) {
            LOG_0(Trace_Error, "ERROR BoosterCore::Create mismatch in objective class model type");
            return Error_IllegalParamVal;
         }
      }

      if(0 != cSamples) {
         if(EBM_FALSE != pBoosterCore->CheckTargets(cSamples, aTargets)) {
            LOG_0(Trace_Warning, "WARNING BoosterCore::Create invalid target value");
            return Error_ObjectiveIllegalTarget;
         }
         LOG_0(Trace_Info, "INFO BoosterCore::Create Targets verified");

         if(CheckBoosterRestrictions(pBoosterCore, &pBoosterCore->m_objectiveCpu, cTensorBinsMax)) {
            LOG_0(Trace_Warning, "WARNING BoosterCore::Create cannot fit indexes in the cpu zone");
            return Error_IllegalParamVal;
         }
         if(0 != pBoosterCore->m_objectiveSIMD.m_cUIntBytes) {
            if(CheckBoosterRestrictions(pBoosterCore, &pBoosterCore->m_objectiveSIMD, cTensorBinsMax)) {
               FreeObjectiveWrapperInternals(&pBoosterCore->m_objectiveSIMD);
               InitializeObjectiveWrapperUnfailing(&pBoosterCore->m_objectiveSIMD);
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
         // because float32 values stop incrementing at 2^24 where the value 1 is below the threshold incrementing a
         // float
         const bool bForceMultipleSubsets = sizeof(UIntSmall) == pBoosterCore->m_objectiveCpu.m_cUIntBytes ||
               sizeof(FloatSmall) == pBoosterCore->m_objectiveCpu.m_cFloatBytes ||
               sizeof(UIntSmall) == pBoosterCore->m_objectiveSIMD.m_cUIntBytes ||
               sizeof(FloatSmall) == pBoosterCore->m_objectiveSIMD.m_cFloatBytes;

         const bool bHessian = pBoosterCore->IsHessian();

         pBoosterCore->m_cInnerBags = cInnerBags; // this is used to destruct m_trainingSet, so store it first
         error = pBoosterCore->m_trainingSet.InitDataSetBoosting(true,
               bHessian,
               !pBoosterCore->IsRmse(),
               !pBoosterCore->IsRmse(),
               true,
               rng,
               cScores,
               bForceMultipleSubsets ? k_cSubsetSamplesMax : SIZE_MAX,
               &pBoosterCore->m_objectiveCpu,
               &pBoosterCore->m_objectiveSIMD,
               pDataSetShared,
               aIntercept,
               BagEbm{1},
               cSamples,
               aBag,
               aInitScores,
               cTrainingSamples,
               cInnerBags,
               cWeights,
               cTerms,
               pBoosterCore->m_apTerms,
               aiTermFeatures);
         if(Error_None != error) {
            return error;
         }

         error = pBoosterCore->m_validationSet.InitDataSetBoosting(pBoosterCore->IsRmse(),
               false,
               !pBoosterCore->IsRmse(),
               !pBoosterCore->IsRmse(),
               false,
               rng,
               cScores,
               bForceMultipleSubsets ? k_cSubsetSamplesMax : SIZE_MAX,
               &pBoosterCore->m_objectiveCpu,
               &pBoosterCore->m_objectiveSIMD,
               pDataSetShared,
               aIntercept,
               BagEbm{-1},
               cSamples,
               aBag,
               aInitScores,
               cValidationSamples,
               0,
               cWeights,
               cTerms,
               pBoosterCore->m_apTerms,
               aiTermFeatures);
         if(Error_None != error) {
            return error;
         }

         size_t cBytesPerFastBinMax = 0;
#if 0 < HESSIAN_PARALLEL_BIN_BYTES_MAX || 0 < GRADIENT_PARALLEL_BIN_BYTES_MAX || 0 < MULTISCORE_PARALLEL_BIN_BYTES_MAX
         size_t cBytesParallelMax;
         if(bHessian) {
            if(size_t {1} == cScores) {
               // the caller can specify gradient boosting as an option for an objective with a hessian
               cBytesParallelMax = EbmMax(HESSIAN_PARALLEL_BIN_BYTES_MAX, GRADIENT_PARALLEL_BIN_BYTES_MAX);
            } else {
               cBytesParallelMax = MULTISCORE_PARALLEL_BIN_BYTES_MAX;
            }
         } else {
            if(size_t {1} == cScores) {
               cBytesParallelMax = GRADIENT_PARALLEL_BIN_BYTES_MAX;
            } else {
               // don't allow parallel gradient multiclass boosting. multiclass should be hessian boosting
               cBytesParallelMax = 0;
            }
         }
         size_t cBytesParallelBoostTrainingMax = 0;
#endif

         if(0 != cTrainingSamples) {
            DataSubsetBoosting* pSubset = pBoosterCore->GetTrainingSet()->GetSubsets();
            const DataSubsetBoosting* const pSubsetsEnd = pSubset + pBoosterCore->GetTrainingSet()->GetCountSubsets();
            do {
               size_t cBytesPerFastBin;
               if(sizeof(UIntBig) == pSubset->GetObjectiveWrapper()->m_cUIntBytes) {
                  if(sizeof(FloatBig) == pSubset->GetObjectiveWrapper()->m_cFloatBytes) {
                     cBytesPerFastBin = GetBinSize<FloatBig, UIntBig>(false, false, bHessian, cScores);
                  } else {
                     EBM_ASSERT(sizeof(FloatSmall) == pSubset->GetObjectiveWrapper()->m_cFloatBytes);
                     cBytesPerFastBin = GetBinSize<FloatSmall, UIntBig>(false, false, bHessian, cScores);
                  }
               } else {
                  EBM_ASSERT(sizeof(UIntSmall) == pSubset->GetObjectiveWrapper()->m_cUIntBytes);
                  if(sizeof(FloatBig) == pSubset->GetObjectiveWrapper()->m_cFloatBytes) {
                     cBytesPerFastBin = GetBinSize<FloatBig, UIntSmall>(false, false, bHessian, cScores);
                  } else {
                     EBM_ASSERT(sizeof(FloatSmall) == pSubset->GetObjectiveWrapper()->m_cFloatBytes);
                     cBytesPerFastBin = GetBinSize<FloatSmall, UIntSmall>(false, false, bHessian, cScores);
                  }
               }
               cBytesPerFastBinMax = EbmMax(cBytesPerFastBinMax, cBytesPerFastBin);

#if 0 < HESSIAN_PARALLEL_BIN_BYTES_MAX || 0 < GRADIENT_PARALLEL_BIN_BYTES_MAX || 0 < MULTISCORE_PARALLEL_BIN_BYTES_MAX
               if(1 != pSubset->GetObjectiveWrapper()->m_cSIMDPack) {
                  if(IsMultiplyError(
                           cBytesPerFastBin, cTensorBinsMax, pSubset->GetObjectiveWrapper()->m_cSIMDPack)) {
                     cBytesParallelBoostTrainingMax = cBytesParallelMax;
                  } else {
                     size_t cBytesParallelBoostTraining =
                           cBytesPerFastBin * cTensorBinsMax * pSubset->GetObjectiveWrapper()->m_cSIMDPack;
                     cBytesParallelBoostTraining = EbmMin(cBytesParallelBoostTraining, cBytesParallelMax);

                     cBytesParallelBoostTrainingMax =
                           EbmMax(cBytesParallelBoostTrainingMax, cBytesParallelBoostTraining);
                  }
               }
#endif

               ++pSubset;
            } while(pSubsetsEnd != pSubset);
         }

         if(0 != cValidationSamples) {
            DataSubsetBoosting* pSubset = pBoosterCore->GetValidationSet()->GetSubsets();
            const DataSubsetBoosting* const pSubsetsEnd = pSubset + pBoosterCore->GetValidationSet()->GetCountSubsets();
            do {
               size_t cBytesPerFastBin;
               if(sizeof(UIntBig) == pSubset->GetObjectiveWrapper()->m_cUIntBytes) {
                  if(sizeof(FloatBig) == pSubset->GetObjectiveWrapper()->m_cFloatBytes) {
                     cBytesPerFastBin = GetBinSize<FloatBig, UIntBig>(false, false, bHessian, cScores);
                  } else {
                     EBM_ASSERT(sizeof(FloatSmall) == pSubset->GetObjectiveWrapper()->m_cFloatBytes);
                     cBytesPerFastBin = GetBinSize<FloatSmall, UIntBig>(false, false, bHessian, cScores);
                  }
               } else {
                  EBM_ASSERT(sizeof(UIntSmall) == pSubset->GetObjectiveWrapper()->m_cUIntBytes);
                  if(sizeof(FloatBig) == pSubset->GetObjectiveWrapper()->m_cFloatBytes) {
                     cBytesPerFastBin = GetBinSize<FloatBig, UIntSmall>(false, false, bHessian, cScores);
                  } else {
                     EBM_ASSERT(sizeof(FloatSmall) == pSubset->GetObjectiveWrapper()->m_cFloatBytes);
                     cBytesPerFastBin = GetBinSize<FloatSmall, UIntSmall>(false, false, bHessian, cScores);
                  }
               }
               cBytesPerFastBinMax = EbmMax(cBytesPerFastBinMax, cBytesPerFastBin);
               ++pSubset;
            } while(pSubsetsEnd != pSubset);
         }

         if(IsMultiplyError(cBytesPerFastBinMax, cTensorBinsMax)) {
            LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsMultiplyError(cBytesPerFastBinMax, cTensorBinsMax)");
            return Error_OutOfMemory;
         }
         cBytesPerFastBinMax *= cTensorBinsMax;
#if 0 < HESSIAN_PARALLEL_BIN_BYTES_MAX || 0 < GRADIENT_PARALLEL_BIN_BYTES_MAX || 0 < MULTISCORE_PARALLEL_BIN_BYTES_MAX
         cBytesPerFastBinMax = EbmMax(cBytesParallelBoostTrainingMax, cBytesPerFastBinMax);
#endif
         pBoosterCore->m_cBytesFastBins = cBytesPerFastBinMax;

         if(IsOverflowBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores)) {
            LOG_0(Trace_Warning, "WARNING BoosterCore::Create bin size overflow");
            return Error_OutOfMemory;
         }

         const size_t cBytesPerMainBin = GetBinSize<FloatMain, UIntMain>(true, true, bHessian, cScores);
         if(IsMultiplyError(cBytesPerMainBin, cMainBinsMax)) {
            LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsMultiplyError(cBytesPerMainBin, cMainBinsMax)");
            return Error_OutOfMemory;
         }
         size_t cBytesMainBins = cBytesPerMainBin * cMainBinsMax;

         if(0 != cSingleDimensionBinsMax) {
            if(IsAddError(cBytesPerMainBin, sizeof(void*))) {
               LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsAddError(cBytesPerMainBin, sizeof(void*))");
               return Error_OutOfMemory;
            }
            const size_t cBytesMainBinPlusPointer = cBytesPerMainBin + sizeof(void*);
            if(IsMultiplyError(cBytesMainBinPlusPointer, cSingleDimensionBinsMax)) {
               LOG_0(Trace_Warning,
                     "WARNING BoosterCore::Create IsMultiplyError(cBytesMainBinPlusPointer, cSingleDimensionBinsMax)");
               return Error_OutOfMemory;
            }
            // we also allocate enough space to create an additional array of pointers
            cBytesMainBins = EbmMax(cBytesMainBins, cBytesMainBinPlusPointer * cSingleDimensionBinsMax);

            if(IsOverflowTreeNodeSize(bHessian, cScores) || IsOverflowSplitPositionSize(bHessian, cScores)) {
               LOG_0(Trace_Warning, "WARNING BoosterCore::Create bin tracking size overflow");
               return Error_OutOfMemory;
            }

            const size_t cSingleDimensionSplitsMax = cSingleDimensionBinsMax - 1;
            const size_t cBytesPerSplitPosition = GetSplitPositionSize(bHessian, cScores);
            // for missing values we can traverse in reverse and examine 2x the number of locations
            if(IsMultiplyError(size_t{2}, cBytesPerSplitPosition, cSingleDimensionSplitsMax)) {
               LOG_0(Trace_Warning,
                     "WARNING BoosterCore::Create IsMultiplyError(cBytesPerSplitPosition, "
                     "cSingleDimensionSplitsMax)");
               return Error_OutOfMemory;
            }
            // TODO : someday add equal gain multidimensional randomized picking.  I think for that we should
            // generate
            //        random numbers as we find equal gains, so we won't need this memory if we do that
            pBoosterCore->m_cBytesSplitPositions = size_t{2} * cBytesPerSplitPosition * cSingleDimensionSplitsMax;

            // If we have N bins, then we can have at most N - 1 splits.
            // At maximum if all splits are made, then we'll have a tree with N - 1 nodes.
            // Each node will contain a the total gradient sums of their left and right sides
            // Each of the N bins will also have a leaf in the tree, which will also consume a TreeNode structure
            // because each split needs to preserve the gradient sums of its left and right sides, which in this
            // case are individual bins.
            // So, in total we consume N + N - 1 TreeNodes

            if(IsAddError(cSingleDimensionSplitsMax, cSingleDimensionBinsMax)) {
               LOG_0(Trace_Warning,
                     "WARNING BoosterCore::Create IsAddError(cSingleDimensionSplitsMax, cSingleDimensionBinsMax)");
               return Error_OutOfMemory;
            }
            const size_t cTreeNodes = cSingleDimensionSplitsMax + cSingleDimensionBinsMax;

            const size_t cBytesPerTreeNode = GetTreeNodeSize(bHessian, cScores);
            if(IsMultiplyError(cBytesPerTreeNode, cTreeNodes)) {
               LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsMultiplyError(cBytesPerTreeNode, cTreeNodes)");
               return Error_OutOfMemory;
            }
            pBoosterCore->m_cBytesTreeNodes = cTreeNodes * cBytesPerTreeNode;
         } else {
            EBM_ASSERT(0 == pBoosterCore->m_cBytesSplitPositions);
            EBM_ASSERT(0 == pBoosterCore->m_cBytesTreeNodes);
         }
         pBoosterCore->m_cBytesMainBins = cBytesMainBins;
      }
      if(0 != cTerms) {
         error = InitializeTensors(cTerms, pBoosterCore->m_apTerms, cScores, &pBoosterCore->m_apCurrentTermTensors);
         if(Error_None != error) {
            return error;
         }
         error = InitializeTensors(cTerms, pBoosterCore->m_apTerms, cScores, &pBoosterCore->m_apBestTermTensors);
         if(Error_None != error) {
            return error;
         }
      }
   }

   LOG_0(Trace_Info, "Exited BoosterCore::Create");
   return Error_None;
}

ErrorEbm BoosterCore::InitializeBoosterGradientsAndHessians(
      void* const aMulticlassMidwayTemp, FloatScore* const aUpdateScores) {
   DataSetBoosting* const pDataSet = GetTrainingSet();
   if(size_t{0} != pDataSet->GetCountSamples()) {
      const size_t cScores = GetCountScores();

#ifndef NDEBUG
      // we should be initted to zero
      for(size_t iScore = 0; iScore < cScores; ++iScore) {
         EBM_ASSERT(0 == aUpdateScores[iScore]);
      }
#endif // NDEBUG

      EBM_ASSERT(1 <= pDataSet->GetCountSubsets());

      DataSubsetBoosting* pSubset = pDataSet->GetSubsets();
      EBM_ASSERT(nullptr != pSubset);
      const DataSubsetBoosting* const pSubsetsEnd = pSubset + pDataSet->GetCountSubsets();
      do {
         EBM_ASSERT(1 <= pSubset->GetCountSamples());

         ApplyUpdateBridge data;
         data.m_cScores = cScores;
         data.m_cPack = k_cItemsPerBitPackUndefined;
         data.m_bHessianNeeded = IsHessian() ? EBM_TRUE : EBM_FALSE;
         data.m_bUseApprox = IsUseApprox();
         data.m_bValidation = EBM_FALSE;
         data.m_aMulticlassMidwayTemp = aMulticlassMidwayTemp;
         // if FloatScore is type FloatSmall then some of the zones might use FloatBig as their type and then read
         // past the end of the aUpdateScores memory, which should always contain zeros.If we want to handle this
         // scenario then we need to allocate a larger buffer of memory to zero out instead of using aUpdateScores
         EBM_ASSERT(pSubset->GetObjectiveWrapper()->m_cFloatBytes <= sizeof(FloatScore));
         data.m_aUpdateTensorScores = aUpdateScores;
         data.m_cSamples = pSubset->GetCountSamples();
         data.m_aPacked = nullptr;
         data.m_aTargets = pSubset->GetTargetData();
         data.m_aWeights = nullptr;
         data.m_aSampleScores = pSubset->GetSampleScores();
         data.m_aGradientsAndHessians = pSubset->GetGradHess();
         data.m_metricOut = 0.0;
         const ErrorEbm error = pSubset->ObjectiveApplyUpdate(&data);
         if(Error_None != error) {
            return error;
         }

         ++pSubset;
      } while(pSubsetsEnd != pSubset);
   }
   return Error_None;
}

} // namespace DEFINED_ZONE_NAME
