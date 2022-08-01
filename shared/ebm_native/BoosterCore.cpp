// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits
#include <thread>

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "common_cpp.hpp"
#include "ebm_internal.hpp"

#include "compute_accessors.hpp"

#include "data_set_shared.hpp"
#include "RandomStream.hpp"
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
#include "TreeSweep.hpp"

#include "BoosterShell.hpp"
#include "BoosterCore.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern ErrorEbm InitializeGradientsAndHessians(
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const BagEbm * const aBag,
   const double * const aInitScores,
   const size_t cSetSamples,
   FloatFast * const aGradientAndHessian
);

extern ErrorEbm Unbag(
   const size_t cSamples,
   const BagEbm * const aBag,
   size_t * const pcTrainingSamplesOut,
   size_t * const pcValidationSamplesOut
);

extern ErrorEbm ExtractWeights(
   const unsigned char * const pDataSetShared,
   const BagEbm direction,
   const size_t cAllSamples,
   const BagEbm * const aBag,
   const size_t cSetSamples,
   FloatFast ** ppWeightsOut
);

INLINE_ALWAYS static size_t GetCountItemsBitPacked(const size_t cBits) {
   EBM_ASSERT(size_t { 1 } <= cBits);
   return k_cBitsForStorageType / cBits;
}

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

   EBM_ASSERT(0 < cTerms);
   EBM_ASSERT(nullptr != apTerms);
   EBM_ASSERT(1 <= cScores);
   EBM_ASSERT(nullptr != papTensorsOut);
   EBM_ASSERT(nullptr == *papTensorsOut);

   ErrorEbm error;

   Tensor ** const apTensors = EbmMalloc<Tensor *>(cTerms);
   if(UNLIKELY(nullptr == apTensors)) {
      LOG_0(Trace_Warning, "WARNING InitializeTensors nullptr == apTensors");
      return Error_OutOfMemory;
   }
   for(size_t iTerm = 0; iTerm < cTerms; ++iTerm) {
      apTensors[iTerm] = nullptr;
   }
   *papTensorsOut = apTensors; // transfer ownership for future deletion

   Tensor ** ppTensor = apTensors;
   for(size_t iTerm = 0; iTerm < cTerms; ++iTerm) {
      const Term * const pTerm = apTerms[iTerm];
      Tensor * const pTensors = 
         Tensor::Allocate(pTerm->GetCountDimensions(), cScores);
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

      ++ppTensor;
   }

   LOG_0(Trace_Info, "Exited InitializeTensors");
   return Error_None;
}

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
   BoosterShell * const pBoosterShell,
   const size_t cTerms,
   const size_t cSamplingSets,
   const double * const experimentalParams,
   const IntEbm * const acTermDimensions,
   const IntEbm * const aiTermFeatures, 
   const unsigned char * const pDataSetShared,
   const BagEbm * const aBag,
   const double * const aInitScores
) {
   // experimentalParams isn't used by default.  It's meant to provide an easy way for python or other higher
   // level languages to pass EXPERIMENTAL temporary parameters easily to the C++ code.
   UNUSED(experimentalParams);

   LOG_0(Trace_Info, "Entered BoosterCore::Create");

   EBM_ASSERT(nullptr != pBoosterShell);

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
   // give ownership of our object to pBoosterShell
   pBoosterShell->SetBoosterCore(pBoosterCore);

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
      LOG_0(Trace_Warning, "WARNING BoosterCore::Create size_t { 1 } < cWeights");
      return Error_IllegalParamValue;
   }
   if(size_t { 1 } != cTargets) {
      LOG_0(Trace_Warning, "WARNING BoosterCore::Create 1 != cTargets");
      return Error_IllegalParamValue;
   }

   ptrdiff_t cClasses;
   GetDataSetSharedTarget(pDataSetShared, 0, &cClasses);

   size_t cTrainingSamples;
   size_t cValidationSamples;
   error = Unbag(cSamples, aBag, &cTrainingSamples, &cValidationSamples);
   if(Error_None != error) {
      // already logged
      return error;
   }

   const size_t cScores = GetCountScores(cClasses);

   LOG_0(Trace_Info, "BoosterCore::Create starting feature processing");
   if(0 != cFeatures) {
      pBoosterCore->m_cFeatures = cFeatures;
      pBoosterCore->m_aFeatures = EbmMalloc<Feature>(cFeatures);
      if(nullptr == pBoosterCore->m_aFeatures) {
         LOG_0(Trace_Warning, "WARNING BoosterCore::Create nullptr == pBoosterCore->m_aFeatures");
         return Error_OutOfMemory;
      }

      size_t iFeatureInitialize = size_t { 0 };
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
         if(0 == cBins && (0 != cTrainingSamples || 0 != cValidationSamples)) {
            LOG_0(Trace_Error, "ERROR BoosterCore::Create countBins cannot be zero if either 0 < cTrainingSamples OR 0 < cValidationSamples");
            return Error_IllegalParamValue;
         }
         if(0 == cBins) {
            // we can handle 0 == cBins even though that's a degenerate case that shouldn't be boosted on.  0 bins
            // can only occur if there were zero training and zero validation cases since the 
            // features would require a value, even if it was 0.
            LOG_0(Trace_Info, "INFO BoosterCore::Create feature with 0 values");
         } else if(1 == cBins) {
            // Dimensions with 1 bin don't contribute anything to the model since they always have the same value, but 
            // the user can specify interactions, so we handle them anyways in a consistent way by boosting on them
            LOG_0(Trace_Info, "INFO BoosterCore::Create feature with 1 value");
         }
         pBoosterCore->m_aFeatures[iFeatureInitialize].Initialize(iFeatureInitialize, cBins, bMissing, bUnknown, bNominal);

         ++iFeatureInitialize;
      } while(cFeatures != iFeatureInitialize);
   }
   LOG_0(Trace_Info, "BoosterCore::Create done feature processing");

   const bool bClassification = IsClassification(cClasses);
   size_t cBytesArrayEquivalentSplitMax = 0;

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

      if(GetTreeSweepSizeOverflow(bClassification, cScores)) {
         LOG_0(Trace_Warning, "WARNING BoosterCore::Create GetTreeSweepSizeOverflow(bClassification, cScores)");
         return Error_OutOfMemory;
      }
      const size_t cBytesPerTreeSweep = GetTreeSweepSize(bClassification, cScores);

      const IntEbm * piTermFeature = aiTermFeatures;
      size_t iTerm = 0;
      do {
         const IntEbm countDimensions = acTermDimensions[iTerm];
         if(countDimensions < IntEbm { 0 }) {
            LOG_0(Trace_Error, "ERROR BoosterCore::Create countDimensions cannot be negative");
            return Error_IllegalParamValue;
         }
         if(IntEbm { k_cDimensionsMax } < countDimensions) {
            LOG_0(Trace_Error, "WARNING BoosterCore::Create countDimensions too large and would cause out of memory condition");
            return Error_OutOfMemory;
         }
         const size_t cDimensions = static_cast<size_t>(countDimensions);
         Term * const pTerm = Term::Allocate(cDimensions, iTerm);
         if(nullptr == pTerm) {
            LOG_0(Trace_Warning, "WARNING BoosterCore::Create nullptr == pTerm");
            return Error_OutOfMemory;
         }
         // assign our pointer directly to our array right now so that we can't loose the memory if we decide to exit due to an error below
         pBoosterCore->m_apTerms[iTerm] = pTerm;

         size_t cSignificantDimensions = 0;
         ptrdiff_t cItemsPerBitPack = k_cItemsPerBitPackNone;
         size_t cTensorBins = 1;
         if(UNLIKELY(0 == cDimensions)) {
            LOG_0(Trace_Info, "INFO BoosterCore::Create empty feature group");
         } else {
            if(nullptr == piTermFeature) {
               LOG_0(Trace_Error, "ERROR BoosterCore::Create aiTermFeatures is null when there are Terms with non-zero numbers of features");
               return Error_IllegalParamValue;
            }
            size_t cEquivalentSplits = 1;
            TermEntry * pTermEntry = pTerm->GetTermEntries();
            TermEntry * pTermEntriesEnd = pTermEntry + cDimensions;
            do {
               const IntEbm indexFeature = *piTermFeature;
               if(indexFeature < 0) {
                  LOG_0(Trace_Error, "ERROR BoosterCore::Create aiTermFeatures value cannot be negative");
                  return Error_IllegalParamValue;
               }
               if(IsConvertError<size_t>(indexFeature)) {
                  LOG_0(Trace_Error, "ERROR BoosterCore::Create aiTermFeatures value too big to reference memory");
                  return Error_IllegalParamValue;
               }
               const size_t iFeature = static_cast<size_t>(indexFeature);

               if(cFeatures <= iFeature) {
                  LOG_0(Trace_Error, "ERROR BoosterCore::Create aiTermFeatures value must be less than the number of features");
                  return Error_IllegalParamValue;
               }

               EBM_ASSERT(1 <= cFeatures);
               EBM_ASSERT(nullptr != pBoosterCore->m_aFeatures);

               Feature * const pInputFeature = &pBoosterCore->m_aFeatures[iFeature];
               pTermEntry->m_pFeature = pInputFeature;

               const size_t cBins = pInputFeature->GetCountBins();
               if(LIKELY(size_t { 1 } < cBins)) {
                  // if we have only 1 bin, then we can eliminate the feature from consideration since the resulting tensor loses one dimension but is 
                  // otherwise indistinquishable from the original data
                  ++cSignificantDimensions;
                  if(IsMultiplyError(cTensorBins, cBins)) {
                     // if this overflows, we definetly won't be able to allocate it
                     LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsMultiplyError(cTensorStates, cBins)");
                     return Error_OutOfMemory;
                  }
                  cTensorBins *= cBins;
                  cEquivalentSplits *= cBins - 1; // we can only split between the bins
               } else {
                  LOG_0(Trace_Info, "INFO BoosterCore::Create feature group with no useful features");
               }

               ++piTermFeature;
               ++pTermEntry;
            } while(pTermEntriesEnd != pTermEntry);

            if(LIKELY(0 != cSignificantDimensions)) {
               EBM_ASSERT(1 < cTensorBins);

               size_t cBytesArrayEquivalentSplit;
               if(1 == cSignificantDimensions) {
                  if(IsMultiplyError(cBytesPerTreeSweep, cEquivalentSplits)) {
                     LOG_0(Trace_Warning, "WARNING BoosterCore::Create IsMultiplyError(cBytesPerTreeSweep, cEquivalentSplits)");
                     return Error_OutOfMemory;
                  }
                  cBytesArrayEquivalentSplit = cBytesPerTreeSweep * cEquivalentSplits;
               } else {
                  // TODO : someday add equal gain multidimensional randomized picking.  It's rather hard though with the existing sweep functions for 
                  // multidimensional right now
                  cBytesArrayEquivalentSplit = 0;
               }
               if(cBytesArrayEquivalentSplitMax < cBytesArrayEquivalentSplit) {
                  cBytesArrayEquivalentSplitMax = cBytesArrayEquivalentSplit;
               }

               const size_t cBitsRequiredMin = CountBitsRequired(cTensorBins - 1);
               EBM_ASSERT(1 <= cBitsRequiredMin); // 1 < cTensorBins otherwise we'd have filtered it out above
               cItemsPerBitPack = static_cast<ptrdiff_t>(GetCountItemsBitPacked(cBitsRequiredMin));
            }
         }
         pTerm->SetCountTensorBins(cTensorBins);
         pTerm->SetCountSignificantFeatures(cSignificantDimensions);
         pTerm->SetBitPack(cItemsPerBitPack);

         ++iTerm;
      } while(iTerm < cTerms);

      if(!bClassification || ptrdiff_t { 2 } <= cClasses) {
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

   pBoosterCore->m_cBytesArrayEquivalentSplitMax = cBytesArrayEquivalentSplitMax;

   error = pBoosterCore->m_trainingSet.Initialize(
      cClasses,
      true,
      bClassification,
      bClassification,
      bClassification,
      pDataSetShared,
      BagEbm { 1 },
      aBag,
      aInitScores,
      cTrainingSamples,
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
      bClassification,
      bClassification,
      pDataSetShared,
      BagEbm { -1 },
      aBag,
      aInitScores,
      cValidationSamples,
      cTerms,
      pBoosterCore->m_apTerms
   );
   if(Error_None != error) {
      LOG_0(Trace_Warning, "WARNING BoosterCore::Create m_validationSet.Initialize");
      return error;
   }

   EBM_ASSERT(nullptr == pBoosterCore->m_apSamplingSets);
   if(0 != cTrainingSamples) {
      FloatFast * aWeights = nullptr;
      if(0 != cWeights) {
         error = ExtractWeights(
            pDataSetShared,
            BagEbm { 1 },
            cSamples, 
            aBag, 
            cTrainingSamples,
            &aWeights
         );
         if(Error_None != error) {
            // error already logged
            return error;
         }
      }
      pBoosterCore->m_cSamplingSets = cSamplingSets;
      // TODO: we could steal the aWeights in GenerateSamplingSets for flat sampling sets
      pBoosterCore->m_apSamplingSets = SamplingSet::GenerateSamplingSets(
         pBoosterShell->GetRandomDeterministic(),
         &pBoosterCore->m_trainingSet, 
         aWeights, 
         cSamplingSets
      );
      free(aWeights);
      if(UNLIKELY(nullptr == pBoosterCore->m_apSamplingSets)) {
         LOG_0(Trace_Warning, "WARNING BoosterCore::Create nullptr == m_apSamplingSets");
         return Error_OutOfMemory;
      }
   }

   EBM_ASSERT(nullptr == pBoosterCore->m_aValidationWeights);
   pBoosterCore->m_validationWeightTotal = static_cast<FloatBig>(cValidationSamples);
   if(0 != cWeights && 0 != cValidationSamples) {
      error = ExtractWeights(
         pDataSetShared,
         BagEbm { -1 },
         cSamples, 
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
            return Error_UserParamValue;
         }
         // if they were all zero then we'd ignore the weights param.  If there are negative numbers it might add
         // to zero though so check it after checking for negative
         EBM_ASSERT(0 != total);
         pBoosterCore->m_validationWeightTotal = total;
      }
   }

   if(bClassification) {
      if(0 != cTrainingSamples) {
         error = InitializeGradientsAndHessians(
            pDataSetShared,
            BagEbm { 1 },
            aBag,
            aInitScores,
            cTrainingSamples,
            pBoosterCore->m_trainingSet.GetGradientsAndHessiansPointer()
         );
         if(Error_None != error) {
            // error already logged
            return error;
         }
      }
   } else {
      EBM_ASSERT(IsRegression(cClasses));
      if(0 != cTrainingSamples) {
#ifndef NDEBUG
         const ErrorEbm errorDebug =
#endif // NDEBUG
         InitializeGradientsAndHessians(
            pDataSetShared,
            BagEbm { 1 },
            aBag,
            aInitScores,
            cTrainingSamples,
            pBoosterCore->m_trainingSet.GetGradientsAndHessiansPointer()
         );
         EBM_ASSERT(Error_None == errorDebug); // InitializeGradientsAndHessians doesn't allocate on regression
      }
      if(0 != cValidationSamples) {
#ifndef NDEBUG
         const ErrorEbm errorDebug =
#endif // NDEBUG
         InitializeGradientsAndHessians(
            pDataSetShared,
            BagEbm { -1 },
            aBag,
            aInitScores,
            cValidationSamples,
            pBoosterCore->m_validationSet.GetGradientsAndHessiansPointer()
         );
         EBM_ASSERT(Error_None == errorDebug); // InitializeGradientsAndHessians doesn't allocate on regression
      }
   }

   pBoosterCore->m_cClasses = cClasses;
   pBoosterCore->m_bestModelMetric = std::numeric_limits<double>::max();

   LOG_0(Trace_Info, "Exited BoosterCore::Create");
   return Error_None;
}

} // DEFINED_ZONE_NAME
