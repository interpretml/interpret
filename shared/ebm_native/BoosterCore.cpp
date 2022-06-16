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
// FeatureGroup.h depends on FeatureInternal.h
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

extern ErrorEbmType InitializeGradientsAndHessians(
   const unsigned char * const pDataSetShared,
   const BagEbmType direction,
   const BagEbmType * const aBag,
   const double * const aPredictorScores,
   const size_t cSetSamples,
   FloatFast * const aGradientAndHessian
);

extern ErrorEbmType Unbag(
   const size_t cSamples,
   const BagEbmType * const aBag,
   size_t * const pcTrainingSamplesOut,
   size_t * const pcValidationSamplesOut
);

extern ErrorEbmType ExtractWeights(
   const unsigned char * const pDataSetShared,
   const BagEbmType direction,
   const size_t cAllSamples,
   const BagEbmType * const aBag,
   const size_t cSetSamples,
   FloatFast ** ppWeightsOut
);

INLINE_ALWAYS static size_t GetCountItemsBitPacked(const size_t cBits) {
   EBM_ASSERT(size_t { 1 } <= cBits);
   return k_cBitsForStorageType / cBits;
}

void BoosterCore::DeleteCompressibleTensors(const size_t cFeatureGroups, CompressibleTensor ** const apCompressibleTensors) {
   LOG_0(TraceLevelInfo, "Entered DeleteCompressibleTensors");

   if(UNLIKELY(nullptr != apCompressibleTensors)) {
      EBM_ASSERT(0 < cFeatureGroups);
      CompressibleTensor ** ppCompressibleTensors = apCompressibleTensors;
      const CompressibleTensor * const * const ppCompressibleTensorsEnd = &apCompressibleTensors[cFeatureGroups];
      do {
         CompressibleTensor::Free(*ppCompressibleTensors);
         ++ppCompressibleTensors;
      } while(ppCompressibleTensorsEnd != ppCompressibleTensors);
      free(apCompressibleTensors);
   }
   LOG_0(TraceLevelInfo, "Exited DeleteCompressibleTensors");
}

ErrorEbmType BoosterCore::InitializeCompressibleTensors(
   const size_t cFeatureGroups, 
   const FeatureGroup * const * const apFeatureGroups, 
   const size_t cVectorLength,
   CompressibleTensor *** papCompressibleTensorsOut)
{
   LOG_0(TraceLevelInfo, "Entered InitializeCompressibleTensors");

   EBM_ASSERT(0 < cFeatureGroups);
   EBM_ASSERT(nullptr != apFeatureGroups);
   EBM_ASSERT(1 <= cVectorLength);
   EBM_ASSERT(nullptr != papCompressibleTensorsOut);
   EBM_ASSERT(nullptr == *papCompressibleTensorsOut);

   ErrorEbmType error;

   CompressibleTensor ** const apCompressibleTensors = EbmMalloc<CompressibleTensor *>(cFeatureGroups);
   if(UNLIKELY(nullptr == apCompressibleTensors)) {
      LOG_0(TraceLevelWarning, "WARNING InitializeCompressibleTensors nullptr == apCompressibleTensors");
      return Error_OutOfMemory;
   }
   for(size_t i = 0; i < cFeatureGroups; ++i) {
      apCompressibleTensors[i] = nullptr;
   }
   *papCompressibleTensorsOut = apCompressibleTensors; // transfer ownership for future deletion

   CompressibleTensor ** ppCompressibleTensors = apCompressibleTensors;
   for(size_t iFeatureGroup = 0; iFeatureGroup < cFeatureGroups; ++iFeatureGroup) {
      const FeatureGroup * const pFeatureGroup = apFeatureGroups[iFeatureGroup];
      CompressibleTensor * const pCompressibleTensors = 
         CompressibleTensor::Allocate(pFeatureGroup->GetCountDimensions(), cVectorLength);
      if(UNLIKELY(nullptr == pCompressibleTensors)) {
         LOG_0(TraceLevelWarning, "WARNING InitializeCompressibleTensors nullptr == pCompressibleTensors");
         return Error_OutOfMemory;
      }
      *ppCompressibleTensors = pCompressibleTensors; // transfer ownership for future deletion

      error = pCompressibleTensors->Expand(pFeatureGroup);
      if(Error_None != error) {
         // already logged
         return error;
      }

      ++ppCompressibleTensors;
   }

   LOG_0(TraceLevelInfo, "Exited InitializeCompressibleTensors");
   return Error_None;
}

void BoosterCore::Free(BoosterCore * const pBoosterCore) {
   LOG_0(TraceLevelInfo, "Entered BoosterCore::Free");
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
         LOG_0(TraceLevelInfo, "INFO BoosterCore::Free deleting BoosterCore");
         delete pBoosterCore;
      }
   }
   LOG_0(TraceLevelInfo, "Exited BoosterCore::Free");
}

//static int g_TODO_removeThisThreadTest = 0;
//void TODO_removeThisThreadTest() {
//   g_TODO_removeThisThreadTest = 1;
//}

ErrorEbmType BoosterCore::Create(
   BoosterShell * const pBoosterShell,
   const size_t cFeatureGroups,
   const size_t cSamplingSets,
   const double * const optionalTempParams,
   const IntEbmType * const aFeatureGroupsDimensionCounts,
   const IntEbmType * const aFeatureGroupsFeatureIndexes, 
   const unsigned char * const pDataSetShared,
   const BagEbmType * const aBag,
   const double * const aPredictorScores
) {
   // optionalTempParams isn't used by default.  It's meant to provide an easy way for python or other higher
   // level languages to pass EXPERIMENTAL temporary parameters easily to the C++ code.
   UNUSED(optionalTempParams);

   LOG_0(TraceLevelInfo, "Entered BoosterCore::Create");

   EBM_ASSERT(nullptr != pBoosterShell);

   ErrorEbmType error;

   //try {
   //   // TODO: eliminate this code I added to test that threads are available on the majority of our systems
   //   std::thread testThread(TODO_removeThisThreadTest);
   //   testThread.join();
   //   if(0 == g_TODO_removeThisThreadTest) {
   //      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create thread not started");
   //      return Error_UnexpectedInternal;
   //   }
   //} catch(const std::bad_alloc &) {
   //   LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create thread start out of memory");
   //   return Error_OutOfMemory;
   //} catch(...) {
   //   // the C++ standard doesn't really seem to say what kind of exceptions we'd get for various errors, so
   //   // about the best we can do is catch(...) since the exact exceptions seem to be implementation specific
   //   LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create thread start failed");
   //   return Error_ThreadStartFailed;
   //}
   //LOG_0(TraceLevelInfo, "INFO BoosterCore::Create thread started");

   BoosterCore * pBoosterCore;
   try {
      pBoosterCore = new BoosterCore();
   } catch(const std::bad_alloc &) {
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create Out of memory allocating BoosterCore");
      return Error_OutOfMemory;
   } catch(...) {
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create Unknown error");
      return Error_UnexpectedInternal;
   }
   if(nullptr == pBoosterCore) {
      // this should be impossible since bad_alloc should have been thrown, but let's be untrusting
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == pBoosterCore");
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
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create size_t { 1 } < cWeights");
      return Error_IllegalParamValue;
   }
   if(size_t { 1 } != cTargets) {
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create 1 != cTargets");
      return Error_IllegalParamValue;
   }

   ptrdiff_t runtimeLearningTypeOrCountTargetClasses;
   GetDataSetSharedTarget(pDataSetShared, 0, &runtimeLearningTypeOrCountTargetClasses);

   size_t cTrainingSamples;
   size_t cValidationSamples;
   error = Unbag(cSamples, aBag, &cTrainingSamples, &cValidationSamples);
   if(Error_None != error) {
      // already logged
      return error;
   }

   const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);

   LOG_0(TraceLevelInfo, "BoosterCore::Create starting feature processing");
   if(0 != cFeatures) {
      pBoosterCore->m_cFeatures = cFeatures;
      pBoosterCore->m_aFeatures = EbmMalloc<Feature>(cFeatures);
      if(nullptr == pBoosterCore->m_aFeatures) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == pBoosterCore->m_aFeatures");
         return Error_OutOfMemory;
      }

      size_t iFeatureInitialize = size_t { 0 };
      do {
         size_t cBins;
         bool bMissing;
         bool bUnknown;
         bool bNominal;
         bool bSparse;
         SharedStorageDataType defaultValueSparse;
         size_t cNonDefaultsSparse;
         GetDataSetSharedFeature(
            pDataSetShared,
            iFeatureInitialize,
            &cBins,
            &bMissing,
            &bUnknown,
            &bNominal,
            &bSparse,
            &defaultValueSparse,
            &cNonDefaultsSparse
         );
         if(0 == cBins && (0 != cTrainingSamples || 0 != cValidationSamples)) {
            LOG_0(TraceLevelError, "ERROR BoosterCore::Create countBins cannot be zero if either 0 < cTrainingSamples OR 0 < cValidationSamples");
            return Error_IllegalParamValue;
         }
         if(0 == cBins) {
            // we can handle 0 == cBins even though that's a degenerate case that shouldn't be boosted on.  0 bins
            // can only occur if there were zero training and zero validation cases since the 
            // features would require a value, even if it was 0.
            LOG_0(TraceLevelInfo, "INFO BoosterCore::Create feature with 0 values");
         } else if(1 == cBins) {
            // Dimensions with 1 bin don't contribute anything to the model since they always have the same value, but 
            // the user can specify interactions, so we handle them anyways in a consistent way by boosting on them
            LOG_0(TraceLevelInfo, "INFO BoosterCore::Create feature with 1 value");
         }
         pBoosterCore->m_aFeatures[iFeatureInitialize].Initialize(iFeatureInitialize, cBins, bMissing, bUnknown, bNominal);

         ++iFeatureInitialize;
      } while(cFeatures != iFeatureInitialize);
   }
   LOG_0(TraceLevelInfo, "BoosterCore::Create done feature processing");

   const bool bClassification = IsClassification(runtimeLearningTypeOrCountTargetClasses);
   size_t cBytesArrayEquivalentSplitMax = 0;

   EBM_ASSERT(nullptr == pBoosterCore->m_apCurrentModel);
   EBM_ASSERT(nullptr == pBoosterCore->m_apBestModel);

   LOG_0(TraceLevelInfo, "BoosterCore::Create starting feature group processing");
   if(0 != cFeatureGroups) {
      pBoosterCore->m_cFeatureGroups = cFeatureGroups;
      pBoosterCore->m_apFeatureGroups = FeatureGroup::AllocateFeatureGroups(cFeatureGroups);
      if(UNLIKELY(nullptr == pBoosterCore->m_apFeatureGroups)) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create 0 != m_cFeatureGroups && nullptr == m_apFeatureGroups");
         return Error_OutOfMemory;
      }

      if(GetTreeSweepSizeOverflow(bClassification, cVectorLength)) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create GetTreeSweepSizeOverflow(bClassification, cVectorLength)");
         return Error_OutOfMemory;
      }
      const size_t cBytesPerTreeSweep = GetTreeSweepSize(bClassification, cVectorLength);

      const IntEbmType * pFeatureGroupFeatureIndexes = aFeatureGroupsFeatureIndexes;
      size_t iFeatureGroup = 0;
      do {
         const IntEbmType countDimensions = aFeatureGroupsDimensionCounts[iFeatureGroup];
         if(countDimensions < IntEbmType { 0 }) {
            LOG_0(TraceLevelError, "ERROR BoosterCore::Create countDimensions cannot be negative");
            return Error_IllegalParamValue;
         }
         if(IntEbmType { k_cDimensionsMax } < countDimensions) {
            LOG_0(TraceLevelError, "WARNING BoosterCore::Create countDimensions too large and would cause out of memory condition");
            return Error_OutOfMemory;
         }
         const size_t cDimensions = static_cast<size_t>(countDimensions);
         FeatureGroup * const pFeatureGroup = FeatureGroup::Allocate(cDimensions, iFeatureGroup);
         if(nullptr == pFeatureGroup) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == pFeatureGroup");
            return Error_OutOfMemory;
         }
         // assign our pointer directly to our array right now so that we can't loose the memory if we decide to exit due to an error below
         pBoosterCore->m_apFeatureGroups[iFeatureGroup] = pFeatureGroup;

         size_t cSignificantDimensions = 0;
         ptrdiff_t cItemsPerBitPack = k_cItemsPerBitPackNone;
         size_t cTensorBins = 1;
         if(UNLIKELY(0 == cDimensions)) {
            LOG_0(TraceLevelInfo, "INFO BoosterCore::Create empty feature group");
         } else {
            if(nullptr == pFeatureGroupFeatureIndexes) {
               LOG_0(TraceLevelError, "ERROR BoosterCore::Create aFeatureGroupsFeatureIndexes is null when there are FeatureGroups with non-zero numbers of features");
               return Error_IllegalParamValue;
            }
            size_t cEquivalentSplits = 1;
            FeatureGroupEntry * pFeatureGroupEntry = pFeatureGroup->GetFeatureGroupEntries();
            FeatureGroupEntry * pFeatureGroupEntryEnd = pFeatureGroupEntry + cDimensions;
            do {
               const IntEbmType indexFeatureInterop = *pFeatureGroupFeatureIndexes;
               if(indexFeatureInterop < 0) {
                  LOG_0(TraceLevelError, "ERROR BoosterCore::Create aFeatureGroupsFeatureIndexes value cannot be negative");
                  return Error_IllegalParamValue;
               }
               if(IsConvertError<size_t>(indexFeatureInterop)) {
                  LOG_0(TraceLevelError, "ERROR BoosterCore::Create aFeatureGroupsFeatureIndexes value too big to reference memory");
                  return Error_IllegalParamValue;
               }
               const size_t iFeature = static_cast<size_t>(indexFeatureInterop);

               if(cFeatures <= iFeature) {
                  LOG_0(TraceLevelError, "ERROR BoosterCore::Create aFeatureGroupsFeatureIndexes value must be less than the number of features");
                  return Error_IllegalParamValue;
               }

               EBM_ASSERT(1 <= cFeatures);
               EBM_ASSERT(nullptr != pBoosterCore->m_aFeatures);

               Feature * const pInputFeature = &pBoosterCore->m_aFeatures[iFeature];
               pFeatureGroupEntry->m_pFeature = pInputFeature;

               const size_t cBins = pInputFeature->GetCountBins();
               if(LIKELY(size_t { 1 } < cBins)) {
                  // if we have only 1 bin, then we can eliminate the feature from consideration since the resulting tensor loses one dimension but is 
                  // otherwise indistinquishable from the original data
                  ++cSignificantDimensions;
                  if(IsMultiplyError(cTensorBins, cBins)) {
                     // if this overflows, we definetly won't be able to allocate it
                     LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create IsMultiplyError(cTensorStates, cBins)");
                     return Error_OutOfMemory;
                  }
                  cTensorBins *= cBins;
                  cEquivalentSplits *= cBins - 1; // we can only split between the bins
               } else {
                  LOG_0(TraceLevelInfo, "INFO BoosterCore::Create feature group with no useful features");
               }

               ++pFeatureGroupFeatureIndexes;
               ++pFeatureGroupEntry;
            } while(pFeatureGroupEntryEnd != pFeatureGroupEntry);

            if(LIKELY(0 != cSignificantDimensions)) {
               EBM_ASSERT(1 < cTensorBins);

               size_t cBytesArrayEquivalentSplit;
               if(1 == cSignificantDimensions) {
                  if(IsMultiplyError(cBytesPerTreeSweep, cEquivalentSplits)) {
                     LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create IsMultiplyError(cBytesPerTreeSweep, cEquivalentSplits)");
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
         pFeatureGroup->SetCountTensorBins(cTensorBins);
         pFeatureGroup->SetCountSignificantFeatures(cSignificantDimensions);
         pFeatureGroup->SetBitPack(cItemsPerBitPack);

         ++iFeatureGroup;
      } while(iFeatureGroup < cFeatureGroups);

      if(!bClassification || ptrdiff_t { 2 } <= runtimeLearningTypeOrCountTargetClasses) {
         error = InitializeCompressibleTensors(cFeatureGroups, pBoosterCore->m_apFeatureGroups, cVectorLength, &pBoosterCore->m_apCurrentModel);
         if(Error_None != error) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == m_apCurrentModel");
            return error;
         }
         error = InitializeCompressibleTensors(cFeatureGroups, pBoosterCore->m_apFeatureGroups, cVectorLength, &pBoosterCore->m_apBestModel);
         if(Error_None != error) {
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == m_apBestModel");
            return error;
         }
      }
   }
   LOG_0(TraceLevelInfo, "BoosterCore::Create finished feature group processing");

   pBoosterCore->m_cBytesArrayEquivalentSplitMax = cBytesArrayEquivalentSplitMax;

   error = pBoosterCore->m_trainingSet.Initialize(
      runtimeLearningTypeOrCountTargetClasses,
      true,
      bClassification,
      bClassification,
      bClassification,
      pDataSetShared,
      BagEbmType { 1 },
      aBag,
      aPredictorScores,
      cTrainingSamples,
      cFeatureGroups,
      pBoosterCore->m_apFeatureGroups
   );
   if(Error_None != error) {
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create m_trainingSet.Initialize");
      return error;
   }

   error = pBoosterCore->m_validationSet.Initialize(
      runtimeLearningTypeOrCountTargetClasses,
      !bClassification,
      false,
      bClassification,
      bClassification,
      pDataSetShared,
      BagEbmType { -1 },
      aBag,
      aPredictorScores,
      cValidationSamples,
      cFeatureGroups,
      pBoosterCore->m_apFeatureGroups
   );
   if(Error_None != error) {
      LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create m_validationSet.Initialize");
      return error;
   }

   EBM_ASSERT(nullptr == pBoosterCore->m_apSamplingSets);
   if(0 != cTrainingSamples) {
      FloatFast * aWeights = nullptr;
      if(0 != cWeights) {
         error = ExtractWeights(
            pDataSetShared,
            BagEbmType { 1 },
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
         pBoosterShell->GetRandomStream(), 
         &pBoosterCore->m_trainingSet, 
         aWeights, 
         cSamplingSets
      );
      free(aWeights);
      if(UNLIKELY(nullptr == pBoosterCore->m_apSamplingSets)) {
         LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create nullptr == m_apSamplingSets");
         return Error_OutOfMemory;
      }
   }

   EBM_ASSERT(nullptr == pBoosterCore->m_aValidationWeights);
   pBoosterCore->m_validationWeightTotal = static_cast<FloatBig>(cValidationSamples);
   if(0 != cWeights && 0 != cValidationSamples) {
      error = ExtractWeights(
         pDataSetShared,
         BagEbmType { -1 },
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
            LOG_0(TraceLevelWarning, "WARNING BoosterCore::Create std::isnan(total) || std::isinf(total) || total <= 0");
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
            BagEbmType { 1 },
            aBag,
            aPredictorScores,
            cTrainingSamples,
            pBoosterCore->m_trainingSet.GetGradientsAndHessiansPointer()
         );
         if(Error_None != error) {
            // error already logged
            return error;
         }
      }
   } else {
      EBM_ASSERT(IsRegression(runtimeLearningTypeOrCountTargetClasses));
      if(0 != cTrainingSamples) {
#ifndef NDEBUG
         const ErrorEbmType errorDebug =
#endif // NDEBUG
         InitializeGradientsAndHessians(
            pDataSetShared,
            BagEbmType { 1 },
            aBag,
            aPredictorScores,
            cTrainingSamples,
            pBoosterCore->m_trainingSet.GetGradientsAndHessiansPointer()
         );
         EBM_ASSERT(Error_None == errorDebug); // InitializeGradientsAndHessians doesn't allocate on regression
      }
      if(0 != cValidationSamples) {
#ifndef NDEBUG
         const ErrorEbmType errorDebug =
#endif // NDEBUG
         InitializeGradientsAndHessians(
            pDataSetShared,
            BagEbmType { -1 },
            aBag,
            aPredictorScores,
            cValidationSamples,
            pBoosterCore->m_validationSet.GetGradientsAndHessiansPointer()
         );
         EBM_ASSERT(Error_None == errorDebug); // InitializeGradientsAndHessians doesn't allocate on regression
      }
   }

   pBoosterCore->m_runtimeLearningTypeOrCountTargetClasses = runtimeLearningTypeOrCountTargetClasses;
   pBoosterCore->m_bestModelMetric = std::numeric_limits<double>::max();

   LOG_0(TraceLevelInfo, "Exited BoosterCore::Create");
   return Error_None;
}

} // DEFINED_ZONE_NAME
