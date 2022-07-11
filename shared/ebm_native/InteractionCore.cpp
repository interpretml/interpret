// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "precompiled_header_cpp.hpp"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "data_set_shared.hpp"

// feature includes
#include "Feature.hpp"
#include "FeatureGroup.hpp"
// dataset depends on features
#include "DataSetInteraction.hpp"
#include "InteractionShell.hpp"

#include "InteractionCore.hpp"
#include "InteractionShell.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

extern ErrorEbmType Unbag(
   const size_t cSamples,
   const BagEbmType * const aBag,
   size_t * const pcTrainingSamplesOut,
   size_t * const pcValidationSamplesOut
);

void InteractionCore::Free(InteractionCore * const pInteractionCore) {
   LOG_0(TraceLevelInfo, "Entered InteractionCore::Free");

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
         LOG_0(TraceLevelInfo, "INFO InteractionCore::Free deleting InteractionCore");
         delete pInteractionCore;
      }
   }

   LOG_0(TraceLevelInfo, "Exited InteractionCore::Free");
}

ErrorEbmType InteractionCore::Create(
   InteractionShell * const pInteractionShell,
   const unsigned char * const pDataSetShared,
   const BagEbmType * const aBag,
   const double * const aInitScores,
   const double * const optionalTempParams
) {
   // optionalTempParams isn't used by default.  It's meant to provide an easy way for python or other higher
   // level languages to pass EXPERIMENTAL temporary parameters easily to the C++ code.
   UNUSED(optionalTempParams);

   LOG_0(TraceLevelInfo, "Entered InteractionCore::Allocate");

   EBM_ASSERT(nullptr != pInteractionShell);
   EBM_ASSERT(nullptr != pDataSetShared);

   ErrorEbmType error;

   InteractionCore * pRet;
   try {
      pRet = new InteractionCore();
   } catch(const std::bad_alloc &) {
      LOG_0(TraceLevelWarning, "WARNING InteractionCore::Create Out of memory allocating InteractionCore");
      return Error_OutOfMemory;
   } catch(...) {
      LOG_0(TraceLevelWarning, "WARNING InteractionCore::Create Unknown error");
      return Error_UnexpectedInternal;
   }
   if(nullptr == pRet) {
      // this should be impossible since bad_alloc should have been thrown, but let's be untrusting
      LOG_0(TraceLevelWarning, "WARNING InteractionCore::Create nullptr == pInteractionCore");
      return Error_OutOfMemory;
   }
   // give ownership of our object to pInteractionShell
   pInteractionShell->SetInteractionCore(pRet);

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

   LOG_0(TraceLevelInfo, "InteractionCore::Allocate starting feature processing");
   if(0 != cFeatures) {
      pRet->m_cFeatures = cFeatures;
      Feature * const aFeatures = EbmMalloc<Feature>(cFeatures);
      if(nullptr == aFeatures) {
         LOG_0(TraceLevelWarning, "WARNING InteractionCore::Allocate nullptr == aFeatures");
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
         if(0 == cBins && 0 != cSamples) {
            LOG_0(TraceLevelError, "ERROR InteractionCore::Allocate countBins cannot be zero if 0 < cSamples");
            return Error_IllegalParamValue;
         }
         if(0 == cBins) {
            // we can handle 0 == cBins even though that's a degenerate case that shouldn't be boosted on.  0 bins
            // can only occur if there were zero training and zero validation cases since the 
            // features would require a value, even if it was 0.
            LOG_0(TraceLevelInfo, "INFO InteractionCore::Allocate feature with 0 values");
         } else if(1 == cBins) {
            // we can handle 1 == cBins even though that's a degenerate case that shouldn't be boosted on. 
            // Dimensions with 1 bin don't contribute anything since they always have the same value.
            LOG_0(TraceLevelInfo, "INFO InteractionCore::Allocate feature with 1 value");
         }
         aFeatures[iFeatureInitialize].Initialize(iFeatureInitialize, cBins, bMissing, bUnknown, bNominal);

         ++iFeatureInitialize;
      } while(cFeatures != iFeatureInitialize);
   }
   LOG_0(TraceLevelInfo, "InteractionCore::Allocate done feature processing");

   pRet->m_runtimeLearningTypeOrCountTargetClasses = runtimeLearningTypeOrCountTargetClasses;

   error = pRet->m_dataFrame.Initialize(
      IsClassification(runtimeLearningTypeOrCountTargetClasses),
      pDataSetShared,
      cSamples,
      aBag,
      aInitScores,
      cTrainingSamples,
      cWeights,
      cFeatures
   );
   if(Error_None != error) {
      LOG_0(TraceLevelWarning, "WARNING InteractionCore::Allocate m_dataFrame.Initialize");
      return error;
   }

   LOG_0(TraceLevelInfo, "Exited InteractionCore::Allocate");
   return Error_None;
}

} // DEFINED_ZONE_NAME
