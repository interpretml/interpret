// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "EbmInternal.h"

#include "SegmentedTensor.h"

#include "HistogramTargetEntry.h"

#include "Booster.h"

#include "ThreadStateBoosting.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

void ThreadStateBoosting::Free(ThreadStateBoosting * const pThreadStateBoosting) {
   LOG_0(TraceLevelInfo, "Entered ThreadStateBoosting::Free");

   if(nullptr != pThreadStateBoosting) {
      SegmentedTensor::Free(pThreadStateBoosting->m_pSmallChangeToModelAccumulatedFromSamplingSets);
      SegmentedTensor::Free(pThreadStateBoosting->m_pSmallChangeToModelOverwriteSingleSamplingSet);
      free(pThreadStateBoosting->m_aThreadByteBuffer1);
      free(pThreadStateBoosting->m_aThreadByteBuffer2);
      free(pThreadStateBoosting->m_aSumHistogramTargetEntry);
      free(pThreadStateBoosting->m_aSumHistogramTargetEntryLeft);
      free(pThreadStateBoosting->m_aSumHistogramTargetEntryRight);
      free(pThreadStateBoosting->m_aTempFloatVector);
      free(pThreadStateBoosting->m_aEquivalentSplits);

      free(pThreadStateBoosting);
   }

   LOG_0(TraceLevelInfo, "Exited ThreadStateBoosting::Free");
}

ThreadStateBoosting * ThreadStateBoosting::Allocate(Booster * const pBooster) {
   LOG_0(TraceLevelInfo, "Entered ThreadStateBoosting::Allocate");

   ThreadStateBoosting * const pNew = EbmMalloc<ThreadStateBoosting>();
   if(LIKELY(nullptr != pNew)) {
      pNew->InitializeZero();

      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = pBooster->GetRuntimeLearningTypeOrCountTargetClasses();
      const size_t cVectorLength = GetVectorLength(runtimeLearningTypeOrCountTargetClasses);
      const size_t cBytesPerItem = IsClassification(runtimeLearningTypeOrCountTargetClasses) ?
         sizeof(HistogramTargetEntry<true>) : sizeof(HistogramTargetEntry<false>);

      SegmentedTensor * const pSmallChangeToModelAccumulatedFromSamplingSets =
         SegmentedTensor::Allocate(k_cDimensionsMax, cVectorLength);
      if(LIKELY(nullptr != pSmallChangeToModelAccumulatedFromSamplingSets)) {
         pNew->m_pSmallChangeToModelAccumulatedFromSamplingSets = pSmallChangeToModelAccumulatedFromSamplingSets;
         SegmentedTensor * const pSmallChangeToModelOverwriteSingleSamplingSet =
            SegmentedTensor::Allocate(k_cDimensionsMax, cVectorLength);
         if(LIKELY(nullptr != pSmallChangeToModelOverwriteSingleSamplingSet)) {
            pNew->m_pSmallChangeToModelOverwriteSingleSamplingSet = pSmallChangeToModelOverwriteSingleSamplingSet;
            HistogramTargetEntryBase * const aSumHistogramTargetEntry =
               EbmMalloc<HistogramTargetEntryBase>(cVectorLength, cBytesPerItem);
            if(LIKELY(nullptr != aSumHistogramTargetEntry)) {
               pNew->m_aSumHistogramTargetEntry = aSumHistogramTargetEntry;
               HistogramTargetEntryBase * const aSumHistogramTargetEntryLeft =
                  EbmMalloc<HistogramTargetEntryBase>(cVectorLength, cBytesPerItem);
               if(LIKELY(nullptr != aSumHistogramTargetEntryLeft)) {
                  pNew->m_aSumHistogramTargetEntryLeft = aSumHistogramTargetEntryLeft;
                  HistogramTargetEntryBase * const aSumHistogramTargetEntryRight =
                     EbmMalloc<HistogramTargetEntryBase>(cVectorLength, cBytesPerItem);
                  if(LIKELY(nullptr != aSumHistogramTargetEntryRight)) {
                     pNew->m_aSumHistogramTargetEntryRight = aSumHistogramTargetEntryRight;
                     FloatEbmType * const aTempFloatVector = EbmMalloc<FloatEbmType>(cVectorLength);
                     if(LIKELY(nullptr != aTempFloatVector)) {
                        pNew->m_aTempFloatVector = aTempFloatVector;
                        const size_t cBytesArrayEquivalentSplitMax = pBooster->GetCountBytesArrayEquivalentSplitMax();
                        if(0 != cBytesArrayEquivalentSplitMax) {
                           void * aEquivalentSplits = EbmMalloc<void>(cBytesArrayEquivalentSplitMax);
                           if(UNLIKELY(nullptr == aEquivalentSplits)) {
                              goto exit_error;
                           }
                           pNew->m_aEquivalentSplits = aEquivalentSplits;
                        }
                        pNew->m_pBooster = pBooster;

                        LOG_0(TraceLevelInfo, "Exited ThreadStateBoosting::Allocate");
                        return pNew;
                     }
                  }
               }
            }
         }
      }
   exit_error:;
      Free(pNew);
   }
   LOG_0(TraceLevelWarning, "WARNING Exited ThreadStateBoosting::Allocate with error");
   return nullptr;
}

HistogramBucketBase * ThreadStateBoosting::GetHistogramBucketBase(const size_t cBytesRequired) {
   HistogramBucketBase * aBuffer = m_aThreadByteBuffer1;
   if(UNLIKELY(m_cThreadByteBufferCapacity1 < cBytesRequired)) {
      m_cThreadByteBufferCapacity1 = cBytesRequired << 1;
      LOG_N(TraceLevelInfo, "Growing ThreadStateBoosting::ThreadByteBuffer1 to %zu", m_cThreadByteBufferCapacity1);

      free(aBuffer);
      aBuffer = static_cast<HistogramBucketBase *>(EbmMalloc<void>(m_cThreadByteBufferCapacity1));
      m_aThreadByteBuffer1 = aBuffer;
   }
   return aBuffer;
}

bool ThreadStateBoosting::GrowThreadByteBuffer2(const size_t cByteBoundaries) {
   // by adding cByteBoundaries and shifting our existing size, we do 2 things:
   //   1) we ensure that if we have zero size, we'll get some size that we'll get a non-zero size after the shift
   //   2) we'll always get back an odd number of items, which is good because we always have an odd number of TreeNodeChilden
   EBM_ASSERT(0 == m_cThreadByteBufferCapacity2 % cByteBoundaries);
   m_cThreadByteBufferCapacity2 = cByteBoundaries + (m_cThreadByteBufferCapacity2 << 1);
   LOG_N(TraceLevelInfo, "Growing ThreadStateBoosting::ThreadByteBuffer2 to %zu", m_cThreadByteBufferCapacity2);

   // our tree objects have internal pointers, so we're going to dispose of our work anyways
   // We can't use realloc since there is no way to check if the array was re-allocated or not without 
   // invoking undefined behavior, so we don't get a benefit if the array can be resized with realloc

   void * aBuffer = m_aThreadByteBuffer2;
   free(aBuffer);
   aBuffer = EbmMalloc<void>(m_cThreadByteBufferCapacity2);
   m_aThreadByteBuffer2 = aBuffer;
   if(UNLIKELY(nullptr == aBuffer)) {
      return true;
   }
   return false;
}

EBM_NATIVE_IMPORT_EXPORT_BODY ThreadStateBoostingHandle EBM_NATIVE_CALLING_CONVENTION CreateThreadStateBoosting(
   BoosterHandle boosterHandle
) {
   LOG_N(TraceLevelInfo, "Entered CreateThreadStateBoosting: boosterHandle=%p", static_cast<void *>(boosterHandle));

   Booster * const pBooster = reinterpret_cast<Booster *>(boosterHandle);
   if(nullptr == pBooster) {
      LOG_0(TraceLevelError, "ERROR CreateThreadStateBoosting boosterHandle cannot be nullptr");
      return nullptr;
   }

   ThreadStateBoosting * const pThreadStateBoosting = ThreadStateBoosting::Allocate(pBooster);
   if(UNLIKELY(nullptr == pThreadStateBoosting)) {
      LOG_0(TraceLevelWarning, "WARNING CreateThreadStateBoosting nullptr == pThreadStateBoosting");
      return nullptr;
   }

   LOG_N(TraceLevelInfo, "Exited CreateThreadStateBoosting: %p", static_cast<void *>(pThreadStateBoosting));
   return reinterpret_cast<ThreadStateBoostingHandle>(pThreadStateBoosting);
}

EBM_NATIVE_IMPORT_EXPORT_BODY void EBM_NATIVE_CALLING_CONVENTION FreeThreadStateBoosting(
   ThreadStateBoostingHandle threadStateBoostingHandle
) {
   LOG_N(TraceLevelInfo, 
      "Entered FreeThreadStateBoosting: threadStateBoostingHandle=%p", 
      static_cast<void *>(threadStateBoostingHandle)
   );

   ThreadStateBoosting * const pThreadStateBoostingHandle = 
      reinterpret_cast<ThreadStateBoosting *>(threadStateBoostingHandle);

   // it's legal to call free on nullptr, just like for free().  This is checked inside ThreadStateBoostingHandle::Free()
   ThreadStateBoosting::Free(pThreadStateBoostingHandle);

   LOG_0(TraceLevelInfo, "Exited FreeThreadStateBoosting");
}

} // DEFINED_ZONE_NAME
