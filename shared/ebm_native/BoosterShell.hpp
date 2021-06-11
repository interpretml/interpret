// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef BOOSTER_SHELL_HPP
#define BOOSTER_SHELL_HPP

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "HistogramTargetEntry.hpp"
#include "BoosterCore.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

struct HistogramBucketBase;

class BoosterShell final {
   static constexpr size_t k_handleVerification = 25077; // random 15 bit number
   size_t m_handleVerification; // this needs to be at the top and make it pointer sized to keep best alignment

   BoosterCore * m_pBoosterCore;
   size_t m_iFeatureGroup;

   SegmentedTensor * m_pSmallChangeToModelAccumulatedFromSamplingSets;
   SegmentedTensor * m_pSmallChangeToModelOverwriteSingleSamplingSet;

   // TODO: can I preallocate m_aThreadByteBuffer1 and m_aThreadByteBuffer2 without resorting to grow them if I examine my inputs

   HistogramBucketBase * m_aThreadByteBuffer1;
   size_t m_cThreadByteBufferCapacity1;

   void * m_aThreadByteBuffer2;
   size_t m_cThreadByteBufferCapacity2;

   FloatEbmType * m_aTempFloatVector;
   void * m_aEquivalentSplits; // we use different structures for mains and multidimension and between classification and regression

   HistogramTargetEntryBase * m_aSumHistogramTargetEntry;
   HistogramTargetEntryBase * m_aSumHistogramTargetEntryLeft;
   HistogramTargetEntryBase * m_aSumHistogramTargetEntryRight;

#ifndef NDEBUG
   const unsigned char * m_aHistogramBucketsEndDebug;
#endif // NDEBUG

public:

   BoosterShell() = default; // preserve our POD status
   ~BoosterShell() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   constexpr static size_t k_illegalFeatureGroupIndex = size_t { static_cast<size_t>(ptrdiff_t { -1 }) };

   INLINE_ALWAYS void InitializeZero() {
      m_handleVerification = k_handleVerification;
      m_pBoosterCore = nullptr;
      m_iFeatureGroup = k_illegalFeatureGroupIndex;
      m_pSmallChangeToModelAccumulatedFromSamplingSets = nullptr;
      m_pSmallChangeToModelOverwriteSingleSamplingSet = nullptr;
      m_aThreadByteBuffer1 = nullptr;
      m_cThreadByteBufferCapacity1 = 0;
      m_aThreadByteBuffer2 = nullptr;
      m_cThreadByteBufferCapacity2 = 0;
      m_aTempFloatVector = nullptr;
      m_aEquivalentSplits = nullptr;
      m_aSumHistogramTargetEntry = nullptr;
      m_aSumHistogramTargetEntryLeft = nullptr;
      m_aSumHistogramTargetEntryRight = nullptr;
   }

   static void Free(BoosterShell * const pBoosterShell);
   static BoosterShell * Create();
   ErrorEbmType FillAllocations();

   INLINE_ALWAYS bool IsValid() {
      // TODO : call this function before using this object whenever we get it from outside our control, just to verify
      // TODO : make a similar verification for the InteractionShell
      return k_handleVerification == m_handleVerification;
   }

   INLINE_ALWAYS BoosterCore * GetBoosterCore() {
      EBM_ASSERT(nullptr != m_pBoosterCore);
      return m_pBoosterCore;
   }

   INLINE_ALWAYS void SetBoosterCore(BoosterCore * const pBoosterCore) {
      EBM_ASSERT(nullptr != pBoosterCore);
      EBM_ASSERT(nullptr == m_pBoosterCore); // only set it once
      m_pBoosterCore = pBoosterCore;
   }

   INLINE_ALWAYS size_t GetFeatureGroupIndex() {
      return m_iFeatureGroup;
   }

   INLINE_ALWAYS void SetFeatureGroupIndex(const size_t val) {
      m_iFeatureGroup = val;
   }

   INLINE_ALWAYS SegmentedTensor * GetAccumulatedModelUpdate() {
      return m_pSmallChangeToModelAccumulatedFromSamplingSets;
   }

   INLINE_ALWAYS SegmentedTensor * GetOverwritableModelUpdate() {
      return m_pSmallChangeToModelOverwriteSingleSamplingSet;
   }

   HistogramBucketBase * GetHistogramBucketBase(const size_t cBytesRequired);

   INLINE_ALWAYS HistogramBucketBase * GetHistogramBucketBase() {
      // call this if the histograms were already allocated and we just need the pointer
      return m_aThreadByteBuffer1;
   }

   bool GrowThreadByteBuffer2(const size_t cByteBoundaries);

   INLINE_ALWAYS void * GetThreadByteBuffer2() {
      return m_aThreadByteBuffer2;
   }

   INLINE_ALWAYS size_t GetThreadByteBuffer2Size() const {
      return m_cThreadByteBufferCapacity2;
   }

   INLINE_ALWAYS FloatEbmType * GetTempFloatVector() {
      return m_aTempFloatVector;
   }

   INLINE_ALWAYS void * GetEquivalentSplits() {
      return m_aEquivalentSplits;
   }

   INLINE_ALWAYS HistogramTargetEntryBase * GetSumHistogramTargetEntryArray() {
      return m_aSumHistogramTargetEntry;
   }

   template<bool bClassification>
   INLINE_ALWAYS HistogramTargetEntry<bClassification> * GetSumHistogramTargetEntryLeft() {
      return static_cast<HistogramTargetEntry<bClassification> *>(m_aSumHistogramTargetEntryLeft);
   }

   template<bool bClassification>
   INLINE_ALWAYS HistogramTargetEntry<bClassification> * GetSumHistogramTargetEntryRight() {
      return static_cast<HistogramTargetEntry<bClassification> *>(m_aSumHistogramTargetEntryRight);
   }

#ifndef NDEBUG
   INLINE_ALWAYS const unsigned char * GetHistogramBucketsEndDebug() const {
      return m_aHistogramBucketsEndDebug;
   }

   INLINE_ALWAYS void SetHistogramBucketsEndDebug(const unsigned char * const val) {
      m_aHistogramBucketsEndDebug = val;
   }
#endif // NDEBUG
};
static_assert(std::is_standard_layout<BoosterShell>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<BoosterShell>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<BoosterShell>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // BOOSTER_SHELL_HPP
