// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_BOOSTING_HPP
#define DATA_SET_BOOSTING_HPP

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "FeatureGroup.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class DataSetBoosting final {
   FloatFast * m_aGradientsAndHessians;
   FloatFast * m_aSampleScores;
   StorageDataType * m_aTargetData;
   StorageDataType * * m_aaInputData;
   size_t m_cSamples;
   size_t m_cTerms;

public:

   DataSetBoosting() = default; // preserve our POD status
   ~DataSetBoosting() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   INLINE_ALWAYS void InitializeUnfailing() {
      m_aGradientsAndHessians = nullptr;
      m_aSampleScores = nullptr;
      m_aTargetData = nullptr;
      m_aaInputData = nullptr;
      m_cSamples = 0;
      m_cTerms = 0;
   }

   void Destruct();

   ErrorEbmType Initialize(
      const ptrdiff_t runtimeLearningTypeOrCountTargetClasses,
      const bool bAllocateGradients,
      const bool bAllocateHessians,
      const bool bAllocateSampleScores,
      const bool bAllocateTargetData,
      const unsigned char * const pDataSetShared,
      const BagEbmType direction,
      const BagEbmType * const aBag,
      const double * const aInitScores,
      const size_t cSetSamples,
      const size_t cTerms,
      const Term * const * const apTerms
   );

   INLINE_ALWAYS FloatFast * GetGradientsAndHessiansPointer() {
      EBM_ASSERT(nullptr != m_aGradientsAndHessians);
      return m_aGradientsAndHessians;
   }
   INLINE_ALWAYS const FloatFast * GetGradientsAndHessiansPointer() const {
      EBM_ASSERT(nullptr != m_aGradientsAndHessians);
      return m_aGradientsAndHessians;
   }
   INLINE_ALWAYS FloatFast * GetSampleScores() {
      EBM_ASSERT(nullptr != m_aSampleScores);
      return m_aSampleScores;
   }
   INLINE_ALWAYS const StorageDataType * GetTargetDataPointer() const {
      EBM_ASSERT(nullptr != m_aTargetData);
      return m_aTargetData;
   }
   // TODO: we can change this to take the GetIndexTerm() value directly, which we get from a loop index
   INLINE_ALWAYS const StorageDataType * GetInputDataPointer(const Term * const pTerm) const {
      EBM_ASSERT(nullptr != pTerm);
      EBM_ASSERT(pTerm->GetIndexTerm() < m_cTerms);
      EBM_ASSERT(nullptr != m_aaInputData);
      return m_aaInputData[pTerm->GetIndexTerm()];
   }
   INLINE_ALWAYS size_t GetCountSamples() const {
      return m_cSamples;
   }
   INLINE_ALWAYS size_t GetCountTerms() const {
      return m_cTerms;
   }
};
static_assert(std::is_standard_layout<DataSetBoosting>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<DataSetBoosting>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<DataSetBoosting>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // DATA_SET_BOOSTING_HPP
