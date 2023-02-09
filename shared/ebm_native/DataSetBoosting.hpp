// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_BOOSTING_HPP
#define DATA_SET_BOOSTING_HPP

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h" // ErrorEbm
#include "logging.h" // EBM_ASSERT
#include "common_c.h" // FloatFast
#include "bridge_c.h" // StorageDataType
#include "zones.h"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class Term;

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

   inline void InitializeUnfailing() {
      m_aGradientsAndHessians = nullptr;
      m_aSampleScores = nullptr;
      m_aTargetData = nullptr;
      m_aaInputData = nullptr;
      m_cSamples = 0;
      m_cTerms = 0;
   }

   void Destruct();

   ErrorEbm Initialize(
      const ptrdiff_t cClasses,
      const bool bAllocateGradients,
      const bool bAllocateHessians,
      const bool bAllocateSampleScores,
      const bool bAllocateTargetData,
      const unsigned char * const pDataSetShared,
      const size_t cSharedSamples,
      const BagEbm direction,
      const BagEbm * const aBag,
      const double * const aInitScores,
      const size_t cSetSamples,
      const IntEbm * const aiTermFeatures,
      const size_t cTerms,
      const Term * const * const apTerms
   );

   inline bool IsGradientsAndHessiansNull() {
      // TODO: remove this and just use GetGradientsAndHessiansPointer
      return nullptr == m_aGradientsAndHessians;
   }

   inline FloatFast * GetGradientsAndHessiansPointer() {
      return m_aGradientsAndHessians;
   }
   inline const FloatFast * GetGradientsAndHessiansPointer() const {
      return m_aGradientsAndHessians;
   }
   inline FloatFast * GetSampleScores() {
      return m_aSampleScores;
   }
   inline const StorageDataType * GetTargetDataPointer() const {
      return m_aTargetData;
   }
   inline const StorageDataType * GetInputDataPointer(const size_t iTerm) const {
      EBM_ASSERT(iTerm < m_cTerms);
      EBM_ASSERT(nullptr != m_aaInputData);
      return m_aaInputData[iTerm];
   }
   inline size_t GetCountSamples() const {
      return m_cSamples;
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
