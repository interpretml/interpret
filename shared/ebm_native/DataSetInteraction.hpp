// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef DATA_SET_INTERACTION_HPP
#define DATA_SET_INTERACTION_HPP

#include <stddef.h> // size_t, ptrdiff_t

#include "ebm_native.h"
#include "logging.h"
#include "zones.h"

#include "ebm_internal.hpp"

#include "Feature.hpp"

namespace DEFINED_ZONE_NAME {
#ifndef DEFINED_ZONE_NAME
#error DEFINED_ZONE_NAME must be defined
#endif // DEFINED_ZONE_NAME

class DataSetInteraction final {
   FloatFast * m_aGradientsAndHessians;
   StorageDataType * * m_aaInputData;
   size_t m_cSamples;
   size_t m_cFeatures;

   FloatFast * m_aWeights;
   FloatBig m_weightTotal;

public:

   DataSetInteraction() = default; // preserve our POD status
   ~DataSetInteraction() = default; // preserve our POD status
   void * operator new(std::size_t) = delete; // we only use malloc/free in this library
   void operator delete (void *) = delete; // we only use malloc/free in this library

   void Destruct();

   INLINE_ALWAYS void InitializeUnfailing() {
      m_aGradientsAndHessians = nullptr;
      m_aaInputData = nullptr;
      m_cSamples = 0;
      m_cFeatures = 0;
      m_aWeights = nullptr;
      m_weightTotal = 0;
   }

   ErrorEbmType Initialize(
      const bool bAllocateHessians,
      const unsigned char * const pDataSetShared,
      const size_t cAllSamples,
      const BagEbmType * const aBag,
      const double * const aInitScores,
      const size_t cSetSamples,
      const size_t cWeights,
      const size_t cFeatures
   );

   INLINE_ALWAYS const FloatFast * GetWeights() const {
      return m_aWeights;
   }
   INLINE_ALWAYS FloatBig GetWeightTotal() const {
      return m_weightTotal;
   }

   INLINE_ALWAYS const FloatFast * GetGradientsAndHessiansPointer() const {
      EBM_ASSERT(nullptr != m_aGradientsAndHessians);
      return m_aGradientsAndHessians;
   }
   // TODO: we can change this to take the m_iFeatureData value directly, which we get from a loop index
   INLINE_ALWAYS const StorageDataType * GetInputDataPointer(const Feature * const pFeature) const {
      EBM_ASSERT(nullptr != pFeature);
      EBM_ASSERT(pFeature->GetIndexFeatureData() < m_cFeatures);
      EBM_ASSERT(nullptr != m_aaInputData);
      return m_aaInputData[pFeature->GetIndexFeatureData()];
   }
   INLINE_ALWAYS size_t GetCountSamples() const {
      return m_cSamples;
   }
   INLINE_ALWAYS size_t GetCountFeatures() const {
      return m_cFeatures;
   }
};
static_assert(std::is_standard_layout<DataSetInteraction>::value,
   "We use the struct hack in several places, so disallow non-standard_layout types in general");
static_assert(std::is_trivial<DataSetInteraction>::value,
   "We use memcpy in several places, so disallow non-trivial types in general");
static_assert(std::is_pod<DataSetInteraction>::value,
   "We use a lot of C constructs, so disallow non-POD types in general");

} // DEFINED_ZONE_NAME

#endif // DATA_SET_INTERACTION_HPP
